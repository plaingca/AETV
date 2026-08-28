#!/usr/bin/env python3
"""Train a causal pixel-domain state over a complete contiguous GOP stream."""
from pathlib import Path
import argparse, json, random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from aetv.config import AETV_MODES
from aetv.stateful_gop_corrector import StatefulGOPCorrector
from scripts.experiment_gop_boundaries import DEFAULT_CELLS, SequenceCache, load_model

class Indexed(Dataset):
    def __init__(self, d): self.d=d
    def __len__(self): return len(self.d)
    def __getitem__(self,i): return i,self.d[i]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--steps',type=int,default=1500); ap.add_argument('--batch',type=int,default=2); ap.add_argument('--lr',type=float,default=1e-4); ap.add_argument('--out',type=Path,default=Path('runs/v8-recurrent-pixel-state')); ap.add_argument('--device',default='cuda'); ap.add_argument('--seed',type=int,default=20260827); a=ap.parse_args(); torch.manual_seed(a.seed); random.seed(a.seed)
    mode=AETV_MODES['V8']; d=torch.device(a.device); base=load_model(Path('models/v8-hf3k-face-gan.pt'),mode,d).eval();
    for p in base.parameters(): p.requires_grad_(False)
    ds=SequenceCache(Path('runs/gop-boundary-data/v8_192x108_5gop_real_train'),max_frames=30); rx=torch.load('runs/v8-stateful-5gop-train-rx.pt',map_location='cpu',weights_only=False); cells=list(DEFAULT_CELLS)
    loader=DataLoader(Indexed(ds),batch_size=a.batch,shuffle=True,drop_last=True,num_workers=0,generator=torch.Generator().manual_seed(a.seed)); it=iter(loader)
    corrector=StatefulGOPCorrector(width=64,blocks=6,spatial_scale=2,max_residual=.5,context_mode='full',taper_floor=.0).to(d); opt=torch.optim.AdamW(corrector.parameters(),lr=a.lr,weight_decay=1e-4)
    for step in range(1,a.steps+1):
        try: indices,target=next(it)
        except StopIteration: it=iter(loader); indices,target=next(it)
        target=target.to(d).float(); ids=torch.randint(0,len(cells),(target.shape[0],)); lat=[]; wt=[]
        for cid,i in zip(ids.tolist(),indices.tolist()): lat.append(rx['received'][cells[cid].label][int(i)]); wt.append(rx['weights'][cells[cid].label][int(i)])
        lat=torch.stack(lat).to(d); wt=torch.stack(wt).to(d); B,G=lat.shape[:2]
        with torch.inference_mode():
            decoded=base.decoder(lat.flatten(0,1),wt.flatten(0,1)).reshape(B,G,3,6,108,192)
        out=[]; prev=None
        for j in range(G):
            cur=decoded[:,j]
            cur=corrector(cur,prev,wt[:,j].mean(-1)) if prev is not None else cur
            out.append(cur); prev=cur
        recon=torch.stack(out,1).permute(0,2,1,3,4,5).reshape(B,3,30,108,192)
        delta=recon[:,:,1:]-recon[:,:,:-1]; td=target[:,:,1:]-target[:,:,:-1]; seams=torch.arange(5,24,6,device=d); mask=torch.ones(29,dtype=torch.bool,device=d); mask[seams]=False
        boundary=(delta[:,:,seams]-td[:,:,seams]).abs().mean(); within=(delta[:,:,mask]-td[:,:,mask]).abs().mean(); loss=F.l1_loss(recon,target)+24*boundary+4*within+F.l1_loss(recon,target)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(corrector.parameters(),1.0); opt.step()
        if step==1 or step%100==0 or step==a.steps: print(json.dumps({'step':step,'loss':float(loss.detach()),'boundary':float(boundary.detach()),'within':float(within.detach())}),flush=True)
    a.out.mkdir(parents=True,exist_ok=True); torch.save({'kind':StatefulGOPCorrector.checkpoint_kind,'adapter_config':{'width':64,'blocks':6,'spatial_scale':2,'max_residual':.5,'context_mode':'full','taper_floor':0.0},'adapter_state_dict':corrector.state_dict()},a.out/'corrector.pt')
if __name__=='__main__': main()
