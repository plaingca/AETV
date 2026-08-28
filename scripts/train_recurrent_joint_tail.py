#!/usr/bin/env python3
"""Jointly tune the causal bottleneck state and the released decoder tail."""
from pathlib import Path
import argparse, json, random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from aetv.decoder_context_adapter import V8DecoderContextAdapter
from scripts.experiment_gop_boundaries import DEFAULT_CELLS, SequenceCache

class Indexed(Dataset):
    def __init__(self,d): self.d=d
    def __len__(self): return len(self.d)
    def __getitem__(self,i): return i,self.d[i]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--steps',type=int,default=800); ap.add_argument('--batch',type=int,default=2); ap.add_argument('--lr',type=float,default=2e-5); ap.add_argument('--boundary-weight',type=float,default=40.0); ap.add_argument('--init',type=Path); ap.add_argument('--out',type=Path,default=Path('runs/v8-recurrent-joint-tail')); ap.add_argument('--device',default='cuda'); ap.add_argument('--seed',type=int,default=20260827); a=ap.parse_args(); torch.manual_seed(a.seed); random.seed(a.seed)
    d=torch.device(a.device); model=V8DecoderContextAdapter.from_v8_checkpoint('models/v8-hf3k-face-gan.pt',adapter_width=192,attention_dim=96,adapter_blocks=5,freeze_base=True).to(d)
    model.context_adapter.load_state_dict(torch.load('runs/v8-runtime-bottleneck-adapter/adapter.pt',map_location='cpu',weights_only=False)['adapter_state_dict'])
    inner=model.decoder.decoder
    if a.init:
        initial=torch.load(a.init,map_location='cpu',weights_only=False); model.context_adapter.load_state_dict(initial['adapter_state_dict']); inner.output.load_state_dict(initial['tail_state_dict']['output']); inner.temporal_skip.load_state_dict(initial['tail_state_dict']['temporal_skip'])
    for p in model.parameters(): p.requires_grad_(False)
    trainable=list(model.context_adapter.parameters())+list(inner.output.parameters())+list(inner.temporal_skip.parameters())
    for p in trainable: p.requires_grad_(True)
    model.train(); model.encoder.eval()
    ds=SequenceCache(Path('runs/gop-boundary-data/v8_192x108_5gop_real_train'),max_frames=30); rx=torch.load('runs/v8-stateful-5gop-train-rx.pt',map_location='cpu',weights_only=False); cells=list(DEFAULT_CELLS)
    loader=DataLoader(Indexed(ds),batch_size=a.batch,shuffle=True,drop_last=True,num_workers=0,generator=torch.Generator().manual_seed(a.seed)); it=iter(loader); opt=torch.optim.AdamW(trainable,lr=a.lr,weight_decay=1e-5)
    for step in range(1,a.steps+1):
        try: indices,target=next(it)
        except StopIteration: it=iter(loader); indices,target=next(it)
        target=target.to(d).float(); ids=torch.randint(0,len(cells),(target.shape[0],)); lat=[]; wt=[]
        for cid,i in zip(ids.tolist(),indices.tolist()): lat.append(rx['received'][cells[cid].label][int(i)]); wt.append(rx['weights'][cells[cid].label][int(i)])
        lat=torch.stack(lat).to(d); wt=torch.stack(wt).to(d); recon,base=model.decode_sequence(lat,wt,recurrent_state=True,return_base=True)
        delta=recon[:,:,1:]-recon[:,:,:-1]; td=target[:,:,1:]-target[:,:,:-1]; seams=torch.arange(5,24,6,device=d); mask=torch.ones(29,dtype=torch.bool,device=d); mask[seams]=False
        boundary=(delta[:,:,seams]-td[:,:,seams]).abs().mean(); within=(delta[:,:,mask]-td[:,:,mask]).abs().mean(); loss=F.l1_loss(recon,target)+F.l1_loss(recon,base)+a.boundary_weight*boundary+4*within
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(trainable,1.0); opt.step()
        if step==1 or step%100==0 or step==a.steps:
            print(json.dumps({'step':step,'loss':float(loss.detach()),'boundary':float(boundary.detach()),'within':float(within.detach())}),flush=True); a.out.mkdir(parents=True,exist_ok=True); torch.save({'kind':model.checkpoint_kind,'base_checkpoint':str(Path('models/v8-hf3k-face-gan.pt').resolve()),'model_config':model.config(),'adapter_state_dict':model.context_adapter.state_dict(),'tail_state_dict':{'output':inner.output.state_dict(),'temporal_skip':inner.temporal_skip.state_dict()},'recurrent_state':True,'step':step},a.out/f'adapter_step_{step:04d}.pt')
    torch.save({'kind':model.checkpoint_kind,'base_checkpoint':str(Path('models/v8-hf3k-face-gan.pt').resolve()),'model_config':model.config(),'adapter_state_dict':model.context_adapter.state_dict(),'tail_state_dict':{'output':inner.output.state_dict(),'temporal_skip':inner.temporal_skip.state_dict()},'recurrent_state':True,'step':a.steps},a.out/'adapter.pt')
if __name__=='__main__': main()
