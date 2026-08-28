#!/usr/bin/env python3
"""Train a carried-scene corrector that can adjust all six current-GOP frames."""

from pathlib import Path
import argparse, json, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from aetv.config import AETV_MODES
from scripts.experiment_gop_boundaries import DEFAULT_CELLS, SequenceCache, boundary_losses, decode_independent_gops, join_gops
from scripts.experiment_gop_boundaries import load_model


class Residual(nn.Module):
    def __init__(self, width):
        super().__init__(); self.n1=nn.GroupNorm(min(8,width),width); self.c1=nn.Conv2d(width,width,3,padding=1); self.n2=nn.GroupNorm(min(8,width),width); self.c2=nn.Conv2d(width,width,3,padding=1)
    def forward(self,x): return x+self.c2(F.silu(self.n2(self.c1(F.silu(self.n1(x))))))


class WholeGOPSceneCorrector(nn.Module):
    def __init__(self,width=128,blocks=8,scale=1,max_residual=.35):
        super().__init__(); self.max_residual=max_residual; self.scale=scale; self.input=nn.Conv2d(36,width,3,padding=1); self.body=nn.Sequential(*(Residual(width) for _ in range(blocks))); self.output=nn.Conv2d(width,18,3,padding=1); nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)
    def forward(self,previous,current):
        if previous.shape != current.shape or previous.ndim != 5: raise ValueError('expected equal BCTHW GOP tensors')
        b,c,t,h,w=current.shape; x=torch.cat((previous,current),dim=1).flatten(1,2); lh=max(1,h//self.scale); lw=max(1,w//self.scale); x=F.interpolate(x,(lh,lw),mode='bilinear',align_corners=False); x=self.body(F.silu(self.input(x))); r=F.interpolate(self.output(x),(h,w),mode='bilinear',align_corners=False); return self.max_residual*torch.tanh(r).reshape(b,6,3,h,w).permute(0,2,1,3,4)


class Indexed(torch.utils.data.Dataset):
    def __init__(self,dataset): self.dataset=dataset
    def __len__(self): return len(self.dataset)
    def __getitem__(self,index): return index,self.dataset[index]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--steps',type=int,default=3000); ap.add_argument('--batch',type=int,default=2); ap.add_argument('--lr',type=float,default=5e-5); ap.add_argument('--out',type=Path,default=Path('runs/v8-whole-gop-scene-corrector')); ap.add_argument('--device',default='cuda'); ap.add_argument('--seed',type=int,default=20260827); a=ap.parse_args(); torch.manual_seed(a.seed); random.seed(a.seed)
    mode=AETV_MODES['V8']; d=torch.device(a.device); base=load_model(Path('models/v8-hf3k-face-gan.pt'),mode,d).eval();
    for p in base.parameters(): p.requires_grad_(False)
    ds=SequenceCache(Path('runs/gop-boundary-data/v8_192x108_3gop_train')); rx=torch.load('runs/v8-two-gop-boundary-sweep-lr1e5/train-runtime-rx.pt',map_location='cpu',weights_only=False); cells=list(DEFAULT_CELLS); loader=DataLoader(Indexed(ds),batch_size=a.batch,shuffle=True,drop_last=True,num_workers=0,generator=torch.Generator().manual_seed(a.seed)); it=iter(loader); model=WholeGOPSceneCorrector().to(d); opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    for step in range(1,a.steps+1):
        try: indices,source=next(it)
        except StopIteration: it=iter(loader); indices,source=next(it)
        source=source.to(d).float(); cell_ids=torch.randint(0,len(cells),(source.shape[0],)); rows=[]
        for cell_id,source_id in zip(cell_ids.tolist(),indices.tolist()):
            label=cells[cell_id].label; z=rx['received'][label][int(source_id)].unsqueeze(0).to(d); w=rx['weights'][label][int(source_id)].unsqueeze(0).to(d); rows.append(decode_independent_gops(base,z,w,mode).reshape(1,2,3,6,108,192))
        g=torch.cat(rows); corrected=g.clone(); corrected[:,1]= (corrected[:,1]+model(g[:,0],g[:,1])).clamp(0,1); recon=join_gops(corrected.flatten(0,1),source.shape[0],2); anchor=join_gops(g.flatten(0,1),source.shape[0],2); cross=boundary_losses(recon,source,6); temporal=((recon[:,:,1:]-recon[:,:,:-1])-(source[:,:,1:]-source[:,:,:-1])).abs().mean(); loss=F.l1_loss(recon,source)+F.l1_loss(recon,anchor)+24*cross['boundary_rgb_delta']+4*cross['boundary_lowpass_step']+2*cross['boundary_acceleration']+2*temporal
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if step==1 or step%100==0 or step==a.steps: print(json.dumps({'step':step,'loss':float(loss.detach()),'boundary':float(cross['boundary_rgb_delta'].detach())}),flush=True)
    a.out.mkdir(parents=True,exist_ok=True); torch.save({'kind':'aetv-whole-gop-scene-corrector','config':{'width':128,'blocks':8,'scale':1,'max_residual':.35},'state_dict':model.state_dict()},a.out/'corrector.pt')


if __name__=='__main__': main()
