#!/usr/bin/env python3
"""Train a boundary-only predictor using two decoded GOPs as receiver context."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from aetv.config import AETV_MODES
from experiment_gop_boundaries import DEFAULT_CELLS, ChannelCell, SequenceCache, boundary_losses, cache_name, join_gops, load_model
from experiment_gop_context import decode_base_gops

class Residual(nn.Module):
    def __init__(self, width):
        super().__init__(); self.n1=nn.GroupNorm(min(8,width),width); self.c1=nn.Conv2d(width,width,3,padding=1); self.n2=nn.GroupNorm(min(8,width),width); self.c2=nn.Conv2d(width,width,3,padding=1)
    def forward(self,x): return x+self.c2(F.silu(self.n2(self.c1(F.silu(self.n1(x))))))

class BoundaryFramePredictor(nn.Module):
    def __init__(self,width=64,blocks=6,scale=2,max_residual=.5):
        super().__init__(); self.width=width; self.blocks=blocks; self.scale=scale; self.max_residual=max_residual
        self.input=nn.Conv2d(36,width,3,padding=1); self.body=nn.Sequential(*(Residual(width) for _ in range(blocks))); self.output=nn.Conv2d(width,3,3,padding=1)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)
    def forward(self,current,previous):
        if current.shape != previous.shape or current.ndim != 5: raise ValueError('expected equal BGCTHW GOP tensors')
        b, c, t, h, w=current.shape
        x=torch.cat((current,previous),dim=1).flatten(1,2)
        lh=max(1,h//self.scale); lw=max(1,w//self.scale)
        x=F.interpolate(x,(lh,lw),mode='bilinear',align_corners=False)
        r=self.output(self.body(F.silu(self.input(x))))
        r=F.interpolate(r,(h,w),mode='bilinear',align_corners=False)
        return (current[:,:,0]+self.max_residual*torch.tanh(r)).clamp(0,1)

def apply(predictor,gops):
    out=gops.clone()
    for i in range(1,gops.shape[1]): out[:,i,:,0]=predictor(gops[:,i],out[:,i-1])
    return out

def train(a):
    mode=AETV_MODES[a.mode]; dev=torch.device(a.device); base=load_model(a.checkpoint,mode,dev).eval()
    for p in base.parameters(): p.requires_grad_(False)
    predictor=BoundaryFramePredictor(a.width,a.blocks,a.scale,a.max_residual).to(dev); opt=torch.optim.AdamW(predictor.parameters(),lr=a.lr,weight_decay=1e-4)
    ds=SequenceCache(a.data_dir/cache_name(mode,3,'train')); loader=DataLoader(ds,batch_size=a.batch,shuffle=True,drop_last=True,num_workers=0); it=iter(loader)
    for step in range(1,a.steps+1):
        try: source=next(it)
        except StopIteration: it=iter(loader); source=next(it)
        source=source.to(dev).float()[:, :, :12]
        with torch.no_grad(): base_gops,_=decode_base_gops(base,source,mode,ChannelCell('clean','clean',None,None)); base_gops=base_gops[:, :2]
        opt.zero_grad(set_to_none=True); corrected=apply(predictor,base_gops); recon=join_gops(corrected.flatten(0,1),source.shape[0],2)
        cross=boundary_losses(recon,source,6)
        loss=a.source_weight*F.l1_loss(recon,source)+a.anchor_weight*F.l1_loss(recon,join_gops(base_gops.flatten(0,1),source.shape[0],2))+a.boundary_weight*cross['boundary_rgb_delta']+a.lowpass_weight*cross['boundary_lowpass_step']+a.within_weight*cross['within_gop_temporal_error']
        loss.backward(); torch.nn.utils.clip_grad_norm_(predictor.parameters(),1.0); opt.step()
        if step==1 or step%a.log_interval==0 or step==a.steps: print(json.dumps({'step':step,'loss':float(loss.detach()),'boundary':float(cross['boundary_rgb_delta'].detach())}),flush=True)
    a.out.mkdir(parents=True,exist_ok=True); torch.save({'kind':'aetv-boundary-frame-predictor','config':{'width':a.width,'blocks':a.blocks,'scale':a.scale,'max_residual':a.max_residual},'state_dict':predictor.state_dict(),'base_checkpoint':str(a.checkpoint.resolve())},a.out/'predictor.pt')

def evaluate(a):
    mode=AETV_MODES[a.mode]; dev=torch.device(a.device); base=load_model(a.checkpoint,mode,dev).eval(); pld=torch.load(a.predictor,map_location='cpu',weights_only=False); predictor=BoundaryFramePredictor(**pld['config']).to(dev); predictor.load_state_dict(pld['state_dict']); predictor.eval(); ds=SequenceCache(a.data_dir/cache_name(mode,3,'eval')); rows={c.label:{'base':[],'candidate':[]} for c in DEFAULT_CELLS}
    with torch.inference_mode():
        for i in range(min(a.eval_sequences,len(ds))):
            source=ds[i].unsqueeze(0).to(dev).float()
            for cell in DEFAULT_CELLS:
                g,_=decode_base_gops(base,source,mode,cell); g=g[:, :2]; cand=apply(predictor,g)
                for n,v in [('base',g),('candidate',cand)]: rows[cell.label][n].append(sequence_metrics(join_gops(v.flatten(0,1),1,2),source,6,dev,include_lpips=True))
    result={'cells':{}}
    for cell,vals in rows.items():
        result['cells'][cell]={}
        for metric in vals['base'][0]:
            b=sum(x[metric] for x in vals['base'])/len(vals['base']); c=sum(x[metric] for x in vals['candidate'])/len(vals['candidate']); result['cells'][cell][metric]={'baseline_mean':b,'candidate_mean':c,'reduction_percent':100*(b-c)/max(abs(b),1e-12)}
    a.out.mkdir(parents=True,exist_ok=True); (a.out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))

from experiment_gop_boundaries import sequence_metrics
def main():
    p=argparse.ArgumentParser(); p.add_argument('command',choices=('train','eval')); p.add_argument('--checkpoint',type=Path,default=Path('models/v8-hf3k-face-gan.pt')); p.add_argument('--predictor',type=Path,default=Path('runs/v8-boundary-frame/predictor.pt')); p.add_argument('--out',type=Path,default=Path('runs/v8-boundary-frame')); p.add_argument('--data-dir',type=Path,default=Path('runs/gop-boundary-data')); p.add_argument('--mode',default='V8'); p.add_argument('--steps',type=int,default=1000); p.add_argument('--batch',type=int,default=2); p.add_argument('--eval-sequences',type=int,default=32); p.add_argument('--width',type=int,default=64); p.add_argument('--blocks',type=int,default=6); p.add_argument('--scale',type=int,default=2); p.add_argument('--max-residual',type=float,default=.5); p.add_argument('--lr',type=float,default=1e-4); p.add_argument('--source-weight',type=float,default=.5); p.add_argument('--anchor-weight',type=float,default=1.); p.add_argument('--boundary-weight',type=float,default=16.); p.add_argument('--lowpass-weight',type=float,default=8.); p.add_argument('--within-weight',type=float,default=1.); p.add_argument('--log-interval',type=int,default=50); p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); a=p.parse_args(); train(a) if a.command=='train' else evaluate(a)
if __name__=='__main__': main()
