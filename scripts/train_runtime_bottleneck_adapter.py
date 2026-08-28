#!/usr/bin/env python3
"""Train the carried decoder-bottleneck adapter on fixed runtime RX latents."""

from pathlib import Path
import argparse, json, random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from aetv.config import AETV_MODES
from aetv.decoder_context_adapter import V8DecoderContextAdapter
from scripts.experiment_gop_boundaries import DEFAULT_CELLS, SequenceCache, boundary_losses


class Indexed(torch.utils.data.Dataset):
    def __init__(self, dataset): self.dataset = dataset
    def __len__(self): return len(self.dataset)
    def __getitem__(self, index): return index, self.dataset[index]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--boundary-weight", type=float, default=24)
    ap.add_argument("--anchor-weight", type=float, default=1)
    ap.add_argument("--out", type=Path, default=Path("runs/v8-runtime-bottleneck-adapter"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=20260827)
    a = ap.parse_args(); torch.manual_seed(a.seed); random.seed(a.seed)
    device = torch.device(a.device); mode = AETV_MODES["V8"]
    model = V8DecoderContextAdapter.from_v8_checkpoint("models/v8-hf3k-face-gan.pt", adapter_width=192, attention_dim=96, adapter_blocks=5, freeze_base=True).to(device)
    model.train(); model.encoder.eval(); model.decoder.eval()
    train = SequenceCache(Path("runs/gop-boundary-data/v8_192x108_3gop_train"))
    rx = torch.load("runs/v8-two-gop-boundary-sweep-lr1e5/train-runtime-rx.pt", map_location="cpu", weights_only=False)
    cells = list(DEFAULT_CELLS); loader = DataLoader(Indexed(train), batch_size=a.batch, shuffle=True, drop_last=True, num_workers=0, generator=torch.Generator().manual_seed(a.seed)); iterator=iter(loader)
    opt=torch.optim.AdamW(model.context_adapter.parameters(),lr=a.lr,weight_decay=1e-4)
    for step in range(1,a.steps+1):
        try: indices, source=next(iterator)
        except StopIteration: iterator=iter(loader); indices,source=next(iterator)
        source=source.to(device).float(); cell_ids=torch.randint(0,len(cells),(source.shape[0],)); received=[];weights=[]
        for cell_id,source_id in zip(cell_ids.tolist(),indices.tolist()):
            label=cells[cell_id].label; received.append(rx['received'][label][int(source_id)]);weights.append(rx['weights'][label][int(source_id)])
        latents=torch.stack(received).to(device); conf=torch.stack(weights).to(device)
        recon, teacher = model.decode_sequence(latents,conf,return_base=True)
        loss_terms=boundary_losses(recon,source,6)
        loss=F.l1_loss(recon,source)+a.anchor_weight*F.l1_loss(recon,teacher)+a.boundary_weight*loss_terms['boundary_rgb_delta']+2*loss_terms['boundary_lowpass_step']+loss_terms['boundary_acceleration']
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.context_adapter.parameters(),1.0);opt.step()
        if step==1 or step%100==0 or step==a.steps: print(json.dumps({'step':step,'loss':float(loss.detach()),'boundary':float(loss_terms['boundary_rgb_delta'].detach())}),flush=True)
    a.out.mkdir(parents=True,exist_ok=True);torch.save({'kind':model.checkpoint_kind,'base_checkpoint':str(Path('models/v8-hf3k-face-gan.pt').resolve()),'model_config':model.config(),'adapter_state_dict':model.context_adapter.state_dict()},a.out/'adapter.pt')


if __name__=='__main__': main()
