#!/usr/bin/env python3
"""Train the decoder-context adapter as a causal recurrent GOP state."""

from pathlib import Path
import argparse, json, random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from aetv.config import AETV_MODES
from aetv.decoder_context_adapter import V8DecoderContextAdapter
from scripts.experiment_gop_boundaries import DEFAULT_CELLS, SequenceCache


class Indexed(Dataset):
    def __init__(self, dataset): self.dataset = dataset
    def __len__(self): return len(self.dataset)
    def __getitem__(self, index): return index, self.dataset[index]


def loss_terms(recon, target, gop_frames):
    delta = recon[:, :, 1:] - recon[:, :, :-1]
    target_delta = target[:, :, 1:] - target[:, :, :-1]
    seams = torch.arange(gop_frames - 1, recon.shape[2] - 1, gop_frames, device=recon.device)
    mask = torch.ones(delta.shape[2], dtype=torch.bool, device=recon.device)
    mask[seams] = False
    boundary = (delta[:, :, seams] - target_delta[:, :, seams]).abs().mean()
    within = (delta[:, :, mask] - target_delta[:, :, mask]).abs().mean()
    acceleration = (delta[:, :, seams] - delta[:, :, seams - 1]).abs().mean()
    return boundary, within, acceleration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out", type=Path, default=Path("runs/v8-recurrent-decoder-state"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=20260827)
    args = ap.parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed)
    device = torch.device(args.device)
    model = V8DecoderContextAdapter.from_v8_checkpoint(
        "models/v8-hf3k-face-gan.pt", adapter_width=192, attention_dim=96,
        adapter_blocks=5, freeze_base=True,
    ).to(device)
    model.train(); model.encoder.eval(); model.decoder.eval()
    data = SequenceCache(Path("runs/gop-boundary-data/v8_192x108_5gop_real_train"), max_frames=30)
    rx = torch.load("runs/v8-stateful-5gop-train-rx.pt", map_location="cpu", weights_only=False)
    loader = DataLoader(Indexed(data), batch_size=args.batch, shuffle=True, drop_last=True,
                        num_workers=0, generator=torch.Generator().manual_seed(args.seed))
    iterator = iter(loader); opt = torch.optim.AdamW(model.context_adapter.parameters(), lr=args.lr, weight_decay=1e-4)
    cells = list(DEFAULT_CELLS)
    for step in range(1, args.steps + 1):
        try: indices, target = next(iterator)
        except StopIteration: iterator = iter(loader); indices, target = next(iterator)
        target = target.to(device).float()
        cell_ids = torch.randint(0, len(cells), (target.shape[0],))
        latents = torch.stack([rx["received"][cells[c].label][int(i)] for c, i in zip(cell_ids.tolist(), indices.tolist())]).to(device)
        weights = torch.stack([rx["weights"][cells[c].label][int(i)] for c, i in zip(cell_ids.tolist(), indices.tolist())]).to(device)
        recon, base = model.decode_sequence(latents, weights, recurrent_state=True, return_base=True)
        boundary, within, acceleration = loss_terms(recon, target, 6)
        loss = F.l1_loss(recon, target) + 24 * boundary + 2 * within + acceleration + F.l1_loss(recon, base)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.context_adapter.parameters(), 1.0); opt.step()
        if step % 100 == 0 or step == args.steps:
            print(json.dumps({"step": step, "loss": float(loss.detach()), "boundary": float(boundary.detach()), "within": float(within.detach())}), flush=True)
            args.out.mkdir(parents=True, exist_ok=True)
            torch.save({"kind": model.checkpoint_kind, "base_checkpoint": str(Path("models/v8-hf3k-face-gan.pt").resolve()), "model_config": model.config(), "adapter_state_dict": model.context_adapter.state_dict(), "recurrent_state": True, "step": step}, args.out / f"adapter_step_{step:04d}.pt")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": model.checkpoint_kind, "base_checkpoint": str(Path("models/v8-hf3k-face-gan.pt").resolve()), "model_config": model.config(), "adapter_state_dict": model.context_adapter.state_dict(), "recurrent_state": True, "step": args.steps}, args.out / "adapter.pt")


if __name__ == "__main__": main()
