"""GOP-boundary metrics and a receiver-side temporal refiner for V8.

V8 decodes each 6-frame GOP on its own, so errors are independent across GOPs
and the picture jumps at every GOP boundary. The metrics here measure that
jump; ``BoundaryRefiner`` removes it at the receiver by filtering the decoded
stream across GOPs. The wire and the codec are unchanged.

Temporal error at transition ``t`` (frame ``t-1`` to ``t``) is the MSE between
the reconstructed and source frame differences, reported as a PSNR:
``10 log10(1 / mean(((r_t - r_{t-1}) - (s_t - s_{t-1}))^2))``. A GOP-boundary
transition is one where frame ``t`` starts a GOP.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _psnr(mse: float) -> float:
    return 100.0 if mse <= 1e-10 else 10.0 * math.log10(1.0 / mse)


def frame_psnr(source: torch.Tensor, recon: torch.Tensor) -> list[float]:
    """Per-frame PSNR of (3, T, H, W) clips."""
    mse = (source.float() - recon.float()).square().mean(dim=(0, 2, 3))
    return [_psnr(float(m)) for m in mse]


def transition_error(source: torch.Tensor, recon: torch.Tensor) -> list[float]:
    """Temporal-difference MSE for transitions 1..T-1 of (3, T, H, W) clips."""
    ds = source[:, 1:].float() - source[:, :-1].float()
    dr = recon[:, 1:].float() - recon[:, :-1].float()
    return [float(m) for m in (dr - ds).square().mean(dim=(0, 2, 3))]


def excess_jump(source: torch.Tensor, recon: torch.Tensor) -> list[float]:
    """Mean |r_t - r_{t-1}| minus mean |s_t - s_{t-1}|, in 8-bit levels, per transition."""
    ds = (source[:, 1:].float() - source[:, :-1].float()).abs().mean(dim=(0, 2, 3))
    dr = (recon[:, 1:].float() - recon[:, :-1].float()).abs().mean(dim=(0, 2, 3))
    return [float(v) * 255.0 for v in (dr - ds)]


def boundary_record(source: torch.Tensor, recon: torch.Tensor, gop: int) -> dict:
    """Per-clip boundary statistics for a (3, T, H, W) clip decoded in ``gop``-frame GOPs."""
    frames = source.shape[1]
    fpsnr = frame_psnr(source, recon)
    terr = transition_error(source, recon)
    jump = excess_jump(source, recon)
    boundary = [t for t in range(1, frames) if t % gop == 0]
    interior = [t for t in range(1, frames) if t % gop != 0]
    by_position = [float(np.mean([fpsnr[t] for t in range(frames) if t % gop == k])) for k in range(gop)]
    b_err = float(np.mean([terr[t - 1] for t in boundary]))
    i_err = float(np.mean([terr[t - 1] for t in interior]))
    return {
        "frame_psnr": fpsnr,
        "position_psnr": by_position,
        "boundary_tpsnr": _psnr(b_err),
        "interior_tpsnr": _psnr(i_err),
        "boundary_gap_db": _psnr(i_err) - _psnr(b_err),
        "boundary_jump": float(np.mean([jump[t - 1] for t in boundary])),
        "interior_jump": float(np.mean([jump[t - 1] for t in interior])),
        "transition_tpsnr": [_psnr(e) for e in terr],
    }


def phase_channels(frames: int, gop: int, offset: int = 0) -> torch.Tensor:
    """(gop, T) one-hot of each frame's position inside its GOP."""
    phase = (torch.arange(frames) + offset) % gop
    return F.one_hot(phase, gop).T.float()


class _Conv(nn.Conv3d):
    """3D conv with temporal kernel 3; ``causal`` pads the past only."""

    def __init__(self, cin: int, cout: int, spatial: int = 3, stride: int = 1, causal: bool = False):
        super().__init__(cin, cout, (3, spatial, spatial), stride=(1, stride, stride), padding=0)
        self.causal = causal
        self.pad_s = (spatial - stride + 1) // 2 if stride > 1 else spatial // 2

    def forward(self, x):
        t_pad = (2, 0) if self.causal else (1, 1)
        x = F.pad(x, (self.pad_s, self.pad_s, self.pad_s, self.pad_s, *t_pad), mode="replicate")
        return super().forward(x)


class _FrameNorm(nn.GroupNorm):
    """GroupNorm with statistics per frame, so no frame sees another's statistics."""

    def forward(self, x):
        b, c, t, h, w = x.shape
        y = super().forward(x.transpose(1, 2).reshape(b * t, c, h, w))
        return y.reshape(b, t, c, h, w).transpose(1, 2)


class _Res3d(nn.Module):
    def __init__(self, ch: int, causal: bool):
        super().__init__()
        self.body = nn.Sequential(
            _Conv(ch, ch, causal=causal), _FrameNorm(8, ch), nn.SiLU(),
            _Conv(ch, ch, causal=causal), _FrameNorm(8, ch),
        )

    def forward(self, x):
        return F.silu(x + self.body(x))


class BoundaryRefiner(nn.Module):
    """Residual 3D U-Net over a decoded stream window (B, 3, T, H, W).

    Inputs per frame are the decoded RGB, the frame's GOP-position one-hot and
    its GOP's mean modem confidence. With ``causal=False`` a frame's output
    depends on decoded frames on both sides, across GOP boundaries (run on a
    window of the previous, current and next GOP: one GOP of latency). With
    ``causal=True`` it depends only on the current and earlier frames, so the
    first frames of a GOP are conditioned on the previous GOP's decoded frames
    at no added latency. The last layer starts at zero: an untrained refiner
    is the identity. Temporal resolution is never reduced.
    """

    def __init__(self, gop: int = 6, width: int = 64, blocks: int = 3, causal: bool = False, per_gop: bool = False):
        super().__init__()
        self.gop, self.per_gop = gop, per_gop
        cin = 3 + gop + 1
        w1, w2, w3 = width, width * 3 // 2, width * 2
        self.head = nn.Sequential(_Conv(cin, w1, causal=causal), nn.SiLU(), _Res3d(w1, causal))
        self.down1 = _Conv(w1, w2, 4, 2, causal)
        self.mid1 = nn.Sequential(*[_Res3d(w2, causal) for _ in range(blocks)])
        self.down2 = _Conv(w2, w3, 4, 2, causal)
        self.mid2 = nn.Sequential(*[_Res3d(w3, causal) for _ in range(blocks)])
        self.up2 = _Conv(w3 + w2, w2, causal=causal)
        self.res2 = _Res3d(w2, causal)
        self.up1 = _Conv(w2 + w1, w1, causal=causal)
        self.res1 = _Res3d(w1, causal)
        self.tail = _Conv(w1, 3, causal=causal)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, video: torch.Tensor, gop_confidence: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """``gop_confidence``: (B, T) mean confidence of each frame's GOP.

        ``per_gop`` refines each GOP on its own (an ablation with no cross-GOP view).
        """
        b, _, t, h, w = video.shape
        if self.per_gop and t > self.gop:
            g = t // self.gop
            split = video.reshape(b, 3, g, self.gop, h, w).transpose(1, 2).reshape(b * g, 3, self.gop, h, w)
            out = self._refine(split, gop_confidence.reshape(b * g, self.gop), 0)
            return out.reshape(b, g, 3, self.gop, h, w).transpose(1, 2).reshape(b, 3, t, h, w)
        return self._refine(video, gop_confidence, offset)

    def _refine(self, video: torch.Tensor, gop_confidence: torch.Tensor, offset: int) -> torch.Tensor:
        b, _, t, h, w = video.shape
        phase = phase_channels(t, self.gop, offset).to(video)[None, :, :, None, None].expand(b, -1, -1, h, w)
        conf = gop_confidence.to(video)[:, None, :, None, None].expand(-1, 1, -1, h, w)
        x = torch.cat([video * 2 - 1, phase, conf], 1)
        pad_h, pad_w = (-h) % 4, (-w) % 4
        x = F.pad(x, (0, pad_w, 0, pad_h, 0, 0), mode="replicate")
        s1 = self.head(x)
        s2 = self.mid1(F.silu(self.down1(s1)))
        s3 = self.mid2(F.silu(self.down2(s2)))
        u2 = F.interpolate(s3, size=s2.shape[2:], mode="trilinear", align_corners=False)
        u2 = self.res2(F.silu(self.up2(torch.cat([u2, s2], 1))))
        u1 = F.interpolate(u2, size=s1.shape[2:], mode="trilinear", align_corners=False)
        u1 = self.res1(F.silu(self.up1(torch.cat([u1, s1], 1))))
        residual = self.tail(u1)[..., :h, :w]
        return (video + residual).clamp(0, 1)


def load_refiner(path, device) -> BoundaryRefiner:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = BoundaryRefiner(**payload["config"])
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval()
