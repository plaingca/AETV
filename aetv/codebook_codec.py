"""From-scratch codebook JSCC codec for the 2,816-real, 2.2 kHz budget.

The wire is not a frozen AC2K latent and it is not a pixel residual. An encoder
with spatiotemporal attention picks 84 spatial tokens and 4 semantic queries.
Each token is a residual against a learned codebook of feature types. The
decoder re-associates a faded token with that codebook before synthesis, so a
carrier null falls back to a known feature instead of an erased pixel.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .framing import GOP_DEINTERLEAVER_W
from .narrowband_jscc import pad_gop, unit_rms, unpad_gop

GOP_VALUES = 2816
LATENT_CARRIERS = 44
CODE_DIM = 32
SPATIAL_TOKENS = 7 * 12
SEMANTIC_TOKENS = 4
TOKENS = SPATIAL_TOKENS + SEMANTIC_TOKENS  # 88 * 32 = 2816
CODES = 4096


def carrier_index() -> torch.Tensor:
    raw = torch.as_tensor(GOP_DEINTERLEAVER_W.copy(), dtype=torch.long)
    if raw.numel() != GOP_VALUES:
        raise RuntimeError(f"interleaver length {raw.numel()} != {GOP_VALUES}")
    return (raw // 2) % LATENT_CARRIERS


def apply_carrier_fade(
    wire: torch.Tensor, sigma: float, generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Equalized two-path fade used while training. Scoring uses the real modem."""
    if wire.shape[-1] != GOP_VALUES:
        raise ValueError(f"wire length {wire.shape[-1]} != {GOP_VALUES}")
    batch = wire.shape[0]
    device = wire.device
    g1 = torch.randn(batch, 1, device=device, generator=generator) + 1j * torch.randn(
        batch, 1, device=device, generator=generator
    )
    g2 = torch.randn(batch, 1, device=device, generator=generator) + 1j * torch.randn(
        batch, 1, device=device, generator=generator
    )
    tones = torch.arange(LATENT_CARRIERS, device=device, dtype=torch.float32)
    phase = torch.exp(-2j * math.pi * tones * 50.0 * 0.002)
    response = (g1 + g2 * phase) / math.sqrt(2)
    power = response.abs().square().mean(dim=1, keepdim=True).clamp_min(1e-8)
    magnitude = (response / power.sqrt()).abs()
    tone = magnitude[:, carrier_index().to(device)].clamp_min(0.08)
    noise = torch.randn(wire.shape, device=device, dtype=wire.dtype, generator=generator)
    received = wire + noise * (sigma / tone)
    strength = tone.square()
    confidence = (strength / (strength + sigma * sigma)).to(wire.dtype)
    return received, confidence


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = 8 if channels % 8 == 0 else 1
        self.net = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class Mixer(nn.Module):
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, queries: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(queries)
        memory = h if context is None else self.norm1(context)
        mixed, _ = self.attn(h, memory, memory, need_weights=False)
        x = queries + mixed
        return x + self.mlp(self.norm2(x))


class Codebook(nn.Module):
    """Database of feature types. The wire carries one code plus a residual.

    Entries are unit vectors updated by EMA, the same way a VQ-VAE codebook
    tracks encoder outputs. A low temperature makes a token that sits on its
    code win the softmax instead of blending all 4096 entries.
    """

    def __init__(self, codes: int = CODES, dim: int = CODE_DIM, tau: float = 0.07, decay: float = 0.99):
        super().__init__()
        self.tau = tau
        self.decay = decay
        book = F.normalize(torch.randn(codes, dim), dim=-1)
        self.register_buffer("book", book)
        self.register_buffer("embed_avg", book.clone())
        self.register_buffer("cluster_size", torch.zeros(codes))
        self.last_commit = None
        self.last_entropy = None
        self.last_token_h = None
        self.last_unique = 0

    def assign(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat = F.normalize(tokens.reshape(-1, tokens.shape[-1]), dim=-1)
        book = F.normalize(self.book, dim=-1)
        logits = flat @ book.T / self.tau
        weights = logits.softmax(dim=-1)
        chosen = (weights @ book).reshape_as(tokens)
        return chosen, weights

    def quantize(self, tokens: torch.Tensor) -> torch.Tensor:
        flat = F.normalize(tokens.reshape(-1, tokens.shape[-1]), dim=-1)
        _soft, weights = self.assign(tokens)
        index = weights.argmax(dim=-1)
        hard = F.normalize(self.book, dim=-1)[index].reshape_as(tokens)
        direction = flat.reshape_as(tokens)
        quantized = direction + (hard - direction).detach()
        self.last_commit = F.mse_loss(direction, hard.detach())
        log_w = weights.clamp_min(1e-8).log()
        # Low per-token entropy means the token picked a code. High average
        # entropy means those picks are spread across the book. A flat softmax
        # hides collapse if only the average is measured.
        self.last_token_h = -(weights * log_w).sum(dim=-1).mean() / math.log(self.book.shape[0])
        average = weights.mean(dim=0).clamp_min(1e-8)
        self.last_entropy = -(average * average.log()).sum() / math.log(self.book.shape[0])
        self.last_unique = int(torch.unique(index.detach()).numel())
        if self.training:
            self._ema(flat, index.detach().reshape(-1))
        return quantized

    @torch.no_grad()
    def _ema(self, flat: torch.Tensor, index: torch.Tensor) -> None:
        onehot = torch.zeros(flat.shape[0], self.book.shape[0], device=flat.device, dtype=flat.dtype)
        onehot.scatter_(1, index[:, None], 1.0)
        self.cluster_size.mul_(self.decay).add_(onehot.sum(0), alpha=1.0 - self.decay)
        self.embed_avg.mul_(self.decay).add_(onehot.T @ flat, alpha=1.0 - self.decay)
        n = self.cluster_size.sum().clamp_min(1.0)
        smoothed = (self.cluster_size + 1e-5) / (n + self.book.shape[0] * 1e-5) * n
        self.book.copy_(self.embed_avg / smoothed.unsqueeze(1).clamp_min(1e-5))
        # Recycle unused entries onto live tokens, with noise so copies do not tie.
        dead = torch.nonzero(self.cluster_size < 1e-3).flatten()
        if dead.numel() == 0:
            return
        take = min(int(dead.numel()), int(flat.shape[0]))
        pick = torch.randperm(flat.shape[0], device=flat.device)[:take]
        noise = 0.15 * torch.randn(take, flat.shape[-1], device=flat.device, dtype=flat.dtype)
        fresh = F.normalize(flat[pick] + noise, dim=-1)
        self.book[dead[:take]] = fresh
        self.embed_avg[dead[:take]] = fresh
        self.cluster_size[dead[:take]] = 1.0


class CodebookCodec(nn.Module):
    architecture = "codebook-jscc-v1"

    def __init__(self, codes: int = CODES, dim: int = CODE_DIM, width: int = 128):
        super().__init__()
        if SPATIAL_TOKENS + SEMANTIC_TOKENS != TOKENS or TOKENS * dim != GOP_VALUES:
            raise RuntimeError("token layout does not fill 2816 reals")
        self.codes = int(codes)
        self.dim = int(dim)
        self.width = int(width)
        self.codebook = Codebook(codes, dim)
        self.queries = nn.Parameter(torch.randn(1, SEMANTIC_TOKENS, width) * 0.02)
        self.state_dim = 64
        self.state_to_query = nn.Linear(self.state_dim, width)
        self.to_state = nn.Linear(width, self.state_dim)
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            ResBlock(64),
            nn.Conv3d(64, 64, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            ResBlock(64),
            nn.Conv3d(64, 96, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            ResBlock(96),
            nn.Conv3d(96, width, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            ResBlock(width),
        )
        self.select = Mixer(width)
        self.semantic = Mixer(width)
        self.mix = Mixer(width)
        self.gate = nn.Linear(width, 1)
        self.to_code = nn.Linear(width, dim)
        # sigmoid(0) = 0.5, and the scale below caps the residual at 0.4.
        self.residual_mix = nn.Parameter(torch.tensor(0.0))
        self.fuse = nn.Sequential(
            nn.Linear(dim * 2 + 1, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(width, 96, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            ResBlock(96),
            nn.ConvTranspose3d(96, 64, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            ResBlock(64),
            nn.ConvTranspose3d(64, 64, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            ResBlock(64),
            nn.ConvTranspose3d(64, 32, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.GELU(),
            nn.Conv3d(32, 3, 3, padding=1),
        )
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)
        self._tx_state: torch.Tensor | None = None
        self._rx_state: torch.Tensor | None = None

    def reset(self) -> None:
        self._tx_state = None
        self._rx_state = None

    def _tokens(self, video: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        features = self.encoder(pad_gop(video))
        if features.shape[-3:] != (1, 7, 12):
            raise RuntimeError(f"encoder grid {tuple(features.shape[-3:])} is not 1x7x12")
        spatial = features.flatten(2).transpose(1, 2)
        spatial = self.select(spatial)
        queries = self.queries.expand(video.shape[0], -1, -1) + self.state_to_query(state).unsqueeze(1)
        semantic = self.semantic(queries, spatial)
        tokens = self.mix(torch.cat([spatial, semantic], dim=1))
        scores = self.gate(tokens).squeeze(-1)
        amplitude = torch.sigmoid(scores) * 1.5 + 0.25
        amplitude = amplitude / amplitude.mean(dim=-1, keepdim=True).clamp_min(1e-4)
        return tokens * amplitude.unsqueeze(-1), semantic

    def encode_gop(self, video: torch.Tensor, retain_state: bool = True, **kwargs) -> torch.Tensor:
        del kwargs
        if video.ndim != 5 or video.shape[1:] != (3, 4, 108, 192):
            raise ValueError(f"expected (B, 3, 4, 108, 192), got {tuple(video.shape)}")
        state = self._tx_state
        if state is None or state.shape[0] != video.shape[0]:
            state = video.new_zeros(video.shape[0], self.state_dim)
        tokens, semantic = self._tokens(video, state)
        coded = self.to_code(tokens)
        quantized = self.codebook.quantize(coded)
        # The code is a unit direction. A capped residual carries what that
        # direction does not, without rotating the token off its codebook entry.
        scale = 0.4 * torch.sigmoid(self.residual_mix)
        residual = F.normalize(coded, dim=-1) - quantized
        payload = quantized + scale * residual
        if retain_state:
            self._tx_state = self.to_state(semantic.mean(dim=1))
        return unit_rms(payload.reshape(video.shape[0], GOP_VALUES))

    def decode_gop(
        self,
        wire: torch.Tensor,
        confidence: torch.Tensor | None = None,
        retain_state: bool = True,
        **kwargs,
    ):
        del kwargs
        if wire.shape[-1] != GOP_VALUES:
            raise ValueError(f"wire length {wire.shape[-1]} != {GOP_VALUES}")
        if confidence is None:
            confidence = torch.ones_like(wire)
        batch = wire.shape[0]
        tokens = wire.reshape(batch, TOKENS, self.dim)
        conf = confidence.reshape(batch, TOKENS, self.dim).mean(dim=-1, keepdim=True)
        prior = self.codebook.assign(tokens)[0]
        fused = self.fuse(torch.cat([tokens, prior, conf], dim=-1))
        state = self._rx_state
        if state is None or state.shape[0] != batch:
            state = fused.new_zeros(batch, self.state_dim)
        fused = fused + self.state_to_query(state).unsqueeze(1)
        spatial = fused[:, :SPATIAL_TOKENS]
        semantic = fused[:, SPATIAL_TOKENS:]
        volume = spatial.transpose(1, 2).reshape(batch, self.width, 1, 7, 12)
        film = semantic.mean(dim=1)
        volume = volume * (1 + film[:, :, None, None, None].tanh())
        recon = unpad_gop(self.decoder(volume) + 0.5).clamp(0, 1)
        if retain_state:
            self._rx_state = self.to_state(semantic.mean(dim=1))
        outage = wire.new_zeros(batch, 4)
        return recon, outage

    def aux_loss(self) -> torch.Tensor:
        commit = self.codebook.last_commit
        token_h = self.codebook.last_token_h
        usage_h = self.codebook.last_entropy
        if commit is None or token_h is None or usage_h is None:
            return torch.zeros((), device=self.codebook.book.device)
        return commit + token_h + (1.0 - usage_h)


def load_codebook_codec(
    path: str | Path, device: torch.device | str = "cpu"
) -> tuple[CodebookCodec, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = CodebookCodec(
        codes=int(payload.get("codes", CODES)),
        dim=int(payload.get("dim", CODE_DIM)),
        width=int(payload.get("width", 128)),
    )
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    return model, payload
