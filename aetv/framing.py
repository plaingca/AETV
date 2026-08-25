"""Framing and interleaver mapping for AETV.

Provides per-GOP interleaving and progressive group structuring for both
Variant N (1,472 latents/GOP) and Variant W (2,816 latents/GOP).
"""

from __future__ import annotations

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # Portable operator builds intentionally omit PyTorch.
    torch = None

from .config import (
    BANDS,
    DATA_SYMS_PER_FRAME,
    FRAMES_PER_GOP,
    LATENTS_PER_GOP_N,
    LATENTS_PER_GOP_W,
    LATENTS_PER_GOP_U,
)

AETV_INTERLEAVER_SEED = 20260818


def derive_gop_interleaver(
    latents_per_gop: int, seed: int = AETV_INTERLEAVER_SEED
) -> np.ndarray:
    """Deterministic permutation mapping GOP latents to OFDM time-frequency slots."""
    rng = np.random.default_rng(seed)
    return rng.permutation(latents_per_gop).astype(np.int32)


# Precomputed GOP interleaver permutations for N, W, and U
GOP_INTERLEAVER_N = derive_gop_interleaver(LATENTS_PER_GOP_N)
GOP_INTERLEAVER_W = derive_gop_interleaver(LATENTS_PER_GOP_W)
GOP_INTERLEAVER_U = derive_gop_interleaver(LATENTS_PER_GOP_U)

GOP_DEINTERLEAVER_N = np.empty_like(GOP_INTERLEAVER_N)
GOP_DEINTERLEAVER_N[GOP_INTERLEAVER_N] = np.arange(len(GOP_INTERLEAVER_N))

GOP_DEINTERLEAVER_W = np.empty_like(GOP_INTERLEAVER_W)
GOP_DEINTERLEAVER_W[GOP_INTERLEAVER_W] = np.arange(len(GOP_INTERLEAVER_W))

GOP_DEINTERLEAVER_U = np.empty_like(GOP_INTERLEAVER_U)
GOP_DEINTERLEAVER_U[GOP_INTERLEAVER_U] = np.arange(len(GOP_INTERLEAVER_U))



def pack_gop_symbols(
    latents: np.ndarray | torch.Tensor,
    beacon_chips: np.ndarray | torch.Tensor,
    band: str = "W",
    interleave: bool = True,
) -> np.ndarray:
    """Pack 1 GOP (8 frames) of latents and beacon chips into (32, NC) complex symbols.
    
    Returns array of shape (32, NC) complex64.
    """
    is_torch = torch is not None and isinstance(latents, torch.Tensor)
    if is_torch:
        latents = latents.detach().cpu().numpy()
        beacon_chips = beacon_chips.detach().cpu().numpy()
    
    geom = BANDS[band]
    if len(latents) != geom.latents_per_gop:
        raise ValueError(
            f"expected {geom.latents_per_gop} latents for band {band}, got {len(latents)}"
        )
    
    # Interleave
    if interleave:
        if band == "N":
            perm = GOP_INTERLEAVER_N
        elif band == "W":
            perm = GOP_INTERLEAVER_W
        else:
            perm = GOP_INTERLEAVER_U
        tx_latents = latents[perm]
    else:
        tx_latents = latents
    
    # Pack into (8 frames * 4 data syms = 32 data symbols) x NC carriers
    n_data_syms = FRAMES_PER_GOP * DATA_SYMS_PER_FRAME  # 32
    symbols = np.zeros((n_data_syms, geom.carriers), dtype=np.complex64)
    
    # Pair I and Q into complex numbers
    complex_latents = tx_latents[0::2] + 1j * tx_latents[1::2]  # len = latents_per_gop // 2
    
    # Fill data carriers (0..NC-2)
    latent_matrix = complex_latents.reshape(n_data_syms, geom.latent_carriers)
    symbols[:, :geom.latent_carriers] = latent_matrix
    
    # V7/U has one otherwise-unused guard carrier. Carry four real beacon chips
    # per data symbol on I/Q of that carrier and the normal beacon carrier.
    # Unit I/Q gives each chip 3 dB more energy than normalized QPSK. These are
    # only two of 160 carriers, so the overall waveform power increase is tiny
    # while the OTA beacon gets useful margin at the Golay correction boundary.
    if band == "U" and len(beacon_chips) >= 4 * n_data_syms:
        # Legacy v3 four-lane fast beacon, retained for decoding old captures.
        scale = np.float32(1.0)
        chips = beacon_chips[: 4 * n_data_syms].reshape(n_data_syms, 4)
        symbols[:, geom.latent_carriers] = scale * (chips[:, 0] + 1j * chips[:, 1])
        symbols[:, geom.beacon_carrier] = scale * (chips[:, 2] + 1j * chips[:, 3])
    elif band == "U" and len(beacon_chips) >= n_data_syms:
        # Repeat each logical beacon chip on I/Q of both reserved carriers.
        # Combining provides 6 dB without consuming more RF time. Unit I/Q
        # keeps each reserved carrier at the same average power as a latent
        # carrier; the former 2x amplitude unnecessarily drove the clipper.
        chips = np.asarray(beacon_chips[:n_data_syms], dtype=np.float32)
        repeated = chips + 1j * chips
        symbols[:, geom.latent_carriers] = repeated
        symbols[:, geom.beacon_carrier] = repeated
    elif len(beacon_chips) >= n_data_syms:
        symbols[:, geom.beacon_carrier] = beacon_chips[:n_data_syms]
    else:
        symbols[:len(beacon_chips), geom.beacon_carrier] = beacon_chips
        
    return symbols


def unpack_gop_symbols(
    data_symbols: np.ndarray,
    data_weights: np.ndarray,
    band: str = "W",
    interleave: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Unpack (32, NC) complex data symbols and weights into GOP latents.
    
    Returns (latents, weights) each of shape (latents_per_gop,).
    """
    geom = BANDS[band]
    n_data_syms = FRAMES_PER_GOP * DATA_SYMS_PER_FRAME  # 32
    
    # Extract latent carriers (0..NC-2)
    latent_syms = data_symbols[:n_data_syms, :geom.latent_carriers].reshape(-1)
    latent_w = data_weights[:n_data_syms, :geom.latent_carriers].reshape(-1)
    
    # Unpack I and Q
    raw_latents = np.empty(geom.latents_per_gop, dtype=np.float32)
    raw_latents[0::2] = np.real(latent_syms)
    raw_latents[1::2] = np.imag(latent_syms)
    
    raw_weights = np.empty(geom.latents_per_gop, dtype=np.float32)
    raw_weights[0::2] = latent_w
    raw_weights[1::2] = latent_w
    
    # De-interleave
    if interleave:
        if band == "N":
            inv_perm = GOP_DEINTERLEAVER_N
        elif band == "W":
            inv_perm = GOP_DEINTERLEAVER_W
        else:
            inv_perm = GOP_DEINTERLEAVER_U
        latents = raw_latents[inv_perm]
        weights = raw_weights[inv_perm]
    else:
        latents = raw_latents
        weights = raw_weights
        
    return latents, weights
