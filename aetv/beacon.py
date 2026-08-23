"""Beacon side-channel for AETV: continuous self-describing sync, counter, callsign, and mode ID.

Carried as BPSK chips on the beacon carrier (carrier 23 for N, carrier 44 for W)
at BEACON_CHIPS_PER_FRAME (4 chips/frame) on every data symbol.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import golay
from .config import (
    BEACON_CALLSIGN_BITS,
    BEACON_CALLSIGN_CHARS,
    BEACON_CHIPS_PER_FRAME,
    BEACON_COUNTER_BITS,
    BEACON_CRC_BITS,
    BEACON_MODE_BITS,
    BEACON_SYNC,
)

SYNC = np.array(BEACON_SYNC, dtype=np.float64)
SYNC_LEN = len(SYNC)

_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/-. " + "?!@#$%^&*()_+=~[]{}<>:;,"
assert len(_ALPHABET) == 64
_CHAR_TO_CODE = {c: i for i, c in enumerate(_ALPHABET)}

_PAYLOAD_BITS = (
    BEACON_COUNTER_BITS + BEACON_CALLSIGN_BITS + BEACON_MODE_BITS + BEACON_CRC_BITS
)  # 78
N_CHUNKS = -(-_PAYLOAD_BITS // 12)  # 7
PADDED_PAYLOAD_BITS = N_CHUNKS * 12  # 84
CODED_LEN = N_CHUNKS * 24  # 168
SUPERFRAME_LEN = SYNC_LEN + CODED_LEN  # 181
MAX_FRAME_COUNTER = (1 << BEACON_COUNTER_BITS) - 1


@dataclass(frozen=True)
class AETVBeaconResult:
    chip_offset: int  # chip index where sync starts
    frame_index: int  # absolute frame index (0..1023)
    callsign: str
    mode_index: int  # 0..15 mode index (e.g. 0=V0, 1=V1, etc.)
    gop_index: int  # frame_index // 8
    gop_phase: int  # frame_index % 8


def callsign_to_codes(callsign: str) -> np.ndarray:
    s = callsign.upper()[:BEACON_CALLSIGN_CHARS].ljust(BEACON_CALLSIGN_CHARS)
    return np.array([_CHAR_TO_CODE.get(c, _CHAR_TO_CODE[" "]) for c in s])


def codes_to_callsign(codes: np.ndarray) -> str:
    return "".join(_ALPHABET[int(c) & 0x3F] for c in codes).rstrip()


def _int_to_bits(value: int, width: int) -> np.ndarray:
    return ((value >> np.arange(width - 1, -1, -1)) & 1).astype(np.int64)


def _bits_to_int(bits: np.ndarray) -> int:
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return int(v)


def crc16(bits: np.ndarray) -> int:
    crc = 0xFFFF
    for bit in bits:
        crc ^= int(bit) << 15
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def encode_superframe(frame_counter: int, callsign: str, mode_index: int = 1) -> np.ndarray:
    """Pack counter, callsign, and mode index into a 181-chip BPSK superframe."""
    counter_bits = _int_to_bits(frame_counter & MAX_FRAME_COUNTER, BEACON_COUNTER_BITS)
    codes = callsign_to_codes(callsign)
    callsign_bits = ((codes[:, None] >> np.arange(5, -1, -1)) & 1).reshape(-1)
    mode_bits = _int_to_bits(mode_index & 0x0F, BEACON_MODE_BITS)
    data_bits = np.concatenate([counter_bits, callsign_bits, mode_bits])
    crc_val = crc16(data_bits)
    crc_bits = _int_to_bits(crc_val, BEACON_CRC_BITS)
    raw_payload = np.concatenate([data_bits, crc_bits])
    padded = np.pad(raw_payload, (0, PADDED_PAYLOAD_BITS - len(raw_payload)))

    coded_chunks = []
    for i in range(N_CHUNKS):
        chunk_bits = padded[i * 12 : (i + 1) * 12]
        chunk_val = _bits_to_int(chunk_bits)
        coded_bits = golay.codeword_bits(chunk_val)
        coded_chunks.append(coded_bits)
    coded = np.concatenate(coded_chunks)
    # BPSK mapping: 0 -> +1, 1 -> -1
    chips = np.concatenate([SYNC, 1.0 - 2.0 * coded])
    return chips


def generate_beacon_chips(
    n_frames: int, start_frame: int = 0, callsign: str = "N0CALL", mode_index: int = 1
) -> np.ndarray:
    """Generate a continuous sequence of BPSK chips for n_frames frames."""
    total_chips = n_frames * BEACON_CHIPS_PER_FRAME
    chips = np.empty(total_chips, dtype=np.float64)
    chip_idx = 0
    while chip_idx < total_chips:
        frame_idx = (start_frame + chip_idx // BEACON_CHIPS_PER_FRAME) % (MAX_FRAME_COUNTER + 1)
        sf = encode_superframe(frame_idx, callsign, mode_index)
        n = min(len(sf), total_chips - chip_idx)
        chips[chip_idx : chip_idx + n] = sf[:n]
        chip_idx += len(sf)
    return chips


def decode_superframe(soft_chips: np.ndarray) -> tuple[int, str, int] | None:
    """Decode a 168-chip Golay payload; returns (frame_counter, callsign, mode_index) or None."""
    if len(soft_chips) != CODED_LEN:
        return None
    decoded_bits = []
    for i in range(N_CHUNKS):
        chunk_soft = soft_chips[i * 24 : (i + 1) * 24]
        decoded_val = golay.decode_soft(chunk_soft)
        chunk_bits = _int_to_bits(decoded_val, 12)
        decoded_bits.append(chunk_bits)
    payload = np.concatenate(decoded_bits)[:_PAYLOAD_BITS]

    counter_bits = payload[:BEACON_COUNTER_BITS]
    callsign_bits = payload[
        BEACON_COUNTER_BITS : BEACON_COUNTER_BITS + BEACON_CALLSIGN_BITS
    ]
    mode_bits = payload[
        BEACON_COUNTER_BITS
        + BEACON_CALLSIGN_BITS : BEACON_COUNTER_BITS
        + BEACON_CALLSIGN_BITS
        + BEACON_MODE_BITS
    ]
    crc_bits = payload[
        BEACON_COUNTER_BITS
        + BEACON_CALLSIGN_BITS
        + BEACON_MODE_BITS : BEACON_COUNTER_BITS
        + BEACON_CALLSIGN_BITS
        + BEACON_MODE_BITS
        + BEACON_CRC_BITS
    ]

    data_bits = np.concatenate([counter_bits, callsign_bits, mode_bits])
    expected_crc = crc16(data_bits)
    received_crc = _bits_to_int(crc_bits)
    if expected_crc != received_crc:
        return None

    counter = _bits_to_int(counter_bits)
    codes = callsign_bits.reshape(BEACON_CALLSIGN_CHARS, 6)
    code_vals = [_bits_to_int(c) for c in codes]
    callsign = codes_to_callsign(np.array(code_vals))
    mode_idx = _bits_to_int(mode_bits)
    return counter, callsign, mode_idx


def find_beacon_superframe(
    soft_stream: np.ndarray, threshold: float = 0.5
) -> AETVBeaconResult | None:
    """Scan a soft chip stream for Barker-13 sync and decode the superframe."""
    if len(soft_stream) < SUPERFRAME_LEN:
        return None
    stream = np.asarray(soft_stream, dtype=np.float64)
    windows = np.lib.stride_tricks.sliding_window_view(stream, SYNC_LEN)
    sync_norm = np.linalg.norm(SYNC)
    window_norms = np.linalg.norm(windows, axis=1)
    corr = (windows @ SYNC) / np.maximum(window_norms * sync_norm, 1e-12)
    peaks = np.where(np.abs(corr) > threshold)[0]
    peaks = peaks[np.argsort(np.abs(corr[peaks]))[::-1]]
    for peak_idx in peaks:
        if peak_idx + SUPERFRAME_LEN <= len(soft_stream):
            polarity = 1.0 if corr[peak_idx] >= 0 else -1.0
            payload_soft = polarity * stream[peak_idx + SYNC_LEN : peak_idx + SUPERFRAME_LEN]
            decoded = decode_superframe(payload_soft)
            if decoded is not None:
                counter, callsign, mode_idx = decoded
                return AETVBeaconResult(
                    chip_offset=int(peak_idx),
                    frame_index=counter,
                    callsign=callsign,
                    mode_index=mode_idx,
                    gop_index=counter // 8,
                    gop_phase=counter % 8,
                )
    return None
