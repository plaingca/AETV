"""Beacon side-channel for AETV: continuous self-describing sync, counter, callsign, and mode ID.

Carried as BPSK chips on the beacon carrier (carrier 23 for N, carrier 44 for W)
at BEACON_CHIPS_PER_FRAME (4 chips/frame) on every data symbol.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import heapq

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


def _decode_payload_values(values: np.ndarray) -> tuple[int, str, int] | None:
    padded = ((values[:, None] >> np.arange(11, -1, -1)) & 1).reshape(-1)
    if np.any(padded[_PAYLOAD_BITS:]):
        return None
    payload = padded[:_PAYLOAD_BITS]

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


def _payload_scores(soft_chips: np.ndarray, expected_mode: int | None) -> np.ndarray:
    scores = golay.soft_scores(np.asarray(soft_chips).reshape(N_CHUNKS, 24))
    messages = np.arange(4096)
    padding = PADDED_PAYLOAD_BITS - _PAYLOAD_BITS
    # These bits are transmitted as zeros, so they are useful coding evidence.
    scores[-1, (messages & ((1 << padding) - 1)) != 0] = -np.inf
    if expected_mode is not None:
        if not 0 <= expected_mode < (1 << BEACON_MODE_BITS):
            raise ValueError("expected beacon mode is outside the wire format")
        mode_start = BEACON_COUNTER_BITS + BEACON_CALLSIGN_BITS
        for bit in range(BEACON_MODE_BITS):
            word, position = divmod(mode_start + bit, 12)
            required = (expected_mode >> (BEACON_MODE_BITS - bit - 1)) & 1
            scores[word, ((messages >> (11 - position)) & 1) != required] = -np.inf
    return scores


def decode_superframe(
    soft_chips: np.ndarray, *, expected_mode: int | None = None
) -> tuple[int, str, int] | None:
    """Decode a Golay payload using its padding, then check the optional mode.

    Returns (frame_counter, callsign, mode_index) only when its CRC passes.
    The wire format and CRC are unchanged.
    """
    if len(soft_chips) != CODED_LEN or not np.all(np.isfinite(soft_chips)):
        return None
    # For a single observation, keep mode bits as independent evidence. Forcing
    # them before the CRC check increases false identification on random noise.
    # Mode-assisted soft candidates are reserved for the two-frame fallback.
    decoded = _decode_payload_values(np.argmax(_payload_scores(soft_chips, None), axis=1))
    if decoded is not None and expected_mode is not None and decoded[2] != expected_mode:
        return None
    return decoded


@lru_cache(maxsize=1)
def _crc_syndromes() -> tuple[np.ndarray, int]:
    """Tabulate the affine CRC check for bounded soft-list decoding."""
    data_len = _PAYLOAD_BITS - BEACON_CRC_BITS
    affine = crc16(np.zeros(data_len, dtype=int))
    table = np.zeros((N_CHUNKS, 4096), dtype=np.int64)
    messages = np.arange(4096)
    for bit in range(_PAYLOAD_BITS):
        basis = np.zeros(_PAYLOAD_BITS, dtype=int)
        basis[bit] = 1
        syndrome = crc16(basis[:data_len]) ^ _bits_to_int(basis[data_len:]) ^ affine
        word, position = divmod(bit, 12)
        table[word] ^= ((messages >> (11 - position)) & 1) * syndrome
    return table, affine


def _payload_candidates(
    soft_chips: np.ndarray, expected_mode: int | None, budget: int = 256
) -> list[tuple[int, str, int]]:
    """Find CRC-valid candidates among a bounded list of soft decisions.

    These candidates MUST NOT identify a station by themselves: searching more
    hypotheses weakens a single CRC check. The caller requires a second frame
    with its own observed counter/CRC, matching identity and counter spacing.
    """
    scores = _payload_scores(soft_chips, expected_mode)
    order = np.argsort(-scores, axis=1)[:, :64]
    costs = scores.max(axis=1)[:, None] - np.take_along_axis(scores, order, axis=1)
    syndromes, affine = _crc_syndromes()
    rows = np.arange(N_CHUNKS)
    initial = (0,) * N_CHUNKS
    heap = [(0.0, initial)]
    seen = {initial}
    found = []
    for _ in range(budget):
        if not heap:
            break
        cost, indices = heapq.heappop(heap)
        values = order[rows, indices]
        if np.bitwise_xor.reduce(syndromes[rows, values]) == affine:
            decoded = _decode_payload_values(values)
            if decoded is not None:
                found.append(decoded)
        for word in range(N_CHUNKS):
            next_indices = list(indices)
            next_indices[word] += 1
            following = tuple(next_indices)
            if following[word] >= order.shape[1] or following in seen:
                continue
            next_cost = cost - costs[word, indices[word]] + costs[word, following[word]]
            if np.isfinite(next_cost):
                heapq.heappush(heap, (float(next_cost), following))
                seen.add(following)
    return found


def _beacon_result(offset: int, decoded: tuple[int, str, int]) -> AETVBeaconResult:
    counter, callsign, mode_idx = decoded
    return AETVBeaconResult(
        chip_offset=int(offset), frame_index=counter, callsign=callsign,
        mode_index=mode_idx, gop_index=counter // 8, gop_phase=counter % 8,
    )


def find_beacon_superframe(
    soft_stream: np.ndarray, threshold: float = 0.5, *,
    expected_mode: int | None = None,
) -> AETVBeaconResult | None:
    """Scan for a CRC-valid beacon, then try corroborated repeated frames.

    The fallback combines only invariant whole Golay words (callsign/mode).
    Counter and CRC words remain separate observations in each frame.
    """
    if len(soft_stream) < SUPERFRAME_LEN:
        return None
    stream = np.asarray(soft_stream, dtype=np.float64)
    if not np.all(np.isfinite(stream)):
        return None
    windows = np.lib.stride_tricks.sliding_window_view(stream, SYNC_LEN)
    sync_norm = np.linalg.norm(SYNC)
    window_norms = np.linalg.norm(windows, axis=1)
    corr = (windows @ SYNC) / np.maximum(window_norms * sync_norm, 1e-12)
    peaks = np.where(np.abs(corr[:len(stream) - SUPERFRAME_LEN + 1]) > threshold)[0]
    peaks = peaks[np.argsort(np.abs(corr[peaks]))[::-1]]
    for peak_idx in peaks:
        if peak_idx + SUPERFRAME_LEN <= len(soft_stream):
            polarity = 1.0 if corr[peak_idx] >= 0 else -1.0
            payload_soft = polarity * stream[peak_idx + SYNC_LEN : peak_idx + SUPERFRAME_LEN]
            decoded = decode_superframe(payload_soft, expected_mode=expected_mode)
            if decoded is not None:
                return _beacon_result(peak_idx, decoded)

    # Bound the work even when a long or noisy stream has many sync peaks.
    # At most three superframes of history are useful to the streaming RX.
    peak_set = set(int(p) for p in peaks)
    pairs = [
        (min(abs(corr[a]), abs(corr[a + distance])), a, a + distance)
        for a in peak_set
        for distance in (SUPERFRAME_LEN, 2 * SUPERFRAME_LEN)
        if a + distance in peak_set
    ]
    # Whole words 1..4 contain only callsign and mode bits. Derive these bounds
    # so a future layout change cannot accidentally combine counters or CRCs.
    fixed_start = -(-BEACON_COUNTER_BITS // 12) * 24
    fixed_end = ((_PAYLOAD_BITS - BEACON_CRC_BITS) // 12) * 24
    for _strength, first, second in sorted(pairs, reverse=True)[:8]:
        left = stream[first + SYNC_LEN:first + SUPERFRAME_LEN] * np.sign(corr[first])
        right = stream[second + SYNC_LEN:second + SUPERFRAME_LEN] * np.sign(corr[second])
        combined = left[fixed_start:fixed_end] + right[fixed_start:fixed_end]
        left[fixed_start:fixed_end] = combined
        right[fixed_start:fixed_end] = combined
        earlier = _payload_candidates(left, expected_mode)
        if not earlier:
            continue
        later = _payload_candidates(right, expected_mode)
        distance = second - first
        steps = {distance // BEACON_CHIPS_PER_FRAME,
                 -(-distance // BEACON_CHIPS_PER_FRAME)}
        matches = {
            b for a in earlier for b in later
            if a[1:] == b[1:]
            and ((b[0] - a[0]) & MAX_FRAME_COUNTER) in steps
        }
        if len(matches) == 1:
            return _beacon_result(second, matches.pop())
    return None
