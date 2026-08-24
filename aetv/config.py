"""Central waveform, framing, and mode definitions for AETV.

Autoencoder Television (AETV): Live HF video from autoencoder latents
over 8 kHz soundcard audio on HF SSB channels.
"""

from __future__ import annotations

from dataclasses import dataclass

FS = 8000  # audio sample rate, Hz

# --- OFDM waveform numerology (Display-Friendly 8 fps) --------------------
RS = 50  # carrier spacing == symbol rate of one carrier, Hz
M = FS // RS  # useful symbol length, samples (160 samples = 20 ms)
NCP = 40  # cyclic prefix, samples (5.0 ms @ 8 kHz)
NSYM = M + NCP  # full symbol length, samples (200 samples = 25 ms, 40 sym/s)

# --- Framing ---------------------------------------------------------------
# 5 symbols per frame = 1 pilot + 4 data = 1000 samples = 125 ms = exactly 8 frames/s
SYMS_PER_FRAME = 5
DATA_SYMS_PER_FRAME = SYMS_PER_FRAME - 1  # 4 data symbols / frame
FRAME_SAMPLES = SYMS_PER_FRAME * NSYM  # 1000 samples = 125 ms

# GOP structure: 8 OFDM frames = 1.000 s = 8000 samples = 1 GOP
FRAMES_PER_GOP = 8
GOP_SAMPLES = FRAMES_PER_GOP * FRAME_SAMPLES  # 8000 samples = 1.000 s

# Bandwidth Variants:
# Variant N (Narrow): 24 carriers (23 latent + 1 beacon), 950-2100 Hz (~1.2 kHz)
NC_N = 24
CARRIER0_N = 950
FCENTER_N = 1600
BEACON_CARRIER_N = NC_N - 1  # index 23
NC_LATENT_N = NC_N - 1  # 23
LATENTS_PER_FRAME_N = NC_LATENT_N * DATA_SYMS_PER_FRAME * 2  # 23 * 4 * 2 = 184 real values
LATENTS_PER_GOP_N = FRAMES_PER_GOP * LATENTS_PER_FRAME_N  # 8 * 184 = 1472 real values

# Variant W (Wide): 45 carriers (44 latent + 1 beacon), 450-2650 Hz (~2.25 kHz)
NC_W = 45
CARRIER0_W = 450
FCENTER_W = 1600
BEACON_CARRIER_W = NC_W - 1  # index 44
NC_LATENT_W = NC_W - 1  # 44
LATENTS_PER_FRAME_W = NC_LATENT_W * DATA_SYMS_PER_FRAME * 2  # 44 * 4 * 2 = 352 real values
LATENTS_PER_GOP_W = FRAMES_PER_GOP * LATENTS_PER_FRAME_W  # 8 * 352 = 2816 real values

# Variant U (Ultra-Wide / Flex-8k): 160 carriers (158 latent + 1 beacon + 1 guard), 1000-9000 Hz (~8.0 kHz)
NC_U = 160
CARRIER0_U = 1000
FCENTER_U = 5000
BEACON_CARRIER_U = NC_U - 1  # index 159
NC_LATENT_U = NC_U - 2  # 158
LATENTS_PER_FRAME_U = NC_LATENT_U * DATA_SYMS_PER_FRAME * 2  # 158 * 4 * 2 = 1264 real values
LATENTS_PER_GOP_U = FRAMES_PER_GOP * LATENTS_PER_FRAME_U  # 8 * 1264 = 10112 real values
FS_U = 24000  # 24 kHz audio sample rate for Ultra-Wide / Flex VITA-49

# --- Beacon side-channel ---------------------------------------------------
# Base beacon rate is 4 chips/frame = 32 chips/s. U/V7 uses I/Q on its
# beacon and spare guard carriers for four lanes = 128 chips/s.
BEACON_CHIPS_PER_FRAME = DATA_SYMS_PER_FRAME  # 4
BEACON_SYNC = (1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1)  # Barker-13
BEACON_COUNTER_BITS = 10  # mod 1024 (128 s wrap @ 8 frames/s)
BEACON_CALLSIGN_CHARS = 8
BEACON_CALLSIGN_CHAR_BITS = 6  # 64-symbol alphabet
BEACON_CALLSIGN_BITS = BEACON_CALLSIGN_CHARS * BEACON_CALLSIGN_CHAR_BITS  # 48
BEACON_MODE_BITS = 4  # 0..15 mode index
BEACON_CRC_BITS = 16
BEACON_PAYLOAD_BITS = BEACON_COUNTER_BITS + BEACON_CALLSIGN_BITS + BEACON_MODE_BITS + BEACON_CRC_BITS  # 78 bits
# Superframe length in chips = sync (13) + payload (78) = 91 chips (~22.75 frames = 2.84 s)
BEACON_SUPERFRAME_CHIPS = len(BEACON_SYNC) + BEACON_PAYLOAD_BITS

# --- Preamble & Sync -------------------------------------------------------
PREAMBLE_REPEATS = 12
PREAMBLE_CP = 2 * NCP  # 80 samples (10.0 ms)
PREAMBLE_SAMPLES = PREAMBLE_CP + PREAMBLE_REPEATS * M  # 2,000 samples (250 ms)
PREAMBLE_CORR_WINDOW = (PREAMBLE_REPEATS - 1) * M  # 1,760 samples
PREAMBLE_THRESHOLD = 0.42
ACQUIRE_MAX_BINS = 12
TEMPLATE_SCORE_THRESHOLD = 0.14

HEADER_SYMS = 8  # repeated Golay-coded BPSK symbols
HEADER_SAMPLES = HEADER_SYMS * NSYM  # 1,600 samples

LEADIN_SAMPLES = 800  # 100 ms lead-in
LEADOUT_SAMPLES = 800

# --- TX & Channel conditioning ---------------------------------------------
CLIP_HEADROOM_DB = 0.5
TX_BANDPASS_N = (850.0, 2200.0)  # Hz
TX_BANDPASS_W = (350.0, 2750.0)  # Hz
TX_BANDPASS_U = (500.0, 9500.0)  # Hz; margin around 1000..8950 Hz carriers
DEMOD_BACKOFF = 8  # samples
SNR_REF_BW_HZ = 2500.0
PROTOCOL_VERSION = 4


def reference_noise_bandwidth_scale(fs: int | float) -> float:
    """White-noise power multiplier for the 2.5 kHz SNR convention."""
    return (float(fs) / 2.0) / SNR_REF_BW_HZ

# --- Drift tracking loop ---------------------------------------------------
DRIFT_SLOW_ALPHA = 0.1
DRIFT_SLOW_BETA = 0.01
DRIFT_FAST_ALPHA = 0.3
DRIFT_FAST_BETA = 0.05
DRIFT_TRACK_MODES = ("off", "slow", "fast")

# --- Pilot Quadrants -------------------------------------------------------
PILOT_QUADRANTS_24 = (
    0, 3, 2, 1, 1, 3, 0, 2, 0, 0, 2, 3,
    2, 3, 2, 3, 2, 0, 3, 1, 2, 1, 0, 3,
)

PILOT_QUADRANTS_45 = (
    0, 3, 2, 0, 2, 3, 0, 1, 2, 0, 1, 2, 3, 3, 0,
    0, 2, 3, 1, 2, 2, 2, 0, 3, 2, 1, 2, 0, 3, 1,
    0, 3, 1, 0, 0, 1, 0, 2, 3, 2, 3, 2, 0, 3, 3,
)

PILOT_QUADRANTS_160 = (
    2, 2, 1, 1, 0, 1, 2, 3, 0, 2, 1, 3, 0, 2, 1, 3, 1, 2, 3, 2, 1, 1, 0, 3, 3, 1, 2, 2, 0, 0, 0, 1,
    0, 2, 1, 3, 3, 3, 3, 3, 1, 1, 3, 0, 3, 0, 0, 3, 2, 0, 1, 0, 2, 0, 3, 1, 3, 3, 3, 0, 0, 2, 1, 1,
    2, 1, 3, 1, 2, 2, 0, 3, 0, 3, 3, 3, 0, 2, 0, 0, 2, 1, 2, 3, 2, 1, 0, 2, 1, 1, 3, 3, 3, 1, 2, 0,
    1, 1, 1, 3, 1, 3, 0, 1, 0, 1, 3, 0, 1, 1, 1, 0, 0, 3, 3, 1, 1, 0, 0, 2, 0, 3, 3, 1, 1, 3, 1, 1,
    3, 1, 3, 2, 3, 3, 2, 1, 2, 1, 3, 2, 1, 3, 2, 1, 0, 3, 3, 2, 2, 2, 3, 2, 0, 3, 3, 1, 3, 3, 3, 3,
)


@dataclass(frozen=True)
class AETVBandGeometry:
    name: str  # "N", "W", or "U"
    carriers: int
    carrier0_hz: int
    fcenter_hz: int
    beacon_carrier: int
    latent_carriers: int
    latents_per_frame: int
    latents_per_gop: int
    tx_bandpass: tuple[float, float]
    pilot_quadrants: tuple[int, ...]
    fs: int = FS


BAND_N = AETVBandGeometry(
    name="N",
    carriers=NC_N,
    carrier0_hz=CARRIER0_N,
    fcenter_hz=FCENTER_N,
    beacon_carrier=BEACON_CARRIER_N,
    latent_carriers=NC_LATENT_N,
    latents_per_frame=LATENTS_PER_FRAME_N,
    latents_per_gop=LATENTS_PER_GOP_N,
    tx_bandpass=TX_BANDPASS_N,
    pilot_quadrants=PILOT_QUADRANTS_24,
    fs=FS,
)

BAND_W = AETVBandGeometry(
    name="W",
    carriers=NC_W,
    carrier0_hz=CARRIER0_W,
    fcenter_hz=FCENTER_W,
    beacon_carrier=BEACON_CARRIER_W,
    latent_carriers=NC_LATENT_W,
    latents_per_frame=LATENTS_PER_FRAME_W,
    latents_per_gop=LATENTS_PER_GOP_W,
    tx_bandpass=TX_BANDPASS_W,
    pilot_quadrants=PILOT_QUADRANTS_45,
    fs=FS,
)

BAND_U = AETVBandGeometry(
    name="U",
    carriers=NC_U,
    carrier0_hz=CARRIER0_U,
    fcenter_hz=FCENTER_U,
    beacon_carrier=BEACON_CARRIER_U,
    latent_carriers=NC_LATENT_U,
    latents_per_frame=LATENTS_PER_FRAME_U,
    latents_per_gop=LATENTS_PER_GOP_U,
    tx_bandpass=TX_BANDPASS_U,
    pilot_quadrants=PILOT_QUADRANTS_160,
    fs=FS_U,
)

BANDS = {"N": BAND_N, "W": BAND_W, "U": BAND_U}



@dataclass(frozen=True)
class AETVModeSpec:
    name: str  # e.g. "V0", "V1", etc.
    index: int
    band: str  # "N", "W", or "U"
    width: int
    height: int
    fps: float
    gop_frames: int  # video frames per 1.0 s GOP
    latents_per_gop: int
    causal: bool = False
    description: str = ""

    @property
    def geometry(self) -> AETVBandGeometry:
        return BANDS[self.band]

    @property
    def latents_per_frame(self) -> float:
        return self.latents_per_gop / self.gop_frames

    @property
    def pixels_per_latent(self) -> float:
        luma_px = self.width * self.height * self.gop_frames
        return luma_px / self.latents_per_gop


AETV_MODES: dict[str, AETVModeSpec] = {
    "V0": AETVModeSpec(
        name="V0",
        index=0,
        band="N",
        width=64,
        height=48,
        fps=6.0,
        gop_frames=6,
        latents_per_gop=LATENTS_PER_GOP_N,  # 1472
        causal=False,
        description="NVIS: 64x48 color @ 6 fps, 1.2 kHz band",
    ),
    "V1": AETVModeSpec(
        name="V1",
        index=1,
        band="W",
        width=96,
        height=72,
        fps=6.0,
        gop_frames=6,
        latents_per_gop=LATENTS_PER_GOP_W,  # 2816
        causal=False,
        description="Classic: 96x72 color @ 6 fps, 2.25 kHz band",
    ),
    "V2": AETVModeSpec(
        name="V2",
        index=2,
        band="W",
        width=96,
        height=72,
        fps=12.0,
        gop_frames=12,
        latents_per_gop=LATENTS_PER_GOP_W,  # 2816
        causal=False,
        description="Motion: 96x72 color @ 12 fps, 2.25 kHz band",
    ),
    "V3": AETVModeSpec(
        name="V3",
        index=3,
        band="W",
        width=160,
        height=120,
        fps=6.0,
        gop_frames=6,
        latents_per_gop=LATENTS_PER_GOP_W,  # 2816
        causal=False,
        description="Detail: 160x120 color @ 6 fps, 2.25 kHz band",
    ),
    "V4": AETVModeSpec(
        name="V4",
        index=4,
        band="W",
        width=320,
        height=240,
        fps=1.0,
        gop_frames=1,
        latents_per_gop=LATENTS_PER_GOP_W,  # 2816
        causal=False,
        description="Still+: 320x240 color @ 1 fps, 2.25 kHz band",
    ),
    "V5": AETVModeSpec(
        name="V5",
        index=5,
        band="W",
        width=96,
        height=72,
        fps=12.0,
        gop_frames=12,
        latents_per_gop=LATENTS_PER_GOP_W,  # 2816
        causal=True,
        description="Convo: 96x72 color @ 12 fps, low-latency causal profile",
    ),
    "V6": AETVModeSpec(
        name="V6",
        index=6,
        band="U",
        width=128,
        height=128,
        fps=10.0,
        gop_frames=10,
        latents_per_gop=LATENTS_PER_GOP_U,  # 10112
        causal=False,
        description="Flex-8k: 128x128 color @ 10 fps, 8 kHz wide channel (24 kHz audio)",
    ),
    "V7": AETVModeSpec(
        name="V7",
        index=7,
        band="U",
        width=256,
        height=144,
        fps=12.0,
        gop_frames=12,
        latents_per_gop=LATENTS_PER_GOP_U,  # 10112
        causal=False,
        description="Wide 8 kHz: 256x144 16:9 @ 12 fps for wide transmit audio",
    ),
    "V8": AETVModeSpec(
        name="V8",
        index=8,
        band="W",
        width=192,
        height=108,
        fps=6.0,
        gop_frames=6,
        latents_per_gop=LATENTS_PER_GOP_W,  # 2816
        causal=False,
        description="Standard channel: 192x108 16:9 @ 6 fps for typical HF/VHF SSB audio",
    ),
}


# Modes with pinned, checksum-verified release checkpoints. Historical modes
# remain decodable at the protocol layer, but are intentionally hidden from the
# release GUI until they have validated weights of their own.
RELEASE_MODES: tuple[str, ...] = ("V8", "V7")
RELEASE_MODE_LABELS = {
    "V8": "Standard channel — 192×108 @ 6 fps",
    "V7": "Wide 8 kHz — 256×144 @ 12 fps",
}


AETV_MODES_BY_INDEX: dict[int, AETVModeSpec] = {m.index: m for m in AETV_MODES.values()}
