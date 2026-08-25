"""AETV (Autoencoder Television): analog video over HF OFDM.

Training-only PyTorch modules are loaded lazily so the portable operator app
can use the compact ONNX Runtime without accidentally bundling PyTorch.
"""

from importlib import import_module

from .config import (
    AETV_MODES,
    AETV_MODES_BY_INDEX,
    RELEASE_MODES,
    RELEASE_MODE_LABELS,
    AETVBandGeometry,
    AETVModeSpec,
    BAND_N,
    BAND_U,
    BAND_W,
    BANDS,
    FS,
    RS,
    M,
    NCP,
    NSYM,
    SYMS_PER_FRAME,
    DATA_SYMS_PER_FRAME,
    FRAME_SAMPLES,
    FRAMES_PER_GOP,
    LATENTS_PER_GOP_N,
    LATENTS_PER_GOP_W,
    LATENTS_PER_GOP_U,
)
_LAZY_EXPORTS = {
    "AETVChannelConfig": (".channel", "AETVChannelConfig"),
    "AETVLatentChannel": (".channel", "AETVLatentChannel"),
    "AETVWaveformChannel": (".channel", "AETVWaveformChannel"),
    "AETVOpenVidStreamDataset": (".data", "AETVOpenVidStreamDataset"),
    "AETVSyntheticVideoDataset": (".data", "AETVSyntheticVideoDataset"),
    "AETVAutoencoder": (".models", "AETVAutoencoder"),
    "AETVEncoder": (".models", "AETVEncoder"),
    "AETVDecoder": (".models", "AETVDecoder"),
    "ShallowVGGPerceptualLoss": (".models", "ShallowVGGPerceptualLoss"),
    "SpatioTemporalDiscriminator3D": (".models", "SpatioTemporalDiscriminator3D"),
    "SpatioTemporalPatchGAN3D": (".models", "SpatioTemporalPatchGAN3D"),
    "AETVDemodResult": (".modem", "AETVDemodResult"),
    "StreamingDemodulator": (".modem", "StreamingDemodulator"),
    "demodulate_gop_stream": (".modem", "demodulate_gop_stream"),
    "modulate_gop_chunks": (".modem", "modulate_gop_chunks"),
    "modulate_gop_stream": (".modem", "modulate_gop_stream"),
    "Acquisition": (".sync", "Acquisition"),
    "SyncError": (".sync", "SyncError"),
    "acquire": (".sync", "acquire"),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value

__all__ = [
    "AETV_MODES",
    "AETV_MODES_BY_INDEX",
    "RELEASE_MODES",
    "RELEASE_MODE_LABELS",
    "AETVBandGeometry",
    "AETVModeSpec",
    "BAND_N",
    "BAND_U",
    "BAND_W",
    "BANDS",
    "FS",
    "RS",
    "M",
    "NCP",
    "NSYM",
    "SYMS_PER_FRAME",
    "DATA_SYMS_PER_FRAME",
    "FRAME_SAMPLES",
    "FRAMES_PER_GOP",
    "LATENTS_PER_GOP_N",
    "LATENTS_PER_GOP_W",
    "LATENTS_PER_GOP_U",
    "AETVChannelConfig",
    "AETVLatentChannel",
    "AETVWaveformChannel",
    "AETVOpenVidStreamDataset",
    "AETVSyntheticVideoDataset",
    "AETVAutoencoder",
    "AETVEncoder",
    "AETVDecoder",
    "ShallowVGGPerceptualLoss",
    "SpatioTemporalDiscriminator3D",
    "SpatioTemporalPatchGAN3D",
    "AETVDemodResult",
    "modulate_gop_stream",
    "modulate_gop_chunks",
    "StreamingDemodulator",
    "demodulate_gop_stream",
    "Acquisition",
    "SyncError",
    "acquire",
]
