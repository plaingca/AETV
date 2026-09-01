"""Perceptual audio metrics shared by analog AETV experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> torch.Tensor:
    def hz_to_mel(hz: torch.Tensor) -> torch.Tensor:
        return 2595.0 * torch.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
        return 700.0 * (torch.pow(10.0, mel / 2595.0) - 1.0)

    mel_points = torch.linspace(
        hz_to_mel(torch.tensor(fmin)), hz_to_mel(torch.tensor(fmax)), n_mels + 2
    )
    hz_points = mel_to_hz(mel_points)
    bins = torch.floor((n_fft + 1) * hz_points / sample_rate).long()
    filters = torch.zeros(n_mels, n_fft // 2 + 1)
    for index in range(n_mels):
        left, center, right = bins[index : index + 3]
        if center > left:
            filters[index, left:center] = torch.linspace(0, 1, center - left)
        if right > center:
            filters[index, center:right] = torch.linspace(1, 0, right - center)
    return filters


class AudioPerceptualLoss(nn.Module):
    """MR-STFT, log-mel, SI-SDR, and waveform reconstruction metrics."""

    def __init__(
        self,
        sample_rate: int = 8_000,
        si_sdr_weight: float = 0.15,
        low_hz: float = 200.0,
        high_hz: float = 2_700.0,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.si_sdr_weight = si_sdr_weight
        self.resolutions = ((512, 120, 400), (256, 60, 200), (128, 30, 100))
        for n_fft, _, window_length in self.resolutions:
            self.register_buffer(f"window_{n_fft}", torch.hann_window(window_length))
        self.register_buffer(
            "mel_filters",
            _mel_filterbank(sample_rate, 512, 40, low_hz, high_hz),
        )

    @staticmethod
    def _si_sdr_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred - pred.mean(dim=-1, keepdim=True)
        target = target - target.mean(dim=-1, keepdim=True)
        scale = (pred * target).sum(-1, keepdim=True) / target.square().sum(
            -1, keepdim=True
        ).clamp_min(1e-7)
        projection = scale * target
        noise = pred - projection
        ratio = projection.square().sum(-1) / noise.square().sum(-1).clamp_min(1e-7)
        return -10.0 * torch.log10(ratio.clamp_min(1e-7)).mean()

    def _magnitude(
        self,
        audio: torch.Tensor,
        n_fft: int,
        hop: int,
        window_length: int,
    ) -> torch.Tensor:
        return torch.stft(
            audio.float(),
            n_fft=n_fft,
            hop_length=hop,
            win_length=window_length,
            window=getattr(self, f"window_{n_fft}"),
            return_complex=True,
        ).abs().clamp_min(1e-7)

    def components(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        spectral = pred.new_zeros(())
        log_spectral = pred.new_zeros(())
        for n_fft, hop, window_length in self.resolutions:
            pred_magnitude = self._magnitude(pred, n_fft, hop, window_length)
            target_magnitude = self._magnitude(target, n_fft, hop, window_length)
            spectral = spectral + (pred_magnitude - target_magnitude).norm(
                p="fro"
            ) / target_magnitude.norm(p="fro").clamp_min(1e-7)
            log_spectral = log_spectral + F.l1_loss(
                pred_magnitude.log(), target_magnitude.log()
            )
        spectral = spectral / len(self.resolutions)
        log_spectral = log_spectral / len(self.resolutions)

        pred_512 = self._magnitude(pred, 512, 120, 400)
        target_512 = self._magnitude(target, 512, 120, 400)
        mel = F.l1_loss(
            torch.log(
                torch.einsum("mf,bft->bmt", self.mel_filters, pred_512).clamp_min(1e-7)
            ),
            torch.log(
                torch.einsum("mf,bft->bmt", self.mel_filters, target_512).clamp_min(1e-7)
            ),
        )
        return {
            "mr_stft": spectral + log_spectral,
            "mel": mel,
            "si_sdr": self._si_sdr_loss(pred, target),
            "waveform": F.l1_loss(pred, target),
        }

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = self.components(pred, target)
        return (
            losses["mr_stft"]
            + 0.5 * losses["mel"]
            + self.si_sdr_weight * losses["si_sdr"]
            + 2.0 * losses["waveform"]
        )
