"""QRL? spectrum check: identical CW-over-SSB queries across the selected band.

Each query is a keyed audio tone as heard by a USB receiver whose suppressed
carrier sits on a 2.5 kHz slot boundary above the waveform's own carrier
(0, 2.5, 5, ... kHz). Only slots whose tone falls inside the selected mode's
transmit band are keyed, so a 20 kHz AC16 A/V selection sends eight queries
and a 2.25 kHz V1 selection sends one. Replies then show on the waterfall.
"""

from __future__ import annotations

import numpy as np

from .analog_av import composite_profile
from .config import AETV_MODES

SLOT_HZ = 2_500.0
TONE_HZ = 1_500.0  # centre of a 300-2700 Hz SSB voice channel
WPM = 20
EDGE_S = 0.005

MORSE = {
    "A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".", "F": "..-.",
    "G": "--.", "H": "....", "I": "..", "J": ".---", "K": "-.-", "L": ".-..",
    "M": "--", "N": "-.", "O": "---", "P": ".--.", "Q": "--.-", "R": ".-.",
    "S": "...", "T": "-", "U": "..-", "V": "...-", "W": ".--", "X": "-..-",
    "Y": "-.--", "Z": "--..", "0": "-----", "1": ".----", "2": "..---",
    "3": "...--", "4": "....-", "5": ".....", "6": "-....", "7": "--...",
    "8": "---..", "9": "----.", "/": "-..-.", "?": "..--..",
}


def qrl_message(callsign: str) -> str:
    return f"QRL? DE {callsign.strip().upper()}"


def qrl_band(settings) -> tuple[float, float, int]:
    """Return the selected waveform's (low Hz, high Hz, sample rate)."""
    if settings.waveform_mode == "analog_av":
        profile = composite_profile(settings.mode)
        return 0.0, float(profile.bandwidth_hz), int(profile.fs)
    geometry = AETV_MODES[settings.mode].geometry
    low, high = geometry.tx_bandpass
    return float(low), float(high), int(geometry.fs)


def qrl_offsets(low_hz: float, high_hz: float, tone_hz: float = TONE_HZ) -> list[float]:
    """Suppressed-carrier offsets of every slot whose tone lies in the band."""
    offsets = []
    k = 0
    while k * SLOT_HZ + tone_hz <= high_hz:
        if k * SLOT_HZ + tone_hz >= low_hz:
            offsets.append(k * SLOT_HZ)
        k += 1
    return offsets


def keying_envelope(text: str, fs: int, wpm: float = WPM) -> np.ndarray:
    """PARIS-timed on/off keying with raised-cosine edges against key clicks."""
    unit = round(fs * 1.2 / wpm)
    keyed: list[np.ndarray] = []
    for word_index, word in enumerate(text.upper().split()):
        if word_index:
            keyed.append(np.zeros(7 * unit))
        for char_index, char in enumerate(word):
            if char not in MORSE:
                raise ValueError(f"no Morse code for {char!r}")
            if char_index:
                keyed.append(np.zeros(3 * unit))
            for element_index, element in enumerate(MORSE[char]):
                if element_index:
                    keyed.append(np.zeros(unit))
                keyed.append(np.ones((3 if element == "-" else 1) * unit))
    edge = max(1, round(fs * EDGE_S))
    padded = np.concatenate([np.zeros(edge), *keyed, np.zeros(edge)])
    kernel = np.hanning(edge + 2)[1:-1]
    return np.convolve(padded, kernel / kernel.sum(), mode="same")


def qrl_waveform(
    callsign: str,
    fs: int,
    offsets: list[float],
    *,
    peak: float = 1.0,
    tone_hz: float = TONE_HZ,
    wpm: float = WPM,
) -> np.ndarray:
    if not offsets:
        raise ValueError("the selected band has no room for a QRL? query")
    if max(offsets) + tone_hz >= fs / 2:
        raise ValueError("QRL? tone is above the transmit sample rate's Nyquist limit")
    envelope = keying_envelope(qrl_message(callsign), fs, wpm)
    t = np.arange(len(envelope)) / fs
    count = len(offsets)
    # Newman phases keep the crest factor of the equal-amplitude comb low.
    carrier = sum(
        np.cos(2 * np.pi * (tone_hz + offset) * t + np.pi * k * k / count)
        for k, offset in enumerate(offsets)
    )
    wave = envelope * carrier
    return (wave * (peak / np.max(np.abs(wave)))).astype(np.float32)
