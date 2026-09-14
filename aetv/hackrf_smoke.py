"""No-hardware AC16 signed-IQ transport check used by frozen packages."""

import time

import numpy as np

from .hackrf import SAMPLE_RATE, decode_iq, encode_iq
from .modem import StreamingDemodulator, modulate_continuous_chunks
from .sdr_dsp import IQDecimator, IQToModem, ModemToIQ


def transport_smoke():
    started = time.perf_counter()
    sent = np.random.default_rng(439).standard_normal((3, 19200)).astype(np.float32)
    tx = ModemToIQ(SAMPLE_RATE)
    decimator = IQDecimator(SAMPLE_RATE // 960000)
    rx = IQToModem(signal_offset_hz=-100000)
    demod = StreamingDemodulator("A", continuous=True, mode_name="AC16", boundary_tracking=True)
    output = []
    count = 0
    for audio in list(modulate_continuous_chunks(sent, mode_name="AC16")) + [np.zeros(9600)]:
        for pos in range(0, len(audio), 4800):
            iq = decode_iq(encode_iq(tx.feed(audio[pos:pos + 4800] * .2)))
            # TX LO is RF-100k, RX LO RF+100k: translate by their difference.
            times = count + np.arange(len(iq))
            iq *= np.exp(-2j * np.pi * np.remainder(times * (200000 / SAMPLE_RATE), 1))
            count += len(iq)
            iq = decode_iq(encode_iq(iq))
            # USB transfers need not divide the decimation factor.
            for begin in range(0, len(iq), 131072):
                converted = rx.feed(decimator.feed(iq[begin:begin + 131072]))
                for result in demod.feed(converted):
                    output.extend(result.gops_latents)
    if np.asarray(output).shape != sent.shape:
        raise RuntimeError(f"HackRF signed-IQ check decoded {len(output)}/3 GOPs")
    cosines = [float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
               for a, b in zip(sent, output)]
    if min(cosines) < .9:
        raise RuntimeError(f"HackRF signed-IQ payload mismatch: {cosines}")
    return {"passed": True, "radio_opened": False, "sample_rate": SAMPLE_RATE,
            "sample_format": "interleaved signed int8 I,Q", "gops": len(output),
            "latent_cosines": cosines, "elapsed_s": time.perf_counter() - started,
            "signal_seconds": count / SAMPLE_RATE}
