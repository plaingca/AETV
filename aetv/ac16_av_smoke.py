"""AC16 A/V software validation, including the production compositor and RF DSP."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy import signal

from .analog_av import AC16CompositeSeparator, AC16ProgramAudio
from .audio_io import StreamResampler
from .hackrf import decode_iq, encode_iq
from .modem import StreamingDemodulator, modulate_continuous_chunks
from .sdr_dsp import IQDecimator, IQToModem, ModemToIQ
from .settings import StationSettings
from .station import Station, TxEngine


def av_smoke(codec=None, *, output: Path | None = None, source: Path | None = None,
             sample_rate: int = 9600000) -> dict:
    started = time.perf_counter()
    gops = 3
    rng = np.random.default_rng(2033)
    frames = None
    if codec is None:
        sent = rng.normal(size=(gops, 19200)).astype(np.float32)
    else:
        if codec.mode.name != 'AC16':
            raise ValueError('A/V validation requires the AC16 codec')
        if source is not None:
            frames = np.load(source, mmap_mode='r')[:30].copy()
        else:
            # Deterministic moving RGB gradients test the real neural path.
            y, x = np.mgrid[:144, :256]
            frames = np.stack([np.stack(((x + i*3) % 256, (y*2 + i*2) % 256,
                                        (x//2 + y + i*4) % 256), axis=-1)
                               for i in range(30)]).astype(np.uint8)
        sent = np.stack([codec.encode_gop(frames[p:p+10]) for p in range(0, 30, 10)])
    tones = (440, 1100, 3100)
    t = np.arange(8000) / 8000
    envelope = np.sin(np.pi * np.clip((t - .15) / .7, 0, 1))**2
    source_voice = np.concatenate([.3 * envelope * np.sin(2*np.pi*f*t) for f in tones])
    settings = StationSettings(mode='AC16', waveform_mode='analog_av', av_microphone_mix=0,
                               av_video_power=.7)
    engine = TxEngine(Station(settings))
    chunks = list(engine._composite_chunks(modulate_continuous_chunks(sent, 'AC16'),
                                          source_voice, gops, capture_microphone=False))
    composite = np.concatenate(chunks)
    f, p = signal.welch(composite, fs=48000, nperseg=48000)
    outband = float(p[f > 20000].sum() / p.sum())
    guard = float(p[(f > 3400) & (f < 4100)].sum() / p.sum())
    tx = ModemToIQ(sample_rate, center_hz=10000)
    decimator = IQDecimator(sample_rate // 960000) if sample_rate != 960000 else None
    rx = IQToModem(signal_offset_hz=-100000, center_hz=10000)
    separator = AC16CompositeSeparator()
    voice_resampler = StreamResampler(1, 6)
    demod = StreamingDemodulator('A', continuous=True, mode_name='AC16', boundary_tracking=True)
    program = AC16ProgramAudio()
    received, audio, weights = [], [], []
    count = 0
    for chunk in chunks + [np.zeros(4800)]:
        for pos in range(0, len(chunk), 4800):
            iq = decode_iq(encode_iq(tx.feed(chunk[pos:pos+4800] * .2)))
            positions = count + np.arange(len(iq))
            iq *= np.exp(-2j*np.pi*np.remainder(positions * (200000 / sample_rate), 1))
            count += len(iq)
            iq = decode_iq(encode_iq(iq))
            for start in range(0, len(iq), 131072):
                block = iq[start:start+131072]
                if decimator is not None:
                    block = decimator.feed(block)
                voice, video = separator.process(rx.feed(block))
                program.voice.write(voice_resampler(voice))
                for result in demod.feed(video):
                    program.add(result, (result.gops_latents[0], result.gops_weights[0]))
                for (latent, confidence), voice_gop in program.ready():
                    received.append(latent)
                    weights.append(confidence)
                    audio.append(voice_gop)
    if np.asarray(received).shape != sent.shape or len(audio) != gops:
        raise RuntimeError(f'AC16 A/V recovered {len(received)}/{gops} paired GOPs')
    cosines = [float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))) for a,b in zip(sent, received)]
    peaks = [float(np.fft.rfftfreq(8000, 1/8000)[np.argmax(abs(np.fft.rfft(v)))]) for v in audio]
    # Envelope timing is independent of carrier phase and sideband conversion.
    offsets = []
    for v in audio:
        observed = abs(signal.hilbert(v))
        lag = int(np.argmax(signal.correlate(observed, envelope)) - (len(envelope)-1))
        offsets.append(lag / 8000)
    if min(cosines) < .9 or outband > 1e-5 or guard > 1e-5:
        raise RuntimeError(f'AC16 A/V RF quality check failed: {cosines}, {outband}, {guard}')
    if max(abs(a-b) for a,b in zip(peaks, tones)) > 5 or max(map(abs, offsets)) > .02:
        raise RuntimeError(f'AC16 A/V audio alignment failed: tones={peaks}, delays={offsets}')
    result = dict(passed=True, radio_opened=False, mode='AC16 A/V', bandwidth_hz=20000,
                  audio_band_hz=[0, 3300], composite_rate=48000, hardware_rate=sample_rate,
                  sample_format='signed int8 I,Q; simulated TX/RX LO difference',
                  paired_gops=len(received), latent_cosines=cosines, audio_tones_hz=peaks,
                  audio_envelope_offsets_s=offsets, above_20khz_power_fraction=outband,
                  guard_power_fraction=guard)
    if codec is not None:
        from .source import write_mp4, ffmpeg_executable
        import subprocess
        decoded = np.concatenate([codec.decode_gop(z,w) for z,w in zip(received, weights)])
        direct = np.concatenate([codec.decode_gop(z) for z in sent])
        mse = float(np.mean((decoded.astype(float) - direct.astype(float))**2))
        result.update(device=str(codec.device), backend=codec.backend, checkpoint=str(codec.checkpoint_path),
                      decoded_frames=len(decoded), rgb_rmse_vs_direct=float(np.sqrt(mse)))
        if decoded.shape != frames.shape or np.sqrt(mse) > 30:
            raise RuntimeError('AC16 A/V neural reconstruction differs excessively from direct decode')
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            write_mp4(decoded, output, 10, audio=np.concatenate(audio), audio_rate=8000)
            video_bytes = subprocess.run([ffmpeg_executable(), '-v', 'error', '-i', str(output),
                                          '-map', '0:v:0', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                                         check=True, capture_output=True).stdout
            audio_bytes = subprocess.run([ffmpeg_executable(), '-v', 'error', '-i', str(output),
                                          '-map', '0:a:0', '-f', 'f32le', '-ar', '8000', '-ac', '1', '-'],
                                         check=True, capture_output=True).stdout
            saved_frames = len(video_bytes) // (256 * 144 * 3)
            track = np.frombuffer(audio_bytes, dtype='<f4')
            if saved_frames != 30 or not 24000 <= len(track) <= 25024 or np.sqrt(np.mean(track**2)) < .005:
                raise RuntimeError('Saved AC16 A/V video or audio did not decode')
            result.update(saved_video=str(output), saved_frames=saved_frames, saved_audio_samples=len(track))
    result['elapsed_s'] = time.perf_counter() - started
    return result
