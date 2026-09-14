"""AC16 A/V spectral isolation and actual modem round trips without a radio."""
import numpy as np
import pytest
from scipy import signal

from aetv.analog_av import AC16_AV, AC16CompositeSeparator, mix_composite_chunk
from aetv.modem import StreamingDemodulator, modulate_continuous_chunks
from aetv.sdr_dsp import IQToModem, ModemToIQ
from aetv.settings import StationSettings
from aetv.station import Station, TxEngine


def composite_fixture(monkeypatch, count=3):
    def no_microphone(*args, **kwargs):
        raise RuntimeError('No microphone in software validation')
    monkeypatch.setattr('aetv.station.open_input_stream', no_microphone)
    sent = np.random.default_rng(2033).normal(size=(count, 19200)).astype(np.float32)
    t = np.arange(count * 8000) / 8000
    voice = .2 * np.sin(2 * np.pi * 1000 * t) + .1 * np.sin(2 * np.pi * 3100 * t)
    settings = StationSettings(mode='AC16', waveform_mode='analog_av', av_microphone_mix=0,
                               av_video_power=.7)
    chunks = list(TxEngine(Station(settings))._composite_chunks(
        modulate_continuous_chunks(sent, 'AC16'), voice, count))
    return sent, chunks


def test_ac16_audio_filter_and_video_image_rejection():
    t = np.arange(96000) / 48000
    source = sum(np.sin(2*np.pi*f*t) for f in (1000, 3100, 4500, 13500, 19500))
    separator = AC16CompositeSeparator()
    parts = [separator.process(source[p:p+3791]) for p in range(0, len(t), 3791)]
    voice, video = [np.concatenate([pair[i] for pair in parts])[48000:] for i in range(2)]
    def amplitude(x, freq):
        return abs(np.fft.rfft(x)[freq])
    assert amplitude(voice, 3100) / amplitude(voice, 4500) > 1e4
    assert amplitude(video, 500) > 20000
    assert amplitude(video, 9500) > 20000
    assert amplitude(video, 15500) > 20000
    assert amplitude(video, 11500) / amplitude(video, 15500) < 1e-4  # conjugate mixer image
    assert amplitude(video, 900) / amplitude(video, 500) < 1e-4  # translated voice


def test_ac16_composite_spectral_mask(monkeypatch):
    _, chunks = composite_fixture(monkeypatch)
    waveform = np.concatenate(chunks)
    f,p = signal.welch(waveform, 48000, nperseg=48000)
    total = p.sum()
    assert p[f > 20000].sum() / total < 1e-5
    assert p[(f > 3400) & (f < 4100)].sum() / total < 1e-5
    assert np.max(abs(waveform)) <= .950001
    assert len(waveform) == round(4.65 * 48000)


@pytest.mark.parametrize('through_iq', [False, True])
def test_ac16_composite_recovers_all_modem_payloads(monkeypatch, through_iq):
    sent, chunks = composite_fixture(monkeypatch)
    separator = AC16CompositeSeparator()
    tx, rx = ModemToIQ(960000, center_hz=10000), IQToModem(signal_offset_hz=100000, center_hz=10000)
    demod = StreamingDemodulator('A', mode_name='AC16', continuous=True, boundary_tracking=True)
    decoded, voice = [], []
    for chunk in chunks + [np.zeros(4800)]:
        for p in range(0, len(chunk), 4800):
            block = chunk[p:p+4800] * .2
            if through_iq:
                block = rx.feed(tx.feed(block))
            audio, native = separator.process(block)
            voice.append(audio)
            for result in demod.feed(native):
                decoded.extend(result.gops_latents)
    assert np.asarray(decoded).shape == sent.shape
    for a,b in zip(sent, decoded):
        assert np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)) > .9
    # Last voice interval must survive after the final video GOP.
    tail = np.concatenate(voice)[round(3.7*48000):round(4.4*48000)]
    assert np.sqrt(np.mean(tail**2)) > .01


def test_ac16_av_settings_accept_sdr_and_reject_narrow_native_sources():
    for tx in ('pluto', 'hackrf', 'audio'):
        for rx in ('pluto', 'rtlsdr', 'hackrf', 'soundcard'):
            settings = StationSettings(mode='AC16', waveform_mode='analog_av', tx_backend=tx, rx_source=rx)
            assert settings.validate() == []
    settings.rx_source = 'flex'
    settings.flex_host = 'example'
    assert any('20 kHz' in p for p in settings.validate())


@pytest.mark.parametrize('kind,power,cfo', [('silent', .7, -20000), ('tones', .5, -5050),
                                          ('noise', .2, 5075), ('noise', .7, 20000)])
def test_av_auto_frequency_correction_ignores_audio_level(kind, power, cfo):
    from aetv.sdr_dsp import estimate_signal_offset
    sent = np.random.default_rng(2033).normal(size=(3, 19200)).astype(np.float32)
    video = list(modulate_continuous_chunks(sent, 'AC16'))[1]
    rng = np.random.default_rng(33)
    t = np.arange(8000)/8000
    voice = (np.zeros(8000) if kind == 'silent' else np.sin(2*np.pi*1000*t)
             if kind == 'tones' else rng.normal(size=8000))
    wave = mix_composite_chunk(video, voice, profile=AC16_AV, video_power=power)
    analytic = signal.resample_poly(signal.hilbert(wave), 20, 1)
    iq = analytic * np.exp(2j*np.pi*(-110000+cfo)*np.arange(len(analytic))/960000)
    iq += .002 * (rng.normal(size=len(iq)) + 1j*rng.normal(size=len(iq)))
    result = estimate_signal_offset(iq, composite=True)
    assert abs(result['offset_hz'] + 100000 - cfo) <= 50
    with pytest.raises(ValueError):
        estimate_signal_offset(rng.normal(size=960000) + 1j*rng.normal(size=960000), composite=True)


def test_delayed_audio_tracks_received_gops_instead_of_decode_order():
    from types import SimpleNamespace
    from aetv.analog_av import AC16ProgramAudio
    program = AC16ProgramAudio()
    # Second video GOP was lost. Its audio must not move onto the third GOP.
    program.add(SimpleNamespace(stream_start_sample=24000), 'first')
    program.add(SimpleNamespace(stream_start_sample=120000), 'third')
    program.voice.write(np.r_[np.zeros(12000), np.ones(8000), np.full(8000, 2)])
    ready = list(program.ready())
    assert len(ready) == 1 and ready[0][0] == 'first'
    assert np.all(ready[0][1] == 1)
    program.voice.write(np.full(8000, 3))
    ready = list(program.ready())
    assert len(ready) == 1 and ready[0][0] == 'third'
    assert np.all(ready[0][1] == 3)


@pytest.mark.parametrize('silent', [False, True])
def test_live_receive_engine_pairs_and_retains_the_final_audio_gop(monkeypatch, silent):
    import threading
    from types import SimpleNamespace
    from aetv.config import AETV_MODES
    from aetv.station import RxEngine

    stream = SimpleNamespace(stop=lambda: None, close=lambda: None)
    playback = []
    monkeypatch.setattr('aetv.station.open_input_stream', lambda *a, **k: (stream, 48000))
    monkeypatch.setattr('aetv.station.AudioPlaybackStream', lambda *a, **k: SimpleNamespace(
        write=lambda samples: playback.append(samples.copy()), close=lambda: None))
    sent = np.random.default_rng(2033).normal(size=(3, 19200)).astype(np.float32)
    settings = StationSettings(mode='AC16', waveform_mode='analog_av', av_microphone_mix=0,
                               debug_capture=False, autosave=False, decode_every_s=.05)
    station = Station(settings)
    station.codec = SimpleNamespace(mode=AETV_MODES['AC16'], device='test',
        decode_gop=lambda z,w: np.zeros((10,144,256,3), np.uint8))
    t = np.arange(8000)/8000
    voice = np.zeros(24000) if silent else np.concatenate([np.sin(2*np.pi*f*t) for f in (440,1100,3100)])
    chunks = list(TxEngine(station)._composite_chunks(modulate_continuous_chunks(sent, 'AC16'),
                                                   voice, 3, capture_microphone=False))
    done = threading.Event()
    errors = []
    engine = RxEngine(station, on_error=errors.append,
                           on_video=lambda v,s: done.set() if s.gops == 3 else None)
    try:
        engine.start()
        engine.ring.write(np.concatenate(chunks))
        assert done.wait(10), errors
        assert not errors
        assert len(engine.last_video) == 30
        assert len(engine.last_audio) == 24000
        assert len(playback) == 3
        if not silent:
            for i,f in enumerate((440,1100,3100)):
                assert np.argmax(abs(np.fft.rfft(engine.last_audio[i*8000:(i+1)*8000]))) == f
    finally:
        engine.stop()
