import asyncio

import numpy as np

from aetv.kiwi import (
    IQ_GPS_HEADER,
    IqToPassband,
    _decode_snd_iq,
    normalize_kiwi_host,
    parse_directory,
)
from aetv.kiwi import KiwiCapture


def test_receiverbook_directory_is_parsed_without_global_probing():
    blob = '''
    <script>var receivers = [{
      "label": "British Columbia",
      "location": {"coordinates": [-120.38, 50.70], "type": "Point"},
      "receivers": [{
        "label": "VE7 test Kiwi", "version": "1.902.0",
        "url": "http://ve7.example:8073/", "type": "KiwiSDR"
      }]
    }];</script>
    '''
    radios = parse_directory(blob)
    assert len(radios) == 1
    assert radios[0].host == "ve7.example:8073"
    assert radios[0].lat == 50.70
    assert radios[0].lon == -120.38


def test_canonical_kiwi_directory_metadata_is_parsed_without_probing():
    blob = '''
    var kiwisdr_com = [{
      "status": "active", "offline": "no", "name": "VE7FSR KiwiSDR",
      "users": "1", "users_max": "4", "ext_api": "3",
      "gps": "(49.123, -122.456)", "loc": "South Surrey, BC",
      "url": "http://ve7fsr.dyndns-home.com:8073"
    },];
    '''
    radios = parse_directory(blob)
    assert len(radios) == 1
    assert radios[0].host == "ve7fsr.dyndns-home.com:8073"
    assert radios[0].ext_api == 3
    assert radios[0].free == 3
    assert radios[0].usable


def test_pasted_kiwi_urls_are_normalized_for_websocket_use():
    assert normalize_kiwi_host("http://207.102.144.154:8073/") == "207.102.144.154:8073"
    assert normalize_kiwi_host("https://ve7fsr.dyndns-home.com/") == "ve7fsr.dyndns-home.com:8073"
    assert normalize_kiwi_host("ve7fsr.dyndns-home.com:8073") == "ve7fsr.dyndns-home.com:8073"


def test_kiwi_iq_network_samples_are_big_endian():
    interleaved = np.array([16384, -8192, -32768, 32767], dtype=np.int16)
    iq = _decode_snd_iq(interleaved.astype(">i2").tobytes())
    assert np.allclose(iq.real, [0.5, -1.0])
    assert np.allclose(iq.imag, [-0.25, 32767 / 32768])
    assert IQ_GPS_HEADER == 10


def test_fractional_kiwi_clock_is_streamed_without_second_long_batches():
    rate = 11998.881265
    converter = IqToPassband(rate, 24000, -5000)
    outputs = []
    phase = 0.0
    for _ in range(48):
        n = np.arange(512)
        # A 1 kHz USB tone appears at -4 kHz in IQ centred 5 kHz up.
        iq = np.exp(1j * (phase - 2 * np.pi * 4000 * n / rate))
        phase = float((phase - 2 * np.pi * 4000 * 512 / rate) % (2 * np.pi))
        outputs.append(converter(iq))
    nonempty = [chunk for chunk in outputs if len(chunk)]
    assert nonempty
    assert len(nonempty[0]) < 1200  # no former 24k-sample startup batch
    audio = np.concatenate(nonempty)
    assert len(audio) > 47000
    spectrum = np.abs(np.fft.rfft(audio[-24000:] * np.hanning(24000)))
    peak_hz = int(np.argmax(spectrum))
    assert abs(peak_hz - 1000) <= 1


def test_kiwi_tune_finishes_with_keepalive_without_deprecated_override():
    class FakeSocket:
        def __init__(self):
            self.sent = []

        async def send(self, message):
            self.sent.append(message)

    socket = FakeSocket()
    capture = KiwiCapture("kiwi.example:8073", 7.2, 5000, 24000, ring=None)
    asyncio.run(capture._tune(socket, 11998.881265))
    assert not any("OVERRIDE inactivity_timeout" in message for message in socket.sent)
    assert socket.sent[-1] == "SET keepalive"
