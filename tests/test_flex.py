import struct

import numpy as np

from aetv.flex import (
    FlexVitaSession,
    discover_path_mtu,
    parse_discovery_packet,
    vita_tx_samples_for_mtu,
    with_probed_path_mtu,
)


def test_flex_discovery_parses_vita_payload():
    payload = (
        b"model=FLEX-6600 serial=1234 version=3.8.23 "
        b"nickname=Shack callsign=VE7ABC ip=192.0.2.10 port=4992\x00"
    )
    packet = bytes(28) + payload
    radio = parse_discovery_packet(packet, "192.0.2.11")
    assert radio is not None
    assert radio.ip == "192.0.2.10"
    assert radio.nickname == "Shack"
    assert radio.callsign == "VE7ABC"


def test_flex_vita_int16_audio_decode():
    samples = np.array([-32768, -1, 0, 32767], dtype=np.int16)
    payload = samples.astype(">i2").tobytes()
    words = 7 + len(payload) // 4
    header = struct.pack(
        ">7I", (1 << 28) | (1 << 27) | words, 0x40000001, 0x001C2D,
        (0x534C << 16) | FlexVitaSession.AUDIO_INT16, 0, 0, 0,
    )
    decoded = FlexVitaSession._decode_audio_packet(header + payload, 0x40000001)
    assert decoded is not None
    assert np.allclose(decoded, samples.astype(np.float32) / 32768.0)


def test_flex_vita_float_stereo_is_converted_from_48k_to_24k():
    left = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    right = -left
    stereo = np.column_stack([left, right]).reshape(-1)
    payload = stereo.astype(">f4").tobytes()
    words = 7 + len(payload) // 4
    header = struct.pack(
        ">7I", (1 << 28) | (1 << 27) | words, 0x40000002, 0x001C2D,
        (0x534C << 16) | FlexVitaSession.AUDIO_FLOAT32, 0, 0, 0,
    )
    decoded = FlexVitaSession._decode_audio_packet(header + payload, 0x40000002)
    assert decoded is not None
    assert np.allclose(decoded, [0.1, 0.3])


def test_flex_session_uses_documented_filter_command(monkeypatch):
    commands = []

    class FakeControl:
        def __init__(self, host, port):
            self.host = host

        def command(self, body):
            commands.append(body)
            if body == "client gui":
                return ["R10|0|client-id|"]
            if body == "sub slice all":
                return ["S1|slice 0 in_use=1 tx=1 mode=DIGU"]
            return ["R10|0||"]

        def _receive(self, seconds):
            return []

        def close(self):
            pass

    class FakeUdp:
        def setsockopt(self, *args):
            pass

        def bind(self, address):
            pass

        def settimeout(self, value):
            pass

        def getsockname(self):
            return ("0.0.0.0", 55000)

        def sendto(self, data, address):
            pass

        def close(self):
            pass

    monkeypatch.setattr("aetv.flex.FlexClient", FakeControl)
    monkeypatch.setattr("aetv.flex.socket.socket", lambda *args: FakeUdp())
    session = FlexVitaSession("192.0.2.1", filter_low=800, filter_high=9200)
    session.close()
    assert "filt 0 800 9200" in commands
    assert "client set enforce_network_mtu=1 network_mtu=1200" in commands
    assert not any("filter_lo=" in command for command in commands)
    assert not any(command.startswith("client program") for command in commands)


def test_flex_tx_packet_uses_flexlib_reduced_bandwidth_wire_format():
    session = object.__new__(FlexVitaSession)
    session.tx_stream_id = 0x84000001
    session._packet_count = 9
    values = np.array([-1.0, -0.25, 0.0, 0.25, 0.5, 1.0], dtype=np.float32)
    packet = session._encode_tx_packet(values)
    words = struct.unpack_from(">7I", packet)
    assert words[1] == 0x84000001
    assert words[2] == FlexVitaSession.FLEX_OUI
    assert words[3] == (0x534C << 16) | FlexVitaSession.AUDIO_INT16
    assert words[0] & 0xFFFF == 7 + (len(values) * 2) // 4
    assert (words[0] >> 16) & 0xF == 9
    expected = (values * 32767).astype(np.int16)
    assert np.array_equal(np.frombuffer(packet[28:], dtype=">i2"), expected)


def test_prepare_tx_creates_and_claims_stream():
    commands = []

    class Control:
        def command(self, body):
            commands.append(body)
            return ["R10|0|84000001|"]

    session = object.__new__(FlexVitaSession)
    session.control = Control()
    session.tx_stream_id = None
    session._tx_claimed = False
    session.path_mtu = 1210
    session.tx_packet_samples = 128
    session._radio_mtu = 1210
    session.radio_mtu_enforced = True
    session.prepare_tx()
    session.prepare_tx()
    assert session.tx_stream_id == 0x84000001
    assert commands == [
        "stream create type=dax_tx",
        "stream set 0x84000001 tx=1",
    ]


def test_path_mtu_uses_connected_socket_estimate():
    class Sock:
        def getsockopt(self, level, option):
            return 1210

    assert discover_path_mtu(Sock()) == 1210


def test_vita_tx_packet_size_respects_small_path_mtu():
    assert vita_tx_samples_for_mtu(1210) == 128
    samples = vita_tx_samples_for_mtu(180, preferred_samples=128)
    assert samples == 62
    wire_bytes = 20 + 8 + 28 + samples * 2
    assert wire_bytes <= 180


def test_flex_continuous_stream_pads_only_once(monkeypatch):
    packets = []

    class Udp:
        def sendto(self, packet, address):
            packets.append(packet)

    session = object.__new__(FlexVitaSession)
    session.udp = Udp()
    session.host = "192.0.2.1"
    session.tx_stream_id = 0x84000001
    session._packet_count = 0
    session.tx_packet_samples = 128
    session.path_mtu = 1210
    session.prepare_tx = lambda: None
    session._refresh_path_mtu = lambda: None
    monkeypatch.setattr("aetv.flex.time.sleep", lambda _seconds: None)
    chunks = [np.ones(100, np.float32), np.ones(100, np.float32)]
    assert session.send_audio_stream(chunks, 24000)
    payload_samples = sum((len(packet) - 28) // 2 for packet in packets)
    assert len(packets) == 2
    assert payload_samples == 256
    decoded = np.concatenate(
        [np.frombuffer(packet[28:], dtype=">i2") for packet in packets]
    )
    assert np.all(decoded[:200] != 0)
    assert np.all(decoded[200:] == 0)


def test_discovered_radio_can_be_annotated_with_path_mtu(monkeypatch):
    monkeypatch.setattr("aetv.flex.probe_radio_path_mtu", lambda host, port, timeout: 1210)
    radio = parse_discovery_packet(b"model=FLEX-6600 ip=192.0.2.10")
    measured = with_probed_path_mtu(radio)
    assert measured.path_mtu == 1210
    assert "MTU 1210" in measured.label
