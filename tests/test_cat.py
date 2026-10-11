"""CAT safety contracts exercised against loopback TCP only, never radio hardware."""
import ctypes
import socketserver
import subprocess
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from aetv import cat


@contextmanager
def rigctld(handler):
    commands = []

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            for raw in self.rfile:
                command = raw.decode().strip()
                commands.append(command)
                if handler(self, command) is False:
                    return

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    with Server(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = cat.RigctldClient("127.0.0.1", server.server_address[1], timeout=0.15)
        try:
            yield client, commands
        finally:
            client.close()
            server.shutdown()
            thread.join(timeout=2)


def respond(connection, data):
    try:
        connection.wfile.write(data)
        connection.wfile.flush()
    except OSError:
        pass


def test_fragmented_mode_and_width_do_not_poison_ptt_ack():
    def handler(conn, command):
        if command == "m":
            respond(conn, b"PKT")
            time.sleep(0.005)
            respond(conn, b"USB\n")
            time.sleep(0.005)
            respond(conn, b"2400\n")
        else:
            respond(conn, b"RPRT -10\n" if command == "T 1" else b"RPRT 0\n")

    with rigctld(handler) as (client, commands):
        assert client.get_mode() == "PKTUSB"
        with pytest.raises(cat.CatError, match="-10"):
            client.set_ptt(True)
        client.set_ptt(False)
        assert commands == ["m", "T 1", "T 0"]


@pytest.mark.parametrize("reply", [b"RPRT -10\n", b"RPRT -20\n", b"RPRT 10\n", b"RPRT nonsense\n", b"0\n", b"\xff\n", b"RPRT 0 garbage\n"])
def test_ptt_requires_exact_success_status(reply):
    with rigctld(lambda conn, cmd: respond(conn, reply if cmd == "T 1" else b"RPRT 0\n")) as (client, _):
        with pytest.raises(cat.CatError):
            client.set_ptt(True)


@pytest.mark.parametrize("reply", [b"nan\n", b"inf\n", b"-1\n", b"0\n", b"garbage\n", b"\n", b"RPRT 0\n", b"1" * 9000 + b"\n"])
def test_bad_frequency_fails_closed(reply):
    with rigctld(lambda conn, cmd: respond(conn, reply if cmd == "f" else b"RPRT 0\n")) as (client, _):
        with pytest.raises(cat.CatError):
            client.get_frequency_hz()
        assert client.sock is None


def test_unsupported_query_does_not_disable_ptt():
    with rigctld(lambda conn, cmd: respond(conn, b"RPRT -11\n" if cmd == "m" else b"RPRT 0\n")) as (client, _):
        with pytest.raises(cat.CatError, match="-11"):
            client.get_mode()
        assert client.sock is not None
        client.set_ptt(False)


def test_timeout_never_replays_key_on_and_off_reconnects():
    state = {"ptt": False}
    connections = []

    def handler(conn, command):
        connections.append(conn.client_address)
        if command == "T 1":
            state["ptt"] = True  # simulated radio keys before its ACK is lost
            time.sleep(0.3)
        else:
            if command == "T 0":
                state["ptt"] = False
            respond(conn, b"RPRT 0\n")

    with rigctld(handler) as (client, commands):
        start = time.monotonic()
        with pytest.raises(cat.CatError):
            client.set_ptt(True)
        assert time.monotonic() - start < 0.8
        assert state["ptt"] is True  # lost ACK must not be mistaken for safe receive
        with pytest.raises(cat.CatError, match="closed"):
            client.set_ptt(True)
        client.set_ptt(False)
        assert commands == ["T 1", "T 0"]
        assert connections[0] != connections[1]  # OFF used a new TCP connection
        assert state["ptt"] is False
    assert state["ptt"] is False


def test_disconnect_requires_reopen_except_for_off():
    def handler(conn, command):
        if command == "f":
            respond(conn, b"7088")
            return False
        respond(conn, b"RPRT 0\n")

    with rigctld(handler) as (client, commands):
        with pytest.raises(cat.CatError, match="disconnected"):
            client.get_frequency_hz()
        client.close()
        client.close()
        client.set_ptt(False)
        with pytest.raises(cat.CatError, match="closed"):
            client.set_ptt(True)
        assert commands == ["f", "T 0"]


def test_concurrent_reads_are_serialized():
    def handler(conn, cmd):
        respond(conn, b"USB\n" if cmd == "m" else b"RPRT 0\n")
        if cmd == "m":
            time.sleep(0.015)
            respond(conn, b"2800\n")

    with rigctld(handler) as (client, _):
        results = []
        threads = [threading.Thread(target=lambda: results.append(client.get_mode())) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2)
        assert results == ["USB"] * 4


class Function:
    def __init__(self, callback=lambda *args: 0):
        self.callback = callback
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        return self.callback(*args)


@pytest.fixture
def hamlib(monkeypatch):
    library = SimpleNamespace(**{name: Function() for name in (
        "rig_cleanup", "rig_close", "rig_open", "rig_set_conf", "rig_set_ptt", "rig_set_freq", "rig_get_freq", "rig_get_mode"
    )})
    library.rig_init = Function(lambda model: 123)
    library.rig_token_lookup = Function(lambda rig, name: 2**33)
    library.rigerror = Function(lambda code: b"unsupported capability")
    library.rig_strrmode = Function(lambda mode: b"PKTUSB")
    monkeypatch.setattr(cat, "_find_hamlib_library", lambda: "fake-library")
    monkeypatch.setattr(cat.ctypes, "CDLL", lambda path: library)
    return library


def test_hamlib_native_abi_and_mode_labels(hamlib):
    def get_mode(rig, vfo, mode, width):
        assert isinstance(mode._obj, ctypes.c_uint64)
        assert isinstance(width._obj, ctypes.c_long)
        mode._obj.value = 1 << 40
        width._obj.value = 12000
        return 0

    hamlib.rig_get_mode.callback = get_mode
    client = cat.HamlibDirect(1, "")
    assert client.get_mode() == "PKTUSB"
    assert hamlib.rig_token_lookup.restype is ctypes.c_long
    assert hamlib.rig_set_conf.argtypes[1] is ctypes.c_long
    assert hamlib.rig_strrmode.calls == [(1 << 40,)]
    client.close()


@pytest.mark.parametrize("model,device,baud", [(1, "", 0), (3073, "COM7", 19200), (1035, "/dev/serial/by-id/radio", 115200), (2, "127.0.0.1:4532", 0)])
def test_hamlib_station_configurations(hamlib, model, device, baud):
    client = cat.HamlibDirect(model, device, baud)
    values = [args[2] for args in hamlib.rig_set_conf.calls]
    assert values == ([device.encode()] if device else []) + ([str(baud).encode()] if baud else []) + ([b"RIG"] if model == 1 else [])
    assert hamlib.rig_set_ptt.calls == []  # opening never keys PTT
    client.close()


def test_hamlib_unsupported_ptt_still_cleans_up_once(hamlib):
    client = cat.HamlibDirect(1, "")
    hamlib.rig_set_ptt.callback = lambda *args: -11
    with pytest.raises(cat.CatError, match="unsupported capability"):
        client.set_ptt(True)
    client.close()
    client.close()
    client.set_ptt(False)
    assert len(hamlib.rig_close.calls) == len(hamlib.rig_cleanup.calls) == 1
    with pytest.raises(cat.CatError, match="closed"):
        client.set_ptt(True)
    with pytest.raises(cat.CatError, match="closed"):
        client.get_mode()


def test_hamlib_failed_open_releases_handle(hamlib):
    hamlib.rig_open.callback = lambda rig: -6
    with pytest.raises(cat.CatError, match="open radio"):
        cat.HamlibDirect(3073, "COM7")
    assert hamlib.rig_cleanup.calls == [(123,)]
    assert hamlib.rig_set_ptt.calls == []


@pytest.mark.parametrize("frequency", [float("nan"), float("inf"), -1, 0])
def test_invalid_tuning_never_reaches_backend(hamlib, frequency):
    client = cat.HamlibDirect(1, "")
    with pytest.raises(cat.CatError, match="finite and positive"):
        client.set_frequency_hz(frequency)
    assert hamlib.rig_set_freq.calls == []
    client.close()


def test_model_discovery_timeout_is_nonfatal(monkeypatch):
    monkeypatch.setattr(cat, "_rigctl_path", lambda: "rigctl")
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("rigctl", 8)
    monkeypatch.setattr(cat.subprocess, "run", timeout)
    assert cat.list_hamlib_models() == []


@pytest.mark.parametrize("reply", [b"USB\nnot-a-width\n", b"USB\nRPRT -11\n", b"USB\n"])
def test_malformed_or_missing_mode_width_invalidates_connection(reply):
    with rigctld(lambda conn, cmd: respond(conn, reply if cmd == "m" else b"RPRT 0\n")) as (client, _):
        with pytest.raises(cat.CatError):
            client.get_mode()
        assert client.sock is None


def test_close_unkeys_an_active_connection_once():
    with rigctld(lambda conn, cmd: respond(conn, b"RPRT 0\n")) as (client, commands):
        client.set_ptt(True)
        client.close()
        client.close()
        assert commands == ["T 1", "T 0"]


def test_hamlib_close_waits_for_inflight_call_before_cleanup(hamlib):
    entered = threading.Event()
    release = threading.Event()
    def tune(*args):
        entered.set()
        assert release.wait(2)
        return 0
    hamlib.rig_set_freq.callback = tune
    client = cat.HamlibDirect(1, "")
    tuner = threading.Thread(target=lambda: client.set_frequency_hz(1_296_000_000))
    closer = threading.Thread(target=client.close)
    tuner.start()
    assert entered.wait(1)
    closer.start()
    assert hamlib.rig_cleanup.calls == []
    release.set()
    tuner.join(timeout=2)
    closer.join(timeout=2)
    assert not tuner.is_alive() and not closer.is_alive()
    assert hamlib.rig_cleanup.calls == [(123,)]


def test_hamlib_missing_configuration_token_cleans_up(hamlib):
    hamlib.rig_token_lookup.callback = lambda *args: 0
    with pytest.raises(cat.CatError, match="does not expose rig_pathname"):
        cat.HamlibDirect(3073, "COM7")
    assert hamlib.rig_cleanup.calls == [(123,)]
    assert hamlib.rig_open.calls == []


def test_open_ptt_failed_tuning_closes_client(hamlib):
    hamlib.rig_set_freq.callback = lambda *args: -11
    with pytest.raises(cat.CatError, match="set frequency"):
        cat.open_ptt(cat.CatConfig(backend="hamlib", hamlib_model=1, freq_mhz=7.088))
    assert hamlib.rig_set_ptt.calls == [(123, cat.HamlibDirect._VFO_CURR, 0)]
    assert hamlib.rig_cleanup.calls == [(123,)]


def test_hamlib_cleanup_survives_close_exception(hamlib):
    def fail(*args):
        raise RuntimeError("native close failed")
    client = cat.HamlibDirect(1, "")
    hamlib.rig_close.callback = fail
    with pytest.raises(RuntimeError, match="native close failed"):
        client.close()
    client.close()
    assert client.rig is None
    assert hamlib.rig_cleanup.calls == [(123,)]


@pytest.mark.parametrize("reply", [b"garbage\n", b"-1\n", b"4\n", b"RPRT 0\n"])
def test_invalid_ptt_readback_fails_closed(reply):
    with rigctld(lambda conn, cmd: respond(conn, reply if cmd == "t" else b"RPRT 0\n")) as (client, _):
        with pytest.raises(cat.CatError):
            client.get_ptt()
        assert client.sock is None


@pytest.mark.parametrize("state", [0, 1, 2, 3])
def test_ptt_readback_accepts_hamlib_states(state):
    with rigctld(lambda conn, cmd: respond(conn, f"{state}\n".encode() if cmd == "t" else b"RPRT 0\n")) as (client, _):
        assert client.get_ptt() is (state != 0)


def test_native_ptt_readback_uses_int_abi(hamlib):
    def get_ptt(rig, vfo, value):
        assert isinstance(value._obj, ctypes.c_int)
        value._obj.value = 3
        return 0
    hamlib.rig_get_ptt = Function(get_ptt)
    client = cat.HamlibDirect(1, "")
    assert client.get_ptt() is True
    client.close()
