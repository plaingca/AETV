"""Integration tests using real Hamlib's model 1; never connect to real radios.

Install libhamlib-utils and libhamlib4t64 (Debian), then run this file.
Tests skip explicitly when the native runtime is absent. All daemons are owned
by this test, bind only to loopback and are terminated in fixture cleanup.
"""
import shutil
import socket
import subprocess
import time

import numpy as np
import pytest

from aetv.cat import CatError, HamlibDirect, RigctldClient, _find_hamlib_library, list_hamlib_models


@pytest.fixture
def native_dummy():
    try:
        _find_hamlib_library()
    except CatError as error:
        pytest.skip(str(error))
    client = HamlibDirect(1, "")
    try:
        yield client
    finally:
        client.close()


@pytest.fixture
def dummy_daemon():
    executable = shutil.which("rigctld")
    if not executable:
        pytest.skip("actual Hamlib rigctld is not installed")
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    process = subprocess.Popen(
        [executable, "-m", "1", "-P", "RIG", "-T", "127.0.0.1", "-t", str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 5
        while True:
            if process.poll() is not None:
                pytest.fail(f"dummy rigctld exited: {process.stderr.read().decode()}")
            try:
                probe = RigctldClient("127.0.0.1", port, timeout=0.3)
                probe.close()
                break
            except CatError:
                if time.monotonic() > deadline:
                    pytest.fail("dummy rigctld did not listen within 5 seconds")
                time.sleep(0.02)
        yield port, process
    finally:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)
        process.stderr.close()


def test_actual_native_dummy_tuning_mode_and_ptt(native_dummy):
    client = native_dummy
    assert client.get_ptt() is False
    assert client.get_mode() == "FM"
    for frequency in (7_088_000, 14_230_000, 1_296_000_000):
        client.set_frequency_hz(frequency)
        assert client.get_frequency_hz() == frequency
    for _ in range(20):
        client.set_ptt(True)
        assert client.get_ptt() is True
        client.set_ptt(False)
        assert client.get_ptt() is False
    client.close()
    client.close()
    client.set_ptt(False)
    with pytest.raises(CatError, match="closed"):
        client.set_ptt(True)


def test_actual_model_names_keep_spaces(dummy_daemon):
    models = {model.model_id: model for model in list_hamlib_models()}
    assert models[1].manufacturer == "Hamlib"
    assert models[1].model == "Dummy"
    assert models[2].model == "NET rigctl"


def test_actual_tcp_dummy_tuning_mode_ptt_and_close(dummy_daemon):
    port, _ = dummy_daemon
    client = RigctldClient("127.0.0.1", port, timeout=0.5)
    observer = RigctldClient("127.0.0.1", port, timeout=0.5)
    try:
        client.set_frequency_hz(7_088_000)
        assert client.get_frequency_hz() == 7_088_000
        assert client.get_mode() == "FM"
        for _ in range(20):
            client.set_ptt(True)
            assert client.get_ptt() is True
            client.set_ptt(False)
            assert client.get_ptt() is False
        client.set_ptt(True)
        client.close()
        assert observer.get_ptt() is False
        client.close()
    finally:
        client.close()
        observer.close()


def test_actual_tcp_off_reconnects_after_lost_socket(dummy_daemon):
    port, _ = dummy_daemon
    client = RigctldClient("127.0.0.1", port, timeout=0.3)
    observer = RigctldClient("127.0.0.1", port, timeout=0.3)
    try:
        client.set_ptt(True)
        assert observer.get_ptt() is True
        client.sock.shutdown(socket.SHUT_RDWR)
        with pytest.raises(CatError):
            client.get_frequency_hz()
        assert client.sock is None
        with pytest.raises(CatError, match="closed"):
            client.set_ptt(True)
        client.set_ptt(False)
        assert observer.get_ptt() is False
    finally:
        client.close()
        observer.close()


def test_actual_tcp_dead_daemon_fails_with_bounded_off(dummy_daemon):
    port, process = dummy_daemon
    client = RigctldClient("127.0.0.1", port, timeout=0.2)
    process.kill()  # Deliberately simulate a daemon crash, not graceful shutdown.
    process.wait(timeout=3)
    start = time.monotonic()
    try:
        with pytest.raises(CatError):
            client.set_ptt(False)
        assert time.monotonic() - start < 1.5
    finally:
        client.close()


@pytest.mark.parametrize("outcome", ["complete", "cancel", "player_error"])
def test_actual_dummy_station_unkeys_on_every_exit(dummy_daemon, outcome):
    from aetv.settings import StationSettings
    from aetv.station import Station, TxEngine, TxPhase

    port, _ = dummy_daemon
    client = RigctldClient("127.0.0.1", port, timeout=0.5)
    observer = RigctldClient("127.0.0.1", port, timeout=0.5)
    station = Station(StationSettings(ptt_lead_s=0, ptt_tail_s=0))
    played = []

    def player(wave, fs, device=None, should_stop=None, on_progress=None):
        assert observer.get_ptt() is True
        played.append(True)
        if outcome == "cancel":
            engine.cancel()
            assert should_stop()
            return False
        if outcome == "player_error":
            raise RuntimeError("simulated audio device failure")
        return True

    engine = TxEngine(station, ptt=client, player=player)
    try:
        assert engine._keyed_send(np.zeros(2400, dtype=np.float32), 24000) == (outcome == "complete")
        assert played == [True]
        assert observer.get_ptt() is False
        assert engine.state.phase == {"complete": TxPhase.DONE, "cancel": TxPhase.CANCELLED, "player_error": TxPhase.FAILED}[outcome]
    finally:
        client.close()
        observer.close()


def test_actual_native_close_requests_off_before_cleanup(native_dummy):
    client = native_dummy
    client.set_ptt(True)
    assert client.get_ptt() is True
    original = client.set_ptt
    observed = []
    def record_off(on):
        original(on)
        observed.append(client.get_ptt())
    client.set_ptt = record_off
    client.close()
    assert observed == [False]
    assert client.rig is None


@pytest.mark.parametrize("outcome", ["cancel", "player_error"])
def test_actual_native_station_releases_before_cleanup(native_dummy, outcome):
    from aetv.settings import StationSettings
    from aetv.station import Station, TxEngine, TxPhase

    client = native_dummy
    original = client.set_ptt
    states = []
    def record(on):
        original(on)
        states.append(client.get_ptt())
    client.set_ptt = record
    def player(*args, should_stop=None, **kwargs):
        assert client.get_ptt() is True
        if outcome == "cancel":
            engine.cancel()
            assert should_stop()
            return False
        raise RuntimeError("simulated audio device failure")
    engine = TxEngine(Station(StationSettings(ptt_lead_s=0, ptt_tail_s=0)), ptt=client, player=player)
    assert engine._keyed_send(np.zeros(2400, dtype=np.float32), 24000) is False
    assert engine.state.phase == (TxPhase.CANCELLED if outcome == "cancel" else TxPhase.FAILED)
    assert states[0] is True
    assert states[1:] == [False, False]
    assert client.rig is None


@pytest.mark.parametrize("backend", ["native", "tcp"])
@pytest.mark.parametrize("outcome", ["complete", "cancel", "output_error"])
def test_station_stream_hf_audio_and_actual_dummy_ptt(
    native_dummy, dummy_daemon, monkeypatch, backend, outcome
):
    """Single production station/audio/modem chain, with only endpoints virtual.

    Synthetic latents avoid model downloads. Real modulation, native-rate writes,
    HF impairment, capture callbacks, rate conversion and RX ring all execute.
    Hamlib controls only model 1. No audio device or radio can be selected here.
    """
    from aetv.analog_validation import unit_rms_latents
    from aetv.audio_validation import AudioPathScenario, virtual_audio_path
    from aetv.config import AETV_MODES
    from aetv.modem import StreamingDemodulator, modulate_continuous_chunks
    from aetv.settings import StationSettings
    from aetv.station import Station, TxEngine, TxPhase

    monkeypatch.setenv("AETV_AUDIO_WORKER_CHILD", "1")
    port, _ = dummy_daemon
    client = native_dummy if backend == "native" else RigctldClient("127.0.0.1", port, timeout=0.5)
    observer = None if backend == "native" else RigctldClient("127.0.0.1", port, timeout=0.5)
    mode = AETV_MODES["V8"]
    originals = [unit_rms_latents("V8", 714 + index) for index in range(2)]
    chunks = [chunk * (.6 / max(float(np.max(np.abs(chunk))), 1e-12))
              for chunk in modulate_continuous_chunks(originals, "V8", "N0CALL")]
    receiver = StreamingDemodulator(mode.band, mode_name="V8", continuous=True)
    recovered, ptt_states, errors, writes = [], [], [], []
    original_set_ptt = client.set_ptt

    def record_ptt(on):
        original_set_ptt(on)
        ptt_states.append(client.get_ptt())
    client.set_ptt = record_ptt
    engine = TxEngine(Station(StationSettings(mode="V8", ptt_lead_s=0, ptt_tail_s=0)),
                      ptt=client, on_error=errors.append)

    def receive(samples):
        for result in receiver.feed(samples):
            recovered.extend(result.gops_latents)

    def send(chunks, rate):
        return engine._keyed_send_stream(chunks, rate, len(originals))

    def output_write(samples, state):
        assert (observer or client).get_ptt() is True
        writes.append(len(samples))
        if state["output_samples"] / 48000 >= .35:
            if outcome == "cancel":
                engine.cancel()
            elif outcome == "output_error":
                raise RuntimeError("injected virtual endpoint write failure")

    try:
        _, health = virtual_audio_path(
            chunks, mode.geometry.fs,
            AudioPathScenario("station_hamlib_hf", rx_rate=44100, snr_db=12),
            seed=23, on_samples=receive, send_chunks=send, on_output_write=output_write,
        )
        assert writes
        assert ptt_states[0] is True
        assert ptt_states[1:] == [False, False]
        if observer:
            assert observer.get_ptt() is False
        assert health["input_closed"] and health["output_closed"]
        assert health["ring_overruns"] == 0
        assert health["max_write_ms"] <= 20
        assert health["completed"] is (outcome == "complete")
        assert engine.state.phase == {"complete": TxPhase.DONE, "cancel": TxPhase.CANCELLED,
                                      "output_error": TxPhase.FAILED}[outcome]
        if outcome == "complete":
            assert not errors
            assert len(recovered) == 2
            for actual, expected in zip(recovered, originals):
                assert np.isfinite(actual).all()
                assert np.corrcoef(actual, expected)[0, 1] > .9
        elif outcome == "cancel":
            assert health["aborted"]
            assert .35 <= health["output_seconds"] <= .37
            assert not errors
        else:
            assert any("injected virtual endpoint write failure" in error for error in errors)
            assert .35 <= health["output_seconds"] <= .37
    finally:
        client.close()
        if observer:
            observer.close()
