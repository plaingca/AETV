"""Receive pane: decoded video, source picker, KiwiSDR list, start/stop."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import math
import threading
import time

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QDoubleSpinBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from aetv.audio_io import AudioUnavailable, list_audio_devices
from aetv.kiwi import KiwiReceiver, find_receivers, normalize_kiwi_host, probe_receiver
from aetv.propagation import (
    PropagationPredictor,
    antenna_gain_db,
    fetch_space_weather,
    frequency_search_radius_km,
    initial_bearing_deg,
    parse_planning_frequencies,
    record_ota_failure,
    record_ota_measurement,
)
from aetv.settings import normalize_callsign
from aetv.station import RxEngine, RxState
from aetv.gui.widgets import ElidingLabel, VideoView


class _KiwiListThread(QThread):
    finished_list = Signal(object, str, str)

    def __init__(self, settings, configured_host: str = "", parent=None):
        super().__init__(parent)
        self._lat = float(settings.kiwi_lat)
        self._lon = float(settings.kiwi_lon)
        self._max_km = float(settings.kiwi_max_km)
        self._frequency = float(settings.kiwi_dial_mhz)
        self._power = max(0.1, float(settings.flex_power))
        self._antenna_pattern = settings.prop_antenna_pattern
        self._antenna_azimuth = float(settings.prop_antenna_azimuth_deg)
        self._antenna_gain = float(settings.prop_antenna_gain_dbi)
        try:
            self._configured_host = normalize_kiwi_host(configured_host)
        except ValueError:
            self._configured_host = ""

    def run(self) -> None:
        try:
            receivers = find_receivers(self._lat, self._lon, max_km=self._max_km)
            if self._configured_host and not any(
                item.host == self._configured_host for item in receivers
            ):
                configured = probe_receiver(self._configured_host, timeout=4.0)
                if configured is not None:
                    receivers.insert(0, configured)
            weather = fetch_space_weather()
            predictor = PropagationPredictor()

            def estimate(receiver):
                bearing = initial_bearing_deg(
                    self._lat, self._lon, receiver.lat, receiver.lon
                )
                gain = antenna_gain_db(
                    self._antenna_pattern,
                    bearing,
                    self._antenna_azimuth,
                    self._antenna_gain,
                )
                return predictor.predict(
                    receiver,
                    self._lat,
                    self._lon,
                    self._frequency,
                    self._power,
                    weather=weather,
                    tx_gain_dbi=gain,
                )

            usable = [receiver for receiver in receivers if receiver.usable][:60]
            with ThreadPoolExecutor(max_workers=min(8, max(1, len(usable)))) as pool:
                pending = {pool.submit(estimate, receiver): receiver for receiver in usable}
                for future in as_completed(pending):
                    if self.isInterruptionRequested():
                        for job in pending:
                            job.cancel()
                        return
                    receiver = pending[future]
                    try:
                        result = future.result()
                    except Exception:
                        continue
                    receiver.predicted_snr_db = result.calibrated_snr_db
                    receiver.predicted_uncertainty_db = result.uncertainty_db
                    receiver.success_probability = result.success_probability
                    receiver.prediction_engine = result.engine
            receivers.sort(
                key=lambda item: (
                    not item.usable,
                    -(item.success_probability if item.success_probability is not None else -1.0),
                    item.km if item.km is not None else 1e9,
                )
            )
            source = f"SSN {weather.ssn:.0f}, Kp {weather.kp:.1f} · {weather.source}"
            self.finished_list.emit(receivers, "", source)
        except Exception as error:
            self.finished_list.emit([], str(error), "")


class _RfPairThread(QThread):
    """Find the strongest frequency/receiver pair without ever tuning or keying."""

    finished_pairs = Signal(object, str, str)

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self._lat = float(settings.kiwi_lat)
        self._lon = float(settings.kiwi_lon)
        self._base_radius = float(settings.kiwi_max_km)
        self._frequencies = parse_planning_frequencies(
            settings.prop_candidate_frequencies_mhz, settings.kiwi_dial_mhz
        )
        self._power = max(0.1, float(settings.flex_power))
        self._antenna_pattern = settings.prop_antenna_pattern
        self._antenna_azimuth = float(settings.prop_antenna_azimuth_deg)
        self._antenna_gain = float(settings.prop_antenna_gain_dbi)

    def run(self) -> None:
        try:
            radii = {
                frequency: frequency_search_radius_km(frequency, self._base_radius)
                for frequency in self._frequencies
            }
            receivers = find_receivers(
                self._lat, self._lon, max_km=max(radii.values())
            )
            usable = [receiver for receiver in receivers if receiver.usable]
            if not usable:
                self.finished_pairs.emit([], "no reachable KiwiSDRs found", "")
                return
            weather = fetch_space_weather()
            predictor = PropagationPredictor()

            def gain_for(receiver):
                bearing = initial_bearing_deg(
                    self._lat, self._lon, receiver.lat, receiver.lon
                )
                return antenna_gain_db(
                    self._antenna_pattern,
                    bearing,
                    self._antenna_azimuth,
                    self._antenna_gain,
                )

            # Cheap pre-ranking keeps native P.533 work bounded, while distance
            # sampling preserves longer skip candidates that a nearest-only list loses.
            shortlist = []
            for frequency in self._frequencies:
                in_range = [
                    receiver for receiver in usable
                    if (receiver.km or 0.0) <= radii[frequency]
                ]
                coarse = sorted(
                    in_range,
                    key=lambda receiver: PropagationPredictor._predict_fallback(
                        receiver.km or 50.0,
                        frequency,
                        self._power,
                        datetime.now(timezone.utc),
                        weather,
                        gain_for(receiver),
                    )["snr"],
                    reverse=True,
                )
                selected = coarse[:8]
                by_distance = sorted(in_range, key=lambda receiver: receiver.km or 0.0)
                if by_distance:
                    step = max(1, len(by_distance) // 8)
                    selected.extend(by_distance[::step][:8])
                seen = set()
                for receiver in selected:
                    key = (receiver.host, frequency)
                    if key not in seen:
                        seen.add(key)
                        shortlist.append((receiver, frequency))

            def estimate(receiver, frequency):
                return predictor.predict(
                    receiver,
                    self._lat,
                    self._lon,
                    frequency,
                    self._power,
                    weather=weather,
                    tx_gain_dbi=gain_for(receiver),
                )

            pairs = []
            with ThreadPoolExecutor(max_workers=min(8, max(1, len(shortlist)))) as pool:
                pending = {
                    pool.submit(estimate, receiver, frequency): (receiver, frequency)
                    for receiver, frequency in shortlist
                }
                for future in as_completed(pending):
                    if self.isInterruptionRequested():
                        for job in pending:
                            job.cancel()
                        return
                    receiver, frequency = pending[future]
                    try:
                        pairs.append((receiver, frequency, future.result()))
                    except Exception:
                        continue
            pairs.sort(
                key=lambda pair: (
                    -pair[2].success_probability,
                    -pair[2].calibrated_snr_db,
                    pair[2].distance_km,
                )
            )
            source = (
                f"{len(self._frequencies)} dials · adaptive radius to "
                f"{max(radii.values()):.0f} km · SSN {weather.ssn:.0f}, Kp {weather.kp:.1f}"
            )
            self.finished_pairs.emit(pairs, "", source)
        except Exception as error:
            self.finished_pairs.emit([], str(error), "")


class ReceivePanel(QWidget):
    pathPlannerRequested = Signal()
    statusChanged = Signal(str)
    listeningChanged = Signal(bool)
    logMessage = Signal(str)
    stopFinished = Signal()
    _stateArrived = Signal(object)
    _errorArrived = Signal(str)
    _videoArrived = Signal(object, object)
    _ringArrived = Signal(object)
    _stopArrived = Signal()

    def __init__(self, station, parent=None):
        super().__init__(parent)
        self.station = station
        self.engine = RxEngine(
            station,
            on_state=self._stateArrived.emit,
            on_error=self._errorArrived.emit,
            on_video=lambda video, state: self._videoArrived.emit(video, state),
            on_ring=self._ringArrived.emit,
        )
        self._waterfall = None
        self._kiwi_thread: _KiwiListThread | None = None
        self._rf_pair_thread: _RfPairThread | None = None
        self._receivers: list[KiwiReceiver] = []
        self._probe_receiver: KiwiReceiver | None = None
        self._probe_tx_active = False
        self._probe_tx_started = 0.0
        self._probe_tx_generation = 0
        self._probe_decode_seen = False
        self._probe_tx_receiver: KiwiReceiver | None = None
        self._recommended_receiver: KiwiReceiver | None = None
        self._prediction_key = None
        self._kiwi_force_auto = False
        self._start_after_kiwi_pick = False
        self._stop_thread: threading.Thread | None = None
        self._stateArrived.connect(self._apply_state, Qt.ConnectionType.QueuedConnection)
        self._errorArrived.connect(self._on_error, Qt.ConnectionType.QueuedConnection)
        self._videoArrived.connect(self._show_video, Qt.ConnectionType.QueuedConnection)
        self._ringArrived.connect(self._apply_ring, Qt.ConnectionType.QueuedConnection)
        self._stopArrived.connect(self._finish_stop, Qt.ConnectionType.QueuedConnection)
        self._build()

    def attach_waterfall(self, waterfall) -> None:
        self._waterfall = waterfall

    def control_strip(self) -> QWidget:
        return self._strip

    def picture_area(self) -> QWidget:
        return self.preview

    def listening(self) -> bool:
        return self.engine.listening

    def sync_from_config(self) -> None:
        settings = self.station.settings
        prediction_key = (
            settings.kiwi_lat,
            settings.kiwi_lon,
            settings.kiwi_max_km,
            settings.kiwi_dial_mhz,
            settings.flex_power,
            settings.prop_antenna_pattern,
            settings.prop_antenna_azimuth_deg,
            settings.prop_antenna_gain_dbi,
        )
        prediction_changed = prediction_key != self._prediction_key
        self._prediction_key = prediction_key
        if prediction_changed:
            self._receivers = []
            self._recommended_receiver = None
        self.source.setCurrentIndex(max(0, self.source.findData(settings.rx_source)))
        self._fill_inputs()
        self.kiwi_host.setText(settings.kiwi_host)
        self.kiwi_dial.setValue(settings.kiwi_dial_mhz)
        self.auto_kiwi.setChecked(settings.kiwi_auto_select)
        self._sync_source_visibility()
        if (
            prediction_changed
            and settings.rx_source == "kiwi"
            and settings.kiwi_auto_select
        ):
            QTimer.singleShot(0, self._refresh_kiwis)

    def start(self) -> bool:
        if self._rf_pair_thread is not None and self._rf_pair_thread.isRunning():
            self._start_after_kiwi_pick = True
            self.status.setText("waiting for the best RF path…")
            return False
        if self.source.currentData() == "kiwi" and self.auto_kiwi.isChecked():
            if self._kiwi_thread is not None and self._kiwi_thread.isRunning():
                self._start_after_kiwi_pick = True
                self.status.setText("waiting for the best Kiwi path…")
                return False
            if self._recommended_receiver is None:
                self._start_after_kiwi_pick = True
                self._refresh_kiwis(force_auto=True)
                return False
        try:
            self._apply_panel_settings()
        except ValueError as error:
            self.status.setText(str(error))
            self.statusChanged.emit(str(error))
            return False
        problems = self.station.settings.validate()
        if self.station.settings.rx_source == "kiwi" and not self.station.settings.kiwi_host:
            problems.append("pick a KiwiSDR or type a host:port")
        if problems:
            self.status.setText(problems[0])
            self.statusChanged.emit(problems[0])
            return False
        try:
            self.preview.clear()
            self.engine.start()
        except Exception as error:
            self.status.setText(str(error))
            self.statusChanged.emit(str(error))
            self.logMessage.emit(str(error))
            return False
        self._set_listening(True)
        return True

    def stop(self) -> None:
        if self._stop_thread is not None and self._stop_thread.is_alive():
            return
        self.status.setText("stopping…")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self._stop_thread = threading.Thread(
            target=self._stop_worker, daemon=True, name="aetv-rx-stop"
        )
        self._stop_thread.start()

    def _stop_worker(self) -> None:
        try:
            self.engine.stop()
        except Exception as error:
            self._errorArrived.emit(str(error))
        finally:
            self._stopArrived.emit()

    def _finish_stop(self) -> None:
        self.progress.setValue(0)
        self._set_listening(False)
        if self._waterfall is not None:
            self._waterfall.set_ring(None)
        self.stopFinished.emit()

    def stop_blocking(self) -> None:
        """Stop synchronously during application shutdown."""
        thread = self._stop_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=6.0)
        elif self.engine.listening:
            self.engine.stop()

    def save_current(self) -> None:
        try:
            path = self.engine.save_current()
        except Exception as error:
            self.status.setText(str(error))
            return
        if path is None:
            self.status.setText("nothing to save yet")
            return
        self.logMessage.emit(f"saved {path}")
        self.status.setText(f"saved {path.name}")

    def _build(self) -> None:
        self.preview = VideoView("Start receiving")
        self.status = ElidingLabel("Stopped")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.source = QComboBox()
        self.source.addItem("Soundcard", "soundcard")
        self.source.addItem("FlexRadio (network)", "flex")
        self.source.addItem("Public KiwiSDR", "kiwi")
        self.source.currentIndexChanged.connect(self._sync_source_visibility)
        self.input_device = QComboBox()
        self.kiwi_host = QLineEdit()
        self.kiwi_host.setMinimumWidth(190)
        self.kiwi_host.setPlaceholderText("Paste http://host:8073/ or host:port")
        self.kiwi_host.textEdited.connect(self._on_manual_kiwi_host_edited)
        self.kiwi_host.editingFinished.connect(self._normalize_kiwi_entry)
        self.kiwi_list = QComboBox()
        self.kiwi_list.setMinimumWidth(180)
        self.kiwi_list.activated.connect(self._apply_kiwi_choice)
        self.kiwi_dial = QDoubleSpinBox()
        self.kiwi_dial.setDecimals(6)
        self.kiwi_dial.setRange(0.1, 30.0)
        self.kiwi_dial.setSingleStep(0.001)
        self.kiwi_dial.setSuffix(" MHz")
        self.kiwi_dial.editingFinished.connect(self._on_kiwi_dial_changed)
        self.refresh_audio = QPushButton("Refresh")
        self.refresh_audio.clicked.connect(self._fill_inputs)
        self.find_button = QPushButton("Find Kiwis")
        self.find_button.setText("Find best RF path")
        self.find_button.setToolTip(
            "Compare your configured HF planning dials and reachable KiwiSDRs; "
            "this selects settings but never starts a transmission."
        )
        self.find_button.clicked.connect(self._find_best_rf_pair)
        self.auto_kiwi = QCheckBox(
            "Automatically replace the receiver with the predictor's best path"
        )
        self.auto_kiwi.setChecked(self.station.settings.kiwi_auto_select)
        self.auto_kiwi.toggled.connect(self._on_auto_kiwi_toggled)
        self.kiwi_recommendation = QLabel("Best receiver not calculated yet")
        self.kiwi_recommendation.setWordWrap(True)
        self.kiwi_recommendation.setStyleSheet(
            "QLabel { padding: 5px 8px; border: 1px solid #777; border-radius: 4px; }"
        )
        self.path_planner_button = QPushButton("Compare paths…")
        self.path_planner_button.clicked.connect(self.pathPlannerRequested.emit)
        self.start_button = QPushButton("Start receiving")
        self.stop_button = QPushButton("Stop")
        self.save_button = QPushButton("Save video…")
        self.stop_button.setEnabled(False)
        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)
        self.save_button.clicked.connect(self.save_current)

        self._strip = QWidget()
        strip = QVBoxLayout(self._strip)
        strip.setContentsMargins(0, 0, 0, 0)
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Source"))
        row1.addWidget(self.source)
        row1.addWidget(self.input_device, 1)
        row1.addWidget(self.refresh_audio)
        row2 = QHBoxLayout()
        self.kiwi_label = QLabel("Receiver address")
        row2.addWidget(self.kiwi_label)
        row2.addWidget(self.kiwi_host, 1)
        row3 = QHBoxLayout()
        self.kiwi_dial_label = QLabel("TX dial")
        row3.addWidget(self.kiwi_dial_label)
        row3.addWidget(self.kiwi_dial)
        row3.addWidget(self.kiwi_list, 1)
        row3.addWidget(self.find_button)
        path_row = QHBoxLayout()
        path_row.addWidget(self.kiwi_recommendation, 1)
        path_row.addWidget(self.path_planner_button)
        auto_row = QHBoxLayout()
        auto_row.addWidget(self.auto_kiwi)
        auto_row.addStretch(1)
        buttons = QHBoxLayout()
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        buttons.addWidget(self.save_button)
        buttons.addStretch(1)
        strip.addLayout(row1)
        strip.addLayout(row2)
        strip.addLayout(row3)
        strip.addLayout(path_row)
        strip.addLayout(auto_row)
        strip.addWidget(self.status)
        strip.addWidget(self.progress)
        strip.addLayout(buttons)

        layout = QVBoxLayout(self)
        layout.addWidget(self.preview, 1)
        layout.addWidget(self._strip, 0)
        self.sync_from_config()
        if self.source.currentData() == "kiwi" and self.auto_kiwi.isChecked():
            QTimer.singleShot(400, self._refresh_kiwis)

    def _sync_source_visibility(self) -> None:
        source = self.source.currentData()
        kiwi = source == "kiwi"
        soundcard = source == "soundcard"
        self.input_device.setVisible(soundcard)
        self.refresh_audio.setVisible(soundcard)
        self.kiwi_label.setVisible(kiwi)
        self.kiwi_host.setVisible(kiwi)
        self.kiwi_list.setVisible(kiwi)
        self.kiwi_dial_label.setVisible(kiwi)
        self.kiwi_dial.setVisible(kiwi)
        self.find_button.setVisible(kiwi)
        self.auto_kiwi.setVisible(kiwi)
        self.kiwi_recommendation.setVisible(kiwi)
        self.path_planner_button.setVisible(kiwi)
        if kiwi and self.auto_kiwi.isChecked() and not self._receivers:
            QTimer.singleShot(0, self._refresh_kiwis)

    def _fill_inputs(self) -> None:
        current = self.station.settings.audio_input
        self.input_device.clear()
        self.input_device.addItem("System default", "")
        try:
            for item in list_audio_devices("input"):
                self.input_device.addItem(item.label(), item.name)
        except AudioUnavailable:
            return
        index = self.input_device.findData(current)
        if index >= 0:
            self.input_device.setCurrentIndex(index)

    def _apply_panel_settings(self) -> None:
        settings = self.station.settings
        settings.rx_source = self.source.currentData()
        settings.audio_input = self.input_device.currentData() or ""
        settings.kiwi_host = normalize_kiwi_host(self.kiwi_host.text())
        self.kiwi_host.setText(settings.kiwi_host)
        settings.kiwi_dial_mhz = float(self.kiwi_dial.value())
        if settings.rx_source == "kiwi":
            settings.freq_mhz = settings.kiwi_dial_mhz
        settings.kiwi_auto_select = self.auto_kiwi.isChecked()

    def _on_auto_kiwi_toggled(self, enabled: bool) -> None:
        self.station.settings.kiwi_auto_select = bool(enabled)
        if enabled and self.source.currentData() == "kiwi" and not self.listening():
            self._refresh_kiwis(force_auto=True)

    def _on_kiwi_dial_changed(self) -> None:
        frequency = float(self.kiwi_dial.value())
        self.station.settings.kiwi_dial_mhz = frequency
        # This field is explicitly the AETV TX dial. Keep CAT/Flex and the
        # remote IQ receiver together when the operator edits it manually.
        self.station.settings.freq_mhz = frequency
        self._recommended_receiver = None
        if self.auto_kiwi.isChecked() and self.source.currentData() == "kiwi":
            self._refresh_kiwis(force_auto=True)

    def _find_best_rf_pair(self) -> None:
        if self.listening():
            self.status.setText("stop receiving before changing RF path")
            return
        if self._rf_pair_thread is not None and self._rf_pair_thread.isRunning():
            return
        if self._kiwi_thread is not None and self._kiwi_thread.isRunning():
            self.status.setText("finish the current receiver ranking, then try again")
            return
        settings = self.station.settings
        settings.kiwi_dial_mhz = float(self.kiwi_dial.value())
        self.find_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.kiwi_recommendation.setText(
            "Comparing frequency and receiver combinations…"
        )
        self.status.setText(
            "finding the best RF path with frequency-aware skip distances…"
        )
        self._rf_pair_thread = _RfPairThread(settings, self)
        self._rf_pair_thread.finished_pairs.connect(self._on_rf_pairs)
        self._rf_pair_thread.start()

    def _on_rf_pairs(self, pairs, error: str, source: str) -> None:
        self.find_button.setEnabled(not self.listening())
        if not self.listening():
            self.start_button.setEnabled(self.station.codec is not None)
        if error or not pairs:
            message = error or "no usable frequency/receiver combination found"
            self.status.setText(message)
            self.kiwi_recommendation.setText(f"RF path search failed: {message}")
            self.logMessage.emit(message)
            self._start_after_kiwi_pick = False
            return
        _best_receiver, best_frequency, best_estimate = pairs[0]
        selected_receivers = []
        seen = set()
        for receiver, frequency, estimate in pairs:
            if abs(frequency - best_frequency) > 1e-9 or receiver.host in seen:
                continue
            seen.add(receiver.host)
            receiver.predicted_snr_db = estimate.calibrated_snr_db
            receiver.predicted_uncertainty_db = estimate.uncertainty_db
            receiver.success_probability = estimate.success_probability
            receiver.prediction_engine = estimate.engine
            receiver.km = estimate.distance_km
            selected_receivers.append(receiver)
        self.kiwi_dial.setValue(best_frequency)
        self.station.settings.kiwi_dial_mhz = best_frequency
        # Keep transmitter CAT/Flex tuning and the remote receiver on one dial.
        self.station.settings.freq_mhz = best_frequency
        self._kiwi_force_auto = True
        self._on_kiwi_list(
            selected_receivers,
            "",
            f"best of {source}",
        )
        place = _best_receiver.loc or _best_receiver.name or _best_receiver.host
        self.status.setText(
            f"selected {best_frequency:.6f} MHz + {place}; propagation estimate only — listen before transmitting"
        )
        self.logMessage.emit(
            f"best RF path selected: {best_frequency:.6f} MHz to {_best_receiver.host}, "
            f"{best_estimate.calibrated_snr_db:.1f} dB, "
            f"{best_estimate.success_probability:.0f}% predicted success"
        )

    def _refresh_kiwis(self, force_auto: bool = False) -> None:
        if self._kiwi_thread is not None and self._kiwi_thread.isRunning():
            self._kiwi_force_auto = self._kiwi_force_auto or force_auto
            return
        settings = self.station.settings
        settings.kiwi_dial_mhz = float(self.kiwi_dial.value())
        self._kiwi_force_auto = bool(force_auto)
        self.find_button.setEnabled(False)
        if not self.listening():
            self.start_button.setEnabled(False)
        self.status.setText("ranking live Kiwi paths with P.533…")
        self.kiwi_recommendation.setText("Finding the best available receiver…")
        self._kiwi_thread = _KiwiListThread(
            settings,
            self.kiwi_host.text(),
            self,
        )
        self._kiwi_thread.finished_list.connect(self._on_kiwi_list)
        self._kiwi_thread.start()

    def refresh_propagation_calibration(self) -> None:
        """Discard cached path scores after an OTA calibration import."""
        self._recommended_receiver = None
        for receiver in self._receivers:
            receiver.predicted_snr_db = None
            receiver.predicted_uncertainty_db = None
            receiver.success_probability = None
            receiver.prediction_engine = ""
        if self.source.currentData() == "kiwi":
            self._refresh_kiwis(force_auto=False)

    def _on_kiwi_list(self, receivers, error: str, source: str = "") -> None:
        self.find_button.setEnabled(True)
        if not self.listening():
            self.start_button.setEnabled(self.station.codec is not None)
        if error:
            self.status.setText(error)
            self.kiwi_recommendation.setText(f"Automatic path selection failed: {error}")
            self.logMessage.emit(error)
            self._start_after_kiwi_pick = False
            return
        self._receivers = list(receivers)
        self.kiwi_list.clear()
        usable = [item for item in self._receivers if item.usable]
        try:
            current = normalize_kiwi_host(self.kiwi_host.text())
        except ValueError:
            current = self.kiwi_host.text().strip()
        shown = self._receivers[:100]
        if current and not any(item.host == current for item in shown):
            self.kiwi_list.addItem(f"configured  {current}", current)
        if not shown:
            if not current:
                self.kiwi_list.addItem("no reachable receivers", "")
                self.status.setText("no reachable KiwiSDRs in range")
            else:
                self.status.setText("configured receiver retained; no directory results")
            return
        for item in shown:
            if item.offline != "no":
                mark = "offline"
            elif item.ext_api <= 0:
                mark = "browser only"
            elif item.free <= 0:
                mark = f"full · API {item.ext_api}"
            else:
                mark = f"API {item.ext_api}"
            if item.success_probability is not None:
                mark = (
                    f"{item.success_probability:.0f}% · "
                    f"{item.predicted_snr_db:.1f} dB · {mark}"
                )
            if item.host == current:
                mark = f"current · {mark}"
            self.kiwi_list.addItem(f"{mark}  {item.label()}", item.host)
        self.status.setText(
            f"ranked {len(usable)} usable Kiwi paths · {source}"
        )
        best = next(
            (item for item in usable if item.success_probability is not None),
            usable[0] if usable else None,
        )
        self._recommended_receiver = best
        if best is not None:
            probability = (
                f"{best.success_probability:.0f}% chance of a usable decode"
                if best.success_probability is not None
                else "prediction unavailable"
            )
            snr = (
                f"{best.predicted_snr_db:.1f} ± {best.predicted_uncertainty_db:.1f} dB"
                if best.predicted_snr_db is not None
                else "unknown SNR"
            )
            place = best.loc or best.name or best.host
            condition = (
                "Good" if (best.success_probability or 0.0) >= 70 else
                "Marginal" if (best.success_probability or 0.0) >= 35 else "Poor"
            )
            self.kiwi_recommendation.setText(
                f"Best now: {place} at {self.station.settings.kiwi_dial_mhz:.6f} MHz\n"
                f"{condition} conditions · {probability} · "
                f"{snr} · {best.km or 0:.0f} km"
            )
            probability_value = best.success_probability or 0.0
            color = "#285b32" if probability_value >= 70 else "#725c1d" if probability_value >= 35 else "#742d2d"
            self.kiwi_recommendation.setStyleSheet(
                f"QLabel {{ padding: 5px 8px; border: 1px solid {color}; "
                f"border-radius: 4px; background: {color}; color: white; }}"
            )
        auto_pick = (self.auto_kiwi.isChecked() or self._kiwi_force_auto) and not self.listening()
        if best is not None and auto_pick:
            self.kiwi_host.setText(best.host)
            self.station.settings.kiwi_host = best.host
            self.set_probe_receiver(best)
            current = best.host
            self.logMessage.emit(
                f"automatically selected Kiwi {best.host}: "
                f"{best.predicted_snr_db:.1f} dB, {best.success_probability:.0f}% predicted success"
                if best.predicted_snr_db is not None
                else f"automatically selected Kiwi {best.host}"
            )
        if not current and usable:
            self.kiwi_host.setText(usable[0].host)
            self.station.settings.kiwi_host = usable[0].host
        elif current:
            index = self.kiwi_list.findData(current)
            if index >= 0:
                self.kiwi_list.setCurrentIndex(index)
        self._kiwi_force_auto = False
        if self._start_after_kiwi_pick and best is not None:
            self._start_after_kiwi_pick = False
            QTimer.singleShot(0, self.start)

    def _apply_kiwi_choice(self, index: int) -> None:
        host = self.kiwi_list.itemData(index)
        if host:
            # Activating a specific list entry is an explicit operator choice.
            # Pin it until automatic path selection is deliberately re-enabled.
            self.auto_kiwi.setChecked(False)
            self._kiwi_force_auto = False
            self.kiwi_host.setText(str(host))
            self.station.settings.kiwi_host = str(host)
            self._probe_receiver = next(
                (item for item in self._receivers if item.host == str(host)), None
            )
            self._show_manual_kiwi(str(host))

    def _on_manual_kiwi_host_edited(self, _text: str) -> None:
        """Treat typed receiver addresses as pinned operator selections."""
        self.auto_kiwi.setChecked(False)
        self._kiwi_force_auto = False
        self._recommended_receiver = None
        self.station.settings.kiwi_auto_select = False
        self.kiwi_list.setCurrentIndex(-1)
        self._show_manual_kiwi(_text.strip())
        if not self.listening():
            self.status.setText(
                "manual Kiwi pinned; automatic path selection is off"
            )

    def _show_manual_kiwi(self, host: str) -> None:
        display = host or "type a receiver address"
        self.kiwi_recommendation.setText(
            f"Manual receiver pinned: {display}\n"
            "The propagation predictor will not replace it."
        )
        self.kiwi_recommendation.setStyleSheet(
            "QLabel { padding: 5px 8px; border: 1px solid #52657a; "
            "border-radius: 4px; background: #263747; color: white; }"
        )

    def set_probe_receiver(self, receiver: KiwiReceiver) -> None:
        """Remember exact receiver coordinates for the next matching OTA decode."""
        self._probe_receiver = receiver
        if not any(item.host == receiver.host for item in self._receivers):
            self._receivers.append(receiver)

    def on_local_ptt_changed(self, keyed: bool) -> None:
        """Collect successful and failed OTA outcomes while a Kiwi monitors TX."""
        settings = self.station.settings
        eligible = (
            settings.tx_channel_profile == "radio"
            and not settings.audio_only
            and self.listening()
            and self.source.currentData() == "kiwi"
        )
        if keyed:
            if not eligible or self._probe_tx_active:
                return
            self._probe_tx_generation += 1
            self._probe_tx_active = True
            self._probe_tx_started = time.monotonic()
            self._probe_decode_seen = False
            self._probe_tx_receiver = self._probe_receiver
            return
        if not self._probe_tx_active:
            return
        self._probe_tx_active = False
        duration = time.monotonic() - self._probe_tx_started
        generation = self._probe_tx_generation
        if duration < 3.0:
            return
        # Give the streaming decoder time to finish the final buffered GOP.
        QTimer.singleShot(
            5000, lambda: self._finish_failed_probe(generation, duration)
        )

    def _finish_failed_probe(self, generation: int, duration: float) -> None:
        if (
            generation != self._probe_tx_generation
            or self._probe_tx_active
            or self._probe_decode_seen
        ):
            return
        settings = self.station.settings
        try:
            host = normalize_kiwi_host(settings.kiwi_host)
        except ValueError:
            return
        receiver = self._probe_tx_receiver
        if receiver is None or receiver.host != host:
            receiver = next((item for item in self._receivers if item.host == host), None)
        lat, lon = float(settings.kiwi_lat), float(settings.kiwi_lon)
        frequency = float(settings.kiwi_dial_mhz)
        power = max(0.1, float(settings.flex_power))
        callsign = normalize_callsign(settings.callsign)

        def save_failure() -> None:
            target = receiver or probe_receiver(host, timeout=5.0)
            if target is None:
                self.logMessage.emit(f"failed probe skipped: no coordinates for {host}")
                return
            if not (-90.0 <= target.lat <= 90.0 and -180.0 <= target.lon <= 180.0):
                self.logMessage.emit(f"failed probe skipped: invalid coordinates for {host}")
                return
            try:
                bearing = initial_bearing_deg(lat, lon, target.lat, target.lon)
                gain = antenna_gain_db(
                    settings.prop_antenna_pattern,
                    bearing,
                    settings.prop_antenna_azimuth_deg,
                    settings.prop_antenna_gain_dbi,
                )
                measurement = record_ota_failure(
                    target, lat, lon, frequency, power, gain, callsign
                )
                self.logMessage.emit(
                    f"propagation miss saved: {host} did not decode {duration:.1f} s "
                    f"at {frequency:.6f} MHz; P.533 predicted "
                    f"{measurement.predicted_snr_db:.1f} dB"
                )
            except Exception as error:
                self.logMessage.emit(f"failed probe calibration error: {error}")

        threading.Thread(
            target=save_failure, daemon=True, name="aetv-probe-failure"
        ).start()

    def _normalize_kiwi_entry(self) -> None:
        text = self.kiwi_host.text().strip()
        if not text:
            return
        try:
            host = normalize_kiwi_host(text)
        except ValueError as error:
            self.status.setText(str(error))
            return
        self.kiwi_host.setText(host)
        self.station.settings.kiwi_host = host

    def _set_listening(self, on: bool) -> None:
        self.start_button.setEnabled(not on)
        self.stop_button.setEnabled(on)
        self.find_button.setEnabled(not on)
        self.listeningChanged.emit(on)

    def _apply_state(self, state: RxState) -> None:
        self.status.setText(state.message)
        self.statusChanged.emit(state.message)
        self.progress.setValue(min(100, state.gops * 10) if state.listening else 0)

    def _on_error(self, message: str) -> None:
        self.status.setText(message)
        self.logMessage.emit(message)

    def _show_video(self, video, state: RxState) -> None:
        mode = self.station.require_codec().mode
        self.preview.enqueue_rgb(
            video,
            fps=mode.fps,
            prebuffer_frames=2 * mode.gop_frames,
            boundary_blend_frames=4,
        )
        self.status.setText(state.message)
        self.statusChanged.emit(state.message)
        self.progress.setValue(min(100, max(5, state.gops * 8)))
        self._record_probe_if_ours(state)

    def _record_probe_if_ours(self, state: RxState) -> None:
        settings = self.station.settings
        if (
            state.source != "kiwi"
            or not math.isfinite(state.snr_db)
            or normalize_callsign(state.callsign) != normalize_callsign(settings.callsign)
        ):
            return
        self._probe_decode_seen = True
        try:
            host = normalize_kiwi_host(settings.kiwi_host)
        except ValueError:
            return
        receiver = self._probe_receiver
        if receiver is None or receiver.host != host:
            receiver = next((item for item in self._receivers if item.host == host), None)
        measured_snr = float(state.snr_db)
        lat, lon = float(settings.kiwi_lat), float(settings.kiwi_lon)
        frequency = float(settings.kiwi_dial_mhz)
        power = max(0.1, float(settings.flex_power))
        callsign = normalize_callsign(settings.callsign)

        def save_measurement() -> None:
            target = receiver or probe_receiver(host, timeout=5.0)
            if target is None:
                self.logMessage.emit(f"probe calibration skipped: no coordinates for {host}")
                return
            try:
                bearing = initial_bearing_deg(lat, lon, target.lat, target.lon)
                gain = antenna_gain_db(
                    settings.prop_antenna_pattern,
                    bearing,
                    settings.prop_antenna_azimuth_deg,
                    settings.prop_antenna_gain_dbi,
                )
                measurement = record_ota_measurement(
                    target,
                    lat,
                    lon,
                    frequency,
                    power,
                    gain,
                    measured_snr,
                    callsign,
                )
                self.logMessage.emit(
                    f"propagation probe saved: {host} measured {measured_snr:.1f} dB, "
                    f"P.533 {measurement.predicted_snr_db:.1f} dB, "
                    f"residual {measurement.residual_db:+.1f} dB"
                )
            except Exception as error:
                self.logMessage.emit(f"probe calibration failed: {error}")

        threading.Thread(
            target=save_measurement, daemon=True, name="aetv-probe-calibration"
        ).start()

    def prepare_emulator(self, label: str) -> None:
        self.preview.clear()
        self.status.setText(f"Waiting for {label} loopback…")
        self.statusChanged.emit(self.status.text())
        self.progress.setValue(0)

    def show_emulated(self, video, state: RxState) -> None:
        """Display locally recovered modem video in the normal receive pane."""
        mode = self.station.require_codec().mode
        self.preview.enqueue_rgb(
            video,
            fps=mode.fps,
            prebuffer_frames=mode.gop_frames,
            boundary_blend_frames=4,
        )
        self.status.setText(state.message)
        self.statusChanged.emit(state.message)
        self.progress.setValue(min(100, max(5, state.gops * 8)))

    def _apply_ring(self, ring) -> None:
        if self._waterfall is None:
            return
        self._waterfall.set_mode(self.station.settings.mode)
        self._waterfall.set_ring(ring)
        if ring is None:
            self._waterfall.clear()
