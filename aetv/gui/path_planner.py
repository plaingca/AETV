"""Propagation-ranked KiwiSDR path planner."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFrame,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from aetv.kiwi import find_receivers, normalize_kiwi_host, probe_receiver
from aetv.propagation import (
    CalibrationStore,
    PropagationPredictor,
    antenna_gain_db,
    fetch_space_weather,
    initial_bearing_deg,
    native_runtime_status,
)


class _PlannerThread(QThread):
    finished_rows = Signal(object, str, str)

    def __init__(self, settings, seed_receivers=None, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.seed_receivers = list(seed_receivers or [])

    def run(self) -> None:
        try:
            settings = self.settings
            receivers = self.seed_receivers or find_receivers(
                settings.kiwi_lat, settings.kiwi_lon, max_km=settings.kiwi_max_km
            )
            try:
                configured_host = normalize_kiwi_host(settings.kiwi_host)
            except ValueError:
                configured_host = ""
            if configured_host and not any(item.host == configured_host for item in receivers):
                configured = probe_receiver(configured_host, timeout=4.0)
                if configured is not None:
                    receivers.insert(0, configured)
            candidates = [item for item in receivers if item.usable]
            candidates.sort(
                key=lambda item: (
                    -(item.success_probability or -1.0), item.km or 1e9
                )
            )
            # A shortlist is more useful to a human and keeps native P.533 forecasts quick.
            candidates = candidates[:20]
            weather = fetch_space_weather()
            predictor = PropagationPredictor()
            now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
            rows_by_host = {}

            def estimate(receiver, offset):
                bearing = initial_bearing_deg(
                    settings.kiwi_lat, settings.kiwi_lon, receiver.lat, receiver.lon
                )
                gain = antenna_gain_db(
                    settings.prop_antenna_pattern,
                    bearing,
                    settings.prop_antenna_azimuth_deg,
                    settings.prop_antenna_gain_dbi,
                )
                return predictor.predict(
                    receiver,
                    settings.kiwi_lat,
                    settings.kiwi_lon,
                    settings.kiwi_dial_mhz,
                    max(0.1, float(settings.flex_power)),
                    tx_gain_dbi=gain,
                    when=now + timedelta(hours=offset),
                    weather=weather,
                )

            jobs = []
            for receiver in candidates:
                # Always recompute "now": seed receiver predictions may predate
                # an FT8 calibration import and must never bypass fresh samples.
                rows_by_host[receiver.host] = [receiver, None, []]
                jobs.append((receiver, 0))
                jobs.extend((receiver, offset) for offset in (2, 4, 6, 8, 10, 12))

            with ThreadPoolExecutor(max_workers=min(8, max(1, len(jobs)))) as pool:
                pending = {pool.submit(estimate, receiver, offset): (receiver, offset)
                           for receiver, offset in jobs}
                for future in as_completed(pending):
                    if self.isInterruptionRequested():
                        for job in pending:
                            job.cancel()
                        return
                    receiver, offset = pending[future]
                    try:
                        result = future.result()
                    except Exception:
                        continue
                    entry = rows_by_host[receiver.host]
                    if offset == 0:
                        entry[1] = result
                    entry[2].append(result)

            rows = []
            for receiver, current, estimates in rows_by_host.values():
                if current is None or not estimates:
                    continue
                best = max(estimates, key=lambda item: item.success_probability)
                rows.append((receiver, current, best))
            rows.sort(
                key=lambda row: (
                    -row[1].success_probability,
                    -row[1].calibrated_snr_db,
                    row[0].km or 1e9,
                )
            )
            runtime, _ = native_runtime_status()
            source = f"SSN {weather.ssn:.0f}, Kp {weather.kp:.1f} · {weather.source}"
            if not runtime:
                source += " · coarse fallback (install P.533 runtime for absolute estimates)"
            self.finished_rows.emit(rows, "", source)
        except Exception as error:
            self.finished_rows.emit([], str(error), "")


class PathPlannerDialog(QDialog):
    def __init__(self, station, receive_panel, parent=None):
        super().__init__(parent)
        self.station = station
        self.receive_panel = receive_panel
        self._thread: _PlannerThread | None = None
        self._rows = []
        self.setWindowTitle("AETV Kiwi path planner")
        self.resize(960, 650)

        self.hero = QFrame()
        self.hero.setObjectName("pathHero")
        self.hero.setStyleSheet(
            "QFrame#pathHero { border: 1px solid #555; border-radius: 8px; "
            "background: #25272b; }"
        )
        hero_layout = QVBoxLayout(self.hero)
        hero_layout.setContentsMargins(16, 12, 16, 12)
        self.best_title = QLabel("Finding the best receiver…")
        title_font = QFont(self.best_title.font())
        title_font.setPointSize(title_font.pointSize() + 3)
        title_font.setBold(True)
        self.best_title.setFont(title_font)
        self.best_detail = QLabel("Comparing current conditions and the next 12 hours")
        self.best_detail.setWordWrap(True)
        hero_layout.addWidget(self.best_title)
        hero_layout.addWidget(self.best_detail)

        self.summary = QLabel("Loading the Kiwi directory…")
        self.summary.setWordWrap(True)
        self.busy = QProgressBar()
        self.busy.setRange(0, 0)
        self.busy.setMaximumHeight(4)
        self.busy.setTextVisible(False)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            [
                "Receiver",
                "Location",
                "Conditions now",
                "Chance now",
                "Best time",
                "Chance then",
                "Distance",
            ]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.doubleClicked.connect(self.use_selected)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)

        self.selection_detail = QLabel("Select a receiver to see the path details.")
        self.selection_detail.setWordWrap(True)
        self.selection_detail.setStyleSheet("QLabel { color: #bbb; padding: 4px; }")

        self.refresh_button = QPushButton("Refresh forecast")
        self.use_button = QPushButton("Use selected receiver")
        self.use_button.setDefault(True)
        self.calibrate_button = QPushButton("Calibrate with FT8…")
        close_button = QPushButton("Close")
        self.refresh_button.clicked.connect(self.refresh)
        self.use_button.clicked.connect(self.use_selected)
        self.calibrate_button.clicked.connect(self._open_ft8_calibration)
        close_button.clicked.connect(self.accept)
        buttons = QHBoxLayout()
        buttons.addWidget(self.refresh_button)
        buttons.addWidget(self.use_button)
        buttons.addWidget(self.calibrate_button)
        buttons.addStretch(1)
        buttons.addWidget(close_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.hero)
        layout.addWidget(self.summary)
        layout.addWidget(self.busy)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.selection_detail)
        layout.addLayout(buttons)
        self.refresh()

    def refresh(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            return
        self.refresh_button.setEnabled(False)
        self.use_button.setEnabled(False)
        self.best_title.setText("Finding the best receiver…")
        self.best_detail.setText("Comparing current conditions and the next 12 hours")
        self.summary.setText("Calculating a shortlist. This usually takes a few seconds.")
        self.busy.show()
        self.table.setRowCount(0)
        self._thread = _PlannerThread(
            self.station.settings, getattr(self.receive_panel, "_receivers", []), self
        )
        self._thread.finished_rows.connect(self._apply_rows)
        self._thread.start()

    def _apply_rows(self, rows, error: str, source: str) -> None:
        self.refresh_button.setEnabled(True)
        self.use_button.setEnabled(bool(rows))
        self.busy.hide()
        if error:
            self.best_title.setText("Forecast unavailable")
            self.best_detail.setText(error)
            self.summary.setText("Check your network connection and station location, then retry.")
            return
        self._rows = list(rows)
        count = len(CalibrationStore().load())
        self.summary.setText(
            f"{len(rows)} strongest reachable receivers at "
            f"{self.station.settings.kiwi_dial_mhz:.6f} MHz  •  {count} OTA calibration "
            f"sample{'s' if count != 1 else ''}  •  {source}"
        )
        self.table.setRowCount(len(rows))
        for row_index, (receiver, now, best) in enumerate(rows):
            place = receiver.loc or receiver.name or receiver.host
            condition = self._condition(now.success_probability)
            best_time = datetime.fromisoformat(best.when_utc).astimezone().strftime("%H:%M")
            values = [
                receiver.host,
                place,
                f"{condition} · {now.calibrated_snr_db:+.1f} dB "
                f"({now.correction_db:+.1f} cal)",
                f"{now.success_probability:.0f}%",
                f"{best_time} local",
                f"{best.success_probability:.0f}%",
                f"{best.distance_km:.0f} km",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column in {3, 5, 6}:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if column == 2:
                    item.setForeground(QColor(self._condition_color(now.success_probability)))
                self.table.setItem(row_index, column, item)
        if rows:
            self.table.selectRow(0)
            receiver, now, best = rows[0]
            place = receiver.loc or receiver.name or receiver.host
            self.best_title.setText(f"Best right now: {place}")
            self.best_detail.setText(
                f"{self._condition(now.success_probability)} conditions  •  "
                f"{now.success_probability:.0f}% chance of a usable decode  •  "
                f"expected pilot SNR {now.calibrated_snr_db:.1f} ± {now.uncertainty_db:.1f} dB  •  "
                f"calibration {now.correction_db:+.1f} dB from {now.samples} samples"
            )
        else:
            self.best_title.setText("No usable receivers found")
            self.best_detail.setText("Try increasing the maximum search distance in Settings.")

    @staticmethod
    def _condition(probability: float) -> str:
        if probability >= 70:
            return "Good"
        if probability >= 35:
            return "Marginal"
        return "Poor"

    @staticmethod
    def _condition_color(probability: float) -> str:
        if probability >= 70:
            return "#78d78a"
        if probability >= 35:
            return "#e4c65d"
        return "#ef8d8d"

    def _selection_changed(self) -> None:
        row = self.table.currentRow()
        if row < 0 or row >= len(self._rows):
            return
        receiver, now, best = self._rows[row]
        best_utc = datetime.fromisoformat(best.when_utc).strftime("%H:%M UTC")
        self.selection_detail.setText(
            f"{receiver.host}  •  bearing {best.bearing_deg:.0f}°  •  "
            f"best {best.calibrated_snr_db:.1f} ± {best.uncertainty_db:.1f} dB at {best_utc}  •  "
            f"calibration {best.correction_db:+.1f} dB  •  "
            f"{best.engine}  •  {best.samples} matching calibration samples"
        )

    def use_selected(self, *_args) -> None:
        row = self.table.currentRow()
        if row < 0 or row >= len(self._rows):
            return
        receiver, _now, best = self._rows[row]
        self.station.settings.kiwi_host = receiver.host
        self.receive_panel.kiwi_host.setText(receiver.host)
        index = self.receive_panel.source.findData("kiwi")
        if index >= 0:
            self.receive_panel.source.setCurrentIndex(index)
        self.receive_panel.set_probe_receiver(receiver)
        restart = " Stop and restart receive to retune." if self.receive_panel.listening() else ""
        place = receiver.loc or receiver.name or receiver.host
        self.best_title.setText(f"Selected: {place}")
        self.best_detail.setText(
            f"This receiver will be used when you start receiving.{restart}"
        )

    def _open_ft8_calibration(self) -> None:
        owner = self.parent()
        opener = getattr(owner, "open_ft8_calibration", None)
        if opener is not None:
            opener()

    def closeEvent(self, event) -> None:
        if self._thread is not None and self._thread.isRunning():
            self._thread.requestInterruption()
        super().closeEvent(event)
