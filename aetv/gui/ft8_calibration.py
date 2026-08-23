"""Operator-controlled FT8 propagation calibration wizard."""

from __future__ import annotations

import time
from datetime import datetime, timezone

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from aetv.ft8_probe import (
    fetch_psk_reporter_spots,
    import_no_report_calibration,
    import_spot_calibration,
    load_probe_runs,
    maidenhead_grid,
    parse_ft8_frequencies,
    spots_for_probe_runs,
    transmit_ft8_probe,
)


class _ProbeTxThread(QThread):
    status = Signal(str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, settings, frequency: float, message: str, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.frequency = frequency
        self.message = message

    def run(self) -> None:
        try:
            run = transmit_ft8_probe(
                self.settings,
                self.frequency,
                self.message,
                on_status=self.status.emit,
            )
            self.completed.emit(run)
        except Exception as error:
            self.failed.emit(str(error))


class _SpotImportThread(QThread):
    completed = Signal(object, int, int)
    failed = Signal(str)

    def __init__(self, settings, runs, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.runs = list(runs)

    def run(self) -> None:
        try:
            since = max(int(time.time()) - 24 * 3600, min(run.timestamp for run in self.runs) - 120)
            spots = fetch_psk_reporter_spots(self.settings.callsign, since)
            spots = spots_for_probe_runs(spots, self.runs)
            imported = import_spot_calibration(spots, self.settings, self.runs)
            misses = import_no_report_calibration(spots, self.runs, self.settings)
            self.completed.emit(spots, imported, misses)
        except Exception as error:
            self.failed.emit(str(error))


class Ft8CalibrationDialog(QDialog):
    calibrationImported = Signal(int)

    def __init__(self, station, parent=None):
        super().__init__(parent)
        self.station = station
        self._tx_thread: _ProbeTxThread | None = None
        self._import_thread: _SpotImportThread | None = None
        self._last_query = 0.0
        self.setWindowTitle("AETV FT8 propagation calibration")
        self.resize(760, 560)

        settings = station.settings
        self.grid = maidenhead_grid(settings.kiwi_lat, settings.kiwi_lon, 4)
        self.message = f"CQ {settings.callsign} {self.grid}"
        title = QLabel(f"Transmit one standard FT8 CQ per band:  {self.message}")
        title.setStyleSheet("QLabel { font-size: 16px; font-weight: bold; }")
        explanation = QLabel(
            "Select a band and explicitly authorize one 15-second transmission. "
            "AETV waits for the next UTC FT8 slot, tunes the Flex, sends the CQ, "
            "and unkeys. After at least five minutes, import PSK Reporter spots "
            "to calibrate P.533 with measured 2.5 kHz-reference SNR."
        )
        explanation.setWordWrap(True)
        warning = QLabel(
            "Propagation calibration does not check your licence, regional band plan, "
            "antenna limits, or whether the selected audio offset is clear. Listen first."
        )
        warning.setWordWrap(True)
        warning.setStyleSheet(
            "QLabel { padding: 8px; border: 1px solid #8a6d2b; "
            "border-radius: 5px; background: #4b3c1d; color: white; }"
        )

        self.frequencies = parse_ft8_frequencies(settings.ft8_probe_frequencies_mhz)
        self.table = QTableWidget(len(self.frequencies), 4)
        self.table.setHorizontalHeaderLabels(["Band dial", "RF tones", "Status", "Spots"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        for row, frequency in enumerate(self.frequencies):
            values = [
                f"{frequency:.3f} MHz",
                f"{frequency + 0.001:.3f}–{frequency + 0.001050:.6f} MHz",
                "Not sent",
                "—",
            ]
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(value))
        self.table.resizeColumnsToContents()
        if self.frequencies:
            self.table.selectRow(0)

        self.status = QLabel("Ready. Confirm the radio is connected and the selected FT8 offset is clear.")
        self.status.setWordWrap(True)
        self.transmit_button = QPushButton("Transmit selected FT8 probe…")
        self.import_button = QPushButton("Import PSK Reporter spots")
        close_button = QPushButton("Close")
        self.transmit_button.clicked.connect(self._confirm_transmit)
        self.import_button.clicked.connect(self._import_spots)
        close_button.clicked.connect(self.accept)
        buttons = QHBoxLayout()
        buttons.addWidget(self.transmit_button)
        buttons.addWidget(self.import_button)
        buttons.addStretch(1)
        buttons.addWidget(close_button)

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(explanation)
        layout.addWidget(warning)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.status)
        layout.addLayout(buttons)

    def busy(self) -> bool:
        return bool(
            (self._tx_thread is not None and self._tx_thread.isRunning())
            or (self._import_thread is not None and self._import_thread.isRunning())
        )

    def transmitting(self) -> bool:
        return self._tx_thread is not None and self._tx_thread.isRunning()

    def closeEvent(self, event) -> None:
        if self.busy():
            self.status.setText("Wait for the active FT8 operation to finish before closing.")
            event.ignore()
            return
        super().closeEvent(event)

    def _confirm_transmit(self) -> None:
        row = self.table.currentRow()
        if row < 0 or row >= len(self.frequencies):
            return
        main = self.parent()
        tx_panel = getattr(main, "tx", None)
        if tx_panel is not None and tx_panel.transmitting():
            self.status.setText("Finish or cancel the current AETV transmission first.")
            return
        frequency = self.frequencies[row]
        answer = QMessageBox.question(
            self,
            "Authorize one FT8 probe",
            f"Transmit {self.message}\n\n"
            f"Dial: {frequency:.3f} MHz DIGU\n"
            f"Audio: 1000–1050 Hz\n"
            f"Power: {self.station.settings.flex_power} W\n\n"
            "Confirm this emission is authorized and the offset is clear.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.transmit_button.setEnabled(False)
        self.import_button.setEnabled(False)
        self.table.item(row, 2).setText("Preparing…")
        self._tx_thread = _ProbeTxThread(
            self.station.settings, frequency, self.message, self
        )
        self._tx_thread.status.connect(self._set_tx_status)
        self._tx_thread.completed.connect(lambda run: self._tx_complete(row, run))
        self._tx_thread.failed.connect(lambda error: self._tx_failed(row, error))
        self._tx_thread.start()

    def _set_tx_status(self, message: str) -> None:
        self.status.setText(message)

    def _tx_complete(self, row: int, run) -> None:
        self.transmit_button.setEnabled(True)
        self.import_button.setEnabled(True)
        sent = datetime.fromtimestamp(run.timestamp, timezone.utc).strftime("%H:%M:%S UTC")
        ready = datetime.fromtimestamp(run.timestamp + 300, timezone.utc).strftime("%H:%M UTC")
        self.table.item(row, 2).setText(f"Sent {sent}")
        self.status.setText(
            f"Probe sent and unkeyed. PSK Reporter import is recommended after {ready}."
        )

    def _tx_failed(self, row: int, error: str) -> None:
        self.transmit_button.setEnabled(True)
        self.import_button.setEnabled(True)
        self.table.item(row, 2).setText("Failed")
        self.status.setText(error)

    def _import_spots(self) -> None:
        now = time.time()
        if now - self._last_query < 300:
            self.status.setText(
                f"PSK Reporter requests no more than one query per five minutes; retry in "
                f"{int(300 - (now - self._last_query))} seconds."
            )
            return
        runs = [run for run in load_probe_runs() if now - run.timestamp <= 24 * 3600]
        if not runs:
            self.status.setText("No FT8 probes from the last 24 hours are recorded.")
            return
        wait_seconds = 300 - int(now - max(run.timestamp for run in runs))
        if wait_seconds > 0:
            self.status.setText(
                f"Allow PSK Reporter five minutes to receive the newest probe; retry in "
                f"{wait_seconds} seconds."
            )
            return
        self._last_query = now
        self.transmit_button.setEnabled(False)
        self.import_button.setEnabled(False)
        self.status.setText("Waiting for PSK Reporter…")
        self._import_thread = _SpotImportThread(self.station.settings, runs, self)
        self._import_thread.completed.connect(self._spots_imported)
        self._import_thread.failed.connect(self._import_failed)
        self._import_thread.start()

    def _spots_imported(self, spots, imported: int, misses: int) -> None:
        self.transmit_button.setEnabled(True)
        self.import_button.setEnabled(True)
        for row, dial in enumerate(self.frequencies):
            matching = [
                spot for spot in spots
                if abs(spot.frequency_hz / 1_000_000.0 - (dial + 0.001)) < 0.005
            ]
            if matching:
                median = sorted(spot.snr_db for spot in matching)[len(matching) // 2]
                self.table.item(row, 3).setText(f"{len(matching)} · median {median:+.0f} dB")
            elif any(abs(run.dial_mhz - dial) < 0.0005 for run in load_probe_runs()):
                self.table.item(row, 3).setText("No reports · miss recorded")
        self.status.setText(
            f"Imported {imported} FT8 reports and {misses} no-report probe"
            f"{'s' if misses != 1 else ''}; forecasts are refreshing."
        )
        self.calibrationImported.emit(imported + misses)

    def _import_failed(self, error: str) -> None:
        self.transmit_button.setEnabled(True)
        self.import_button.setEnabled(True)
        self.status.setText(f"PSK Reporter import failed: {error}")
