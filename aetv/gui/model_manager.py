"""Explicit inventory and installer UI for checksum-pinned release models."""

from __future__ import annotations

import os

from PySide6.QtCore import Qt, QThread, QUrl, Signal
from PySide6.QtGui import QCloseEvent, QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from aetv.codec import (
    ReleaseModelStatus,
    download_runtime_bundle,
    inspect_release_models,
    model_cache_dir,
    runtime_bundle_bytes,
)
from aetv.config import RELEASE_MODE_LABELS, RELEASE_MODES


def _mib(value: int) -> str:
    return f"{value / (1024 * 1024):.0f} MiB"


class ModelInventoryThread(QThread):
    """Hash release files away from the GUI thread."""

    complete = Signal(object, str)

    def run(self) -> None:
        try:
            self.complete.emit(inspect_release_models(RELEASE_MODES), "")
        except Exception as error:
            self.complete.emit({}, str(error))


class _ModelDownloadThread(QThread):
    progress = Signal(str, int, int, str)
    complete = Signal(object, str)

    def __init__(self, modes: list[str], parent=None):
        super().__init__(parent)
        self._modes = modes

    def run(self) -> None:
        error_message = ""
        try:
            for mode in self._modes:
                download_runtime_bundle(
                    mode,
                    progress=lambda done, total, detail, selected=mode: self.progress.emit(
                        selected, done, total, detail
                    ),
                )
        except Exception as error:
            error_message = str(error)
        try:
            statuses = inspect_release_models(RELEASE_MODES)
        except Exception as error:
            statuses = {}
            error_message = error_message or str(error)
        self.complete.emit(statuses, error_message)


class ModelManagerDialog(QDialog):
    """Show installed release models and download selected missing modes."""

    modelsChanged = Signal(object)

    def __init__(
        self,
        selected_mode: str,
        parent=None,
        *,
        statuses: dict[str, ReleaseModelStatus] | None = None,
        first_run: bool = False,
    ):
        super().__init__(parent)
        self.setWindowTitle("AETV Model Manager")
        self.setModal(True)
        self._selected_mode = selected_mode
        self._statuses: dict[str, ReleaseModelStatus] = statuses or {}
        self._inventory_thread: ModelInventoryThread | None = None
        self._download_thread: _ModelDownloadThread | None = None
        self._default_selection_applied = False
        self._checks: dict[str, QCheckBox] = {}
        self._status_labels: dict[str, QLabel] = {}

        layout = QVBoxLayout(self)
        intro = QLabel(
            (
                "AETV needs a video model before Send and Receive are available. "
                "Choose one or both checksum-verified release models to download."
                if first_run
                else "Install and verify AETV's checksum-pinned release models."
            )
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        grid = QGridLayout()
        grid.addWidget(QLabel("Model"), 0, 0)
        grid.addWidget(QLabel("Status"), 0, 1)
        for row, mode in enumerate(RELEASE_MODES, start=1):
            check = QCheckBox(RELEASE_MODE_LABELS[mode])
            check.toggled.connect(self._refresh_controls)
            status_label = QLabel("Checking…")
            self._checks[mode] = check
            self._status_labels[mode] = status_label
            grid.addWidget(check, row, 0)
            grid.addWidget(status_label, row, 1)
        grid.setColumnStretch(0, 1)
        layout.addLayout(grid)

        cache_caption = QLabel("Downloads are stored in your per-user model cache:")
        cache_caption.setWordWrap(True)
        layout.addWidget(cache_caption)
        cache_path = QLabel(str(model_cache_dir().absolute()))
        cache_path.setTextInteractionFlags(
            cache_path.textInteractionFlags()
            | Qt.TextInteractionFlag.TextSelectableByMouse
        )
        cache_path.setWordWrap(True)
        layout.addWidget(cache_path)

        source = QLabel(
            'Source: <a href="https://huggingface.co/AETV/AETV">AETV/AETV on Hugging Face</a>'
        )
        source.setOpenExternalLinks(True)
        layout.addWidget(source)

        self.progress_label = QLabel("")
        self.progress_bar = QProgressBar()
        self.progress_label.hide()
        self.progress_bar.hide()
        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress_bar)

        actions = QHBoxLayout()
        self.open_folder_button = QPushButton("Open model folder")
        self.open_folder_button.clicked.connect(self._open_model_folder)
        self.download_button = QPushButton("Download selected")
        self.download_button.clicked.connect(self._download_selected)
        actions.addWidget(self.open_folder_button)
        actions.addStretch(1)
        actions.addWidget(self.download_button)
        layout.addLayout(actions)

        self.buttons = QDialogButtonBox()
        if first_run:
            self.continue_button = self.buttons.addButton(
                "Continue", QDialogButtonBox.ButtonRole.AcceptRole
            )
            self.buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
            self.buttons.accepted.connect(self.accept)
            self.buttons.rejected.connect(self.reject)
        else:
            self.continue_button = None
            self.buttons.addButton(QDialogButtonBox.StandardButton.Close)
            self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.resize(720, 330)

        if statuses is None:
            self._set_inventory_busy(True)
            self._inventory_thread = ModelInventoryThread(self)
            self._inventory_thread.complete.connect(self._inventory_complete)
            self._inventory_thread.finished.connect(self._inventory_finished)
            self._inventory_thread.start()
        else:
            self._apply_statuses(statuses)

    @property
    def statuses(self) -> dict[str, ReleaseModelStatus]:
        return dict(self._statuses)

    def installed_modes(self) -> list[str]:
        return [
            mode
            for mode in RELEASE_MODES
            if self._statuses.get(mode) is not None
            and self._statuses[mode].installed
        ]

    def _inventory_complete(
        self, statuses: dict[str, ReleaseModelStatus], error_message: str
    ) -> None:
        self._set_inventory_busy(False)
        if error_message:
            QMessageBox.warning(self, "AETV Model Manager", error_message)
        self._apply_statuses(statuses)

    def _inventory_finished(self) -> None:
        self._inventory_thread = None
        self._apply_statuses(self._statuses)

    def _apply_statuses(self, statuses: dict[str, ReleaseModelStatus]) -> None:
        self._statuses = statuses
        offline = bool(os.environ.get("AETV_OFFLINE"))
        for mode in RELEASE_MODES:
            status = statuses.get(mode)
            check = self._checks[mode]
            label = self._status_labels[mode]
            if status is not None and status.installed:
                check.setChecked(False)
                check.setEnabled(False)
                label.setText(f"Installed · {status.backend}")
                label.setToolTip(str(status.path or ""))
            else:
                check.setEnabled(not offline and not self._download_busy())
                problem = status.problem if status is not None else "could not inspect"
                prefix = "Needs repair" if "checksum" in problem else "Not installed"
                label.setText(f"{prefix} · {_mib(runtime_bundle_bytes(mode))} download")
                label.setToolTip(problem)

        if not self._default_selection_applied:
            preferred = self._checks.get(self._selected_mode)
            if preferred is not None and preferred.isEnabled():
                preferred.setChecked(True)
            self._default_selection_applied = True
        if offline:
            self.progress_label.setText(
                "Downloads are disabled because AETV_OFFLINE is set. "
                "A local runtime can still be selected in Settings."
            )
            self.progress_label.show()
        self._refresh_controls()

    def _set_inventory_busy(self, busy: bool) -> None:
        for check in self._checks.values():
            check.setEnabled(not busy)
        self.download_button.setEnabled(not busy)
        self.open_folder_button.setEnabled(not busy)
        if self.continue_button is not None:
            self.continue_button.setEnabled(not busy and bool(self.installed_modes()))

    def _download_busy(self) -> bool:
        return self._download_thread is not None and self._download_thread.isRunning()

    def _refresh_controls(self) -> None:
        busy = self._download_busy() or (
            self._inventory_thread is not None and self._inventory_thread.isRunning()
        )
        selected = any(
            check.isChecked() and check.isEnabled() for check in self._checks.values()
        )
        self.download_button.setEnabled(
            selected and not busy and not bool(os.environ.get("AETV_OFFLINE"))
        )
        self.open_folder_button.setEnabled(not busy)
        if self.continue_button is not None:
            self.continue_button.setEnabled(bool(self.installed_modes()) and not busy)

    def _download_selected(self) -> None:
        modes = [
            mode
            for mode in RELEASE_MODES
            if self._checks[mode].isChecked() and self._checks[mode].isEnabled()
        ]
        if not modes:
            return
        for check in self._checks.values():
            check.setEnabled(False)
        self.download_button.setEnabled(False)
        self.open_folder_button.setEnabled(False)
        if self.continue_button is not None:
            self.continue_button.setEnabled(False)
        self.progress_label.setText("Starting model download…")
        self.progress_label.show()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self._download_thread = _ModelDownloadThread(modes, self)
        self._download_thread.progress.connect(self._download_progress)
        self._download_thread.complete.connect(self._download_complete)
        self._download_thread.finished.connect(self._download_finished)
        self._download_thread.start()

    def _download_progress(
        self, mode: str, done: int, total: int, detail: str
    ) -> None:
        label = RELEASE_MODE_LABELS[mode].split(" — ", 1)[0]
        self.progress_label.setText(
            f"{label}: {detail} · {_mib(done)} of {_mib(total)}"
        )
        self.progress_bar.setValue(round(1000 * done / max(1, total)))

    def _download_complete(
        self, statuses: dict[str, ReleaseModelStatus], error_message: str
    ) -> None:
        self.progress_bar.hide()
        if error_message:
            self.progress_label.setText(
                "Download failed. You can retry the selected model."
            )
            QMessageBox.warning(self, "AETV model download", error_message)
        else:
            self.progress_label.setText("Selected models are installed and verified.")
        self._apply_statuses(statuses)
        self.modelsChanged.emit(statuses)

    def _download_finished(self) -> None:
        self._download_thread = None
        self._apply_statuses(self._statuses)

    def _open_model_folder(self) -> None:
        folder = model_cache_dir()
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder.resolve())))

    def reject(self) -> None:
        if self._download_busy() or (
            self._inventory_thread is not None and self._inventory_thread.isRunning()
        ):
            return
        super().reject()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._download_busy() or (
            self._inventory_thread is not None and self._inventory_thread.isRunning()
        ):
            event.ignore()
            return
        super().closeEvent(event)
