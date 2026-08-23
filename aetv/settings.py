"""Persisted ham-station settings. JSON in the user config directory.

Device pickers store names, not PortAudio indices: a USB interface
that moved from index 3 to 5 must not silently key the wrong card.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .config import AETV_MODES
from .hfchannel import CHANNEL_PROFILES
from .kiwi import normalize_kiwi_host

CALLSIGN_RE = re.compile(r"^[A-Z0-9/]{1,8}$")


def config_dir() -> Path:
    if os.name == "nt":
        root = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    else:
        root = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    path = root / "AETV"
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_receive_dir() -> Path:
    path = Path.home() / "AETV" / "received"
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_callsign(text: str) -> str:
    token = text.strip().split()[0] if text.strip() else ""
    return re.sub(r"[^A-Z0-9/]", "", token.upper())[:8]


@dataclass
class StationSettings:
    callsign: str = "N0CALL"
    mode: str = "V7"
    checkpoint: str = ""
    torch_device: str = ""

    audio_input: str = ""
    audio_output: str = ""
    tx_level: float = 0.7
    rx_source: str = "soundcard"  # soundcard | flex | kiwi
    buffer_seconds: float = 90.0
    decode_every_s: float = 0.1

    kiwi_host: str = ""
    kiwi_password: str = ""
    kiwi_user: str = ""
    kiwi_dial_mhz: float = 7.088
    kiwi_lat: float = 49.26
    kiwi_lon: float = -123.11
    kiwi_max_km: float = 2500.0

    cat_backend: str = "none"  # none | hamlib | flex | rts | dtr
    hamlib_model: int = 0
    hamlib_device: str = ""
    hamlib_baud: int = 0
    rigctld_host: str = "127.0.0.1"
    rigctld_port: int = 4532
    flex_host: str = ""
    flex_power: int = 5
    flex_native_audio: bool = True
    freq_mhz: float | None = None
    require_mode: str = "DIGU"
    serial_port: str = ""
    ptt_lead_s: float = 0.3
    ptt_tail_s: float = 0.3
    audio_only: bool = False

    camera_index: int = 0
    gops: int = 10
    receive_dir: str = ""
    autosave: bool = True
    debug_capture: bool = True
    window_layout: str = "split"
    tx_channel_profile: str = "radio"  # radio | a CHANNEL_PROFILES key

    def validate(self) -> list[str]:
        problems: list[str] = []
        self.callsign = normalize_callsign(self.callsign) or "N0CALL"
        if not CALLSIGN_RE.match(self.callsign):
            problems.append("callsign must be 1-8 characters from A-Z, 0-9, /")
        if self.mode not in AETV_MODES:
            problems.append(f"unknown mode {self.mode!r}")
        if not 0.05 <= self.tx_level <= 1.0:
            problems.append("TX level must be between 0.05 and 1.0")
        if self.gops < 1:
            problems.append("GOP count must be >= 1")
        if self.tx_channel_profile != "radio" and self.tx_channel_profile not in CHANNEL_PROFILES:
            problems.append(f"unknown TX channel profile {self.tx_channel_profile!r}")
        if self.cat_backend not in {"none", "hamlib", "rigctld", "flex", "rts", "dtr"}:
            problems.append(f"unknown CAT backend {self.cat_backend!r}")
        if self.rx_source not in {"soundcard", "flex", "kiwi"}:
            problems.append(f"unknown receive source {self.rx_source!r}")
        if self.rx_source == "kiwi" and not self.kiwi_host:
            problems.append("KiwiSDR host is empty")
        elif self.kiwi_host:
            try:
                self.kiwi_host = normalize_kiwi_host(self.kiwi_host)
            except ValueError as error:
                problems.append(str(error))
        if self.cat_backend == "flex" and not self.flex_host:
            problems.append("Flex host is empty")
        if self.cat_backend == "hamlib" and self.hamlib_model <= 0:
            problems.append("Hamlib radio model is not selected")
        if self.rx_source == "flex" and not self.flex_host:
            problems.append("Flex host is empty")
        if self.cat_backend in {"rts", "dtr"} and not self.serial_port:
            problems.append("serial PTT port is empty")
        return problems

    def receive_path(self) -> Path:
        return Path(self.receive_dir) if self.receive_dir else default_receive_dir()


def settings_path() -> Path:
    return config_dir() / "settings.json"


def load_settings(path: Path | None = None) -> StationSettings:
    target = path or settings_path()
    if not target.is_file():
        settings = StationSettings()
        settings.receive_dir = str(default_receive_dir())
        return settings
    raw = json.loads(target.read_text(encoding="utf-8"))
    allowed = {item.name for item in fields(StationSettings)}
    clean = {key: value for key, value in raw.items() if key in allowed}
    settings = StationSettings(**clean)
    if settings.kiwi_host:
        try:
            settings.kiwi_host = normalize_kiwi_host(settings.kiwi_host)
        except ValueError:
            pass
    if (
        settings.cat_backend == "flex"
        and settings.flex_native_audio
        and settings.rx_source == "soundcard"
    ):
        settings.rx_source = "flex"
    if not settings.receive_dir:
        settings.receive_dir = str(default_receive_dir())
    return settings


def save_settings(settings: StationSettings, path: Path | None = None) -> Path:
    target = path or settings_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(asdict(settings), indent=2) + "\n", encoding="utf-8")
    return target
