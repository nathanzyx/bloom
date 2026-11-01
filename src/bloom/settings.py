"""
    settings.py
    
    settings as displayed on the control panel.

User-facing settings, shared by the control panel and the display window.

The panel edits a Settings object; the app compares it with what is currently
built and rebuilds only what changed (a new shape re-bakes geometry, a colour
is just a uniform). Colours always come from the now-playing cover art.

Settings persist between runs as JSON in the per-user config folder (see
`default_path`). The app saves a second after the last change rather than on
every slider tick, and once more on exit. Loading is forgiving: a missing or
corrupt file, an unknown key or a wrongly typed value just falls back to the
default for that field, so an old file never stops the app from starting.
"""

from __future__ import annotations
import json
import os
import sys
import time
from dataclasses import dataclass, fields
from pathlib import Path
from .shape import CIRCLE


@dataclass
class Settings:
    shape: str = CIRCLE # circular (default), fullscreen, or an SVG file path
    background: tuple[float, float, float] = (0.0, 0.0, 0.0) # sRGB: the display color when idle
    frame: tuple[float, float, float] = (0.0, 0.0, 0.0) # sRGB: the material color outside the display
    idle_glow: float = 0.5 # >0: idle screen = dim palette colour (0..1). overrides background color if >0
    # (these five are the "Medium" preset, see PRESETS)
    energy: float = 0.29 # 0 = low reactance, 1 = high reactance
    balance: float = 0.21 # 0 = mostly low intensity, 1 = mostly high
    size: float = 1.3 # overall blob size: 0 = small .. SIZE_MAX = largest
    branching: float = 0.64 # how often noise pulses swell out: 0 = never
    waves: float = 0.3 # snare waves (the pulsing ring): 0 = off
    edge_margin: float = 0.385 # fraction of the shape's depth

    def copy(self) -> Settings:
        return Settings(**{k: getattr(self, k) for k in self.__dataclass_fields__})

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def reset(self, keys) -> None:
        """Put the given fields back to their defaults."""
        d = Settings()
        for k in keys: setattr(self, k, getattr(d, k))

    @classmethod
    def from_dict(cls, data: dict) -> Settings:
        """Build from saved data, keeping the default for anything invalid."""
        s = cls()
        if not isinstance(data, dict): return s
        for f in fields(cls):
            if f.name not in data: continue
            value = _coerce(getattr(s, f.name), data[f.name], f.name)
            if value is not _INVALID: setattr(s, f.name, value)
        return s


_INVALID = object()

# what each panel tab's "Reset to defaults" button covers.
REACTANCE_KEYS = ("energy", "size", "balance", "branching", "waves")

# The Bloom tab's preset buttons. Medium is also the default.
PRESETS = {
    "Calm": dict(energy=0.08, size=1.3, balance=0.2, branching=0.3, waves=0.0),
    "Medium (Default)": dict(energy=0.29, size=1.3, balance=0.21, branching=0.64, waves=0.3),
    "More Reactive": dict(energy=1.0, size=1.1, balance=0.21, branching=1.0, waves=1.0),
}
DISPLAY_KEYS = ("shape", "background", "frame", "idle_glow", "edge_margin")


def _coerce(default, value, name: str):
    """`value` converted to the type of `default`, or _INVALID."""
    try:
        if isinstance(default, float):
            v = float(value)
            return v if v == v else _INVALID # reject NaN
        if isinstance(default, tuple):
            v = tuple(float(c) for c in value)
            return v if len(v) == len(default) else _INVALID
        if isinstance(default, str):
            return value if isinstance(value, str) else _INVALID
    except (TypeError, ValueError): pass
    return _INVALID


def default_path() -> Path:
    """
    settings.json in the per-user config folder:
    %APPDATA%\\reactance on Windows, 
    ~/Library/Application Support/reactance on macOS, 
    $XDG_CONFIG_HOME (or ~/.config)/reactance elsewhere.
    """
    if sys.platform == "win32": base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin": base = Path.home() / "Library" / "Application Support"
    else: base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "reactance" / "settings.json"


def load(path: Path) -> Settings | None:
    """The saved settings, or None if there is no (readable) file."""
    try:
        return Settings.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return None


def save(settings: Settings, path: Path) -> None:
    """
    Write atomically (temp file + rename), so a crash mid-write can't
    leave a half-written file behind.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(settings.to_dict(), indent=2), encoding="utf-8")
    os.replace(tmp, path)


class SettingsSaver:
    """
    Saves settings shortly after they stop changing.

    Call `poll()` once per frame. Dragging a slider changes the value every
    frame; saving waits until it has been still for `delay_s`, so a drag is
    one write, not hundreds. `flush()` writes any pending change immediately
    (on exit).

    `saved` is what the file currently holds (None if there is no file), so
    command-line overrides and first-run defaults get written too.
    """

    def __init__(self, settings: Settings, path: Path, saved: Settings | None = None, delay_s: float = 1.0) -> None:
        self.settings, self.path, self.delay_s = settings, Path(path), delay_s
        self._saved = saved.to_dict() if saved is not None else None
        self._last_seen = self._saved
        self._changed_at: float | None = None

    def poll(self, now: float | None = None) -> bool:
        """Returns True if it wrote the file."""
        now = time.monotonic() if now is None else now
        current = self.settings.to_dict()
        if current == self._saved:
            self._changed_at = None
            return False
        if current != self._last_seen or self._changed_at is None:
            self._last_seen, self._changed_at = current, now # still changing: restart wait
            return False
        if now - self._changed_at < self.delay_s:
            return False
        return self._write(current)

    def flush(self) -> bool:
        current = self.settings.to_dict()
        return current != self._saved and self._write(current)

    def _write(self, current: dict) -> bool:
        try: save(self.settings, self.path)
        except OSError as e:
            print(f"could not save settings to {self.path}: {e}")
            self._saved = current  # don't retry every frame
            return False
        self._saved, self._changed_at = current, None
        return True
