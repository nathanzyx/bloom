"""Entry point: display window + control panel, and the loop that drives them.

  python -m bloom                                  # live audio, colours from now playing
  python -m bloom --shape my_shape.svg --energy 0.8
  python -m bloom --wav song.wav                   # analyse a file instead
  python -m bloom --demo                           # built-in drum groove
  python -m bloom --no-panel                       # display only
  python -m bloom --reset-settings                 # start from the defaults

Settings are saved automatically (to the per-user config folder) and restored
on the next run. A flag given on the command line overrides the saved value.
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np

from .capture import SR, AnalyzerThread, FeatureBus, FileSource, LoopbackSource, demo_loop, save_wav
from .cover import NowPlayingWatcher, ProviderUnavailable, create_provider, extract_from_pixels
from .motion import AudioMapper, Reactivity
from .render import DisplayWindow
from .settings import Settings, SettingsSaver, default_path, load
from .shape import CIRCLE, FULLSCREEN, bake, load_shape

BAKE_RESOLUTION = 512


def default_palette():
    """Shown until the first cover art arrives (or if now-playing is unavailable)."""
    return extract_from_pixels(np.full((2, 2, 3), (0.25, 0.45, 0.9)))   # a soft blue


def make_source(args):
    if args.demo:
        tmp = Path(tempfile.gettempdir()) / "reactance_demo.wav"
        save_wav(tmp, demo_loop(bars=4), SR)
        return FileSource(tmp)
    if args.wav:
        return FileSource(args.wav)
    return LoopbackSource(args.device)


class App:
    """Applies Settings to the running pipeline, rebuilding only what changed."""

    PALETTE_FADE_S = 1.2

    def __init__(self, settings, window, watcher=None) -> None:
        self.settings = settings
        self.window = window
        self.watcher = watcher
        self.applied = None
        self.shape = self.palette = self.mapper = None   # palette: the cover's colour
        self._aspect = 1.0          # aspect the fullscreen shape was built for
        self._resize_seen = None    # (aspect, time) of a pending resize

    def _fullscreen_needs_rebuild(self) -> bool:
        """The fullscreen shape must match the window; rebuild once a resize settles."""
        if self.settings.shape != FULLSCREEN or self.applied is None:
            return False
        w, h = self.window.size()
        if w <= 0 or h <= 0:
            return False
        aspect = w / h
        if abs(aspect - self._aspect) / self._aspect < 0.01:
            self._resize_seen = None
            return False
        now = time.perf_counter()
        if self._resize_seen is None or abs(self._resize_seen[0] - aspect) > 1e-3:
            self._resize_seen = (aspect, now)   # still dragging: wait for it to settle
            return False
        return now - self._resize_seen[1] > 0.25

    def poll_now_playing(self, panel=None) -> None:
        """Apply cover-art changes from the watcher (called once per frame)."""
        if self.watcher is None:
            return
        upd = self.watcher.take()
        if upd is None:
            return
        if panel:
            panel.show_track(upd.track)
        if upd.palette is not None:
            # Fade between covers; the first cover after launch just appears
            # (fading in from the placeholder palette would show a muddy mix).
            fade = self.PALETTE_FADE_S if getattr(self, "_had_cover", False) else 0.0
            self._had_cover = True
            self.palette = upd.palette
            self.window.renderer.set_palette(upd.palette, fade_s=fade)
            if panel:
                panel.show_palette(upd.palette)
            if upd.track:
                print(f"now playing: {upd.track.label()} -> {upd.palette.describe()}")

    def sync(self, panel=None) -> None:
        s, old = self.settings, self.applied
        rebuild_shape = (old is None or s.shape != old.shape
                         or self._fullscreen_needs_rebuild())
        if self.palette is None:
            self.palette = (self.watcher.palette if self.watcher else None) or default_palette()
            if panel:
                panel.show_palette(self.palette)
        if rebuild_shape:
            w, h = self.window.size()
            self._aspect = w / h if w > 0 and h > 0 else 1.0
            self._resize_seen = None
            try:
                self.shape = load_shape(s.shape, self._aspect)
            except Exception as e:
                if old is None:
                    raise
                print(f"could not load shape {s.shape!r}: {e}")
                s.shape = old.shape
                return
            baked = bake(self.shape, BAKE_RESOLUTION)
            self.window.set_shape(baked, self.palette, s.background, s.frame, s.idle_glow)
            prev = self.mapper
            self.mapper = AudioMapper(self.shape, Reactivity(s.energy), edge_margin=s.edge_margin)
            if prev is not None:
                self.mapper.state.time = prev.state.time
        elif (s.background, s.frame, s.idle_glow) != (old.background, old.frame, old.idle_glow):
            self.window.renderer.set_colors(s.background, s.frame, s.idle_glow)
        self.mapper.reactivity.energy = s.energy
        self.mapper.state.balance = s.balance
        self.mapper.state.branching = s.branching
        self.mapper.waves = s.waves
        self.mapper.size = s.size
        if old is None or rebuild_shape or s.edge_margin != old.edge_margin:
            self.mapper.set_edge_margin(s.edge_margin)
        self.applied = s.copy()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="bloom", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Defaults are None: a flag left out keeps the saved value (or the default).
    ap.add_argument("--shape", help='"circle", "fullscreen", or an SVG file')
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--wav", help="analyse a WAV file instead of live audio")
    src.add_argument("--demo", action="store_true", help="use a synthetic drum groove")
    ap.add_argument("--device", help="loopback device name or index")
    ap.add_argument("--energy", type=float,
                    help="0 = low reactance (gradual, with momentum), 1 = high (immediate)")
    ap.add_argument("--edge-fade", type=float, dest="edge_margin",
                    help="edge fade as a fraction of the shape's depth (0 = hard edge)")
    ap.add_argument("--balance", type=float, help="0 = mostly low intensity, 1 = mostly high")
    ap.add_argument("--blob-size", type=float, dest="size",
                    help="blob size: 0 = small .. 1.3 = largest")
    ap.add_argument("--branching", type=float,
                    help="how often noise pulses swell out of the centre: 0 = never, 1 = often")
    ap.add_argument("--waves", type=float, help="snare waves: 0 = off, 1 = strong")
    ap.add_argument("--idle-glow", type=float,
                    help="idle screen shows the palette colour at this level (0 = off)")
    ap.add_argument("--size", type=int, nargs=2, default=(720, 720), dest="window_size",
                    metavar=("W", "H"), help="window size in pixels")
    ap.add_argument("--settings", type=Path, help="settings file (default: per-user config)")
    ap.add_argument("--reset-settings", action="store_true",
                    help="ignore the saved settings and start from the defaults")
    ap.add_argument("--no-panel", action="store_true")
    args = ap.parse_args(argv)

    settings_path = args.settings or default_path()
    on_disk = load(settings_path)
    settings = None if args.reset_settings else on_disk
    print(f"settings: {settings_path}" + (" (restored)" if settings else " (defaults)"))
    settings = settings.copy() if settings else Settings()
    for key in ("shape", "energy", "edge_margin", "balance", "idle_glow", "branching",
                "waves", "size"):
        value = getattr(args, key)
        if value is not None:
            setattr(settings, key, value)
    if settings.shape not in (CIRCLE, FULLSCREEN) and not Path(settings.shape).is_file():
        print(f"shape {settings.shape!r} not found; using the circle")
        settings.shape = CIRCLE
    saver = SettingsSaver(settings, settings_path, saved=on_disk)

    watcher = None
    try:
        provider = create_provider()
        watcher = NowPlayingWatcher(provider)
        watcher.start()
        np_status = f"now playing: via {provider.name}"
    except ProviderUnavailable as e:
        np_status = f"now playing unavailable ({e}); using default colours"
    print(np_status)

    panel = None
    if not args.no_panel:
        from .panel import ControlPanel  # tkinter: only loaded when the panel is shown

        panel = ControlPanel(settings)
        panel.show_track(None, np_status)
    window = DisplayWindow(tuple(args.window_size))
    bus = FeatureBus()
    analyzer = AnalyzerThread(make_source(args), bus)
    analyzer.start()
    app = App(settings, window, watcher)
    app.sync(panel)
    last = time.perf_counter()
    n = 0
    try:
        while not window.should_close():
            if panel is not None and not panel.pump():
                break
            app.sync(panel)
            app.poll_now_playing(panel)
            now = time.perf_counter()
            dt, last = min(now - last, 0.1), now
            st = app.mapper.update(bus.consume(), dt)
            window.draw(st)
            n += 1
            if panel is not None and n % 3 == 0:
                panel.show_input(app.mapper.meters())
                panel.show_device(analyzer.device_name)
            saver.poll()
    finally:
        saver.flush()
        analyzer.stop()
        if watcher is not None:
            watcher.stop()
        if panel is not None:
            panel.destroy()
        window.close()


if __name__ == "__main__":
    main()
