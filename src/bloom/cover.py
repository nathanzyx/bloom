"""
    cover.py
    
    Cover art for whatever is currently playing, as retreived by the OS.

    A single color is used from the cover for the visual effects.
    
    For platforms with no provider, that app color reamins it's default.

    Album art -> one colour. Rules (in order):

    1. Downsample to ~256 px. Large enough that thin, bright details (text,
        a logo) survive; averaging to a tiny thumbnail would blend them away.
    2. Split pixels into NEUTRAL (low OKLab chroma: blacks, whites, greys) and
        CHROMATIC. k-means the chromatic pixels and merge clusters that share a
        hue. Clusters that differ only in lightness are one colour here, because
        the display drives lightness from intensity anyway.
    3. The largest MAJOR colour (a chromatic cluster covering at least
        `min_share` of the image) is the colour. Grey never wins over colour:
        on an emissive display it reads as dirty (Brat -> lime, not black).
    4. No major colour: a vivid detail is the colour (a red logo on a black
        cover), but only if there are at most `max_rare` such details. One or
        two read as deliberate; many (confetti) read as noise.
    5. Otherwise the cover is greyscale.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .colour import Palette, Swatch, make_palette, srgb_to_oklab

# ------------------------------------------------------------ colour --

@dataclass
class PaletteConfig:
    sample_size: int = 256
    neutral_chroma: float = 0.04 # OKLab chroma; vivid colours are ~0.1-0.3
    min_share: float = 0.06 # of the whole image, for a MAJOR colour
    max_clusters: int = 8
    merge_distance: float = 0.08 # in hue-weighted OKLab (see lightness_weight)
    # Lightness barely counts in distances: a colour in light and in shadow
    # (Ye's dark sky and pale sky) is still one colour.
    lightness_weight: float = 0.15
    salient_chroma: float = 0.10 # how vivid a detail must be (rule 4)
    rare_min_share: float = 0.002 # ignore specks below this (JPEG noise)
    max_rare: int = 2
    hue_merge_deg: float = 15.0 # same hue (any chroma/lightness) = same colour
    kmeans_iters: int = 25
    seed: int = 0


def _weighted(lab: np.ndarray, wl: float) -> np.ndarray:
    return lab * np.array([wl, 1.0, 1.0])


def kmeans(x: np.ndarray, k: int, iters: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Plain Lloyd's k-means with k-means++ seeding. Deterministic for a seed."""
    rng = np.random.default_rng(seed)
    k = min(k, len(x))
    centers = [x[rng.integers(len(x))]]
    
    for _ in range(1, k):
        d2 = np.min(((x[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
        if d2.sum() <= 0: break
        centers.append(x[rng.choice(len(x), p=d2 / d2.sum())])
        
    c = np.array(centers)
    labels = np.zeros(len(x), dtype=int)
    
    for _ in range(iters):
        labels = np.argmin(((x[:, None, :] - c[None]) ** 2).sum(-1), axis=1)
        new = np.array([x[labels == j].mean(0) if np.any(labels == j) else c[j] for j in range(len(c))])
        if np.allclose(new, c): break
        c = new
        
    return c, labels


def _merge(swatches: list[Swatch], dist: float, wl: float, hue_deg: float = 15.0) -> list[Swatch]:
    """Repeatedly merge the closest pair that is near in colour OR in hue."""
    sw = list(swatches)
    
    while len(sw) > 1:
        lab = np.array([s.lab for s in sw])
        labs = _weighted(lab, wl)
        d = np.sqrt(((labs[:, None] - labs[None]) ** 2).sum(-1))
        h = np.arctan2(lab[:, 2], lab[:, 1])
        dh = np.degrees(np.abs((h[:, None] - h[None] + np.pi) % (2 * np.pi) - np.pi))
        cand = np.where((d < dist) | (dh < hue_deg), d, np.inf)
        np.fill_diagonal(cand, np.inf)
        i, j = np.unravel_index(np.argmin(cand), cand.shape)
        if not np.isfinite(cand[i, j]): break
        
        a, b = sw[i], sw[j]
        share = a.share + b.share
        merged = Swatch((a.lab * a.share + b.lab * b.share) / share, share, False)
        sw = [s for n, s in enumerate(sw) if n not in (i, j)] + [merged]
    return sw


def _cluster(pixels: np.ndarray, n_total: int, cfg: PaletteConfig, vivid: bool) -> list[Swatch]:
    if len(pixels) == 0: return []
    centers, labels = kmeans(_weighted(pixels, cfg.lightness_weight), cfg.max_clusters, cfg.kmeans_iters, cfg.seed)
    out = []
    for j in range(len(centers)):
        members = pixels[labels == j]
        if not len(members): continue
        if vivid:
            # Thin details are anti-aliased against their surroundings; the
            # most saturated half of the cluster is the colour actually drawn.
            c = np.hypot(members[:, 1], members[:, 2])
            members = members[c >= np.median(c)]
            
        out.append(Swatch(members.mean(0), (labels == j).sum() / n_total, False))
    return _merge(out, cfg.merge_distance, cfg.lightness_weight, cfg.hue_merge_deg)


def extract_from_pixels(srgb: np.ndarray, cfg: PaletteConfig | None = None) -> Palette:
    """srgb: (..., 3) floats in [0, 1]."""
    cfg = cfg or PaletteConfig()
    lab = srgb_to_oklab(srgb.reshape(-1, 3))
    n = len(lab)
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    neutral = chroma < cfg.neutral_chroma

    majors = [c for c in _cluster(lab[~neutral], n, cfg, vivid=False) if c.share >= cfg.min_share]
    if majors: return make_palette(max(majors, key=lambda s: s.share))
    vivid = ~neutral & (chroma >= cfg.salient_chroma)
    rare = [r for r in _cluster(lab[vivid], n, cfg, vivid=True) if r.share >= cfg.rare_min_share]
    if 1 <= len(rare) <= cfg.max_rare: return make_palette(max(rare, key=lambda s: s.share))
    if neutral.any(): return make_palette(Swatch(lab[neutral].mean(0), float(neutral.mean()), True))
    return make_palette(Swatch(lab.mean(0), 1.0, True)) # only faint colour specks


def extract_from_image(path: str | Path, cfg: PaletteConfig | None = None) -> Palette:
    from PIL import Image

    cfg = cfg or PaletteConfig()
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((cfg.sample_size, cfg.sample_size), Image.Resampling.BOX)
        px = np.asarray(im, dtype=np.float64) / 255.0
    return extract_from_pixels(px, cfg)


# ------------------------------------------------------------- tracks --

@dataclass(frozen=True)
class Track:
    title: str = ""
    artist: str = ""
    app: str = ""
    playing: bool = False
    art: bytes | None = None # cover image given by OS

    @property
    def art_hash(self) -> str | None:
        return hashlib.sha1(self.art).hexdigest() if self.art else None

    def label(self) -> str:
        if not (self.title or self.artist): return "(untitled)"
        return " — ".join(x for x in (self.title, self.artist) if x)


class ProviderUnavailable(RuntimeError):
    """The platform or its dependencies can't supply now-playing info."""


class Provider(ABC):
    name = "provider"

    @abstractmethod
    def poll(self) -> Track | None:
        """
        Blocking query of the current track. None = nothing playing / no session.

        Called from a background thread, roughly twice a second. May raise;
        the watcher logs and carries on.
        """

    def close(self) -> None:  # noqa: B027 - optional hook
        pass


# ------------------------------------------------------ Windows (SMTC) --
#
# Uses the pywinrt projection (`pip install -e .[nowplaying]`). The API is
# async; the provider owns a private asyncio loop and runs each poll to
# completion on it, so callers just see a blocking `poll()`.

class SmtcProvider(Provider):
    name = "Windows media controls"

    def __init__(self) -> None:
        try:
            from winrt.windows.media.control import (
                GlobalSystemMediaTransportControlsSessionManager as Manager,
            )
            from winrt.windows.media.control import (
                GlobalSystemMediaTransportControlsSessionPlaybackStatus as Status,
            )
            from winrt.windows.storage.streams import Buffer, InputStreamOptions
        except ImportError as e:
            raise ProviderUnavailable("install the now-playing extra: pip install -e .[nowplaying]") from e
        self._Manager, self._Status = Manager, Status
        self._Buffer, self._ReadAhead = Buffer, InputStreamOptions.READ_AHEAD
        self._loop = asyncio.new_event_loop()
        self._manager = None

    def poll(self) -> Track | None:
        return self._loop.run_until_complete(self._poll())

    async def _poll(self) -> Track | None:
        if self._manager is None: self._manager = await self._Manager.request_async()
        # Prefer a session that is actually playing; Windows' "current" session
        # is the one last interacted with, which can be a paused app.
        session = None
        for s in self._manager.get_sessions():
            info = s.get_playback_info()
            if info is not None and info.playback_status == self._Status.PLAYING:
                session = s
                break
        if session is None: session = self._manager.get_current_session()
        if session is None: return None
        props = await session.try_get_media_properties_async()
        info = session.get_playback_info()
        playing = info is not None and info.playback_status == self._Status.PLAYING
        title, artist = props.title or "", props.artist or ""
        app = session.source_app_user_model_id or ""
        art = None
        if props.thumbnail is not None: art = await self._read_thumbnail(props.thumbnail)
        return Track(title=title, artist=artist, app=app, playing=playing, art=art)

    async def _read_thumbnail(self, ref) -> bytes | None:
        stream = await ref.open_read_async()
        size = int(stream.size)
        if size <= 0: return None
        buf = self._Buffer(size)
        result = await stream.read_async(buf, size, self._ReadAhead)
        try: return bytes(memoryview(result))       # pywinrt 2.x: IBuffer is a buffer
        except TypeError:
            from winrt.windows.storage.streams import DataReader

            reader = DataReader.from_buffer(result)
            data = bytearray(result.length)
            reader.read_bytes(data)
            return bytes(data)

    def close(self) -> None:
        self._loop.close()

def create_provider() -> Provider:
    """The provider for this OS. Raises ProviderUnavailable with a reason."""
    if sys.platform != "win32": raise ProviderUnavailable("now playing is only implemented on Windows")
    return SmtcProvider()


# ------------------------------------------------------------- watcher --
#
# Turns a stream of Track polls into palette changes.
#
# The rule is: **the palette follows the artwork, not the title.**
#
#   * Apps often publish the new title a moment before the new art. If we
#     reacted to the title we'd briefly re-extract the *previous* song's cover.
#     Keying on the art itself (its hash) sidesteps that: nothing happens until
#     different art actually arrives.
#   * Consecutive tracks from one album share art, so their palette rightly
#     doesn't change.
#   * Missing art (an app that publishes none, or a gap between tracks) keeps
#     the current palette instead of flashing grey.
#   * New art must be seen on two consecutive polls before it's used, which
#     filters out placeholder images some apps show for a split second.
#
# Extraction runs on this background thread; the app collects the result with
# `take()` once per frame, so the render loop never waits on I/O.

@dataclass
class Update:
    track: Track | None
    palette: Palette | None # None = keep the current palette


def palette_from_bytes(data: bytes) -> Palette:
    from PIL import Image

    cfg = PaletteConfig()
    with Image.open(io.BytesIO(data)) as im:
        im = im.convert("RGB")
        im.thumbnail((cfg.sample_size, cfg.sample_size), Image.Resampling.BOX)
        px = np.asarray(im, dtype=np.float64) / 255.0
    return extract_from_pixels(px, cfg)


class NowPlayingWatcher:
    def __init__(self, provider: Provider, interval_s: float = 0.5) -> None:
        self.provider = provider
        self.interval_s = interval_s
        self._lock = threading.Lock()
        self._pending: Update | None = None
        self.track: Track | None = None
        self.palette: Palette | None = None # palette of the art currently applied
        self.error: str | None = None
        self._applied_hash: str | None = None
        self._candidate: str | None = None
        self._running = threading.Event()
        self._thread: threading.Thread | None = None

    # ---------------------------------------------------------- the logic --
    def step(self, track: Track | None) -> Update | None:
        """
        Feed one poll result; returns an Update when something changed.

        Pure logic (no threads, no I/O except palette extraction), so tests can
        drive it with scripted sequences of tracks.
        """
        changed_track = track != self.track and (
            track is None or self.track is None
            or (track.title, track.artist, track.app, track.playing)
            != (self.track.title, self.track.artist, self.track.app, self.track.playing))
        self.track = track
        h = track.art_hash if track else None
        new_palette = None
        if h is None or h == self._applied_hash: self._candidate = None
        elif h != self._candidate: self._candidate = h # seen once: wait for the next poll
        else:
            try:
                new_palette = palette_from_bytes(track.art)
                self._applied_hash = h
                self.palette = new_palette
            except Exception as e:         # corrupt/unsupported image: ignore it
                self.error = f"could not read cover art: {e}"
                self._applied_hash = h
            self._candidate = None
            
        if changed_track or new_palette is not None: return Update(track, new_palette)
        return None

    # ---------------------------------------------------------- threading --
    def start(self) -> None:
        self._running.set()
        self._thread = threading.Thread(target=self._run, name="now-playing", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        last_error = None
        while self._running.is_set():
            t0 = time.perf_counter()
            try:
                upd = self.step(self.provider.poll())
                self.error = None
            except Exception as e:  # keep going; sessions come and go
                upd = None
                self.error = f"{type(e).__name__}: {e}"
                if self.error != last_error:
                    print(f"now playing: {self.error}")
                last_error = self.error
            if upd is not None:
                with self._lock:
                    if self._pending is not None and upd.palette is None: upd.palette = self._pending.palette  # don't drop an unread palette
                    self._pending = upd
            time.sleep(max(0.0, self.interval_s - (time.perf_counter() - t0)))

    def take(self) -> Update | None:
        with self._lock: upd, self._pending = self._pending, None
        return upd

    def stop(self) -> None:
        self._running.clear()
        if self._thread is not None: self._thread.join(timeout=3.0)
        self.provider.close()
