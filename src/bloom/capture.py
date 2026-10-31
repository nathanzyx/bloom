"""
    capture.py
    
    Capture is where samples come from, and the thread that analyses them.

    Every source pushes float32 frames of shape (n, channels) into a RingBuffer.
    The analyzer never talks to a device directly, so the analysis code is
    identical for live loopback capture and for WAV files (which is what makes the
    detectors testable).

    Threading model:
    audio callback  --(RingBuffer)-->  analyzer thread  --(FeatureBus)-->  render thread
"""

from __future__ import annotations

import copy
import sys
import threading
import time
import wave
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from .analysis import Analyzer, DetectorConfig, ElementFrame

# ----------------------------------------------------------- ring buffer --
#
# Single-producer / single-consumer ring buffer for audio samples.
#
# The audio callback (producer) must never block or allocate, because a stalled
# callback causes dropouts. So the producer only copies into a preallocated
# array and bumps a counter; the consumer (analysis thread) reads the most
# recent N samples. Python's GIL makes the integer write atomic, and the
# consumer tolerates reading a frame that is being overwritten at the very tail,
# because it only ever reads behind the write head.

class RingBuffer:
    def __init__(self, capacity: int, channels: int) -> None:
        self._buf = np.zeros((capacity, channels), dtype=np.float32)
        self._capacity = capacity
        self._written = 0  # total frames ever written (monotonic)

    @property
    def channels(self) -> int:
        return self._buf.shape[1]

    @property
    def written(self) -> int:
        return self._written

    def write(self, frames: np.ndarray) -> None:
        """Append frames of shape (n, channels). Oldest data is overwritten."""
        n = frames.shape[0]
        if n >= self._capacity:
            frames = frames[-self._capacity :]
            n = self._capacity
        start = self._written % self._capacity
        end = start + n
        if end <= self._capacity:
            self._buf[start:end] = frames
        else:
            split = self._capacity - start
            self._buf[start:] = frames[:split]
            self._buf[: n - split] = frames[split:]
        self._written += n

    def read_window(self, end_frame: int, length: int) -> np.ndarray:
        """Return `length` frames ending at absolute frame index `end_frame`.

        Frames before the start of the stream are returned as zeros.
        """
        if length > self._capacity: raise ValueError("window longer than buffer capacity")
        if end_frame > self._written: raise ValueError("cannot read frames that have not been written")
        out = np.zeros((length, self.channels), dtype=np.float32)
        start_frame = end_frame - length
        valid_from = max(start_frame, 0, self._written - self._capacity)
        if valid_from >= end_frame: return out
        offset = valid_from - start_frame
        idx = np.arange(valid_from, end_frame) % self._capacity
        out[offset:] = self._buf[idx]
        return out


# --------------------------------------------------------------- sources --

class AudioSource(ABC):
    sample_rate: int
    channels: int

    def __init__(self) -> None:
        self.buffer: RingBuffer | None = None

    def attach(self, buffer_seconds: float = 2.0) -> RingBuffer:
        self.buffer = RingBuffer(int(self.sample_rate * buffer_seconds), self.channels)
        return self.buffer

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...


def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Load a PCM WAV (8/16/24/32-bit int) as float32 (n, channels) in [-1, 1]."""
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        width = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
        
    if width == 1: data = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2: data = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    elif width == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v >= 1 << 23, v - (1 << 24), v)
        data = v.astype(np.float32) / float(1 << 23)
    elif width == 4: data = np.frombuffer(raw, "<i4").astype(np.float32) / float(1 << 31)
    else: raise ValueError(f"unsupported sample width: {width}")
    return data.reshape(-1, ch), sr


def save_wav(path: str | Path, data: np.ndarray, sample_rate: int) -> None:
    """Write float data (n,) or (n, channels) as 16-bit PCM."""
    if data.ndim == 1: data = data[:, None]
    pcm = (np.clip(data, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(data.shape[1])
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


class FileSource(AudioSource):
    """Plays a WAV file into the ring buffer in real time (or as fast as asked).

    It does not output sound. Play the same file in a media player if you want
    to hear it; the point is a reproducible input for tuning and tests.
    """

    def __init__(self, path: str | Path, chunk: int = 480, realtime: bool = True, loop: bool = True) -> None:
        super().__init__()
        self.data, self.sample_rate = load_wav(path)
        self.device_name = f"{Path(path).name} (file)"
        self.channels = self.data.shape[1]
        self.chunk = chunk
        self.realtime = realtime
        self.loop = loop
        self._thread: threading.Thread | None = None
        self._running = threading.Event()

    def start(self) -> None:
        if self.buffer is None: self.attach()
        self._running.set()
        self._thread = threading.Thread(target=self._run, name="file-source", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        assert self.buffer is not None
        pos = 0
        period = self.chunk / self.sample_rate
        next_t = time.perf_counter()
        while self._running.is_set():
            block = self.data[pos : pos + self.chunk]
            pos += self.chunk
            if pos >= len(self.data):
                if not self.loop:
                    self.buffer.write(block)
                    break
                pos = 0
            self.buffer.write(block)
            if self.realtime:
                next_t += period
                delay = next_t - time.perf_counter()
                if delay > 0: time.sleep(delay)

    def stop(self) -> None:
        self._running.clear()
        if self._thread is not None: self._thread.join(timeout=1.0)


class LoopbackSource(AudioSource):
    """
    Captures whatever the system is playing, following the default output.

    WASAPI loopback via PyAudioWPatch (the stock PortAudio builds don't expose
    loopback devices on Windows). Windows only.

    Following the default device (when no `device` is given):
    loopback capture is tied to one output device. When Windows switches the
    default output (headphones plugged in, Bluetooth connects), the old
    device's loopback simply goes quiet. PortAudio can't tell us, because it
    enumerates devices once at initialisation and caches the list. So a
    watcher thread waits for the input to go quiet, then closes the stream,
    fully re-initialises PortAudio (which re-enumerates devices), and reopens
    on whatever the default output is now. Probing only during silence means
    music that is actually playing is never interrupted; the cost is a short
    gap (about `probe_after_s`) after a switch.

    A device switch can change the sample rate and channel count, so each
    (re)open that changes them creates a new ring buffer and bumps
    `generation`. AnalyzerThread watches `generation` and rebuilds itself.
    """

    def __init__(self, device: str | int | None = None, chunk: int = 480, probe_after_s: float = 1.0) -> None:
        super().__init__()
        if sys.platform != "win32": raise RuntimeError("live capture is only implemented on Windows; use --wav or --demo")
        self.device = device
        self.chunk = chunk
        self.follow_default = device is None
        self.probe_after_s = probe_after_s
        self.generation = 0
        self.device_name = "?"
        self._stream = None
        self._pa = None
        self._info: dict | None = None
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._watcher: threading.Thread | None = None
        self._last_audio = time.perf_counter()
        self._init_wasapi()

    def _wasapi(self):
        try: import pyaudiowpatch as pyaudio
        except ImportError as e: raise RuntimeError("install the 'audio' extra: pip install -e .[audio]") from e
        return pyaudio

    def _init_wasapi(self) -> None:
        pyaudio = self._wasapi()
        self._pa = pyaudio.PyAudio()
        if self.device is None: info = self._pa.get_default_wasapi_loopback()
        else:
            info = next(
                d for d in self._pa.get_loopback_device_info_generator()
                if self.device in (d["index"], d["name"])
            )
        self._info = info
        self.sample_rate = int(info["defaultSampleRate"])
        self.channels = int(info["maxInputChannels"])
        self.device_name = str(info["name"]).replace(" [Loopback]", "")

    def _open_wasapi_stream(self) -> None:
        pyaudio = self._wasapi()
        buf, ch = self.buffer, self.channels
        assert buf is not None and self._info is not None

        def callback(in_data, frame_count, time_info, status):
            block = np.frombuffer(in_data, np.float32).reshape(-1, ch)
            buf.write(block)
            if block.size and float(np.abs(block).max()) > 1e-6: self._last_audio = time.perf_counter()
            return (None, pyaudio.paContinue)

        self._stream = self._pa.open(
            format=pyaudio.paFloat32, channels=ch, rate=self.sample_rate,
            input=True, input_device_index=self._info["index"],
            frames_per_buffer=self.chunk, stream_callback=callback,
        )
        self._stream.start_stream()

    def _close_wasapi(self) -> None:
        # The device may already be gone (headphones unplugged), in which case
        # closing can raise; the stream is dead either way.
        for fn in (lambda: self._stream.stop_stream(), lambda: self._stream.close()):
            if self._stream is not None:
                try: fn()
                except Exception: pass
        self._stream = None
        if self._pa is not None:
            try: self._pa.terminate()
            except Exception: pass
            self._pa = None

    def _reopen_default(self) -> bool:
        """Re-enumerate devices and reopen on the current default. True if it changed."""
        with self._lock:
            old = (self._info or {}).get("name"), self.sample_rate, self.channels
            self._close_wasapi()
            try:
                self._init_wasapi()
            except Exception as e: # e.g. no output device at all right now
                print(f"audio: no default output device ({e}); retrying")
                self._info = None
                return False
            new = self._info["name"], self.sample_rate, self.channels
            changed = new != old
            if (self.sample_rate, self.channels) != old[1:]:
                self.attach() # new rate/layout: fresh buffer
                self.generation += 1
            self._open_wasapi_stream()
        if changed:
            print(f"audio: now listening to {self.device_name}")
        return changed

    def _watch(self) -> None:
        last_probe = 0.0
        while self._running.is_set():
            time.sleep(0.2)
            now = time.perf_counter()
            quiet = now - self._last_audio
            if quiet >= self.probe_after_s and now - last_probe >= self.probe_after_s:
                last_probe = now
                self._reopen_default()

    def start(self) -> None:
        if self.buffer is None: self.attach()
        self._running.set()
        with self._lock:
            self._open_wasapi_stream()
        print(f"audio: listening to {self.device_name}")
        if self.follow_default:
            self._watcher = threading.Thread(target=self._watch, name="audio-device-watch", daemon=True)
            self._watcher.start()

    def stop(self) -> None:
        self._running.clear()
        if self._watcher is not None: self._watcher.join(timeout=2.0)
        with self._lock: self._close_wasapi()


# -------------------------------------------------------------- analysis --

_EVENT_FIELDS = ("kick", "snare", "hat")


class FeatureBus:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest = ElementFrame()
        self._events = dict.fromkeys(_EVENT_FIELDS, 0.0)

    def publish(self, frame: ElementFrame) -> None:
        with self._lock:
            self._latest = frame
            for k in _EVENT_FIELDS: self._events[k] = max(self._events[k], getattr(frame, k))

    def consume(self) -> ElementFrame:
        """Latest frame, with onset fields replaced by the max since the last consume."""
        with self._lock:
            out = copy.copy(self._latest)
            for k in _EVENT_FIELDS:
                setattr(out, k, self._events[k])
                self._events[k] = 0.0
        return out


class AnalyzerThread:
    """
    Feeds the analyzer from a source's ring buffer, one hop at a time.

    Two situations are handled here rather than in the sources:

    * No data arriving. WASAPI loopback delivers *nothing* (not zeros) while
      nothing is playing, and nothing during a device switch. Waiting for data
      would freeze the display on the last loud frame. So when the buffer
      stalls for longer than `stall_s`, the thread analyses silent windows at
      the normal hop rate. The detectors then decay exactly as they would on
      real silence, and the display fades to background.
    * The source switching devices (`generation` changes), possibly to a new
      sample rate. The analyzer is rebuilt for the new rate and reads the new
      buffer from its current position.
    """

    def __init__(self, source: AudioSource, bus: FeatureBus, detector_cfg: DetectorConfig | None = None, stall_s: float = 0.05) -> None:
        self.source = source
        self.bus = bus
        self.detector_cfg = detector_cfg
        self.stall_s = stall_s
        self.analyzer = Analyzer(source.sample_rate, detector_cfg)
        self._buf: RingBuffer = source.buffer or source.attach()
        self._gen = getattr(source, "generation", 0)
        self._running = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def device_name(self) -> str:
        return getattr(self.source, "device_name", "file")

    def start(self) -> None:
        self._running.set()
        self.source.start()
        self._thread = threading.Thread(target=self._run, name="analyzer", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        cfg = self.analyzer.stft_cfg
        next_end = cfg.n_fft
        silence = np.zeros((cfg.n_fft, self._buf.channels), np.float32)
        stalled_since = None
        next_silent = 0.0
        while self._running.is_set():
            gen = getattr(self.source, "generation", 0)
            if gen != self._gen and self.source.buffer is not None:
                self._gen = gen
                self._buf = self.source.buffer
                self.analyzer = Analyzer(self.source.sample_rate, self.detector_cfg)
                cfg = self.analyzer.stft_cfg
                silence = np.zeros((cfg.n_fft, self._buf.channels), np.float32)
                next_end = max(self._buf.written, cfg.n_fft)

            if self._buf.written < next_end:
                now = time.perf_counter()
                stalled_since = stalled_since or now
                if now - stalled_since >= self.stall_s and now >= next_silent:
                    self.bus.publish(self.analyzer.process_window(silence))
                    next_silent = now + cfg.frame_dt
                time.sleep(0.002)
                continue
            if stalled_since is not None:
                # data resumed after a gap: start from the live edge
                stalled_since = None
                next_end = max(next_end, self._buf.written - cfg.hop)
            # If we fell behind (e.g. GC pause) skip ahead rather than lagging forever.
            if self._buf.written - next_end > 4 * cfg.hop: next_end = self._buf.written
            window = self._buf.read_window(next_end, cfg.n_fft)
            self.bus.publish(self.analyzer.process_window(window))
            next_end += cfg.hop

    def stop(self) -> None:
        self._running.clear()
        self.source.stop()
        if self._thread is not None: self._thread.join(timeout=1.0)


# ------------------------------------------------------------ demo groove --
#
# Synthetic drum/tonal signals with known ground truth.
#
# Real recordings are better for tuning, but synthetic hits let the tests assert
# "a kick happened at exactly 0.5 s" without shipping copyrighted audio.

SR = 48_000


def _env(n: int, decay_s: float, sr: int = SR) -> np.ndarray:
    t = np.arange(n) / sr
    return np.exp(-t / decay_s)


def kick(sr: int = SR, rng: np.random.Generator | None = None) -> np.ndarray:
    n = int(0.35 * sr)
    t = np.arange(n) / sr
    f = 50.0 + 100.0 * np.exp(-t / 0.03)  # pitch sweep 150 -> 50 Hz
    phase = 2 * np.pi * np.cumsum(f) / sr
    return 0.9 * np.sin(phase) * _env(n, 0.12, sr)


def snare(sr: int = SR, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng(1)
    n = int(0.25 * sr)
    t = np.arange(n) / sr
    body = np.sin(2 * np.pi * 200.0 * t) * _env(n, 0.05, sr)
    noise = rng.standard_normal(n)
    # crude band emphasis around 1-5 kHz: difference of two moving averages
    lp_short = np.convolve(noise, np.ones(4) / 4, "same")
    lp_long = np.convolve(noise, np.ones(24) / 24, "same")
    crack = (lp_short - lp_long) * _env(n, 0.07, sr)
    return 0.5 * body + 0.9 * crack


def hat(sr: int = SR, rng: np.random.Generator | None = None, decay_s: float = 0.03) -> np.ndarray:
    rng = rng or np.random.default_rng(2)
    n = int(max(0.12, decay_s * 6) * sr)
    noise = rng.standard_normal(n)
    hp = noise - np.convolve(noise, np.ones(3) / 3, "same")  # crude high-pass
    return 0.5 * hp * _env(n, decay_s, sr)


def tone(freqs: list[float], seconds: float, sr: int = SR, amp: float = 0.2) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    out = sum(np.sin(2 * np.pi * f * t) for f in freqs) / max(1, len(freqs))
    fade = np.minimum(1.0, np.minimum(t, t[-1] - t) / 0.02)
    return amp * out * fade


def place(length_s: float, events: list[tuple[float, np.ndarray]], sr: int = SR) -> np.ndarray:
    out = np.zeros(int(length_s * sr), np.float32)
    for at, sig in events:
        i = int(at * sr)
        j = min(len(out), i + len(sig))
        out[i:j] += sig[: j - i]
    return out


def stereo(mid: np.ndarray, side: np.ndarray | None = None) -> np.ndarray:
    side = np.zeros_like(mid) if side is None else side
    return np.stack([mid + side, mid - side], axis=1).astype(np.float32)


def demo_loop(bars: int = 2, bpm: float = 120.0, sr: int = SR) -> np.ndarray:
    """A small groove: kick on 1/3, snare on 2/4, 8th-note hats, a centred
    chord, and a wide pad. Returns stereo (n, 2)."""
    beat = 60.0 / bpm
    length = bars * 4 * beat
    rng = np.random.default_rng(0)
    ev_mid = []
    for b in range(bars * 4):
        t = b * beat
        if b % 2 == 0: ev_mid.append((t, kick(sr)))
        else: ev_mid.append((t, snare(sr, rng)))
        ev_mid.append((t, hat(sr, rng)))
        ev_mid.append((t + beat / 2, hat(sr, rng)))
    mid = place(length, ev_mid, sr)
    mid[: int(length * sr)] += tone([261.6, 329.6, 392.0], length, sr, 0.08)[: len(mid)]
    pad = tone([523.3, 659.3], length, sr, 0.05)[: len(mid)]
    return stereo(mid, pad)
