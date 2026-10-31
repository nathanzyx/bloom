"""
    analysis.py
    
    Audio analyses; a stereo window in and music elements out

    Splitting a mix into the musical elements that DSP can actually separate.

    What is separable without machine learning, and how:

    Axis              Method                              Gives
    ----------------  ----------------------------------  ---------------------------------
    percussive/tonal  median-filter HPSS (StreamingHPSS)  drums vs sustained notes
    frequency         band energies on each HPSS stream   kick vs snare vs hats; bass vs leads
    noisiness         spectral flatness                   hats/cymbals (noise) vs tonal highs
    time              decay / sustain of a band           hat ticks vs cymbal wash
    stereo            mid/side split (STFT)               centred lead vs wide pads/reverb

    Combining the axes gives these outputs (ElementFrame):

    kick        percussive, 40-120 Hz onset
    snare       percussive onset in BOTH 150-400 Hz (body) AND 1-5 kHz (crack).
                Using the geometric mean of the two bands means a kick (no crack)
                and a hat (no body) barely move it. Claps land here too.
    hat         percussive onset 6-16 kHz, noisy (high flatness), not a snare
    cymbal      sustained noisy 6-16 kHz energy: rides, crashes, shakers, wash
    bass        tonal 40-250 Hz
    body        tonal 250-2000 Hz in the centre channel: lead vocal, keys, guitar
    lead        tonal 2-8 kHz in the centre channel: brightness of leads/vocals
    wide        side-channel 250-8000 Hz: pads, stereo synths, reverb, doubled parts

    The pieces, in order: an STFT on mid and side, streaming harmonic/percussive
    separation of the mid channel, band energies on each stream, and adaptive
    onset detection. Each window is 2048 samples at a 10 ms hop.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

EPS = 1e-10


@dataclass(frozen=True)
class STFTConfig:
    sample_rate: int = 48_000
    n_fft: int = 2048   # 42.7 ms at 48 kHz: 23 Hz bins, enough to resolve kick vs bass
    hop: int = 480      # 10 ms: the feature update rate (100 Hz)

    @property
    def n_bins(self) -> int:
        return self.n_fft // 2 + 1

    @property
    def frame_dt(self) -> float:
        return self.hop / self.sample_rate

    def bin_freqs(self) -> np.ndarray:
        return np.fft.rfftfreq(self.n_fft, 1.0 / self.sample_rate)


class STFT:
    """
    Turns a window of stereo samples into mid and side power spectra.

    Mid = (L+R)/2 carries the centre-panned material (kick, bass, lead vocal);
    side = (L-R)/2 carries stereo-wide material (pads, reverb, doubled guitars).
    Splitting them is the one source separation that is exact rather than
    heuristic, so it is done up front.
    """

    def __init__(self, cfg: STFTConfig) -> None:
        self.cfg = cfg
        self.window = np.hanning(cfg.n_fft).astype(np.float32)
        # Normalise so a full-scale sine reads ~1.0 in power.
        self._scale = 2.0 / np.sum(self.window)

    def power(self, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """frames: (n_fft, channels). Returns (mid_power, side_power), each (n_bins,)."""
        if frames.shape[1] >= 2:
            mid = 0.5 * (frames[:, 0] + frames[:, 1])
            side = 0.5 * (frames[:, 0] - frames[:, 1])
        else:
            mid = frames[:, 0]
            side = np.zeros_like(mid)
        m = np.fft.rfft(mid * self.window) * self._scale
        s = np.fft.rfft(side * self.window) * self._scale
        return (m.real**2 + m.imag**2).astype(np.float32), (s.real**2 + s.imag**2).astype(
            np.float32
        )


def band_mask(freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    m = (freqs >= lo) & (freqs < hi)
    # very narrow band at low resolution: take the nearest bin 
    if not m.any(): m[np.argmin(np.abs(freqs - 0.5 * (lo + hi)))] = True
    return m


def flatness(power: np.ndarray) -> float:
    """Spectral flatness (geometric / arithmetic mean), 0 = pure tone, 1 = white noise."""
    p = power + EPS
    return float(np.exp(np.mean(np.log(p))) / np.mean(p))


# ------------------------------------------------------------------- HPSS --
#
# Streaming harmonic/percussive source separation (median filtering).
#
# Fitzgerald (2010): in a spectrogram, sustained tonal sounds are horizontal
# lines (stable in time) and drum hits are vertical lines (broadband, short).
#   * a median *across time* at each bin keeps horizontal lines -> harmonic
#   * a median *across frequency* in each frame keeps vertical lines -> percussive
# Soft (Wiener-style) masks then split the current frame's power between the two.
#
# Real time forces one change from the offline algorithm: the time median can
# only look backwards. The consequence is that the first ~half window of a new
# sustained note is classified as percussive before it "settles" into harmonic.
# For visuals this is acceptable, and arguably correct: a note's attack *is*
# percussive. With the defaults (17 frames at a 10 ms hop) that is ~85 ms.

class StreamingHPSS:
    def __init__(self, n_bins: int, time_kernel: int = 17, freq_kernel: int = 17, power: float = 2.0) -> None:
        if freq_kernel % 2 == 0: raise ValueError("freq_kernel must be odd")
        self.time_kernel = time_kernel
        self.freq_kernel = freq_kernel
        self.power = power
        self._hist = np.zeros((time_kernel, n_bins), dtype=np.float32)
        self._pos = 0
        self._filled = 0

    def process(self, mag: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """mag: magnitude spectrum (n_bins,). Returns (harmonic, percussive) magnitudes."""
        self._hist[self._pos] = mag
        self._pos = (self._pos + 1) % self.time_kernel
        self._filled = min(self._filled + 1, self.time_kernel)

        harm = np.median(self._hist[: self._filled], axis=0)
        pad = self.freq_kernel // 2
        padded = np.pad(mag, pad, mode="edge")
        perc = np.median(sliding_window_view(padded, self.freq_kernel), axis=1)

        h = harm**self.power
        p = perc**self.power
        denom = h + p + 1e-12
        mask_h = h / denom
        return mag * mask_h, mag * (1.0 - mask_h)


# -------------------------------------------------------------- envelopes --

def coeff(time_s: float, dt: float) -> float:
    """One-pole smoothing coefficient reaching ~63% of a step after `time_s`."""
    if time_s <= 0: return 0.0
    return math.exp(-dt / time_s)


class EnvelopeFollower:
    """
    Asymmetric one-pole follower: fast attack, slow release.

    This is the main "feel" control. A short attack makes hits land on time; a
    long release makes them decay gracefully instead of strobing. The
    reactivity knob mostly moves these two numbers.
    """

    def __init__(self, attack_s: float, release_s: float, dt: float) -> None:
        self.dt = dt
        self.value = 0.0
        self.set_times(attack_s, release_s)

    def set_times(self, attack_s: float, release_s: float) -> None:
        self._a = coeff(attack_s, self.dt)
        self._r = coeff(release_s, self.dt)

    def update(self, x: float) -> float:
        c = self._a if x > self.value else self._r
        self.value = c * self.value + (1.0 - c) * x
        return self.value


class PeakNormalizer:
    """
    Automatic gain control: divides by a slowly decaying running peak.

    Without this, a quiet acoustic track barely moves the display while a
    brickwalled master saturates it. The slow release (default 8 s) adapts to
    a song's overall level but not to individual hits, so dynamics within a
    song survive. `floor` stops silence from being amplified into noise.
    """

    def __init__(self, dt: float, release_s: float = 8.0, floor: float = 1e-4) -> None:
        self._r = coeff(release_s, dt)
        self.floor = floor
        self.peak = floor

    def update(self, x: float) -> float:
        self.peak = max(x, self.peak * self._r, self.floor)
        return min(x / self.peak, 1.0)


class OnsetTracker:
    """
    Adaptive-threshold onset detector on a positive-flux novelty signal.

    novelty = max(0, e[n] - e[n-1]) on a log-compressed band energy. An onset
    fires when novelty exceeds  mean + k * std  of its recent history (plus an
    absolute floor) and the refractory period has passed. Using the band's own
    statistics, rather than a fixed threshold, is what lets one detector work
    on both a sparse jazz kit and a dense EDM mix.
    """

    def __init__(self, dt: float, k: float = 2.0, history_s: float = 1.0,
                 refractory_s: float = 0.08, floor: float = 0.1) -> None:
        self.dt = dt
        self.k = k
        self.floor = floor
        self.refractory = max(1, int(round(refractory_s / dt)))
        self._hist: deque[float] = deque(maxlen=max(4, int(history_s / dt)))
        self._prev = 0.0
        self._since = self.refractory
        self.novelty = 0.0

    def update(self, energy_log: float) -> float:
        """Returns onset strength (0 when no onset, else >0, roughly 0..1)."""
        nov = max(0.0, energy_log - self._prev)
        self._prev = energy_log
        self.novelty = nov
        self._since += 1
        if len(self._hist) >= 4:
            h = np.fromiter(self._hist, dtype=np.float32)
            thr = float(h.mean() + self.k * h.std()) + self.floor
        else:
            thr = float("inf")
        self._hist.append(nov)
        if nov > thr and self._since >= self.refractory:
            self._since = 0
            # Map "how far over threshold" to 0.2..1 so any onset is visible.
            return 0.2 + 0.8 * float(min(1.0, (nov - thr) / (thr + 1e-6)))
        return 0.0


# --------------------------------------------------------------- elements --

@dataclass(frozen=True)
class ElementBands:
    kick: tuple[float, float] = (40.0, 120.0)
    snare_body: tuple[float, float] = (150.0, 400.0)
    snare_crack: tuple[float, float] = (1000.0, 5000.0)
    hat: tuple[float, float] = (6000.0, 16000.0)
    bass: tuple[float, float] = (40.0, 250.0)
    body: tuple[float, float] = (250.0, 2000.0)
    lead: tuple[float, float] = (2000.0, 8000.0)
    wide: tuple[float, float] = (250.0, 8000.0)


@dataclass
class DetectorConfig:
    bands: ElementBands = field(default_factory=ElementBands)
    onset_k: float = 2.0 # threshold = mean + k*std of recent novelty
    hat_min_flatness: float = 0.15 # below this the "hat" band is tonal, not a hat
    compress: float = 1000.0 # log1p(compress * energy)
    agc_release_s: float = 8.0
    # The AGC never normalises against a peak quieter than this (dBFS). Without
    # a floor, a long silence lets the running peak decay until the noise
    # floor is amplified to full scale and the display "reacts" to nothing.
    agc_floor_db: float = -50.0


@dataclass
class ElementFrame:
    time: float = 0.0
    # global
    level_db: float = -120.0 # absolute level of the mix (dBFS-ish), NOT normalised
    loudness: float = 0.0
    # percussive: onset strengths (0 = no onset this frame)
    kick: float = 0.0
    snare: float = 0.0
    hat: float = 0.0
    cymbal: float = 0.0
    # tonal
    bass: float = 0.0
    body: float = 0.0
    lead: float = 0.0
    wide: float = 0.0


class ElementDetector:
    def __init__(self, stft: STFTConfig, cfg: DetectorConfig | None = None) -> None:
        self.stft = stft
        self.cfg = cfg = cfg or DetectorConfig()
        f = stft.bin_freqs()
        b = cfg.bands
        dt = stft.frame_dt
        self._m = {name: band_mask(f, *getattr(b, name)) for name in b.__dataclass_fields__}
        self.hpss = StreamingHPSS(stft.n_bins)

        k = cfg.onset_k
        self._on_kick = OnsetTracker(dt, k=k, refractory_s=0.10)
        self._on_snare = OnsetTracker(dt, k=k, refractory_s=0.10)
        self._on_hat = OnsetTracker(dt, k=k, refractory_s=0.05)

        floor = float(np.log1p(cfg.compress * 10 ** (cfg.agc_floor_db / 10)))

        def agc() -> PeakNormalizer: return PeakNormalizer(dt, release_s=cfg.agc_release_s, floor=floor)

        # Bands within a stream share one normaliser, so their balance survives:
        # a quiet pad under a loud lead reads as quiet, not as "1.0 of itself".
        self._agc = {n: agc() for n in ("loud", "tonal", "cymbal")}
        # Cymbal wash is defined by sustain, so it gets its own slow-release follower.
        self._cymbal_env = EnvelopeFollower(0.02, 0.35, dt)
        self._prev: dict[str, float] = {}
        self._frame = 0

    def _c(self, e: float) -> float:
        return float(np.log1p(self.cfg.compress * e))

    def process(self, mid_power: np.ndarray, side_power: np.ndarray) -> ElementFrame:
        cfg, m = self.cfg, self._m
        harm_mag, perc_mag = self.hpss.process(np.sqrt(mid_power))
        hp = harm_mag**2
        pp = perc_mag**2
        c = self._c
        agc = self._agc

        out = ElementFrame(time=self._frame * self.stft.frame_dt)
        self._frame += 1

        total = float(mid_power.sum())
        out.level_db = float(10.0 * np.log10(total + 1e-12))
        out.loudness = agc["loud"].update(c(total))

        # ---- percussive ----
        e_kick = float(pp[m["kick"]].sum())
        e_body = float(pp[m["snare_body"]].sum())
        e_crack = float(pp[m["snare_crack"]].sum())
        e_hat = float(pp[m["hat"]].sum())
        lk, lb, lc, lh = c(e_kick), c(e_body), c(e_crack), c(e_hat)
        # Snare needs BOTH body and crack: the min of the two log energies only
        # rises when both do, so a kick (no crack) or a hat (no body) can't fire it.
        l_snare = min(lb, lc)

        # Every drum hit is broadband to some degree (a snare has low thump, a
        # kick has a click), so a band rising is not enough. Each detector is
        # also gated on the *shape* of the rise: which band rose the most.
        nov = {k: max(0.0, v - self._prev.get(k, 0.0)) for k, v in (("k", lk), ("c", lc), ("b", lb), ("h", lh))}
        self._prev.update(k=lk, c=lc, b=lb, h=lh)

        kick_on = self._on_kick.update(lk)
        snare_on = self._on_snare.update(l_snare)
        hat_on = self._on_hat.update(lh)
        hat_flat = flatness(mid_power[m["hat"]])

        out.snare = snare_on
        if kick_on and nov["k"] >= nov["c"]: out.kick = kick_on
        # Hats are gated on the snare's *crack* band, not its body: a kick has
        # a big low-mid rise but almost no crack, so a hat landing on the same
        # beat as a kick still registers. A snare's crack rises more than its
        # hat-band spill, so snares don't double as hats.
        if hat_on and hat_flat >= cfg.hat_min_flatness and nov["h"] >= nov["c"]: out.hat = hat_on

        # Cymbal wash: total (not percussive-only) high-band energy weighted by
        # noisiness. HPSS can't hold a sustained noise in either stream, so we
        # measure it on the full spectrum instead.
        e_high = float(mid_power[m["hat"]].sum()) * min(1.0, hat_flat * 2.0)
        out.cymbal = self._cymbal_env.update(agc["cymbal"].update(c(e_high)))

        # ---- tonal ----
        e_body_h = float(hp[m["body"]].sum())
        e_lead_h = float(hp[m["lead"]].sum())
        tonal = (c(float(hp[m["bass"]].sum())), c(e_body_h), c(e_lead_h), c(float(side_power[m["wide"]].sum())))
        agc["tonal"].update(max(tonal))
        out.bass, out.body, out.lead, out.wide = (min(1.0, v / agc["tonal"].peak) for v in tonal)
        return out


class Analyzer:
    def __init__(self, sample_rate: int, detector_cfg: DetectorConfig | None = None,
                 n_fft: int = 2048, hop: int | None = None) -> None:
        hop = hop or round(sample_rate / 100)  # 10 ms regardless of device rate
        self.stft_cfg = STFTConfig(sample_rate=sample_rate, n_fft=n_fft, hop=hop)
        self.stft = STFT(self.stft_cfg)
        self.detector = ElementDetector(self.stft_cfg, detector_cfg)

    def process_window(self, frames: np.ndarray) -> ElementFrame:
        mid, side = self.stft.power(frames)
        return self.detector.process(mid, side)

    def analyze_offline(self, data: np.ndarray) -> list[ElementFrame]:
        """Run over a whole signal (n, ch). Used by tests and the feature viewer."""
        if data.ndim == 1: data = data[:, None]
        n_fft, hop = self.stft_cfg.n_fft, self.stft_cfg.hop
        padded = np.concatenate([np.zeros((n_fft, data.shape[1]), np.float32), data])
        
        return [
            self.process_window(padded[end - n_fft : end])
            for end in range(n_fft + hop, len(padded) + 1, hop)
        ]
