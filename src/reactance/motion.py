"""
    motion.py
    
    This is where audio elements become patterns which the share will draw.
    
    Audio elements become pattern parameters.
    
    Silence occurs when presence (ant sound intensity) becomes 0. Silence
    causes the pattern to draw back.

    The reactance knob sets the RATE at which the blob follows the music, with
    momentum so slow settings still reach intense big moments.

    element        visual                              why
    -------------  ----------------------------------  ---------------------------------
    loudness+bass  reach (how much of the shape lit)   volume = size, the most intuitive
                    and overall level
    kick           reach surge + core pulse            the heaviest hit pushes outward
    snare / clap   a wave travelling out from the      sharp event that stays attached
                    core through the body               to the body
    loudness/kick  noise-pulse rate (noise_pulses.py)  busier music emits more swells
    hat            small brightness lift               (no scattered specks)
    body (tonal)   warp: organic bending               sustained notes make it flow
    lead + wide    noise-pulse rate (with loudness)    melodic content adds swells
    loudness       pattern speed (0 in silence)        busier music moves faster
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from .analysis import ElementFrame
from .shape import Shape

# --------------------------------------------------------------- reactance --
#
# The "low reactance <-> high reactance" knob: the RATE of change, with momentum.
#
# Both ends show the same intensities: the same whiteness at big moments, the
# same size for the same music. What changes is how the blob gets there:
#
#   low reactance   The blob moves like a heavy object on a soft spring. It
#                   accelerates gradually toward where the music wants it and
#                   carries momentum, so a burst of rapid hits doesn't make it
#                   jitter; it swells smoothly instead.
#   high reactance  A stiff spring: it snaps to the music almost immediately.
#
# The problem with simply slowing things down is that an intense moment can
# end before a slow follower ever reaches it, so the big moments would never
# show. Momentum fixes that in two parts (see mapping._Momentum):
#
#   * hold    An intense moment raises a *target* that is held and released
#             gradually, even after the music itself has dropped back.
#   * spring  The blob accelerates toward the target with a second-order
#             (mass-spring-damper) response, so it has velocity: it keeps
#             rising after the moment until it reaches what the moment
#             deserved, then eases back.
#
# Falling uses a stiffer spring than rising, so quiet passages still clear
# reasonably quickly at low reactance.

def lerp(a: float, b: float, t: float) -> float: return a + (b - a) * t

def glerp(a: float, b: float, t: float) -> float: return a * (b / a) ** t


@dataclass(frozen=True)
class Timing:
    attack_s: float = 0.03 # feature smoothing (before momentum)
    release_s: float = 0.15
    impulse_release_s: float = 0.2
    presence_release_s: float = 0.35
    baseline_s: float = 2.0 # window for the "recent average" of each feature


TIMING = Timing()


@dataclass(frozen=True)
class Derived:
    spring_up_hz: float # natural frequency while rising (lower = more momentum)
    spring_down_hz: float # ... while falling
    hold_s: float # how long an intense moment's target is held
    deviation_gain: float
    onset_min: float
    narrow_gain: float
    speed_base: float
    speed_audio: float
    reach_base: float
    reach_audio: float
    reach_kick: float
    reach_snare: float
    branch_audio: float
    warp_base: float
    warp_audio: float
    pulse_gain: float
    ripple_gain: float
    ripple_min_interval_s: float


@dataclass
class Reactivity:
    energy: float = 0.5

    def derive(self) -> Derived:
        e = min(1.0, max(0.0, self.energy))
        n = 0.8 # what it reacts to is the same at every setting
        return Derived(
            # Wide on purpose: 0 is a slow swell (a big moment takes ~2 s to
            # build), 1 tracks individual kicks.
            spring_up_hz=glerp(0.25, 8.0, e),
            spring_down_hz=glerp(0.5, 8.0, e),
            hold_s=glerp(2.0, 0.04, e),
            deviation_gain=1.0,
            onset_min=0.3,
            narrow_gain=n,
            speed_base=0.05,
            speed_audio=lerp(0.06, 0.5, e),
            reach_base=0.44,
            reach_audio=0.22,
            reach_kick=0.06 * n,
            reach_snare=0.04 * n,
            branch_audio=0.6 * n,
            warp_base=0.18,
            warp_audio=0.2 * n,
            pulse_gain=0.25 * n,
            ripple_gain=0.45 * n,
            ripple_min_interval_s=lerp(1.2, 0.1, e),
        )


# ------------------------------------------------------------------ state --
#
# PatternState: everything the shader needs for one frame, as plain values.
#
# The renderer is a pure function of (shape, palette, settings, PatternState).
# Nothing in here knows about audio, so the shader can be driven by recorded
# states, a test, or a UI just as well as by music.

MAX_RIPPLES = 8
MAX_PULSES = 3


@dataclass
class Ripple:
    radius: float # in core-distance units (0 at the core, 1 at the edge)
    max_radius: float
    speed: float
    amp: float

    @property
    def strength(self) -> float:
        return self.amp * max(0.0, 1.0 - self.radius / self.max_radius) ** 2


@dataclass
class PatternState:
    time: float = 0.0
    # geometry-derived look (set by initial_state)
    scale: float = 1.0 # warp feature size, shape units
    branch_freq: float = 0.55 # how many lobes in the outline's wobble
    balance: float = 0.5 # 0 = mostly low intensity, 1 = mostly high
    branching: float = 0.5 # how often noise pulses are emitted: 0 = never
    size: float = 1.0 # overall blob size multiplier (the size control)
    # Noise pulses: up to MAX_PULSES round swells of the main noise drifting out
    # from the core, as (centre x, centre y, radius, strength); centre and
    # radius are in multiples of the blob's radius. Set by the mapper.
    noise_pulses: list[tuple[float, float, float, float]] = field(default_factory=list)
    # audio-driven
    level: float = 0.0 # overall intensity (0 in silence)
    reach: float = 0.0 # how far the body extends from the core (0..1)
    branch: float = 0.0 # tendril amplitude, fraction of reach
    warp: float = 0.15 # organic bending of the body, fraction of scale
    pulse: float = 0.0
    ripples: list[Ripple] = field(default_factory=list)
    ripple_width: float = 0.08
    edge_margin: float = 0.15 # shape units

    def noise_pulse_arrays(self) -> np.ndarray:
        a = np.zeros((MAX_PULSES, 4), np.float32)
        for k, p in enumerate(self.noise_pulses[:MAX_PULSES]): a[k] = p
        return a

    def ripple_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        r = np.zeros(MAX_RIPPLES, np.float32)
        a = np.zeros(MAX_RIPPLES, np.float32)
        for k, rp in enumerate(self.ripples[:MAX_RIPPLES]): r[k], a[k] = rp.radius, rp.strength
        return r, a


def initial_state(shape: Shape) -> PatternState:
    """Geometry-derived defaults: feature sizes scale with the shape."""
    lo, hi = shape.bounds()
    return PatternState(scale=float(np.max(hi - lo)) / 2.5, ripple_width=0.08)


# ----------------------------------------------------------- noise pulses --
#
# Noise pulses: extra noise that starts in the middle and melts outward.
#
# There are no separate shapes. A pulse is a second field of the same kind as
# the main blob's, centred on a point that starts at the core and drifts
# outward. The shader ADDS the fields (metaballs) and reads the noise's profile
# from the sum, so the main noise map simply swells: a round, soft bulge that
# begins inside the core, carries its heat outward, and dissolves as it moves
# away, with no seams or circles-within-circles.
#
# How often pulses appear is the pulses control, scaled by the music: louder
# passages emit more, and a kick can emit one directly. At most MAX_PULSES are
# alive at once, sent in directions spread around the blob.

# How far a pulse's centre drifts, in multiples of the blob's radius. Kept
# under ~0.75 so the pulse always overlaps the core: it swells the outline
# rather than floating off as a separate bubble.
TRAVEL = (0.45, 0.75)
LIFE = (1.2, 2.4) # seconds


def _ease_out(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return 1.0 - (1.0 - u) ** 3


@dataclass
class NoisePulse:
    theta: float # direction of travel (radians)
    drift: float # slight change of direction over the life (radians / s)
    travel: float # final offset of the centre, in blob radii
    life: float # seconds
    age: float = 0.0

    def as_uniform(self) -> tuple[float, float, float, float]:
        """(centre x, centre y) in blob radii, radius in blob radii, strength."""
        u = min(1.0, self.age / self.life)
        d = self.travel * _ease_out(u) # glides out, slowing down
        radius = 0.85 + 0.25 * u # spreads a little as it goes
        # Fades in over the first moments (it starts inside the core), carries
        # the core's heat out for the first ~30% of its life, then melts.
        a = min(1.0, u / 0.12)
        t = min(1.0, max(0.0, (u - 0.3) / 0.7))
        strength = a * a * (3 - 2 * a) * (1.0 - t * t * (3 - 2 * t))
        return (d * math.cos(self.theta), d * math.sin(self.theta), radius, strength)


class PulseEmitter:
    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)
        self.pulses: list[NoisePulse] = []
        self.spawned = 0 # total pulses ever emitted (for tests/diagnostics)

    def rate(self, amount: float, intensity: float) -> float:
        """Pulses per second: the pulses control, loosely scaled by the music."""
        return amount * (0.2 + 1.2 * intensity)

    def update(self, dt: float, amount: float, intensity: float, kick: float = 0.0,
               presence: float = 1.0) -> list[tuple[float, float, float, float]]:
        for p in self.pulses:
            p.age += dt
            p.theta += p.drift * dt
        self.pulses = [p for p in self.pulses if p.age < p.life]
        if amount > 0 and presence > 0.2 and len(self.pulses) < MAX_PULSES:
            chance = self.rate(amount, intensity) * dt
            if kick > 0: chance += 0.35 * amount * kick # a kick can emit one directly
            if self.rng.random() < chance: self._emit(intensity)
        return [p.as_uniform() for p in self.pulses]

    def _emit(self, intensity: float) -> None:
        rng = self.rng
        for _ in range(8):   # a direction away from pulses already travelling
            theta = rng.uniform(0, 2 * math.pi)
            if all(abs((theta - p.theta + math.pi) % (2 * math.pi) - math.pi) > 1.2 for p in self.pulses): break
        lo, hi = TRAVEL
        
        self.spawned += 1
        self.pulses.append(NoisePulse(
            theta=theta,
            drift=rng.uniform(-0.25, 0.25),
            travel=(lo + (hi - lo) * min(1.0, intensity)) * rng.uniform(0.9, 1.1),
            life=rng.uniform(*LIFE),
        ))


# ------------------------------------------------------ size compensation ----
#
# Each pulse is a field as big as the core's, so adding one grows the visible
# body a lot (an egg plus most of a second disc). Left alone, the blob's size
# would swing with every pulse, on top of the swing the music already gives
# it. Instead the body's overall scale is reduced while pulses are alive so
# its visible AREA stays what the music asked for: pulses reshape and pull
# the body around, they don't inflate it.
#
# VISIBLE_X is where the edge reads as lit, in blob radii (measured: the lit
# diameter is ~0.65 x reach on a circle at the floor level). The area above
# the matching field level is computed on a small grid, mirroring the shader:
# F = kernel(|p|) + sum_k w_k * kernel(|p - c_k| / r_k), kernel(d) = (1 - d^2)^2.

VISIBLE_X = 0.65
_F_EDGE = (1.0 - VISIBLE_X ** 2) ** 2
_N = 72
_ax = np.linspace(-2.2, 2.2, _N)
_GX, _GY = np.meshgrid(_ax, _ax)


def _kernel(d: np.ndarray) -> np.ndarray:
    return np.clip(1.0 - d * d, 0.0, None) ** 2


def area_compensation(pulses: list[tuple[float, float, float, float]]) -> float:
    """
    Scale (<= 1) for the body's reach that keeps its lit area unchanged
    by the given pulses (as from NoisePulse.as_uniform).
    """
    if not pulses: return 1.0
    F = _kernel(np.hypot(_GX, _GY))
    base = int((F > _F_EDGE).sum())
    for cx, cy, r, w in pulses:
        if w > 0: F = F + w * _kernel(np.hypot(_GX - cx, _GY - cy) / max(r, 1e-3))
    grown = int((F > _F_EDGE).sum())
    return float(np.sqrt(base / max(grown, base)))


# ----------------------------------------------------------------- mapping --

GATE_LO_DB = -62.0 # below this: silence
GATE_HI_DB = -45.0 # above this: fully present
PRESENCE_FULL = 0.35 # presence at which the body is at full (minimum) size
LEVEL_MIN = 0.62 # body level for anything audibly playing


def _smoothstep(e0: float, e1: float, x: float) -> float:
    t = min(1.0, max(0.0, (x - e0) / (e1 - e0)))
    return t * t * (3 - 2 * t)


SIZE_MAX = 1.3 # the size control runs 0..SIZE_MAX (1.0 was the old maximum)


def size_scale(size: float) -> float:
    """The size control (0..SIZE_MAX) as a multiplier on the blob's reach."""
    return 0.45 + 0.85 * min(SIZE_MAX, max(0.0, size))


class _Momentum:
    """
    Peak hold + mass-spring-damper: slow, but never misses a big moment.

    target  max(input, held value); the held value releases over hold_s, so an
            intense moment keeps pulling after it has passed.
    x, v    position and velocity of a damped spring chasing the target
            (zeta ~0.85: barely overshoots). A low natural frequency gives
            gradual acceleration and momentum; a high one tracks instantly.
    """

    ZETA = 0.85

    def __init__(self) -> None:
        self.held = 0.0
        self.x = 0.0
        self.v = 0.0

    def update(self, target: float, dt: float, up_hz: float, down_hz: float, hold_s: float) -> float:
        if target >= self.held: self.held = target
        else: self.held += (target - self.held) * (1.0 - math.exp(-dt / max(hold_s, 1e-3)))
        
        goal = self.held
        w = 2 * math.pi * (up_hz if goal >= self.x else down_hz)
        # semi-implicit Euler, sub-stepped so stiff springs stay stable at 30 fps
        steps = max(1, int(math.ceil(dt * w / 0.5)))
        h = dt / steps
        for _ in range(steps):
            self.v += (w * w * (goal - self.x) - 2 * self.ZETA * w * self.v) * h
            self.x += self.v * h
        self.x = max(0.0, self.x)
        return self.x


class _Follower:
    def __init__(self) -> None:
        self.v = 0.0

    def smooth(self, x: float, dt: float, attack: float, release: float) -> float:
        tau = attack if x > self.v else release
        self.v += (x - self.v) * (1.0 - np.exp(-dt / max(tau, 1e-6)))
        return self.v

    def impulse(self, x: float, dt: float, release: float) -> float:
        """Instant attack, exponential release: for one-frame onset events."""
        self.v = max(x, self.v * np.exp(-dt / max(release, 1e-6)))
        return self.v


class _Sensitive:
    """A feature split into slow baseline + deviation; output = base + gain * dev."""

    def __init__(self) -> None:
        self.fast = _Follower()
        self.base = _Follower()

    def update(self, x: float, dt: float, gain: float) -> float:
        tm = TIMING
        fast = self.fast.smooth(x, dt, tm.attack_s, tm.release_s)
        base = self.base.smooth(x, dt, tm.baseline_s, tm.baseline_s)
        return min(1.0, max(0.0, base + gain * (fast - base)))


class AudioMapper:
    def __init__(self, shape: Shape, reactivity: Reactivity | None = None, edge_margin: float = 0.25, seed: int = 0) -> None:
        self.shape = shape
        self.reactivity = reactivity or Reactivity()
        self.state = initial_state(shape)
        self.set_edge_margin(edge_margin)
        # (cymbal is only followed for the panel's input meters)
        self._f = {k: _Follower() for k in ("presence", "kick", "snare", "hat", "cymbal")}
        self._s = {k: _Sensitive() for k in ("loud", "bass", "body", "lead", "wide")}
        self._since_ripple = 1e9
        self.waves = 0.5 # snare-wave strength: 0 = off, 1 = strong
        self.size = 0.65 # blob size control (0..SIZE_MAX); see size_scale
        self._emitter = PulseEmitter(seed)
        self._m = {k: _Momentum() for k in ("reach", "level")}

    def set_edge_margin(self, margin: float) -> None:
        """margin: fraction of the shape's thickness over which the edge fade acts."""
        self.edge_margin = margin
        self.state.edge_margin = float(margin) * self.shape.thickness

    def update(self, fr: ElementFrame, dt: float) -> PatternState:
        p = self.reactivity.derive()
        tm = TIMING
        f, sv = self._f, self._s
        st = self.state
        g = p.deviation_gain

        def onset(x: float) -> float:
            # weak hits don't count at low reactance
            return x if x >= p.onset_min else 0.0

        gate = _smoothstep(GATE_LO_DB, GATE_HI_DB, fr.level_db)
        presence = f["presence"].smooth(gate, dt, 0.05, tm.presence_release_s)
        loud = sv["loud"].update(fr.loudness, dt, g)
        bass = sv["bass"].update(fr.bass, dt, g)
        body = sv["body"].update(fr.body, dt, g)
        lead = sv["lead"].update(fr.lead, dt, g)
        wide = sv["wide"].update(fr.wide, dt, g)
        kick = f["kick"].impulse(onset(fr.kick), dt, tm.impulse_release_s)
        snare = f["snare"].impulse(onset(fr.snare), dt, tm.impulse_release_s)
        hat = f["hat"].impulse(onset(fr.hat), dt, tm.impulse_release_s * 0.6)
        f["cymbal"].smooth(fr.cymbal, dt, tm.attack_s, tm.release_s)
        drive = 0.55 * loud + 0.45 * bass

        st.time += dt * presence * (p.speed_base + p.speed_audio * loud)
        mo, up, down, hold = self._m, p.spring_up_hz, p.spring_down_hz, p.hold_s
        # Level and reach chase their targets with momentum (see _Momentum).
        # Presence decides WHETHER the body is there, not how big it is: it
        # saturates early, so a quiet song (which sits low in the gate) still
        # gets the full minimum body, and only near-silence shrinks it away.
        # The minimums (LEVEL_MIN, reach_base) keep quiet songs from shrinking
        # to a dim dot; the audio terms on top are deliberately modest so the
        # size swings over a narrow range.
        on = _smoothstep(0.0, PRESENCE_FULL, presence)
        level_t = on * (LEVEL_MIN + (1.0 - LEVEL_MIN) * drive + 0.08 * p.narrow_gain * hat)
        reach_t = on * (p.reach_base + p.reach_audio * drive + p.reach_kick * kick + p.reach_snare * snare)
        st.level = mo["level"].update(level_t, dt, up, down, hold)
        st.reach = mo["reach"].update(reach_t, dt, up, down, hold) * size_scale(self.size)
        st.size = size_scale(self.size)
        st.branch = min(1.0, p.branch_audio * (lead + 0.5 * wide + 0.6 * hat))
        st.noise_pulses = self._emitter.update(dt, st.branching, min(1.0, drive + 0.3 * st.branch), kick, presence)
        # Pulses reshape the body without inflating it (see area_compensation).
        st.reach *= area_compensation(st.noise_pulses)
        st.warp = p.warp_base + p.warp_audio * body
        st.pulse = presence * p.pulse_gain * kick
        self._update_ripples(onset(fr.snare), dt, p, presence)
        return st

    def _update_ripples(self, snare: float, dt: float, p, presence: float) -> None:
        st = self.state
        for rp in st.ripples: rp.radius += rp.speed * dt
        st.ripples = [rp for rp in st.ripples if rp.radius < rp.max_radius]
        self._since_ripple += dt
        
        if not (snare > 0 and p.ripple_gain > 0 and self.waves > 0
            and len(st.ripples) < MAX_RIPPLES
            and self._since_ripple >= p.ripple_min_interval_s): return
        
        amp = p.ripple_gain * snare * presence * 2.0 * self.waves
        st.ripples.append(Ripple(0.0, 1.0, 1.4, amp))
        self._since_ripple = 0.0

    def meters(self) -> dict[str, float]:
        """Current smoothed envelopes, for a level-meter display."""
        out = {k: float(f.v) for k, f in self._f.items()}
        out.update({k: float(s.fast.v) for k, s in self._s.items()})
        return out

    def snapshot(self) -> PatternState:
        return replace(self.state, ripples=list(self.state.ripples), noise_pulses=list(self.state.noise_pulses))
