"""
    colour.py
    
    The cover art's derived colors from cover.py. OKLab conversions
    

    Why OKLab rather than RGB or HSV: Euclidean distance in OKLab tracks how
    different two colours *look*, so clustering finds the colours a person would
    name, and interpolating between two colours doesn't pass through a muddy grey
    (the classic RGB red->green midpoint problem). HSV hue is not perceptually
    uniform either: equal hue steps look like very unequal colour changes.

    A palette is ONE colour, taken from the cover. Intensity is the single source
    of truth for brightness. Reading upward from zero:

    I = 0            the BACKGROUND colour (the "off" screen; user-chosen)
    0 .. i_full      background fades into the colour
    i_full .. white_hi
                    lightness is set by intensity alone, rising from the
                    colour's lightness to pure white
    white_lo .. white_hi
                    the saturation plateau: chroma drops quickly to zero
    white_hi .. 1    pure white

    So hotter is always brighter. Chroma is pulled in as lightness approaches 1,
    which is both what washes the colour out and what keeps it inside sRGB.

    The white plateau exists because a lightness curve alone only reaches white at
    I = 1, which a soft-kneed intensity almost never hits: the core read as a pale
    tint. With a band that is fully white, the hottest part of the body is white
    for any reasonably present music.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_M1 = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
])
_M2 = np.array([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
])
_M1_INV = np.linalg.inv(_M1)
_M2_INV = np.linalg.inv(_M2)


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, None)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def linear_to_oklab(rgb: np.ndarray) -> np.ndarray:
    lms = rgb @ _M1.T
    return np.cbrt(lms) @ _M2.T


def oklab_to_linear(lab: np.ndarray) -> np.ndarray:
    lms = (np.asarray(lab) @ _M2_INV.T) ** 3
    return lms @ _M1_INV.T


def srgb_to_oklab(srgb: np.ndarray) -> np.ndarray:
    return linear_to_oklab(srgb_to_linear(srgb))


def oklab_to_srgb(lab: np.ndarray) -> np.ndarray:
    return np.clip(linear_to_srgb(oklab_to_linear(lab)), 0.0, 1.0)


def oklab_to_oklch(lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab)
    c = np.hypot(lab[..., 1], lab[..., 2])
    h = np.arctan2(lab[..., 2], lab[..., 1])
    return np.stack([lab[..., 0], c, h], axis=-1)


def oklch_to_oklab(lch: np.ndarray) -> np.ndarray:
    lch = np.asarray(lch)
    return np.stack(
        [lch[..., 0], lch[..., 1] * np.cos(lch[..., 2]), lch[..., 1] * np.sin(lch[..., 2])],
        axis=-1,
    )


def in_gamut(lab: np.ndarray, tol: float = 1e-4) -> np.ndarray:
    rgb = oklab_to_linear(lab)
    return np.all((rgb >= -tol) & (rgb <= 1 + tol), axis=-1)


def gamut_map(lab: np.ndarray, steps: int = 20) -> np.ndarray:
    """
    Bring colours into sRGB by reducing chroma at constant L and hue.

    Plain RGB clipping shifts hue (a saturated blue clips toward purple);
    reducing chroma keeps the colour recognisably the same, just less vivid.
    """
    lab = np.atleast_2d(np.asarray(lab, dtype=np.float64)).copy()
    lch = oklab_to_oklch(lab)
    lo = np.zeros(len(lab))
    hi = lch[:, 1].copy()
    ok = in_gamut(lab)
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        trial = oklch_to_oklab(np.stack([lch[:, 0], mid, lch[:, 2]], axis=-1))
        g = in_gamut(trial)
        lo = np.where(g, mid, lo)
        hi = np.where(g, hi, mid)
    mapped = oklch_to_oklab(np.stack([lch[:, 0], lo, lch[:, 2]], axis=-1))
    return np.where(ok[:, None], lab, mapped)


# ----------------------------------------------------------------- palette --

# Display chroma floor for chromatic palettes: a washed-out grey-blue cover
# shows as a clear (if gentle) blue instead of a muddy grey.
MIN_DISPLAY_CHROMA = 0.09


@dataclass(eq=False)  # identity comparison: lab is an array
class Swatch:
    lab: np.ndarray
    share: float
    neutral: bool

    @property
    def srgb(self) -> np.ndarray:
        return oklab_to_srgb(self.lab)

    @property
    def hex(self) -> str:
        r, g, b = (np.round(self.srgb * 255)).astype(int)
        return f"#{r:02x}{g:02x}{b:02x}"


@dataclass
class Palette:
    mode: str                     # "grayscale" or "chromatic"
    base: Swatch                  # the colour as found on the cover
    base_display: np.ndarray      # (3,) OKLab, display-ready

    def describe(self) -> str:
        return f"{self.mode}: {self.base.hex} ({self.base.share:.0%} of the cover)"

    def to_uniforms(self) -> dict:
        return {"u_base": self.base_display.astype(np.float32)}


def make_palette(sw: Swatch) -> Palette:
    """The display version of a colour. Its lightness is pinned to a mid range:
    it is the colour of "some reactance", and must sit clearly between the
    background and the white core."""
    lch = oklab_to_oklch(sw.lab)
    lch[0] = np.clip(lch[0], 0.45, 0.62)
    lch[1] = 0.0 if sw.neutral else max(lch[1], MIN_DISPLAY_CHROMA)
    display = gamut_map(oklch_to_oklab(lch))[0]
    return Palette("grayscale" if sw.neutral else "chromatic", sw, display)


# ----------------------------------------------------------------- shading --

@dataclass
class ShadeParams:
    i_full: float = 0.4          # intensity at which the colour is fully shown
    white_hi: float = 0.84       # pure white from this intensity up
    # Width of the colour -> white transition below white_hi. THE knob for how
    # gradually colour fades into the white core: larger = smoother, smaller =
    # crisper.
    white_band: float = 0.38
    white_gamma: float = 1.6     # >1 keeps mid intensities colourful, rushes to white at the top

    @property
    def white_lo(self) -> float:
        """Intensity where colour starts giving way to white."""
        return self.white_hi - self.white_band


def smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, dtype=np.float64) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def max_chroma(lightness: np.ndarray) -> np.ndarray:
    """Chroma allowance that shrinks to 0 at white: the white-wash."""
    return 0.37 * np.clip((1.0 - lightness) / 0.3, 0.0, 1.0)


def shade(palette: Palette, intensity, background_srgb=(0.0, 0.0, 0.0),
          p: ShadeParams | None = None) -> np.ndarray:
    """CPU mirror of shade() in shaders/color.glsl. Returns OKLab (..., 3)."""
    p = p or ShadeParams()
    i = np.clip(np.asarray(intensity, dtype=np.float64), 0.0, 1.0)
    bg = srgb_to_oklab(np.asarray(background_srgb, dtype=np.float64))
    base = palette.base_display
    k = np.clip((i - p.i_full) / (p.white_hi - p.i_full), 0, 1)
    light = base[0] + (1.0 - base[0]) * k**p.white_gamma
    ab = base[1:] * (1.0 - smoothstep(p.white_lo, p.white_hi, i))[..., None]
    c = np.hypot(ab[..., 0], ab[..., 1])
    ab = ab * np.minimum(1.0, max_chroma(light) / np.maximum(c, 1e-5))[..., None]
    lit = np.concatenate([light[..., None], ab], -1)
    f = smoothstep(0.0, p.i_full, i)[..., None]
    return bg + (lit - bg) * f


def blend_uniforms(a: dict, b: dict, t: float) -> dict:
    """
    Palette uniforms part-way from `a` to `b` (t in 0..1), for crossfades.
    Colours blend in OKLab, which keeps the midpoint perceptually between the
    two rather than muddy."""
    t = float(np.clip(t, 0.0, 1.0))
    t = t * t * (3 - 2 * t)
    ba, bb = np.asarray(a["u_base"], np.float32), np.asarray(b["u_base"], np.float32)
    return {"u_base": ba + (bb - ba) * t}
