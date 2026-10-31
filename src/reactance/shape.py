"""
    shape.py
    
    A displays outline, this is the shape of the reactant.
    
    This file incorporates a circular and rectangular display, but 
    custom SVGs are allowed.

    depth: 0 at the edge -> 1 at the deepest interior point (the core)
    d: signed distance to the nearest edge in shape units (> 0 inside)

    These are continuous functions, a display of any resolution just 
    evaluates them at its pixel centres.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

CIRCLE = "circle"
FULLSCREEN = "fullscreen"
CHUNK = 8192  # points per batch; bounds memory at CHUNK x n_edges floats


@dataclass
class Coords:
    depth: np.ndarray
    d: np.ndarray


class Shape:
    thickness: float        # depth at the core, in shape units
    core: np.ndarray        # deepest interior point: where the body grows from
    fill_view = False       # True: fill the window edge to edge (fullscreen)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def coords(self, p: np.ndarray) -> Coords:
        raise NotImplementedError


class Disc(Shape):
    def __init__(self, center=(0.0, 0.0), radius: float = 1.0) -> None:
        self.center = np.asarray(center, dtype=np.float64)
        self.radius = float(radius)
        self.thickness = self.radius
        self.core = self.center

    def bounds(self):
        r = np.array([self.radius, self.radius])
        return self.center - r, self.center + r

    def coords(self, p):
        rho = np.hypot(*(np.asarray(p, dtype=np.float64) - self.center).T)
        return Coords(depth=1.0 - rho / self.radius, d=self.radius - rho)

def _distances(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each point p (M,2) to each segment a->b (K,2): (M,K)."""
    ab = b - a
    ab2 = np.maximum((ab**2).sum(1), 1e-12)
    ap = p[:, None, :] - a[None]
    u = np.clip((ap * ab[None]).sum(-1) / ab2, 0.0, 1.0)
    diff = ap - u[..., None] * ab[None]
    return np.sqrt((diff**2).sum(-1))


def _inside(p: np.ndarray, rings: list[np.ndarray]) -> np.ndarray:
    """Even-odd rule over all rings, so holes work without orientation rules."""
    inside = np.zeros(len(p), dtype=bool)
    x, y = p[:, 0:1], p[:, 1:2]
    for ring in rings:
        a, b = ring, np.roll(ring, -1, axis=0)
        ya, yb = a[None, :, 1], b[None, :, 1]
        crosses = (ya > y) != (yb > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = a[None, :, 0] + (y - ya) * (b[None, :, 0] - a[None, :, 0]) / (yb - ya)
        inside ^= ((crosses & (x < xi)).sum(1) % 2).astype(bool)
    return inside


class Polygon(Shape):
    """Filled region bounded by one or more rings (holes via even-odd).

    Depth uses a SMOOTH minimum over the edges. With a plain min, "distance to
    the nearest edge" has creases wherever the nearest edge switches: along
    every corner's diagonal and along the medial axis (the centre line of a
    long rectangle). A glow drawn from depth turns each crease into a visible
    line. The soft-min rounds them off. d stays exact, so the outline and
    anti-aliasing are unaffected.
    """

    def __init__(self, rings, depth_probe: int = 96, soft: float = 0.15) -> None:
        self.rings = [np.asarray(r, dtype=np.float64) for r in rings]
        self.rings = [r[:-1] if np.allclose(r[0], r[-1]) else r for r in self.rings]
        self._a = np.concatenate(self.rings)
        self._b = np.concatenate([np.roll(r, -1, axis=0) for r in self.rings])
        self._tau = 0.0
        # Estimate the deepest interior point so depth is normalised to [0, 1].
        lo, hi = self.bounds()
        g = np.stack(np.meshgrid(np.linspace(lo[0], hi[0], depth_probe),
                                 np.linspace(lo[1], hi[1], depth_probe)), -1).reshape(-1, 2)
        d = self._signed(g)[0]
        exact_max = max(float(d.max()), 1e-9)
        # Core = centroid of the deepest few percent, not a single argmax, so a
        # shape with a ridge of equal depth (a long rectangle) gets the middle.
        self.core = g[d >= exact_max * 0.97].mean(0)
        self._tau = soft * exact_max
        self.thickness = max(float(self._signed(g)[1].max()), 1e-9)

    def bounds(self):
        allp = np.concatenate(self.rings)
        return allp.min(0), allp.max(0)

    def _signed(self, p):
        dist = _distances(p, self._a, self._b)
        dmin = dist.min(1)
        inside = _inside(p, self.rings)
        
        if self._tau > 0:
            w = np.exp(-(dist - dmin[:, None]) / self._tau).sum(1)
            dsoft = np.maximum(dmin - self._tau * np.log(w), 0.0)
        else: dsoft = dmin
        
        return np.where(inside, dmin, -dmin), np.where(inside, dsoft, 0.0)

    def coords(self, p):
        p = np.asarray(p, dtype=np.float64)
        d, ds = np.empty(len(p)), np.empty(len(p))
        
        for i in range(0, len(p), CHUNK): d[i:i + CHUNK], ds[i:i + CHUNK] = self._signed(p[i:i + CHUNK])
            
        return Coords(depth=np.clip(ds / self.thickness, 0.0, 1.0), d=d)


def rect(w: float, h: float, center=(0.0, 0.0)) -> Polygon:
    cx, cy = center
    x, y = w / 2, h / 2
    return Polygon([[(cx - x, cy - y), (cx + x, cy - y), (cx + x, cy + y), (cx - x, cy + y)]])


def fullscreen(aspect: float, height: float = 100.0) -> Polygon:
    """A filled rectangle matching the window's aspect: the whole window is screen.

    It has to be rebuilt when the window's aspect changes, which is cheap
    (four edges).
    """
    shape = rect(height * aspect, height)
    shape.fill_view = True
    return shape


# --------------------------------------------------------------------- SVG --

def load_svg(path: str | Path, curve_steps: int = 24) -> Shape:
    """The first filled element of an SVG: a <circle> becomes a Disc, anything
    else a Polygon (all subpaths; holes by even-odd). So any vector editor is
    the shape editor."""
    try:
        from svgelements import SVG, Circle, Close, Color, Line, Move
        from svgelements import Path as SvgPath
        from svgelements import Shape as SvgShape
    except ImportError as e:
        raise RuntimeError("install the 'svg' extra: pip install -e .[svg]") from e

    def flatten(subpath) -> np.ndarray:
        pts: list[tuple[float, float]] = []
        
        for seg in subpath:
            if isinstance(seg, (Move, Line)): pts.append((seg.end.x, seg.end.y))
            elif not isinstance(seg, Close): pts.extend((q[0], q[1]) for q in seg.npoint(np.linspace(0, 1, curve_steps + 1)[1:]))
            
        arr = np.array(pts, dtype=np.float64).reshape(-1, 2)
        
        if len(arr) > 2 and np.allclose(arr[0], arr[-1]): arr = arr[:-1]
        
        keep = np.concatenate([[True], np.any(np.abs(np.diff(arr, axis=0)) > 1e-9, axis=1)])
        return arr[keep]

    for el in SVG.parse(str(path)).elements():
        if not isinstance(el, SvgShape): continue
        if el.fill is None or el.fill.value is None or el.fill == Color("none"): continue
        if isinstance(el, Circle) and abs(el.rx - el.ry) < 1e-9: return Disc((el.cx, el.cy), el.rx)
        
        p = SvgPath(el)
        p.reify()
        rings = [r for r in (flatten(sp) for sp in p.as_subpaths()) if len(r) >= 3]
        
        if rings: return Polygon(rings)
    raise ValueError(f"no filled element in {path} (outlines/strokes aren't supported)")


def load_shape(name: str, aspect: float = 1.0) -> Shape:
    """CIRCLE, FULLSCREEN, or a path to an SVG file."""
    if name == CIRCLE: return Disc((110.0, 110.0), 100.0)
    if name == FULLSCREEN: return fullscreen(aspect)
    return load_svg(name)


# ------------------------------------------------------------------ baking --

@dataclass
class BakedShape:
    """(depth, d) sampled on a grid, for the GPU.

    The shader evaluates noise per pixel at that pixel's continuous position;
    only the geometry is precomputed. Those fields are smooth, so a bilinearly
    filtered texture reproduces them accurately at any display resolution.
    """

    texture: np.ndarray        # (H, W, 2) float32, row 0 = min y
    origin: np.ndarray         # world position of the texel grid corner (min x, min y)
    size: np.ndarray           # world extent covered by the texture
    shape: Shape


def bake(shape: Shape, resolution: int = 512, pad: float = 0.05) -> BakedShape:
    lo, hi = shape.bounds()
    ext = hi - lo
    lo, hi = lo - ext * pad, hi + ext * pad
    ext = hi - lo
    aspect = ext[0] / ext[1]
    w = resolution if aspect >= 1 else max(8, int(round(resolution * aspect)))
    h = resolution if aspect < 1 else max(8, int(round(resolution / aspect)))
    xs = lo[0] + (np.arange(w) + 0.5) / w * ext[0]
    ys = lo[1] + (np.arange(h) + 0.5) / h * ext[1]
    gx, gy = np.meshgrid(xs, ys)
    c = shape.coords(np.stack([gx.ravel(), gy.ravel()], 1))
    tex = np.stack([c.depth, c.d], -1)
    return BakedShape(tex.reshape(h, w, 2).astype(np.float32), lo, ext, shape)
