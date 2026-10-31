"""
    render.py
    
    OpenGL rendering, including display window, and a headless renderer.

    The baked shape is uploaded once; every frame is a few dozen uniform writes,
    and all per-pixel work (noise, shading, anti-aliasing) is in the fragment
    shader. That is why Python is fine here: the language never touches a pixel.
"""

from __future__ import annotations
import re
import time
from pathlib import Path
import numpy as np
from .colour import Palette, ShadeParams, blend_uniforms, srgb_to_oklab
from .motion import PatternState
from .shape import BakedShape, Shape

SHADERS = Path(__file__).parent / "shaders"
_INCLUDE = re.compile(r'^\s*#include\s+"([^"]+)"\s*$', re.MULTILINE)


def load_shader(name: str) -> str:
    """GLSL source with `#include "file"` lines resolved."""
    src = (SHADERS / name).read_text(encoding="utf-8")
    return _INCLUDE.sub(lambda m: load_shader(m.group(1)), src)


def _lab(srgb) -> tuple[float, float, float]:
    return tuple(float(v) for v in srgb_to_oklab(np.asarray(srgb, dtype=np.float64)))


def fit_view(shape: Shape, width: int, height: int, pad: float = 0.06):
    """World rectangle (min, size) that fits the shape into a window, aspect
    preserved. The fullscreen shape gets no padding."""
    if shape.fill_view: pad = 0.0
    
    lo, hi = shape.bounds()
    ext = (hi - lo) * (1 + 2 * pad)
    c = 0.5 * (lo + hi)
    aspect = width / height
    
    if ext[0] / ext[1] > aspect: size = np.array([ext[0], ext[0] / aspect])
    else: size = np.array([ext[1] * aspect, ext[1]])
    
    return c - 0.5 * size, size


class PatternRenderer:
    def __init__(self, ctx, baked: BakedShape, palette: Palette, shade: ShadeParams | None = None) -> None:
        import moderngl

        self.prog = ctx.program(vertex_shader=load_shader("fullscreen.vert"), fragment_shader=load_shader("flow.frag"))
        self.vao = ctx.vertex_array(self.prog, [])
        self.vao.vertices = 3
        tex = baked.texture
        self.coords = ctx.texture((tex.shape[1], tex.shape[0]), 2, tex.tobytes(), dtype="f4")
        self.coords.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.coords.repeat_x = self.coords.repeat_y = False
        self.baked = baked
        shape = baked.shape
        s = self._set
        s("u_coords", 0)
        s("u_origin", tuple(map(float, baked.origin)))
        s("u_size", tuple(map(float, baked.size)))
        s("u_thickness", float(shape.thickness))
        s("u_core", tuple(map(float, shape.core)))
        self._background = (0.0, 0.0, 0.0)
        self._idle_glow = 0.0
        self._pal_u: dict | None = None # palette uniforms currently shown
        self._fade: tuple[dict, dict, float, float] | None = None # from, to, t0, dur
        self.set_palette(palette)
        self.set_shade(shade or ShadeParams())
        self.set_colors((0, 0, 0), (0.07, 0.07, 0.08))

    def release(self) -> None:
        for obj in (self.vao, self.prog, self.coords): obj.release()

    def _set(self, name: str, value) -> None:
        # GLSL drops unused uniforms, and moderngl raises on unknown names.
        u = self.prog.get(name, None)
        if u is not None: u.value = value

    def set_palette(self, palette: Palette, fade_s: float = 0.0, now: float | None = None) -> None:
        """Show a palette, optionally crossfading from the current one."""
        self.palette = palette
        target = palette.to_uniforms()
        if fade_s > 0 and self._pal_u is not None:
            t0 = time.perf_counter() if now is None else now
            self._fade = (self._pal_u, target, t0, fade_s)
        else:
            self._fade = None
            self._apply_palette(target)

    def _apply_palette(self, u: dict) -> None:
        self._pal_u = u
        self._set("u_base", tuple(map(float, u["u_base"])))
        if hasattr(self, "_idle_glow"): self._update_background()

    def _tick_fade(self, now: float | None = None) -> None:
        if self._fade is None: return
        a, b, t0, dur = self._fade
        t = ((time.perf_counter() if now is None else now) - t0) / dur
        if t >= 1.0:
            self._fade = None
            self._apply_palette(b)
        else:
            self._apply_palette(blend_uniforms(a, b, t))

    def set_shade(self, p: ShadeParams) -> None:
        for name in ("i_full", "white_hi", "white_band", "white_gamma"):
            self._set(f"u_{name}", float(getattr(p, name)))

    def set_colors(self, background, frame, idle_glow: float | None = None) -> None:
        """background: sRGB idle-screen colour. frame: sRGB surrounding material.

        idle_glow > 0 replaces the background with a dim version of the
        palette's colour, i.e. the screen idles at a low intensity of its own
        colour rather than a separate colour.
        """
        self._background = tuple(background)
        if idle_glow is not None: self._idle_glow = float(idle_glow)
        self._set("u_frame", _lab(frame))
        self._update_background()

    def _update_background(self) -> None:
        if self._idle_glow > 0:
            # Scaling L, a and b together dims a colour toward black while
            # keeping its hue: the same colour at lower emission. It follows
            # the crossfade, so the idle colour changes with the track.
            lab = np.asarray(self._pal_u["u_base"]) * self._idle_glow
            self._set("u_background", tuple(float(v) for v in lab))
        else:
            self._set("u_background", _lab(self._background))

    def render(self, st: PatternState, width: int, height: int, now: float | None = None) -> None:
        self._tick_fade(now)
        vmin, vsize = fit_view(self.baked.shape, width, height)
        s = self._set
        s("u_resolution", (float(width), float(height)))
        s("u_view_min", tuple(map(float, vmin)))
        s("u_view_size", tuple(map(float, vsize)))
        s("u_edge_margin", float(st.edge_margin))
        for name in ("time", "scale", "branch_freq", "balance", "branching", "level", "reach", "warp", "pulse", "ripple_width"): s(f"u_{name}", float(getattr(st, name)))
        s("u_npulses", [tuple(map(float, a)) for a in st.noise_pulse_arrays()])
        rip, amp = st.ripple_arrays()
        s("u_ripple_radius", [float(r) for r in rip])
        s("u_ripple_amp", [float(a) for a in amp])
        s("u_ripple_count", min(len(st.ripples), len(amp)))
        self.coords.use(0)
        self.vao.render()


def read_pixels(fbo, width: int, height: int) -> np.ndarray:
    # explicit viewport: after a window resize the framebuffer's stored
    # viewport can still be the old size, which would scramble the rows
    data = np.frombuffer(fbo.read(viewport=(0, 0, width, height), components=3), np.uint8).reshape(height, width, 3)
    return np.flipud(data)  # GL rows start at the bottom


class DisplayWindow:
    """
    The display window (glfw + moderngl). Owns the GL context and swaps the
    renderer when the shape changes; colours and palette are applied without
    rebuilding anything.
    """

    def __init__(self, size: tuple[int, int] = (720, 720), title: str = "reactance") -> None:
        try:
            import glfw
            import moderngl
        except ImportError as e:
            raise RuntimeError("install the 'render' extra: pip install -e .[render]") from e
        self.glfw = glfw
        
        if not glfw.init(): raise RuntimeError("glfw failed to initialise")
        
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        
        self.win = glfw.create_window(size[0], size[1], title, None, None)
        if not self.win:
            glfw.terminate()
            raise RuntimeError("could not create an OpenGL 3.3 window")
        
        glfw.make_context_current(self.win)
        glfw.swap_interval(1)
        glfw.set_key_callback(self.win, self._on_key)
        self.ctx = moderngl.create_context()
        self.renderer: PatternRenderer | None = None

    def _on_key(self, _w, key, _sc, action, _mods):
        if action == self.glfw.PRESS and key == self.glfw.KEY_ESCAPE: self.glfw.set_window_should_close(self.win, True)

    def size(self) -> tuple[int, int]:
        return self.glfw.get_framebuffer_size(self.win)

    def set_shape(self, baked: BakedShape, palette: Palette, background, frame, idle_glow: float = 0.0) -> None:
        if self.renderer is not None: self.renderer.release()
        self.renderer = PatternRenderer(self.ctx, baked, palette)
        self.renderer.set_colors(background, frame, idle_glow)

    def should_close(self) -> bool:
        return self.glfw.window_should_close(self.win)

    def draw(self, st: PatternState) -> None:
        glfw = self.glfw
        w, h = glfw.get_framebuffer_size(self.win)
        self.ctx.detect_framebuffer().use()
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.clear()
        if w > 0 and h > 0 and self.renderer is not None: self.renderer.render(st, w, h)
        glfw.swap_buffers(self.win)
        glfw.poll_events()

    def close(self) -> None:
        self.glfw.terminate()


class OfflineRenderer:
    """
    Headless rendering (stills, tests) on a standalone OpenGL context: EGL on
    Linux, a hidden window on Windows.
    """

    def __init__(self, baked: BakedShape, palette: Palette, width: int, height: int,
        background=(0.0, 0.0, 0.0), frame=(0.07, 0.07, 0.08),
        idle_glow: float = 0.0) -> None:
        import moderngl

        try: self.ctx = moderngl.create_context(standalone=True, require=330)
        except Exception: self.ctx = moderngl.create_context(standalone=True, backend="egl", require=330)
        
        self.width, self.height = width, height
        self.fbo = self.ctx.simple_framebuffer((width, height))
        self.renderer = PatternRenderer(self.ctx, baked, palette)
        self.renderer.set_colors(background, frame, idle_glow)

    def frame(self, st: PatternState) -> np.ndarray:
        self.fbo.use()
        self.fbo.clear()
        self.ctx.viewport = (0, 0, self.width, self.height)
        self.renderer.render(st, self.width, self.height)
        return read_pixels(self.fbo, self.width, self.height)
