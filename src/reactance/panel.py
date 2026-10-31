"""
    panel.py
    
    This is the control panel, te second window which allows settings to be changed.

    Control panel is seperated from the renderer, they share the settings object.
"""

from __future__ import annotations
import tkinter as tk
from tkinter import colorchooser, filedialog, ttk
from .colour import Palette
from .motion import SIZE_MAX
from .settings import DISPLAY_KEYS, REACTANCE_KEYS, Settings
from .shape import CIRCLE, FULLSCREEN


def _hex(rgb) -> str:
    r, g, b = (max(0, min(255, round(c * 255))) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


class ControlPanel:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.alive = True
        root = self.root = tk.Tk()
        root.title("Control Panel")
        root.protocol("WM_DELETE_WINDOW", self._close)
        root.resizable(False, True)
        pad = {"padx": 10, "pady": 4}
        # Tabs keep the window short enough for a 1080p screen with display
        # scaling; the input meters stay visible below them.
        outer = ttk.Frame(root, padding=8)
        outer.grid(sticky="nsew")
        notebook = ttk.Notebook(outer)
        notebook.grid(row=0, column=0, sticky="nsew")
        tab_react = ttk.Frame(notebook, padding=6)
        tab_display = ttk.Frame(notebook, padding=6)
        notebook.add(tab_react, text="Reactance")
        notebook.add(tab_display, text="Display")
        row = 0
        frm = tab_react

        def section(text):
            nonlocal row
            ttk.Label(frm, text=text, font=("Segoe UI", 10, "bold")).grid(
                row=row, column=0, columnspan=3, sticky="w", pady=(10, 2))
            row += 1

        # ======== Reactance tab ========
        # ---- reactivity ----
        row = self._slider(frm, row, "Low reactance (gradual)", "High (immediate)", "energy", 0.0, 1.0)
        row = self._slider(frm, row, "Small blob", "Large blob", "size", 0.0, SIZE_MAX)
        row = self._slider(frm, row, "Mostly low intensity", "Mostly high", "balance", 0.0, 1.0)
        row = self._slider(frm, row, "No pulses", "Frequent pulses", "branching", 0.0, 1.0)
        row = self._slider(frm, row, "Snare waves: off", "Strong", "waves", 0.0, 1.0)
        ttk.Button(frm, text="Reset to defaults", command=lambda: self._reset(REACTANCE_KEYS)).grid(
            row=row, column=0, columnspan=3, sticky="e", padx=10, pady=(12, 4))
        row += 1

        # ======== Display tab ========
        frm, row = tab_display, 0
        # ---- shape ----
        # The circle is the built-in shape; anything else is a custom SVG.
        # Fullscreen (the whole window) isn't an SVG, so it stays a choice.
        section("Shape")
        btns = ttk.Frame(frm)
        btns.grid(row=row, column=0, columnspan=3, sticky="w", padx=6, pady=2)
        ttk.Button(btns, text="Circle", command=lambda: self._set_shape(CIRCLE)).grid(row=0, column=0, padx=4)
        ttk.Button(btns, text="Fullscreen", command=lambda: self._set_shape(FULLSCREEN)).grid(row=0, column=1, padx=4)
        ttk.Button(btns, text="Custom SVG…", command=self._browse_shape).grid(row=0, column=2, padx=4)
        row += 1

        # ---- palette ----
        section("Now playing")
        self.swatch = tk.Canvas(frm, height=22, width=22, highlightthickness=0)
        self.swatch.grid(row=row, column=0, sticky="w", padx=10, pady=4)
        self.track_label = ttk.Label(frm, text="", width=46, foreground="#555")
        self.track_label.grid(row=row, column=1, columnspan=2, sticky="w")
        row += 1

        # ---- colours ----
        section("Colours")
        self.bg_btn = self._color_row(frm, row, "Background (idle screen)", "background")
        row += 1
        self.frame_btn = self._color_row(frm, row, "Frame / edge-fade material", "frame")
        row += 1
        self.glow_on = tk.BooleanVar(value=settings.idle_glow > 0)
        ttk.Checkbutton(frm, text="Idle glow (dim palette colour instead of background)",
            variable=self.glow_on, command=self._apply_glow).grid(
            row=row, column=0, columnspan=3, sticky="w", **pad)
        row += 1
        self.glow = tk.DoubleVar(value=settings.idle_glow or 0.25)
        ttk.Scale(frm, from_=0.05, to=0.6, variable=self.glow, length=300,
            command=lambda _v: self._apply_glow()).grid(
            row=row, column=0, columnspan=3, sticky="we", **pad)
        row += 1

        section("Edge fade")
        ttk.Label(frm, text="Width").grid(row=row, column=0, sticky="w", padx=10)
        self.margin_lbl = ttk.Label(frm, text="")
        self.margin_lbl.grid(row=row, column=2, sticky="e", padx=10)
        row += 1
        self.margin = tk.DoubleVar(value=settings.edge_margin)
        ttk.Scale(frm, from_=0.0, to=0.6, variable=self.margin, length=300,
            command=lambda v: self._set_margin(float(v))).grid(
            row=row, column=0, columnspan=3, sticky="we", **pad)
        row += 1

        ttk.Button(frm, text="Reset to defaults", command=lambda: self._reset(DISPLAY_KEYS)).grid(
            row=row, column=0, columnspan=3, sticky="e", padx=10, pady=(12, 4))
        row += 1
        self.refresh()

        # ======== always visible ========
        frm, row = ttk.Frame(outer), 0
        frm.grid(row=1, column=0, sticky="we")
        # ---- meters ----
        section("Input")
        self.device_lbl = ttk.Label(frm, text="Listening to: …")
        self.device_lbl.grid(row=row, column=0, columnspan=3, sticky="w", padx=10)
        row += 1
        self.meters = tk.Canvas(frm, height=126, width=300, highlightthickness=0, background="#16161a")
        self.meters.grid(row=row, column=0, columnspan=3, sticky="we", **pad)
        row += 1

    # ------------------------------------------------------------ helpers --
    def _close(self) -> None:
        self.alive = False

    def _set(self, key: str, value) -> None:
        setattr(self.settings, key, value)

    def _slider(self, frm, row: int, left: str, right: str, key: str, lo: float, hi: float) -> int:
        ttk.Label(frm, text=left).grid(row=row, column=0, sticky="w", padx=10)
        ttk.Label(frm, text=right).grid(row=row, column=2, sticky="e", padx=10)
        var = tk.DoubleVar(value=getattr(self.settings, key))
        setattr(self, f"_{key}_var", var)
        ttk.Scale(frm, from_=lo, to=hi, variable=var, length=300, command=lambda v: self._set(key, float(v))).grid(
            row=row + 1, column=0, columnspan=3, sticky="we", padx=10, pady=4)
        return row + 2

    # ---- shape ----
    def _set_shape(self, shape: str) -> None:
        self._set("shape", shape)

    # ---- reset ----
    def _reset(self, keys) -> None:
        self.settings.reset(keys)
        self.refresh()

    def refresh(self) -> None:
        """Make every control show what `settings` currently holds."""
        s = self.settings
        
        for key in REACTANCE_KEYS: getattr(self, f"_{key}_var").set(getattr(s, key))
        
        self.bg_btn.config(bg=_hex(s.background))
        self.frame_btn.config(bg=_hex(s.frame))
        self.glow_on.set(s.idle_glow > 0)
        if s.idle_glow > 0: self.glow.set(s.idle_glow)
        self.margin.set(s.edge_margin)
        self._set_margin(s.edge_margin)

    def _apply_glow(self) -> None:
        self._set("idle_glow", float(self.glow.get()) if self.glow_on.get() else 0.0)

    def _set_margin(self, v: float) -> None:
        self._set("edge_margin", v)
        self.margin_lbl.config(text="hard edge" if v < 0.01 else f"{v:.0%} of depth")

    def _browse_shape(self) -> None:
        path = filedialog.askopenfilename(title="Shape", filetypes=[("SVG", "*.svg")])
        if path: self._set_shape(path)

    def show_track(self, track, status: str | None = None) -> None:
        if status is not None: self._np_status = status
        if track is None:
            text = getattr(self, "_np_status", "nothing playing")
            if status is None: text = "nothing playing"
        else:
            state = "♪" if track.playing else "❚❚"
            art = "" if track.art else "  (no cover art)"
            text = f"{state} {track.label()}{art}"
        if len(text) > 60: text = text[:57] + "…"
        self.track_label.config(text=text)

    def _color_row(self, frm, row: int, label: str, key: str) -> tk.Button:
        ttk.Label(frm, text=label).grid(row=row, column=0, columnspan=2, sticky="w", padx=10)
        btn = tk.Button(frm, width=6, relief="flat", bg=_hex(getattr(self.settings, key)))

        def pick():
            _, hx = colorchooser.askcolor(color=_hex(getattr(self.settings, key)), title=label)
            if hx:
                rgb = tuple(int(hx[i:i + 2], 16) / 255 for i in (1, 3, 5))
                self._set(key, rgb)
                btn.config(bg=hx)

        btn.config(command=pick)
        btn.grid(row=row, column=2, sticky="e", padx=10, pady=2)
        return btn

    # ------------------------------------------------------------- public --
    def show_palette(self, pal: Palette) -> None:
        """The cover's colour, as a swatch next to the track name."""
        self.swatch.delete("all")
        self.swatch.create_rectangle(1, 1, 21, 21, fill=pal.base.hex, outline="")

    def show_input(self, m: dict[str, float]) -> None:
        """m: the mapper's smoothed envelopes (see AudioMapper.meters)."""
        c = self.meters
        c.delete("all")
        for i, name in enumerate(("presence", "loud", "bass", "body", "lead", "wide", "cymbal")):
            y = 8 + i * 17
            c.create_text(8, y + 6, text=name, anchor="w", fill="#9a9aa2", font=("Segoe UI", 8))
            v = max(0.0, min(1.0, m.get(name, 0.0)))
            c.create_rectangle(70, y, 70 + 220 * v, y + 11, fill="#d9d9e0", width=0)

    def show_device(self, name: str) -> None:
        text = f"Listening to: {name}"
        if self.device_lbl.cget("text") != text:self.device_lbl.config(text=text)

    def pump(self) -> bool:
        if self.alive:
            try: self.root.update()
            except tk.TclError: self.alive = False
        return self.alive

    def destroy(self) -> None:
        try: self.root.destroy()
        except tk.TclError: pass
