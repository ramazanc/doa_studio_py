"""Reusable Tk building blocks: form rows, panels and embedded figures."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Iterable, Optional, Sequence

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure                              # noqa: E402

from . import theme                                               # noqa: E402


# =====================================================================
# Panels
# =====================================================================
class Panel(ttk.LabelFrame):
    """A titled panel with consistent padding."""

    def __init__(self, master, title: str, padding=(8, 6, 8, 8), **kw):
        super().__init__(master, text=" " + title + " ", padding=padding, **kw)


class PlotPanel(Panel):
    """A titled panel holding one embedded Matplotlib figure.

    ``self.fig`` is the Figure; call :meth:`draw` after mutating it.  The
    canvas is created once and reused, so a redraw never rebuilds widgets.
    """

    def __init__(self, master, title: str, figsize=(4.0, 3.0), dpi=100, **kw):
        super().__init__(master, title, padding=(4, 2, 4, 4), **kw)
        self.fig = Figure(figsize=figsize, dpi=dpi, facecolor=theme.PANEL_BG)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        widget = self.canvas.get_tk_widget()
        widget.configure(background=theme.PANEL_BG, highlightthickness=0)
        widget.pack(fill="both", expand=True)

    def draw(self):
        try:
            self.fig.tight_layout(pad=0.9)
        except Exception:
            # tight_layout can fail on a degenerate axes (e.g. an empty
            # compass before the first run); the plot is still valid.
            pass
        self.canvas.draw_idle()

    def clear(self):
        self.fig.clear()


# =====================================================================
# Form helpers
# =====================================================================
class FormGrid(ttk.Frame):
    """A label/control grid that keeps rows aligned.

    Four underlying columns (label, control, label, control) so that short
    numeric fields can be paired on one row with :meth:`add_pair`, which
    matters on a laptop screen where vertical space is the scarce resource.
    """

    def __init__(self, master, label_width: int = 18, pady=(0, 3), **kw):
        super().__init__(master, **kw)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(3, weight=1)
        self._row = 0
        self._label_width = label_width
        self._pady = pady

    def _label(self, text: str, column: int, width=None) -> ttk.Label:
        lbl = ttk.Label(self, text=text, anchor="w",
                        width=width if width is not None else self._label_width)
        lbl.grid(row=self._row, column=column, sticky="w", pady=self._pady,
                 padx=(0, 6) if column == 0 else (10, 6))
        return lbl

    def add(self, label: str, widget: tk.Widget) -> ttk.Label:
        """One label plus one control spanning the rest of the row."""
        lbl = self._label(label, 0)
        widget.grid(row=self._row, column=1, columnspan=3, sticky="ew",
                    pady=self._pady)
        self._row += 1
        return lbl

    def add_pair(self, label_a: str, widget_a: tk.Widget,
                 label_b: str, widget_b: tk.Widget, label_b_width: int = 13):
        """Two label/control pairs sharing one row."""
        lbl_a = self._label(label_a, 0)
        widget_a.grid(row=self._row, column=1, sticky="ew", pady=self._pady)
        lbl_b = self._label(label_b, 2, width=label_b_width)
        widget_b.grid(row=self._row, column=3, sticky="ew", pady=self._pady)
        self._row += 1
        return lbl_a, lbl_b

    def add_full(self, widget: tk.Widget):
        widget.grid(row=self._row, column=0, columnspan=4, sticky="ew",
                    pady=self._pady)
        self._row += 1

    def add_separator(self, pady=(6, 6)):
        sep = ttk.Separator(self, orient="horizontal")
        sep.grid(row=self._row, column=0, columnspan=4, sticky="ew", pady=pady)
        self._row += 1


def entry(master, textvariable, width=14, on_change: Optional[Callable] = None,
          **kw) -> ttk.Entry:
    """An entry that reports edits on Return and on focus-out."""
    e = ttk.Entry(master, textvariable=textvariable, width=width, **kw)
    if on_change is not None:
        e.bind("<Return>", lambda _e: on_change())
        e.bind("<FocusOut>", lambda _e: on_change())
    return e


def combo(master, textvariable, values: Sequence[str], width=18,
          on_change: Optional[Callable] = None, **kw) -> ttk.Combobox:
    c = ttk.Combobox(master, textvariable=textvariable, values=list(values),
                     state="readonly", width=width, **kw)
    if on_change is not None:
        c.bind("<<ComboboxSelected>>", lambda _e: on_change())
    return c


# =====================================================================
# Value coercion
# =====================================================================
def read_float(var: tk.Variable, default: float,
               lo: Optional[float] = None, hi: Optional[float] = None) -> float:
    """Read a possibly half-typed entry as a float, clamped into range."""
    try:
        v = float(str(var.get()).strip().replace(",", "."))
    except (ValueError, tk.TclError):
        return default
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v


def read_int(var: tk.Variable, default: int,
             lo: Optional[int] = None, hi: Optional[int] = None) -> int:
    return int(round(read_float(var, float(default), lo, hi)))


# =====================================================================
# Read-only text
# =====================================================================
class ReadOnlyText(tk.Text):
    """A Text widget that looks like a note field and cannot be typed into."""

    # tk.Text defaults to 80 characters, which would make this widget - not
    # the form above it - decide how wide the whole left column has to be.
    # A small floor lets the geometry manager set the real width instead.
    DEFAULT_WIDTH = 26

    def __init__(self, master, height=8, wrap="word", font=None,
                 width=DEFAULT_WIDTH, **kw):
        super().__init__(master, height=height, width=width, wrap=wrap,
                         relief="solid", borderwidth=1,
                         highlightthickness=0, padx=8, pady=6,
                         background=theme.PANEL_BG, foreground=theme.TEXT,
                         insertwidth=0, **kw)
        if font is not None:
            self.configure(font=font)
        self.configure(state="disabled", cursor="arrow")

    def set_text(self, text: str):
        self.configure(state="normal")
        self.delete("1.0", "end")
        self.insert("1.0", text)
        self.configure(state="disabled")


class ScrolledReadOnlyText(ttk.Frame):
    """:class:`ReadOnlyText` with a vertical scrollbar.

    Used wherever the content length is not known in advance - a limitation
    explanation can be one line or a full paragraph.
    """

    def __init__(self, master, height=6, font=None, **kw):
        super().__init__(master)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.text = ReadOnlyText(self, height=height, font=font, **kw)
        self.text.grid(row=0, column=0, sticky="nsew")
        self._sb = ttk.Scrollbar(self, orient="vertical", command=self.text.yview)
        self._sb.grid(row=0, column=1, sticky="ns")
        self.text.configure(yscrollcommand=self._sb.set)

    def set_text(self, text: str):
        self.text.set_text(text)

    # Delegate the Text API the callers actually use.
    def configure_tag(self, name, **kw):
        self.text.tag_configure(name, **kw)

    def replace(self, chunks):
        """Rewrite the whole box from ``(text, tag)`` pairs."""
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        for body, tag in chunks:
            self.text.insert("end", body, tag)
        self.text.configure(state="disabled")
        self.text.yview_moveto(0.0)


def note_box(master, lines: Iterable[str], font=None, height=None) -> ReadOnlyText:
    """A static explanatory note, sized to its content."""
    lines = list(lines)
    box = ReadOnlyText(master, height=height or max(4, len(lines)), font=font)
    box.set_text("\n".join(lines))
    return box


# =====================================================================
# Tables
# =====================================================================
def make_table(master, columns: Sequence[tuple], height: int = 6) -> ttk.Treeview:
    """A Treeview configured as a compact read-only table.

    ``columns`` holds ``(heading, width, anchor)`` or
    ``(heading, width, anchor, stretch)`` tuples.  Pinning the narrow numeric
    columns with ``stretch=False`` keeps them from being widened past the
    panel edge when only one column should absorb the slack.
    """
    names = ["c%d" % i for i in range(len(columns))]
    tv = ttk.Treeview(master, columns=names, show="headings", height=height,
                      selectmode="browse")
    for name, spec in zip(names, columns):
        heading, width, anchor = spec[0], spec[1], spec[2]
        stretch = spec[3] if len(spec) > 3 else True
        tv.heading(name, text=heading, anchor=anchor)
        tv.column(name, width=width, minwidth=width, anchor=anchor,
                  stretch=stretch)
    return tv


def fill_table(tv: ttk.Treeview, rows: Sequence[Sequence[str]],
               tags: Optional[Sequence[str]] = None):
    tv.delete(*tv.get_children())
    for i, row in enumerate(rows):
        tag = (tags[i],) if tags else ()
        tv.insert("", "end", values=list(row), tags=tag)
