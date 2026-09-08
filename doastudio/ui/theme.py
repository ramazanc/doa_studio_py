"""Colours, fonts and ttk styling for the whole app.

Keeping every visual constant here means the tabs contain layout code only,
and a change of palette is a one-file edit.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

# ---------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------
NAVY = "#193d66"            # header banner
NAVY_LIGHT = "#cddceb"      # header subtitle text
NAVY_FAINT = "#b3c8de"      # header right-hand text
ACCENT = "#2273bf"          # primary button
ACCENT_DARK = "#1a5c9a"     # primary button, pressed
ACCENT_LIGHT = "#e8f1fa"    # selection wash

BG = "#f0f0f0"              # window background
PANEL_BG = "#ffffff"        # panel / plot background
BORDER = "#c8ccd0"
TEXT = "#1c1c1c"
TEXT_MUTED = "#5a5f66"
TEXT_FAINT = "#8a9099"

OK_GREEN = "#2f7d32"
WARN_AMBER = "#a8690a"
CRIT_RED = "#b3261e"

# Severity colours for the limitation panel, keyed by the core's severity
# strings so the UI never re-derives them.
SEVERITY_FG = {"crit": CRIT_RED, "warn": WARN_AMBER, "info": "#1a4a7a"}

# ---------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------
FAMILY = "Segoe UI"
MONO = "Consolas"


def fonts() -> dict:
    """Named fonts, created once a Tk root exists."""
    return {
        "title": tkfont.Font(family=FAMILY, size=16, weight="bold"),
        "subtitle": tkfont.Font(family=FAMILY, size=10),
        "heading": tkfont.Font(family=FAMILY, size=10, weight="bold"),
        "body": tkfont.Font(family=FAMILY, size=9),
        "small": tkfont.Font(family=FAMILY, size=8),
        "italic": tkfont.Font(family=FAMILY, size=8, slant="italic"),
        "button": tkfont.Font(family=FAMILY, size=11, weight="bold"),
        "mono": tkfont.Font(family=MONO, size=9),
        "mono_small": tkfont.Font(family=MONO, size=8),
        # comparison page
        "doc_h1": tkfont.Font(family=FAMILY, size=15, weight="bold"),
        "doc_h2": tkfont.Font(family=FAMILY, size=11, weight="bold"),
        "doc_body": tkfont.Font(family=FAMILY, size=10),
        "doc_bold": tkfont.Font(family=FAMILY, size=10, weight="bold"),
        "doc_italic": tkfont.Font(family=FAMILY, size=10, slant="italic"),
        "doc_mono": tkfont.Font(family=MONO, size=9),
        "doc_nav": tkfont.Font(family=FAMILY, size=9),
    }


def apply_style(root: tk.Misc) -> ttk.Style:
    """Install the ttk theme used by every tab."""
    style = ttk.Style(root)
    # 'clam' is the only built-in theme that honours background colours on
    # every platform, which the panels and the notebook both rely on.
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    style.configure(".", background=BG, foreground=TEXT, font=(FAMILY, 9))
    style.configure("TFrame", background=BG)
    style.configure("Panel.TFrame", background=PANEL_BG)
    style.configure("Header.TFrame", background=NAVY)

    style.configure("TLabel", background=BG, foreground=TEXT)
    style.configure("Panel.TLabel", background=PANEL_BG, foreground=TEXT)
    style.configure("Muted.TLabel", background=BG, foreground=TEXT_MUTED,
                    font=(FAMILY, 8, "italic"))
    style.configure("Status.TLabel", background=BG, foreground=TEXT_MUTED,
                    font=(FAMILY, 8))
    style.configure("HeaderTitle.TLabel", background=NAVY, foreground="#ffffff",
                    font=(FAMILY, 16, "bold"))
    style.configure("HeaderSub.TLabel", background=NAVY, foreground=NAVY_LIGHT,
                    font=(FAMILY, 10))
    style.configure("HeaderRight.TLabel", background=NAVY, foreground=NAVY_FAINT,
                    font=(FAMILY, 9))

    style.configure("TLabelframe", background=BG, bordercolor=BORDER,
                    relief="solid", borderwidth=1)
    style.configure("TLabelframe.Label", background=BG, foreground=NAVY,
                    font=(FAMILY, 9, "bold"))

    style.configure("TCheckbutton", background=BG, foreground=TEXT)
    style.map("TCheckbutton", background=[("active", BG)])

    style.configure("TNotebook", background=BG, borderwidth=0, tabmargins=(6, 4, 6, 0))
    style.configure("TNotebook.Tab", padding=(18, 7), font=(FAMILY, 9, "bold"),
                    background="#dfe3e8", foreground=TEXT_MUTED)
    style.map("TNotebook.Tab",
              background=[("selected", PANEL_BG)],
              foreground=[("selected", NAVY)])

    style.configure("Accent.TButton", background=ACCENT, foreground="#ffffff",
                    font=(FAMILY, 11, "bold"), borderwidth=0, padding=(10, 8))
    style.map("Accent.TButton",
              background=[("pressed", ACCENT_DARK), ("active", ACCENT_DARK),
                          ("disabled", "#9db8d1")])

    style.configure("Small.TButton", font=(FAMILY, 8), padding=(6, 3))

    style.configure("TCombobox", fieldbackground=PANEL_BG, background=PANEL_BG)
    style.configure("TEntry", fieldbackground=PANEL_BG)

    style.configure("Treeview", background=PANEL_BG, fieldbackground=PANEL_BG,
                    rowheight=20, font=(FAMILY, 8))
    style.configure("Treeview.Heading", font=(FAMILY, 8, "bold"),
                    background="#e4e8ec", foreground=NAVY)
    style.map("Treeview", background=[("selected", ACCENT_LIGHT)],
              foreground=[("selected", TEXT)])

    style.configure("TProgressbar", background=ACCENT, troughcolor="#dfe3e8",
                    borderwidth=0)

    return style


# ---------------------------------------------------------------------
# Matplotlib
# ---------------------------------------------------------------------
def mpl_rc() -> dict:
    """rcParams that make embedded figures match the Tk chrome."""
    return {
        "figure.facecolor": PANEL_BG,
        "axes.facecolor": PANEL_BG,
        "axes.edgecolor": "#9aa0a6",
        "axes.labelcolor": TEXT,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.color": "#dcdfe3",
        "grid.linewidth": 0.7,
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "legend.framealpha": 0.92,
        "font.family": "sans-serif",
        "font.sans-serif": [FAMILY, "DejaVu Sans"],
        "figure.autolayout": False,
    }
