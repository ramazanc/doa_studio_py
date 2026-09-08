"""The main window: header banner, four tabs and a status bar."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import matplotlib

from . import theme
from .tab_comparison import ComparisonTab
from .tab_single import SingleRunTab
from .tab_snr import SnrTab
from .tab_sweep import SweepTab

WINDOW_TITLE = "DoA Studio  |  RF Direction Finding"
MIN_SIZE = (1180, 720)
DEFAULT_SIZE = (1520, 940)


class DoAStudioApp(tk.Tk):
    """Six DoA estimators, three array geometries, one window."""

    def __init__(self):
        super().__init__()
        self.title(WINDOW_TITLE)
        self.configure(background=theme.BG)
        self.minsize(*MIN_SIZE)
        self._centre(*DEFAULT_SIZE)

        matplotlib.rcParams.update(theme.mpl_rc())
        self.f = theme.fonts()
        theme.apply_style(self)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self._build_header()
        self._build_tabs()
        self._build_status()

        # Show something immediately, exactly as the MATLAB app did.
        self.after(60, self.single.run_single)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # -----------------------------------------------------------------
    def _centre(self, w: int, h: int):
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w = min(w, sw - 60)
        h = min(h, sh - 100)
        x = max(0, (sw - w) // 2)
        y = max(0, (sh - h) // 2 - 20)
        self.geometry("%dx%d+%d+%d" % (w, h, x, y))

    def _build_header(self):
        hdr = ttk.Frame(self, style="Header.TFrame", padding=(20, 10, 20, 10))
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.columnconfigure(0, weight=14)
        hdr.columnconfigure(1, weight=12)
        hdr.columnconfigure(2, weight=10)

        ttk.Label(hdr, text="DoA Studio   |   RF Direction Finding",
                  style="HeaderTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(hdr, text="Array Signal Processing   |   Direction of Arrival "
                            "Estimation",
                  style="HeaderSub.TLabel").grid(row=0, column=1)
        ttk.Label(hdr, text="Python  |  NumPy · SciPy · Matplotlib",
                  style="HeaderRight.TLabel").grid(row=0, column=2, sticky="e")

    def _build_tabs(self):
        self.nb = ttk.Notebook(self)
        self.nb.grid(row=1, column=0, sticky="nsew", padx=8, pady=(8, 4))

        self.single = SingleRunTab(self.nb, self)
        self.sweep = SweepTab(self.nb, self)
        self.snr = SnrTab(self.nb, self)
        self.comparison = ComparisonTab(self.nb, self)

        self.nb.add(self.single, text="Single Run")
        self.nb.add(self.sweep, text="Sweep Study")
        self.nb.add(self.snr, text="SNR Performance")
        self.nb.add(self.comparison, text="Comparison")

    def _build_status(self):
        bar = ttk.Frame(self, padding=(12, 3, 12, 5))
        bar.grid(row=2, column=0, sticky="ew")
        bar.columnconfigure(0, weight=1)
        self.var_status = tk.StringVar(value="Ready")
        ttk.Label(bar, textvariable=self.var_status,
                  style="Status.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(bar, text="scan -90…+90° · step 0.5° · positions in wavelengths",
                  style="Status.TLabel").grid(row=0, column=1, sticky="e")

    # -----------------------------------------------------------------
    def set_status(self, text: str):
        self.var_status.set(text)
        self.update_idletasks()

    def _on_close(self):
        # Ask any running sweep to stop so the worker threads exit promptly.
        for tab in (self.sweep, self.snr):
            runner = getattr(tab, "runner", None)
            if runner is not None:
                runner.cancel()
        self.destroy()


def main():
    app = DoAStudioApp()
    app.mainloop()


if __name__ == "__main__":
    main()
