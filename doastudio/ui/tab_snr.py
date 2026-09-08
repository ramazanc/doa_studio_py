"""SNR Performance tab: a dedicated, always-available RMSE-vs-SNR study."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Optional

from ..core import SweepResult, sweep_values
from . import plots
from .runner import SweepRunner
from .widgets import (FormGrid, Panel, PlotPanel, combo, entry, note_box,
                      read_float, read_int)

NOTES = [
    "RMSE vs SNR is the standard way",
    "to compare DF estimators.",
    "",
    "A healthy curve has three regions:",
    "",
    "1. Below threshold. The subspace",
    "   estimators break down and RMSE",
    "   saturates near the width of the",
    "   scan sector.",
    "",
    "2. The knee. This is the threshold",
    "   SNR, and where extra elements or",
    "   snapshots buy the most.",
    "",
    "3. Asymptotic region. RMSE falls",
    "   about one decade per 20 dB,",
    "   tracking the Cramer-Rao bound.",
    "",
    "Bartlett and CDF flatten out at a",
    "floor set by beamwidth and grid",
    "step rather than by noise, so they",
    "can beat the subspace methods below",
    "threshold and lose badly above it.",
]


class SnrTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=8)
        self.app = app
        self.f = app.f
        self.result: Optional[SweepResult] = None

        self.columnconfigure(0, minsize=320, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self._build_left()
        self._build_right()

        self.runner = SweepRunner(self, self._on_progress, self._on_done)

    # -----------------------------------------------------------------
    def _build_left(self):
        left = ttk.Frame(self)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(2, weight=1)

        pn = Panel(left, "SNR Study")
        pn.grid(row=0, column=0, sticky="ew")
        pn.columnconfigure(0, weight=1)
        fg = FormGrid(pn, label_width=16)
        fg.grid(row=0, column=0, sticky="ew")

        self.var_start = tk.StringVar(value="-15")
        fg.add("SNR start (dB):", entry(fg, self.var_start))
        self.var_stop = tk.StringVar(value="25")
        fg.add("SNR stop (dB):", entry(fg, self.var_stop))
        self.var_step = tk.StringVar(value="5")
        fg.add("SNR step (dB):", entry(fg, self.var_step))
        self.var_trials = tk.StringVar(value="60")
        fg.add("Trials / point:", entry(fg, self.var_trials))
        self.var_scale = tk.StringVar(value="log")
        fg.add("Y-axis scale:", combo(fg, self.var_scale, ["log", "linear"],
                                      width=8, on_change=self.replot))

        self.pb = ttk.Progressbar(pn, mode="determinate", maximum=1.0)
        self.pb.grid(row=1, column=0, sticky="ew", pady=(6, 2))
        self.lbl_status = ttk.Label(pn, text="idle", style="Status.TLabel")
        self.lbl_status.grid(row=2, column=0, sticky="w")

        btns = ttk.Frame(left)
        btns.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        btns.columnconfigure(0, weight=1)
        self.btn_run = ttk.Button(btns, text="Run SNR Study", style="Accent.TButton",
                                  command=self.run_study)
        self.btn_run.grid(row=0, column=0, sticky="ew")
        self.btn_cancel = ttk.Button(btns, text="Stop", style="Small.TButton",
                                     command=self.cancel, state="disabled")
        self.btn_cancel.grid(row=0, column=1, sticky="ns", padx=(6, 0))

        pi = Panel(left, "Reading this chart")
        pi.grid(row=2, column=0, sticky="nsew", pady=(8, 0))
        pi.columnconfigure(0, weight=1)
        pi.rowconfigure(0, weight=1)
        note_box(pi, NOTES, font=self.f["body"]).grid(row=0, column=0, sticky="nsew")

    def _build_right(self):
        right = ttk.Frame(self)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=6)
        right.rowconfigure(1, weight=4)

        self.pnl_rmse = PlotPanel(right, "RMSE vs SNR", figsize=(8.0, 3.8))
        self.pnl_rmse.grid(row=0, column=0, sticky="nsew")

        bot = ttk.Frame(right)
        bot.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        bot.rowconfigure(0, weight=1)
        bot.columnconfigure(0, weight=1)
        bot.columnconfigure(1, weight=1)

        self.pnl_bias = PlotPanel(bot, "Bias vs SNR", figsize=(4.0, 2.6))
        self.pnl_bias.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.pnl_res = PlotPanel(bot, "Resolution Probability vs SNR",
                                 figsize=(4.0, 2.6))
        self.pnl_res.grid(row=0, column=1, sticky="nsew")

        for pnl, fn in ((self.pnl_rmse, plots.plot_sweep_rmse),
                        (self.pnl_bias, plots.plot_sweep_bias),
                        (self.pnl_res, plots.plot_sweep_resolution)):
            fn(pnl.fig, None)
            pnl.draw()

    # -----------------------------------------------------------------
    def run_study(self):
        if self.runner.busy:
            return
        cfg = self.app.single.scenario()
        if not cfg.keys:
            messagebox.showwarning(
                "Nothing to run",
                "Select at least one algorithm on the Single Run tab.", parent=self)
            return

        values = sweep_values(read_float(self.var_start, -15.0),
                              read_float(self.var_stop, 25.0),
                              read_float(self.var_step, 5.0))
        if values.size == 0:
            messagebox.showwarning("Bad range",
                                   "SNR start/stop/step define no points.",
                                   parent=self)
            return

        trials = read_int(self.var_trials, 60, 1, 5000)
        self._cfg = cfg
        self.btn_run.configure(state="disabled")
        self.btn_cancel.configure(state="normal")
        self.pb.configure(value=0.0)
        self.lbl_status.configure(text="starting ...")
        self.app.set_status("SNR study running: %d points x %d trials"
                            % (values.size, trials))
        self.runner.start(cfg, "snr", values, trials, cfg.keys)

    def cancel(self):
        self.runner.cancel()
        self.lbl_status.configure(text="stopping ...")

    def _on_progress(self, frac: float, text: str):
        self.pb.configure(value=max(0.0, min(1.0, frac)))
        self.lbl_status.configure(text=text)

    def _on_done(self, S: Optional[SweepResult], error: Optional[str]):
        self.btn_run.configure(state="normal")
        self.btn_cancel.configure(state="disabled")
        if error is not None:
            self.lbl_status.configure(text="failed")
            self.app.set_status("SNR study failed")
            messagebox.showerror("SNR study failed", error, parent=self)
            return

        self.result = S
        self.replot()
        done = "done" if S.completed else "stopped early"
        self.lbl_status.configure(text="%s: %d pts x %d trials"
                                  % (done, S.values.size, S.trials))
        self.app.set_status("SNR study %s  |  %d trials per point" % (done, S.trials))

    # -----------------------------------------------------------------
    def replot(self):
        S = self.result
        title = None
        if S is not None:
            cfg = getattr(self, "_cfg", None)
            if cfg is not None:
                title = ("DoA RMSE vs SNR   (%d trials/point, N = %d, M = %d, K = %d)"
                         % (S.trials, cfg.N, cfg.M, cfg.K))
        plots.plot_sweep_rmse(self.pnl_rmse.fig, S, yscale=self.var_scale.get(),
                              title=title)
        self.pnl_rmse.draw()
        plots.plot_sweep_bias(self.pnl_bias.fig, S)
        self.pnl_bias.draw()
        plots.plot_sweep_resolution(self.pnl_res.fig, S,
                                    title="Resolution probability")
        self.pnl_res.draw()
