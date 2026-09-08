"""Sweep Study tab: Monte-Carlo RMSE against any one scenario parameter."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Optional

import numpy as np

from ..core import (SWEEP_DEFAULTS, SWEEP_LABELS, SweepResult, sweep_values)
from . import plots
from .runner import SweepRunner
from .widgets import (FormGrid, Panel, PlotPanel, combo, entry, fill_table,
                      make_table, note_box, read_float, read_int)

SWEEP_KEY_BY_LABEL = {v: k for k, v in SWEEP_LABELS.items()}

NOTES = [
    "The sweep takes the scenario on the",
    "Single Run tab as its baseline and",
    "replaces only the swept parameter.",
    "",
    "Trials/point here is independent of",
    "the single-run trial count.",
    "",
    "Sweep points that are physically",
    "impossible (K >= M) are skipped and",
    "appear as gaps in the curves.",
    "",
    "Resolution probability is the",
    "fraction of trials in which every",
    "source was matched inside the",
    "tolerance: half the true separation,",
    "capped at 5 deg. It is the headline",
    "metric for the source-count sweep.",
]


class SweepTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=8)
        self.app = app
        self.f = app.f
        self.result: Optional[SweepResult] = None

        self.columnconfigure(0, minsize=350, weight=0)
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
        left.rowconfigure(3, weight=1)

        pn = Panel(left, "Sweep Configuration")
        pn.grid(row=0, column=0, sticky="ew")
        pn.columnconfigure(0, weight=1)
        fg = FormGrid(pn, label_width=16)
        fg.grid(row=0, column=0, sticky="ew")

        self.var_type = tk.StringVar(value=SWEEP_LABELS["snr"])
        fg.add("Sweep type:", combo(fg, self.var_type, list(SWEEP_LABELS.values()),
                                    width=20, on_change=self.on_type_changed))

        self.var_start = tk.StringVar(value="-10")
        fg.add("Start:", entry(fg, self.var_start))
        self.var_stop = tk.StringVar(value="20")
        fg.add("Stop:", entry(fg, self.var_stop))
        self.var_step = tk.StringVar(value="5")
        fg.add("Step:", entry(fg, self.var_step))
        self.var_trials = tk.StringVar(value="50")
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
        self.btn_run = ttk.Button(btns, text="Run Sweep", style="Accent.TButton",
                                  command=self.run_sweep)
        self.btn_run.grid(row=0, column=0, sticky="ew")
        self.btn_cancel = ttk.Button(btns, text="Stop", style="Small.TButton",
                                     command=self.cancel, state="disabled")
        self.btn_cancel.grid(row=0, column=1, sticky="ns", padx=(6, 0))

        pi = Panel(left, "Notes")
        pi.grid(row=3, column=0, sticky="nsew", pady=(8, 0))
        pi.columnconfigure(0, weight=1)
        pi.rowconfigure(0, weight=1)
        note_box(pi, NOTES, font=self.f["body"]).grid(row=0, column=0, sticky="nsew")

    def _build_right(self):
        right = ttk.Frame(self)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=6)
        right.rowconfigure(1, weight=4)

        self.pnl_rmse = PlotPanel(right, "RMSE vs Swept Parameter", figsize=(8.0, 3.8))
        self.pnl_rmse.grid(row=0, column=0, sticky="nsew")

        bot = ttk.Frame(right)
        bot.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        bot.rowconfigure(0, weight=1)
        bot.columnconfigure(0, weight=12)
        bot.columnconfigure(1, weight=10)

        self.pnl_res = PlotPanel(bot, "Resolution Probability", figsize=(4.4, 2.6))
        self.pnl_res.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        pt = Panel(bot, "Mean Compute Time / Call")
        pt.grid(row=0, column=1, sticky="nsew")
        pt.columnconfigure(0, weight=1)
        pt.rowconfigure(0, weight=1)
        self.tbl = make_table(pt, [
            ("Algorithm", 100, "w"),
            ("ms", 54, "e"),
            ("RMSE @ end", 90, "e"),
            ("Bias @ end", 90, "e"),
        ], height=7)
        self.tbl.grid(row=0, column=0, sticky="nsew")

        plots.plot_sweep_rmse(self.pnl_rmse.fig, None)
        self.pnl_rmse.draw()
        plots.plot_sweep_resolution(self.pnl_res.fig, None)
        self.pnl_res.draw()

    # -----------------------------------------------------------------
    @property
    def sweep_type(self) -> str:
        return SWEEP_KEY_BY_LABEL.get(self.var_type.get(), "snr")

    def on_type_changed(self):
        """Preload a sensible range for each sweep type."""
        start, stop, step = SWEEP_DEFAULTS[self.sweep_type]
        self.var_start.set("%g" % start)
        self.var_stop.set("%g" % stop)
        self.var_step.set("%g" % step)

    # -----------------------------------------------------------------
    def run_sweep(self):
        if self.runner.busy:
            return
        cfg = self.app.single.scenario()
        if not cfg.keys:
            messagebox.showwarning(
                "Nothing to sweep",
                "Select at least one algorithm on the Single Run tab.", parent=self)
            return

        values = sweep_values(read_float(self.var_start, -10.0),
                              read_float(self.var_stop, 20.0),
                              read_float(self.var_step, 5.0))
        if values.size == 0:
            messagebox.showwarning("Bad sweep range",
                                   "Start/stop/step do not define any sweep points.",
                                   parent=self)
            return

        trials = read_int(self.var_trials, 50, 1, 5000)
        self.btn_run.configure(state="disabled")
        self.btn_cancel.configure(state="normal")
        self.pb.configure(value=0.0)
        self.lbl_status.configure(text="starting ...")
        self.app.set_status("Sweep running: %d points x %d trials"
                            % (values.size, trials))
        self.runner.start(cfg, self.sweep_type, values, trials, cfg.keys)

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
            self.app.set_status("Sweep failed")
            messagebox.showerror("Sweep failed", error, parent=self)
            return

        self.result = S
        self.replot()
        done = "done" if S.completed else "stopped early"
        self.lbl_status.configure(text="%s: %d points x %d trials"
                                  % (done, S.values.size, S.trials))
        self.pb.configure(value=1.0 if S.completed else self.pb["value"])
        self.app.set_status("Sweep %s  |  %s  |  %d trials per point"
                            % (done, S.xlabel, S.trials))

    # -----------------------------------------------------------------
    def replot(self):
        S = self.result
        plots.plot_sweep_rmse(self.pnl_rmse.fig, S, yscale=self.var_scale.get())
        self.pnl_rmse.draw()
        plots.plot_sweep_resolution(self.pnl_res.fig, S)
        self.pnl_res.draw()
        if S is None:
            return

        rows = []
        for a in range(len(S.keys)):
            col_t = S.time[:, a]
            finite_t = col_t[~np.isnan(col_t)]
            mt = 1000.0 * float(finite_t.mean()) if finite_t.size else np.nan
            valid = np.nonzero(~np.isnan(S.rmse[:, a]))[0]
            if valid.size == 0:
                rows.append([S.labels[a], "-", "-", "-"])
            else:
                last = valid[-1]
                rows.append([S.labels[a], "%.2f" % mt,
                             "%.3f" % S.rmse[last, a], "%+.3f" % S.bias[last, a]])
        fill_table(self.tbl, rows)
