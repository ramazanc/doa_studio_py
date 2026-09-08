"""Single Run tab: one realisation, plus the interactive limitation panel.

This tab also owns the scenario controls; the Sweep and SNR tabs read their
baseline from :meth:`SingleRunTab.scenario`, exactly as the MATLAB app did.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import List, Optional

import numpy as np

from ..core import (ALGORITHMS, ArrayGeometry, Scenario, array_geometry,
                    default_doa, default_scan, doa_limitations, doa_metrics,
                    esprit_subarrays, generate_snapshots, nla_preset,
                    run_algorithms)
from ..core.algorithms import AlgoResult
from ..core.limitations import SEVERITY_TAG, Limitation
from ..core.numeric import format_number_list, parse_number_list
from . import plots, theme
from .widgets import (FormGrid, Panel, PlotPanel, ScrolledReadOnlyText, combo,
                      entry, fill_table, make_table, read_float, read_int)

SOURCE_CHOICES = [str(k) for k in range(1, 7)]
ARRAY_CHOICES = {
    "ULA (Uniform Linear Array)": "ULA",
    "UCA (Uniform Circular Array)": "UCA",
    "NLA (Non-uniform Linear Array)": "NLA",
}
ARRAY_LABEL_BY_KEY = {v: k for k, v in ARRAY_CHOICES.items()}


class SingleRunTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=8)
        self.app = app
        self.f = app.f                      # named fonts

        self.results: List[AlgoResult] = []
        self.last_data = None
        self.last_geo: Optional[ArrayGeometry] = None
        self.limitations: List[Limitation] = []
        self._esprit_auto_off = False       # True when WE unchecked ESPRIT
        self._building = True

        self.columnconfigure(0, minsize=430, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self._build_left()
        self._build_right()

        self._building = False
        self.on_array_type_changed()

    # =================================================================
    # Layout
    # =================================================================
    def _build_left(self):
        left = ttk.Frame(self)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(3, weight=1)          # limitation panel expands

        # ---- Scenario Settings ----
        pn = Panel(left, "Scenario Settings")
        pn.grid(row=0, column=0, sticky="ew")
        pn.columnconfigure(0, weight=1)
        fg = FormGrid(pn, label_width=17)
        fg.grid(row=0, column=0, sticky="ew")

        self.var_array = tk.StringVar(value=ARRAY_LABEL_BY_KEY["ULA"])
        fg.add("Array Type:", combo(fg, self.var_array, list(ARRAY_CHOICES),
                                    width=26, on_change=self.on_array_type_changed))

        # Short numeric fields share a row so the limitation panel below keeps
        # usable height on a 768-pixel-tall screen.
        self.var_M = tk.StringVar(value="8")
        self.ent_M = entry(fg, self.var_M, width=8,
                           on_change=self.on_geometry_changed)
        self.var_spacing = tk.StringVar(value="0.5")
        self.ent_spacing = entry(fg, self.var_spacing, width=8,
                                 on_change=self.on_geometry_changed)
        self.lbl_M, self.lbl_spacing = fg.add_pair(
            "Elements (M):", self.ent_M, "Spacing / λ:", self.ent_spacing)

        self.var_pos = tk.StringVar(value="0, 0.5, 2, 3")
        self.ent_pos = entry(fg, self.var_pos, on_change=self.on_geometry_changed)
        self.lbl_pos = fg.add("Positions (λ):", self.ent_pos)

        self.var_sense = tk.BooleanVar(value=True)
        fg.add("Sense antenna:",
               ttk.Checkbutton(fg, text="enabled (at array centroid)",
                               variable=self.var_sense,
                               command=self.on_geometry_changed))

        self.var_sources = tk.StringVar(value="2")
        self.var_doa = tk.StringVar(value="-20, 30")
        fg.add_pair("Sources:",
                    combo(fg, self.var_sources, SOURCE_CHOICES, width=5,
                          on_change=self.on_sources_changed),
                    "DoA (deg):",
                    entry(fg, self.var_doa, on_change=self.refresh_limitations))

        self.var_snr = tk.StringVar(value="10")
        self.var_N = tk.StringVar(value="200")
        fg.add_pair("SNR (dB):",
                    entry(fg, self.var_snr, width=8,
                          on_change=self.refresh_limitations),
                    "Snapshots (N):",
                    entry(fg, self.var_N, width=8,
                          on_change=self.refresh_limitations),
                    label_b_width=14)

        self.var_trials = tk.StringVar(value="1")
        fg.add("Monte Carlo trials:", entry(fg, self.var_trials, width=8))

        # ---- Algorithm Selection ----
        pa = Panel(left, "Algorithm Selection")
        pa.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        pa.columnconfigure(0, weight=1)

        self.alg_vars = {}
        self.alg_checks = {}
        for i, spec in enumerate(ALGORITHMS):
            var = tk.BooleanVar(value=True)
            cb = ttk.Checkbutton(pa, text=spec.label, variable=var,
                                 command=self.refresh_limitations)
            cb.grid(row=i, column=0, sticky="w", pady=1)
            # A colour chip ties each row to its curve in the spectrum plot.
            tk.Frame(pa, background=spec.hex_color, width=14, height=3).grid(
                row=i, column=1, sticky="e", padx=(6, 2))
            self.alg_vars[spec.key] = var
            self.alg_checks[spec.key] = cb

        self.lbl_scan = ttk.Label(
            pa, style="Muted.TLabel",
            text="Scan: -90 to +90 deg  |  Step: 0.5 deg  |  RMSE = Monte-Carlo")
        self.lbl_scan.grid(row=len(ALGORITHMS), column=0, columnspan=2,
                           sticky="w", pady=(6, 0))

        # ---- Run button ----
        self.btn_run = ttk.Button(left, text="Run Estimation", style="Accent.TButton",
                                  command=self.run_single)
        self.btn_run.grid(row=2, column=0, sticky="ew", pady=(8, 0))

        # ---- Limitation panel ----
        pl = Panel(left, "Scenario Limitations  (click an item to read why)")
        pl.grid(row=3, column=0, sticky="nsew", pady=(8, 0))
        pl.columnconfigure(0, weight=1)
        pl.rowconfigure(0, weight=2, minsize=76)
        pl.rowconfigure(1, weight=3, minsize=92)

        list_wrap = ttk.Frame(pl)
        list_wrap.grid(row=0, column=0, sticky="nsew")
        list_wrap.columnconfigure(0, weight=1)
        list_wrap.rowconfigure(0, weight=1)

        self.lb_limit = tk.Listbox(
            list_wrap, height=4, activestyle="none", exportselection=False,
            font=self.f["body"], relief="solid", borderwidth=1,
            highlightthickness=0, background=theme.PANEL_BG,
            selectbackground=theme.ACCENT_LIGHT, selectforeground=theme.TEXT)
        self.lb_limit.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(list_wrap, orient="vertical", command=self.lb_limit.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.lb_limit.configure(yscrollcommand=sb.set)
        self.lb_limit.bind("<<ListboxSelect>>", lambda _e: self.on_limit_selected())

        self.txt_limit = ScrolledReadOnlyText(pl, height=5, font=self.f["body"])
        self.txt_limit.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        self.txt_limit.configure_tag("affects", font=self.f["italic"],
                                     foreground=theme.TEXT_MUTED)
        self.txt_limit.set_text(
            "Run an estimation to see which assumptions the current scenario breaks.")

    def _build_right(self):
        right = ttk.Frame(self)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=5)
        right.rowconfigure(1, weight=4)

        self.pnl_spec = PlotPanel(right, "DoA Spectrum", figsize=(8.0, 3.6))
        self.pnl_spec.grid(row=0, column=0, sticky="nsew")

        bottom = ttk.Frame(right)
        bottom.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        bottom.rowconfigure(0, weight=1)
        for c, w in enumerate((10, 9, 10, 13)):
            bottom.columnconfigure(c, weight=w, uniform="b")

        self.pnl_geo = PlotPanel(bottom, "Array Geometry (Top View)", figsize=(2.4, 2.4))
        self.pnl_geo.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        self.pnl_compass = PlotPanel(bottom, "Compass (true vs estimated)",
                                     figsize=(2.4, 2.4))
        self.pnl_compass.grid(row=0, column=1, sticky="nsew", padx=(0, 6))

        self.pnl_signal = PlotPanel(bottom, "Received Signal with AWGN",
                                    figsize=(2.4, 2.4))
        self.pnl_signal.grid(row=0, column=2, sticky="nsew", padx=(0, 6))

        pt = Panel(bottom, "Per-Algorithm Summary")
        pt.grid(row=0, column=3, sticky="nsew")
        pt.columnconfigure(0, weight=1)
        pt.rowconfigure(0, weight=1)
        self.tbl_run = make_table(pt, [
            ("Algorithm", 76, "w", False),
            ("Est. DoA (deg)", 84, "w", True),
            ("RMSE", 46, "e", False),
            ("Bias", 48, "e", False),
            ("ms", 36, "e", False),
        ], height=7)
        self.tbl_run.grid(row=0, column=0, sticky="nsew")
        self.tbl_run.tag_configure("na", foreground=theme.TEXT_FAINT)

    # =================================================================
    # Scenario assembly
    # =================================================================
    @property
    def array_type(self) -> str:
        return ARRAY_CHOICES.get(self.var_array.get(), "ULA")

    def selected_keys(self) -> List[str]:
        return [s.key for s in ALGORITHMS
                if self.alg_vars[s.key].get()
                and str(self.alg_checks[s.key]["state"]) != "disabled"]

    def scenario(self) -> Scenario:
        """Build a :class:`Scenario` from the current control values."""
        array_type = self.array_type
        M = read_int(self.var_M, 8, 2, 64)
        spacing = read_float(self.var_spacing, 0.5, 0.05, 5.0)

        pos_vec = parse_number_list(self.var_pos.get())
        # A blank or malformed position list would make array_geometry raise
        # from inside a widget callback, so fall back to a sparse preset.
        if pos_vec.size < 2:
            pos_vec = nla_preset(M, 0.5)

        K = int(self.var_sources.get())
        doa = parse_number_list(self.var_doa.get())
        if doa.size < K:
            doa = np.concatenate([doa, default_doa(K)[doa.size:K]])
        doa = doa[:K]

        return Scenario(
            array_type=array_type,
            M=M,
            spacing=spacing,
            radius=spacing,                 # same control, UCA meaning
            pos_vec=pos_vec,
            sense=bool(self.var_sense.get()),
            doa=doa,
            K=K,
            snr=read_float(self.var_snr, 10.0, -40.0, 60.0),
            N=read_int(self.var_N, 200, 2, 100000),
            trials=read_int(self.var_trials, 1, 1, 2000),
            tol=None,
            scan=default_scan(),
            keys=self.selected_keys(),
        )

    def geometry(self) -> ArrayGeometry:
        return array_geometry(self.scenario())

    # =================================================================
    # Control callbacks
    # =================================================================
    def on_array_type_changed(self):
        if self._building:
            return
        t = self.array_type
        is_nla, is_uca = t == "NLA", t == "UCA"

        self.lbl_spacing.configure(text="Radius / λ:" if is_uca else "Spacing / λ:")
        for w in (self.lbl_spacing, self.ent_spacing, self.lbl_M, self.ent_M):
            w.configure(state="disabled" if is_nla else "normal")
        for w in (self.lbl_pos, self.ent_pos):
            w.configure(state="normal" if is_nla else "disabled")

        if is_nla:
            # Seed the position box with a sparse preset for the current M.
            M = read_int(self.var_M, 8, 2, 64)
            self.var_pos.set(format_number_list(nla_preset(M, 0.5), "%.2f"))

        self.on_geometry_changed()

    def on_geometry_changed(self):
        if self._building:
            return
        # ESPRIT is only meaningful where rotational invariance holds.
        geo = self.geometry()
        ok = True
        if geo.type == "NLA":
            i1, _, _ = esprit_subarrays(geo.pos)
            ok = i1 is not None

        spec = next(s for s in ALGORITHMS if s.key == "esprit")
        cb = self.alg_checks["esprit"]
        if ok:
            cb.configure(state="normal", text=spec.label)
            if self._esprit_auto_off:
                # Returning to a valid geometry restores the user's original
                # selection instead of silently leaving ESPRIT out.
                self.alg_vars["esprit"].set(True)
                self._esprit_auto_off = False
        else:
            if self.alg_vars["esprit"].get():
                self._esprit_auto_off = True
            self.alg_vars["esprit"].set(False)
            cb.configure(state="disabled",
                         text=spec.label + "  -  N/A for this geometry")

        plots.draw_geometry(self.pnl_geo.fig, geo)
        self.pnl_geo.draw()
        self.refresh_limitations()

    def on_sources_changed(self):
        if self._building:
            return
        K = int(self.var_sources.get())
        cur = parse_number_list(self.var_doa.get())
        if cur.size != K:
            self.var_doa.set(format_number_list(default_doa(K)))
        self.refresh_limitations()

    # =================================================================
    # Limitation panel
    # =================================================================
    def refresh_limitations(self):
        if self._building:
            return
        cfg = self.scenario()
        geo = array_geometry(cfg)
        self.limitations = doa_limitations(cfg, geo, self.results)

        self.lb_limit.delete(0, "end")
        for i, lim in enumerate(self.limitations):
            self.lb_limit.insert("end", "  " + lim.list_entry)
            self.lb_limit.itemconfig(i, foreground=theme.SEVERITY_FG.get(
                lim.severity, theme.TEXT))

        if self.limitations:
            self.lb_limit.selection_clear(0, "end")
            self.lb_limit.selection_set(0)
            self.lb_limit.see(0)
            self.on_limit_selected()

    def on_limit_selected(self):
        sel = self.lb_limit.curselection()
        if not sel or sel[0] >= len(self.limitations):
            return
        lim = self.limitations[sel[0]]
        tag = SEVERITY_TAG.get(lim.severity, "NOTE")

        self.txt_limit.configure_tag(
            "head", font=self.f["heading"],
            foreground=theme.SEVERITY_FG.get(lim.severity, theme.TEXT))
        self.txt_limit.replace([
            ("[%s]  %s\n\n" % (tag, lim.title), "head"),
            (lim.text + "\n\n", None),
            ("Affects: %s" % lim.algos, "affects"),
        ])

    # =================================================================
    # Run
    # =================================================================
    def run_single(self):
        cfg = self.scenario()
        if not cfg.keys:
            messagebox.showwarning("Nothing to run",
                                   "Select at least one algorithm.", parent=self)
            return

        geo = array_geometry(cfg)
        if cfg.K >= geo.M:
            messagebox.showwarning(
                "Too many sources",
                "%d sources cannot be estimated with %d elements.\n\n"
                "Reduce the source count or add elements." % (cfg.K, geo.M),
                parent=self)
            return

        self.btn_run.configure(state="disabled")
        self.app.set_status("Running estimation ...")
        self.update_idletasks()
        try:
            # ---- one displayed realisation ----
            data = generate_snapshots(geo, cfg.doa, cfg.snr, cfg.N)
            res = run_algorithms(geo, data, cfg.scan, cfg.K, cfg.keys)

            # ---- Monte-Carlo average for the summary table ----
            nA = len(res)
            acc_sq = np.zeros(nA)
            acc_b = np.zeros(nA)
            acc_t = np.zeros(nA)
            n_ok = np.zeros(nA)
            for it in range(cfg.trials):
                if it == 0:
                    r = res                              # reuse the displayed trial
                else:
                    d = generate_snapshots(geo, cfg.doa, cfg.snr, cfg.N)
                    r = run_algorithms(geo, d, cfg.scan, cfg.K, cfg.keys)
                for a, ra in enumerate(r):
                    if not ra.valid:
                        continue
                    m = doa_metrics(ra.angles, cfg.doa, cfg.tol)
                    acc_sq[a] += m.rmse ** 2
                    acc_b[a] += m.bias
                    acc_t[a] += ra.time
                    n_ok[a] += 1
        except Exception as exc:                          # noqa: BLE001
            self.btn_run.configure(state="normal")
            self.app.set_status("Estimation failed")
            messagebox.showerror("Estimation failed", str(exc), parent=self)
            return

        self.results = res
        self.last_data = data
        self.last_geo = geo

        plots.plot_spectrum(self.pnl_spec.fig, cfg, res)
        self.pnl_spec.draw()
        plots.draw_geometry(self.pnl_geo.fig, geo)
        self.pnl_geo.draw()
        plots.draw_compass(self.pnl_compass.fig, cfg, res)
        self.pnl_compass.draw()
        plots.plot_signal(self.pnl_signal.fig, cfg, data)
        self.pnl_signal.draw()

        self._fill_run_table(res, acc_sq, acc_b, acc_t, n_ok)
        self.refresh_limitations()

        self.btn_run.configure(state="normal")
        self.app.set_status(
            "Single run complete  |  %s  |  %d source(s), SNR %g dB, N = %d, "
            "%d trial(s)" % (geo.label, cfg.K, cfg.snr, cfg.N, cfg.trials))

    def _fill_run_table(self, res, acc_sq, acc_b, acc_t, n_ok):
        rows, tags = [], []
        for a, r in enumerate(res):
            if not r.valid:
                rows.append([r.short, "not applicable", "-", "-", "-"])
                tags.append("na")
                continue
            if n_ok[a] > 0:
                rmse = float(np.sqrt(acc_sq[a] / n_ok[a]))
                bias = float(acc_b[a] / n_ok[a])
                ms = 1000.0 * acc_t[a] / n_ok[a]
                rows.append([r.short, r.angles_text, "%.3f" % rmse,
                             "%+.3f" % bias, "%.2f" % ms])
            else:
                rows.append([r.short, r.angles_text, "-", "-", "-"])
            tags.append("")
        fill_table(self.tbl_run, rows, tags)
