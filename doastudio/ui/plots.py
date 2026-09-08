"""All Matplotlib drawing, kept out of the tab layout code.

Every function takes a :class:`matplotlib.figure.Figure`, clears it and draws
one panel, so the tabs only decide *when* to redraw.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..core import ArrayGeometry, Scenario, Snapshots, SweepResult
from ..core.algorithms import AlgoResult
from ..core.numeric import cosd, sind
from . import theme

TRUE_GREY = (0.35, 0.35, 0.35)


def _style_axes(ax):
    ax.set_facecolor(theme.PANEL_BG)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(True, color="#dcdfe3", linewidth=0.7)
    ax.set_axisbelow(True)


def _empty(fig, message: str):
    fig.clear()
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center",
            fontsize=9, color=theme.TEXT_FAINT, wrap=True)
    return ax


# =====================================================================
# Single Run
# =====================================================================
def plot_spectrum(fig, cfg: Scenario, results: Sequence[AlgoResult]):
    """DoA spectrum with the true bearings marked."""
    fig.clear()
    ax = fig.add_subplot(111)
    _style_axes(ax)

    ylim = (-60.0, 5.0)
    for r in results:
        if not r.valid:
            continue
        # A search-free estimator has no real spectrum: draw its cosmetic
        # marker dotted and say so in the legend, so the curve is never
        # mistaken for a power measurement.
        style = "-" if r.has_spectrum else ":"
        width = 1.6 if r.has_spectrum else 1.2
        label = r.short if r.has_spectrum else r.short + " (estimate only)"
        ax.plot(cfg.scan, r.spectrum, style, color=r.color, linewidth=width,
                label=label)

    for k, d in enumerate(np.atleast_1d(cfg.doa)):
        ax.plot([d, d], ylim, "--", color=TRUE_GREY, linewidth=1.0,
                label="True DoA" if k == 0 else None)

    ax.set_xlim(-90, 90)
    ax.set_ylim(*ylim)
    ax.set_xticks(np.arange(-90, 91, 15))
    ax.set_xlabel("Angle of Arrival / DoA (deg)")
    ax.set_ylabel("Normalized Power (dB)")
    ax.set_title("DoA / Direction Finding Spectrum   (SNR = %g dB, N = %d, K = %d)"
                 % (cfg.snr, cfg.N, cfg.K))
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="lower left", fontsize=7.5, ncol=2)


def draw_geometry(fig, geo: ArrayGeometry):
    """Array top view with numbered elements and the sense antenna."""
    fig.clear()
    ax = fig.add_subplot(111)
    _style_axes(ax)

    p = geo.pos
    ax.plot(p[:, 0], p[:, 1], "o", markersize=11,
            markerfacecolor=(0.85, 0.33, 0.10), markeredgecolor="k",
            markeredgewidth=0.8, linestyle="none")
    for m in range(p.shape[0]):
        ax.text(p[m, 0], p[m, 1], str(m + 1), ha="center", va="center",
                fontsize=6.5, fontweight="bold", color="w")

    if geo.sense_pos is not None:
        s = geo.sense_pos.ravel()
        ax.plot(s[0], s[1], "D", markersize=11,
                markerfacecolor=(0.20, 0.60, 0.95), markeredgecolor="k",
                markeredgewidth=1.0, linestyle="none")
        ax.annotate("sense", (s[0], s[1]), textcoords="offset points",
                    xytext=(0, -18), ha="center", fontsize=7.5,
                    color="#1a5aa6", fontweight="bold")

    ax.set_aspect("equal", adjustable="box")
    r = max(float(np.abs(p).max()), 1.0) * 1.45
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_xlabel("X position (wavelengths)")
    ax.set_ylabel("Y (wavelengths)")
    ax.set_title(geo.label, fontsize=8.5)


def draw_compass(fig, cfg: Scenario, results: Sequence[AlgoResult]):
    """True versus estimated bearings as needles on a hand-drawn compass.

    Drawn on ordinary Cartesian axes rather than a polar projection so the
    0 deg = broadside = "up", +90 deg = endfire-right convention used
    everywhere else in the app is preserved exactly.
    """
    fig.clear()
    ax = fig.add_subplot(111)
    ax.set_facecolor(theme.PANEL_BG)

    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), color="#9aa0a6", linewidth=0.9)
    ax.plot(0.5 * np.cos(th), 0.5 * np.sin(th), ":", color="#cdd2d7", linewidth=0.8)

    for a in range(-180, 180, 30):
        u = np.array([sind(a), cosd(a)])
        ax.plot([0.94 * u[0], u[0]], [0.94 * u[1], u[1]], color="#6b7280",
                linewidth=0.8)
        ax.text(1.17 * u[0], 1.17 * u[1], str(a), ha="center", va="center",
                fontsize=6.5, color=theme.TEXT_MUTED)

    for d in np.atleast_1d(cfg.doa):
        u = np.array([sind(d), cosd(d)])
        ax.plot([0, u[0]], [0, u[1]], "-", linewidth=3.0, color=TRUE_GREY,
                solid_capstyle="round")

    handles, labels = [], []
    for r in results:
        if not r.valid:
            continue
        first = True
        for a in np.atleast_1d(r.angles):
            if np.isnan(a):
                continue
            u = np.array([sind(a), cosd(a)])
            line, = ax.plot([0, 0.92 * u[0]], [0, 0.92 * u[1]], "-",
                            linewidth=1.5, color=r.color)
            ax.plot(0.92 * u[0], 0.92 * u[1], "o", markersize=4,
                    markerfacecolor=r.color, markeredgecolor="none")
            if first:
                handles.append(line)
                labels.append(r.short)
                first = False

    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_title("grey needles = true bearing", fontsize=8.5, fontweight="normal")
    if handles:
        ax.legend(handles, labels, loc="upper center",
                  bbox_to_anchor=(0.5, -0.02), fontsize=6.5, ncol=2,
                  frameon=False, handlelength=1.4, columnspacing=1.0)


def plot_signal(fig, cfg: Scenario, data: Snapshots):
    """Element-1 waveform before and after AWGN, plus the measured SNR."""
    fig.clear()
    ax = fig.add_subplot(111)
    _style_axes(ax)

    n_show = min(120, data.X.shape[1])
    t = np.arange(1, n_show + 1)
    ax.plot(t, np.real(data.X_clean[0, :n_show]), "-", color=(0.20, 0.55, 0.20),
            linewidth=1.4, label="signal only")
    ax.plot(t, np.real(data.X[0, :n_show]), "-", color=(0.85, 0.33, 0.10),
            linewidth=0.9, label="signal + AWGN")
    if data.xs is not None:
        ax.plot(t, np.real(data.xs[:n_show]), ":", color=(0.20, 0.60, 0.95),
                linewidth=1.0, label="sense ch.")

    ax.set_xlabel("Snapshot index n")
    ax.set_ylabel(r"Re$\{x_1(n)\}$")
    ax.set_title("measured SNR = %.1f dB" % data.measured_snr_db, fontsize=8.5)
    # Inside the axes: tight_layout does not reserve room for a legend
    # anchored below, so an outside legend gets clipped in a short panel.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.42 * (hi - lo))
    ax.legend(loc="upper center", fontsize=6.0, ncol=3, frameon=True,
              framealpha=0.88, borderpad=0.3, handlelength=1.3,
              columnspacing=0.9, handletextpad=0.4)


# =====================================================================
# Sweeps
# =====================================================================
def _integer_ticks(ax, values: np.ndarray):
    """One tick per swept value when the sweep is short and integer valued.

    Fractional ticks like "1.5 sources" are meaningless, so they are
    suppressed rather than left to the auto-locator.
    """
    v = np.asarray(values, dtype=float).ravel()
    if v.size <= 14 and np.all(np.abs(v - np.round(v)) < 1e-9):
        ax.set_xticks(np.unique(v))


def _plot_metric(ax, S: SweepResult, matrix: np.ndarray, marker="o",
                 linewidth=1.6, legend=False):
    handles, labels = [], []
    for a in range(len(S.keys)):
        y = matrix[:, a]
        if np.all(np.isnan(y)):
            continue
        line, = ax.plot(S.values, y, "-" + marker, linewidth=linewidth,
                        markersize=4, color=S.colors[a],
                        markerfacecolor=S.colors[a], label=S.labels[a])
        handles.append(line)
        labels.append(S.labels[a])
    if legend and handles:
        ax.legend(handles, labels, loc="best", fontsize=7.5)
    return handles


def plot_sweep_rmse(fig, S: Optional[SweepResult], yscale: str = "log",
                    title: Optional[str] = None):
    if S is None:
        return _empty(fig, "Run a sweep to see RMSE against the swept parameter.")
    fig.clear()
    ax = fig.add_subplot(111)
    _style_axes(ax)
    _plot_metric(ax, S, S.rmse, legend=True)
    if yscale == "log":
        ax.set_yscale("log")
    _integer_ticks(ax, S.values)
    ax.set_xlabel(S.xlabel)
    ax.set_ylabel("RMSE (deg)")
    ax.set_title(title or ("DoA RMSE vs %s   (%d Monte-Carlo trials per point)"
                           % (S.xlabel, S.trials)))
    return ax


def plot_sweep_resolution(fig, S: Optional[SweepResult], title: Optional[str] = None):
    if S is None:
        return _empty(fig, "Resolution probability appears here after a sweep.")
    fig.clear()
    ax = fig.add_subplot(111)
    _style_axes(ax)
    _plot_metric(ax, S, S.res_prob, marker="s", linewidth=1.4)
    ax.set_ylim(-0.05, 1.05)
    _integer_ticks(ax, S.values)
    ax.set_xlabel(S.xlabel)
    ax.set_ylabel("P(all sources resolved)")
    ax.set_title(title or ("Resolution probability vs source count"
                           if S.type == "sources" else "Resolution probability"))
    return ax


def plot_sweep_bias(fig, S: Optional[SweepResult], title: Optional[str] = None):
    if S is None:
        return _empty(fig, "Bias appears here after a study.")
    fig.clear()
    ax = fig.add_subplot(111)
    _style_axes(ax)
    _plot_metric(ax, S, S.bias, linewidth=1.4)
    ax.axhline(0.0, color="#6b7280", linestyle=":", linewidth=0.9)
    _integer_ticks(ax, S.values)
    ax.set_xlabel(S.xlabel)
    ax.set_ylabel("Bias (deg)")
    ax.set_title(title or "Mean signed error")
    return ax
