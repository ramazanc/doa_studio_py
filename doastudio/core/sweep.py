"""Monte-Carlo performance sweep over one scenario parameter.

Python port of ``doa_sweep.m``, plus a cancellation hook so the GUI can abort
a long study without killing the process.

Speed notes: the geometry is rebuilt only once per sweep point, the
shift-invariant subarray search and the circular-mode analysis are memoised
on the geometry, and every estimator is internally vectorised over the whole
scan grid, so the only genuine loop is over trials.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np

from .algorithms import ALGORITHM_BY_KEY, run_algorithms
from .geometry import array_geometry, nla_preset
from .metrics import doa_metrics
from .scenario import Scenario, default_doa
from .signals import generate_snapshots

#: Sweep types the engine understands, mapped to their x-axis label.
SWEEP_TYPES = {
    "snr":        "SNR (dB)",
    "snapshots":  "Snapshots N (samples)",
    "separation": "Source separation (deg)",
    "elements":   "Number of elements M (count)",
    "spacing":    "Element spacing / radius (wavelengths)",
    "sources":    "Number of sources K (count)",
}

#: Human-readable names for the sweep-type dropdown.
SWEEP_LABELS = {
    "snr":        "SNR",
    "snapshots":  "Snapshots",
    "separation": "DoA separation",
    "elements":   "Elements (M)",
    "spacing":    "Spacing / radius",
    "sources":    "Sources count",
}

#: Sensible default start / stop / step for each sweep type.
SWEEP_DEFAULTS = {
    "snr":        (-10.0, 20.0, 5.0),
    "snapshots":  (20.0, 500.0, 60.0),
    "separation": (1.0, 30.0, 2.0),
    "elements":   (4.0, 20.0, 2.0),
    "spacing":    (0.2, 1.2, 0.1),
    "sources":    (1.0, 6.0, 1.0),
}


@dataclass
class SweepResult:
    """Outcome of one Monte-Carlo sweep."""

    type: str
    values: np.ndarray                  # (V,) swept parameter values
    xlabel: str
    keys: List[str]
    labels: List[str]
    colors: List[tuple]
    rmse: np.ndarray                    # (V, A) degrees
    bias: np.ndarray                    # (V, A) degrees
    res_prob: np.ndarray                # (V, A) fraction of resolved trials
    time: np.ndarray                    # (V, A) mean seconds per call
    trials: int
    completed: bool = True              # False when the user cancelled


def sweep_values(start: float, stop: float, step: float) -> np.ndarray:
    """Build the sweep grid, tolerating a descending or zero-step range."""
    if step == 0:
        return np.array([float(start)])
    if (stop - start) * step < 0:
        step = -step                    # the user typed a descending range
    n = int(np.floor((stop - start) / step + 1e-9)) + 1
    if n < 1:
        return np.array([float(start)])
    return start + step * np.arange(n)


def apply_sweep(cfg: Scenario, sweep_type: str, v: float) -> Scenario:
    """Return the baseline scenario with one parameter replaced."""
    t = sweep_type.lower()

    if t == "snr":
        return cfg.copy(snr=float(v))

    if t == "snapshots":
        return cfg.copy(N=max(2, int(round(v))))

    if t == "separation":
        # Keep the source constellation centred on its original mean and set
        # the spacing between adjacent sources to v degrees.
        K = cfg.K
        centre = float(np.mean(cfg.doa))
        doa = centre + (np.arange(K) - (K - 1) / 2.0) * float(v)
        return cfg.copy(doa=doa, tol=None)   # recompute tolerance from separation

    if t == "elements":
        M = max(2, int(round(v)))
        if cfg.array_type.upper() == "NLA":
            # Re-derive a sparse layout with the requested count.
            return cfg.copy(M=M, pos_vec=nla_preset(M, cfg.spacing))
        return cfg.copy(M=M)

    if t == "spacing":
        # For a UCA the same control is the radius.
        return cfg.copy(spacing=float(v), radius=float(v))

    if t == "sources":
        # Spread K sources evenly across a fixed +-50 deg sector so the
        # separation shrinks as K grows - the interesting regime.
        K = max(1, int(round(v)))
        return cfg.copy(K=K, doa=default_doa(K), tol=None)

    raise ValueError('Unknown sweep type "%s".' % sweep_type)


def doa_sweep(cfg: Scenario, sweep_type: str, values: Sequence[float],
              trials: int, keys: Sequence[str],
              progress: Optional[Callable[[float, str], None]] = None,
              should_stop: Optional[Callable[[], bool]] = None) -> SweepResult:
    """Run ``trials`` Monte-Carlo trials at every point of ``values``.

    ``progress`` is called as ``progress(fraction, text)`` after each sweep
    point; ``should_stop`` is polled at the same cadence so a GUI can cancel.
    Sweep points that are physically impossible (K >= M) are skipped and
    appear as gaps in the curves rather than raising.
    """
    keys = [k for k in keys if k in ALGORITHM_BY_KEY]
    specs = [ALGORITHM_BY_KEY[k] for k in keys]
    values = np.asarray(values, dtype=float).ravel()
    nV, nA = values.size, len(keys)

    out = SweepResult(
        type=sweep_type,
        values=values,
        xlabel=SWEEP_TYPES.get(sweep_type.lower(), "Swept parameter"),
        keys=keys,
        labels=[s.short for s in specs],
        colors=[s.color for s in specs],
        rmse=np.full((nV, nA), np.nan),
        bias=np.full((nV, nA), np.nan),
        res_prob=np.full((nV, nA), np.nan),
        time=np.full((nV, nA), np.nan),
        trials=int(trials),
    )
    if nA == 0 or nV == 0:
        return out

    scan = cfg.scan

    for iv, v in enumerate(values):
        if should_stop is not None and should_stop():
            out.completed = False
            break

        c = apply_sweep(cfg, sweep_type, v)
        geo = array_geometry(c)

        # Skip physically impossible points instead of erroring out.
        if c.K >= geo.M:
            continue

        acc_sq = np.zeros(nA)
        acc_err = np.zeros(nA)
        acc_res = np.zeros(nA)
        acc_t = np.zeros(nA)
        n_ok = np.zeros(nA)

        for _ in range(int(trials)):
            data = generate_snapshots(geo, c.doa, c.snr, c.N)
            res = run_algorithms(geo, data, scan, c.K, keys)

            for a, r in enumerate(res):
                if not r.valid:
                    continue
                m = doa_metrics(r.angles, c.doa, c.tol)
                acc_sq[a] += m.rmse ** 2
                acc_err[a] += m.bias
                acc_res[a] += float(m.resolved)
                acc_t[a] += r.time
                n_ok[a] += 1

        good = n_ok > 0
        out.rmse[iv, good] = np.sqrt(acc_sq[good] / n_ok[good])
        out.bias[iv, good] = acc_err[good] / n_ok[good]
        out.res_prob[iv, good] = acc_res[good] / n_ok[good]
        out.time[iv, good] = acc_t[good] / n_ok[good]

        if progress is not None:
            progress((iv + 1) / nV,
                     "%s = %g  (%d of %d)" % (sweep_type, v, iv + 1, nV))

    return out
