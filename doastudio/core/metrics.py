"""Per-trial accuracy metrics for one algorithm.

Python port of ``doa_metrics.m``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Worst-case penalty, in degrees, for an estimate that was never produced.
#: Without it an estimator could win the RMSE comparison by refusing to
#: report a source it is unsure about.
PENALTY = 180.0


@dataclass
class DoAMetrics:
    err: np.ndarray        # signed per-source error (est - true), degrees
    rmse: float            # sqrt(mean(err**2)) over sources
    bias: float            # mean(err) over sources
    resolved: bool         # True when EVERY source matched within tol
    tol: float             # tolerance actually used, degrees


def doa_metrics(est_deg, true_deg, tol_deg=None) -> DoAMetrics:
    """Accuracy of one estimate against the true bearings.

    Both vectors are sorted ascending and paired in order, which is the
    correct assignment whenever the estimator has not swapped sources - and a
    swapped pairing would in any case be counted as an unresolved trial.
    Missing estimates (NaN, or fewer estimates than sources) are penalised
    with :data:`PENALTY`.

    ``tol_deg`` defaults to half the smallest true source separation, capped
    at 5 deg: the usual "correctly resolved" criterion, where an estimate must
    be closer to its own source than to any other.
    """
    t = np.sort(np.atleast_1d(np.asarray(true_deg, dtype=float)).ravel())
    e = np.sort(np.atleast_1d(np.asarray(est_deg, dtype=float)).ravel())
    K = t.size

    if tol_deg is None:
        tol_deg = min(5.0, float(np.min(np.diff(t))) / 2.0) if K > 1 else 5.0
    tol_deg = float(tol_deg)

    err = np.full(K, PENALTY, dtype=float)
    n = min(K, e.size)
    if n > 0:
        d = e[:n] - t[:n]
        good = np.nonzero(~np.isnan(d))[0]
        err[good] = d[good]

    return DoAMetrics(
        err=err,
        rmse=float(np.sqrt(np.mean(err ** 2))),
        bias=float(np.mean(err)),
        resolved=bool(np.all(np.abs(err) <= tol_deg)),
        tol=tol_deg,
    )
