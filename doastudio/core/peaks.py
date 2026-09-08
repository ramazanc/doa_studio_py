"""K strongest local maxima of a 1-D spectrum.

Python port of ``pick_peaks.m``.  Written without SciPy's ``find_peaks`` for
the same reason the MATLAB original avoids ``findpeaks``: the peak picker is
part of the estimator, and its sub-grid interpolation directly sets the RMSE
floor at high SNR, so it should be visible rather than delegated.
"""

from __future__ import annotations

import numpy as np


def pick_peaks(spec: np.ndarray, scan_deg: np.ndarray, K: int) -> np.ndarray:
    """Return up to ``K`` angles, ascending, at the tallest interior maxima.

    A quadratic interpolation through each peak and its two neighbours
    refines the estimate below the scan-grid resolution, which matters when
    the grid step (0.5 deg by default) would otherwise dominate the RMSE at
    high SNR.
    """
    spec = np.asarray(spec, dtype=float).ravel()
    scan = np.asarray(scan_deg, dtype=float).ravel()
    n = spec.size
    K = max(1, int(K))

    # Interior local maxima (strictly greater than the left neighbour, at
    # least equal to the right one - the asymmetry breaks ties on a plateau).
    if n >= 3:
        interior = np.nonzero((spec[1:-1] > spec[:-2]) & (spec[1:-1] >= spec[2:]))[0] + 1
    else:
        interior = np.zeros(0, dtype=int)

    # Fall back to the global maximum if the spectrum is flat or monotonic.
    if interior.size == 0:
        interior = np.array([int(np.argmax(spec))])

    # Keep the K tallest.
    order = np.argsort(spec[interior])[::-1]
    idx = interior[order[:min(K, interior.size)]]

    ang = np.empty(idx.size, dtype=float)
    for q, i0 in enumerate(idx):
        if 0 < i0 < n - 1:
            # Parabolic vertex through (i0-1, i0, i0+1).
            y1, y2, y3 = spec[i0 - 1], spec[i0], spec[i0 + 1]
            den = y1 - 2.0 * y2 + y3
            if den != 0.0:
                delta = 0.5 * (y1 - y3) / den          # in grid samples
                delta = float(np.clip(delta, -1.0, 1.0))
            else:
                delta = 0.0
            step = (scan[min(i0 + 1, n - 1)] - scan[max(i0 - 1, 0)]) / 2.0
            ang[q] = scan[i0] + delta * step
        else:
            ang[q] = scan[i0]

    return np.sort(ang)
