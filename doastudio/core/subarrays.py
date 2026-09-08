"""Geometry analysis that decides where ESPRIT and Watson-Watt are legal.

Python port of ``esprit_subarrays.m`` and ``uca_modes.m``.  Both are pure
functions of the geometry, and both are called once per Monte-Carlo trial by
the estimators, so both are memoised - a sweep of 60 points x 200 trials
would otherwise repeat the same O(M^4) search 12000 times.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.special import jv

# --- thresholds that define a "usable" circular-array phase mode ----------
GAIN_MIN = 0.05     # below this Bessel gain the mode is nulled
RATIO_MAX = 0.10    # above this alias-to-signal ratio the mode phase is corrupt


# =====================================================================
# Shift-invariant subarray pairs (element-space ESPRIT)
# =====================================================================
def esprit_subarrays(pos, tol: float = 1e-9):
    """Find the largest translationally-invariant subarray pair.

    Searches the element positions ``pos`` (M x 2, wavelengths) for the
    largest set of element pairs sharing ONE common displacement vector
    ``delta``::

        pos[i2[q], :] == pos[i1[q], :] + delta      for every q

    ``i1`` and ``i2`` index the two subarrays; they may overlap (for a ULA
    the standard choice 0..M-2 / 1..M-1 shares M-2 elements).

    Why this matters: ESPRIT does not search a grid, it solves for a
    rotation.  That only works when the two subarrays see the same signal
    subspace up to a diagonal phase rotation, i.e. when they are identical
    arrays rigidly displaced by ``delta``.  A ULA satisfies this by
    construction; an arbitrary non-uniform array generally does not, and this
    function then returns ``(None, None, None)`` so the caller can disable
    ESPRIT instead of producing meaningless numbers.
    """
    pos = np.atleast_2d(np.asarray(pos, dtype=float))
    key = (pos.tobytes(), pos.shape, float(tol))
    i1, i2, delta = _esprit_subarrays_cached(key)
    if i1 is None:
        return None, None, None
    return np.asarray(i1, dtype=int), np.asarray(i2, dtype=int), np.asarray(delta)


@lru_cache(maxsize=256)
def _esprit_subarrays_cached(key):
    raw, shape, tol = key
    pos = np.frombuffer(raw, dtype=float).reshape(shape)
    M = shape[0]

    # ---- fast path: a uniform linear array --------------------------
    # Detecting this directly avoids the brute-force search below, which
    # matters when a sweep rebuilds the geometry thousands of times.
    if M >= 2 and np.all(np.abs(pos[:, 1] - pos[0, 1]) < tol):
        order = np.argsort(pos[:, 0])
        dx = np.diff(pos[order, 0])
        if dx.size and np.all(np.abs(dx - dx[0]) < tol) and dx[0] > tol:
            return tuple(order[:-1]), tuple(order[1:]), (float(dx[0]), 0.0)

    best_n = 0
    best = (None, None, None)

    # Candidate displacements: every ordered pair of distinct elements.
    for a in range(M):
        for b in range(M):
            if a == b:
                continue
            d = pos[b, :] - pos[a, :]
            # Two elements at the same coordinate (a user typing duplicate
            # positions) would offer the zero displacement, which every
            # element trivially satisfies and which carries no phase
            # information at all - it would make ESPRIT return 0 deg.
            if np.linalg.norm(d) < tol:
                continue

            # Which elements have a partner at +d?  L1 distance from every
            # shifted element to every real one, in one broadcast.
            shifted = pos + d                                    # M x 2
            dist = np.abs(pos[None, :, :] - shifted[:, None, :]).sum(axis=2)
            nearest = dist.argmin(axis=1)
            hit = dist[np.arange(M), nearest] < tol

            n_hit = int(hit.sum())
            if n_hit > best_n:
                best_n = n_hit
                p1 = np.nonzero(hit)[0]
                best = (tuple(p1), tuple(nearest[p1]), (float(d[0]), float(d[1])))

    if best_n < 2:
        return None, None, None
    return best


# =====================================================================
# Circular-array phase modes (beamspace ESPRIT, Watson-Watt)
# =====================================================================
@dataclass
class UcaModes:
    """Which phase modes of a uniform circular array carry usable bearing."""

    z: float                 # 2*pi*R
    arc: float               # inter-element arc spacing in wavelengths
    Hmax: int                # aliasing cap, floor((M-1)/2)
    H: int                   # highest contiguous usable mode (0 = mode 1 fails)
    J: np.ndarray            # J_h(z) for h = 0..Hmax
    ratio: np.ndarray        # alias-to-signal amplitude ratio for h = 1..Hmax
    mode1_ok: bool           # True when mode 1 (the loop channel) is trustworthy
    why: str                 # the binding limit, empty string when all is well

    @property
    def n_modes(self) -> int:
        """Number of virtual ULA elements the Davies transform can offer."""
        return 2 * self.H + 1

    @property
    def J0(self) -> float:
        return float(self.J[0]) if self.J.size >= 1 else 0.0

    @property
    def J1(self) -> float:
        return float(self.J[1]) if self.J.size >= 2 else 0.0

    @property
    def ratio1(self) -> float:
        return float(self.ratio[0]) if self.ratio.size else 0.0


def uca_modes(R: float, M: int) -> UcaModes:
    """Usable phase modes of a uniform circular array of radius ``R``.

    Both the Watson-Watt adaptation and beamspace ESPRIT work in the phase
    mode (Davies) domain, where the M element outputs become modes h = -H..H::

        Y_h = (1/M) * sum_m x_m * exp(j*h*phi_m)
            = j^h * J_h(2*pi*R) * exp(j*h*psi)

    Two independent effects limit which modes are actually usable, and both
    are properties of the GEOMETRY, not of the algorithm:

    1. Bessel weighting.  Mode h has gain J_h(2*pi*R).  Near a zero of J_h
       that mode carries no signal.  J_1 first vanishes at 2*pi*R = 3.8317
       (R = 0.6098 lambda), a hard blind spot for any mode-1 technique such
       as Watson-Watt.

    2. Mode aliasing.  The M-point spatial DFT around the circle cannot
       separate mode h from modes h +- M, h +- 2M, ...  The measured Y_h is
       therefore contaminated by the h-M and h+M terms.  This is the circular
       equivalent of exceeding lambda/2 spacing: the arc distance between
       neighbouring elements is 2*pi*R/M, and the contamination becomes
       serious exactly when that exceeds about lambda/2.
    """
    return _uca_modes_cached(round(float(R), 12), int(M))


@lru_cache(maxsize=512)
def _uca_modes_cached(R: float, M: int) -> UcaModes:
    z = 2.0 * np.pi * R
    Hmax = max(0, (M - 1) // 2)

    J = np.atleast_1d(jv(np.arange(0, Hmax + 1), z)).astype(float)
    ratio = np.zeros(Hmax)
    for h in range(1, Hmax + 1):
        alias = abs(jv(h - M, z)) + abs(jv(h + M, z))
        ratio[h - 1] = alias / max(abs(J[h]), np.finfo(float).eps)

    # Contiguous run of trustworthy modes starting at 1.
    H = 0
    for h in range(1, Hmax + 1):
        if abs(J[h]) > GAIN_MIN and ratio[h - 1] < RATIO_MAX:
            H = h
        else:
            break

    arc = z / M
    if Hmax < 1:
        why = "Only %d elements: no mode-1 channel exists." % M
    elif abs(J[1]) <= GAIN_MIN:
        why = ("2*pi*R = %.3f sits on the first zero of J1 (3.832), so the "
               "mode-1 gain is only %.4f." % (z, J[1]))
    elif ratio[0] >= RATIO_MAX:
        why = ("Mode aliasing: with M = %d elements around a circle of "
               "circumference %.2f wavelengths the arc spacing is %.2f "
               "wavelengths (> half a wavelength), so mode 1 is contaminated "
               "by mode %d at %.0f%% of its own amplitude."
               % (M, z, arc, 1 - M, 100 * ratio[0]))
    else:
        why = ""

    return UcaModes(z=z, arc=arc, Hmax=Hmax, H=H, J=J, ratio=ratio,
                    mode1_ok=(H >= 1), why=why)
