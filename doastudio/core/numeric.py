"""Small numeric helpers shared by the whole engine.

MATLAB has degree-argument trigonometry (``sind``/``cosd``/``asind``) and a
1-norm reciprocal-condition estimator (``rcond``) in the language itself.
NumPy does not, so they live here and every other module imports them from
one place.  Keeping them together also keeps the DSP modules readable: the
ported formulas then look almost exactly like the MATLAB originals.
"""

from __future__ import annotations

import numpy as np

# MATLAB's ``eps`` (2.2204e-16).  Used as the floor before every log10 so a
# zero-valued spectrum bin cannot produce -inf dB.
EPS = float(np.finfo(float).eps)


def sind(deg):
    """Sine of an angle given in degrees."""
    return np.sin(np.deg2rad(deg))


def cosd(deg):
    """Cosine of an angle given in degrees."""
    return np.cos(np.deg2rad(deg))


def asind(x):
    """Arcsine in degrees."""
    return np.rad2deg(np.arcsin(x))


def rcond(A: np.ndarray) -> float:
    """Reciprocal condition number in the 1-norm, MATLAB's ``rcond``.

    Returns 0.0 for a numerically singular matrix instead of raising, which
    is what the callers want: they use the value purely to decide between a
    direct inverse and a pseudo-inverse.
    """
    A = np.asarray(A)
    if A.size == 0:
        return 0.0
    try:
        c = np.linalg.cond(A, 1)
    except np.linalg.LinAlgError:
        return 0.0
    if not np.isfinite(c) or c == 0.0:
        return 0.0
    return float(1.0 / c)


def i_pow(h: np.ndarray) -> np.ndarray:
    """``1j ** h`` for integer ``h``, computed exactly.

    ``np.power(1j, h)`` goes through complex log/exp and leaves rounding
    dust of order 1e-17 in the component that should be exactly zero.  The
    Davies phase-mode transform divides by this factor, so an exact table
    lookup on ``h mod 4`` keeps the virtual-ULA manifold clean.
    """
    table = np.array([1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j])
    return table[np.mod(np.asarray(h, dtype=int), 4)]


def hermitian_eig(R: np.ndarray):
    """Eigendecomposition of a Hermitian matrix, eigenvalues descending.

    Returns ``(eigenvalues, eigenvectors)`` with the columns of the second
    output ordered to match.  ``R`` is symmetrised first because a sample
    covariance accumulated in floating point is only Hermitian to within
    rounding, and ``eigh`` reads a single triangle.
    """
    Rh = (R + R.conj().T) / 2.0
    w, V = np.linalg.eigh(Rh)          # ascending
    order = np.argsort(w)[::-1]        # descending
    return w[order], V[:, order]


def db10(power: np.ndarray) -> np.ndarray:
    """10*log10 of a power spectrum, normalised so the peak sits at 0 dB."""
    p = np.maximum(np.asarray(power, dtype=float), EPS)
    return 10.0 * np.log10(p / p.max())


def db20(amplitude: np.ndarray) -> np.ndarray:
    """20*log10 of an amplitude spectrum, normalised to a 0 dB peak."""
    a = np.maximum(np.asarray(amplitude, dtype=float), EPS)
    return 20.0 * np.log10(a / a.max())


def parse_number_list(text: str) -> np.ndarray:
    """Parse "-20, 30" / "0 0.5 2 3" into a float array.

    Replaces MATLAB's ``str2num`` for the free-text entry boxes.  Anything
    that is not a number is skipped rather than raising, because these
    strings arrive straight from a widget while the user is still typing.
    """
    if text is None:
        return np.zeros(0)
    out = []
    for tok in str(text).replace(";", ",").replace("[", " ").replace("]", " ").replace(",", " ").split():
        try:
            out.append(float(tok))
        except ValueError:
            continue
    return np.asarray(out, dtype=float)


def format_number_list(values, fmt: str = "%g") -> str:
    """Inverse of :func:`parse_number_list`, for writing back into a widget."""
    return ", ".join(fmt % v for v in np.atleast_1d(values))
