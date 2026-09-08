"""Element positions and the narrowband array manifold.

Python port of ``steering_vector.m``, ``array_geometry.m`` and
``nla_preset.m``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .numeric import cosd, sind


# =====================================================================
# Array manifold
# =====================================================================
def steering_vector(pos: np.ndarray, theta_deg) -> np.ndarray:
    """Narrowband array manifold for an arbitrary 2-D geometry.

    Parameters
    ----------
    pos : (M, 2) array
        Element positions in WAVELENGTHS, column 0 = x, column 1 = y.  A ULA
        lives on the x-axis, a UCA on a circle, a non-uniform linear array on
        the x-axis with irregular spacings.  Because the model is purely
        positional, one routine serves ULA, UCA and NLA with no special cases.
    theta_deg : (K,) array_like
        Azimuth angles in degrees, measured from broadside (+y) toward +x,
        i.e. the classic ULA convention where 0 deg is broadside and +90 deg
        is endfire.

    Returns
    -------
    (M, K) complex array
        The plane-wave phase seen at element m for a unit-amplitude source
        arriving from theta is

            a_m(theta) = exp(+1j*2*pi*(x_m*sin(theta) + y_m*cos(theta)))

        with x_m, y_m in wavelengths so the 2*pi/lambda factor is 1.  For a
        ULA (y_m = 0, x_m = m*d) this collapses to exp(1j*2*pi*d*m*sin(theta)).
    """
    pos = np.atleast_2d(np.asarray(pos, dtype=float))
    theta = np.atleast_1d(np.asarray(theta_deg, dtype=float)).ravel()

    # Projection of each element position onto the arrival direction, in
    # wavelengths.  The (M,1)x(1,K) outer products keep this fully vectorised.
    delay = np.outer(pos[:, 0], sind(theta)) + np.outer(pos[:, 1], cosd(theta))
    return np.exp(1j * 2.0 * np.pi * delay)


# =====================================================================
# Geometry container
# =====================================================================
@dataclass
class ArrayGeometry:
    """Element layout of one array.

    The sense antenna is deliberately kept OUT of ``pos``: every subspace
    algorithm (Bartlett, Capon, MUSIC, ESPRIT, CDF) uses only the main array,
    while Watson-Watt additionally receives ``sense_pos`` to resolve the
    180 degree front/back ambiguity.  Mixing it into the manifold would change
    the covariance dimension for all algorithms, which is not what a classical
    sense channel does.
    """

    type: str                       # 'ULA' | 'UCA' | 'NLA'
    pos: np.ndarray                 # (M, 2) positions in wavelengths
    M: int                          # element count, sense antenna NOT included
    label: str                      # short description for the plot title
    sense_pos: Optional[np.ndarray] = None   # (1, 2) or None

    @property
    def radius(self) -> float:
        """Circle radius in wavelengths (meaningful for a UCA)."""
        return float(np.linalg.norm(self.pos[0, :]))

    @property
    def aperture(self) -> float:
        """Largest extent of the array in wavelengths.

        For a linear array this is the x-extent; for a circle it is the
        diameter.  Beamwidth scales as ~0.886/aperture radians.
        """
        if self.type == "UCA":
            return 2.0 * self.radius
        return float(self.pos[:, 0].max() - self.pos[:, 0].min())

    @property
    def element_angles(self) -> np.ndarray:
        """Angular position of each element around the origin, radians."""
        return np.arctan2(self.pos[:, 1], self.pos[:, 0])


def array_geometry(cfg) -> ArrayGeometry:
    """Build element positions (in wavelengths) for the selected array.

    ``cfg`` is a :class:`~doastudio.core.scenario.Scenario` (or anything with
    the same attribute names).
    """
    array_type = str(cfg.array_type).upper()

    if array_type == "ULA":
        # Uniform linear array on the x-axis, centred on the origin so the
        # phase centre coincides with the geometric centre.
        M = max(2, int(round(cfg.M)))
        d = float(cfg.spacing)
        x = (np.arange(M) - (M - 1) / 2.0) * d
        pos = np.column_stack([x, np.zeros(M)])
        label = f"ULA  |  M = {M}  |  d = {d:.2f}λ"

    elif array_type == "UCA":
        # M elements equally spaced on a circle of radius R wavelengths,
        # first element on the +x axis.
        M = max(3, int(round(cfg.M)))
        R = float(cfg.radius)
        phi = np.arange(M) * (2.0 * np.pi / M)
        pos = np.column_stack([R * np.cos(phi), R * np.sin(phi)])
        label = f"UCA  |  M = {M}  |  R = {R:.2f}λ"

    elif array_type == "NLA":
        # Non-uniform (sparse / minimum-redundancy) linear array.  Positions
        # come straight from the user-editable vector, so any irregular
        # layout is supported.  Centre it for symmetry.
        x = np.asarray(cfg.pos_vec, dtype=float).ravel()
        if x.size < 2:
            raise ValueError("NLA needs at least 2 element positions.")
        x = x - x.mean()
        M = x.size
        pos = np.column_stack([x, np.zeros(M)])
        label = f"NLA  |  M = {M}  |  aperture = {x.max() - x.min():.2f}λ"

    else:
        raise ValueError(f'Unknown array type "{cfg.array_type}".')

    sense_pos = pos.mean(axis=0, keepdims=True) if bool(cfg.sense) else None
    return ArrayGeometry(type=array_type, pos=pos, M=M, label=label,
                         sense_pos=sense_pos)


# =====================================================================
# Sparse-array presets
# =====================================================================
# Known minimum-redundancy linear array generators, in grid units.
_MRLA = {
    2:  [0, 1],
    3:  [0, 1, 3],
    4:  [0, 1, 4, 6],
    5:  [0, 1, 4, 9, 11],
    6:  [0, 1, 4, 10, 12, 17],
    7:  [0, 1, 4, 10, 18, 23, 25],
    8:  [0, 1, 4, 10, 16, 22, 28, 30],
    9:  [0, 1, 4, 10, 16, 22, 28, 33, 35],
    10: [0, 1, 4, 10, 16, 22, 28, 34, 39, 41],
}


def nla_preset(M: int, unit: float = 0.5) -> np.ndarray:
    """Sparse (minimum-redundancy style) linear array positions.

    Returns ``M`` element x-positions in wavelengths, spaced on a ``unit``
    grid (default lambda/2), using a minimum-redundancy layout where one is
    known and a nested layout otherwise.

    Why sparse arrays are interesting here: the co-array (the set of pairwise
    differences) determines resolution, so a well-chosen sparse layout
    achieves the aperture - and therefore the beamwidth - of a much larger
    ULA with fewer elements.  The price is a ragged sidelobe structure and,
    for ESPRIT, the loss of shift invariance.
    """
    M = max(2, int(round(M)))
    if unit is None or unit <= 0:
        unit = 0.5

    if M in _MRLA:
        g = list(_MRLA[M])
    else:
        # Nested array: a dense inner ULA plus a sparse outer ULA.
        m1 = M // 2
        m2 = M - m1
        inner = list(range(m1))
        outer = [m1 * k for k in range(1, m2 + 1)]
        g = sorted(set(inner + outer))
        while len(g) < M:
            g.append(g[-1] + m1)
        g = g[:M]

    return np.asarray(g, dtype=float) * float(unit)
