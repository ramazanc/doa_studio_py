"""The scenario description shared by every part of the app.

The MATLAB app passed an ad-hoc ``cfg`` struct around; a dataclass makes the
same information self-documenting and lets the sweep engine produce modified
copies with :func:`dataclasses.replace` semantics.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .geometry import nla_preset

#: Default scan grid: -90 to +90 degrees.
SCAN_STEP = 0.5


def default_scan(step: float = SCAN_STEP) -> np.ndarray:
    return np.arange(-90.0, 90.0 + step / 2, step)


def default_doa(K: int) -> np.ndarray:
    """A sensible source constellation spanning +-50 degrees."""
    K = max(1, int(K))
    if K == 1:
        return np.array([0.0])
    return np.linspace(-50.0, 50.0, K)


@dataclass
class Scenario:
    """Everything needed to synthesise data and run the estimators."""

    array_type: str = "ULA"          # 'ULA' | 'UCA' | 'NLA'
    M: int = 8                       # element count (ULA / UCA)
    spacing: float = 0.5             # inter-element spacing, wavelengths
    radius: float = 0.5              # circle radius, wavelengths (UCA)
    pos_vec: np.ndarray = field(default_factory=lambda: nla_preset(8, 0.5))
    sense: bool = True               # omni sense antenna at the centroid

    doa: np.ndarray = field(default_factory=lambda: np.array([-20.0, 30.0]))
    K: int = 2                       # number of sources
    snr: float = 10.0                # per-element SNR in dB
    N: int = 200                     # snapshots
    trials: int = 1                  # Monte-Carlo trials for the summary table

    tol: Optional[float] = None      # resolution tolerance; None = auto
    scan: np.ndarray = field(default_factory=default_scan)
    keys: List[str] = field(default_factory=lambda: [
        "bartlett", "capon", "cdf", "music", "esprit", "ww"])

    # -----------------------------------------------------------------
    def __post_init__(self):
        self.doa = np.atleast_1d(np.asarray(self.doa, dtype=float)).ravel()
        self.pos_vec = np.atleast_1d(np.asarray(self.pos_vec, dtype=float)).ravel()
        self.scan = np.asarray(self.scan, dtype=float).ravel()
        if self.array_type.upper() == "NLA":
            self.M = int(self.pos_vec.size)

    def copy(self, **changes) -> "Scenario":
        """Return a copy with the named fields replaced."""
        return dataclasses.replace(self, **changes)

    @property
    def element_spacing(self) -> float:
        """The spacing control, interpreted for the current array type."""
        return self.radius if self.array_type.upper() == "UCA" else self.spacing
