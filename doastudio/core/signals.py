"""Narrowband multi-source array data with AWGN.

Python port of ``generate_snapshots.m`` and ``add_awgn.m``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .geometry import ArrayGeometry, steering_vector

#: Module-level generator so a caller who does not care about reproducibility
#: gets fresh noise on every run, exactly like MATLAB's global ``randn``.
_DEFAULT_RNG = np.random.default_rng()


def _as_rng(rng) -> np.random.Generator:
    if rng is None:
        return _DEFAULT_RNG
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def add_awgn(x: np.ndarray, snr_db: float, rng=None) -> Tuple[np.ndarray, float]:
    """Additive white Gaussian noise at a requested SNR.

    Adds noise to ``x`` so that the ratio of the MEASURED signal power in
    ``x`` to the injected noise power equals ``snr_db`` decibels - the
    behaviour of MATLAB's ``awgn(x, snr, 'measured')``, reimplemented so the
    engine depends on nothing but NumPy.

    Returns ``(y, sigma)`` where ``sigma`` is the per-component noise standard
    deviation actually used, so callers can report or reuse the exact level.
    """
    rng = _as_rng(rng)
    x = np.asarray(x)

    Ps = float(np.mean(np.abs(x) ** 2))
    if Ps <= 0:
        Ps = 1.0                                    # degenerate: assume unit power
    Pn = Ps / (10.0 ** (snr_db / 10.0))

    if np.isrealobj(x):
        sigma = np.sqrt(Pn)
        n = sigma * rng.standard_normal(x.shape)
    else:
        # Split the power equally between real and imaginary parts so that
        # E{|n|^2} = Pn.
        sigma = np.sqrt(Pn / 2.0)
        n = sigma * (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape))

    return x + n, float(sigma)


@dataclass
class Snapshots:
    """One block of received data."""

    X: np.ndarray                    # (M, N) noisy snapshots
    X_clean: np.ndarray              # (M, N) noise-free snapshots
    xs: Optional[np.ndarray]         # (N,) noisy sense channel, or None
    S: np.ndarray                    # (K, N) source waveforms
    sigma: float                     # per-component noise std
    R: np.ndarray                    # (M, M) sample covariance X X^H / N
    K: int                           # number of sources

    @property
    def measured_snr_db(self) -> float:
        """SNR actually realised in this block, for the signal plot title."""
        Ps = float(np.mean(np.abs(self.X_clean) ** 2))
        Pn = 2.0 * self.sigma ** 2
        return 10.0 * np.log10(Ps / max(Pn, np.finfo(float).eps))


def generate_snapshots(geo: ArrayGeometry, doa_deg, snr_db: float, N: int,
                       rng=None) -> Snapshots:
    """Synthesise ``N`` complex baseband snapshots for ``geo``.

    Signal model::

        X  = A(theta) @ S + noise                  (M x N)
        xs = a_sense(theta) @ S + noise            (1 x N)

    Sources are independent circular Gaussian - a good stand-in for modulated
    RF traffic - so the source covariance is diagonal and the subspace
    assumptions behind MUSIC and ESPRIT hold.
    """
    rng = _as_rng(rng)
    doa = np.atleast_1d(np.asarray(doa_deg, dtype=float)).ravel()
    K = doa.size
    N = int(N)

    A = steering_vector(geo.pos, doa)                      # M x K

    # Unit-power complex Gaussian sources, mutually uncorrelated.
    S = (rng.standard_normal((K, N)) + 1j * rng.standard_normal((K, N))) / np.sqrt(2.0)

    X_clean = A @ S

    # Noise is referenced to the measured array-signal power, i.e. the SNR
    # printed in the UI is the per-element SNR before beamforming gain.
    X, sigma = add_awgn(X_clean, snr_db, rng=rng)

    # ---- Sense channel ----------------------------------------------
    # The sense antenna is an omnidirectional element at the array centroid.
    # Its response therefore has (ideally) unit gain and zero phase slope
    # versus angle, which is exactly the property Watson-Watt exploits to
    # break the 180 degree ambiguity of a difference channel.
    if geo.sense_pos is not None:
        a_s = steering_vector(geo.sense_pos, doa)          # 1 x K
        xs_clean = (a_s @ S).ravel()
        # Reuse the same sigma so the sense channel sits at the same noise
        # floor as the array elements.
        xs = xs_clean + sigma * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
    else:
        xs = None

    return Snapshots(X=X, X_clean=X_clean, xs=xs, S=S, sigma=sigma,
                     R=(X @ X.conj().T) / N, K=K)
