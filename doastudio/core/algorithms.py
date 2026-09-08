"""The six direction-finding estimators, their registry and the dispatcher.

Python port of ``bartlett_doa.m``, ``capon_doa.m``,
``corr_interferometry_doa.m``, ``music_doa.m``, ``esprit_doa.m``,
``watson_watt_doa.m``, ``doa_algorithm_list.m`` and ``doa_run_algorithms.m``.

Every estimator takes the same arguments - ``(geo, data, scan_deg, K)`` - and
returns an :class:`AlgoResult`, so the dispatcher is a plain loop and adding a
seventh estimator only touches :data:`ALGORITHMS`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import numpy as np
from scipy.special import jv

from .geometry import ArrayGeometry, steering_vector
from .numeric import EPS, asind, db10, db20, hermitian_eig, i_pow, rcond
from .peaks import pick_peaks
from .signals import Snapshots
from .subarrays import esprit_subarrays, uca_modes

#: dB floor of the cosmetic pseudo-spectrum drawn for search-free estimators.
SPECTRUM_FLOOR = -60.0


# =====================================================================
# Result container
# =====================================================================
@dataclass
class AlgoResult:
    """What every estimator returns."""

    name: str                                   # display name
    spectrum: np.ndarray                        # dB, normalised to a 0 dB peak
    angles: np.ndarray                          # K estimated DoAs, ascending
    time: float                                 # seconds spent estimating
    valid: bool = True                          # False = refused to run
    note: str = ""                              # why it refused, or a caveat
    has_spectrum: bool = True                   # False = curve is cosmetic only
    ambiguous: bool = False                     # bearing is ambiguous

    # filled in by the dispatcher from the registry
    key: str = ""
    short: str = ""
    color: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def angles_text(self) -> str:
        if not self.valid:
            return "not applicable"
        return ", ".join("%.2f" % a for a in np.atleast_1d(self.angles)
                         if not np.isnan(a)) or "-"


def _floor_spectrum(scan_deg: np.ndarray) -> np.ndarray:
    return np.full(np.asarray(scan_deg).size, SPECTRUM_FLOOR, dtype=float)


def _pseudo_spectrum(angles, scan_deg, width: float = 0.75) -> np.ndarray:
    """Cosmetic curve so a search-free estimator can share the spectrum axes.

    A -60 dB floor with narrow 0 dB spikes at the estimates.  ``has_spectrum``
    is False on those results so the UI can draw them dotted and label them
    "estimate only" - the curve carries no information beyond the peak
    positions.
    """
    scan = np.asarray(scan_deg, dtype=float).ravel()
    spec = np.full(scan.size, SPECTRUM_FLOOR, dtype=float)
    for a in np.atleast_1d(np.asarray(angles, dtype=float)):
        if np.isnan(a):
            continue
        spec = np.maximum(spec, SPECTRUM_FLOOR
                          - SPECTRUM_FLOOR * np.exp(-((scan - a) / width) ** 2))
    return spec


# =====================================================================
# 1. Bartlett - conventional beamforming
# =====================================================================
def bartlett_doa(geo: ArrayGeometry, data: Snapshots, scan_deg, K: int) -> AlgoResult:
    """Conventional (Bartlett) beamforming spatial spectrum.

    Steer the array to each candidate angle and read the output power.  With
    ``w = a(theta)/||a||`` the output power is

        P(theta) = w^H R w = a^H(theta) R a(theta) / (a^H a)

    This is the Fourier / periodogram estimator: robust and never
    ill-conditioned, but its resolution is limited by the beamwidth
    (~lambda/aperture), so two sources closer than one beamwidth merge into a
    single lobe no matter how high the SNR is.
    """
    t0 = time.perf_counter()

    A = steering_vector(geo.pos, scan_deg)                  # M x G
    R = data.R

    # Vectorised quadratic form: sum over elements of conj(a) * (R a).
    num = np.real(np.sum(A.conj() * (R @ A), axis=0))       # 1 x G
    den = np.sum(np.abs(A) ** 2, axis=0)                    # normalisation (= M)
    P = num / den

    spec = db10(np.maximum(P, EPS))
    return AlgoResult(name="Bartlett", spectrum=spec,
                      angles=pick_peaks(spec, scan_deg, K),
                      time=time.perf_counter() - t0)


# =====================================================================
# 2. Capon / MVDR
# =====================================================================
def capon_doa(geo: ArrayGeometry, data: Snapshots, scan_deg, K: int) -> AlgoResult:
    """Capon / MVDR (minimum variance distortionless response) spectrum.

    Minimise output power subject to unit gain toward theta,

        min w^H R w  s.t.  w^H a(theta) = 1   ->   w = R^-1 a / (a^H R^-1 a)

    giving ``P(theta) = 1 / (a^H(theta) R^-1 a(theta))``.

    The inverse covariance places nulls on interferers, so MVDR resolves
    sources well inside the Bartlett beamwidth.  The price is that R must be
    invertible: it needs N > M snapshots, and it degrades badly when the
    sources are correlated or N is close to M.  A small diagonal loading term
    keeps the inverse numerically sane in exactly those cases, and a
    pseudo-inverse is the fallback if R is still rank deficient.
    """
    t0 = time.perf_counter()

    R = data.R
    M = R.shape[0]

    # Diagonal loading at ~ -30 dB of the average eigenvalue.  This is the
    # standard robust-MVDR fix for snapshot-starved covariance estimates.
    load_level = 1e-3 * float(np.real(np.trace(R))) / M
    Rl = R + load_level * np.eye(M)

    if rcond(Rl) > 1e-12:
        Rinv = np.linalg.inv(Rl)
    else:
        Rinv = np.linalg.pinv(Rl)      # rank deficient: pseudo-inverse

    A = steering_vector(geo.pos, scan_deg)                       # M x G
    den = np.real(np.sum(A.conj() * (Rinv @ A), axis=0))         # a^H R^-1 a
    P = 1.0 / np.maximum(den, EPS)

    spec = db10(P)
    return AlgoResult(name="Capon / MVDR", spectrum=spec,
                      angles=pick_peaks(spec, scan_deg, K),
                      time=time.perf_counter() - t0)


# =====================================================================
# 3. Correlative interferometry (CDF)
# =====================================================================
def corr_interferometry_doa(geo: ArrayGeometry, data: Snapshots, scan_deg,
                            K: int) -> AlgoResult:
    """Correlative interferometry / correlative direction finding.

    The technique used by most operational DF receivers.  Instead of forming
    a beam it compares the MEASURED complex inter-element pattern against a
    precomputed table of reference patterns (the "correlation table", or
    antenna calibration table) and reports the angle whose reference pattern
    correlates best.

    1. Reference table: ``A[:, g]`` = steering vector on the scan grid.  In a
       real receiver this table is measured on a test range, which is why CDF
       tolerates arbitrary, even badly calibrated, geometries.
    2. Measured pattern: the dominant eigenvector of the sample covariance.
       Using the principal eigenvector rather than a single snapshot averages
       the noise over all N samples and is the maximum-likelihood
       single-wavefront estimate.
    3. Correlation coefficient, magnitude only so that an unknown common
       phase and gain do not matter::

           rho(g) = |a_g^H v| / (||a_g|| * ||v||)        in [0, 1]

    4. The reported spectrum is ``20*log10(rho)`` normalised to 0 dB.

    Character: CDF is a single-wavefront estimator.  With two co-channel
    emitters the best-correlating reference lies between them, so CDF biases
    toward the power centroid rather than resolving both.  That behaviour is
    deliberate here, and the app flags it.
    """
    t0 = time.perf_counter()

    A = steering_vector(geo.pos, scan_deg)              # M x G reference table

    # Dominant eigenvector = measured wavefront estimate.
    _, U = hermitian_eig(data.R)
    v = U[:, 0]

    num = np.abs(A.conj().T @ v)                        # G
    den = np.sqrt(np.sum(np.abs(A) ** 2, axis=0)) * np.linalg.norm(v)
    rho = num / np.maximum(den, EPS)

    spec = db20(np.maximum(rho, EPS))
    return AlgoResult(name="CDF (Correlative)", spectrum=spec,
                      angles=pick_peaks(spec, scan_deg, K),
                      time=time.perf_counter() - t0)


# =====================================================================
# 4. MUSIC
# =====================================================================
def music_doa(geo: ArrayGeometry, data: Snapshots, scan_deg, K: int) -> AlgoResult:
    """MUltiple SIgnal Classification via noise-subspace projection.

    Eigendecompose ``R = U L U^H``.  The K eigenvectors with the largest
    eigenvalues span the signal subspace; the remaining M-K span the noise
    subspace ``En``.  Any true steering vector is orthogonal to ``En``, so

        P(theta) = 1 / || En^H a(theta) ||^2

    peaks sharply at the true DoAs.  Resolution is not beamwidth-limited,
    which is why MUSIC separates sources Bartlett cannot - provided that
    (a) K is known and correct, (b) K < M, and (c) the sources are not fully
    correlated (coherent multipath collapses the signal subspace rank and
    MUSIC then misses sources).
    """
    t0 = time.perf_counter()

    R = data.R
    M = R.shape[0]
    K = max(1, min(int(K), M - 1))          # need at least one noise eigenvector

    w, U = hermitian_eig(R)
    En = U[:, K:]                           # noise subspace, M x (M-K)

    A = steering_vector(geo.pos, scan_deg)  # M x G
    proj = En.conj().T @ A                  # (M-K) x G
    den = np.sum(np.abs(proj) ** 2, axis=0)
    P = 1.0 / np.maximum(den, EPS)

    spec = db10(P)
    return AlgoResult(name="MUSIC", spectrum=spec,
                      angles=pick_peaks(spec, scan_deg, K),
                      time=time.perf_counter() - t0)


# =====================================================================
# 5. ESPRIT
# =====================================================================
def _esprit_core(R: np.ndarray, K: int, i1, i2):
    """TLS-ESPRIT rotation phases from a covariance matrix.

    Returns the K rotation phases in radians, or None when the covariance is
    rank deficient.
    """
    w, U = hermitian_eig(R)
    K = max(1, min(int(K), R.shape[0] - 1))
    if w[K - 1] <= 0:
        return None
    Es = U[:, :K]                           # signal subspace

    E1 = Es[np.asarray(i1, dtype=int), :]
    E2 = Es[np.asarray(i2, dtype=int), :]

    # --- Total least squares -----------------------------------------
    # Solve E1 @ PSI ~ E2 in the TLS sense: the null space of [E1 E2] gives
    # the balanced solution.  Both matrices carry noise, so TLS is the right
    # error model (plain LS assumes E1 is exact).
    C = np.hstack([E1, E2])
    # full_matrices=True so V is always 2K x 2K, matching MATLAB's svd(C, 0).
    _, _, Vh = np.linalg.svd(C, full_matrices=True)
    V = Vh.conj().T
    V12 = V[:K, K:2 * K]
    V22 = V[K:2 * K, K:2 * K]

    if rcond(V22) < 1e-12:
        PSI = np.linalg.pinv(E1) @ E2       # fall back to LS
    else:
        PSI = -V12 @ np.linalg.inv(V22)

    return np.angle(np.linalg.eigvals(PSI))


def _fold_into_sector(ang: np.ndarray) -> np.ndarray:
    """Wrap angles into the -90..+90 scan sector the app displays."""
    ang = np.mod(ang + 90.0, 360.0) - 90.0
    ang = np.where(ang > 90.0, 180.0 - ang, ang)
    ang = np.where(ang < -90.0, -180.0 - ang, ang)
    return ang


def esprit_doa(geo: ArrayGeometry, data: Snapshots, scan_deg, K: int) -> AlgoResult:
    """Estimation of Signal Parameters via Rotational Invariance Techniques.

    ESPRIT is search-free, so ``scan_deg`` is used only to render a display
    pseudo-spectrum; ``has_spectrum`` is False to flag that the curve is
    cosmetic.

    Subarray construction - the part that decides where ESPRIT is legal.
    ESPRIT needs two subarrays that are identical arrays rigidly displaced by
    a fixed vector.  Their signal subspaces are then related by a diagonal
    rotation whose phases are the DoAs::

        A2 = A1 @ PHI,   PHI = diag(exp(j*2*pi*<delta, u(theta_k)>))

    1. ULA / uniform-subset NLA: :func:`esprit_subarrays` scans the geometry
       for the largest set of element pairs sharing one displacement.  For a
       ULA that is the classic 0..M-2 / 1..M-1 split.  A sparse NLA only
       qualifies if it happens to contain a shift-invariant subset.

    2. UCA: a circular array has NO shift-invariant element pair - every
       chord points in a different direction.  Invariance is instead
       manufactured in BEAMSPACE with the Davies phase-mode transform.
       Jacobi-Anger expands the UCA manifold as

           a_m(psi) = sum_h j^h J_h(2*pi*R) exp(j*h*(psi - phi_m))

       so pre-multiplying the data by

           F = diag(1 / (j^h J_h(2*pi*R))) @ (1/M) exp(j*h*phi_m)

       yields a VIRTUAL ULA whose h-th "element" responds as exp(j*h*psi).
       Standard ESPRIT then applies, with psi = 90 deg - theta.  Caveats
       surfaced as limitations in the UI: the transform colours the noise,
       needs M >= 2H+1 modes, and only modes with a non-vanishing
       J_h(2*pi*R) are usable.
    """
    t0 = time.perf_counter()
    scan = np.asarray(scan_deg, dtype=float).ravel()

    def refuse(note: str) -> AlgoResult:
        return AlgoResult(name="ESPRIT", spectrum=_floor_spectrum(scan),
                          angles=np.full(K, np.nan), time=time.perf_counter() - t0,
                          valid=False, note=note, has_spectrum=False)

    X = data.X
    N = X.shape[1]

    if geo.type == "UCA":
        # ---------------- Beamspace (phase-mode) route -----------------
        R_circ = geo.radius
        phi = geo.element_angles
        M = geo.pos.shape[0]

        # Usable modes: Bessel gain must be significant AND the mode must not
        # be swamped by its aliases at h +- M.  uca_modes owns that rule.
        u = uca_modes(R_circ, M)
        n_modes = u.n_modes

        if n_modes < K + 1:
            return refuse(
                "UCA beamspace supports only %d usable phase mode(s) for "
                "R = %.2f wavelengths with M = %d, but ESPRIT needs at least "
                "K+1 = %d virtual elements. %s"
                % (n_modes, R_circ, M, K + 1, u.why))

        h = np.arange(-u.H, u.H + 1)
        W = np.exp(1j * np.outer(h, phi)) / M                # (2H+1) x M
        scale = i_pow(h) * jv(h, 2.0 * np.pi * R_circ)       # Bessel/mode weights
        F = (1.0 / scale)[:, None] * W                       # Davies transform

        Xv = F @ X                                           # virtual-ULA snapshots
        # Virtual element h responds as exp(j*h*psi) -> unit "spacing" in the
        # psi variable, so the rotation phase IS psi (radians).
        Rv = (Xv @ Xv.conj().T) / N
        phases = _esprit_core(Rv, K, np.arange(n_modes - 1), np.arange(1, n_modes))
        if phases is None:
            return refuse("Beamspace covariance was rank deficient; ESPRIT aborted.")

        psi_deg = np.rad2deg(phases)
        ang = _fold_into_sector(90.0 - psi_deg)              # back to broadside ref

    else:
        # ---------------- Element-space route (ULA / uniform NLA) -------
        i1, i2, delta = esprit_subarrays(geo.pos)
        if i1 is None or i1.size < K + 1:
            return refuse(
                "No translationally invariant subarray pair of sufficient "
                "size exists for this geometry, so the rotational-invariance "
                "model does not hold.")

        R = (X @ X.conj().T) / N
        phases = _esprit_core(R, K, i1, i2)
        if phases is None:
            return refuse("Sample covariance was rank deficient; ESPRIT aborted.")

        # phase = 2*pi*<delta, u(theta)>.  For a linear array delta is along
        # x, so phase = 2*pi*dx*sin(theta).
        dx = float(delta[0])
        s = np.clip(phases / (2.0 * np.pi * dx), -1.0, 1.0)  # |sin| <= 1
        ang = asind(s)

    ang = np.sort(np.asarray(ang, dtype=float).ravel())
    return AlgoResult(name="ESPRIT", spectrum=_pseudo_spectrum(ang, scan),
                      angles=ang, time=time.perf_counter() - t0,
                      has_spectrum=False)


# =====================================================================
# 6. Watson-Watt
# =====================================================================
def watson_watt_doa(geo: ArrayGeometry, data: Snapshots, scan_deg,
                    K: int) -> AlgoResult:
    """Watson-Watt DF adapted to an array geometry, with sense antenna.

    Watson-Watt is search-free and single-wavefront, so it always returns ONE
    angle (replicated to length K for bookkeeping) and ``has_spectrum`` is
    False: the plotted curve is a cosmetic marker.

    IMPORTANT SIMPLIFICATION.  True Watson-Watt uses two physically
    ORTHOGONAL crossed LOOPS whose figure-of-eight voltage patterns are
    ``X = E cos(theta)`` (N-S loop) and ``Y = E sin(theta)`` (E-W loop) plus
    an omnidirectional SENSE antenna ``S = E``.  The bearing is then
    ``theta = atan2(Y, X)`` and, because a figure-of-eight pattern is
    symmetric, the result is ambiguous by 180 deg.  Comparing the loop phase
    with the sense phase (the "cardioid conversion") removes that ambiguity.

    This app has no loops - only omnidirectional array elements.  The X/Y
    (sum/difference) channel pair is therefore SYNTHESISED from array
    outputs.  This is an adaptation, not the classical instrument, and the
    two geometries need different constructions:

    (A) Linear array -> symmetric pair sum-difference.
        Take a symmetric element pair at +h and -h (baseline b = 2h).  Their
        cross-correlation over the snapshots is

            c = <x(+h) x*(-h)> = Ps * exp(j*2*pi*b*sin(theta))

        so with ``X = Re{c}`` and ``Y = Im{c}``,
        ``u = atan2(Y, X) = 2*pi*b*sin(theta)`` - exactly the Watson-Watt
        arctangent applied to a sum/difference channel pair.

        ROLE OF THE SENSE ANTENNA HERE: correlating each element against the
        centroid sense element gives the HALF-baseline phases
        ``+-2*pi*h*sin(theta)``, which are unambiguous for h <= 0.5 lambda
        even when the full baseline b is not.  The sense channel is used to
        UNWRAP u, extending the unambiguous baseline from 0.5 to about
        1 wavelength.

        WHY NOT FRONT/BACK: a line of omni elements cannot distinguish theta
        from 180-theta at all - that ambiguity is geometric, and an omni
        sense element at the centroid carries no directional information to
        break it.  (The app scans -90..+90 deg, so the mirror image simply
        lies outside the displayed sector.)

    (B) UCA -> phase modes 1 and 0 behave like a loop and a sense.
        Jacobi-Anger on the circular manifold gives, for mode h,

            Y_h = (1/M) sum_m x_m exp(j*h*phi_m)
                = j^h J_h(2*pi*R) exp(j*h*psi),     psi = 90 - theta

        Mode +-1 is the direct analogue of a crossed-loop pair (its pattern
        is a rotating figure-of-eight) and mode 0 the analogue of the sense
        antenna (omnidirectional, J_0 weighted).  Correlating mode 1 against
        the sense reference and dividing out the known ``j*J_1`` factor
        yields ``z ~ exp(j*psi)``, whose real and imaginary parts ARE the
        Watson-Watt X and Y channels.  This is the genuine cardioid
        conversion: with the sense reference the bearing is unambiguous over
        the full 360 deg; without it only ``Y_1 conj(Y_-1)`` is available,
        which gives ``2*psi`` and therefore the classical 180 deg ambiguity.
    """
    t0 = time.perf_counter()
    scan = np.asarray(scan_deg, dtype=float).ravel()

    X = data.X
    xs = data.xs                       # sense channel, None if disabled
    note = ""
    ambiguous = False

    if geo.type == "UCA":
        # -------------------- (B) circular array ----------------------
        R_circ = geo.radius
        phi = geo.element_angles
        M = geo.pos.shape[0]

        J1 = float(jv(1, 2.0 * np.pi * R_circ))
        J0 = float(jv(0, 2.0 * np.pi * R_circ))

        # The mode-1 loop gain IS J_1(2*pi*R), and mode 1 is also aliased by
        # modes 1 +- M.  Either effect can leave the loop channel carrying no
        # usable bearing information, in which case the answer would be wrong
        # rather than merely noisy - so refuse instead of reporting.
        u = uca_modes(R_circ, M)
        if not u.mode1_ok:
            return AlgoResult(
                name="Watson-Watt", spectrum=_floor_spectrum(scan),
                angles=np.full(K, np.nan), time=time.perf_counter() - t0,
                valid=False, has_spectrum=False,
                note=("Watson-Watt needs a trustworthy mode-1 (loop) channel "
                      "on a circular array, and this geometry does not provide "
                      "one. %s Radii near 0.5 wavelengths with M >= 8, or "
                      "1.0 wavelengths with M >= 12, work well." % u.why))

        y1 = (np.exp(1j * phi) @ X) / M            # mode +1 ("loop")
        ym1 = (np.exp(-1j * phi) @ X) / M          # mode -1

        if xs is not None:
            ref = xs                               # true omni sense antenna
            gref = 1.0 + 0j                        # its response is unity
        else:
            ref = X.mean(axis=0)                   # mode 0 = synthetic sense
            gref = J0 + 0j

        # Sense-referenced loop correlation -> Watson-Watt X/Y channels.
        c = np.mean(y1 * np.conj(ref))             # ~ Ps * j*J1*conj(gref)*exp(j*psi)
        z = c / (1j * J1 * np.conj(gref))          # ~ Ps * exp(j*psi)
        psi = np.arctan2(np.imag(z), np.real(z))

        if xs is None and abs(J0) < 1e-3:
            # Mode 0 is itself nulled: fall back to the ambiguous loop-only
            # estimate and say so.
            psi = np.angle(np.mean(y1 * np.conj(ym1))) / 2.0
            ambiguous = True
            note = "No usable sense reference: bearing is ambiguous by 180 deg."

        ang = float(_fold_into_sector(np.array([90.0 - np.rad2deg(psi)]))[0])

    else:
        # -------------------- (A) linear arrays -----------------------
        x = geo.pos[:, 0]
        order = np.argsort(x)          # symmetric pairs about the phase centre
        n = order.size

        # Choose the widest symmetric pair whose baseline stays inside the
        # unambiguous region, unless the sense antenna is available to
        # unwrap - then allow the widest pair up to 1 wavelength.
        limit = 1.0 if xs is not None else 0.5

        best_pair = None
        best_b = 0.0
        for q in range(n // 2):
            p1 = int(order[q])
            p2 = int(order[n - 1 - q])
            b = abs(x[p2] - x[p1])
            if b <= limit + 1e-9 and b > best_b:
                best_b = b
                best_pair = (p1, p2)

        if best_pair is None:
            # Every symmetric pair is too wide: take the narrowest one and
            # declare the result ambiguous (grating lobes in sin(theta)).
            q = n // 2
            best_pair = (int(order[q - 1]), int(order[q]))
            best_b = abs(x[best_pair[1]] - x[best_pair[0]])
            ambiguous = True
            note = ("Narrowest available element pair is %.2f wavelengths apart "
                    "(> 0.5), so the sum/difference phase wraps and the bearing "
                    "is ambiguous." % best_b)

        p_minus, p_plus = best_pair          # elements at -h and +h
        b = float(x[p_plus] - x[p_minus])    # signed baseline, wavelengths

        c = np.mean(X[p_plus, :] * np.conj(X[p_minus, :]))
        u_phase = np.arctan2(np.imag(c), np.real(c))   # = 2*pi*b*sin(theta)

        if xs is not None:
            # Sense-referenced half-baseline phases, unambiguous for h <= 0.5.
            up = np.angle(np.mean(X[p_plus, :] * np.conj(xs)))
            um = np.angle(np.mean(X[p_minus, :] * np.conj(xs)))
            u_coarse = up - um                          # ~ 2*pi*b*sin(theta)
            # Unwrap the fine estimate onto the coarse one.
            u_phase = u_phase + 2.0 * np.pi * round((u_coarse - u_phase) / (2.0 * np.pi))

        s = u_phase / (2.0 * np.pi * b)
        if abs(s) > 1.0:
            s = float(np.clip(s, -1.0, 1.0))
            ambiguous = True
            if not note:
                note = "Sum/difference phase exceeded the unambiguous range."
        ang = float(asind(s))

    angles = np.full(max(1, int(K)), ang, dtype=float)   # single-wavefront
    return AlgoResult(name="Watson-Watt",
                      spectrum=_pseudo_spectrum([ang], scan, width=1.0),
                      angles=angles, time=time.perf_counter() - t0,
                      note=note, has_spectrum=False, ambiguous=ambiguous)


# =====================================================================
# Registry + dispatcher
# =====================================================================
@dataclass(frozen=True)
class AlgorithmSpec:
    key: str
    label: str
    short: str
    fcn: Callable[..., AlgoResult]
    color: Tuple[float, float, float]
    search_free: bool

    @property
    def hex_color(self) -> str:
        r, g, b = self.color
        return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))


#: Single source of truth.  The app builds its checkbox column, plot legend
#: and sweep lines from this list, so adding an estimator here is enough to
#: expose it everywhere.
ALGORITHMS: List[AlgorithmSpec] = [
    AlgorithmSpec("bartlett", "Bartlett (Conventional Beamforming)", "Bartlett",
                  bartlett_doa, (0.00, 0.45, 0.74), False),
    AlgorithmSpec("capon", "Capon / MVDR", "Capon/MVDR",
                  capon_doa, (0.85, 0.33, 0.10), False),
    AlgorithmSpec("cdf", "Correlative Direction Finding (CDF)", "CDF",
                  corr_interferometry_doa, (0.93, 0.69, 0.13), False),
    AlgorithmSpec("music", "MUSIC", "MUSIC",
                  music_doa, (0.49, 0.18, 0.56), False),
    AlgorithmSpec("esprit", "ESPRIT", "ESPRIT",
                  esprit_doa, (0.47, 0.67, 0.19), True),
    AlgorithmSpec("ww", "Watson-Watt (+ sense antenna)", "Watson-Watt",
                  watson_watt_doa, (0.30, 0.75, 0.93), True),
]

ALGORITHM_BY_KEY = {a.key: a for a in ALGORITHMS}


def run_algorithms(geo: ArrayGeometry, data: Snapshots, scan_deg, K: int,
                   keys: Sequence[str]) -> List[AlgoResult]:
    """Run the selected estimators on one snapshot block.

    Returns one :class:`AlgoResult` per requested key, in the requested
    order.  Every estimator shares the signature ``(geo, data, scan_deg, K)``,
    so this dispatcher is a plain loop with no per-algorithm special cases.
    """
    results: List[AlgoResult] = []
    for key in keys:
        spec = ALGORITHM_BY_KEY.get(key)
        if spec is None:
            continue
        r = spec.fcn(geo, data, scan_deg, K)
        r.key = spec.key
        r.short = spec.short
        r.color = spec.color
        results.append(r)
    return results
