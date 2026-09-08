"""Diagnose why the current scenario is hard, and for whom.

Python port of ``doa_limitations.m``.  Every entry is written as physics, not
as a warning label: the point is that the user can see the estimator failing
on the plot and read the reason next to it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .algorithms import AlgoResult
from .geometry import ArrayGeometry
from .scenario import Scenario
from .subarrays import esprit_subarrays, uca_modes

CRIT, WARN, INFO = "crit", "warn", "info"

#: Prefix shown in the list box for each severity.
SEVERITY_PREFIX = {CRIT: "[!]", WARN: "[~]", INFO: "[i]"}
SEVERITY_TAG = {
    CRIT: "BLOCKING LIMITATION",
    WARN: "PERFORMANCE LIMITATION",
    INFO: "NOTE",
}


@dataclass
class Limitation:
    severity: str        # 'crit' | 'warn' | 'info'
    title: str           # one-line headline shown in the list box
    text: str            # full explanation of WHY the scenario is limiting
    algos: str           # which estimators are affected

    @property
    def list_entry(self) -> str:
        return "%s  %s" % (SEVERITY_PREFIX.get(self.severity, "[i]"), self.title)


def doa_limitations(cfg: Scenario, geo: ArrayGeometry,
                    results: Optional[Sequence[AlgoResult]] = None) -> List[Limitation]:
    """Return the findings that populate the interactive limitation panel."""
    out: List[Limitation] = []

    def add(severity, title, text, algos):
        out.append(Limitation(severity, title, text, algos))

    M = geo.M
    K = int(cfg.K)
    N = int(cfg.N)
    snr = float(cfg.snr)
    doa = np.sort(np.asarray(cfg.doa, dtype=float).ravel())
    sel = set(cfg.keys)

    # ---------------------------------------------------------------
    # 1. Degrees of freedom: a covariance matrix cannot hold more
    #    sources than it has noise-subspace dimensions.
    # ---------------------------------------------------------------
    if K >= M:
        add(CRIT,
            "K = %d sources with only M = %d elements" % (K, M),
            "An M-element array spans an M-dimensional observation space. MUSIC "
            "needs at least one noise eigenvector, so it can handle at most "
            "M-1 = %d sources; ESPRIT needs the signal subspace to be smaller "
            "than each subarray. With K >= M the signal subspace fills the whole "
            "space, the noise subspace is empty and the subspace estimators have "
            "nothing to project onto. Bartlett and Capon still produce a spectrum "
            "but it can only show M-1 independent lobes." % (M - 1),
            "MUSIC, ESPRIT")
    elif K > M - 2:
        add(WARN,
            "K = %d sources leaves only %d noise eigenvector(s)" % (K, M - K),
            "With almost no noise subspace left, the MUSIC denominator is "
            "estimated from very few eigenvectors, so its peaks broaden and its "
            "variance rises sharply. More elements or fewer sources restores the "
            "usual sharp nulls.",
            "MUSIC, ESPRIT")

    # ---------------------------------------------------------------
    # 2. Snapshot support for the sample covariance.
    # ---------------------------------------------------------------
    if N < M:
        add(CRIT,
            "N = %d snapshots < M = %d elements" % (N, M),
            "The sample covariance R = X X^H / N has rank at most N. With N < M "
            "it is singular, so the Capon inverse does not exist (the app falls "
            "back to diagonal loading and a pseudo-inverse, which biases and "
            "broadens the spectrum) and the MUSIC eigenvectors are dominated by "
            "estimation noise rather than by the true subspaces.",
            "Capon, MUSIC, ESPRIT")
    elif N < 2 * M:
        add(WARN,
            "N = %d snapshots is barely above M = %d" % (N, M),
            "Covariance estimation error scales roughly as sqrt(M/N). Below about "
            "N = 2M the inverse used by Capon is poorly conditioned and the MUSIC "
            "subspace split becomes unstable, which shows up as peak splitting and "
            "a raised noise floor.",
            "Capon, MUSIC")

    # ---------------------------------------------------------------
    # 3. Spatial aliasing / grating lobes (linear geometries).
    # ---------------------------------------------------------------
    if geo.type == "ULA":
        d = float(cfg.spacing)
        if d > 0.5 + 1e-9:
            th_g = np.rad2deg(np.arcsin(min(1.0, 1.0 / d - 1.0)))
            add(CRIT,
                "Spacing %.2f wavelengths exceeds the half-wavelength sampling limit" % d,
                "The array samples the wavefront in space; d > lambda/2 is spatial "
                "undersampling. Phase differences of 2*pi*d*sin(theta) wrap, so "
                "directions separated by about %.1f deg become indistinguishable and "
                "grating lobes appear in the spectrum. Every estimator inherits this "
                "- it is a property of the manifold, not of the algorithm - and the "
                "ESPRIT rotation phase wraps in exactly the same way."
                % max(1.0, 2 * th_g),
                "all")
        elif d < 0.25:
            add(INFO,
                "Spacing %.2f wavelengths gives a small aperture" % d,
                "Angular resolution is set by aperture, not element count. A tightly "
                "packed array has a wide beam, and mutual coupling (not modelled "
                "here) would degrade it further in hardware.",
                "all")

    # ---------------------------------------------------------------
    # 4. Source separation vs beamwidth (the Bartlett resolution wall).
    # ---------------------------------------------------------------
    if K > 1:
        sep = float(np.min(np.diff(doa)))
        aperture = geo.aperture
        if aperture > 0:
            bw = np.rad2deg(0.886 / aperture)          # 3 dB beamwidth, degrees
            if sep < bw:
                add(WARN,
                    "Source separation %.1f deg is inside the %.1f deg beamwidth"
                    % (sep, bw),
                    "Conventional beamforming cannot resolve sources closer than "
                    "about one beamwidth (0.886*lambda/aperture = %.1f deg here): the "
                    "two lobes merge into one. Capon and MUSIC are super-resolution "
                    "methods and can still separate them, but only if SNR and "
                    "snapshot count are high enough - below their threshold they "
                    "collapse to a single peak too. Watch the Bartlett curve show one "
                    "lobe while MUSIC shows two." % bw,
                    "Bartlett, CDF")

    # ---------------------------------------------------------------
    # 5. Threshold effect: low SNR combined with few snapshots.
    # ---------------------------------------------------------------
    if snr < 0 and N < 200:
        add(WARN,
            "SNR %.0f dB with only %d snapshots" % (snr, N),
            "Subspace methods have a threshold: below a critical SNR*N product the "
            "estimated signal and noise eigenvectors swap, MUSIC picks a noise peak "
            "and the RMSE jumps by orders of magnitude instead of degrading "
            "smoothly. This is why the RMSE-vs-SNR curve has a knee rather than a "
            "straight slope. Bartlett degrades gracefully and often wins below the "
            "knee.",
            "MUSIC, ESPRIT, Capon")

    # ---------------------------------------------------------------
    # 6. Endfire degradation.
    # ---------------------------------------------------------------
    if np.any(np.abs(doa) > 70) and geo.type != "UCA":
        add(INFO,
            "Source close to endfire (|DoA| > 70 deg)",
            "A linear array measures sin(theta), and d(sin)/d(theta) = cos(theta) "
            "goes to zero at endfire. Near +-90 deg a large bearing change produces "
            "almost no phase change, so the effective aperture collapses, lobes "
            "broaden and the variance of every estimator grows as 1/cos^2(theta). A "
            "circular array does not have this blind cone.",
            "all")

    # ---------------------------------------------------------------
    # 7. Geometry-specific estimator validity.
    # ---------------------------------------------------------------
    if "esprit" in sel and geo.type == "NLA":
        i1, _, _ = esprit_subarrays(geo.pos)
        if i1 is None or i1.size < K + 1:
            add(CRIT,
                "ESPRIT is not applicable to this non-uniform array",
                "ESPRIT replaces the spectral search with an eigenvalue problem, but "
                "that requires two identical subarrays displaced by a fixed vector. "
                "The element positions entered here contain no shift-invariant subset "
                "large enough, so the relation A2 = A1 PHI does not hold and any "
                "result would be meaningless. The checkbox is disabled for this "
                "geometry.",
                "ESPRIT")
        else:
            add(INFO,
                "ESPRIT is using a uniform subset of the sparse array",
                "This non-uniform layout happens to contain a shift-invariant element "
                "subset, so ESPRIT runs on that subset only. It therefore uses less "
                "than the full aperture and will not match the accuracy of the "
                "grid-search methods, which exploit every element.",
                "ESPRIT")

    if "esprit" in sel and geo.type == "UCA":
        add(INFO,
            "ESPRIT on a UCA runs in beamspace, not element space",
            "A circle has no two elements sharing a common displacement, so "
            "rotational invariance is manufactured with the Davies phase-mode "
            "transform, which maps the UCA onto a virtual ULA of 2H+1 modes. Two "
            "costs follow: the transform colours the white noise (so ESPRIT loses "
            "its optimality), and the number of usable modes is limited by "
            "J_h(2*pi*R) - with R = %.2f wavelengths only the low modes survive, "
            "capping the number of resolvable sources." % geo.radius,
            "ESPRIT")

    # ---------------------------------------------------------------
    # 7b. Circular-array Bessel nulls: the phase-mode gains are
    #     J_h(2*pi*R), so certain radii blind specific modes.
    # ---------------------------------------------------------------
    if geo.type == "UCA":
        u = uca_modes(geo.radius, M)

        if u.arc > 0.5 + 1e-9:
            add(WARN,
                "UCA arc spacing %.2f wavelengths exceeds half a wavelength" % u.arc,
                "A circular array samples the wavefront along its circumference. With "
                "M = %d elements on a circle of circumference 2*pi*R = %.2f "
                "wavelengths the arc spacing is %.2f wavelengths, which is spatial "
                "undersampling just as d > lambda/2 is for a line. Its signature here "
                "is phase-mode aliasing: the M-point spatial DFT cannot separate mode "
                "h from mode h-%d, so the mode-1 channel is contaminated at %.0f%% of "
                "its own amplitude. Add elements or shrink the radius."
                % (M, u.z, u.arc, M, 100 * u.ratio1),
                "Watson-Watt, ESPRIT (beamspace)")

        if not u.mode1_ok and "ww" in sel:
            add(CRIT,
                "Circular geometry has no usable mode-1 channel",
                "Watson-Watt needs the mode-1 (loop) channel of the circle, whose gain "
                "is J1(2*pi*R) = %.4f here. %s Because the loop channel would then "
                "contain mostly aliasing residue and noise, the bearing would be "
                "confidently wrong rather than noisy, so the estimator refuses to "
                "report. Note that MUSIC, Capon and Bartlett are unaffected: they use "
                "the full element-space manifold and never form modes."
                % (u.J1, u.why),
                "Watson-Watt")

        if "esprit" in sel and u.n_modes < K + 1:
            add(CRIT,
                "Only %d usable beamspace mode(s) for ESPRIT" % u.n_modes,
                "Beamspace ESPRIT needs at least K+1 = %d virtual elements. The usable "
                "phase modes are squeezed from two sides: aliasing caps them at "
                "(M-1)/2 = %d, and the Bessel gains J_h(%.2f) plus their alias terms "
                "rule out everything above mode %d. %s A larger radius buys more modes "
                "(more aperture in wavelengths) and more elements raise the aliasing "
                "cap - both are needed together."
                % (K + 1, u.Hmax, u.z, u.H, u.why),
                "ESPRIT")

    # ---------------------------------------------------------------
    # 8. Single-wavefront estimators facing multiple emitters.
    # ---------------------------------------------------------------
    if K > 1 and "cdf" in sel:
        add(WARN,
            "CDF is a single-wavefront estimator facing multiple emitters",
            "Correlative DF compares the measured phase pattern against a table of "
            "reference patterns, each generated by ONE source. With several "
            "co-channel emitters the measured pattern is a superposition that matches "
            "no table entry; the best correlation lands between the sources, so the "
            "estimate is biased toward the power centroid rather than being noisy. "
            "Expect a systematic error, not a random one.",
            "CDF")

    if K > 1 and "ww" in sel:
        add(WARN,
            "Watson-Watt reports one bearing regardless of source count",
            "The Watson-Watt arctangent maps a single pair of sum/difference channel "
            "voltages to a single bearing. Two simultaneous signals add vectorially in "
            "those channels, so the arctangent returns their instantaneous vector sum "
            "- a bearing that lies between the emitters and wanders with their "
            "relative phase. This is the classical wandering bearing seen on real "
            "Watson-Watt sets in a multipath environment.",
            "Watson-Watt")

    # ---------------------------------------------------------------
    # 9. Sense antenna / Watson-Watt specifics.
    # ---------------------------------------------------------------
    if "ww" in sel:
        if not cfg.sense and geo.type == "UCA":
            add(WARN,
                "Watson-Watt without a sense antenna: 180 deg ambiguity",
                "The mode-1 loop channel of a circular array has a figure-of-eight "
                "pattern, which is identical for theta and theta+180 deg. The sense "
                "antenna supplies an omnidirectional phase reference; adding it to the "
                "loop output forms a cardioid whose single null resolves the "
                "front/back pair. Without it the app falls back to the mode-0 channel "
                "as a synthetic sense, and if that is Bessel-nulled the bearing is "
                "genuinely ambiguous.",
                "Watson-Watt")
        if geo.type != "UCA":
            add(INFO,
                "Watson-Watt on a linear array is an adaptation, not the classical set",
                "True Watson-Watt uses two orthogonal loops. Here the X/Y channels are "
                "synthesised as the sum and difference of a symmetric element pair, so "
                "the arctangent returns 2*pi*b*sin(theta) rather than theta directly. A "
                "line of omni elements also cannot separate theta from 180-theta at "
                "all: that ambiguity is geometric, and an omni sense antenna at the "
                "centroid carries no information to break it. Here the sense channel is "
                "used instead to unwrap the pair phase, extending the unambiguous "
                "baseline from 0.5 to about 1 wavelength.",
                "Watson-Watt")

    # ---------------------------------------------------------------
    # 10. Runtime refusals reported by the estimators themselves.
    # ---------------------------------------------------------------
    for r in (results or []):
        if not r.valid and r.note:
            add(CRIT, "%s could not run on this scenario" % r.short, r.note, r.short)
        elif r.ambiguous and r.note:
            add(WARN, "%s returned an ambiguous bearing" % r.short, r.note, r.short)

    if not out:
        add(INFO, "No limiting conditions detected",
            "Element count, snapshot support, spacing, source separation and SNR are "
            "all inside the regime where every selected estimator is expected to work. "
            "Differences between the curves now reflect estimator efficiency rather "
            "than a broken assumption.",
            "all")

    return out
