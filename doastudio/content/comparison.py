"""Background reference for the Comparison tab.

This module is pure content: no computation, no simulation, no dependency on
the engine.  It describes what each estimator actually does, what physics
bounds all of them, and how to choose between them.  The renderer in
``ui/tab_comparison.py`` turns these blocks into a formatted document.

Block grammar
-------------
``("h2", text)``            sub-heading
``("p", text)``             paragraph, word-wrapped by the renderer
``("bullets", [text, ...])``
``("numbers", [text, ...])``
``("table", [headers], [[cell, ...], ...])``
``("formula", text)``       monospace block, newlines preserved
``("kv", [(key, value), ...])``
``("note", text)``          call-out, indented and tinted
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Section:
    id: str
    title: str
    blocks: List[tuple]


# =====================================================================
# 1. The physics
# =====================================================================
_OVERVIEW = Section("overview", "1.  What direction finding actually measures", [
    ("p",
     "Direction of arrival (DoA) estimation answers one question: from which "
     "bearing did this radio signal come? An antenna array answers it not by "
     "pointing a dish, but by measuring the small time delays with which one "
     "wavefront reaches spatially separated elements. At a single carrier "
     "frequency those delays appear as phase differences, and the whole "
     "subject is the business of turning a vector of measured phases back "
     "into an angle."),

    ("h2", "The plane-wave model"),
    ("p",
     "A source far enough away that its wavefront is effectively flat across "
     "the array (the far field, beyond roughly 2D^2/lambda for an aperture D) "
     "arrives at element m with an extra path length that depends only on the "
     "projection of the element position onto the arrival direction. With "
     "positions measured in wavelengths, the response of the array to a unit "
     "amplitude source at bearing theta is the steering vector:"),
    ("formula",
     "a_m(theta) = exp( j*2*pi * ( x_m*sin(theta) + y_m*cos(theta) ) )\n"
     "\n"
     "For a uniform linear array on the x-axis (y_m = 0, x_m = m*d):\n"
     "\n"
     "a_m(theta) = exp( j*2*pi*d*m*sin(theta) )"),
    ("p",
     "Two consequences follow immediately, and they shape every algorithm in "
     "this app. First, a linear array measures sin(theta), not theta - so its "
     "angular sensitivity is proportional to cos(theta) and collapses at "
     "endfire. Second, phase is only defined modulo 2*pi, so once the "
     "inter-element spacing d exceeds half a wavelength, distinct bearings "
     "produce identical phase vectors. That is spatial aliasing, and it is "
     "the direct analogue of the Nyquist limit in time sampling."),

    ("h2", "The data model"),
    ("p",
     "With K narrowband sources, M elements and N snapshots, the received "
     "data is a superposition of steered sources plus noise:"),
    ("formula",
     "X  =  A(theta) S  +  Noise            X is M x N\n"
     "A  =  [ a(theta_1)  ...  a(theta_K) ]  the M x K array manifold\n"
     "S  =  source waveforms                 K x N\n"
     "\n"
     "R  =  E{ x x^H }  =  A Rs A^H  +  sigma^2 I      the covariance"),
    ("p",
     "Almost every estimator works from the sample covariance R rather than "
     "from the raw snapshots, because R compresses N samples into an M x M "
     "matrix that contains everything a second-order method can use. The "
     "structure of that matrix - a rank-K signal term plus a scaled identity "
     "- is exactly what the subspace methods exploit."),

    ("h2", "Angle convention used throughout this app"),
    ("bullets", [
        "0 degrees is broadside: perpendicular to a linear array, along +y.",
        "+90 degrees is endfire toward +x; -90 degrees is endfire toward -x.",
        "Element positions are always in wavelengths, so the 2*pi/lambda "
        "factor is 1 and every geometry number is frequency-independent.",
        "The scan sector is -90 to +90 degrees. A linear array of "
        "omnidirectional elements genuinely cannot distinguish theta from "
        "180-theta, so the rear hemisphere is not displayed.",
    ]),
])


# =====================================================================
# 2. Taxonomy
# =====================================================================
_FAMILIES = Section("families", "2.  Four families of estimator", [
    ("p",
     "The six algorithms compared here are not six variations on one idea. "
     "They belong to four genuinely different families, with different "
     "assumptions, different failure modes and different hardware histories. "
     "Knowing which family a method belongs to predicts its behaviour better "
     "than any single performance number."),

    ("table",
     ["Family", "Members", "Core idea", "Breaks when"],
     [
         ["Beamforming",
          "Bartlett",
          "Steer, measure power, repeat",
          "Sources closer than one beamwidth"],
         ["Adaptive beamforming",
          "Capon / MVDR",
          "Steer while nulling everything else",
          "R is singular or sources are correlated"],
         ["Subspace",
          "MUSIC, ESPRIT",
          "Split R into signal and noise subspaces",
          "K unknown, K >= M, coherent sources, low SNR*N"],
         ["Pattern matching",
          "CDF (correlative)",
          "Match measured phases to a calibration table",
          "More than one wavefront at a time"],
         ["Instrument / amplitude",
          "Watson-Watt",
          "Arctangent of two orthogonal channel voltages",
          "More than one wavefront; loop channel nulled"],
     ]),

    ("h2", "Why the split matters"),
    ("bullets", [
        "Beamforming methods are non-parametric. They make no assumption "
        "about how many sources are present, never diverge, and degrade "
        "gracefully. Their resolution is fixed by the aperture and cannot be "
        "improved by raising the SNR.",

        "Subspace methods are parametric. They assume exactly K sources and "
        "exploit the algebraic structure that assumption creates. In return "
        "they resolve far inside the beamwidth - but if K is wrong, or the "
        "SNR falls below their threshold, they fail abruptly rather than "
        "gradually.",

        "Pattern-matching methods make no model assumption at all beyond "
        "'one wavefront', which is why they tolerate arbitrary and even "
        "poorly calibrated geometries. That robustness is the reason most "
        "fielded DF receivers use them.",

        "Instrument methods come from an era before digital arrays. They are "
        "extremely cheap, work on a handful of channels, and are still in "
        "service - but they report a single bearing whatever the scene "
        "contains.",
    ]),
])


# =====================================================================
# 3. Comparison matrix
# =====================================================================
_MATRIX = Section("matrix", "3.  Side-by-side comparison", [
    ("p",
     "The table below summarises the properties that actually decide which "
     "estimator to reach for. 'DoF' is degrees of freedom: the maximum number "
     "of sources the method can in principle separate with M elements."),

    ("table",
     ["Property", "Bartlett", "Capon", "CDF", "MUSIC", "ESPRIT", "Watson-Watt"],
     [
         ["Family", "Beamform", "Adaptive", "Match", "Subspace", "Subspace",
          "Instrument"],
         ["Needs K in advance", "no", "no", "no", "yes", "yes", "no"],
         ["Sources resolvable", "aperture", "M-1", "1", "M-1", "M-1", "1"],
         ["Super-resolution", "no", "yes", "no", "yes", "yes", "no"],
         ["Grid search", "yes", "yes", "yes", "yes", "no", "no"],
         ["Needs N > M", "no", "yes", "no", "yes", "yes", "no"],
         ["Coherent sources", "works", "fails", "n/a", "fails", "fails", "n/a"],
         ["Arbitrary geometry", "yes", "yes", "yes", "yes", "no", "adapted"],
         ["Sidelobe / bias risk", "high", "low", "medium", "low", "none",
          "n/a"],
         ["Cost per estimate", "low", "medium", "low", "high", "medium",
          "trivial"],
         ["Fails", "gracefully", "gracefully", "gracefully", "abruptly",
          "abruptly", "silently"],
     ]),

    ("h2", "Computational cost"),
    ("p",
     "M is the element count, N the snapshot count, G the number of scan-grid "
     "points and K the source count. The covariance itself costs O(M^2 N) and "
     "is shared by every method that uses it."),
    ("table",
     ["Algorithm", "Dominant cost", "Grid dependent", "Comment"],
     [
         ["Bartlett", "O(M^2 G)", "yes",
          "One matrix product; trivially parallel."],
         ["Capon", "O(M^3 + M^2 G)", "yes",
          "One inverse, then the same product as Bartlett."],
         ["CDF", "O(M^2 N + M G)", "yes",
          "Table lookup; the table can be precomputed once."],
         ["MUSIC", "O(M^3 + M(M-K) G)", "yes",
          "Eigendecomposition plus a projection at every grid point."],
         ["ESPRIT", "O(M^3 + K^3)", "no",
          "Search-free: cost is independent of the required precision."],
         ["Watson-Watt", "O(M N)", "no",
          "Two correlations and one arctangent."],
     ]),
    ("note",
     "The grid-dependent methods pay for angular precision twice: once in "
     "compute time, and once in an RMSE floor set by the grid step. This app "
     "interpolates a parabola through each spectral peak precisely so that "
     "floor does not dominate the high-SNR comparison."),
])


# =====================================================================
# 4-9. Per-algorithm detail
# =====================================================================
def _algorithm_section(sid, title, blocks) -> Section:
    return Section(sid, title, blocks)


_BARTLETT = _algorithm_section("bartlett", "4.  Bartlett - conventional beamforming", [
    ("kv", [
        ("Also known as", "Delay-and-sum, Fourier method, periodogram beamformer"),
        ("Origin", "M. S. Bartlett, 1948 - the spatial analogue of the "
                   "smoothed periodogram"),
        ("Class", "Non-parametric beamforming"),
    ]),

    ("h2", "Principle"),
    ("p",
     "Point the array at a candidate bearing by applying the conjugate of the "
     "steering vector to each element, sum the elements, and measure the "
     "output power. Sweep the candidate bearing across the scan sector and the "
     "resulting curve peaks where energy is arriving. This is exactly what a "
     "mechanically rotated antenna does, performed numerically and "
     "simultaneously in every direction."),
    ("formula",
     "w(theta) = a(theta) / ||a(theta)||\n"
     "\n"
     "P(theta) = w^H R w = a^H(theta) R a(theta) / ( a^H(theta) a(theta) )"),

    ("h2", "Resolution: the hard wall"),
    ("p",
     "The Bartlett spectrum is the true angular spectrum convolved with the "
     "array's beam pattern. Two sources closer than roughly one 3 dB beamwidth "
     "merge into a single lobe, and no amount of extra SNR or extra snapshots "
     "will separate them - the information has been smeared away by the "
     "aperture itself, not buried in noise."),
    ("formula",
     "3 dB beamwidth  ~  0.886 * lambda / D   radians\n"
     "\n"
     "where D is the physical aperture (the array's total extent).\n"
     "Element COUNT does not set resolution; aperture does."),

    ("h2", "Strengths"),
    ("bullets", [
        "Never diverges and never needs a matrix inverse. It works with a "
        "single snapshot, with a rank-deficient covariance, and with fully "
        "coherent multipath.",
        "Requires no knowledge of the source count.",
        "Below the threshold SNR of the subspace methods it is frequently the "
        "most accurate estimator available, because it has no threshold.",
        "Lowest sidelobe risk to reason about: the pattern is known in "
        "advance and can be shaped by amplitude tapering.",
    ]),
    ("h2", "Weaknesses"),
    ("bullets", [
        "Resolution is beamwidth-limited, full stop.",
        "Sidelobes of a strong emitter can mask a weak one nearby - the "
        "classic dynamic-range problem of an untapered array.",
        "The peak of a two-source lobe is biased toward the stronger source.",
    ]),
    ("note",
     "Use Bartlett as the reference curve. If a super-resolution method is "
     "not clearly beating it, that method is operating below its threshold "
     "and its extra complexity is buying nothing."),
])


_CAPON = _algorithm_section("capon", "5.  Capon / MVDR - adaptive beamforming", [
    ("kv", [
        ("Also known as", "Minimum Variance Distortionless Response, "
                          "maximum-likelihood method"),
        ("Origin", "J. Capon, 1969, 'High-resolution frequency-wavenumber "
                   "spectrum analysis'"),
        ("Class", "Adaptive (data-dependent) beamforming"),
    ]),

    ("h2", "Principle"),
    ("p",
     "Instead of fixing the beam shape in advance, let the data choose it. "
     "Minimise the total output power subject to the constraint that a signal "
     "arriving from the look direction passes with unit gain. Everything the "
     "beamformer is free to suppress - interferers, sidelobe leakage - it "
     "will suppress, so the output power that remains is a much sharper "
     "measure of energy genuinely arriving from that bearing."),
    ("formula",
     "minimise  w^H R w    subject to   w^H a(theta) = 1\n"
     "\n"
     "w(theta) = R^-1 a(theta) / ( a^H(theta) R^-1 a(theta) )\n"
     "\n"
     "P(theta) = 1 / ( a^H(theta) R^-1 a(theta) )"),
    ("p",
     "The inverse covariance is doing the work: it places nulls on every "
     "strong arrival that is not the look direction. That is why Capon "
     "resolves sources comfortably inside the Bartlett beamwidth while still "
     "producing a genuine, interpretable power spectrum."),

    ("h2", "The price of the inverse"),
    ("bullets", [
        "R must be invertible, which needs at least N > M snapshots and in "
        "practice more like N > 2M for a well-conditioned result.",
        "Correlated or coherent sources break it badly. If two arrivals are "
        "phase-locked - the direct path and its specular reflection, for "
        "instance - the beamformer can cancel the signal against itself, and "
        "the peak at the true bearing disappears.",
        "A steering-vector mismatch (calibration error, mutual coupling) is "
        "punished harshly: the constraint no longer protects the real signal, "
        "so the adaptive weights null it out.",
    ]),
    ("h2", "Diagonal loading"),
    ("p",
     "The standard fix for all three problems is to add a small multiple of "
     "the identity to R before inverting. This trades a little resolution for "
     "a great deal of robustness by bounding the white-noise gain of the "
     "weight vector, and it is what makes MVDR usable on real hardware. This "
     "app loads at about -30 dB of the mean eigenvalue and falls back to a "
     "pseudo-inverse if the matrix is still rank deficient."),
    ("note",
     "Capon sits between Bartlett and MUSIC in every respect: better "
     "resolution than Bartlett, more robust than MUSIC, no need to know K, "
     "but a real dependence on snapshot support."),
])


_CDF = _algorithm_section("cdf", "6.  Correlative interferometry (CDF)", [
    ("kv", [
        ("Also known as", "Correlative direction finding, correlation "
                          "interferometry, phase-pattern matching"),
        ("Origin", "Operational DF receiver practice from the 1980s onward; "
                   "the dominant technique in fielded monitoring equipment"),
        ("Class", "Pattern matching against a calibration table"),
    ]),

    ("h2", "Principle"),
    ("p",
     "Do not model the array at all. Measure, on a test range, the complex "
     "voltage pattern the real antenna system produces for a source at every "
     "bearing, and store it as a correlation table. In operation, measure the "
     "actual inter-element pattern and report the bearing whose stored "
     "reference correlates best with it."),
    ("formula",
     "rho(theta) = | a^H(theta) v |  /  ( ||a(theta)|| * ||v|| )\n"
     "\n"
     "v  =  the measured wavefront (this app uses the dominant\n"
     "      eigenvector of R, the maximum-likelihood single-\n"
     "      wavefront estimate over all N snapshots)\n"
     "\n"
     "The magnitude discards unknown common phase and gain,\n"
     "so rho lies in [0, 1] and 1 is a perfect match."),

    ("h2", "Why fielded receivers prefer it"),
    ("bullets", [
        "It never needs the array to match a mathematical model. Mutual "
        "coupling, a mast, a vehicle roof, radome distortion - all of it is "
        "baked into the measured table and therefore corrected for free.",
        "It tolerates irregular geometries and mixed element types.",
        "The correlation value itself is a natural quality metric: a low peak "
        "correlation flags a multipath or interference situation, and "
        "operators use it exactly that way.",
        "Cost is low and the table can be precomputed once.",
    ]),
    ("h2", "The single-wavefront assumption"),
    ("p",
     "Every entry in the reference table was generated by ONE source. When "
     "two co-channel emitters are present, the measured pattern is their "
     "superposition and matches no table entry at all. The best-correlating "
     "reference then lands somewhere between the two sources, biased toward "
     "the stronger one. The result is a systematic error, not a noisy one - "
     "averaging more snapshots will not remove it."),
    ("note",
     "This is the defining difference between CDF and the subspace methods. "
     "CDF is superb for one emitter on arbitrary hardware, and structurally "
     "wrong for two. Operational systems handle this by detecting the low "
     "correlation peak and flagging the bearing as unreliable."),
])


_MUSIC = _algorithm_section("music", "7.  MUSIC - MUltiple SIgnal Classification", [
    ("kv", [
        ("Origin", "R. O. Schmidt, 1979; published in IEEE Trans. Antennas "
                   "and Propagation, 1986"),
        ("Class", "Noise-subspace method"),
        ("Significance", "The paper that made super-resolution DF practical; "
                         "still the benchmark against which new methods are "
                         "measured"),
    ]),

    ("h2", "Principle"),
    ("p",
     "The covariance of K sources in white noise has a very particular "
     "structure: its K largest eigenvalues carry signal plus noise, and its "
     "remaining M-K eigenvalues are all equal to the noise power. The "
     "eigenvectors split correspondingly into a K-dimensional signal subspace "
     "and an (M-K)-dimensional noise subspace, and those two subspaces are "
     "orthogonal."),
    ("p",
     "Crucially, the signal subspace is spanned by the true steering vectors. "
     "So every true steering vector is orthogonal to the entire noise "
     "subspace, while every other steering vector is not. Projecting a "
     "candidate steering vector onto the noise subspace and taking the "
     "reciprocal therefore produces a function that blows up at the true "
     "bearings:"),
    ("formula",
     "R = U L U^H,   eigenvalues descending\n"
     "\n"
     "Es = U[:, :K]      signal subspace   (K columns)\n"
     "En = U[:, K:]      noise subspace    (M-K columns)\n"
     "\n"
     "P(theta) = 1 / || En^H a(theta) ||^2"),

    ("h2", "Why the peaks are sharp"),
    ("p",
     "The MUSIC 'spectrum' is not a power spectrum. Nothing about it is "
     "calibrated in watts; the peak height carries no information about "
     "source strength. It is a nulling function whose denominator goes to "
     "zero at the true bearings, which is why its peaks can be arbitrarily "
     "narrow and why its resolution is not bounded by the beamwidth. Two "
     "sources a tenth of a beamwidth apart are separable in principle - "
     "given enough SNR and enough snapshots."),

    ("h2", "The four ways MUSIC fails"),
    ("numbers", [
        "K is wrong. Overestimate it and real sources get absorbed into a "
        "signal subspace that is too large, producing spurious peaks; "
        "underestimate it and a real source is left in the noise subspace and "
        "vanishes entirely. K must be estimated separately, typically with an "
        "information criterion such as AIC or MDL applied to the eigenvalues.",

        "K >= M. The noise subspace is empty and there is nothing to project "
        "onto. In practice the useful limit is well below M-1, because the "
        "projection is estimated from ever fewer eigenvectors.",

        "Coherent sources. Fully correlated arrivals - the direct path and a "
        "specular reflection - make the source covariance rank deficient, so "
        "the signal subspace collapses to fewer than K dimensions and sources "
        "are simply missed. Spatial smoothing on a ULA restores the rank at "
        "the cost of aperture.",

        "The threshold effect. Below a critical SNR*N product, an estimated "
        "noise eigenvector and an estimated signal eigenvector swap places. "
        "MUSIC then locks onto a noise peak and the error jumps by orders of "
        "magnitude. This is why an RMSE-vs-SNR curve has a knee rather than a "
        "straight slope, and it is a property of the estimator, not of the "
        "implementation.",
    ]),
    ("note",
     "Above its threshold MUSIC is asymptotically efficient - its variance "
     "approaches the Cramer-Rao bound. Below it, plain Bartlett is often more "
     "accurate. Almost every practical argument about MUSIC is really an "
     "argument about where that threshold sits."),
])


_ESPRIT = _algorithm_section("esprit", "8.  ESPRIT - rotational invariance", [
    ("kv", [
        ("Full name", "Estimation of Signal Parameters via Rotational "
                      "Invariance Techniques"),
        ("Origin", "R. Roy and T. Kailath, IEEE Trans. ASSP, 1989"),
        ("Class", "Signal-subspace method, search-free"),
    ]),

    ("h2", "Principle"),
    ("p",
     "MUSIC finds bearings by searching. ESPRIT observes that if the array "
     "contains two identical subarrays displaced by a fixed vector, the "
     "bearings can be read directly off an eigenvalue problem - no search at "
     "all. The displacement imposes a pure phase shift on each source, so the "
     "two subarrays see manifolds related by a diagonal matrix whose entries "
     "encode the DoAs:"),
    ("formula",
     "A2 = A1 * PHI,     PHI = diag( exp( j*2*pi*<delta, u(theta_k)> ) )\n"
     "\n"
     "The same relation must hold between the two subarrays' signal\n"
     "subspaces, up to an unknown invertible T:\n"
     "\n"
     "Es2 = Es1 * PSI,   PSI = T^-1 PHI T\n"
     "\n"
     "so eig(PSI) recovers the phases, and hence the bearings,\n"
     "in closed form."),

    ("h2", "Total least squares"),
    ("p",
     "Both subarray subspaces are estimated from noisy data, so solving "
     "Es1 PSI = Es2 by ordinary least squares is the wrong error model - it "
     "assumes the left-hand side is exact. Total least squares, obtained from "
     "the null space of the stacked matrix [Es1 Es2], distributes the error "
     "symmetrically and is markedly better conditioned at low SNR. That is "
     "the variant implemented here."),

    ("h2", "The geometry requirement is absolute"),
    ("p",
     "ESPRIT's speed comes entirely from the shift-invariance assumption, and "
     "where that assumption does not hold the method does not degrade - it "
     "returns confident nonsense. A uniform linear array satisfies it by "
     "construction. An arbitrary sparse array satisfies it only if it happens "
     "to contain a shift-invariant subset, and then only that subset's "
     "aperture is used."),
    ("p",
     "A circular array satisfies it nowhere: every chord of a circle points "
     "in a different direction, so no two element pairs share a displacement "
     "vector. Invariance must instead be manufactured in beamspace using the "
     "Davies phase-mode transform, which maps the circle onto a virtual "
     "linear array of phase modes:"),
    ("formula",
     "Jacobi-Anger expansion of the circular manifold:\n"
     "\n"
     "a_m(psi) = sum_h  j^h  J_h(2*pi*R)  exp( j*h*(psi - phi_m) )\n"
     "\n"
     "Transform:  F = diag( 1 / ( j^h J_h(2*pi*R) ) ) * (1/M) exp( j h phi_m )\n"
     "\n"
     "Virtual element h then responds as exp( j*h*psi ) - a ULA in h."),
    ("p",
     "Two costs come with the transform and both are real. It colours the "
     "white noise, so ESPRIT loses its statistical optimality; and only modes "
     "with a non-vanishing Bessel gain J_h(2*pi*R) survive, so a small circle "
     "supports too few virtual elements to resolve many sources."),

    ("h2", "Trade-offs versus MUSIC"),
    ("bullets", [
        "No grid, so no grid-induced RMSE floor and no cost-versus-precision "
        "trade. Estimates are continuous-valued by construction.",
        "Much cheaper when high angular precision is required.",
        "Slightly worse variance than MUSIC on the same array, because each "
        "subarray uses less than the full aperture.",
        "Shares every one of MUSIC's assumptions: K must be known, K < M, "
        "coherent sources still break it, and it has the same threshold.",
        "Cannot run on an arbitrary geometry, which MUSIC can.",
    ]),
])


_WW = _algorithm_section("ww", "9.  Watson-Watt - the classical instrument", [
    ("kv", [
        ("Origin", "Robert Watson-Watt, 1926, building on the Adcock antenna "
                   "of 1919"),
        ("Class", "Amplitude-comparison instrument, single wavefront"),
        ("Still used in", "HF and VHF monitoring, maritime and emergency "
                          "beacon DF, compact mobile installations"),
    ]),

    ("h2", "The original instrument"),
    ("p",
     "Two physically orthogonal loop antennas produce figure-of-eight voltage "
     "patterns. A signal at bearing theta induces voltages proportional to "
     "cos(theta) in the north-south loop and sin(theta) in the east-west "
     "loop, so the bearing falls straight out of an arctangent - historically "
     "displayed as a line on a cathode-ray tube, with no computation at all."),
    ("formula",
     "X = E cos(theta)      N-S loop\n"
     "Y = E sin(theta)      E-W loop\n"
     "S = E                 omnidirectional sense antenna\n"
     "\n"
     "theta = atan2( Y, X )        ambiguous by 180 degrees\n"
     "\n"
     "Adding the sense channel to a loop forms a cardioid with a\n"
     "single null, which resolves front from back."),
    ("p",
     "Because a figure-of-eight pattern is symmetric, the loop pair alone "
     "cannot tell a bearing from its reciprocal. The sense antenna exists "
     "purely to break that ambiguity - the 'cardioid conversion' - and it is "
     "the reason a real Watson-Watt set has three channels rather than two."),

    ("h2", "Adapting it to an array of omni elements"),
    ("p",
     "This app has no loops, only omnidirectional elements, so the X/Y "
     "channel pair must be synthesised. The construction differs by geometry, "
     "and the two cases are genuinely different in what they can achieve:"),
    ("bullets", [
        "Linear array. The cross-correlation of a symmetric element pair at "
        "+h and -h has phase 2*pi*b*sin(theta) for baseline b, so its real "
        "and imaginary parts play the roles of the sum and difference "
        "channels. Here the sense antenna cannot resolve front from back - "
        "that ambiguity is geometric, and an omni element at the centroid "
        "carries no directional information. Instead the sense channel "
        "supplies half-baseline phases that unwrap the pair phase, extending "
        "the unambiguous baseline from 0.5 to about 1 wavelength.",

        "Circular array. Phase mode 1 has a rotating figure-of-eight pattern "
        "- a genuine electronic loop - and mode 0 is omnidirectional, a "
        "genuine sense channel. Correlating mode 1 against mode 0 and "
        "dividing out the known j*J_1(2*pi*R) factor gives a complex number "
        "whose real and imaginary parts ARE the Watson-Watt X and Y. This is "
        "the true cardioid conversion and it resolves the full 360 degrees.",
    ]),

    ("h2", "The wandering bearing"),
    ("p",
     "Watson-Watt maps one pair of channel voltages to one bearing. Two "
     "simultaneous signals add vectorially in those channels, so the "
     "arctangent returns the instantaneous vector sum: a bearing that lies "
     "between the emitters and wanders as their relative phase drifts. Real "
     "Watson-Watt sets show exactly this in multipath, and operators learn to "
     "read the wander itself as a multipath indicator."),
    ("note",
     "The practical consequence is that a Watson-Watt error-versus-SNR curve "
     "for two sources is flat. The error is not noise, it is the estimator "
     "answering a different question from the one being asked - and no amount "
     "of signal-to-noise ratio will change the answer."),

    ("h2", "Bessel nulls: a circular-array blind spot"),
    ("p",
     "On a circle the mode-1 gain is J_1(2*pi*R), and J_1 first vanishes at "
     "2*pi*R = 3.8317, i.e. R = 0.6098 wavelengths. At that radius the loop "
     "channel carries no signal at all and the bearing would be confidently "
     "wrong rather than merely noisy. Any honest implementation must detect "
     "this and refuse, which is what this app does. The same Bessel weighting "
     "caps how many modes beamspace ESPRIT can use."),
])


# =====================================================================
# 10. Geometry
# =====================================================================
_GEOMETRY = Section("geometry", "10.  Array geometries", [
    ("p",
     "The geometry decides more about achievable performance than the choice "
     "of algorithm does. Aperture sets resolution, element count sets degrees "
     "of freedom, and the layout decides which algorithms are legal at all."),

    ("table",
     ["", "ULA", "UCA", "NLA (sparse)"],
     [
         ["Coverage", "180 deg, front/back ambiguous", "full 360 deg",
          "180 deg, front/back ambiguous"],
         ["Endfire blind cone", "yes", "no", "yes"],
         ["Beamwidth uniform in angle", "no, widens as 1/cos(theta)", "yes",
          "no"],
         ["Aperture per element", "low", "low", "high"],
         ["ESPRIT", "native", "beamspace only", "only if a shift-invariant "
                                                "subset exists"],
         ["Watson-Watt", "adapted, no front/back", "native, full 360",
          "adapted, often ambiguous"],
         ["Aliasing rule", "d <= lambda/2", "arc 2*pi*R/M <= lambda/2",
          "co-array must have no holes"],
         ["Typical use", "radar, sonar, textbooks", "HF/VHF monitoring, "
                                                    "fixed DF sites",
          "radio astronomy, compressed arrays"],
     ]),

    ("h2", "Uniform linear array"),
    ("p",
     "The reference geometry for the whole field. Its manifold is a Vandermonde "
     "vector in sin(theta), which is what makes ESPRIT, root-MUSIC and spatial "
     "smoothing possible. Its weaknesses are structural: it measures only "
     "sin(theta), so sensitivity dies at endfire; and a line of "
     "omnidirectional elements cannot distinguish theta from 180-theta."),

    ("h2", "Uniform circular array"),
    ("p",
     "The natural choice when full azimuth coverage is required, which is why "
     "fixed monitoring sites use it. Its beamwidth is essentially independent "
     "of bearing and it has no endfire blind cone. The cost is that its "
     "manifold is not Vandermonde, so the elegant linear-array algorithms need "
     "the phase-mode transform to apply - and that transform brings the Bessel "
     "weighting and mode aliasing with it. Note that a circle's aperture is "
     "only its diameter, so a UCA of M elements has considerably worse "
     "resolution than a ULA of the same M at half-wavelength spacing."),

    ("h2", "Sparse and non-uniform arrays"),
    ("p",
     "Resolution follows aperture, and aperture costs elements only if the "
     "array is filled. A minimum-redundancy or nested layout spreads few "
     "elements over a wide aperture while keeping the co-array - the set of "
     "all pairwise differences - free of holes, so the covariance still "
     "contains every spatial lag a filled array would have provided. Such "
     "arrays can resolve more sources than they have elements when combined "
     "with co-array processing."),
    ("bullets", [
        "Minimum-redundancy arrays (Moffet, 1968) minimise repeated lags for "
        "a given element count; optimal layouts are known only for small M "
        "and are tabulated rather than derived.",
        "Nested arrays (Pal and Vaidyanathan, 2010) give a closed-form "
        "construction with a guaranteed hole-free co-array.",
        "The price: ragged sidelobes, greater sensitivity to calibration "
        "error, and the loss of shift invariance that ESPRIT depends on.",
    ]),
])


# =====================================================================
# 11. Physics limits
# =====================================================================
_LIMITS = Section("limits", "11.  The limits every estimator shares", [
    ("p",
     "Some constraints belong to the algorithm and can be engineered around. "
     "Others belong to the physics of a sampled aperture and bind every method "
     "equally. Confusing the two is the most common source of unrealistic "
     "expectations in DF work."),

    ("h2", "Spatial aliasing - a property of the array"),
    ("p",
     "Phase is measured modulo 2*pi. If neighbouring elements are more than "
     "half a wavelength apart, two different bearings can produce identical "
     "phase vectors, and grating lobes appear. No algorithm can undo this, "
     "because the information is not present in the measurement. On a circle "
     "the same rule applies to the arc distance between neighbours."),

    ("h2", "Beamwidth - a property of the aperture"),
    ("p",
     "The classical resolution limit 0.886*lambda/D binds the non-parametric "
     "methods absolutely. Super-resolution methods can beat it, but only by "
     "spending assumptions (a known source count, uncorrelated sources) and "
     "SNR. The aperture limit reappears as a threshold: the closer two sources "
     "are relative to the beamwidth, the higher the SNR needed to separate "
     "them, and the relationship is steep."),

    ("h2", "Degrees of freedom"),
    ("p",
     "An M-element array spans an M-dimensional observation space. MUSIC needs "
     "at least one noise eigenvector, so K <= M-1 is a hard ceiling and K <= "
     "M-2 is a practical one. Sparse arrays evade this by processing the "
     "co-array instead of the physical array, which is the main reason anyone "
     "builds them."),

    ("h2", "Snapshot support"),
    ("p",
     "The sample covariance has rank at most N. With N < M it is singular, so "
     "the Capon inverse does not exist and the MUSIC eigenvectors are "
     "dominated by estimation error rather than by the true subspaces. "
     "Covariance error falls roughly as sqrt(M/N), so N > 2M is the usual "
     "working minimum and N > 10M is comfortable."),

    ("h2", "The threshold effect"),
    ("p",
     "Subspace estimators do not degrade smoothly. Below a critical SNR*N "
     "product the estimated eigenvectors swap and the error jumps by orders of "
     "magnitude. The transition is sharp enough that RMSE-versus-SNR curves "
     "are conventionally described in three regions: a saturated region below "
     "threshold, a knee, and an asymptotic region where RMSE falls about one "
     "decade per 20 dB, tracking the Cramer-Rao bound."),

    ("h2", "The Cramer-Rao bound"),
    ("p",
     "The CRB is the variance floor no unbiased estimator can beat. For a "
     "single source on a ULA it scales as:"),
    ("formula",
     "var(theta)  ~   1  /  ( SNR * N * M * (aperture)^2 * cos^2(theta) )"),
    ("p",
     "Every term is a design lever, and the aperture term is squared - which "
     "is why doubling the array length buys four times more than doubling the "
     "SNR. The cos^2(theta) term is the endfire penalty, and it is why a "
     "circular array is preferred whenever full coverage matters."),

    ("h2", "Coherent sources"),
    ("p",
     "Multipath produces arrivals that are phase-locked to each other. The "
     "source covariance is then rank deficient, the signal subspace collapses, "
     "and every subspace method fails - not gracefully, but by losing sources "
     "entirely. Spatial smoothing (averaging covariances over overlapping "
     "subarrays) restores the rank at the cost of aperture, and requires a "
     "shift-invariant geometry. Bartlett is unaffected, which is a real "
     "argument for keeping it in the comparison."),

    ("h2", "Endfire degradation"),
    ("p",
     "A linear array measures sin(theta), whose derivative cos(theta) vanishes "
     "at +-90 degrees. Near endfire a large bearing change produces almost no "
     "phase change, so the effective aperture collapses, lobes broaden, and "
     "the variance of every estimator grows as 1/cos^2(theta). A circular "
     "array has no such blind cone."),
])


# =====================================================================
# 12. Choosing
# =====================================================================
_CHOOSING = Section("choosing", "12.  Choosing an estimator", [
    ("p",
     "There is no best algorithm, only a best match to a situation. The table "
     "below maps the question actually being asked onto the method that "
     "answers it most reliably."),

    ("table",
     ["Situation", "Reach for", "Because"],
     [
         ["One emitter, real hardware, unknown calibration",
          "CDF",
          "The measured table absorbs every hardware imperfection."],
         ["One emitter, minimal channels, low cost",
          "Watson-Watt",
          "Three channels and an arctangent; decades of field history."],
         ["Two or more emitters, good SNR, known count",
          "MUSIC",
          "Super-resolution and near-optimal variance above threshold."],
         ["Same, but angular precision matters more than compute",
          "ESPRIT",
          "Search-free, so no grid floor and no precision/cost trade."],
         ["Unknown source count",
          "Bartlett or Capon",
          "Neither needs K; estimate K from the eigenvalues afterwards."],
         ["Very low SNR or very few snapshots",
          "Bartlett",
          "No threshold, no inverse, no subspace to get wrong."],
         ["Strong multipath / coherent arrivals",
          "Bartlett, or MUSIC with spatial smoothing",
          "Coherence destroys the subspace the others rely on."],
         ["Strong interferer near a weak signal",
          "Capon",
          "The adaptive null suppresses the interferer directly."],
         ["Wide aperture, few elements",
          "MUSIC on a sparse array",
          "Co-array processing exceeds the physical degrees of freedom."],
         ["Full 360 degree coverage required",
          "Any method on a UCA",
          "No endfire blind cone; bearing-independent beamwidth."],
     ]),

    ("h2", "A practical order of work"),
    ("numbers", [
        "Fix the geometry first. Aperture sets resolution and element count "
        "sets degrees of freedom; no algorithm recovers what the array did "
        "not measure.",
        "Check the sampling. Element spacing at or below half a wavelength, "
        "arc spacing likewise on a circle. Aliasing is unrecoverable.",
        "Run Bartlett as the control. It always works, and it tells you the "
        "beamwidth you are trying to beat.",
        "Estimate the source count from the eigenvalue profile before "
        "trusting any subspace result.",
        "Only then apply a super-resolution method - and compare it against "
        "the Bartlett control. If it is not clearly better, it is below "
        "threshold.",
        "Check the snapshot support before believing a Capon or MUSIC result: "
        "N > 2M at minimum.",
    ]),
])


# =====================================================================
# 13-14. Reference
# =====================================================================
_GLOSSARY = Section("glossary", "13.  Notation and glossary", [
    ("table",
     ["Symbol", "Meaning"],
     [
         ["M", "number of array elements"],
         ["K", "number of sources (emitters)"],
         ["N", "number of snapshots (time samples)"],
         ["G", "number of points on the angular scan grid"],
         ["theta", "bearing, measured from broadside"],
         ["psi", "bearing measured from the +x axis (circular arrays)"],
         ["lambda", "wavelength; all positions are expressed in wavelengths"],
         ["d", "inter-element spacing of a uniform linear array"],
         ["R (scalar)", "radius of a uniform circular array"],
         ["R (matrix)", "M x M spatial covariance of the array output"],
         ["a(theta)", "steering vector: the array's response to one source"],
         ["A", "array manifold, the M x K matrix of steering vectors"],
         ["Es, En", "signal and noise subspaces from the eigendecomposition"],
         ["J_h(z)", "Bessel function of the first kind, order h"],
         ["PHI, PSI", "the diagonal rotation ESPRIT solves for"],
         ["SNR", "per-element signal-to-noise ratio, before array gain"],
     ]),

    ("h2", "Terms"),
    ("kv", [
        ("Aperture", "The physical extent of the array. Sets resolution."),
        ("Beamwidth", "Angular width of the main lobe, ~0.886*lambda/D."),
        ("Grating lobe", "A false main lobe caused by spatial undersampling."),
        ("Co-array", "The set of all pairwise element separations. Determines "
                     "how many sources a sparse array can resolve."),
        ("Coherent sources", "Arrivals that are fully correlated, typically "
                             "direct path plus reflection."),
        ("Diagonal loading", "Adding a scaled identity to R before inversion, "
                             "to bound white-noise gain."),
        ("Phase mode", "A spatial-DFT component around a circular array; the "
                       "circular analogue of a plane-wave component."),
        ("Sense antenna", "An omnidirectional reference channel used to "
                          "resolve a front/back ambiguity."),
        ("Spatial smoothing", "Averaging covariance over overlapping "
                              "subarrays to restore rank lost to coherence."),
        ("Threshold effect", "The abrupt breakdown of a subspace estimator "
                             "below a critical SNR*N product."),
    ]),
])


_READING = Section("reading", "14.  Further reading", [
    ("h2", "Founding papers"),
    ("bullets", [
        "J. Capon, 'High-resolution frequency-wavenumber spectrum analysis', "
        "Proceedings of the IEEE, vol. 57, no. 8, 1969.",
        "R. O. Schmidt, 'Multiple emitter location and signal parameter "
        "estimation', IEEE Transactions on Antennas and Propagation, vol. 34, "
        "no. 3, 1986 (originally presented 1979).",
        "R. Roy and T. Kailath, 'ESPRIT - estimation of signal parameters via "
        "rotational invariance techniques', IEEE Transactions on Acoustics, "
        "Speech and Signal Processing, vol. 37, no. 7, 1989.",
        "D. E. N. Davies, 'A transformation between the phasing techniques "
        "required for linear and circular aerial arrays', Proceedings of the "
        "IEE, vol. 112, no. 11, 1965.",
        "A. T. Moffet, 'Minimum-redundancy linear arrays', IEEE Transactions "
        "on Antennas and Propagation, vol. 16, no. 2, 1968.",
        "P. Pal and P. P. Vaidyanathan, 'Nested arrays: a novel approach to "
        "array processing with enhanced degrees of freedom', IEEE "
        "Transactions on Signal Processing, vol. 58, no. 8, 2010.",
    ]),

    ("h2", "Performance analysis"),
    ("bullets", [
        "P. Stoica and A. Nehorai, 'MUSIC, maximum likelihood, and "
        "Cramer-Rao bound', IEEE Transactions on ASSP, vol. 37, no. 5, 1989.",
        "M. Kaveh and A. Barabell, 'The statistical performance of the MUSIC "
        "and the minimum-norm algorithms in resolving plane waves in noise', "
        "IEEE Transactions on ASSP, vol. 34, no. 2, 1986.",
        "T.-J. Shan, M. Wax and T. Kailath, 'On spatial smoothing for "
        "direction-of-arrival estimation of coherent signals', IEEE "
        "Transactions on ASSP, vol. 33, no. 4, 1985.",
    ]),

    ("h2", "Books"),
    ("bullets", [
        "H. L. Van Trees, 'Optimum Array Processing' (Detection, Estimation "
        "and Modulation Theory, Part IV), Wiley, 2002. The standard "
        "reference.",
        "P. Stoica and R. Moses, 'Spectral Analysis of Signals', Prentice "
        "Hall, 2005. Concise and rigorous on the subspace methods.",
        "R. Poisel, 'Electronic Warfare Target Location Methods', Artech "
        "House, 2012. The operational and instrument side, including "
        "Watson-Watt and correlative DF.",
    ]),
])


#: Every section, in reading order.
SECTIONS: List[Section] = [
    _OVERVIEW,
    _FAMILIES,
    _MATRIX,
    _BARTLETT,
    _CAPON,
    _CDF,
    _MUSIC,
    _ESPRIT,
    _WW,
    _GEOMETRY,
    _LIMITS,
    _CHOOSING,
    _GLOSSARY,
    _READING,
]

#: Short labels for the navigation pane.
NAV_LABELS: List[Tuple[str, str]] = [
    ("overview", "1.  What DF measures"),
    ("families", "2.  Four families"),
    ("matrix", "3.  Side-by-side table"),
    ("bartlett", "4.  Bartlett"),
    ("capon", "5.  Capon / MVDR"),
    ("cdf", "6.  Correlative (CDF)"),
    ("music", "7.  MUSIC"),
    ("esprit", "8.  ESPRIT"),
    ("ww", "9.  Watson-Watt"),
    ("geometry", "10.  Array geometries"),
    ("limits", "11.  Shared limits"),
    ("choosing", "12.  Choosing one"),
    ("glossary", "13.  Notation"),
    ("reading", "14.  Further reading"),
]

INTRO = (
    "A reference on how the six estimators in this app differ - what each one "
    "assumes, what it computes, where it excels and how it fails. Nothing on "
    "this page is simulated: it is background, to be read alongside the "
    "results the other tabs produce."
)
