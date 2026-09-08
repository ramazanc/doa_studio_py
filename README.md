# DoA Studio (Python) — RF Direction Finding algorithm comparison

A self-contained desktop app that compares six direction-of-arrival estimators
on three array geometries, on a single realisation and over Monte-Carlo
parameter sweeps — plus a descriptive comparison reference explaining how the
estimators differ.

This is a Python port of the MATLAB
[DoA-Studio](https://github.com/ramazanc/DoA-Studio), rewritten for
NumPy/SciPy/Matplotlib with a Tkinter front end. The estimator behaviour,
including every refusal and ambiguity warning, matches the original.

**No toolboxes, no heavyweight GUI stack.** The engine needs only NumPy and
SciPy (SciPy purely for `scipy.special.jv`, the Bessel function that the
circular-array phase modes are built on). The UI uses Tkinter from the standard
library plus Matplotlib. No PyQt, no App Designer equivalent, no `.ui` files.

---

## Run it

```bash
pip install -r requirements.txt
python run.py                 # or:  python -m doastudio
```

Requires Python 3.9 or later.

The app opens and runs one estimation immediately, so there is something on
screen before you touch a control.

### Drive it from a script

The engine is independent of the UI, so batch studies need no window:

```python
import numpy as np
from doastudio.core import (Scenario, array_geometry, generate_snapshots,
                            run_algorithms, doa_metrics, doa_sweep, sweep_values)

cfg = Scenario(array_type="UCA", M=12, radius=0.5, spacing=0.5,
               doa=[-30, 20], K=2, snr=8, N=400,
               keys=["music", "esprit"])

geo  = array_geometry(cfg)
data = generate_snapshots(geo, cfg.doa, cfg.snr, cfg.N)
for r in run_algorithms(geo, data, cfg.scan, cfg.K, cfg.keys):
    print(r.short, r.angles_text, doa_metrics(r.angles, cfg.doa).rmse)

S = doa_sweep(cfg, "snr", sweep_values(-15, 25, 5), trials=200, keys=cfg.keys)
print(S.rmse)
```

Pass `rng=` to `generate_snapshots` for reproducible noise.

---

## Tabs

**Single Run** — one realisation. DoA spectrum with the true bearings marked,
array geometry top view with numbered elements and the sense antenna, a bearing
compass showing true versus estimated needles, the element-1 waveform before and
after AWGN, a per-algorithm summary table (estimate, RMSE, bias, milliseconds),
and the interactive limitation panel.

**Sweep Study** — Monte-Carlo RMSE against SNR, snapshots, DoA separation,
element count, spacing/radius or source count. Linear/log y-axis, automatic
x-axis units, resolution probability subplot, and a mean-compute-time table.
Sweeps run on a worker thread with a progress bar and a **Stop** button, so the
window stays responsive.

**SNR Performance** — a dedicated, always-available RMSE-vs-SNR study, with bias
and resolution probability on the same sweep.

**Comparison** — a descriptive reference with **no simulation on it at all**:
background only. Fourteen sections covering the physics of a sampled aperture,
the four families of estimator, a side-by-side property matrix, a detailed
treatment of each of the six algorithms (principle, formula, strengths, failure
modes, history), the three array geometries, the limits that bind every method
equally, a decision guide for choosing one, notation, and references. A contents
pane on the left jumps to any section.

---

## Interactive limitation panel

`limitations.py` diagnoses the running scenario and lists what it breaks.
Clicking an entry prints the physics behind it. Entries are graded `[!]`
blocking, `[~]` performance, `[i]` note, and cover:

- more sources than the array has degrees of freedom (`K >= M`, and `K > M-2`)
- snapshot starvation (`N < M`, `N < 2M`) and what it does to the Capon inverse
- spatial aliasing for `d > lambda/2`, with the resulting ambiguity angle
- source separation inside one beamwidth, which is the Bartlett resolution wall
- the subspace threshold effect at low SNR with few snapshots
- endfire degradation, where `d(sin theta)/d(theta)` vanishes
- ESPRIT on a non-uniform array, where shift invariance does not hold
- UCA arc spacing above `lambda/2`, i.e. phase-mode aliasing
- UCA radii on a Bessel null, which blind the mode-1 channel
- single-wavefront estimators (CDF, Watson-Watt) facing multiple emitters
- the Watson-Watt sense antenna, and what it can and cannot resolve

Estimators that cannot run on a geometry refuse and say why, rather than
returning a plausible-looking wrong number. Those refusals are surfaced in the
same panel and shown as "not applicable" in the summary table.

---

## Layout

```
doa_studio_py/
├── run.py                        launcher
├── requirements.txt
├── doastudio/
│   ├── core/                     the engine — no Tk, no Matplotlib
│   │   ├── numeric.py            degree trig, rcond, exact i**h, parsing
│   │   ├── geometry.py           steering_vector, array_geometry, nla_preset
│   │   ├── scenario.py           the Scenario dataclass
│   │   ├── signals.py            generate_snapshots, add_awgn
│   │   ├── peaks.py              pick_peaks with parabolic interpolation
│   │   ├── metrics.py            per-trial RMSE, bias, resolution flag
│   │   ├── subarrays.py          esprit_subarrays, uca_modes (memoised)
│   │   ├── algorithms.py         the six estimators, registry, dispatcher
│   │   ├── sweep.py              Monte-Carlo sweep engine
│   │   └── limitations.py        scenario diagnosis
│   ├── ui/
│   │   ├── app.py                main window and tabs
│   │   ├── theme.py              palette, fonts, ttk style, mpl rcParams
│   │   ├── widgets.py            panels, form grids, embedded figures
│   │   ├── plots.py              all Matplotlib drawing
│   │   ├── runner.py             threaded sweep runner
│   │   ├── tab_single.py         Single Run
│   │   ├── tab_sweep.py          Sweep Study
│   │   ├── tab_snr.py            SNR Performance
│   │   └── tab_comparison.py     Comparison (renderer)
│   └── content/
│       └── comparison.py         Comparison page text — pure content
└── tests/
    └── test_core.py              64 regression tests
```

Every estimator takes the same arguments — `(geo, data, scan_deg, K)` — and
returns an `AlgoResult` with `.spectrum`, `.angles`, `.time`, `.valid`, `.note`,
`.has_spectrum`, `.ambiguous`. The dispatcher is a plain loop, so adding a
seventh estimator only touches the `ALGORITHMS` registry in `algorithms.py`.

---

## Two adaptations worth reading the comments for

**Watson-Watt.** The classical instrument uses two orthogonal loops plus an omni
sense antenna, and takes `theta = atan2(Y, X)`. This app has only
omnidirectional array elements, so the X/Y channel pair is synthesised, and the
construction differs by geometry.

On a *linear* array the sum and difference of a symmetric element pair give
`atan2(Im{c}, Re{c}) = 2*pi*b*sin(theta)` for baseline `b`. The sense antenna is
used to unwrap that phase from the half-baseline correlations, extending the
unambiguous baseline from 0.5 to about 1 wavelength. It cannot resolve front
from back, because a line of omni elements genuinely cannot separate `theta`
from `180-theta` and an omni at the centroid carries no directional information.

On a *circular* array phase mode 1 has a rotating figure-of-eight pattern (a
loop) and mode 0 is omnidirectional (a sense channel). Correlating mode 1
against the sense reference and dividing out the known `j*J_1(2*pi*R)` factor
gives `z ~ exp(j*psi)`, whose real and imaginary parts are the true Watson-Watt
X and Y channels. This is the genuine cardioid conversion, and it does resolve
the full 360 degrees. Without a sense reference only `Y_1*conj(Y_-1)` is
available, which yields `2*psi` and hence the classical 180 degree ambiguity.

Watson-Watt reports one bearing whatever the source count. With two emitters it
returns their instantaneous vector sum, which is the "wandering bearing" of real
Watson-Watt sets in multipath. The RMSE-vs-SNR curve is flat for that reason,
and the limitation panel says so.

**ESPRIT subarrays.** ESPRIT needs two identical subarrays displaced by a fixed
vector, so that `A2 = A1*PHI`. `esprit_subarrays` searches the geometry for the
largest set of element pairs sharing one displacement: for a ULA that is the
classic `0..M-2` / `1..M-1` split, and a sparse array qualifies only if it
happens to contain a shift-invariant subset. Where none exists the checkbox is
disabled and labelled N/A.

A circle has no such pair — every chord points in a different direction — so
invariance is manufactured with the Davies phase-mode transform, which maps the
UCA onto a virtual ULA of `2H+1` modes. `uca_modes` decides how many modes are
honest, rejecting those that are Bessel-nulled or swamped by their aliases at
`h +/- M`. Both costs are real and documented: the transform colours the noise,
and small circles support too few modes to resolve many sources.

---

## Notes on the port

The engine is a faithful translation, but a few things are done the Python way:

- **`Scenario` dataclass** replaces the ad-hoc MATLAB `cfg` struct, and the
  sweep engine derives modified copies with `Scenario.copy(...)`.
- **`AlgoResult` dataclass** replaces the struct array, so the dispatcher no
  longer has to strip optional fields to keep the array homogeneous.
- **Memoisation.** `esprit_subarrays` and `uca_modes` are pure functions of the
  geometry and are called once per Monte-Carlo trial, so both are cached. A
  60-point × 200-trial sweep would otherwise repeat the same subarray search
  12 000 times.
- **Explicit RNG.** `generate_snapshots(..., rng=...)` accepts a seed or a
  `numpy.random.Generator` for reproducible runs; MATLAB's global `randn` had no
  equivalent.
- **Threaded sweeps.** Long studies run off the UI thread with a progress bar
  and a working Stop button.
- **Index base.** MATLAB's 1-based indexing is translated throughout; element
  numbering in the geometry plot still starts at 1 for the display only.

MATLAB constructs and their replacements: `eig` on a Hermitian matrix →
`numpy.linalg.eigh` with an explicit descending sort; `svd(C, 0)` →
`numpy.linalg.svd(C, full_matrices=True)` (MATLAB returns a full `V` for a wide
matrix, NumPy's economy mode does not); `-V12 / V22` → an explicit inverse;
`rcond` → `1/numpy.linalg.cond(A, 1)`; `besselj` → `scipy.special.jv`; `sind`,
`cosd`, `asind` → helpers in `core/numeric.py`.

---

## Tests

```bash
python -m unittest discover -s tests -t . -v
```

64 tests covering the array manifold, the signal model, peak interpolation,
metrics, all six estimators, shift-invariance detection, the circular-array
Bessel nulls, the sweep engine and the limitation diagnoses. They assert
physics rather than implementation: a steering vector must reproduce the
textbook ULA phase ramp, MUSIC must beat Bartlett above threshold, Watson-Watt's
two-source error must stay flat with SNR, and the estimators that cannot legally
run on a geometry must refuse.
