"""Toolbox-free DoA estimation engine (NumPy/SciPy only)."""

from .algorithms import (ALGORITHM_BY_KEY, ALGORITHMS, AlgoResult, AlgorithmSpec,
                         bartlett_doa, capon_doa, corr_interferometry_doa,
                         esprit_doa, music_doa, run_algorithms, watson_watt_doa)
from .geometry import ArrayGeometry, array_geometry, nla_preset, steering_vector
from .limitations import CRIT, INFO, WARN, Limitation, doa_limitations
from .metrics import DoAMetrics, doa_metrics
from .peaks import pick_peaks
from .scenario import SCAN_STEP, Scenario, default_doa, default_scan
from .signals import Snapshots, add_awgn, generate_snapshots
from .subarrays import UcaModes, esprit_subarrays, uca_modes
from .sweep import (SWEEP_DEFAULTS, SWEEP_LABELS, SWEEP_TYPES, SweepResult,
                    doa_sweep, sweep_values)

__all__ = [
    "ALGORITHMS", "ALGORITHM_BY_KEY", "AlgoResult", "AlgorithmSpec",
    "ArrayGeometry", "CRIT", "DoAMetrics", "INFO", "Limitation", "SCAN_STEP",
    "SWEEP_DEFAULTS", "SWEEP_LABELS", "SWEEP_TYPES", "Scenario", "Snapshots",
    "SweepResult", "UcaModes", "WARN", "add_awgn", "array_geometry",
    "bartlett_doa", "capon_doa", "corr_interferometry_doa", "default_doa",
    "default_scan", "doa_limitations", "doa_metrics", "doa_sweep",
    "esprit_doa", "esprit_subarrays", "generate_snapshots", "music_doa",
    "nla_preset", "pick_peaks", "run_algorithms", "steering_vector",
    "sweep_values", "uca_modes", "watson_watt_doa",
]
