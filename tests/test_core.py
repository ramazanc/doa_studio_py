"""Regression tests for the DoA engine.

Run with:  python -m unittest discover -s tests -v
(or with pytest, which picks the same classes up.)

The tests assert physics, not implementation details: a steering vector must
reproduce the textbook ULA phase ramp, an estimator must find a source it was
given, and the estimators that cannot legally run on a geometry must refuse
rather than return a plausible-looking wrong number.
"""

from __future__ import annotations

import unittest

import numpy as np

from doastudio.core import (ALGORITHMS, Scenario, array_geometry, doa_limitations,
                            doa_metrics, doa_sweep, esprit_subarrays,
                            generate_snapshots, nla_preset, pick_peaks,
                            run_algorithms, steering_vector, sweep_values,
                            uca_modes)
from doastudio.core.algorithms import (bartlett_doa, capon_doa,
                                       corr_interferometry_doa, esprit_doa,
                                       music_doa, watson_watt_doa)
from doastudio.core.numeric import parse_number_list, sind

ALL_KEYS = [a.key for a in ALGORITHMS]


def scenario(**kw) -> Scenario:
    base = dict(array_type="ULA", M=8, spacing=0.5, radius=0.5,
                doa=[-20.0, 30.0], K=2, snr=20.0, N=400, keys=list(ALL_KEYS))
    base.update(kw)
    return Scenario(**base)


def snapshots(cfg, seed=0):
    geo = array_geometry(cfg)
    return geo, generate_snapshots(geo, cfg.doa, cfg.snr, cfg.N,
                                   rng=np.random.default_rng(seed))


# =====================================================================
class TestSteeringVector(unittest.TestCase):
    def test_ula_phase_ramp(self):
        """A ULA manifold must be exp(j*2*pi*d*m*sin(theta))."""
        M, d, theta = 6, 0.5, 25.0
        x = (np.arange(M) - (M - 1) / 2.0) * d
        pos = np.column_stack([x, np.zeros(M)])
        a = steering_vector(pos, theta).ravel()
        expected = np.exp(1j * 2 * np.pi * x * sind(theta))
        np.testing.assert_allclose(a, expected, atol=1e-12)

    def test_broadside_is_all_ones(self):
        """At 0 degrees every element of a linear array is co-phased."""
        pos = np.column_stack([np.linspace(-2, 2, 5), np.zeros(5)])
        a = steering_vector(pos, 0.0).ravel()
        np.testing.assert_allclose(a, np.ones(5), atol=1e-12)

    def test_unit_modulus(self):
        pos = np.column_stack([np.random.default_rng(1).normal(size=7),
                               np.random.default_rng(2).normal(size=7)])
        A = steering_vector(pos, np.arange(-90, 91, 10.0))
        np.testing.assert_allclose(np.abs(A), 1.0, atol=1e-12)


# =====================================================================
class TestGeometry(unittest.TestCase):
    def test_ula_is_centred(self):
        geo = array_geometry(scenario(array_type="ULA", M=8, spacing=0.5))
        self.assertAlmostEqual(float(geo.pos[:, 0].mean()), 0.0, places=12)
        self.assertAlmostEqual(geo.aperture, 3.5, places=12)
        self.assertEqual(geo.M, 8)

    def test_uca_radius_and_spacing(self):
        geo = array_geometry(scenario(array_type="UCA", M=12, radius=0.8))
        self.assertAlmostEqual(geo.radius, 0.8, places=12)
        np.testing.assert_allclose(np.linalg.norm(geo.pos, axis=1), 0.8, atol=1e-12)
        self.assertAlmostEqual(geo.aperture, 1.6, places=12)

    def test_sense_antenna_is_at_the_centroid_and_excluded(self):
        geo = array_geometry(scenario(sense=True))
        self.assertIsNotNone(geo.sense_pos)
        np.testing.assert_allclose(geo.sense_pos.ravel(),
                                   geo.pos.mean(axis=0), atol=1e-12)
        # The sense element must NOT change the covariance dimension.
        self.assertEqual(geo.pos.shape[0], 8)

        geo_off = array_geometry(scenario(sense=False))
        self.assertIsNone(geo_off.sense_pos)

    def test_nla_preset_is_minimum_redundancy(self):
        x = nla_preset(4, 0.5)
        np.testing.assert_allclose(x, np.array([0, 1, 4, 6]) * 0.5)
        self.assertEqual(nla_preset(15, 0.5).size, 15)   # nested fallback

    def test_unknown_array_type_raises(self):
        with self.assertRaises(ValueError):
            array_geometry(scenario(array_type="XYZ"))


# =====================================================================
class TestSignalModel(unittest.TestCase):
    def test_requested_snr_is_realised(self):
        cfg = scenario(snr=7.0, N=20000)
        _, data = snapshots(cfg, seed=3)
        self.assertAlmostEqual(data.measured_snr_db, 7.0, delta=0.15)

    def test_covariance_is_hermitian(self):
        _, data = snapshots(scenario(), seed=4)
        np.testing.assert_allclose(data.R, data.R.conj().T, atol=1e-12)

    def test_sense_channel_present_only_when_enabled(self):
        _, on = snapshots(scenario(sense=True), seed=5)
        _, off = snapshots(scenario(sense=False), seed=5)
        self.assertIsNotNone(on.xs)
        self.assertIsNone(off.xs)
        self.assertEqual(on.xs.shape, (on.X.shape[1],))

    def test_reproducible_with_a_seed(self):
        _, a = snapshots(scenario(), seed=11)
        _, b = snapshots(scenario(), seed=11)
        np.testing.assert_allclose(a.X, b.X)


# =====================================================================
class TestPeakPicking(unittest.TestCase):
    def test_finds_two_peaks_in_order(self):
        scan = np.arange(-90, 90.5, 0.5)
        spec = (np.exp(-((scan + 20) / 2.0) ** 2)
                + np.exp(-((scan - 30) / 2.0) ** 2))
        ang = pick_peaks(spec, scan, 2)
        np.testing.assert_allclose(ang, [-20.0, 30.0], atol=0.05)

    def test_sub_grid_interpolation_beats_the_grid_step(self):
        """The parabolic vertex must resolve a peak off the sample grid."""
        scan = np.arange(-90, 90.5, 0.5)
        true = 12.37                                  # deliberately off-grid
        spec = np.exp(-((scan - true) / 1.5) ** 2)
        ang = pick_peaks(spec, scan, 1)[0]
        self.assertLess(abs(ang - true), 0.25)        # better than the grid

    def test_flat_spectrum_falls_back_to_the_maximum(self):
        scan = np.arange(-90, 90.5, 0.5)
        ang = pick_peaks(np.ones_like(scan), scan, 2)
        self.assertEqual(ang.size, 1)


# =====================================================================
class TestMetrics(unittest.TestCase):
    def test_perfect_estimate(self):
        m = doa_metrics([-20.0, 30.0], [-20.0, 30.0])
        self.assertAlmostEqual(m.rmse, 0.0)
        self.assertAlmostEqual(m.bias, 0.0)
        self.assertTrue(m.resolved)

    def test_missing_estimate_is_penalised(self):
        """An estimator must not win by refusing to report a source."""
        m = doa_metrics([-20.0], [-20.0, 30.0])
        self.assertGreater(m.rmse, 100.0)
        self.assertFalse(m.resolved)

    def test_nan_estimate_is_penalised(self):
        m = doa_metrics([np.nan, np.nan], [-20.0, 30.0])
        self.assertGreater(m.rmse, 100.0)

    def test_tolerance_defaults_to_half_the_separation(self):
        m = doa_metrics([0.0, 4.0], [0.0, 4.0])
        self.assertAlmostEqual(m.tol, 2.0)
        m2 = doa_metrics([0.0, 40.0], [0.0, 40.0])
        self.assertAlmostEqual(m2.tol, 5.0)          # capped


# =====================================================================
class TestEstimatorsOnULA(unittest.TestCase):
    """At 20 dB with 400 snapshots every estimator should find the sources."""

    def setUp(self):
        self.cfg = scenario(snr=20.0, N=400)
        self.geo, self.data = snapshots(self.cfg, seed=7)

    def _check(self, fn, tol):
        res = fn(self.geo, self.data, self.cfg.scan, self.cfg.K)
        self.assertTrue(res.valid, res.note)
        np.testing.assert_allclose(np.sort(res.angles), [-20.0, 30.0], atol=tol)
        self.assertGreater(res.time, 0.0)
        return res

    def test_bartlett(self):
        self._check(bartlett_doa, 0.6)

    def test_capon(self):
        self._check(capon_doa, 0.3)

    def test_music(self):
        self._check(music_doa, 0.2)

    def test_esprit(self):
        res = self._check(esprit_doa, 0.3)
        self.assertFalse(res.has_spectrum)      # search-free: cosmetic curve

    def test_cdf_finds_both_but_less_accurately(self):
        self._check(corr_interferometry_doa, 2.0)

    def test_spectra_are_normalised_to_zero_db(self):
        for fn in (bartlett_doa, capon_doa, music_doa, corr_interferometry_doa):
            res = fn(self.geo, self.data, self.cfg.scan, self.cfg.K)
            self.assertAlmostEqual(float(res.spectrum.max()), 0.0, places=9)

    def test_music_beats_bartlett_at_high_snr(self):
        """Super-resolution must actually pay off above threshold."""
        b = doa_metrics(bartlett_doa(self.geo, self.data, self.cfg.scan, 2).angles,
                        self.cfg.doa)
        m = doa_metrics(music_doa(self.geo, self.data, self.cfg.scan, 2).angles,
                        self.cfg.doa)
        self.assertLess(m.rmse, b.rmse)


# =====================================================================
class TestSingleSourceAccuracy(unittest.TestCase):
    """Watson-Watt and CDF are single-wavefront: give them one wavefront."""

    def test_watson_watt_on_ula(self):
        cfg = scenario(doa=[25.0], K=1, snr=25.0, N=800)
        geo, data = snapshots(cfg, seed=9)
        res = watson_watt_doa(geo, data, cfg.scan, 1)
        self.assertTrue(res.valid, res.note)
        self.assertAlmostEqual(float(res.angles[0]), 25.0, delta=1.0)

    def test_watson_watt_on_uca_resolves_full_circle(self):
        cfg = scenario(array_type="UCA", M=12, radius=0.5, spacing=0.5,
                       doa=[-40.0], K=1, snr=25.0, N=800)
        geo, data = snapshots(cfg, seed=10)
        res = watson_watt_doa(geo, data, cfg.scan, 1)
        self.assertTrue(res.valid, res.note)
        self.assertAlmostEqual(float(res.angles[0]), -40.0, delta=1.0)

    def test_cdf_single_source(self):
        cfg = scenario(doa=[15.0], K=1, snr=25.0, N=800)
        geo, data = snapshots(cfg, seed=12)
        res = corr_interferometry_doa(geo, data, cfg.scan, 1)
        self.assertAlmostEqual(float(res.angles[0]), 15.0, delta=0.5)

    def test_watson_watt_returns_one_bearing_per_source_slot(self):
        """With K sources it still reports a single wandering bearing."""
        cfg = scenario(K=2, doa=[-20.0, 30.0])
        geo, data = snapshots(cfg, seed=13)
        res = watson_watt_doa(geo, data, cfg.scan, 2)
        self.assertEqual(res.angles.size, 2)
        self.assertAlmostEqual(res.angles[0], res.angles[1])


# =====================================================================
class TestEspritSubarrays(unittest.TestCase):
    def test_ula_splits_into_the_classic_pair(self):
        geo = array_geometry(scenario(array_type="ULA", M=6, spacing=0.5))
        i1, i2, delta = esprit_subarrays(geo.pos)
        self.assertEqual(i1.size, 5)
        np.testing.assert_array_equal(i1, np.arange(5))
        np.testing.assert_array_equal(i2, np.arange(1, 6))
        np.testing.assert_allclose(delta, [0.5, 0.0], atol=1e-12)

    def test_mrla5_has_no_repeated_lag(self):
        """Every lag of the 5-element MRLA is unique, so ESPRIT cannot run."""
        pos = np.column_stack([nla_preset(5, 0.5), np.zeros(5)])
        i1, i2, delta = esprit_subarrays(pos)
        self.assertIsNone(i1)

    def test_nested_layout_does_contain_an_invariant_subset(self):
        pos = np.column_stack([nla_preset(8, 0.5), np.zeros(8)])
        i1, _, _ = esprit_subarrays(pos)
        self.assertIsNotNone(i1)
        self.assertGreaterEqual(i1.size, 2)

    def test_circle_has_no_shift_invariant_pair(self):
        geo = array_geometry(scenario(array_type="UCA", M=7, radius=0.5))
        i1, _, _ = esprit_subarrays(geo.pos)
        self.assertIsNone(i1)

    def test_esprit_refuses_on_a_non_invariant_array(self):
        cfg = scenario(array_type="NLA", pos_vec=nla_preset(5, 0.5), K=2)
        geo, data = snapshots(cfg, seed=14)
        res = esprit_doa(geo, data, cfg.scan, cfg.K)
        self.assertFalse(res.valid)
        self.assertIn("invariant", res.note)


# =====================================================================
class TestUcaModes(unittest.TestCase):
    def test_bessel_null_blinds_mode_one(self):
        """J1 first vanishes at 2*pi*R = 3.8317, i.e. R = 0.6098."""
        u = uca_modes(3.8317 / (2 * np.pi), 12)
        self.assertFalse(u.mode1_ok)
        self.assertIn("J1", u.why)

    def test_healthy_radius_supports_mode_one(self):
        u = uca_modes(0.5, 12)
        self.assertTrue(u.mode1_ok)
        self.assertEqual(u.why, "")
        self.assertGreaterEqual(u.n_modes, 3)

    def test_arc_spacing_above_half_wavelength_aliases(self):
        """Few elements on a big circle undersample the wavefront."""
        u = uca_modes(2.0, 6)                       # arc = 2*pi*2/6 = 2.09
        self.assertGreater(u.arc, 0.5)
        self.assertFalse(u.mode1_ok)

    def test_watson_watt_refuses_at_the_bessel_null(self):
        cfg = scenario(array_type="UCA", M=12, radius=0.6098, spacing=0.6098,
                       doa=[10.0], K=1)
        geo, data = snapshots(cfg, seed=15)
        res = watson_watt_doa(geo, data, cfg.scan, 1)
        self.assertFalse(res.valid)
        self.assertIn("mode-1", res.note)

    def test_esprit_refuses_when_too_few_modes(self):
        cfg = scenario(array_type="UCA", M=12, radius=0.6098, spacing=0.6098,
                       doa=[-10.0, 20.0], K=2)
        geo, data = snapshots(cfg, seed=16)
        res = esprit_doa(geo, data, cfg.scan, cfg.K)
        self.assertFalse(res.valid)
        self.assertIn("phase mode", res.note)

    def test_element_space_methods_are_unaffected_by_a_bessel_null(self):
        """Bartlett/Capon/MUSIC never form modes, so the null cannot hurt."""
        cfg = scenario(array_type="UCA", M=12, radius=0.6098, spacing=0.6098,
                       doa=[15.0], K=1, snr=20.0, N=400)
        geo, data = snapshots(cfg, seed=17)
        for fn in (bartlett_doa, capon_doa, music_doa):
            res = fn(geo, data, cfg.scan, 1)
            self.assertTrue(res.valid)
            self.assertAlmostEqual(float(res.angles[0]), 15.0, delta=1.0)


# =====================================================================
class TestDispatcher(unittest.TestCase):
    def test_returns_one_result_per_key_in_order(self):
        cfg = scenario()
        geo, data = snapshots(cfg, seed=18)
        keys = ["music", "bartlett", "esprit"]
        res = run_algorithms(geo, data, cfg.scan, cfg.K, keys)
        self.assertEqual([r.key for r in res], keys)
        for r in res:
            self.assertTrue(r.short)
            self.assertEqual(len(r.color), 3)

    def test_unknown_key_is_skipped(self):
        cfg = scenario()
        geo, data = snapshots(cfg, seed=19)
        res = run_algorithms(geo, data, cfg.scan, cfg.K, ["music", "nope"])
        self.assertEqual(len(res), 1)

    def test_every_registered_algorithm_runs_on_a_ula(self):
        cfg = scenario()
        geo, data = snapshots(cfg, seed=20)
        res = run_algorithms(geo, data, cfg.scan, cfg.K, ALL_KEYS)
        self.assertEqual(len(res), 6)
        self.assertTrue(all(r.valid for r in res))


# =====================================================================
class TestSweep(unittest.TestCase):
    def test_rmse_falls_with_snr_for_subspace_methods(self):
        cfg = scenario(N=200)
        S = doa_sweep(cfg, "snr", [0.0, 10.0, 20.0], 12, ["music", "capon"])
        for a in range(2):
            self.assertGreater(S.rmse[0, a], S.rmse[2, a])
        self.assertTrue(S.completed)

    def test_watson_watt_rmse_is_flat_with_two_sources(self):
        """The wandering bearing is a bias, not noise: SNR cannot fix it."""
        cfg = scenario(N=200)
        S = doa_sweep(cfg, "snr", [0.0, 20.0], 12, ["ww"])
        self.assertGreater(S.rmse[0, 0], 10.0)
        self.assertGreater(S.rmse[1, 0], 10.0)

    def test_impossible_points_are_skipped_not_raised(self):
        """K >= M must leave a gap in the curve."""
        cfg = scenario(array_type="ULA", M=3, K=2, doa=[-20.0, 30.0])
        S = doa_sweep(cfg, "sources", [1.0, 2.0, 5.0], 4, ["music"])
        self.assertFalse(np.isnan(S.rmse[0, 0]))
        self.assertTrue(np.isnan(S.rmse[2, 0]))      # K = 5 > M = 3

    def test_separation_sweep_recentres_the_constellation(self):
        from doastudio.core.sweep import apply_sweep
        cfg = scenario(doa=[-20.0, 30.0], K=2)
        c = apply_sweep(cfg, "separation", 10.0)
        self.assertAlmostEqual(float(np.mean(c.doa)), 5.0)
        self.assertAlmostEqual(float(np.diff(c.doa)[0]), 10.0)

    def test_elements_sweep_rebuilds_a_sparse_layout(self):
        from doastudio.core.sweep import apply_sweep
        cfg = scenario(array_type="NLA", pos_vec=nla_preset(5, 0.5), K=1,
                       doa=[0.0])
        c = apply_sweep(cfg, "elements", 7)
        self.assertEqual(c.pos_vec.size, 7)
        self.assertEqual(c.M, 7)

    def test_spacing_sweep_moves_radius_too(self):
        from doastudio.core.sweep import apply_sweep
        c = apply_sweep(scenario(array_type="UCA"), "spacing", 0.9)
        self.assertAlmostEqual(c.spacing, 0.9)
        self.assertAlmostEqual(c.radius, 0.9)

    def test_cancellation_marks_the_result_incomplete(self):
        S = doa_sweep(scenario(), "snr", [0.0, 5.0, 10.0], 5, ["music"],
                      should_stop=lambda: True)
        self.assertFalse(S.completed)

    def test_sweep_values_handles_descending_and_zero_step(self):
        np.testing.assert_allclose(sweep_values(0, 10, 5), [0, 5, 10])
        np.testing.assert_allclose(sweep_values(10, 0, 5), [10, 5, 0])
        np.testing.assert_allclose(sweep_values(3, 9, 0), [3])

    def test_unknown_sweep_type_raises(self):
        with self.assertRaises(ValueError):
            doa_sweep(scenario(), "banana", [1.0], 1, ["music"])


# =====================================================================
class TestLimitations(unittest.TestCase):
    def _titles(self, cfg):
        geo = array_geometry(cfg)
        return [lim.title for lim in doa_limitations(cfg, geo)]

    def test_too_many_sources_is_blocking(self):
        cfg = scenario(M=4, K=4, doa=[-40.0, -10.0, 10.0, 40.0])
        geo = array_geometry(cfg)
        lims = doa_limitations(cfg, geo)
        self.assertTrue(any(l.severity == "crit" for l in lims))
        self.assertTrue(any("4 sources" in l.title for l in lims))

    def test_snapshot_starvation_is_reported(self):
        titles = " ".join(self._titles(scenario(M=8, N=5)))
        self.assertIn("snapshots", titles)

    def test_spatial_aliasing_is_reported(self):
        titles = " ".join(self._titles(scenario(spacing=0.9)))
        self.assertIn("sampling limit", titles)

    def test_endfire_is_reported(self):
        titles = " ".join(self._titles(scenario(doa=[80.0], K=1)))
        self.assertIn("endfire", titles)

    def test_clean_scenario_reports_nothing_blocking(self):
        cfg = scenario(M=10, N=1000, snr=20.0, doa=[-30.0, 30.0],
                       keys=["bartlett", "capon", "music"])
        geo = array_geometry(cfg)
        lims = doa_limitations(cfg, geo)
        self.assertFalse(any(l.severity == "crit" for l in lims))

    def test_runtime_refusals_are_surfaced(self):
        cfg = scenario(array_type="NLA", pos_vec=nla_preset(5, 0.5), K=2)
        geo, data = snapshots(cfg, seed=21)
        results = run_algorithms(geo, data, cfg.scan, cfg.K, cfg.keys)
        lims = doa_limitations(cfg, geo, results)
        self.assertTrue(any("could not run" in l.title for l in lims))

    def test_every_limitation_has_prose(self):
        for cfg in (scenario(), scenario(array_type="UCA", M=12, radius=0.61,
                                         spacing=0.61),
                    scenario(M=4, K=3, doa=[-30.0, 0.0, 30.0], N=6, spacing=0.8)):
            for lim in doa_limitations(cfg, array_geometry(cfg)):
                self.assertTrue(lim.title.strip())
                self.assertGreater(len(lim.text), 60, lim.title)
                self.assertTrue(lim.algos.strip())
                self.assertIn(lim.severity, ("crit", "warn", "info"))


# =====================================================================
class TestParsing(unittest.TestCase):
    def test_number_list_accepts_several_separators(self):
        for text in ("-20, 30", "-20 30", "[-20, 30]", "-20;30"):
            np.testing.assert_allclose(parse_number_list(text), [-20.0, 30.0])

    def test_garbage_is_skipped_not_raised(self):
        np.testing.assert_allclose(parse_number_list("-20, abc, 30"),
                                   [-20.0, 30.0])
        self.assertEqual(parse_number_list("").size, 0)
        self.assertEqual(parse_number_list(None).size, 0)


# =====================================================================
class TestContent(unittest.TestCase):
    """The comparison page is static content, but it must stay well formed."""

    def test_sections_and_nav_agree(self):
        from doastudio.content.comparison import NAV_LABELS, SECTIONS
        self.assertEqual([s.id for s in SECTIONS], [n[0] for n in NAV_LABELS])

    def test_every_block_is_renderable(self):
        from doastudio.content.comparison import SECTIONS
        known = {"h2", "p", "bullets", "numbers", "table", "formula", "kv",
                 "note"}
        for sec in SECTIONS:
            self.assertTrue(sec.title.strip())
            self.assertTrue(sec.blocks, sec.id)
            for block in sec.blocks:
                self.assertIn(block[0], known, "%s: %s" % (sec.id, block[0]))
                if block[0] == "table":
                    headers, rows = block[1], block[2]
                    for row in rows:
                        self.assertEqual(len(row), len(headers),
                                         "%s: ragged table row %r" % (sec.id, row[0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
