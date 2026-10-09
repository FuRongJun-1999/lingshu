"""Measurement guards: denominator, numeric validity, real call coverage and purity."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.measure_hex_activity import ActivationCounts, _MeasuredHexHierNet, measure
from lingshu.nn.hex_hier import HexHierNet


class ActivationMeasurementTests(unittest.TestCase):
    def test_aggregate_weights_elements_instead_of_averaging_layer_rates(self):
        meter = ActivationCounts()
        meter.record("small", np.array([0.0]))
        meter.record("large", np.ones(9))
        report = meter.report()
        self.assertEqual(report["layers"]["small"]["activation_sparsity"], 1.0)
        self.assertEqual(report["aggregate"]["zero_elements"], 1)
        self.assertEqual(report["aggregate"]["elements"], 10)
        self.assertEqual(report["aggregate"]["activation_sparsity"], 0.1)

    def test_exact_zero_does_not_turn_small_nonzero_values_into_zero(self):
        meter = ActivationCounts()
        meter.record("layer", np.array([0.0, -0.0, 1e-12, -1e-12, 1.0]))
        self.assertEqual(meter.report()["aggregate"]["zero_elements"], 2)
        self.assertEqual(meter.report()["aggregate"]["activation_sparsity"], 0.4)

    def test_empty_measurement_is_explicit_and_has_no_rate(self):
        for values in (None, np.array([])):
            meter = ActivationCounts()
            if values is not None:
                meter.record("layer", values)
            report = meter.report()["aggregate"]
            self.assertEqual(report["status"], "not_measured")
            self.assertEqual(report["elements"], 0)
            self.assertIsNone(report["activation_sparsity"])

    def test_nonfinite_values_invalidate_rate_without_dropping_denominator(self):
        meter = ActivationCounts()
        meter.record("layer", np.array([0.0, 1.0, np.nan, np.inf, -np.inf]))
        report = meter.report()["aggregate"]
        self.assertEqual(report["elements"], 5)
        self.assertEqual(report["nonfinite_elements"], 3)
        self.assertEqual(report["status"], "invalid")
        self.assertIsNone(report["activation_sparsity"])

    def test_chunking_does_not_change_counts_or_weighted_rate(self):
        values = np.array([0.0, 1.0, 0.0, 2.0, 3.0, 4.0])
        whole, chunked = ActivationCounts(), ActivationCounts()
        whole.record("layer", values)
        chunked.record("layer", values[:1])
        chunked.record("layer", values[1:])
        for key in ("elements", "zero_elements", "nonfinite_elements", "activation_sparsity"):
            self.assertEqual(whole.report()["aggregate"][key],
                             chunked.report()["aggregate"][key])

    def test_instrumentation_preserves_forward_outputs_and_parameters(self):
        lattices = np.random.default_rng(3).normal(size=(2, 5, 4, 3))
        for stacked in (False, True):
            with self.subTest(stacked=stacked):
                plain = HexHierNet(seed=7, stacked=stacked)
                measured = _MeasuredHexHierNet(seed=7, stacked=stacked)
                before = measured.get_vec().copy()
                expected = plain.l2_features(lattices)
                actual = measured.l2_features(lattices)
                for left, right in zip(expected, actual):
                    np.testing.assert_array_equal(left, right)
                np.testing.assert_array_equal(before, measured.get_vec())
                layers = measured.activity.report()["layers"]
                self.assertEqual(layers["L1"]["elements"], 2 * 5 * 4 * plain.K)
                self.assertEqual("L1.5" in layers, stacked)
                if stacked:
                    self.assertEqual(layers["L1.5"]["elements"], 2 * 5 * 4 * plain.K2)
                # Every zero tensor is exactly zero after the real activation call.
                blank = _MeasuredHexHierNet(seed=7, stacked=stacked)
                blank.l2_features(np.zeros_like(lattices))
                self.assertEqual(blank.activity.report()["aggregate"]["activation_sparsity"], 1.0)

    def test_repeated_forwards_keep_layer_labels_and_count_all_batches(self):
        net = _MeasuredHexHierNet(n_kernels=4, n_kernels2=2, seed=7)
        for _ in range(2):
            net.l2_features(np.ones((3, 5, 4, 3)))
        layers = net.activity.report()["layers"]
        self.assertEqual(layers["L1"]["elements"], 2 * 3 * 5 * 4 * 4)
        self.assertEqual(layers["L1.5"]["elements"], 2 * 3 * 5 * 4 * 2)

    def test_seeded_report_is_repeatable_and_declares_scope(self):
        first = measure(samples=2, cells_across=4, seed=7)
        self.assertEqual(first, measure(samples=2, cells_across=4, seed=7))
        self.assertFalse(first["conditions"]["trained"])
        self.assertEqual(first["scope"], ["L1 post-LeakyReLU", "L1.5 post-LeakyReLU"])
        second = measure(samples=2, cells_across=4, seed=8)
        self.assertNotEqual(first["conditions"]["input_sha256"], second["conditions"]["input_sha256"])

    def test_zero_samples_are_rejected(self):
        with self.assertRaises(ValueError):
            measure(samples=0)

    def test_cli_runs_outside_checkout_and_emits_strict_json(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "measure_hex_activity.py"
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-X", "utf8", str(script), "--samples", "2",
                 "--cells-across", "4", "--unstacked"],
                cwd=directory, capture_output=True, text=True, encoding="utf-8", timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["scope"], ["L1 post-LeakyReLU"])
        self.assertEqual(report["aggregate"]["status"], "measured")
        json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
