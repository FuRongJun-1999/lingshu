"""Measurement guards: denominator, numeric validity, real call coverage and purity."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.measure_hex_activity import ActivationCounts, _MeasuredHexHierNet, measure
from lingshu.nn.hex_hier import HexHierNet
from lingshu.nn.hex_train import hex_conv_batch_multi


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
        for stacked, k1, k2 in ((False, 6, 6), (True, 6, 6),
                                (True, 4, 2), (True, 2, 4)):
            with self.subTest(stacked=stacked, k1=k1, k2=k2):
                kwargs = dict(seed=7, stacked=stacked, n_kernels=k1, n_kernels2=k2)
                plain = HexHierNet(**kwargs)
                measured = _MeasuredHexHierNet(**kwargs)
                before = measured.get_vec().copy()
                expected = plain.l2_features(lattices)
                actual = measured.l2_features(lattices)
                for left, right in zip(expected, actual):
                    np.testing.assert_array_equal(left, right)
                np.testing.assert_array_equal(before, measured.get_vec())
                layers = measured.activity.report()["layers"]
                self.assertEqual(layers["L1"]["elements"], 2 * 5 * 4 * plain.K)
                h1 = plain._h1(lattices)
                self.assertEqual(layers["L1"]["zero_elements"],
                                 int(np.count_nonzero(h1 == 0)))
                self.assertEqual(layers["L1"]["calls"], 1)
                self.assertEqual("L1.5" in layers, stacked)
                if stacked:
                    self.assertEqual(layers["L1.5"]["elements"], 2 * 5 * 4 * plain.K2)
                    h2 = plain._lrelu(hex_conv_batch_multi(h1, plain.conv2))
                    self.assertEqual(layers["L1.5"]["zero_elements"],
                                     int(np.count_nonzero(h2 == 0)))
                    self.assertEqual(layers["L1.5"]["calls"], 1)
                self.assertIsNone(measured._activation_layer)
                # Every zero tensor is exactly zero after the real activation call.
                blank = _MeasuredHexHierNet(**kwargs)
                blank.l2_features(np.zeros_like(lattices))
                self.assertEqual(blank.activity.report()["aggregate"]["activation_sparsity"], 1.0)

    def test_repeated_forwards_keep_layer_labels_and_count_all_batches(self):
        for k1, k2 in ((6, 6), (4, 2), (2, 4)):
            with self.subTest(k1=k1, k2=k2):
                net = _MeasuredHexHierNet(n_kernels=k1, n_kernels2=k2, seed=7)
                net.l2_features(np.zeros((3, 5, 4, 3)))
                net.l2_features(np.ones((1, 5, 4, 3)))
                layers = net.activity.report()["layers"]
                self.assertEqual(layers["L1"]["elements"], 4 * 5 * 4 * k1)
                self.assertEqual(layers["L1.5"]["elements"], 4 * 5 * 4 * k2)
                for layer in layers.values():
                    self.assertEqual(layer["calls"], 2)
                self.assertIsNone(net._activation_layer)

    def test_layer_counts_match_hand_computable_activations(self):
        for k1, k2 in ((6, 6), (4, 2), (2, 4)):
            with self.subTest(k1=k1, k2=k2):
                net = _MeasuredHexHierNet(n_kernels=k1, n_kernels2=k2, seed=7)
                # Only the first L1 kernel is zero; L1.5 reads a nonzero kernel.
                net.conv.fill(0)
                net.conv[1:, 0] = 1
                net.conv2.fill(0)
                net.conv2[:, 1, 0] = 1
                net.l2_features(np.ones((2, 5, 4, 3)))
                report = net.activity.report()
                self.assertEqual(report["layers"]["L1"]["elements"], 40 * k1)
                self.assertEqual(report["layers"]["L1"]["zero_elements"], 40)
                self.assertEqual(report["layers"]["L1.5"]["elements"], 40 * k2)
                self.assertEqual(report["layers"]["L1.5"]["zero_elements"], 0)
                self.assertEqual(report["aggregate"]["activation_sparsity"],
                                 1 / (k1 + k2))

    def test_only_l2_features_activations_are_measured(self):
        net = _MeasuredHexHierNet(seed=7)
        lat = np.ones((1, 5, 4, 3))
        net._h1(lat)
        net.l1(lat)
        values = np.array([-2.0, 0.0, 1.0])
        np.testing.assert_array_equal(net._lrelu(values, a=0.2),
                                      HexHierNet._lrelu(values, a=0.2))
        self.assertEqual(net.activity.report()["aggregate"]["elements"], 0)
        net.l2_features(lat)
        report = net.activity.report()
        net.l1(lat)
        self.assertEqual(net.activity.report(), report)

    def test_context_is_restored_after_parent_or_recorder_failure(self):
        lat = np.ones((1, 5, 4, 3))
        net = _MeasuredHexHierNet(seed=7)

        def fail_after_h1(model, values):
            model._h1(values)
            self.assertEqual(model._activation_layer, "L1.5")
            raise RuntimeError("after L1")

        def fail_in_h1(model, values):
            self.assertEqual(model._activation_layer, "L1")
            raise RuntimeError("during L1")

        for method, replacement in (("l2_features", fail_after_h1),
                                    ("_h1", fail_in_h1)):
            with self.subTest(method=method):
                with patch.object(HexHierNet, method, replacement):
                    with self.assertRaises(RuntimeError):
                        net.l2_features(lat)
                self.assertIsNone(net._activation_layer)
        with patch.object(net.activity, "record", side_effect=RuntimeError("recorder")):
            with self.assertRaises(RuntimeError):
                net.l2_features(lat)
        self.assertIsNone(net._activation_layer)
        # A subsequent real forward still measures exactly one activation per layer.
        net.activity = ActivationCounts()
        net.l2_features(lat)
        for counts in net.activity.report()["layers"].values():
            self.assertEqual(counts["calls"], 1)

    def test_nested_forward_restores_the_outer_measurement_context(self):
        net = _MeasuredHexHierNet(seed=7)
        lat = np.ones((1, 5, 4, 3))
        original = HexHierNet.l2_features
        entered = False

        def nested_forward(model, values):
            nonlocal entered
            if not entered:
                entered = True
                model.l2_features(values)
                self.assertEqual(model._activation_layer, "L1.5")
            return original(model, values)

        with patch.object(HexHierNet, "l2_features", nested_forward):
            net.l2_features(lat)
        self.assertIsNone(net._activation_layer)
        for counts in net.activity.report()["layers"].values():
            self.assertEqual(counts["calls"], 2)
            self.assertEqual(counts["elements"], 2 * 5 * 4 * 6)

    def test_caught_h1_failure_restores_l15_context(self):
        net = _MeasuredHexHierNet(seed=7)

        def continue_after_failure(model, values):
            try:
                model._h1(values)
            except RuntimeError:
                pass
            self.assertEqual(model._activation_layer, "L1.5")
            return model._lrelu(np.array([0.0, 1.0, -1.0]))

        with patch.object(HexHierNet, "l2_features", continue_after_failure):
            with patch.object(HexHierNet, "_h1", side_effect=RuntimeError("L1")):
                net.l2_features(np.ones((1, 5, 4, 3)))
        self.assertIsNone(net._activation_layer)
        layers = net.activity.report()["layers"]
        self.assertEqual(set(layers), {"L1.5"})
        self.assertEqual(layers["L1.5"]["elements"], 3)
        self.assertEqual(layers["L1.5"]["zero_elements"], 1)

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
        for stacked in (False, True):
            with self.subTest(stacked=stacked):
                args = [sys.executable, "-X", "utf8", str(script), "--samples", "2",
                        "--cells-across", "4"] + ([] if stacked else ["--unstacked"])
                with tempfile.TemporaryDirectory() as directory:
                    result = subprocess.run(args, cwd=directory, capture_output=True,
                                            text=True, encoding="utf-8", timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                report = json.loads(result.stdout)
                self.assertEqual(report, measure(samples=2, cells_across=4,
                                                  seed=7, stacked=stacked))
                self.assertEqual(report["aggregate"]["status"], "measured")
                self.assertEqual(set(report["layers"]),
                                 {"L1", "L1.5"} if stacked else {"L1"})
                json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
