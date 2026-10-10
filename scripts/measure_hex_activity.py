#!/usr/bin/env python3
"""Measure exact-zero activity at HexHierNet's L1/L1.5 LeakyReLU outputs.

Independent measurement only: no training, threshold changes or new dependencies.
The measured scope is explicit; this is not a full NeuroBench benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

if __package__ in (None, ""):  # Allow direct invocation from a source checkout.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lingshu.nn.hex_cnn import image_to_grid
from lingshu.nn.hex_hier import HexHierNet, make_scene_dataset
from lingshu.nn.hex_train import normalize_lattices


class ActivationCounts:
    """Accumulate integer counts, weighting batches/layers by their element counts."""

    def __init__(self):
        self.layers = {}

    def record(self, layer: str, values: np.ndarray) -> None:
        values = np.asarray(values)
        if values.dtype.kind not in "biuf":
            raise ValueError("activations must be real numeric arrays")
        counts = self.layers.setdefault(
            layer, dict(calls=0, elements=0, zero_elements=0, nonfinite_elements=0))
        counts["calls"] += 1
        counts["elements"] += int(values.size)
        counts["zero_elements"] += int(np.count_nonzero(values == 0))
        counts["nonfinite_elements"] += int(np.count_nonzero(~np.isfinite(values)))

    @staticmethod
    def _summary(counts):
        result = dict(counts)
        valid = counts["elements"] > 0 and counts["nonfinite_elements"] == 0
        result["status"] = ("invalid" if counts["nonfinite_elements"] else
                            "measured" if counts["elements"] else "not_measured")
        result["activation_sparsity"] = (
            counts["zero_elements"] / counts["elements"] if valid else None)
        return result

    def report(self):
        layers = {name: self._summary(counts)
                  for name, counts in sorted(self.layers.items())}
        total = {key: sum(layer[key] for layer in self.layers.values())
                 for key in ("calls", "elements", "zero_elements", "nonfinite_elements")}
        return {"layers": layers, "aggregate": self._summary(total)}


class _MeasuredHexHierNet(HexHierNet):
    """Record the actual activation call results; delegate arithmetic to the model."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.activity = ActivationCounts()
        self._activation_calls = 0

    def _lrelu(self, values, a=0.05):
        result = super()._lrelu(values, a)
        # l2_features calls K L1 kernels, then K2 L1.5 kernels (if stacked).
        phase = self._activation_calls % (self.K + self.K2)
        layer = "L1" if phase < self.K else "L1.5"
        self.activity.record(layer, result)
        self._activation_calls += 1
        return result


def measure(samples: int = 8, cells_across: int = 16, seed: int = 7,
            stacked: bool = True):
    """Measure a seeded, untrained model on the existing synthetic scene generator."""
    if samples < 1 or cells_across < 1:
        raise ValueError("samples and cells_across must be positive")
    images, _ = make_scene_dataset(samples, size=48, seed=seed)
    lattices = normalize_lattices(np.stack([
        image_to_grid(image, cells_across=cells_across)[0] / 255.0
        for image in images]))
    net = _MeasuredHexHierNet(seed=seed, stacked=stacked)
    before = net.get_vec().copy()
    net.l2_features(lattices)
    if not np.array_equal(before, net.get_vec()):
        raise RuntimeError("measurement changed model parameters")
    return {
        "schema": "hex-activity-0.1",
        "definition": "exact zero activations / total measured activations",
        "scope": ["L1 post-LeakyReLU"] + (["L1.5 post-LeakyReLU"] if stacked else []),
        "conditions": {
            "model": "HexHierNet", "trained": False, "seed": seed,
            "stacked": stacked, "samples": samples, "scene_size": 48,
            "cells_across": cells_across, "lattice_shape": list(lattices.shape),
            "input_sha256": hashlib.sha256(images.tobytes()).hexdigest(),
            "parameters_sha256": hashlib.sha256(
                before.astype("<f8").tobytes()).hexdigest(),
        },
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        **net.activity.report(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--cells-across", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--unstacked", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = measure(args.samples, args.cells_across, args.seed, not args.unstacked)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0 if report["aggregate"]["status"] == "measured" else 1


if __name__ == "__main__":
    raise SystemExit(main())
