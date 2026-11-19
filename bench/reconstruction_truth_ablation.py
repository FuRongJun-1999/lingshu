"""Replay alpha-mask alignment and confidence fusion from saved neural depth.

This deliberately does not report end-to-end runtime or peak memory: SfM and
neural inference are reused. The production alpha-mask/undistortion contract
is replayed from input RGBA, never from independent truth depth or masks.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np


def _load_evaluator():
    path = Path(__file__).with_name("reconstruction_truth.py")
    spec = importlib.util.spec_from_file_location("truth_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _link_or_copy(source, destination):
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--producer-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--percentiles", type=float, nargs="+", default=[0, 5, 25])
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    if args.threads < 1 or any(not 0 <= value < 100 for value in args.percentiles):
        parser.error("positive threads and confidence percentiles in [0,100) are required")
    if args.run_root.exists():
        parser.error("--run-root must be a new directory, preserving earlier experiments")
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    sys.path.insert(0, str(args.producer_root.resolve()))
    import cv2
    from lingshu.world.reconstruct.alignment import estimate_alignment, write_aligned_depths
    from lingshu.world.reconstruct.colmap import choose_model
    from lingshu.world.reconstruct.fusion import fuse_scene
    from lingshu.world.reconstruct.images import normalize_images, undistort_valid_masks
    from lingshu.world.reconstruct.pipeline import estimate_depth_truncation

    cv2.setNumThreads(args.threads)
    evaluator = _load_evaluator()
    source_report = evaluator._read_json(args.run / "report.json")
    if source_report["absolute_metric_scale"]:
        parser.error("this replay currently expects an unscaled SfM-relative source run")
    poses = evaluator._read_json(args.run / "02_sparse/poses.json")
    original_manifest = evaluator._read_json(args.run / "00_input/input_manifest.json")
    tracks = [json.loads(line) for line in (args.run / "02_sparse/tracks.jsonl").read_text(encoding="utf-8").splitlines()]
    base = args.run_root / "prepared-alpha-inputs"
    base.mkdir(parents=True)
    manifest = normalize_images(args.scene / "images", base / "00_input/images",
                                max(max(record["width"], record["height"]) for record in original_manifest))
    for name in ("01_sfm", "02_sparse", "02_undistorted"):
        shutil.copytree(args.run / name, base / name, copy_function=_link_or_copy,
                        ignore=shutil.ignore_patterns("masks"))
    masks = undistort_valid_masks(manifest, base / "00_input", base / source_report["stages"]["sfm"]["model"],
                                  choose_model(base / "02_undistorted/sparse"), base / "02_undistorted/masks")
    (base / "03_depth").mkdir()
    for name in poses:
        source = args.run / "03_depth" / f"{Path(name).stem}.npz"
        with np.load(source) as data:
            record = {key: data[key] for key in data.files}
        if name in masks:
            mask = cv2.imread(str(base / "02_undistorted/masks" / masks[name]), cv2.IMREAD_GRAYSCALE)
            height, width = record["depth"].shape
            record["input_valid_mask"] = (cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)
        np.savez(base / "03_depth" / source.name, **record)
    result = {"schema": "lingshu.truth-confidence-replay/1", "scene": str(args.scene.resolve()),
              "source_run": str(args.run.resolve()), "producer_root": str(args.producer_root.resolve()),
              "producer_source_sha256": evaluator._fingerprint(args.producer_root),
              "scope": "saved SfM and neural depth, RGBA alpha-mask preparation, alignment and CPU TSDF replay",
              "not_end_to_end_runtime_or_memory": True, "configurations": []}
    for percentile in args.percentiles:
        run = args.run_root / f"confidence-{percentile:g}"
        run.mkdir()
        for name in ("00_input", "01_sfm", "02_sparse", "02_undistorted", "03_depth"):
            shutil.copytree(base / name, run / name, copy_function=_link_or_copy)
        report = copy.deepcopy(source_report)
        started = time.perf_counter()
        alignment = estimate_alignment(tracks, run / "03_depth", poses, percentile)
        conversion = write_aligned_depths(run / "03_depth", run / "04_alignment/depth_aligned", poses, alignment)
        alignment["external_metric_conversion"] = conversion
        evaluator._write_json(run / "04_alignment/scale_report.json", alignment)
        fusion = fuse_scene(poses, run / "04_alignment/depth_aligned", run / "02_undistorted/images",
                            run / "05_fused", 1., source_report["stages"]["fusion"]["voxel_size"],
                            percentile, estimate_depth_truncation(run / "04_alignment/depth_aligned", poses),
                            report["units"])
        replay_seconds = time.perf_counter() - started
        report["stages"]["alignment"], report["stages"]["fusion"] = alignment, fusion
        report["timings_seconds"]["total"] = None
        report["resource_usage"]["peak_process_tree_rss_mb"] = None
        report["saved_depth_replay"] = {"seconds_alignment_and_fusion": replay_seconds,
                                        "confidence_percentile": percentile, "inference_rerun": False,
                                        "input_domain_source": "RGBA alpha>0 via production camera undistortion"}
        evaluator._write_json(run / "report.json", report)
        measured = evaluator.evaluate_scene(args.scene, run, threads=args.threads)
        row = {"confidence_percentile": percentile, "run": str(run.resolve()),
               "alignment_and_fusion_replay_seconds": replay_seconds, **measured}
        result["configurations"].append(row)
        evaluator._write_json(args.output, result)
        print(json.dumps({"confidence_percentile": percentile,
                          "dense_mean_abs_rel": measured["dense_depth_abs_rel"]["mean"],
                          "gt_coverage": measured["dense_coverage"]["fused_confidence_fraction"],
                          "silhouette_iou": measured["silhouette_iou"]["mean"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
