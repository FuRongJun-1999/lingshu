"""Diagnose failed independent reconstruction truth metrics without refitting output.

The primary center-Umeyama evaluation remains unchanged. Orientation-gauge
results are competing explanations for a short camera trajectory, not an
alternative acceptance score. Silhouette masks use the same independently
validated Blender camera convention as the truth-asset checker.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def _load_evaluator():
    path = Path(__file__).resolve().parents[2] / "bench/reconstruction_truth.py"
    spec = importlib.util.spec_from_file_location("truth_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _proper_rotation(matrices):
    u, _, vt = np.linalg.svd(np.asarray(matrices).sum(axis=0))
    return u @ np.diag((1., 1., np.linalg.det(u @ vt))) @ vt


def _angle(matrix):
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(matrix).as_rotvec())))


def evaluate(scene, run, threads=2, output_images=None):
    import open3d as o3d
    from PIL import Image

    evaluator = _load_evaluator()
    scene, run = Path(scene), Path(run)
    truth = evaluator._read_json(scene / "truth/cameras.json")
    cameras = {camera["image"]: camera for camera in truth["cameras"]}
    poses = evaluator._read_json(run / "02_sparse/poses.json")
    report = evaluator._read_json(run / "report.json")
    manifest = evaluator._read_json(run / "00_input/input_manifest.json")
    source_names = {record["name"]: Path(record["source"]).name for record in manifest}
    names = sorted([name for name in poses if source_names.get(name) in cameras],
                   key=lambda name: poses[name]["source_index"])
    predicted = np.asarray([poses[name]["cam_from_world"] for name in names])
    expected = np.asarray([cameras[source_names[name]]["cam_from_world"] for name in names])
    similarity = evaluator.fit_similarity(predicted, expected)
    source_centers = np.asarray([evaluator.camera_center(pose) for pose in predicted])
    truth_centers = np.asarray([evaluator.camera_center(pose) for pose in expected])
    source_zero = source_centers - source_centers.mean(axis=0)
    truth_zero = truth_centers - truth_centers.mean(axis=0)
    singular_source = np.linalg.svd(source_zero, compute_uv=False)
    singular_truth = np.linalg.svd(truth_zero, compute_uv=False)
    source_cameras = evaluator._source_camera_map(run, report)
    align_rotation = np.asarray(similarity["rotation"])
    residual_maps = [g[:, :3].T @ p[:, :3] @ align_rotation.T for p, g in zip(predicted, expected)]
    common_map = _proper_rotation(residual_maps)
    rotation_rows = []
    for name, residual in zip(names, residual_maps):
        rotation_rows.append({"image": source_names[name], "absolute_degrees": _angle(residual),
                              "world_rotation_vector_degrees": np.degrees(Rotation.from_matrix(residual).as_rotvec()).tolist(),
                              "after_removing_common_rotation_degrees": _angle(residual @ common_map.T)})
    pairwise = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            p_relative = predicted[i, :, :3] @ predicted[j, :, :3].T
            g_relative = expected[i, :, :3] @ expected[j, :, :3].T
            pairwise.append({"a": source_names[names[i]], "b": source_names[names[j]],
                             "rotation_error_degrees": _angle(p_relative @ g_relative.T)})
    orientation_rotation = _proper_rotation([g[:, :3].T @ p[:, :3] for p, g in zip(predicted, expected)])
    orientation_scale = float(np.sum((source_zero @ orientation_rotation.T) * truth_zero)
                              / np.sum(source_zero ** 2))
    orientation_translation = truth_centers.mean(axis=0) - orientation_scale * orientation_rotation @ source_centers.mean(axis=0)
    orientation_similarity = {"scale": orientation_scale, "rotation": orientation_rotation.tolist(),
                              "translation": orientation_translation.tolist()}
    mesh = o3d.io.read_triangle_mesh(str(run / "05_fused/mesh.ply"))
    output_scale = report["stages"]["alignment"]["external_metric_conversion"]["meters_per_sfm_unit"] or 1.
    mesh_vertices = evaluator.transform_points(np.asarray(mesh.vertices) / output_scale, similarity)
    mesh_triangles = np.asarray(mesh.triangles)
    with np.load(scene / "truth/mesh.npz") as data:
        truth_vertices = data["vertices"]
    minimum, maximum = truth_vertices.min(axis=0), truth_vertices.max(axis=0)
    margin = float(np.linalg.norm(maximum - minimum) * .02)
    outside = np.any((mesh_vertices < minimum - margin) | (mesh_vertices > maximum + margin), axis=1)
    silhouettes, depth_background = [], []
    depth_upper_bounds = []
    for name in names:
        depth_path = run / "04_alignment/depth_aligned" / f"{Path(name).stem}.npz"
        if depth_path.is_file():
            with np.load(depth_path) as data:
                values = data["depth"]
                values = values[np.isfinite(values) & (values > 0)]
                if values.size:
                    depth_upper_bounds.append(float(np.quantile(values, .995)))
    max_depth_units = max(float(np.median(depth_upper_bounds)) * 1.1, 1e-4)
    if output_images is not None:
        output_images = Path(output_images)
        output_images.mkdir(parents=True, exist_ok=True)
    for name in names:
        camera = cameras[source_names[name]]
        gt_depth, gt_mask = evaluator._depth_truth(scene, camera)
        mesh_mask = evaluator._raycast_mask(mesh_vertices, mesh_triangles, camera, threads)
        true_positive = int(np.count_nonzero(mesh_mask & gt_mask))
        false_positive = int(np.count_nonzero(mesh_mask & ~gt_mask))
        false_negative = int(np.count_nonzero(~mesh_mask & gt_mask))
        row = {"image": source_names[name], "tp": true_positive, "fp": false_positive, "fn": false_negative,
               "precision": true_positive / (true_positive + false_positive),
               "recall": true_positive / (true_positive + false_negative),
               "iou": evaluator.mask_iou(mesh_mask, gt_mask)}
        if output_images is not None:
            visualization = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
            visualization[gt_mask & mesh_mask] = [50, 160, 50]
            visualization[~gt_mask & mesh_mask] = [240, 40, 30]
            visualization[gt_mask & ~mesh_mask] = [40, 70, 240]
            path = output_images / camera["image"]
            Image.fromarray(visualization).save(path)
            row["tp_green_fp_red_fn_blue_image"] = str(path.resolve())
        silhouettes.append(row)
        path = run / "04_alignment/depth_aligned" / f"{Path(name).stem}.npz"
        if not path.is_file():
            continue
        with np.load(path) as data:
            depth, confidence = data["depth"], data["confidence"]
            input_mask = data["input_valid_mask"] if "input_valid_mask" in data else None
        ys, xs = np.mgrid[:depth.shape[0], :depth.shape[1]]
        pixels = np.column_stack((xs.ravel() + .5, ys.ravel() + .5))
        mapped = evaluator.source_pixels(pixels, poses[name], source_cameras[name], camera, depth.shape)
        finite = np.isfinite(mapped).all(axis=1)
        indices = np.zeros((len(mapped), 2), dtype=np.int64)
        indices[finite] = np.floor(mapped[finite]).astype(np.int64)
        inside = (finite & (indices[:, 0] >= 0) & (indices[:, 0] < camera["width"])
                  & (indices[:, 1] >= 0) & (indices[:, 1] < camera["height"]))
        foreground = np.zeros(len(mapped), dtype=bool)
        foreground[inside] = gt_mask[indices[inside, 1], indices[inside, 0]]
        positive, accepted, cutoff = evaluator.depth_acceptance(
            depth, confidence, report["stages"]["fusion"]["confidence_percentile"], input_mask)
        positive = positive.ravel()
        fused_candidate = accepted.ravel() & (depth.ravel() < max_depth_units)
        background = inside & ~foreground
        depth_background.append({"image": camera["image"], "source_background_samples": int(background.sum()),
                                 "positive_depth_background_samples": int((background & positive).sum()),
                                 "accepted_background_samples": int((background & fused_candidate).sum()),
                                 "accepted_foreground_samples": int((inside & foreground & fused_candidate).sum()),
                                 "input_valid_mask_supplied": input_mask is not None,
                                 "confidence_cutoff": float(cutoff)})
    return {"primary_center_umeyama_unchanged": similarity,
            "center_singular_values": {"predicted": singular_source.tolist(), "truth_m": singular_truth.tolist(),
                                       "predicted_second_over_first": float(singular_source[1] / singular_source[0]),
                                       "truth_second_over_first": float(singular_truth[1] / singular_truth[0]),
                                       "predicted_third_over_first": float(singular_source[2] / singular_source[0]),
                                       "truth_third_over_first": float(singular_truth[2] / singular_truth[0])},
            "primary_rotation_per_view": rotation_rows,
            "common_world_rotation_degrees": _angle(common_map),
            "common_world_rotation_vector_degrees": np.degrees(Rotation.from_matrix(common_map).as_rotvec()).tolist(),
            "gauge_invariant_pairwise_rotation": {"summary_degrees": evaluator._summary([row["rotation_error_degrees"] for row in pairwise]),
                                                   "pairs": pairwise},
            "orientation_gauge_diagnostic_only": {"transform": orientation_similarity,
                                                  "pose": evaluator.pose_errors(predicted, expected, orientation_similarity),
                                                  "used_for_acceptance": False},
            "silhouette": silhouettes, "depth_background": depth_background,
            "depth_truncation_diagnostic": {"max_depth_units": max_depth_units,
                                             "source": "replayed default 1.1 * median of per-view positive-depth p99.5; original report omits this setting"},
            "mesh_extent": {"truth_min": minimum.tolist(), "truth_max": maximum.tolist(),
                            "predicted_min": mesh_vertices.min(axis=0).tolist(),
                            "predicted_max": mesh_vertices.max(axis=0).tolist(),
                            "gt_box_margin_m": margin, "vertices": len(mesh_vertices),
                            "vertices_outside_gt_box_plus_margin": int(outside.sum()),
                            "outside_fraction": float(outside.mean())}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    record = evaluate(args.scene, args.run, args.threads, args.output.parent / (args.output.stem + "-masks"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(str(args.output.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
