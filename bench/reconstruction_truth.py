"""Independent geometry truth evaluation and sequential CPU reconstruction runs.

Ground truth is supplied by ``tools/reconstruct/generate_truth.py``. One scene
similarity aligns the SfM gauge to truth; no per-view depth fit is allowed.
Absolute metric scale is not inferred by that evaluation-only alignment.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.spatial import cKDTree


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    pending.replace(path)


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"samples": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {"samples": int(values.size), "mean": float(values.mean()),
            "median": float(np.median(values)), "p95": float(np.quantile(values, .95)),
            "max": float(values.max())}


def camera_center(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    return -matrix[:, :3].T @ matrix[:, 3]


def fit_similarity(predicted, truth):
    """Fit a single proper Sim3, using orientations only for collinear centers.

    Both inputs are ordered lists of world-to-camera 3x4 matrices. The returned
    transform maps *raw SfM* world coordinates into truth world meters. Using
    truth to set this gauge does not make metric scale observable to the model.
    """
    if len(predicted) != len(truth) or len(truth) < 3:
        raise ValueError("at least three corresponding camera poses are required")
    predicted = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if predicted.shape[1:] != (3, 4) or truth.shape[1:] != (3, 4):
        raise ValueError("camera poses must be 3x4 matrices")
    if not np.isfinite(predicted).all() or not np.isfinite(truth).all():
        raise ValueError("camera poses must be finite")
    source = np.asarray([camera_center(p) for p in predicted])
    target = np.asarray([camera_center(p) for p in truth])
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    source_zero, target_zero = source - source_mean, target - target_mean
    variance = float(np.sum(source_zero ** 2))
    if variance <= 1e-12 or float(np.sum(target_zero ** 2)) <= 1e-12:
        raise ValueError("camera baseline is degenerate; scale is unobservable")
    singular = np.linalg.svd(source_zero, compute_uv=False)
    rank = int(np.sum(singular > max(singular[0] * 1e-6, 1e-10)))
    if rank >= 2:
        u, _, vt = np.linalg.svd(target_zero.T @ source_zero)
        correction = np.diag((1.0, 1.0, np.linalg.det(u @ vt)))
        rotation = u @ correction @ vt
        method = "all_camera_centers_umeyama"
    else:
        orientation_maps = np.asarray([g[:, :3].T @ p[:, :3] for p, g in zip(predicted, truth)])
        u, _, vt = np.linalg.svd(orientation_maps.sum(axis=0))
        rotation = u @ np.diag((1.0, 1.0, np.linalg.det(u @ vt))) @ vt
        method = "collinear_centers_with_mean_orientation_gauge"
    mapped_zero = source_zero @ rotation.T
    scale = float(np.sum(mapped_zero * target_zero) / variance)
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError("camera similarity has no positive finite scale")
    translation = target_mean - scale * rotation @ source_mean
    return {"scale": scale, "rotation": rotation.tolist(), "translation": translation.tolist(),
            "method": method, "camera_center_rank": rank,
            "scale_source": "ground truth used by evaluator only",
            "absolute_scale_observable": False}


def transform_points(points, similarity):
    points = np.asarray(points, dtype=np.float64)
    return (similarity["scale"] * points @ np.asarray(similarity["rotation"]).T
            + np.asarray(similarity["translation"]))


def pose_errors(predicted, truth, similarity):
    rotation = np.asarray(similarity["rotation"])
    rotations, translations = [], []
    for p, g in zip(np.asarray(predicted), np.asarray(truth)):
        aligned_camera_rotation = p[:, :3] @ rotation.T
        delta = aligned_camera_rotation @ g[:, :3].T
        cosine = float(np.clip((np.trace(delta) - 1) / 2, -1, 1))
        rotations.append(math.degrees(math.acos(cosine)))
        aligned_center = transform_points(camera_center(p), similarity)
        translations.append(float(np.linalg.norm(aligned_center - camera_center(g))))
    return {"rotation_degrees": _summary(rotations), "translation_m_after_sim3": _summary(translations)}


def depth_statistics(predicted_depth, truth_depth, valid):
    predicted_depth = np.asarray(predicted_depth, dtype=np.float64)
    truth_depth = np.asarray(truth_depth, dtype=np.float64)
    valid = (np.asarray(valid, dtype=bool) & np.isfinite(predicted_depth) & (predicted_depth > 0)
             & np.isfinite(truth_depth) & (truth_depth > 0))
    errors = np.abs(predicted_depth[valid] - truth_depth[valid]) / truth_depth[valid]
    return _summary(errors)


def mask_iou(predicted, truth):
    predicted, truth = np.asarray(predicted, dtype=bool), np.asarray(truth, dtype=bool)
    union = int(np.count_nonzero(predicted | truth))
    return float(np.count_nonzero(predicted & truth) / union) if union else 1.0


def sample_surface(vertices, triangles, count=12000, seed=0):
    vertices, triangles = np.asarray(vertices), np.asarray(triangles, dtype=np.int64)
    if not len(triangles):
        raise ValueError("mesh has no triangles")
    corners = vertices[triangles]
    areas = np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)
    if not np.isfinite(areas).all() or float(areas.sum()) <= 0:
        raise ValueError("mesh has no finite surface area")
    rng = np.random.default_rng(seed)
    picked = corners[rng.choice(len(triangles), count, p=areas / areas.sum())]
    uv = rng.random((count, 2))
    fold = uv.sum(axis=1) > 1
    uv[fold] = 1 - uv[fold]
    return picked[:, 0] + uv[:, :1] * (picked[:, 1] - picked[:, 0]) + uv[:, 1:] * (picked[:, 2] - picked[:, 0])


def surface_statistics(predicted_points, truth_points, tolerance):
    predicted_points, truth_points = np.asarray(predicted_points), np.asarray(truth_points)
    if not len(predicted_points) or not len(truth_points):
        raise ValueError("surface statistics require both point samples")
    prediction_to_truth = cKDTree(truth_points).query(predicted_points, workers=1)[0]
    truth_to_prediction = cKDTree(predicted_points).query(truth_points, workers=1)[0]
    return {"chamfer_mean_euclidean_m": float((prediction_to_truth.mean() + truth_to_prediction.mean()) / 2),
            "prediction_to_truth_m": _summary(prediction_to_truth),
            "truth_to_prediction_m": _summary(truth_to_prediction),
            "tolerance_m": float(tolerance),
            "coverage": float(np.mean(truth_to_prediction <= tolerance)),
            "precision": float(np.mean(prediction_to_truth <= tolerance))}


def _source_camera_map(run, report):
    import pycolmap
    model_path = run / Path(report["stages"]["sfm"]["model"])
    model = pycolmap.Reconstruction(str(model_path))
    return {image.name: model.camera(image.camera_id) for image in model.images.values() if image.has_pose}


def source_pixels(pixels, pose, source_camera, truth_camera, depth_shape=None):
    """Map rectified/depth pixel centers back through the original camera model."""
    pixels = np.asarray(pixels, dtype=np.float64)
    k = np.asarray(pose["intrinsics"], dtype=np.float64)
    if depth_shape is not None:
        k = np.diag((depth_shape[1] / pose["width"], depth_shape[0] / pose["height"], 1.0)) @ k
    rays = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(k).T
    native = np.asarray(source_camera.img_from_cam(rays / rays[:, 2:3]), dtype=np.float64)
    return native * np.array((truth_camera["width"] / source_camera.width,
                              truth_camera["height"] / source_camera.height))


def _truth_at_pixels(pixels, depth, mask):
    finite = np.isfinite(pixels).all(axis=1)
    indices = np.zeros((len(pixels), 2), dtype=np.int64)
    indices[finite] = np.floor(pixels[finite]).astype(np.int64)
    valid = (finite & (indices[:, 0] >= 0) & (indices[:, 0] < depth.shape[1])
             & (indices[:, 1] >= 0) & (indices[:, 1] < depth.shape[0]))
    sampled = np.full(len(pixels), np.nan, dtype=np.float64)
    chosen = np.flatnonzero(valid)
    sampled[chosen] = depth[indices[chosen, 1], indices[chosen, 0]]
    valid[chosen] &= mask[indices[chosen, 1], indices[chosen, 0]]
    valid &= np.isfinite(sampled) & (sampled > 0)
    return sampled, valid


def covered_truth_pixels(pixels, valid, truth_mask):
    """Count distinct covered GT pixels, never using predicted pixels as denominator."""
    pixels, valid = np.asarray(pixels), np.asarray(valid, dtype=bool)
    selected = pixels[valid]
    selected = selected[np.isfinite(selected).all(axis=1)]
    if not len(selected):
        return 0
    indices = np.floor(selected).astype(np.int64)
    h, w = truth_mask.shape
    inside = ((indices[:, 0] >= 0) & (indices[:, 0] < w)
              & (indices[:, 1] >= 0) & (indices[:, 1] < h))
    indices = indices[inside]
    if not len(indices):
        return 0
    indices = indices[truth_mask[indices[:, 1], indices[:, 0]]]
    return int(len(np.unique(indices[:, 1] * w + indices[:, 0])))


def depth_acceptance(depth, confidence, confidence_percentile, input_valid_mask=None):
    """Mirror the production input domain and its confidence percentile cutoff."""
    depth, confidence = np.asarray(depth), np.asarray(confidence)
    if confidence.shape != depth.shape:
        raise ValueError("depth and confidence shapes differ")
    domain = np.ones(depth.shape, dtype=bool)
    if input_valid_mask is not None:
        mask = np.asarray(input_valid_mask)
        if mask.shape != depth.shape or mask.dtype.kind not in "biu" or np.any((mask != 0) & (mask != 1)):
            raise ValueError("input_valid_mask must be binary bool/integer and match depth")
        domain &= mask.astype(bool)
    confidence_domain = domain & np.isfinite(confidence)
    cutoff = float(np.percentile(confidence[confidence_domain], confidence_percentile)) if confidence_domain.any() else float("inf")
    positive = domain & np.isfinite(depth) & (depth > 0)
    accepted = positive & np.isfinite(confidence) & (confidence >= cutoff)
    return positive, accepted, cutoff


def backproject_pixels(pixels, depth, camera):
    k, pose = np.asarray(camera["intrinsics"]), np.asarray(camera["cam_from_world"])
    rays = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(k).T
    camera_points = rays * np.asarray(depth)[:, None]
    return (camera_points - pose[:, 3]) @ pose[:, :3]


def _depth_truth(scene, truth_camera):
    with np.load(scene / truth_camera["depth"]) as record:
        return record["depth_z"].astype(np.float64), record["mask"].astype(bool)


def _raycast_mask(vertices, triangles, camera, threads):
    import open3d as o3d
    raycaster = o3d.t.geometry.RaycastingScene(nthreads=threads)
    raycaster.add_triangles(o3d.core.Tensor(np.asarray(vertices, dtype=np.float32)),
                           o3d.core.Tensor(np.asarray(triangles, dtype=np.uint32)))
    h, w = camera["height"], camera["width"]
    ys, xs = np.mgrid[:h, :w]
    pixels = np.column_stack((xs.ravel() + .5, ys.ravel() + .5))
    pose, k = np.asarray(camera["cam_from_world"]), np.asarray(camera["intrinsics"])
    directions = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(k).T @ pose[:, :3]
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    origins = np.broadcast_to(camera_center(pose), directions.shape)
    rays = np.column_stack((origins, directions)).astype(np.float32).reshape(h, w, 6)
    return np.isfinite(raycaster.cast_rays(o3d.core.Tensor(rays), nthreads=threads)["t_hit"].numpy())


def raycast_truth_points(pixels, camera, raycaster, threads=2):
    """Intersect continuous source pixels with independent mesh geometry."""
    import open3d as o3d
    pixels = np.asarray(pixels, dtype=np.float64)
    pose, k = np.asarray(camera["cam_from_world"]), np.asarray(camera["intrinsics"])
    directions = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(k).T @ pose[:, :3]
    origins = np.broadcast_to(camera_center(pose), directions.shape)
    rays = np.column_stack((origins, directions)).astype(np.float32)
    depth = raycaster.cast_rays(o3d.core.Tensor(rays), nthreads=threads)["t_hit"].numpy().astype(np.float64)
    valid = np.isfinite(depth) & (depth > 0)
    points = np.full((len(pixels), 3), np.nan, dtype=np.float64)
    points[valid] = backproject_pixels(pixels[valid], depth[valid], camera)
    return points, depth, valid


def evaluate_scene(scene, run, *, threads=2, stride=4):
    """Evaluate one saved reconstruction without changing its fitted geometry."""
    import open3d as o3d
    scene, run = Path(scene), Path(run)
    truth = _read_json(scene / "truth/cameras.json")
    truth_cameras = {record["image"]: record for record in truth["cameras"]}
    report = _read_json(run / "report.json")
    poses = _read_json(run / "02_sparse/poses.json")
    manifest = _read_json(run / "00_input/input_manifest.json")
    source_names = {record["name"]: Path(record["source"]).name for record in manifest}
    names = [name for name in poses if source_names.get(name) in truth_cameras]
    names.sort(key=lambda name: poses[name]["source_index"])
    predicted_poses = [poses[name]["cam_from_world"] for name in names]
    expected_poses = [truth_cameras[source_names[name]]["cam_from_world"] for name in names]
    similarity = fit_similarity(predicted_poses, expected_poses)
    source_cameras = _source_camera_map(run, report)
    scale_out = report["stages"]["alignment"]["external_metric_conversion"]["meters_per_sfm_unit"] or 1.0
    rotation, translation, scene_scale = (np.asarray(similarity["rotation"]),
                                           np.asarray(similarity["translation"]), similarity["scale"])
    dense_errors, all_dense_errors, dense_rows, observed_surfaces = [], [], [], []
    covered_filtered, covered_positive = 0, 0
    truth_cache = {name: _depth_truth(scene, camera) for name, camera in truth_cameras.items()}
    with np.load(scene / "truth/mesh.npz") as data:
        truth_vertices, truth_triangles = data["vertices"], data["triangles"]
    truth_raycaster = o3d.t.geometry.RaycastingScene(nthreads=threads)
    truth_raycaster.add_triangles(o3d.core.Tensor(np.asarray(truth_vertices, dtype=np.float32)),
                                 o3d.core.Tensor(np.asarray(truth_triangles, dtype=np.uint32)))
    gt_foreground_pixels = sum(int(np.count_nonzero(mask & np.isfinite(depth) & (depth > 0)))
                               for depth, mask in truth_cache.values())
    for camera_name, camera in truth_cameras.items():
        truth_depth, truth_mask = truth_cache[camera_name]
        ys, xs = np.mgrid[stride // 2:camera["height"]:stride, stride // 2:camera["width"]:stride]
        valid = truth_mask[ys, xs] & np.isfinite(truth_depth[ys, xs]) & (truth_depth[ys, xs] > 0)
        pixels = np.column_stack((xs[valid] + .5, ys[valid] + .5))
        observed_surfaces.append(backproject_pixels(pixels, truth_depth[ys[valid], xs[valid]], camera))
    for name in names:
        pose, truth_camera = poses[name], truth_cameras[source_names[name]]
        depth_path = run / "04_alignment/depth_aligned" / f"{Path(name).stem}.npz"
        if not depth_path.is_file():
            dense_rows.append({"image": source_names[name], "status": "no_inferred_depth"})
            continue
        with np.load(depth_path) as record:
            depth, confidence = record["depth"], record["confidence"]
            input_mask = record["input_valid_mask"] if "input_valid_mask" in record else None
        ys, xs = np.mgrid[:depth.shape[0], :depth.shape[1]]
        pixels = np.column_stack((xs.ravel() + .5, ys.ravel() + .5))
        depth_values = depth[ys, xs].ravel().astype(np.float64) / scale_out
        mapped = source_pixels(pixels, pose, source_cameras[name], truth_camera, depth.shape)
        expected, valid = _truth_at_pixels(mapped, *truth_cache[source_names[name]])
        k = np.diag((depth.shape[1] / pose["width"], depth.shape[0] / pose["height"], 1)) @ np.asarray(pose["intrinsics"])
        rays = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(k).T
        p = np.asarray(pose["cam_from_world"])
        native_world = (rays * depth_values[:, None] - p[:, 3]) @ p[:, :3]
        aligned_world = scene_scale * native_world @ rotation.T + translation
        g = np.asarray(truth_camera["cam_from_world"])
        predicted_z = (aligned_world @ g[:, :3].T + g[:, 3])[:, 2]
        positive, accepted, cutoff = depth_acceptance(
            depth, confidence, report["stages"]["fusion"]["confidence_percentile"], input_mask)
        valid &= np.isfinite(predicted_z) & (predicted_z > 0) & positive.ravel()
        filtered = valid & accepted.ravel()
        errors = np.abs(predicted_z - expected) / expected
        positive_coverage = covered_truth_pixels(mapped, valid, truth_cache[source_names[name]][1])
        filtered_coverage = covered_truth_pixels(mapped, filtered, truth_cache[source_names[name]][1])
        covered_positive += positive_coverage
        covered_filtered += filtered_coverage
        all_dense_errors.extend(errors[valid].tolist())
        dense_errors.extend(errors[filtered].tolist())
        dense_rows.append({"image": source_names[name], "status": "measured",
                           "all_positive_abs_rel": _summary(errors[valid]),
                           "fused_confidence_abs_rel": _summary(errors[filtered]),
                           "sampled_pixels": len(pixels), "truth_foreground_samples": int(np.count_nonzero(np.isfinite(expected))),
                           "accepted_samples": int(filtered.sum()),
                           "confidence_cutoff_on_input_domain": cutoff if np.isfinite(cutoff) else None,
                           "input_valid_mask_supplied": input_mask is not None,
                           "gt_foreground_pixels": int(np.count_nonzero(truth_cache[source_names[name]][1])),
                           "gt_pixels_covered_positive": positive_coverage,
                           "gt_pixels_covered_fused_confidence": filtered_coverage})
    sparse_errors, sparse_relative, nearest_errors, nearest_relative = [], [], [], []
    observations_by_view = {}
    sparse_total_observations = 0
    with (run / "02_sparse/tracks.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            aligned = transform_points(record["xyz"], similarity)
            for observation in record["observations"]:
                sparse_total_observations += 1
                name = observation["image"]
                if name not in poses or name not in source_names or source_names[name] not in truth_cameras:
                    continue
                observations_by_view.setdefault(name, {"pixels": [], "points": []})
                observations_by_view[name]["pixels"].append(observation["pixel_xy"])
                observations_by_view[name]["points"].append(aligned)
    for name, observations in observations_by_view.items():
        camera = truth_cameras[source_names[name]]
        pixels = source_pixels(observations["pixels"], poses[name], source_cameras[name], camera)
        reconstructed = np.asarray(observations["points"])
        points, expected_depth, valid = raycast_truth_points(pixels, camera, truth_raycaster, threads)
        errors = np.linalg.norm(reconstructed[valid] - points[valid], axis=1)
        sparse_errors.extend(errors.tolist())
        sparse_relative.extend((errors / expected_depth[valid]).tolist())
        nearest_depth, nearest_valid = _truth_at_pixels(pixels, *truth_cache[source_names[name]])
        nearest_points = backproject_pixels(pixels[nearest_valid], nearest_depth[nearest_valid], camera)
        errors = np.linalg.norm(reconstructed[nearest_valid] - nearest_points, axis=1)
        nearest_errors.extend(errors.tolist())
        nearest_relative.extend((errors / nearest_depth[nearest_valid]).tolist())
    mesh = o3d.io.read_triangle_mesh(str(run / "05_fused/mesh.ply"))
    vertices = transform_points(np.asarray(mesh.vertices) / scale_out, similarity)
    triangles = np.asarray(mesh.triangles)
    predicted_samples = sample_surface(vertices, triangles)
    truth_samples = sample_surface(truth_vertices, truth_triangles)
    diagonal = float(np.linalg.norm(truth_vertices.max(axis=0) - truth_vertices.min(axis=0)))
    visible_samples = np.concatenate(observed_surfaces)
    if len(visible_samples) > 24000:
        visible_samples = visible_samples[np.linspace(0, len(visible_samples) - 1, 24000, dtype=int)]
    ious = []
    for camera_name, camera in truth_cameras.items():
        predicted_mask = _raycast_mask(vertices, triangles, camera, threads)
        ious.append({"image": camera_name, "iou": mask_iou(predicted_mask, truth_cache[camera_name][1])})
    result = {
        "status": "measured", "truth_source": truth["truth_source"],
        "input_views": len(truth_cameras), "registered_views": len(names),
        "unregistered_views": sorted(set(truth_cameras) - {source_names[name] for name in names}),
        "dense_views": sum(row["status"] == "measured" for row in dense_rows),
        "sim3": similarity, "pose": pose_errors(predicted_poses, expected_poses, similarity),
        "sparse_point_error_m": _summary(sparse_errors), "sparse_point_error_relative_to_depth": _summary(sparse_relative),
        "sparse_point_truth_method": "exact GT mesh raycast at continuous original source pixels, batched by view",
        "sparse_nearest_rendered_pixel_diagnostic": {"error_m": _summary(nearest_errors),
                                                     "error_relative_to_depth": _summary(nearest_relative),
                                                     "used_for_acceptance": False},
        "sparse_total_observations": sparse_total_observations,
        "dense_depth_abs_rel": _summary(dense_errors), "dense_depth_all_positive_abs_rel": _summary(all_dense_errors),
        "dense_coverage": {"gt_foreground_pixel_denominator_all_input_views": gt_foreground_pixels,
                           "covered_positive_pixels": covered_positive,
                           "covered_fused_confidence_pixels": covered_filtered,
                           "positive_fraction": covered_positive / gt_foreground_pixels if gt_foreground_pixels else 0.0,
                           "fused_confidence_fraction": covered_filtered / gt_foreground_pixels if gt_foreground_pixels else 0.0,
                           "missing_camera_or_depth_views_contribute_zero": True},
        "dense_views_metrics": dense_rows,
        "mesh_observed_surface": surface_statistics(predicted_samples, visible_samples, diagonal * .02),
        "mesh_full_surface": surface_statistics(predicted_samples, truth_samples, diagonal * .02),
        "silhouette_iou": {**_summary([row["iou"] for row in ious]), "views": ious},
        "reprojection_rmse_px": report["verification"]["reprojection"]["rmse_px"],
        "runtime_seconds": report["timings_seconds"]["total"],
        "peak_process_tree_rss_mib": report["resource_usage"]["peak_process_tree_rss_mb"],
        "method": {"depth_pixels": "original source pixels mapped from rectified predictions using estimated input distortion",
                   "dense_primary_domain": "positive depth within optional input_valid_mask and GT foreground passing production confidence percentile computed only within input_valid_mask",
                   "mesh_samples": 12000, "surface_tolerance": "2 percent of GT scene diagonal; diagnostic only",
                   "pixel_center_offset": .5, "independent_scale_anchor_supplied": report["absolute_metric_scale"]},
    }
    pose = result["pose"]
    checks = {
        "at_least_three_registered_views": len(names) >= 3,
        "at_least_three_dense_views": result["dense_views"] >= 3,
        "rotation_max_le_0_5_deg": pose["rotation_degrees"]["max"] <= .5,
        "aligned_translation_max_le_5_cm": pose["translation_m_after_sim3"]["max"] <= .05,
        "reprojection_rmse_le_2_px": result["reprojection_rmse_px"] is not None and result["reprojection_rmse_px"] <= 2,
        "sparse_mean_relative_error_le_2_percent": bool(sparse_relative) and result["sparse_point_error_relative_to_depth"]["mean"] <= .02,
        "dense_mean_abs_rel_le_5_percent": bool(dense_errors) and result["dense_depth_abs_rel"]["mean"] <= .05,
    }
    result["w3_sim3_aligned_checks"] = checks
    result["w3_sim3_aligned_pass"] = all(checks.values())
    result["r4_silhouette_mean_ge_0_85"] = result["silhouette_iou"]["mean"] >= .85
    return result


def _fingerprint(root):
    digest = hashlib.sha256()
    for path in sorted((root / "lingshu/world/reconstruct").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True, help="independent truth suite directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--resource-root", type=Path)
    parser.add_argument("--producer-root", type=Path,
                        help="frozen checkout or extracted wheel directory; defaults to this repository")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--depth-threads", type=int, default=4)
    parser.add_argument("--max-image-size", type=int, default=448)
    parser.add_argument("--cases", nargs="*", help="selected scene IDs; other denominator rows remain pending")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--rerun", action="store_true", help="create a new attempt even when a completed run exists")
    parser.add_argument("producer_args", nargs=argparse.REMAINDER, help="extra producer options after --")
    args = parser.parse_args(argv)
    suite = _read_json(args.suite / "suite.json")
    cases = suite["scenes"]
    ids = {record["id"] for record in cases}
    if args.cases is not None and not set(args.cases) <= ids:
        parser.error("--cases contains an unknown fixed suite ID")
    if args.threads < 1 or args.depth_threads < 1:
        parser.error("thread counts must be positive")
    repo = Path(__file__).resolve().parents[1]
    producer_root = (args.producer_root or repo).resolve()
    if not (producer_root / "lingshu/world/reconstruct").is_dir():
        parser.error("--producer-root must contain lingshu/world/reconstruct")
    result = {"schema": "lingshu.reconstruction-truth/1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "suite": str(args.suite.resolve()), "fixed_denominator": len(cases),
              "truth_source": suite["truth_source"], "producer_source_root": str(producer_root),
              "producer_source_sha256": _fingerprint(producer_root),
              "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "truth_generator_sha256": suite.get("generator_sha256"),
              "acceptance": "W3 after a single evaluation-only Sim3; no absolute metric scale claim",
              "scenes": [{"id": case["id"], "family": case["family"], "status": "pending_not_run"}
                         for case in cases]}
    args.run_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.output, result)
    for case, row in zip(cases, result["scenes"]):
        scene, run = args.suite / case["id"], args.run_root / case["id"]
        if args.cases is not None and case["id"] not in args.cases:
            continue
        started = time.perf_counter()
        row["producer_source_sha256"] = _fingerprint(producer_root)
        try:
            if not (scene / "scene.json").is_file():
                raise FileNotFoundError(f"truth scene not generated: {scene}")
            if not args.evaluate_only and (args.rerun or not (run / "report.json").is_file()):
                if run.exists():
                    attempt = 2
                    while (args.run_root / f"{case['id']}-attempt-{attempt:02d}").exists():
                        attempt += 1
                    run = args.run_root / f"{case['id']}-attempt-{attempt:02d}"
                row["run"] = str(run.resolve())
                command = [args.python, "-m", "lingshu.world.reconstruct", str((scene / "images").resolve()),
                           "--output", str(run.resolve()), "--threads", str(args.threads),
                           "--depth-threads", str(args.depth_threads), "--depth-views", "all",
                           "--max-image-size", str(args.max_image_size)]
                if args.resource_root:
                    command += ["--resource-root", str(args.resource_root.resolve())]
                command += args.producer_args[1:] if args.producer_args[:1] == ["--"] else args.producer_args
                row["command"] = command
                log = args.output.parent / f"{args.output.stem}-{case['id']}-producer.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open("w", encoding="utf-8") as stream:
                    process = subprocess.run(command, cwd=producer_root, stdout=stream, stderr=subprocess.STDOUT)
                row["producer_exit_code"], row["producer_log"] = process.returncode, str(log.resolve())
                if process.returncode:
                    raise RuntimeError(f"producer failed with exit code {process.returncode}; see {log}")
            row["run"] = str(run.resolve())
            row.update(evaluate_scene(scene, run, threads=args.threads))
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
            row.update(status="failed", error=f"{type(error).__name__}: {error}", w3_sim3_aligned_pass=False)
        row["benchmark_wall_seconds"] = round(time.perf_counter() - started, 3)
        result["summary"] = {"measured": sum(r["status"] == "measured" for r in result["scenes"]),
                             "failed": sum(r["status"] == "failed" for r in result["scenes"]),
                             "pending": sum(r["status"] == "pending_not_run" for r in result["scenes"]),
                             "w3_sim3_aligned_pass": sum(r.get("w3_sim3_aligned_pass", False) for r in result["scenes"]),
                             "fixed_denominator": len(cases)}
        _write_json(args.output, result)
        print(json.dumps({"id": case["id"], "status": row["status"],
                          "w3_pass": row.get("w3_sim3_aligned_pass"),
                          "error": row.get("error")}, ensure_ascii=False), flush=True)
    _write_json(args.output, result)
    return 0 if len(result["scenes"]) == len(cases) and all(
        r.get("w3_sim3_aligned_pass", False) for r in result["scenes"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
