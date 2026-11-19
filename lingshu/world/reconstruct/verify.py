from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .alignment import depth_valid_mask

def reprojection_report(tracks: list[dict], poses: dict) -> dict:
    by_view = {name: [] for name in poses}
    for track in tracks:
        for observation in track["observations"]:
            if observation["image"] in by_view:
                by_view[observation["image"]].append((track["xyz"], observation["pixel_xy"]))
    errors = []
    for name, pairs in by_view.items():
        if not pairs:
            continue
        xyz, pixels = (np.asarray(values, dtype=np.float64) for values in zip(*pairs))
        extrinsic = np.asarray(poses[name]["cam_from_world"], dtype=np.float64)
        intrinsic = np.asarray(poses[name]["intrinsics"], dtype=np.float64)
        camera_xyz = xyz @ extrinsic[:, :3].T + extrinsic[:, 3]
        visible = camera_xyz[:, 2] > 1e-9
        if visible.any():
            uvw = camera_xyz[visible] @ intrinsic.T
            errors.append(np.linalg.norm(uvw[:, :2] / uvw[:, 2:3] - pixels[visible], axis=1))
    if not errors:
        return {"observations": 0, "mean_px": None, "median_px": None, "p95_px": None, "rmse_px": None}
    values = np.concatenate(errors)
    return {
        "observations": int(values.size),
        "mean_px": float(values.mean()),
        "median_px": float(np.median(values)),
        "p95_px": float(np.quantile(values, 0.95)),
        "rmse_px": float(np.sqrt(np.mean(values ** 2))),
    }


def dense_cross_view_report(poses: dict, depth_dir: Path, confidence_percentile: float = 25.0,
                            stride: int = 8, *, coordinate_scale: float = 1.0) -> dict:
    ordered = sorted(
        ((name, pose, depth_dir / f"{Path(name).stem}.npz") for name, pose in poses.items()),
        key=lambda row: row[1]["source_index"],
    )
    ordered = [row for row in ordered if row[2].is_file()]
    pair_reports = []
    relative_error_batches = []
    def load(path):
        with np.load(path) as record:
            depth, confidence = record["depth"], record["confidence"]
            input_mask = depth_valid_mask(record, depth.shape)
        domain = np.isfinite(confidence)
        if input_mask is not None:
            domain &= input_mask
        finite = confidence[domain]
        cutoff = float(np.percentile(finite, confidence_percentile)) if finite.size else np.inf
        return depth, confidence, cutoff

    previous = load(ordered[0][2]) if len(ordered) >= 2 else None
    for (source_name, source_pose, _), (target_name, target_pose, target_path) in zip(ordered, ordered[1:]):
        source_depth, source_conf, source_cutoff = previous
        previous = load(target_path)
        target_depth, target_conf, target_cutoff = previous
        if not np.isfinite(source_cutoff) or not np.isfinite(target_cutoff):
            continue
        source_h, source_w = source_depth.shape
        target_h, target_w = target_depth.shape
        source_image_scale = np.diag((source_w / source_pose["width"], source_h / source_pose["height"], 1.0))
        target_image_scale = np.diag((target_w / target_pose["width"], target_h / target_pose["height"], 1.0))
        source_k = source_image_scale @ np.asarray(source_pose["intrinsics"], dtype=np.float64)
        target_k = target_image_scale @ np.asarray(target_pose["intrinsics"], dtype=np.float64)

        ys, xs = np.mgrid[stride // 2:source_h:stride, stride // 2:source_w:stride]
        sampled_depth = source_depth[ys, xs]
        sampled_conf = source_conf[ys, xs]
        valid = (np.isfinite(sampled_depth) & (sampled_depth > 0) &
                 np.isfinite(sampled_conf) & (sampled_conf >= source_cutoff))
        ys, xs, z = ys[valid].astype(np.float64), xs[valid].astype(np.float64), sampled_depth[valid].astype(np.float64)
        if not len(z):
            continue

        camera_points = np.column_stack((
            (xs + 0.5 - source_k[0, 2]) * z / source_k[0, 0],
            (ys + 0.5 - source_k[1, 2]) * z / source_k[1, 1],
            z,
        ))
        source_extrinsic = np.asarray(source_pose["cam_from_world"], dtype=np.float64)
        target_extrinsic = np.asarray(target_pose["cam_from_world"], dtype=np.float64)
        world_points = (camera_points - coordinate_scale * source_extrinsic[:, 3]) @ source_extrinsic[:, :3]
        target_points = world_points @ target_extrinsic[:, :3].T + coordinate_scale * target_extrinsic[:, 3]
        positive = target_points[:, 2] > 1e-8
        projected_x = np.floor(target_k[0, 0] * target_points[:, 0] / np.maximum(target_points[:, 2], 1e-8) + target_k[0, 2]).astype(np.int64)
        projected_y = np.floor(target_k[1, 1] * target_points[:, 1] / np.maximum(target_points[:, 2], 1e-8) + target_k[1, 2]).astype(np.int64)
        in_frame = positive & (projected_x >= 0) & (projected_x < target_w) & (projected_y >= 0) & (projected_y < target_h)
        candidate = np.flatnonzero(in_frame)
        if not len(candidate):
            continue
        observed = target_depth[projected_y[candidate], projected_x[candidate]].astype(np.float64)
        confidence = target_conf[projected_y[candidate], projected_x[candidate]]
        projected_depth = target_points[candidate, 2]
        valid_target = np.isfinite(observed) & (observed > 0) & np.isfinite(confidence) & (confidence >= target_cutoff)
        # Ignore source points hidden behind the nearer surface in the target view.
        visible = projected_depth <= observed + np.maximum(0.02 * observed, 1e-4 * coordinate_scale)
        keep = valid_target & visible
        if not keep.any():
            continue
        errors = np.abs(projected_depth[keep] - observed[keep]) / observed[keep]
        relative_error_batches.append(errors.astype(np.float32, copy=False))
        pair_reports.append({
            "source": source_name,
            "target": target_name,
            "tested_correspondences": int(keep.sum()),
            "median_abs_rel": float(np.median(errors)),
            "p95_abs_rel": float(np.quantile(errors, 0.95)),
        })

    if not relative_error_batches:
        return {"status": "no_valid_overlap", "pairs": pair_reports, "tested_correspondences": 0}
    values = np.concatenate(relative_error_batches).astype(np.float64, copy=False)
    return {
        "status": "measured",
        "sampling_stride_px": int(stride),
        "confidence_percentile": float(confidence_percentile),
        "tested_correspondences": int(values.size),
        "median_abs_rel": float(np.median(values)),
        "p95_abs_rel": float(np.quantile(values, 0.95)),
        "within_10_percent_ratio": float(np.mean(values <= 0.10)),
        "pairs": pair_reports,
    }


def write_report(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
