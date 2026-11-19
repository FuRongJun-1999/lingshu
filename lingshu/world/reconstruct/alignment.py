from __future__ import annotations

from pathlib import Path

import numpy as np


class AlignmentError(RuntimeError):
    """Sparse anchors did not support a usable calibration."""


def depth_valid_mask(record, shape):
    """Read an optional input observation domain; confidence cannot restore it."""
    if "input_valid_mask" not in record:
        return None
    mask = np.asarray(record["input_valid_mask"])
    if mask.shape != tuple(shape) or mask.dtype.kind not in "biu" or np.any((mask != 0) & (mask != 1)):
        raise ValueError("input_valid_mask must be a binary bool/integer array matching depth shape")
    return mask.astype(bool, copy=False)


def estimate_alignment(tracks: list[dict], depth_dir: Path, poses: dict,
                       confidence_percentile: float = 20.0, min_samples: int = 30) -> dict:
    observations_by_image = {name: [] for name in poses}
    for track in tracks:
        for observation in track["observations"]:
            if observation["image"] in observations_by_image:
                observations_by_image[observation["image"]].append(observation)

    depth_values = []
    sfm_z_values = []
    confidence_values = []
    image_indices = []
    names = list(poses)
    for image_index, (name, observations) in enumerate(observations_by_image.items()):
        depth_file = depth_dir / f"{Path(name).stem}.npz"
        if not observations or not depth_file.exists():
            continue
        with np.load(depth_file) as record:
            depth = record["depth"]
            confidence = record["confidence"]
            input_mask = depth_valid_mask(record, depth.shape)
            pose = poses[name]
            confidence_domain = np.isfinite(confidence)
            if input_mask is not None:
                confidence_domain &= input_mask
            valid_confidence = confidence[confidence_domain]
            if not valid_confidence.size:
                continue
            confidence_cutoff = float(np.percentile(valid_confidence, confidence_percentile))
            h, w = depth.shape
            xs, ys = np.asarray([obs["pixel_xy"] for obs in observations]).T
            # Native COLMAP coordinates have centers at i+.5; depth arrays
            # use integer indices. The containing pixel is floor, not rint.
            ix = np.clip(np.floor(xs * w / pose["width"]), 0, w - 1).astype(np.intp)
            iy = np.clip(np.floor(ys * h / pose["height"]), 0, h - 1).astype(np.intp)
            predicted_depth, conf = depth[iy, ix], confidence[iy, ix]
            sfm_z = np.asarray([obs["camera_z"] for obs in observations], dtype=np.float64)
            valid = (np.isfinite(predicted_depth) & (predicted_depth > 0) &
                     np.isfinite(conf) & (conf >= confidence_cutoff) & np.isfinite(sfm_z) & (sfm_z > 0))
            if input_mask is not None:
                valid &= input_mask[iy, ix]
            depth_values.append(predicted_depth[valid])
            sfm_z_values.append(sfm_z[valid])
            confidence_values.append(conf[valid])
            image_indices.append(np.full(np.count_nonzero(valid), image_index, dtype=np.int32))

    x = np.concatenate(depth_values, dtype=np.float64) if depth_values else np.empty(0)
    if len(x) < min_samples:
        raise AlignmentError(f"depth alignment needs at least {min_samples} valid sparse/depth observations; found {len(x)}")
    y = np.concatenate(sfm_z_values)
    confidences = np.concatenate(confidence_values)
    image_indices = np.concatenate(image_indices)

    y_spread = float(np.quantile(y, 0.9) - np.quantile(y, 0.1))
    x_spread = float(np.quantile(x, 0.9) - np.quantile(x, 0.1))
    correlation = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else None
    threshold = max(0.025 * y_spread, 0.015 * float(np.median(np.abs(y))), 1e-8)
    rng = np.random.default_rng(73021)
    best_inliers = None
    best_score = (-1, -float("inf"))
    min_delta = max(1e-9, float(np.ptp(x)) * 1e-6)
    for _ in range(min(1200, max(256, len(x) * 20))):
        first, second = rng.choice(len(x), size=2, replace=False)
        delta_x = x[second] - x[first]
        if abs(delta_x) <= min_delta:
            continue
        slope = (y[second] - y[first]) / delta_x
        if not np.isfinite(slope) or slope <= 0:
            continue
        intercept = y[first] - slope * x[first]
        residuals = np.abs(y - (slope * x + intercept))
        inliers = residuals <= threshold
        count = int(inliers.sum())
        if count < best_score[0]:
            continue
        median_error = float(np.median(residuals[inliers])) if count else float("inf")
        if (count, -median_error) > best_score:
            best_score = (count, -median_error)
            best_inliers = inliers
    if best_inliers is None or int(best_inliers.sum()) < min_samples:
        raise AlignmentError("no stable positive depth scale was supported by the COLMAP tracks")

    inliers = best_inliers
    for _ in range(4):
        design = np.column_stack((x[inliers], np.ones(int(inliers.sum()), dtype=np.float64)))
        slope, intercept = np.linalg.lstsq(design, y[inliers], rcond=None)[0]
        if not np.isfinite(slope) or slope <= 0:
            raise AlignmentError(
                "sparse/depth fit produced a non-positive scale; "
                f"anchors={len(x)}, Pearson r={correlation}, "
                f"DA3 depth P10-P90 spread={x_spread:.6g}, COLMAP camera-Z P10-P90 spread={y_spread:.6g}"
            )
        residuals = y - (slope * x + intercept)
        median = float(np.median(residuals[inliers]))
        mad = float(np.median(np.abs(residuals[inliers] - median)))
        refined_threshold = max(threshold, 3.5 * 1.4826 * mad)
        refined = np.abs(residuals) <= refined_threshold
        if int(refined.sum()) < min_samples or np.array_equal(refined, inliers):
            break
        inliers = refined

    design = np.column_stack((x[inliers], np.ones(int(inliers.sum()), dtype=np.float64)))
    slope, intercept = np.linalg.lstsq(design, y[inliers], rcond=None)[0]
    if slope <= 0:
        raise AlignmentError("sparse/depth fit produced a non-positive scale")
    residuals = np.abs(y - (slope * x + intercept))
    inlier_residuals = residuals[inliers]
    frames = {}
    for index, name in enumerate(names):
        selected = (image_indices == index) & inliers
        if not selected.any():
            continue
        frame_residuals = residuals[selected]
        frames[name] = {
            "inlier_observations": int(selected.sum()),
            "median_abs_residual_sfm_units": float(np.median(frame_residuals)),
            "confidence_median": float(np.median(confidences[selected])),
        }
    inlier_ratio = float(inliers.mean())
    if inlier_ratio < 0.30:
        raise AlignmentError(f"depth alignment is unstable: robust inlier ratio {inlier_ratio:.3f} < 0.30")
    y_scale = max(float(np.median(np.abs(y[inliers]))), 1e-9)
    return {
        "method": "global RANSAC affine fit: COLMAP camera Z = scale * DA3 depth + shift",
        "da3_depth_units": "relative",
        "aligned_units": "COLMAP relative units",
        "scale_sfm_units_per_da3_unit": float(slope),
        "shift_sfm_units": float(intercept),
        "input_observations": int(len(x)),
        "da3_depth_p10_p90_spread": x_spread,
        "colmap_camera_z_p10_p90_spread": y_spread,
        "raw_anchor_pearson_correlation": correlation,
        "inlier_observations": int(inliers.sum()),
        "inlier_ratio": inlier_ratio,
        "median_abs_residual_sfm_units": float(np.median(inlier_residuals)),
        "p95_abs_residual_sfm_units": float(np.quantile(inlier_residuals, 0.95)),
        "median_relative_anchor_error": float(np.median(inlier_residuals) / y_scale),
        "p95_relative_anchor_error": float(np.quantile(inlier_residuals, 0.95) / y_scale),
        "confidence_percentile_cutoff": float(confidence_percentile),
        "frames": frames,
    }


def write_aligned_depths(depth_dir: Path, output_dir: Path, poses: dict, alignment: dict,
                         meters_per_sfm_unit: float | None = None) -> dict:
    if meters_per_sfm_unit is not None and meters_per_sfm_unit <= 0:
        raise ValueError("meters_per_sfm_unit must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    scale = alignment["scale_sfm_units_per_da3_unit"]
    shift = alignment["shift_sfm_units"]
    unit_scale = meters_per_sfm_unit or 1.0
    for name in poses:
        source = depth_dir / f"{Path(name).stem}.npz"
        if not source.exists():
            continue
        with np.load(source) as record:
            depth = record["depth"].astype(np.float32)
            confidence = record["confidence"].astype(np.float32)
            input_mask = depth_valid_mask(record, depth.shape)
        valid_source = np.isfinite(depth) & (depth > 0) & np.isfinite(confidence)
        if input_mask is not None:
            valid_source &= input_mask
        aligned = (depth * scale + shift) * unit_scale
        aligned[~valid_source | ~np.isfinite(aligned) | (aligned <= 0)] = 0
        extra = {"input_valid_mask": input_mask.astype(np.uint8)} if input_mask is not None else {}
        np.savez(output_dir / f"{Path(name).stem}.npz", depth=aligned, confidence=confidence, **extra)
    return {
        "input_units": "COLMAP relative units",
        "output_units": "meters" if meters_per_sfm_unit is not None else "COLMAP relative units",
        "meters_per_sfm_unit": meters_per_sfm_unit,
        "aligned_frames": sum((output_dir / f"{Path(name).stem}.npz").exists() for name in poses),
    }
