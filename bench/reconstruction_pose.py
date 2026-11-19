"""Evaluate saved reconstruction poses against an independent camera par file.

The par format is a count followed by ``name + 9 K + 9 R + 3 t`` per line,
with row-major K and world-to-camera [R|t]. Calibration is evaluation input
only. Lengths remain in unspecified dataset units; this is not a dense, mesh,
absolute-scale, or fixed-20 W3 evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path, PurePosixPath

import numpy as np


def _load_math():
    # Resolve beside this file, never through the caller's working directory.
    path = Path(__file__).resolve().with_name("reconstruction_truth.py")
    spec = importlib.util.spec_from_file_location("_reconstruction_pose_truth_math", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load reconstruction math: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


truth = _load_math()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_name(value, label):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty relative image name")
    name = PurePosixPath(value.replace("\\", "/"))
    if name.is_absolute() or ":" in value or any(part in (".", "..") for part in name.parts):
        raise ValueError(f"invalid {label}: {value}")
    if str(name) == ".":
        raise ValueError(f"invalid {label}: {value}")
    return name.as_posix()


def _rotation(matrix, label):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"{label} must be a finite 3x3 rotation")
    if (not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-6, rtol=0)
            or abs(np.linalg.det(matrix) - 1) > 1e-6):
        raise ValueError(f"{label} is not a proper orthonormal rotation")
    return matrix


def parse_calibration(path):
    """Read strict count/name/K/R/t records without assuming SI length units."""
    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not lines:
        raise ValueError("calibration is empty")
    try:
        declared = int(lines[0])
    except ValueError as error:
        raise ValueError("calibration first line must be an integer camera count") from error
    if declared < 1 or len(lines) != declared + 1:
        raise ValueError("calibration declared count does not match its camera records")
    records = {}
    for number, line in enumerate(lines[1:], 2):
        fields = line.split()
        if len(fields) != 22:
            raise ValueError(f"calibration line {number} needs name + 21 numbers")
        name = _relative_name(fields[0], "calibration name")
        if "/" in name or name != fields[0]:
            raise ValueError(f"calibration name must be a basename: {fields[0]}")
        if name in records:
            raise ValueError(f"duplicate calibration name: {name}")
        try:
            values = np.asarray(fields[1:], dtype=float)
        except ValueError as error:
            raise ValueError(f"calibration {name} contains invalid numbers") from error
        if not np.isfinite(values).all():
            raise ValueError(f"calibration {name} contains non-finite values")
        intrinsic = values[:9].reshape(3, 3)
        if (intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0
                or not np.allclose(intrinsic[2], [0, 0, 1], atol=1e-10, rtol=0)
                or abs(intrinsic[1, 0]) > 1e-10):
            raise ValueError(f"calibration {name} has invalid K")
        rotation = _rotation(values[9:18].reshape(3, 3), f"calibration {name} R")
        records[name] = {"K": intrinsic, "pose": np.column_stack((rotation, values[18:]))}
    return records


def _manifest_map(path):
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("input manifest must be a nonempty list")
    names, sources = {}, set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("input manifest entries must be objects")
        name = _relative_name(record.get("name"), "manifest name")
        source = record.get("source")
        if not isinstance(source, str) or not source:
            raise ValueError(f"manifest {name} needs a source path")
        source_name = PurePosixPath(source.replace("\\", "/")).name
        if not source_name or source_name in (".", ".."):
            raise ValueError(f"manifest {name} has an invalid source basename")
        if name in names or source_name in sources:
            raise ValueError("duplicate manifest native name or source basename")
        names[name] = source_name
        sources.add(source_name)
    return names


def _has_model(path):
    return any((path / f"cameras.{suffix}").is_file() and (path / f"images.{suffix}").is_file()
               for suffix in ("bin", "txt"))


def _confined_model(run, value):
    if not isinstance(value, str) or not value:
        return None
    candidate = Path(value.replace("\\", "/"))
    candidate = (candidate if candidate.is_absolute() else run / candidate).resolve()
    return candidate if candidate.is_relative_to(run) and _has_model(candidate) else None


def select_native_model(run, report):
    """Prefer the reported original model, then an unambiguous 01_sfm model."""
    run = Path(run).resolve()
    original = (run / "01_sfm").resolve()
    if not isinstance(report, dict):
        raise ValueError("run report must be an object")
    stages = report.get("stages", {})
    if not isinstance(stages, dict):
        raise ValueError("run report stages must be an object")
    sfm = stages.get("sfm", {})
    if not isinstance(sfm, dict):
        raise ValueError("run report stages.sfm must be an object")
    reported = _confined_model(run, sfm.get("model"))
    if reported is not None and reported.is_relative_to(original):
        return reported, "original_01_sfm_reported"
    components = original / "sparse/components.json"
    if components.is_file():
        component_record = json.loads(components.read_text(encoding="utf-8"))
        if not isinstance(component_record, dict):
            raise ValueError("original SfM components.json must be an object")
        selected = component_record.get("selected_model")
        candidate = _confined_model(run, str((components.parent / selected).resolve())) if isinstance(selected, str) else None
        if candidate is None or not candidate.is_relative_to(original):
            raise ValueError("invalid original SfM selected model in components.json")
        return candidate, "original_01_sfm_components"
    candidates = set()
    if original.is_dir():
        for filename in ("cameras.bin", "cameras.txt"):
            for path in original.rglob(filename):
                candidate = path.parent.resolve()
                if candidate.is_relative_to(original) and _has_model(candidate):
                    candidates.add(candidate)
    if len(candidates) == 1:
        return candidates.pop(), "original_01_sfm_discovered"
    if len(candidates) > 1:
        raise ValueError("ambiguous original SfM models; report must identify the selected component")
    if reported is not None:
        return reported, "reported_model_fallback_no_original_01_sfm"
    raise ValueError("no native reconstruction model found in run")


def _read_native_poses(model):
    import pycolmap

    reconstruction = pycolmap.Reconstruction(str(model))
    records = {}
    for image in reconstruction.images.values():
        if not image.has_pose:
            continue
        name = _relative_name(image.name, "native image name")
        if name in records:
            raise ValueError(f"duplicate native image name: {name}")
        pose = np.asarray(image.cam_from_world().matrix(), dtype=float)
        if pose.shape != (3, 4) or not np.isfinite(pose).all():
            raise ValueError(f"native {name} has a non-finite or invalid pose")
        _rotation(pose[:, :3], f"native {name} R")
        records[name] = pose
    return records


def _center_rank(poses):
    centers = np.asarray([truth.camera_center(pose) for pose in poses])
    singular = np.linalg.svd(centers - centers.mean(axis=0), compute_uv=False)
    return int(np.sum(singular > max(singular[0] * 1e-6, 1e-10)))


def evaluate_pose_run(run, calibration):
    """Read a saved native run and return independent calibrated pose metrics."""
    run, calibration = Path(run).resolve(), Path(calibration).resolve()
    manifest_path, report_path = run / "00_input/input_manifest.json", run / "report.json"
    source_map = _manifest_map(manifest_path)
    expected = parse_calibration(calibration)
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    if not isinstance(report, dict):
        raise ValueError("run report must be an object")
    model, selection = select_native_model(run, report)
    native = _read_native_poses(model)
    unexpected = sorted(set(native) - set(source_map))
    if unexpected:
        raise ValueError(f"native image names do not match input manifest: {unexpected}")
    names = sorted(name for name in native if source_map[name] in expected)
    if len(names) < 3:
        raise ValueError("at least three common calibrated registered poses are required")
    predicted = np.asarray([native[name] for name in names])
    target = np.asarray([expected[source_map[name]]["pose"] for name in names])
    if min(_center_rank(predicted), _center_rank(target)) < 2:
        raise ValueError("center-only Sim3 requires non-collinear, non-collapsed camera centers")
    similarity = truth.fit_similarity(predicted, target)
    if similarity["method"] != "all_camera_centers_umeyama":
        raise ValueError("center-only Sim3 must not use camera orientations for its gauge")
    similarity.update(scale_source="camera calibration used by evaluator only; dataset units unspecified",
                      coordinate_units="dataset", absolute_scale_observable=False)
    errors = truth.pose_errors(predicted, target, similarity)
    centers = np.asarray([truth.camera_center(pose) for pose in target])
    diameter = float(np.max(np.linalg.norm(centers[:, None] - centers[None, :], axis=2)))
    translations = np.linalg.norm(truth.transform_points(
        [truth.camera_center(pose) for pose in predicted], similarity) - centers, axis=1)
    identity = {"scale": 1., "rotation": np.eye(3).tolist(), "translation": [0., 0., 0.]}
    views = []
    for name, p, g, distance in zip(names, predicted, target, translations):
        view = truth.pose_errors([p], [g], similarity)
        views.append({"native_name": name, "source_basename": source_map[name],
                      "rotation_degrees": view["rotation_degrees"]["mean"],
                      "center_error_dataset_units": float(distance)})
    pairs = []
    for i, j in itertools.combinations(range(len(names)), 2):
        rp = np.column_stack((predicted[i, :, :3] @ predicted[j, :, :3].T, np.zeros(3)))
        rg = np.column_stack((target[i, :, :3] @ target[j, :, :3].T, np.zeros(3)))
        pair_error = truth.pose_errors([rp], [rg], identity)["rotation_degrees"]["mean"]
        pairs.append({"a": source_map[names[i]], "b": source_map[names[j]],
                      "relative_rotation_degrees": pair_error})
    files = [manifest_path]
    if report_path.is_file():
        files.append(report_path)
    files.extend(model / f"{part}.{suffix}" for part in ("cameras", "images", "points3D", "rigs", "frames")
                 for suffix in ("bin", "txt") if (model / f"{part}.{suffix}").is_file())
    run_hashes = {path.relative_to(run).as_posix(): _sha256(path) for path in sorted(files)}
    source_files = [Path(__file__).resolve(), Path(truth.__file__).resolve()]
    source_hashes = {path.name: _sha256(path) for path in source_files}
    calibrated_sources = set(expected)
    input_sources = set(source_map.values())
    common_sources = {source_map[name] for name in names}
    return {
        "schema": "lingshu.independent-reconstruction-pose/1",
        "scope": "Independent saved-run camera pose evaluation only; GT not supplied to production. No dense/mesh or absolute metric scale claim; separate from fixed-20 W3 evidence.",
        "run": str(run), "calibration": str(calibration), "native_model": str(model),
        "model_selection": selection,
        "conventions": {"calibration_format": "count, then basename + row-major 9K + 9R + 3t; world-to-camera K[R|t]",
                        "alignment": "one center-only proper Sim3 for all common views",
                        "length_units": "dataset", "length_units_declared_by_par_format": False,
                        "absolute_metric_scale_observable": False},
        "fingerprints": {"source_sha256": source_hashes, "run_files_sha256": run_hashes,
                         "run_sha256": hashlib.sha256(json.dumps(run_hashes, sort_keys=True).encode()).hexdigest(),
                         "calibration_sha256": _sha256(calibration)},
        "coverage": {"calibration_cameras": len(expected), "input_images": len(source_map),
                     "registered_native_images": len(native), "common_calibrated_registered_images": len(names),
                     "source_names_one_to_one": True,
                     "registered_all_inputs": set(native) == set(source_map),
                     "evaluated_all_calibrations": common_sources == calibrated_sources,
                     "unregistered_input_native_names": sorted(set(source_map) - set(native)),
                     "input_sources_without_calibration": sorted(input_sources - calibrated_sources),
                     "calibrations_without_input": sorted(calibrated_sources - input_sources),
                     "calibrations_without_registered_pose": sorted(calibrated_sources - common_sources)},
        "evaluation_similarity": similarity,
        "pose": {"rotation_degrees": errors["rotation_degrees"],
                 "center_translation_dataset_units": errors["translation_m_after_sim3"],
                 "center_translation_fraction_of_GT_center_diameter": truth._summary(translations / diameter)},
        "gauge_invariant_relative_rotation_degrees": truth._summary([p["relative_rotation_degrees"] for p in pairs]),
        "per_view": views, "relative_pairs": pairs,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = evaluate_pose_run(args.run, args.calibration)
        protected = {args.calibration.resolve(), Path(__file__).resolve(), Path(truth.__file__).resolve()}
        protected.update(args.run.resolve() / name for name in result["fingerprints"]["run_files_sha256"])
        if args.output.resolve() in protected:
            raise ValueError("output must not overwrite calibration, evaluator, or run input files")
        if args.output.exists():
            if not args.output.is_file():
                raise ValueError("existing output must be this evaluator's JSON receipt")
            try:
                previous = json.loads(args.output.read_text(encoding="utf-8"))
            except (UnicodeError, json.JSONDecodeError) as error:
                raise ValueError("existing output must be this evaluator's JSON receipt") from error
            if not isinstance(previous, dict) or previous.get("schema") != result["schema"]:
                raise ValueError("existing output must be this evaluator's JSON receipt")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        parser.error(str(error))
    print(json.dumps({"output": str(args.output.resolve()), "coverage": result["coverage"], "pose": result["pose"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
