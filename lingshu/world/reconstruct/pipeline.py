from __future__ import annotations

import json
from pathlib import Path
import threading
import time

import cv2
import numpy as np
import psutil

from .alignment import AlignmentError, depth_valid_mask, estimate_alignment, write_aligned_depths
from .colmap import choose_model, map_global_scene, prepare_matches, read_model, run_colmap, write_sparse_exports
from .config import validate_options
from .depth import DepthEngine, convolution_mode, depth_convolution
from .fusion import fuse_scene
from .images import export_feature_masks, normalize_images, undistort_valid_masks
from .keyframes import select_depth_frames
from .verify import dense_cross_view_report, reprojection_report, write_report


class ResourceMonitor:
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.peak_rss = 0
        self.stage = "startup"
        self.stage_peaks: dict[str, int] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        process = psutil.Process()
        while not self._stop.is_set():
            stage = self.stage
            rss = process.memory_info().rss
            for child in process.children(recursive=True):
                try:
                    rss += child.memory_info().rss
                except (psutil.Error, OSError):
                    pass
            self.peak_rss = max(self.peak_rss, rss)
            self.stage_peaks[stage] = max(self.stage_peaks.get(stage, 0), rss)
            self._stop.wait(self.interval)

    def __enter__(self) -> "ResourceMonitor":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        self._thread.join()


def scene_diagonal(tracks: list[dict], poses: dict, coordinate_scale: float) -> float:
    low = np.full(3, np.inf, dtype=np.float64)
    high = np.full(3, -np.inf, dtype=np.float64)
    for track in tracks:
        xyz = np.asarray(track["xyz"], dtype=np.float64)
        low = np.minimum(low, xyz)
        high = np.maximum(high, xyz)
    for pose in poses.values():
        matrix = np.asarray(pose["cam_from_world"], dtype=np.float64)
        center = -matrix[:3, :3].T @ matrix[:3, 3]
        low = np.minimum(low, center)
        high = np.maximum(high, center)
    diagonal = float(np.linalg.norm(high - low)) * coordinate_scale
    if not np.isfinite(diagonal) or diagonal <= 0:
        raise RuntimeError("could not estimate the sparse scene extent for TSDF settings")
    return diagonal


def estimate_depth_truncation(depth_dir: Path, poses: dict) -> float:
    upper_bounds = []
    for name in poses:
        path = depth_dir / f"{Path(name).stem}.npz"
        if not path.exists():
            continue
        with np.load(path) as record:
            depth = record["depth"]
            values = depth[np.isfinite(depth) & (depth > 0)]
            if values.size:
                upper_bounds.append(float(np.quantile(values, 0.995)))
    if not upper_bounds:
        raise RuntimeError("no valid aligned depth values are available for TSDF")
    return max(float(np.median(upper_bounds)) * 1.1, 1e-4)


def _save_depth(output_dir: Path, name: str, result: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    camera = {key: result[key] for key in ("intrinsics", "extrinsics") if result[key] is not None}
    input_mask = depth_valid_mask(result, result["depth"].shape)
    if input_mask is not None:
        camera["input_valid_mask"] = input_mask.astype(np.uint8)
    np.savez(
        output_dir / f"{Path(name).stem}.npz",
        depth=result["depth"].astype(np.float32),
        confidence=result["confidence"].astype(np.float32),
        is_metric=np.asarray(result["is_metric"], dtype=np.uint8),
        **camera,
    )


class ReconstructionPipeline:
    def __init__(self, *, input_path: Path, output_path: Path, colmap: Path, model: Path,
                 depth_library: Path, runtime_bin: Path, threads: int | None = None, max_image_size: int = 1600,
                 voxel_size: float | None = None, voxel_fraction: float = 0.006,
                 confidence_percentile: float = 0.0, max_depth_units: float | None = None,
                 meters_per_sfm_unit: float | None = None, camera_model: str = "SIMPLE_RADIAL",
                 depth_views: str = "all", include_da3_pose: bool = False,
                 depth_threads: int | None = None, depth_conv: str | None = None,
                 sfm_features: str = "full", matching_mode: str = "auto"):
        validate_options(
            threads=threads, depth_threads=depth_threads, max_image_size=max_image_size,
            voxel_size=voxel_size, voxel_fraction=voxel_fraction,
            confidence_percentile=confidence_percentile, max_depth_units=max_depth_units,
            meters_per_sfm_unit=meters_per_sfm_unit, depth_views=depth_views,
            depth_conv=depth_conv, sfm_features=sfm_features, matching_mode=matching_mode,
        )
        self.input_path = input_path.resolve()
        self.output_path = output_path.resolve()
        self.colmap = colmap.resolve()
        self.model = model.resolve()
        self.depth_library = depth_library.resolve()
        self.runtime_bin = runtime_bin.resolve()
        self.threads = threads if threads is not None else 2
        if depth_threads is None:
            depth_threads = self.threads if threads is not None else min(4, psutil.cpu_count(logical=False) or 2)
        self.depth_threads = depth_threads
        self.sfm_features = sfm_features
        self.matching_mode = matching_mode
        self.depth_conv = convolution_mode(depth_conv)
        self.depth_views = depth_views
        self.include_da3_pose = include_da3_pose
        self.max_image_size = max_image_size
        self.voxel_size = voxel_size
        self.voxel_fraction = voxel_fraction
        self.confidence_percentile = confidence_percentile
        self.max_depth_units = max_depth_units
        self.meters_per_sfm_unit = meters_per_sfm_unit
        self.camera_model = camera_model

    def _progress(self, message: str) -> None:
        print(message, flush=True)

    def run(self) -> dict:
        for path in (self.input_path, self.colmap, self.model, self.depth_library):
            if not path.exists():
                raise FileNotFoundError(path)
        if self.output_path.exists() and any(self.output_path.iterdir()):
            raise FileExistsError(f"output directory is not empty: {self.output_path}")
        self.output_path.mkdir(parents=True, exist_ok=True)
        logs_dir = self.output_path / "logs"
        logs_dir.mkdir(exist_ok=True)
        timings = {}
        report = {
            "schema_version": 1,
            "pipeline": "GLOMAP CPU SfM + Depth Anything 3 Base relative depth + global affine alignment + Open3D CPU TSDF",
            "input": str(self.input_path),
            "output": str(self.output_path),
            "units": "meters" if self.meters_per_sfm_unit is not None else "COLMAP relative units",
            "absolute_metric_scale": self.meters_per_sfm_unit is not None,
            "timings_seconds": timings,
            "stages": {},
        }
        cv2.setNumThreads(self.threads)
        started = time.perf_counter()
        with ResourceMonitor() as monitor:
            monitor.stage = "normalize"
            self._progress("[0/6] Normalizing input images")
            tick = time.perf_counter()
            manifest = normalize_images(
                self.input_path, self.output_path / "00_input" / "images", self.max_image_size
            )
            feature_masks = export_feature_masks(
                manifest, self.output_path / "00_input", self.output_path / "00_input" / "feature_masks"
            )
            timings["normalize"] = time.perf_counter() - tick
            report["stages"]["input"] = {
                "images": len(manifest), "manifest": "00_input/input_manifest.json",
                "input_domain_masked_images": sum(bool(frame.get("valid_mask")) for frame in manifest),
                "input_domain_semantics": "optional input alpha>0; RGB/JPEG pixels are never classified by color",
            }

            image_dir = self.output_path / "00_input" / "images"
            database = self.output_path / "01_sfm" / "database.db"
            sparse_dir = self.output_path / "01_sfm" / "sparse"
            sparse_dir.mkdir(parents=True)
            monitor.stage = "sfm"
            self._progress("[1/6] CPU SIFT extraction, matching, and GLOMAP global mapping")
            tick = time.perf_counter()
            matching_mode = self.matching_mode
            if matching_mode == "auto":
                matching_mode = "batched" if len(manifest) >= 16 and self.threads >= 4 else "sequential"
            match_times = prepare_matches(self.colmap, database, image_dir, logs_dir, self.threads,
                                          self.max_image_size, self.camera_model,
                                          sift_first_octave={"full": -1, "native": 0}[self.sfm_features],
                                          matching_mode=matching_mode, mask_path=feature_masks)
            after_matches = time.perf_counter()
            initial_model = map_global_scene(self.colmap, database, image_dir, sparse_dir,
                                             logs_dir / "mapper.log", self.threads)
            after_mapper = time.perf_counter()
            components = json.loads((sparse_dir / "components.json").read_text(encoding="utf-8"))
            timings["sfm"] = after_mapper - tick
            report["stages"]["sfm"] = {
                "model": str(initial_model.relative_to(self.output_path)), "log_dir": "logs",
                "components_manifest": "01_sfm/sparse/components.json",
                "component_count": components["component_count"],
                "all_component_registered_images": components["all_component_registered_images"],
                "coordinate_relationship": components["coordinate_relationship"],
                "mapper": "global", "mapper_threads": self.threads,
                "feature_profile": self.sfm_features,
                "input_domain_masked_images": sum(bool(frame.get("valid_mask")) for frame in manifest),
                "feature_mask_directory": "00_input/feature_masks" if feature_masks is not None else None,
                "matching_mode": matching_mode,
                **match_times,
                "mapping_seconds": round(after_mapper - after_matches, 3),
            }

            monitor.stage = "sparse_parse"
            self._progress("[2/6] Undistorting registered views and exporting COLMAP tracks")
            tick = time.perf_counter()
            dense_workspace = self.output_path / "02_undistorted"
            run_colmap(self.colmap, "image_undistorter", [
                "--image_path", str(image_dir), "--input_path", str(initial_model),
                "--output_path", str(dense_workspace), "--output_type", "COLMAP",
                "--max_image_size", str(self.max_image_size), "--num_threads", str(self.threads),
            ], logs_dir / "image_undistorter.log", self.output_path)
            rectified_model = choose_model(dense_workspace / "sparse")
            valid_masks = undistort_valid_masks(
                manifest, self.output_path / "00_input", initial_model, rectified_model,
                dense_workspace / "masks",
            )
            reconstruction, poses, tracks = read_model(rectified_model)
            del reconstruction
            if len(poses) < 2 or len(tracks) < 30:
                raise RuntimeError(f"sparse model is insufficient: {len(poses)} registered views, {len(tracks)} points")
            unregistered = [frame["source"] for frame in manifest if frame["name"] not in poses]
            sparse_summary = write_sparse_exports(rectified_model, poses, tracks, self.output_path / "02_sparse", unregistered)
            timings["sparse_parse"] = time.perf_counter() - tick
            report["stages"]["sparse"] = sparse_summary
            report["coverage"] = {
                "input_images": len(manifest),
                "selected_model_registered_images": len(poses),
                "all_independent_components_registered_images": components["all_component_registered_images"],
                "single_coordinate_frame_complete": len(poses) == len(manifest),
                "unregistered_in_selected_model": unregistered,
            }

            monitor.stage = "depth_and_alignment"
            tick = time.perf_counter()
            frame_names, selection = select_depth_frames(poses, tracks, all_views=self.depth_views == "all")
            selection["planned_views"] = frame_names
            deferred = sorted(set(poses) - set(frame_names), key=lambda name: poses[name]["source_index"])
            self._progress(f"[3/6] DA3 CPU inference for {len(frame_names)}/{len(poses)} registered views")
            depth_dir = self.output_path / "03_depth"
            metric_flags = set()
            inferred = []
            depth_seconds = 0.0

            def infer(engine, names):
                nonlocal depth_seconds
                depth_start = time.perf_counter()
                for name in names:
                    prediction = engine.predict(dense_workspace / "images" / name, include_pose=self.include_da3_pose)
                    if name in valid_masks:
                        mask_path = dense_workspace / "masks" / valid_masks[name]
                        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                        if mask is None:
                            raise RuntimeError(f"cannot decode input valid mask for {name}: {mask_path}")
                        height, width = prediction["depth"].shape
                        prediction["input_valid_mask"] = (
                            cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST) > 0
                        ).astype(np.uint8)
                        if not prediction["input_valid_mask"].any():
                            raise RuntimeError(f"input valid mask has no supported depth pixels for {name}")
                    metric_flags.add(prediction["is_metric"])
                    _save_depth(depth_dir, name, prediction)
                    if self.include_da3_pose:
                        (depth_dir / f"{Path(name).stem}-pose.json").write_text(json.dumps({
                            "model_pose_ext": prediction["extrinsics"].tolist(),
                            "model_pose_intrinsics": prediction["intrinsics"].tolist(),
                            "depth_shape": list(prediction["depth"].shape),
                            "is_metric": prediction["is_metric"],
                        }, indent=2), encoding="utf-8")
                    inferred.append(name)
                    self._progress(f"      {len(inferred)}/{len(poses)} {name}: "
                                   f"{prediction['depth'].shape[1]}x{prediction['depth'].shape[0]}")
                depth_seconds += time.perf_counter() - depth_start

            with depth_convolution(self.depth_conv), DepthEngine(
                self.depth_library, self.model, runtime_dir=self.runtime_bin, threads=self.depth_threads,
            ) as depth_engine:
                depth_seconds += time.perf_counter() - tick
                infer(depth_engine, frame_names)
                self._progress("[4/6] Robust global scale+shift alignment from sparse tracks")
                try:
                    alignment = estimate_alignment(tracks, depth_dir, poses, self.confidence_percentile)
                    if deferred and (alignment["inlier_ratio"] < 0.45 or
                                     sum(f["inlier_observations"] >= 15 for f in alignment["frames"].values())
                                     < max(2, (len(frame_names) + 1) // 2)):
                        raise AlignmentError("too few selected views or anchors support the shared scale")
                except AlignmentError as error:
                    if not deferred:
                        raise
                    selection["alignment_fallback"] = str(error)
                    self._progress(f"      Scale support weak; processing {len(deferred)} deferred views")
                    infer(depth_engine, deferred)
                    alignment = estimate_alignment(tracks, depth_dir, poses, self.confidence_percentile)
            if len(metric_flags) != 1:
                raise RuntimeError(f"DA3 checkpoint returned inconsistent scale flags: {sorted(metric_flags)}")
            timings["depth"] = depth_seconds
            report["stages"]["depth"] = {
                "checkpoint": str(self.model),
                "engine": "depth-anything.cpp C API / ggml CPU",
                "quantization": "q8_0",
                "registered_views": len(poses),
                "inferred_views": len(inferred),
                "inferred_images": inferred,
                "skipped_images": [name for name in deferred if name not in inferred],
                "selection": selection,
                "metric": metric_flags == {True},
                "model_pose_exported": self.include_da3_pose,
                "threads": self.depth_threads,
                "convolution_mode": self.depth_conv,
                "input_domain_masked_images": [name for name in inferred if name in valid_masks],
                "depth_semantics": "relative; calibrated to COLMAP coordinates below",
            }

            alignment_dir = self.output_path / "04_alignment"
            alignment_dir.mkdir(exist_ok=True)
            conversion = write_aligned_depths(
                depth_dir, alignment_dir / "depth_aligned", poses, alignment, self.meters_per_sfm_unit
            )
            alignment["external_metric_conversion"] = conversion
            (alignment_dir / "scale_report.json").write_text(json.dumps(alignment, ensure_ascii=False, indent=2), encoding="utf-8")
            timings["alignment"] = time.perf_counter() - tick - depth_seconds
            report["stages"]["alignment"] = alignment

            monitor.stage = "fusion"
            self._progress("[5/6] Integrating confidence-filtered CPU TSDF and voxel provenance")
            tick = time.perf_counter()
            coordinate_scale = self.meters_per_sfm_unit or 1.0
            scene_size = scene_diagonal(tracks, poses, coordinate_scale)
            voxel_size = self.voxel_size or max(scene_size * self.voxel_fraction, 1e-5)
            max_depth_units = self.max_depth_units or estimate_depth_truncation(alignment_dir / "depth_aligned", poses)
            fusion = fuse_scene(
                poses, alignment_dir / "depth_aligned", dense_workspace / "images", self.output_path / "05_fused",
                coordinate_scale, voxel_size, self.confidence_percentile, max_depth_units, report["units"],
            )
            timings["fusion"] = time.perf_counter() - tick
            report["stages"]["fusion"] = fusion
            report["coverage"]["inferred_images"] = len(inferred)
            report["coverage"]["all_inputs_have_dense_depth"] = len(inferred) == len(manifest)

            monitor.stage = "verification"
            self._progress("[6/6] Sparse reprojection and dense cross-view depth checks")
            tick = time.perf_counter()
            reprojection = reprojection_report(tracks, poses)
            cross_view = dense_cross_view_report(
                poses, alignment_dir / "depth_aligned", self.confidence_percentile, coordinate_scale=coordinate_scale
            )
            report["verification"] = {
                "reprojection": reprojection,
                "dense_cross_view_depth_abs_rel": cross_view,
                "sparse_depth_anchor_abs_rel_median": alignment["median_relative_anchor_error"],
                "sparse_depth_anchor_abs_rel_p95": alignment["p95_relative_anchor_error"],
                "contour_iou": {"status": "not_available", "reason": "no reference object masks were supplied"},
                "scale_source": "user-provided meters_per_sfm_unit" if self.meters_per_sfm_unit is not None else "GLOMAP sparse reconstruction",
                "absolute_scale_status": "metric anchor supplied" if self.meters_per_sfm_unit is not None else "relative scene scale; no measured baseline was provided",
            }
            timings["verification"] = time.perf_counter() - tick
        report["timings_seconds"]["total"] = time.perf_counter() - started
        report["resource_usage"] = {
            "peak_process_tree_rss_mb": round(monitor.peak_rss / (1024 * 1024), 1),
            "stage_peak_process_tree_rss_mb": {
                stage: round(peak / (1024 * 1024), 1) for stage, peak in monitor.stage_peaks.items()
            },
        }
        report["timings_seconds"] = {key: round(value, 3) for key, value in timings.items()}
        report["timings_seconds"]["total"] = round(report["timings_seconds"].get("total", 0.0), 3)
        write_report(self.output_path / "report.json", report)
        self._progress(f"Completed in {report['timings_seconds']['total']:.1f}s; report: {self.output_path / 'report.json'}")
        return report
