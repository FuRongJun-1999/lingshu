from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import importlib.util
import os
import subprocess
import sys
from unittest.mock import patch

from lingshu.world.reconstruct.config import validate_options

RECONSTRUCTION_AVAILABLE = all(importlib.util.find_spec(name) is not None
                               for name in ("numpy", "cv2", "pycolmap", "open3d", "scipy", "psutil"))
if RECONSTRUCTION_AVAILABLE:
    import numpy as np
    from lingshu.world.reconstruct.alignment import estimate_alignment
    from lingshu.world.reconstruct.fusion import VoxelProvenance, _transform_points, _write_mesh_provenance, fuse_scene
    from lingshu.world.reconstruct.verify import dense_cross_view_report, reprojection_report


@unittest.skipUnless(RECONSTRUCTION_AVAILABLE, "optional world-reconstruct dependencies are not installed")
class GeometryTests(unittest.TestCase):
    def test_global_scale_rejects_sparse_track_outliers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            depth_dir = root / "depth"
            depth_dir.mkdir()
            poses = {}
            tracks = []
            for image_index in range(3):
                name = f"{image_index:06d}.jpg"
                poses[name] = {"width": 20, "height": 10}
                depth = np.zeros((10, 20), dtype=np.float32)
                confidence = np.ones((10, 20), dtype=np.float32)
                for point_index in range(15):
                    z = 2.0 + point_index * 0.1
                    depth_value = 2.0 * z + 0.3
                    if image_index == 2 and point_index == 14:
                        z = 0.2
                    x = point_index + 2
                    depth[4, x] = depth_value
                np.savez_compressed(depth_dir / f"{image_index:06d}.npz", depth=depth, confidence=confidence)
                for point_index in range(15):
                    z = 2.0 + point_index * 0.1
                    if image_index == 2 and point_index == 14:
                        z = 0.2
                    tracks.append({"observations": [{
                        "image": name,
                        "pixel_xy": [float(point_index + 2), 4.0],
                        "camera_z": z,
                    }]})
            report = estimate_alignment(tracks, depth_dir, poses, confidence_percentile=0, min_samples=20)
            self.assertAlmostEqual(report["scale_sfm_units_per_da3_unit"], 0.5, places=5)
            self.assertAlmostEqual(report["shift_sfm_units"], -0.15, places=5)
            self.assertGreater(report["inlier_ratio"], 0.9)

    def test_camera_to_world_backprojection_and_metric_translation(self):
        depth = np.full((2, 2), 2.0, dtype=np.float32)
        confidence = np.full((2, 2), 0.8, dtype=np.float32)
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        pose = {
            "name": "000000.jpg", "width": 2, "height": 2,
            "intrinsics": [[1.0, 0, 0.5], [0, 1.0, 0.5], [0, 0, 1.0]],
            "cam_from_world": [[1, 0, 0, -1], [0, 1, 0, 0], [0, 0, 1, 0]],
        }
        points, colors, keys, pixels = _transform_points(depth, confidence, image, pose, 2.0, 0.5, 0, 10)
        center = points[np.argmin(np.linalg.norm(points - np.array([2.0, 0.0, 2.0]), axis=1))]
        np.testing.assert_allclose(center, [2.0, 0.0, 2.0], atol=1e-6)
        self.assertEqual(len(points), 4)
        self.assertEqual(colors.shape, (4, 3))
        self.assertEqual(keys.shape, (4, 3))
        self.assertEqual(pixels.shape, (4,))

    def test_voxel_provenance_records_each_source_frame(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = VoxelProvenance(root / "provenance.sqlite")
            point = np.array([[0.1, 0.2, 0.3]], dtype=np.float64)
            color = np.array([[10, 20, 30]], dtype=np.uint8)
            key = np.array([[1, 2, 3]], dtype=np.int32)
            pixel = np.array([15], dtype=np.int64)
            for index in range(2):
                store.add_frame(point, color, key, pixel, {
                    "source": f"original-{index}.jpg", "name": f"{index:06d}.jpg", "image_id": index + 1,
                }, 5)
            store.close()
            connection = sqlite3.connect(root / "provenance.sqlite")
            row = connection.execute("SELECT observations, sources FROM voxels").fetchone()
            connection.close()
            self.assertEqual(row[0], 2)
            sources = json.loads(row[1])
            self.assertEqual([source["frame"] for source in sources], ["original-0.jpg", "original-1.jpg"])

    def test_mesh_vertices_get_nearest_voxel_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = VoxelProvenance(root / "provenance.sqlite")
            store.add_frame(
                np.array([[0.1, 0.2, 0.3]]), np.array([[10, 20, 30]], dtype=np.uint8),
                np.array([[1, 2, 3]], dtype=np.int32), np.array([15]),
                {"source": "original.jpg", "name": "000000.jpg", "image_id": 1}, 5,
            )
            store.close()
            mesh_report = _write_mesh_provenance(
                np.array([[0.1, 0.2, 0.3], [10, 10, 10]], dtype=np.float32),
                root / "provenance.sqlite", root / "mesh_provenance.jsonl", 0.1, "relative",
            )
            rows = [json.loads(line) for line in (root / "mesh_provenance.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(mesh_report["mapped_vertices"], 1)
            self.assertEqual(rows[0]["source"]["sources"][0]["frame"], "original.jpg")
            self.assertIsNone(rows[1]["source"])

    def test_cpu_tsdf_extracts_a_plane_seen_from_two_views(self):
        import cv2

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            depth = np.full((48, 64), 2, dtype=np.float32)
            poses = {}
            for i in range(2):
                name = f"view-{i}.png"
                self.assertTrue(cv2.imwrite(str(root / name), np.full((48, 64, 3), 127, dtype=np.uint8)))
                np.savez(root / f"view-{i}.npz", depth=depth, confidence=np.ones_like(depth))
                poses[name] = {
                    "name": name, "source_index": i, "source": name, "image_id": i + 1,
                    "width": 64, "height": 48,
                    "intrinsics": [[48, 0, 31.5], [0, 48, 23.5], [0, 0, 1]],
                    "cam_from_world": [[1, 0, 0, -0.1 * i], [0, 1, 0, 0], [0, 0, 1, 0]],
                }
            result = fuse_scene(poses, root, root, root / "fused", 1, 0.1, 25, 5, "relative")
            self.assertEqual(result["integrated_frames"], 2)
            self.assertEqual(result["mesh_weight_threshold"], 0)
            self.assertEqual(result["mesh_support"]["minimum_voxel_observations"], 1)
            self.assertGreater(result["mesh_triangles"], 0)
            self.assertGreater(result["mesh_provenance"]["mapping_ratio"], 0.95)

    def test_dense_cross_view_depth_projection_reports_consistent_overlap(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            depth_dir = root / "depth"
            depth_dir.mkdir()
            depth = np.full((8, 8), 4.0, dtype=np.float32)
            confidence = np.ones_like(depth)
            for index in range(2):
                np.savez_compressed(depth_dir / f"{index:06d}.npz", depth=depth, confidence=confidence)
            poses = {
                "000000.jpg": {
                    "source_index": 0, "width": 8, "height": 8,
                    "intrinsics": [[4, 0, 3.5], [0, 4, 3.5], [0, 0, 1]],
                    "cam_from_world": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
                },
                "000001.jpg": {
                    "source_index": 1, "width": 8, "height": 8,
                    "intrinsics": [[4, 0, 3.5], [0, 4, 3.5], [0, 0, 1]],
                    "cam_from_world": [[1, 0, 0, -0.5], [0, 1, 0, 0], [0, 0, 1, 0]],
                },
            }
            report = dense_cross_view_report(poses, depth_dir, confidence_percentile=0, stride=2)
            self.assertEqual(report["status"], "measured")
            self.assertGreater(report["tested_correspondences"], 0)
            self.assertAlmostEqual(report["median_abs_rel"], 0.0)

    def test_reprojection_error_uses_camera_intrinsics_and_pose(self):
        pose = {
            "cam_from_world": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
            "intrinsics": [[100, 0, 5], [0, 100, 4], [0, 0, 1]],
        }
        tracks = [{"xyz": [0, 0, 2], "observations": [{"image": "a.jpg", "pixel_xy": [5, 4]}]}]
        report = reprojection_report(tracks, {"a.jpg": pose})
        self.assertEqual(report["observations"], 1)
        self.assertAlmostEqual(report["mean_px"], 0.0)


class ReconstructionConfigurationTests(unittest.TestCase):
    def test_dense_coverage_is_complete_by_default(self):
        from lingshu.world.reconstruct.__main__ import parser
        self.assertEqual(parser().parse_args(["photos"]).depth_views, "all")
        self.assertEqual(parser().parse_args(["photos"]).confidence_percentile, 0)
        self.assertEqual(parser().parse_args(["photos", "--depth-views", "auto"]).depth_views, "auto")
        if RECONSTRUCTION_AVAILABLE:
            import inspect
            from lingshu.world.reconstruct.pipeline import ReconstructionPipeline
            self.assertEqual(inspect.signature(ReconstructionPipeline).parameters["depth_views"].default, "all")
            self.assertEqual(inspect.signature(ReconstructionPipeline).parameters["confidence_percentile"].default, 0)

    def test_cli_help_works_without_site_packages(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, "-S", "-m", "lingshu.world.reconstruct", "--help"],
                                cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--resource-root", result.stdout)
        self.assertNotIn("--mapper", result.stdout)

    def test_package_import_loads_no_optional_runtime(self):
        root = Path(__file__).resolve().parents[1]
        code = ('import sys; import lingshu.world.reconstruct; '
                'assert not {"numpy", "cv2", "pycolmap", "open3d", "torch"}.intersection(sys.modules)')
        result = subprocess.run([sys.executable, "-S", "-c", code], cwd=root,
                                capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cli_reports_missing_extra_without_traceback(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, "-S", "-m", "lingshu.world.reconstruct", "."],
                                cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.assertEqual(result.returncode, 1)
        self.assertIn("lingshu[world-reconstruct]", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_cli_rejects_nonfinite_configuration_before_import(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, "-S", "-m", "lingshu.world.reconstruct", ".",
                                 "--voxel-size", "nan"], cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.assertEqual(result.returncode, 2)
        self.assertIn("voxel_size must be a finite number", result.stderr)
        self.assertNotIn("Missing import", result.stderr)

    def test_rejects_invalid_numeric_options(self):
        for options in ({"threads": 0}, {"threads": True}, {"threads": 1.5},
                        {"depth_threads": -1}, {"max_image_size": 0},
                        {"voxel_size": float("nan")}, {"voxel_fraction": float("inf")},
                        {"meters_per_sfm_unit": float("nan")}, {"max_depth_units": 0},
                        {"confidence_percentile": -1}, {"confidence_percentile": 101},
                        {"confidence_percentile": float("nan")}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                validate_options(**options)
        validate_options(threads=20, depth_threads=16, meters_per_sfm_unit=0.5,
                         confidence_percentile=0, depth_views="all")

    def test_rejects_unknown_programmatic_options(self):
        for options in ({"depth_views": "partial"}, {"matching_mode": "unknown"},
                        {"sfm_features": "lossless_fast"}, {"depth_conv": "tensor"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                validate_options(**options)

    def test_resource_paths_respect_explicit_root_and_overrides(self):
        from lingshu.world.reconstruct.__main__ import parser, resource_paths
        args = parser().parse_args(["photos", "--resource-root", "assets", "--model", "custom.gguf"])
        paths = resource_paths(args)
        self.assertEqual(paths["model"], Path("custom.gguf"))
        self.assertTrue(paths["colmap"].is_relative_to(Path("assets")))
        with patch.dict(os.environ, {"LINGSHU_RECONSTRUCTION_HOME": "env-assets"}):
            env_paths = resource_paths(parser().parse_args(["photos"]))
        self.assertTrue(env_paths["model"].is_relative_to(Path("env-assets")))

    @unittest.skipUnless(RECONSTRUCTION_AVAILABLE, "optional world-reconstruct dependencies are not installed")
    def test_requested_threads_reach_colmap_environment(self):
        from lingshu.world.reconstruct.colmap import run_colmap
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch("lingshu.world.reconstruct.colmap.subprocess.run") as run:
                run.return_value.returncode = 0
                run_colmap(root / "colmap", "global_mapper", ["--GlobalMapper.num_threads", "20"],
                           root / "mapper.log", root)
                self.assertEqual(run.call_args.kwargs["env"]["OMP_NUM_THREADS"], "20")

    @unittest.skipUnless(RECONSTRUCTION_AVAILABLE, "optional world-reconstruct dependencies are not installed")
    def test_matching_worker_uses_installed_module(self):
        from lingshu.world.reconstruct.colmap import run_batched_matching
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch("lingshu.world.reconstruct.colmap.subprocess.run") as run:
                run.return_value.returncode = 0
                run_batched_matching(root / "database.db", root, 4)
                self.assertEqual(run.call_args.args[0][:3],
                                 [sys.executable, "-m", "lingshu.world.reconstruct.match_pairs"])

    @unittest.skipUnless(RECONSTRUCTION_AVAILABLE, "optional world-reconstruct dependencies are not installed")
    def test_native_library_without_required_adapter_fails_clearly(self):
        from lingshu.world.reconstruct.depth import DepthEngine
        with patch("lingshu.world.reconstruct.depth.ctypes.CDLL", return_value=object()):
            with self.assertRaisesRegex(RuntimeError, "missing the reconstruction C API"):
                DepthEngine(Path("missing/adapter"), Path("model.gguf"))


if __name__ == "__main__":
    unittest.main()
