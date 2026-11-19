"""Check saved CPU reconstructions through Lingshu's actual geometry consumer.

This checks contracts and projection arithmetic, not accuracy to external truth.
Run from an installed checkout: python tools/reconstruct/check_integration.py
--run /path/to/run --output /path/to/result.json
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import statistics
import time

from lingshu.world.scene_model import WorldModel


def check_run(directory: Path) -> dict:
    started = time.perf_counter()
    model = WorldModel()
    original_scene = model.scene
    model.load_reconstruction(directory)
    reconstruction = model.reconstruction
    if model.entities or model.scene is not original_scene:
        raise AssertionError("loading static geometry changed semantic scene state")
    errors = []
    max_inverse_error = max_virtual_error = 0.0
    tracks = list(reconstruction.iter_tracks())
    for track in tracks:
        point = track["xyz"]
        for observation in track["observations"]:
            camera = reconstruction.cameras[observation["image"]]
            pixel = camera.project(point)
            if pixel is None:
                raise AssertionError("sparse observation is behind its camera")
            errors.append(math.dist(pixel, observation["pixel_xy"]))
            recovered = camera.unproject(*pixel, camera.camera_depth(point))
            inverse_error = math.dist(point, recovered) / max(1.0, math.dist(point, (0, 0, 0)))
            max_inverse_error = max(max_inverse_error, inverse_error)
            virtual = camera.as_camera3d().project(point, camera.width, camera.height)
            if virtual is not None:  # Existing renderer has a fixed 0.1 near plane.
                max_virtual_error = max(max_virtual_error,
                                        math.dist(virtual, camera.to_virtual_pixel(*pixel)))
    rmse = math.sqrt(statistics.fmean(e * e for e in errors))
    reported = reconstruction.report["verification"]["reprojection"]["rmse_px"]
    if not math.isclose(rmse, reported, abs_tol=1e-7, rel_tol=1e-7):
        raise AssertionError("consumer projection differs from the producer's sparse RMSE")
    if max_inverse_error > 1e-9 or max_virtual_error > 1e-7:
        raise AssertionError("camera conversion changed projection arithmetic")
    # Evenly sample the existing tracks; source pixels retain their BA residuals.
    sampled = tracks[::max(1, len(tracks) // 32)][:32]
    deltas = []
    for track in sampled:
        recovered = reconstruction.triangulate_track(track["point3D_id"])
        if recovered is not None:
            deltas.append(math.dist(recovered, track["xyz"]))
    if not deltas or any(not math.isfinite(delta) for delta in deltas):
        raise AssertionError("existing multiview triangulation produced no finite geometry")
    report = reconstruction.report
    return {
        "case": directory.name,
        "status": "passed",
        "scope": "static geometry integration; no independent pose/depth ground truth",
        "units": reconstruction.units,
        "coordinate_scale": reconstruction.coordinate_scale,
        "input_images": len(reconstruction.manifest),
        "registered_images": len(reconstruction.cameras),
        "dense_images": len(reconstruction.inferred_images),
        "absent_from_main_model": len(reconstruction.unregistered_images),
        "sparse_points": len(tracks),
        "sparse_observations": len(errors),
        "consumer_reprojection_rmse_px": rmse,
        "max_normalized_projection_roundtrip_error": max_inverse_error,
        "max_virtual_projection_error_px": max_virtual_error,
        "existing_multiview_sampled_tracks": len(sampled),
        "existing_multiview_finite_results": len(deltas),
        "existing_multiview_median_point_difference_output_units": statistics.median(deltas),
        "semantic_entities_created": len(model.entities),
        "mesh_triangles": report["stages"]["fusion"]["mesh_triangles"],
        "mesh_provenance": report["stages"]["fusion"]["mesh_provenance"],
        "timings_seconds": report["timings_seconds"],
        "peak_process_tree_rss_mib": report["resource_usage"]["peak_process_tree_rss_mb"],
        "dense_cross_view": {key: value for key, value in
                             report["verification"]["dense_cross_view_depth_abs_rel"].items()
                             if key != "pairs"},
        "consumer_check_seconds": round(time.perf_counter() - started, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "python": platform.python_version(),
        "platform": platform.system(),
        "dependencies": {name: importlib.metadata.version(name) for name in
                         ("numpy", "Pillow", "open3d", "pycolmap", "scipy", "psutil")},
        "runs": [check_run(directory) for directory in args.run],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
