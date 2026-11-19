"""Check independent Blender camera/depth truth against its exported mesh.

This checks the truth assets only, before reconstruction. Open3D CPU raycasts
the exact Blender mesh with the declared K/extrinsics and pixel centers. Rays
have camera-forward z=1, so the returned hit parameter is camera-z in meters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"samples": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {"samples": int(values.size), "mean": float(values.mean()),
            "median": float(np.median(values)), "p95": float(np.quantile(values, .95)),
            "max": float(values.max())}


def camera_rays(camera):
    """Return world rays whose t parameter is camera-forward depth in meters."""
    pose = np.asarray(camera["cam_from_world"], dtype=np.float64)
    k = np.asarray(camera["intrinsics"], dtype=np.float64)
    height, width = camera["height"], camera["width"]
    ys, xs = np.mgrid[:height, :width]
    pixels = np.column_stack((xs.ravel() + .5, ys.ravel() + .5, np.ones(height * width)))
    directions = pixels @ np.linalg.inv(k).T @ pose[:, :3]
    center = -pose[:, :3].T @ pose[:, 3]
    origins = np.broadcast_to(center, directions.shape)
    return np.column_stack((origins, directions)).astype(np.float32).reshape(height, width, 6)


def validate_scene(scene, threads=2):
    import open3d as o3d

    scene = Path(scene)
    record = json.loads((scene / "truth/cameras.json").read_text(encoding="utf-8"))
    with np.load(scene / "truth/mesh.npz") as data:
        vertices, triangles = data["vertices"], data["triangles"]
    raycaster = o3d.t.geometry.RaycastingScene(nthreads=threads)
    raycaster.add_triangles(o3d.core.Tensor(np.asarray(vertices, dtype=np.float32)),
                           o3d.core.Tensor(np.asarray(triangles, dtype=np.uint32)))
    rows = []
    for camera in record["cameras"]:
        with np.load(scene / camera["depth"]) as data:
            rendered_depth, rendered_mask = data["depth_z"], data["mask"].astype(bool)
        hits = raycaster.cast_rays(o3d.core.Tensor(camera_rays(camera)), nthreads=threads)["t_hit"].numpy()
        hit_mask = np.isfinite(hits) & (hits > 0)
        rendered_mask &= np.isfinite(rendered_depth) & (rendered_depth > 0)
        domain = hit_mask & rendered_mask
        union = int(np.count_nonzero(hit_mask | rendered_mask))
        iou = float(np.count_nonzero(domain) / union) if union else 0.0
        error = _summary(np.abs(hits[domain] - rendered_depth[domain]))
        checks = {"foreground_iou_ge_0_995": iou >= .995,
                  "median_abs_z_error_le_1_mm": error["median"] is not None and error["median"] <= .001,
                  "p95_abs_z_error_le_1_cm": error["p95"] is not None and error["p95"] <= .01}
        rows.append({"image": camera["image"], "foreground_iou": iou,
                     "rendered_foreground_pixels": int(rendered_mask.sum()),
                     "mesh_hit_pixels": int(hit_mask.sum()), "abs_z_error_m": error,
                     "checks": checks, "passed": all(checks.values())})
    return {"status": "measured", "truth_source": record["truth_source"],
            "method": "Open3D CPU raycast of exact Blender-exported mesh using declared cameras; t is camera-z",
            "views": rows, "passed": bool(rows) and all(row["passed"] for row in rows)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="*")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    suite = json.loads((args.suite / "suite.json").read_text(encoding="utf-8"))
    cases = suite["scenes"]
    if args.cases is not None and not set(args.cases) <= {case["id"] for case in cases}:
        parser.error("unknown --cases ID")
    result = {"schema": "lingshu.independent-truth-self-check/1", "fixed_denominator": len(cases),
              "source": str(args.suite.resolve()), "scenes": []}
    for case in cases:
        row = {"id": case["id"], "status": "pending_not_run", "passed": False}
        if args.cases is None or case["id"] in args.cases:
            try:
                row.update(validate_scene(args.suite / case["id"], args.threads))
            except (OSError, KeyError, ValueError, RuntimeError) as error:
                row.update(status="failed", error=f"{type(error).__name__}: {error}")
            print(json.dumps({"id": case["id"], "status": row["status"], "passed": row["passed"]}), flush=True)
        result["scenes"].append(row)
    result["summary"] = {"measured": sum(row["status"] == "measured" for row in result["scenes"]),
                         "passed": sum(row["passed"] for row in result["scenes"]),
                         "fixed_denominator": len(cases)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(args.output)
    return 0 if result["summary"]["passed"] == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
