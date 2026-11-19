from __future__ import annotations

from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from time import perf_counter

import numpy as np
import pycolmap

from .config import positive_int


def run_colmap(executable: Path, command: str, args: list[str], log_path: Path, cwd: Path) -> None:
    invocation = [str(executable.resolve()), command, *args]
    env = os.environ.copy()
    thread_args = [args[index + 1] for index, value in enumerate(args[:-1])
                   if value.endswith("num_threads")]
    threads = positive_int(int(thread_args[0]), "threads") if thread_args else 2
    env["OMP_NUM_THREADS"] = str(threads)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(invocation, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    if proc.returncode:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
        raise RuntimeError(f"COLMAP {command} failed ({proc.returncode}). Log: {log_path}\n{tail}")


def run_batched_matching(database: Path, logs: Path, threads: int) -> None:
    positive_int(threads, "threads")
    log_path = logs / "sequential_matcher.log"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run([sys.executable, "-m", "lingshu.world.reconstruct.match_pairs", str(database.resolve()),
                               str((logs / "image_pairs.txt").resolve()), str(threads)],
                              stdout=log, stderr=subprocess.STDOUT, check=False)
    if proc.returncode:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
        raise RuntimeError(f"Batched COLMAP matching failed ({proc.returncode}). Log: {log_path}\n{tail}")


_FEATURE_TABLES = ("keypoints", "descriptors")
_LARGE_TABLES = {*_FEATURE_TABLES, "matches", "two_view_geometries"}


def _quoted(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _open_database(path: Path, mode: str = "ro") -> sqlite3.Connection:
    return sqlite3.connect(path.resolve().as_uri() + f"?mode={mode}", uri=True)


def _feature_counts(db: sqlite3.Connection, selected: set[int] | None = None) -> list[tuple]:
    records = []
    for image_id, name, points, descriptors in db.execute(
        "SELECT i.image_id,i.name,k.rows,d.rows FROM images AS i "
        "LEFT JOIN keypoints AS k ON k.image_id=i.image_id "
        "LEFT JOIN descriptors AS d ON d.image_id=i.image_id ORDER BY i.image_id"
    ):
        if selected is not None and image_id not in selected:
            continue
        if points is None or descriptors is None or points != descriptors or points < 0:
            raise RuntimeError(f"incomplete or inconsistent SIFT records for {name}")
        records.append((image_id, name, points))
    return records


def _require_unmatched(db: sqlite3.Connection) -> None:
    if any(db.execute(f"SELECT 1 FROM {_quoted(table)} LIMIT 1").fetchone()
           for table in ("matches", "two_view_geometries")):
        raise RuntimeError("SIFT recovery refuses a database containing match records")


def _identity(db: sqlite3.Connection) -> tuple:
    return tuple((table, tuple(db.execute(f"SELECT * FROM {_quoted(table)} ORDER BY 1")))
                 for table in ("cameras", "images"))


def _feature_signature(db: sqlite3.Connection, image_ids: list[int]) -> str:
    digest = hashlib.sha256()
    for table in _FEATURE_TABLES:
        for image_id in image_ids:
            row = db.execute(f"SELECT * FROM {_quoted(table)} WHERE image_id=?", (image_id,)).fetchone()
            if row is None:
                raise RuntimeError(f"missing {table} for image {image_id}")
            digest.update(table.encode())
            for value in row:
                data = value if isinstance(value, bytes) else repr(value).encode()
                digest.update(len(data).to_bytes(8, "little"))
                digest.update(data)
    return digest.hexdigest()


def _metadata_scratch(source: sqlite3.Connection, destination: Path) -> None:
    """Copy native schema and identities without copying large feature blobs."""
    schema = source.execute("SELECT type,name,sql FROM sqlite_master "
                            "WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%'").fetchall()
    with closing(sqlite3.connect(destination)) as target, target:
        for kind, name, sql in schema:
            if kind == "table":
                target.execute(sql)
        for kind, name, _ in schema:
            if kind != "table" or name in _LARGE_TABLES:
                continue
            cursor = source.execute(f"SELECT * FROM {_quoted(name)}")
            placeholders = ",".join("?" for _ in cursor.description)
            while rows := cursor.fetchmany(256):
                target.executemany(f"INSERT INTO {_quoted(name)} VALUES({placeholders})", rows)
        for kind, _, sql in schema:
            if kind != "table":
                target.execute(sql)


def _recover_weak_sift(executable: Path, database: Path, logs: Path, feature_args: list[str], *,
                       database_preexisting: bool) -> dict:
    """Retry weak new captures once, importing only completed scratch features."""
    started = perf_counter()
    record = {"status": "checking", "minimum_features": 512,
              "initial_peak_threshold": 1 / 150, "recovery_peak_threshold": 0.002,
              "retried_images": [], "initial_counts": {}, "recovered_counts": {},
              "scratch_metadata_only": True}
    try:
        with closing(_open_database(database)) as source:
            first = _feature_counts(source)
            record["initial_counts"] = {name: count for _, name, count in first}
            if database_preexisting:
                record["status"] = "skipped_preexisting_database"
                return record
            _require_unmatched(source)
            weak = [(image_id, name) for image_id, name, count in first if count < 512]
            if not weak:
                record["status"] = "not_needed"
                return record
            image_ids = [image_id for image_id, _ in weak]
            names = [name for _, name in weak]
            if any("\n" in name or "\r" in name for name in names):
                raise RuntimeError("SIFT image-list names cannot contain line breaks")
            record["retried_images"] = names
            original_identity = _identity(source)
            original_features = _feature_signature(source, image_ids)
            scratch_dir = logs / "sift_recovery"
            scratch_dir.mkdir(exist_ok=False)
            scratch = scratch_dir / "database.db"
            _metadata_scratch(source, scratch)
        image_list = scratch_dir / "images.txt"
        image_list.write_text("\n".join(names) + "\n", encoding="utf-8")
        args = list(feature_args)
        args[args.index("--database_path") + 1] = str(scratch.resolve())
        args += ["--image_list_path", str(image_list.resolve()), "--SiftExtraction.peak_threshold", "0.002"]
        run_colmap(executable, "feature_extractor", args, logs / "feature_recovery.log", logs.parent)
        with closing(_open_database(scratch)) as recovered:
            if _identity(recovered) != original_identity:
                raise RuntimeError("SIFT scratch camera/image identity changed")
            _require_unmatched(recovered)
            counts = _feature_counts(recovered, set(image_ids))
            for table in _FEATURE_TABLES:
                extracted = {row[0] for row in recovered.execute(f"SELECT image_id FROM {_quoted(table)}")}
                if extracted != set(image_ids):
                    raise RuntimeError("SIFT scratch features do not match the weak image list")
            record["recovered_counts"] = {name: count for _, name, count in counts}
        with closing(_open_database(database, "rw")) as target:
            target.execute("ATTACH DATABASE ? AS sift_recovery", (str(scratch.resolve()),))
            with target:
                target.execute("BEGIN IMMEDIATE")
                _require_unmatched(target)
                if (_identity(target) != original_identity or
                        _feature_signature(target, image_ids) != original_features):
                    raise RuntimeError("original camera/image identity or features changed during SIFT recovery")
                for table in _FEATURE_TABLES:
                    target.executemany(f"DELETE FROM {_quoted(table)} WHERE image_id=?", [(i,) for i in image_ids])
                    target.executemany(f"INSERT INTO {_quoted(table)} SELECT * FROM sift_recovery.{_quoted(table)} "
                                       "WHERE image_id=?", [(i,) for i in image_ids])
        record["status"] = "recovered"
        return record
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["seconds"] = round(perf_counter() - started, 6)
        (logs / "sift_support.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_matches(executable: Path, database: Path, images: Path, logs: Path, threads: int,
                    max_image_size: int, camera_model: str, *, sift_first_octave: int = -1,
                    matching_mode: str = "sequential", mask_path: Path | None = None) -> dict:
    logs.mkdir(parents=True, exist_ok=True)
    database_preexisting = database.exists()
    start = perf_counter()
    feature_args = [
        "--database_path", str(database), "--image_path", str(images),
        "--ImageReader.camera_model", camera_model, "--ImageReader.single_camera", "1",
        "--FeatureExtraction.use_gpu", "0", "--FeatureExtraction.num_threads", str(threads),
        "--FeatureExtraction.max_image_size", str(max_image_size), "--SiftExtraction.max_num_features", "8192",
        "--SiftExtraction.first_octave", str(sift_first_octave),
    ]
    if mask_path is not None:
        feature_args += ["--ImageReader.mask_path", str(mask_path.resolve())]
    run_colmap(executable, "feature_extractor", feature_args, logs / "feature_extractor.log", logs.parent)
    feature_support = _recover_weak_sift(executable, database, logs, feature_args,
                                         database_preexisting=database_preexisting)
    after_features = perf_counter()
    if matching_mode == "batched":
        run_batched_matching(database, logs, threads)
    elif matching_mode == "sequential":
        run_colmap(executable, "sequential_matcher", [
            "--database_path", str(database),
            "--FeatureMatching.use_gpu", "0", "--FeatureMatching.num_threads", str(threads),
            "--SequentialMatching.overlap", "10",
        ], logs / "sequential_matcher.log", logs.parent)
    elif matching_mode == "exhaustive":
        run_colmap(executable, "exhaustive_matcher", [
            "--database_path", str(database),
            "--FeatureMatching.use_gpu", "0", "--FeatureMatching.num_threads", str(threads),
        ], logs / "exhaustive_matcher.log", logs.parent)
    else:
        raise ValueError(f"unknown matching mode: {matching_mode}")
    return {"feature_seconds": round(after_features - start, 3),
            "matching_seconds": round(perf_counter() - after_features, 3),
            "feature_support": feature_support}


def map_global_scene(executable: Path, database: Path, images: Path, output: Path,
                     log: Path, threads: int) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    run_colmap(executable, "global_mapper", [
        "--database_path", str(database.resolve()), "--image_path", str(images.resolve()),
        "--output_path", str(output.resolve()), "--GlobalMapper.num_threads", str(threads),
        "--GlobalMapper.gp_use_gpu", "0", "--GlobalMapper.ba_ceres_use_gpu", "0",
    ], log, output.parent)
    components = describe_model_components(output)
    (output / "components.json").write_text(
        json.dumps(components, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output / components["selected_model"]


def describe_model_components(output_dir: Path) -> dict:
    """Inventory native models without implying a shared coordinate frame."""
    if (output_dir / "cameras.bin").exists():
        candidates = [output_dir]
    else:
        candidates = sorted(p for p in output_dir.iterdir()
                            if p.is_dir() and (p / "cameras.bin").exists())
    if not candidates:
        raise RuntimeError(f"COLMAP produced no registered model in {output_dir}")
    components = []
    for path in candidates:
        reconstruction = pycolmap.Reconstruction(str(path))
        components.append({
            "model_path": path.relative_to(output_dir).as_posix(),
            "registered_images": reconstruction.num_reg_images(),
            "sparse_points": reconstruction.num_points3D(),
            "image_names": sorted(image.name for image in reconstruction.images.values() if image.has_pose),
        })
    selected = max(components, key=lambda item: (item["registered_images"], item["sparse_points"]))
    for component in components:
        component["selected"] = component is selected
    return {
        "schema_version": 1,
        "coordinate_relationship": "independent_per_component",
        "selection_policy": "most_registered_images_then_sparse_points",
        "selected_model": selected["model_path"],
        "component_count": len(components),
        "all_component_registered_images": len({name for item in components for name in item["image_names"]}),
        "components": components,
    }


def choose_model(output_dir: Path) -> Path:
    return output_dir / describe_model_components(output_dir)["selected_model"]


def read_model(path: Path) -> tuple[pycolmap.Reconstruction, dict, list[dict]]:
    reconstruction = pycolmap.Reconstruction(str(path))
    manifest_candidates = []
    for parent in (path, *path.parents):
        manifest_candidates.extend((parent / "input_manifest.json", parent / "00_input" / "input_manifest.json"))
    manifest_path = next((candidate for candidate in manifest_candidates if candidate.exists()), None)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path else []
    source_by_name = {entry["name"]: entry for entry in manifest}
    poses = {}
    tracks = []
    projection_rows = {}
    registered_images = {}
    for image_id, image in reconstruction.images.items():
        if not image.has_pose:
            continue
        camera = reconstruction.camera(image.camera_id)
        pose = np.asarray(image.cam_from_world().matrix(), dtype=np.float64)
        projection_rows[image_id] = pose[2]
        registered_images[image_id] = image
        source = source_by_name.get(image.name, {"source": image.name, "index": int(image_id)})
        poses[image.name] = {
            "image_id": int(image_id),
            "camera_id": int(image.camera_id),
            "source": source["source"],
            "source_index": source["index"],
            "width": int(camera.width),
            "height": int(camera.height),
            "model": camera.model_name,
            "params": [float(v) for v in camera.params],
            "intrinsics": np.asarray(camera.calibration_matrix(), dtype=np.float64).tolist(),
            "cam_from_world": pose.tolist(),
        }
    for point_id, point in reconstruction.points3D.items():
        observations = []
        xyz = np.asarray(point.xyz, dtype=np.float64)
        for element in point.track.elements:
            image = registered_images.get(element.image_id)
            if image is None:
                continue
            pixel = np.asarray(image.point2D(element.point2D_idx).xy, dtype=np.float64)
            row = projection_rows[element.image_id]
            observations.append({
                "image": image.name,
                "image_id": int(element.image_id),
                "pixel_xy": pixel.tolist(),
                "camera_z": float(row[:3] @ xyz + row[3]),
            })
        if observations:
            tracks.append({
                "point3D_id": int(point_id),
                "xyz": xyz.tolist(),
                "rgb": [int(v) for v in point.color],
                "reprojection_error_px": float(point.error),
                "observations": observations,
            })
    return reconstruction, poses, tracks


def write_sparse_exports(path: Path, poses: dict, tracks: list[dict], out_dir: Path,
                         unregistered_images: list[str]) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "poses.json").write_text(json.dumps(poses, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "tracks.jsonl").open("w", encoding="utf-8") as stream:
        for track in tracks:
            stream.write(json.dumps(track, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = {
        "model_path": str(path.resolve()),
        "registered_images": len(poses),
        "sparse_points": len(tracks),
        "observations": sum(len(track["observations"]) for track in tracks),
        "unregistered_images": unregistered_images,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
