from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from .alignment import depth_valid_mask

def _transform_points(depth: np.ndarray, confidence: np.ndarray, image: np.ndarray, pose: dict,
                     coordinate_scale: float, voxel_size: float, confidence_percentile: float,
                     max_depth_units: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _voxelize(*_prepare_frame(depth, confidence, image, pose, coordinate_scale,
                                    confidence_percentile, max_depth_units), voxel_size)


def _prepare_frame(depth, confidence, image, pose, coordinate_scale, confidence_percentile, max_depth_units,
                   input_valid_mask=None):
    height, width = depth.shape
    color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if color.shape[:2] != depth.shape:
        color = cv2.resize(color, (width, height), interpolation=cv2.INTER_AREA)

    confidence_domain = np.isfinite(confidence)
    if input_valid_mask is not None:
        confidence_domain &= input_valid_mask
    valid_conf = confidence[confidence_domain]
    if not valid_conf.size:
        raise RuntimeError(f"DA3 returned no finite confidence values for {pose['name']}")
    threshold = float(np.percentile(valid_conf, confidence_percentile))
    valid = np.isfinite(depth) & (depth > 0) & (depth < max_depth_units) & np.isfinite(confidence) & (confidence >= threshold)
    if input_valid_mask is not None:
        valid &= input_valid_mask
    image_scale = np.diag((width / pose["width"], height / pose["height"], 1.0))
    intrinsics = image_scale @ np.asarray(pose["intrinsics"], dtype=np.float64)
    # COLMAP's continuous centers are i+.5, whereas Open3D depth raster
    # indices are i. Shift only the internal raster K, never exported K.
    intrinsics[:2, 2] -= 0.5
    original_pose = np.asarray(pose["cam_from_world"], dtype=np.float64)
    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3, :3] = original_pose[:, :3]
    world_to_camera[:3, 3] = coordinate_scale * original_pose[:, 3]
    return np.where(valid, depth, 0), color, intrinsics, world_to_camera


def _voxelize(depth, color, intrinsics, world_to_camera, voxel_size):
    width = depth.shape[1]
    ys, xs = np.nonzero(depth)
    z = depth[ys, xs].astype(np.float64)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    camera_points = np.column_stack(((xs - cx) * z / fx, (ys - cy) * z / fy, z))

    world_points = (camera_points - world_to_camera[:3, 3]) @ world_to_camera[:3, :3]
    rgb = color[ys, xs]

    voxel_keys = np.floor(world_points / voxel_size).astype(np.int32)
    # Pack coordinates relative to this frame's origin range for a single integer sort.
    if len(voxel_keys):
        lo = voxel_keys.min(axis=0).astype(np.int64)
        sizes = voxel_keys.max(axis=0).astype(np.int64) - lo + 1
    if len(voxel_keys) and int(sizes[0]) * int(sizes[1]) * int(sizes[2]) <= 2**63:
        shifted = voxel_keys - lo
        flat = (shifted[:, 0] * sizes[1] + shifted[:, 1]) * sizes[2] + shifted[:, 2]
        _, first_positions, inverse = np.unique(flat, return_index=True, return_inverse=True)
        unique_keys = voxel_keys[first_positions]
    else:
        unique_keys, first_positions, inverse = np.unique(
            voxel_keys, axis=0, return_index=True, return_inverse=True
        )
    counts = np.bincount(inverse).astype(np.int64)
    xyz_sums = np.column_stack([np.bincount(inverse, weights=world_points[:, axis]) for axis in range(3)])
    rgb_sums = np.column_stack([np.bincount(inverse, weights=rgb[:, axis]) for axis in range(3)])
    xyz_means = xyz_sums / counts[:, None]
    rgb_means = np.clip(np.rint(rgb_sums / counts[:, None]), 0, 255).astype(np.uint8)
    pixel_indices = ys[first_positions] * width + xs[first_positions]
    return xyz_means, rgb_means, unique_keys, pixel_indices


class VoxelProvenance:
    def __init__(self, path: Path):
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("""
            CREATE TABLE voxels (
                ix INTEGER NOT NULL, iy INTEGER NOT NULL, iz INTEGER NOT NULL,
                sx REAL NOT NULL, sy REAL NOT NULL, sz REAL NOT NULL,
                sr INTEGER NOT NULL, sg INTEGER NOT NULL, sb INTEGER NOT NULL,
                observations INTEGER NOT NULL,
                sources TEXT NOT NULL,
                PRIMARY KEY(ix, iy, iz)
            )
        """)

    def add_frame(self, points: np.ndarray, colors: np.ndarray, keys: np.ndarray,
                  pixels: np.ndarray, frame_record: dict, width: int) -> None:
        prefix = json.dumps({
            "frame": frame_record["source"], "image": frame_record["name"], "image_id": frame_record["image_id"],
        }, ensure_ascii=False, separators=(",", ":"))[:-1]

        def rows():
            for point, color, key, pixel in zip(points.tolist(), colors.tolist(), keys.tolist(), pixels.tolist()):
                x, y = pixel % width, pixel // width
                source = prefix + f',"pixel_xy":[{x},{y}]' + "}"
                yield (*key, *point, *color, 1, "[" + source + "]", source, source)
        self.db.executemany("""
            INSERT INTO voxels VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ix, iy, iz) DO UPDATE SET
                sx=sx+excluded.sx, sy=sy+excluded.sy, sz=sz+excluded.sz,
                sr=sr+excluded.sr, sg=sg+excluded.sg, sb=sb+excluded.sb,
                observations=observations+1,
                sources=CASE WHEN instr(sources, ?) > 0 THEN sources
                    ELSE substr(sources, 1, length(sources)-1) || ',' || ? || ']' END
        """, rows())

    def close(self) -> None:
        self.db.commit()
        self.db.close()

    def __enter__(self) -> VoxelProvenance:
        return self

    def __exit__(self, exc_type, *_exc) -> None:
        if exc_type is None:
            self.close()
        else:
            self.db.close()


def _write_voxel_cloud(provenance_path: Path, cloud_path: Path, jsonl_path: Path,
                       voxel_size: float, units: str) -> int:
    connection = sqlite3.connect(provenance_path)
    count = connection.execute("SELECT COUNT(*) FROM voxels").fetchone()[0]
    with cloud_path.open("w", encoding="ascii", newline="\n") as ply, jsonl_path.open("w", encoding="utf-8", newline="\n") as provenance:
        ply.write("ply\nformat ascii 1.0\n")
        ply.write(f"element vertex {count}\nproperty float x\nproperty float y\nproperty float z\n")
        ply.write("property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uint observation_count\nend_header\n")
        cursor = connection.execute("SELECT ix,iy,iz,sx,sy,sz,sr,sg,sb,observations,sources FROM voxels ORDER BY ix,iy,iz")
        for ix, iy, iz, sx, sy, sz, sr, sg, sb, observations, sources in cursor:
            ply.write(f"{sx / observations:.6f} {sy / observations:.6f} {sz / observations:.6f} "
                      f"{round(sr / observations)} {round(sg / observations)} {round(sb / observations)} {observations}\n")
            prefix = json.dumps({
                "voxel": [ix, iy, iz],
                "center": [(ix + 0.5) * voxel_size, (iy + 0.5) * voxel_size, (iz + 0.5) * voxel_size],
                "units": units,
                "observations": observations,
            }, ensure_ascii=False, separators=(",", ":"))[:-1]
            provenance.write(prefix + ',"sources":' + sources + "}\n")
    connection.close()
    return count


def _write_mesh_provenance(vertices: np.ndarray, provenance_path: Path, jsonl_path: Path,
                           voxel_size: float, units: str) -> dict:
    connection = sqlite3.connect(provenance_path)
    count = int(connection.execute("SELECT COUNT(*) FROM voxels").fetchone()[0])
    rowids = np.empty(count, dtype=np.int64)
    points = np.empty((count, 3), dtype=np.float32)
    cursor = connection.execute("SELECT rowid,sx,sy,sz,observations FROM voxels")
    offset = 0
    while rows := cursor.fetchmany(8192):
        end = offset + len(rows)
        chunk = np.asarray(rows, dtype=np.float64)
        rowids[offset:end] = chunk[:, 0].astype(np.int64)
        points[offset:end] = chunk[:, 1:4] / chunk[:, 4:5]
        offset = end

    if count and len(vertices):
        distances, indices = cKDTree(points).query(
            np.asarray(vertices, dtype=np.float32), k=1, distance_upper_bound=voxel_size * 4
        )
        mapped = np.isfinite(distances) & (indices < count)
    else:
        distances = np.full(len(vertices), np.inf, dtype=np.float64)
        indices = np.full(len(vertices), count, dtype=np.int64)
        mapped = np.zeros(len(vertices), dtype=bool)

    selected_rowids = np.unique(rowids[indices[mapped]]) if mapped.any() else np.empty(0, dtype=np.int64)
    source_by_rowid = {}
    for start in range(0, len(selected_rowids), 500):
        batch = selected_rowids[start:start + 500].tolist()
        placeholders = ",".join("?" for _ in batch)
        for rowid, ix, iy, iz, observations, sources in connection.execute(
            f"SELECT rowid,ix,iy,iz,observations,sources FROM voxels WHERE rowid IN ({placeholders})", batch
        ):
            source_by_rowid[rowid] = {
                "voxel": [ix, iy, iz],
                "observations": observations,
                "sources": json.loads(sources),
            }

    with jsonl_path.open("w", encoding="utf-8", newline="\n") as stream:
        for index, vertex in enumerate(vertices):
            record = {
                "mesh_vertex": index,
                "xyz": np.asarray(vertex, dtype=np.float64).tolist(),
                "units": units,
                "source_mapping": "nearest observed depth voxel within four TSDF voxels",
                "nearest_distance": float(distances[index]) if mapped[index] else None,
                "source": source_by_rowid.get(int(rowids[indices[index]])) if mapped[index] else None,
            }
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    connection.close()
    return {
        "mesh_vertices": int(len(vertices)),
        "mapped_vertices": int(mapped.sum()),
        "unmapped_vertices": int((~mapped).sum()),
        "mapping_ratio": float(mapped.mean()) if len(vertices) else 1.0,
        "max_mapping_radius": float(voxel_size * 4),
        "max_mapped_distance": float(distances[mapped].max()) if mapped.any() else None,
        "p95_mapped_distance": float(np.quantile(distances[mapped], 0.95)) if mapped.any() else None,
        "method": "nearest observed depth voxel within four TSDF voxels",
    }


def fuse_scene(poses: dict, depth_dir: Path, images_dir: Path, output_dir: Path,
               coordinate_scale: float, voxel_size: float, confidence_percentile: float,
               max_depth_units: float, units: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    max_depth_units = float(max_depth_units)
    tensor = o3d.core.Tensor
    f32 = o3d.core.Dtype.Float32
    mesh_volume = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(f32, f32, f32),
        attr_channels=((1,), (1,), (3,)),
        voxel_size=voxel_size, block_resolution=16, block_count=128,
        device=o3d.core.Device("CPU:0"),
    )
    provenance_path = output_dir / "voxel_provenance.sqlite"
    integrated_frames = 0
    valid_points = 0
    with VoxelProvenance(provenance_path) as provenance:
        for name, pose_record in sorted(poses.items(), key=lambda pair: pair[1]["source_index"]):
            depth_file = depth_dir / f"{Path(name).stem}.npz"
            image_file = images_dir / name
            if not depth_file.exists():
                continue
            if not image_file.is_file():
                raise RuntimeError(f"aligned depth frame {name} is missing its undistorted image: {image_file}")
            with np.load(depth_file) as record:
                depth = record["depth"].astype(np.float32)
                confidence = record["confidence"].astype(np.float32)
                input_mask = depth_valid_mask(record, depth.shape)
            image = cv2.imread(str(image_file), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"cannot decode undistorted image for aligned depth frame {name}")
            pose = {**pose_record, "name": name}
            depth_for_fusion, rgb, intrinsic_matrix, world_to_camera = _prepare_frame(
                depth, confidence, image, pose, coordinate_scale, confidence_percentile, max_depth_units,
                input_valid_mask=input_mask,
            )
            points, colors, keys, pixels = _voxelize(depth_for_fusion, rgb, intrinsic_matrix, world_to_camera, voxel_size)
            if not len(points):
                raise RuntimeError(f"aligned depth frame {name} has no valid points after filtering")
            valid_points += len(points)
            provenance.add_frame(points, colors, keys, pixels, {
                "source": pose_record["source"], "name": name, "image_id": pose_record["image_id"],
            }, depth.shape[1])

            depth_o3d = o3d.t.geometry.Image(tensor.from_numpy(depth_for_fusion))
            rgb_o3d = o3d.t.geometry.Image(tensor.from_numpy(rgb.astype(np.float32) / 255.0))
            intrinsic, extrinsic = tensor.from_numpy(intrinsic_matrix), tensor.from_numpy(world_to_camera)
            blocks = mesh_volume.compute_unique_block_coordinates(depth_o3d, intrinsic, extrinsic, 1.0, max_depth_units, 4.0)
            if not blocks.shape[0]:
                raise RuntimeError(f"CPU TSDF found no blocks for aligned depth frame {name}")
            mesh_volume.integrate(blocks, depth_o3d, rgb_o3d, intrinsic, extrinsic, 1.0, max_depth_units, 4.0)
            integrated_frames += 1
    if not integrated_frames:
        raise RuntimeError("no registered views were integrated into the CPU TSDF")
    # Open3D rejects a marching-cubes cell when any corner's weight is <=
    # this threshold. Zero therefore excludes unobserved space while keeping
    # valid single-observation boundaries, even when an overlap mesh exists.
    mesh_weight_threshold = 0.0
    mesh = mesh_volume.extract_triangle_mesh(weight_threshold=mesh_weight_threshold).to_legacy()
    if not len(mesh.triangles):
        raise RuntimeError("CPU TSDF produced no surface from the filtered aligned depth maps")
    if not o3d.io.write_triangle_mesh(str(output_dir / "mesh.ply"), mesh):
        raise RuntimeError("Open3D failed to write fused mesh")
    provenance_count = _write_voxel_cloud(
        provenance_path, output_dir / "cloud.ply", output_dir / "provenance.jsonl", voxel_size, units
    )
    mesh_provenance = _write_mesh_provenance(
        np.asarray(mesh.vertices), provenance_path, output_dir / "mesh_provenance.jsonl", voxel_size, units
    )
    return {
        "integrated_frames": integrated_frames,
        "backend": "Open3D CPU VoxelBlockGrid",
        "active_voxel_blocks": int(mesh_volume.hashmap().size()),
        "mesh_weight_threshold": mesh_weight_threshold,
        "mesh_support": {
            "minimum_voxel_observations": 1,
            "unknown_voxels_excluded": True,
            "single_view_surfaces_retained": True,
            "tradeoff": "single-view depth noise is retained with its observation provenance",
        },
        "valid_voxels": provenance_count,
        "source_voxel_samples": valid_points,
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_triangles": int(len(mesh.triangles)),
        "mesh_provenance": mesh_provenance,
        "voxel_size": float(voxel_size),
        "units": units,
        "confidence_percentile": float(confidence_percentile),
        "files": {
            "mesh": str((output_dir / "mesh.ply").resolve()),
            "cloud": str((output_dir / "cloud.ply").resolve()),
            "provenance_jsonl": str((output_dir / "provenance.jsonl").resolve()),
            "provenance_sqlite": str(provenance_path.resolve()),
            "mesh_provenance_jsonl": str((output_dir / "mesh_provenance.jsonl").resolve()),
        },
    }
