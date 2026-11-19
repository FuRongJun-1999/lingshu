"""Explicit content checks for CPU reconstruction artifacts.

PLY/SQLite decoding uses the standard library. NumPy and OpenCV are imported
only by ``validate_artifacts``: ordinary metadata loading remains lightweight.
This checks file contents and bundle consistency, not reconstruction accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import closing
from itertools import zip_longest
import math
from pathlib import Path
import sqlite3
import struct


_TYPES = {
    "char": ("b", -128, 127), "int8": ("b", -128, 127),
    "uchar": ("B", 0, 255), "uint8": ("B", 0, 255),
    "short": ("h", -32768, 32767), "int16": ("h", -32768, 32767),
    "ushort": ("H", 0, 65535), "uint16": ("H", 0, 65535),
    "int": ("i", -(2**31), 2**31-1), "int32": ("i", -(2**31), 2**31-1),
    "uint": ("I", 0, 2**32-1), "uint32": ("I", 0, 2**32-1),
    "float": ("f", None, None), "float32": ("f", None, None),
    "double": ("d", None, None), "float64": ("d", None, None),
}


@dataclass
class PlyGeometry:
    vertices: list
    faces: list
    colors: list


def read_ply(path, *, mesh=False):
    """Read finite ASCII/little- or big-endian PLY with triangular mesh faces.

    Fail closed on unsupported properties, truncated payloads and out-of-range
    indices. Vertex colors are optional (neutral gray when absent). Files are
    read anew so rendering never relies on a previous validation of mutable data.
    """
    path = Path(path)
    try:
        file_size = path.stat().st_size
        with path.open("rb") as stream:
            if stream.readline() != b"ply\n":
                stream.seek(0)
                if stream.readline().rstrip(b"\r\n") != b"ply":
                    raise ValueError("missing PLY signature")
            elements, current, fmt, header_bytes = [], None, None, 0
            while True:
                raw = stream.readline(65537)
                header_bytes += len(raw)
                if not raw or header_bytes > 65536:
                    raise ValueError("missing or oversized PLY header")
                tokens = raw.decode("ascii").split()
                if not tokens or tokens[0] in ("comment", "obj_info"):
                    continue
                if tokens == ["end_header"]:
                    break
                if tokens[0] == "format" and len(tokens) == 3:
                    if fmt is not None or tokens[2] != "1.0" or tokens[1] not in (
                            "ascii", "binary_little_endian", "binary_big_endian"):
                        raise ValueError("unsupported PLY format")
                    fmt = tokens[1]
                elif tokens[0] == "element" and len(tokens) == 3:
                    count = int(tokens[2])
                    if count < 0 or count > file_size:
                        raise ValueError("invalid PLY element count")
                    if tokens[1] not in ("vertex", "face") or any(e[0] == tokens[1] for e in elements):
                        raise ValueError("unsupported or duplicate PLY element")
                    current = [tokens[1], count, []]
                    elements.append(current)
                elif tokens[0] == "property" and current is not None:
                    if len(tokens) == 3 and tokens[1] in _TYPES:
                        prop = (tokens[2], tokens[1], None)
                    elif (len(tokens) == 5 and tokens[1] == "list" and tokens[2] in _TYPES
                          and _TYPES[tokens[2]][1] is not None and tokens[3] in _TYPES):
                        prop = (tokens[4], tokens[3], tokens[2])
                    else:
                        raise ValueError("unsupported PLY property")
                    if any(p[0] == prop[0] for p in current[2]):
                        raise ValueError("duplicate PLY property")
                    current[2].append(prop)
                else:
                    raise ValueError("invalid PLY header")
            if fmt is None or not elements or elements[0][0] != "vertex":
                raise ValueError("PLY needs vertices before faces")
            vprops = {name: (kind, count) for name, kind, count in elements[0][2]}
            if not all(name in vprops and vprops[name][1] is None for name in ("x", "y", "z")):
                raise ValueError("PLY needs scalar x/y/z")
            color_names = ("red", "green", "blue")
            if any(name in vprops for name in color_names) and not all(
                    name in vprops and vprops[name][1] is None for name in color_names):
                raise ValueError("PLY colors need scalar red/green/blue")
            if mesh and (len(elements) != 2 or elements[1][0] != "face"):
                raise ValueError("mesh PLY needs faces")
            vertices, faces, colors = [], [], []
            endian = "<" if fmt == "binary_little_endian" else ">"
            for element, count, properties in elements:
                if not properties:
                    raise ValueError("PLY element needs properties")
                for _ in range(count):
                    tokens = iter(stream.readline().decode("ascii").split()) if fmt == "ascii" else None

                    def scalar(kind):
                        code, low, high = _TYPES[kind]
                        if tokens is not None:
                            token = next(tokens)
                            value = float(token) if low is None else int(token)
                        else:
                            value = struct.unpack(endian + code, stream.read(struct.calcsize(code)))[0]
                        if not math.isfinite(value) or (low is not None and not low <= value <= high):
                            raise ValueError("non-finite or out-of-range PLY value")
                        return value

                    row = {}
                    for name, kind, length_kind in properties:
                        if length_kind is None:
                            row[name] = scalar(kind)
                        else:
                            length = scalar(length_kind)
                            if not 0 <= length <= 1_000_000 or length > file_size:
                                raise ValueError("invalid PLY list length")
                            row[name] = [scalar(kind) for _ in range(length)]
                    if tokens is not None and next(tokens, None) is not None:
                        raise ValueError("extra PLY row values")
                    if element == "vertex":
                        vertices.append(tuple(row[name] for name in ("x", "y", "z")))
                        color = tuple(row[name] for name in color_names) if "red" in row else (180, 180, 180)
                        if any(type(c) is not int or not 0 <= c <= 255 for c in color):
                            raise ValueError("PLY color must be an integer in [0,255]")
                        colors.append(color)
                    else:
                        indices = row.get("vertex_indices", row.get("vertex_index"))
                        if (not isinstance(indices, list) or len(indices) != 3
                                or any(type(i) is not int or not 0 <= i < len(vertices) for i in indices)
                                or len(set(indices)) != 3):
                            raise ValueError("PLY face needs three distinct valid vertex indices")
                        a, b, c = (vertices[i] for i in indices)
                        u, v = (tuple(p-q for p, q in zip(point, a)) for point in (b, c))
                        cross = (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
                        if not any(cross) or not all(math.isfinite(value) for value in cross):
                            raise ValueError("PLY face has degenerate or non-finite area")
                        faces.append(tuple(indices))
            if stream.read().strip():
                raise ValueError("extra PLY payload")
            if not vertices or (mesh and not faces):
                raise ValueError("PLY geometry must be nonempty")
            return PlyGeometry(vertices, faces, colors)
    except (OSError, UnicodeError, StopIteration, struct.error, OverflowError, ValueError) as exc:
        raise ValueError(f"invalid PLY {path}: {exc}") from exc


def _depth_archive(path, *, raw):
    import numpy as np

    try:
        with np.load(path, allow_pickle=False) as archive:
            if not {"depth", "confidence"} <= set(archive.files):
                raise ValueError("missing depth/confidence arrays")
            arrays = {key: archive[key] for key in archive.files}
            for key, value in arrays.items():
                kinds = "biu" if key == "input_valid_mask" else "fiu"
                if value.dtype.kind not in kinds or not np.isfinite(value).all():
                    raise ValueError(f"{key} must contain finite real numeric values")
            depth, confidence = arrays["depth"], arrays["confidence"]
            if (depth.ndim != 2 or 0 in depth.shape or confidence.shape != depth.shape
                    or (depth < 0).any() or not (depth > 0).any()):
                raise ValueError("depth/confidence need matching nonempty HxW shapes and positive depth support")
            input_mask = arrays.get("input_valid_mask")
            if input_mask is not None and (input_mask.shape != depth.shape
                    or not ((input_mask == 0) | (input_mask == 1)).all()
                    or not ((depth > 0) & (input_mask != 0)).any()):
                raise ValueError("input_valid_mask needs matching HxW binary values and valid positive depth support")
            if raw:
                flag = arrays.get("is_metric")
                if flag is None or flag.shape != () or flag.item() not in (0, 1):
                    raise ValueError("raw depth needs scalar is_metric 0/1")
            for key, shapes in (("intrinsics", {(3, 3)}), ("extrinsics", {(3, 4), (4, 4)})):
                if key in arrays and arrays[key].shape not in shapes:
                    raise ValueError(f"invalid {key} shape")
            summary = {"shape": list(depth.shape), "positive_pixels": int((depth > 0).sum()),
                       "minimum_positive": float(depth[depth > 0].min()), "maximum": float(depth.max())}
            if input_mask is not None:
                summary["input_valid_pixels"] = int((input_mask != 0).sum())
            return summary
    except Exception as exc:
        raise ValueError(f"invalid NPZ {path}: {exc}") from exc


def validate_artifacts(reconstruction):
    """Decode every native product; expensive opt-in validation, no state changes.

    Checks all normalized inputs, registered undistorted images, inferred raw and
    aligned depth, both PLYs, SQLite integrity/schema/content and provenance
    correspondence. Zero depth is a mask; optional input_valid_mask is binary
    in the depth pixel domain; confidence is a finite model score, not a
    probability. This is intentionally separate from metadata-only load.
    """
    import cv2
    import numpy as np
    from .reconstruction import _decode, _file, _json

    recon = reconstruction
    mesh = read_ply(recon.artifacts["mesh"], mesh=True)
    cloud = read_ply(recon.artifacts["cloud"])
    fusion = recon.report["stages"]["fusion"]
    if (len(mesh.vertices) != fusion["mesh_vertices"] or len(mesh.faces) != fusion["mesh_triangles"]
            or len(cloud.vertices) != fusion["valid_voxels"]):
        raise ValueError("PLY geometry counts differ from report")
    for vertex, provenance in zip_longest(mesh.vertices, recon.iter_mesh_provenance()):
        if (vertex is None or provenance is None or any(not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
                for a, b in zip(vertex, provenance["xyz"]))):
            raise ValueError("mesh PLY vertex differs from provenance")
    images = 0
    for frame in recon.manifest:
        paths = [(f"00_input/images/{frame['name']}", (frame["width"], frame["height"]))]
        camera = recon.cameras.get(frame["name"])
        if camera is not None:
            paths.append((f"02_undistorted/images/{frame['name']}", (camera.width, camera.height)))
        for relative, size in paths:
            path = _file(recon.root, relative)
            try:
                decoded = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                if (decoded is None or decoded.ndim not in (2, 3)
                        or (decoded.ndim == 3 and decoded.shape[2] not in (1, 3, 4))):
                    raise ValueError("image does not decode to a supported pixel array")
                actual_size = (decoded.shape[1], decoded.shape[0])
                if actual_size != size:
                    raise ValueError(f"image dimensions {actual_size} differ from {size}")
            except Exception as exc:
                raise ValueError(f"invalid image {path}: {exc}") from exc
            images += 1
    depths = {}
    for name in recon.inferred_images:
        stem = Path(name).stem
        raw = _depth_archive(_file(recon.root, f"03_depth/{stem}.npz"), raw=True)
        aligned = _depth_archive(_file(recon.root, f"04_alignment/depth_aligned/{stem}.npz"), raw=False)
        if raw["shape"] != aligned["shape"]:
            raise ValueError("raw and aligned depth dimensions differ")
        depths[name] = {"raw": raw, "aligned": aligned}
    poses = _json(recon.artifacts["poses"])
    path = recon.artifacts["provenance_sqlite"]
    expected = {"ix": "INTEGER", "iy": "INTEGER", "iz": "INTEGER", "sx": "REAL", "sy": "REAL",
                "sz": "REAL", "sr": "INTEGER", "sg": "INTEGER", "sb": "INTEGER",
                "observations": "INTEGER", "sources": "TEXT"}
    count = 0
    try:
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
            if connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                raise ValueError("SQLite integrity_check failed")
            columns = connection.execute("PRAGMA table_info(voxels)").fetchall()
            if ({row[1]: row[2].upper() for row in columns} != expected
                    or any(not row[3] for row in columns)
                    or {row[1]: row[5] for row in columns if row[5]} != {"ix": 1, "iy": 2, "iz": 3}):
                raise ValueError("SQLite voxels schema differs from producer schema")
            rows = connection.execute("SELECT ix,iy,iz,sx,sy,sz,sr,sg,sb,observations,sources FROM voxels ORDER BY ix,iy,iz")
            for row, provenance in zip_longest(rows, recon.iter_voxel_provenance()):
                if row is None or provenance is None:
                    raise ValueError("SQLite/JSONL voxel counts differ")
                obs = row[9]
                if (any(type(v) is not int for v in row[:3]) or type(obs) is not int or obs < 1
                        or any(not isinstance(v, (float, int)) or not math.isfinite(v) for v in row[3:6])
                        or any(type(v) is not int or not 0 <= v <= 255*obs for v in row[6:9])):
                    raise ValueError("invalid SQLite voxel values")
                sources = _decode(row[10], "SQLite sources")
                recon._sources(sources)
                if (list(row[:3]) != provenance["voxel"] or obs != provenance["observations"]
                        or sources != provenance["sources"] or len(sources) > obs):
                    raise ValueError("SQLite voxel differs from JSONL provenance")
                if count >= len(cloud.vertices) or any(not math.isclose(cloud.vertices[count][axis], row[3+axis]/obs,
                                        abs_tol=5.1e-7, rel_tol=1e-12) for axis in range(3)):
                    raise ValueError("cloud PLY coordinate differs from SQLite voxel average")
                for source in sources:
                    name = source["image"]
                    height, width = depths[name]["aligned"]["shape"]
                    x, y = source["pixel_xy"]
                    if (any(type(v) is not int for v in (x, y)) or not 0 <= x < width or not 0 <= y < height
                            or source.get("image_id") != poses[name]["image_id"]):
                        raise ValueError("SQLite source pixel or image identity differs from depth/pose")
                count += 1
    except (sqlite3.Error, TypeError) as exc:
        raise ValueError(f"invalid SQLite {path}: {exc}") from exc
    if count != fusion["valid_voxels"]:
        raise ValueError("SQLite voxel count differs from report")
    return {"validation": "decoded artifact content and cross-file consistency",
            "mesh_vertices": len(mesh.vertices), "mesh_triangles": len(mesh.faces),
            "cloud_points": len(cloud.vertices), "sqlite_voxels": count,
            "decoded_images": images, "depth_frames": depths,
            "geometric_accuracy": "not evaluated by content validation"}
