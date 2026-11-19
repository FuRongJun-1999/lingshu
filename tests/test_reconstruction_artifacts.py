"""Native artifact content corruption must fail an explicitly requested check."""
import json
import importlib.util
from pathlib import Path
import sqlite3
import struct

import pytest

from lingshu.world.reconstruction import Reconstruction
from lingshu.world.reconstruction_artifacts import read_ply

# Load this repository-owned helper by its controlled absolute path. This does
# not add tests/CWD to sys.path or exempt test_ names from dependency guards.
_spec = importlib.util.spec_from_file_location(
    "_reconstruction_metadata_fixture", Path(__file__).resolve().with_name("test_reconstruction_import.py"))
_metadata = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_metadata)
bundle = _metadata.bundle
report_at = _metadata.report_at
write_json = _metadata.write_json
write_rows = _metadata.write_rows


def write_ply(path, vertices, faces=(), *, binary=False):
    header = "ply\nformat " + ("binary_little_endian" if binary else "ascii") + " 1.0\n"
    header += f"element vertex {len(vertices)}\nproperty double x\nproperty double y\nproperty double z\n"
    if faces:
        header += f"element face {len(faces)}\nproperty list uchar uint vertex_indices\n"
    header += "end_header\n"
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        for vertex in vertices:
            stream.write(struct.pack("<ddd", *vertex) if binary else (" ".join(map(str, vertex))+"\n").encode())
        for face in faces:
            stream.write(struct.pack("<BIII", 3, *face) if binary else ("3 "+" ".join(map(str, face))+"\n").encode())


@pytest.fixture
def native_bundle(bundle):
    np = pytest.importorskip("numpy")
    cv2 = pytest.importorskip("cv2")
    vertices = [(0, 0, 5), (1, 0, 5), (0, 1, 5)]
    write_ply(bundle / "05_fused/mesh.ply", vertices, [(0, 1, 2)], binary=True)
    write_ply(bundle / "05_fused/cloud.ply", [(1.5, 2.5, 3.5)])
    path = bundle / "05_fused/mesh_provenance.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows.append({**rows[1], "mesh_vertex": 2, "xyz": list(vertices[2])})
    write_rows(path, rows)
    report = report_at(bundle)
    report["stages"]["fusion"]["mesh_vertices"] = 3
    write_json(bundle / "report.json", report)
    manifest = json.loads((bundle / "00_input/input_manifest.json").read_text())
    for frame in manifest:
        encoded = cv2.imencode(".jpg", np.full((80, 100, 3), 90, dtype=np.uint8))[1]
        path = bundle / "00_input/images" / frame["name"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded.tobytes())
        if frame["index"] < 2:
            (bundle / "02_undistorted/images" / frame["name"]).write_bytes(encoded.tobytes())
            depth = np.ones((20, 30), dtype=np.float32)
            depth[0, 0] = 0  # masked zero is a valid depth representation
            confidence = np.full_like(depth, 1.03)  # confidence is not a probability
            np.savez(bundle / f"03_depth/{frame['index']:06d}.npz", depth=depth,
                     confidence=confidence, is_metric=np.array(0, dtype=np.uint8))
            np.savez(bundle / f"04_alignment/depth_aligned/{frame['index']:06d}.npz",
                     depth=depth * 5, confidence=confidence)
    path = bundle / "05_fused/voxel_provenance.sqlite"
    path.unlink()
    provenance = json.loads((bundle / "05_fused/provenance.jsonl").read_text())
    with sqlite3.connect(path) as connection:
        connection.execute("""CREATE TABLE voxels (
            ix INTEGER NOT NULL, iy INTEGER NOT NULL, iz INTEGER NOT NULL,
            sx REAL NOT NULL, sy REAL NOT NULL, sz REAL NOT NULL,
            sr INTEGER NOT NULL, sg INTEGER NOT NULL, sb INTEGER NOT NULL,
            observations INTEGER NOT NULL, sources TEXT NOT NULL, PRIMARY KEY(ix,iy,iz))""")
        connection.execute("INSERT INTO voxels VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                           (1, 2, 3, 1.5, 2.5, 3.5, 100, 100, 100, 1, json.dumps(provenance["sources"])))
    return bundle


def test_explicit_deep_check_decodes_native_files(native_bundle):
    result = Reconstruction.load(native_bundle).validate_artifacts()
    assert result["mesh_triangles"] == 1 and result["mesh_vertices"] == 3
    assert result["decoded_images"] == 5 and result["sqlite_voxels"] == 1
    assert result["depth_frames"]["000000.jpg"]["raw"]["positive_pixels"] == 599


@pytest.mark.parametrize("kind", ["truncated", "nan", "bad_index", "empty", "vertex_mismatch", "count_mismatch", "degenerate"])
def test_deep_check_rejects_corrupt_mesh(native_bundle, kind):
    path = native_bundle / "05_fused/mesh.ply"
    if kind == "truncated":
        path.write_bytes(path.read_bytes()[:-2])
    else:
        vertices = [(0, 0, 5), (1, 0, 5), (0, 1, 5)]
        faces = [(0, 1, 2)]
        if kind == "nan":
            vertices[0] = (float("nan"), 0, 5)
        elif kind == "bad_index":
            faces = [(0, 1, 8)]
        elif kind == "empty":
            vertices, faces = [], []
        elif kind == "vertex_mismatch":
            vertices[0] = (4, 0, 5)
        elif kind == "count_mismatch":
            faces.append((0, 2, 1))
        elif kind == "degenerate":
            vertices[2] = (2, 0, 5)
        write_ply(path, vertices, faces)
    with pytest.raises(ValueError):
        Reconstruction.load(native_bundle).validate_artifacts()


@pytest.mark.parametrize("kind", ["nonfinite", "coordinate_mismatch"])
def test_deep_check_rejects_corrupt_cloud(native_bundle, kind):
    write_ply(native_bundle / "05_fused/cloud.ply",
              [(float("inf"), 2.5, 3.5) if kind == "nonfinite" else (8, 2.5, 3.5)])
    with pytest.raises(ValueError):
        Reconstruction.load(native_bundle).validate_artifacts()


@pytest.mark.parametrize("kind", ["bytes", "missing_key", "shape", "nan", "negative", "empty_depth", "bad_metric", "bad_pose"])
def test_deep_check_rejects_corrupt_depth(native_bundle, kind):
    import numpy as np

    path = native_bundle / "03_depth/000000.npz"
    if kind == "bytes":
        path.write_bytes(b"not an archive")
    else:
        with np.load(path) as data:
            arrays = {key: data[key] for key in data.files}
        if kind == "missing_key":
            del arrays["confidence"]
        elif kind == "shape":
            arrays["confidence"] = np.ones((1, 2))
        elif kind == "nan":
            arrays["confidence"][0, 0] = np.nan
        elif kind == "negative":
            arrays["depth"][0, 0] = -1
        elif kind == "empty_depth":
            arrays["depth"][:] = 0
        elif kind == "bad_metric":
            arrays["is_metric"] = np.array(2)
        else:
            arrays["intrinsics"] = np.ones((2, 3))
        np.savez(path, **arrays)
    with pytest.raises(ValueError, match="invalid NPZ"):
        Reconstruction.load(native_bundle).validate_artifacts()


@pytest.mark.parametrize("kind", ["bytes", "schema", "nonfinite", "source_mismatch", "out_of_bounds"])
def test_deep_check_rejects_corrupt_sqlite(native_bundle, kind):
    path = native_bundle / "05_fused/voxel_provenance.sqlite"
    if kind == "bytes":
        path.write_bytes(b"not SQLite")
    else:
        with sqlite3.connect(path) as connection:
            if kind == "schema":
                connection.execute("ALTER TABLE voxels ADD COLUMN unsupported TEXT")
            elif kind == "nonfinite":
                connection.execute("UPDATE voxels SET sx=?", (float("inf"),))
            elif kind == "source_mismatch":
                connection.execute("UPDATE voxels SET observations=2")
            else:
                source = json.loads(connection.execute("SELECT sources FROM voxels").fetchone()[0])
                source[0]["pixel_xy"] = [1000, 10]
                connection.execute("UPDATE voxels SET sources=?", (json.dumps(source),))
                jsonl = native_bundle / "05_fused/provenance.jsonl"
                row = json.loads(jsonl.read_text())
                row["sources"] = source
                write_rows(jsonl, [row])
    with pytest.raises(ValueError):
        Reconstruction.load(native_bundle).validate_artifacts()


@pytest.mark.parametrize("kind", ["bytes", "dimensions"])
def test_deep_check_rejects_corrupt_images(native_bundle, kind):
    import cv2
    import numpy as np

    path = native_bundle / "02_undistorted/images/000000.jpg"
    path.write_bytes(b"invalid jpg" if kind == "bytes" else
                     cv2.imencode(".jpg", np.ones((10, 10, 3), dtype=np.uint8))[1].tobytes())
    with pytest.raises(ValueError, match="invalid image"):
        Reconstruction.load(native_bundle).validate_artifacts()


def test_ascii_and_binary_ply_decode_identically(tmp_path):
    vertices = [(0, 0, 2), (1, 0, 2), (0, 1, 3)]
    results = []
    for binary in (False, True):
        path = tmp_path / f"mesh-{binary}.ply"
        write_ply(path, vertices, [(0, 1, 2)], binary=binary)
        results.append(read_ply(path, mesh=True))
    assert results[0] == results[1]


@pytest.mark.parametrize("dtype", ["bool", "uint8"])
def test_optional_input_valid_mask_is_accepted_without_discarding_raw_prediction(native_bundle, dtype):
    import numpy as np

    path = native_bundle / "03_depth/000000.npz"
    with np.load(path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    mask = np.ones(arrays["depth"].shape, dtype=dtype)
    mask[1, 1] = 0  # raw network predictions outside input support may be retained
    arrays["input_valid_mask"] = mask
    np.savez(path, **arrays)
    result = Reconstruction.load(native_bundle).validate_artifacts()
    assert result["depth_frames"]["000000.jpg"]["raw"]["input_valid_pixels"] == 599


@pytest.mark.parametrize("kind", ["shape", "nonbinary", "empty_support", "float_dtype"])
def test_optional_input_mask_content_is_validated(native_bundle, kind):
    import numpy as np

    path = native_bundle / "03_depth/000000.npz"
    with np.load(path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    mask = np.ones(arrays["depth"].shape, dtype=np.uint8)
    if kind == "shape":
        mask = mask[:1]
    elif kind == "nonbinary":
        mask[1, 1] = 2
    elif kind == "empty_support":
        mask[:] = 0
        mask[0, 0] = 1  # the only unmasked pixel has zero depth
    else:
        mask = mask.astype(np.float32)
    arrays["input_valid_mask"] = mask
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="input_valid_mask"):
        Reconstruction.load(native_bundle).validate_artifacts()
