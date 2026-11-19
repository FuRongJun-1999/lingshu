"""Static geometry import contracts; synthetic metadata is E1 evidence only."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from lingshu.world.reconstruction import Reconstruction
from lingshu.world.scene_model import WorldModel


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture
def bundle(tmp_path):
    """Opaque artifacts are placeholders; tests exercise metadata, not decoders."""
    root = tmp_path / "run"
    units = "COLMAP relative units"
    manifest = [{"name": f"{i:06d}.jpg", "index": i,
                 "source": f"missing-original/frame-{i}.png", "width": 100,
                 "height": 80, "source_width": 100, "source_height": 80} for i in range(3)]
    poses = {}
    for i in range(2):
        frame = manifest[i]
        poses[frame["name"]] = {
            "image_id": i + 1, "camera_id": 1, "source": frame["source"],
            "source_index": i, "width": 100, "height": 80, "model": "PINHOLE",
            "params": [100, 120, 45, 35],
            "intrinsics": [[100, 0, 45], [0, 120, 35], [0, 0, 1]],
            "cam_from_world": [[1, 0, 0, -i], [0, 1, 0, 0], [0, 0, 1, 0]],
        }
        for rel in (f"02_undistorted/images/{frame['name']}",
                    f"03_depth/{i:06d}.npz", f"04_alignment/depth_aligned/{i:06d}.npz"):
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"opaque native artifact")
    source = {"frame": manifest[0]["source"], "image": manifest[0]["name"],
              "image_id": 1, "pixel_xy": [20, 10]}
    voxel = {"voxel": [1, 2, 3], "center": [1.5, 2.5, 3.5], "units": units,
             "observations": 1, "sources": [source]}
    mesh = [{"mesh_vertex": i, "xyz": [i, 0, 5], "units": units,
             "nearest_distance": 0.1 if i == 0 else None,
             "source": {"voxel": [1, 2, 3], "observations": 1, "sources": [source]}
             if i == 0 else None} for i in range(2)]
    files = {"mesh": "05_fused/mesh.ply", "cloud": "05_fused/cloud.ply",
             "provenance_jsonl": "05_fused/provenance.jsonl",
             "provenance_sqlite": "05_fused/voxel_provenance.sqlite",
             "mesh_provenance_jsonl": "05_fused/mesh_provenance.jsonl"}
    for rel in files.values():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"opaque native artifact")
    write_rows(root / files["provenance_jsonl"], [voxel])
    write_rows(root / files["mesh_provenance_jsonl"], mesh)
    track = {"point3D_id": 7, "xyz": [1, 0.5, 5], "rgb": [100, 110, 120],
             "reprojection_error_px": 0,
             "observations": [{"image": f"{i:06d}.jpg", "image_id": i + 1,
                               "pixel_xy": [65 - 20*i, 47], "camera_z": 5}
                              for i in range(2)]}
    alignment = {"external_metric_conversion": {"input_units": units, "output_units": units,
                 "meters_per_sfm_unit": None, "aligned_frames": 2}}
    report = {"schema_version": 1, "output": str(root), "units": units,
              "absolute_metric_scale": False,
              "stages": {"input": {"images": 3},
                         "sparse": {"registered_images": 2, "sparse_points": 1,
                                    "observations": 2, "unregistered_images": [manifest[2]["source"]]},
                         "depth": {"registered_views": 2, "inferred_views": 2,
                                   "inferred_images": list(poses), "skipped_images": []},
                         "alignment": alignment,
                         "fusion": {"integrated_frames": 2, "units": units, "valid_voxels": 1,
                                    "mesh_vertices": 2, "mesh_triangles": 1, "files": files}},
              "verification": {"reprojection": {"rmse_px": 0}}}
    write_json(root / "00_input/input_manifest.json", manifest)
    write_json(root / "02_sparse/poses.json", poses)
    write_rows(root / "02_sparse/tracks.jsonl", [track])
    write_json(root / "04_alignment/scale_report.json", alignment)
    write_json(root / "report.json", report)
    return root


def report_at(root):
    return json.loads((root / "report.json").read_text(encoding="utf-8"))


def test_static_attachment_retains_missing_views_and_existing_entities(bundle):
    world = WorldModel().add_entity("known", "tree", (1, 0, 2))
    scene = world.scene
    entities = deepcopy(world.entities)
    assert world.load_reconstruction(bundle) is world
    recon = world.reconstruction
    assert world.scene is scene and world.entities == entities
    assert recon.unregistered_images == ["missing-original/frame-2.png"]
    assert recon.to_dict()["frames"][2] == {
        "name": "000002.jpg", "source": "missing-original/frame-2.png",
        "source_index": 2, "registered": False, "dense": False}
    assert not recon.absolute_metric_scale and recon.coordinate_scale == 1
    assert recon.triangulate_track(7) == pytest.approx((1, 0.5, 5), abs=1e-12)
    assert Reconstruction.load(bundle).run_id == recon.run_id
    world.build()
    assert world.reconstruction is recon and world.entities == entities
    other = WorldModel()
    assert other.reconstruction is None


def test_metric_conversion_scales_points_camera_depth_and_camera_translation_once(bundle):
    report = report_at(bundle)
    report["units"] = "meters"
    report["absolute_metric_scale"] = True
    conversion = report["stages"]["alignment"]["external_metric_conversion"]
    conversion.update(output_units="meters", meters_per_sfm_unit=0.25)
    report["stages"]["fusion"]["units"] = "meters"
    write_json(bundle / "report.json", report)
    write_json(bundle / "04_alignment/scale_report.json", report["stages"]["alignment"])
    for file in ("provenance.jsonl", "mesh_provenance.jsonl"):
        path = bundle / "05_fused" / file
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row["units"] = "meters"
        write_rows(path, rows)
    recon = Reconstruction.load(bundle)
    track = next(recon.iter_tracks())
    assert track["xyz"] == [0.25, 0.125, 1.25]
    assert track["observations"][0]["camera_z"] == 1.25
    assert recon.cameras["000001.jpg"].unproject(45, 47, 1.25) == pytest.approx(track["xyz"])
    assert recon.triangulate_track(7) == pytest.approx(track["xyz"], abs=1e-12)
    # Fusion files were already converted by the producer, so they are not scaled again.
    assert next(recon.iter_voxel_provenance())["center"] == [1.5, 2.5, 3.5]


@pytest.mark.parametrize("units,metric,scale", [
    ("meters", False, None), ("COLMAP relative units", True, 0.25),
    ("meters", True, None), ("meters", True, 0), ("meters", True, -1),
    ("meters", True, True), ("COLMAP relative units", False, 1),
])
def test_bad_scale_contract_rejected_before_replacing_attachment(bundle, units, metric, scale):
    world = WorldModel().load_reconstruction(bundle)
    held = world.reconstruction
    report = report_at(bundle)
    report.update(units=units, absolute_metric_scale=metric)
    report["stages"]["alignment"]["external_metric_conversion"]["meters_per_sfm_unit"] = scale
    write_json(bundle / "report.json", report)
    with pytest.raises(ValueError):
        world.load_reconstruction(bundle)
    assert world.reconstruction is held


@pytest.mark.parametrize("mutation", ["xyz", "depth", "unknown_frame", "counts", "duplicate_track", "bad_json"])
def test_malformed_sparse_geometry_is_rejected_as_a_bundle(bundle, mutation):
    path = bundle / "02_sparse/tracks.jsonl"
    track = json.loads(path.read_text())
    if mutation == "xyz":
        track["xyz"][0] = float("nan")
    elif mutation == "depth":
        track["observations"][0]["camera_z"] = 50
    elif mutation == "unknown_frame":
        track["observations"][0]["image"] = "unregistered.jpg"
    elif mutation == "counts":
        report = report_at(bundle)
        report["stages"]["sparse"]["observations"] = 99
        write_json(bundle / "report.json", report)
    write_rows(path, [track, track] if mutation == "duplicate_track" else [track])
    if mutation == "bad_json":
        path.write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError):
        Reconstruction.load(bundle)


def test_relocation_rebases_old_absolute_artifact_paths(bundle, tmp_path):
    report = report_at(bundle)
    for key, path in report["stages"]["fusion"]["files"].items():
        report["stages"]["fusion"]["files"][key] = str(bundle / path)
    write_json(bundle / "report.json", report)
    copied = tmp_path / "relocated"
    shutil.copytree(bundle, copied)
    recon = Reconstruction.load(copied)
    assert recon.artifacts["mesh"] == copied / "05_fused/mesh.ply"
    assert recon.cameras["000000.jpg"].source == "missing-original/frame-0.png"


@pytest.mark.parametrize("bad_path", ["../outside.ply", "05_fused/../../outside.ply", "C:\\elsewhere\\mesh.ply"])
def test_artifact_path_escape_rejected(bundle, bad_path):
    report = report_at(bundle)
    report["stages"]["fusion"]["files"]["mesh"] = bad_path
    write_json(bundle / "report.json", report)
    with pytest.raises(ValueError):
        Reconstruction.load(bundle)


def test_missing_artifact_is_rejected(bundle):
    (bundle / "05_fused/mesh.ply").unlink()
    with pytest.raises(ValueError, match="missing reconstruction artifact"):
        Reconstruction.load(bundle)


def test_source_provenance_is_streamed_and_unmapped_vertices_retained(bundle):
    recon = Reconstruction.load(bundle)
    assert iter(recon.iter_tracks()) is not None
    mesh = list(recon.iter_mesh_provenance())
    assert mesh[0]["source"]["sources"][0]["image"] == "000000.jpg"
    assert mesh[1]["source"] is None
    path = bundle / "05_fused/provenance.jsonl"
    row = json.loads(path.read_text())
    row["sources"][0]["image"] = "000002.jpg"
    write_rows(path, [row])
    with pytest.raises(ValueError, match="no inferred depth"):
        Reconstruction.load(bundle)


def test_multiple_feature_observations_from_one_view_are_preserved(bundle):
    path = bundle / "02_sparse/tracks.jsonl"
    track = json.loads(path.read_text())
    extra = deepcopy(track["observations"][0])
    extra["pixel_xy"][0] += 0.1
    track["observations"].append(extra)
    write_rows(path, [track])
    report = report_at(bundle)
    report["stages"]["sparse"]["observations"] = 3
    write_json(bundle / "report.json", report)
    recon = Reconstruction.load(bundle)
    assert len(next(recon.iter_tracks())["observations"]) == 3


def test_import_has_no_site_packages_or_native_dependency_requirement():
    source = (
        "import sys; from lingshu.world.reconstruction import Reconstruction; "
        "from lingshu.world.scene_model import WorldModel; "
        "assert not {'numpy', 'cv2', 'open3d', 'torch', 'pycolmap'} & set(sys.modules); "
        "assert WorldModel().reconstruction is None"
    )
    result = subprocess.run([sys.executable, "-S", "-c", source],
                            cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert result.returncode == 0, result.stderr
