"""GLOMAP component selection must retain evidence for every native model."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("numpy")
pytest.importorskip("pycolmap")

from lingshu.world.reconstruct import colmap


def _models(monkeypatch, output, specifications):
    records = {}
    for name, images, points in specifications:
        path = output / name
        path.mkdir(parents=True, exist_ok=True)
        (path / "cameras.bin").write_bytes(b"native-model-preserved")
        records[path] = SimpleNamespace(
            num_reg_images=lambda images=images: len(images),
            num_points3D=lambda points=points: points,
            images={index: SimpleNamespace(name=image, has_pose=True)
                    for index, image in enumerate(images)},
        )
    monkeypatch.setattr(colmap.pycolmap, "Reconstruction", lambda path: records[Path(path)])


def test_mapper_reports_all_independent_components_without_merging(tmp_path, monkeypatch):
    output = tmp_path / "sparse"
    _models(monkeypatch, output, [("0", ["c.jpg", "d.jpg"], 200),
                                  ("1", ["a.jpg", "b.jpg", "e.jpg"], 100)])
    calls = []
    monkeypatch.setattr(colmap, "run_colmap", lambda *args: calls.append(args))
    selected = colmap.map_global_scene(tmp_path / "colmap", tmp_path / "database.db",
                                      tmp_path / "images", output, tmp_path / "mapper.log", 2)
    manifest = json.loads((output / "components.json").read_text(encoding="utf-8"))
    assert selected == output / "1"
    assert manifest["coordinate_relationship"] == "independent_per_component"
    assert manifest["component_count"] == 2
    assert manifest["all_component_registered_images"] == 5
    assert manifest["selected_model"] == "1"
    assert [item["selected"] for item in manifest["components"]] == [False, True]
    assert [item["image_names"] for item in manifest["components"]] == [
        ["c.jpg", "d.jpg"], ["a.jpg", "b.jpg", "e.jpg"]]
    assert calls[0][1] == "global_mapper"
    assert "--GlobalMapper.gp_use_gpu" in calls[0][2]
    for component in manifest["components"]:
        assert (output / component["model_path"] / "cameras.bin").read_bytes() == b"native-model-preserved"


@pytest.mark.parametrize("second_points,expected", [(101, "1"), (100, "0")])
def test_component_selection_uses_points_then_stable_path_order(tmp_path, monkeypatch, second_points, expected):
    _models(monkeypatch, tmp_path, [("1", ["c.jpg", "d.jpg"], second_points),
                                   ("0", ["a.jpg", "b.jpg"], 100)])
    assert colmap.choose_model(tmp_path) == tmp_path / expected


def test_component_inventory_supports_native_model_at_output_root(tmp_path, monkeypatch):
    _models(monkeypatch, tmp_path, [(".", ["a.jpg", "b.jpg", "c.jpg"], 12)])
    manifest = colmap.describe_model_components(tmp_path)
    assert manifest["selected_model"] == "."
    assert manifest["component_count"] == 1
    assert manifest["components"][0]["sparse_points"] == 12
    assert colmap.choose_model(tmp_path) == tmp_path


def test_missing_native_models_fail_without_claiming_component_coverage(tmp_path):
    with pytest.raises(RuntimeError, match="no registered model"):
        colmap.describe_model_components(tmp_path)


def _prepare_matching(tmp_path, monkeypatch, mode):
    calls, batches = [], []
    monkeypatch.setattr(colmap, "run_colmap", lambda *args: calls.append(args))
    monkeypatch.setattr(colmap, "run_batched_matching", lambda *args: batches.append(args))
    monkeypatch.setattr(colmap, "_recover_weak_sift", lambda *args, **kwargs: {"status": "not_needed"})
    result = colmap.prepare_matches(tmp_path / "colmap", tmp_path / "database.db",
                                    tmp_path / "images", tmp_path / "logs", 2, 1600,
                                    "SIMPLE_RADIAL", matching_mode=mode)
    assert set(result) == {"feature_seconds", "matching_seconds", "feature_support"}
    assert calls[0][1] == "feature_extractor"
    assert calls[0][2][calls[0][2].index("--FeatureExtraction.use_gpu") + 1] == "0"
    return calls, batches


def test_exhaustive_matching_explicitly_calls_cpu_exhaustive_matcher(tmp_path, monkeypatch):
    calls, batches = _prepare_matching(tmp_path, monkeypatch, "exhaustive")
    assert batches == [] and len(calls) == 2
    assert calls[1] == (tmp_path / "colmap", "exhaustive_matcher", [
        "--database_path", str(tmp_path / "database.db"),
        "--FeatureMatching.use_gpu", "0", "--FeatureMatching.num_threads", "2",
    ], tmp_path / "logs/exhaustive_matcher.log", tmp_path)


def test_sequential_matching_retains_cpu_overlap_ten_options(tmp_path, monkeypatch):
    calls, batches = _prepare_matching(tmp_path, monkeypatch, "sequential")
    assert batches == [] and len(calls) == 2
    assert calls[1] == (tmp_path / "colmap", "sequential_matcher", [
        "--database_path", str(tmp_path / "database.db"),
        "--FeatureMatching.use_gpu", "0", "--FeatureMatching.num_threads", "2",
        "--SequentialMatching.overlap", "10",
    ], tmp_path / "logs/sequential_matcher.log", tmp_path)


def test_batched_matching_retains_bounded_worker_dispatch(tmp_path, monkeypatch):
    calls, batches = _prepare_matching(tmp_path, monkeypatch, "batched")
    assert len(calls) == 1
    assert batches == [(tmp_path / "database.db", tmp_path / "logs", 2)]
