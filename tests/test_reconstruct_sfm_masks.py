"""Declared input validity must reach native SIFT without changing matching."""

import pytest

pytest.importorskip("numpy")
pytest.importorskip("pycolmap")

from lingshu.world.reconstruct import colmap


@pytest.mark.parametrize("matching_mode", ["sequential", "exhaustive", "batched"])
def test_feature_mask_is_passed_only_to_native_extractor(tmp_path, monkeypatch, matching_mode):
    calls, batches = [], []
    monkeypatch.setattr(colmap, "run_colmap", lambda *args: calls.append(args))
    monkeypatch.setattr(colmap, "run_batched_matching", lambda *args: batches.append(args))
    monkeypatch.setattr(colmap, "_recover_weak_sift", lambda *args, **kwargs: {"status": "not_needed"})
    mask_root = tmp_path / "feature-masks"
    result = colmap.prepare_matches(
        tmp_path / "colmap", tmp_path / "database.db", tmp_path / "images",
        tmp_path / "logs", 2, 448, "SIMPLE_RADIAL",
        matching_mode=matching_mode, mask_path=mask_root,
    )
    feature = calls[0]
    assert feature[1] == "feature_extractor"
    args = feature[2]
    assert args.count("--ImageReader.mask_path") == 1
    assert args[args.index("--ImageReader.mask_path") + 1] == str(mask_root.resolve())
    assert args[args.index("--ImageReader.single_camera") + 1] == "1"
    assert args[args.index("--ImageReader.camera_model") + 1] == "SIMPLE_RADIAL"
    assert args[args.index("--FeatureExtraction.num_threads") + 1] == "2"
    assert args[args.index("--FeatureExtraction.use_gpu") + 1] == "0"
    assert all("--ImageReader.mask_path" not in call[2] for call in calls[1:])
    assert bool(batches) == (matching_mode == "batched")
    assert set(result) == {"feature_seconds", "matching_seconds", "feature_support"}


def test_rgb_only_feature_extraction_preserves_the_unmasked_native_call(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(colmap, "run_colmap", lambda *args: calls.append(args))
    monkeypatch.setattr(colmap, "_recover_weak_sift", lambda *args, **kwargs: {"status": "not_needed"})
    colmap.prepare_matches(
        tmp_path / "colmap", tmp_path / "database.db", tmp_path / "images",
        tmp_path / "logs", 2, 448, "SIMPLE_RADIAL", mask_path=None,
    )
    assert "--ImageReader.mask_path" not in calls[0][2]
    assert [call[1] for call in calls] == ["feature_extractor", "sequential_matcher"]
