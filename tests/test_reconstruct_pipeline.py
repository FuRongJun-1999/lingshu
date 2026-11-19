"""Producer sizing and preflight checks run before any native reconstruction."""

from unittest.mock import patch

import pytest

np = pytest.importorskip("numpy")
for dependency in ("cv2", "open3d", "pycolmap", "scipy", "psutil"):
    pytest.importorskip(dependency)

from lingshu.world.reconstruct.pipeline import (
    ReconstructionPipeline,
    estimate_depth_truncation,
    scene_diagonal,
)


def test_scene_extent_includes_points_camera_centers_and_output_scale():
    tracks = [{"xyz": [2, 4, 3]}, {"xyz": [0, -2, 1]}]
    poses = {"view.png": {"cam_from_world": [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3]]}}
    # Camera center (-1,-2,-3) to point (2,4,3) spans (3,6,6).
    assert scene_diagonal(tracks, poses, 2.5) == pytest.approx(22.5)


@pytest.mark.parametrize("tracks", [[], [{"xyz": [0, 0, 0]}]])
def test_empty_or_zero_scene_extent_is_rejected(tracks):
    with pytest.raises(RuntimeError, match="scene extent"):
        scene_diagonal(tracks, {}, 1)


def test_depth_truncation_filters_invalid_values_and_uses_per_view_quantiles(tmp_path):
    np.savez(tmp_path / "first.npz", depth=np.array([0, np.nan, np.inf, -2, 2, 4]))
    np.savez(tmp_path / "second.npz", depth=np.array([10, 10]))
    np.savez(tmp_path / "empty.npz", depth=np.array([0, np.nan]))
    poses = {name: {} for name in ("first.png", "second.png", "empty.png", "deferred.png")}
    expected = float(np.median([np.quantile([2, 4], 0.995), 10])) * 1.1
    assert estimate_depth_truncation(tmp_path, poses) == pytest.approx(expected)


def test_no_valid_depth_has_no_truncation_estimate(tmp_path):
    np.savez(tmp_path / "empty.npz", depth=np.array([0, -1, np.nan, np.inf]))
    with pytest.raises(RuntimeError, match="no valid aligned depth"):
        estimate_depth_truncation(tmp_path, {"empty.png": {}, "deferred.png": {}})


def _pipeline(root):
    images = root / "images"
    images.mkdir()
    for name in ("colmap", "model.gguf", "adapter.dll"):
        (root / name).write_bytes(b"test preflight only")
    return ReconstructionPipeline(
        input_path=images, output_path=root / "output", colmap=root / "colmap",
        model=root / "model.gguf", depth_library=root / "adapter.dll", runtime_bin=root,
        threads=1, depth_threads=1, depth_conv="direct",
    )


@pytest.mark.parametrize("missing", ["input_path", "colmap", "model", "depth_library"])
def test_missing_required_resources_fail_before_native_work(tmp_path, missing):
    pipeline = _pipeline(tmp_path)
    target = getattr(pipeline, missing)
    target.rmdir() if target.is_dir() else target.unlink()
    with patch("lingshu.world.reconstruct.colmap.subprocess.run") as native_process, \
         patch("lingshu.world.reconstruct.pipeline.DepthEngine") as depth_engine, \
         patch("lingshu.world.reconstruct.pipeline.normalize_images") as normalize:
        with pytest.raises(FileNotFoundError):
            pipeline.run()
        native_process.assert_not_called()
        depth_engine.assert_not_called()
        normalize.assert_not_called()
    assert not pipeline.output_path.exists()


def test_nonempty_output_is_preserved_before_native_work(tmp_path):
    pipeline = _pipeline(tmp_path)
    pipeline.output_path.mkdir()
    sentinel = pipeline.output_path / "existing.txt"
    sentinel.write_bytes(b"preserve existing run")
    with patch("lingshu.world.reconstruct.colmap.subprocess.run") as native_process, \
         patch("lingshu.world.reconstruct.pipeline.DepthEngine") as depth_engine, \
         patch("lingshu.world.reconstruct.pipeline.normalize_images") as normalize:
        with pytest.raises(FileExistsError, match="not empty"):
            pipeline.run()
        native_process.assert_not_called()
        depth_engine.assert_not_called()
        normalize.assert_not_called()
    assert sentinel.read_bytes() == b"preserve existing run"
    assert list(pipeline.output_path.iterdir()) == [sentinel]
