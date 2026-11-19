"""An inferred depth frame must either integrate or fail with its identity."""

from unittest.mock import patch

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
o3d = pytest.importorskip("open3d")

from lingshu.world.reconstruct.fusion import fuse_scene


def _views(root):
    poses = {}
    for index in range(2):
        name = f"view-{index}.png"
        assert cv2.imwrite(str(root / name), np.full((48, 64, 3), 127, dtype=np.uint8))
        np.savez(root / f"view-{index}.npz", depth=np.full((48, 64), 2, np.float32),
                 confidence=np.ones((48, 64), np.float32))
        poses[name] = {"name": name, "source_index": index, "source": name,
                       "image_id": index + 1, "width": 64, "height": 48,
                       "intrinsics": [[48, 0, 31.5], [0, 48, 23.5], [0, 0, 1]],
                       "cam_from_world": [[1, 0, 0, -.1 * index], [0, 1, 0, 0], [0, 0, 1, 0]]}
    return poses


@pytest.mark.parametrize("failure", ["missing_image", "unreadable_image", "no_points"])
def test_inferred_view_cannot_be_silently_dropped(tmp_path, failure):
    poses = _views(tmp_path)
    if failure == "missing_image":
        (tmp_path / "view-1.png").unlink()
    elif failure == "unreadable_image":
        (tmp_path / "view-1.png").write_bytes(b"not an image")
    else:
        np.savez(tmp_path / "view-1.npz", depth=np.full((48, 64), 20, np.float32),
                 confidence=np.ones((48, 64), np.float32))
    with pytest.raises(RuntimeError, match="view-1.png"):
        fuse_scene(poses, tmp_path, tmp_path, tmp_path / "fused", 1, .1, 25, 5, "relative")


def test_inferred_view_without_blocks_names_failed_frame(tmp_path):
    poses = _views(tmp_path)
    with patch("lingshu.world.reconstruct.fusion.o3d.t.geometry.VoxelBlockGrid") as factory:
        factory.return_value.compute_unique_block_coordinates.return_value = np.empty((0, 3))
        with pytest.raises(RuntimeError, match="view-0.png"):
            fuse_scene(poses, tmp_path, tmp_path, tmp_path / "fused", 1, .1, 25, 5, "relative")


def test_registered_view_without_inferred_depth_is_deferred(tmp_path):
    poses = _views(tmp_path)
    (tmp_path / "view-1.npz").unlink()
    result = fuse_scene(poses, tmp_path, tmp_path, tmp_path / "fused", 1, .1, 25, 5, "relative")
    assert result["integrated_frames"] == 1


def test_partly_overlapping_views_keep_observed_boundary_surfaces(tmp_path):
    """Overlap must not suppress valid surface seen by only one input view."""
    width, height, focal = 80, 60, 70
    poses = {}
    for index, (left, right) in enumerate(((8, 50), (30, 72))):
        name = f"view-{index}.png"
        assert cv2.imwrite(str(tmp_path / name), np.full((height, width, 3), 127, np.uint8))
        valid = np.zeros((height, width), dtype=bool)
        valid[8:52, left:right] = True
        np.savez(tmp_path / f"view-{index}.npz", depth=np.full((height, width), 2, np.float32),
                 confidence=np.ones((height, width), np.float32), input_valid_mask=valid)
        poses[name] = dict(name=name, source_index=index, source=name, image_id=index+1,
                           width=width, height=height,
                           intrinsics=[[focal, 0, width/2], [0, focal, height/2], [0, 0, 1]],
                           cam_from_world=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

    result = fuse_scene(poses, tmp_path, tmp_path, tmp_path / "fused", 1, .025, 0, 5, "relative")
    mesh = o3d.io.read_triangle_mesh(result["files"]["mesh"])
    scene = o3d.t.geometry.RaycastingScene(nthreads=1)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    # Left-only, overlap, right-only, then a never-observed pixel. Rays use
    # exported continuous image coordinates, independent of the TSDF adapter.
    pixels = np.array([[15.5, 30.5], [40.5, 30.5], [65.5, 30.5], [2.5, 30.5]])
    directions = np.column_stack(((pixels[:, 0]-width/2)/focal,
                                  (pixels[:, 1]-height/2)/focal, np.ones(len(pixels))))
    rays = np.column_stack((np.zeros((len(pixels), 3)), directions)).astype(np.float32)
    depths = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    np.testing.assert_allclose(depths[:3], 2, atol=.03)
    assert np.isinf(depths[3]), "unobserved TSDF weight-zero space must not become a surface"
    assert result["mesh_provenance"]["unmapped_vertices"] == 0
    assert result["mesh_support"]["minimum_voxel_observations"] == 1
    assert result["mesh_support"]["unknown_voxels_excluded"] is True
