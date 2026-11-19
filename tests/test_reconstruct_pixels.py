"""Check the COLMAP half-center to Open3D integer-grid boundary independently."""
import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
o3d = pytest.importorskip("open3d")
pycolmap = pytest.importorskip("pycolmap")
pytest.importorskip("scipy")

from lingshu.world.reconstruct.alignment import estimate_alignment
from lingshu.world.reconstruct.fusion import _prepare_frame, _voxelize


def test_provenance_and_open3d_use_the_same_native_colmap_pixel_ray():
    depth = np.array([[2, 0], [0, 0]], dtype=np.float32)
    confidence = np.ones_like(depth)
    pose = dict(name='view.png', width=2, height=2,
                intrinsics=[[100, 0, 1], [0, 120, 1], [0, 0, 1]],
                cam_from_world=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    native = pycolmap.Camera(model='PINHOLE', width=2, height=2, params=[100, 120, 1, 1])
    ray = np.r_[native.cam_from_img([.5, .5]), 1.] * 2
    prepared = _prepare_frame(depth, confidence, np.zeros((2, 2, 3), dtype=np.uint8), pose, 1, 0, 10)
    points, _, _, _ = _voxelize(*prepared, voxel_size=.001)
    np.testing.assert_allclose(points[0], ray, atol=1e-12)
    # Open3D's depth raster uses integer array indices, independently checked here.
    image = o3d.geometry.Image(depth)
    k = prepared[2]
    intrinsics = o3d.camera.PinholeCameraIntrinsic(2, 2, k[0, 0], k[1, 1], k[0, 2], k[1, 2])
    cloud = o3d.geometry.PointCloud.create_from_depth_image(image, intrinsics, depth_scale=1)
    np.testing.assert_allclose(np.asarray(cloud.points)[0], ray, atol=1e-8)


def test_sparse_anchor_sampling_selects_the_pixel_containing_its_half_center(tmp_path):
    depth = np.random.default_rng(45).uniform(1, 10, (8, 8)).astype(np.float32)
    np.savez(tmp_path / 'view.npz', depth=depth, confidence=np.ones_like(depth))
    observations = [dict(image='view.png', pixel_xy=[x+.5, y+.5], camera_z=2*float(depth[y, x])+3)
                    for y in range(8) for x in range(8)]
    fit = estimate_alignment([dict(observations=observations)], tmp_path,
                             {'view.png': dict(width=8, height=8)}, min_samples=30)
    assert fit['inlier_observations'] == 64
    assert fit['scale_sfm_units_per_da3_unit'] == pytest.approx(2)
    assert fit['shift_sfm_units'] == pytest.approx(3)
