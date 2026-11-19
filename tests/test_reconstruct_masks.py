"""Input alpha is an explicit valid domain, never an inferred foreground mask."""

from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
pycolmap = pytest.importorskip("pycolmap")

from lingshu.world.reconstruct import images


def _write_image(path, pixels):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, pixels)
    assert ok
    encoded.tofile(path)


@pytest.mark.parametrize("dtype,maximum", [(np.uint8, 255), (np.uint16, 65535)])
def test_alpha_mask_survives_normalization_and_nearest_resize(tmp_path, dtype, maximum):
    source = tmp_path / "source"
    pixels = np.full((4, 6, 4), maximum, dtype=dtype)
    pixels[:, :, 0] = np.arange(24).reshape(4, 6) * (maximum // 24)
    pixels[0, 0, 3] = 0
    pixels[2, 2, 3] = 1  # Partial transparency stays valid under alpha > 0.
    for name in ("a.png", "b.png"):
        _write_image(source / name, pixels)
    destination = tmp_path / "00_input/images"
    manifest = images.normalize_images(source, destination, 3)
    for record in manifest:
        assert record["alpha_status"] == "transparent"
        assert record["alpha_dtype"] == np.dtype(dtype).name
        assert record["valid_mask_rule"] == "alpha>0"
        assert record["valid_mask"] == "masks/" + Path(record["name"]).stem + ".png"
        mask = cv2.imread(str(destination.parent / record["valid_mask"]), cv2.IMREAD_UNCHANGED)
        expected = cv2.resize((pixels[:, :, 3] > 0).astype(np.uint8) * 255,
                              (3, 2), interpolation=cv2.INTER_NEAREST)
        np.testing.assert_array_equal(mask, expected)
        assert mask.dtype == np.uint8 and mask.shape == (2, 3)
        assert cv2.imread(str(destination / record["name"])).shape == (2, 3, 3)
        bgr = (pixels[:, :, :3] >> 8).astype(np.uint8) if dtype == np.uint16 else pixels[:, :, :3]
        resized = cv2.resize(bgr, (3, 2), interpolation=cv2.INTER_AREA)
        ok, expected_jpeg = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
        assert ok and (destination / record["name"]).read_bytes() == expected_jpeg.tobytes()


def test_rgb_and_opaque_alpha_do_not_create_masks_or_guess_black_background(tmp_path):
    source = tmp_path / "source"
    _write_image(source / "black.jpg", np.zeros((3, 4, 3), dtype=np.uint8))
    _write_image(source / "black.png", np.zeros((3, 4, 3), dtype=np.uint8))
    opaque = np.zeros((3, 4, 4), dtype=np.uint8)
    opaque[:, :, 3] = 255
    _write_image(source / "opaque.png", opaque)
    destination = tmp_path / "00_input/images"
    manifest = images.normalize_images(source, destination, 100)
    assert [record["alpha_status"] for record in manifest] == ["absent", "absent", "opaque"]
    assert all(record["valid_mask"] is None and record["valid_mask_rule"] is None for record in manifest)
    assert not (destination.parent / "masks").exists()
    for record in manifest:
        assert not np.any(cv2.imread(str(destination / record["name"])))


def test_fully_transparent_input_fails_with_its_source_path(tmp_path):
    source = tmp_path / "source"
    for name in ("a-transparent.png", "b-transparent.png"):
        _write_image(source / name, np.zeros((3, 4, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="fully transparent.*a-transparent.png"):
        images.normalize_images(source, tmp_path / "00_input/images", 100)


def test_partial_alpha_positive_everywhere_is_explicitly_all_valid(tmp_path):
    pixels = np.full((3, 4, 4), 128, dtype=np.uint8)
    for name in ("a.png", "b.png"):
        _write_image(tmp_path / "source" / name, pixels)
    destination = tmp_path / "00_input/images"
    manifest = images.normalize_images(tmp_path / "source", destination, 100)
    assert all(record["alpha_status"] == "transparent" for record in manifest)
    for record in manifest:
        mask = cv2.imread(str(destination.parent / record["valid_mask"]), cv2.IMREAD_UNCHANGED)
        assert np.all(mask == 255)


def test_resize_cannot_silently_remove_the_entire_valid_domain(tmp_path):
    pixels = np.zeros((4, 6, 4), dtype=np.uint8)
    pixels[-1, -1, 3] = 255
    for name in ("a.png", "b.png"):
        _write_image(tmp_path / "source" / name, pixels)
    with pytest.raises(ValueError, match="no valid pixels after resizing.*a.png"):
        images.normalize_images(tmp_path / "source", tmp_path / "00_input/images", 3)


def test_alpha_dtype_outside_uint8_uint16_is_rejected(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("a.png", "b.png"):
        (source / name).write_bytes(b"mock decoded input")
    monkeypatch.setattr(images.cv2, "imdecode", lambda *args: np.ones((2, 2, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="alpha.*uint8.*uint16"):
        images.normalize_images(source, tmp_path / "00_input/images", 100)


def _camera(model="PINHOLE", width=5, height=5, params=None):
    return pycolmap.Camera(model=model, width=width, height=height,
                           params=params if params is not None else [2.0, 2.0, width / 2, height / 2])


def _install_models(monkeypatch, source_camera, target_camera, *, source_pose=None, target_pose=None,
                    source_name="000000.jpg", target_name="000000.jpg"):
    identity = np.column_stack((np.eye(3), np.zeros(3)))
    source_pose = identity if source_pose is None else source_pose
    target_pose = identity if target_pose is None else target_pose

    def reconstruction(camera, pose, name):
        image = SimpleNamespace(name=name, camera_id=1, has_pose=True,
                                cam_from_world=lambda: SimpleNamespace(matrix=lambda: pose))
        return SimpleNamespace(images={1: image}, camera=lambda camera_id: camera)

    models = {"source": reconstruction(source_camera, source_pose, source_name),
              "target": reconstruction(target_camera, target_pose, target_name)}
    monkeypatch.setattr(pycolmap, "Reconstruction", lambda path: models[Path(path).name])


def _mask_fixture(tmp_path, mask, name="000000.jpg"):
    root = tmp_path / "00_input"
    relative = "masks/" + Path(name).stem + ".png"
    _write_image(root / relative, mask)
    return root, [{"name": name, "valid_mask": relative}], tmp_path / "02_undistorted/masks"


def _undistort(manifest, root, destination):
    return images.undistort_valid_masks(manifest, root, Path("source"), Path("target"), destination)


def test_identity_mask_remap_uses_native_camera_and_preserves_size(tmp_path, monkeypatch):
    mask = np.array([[0, 255, 0, 255, 0], [255, 255, 0, 0, 0], [0, 255, 255, 255, 0]], dtype=np.uint8)
    camera = _camera(width=5, height=3)
    _install_models(monkeypatch, camera, camera)
    root, manifest, destination = _mask_fixture(tmp_path, mask)
    result = _undistort(manifest, root, destination)
    assert result == {"000000.jpg": "000000.png"}
    np.testing.assert_array_equal(cv2.imread(str(destination / result["000000.jpg"]), cv2.IMREAD_UNCHANGED), mask)


def test_camera_rotation_rotates_rays_and_mask_without_translation(tmp_path, monkeypatch):
    mask = np.array([[255, 0, 0], [255, 255, 0], [0, 0, 255]], dtype=np.uint8)
    camera = _camera(width=3, height=3, params=[1, 1, 1.5, 1.5])
    center = np.array([1., 2., 3.])
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    source_pose = np.column_stack((np.eye(3), -center))
    target_pose = np.column_stack((rotation, -rotation @ center))
    _install_models(monkeypatch, camera, camera, source_pose=source_pose, target_pose=target_pose)
    root, manifest, destination = _mask_fixture(tmp_path, mask)
    result = _undistort(manifest, root, destination)
    actual = cv2.imread(str(destination / result["000000.jpg"]), cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(actual, np.rot90(mask, -1))


def test_radial_distortion_and_changed_intrinsics_resize_mask_by_camera_rays(tmp_path, monkeypatch):
    mask = np.zeros((9, 11), dtype=np.uint8)
    mask[1:8, 2:9:2] = 255
    source_camera = _camera("SIMPLE_RADIAL", 11, 9, [4.1, 5.5, 4.5, 0.25])
    target_camera = _camera(width=7, height=5, params=[2.2, 2.4, 3.5, 2.5])
    _install_models(monkeypatch, source_camera, target_camera)
    root, manifest, destination = _mask_fixture(tmp_path, mask)
    result = _undistort(manifest, root, destination)
    actual = cv2.imread(str(destination / result["000000.jpg"]), cv2.IMREAD_UNCHANGED)
    expected = np.zeros((5, 7), dtype=np.uint8)
    for y in range(5):
        for x in range(7):
            xn, yn = (x + .5 - 3.5) / 2.2, (y + .5 - 2.5) / 2.4
            radial = 1 + .25 * (xn*xn + yn*yn)
            u, v = 4.1*xn*radial + 5.5, 4.1*yn*radial + 4.5
            if 0 <= u < 11 and 0 <= v < 9:
                expected[y, x] = mask[int(np.rint(v-.5)), int(np.rint(u-.5))]
    assert actual.shape == (5, 7)
    np.testing.assert_array_equal(actual, expected)


def test_fisheye_source_uses_the_full_native_projection_model(tmp_path, monkeypatch):
    mask = np.zeros((9, 9), dtype=np.uint8)
    mask[1::2, :] = 255
    source_camera = _camera("OPENCV_FISHEYE", 9, 9, [3.4, 3.7, 4.5, 4.5, .03, -.01, .001, 0])
    target_camera = _camera(width=5, height=5, params=[2.1, 2.4, 2.5, 2.5])
    _install_models(monkeypatch, source_camera, target_camera)
    root, manifest, destination = _mask_fixture(tmp_path, mask)
    result = _undistort(manifest, root, destination)
    actual = cv2.imread(str(destination / result["000000.jpg"]), cv2.IMREAD_UNCHANGED)
    expected = np.zeros((5, 5), dtype=np.uint8)
    for y in range(5):
        for x in range(5):
            ray = target_camera.cam_from_img(np.array([x+.5, y+.5]))
            projected = source_camera.img_from_cam(np.array([ray[0], ray[1], 1.]))
            if projected is not None and 0 <= projected[0] < 9 and 0 <= projected[1] < 9:
                expected[y, x] = mask[int(np.rint(projected[1]-.5)), int(np.rint(projected[0]-.5))]
    np.testing.assert_array_equal(actual, expected)


def test_out_of_bounds_or_behind_camera_rays_are_invalid(tmp_path, monkeypatch):
    camera = _camera(width=3, height=3, params=[1, 1, 1.5, 1.5])
    flipped = np.column_stack((np.diag([-1, 1, -1]), np.zeros(3)))
    _install_models(monkeypatch, camera, camera, target_pose=flipped)
    root, manifest, destination = _mask_fixture(tmp_path, np.full((3, 3), 255, dtype=np.uint8))
    result = _undistort(manifest, root, destination)
    assert not np.any(cv2.imread(str(destination / result["000000.jpg"]), cv2.IMREAD_UNCHANGED))


def test_different_camera_centers_are_rejected_without_depth(tmp_path, monkeypatch):
    camera = _camera()
    shifted = np.column_stack((np.eye(3), np.array([.01, 0, 0])))
    _install_models(monkeypatch, camera, camera, target_pose=shifted)
    root, manifest, destination = _mask_fixture(tmp_path, np.full((5, 5), 255, dtype=np.uint8))
    with pytest.raises(ValueError, match="camera centers"):
        _undistort(manifest, root, destination)


def test_rgb_only_manifest_needs_no_native_models_or_mask_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(pycolmap, "Reconstruction", lambda path: pytest.fail("RGB input needs no mask model read"))
    destination = tmp_path / "masks"
    assert _undistort([{"name": "000000.jpg", "valid_mask": None}], tmp_path, destination) == {}
    assert not destination.exists()


def test_unregistered_masked_inputs_are_not_given_a_pose(tmp_path, monkeypatch):
    camera = _camera()
    _install_models(monkeypatch, camera, camera, target_name="other.jpg")
    root, manifest, destination = _mask_fixture(tmp_path, np.full((5, 5), 255, dtype=np.uint8))
    assert _undistort(manifest, root, destination) == {}


@pytest.mark.parametrize("relative", ["../outside.png", "masks/../../outside.png", "/outside.png", "C:\\outside.png", "\\\\server\\share\\outside.png"])
def test_manifest_mask_paths_cannot_escape_input_root(tmp_path, relative):
    with pytest.raises(ValueError, match="relative|escape"):
        _undistort([{"name": "000000.jpg", "valid_mask": relative}], tmp_path, tmp_path / "out")


@pytest.mark.parametrize("name", ["../image.jpg", "nested/image.jpg", "C:\\image.jpg", "/image.jpg"])
def test_manifest_image_names_are_confined_filenames(tmp_path, name):
    with pytest.raises(ValueError, match="filename|relative|escape"):
        _undistort([{"name": name, "valid_mask": "masks/frame.png"}], tmp_path, tmp_path / "out")


def test_resolved_mask_path_must_stay_inside_input_root(tmp_path, monkeypatch):
    root = tmp_path / "input"
    masked = root / "masks/link.png"
    original_resolve = type(root).resolve

    def resolve(path, *args, **kwargs):
        # Simulate a resolved link crossing the root without needing Windows
        # symlink privileges; the resolver's destination is the trust boundary.
        if path == masked:
            return tmp_path / "outside.png"
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(type(root), "resolve", resolve)
    with pytest.raises(ValueError, match="outside.*escape"):
        _undistort([{"name": "000000.jpg", "valid_mask": "masks/link.png"}], root, tmp_path / "out")


def test_mask_dimensions_must_match_native_source_camera(tmp_path, monkeypatch):
    _install_models(monkeypatch, _camera(), _camera())
    root, manifest, destination = _mask_fixture(tmp_path, np.full((3, 3), 255, dtype=np.uint8))
    with pytest.raises(ValueError, match="mask dimensions"):
        _undistort(manifest, root, destination)


def test_feature_masks_cover_mixed_inputs_with_native_full_filenames(tmp_path):
    root = tmp_path / "input"
    mask = np.array([[255, 0, 255], [0, 255, 255]], dtype=np.uint8)
    _write_image(root / "masks/000000.png", mask)
    manifest = [
        dict(name="000000.jpg", width=3, height=2, valid_mask="masks/000000.png"),
        dict(name="000001.jpg", width=4, height=3, valid_mask=None),
    ]
    destination = tmp_path / "feature_masks"
    assert images.export_feature_masks(manifest, root, destination) == destination
    np.testing.assert_array_equal(cv2.imread(str(destination / "000000.jpg.png"), 0), mask)
    np.testing.assert_array_equal(cv2.imread(str(destination / "000001.jpg.png"), 0),
                                  np.full((3, 4), 255, dtype=np.uint8))
    assert not (destination / "000000.png").exists()


def test_feature_masks_are_absent_for_all_rgb_inputs(tmp_path):
    destination = tmp_path / "feature_masks"
    assert images.export_feature_masks(
        [dict(name="000000.jpg", valid_mask=None)], tmp_path, destination
    ) is None
    assert not destination.exists()


def test_feature_mask_dimension_mismatch_is_rejected(tmp_path):
    _write_image(tmp_path / "mask.png", np.full((2, 3), 255, dtype=np.uint8))
    with pytest.raises(ValueError, match="mask dimensions"):
        images.export_feature_masks(
            [dict(name="view.jpg", width=5, height=4, valid_mask="mask.png")],
            tmp_path, tmp_path / "feature_masks",
        )
