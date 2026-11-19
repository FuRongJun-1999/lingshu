"""Input alpha defines the observation domain, independently of model confidence."""
import pytest

np = pytest.importorskip("numpy")

from lingshu.world.reconstruct.alignment import estimate_alignment, write_aligned_depths


def test_alignment_uses_only_input_valid_observations(tmp_path):
    depth = np.arange(1, 65, dtype=np.float32).reshape(8, 8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[:, 0] = 1
    np.savez(tmp_path / 'view.npz', depth=depth, confidence=np.ones_like(depth), input_valid_mask=mask)
    observations = [dict(image='view.png', pixel_xy=[x, y],
                         camera_z=(2 * float(depth[y, x]) + 3 if x == 0 else 4 * float(depth[y, x]) + 1000))
                    for y in range(8) for x in range(8)]
    fit = estimate_alignment([dict(observations=observations)], tmp_path,
                             {'view.png': dict(width=8, height=8)}, min_samples=6)
    assert fit['input_observations'] == 8
    assert fit['scale_sfm_units_per_da3_unit'] == pytest.approx(2)
    assert fit['shift_sfm_units'] == pytest.approx(3)


def test_aligned_depth_preserves_mask_and_never_fuses_outside_it(tmp_path):
    raw, output = tmp_path / 'raw', tmp_path / 'aligned'
    raw.mkdir()
    mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    np.savez(raw / 'view.npz', depth=np.full((2, 2), 4, dtype=np.float32),
             confidence=np.ones((2, 2), dtype=np.float32), input_valid_mask=mask)
    fit = dict(scale_sfm_units_per_da3_unit=2, shift_sfm_units=1)
    write_aligned_depths(raw, output, {'view.png': {}}, fit, .1)
    with np.load(output / 'view.npz') as record:
        np.testing.assert_array_equal(record['input_valid_mask'], mask)
        np.testing.assert_allclose(record['depth'], [[.9, 0], [0, .9]])


@pytest.mark.parametrize('mask', [np.ones((1, 2), dtype=np.uint8),
                                  np.full((2, 2), 2, dtype=np.uint8),
                                  np.ones((2, 2), dtype=np.float32)])
def test_bad_input_mask_is_rejected_instead_of_becoming_full_support(tmp_path, mask):
    raw = tmp_path / 'raw'
    raw.mkdir()
    np.savez(raw / 'view.npz', depth=np.ones((2, 2)), confidence=np.ones((2, 2)), input_valid_mask=mask)
    with pytest.raises(ValueError, match='input_valid_mask'):
        write_aligned_depths(raw, tmp_path / 'aligned', {'view.png': {}},
                             dict(scale_sfm_units_per_da3_unit=1, shift_sfm_units=0))
