"""Batch depth samples: camera delegation, strict preflight and JSON output."""

import builtins
from copy import deepcopy
from fractions import Fraction
import importlib
import json
import math
from unittest.mock import Mock, call, patch

import pytest

from lingshu.world import depth_samples
from lingshu.world.depth_samples import unproject_depth_samples
from lingshu.world.world3d import Camera3D


@pytest.fixture
def camera():
    return Camera3D(cx=0.0, cy=0.0, cz=0.0)


def test_public_world_export_is_lazy_and_does_not_load_optional_images():
    import subprocess
    import sys
    from pathlib import Path

    repo = str(Path(__file__).resolve().parents[1])
    code = """
import sys
sys.path.insert(0, sys.argv[1])
import lingshu.world
assert 'lingshu.world.depth_samples' not in sys.modules
from lingshu.world import unproject_depth_samples
from lingshu.world.world3d import Camera3D
assert unproject_depth_samples([(0, 0, 1)], Camera3D(), (1, 1))['sample_count'] == 1
assert not any(name.split('.')[0] in ('PIL', 'numpy') for name in sys.modules)
"""
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code, repo],
                            capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_center_corners_order_and_exact_delegation(camera):
    samples = [(400, 300, 4), [0, 0, 2], (799.75, 599.5, 8),
               [0, 599.75, 3], (799.5, 0, 6)]
    expected = [camera.unproject(*sample, 800, 600) for sample in samples]
    with patch.object(camera, "unproject", wraps=camera.unproject) as inverse:
        result = unproject_depth_samples(samples, camera, (800, 600))
    assert set(result) == {"width", "height", "sample_count", "points"}
    assert (result["width"], result["height"], result["sample_count"]) == (800, 600, 5)
    assert inverse.call_args_list == [call(*sample, 800, 600) for sample in samples]
    for point, sample, world in zip(result["points"], samples, expected):
        assert set(point) == {"pixel", "depth", "world"}
        assert point["pixel"] == list(sample[:2])
        assert point["depth"] == sample[2]
        assert point["world"] == pytest.approx(world)
        assert all(type(value) is float for value in
                   point["pixel"] + [point["depth"]] + point["world"])
    assert result["points"][0]["world"] == [0.0, 0.0, 4.0]
    assert result["points"][1]["world"][0] < 0
    assert result["points"][1]["world"][1] > 0
    assert result["points"][2]["world"][0] > 0
    assert result["points"][2]["world"][1] < 0
    assert json.loads(json.dumps(result, allow_nan=False)) == result


@pytest.mark.parametrize("orientation", [
    {"yaw": 0.43}, {"pitch": -0.27}, {"roll": 0.61},
    {"yaw": -0.36, "pitch": 0.21, "roll": -0.48},
    {"basis": ((0., 0., 1.), (0., 1., 0.), (-1., 0., 0.))},
    {"basis": ((0., 0., 1.), (0., 1., 0.), (-1., 0., 0.)),
     "yaw": 1.2, "pitch": -0.7, "roll": 0.4},
])
def test_orientation_project_unproject_roundtrip(orientation):
    cam = Camera3D(cx=1.25, cy=-2.5, cz=3.75, **orientation)
    samples = [(400, 300, 5), (0, 0, 2), (799.75, 599.5, 8),
               (125.25, 472.125, 0.1001)]
    result = unproject_depth_samples(samples, cam, (800, 600))
    for point, sample in zip(result["points"], samples):
        assert point["world"] == pytest.approx(cam.unproject(*sample, 800, 600))
        assert cam.project(point["world"], 800, 600) == pytest.approx(sample[:2], abs=1e-9)
        assert cam.depth_of(point["world"]) == pytest.approx(sample[2], abs=1e-12)


def test_explicit_basis_overrides_yaw_pitch_and_roll():
    basis = ((0., 0., 1.), (0., 1., 0.), (-1., 0., 0.))
    plain = Camera3D(basis=basis)
    angled = Camera3D(basis=basis, yaw=1.2, pitch=-0.7, roll=0.4)
    samples = [(100, 200, 7), (400, 300, 3)]
    assert unproject_depth_samples(samples, plain, (800, 600)) == (
        unproject_depth_samples(samples, angled, (800, 600)))


def test_camera_translation_adds_world_offset():
    pose = {"yaw": 0.2, "pitch": -0.4, "roll": 0.3}
    origin = Camera3D(cx=0, cy=0, cz=0, **pose)
    moved = Camera3D(cx=7, cy=-3, cz=2, **pose)
    samples = [(100, 200, 5), (400, 300, 3)]
    before = unproject_depth_samples(samples, origin, (800, 600))
    after = unproject_depth_samples(samples, moved, (800, 600))
    for a, b in zip(before["points"], after["points"]):
        assert b["world"] == pytest.approx([v + d for v, d in zip(a["world"], (7, -3, 2))])


@pytest.mark.parametrize("samples", [[], ()])
def test_empty_samples(camera, samples):
    with patch.object(camera, "unproject") as inverse:
        assert unproject_depth_samples(samples, camera, (800, 600)) == {
            "width": 800, "height": 600, "sample_count": 0, "points": []}
    inverse.assert_not_called()


@pytest.mark.parametrize("limit", [1, 4096])
def test_max_samples_inclusive_limits(camera, limit):
    samples = [(0, 0, 1)] * limit
    assert unproject_depth_samples(samples, camera, (1, 1), max_samples=limit)["sample_count"] == limit
    with patch.object(camera, "unproject") as inverse:
        with pytest.raises(ValueError, match="exceeds"):
            unproject_depth_samples(samples + [(0, 0, 1)], camera, (1, 1), max_samples=limit)
    inverse.assert_not_called()


@pytest.mark.parametrize("limit", [0, -1, 4097, True, False, 1.0, "1", None, math.inf, math.nan])
def test_invalid_max_samples_never_calls_camera(camera, limit):
    with patch.object(camera, "unproject") as inverse:
        with pytest.raises((TypeError, ValueError)):
            unproject_depth_samples([], camera, (800, 600), max_samples=limit)
    inverse.assert_not_called()


@pytest.mark.parametrize("screen", [
    None, "800,600", 800, {}, (), (800,), (800, 600, 1),
    (0, 600), (800, -1), (-800, 600), (800, 0),
    (True, 600), (800, False), (800.0, 600), (800, 600.0),
    ("800", 600), (800, None), (math.inf, 600), (800, math.nan),
])
def test_invalid_screen_never_calls_camera(camera, screen):
    with patch.object(camera, "unproject") as inverse:
        with pytest.raises((TypeError, ValueError)):
            unproject_depth_samples([], camera, screen)
    inverse.assert_not_called()


@pytest.mark.parametrize("bad_camera", [None, {}, object(), Camera3D, Mock()])
def test_invalid_camera_never_calls_unproject(bad_camera):
    with patch.object(Camera3D, "unproject") as inverse:
        with pytest.raises(TypeError, match="Camera3D"):
            unproject_depth_samples([(0, 0, 1)], bad_camera, (800, 600))
    inverse.assert_not_called()
    if isinstance(bad_camera, Mock):
        bad_camera.unproject.assert_not_called()


@pytest.mark.parametrize("samples", [None, "0,0,1", {}, {1, 2, 3}, iter([(0, 0, 1)])])
def test_samples_must_be_list_or_tuple(camera, samples):
    with patch.object(camera, "unproject") as inverse:
        with pytest.raises(TypeError):
            unproject_depth_samples(samples, camera, (800, 600))
    inverse.assert_not_called()


@pytest.mark.parametrize("bad", [None, 1, "123", {}, (), (1,), (1, 2), (1, 2, 3, 4), {1, 2, 3}])
def test_complete_preflight_rejects_late_bad_structure(camera, bad):
    with patch.object(camera, "unproject") as inverse:
        with pytest.raises(TypeError):
            unproject_depth_samples([(0, 0, 1), bad], camera, (800, 600))
    inverse.assert_not_called()


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, True, False, "1", None, 1j, 10 ** 400])
def test_complete_preflight_rejects_invalid_scalar(camera, axis, bad):
    sample = [100, 200, 3]
    sample[axis] = bad
    with patch.object(camera, "unproject") as inverse:
        with pytest.raises((TypeError, ValueError)):
            unproject_depth_samples([(0, 0, 1), sample], camera, (800, 600))
    inverse.assert_not_called()


@pytest.mark.parametrize("sample", [
    (-0.01, 0, 1), (0, -0.01, 1), (800, 0, 1), (0, 600, 1),
    (800.1, 200, 1), (100, 600.1, 1), (100, 200, -1),
    (100, 200, 0), (100, 200, 0.1), (100, 200, 0.099999),
])
def test_complete_preflight_rejects_range_errors(camera, sample):
    with patch.object(camera, "unproject") as inverse:
        with pytest.raises(ValueError):
            unproject_depth_samples([(0, 0, 1), sample], camera, (800, 600))
    inverse.assert_not_called()


def test_real_fraction_samples_and_list_screen_are_supported(camera):
    sample = (Fraction(1, 2), Fraction(3, 4), Fraction(1, 4))
    result = unproject_depth_samples((sample,), camera, [1, 1])
    assert result["points"][0]["pixel"] == [0.5, 0.75]
    assert result["points"][0]["depth"] == 0.25
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("axis", range(3))
def test_nonfinite_world_output_is_value_error(camera, bad, axis):
    world = [1., 2., 3.]
    world[axis] = bad
    with patch.object(camera, "unproject", return_value=world) as inverse:
        with pytest.raises(ValueError, match="finite"):
            unproject_depth_samples([(0, 0, 1)], camera, (800, 600))
    inverse.assert_called_once_with(0., 0., 1., 800, 600)


def test_nonfinite_actual_camera_is_rejected_before_unproject():
    cam = Camera3D(cx=math.inf)
    with patch.object(cam, "unproject") as inverse:
        with pytest.raises(ValueError, match="finite"):
            unproject_depth_samples([(400, 300, 2)], cam, (800, 600))
    inverse.assert_not_called()


def test_inputs_and_camera_are_unchanged_and_result_is_independent(camera):
    samples = [[400, 300, 2], (0.25, 0.5, 4)]
    screen = [800, 600]
    before = deepcopy((samples, screen, camera.to_meta()))
    row = samples[0]
    result = unproject_depth_samples(samples, camera, screen)
    assert (samples, screen, camera.to_meta()) == before
    assert samples[0] is row
    result["points"][0]["pixel"][0] = 99
    result["points"][0]["world"][0] = 99
    assert (samples, screen, camera.to_meta()) == before


@pytest.mark.parametrize("field", ["cx", "cy", "cz", "yaw", "pitch", "roll", "fov_deg"])
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, True, False, "1", None, 1j, 10 ** 400])
def test_invalid_camera_scalar_preflight_and_unchanged(field, bad):
    cam = Camera3D(**{field: bad})
    before = repr(vars(cam))
    with patch.object(cam, "unproject") as inverse:
        with pytest.raises((TypeError, ValueError), match=field):
            unproject_depth_samples([(0, 0, 1)], cam, (800, 600))
    inverse.assert_not_called()
    assert repr(vars(cam)) == before


@pytest.mark.parametrize("fov", [-1, 0, 180, 181])
def test_invalid_camera_fov_range_preflight(fov):
    cam = Camera3D(fov_deg=fov)
    with patch.object(cam, "unproject") as inverse:
        with pytest.raises(ValueError, match="fov_deg"):
            unproject_depth_samples([], cam, (800, 600))
    inverse.assert_not_called()
    assert cam.fov_deg == fov


@pytest.mark.parametrize("basis", [
    (), [], "123", {}, ((1, 0, 0),),
    ((1, 0, 0), (0, 1, 0)),
    ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 1)),
    ((1, 0), (0, 1, 0), (0, 0, 1)),
    ((1, 0, 0), None, (0, 0, 1)),
    ((2, 0, 0), (0, 1, 0), (0, 0, 1)),
    ((1, 0, 0), (0.1, 1, 0), (0, 0, 1)),
    ((1, 0, 0), (1, 0, 0), (0, 0, 1)),
    ((1, 0, 0), (0.6, 0.8, 0), (0, 0, 1)),
    ((-1, 0, 0), (0, 1, 0), (0, 0, 1)),
    ((0, 1, 0), (1, 0, 0), (0, 0, 1)),
    ((1e308, 0, 0), (0, 1, 0), (0, 0, 1)),
])
def test_invalid_basis_preflight_and_unchanged(basis):
    cam = Camera3D(basis=basis)
    before = repr(vars(cam))
    with patch.object(cam, "unproject") as inverse:
        with pytest.raises((TypeError, ValueError), match="basis"):
            unproject_depth_samples([(0, 0, 1)], cam, (800, 600))
    inverse.assert_not_called()
    assert cam.basis is basis
    assert repr(vars(cam)) == before


@pytest.mark.parametrize("axis", range(9))
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, True, False, "1", None, 1j, 10 ** 400])
def test_invalid_basis_scalar_preflight(axis, bad):
    basis = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    basis[axis // 3][axis % 3] = bad
    cam = Camera3D(basis=basis)
    before = repr(vars(cam))
    with patch.object(cam, "unproject") as inverse:
        with pytest.raises((TypeError, ValueError), match="basis"):
            unproject_depth_samples([(0, 0, 1)], cam, (800, 600))
    inverse.assert_not_called()
    assert cam.basis is basis
    assert repr(vars(cam)) == before


@pytest.mark.parametrize("offset,accepted", [(2e-10, True), (2e-9, False)])
def test_basis_orthogonality_absolute_tolerance(offset, accepted):
    basis = [[1, 0, 0], [offset, 1, 0], [0, 0, 1]]
    cam = Camera3D(basis=basis)
    before = deepcopy(vars(cam))
    with patch.object(cam, "unproject", wraps=cam.unproject) as inverse:
        if accepted:
            unproject_depth_samples([(400, 300, 1)], cam, (800, 600))
            inverse.assert_called_once()
        else:
            with pytest.raises(ValueError, match="orthogonal"):
                unproject_depth_samples([(400, 300, 1)], cam, (800, 600))
            inverse.assert_not_called()
    assert vars(cam) == before
    assert cam.basis is basis


def test_basis_determinant_checked_even_when_row_norms_are_within_tolerance():
    scale = 1 + 4e-10
    cam = Camera3D(basis=((scale, 0, 0), (0, scale, 0), (0, 0, scale)))
    with patch.object(cam, "unproject") as inverse:
        with pytest.raises(ValueError, match="determinant"):
            unproject_depth_samples([(0, 0, 1)], cam, (800, 600))
    inverse.assert_not_called()


@pytest.mark.parametrize("field", ["yaw", "pitch", "roll"])
def test_ignored_euler_parameters_still_require_validation_with_basis(field):
    cam = Camera3D(basis=((1, 0, 0), (0, 1, 0), (0, 0, 1)), **{field: math.inf})
    with patch.object(cam, "unproject") as inverse:
        with pytest.raises(ValueError, match=field):
            unproject_depth_samples([], cam, (800, 600))
    inverse.assert_not_called()


@pytest.mark.parametrize("screen", [(10 ** 400, 600), (800, 10 ** 400)])
def test_unrepresentable_screen_dimension_preflight(camera, screen):
    with patch.object(camera, "unproject") as inverse:
        with pytest.raises(ValueError, match="finite float"):
            unproject_depth_samples([(0, 0, 1)], camera, screen)
    inverse.assert_not_called()


def test_large_representable_screen_has_no_arbitrary_resolution_limit(camera):
    dimension = 10 ** 100
    result = unproject_depth_samples([(0, 0, 1)], camera, (dimension, dimension))
    assert result["width"] == result["height"] == dimension
    json.dumps(result, allow_nan=False)


def test_valid_list_basis_and_camera_state_are_unchanged():
    basis = [[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]]
    cam = Camera3D(cx=1, cy=2, cz=3, basis=basis, yaw=0.1, pitch=0.2, roll=0.3)
    before = deepcopy(vars(cam))
    unproject_depth_samples([(0, 0, 1), (400, 300, 2)], cam, (800, 600))
    assert vars(cam) == before
    assert cam.basis is basis


def test_import_and_call_do_not_import_numpy_or_pillow(monkeypatch, camera):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name.split(".")[0] not in {"numpy", "PIL", "Pillow"}
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    module = importlib.reload(depth_samples)
    assert module.unproject_depth_samples([(0, 0, 1)], camera, (1, 1))["sample_count"] == 1
