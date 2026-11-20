"""Contract tests for bounded geometry of synthetic boolean masks."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from lingshu.vision import mask_geometry
from lingshu.vision.mask_geometry import analyze_binary_components


def _mask(*rows):
    return [[cell == "#" for cell in row] for row in rows]


def _component(area, bbox, centroid, touches_boundary):
    return dict(area=area, bbox=bbox, centroid=centroid, touches_boundary=touches_boundary)


def test_two_rectangles_json_shape_and_input_unchanged():
    mask = _mask("##......", "##......", "....###.", "....###.", "........")
    before = [row[:] for row in mask]
    row_ids = list(map(id, mask))
    result = analyze_binary_components(mask)
    assert result == {
        "width": 8,
        "height": 5,
        "pixel_count": 40,
        "foreground_pixels": 10,
        "connectivity": 4,
        "min_area": 1,
        "component_count": 2,
        "filtered_component_count": 2,
        "reported_component_count": 2,
        "omitted_component_count": 0,
        "components": [
            _component(6, [4, 2, 7, 4], [5.5, 3.0], False),
            _component(4, [0, 0, 2, 2], [1.0, 1.0], True),
        ],
    }
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert mask == before
    assert list(map(id, mask)) == row_ids


def test_diagonal_connectivity():
    mask = ((True, False), (False, True))
    four = analyze_binary_components(mask, connectivity=4)
    eight = analyze_binary_components(mask, connectivity=8)
    assert four["component_count"] == 2
    assert four["components"] == [
        _component(1, [0, 0, 1, 1], [0.5, 0.5], True),
        _component(1, [1, 1, 2, 2], [1.5, 1.5], True),
    ]
    assert eight["connectivity"] == 8
    assert eight["component_count"] == 1
    assert eight["foreground_pixels"] == 2
    assert eight["components"] == [_component(2, [0, 0, 2, 2], [1.0, 1.0], True)]


@pytest.mark.parametrize("connectivity", [4, 8])
def test_ring_does_not_fill_off_center_hole(connectivity):
    mask = _mask("......", ".####.", ".#.##.", ".####.", "......")
    result = analyze_binary_components(mask, connectivity=connectivity)
    assert result["foreground_pixels"] == 11
    assert result["component_count"] == 1
    assert result["components"] == [
        _component(11, [1, 1, 5, 4], [33.5 / 11, 2.5], False),
    ]


@pytest.mark.parametrize(
    ("mask", "components"),
    [
        (_mask("...", "..."), []),
        (_mask("###", "###"), [_component(6, [0, 0, 3, 2], [1.5, 1.0], True)]),
        ([[True]], [_component(1, [0, 0, 1, 1], [0.5, 0.5], True)]),
        ([[False]], []),
    ],
)
def test_empty_foreground_full_and_single_pixel(mask, components):
    result = analyze_binary_components(mask)
    assert result["foreground_pixels"] == sum(c["area"] for c in components)
    assert result["component_count"] == len(components)
    assert result["filtered_component_count"] == len(components)
    assert result["reported_component_count"] == len(components)
    assert result["omitted_component_count"] == 0
    assert result["components"] == components


def test_min_area_filters_inclusively_and_top_k_replaces_earlier_small_region():
    mask = _mask("##.#.....", ".........", "......###")
    result = analyze_binary_components(mask, min_area=2, max_components=1)
    assert result["min_area"] == 2
    assert result["foreground_pixels"] == 6
    assert result["component_count"] == 3
    assert result["filtered_component_count"] == 2
    assert result["reported_component_count"] == 1
    assert result["omitted_component_count"] == 1
    assert result["components"] == [_component(3, [6, 2, 9, 3], [7.5, 2.5], True)]


def test_equal_area_uses_bbox_top_then_left_not_first_pixel_x():
    # The first seed is at x=2, but the later seed's region extends left to x=0.
    mask = _mask(
        "..#####.#....",
        "..#####.#....",
        "..###...#....",
        "........#....",
        "#########....",
        ".............",
        "#############",
    )
    expected = [[0, 0, 9, 5], [2, 0, 7, 3], [0, 6, 13, 7]]
    for limit in (1, 3):
        components = analyze_binary_components(mask, max_components=limit)["components"]
        assert [c["area"] for c in components] == [13] * limit
        assert [c["bbox"] for c in components] == expected[:limit]


@pytest.mark.parametrize("max_components", [3, 128])
def test_checkerboard_at_pixel_budget_truncates_but_counts_all(max_components):
    odd = tuple(x % 2 == 1 for x in range(512))
    even = odd[::-1]
    mask = (even, odd) * 256
    result = analyze_binary_components(mask, max_components=max_components)
    assert result["pixel_count"] == 262_144
    assert result["foreground_pixels"] == 131_072
    assert result["component_count"] == 131_072
    assert result["filtered_component_count"] == 131_072
    assert result["reported_component_count"] == max_components
    assert result["omitted_component_count"] == 131_072 - max_components
    assert result["components"] == [
        _component(1, [x, 0, x + 1, 1], [x + 0.5, 0.5], True)
        for x in range(0, 2 * max_components, 2)
    ]


@pytest.mark.parametrize(
    "mask", [None, "#", {0: [True]}, iter([[True]]), [True], ["#"], [range(2)],
             [[0]], [[1]], [[None]], [[[True]]]]
)
def test_invalid_mask_types(mask):
    with pytest.raises(TypeError):
        analyze_binary_components(mask)


@pytest.mark.parametrize("mask", [[], (), [[]], [[], []], [[True], []], [[0], [True, True]]])
def test_invalid_shape_precedes_cell_validation(mask):
    with pytest.raises(ValueError):
        analyze_binary_components(mask)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("connectivity", 6), ("connectivity", 4.0), ("connectivity", "4"),
        ("connectivity", True),
        ("min_area", 0), ("min_area", 262_145), ("min_area", 1.0), ("min_area", True),
        ("max_components", 0), ("max_components", 129), ("max_components", 1.0),
        ("max_components", True),
    ],
)
def test_invalid_parameters(name, value):
    with pytest.raises(ValueError, match=name):
        analyze_binary_components([[True]], **{name: value})


def test_maximum_min_area_is_valid():
    assert analyze_binary_components([[True]], min_area=262_144)["filtered_component_count"] == 0


def test_budget_and_cell_validation_precede_work_storage(monkeypatch):
    def fail(*args):
        pytest.fail("allocated BFS storage before prevalidation")

    monkeypatch.setattr(mask_geometry, "bytearray", fail, raising=False)
    monkeypatch.setattr(mask_geometry, "deque", fail)
    # An invalid integer cell would raise TypeError if scanned before the budget.
    width_512_invalid_row = (0,) + (False,) * 511
    with pytest.raises(ValueError, match="pixel budget"):
        analyze_binary_components((width_512_invalid_row,) * 513)
    with pytest.raises(TypeError, match="mask cells"):
        analyze_binary_components(((True, 0),))


def test_public_vision_export_cold_import_is_lightweight():
    repo = Path(__file__).resolve().parents[1]
    code = """
import importlib.abc
import sys
sys.path.insert(0, sys.argv[1])
class RejectOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'numpy' or fullname.startswith('numpy.') or fullname == 'PIL' or fullname.startswith('PIL.'):
            raise AssertionError('optional image dependency imported: ' + fullname)
sys.meta_path.insert(0, RejectOptional())
import lingshu.vision
from lingshu.vision import analyze_binary_components
assert analyze_binary_components([[True]])['components'][0]['area'] == 1
assert not any(name.split('.')[0] in ('numpy', 'PIL') for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code, str(repo)],
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_module_cold_import_does_not_load_numpy_or_pillow():
    # Load the module itself; the public package has its own import contract.
    code = """
import importlib.util
import sys

spec = importlib.util.spec_from_file_location('mask_geometry', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.analyze_binary_components([[True]])['foreground_pixels'] == 1
assert not any(name.split('.')[0] in ('numpy', 'PIL') for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code, mask_geometry.__file__],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
    )
    assert completed.returncode == 0, completed.stderr
