"""HexGrid.neighbors must match its mode's Cartesian lattice, in direction order.

The oracle enumerates documented Cartesian centers and sorts the nearest six
geometrically. It does not use HEX_DIRS, hex_conv, or pixel_to_cell.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lingshu.nn.hex_cnn import HexGrid, image_to_grid


def _center(mode, size, origin, q, r):
    if mode == "odd_r_offset":
        return (math.sqrt(3) * size * (q + 0.5 * (r % 2)) + origin[0],
                1.5 * size * r + origin[1])
    return (math.sqrt(3) * size * (q + r / 2), 1.5 * size * r)


def _nearest_six(grid, q, r):
    """7x7 point-cloud oracle; angle order is E, NE, NW, W, SW, SE."""
    cx, cy = _center(grid.mode, grid.size, grid.origin, q, r)
    candidates = []
    for row in range(r - 3, r + 4):
        for col in range(q - 3, q + 4):
            if (col, row) == (q, r):
                continue
            x, y = _center(grid.mode, grid.size, grid.origin, col, row)
            dx, dy = x - cx, y - cy
            candidates.append((dx * dx + dy * dy,
                               (-math.atan2(dy, dx)) % math.tau, (col, row)))
    nearest = sorted(candidates)[:6]
    assert all(math.isclose(item[0], 3 * grid.size ** 2, rel_tol=1e-10,
                            abs_tol=1e-12) for item in nearest)
    return [item[2] for item in sorted(nearest, key=lambda item: item[1])]


@pytest.mark.parametrize("q,r", [
    (3, 2), (3, 3), (-3, 2), (-3, 3),
    (3, -2), (3, -1), (-3, -2), (-3, -3),
], ids=["even", "odd", "negative_col_even", "negative_col_odd",
        "negative_even", "negative_odd", "both_negative_even", "both_negative_odd"])
def test_odd_r_neighbors_keep_direction_order(q, r):
    grid = HexGrid(8, 7, mode="odd_r_offset")
    assert grid.neighbors(q, r) == _nearest_six(grid, q, r), (
        "odd-r neighbors must follow E, NE, NW, W, SW, SE")


@pytest.mark.parametrize("size,origin", [
    (0.25, (0.0, 0.0)), (1.0, (-9.3, 13.75)),
    (2.5, (1024.0, -2048.0)), (10.0, (-17.0, 31.0)),
], ids=["small", "translated", "large_origin", "large_size"])
@pytest.mark.parametrize("mode", ["axial", "odd_r_offset"])
def test_neighbors_match_independent_cartesian_oracle(mode, size, origin):
    # The axial contract fixes its origin at zero; offset grids allow translation.
    grid = HexGrid(8, 7, size, mode=mode,
                   origin=origin if mode == "odd_r_offset" else (0.0, 0.0))
    for r in range(-4, 5):
        for q in range(-4, 5):
            assert grid.neighbors(q, r) == _nearest_six(grid, q, r), (
                "six ordered neighbors must be the six nearest Cartesian centers",
                mode, size, origin, q, r)


@pytest.mark.parametrize("width,height", [(1, 1), (2, 1), (1, 2), (7, 3)],
                         ids=["single", "one_row", "one_column", "rectangle"])
@pytest.mark.parametrize("mode", ["axial", "odd_r_offset"])
def test_neighbors_remain_unclipped_at_shape_boundaries(mode, width, height):
    grid = HexGrid(width, height, 0.75, mode=mode)
    for q, r in [(0, 0), (width - 1, height - 1), (-2, -1), (width, height)]:
        actual = grid.neighbors(q, r)
        assert len(actual) == 6, "neighbors must return six cells before in_bounds filtering"
        assert len(set(actual)) == 6
        assert actual == _nearest_six(grid, q, r)
        assert any(not grid.in_bounds(*cell) for cell in actual)


def test_odd_r_neighbors_are_reciprocal():
    grid = HexGrid(7, 7, mode="odd_r_offset")
    for r in range(-4, 5):
        for q in range(-4, 5):
            assert all((q, r) in grid.neighbors(*cell) for cell in grid.neighbors(q, r))


@pytest.mark.parametrize("height,width,across,interior_count", [
    (64, 64, 8, 42), (48, 96, 16, 98), (96, 48, 12, 260),
], ids=["square", "landscape", "portrait"])
def test_image_to_grid_exposes_correct_odd_r_neighbors(height, width, across, interior_count):
    feature, grid = image_to_grid(np.full((height, width, 3), 127, dtype=np.uint8), across)
    assert grid.mode == "odd_r_offset"
    assert feature.shape == (grid.H, grid.W, 3)
    assert feature.dtype == np.float32 and np.all(feature == 127)
    checked = 0
    for r in range(1, grid.H - 1):
        for q in range(1, grid.W - 1):
            expected = _nearest_six(grid, q, r)
            assert all(grid.in_bounds(*cell) for cell in expected)
            assert grid.neighbors(q, r) == expected, (
                "image_to_grid's returned grid must expose its nearest six centers")
            checked += 1
    assert checked == interior_count


@pytest.mark.parametrize("explicit_mode", [False, True], ids=["default", "explicit"])
def test_axial_neighbor_list_and_order_are_unchanged(explicit_mode):
    grid = HexGrid(1, 1, **({"mode": "axial"} if explicit_mode else {}))
    for q, r in [(3, 2), (3, 3), (-3, -2), (-3, -1), (0, 0)]:
        assert grid.neighbors(q, r) == [
            (q + 1, r), (q + 1, r - 1), (q, r - 1),
            (q - 1, r), (q - 1, r + 1), (q, r + 1),
        ], "axial neighbor list and direction order must remain unchanged"
