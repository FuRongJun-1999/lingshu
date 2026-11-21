"""odd_r_offset pixel lookup must return the nearest cell center."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lingshu.nn.hex_cnn import HexGrid


def _distance2(grid, point, cell):
    center = grid.center_px(*cell)
    return (center[0] - point[0]) ** 2 + (center[1] - point[1]) ** 2


def test_odd_r_checks_both_neighbor_rows_for_nearest_center():
    grid = HexGrid(5, 5, 1.0, mode="odd_r_offset")
    point = (math.sqrt(3.0) / 2.0, 0.74)
    expected = (0, 1)
    actual = grid.pixel_to_cell(*point)
    assert actual == expected, "nearest odd-r center is in the adjacent staggered row"
    assert math.sqrt(_distance2(grid, point, actual)) == pytest.approx(0.76)


@pytest.mark.parametrize("size,origin", [
    (0.5, (0.0, 0.0)),
    (1.0, (2.25, -0.75)),
    (2.75, (-4.0, 3.5)),
])
def test_odd_r_pixel_lookup_matches_brute_force_nearby_centers(size, origin):
    grid = HexGrid(8, 7, size, mode="odd_r_offset", origin=origin)
    spacing_x = math.sqrt(3.0) * size
    spacing_y = 1.5 * size
    ox, oy = origin
    for iy in range(-8, 25):
        y = oy + iy * size * 0.23
        rf = (2.0 / 3.0 * (y - oy)) / size
        row0 = math.floor(rf)
        for ix in range(-12, 28):
            x = ox + ix * size * 0.19
            qf = (x - ox) / spacing_x
            q0 = math.floor(qf)
            candidates = [
                (q, r)
                for r in range(row0 - 2, row0 + 4)
                for q in range(q0 - 3, q0 + 4)
            ]
            expected = min(_distance2(grid, (x, y), cell) for cell in candidates)
            actual = _distance2(grid, (x, y), grid.pixel_to_cell(x, y))
            assert actual == pytest.approx(expected, abs=1e-12), (x, y, actual, expected)
