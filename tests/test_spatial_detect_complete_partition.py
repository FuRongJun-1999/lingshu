"""The nine spatial crops must partition the complete positive-size lattice.

The coverage oracle counts unique input cell IDs passed through the public
detector's classification interface; it does not reproduce its slice formula.
Constructive color weights then exercise the real HexHierNet and image/text
pipeline without training noise or hidden ground-truth injection.
"""
from __future__ import annotations

import numpy as np
import pytest

from lingshu.nn.hex_cnn import image_to_grid
from lingshu.nn.hex_hier import HexHierNet, OBJ
from lingshu.nn.hex_text import multimodal_check, spatial_detect


class CropRecorder:
    """Minimal network protocol records every cell seen by classification."""

    def __init__(self):
        self.crops = []

    def l1(self, lat):
        return np.abs(lat[..., :1]), lat.mean(axis=(1, 2))

    def l2_features(self, lat):
        assert lat.shape[1] > 0 and lat.shape[2] > 0, "classified an empty crop"
        self.crops.append(lat.copy())
        return np.ones((len(lat), 1)), np.ones((len(lat), 3))

    def l3_logits(self, sf, cf):
        return np.tile(np.arange(len(OBJ)), (len(sf), 1))


def _color_net():
    net = HexHierNet(n_kernels=1, stacked=False)
    vec = np.zeros_like(net.get_vec())
    vec[0] = 1  # identity spatial kernel: no neighboring-cell leakage
    head = vec[7:].reshape(net.head.shape)
    for color_index, color in enumerate(("red", "green", "blue")):
        head[OBJ.index("circle|" + color), 1 + color_index] = 100
    net.set_vec(vec)
    return net


def test_quadrant_crops_cover_each_cell_exactly_once():
    """All 400 row/column pairs cover every cell, including sub-3 axes."""
    for rows in range(1, 21):
        for cols in range(1, 21):
            net = CropRecorder()
            ids = np.arange(1, rows * cols + 1).reshape(1, rows, cols, 1)
            spatial_detect(net, ids, exist_mult=0)
            seen = np.concatenate([p.ravel() for p in net.crops])
            assert np.array_equal(np.sort(seen), ids.ravel()), (
                "spatial crops omitted or repeated input cells", rows, cols, seen)
            # Scaled thirds have widths/heights differing by at most one.
            heights = [p.shape[1] for p in net.crops]
            widths = [p.shape[2] for p in net.crops]
            assert max(heights) - min(heights) <= 1
            assert max(widths) - min(widths) <= 1


@pytest.mark.parametrize("rows,cols", [(4,4), (6,7), (7,6), (18,16), (19,17), (6,6)])
def test_last_cell_energy_and_color_reach_bottom_right(rows, cols):
    lat = np.zeros((1, rows, cols, 3))
    lat[0, -1, -1, 2] = 1
    result = spatial_detect(_color_net(), lat)[0]
    assert [(d["pos"], d["obj"], d["share"]) for d in result] == [
        ("r8", "circle|blue", 1.0)], "last-cell evidence was lost or assigned to another zone"


@pytest.mark.parametrize("cells_across", [16,17,19,23])
def test_real_image_final_column_reaches_multimodal_check(cells_across):
    image = np.zeros((48,48,3), dtype=np.uint8)
    image[:, -1, 2] = 255
    lattice, _ = image_to_grid(image, cells_across=cells_across)
    net = _color_net()
    lat = lattice[None] / 255
    # Prove this source pixel is genuinely sampled into the final column.
    assert lattice[..., 2].sum() > 0
    assert np.count_nonzero(lattice[:, :-1, 2]) == 0
    found = spatial_detect(net, lat)[0]
    assert {d["pos"] for d in found} == {"r2", "r5", "r8"}, "sampled final image column disappeared"
    assert {d["obj"] for d in found} == {"circle|blue"}
    check = multimodal_check(net, lat, "右中有蓝色圆形", n_expected=3)
    assert (check["verdict"], check["align"], check["count_match"]) == ("ACCEPT", 1.0, True)


def test_energy_and_classification_use_the_same_uneven_crops():
    net = CropRecorder()
    lat = np.arange(1, 6*8+1, dtype=np.float64).reshape(1,6,8,1)
    found = spatial_detect(net, lat, exist_mult=0)[0]
    assert len(found) == len(net.crops) == 9
    expected = {f"r{q}": round(float(p.sum()/lat.sum()), 3)
                for q, p in enumerate(net.crops)}
    assert {d["pos"]: d["share"] for d in found} == expected, (
        "energy shares and classified crops use different partitions")


def test_divisible_grid_keeps_original_crops_and_results():
    net = CropRecorder()
    lat = np.arange(1, 6*9+1, dtype=np.float64).reshape(1,6,9,1)
    found = spatial_detect(net, lat, exist_mult=0)[0]
    legacy = [lat[:, y:y+2, x:x+3] for y in (0,2,4) for x in (0,3,6)]
    assert all(np.array_equal(a,b) for a,b in zip(net.crops,legacy))
    assert len(found) == 9
    assert {d["obj"] for d in found} == {OBJ[-1]}


def test_batched_samples_partition_independently():
    lat = np.zeros((2,4,5,3))
    lat[0,-1,-1,2] = 1
    lat[1,-1,-1,1] = 1
    result = spatial_detect(_color_net(),lat)
    assert [[(d["pos"],d["obj"]) for d in sample] for sample in result] == [
        [("r8","circle|blue")],[("r8","circle|green")]]


def test_empty_batch_remains_empty():
    assert spatial_detect(_color_net(),np.zeros((0,4,5,3))) == []


@pytest.mark.parametrize("rows,cols", [(1,1), (2,2), (1,5), (5,2)])
def test_dark_tiny_lattice_does_not_classify_empty_quadrants(rows, cols):
    net = CropRecorder()
    assert spatial_detect(net,np.zeros((1,rows,cols,1))) == [[]]
    assert not net.crops
