import pytest
import numpy as np
from lingshu.nn import hex_gen as G

def test_verify_readback_empty_parts_returns_none():
    """dsh-bounty #2223: verify_readback with empty parts must return pattern_acc=None instead of 0.0."""
    img = np.zeros((96, 96, 3), dtype=np.uint8)
    res = G.verify_readback(img, [], [])
    assert res["n_pattern"] == 0
    assert res["pattern_acc"] is None
    assert res["n_size"] == 0
    assert res["size_acc"] is None

def test_verify_readback_non_rgb_color_returns_none():
    """dsh-bounty #2223: parts with color outside READBACK_COLORS (e.g. orange) are skipped by _readback_records,
    so pattern_acc must return None instead of masquerading as 0.0 accuracy."""
    parts = [{"shape": "circle", "color": "orange", "zone": "r4",
              "pattern": "solid", "size": "medium"}]
    img, log = G.render_relations(parts, size=96, seed=7, noise=False)
    res = G.verify_readback(img, parts, log)
    assert res["n_pattern"] == 0
    assert res["pattern_acc"] is None
    assert res["n_size"] == 0
    assert res["size_acc"] is None

def test_verify_readback_valid_rgb_color_returns_float():
    """dsh-bounty #2223: valid RGB parts calculate normal positive accuracy."""
    parts = [{"shape": "circle", "color": "red", "zone": "r4",
              "pattern": "solid", "size": "medium"}]
    img, log = G.render_relations(parts, size=96, seed=7, noise=False)
    res = G.verify_readback(img, parts, log)
    assert res["n_pattern"] == 1
    assert res["pattern_acc"] == 1.0
