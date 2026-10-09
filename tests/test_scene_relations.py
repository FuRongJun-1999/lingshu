import pytest

from lingshu.world.scene_model import WorldModel


@pytest.mark.parametrize("kind,offset", [
    ("left_of", 2), ("left_of", -2), ("left_of", 0),
    ("right_of", -2), ("right_of", 2), ("right_of", 0),
])
def test_build_enforces_horizontal_relation_and_preserves_valid_position(kind, offset):
    wm = WorldModel()
    original_pos = (10 + offset, 0.45, 5)
    wm.add_entity("chair", "chair", original_pos)
    wm.add_entity("table", "table", (10, 0.45, 5))
    wm.relation("chair", kind, "table").build()
    pos = wm.entities["chair"].pos
    assert pos[0] < 10 if kind == "left_of" else pos[0] > 10
    if (kind == "left_of" and offset < 0) or (kind == "right_of" and offset > 0):
        assert pos == original_pos
    wm.build()
    assert wm.entities["chair"].pos == pos


@pytest.mark.parametrize("kind,offset", [("in_front", 2), ("behind", -2)])
def test_depth_relations_remain_enforced(kind, offset):
    wm = WorldModel()
    wm.add_entity("chair", "chair", (0, 0.45, 5 + offset))
    wm.add_entity("table", "table", (0, 0.45, 5))
    wm.relation("chair", kind, "table").build()
    z = wm.entities["chair"].pos[2]
    assert z < 5 if kind == "in_front" else z > 5
