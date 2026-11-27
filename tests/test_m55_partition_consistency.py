"""The M5.5 bridge and detector must consume the same complete quadrant evidence."""
import numpy as np
import pytest

from lingshu.core.core import SpacetimeMemoryEngine
from lingshu.nn.hex_hier import HexHierNet, OBJ
from lingshu.nn.hex_text import (
    extract_perception_cards,
    ingest_perception_cards,
    query_visual_memory,
    spatial_detect,
)


class CardCropRecorder(HexHierNet):
    def __init__(self):
        super().__init__(n_kernels=1, stacked=False)
        self.crops = []

    def l1(self, lat):
        return np.abs(lat[..., :1]), lat.mean(axis=(1, 2))

    def l2_features(self, lat):
        assert lat.shape[1] and lat.shape[2], "bridge classified an empty region"
        self.crops.append(lat.copy())
        return np.ones((len(lat), 1)), np.ones((len(lat), 3))

    def l3_logits(self, sf, cf):
        return np.tile(np.arange(len(OBJ)), (len(sf), 1))


def color_net():
    net = HexHierNet(n_kernels=1, stacked=False)
    vector = np.zeros_like(net.get_vec())
    vector[0] = 1
    head = vector[7:].reshape(net.head.shape)
    for index, color in enumerate(("red", "green", "blue")):
        head[OBJ.index("circle|" + color), 1 + index] = 100
    net.set_vec(vector)
    return net


def test_card_crops_cover_all_cells_with_independent_region_ownership():
    net = CardCropRecorder()
    for rows in range(3, 21):
        for cols in range(3, 21):
            ids = np.arange(1, rows * cols + 1).reshape(1, rows, cols, 1)
            net.crops.clear()
            cards = extract_perception_cards(net, ids, exist_mult=1e-12)[0]
            seen = np.concatenate([crop.ravel() for crop in net.crops])
            assert np.array_equal(np.sort(seen), ids.ravel()), (
                "bridge omitted or repeated finite lattice cells", rows, cols)
            expected = {}
            for y in range(rows):
                for x in range(cols):
                    # Invert per-cell ownership, independently from crop boundaries.
                    q = ((3 * (y + 1) - 1) // rows) * 3 + (3 * (x + 1) - 1) // cols
                    expected.setdefault(q, []).append(int(ids[0, y, x, 0]))
            assert len(net.crops) == len(cards) == len(expected)
            for crop, q in zip(net.crops, sorted(expected)):
                assert sorted(crop.ravel()) == expected[q]
            denominator = sum(range(1, rows * cols + 1))
            for card in cards:
                q = int(card["pos"][1:])
                assert card["share"] == pytest.approx(sum(expected[q]) / denominator, abs=0.000050001)


@pytest.mark.parametrize("rows,cols", [(4, 4), (6, 7), (7, 6), (18, 16), (19, 17), (6, 6)])
def test_border_evidence_gives_the_same_card_and_detection(rows, cols):
    lat = np.zeros((1, rows, cols, 3))
    lat[0, -1, -1, 2] = 1
    net = color_net()
    before = lat.copy()
    cards = extract_perception_cards(net, lat)[0]
    detection = spatial_detect(net, lat)[0]
    assert [(c["pos"], c["obj"], c["share"]) for c in cards] == [
        ("r8", "circle|blue", 1.0)], "bridge lost the final-cell evidence"
    assert [(d["pos"], d["obj"], d["share"]) for d in detection] == [
        ("r8", "circle|blue", 1.0)]
    assert cards[0]["card_verdict"] == "ACCEPT"
    assert cards[0]["conf"] == pytest.approx(detection[0]["conf"], abs=0.0005)
    assert np.array_equal(lat, before)


@pytest.mark.parametrize("rows,cols", [(3, 4), (4, 4), (5, 5), (18, 16)])
@pytest.mark.parametrize("value", [0.0, 0.25, 1.0])
def test_uniform_evidence_is_scored_by_actual_region_area(rows, cols, value):
    lat = np.zeros((1, rows, cols, 3))
    lat[..., 0] = value
    net = color_net()
    cards = extract_perception_cards(net, lat)[0]
    detection = spatial_detect(net, lat)[0]
    assert len(cards) == len(detection) == 1, "bridge mistook unequal areas for extra evidence"
    assert cards[0]["pos"] == detection[0]["pos"]
    if value:
        chosen = int(cards[0]["pos"][1:])
        cells = sum((((3 * (y + 1) - 1) // rows) * 3
                     + (3 * (x + 1) - 1) // cols) == chosen
                    for y in range(rows) for x in range(cols))
        assert cards[0]["share"] == round(cells / (rows * cols), 4)


def test_last_cell_reaches_ingestion_and_knowledge_query():
    lat = np.zeros((1, 4, 4, 3))
    lat[0, -1, -1, 2] = 1
    engine = SpacetimeMemoryEngine(":memory:")
    try:
        result = ingest_perception_cards(engine, color_net(), lat, image_id="edge")
        assert len(result["node_ids"]) == 1
        node = engine.store.get_node(result["node_ids"][0])
        assert node.semantic_coordinates["pos"] == "r8"
        assert node.semantic_coordinates["color"] == "blue"
        assert node.state_attributes["qualified"] is True
        answer = query_visual_memory(engine, "图像里有什么", image_id="edge")
        assert answer["found"] is True
        assert "右下" in answer["answer"] and "蓝圆形" in answer["answer"]
    finally:
        engine.close()