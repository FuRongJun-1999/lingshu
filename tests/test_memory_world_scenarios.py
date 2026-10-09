"""Business journeys through persistent memory and the observation/prediction loop.

The expected graph and trajectories belong to the scenario, not to the system
under test. Only public APIs are used; files and databases stay in ``tmp_path``.
"""

from contextlib import closing
import json
import math

import pytest

from lingshu.core.core import ConditionSpace, EdgeType, SpacetimeMemoryEngine
from lingshu.world.scene_simulator import SceneSimulator
from lingshu.world.wm_simloop import SimulationLoop
from lingshu.world.world_model import UnifiedWorldModel


def test_observations_and_causal_evidence_survive_a_memory_restart(tmp_path):
    database = str(tmp_path / "greenhouse.db")
    source = ConditionSpace(
        observation_position="greenhouse-A",
        observation_tool="maintenance-log",
        time_window=(1_700_000_000.0, 1_700_003_600.0),
        existence_constraint="irrigation system in service",
    )
    with closing(SpacetimeMemoryEngine(database)) as memory:
        pump = memory.add_perception(
            "pump offline", condition_space=source,
            spatial_coordinates={"x": 3.0, "z": 7.0},
            tags=["incident:irrigation"],
        )
        soil = memory.add_perception("soil moisture fell", condition_space=source)
        plants = memory.add_perception("seedlings wilted", condition_space=source)
        stopped_watering = memory.add_edge(
            pump.id, soil.id, EdgeType.CAUSAL, confidence=0.8,
            condition_space=source, source_evidence="extracted",
        )
        wilted = memory.add_edge(
            soil.id, plants.id, EdgeType.CAUSAL, confidence=0.6,
            condition_space=source, source_evidence="inferred",
        )
        # A correlation is useful evidence, but is not a causal shortcut.
        memory.add_edge(pump.id, plants.id, EdgeType.CORRELATIONAL,
                        condition_space=source)
        memory.verify_edge(wilted.id, new_confidence=0.85)

    with closing(SpacetimeMemoryEngine(database)) as reopened:
        recalled = reopened.recall("pump offline")
        assert [node.id for node, _ in recalled] == [pump.id]
        restored = recalled[0][0]
        assert restored.content == "pump offline"
        assert restored.condition_space == source
        assert restored.spatial_coordinates == {"x": 3.0, "z": 7.0}
        assert "incident:irrigation" in restored.tags

        paths = reopened.reason_causal(pump.id, plants.id)
        assert [[edge.id for edge in path] for path in paths] == [
            [stopped_watering.id, wilted.id]
        ]
        assert all(edge.condition_space == source for edge in paths[0])
        assert [edge.source_evidence for edge in paths[0]] == ["extracted", "inferred"]
        assert paths[0][1].verified is True
        assert paths[0][1].confidence == pytest.approx(0.85)


@pytest.fixture
def moving_cart():
    """Three camera frames establish motion of one metre per observation."""
    model = UnifiedWorldModel()
    for x in (2, 3, 4):
        model.perceive(
            [{"eid": "cart", "category": "cart", "pos": (x, 1.5, 4)}],
            tool="aisle-camera",
        )
    return model


def test_next_camera_frame_confirms_the_predicted_motion(moving_cart):
    confidence_before = moving_cart.graph()["nodes"]["cart"]["confidence"]
    prediction = moving_cart.generate()
    assert prediction["predictions"]["cart"]["predicted"] == [5.0, 1.5, 4.0]

    observed = moving_cart.perceive(
        [{"eid": "cart", "category": "cart", "pos": (5, 1.5, 4)}],
        tool="aisle-camera",
    )
    verified = moving_cart.verify()

    assert observed["matched"] == 1
    assert observed["consistent"] == 1
    assert (verified["hits"], verified["total"], verified["pending"]) == (1, 1, 0)
    assert verified["hit_rate"] == 1.0
    detail = verified["details"][0]
    assert detail["actual"] == [5.0, 1.5, 4.0]
    assert detail["distance"] == 0.0
    assert detail["hit"] is True
    node = moving_cart.graph()["nodes"]["cart"]
    assert node["confidence"] > confidence_before
    assert node["conditions"]["observation_tool"] == "aisle-camera"
    assert (node["first_seen"], node["last_seen"]) == (1, 4)


def test_a_real_turn_updates_the_world_and_records_the_failed_prediction(moving_cart):
    confidence_before = moving_cart.graph()["nodes"]["cart"]["confidence"]
    moving_cart.generate()
    # The cart turns into a different aisle instead of continuing along x.
    observed = moving_cart.perceive(
        [{"eid": "cart", "category": "cart", "pos": (4, 1.5, 6)}],
        tool="aisle-camera",
    )
    verified = moving_cart.verify()

    assert observed["anomaly_events"] == 1
    assert (verified["hits"], verified["total"]) == (0, 1)
    assert verified["hit_rate"] == 0.0
    detail = verified["details"][0]
    assert detail["predicted"] == [5.0, 1.5, 4.0]
    assert detail["actual"] == [4.0, 1.5, 6.0]
    assert detail["distance"] == pytest.approx(math.sqrt(5), abs=0.00005)
    assert detail["hit"] is False

    anomaly, = moving_cart.anomalies()
    assert anomaly["entity"] == "cart"
    assert anomaly["expected"] == [5.0, 1.5, 4.0]
    assert anomaly["observed"] == [4.0, 1.5, 6.0]
    node = moving_cart.graph()["nodes"]["cart"]
    assert node["pos"] == (4.0, 1.5, 6.0)
    assert node["confidence"] < confidence_before


def test_simulation_journal_and_memory_payload_carry_the_observed_world(tmp_path, monkeypatch):
    monkeypatch.delenv("MDCG_ROOT", raising=False)
    scene = SceneSimulator()
    cart = scene.add_entity("cart", pos=(2, 1.5, 4), speed=0)
    journal = tmp_path / "simulation.jsonl"
    loop = SimulationLoop(scene=scene, wal_path=str(journal), clock=lambda: 1000.0)
    loop.wm.perceive()

    def advance_cart(world, tick):
        world.entities[cart].pos = (1 + tick, 1.5, 4)

    loop.step(n=3, external=advance_cart)
    records = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    assert [record["seq"] for record in records] == [0, 1, 2]
    assert [record["kind"] for record in records] == ["episode"] * 3
    assert [record["tick"] for record in records] == [2, 3, 4]
    assert [next(iter(record["payload"]["nodes"].values()))["pos"] for record in records] == [
        [3.0, 1.5, 4.0], [4.0, 1.5, 4.0], [5.0, 1.5, 4.0]
    ]

    # Read the persisted artifact in a fresh loop, not its own in-memory WAL.
    replayed = SimulationLoop().replay(records)
    assert replayed["episodes"] == 3
    assert replayed["tick"] == 4
    replayed_cart, = replayed["nodes"].values()
    assert replayed_cart["category"] == "cart"
    assert replayed_cart["pos"] == [5.0, 1.5, 4.0]

    payload_path = tmp_path / "contextual" / "cart-run.md"
    exported = loop.flush_payloads(out_path=str(payload_path))
    assert payload_path.read_text(encoding="utf-8") == exported["payload"]["body_md"]
    assert exported["payload"]["frontmatter"]["verification_basis"] == "wal_replay"

    # load_priors promises position/category priors; it does not restore behavior.
    resumed = SimulationLoop()
    loaded = resumed.load_priors(str(tmp_path))
    assert loaded["seeded"] == 1
    restored_cart, = resumed.scene.entities.values()
    assert restored_cart.category == "cart"
    assert restored_cart.pos == (5.0, 1.5, 4.0)
