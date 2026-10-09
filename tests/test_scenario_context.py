"""跨任务场景验收：同图同源、角色/会话隔离、候选证据、预算与真实 MCP。"""
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from lingshu.core.core import ConditionSpace, EdgeType, LayeredStore, STNode, STEdge
from lingshu.world.scenario import ScenarioContext


def add(store, nid, text, tags=(), when=100, pos=None, state=None):
    n = STNode(nid, text, "text", pos or {}, when,
               ConditionSpace("archive-A", "document-reader", (when, when + 10), "scenario-A"),
               tags=list(tags), state_attributes={"state": state} if state else {})
    store.add_node(n)
    return n


def link(store, eid, source, target, rel=EdgeType.CAUSAL, verified=False):
    store.add_edge(STEdge(eid, source, target, rel,
        ConditionSpace("archive-A", "document-reader", (100, 300), "scenario-A"),
        verified=verified, source_evidence="extracted"))


@pytest.fixture
def journey(tmp_path):
    store = LayeredStore(str(tmp_path / "scenario.db"))
    add(store, "alice", "Alice, lighthouse keeper", ["roleplay", "role:alice", "roleplay:role"])
    add(store, "rules", "The island lighthouse uses blue crystal", ["sub-knowledge", "role:alice"])
    add(store, "lamp0", "Lamp unlit", ["role:alice", "ent:lamp", "cat:table"],
        when=110, pos={"x": 1, "y": 2, "z": 3}, state="off")
    add(store, "lamp1", "Lamp lit", ["role:alice", "ent:lamp", "cat:table"],
        when=120, pos={"x": 2, "y": 2, "z": 3}, state="on")
    # Late observation of the cause: causal order must still precede its effect.
    add(store, "switch", "Alice activates crystal", ["role:alice", "event"], when=150)
    add(store, "turn", "I have lit the lamp", ["roleplay", "role:alice", "session:s1", "turn:assistant"])
    add(store, "private", "A different conversation", ["role:alice", "session:s2", "turn:user"])
    add(store, "bob", "Bob's planet has two suns", ["role:bob", "sub-knowledge"])
    add(store, "orphan", "Unassigned fictional lore", ["sub-knowledge"])
    add(store, "real", "Catalogue describes an oil lamp", ["external", "vref:catalogue:42"])
    add(store, "guess", "Perhaps the lamp was ceremonial", ["assistant", "session:s1"])
    link(store, "lit", "switch", "lamp1")
    link(store, "cross-role", "bob", "lamp1")
    link(store, "correlation", "switch", "lamp0", EdgeType.CORRELATIONAL)
    ids = ["alice", "rules", "lamp0", "lamp1", "switch", "turn", "private", "bob", "orphan", "real", "guess"]
    try:
        yield store, ids
    finally:
        store.conn.close()


def test_one_persistent_graph_drives_five_tasks_without_writes(journey):
    store, ids = journey
    before = list(store.conn.iterdump())
    ctx = ScenarioContext.from_store(store, ids, role_id="alice", session_id="s1")
    role = ctx.view("roleplay")
    assert role["payload"]["identity"][0]["id"] == "alice"
    assert [n["id"] for n in role["payload"]["lore"]] == ["rules"]
    assert [n["id"] for n in role["payload"]["turns"]] == ["turn"]
    world = ctx.view("world")
    lamp, = world["payload"]["entities"]
    assert lamp == {"name": "lamp", "node_id": "lamp1", "category": "table",
                    "pos": [2, 2, 3], "state": "on"}
    story = ctx.view("story")
    outline = [step["node_id"] for step in story["payload"]["outline"]]
    assert outline.index("switch") < outline.index("lamp1")
    assert story["payload"]["fiction"] is True
    assert "bob" not in outline and "private" not in outline
    history = ctx.view("history")
    assert {n["id"] for n in history["facts"]} == {"real", "guess"}
    assert history["state"] == "DEFER"  # 本地记录没有脑端资格，引用不等于确认。
    assert history["payload"]["evidence_timeline"] == []
    assert {n["id"] for n in history["payload"]["unresolved"]} == {"real", "guess"}
    causal = ctx.view("causal", start_id="switch", end_id="lamp1")
    path, = causal["payload"]["paths"]
    assert path["node_ids"] == ["switch", "lamp1"]
    assert [e["id"] for e in path["edges"]] == ["lit"]
    assert path["hypothesis"] is True and causal["state"] == "DEFER"
    assert path["edges"][0]["conditions"]["existence_constraint"] == "scenario-A"
    assert ctx.view("causal", start_id="bob", end_id="lamp1")["state"] == "REJECT"
    assert ctx.view("causal", start_id="absent", end_id="lamp1")["state"] == "BLINDSPOT"
    assert ctx.view("causal", start_id="switch", end_id="lamp0")["payload"]["paths"] == []
    assert list(store.conn.iterdump()) == before


def test_no_role_or_session_scope_does_not_guess_visibility(journey):
    store, ids = journey
    ctx = ScenarioContext.from_store(store, ids)
    assert {n["id"] for n in ctx.view("world")["facts"]} == {"real"}
    assert ctx.view("roleplay")["state"] == "BLINDSPOT"
    alice = ScenarioContext.from_store(store, ids, role_id="alice")
    visible = {n["id"] for n in alice.view("roleplay")["facts"]}
    assert "turn" not in visible and "private" not in visible


def test_existing_role_imports_stay_constraints_instead_of_story_events(journey):
    store, ids = journey
    add(store, "anchor", "Never betray the lighthouse", ["role:alice", "roleplay:anchors"])
    add(store, "value", "Prefer honesty", ["role:alice", "roleplay:values"])
    add(store, "memory", "Remember the former keeper", ["role:alice", "roleplay:memory"])
    ctx = ScenarioContext.from_store(store, ids + ["anchor", "value", "memory"], role_id="alice")
    role = ctx.view("roleplay")["payload"]
    assert [n["id"] for n in role["anchors"]] == ["anchor"]
    assert [n["id"] for n in role["values"]] == ["value"]
    assert [n["id"] for n in role["memory"]] == ["memory"]
    story = ctx.view("story")["payload"]
    assert {"anchor", "value", "memory", "rules", "alice"}.issubset({n["id"] for n in story["background"]})
    assert not {"anchor", "value", "memory", "rules", "alice"}.intersection({n["node_id"] for n in story["outline"]})


def test_fictional_cause_cannot_become_a_real_world_evidence_path(journey):
    store, ids = journey
    link(store, "fiction-to-real", "switch", "real", verified=True)
    ctx = ScenarioContext.from_store(store, ids, role_id="alice")
    report = ctx.view("causal", start_id="switch", end_id="real")
    assert report["payload"]["paths"] == []
    assert "cross_domain_edge:fiction-to-real" in report["gaps"]
    assert ctx.view("history")["domain"] == "real"
    assert ctx.view("world")["domain"] == "role"


def test_snapshots_and_return_values_cannot_change_store_or_other_views(journey):
    store, ids = journey
    ctx = ScenarioContext.from_store(store, ids, role_id="alice")
    report = ctx.view("world")
    report["facts"][0]["conditions"]["existence_constraint"] = "foreign world"
    report["facts"].clear()
    store.conn.execute("UPDATE nodes SET content='changed later' WHERE id='lamp1'")
    store.conn.commit()
    assert next(n for n in ctx.view("world")["facts"] if n["id"] == "lamp1")["content"] == "Lamp lit"
    assert all(n["conditions"]["existence_constraint"] == "scenario-A" for n in ctx.view("world")["facts"])


def test_restart_and_existing_world_model_share_latest_entity_state(journey):
    store, ids = journey
    reopened = LayeredStore(store.db_path)
    try:
        ctx = ScenarioContext.from_store(reopened, list(reversed(ids)), role_id="alice", session_id="s1")
        model = ctx.to_world_model()
        assert set(model.entities) == {"lamp"}
        assert model.entities["lamp"].pos == (2, 2, 3)
        assert model.entities["lamp"].state == "on"
    finally:
        reopened.conn.close()


def test_causal_depth_cycles_and_parallel_paths_are_explicit(tmp_path):
    store = LayeredStore(str(tmp_path / "paths.db"))
    try:
        for index in range(9):
            add(store, str(index), f"stage {index}")
        for index in range(8):
            for branch in range(2):
                link(store, f"{index}-{branch}", str(index), str(index + 1))
        link(store, "cycle", "2", "0")
        ctx = ScenarioContext.from_store(store, [str(i) for i in range(9)])
        short = ctx.view("causal", start_id="0", end_id="8", max_depth=5)
        assert short["payload"]["paths"] == [] and short["payload"]["truncated"] is True
        full = ctx.view("causal", start_id="0", end_id="8", max_paths=300)
        assert len(full["payload"]["paths"]) == 256
        assert full["payload"]["truncated"] is False and full["payload"]["cyclic"] is True
        assert all(p["node_ids"] == [str(i) for i in range(9)] for p in full["payload"]["paths"])
        bounded = ctx.view("causal", start_id="0", end_id="8", max_paths=3)
        assert len(bounded["payload"]["paths"]) == 3 and bounded["payload"]["truncated"] is True
        exhausted = ctx.view("causal", start_id="0", end_id="8", max_steps=2)
        assert exhausted["payload"]["steps"] == 2 and exhausted["payload"]["truncated"] is True
        assert ctx.view("story")["payload"]["cyclic"]
    finally:
        store.conn.close()


def test_missing_nodes_and_edge_budget_do_not_silently_claim_completeness(journey):
    store, ids = journey
    ctx = ScenarioContext.from_store(store, ids + ["missing"], role_id="alice", max_edges=1)
    assert "missing_node:missing" in ctx.view("story")["gaps"]
    assert "edge_budget_exhausted" in ctx.view("story")["gaps"]
    assert ctx.view("story")["state"] == "DEFER"
    with pytest.raises(ValueError):
        ScenarioContext.from_store(store, ids, max_nodes=1)


class BrainRead:
    """协议单元测试只覆盖裁决/分页元数据；真传输另由 MCP 验收。"""
    def __init__(self, results):
        self.results = results
        self.calls = []

    def call(self, name, args):
        self.calls.append((name, args))
        return {"results": self.results}


def record(nid, state="ACCEPT", tags=None, cs=None, **extra):
    return {"state": state, "node": {"id": nid, "content": "archival record",
        "frontmatter": {"tags": tags or ["external", "vref:catalogue:42"],
            "condition_space": cs or asdict(ConditionSpace("archive-A", "scanner", (100, 110), "museum"))},
        **extra}}


def test_brain_qualification_sources_and_truncation_remain_visible():
    brain = BrainRead([record("ok"), record("rejected", "REJECT"), record("pending", "DEFER"),
                       record("clip", truncated=True, next_offset=20),
                       record("fiction", tags=["role:alice", "sub-knowledge", "external", "vref:fiction"]),
                       record("explicit", tags=["external", "novel_prefeed", "vref:document"])])
    ctx = ScenarioContext.from_brain(brain, "archival", role_id="alice", session_id="s1", k=8)
    report = ctx.view("history")
    assert "rejected" not in {n["id"] for n in report["facts"]}
    assert "fiction" not in {n["id"] for n in report["facts"]}
    assert "body_window_truncated:clip" in report["gaps"]
    assert [n["id"] for n in report["payload"]["unresolved"]] == ["clip", "pending"]
    explicit = next(n for n in report["facts"] if n["id"] == "explicit")
    assert explicit["provenance"]["source"] == "external"
    assert brain.calls == [("cg", {"op": "read", "query": "archival", "k": 8, "session": "s1"})]
    assert "recall_limit_reached" in ScenarioContext.from_brain(brain, "archival", k=6).view("history")["gaps"]


def test_host_state_does_not_leak_into_same_named_role_entity():
    real = record("real-keeper", tags=["external", "ent:keeper"])
    role = record("role-keeper", tags=["role:alice", "ent:keeper"])
    for r in (real, role):
        r["node"]["frontmatter"]["spatial"] = {"coords3d": {"x": 0, "y": 1, "z": 5}}
    class States(BrainRead):
        def call(self, name, args):
            if name == "stg":
                return {"items": [{"subject": "keeper", "slot": "状态", "state": "active", "value": "happy"}]}
            return super().call(name, args)
    ctx = ScenarioContext.from_brain(States([real, role]), "keeper", role_id="alice")
    report = ctx.view("world")
    assert report["payload"]["entities"] == []
    assert "ambiguous_entity_domain:keeper" in report["gaps"]
    assert next(n for n in report["facts"] if n["id"] == "real-keeper")["attributes"]["state"] == "happy"
    assert next(n for n in report["facts"] if n["id"] == "role-keeper")["attributes"] == {}


def test_invalid_selection_and_oversized_brain_response_do_not_change_budget(journey):
    store, _ = journey
    for selection in ("alice", [None], [""]):
        with pytest.raises(ValueError):
            ScenarioContext.from_store(store, selection)
    with pytest.raises(RuntimeError, match="more candidates"):
        ScenarioContext.from_brain(BrainRead([record("a"), record("b")]), "record", k=1)


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_invalid_causal_budget_fails_explicitly(journey, bad):
    store, ids = journey
    with pytest.raises(ValueError):
        ScenarioContext.from_store(store, ids).view("causal", start_id="real", end_id="real", max_steps=bad)


def test_undeclared_conditions_unknown_modes_and_bad_brain_responses(journey):
    store, ids = journey
    with pytest.raises(ValueError):
        ScenarioContext.from_store(store, ids).view("unknown")
    ctx = ScenarioContext.from_brain(BrainRead([record("unknown", cs={"existence_constraint": "museum"})]), "record")
    assert "undeclared_conditions:unknown" in ctx.view("history")["gaps"]
    class Broken:
        def call(self, *args):
            return {"ok": False, "error": "denied"}
    with pytest.raises(RuntimeError, match="denied"):
        ScenarioContext.from_brain(Broken(), "record")


def test_scenario_import_needs_no_world_extras():
    repo = str(Path(__file__).resolve().parents[1])
    code = "import sys; sys.path.insert(0, sys.argv[1]); from lingshu.world.scenario import ScenarioContext; assert 'numpy' not in sys.modules; assert 'PIL' not in sys.modules"
    result = subprocess.run([sys.executable, "-I", "-S", "-c", code, repo],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_real_brain_mcp_role_lore_and_history_are_separate(tmp_path):
    brain_path = os.environ.get("MDCG_BRAIN_PYTHONPATH")
    if not brain_path:
        pytest.skip("MDCG_BRAIN_PYTHONPATH absent: real MCP brain not available")
    from lingshu.world.brain_store import connect
    agent = connect(root=str(tmp_path / "brain"), pythonpath=brain_path,
        extra_env={"MDCG_LEGACY_ENV_AUTH": "1", "MDCG_LEGACY_ENV_ADMIN": "1", "MDCG_CAN_ADMIN": "1"})
    try:
        cs = json.loads(ConditionSpace("scenario-lab", "fixture-loader", (100, 200), "role-world").to_json())
        for nid, tags, text in [
            ("scenario-alice", ["roleplay", "role:alice", "roleplay:role"], "scenario-probe Alice lighthouse keeper"),
            ("scenario-lore", ["sub-knowledge", "role:alice"], "scenario-probe Blue crystal powers the lamp"),
            ("scenario-bob", ["sub-knowledge", "role:bob"], "scenario-probe Bob lives on a different planet"),
            ("scenario-real", ["external", "vref:test-catalogue"], "scenario-probe An archival catalogue of an oil lamp")]:
            written = agent.store.client.call("mdcg_remember", {"node_id": nid, "content": text,
                "layer": "knowledge", "gated": False, "tags": tags, "condition_space": cs})
            assert written.get("ok"), written
        ctx = ScenarioContext.from_brain(agent.store.client, "scenario-probe", k=16, role_id="alice")
        role = ctx.view("roleplay")
        assert {n["id"] for n in role["facts"]} == {"scenario-alice", "scenario-lore"}
        assert role["payload"]["identity"][0]["id"] == "scenario-alice"
        assert {n["id"] for n in ctx.view("history")["facts"]} == {"scenario-real"}
        assert all(n["conditions"] == cs for n in role["facts"])
    finally:
        agent.store.client.close()


def test_real_brain_world_projection_consumes_existing_state_event_loop(tmp_path):
    brain_path = os.environ.get("MDCG_BRAIN_PYTHONPATH")
    if not brain_path:
        pytest.skip("MDCG_BRAIN_PYTHONPATH absent: real MCP brain not available")
    from lingshu.world.brain_store import connect, ingest_scene_to_brain
    agent = connect(root=str(tmp_path / "brain"), pythonpath=brain_path,
        extra_env={"MDCG_LEGACY_ENV_AUTH": "1", "MDCG_LEGACY_ENV_ADMIN": "1", "MDCG_CAN_ADMIN": "1"})
    try:
        ingest_scene_to_brain(agent, "keeper|person|0,1,5|happy")
        ctx = ScenarioContext.from_brain(agent.store.client, "场景实体", k=16)
        entity, = ctx.view("world")["payload"]["entities"]
        assert entity["name"] == "keeper" and entity["pos"] == [0.0, 1.0, 5.0]
        assert entity["state"] == "happy"
        node, = ctx.view("world")["facts"]
        assert node["attributes"]["state_evidence"]["slot"] == "状态"
        assert node["attributes"]["state_evidence"]["state"] == "active"
        assert ctx.to_world_model().entities["keeper"].state == "happy"
    finally:
        agent.store.client.close()
