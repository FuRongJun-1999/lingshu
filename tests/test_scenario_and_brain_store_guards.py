# -*- coding: utf-8 -*-
"""Regression & isolated-sandbox mutation guards for scenario, scenario_cli, and brain_store."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO = Path(__file__).resolve().parent.parent


def _node_dict(nid: str):
    return {
        "id": nid, "content": nid, "tags": ["user"], "layer": "knowledge",
        "conditions": {"observation_position": "p", "observation_tool": "t",
                       "time_window": [0, 1], "existence_constraint": "e"},
        "provenance": {"source": "user", "verify_status": "verified",
                       "references": [], "session_ids": []},
        "qualification": "ACCEPT", "spatial": {}, "attributes": {},
        "record_role": None, "body_window": {},
    }


def test_scenario_from_store_and_from_brain_sanitize_edge_nonfinite_conditions():
    from lingshu.core.core import ConditionSpace
    from lingshu.world.scenario import ScenarioContext

    cs_inf = ConditionSpace("pos", "tool", (0.0, float("inf")), "c")
    nodes = {
        "n1": SimpleNamespace(id="n1", content="A", tags=["user"], layer="knowledge",
                              condition_space=cs_inf, spatial_coordinates={}, state_attributes={}),
        "n2": SimpleNamespace(id="n2", content="B", tags=["user"], layer="knowledge",
                              condition_space=cs_inf, spatial_coordinates={}, state_attributes={}),
    }
    edges = {
        "n1": [SimpleNamespace(id="e1", source_id="n1", target_id="n2", relation_type="causal",
                               condition_space=cs_inf, verified=True, source_evidence="extracted")],
        "n2": [],
    }
    store = SimpleNamespace(get_node=lambda nid: nodes.get(nid),
                            get_outgoing_edges=lambda nid: edges.get(nid, []))
    ctx = ScenarioContext.from_store(store, ["n1", "n2"])
    v = ctx.view("causal", start_id="n1", end_id="n2")
    assert len(v["payload"]["paths"]) == 1
    assert v["payload"]["paths"][0]["edges"][0]["conditions"]["time_window"] == [0.0, None]

    # from_brain also sanitizes edge condition_space and honors state_response["truncated"]
    class FakeClient:
        def call(self, tool, args):
            if tool == "cg":
                return {
                    "ok": True,
                    "results": [{
                        "state": "ACCEPT",
                        "node": {
                            "id": "n1", "content": "A",
                            "frontmatter": {
                                "tags": ["spatial", "user"], "layer": "knowledge",
                                "spatial": {"coords3d": {"x": 0.0, "y": 1.0, "z": 2.0}},
                                "edges": [{"target": "n1", "relation_type": "causal",
                                           "condition_space": {"time_window": [0.0, float("inf")]},
                                           "verified": True}],
                            },
                        },
                    }],
                }
            return {"items": [], "truncated": True}

    ctx_b = ScenarioContext.from_brain(FakeClient(), "q", k=4)
    assert "state_projection_limit_reached" in ctx_b.view("world")["gaps"]


def test_scenario_max_nodes_validation_before_comparison():
    from lingshu.world.scenario import ScenarioContext

    store = SimpleNamespace(get_node=lambda _nid: None, get_outgoing_edges=lambda _nid: [])
    for bad in (True, False, 0, -5, "10"):
        with pytest.raises(ValueError, match="max_nodes"):
            ScenarioContext.from_store(store, ["a"], max_nodes=bad)
        with pytest.raises(ValueError, match="max_nodes"):
            ScenarioContext.from_brain(SimpleNamespace(), "q", k=1, max_nodes=bad)


def test_scenario_causal_preserves_simple_path_and_zero_edge_contract():
    from lingshu.world.scenario import ScenarioContext

    # Zero-edge query a->a stays empty and cyclic=True when self-loop exists (#1319)
    ctx_cyc = ScenarioContext(
        nodes=[_node_dict("a"), _node_dict("b")],
        edges=[
            {"id": "e1", "source_id": "a", "target_id": "a", "relation_type": "causal",
             "conditions": {}, "verified": True, "source_evidence": "extracted"},
            {"id": "e2", "source_id": "a", "target_id": "b", "relation_type": "causal",
             "conditions": {}, "verified": True, "source_evidence": "extracted"},
        ],
    )
    rc = ctx_cyc.view("causal", start_id="a", end_id="a")
    assert rc["payload"]["paths"] == []
    assert rc["payload"]["cyclic"] is True
    rab = ctx_cyc.view("causal", start_id="a", end_id="b")
    assert [p["node_ids"] for p in rab["payload"]["paths"]] == [["a", "b"]]


def test_scenario_cli_rejects_unc_brain_root(tmp_path, capsys):
    from lingshu.world import scenario_cli

    out_png = tmp_path / "out.png"
    rc = scenario_cli.main([
        "brain", "--root", r"\\?\C:\fake_brain_root",
        "--brain-path", str(tmp_path), "--query", "q",
        "--mode", "world", "--render", str(out_png),
    ])
    assert rc == 1
    captured = capsys.readouterr()
    assert "brain root must not use a UNC/device path" in (captured.err + captured.out)


def test_brain_store_rejects_bool_and_nonfinite_inputs():
    from lingshu.world.brain_store import BrainEngine, BrainStore, MCPClient

    for bad_t in (True, False, float("nan"), float("inf"), -1.0, "30"):
        with pytest.raises(ValueError, match="request_timeout"):
            MCPClient(pythonpath="/tmp", request_timeout=bad_t)

    class StubClient:
        def __init__(self):
            self.calls = []

        def call(self, tool, args):
            self.calls.append((tool, args))
            if tool == "cg" and args.get("op") == "read":
                return {
                    "results": [{
                        "state": "ACCEPT",
                        "node": {
                            "id": "n_bad", "content": "c",
                            "frontmatter": {
                                "tags": ["spatial"],
                                "spatial": {"coords3d": {"x": float("nan"), "y": 1.0, "z": 2.0}},
                            },
                        },
                    }],
                    "meta": {"pre_cap": 1},
                }
            if tool == "stg":
                return {"items": [], "truncated": False}
            return {"ok": True, "id": "ok1"}

    client = StubClient()
    store = BrainStore(client)
    for bad_lim in (True, False, -1, 1.5):
        with pytest.raises(ValueError, match="limit"):
            store.get_nodes_by_tag("spatial", limit=bad_lim)

    # limit=0 still maps to want=1 per docstring, and non-finite coords3d drops coords instead of leaking NaN
    nodes = store.get_nodes_by_tag("spatial", limit=0)
    assert len(nodes) == 1
    assert nodes[0].spatial_coordinates is None

    engine = BrainEngine(store)
    for bad_imp in (True, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="importance"):
            engine.add_perception("x", importance=bad_imp)
    for bad_coords in ({"x": float("nan")}, {"x": float("inf")}, {"x": True}):
        with pytest.raises(ValueError, match="spatial_coordinates"):
            engine.add_perception("x", spatial_coordinates=bad_coords)


_MUTATION_ARMS = {
    "M1_edge_inf_sanitize": (
        "lingshu/world/scenario.py",
        '"conditions": _finite_or_none(_condition_dict(e.condition_space)),',
        '"conditions": _condition_dict(e.condition_space),',
        "test_scenario_from_store_and_from_brain_sanitize_edge_nonfinite_conditions",
    ),
    "M2_state_truncated_gap": (
        "lingshu/world/scenario.py",
        'if state_response.get("truncated") or len(state_response["items"]) >= 500:',
        'if len(state_response["items"]) >= 500:',
        "test_scenario_from_store_and_from_brain_sanitize_edge_nonfinite_conditions",
    ),
    "M3_unc_brain_root": (
        "lingshu/world/scenario_cli.py",
        'if _raw_root.startswith("//") or _raw_root.startswith("\\\\\\\\"):',
        "if False:",
        "test_scenario_cli_rejects_unc_brain_root",
    ),
    "M4_brain_timeout_bool": (
        "lingshu/world/brain_store.py",
        "                isinstance(request_timeout, bool)\n"
        "                or not isinstance(request_timeout, (int, float))\n"
        "                or not math.isfinite(request_timeout)",
        "                not math.isfinite(request_timeout)",
        "test_brain_store_rejects_bool_and_nonfinite_inputs",
    ),
}


def _run_mutate(want: str) -> int:
    arms = list(_MUTATION_ARMS) if want == "ALL" else [want]
    failed = 0
    for arm in arms:
        if arm not in _MUTATION_ARMS:
            print(f"[ANCHOR-MISS] unknown arm {arm}")
            return 2
        rel, old, new, test_name = _MUTATION_ARMS[arm]
        src_path = _REPO / rel
        orig_bytes = src_path.read_bytes()
        tmp = Path(tempfile.mkdtemp(prefix=f"mut_l2_{arm}_"))
        try:
            shutil.copytree(_REPO / "lingshu", tmp / "lingshu",
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            (tmp / "tests").mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(__file__), tmp / "tests" / Path(__file__).name)
            target = tmp / rel
            text = target.read_bytes().decode("utf-8").replace("\r\n", "\n")
            if text.count(old) != 1:
                print(f"[ANCHOR-MISS] {arm}: count={text.count(old)} in {rel}")
                return 2
            target.write_bytes(text.replace(old, new, 1).encode("utf-8"))
            env = os.environ.copy()
            env["PYTHONPATH"] = str(tmp)
            r = subprocess.run(
                [sys.executable, "-X", "utf8", "-m", "pytest", "-q",
                 str(tmp / "tests" / Path(__file__).name), "-k", test_name],
                cwd=str(tmp), env=env, capture_output=True, text=True,
                encoding="utf-8", errors="replace",
            )
            untouched = src_path.read_bytes() == orig_bytes
            if r.returncode != 0 and untouched:
                print(f"[MUTATE-OK] {arm} killed (rc={r.returncode}, repo_files_untouched={untouched})")
            else:
                print(f"[MUTATE-FAIL] {arm} survived (rc={r.returncode}, repo_files_untouched={untouched})")
                failed += 1
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    want = None
    for a in sys.argv[1:]:
        if a == "--mutate":
            want = "ALL"
        elif a.startswith("--mutate="):
            want = a.split("=", 1)[1]
    if want is not None:
        raise SystemExit(_run_mutate(want))
    raise SystemExit(pytest.main(["-q", __file__]))
