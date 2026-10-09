"""交付入口验收：安装形态命令、只读 SQLite、真实脑端与错误出口。"""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from lingshu.core.core import ConditionSpace, EdgeType, LayeredStore, STNode, STEdge
from lingshu.world.scenario import MODES

REPO = Path(__file__).resolve().parents[1]


def cli(*args, isolated=False, env=None):
    child_env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONUTF8="1")
    child_env.update(env or {})
    if isolated:
        code = ("import sys; sys.path.insert(0, sys.argv[1]); "
                "from lingshu.world.scenario_cli import main; "
                "rc=main(sys.argv[2:]); "
                "assert 'numpy' not in sys.modules and 'PIL' not in sys.modules; "
                "raise SystemExit(rc)")
        cmd = [sys.executable, "-I", "-S", "-c", code, str(REPO)]
    else:
        cmd = [sys.executable, "-X", "utf8", "-m", "lingshu.world.scenario_cli"]
    return subprocess.run(cmd + list(args), env=child_env, cwd=REPO.parent,
                          capture_output=True, text=True, encoding="utf-8", timeout=60)


@pytest.mark.parametrize("mode", ["all", *MODES])
def test_demo_runs_all_five_tasks_without_extras(mode):
    result = cli("demo", "--mode", mode, isolated=True)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["schema_version"] == 1 and data["fixture"] is True
    assert set(data["views"]) == (set(MODES) if mode == "all" else {mode})
    for task, view in data["views"].items():
        assert view["mode"] == task and view["state"] == "DEFER"
        assert all(n["provenance"]["source"] == "fixture" for n in view["facts"])
    if mode == "all":
        views = data["views"]
        assert views["roleplay"]["payload"]["identity"][0]["id"] == "keeper"
        assert "other-role" not in {n["id"] for n in views["roleplay"]["facts"]}
        assert "other-session" not in {n["id"] for n in views["roleplay"]["facts"]}
        assert {n["id"] for n in views["history"]["facts"]} == {"catalogue"}
        assert views["world"]["payload"]["entities"][0]["state"] == "on"
        path, = views["causal"]["payload"]["paths"]
        assert len(path["edges"]) == 8 and path["hypothesis"] is True
        outline = [n["node_id"] for n in views["story"]["payload"]["outline"]]
        assert outline.index("action") < outline.index("result")


def test_local_input_is_readonly_and_does_not_create_or_migrate_tables(tmp_path):
    db = tmp_path / "existing # graph.db"
    store = LayeredStore(str(db))
    cs = ConditionSpace("lab", "fixture", (100, 200), "test")
    for nid in ("cause'--", "effect"):
        store.add_node(STNode(nid, nid, "text", {}, 100, cs, tags=["fixture"]))
    store.add_edge(STEdge("link", "cause'--", "effect", EdgeType.CAUSAL, cs))
    before_dump = list(store.conn.iterdump())
    store.conn.close()
    before_hash = hashlib.sha256(db.read_bytes()).hexdigest()
    result = cli("local", "--db", str(db), "--node-id", "cause'--", "--node-id", "effect",
                 "--mode", "causal", "--start-id", "cause'--", "--end-id", "effect")
    assert result.returncode == 0, result.stderr
    view = json.loads(result.stdout)["views"]["causal"]
    assert view["payload"]["paths"][0]["node_ids"] == ["cause'--", "effect"]
    assert view["state"] == "DEFER"
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before_hash
    with sqlite3.connect(db) as conn:
        assert list(conn.iterdump()) == before_dump
        assert conn.execute("SELECT SUM(access_count) FROM nodes").fetchone()[0] == 0


@pytest.mark.parametrize("args", [
    ["demo", "--max-steps", "0"],
    ["demo", "--role-id", " "],
    ["demo", "--mode", "history", "--render", "example.png"],
    ["local", "--db", "does-not-exist.db", "--node-id", "a", "--mode", "causal"],
])
def test_invalid_options_fail_without_success_json(args):
    result = cli(*args)
    assert result.returncode == 2 and result.stdout == ""
    assert "error" in result.stderr.lower()


def test_missing_database_is_not_created_and_wrong_schema_is_not_migrated(tmp_path):
    missing = tmp_path / "missing.db"
    result = cli("local", "--db", str(missing), "--node-id", "a", "--mode", "world")
    assert result.returncode == 1 and not missing.exists() and not result.stdout
    wrong = tmp_path / "wrong.db"
    with sqlite3.connect(wrong) as conn:
        conn.execute("CREATE TABLE unrelated (id TEXT)")
    before = wrong.read_bytes()
    result = cli("local", "--db", str(wrong), "--node-id", "a", "--mode", "world")
    assert result.returncode == 1 and not result.stdout
    assert wrong.read_bytes() == before


def test_demo_render_reuses_existing_world_model(tmp_path):
    from PIL import Image
    output = tmp_path / "scene.png"
    result = cli("demo", "--mode", "world", "--render", str(output))
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["render"]["evidence"] is False
    assert data["views"]["world"]["state"] == "DEFER"
    with Image.open(output) as img:
        assert img.format == "PNG" and img.size == (500, 500)
        assert len(img.getcolors(250001)) > 1


def test_render_cannot_overwrite_a_source_or_write_into_brain_root(tmp_path):
    db = tmp_path / "source.png"
    store = LayeredStore(str(db))
    store.conn.close()
    before = db.read_bytes()
    result = cli("local", "--db", str(db), "--node-id", "a", "--mode", "world", "--render", str(db))
    assert result.returncode == 1 and "overwrite" in result.stderr and not result.stdout
    assert db.read_bytes() == before
    output = tmp_path / "brain-render.png"
    result = cli("brain", "--root", str(tmp_path), "--brain-path", str(tmp_path),
        "--query", "test", "--mode", "world", "--render", str(output))
    assert result.returncode == 1 and "outside" in result.stderr and not result.stdout
    assert not output.exists()


def test_brain_config_cannot_override_explicit_root_or_print_secrets(tmp_path):
    config = tmp_path / "identity.json"
    config.write_text(json.dumps({"MDCG_ROOT": "other", "MDCG_TOKEN": "synthetic-secret"}),
                      encoding="utf-8")
    result = cli("brain", "--root", str(tmp_path), "--brain-path", str(tmp_path),
                 "--brain-env-file", str(config), "--query", "test", "--mode", "history")
    assert result.returncode == 1 and not result.stdout
    assert "reserved" in result.stderr.lower() and "synthetic-secret" not in result.stderr


def test_real_brain_cli_uses_selected_root_and_shares_one_context(tmp_path):
    brain_path = os.environ.get("MDCG_BRAIN_PYTHONPATH")
    if not brain_path:
        pytest.skip("MDCG_BRAIN_PYTHONPATH absent: real MCP brain not available")
    from lingshu.world.brain_store import connect, ingest_scene_to_brain
    target_root = tmp_path / "brain"
    agent = connect(root=str(target_root), pythonpath=brain_path, extra_env={
        "MDCG_LEGACY_ENV_AUTH": "1", "MDCG_LEGACY_ENV_ADMIN": "1", "MDCG_CAN_ADMIN": "1"})
    try:
        cs = json.loads(ConditionSpace("lab", "fixture", (100, 200), "scenario").to_json())
        for nid, tags in [("cli-role", ["roleplay", "role:alice", "roleplay:role", "fixture"]),
                          ("cli-real", ["fixture", "vref:test-catalogue"])]:
            result = agent.store.client.call("mdcg_remember", {"node_id": nid,
                "content": "场景实体 cli-probe", "gated": False, "layer": "knowledge",
                "tags": tags, "condition_space": cs})
            assert result.get("ok"), result
        ingest_scene_to_brain(agent, "keeper|person|0,1,5|happy")
    finally:
        agent.store.client.close()
    config = tmp_path / "identity.json"
    config.write_text(json.dumps({"MDCG_LEGACY_ENV_AUTH": "1"}), encoding="utf-8")
    wrong_root = tmp_path / "inherited-wrong-root"
    result = cli("brain", "--root", str(target_root), "--brain-path", brain_path,
        "--brain-env-file", str(config), "--query", "场景实体", "--mode", "all", "--k", "16",
        "--role-id", "alice", "--start-id", "cli-role", "--end-id", "cli-real",
        env={"MDCG_ROOT": str(wrong_root)})
    assert result.returncode == 0, result.stderr
    views = json.loads(result.stdout)["views"]
    assert set(views) == set(MODES) and not wrong_root.exists()
    assert {n["id"] for n in views["roleplay"]["facts"]} == {"cli-role"}
    assert "cli-role" not in {n["id"] for n in views["history"]["facts"]}
    assert views["world"]["payload"]["entities"][0]["state"] == "happy"
    assert views["causal"]["payload"]["paths"] == []
