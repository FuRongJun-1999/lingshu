# Copyright 2026 灵枢 (Lingshu) · MIT
"""场景整合运行入口：demo / 本地图只读 / 既有脑端 MCP；无需前端或插件。

运行：python -m lingshu.world.scenario_cli demo --mode all
JSON 写 stdout，错误写 stderr；资格为 DEFER/BLINDSPOT 的成功消费仍返回 0。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sqlite3
import sys

from .scenario import MODES, ScenarioContext


class _ReadOnlyStore:
    """消费已有 LayeredStore schema，不调用会建表/迁移的构造器。"""

    def __init__(self, path):
        path = Path(path).resolve()
        if not path.is_file():
            raise ValueError("local database must already exist")
        self.conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
        try:
            self.conn.execute("PRAGMA query_only=ON")
            self.conn.execute("BEGIN")  # 同一选取集的节点/边来自同一 SQLite 读快照。
        except BaseException:
            self.conn.close()
            raise

    def get_node(self, node_id):
        from ..core.core import STNode
        row = self.conn.execute("SELECT * FROM nodes WHERE id=?", (node_id,)).fetchone()
        return STNode.from_row(row) if row is not None else None

    def get_outgoing_edges(self, node_id):
        from ..core.core import STEdge
        rows = self.conn.execute("SELECT * FROM edges WHERE source_id=?", (node_id,)).fetchall()
        return [STEdge.from_row(row) for row in rows]

    def close(self):
        self.conn.close()


def _demo_context(**scope):
    """只在内存构造合成灯塔图；八条因果边、其他角色和其他会话用于验收。"""
    from ..core.core import ConditionSpace, EdgeType, LayeredStore, STNode, STEdge

    store = LayeredStore(":memory:")
    ids = []
    try:
        def add(nid, text, tags, when=100, pos=None, state=None):
            cs = ConditionSpace("合成灯塔观察位", "离线验收", (when, when + 20), "合成灯塔场景")
            store.add_node(STNode(nid, text, "text", pos or {}, when, cs,
                tags=["fixture", *tags], state_attributes={"state": state} if state else {}))
            ids.append(nid)

        add("keeper", "守灯人 Alice", ["roleplay", "role:alice", "roleplay:role"])
        add("rule", "本角色世界的灯塔以晶石供能", ["role:alice", "sub-knowledge"])
        add("memory", "前任守灯人留下操作手册", ["role:alice", "roleplay:memory"])
        add("anchor", "守护灯塔", ["role:alice", "roleplay:anchors"])
        add("value", "如实报告未知状态", ["role:alice", "roleplay:values"])
        add("turn", "准备启动晶石", ["role:alice", "session:demo", "turn:user"])
        add("other-role", "Bob 的星球有两颗太阳", ["role:bob", "sub-knowledge"])
        add("other-session", "另一个会话的转录", ["role:alice", "session:other", "turn:user"])
        add("catalogue", "油灯目录（合成待核史料，无真实史实声明）", ["vref:synthetic-catalogue"])
        chain = ["action", *(f"stage-{i}" for i in range(1, 8)), "result"]
        for i, nid in enumerate(chain):
            add(nid, "启动晶石" if i == 0 else "灯塔亮起" if i == 8 else f"供能步骤 {i}",
                ["role:alice", "event", *(["ent:灯塔", "cat:house"] if i == 8 else [])],
                when=200 if i == 0 else 100 + i * 10,
                pos={"x": 0, "y": 1, "z": 18} if i == 8 else None,
                state="on" if i == 8 else None)
        for i in range(8):
            store.add_edge(STEdge(f"path-{i}", chain[i], chain[i + 1], EdgeType.CAUSAL,
                ConditionSpace("合成灯塔观察位", "离线验收", (100, 300), "合成灯塔场景"),
                source_evidence="inferred"))
        return ScenarioContext.from_store(store, ids, **scope)
    finally:
        store.conn.close()


def _brain_env_file(path):
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        text = stream.read(65537)
    if len(text) > 65536:
        raise ValueError("brain env file exceeds 64 KiB characters")
    try:
        env = json.loads(text)
    except ValueError:
        raise ValueError("brain env file must contain a JSON object") from None
    if not isinstance(env, dict) or any(not key.startswith("MDCG_") or
            (value is not None and not isinstance(value, str)) for key, value in env.items()):
        raise ValueError("brain env file must map MDCG_ keys to strings or null")
    if {"MDCG_ROOT", "MDCG_STG_STATE"}.intersection(env):
        raise ValueError("brain env file contains reserved root/state keys")
    return env  # 不打印身份值，不隐式继承宿主 MDCG_*。


def _positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _positive_seconds(text):
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be positive finite seconds")
    return value


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="source", required=True)
    for source, help_text in (("demo", "内存合成图，无需库或脑"),
                              ("local", "只读已有 LayeredStore SQLite 图"),
                              ("brain", "经既有 MCP 读取显式指定的脑库")):
        child = commands.add_parser(source, help=help_text)
        child.add_argument("--mode", choices=("all", *MODES), default="all")
        child.add_argument("--role-id", default="alice" if source == "demo" else None)
        child.add_argument("--session-id", default="demo" if source == "demo" else None)
        child.add_argument("--start-id", default="action" if source == "demo" else None)
        child.add_argument("--end-id", default="result" if source == "demo" else None)
        for name, default in (("nodes", 256), ("edges", 1024), ("depth", 16),
                              ("paths", 32), ("steps", 4096)):
            child.add_argument("--max-" + name, type=_positive_int, default=default)
        child.add_argument("--render", metavar="OUTPUT.png", help="复用 WorldModel 渲染；需要 .[full]")
        if source == "local":
            child.add_argument("--db", required=True)
            child.add_argument("--node-id", required=True, action="append")
        elif source == "brain":
            child.add_argument("--root", required=True, help="已存在的明确目标库目录")
            child.add_argument("--brain-path", required=True, help="包含 md_cg 包的目录")
            child.add_argument("--brain-python", help="脑端解释器；默认沿用既有适配器选择")
            child.add_argument("--brain-env-file", help="库外身份配置 JSON；不打印值")
            child.add_argument("--query", required=True)
            child.add_argument("--k", type=_positive_int, default=64)
            child.add_argument("--timeout", type=_positive_seconds, default=30.0)
    return parser


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = _parser()
    args = parser.parse_args(argv)
    modes = MODES if args.mode == "all" else (args.mode,)
    if "causal" in modes and (not args.start_id or not args.end_id):
        parser.error("causal/all requires --start-id and --end-id; endpoints are never guessed")
    for value in (args.role_id, args.session_id):
        if value is not None and not value.strip():
            parser.error("role-id/session-id must be nonempty")
    if args.render and "world" not in modes:
        parser.error("--render requires world/all mode")
    if args.source == "brain" and args.k > args.max_nodes:
        parser.error("--k must not exceed --max-nodes")
    scope = dict(role_id=args.role_id, session_id=args.session_id,
                 max_nodes=args.max_nodes, max_edges=args.max_edges)
    try:
        render = Path(args.render).resolve() if args.render else None
        if render:
            if render.suffix.lower() != ".png":
                raise ValueError("render output must be a .png file")
            if args.source == "local" and (render == Path(args.db).resolve() or
                    (render.exists() and Path(args.db).is_file() and render.samefile(args.db))):
                raise ValueError("render output cannot overwrite the source database")
            if args.source == "brain" and render.is_relative_to(Path(args.root).resolve()):
                raise ValueError("render output must be outside the brain root")
        if args.source == "demo":
            context = _demo_context(**scope)
        elif args.source == "local":
            store = _ReadOnlyStore(args.db)
            try:
                context = ScenarioContext.from_store(store, args.node_id, **scope)
            finally:
                store.close()
        else:
            env = _brain_env_file(args.brain_env_file)
            if not Path(args.root).is_dir():
                raise ValueError("brain root must already exist; no default/new root is selected")
            if not (Path(args.brain_path) / "md_cg" / "mcp_server.py").is_file():
                raise ValueError("brain-path must contain md_cg/mcp_server.py")
            from .brain_store import MCPClient
            client = MCPClient(root=str(Path(args.root).resolve()),
                pythonpath=str(Path(args.brain_path).resolve()), python=args.brain_python,
                extra_env=env, request_timeout=args.timeout)
            try:
                context = ScenarioContext.from_brain(client, args.query, k=args.k, **scope)
            finally:
                client.close()
        views = {mode: context.view(mode, start_id=args.start_id, end_id=args.end_id,
            max_depth=args.max_depth, max_paths=args.max_paths, max_steps=args.max_steps) for mode in modes}
        result = {"schema_version": 1, "fixture": args.source == "demo", "views": views}
        if render:
            render.parent.mkdir(parents=True, exist_ok=True)
            context.to_world_model().render().save(render, format="PNG")
            result["render"] = {"path": str(render), "evidence": False,
                "boundary": "当前候选实体的呈现，不能证明材料或世界为真"}
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, sqlite3.Error, ImportError) as exc:
        print("scenario error: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
