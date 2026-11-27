# Copyright 2026 灵枢 (Lingshu) · MIT
"""场景整合运行入口：demo / 本地图只读 / 既有脑端 MCP；无需前端或插件。

运行：python -m lingshu.world.scenario_cli demo --mode all
JSON 写 stdout，错误写 stderr；资格为 DEFER/BLINDSPOT 的成功消费仍返回 0。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sqlite3
import sys

from .scenario import CONDITION_FIELDS, MODES, ScenarioContext, _node

# LayeredStore SQLite 只读反序列化契约（与 core.MemoryLayer / EdgeType / ConditionSpace 同口径；
# world 层不反向导入 core，issue #434）。
_MEMORY_LAYERS = frozenset({"anchor", "structure", "knowledge", "context", "self"})
_EDGE_TYPES = frozenset({
    "causal", "sequential", "correlational", "cyclic", "hierarchical",
    "spatial_adjacent", "spatial_contains", "spatial_connected",
    "similar", "opposite", "applies_to",
})


def _parse_condition_space(raw):
    """兼容 core.ConditionSpace.from_json 的纯标准库反序列化。"""
    if not isinstance(raw, str):
        raise ValueError("condition_space must be a JSON string")
    parsed = json.loads(raw)
    cs = {k: v for k, v in parsed.items() if k in CONDITION_FIELDS} if isinstance(parsed, dict) else {}
    cs.setdefault("observation_position", "外部观测位")
    cs.setdefault("observation_tool", "文本语义分析")
    cs.setdefault("existence_constraint", "（未声明）")
    if "time_window" not in cs:
        cs["time_window"] = (0.0, 0.0)
    elif not isinstance(cs["time_window"], (tuple, list)):
        cs["time_window"] = (cs["time_window"], cs["time_window"])
    tw = tuple(cs["time_window"])
    if not all(v is None for v in tw):
        if (len(tw) != 2
                or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in tw)):
            tw = (0.0, 0.0)
    cs["time_window"] = tw
    return cs


def _parse_json_field(raw, name, default=None):
    if raw is None:
        if default is not None:
            return default()
        raise ValueError(f"{name} must be a JSON string")
    if not isinstance(raw, str):
        raise ValueError(f"{name} must be a JSON string")
    return json.loads(raw)


@dataclass
class _ReadOnlyNode:
    """只读 SQLite 节点轻量载体（避免反向导入 core.STNode）。"""
    id: str
    content: str
    tags: list
    layer: str
    condition_space: dict
    spatial_coordinates: dict
    state_attributes: dict


@dataclass
class _ReadOnlyEdge:
    """只读 SQLite 边轻量载体（避免反向导入 core.STEdge）。"""
    id: str
    source_id: str
    target_id: str
    relation_type: str
    condition_space: dict
    verified: bool
    source_evidence: str


class _ReadOnlyStore:
    """消费已有 LayeredStore schema，不调用会建表/迁移的构造器，也不反向依赖 core。"""

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
        cur = self.conn.execute("SELECT * FROM nodes WHERE id=?", (node_id,))
        row = cur.fetchone()
        if row is None:
            return None
        cols = [desc[0] for desc in cur.description]
        rec = dict(zip(cols, row))
        required = {"id", "content", "spatial_coordinates", "condition_space", "layer", "tags"}
        if not required.issubset(rec):
            raise ValueError("incompatible nodes table schema")
        layer = rec["layer"]
        if layer not in _MEMORY_LAYERS:
            raise ValueError(f"invalid memory layer: {layer!r}")
        raw_state = rec.get("state_attributes")
        state_attributes = _parse_json_field(raw_state, "state_attributes", dict) if raw_state else {}
        return _ReadOnlyNode(
            id=rec["id"],
            content=rec["content"],
            tags=_parse_json_field(rec["tags"], "tags"),
            layer=layer,
            condition_space=_parse_condition_space(rec["condition_space"]),
            spatial_coordinates=_parse_json_field(rec["spatial_coordinates"], "spatial_coordinates"),
            state_attributes=state_attributes,
        )

    def get_outgoing_edges(self, node_id):
        cur = self.conn.execute("SELECT * FROM edges WHERE source_id=?", (node_id,))
        rows = cur.fetchall()
        if not rows:
            return []
        cols = [desc[0] for desc in cur.description]
        required = {"id", "source_id", "target_id", "relation_type", "condition_space", "verified"}
        if not required.issubset(cols):
            raise ValueError("incompatible edges table schema")
        edges = []
        for row in rows:
            rec = dict(zip(cols, row))
            rel_type = rec["relation_type"]
            if rel_type not in _EDGE_TYPES:
                raise ValueError(f"invalid edge relation_type: {rel_type!r}")
            edges.append(_ReadOnlyEdge(
                id=rec["id"],
                source_id=rec["source_id"],
                target_id=rec["target_id"],
                relation_type=rel_type,
                condition_space=_parse_condition_space(rec["condition_space"]),
                verified=bool(rec["verified"]),
                source_evidence=rec["source_evidence"] if "source_evidence" in rec else "extracted",
            ))
        return edges

    def close(self):
        self.conn.close()


def _demo_context(**scope):
    """只在内存构造合成灯塔图；八条因果边、其他角色和其他会话用于验收。"""
    nodes, edges = [], []

    def add(nid, text, tags, when=100, pos=None, state=None):
        cs = {
            "observation_position": "合成灯塔观察位",
            "observation_tool": "离线验收",
            "time_window": (when, when + 20),
            "existence_constraint": "合成灯塔场景",
        }
        nodes.append(_node({
            "id": nid,
            "content": text,
            "frontmatter": {
                "tags": ["fixture", *tags],
                "layer": "knowledge",
                "condition_space": cs,
                "spatial": {"coords3d": pos or {}},
                "state_attributes": {"state": state} if state else {},
            },
        }))

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
    edge_cs = {
        "observation_position": "合成灯塔观察位",
        "observation_tool": "离线验收",
        "time_window": (100, 300),
        "existence_constraint": "合成灯塔场景",
    }
    for i in range(8):
        edges.append({
            "id": f"path-{i}",
            "source_id": chain[i],
            "target_id": chain[i + 1],
            "relation_type": "causal",
            "conditions": edge_cs,
            "verified": False,
            "source_evidence": "inferred",
        })
    return ScenarioContext(nodes, edges, origin="selected", **scope)


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
    # #2564（安全 P1）：保留键检查此前**大小写敏感**——`{"MDCG_root": ...}` 放行，
    # 而 Windows 环境块名不区分大小写，`_brain_env` 先写 `MDCG_ROOT` 再写
    # `MDCG_root` ⇒ 后者生效，脑端库根被重定向到攻击者自备的库（世界先验全部由
    # 攻击者决定），`MDCG_STG_state="0"` 还能关掉状态读取。docs/scenario_usage.md
    # 明确承诺这两个键「不能经身份文件覆盖」。改为大小写不敏感比较。
    if {"MDCG_ROOT", "MDCG_STG_STATE"} & {key.upper() for key in env}:
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
    def _is_unc_or_device(raw_path: str) -> bool:
        return len(raw_path) >= 2 and raw_path[0] in ("/", "\\") and raw_path[1] in ("/", "\\")

    try:
        if args.source == "brain" and _is_unc_or_device(str(args.root)):
            raise ValueError("brain root must not use a UNC/device path")
        if args.render and _is_unc_or_device(str(args.render)):
            raise ValueError("render output must not use a UNC/device path")
        render = Path(args.render).resolve() if args.render else None
        if render:
            if render.suffix.lower() != ".png":
                raise ValueError("render output must be a .png file")
            if args.source == "local" and (render == Path(args.db).resolve() or
                    (render.exists() and Path(args.db).is_file() and render.samefile(args.db))):
                raise ValueError("render output cannot overwrite the source database")
            if args.source == "brain":
                # #2565：`Path("\\?\C:\...").resolve()` **保留** `\?\` 前缀，
                # 而 `Path(root).resolve()` 不带 ⇒ `is_relative_to` 恒 False ⇒ 围栏放行，
                # 但 `os.path.realpath(verbatim)` 仍保留前缀、实际写入落在脑根内（实测
                # plain=True / verbatim=False，两者指向同一文件）。同仓
                # `_screen_expected_allowed` 已有 `\?\` 拒绝的现成写法。
                if render.is_relative_to(Path(args.root).resolve()):
                    raise ValueError(
                        "render output must be outside the brain root")
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
