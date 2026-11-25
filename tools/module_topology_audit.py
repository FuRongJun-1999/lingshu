#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""module_topology_audit · 全仓模块主链连通性 AST 实测盘点工具与漂移门禁（#172）

背景（issue #172 后续第 ② 项）：
    盘点 `lingshu/` 下全部 `.py` 模块的真实导入连通性，以纯标准库 AST 静态扫描
    （含顶层静态 `import`/`from`、函数/方法体惰性导入以及 `importlib.import_module` /
    `__import__("lingshu....")`），将全部非 `__init__` 模块客观划分为三类：

      1. `main_chain`（核心主入口可达）：从主入口集合
         `{lingshu/core/core.py, lingshu/world/world_model.py, lingshu/world/scene_model.py}`
         沿有向 AST 导入边可达（含 `core.py` 内惰性 `from ..world...` 边）；
      2. `subgraph_only`（包内子图互调）：不在 `main_chain` 内，但在 `lingshu/` 内
         入度 > 0（被同包或跨包其它模块导入，含 `__init__.py` 如
         `lingshu/world/__init__.py::register_world_process` 等静态解析包内边）；
      3. `standalone_leaf`（独立原型叶模块）：`lingshu/` 包内入度 == 0（包内无人导入，
         仅由 `tests/`、`bench/`、`tools/` 或直接 CLI 消费）。

用法：
    python -X utf8 tools/module_topology_audit.py --json
    python -X utf8 tools/module_topology_audit.py --markdown [--with-test-refs]
    python -X utf8 tools/module_topology_audit.py --write
    python -X utf8 tools/module_topology_audit.py --check

说明：
    默认 `--write` 生成的台账与 `--check` 门禁仅锁定 `lingshu/` 包内拓扑结构
    （`total_py` / `init_py` / `non_init_modules` / `by_status` / `by_layer` 与
    逐模块 `{path, layer, status, reach_via, in_degree, importers, out_degree}`），
    不把随单测/基准增长而变化的 `tests/`、`bench/` 引用计数纳入阻断面，避免
    并行 PR 新增测试时误触发台账漂移；需实时查看引用计数可使用 `--json` 或
    `--markdown --with-test-refs`。

退出码：
    0  成功 / 台账与当前仓 AST 实算逐字一致
    1  `--check` 检测到台账缺失或与实算结果漂移
    2  用法或路径错误
"""
from __future__ import annotations

import argparse
import ast
import difflib
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

AUDIT_VERSION = 1

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER_REL = "docs/module-connectivity.md"

MAIN_CHAIN_ROOTS: Tuple[str, ...] = (
    "lingshu/core/core.py",
    "lingshu/world/world_model.py",
    "lingshu/world/scene_model.py",
)

STATUS_MAIN_CHAIN = "main_chain"
STATUS_SUBGRAPH_ONLY = "subgraph_only"
STATUS_STANDALONE_LEAF = "standalone_leaf"

STATUS_LABELS: Dict[str, str] = {
    STATUS_MAIN_CHAIN: "核心主入口可达",
    STATUS_SUBGRAPH_ONLY: "包内子图互调",
    STATUS_STANDALONE_LEAF: "独立原型叶模块",
}

LAYER_ORDER: Tuple[str, ...] = ("root", "core", "world", "nn", "gen", "vision")


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", newline="\n")
        except (OSError, ValueError, TypeError):
            pass


def _layer_of(rel_path: str) -> str:
    parts = rel_path.split("/")
    if len(parts) == 2:
        return "root"
    return parts[1]


def _discover_modules(repo_root: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """扫描 `lingshu/**/*.py`，返回 `(path_to_mod, mod_to_path)`。"""
    pkg_dir = repo_root / "lingshu"
    path_to_mod: Dict[str, str] = {}
    mod_to_path: Dict[str, str] = {}
    for p in sorted(pkg_dir.rglob("*.py"), key=lambda q: q.as_posix()):
        if "__pycache__" in p.parts:
            continue
        rel = p.relative_to(repo_root).as_posix()
        noext = p.relative_to(repo_root).with_suffix("").as_posix()
        mod = noext.replace("/", ".")
        if mod.endswith(".__init__"):
            mod = mod[: -len(".__init__")]
        path_to_mod[rel] = mod
        mod_to_path[mod] = rel
    return path_to_mod, mod_to_path


def _resolve_import_from(
    cur_rel: Optional[str],
    level: int,
    module: Optional[str],
    names: Sequence[ast.alias],
    mod_to_path: Dict[str, str],
) -> List[str]:
    if level > 0:
        if not cur_rel:
            return []
        pkg_parts = cur_rel.split("/")[:-1]
        up = level - 1
        if up > len(pkg_parts):
            return []
        base_parts = pkg_parts[: len(pkg_parts) - up]
        mod_parts = module.split(".") if module else []
        base_mod = ".".join(base_parts + mod_parts)
    else:
        base_mod = module or ""

    sub_hits: List[str] = []
    for alias in names:
        sub = f"{base_mod}.{alias.name}" if base_mod else alias.name
        if sub in mod_to_path:
            sub_hits.append(mod_to_path[sub])
    if sub_hits:
        return sub_hits
    if base_mod in mod_to_path:
        return [mod_to_path[base_mod]]
    return []


def _scan_file_imports(
    file_path: Path,
    mod_to_path: Dict[str, str],
    cur_rel: Optional[str] = None,
) -> Set[str]:
    """用 AST 提取单个文件导入的 `lingshu/` 内部模块相对路径集合。"""
    text = file_path.read_text(encoding="utf-8-sig", errors="replace")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(text, filename=file_path.name)

    targets: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                for k in range(len(parts), 0, -1):
                    cand = ".".join(parts[:k])
                    if cand in mod_to_path:
                        targets.add(mod_to_path[cand])
                        break
        elif isinstance(node, ast.ImportFrom):
            level = node.level or 0
            for hit in _resolve_import_from(
                cur_rel, level, node.module, node.names, mod_to_path
            ):
                targets.add(hit)
        elif isinstance(node, ast.Call):
            fn = node.func
            fn_name = (
                fn.id
                if isinstance(fn, ast.Name)
                else (fn.attr if isinstance(fn, ast.Attribute) else None)
            )
            if fn_name in ("import_module", "__import__") and node.args:
                arg0 = node.args[0]
                if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                    val = arg0.value
                    if val in mod_to_path:
                        targets.add(mod_to_path[val])
    return targets


def _bfs(seeds: Sequence[str], edges: Dict[str, Set[str]]) -> Set[str]:
    seen: Set[str] = set()
    queue: List[str] = []
    for s in seeds:
        if s in edges and s not in seen:
            seen.add(s)
            queue.append(s)
    while queue:
        cur = queue.pop(0)
        for nxt in sorted(edges.get(cur, ())):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def _scan_external_dir_refs(
    repo_root: Path,
    subdir: str,
    mod_to_path: Dict[str, str],
    edges: Dict[str, Set[str]],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """统计 `subdir/`（tests 或 bench）对各 `lingshu/` 模块的直接 AST 导入数与传递可达数。"""
    direct: Dict[str, int] = {rel: 0 for rel in mod_to_path.values()}
    transitive: Dict[str, int] = {rel: 0 for rel in mod_to_path.values()}
    target_dir = repo_root / subdir
    if not target_dir.is_dir():
        return direct, transitive

    for py_file in sorted(target_dir.rglob("*.py"), key=lambda q: q.as_posix()):
        if "__pycache__" in py_file.parts:
            continue
        try:
            hits = _scan_file_imports(py_file, mod_to_path, cur_rel=None)
        except SyntaxError:
            continue
        for h in hits:
            direct[h] += 1
        reached = _bfs(sorted(hits), edges)
        for r in reached:
            transitive[r] += 1
    return direct, transitive


def analyze(repo_root: Path, include_external_refs: bool = True) -> Dict:
    """对 `repo_root` 执行全仓模块连通性 AST 实测盘点，返回确定性报告字典。"""
    path_to_mod, mod_to_path = _discover_modules(repo_root)
    edges: Dict[str, Set[str]] = {}
    for rel in sorted(path_to_mod):
        edges[rel] = _scan_file_imports(repo_root / rel, mod_to_path, cur_rel=rel) - {rel}

    indeg: Dict[str, Set[str]] = {rel: set() for rel in path_to_mod}
    for src, dsts in edges.items():
        for dst in dsts:
            indeg[dst].add(src)

    core_reach = _bfs(["lingshu/core/core.py"], edges)
    wm_reach = _bfs(["lingshu/world/world_model.py", "lingshu/world/scene_model.py"], edges)
    main_reach = core_reach | wm_reach

    if include_external_refs:
        test_direct, test_trans = _scan_external_dir_refs(repo_root, "tests", mod_to_path, edges)
        bench_direct, bench_trans = _scan_external_dir_refs(repo_root, "bench", mod_to_path, edges)
    else:
        zero_refs = {rel: 0 for rel in path_to_mod}
        test_direct, test_trans = zero_refs, zero_refs
        bench_direct, bench_trans = zero_refs, zero_refs

    modules: List[Dict] = []
    init_modules: List[Dict] = []
    by_status = {
        STATUS_MAIN_CHAIN: 0,
        STATUS_SUBGRAPH_ONLY: 0,
        STATUS_STANDALONE_LEAF: 0,
    }
    by_layer: Dict[str, Dict[str, int]] = {}

    for rel in sorted(path_to_mod):
        layer = _layer_of(rel)
        is_init = rel.endswith("/__init__.py")
        importers = sorted(indeg[rel])
        imports = sorted(edges[rel])

        if is_init:
            init_modules.append({
                "path": rel,
                "module": path_to_mod[rel],
                "layer": layer,
                "in_degree": len(importers),
                "importers": importers,
                "out_degree": len(imports),
                "imports": imports,
                "test_direct": test_direct[rel],
                "test_transitive": test_trans[rel],
                "bench_direct": bench_direct[rel],
            })
            continue

        if rel in main_reach:
            status = STATUS_MAIN_CHAIN
        elif importers:
            status = STATUS_SUBGRAPH_ONLY
        else:
            status = STATUS_STANDALONE_LEAF

        by_status[status] += 1
        layer_bucket = by_layer.setdefault(
            layer,
            {
                "total": 0,
                STATUS_MAIN_CHAIN: 0,
                STATUS_SUBGRAPH_ONLY: 0,
                STATUS_STANDALONE_LEAF: 0,
            },
        )
        layer_bucket["total"] += 1
        layer_bucket[status] += 1

        reach_via: List[str] = []
        if rel in core_reach:
            reach_via.append("core.py")
        if rel in wm_reach:
            reach_via.append("WorldModel")

        modules.append({
            "path": rel,
            "module": path_to_mod[rel],
            "layer": layer,
            "status": status,
            "status_zh": STATUS_LABELS[status],
            "reach_via": reach_via,
            "in_degree": len(importers),
            "importers": importers,
            "out_degree": len(imports),
            "imports": imports,
            "test_direct": test_direct[rel],
            "test_transitive": test_trans[rel],
            "bench_direct": bench_direct[rel],
            "bench_transitive": bench_trans[rel],
        })

    ordered_by_layer = {
        layer: by_layer[layer]
        for layer in LAYER_ORDER
        if layer in by_layer
    }
    for layer in sorted(by_layer):
        if layer not in ordered_by_layer:
            ordered_by_layer[layer] = by_layer[layer]

    return {
        "version": AUDIT_VERSION,
        "roots": [r for r in MAIN_CHAIN_ROOTS if r in path_to_mod],
        "summary": {
            "total_py": len(path_to_mod),
            "init_py": len(init_modules),
            "non_init_modules": len(modules),
            "by_status": by_status,
            "by_layer": ordered_by_layer,
        },
        "modules": modules,
        "init_modules": init_modules,
    }


def render_markdown(report: Dict, include_test_refs: bool = False) -> str:
    """将 AST 实测报告渲染为确定性 Markdown 台账（`docs/module-connectivity.md`）。"""
    s = report["summary"]
    bs = s["by_status"]
    lines: List[str] = [
        "# 灵枢模块主链连通性实测台账（#172）",
        "",
        "> **生成方式**：由 `python -X utf8 tools/module_topology_audit.py --write` 基于 AST 静态与惰性导入扫描机械生成，",
        "> 并由 `--check` 门禁及 `tests/test_issue172_module_topology_audit.py` 守护（台账漂移即红）。",
        "",
        "## 一、盘点口径与白箱定义",
        "",
        "- **扫描范围**：`lingshu/` 下全部 `.py` 文件（共 **%d** 个文件：**%d** 个非 `__init__` 模块 + **%d** 个 `__init__.py`）。"
        % (s["total_py"], s["non_init_modules"], s["init_py"]),
        "- **导入边识别**：AST 扫描 `import ...`、`from ... import ...`（含相对/绝对/子包解析）及 `importlib.import_module(\"lingshu....\")` / `__import__(\"lingshu....\")`（含 `core.py` 方法体内的惰性导入）。",
        "- **主入口根节点**：`lingshu/core/core.py`（`SpacetimeMemoryEngine`）、`lingshu/world/world_model.py`（`UnifiedWorldModel`）、`lingshu/world/scene_model.py`（`WorldModel`）。",
        "- **三分类定义**（对全部 **%d** 个非 `__init__` 模块互斥完备划分）：" % s["non_init_modules"],
        "  1. `main_chain`（**核心主入口可达**）：从主入口根节点沿 `lingshu/` 内有向 import 边可达；",
        "  2. `subgraph_only`（**包内子图互调**）：未从主入口直接可达，但在 `lingshu/` 内入度 > 0（含通过包 `__init__.py` 如 `lingshu/world/__init__.py::register_world_process` 等静态解析包内边 接入者）；",
        "  3. `standalone_leaf`（**独立原型叶模块**）：`lingshu/` 包内入度 == 0（包内无任何模块导入它，仅被 `tests/`、`bench/`、`tools/` 或直接 CLI 消费）。",
        "",
        "## 二、汇总读数（实测）",
        "",
        "| 分类 | 模块数 | 占比 | 说明 |",
        "|---|---:|---:|---|",
    ]
    total_non_init = max(1, s["non_init_modules"])
    for key, desc in (
        (STATUS_MAIN_CHAIN, "由 `core.py`（含惰性边界）或 `WorldModel` / `UnifiedWorldModel` 静态/惰性可达"),
        (STATUS_SUBGRAPH_ONLY, "仅在子包/扩展内部或经 `__init__.py`（含依赖倒置注册接口）互调"),
        (STATUS_STANDALONE_LEAF, "包内入度为 0，仅被测试 / 基准 / CLI 独立调用"),
    ):
        cnt = bs[key]
        lines.append("| `%s`（%s） | %d | %.1f%% | %s |" % (
            key, STATUS_LABELS[key], cnt, (cnt * 100.0 / total_non_init), desc))
    lines.append("| **非 `__init__` 合计** | **%d** | **100.0%%** | 与 `docs/architecture.md` 计数同口径 |" % s["non_init_modules"])
    lines.append("")

    lines.extend([
        "### 按层分布",
        "",
        "| 层 | 非 `__init__` 数 | `main_chain` | `subgraph_only` | `standalone_leaf` |",
        "|---|---:|---:|---:|---:|",
    ])
    for layer, b in s["by_layer"].items():
        lines.append("| `%s` | %d | %d | %d | %d |" % (
            layer, b["total"], b[STATUS_MAIN_CHAIN], b[STATUS_SUBGRAPH_ONLY], b[STATUS_STANDALONE_LEAF]))
    lines.append("")

    mod_header = (
        "| 模块路径 | 层 | 状态 | 主入口可达径 | 包内入度 / 导入方 | 包内出度 | `tests/` 直接(传递) | `bench/` 直接(传递) |"
        if include_test_refs
        else "| 模块路径 | 层 | 状态 | 主入口可达径 | 包内入度 / 导入方 | 包内出度 |"
    )
    mod_sep = (
        "|---|---|---|---|---:|---:|---:|---:|"
        if include_test_refs
        else "|---|---|---|---|---:|---:|"
    )
    lines.extend([
        "## 三、全量模块逐件台账（%d 个非 `__init__` 模块）" % s["non_init_modules"],
        "",
        mod_header,
        mod_sep,
    ])
    for m in report["modules"]:
        via = ", ".join(m["reach_via"]) if m["reach_via"] else "—"
        if m["importers"]:
            imp_preview = ", ".join(f"`{x}`" for x in m["importers"][:4])
            if len(m["importers"]) > 4:
                imp_preview += f" 等 {len(m['importers'])} 处"
            in_col = f"{m['in_degree']}（{imp_preview}）"
        else:
            in_col = "0"
        if include_test_refs:
            lines.append(
                "| `%s` | `%s` | `%s`（%s） | %s | %s | %d | %d(%d) | %d(%d) |"
                % (
                    m["path"],
                    m["layer"],
                    m["status"],
                    m["status_zh"],
                    via,
                    in_col,
                    m["out_degree"],
                    m["test_direct"],
                    m["test_transitive"],
                    m["bench_direct"],
                    m["bench_transitive"],
                )
            )
        else:
            lines.append(
                "| `%s` | `%s` | `%s`（%s） | %s | %s | %d |"
                % (
                    m["path"],
                    m["layer"],
                    m["status"],
                    m["status_zh"],
                    via,
                    in_col,
                    m["out_degree"],
                )
            )
    lines.append("")

    init_header = (
        "| 路径 | 层 | 包内入度 | 包内出度（重导出/注册目标） | `tests/` 直接(传递) |"
        if include_test_refs
        else "| 路径 | 层 | 包内入度 | 包内出度（重导出/注册目标） |"
    )
    init_sep = (
        "|---|---|---:|---|---:|"
        if include_test_refs
        else "|---|---|---:|---|"
    )
    lines.extend([
        "## 四、包初始化模块（%d 个 `__init__.py`）" % s["init_py"],
        "",
        init_header,
        init_sep,
    ])
    for im in report["init_modules"]:
        outs = ", ".join(f"`{x}`" for x in im["imports"]) if im["imports"] else "—"
        if include_test_refs:
            lines.append("| `%s` | `%s` | %d | %s | %d(%d) |" % (
                im["path"], im["layer"], im["in_degree"], outs, im["test_direct"], im["test_transitive"]))
        else:
            lines.append("| `%s` | `%s` | %d | %s |" % (
                im["path"], im["layer"], im["in_degree"], outs))
    lines.append("")

    lines.extend([
        "## 五、诚实边界说明",
        "",
        "1. 本台账按当前仓库树上 AST 静态/惰性 import 实测生成：凡未从核心主入口导入的模块均如实标为 `subgraph_only` 或 `standalone_leaf`，不把「有单测覆盖」伪装为「已接入主链」。",
        "2. `lingshu/world/__init__.py::register_world_process(engine)` 采用依赖倒置（调用方显式把 `world_process` / `motion_groups` / `process_ops` 注册给 `SpacetimeMemoryEngine` 实例），静态图中 `core.py` 不反向导入它，故三模块在此客观记为 `subgraph_only`（由 `lingshu/world/__init__.py` 导入）。",
        "3. `lingshu/vision/__init__.py` 通过相对动态导入延迟暴露 `lineart` 符号；默认 AST 扫描仅解析静态导入及字面绝对 `lingshu.*` 动态导入，因此该边界未计入包内图，`lineart.py` 按当前扫描结果保持 `standalone_leaf`。默认 `--check` 仅锁定 `lingshu/` 包内拓扑；查看 tests/bench 外部引用可运行 `python -X utf8 tools/module_topology_audit.py --markdown --with-test-refs` 或 `--json`。",
        "4. 若后续 PR 修改 `lingshu/` 模块集合或包内跨模块导入关系（如切断 `world -> core` 反向边或将原型模块接入主链），须同步执行 `python -X utf8 tools/module_topology_audit.py --write` 更新本台账，否则 `--check` 门禁将精确报出漂移；仅增删 `tests/` 或 `bench/` 文件不会触发本门禁漂移。",
        "",
    ])
    return "\n".join(lines)


def _normalize_lf(text: str) -> str:
    return text.replace("\r\n", "\n")


def run(argv: Optional[Sequence[str]] = None) -> Tuple[int, str, Dict]:
    parser = argparse.ArgumentParser(
        prog="module_topology_audit.py",
        description="全仓模块主链连通性 AST 实测盘点与台账漂移守卫（#172）",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--json", action="store_true", help="输出确定性 JSON（sort_keys=True）")
    mode.add_argument("--markdown", action="store_true", help="输出生成的 Markdown 台账文本")
    mode.add_argument("--write", action="store_true", help="将实测台账写入 --ledger（默认 docs/module-connectivity.md）")
    mode.add_argument("--check", action="store_true", help="核对 --ledger 与当前仓 AST 实算是否逐字一致")
    parser.add_argument(
        "--with-test-refs",
        action="store_true",
        help="在 --markdown 输出中附带 tests/ 与 bench/ 实时直接/传递引用计数列（不纳入默认 --check 阻断面）",
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT), help="仓库根目录")
    parser.add_argument("--ledger", default=DEFAULT_LEDGER_REL, help="台账 Markdown 路径（相对仓库根或绝对路径）")
    args = parser.parse_args(argv)

    root = Path(args.repo_root).resolve()
    if not root.is_dir() or not (root / "lingshu").is_dir():
        return 2, "仓库根或 lingshu/ 目录不存在\n", {"exit_code": 2}

    ledger_arg = Path(args.ledger)
    ledger_path = ledger_arg if ledger_arg.is_absolute() else root / ledger_arg

    report = analyze(root, include_external_refs=bool(args.json or args.with_test_refs))
    expected_md = render_markdown(report, include_test_refs=False)

    if args.json:
        payload = {
            **report,
            "exit_code": 0,
        }
        return 0, json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n", payload

    if args.markdown:
        md_out = render_markdown(report, include_test_refs=bool(args.with_test_refs))
        return 0, md_out, {"exit_code": 0, **report["summary"]}

    if args.write:
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger_path.write_text(expected_md, encoding="utf-8", newline="\n")
        s = report["summary"]
        bs = s["by_status"]
        msg = (
            f"已写入 {ledger_path.relative_to(root).as_posix() if ledger_path.is_relative_to(root) else ledger_path.name}："
            f"total_py={s['total_py']} non_init={s['non_init_modules']} "
            f"main_chain={bs[STATUS_MAIN_CHAIN]} subgraph_only={bs[STATUS_SUBGRAPH_ONLY]} "
            f"standalone_leaf={bs[STATUS_STANDALONE_LEAF]}\n"
        )
        return 0, msg, {"exit_code": 0, **s}

    if args.check:
        if not ledger_path.is_file():
            return 1, "FAIL: 台账文件不存在（请运行 --write 生成）\n", {"exit_code": 1, "missing_ledger": True}
        actual_md = _normalize_lf(ledger_path.read_text(encoding="utf-8-sig", errors="replace"))
        if actual_md == expected_md:
            s = report["summary"]
            bs = s["by_status"]
            return 0, (
                f"OK: 模块连通性台账一致（total_py={s['total_py']} non_init={s['non_init_modules']} "
                f"main_chain={bs[STATUS_MAIN_CHAIN]} subgraph_only={bs[STATUS_SUBGRAPH_ONLY]} "
                f"standalone_leaf={bs[STATUS_STANDALONE_LEAF]}）\n"
            ), {"exit_code": 0, **s}
        diff_lines = list(difflib.unified_diff(
            actual_md.splitlines(),
            expected_md.splitlines(),
            fromfile="recorded",
            tofile="actual",
            lineterm="",
            n=1,
        ))
        drifted = [ln for ln in diff_lines if ln.startswith(("+|", "-|"))]
        drifted_modules = sorted({
            m.group(1)
            for ln in drifted
            for m in [re.search(r"`(lingshu/[^`]+\.py)`", ln)]
            if m
        })
        mods_line = (
            f"漂移模块 ({len(drifted_modules)}): " + ", ".join(drifted_modules) + "\n"
            if drifted_modules
            else ""
        )
        msg = "FAIL: 模块连通性台账与 AST 实算漂移（差异行 %d）:\n%s%s\n" % (
            len(drifted),
            mods_line,
            "\n".join(diff_lines[:40]),
        )
        return 1, msg, {
            "exit_code": 1,
            "drifted_modules": drifted_modules,
            "drifted_lines": drifted,
            "diff": diff_lines[:60],
        }

    # 默认打印简表
    s = report["summary"]
    bs = s["by_status"]
    out = (
        f"total_py={s['total_py']} init={s['init_py']} non_init={s['non_init_modules']} | "
        f"main_chain={bs[STATUS_MAIN_CHAIN]} subgraph_only={bs[STATUS_SUBGRAPH_ONLY]} "
        f"standalone_leaf={bs[STATUS_STANDALONE_LEAF]}\n"
    )
    return 0, out, {"exit_code": 0, **s}


def main(argv: Optional[Sequence[str]] = None) -> int:
    _configure_stdout()
    code, out, _ = run(argv)
    if out:
        sys.stdout.write(out)
    return code


if __name__ == "__main__":
    sys.exit(main())
