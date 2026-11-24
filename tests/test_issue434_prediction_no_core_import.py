# -*- coding: utf-8 -*-
"""test_issue434_prediction_no_core_import · #434 prediction 零 core/旧名反向导入守卫
========================================================================================
目标（issue #434 / 准备稿 §二 硬前置 1）：
  切断 `lingshu/world/prediction.py` 对 `lingshu.core` 及旧包名 `spacetime_memory_core`
  的反向导入；改由鸭子类型 `PredictionStoreProtocol` / `PredictionEngineHostProtocol`
  与纯标准库 `_char_bigram_jaccard` 回退承接。

断言组：
  1. AST 静态扫描：`prediction.py` 不得出现任何对 `core`（相对 `..core`、绝对
     `lingshu.core`、动态 `import_module` / `__import__`）或 `spacetime_memory_core`
     的导入与代码/注释残留（`stmc_refs == 0`）。
  2. 阻断 `lingshu.core` 与 `spacetime_memory_core` 后独立导入与核心预测闭环：
     在干净子进程预先将 `lingshu.core` / `lingshu.core.core` / `spacetime_memory_core`
     置为 `None`（强制阻塞导入）并导入 `lingshu.world.prediction`，验证：
     - `PredictionEngine` 构造、`semantic_neighbors`、`predict_routes`（含 `start_id`
       与 `blindspot_id`）、`update_prediction_feedback`（命中强化 & 未命中登记/自动
       条件化）、`render_semantic_map_2d`、`render_semantic_cube_3d` 全部跑通；
     - `sys.modules` 中不存在已加载的 `lingshu.core*` 或 `spacetime_memory_core`。
  3. 协议与相似度口径一致：
     - `PredictionStoreProtocol` / `PredictionEngineHostProtocol` 鸭子类型检查通过；
     - `_char_bigram_jaccard` 与 `LayeredStore.char_bigram_jaccard`（在非阻断环境对拍）
       逐位一致（含非字符串 `TypeError`）。
  4. 定点变异自证（含 `--mutate` CLI）：
     - 向源码副本注入 `from ..core.core import LayeredStore` ⇒ AST 扫描必红；
     - 向源码副本注入 `from spacetime_memory_core import LayeredStore` ⇒ AST 扫描必红；
     - 还原后全绿。

运行（仓根）：
  python -X utf8 tests/test_issue434_prediction_no_core_import.py
  python -X utf8 tests/test_issue434_prediction_no_core_import.py --mutate
  python -X utf8 -m pytest tests/test_issue434_prediction_no_core_import.py -q --no-header
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PRED_PATH = REPO_ROOT / "lingshu" / "world" / "prediction.py"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _scan_prediction_source(src: str) -> list:
    """AST + 文本扫描源码中对 core / spacetime_memory_core 的反向依赖与残留。

    返回违规描述列表（空列表 = 干净）。
    """
    violations = []
    tree = ast.parse(src)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if (
                    name == "spacetime_memory_core"
                    or name.startswith("spacetime_memory_core.")
                    or name == "lingshu.core"
                    or name.startswith("lingshu.core.")
                ):
                    violations.append(f"L{node.lineno}: import {name}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            # 相对导入（..core / ..core.core / .. import core）
            if node.level and node.level >= 2:
                if mod == "core" or mod.startswith("core."):
                    violations.append(
                        f"L{node.lineno}: from {'.' * node.level}{mod} import ..."
                    )
                if not mod:
                    for alias in node.names:
                        if alias.name == "core":
                            violations.append(
                                f"L{node.lineno}: from {'.' * node.level} import core"
                            )
            # 绝对导入
            if (
                mod == "spacetime_memory_core"
                or mod.startswith("spacetime_memory_core.")
                or mod == "lingshu.core"
                or mod.startswith("lingshu.core.")
            ):
                violations.append(f"L{node.lineno}: from {mod} import ...")
        elif isinstance(node, ast.Call):
            fn = node.func
            fn_name = None
            if isinstance(fn, ast.Name):
                fn_name = fn.id
            elif isinstance(fn, ast.Attribute):
                fn_name = fn.attr
            if fn_name in ("import_module", "__import__") and node.args:
                arg0 = node.args[0]
                if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                    target = arg0.value
                    if (
                        "spacetime_memory_core" in target
                        or "lingshu.core" in target
                        or target.startswith("..core")
                    ):
                        violations.append(
                            f"L{node.lineno}: {fn_name}({target!r})"
                        )

    for lineno, line in enumerate(src.splitlines(), start=1):
        if "spacetime_memory_core" in line:
            violations.append(f"L{lineno}: literal token spacetime_memory_core")

    return violations


def test_ast_zero_core_and_stmc_imports():
    """① AST 静态守卫：`prediction.py` 零 `core` / 零 `spacetime_memory_core` 引用。"""
    src = PRED_PATH.read_text(encoding="utf-8")
    violations = _scan_prediction_source(src)
    assert violations == [], (
        "lingshu/world/prediction.py 存在反向依赖或旧名残留: %r" % (violations,)
    )


def test_standalone_import_and_core_flow_with_core_blocked():
    """② 阻断 `lingshu.core` 与 `spacetime_memory_core` 后独立导入并跑通预测闭环。"""
    probe = r"""
import json
import sys

# 强阻断：任何尝试导入 lingshu.core / spacetime_memory_core 的行为立即抛 ModuleNotFoundError
for mod in (
    "lingshu.core",
    "lingshu.core.core",
    "lingshu.core.provenance",
    "lingshu.core.component_resolver",
    "spacetime_memory_core",
):
    sys.modules[mod] = None

from lingshu.world.prediction import (
    PredictionEngine,
    PredictionEngineHostProtocol,
    PredictionStoreProtocol,
)

class _Rel:
    def __init__(self, value):
        self.value = value

class _CS:
    def __init__(self, ec="", pos=""):
        self._d = {"existence_constraint": ec, "observation_position": pos}
    def to_json(self):
        return json.dumps(self._d, ensure_ascii=False)

class _Node:
    def __init__(self, nid, content, t=1.0, tags=None, sc=None, ec="", eid=None):
        self.id = nid
        self.content = content
        self.temporal_coordinate = t
        self.tags = tags
        self.semantic_coordinates = sc or {}
        self.condition_space = _CS(ec=ec) if ec else None
        self.entity_id = eid

class _Edge:
    def __init__(self, eid, src, dst, rel="causal", conf=0.7, verified=False, ec=""):
        self.id = eid
        self.source_id = src
        self.target_id = dst
        self.relation_type = _Rel(rel)
        self.confidence = conf
        self.verified = verified
        self.condition_space = _CS(ec=ec) if ec else None

class _DuckStore:
    def __init__(self):
        self.nodes = {
            "n0": _Node("n0", "机器状态正常预测起点", t=1.0, tags=["boundary", "ent:e1"],
                        sc={"protocol": {"concept": {"stability": 0.8, "risk": 0.1}}},
                        ec="电源已接通", eid="e1"),
            "n1": _Node("n1", "机器状态温度上升趋势", t=2.0, tags=[],
                        sc={"protocol": {"concept": {"stability": 0.6, "risk": 0.4}}},
                        ec="负载持续上升", eid="e1"),
            "n2": _Node("n2", "机器状态正常预测邻居", t=3.0, tags=None,
                        sc={"protocol": {"concept": {"stability": 0.7, "risk": 0.2}}},
                        ec="", eid="e1"),
            "n3": _Node("n3", "完全不相关的食堂菜单", t=4.0, tags=[],
                        sc={"protocol": {"concept": {"food": 0.9}}},
                        ec="内部条件存在", eid="e2"),
        }
        self.edges = [
            _Edge("e01", "n0", "n1", rel="causal", conf=0.85, verified=False, ec="边存在约束"),
            _Edge("e12", "n1", "n2", rel="sequential", conf=0.65, verified=False),
        ]
        self.verified_updates = []

    def get_node(self, node_id):
        return self.nodes.get(node_id)

    def query_nodes(self, limit=300):
        return list(self.nodes.values())[:limit]

    def get_outgoing_edges(self, node_id):
        return [e for e in self.edges if e.source_id == node_id]

    def get_incoming_edges(self, node_id):
        return [e for e in self.edges if e.target_id == node_id]

    def verify_edge(self, edge_id, new_confidence):
        self.verified_updates.append((edge_id, new_confidence))
        for e in self.edges:
            if e.id == edge_id:
                e.verified = True
                e.confidence = new_confidence

class _DuckEngine:
    def __init__(self, store):
        self.store = store
        self.rejected = []
        self.perceptions = []

    def list_blindspots(self):
        return [
            {"id": "bs_know", "predictability": "predictable", "description": "机器状态"},
            {"id": "bs_unk", "predictability": "unknowable", "description": "不可知"},
        ]

    def search_content(self, query, limit=3):
        hits = [n for n in self.store.query_nodes() if query in n.content]
        return [(n, 0.9) for n in hits[:limit]]

    def register_rejected_path(self, **kwargs):
        self.rejected.append(kwargs)
        return "rej_1"

    def add_perception(self, **kwargs):
        nid = f"cond_{len(self.perceptions) + 1}"
        self.perceptions.append(kwargs)
        return type("N", (), {"id": nid})()

store = _DuckStore()
eng = _DuckEngine(store)
assert isinstance(store, PredictionStoreProtocol)
assert isinstance(eng, PredictionEngineHostProtocol)

pe = PredictionEngine(eng)
nbrs = [n.id for n in pe.semantic_neighbors("n0", k=5)]
assert "n2" in nbrs and "n3" not in nbrs, nbrs

res = pe.predict_routes(start_id="n0", horizon=2, max_branches=5)
paths = [r["path"] for r in res["routes"]]
assert ["n0", "n1"] in paths and ["n0", "n1", "n2"] in paths, paths

bs_res = pe.predict_routes(blindspot_id="bs_know", horizon=2)
assert bs_res["meta"]["blindspot_id"] == "bs_know", bs_res

fb_hit = pe.update_prediction_feedback("n1", "n1", hit=True)
assert fb_hit["samples"] == 1
assert store.verified_updates == [("e01", 0.9)], store.verified_updates

bs_unk = pe.predict_routes(blindspot_id="bs_unk", horizon=2)
assert bs_unk["status"] == "unpredictable", bs_unk

fb_miss = pe.update_prediction_feedback("n1", "n3", hit=False, note="test")
assert fb_miss["samples"] == 2
assert len(eng.rejected) == 1 and len(eng.perceptions) >= 1

m2d = pe.render_semantic_map_2d(limit=10)
assert m2d["count"] == 4 and len(m2d["axes"]) == 2, m2d

c3d = pe.render_semantic_cube_3d(entity_id="e1", limit=10)
assert len(c3d["trajectory"]) == 3, c3d

loaded_core = [k for k, v in sys.modules.items() if v is not None and (k == "lingshu.core" or k.startswith("lingshu.core.") or k == "spacetime_memory_core")]
assert not loaded_core, f"core unexpectedly loaded: {loaded_core}"
print("STANDALONE_OK")
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["PYTHONUTF8"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", probe],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert proc.returncode == 0, (
        f"子进程独立导入/预测失败 rc={proc.returncode}\n"
        f"stdout={proc.stdout}\nstderr={proc.stderr}"
    )
    assert "STANDALONE_OK" in proc.stdout, proc.stdout


def test_jaccard_parity_with_core_and_type_checks():
    """③ 纯标准库 `_char_bigram_jaccard` 与 `LayeredStore.char_bigram_jaccard` 逐位一致。"""
    from lingshu.core.core import LayeredStore
    from lingshu.world.prediction import _char_bigram_jaccard

    pairs = [
        ("", ""),
        ("", "abc"),
        ("单", "单"),
        ("单", "双"),
        ("预测引擎语义通道在本仓整体静默失效", "预测引擎语义通道在本仓整体失效"),
        ("hello world", "hello  world!"),
        ("完全无关的另一种内容", "机器状态正常预测起点"),
    ]
    for a, b in pairs:
        assert _char_bigram_jaccard(a, b) == LayeredStore.char_bigram_jaccard(a, b), (a, b)

    for bad_a, bad_b in ((123, "abc"), ("abc", None)):
        try:
            _char_bigram_jaccard(bad_a, bad_b)
            raise AssertionError("expected TypeError for (%r, %r)" % (bad_a, bad_b))
        except TypeError:
            pass


def run_mutation_check() -> dict:
    """④ 定点变异自证：向源码副本注入假 core/stmc 导入 ⇒ 必红，还原 ⇒ 绿。"""
    clean_src = PRED_PATH.read_text(encoding="utf-8")
    clean_hits = _scan_prediction_source(clean_src)
    assert clean_hits == [], f"基线应干净，实得 {clean_hits}"

    mut_core_src = "from ..core.core import LayeredStore\n" + clean_src
    mut_core_hits = _scan_prediction_source(mut_core_src)
    assert mut_core_hits, "注入假 `from ..core.core import LayeredStore` 后守卫竟未判红"

    mut_stmc_src = "from spacetime_memory_core import LayeredStore\n" + clean_src
    mut_stmc_hits = _scan_prediction_source(mut_stmc_src)
    assert mut_stmc_hits, "注入假 `from spacetime_memory_core import LayeredStore` 后守卫竟未判红"

    # 临时副本文件验证（模拟真实盘上文件还原）
    with tempfile.TemporaryDirectory(prefix="pred_mut_") as td:
        copy_file = Path(td) / "prediction.py"
        copy_file.write_text(mut_core_src, encoding="utf-8")
        assert _scan_prediction_source(copy_file.read_text(encoding="utf-8"))
        copy_file.write_text(clean_src, encoding="utf-8")
        assert _scan_prediction_source(copy_file.read_text(encoding="utf-8")) == []

    return {
        "clean_hits": clean_hits,
        "mut_core_hits": mut_core_hits,
        "mut_stmc_hits": mut_stmc_hits,
    }


def test_mutation_self_proof():
    """④ 定点变异自证（pytest 入口）。"""
    res = run_mutation_check()
    assert res["clean_hits"] == []
    assert len(res["mut_core_hits"]) >= 1
    assert len(res["mut_stmc_hits"]) >= 2  # AST import + token


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    test_ast_zero_core_and_stmc_imports()
    test_standalone_import_and_core_flow_with_core_blocked()
    test_jaccard_parity_with_core_and_type_checks()
    mut = run_mutation_check()
    if "--mutate" in args:
        print("MUTATE_CHECK_OK", json.dumps(mut, ensure_ascii=False))
    else:
        print("ISSUE434_PREDICTION_NO_CORE_IMPORT_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
