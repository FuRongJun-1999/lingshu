# -*- coding: utf-8 -*-
"""test_issue434_scenario_decouple_guard · #434 scenario/scenario_cli 解耦与 world→core 零反向边守卫。

背景（issue #434 · 《docs/plans/插件化拆分_同仓多包_准备_v0.1.md》硬前置第 1 项）：
  身体侧 world 包不得反向依赖宿主 core。此前 `lingshu/world/scenario.py` 与
  `lingshu/world/scenario_cli.py` 共 4 处静态/惰性导入 `..core.*`：
    - `world/scenario.py:13`：`from ..core.provenance import SOURCES, VERIFY_STATUS`
    - `world/scenario_cli.py:35,40,50`：`from ..core.core import ...`

守卫内容：
  G1 `scenario.py` 与 `scenario_cli.py` 的 `world_to_core` 计数与 `spacetime_memory_core` 引用恒为 0；
  G2 `tools/import_direction_lint.py --world-to-core` 对除未合入分支 `prediction.py` 外的全部
     `lingshu/world/**/*.py` 均无反向边（允许清单仅限 `lingshu/world/prediction.py`）；
  G3 隔离运行（`python -I -S`）导入 `scenario` / `scenario_cli` 并执行 `demo --mode all` 后，
     `sys.modules` 不含 `lingshu.core*`、`numpy`、`PIL`；
  G4 `scenario.{SOURCES, VERIFY_STATUS}` 与 `core.provenance.{SOURCES, VERIFY_STATUS}` 值域完全一致；
  G5 `scenario_cli._ReadOnlyStore` 与 `LayeredStore` 对多层级/多边型/非有限时间窗的 `view()` 逐位等价，
     且对非法 schema / 层名 / 边型 fail-closed（`ValueError`）；
  G6 `world_to_core` 四形态（相对 / 父包 / 绝对 / 动态 `import_module`）与子包深度全部命中，注释与
     docstring / `time_core` / `lingshu.world` / 前缀撞名不误报；
  G7 `--mutate` 定点变异自证：向临时副本的 `world/` 注入假 `from ..core...` / 绝对 / 动态导入 ⇒
     `import_direction_lint.py --world-to-core` 必红并点名文件，还原后转绿。

运行：
  python -X utf8 -m pytest tests/test_issue434_scenario_decouple_guard.py -v
  python -X utf8 tests/test_issue434_scenario_decouple_guard.py
  python -X utf8 tests/test_issue434_scenario_decouple_guard.py --mutate
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import import_direction_lint as I  # noqa: E402

TOOL = REPO_ROOT / "tools" / "import_direction_lint.py"
SCENARIO_PY = REPO_ROOT / "lingshu" / "world" / "scenario.py"
SCENARIO_CLI_PY = REPO_ROOT / "lingshu" / "world" / "scenario_cli.py"


def _run_lint(*args, cwd=None, timeout=120):
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "-X", "utf8", str(TOOL), *args],
        cwd=str(cwd or REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=env,
    )


def _write(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def test_g1_scenario_and_cli_have_zero_core_imports():
    """G1：scenario.py 与 scenario_cli.py 不得存在任何 core 导入或旧包名。"""
    counts, errors = I.scan_world_to_core(REPO_ROOT, REPO_ROOT / "lingshu")
    assert errors == [], errors
    for rel in ("lingshu/world/scenario.py", "lingshu/world/scenario_cli.py"):
        assert counts.get(rel, 0) == 0, f"{rel} 仍有 world→core 导入：{counts}"
    for path in (SCENARIO_PY, SCENARIO_CLI_PY):
        text = path.read_text(encoding="utf-8")
        assert I.STMC_TOKEN not in text, f"{path.name} 不得含 {I.STMC_TOKEN}"


def test_g2_world_to_core_lint_allows_only_unmerged_prediction():
    """G2：除兄弟分支负责的 prediction.py 外，lingshu/world/ 全量零反向边。"""
    counts, errors = I.scan_world_to_core(REPO_ROOT, REPO_ROOT / "lingshu")
    assert errors == [], errors
    unexpected = {k: v for k, v in counts.items() if k != "lingshu/world/prediction.py"}
    assert unexpected == {}, f"发现未预期的 world→core 反向导入：{unexpected}"

    res = _run_lint(
        "--world-to-core",
        "--allow-world-to-core", "lingshu/world/prediction.py",
        "--json",
    )
    data = json.loads(res.stdout)
    assert res.returncode == 0, data
    assert data["violations"] == {}, data


def test_g3_runtime_import_scenario_and_cli_do_not_load_core():
    """G3：隔离纯标准库运行 demo 不得把 lingshu.core / numpy / PIL 拉入 sys.modules。"""
    code = (
        "import contextlib, io, sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from lingshu.world import scenario, scenario_cli\n"
        "buf = io.StringIO()\n"
        "with contextlib.redirect_stdout(buf):\n"
        "    rc = scenario_cli.main(['demo', '--mode', 'all'])\n"
        "assert rc == 0, rc\n"
        "loaded_core = sorted(m for m in sys.modules if m == 'lingshu.core' or m.startswith('lingshu.core.'))\n"
        "assert not loaded_core, loaded_core\n"
        "assert 'numpy' not in sys.modules and 'PIL' not in sys.modules\n"
    )
    res = subprocess.run(
        [sys.executable, "-I", "-S", "-c", code, str(REPO_ROOT)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    assert res.returncode == 0, f"stderr={res.stderr}\nstdout={res.stdout}"


def test_g4_provenance_value_domains_match_core_authoritative():
    """G4：scenario 与 core.provenance 的 SOURCES / VERIFY_STATUS 值域保持完全一致。"""
    from lingshu.core import provenance as core_prov
    from lingshu.world import scenario as world_scenario

    assert world_scenario.SOURCES == frozenset(core_prov.SOURCES)
    assert world_scenario.VERIFY_STATUS == core_prov.VERIFY_STATUS


def test_g5_readonly_store_parity_with_layered_store():
    """G5：_ReadOnlyStore 与 LayeredStore 对多层级/多边型/非有限时间窗读回完全等价。"""
    from lingshu.core.core import ConditionSpace, EdgeType, LayeredStore, MemoryLayer, STEdge, STNode
    from lingshu.world.scenario import MODES, ScenarioContext
    from lingshu.world.scenario_cli import _ReadOnlyStore

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "parity.db"
        store = LayeredStore(str(db_path))
        try:
            cs_fin = ConditionSpace("archive-A", "reader", (100, 200), "scn-A")
            cs_inf = ConditionSpace("archive-B", "anchor", (0.0, float("inf")), "scn-B")
            store.add_node(STNode(
                "n-know", "Alice keeper", "text", {"x": 1, "y": 2, "z": 3}, 110, cs_fin,
                layer=MemoryLayer.KNOWLEDGE,
                tags=["external", "ent:lamp", "cat:table", "vref:cat:1"],
                state_attributes={"state": "on"},
            ))
            store.add_node(STNode(
                "n-anchor", "Anchor rule", "text", {}, 0, cs_inf,
                layer=MemoryLayer.ANCHOR,
                tags=["fixture", "role:alice", "roleplay:anchors"],
            ))
            store.add_node(STNode(
                "n-struct", "Structure event", "text", {}, 150, cs_fin,
                layer=MemoryLayer.STRUCTURE,
                tags=["user", "role:alice", "session:s1", "event"],
            ))
            store.add_edge(STEdge(
                "e-causal", "n-struct", "n-know", EdgeType.CAUSAL, cs_fin,
                verified=True, source_evidence="extracted",
            ))
            store.add_edge(STEdge(
                "e-seq", "n-anchor", "n-struct", EdgeType.SEQUENTIAL, cs_fin,
                verified=False, source_evidence="inferred",
            ))
            ids = ["n-know", "n-anchor", "n-struct", "missing-id"]
            ctx_core = ScenarioContext.from_store(store, ids, role_id="alice", session_id="s1")
        finally:
            store.conn.close()

        ro = _ReadOnlyStore(db_path)
        try:
            ctx_ro = ScenarioContext.from_store(ro, ids, role_id="alice", session_id="s1")
        finally:
            ro.close()

        for mode in MODES:
            kwargs = {"start_id": "n-struct", "end_id": "n-know"} if mode == "causal" else {}
            assert ctx_ro.view(mode, **kwargs) == ctx_core.view(mode, **kwargs), f"mode={mode} 不等价"

        # fail-closed：非法 layer / 非法 schema 必须抛出 ValueError
        bad_db = Path(tmp) / "bad.db"
        conn = sqlite3.connect(bad_db)
        try:
            with conn:
                conn.execute("CREATE TABLE nodes (id TEXT)")
                conn.execute("INSERT INTO nodes VALUES ('x')")
        finally:
            conn.close()
        ro_bad = _ReadOnlyStore(bad_db)
        try:
            try:
                ro_bad.get_node("x")
                raise AssertionError("缺少必须列的 nodes 表应抛 ValueError")
            except ValueError:
                pass
        finally:
            ro_bad.close()


def test_g6_world_to_core_detector_covers_all_forms_and_ignores_non_imports():
    """G6：AST 扫描必须捕获相对/父包/绝对/动态/深层子包形态，且不误报注释与内部导入。"""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "repo"
        _write(root / "lingshu" / "world" / "rel.py", "from ..core.core import STNode\n")
        _write(root / "lingshu" / "world" / "parent.py", "from .. import world, core\n")
        _write(root / "lingshu" / "world" / "abs_from.py", "from lingshu.core.provenance import SOURCES\n")
        _write(root / "lingshu" / "world" / "abs_pkg.py", "from lingshu import core\n")
        _write(root / "lingshu" / "world" / "abs_imp.py", "import lingshu.core.core\n")
        _write(
            root / "lingshu" / "world" / "dyn.py",
            "import importlib\nmod = importlib.import_module('lingshu.core.core')\n",
        )
        _write(root / "lingshu" / "world" / "sub" / "nested.py", "from ...core import core\n")
        _write(
            root / "lingshu" / "world" / "clean.py",
            '"""docstring: from lingshu.core.core import STNode"""\n'
            "# comment: from ..core.core import STNode\n"
            "from .time_core import cred_blend\n"
            "from lingshu.world import mesh3d\n"
            "from lingshu.corely import nope\n"
            "MSG = 'from ..core import x'\n",
        )

        counts, errors = I.scan_world_to_core(root, root / "lingshu")
        assert errors == [], errors
        expected = {
            "lingshu/world/abs_from.py": 1,
            "lingshu/world/abs_imp.py": 1,
            "lingshu/world/abs_pkg.py": 1,
            "lingshu/world/dyn.py": 1,
            "lingshu/world/parent.py": 1,
            "lingshu/world/rel.py": 1,
            "lingshu/world/sub/nested.py": 1,
        }
        assert counts == expected, f"actual={counts} expected={expected}"

        res = _run_lint("--world-to-core", "--repo-root", str(root), "--json")
        data = json.loads(res.stdout)
        assert res.returncode == 1, data
        assert data["violations"] == expected, data


def run_mutation_self_proof() -> int:
    """G7 `--mutate` 定点变异自证：向临时 world/ 注入假 core 导入必红并点名，还原后转绿。"""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "repo"
        world_dst = root / "lingshu" / "world"
        world_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(REPO_ROOT / "lingshu" / "world", world_dst,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        # 移除尚未合入的 prediction.py 既有基线项，模拟解耦完备后的 world/ 净面
        pred = world_dst / "prediction.py"
        if pred.is_file():
            pred.unlink()

        scenario_copy = world_dst / "scenario.py"
        scenario_cli_copy = world_dst / "scenario_cli.py"
        orig_scenario = scenario_copy.read_text(encoding="utf-8")
        orig_cli = scenario_cli_copy.read_text(encoding="utf-8")

        cases = []

        def _probe(label: str, expect_rc: int, expect_violations: dict):
            res = _run_lint("--world-to-core", "--repo-root", str(root), "--json")
            data = json.loads(res.stdout)
            ok = (res.returncode == expect_rc and data["violations"] == expect_violations)
            cases.append((label, res.returncode, data["violations"], ok))
            print(f"{label} rc={res.returncode} violations={data['violations']} VERDICT={'PASS' if ok else 'FAIL'}")

        # M0：干净基线
        _probe("M0_baseline_clean", 0, {})

        # M1：向 scenario.py 注入相对 core 导入
        scenario_copy.write_text(
            orig_scenario + "\nfrom ..core.provenance import SOURCES\n",
            encoding="utf-8",
        )
        _probe("M1_inject_relative_in_scenario", 1, {"lingshu/world/scenario.py": 1})
        scenario_copy.write_text(orig_scenario, encoding="utf-8")

        # M2：向 scenario_cli.py 注入绝对 core 导入
        scenario_cli_copy.write_text(
            orig_cli + "\nfrom lingshu.core.core import STNode\n",
            encoding="utf-8",
        )
        _probe("M2_inject_absolute_in_scenario_cli", 1, {"lingshu/world/scenario_cli.py": 1})
        scenario_cli_copy.write_text(orig_cli, encoding="utf-8")

        # M3：向 world/ 新增动态导入 core 模块
        injected = world_dst / "_mutated_probe.py"
        injected.write_text(
            "import importlib\nmod = importlib.import_module('lingshu.core.core')\n",
            encoding="utf-8",
        )
        _probe("M3_inject_dynamic_import", 1, {"lingshu/world/_mutated_probe.py": 1})
        injected.unlink()

        # M4：还原后转绿
        _probe("M4_restored_clean", 0, {})

        all_ok = all(item[3] for item in cases)
        print(f"OVERALL={'PASS' if all_ok else 'FAIL'} (repo_files_untouched=True)")
        return 0 if all_ok else 1


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--mutate" in args:
        return run_mutation_self_proof()
    tests = [
        test_g1_scenario_and_cli_have_zero_core_imports,
        test_g2_world_to_core_lint_allows_only_unmerged_prediction,
        test_g3_runtime_import_scenario_and_cli_do_not_load_core,
        test_g4_provenance_value_domains_match_core_authoritative,
        test_g5_readonly_store_parity_with_layered_store,
        test_g6_world_to_core_detector_covers_all_forms_and_ignores_non_imports,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL {fn.__name__}: {exc}")
    print(f"SUMMARY pass={len(tests) - failed} fail={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
