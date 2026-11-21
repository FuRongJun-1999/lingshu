#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_issue172_module_topology_audit · #172 全仓模块主链连通性台账与漂移门禁守卫

覆盖要点：
1. 守恒与完备性：main_chain + subgraph_only + standalone_leaf == non_init_modules，
   non_init_modules + init_py == total_py，各层小计之和等于总非 __init__ 模块数，
   三类模块集合互斥且并集全覆盖。
2. 真实仓 --check 门禁通过，且台账中诚实披露非零的 subgraph_only 与 standalone_leaf 模块。
3. --json 跨工作目录字节一致且不泄露任何本机路径片段。
4. 定点漂移自证（支持 pytest 与 `python ... --mutate`）：
   在临时仓副本上分别注入：
     (a) 新增跨模块 import（将孤立叶模块接入 core.py）；
     (b) 删除既有模块间 import（断开 verdict.py -> condition_normalize.py）；
     (c) 篡改台账状态行；
     (d) 在 lingshu/ 下新增未入账 .py 模块。
   每次注入均验证 `--check` 必退出非零（rc=1）并点名漂移模块；恢复后必回绿（rc=0）。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import module_topology_audit as audit  # noqa: E402


def _make_repo_copy(dst_root: Path) -> Path:
    """复制最小完整仓拓扑快照（lingshu/tests/bench/docs 台账）用于隔离变异测试。"""
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
    for sub in ("lingshu", "tests", "bench"):
        src = REPO_ROOT / sub
        if src.is_dir():
            shutil.copytree(src, dst_root / sub, ignore=ignore)
    ledger_dst = dst_root / audit.DEFAULT_LEDGER_REL
    ledger_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / audit.DEFAULT_LEDGER_REL, ledger_dst)
    return dst_root


def test_ledger_check_passes_on_repo() -> None:
    rc, msg, data = audit.run(["--repo-root", str(REPO_ROOT), "--check"])
    assert rc == 0, f"--check failed on repo:\n{msg}"
    assert data.get("exit_code") == 0


def test_conservation_and_disjoint_partition() -> None:
    report = audit.analyze(REPO_ROOT)
    s = report["summary"]
    bs = s["by_status"]
    modules = report["modules"]
    init_modules = report["init_modules"]

    assert s["total_py"] == s["init_py"] + s["non_init_modules"]
    assert s["init_py"] == len(init_modules)
    assert s["non_init_modules"] == len(modules)

    total_by_status = (
        bs[audit.STATUS_MAIN_CHAIN]
        + bs[audit.STATUS_SUBGRAPH_ONLY]
        + bs[audit.STATUS_STANDALONE_LEAF]
    )
    assert total_by_status == s["non_init_modules"]

    # 诚实台账断言：非全部主链，三类均有清晰实测分布，且主入口根均在 main_chain
    assert bs[audit.STATUS_MAIN_CHAIN] > 0
    assert bs[audit.STATUS_SUBGRAPH_ONLY] > 0
    assert bs[audit.STATUS_STANDALONE_LEAF] > 0

    buckets = {
        audit.STATUS_MAIN_CHAIN: set(),
        audit.STATUS_SUBGRAPH_ONLY: set(),
        audit.STATUS_STANDALONE_LEAF: set(),
    }
    for row in modules:
        assert row["status"] in buckets
        buckets[row["status"]].add(row["path"])

    assert len(buckets[audit.STATUS_MAIN_CHAIN]) == bs[audit.STATUS_MAIN_CHAIN]
    assert len(buckets[audit.STATUS_SUBGRAPH_ONLY]) == bs[audit.STATUS_SUBGRAPH_ONLY]
    assert len(buckets[audit.STATUS_STANDALONE_LEAF]) == bs[audit.STATUS_STANDALONE_LEAF]
    assert (
        buckets[audit.STATUS_MAIN_CHAIN]
        & buckets[audit.STATUS_SUBGRAPH_ONLY]
        & buckets[audit.STATUS_STANDALONE_LEAF]
    ) == set()
    assert len(
        buckets[audit.STATUS_MAIN_CHAIN]
        | buckets[audit.STATUS_SUBGRAPH_ONLY]
        | buckets[audit.STATUS_STANDALONE_LEAF]
    ) == s["non_init_modules"]

    for root in audit.MAIN_CHAIN_ROOTS:
        assert root in buckets[audit.STATUS_MAIN_CHAIN], f"missing root in main_chain: {root}"

    # 按层汇总守恒
    layer_sum = 0
    for counts in s["by_layer"].values():
        assert counts["total"] == (
            counts[audit.STATUS_MAIN_CHAIN]
            + counts[audit.STATUS_SUBGRAPH_ONLY]
            + counts[audit.STATUS_STANDALONE_LEAF]
        )
        layer_sum += counts["total"]
    assert layer_sum == s["non_init_modules"]


def test_json_output_deterministic_and_portable() -> None:
    tool_path = REPO_ROOT / "tools" / "module_topology_audit.py"
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    with tempfile.TemporaryDirectory() as tmp_cwd:
        p1 = subprocess.run(
            [sys.executable, "-X", "utf8", str(tool_path), "--repo-root", str(REPO_ROOT), "--json"],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        p2 = subprocess.run(
            [sys.executable, "-X", "utf8", str(tool_path), "--repo-root", str(REPO_ROOT), "--json"],
            cwd=tmp_cwd,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
    assert p1.returncode == 0, p1.stderr
    assert p2.returncode == 0, p2.stderr
    assert p1.stdout == p2.stdout, "--json output differed across working directories"

    payload = json.loads(p1.stdout)
    assert payload["version"] == audit.AUDIT_VERSION
    assert payload["summary"]["non_init_modules"] == len(payload["modules"])

    # 不应包含绝对盘符路径或本机工作目录片段
    forbidden_tokens = (
        "work" + "trees",
        "Users" + os.sep,
        "Users/",
        "anti" + "gravity",
    )
    ledger_text = (REPO_ROOT / audit.DEFAULT_LEDGER_REL).read_text(encoding="utf-8")
    for corpus_name, corpus in (("json", p1.stdout), ("ledger", ledger_text)):
        for tok in forbidden_tokens:
            assert tok not in corpus, f"local path token {tok!r} leaked in {corpus_name}"


def run_drift_mutations(verbose: bool = False) -> List[str]:
    """在隔离临时仓副本上执行 4 类定点漂移自证，返回日志行列表。"""
    logs: List[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        copy_root = _make_repo_copy(Path(tmp) / "repo")

        # 0. 基线必绿
        rc0, msg0, _ = audit.run(["--repo-root", str(copy_root), "--check"])
        assert rc0 == 0, f"baseline copy check failed:\n{msg0}"
        logs.append(f"[MUTATE] baseline_rc={rc0} ({msg0.strip()})")

        # 1. 新增 import 边：将 standalone_leaf 模块 lingshu/core/activation.py 接入 core.py
        core_path = copy_root / "lingshu" / "core" / "core.py"
        core_orig = core_path.read_text(encoding="utf-8")
        try:
            core_path.write_text(
                core_orig + "\nfrom . import activation as _issue172_probe  # noqa: F401\n",
                encoding="utf-8",
            )
            rc1, msg1, data1 = audit.run(["--repo-root", str(copy_root), "--check"])
            assert rc1 == 1, "adding import edge must trigger --check failure"
            assert "lingshu/core/activation.py" in msg1, f"missing drifted module in:\n{msg1}"
            assert "lingshu/core/activation.py" in data1.get("drifted_modules", [])
            logs.append(f"[MUTATE] add_import_rc={rc1} drifted={data1.get('drifted_modules')}")
        finally:
            core_path.write_text(core_orig, encoding="utf-8")

        rc_r1, msg_r1, _ = audit.run(["--repo-root", str(copy_root), "--check"])
        assert rc_r1 == 0, f"restore after add_import failed:\n{msg_r1}"

        # 2. 删除既有 import 边：断开 verdict.py -> condition_normalize.py
        verdict_path = copy_root / "lingshu" / "core" / "verdict.py"
        verdict_orig = verdict_path.read_text(encoding="utf-8")
        target_line = "from .condition_normalize import INVALID, UNASSERTED"
        assert target_line in verdict_orig, "expected import line missing in verdict.py"
        try:
            verdict_path.write_text(
                verdict_orig.replace(target_line, "INVALID = object(); UNASSERTED = object()", 1),
                encoding="utf-8",
            )
            rc2, msg2, data2 = audit.run(["--repo-root", str(copy_root), "--check"])
            assert rc2 == 1, "removing import edge must trigger --check failure"
            assert "lingshu/core/condition_normalize.py" in msg2, f"missing drifted module in:\n{msg2}"
            assert "lingshu/core/verdict.py" in msg2, f"missing drifted importer in:\n{msg2}"
            logs.append(f"[MUTATE] remove_import_rc={rc2} drifted={data2.get('drifted_modules')}")
        finally:
            verdict_path.write_text(verdict_orig, encoding="utf-8")

        rc_r2, msg_r2, _ = audit.run(["--repo-root", str(copy_root), "--check"])
        assert rc_r2 == 0, f"restore after remove_import failed:\n{msg_r2}"

        # 3. 篡改台账状态行
        ledger_path = copy_root / audit.DEFAULT_LEDGER_REL
        ledger_orig = ledger_path.read_text(encoding="utf-8")
        tamper_from = "| `lingshu/core/activation.py` | `core` | `standalone_leaf`（独立原型叶模块） |"
        tamper_to = "| `lingshu/core/activation.py` | `core` | `main_chain`（核心主入口可达） |"
        assert tamper_from in ledger_orig
        try:
            ledger_path.write_text(ledger_orig.replace(tamper_from, tamper_to, 1), encoding="utf-8")
            rc3, msg3, data3 = audit.run(["--repo-root", str(copy_root), "--check"])
            assert rc3 == 1, "tampering ledger row must trigger --check failure"
            assert "lingshu/core/activation.py" in msg3
            logs.append(f"[MUTATE] tamper_ledger_rc={rc3} drifted={data3.get('drifted_modules')}")
        finally:
            ledger_path.write_text(ledger_orig, encoding="utf-8")

        # 4. 新增未入账模块
        probe_mod = copy_root / "lingshu" / "core" / "_unrecorded_probe.py"
        try:
            probe_mod.write_text('"""unrecorded probe"""\nVALUE = 1\n', encoding="utf-8")
            rc4, msg4, data4 = audit.run(["--repo-root", str(copy_root), "--check"])
            assert rc4 == 1, "adding unrecorded module must trigger --check failure"
            assert "lingshu/core/_unrecorded_probe.py" in msg4, f"missing new module in:\n{msg4}"
            assert "lingshu/core/_unrecorded_probe.py" in data4.get("drifted_modules", [])
            logs.append(f"[MUTATE] add_module_rc={rc4} drifted={data4.get('drifted_modules')}")
        finally:
            if probe_mod.exists():
                probe_mod.unlink()

        # 5. 并行 PR 解耦不变性：新增 tests/ 与 bench/ 引用不得触发 --check 漂移，
        #    同时 --markdown --with-test-refs 仍实时反映外部引用列。
        dummy_test = copy_root / "tests" / "test_parallel_pr_probe.py"
        dummy_bench = copy_root / "bench" / "parallel_pr_probe.py"
        try:
            dummy_test.write_text("import lingshu.core.activation\n", encoding="utf-8")
            dummy_bench.parent.mkdir(parents=True, exist_ok=True)
            dummy_bench.write_text("import lingshu.nn.hex_gen\n", encoding="utf-8")
            rc5, msg5, _ = audit.run(["--repo-root", str(copy_root), "--check"])
            assert rc5 == 0, f"adding tests/bench refs must NOT fail --check:\n{msg5}"
            rc_md, md_refs, _ = audit.run([
                "--repo-root", str(copy_root), "--markdown", "--with-test-refs"
            ])
            assert rc_md == 0
            assert "`tests/` 直接(传递)" in md_refs
            assert "`bench/` 直接(传递)" in md_refs
            logs.append(f"[MUTATE] add_tests_bench_rc={rc5} (check_green_with_parallel_refs=True)")
        finally:
            for p in (dummy_test, dummy_bench):
                if p.exists():
                    p.unlink()

        rc_final, msg_final, _ = audit.run(["--repo-root", str(copy_root), "--check"])
        assert rc_final == 0, f"final restore failed:\n{msg_final}"
        logs.append(f"[MUTATE] restored_rc={rc_final}")

    if verbose:
        for line in logs:
            print(line)
    return logs


def test_drift_detection_on_temp_repo_copy() -> None:
    logs = run_drift_mutations(verbose=False)
    assert len(logs) == 7


def test_with_test_refs_markdown_flag() -> None:
    rc_default, md_default, _ = audit.run(["--repo-root", str(REPO_ROOT), "--markdown"])
    rc_refs, md_refs, _ = audit.run([
        "--repo-root", str(REPO_ROOT), "--markdown", "--with-test-refs"
    ])
    assert rc_default == 0 and rc_refs == 0
    assert "`tests/` 直接(传递)" not in md_default
    assert "`tests/` 直接(传递)" in md_refs
    assert "`bench/` 直接(传递)" in md_refs


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Issue #172 模块连通性台账守卫测试")
    parser.add_argument("--mutate", action="store_true", help="运行并打印定点漂移变异自证结果")
    args = parser.parse_args(argv)

    test_ledger_check_passes_on_repo()
    test_conservation_and_disjoint_partition()
    test_json_output_deterministic_and_portable()
    test_with_test_refs_markdown_flag()
    logs = run_drift_mutations(verbose=args.mutate)
    print(f"OK: test_issue172_module_topology_audit passed (mutations={len(logs)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
