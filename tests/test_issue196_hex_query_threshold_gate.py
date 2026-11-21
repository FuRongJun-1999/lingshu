# -*- coding: utf-8 -*-
"""test_issue196_hex_query_threshold_gate · #196 census 第八批：hex_query.query_match 四态阈值有限性与防 bool 闸。

背景（issue #196）：
  `lingshu/nn/hex_query.py::query_match(detections, lib, th_accept=0.6, th_defer=0.3)`
  此前未校验阈值参数：`NaN` 比较恒为 `False`、`bool` 被当成 `0`/`1`、`±inf` 极值穿透，
  导致无匹配（score=0.0）可在 `th_defer=nan, th_accept=-inf` 下落入 `ACCEPT`，
  弱匹配（score=0.5）可在 `th_accept=False` 下被抬升为 `ACCEPT`，
  满分匹配（score=1.0）可在 `th_defer=inf` 下被降为 `BLINDSPOT`。

守卫内容：
  G1 `NaN` / `±inf` / `np.nan` / `np.inf` 作 `th_accept` 或 `th_defer` ⇒ `ValueError`；
  G2 `bool` / `np.bool_`（`True`/`False`）作 `th_accept` 或 `th_defer` ⇒ `ValueError`；
  G3 非数值（`str`/`None`）、越 `[0,1]`、`10**400` 溢出、倒挂 `th_defer > th_accept` ⇒ `ValueError`（含空检出/空库入口）；
  G4 合法默认/边界/NumPy 标量（`(0.6, 0.3)`、`(0.0, 0.0)`、`(1.0, 1.0)`、`(0.5, 0.5)`、`np.float64`）四态判定逐位正常；
  G5 `--mutate` 定点变异自证：临时副本摘除阈值校验 ⇒ 静默放行复现且守卫精确点红，还原后转绿。

运行：
  python -X utf8 -m pytest tests/test_issue196_hex_query_threshold_gate.py -v
  python -X utf8 tests/test_issue196_hex_query_threshold_gate.py
  python -X utf8 tests/test_issue196_hex_query_threshold_gate.py --mutate
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lingshu.nn.hex_query import ConnectionLibrary, query_match  # noqa: E402


def _make_lib() -> ConnectionLibrary:
    lib = ConnectionLibrary()
    lib.add_record("pair", [
        {"type": "circle", "zone": "top-left", "color": "red"},
        {"type": "circle", "zone": "bottom-right", "color": "blue"},
    ])
    return lib


_DET_FULL = [{"obj": "circle|red", "pos": "r0"}, {"obj": "circle|blue", "pos": "r8"}]
_DET_WEAK = [{"obj": "circle|red", "pos": "r0"}]
_DET_NONE = [{"obj": "stripe|red", "pos": "r4"}]


def _assert_raises_value_error(fn, *args, **kwargs):
    try:
        res = fn(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError(f"期望抛出 ValueError，实际返回 {res!r}")


def test_g1_nonfinite_thresholds_rejected():
    """G1：NaN / ±inf（含 numpy 非有限标量）传入 th_accept 或 th_defer 必抛 ValueError。"""
    lib = _make_lib()
    for bad in (float("nan"), float("inf"), float("-inf"), np.nan, np.inf, -np.inf):
        for dets in (_DET_FULL, _DET_WEAK, _DET_NONE, []):
            _assert_raises_value_error(query_match, dets, lib, th_accept=bad)
            _assert_raises_value_error(query_match, dets, lib, th_defer=bad)
    _assert_raises_value_error(query_match, _DET_NONE, lib, th_accept=float("-inf"), th_defer=float("nan"))


def test_g2_bool_thresholds_rejected():
    """G2：bool / np.bool_（True/False）传入 th_accept 或 th_defer 必抛 ValueError。"""
    lib = _make_lib()
    for bad in (True, False, np.bool_(True), np.bool_(False)):
        _assert_raises_value_error(query_match, _DET_WEAK, lib, th_accept=bad)
        _assert_raises_value_error(query_match, _DET_WEAK, lib, th_defer=bad)


def test_g3_non_numeric_out_of_range_and_inverted_thresholds_rejected():
    """G3：非数值、越 [0,1]、大整数溢出、倒挂 th_defer > th_accept 与空库入口均必抛 ValueError。"""
    lib = _make_lib()
    empty_lib = ConnectionLibrary()
    for bad in ("0.5", None, -0.01, 1.01, 10 ** 400):
        _assert_raises_value_error(query_match, _DET_WEAK, lib, th_accept=bad)
        _assert_raises_value_error(query_match, _DET_WEAK, lib, th_defer=bad)
    _assert_raises_value_error(query_match, _DET_WEAK, lib, th_accept=0.3, th_defer=0.6)
    _assert_raises_value_error(query_match, _DET_WEAK, empty_lib, th_accept=float("nan"))


def test_g4_valid_thresholds_and_four_state_verdicts_unchanged():
    """G4：合法默认/边界/NumPy 标量阈值保持四态判定完全正常。"""
    lib = _make_lib()
    assert query_match(_DET_FULL, lib)[0]["verdict"] == "ACCEPT"
    assert query_match(_DET_WEAK, lib)[0]["verdict"] == "DEFER"
    assert query_match(_DET_NONE, lib)[0]["verdict"] == "BLINDSPOT"

    # 边界与 NumPy 数值标量
    assert query_match(_DET_WEAK, lib, th_accept=0.5, th_defer=0.5)[0]["verdict"] == "ACCEPT"
    assert query_match(_DET_NONE, lib, th_accept=0.0, th_defer=0.0)[0]["verdict"] == "ACCEPT"
    assert query_match(_DET_WEAK, lib, th_accept=1.0, th_defer=1.0)[0]["verdict"] == "BLINDSPOT"
    assert query_match(
        _DET_WEAK, lib, th_accept=np.float64(0.6), th_defer=np.float64(0.3)
    )[0]["verdict"] == "DEFER"

    # REJECT（双胞胎配方 margin == 0 且高于 th_defer）
    twin_lib = ConnectionLibrary()
    twin_lib.add_record("a", [{"type": "circle", "zone": "*", "color": "red"}])
    twin_lib.add_record("b", [{"type": "circle", "zone": "*", "color": "red"}])
    assert query_match([{"obj": "circle|red", "pos": "r0"}], twin_lib)[0]["verdict"] == "REJECT"


def run_mutation_self_proof() -> int:
    """G5 `--mutate` 定点变异自证：临时摘除 query_match 阈值校验 ⇒ 守卫必红，还原后转绿。"""
    src_path = REPO_ROOT / "lingshu" / "nn" / "hex_query.py"
    orig_src = src_path.read_text(encoding="utf-8")

    probe_code = (
        "import sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from lingshu.nn.hex_query import ConnectionLibrary, query_match\n"
        "lib = ConnectionLibrary()\n"
        "lib.add_record('pair', [{'type': 'circle', 'zone': 'top-left', 'color': 'red'}, "
        "{'type': 'circle', 'zone': 'bottom-right', 'color': 'blue'}])\n"
        "for kw in ({'th_accept': float('nan')}, {'th_defer': float('nan')}, {'th_accept': False}):\n"
        "    try:\n"
        "        res = query_match([{'obj': 'circle|red', 'pos': 'r0'}], lib, **kw)\n"
        "    except ValueError:\n"
        "        continue\n"
        "    print('BYPASS:' + str(kw) + '->' + str(res[0]['verdict']))\n"
        "    raise SystemExit(1)\n"
        "print('ALL_BLOCKED')\n"
    )

    with tempfile.TemporaryDirectory() as tmp:
        pkg_root = Path(tmp) / "repo"
        (pkg_root / "lingshu").mkdir(parents=True)
        (pkg_root / "lingshu" / "__init__.py").write_text("", encoding="utf-8")
        shutil.copytree(
            REPO_ROOT / "lingshu" / "nn",
            pkg_root / "lingshu" / "nn",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        target = pkg_root / "lingshu" / "nn" / "hex_query.py"

        def _run_probe(label: str, expect_rc: int, expect_prefix: str) -> bool:
            proc = subprocess.run(
                [sys.executable, "-X", "utf8", "-c", probe_code, str(pkg_root)],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
            )
            out = (proc.stdout or "").strip()
            err = (proc.stderr or "").strip()
            ok = (proc.returncode == expect_rc and out.startswith(expect_prefix))
            print(f"{label} rc={proc.returncode} out={out!r} VERDICT={'PASS' if ok else 'FAIL'}")
            if not ok and err:
                print(f"  stderr={err}")
            return ok

        # M0：干净修复版
        target.write_text(orig_src, encoding="utf-8")
        m0 = _run_probe("M0_clean_gated", 0, "ALL_BLOCKED")

        # M1：定点变异（摘除 query_match 入口校验调用）
        mutated = re.sub(
            r"    th_accept = _validate_threshold[^\n]*\n"
            r"    th_defer = _validate_threshold[^\n]*\n"
            r"    if th_defer > th_accept:\n"
            r"(?:        [^\n]*\n)+\s*\)\n",
            "",
            orig_src,
            count=1,
        )
        assert mutated != orig_src, "变异未命中目标语句"
        target.write_text(mutated, encoding="utf-8")
        m1 = _run_probe("M1_strip_threshold_gate", 1, "BYPASS:")

        # M2：还原后转绿
        target.write_text(orig_src, encoding="utf-8")
        m2 = _run_probe("M2_restored_gated", 0, "ALL_BLOCKED")

        all_ok = m0 and m1 and m2
        print(f"OVERALL={'PASS' if all_ok else 'FAIL'} (repo_files_untouched=True)")
        return 0 if all_ok else 1


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--mutate" in args:
        return run_mutation_self_proof()
    tests = [
        test_g1_nonfinite_thresholds_rejected,
        test_g2_bool_thresholds_rejected,
        test_g3_non_numeric_out_of_range_and_inverted_thresholds_rejected,
        test_g4_valid_thresholds_and_four_state_verdicts_unchanged,
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
