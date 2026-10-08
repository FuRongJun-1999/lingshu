# -*- coding: utf-8 -*-
"""test_gate_coverage · 门禁覆盖集守卫（防「同族盲区」第四次出现）
============================================================================
背景（2026-10-08/09 实测）：
    tests/ 下同时存在两式测试件——
      · **脚本式**：模块顶层即执行 / 只有 main()，**无 `test_*` 函数**
        ⇒ `pytest --collect-only` 收 **0 件**（rc=5）。它们**只能**靠
        `.github/workflows/gate.yml` 的 `SCRIPT_TESTS` 逐件单跑才被门禁执行。
      · **pytest 式**：含 `test_*` 函数 ⇒ 被 `pytest tests/` 自动收集执行。
    当一件**既是**脚本式、又**没被**列入 SCRIPT_TESTS ⇒ **门禁两条腿都看不到它**，
    它绿 / 红都无人知——此「同族盲区」已出现三次（#69 / #26 / #145）。

判据（本件）：`tests/test_*.py` 里的**每一件**必须满足其一——
    (a) 出现在 gate.yml 的 `SCRIPT_TESTS` 里（走**脚本腿**）；或
    (b) `pytest --collect-only <file>` 收集数 > 0（走 **pytest 腿**）。
    二者皆非 ⇒ **判红**并点名该文件。

实现：
    · 从仓根读 gate.yml 的 `SCRIPT_TESTS`——它是 YAML 折叠标量（`>-`）⇒
      按空白分词即可（不引第三方 YAML，避免守卫自身新增依赖）。
    · 对**未列入**者逐件 `pytest --collect-only`（**只收集不执行**，短超时）。
    · 三档打印：走脚本腿 / 走 pytest 腿（附收集数）/ **都没走到**。

自身要求：本件**必须自身可被 pytest 收集**——否则它自己就落在盲区里
    （由 test_guard_itself_is_collectable 自证）。

运行：pytest tests/test_gate_coverage.py
      （或 python -X utf8 tests/test_gate_coverage.py 直接打报告）
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATE_YML = os.path.join(REPO, ".github", "workflows", "gate.yml")
TESTS_DIR = os.path.join(REPO, "tests")
SELF_REL = "tests/test_gate_coverage.py"
COLLECT_TIMEOUT = 90          # 秒；某件偏高（导入重）也不致误判为超时
_COUNT_RE = re.compile(r"(\d+)\s+tests?\s+collected")


def _script_tests_from_gate_yml():
    """读 gate.yml 的 SCRIPT_TESTS（YAML 折叠标量 ⇒ 换行折为空格，按空白分词）。

    只做本工件所需的极窄解析（`SCRIPT_TESTS:` 行 + 其后缩进块），不引 YAML。
    """
    with open(GATE_YML, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    entries = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if re.match(r"\s*SCRIPT_TESTS\s*:", line):
            tail = line.split(":", 1)[1].strip()
            # 折叠/块标量指示符（>-、|- …）不是值本身；同行的其余 token 才是值
            if tail and tail not in (">", ">-", "|", "|-", ">+", "|+"):
                entries.extend(tail.split())
            i += 1
            while i < n:
                nxt = lines[i]
                if nxt.strip() == "" or not nxt[:1].isspace():
                    break             # 缩进块到此结束（空行或回到 0 缩进）
                entries.extend(nxt.split())
                i += 1
            break
        i += 1
    return [e for e in entries if e.startswith("tests/")]


def _collect_count(rel_path):
    """`pytest --collect-only` 该件的收集数（**只收集不执行**）；短超时。

    返回 (count, diag)：count 为 None 表示超时；否则为收集数（0 = 没被收集）。
    """
    env = dict(os.environ)
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    cmd = [sys.executable, "-X", "utf8", "-m", "pytest",
           "--collect-only", "-q", os.path.join(REPO, rel_path)]
    try:
        proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                              encoding="utf-8", errors="replace",
                              timeout=COLLECT_TIMEOUT, env=env)
    except subprocess.TimeoutExpired:
        return None, "collect-only 超时（>%ss）" % COLLECT_TIMEOUT
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    hits = _COUNT_RE.findall(out)
    count = int(hits[-1]) if hits else 0     # 取末行汇总（防中间噪声行）
    return count, "rc=%d" % proc.returncode


def _test_files():
    return sorted(f for f in os.listdir(TESTS_DIR)
                  if f.startswith("test_") and f.endswith(".py"))


def _coverage_report():
    """三档归类 + 明细行（供各断言与打印共用）。"""
    script_tests = set(_script_tests_from_gate_yml())
    script_leg, pytest_leg, uncovered = [], [], []
    for f in _test_files():
        rel = "tests/" + f
        if rel in script_tests:
            script_leg.append(rel)
            continue
        count, diag = _collect_count(rel)
        if count is None:
            uncovered.append((rel, diag))
        elif count > 0:
            pytest_leg.append((rel, count))
        else:
            uncovered.append((rel, "collect-only 收 0 件（%s）" % diag))
    return {
        "script_tests": script_tests,
        "script_leg": script_leg,
        "pytest_leg": pytest_leg,
        "uncovered": uncovered,
    }


def _print_report(rep):
    print("=== gate 覆盖集判据：每件必须满足 (a) 列入 SCRIPT_TESTS 或 (b) pytest 可收集 ===")
    print("SCRIPT_TESTS（%d 件）= %s" % (len(rep["script_tests"]),
                                         sorted(rep["script_tests"])))
    print("--- (a) 走脚本腿（列在 SCRIPT_TESTS）%d 件 ---" % len(rep["script_leg"]))
    for rel in rep["script_leg"]:
        print("    [script] %s" % rel)
    print("--- (b) 走 pytest 腿（未被列入，但 pytest 可收集）%d 件 ---"
          % len(rep["pytest_leg"]))
    for rel, cnt in rep["pytest_leg"]:
        print("    [pytest] %-45s collected=%d" % (rel, cnt))
    print("--- (c) **都没走到**（盲区）%d 件 ---" % len(rep["uncovered"]))
    for rel, why in rep["uncovered"]:
        print("    [GAP]    %-45s %s" % (rel, why))


# ---------------------------------------------------------------- 量具自检

def test_gate_yml_script_tests_parsed():
    """量具自检①：必须真的从 gate.yml 读到 SCRIPT_TESTS，且所列文件都在场。"""
    script_tests = _script_tests_from_gate_yml()
    assert script_tests, (
        "没从 %s 读到任何 SCRIPT_TESTS 条目 ⇒ 解析器坏了，本判据此时**无判别力**"
        "（会静默通过）。" % os.path.relpath(GATE_YML, REPO))
    missing = [p for p in script_tests
               if not os.path.isfile(os.path.join(REPO, p))]
    assert not missing, (
        "SCRIPT_TESTS 里列了**不存在的文件**（脚本腿会直接失败，或等于白列）：\n"
        + "\n".join("  · %s" % m for m in missing))


def test_guard_itself_is_collectable():
    """量具自检②：本件自身必须被 pytest 收集——否则它自己也落在盲区里。"""
    count, diag = _collect_count(SELF_REL)
    assert count, (
        "本件 %s 没被 pytest 收集（%s）⇒ 它自己就落在它所防的盲区里，"
        "本判据形同虚设。" % (SELF_REL, diag))


# ---------------------------------------------------------------- 主判据

def test_every_test_file_is_covered():
    """主判据：tests/test_*.py 每一件必须走脚本腿或 pytest 腿；二者皆非 ⇒ 红。"""
    rep = _coverage_report()
    _print_report(rep)
    assert rep["script_leg"], "SCRIPT_TESTS 一件都没匹配上 tests/ 下的文件 ⇒ 解析/清单失配"
    assert rep["pytest_leg"], "没有任何一件走 pytest 腿 ⇒ 收集都没跑起来（量具失效）"
    if rep["uncovered"]:
        detail = "\n".join("    · %s   ← %s" % (rel, why)
                           for rel, why in rep["uncovered"])
        raise AssertionError(
            "以下 tests/test_*.py **门禁两条腿都没走到**（既是脚本式、又没列入 "
            "SCRIPT_TESTS，且 pytest 收集 0 件）——它们的绿 / 红无人知：\n"
            + detail
            + "\n\n修法：把它加进 .github/workflows/gate.yml 的 `SCRIPT_TESTS`"
            "（脚本式），或改成可被 pytest 收集（含 test_* 函数）。")
    # 契约：覆盖集 = SCRIPT_TESTS(∩tests) ∪ pytest 腿
    covered = set(rep["script_leg"]) | {rel for rel, _ in rep["pytest_leg"]}
    all_rel = {"tests/" + f for f in _test_files()}
    assert covered == all_rel, (
        "覆盖集与 tests/ 实况不符：未覆盖 %s" % sorted(all_rel - covered))


if __name__ == "__main__":
    _rep = _coverage_report()
    _print_report(_rep)
    sys.exit(1 if _rep["uncovered"] else 0)
