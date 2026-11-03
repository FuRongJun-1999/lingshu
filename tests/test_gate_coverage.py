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

判据（本件）：`tests/` 下（**递归**）属「测试件」的**每一件**必须满足其一——
    (a) 出现在 gate.yml 的 `SCRIPT_TESTS` 里（走**脚本腿**）；或
    (b) `pytest --collect-only <file>` 收集数 > 0（走 **pytest 腿**）。
    二者皆非 ⇒ **判红**并点名该文件。
    「测试件」= 行为判据三族之一（`_is_test_artifact`）：文件名符合 pytest 默认收集式、
    顶层定义了 `test_*` 函数、或含 `if __name__ == "__main__"` 块（脚本式）。
    #491：枚举面原为 `os.listdir(tests/) × test_*.py`（**顶层 + 按名**）⇒ 子目录 /
    异名的脚本式守卫会从 pytest 腿、脚本腿、本守卫**三个视野同时消失**。

实现：
    · 从仓根读 gate.yml 的 `SCRIPT_TESTS`——它是 YAML 折叠标量（`>-`）⇒
      按空白分词即可（不引第三方 YAML，避免守卫自身新增依赖）。
    · 对**未列入**者**一次**批量 `pytest --collect-only tests/`（#474：原实现逐件
      起子进程，207 次 spawn / 240.9 秒；批量后 1 次 / 约 3 秒，逐件收集数一致）。
      `--ignore` 掉 SCRIPT_TESTS（它们模块级 `sys.exit()`，混进同一次收集会把
      整个 pytest 进程带走），并加 `--continue-on-collection-errors` 防单件导入
      错误吞掉其余件的收集。批量失败（超时/解析不出）⇒ 自动回落逐件口径。
    · 三档打印：走脚本腿 / 走 pytest 腿（附收集数）/ **都没走到**。

自身要求：本件**必须自身可被 pytest 收集**——否则它自己就落在盲区里
    （由 test_guard_itself_is_collectable 自证）。

运行：pytest tests/test_gate_coverage.py
      （或 python -X utf8 tests/test_gate_coverage.py 直接打报告）
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATE_YML = os.path.join(REPO, ".github", "workflows", "gate.yml")
TESTS_DIR = os.path.join(REPO, "tests")
SELF_REL = "tests/test_gate_coverage.py"
COLLECT_TIMEOUT = 90          # 秒；某件偏高（导入重）也不致误判为超时（逐件兜底口径）
BATCH_TIMEOUT = 300           # 秒；一次批量收集的上界（#474：把「件数 × 90s」的无界上界
                              # 收窄为单次 300s —— 225 件时最坏 5.2 小时 → 5 分钟）
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


def _parse_collected_counts(text):
    """从 `pytest --collect-only -q tests/` 的输出里数出**逐件**收集数。

    只认 node id 行（`tests/<路径>.py::...`）——汇总行（`NNN tests collected`）
    与 `ERROR ...` 行都不含 `::`，自然不计入；参数化用例的每个 node id 各计一次，
    与逐件 `--collect-only <file>` 的汇总口径一致（#474 的对账组：207 件全等）。
    """
    counts = {}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("tests/") and "::" in line:
            rel = line.split("::", 1)[0]
            counts[rel] = counts.get(rel, 0) + 1
    return counts


def _collect_counts_batch(script_tests):
    """**一次** `pytest --collect-only` 取全仓逐件收集数（#474）。

    镜像 gate.yml 的 pytest 腿：`--ignore` 掉 `SCRIPT_TESTS`——那些件模块级即
    `sys.exit()`，混进同一次收集会把整个 pytest 进程带走。`--continue-on-collection-
    errors` 让单件导入错误只把该件收成 0 件（判红），不连坐其余件。

    返回 (counts | None, diag)：None 表示批量不可用（超时），调用方回落逐件口径。
    """
    env = dict(os.environ)
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    cmd = [sys.executable, "-X", "utf8", "-m", "pytest",
           "--collect-only", "-q", TESTS_DIR,
           "--continue-on-collection-errors"]
    for rel in sorted(script_tests):
        cmd.append("--ignore=" + os.path.join(REPO, rel))
    try:
        proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                              encoding="utf-8", errors="replace",
                              timeout=BATCH_TIMEOUT, env=env)
    except subprocess.TimeoutExpired:
        return None, "batch collect-only 超时（>%ss）" % BATCH_TIMEOUT
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return _parse_collected_counts(out), "batch rc=%d" % proc.returncode


_PYTEST_FILE_RE = re.compile(r"^(?:test_.+|.+_test)\.py$")


def _is_test_artifact(path):
    """行为判据（#491）：该 `.py` 是否属「门禁两条腿之一应当走到」的**测试件**。

    三族之一即算——
      (a) 文件名符合 pytest 默认收集式（`test_*.py` / `*_test.py`）⇒ pytest 腿候选；
      (b) 顶层定义了 `test_*` 函数（或 `Test*` 类内的 `test_*` 方法）⇒ pytest 腿候选；
      (c) 含 `if __name__ == "__main__"` 块 ⇒ **脚本式**，只能靠 SCRIPT_TESTS 走脚本腿。
    三族皆非 ⇒ 合法辅助模块（如 `tests/_helpers.py`、`conftest.py`），不在本件视野内。
    解析不了（SyntaxError）⇒ 收进视野（宁可报出、不静默失明）。
    """
    path = os.fspath(path)
    if _PYTEST_FILE_RE.match(os.path.basename(path)):
        return True
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            tree = ast.parse(fh.read(), filename=str(path))
    except SyntaxError:
        return True
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "__main__" in ast.unparse(node.test):
            return True
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_"):
                return True
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            if any(isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and m.name.startswith("test_") for m in node.body):
                return True
    return False


def _test_files():
    """枚举 `tests/` 下**递归**的测试件（仓内相对 POSIX 路径，已排序）。

    #491：原实现是 `os.listdir(TESTS_DIR)` + `startswith("test_")` —— **顶层 + 按名**
    ⇒ 子目录 / 异名脚本式守卫会从 pytest 腿、脚本腿、本守卫**三个视野同时消失**。
    """
    root = os.path.dirname(os.path.normpath(TESTS_DIR))
    found = []
    for dirpath, dirnames, filenames in os.walk(TESTS_DIR):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for name in sorted(filenames):
            if not name.endswith(".py"):
                continue
            full = os.path.join(dirpath, name)
            if _is_test_artifact(full):
                found.append(os.path.relpath(full, root).replace(os.sep, "/"))
    return sorted(found)


def _coverage_report():
    """三档归类 + 明细行（供各断言与打印共用）。"""
    script_tests = set(_script_tests_from_gate_yml())
    script_leg, pytest_leg, uncovered = [], [], []
    need = []
    for rel in _test_files():
        if rel in script_tests:
            script_leg.append(rel)
        else:
            need.append(rel)
    # #474：一次批量收集替代逐件 spawn（207 次 / 240.9s → 1 次 / ~3s）。
    counts, batch_diag = _collect_counts_batch(script_tests)
    for rel in need:
        if counts is None:
            # 批量不可用（超时）⇒ 回落逐件口径，判据不降级。
            count, diag = _collect_count(rel)
        else:
            count, diag = counts.get(rel, 0), batch_diag
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


def test_batch_parser_counts_per_file():
    """量具自检④（#474）：批量输出的逐件计数解析（纯函数，不起子进程）。

    只认 `tests/<路径>.py::...` 的 node id 行；汇总行与 ERROR 行不得计入；
    参数化用例的每个 node id 各计一次（与逐件 `--collect-only` 的汇总口径一致）。
    """
    text = (
        "tests/test_a.py::test_one\n"
        "tests/test_a.py::test_two\n"
        "tests/sub/test_b.py::TestX::test_three\n"
        "tests/test_a.py::test_param[case-1]\n"
        "\n"
        "4 tests collected in 0.02s\n"
        "ERROR tests/test_broken.py - ImportError while importing test module\n"
    )
    counts = _parse_collected_counts(text)
    assert counts == {"tests/test_a.py": 3, "tests/sub/test_b.py": 1}, (
        "批量计数解析错：%r ⇒ 覆盖表会与逐件口径不一致（#474 的对账前提）" % (counts,))


def test_batch_and_per_file_counts_agree_on_sample():
    """量具自检⑤（#474 对账组）：真实仓上批量口径与逐件口径必须同数。

    取前三件做抽检（控制成本）：批量若少收/多收，这里直接点名。
    """
    script_tests = set(_script_tests_from_gate_yml())
    need = [rel for rel in _test_files() if rel not in script_tests]
    assert need, "没有需要收集的件 ⇒ 量具失效"
    counts, diag = _collect_counts_batch(script_tests)
    assert counts is not None, diag
    for rel in need[:3]:
        per_file, pdiag = _collect_count(rel)
        assert counts.get(rel, 0) == per_file, (
            "批量与逐件收集数不一致：%s 批量=%r 逐件=%r（%s / %s）"
            % (rel, counts.get(rel, 0), per_file, diag, pdiag))


def test_enumeration_face_is_recursive_and_behavior_based(tmp_path, monkeypatch):
    """量具自检③（#491）：枚举面必须**递归**且按**行为**判件，否则三视野同时失明。

    合成四件在临时树里实算（**不写工作树**）：
      · `tests/sub/test_script_style_guard.py` —— 脚本式 + 子目录 ⇒ 必须枚举到；
      · `tests/check_named_guard.py`        —— 脚本式 + 异名顶层 ⇒ 必须枚举到；
      · `tests/named_pytest_style_test.py`  —— 异名但合 pytest 默认收集式 ⇒ 必须枚举到；
      · `tests/helper.py`                   —— 合法辅助模块 ⇒ **不得**被枚举（防误报）。
    若有人把 `_test_files()` 改回「`os.listdir` + `test_` 前缀」，本条即变红。
    """
    script_src = ('def main():\n'
                  '    assert 1 + 1 == 2\n'
                  '\n'
                  'if __name__ == "__main__":\n'
                  '    main()\n')
    fake = tmp_path / "tests"
    (fake / "sub").mkdir(parents=True)
    (fake / "sub" / "test_script_style_guard.py").write_text(script_src, encoding="utf-8")
    (fake / "check_named_guard.py").write_text(script_src, encoding="utf-8")
    (fake / "named_pytest_style_test.py").write_text(
        "def test_ok():\n    assert True\n", encoding="utf-8")
    (fake / "helper.py").write_text("def helper():\n    return 42\n", encoding="utf-8")

    monkeypatch.setattr(sys.modules[__name__], "TESTS_DIR", str(fake))
    found = _test_files()
    for want in ("tests/sub/test_script_style_guard.py",
                 "tests/check_named_guard.py",
                 "tests/named_pytest_style_test.py"):
        assert want in found, (
            "枚举面漏掉 %s（实测 %s）⇒ 该件会从 pytest 腿 / 脚本腿 / 本守卫"
            "三个视野同时消失（#491 的机制）。" % (want, found))
    assert "tests/helper.py" not in found, (
        "枚举面把合法辅助模块 tests/helper.py 也收进来了（%s）⇒ 会制造误报，"
        "判据应收窄到「文件名符合收集式 ∨ 顶层 test_* 函数 ∨ 有 __main__ 块」。" % found)


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
    all_rel = set(_test_files())
    assert covered == all_rel, (
        "覆盖集与 tests/ 实况不符：未覆盖 %s" % sorted(all_rel - covered))


# ---------------------------------------------------------------- 名单基线

#: 守卫名单基线（#2294）：tests/ 下全部测试件的**冻结清单**（仓内相对 POSIX 路径，
#: 每行一条，`#` 开头为注释）。
#:
#: 本守卫的三视野枚举源与被检面**同源** ⇒ 文件级消失（PR 误删 / git rm / 重构
#: 挪出 tests/）对枚举不可见，守卫面可静默缩水（#2294 实测：把一件守卫 mv 到
#: scripts/ 后本守卫仍 rc=0、盲区 0 件——文件消失这件「事」三个视野都看不见）。
#: 基线反着钉：**清单里每一件都必须仍被枚举为测试件**，缺员即红。
#: 纪律（只增不减，同 time_core_lint_baseline 手法）：新增测试件时追加进基线；
#: 删除基线条目必须是有意的可审 diff（有意删除 ⇒ 同步改基线；误删 ⇒ 恢复文件）。
BASELINE_REL = "tests/gate_baseline.txt"


def _read_baseline():
    """读基线名册；文件缺失 / 为空 / 有重复条目一律 fail-closed。"""
    path = os.path.join(REPO, BASELINE_REL)
    if not os.path.isfile(path):
        raise AssertionError(
            "守卫名单基线 %s 缺失 ⇒ #2294 判据失效（fail-closed）。"
            "重新生成：把 tests/ 下全部测试件的仓内相对路径写入该文件"
            "（每行一条）。" % BASELINE_REL)
    with open(path, encoding="utf-8") as fh:
        names = [ln.strip() for ln in fh
                 if ln.strip() and not ln.strip().startswith("#")]
    assert names, "基线为空 ⇒ 判据无判别力（fail-closed）"
    dupes = sorted({n for n in names if names.count(n) > 1})
    assert not dupes, "基线含重复条目：%r" % (dupes,)
    return names


def test_guard_roster_baseline_no_member_disappears():
    """#2294：基线名册每一件必须仍被枚举为测试件——缺员即红。

    「枚举源 = 被检面」的结构性自指 ⇒ 文件级消失对三视野不可见；本判据把
    枚举结果与冻结名册比对，任何一件缺席都点名。
    """
    names = _read_baseline()
    actual = set(_test_files())
    missing = sorted(n for n in names if n not in actual)
    assert not missing, (
        "守卫名单基线缺员 %d 件（文件级消失对三视野枚举不可见，#2294）：\n"
        "%s\n若是有意删除/挪动：同步更新 %s（只增不减纪律）；若是误删：恢复文件。"
        % (len(missing), "\n".join("  · %s" % m for m in missing), BASELINE_REL))


if __name__ == "__main__":
    _rep = _coverage_report()
    _print_report(_rep)
    _rc = 1 if _rep["uncovered"] else 0
    try:
        test_guard_roster_baseline_no_member_disappears()
    except AssertionError as _ex:
        print("[GAP] %s" % _ex)
        _rc = 1
    sys.exit(_rc)
