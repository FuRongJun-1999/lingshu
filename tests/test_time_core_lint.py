# -*- coding: utf-8 -*-
"""PR-2 · `tools/time_core_lint.py` 的验收测试

覆盖 PR-SPEC「PR-2 / 文件 C」的 6 条冻结断言，另加**判定豁免回归**：

1. R1 合成违规          → 被检出、退出码 1
2. R2/R3 合成违规       → 被检出
3. 只含四类豁免形态     → 0 检出
4. 真实仓库（默认 baseline）→ 退出码 0、`new_violations == 0`
5. 空 baseline 跑真实仓库   → 退出码 1 且命中 `lingshu/world/stable_lease.py:67:R1`
6. `--json` 两次运行逐字节相同（另：换 cwd 亦逐字节相同）
7+. 修正轮（task-11）追加的回归：
   · EX5 是**位置级**豁免：`x = cred_step(a,b) + x * (1 - gamma)` 必须命中（同行混写），
     两行写法同样命中，而 `cred_blend(..., retain=1.0 - decay)` 仍豁免
   · 裸 `keep` 计入保持率词表（`x = x * keep` 必须命中）
   · 同一物理行 `;` 分隔的语句分别判定（`a = a*(1-pc); b = z*pc` 必须命中，行号仍精确）
   · R3 数值形状只认 `0 < literal < 1`（`x = 2.0*x` 不再误报，`v = 0.9*v + 0.1*g` 仍报）
   · 判定回归带**源行守卫**（`EXPECTED_MARKERS`），行号漂移会当场报错而非恒真
   · 同一语句 R2+R3 ⇒ 输出 2 行、baseline 需 2 行
   · 显式 `--baseline FILE --update-baseline` 指向不存在文件 ⇒ 退出 2 且不创建
   · 退出码 2、baseline 注册、掩码、规范模块豁免、`;` 切分行号回归

本文件只读仓库源码，只写 `tmp_path`；不改 `lingshu/` 下任何文件。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "tools" / "time_core_lint.py"
BASELINE = REPO_ROOT / "tools" / "time_core_lint_baseline.txt"

if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

import time_core_lint as T  # noqa: E402


# --------------------------------------------------------------------- 工具函数
def run_lint(*args, cwd=None, timeout=300):
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "-X", "utf8", str(TOOL), *args],
        cwd=str(cwd or REPO_ROOT), capture_output=True, timeout=timeout, env=env)


def payload(res):
    assert res.stdout, "工具没有输出：stderr=%r" % res.stderr[:2000]
    return json.loads(res.stdout.decode("utf-8"))


def hit_set(data):
    return {(h["file"], h["line"], h["rule"]) for h in data["new_violations"]}


def write_module(directory: Path, name: str, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(body, encoding="utf-8")
    return path


@pytest.fixture
def empty_baseline(tmp_path):
    path = tmp_path / "empty_baseline.txt"
    path.write_text("", encoding="utf-8")
    return path


@pytest.fixture
def lint_dir(tmp_path, empty_baseline):
    """返回闭包：写一段合成源码到**独立目录**并跑工具（每次调用一个新目录，互不污染）。"""
    counter = {"n": 0}

    def _lint(body: str, name: str = "sample.py"):
        counter["n"] += 1
        directory = tmp_path / ("pkg%d" % counter["n"])
        write_module(directory, name, body)
        return run_lint(str(directory), "--baseline", str(empty_baseline), "--json")

    return _lint


# ------------------------------------------------------- 1. R1 / R2 / R3 合成检出
def test_r1_exp_decay_detected(lint_dir):
    body = (
        "import math\n"
        "import numpy as np\n"
        "\n"
        "\n"
        "def decayed(conf, gamma, age, tau):\n"
        "    a = conf * math.exp(-gamma * age)\n"
        "    b = np.exp(-age / tau)\n"
        "    import math as _m\n"
        "    c = _m.exp(-tau)\n"
        "    return a + b + c\n"
    )
    res = lint_dir(body)
    data = payload(res)
    assert res.returncode == 1
    rules = {h["rule"] for h in data["new_violations"]}
    assert rules == {"R1"}, data["new_violations"]
    lines = sorted(h["line"] for h in data["new_violations"])
    assert lines == [6, 7, 9], data["new_violations"]
    assert data["summary"]["new"] == 3
    assert all(h["baseline"] is False for h in data["new_violations"])


@pytest.mark.parametrize("body,expect_lines", [
    # R2(a) 乘性保持率改写
    ("def f(x, factor):\n    return x * (1 - factor)\n", [2]),
    # R2(b) 保持率命名赋值
    ("def f(factor):\n    retain = 1.0 - factor\n    return retain\n", [2]),
    # R2(c) 括号包裹的独立保持率表达式
    ("def f(decay):\n    e = (1.0 - decay)\n    return e\n", [2]),
])
def test_r2_keep_rate_detected(lint_dir, body, expect_lines):
    res = lint_dir(body)
    data = payload(res)
    assert res.returncode == 1
    assert [(h["line"], h["rule"]) for h in data["new_violations"]] == \
           [(ln, "R2") for ln in expect_lines], data["new_violations"]


@pytest.mark.parametrize("body,expect_rule", [
    # R3(a) 自指乘性更新
    ("def f(x, retain_gain):\n    x = x * retain_gain\n    return x\n", "R3"),
    # R3(a) 保持率形状在右侧
    ("def f(x, keep_rate, y):\n    x = (1 - keep_rate) * x + keep_rate * y\n    return x\n", "R3"),
    # R3(b) 增量乘性赋值
    ("def f(x, momentum):\n    x *= momentum\n    return x\n", "R3"),
    # R3(c) 赋值里出现 EMA/保持率命名的乘性因子
    ("def f(x, y, decay_rate):\n    z = y * decay_rate\n    return z\n", "R3"),
])
def test_r3_ema_detected(lint_dir, body, expect_rule):
    res = lint_dir(body)
    data = payload(res)
    assert res.returncode == 1
    assert data["new_violations"], "R3 未被检出：%r" % body
    assert {h["rule"] for h in data["new_violations"]} == {expect_rule}


# ------------------------------------------------------------- 3. 四类豁免形态
EXEMPT_KITCHEN_SINK = '''# -*- coding: utf-8 -*-
"""四类非衰减用法的合成样例（PR-SPEC R2 豁免表）。"""
import math

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_plain(x):
    return 1 / (1 + math.exp(-x))


def softmax_oneline(z):
    return np.exp(-z) / np.exp(-z).sum()


def softmax_two_statement(z):
    e = np.exp(-z)
    return e / e.sum()


def bilinear(mm, y0, x0, fy, fx):
    v = (mm[y0, x0] * (1 - fy) * (1 - fx) + mm[y0 + 1, x0] * fy * (1 - fx)
         + mm[y0, x0 + 1] * (1 - fy) * fx + mm[y0 + 1, x0 + 1] * fy * fx)
    return v


def alpha_composite(obj, dark, pc):
    obj = obj * (1.0 - pc) + dark * pc
    return obj


def bernoulli_stderr(cer, n):
    return math.sqrt(cer * (1 - cer) / n)
'''


def test_exemptions_produce_zero_hits(lint_dir):
    res = lint_dir(EXEMPT_KITCHEN_SINK)
    data = payload(res)
    assert data["new_violations"] == [], data["new_violations"]
    assert data["summary"]["total"] == 0
    assert res.returncode == 0


# ------------------------------------------- 判定边界：不属违规的形态一律不报
JUDGEMENT_BOUNDARY = '''"""判定边界样例：全部判为「非违规」（见 lint 头部 L3/L8）。"""
import math


def taper(tail_w, i):
    w0 = tail_w * (1 - i * 0.18)
    w1 = tail_w * (1 - (i + 1) * 0.18)
    return w0, w1


def safety_line(rel, TOL, d_obj):
    return min(rel, max(0.0, 2.0 * (1.0 - 1.5 * TOL / d_obj)))


def implicit_rate(x):
    return x * (1 - 0.02)


def growth_gate(d_before, d_after, retain_gain=0.98):
    if d_after <= d_before * retain_gain:
        return True
    return False


def evidence_weight(alpha):
    evidence = 1.0 - alpha
    return evidence


def orthogonalize(t, w):
    t = (t - float((t * w).sum())).astype(float)
    return t


def rotation(x, z, cos_y, sin_y):
    x, z = x * cos_y + z * sin_y, -x * sin_y + z * cos_y
    return x, z


def explained_variance(r, norm0):
    return round(1.0 - (float(r) / norm0) ** 2, 4)
'''


def test_judgement_boundary_produces_zero_hits(lint_dir):
    res = lint_dir(JUDGEMENT_BOUNDARY)
    data = payload(res)
    assert data["new_violations"] == [], data["new_violations"]
    assert res.returncode == 0


def test_canonical_call_through_exempt(lint_dir):
    """`retain=1.0 - decay` 交给规范核 = 核的既定参数形态，不是同核重写（EX5）。"""
    body = (
        "from lingshu.world.time_core import cred_blend\n"
        "\n"
        "\n"
        "def update(last, growth, decay):\n"
        "    return max(0.0, cred_blend(last, growth, retain=1.0 - decay))\n"
    )
    res = lint_dir(body)
    data = payload(res)
    assert data["new_violations"] == [], data["new_violations"]


# ------------------------------------------- EX5 是位置级豁免（回归：曾整句洗白）
def test_ex5_scope_same_statement_multiply_is_reported(lint_dir):
    """★ 回归：同一语句里 `cred_step(...)` 之外手写衰减核，必须照样报红。"""
    body = ("def f(x, a, b, gamma):\n"
            "    x = cred_step(a, b) + x * (1 - gamma)\n"
            "    return x\n")
    data = payload(lint_dir(body))
    assert {(h["line"], h["rule"]) for h in data["new_violations"]} == {(2, "R2"), (2, "R3")}


def test_ex5_scope_same_statement_keep_rate_is_reported(lint_dir):
    body = ("def f(x, a, b, keep_rate):\n"
            "    x = cred_step(a, b) + x * keep_rate\n"
            "    return x\n")
    data = payload(lint_dir(body))
    assert {(h["line"], h["rule"]) for h in data["new_violations"]} == {(2, "R3")}


def test_ex5_scope_split_statements_are_reported(lint_dir):
    """两行写法（核调用与手写核分属不同语句）同样必须命中。"""
    body = ("def f(x, a, b, gamma):\n"
            "    x = x * (1 - gamma)\n"
            "    y = cred_step(a, b)\n"
            "    return x + y\n")
    data = payload(lint_dir(body))
    assert {(h["line"], h["rule"]) for h in data["new_violations"]} == {(2, "R2"), (2, "R3")}


def test_canon_arg_scope_helper():
    """EX5 判定的最小单元：实参内 True、实参外 False、未给位置 False（失败朝报红）。"""
    stmt = "x = cred_blend(last, growth, retain=1.0 - decay) + x * (1 - keep)"
    assert T._in_canon_args(stmt, stmt.index("retain")) is True
    assert T._in_canon_args(stmt, stmt.index("x * (1 - keep)")) is False
    assert T._in_canon_args(stmt, -1) is False
    span = T._canon_arg_spans(stmt)[0]
    assert span[0] < stmt.index("retain") < span[1]


# ------------------------------------------------------- 裸 keep / `;` / 数值形状
def test_bare_keep_is_a_keep_rate_name(lint_dir):
    """裸 `keep` 与 `keep_rate` 同权：不能出现 `x*(1-keep)` 报而 `x*keep` 不报。"""
    body = ("def f(x, keep):\n"
            "    x = x * keep\n"
            "    return x\n"
            "\n"
            "\n"
            "def g(factor):\n"
            "    keep = 1.0 - factor\n"
            "    return keep\n")
    data = payload(lint_dir(body))
    assert {(h["line"], h["rule"]) for h in data["new_violations"]} == {(2, "R3"), (7, "R2")}


def test_semicolon_separated_statements_judged_separately(lint_dir):
    """★ 回归：同一物理行 `;` 分隔的两条语句分别判定（不再被 EX4 整体洗白）。"""
    same_line = ("def f(a, z, pc):\n"
                 "    a = a * (1 - pc); b = z * pc\n"
                 "    return a, b\n")
    split_lines = ("def f(a, z, pc):\n"
                   "    a = a * (1 - pc)\n"
                   "    b = z * pc\n"
                   "    return a, b\n")
    one = payload(lint_dir(same_line, name="one.py"))
    two = payload(lint_dir(split_lines, name="two.py"))
    assert {(h["line"], h["rule"]) for h in one["new_violations"]} == {(2, "R2"), (2, "R3")}
    assert {(h["line"], h["rule"]) for h in two["new_violations"]} == {(2, "R2"), (2, "R3")}


def test_semicolon_split_keeps_line_numbers(lint_dir):
    """`;` 切分后行号仍精确（回归：曾把整文件片段记成第 1 行）。"""
    body = ("".join("# 填充 %d\n" % i for i in range(5))
            + "a = a * (1 - pc); b = z * pc\n"
            + "c = 1\n"
            + "d = d * (1 - f)\n")
    data = payload(lint_dir(body))
    assert sorted(h["line"] for h in data["new_violations"]) == [6, 6, 8, 8]
    assert {(h["line"], h["rule"]) for h in data["new_violations"]} == \
           {(6, "R2"), (6, "R3"), (8, "R2"), (8, "R3")}


def test_alpha_composite_self_update_still_exempt(lint_dir):
    """真 alpha 合成（自指形态）在 `;` 收紧后仍不误报。"""
    body = ("def composite(obj, dark, pc):\n"
            "    obj = obj * (1.0 - pc) + dark * pc\n"
            "    return obj\n")
    data = payload(lint_dir(body))
    assert data["new_violations"] == [], data["new_violations"]


def test_numeric_keep_shape_only_below_one(lint_dir):
    """R3 的数值形状只认 0<literal<1：`2.0*x` 不报，`0.9*v+0.1*g` 与 `w*0.5` 报。"""
    amplified = ("def f(x):\n"
                 "    x = 2.0 * x\n"
                 "    y = x * 1.5\n"
                 "    z = x * 100\n"
                 "    k = 1.0 * x\n"
                 "    return x, y, z, k\n")
    data = payload(lint_dir(amplified, name="amp.py"))
    assert data["new_violations"] == [], data["new_violations"]

    decayed = ("def f(v, g, w, u):\n"
               "    v = 0.9 * v + 0.1 * g\n"
               "    w = w * 0.5\n"
               "    u = .5 * u\n"
               "    return v, w, u\n")
    data2 = payload(lint_dir(decayed, name="dec.py"))
    assert {(h["line"], h["rule"]) for h in data2["new_violations"]} == \
           {(2, "R3"), (3, "R3"), (4, "R3")}


def test_same_line_r2_r3_need_two_baseline_lines(tmp_path):
    """同一语句命中 R2+R3 输出 2 行 ⇒ baseline 也要 2 行（docstring 运维须知 ②）。"""
    root = tmp_path / "repo"
    write_module(root / "lingshu" / "pkg", "mod.py",
                 "def f(x, decay_rate):\n"
                 "    x = x * (1 - decay_rate)\n"
                 "    return x\n")
    baseline = root / "tools" / "time_core_lint_baseline.txt"

    data = payload(run_lint(str(root / "lingshu"), "--repo-root", str(root), "--json"))
    assert {(h["line"], h["rule"]) for h in data["new_violations"]} == {(2, "R2"), (2, "R3")}

    baseline.parent.mkdir(parents=True, exist_ok=True)
    baseline.write_text("lingshu/pkg/mod.py:2:R2\n", encoding="utf-8")
    partial = payload(run_lint(str(root / "lingshu"), "--repo-root", str(root), "--json"))
    assert {(h["line"], h["rule"]) for h in partial["new_violations"]} == {(2, "R3")}

    baseline.write_text("lingshu/pkg/mod.py:2:R2\nlingshu/pkg/mod.py:2:R3\n", encoding="utf-8")
    assert run_lint(str(root / "lingshu"), "--repo-root", str(root)).returncode == 0


def test_update_baseline_refuses_to_create_explicit_missing_file(tmp_path):
    """显式 `--baseline FILE --update-baseline` 而 FILE 不存在 → rc=2 且不创建（防误写）。"""
    target = tmp_path / "not_yet.txt"
    res = run_lint("--baseline", str(target), "--update-baseline")
    assert res.returncode == 2
    assert not target.exists()
    assert "不存在" in res.stdout.decode("utf-8")


def test_comments_and_docstrings_are_masked(lint_dir):
    body = (
        '"""文档示例：exp(-t/tau) 与 ×(1-factor)、EMA 保持率都是违规形态。"""\n'
        "# 注释里的 exp(-gamma * dt) 与 retain = 1 - decay 也不该命中\n"
        "X = 1  # exp(-x)\n"
    )
    res = lint_dir(body)
    data = payload(res)
    assert data["new_violations"] == [], data["new_violations"]


def test_canonical_modules_fully_exempt(tmp_path):
    """规范模块自身（唯一定义处）整文件豁免（EX0）。"""
    root = tmp_path / "repo"
    for rel in ("lingshu/core/time_core.py", "lingshu/world/time_core.py"):
        write_module(root / Path(rel).parent, Path(rel).name,
                     "import math\n\n\n"
                     "def cred_factor(gamma, dt):\n"
                     "    return math.exp(-gamma * dt)\n\n\n"
                     "def cred_step(x, factor):\n"
                     "    return x * (1.0 - factor)\n")
    empty = tmp_path / "e.txt"
    empty.write_text("", encoding="utf-8")
    res = run_lint(str(root / "lingshu"), "--baseline", str(empty),
                   "--repo-root", str(root), "--json")
    data = payload(res)
    assert data["new_violations"] == [], data["new_violations"]
    assert data["files_scanned"] == 0
    assert res.returncode == 0


# ------------------------------------------------- 4/5. 真实仓库：默认与空 baseline
def test_real_repo_default_baseline_exit_zero():
    res = run_lint("--json")
    data = payload(res)
    assert res.returncode == 0, data
    assert data["new_violations"] == []
    assert data["summary"]["new"] == 0
    assert data["summary"]["total"] >= 1, "baseline 里没有任何真实命中，规则可能失效"
    registered = {(h["file"], h["line"], h["rule"]) for h in data["registered_violations"]}
    assert ("lingshu/world/stable_lease.py", 67, "R1") in registered
    assert data["stale_baseline_entries"] == []


def test_real_repo_empty_baseline_exit_one_hits_stable_lease(empty_baseline):
    res = run_lint("--baseline", str(empty_baseline), "--json")
    data = payload(res)
    assert res.returncode == 1
    assert data["summary"]["new"] >= 1
    hits = hit_set(data)
    assert ("lingshu/world/stable_lease.py", 67, "R1") in hits
    assert data["summary"]["new"] == len(data["new_violations"])


def test_real_repo_empty_baseline_text_mode_shows_hit(empty_baseline):
    res = run_lint("--baseline", str(empty_baseline))
    out = res.stdout.decode("utf-8")
    assert res.returncode == 1
    assert "lingshu/world/stable_lease.py:67:R1" in out
    assert "[新]" in out


# --------------------------------------------------------------- 6. 逐字节确定性
def test_json_byte_identical_across_runs_and_cwd(tmp_path):
    first = run_lint("--json")
    second = run_lint("--json")
    assert first.stdout == second.stdout
    assert first.returncode == second.returncode == 0
    # 任意 cwd 都必须可用（路径由 __file__ 推导，默认 baseline 仍能定位）
    third = run_lint("--json", cwd=tmp_path)
    assert third.returncode == 0
    assert third.stdout == first.stdout
    assert b"\r\n" not in first.stdout


def test_text_mode_runs_from_other_cwd(tmp_path):
    res = run_lint(cwd=tmp_path)
    assert res.returncode == 0
    assert "退出码 0" in res.stdout.decode("utf-8")


# ------------------------------------------- 判定回归：真实仓库当前唯一违规
def test_real_repo_current_hit_set_is_exactly_stable_lease(empty_baseline):
    """真实仓库（空 baseline）当前**有且只有** stable_lease.py:67:R1 一条。

    这是判定回归：若新增了其它命中，须先按 lint 头部豁免表复核，
    再决定登记 baseline 还是加豁免——不要直接改这条断言。
    """
    data = payload(run_lint("--baseline", str(empty_baseline), "--json"))
    assert hit_set(data) == {("lingshu/world/stable_lease.py", 67, "R1")}


# 判定回归的「源行守卫」：{（文件, 行号）: 该行必须仍含的片段}
# 没有它的话，只要行号漂移，参数化断言就恒真（豁免失效也看不出来）。
EXPECTED_MARKERS = {
    ("lingshu/world/gap_dual.py", 103): "retain=1.0 - self.decay",        # EX5（实参内）
    ("lingshu/nn/hex_train.py", 289): "d_before * retain_gain",           # 阈值比较，非更新
    ("lingshu/world/channel_credibility.py", 106): "evidence = 1.0 - alpha",
    ("lingshu/core/core.py", 1503): "math.sqrt(cer * (1 - cer) / n)",     # EX3
    ("lingshu/gen/hexgen_c1_real.py", 464): "* (1 - fy)",                 # EX4 插值
    ("lingshu/gen/hexgen_c1_real.py", 465): "fy * fx",                    # EX4 插值
    ("lingshu/gen/hexgen_self_source.py", 236): "* (1.0 - pc",            # EX4 合成
    ("lingshu/gen/hexgen_self_source.py", 411): "* (1.0 - pc",            # EX4 合成
    ("lingshu/world/confirmation.py", 34): "1.0 + math.exp(-x)",          # EX1 sigmoid
}


@pytest.mark.parametrize("rel,line", sorted(EXPECTED_MARKERS))
def test_judgement_regressions_not_reported(empty_baseline, rel, line):
    # 先守卫：目标行必须仍是当初被评估的那一行（否则下面的"不在命中里"恒真）
    src_line = (REPO_ROOT / rel).read_text(encoding="utf-8").splitlines()[line - 1]
    assert EXPECTED_MARKERS[(rel, line)] in src_line, \
        "行号已漂移：%s:%d 现在是 %r（请同步 EXPECTED_MARKERS 与 baseline 注释）" \
        % (rel, line, src_line.strip())

    data = payload(run_lint("--baseline", str(empty_baseline), "--json"))
    assert not [h for h in data["new_violations"] if h["file"] == rel and h["line"] == line], \
        "%s:%d 被判为豁免，不应出现在命中里" % (rel, line)


# ------------------------------------------------------------- baseline 机制
def test_shipped_baseline_registers_stable_lease():
    text = BASELINE.read_text(encoding="utf-8")
    assert "lingshu/world/stable_lease.py:67:R1" in text
    entries, errors = T.parse_baseline(text)
    assert errors == []
    assert ("lingshu/world/stable_lease.py", 67, "R1") in entries
    # 每条登记都要有性质注释（机械可核：登记行后必须有 `#`）
    for raw in text.split("\n"):
        if "lingshu/" in raw and raw.lstrip().startswith("lingshu/"):
            assert "#" in raw, "baseline 条目缺少性质注释：%s" % raw


def test_baseline_registration_synthetic_repo(tmp_path):
    root = tmp_path / "repo"
    src = ("import math\n"
           "\n"
           "\n"
           "def f(conf, gamma, age):\n"
           "    return conf * math.exp(-gamma * age)\n")
    write_module(root / "lingshu" / "pkg", "mod.py", src)
    baseline = root / "tools" / "time_core_lint_baseline.txt"

    # 未登记 → 退出 1
    res = run_lint(str(root / "lingshu"), "--repo-root", str(root), "--json")
    data = payload(res)
    assert res.returncode == 1
    assert data["new_violations"][0]["file"] == "lingshu/pkg/mod.py"
    assert data["new_violations"][0]["line"] == 5
    assert data["new_violations"][0]["rule"] == "R1"

    # --update-baseline 落盘 → 再跑退出 0
    res2 = run_lint(str(root / "lingshu"), "--repo-root", str(root), "--update-baseline")
    assert res2.returncode == 0
    assert baseline.exists()
    assert "lingshu/pkg/mod.py:5:R1" in baseline.read_text(encoding="utf-8")

    res3 = run_lint(str(root / "lingshu"), "--repo-root", str(root), "--json")
    data3 = payload(res3)
    assert res3.returncode == 0
    assert data3["new_violations"] == []
    assert data3["summary"]["registered"] == 1

    # 清空 → 退出 1（证明规则不是靠 baseline 空转）
    baseline.write_text("", encoding="utf-8")
    assert run_lint(str(root / "lingshu"), "--repo-root", str(root)).returncode == 1


def test_baseline_parse_accepts_rule_names_and_comments():
    text = ("# 注释\n"
            "\n"
            "lingshu/a.py:3:R1\n"
            "lingshu/b.py:9:R2    # 既有：样例\n"
            "lingshu/c.py:12:exp_decay\n")
    entries, errors = T.parse_baseline(text)
    assert errors == []
    assert entries == {("lingshu/a.py", 3, "R1"),
                       ("lingshu/b.py", 9, "R2"),
                       ("lingshu/c.py", 12, "R1")}


def test_render_baseline_roundtrip():
    entries = {("lingshu/z.py", 9, "R2"), ("lingshu/a.py", 3, "R1")}
    text = T.render_baseline(entries)
    parsed, errors = T.parse_baseline(text)
    assert errors == []
    assert parsed == entries


# --------------------------------------------------------------- 退出码 2
def test_usage_errors_return_two(tmp_path, empty_baseline):
    # 路径不存在
    assert run_lint("no/such/path").returncode == 2
    # 显式 baseline 不存在
    assert run_lint("--baseline", str(tmp_path / "nope.txt")).returncode == 2
    # baseline 格式错
    bad = tmp_path / "bad.txt"
    bad.write_text("这不是一条合法条目\n", encoding="utf-8")
    res = run_lint("--baseline", str(bad))
    assert res.returncode == 2
    assert "无法解析" in res.stdout.decode("utf-8")
    # 规则名未知
    bad2 = tmp_path / "bad2.txt"
    bad2.write_text("lingshu/a.py:1:R9\n", encoding="utf-8")
    assert run_lint("--baseline", str(bad2)).returncode == 2
    # --update-baseline 与 --json 同用
    assert run_lint("--update-baseline", "--json").returncode == 2
    # 仓库根不存在
    assert run_lint("--repo-root", str(tmp_path / "no-root")).returncode == 2
