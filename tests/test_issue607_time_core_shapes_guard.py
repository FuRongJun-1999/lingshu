# -*- coding: utf-8 -*-
"""issue #607 · time_core_lint R3「手写衰减核」闸三处形态缺口的守卫测试

缺陷（#607，P1）：三处互相独立的形态缺口（改前实测，HEAD 375cf55，空 baseline）：
  ① _KEEP_WORD 词表只认 retain*/keep*/momentum*/ema*/decay*/forget*/halflife/
     half_life/smoothing*——gamma/alpha/rho/tau/discount/lr 不在册，而同文件
     RULE_R2C 自己把 gamma/lambda 列为衰减名 ⇒ 语义等价写法判读相反：
     8 种折扣写法夹具 6 条零命中（`x = x * (1 - gamma)` 被报、`x = x * gamma` 不报）。
  ② add() 以 (line, rule) 去重 ⇒ 同一物理行第 2/3 条同类违规被丢
     （`;` 同行 5 条逻辑违规实报 3 条）。
  ③ _is_depth0 两道 finditer 要求乘法在括号深度 0 ⇒ 顶层表达式里被一对冗余括号
     包住的自指乘法全部逃逸（7 种等价写法只 1 条命中）。

修复（本守卫锁定的行为）：
  ① _KEEP_WORD 补 gamma/alpha/rho/tau/discount/lr（`\b` 精确词，与 RULE_R2C 对齐）；
  ② 命中逐条入账——去重只在「同一语句内、同一规则、同一处文本」（R3(a)×R3(c)、
     R2(a)×R2(c) 对同一处的重复检出仍合并），同行多语句的同类违规各自成条；
  ③ 深度判定改 _is_depth0_peeled：剥掉**成对且恰好包裹整个乘法**的冗余外层括号，
     调用/属性访问伴生括号（函数实参、`(t * w).sum()`、`(…)[0]`）不剥。
基线已随词表收紧重测（11→17，physics.py +6 条 R3 形状级命中，见
tools/time_core_lint_baseline.txt 的 2026-10-15 #607 段）。

断言组
------
G1 词表 6 新词各自命中（含 `(1-gamma)` 与裸 `x = x * gamma` 等价双写）＋
   词边界防误伤（`gamma_t`/`alphas`/`lru`/`agamma` 不命中）；
G2 同行 3 条违规全报（`;` 切分）＋同语句多路径检出仍合并（防多报）＋
   baseline 三元组 1 行覆盖同行多条；
G3 括号 6+ 形态命中＋防过修（`(t * w).sum()`、`*fill` 解包、函数实参括号不误报）；
G4 仓内全量跑 exit 0（锁定新基线数 17）。

变异锚点：_KEEP_WORD 回退掉 `|gamma\b|…|lr\b` 一行 ⇒ G1 全红；
_self_multiplicative_pos 回退深度 0 判定（_is_depth0_peeled 换回 _is_depth0）
⇒ G3 命中面全红；add() 回退 (line, rule) 去重 ⇒ G2 同行计数断言红。恢复 ⇒ 全绿。
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

# 按文件路径加载（lingshu #288 补报帖 · 2026-10-31 收口）：本件不 import lingshu，
# 但与 import lingshu 的测试件**同进程收集**时，pathguard 的解析期拒绝 finder
# （#150A 语义）会把「cwd 面才能解析的裸顶层名」拒掉——`from tools import …`
# 在该时序下收集期 ImportError（各自单跑均绿）。仓根自足 sys.path 头部**救不了**
# 它（REPO_ROOT realpath 归一后就是 cwd，属拒绝面）。故按文件路径直接加载工具
# 模块，绕开包名解析（不碰 pathguard，符合 #273 裁定）。
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("time_core_lint_under_test", str(TOOL))
T = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(T)


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


def hit_multiset(data):
    """命中多重集：同一 (line, rule) 的多条同类违规**逐条**计数（#607 ②）。"""
    out = {}
    for h in data["new_violations"]:
        key = (h["line"], h["rule"])
        out[key] = out.get(key, 0) + 1
    return out


@pytest.fixture
def empty_baseline(tmp_path):
    path = tmp_path / "empty_baseline.txt"
    path.write_text("", encoding="utf-8")
    return path


_COUNTER = {"n": 0}


def lint_body(body: str, name: str, empty_baseline: Path, tmp_path: Path):
    """写一段合成源码到独立目录并跑工具（JSON 模式，互不污染）。"""
    _COUNTER["n"] += 1
    directory = tmp_path / ("pkg%d" % _COUNTER["n"])
    directory.mkdir(parents=True)
    (directory / name).write_text(body, encoding="utf-8")
    return payload(run_lint(str(directory), "--baseline", str(empty_baseline), "--json"))


# ---------------------------------------------- G1 词表：6 新词各自命中＋防误伤
#: #607 ① 的准绳：「语义等价写法判读一致」——裸 `x = x * WORD` 与 `x = x * (1 - WORD)`
#: 都是衰减写法，都必须命中（改前裸写法 6 条零命中）。
G1_WORDS = ("gamma", "alpha", "rho", "tau", "discount", "lr")


def test_g1_new_vocab_words_each_hit(tmp_path, empty_baseline):
    """6 个新词的裸乘写法必须各自命中 R3（改前 6 条零命中）。"""
    lines = []
    for i, w in enumerate(G1_WORDS):
        lines += ["def f%d(x, %s):" % (i + 1, w),
                  "    x = x * %s" % w,
                  "    return x",
                  ""]
    data = lint_body("\n".join(lines) + "\n", "g1_bare.py", empty_baseline, tmp_path)
    assert data["summary"]["new"] == 6, data["new_violations"]
    assert hit_multiset(data) == {(2 + 4 * i, "R3"): 1 for i in range(6)}, \
        data["new_violations"]


def test_g1_equivalent_writings_agree(tmp_path, empty_baseline):
    """`(1 - gamma)` 与裸 `x * gamma` 等价双写：两形态都命中（判读不再相反）。"""
    body = ("def f(x, gamma):\n"
            "    x = x * (1 - gamma)\n"
            "    x = x * gamma\n"
            "    return x\n")
    data = lint_body(body, "g1_equiv.py", empty_baseline, tmp_path)
    # `(1 - gamma)` 行：R2(a)/R2(c) 合并 1 条 R2 + R3(a) `(1-…)` 形 1 条；裸写法行：R3 1 条
    assert hit_multiset(data) == {(2, "R2"): 1, (2, "R3"): 1, (3, "R3"): 1}, \
        data["new_violations"]


def test_g1_word_boundary_no_prefix_or_suffix_hits(tmp_path, empty_baseline):
    """词边界防误伤：前后缀扩展与拼前缀一律不命中（`\b` 精确词口径，与 R2C 一致）。"""
    body = (
        "def f(x, gamma_t, alphas, discount_rate, w, agamma):\n"
        "    a = x * gamma_t\n"          # gamma 精确词，不认 _t 后缀（与 R2C 口径一致）
        "    b = x * alphas\n"           # 不认 s 后缀
        "    c = x * discount_rate\n"    # 不认 _rate 后缀
        "    d = w * agamma\n"           # `* agamma`：R3(c) 词表锚在 `* ` 之后，拼前缀起不了匹配
        "    e = w * lr0\n"              # lr 精确词，不认 0 后缀
        "    return a, b, c, d, e\n"
    )
    data = lint_body(body, "g1_bounds.py", empty_baseline, tmp_path)
    assert data["new_violations"] == [], data["new_violations"]


def test_g1_new_words_in_return_and_attr_forms(tmp_path, empty_baseline):
    """新词在 return 表达式与 `self.x` 属性形态下同样命中。"""
    body = ("def f(self, g):\n"
            "    return self.v * self.alpha\n"
            "\n"
            "def h(v, tau):\n"
            "    return v * tau\n")
    data = lint_body(body, "g1_forms.py", empty_baseline, tmp_path)
    assert hit_multiset(data) == {(2, "R3"): 1, (5, "R3"): 1}, data["new_violations"]


# ---------------------------------------- G2 同行多违规划账（`;` 切分逐条入账）
def test_g2_three_same_rule_violations_on_one_line_all_counted(tmp_path, empty_baseline):
    """★ 改前读数复现点：`a = a * retain; b = b * retain; c = c * retain`
    逻辑 3 条同类违规，改前只报 1 条（(line, rule) 去重吞掉第 2/3 条）。"""
    body = ("def f(a, b, c, retain):\n"
            "    a = a * retain; b = b * retain; c = c * retain\n"
            "    return a, b, c\n")
    data = lint_body(body, "g2_three.py", empty_baseline, tmp_path)
    assert hit_multiset(data) == {(2, "R3"): 3}, data["new_violations"]
    assert data["summary"]["new"] == 3


def test_g2_mixed_rules_same_line_full_accounting(tmp_path, empty_baseline):
    """同行＋后续行的 5 条逻辑违规（3×R3 同行同类 + 1×R3 + 1×R2）全部入账
    （改前实报 3 条——同行第 2/3 条被吞）。"""
    body = ("def g(a, b, c, retain, keep, f):\n"
            "    a = a * retain; b = b * retain; c = c * retain\n"
            "    k2 = keep * retain\n"
            "    z = z2 * (1 - f)\n"
            "    return a, b, c, k2, z\n")
    data = lint_body(body, "g2_mixed.py", empty_baseline, tmp_path)
    assert hit_multiset(data) == {(2, "R3"): 3, (3, "R3"): 1, (4, "R2"): 1}, \
        data["new_violations"]
    assert data["summary"]["new"] == 5


def test_g2_split_lines_equivalent_count(tmp_path, empty_baseline):
    """同行多语句的计数与拆行写法**一致**（`;` 同行不应低报；规则构成也一致）。"""
    same_line = ("def f(a, b, c, retain):\n"
                 "    a = a * retain; b = b * retain; c = c * retain\n")
    split = ("def f(a, b, c, retain):\n"
             "    a = a * retain\n"
             "    b = b * retain\n"
             "    c = c * retain\n")
    one = lint_body(same_line, "g2_same.py", empty_baseline, tmp_path)
    two = lint_body(split, "g2_split.py", empty_baseline, tmp_path)
    assert one["summary"]["new"] == two["summary"]["new"] == 3
    assert hit_multiset(one) == {(2, "R3"): 3}
    assert hit_multiset(two) == {(2, "R3"): 1, (3, "R3"): 1, (4, "R3"): 1}


def test_g2_single_violation_multi_path_detection_still_merged(tmp_path, empty_baseline):
    """防多报：同一处乘法被多条规则路径检出（R3(a)×R3(c)、R2(a)×R2(c)）仍合并为 1 条。"""
    body = ("def f(x, gamma, d):\n"
            "    x = x * gamma\n"          # R3(a) 词形 + R3(c) 词名 → 1 条
            "    y = y * (1 - d)\n"        # R2(a)+R2(c) → 1 条 R2；R3(a) `(1-…)` 形 → 1 条 R3
            "    return x, y\n")
    data = lint_body(body, "g2_merge.py", empty_baseline, tmp_path)
    assert hit_multiset(data) == {(2, "R3"): 1, (3, "R2"): 1, (3, "R3"): 1}, \
        data["new_violations"]


def test_g2_baseline_single_entry_covers_same_triple_duplicates(tmp_path):
    """baseline 三元组集合语义：同行 3 条同类违规登记 1 行即全部覆盖（exit 0），
    `--update-baseline` 机械重写也按集合去重（同一三元组只落 1 行）。"""
    root = tmp_path / "repo"
    (root / "lingshu" / "pkg").mkdir(parents=True)
    (root / "lingshu" / "pkg" / "mod.py").write_text(
        "def f(a, b, c, retain):\n"
        "    a = a * retain; b = b * retain; c = c * retain\n",
        encoding="utf-8")
    res = run_lint(str(root / "lingshu"), "--repo-root", str(root), "--json")
    data = payload(res)
    assert res.returncode == 1 and data["summary"]["new"] == 3

    baseline = root / "tools" / "time_core_lint_baseline.txt"
    baseline.parent.mkdir(parents=True, exist_ok=True)
    baseline.write_text("lingshu/pkg/mod.py:2:R3\n", encoding="utf-8")
    assert run_lint(str(root / "lingshu"), "--repo-root", str(root)).returncode == 0
    assert run_lint(str(root / "lingshu"), "--repo-root", str(root),
                    "--update-baseline").returncode == 0
    assert baseline.read_text(encoding="utf-8").count("lingshu/pkg/mod.py:2:R3") == 1


# --------------------------- G3 括号：冗余包裹命中＋调用/解包伴生括号防过修
G3_EQUIV_FORMS = (
    ("def h1(v):\n    v = v * 0.9\n    return v\n", "深度 0（改前唯一命中的形态）"),
    ("def h2(v):\n    v = (v * 0.9)\n    return v\n", "整段冗余包裹"),
    ("def h3(v):\n    v = ((v * 0.9))\n    return v\n", "双层冗余包裹"),
    ("def h4(v, g):\n    v = (v * 0.9) + 0.1 * g\n    return v\n", "括号项在和式左侧"),
    ("def h5(v, g):\n    v = 0.1 * g + (v * 0.9)\n    return v\n", "括号项在和式右侧"),
    ("def h6(self, g):\n    self.v = (self.v * 0.9) + 0.1 * g\n    return self.v\n",
     "self.x 属性形态"),
    ("def h7(v, w):\n    v = (v * 0.9) * w\n    return v\n", "剥括号后仍是顶层乘法"),
)


@pytest.mark.parametrize("form,desc", G3_EQUIV_FORMS, ids=[d for _, d in G3_EQUIV_FORMS])
def test_g3_redundant_paren_wrapped_self_update_hits(tmp_path, empty_baseline, form, desc):
    """★ 改前读数复现点：7 种等价写法只 1 条命中 ⇒ 全部命中（R3 各 1 条）。"""
    data = lint_body(form, "g3_form.py", empty_baseline, tmp_path)
    assert hit_multiset(data) == {(2, "R3"): 1}, \
        "等价写法（%s）判定异常：%r → %r" % (desc, form, data["new_violations"])


def test_g3_paren_wrapped_word_form_hits(tmp_path, empty_baseline):
    """词形版：`(x * gamma)` / `((y * ema))` 冗余包裹同样命中（各自自指）。"""
    body = ("def f(x, y, gamma, ema):\n"
            "    x = (x * gamma)\n"
            "    y = ((y * ema))\n"
            "    return x, y\n")
    data = lint_body(body, "g3_word.py", empty_baseline, tmp_path)
    assert hit_multiset(data) == {(2, "R3"): 1, (3, "R3"): 1}, data["new_violations"]


def test_g3_call_arg_and_method_parens_not_reported(tmp_path, empty_baseline):
    """防过修：函数/方法实参括号里的自指乘法不误报（开括号前是名字 ⇒ 不剥）。"""
    body = (
        "def f(v, foo, obj, arr_fn):\n"
        "    a = foo(v * 0.9)\n"            # 调用实参
        "    b = foo((v * 0.9))\n"          # 实参内再裹一层（两层都是实参括号）
        "    c = obj.m(v * 0.9)\n"          # 方法实参
        "    d = arr_fn(v * 0.9)[0]\n"      # 调用后下标取用
        "    return a, b, c, d\n"
    )
    data = lint_body(body, "g3_args.py", empty_baseline, tmp_path)
    assert data["new_violations"] == [], data["new_violations"]


def test_g3_orthogonalize_sum_fill_not_reported(tmp_path, empty_baseline):
    """防过修：`(t * w).sum()` 正交化与 `(*fill, 255)` 解包不误报（伴生括号不剥；
    形态取自仓内 physics 惯用式与 silhouette3d.py:197 的 draw.polygon 实参）。"""
    body = (
        "def orthogonalize(t, w):\n"
        "    t = (t - float((t * w).sum())).astype(float)\n"
        "    return t\n"
        "\n"
        "def draw_poly(poly, fill, part):\n"
        "    out = draw.polygon(poly, fill=(*fill, 255), outline=(*part.outline, 255))\n"
        "    return out\n"
    )
    data = lint_body(body, "g3_nofix.py", empty_baseline, tmp_path)
    assert data["new_violations"] == [], data["new_violations"]


def test_g3_peeled_helper_unit():
    """_is_depth0_peeled 的最小单元：冗余包裹 True、实参/伴生括号 False。"""
    P = T._is_depth0_peeled
    # 冗余包裹（成对且恰包整个乘法）——剥后深度 0
    assert P("(v * 0.9)", 1, 8) is True
    assert P("((v * 0.9))", 2, 9) is True
    assert P("(self.v * 0.9) + 0.1 * g", 1, 13) is True
    assert P("0.1 * g + (v * 0.9)", 11, 18) is True
    assert P("(v * 0.9) * w", 1, 8) is True
    # 深度 0（无括号）照旧
    assert P("v * 0.9", 0, 7) is True
    # 调用实参（开括号前是名字）——即使恰好只包这个乘法
    assert P("foo(v * 0.9)", 4, 11) is False
    assert P("foo((v * 0.9))", 5, 12) is False
    assert P("obj.m(v * 0.9)", 6, 13) is False
    # 伴生括号（闭括号后是 `.` / `(` / `[`）
    assert P("(t * w).sum()", 1, 6) is False
    assert P("(t * w)(x)", 1, 6) is False
    assert P("(t * w)[0]", 1, 6) is False
    # 括号里不止这个乘法（如实参之一 / 元组元素）——不是冗余包裹
    assert P("foo(a, v * 0.9)", 7, 14) is False
    assert P("(v * 0.9, x)", 1, 8) is False


# ------------------------------------------------- G4 仓内全量：exit 0 锁基线
def test_g4_repo_full_run_exit_zero_locks_baseline():
    """全仓实跑：出货 baseline 下 exit 0、new==0，命中总数锁定为**当前基线值**。

    变更史（均按本仓程序：先逐条复核豁免表、再更新断言）：
      · 17 条：2026-10-15 #607 词表收紧使 physics.py +6 条 R3（见 baseline #607 段）。
      · 23 条：2026-10-16 **物理仿真深化 + 碰撞箱**（`drag2`/`slope_friction`/`rolling`/`collision`
        + `Environment`/`simulate` + AABB/OBB-SAT/MTV/CCD）在 `physics.py` 行号之上插入大量代码
        ⇒ 既有条目全部 stale、同一违规以新行号出现。**逐条复核 20 条新命中**后确认全部属既有两类豁免
        （R1 = 物理阻尼/振子**闭式解**与 `_log_cosh` softplus；R3 = **运动学外推**，`tau` 是流逝时间
        非保持率），`(file,rule)` 多重集 = base 17 + 本笔 6，**无一条被删除** ⇒ **不是放宽判据**。
    """
    res = run_lint("--json")
    assert res.returncode == 0
    data = payload(res)
    assert data["new_violations"] == []
    assert data["summary"]["total"] == 23, data["summary"]
    assert data["stale_baseline_entries"] == []
    #: ★改为**形态锚**（照本件 docstring 的声明"锚的是登记存在性而非物理行号本身"）：
    #: 行号会随 physics.py 的插入/删除漂移，钉死行号会让守卫在**合法重编号**时误报。
    physics_r3 = [h for h in data["registered_violations"]
                  if h["file"] == "lingshu/world/physics.py" and h["rule"] == "R3"]
    assert len(physics_r3) == 11, sorted(h["line"] for h in physics_r3)   # #607 的 6 条 + 本笔 5 条
    physics_r1 = [h for h in data["registered_violations"]
                  if h["file"] == "lingshu/world/physics.py" and h["rule"] == "R1"]
    assert len(physics_r1) == 9, sorted(h["line"] for h in physics_r1)    # 物理阻尼 8 条 + `_log_cosh` 1 条


def test_g4_baseline_documents_607_retest():
    """baseline 留痕：#607 重测记录与 6 条新登记都在文件里（可机械核；
    行号随 #662 的 +6 插入同步移位，锚的是登记存在性而非物理行号本身）。"""
    text = BASELINE.read_text(encoding="utf-8")
    assert "#607" in text
    #: ★形态锚（不钉物理行号）：physics.py 的 R3 条目数 = #607 的 6 条 + 本笔 5 条
    r3_lines = [l for l in text.split("\n")
                if l.startswith("lingshu/world/physics.py:") and l.split(":")[2].split()[0] == "R3"]
    assert len(r3_lines) == 11, r3_lines
    entries, errors = T.parse_baseline(text)
    assert errors == []
    assert len(entries) == 23
    # 每条登记都要有性质注释（与既有守卫 test_time_core_lint.py 同一口径）
    for raw in text.split("\n"):
        if raw.lstrip().startswith("lingshu/"):
            assert "#" in raw, "baseline 条目缺少性质注释：%s" % raw
