# -*- coding: utf-8 -*-
"""test_doc_module_counts_guard · 文档模块计数必须等于实读（lingshu issue #78）
============================================================================
缺陷（#78，成立）：
    `README.md`「代码主体」节与 `docs/architecture.md` 声明的模块计数**不随
    引入登记更新**——声明 world 24 件 / 合计 52 模块 / import 冒烟 57/57，
    而实读 `lingshu/**/*.py` = **60**、world = **26**（非 `__init__` 25）、
    `__init__` = 5 ⇒ 早已过期（模块在 v0.2 与后续各批修复中增了
    `world/brain_store.py`、`core/component_resolver.py` 等）。

判据（本件，三层）：
  A 量具自检：README 与 architecture.md 的计数解析器必须真读出四个层名与数字，
    且实读计数不为 0——否则「没读到 ⇒ 跳过 ⇒ 恒绿」会成为坏解析器掩盖的假绿。
  B 主判据（本件机制）：README 四个层计数、architecture.md「当前计数」行的
    总文件数 / 非 `__init__` 数 / 四层计数 / 冒烟分子分母，必须**逐位等于**
    本件从文件系统实读的值。
  C 定点夹具（只让本件机制生效）：把 README 文本改回 `world 世界模型 24 件` 喂给
    同一检查器 ⇒ 必报 world 不符；并断言**仓内无第二件**读 README/architecture 的
    模块计数（否则两守卫互相代偿）。

判据来源：
  · 「件」= 该层非 `__init__` 的 `.py` 数——由 v0.1 基线反推：`ac6c0ca` 实读
    core 7 / world 24 / nn 14 / gen 7 = 52 非 `__init__`，而 v0.1 声明「52 模块」
    且「import 冒烟 57/57」＝ 52 + 5 个 `__init__`（`git ls-tree ac6c0ca lingshu`
    实读 57 个 `.py`）⇒ 口径自洽（经验标定：`ac6c0ca` 的 ls-tree 计数）。
  · 「import 冒烟 60/60」——本批修复时**本机实跑**读数（逐模块
    `importlib.import_module('lingshu.<...>')`，60 个模块全通过；命令与读数见
    `docs/architecture.md` 该行；本守卫只**核分子分母相等且等于实读 `.py` 总数**，
    不重跑 import 冒烟）。

不适用条件 / 已知边界：
  · 本件**不**改写 `docs/architecture.md` §七「引入登记」表内各**历史行**的计数
    ——那些行是逐次聚合动作的快照（日期 + 提交号），其计数对各自提交是**准确**的
    （如 `ac6c0ca` 确为 52/57）；本件只钉「当前计数」这一处活文档声明与 README。
  · 本件**不**重跑 `import` 冒烟（那是 `_smoke_count` 口径的一次性实跑读数；导入面
    已由 `tests/test_issue156_component_discovery.py` 等覆盖）。

定点变异自证（抽掉修复 ⇒ 必红）：
  把 `README.md` 的 `world 世界模型 25 件` 改回 `24 件`（或把 architecture.md
  的「当前计数」行改回 `52 模块 / 57/57`）⇒ B/C 两组当场红并点名不符项。

运行（lingshu 仓根）：python -X utf8 -m pytest tests/test_doc_module_counts_guard.py -q --no-header
退出码：0 = 全过；1 = 有断言失败
"""
from __future__ import annotations

import os
import re
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(REPO, "lingshu")
README = os.path.join(REPO, "README.md")
ARCH = os.path.join(REPO, "docs", "architecture.md")

LAYERS = ("core", "nn", "gen", "world", "vision")


# ---------------------------------------------------------------- 实读计数

def _actual_counts():
    """本仓实读：{层: 非 __init__ 的 .py 数} + 包根 + 总 .py + 非 __init__ 总数。"""
    def noninit(path):
        if not os.path.isdir(path):
            return 0
        return sum(1 for f in os.listdir(path)
                   if f.endswith(".py") and f != "__init__.py")

    def total_py(path):
        n = 0
        for dp, dn, fn in os.walk(path):
            dn[:] = [d for d in dn if d != "__pycache__"]
            n += sum(1 for f in fn if f.endswith(".py"))
        return n

    per = {layer: noninit(os.path.join(PKG, layer)) for layer in LAYERS}
    root = noninit(PKG)
    all_py = total_py(PKG)
    return {"per": per, "root": root, "all_py": all_py,
            "noninit_total": sum(per.values()) + root}


# ---------------------------------------------------------------- 文档解析

_README_RE = {
    "core": r"core 存算一体 (\d+) 件",
    "world": r"world 世界模型 (\d+) 件",
    "nn": r"nn 自研神经网络 (\d+) 件",
    "gen": r"gen 图像生成 (\d+) 件",
    "vision": r"vision 视觉线稿 (\d+) 件",
}
_ARCH_TOTAL_RE = r"\*\*(\d+)\*\* 个 `\.py`"
_ARCH_NONINIT_RE = r"\*\*(\d+)\*\* 个非 `__init__` 模块"
_ARCH_PER_RE = (r"core (\d+) / nn (\d+) / gen (\d+) / world (\d+) / "
                r"vision (\d+) / 包根 (\d+)")
_ARCH_SMOKE_RE = r"import 冒烟 \*\*(\d+)/(\d+)\*\*"


def _readme_counts(text):
    """从 README 文本取四层计数；缺失即 None（供量具自检判红）。"""
    out = {}
    for layer, pat in _README_RE.items():
        m = re.search(pat, text)
        if m:
            out[layer] = int(m.group(1))
    return out


def _arch_counts(text):
    """从 architecture.md「当前计数」行取总数/非 __init__/四层/冒烟。"""
    out = {}
    m = re.search(_ARCH_TOTAL_RE, text)
    if m:
        out["all_py"] = int(m.group(1))
    m = re.search(_ARCH_NONINIT_RE, text)
    if m:
        out["noninit_total"] = int(m.group(1))
    m = re.search(_ARCH_PER_RE, text)
    if m:
        (out["core"], out["nn"], out["gen"], out["world"], out["vision"],
         out["root"]) = (int(g) for g in m.groups())
    m = re.search(_ARCH_SMOKE_RE, text)
    if m:
        out["smoke_num"], out["smoke_den"] = int(m.group(1)), int(m.group(2))
    return out


def _mismatches(readme_text, arch_text, actual):
    """返回不符项列表（空 = 全一致）。B/C 共用，保证夹具与主判据同机制。"""
    bad = []
    rc = _readme_counts(readme_text)
    for layer in LAYERS:
        if layer not in rc:
            bad.append("README 未声明 %s 计数（解析失败或已删）" % layer)
        elif rc[layer] != actual["per"][layer]:
            bad.append("README %s：声明 %d，实读 %d" % (layer, rc[layer], actual["per"][layer]))

    ac = _arch_counts(arch_text)
    want = {
        "all_py": actual["all_py"],
        "noninit_total": actual["noninit_total"],
        "core": actual["per"]["core"], "nn": actual["per"]["nn"],
        "gen": actual["per"]["gen"], "world": actual["per"]["world"],
        "vision": actual["per"]["vision"],
        "root": actual["root"],
        "smoke_num": actual["all_py"], "smoke_den": actual["all_py"],
    }
    for key, val in want.items():
        if key not in ac:
            bad.append("architecture.md「当前计数」行缺 %s（解析失败）" % key)
        elif ac[key] != val:
            bad.append("architecture.md %s：声明 %d，实读 %d" % (key, ac[key], val))
    return bad


# ---------------------------------------------------------------- A 量具自检

def test_parsers_read_counts():
    """A：解析器必须真读出 README 四层 + architecture.md 计数行，且实读非 0。"""
    actual = _actual_counts()
    assert actual["all_py"] > 0 and actual["noninit_total"] > 0, (
        "实读计数为 0 ⇒ 扫描器坏了（PKG 路径失效），本判据会静默通过。实读=%r" % actual)
    assert all(actual["per"][l] > 0 for l in LAYERS), (
        "某层实读 0 ⇒ 扫描器或目录结构坏了。实读=%r" % actual)

    rc = _readme_counts(open(README, encoding="utf-8").read())
    assert set(rc) == set(LAYERS), (
        "README 解析器没读出全部四层计数（实读 %r）⇒ 本判据会静默跳过" % rc)
    ac = _arch_counts(open(ARCH, encoding="utf-8").read())
    assert {"all_py", "noninit_total", "smoke_num", "smoke_den"} <= set(ac), (
        "architecture.md 解析器没读出「当前计数」行（实读 %r）⇒ 本判据会静默跳过" % ac)


# ---------------------------------------------------------------- B 主判据

def test_docs_module_counts_match_reality():
    """B：README 与 architecture.md 的计数逐位等于本仓实读。"""
    actual = _actual_counts()
    bad = _mismatches(open(README, encoding="utf-8").read(),
                      open(ARCH, encoding="utf-8").read(), actual)
    assert not bad, (
        "文档模块计数与实读不符（#78——计数未随引入登记更新）：\n"
        + "\n".join("  · " + b for b in bad)
        + "\n\n实读 = %r" % actual
        + "\n修法：同步 README「代码主体」节与 docs/architecture.md 的「当前计数」行。")


# ---------------------------------------------------------------- C 定点夹具

def test_mutation_readme_stale_count_is_red():
    """C-1：把 README 改回 `world 世界模型 24 件` ⇒ 同一检查器必报 world 不符。

    **只让本件机制生效**：本检查器是仓内唯一读 README/architecture 模块计数的地方
    （见 C-2），故不存在与其它守卫互相代偿。
    """
    readme = open(README, encoding="utf-8").read()
    actual = _actual_counts()
    arch = open(ARCH, encoding="utf-8").read()
    assert not _mismatches(readme, arch, actual), "基线应无不符（前置条件不满足）"
    mutated = readme.replace("world 世界模型 %d 件" % actual["per"]["world"],
                             "world 世界模型 24 件")
    assert mutated != readme, "README 里没有可替换的 world 计数 ⇒ 本夹具须重写"
    bad = _mismatches(mutated, arch, actual)
    assert any("README world" in b for b in bad), (
        "变异自证失败：把 README 的 world 计数改回 24 后，检查器**没**报 world 不符 "
        "⇒ 本件判据无判别力。实报 = %r" % bad)


def test_mutation_arch_stale_counts_is_red():
    """C-2：把 architecture.md 的「当前计数」行改回 52/57 旧口径 ⇒ 必报不符。"""
    arch = open(ARCH, encoding="utf-8").read()
    actual = _actual_counts()
    readme = open(README, encoding="utf-8").read()
    mutated = arch.replace("**%d** 个 `.py`" % actual["all_py"], "**52** 个 `.py`")
    mutated = mutated.replace("import 冒烟 **%d/%d**" % (actual["all_py"], actual["all_py"]),
                              "import 冒烟 **57/57**")
    assert mutated != arch, "architecture.md 里没有可替换的计数 ⇒ 本夹具须重写"
    bad = _mismatches(readme, mutated, actual)
    assert any("architecture.md all_py" in b for b in bad), (
        "变异自证失败：把 architecture.md 计数改回 52/57 后检查器**没**报不符 "
        "⇒ 无判别力。实报 = %r" % bad)


def test_no_other_guard_reads_doc_counts():
    """C-3：仓内无第二件**同时**读文档计数并解析计数形态 ⇒ 不可能代偿。

    口径（AST，非全文搜词）：另一测试件必须**两样都占**才算重叠——
      ① 出现裸文件名常量 `'README.md'` / `'architecture.md'`（真读该文档的形态）；
      **且** ② 出现**本件专有的计数解析形态**（非 docstring 的字符串常量里含
      `件` 计数正则片段或 `个 `\\.py`` / `import 冒烟`——即本文件 `_README_RE` /
      `_ARCH_*_RE` 的解析器源）。
    **docstring 里提一句不算**（只在注释/说明里点名计数不构成解析）。
    只读 architecture.md 的 **§三 层树**（件 #79 的机制，其正则源片段是 box-drawing
    字符）不命中 ②，故两件不会被判为重叠——这正是「各钉各的缺陷面」。
    """
    import ast
    targets = {"README.md", "architecture.md"}
    me = os.path.basename(__file__)

    def _docstring_values(tree):
        """docstring 的**原始**字符串值集合（`ast.get_docstring` 会 dedent，与
        Constant 的原值不等，故直接取 body[0].value.value）。"""
        vals = set()
        for n in ast.walk(tree):
            body = getattr(n, "body", None)
            if (isinstance(body, list) and body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                vals.add(body[0].value.value)
        return vals

    def _count_parser_marker(src):
        tree = ast.parse(src)
        docs = _docstring_values(tree)
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) \
                    and n.value not in docs:
                v = n.value
                if ("\\d" in v and "件" in v) or "个 `\\.py`" in v or "import 冒烟" in v:
                    return True
        return False

    hits = []
    for dp, dn, fn in os.walk(os.path.join(REPO, "tests")):
        dn[:] = [d for d in dn if d != "__pycache__"]
        for f in sorted(fn):
            if not f.endswith(".py") or f == me:
                continue
            p = os.path.join(dp, f)
            src = open(p, encoding="utf-8").read()
            if not _count_parser_marker(src):
                continue
            tree = ast.parse(src)
            docstrings = _docstring_values(tree)
            for n in ast.walk(tree):
                if (isinstance(n, ast.Constant) and isinstance(n.value, str)
                        and n.value in targets and n.value not in docstrings):
                    hits.append("%s:%d" % (os.path.relpath(p, REPO).replace("\\", "/"),
                                           n.lineno))
    assert not hits, (
        "以下测试件也真读文档并解析**模块计数**（%r）⇒ 与本件机制重叠、可能互相代偿；"
        "须重新设计只让本件生效的夹具（或合并判据）。" % hits)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
