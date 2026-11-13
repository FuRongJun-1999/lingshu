# -*- coding: utf-8 -*-
"""#132 剩余两支 · 守卫（lingshu/core/core.py）

背景：#132 首轮修复只做了 D-44 点名的 **ASCII 词边界**（`EMAIL`→`AI`），
`tests/test_issue132_synonym_word_boundary_guard.py` 已钉住那一支。
本守卫补的是同一 issue 里**当时未做**的两条（判定方 2026-10-24 复核确认仍坏）：

  A 支 · 中文**单字**触发：`expand_query_terms("数据库迁移")` 曾展开出
      `['数据库迁移', '记忆', '记录', '库']` —— 单字「库」把整组灌进 LIKE 预筛。
      issue 建议 1 后半：「中文至少禁止单字触发（「库」）」。
  B 支 · 非同义关系：`expand_query_terms("实现登录")` 曾展开出
      `['实现登录', '测试', '试验', '实验', '实现', '集成', '验证']` ——
      issue 建议 2：「从通用同义表里移除项目专名（「灵枢」）和非同义关系
      （「实现」↔「测试」）」。本件只移除 `实现`（`灵枢` 保留：它是项目专名，
      但移除会打红既有 B 组「AI 与灵枢」断言，属另一裁定面，未在本件做）。

修法（最小）：
  1. `_synonym_trigger_hit`：非 ASCII 词长度 < 2 ⇒ 不触发（不引入分词器，
     只要求中文「词」至少两字）。
  2. `SYNONYM_GROUPS`：`{"实验","试验","集成","实现","验证","测试"}`
     → 移除 `实现`。

★ 与既有守卫的兼容性（已实测，不是推断）：
  · C 组「`记忆库` 仍展开出 `记忆`/`记录`/`库`」**仍通过** —— `记忆` 是两字词，
    照旧触发整组；单字禁令只挡「整句里只有单字命中」的情形。
  · B 组「独立 `AI` 仍触发」不受影响。

断言组（把修复抽掉 ⇒ A/B 支必红）：
  A1 `数据库迁移` 不再展开出 `记忆`/`记录`/`库`。
  A2 对照：`记忆库` 仍展开出 `记忆`/`记录`/`库`（两字词照旧触发，防过度收紧）。
  A3 对照：单字查询 `库` 本身不得触发（它是触发源本身，不是被扩展词）。
  B1 `实现登录` 不再展开出 `实验`/`试验`/`集成`/`验证`/`测试`。
  B2 对照：`实验`/`测试` 等词本身仍能触发该组（组未被整体删除）。
  C  静态有牙：AST 扫 `_synonym_trigger_hit`，必须存在「非 ASCII 且 len<2 ⇒ 返回」
     的单字禁令；扫 `SYNONYM_GROUPS`，`实现` 不得出现在「实验」组里。

运行（仓根）：python -X utf8 -m pytest tests/test_issue132_chinese_single_char_and_nonsynonym_guard.py -q --no-header
"""
from __future__ import annotations

import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingshu.core.core import LayeredStore  # noqa: E402

_MEM_GROUP = {"记忆", "记录", "库"}
_EXP_GROUP = {"实验", "试验", "集成", "验证", "测试"}


# ---------- A 支：中文单字不得触发 ----------

def test_a1_chinese_single_char_does_not_trigger():
    """A1：`数据库迁移` 的「库」是单字，不得触发记忆组。"""
    terms = set(LayeredStore.expand_query_terms("数据库迁移"))
    leaked = _MEM_GROUP & terms
    assert not leaked, (
        "#132 A 支：中文单字「库」仍触发记忆组，展开出 %r（issue 原报 "
        "['数据库迁移','记忆','记录','库']）" % (sorted(leaked),))


def test_a2_two_char_chinese_still_triggers():
    """A2 对照：两字词 `记忆` 仍触发整组（防过度收紧，兼容既有 C 组）。"""
    terms = set(LayeredStore.expand_query_terms("记忆库"))
    assert _MEM_GROUP <= terms, (
        "#132 A2：两字词不再触发记忆组（过度收紧）——实际 %r" % (sorted(terms),))


def test_a3_bare_single_char_query_does_not_trigger():
    """A3：单字查询 `库` 本身不得作为触发词展开整组。"""
    terms = set(LayeredStore.expand_query_terms("库"))
    # 严格判据：`库` 是单字 ⇒ 不触发，展开结果应只有查询本身
    assert terms == {"库"}, (
        "#132 A3：单字查询 `库` 仍触发了记忆组——实际 %r" % (sorted(terms),))


# ---------- B 支：非同义关系不得留在通用表 ----------

def test_b1_implementation_does_not_trigger_experiment_group():
    """B1：`实现登录` 的「实现」不得触发「实验/测试」组（实现 ≠ 测试）。"""
    terms = set(LayeredStore.expand_query_terms("实现登录"))
    leaked = _EXP_GROUP & terms
    assert not leaked, (
        "#132 B 支：非同义关系「实现→测试」仍在，展开出 %r（issue 原报 "
        "['实现登录','测试','试验','实验','实现','集成','验证']）" % (sorted(leaked),))


def test_b2_experiment_words_still_trigger():
    """B2 对照：该组词本身仍能触发（组未被整体删除）。"""
    for w in ("实验", "测试", "验证"):
        terms = set(LayeredStore.expand_query_terms(w + "报告"))
        assert _EXP_GROUP <= terms, (
            "#132 B2：%r 不再触发实验组（组被整体删除，过度收紧）——实际 %r"
            % (w, sorted(terms)))


# ---------- C 静态有牙 ----------

def _core_tree():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "lingshu", "core", "core.py"), encoding="utf-8") as f:
        return ast.parse(f.read())


def test_c_static_single_char_ban_and_group_cleanup():
    """C：源码层面必须存在单字禁令，且 `实现` 不在实验组。"""
    tree = _core_tree()

    # C1：`_synonym_trigger_hit` 内必须有「非 ASCII 且 len<2 ⇒ return」的判据
    src_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "lingshu", "core", "core.py")
    src_text = open(src_path, encoding="utf-8").read()
    fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_synonym_trigger_hit":
            fn = node
            break
    assert fn is not None, "#132 C1：找不到 _synonym_trigger_hit"
    has_ban = False
    for node in ast.walk(fn):
        if isinstance(node, ast.If):
            seg = (ast.get_source_segment(src_text, node.test) or "").replace(" ", "")
            if "isascii" in seg and "len(word)<2" in seg:
                has_ban = True
    assert has_ban, (
        "#132 C1：`_synonym_trigger_hit` 内没有「非 ASCII 且 len<2 ⇒ 不触发」的单字禁令")

    # C2：`实现` 不得出现在「实验」组
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "SYNONYM_GROUPS":
                    groups = ast.literal_eval(node.value)
                    for g in groups:
                        if "实验" in g:
                            assert "实现" not in g, (
                                "#132 C2：`实现` 仍在「实验」同义组里：%r" % (sorted(g),))
                            return
    raise AssertionError("#132 C2：找不到含「实验」的同义组")
