"""dsh-bounty #1795 守卫：search_content 族「None＝全量、-1＝全量」口径。

缺陷（修复前实测，锚定 ed8700d）：
  1. `LINGSHU_FTS=1` 时 `search_content(limit=None)` 抛 `TypeError`
     （`_search_content_fts` 的 `max(int(limit) * 5, 100)` 对 None 崩，
     且该 TypeError 不在 `except sqlite3.Error` 内）——LIKE 腿同输入却正常
     全量，同一公开入口两条腿行为分叉。
  2. `search_content(limit=-1)` 走 Python 负切片 `scored[:-1]`——5 条命中
     只回 4 条，静默丢尾；同族 `query_nodes(limit=-1)` 是全量（#658 G2）。
  3. 同款负切片还有 `semantic_search` / `recall` / `query_by_semantics`。

判据来源：dsh-bounty #1795（含逐字读数）；契约＝#658 守卫
tests/test_issue658_limit_none_full_return_guard.py 的 G1/G2 判据
（None＝全量、0＝空、-1＝全量、显式 k＝精确截断）。
本文件是「回退即红」对照：撤掉任一入口的 `_normalize_limit(limit)`，
对应用例失败。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingshu.core.core import SpacetimeMemoryEngine  # noqa: E402


def _seeded(n=5):
    e = SpacetimeMemoryEngine(":memory:")
    for i in range(n):
        e.add_context("情境内容条目%d：检索上限语义" % i)
    return e


# ------------------------------------------------- G1 None＝全量（两腿一致）

def test_g1_like_leg_limit_none_full():
    e = _seeded(5)
    got = e.store.search_content("情境内容", limit=None)
    assert len(got) == 5


def test_g1_fts_leg_limit_none_full(monkeypatch):
    monkeypatch.setenv("LINGSHU_FTS", "1")
    e = SpacetimeMemoryEngine(":memory:")
    for i in range(5):
        e.add_context("全文检索条目%d：上限语义" % i)
    got = e.store.search_content("全文检索", limit=None)  # 修复前：TypeError
    assert len(got) == 5


# ------------------------------------------------------- G2 -1＝全量

def test_g2_limit_minus_one_full():
    e = _seeded(5)
    got = e.store.search_content("情境内容", limit=-1)
    assert len(got) == 5  # 修复前：4（scored[:-1] 丢尾条）


def test_g2_recall_limit_minus_one_full():
    e = SpacetimeMemoryEngine(":memory:")
    for i in range(5):
        e.add_perception("组合联想条目%d：上限语义与复用" % i, importance=0.6)
    got = e.recall("组合联想条目", limit=-1)
    assert len(got) == 5


def test_g2_semantic_and_query_by_semantics_no_negative_truncation():
    """semantic_search / query_by_semantics 在无 provider 时走各自的
    回退/空路径也要先过归一——用 -1 直接调用不崩、不吞尾。"""
    e = SpacetimeMemoryEngine(":memory:")  # 无 semantic provider
    for i in range(5):
        e.add_context("语义坐标条目%d：上限语义" % i)
    got = e.semantic_search("语义坐标条目", limit=-1)  # 回退腿：store.search_content
    assert len(got) == 5


# --------------------------------------- G3 防过修正向：显式 k 仍精确截断

def test_g3_explicit_limit_still_truncates():
    e = _seeded(5)
    assert len(e.store.search_content("情境内容", limit=3)) == 3
    assert e.store.search_content("情境内容", limit=0) == []  # 0＝空（#658 G2）
