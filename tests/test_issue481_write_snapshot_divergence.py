#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_issue481_write_snapshot_divergence —— #481 命中腿的「断言分歧」守卫。

缺陷（lingshu #481）：`LongTermMemoryGate.write_snapshot` 的「已存在」命中腿旧实现
只按 `search_content` 得分判同——`if hits and hits[0][1] >= 0.95:` 直接取
`existing_id = hits[0][0].id` 走合并，**完全不问两条断言是否分歧**（longterm_gate.py
全文 grep divergence|assertion|negation = 0 命中；判据 `_assertion_divergence` 定义在
core.py:5904，全仓唯一调用点是 core.py:6094 的 `add_perception` M5 去重腿）。

后果：否定更正（「仍然可用」→「仍然不可用」，char-bigram jaccard 0.960526 ≥ 0.95）被
并进旧节点 ⇒ 更正文本不落库（content 行数为 0）、旧节点 importance 被抬到更正值，
importance_hint ≥ 0.7 时更被 `protect_node` 打 `no_forget`（**错误记忆被永久保护**）；
同一对文本走 `add_perception` 却会另立新节点（M5 腿有 divergence 闸）。

本件断言修复后的行为（全部 `:memory:` 隔离库，不落盘、不碰在役记忆库）：
  G1 negation 更正另立新节点，且**更正文本真的落库**（旧行为 content 行数为 0）。
  G2 以更正原话 recall，命中新节点且返回正文 == 更正文本（旧行为返回错误事实）。
  G3 纯重复文本仍走原合并：节点数不变、node_id 不变、tags/importance 合并语义不变。
  G4 无分歧的换说法（jaccard 同级）仍合并（新闸不误杀重复判定）。
  G5 旧节点不被提权：importance 保持原值（hint=0.6 / 0.9 两种）。
  G6 旧节点不被打 `no_forget`、不在 protections 表里；hint ≥ 0.7 时被保护的是**新节点**。
  G7 0.95 阈值与 tag_bonus 计分未动：raw jaccard < 0.95 但（含标签加分后的）命中分
     ≥ 0.95 且无分歧时，仍按原逻辑合并。

运行（lingshu 仓根）：python -X utf8 -m pytest tests/test_issue481_write_snapshot_divergence.py -q --no-header
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.core.core import (  # noqa: E402
    SpacetimeMemoryEngine, LayeredStore, MemoryLayer)

# 原命题（含否定标记的一侧为「不可用」的更正句）。长句保证 char-bigram jaccard ≥ 0.95。
A = ("本次独立复核的结论是该方法在低资源的小样本场景下仍然可用，"
     "误差区间与上一次的测量结果完全一致，因此可以直接进入下一步的"
     "集成测试流程，标定工作已经完成。")
B_NEG = A.replace("仍然可用", "仍然不可用")      # 否定更正：divergence == "negation"
B_NEUTRAL = A.replace("可用，误差", "可用，且误差")  # 无分歧换说法：divergence == ""

TAG_WORD = "仍然可用"   # 作为节点标签，出现在查询里 ⇒ search_content 命中分 +0.05 tag_bonus


def _eng():
    return SpacetimeMemoryEngine(":memory:")


def _n_nodes(e):
    return e.store.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]


def _rows_of(e, content):
    return e.store.conn.execute(
        "SELECT COUNT(*) FROM nodes WHERE content=?", (content,)).fetchone()[0]


def _importance(e, nid):
    row = e.store.conn.execute(
        "SELECT importance FROM nodes WHERE id=?", (nid,)).fetchone()
    return None if row is None else float(row[0])


def _hit(e, text):
    """(节点, 命中分) —— 与 gate 命中腿同一查询（层与 limit 同口径）。"""
    hits = e.store.search_content(
        text, layers=[MemoryLayer.KNOWLEDGE, MemoryLayer.CONTEXT], limit=1)
    return (None, None) if not hits else (hits[0][0], float(hits[0][1]))


def _seed(e, base=A, tags=None, hint=0.5):
    """写入原命题，返回其 node_id。"""
    res = e.longterm_snapshot(base, tags=(tags or ["pbase"]), importance_hint=hint)
    assert res.get("node_id"), "原命题未落库：%r" % (res,)
    assert res.get("write_failed") is False, "原命题写入失败：%r" % (res,)
    return res["node_id"]


def test_g1_negation_correction_is_stored_as_new_node():
    """G1：否定更正另立新节点，且更正文本落库。"""
    e = _eng()
    old = _seed(e)
    assert _n_nodes(e) == 1
    raw = LayeredStore.char_bigram_jaccard(B_NEG, A)
    assert raw >= 0.95, "用例前提：raw jaccard 必须 ≥0.95（旧实现正是因此误并），实得 %s" % raw
    assert SpacetimeMemoryEngine._assertion_divergence(B_NEG, A) == "negation"
    hit_node, hit_score = _hit(e, B_NEG)
    assert hit_node is not None and hit_node.id == old and hit_score >= 0.95, \
        "用例前提：更正文本必须命中旧节点且命中分 ≥0.95，实得 %s/%s" % (hit_node, hit_score)

    res = e.longterm_snapshot(B_NEG, tags=["pcorr"], importance_hint=0.6)
    assert res.get("write_failed") is False, res
    assert res["node_id"] != old, "更正被并进旧节点（#481 缺陷形态）：%r" % (res,)
    assert _n_nodes(e) == 2, "应另立新节点（1→2），实得 %s" % _n_nodes(e)
    assert _rows_of(e, B_NEG) == 1, "更正文本必须落库（旧行为为 0 行）"
    old_node = e.store.get_node(old)
    assert old_node.content == A, "旧节点正文被改写"


def test_g2_correction_recallable_by_its_own_text():
    """G2：以更正原话 recall → 新节点、正文 == 更正文本。"""
    e = _eng()
    old = _seed(e)
    res = e.longterm_snapshot(B_NEG, tags=["pcorr"], importance_hint=0.6)
    new = res["node_id"]
    node, score = _hit(e, B_NEG)
    assert node is not None and node.id == new, \
        "更正原话未召回新节点（旧行为返回旧节点=错误事实）：%s" % (node.id if node else None,)
    assert node.content == B_NEG, "召回正文不是更正文本"
    assert score >= 0.95
    assert node.id != old


def test_g3_pure_duplicate_still_merges():
    """G3：纯重复文本仍按原逻辑合并——节点数不变、node_id 不变。"""
    e = _eng()
    old = _seed(e)
    res = e.longterm_snapshot(A, tags=["pcorr"], importance_hint=0.6)
    assert res["node_id"] == old, "纯重复未合并：%r" % (res,)
    assert _n_nodes(e) == 1, "纯重复新增了节点：%s" % _n_nodes(e)
    assert res["write_failed"] is False
    merged = e.store.get_node(old)
    assert abs(_importance(e, old) - 0.6) < 1e-9, "合并腿 importance 取大语义被改"
    assert "pcorr" in (merged.tags or []), "合并腿 tags 并入语义被改"


def test_g4_no_divergence_rewrite_still_merges():
    """G4：无分歧换说法（jaccard 同级）仍合并——新闸不误杀。"""
    e = _eng()
    old = _seed(e)
    assert LayeredStore.char_bigram_jaccard(B_NEUTRAL, A) >= 0.95
    assert SpacetimeMemoryEngine._assertion_divergence(B_NEUTRAL, A) == ""
    res = e.longterm_snapshot(B_NEUTRAL, tags=["pcorr"], importance_hint=0.6)
    assert res["node_id"] == old and _n_nodes(e) == 1, res


def test_g5_old_node_not_promoted():
    """G5：旧节点不被提权——hint=0.6 与 hint=0.9 两种。"""
    for hint in (0.6, 0.9):
        e = _eng()
        old = _seed(e, hint=0.5)
        assert abs(_importance(e, old) - 0.5) < 1e-9
        res = e.longterm_snapshot(B_NEG, tags=["pcorr"], importance_hint=hint)
        assert res["node_id"] != old, "hint=%s：更正被并进旧节点" % hint
        assert abs(_importance(e, old) - 0.5) < 1e-9, \
            "hint=%s：旧节点被提权到 %s（旧行为 0.5→%s）" % (hint, _importance(e, old), hint)


def test_g6_old_node_not_protected_no_forget():
    """G6：旧节点不打 no_forget/不进 protections；hint≥0.7 时保护的是新节点。"""
    e = _eng()
    old = _seed(e, hint=0.5)
    res = e.longterm_snapshot(B_NEG, tags=["pcorr"], importance_hint=0.9)
    new = res["node_id"]
    assert new != old
    old_node = e.store.get_node(old)
    assert "no_forget" not in (old_node.tags or []), \
        "旧节点被打 no_forget（错误记忆被永久保护）：%s" % (old_node.tags,)
    assert old not in set(e.store.get_protected_nodes()), "旧节点进了 protections 表"
    assert res.get("protected") is True, "长期层更正未保护新节点：%r" % (res,)
    assert new in set(e.store.get_protected_nodes()), "新节点未进 protections 表"


def test_g7_threshold_and_tag_bonus_untouched():
    """G7：raw jaccard <0.95 但含 tag_bonus 的命中分 ≥0.95、无分歧 → 仍合并。"""
    cand = None
    for old_s, new_s in (("低资源的小样本", "资源"), ("低资源的小样本场景", "资源场景"),
                         ("低资源的小样本场景下", "资源场景下"),
                         ("误差区间与上一次的测量结果", "误差区间与上次测量结果")):
        c = A.replace(old_s, new_s)
        if c == A or TAG_WORD not in c:
            continue
        raw = LayeredStore.char_bigram_jaccard(c, A)
        if 0.90 <= raw < 0.95:
            cand = (c, raw)
            break
    assert cand is not None, "未能构造 tag_bonus 用例（raw 需落在 [0.90,0.95)）"
    text, raw = cand
    assert SpacetimeMemoryEngine._assertion_divergence(text, A) == "", "该用例不应有分歧"

    e = _eng()
    old = _seed(e, tags=[TAG_WORD], hint=0.5)
    node, score = _hit(e, text)
    assert node is not None and node.id == old
    assert raw < 0.95 <= score, \
        "用例前提：raw(%s) < 0.95 ≤ 命中分(%s)，tag_bonus 未生效" % (raw, score)
    res = e.longterm_snapshot(text, tags=["pcorr"], importance_hint=0.6)
    assert res["node_id"] == old and _n_nodes(e) == 1, \
        "tag_bonus 命中的无分歧文本必须仍合并：%r" % (res,)


def test_g8_engine_judge_is_the_single_source():
    """G8：gate 的判据来自引擎侧 `_assertion_divergence`（同源，不另造一套）。"""
    from lingshu.core.longterm_gate import LongTermMemoryGate
    e = _eng()
    gate = LongTermMemoryGate(e)
    assert gate._assertion_divergence(B_NEG, A) == "negation"
    assert gate._assertion_divergence(A, A) == ""
    assert gate._assertion_divergence(B_NEG, A) == \
        SpacetimeMemoryEngine._assertion_divergence(B_NEG, A)


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    bad = 0
    for fn in fns:
        try:
            fn()
            print("PASS %s" % fn.__name__)
        except Exception:
            bad += 1
            print("FAIL %s" % fn.__name__)
            traceback.print_exc()
    print("%d/%d passed" % (len(fns) - bad, len(fns)))
    sys.exit(1 if bad else 0)
