#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_issue462_known_snapshot_incremental_guard —— lingshu #462 守卫（5 组判据）。

缺陷（P1）：`LongTermMemoryGate._known_snapshot()` 的全库词表缓存用
「签名三项都变大」当作「只是追加」的判据（节点数 / MAX(rowid) / 正文总长），
而增量分支只 `SELECT id, content FROM nodes WHERE rowid > old_max` 并入新行、
从不剔除已删节点。于是「删 1 条再增 >=2 条」后三项同样全变大 ⇒ 误走增量，
**被删节点的 gram 永久留在参考集里**：被遗忘的内容再次出现时 novelty 被算成
0.0、写入层级从 knowledge 掉到 context（新知识被当旧知识降级）。
修法：增量并入之前加「前缀不变校验」——查 `rowid <= old_max` 前缀的
(COUNT(*), MAX(rowid), SUM(LENGTH(content)))，与缓存时刻三项全等才走增量，
否则全量重建。纯追加时前缀不动，#389 的快路径（不重切全库正文）不被回退。

判据（全部 :memory: 隔离库，不落盘、不触在役记忆库）：
  G1 精确复现：add A -> delete_node(A) -> add B,C ⇒ 被删节点的 gram 不在 union
     也不在 per_node，novelty(A) 回到 1.0，evaluate(A) 的 layer 回到 knowledge。
  G2 阴性对照：纯追加（无删除）⇒ 仍复用**同一** union 对象（快路径未被禁用）。
  G3 改写旧行：UPDATE 已有行 content（count/rowid 不变、正文总长变化）**再追加
     1 条**（全表三项仍全变大、旧判据被掩盖）⇒ 被识破、旧 gram 被剔除、新 gram 入集。
  G4 签名未变：连续两次取词表返回同一 union/counts/per_node 对象。
  G5 定点变异自证：把前缀校验改成恒真（LINGSHU462_MUTATE=1 时 monkeypatch
     `_prefix_intact` 为恒真）⇒ G1/G3 必须变红，本件断言缺陷形态复现。

运行（lingshu 仓根，绿）：
  PYTHONUTF8=1 python -X utf8 -m pytest tests/test_issue462_known_snapshot_incremental_guard.py -q --no-header
变异（红，预期 G1/G3 failed、G2/G4/G5 passed）：
  LINGSHU462_MUTATE=1 PYTHONUTF8=1 python -X utf8 -m pytest tests/test_issue462_known_snapshot_incremental_guard.py -q --no-header
"""
from __future__ import annotations

import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import lingshu.core.longterm_gate as lg  # noqa: E402
from lingshu.core.core import SpacetimeMemoryEngine  # noqa: E402
from lingshu.core.longterm_gate import LongTermMemoryGate  # noqa: E402

# 与母栏复现脚本 probe462.py 同一条读数（A 15 字；B+C 29 字 > A，保证三项全变大）
A = "紫晶密钥环在第七码头的存放位置"
B = "琥珀色铜制罗盘校准记录在仓库西侧"
C = "青灰色陶罐封蜡批次编号在东门"
D = "靛蓝色琉璃沙漏的砂流速率标定记录存放在地窖第七层"   # 改写用：更长且词不同

MUTATE = os.environ.get("LINGSHU462_MUTATE") == "1"


@pytest.fixture(autouse=True)
def _arm_mutation(monkeypatch):
    """LINGSHU462_MUTATE=1 ⇒ 把前缀校验改成恒真（模拟「去掉那一行」）。"""
    if MUTATE:
        monkeypatch.setattr(lg, "_prefix_intact", lambda *a, **k: True)
    yield


def _eng():
    return SpacetimeMemoryEngine(":memory:")


def _counts(engine):
    return tuple(engine.store.conn.execute(
        "SELECT count(*), COALESCE(MAX(rowid),0), "
        "COALESCE(SUM(LENGTH(content)),0) FROM nodes").fetchone())


# ---------------- G1：精确复现 ----------------

def test_g1_deleted_node_grams_evicted_and_novelty_restored():
    """删 1 增 2 后：被删节点的 gram 必须从 union / per_node 剔除，
    novelty(A) 回到 1.0、evaluate(A) 回到 knowledge 层。"""
    e = _eng()
    g = LongTermMemoryGate(e)          # 冷缓存：独立 gate，保证首次全量建缓存
    n = e.add_perception(A, importance=0.5)
    gA = g._core_grams(A)
    u1 = g._known_grams()
    assert gA and gA <= u1, "STEP1 前提：A 已入库且其 gram 在参考集里"
    assert _counts(e) == (1, 1, len(A)), _counts(e)

    assert e.store.delete_node(n.id) is True
    assert e.store.get_node(n.id) is None

    e.add_perception(B, importance=0.5)
    e.add_perception(C, importance=0.5)

    union, counts, per_node, sig = g._known_snapshot()
    # 注意：本仓 nodes 表删空后 rowid 会从 1 重新分配，故 MAX(rowid)=2 而非 3
    # （母栏 probe462 的 STEP3 实测 sig=(2,2,30)）；判据用「节点数 + 正文总长」。
    assert sig[0] == 2 and sig[2] == len(B) + len(C), sig
    assert len(per_node) == 2, sorted(per_node)
    assert not (gA <= union), "被删节点的 gram 仍在 union（#462 未修）"
    assert str(n.id) not in per_node, "被删节点仍在 per_node（#462 未修）"
    assert all(counts.get(gram, 0) == 0 for gram in gA), "counts 仍留被删节点的词"
    assert g._core_grams(B) <= union and g._core_grams(C) <= union, "全量重建不完整"

    assert g._novelty(A) == 1.0, g._novelty(A)
    ev = g.evaluate(A, "snapshot")
    assert ev["features"]["novelty"] == 1.0, ev
    assert ev["layer"] == "knowledge", ev
    assert ev["layer"] != "context", ev


# ---------------- G2：阴性对照（纯追加仍走快路径） ----------------

def test_g2_pure_append_still_reuses_cached_union_object():
    """无删除的纯追加必须仍复用同一 union/counts/per_node 对象
    （修复不得把 #389 的增量快路径废掉）。"""
    e = _eng()
    g = LongTermMemoryGate(e)
    for i in range(5):
        e.add_perception("高权无关条目第%03d号" % i, importance=0.9)
    u1, c1, p1, s1 = g._known_snapshot()
    n1 = len(p1)          # 增量路径复用同一对象，必须先取长度（否则读到变异后的值）
    e.add_perception("追加的一条全新知识记录用于增量验证", importance=0.5)
    u2, c2, p2, s2 = g._known_snapshot()
    assert u2 is u1, "纯追加必须走增量并入（union 对象不变）"
    assert c2 is c1 and p2 is p1, "counts/per_node 也必须复用缓存对象"
    assert s2[0] == s1[0] + 1 and s2[1] > s1[1] and s2[2] > s1[2], (s1, s2)
    assert len(p2) == n1 + 1, (n1, len(p2))


# ---------------- G3：改写旧行 ----------------

def test_g3_rewritten_old_row_is_detected():
    """UPDATE 已有行 content + 追加 1 条 ⇒ 改写必须被识破、旧 gram 被剔除。

    注意（判据前提）：**只**改写而不追加时，节点数不增 ⇒ 旧判据的
    `sig[0] > old_cnt` 已为假、本就落全量重建，改写的破坏力被「节点数」掩盖；
    真正被掩盖的是「改写 + 追加」——三项全变大，旧判据误走增量分支。
    """
    e = _eng()
    g = LongTermMemoryGate(e)
    n = e.add_perception(A, importance=0.5)
    old_grams = g._core_grams(A)
    g._known_snapshot()
    assert _counts(e) == (1, 1, len(A))

    # 改写旧行：count/rowid 不变，仅正文总长变化
    e.store.conn.execute("UPDATE nodes SET content=? WHERE id=?", (D, n.id))
    e.store.conn.commit()
    assert _counts(e) == (1, 1, len(D)) and len(D) != len(A), _counts(e)

    # 追加 1 条：全表三项于是全变大（旧判据被掩盖）
    e.add_perception(B, importance=0.5)
    sig_after = _counts(e)
    assert sig_after == (2, 2, len(D) + len(B)), sig_after

    new_grams = g._core_grams(D)
    union, counts, per_node, sig = g._known_snapshot()
    assert not (old_grams <= union), "改写前旧正文的 gram 仍在参考集（#462 未修）"
    assert not (old_grams <= set(per_node.get(str(n.id), set()))), per_node.get(str(n.id))
    assert new_grams <= union, "改写后的新正文 gram 未入参考集"
    assert per_node[str(n.id)] == new_grams, per_node.get(str(n.id))
    assert all(counts.get(gram, 0) == 0 for gram in old_grams - new_grams)


# ---------------- G4：签名未变仍复用 ----------------

def test_g4_unchanged_signature_reuses_cache():
    """连续两次取词表（库未动）必须返回同一批缓存对象。"""
    e = _eng()
    g = LongTermMemoryGate(e)
    e.add_perception(A, importance=0.5)
    u1, c1, p1, s1 = g._known_snapshot()
    u2, c2, p2, s2 = g._known_snapshot()
    assert u1 is u2 and c1 is c2 and p1 is p2, "签名未变时必须复用缓存"
    assert s1 == s2 == _counts(e)


# ---------------- G5：定点变异自证 ----------------

def test_g5_mutation_selfproof_prefix_check_is_load_bearing():
    """前缀校验那一行是承重的：恒真 ⇒ 缺陷复现（G1/G3 变红）。

    正常模式：`_prefix_intact` 对被删前缀必须判 False，且被删 gram 被剔除。
    变异模式（LINGSHU462_MUTATE=1）：它恒真 ⇒ 被删 gram 留在参考集里。
    """
    e = _eng()
    g = LongTermMemoryGate(e)
    n = e.add_perception(A, importance=0.5)
    gA = g._core_grams(A)
    g._known_snapshot()
    cached = g._known_sig
    assert e.store.delete_node(n.id) is True
    e.add_perception(B, importance=0.5)
    e.add_perception(C, importance=0.5)

    assert lg._prefix_intact(e.store.conn, cached) == MUTATE, (
        "前缀校验未被变异/未被击穿，变异自证不成立")
    union = g._known_grams()
    if MUTATE:
        assert gA <= union, "变异模式下缺陷应复现（此时 G1/G3 红）"
    else:
        assert not (gA <= union), "前缀校验必须剔除被删节点的 gram"
