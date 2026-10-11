# -*- coding: utf-8 -*-
"""#36「不可遗忘」保护在两条遗忘路径上失效 —— 守卫（lingshu/core/core.py）

缺陷（lingshu issue #36；核验件 `.tmp/wave_verify_mem.md` §三 实测成立）：
  ① `LayeredStore.decay_cycle` 的情境层删除段是**就地内联 DELETE**，既不查
     `protections` 表也不查 `no_forget` 标签 ⇒ 受保护 context 节点经 40 轮衰减被删
     （同函数上半段对**边**却已做保护排除——同函数内上下两段判据不一致）。
  ② `LayeredStore.enforce_context_cap` 纯 `created_at` 升序 FIFO 淘汰，零保护过滤
     ⇒ 受保护的最旧情境节点照样被删。
  ③ 保护登记行删除后不清理（悬空）。

判据来源：
  · 修法落点 —— 设计稿 `docs/plans/删除恢复持久化完整性_修复设计_v0.1.md` §3.3
    「保护判定下沉到唯一删除原语 `LayeredStore.delete_node`」（可一次覆盖 ②），
    衰减段单独加判据并与其共用同一谓词（§3.3 的「若不收敛」分支）。
  · 谓词口径 —— 设计稿 §3.3「可复用的既有机制（无需新造）：`get_protected_nodes()`
    ＋ `no_forget` 标签」；待裁清单 D-15「应用层单一谓词」。
  · FIFO 与保护冲突语义 —— 待裁清单 **D-23** 推荐「保护优先、允许暂时超限」。

断言组（抽掉任一修复即红）：
  P1 衰减路径：`protect_node` 过的情境节点经 40 轮 `decay_cycle` 仍存在；
     未保护的同类节点**被删**（防「一律不删」的假修复）。
  P2 FIFO 路径：受保护的最旧情境节点经 `enforce_context_cap` 仍存在；
     未保护的更旧节点被删（保护不豁免全部）。
  P3 悬空登记：删除路径一旦拒删，`get_protected_nodes()` 中每个 id 都有节点。
  P4 判据一致：同一节点在衰减路径与 FIFO 路径下**命运相同**（钉住「同函数上下段
     判据不一致」这类漏洞）。
  P5 `no_forget` 标签腿（无 `protections` 行登记）同样受保护。
  P6 拒删契约：`delete_node` 对受保护节点返回 **False**（bool 契约不变，对齐
     `tests/test_issue109_external_anchor_gate.py` 的既有断言形态）。

运行（仓根）：python -X utf8 -m pytest tests/test_issue36_forget_protection_guard.py -q --no-header
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingshu.core.core import SpacetimeMemoryEngine, MemoryLayer  # noqa: E402

# 低于情境层删除线的 importance（min_confidence=0.1）——缺陷形态下才足以触发删除。
_LOW = 0.12


def _live(e, node):
    return e.store.get_node(node.id) is not None


def test_p1_decay_path_respects_protection():
    """P1：衰减周期的情境层删除段必须查保护（旧：内联 DELETE 不查）。"""
    e = SpacetimeMemoryEngine(":memory:")
    kept = e.add_context("受保护的情境记忆", importance=_LOW)
    plain = e.add_context("未保护的情境记忆", importance=_LOW)
    e.store.protect_node(kept.id, "3.2 节不可遗忘")
    for _ in range(40):
        e.decay_cycle()
    assert _live(e, kept), "受保护情境节点经 40 轮衰减被删（旧缺陷行为）"
    assert not _live(e, plain), "未保护情境节点未被删除——衰减段整体失效（假修复）"


def test_p2_fifo_cap_respects_protection():
    """P2：情境层 FIFO 上限必须查保护（旧：纯 created_at 升序淘汰）。"""
    e = SpacetimeMemoryEngine(":memory:")
    nodes = [e.add_context("情境条目%d" % i) for i in range(5)]
    e.store.protect_node(nodes[0].id, "最旧且受保护")
    e.store.enforce_context_cap(2)
    assert _live(e, nodes[0]), "受保护（最旧）情境节点被 FIFO 淘汰（旧缺陷行为）"
    assert not _live(e, nodes[1]), "未保护的更旧节点未被淘汰——FIFO 整体失效（假修复）"


def test_p3_no_dangling_protection_rows():
    """P3：删除路径拒删后，保护登记不悬空。"""
    e = SpacetimeMemoryEngine(":memory:")
    nodes = [e.add_context("情境条目%d" % i) for i in range(5)]
    e.store.protect_node(nodes[0].id, "保护")
    e.store.enforce_context_cap(1)
    for nid in e.store.get_protected_nodes():
        assert e.store.get_node(nid) is not None, "保护登记悬空：%s" % nid


def test_p4_both_paths_agree_on_same_node():
    """P4：同一节点在衰减路径与 FIFO 路径下命运相同（判据一致性）。"""
    e = SpacetimeMemoryEngine(":memory:")
    n = e.add_context("受保护节点", importance=_LOW)
    e.store.protect_node(n.id, "保护")
    e.decay_cycle()
    survived_decay = _live(e, n)
    e.store.enforce_context_cap(0)
    survived_cap = _live(e, n)
    assert survived_decay == survived_cap is True, (
        "两路径判据不一致：decay 存活=%s / cap 存活=%s" % (survived_decay, survived_cap))


def test_p5_no_forget_tag_leg_protects():
    """P5：仅 `no_forget` 标签（无 protections 行）同样受保护。"""
    e = SpacetimeMemoryEngine(":memory:")
    n = e.add_context("仅带标签", importance=_LOW)
    e.store.tag_node(n.id, "no_forget")
    for _ in range(40):
        e.decay_cycle()
    assert _live(e, n), "no_forget 标签腿未生效"


def test_p6_delete_node_bool_contract_unchanged():
    """P6：受保护节点 ⇒ `delete_node` 返回 False（bool 契约不变）。"""
    e = SpacetimeMemoryEngine(":memory:")
    n = e.add_context("受保护", importance=_LOW)
    e.store.protect_node(n.id, "保护")
    assert e.store.delete_node(n.id) is False, "受保护节点未被拒删"
    assert _live(e, n)
