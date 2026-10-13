# -*- coding: utf-8 -*-
"""归档层回归测试：归档状态机必须闭合（对应 issue #448）。

此前 forget_advisor 打下的 `archived` 标签没有任何公开 API 能清除，
且 `if "archived" in tags: continue` 让节点被永久跳过 ⇒ 「可逆」不成立。
"""
import os
import sys
import tempfile

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lingshu.core.core import SpacetimeMemoryEngine  # noqa: E402


@pytest.fixture()
def engine():
    db = os.path.join(tempfile.mkdtemp(), "archive.db")
    e = SpacetimeMemoryEngine(db)
    yield e
    try:
        os.remove(db)
    except OSError:
        pass


def _archived_node(e, importance=0.05, content="归档候选"):
    n = e.add_context(content, importance=importance)
    e.forget_advisor(stale_days=0, low_value=0.2)
    return n


def test_untag_node_is_symmetric_to_tag_node(engine):
    """untag_node 能清除 tag_node 加上的标签（幂等）。"""
    n = engine.add_context("标签对称性", importance=0.5)
    engine.store.tag_node(n.id, "probe_tag")
    assert "probe_tag" in engine.store.get_node(n.id).tags

    engine.store.untag_node(n.id, "probe_tag")
    assert "probe_tag" not in engine.store.get_node(n.id).tags

    # 幂等：重复移除不报错
    engine.store.untag_node(n.id, "probe_tag")
    assert "probe_tag" not in engine.store.get_node(n.id).tags


def test_restore_node_clears_archived_tag(engine):
    """restore_node 必须清掉 archived 标签 —— 这是本issue 的核心。"""
    n = _archived_node(engine)
    row = engine.store.get_node(n.id)
    assert "archived" in row.tags

    assert engine.store.restore_node(n.id) is True
    row = engine.store.get_node(n.id)
    assert "archived" not in row.tags
    # 也不该再被 get_nodes_by_tag('archived') 计入
    assert n.id not in [x.id for x in engine.store.get_nodes_by_tag("archived", limit=50)]


def test_restore_node_restores_pre_archive_importance(engine):
    """归档前的 importance 应当被还原（0.05 -> archived 0.1 -> 回 0.05）。"""
    n = _archived_node(engine, importance=0.05)
    assert round(engine.store.get_node(n.id).importance, 3) == 0.1

    engine.store.restore_node(n.id)
    assert round(engine.store.get_node(n.id).importance, 3) == 0.05
    # 内部用的 pre_archive_* 标签不该留在库里
    assert not any(t.startswith("pre_archive_importance=")
                   for t in engine.store.get_node(n.id).tags)


def test_archive_restore_roundtrip_is_closed(engine):
    """归档 -> 还原 往返后，tags 与 importance 回到初态。"""
    n = engine.add_context("往返一致", importance=0.05)
    before_tags = set(engine.store.get_node(n.id).tags)
    before_imp = round(engine.store.get_node(n.id).importance, 6)

    engine.forget_advisor(stale_days=0, low_value=0.2)
    assert "archived" in engine.store.get_node(n.id).tags

    engine.store.restore_node(n.id)
    after = engine.store.get_node(n.id)
    assert round(after.importance, 6) == before_imp
    assert set(after.tags) == before_tags


def test_restore_node_is_noop_when_not_archived(engine):
    """未归档的节点调restore_node 返回 False，不应改动数据。"""
    n = engine.add_context("没归档", importance=0.7)
    assert engine.store.restore_node(n.id) is False
    assert "archived" not in engine.store.get_node(n.id).tags


def test_restored_node_is_considered_again_by_forget_advisor(engine):
    """还原后不应再被 'archived' 标签永久跳过（:5454 的语义）。"""
    n = _archived_node(engine)
    engine.store.restore_node(n.id)

    # 再跑一次：此时 importance 已回到 0.05（低于 low_value），应重新归档
    out = engine.forget_advisor(stale_days=0, low_value=0.2)
    assert out["archived"] == 1
    assert "archived" in engine.store.get_node(n.id).tags


# ---------------------------------------------------------------------------
# 保护闸（作者独立复核建议 1/3：新增写原语必须对齐 delete_node 的保护口径）
# ---------------------------------------------------------------------------

def test_untag_node_refuses_to_strip_no_forget(engine):
    """剥 `no_forget` 标签等同解除保护 ⇒ 默认必须拒。"""
    n = engine.add_context("受保护节点", importance=0.5)
    engine.store.tag_node(n.id, "no_forget")
    assert engine.store._is_forget_protected(engine.store.get_node(n.id)) is True

    # 默认拒
    assert engine.store.untag_node(n.id, "no_forget") is False
    assert "no_forget" in engine.store.get_node(n.id).tags
    assert engine.store._is_forget_protected(engine.store.get_node(n.id)) is True

    # 显式 override 才放行
    assert engine.store.untag_node(n.id, "no_forget", override=True) is True
    assert "no_forget" not in engine.store.get_node(n.id).tags
    assert engine.store._is_forget_protected(engine.store.get_node(n.id)) is False


def test_untag_node_protection_bypass_cannot_reach_delete(engine):
    """核心旁路：untag 剥不掉 no_forget ⇒ delete_node 仍旧拒删。"""
    n = engine.add_context("不可删节点", importance=0.5)
    engine.store.tag_node(n.id, "no_forget")

    engine.store.untag_node(n.id, "no_forget")   # 静默失败
    assert engine.store.delete_node(n.id) is False
    assert engine.store.get_node(n.id) is not None

    # 只有显式 override 后才能走删除路径
    engine.store.untag_node(n.id, "no_forget", override=True)
    assert engine.store.delete_node(n.id) is True


def test_restore_node_refuses_immutable_layer(engine):
    """不可变层（ANCHOR/STRUCTURE/SELF）节点拒还原，返回 False。"""
    from lingshu.core.core import STNode, MemoryLayer, ConditionSpace
    import time

    for layer in engine.store.IMMUTABLE_LAYERS:
        node = STNode(
            id="immut_" + layer.name.lower(), content="不可变层节点",
            modality="text", spatial_coordinates={"x": 0.0},
            temporal_coordinate=time.time(),
            condition_space=ConditionSpace(
                observation_position="桌面", observation_tool="文本语义分析",
                time_window=(0.0, 0.0), existence_constraint="（未声明）"),
            importance=0.9, layer=layer,
            tags=["archived", "pre_archive_importance=0.1"])
        engine.store.add_node(node)

        assert engine.store.restore_node(node.id) is False, layer
        row = engine.store.get_node(node.id)
        # 数据纹丝不动：importance 仍是 0.9，标签仍是归档态
        assert round(row.importance, 6) == 0.9, layer
        assert "archived" in row.tags, layer


def test_pre_archive_importance_has_no_type_drift(engine):
    """原 importance 为 int 0 时，还原后不得因 repr 漂移成 float。"""
    n = engine.add_context("整数重要度", importance=0)
    assert isinstance(n.importance, int)

    engine.forget_advisor(stale_days=0, low_value=0.2)
    tags = engine.store.get_node(n.id).tags
    marker = [t for t in tags if t.startswith("pre_archive_importance=")]
    assert marker, tags
    # %.6g 固定格式化：int 0 落成 "0"，不含小数点
    assert marker[0] == "pre_archive_importance=0", marker

    engine.store.restore_node(n.id)
    row = engine.store.get_node(n.id)
    assert row.importance == 0
    assert not any(t.startswith("pre_archive_importance=") for t in row.tags)