# -*- coding: utf-8 -*-
"""结构层记录惯例的 extra_tags 守卫（对应 issue #571）。

`_write_structure_record` 是 #329 收口后的唯一真源。它的 PRIMARY 分支调用
`add_structure_node(record, importance=0.9)`（该原语**无 tags 形参**），原先只在
事后补 `tag` 一个 ⇒ `extra_tags` 在主控角色下被静默丢弃；SUB 分支却正常落库。
本守卫钉住「形参在两个角色分支都生效」。
"""
import json
import os
import shutil
import tempfile

import pytest

from lingshu.core.core import Role, SpacetimeMemoryEngine  # noqa: E402


@pytest.fixture()
def prim():
    os.environ["AEIS_DESIGNER_KEY"] = "k-demo-571"
    d = tempfile.mkdtemp()
    e = SpacetimeMemoryEngine(os.path.join(d, "p.db"), role=Role.PRIMARY)
    yield e
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture()
def sub():
    os.environ["AEIS_DESIGNER_KEY"] = "k-demo-571"
    d = tempfile.mkdtemp()
    e = SpacetimeMemoryEngine(os.path.join(d, "s.db"), role=Role.SUB)
    yield e
    shutil.rmtree(d, ignore_errors=True)


def _tags(engine, content):
    """读回 tags 列——该列存的是 JSON 文本（`_dumps_tags` 写入），需解析。"""
    row = engine.store.conn.execute(
        "SELECT tags FROM nodes WHERE content=?", (content,)).fetchone()
    if not row or row[0] is None:
        return None
    return json.loads(row[0])


def test_extra_tags_land_on_primary_structure_record(prim):
    """PRIMARY + extra_tags ⇒ 结构层节点必须带上 extra_tags。"""
    prim._write_structure_record("迁移事件A", "migration", extra_tags=["v1.7"])
    tags = _tags(prim, "迁移事件A")
    assert tags is not None, tags
    assert "v1.7" in tags, tags
    assert "migration" in tags, tags
    assert "structure" in tags, tags


def test_extra_tags_land_on_sub_copy(sub):
    """SUB + extra_tags ⇒ 知识层副本带 extra_tags 与 pending_sync。"""
    sub._write_structure_record("迁移事件B", "migration", extra_tags=["v1.7"])
    tags = _tags(sub, "迁移事件B")
    assert tags is not None, tags
    assert "v1.7" in tags, tags
    assert "pending_sync" in tags, tags


def test_both_roles_keep_the_same_extra_tags(prim, sub):
    """同一调用 ⇒ 两个角色落库的标签集只在层标签 / pending_sync 上有差。"""
    prim._write_structure_record("同参迁移", "migration", extra_tags=["v1.7", "d3"])
    sub._write_structure_record("同参迁移", "migration", extra_tags=["v1.7", "d3"])
    tp, ts = set(_tags(prim, "同参迁移")), set(_tags(sub, "同参迁移"))
    extra = {"v1.7", "d3", "migration"}
    assert extra <= tp, tp
    assert extra <= ts, ts
    # 角色差异只在 structure / pending_sync
    assert tp - ts == {"structure"}, (tp, ts)
    assert ts - tp == {"pending_sync"}, (tp, ts)


def test_no_extra_tags_still_records_tag(prim):
    """不传 extra_tags 时行为不变（回归护栏）。"""
    prim._write_structure_record("注意力记录", "attention_weight")
    tags = _tags(prim, "注意力记录")
    assert tags is not None and "attention_weight" in tags, tags


def test_multiple_extra_tags_all_land(prim):
    """多个 extra_tags 都要落上。"""
    prim._write_structure_record("多标签事件", "migration",
                                 extra_tags=["v1.7", "b2", "flag-x"])
    tags = _tags(prim, "多标签事件")
    for t in ("v1.7", "b2", "flag-x"):
        assert t in tags, (t, tags)


def test_no_duplicate_tags(prim):
    """tag_node 幂等：不得产生重复标签。"""
    prim._write_structure_record("去重事件", "migration", extra_tags=["v1.7"])
    tags = _tags(prim, "去重事件")
    assert tags is not None
    assert len(tags) == len(set(tags)), tags


def test_guard_is_discriminating(prim):
    """判别力锚定：修前实现只在事后补 `tag`，extra_tags 必然缺席。"""
    prim._write_structure_record("判别事件", "migration", extra_tags=["v1.7"])
    tags = set(_tags(prim, "判别事件"))
    # 修前语义：tags 里只有 structure + migration
    assert "migration" in tags
    # 现语义：extra_tags 已在
    assert "v1.7" in tags, tags
