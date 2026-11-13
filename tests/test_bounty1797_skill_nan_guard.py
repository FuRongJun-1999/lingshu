"""dsh-bounty #1797 守卫：add_skill 对 confidence 的非有限值 fail-closed 拒收。

缺陷（修复前实测，锚定 ed8700d）：`add_skill(confidence=float('nan'))` 零校验
直插 ⇒ NaN 被 SQLite 存成 NULL；此后 `update_skill_confidence` 的
`MIN(1.0, MAX(0.0, confidence + ?))` 对 NULL 恒回 NULL——该技能卡的增减信
面**永久失效**（docstring 承诺的 clamp 永不兑现），`search_skills` 的
`ORDER BY confidence DESC` 里 NULL 恒沉底。症状类＝#185（nodes/edges 两表
已收口），skills 表漏网。

判据来源：dsh-bounty #1797（含逐字读数）；闸＝#206 的 `_require_finite`
单点（fail-closed 拒收，写侧读侧共用）。本文件是「回退即红」对照：
撤掉 add_skill 入口的 `_require_finite` 行，G1 失败。
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingshu.core.core import SpacetimeMemoryEngine  # noqa: E402


@pytest.fixture()
def e():
    return SpacetimeMemoryEngine(":memory:")


# ------------------------------------------------ G1 非有限值 fail-closed

def test_g1_nan_inf_and_nonnumber_rejected(e):
    for bad in (float("nan"), float("inf"), float("-inf"), "abc", None):
        with pytest.raises(ValueError):
            e.store.add_skill("备份", "描述", "步骤", confidence=bad)
    # 拒收后库里没有半写行
    assert e.store.get_skill_by_name("备份") is None


def test_g1b_rejected_row_not_partially_written(e):
    """拒收发生在 INSERT 之前——skills 表行数为 0。"""
    with pytest.raises(ValueError):
        e.store.add_skill("x", "y", "z", confidence=float("nan"))
    cur = e.store.conn.cursor()
    cur.execute("SELECT COUNT(*) FROM skills")
    assert cur.fetchone()[0] == 0


# ------------------------------------- G2 防过修正向：合法路径逐字不变

def test_g2_legal_confidence_paths_unchanged(e):
    sid = e.store.add_skill("合法", "描述", "步骤")           # 缺省 0.5
    assert e.store.get_skill_by_name("合法")["confidence"] == 0.5
    sid0 = e.store.add_skill("零", "d", "p", confidence=0)    # 边界 0
    sid1 = e.store.add_skill("壹", "d", "p", confidence=1)    # 边界 1
    assert e.store.get_skill_by_name("零")["confidence"] == 0
    assert e.store.get_skill_by_name("壹")["confidence"] == 1
    # 增减信 clamp 语义不变
    e.store.update_skill_confidence(sid, 0.5)
    assert e.store.get_skill_by_name("合法")["confidence"] == 1.0
    e.store.update_skill_confidence(sid0, -0.25)
    assert e.store.get_skill_by_name("零")["confidence"] == 0.0
    assert sid1  # 未用变量仅为落库断言铺垫
