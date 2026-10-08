# -*- coding: utf-8 -*-
"""self_check 自检诚实性回归。

self_ok / self_model_exists 必须以持久化 SELF 层为准，不能用 in-memory
self_model.identity 兜底——后者在 __init__ 无条件构造、identity 恒为非空
默认值（"协议实例"），作判据会使 self_ok 恒 True，无法发现失忆/无自我，
且与同函数 anchor_ok / structure_ok 的纯持久化口径不一致。
"""


def test_self_check_empty_self_reports_not_ok():
    """全新引擎 SELF 层 0 节点 → self_ok 必须为 False（修复前恒 True）。"""
    from lingshu.core.core import SpacetimeMemoryEngine

    e = SpacetimeMemoryEngine(":memory:")
    sc = e.self_check()
    assert sc["self_ok"] is False
    assert sc["self_model_exists"] is False


def test_self_check_after_update_self_reports_ok():
    """update_self 写入 SELF 层后 → self_ok 应转为 True。"""
    from lingshu.core.core import SpacetimeMemoryEngine

    e = SpacetimeMemoryEngine(":memory:")
    e.update_self({"goal": "测试目标"})
    sc = e.self_check()
    assert sc["self_ok"] is True
    assert sc["self_model_exists"] is True
