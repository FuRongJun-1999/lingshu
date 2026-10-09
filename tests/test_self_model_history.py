# -*- coding: utf-8 -*-
"""SelfModel.history 上界钳制回归。

SelfModel.update（update_self 经此入口）往 self.history append，原代码无 MAX/钳制
→ 随 update_self 调用线性无界增长（长程运行 OOM 慢性泄漏）。修复后对齐同类
trust_history：append 后钳到 HISTORY_MAX，保留最近尾部（FIFO）。

对照：trust_history 已有 TRUST_HISTORY_MAX=30 且在 update_trust_state 里钳制，
history 修复前两者皆无——同类内一处有界一处无界 = 遗漏。
"""


def test_history_capped_after_many_updates():
    """update_self 超过 HISTORY_MAX 次 → history 长度钳到 HISTORY_MAX（修复前无界 == N）。"""
    from lingshu.core.core import SpacetimeMemoryEngine, SelfModel

    e = SpacetimeMemoryEngine(":memory:")
    N = SelfModel.HISTORY_MAX + 50
    for i in range(N):
        e.update_self({"current_goal": f"g{i}"})
    assert len(e.self_model.history) == SelfModel.HISTORY_MAX, (
        f"history 应钳到 HISTORY_MAX={SelfModel.HISTORY_MAX}，实际 {len(e.self_model.history)}")


def test_history_keeps_recent_tail():
    """钳制后保留最近 HISTORY_MAX 条（尾部 FIFO），最早一批被淘汰。"""
    from lingshu.core.core import SpacetimeMemoryEngine, SelfModel

    e = SpacetimeMemoryEngine(":memory:")
    N = SelfModel.HISTORY_MAX + 10
    for i in range(N):
        e.update_self({"current_goal": f"g{i}"})
    # 尾部应是最近一次调用
    last = e.self_model.history[-1]
    assert last["changes"].get("current_goal") == f"g{N - 1}"
    # 头部不应是最早的 g0（已被淘汰）
    first = e.self_model.history[0]
    assert first["changes"].get("current_goal") != "g0"


def test_trust_history_still_capped():
    """对照：trust_history 仍钳到 TRUST_HISTORY_MAX=30（修复未碰它，回归保护）。"""
    from lingshu.core.core import SpacetimeMemoryEngine, SelfModel

    e = SpacetimeMemoryEngine(":memory:")
    for i in range(100):
        e.update_trust_state(t_total=float(i), round_no=i, p_trust=0.5, p_gap=0.5)
    assert len(e.self_model.trust_history) == SelfModel.TRUST_HISTORY_MAX
