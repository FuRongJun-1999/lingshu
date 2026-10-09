# -*- coding: utf-8 -*-
"""_track_identity eid 查重回归属回测。

_track_identity（world_model.py:112）对无 eid 观测生成 "wm_"+6 hex eid（16M 空间），
原代码不与既有 self.nodes 键查重——碰撞时 perceive（:147）走 matched 分支把两个
不同实体静默合并（长程生日悖论必然碰撞，≈4824 次即 >50%）。修复后生成 eid 查重
self.nodes，碰撞重生成（兜底 12 位）。
"""


def test_no_silent_merge_on_eid_collision():
    """大量匿名观测（唯一 category 必走随机生成）→ 节点数应 == 观测数（无合并）。

    修复前：碰撞走 matched 分支 → 节点数 < 观测数（静默合并）。
    """
    from lingshu.world.world_model import UnifiedWorldModel

    wm = UnifiedWorldModel(size=50, seed=123)
    N = 5000
    for i in range(N):
        wm.perceive([{"category": f"uniq_{i}", "pos": (float(i % 100), 0.0, 0.0)}])
    assert len(wm.nodes) == N, (
        f"应无合并：节点数应 == 观测数 {N}，实际 {len(wm.nodes)}（碰撞导致静默合并）")


def test_matched_branch_still_reuses_nearby_eid():
    """同类别 + 2 体素内的匿名观测应复用既有 eid（matched 路径未受修复影响）。"""
    from lingshu.world.world_model import UnifiedWorldModel

    wm = UnifiedWorldModel(size=50, seed=7)
    wm.perceive([{"category": "cat", "pos": (0.0, 0.0, 0.0)}])   # 首个匿名 → 生成
    eids_before = set(wm.nodes.keys())
    wm.perceive([{"category": "cat", "pos": (0.5, 0.0, 0.0)}])   # 同类 + 0.5 距离 → 应复用
    assert set(wm.nodes.keys()) == eids_before, (
        "近邻同类别应复用既有 eid（matched 路径不应被修复破坏）")
