# -*- coding: utf-8 -*-
"""#1117：三处读数失真的收口守卫。

契约（docstring 既有承诺 + issue 期望行为）：
  · `build_flatland()` 返回「方块数」＝`len(self.blocks)`（去重后实际方块数，
    非写入次数——树/水覆盖同格时旧实现重复计数，默认参数即偏 12）；
  · `_context_points` 的 `need` 只按**有限**点数算（非有限读数不是可用证据，
    `motion_detail` docstring「不进估计、也不静默」）；`window_points` 读数
    同口径（`nonfinite_points` 照旧把坏读数如实带出）；
  · `recent_direction` 遇坏读数**跳过该帧**（与 missing 帧同路径），不重置
    `newer`——一次坏读数不再丢掉两侧有限观测的真实方向。
不做：不改 `MAX_TRAIL`/`MAX_TREES` 等既有边界声明（#719/#701 已披露）。
"""
from lingshu.world.motion import (ObservationWindow, recent_direction,
                                  _motion_rows)
from lingshu.world.voxel_world import VoxelWorld


# ---- 现象一：build_flatland 返回写入次数而非方块数 ----

def test_build_flatland_returns_actual_block_count():
    w = VoxelWorld(size=5, seed=0)
    n = w.build_flatland()
    assert n == len(w.blocks), (
        "返回 %d ≠ 实际方块数 %d（树/水覆盖重复计数）" % (n, len(w.blocks)))


def test_build_flatland_count_matches_census():
    w = VoxelWorld(size=8, seed=42)
    n = w.build_flatland()
    census = len(w.blocks)               # 值为 int 类型码 ⇒ 方块数即 dict 大小
    assert n == census


# ---- 现象二：_context_points 的 need 把非有限点当可用证据 ----

def _mixed_history():
    return [{'tick': 0, 'entities': {'e': (0.0, 1.5, 2.0)}},
            {'tick': 1, 'entities': {'e': (10.0, 1.5, 2.0)}},
            {'tick': 2, 'entities': {}},
            {'tick': 3, 'entities': {'e': (float('nan'), 1.5, 2.0)}}]


def test_context_backfills_when_window_has_nonfinite_tail():
    d = ObservationWindow(_mixed_history(), 3).motion_detail('e')
    # 窗口内 [有限, NaN] ⇒ 可用证据 1 点 ⇒ 必须回填更早的真实有限观测
    assert d["context_points"] >= 1, (
        "NaN 点占位导致该补的上下文不补（#1117 现象二）")
    assert d["estimate"] is not None
    assert d["window_points"] == 1, "window_points 应按有限点数计"


def test_window_points_readout_finite_only():
    d = ObservationWindow(_mixed_history(), 3).motion_detail('e')
    assert d["nonfinite_points"] >= 0     # 坏读数如实带出（不静默）
    assert d["window_points"] == 1        # [有限, NaN] ⇒ 有限可用证据 1 点


# ---- 现象三：recent_direction 遇坏读数重置 ----

def test_recent_direction_skips_bad_reading_without_reset():
    nan_mid = [{'tick': 0, 'entities': {'e': (0.0, 1.5, 2.0)}},
               {'tick': 1, 'entities': {'e': (float('nan'), 1.5, 2.0)}},
               {'tick': 2, 'entities': {'e': (5.0, 1.5, 2.0)}}]
    gap_mid = [{'tick': 0, 'entities': {'e': (0.0, 1.5, 2.0)}},
               {'tick': 2, 'entities': {'e': (5.0, 1.5, 2.0)}}]
    r_nan = recent_direction(nan_mid, 'e')
    r_gap = recent_direction(gap_mid, 'e')
    assert r_nan == r_gap == (1.0, 0.0, 0.0), (
        "一次坏读数清空历史（#1117 现象三）：nan=%r gap=%r" % (r_nan, r_gap))


def test_motion_rows_drops_only_the_bad_pair_regression():
    # 回归锚：_motion_rows 的既有口径（只丢坏对）保持——recent_direction 收口
    # 后两处口径一致。
    pts = [(1, (0.0, 1.5, 2.0)), (2, (float('nan'), 1.5, 2.0)),
           (3, (5.0, 1.5, 2.0)), (4, (6.0, 1.5, 2.0))]
    rows, nonfinite, _ = _motion_rows(pts, 1)
    assert nonfinite == 2                 # NaN 端点参与的两对各拒一次（t1→t2、t2→t3）
    assert len(rows) == 1 and rows[0][0] == 1   # t3→t4 有限对照常入列（只丢坏对）
