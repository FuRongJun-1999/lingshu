# -*- coding: utf-8 -*-
"""#1115/#1116：`world/motion` 两处输入防护收口守卫。

契约（模块既有口径 + issue 期望行为）：
  · #1115：非有限（NaN/±Inf）读数的位移对**不是已测间隔**——不计 `intervals`、
    不进 `samples`（模块口径「非有限坐标一律拒绝」在 `pair()` 侧同样成立），
    下游 `infer_patterns` / `learn` 不再在 `motion(a)[3]` 上 None 下标崩溃；
  · #1116：`window` 任何入口统一 `max(1, int(window))` 钳制（`window<=0` 曾
    直接 IndexError；`window=None` 保留「全历史」语义）；`window_start` 未知
    （None）时窗口外位移对不计入 `samples`。
不做：不改 #418 的稀疏窗口回填策略（报告者明示）。
"""
import math

import pytest

from lingshu.world.motion import (ObservationWindow, _motion_rows,
                                  estimate_motion)

H2 = [{'tick': 1, 'entities': {'e': (0.0, 1.5, 2.0)}},
      {'tick': 2, 'entities': {'e': (1.0, 1.5, 2.0)}}]


# ---- #1115：非有限读数计入 intervals/samples ----

def test_pair_nan_endpoints_not_counted_as_measured():
    history = [{'tick': i, 'entities': {'a': (float('nan'), 1.5, 2.0),
                                        'b': (float(i), 1.5, 2.0)}}
               for i in range(4)]
    ow = ObservationWindow(history, 4)
    assert ow.pair('a', 'b') == (0.0, 0, 0), (
        "含 NaN 端点的位移对被当作已测间隔（#1115）")
    assert ow.motion('a') is None       # 估计侧正确拒绝（既有口径）


def test_nan_pair_chain_keeps_previous_link():
    # NaN 对之后链不中断：后续有限对的位移以它之前最后有限帧为参照。
    # a: 0 → NaN → 3；b: 0 → 1 → 2（t2 两物不重合，位移向量有效）。
    history = [{'tick': 0, 'entities': {'a': (0.0, 1.5, 2.0), 'b': (0.0, 1.5, 2.0)}},
               {'tick': 1, 'entities': {'a': (float('nan'), 1.5, 2.0), 'b': (1.0, 1.5, 2.0)}},
               {'tick': 2, 'entities': {'a': (3.0, 1.5, 2.0), 'b': (2.0, 1.5, 2.0)}}]
    mean, samples, intervals = ObservationWindow(history, 3).pair('a', 'b')
    assert samples == 1 and intervals == 1   # NaN 对不计，有限对正常计


# ---- #1116：window<=0 与 window_start=None ----

@pytest.mark.parametrize("w", [0, -1, -5])
def test_estimate_motion_nonpositive_window_clamped(w):
    r = estimate_motion(H2, 'e', w)          # 改前：IndexError
    assert r == (1.0, 1.0, 1, (1.0, 0.0, 0.0))


@pytest.mark.parametrize("w", [-1, -5])
def test_observation_window_nonpositive_window_clamped(w):
    assert ObservationWindow(H2, w).motion('e') == (1.0, 1.0, 1, (1.0, 0.0, 0.0))


def test_normal_window_regression():
    assert estimate_motion(H2, 'e', 2) == (1.0, 1.0, 1, (1.0, 0.0, 0.0))


def test_motion_rows_unknown_window_start_counts_no_samples():
    # #1116 期望第二句：window_start 未知（None）时窗口外位移对不计入 samples。
    pts = [(1, (0.0, 1.5, 2.0)), (2, (1.0, 1.5, 2.0)), (3, (2.0, 1.5, 2.0))]
    rows, nonfinite, _ = _motion_rows(pts, None)
    assert nonfinite == 0
    assert all(r[-1] is False for r in rows), "window_start=None 不得放行任何对"
    stats = _motion_rows.__globals__  # noqa: F841  （占位：真实断言经 _estimate_detail）
    from lingshu.world.motion import _estimate_detail
    d = _estimate_detail(pts, None)
    assert d is not None and d["samples"] == 0, (
        "window_start=None 时 samples 应为 0（窗口外不计）")
