# -*- coding: utf-8 -*-
"""守卫 · 亮灭周期检测不得被双沿减半（lingshu issue #76）

缺陷（HEAD 修复前，`lingshu/nn/stcnn.py`）
    `extract_spatiotemporal_primitives` 把 `frame_diff(frames)`（帧间 |Δ| 掩码）的
    逐帧面积当周期信号喂给 `detect_period`。|Δ| 对「暗→亮」与「亮→暗」各出一个沿，
    所以**单个亮灭周期产生双沿**：两个前沿之间的间隔是真实周期的一半，读出的周期
    因此**减半**。实测（本机 Python 3.12.10）：

        synth_blinking(frames=12, period=3) → period 3（亮:灭 = 2:1，双沿恰好重合 ⇒ 侥幸正确）
        synth_blinking(frames=24, period=4) → period 2（应为 4，减半）
        synth_blinking(frames=24, period=6) → period 3（应为 6，减半）

    既有的自检只覆盖 `synth_blinking(frames=12, period=3)` 这一档「唯一正确档」，
    缺陷因此长期不可见。

修复
    周期信号改用 `frame_rise(frames)`：只保留正向亮度变化（亮起前沿），每个亮灭
    周期恰一个沿。`frame_diff` 的 |Δ| 掩码语义不变（仍用于运动区域/质心）。

判据来源
    · 理论章节：`lingshu/nn/stcnn.py` 模块头引《白箱自举·LLM替代与3D时空多模态》
      §3——时空原语含「周期」，周期是信号**一个完整循环**的时长；一个循环的时长
      与半个循环的间隔不是同一个量。
    · 阈值（detect_period 的 0.05 前沿阈值、frame_diff/frame_rise 的 0.08）**未改**。
    · 合成数据 `synth_blinking(frames, period)` 自声明周期为 period（构造期真值），
      属经验标定（合成器定义），非理论推导。

定点变异自证（抽掉修复 ⇒ 必红）
    把 `extract_spatiotemporal_primitives` 的 `frame_rise(frames)` 改回 `diffs`
    （即 `frame_diff` 的 |Δ| 掩码）⇒ test_period_not_halved_for_non_2to1_duty 红
    （period=4 读成 2、period=6 读成 3）。

运行（仓根）：python -X utf8 -m pytest tests/test_stcnn_period_edges.py -q --no-header
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from lingshu.nn.stcnn import (  # noqa: E402
    detect_period,
    extract_spatiotemporal_primitives,
    frame_diff,
    frame_rise,
    synth_blinking,
)

FRAMES = 24


def _edges(sig):
    """信号里「低→高」前沿的位置（与 detect_period 同口径）。"""
    return [i for i in range(1, len(sig)) if sig[i] > 0.05 and sig[i - 1] <= 0.05]


def test_period_not_halved_for_non_2to1_duty():
    """★ 核心：亮灭占空比非 2:1 时，读出的周期必须等于真值（旧实现减半）。

    旧实现（喂 |Δ| 掩码）：period=4 → 2、period=6 → 3（各减半）⇒ 本条必红。
    """
    for period in (3, 4, 5, 6):
        prims, _ = extract_spatiotemporal_primitives(
            synth_blinking(frames=FRAMES, period=period))
        assert prims["period"] == period, (period, prims["period"])


def test_frame_rise_signal_has_one_edge_per_cycle():
    """前沿信号每个亮灭周期恰一个沿（旧 |Δ| 信号是每个周期两个沿）。"""
    period = 4
    frames = synth_blinking(frames=FRAMES, period=period)
    rise_signal = [float(region.sum()) for region in frame_rise(frames)]
    diff_signal = [float(region.sum()) for region in frame_diff(frames)]

    rise_edges = _edges(rise_signal)
    diff_edges = _edges(diff_signal)
    # 亮起前沿的间隔 = 真实周期；旧 |Δ| 信号的间隔是它的一半（双沿 ⇒ 减半）
    rise_intervals = {rise_edges[i + 1] - rise_edges[i]
                      for i in range(len(rise_edges) - 1)}
    diff_intervals = {diff_edges[i + 1] - diff_edges[i]
                      for i in range(len(diff_edges) - 1)}
    assert rise_intervals == {period}, rise_edges
    assert diff_intervals == {period // 2}, diff_edges


def test_legacy_abs_diff_signal_would_halve_period():
    """缺陷形态标定：把 |Δ| 掩码当周期信号 ⇒ detect_period 读出真值的一半。

    这条证明「旧信号确实会减半」——修复若退回该信号，本条与上面两条同时红。
    """
    period = 4
    frames = synth_blinking(frames=FRAMES, period=period)
    legacy_signal = [float(region.sum()) for region in frame_diff(frames)]
    assert detect_period(legacy_signal) == period // 2  # 减半：2


def test_period_3_regression_still_detected():
    """回归：既有自检覆盖的唯一档（period=3, frames=12）仍读出 3。"""
    prims, _ = extract_spatiotemporal_primitives(synth_blinking(frames=12, period=3))
    assert prims["period"] == 3


def test_frame_rise_only_keeps_positive_change():
    """frame_rise 只保留正向变化：逐元素 >= 0，且与 |Δ| 的正向部分一致。"""
    frames = synth_blinking(frames=FRAMES, period=4)
    rise = frame_rise(frames)
    arr = np.asarray(frames, dtype=np.float32)
    assert (rise >= 0).all()
    expected = np.clip(np.diff(arr, axis=0), 0.0, None)
    assert np.array_equal(rise, expected)
