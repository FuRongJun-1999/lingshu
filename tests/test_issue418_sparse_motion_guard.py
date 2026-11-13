# -*- coding: utf-8 -*-
"""test_issue418_sparse_motion_guard · 稀疏观测下必须学到运动速度（#418 守卫）
============================================================================
缺陷（issue #418，成立）
------------------------
有限带宽（`budget=1` round_robin）下正常历史是「观测→缺帧→观测」，`speed_est`
本应学到真值（本件用例 0.8 / 0.4），实测却**卡在默认 0.3、persistence 0.0**
（＝`_motion_stats` 的回落默认值对 `(0.0, 0.3)`）。

根因（本件钉住的那一处）：`motion.ObservationWindow` 的回填条件
    `if points and self.start and len(points) < len(self.frames)`
有两处丢样本：
  ① 窗口内**一个**观测都没有时（长期遮挡 / 观测周期 > 窗口）**整段跳过**回填
     ⇒ `motion()` 返回 None ⇒ 下游回落默认 0.3/0.0 —— "上一次真实观测连同它的
     时间戳一起被丢"（**不是样本不足，是可用样本被丢**）；
  ② 即便触发也只回填**一个**点 ⇒ 窗口内 0 点时仍不足 2 点，构不成一次位移。
另有第二处：配对时间差没有任何**有效期**概念，长间隔（如 dt=17 的陈旧跨度）与
1 tick 的新鲜位移**同权**参与算术平均 ⇒ 陈旧跨度把速度拉偏，且调用方看不出置信。

修法
----
1. `_context_points`：按需回填——窗口内 0 点 ⇒ 取更早历史里**最近的两个**真实
   观测配对；窗口内 ≥1 点但不满窗 ⇒ 沿用"补 1 个前置点（给首个位移供真实 dt）"
   （该路径行为逐位不变）。只搬运真实观测，**不补造帧**、不把缺失当静止（#39）。
2. `_estimate_detail`：`dt > interval_limit(window)` 的配对按 `limit/dt` **降权**
   （不丢弃证据、不冒充高置信）；`confidence` / `max_obs_dt` /
   `long_interval_pairs` / `stale_ticks` / `evidence` 做成**可读读数**
   （`motion_detail()`），并写进 `world_learner.learn()` 的 `per_entity`。
3. `motion()` 的四元组契约与 `samples`（#148 新证据计数）口径不变；
   `_motion_stats` 只在 `stats[2] > 0`（有新证据）时才允许清空方向缓存。

判据来源
--------
· 复现与期望值：issue #418 正文（两实体直线 0.8/0.4；`round_robin` 预算 1 vs 2；
  真实时间戳独立复算在稀疏组同样得到 0.8/0.4）。
· "不把缺失当静止"：#39 的既有口径（真实观测才进历史；缺失帧不产生测量）。
· 观测计数/时间记录口径：#148（`samples` 只数窗口内有真实样本参与的位移对）。
· "判据必须能红 + 口径写进数据"体例：`wm-verify/window_caliber.py`（定点变异必须
  恰好转红）与 `wm-verify/eval/registry.py`（一切写成字段）；本件自包含在 `tests/`。

断言组（抽掉修复 ⇒ 哪组红）
--------------------------
  A 组 稀疏控制口径 ──────── 抽掉回填（`_context_points` 回退）⇒ **A 红**
     （budget=1 的 speed_est 由 0.8/0.4 掉回 0.3/0.3）
  B 组 缺帧不清空上次真实观测 ─ 同上回退 ⇒ **B 红**（0.8 → 0.3、persistence 1.0 → 0.0）
  C 组 长间隔降权 + 可读读数 ── 把权重退回 `w ≡ 1` ⇒ **C 红**（confidence 0.853 → 1.0、
     long_interval_pairs 1 → 0；混合读数 0.469 → 0.4）
  D 组 契约与"不补造帧" ──── 让回填插入非真实观测/单 tick 假 dt ⇒ **D 红**
  E 组 真 dt 口径 ────────── 把 dt 换成"帧序差 1" ⇒ **E 红**
  F 组 两处 `_motion_stats` 同口径 ── 把 `UnifiedWorldModel._motion_stats` 退回
     "无条件接受 stats[:2]" ⇒ **F 红**（c 场景 0.68 → 0.2；实测见 `align418.json`）
  G 组（第三方复核 F1/F3）────────
     G1/G2 `window=1` 的可用真样本 ── 把 `_context_points` 的 need 判据退回旧式
        （按"缺了几帧"判）⇒ **G1/G3 红**（window=1：0.8 → None/回落 0.3）
     G4-G7 NaN/±Inf 有限性 ────── 抽掉 `_finite_pos` 过滤 ⇒ **G4/G5 红**
        （NaN 传染进 speed/confidence；实测 `estimate=(nan, …, 0, …)`、confidence=1.0）

回退即红的实测读数（本件运行时实时打印，见 `_readings` 快照）
------------------------------------------------------------
  稀疏控制 budget=1：修复 0.8 / 0.4 ；回退 0.3 / 0.3
  长遮挡（2 次真实观测后 38 tick 未观测）：修复 0.8 / pers 1.0 ；回退 0.3 / pers 0.0
  长间隔混合（dt 17 + dt 1）：修复 speed 0.469 / confidence 0.8529 / long 1
                              ；回退 speed 0.400 / confidence 1.0 / long 0

运行命令
--------
（需已装 pytest；系统 python3 未装时，用任意含 pytest 的解释器即可）
    cd <仓根>
    python3 -m pytest tests/test_issue418_sparse_motion_guard.py -q --no-header
纯标准库实现路径：本件只读 `lingshu/world/*`（motion/world_learner/curiosity_explorer/
seven_layer_loop），不引入 numpy/PIL，不依赖 `wm-verify/`。
"""
from __future__ import annotations

import math
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lingshu.world import motion as M                              # noqa: E402
from lingshu.world.curiosity_explorer import CuriosityExplorer     # noqa: E402
from lingshu.world.motion import ObservationWindow, interval_limit  # noqa: E402
from lingshu.world.seven_layer_loop import SevenLayerLoop          # noqa: E402
from lingshu.world.world_model import UnifiedWorldModel            # noqa: E402

#: 稀疏控制口径的真值（issue #418：两实体直线速度）
TRUTH_A, TRUTH_B = 0.8, 0.4


# --------------------------------------------------------------------------- #
# 回归形态（定点变异的注入面，**保持与修复前逐字一致**）
# --------------------------------------------------------------------------- #
def legacy_context_points(self, eid, points):
    """#418 修复前的回填逻辑：0 点时整段跳过；触发时只回填 1 个点。"""
    if not self.start or not points:
        return points, 0
    if len(points) >= len(self.frames):
        return points, 0
    for index in range(self.start - 1, -1, -1):
        cur = self.history[index]["entities"].get(eid)
        if cur is not None:
            return ([(self.history[index]["tick"],
                      cur["pos"] if self.nested else cur)] + list(points)), 1
    return points, 0


def legacy_estimate_detail(points, window_start=None, limit=None):
    """#418 修复前的估计口径：无有效期、配对等权（等价于 `w ≡ 1`）。"""
    return _ORIGINAL_ESTIMATE_DETAIL(points, None, None)


_ORIGINAL_CONTEXT_POINTS = ObservationWindow._context_points
_ORIGINAL_ESTIMATE_DETAIL = M._estimate_detail


@pytest.fixture
def revert_backfill(monkeypatch):
    """定点变异 A/B：把"按需回填"退回"0 点跳过 + 只回填一点"。"""
    monkeypatch.setattr(ObservationWindow, "_context_points", legacy_context_points)
    return True


@pytest.fixture
def revert_interval_policy(monkeypatch):
    """定点变异 C：把"有效期降权"退回"等权 + 无有效期读数"。"""
    monkeypatch.setattr(M, "_estimate_detail", legacy_estimate_detail)
    return True


# --------------------------------------------------------------------------- #
# 夹具：控制口径 / 长遮挡 / 长间隔
# --------------------------------------------------------------------------- #
def _loop_control(budget, ticks=30, window=6, size=100, seed=42):
    """issue #418 控制口径：两实体直线 0.8 / 0.4，round_robin，budget=1 / 2。"""
    loop = SevenLayerLoop(size=size, seed=seed, window=window, budget=budget,
                          policy="round_robin")
    a = loop.add_entity("A", speed=0.8, pos=(2.0, 1.5, 2.0))
    b = loop.add_entity("B", speed=0.4, pos=(2.0, 1.5, 60.0))
    for _ in range(ticks):
        loop.world.set_command(a, "east", 0.8)
        loop.world.set_command(b, "east", 0.4)
        loop.step()
    model = loop.explorer.model["per_entity"]
    return {"a": model[a], "b": model[b], "id_a": a, "id_b": b,
            "obs_a": loop.explorer.obs_counts.get(a, 0),
            "obs_b": loop.explorer.obs_counts.get(b, 0),
            "history_len": len(loop.explorer.history)}


def _occlusion_speed(occluded=38, window=6):
    """长遮挡：a 只在最早 2 tick 被真实观测（速度 0.8），此后只观测 b。"""
    ex = CuriosityExplorer(size=100, window=window, seed=42)
    a = ex.world.add_entity("a", speed=0.8, pos=(2.0, 1.5, 2.0))
    b = ex.world.add_entity("b", speed=0.0, pos=(2.0, 1.5, 90.0))
    for tick in range(1, 3 + occluded):
        if tick <= 2:
            ex.world.entities[a].pos = (2.0 + 0.8 * (tick - 1), 1.5, 2.0)
            ex.observe(entities=[a])
        else:
            ex.observe(entities=[b])
    ex.learn()
    return ex, a, b


def _frame(tick, **kwargs):
    return {"tick": tick,
            "entities": {k: {"pos": v} for k, v in kwargs.items()}}


def _long_single(window=6):
    """单长间隔：t=1 与 t=21 各一次真实观测（跨度 20 > 有效期 18）。"""
    history = [_frame(1, a=(2.0, 1.5, 2.0))]
    history += [_frame(t, b=(0.0, 1.5, 0.0)) for t in range(2, 21)]
    history += [_frame(21, a=(2.0 + 0.8 * 20, 1.5, 2.0))]
    return ObservationWindow(history, window, nested=True)


def _long_mixed(window=3):
    """混合间隔：窗口内 dt=1 的新鲜位移 + 回填上下文 dt=17（> 有效期 12）。"""
    history = [_frame(1, a=(0.0, 1.5, 0.0))]
    history += [_frame(t, b=(0.0, 1.5, 0.0)) for t in range(2, 18)]
    history += [_frame(18, a=(0.0, 1.5, 0.0))]
    history += [_frame(19, a=(0.8, 1.5, 0.0))]
    return ObservationWindow(history, window, nested=True)


# --------------------------------------------------------------------------- #
# A 组：稀疏控制口径必须学到真值
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("budget,expect_b", [(1, True), (2, True)])
def test_A_sparse_straight_line_learns_true_speed(budget, expect_b):
    """A：budget=1（稀疏）与 budget=2（全带宽）都必须学到 0.8 / 0.4。

    修复前：budget=1 时可用样本被丢 ⇒ 0.3 / 0.3（默认值对）。
    """
    r = _loop_control(budget)
    assert r["a"]["speed_est"] == pytest.approx(TRUTH_A, abs=0.05), r["a"]
    assert r["b"]["speed_est"] == pytest.approx(TRUTH_B, abs=0.05), r["b"]
    assert r["a"]["persistence"] == pytest.approx(1.0, abs=0.05), r["a"]
    # 稀疏组每实体 15 次真观测（30 tick / 2 实体）；全带宽 30 次
    assert r["obs_a"] == (15 if budget == 1 else 30), (r["obs_a"], r["obs_b"])
    # 读数必须自证证据来源与置信（#418：口径写成数据）
    assert r["a"]["motion_evidence"] in ("window", "window+context")
    assert r["a"]["motion_confidence"] == pytest.approx(1.0, abs=0.05)


def test_A_control_scope_is_self_reported(revert_backfill):
    """如实记录**判别力边界**：控制口径①（cadence 2 ≤ 窗口 6）在"只回填一个点"
    的旧行为下**也已经是 0.8/0.4**（窗口内各有 3 个真样本，旧回填恰好够用）。

    ⇒ 本件的定点变异面在**验收②/④（长遮挡、长间隔）**，不在控制①。这里把回退
    注入后**仍是 0.8**，正是"控制① 不具判别力"的证据（不是判据失效）；判别力在
    `test_B_mutation_*` / `test_C_mutation_*`，那两处回退后必须转红。
    """
    r = _loop_control(1)
    assert r["a"]["speed_est"] == pytest.approx(0.8, abs=0.05), (
        "控制口径在回退注入下的读数变了：%r" % (r["a"],))
    assert r["b"]["speed_est"] == pytest.approx(0.4, abs=0.05), r["b"]


def test_A_mutation_is_exactly_the_regression_fixture():
    """自证：夹具不是空转——它注入的函数与"修复前"语义等价（0 点不回填）。"""
    history = [_frame(1, a=(0.0, 1.5, 0.0)), _frame(2, a=(0.8, 1.5, 0.0))]
    history += [_frame(t, b=(0.0, 1.5, 0.0)) for t in range(3, 12)]
    window = ObservationWindow(history, 6, nested=True)
    assert window.motion("a") == (1, 0.8, 0, (1, 0, 0))       # 修复后：用上次真实观测对
    saved, M.ObservationWindow._context_points = M.ObservationWindow._context_points, legacy_context_points
    try:
        assert ObservationWindow(history, 6, nested=True).motion("a") is None
    finally:
        M.ObservationWindow._context_points = saved


# --------------------------------------------------------------------------- #
# B 组：缺帧不得清空/丢弃"上一次真实观测"
# --------------------------------------------------------------------------- #
def test_B_occlusion_keeps_last_real_observation():
    """B：2 次真实观测后 38 tick 未观测 ⇒ 速度仍是 0.8（不是默认 0.3）。

    且**不补造帧**：`samples` 仍为 0（窗口内没有新证据，#148 口径），
    证据来源标 `context_only`、`stale=True`、陈旧度可读。
    """
    ex, a, b = _occlusion_speed()
    p = ex.model["per_entity"][a]
    assert p["speed_est"] == pytest.approx(TRUTH_A, abs=0.05), p
    assert p["persistence"] == pytest.approx(1.0, abs=0.05), p
    assert p["samples"] == 0, "窗口内没有新证据时 samples 必须为 0（不得补造）"
    assert p["motion_evidence"] == "context_only", p
    assert p["stale"] is True and ex.nodes[a].last_seen == 2
    # 确认历史里确实没有 a 的后 38 条记录（缺失 ≠ 静止，也没有被补造）
    assert all(a not in rec["entities"] for rec in ex.history[-38:])
    # 未被观测的信念不得进历史（#39）：b 的观测没有把 a 的位置写进来
    assert ex.nodes[a].pos == (2.8, 1.5, 2.0)


def test_B_mutation_dropping_backfill_turns_occlusion_red(revert_backfill):
    """B/定点变异①：退回旧行为 ⇒ 长遮挡回到默认 0.3 / persistence 0.0（判据能红）。"""
    ex, a, _b = _occlusion_speed()
    p = ex.model["per_entity"][a]
    assert p["speed_est"] == pytest.approx(0.3, abs=0.05), p
    assert p["persistence"] == pytest.approx(0.0, abs=0.05), p
    assert p["motion_evidence"] == "none", p


def test_B_stale_direction_is_not_cleared_by_missing_frames():
    """B（#39 口径）：无新证据时**不得**清空方向缓存（缺失 ≠ 测得静止）。"""
    ex, a, b = _occlusion_speed()
    # 遮挡前最后一次真实位移方向是 +x；遮挡期间不得被"零位移"顶掉
    assert ex._motion_directions.get(a) == (1.0, 0.0, 0.0), ex._motion_directions
    # 而窗口内"测得静止"（有新证据且位移为零）才允许清方向
    ex2 = CuriosityExplorer(size=100, window=6, seed=42)
    c = ex2.world.add_entity("c", speed=0.0, pos=(5.0, 1.5, 5.0))
    for _ in range(4):
        ex2.observe(entities=[c])
    ex2.learn()
    assert ex2._motion_directions.get(c) is None


# --------------------------------------------------------------------------- #
# C 组：长间隔降权 + 置信读数
# --------------------------------------------------------------------------- #
def test_C_long_interval_is_downweighted_and_readable():
    """C：span 20 的单对 ⇒ confidence = limit/20 = 18/20 = 0.9 且 long_pairs=1。

    （旧口径没有有效期概念：confidence 不存在、long_pairs 恒 0 ⇒ 长间隔被当成
    与 1 tick 新鲜位移同等的证据。）
    """
    d = _long_single(window=6).motion_detail("a")
    assert d["interval_limit"] == interval_limit(6) == 18
    assert d["max_obs_dt"] == 20
    assert d["long_interval_pairs"] == 1
    assert d["confidence"] == pytest.approx(18.0 / 20.0, abs=1e-6), d
    assert d["estimate"][1] == pytest.approx(0.8, abs=0.05), d      # 仍是真实速度
    assert d["schema"] == M.MOTION_SCHEMA
    assert d["stale_ticks"] == 0


def test_C_mixed_intervals_weighting_is_visible():
    """C：混合间隔（dt=17 + dt=1）⇒ 长对被降权，读数与等权口径可区分。

    等权（修复前）：(0/17 + 0.8/1)/2 = 0.400、confidence 1.0、long_pairs 0
    降权（修复后）：(0.706·0 + 1·0.8)/1.706 = 0.469、confidence 0.853、long_pairs 1
    """
    d = _long_mixed(window=3).motion_detail("a")
    assert d["dts"] == [17, 1], d
    assert d["interval_limit"] == 12
    assert d["long_interval_pairs"] == 1
    assert d["confidence"] == pytest.approx((12.0 / 17.0 + 1.0) / 2.0, abs=1e-6), d
    assert d["confidence"] < 1.0
    assert d["estimate"][1] == pytest.approx(0.469, abs=0.002), d
    assert d["estimate"][2] == 2                                   # samples 不变


def test_C_mutation_unweighted_mean_turns_red(revert_interval_policy):
    """C/定点变异②：把权重退回 `w ≡ 1`（无有效期）⇒ 置信读数恒 1、长间隔不可见。

    与修复态（`test_C_long_interval_is_downweighted_and_readable`：0.9 / long=1）
    成对 ⇒ 判据**能红**：同输入下 confidence 1.0 ≠ 0.9、long_pairs 0 ≠ 1、
    混合读数 0.400 ≠ 0.469。
    """
    d = _long_single(window=6).motion_detail("a")
    assert d["confidence"] == pytest.approx(1.0, abs=1e-9), d
    assert d["long_interval_pairs"] == 0, d
    assert d["estimate"][1] == pytest.approx(0.8, abs=0.05), d   # 单对：值本身不变
    mixed = _long_mixed(window=3).motion_detail("a")
    assert mixed["estimate"][1] == pytest.approx(0.4, abs=0.002), mixed
    assert mixed["confidence"] == pytest.approx(1.0, abs=1e-9), mixed


# --------------------------------------------------------------------------- #
# D 组：契约不变 + 不补造帧
# --------------------------------------------------------------------------- #
def test_D_motion_contract_unchanged():
    """D：`motion()` 四元组形状/语义不变；`samples` 仍只数窗口内新证据。"""
    window = ObservationWindow(
        [_frame(1, a=(2.0, 1.5, 2.0))] + [_frame(t) for t in range(2, 11)]
        + [_frame(11, a=(12.0, 1.5, 2.0)), _frame(12)], 3, nested=True)
    assert window.motion("a") == (1, 1, 1, (1, 0, 0))
    assert window.motion("missing") is None
    assert ObservationWindow([], 6).motion("a") is None
    # 窗口内 0 点（长遮挡）⇒ samples 必须为 0（新证据计数口径不变），见 test_B


def test_D_no_fabricated_frames_in_history():
    """D：稀疏观测下历史里**没有**为未观测实体补造记录（缺失 ≠ 观测）。"""
    ex, a, b = _occlusion_speed()
    sparse = ex.history[-38:]
    assert all(a not in rec["entities"] and b in rec["entities"]
               for rec in sparse), "历史里出现了未观测实体的记录（补造帧）"
    # 每个实体在历史里出现的次数 == 真实观测次数
    seen_a = sum(1 for rec in ex.history if a in rec["entities"])
    assert seen_a == ex.obs_counts[a] == 2, (seen_a, ex.obs_counts[a])


def test_D_real_observation_timestamps_are_preserved():
    """D：回填搬运的是**真实 tick**（不是帧序/序号），dt 由真时间戳算。"""
    window = ObservationWindow(
        [_frame(1, a=(0.0, 1.5, 0.0))] + [_frame(t, b=(0.0, 1.5, 0.0))
                                          for t in range(2, 21)]
        + [_frame(21, a=(16.0, 1.5, 0.0))],
        2, nested=True)
    detail = window.motion_detail("a")
    assert detail["estimate"][1] == pytest.approx(0.8, abs=0.02), detail
    assert detail["dts"] == [20], detail


# --------------------------------------------------------------------------- #
# E 组：真 dt 口径（不得把跨帧距离当成单 tick 距离）
# --------------------------------------------------------------------------- #
def test_E_gap_distance_is_not_single_tick_speed():
    """E：两 tick 间隔的 1.6 位移 ⇒ 速度 0.8（不是 1.6）。抽掉真 dt ⇒ 红。"""
    window = ObservationWindow(
        [_frame(1, a=(0.0, 1.5, 0.0)), _frame(3, a=(1.6, 1.5, 0.0))], 6,
        nested=True)
    assert window.motion("a")[1] == pytest.approx(0.8, abs=1e-9)
    detail = window.motion_detail("a")
    assert detail["dts"] == [2] and detail["max_obs_dt"] == 2
    # 反向腿：若把 dt 当成 1（帧序差），速度会是 1.6 —— 判据必须能区分
    assert abs(detail["estimate"][1] - 1.6) > 0.5


def test_E_interval_limit_is_data_not_magic_number():
    """E：有效期是**可读回的数据**（常量 + 函数），不是调用点写死的数。"""
    assert M.MOTION_SCHEMA == "wm.motion-confidence/1"
    assert M.INTERVAL_LIMIT_FACTOR == 3 and M.MIN_INTERVAL_WINDOW == 4
    assert interval_limit(2) == 12 and interval_limit(6) == 18
    assert interval_limit(0) == interval_limit(4) == 12      # 下限生效
    d = _long_single(6).motion_detail("a")
    assert d["interval_limit"] == interval_limit(6)
    assert M.MOTION_SCHEMA in (d["schema"],)



# --------------------------------------------------------------------------- #
# F 组（#418 追加件）：UnifiedWorldModel._motion_stats 与 world_learner 同口径
# --------------------------------------------------------------------------- #
def legacy_world_model_motion_stats(self, eid, window=8, observations=None):
    """#418 追加件**修复前**形态：无条件接受 `stats[:2]`。

    即"窗口内无新样本（`stats[2] == 0`，估计来自更早的真实观测对）也直接覆盖
    上一轮已学估计"——与 `WorldLearner._motion_stats` 的新口径不一致。
    """
    if observations is None:
        observations = ObservationWindow(self.history, window)
    stats = observations.motion(eid)
    if stats is not None:
        self._motion_estimates[eid] = stats[:2]
        if stats[3] is not None:
            self._motion_directions[eid] = stats[3]
        else:
            self._motion_directions.pop(eid, None)
    return self._motion_estimates.get(eid, (0.0, 0.3))


@pytest.fixture
def revert_world_model_stats(monkeypatch):
    """定点变异③：把 world_model 的 `_motion_stats` 退回"无条件接受 stats[:2]"。"""
    monkeypatch.setattr(UnifiedWorldModel, "_motion_stats",
                        legacy_world_model_motion_stats)
    return True


#: 三场景的 x 序列（真速度：a 段 1.0/tick → c 段 0.2/tick，末对真实观测只给 0.2）
_MODEL_SCEN = {
    "moving": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    "stop": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
    "decel": [0.0, 1.0, 2.0, 3.0, 3.2, 3.4],
}


def _model_read(positions, mode="once", occluded=10, window=8):
    """world_model 路径读数；mode=once ⇒ 密集期学一次 + 遮挡后学一次。"""
    model = UnifiedWorldModel(size=100)
    for x in positions:
        model.perceive([{"eid": "c", "category": "c", "pos": (x, 1.5, 2.0)}])
        if mode == "every":
            model.infer_patterns(window=window, force=True)
    if mode == "once":
        model.infer_patterns(window=window, force=True)
    for _ in range(occluded):
        model.perceive([{"eid": "z", "category": "z", "pos": (50.0, 1.5, 50.0)}])
        if mode == "every":
            model.infer_patterns(window=window, force=True)
    estimate = model._motion_stats("c", window=window)
    return model, [round(v, 3) for v in estimate]


def _learner_read(positions, mode="once", occluded=10, window=8):
    """world_learner（CuriosityExplorer）路径读数，同一驱动。"""
    ex = CuriosityExplorer(size=100, window=window, seed=42)
    cid = ex.world.add_entity("c", speed=0)
    zid = ex.world.add_entity("z", speed=0)
    for x in positions:
        ex.world.entities[cid].pos = (x, 1.5, 2.0)
        ex.observe(entities=[cid])
        if mode == "every":
            ex.learn()
    if mode == "once":
        ex.learn()
    for _ in range(occluded):
        ex.world.entities[zid].pos = (50.0, 1.5, 50.0)
        ex.observe(entities=[zid])
        if mode == "every":
            ex.learn()
    ex.learn()
    params = ex.model["per_entity"][cid]
    return ex, params


@pytest.mark.parametrize("mode", ["every", "once"])
def test_F_two_paths_are_aligned_on_all_scenarios(mode):
    """F：两种驱动 × 三场景，`world_model` 与 `world_learner` 读数**必须一致**。

    "every"（每帧都学/推断）＝真实闭环 ⇒ 窗口滑动本身就是证据，估计跟随；
    "once"（密集期一次 + 遮挡后一次）＝遮挡期**无新样本** ⇒ 保留已学估计。
    """
    for name, positions in _MODEL_SCEN.items():
        _m, model_est = _model_read(positions, mode)
        _e, params = _learner_read(positions, mode)
        learner_est = [params["persistence"], params["speed_est"]]
        assert model_est == learner_est, (
            "两处 _motion_stats 走偏（mode=%s / %s）：world_model=%r world_learner=%r"
            % (mode, name, model_est, learner_est))


def test_F_no_new_samples_does_not_overwrite_learned_estimate():
    """F：密集期已学 0.68（1.0 段 + 0.2 段的加权），长遮挡后**不得**被上下文对 0.2 覆盖。"""
    _model, model_est = _model_read(_MODEL_SCEN["decel"], "once")
    _ex, params = _learner_read(_MODEL_SCEN["decel"], "once")
    assert model_est == [1.0, 0.68], model_est
    assert [params["persistence"], params["speed_est"]] == [1.0, 0.68], params
    # 证据来源与"新样本计数"必须如实标出（无新样本）
    assert params["motion_evidence"] == "context_only" and params["samples"] == 0, params


def test_F_measured_stop_still_overrides():
    """F：**测得静止**是证据（测量优先）⇒ 长遮挡后仍是 0.0，不得被陈旧速度顶掉。"""
    for mode in ("every", "once"):
        _m, model_est = _model_read(_MODEL_SCEN["stop"], mode)
        _e, params = _learner_read(_MODEL_SCEN["stop"], mode)
        assert model_est == [0.0, 0.0], (mode, model_est)
        assert [params["persistence"], params["speed_est"]] == [0.0, 0.0], (mode, params)


def test_F_mutation_unconditional_accept_turns_red(revert_world_model_stats):
    """F/定点变异③：把 world_model 退回"无条件接受 stats[:2]" ⇒ **判据转红**。

    实测：`once/减速` 场景由 **0.68（保留已学）→ 0.2（被上下文对覆盖）**；
    而 `every` 三场景与 `once` 的 moving/stop 两场景不受影响（滑动窗口本来就收敛到
    同一对证据）⇒ 这正是该规则**唯一**的可观测差异面，判据在此处必须能红。
    """
    _m, model_est = _model_read(_MODEL_SCEN["decel"], "once")
    assert model_est == [1.0, 0.2], (
        "回退后仍是 %r ⇒ 判据没有判别力（应与对齐口径的 0.68 不同）" % (model_est,))
    # 对齐口径在同一输入下是 0.68（见 test_F_no_new_samples_does_not_overwrite_learned_estimate）
    assert model_est != [1.0, 0.68]
    # 反向腿：变异不影响"测得静止"覆盖（测量优先不因本变异改变）
    _m2, stop_est = _model_read(_MODEL_SCEN["stop"], "once")
    assert stop_est == [0.0, 0.0], stop_est


def test_F_world_model_exposes_same_readable_readout():
    """F：world_model 侧也留**可读读数**（`_motion_detail`），字段与 learner 同语义。"""
    model, _est = _model_read(_MODEL_SCEN["decel"], "once")
    detail = model._motion_detail["c"]
    assert detail["schema"] == M.MOTION_SCHEMA
    assert detail["evidence"] == "context_only"
    assert detail["confidence"] == pytest.approx(1.0, abs=1e-9)
    assert detail["max_obs_dt"] == 1 and detail["interval_limit"] == interval_limit(8)
    _ex, params = _learner_read(_MODEL_SCEN["decel"], "once")
    for key in ("motion_schema", "motion_evidence", "motion_confidence", "max_obs_dt"):
        assert key in params, params
    assert params["motion_schema"] == detail["schema"]
    assert params["motion_evidence"] == detail["evidence"]
    assert params["max_obs_dt"] == detail["max_obs_dt"]



# --------------------------------------------------------------------------- #
# G 组（第三方复核 §F1/§F3）：window=1 的可用样本 + NaN/±Inf 有限性
# --------------------------------------------------------------------------- #
def _history_from_obs(obs, n_frames):
    """`obs: {tick: (x,y,z)}` → 稀疏历史（其余帧该实体**不出现**，不补造）。"""
    return [{"tick": t, "entities": ({"e": {"pos": obs[t]}} if t in obs else {})}
            for t in range(n_frames)]


#: 复核 F1 的最小复现：41 帧、实体只在 tick 0 与 40 被观测，真速度 0.8（dt=40）
_F1_OBS = {0: (0.0, 0.0, 0.0), 40: (32.0, 0.0, 0.0)}


def test_G1_window1_keeps_usable_real_samples():
    """G1（复核 §F1 靶）：`window=1` 时窗口内 1 观测 + 更早 1 观测 ⇒ **必须出估计**。

    旧判据 `need = 1 if len(points) < len(frames) else 0`：`window=1` 且该帧正好观测
    到时 `1 < 1` 为假 ⇒ need=0 ⇒ 只有 1 点 ⇒ 构不成位移对 ⇒ `motion()` 返回 None
    ⇒ 下游回落默认 `speed=0.3`。
    """
    for window, confidence in ((1, 0.3), (2, 0.3), (3, 0.3), (6, 0.45)):
        d = ObservationWindow(_history_from_obs(_F1_OBS, 41), window,
                              nested=True).motion_detail("e")
        assert d["estimate"] is not None, (window, d)
        assert d["estimate"][1] == pytest.approx(0.8, abs=0.02), (window, d)
        assert d["window_points"] == 1 and d["context_points"] == 1, (window, d)
        assert d["evidence"] == "window+context", (window, d)
        assert d["max_obs_dt"] == 40, (window, d)
        # 置信度 = 有效期/间隔 = max(window,4)*3/40
        assert d["confidence"] == pytest.approx(confidence, abs=1e-6), (window, d)
    # window=1 与更大窗口读数**一致**：判据是"能否构成一对位移"，不是"缺了几帧"
    speeds = [ObservationWindow(_history_from_obs(_F1_OBS, 41), w, nested=True).motion("e")[1]
              for w in (1, 2, 3, 6)]
    assert speeds == [0.8, 0.8, 0.8, 0.8], speeds


def test_G2_public_helpers_also_covered():
    """G2：公开可达面同样不得丢样本——`estimate_motion(...1)` / `observed_positions`。"""
    history = _history_from_obs(_F1_OBS, 41)
    for window in (1, 2, 3, 6):
        stats = M.estimate_motion(history, "e", window, nested=True)
        assert stats is not None, (window, stats)
        assert stats[1] == pytest.approx(0.8, abs=0.02), (window, stats)
    assert len(M.observed_positions(history, "e", 1, nested=True)) == 2


def test_G3_mutation_frame_based_need_turns_window1_red(revert_backfill):
    """G3/定点变异①：need 判据退回"按缺帧数"判 ⇒ `window=1` 立刻回落 None（判据能红）。"""
    d = ObservationWindow(_history_from_obs(_F1_OBS, 41), 1,
                          nested=True).motion_detail("e")
    assert d["estimate"] is None, (
        "回退后 window=1 仍出估计 %r ⇒ 判据没有判别力" % (d["estimate"],))
    assert d["reason"] == "insufficient_real_observations", d
    # 更大窗口不受该变异影响（旧判据在那里本来就回填）⇒ 判别面确实只在 window=1
    d6 = ObservationWindow(_history_from_obs(_F1_OBS, 41), 6,
                           nested=True).motion_detail("e")
    assert d6["estimate"] is not None and d6["estimate"][1] == pytest.approx(0.8, abs=0.02), d6


def _nan_history(value):
    return _history_from_obs({0: (0.0, 0.0, 0.0), 2: (value, 0.0, 0.0)}, 10)


@pytest.fixture
def revert_finite_filter(monkeypatch):
    """旁路第一道坐标闸；算术有限性闸仍应独立阻断坏读数。"""
    monkeypatch.setattr(M, "_finite_pos", lambda pos: True)
    return True


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_G4_nonfinite_coords_never_enter_the_estimate(bad):
    """G4（复核 §F3）：非有限坐标的位移对**一律拒绝**，读数不得被污染。"""
    d = ObservationWindow(_nan_history(bad), 6, nested=True).motion_detail("e")
    assert d["estimate"] is None, d
    assert d["confidence"] == 0.0 and d["max_obs_dt"] is None, d
    assert d["nonfinite_points"] == 1 and d["nonfinite_pairs_dropped"] == 1, d
    assert d["reason"] == "nonfinite_observation", d


def test_G5_finite_filter_bypass_is_blocked_by_arithmetic_gate(revert_finite_filter):
    """G5：坐标闸被旁路时，第二道算术闸仍必须拒绝 NaN/Inf。

    旧断言假定旁路唯一防线就会污染；新增独立算术闸后，不应为了该自证放行坏值。
    G4 继续检查正常路径的非有限点/配对诊断，新算术变异件单独证明第二道闸可点红。
    """
    nan = ObservationWindow(_nan_history(float("nan")), 6,
                            nested=True).motion_detail("e")
    assert nan["estimate"] is None and nan["arithmetic_pairs_dropped"] == 1, nan
    assert nan["reason"] == "invalid_motion_arithmetic", nan
    inf = ObservationWindow(_nan_history(float("inf")), 6,
                            nested=True).motion_detail("e")
    assert inf["estimate"] is None and inf["arithmetic_pairs_dropped"] == 1, inf
    assert inf["confidence"] == 0.0 and inf["nonfinite_points"] == 0, inf


def test_G6_finite_pairs_still_used_when_mixed():
    """G6（不误伤）：一帧坏读数不毁掉其余**有限**证据；坏帧被计数、可读。"""
    history = _history_from_obs({0: (0.0, 0.0, 0.0), 2: (1.6, 0.0, 0.0),
                                 4: (float("nan"), 0.0, 0.0),
                                 6: (4.8, 0.0, 0.0)}, 8)
    d = ObservationWindow(history, 8, nested=True).motion_detail("e")
    assert d["estimate"] is not None, d
    assert d["estimate"][1] == pytest.approx(0.8, abs=0.02), d     # 有限对仍给出真速度
    assert d["nonfinite_points"] == 1, d
    assert d["nonfinite_pairs_dropped"] == 2, d                    # (t2,t4) 与 (t4,t6) 各一对
    assert math.isfinite(d["confidence"]) and math.isfinite(d["estimate"][1]), d


def test_G7_nonfinite_never_reaches_the_learned_speed():
    """G7（端到端）：NaN/±Inf 观测不得把非有限值写进学得的 `speed_est`（两条路径）。"""
    model = UnifiedWorldModel(size=100)
    model.perceive([{"eid": "a", "category": "a", "pos": (0.0, 1.5, 2.0)}])
    model.perceive([{"eid": "a", "category": "a", "pos": (float("nan"), 1.5, 2.0)}])
    pattern = model.infer_patterns(window=6, force=True)
    assert math.isfinite(pattern["speed_estimates"]["a"]), pattern["speed_estimates"]
    assert model._motion_detail["a"]["reason"] == "nonfinite_observation", \
        model._motion_detail["a"]

    ex = CuriosityExplorer(size=100, window=6, seed=42)
    eid = ex.world.add_entity("a", speed=0)
    ex.world.entities[eid].pos = (0.0, 1.5, 2.0)
    ex.observe(entities=[eid])
    ex.world.entities[eid].pos = (float("inf"), 1.5, 2.0)
    ex.observe(entities=[eid])
    ex.learn()
    params = ex.model["per_entity"][eid]
    assert math.isfinite(params["speed_est"]), params
    assert params["motion_evidence"] == "none", params


def test_guard_is_collectable():
    assert os.path.basename(__file__).startswith("test_")
    assert M.MOTION_SCHEMA and ObservationWindow is not None
    assert callable(interval_limit) and callable(SevenLayerLoop)
