# -*- coding: utf-8 -*-
"""confirmation · 完全确认与验证收益（智能论 v3.4 · 2.9.3a + 5.3）
============================================================================
完全确认(操作)：
    Confirmed(S) ⟺ ∀i∈ch: c_i ≥ θ_c
                 ∧ ∃强验证 realized_KL_task > θ_g
                 ∧ 跨时间稳定
                 ∧ 无内部矛盾

验证收益双层结构（v1.2 双货币统一）：
    Value(s, v) = ΔD_task(s, v) · σ( Gain_task(s, v) )
  - Gain_task：筛选器（这次验证值不值得做）= 任务相关期望信念更新量
  - ΔD_task：定价器（做完之后对任务值多少）= 图上信息差缩小量
  - σ(·)：单调门控（sigmoid），Gain 低于阈值则整个验证不值钱

期望值（Gain_task）用于决策；实现值（realized_KL）用于确认——分离。

纯标准库 · 零外部依赖（D-005）
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

# 确认阈值（工程默认值，待实测校准）
CONF_THRESHOLD = 0.5        # θ_c：通道可信度达标线
KL_THRESHOLD = 0.05         # θ_g：强验证 realized_KL 达标线
STABLE_ROUNDS = 5           # 跨时间稳定轮数


def _sigmoid(x: float) -> float:
    """σ 门控：单调 S 曲线，x→-∞ →0，x→+∞ →1。"""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def kl_binary(p_after: float, p_before: float) -> float:
    """二值信念分布 KL(p_after || p_before)：信息增益的离散近似。
    KL = p_after·log(p_after/p_before) + (1-p_after)·log((1-p_after)/(1-p_before))
    仅当信念改变时 > 0；p_after==p_before → 0（自我应答自动出局）。
    非有限输入（NaN/±inf）→ 0.0（#402：无信息＝fail-closed，不得被钳制表达式
    放行——实测 min(1-1e-9, nan)=1-1e-9 会把 NaN 概率抬成"几乎必然"）。"""
    p_after = float(p_after)
    p_before = float(p_before)
    if not (math.isfinite(p_after) and math.isfinite(p_before)):
        return 0.0
    p_after = max(1e-9, min(1 - 1e-9, p_after))
    p_before = max(1e-9, min(1 - 1e-9, p_before))
    return (p_after * math.log(p_after / p_before)
            + (1 - p_after) * math.log((1 - p_after) / (1 - p_before)))


def gain_task(p_after: float, p_before: float, task_relevance: float = 1.0) -> float:
    """任务相关期望信息增益（筛选器）：
    Gain_task = 任务相关性 × KL(p_after || p_before)。
    无关维度（task_relevance=0）→ Gain=0 → 无目的猎奇自动出局。"""
    kl = kl_binary(p_after, p_before)
    # #402：非有限任务相关性 fail-closed 归零（不得被钳制表达式抬成满相关）
    task_relevance = float(task_relevance)
    if not math.isfinite(task_relevance):
        task_relevance = 0.0
    return max(0.0, kl * max(0.0, min(1.0, task_relevance)))


def value(s_delta_d: float, gain: float, gain_gate: float = 0.05) -> float:
    """验证收益双层结构：Value = ΔD_task · σ(Gain_task)。
    σ 门控：Gain 低于 gain_gate → Value ≈ 0（验证不值钱）。
    gain=0（信念无改变）→ σ≈0 → Value≈0（自我应答/无目的猎奇出局）。"""
    # 门控：gain≤0 硬归零（信念无改变=无价值）；gain>0 用陡峭 sigmoid 软过渡
    if gain <= 0.0:
        return 0.0
    gate = _sigmoid((gain - gain_gate) * 100.0)
    # #1075②：非有限 ΔD 按 #402 归零（同文件 kl_binary / gain_task 的同款守卫；
    # 旧实现原样返回 ⇒ NaN 参与排序恒 False（与零收益不可区分）、inf 恒排第一）。
    s_delta_d = float(s_delta_d)
    if not math.isfinite(s_delta_d):
        return 0.0
    return s_delta_d * gate


class ConfirmationEvaluator:
    """完全确认评估器（v3.4 2.9.3a + 5.3）。

    输入：
      - channel_creds：{channel: credibility}（来自 B1 注册表）
      - realized_kl：强验证已实现的信念更新量（KL）
      - stable_rounds：跨时间稳定轮数
      - contradiction：是否检测到内部矛盾
    输出：
      - confirmed：完全确认（操作）
      - tier：ACCEPT_weak / ACCEPT_strong / ACCEPT_stable（C4 基础）
    """

    def __init__(self, conf_threshold: float = CONF_THRESHOLD,
                 kl_threshold: float = KL_THRESHOLD,
                 stable_rounds: int = STABLE_ROUNDS):
        self.conf_threshold = conf_threshold
        self.kl_threshold = kl_threshold
        self.stable_rounds = stable_rounds

    def evaluate(self, channel_creds: Dict[str, float],
                 realized_kl: float = 0.0,
                 stable_rounds: int = 0,
                 contradiction: bool = False) -> Dict:
        """评估确认状态：返回确认度分层（weak/strong/stable）与完全确认判定。"""
        # 条件①：所有相关通道可信度达标
        # #1075①：非数值通道（JSON/外部数据解码后的 str/None/list/…）按
        # fail-closed 处理——与同入口对 NaN 的既有路径一致（`nan >= θ` 恒 False
        # ⇒ 条件不满足）。旧实现直接比较 ⇒ TypeError 打断整次评估。
        def _cred_ok(c) -> bool:
            try:
                c = float(c)
            except (TypeError, ValueError):
                return False
            # +inf 会以「≥ 阈值」的面目静默达标（同 #402 的 NaN 放行族）⇒
            # 非有限一律不达标。
            return math.isfinite(c) and c >= self.conf_threshold
        ch_ok = bool(channel_creds) and all(
            _cred_ok(c) for c in channel_creds.values())
        # 条件②：至少一次强验证命中（realized_KL > θ_g）
        # #1958：非数值/非有限 realized_kl 按「无强验证」处理——与条件①（#1075①）及
        # 同文件 kl_binary/gain_task/value（#402）同一 fail-closed 口径。旧实现直接
        # 比较：`inf > θ` 恒真 ⇒ 非有限读数静默换到 ACCEPT_stable（比「零证据」更宽松）；
        # `str`/`None` 抛 TypeError 打断整次评估；`nan` 虽不达标却把 nan 原样回填 values
        # （下游 json.dumps(allow_nan=False) 即炸）。
        try:
            kl_value = float(realized_kl)
        except (TypeError, ValueError):
            kl_value = 0.0
        if not math.isfinite(kl_value):
            kl_value = 0.0
        strong_ok = kl_value > self.kl_threshold
        # 条件③：跨时间稳定
        # #1958：同款前置校验——非数值/非有限轮数按 0 处理（`inf >= θ` 恒真会把
        # 「跨时间稳定」判成已达成，`str` 抛 TypeError）。合法值原样回填。
        try:
            rounds_value = float(stable_rounds)
            rounds_valid = math.isfinite(rounds_value)
        except (TypeError, ValueError):
            rounds_value = 0.0
            rounds_valid = False
        if not rounds_valid:
            rounds_value = 0.0
        stable_ok = rounds_value >= self.stable_rounds
        # 条件④：无内部矛盾
        no_conflict = not contradiction

        confirmed = ch_ok and strong_ok and stable_ok and no_conflict

        # 确认度分层（C4）：weak / strong / stable
        if confirmed:
            tier = "ACCEPT_stable"
        elif ch_ok and strong_ok and no_conflict:
            tier = "ACCEPT_strong"
        elif ch_ok and no_conflict:
            tier = "ACCEPT_weak"
        else:
            tier = "NOT_ACCEPTED"

        # 回填守卫后的读数（#1958）：nan/inf/非数值不得进入返回体——该返回体是逐行
        # JSON-RPC 观测面，`json.dumps(..., allow_nan=False)` 遇 nan/inf 直接抛
        # ValueError，整条链路会在序列化处炸。合法输入原样回填（形态不变）：int 轮数仍回填 int。
        kl_out = round(kl_value, 6)
        rounds_out = int(rounds_value) if type(stable_rounds) is int else rounds_value
        return {
            "confirmed": confirmed,
            "tier": tier,
            "conditions": {
                "channels_ok": ch_ok,
                "strong_verification_ok": strong_ok,
                "stable_ok": stable_ok,
                "no_contradiction": no_conflict,
            },
            "values": {
                "realized_kl": kl_out,
                "stable_rounds": rounds_out,
            },
        }
