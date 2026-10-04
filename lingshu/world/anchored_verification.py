# -*- coding: utf-8 -*-
"""anchored_verification · 锚定分级验证（智能论 v3.4 · 2.9.1b）
============================================================================
锚定分级（打破 hit 自指循环——GLM 短板二）：
  - 弱验证：感知通道互裁（α=0.95 慢更新）——通道间裁决仍可能共谋
  - 强验证：行动/世界裁决（α=0.7 快更新）——行动路径结构独立于感知

锚定分级从设计决策升级为推导结果：感知互裁的 Gain 天然 < 行动裁决的 Gain
（感知通道本身在信念条件集里——自我条件化），无需外加人为系数。

本模块提供：
  - classify(verification_source)：判定验证归属（弱/强）
  - verify(channel, source, conf, registry)：执行一次分级验证并更新可信度
  - 认知基底层：跨时间稳定 + 预测引擎长期命中率（D-006）——"世界长期不骗人"

纯标准库 · 零外部依赖（D-005）
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, Optional


class VerificationLevel(str, Enum):
    WEAK = "weak"       # 感知通道互裁（慢更新）
    STRONG = "strong"   # 行动/世界裁决（快更新）


# 验证来源 → 分级（工程代理：按来源类型归属）
STRONG_SOURCES = {
    "action",      # 行动执行（咬一口/推一下）
    "world",       # 世界反馈（编译/测试/测量）
    "physical",    # 物理基底（代码真跑）
    "external",    # 外部校准（独立第三方）
}
WEAK_SOURCES = {
    "cross_channel",  # 感知通道互裁
    "visual",         # 单通道自我一致性
    "prediction",     # 预测自洽（非外部）
    "search",         # 检索一致性
}


def classify(source: str) -> VerificationLevel:
    """按验证来源判定锚定分级。未知来源默认弱验证（保守）。"""
    src = (source or "").lower()
    if src in STRONG_SOURCES:
        return VerificationLevel.STRONG
    return VerificationLevel.WEAK


class AnchoredVerification:
    """锚定分级验证器：执行分级验证 + 更新通道可信度。

    verify() 返回 {level, source, channel, credibility_after, gain_note}
    gain_note 说明该验证的信念更新量级（强验证 Gain 天然 > 弱验证）。
    """

    def __init__(self, registry=None):
        # registry: ChannelCredibilityRegistry（B1），duck-typed
        self.registry = registry

    def verify(self, channel: str, source: str, conf: float = 1.0,
               hit: bool = True, registry=None) -> Dict:
        """执行一次锚定分级验证。

        channel: 受验通道（visual/tactile/action/prediction/search...）
        source: 验证来源（action/world/physical/external → 强；其余 → 弱）
        conf: 本次验证置信度 [0,1]
        hit: 验证是否命中
        registry: 通道可信度注册表（缺省用 self.registry）
        """
        reg = registry or self.registry
        level = classify(source)
        strong = level == VerificationLevel.STRONG

        result = {"level": level.value, "source": source, "channel": channel,
                  "hit": hit, "conf": round(conf, 3)}
        if reg is not None:
            if hit:
                st = reg.record_hit(channel, conf, strong=strong)
            else:
                st = reg.record_miss(channel, conf, strong=strong)
            result["credibility_after"] = st.get("credibility")
        else:
            result["credibility_after"] = None

        # 锚定分级的推导结果（GLM：弱验证 Gain 天然 < 强验证 Gain）
        result["gain_note"] = (
            "强验证（行动/世界裁决）：信念更新量天然较大——锚定于外部，快速校准"
            if strong else
            "弱验证（通道互裁）：信念更新量天然较小——感知通道在信念条件集内（自我条件化）")
        return result

    def cognitive_baseline(self, hit_rate: Optional[float] = None,
                           stable_rounds: int = 0) -> Dict:
        """认知基底层：跨时间稳定 + 预测长期命中率（D-006）。

        "世界长期不骗人"的操作性假设——注册表不陷入自指的基底。
        hit_rate: 预测引擎长期命中率（D-006 动态校准）
        stable_rounds: 跨时间稳定轮数
        """
        baseline_ok = True
        notes = []
        if hit_rate is not None and hit_rate < 0.4:
            baseline_ok = False
            notes.append("预测命中率低于基线 0.4（D-006）——基底不稳")
        if stable_rounds < 3:
            notes.append("跨时间稳定轮数不足（<3）——基底层未建立")
        return {"baseline_ok": baseline_ok,
                "hit_rate": round(hit_rate, 4) if hit_rate is not None else None,
                "stable_rounds": stable_rounds,
                "notes": notes}
