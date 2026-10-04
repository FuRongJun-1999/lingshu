# -*- coding: utf-8 -*-
"""anchor_verify · 多感知机锚点验证（世界模型阶段1 · 里程碑1.4）
============================================================================
核心（荣）：一个事物不能只有视觉一个层面的信息——还有交互（触觉）、
声音（听觉）、其他交互来实现感知。3D 锚点验证 = **多感知机协同验证**。

为什么不止视觉（v3.4 理论）：
  单一通道 = 自证陷阱（蜡苹果看起来完全像，但内部一致性 ≠ 外部真实性）。
  3D 锚点若只靠视觉多视角确认，永远无法区分"像椅子"与"是椅子"——
  需要独立于视觉的通道打破自证闭环（多重一致性 Multi-Consistency）。

验证流程：
  视觉多视角（弱）→ 触觉接触（强）→ 行动物理（强）→ 听觉（独立）
  → 预测 → 图矛盾检测 → 多通道一致才确认

复用组件：
  - channel_credibility.py：6 通道可信度（visual/tactile/audio/action/prediction/search）
  - anchored_verification.py：弱/强分级（tactile/action/audio → 强）
  - confirmation.py：完全确认四条件 + ACCEPT 分层
  - stable_lease.py：锚点 TTL 租约（过期降级）

纯标准库 · 零外部依赖（D-005）
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional

# 强验证通道（物理/行动/触觉/听觉——独立于视觉，打破自证闭环）
STRONG_CHANNELS = {"tactile", "action", "audio"}
# 弱验证通道（感知/图/预测——可能自证）
WEAK_CHANNELS = {"visual", "search", "prediction", "graph"}

# 确认阈值（复用 confirmation 语义）
CONF_THRESHOLD = 0.5      # 通道可信度达标线
KL_THRESHOLD = 0.05       # 强验证 realized_KL 达标线
STABLE_ROUNDS = 3         # 跨时间稳定轮数
CONFLICT_THRESHOLD = 2    # 矛盾通道数（≥2 个通道冲突 → 降级）


class AnchorVerification:
    """多感知机锚点验证器。

    为每个锚点维护：
      - channel_evidence: {channel: evidence_score}  各通道对该锚点的验证证据
      - channel_conflicts: {channel: reason}         通道矛盾记录
      - verified_rounds: 连续稳定轮数
      - confirmation: ACCEPT_weak/strong/stable
    """

    def __init__(self, graph=None, registry=None, lease=None):
        self.graph = graph                # SemanticAnchorGraph（可选）
        self.registry = registry          # ChannelCredibilityRegistry（可选）
        self.lease = lease                # StableLease（可选）
        self._anchors: Dict[str, Dict] = {}

    # ---- 多通道证据记录 ----

    def add_channel_evidence(self, anchor_id: str, channel: str,
                             evidence: float, strong: Optional[bool] = None) -> Dict:
        """记录某通道对锚点的验证证据。

        channel: visual/tactile/audio/action/prediction/search/graph
        evidence: 该通道证据强度 [0,1]（1=完全支持，0=完全反对）
        strong: 显式指定强弱；缺省按通道类型（tactile/action/audio=强）
        """
        rec = self._anchors.setdefault(anchor_id, {
            "channel_evidence": {}, "channel_conflicts": {},
            "verified_rounds": 0, "confirmation": "ACCEPT_weak",
        })
        evidence = max(0.0, min(1.0, float(evidence)))
        rec["channel_evidence"][channel] = evidence
        # 更新注册表可信度（若有）
        if self.registry is not None:
            is_strong = strong if strong is not None else channel in STRONG_CHANNELS
            if evidence >= 0.5:
                self.registry.record_hit(channel, evidence, strong=is_strong)
            else:
                self.registry.record_miss(channel, evidence, strong=is_strong)
        return self.anchor_state(anchor_id)

    # ---- 确认度判定 ----

    def verify_anchor(self, anchor_id: str) -> Dict:
        """聚合多通道证据 → 确认度判定（复用 confirmation 四条件）。

        条件①：通道可信度达标（证据一致）
        条件②：至少一次强验证（realized_KL = 强通道证据贡献）
        条件③：跨时间稳定（verified_rounds）
        条件④：无矛盾（channel_conflicts 为空）
        """
        rec = self._anchors.get(anchor_id)
        if rec is None:
            return {"anchor_id": anchor_id, "confirmation": "unknown",
                    "error": "锚点无验证记录"}

        ev = rec["channel_evidence"]
        if not ev:
            return {"anchor_id": anchor_id, "confirmation": "ACCEPT_weak",
                    "verified_rounds": 0, "note": "无任何通道证据"}

        # ① 通道一致：证据均值 ≥ 阈值（允许个别弱通道）
        mean_evidence = sum(ev.values()) / len(ev)
        ch_ok = mean_evidence >= CONF_THRESHOLD

        # ② 强验证：至少一个强通道证据 ≥ KL 阈值
        strong_evidence = [v for c, v in ev.items() if c in STRONG_CHANNELS]
        strong_ok = any(v > KL_THRESHOLD for v in strong_evidence) if strong_evidence else False

        # ③ 跨时间稳定
        stable_ok = rec["verified_rounds"] >= STABLE_ROUNDS

        # ④ 无矛盾
        no_conflict = len(rec["channel_conflicts"]) < CONFLICT_THRESHOLD

        # 确认度分层
        # 无矛盾时：按证据/强验证/稳定分层
        if ch_ok and strong_ok and stable_ok and no_conflict:
            confirmation = "ACCEPT_stable"
        elif ch_ok and strong_ok and no_conflict:
            confirmation = "ACCEPT_strong"
        elif ch_ok:
            confirmation = "ACCEPT_weak"
        elif no_conflict and len(ev) > 0:
            confirmation = "ACCEPT_weak"
        else:
            confirmation = "NOT_ACCEPTED"

        rec["confirmation"] = confirmation
        # 稳定轮数推进
        if confirmation in ("ACCEPT_strong", "ACCEPT_stable"):
            rec["verified_rounds"] += 1
        elif confirmation == "NOT_ACCEPTED":
            rec["verified_rounds"] = 0

        return self.anchor_state(anchor_id)

    # ---- 多通道矛盾检测 ----

    def channel_conflict_detect(self, anchor_id: str, channel: str,
                                expected: str, actual: str) -> Dict:
        """检测通道矛盾：某通道观测与锚点声明不符 → 冲突记录。

        expected: 锚点当前声明（如"椅子"）
        actual: 该通道观测（如"箱子"）
        返回冲突记录；冲突通道数 ≥ CONFLICT_THRESHOLD → 建议降级。
        """
        rec = self._anchors.setdefault(anchor_id, {
            "channel_evidence": {}, "channel_conflicts": {},
            "verified_rounds": 0, "confirmation": "ACCEPT_weak",
        })
        conflict = {"channel": channel, "expected": expected, "actual": actual,
                    "ts": time.time()}
        rec["channel_conflicts"][channel] = conflict
        # 冲突 → 该通道证据清零
        rec["channel_evidence"][channel] = 0.0
        result = self.anchor_state(anchor_id)
        result["conflict_detected"] = True
        result["conflict_count"] = len(rec["channel_conflicts"])
        return result

    # ---- 状态 ----

    def anchor_state(self, anchor_id: str) -> Dict:
        rec = self._anchors.get(anchor_id, {})
        return {
            "anchor_id": anchor_id,
            "channel_evidence": dict(rec.get("channel_evidence", {})),
            "channel_conflicts": {k: {"expected": v["expected"], "actual": v["actual"]}
                                  for k, v in rec.get("channel_conflicts", {}).items()},
            "verified_rounds": rec.get("verified_rounds", 0),
            "confirmation": rec.get("confirmation", "ACCEPT_weak"),
        }

    def verification_summary(self) -> Dict:
        """全部锚点验证状态摘要。"""
        out = {}
        for aid, rec in self._anchors.items():
            out[aid] = {
                "confirmation": rec.get("confirmation", "ACCEPT_weak"),
                "channels": len(rec.get("channel_evidence", {})),
                "conflicts": len(rec.get("channel_conflicts", {})),
            }
        return out
