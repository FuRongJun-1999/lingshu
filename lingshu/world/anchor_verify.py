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

# #160：确认（ACCEPT_*）所需的最少**独立通道数**。
#   判据来源：本模块头 7-10「单一通道 = 自证陷阱」、14「多通道一致才确认」，
#   与理论 §3.3 第 2 条「3D锚点可信度 = 多模态交叉 + 物理一致 + 时间一致」
#   （docs/theory/世界模型与语义时空图_完整理论整理与实现路线.md）。
#   单通道时「每维硬下限」all() 恒真 ⇒ 单通道可自称「通道一致」，即自证陷阱。
MIN_CHANNELS = 2

# #160：相邻「稳定轮」的最小时间间隔（秒）——「跨时间稳定」的时间门槛。
#   同一证据在毫秒内被重复喂入只算**同一时刻的一次观测**，不得累加成跨时间稳定轮数。
#   判据来源：条件③「跨时间稳定」与理论 §3.3 第 2 条「时间一致」；具体量级为
#   **工程标定（经验值，追不到理论章节）**，同本文件其它阈值一样可随实测校准
#   （confirmation.py 的 DEV-004 口径）。重新评估条件：实测出现「分钟级复现过严
#   导致合法锚点无法升 stable」或「秒级连调仍被视作跨时间」时重标。
STABLE_MIN_INTERVAL = 60.0


class AnchorVerification:
    """多感知机锚点验证器。

    为每个锚点维护：
      - channel_evidence: {channel: evidence_score}  各通道对该锚点的验证证据
      - channel_conflicts: {channel: reason}         通道矛盾记录
      - verified_rounds: 连续稳定轮数（#160：须时间分离；#400：非稳定轮清零）
      - confirmation: ACCEPT_weak/strong/stable | NOT_ACCEPTED
        （#400：**未验证/无证据时 fail-closed 取 NOT_ACCEPTED**，不得乐观兜底为
          ACCEPT_weak——完全反驳的锚点读数曾是 ACCEPT_weak）
    """

    def __init__(self, graph=None, registry=None, lease=None, clock=None):
        self.graph = graph                # SemanticAnchorGraph（可选）
        self.registry = registry          # ChannelCredibilityRegistry（可选）
        self.lease = lease                # StableLease（可选）
        # 时钟（#160）：可注入以做确定性「跨时间稳定」测试；缺省 wall clock。
        self._clock = clock or time.time
        self._anchors: Dict[str, Dict] = {}
        self._evidence_seq = 0

    # ---- 多通道证据记录 ----

    def add_channel_evidence(self, anchor_id: str, channel: str,
                             evidence: float, strong: Optional[bool] = None, *,
                             observation_tick: Optional[int] = None,
                             evidence_id: Optional[str] = None) -> Dict:
        """记录某通道对锚点的验证证据。

        channel: visual/tactile/audio/action/prediction/search/graph
        evidence: 该通道证据强度 [0,1]（1=完全支持，0=完全反对）
        strong: 显式指定强弱；缺省按通道类型（tactile/action/audio=强）
        observation_tick: 同一观测时刻只可推进一次稳定轮数。
        evidence_id: 同通道重复递交同一证据时幂等（含 registry 更新）。
        缺省时每次提交视为一次新观测；verify 本身不产生新证据。
        """
        rec = self._anchors.setdefault(anchor_id, {
            "channel_evidence": {}, "channel_conflicts": {},
            "verified_rounds": 0, "confirmation": "NOT_ACCEPTED",
        })
        seen = rec.setdefault("_evidence_ids", set())
        key = (channel, evidence_id)
        if evidence_id is not None and key in seen:
            return self.anchor_state(anchor_id)
        if observation_tick is not None:
            if observation_tick < rec.get("_latest_tick", observation_tick):
                return self.anchor_state(anchor_id)
            rec["_latest_tick"] = observation_tick
            rec["_round"] = ("tick", observation_tick)
        else:
            self._evidence_seq += 1
            rec["_round"] = ("submission", self._evidence_seq)
        if evidence_id is not None:
            seen.add(key)
        # #402：NaN/±inf 不得被钳制表达式放行——实测 Python 的 min(1.0, nan)=1.0
        #   会把 NaN 证据抬成"最强支持"（1.0）而绕过全称量闸。非有限值一律
        #   fail-closed 归零（无证据），再钳到 [0,1]。
        evidence = float(evidence)
        finite = math.isfinite(evidence)
        if not finite:
            evidence = 0.0
        evidence = max(0.0, min(1.0, evidence))
        rec["channel_evidence"][channel] = evidence
        # 更新注册表可信度（若有）
        # #119：本方法的 evidence 语义是**支持强度**（1=完全支持，0=完全反对）。
        #   注册表要的是「本次观测的置信度」：命中 ⇒ = evidence；未命中 ⇒ =
        #   **反证强度 = 1 - evidence**。原实现把 evidence 直接当未命中的置信度，
        #   于是反证越强（evidence 越小）扣分越少、evidence=0（完全反驳）时
        #   n_eff=0 零扣分——方向反了。非有限输入无信息（fail-closed，#402）⇒ 不扣分。
        if self.registry is not None:
            is_strong = strong if strong is not None else channel in STRONG_CHANNELS
            if evidence >= 0.5:
                self.registry.record_hit(channel, evidence, strong=is_strong)
            else:
                refutation = (1.0 - evidence) if finite else 0.0
                self.registry.record_miss(channel, refutation, strong=is_strong)
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
            # #400：无任何证据 = 未验证 ⇒ fail-closed 取 NOT_ACCEPTED
            #   （原返回 ACCEPT_weak——零证据被乐观判为「弱确认」）。
            return {"anchor_id": anchor_id, "confirmation": "NOT_ACCEPTED",
                    "verified_rounds": 0, "note": "无任何通道证据"}

        # ① 通道一致：全称量闸（每维硬下限，任一通道证据 < 阈值即资格不过）
        #    对齐 confirmation.py:94-95 口径；不用算术平均——平均会被多弱通道
        #    稀释单通道的强反证（0.0 反证被 4×1.0 弱通道抬到 0.8 而放行＝被淹没的根因）。
        #    #160：单通道时 all() 恒真（＝自证陷阱，见模块头 7-10）⇒ 「通道一致」
        #    至少需要 MIN_CHANNELS 个独立通道，单通道不构成一致，判不过闸。
        ch_ok = (len(ev) >= MIN_CHANNELS
                 and all(v >= CONF_THRESHOLD for v in ev.values()))

        # ② 强验证：至少一个强通道证据 ≥ KL 阈值
        strong_evidence = [v for c, v in ev.items() if c in STRONG_CHANNELS]
        strong_ok = any(v > KL_THRESHOLD for v in strong_evidence) if strong_evidence else False

        # ④ 无矛盾
        no_conflict = len(rec["channel_conflicts"]) < CONFLICT_THRESHOLD

        # ③ 新观测推进稳定轮数；重复读验证结果不得创造新证据。
        fresh = rec.get("_round") != rec.get("_verified_round")
        if ch_ok and strong_ok and no_conflict:
            now = self._clock()
            last = rec.get("last_stable_ts")
            if fresh and (last is None or now - last >= STABLE_MIN_INTERVAL):
                rec["verified_rounds"] += 1
                rec["last_stable_ts"] = now
            rec["_verified_round"] = rec.get("_round")
        else:
            rec["verified_rounds"] = 0
            rec["last_stable_ts"] = None
        stable_ok = rec["verified_rounds"] >= STABLE_ROUNDS

        # 幸存者内加权（只排序）：仅对【已过全称量闸】的幸存通道、按本轮证据
        #   强度加权排序，产出分层依据；零历史依赖（见 _survivor_weighting）。
        survivors = {c: v for c, v in ev.items() if v >= CONF_THRESHOLD} if ch_ok else {}
        rec["tier_basis"] = self._survivor_weighting(survivors)

        # 确认度分层
        # 无矛盾时：按证据/强验证/稳定分层
        # 资格与排序分离：ACCEPT/NOT_ACCEPTED 仅由全称量闸(ch_ok)与②③④决定，
        #   幸存者内加权只产出排序依据，不参与资格判定（故此处无放行支）。
        if ch_ok and strong_ok and stable_ok and no_conflict:
            confirmation = "ACCEPT_stable"
        elif ch_ok and strong_ok and no_conflict:
            confirmation = "ACCEPT_strong"
        elif ch_ok:
            confirmation = "ACCEPT_weak"
        else:
            confirmation = "NOT_ACCEPTED"

        rec["confirmation"] = confirmation
        return self.anchor_state(anchor_id)

    # ---- 幸存者内加权（只排序 · 零历史依赖 · 防马太效应）----

    @staticmethod
    def _survivor_weighting(ev: Dict[str, float]) -> List[Dict[str, float]]:
        """对【已过全称量闸】的幸存通道按【本轮证据强度】加权排序。

        权重 = 该通道本轮证据强度 / 幸存通道证据强度之和。
        权威口径（设计者 2026-10-09 裁决 C-9 / #74）：
          - 资格与排序分离：本函数只产出「分层依据」，不改 ACCEPT/NOT_ACCEPTED；
          - 加权输入用【证据强度】而非票数；
          - 零历史依赖：权重不取自任何累计量（不按累计通过次数自增、不继承历史
            权重），故同一证据重复喂 N 轮权重恒定、不随轮次单调抬升——防马太效应；
            历史权重须等 #119，本件不做。
        """
        if not ev:
            return []
        total = sum(ev.values())
        ranked = sorted(ev.items(), key=lambda kv: (-kv[1], kv[0]))
        return [
            {"channel": ch, "evidence": v,
             "weight": (v / total) if total > 0 else 0.0}
            for ch, v in ranked
        ]

    # ---- 多通道矛盾检测 ----

    def channel_conflict_detect(self, anchor_id: str, channel: str,
                                expected: str, actual: str) -> Dict:
        """检测通道矛盾：某通道观测与锚点声明**不符** → 冲突记录。

        expected: 锚点当前声明（如"椅子"）
        actual: 该通道观测（如"箱子"）
        #401：只有 expected != actual 才是矛盾。原实现**无条件登记**——expected ==
        actual 的「一致观测」也被记成冲突并清零该通道证据，把一致观测译成矛盾。
        返回冲突记录；冲突通道数 ≥ CONFLICT_THRESHOLD → 建议降级。
        """
        rec = self._anchors.setdefault(anchor_id, {
            "channel_evidence": {}, "channel_conflicts": {},
            "verified_rounds": 0, "confirmation": "NOT_ACCEPTED",
        })
        # 归一化后逐字比较（容忍大小写/首尾空白差异；不引入模糊匹配）
        if str(expected).strip().casefold() == str(actual).strip().casefold():
            # 一致观测：不登记冲突、不清零证据（消除「假冲突」）
            result = self.anchor_state(anchor_id)
            result["conflict_detected"] = False
            result["conflict_count"] = len(rec["channel_conflicts"])
            return result
        conflict = {"channel": channel, "expected": expected, "actual": actual,
                    "ts": self._clock()}
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
            "confirmation": rec.get("confirmation", "NOT_ACCEPTED"),
            "tier_basis": list(rec.get("tier_basis", [])),
        }

    def verification_summary(self) -> Dict:
        """全部锚点验证状态摘要。"""
        out = {}
        for aid, rec in self._anchors.items():
            out[aid] = {
                "confirmation": rec.get("confirmation", "NOT_ACCEPTED"),
                "channels": len(rec.get("channel_evidence", {})),
                "conflicts": len(rec.get("channel_conflicts", {})),
            }
        return out
