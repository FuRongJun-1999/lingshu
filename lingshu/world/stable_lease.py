# -*- coding: utf-8 -*-
"""stable_lease · stable 租约机制（智能论 v3.4 · 3.2.2）
============================================================================
stable 不是终态，是带时间戳的租约：
    stable(S, t_verified, TTL)
    S ∈ stable ⟺ (t - t_verified) < TTL ∧ ¬conflict(S_pred_hat, S)

耦合声明（DEV-005）：TTL 与时空记忆图 confidence 衰减共享同一指数衰减核
exp(-γ·t)——TTL 超时等价于 confidence 衰减至弱验证阈值以下。

记录边界（P1-003）：确认度降级作用于状态，不作用于记录——
验证历史记录仍属 3.2 节不可遗忘记录范畴，降级 ≠ 删除。

纯标准库 · 零外部依赖（D-005）
"""
from __future__ import annotations

import math
import time
from typing import Dict, Optional


DEFAULT_TTL = 3600.0          # 默认租约时长（秒，1 小时）
DECAY_GAMMA = 0.0005          # 指数衰减核 γ（与时空记忆图 confidence 衰减统一）
WEAK_THRESHOLD = 0.3          # 弱验证阈值（TTL 超时等价于 confidence 降至此处以下）


class StableLease:
    """stable 租约管理器：记录/检查/续期/降级。

    - acquire(key, ttl)：进入 stable（记录 t_verified）
    - check(key)：检查是否仍在租约内（未超时且无冲突）
    - renew(key)：续期（验证通过后刷新 t_verified）
    - degrade(key)：主动降级（超时/冲突 → weak 重新验证，不删除记录）
    """

    def __init__(self, ttl: float = DEFAULT_TTL, gamma: float = DECAY_GAMMA,
                 weak_threshold: float = WEAK_THRESHOLD):
        self.ttl = ttl
        self.gamma = gamma
        self.weak_threshold = weak_threshold
        self._leases: Dict[str, Dict] = {}

    def acquire(self, key: str, ttl: Optional[float] = None,
                confidence: float = 1.0, history: Optional[list] = None) -> Dict:
        """进入 stable 状态。history：验证历史记录（不可遗忘，仅记录不删除）。"""
        now = time.time()
        eff_ttl = ttl if ttl is not None else self.ttl
        self._leases[key] = {
            "t_verified": now, "ttl": eff_ttl, "confidence": confidence,
            "state": "stable", "expires_at": now + eff_ttl,
        }
        # 记录保留（不可遗忘——P1-003 边界）：历史由调用方持有，这里仅标记
        return self.state(key)

    def check(self, key: str) -> Dict:
        """检查租约状态：stable（有效）/ expired（超时→降级 weak）/ unknown。"""
        lease = self._leases.get(key)
        if lease is None:
            return {"key": key, "state": "unknown", "in_lease": False,
                    "reason": "无租约记录（从未确认或已降级）"}

        now = time.time()
        age = now - lease["t_verified"]
        expired = age >= lease["ttl"]
        # 指数衰减核 exp(-γ·t)：TTL 超时 = confidence 衰减至弱阈值以下
        decayed_conf = lease["confidence"] * math.exp(-self.gamma * age)
        weak_conf = decayed_conf < self.weak_threshold

        if expired or weak_conf:
            lease["state"] = "weak"  # 降级（作用于状态，不删除记录）
            return {"key": key, "state": "weak", "in_lease": False,
                    "reason": "TTL 超时" if expired else "置信度衰减至弱阈值",
                    "age": round(age, 1), "decayed_confidence": round(decayed_conf, 4)}
        return {"key": key, "state": "stable", "in_lease": True,
                "age": round(age, 1), "remaining": round(lease["ttl"] - age, 1),
                "confidence": round(decayed_conf, 4),
                "decayed_confidence": round(decayed_conf, 4)}

    def renew(self, key: str, confidence: float = 1.0) -> Dict:
        """续期：验证通过后刷新 t_verified（确认度从 weak 回到 stable）。"""
        lease = self._leases.get(key)
        if lease is None:
            return self.acquire(key, confidence=confidence)
        now = time.time()
        lease["t_verified"] = now
        lease["confidence"] = confidence
        lease["state"] = "stable"
        lease["expires_at"] = now + lease["ttl"]
        return self.state(key)

    def degrade(self, key: str, reason: str = "主动降级") -> Dict:
        """主动降级：状态 → weak（记录保留，不可遗忘）。"""
        lease = self._leases.get(key)
        if lease is None:
            return {"key": key, "state": "unknown", "reason": reason}
        lease["state"] = "weak"
        return {"key": key, "state": "weak", "reason": reason,
                "note": "降级作用于状态，不作用于记录——验证历史仍属不可遗忘范畴"}

    def state(self, key: str) -> Dict:
        lease = self._leases.get(key)
        if lease is None:
            return {"key": key, "state": "unknown"}
        return {"key": key, "state": lease["state"],
                "t_verified": round(lease["t_verified"], 1),
                "ttl": lease["ttl"], "confidence": lease["confidence"],
                "expires_at": round(lease["expires_at"], 1)}
