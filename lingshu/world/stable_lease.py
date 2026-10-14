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
    - check(key)：检查是否仍在租约内（未超时、无冲突、**未被降级**）
    - renew(key, evidence=)：续期（刷新 t_verified；带证据才可从 weak 回 stable）
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
            "state": "stable" if self._acquire_ok(key, eff_ttl, confidence) else "weak", "expires_at": now + eff_ttl,
        }
        # 记录保留（不可遗忘——P1-003 边界）：历史由调用方持有，这里仅标记
        return self.state(key)

    def check(self, key: str) -> Dict:
        """检查租约状态：stable（有效）/ weak（超时·冲突·已降级·未验证）/ unknown（无记录）。"""
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

        # 非有限输入闸（#196）：NaN/±Inf 使 `age >= ttl` 与 `decayed < weak_threshold`
        # 恒为 False ⇒ 控制流直落末尾 stable 分支（「数值坏了」被译成「租约有效」）。
        # 故先拦：任一非有限 ⇒ 降级 weak（不可判，待重新验证），不给 stable。
        # 判据：本模块 3.2.2 `S ∈ stable ⟺ (t-t_verified) < TTL ∧ …`——非有限时
        # 该合取式无法成立，不得判 stable；降级口径见 P1-003（降级≠删除）。
        if not self._finite(age, lease["ttl"], decayed_conf,
                            lease["confidence"], self.gamma,
                            self.weak_threshold):
            lease["state"] = "weak"  # 降级（作用于状态，不删除记录——P1-003）
            return {"key": key, "state": "weak", "in_lease": False,
                    "reason": "非有限输入（NaN/Inf）——不可判，降级待重新验证"}

        # #101：check 只判时间、不读 lease["state"] —— degrade() 显式降级后 check()
        # 仍报 stable，与 state() 报 weak 直接矛盾（3.2.2 的 ¬conflict 合取项被忽略）。
        # 已被降级（state != stable）的租约不得被 check 报回 stable：无证据不复活。
        # 判据：本模块 3.2.2 `S ∈ stable ⟺ (t-t_verified) < TTL ∧ ¬conflict`——冲突/
        # 降级已被记录在 lease["state"] 上，只判时间等于丢掉合取项的后半。
        if lease["state"] != "stable":
            return {"key": key, "state": "weak", "in_lease": False,
                    "reason": "已降级（%s）——须带证据续期方可回 stable" % lease["state"],
                    "age": round(age, 1)}

        if expired or weak_conf:
            lease["state"] = "weak"  # 降级（作用于状态，不删除记录）
            return {"key": key, "state": "weak", "in_lease": False,
                    "reason": "TTL 超时" if expired else "置信度衰减至弱阈值",
                    "age": round(age, 1), "decayed_confidence": round(decayed_conf, 4)}
        return {"key": key, "state": "stable", "in_lease": True,
                "age": round(age, 1), "remaining": round(lease["ttl"] - age, 1),
                "confidence": round(decayed_conf, 4),
                "decayed_confidence": round(decayed_conf, 4)}

    def renew(self, key: str, confidence: float = 1.0, evidence: bool = False) -> Dict:
        """续期：刷新 t_verified/confidence；**仅带证据时**把确认度从 weak 恢复 stable。

        evidence（#101）：本次续期所依据的验证证据（True = 本次验证已通过）。缺省
        False = 无证据 —— 此时只刷新时间戳与确认度，**不把已降级的租约提升回
        stable**。旧实现 `lease["state"] = "stable"` 无证据入参：任何调用者在未验证
        的情况下 renew() 就能「无证据复活」一个 weak 租约，3.2.2 的 ¬conflict 合取项
        形同虚设（与同模块 check() 忽略 lease["state"] 是同一根）。
        已有 stable 租约无新证据续期不降级（本就有效，非「提升」，不需要证据）。
        """
        lease = self._leases.get(key)
        if lease is None:
            return self.acquire(key, confidence=confidence)
        now = time.time()
        age = now - lease["t_verified"]
        expired = age >= lease["ttl"]
        params_ok = self._params_ok(lease["ttl"], confidence)
        lease["t_verified"] = now
        lease["confidence"] = confidence
        # 无证据（evidence=False）时只在「原本就 stable 且未超时」才维持 stable；
        # 已降级或已超时的租约一律回落 weak，须带证据方可回 stable（#101）。
        if params_ok and (evidence or (lease["state"] == "stable" and not expired)):
            lease["state"] = "stable"
        else:
            lease["state"] = "weak"
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

    # ---- 非有限输入闸（#196）----
    # NaN/±Inf 与任何阈值比较恒为 False，`age >= ttl` / `decayed < weak_threshold`
    # 都不成立 ⇒ 控制流直落末尾 stable 分支（「数值坏了」被译成「租约有效」）。
    # 故 acquire/renew/check 先判有限性，任一非有限一律降级 weak（不可判，待重新
    # 验证），不给 stable。判据（理论）：3.2.2 `S ∈ stable ⟺ (t-t_verified) < TTL
    # ∧ ¬conflict`——TTL/置信度非有限时该合取式无法成立，不得判 stable；降级口径
    # 见 P1-003 边界（降级作用于状态，不删除记录）。

    @staticmethod
    def _finite(*xs) -> bool:
        """全部为有限实数（NaN/±Inf → False）。非有限输入闸的唯一判据。"""
        return all(math.isfinite(float(x)) for x in xs)

    def _params_ok(self, ttl, confidence) -> bool:
        """本参数组能否支撑 stable：租约时长/确认度/衰减核/弱阈值皆须有限。"""
        return self._finite(ttl, confidence, self.gamma, self.weak_threshold)

    def _acquire_ok(self, key, ttl, confidence) -> bool:
        """acquire 能否写 stable（#101）：参数须有限，且**不得在无证据下复活已降级租约**。

        旧实现 `state = "stable" if self._params_ok(...) else "weak"` 只看参数，于是对
        一个已 `degrade()`（或 check 超时降级）的租约再调 acquire() 即可无证据重置回
        stable——3.2.2 的 `¬conflict` 合取项形同虚设（与 check 忽略 lease["state"]、
        renew 无条件写 stable 是同一根）。现在：全新 key（调用方显式进入）或本就 stable
        的租约照旧 stable；已降级者保持 weak，须用 `renew(key, evidence=True)` 带证据恢复。
        """
        prior = self._leases.get(key)
        if prior is not None and prior.get("state") != "stable":
            return False
        return self._params_ok(ttl, confidence)
