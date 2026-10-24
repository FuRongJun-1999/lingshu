# -*- coding: utf-8 -*-
"""stable_lease · stable 租约机制（智能论 v3.4 · 3.2.2）
============================================================================
stable 不是终态，是带时间戳的租约：
    stable(S, t_verified, TTL)
    S ∈ stable ⟺ (t - t_verified) < TTL ∧ ¬conflict(S_pred_hat, S)

其中 ¬conflict 合取项不取自 check() 入参：冲突经 `degrade()` 显式记录在
`lease["state"]` 上（`check()` 读该状态，见 #101）——本模块不提供独立的冲突
入参或冲突字段（issue #19 如实收窄此处声明，不再声称 check() 自带冲突判定）。

耦合声明（DEV-005，issue #19 校正）：TTL 与时空记忆图 confidence 衰减共享同一
指数衰减核 exp(-γ·t)，但**二者并不等价**——租约的实际失效时刻是两者的**较早者**：
    t_weak(confidence) = ln(confidence / weak_threshold) / γ   （confidence > weak_threshold）
    t_expire = min(TTL, t_weak)
「TTL 超时」与「confidence 衰减至弱验证阈值以下」二者**并非等价**，仅当
confidence = confidence* = weak_threshold · e^(γ·TTL) 这**一个点**上才同时发生：
默认参数下 confidence* ≈ 1.8149，而 acquire() 默认 confidence = 1.0 ⇒
t_weak ≈ 2407.9s < TTL = 3600s（差 33%）。
故 state() 报**实际**失效时刻 `expires_at = min(TTL, t_weak)`，并另给名义
`ttl_expires_at`（TTL 到期）与 `weak_at`（弱阈值触发）两个分项；旧实现只报 TTL
名义到期，与实际失效相差 33%（`conflict` 合取项落在 `lease["state"]` 上，见 #101）。

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
WEAK_THRESHOLD = 0.3          # 弱验证阈值（confidence 衰减至此以下即失效，见 #19）


class StableLease:
    """stable 租约管理器：记录/检查/续期/降级。

    - acquire(key, ttl)：进入 stable（记录 t_verified）
    - check(key)：检查是否仍在租约内（未超时、**未被降级**；冲突经 degrade()
      记入状态，不取自入参——见 #101 / #19）
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
        """进入 stable 状态。history：验证历史记录（不可遗忘，仅记录不删除）。

        #859（兑现契约）：旧实现收下 `history` 却一次不读——docstring 承诺「不可遗忘、
        仅记录不删除」，但记录既没落进租约、也不出现在任何对外读数面（`state()` /
        `check()` 都看不到），调用方传了等于蒸发。现改为：
          · 记录随租约保存，`state()`（以及 `acquire()` 的返回体）在存在记录时透出
            `history` 键（列表）；没有记录时**不新增该键**（既有调用方键集逐位不变）；
          · **只增不删**（P1-003）：再次 `acquire` 不带 history 时旧记录仍在；带
            history 时追加到既有记录之后；
          · 入参须为 list/tuple，否则 `TypeError`（fail-closed，且在**任何写之前**
            抛出——失败的 acquire 不会清掉既有租约与历史）；
          · 存的是**浅拷贝列表**（逐个记录对象不深拷贝）：调用方事后 append/remove
            原列表不影响已存记录，但修改记录对象本身仍会反映（记录对象归调用方）。
        """
        if history is not None and not isinstance(history, (list, tuple)):
            raise TypeError("history 须为 list/tuple（验证历史记录序列），实为 %s"
                            % type(history).__name__)
        prior = self._leases.get(key)
        history_records = list(prior.get("history") or []) if prior else []
        if history:
            history_records.extend(history)
        now = time.time()
        eff_ttl = ttl if ttl is not None else self.ttl
        self._leases[key] = {
            "t_verified": now, "ttl": eff_ttl, "confidence": confidence,
            "state": "stable" if self._acquire_ok(key, eff_ttl, confidence) else "weak",
            # issue #19：对外报的 expires_at 必须是**实际**失效时刻（TTL 与弱阈值
            # 触发取早），而非 TTL 名义到期；分项另给 ttl_expires_at / weak_at。
            **self._expiry_fields(now, eff_ttl, confidence),
        }
        # #859：记录保留（不可遗忘——P1-003 边界）：与租约同存、只增不删。
        if history_records:
            self._leases[key]["history"] = history_records
        return self.state(key)

    def check(self, key: str) -> Dict:
        """检查租约状态：stable（有效）/ weak（超时·冲突·已降级·未验证）/ unknown（无记录）。

        「冲突」不取自本方法入参——由 `degrade()` 记入 `lease["state"]`（3.2.2 的
        ¬conflict 合取项落在状态上，见 #101）；本方法只判时间与状态（issue #19）。
        """
        lease = self._leases.get(key)
        if lease is None:
            return {"key": key, "state": "unknown", "in_lease": False,
                    "reason": "无租约记录（从未确认或已降级）"}

        ok, reason, extra = self._stable_now(lease)
        if not ok:
            lease["state"] = "weak"  # 降级（作用于状态，不删除记录——P1-003）
            out = {"key": key, "state": "weak", "in_lease": False, "reason": reason}
            out.update(extra)
            return out
        out = {"key": key, "state": "stable", "in_lease": True}
        out.update(extra)
        return out

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
        lease.update(self._expiry_fields(now, lease["ttl"], confidence))
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
        """租约台账快照：`state` 字段给出**现算**结论（与 `check()` 同谓词）。

        #101（残余事实 2）：旧实现 `out = {"state": lease["state"], ...}` 直接把
        存储字段当结论返回，**不判过期、不过谓词** ⇒ 同一次返回体里同时给出
        `state="stable"` 与 `expires_at <= now` 两个互相冲突的字段（实测
        `expires_at - now = -0.285`，5/5 次复跑一致）。只读 `state()` 而不读
        `check()` 的调用方，会把已过期租约当成 stable。

        现改为：先经 `_stable_now()`（`check()` / `_acquire_ok()` 的同一谓词）求值，
        谓词不成立即报 `weak`。**纯读、无副作用**：不改写 `lease["state"]`
        （改写是 `check()` 的职责，P1-003 的「降级作用于状态」仍由它落）。
        键集与修前**逐位相同**（#859 守卫钉死），不新增 `reason` 等字段——
        需要原因文本的调用方仍走 `check()`。
        """
        lease = self._leases.get(key)
        if lease is None:
            return {"key": key, "state": "unknown"}
        ok, _reason, _extra = self._stable_now(lease)
        out = {"key": key, "state": lease["state"] if ok else "weak",
                "t_verified": round(lease["t_verified"], 1),
                "ttl": lease["ttl"], "confidence": lease["confidence"],
                # issue #19：expires_at = **实际**失效时刻（min(TTL, 弱阈值触发)），
                # 与 check() 首次报 weak 的时刻一致；名义/分项另给。
                "expires_at": round(lease["expires_at"], 1),
                "ttl_expires_at": round(lease.get("ttl_expires_at",
                                                  lease["expires_at"]), 1),
                "weak_at": round(lease.get("weak_at", lease["expires_at"]), 1)}
        # #859：验证历史（不可遗忘记录）在存在时透出；没有记录时不新增该键，
        # 既有调用方看到的键集逐位不变。返回浅拷贝，调用方改返回值不影响台账。
        history_records = lease.get("history")
        if history_records:
            out["history"] = list(history_records)
        return out

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

    def _expiry_fields(self, now: float, ttl: float, confidence: float) -> Dict:
        """由 (now, ttl, confidence) 算出**实际**失效时刻与两个分项（issue #19）。

        · `ttl_expires_at = now + ttl` —— TTL 名义到期；
        · `weak_at = now + ln(confidence / weak_threshold) / γ` —— confidence 衰减
          至弱验证阈值的时刻（confidence ≤ 弱阈值时落在 now 或更早，即「已然 weak」）；
        · `expires_at = min(ttl_expires_at, weak_at)` —— **实际**失效时刻，与
          `check()` 首次返回 `state == "weak"` 的时刻一致。

        旧实现只写 `expires_at = now + ttl`（TTL 名义到期），而 `check()` 的实际
        失效判据是 `expired or weak_conf` 的**较早者** ⇒ 默认参数下 2407.9s 即失效，
        报时却为 3600s（差 33%）。判据：本模块 3.2.2 `S ∈ stable ⟺ (t - t_verified)
        < TTL ∧ ¬conflict` 与 DEV-005 的共享衰减核 `exp(-γ·t)`——两条件并列时失效
        时刻为两者较早者，报时必须与之一致。参数非有限 / γ≤0 / 对数域非法时退化为
        `ttl_expires_at`（此时 `check()` 的非有限闸已把状态判为 weak，本字段仅作报时）。
        """
        ttl_exp = now + ttl
        weak_at = ttl_exp
        try:
            if (self._finite(ttl, confidence, self.gamma, self.weak_threshold)
                    and self.gamma > 0 and confidence > 0
                    and self.weak_threshold > 0):
                weak_at = now + math.log(confidence / self.weak_threshold) / self.gamma
        except Exception:  # pragma: no cover - 防御：非法参数退回名义到期
            weak_at = ttl_exp
        return {"expires_at": min(ttl_exp, weak_at),
                "ttl_expires_at": ttl_exp, "weak_at": weak_at}

    def _params_ok(self, ttl, confidence) -> bool:
        """本参数组能否支撑 stable：租约时长/确认度/衰减核/弱阈值皆须有限。"""
        return self._finite(ttl, confidence, self.gamma, self.weak_threshold)

    def _stable_now(self, lease, now=None):
        """按 3.2.2 **现算**租约是否 stable——`check()` 与 `_acquire_ok()` 的唯一谓词。

        返回 `(ok, reason, extra)`：`reason` 为不成立时的原因（与 `check()` 对外措辞
        逐字一致），`extra` 为该分支对外报数（age / remaining / decayed_confidence）。
        **不修改 `lease`**：是否把 state 落成 weak 由调用方决定（`check()` 落、
        `_acquire_ok()` 不落——acquire 随后整条替换租约）。

        #458（本方法的来历）：`_acquire_ok` 旧实现只读 `prior["state"]`——而 `state`
        是 `check()`/`renew()` 的**副作用产物**，不是谓词真值。于是一个时间上已超时
        （3.2.2 首项 `(t - t_verified) < TTL` 不成立）的租约：先 `check()` 再
        `acquire()` 判 weak、直接 `acquire()` 判 stable ⇒ **观察行为改变授权结果**。
        与 #101（check 忽略 state）是同一根的两端。现把谓词收在本方法，两条路径共用。

        以下闸的口径与出处（自 check() 原样迁入，勿删）：
        · 负 age 闸（#729）：`t_verified` 落在未来（时钟回拨/容器迁移/VM 挂起恢复/
          双机主备切换/NTP 校正）时 age < 0。3.2.2 的 `(t - t_verified)` 是**已流逝
          时长**，物理上非负；且 DEV-005 的共享衰减核 exp(-γ·t) 在 t < 0 时 > 1
          （确认度反增、越界），`gamma * age` 足够负时更直接 `OverflowError`。
          故不可判 ⇒ weak，且**不**把 decayed_conf 钳成 1.0（那会把「时钟坏了」
          静默译成「租约仍有效」）。闸必须落在 exp 计算**之前**：极端回拨下溢出。
        · 非有限输入闸（#196）：NaN/±Inf 使比较恒 False ⇒ 直落 stable 分支（「数值
          坏了」被译成「租约有效」）。任一非有限 ⇒ weak（不可判，待重新验证）。
        · 已降级闸（#101）：`lease["state"] != "stable"` ⇒ weak（¬conflict 合取项落在
          状态上，无证据不复活）。
        """
        now = time.time() if now is None else now
        age = now - lease["t_verified"]
        if age < 0.0:
            return (False,
                    "t_verified 在未来（时钟回拨）——不可判，降级待重新验证",
                    {"age": round(age, 1)})
        decayed_conf = lease["confidence"] * math.exp(-self.gamma * age)
        if not self._finite(age, lease["ttl"], decayed_conf,
                            lease["confidence"], self.gamma,
                            self.weak_threshold):
            return (False, "非有限输入（NaN/Inf）——不可判，降级待重新验证", {})
        if lease["state"] != "stable":
            return (False,
                    "已降级（%s）——须带证据续期方可回 stable" % lease["state"],
                    {"age": round(age, 1)})
        if age >= lease["ttl"]:
            return (False, "TTL 超时",
                    {"age": round(age, 1),
                     "decayed_confidence": round(decayed_conf, 4)})
        if decayed_conf < self.weak_threshold:
            return (False, "置信度衰减至弱阈值",
                    {"age": round(age, 1),
                     "decayed_confidence": round(decayed_conf, 4)})
        return (True, None,
                {"age": round(age, 1), "remaining": round(lease["ttl"] - age, 1),
                 "confidence": round(decayed_conf, 4),
                 "decayed_confidence": round(decayed_conf, 4)})

    def _acquire_ok(self, key, ttl, confidence) -> bool:
        """acquire 能否写 stable（#101 + #458）：参数须有限，且先租约须**现算**仍 stable。

        旧实现 `state = "stable" if self._params_ok(...) else "weak"` 只看参数，于是对
        一个已 `degrade()`（或 check 超时降级）的租约再调 acquire() 即可无证据重置回
        stable——3.2.2 的 `¬conflict` 合取项形同虚设（与 check 忽略 lease["state"]、
        renew 无条件写 stable 是同一根）。#101 修后改为读 `prior["state"]`，但该字段
        只在 check()/renew() 被调用时更新 ⇒ 同一超时租约「先 check 再 acquire」判
        weak、「直接 acquire」判 stable（#458）。现改为经 `_stable_now` **求值同一
        谓词**，不再读缓存字段：全新 key（调用方显式进入）或本就仍有效的租约照旧
        stable；已降级/已超时/已衰减者保持 weak，须 `renew(key, evidence=True)` 带
        证据恢复。
        """
        prior = self._leases.get(key)
        if prior is not None:
            ok, _reason, _extra = self._stable_now(prior)
            if not ok:
                return False
        return self._params_ok(ttl, confidence)
