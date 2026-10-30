# -*- coding: utf-8 -*-
"""守卫 · `renew()` 判据必须与 `_stable_now()` 同谓词（lingshu issue #1956）

缺陷面（issue #1956，锚定 45b8359）
    `StableLease.renew()` 不走本模块声明为**唯一谓词**的 `_stable_now()`，自算一个
    更宽的判据：

        age = now - lease["t_verified"]
        expired = age >= lease["ttl"]                          # 只判 TTL
        params_ok = self._params_ok(lease["ttl"], confidence)  # 只判参数有限
        lease["t_verified"] = now                              # 无条件清零衰减时钟
        lease["confidence"] = confidence                       # 缺省 1.0，确认度刷满
        if params_ok and (evidence or (lease["state"] == "stable" and not expired)):
            lease["state"] = "stable"

    而 `check()` / `state()` / `_acquire_ok()` 全走 `_stable_now()`，其失效判据是
    `age >= ttl` **或** `decayed_conf < weak_threshold`（默认参数下
    t_weak ≈ 2407.9s < TTL 3600s），外加 #729 的负 age 闸与 #196 的非有限闸。

    由此三处与契约相反的行为（均已在本文件钉成判据）：
      A. 已过 t_weak、仍在 TTL 内的租约，`renew()`（无证据）仍判 stable，并把衰减
         时钟清零 ⇒ 无任何验证即可「无限续命」（实测 20 周期 × 3000s = 60000s 仍 stable）。
      B. 同一份状态「直接 renew」判 stable、「先 check 再 renew」判 weak ⇒ 观察行为
         改变授权结果（与 #458 在 `acquire()` 侧修掉的是同一根，`renew()` 侧漏修）。
      C. `t_verified` 在未来（时钟回拨）时 `age >= ttl` 恒 False ⇒ 判 stable，
         并把错误时间戳「修正」成 now 掩盖掉（#729 的闸在 renew 侧缺失）。

修复（最小面）
    `renew()` 改为**先**求值 `self._stable_now(lease, now=now)`（与 check / state /
    _acquire_ok 同一谓词），**再**刷新 `t_verified` / `confidence`：
    `evidence=False` 时只在「谓词现算成立」才维持 stable。带证据的恢复路径不变，
    docstring 承诺的语义不变（未改 docstring 承诺来迁就实现）。

定点变异自证（抽掉修复 ⇒ 必红）
    把 `renew()` 的 `ok, _reason, _extra = self._stable_now(...)` 换回
    `ok = lease["state"] == "stable" and (now - lease["t_verified"]) < lease["ttl"]`
    ⇒ `test_renew_without_evidence_does_not_revive_decayed_lease`、
    `test_renew_does_not_depend_on_check_call`、
    `test_repeated_evidenceless_renew_cannot_keep_lease_alive_forever`、
    `test_future_timestamp_lease_not_revived_by_renew` 四条红。

运行（仓根）：python -m pytest tests/test_issue1956_stable_lease_renew_predicate.py -q
"""
import math
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lingshu.world.stable_lease as _sl  # noqa: E402
from lingshu.world.stable_lease import (  # noqa: E402
    DECAY_GAMMA, DEFAULT_TTL, WEAK_THRESHOLD, StableLease,
)

TTL = DEFAULT_TTL
#: confidence=1.0 时衰减至弱阈值的时刻（issue #1956 用的就是这条算术）。
T_WEAK = math.log(1.0 / WEAK_THRESHOLD) / DECAY_GAMMA


def _forced(age, ttl=TTL, confidence=1.0, state="stable", key="k"):
    """构造 `t_verified = now - age` 的租约（与 test_issue458:27-34 的 `_forced` 同法）。"""
    lea = StableLease()
    lea.acquire(key, ttl=ttl, confidence=confidence)
    lea._leases[key].update({"t_verified": time.time() - age, "ttl": ttl,
                             "confidence": confidence, "state": state})
    return lea


@pytest.fixture()
def frozen_clock(monkeypatch):
    """冻结本模块命名空间里的 `time`（不动 stdlib）：消除墙钟抖动。

    `renew()` 与 `_stable_now()` 都从 `lingshu.world.stable_lease.time` 取时刻，
    替换该名字即可让「age 恰好落在闸上」的用例稳定复现（不 sleep）。
    """
    class _Clock:
        now = 1_700_000_000.0

        @classmethod
        def time(cls):
            return cls.now

        @classmethod
        def advance(cls, dt):
            cls.now += dt

    monkeypatch.setattr(_sl, "time", _Clock)
    return _Clock


def _forced_at(clock, age, ttl=TTL, confidence=1.0, state="stable", key="k"):
    """在冻结时钟下构造 `t_verified = clock.now - age` 的租约。"""
    lea = StableLease()
    lea.acquire(key, ttl=ttl, confidence=confidence)
    lea._leases[key].update({"t_verified": clock.now - age, "ttl": ttl,
                             "confidence": confidence, "state": state})
    return lea


# =============================================== A. 无证据续期不得复活已衰减租约

def test_t_weak_is_below_ttl_probe_is_valid():
    """探针有效性：默认参数下 t_weak 确实明显小于 TTL（否则本组用例无判别力）。"""
    assert T_WEAK < TTL, "t_weak=%.1f 不小于 TTL=%.1f —— 用例构造失效" % (T_WEAK, TTL)
    assert T_WEAK == pytest.approx(2407.9, abs=0.5)


def test_renew_without_evidence_does_not_revive_decayed_lease():
    """★ 核心：age 已过 t_weak 但未超 TTL，无证据 renew 不得判 stable。

    修复前：`_stable_now()` 判 False（'置信度衰减至弱阈值'），`renew()` 却判 stable
    并把 `t_verified` 刷成 now、`confidence` 刷回 1.0 —— 一次无验证的续期即清零
    衰减时钟。
    """
    age = 3000.0                                  # > t_weak(2407.9)，< TTL(3600)
    witness = _forced(age)
    ok, reason, _extra = witness._stable_now(witness._leases["k"])
    assert ok is False and reason == "置信度衰减至弱阈值", (ok, reason)

    lea = _forced(age)
    assert lea.renew("k")["state"] == "weak", "无证据续期复活了已衰减租约"
    # 修复后必须与谓词一致：renew 后的公开读数同样是 weak
    assert lea.check("k")["state"] == "weak"
    assert lea.state("k")["state"] == "weak"


def test_renew_after_decay_requires_evidence_to_revive():
    """对照：同一份已衰减状态，带证据（evidence=True）续期仍可回 stable（不误杀）。"""
    lea = _forced(3000.0)
    assert lea.renew("k", evidence=True)["state"] == "stable"
    assert lea.check("k")["state"] == "stable"


def test_repeated_evidenceless_renew_cannot_keep_lease_alive_forever(frozen_clock):
    """★ 核心：每周期无证据 renew 不得把租约无限续命（issue #1956 用例 D）。

    修复前：20 个周期（每周期 3000s > t_weak）累计 60000s 仍 stable。
    修复后：首个「本周期内已衰减」的周期即回落 weak，此后无证据不得复活。
    """
    lea = StableLease()
    lea.acquire("k", ttl=TTL)
    lea._leases["k"]["t_verified"] = frozen_clock.now
    states = []
    for cycle in range(20):
        frozen_clock.advance(3000.0)           # 每周期流逝 3000s > t_weak
        ok, reason, _e = lea._stable_now(lea._leases["k"], now=frozen_clock.now)
        states.append((cycle + 1, ok, reason, lea.renew("k")["state"]))

    assert states[0][1] is False, "首周期谓词就应为 False（age 3000 > t_weak）"
    assert states[0][2] == "置信度衰减至弱阈值", states[0]
    assert states[0][3] == "weak"
    revived = [row for row in states if row[3] == "stable"]
    assert not revived, (
        "无证据续期把租约无限续命：周期 %r（每周期流逝 3000s > t_weak=%.1f）"
        % (revived, T_WEAK))


@pytest.mark.parametrize("age,expected", [
    (0.0, "stable"),        # 刚验证：无证据续期维持 stable（防误杀正对照）
    (100.0, "stable"),      # 远未衰减
    (2400.0, "stable"),     # t_weak 前一点
    (2500.0, "weak"),       # 过 t_weak（2407.9）
    (3599.0, "weak"),       # 未超 TTL 但已衰减
    (3601.0, "weak"),       # 超 TTL
])
def test_renew_matches_stable_now_across_ages(age, expected):
    """判据矩阵：同一份状态下 `_stable_now()` / `check()` / `renew()` 结论必须一致。"""
    probe = _forced(age)
    ok, _reason, _extra = probe._stable_now(probe._leases["k"])
    checked = _forced(age).check("k")["state"]
    renewed = _forced(age).renew("k")["state"]
    assert checked == ("stable" if ok else "weak"), "_stable_now 与 check 不一致（探针失效）"
    assert renewed == checked == expected, (
        "age=%r：renew=%r check=%r（期望 %r）" % (age, renewed, checked, expected))


# ========================================== B. 观察顺序不得改变授权结果

def test_renew_does_not_depend_on_check_call():
    """★ 核心：`renew()` 的结论不得取决于此前是否调用过 `check()`（#458 同根）。"""
    age = 3000.0

    def run(do_check):
        lea = _forced(age)
        mid = None
        if do_check:
            mid = lea.check("k")["state"]
            assert mid == "weak", "check() 本身须把衰减租约判 weak（探针有效）"
        return mid, lea.renew("k")["state"]

    mid_a, with_check = run(True)
    mid_b, without_check = run(False)
    assert mid_a == "weak" and mid_b is None
    assert with_check == without_check == "weak", (
        "renew 结果取决于是否调用过 check()：先 check=%r / 直接=%r"
        % (with_check, without_check))


def test_renew_does_not_depend_on_state_call():
    """`state()` 是纯读（不落 state），故 renew 结论也不得因它而变——两侧均须 weak。"""
    def run(do_state):
        lea = _forced(3000.0)
        if do_state:
            assert lea.state("k")["state"] == "weak"
            # state() 纯读：台账字段仍是 stable（#101 契约），renew 不得因此改判
            assert lea._leases["k"]["state"] == "stable"
        return lea.renew("k")["state"]

    assert run(True) == run(False) == "weak"


def test_degraded_then_renew_stays_weak_without_evidence():
    """已 degrade() 的租约（即使时间上仍新鲜）无证据续期不得复活（#101 不回退）。"""
    lea = StableLease()
    lea.acquire("k")
    lea.degrade("k", reason="冲突")
    assert lea.renew("k")["state"] == "weak"
    assert lea.renew("k", evidence=True)["state"] == "stable"


def test_positive_control_healthy_lease_still_renewable():
    """正对照（防误杀）：本就有效（谓词现算成立）的租约无证据续期仍 stable。"""
    lea = StableLease()
    lea.acquire("k")
    assert lea.renew("k")["state"] == "stable"
    assert lea.check("k")["state"] == "stable"


# ===================================== C. renew 侧补 #729 负 age 闸 / #196 非有限闸

def test_future_timestamp_lease_not_revived_by_renew():
    """★ 核心：`t_verified` 在未来（时钟回拨）时无证据 renew 不得判 stable（#729）。"""
    for age in (-100.0, -600.0, -1e7):
        probe = _forced(age)
        ok, reason, _e = probe._stable_now(probe._leases["k"])
        assert ok is False and "回拨" in reason, (age, ok, reason)
        assert _forced(age).check("k")["state"] == "weak"
        assert _forced(age).renew("k")["state"] == "weak", "age=%r：renew 复明了回拨租约" % age


def test_future_timestamp_extreme_rollback_does_not_overflow():
    """极端回拨下 renew 不得溢出（闸必须落在 exp 计算之前，与 #729 同口径）。"""
    assert _forced(-1e7).renew("k")["state"] == "weak"


def test_nonfinite_confidence_renew_is_not_stable():
    """#196 不回退：以 NaN confidence 续期不得判 stable（params_ok 闸仍在）。"""
    lea = StableLease()
    assert lea.acquire("k")["state"] == "stable"
    assert lea.renew("k", confidence=float("nan"))["state"] == "weak"
    assert lea.check("k")["state"] == "weak"


def test_frozen_clock_boundary_at_t_weak(frozen_clock):
    """边界：age 恰在 t_weak 两侧时 renew 的判据与 `_stable_now()` 同步翻转。

    `decayed_conf < weak_threshold` 是严格小于，故 age 略小于 t_weak 仍 stable、
    略大于即 weak。冻结时钟让这两点可精确构造（不依赖墙钟抖动）。
    """
    just_before = _forced_at(frozen_clock, T_WEAK - 1.0)
    assert just_before.renew("k")["state"] == "stable"

    frozen_clock.advance(0)                    # 同一冻结时刻
    just_after = _forced_at(frozen_clock, T_WEAK + 1.0)
    assert just_after.renew("k")["state"] == "weak"

    at = _forced_at(frozen_clock, T_WEAK)
    # decayed_conf == weak_threshold 时严格小于不成立 ⇒ 仍 stable（与谓词逐字一致）
    assert at.renew("k")["state"] == "stable"


def test_frozen_clock_ttl_boundary(frozen_clock):
    """边界：`age >= ttl` 这一项的翻转点必须由 renew 与谓词共享。

    ⚠ 构造说明：默认 confidence=1.0 时 t_weak≈2407.9s **先于** TTL 触发，TTL 这一
    项在 3599s 处根本轮不到 —— 那时 renew 本就该判 weak（衰减闸）。要看 TTL 闸本身，
    须把 decayed_conf 抬到 TTL 时刻仍在弱阈值之上：confidence=2.0 时
    `t_weak = ln(2.0/0.3)/0.0005 ≈ 3794s > TTL=3600s`，于是 age∈[3599, 3600) 仍
    因衰减闸放行，而 age=3600 起由 `age >= ttl` 拦下。
    """
    assert math.log(2.0 / WEAK_THRESHOLD) / DECAY_GAMMA > TTL, "构造前提不成立"
    before = _forced_at(frozen_clock, TTL - 1.0, confidence=2.0)
    probe = before._leases["k"]
    assert before._stable_now(probe, now=frozen_clock.now)[0] is True, "探针失效"
    assert before.renew("k")["state"] == "stable"

    at = _forced_at(frozen_clock, TTL, confidence=2.0)
    ok, reason, _e = at._stable_now(at._leases["k"], now=frozen_clock.now)
    assert ok is False and reason == "TTL 超时", (ok, reason)
    assert at.renew("k")["state"] == "weak"
