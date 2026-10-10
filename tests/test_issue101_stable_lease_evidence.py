# -*- coding: utf-8 -*-
"""守卫 · stable 租约不得在无证据下复活 / check 不得与 state 口径矛盾（lingshu issue #101）

缺陷（HEAD 修复前，`lingshu/world/stable_lease.py`）
    ① `check()` 只判时间（`expired = age >= lease["ttl"]`），**不读 `lease["state"]`**：
       `degrade()` 之后 `state()` 报 weak，`check()` 却仍报 stable —— 两个公开接口对
       同一租约给出相反读数。
    ② `renew()` 内 `lease["state"] = "stable"` 无证据入参：未做任何验证即可把 weak
       租约「无证据复活」。
    ③ `acquire()` 对已降级租约同样无条件写 `state = stable`，同一根。

    实测（本机 Python 3.12.10，HEAD）：
        L = StableLease(); L.acquire('k'); L.degrade('k')
        L.state('k')  → {'state': 'weak', ...}
        L.check('k')  → {'state': 'stable', 'in_lease': True, ...}   ← 矛盾
        L.renew('k')  → {'state': 'stable', ...}                     ← 无证据复活

修复
    · `check()` 读 `lease["state"]`：已被降级者一律报 weak（无证据不复活）。
    · `renew(key, confidence, evidence=False)`：仅 `evidence=True` 才可从 weak 回
      stable；已有 stable 且未超时的租约无新证据续期维持 stable（本就有效，非提升）。
    · `acquire()` 经 `_acquire_ok()`：已降级租约不得无证据重置回 stable。

判据来源
    · 理论章节：`lingshu/world/stable_lease.py` 模块头引「智能论 v3.4 · 3.2.2」——
      `S ∈ stable ⟺ (t - t_verified) < TTL ∧ ¬conflict(S_pred_hat, S)`。只判时间等于
      丢掉合取项后半（¬conflict）；降级/冲突记录在 `lease["state"]` 上，check 必须读它。
    · 「无证据不得恢复 stable」的接口形态（evidence 入参）是**经验标定**：3.2.2 只
      要求「有验证」才可称 stable，未规定入参名；本件按同仓 `degrade()` 的既有
      「降级作用于状态、不删除记录」（P1-003）口径落地，**追不到**更早的 commit 出处。
    · 阈值（DEFAULT_TTL / DECAY_GAMMA / WEAK_THRESHOLD）**未改**。

定点变异自证（抽掉修复 ⇒ 必红）
    把 `stable_lease.py` 回退到 HEAD 形态（check 不读 state / renew 无条件 stable /
    acquire 只看参数）⇒ 本文件 test_check_never_reports_stable_after_degrade、
    test_renew_without_evidence_does_not_restore_stable、
    test_acquire_without_evidence_does_not_resurrect 三条红。

运行（仓根）：python -X utf8 -m pytest tests/test_issue101_stable_lease_evidence.py -q --no-header
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lingshu.world.stable_lease import StableLease  # noqa: E402


def _degraded():
    lease = StableLease()
    lease.acquire("k")
    lease.degrade("k", reason="冲突")
    return lease


# ------------------------------------------------- ① check 不得与 state 矛盾

def test_check_never_reports_stable_after_degrade():
    """★ 核心：degrade() 之后 check() 必须报 weak，与 state() 同口径。"""
    lease = _degraded()
    assert lease.state("k")["state"] == "weak"
    checked = lease.check("k")
    assert checked["state"] == "weak", checked
    assert checked["in_lease"] is False, checked


def test_check_and_state_agree_across_transitions():
    """正对照 + 一致性：stable→weak→(带证据)stable 全程 check/state 不打架。"""
    lease = StableLease()
    lease.acquire("k")
    assert lease.check("k")["state"] == "stable"
    assert lease.state("k")["state"] == "stable"

    lease.degrade("k")
    assert lease.check("k")["state"] == lease.state("k")["state"] == "weak"

    lease.renew("k", evidence=True)
    assert lease.check("k")["state"] == lease.state("k")["state"] == "stable"


# ------------------------------------------------- ② renew 无证据不得复活

def test_renew_without_evidence_does_not_restore_stable():
    """★ 核心：对已降级租约 renew()（缺省无证据）不得回到 stable。"""
    lease = _degraded()
    assert lease.renew("k")["state"] == "weak"
    assert lease.state("k")["state"] == "weak"
    assert lease.check("k")["state"] == "weak"


def test_renew_with_evidence_restores_stable():
    """正路径：显式带证据续期 ⇒ 可回 stable（修复未把恢复路径焊死）。"""
    lease = _degraded()
    assert lease.renew("k", evidence=True)["state"] == "stable"
    assert lease.check("k")["state"] == "stable"


def test_renew_on_expired_lease_without_evidence_is_not_stable():
    """已超时的租约无证据续期 ⇒ 不得回 stable（超时即失去 ¬conflict 的支撑）。"""
    lease = StableLease()
    lease.acquire("k", ttl=10.0)
    lease._leases["k"]["t_verified"] -= 100.0  # 模拟时间推进越过 TTL
    assert lease.check("k")["state"] == "weak"  # 超时降级
    assert lease.renew("k")["state"] == "weak"
    assert lease.renew("k", evidence=True)["state"] == "stable"


def test_renew_on_healthy_stable_lease_stays_stable():
    """正对照（防误杀）：本就 stable 且未超时的租约，无证据续期仍是 stable。"""
    lease = StableLease()
    lease.acquire("k")
    assert lease.renew("k")["state"] == "stable"


# ------------------------------------------------- ③ acquire 无证据不得复活

def test_acquire_without_evidence_does_not_resurrect():
    """★ 核心：对已降级租约再 acquire() 不得无证据重置回 stable。"""
    lease = _degraded()
    assert lease.acquire("k")["state"] == "weak"
    assert lease.check("k")["state"] == "weak"


def test_acquire_fresh_key_and_healthy_reacquire_are_stable():
    """正对照：全新 key 照旧 stable；本就 stable 的租约重复 acquire 不掉级。"""
    fresh = StableLease()
    assert fresh.acquire("k")["state"] == "stable"
    assert fresh.acquire("k")["state"] == "stable"
    assert fresh.check("k")["state"] == "stable"


def test_degrade_keeps_record_and_unknown_stays_unknown():
    """P1-003 边界未动：降级只作用于状态、记录保留；无记录仍是 unknown。"""
    lease = _degraded()
    assert lease.state("k")["key"] == "k"
    assert lease.check("k")["state"] == "weak"
    assert StableLease().check("nope")["state"] == "unknown"
