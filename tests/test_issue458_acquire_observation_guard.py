# -*- coding: utf-8 -*-
"""#458 守卫：`acquire()` 的授权结果不得取决于此前是否调用过 `check()`。

缺陷面（公开 issue #458，2026-10-14 独立复核成立）：
`StableLease._acquire_ok()` 用 `prior["state"] != "stable"` 判断「已降级」，而
`state` 只在 `check()`/`renew()` 里惰性更新。于是同一个已超时（3.2.2 首项
`(t - t_verified) < TTL` 不成立）的租约：

    先 check() 再 acquire()  ⇒ weak
    直接 acquire()          ⇒ stable   （无证据复活）

本件把谓词收进 `_stable_now()`，`check()` 与 `_acquire_ok()` 共用；本文件把
「观察行为不得改变授权」锁成判据，并覆盖 #196/#729（非有限、时钟回拨）两条闸在
acquire 侧同样生效。定点变异：删掉 `_stable_now` 的 `age >= lease["ttl"]` 分支
⇒ `test_acquire_does_not_depend_on_check_call` 与矩阵用例必红。
"""
import math
import time

import pytest

from lingshu.world.stable_lease import StableLease

TTL = 3600.0


def _forced(age, ttl=TTL, confidence=1.0, state="stable", key="k"):
    """构造一个 t_verified = now - age 的租约（不经 sleep，直接摆到目标年龄）。"""
    lea = StableLease()
    lea.acquire(key, ttl=ttl, confidence=confidence)
    row = lea._leases[key]
    row.update({"t_verified": time.time() - age, "ttl": ttl,
                "confidence": confidence, "state": state})
    return lea


# ---------------------------------------------------------------- 主判据

def test_acquire_does_not_depend_on_check_call():
    """`acquire` 的授权结果不得取决于此前是否调用过 `check()`（3.2.2 首项须现算）。"""
    def run(do_check):
        lea = StableLease()
        lea.acquire("k", ttl=0.0)          # age >= ttl 恒成立
        mid = None
        if do_check:
            mid = lea.check("k")["state"]
            assert mid == "weak", "check() 本身必须把超时租约判 weak（探针有效）"
        return mid, lea.acquire("k", ttl=0.0)["state"]

    mid_a, with_check = run(True)
    mid_b, without_check = run(False)
    assert mid_b is None
    assert with_check == without_check == "weak", (
        "acquire 的结果取决于是否调用过 check()：先 check=%r / 直接=%r"
        % (with_check, without_check))


def test_control_large_ttl_both_paths_stable():
    """对照组：租约仍有效时两条路径一致 stable（差异只来自超时判定）。"""
    def run(do_check):
        lea = StableLease()
        lea.acquire("k", ttl=TTL)
        if do_check:
            assert lea.check("k")["state"] == "stable"
        return lea.acquire("k", ttl=TTL)["state"]

    assert run(True) == run(False) == "stable"


@pytest.mark.parametrize("age,expected", [
    (-10.0, "weak"),    # 时钟回拨（#729）
    (0.0, "stable"),    # 刚验证
    (100.0, "stable"),
    (3000.0, "weak"),   # 未到 TTL，但 confidence 衰减过弱阈值
    (3700.0, "weak"),   # 超 TTL
])
def test_check_and_acquire_agree_across_ages(age, expected):
    """同一份租约状态下，`check()` 与 `acquire()` 必须给出同一结论。"""
    checked = _forced(age).check("k")["state"]
    acquired = _forced(age).acquire("k", ttl=TTL)["state"]
    assert checked == acquired == expected, (
        "age=%r 两路径不一致：check=%r acquire=%r（期望 %r）"
        % (age, checked, acquired, expected))


def test_weak_confidence_blocks_acquire_without_check():
    """衰减过弱阈值（但未超 TTL）的租约，acquire 也须现算判 weak。"""
    lea = _forced(3000.0, confidence=0.5)
    assert lea.acquire("k", ttl=TTL)["state"] == "weak"


def test_expired_reaquire_requires_evidence_to_revive():
    """超时后 acquire 判 weak；带证据 renew 才可恢复（#101 的方向不被本件放宽）。

    ★ 构造方式重登记（2026-10-24，随 #101「state() 现算」落地）：
      原实现用 `ttl=0.0` 图省事造「一定超时」的租约。但 `ttl=0.0` 是**退化值**：
      任何时刻都有 `expires_at <= now`，于是「已超时」与「永远不可能 stable」
      被混为一谈。`state()` 改为与 `check()` 同谓词现算后，这条退化构造下
      即便带证据续期也只能得到 weak —— 那是对 ttl=0 的正确结论，不是放宽证据闸。
      改为「正常 TTL + 把 t_verified 摆到超龄」（本文件既有的 `_forced` 手法），
      断言链与判据**逐条不变**：acquire→weak、无证据 renew→weak、带证据→stable。
      判别力仍在：删掉 `renew` 的 evidence 分支 ⇒ 第二条断言当场变红。
    """
    lea = _forced(3700.0)                     # t_verified = now − 3700（> TTL=3600）
    assert lea.acquire("k", ttl=TTL)["state"] == "weak"
    assert lea.renew("k")["state"] == "weak", "无证据续期不得复活"
    assert lea.renew("k", evidence=True)["state"] == "stable"


# ---------------------------------------------------------------- 既有闸的 acquire 侧

def test_nonfinite_prior_lease_not_revived_by_acquire():
    """prior 的 ttl/confidence 非有限：acquire 不得判 stable（#196 同口径）。"""
    for field in ("ttl", "confidence"):
        checked = _forced(0.0)          # acquire 会整条替换租约 ⇒ check 用独立实例
        checked._leases["k"][field] = float("nan")
        assert "非有限" in checked.check("k")["reason"], field
        acquired = _forced(0.0)
        acquired._leases["k"][field] = float("nan")
        assert acquired.acquire("k", ttl=TTL)["state"] == "weak", field


def test_future_timestamp_prior_not_revived_by_acquire():
    """t_verified 在未来（时钟回拨）：acquire 不得判 stable（#729 同口径）。"""
    # 回拨 1e7 秒；旧 check 在此量级会 math range error。acquire 会整条替换租约，
    # 故 check 与 acquire 各用一个实例。
    assert "回拨" in _forced(-1e7).check("k")["reason"]
    assert _forced(-1e7).acquire("k", ttl=TTL)["state"] == "weak"


def test_acquire_on_fresh_key_still_stable():
    """全新 key（无 prior）不受影响：显式进入 stable 的契约保持。"""
    lea = StableLease()
    out = lea.acquire("fresh", ttl=TTL)
    assert out["state"] == "stable"
    assert math.isfinite(out["expires_at"])
