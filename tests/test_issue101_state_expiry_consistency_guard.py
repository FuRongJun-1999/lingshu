# -*- coding: utf-8 -*-
"""test_issue101_state_expiry_consistency_guard · `state()` 不得在已过期时报 stable
============================================================================
缺陷（lingshu issue #101 · 残余事实 2）：
`StableLease.state()` 旧实现 `out = {"state": lease["state"], ...}` 直接把**存储字段**
当结论返回，既不判过期、也不过 `_stable_now()` 谓词。于是同一次返回体里同时给出
    state="stable"  与  expires_at <= now
两个互相冲突的字段——实测 `expires_at - now = -0.285`，5/5 次复跑一致。

后果：只读 `state()` 而不读 `check()` 的调用方，会把**已过期**租约当成 stable。
`check()` 在 #101/#458 重构后已改走 `_stable_now()`，`state()` 是**被漏在同一重构
之外**的那一处（`renew()` / `acquire()` 的返回体都经 `state()`，故一并受益）。

修法：`state()` 先经 `_stable_now()`（`check()` / `_acquire_ok()` 的同一谓词）求值，
谓词不成立即报 `weak`。**纯读、无副作用**：不改写 `lease["state"]`（落状态仍是
`check()` 的职责，P1-003「降级作用于状态」不变）。键集与修前**逐位相同**——
不新增 `reason` 等字段（#859 守卫钉死 LEGACY_KEYS），需要原因文本仍走 `check()`。

守卫断言组（**只回滚 `stable_lease.py` 的 `state()` 方法 ⇒ A/B 必红**）：
  A 组（核心判据 · 有牙）：TTL 过期后 `state()["state"] == "weak"`，且同一返回体
     内部不再自相矛盾（`state=="stable"` 与 `expires_at <= now` 不得同时成立）。
     旧实现报 `stable` ⇒ 直接红。
  B 组（一致性 · 有牙）：`state()` 与 `check()` 在同一时刻必须同结论——覆盖
     过期 / 已降级 / 置信度衰减过弱阈值 / 时钟回拨 / 非有限 五种不成立分支。
  C 组（正对照 · 防误杀）：健康租约仍报 `stable`；键集逐位不变（无新增键）。
  D 组（无副作用 · 契约）：`state()` 是纯读——调用前后 `lease["state"]` 逐位不变
     （过期时也不得被 `state()` 改写成 weak，改写只由 `check()` 落）。

判据来源：
  · issue #101 正文「事实 2」——依据取自**返回体自身**（state 与 expires_at 互斥），
    不主张「state() 应当计算过期」这一设计判断，只主张内部矛盾不存在。
  · 3.2.2 `S ∈ stable ⟺ (t-t_verified) < TTL ∧ ¬conflict`（`stable_lease.py` 模块头）。

定点变异自证（抽掉修复 ⇒ 必红）：
  把 `state()` 的 `self._stable_now(lease)` 求值去掉、恢复 `lease["state"]` 直返
  ⇒ A1/A2/B* 红（过期租约报 stable）。实测：回滚源码后 8/19 通过（11 项红）。

运行（仓根，两种都支持）：
    python -X utf8 tests/test_issue101_state_expiry_consistency_guard.py
    python -X utf8 -m pytest tests/test_issue101_state_expiry_consistency_guard.py
"""
from __future__ import annotations

import os
import sys
import time
import warnings

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
warnings.filterwarnings("ignore")

from lingshu.world.stable_lease import StableLease  # noqa: E402

TTL = 3600.0
# 修前 `state()` 的键集（#859 LEGACY_KEYS 同口径）——修后必须逐位不变
LEGACY_KEYS = {"key", "state", "t_verified", "ttl", "confidence",
               "expires_at", "ttl_expires_at", "weak_at"}


def _forced(age, ttl=TTL, confidence=1.0, state="stable"):
    """构造 t_verified = now - age 的租约（不经 sleep）。"""
    lea = StableLease()
    lea.acquire("k", ttl=ttl, confidence=confidence)
    lea._leases["k"].update({"t_verified": time.time() - age, "ttl": ttl,
                             "confidence": confidence, "state": state})
    return lea


def run_guard(verbose=True):
    """执行全部断言，返回 (通过项列表, 失败项列表)。"""
    passed, failed = [], []

    def chk(name, cond, detail=""):
        (passed if cond else failed).append(name)
        if verbose:
            print(f"  {'PASS' if cond else 'FAIL'} {name}"
                  + (f"  <- {detail}" if detail else ""))
        return bool(cond)

    if verbose:
        print("== A 组：#101 事实 2 —— 过期后 state() 不得报 stable ==")

    # A1/A2：真实过期（sleep 超过 TTL），逐次复跑
    for i in range(3):
        lea = StableLease()
        lea.acquire("k", ttl=0.05, confidence=1.0)
        time.sleep(0.30)
        st = lea.state("k")
        delta = st["expires_at"] - time.time()
        chk(f"A1.{i} sleep 过期后 state() 报 weak（delta={delta:+.3f}）",
            st["state"] == "weak", f"<- {st['state']}")
        chk(f"A2.{i} 返回体不再自相矛盾（state=stable 且 expires_at<=now）",
            not (st["state"] == "stable" and delta <= 0),
            f"<- state={st['state']} delta={delta:+.3f}")

    # A3：不经 sleep，直接摆到超 TTL 的年龄
    lea = _forced(TTL + 10.0)
    chk("A3 超 TTL 的租约 state() 报 weak", lea.state("k")["state"] == "weak",
        f"<- {lea.state('k')['state']}")

    if verbose:
        print("== B 组：state() 与 check() 在五种不成立分支上必须同结论 ==")
    cases = [
        ("超 TTL", _forced(TTL + 10.0)),
        ("已降级", _forced(0.0, state="weak")),
        ("置信度衰减过弱阈值", _forced(3000.0, confidence=0.5)),
        ("时钟回拨", _forced(-1e7)),
        ("非有限 confidence", _forced(0.0, confidence=float("nan"))),
    ]
    for label, lea in cases:
        s, c = lea.state("k")["state"], lea.check("k")["state"]
        chk(f"B {label}: state={s} check={c} 一致", s == c,
            f"<- state={s} check={c}")
        chk(f"B {label}: 两者均为 weak", s == "weak" and c == "weak",
            f"<- state={s} check={c}")

    if verbose:
        print("== C 组：正对照 —— 健康租约仍 stable，键集不得变 ==")
    lea = StableLease()
    lea.acquire("k", ttl=TTL)
    chk("C1 健康租约 state() 报 stable", lea.state("k")["state"] == "stable",
        f"<- {lea.state('k')['state']}")
    chk("C2 健康租约 check() 报 stable", lea.check("k")["state"] == "stable")
    chk("C3 键集与修前逐位相同（无新增键）",
        set(lea.state("k").keys()) == LEGACY_KEYS,
        f"<- 差异 {sorted(set(lea.state('k').keys()) ^ LEGACY_KEYS)}")
    chk("C4 未知 key 仍报 unknown",
        StableLease().state("nx")["state"] == "unknown")

    if verbose:
        print("== D 组：state() 无副作用（改写状态仍是 check() 的职责）==")
    lea = _forced(TTL + 10.0)
    before = lea._leases["k"]["state"]
    lea.state("k")
    after = lea._leases["k"]["state"]
    chk("D1 state() 不改写内部 lease['state']", before == after,
        f"<- before={before} after={after}")
    chk("D2 state() 报 weak 但内部仍是原值（纯读）", after == "stable", f"<- {after}")
    lea.check("k")
    chk("D3 check() 才把内部状态落成 weak（P1-003 口径不变）",
        lea._leases["k"]["state"] == "weak", f"<- {lea._leases['k']['state']}")

    return passed, failed


def test_issue101_state_expiry_consistency_guard():
    """pytest 入口：任一断言失败即红。"""
    passed, failed = run_guard(verbose=False)
    assert not failed, f"失败 {len(failed)}/{len(passed) + len(failed)} 项: {failed}"


if __name__ == "__main__":
    _p, _f = run_guard(verbose=True)
    print()
    print("===== SUMMARY %d/%d 通过 =====" % (len(_p), len(_p) + len(_f)))
    if _f:
        print("失败项:", _f)
        sys.exit(1)
    print("VERDICT=PASS（#101 事实 2：state() 按现算谓词报结论、与 check() 一致、"
          "键集不变、纯读无副作用）")
