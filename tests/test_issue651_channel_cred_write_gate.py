# -*- coding: utf-8 -*-
"""test_issue651_channel_cred_write_gate · 写入侧通道名闸守卫（#651）
================================================================================
缺陷（#651，成立）：``ChannelCredibilityRegistry`` 写入侧对通道名零校验——
``_update``（setdefault）与 ``reset``（直接赋值）对任意名字静默建档：
known=True、进 ``registry()``，reset 带 persist 还会 ``save()`` 落盘。同文件
``_load`` 却有 allowlist（``ch not in self._state: continue``）——同文件两条
路径纪律相反。

改前读数（本机，Windows / py3.12，main @ 43ece08）：
  A1 ``reset("__injected__")`` 后 ``registry()`` 含该键；
  A2 该键 ``known=True``；
  A3 带 persist 的 reset 落盘后**文件里**含该键；
  A4 ``record_hit("__ghost__")`` 同样静默建档且无任何警告；
  A6 重启（新实例 _load 同一 persist 文件）后自造键被滤掉——缺口**仅本进程
     有效**，是状态一致性缺口而非裁决污染（未发现改变合法通道读数的证据）。

修法（口径及理由已写入 channel_credibility.py 的 docstring）：
  · 合法通道 = DEFAULT_CHANNELS ∪ 注册表既有键（与 _load 的 allowlist 同口径）；
  · ``reset(未知通道)`` → **raise ValueError**（显式管理 API，传错名属编程
    错误，fail-closed；与 normalize_part / load_shapes 的「非法即 raise」纪律
    一致）。``reset(None)`` 全量复位行为不变；``reset("")`` 修前被 ``if
    channel:`` 当无参走全量复位，现同样 raise（同一个零校验缺口）；
  · ``_update(未知通道)`` → **warn + 跳过本次记账**（数据驱动路径：anchored_
    verification / anchor_verify 把事件自带的通道名透传进来，一条坏通道名的
    一次 hit 不应打断整条验证更新流），不建档、不落盘；
  · ``save()`` 落盘前过滤非 DEFAULT_CHANNELS 键——盘面与 _load 重载面严格
    一致，盘干净是第一道防线。

守卫判据（注入式断言，回退写入闸即红）：
  G1  reset 未知通道：raise ValueError，消息含通道名与合法清单提示；
      registry 不建档；带 persist 时盘上无自造键（raise 先于 save）。
  G2  _update 未知通道（record_hit/record_miss）：发 UserWarning 且不建档、
      不落盘；同一 registry 里合法通道记账不受影响。
  G3（防过修）六条 DEFAULT_CHANNELS 的 record_hit/record_miss/credibility/
      channel_state/reset(单)/reset() 全链零回归（对齐改前读数 A7：
      hit(conf=1.0)+miss(conf=0.5) ⇒ a=12, b=7, cred=12/19）；reset() 无参
      全量复位不变；registry 键集恒为 DEFAULT_CHANNELS。
  G4  save() 盘上无自造键：数据驱动的坏通道被闸后盘干净；直改 _state 模拟
      旧版本残留时 save 仍过滤。

变异验证（两向读数已记录于 #651 修复报告）：
  回退写入闸（恢复 setdefault / 直接赋值 / 不过滤落盘）⇒ G1/G2/G4 红、
  G3 绿；恢复修复 ⇒ 全绿。

判据来源：issue #651 取证 + 同文件 ``_load`` 的 allowlist 自述纪律 +
normalize_part / load_shapes 的「非法即 raise」先例。
运行：pytest tests/test_issue651_channel_cred_write_gate.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingshu.world.channel_credibility import (  # noqa: E402
    DEFAULT_CHANNELS,
    N_BASE,
    PRIOR_A,
    PRIOR_B,
    ALPHA_WEAK,
    ChannelCredibilityRegistry,
)


def _prior_state(ch: str) -> dict:
    return {"channel": ch, "credibility": 0.5, "a": PRIOR_A, "b": PRIOR_B,
            "hits": 0, "misses": 0, "known": True, "status": "ok"}


def _persist_path(prefix: str) -> str:
    return os.path.join(tempfile.mkdtemp(prefix=prefix), "cred.json")


# ---------------------------------------------------------------- G1 reset 闸

def test_g1_reset_unknown_channel_raises_with_legal_list():
    """G1：reset 未知通道 raise ValueError，消息含通道名与合法清单提示。"""
    reg = ChannelCredibilityRegistry()
    before = set(reg.registry())
    with pytest.raises(ValueError) as ei:
        reg.reset("__injected__")
    msg = str(ei.value)
    assert "__injected__" in msg, "错误消息必须点名非法通道（#651）：%r" % msg
    for ch in DEFAULT_CHANNELS:  # 合法清单提示：六条默认通道全列
        assert ch in msg, "错误消息缺合法通道 %r：%r" % (ch, msg)


def test_g1_reset_unknown_channel_does_not_register():
    """G1：reset 未知通道不得建档（registry/known 双读数）。"""
    reg = ChannelCredibilityRegistry()
    with pytest.raises(ValueError):
        reg.reset("__injected__")
    assert "__injected__" not in reg.registry(), (
        "reset('__injected__') 不得静默建档进 registry()（#651）")
    assert reg.channel_state("__injected__")["known"] is False
    assert set(reg.registry()) == set(DEFAULT_CHANNELS)


def test_g1_reset_empty_string_raises_not_full_reset():
    """G1 附带：reset("") 属零校验缺口（修前被 if channel: 当无参全量复位）。"""
    reg = ChannelCredibilityRegistry()
    reg.record_hit("visual", conf=1.0)          # 先制造非先验状态
    with pytest.raises(ValueError):
        reg.reset("")
    assert reg.channel_state("visual")["a"] > PRIOR_A, (
        'reset("") 不得悄悄走全量复位（#651）')


def test_g1_reset_unknown_with_persist_disk_stays_clean():
    """G1：带 persist 的 reset 在 raise 前不落盘——盘上无自造键。"""
    p = _persist_path("issue651g1_")
    reg = ChannelCredibilityRegistry(persist_path=p)
    reg.record_hit("visual", conf=1.0)          # 先产生一次合法落盘
    with open(p, encoding="utf-8") as f:
        snap = json.load(f)
    with pytest.raises(ValueError):
        reg.reset("__injected__")
    with open(p, encoding="utf-8") as f:
        after = json.load(f)
    assert after == snap, "失败的 reset 不得改写盘文件（#651）"
    assert "__injected__" not in after


# ---------------------------------------------------------------- G2 _update 闸

def test_g2_update_unknown_channel_warns_and_skips():
    """G2：record_hit/record_miss 未知通道 warn＋跳过——不建档、不落盘。"""
    p = _persist_path("issue651g2_")
    reg = ChannelCredibilityRegistry(persist_path=p)
    with pytest.warns(UserWarning) as wrec:
        st = reg.record_hit("__ghost__", conf=1.0)
    msg = str(wrec[0].message)
    assert "__ghost__" in msg and "#651" in msg, msg
    # 返回中性先验快照、known=False（与 #539 查询面口径衔接）
    assert st["known"] is False and st["credibility"] == 0.5
    assert st["hits"] == 0 and st["misses"] == 0
    # 不建档
    assert "__ghost__" not in reg.registry()
    assert reg.channel_state("__ghost__")["known"] is False
    # 不落盘：跳过的记账连 save 都不该触发（persist 文件尚不存在）
    assert not os.path.exists(p)
    # record_miss 同闸
    with pytest.warns(UserWarning, match="__ghost2__"):
        reg.record_miss("__ghost2__", conf=1.0)
    assert "__ghost2__" not in reg.registry()


def test_g2_update_unknown_channel_does_not_disturb_legal_accounting():
    """G2：同 registry 里合法通道记账不受坏通道影响。"""
    reg = ChannelCredibilityRegistry()
    with pytest.warns(UserWarning):
        reg.record_hit("__ghost__", conf=1.0)
    st = reg.record_hit("visual", conf=1.0)
    n_eff = 1.0 * N_BASE * min((1.0 - ALPHA_WEAK) * 20.0, 6.0)   # = 10.0
    assert st["a"] == pytest.approx(PRIOR_A + n_eff)
    assert st["hits"] == 1 and st["known"] is True
    assert reg.credibility("visual") == pytest.approx((PRIOR_A + 10.0) / (PRIOR_A + 10.0 + PRIOR_B))


# ---------------------------------------------------------------- G3 防过修

def test_g3_default_channels_full_chain_zero_regression():
    """G3：六条默认通道 record/credibility/state/reset 全链零回归（对齐 A7 基线）。

    ★ 基线重登记（2026-10-24，随 #119 ② 折扣 Beta 落地）：
      本件原断言 a=12.0 / b=7.0 / cred=12/19（A7 改前读数），那是**无遗忘**实现的
      算术。issue #119 ② 要求「每次更新前先 a ← PRIOR_A + α(a − PRIOR_A)」，
      于是同一条 hit→miss 链变成 a = 2 + 0.95×(12 − 2) = 11.5、b 仍 7.0
      （miss 时 b 尚等于先验，折扣对它是恒等），cred = 11.5/18.5。
      这不是「过修」——折扣正是 #119 要的行为；但**必须重登记**而不是放宽判据：
      结构面断言（hits/misses 计数、known、reset 语义、键集、cred 与 a/b 的一致性）
      全部原样保留，只把三个算术常量改到新语义，并在此写明来由。
      判别力仍在：任何**进一步的**未申报算术漂移都会让本件变红。
    """
    reg = ChannelCredibilityRegistry()
    for ch in DEFAULT_CHANNELS:
        reg.record_hit(ch, conf=1.0)
        reg.record_miss(ch, conf=0.5)
        st = reg.channel_state(ch)
        assert st["a"] == pytest.approx(11.5), (ch, st)   # 2 + 0.95×(12−2)（#119 ② 折扣）
        assert st["b"] == pytest.approx(7.0), (ch, st)    # 2 + 5.0（miss 时 b 等于先验，折扣为恒等）
        assert st["hits"] == 1 and st["misses"] == 1
        assert st["known"] is True
        assert reg.credibility(ch) == pytest.approx(11.5 / 18.5)   # A7 原为 12/19
    # 单通道 reset 只复位该通道
    reg.reset("audio")
    assert reg.channel_state("audio") == _prior_state("audio")
    assert reg.channel_state("visual")["a"] == pytest.approx(11.5)
    # 无参全量复位不变
    reg.reset()
    for ch in DEFAULT_CHANNELS:
        assert reg.channel_state(ch) == _prior_state(ch)
    assert set(reg.registry()) == set(DEFAULT_CHANNELS)


def test_g3_reset_none_full_reset_with_persist_roundtrip():
    """G3：reset() 全量复位后 save/_load 回环仍只有六条默认通道。"""
    p = _persist_path("issue651g3_")
    reg = ChannelCredibilityRegistry(persist_path=p)
    for ch in DEFAULT_CHANNELS:
        reg.record_hit(ch, conf=1.0)
    reg.reset()
    reloaded = ChannelCredibilityRegistry(persist_path=p)
    assert set(reloaded.registry()) == set(DEFAULT_CHANNELS)
    for ch in DEFAULT_CHANNELS:
        assert reloaded.channel_state(ch)["a"] == PRIOR_A


# ---------------------------------------------------------------- G4 盘干净

def test_g4_gated_public_path_leaves_disk_clean():
    """G4：数据驱动坏通道被闸后，persist 盘上无自造键。"""
    p = _persist_path("issue651g4_")
    reg = ChannelCredibilityRegistry(persist_path=p)
    reg.record_hit("visual", conf=1.0)          # 合法落盘一次
    with pytest.warns(UserWarning):
        reg.record_hit("__ghost__", conf=1.0)   # 坏通道：warn+跳过
    with open(p, encoding="utf-8") as f:
        disk = json.load(f)
    assert "__ghost__" not in disk, "被闸掉的坏通道不得落盘（#651）"
    assert "visual" in disk


def test_g4_save_filters_non_default_keys():
    """G4：save() 落盘前过滤非注册键——直改 _state 模拟旧版本残留也不落盘。"""
    p = _persist_path("issue651g4s_")
    reg = ChannelCredibilityRegistry(persist_path=p)
    reg.record_hit("visual", conf=1.0)
    # 模拟旧版本进程内残留 / 外部直改内部字典
    reg._state["__legacy__"] = {"a": 1.0, "b": 1.0, "hits": 9.0, "misses": 9.0}
    reg.save()
    with open(p, encoding="utf-8") as f:
        disk = json.load(f)
    assert "__legacy__" not in disk, "save() 须过滤非注册键——盘干净是第一道（#651）"
    assert "visual" in disk
    # 与 _load 重载面一致：重载后也只见六条默认通道
    reloaded = ChannelCredibilityRegistry(persist_path=p)
    assert set(reloaded.registry()) == set(DEFAULT_CHANNELS)


if __name__ == "__main__":
    import pytest as _pytest
    sys.exit(_pytest.main([__file__, "-v"]))
