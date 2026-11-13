"""#1064 守卫：零信息事件（n_eff==0）不得改动后验。

缺陷（修复前实测，锚定 ee4da17）：非有限/零置信度经 #402 归零后 n_eff=0，但 #119
折扣先执行 ⇒ 一次无信息事件扣掉 (1-α) 历史证据（a 201.84→141.89），且 miss-heavy
通道方向反转（可信度 0.0098→0.0139 反而上升；conf=0 时 maintenance→downweighted）。
#402 注释原文承诺「非有限值一律 fail-closed 归零（零伪样本量，后验不动）」。
本文件是「回退即红」对照。
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingshu.world.channel_credibility import ChannelCredibilityRegistry  # noqa: E402


def test_g1_nan_conf_on_trained_channel_leaves_posterior():
    reg = ChannelCredibilityRegistry()
    for _ in range(20):
        reg.record_hit("visual", 1.0, strong=True)
    before = dict(reg._state["visual"])
    reg.record_hit("visual", float("nan"), strong=True)
    after = dict(reg._state["visual"])
    assert after["a"] == before["a"] and after["b"] == before["b"]     # 修复前 a-59.95
    assert after["hits"] == before["hits"]                              # 计数也不动


def test_g2_nan_conf_does_not_raise_credibility_of_miss_heavy_channel():
    reg = ChannelCredibilityRegistry()
    for _ in range(50):
        reg.record_miss("audio", 1.0, strong=True)
    before = reg.channel_state("audio")
    reg.record_miss("audio", float("nan"), strong=True)
    after = reg.channel_state("audio")
    assert after["credibility"] <= before["credibility"] + 1e-12        # 修复前 +0.0041


def test_g3_zero_conf_does_not_upgrade_status():
    reg = ChannelCredibilityRegistry()
    reg.record_miss("audio", 0.1, strong=True)
    reg.record_miss("audio", 0.1, strong=True)
    before = reg.channel_state("audio")
    reg.record_miss("audio", 0.0, strong=True)
    after = reg.channel_state("audio")
    assert after["status"] == before["status"]                          # 修复前 maintenance→downweighted
    assert after["credibility"] == before["credibility"]


def test_g4_normal_evidence_still_updates():
    """防过修正向：正常置信度证据仍按原语义更新与折扣。"""
    reg = ChannelCredibilityRegistry()
    reg.record_hit("visual", 1.0, strong=True)
    assert reg._state["visual"]["a"] > 2.0
    reg.record_miss("visual", 1.0, strong=True)
    assert reg._state["visual"]["b"] > 2.0
