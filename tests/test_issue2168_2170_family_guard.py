# -*- coding: utf-8 -*-
"""test_issue2168_2170_family_guard · nn/core 族三处修复回归钉
============================================================================
本件钉 PR fix/verify-batch14 三处修复的**行为**（每条都是修前形态必红）：

  · #2168 nn/hex_unit.UnitHeadNet.set_vec：theta 与 gain 同口径 clip 到
    ±THETA_BOUND——修前 gain 有界（np.clip 0..GAIN_BOUND）、theta 原样写入
    （set_vec(1e6) → theta=1e6、set_vec(nan) → theta=nan），而类 docstring
    声明「两个有界属性」。
  · #2169 core/contradiction._nums：中文数字折算成数值与阿拉伯数字共用值域
    ——修前阿拉伯走 float 归一（25/25.0 收敛）、中文走 "cn:"+原串 字符串键，
    两条路从不汇合 ⇒ 「数量25」与「数量二十五」误报矛盾。解析复用同仓
    retrieval.cn_number()；无法解析时保留 "cn:"+原串 兜底。
  · #2170 core/verdict.assess：布尔遮挡并入字符串域同档——修前
    `occluded != "无遮挡"` 把 True/False 都当「有遮挡」DEFER（理由文案断言
    有遮挡），而 False 断言的是无遮挡 ⇒ 语义反转。用 `is True`/`is False`
    归一（非 `if occluded:`，否则假值重演 #135 falsy 吞噬）。

运行（仓根）：python -X utf8 -m pytest tests/test_issue2168_2170_family_guard.py
"""
import os
import sys

import numpy as np
import pytest

import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


# ---------------------------------------------------------------- #2168

def _unit():
    from lingshu.nn.hex_unit import UnitHeadNet
    from lingshu.nn.hex_hier import HexHierNet
    return UnitHeadNet(HexHierNet(seed=7))


def test_2168_theta_bounded():
    net = _unit()
    k = int(net.mask.sum())
    net.set_vec(np.full(k + net.n_card, 1e6))
    assert net.gain.max() <= net.GAIN_BOUND, "gain 有界（既有行为）"
    assert net.theta.max() <= net.THETA_BOUND, (
        "theta 须与 gain 同口径有界（修前 1e6 原样写入）：%r" % (net.theta.max(),))
    assert net.theta.min() >= -net.THETA_BOUND


# ---------------------------------------------------------------- #2169

def test_2169_cn_number_merges_with_arabic():
    from lingshu.core.contradiction import _nums
    c_ar = _nums("数量25")
    c_cn = _nums("数量二十五")
    assert 25.0 in c_ar, "阿拉伯数字走 float 归一"
    assert c_cn.get(25) == 1 or c_cn.get(25.0) == 1, (
        "中文数字「二十五」须折算成 25 与阿拉伯汇合（修前 'cn:二十五' 字符串键）：%r"
        % (dict(c_cn),))


def test_2169_unparseable_cn_falls_back():
    from lingshu.core.contradiction import _nums
    # 超出 cn_number 解析域（1-999）的写法保留 "cn:" 兜底
    c = _nums("数量三万五千")
    keys = list(c)
    assert any(k == "cn:三万五千" or k == 35000 for k in keys), (
        "无法解析的中文数字保留 cn: 兜底（不引入新误报）：%r" % (keys,))


def test_2169_conflict_no_longer_false_positive():
    """行为面：「数量25」与「数量二十五」不再误报矛盾。"""
    from lingshu.core.contradiction import conflict
    assert conflict("数量25", "数量二十五") is None, (
        "同值不同写法不得误报矛盾（修前 _nums 两路不汇合 ⇒ 误报）")


# ---------------------------------------------------------------- #2170

def _assess(occlusion):
    from lingshu.core.verdict import assess
    part = {"type": "circle", "bbox": [10, 10, 30, 30]}
    return assess(part, {"occlusion": occlusion, "clothing": None,
                         "contrast": None, "line_edge": None}, 400, 400)


def test_2170_false_occlusion_is_accept():
    r = _assess(False)
    assert r["verdict"] == "ACCEPT", (
        "occlusion=False 断言无遮挡 ⇒ 不得报「clothing/occlusion covers」DEFER"
        "（修前与 True 同判、语义反转）：%r" % (r.get("verdict"),))


def test_2170_true_occlusion_is_defer():
    r = _assess(True)
    assert r["verdict"] == "DEFER", "occlusion=True 断言有遮挡 ⇒ DEFER"


def test_2170_string_domain_unchanged():
    assert _assess("无遮挡")["verdict"] == "ACCEPT", "字符串域不变"
    assert _assess("完全遮挡")["verdict"] == "DEFER", "字符串域不变"
