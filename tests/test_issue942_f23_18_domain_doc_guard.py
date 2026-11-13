# -*- coding: utf-8 -*-
"""#942 / F23-18 守卫：域判定分支行为锁定＋「零容差」边界如实声明。

R23 清单 F23-18：三分支是零容差精确比较（-1e-9 误差翻转入 min-max），
与旧注释「明显超出」矛盾。本件按 #503/#501 先例**只动标注不动判据**：
行为锁定＋声明在位断言；是否加容差属行为面裁定。

运行（lingshu 仓根）：python -X utf8 -m pytest tests/test_issue942_f23_18_domain_doc_guard.py -q
"""
import inspect
import re

import numpy as np
import pytest

from lingshu.nn.stcnn import frames_to_voxel


def _frames(vals):
    return [np.full((4, 4), v, dtype=np.float32) for v in vals]


def test_f23_18_known_domains_unchanged():
    """已知两域行为锁定：0..1 原样、0..255 归一。"""
    a = frames_to_voxel(_frames([0.0, 0.25, 0.5, 1.0]))
    assert abs(float(a.max()) - 1.0) < 1e-6, "0..1 域应原样（不重归一）"
    b = frames_to_voxel(_frames([0.0, 63.75, 127.5, 255.0]))
    assert abs(float(b.max()) - 1.0) < 1e-6, "0..255 域应 /255"
    assert abs(float(b.max() - b.min()) - 1.0) < 1e-6


def test_f23_18_zero_tolerance_is_declared_boundary():
    """零容差是已声明边界。实测注记（如实）：-1e-9 的位移在 float32 于 0.1
    附近的分辨率（eps≈1.2e-8）之下——分支翻转在数值上**不可观测**，F23-18
    的实害仅为「注释宣称明显超出、实现是精确比较」的声明不实（本件已修）。
    若未来加容差改变分支行为，本文件的两域行为锁定仍须成立。"""
    vals = [0.1, 0.5, 1.0]
    frames = [np.full((4, 4), v, dtype=np.float32) for v in vals]
    frames[0] = frames[0] - 1e-9          # 制造浮点误差格（低于 float32 分辨率）
    a = frames_to_voxel(frames)
    b = frames_to_voxel(_frames(vals))
    # 两域行为锁定：0..1 域原样（两者输出均应保持 0.1 原值域语义）
    assert abs(float(a.flatten()[0]) - 0.1) < 1e-6
    assert abs(float(b.flatten()[0]) - 0.1) < 1e-6


def test_f23_18_docstring_declares_zero_tolerance():
    doc = inspect.getsource(frames_to_voxel)
    assert "#942 清单" in doc and "零容差" in doc, (
        "域判定缺「零容差精确比较」的如实声明（F23-18）")
