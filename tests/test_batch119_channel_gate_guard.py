# -*- coding: utf-8 -*-
"""批一百一十九守卫：#2614 split_subjects 通道数 fail-loud（RGBA/灰度裸 reshape 崩）。

覆盖
--------------------------------------------------------------------------
#2614 `split_subjects` 对非 3 通道图像直通管线 ⇒ `bg_color` 的
`reshape(-1, 3)` 裸 ValueError（"cannot reshape array of size 256 into
shape (3)"——不可读且不指参数名，#2528 add_entity pos 裸 float() 已裁决
同款）。修法：入口 fail-loud TypeError（指 shape）。
    G1 行为级：RGBA ⇒ TypeError 且指 shape（M1 撤闸 ⇒ 裸 reshape 崩 ⇒ 红）
    G2 行为级：灰度 2D ⇒ TypeError
    G3 正对照：RGB 3 通道照常走管线（空图 instances=0）
    G4 源码级：闸在位（ndim/shape 校验）

运行（lingshu 仓根）：
    python -X utf8 -m pytest tests/test_batch119_channel_gate_guard.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from lingshu.gen import hexgen_subject_split as SS


def test_G1_rgba_rejected():
    """G1：RGBA ⇒ TypeError 指明 shape（M1 撤闸 ⇒ 裸 reshape ValueError ⇒ 红）。"""
    with pytest.raises(TypeError) as ei:
        SS.split_subjects(np.zeros((8, 8, 4)))
    assert "(8, 8, 4)" in str(ei.value), (
        "错误信息未指明实际 shape：%s" % (ei.value,))


def test_G2_grayscale_rejected():
    """G2：2D 灰度 ⇒ TypeError（同闸的 ndim 腿）。"""
    with pytest.raises(TypeError):
        SS.split_subjects(np.zeros((8, 8)))
    with pytest.raises(TypeError):
        SS.split_subjects(np.zeros((8, 8, 1)))


def test_G3_rgb_unchanged():
    """G3 正对照：RGB 3 通道照常走管线（空图零实例）。"""
    r = SS.split_subjects(np.zeros((8, 8, 3)))
    assert r.get("instances") == [] or len(r.get("instances", [])) == 0


def test_G4_source_pins():
    """G4：源码锚——通道数闸在位（ndim + shape[2] 校验）。"""
    import inspect
    src = inspect.getsource(SS)
    body = src.split("def split_subjects")[1].split("\ndef ")[0]
    assert "img.ndim != 3" in body, "缺 ndim 闸"
    assert 'img.shape[2] != 3' in body, "缺通道数闸"
    assert "#2614" in body, "缺票锚"
