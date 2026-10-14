# -*- coding: utf-8 -*-
"""test_issue523_subject_split_size · split_subjects 不再写死 512（#523）
==========================================================================
缺陷（#523，改前实测复现）：切分步用模块常量 ``SIZE = 512`` 建网格/标签数组
（``np.mgrid[0:SIZE,0:SIZE]``、``np.full((SIZE,SIZE), …)`` ×2 处），而 ``sil``
跟随输入形状 ⇒ 非 512×512 输入广播失败：

    300×300  split_subjects=ValueError: operands could not be broadcast … (300,300) (512,512)
    512×512  split_subjects=2（恰好掩盖）
    800×800  split_subjects=ValueError: … (800,800) (512,512)

修法：切分步与全局合并标签面改按**输入形状**（``sil.shape`` / ``img.shape``）。

判据（本件）：
  G1 三尺寸（300/512/800）同场景（两块不相交前景，真值 2）⇒ 全部正常返回
     且实例数恰 2（基线 count_objects 同为 2，逐尺寸一致）；
  G2 非方阵（如 300×480）也正常（H≠W 各轴独立）；
  G3 定点变异（源码形态锚）：``np.mgrid[0:SIZE, 0:SIZE]`` 与
     ``np.full((SIZE, SIZE)`` 形态不得回潮——回潮即 G1 在非 512 输入上必红。

运行（仓根）：python -X utf8 -m pytest tests/test_issue523_subject_split_size.py -q
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lingshu.gen import hexgen_c1_real as C
from lingshu.gen import hexgen_subject_split as X

SRC = Path(__file__).resolve().parents[1] / "lingshu" / "gen" / "hexgen_subject_split.py"


def _scene(h, w):
    img = np.full((h, w, 3), 255.0)
    img[50:150, 50:150] = 0
    img[180:260, 180:260] = 0
    return img


@pytest.mark.parametrize("s", [300, 512, 800])
def test_square_sizes_all_work(s):
    """G1：三尺寸全部正常返回且实例数＝真值 2（改前 300/800 直接 ValueError）。"""
    img = _scene(s, s)
    cl, _ = C.objects_of(img)
    base = C.count_objects(cl)
    got = X.split_subjects(img)
    n = len(got["instances"])
    assert n == 2, ("#523 %d×%d 实例数 %r（期望 2）" % (s, s, n))
    assert base == 2, "对照基线异常：%r" % base


def test_nonsquare_input_works():
    """G2：非方阵（300×480）H/W 各轴独立，正常返回 2。"""
    img = _scene(300, 480)
    got = X.split_subjects(img)
    assert len(got["instances"]) == 2, ("#523 非方阵失败", got)


def test_mutation_size_literal_absent():
    """G3（形态锚·定点变异自证）：SIZE 字面建数组的形态不得回潮。"""
    src = SRC.read_text(encoding="utf-8")
    assert "np.mgrid[0:SIZE, 0:SIZE]" not in src, (
        "#523：mgrid 又回到 SIZE=512 字面——非 512 输入将再次广播失败")
    assert "np.full((SIZE, SIZE)" not in src, (
        "#523：np.full((SIZE, SIZE)…)) 形态回潮——标签面应按 sil.shape/img.shape 建")


if __name__ == "__main__":
    import subprocess
    import sys
    raise SystemExit(subprocess.call(
        [sys.executable, "-X", "utf8", "-m", "pytest", __file__, "-q", "--no-header"]))
