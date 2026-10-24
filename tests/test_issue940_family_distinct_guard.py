# -*- coding: utf-8 -*-
"""#940 守卫：`extended_prior_family` 的「互异」必须由构造保证。

缺陷面（公开 issue #940，第三方 R23）：微噪变体对常幅基核（如均匀 smooth）加
均匀微噪再 L1 归一，扰动恒被坍缩 ⇒ idx 16/26/36 与 idx 6 逐位相同（「每核唯一」
声明不实；#339 修 grow_kernel 回绕时探针已见 fam dup: 16/6, 26/6, 36/6）。

本件锁：
  ① n=37 全族两两 `array_equal` 去重后仍 37 个（无任何重复对）；
  ② 前 10 项语义锚：六方向 edge + smooth + 粗平滑 + laplacian + center_surround
     各就各位（#339 初始核口径不被本件扰动）；
  ③ n=200 大规模压测仍互异。
定点变异：微噪变体回退均匀扰动（坍缩写法）⇒ ① 中 16/6 对复活必红。
"""
import numpy as np
import pytest

from lingshu.nn.hex_cnn import DEFAULT_KERNELS, extended_prior_family


# ---------------------------------------------------------------- ① 全族互异

def test_family_37_all_distinct():
    fam = extended_prior_family(37)
    assert len(fam) == 37
    dups = [(i, j) for i in range(len(fam)) for j in range(i)
            if np.array_equal(fam[i], fam[j])]
    assert not dups, "核族存在逐位重复对：%r" % (dups,)


# ---------------------------------------------------------------- ② 前 10 项锚

def test_first_ten_semantics_preserved():
    fam = extended_prior_family(10)
    by_name = {name: i for i, name in enumerate(
        ["edge_d0", "edge_d1", "edge_d2", "edge_d3", "edge_d4", "edge_d5",
         "smooth", "coarse", "laplacian", "center_surround"])}
    for name in ("smooth", "laplacian", "center_surround"):
        anchor = DEFAULT_KERNELS.get(name)
        assert anchor is not None
        hits = [i for i, f in enumerate(fam) if np.array_equal(f, anchor)]
        assert hits, "语义锚 %s 在扩展族前 10 项丢失" % name


# ---------------------------------------------------------------- ③ 大规模

def test_family_200_still_distinct():
    fam = extended_prior_family(200)
    arrs = [f.tobytes() for f in fam]
    assert len(set(arrs)) == len(arrs), (
        "n=200 大规模下仍有重复核：%d/%d" % (len(set(arrs)), len(arrs)))
