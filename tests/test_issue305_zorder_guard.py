# -*- coding: utf-8 -*-
"""#305 守卫：`reconstruct` 铺 bbox 必须按面积降序（大区域先铺）。

缺陷面（公开 issue #305）：`reconstruct` 按 `extract_regions` 的扫描序逐个铺
bbox——物体占住 (0,0) 时先于背景入列，被背景整块盖掉（实测 kept=0/25）。
本件修为面积降序（面积是连接描述里已有字段，不引入新语义；同面积稳定排序）。

本文件锁：
  ① 角落方块 (0,0)：kept=25/25（修前 0/25）；
  ② 图内方块 (14,14)：kept=25/25；
  ③ 边缘方块 (0,27)/(27,0)：kept≥20/25（1px 量化偏移为描述格式 4 位小数的
     既有伪影，如实声明，不属层序面）；
  ④ 反向对照：无前景的纯背景重建仍为整幅背景色。
定点变异：去掉排序 ⇒ ① 红。
"""
import numpy as np
import pytest

from lingshu.nn.hex_recon import (extract_regions, pixel_accuracy,
                                  reconstruct, regions_to_connections)


def _case(y0, x0, wd=5):
    img = np.full((32, 32, 3), 200, np.uint8)
    img[y0:y0 + wd, x0:x0 + wd] = (200, 40, 40)
    regs, _ = extract_regions(img)
    rec = reconstruct(regions_to_connections(regs, img.shape))
    return img, rec


def _kept(img, rec, y0, x0, wd=5):
    obj = rec[y0:y0 + wd, x0:x0 + wd].astype(int)
    return int((np.abs(obj - [200, 40, 40]).sum(-1) <= 8).sum())


# ---------------------------------------------------------------- ①② 主判据

@pytest.mark.parametrize("y0,x0", [(0, 0), (14, 14)])
def test_object_block_survives_reconstruction(y0, x0):
    img, rec = _case(y0, x0)
    assert _kept(img, rec, y0, x0) == 25, (
        "方块@(%d,%d) 被背景整块盖掉（kept=%d/25）——层序未修（#305）"
        % (y0, x0, _kept(img, rec, y0, x0)))


# ---------------------------------------------------------------- ③ 边缘方块

@pytest.mark.parametrize("y0,x0", [(0, 27), (27, 0)])
def test_edge_block_kept_at_least_20(y0, x0):
    """边缘方块的 1px 量化偏移（描述格式 4 位小数）是既有伪影，如实下界。"""
    img, rec = _case(y0, x0)
    assert _kept(img, rec, y0, x0) >= 20, (
        "边缘方块@(%d,%d) 大部分丢失：%d/25" % (y0, x0, _kept(img, rec, y0, x0)))


# ---------------------------------------------------------------- ④ 对照

def test_plain_background_reconstructs_to_background():
    img = np.full((32, 32, 3), 200, np.uint8)
    regs, _ = extract_regions(img)
    rec = reconstruct(regions_to_connections(regs, img.shape))
    assert pixel_accuracy(img, rec) > 0.99
