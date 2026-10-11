# -*- coding: utf-8 -*-
"""test_issue171_hex_composite_run_count · 每着色行的「水平色段数」必须一行一段 = 1
============================================================================
背景（lingshu issue #171）：
  `extract_attributes` 的花纹证据 `tx = 每着色行的水平色段数`（见
  `lingshu/nn/hex_composite.py` 该函数 docstring）由下式给出：

      int(np.count_nonzero(row[1:] & ~row[:-1]) + 1)          # 缺陷形态

  `count_nonzero(row[1:] & ~row[:-1])` 数的是掩码里的 **False→True 上升沿**，
  只有「行首为 True」时上升沿数才等于色段数。部件不贴象限左缘时行首为 False
  （实心块、条纹块都是这种多数情形），首段没有上升沿 ⇒ 恒 +1 把 **一行一段
  数成 2**（三段的行数成 4）。实心部件的 `tx` 因此从 1.0 被抬到 2.0。

  连带后果（本件实测，Python 3.12.10 · 合成 v2 数据集 seed=7）：缺省先验
  `tx_solid=1.35`（`extract_attributes` 内 `c = calib or {...}` 缺省值，随
  ac6c0ca 引入，判据来源：经验标定，追不到理论章节）是按**正确口径**标定的
  （实心行 =1.0、非实心最小 ≈1.67，中点 ≈1.33）——缺陷形态下实心件 tx=2.0
  > 1.35 被误判成非实心，**缺省先验的花纹提取精度掉到 37/64 = 57.8%**。

断言组（回退 / 放宽即红）：
  G1 合成掩码直测：一个不贴象限左缘的**实心单段块** ⇒ `tx == 1.0`（缺陷形态
     为 2.0）；行首为 True 的单段块亦为 1.0（防只补不对侧）。
  G2 合成掩码直测：一行**三段**竖条 ⇒ `tx == 3.0`（缺陷形态为 4.0）。
  G3 数据集级：`make_composite_dataset_v2(32, seed=7)` 里所有真值 solid 部件
     在**缺省先验**（calib=None）下 `tx == 1.0`，且缺省先验花纹提取精度 = 100%
     （缺陷形态 57.8%）。
  G4 双副本 `lingshu/gen/hex_composite.py` 与 `lingshu/nn/hex_composite.py`
     逐字节相同（防只修一份；两副本共用同一缺陷）。

判据来源（逐条写清，不编造）：
  · `tx` 的语义 = 每着色行的**水平色段数**——`extract_attributes` docstring 原文，
    一段就是 1，这是本件唯一的判据出处（理论章节无此定义）。
  · 缺省先验 1.35：`extract_attributes` 内 `c = calib or {"tx_solid": 1.35, ...}`；
    随提交 ac6c0ca（lingshu 包模块化导出 v0.1）引入，**经验标定，追不到理论章节**。
  · 本件**不改任何阈值**（1.35 / vert 0.62 / area_cut 均未动），只修正色段计数。

不适用条件 / 已知边界：
  · 本守卫只钉「水平色段数（tx）」这一路的计数；`vert`（垂直游程）与 `span`
    （尺寸）不在本件范围。
  · 只在合成数据集 seed=7 上做数据集级断言；真实语料不在本件范围。

定点变异自证（抽掉修复 ⇒ 必红）：
  把 `+ (1 if row[0] else 0)` 改回 `+ 1` ⇒ G1 报 tx=2.0、G2 报 tx=4.0、G3 精度 57.8%。

运行（仓根）：python -X utf8 tests/test_issue171_hex_composite_run_count.py
             / python -X utf8 -m pytest tests/test_issue171_hex_composite_run_count.py
退出码：0 = 全过；1 = 有断言失败
"""
from __future__ import annotations

import hashlib
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lingshu.nn.hex_composite import (            # noqa: E402
    extract_attributes, calibrate_attributes, make_composite_dataset_v2,
)

COL_RGB = (220, 40, 40)                            # _COLORS_RGB["red"]
COLOR = "red"
QUADRANT = "r0"                                    # 顶-左象限（行 0..31, 列 0..31）


def _blank() -> np.ndarray:
    """96×96 全黑底（与部件色差 |c-ref|.sum=300 > 120 ⇒ 恒不进掩码）。"""
    return np.zeros((96, 96, 3), dtype=np.uint8)


def _tx(img: np.ndarray) -> float:
    r = extract_attributes(img, QUADRANT, COLOR, None)
    assert "tx" in r, f"部件过小未出 tx：{r}"
    return r["tx"]


def test_g1_single_run_row_counts_one():
    """G1：实心单段块（不贴象限左缘）⇒ tx == 1.0。"""
    img = _blank()
    img[8:25, 8:25] = COL_RGB                    # 单段、行首列 8 ≠ 0
    assert _tx(img) == 1.0, f"实心单段行被多数成 {_tx(img)}（应 1.0）"

    img_edge = _blank()
    img_edge[8:25, 0:17] = COL_RGB               # 单段且贴左缘（行首 True）
    assert _tx(img_edge) == 1.0, f"贴左缘单段行 tx={_tx(img_edge)}（应 1.0）"


def test_g2_three_runs_count_three():
    """G2：一行三段竖条 ⇒ tx == 3.0。"""
    img = _blank()
    for c in (8, 12, 16):
        img[8:25, c] = COL_RGB
    assert _tx(img) == 3.0, f"三段行被数成 {_tx(img)}（应 3.0）"


def test_g3_dataset_solid_tx_is_one_and_default_prior_accurate():
    """G3：数据集级——solid 部件缺省先验 tx == 1.0 且花纹提取精度 = 100%。"""
    imgs, metas = make_composite_dataset_v2(32, seed=7)
    solid_tx, ok, n = [], 0, 0
    for img, meta in zip(imgs, metas):
        for p in meta["parts"]:
            r = extract_attributes(img, p["pos"], meta["body_color"], None,
                                   shape_hint=p["shape"])
            if p["pattern"] in ("solid", "striped", "dotted") and r["pattern"] is not None:
                n += 1
                ok += int(r["pattern"] == p["pattern"])
                if p["pattern"] == "solid" and "tx" in r:
                    solid_tx.append(r["tx"])
    assert solid_tx, "数据集里没有可测的 solid 部件"
    assert max(solid_tx) == 1.0, f"solid 部件 tx 上限 {max(solid_tx)}（应 1.0）"
    assert n > 0 and ok == n, f"缺省先验花纹提取精度 {ok}/{n}（应全中）"


def test_g4_dual_copies_byte_identical():
    """G4：gen/ 与 nn/ 的 hex_composite.py 逐字节相同（防只修一份）。"""
    gen = os.path.join(REPO, "lingshu", "gen", "hex_composite.py")
    nn = os.path.join(REPO, "lingshu", "nn", "hex_composite.py")
    with open(gen, "rb") as f:
        hg = hashlib.sha256(f.read()).hexdigest()
    with open(nn, "rb") as f:
        hn = hashlib.sha256(f.read()).hexdigest()
    assert hg == hn, "双副本已漂移：只修了一份？"


if __name__ == "__main__":
    test_g1_single_run_row_counts_one()
    test_g2_three_runs_count_three()
    test_g3_dataset_solid_tx_is_one_and_default_prior_accurate()
    test_g4_dual_copies_byte_identical()
    print("OK #171 守卫全过")
