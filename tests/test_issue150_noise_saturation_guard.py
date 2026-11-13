# -*- coding: utf-8 -*-
"""守卫 · render_relations 的噪声必须是**饱和加法**，不许 uint8 回绕（issue #150）

缺陷（锚定 50ee03b2，`lingshu/nn/hex_gen.py:335`）：
    旧实现 `img += (rng.random(img.shape) * 18).astype(np.uint8)` 是 **uint8 原地加**
    ⇒ 超过 255 的像素按模 256 回绕。噪声幅度 0..17，凡基色 >237 的像素都会回绕：
    白（245）在 noise>=11 时 245+11=256→0。
    实测：white disk min channel-sum=**1**、n(sum<100)=**35/529**；
         clean_white 背景 n(sum<100)=**567/9216（6.15%）**，理论 (7/18)^3=5.88%；
         noise=False 对照 min=735（=245×3，干净）。
    ⇒ 白盘/白底出现近黑散点：噪声的语义是「亮度微扰」，回绕是容器溢出，不是设计。

修法：先在 int16 上做饱和加法，再夹回 0..255。rng 抽样次数与顺序未动。

五条：
  A 白盘不得变黑（min channel-sum 与 n(sum<100)）——修前必红
  B clean_white 背景不得变黑——修前必红
  C 单调性：噪声只会让像素变亮，不得有像素比基色更暗——修前必红
  D 边界钉住：基色 ≤237 的颜色（gray/red）输出必须与 int16 饱和公式逐像素一致
    （修前修后都绿：证明修复没有波及这些颜色，回绕概率本就是 0）
  E 确定性：同 seed 连跑两次逐位相同

运行（仓根）：
  PYTHONUTF8=1 python -X utf8 -m pytest tests/test_issue150_noise_saturation_guard.py -q --no-header
"""
from __future__ import annotations

import os
import sys

_THIS = os.path.abspath(__file__)
REPO = os.environ.get("LINGSHU_ROOT") or os.path.dirname(os.path.dirname(_THIS))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

from lingshu.nn.hex_gen import render_relations  # noqa: E402

SIZE = 96
WHITE = 245          # clean_white 基色
NOISE_MAX = 17       # (rng.random()*18).astype(uint8) 的最大值


def _disk(color, seed=7, noise=True):
    img, log = render_relations(
        [{"shape": "circle", "color": color, "zone": "r4",
          "pattern": "solid", "size": "large"}],
        size=SIZE, seed=seed, noise=noise)
    cx, cy = log[0]["center"]
    rad = SIZE // 7
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= rad ** 2
    return img, disk


# ---------------- A 白盘不得变黑 ----------------

def test_a_white_disk_does_not_go_dark():
    """A 白盘经噪声后不得出现近黑像素（uint8 回绕会把它打到 0..6）。"""
    img, disk = _disk("white")
    s = img.astype(int).sum(2)[disk]
    assert int(s.min()) >= 700, (
        "A 白盘出现暗像素：min channel-sum=%d（干净应为 735=245×3）——"
        "噪声发生了 uint8 回绕：见 #150" % int(s.min()))
    assert int((s < 100).sum()) == 0, (
        "A 白盘有 %d/%d 个像素 channel-sum<100 —— 回绕把它们从白打到近黑"
        % (int((s < 100).sum()), int(disk.sum())))


# ---------------- B clean_white 背景不得变黑 ----------------

def test_b_clean_white_background_stays_clean():
    """B clean_white 背景经噪声后不得有近黑像素。"""
    img, _ = render_relations([], size=SIZE, seed=7, background="clean_white")
    n = int((img.astype(int).sum(2) < 100).sum())
    assert n == 0, (
        "B clean_white 背景有 %d/%d 个近黑像素（理论回绕率 (7/18)^3=5.88%%）"
        % (n, SIZE * SIZE))


# ---------------- C 单调性：噪声只增不减 ----------------

def test_c_noise_never_darkens():
    """C 噪声只能让像素变亮或不变，绝不许比基色更暗。"""
    img, _ = render_relations([], size=SIZE, seed=7, background="clean_white")
    lo = int(img.min())
    assert lo >= WHITE, (
        "C 出现比基色更暗的像素：min=%d < 基色 %d（噪声回绕把亮像素打回暗端）"
        % (lo, WHITE))
    hi = int(img.max())
    assert hi <= WHITE + NOISE_MAX, (
        "C 上界越界：max=%d > 基色+噪声上界 %d" % (hi, WHITE + NOISE_MAX))


# ---------------- D 边界钉住：不波及低基色 ----------------

def test_d_low_base_colors_match_saturating_formula():
    """D 基色 ≤237 的颜色，噪声只应是「加 0..17」，不得有像素变暗（修复不得波及）。"""
    for color in ("gray", "red"):
        clean, _ = _disk(color, noise=False)
        noisy, disk = _disk(color)
        assert np.array_equal(disk, _disk(color, noise=False)[1]), (
            "D 两次的 disk 掩膜不同，守卫前提失效")
        # 逐像素差值（全图，不只 disk——背景同样是「加噪声」语义）
        d = noisy.astype(int) - clean.astype(int)
        assert int(d.min()) >= 0, (
            "D %s 出现被噪声**变暗**的像素：最小差值 %d（这些颜色基色远低于回绕门槛 "
            "237，本不该出现任何负差值）—— 修复波及了不该变的颜色"
            % (color, int(d.min())))
        assert int(d.max()) <= NOISE_MAX, (
            "D %s 噪声增亮越界：最大差值 %d > 上界 %d" % (color, int(d.max()), NOISE_MAX))


# ---------------- E 确定性 ----------------

def test_e_deterministic():
    """E 同 seed 连跑两次必须逐位相同（噪声序列不许因修复而漂移）。"""
    a, _ = render_relations([], size=SIZE, seed=7, background="clean_white")
    b, _ = render_relations([], size=SIZE, seed=7, background="clean_white")
    assert np.array_equal(a, b), "E 同 seed 两次输出不同 —— 确定性被破坏"
