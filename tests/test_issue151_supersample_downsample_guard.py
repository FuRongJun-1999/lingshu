# -*- coding: utf-8 -*-
"""test_issue151_supersample_downsample_guard · lingshu issue #151 守卫

缺陷（issue #151，基线 8a2de8f；在 50ee03b 上实测仍复现）：
  `render_relations` 的 R44 分支把 `size` 当**请求尺寸**去降采，而调用方
  `refine_render` 传入的 `size` 已是 `size*supersample`（**渲染尺寸**）
  ⇒ `resize((size, size))` 与渲染尺寸同值 ⇒ **降采是空操作**：
  `generate_from_text(text, size=48, supersample=4)` 返回 192×192 而非 48×48，
  且注释声称的抗锯齿效果并未发生。

修法（最小 · 按 issue 建议）：降采目标改为**请求尺寸** `size // supersample`。
  另同步修正 `refine_render` 中「supersample≠1 ⇒ 图像尺度变了」的注释/note——
  该理由在修正后已不成立（输出尺度已回到请求尺寸），改为如实理由
  （经 LANCZOS 降采的像素统计与 ATTR_CALIB_96 的直接渲染标定不同源）。

断言组（把 R44 的降采改回 `(size, size)` ⇒ **G1/G3 变红**）：
  G1 验收判据：`generate_from_text(text,size=S,supersample=k)["image"].shape[:2] == (S,S)`
     （S∈{48,96} × k∈{1,2,4}；修前 4/6 不满足）。
  G2 反「另一种错法」的判别力（**降采确实发生**）：k=4 与 k=1 的输出**尺寸相同但像素不同**
     —— 若有人改成「传原始 size 且忽略 supersample」，两者会逐像素相同 ⇒ 本项红。
     （如实标注：这是**代理判据**，不直接度量「渲染分辨率更高」。）
  G3 回归面（不得破坏既有声明）：supersample≠1 ⇒ refine `verifiable=False`、
     `before/after is None`、`stop_reason` 含 `supersample` 与 `不声称成功`。
  G4 边界诚实性（防后人误读）：源码 docstring 必须写明 `log[i]["center"]` 仍是
     **高分辨率坐标**（该换算属维护者决定，#151 附注，本件不动）。

★ 不硬编码仓库路径：根由 `__file__` 上溯两层定位，可用 LINGSHU_ROOT 覆盖。
运行（仓根）：python -X utf8 -m pytest tests/test_issue151_supersample_downsample_guard.py -q --no-header
"""
from __future__ import annotations

import os
import sys
import warnings

ROOT = os.environ.get("LINGSHU_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

from lingshu.nn.hex_gen import (  # noqa: E402
    generate_from_text, render_relations)

TEXT = "左上一个红色的圆"
SIZES = (48, 96)
SUPERSAMPLES = (1, 2, 4)


def test_g1_output_shape_is_requested_size():
    """G1 验收判据：输出边长恒为请求的 size（修前 supersample>1 时为 size*k）。"""
    bad = []
    for size in SIZES:
        for ss in SUPERSAMPLES:
            r = generate_from_text(TEXT, size=size, supersample=ss)
            got = tuple(r["image"].shape[:2])
            if got != (size, size):
                bad.append((size, ss, got))
    assert not bad, "输出尺寸不等于请求尺寸（降采仍是空操作）：%r" % (bad,)


def test_g2_downsampling_actually_used_extra_resolution():
    """G2 判别力：k=4 与 k=1 输出同尺寸但像素不同（挡住「忽略 supersample」的错法）。"""
    parts = [{"shape": "circle", "color": "red", "zone": "r0", "pattern": "solid",
              "size": "medium"}]
    img1, _ = render_relations(parts, size=48, seed=7, supersample=1)
    img4, _ = render_relations(parts, size=48 * 4, seed=7, supersample=4)
    assert img1.shape[:2] == img4.shape[:2] == (48, 48), (img1.shape, img4.shape)
    assert not np.array_equal(img1, img4), (
        "supersample=4 与 =1 的输出逐像素相同 ⇒ 高分辨率渲染未被使用"
        "（很可能是把请求尺寸直接传下去、并忽略了 supersample）")


def test_g3_regression_supersample_still_declines_refinement():
    """G3 回归面：supersample≠1 仍须「不修正、不声称成功」（既有声明不得被破坏）。"""
    for size in SIZES:
        ref = generate_from_text(TEXT, size=size, supersample=2)["refine"]
        assert ref["verifiable"] is False, (size, ref)
        assert ref["before"] is None and ref["after"] is None, (size, ref)
        sr = ref["stop_reason"]
        assert "supersample" in sr and "不声称成功" in sr, (size, sr)


def test_g4_boundary_declared_center_stays_high_res():
    """G4 边界诚实性：源码须声明 log.center 仍是高分辨率坐标（未换算）。"""
    import inspect
    doc = inspect.getdoc(render_relations) or ""
    src = inspect.getsource(render_relations)
    assert "高分辨率坐标" in src, "render_relations 未声明 log.center 的坐标系边界"
    assert "#151" in src, "render_relations 未标注本件的 #151 边界"