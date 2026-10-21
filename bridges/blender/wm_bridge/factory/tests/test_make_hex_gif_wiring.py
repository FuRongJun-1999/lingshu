# -*- coding: utf-8 -*-
"""R13-O01 守卫：六足步态 GIF 产出脚本必须**先烘焙再渲染**（源码级接线判据）。

缺陷（2026-10-18 外部审计 R13-O01，P1）
------------------------------------
`anim/_det/make_hex_gif.py` 只 build_scene 后直接 render_frames，全文件零
`bake_keyframes` 调用 —— 而 baked 关节（ease / prismatic / static）的唯一落地路径
就是 `bake_keyframes`（`rig.eval_driver_expr` 只覆盖 spin/swing 驱动器）。
结果：GIF 里是静置姿态，与 `gate_gait_contact` 验证的那套运动学不是同一套动作。

判据（源码级，无需 Blender）
---------------------------
  1. 脚本源码里必须出现 `bake_keyframes(` 调用；
  2. 该调用必须在 `render_frames(` **之前**（先烘焙后渲染）；
  3. 烘焙结果必须被记录（`res["bake"]`）——证据可回查，不是静默调用。
"""
from __future__ import annotations

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_FACTORY = os.path.dirname(_HERE)
_SRC = os.path.join(_FACTORY, "anim", "_det", "make_hex_gif.py")


def _source():
    with open(_SRC, encoding="utf-8") as f:
        return f.read()


def test_bake_is_called():
    src = _source()
    assert "bake_keyframes(" in src, "make_hex_gif.py 没有调用 bake_keyframes（R13-O01 回归）"


def test_bake_before_render():
    src = _source()
    i_bake = src.find("bake_keyframes(")
    i_render = src.find("render_frames(")
    assert i_bake != -1 and i_render != -1
    assert i_bake < i_render, "bake_keyframes 必须在 render_frames 之前（先烘焙后渲染）"


def test_bake_result_recorded():
    src = _source()
    assert 'res["bake"]' in src, "烘焙结果必须落进 res（可回查证据，不许静默调用）"
