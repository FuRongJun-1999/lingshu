# -*- coding: utf-8 -*-
"""test_issue2097_2120_bridge_family_guard · bridges 族十处修复回归钉
============================================================================
本件钉 PR fix/bridges-batch3 十处修复的**行为**（每条都是修前形态必红）：

  · #2097 rig.build_rig：spec 合法的 `easing="bezier"` 解析式关节不再被编译成
    **线性**驱动器（与 eval 真值偏差 0.60 rad）——归 baked（kind="ease"），
    bpy_anim._bezier_joint_links 与 rig 分类解耦（bake_bezier 遍历面）。
  · #2098 raster.parse_svg：viewBox 的 min-x/min-y 不再丢弃（rasterize 的
    -vx*s 补偿重新生效，带偏移 viewBox 不再整幅错位）。
  · #2099 svg._esc：属性值双引号转义 `&quot;`（label 含 `"` 产出 not
    well-formed SVG → ParseError 的红线，同 svg_backend #1091 已修先例）。
  · #2100 material.material_recipe：NPR shadow 0..1 fail-closed（修前 >1 语义
    反转比基色还亮、<0 静默钳 0；同函数 roughness/metallic 越界却抛）。
  · #2101 material.outline_recipe：自检 (False, reason) 不再被丢弃（style 越界
    抛 RecipeError）+ thickness_m 入口正数闸（负宽度反语义）。
  · #2105 seq.render_track：两层残留死代码（`list*0.0` 调试行 + `del Image`）
    清除——合法输入不再 100% 崩（TypeError / UnboundLocalError）。
  · #2113 audio._add：截断淡出改在副本上做——pan=0 时左右声道不再共享被原地
    改写的缓冲（修前尾部差 220 倍）。
  · #2117 bpy_vec.export_anim_document：keyframes >= 2 fail-closed（修前
    keyframes=1 裸除零，同函数对 morph 拓扑已有降级+计数口径）。
  · #2119 audio._norm_sections：「从 0 开始」改绝对值判等——负 start_s 不再
    静默通过（渲染期整段样点被负索引切片丢弃）。
  · #2120 shot._norm_camera：显式 `easing: []` 不再被 `or` 吞成默认——落入
    既有「须为 4 个控制点」报错；「不写」与「显式 None」仍得默认。

运行（仓根）：python -X utf8 -m pytest tests/test_issue2097_2120_bridge_family_guard.py
"""
import math
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


# ---------------------------------------------------------------- #2097

def test_2097_bezier_joint_is_baked_not_driven():
    from bridges.blender.wm_bridge.factory.anim import demo as _D
    from bridges.blender.wm_bridge.factory.anim import rig as _R
    spec = _D.mini_spec()
    spec["joints"][1] = dict(spec["joints"][1], easing="bezier")
    doc = _R.build_rig(spec)
    assert [d["link"] for d in doc["driven"]] == ["arm"], (
        "bezier 关节不得进 driven（修前错编线性驱动器），实测 %r"
        % ([d["link"] for d in doc["driven"]],))
    tip = [b for b in doc["baked"] if b["link"] == "tip"]
    assert tip and tip[0]["kind"] == "ease", (
        "bezier 关节应归 baked（kind='ease'），实测 %r" % (tip,))
    assert _R.validate_rig(doc) == (True, ""), "bezier 归 baked 后 rig 文档须过自检"


def test_2097_clean_spec_unchanged():
    from bridges.blender.wm_bridge.factory.anim import demo as _D
    from bridges.blender.wm_bridge.factory.anim import rig as _R
    doc = _R.build_rig(_D.mini_spec())
    assert [d["link"] for d in doc["driven"]] == ["arm", "tip"], (
        "无 easing 关节的 spec 行为必须不变（防过度收紧）")
    assert [b["link"] for b in doc["baked"]] == ["base"]


def test_2097_bake_bezier_traverses_spec_not_rdoc():
    from bridges.blender.wm_bridge.factory.anim import bpy_anim as _BA
    from bridges.blender.wm_bridge.factory.anim import demo as _D
    spec = _D.mini_spec()
    spec["joints"][1] = dict(spec["joints"][1], easing="bezier")
    assert _BA._bezier_joint_links(spec) == ["tip"]
    assert _BA._bezier_joint_links(_D.mini_spec()) == []


# ---------------------------------------------------------------- #2098

def test_2098_parse_svg_keeps_viewbox_offset():
    from bridges.blender.wm_bridge.factory.vec import raster as _RA
    svg = ('<svg width="100" height="100" viewBox="50 50 100 100">'
           '<rect x="50" y="50" width="50" height="50" fill="#ff0000"/></svg>')
    _shapes, meta, _rep = _RA.parse_svg(svg)
    assert tuple(meta["viewbox"]) == (50.0, 50.0, 100.0, 100.0), (
        "min-x/min-y 被丢（修前恒 (0,0,vw,vh)）：%r" % (meta["viewbox"],))


def test_2098_rasterize_offset_compensation_restored():
    import numpy as _np
    pytest.importorskip("numpy")
    from bridges.blender.wm_bridge.factory.vec import raster as _RA
    svg = ('<svg width="100" height="100" viewBox="50 50 100 100">'
           '<rect x="50" y="50" width="50" height="50" fill="#ff0000"/></svg>')
    img, _rep = _RA.rasterize(svg, width=100, height=100, ss=1)
    ys, xs = _np.nonzero(_np.asarray(img)[:, :, 0] > 100)
    assert xs.min() == 0 and xs.max() == 49 and ys.min() == 0 and ys.max() == 49, (
        "带偏移 viewBox 的红色矩形应落在左上象限 0..49（修前错位到 50..99），"
        "实测 x[%s,%s] y[%s,%s]" % (xs.min(), xs.max(), ys.min(), ys.max()))


# ---------------------------------------------------------------- #2099

def test_2099_attr_quotes_are_escaped():
    from bridges.blender.wm_bridge.factory.vec import svg as _S
    doc = _S.SVGDocument(100.0, 100.0)
    doc.layer("body", label='part "A" <1>')
    out = doc.to_svg()
    assert "&quot;" in out, "属性值双引号须转义 &quot;（修前裸 \" 破坏文档结构）"
    rep = doc.verify(out)
    assert rep["ok"] is True, "verify 必须可解析（修前 ParseError）"


# ---------------------------------------------------------------- #2100

def test_2100_shadow_range_fail_closed():
    from bridges.blender.wm_bridge.factory.shade import material as _M
    for bad in (1.5, -0.7):
        with pytest.raises(_M.RecipeError):
            _M.material_recipe("metal", style="npr", shadow=bad)


def test_2100_shadow_in_range_still_works():
    from bridges.blender.wm_bridge.factory.shade import material as _M
    rec = _M.material_recipe("metal", style="npr", shadow=0.3)
    assert rec["npr"]["shadow_floor"] == 0.3


# ---------------------------------------------------------------- #2101

def test_2101_outline_recipe_validates_style():
    from bridges.blender.wm_bridge.factory.shade import material as _M
    with pytest.raises(_M.RecipeError) as ei:
        _M.outline_recipe(style="pbr2")
    assert "style 不在闭集" in str(ei.value)


def test_2101_outline_thickness_positive():
    from bridges.blender.wm_bridge.factory.shade import material as _M
    with pytest.raises(_M.RecipeError):
        _M.outline_recipe(thickness_m=-1.0)
    rec = _M.outline_recipe()
    assert rec["outline"]["thickness_m"] > 0


# ---------------------------------------------------------------- #2105

def test_2105_render_track_works_on_legal_input():
    pytest.importorskip("PIL.Image")
    from bridges.blender.wm_bridge.factory.anim import seq as _SQ
    tracks = [(0.0, (0.0, 0.5, 0.0), (1.0, 0, 0)),
              (0.1, (0.1, 0.6, 0.0), (1.0, 0, 0)),
              (0.2, (0.2, 0.7, 0.0), (1.0, 0, 0))]
    out = _SQ.render_track(tracks)
    assert isinstance(out, list) and len(out) == 3


def test_2105_render_track_illegal_input_still_seqerror():
    from bridges.blender.wm_bridge.factory.anim import seq as _SQ
    with pytest.raises(_SQ.SeqError):
        _SQ.render_track([])


# ---------------------------------------------------------------- #2113

def test_2113_pan0_channels_identical():
    from bridges.blender.wm_bridge.factory.film.mv._det import audio as _A
    left, right = [0.0] * 600, [0.0] * 600
    _A._place([left, right], 0, [1.0] * 300, 0.0, limit=250)
    assert left == right, (
        "pan=0 时左右声道必须逐项相等（修前右声道被重复淡出，尾部差 220 倍）")


def test_2113_pan_positive_gains_still_linear():
    from bridges.blender.wm_bridge.factory.film.mv._det import audio as _A
    left, right = [0.0] * 600, [0.0] * 600
    _A._place([left, right], 0, [1.0] * 300, 0.18, limit=250)
    assert left[249] == right[249] * (1.0 - 0.18), (
        "线性声像：pan>0 时左 = 右×(1-pan)，实测 %r vs %r"
        % (left[249], right[249]))


# ---------------------------------------------------------------- #2117

def test_2117_keyframes_floor_fail_closed():
    from bridges.blender.wm_bridge.factory.vec import bpy_vec as _BV
    from bridges.blender.wm_bridge.factory.vec import geom2d as _G
    from bridges.blender.wm_bridge.factory.vec import prims as _P
    obj = _BV.DrawObject("cube", _P.box((1.0, 1.0, 1.0)))
    import tempfile
    with pytest.raises(ValueError):
        _BV.export_anim_document(
            [obj], lambda az: _G.Camera(eye=(3, 0, 1), look_at=(0, 0, 0)),
            os.path.join(tempfile.mkdtemp(prefix="r55_"), "a.svg"),
            keyframes=1)


def test_2117_keyframes_legal_still_works():
    from bridges.blender.wm_bridge.factory.vec import bpy_vec as _BV
    from bridges.blender.wm_bridge.factory.vec import geom2d as _G
    from bridges.blender.wm_bridge.factory.vec import prims as _P
    import tempfile
    obj = _BV.DrawObject("cube", _P.box((1.0, 1.0, 1.0)))
    d = tempfile.mkdtemp(prefix="r55_")
    path = os.path.join(d, "a.svg")
    _BV.export_anim_document(
        [obj], lambda az: _G.Camera(eye=(3, 0, 1), look_at=(0, 0, 0)),
        path, keyframes=3)
    assert os.path.isfile(path)


# ---------------------------------------------------------------- #2119

def test_2119_negative_start_s_rejected():
    from bridges.blender.wm_bridge.factory.film.mv._det import audio as _A
    with pytest.raises(_A.AudioError):
        _A._norm_sections([{"kind": "verse", "start_s": -0.5, "end_s": 0.5},
                           {"kind": "chorus", "start_s": 0.5, "end_s": 1.5}],
                          1.5)


def test_2119_zero_start_still_works():
    from bridges.blender.wm_bridge.factory.film.mv._det import audio as _A
    out = _A._norm_sections([{"kind": "verse", "start_s": 0.0, "end_s": 0.5},
                             {"kind": "chorus", "start_s": 0.5, "end_s": 1.5}],
                            1.5)
    assert out[0]["start_s"] == 0.0


# ---------------------------------------------------------------- #2120

def _base_shot():
    from bridges.blender.wm_bridge.factory.film import shot as _SH
    return {"schema": _SH.SHOT_SCHEMA, "id": "shot_01",
            "spec_ref": {"name": "ref_01", "digest": "a" * 64},
            "frame_base": {"fps": 24, "frame_start": 0, "frame_end": 23},
            "camera": {"path": [
                {"t": 0.0, "pos": [0.0, 1.6, -3.0], "aim": [0.0, 0.9, 0.0],
                 "focal_35eq": 45.0},
                {"t": 1.0, "pos": [1.5, 1.4, -1.4], "aim": [0.0, 0.9, 0.0],
                 "focal_35eq": 60.0}],
                "move_kind": "dolly_in"},
            "compose": {"aspect_ratio": 1.778, "subject_anchor": "center",
                        "headroom_ratio": 0.08, "lookroom_ratio": 0.18,
                        "horizon_v": 0.5, "depth_layers_min": 3,
                        "coverage_target": 0.68},
            "light": {"key_fill_ratio": 4.0, "mood": "low_key",
                      "color_temp_k": {"key": 3200, "fill": 5600},
                      "practicals": [],
                      "atmosphere": {"fog_density": 0.0, "volumetric_gain": 0.0}},
            "transition_out": {"kind": "cut", "quantize_beats": 1,
                               "tolerance_frames": 1}}


def test_2120_explicit_empty_easing_rejected():
    from bridges.blender.wm_bridge.factory.film import shot as _SH
    base = _base_shot()
    with pytest.raises(Exception) as ei:
        _SH.normalize_shot(dict(base, camera=dict(base["camera"], easing=[])))
    assert "4 个控制点" in str(ei.value), (
        "显式 easing: [] 须落入「须为 4 个控制点」报错（修前被 or 吞成默认），"
        "实得 %r" % (str(ei.value),))


def test_2120_default_semantics_unchanged():
    from bridges.blender.wm_bridge.factory.film import shot as _SH
    base = _base_shot()
    d_missing = _SH.normalize_shot(base)
    d_none = _SH.normalize_shot(dict(base, camera=dict(base["camera"],
                                                       easing=None)))
    assert list(d_missing["camera"]["easing"]) == [0.42, 0.0, 0.58, 1.0]
    assert list(d_none["camera"]["easing"]) == [0.42, 0.0, 0.58, 1.0]
    d_explicit = _SH.normalize_shot(dict(
        base, camera=dict(base["camera"], easing=[0.1, 0.2, 0.3, 0.4])))
    assert list(d_explicit["camera"]["easing"]) == [0.1, 0.2, 0.3, 0.4]
