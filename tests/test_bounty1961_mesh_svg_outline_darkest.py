# -*- coding: utf-8 -*-
"""test_bounty1961_mesh_svg_outline_darkest · #1961 守卫
=======================================================
缺陷（修复前实测，锚定 45b8359）：

`lingshu/world/mesh_svg.py` 的 `STYLE["outline_color"]` 契约（`mesh_svg.py:89`）是

    "outline_color": None,      # None ⇒ 按该物体最暗材质色再压暗 0.42

但 `_emit_mesh` 的缺省分支取的是 `mesh.faces[0].color`（首个面的颜色），与「最暗
材质色」无关。两个后果：

  ① 取错色：`faces[0]` 亮、后续面暗时线稿明显偏亮
     （实测 `rgb(96,96,96)` vs 契约 `rgb(16,25,37)`）；
  ② 破坏确定性：同一物体、材质集合完全相同，只把 `faces` 顺序颠倒，线稿色就从
     `rgb(96,96,96)` 变成 `rgb(16,25,37)` —— 而 `_emit_mesh` docstring 与
     `svg_backend` 的设计要点都承诺「同样场景两次导出逐字节相同」。

修法（最小）：缺省分支改为在 `mesh.faces` 的**去重材质色集合**里按 Rec.709 亮度取
最小者（空网格仍回落 `(40,44,52)`），再 `_darken(..., 0.42)`。

断言组（抽掉修复 ⇒ 哪组红）：
  G1 缺省线稿色 == `_darken(最暗材质色, 0.42)`（改回 `faces[0]` ⇒ G1 红）
  G2 面序无关：材质集合相同、`faces` 顺序颠倒 ⇒ 线稿 stroke 集合相同
  G3 显式 `outline_color` 仍原样生效（缺省分支不许吞掉显式值）
  G4 空网格回落 `(40,44,52)` 后压暗（不许 IndexError）
  G5 单材质网格：缺省色 == `_darken(该材质色, 0.42)`（不许因去重逻辑漂移）
  G6 同亮度不同色：线稿色与面序无关（同亮度按分量序定序）

运行（lingshu 仓根）：
  python -X utf8 -m pytest tests/test_bounty1961_mesh_svg_outline_darkest.py -q
"""
from __future__ import annotations

import os
import re
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingshu.world import mesh3d, mesh_svg                    # noqa: E402
from lingshu.world.svg_backend import SVGBackend              # noqa: E402
from lingshu.world.world3d import Camera3D                    # noqa: E402

BRIGHT = (230, 230, 230)      # 亮面
DARK = (40, 60, 90)           # 最暗材质色
MID = (150, 120, 60)

#: 契约口径的期望值（`_darken(最暗材质色, 0.42)`）。
WANT_DEFAULT = mesh_svg._darken(DARK, 0.42)
#: 误取 `faces[0]` 会得到这个 —— 修复前实测读数。
WRONG_FIRST_FACE = mesh_svg._darken(BRIGHT, 0.42)


def _mesh(colors, reverse=False):
    """两面网格（`colors[0]` 在前），`reverse` 只颠倒面序、不改材质集合。"""
    verts = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0)]
    faces = [mesh3d.Face(idx=(0, 1, 2), normal=(0.0, 0.0, -1.0), color=colors[0]),
             mesh3d.Face(idx=(1, 3, 2), normal=(0.0, 0.0, -1.0), color=colors[1])]
    if reverse:
        faces = faces[::-1]
    return mesh3d.MeshGeometry(verts=verts, faces=faces, name="guard1961")


def _emit(colors, style=None, reverse=False, faces=None):
    """走 `_emit_mesh`（`render_mesh_svg` 的网格发射函数，同一条代码路径）。"""
    mesh = _mesh(colors, reverse=reverse) if faces is None else faces
    cam = Camera3D(cx=0.5, cy=0.5, cz=-4.0, fov_deg=50.0)
    world = SimpleNamespace(mesh_shade="lambert", mesh_cull=False,
                            mesh_smooth=False, env_reflect=False)
    obj = SimpleNamespace(rot=None, center=(0.0, 0.0, 0.0), category="guard1961")
    backend = SVGBackend(200, 200, background=(0, 0, 0))
    counts = {"objects": 0, "mesh_objects": 0, "faces": 0, "tris": 0,
              "contours": 0, "creases": 0, "skipped_edges": 0, "subgroups": 0}
    st = dict(mesh_svg.STYLE)
    st.update(style or {})
    mesh_svg._emit_mesh(backend, world, obj, mesh, cam, 200, 200, st, 1.0,
                        counts, layers=False)
    return backend.to_svg(), counts


def _strokes(svg):
    return sorted(set(re.findall(r'stroke="([^"]*)"', svg)))


def _rgb(c):
    return "rgb(%d,%d,%d)" % (int(c[0]), int(c[1]), int(c[2]))


# ---------------------------------------------- G1 缺省取最暗材质色

def test_g1_default_outline_uses_darkest_material():
    svg, counts = _emit([BRIGHT, DARK])
    assert counts["contours"] > 0, "线稿未发射，守卫无从判定"
    assert _strokes(svg) == [_rgb(WANT_DEFAULT)], (
        "缺省 outline_color 应取最暗材质色 %r 压暗 0.42 ⇒ %s；"
        "实测 %s（误取 faces[0] 会得到 %s）"
        % (DARK, _rgb(WANT_DEFAULT), _strokes(svg), _rgb(WRONG_FIRST_FACE)))


# ---------------------------------------------- G2 面序无关

def test_g2_outline_color_independent_of_face_order():
    a, _ = _emit([BRIGHT, DARK])
    b, _ = _emit([BRIGHT, DARK], reverse=True)
    assert _strokes(a) == _strokes(b) == [_rgb(WANT_DEFAULT)], (
        "材质集合相同、仅面序颠倒，线稿色不应变化：顺 %s / 逆 %s"
        % (_strokes(a), _strokes(b)))


# ---------------------------------------------- G3 显式值不被吞掉

def test_g3_explicit_outline_color_still_respected():
    explicit = (10, 200, 30)
    svg, _ = _emit([BRIGHT, DARK], style={"outline_color": explicit})
    assert _strokes(svg) == [_rgb(explicit)]


def test_g3b_preset_technical_keeps_explicit_color():
    # `technical` 预设显式给了 outline_color，不受本修复影响
    preset = dict(mesh_svg.STYLE_PRESETS["technical"])
    assert preset.get("outline_color") is not None
    svg, _ = _emit([BRIGHT, DARK], style=preset)
    assert _rgb(preset["outline_color"]) in _strokes(svg)


# ---------------------------------------------- G4 空网格回落

def test_g4_empty_mesh_falls_back_to_default():
    empty = mesh3d.MeshGeometry(verts=[], faces=[], name="guard1961_empty")
    assert mesh_svg._darkest_face_color([]) == (40, 44, 52)
    assert mesh_svg._darkest_face_color(empty.faces) == (40, 44, 52)
    # 空网格不发线稿，但缺省分支不得抛 IndexError
    svg, counts = _emit([BRIGHT, DARK], faces=empty)
    assert counts["contours"] == 0 and _strokes(svg) == []


# ---------------------------------------------- G5 单材质网格不漂移

def test_g5_single_material_unchanged():
    only = (118, 121, 126)
    svg, _ = _emit([only, only])
    assert _strokes(svg) == [_rgb(mesh_svg._darken(only, 0.42))]


# ---------------------------------------------- G6 同亮度不同色

def test_g6_equal_luminance_tie_is_order_independent():
    # (100, 100, 100) 与 (60, 160, 60)：亮度接近，取色仍须与面序无关
    x, y = (100, 100, 100), (60, 160, 60)
    a, _ = _emit([x, y])
    b, _ = _emit([x, y], reverse=True)
    assert _strokes(a) == _strokes(b)
    expect = mesh_svg._darken(mesh_svg._darkest_face_color(
        [SimpleNamespace(color=x), SimpleNamespace(color=y)]), 0.42)
    assert _strokes(a) == [_rgb(expect)]
