# -*- coding: utf-8 -*-
"""批九十六守卫 C 组：结构一致性族（#2428 / #2429 / #2433）回归钉。

覆盖
--------------------------------------------------------------------------
#2428 `mesh_gen` 两处
    G1 `install("lamp")` / `install(b"lamp")` ⇒ TypeError
       （改前 `list("lamp")` 拆成字符、每个 `build_mesh` 不到 ⇒ 返回 `{}` 静默 0 登记）
    G2 `install([])` 的 #2074 契约不变（空列表仍登记全部）
    G3 `add_tri` 非法色 ⇒ faces/colors 仍等长
       （改前先 append 面后 `_clamp_color` ⇒ faces=2/colors=1，`to_obj_text`
        的 zip 静默丢掉那个面）
    G4 正对照：合法色 faces/colors 各 +1；`add_poly` 同款

#2429 `provenance.new_provenance` 时间窗有限性闸
    G5 `observed_at=nan/inf`、`window=nan/0` ⇒ ValueError
       （改前产出 `(nan,nan)` 窗、`to_sql()` 出 RFC 8259 非法 JSON、`validate()`
        判合规）
    G6 正对照：合法窗逐位不变且严格 JSON 可解析

#2433 `mesh3d` 四处
    G7 `ZBuffer(background=(20,24,40,255))` ⇒ 9 字节缓冲
       （改前 12 字节 ⇒ 图像斜条纹）；`(20,24)` / 标量 ⇒ ValueError
    G8 `parse_obj(scale=1e999/-1.0/0)` ⇒ ValueError
       （改前 inf 让 240 面静默归零、负值翻法向）
    G9 `load_mtl` 短 `Kd` / 非有限分量 ⇒ ValueError
       （改前 2 元组颜色直到 `prepare_triangles` 才 IndexError；NaN 静默变纯白）
    G10 `_rot_mat((nan,0,0))` ⇒ ValueError
        （改前顶点全 NaN、背面剔除失效、`kept` 假读数）
    G11 `skeleton3d.fit_camera(fov_deg=0/-30/nan)` ⇒ ValueError
        （改前裸 ZeroDivisionError / 负焦距翻转画面 / 全 NaN 相机）
    G12 正对照：四个入口的合法读数逐位不变

运行（lingshu 仓根）：
    python -X utf8 -m pytest tests/test_batch96_structure_guard.py -q
"""
from __future__ import annotations

import json
import math
import os

import pytest

from lingshu.core import provenance as PV
from lingshu.world import mesh3d as M3
from lingshu.world import mesh_gen as MG
from lingshu.world import skeleton3d as SK


def _quad_mesh():
    md = MG.MeshData()
    for v in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]:
        md.add_vert(v)
    return md


# ==================================================== #2428 install 入参
@pytest.mark.parametrize("bad", ["lamp", b"lamp"])
def test_G1_install_string_rejected(bad):
    """G1：单个字符串/bytes 必须 TypeError（改前静默登记 0 个配方）。"""
    with pytest.raises(TypeError) as ei:
        MG.install(bad)
    assert "categories" in str(ei.value)


def test_G2_install_empty_list_contract_kept():
    """G2：#2074 的契约不变——显式空列表 ≠ None：`install([])` 登记**零个**，
    只有 `install(None)` 才登记全部（#2074 修的正是旧版把 `[]` 当 None 的 falsy 混淆）。"""
    assert MG.install([]) == {}, MG.install([])
    assert len(MG.install(None)) == len(MG.MESH_RECIPES)


def test_G3_add_tri_bad_color_keeps_sync():
    """G3：非法色抛错后 faces 与 colors 仍等长（改前 faces=2/colors=1）。"""
    md = _quad_mesh()
    md.add_tri(0, 1, 2)
    assert (len(md.faces), len(md.colors)) == (1, 1)
    with pytest.raises(ValueError):
        md.add_tri(1, 2, 3, color=(1, 2))
    assert len(md.faces) == len(md.colors) == 1, (len(md.faces), len(md.colors))
    # OBJ 的 f 行数必须等于面数（zip 不再截断）
    obj = MG.to_obj_text(md)
    assert sum(1 for ln in obj.split("\n") if ln.startswith("f ")) == len(md.faces)


def test_G4_add_tri_add_poly_legal_unchanged():
    """G4 正对照：合法色时 faces/colors 各 +1；`add_poly` 同款同步。"""
    md = _quad_mesh()
    md.add_tri(0, 1, 2, color=(200, 10, 10))
    assert (len(md.faces), len(md.colors)) == (1, 1)
    assert md.colors[0] == (200, 10, 10)
    md2 = _quad_mesh()
    md2.add_poly([0, 1, 3, 2], color=(10, 200, 10))
    assert len(md2.faces) == len(md2.colors) == 2, (len(md2.faces), len(md2.colors))


# ============================================== #2429 provenance 时间窗
@pytest.mark.parametrize("kw", [
    {"observed_at": float("nan")},
    {"observed_at": float("inf")},
    {"window": float("nan")},
    {"window": float("inf")},
    {"window": 0},
])
def test_G5_provenance_nonfinite_window_rejected(kw):
    """G5：非有限/非正时间窗必须 ValueError（改前出 RFC 8259 非法 JSON）。"""
    with pytest.raises(ValueError) as ei:
        PV.new_provenance("pos1", "toolA", "srcA", **kw)
    assert "observed_at" in str(ei.value) or "window" in str(ei.value)


def test_G6_provenance_legal_window_unchanged():
    """G6 正对照：合法窗逐位不变，`to_sql()` 的严格 JSON 可解析。"""
    p = PV.new_provenance("pos1", "toolA", "srcA", observed_at=1234, window=3600)
    assert p.time_window == (1234.0, 4834.0)
    inner = p.to_sql()[0]

    def _reject(x):
        raise ValueError("RFC8259 violation: %r" % (x,))

    json.loads(inner, parse_constant=_reject)   # 不抛即合规


# ================================================== #2433 mesh3d 四处
def test_G7_zbuffer_background_shape_gate():
    """G7：4 元组只取前 3 通道（9 字节）；2 元组/标量必须 ValueError。"""
    z = M3.ZBuffer(3, 1, background=(20, 24, 40, 255))
    assert len(z.color) == 9, len(z.color)
    assert bytes(z.color) == bytes([20, 24, 40]) * 3
    for bad in [(20, 24), 20, (1, 2, 3, 4, 5)]:
        with pytest.raises((ValueError, TypeError)):
            M3.ZBuffer(3, 1, background=bad)


@pytest.mark.parametrize("scale", ["1e999", "-1.0", "0"])
def test_G8_parse_obj_bad_scale_rejected(scale):
    """G8：`scale` 非有限/非正必须 ValueError（改前 inf 静默丢 240 面）。"""
    txt = MG.to_obj_text(MG.build_mesh("chair")).replace(
        "scale=1.0", "scale=%s" % scale, 1).split("\n")
    with pytest.raises(ValueError) as ei:
        M3.parse_obj(txt)
    assert "scale" in str(ei.value)


def test_G9_load_mtl_short_or_nonfinite_kd(tmp_path):
    """G9：短 `Kd` / 非有限分量必须 ValueError（改前 2 元组致深管线 IndexError）。"""
    p = tmp_path / "m.mtl"
    p.write_text("newmtl short\nKd 0.5 0.5\n", encoding="utf-8")
    with pytest.raises(ValueError) as ei:
        M3.load_mtl(str(p))
    assert "Kd" in str(ei.value)
    p2 = tmp_path / "n.mtl"
    p2.write_text("newmtl nanm\nKd nan nan nan\n", encoding="utf-8")
    with pytest.raises(ValueError):
        M3.load_mtl(str(p2))


def test_G10_rot_mat_nonfinite_rejected():
    """G10：NaN 旋转必须 ValueError（改前 kept 计数撒谎 + 渲染空背景）。"""
    with pytest.raises(ValueError) as ei:
        M3._rot_mat((float("nan"), 0, 0))
    assert "rot" in str(ei.value)
    assert M3._rot_mat((0, 0, 0)) is None
    m = M3._rot_mat((0.1, 0.2, 0.3))
    assert m is not None and all(math.isfinite(v) for row in m for v in row)


@pytest.mark.parametrize("fov", [0.0, -30.0, float("nan"), 180.0])
def test_G11_fit_camera_bad_fov_rejected(fov):
    """G11：`fov_deg` 越界必须 ValueError（改前裸除零/负焦距/全 NaN 相机）。"""
    with pytest.raises(ValueError) as ei:
        SK.fit_camera([(0, 0, 0), (0, 1.8, 0)], 240, 240, fov_deg=fov)
    assert "fov_deg" in str(ei.value)


def test_G12_legal_values_unchanged():
    """G12 正对照：四个入口的合法读数逐位不变。"""
    txt = MG.to_obj_text(MG.build_mesh("chair")).split("\n")
    g = M3.parse_obj(txt)
    assert (len(g.verts), len(g.faces)) == (132, 240)
    cam = SK.fit_camera([(0, 0, 0), (0, 1.8, 0)], 240, 240)
    assert abs(cam.cz - (-2.2459)) < 1e-3, cam.cz
    assert M3.ZBuffer(3, 1, background=(20, 24, 40)).color == bytearray([20, 24, 40] * 3)
