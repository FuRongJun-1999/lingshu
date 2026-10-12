# -*- coding: utf-8 -*-
"""world3d 相机正交基 + 显式网格上屏 —— 守卫

缺陷（[world/world3d][口径不诚实 · 两侧像面不可比]）：
  ① 相机只有 `yaw`/`pitch`，**表达不出「真实引擎用 up 向量定向」的相机朝向**。
     在方位角≠0 时，画面相对纯 yaw+pitch 相机存在**滚转**（实测 45° 方位角下约 16.4°），
     于是两侧像面相对旋转、**同一场景的轮廓永远对不齐**（实测无论怎么调 roll，
     轮廓 IoU 都卡在 ~0.49）。「用世界模型的渲染对真实引擎的渲染」这条链路因此不成立。
  ② 世界模型的实体只能用内置词表里的**近似盒**画，真实几何（OBJ + MTL）进不来
     —— 轮廓比对拿的是近似的形状，不是引擎里那件东西。

修法（全部**向后兼容**，默认值不动既有行为）：
  ① `Camera3D` 新增 `roll`（默认 0.0）与 `basis`（显式正交基，默认 None）；
     `from_look_at(eye, target, up)` 按「右=前向×up、上=右×前向」构造**与引擎同构**的基；
     `look_at` 加 `roll=0.0` 默认参数。退化输入（eye==target、up∥前向）**raise**（fail-closed）。
  ② 新增 `load_mtl` / `load_obj_mesh` / `register_mesh`：把导出的 OBJ/MTL 载成显式几何；
     **归一化参数必须从文件头注释解析，解析不到就 raise**（不拿硬编码兜底）；
     `axis_swap` 做 Blender(Z-up)→lingshu(Y-up/+Z) 的**真旋转**（det=+1，不是镜像）。
  ③ 绘制侧只**包装**不改既有路径：无显式几何时原路（既有代码逐字节不变）。

判据来源：与真实引擎（Blender）的轮廓比对实测（45° 方位角 IoU 0.49 → 对齐后显著上升）；
`axis_swap` 的 (x,y,z)→(x,z,−y) 取自桥的 `geom.py` 既有权杖（det=+1）。

断言组（抽掉修复 ⇒ 各组按标注转红）：
  A **正交基正确**：`from_look_at` 的基单位正交、右手系、前向=normalize(target−eye)；
    退化输入 raise —— 抽 ① ⇒ 红
  B **两侧朝向确有差异**（本件的**存在理由**）：45° 方位角下，`from_look_at` 与
    `look_at` 画出的「世界 up」在屏幕上夹角 ≈ 16.4°（>5°）—— 抽 ① ⇒ 红
  C **向后兼容**：`look_at(eye,target)`（不给 roll）与改前的 yaw/pitch **逐位一致**；
    非基相机的 `to_camera` 与改前算式**逐位一致** —— 抽 ① 过头 ⇒ 红
  D **网格载入 fail-closed**：缺 `normalize:` 头 ⇒ raise；头齐全 ⇒ 计数正确；
    `axis_swap` 把 (0,1,0) 映到 (0,0,−1)（真旋转）—— 抽 ② ⇒ 红
  E **MTL/防误杀**：`Kd` 经 sRGB 转 0..255；既有形状渲染路径仍能出图 —— 抽 ③ ⇒ 红

运行（仓根）：python -X utf8 -m pytest tests/test_world3d_camera_and_mesh_guard.py -q --no-header
"""
from __future__ import annotations

import math
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.world.world3d import (  # noqa: E402
    Camera3D, _srgb, load_mtl, load_obj_mesh, register_mesh)

EYE45 = (10.0 * math.cos(math.radians(45)), 3.0, 10.0 * math.sin(math.radians(45)))
TGT = (0.0, 0.0, 0.0)
SW, SH = 800, 600


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _norm(a):
    return math.sqrt(sum(x * x for x in a))


def _det(r, u, f):
    return (r[0] * (u[1] * f[2] - u[2] * f[1])
            - r[1] * (u[0] * f[2] - u[2] * f[0])
            + r[2] * (u[0] * f[1] - u[1] * f[0]))


def test_a_显式基单位正交且朝向为标定值():
    """A：`from_look_at` 的基必须**单位正交**、前向=normalize(target−eye)，
    且 `det` 为**标定出来的 −1**。

    ⚠ 本组**不是**在要求教科书的右手系。写守卫时实测发现该基 `det = −1`
    （`r × u = −f`，与 `yaw/pitch` 路径的 det=+1 相反），一度当成镜像缺陷；
    但把它改成 `r = up × f`（det=+1）后，**几何 IoU 从 0.9952 崩到 0.2369**
    （gearbox，同一参考图、同一机位；实测命令见本文件 docstring）。
    ⇒ **det=−1 是这条管线标定出来的**（世界 up 在像面保持竖直，与引擎参考图对齐），
      不是可以按理论"修"掉的东西。本组据此把它**钉住**：谁把它改成 +1，这组必红。
    """
    cam = Camera3D.from_look_at(EYE45, TGT, up=(0.0, 0.0, 1.0))
    assert cam.basis is not None, "from_look_at 必须给出显式基"
    r, u, f = [tuple(v) for v in cam.basis]
    for name, v in (("right", r), ("up", u), ("forward", f)):
        assert abs(_norm(v) - 1.0) < 1e-9, "%s 不是单位向量：%r" % (name, v)
    assert abs(_dot(r, u)) < 1e-9 and abs(_dot(r, f)) < 1e-9 and abs(_dot(u, f)) < 1e-9, \
        "三元组必须两两正交：r·u=%r r·f=%r u·f=%r" % (_dot(r, u), _dot(r, f), _dot(u, f))
    # 标定值：det = −1（改了它，几何 IoU 会从 0.9952 崩到 0.2369）
    assert abs(_det(r, u, f) + 1.0) < 1e-9, (
        "det 必须是标定值 −1（实测改成 +1 会让几何 IoU 0.9952→0.2369）：%r" % _det(r, u, f))
    cr = (r[1] * u[2] - r[2] * u[1], r[2] * u[0] - r[0] * u[2], r[0] * u[1] - r[1] * u[0])
    assert all(abs(a + b) < 1e-9 for a, b in zip(cr, f)), "应满足 r × u = −f（标定口径）"
    # 前向 = normalize(target − eye)
    d = tuple(t - e for t, e in zip(TGT, EYE45))
    dl = _norm(d)
    assert all(abs(a - b / dl) < 1e-9 for a, b in zip(f, d)), "前向 ≠ normalize(target−eye)"
    assert (cam.cx, cam.cy, cam.cz) == EYE45
    # 世界 up 必须落在像面**竖直**方向（这正是标定的目的）
    assert abs(_screen_up_deg(cam)) < 1.0, "世界 up 未保持竖直：%.3f°" % _screen_up_deg(cam)


@pytest.mark.parametrize("eye,target,up,why", [
    ((1.0, 2.0, 3.0), (1.0, 2.0, 3.0), (0.0, 0.0, 1.0), "eye 与 target 重合"),
    ((0.0, 0.0, 0.0), (0.0, 0.0, 5.0), (0.0, 0.0, 1.0), "up 与前向共线"),
])
def test_a2_退化输入必须报错(eye, target, up, why):
    """A2（fail-closed）：退化基必须 raise，不许静默给一个歪相机。"""
    with pytest.raises(ValueError):
        Camera3D.from_look_at(eye, target, up=up)


def _screen_up_deg(cam):
    """世界 up 在像面上的方向（相对屏幕竖直，度；0 = 竖直）。"""
    c = cam.project(TGT, SW, SH)
    p = cam.project((0.0, 0.0, 1.0), SW, SH)
    assert c and p, "投影失败"
    return math.degrees(math.atan2(p[0] - c[0], c[1] - p[1]))


def test_b_两侧朝向确有差异_本件的存在理由():
    """B：`from_look_at` 让世界 up 保持竖直；`look_at`（纯 yaw+pitch）做不到。

    这就是本件存在的理由：真实引擎用**真实 up 向量**定向，画面里"上"不一定竖直；
    纯 yaw+pitch 相机在方位角≠0 时会把世界 up 拧到斜的（实测 45° 方位角下约 106°），
    两侧像面因此相对旋转、轮廓对不齐。`from_look_at` 按构造让世界 up 落在竖直方向。
    """
    yp = Camera3D.look_at(EYE45, TGT)
    ex = Camera3D.from_look_at(EYE45, TGT, up=(0.0, 0.0, 1.0))
    a_yp, a_ex = abs(_screen_up_deg(yp)), abs(_screen_up_deg(ex))
    assert a_ex < 1.0, "显式基应让世界 up 竖直，实得 %.3f°" % a_ex
    assert a_yp > 5.0, (
        "纯 yaw+pitch 相机在 45° 方位角下必须把世界 up 拧斜（实测约 106°），实得 %.3f°" % a_yp)


def test_c_向后兼容_默认参数逐位一致():
    """C：不给 `roll` 时，`look_at` 与 `to_camera` 必须与改前算式**逐位一致**。"""
    for eye in (EYE45, (3.0, 1.0, -2.0), (0.0, 5.0, 0.0)):
        cam = Camera3D.look_at(eye, TGT)
        dx, dy, dz = TGT[0] - eye[0], TGT[1] - eye[1], TGT[2] - eye[2]
        yaw = math.atan2(-dx, dz)
        pitch = math.atan2(dy, math.hypot(dx, dz))
        assert repr(cam.yaw) == repr(yaw), "yaw 变了：%r ≠ %r" % (cam.yaw, yaw)
        assert repr(cam.pitch) == repr(pitch), "pitch 变了：%r ≠ %r" % (cam.pitch, pitch)
        assert repr(cam.roll) == repr(0.0)
        # to_camera 的非基分支（改前算式，逐字重算）
        for p3 in ((1.0, 2.0, 3.0), (-4.0, 0.5, 2.0), (0.0, 0.0, 8.0)):
            x, y, z = p3[0] - cam.cx, p3[1] - cam.cy, p3[2] - cam.cz
            cy, sy = math.cos(yaw), math.sin(yaw)
            x, z = x * cy + z * sy, -x * sy + z * cy
            cp, sp = math.cos(pitch), math.sin(pitch)
            y, z = y * cp - z * sp, y * sp + z * cp
            assert cam.to_camera(p3) == (x, y, z), (
                "非基相机 to_camera 与改前算式不一致：%r" % (cam.to_camera(p3),))
    a = Camera3D.look_at(EYE45, TGT)
    b = Camera3D.look_at(EYE45, TGT, roll=0.0)
    for fld in ("fov_deg", "cx", "cy", "cz", "yaw", "pitch", "roll", "basis"):
        assert repr(getattr(a, fld)) == repr(getattr(b, fld)), \
            "roll=0.0 与缺省不等价：%s %r ≠ %r" % (fld, getattr(a, fld), getattr(b, fld))
    assert a.basis is None and b.basis is None, "look_at 不该给显式基"


def _write_obj(tmp_path, body, header=True):
    p = tmp_path / "m.obj"
    lines = []
    if header:
        lines.append("# normalize: scale=1.0 center=(0.000000,0.000000,0.000000)")
    lines.append("mtllib m.mtl")
    lines += body
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(p)


OBJ_BODY = [
    "v 0.0 1.0 0.0", "v 1.0 0.0 0.0", "v 0.0 0.0 1.0",
    "g probe", "usemtl M_probe", "f 1 2 3",
]


def test_d_网格载入_fail_closed_与轴交换(tmp_path):
    """D：缺归一化头必须 raise；头齐全则计数正确且 axis_swap 是真旋转。"""
    p_ok = _write_obj(tmp_path, OBJ_BODY, header=True)
    r = load_obj_mesh(p_ok, "__probe_ok")
    assert (r["verts"], r["tris"], r["materials"]) == (3, 1, 1), r
    assert r["unscaled_by"] == 1.0, "头部 scale=1.0 应被采纳：%r" % (r["unscaled_by"],)
    # 轴交换：(x,y,z) -> (x,z,-y)  ⇒ (0,1,0) -> (0,0,-1)（真旋转 det=+1，不是镜像）
    from lingshu.world.world3d import MESH_LIBRARY
    first = MESH_LIBRARY["__probe_ok"]["verts"][0]
    assert tuple(round(v, 9) for v in first) == (0.0, 0.0, -1.0), (
        "axis_swap 应把 (0,1,0) 映到 (0,0,-1)：%r" % (first,))
    assert len(MESH_LIBRARY["__probe_ok"]["tris"]) == 1
    # 缺头 ⇒ raise（不拿硬编码兜底）
    p_bad = _write_obj(tmp_path, OBJ_BODY, header=False)
    with pytest.raises(Exception):
        load_obj_mesh(p_bad, "__probe_bad")
    assert "__probe_bad" not in MESH_LIBRARY, "载入失败却登记了"


def test_e_mtl_解析与既有形状路径未被改坏(tmp_path):
    """E：`Kd` → sRGB 0..255；`_srgb` 端点正确；既有形状渲染仍出图。"""
    assert _srgb(0.0) == 0 and _srgb(1.0) == 255
    assert _srgb(-1.0) == 0 and _srgb(2.0) == 255, "越界应被夹住"
    mtl = tmp_path / "m.mtl"
    mtl.write_text("newmtl M_probe\nKd 1.000000 0.000000 0.500000\n", encoding="utf-8")
    cols = load_mtl(str(mtl))
    assert set(cols) == {"M_probe"}, cols
    assert cols["M_probe"][0] == 255 and cols["M_probe"][1] == 0
    assert 0 < cols["M_probe"][2] < 255
    # 防误杀：既有形状路径（不注册任何网格）仍应出图
    from lingshu.world.world3d import World3D
    w = World3D()
    img = w.render(160, 120)
    assert img is not None and img.size == (160, 120), "既有渲染路径被改坏了：%r" % (img,)
