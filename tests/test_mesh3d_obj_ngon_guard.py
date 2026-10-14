# -*- coding: utf-8 -*-
"""守卫 · mesh3d OBJ 载入：quad / n-gon **不再被静默丢弃**
============================================================================
缺陷（v1 `world3d.load_obj_mesh`）：
    elif ln.startswith("f "):
        idx = [...]
        if len(idx) == 3:          # ← 只收三角形
            tris.append((tuple(idx), cur))
  Blender 导出的 OBJ 里 quad 是**常态**。旧实现对这些面**一个不剩地丢掉，
  且不报错、不计数** —— 表现就是「模型缺面 / 只剩零散三角」，与用户
  「像 blender 一样多面数建模」的诉求直接冲突。

修法（`mesh3d.parse_obj`）：
  · `fan_triangulate(n)`：quad → 2 个三角形；n-gon → n-2 个三角形（扇形）；
  · 退化面（顶点重复 / 零面积 / 非有限）**丢弃但计数**（`stats`）——不静默；
  · 载入返回值带出 `quad_in` / `ngon_in` / `tri_in` / `dropped`，调用方看得见。

断言组（把 `fan_triangulate` 换回 `if len(idx) == 3` ⇒ A/B/C 全红）：
  A 只含 quad 的 OBJ：tris == 2×quad 数（旧实现 ⇒ 0 面 ⇒ ValueError）
  B 含 n-gon 的 OBJ：tris == Σ(n-2)，且 `ngon_in` 计数正确
  C 混合 quad+ngon+tri 的 OBJ：逐个面型的三角面数都对得上
  D 退化面被**计数**而不是静默消失
  E fail-closed：没有 `# normalize:` 头 ⇒ ValueError（不拿硬编码兜底）
  F 索引越界 / scale=0 ⇒ ValueError
  G 真实模型逐面型读数如实（不留空口）

判据来源：OBJ 格式规范（`f` 行是**任意顶点数的多边形**，三角形只是特例）＋
  本仓「不静默丢弃」纪律。「用扇形三角化」是工程选择（凹多边形会失真），
  已在 mesh3d 模块文档里如实标注。

运行（仓根 wm-verify/lingshu）：
    python3 -X utf8 -m pytest tests/test_mesh3d_obj_ngon_guard.py -q --no-header
"""
from __future__ import annotations

import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import pytest  # noqa: E402

from lingshu.world import mesh3d  # noqa: E402

HEADER = "# normalize: scale=1.0 center=(0.000000,0.000000,0.000000)\n"


def _write(tmp_path, body, name="m.obj"):
    p = tmp_path / name
    p.write_text(HEADER + body, encoding="utf-8")
    return str(p)


def _poly_body(sides, count, z0=1.0, radius=0.4):
    """`count` 个 `sides` 边形（**非退化**：正多边形放在 z=z0+i 的平面上）。

    注意：不能把 4 个顶点摆成一条直线——那会得到零面积面，被退化计数拦下，
    fixture 就测不到「三角化」这件事了（第一版 fixture 正是栽在这里）。
    """
    lines = []
    for i in range(count):
        z = z0 + i * 1.0
        base = len(lines) and 0 or 0
        # 顶点编号 = 已写出的 v 行数 + 1
        nv = sum(1 for ln in lines if ln.startswith("v "))
        for k in range(sides):
            a = 2 * math.pi * k / sides
            lines.append("v %.6f %.6f %.6f" % (radius * math.cos(a),
                                               radius * math.sin(a), z))
        lines.append("f " + " ".join(str(nv + k + 1) for k in range(sides)))
        del base
    return "\n".join(lines) + "\n"


def test_a_四边形三角化_tris等于2倍quad数(tmp_path):
    n = 7
    path = _write(tmp_path, _poly_body(4, n))
    with open(path, encoding="utf-8") as f:
        mesh = mesh3d.parse_obj(f, path=path)
    assert mesh.stats["quad_in"] == n, mesh.stats
    assert mesh.stats["tri_in"] == 0, mesh.stats
    assert len(mesh.faces) == 2 * n, (
        "A quad 未被三角化：tris=%d，期望 %d（旧实现这里是 0，随后 raise '没有三角面'）"
        % (len(mesh.faces), 2 * n))


def test_b_五边形按n减2三角化(tmp_path):
    n = 4
    path = _write(tmp_path, _poly_body(5, n))
    with open(path, encoding="utf-8") as f:
        mesh = mesh3d.parse_obj(f, path=path)
    assert mesh.stats["ngon_in"] == n, mesh.stats
    assert len(mesh.faces) == n * (5 - 2), (
        "B n-gon 三角化面数不对：%d != %d" % (len(mesh.faces), n * 3))


def test_c_混合面型逐类对得上(tmp_path):
    quad = _poly_body(4, 3)
    penta = _poly_body(6, 2, z0=50.0)
    nv = sum(1 for ln in (quad + penta).splitlines() if ln.startswith("v "))
    tail = ("v 0 0 0\nv 1 0 0\nv 0 1 0\n"
            "f %d %d %d\n" % (nv + 1, nv + 2, nv + 3))
    path = _write(tmp_path, quad + penta + tail)
    with open(path, encoding="utf-8") as f:
        mesh = mesh3d.parse_obj(f, path=path)
    st = mesh.stats
    expect = 3 * 2 + 2 * 4 + 1
    assert st["quad_in"] == 3 and st["ngon_in"] == 2 and st["tri_in"] == 1, st
    assert len(mesh.faces) == expect, (len(mesh.faces), expect, st)
    assert st["tris_out"] == expect, st


def test_d_退化面计数而不是静默消失(tmp_path):
    body = ("v 0 0 0\nv 1 0 0\nv 0 1 0\n"
            "f 1 1 2\n"            # 顶点重复 ⇒ 退化
            "f 1 2 3\n"            # 正常
            "f 1 2\n")             # 顶点不足 3 ⇒ 退化
    path = _write(tmp_path, body)
    with open(path, encoding="utf-8") as f:
        mesh = mesh3d.parse_obj(f, path=path)
    assert len(mesh.faces) == 1, len(mesh.faces)
    assert mesh.stats["degenerate_dropped"] == 2, mesh.stats


def test_e_无_normalize头则_fail_closed(tmp_path):
    p = tmp_path / "bad.obj"
    p.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    with pytest.raises(ValueError) as e:
        with open(str(p), encoding="utf-8") as f:
            mesh3d.parse_obj(f, path=str(p))
    assert "normalize" in str(e.value), str(e.value)


def test_f_索引越界与scale零都拒收(tmp_path):
    """反例必须锚在**正确的拒绝原因**上（第一版 fixture 自己算错了顶点号）。"""
    p1 = _write(tmp_path, "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 9\n", "a.obj")
    with pytest.raises(ValueError) as e1:
        with open(p1, encoding="utf-8") as f:
            mesh3d.parse_obj(f, path=p1)
    assert "索引越界" in str(e1.value), str(e1.value)
    p2 = tmp_path / "b.obj"
    p2.write_text("# normalize: scale=0.0 center=(0,0,0)\n"
                  "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    with pytest.raises(ValueError) as e2:
        with open(str(p2), encoding="utf-8") as f:
            mesh3d.parse_obj(f, path=str(p2))
    assert "scale=0" in str(e2.value), str(e2.value)


def test_g_真实模型逐面型读数与帧一致(tmp_path):
    """仓库外的真实 OBJ 都是全三角面；读数必须如实（不能凭空多出 quad）。"""
    arm = "/mnt/d/DSH/blender/out/wmtest/arm/mesh.obj"
    if not os.path.isfile(arm):
        pytest.skip("真实模型不在本机：%s" % arm)
    info = mesh3d.load_obj_mesh(arm, "_guard_arm")
    assert info["quad_in"] == 0 and info["ngon_in"] == 0, info
    assert info["tris"] == info["tri_in"] == 344, info
    assert info["dropped"] == 0, info


def test_h_重定心_不丢原始偏移(tmp_path):
    """**实测缺陷**：源 OBJ 带着它在原场景里的世界偏移（本仓 `wmtest/arm` x∈[5.17,6.40]）。
    不重定心 ⇒ 调用方按直觉传 `center=(0,0,0)` 会得到一张空图且不报错。

    修法：载入期把顶点平移到包围盒中心为局部原点，**偏移记进 `recenter_offset`**。
    """
    body = "v 5.0 0 0\nv 6.0 0 0\nv 5.0 1.0 0\nv 6.0 1.0 0\nf 1 2 4 3\n"
    path = _write(tmp_path, body)
    with open(path, encoding="utf-8") as f:
        mesh = mesh3d.parse_obj(f, path=path)
    lo, hi = mesh.bounds()
    c = [(lo[k] + hi[k]) / 2 for k in range(3)]
    assert all(abs(v) < 1e-9 for v in c), ("重定心失败，包围盒中心 %r" % (c,))
    assert abs(mesh.recenter_offset[0] - 5.5) < 1e-9, mesh.recenter_offset
    with open(path, encoding="utf-8") as f:
        raw = mesh3d.parse_obj(f, path=path, recenter=False)
    assert abs(raw.bounds()[0][0] - 5.0) < 1e-9, raw.bounds()
    assert raw.recenter_offset == (0.0, 0.0, 0.0), raw.recenter_offset
