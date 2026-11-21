# -*- coding: utf-8 -*-
"""守卫：wm_bridge.factory.vec.raster SVG `<path>` 契约与子路径闭合语义。

根因（修前）：
  `raster._geometry("path", elem)` 直接返回 `_path_subpaths(d)` 的子路径列表本身，
  而不是 `(subpaths, closed)` 二元组：
    · 0 子路径（空/非法 `d`）→ `ValueError: not enough values to unpack (expected 2, got 0)`
    · 1 子路径 → `ValueError: not enough values to unpack (expected 2, got 1)`
    · 2 子路径 → 第 2 条子路径被误解包为 `closed`，后续 `_apply`/`rasterize` 抛 `AxisError`
    · ≥3 子路径 → `ValueError: too many values to unpack (expected 2)`
  同时旧版 `_path_subpaths` 的词法正则漏掉了 `C/S/Q/T/A` 等曲线指令字母，
  导致不支持指令被静默吞掉、控制点被误当折线顶点；且旧版 `_Ctx.draw` 逐子路径
  独立填充，无法按 even-odd 规则在复合路径中挖洞。

修复语义：
  1. `_path_subpaths(d)` 返回 `(subpaths, closed_flags)`，逐子路径保留 `Z`/`z`
     闭合标记，支持绝对/相对 `M/L/H/V/Z` 与隐式续段，对空/非法 `d` 与 `C/Q/S/T/A`
     等不支持指令 fail-closed 返回 `([], [])` 并记入 `report["unsupported"]`；
  2. `_geometry("path", elem)` 默认返回 `(subpaths, bool)`，并在 `per_subpath=True`
     下返回 `(subpaths, closed_flags)` 供 `parse_svg` 保留混合闭合/开放子路径语义；
  3. `_Ctx.draw` 合并同元素全部闭合子路径执行 even-odd 扫描线填充（多块 + 带洞），
     描边按各子路径独立闭合标记切段（开放路径仅描边不误闭合）；
  4. `parse_svg` 几何解析支持无 numpy 的纯标准库环境（Python 3.12 可直接验证）。

运行（仓根）：
  pytest：python -X utf8 -m pytest tests/test_vec_raster_svg_path.py -q
  直跑：  python -X utf8 tests/test_vec_raster_svg_path.py
  自证：  python -X utf8 tests/test_vec_raster_svg_path.py --mutate
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_BRIDGE_ROOT = str(_REPO / "bridges" / "blender")
if _BRIDGE_ROOT not in sys.path:
    sys.path.insert(0, _BRIDGE_ROOT)

from wm_bridge.factory.vec import raster as RA  # noqa: E402
from wm_bridge.factory.vec import svg as SVG_DOC  # noqa: E402

_CHECKS = []
FAILS = []


def _check(name, cond, detail=""):
    if name not in _CHECKS:
        _CHECKS.append(name)
    if cond:
        print("  PASS " + name)
    else:
        print("  FAIL " + name + ("  " + str(detail)[:240] if detail else ""))
        FAILS.append(name)


def _svg_wrap(inner, w=20, h=20):
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
        'viewBox="0 0 %d %d">%s</svg>' % (w, h, w, h, inner)
    )


def _pts_close(actual, expected, tol=1e-6):
    pts = [(float(p[0]), float(p[1])) for p in actual]
    if len(pts) != len(expected):
        return False
    return all(
        abs(ax - ex) <= tol and abs(ay - ey) <= tol
        for (ax, ay), (ex, ey) in zip(pts, expected)
    )


# ─────────────────────────────────────────────────────────── A 组：纯标准库几何契约
def _check_a01_single_closed_path():
    d = "M 2 2 L 18 2 L 18 18 L 2 18 Z"
    elem = ET.fromstring('<path d="%s"/>' % d)
    res = RA._geometry("path", elem)
    assert isinstance(res, tuple) and len(res) == 2, f"期望 2 元组，实得 {res!r}"
    subs, closed = res
    assert isinstance(closed, bool) and closed is True, f"期望 closed=True，实得 {closed!r}"
    assert len(subs) == 1 and _pts_close(subs[0], [(2, 2), (18, 2), (18, 18), (2, 18)])

    shapes, _meta, rep = RA.parse_svg(_svg_wrap('<path d="%s" fill="#ff0000"/>' % d))
    assert len(shapes) == 1 and rep["unsupported"] == {}
    sh = shapes[0]
    assert sh["kind"] == "path" and sh["closed"] is True
    assert sh["subpath_closed"] == [True]
    assert len(sh["subpaths"]) == 1
    assert _pts_close(sh["subpaths"][0], [(2, 2), (18, 2), (18, 18), (2, 18)])


def _check_a02_single_open_path():
    for d, expected in (
        ("M 2 10 L 18 10", [(2, 10), (18, 10)]),
        ("M 2 2 L 18 2 L 18 18", [(2, 2), (18, 2), (18, 18)]),
    ):
        elem = ET.fromstring('<path d="%s"/>' % d)
        subs, closed = RA._geometry("path", elem)
        assert isinstance(closed, bool) and closed is False, f"{d}: closed={closed!r}"
        assert len(subs) == 1 and _pts_close(subs[0], expected)

        shapes, _meta, rep = RA.parse_svg(
            _svg_wrap('<path d="%s" fill="none" stroke="#00ff00" stroke-width="2"/>' % d)
        )
        assert len(shapes) == 1 and rep["unsupported"] == {}
        sh = shapes[0]
        assert sh["closed"] is False and sh["subpath_closed"] == [False]
        assert _pts_close(sh["subpaths"][0], expected)


def _check_a03_two_subpaths():
    d_blocks = "M 2 2 L 8 2 L 8 8 L 2 8 Z M 12 12 L 18 12 L 18 18 L 12 18 Z"
    elem = ET.fromstring('<path d="%s"/>' % d_blocks)
    subs, closed = RA._geometry("path", elem)
    assert isinstance(closed, bool) and closed is True
    assert len(subs) == 2

    shapes, _meta, rep = RA.parse_svg(_svg_wrap('<path d="%s" fill="#ff0000"/>' % d_blocks))
    assert len(shapes) == 1 and rep["unsupported"] == {}
    sh = shapes[0]
    assert sh["closed"] is True and sh["subpath_closed"] == [True, True]
    assert len(sh["subpaths"]) == 2
    assert _pts_close(sh["subpaths"][0], [(2, 2), (8, 2), (8, 8), (2, 8)])
    assert _pts_close(sh["subpaths"][1], [(12, 12), (18, 12), (18, 18), (12, 18)])

    d_hole = "M 2 2 L 18 2 L 18 18 L 2 18 Z M 6 6 L 14 6 L 14 14 L 6 14 Z"
    shapes_h, _meta_h, rep_h = RA.parse_svg(_svg_wrap('<path d="%s" fill="#ff0000"/>' % d_hole))
    assert len(shapes_h) == 1 and rep_h["unsupported"] == {}
    assert len(shapes_h[0]["subpaths"]) == 2
    assert shapes_h[0]["subpath_closed"] == [True, True]


def _check_a04_three_subpaths():
    d = (
        "M 1 1 L 4 1 L 4 4 L 1 4 Z "
        "M 7 7 L 10 7 L 10 10 L 7 10 Z "
        "M 13 13 L 17 13 L 17 17 L 13 17 Z"
    )
    elem = ET.fromstring('<path d="%s"/>' % d)
    subs, closed = RA._geometry("path", elem)
    assert isinstance(closed, bool) and closed is True
    assert len(subs) == 3

    shapes, _meta, rep = RA.parse_svg(_svg_wrap('<path d="%s" fill="#ff0000"/>' % d))
    assert len(shapes) == 1 and rep["unsupported"] == {}
    assert len(shapes[0]["subpaths"]) == 3
    assert shapes[0]["closed"] is True
    assert shapes[0]["subpath_closed"] == [True, True, True]


def _check_a05_mixed_closed_and_open_subpaths():
    # 先闭合后开放
    d = "M 2 2 L 8 2 L 8 8 L 2 8 Z M 12 2 L 18 2 L 18 18"
    elem = ET.fromstring('<path d="%s"/>' % d)
    subs, closed = RA._geometry("path", elem)
    assert isinstance(closed, bool) and closed is False
    assert len(subs) == 2

    subs_p, flags = RA._geometry("path", elem, per_subpath=True)
    assert len(subs_p) == 2 and flags == [True, False]

    shapes, _meta, rep = RA.parse_svg(
        _svg_wrap('<path d="%s" fill="#ff0000" stroke="#00ff00" stroke-width="2"/>' % d)
    )
    assert len(shapes) == 1 and rep["unsupported"] == {}
    sh = shapes[0]
    assert sh["closed"] is False
    assert sh["subpath_closed"] == [True, False]
    assert len(sh["subpaths"]) == 2

    # 先开放后闭合（覆盖遇新 M 刷新前一段开放子路径的分支）
    d2 = "M 2 2 L 8 2 L 8 8 M 12 2 L 18 2 L 18 8 L 12 8 Z"
    shapes2, _m2, rep2 = RA.parse_svg(
        _svg_wrap('<path d="%s" fill="#ff0000" stroke="#00ff00" stroke-width="2"/>' % d2)
    )
    assert len(shapes2) == 1 and rep2["unsupported"] == {}
    assert shapes2[0]["closed"] is False
    assert shapes2[0]["subpath_closed"] == [False, True]


def _check_a06_abs_rel_commands_and_transform():
    # 绝对 M/L/H/V/Z 与 相对 m/l/h/v/z 等价
    d_abs = "M +2 +2 H 18 V 18 H 2 Z"
    d_rel = "m 2 2 h 16 v 16 h -16 z"
    d_imp = "M 2 2 18 2 18 18 2 18 Z"
    d_dup = "M 2 2 L 18 2 L 18 18 L 2 18 L 2 2 Z"
    expected = [(2, 2), (18, 2), (18, 18), (2, 18)]
    for d in (d_abs, d_rel, d_imp, d_dup):
        subs, flags = RA._path_subpaths(d)
        assert flags == [True], f"{d}: flags={flags!r}"
        assert len(subs) == 1 and _pts_close(subs[0], expected), f"{d}: subs={subs!r}"

    # Z 后相对 m（以子路径起点 (2,2) 为基准）
    d_rel_after_z = "M 2 2 L 8 2 L 8 8 L 2 8 Z m 10 10 h 6 v 6 h -6 z"
    subs2, flags2 = RA._path_subpaths(d_rel_after_z)
    assert flags2 == [True, True] and len(subs2) == 2
    assert _pts_close(subs2[1], [(12, 12), (18, 12), (18, 18), (12, 18)])

    # Z 后无 M 直接续 L（从 (sx, sy)=(2, 2) 起新开放子路径）
    d_cont = "M 2 2 L 8 2 L 8 8 Z L 14 14"
    subs3, flags3 = RA._path_subpaths(d_cont)
    assert flags3 == [True, False] and len(subs3) == 2
    assert _pts_close(subs3[1], [(2, 2), (14, 14)])

    # 科学计数法与紧凑符号、叠加 transform（纯标准库路径也需准确）
    svg = _svg_wrap(
        '<g transform="translate(1.5, 2.5) scale(2)">'
        '<path d="M1e0,2e0 l2-1 v2 z" fill="#ff0000"/>'
        '</g>'
    )
    shapes, _meta, rep = RA.parse_svg(svg)
    assert len(shapes) == 1 and rep["unsupported"] == {}
    # 原顶点 (1,2), (3,1), (3,3) → scale(2) → (2,4),(6,2),(6,6) → translate(1.5,2.5)
    assert _pts_close(shapes[0]["subpaths"][0], [(3.5, 6.5), (7.5, 4.5), (7.5, 8.5)])


def _check_a07_empty_and_malformed_d():
    bad_ds = [
        "",
        "   ",
        "10 20 30 40",
        "M 10",
        "M 0 0 L 10",
        "M 0 0 L Z",
        "M 0 0 L",
        "L 10 10",
        "H 10",
        "V 10",
        "Z",
        "M 0 0 L 10 10 Z 5 5",
        "M 0 0 @@@ L 10 10",
        "M 1e400 0 L 10 10",
        "M 5 5",
    ]
    for d in bad_ds:
        elem = ET.fromstring('<path d="%s"/>' % d)
        subs, closed = RA._geometry("path", elem)
        assert subs == [] and closed is False, f"非法 d={d!r} 应返回 ([], False)，实得 {(subs, closed)!r}"
        shapes, _meta, rep = RA.parse_svg(_svg_wrap('<path d="%s" fill="#ff0000"/>' % d))
        assert shapes == [], f"非法 d={d!r} 不应发射图形：{shapes!r}"
        assert rep["unsupported"].get("path:empty") == 1, f"非法 d={d!r} 应记 unsupported：{rep!r}"

    # 缺失 d 属性也不崩
    shapes_none, _m, rep_none = RA.parse_svg(_svg_wrap('<path fill="#ff0000"/>'))
    assert shapes_none == [] and rep_none["unsupported"].get("path:empty") == 1


def _check_a08_unsupported_curve_commands_fail_closed():
    curve_ds = [
        "M 2 2 C 5 2 15 18 18 18",
        "M 2 2 c 3 0 13 16 16 16",
        "M 2 2 Q 10 18 18 2",
        "M 2 2 q 8 16 16 0",
        "M 2 2 S 15 18 18 18",
        "M 2 2 T 18 2",
        "M 2 2 A 5 5 0 0 1 18 18",
    ]
    for d in curve_ds:
        subs, flags = RA._path_subpaths(d)
        assert subs == [] and flags == [], (
            f"曲线指令 {d!r} 不在 M/L/H/V/Z 自用子集内，必须 fail-closed（不得吞字母画假折线），"
            f"实得 {(subs, flags)!r}"
        )
        shapes, _meta, rep = RA.parse_svg(
            _svg_wrap('<path d="%s" fill="none" stroke="#0000ff"/>' % d)
        )
        assert shapes == [] and rep["unsupported"].get("path:empty") == 1


def _check_a09_non_path_primitives_unchanged():
    svg = _svg_wrap(
        '<rect x="1" y="1" width="4" height="4" fill="#ff0000"/>'
        '<polygon points="1,1 5,1 5,5" fill="#00ff00"/>'
        '<polyline points="1,1 5,5 9,1" stroke="#0000ff"/>'
        '<circle cx="10" cy="10" r="3" fill="#ff00ff"/>'
        '<ellipse cx="10" cy="10" rx="4" ry="2" fill="#00ffff"/>'
        '<line x1="0" y1="0" x2="10" y2="10" stroke="#ffffff"/>'
    )
    shapes, _meta, rep = RA.parse_svg(svg)
    assert rep["unsupported"] == {}
    got = [(sh["kind"], sh["closed"], sh["subpath_closed"]) for sh in shapes]
    assert got == [
        ("rect", True, [True]),
        ("polygon", True, [True]),
        ("polyline", False, [False]),
        ("circle", True, [True]),
        ("ellipse", True, [True]),
        ("line", False, [False]),
    ], f"既有原语契约被改变：{got!r}"


def _check_a10_stdlib_numpy_boundary():
    if RA.np is not None:
        return
    svg = _svg_wrap('<path d="M 2 2 L 18 2 L 18 18 Z" fill="#ff0000"/>')
    try:
        RA.rasterize(svg)
    except ModuleNotFoundError:
        pass
    else:
        raise AssertionError("无 numpy 时 rasterize 应抛 ModuleNotFoundError")
    try:
        RA.iou([], [])
    except ModuleNotFoundError:
        pass
    else:
        raise AssertionError("无 numpy 时 iou 应抛 ModuleNotFoundError")


# ─────────────────────────────────────────────────────────── B 组：光栅化像素覆盖（需 numpy）
def _check_b01_single_closed_fill_and_stroke():
    svg_fill = _svg_wrap('<path d="M 2 2 L 18 2 L 18 18 L 2 18 Z" fill="#ff0000"/>')
    img, rep = RA.rasterize(svg_fill, width=20, height=20, ss=1)
    assert rep["shapes_parsed"] == 1 and rep["shapes_drawn"] == 1 and rep["unsupported"] == {}
    painted = int((img[..., 3] > 0).sum())
    assert painted == 256, f"16×16 实心正方形期望 256 像素，实得 {painted}"
    assert tuple(int(c) for c in img[10, 10]) == (255, 0, 0, 255)
    assert int(img[0, 0, 3]) == 0

    svg_stroke = _svg_wrap(
        '<path d="M 2 2 L 18 2 L 18 18 L 2 18 Z" fill="#ff0000" '
        'stroke="#0000ff" stroke-width="2"/>'
    )
    img_s, _ = RA.rasterize(svg_stroke, width=20, height=20, ss=1)
    # 闭合边 (2,18)→(2,2) 中点 (x=2, y=10) 必须被蓝色描边覆盖
    assert int(img_s[10, 2, 2]) > 200 and int(img_s[10, 2, 3]) == 255
    assert tuple(int(c) for c in img_s[10, 10]) == (255, 0, 0, 255)


def _check_b02_relative_matches_absolute():
    svg_abs = _svg_wrap(
        '<path d="M 2 2 L 18 2 L 18 18 L 2 18 Z" fill="#ff0000" '
        'stroke="#0000ff" stroke-width="2"/>'
    )
    svg_rel = _svg_wrap(
        '<path d="m 2 2 h 16 v 16 h -16 z" fill="#ff0000" '
        'stroke="#0000ff" stroke-width="2"/>'
    )
    img_a, _ = RA.rasterize(svg_abs, width=20, height=20, ss=2)
    img_r, _ = RA.rasterize(svg_rel, width=20, height=20, ss=2)
    assert (img_a == img_r).all(), "绝对 M/L/Z 与相对 m/h/v/z 光栅化结果应逐像素一致"


def _check_b03_open_path_stroke_only_no_false_close():
    # L 形开放路径：即使写了 fill 也不应误闭合填充，且不得画斜边 (18,18)→(2,2)
    svg = _svg_wrap(
        '<path d="M 2 2 L 18 2 L 18 18" fill="#ff0000" '
        'stroke="#00ff00" stroke-width="2"/>'
    )
    img, rep = RA.rasterize(svg, width=20, height=20, ss=1)
    assert rep["shapes_parsed"] == 1 and rep["shapes_drawn"] == 1
    # 顶边与右边被绿色描边
    assert int(img[2, 10, 1]) > 200 and int(img[2, 10, 3]) == 255
    assert int(img[10, 18, 1]) > 200 and int(img[10, 18, 3]) == 255
    # 闭合斜边中点 (10, 10) 与内部点 (14, 6) 必须透明
    assert int(img[10, 10, 3]) == 0, f"开放路径不应画回起点斜边：pixel(10,10)={img[10, 10]!r}"
    assert int(img[6, 14, 3]) == 0, f"开放路径不应被错误填充：pixel(14,6)={img[6, 14]!r}"
    # 整幅无红色像素
    assert int((img[..., 0] > 0).sum()) == 0

    # 两点直线开放路径
    svg_line = _svg_wrap(
        '<path d="M 2 10 L 18 10" fill="none" stroke="#00ff00" stroke-width="2"/>'
    )
    img_l, _ = RA.rasterize(svg_line, width=20, height=20, ss=1)
    assert int(img_l[10, 10, 1]) > 200 and int(img_l[4, 10, 3]) == 0


def _check_b04_two_disjoint_subpaths():
    svg = _svg_wrap(
        '<path d="M 2 2 L 8 2 L 8 8 L 2 8 Z M 12 12 L 18 12 L 18 18 L 12 18 Z" '
        'fill="#ff0000"/>'
    )
    img, rep = RA.rasterize(svg, width=20, height=20, ss=1)
    assert rep["shapes_parsed"] == 1 and rep["shapes_drawn"] == 1
    painted = int((img[..., 3] > 0).sum())
    assert painted == 72, f"两块 6×6 期望 72 像素，实得 {painted}"
    assert tuple(int(c) for c in img[5, 5]) == (255, 0, 0, 255)
    assert tuple(int(c) for c in img[15, 15]) == (255, 0, 0, 255)
    assert int(img[10, 10, 3]) == 0


def _check_b05_compound_path_with_hole():
    svg = _svg_wrap(
        '<path d="M 2 2 L 18 2 L 18 18 L 2 18 Z M 6 6 L 14 6 L 14 14 L 6 14 Z" '
        'fill="#ff0000"/>'
    )
    img, rep = RA.rasterize(svg, width=20, height=20, ss=1)
    assert rep["shapes_parsed"] == 1 and rep["shapes_drawn"] == 1
    painted = int((img[..., 3] > 0).sum())
    assert painted == 192, f"16×16 挖去 8×8 洞期望 192 像素，实得 {painted}"
    assert int(img[10, 10, 3]) == 0, f"洞中心 (10,10) 必须透明，实得 {img[10, 10]!r}"
    assert tuple(int(c) for c in img[10, 4]) == (255, 0, 0, 255)
    assert tuple(int(c) for c in img[10, 16]) == (255, 0, 0, 255)


def _check_b06_three_subpaths():
    svg = _svg_wrap(
        '<path d="M 1 1 L 4 1 L 4 4 L 1 4 Z '
        'M 7 7 L 10 7 L 10 10 L 7 10 Z '
        'M 13 13 L 17 13 L 17 17 L 13 17 Z" fill="#ff0000"/>'
    )
    img, rep = RA.rasterize(svg, width=20, height=20, ss=1)
    assert rep["shapes_parsed"] == 1 and rep["shapes_drawn"] == 1
    painted = int((img[..., 3] > 0).sum())
    assert painted == 34, f"3×3 + 3×3 + 4×4 期望 34 像素，实得 {painted}"
    assert tuple(int(c) for c in img[2, 2]) == (255, 0, 0, 255)
    assert tuple(int(c) for c in img[8, 8]) == (255, 0, 0, 255)
    assert tuple(int(c) for c in img[15, 15]) == (255, 0, 0, 255)
    assert int(img[5, 5, 3]) == 0 and int(img[11, 11, 3]) == 0


def _check_b07_mixed_closed_and_open_rasterize():
    svg = _svg_wrap(
        '<path d="M 2 2 L 8 2 L 8 8 L 2 8 Z M 12 2 L 18 2 L 18 18" '
        'fill="#ff0000" stroke="#00ff00" stroke-width="2"/>'
    )
    img, rep = RA.rasterize(svg, width=20, height=20, ss=1)
    assert rep["shapes_parsed"] == 1 and rep["shapes_drawn"] == 1
    # 子路径 0（闭合）：中心 (5,5) 填充红，闭合左边 (2,5) 描边绿
    assert tuple(int(c) for c in img[5, 5]) == (255, 0, 0, 255)
    assert int(img[5, 2, 1]) > 200 and int(img[5, 2, 3]) == 255
    # 子路径 1（开放）：顶边 (15,2) 与右边 (18,10) 描边绿，内部 (16,6) 与斜边 (15,10) 透明
    assert int(img[2, 15, 1]) > 200 and int(img[2, 15, 3]) == 255
    assert int(img[10, 18, 1]) > 200 and int(img[10, 18, 3]) == 255
    assert int(img[6, 16, 3]) == 0, f"混合路径的开放子路径不应填充：{img[6, 16]!r}"
    assert int(img[10, 15, 3]) == 0, f"混合路径的开放子路径不应闭合描边：{img[10, 15]!r}"

    # 先开放后闭合对称验证
    svg2 = _svg_wrap(
        '<path d="M 2 2 L 8 2 L 8 8 M 12 2 L 18 2 L 18 8 L 12 8 Z" '
        'fill="#ff0000" stroke="#00ff00" stroke-width="2"/>'
    )
    img2, _ = RA.rasterize(svg2, width=20, height=20, ss=1)
    assert tuple(int(c) for c in img2[5, 15]) == (255, 0, 0, 255)
    assert int(img2[5, 12, 1]) > 200 and int(img2[5, 12, 3]) == 255
    assert int(img2[5, 5, 3]) == 0, f"前置开放子路径不应画回起点斜边：{img2[5, 5]!r}"


def _check_b08_empty_and_unsupported_rasterize_no_crash():
    for d in ("", "M 2 2 C 5 2 15 18 18 18", "M 2 2 Q 10 18 18 2", "M 10", "not-a-path"):
        svg = _svg_wrap('<path d="%s" fill="#ff0000" stroke="#00ff00"/>' % d)
        img, rep = RA.rasterize(svg, width=20, height=20, ss=1)
        assert int((img[..., 3] > 0).sum()) == 0, f"{d!r} 不应画出像素"
        assert rep["shapes_parsed"] == 0 and rep["shapes_drawn"] == 0
        assert rep["unsupported"].get("path:empty") == 1


def _check_b09_svg_doc_and_anim_roundtrip():
    doc = SVG_DOC.SVGDocument(40, 40)
    lay = doc.layer("body")
    doc.path("M 4 4 L 36 4 L 36 36 L 4 36 Z M 12 12 L 28 12 L 28 28 L 12 28 Z",
             layer=lay, fill="#cc3300")
    img, rep = RA.rasterize(doc.to_svg(), width=40, height=40, ss=1)
    assert rep["shapes_parsed"] == 1 and rep["shapes_drawn"] == 1
    assert int((img[..., 3] > 0).sum()) == 32 * 32 - 16 * 16
    assert int(img[20, 20, 3]) == 0

    from wm_bridge.factory.vec import bpy_vec as BV
    from wm_bridge.factory.vec import geom2d as G
    from wm_bridge.factory.vec import prims as P

    obj = BV.DrawObject("cube", P.box((1.0, 1.0, 1.0)))
    with tempfile.TemporaryDirectory(prefix="vec_raster_path_") as td:
        out_path = os.path.join(td, "anim.svg")
        BV.export_anim_document(
            [obj],
            lambda az: G.Camera(eye=(3, 0, 1), look_at=(0, 0, 0), width=64, height=64),
            out_path,
            keyframes=2,
        )
        img_a, rep_a = RA.rasterize(out_path, width=64, height=64, ss=1)
        assert rep_a["shapes_parsed"] >= 1 and rep_a["shapes_drawn"] >= 1
        assert int((img_a[..., 3] > 0).sum()) > 0


# ─────────────────────────────────────────────────────────── pytest 入口
def test_a01_single_closed_path():
    _check_a01_single_closed_path()


def test_a02_single_open_path():
    _check_a02_single_open_path()


def test_a03_two_subpaths():
    _check_a03_two_subpaths()


def test_a04_three_subpaths():
    _check_a04_three_subpaths()


def test_a05_mixed_closed_and_open_subpaths():
    _check_a05_mixed_closed_and_open_subpaths()


def test_a06_abs_rel_commands_and_transform():
    _check_a06_abs_rel_commands_and_transform()


def test_a07_empty_and_malformed_d():
    _check_a07_empty_and_malformed_d()


def test_a08_unsupported_curve_commands_fail_closed():
    _check_a08_unsupported_curve_commands_fail_closed()


def test_a09_non_path_primitives_unchanged():
    _check_a09_non_path_primitives_unchanged()


def test_a10_stdlib_numpy_boundary():
    _check_a10_stdlib_numpy_boundary()


def test_b01_single_closed_fill_and_stroke():
    if RA.np is None:
        return
    _check_b01_single_closed_fill_and_stroke()


def test_b02_relative_matches_absolute():
    if RA.np is None:
        return
    _check_b02_relative_matches_absolute()


def test_b03_open_path_stroke_only_no_false_close():
    if RA.np is None:
        return
    _check_b03_open_path_stroke_only_no_false_close()


def test_b04_two_disjoint_subpaths():
    if RA.np is None:
        return
    _check_b04_two_disjoint_subpaths()


def test_b05_compound_path_with_hole():
    if RA.np is None:
        return
    _check_b05_compound_path_with_hole()


def test_b06_three_subpaths():
    if RA.np is None:
        return
    _check_b06_three_subpaths()


def test_b07_mixed_closed_and_open_rasterize():
    if RA.np is None:
        return
    _check_b07_mixed_closed_and_open_rasterize()


def test_b08_empty_and_unsupported_rasterize_no_crash():
    if RA.np is None:
        return
    _check_b08_empty_and_unsupported_rasterize_no_crash()


def test_b09_svg_doc_and_anim_roundtrip():
    if RA.np is None:
        return
    _check_b09_svg_doc_and_anim_roundtrip()


# ─────────────────────────────────────────────────────────── 脚本直跑 + 定点变异自证
_A_GROUP = [
    ("A01", "单子路径闭合 <path ... Z/> 解包与几何", _check_a01_single_closed_path),
    ("A02", "单子路径开放 <path .../> closed=False", _check_a02_single_open_path),
    ("A03", "双子路径（多块与带洞）解包与几何", _check_a03_two_subpaths),
    ("A04", "三子路径解包与几何", _check_a04_three_subpaths),
    ("A05", "混合闭合+开放子路径保留逐条 closed", _check_a05_mixed_closed_and_open_subpaths),
    ("A06", "绝对/相对 M/L/H/V/Z 与 transform", _check_a06_abs_rel_commands_and_transform),
    ("A07", "空/非法 d fail-closed 不崩", _check_a07_empty_and_malformed_d),
    ("A08", "不支持的 C/Q/S/T/A fail-closed 记 unsupported", _check_a08_unsupported_curve_commands_fail_closed),
    ("A09", "非 path 原语契约不变", _check_a09_non_path_primitives_unchanged),
    ("A10", "无 numpy 纯标准库边界", _check_a10_stdlib_numpy_boundary),
]

_B_GROUP = [
    ("B01", "单闭合路径填充+闭合描边像素精确", _check_b01_single_closed_fill_and_stroke),
    ("B02", "相对 m/h/v/z 与绝对 M/L/Z 逐像素一致", _check_b02_relative_matches_absolute),
    ("B03", "开放路径仅描边不误闭合不误填充", _check_b03_open_path_stroke_only_no_false_close),
    ("B04", "双分离闭合子路径像素覆盖", _check_b04_two_disjoint_subpaths),
    ("B05", "带洞复合路径 even-odd 挖洞精确", _check_b05_compound_path_with_hole),
    ("B06", "三闭合子路径像素覆盖", _check_b06_three_subpaths),
    ("B07", "混合闭合+开放路径光栅化区分", _check_b07_mixed_closed_and_open_rasterize),
    ("B08", "空/曲线/非法 d 光栅化零崩零像素", _check_b08_empty_and_unsupported_rasterize_no_crash),
    ("B09", "SVGDocument.path 与 bpy_vec anim.svg 端到端光栅化", _check_b09_svg_doc_and_anim_roundtrip),
]


def _run_one(code, label, fn):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - 测试运行器统一汇总异常为 FAIL
        _check(f"{code} {label}", False, f"{type(exc).__name__}: {exc}")
    else:
        _check(f"{code} {label}", True)


def _run_assertions():
    global FAILS
    FAILS = []
    del _CHECKS[:]
    print("== A 组：SVG <path> 几何契约（纯标准库可用）==")
    for code, label, fn in _A_GROUP:
        _run_one(code, label, fn)
    if RA.np is not None:
        print("== B 组：SVG <path> 光栅化像素覆盖（numpy）==")
        for code, label, fn in _B_GROUP:
            _run_one(code, label, fn)
    else:
        print("== B 组：跳过（当前环境无 numpy，仅运行纯标准库几何校验）==")
    print("通过 %d / 失败 %d" % (len(_CHECKS) - len(FAILS), len(FAILS)))
    return len(FAILS)


_RASTER_REL = "bridges/blender/wm_bridge/factory/vec/raster.py"

_ARMS = {
    "M1": (
        _RASTER_REL,
        [(
            '    elif tag == "path":\n'
            '        subs, flags = _path_subpaths(e.get("d") or "")\n'
            '        if per_subpath:\n'
            '            return subs, flags\n'
            '        return subs, (bool(flags) and all(flags))',
            '    elif tag == "path":\n'
            '        return _path_subpaths(e.get("d") or "")[0]',
        )],
        {"A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08"},
        {"A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08",
         "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09"},
        False,
    ),
    "M2": (
        _RASTER_REL,
        [(
            '    if cur:\n'
            '        _flush(False)',
            '    if cur:\n'
            '        _flush(True)',
        )],
        {"A02", "A05", "A06"},
        {"A02", "A05", "A06", "B03", "B07"},
        False,
    ),
    "M3": (
        _RASTER_REL,
        [(
            '                if ix1 >= ix0 and iy1 >= iy0:\n'
            '                    cov = _fill_subpaths(fill_polys, ix0, ix1, iy0, iy1)\n'
            '                    _composite(self.img, self.painted, cov, fill, fa, ix0, iy0)\n'
            '                    touched += 1',
            '                if ix1 >= ix0 and iy1 >= iy0:\n'
            '                    for poly in fill_polys:\n'
            '                        cov = _fill_coverage(poly, ix0, ix1, iy0, iy1)\n'
            '                        _composite(self.img, self.painted, cov, fill, fa, ix0, iy0)\n'
            '                    touched += 1',
        )],
        set(),
        {"B05", "B09"},
        True,
    ),
    "M4": (
        _RASTER_REL,
        [(
            '_PATH_TOKEN_RE = re.compile(\n'
            '    r"[-+]?(?:\\d*\\.\\d+|\\d+\\.?)(?:[eE][-+]?\\d+)?|[A-Za-z]|[^\\s,]"\n'
            ')',
            '_PATH_TOKEN_RE = re.compile(\n'
            '    r"[MmLlHhVvZz]|-?(?:\\d*\\.\\d+|\\d+\\.?)(?:[eE][-+]?\\d+)?"\n'
            ')',
        )],
        {"A07", "A08"},
        {"A07", "A08", "B08"},
        False,
    ),
}


def _subprocess_fail_codes():
    for cache in (_REPO / "bridges" / "blender" / "wm_bridge" / "factory" / "vec").rglob("__pycache__"):
        for pyc in cache.glob("raster.*.pyc"):
            pyc.unlink(missing_ok=True)
    proc = subprocess.run(
        [sys.executable, "-X", "utf8", __file__, "--assert-only"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(_REPO), timeout=300,
    )
    codes = set(re.findall(r"(?m)^\s*FAIL\s+([AB]\d+)", proc.stdout or ""))
    return codes, (proc.stdout or "") + (proc.stderr or "")


def _mutate(argv):
    want = ""
    for arg in argv:
        if arg == "--mutate":
            want = want or "ALL"
        elif arg.startswith("--mutate="):
            want = arg.split("=", 1)[1]
    if not want:
        return 0
    ids = sorted(_ARMS) if want == "ALL" else [want]
    red = 0
    ran = 0
    has_np = RA.np is not None
    for mid in ids:
        if mid not in _ARMS:
            print("[ANCHOR-MISS] %s：未登记该变异臂" % mid)
            return 2
        rel, edits, expect_stdlib, expect_np, needs_np = _ARMS[mid]
        if needs_np and not has_np:
            print("[MUTATE-SKIP] %s：需要 numpy（纯标准库环境跳过）" % mid)
            continue
        ran += 1
        expect = expect_np if has_np else expect_stdlib
        target = _REPO / rel
        orig = target.read_bytes()
        try:
            text = orig.decode("utf-8")
            nl = "\r\n" if "\r\n" in text else "\n"
            norm = text.replace("\r\n", "\n")
            for old, new in edits:
                cnt = norm.count(old)
                if cnt != 1:
                    print("[ANCHOR-MISS] %s：锚点命中 %d 次（期望 1）" % (mid, cnt))
                    return 2
                norm = norm.replace(old, new)
            target.write_bytes(norm.replace("\n", nl).encode("utf-8"))
            actual, out = _subprocess_fail_codes()
            if actual == expect:
                print("[MUTATE-OK]   %s：指定断言精确转红 %s" % (mid, sorted(actual)))
            else:
                print("[MUTATE-FAIL] %s：期望红 %s，实际红 %s"
                      % (mid, sorted(expect), sorted(actual)))
                for line in out.splitlines():
                    if line.strip().startswith("FAIL"):
                        print("    " + line)
                red += 1
        finally:
            target.write_bytes(orig)
    print("反证结果：%d 条变异，%s" % (ran, "全部恰好命中" if red == 0 else "%d 条未命中" % red))
    return 1 if red else 0


if __name__ == "__main__":
    _f = _run_assertions()
    if "--assert-only" in sys.argv:
        sys.exit(1 if _f else 0)
    _m = _mutate(sys.argv[1:])
    sys.exit(1 if (_f or _m) else 0)
