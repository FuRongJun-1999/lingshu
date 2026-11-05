# -*- coding: utf-8 -*-
"""test_issue2382_render_camera_guard —— world3d 门面 render 相机参数守卫。

缺陷（dsh-bounty #2382）
============================================================================
`lingshu/core/core.py` 的 `world3d("render")` 分支构造入参相机只读 4 参
（yaw/pitch/cx/cy）——`p["cz"]` 与 `p["fov_deg"]` 从不读取，Camera3D 落默认
`cz=0.0`/`fov_deg=60.0`。这是 `Camera3D.to_meta` docstring（F04 段）点名过的
「Camera3D(yaw,pitch,cx,cy) 四参构造静默丢 cz/fov」病；R6 返工把 observe/add/
add_view 摄取侧收口成 strict 六参（resolve_ingest_camera），render 分支残留。
危害：docstring 声明「任意视角透视投影」但视角 Z 分量与视场角不可达；调用方
传 cz/fov_deg 拿到与缺省**逐字节相同**的渲染，零可辨识性。

实现侧口径（本守卫的判据）
--------------------------------------------------------------------------
写面补读两参，默认值与 Camera3D 字段默认逐位相同（cz=0.0/fov_deg=60.0）⇒
不传参调用方行为逐位不变；fov 别名对齐 add 动作六参契约。

守卫用例（PNG 字节 sha256 判等；fixture 保证场景非空——box 先入库）
--------------------------------------------------------------------------
G1 缺陷态：render(cz=100) 与基线渲染必须**不同**（改前逐字节相同）。
G2 缺陷态：render(fov_deg=120) 与基线渲染必须**不同**（改前逐字节相同）。
G3 fov 别名：render(fov=120) 与 render(fov_deg=120) 逐字节相同。
G4 零行为变化：显式缺省（cz=0/fov_deg=60）与隐式缺省渲染逐字节相同。
G5 契约面：world3d docstring 的 render 参数行列出 cz/fov_deg（cy 亦在列）。

运行（lingshu 仓根）：
    python -X utf8 -m pytest tests/test_issue2382_render_camera_guard.py -q
"""
from __future__ import annotations

import hashlib
import inspect

import pytest

from lingshu.core.core import SpacetimeMemoryEngine


def _engine():
    e = SpacetimeMemoryEngine(":memory:")
    r = e.world3d("add", {"category": "box", "bbox": [380, 280, 420, 320]})
    assert r.get("status") == "ok", r  # 场景非空：防"空渲恒同"假绿
    return e


def _h(engine, params):
    r = engine.world3d("render", dict(params))
    assert r.get("status") == "ok", r
    return hashlib.sha256(bytes(r["bytes"])).hexdigest()


def test_2382_render_honors_cz():
    """cz=100 的渲染必须与基线不同（改前逐字节相同）。"""
    e = _engine()
    assert _h(e, {"yaw": 0.0}) != _h(e, {"yaw": 0.0, "cz": 100.0}), (
        "#2382 render 的 cz 仍被静默丢弃（与缺省渲染逐字节相同）")


def test_2382_render_honors_fov_deg():
    """fov_deg=120 的渲染必须与基线不同（改前逐字节相同）。"""
    e = _engine()
    assert _h(e, {"yaw": 0.0}) != _h(e, {"yaw": 0.0, "fov_deg": 120.0}), (
        "#2382 render 的 fov_deg 仍被静默丢弃")


def test_2382_fov_alias_matches_fov_deg():
    """fov 别名与 fov_deg 同效（对齐 add 动作六参契约）。"""
    e = _engine()
    assert _h(e, {"yaw": 0.0, "fov": 120.0}) == _h(e, {"yaw": 0.0, "fov_deg": 120.0})


def test_2382_explicit_defaults_unchanged():
    """零行为变化：显式缺省（cz=0/fov_deg=60）与隐式缺省渲染逐字节相同。"""
    e = _engine()
    assert _h(e, {"yaw": 0.0}) == _h(e, {"yaw": 0.0, "cz": 0.0, "fov_deg": 60.0})


def test_2382_docstring_declares_cz_fov():
    """契约面：world3d docstring 的 render 参数行列出 cz/fov_deg。"""
    from lingshu.core.core import SpacetimeMemoryEngine as E
    src = inspect.getsource(E.world3d)
    render_line = src[src.find("- render:"):src.find("- status:")]
    assert "cz" in render_line and "fov_deg" in render_line, (
        "#2382 docstring render 参数行须列出 cz/fov_deg（「任意视角」须真）")
