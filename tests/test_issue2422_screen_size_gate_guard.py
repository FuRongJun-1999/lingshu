# -*- coding: utf-8 -*-
"""test_issue2422_screen_size_gate_guard —— world3d 门面 screen_w/h 预闸守卫。

缺陷（dsh-bounty #2422）
============================================================================
`world3d` 门面 4 个动作（build/render/add/add_view）的
`sw = int(p.get("screen_w", 800))` 零类型/值域预闸：
  · `screen_w=nan` ⇒ 裸 `ValueError: cannot convert float NaN to integer`
    **穿透门面**（模型驱动面拿到不可读崩溃，非 error dict）；
  · `screen_w=-50` ⇒ int 通过、**PIL 内部**抛 `Width and height must be >= 0`
    （泄露实现层）；
  · `screen_w=800.5` ⇒ 静默截断 800。
#416 的 MAX 闸在 `int()` **之后**——挡得住大数、挡不住这三类。

实现侧口径（本守卫的判据）
--------------------------------------------------------------------------
门面内新增 `resolve_screen` 嵌套 helper（与 `resolve_ingest_camera` 同风格），
4 站点换用：非 int（bool 同拒）/非正 ⇒ `{"status": "error",
"reason": "bad_screen_size", …}`（同门面 fail-closed 惯例）；#416 的 MAX 闸
保持不动（职责不同：MAX=上限语义、本闸=类型/正值语义）；合法输入逐位不变。

守卫用例
--------------------------------------------------------------------------
G1 缺陷态：render 的 screen_w=nan/-50/800.5 ⇒ error reason=bad_screen_size
   （改前裸崩/裸崩/静默截断）。
G2 覆盖面：add/build/add_view 四站点全部预闸（screen_h 同查）。
G3 正对照：合法尺寸 render 照常出图；#416 的 MAX 上限语义保持（直调 API
   不受影响，由既有 #416 守卫承载）。
G5 文案可判读：点名 screen_w/screen_h 与原值。

运行（lingshu 仓根）：
    python -X utf8 -m pytest tests/test_issue2422_screen_size_gate_guard.py -q
"""
from __future__ import annotations

import pytest

from lingshu.core.core import SpacetimeMemoryEngine


def _engine():
    e = SpacetimeMemoryEngine(":memory:")
    r = e.world3d("add", {"category": "box", "bbox": [380, 280, 420, 320]})
    assert r.get("status") == "ok", r
    return e


# ---------------------------------------------------------------- G1

@pytest.mark.parametrize("bad", [float("nan"), -50, 800.5, True])
def test_2422_render_screen_w_rejected(bad):
    e = _engine()
    r = e.world3d("render", {"screen_w": bad, "screen_h": 600})
    assert r.get("status") == "error" and r.get("reason") == "bad_screen_size", (
        "#2422 render 的非法 screen_w %r 未被可读拒绝：%r" % (bad, r))


# ---------------------------------------------------------------- G2

@pytest.mark.parametrize("act,params", [
    ("add", {"category": "ball", "bbox": [1, 2, 3, 4], "screen_w": float("nan")}),
    ("add_view", {"category": "ball", "bbox": [1, 2, 3, 4], "screen_h": 0}),
    ("build", {"screen_w": -1}),
])
def test_2422_all_four_sites_gated(act, params):
    e = _engine()
    r = e.world3d(act, params)
    assert r.get("status") == "error" and r.get("reason") == "bad_screen_size", (
        "#2422 %s 的非法 screen 尺寸未被预闸：%r" % (act, r))


# ---------------------------------------------------------------- G3

def test_2422_legal_size_renders():
    e = _engine()
    r = e.world3d("render", {"screen_w": 640, "screen_h": 480})
    assert r.get("status") == "ok", r


# ---------------------------------------------------------------- G5

def test_2422_message_names_param_and_value():
    e = _engine()
    r = e.world3d("render", {"screen_w": 800.5})
    msg = r.get("error", "")
    assert "screen_w" in msg and "800.5" in msg, msg
