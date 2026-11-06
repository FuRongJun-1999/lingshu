# -*- coding: utf-8 -*-
"""test_issue2418_bbox_finite_gate_guard —— world3d 门面 bbox 有限性守卫。

缺陷（dsh-bounty #2418）
============================================================================
`world3d` 门面的 bbox 三条消费路径两种口径：
  · 带 camera 的持久化路径（persist_bound_observation → format_camera_observation）
    对 bbox 逐元素 _meta_number 闸 ⇒ 非有限被拒（"bbox[0] 必须是有限实数"）；
  · `add`（VPrim 直构）与 `add_view`（无 camera）**零有限性校验**——NaN/±inf
    直通进 VPrim 落世界图 ⇒ `graph`/`status` 导出严格 JSON 当场崩
    （与 #2400/#2404 同族，但落点是门面写入口——修源头优于导出面兜底）。
同函数内 confidence/tick/ts 已有 _mn 闸（#2148）——bbox 是同动作漏掉的同族参数。

实现侧口径（本守卫的判据）
--------------------------------------------------------------------------
两动作在 bbox 长度检查后复用 `_require_finite`（与持久化路径同一文案），
非有限/非数值 ⇒ `{"status": "error", "reason": "bad_bbox", …}`（同门面
fail-closed 惯例）；带 camera 路径行为不变；合法输入逐位不变。

守卫用例
--------------------------------------------------------------------------
G1 缺陷态：add 的 bbox 含 nan / inf / -inf ⇒ error reason=bad_bbox（改前 ok）。
G2 缺陷态：add_view 同型（无 camera 路径）。
G3 缺陷态（导出面）：修前 graph 严格 JSON 崩 ⇒ 修后（拒在入口）graph JSON OK。
G4 正对照：合法 bbox 的 add/add_view 照旧 ok；带 camera 的持久化路径照旧
   拒非有限（既有闸不回退）。
G5 文案可判读：点名 bbox[i] 与「拒绝 NaN/±inf」。

运行（lingshu 仓根）：
    python -X utf8 -m pytest tests/test_issue2418_bbox_finite_gate_guard.py -q
"""
from __future__ import annotations

import json

import pytest

from lingshu.core.core import SpacetimeMemoryEngine


def _engine():
    e = SpacetimeMemoryEngine(":memory:")
    r = e.world3d("add", {"category": "box", "bbox": [380, 280, 420, 320]})
    assert r.get("status") == "ok", r
    return e


def _cam():
    return {"cx": 0.0, "cy": 1.2, "cz": 0.0, "yaw": 0.0,
            "pitch": 0.0, "fov_deg": 60.0}


# ---------------------------------------------------------------- G1

@pytest.mark.parametrize("bad", [[float("nan"), 280, 420, 320],
                                 [380, float("inf"), 420, 320],
                                 [380, 280, 420, float("-inf")]])
def test_2418_add_rejects_nonfinite_bbox(bad):
    e = _engine()
    r = e.world3d("add", {"category": "ball", "bbox": bad,
                          "screen_w": 800, "screen_h": 600})
    assert r.get("status") == "error" and r.get("reason") == "bad_bbox", (
        "#2418 add 的非有限 bbox 仍被接受：%r" % (r,))


# ---------------------------------------------------------------- G2

@pytest.mark.parametrize("bad", [[float("nan"), 280, 420, 320],
                                 [380, 280, float("inf"), 320]])
def test_2418_add_view_rejects_nonfinite_bbox(bad):
    e = _engine()
    r = e.world3d("add_view", {"category": "ball", "bbox": bad,
                               "screen_w": 800, "screen_h": 600})
    assert r.get("status") == "error" and r.get("reason") == "bad_bbox", (
        "#2418 add_view（无 camera）的非有限 bbox 仍被接受：%r" % (r,))


# ---------------------------------------------------------------- G3

def test_2418_graph_export_json_safe():
    """拒在入口 ⇒ 世界图不含 NaN ⇒ graph 严格 JSON 安全（改前当场崩）。"""
    e = _engine()
    r = e.world3d("add", {"category": "ball",
                          "bbox": [float("nan"), 280, 420, 320]})
    assert r.get("status") == "error", r
    g = e.world3d("graph", {})
    json.dumps(g, allow_nan=False)          # 不抛即过


# ---------------------------------------------------------------- G4

def test_2418_legal_paths_unchanged():
    e = _engine()
    assert e.world3d("add", {"category": "ball",
                             "bbox": [100, 200, 300, 400]})["status"] == "ok"
    assert e.world3d("add_view", {"category": "ball", "bbox": [100, 200, 300, 400],
                                  "screen_w": 800, "screen_h": 600})["status"] == "ok"
    # 带 camera 的持久化路径既有闸不回退（非有限 bbox 依旧拒，reason 同为 bad_bbox）
    r = e.world3d("add_view", {"category": "ball",
                               "bbox": [float("nan"), 280, 420, 320],
                               "screen_w": 800, "screen_h": 600,
                               "camera": _cam()})
    assert r.get("status") == "error", r


# ---------------------------------------------------------------- G5

def test_2418_message_names_index_and_rejects_nan():
    e = _engine()
    try:
        e.world3d("add", {"category": "ball",
                          "bbox": [380, float("nan"), 420, 320]})
    except AssertionError:
        raise
    except ValueError:
        raise
    r = e.world3d("add", {"category": "ball",
                          "bbox": [380, float("nan"), 420, 320]})
    msg = r.get("error", "")
    assert "bbox[1]" in msg and "NaN" in msg, msg
