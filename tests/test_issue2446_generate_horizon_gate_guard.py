# -*- coding: utf-8 -*-
"""test_issue2446_generate_horizon_gate_guard —— world_model.generate horizon 闸守卫。

缺陷（dsh-bounty #2446）
============================================================================
`world_model` 门面的 `generate` 动作 `horizon=int(p.get(…))` 双层静默截断/
裸崩：`generate(horizon=nan)` 裸 ValueError（"cannot convert float NaN to
integer"）穿透门面、2.7 静默截断成 2、字符串裸崩 invalid literal。同门面
`counterfactual` 已显式闸 horizon ∈ 1..64（fail-closed）——一图两制。
tick/步数计数非 int fail-loud 裁决已三处落地（#2409/#2412/#2420）。

实现侧口径（本守卫的判据）
--------------------------------------------------------------------------
底层 `UnifiedWorldModel.generate`：horizon 非 None 且非 int（bool 同拒）⇒
TypeError；**0/负数钳 1 的既有行为保持不变**（与既有 `max(1, int(…))` 一致）。
门面 "generate"：判型后非 int 转 error dict（同门面 fail-closed 惯例）。

守卫用例
--------------------------------------------------------------------------
G1 缺陷态：门面 generate horizon=nan/2.7/"x"/True ⇒ error dict（改前裸崩/截断）。
G2 正对照：合法 horizon=3 照常出预测；horizon=0 钳 1 既有行为保持。
G3 底层：generate(horizon=2.7/True) ⇒ TypeError；horizon=None 缺省路径照常。
G4 一图一制：同门面 counterfactual 的既有 1..64 闸不回退（直调断言）。
G5 文案可判读：点名形参名与原值。

运行（lingshu 仓根）：
    python -X utf8 -m pytest tests/test_issue2446_generate_horizon_gate_guard.py -q
"""
from __future__ import annotations

import pytest

from lingshu.core.core import SpacetimeMemoryEngine
from lingshu.world.world_model import UnifiedWorldModel


def _engine():
    e = SpacetimeMemoryEngine(":memory:")
    assert e.world_model("init", {})["status"] == "ok"
    assert e.world_model("create", {})["status"] == "ok"
    return e


# ---------------------------------------------------------------- G1

@pytest.mark.parametrize("bad", [float("nan"), 2.7, "x", True])
def test_2446_facade_generate_horizon_rejected(bad):
    e = _engine()
    r = e.world_model("generate", {"horizon": bad})
    assert r.get("status") == "error" and "horizon" in r.get("error", ""), (
        "#2446 门面 generate horizon=%r 未被可读拒绝：%r" % (bad, r))


# ---------------------------------------------------------------- G2

def test_2446_legal_horizon_and_clamp_unchanged():
    e = _engine()
    r = e.world_model("generate", {"horizon": 3})
    assert r.get("status") == "ok", r
    r = e.world_model("generate", {"horizon": 0})     # 钳 1 既有行为
    assert r.get("status") == "ok", r
    r = e.world_model("generate", {"horizon": -2})    # 负数钳 1 既有行为
    assert r.get("status") == "ok", r


# ---------------------------------------------------------------- G3

def test_2423_underlying_generate_rejects_non_int():
    wm = UnifiedWorldModel(size=24, seed=1)
    with pytest.raises(TypeError) as ei:
        wm.generate(horizon=2.7)
    assert "horizon" in str(ei.value), str(ei.value)
    with pytest.raises(TypeError):
        wm.generate(horizon=True)
    r = wm.generate(horizon=None)                     # 缺省路径照常
    assert isinstance(r, dict) and "predictions" in r, r.keys()


# ---------------------------------------------------------------- G4

def test_2424_counterfactual_gate_unchanged():
    """同文件 counterfactual 的既有 1..64 闸（底层 ValueError）不回退。
    其门面裸穿属 #2423 同族既留面，不在本票范围。"""
    e = _engine()
    wm = e._wmodel
    with pytest.raises(ValueError):
        wm.counterfactual(horizon=0)
    r = wm.counterfactual(horizon=1)
    assert r.get("schema") == "wm.counterfactual/1", r.keys()


# ---------------------------------------------------------------- G5

def test_2446_message_names_param_and_value():
    e = _engine()
    r = e.world_model("generate", {"horizon": 2.7})
    msg = r.get("error", "")
    assert "horizon" in msg and "2.7" in msg, msg
