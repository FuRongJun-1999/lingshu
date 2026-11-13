# -*- coding: utf-8 -*-
"""#1136：三处输入校验缺口的收口守卫。

契约（docstring 既有承诺 + issue 期望行为）：
  · `Environment.up`/`incline_axis`：轴索引须为 **int**（`1.0 in (0,1,2)` 为
    True 的浮点漏网 ⇒ 错误延后到 step 才抛、set_environment 已登记污染状态）
    ——浮点/bool 一律结构化拒绝；
  · `environment_readout` 的 readout 是**副本**（审计读数不被外部污染）；
  · `transfer_demonstration` 的 `scale` 必须为**正的有限数**
    （`nan <= 0.0`/`inf <= 0.0` 均 False 的穿透门收口）。
"""
import math

import pytest

from lingshu.world.scene_simulator import Environment, SceneSimulator
from lingshu.world.world_process import transfer_demonstration


# ---- 现象一：浮点轴索引 ----

@pytest.mark.parametrize("bad_up", [1.0, 2.0, True])
def test_environment_rejects_float_up(bad_up):
    with pytest.raises(ValueError, match="0/1/2"):
        Environment(up=bad_up)


@pytest.mark.parametrize("bad_axis", [2.0, 1.0, True])
def test_environment_rejects_float_incline_axis(bad_axis):
    with pytest.raises(ValueError, match="0/1/2"):
        Environment(up=1, incline_axis=bad_axis)


def test_int_axes_still_accepted_regression():
    env = Environment(up=1, incline_axis=0)
    assert env.up == 1 and env.incline_axis == 0


# ---- 现象二：readout 活引用 ----

def test_environment_readout_returns_copy():
    # readout 只对动力学实体非 None（attrs["physics"]="dynamics" opt-in 标记）。
    s = SceneSimulator(size=24, seed=1)
    e = s.add_entity("e", "wander", (5.0, 1.5, 5.0), 0.3)
    s.entities[e].attrs["physics"] = "dynamics"
    s.set_environment(slope_deg=10.0)
    s.step(1)
    assert s.entities[e].attrs.get("physics_readout") is not None
    ro = s.environment_readout(e)
    assert ro["readout"] is not None
    ro["readout"]["pos"] = (999.0, 999.0, 999.0)
    assert s.entities[e].attrs["physics_readout"]["pos"] != (999.0, 999.0, 999.0), (
        "审计读数是活引用（外部改写永久污染仿真状态）")
    assert s.environment_readout(e)["readout"] is not s.entities[e].attrs["physics_readout"]


# ---- 现象三：transfer_demonstration 的 scale 收 NaN/inf ----

def _record():
    # 真实 record 形态（extract_relative_trajectory 产物）：每采样含
    # ref / rel（actor 相对 ref 的位移）/ dt / tick。
    return {"samples": [
        {"tick": 1, "dt": 1, "ref": [0.0, 0.0, 0.0], "rel": [1.0, 0.0, 0.0],
         "entities": {"a": [1.0, 1.5, 2.0]}},
        {"tick": 2, "dt": 1, "ref": [1.0, 1.5, 2.0], "rel": [1.0, 0.0, 0.0],
         "entities": {"a": [2.0, 1.5, 2.0]}},
    ]}


@pytest.mark.parametrize("bad_scale", [float("nan"), float("inf"), float("-inf")])
def test_transfer_demonstration_rejects_nonfinite_scale(bad_scale):
    with pytest.raises(ValueError, match="正的有限数"):
        transfer_demonstration(_record(), "a2", "r2", scale=bad_scale)


def test_transfer_demonstration_finite_scale_still_works_regression():
    out = transfer_demonstration(_record(), "a2", "r2", scale=2.0)
    assert out["transform"]["scale"] == pytest.approx(2.0)
    assert all(math.isfinite(v) for fr in out["frames"]
               for ents in [fr["entities"].values()] for vec in ents for v in vec)
