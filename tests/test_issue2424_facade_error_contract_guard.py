# -*- coding: utf-8 -*-
"""test_issue2424_facade_error_contract_guard —— 四门面错误契约守卫。

缺陷（dsh-bounty #2424）
============================================================================
core.py 的 12 个 action 门面是同一份复制粘贴模板，错误契约却分两派：
  · 已收口：world3d / wm_simloop（#691 Z3+#2412）/ seven_layer_loop /
    world_process 第 2 组 / spacetime_consistency（#2423）；
  · 裸穿：voxel_world（simulate steps=nan 裸崩）、scene_simulator（create/
    entity/path/step）、world_learner（create/entity/path/run）、
    curiosity_explorer（create/entity/path——#2420 只闸 explore/step）。
坏输入的不可读异常直达模型驱动面（实测 `invalid literal for int()…`、
`could not convert float NaN to integer`、`pos 需为 (x,y,z) 有限坐标三元组…`
三种形态全部裸穿），与已收口门面行为分叉。

实现侧口径（本守卫的判据）
--------------------------------------------------------------------------
四门面内所有调用下游闸/裸 int()/float() 的动作各包 try/except
(ValueError, TypeError) ⇒ error dict（同 #691 Z3 / #2423 惯例）；voxel
simulate 另加 steps 非 int 预闸（nan 裸崩族）。合法输入行为逐位不变。

守卫用例（每门面至少一坏一好）
--------------------------------------------------------------------------
G1 voxel_world：simulate(steps=nan) ⇒ error；spawn 坏 pos ⇒ error；合法 build ⇒ ok。
G2 scene_simulator：create(trees="x")/entity 坏 pos/path 坏 points/step(n="x")
   ⇒ error（step 的 n 闸在 #2409 已落地，门面转 error）；合法 entity ⇒ ok。
G3 world_learner：run(n="x")/entity 坏 pos ⇒ error；合法 create ⇒ ok。
G4 curiosity_explorer：create(trees="x")/path 坏 points ⇒ error（#2420 已闸
   explore/step 不重复）；合法 entity ⇒ ok。
G5 文案可读：error 为下游闸中文原因或门面预闸原因（非实现层裸文本裸算式）。

运行（lingshu 仓根）：
    python -X utf8 -m pytest tests/test_issue2424_facade_error_contract_guard.py -q
"""
from __future__ import annotations

import pytest

from lingshu.core.core import SpacetimeMemoryEngine


def _e():
    return SpacetimeMemoryEngine(":memory:")


# ---------------------------------------------------------------- G1

def test_2424_voxel_simulate_nan_rejected():
    e = _e()
    r = e.voxel_world("simulate", {"steps": float("nan")})
    assert r.get("status") == "error" and "steps" in r.get("error", ""), (
        "#2424 voxel simulate steps=nan 仍裸穿：%r" % (r,))


def test_2424_voxel_spawn_bad_pos_rejected():
    e = _e()
    r = e.voxel_world("spawn", {"category": "a", "pos": ["x", 1, 0]})
    assert r.get("status") == "error", r
    r = e.voxel_world("build", {"trees": "x"})
    assert r.get("status") == "error", r
    r = e.voxel_world("build", {})
    assert r.get("status") == "ok", r


# ---------------------------------------------------------------- G2

def test_2424_scene_facade_bad_inputs_error_dict():
    e = _e()
    assert e.scene_simulator("create", {"trees": "x"}).get("status") == "error"
    assert e.scene_simulator("entity", {"category": "p", "pos": [1, 2]}).get("status") == "error"
    assert e.scene_simulator("path", {"path_id": "p", "points": [[1, 2]]}).get("status") == "error"
    assert e.scene_simulator("step", {"n": "x"}).get("status") == "error"


def test_2424_scene_facade_legal_ok():
    e = _e()
    r = e.scene_simulator("create", {})
    assert r.get("status") == "ok", r
    r = e.scene_simulator("entity", {"category": "a"})
    assert r.get("status") == "ok", r
    r = e.scene_simulator("step", {"n": 2})
    assert r.get("status") == "ok", r


# ---------------------------------------------------------------- G3

def test_2424_wl_facade_bad_inputs_error_dict():
    e = _e()
    assert e.world_learner("init", {})["status"] == "ok"
    assert e.world_learner("create", {"trees": "x"}).get("status") == "error"
    assert e.world_learner("entity", {"category": "p", "pos": [1, 2]}).get("status") == "error"
    assert e.world_learner("path", {"path_id": "p", "points": [[1, 2]]}).get("status") == "error"
    assert e.world_learner("run", {"n": "x"}).get("status") == "error"


def test_2424_wl_facade_legal_ok():
    e = _e()
    assert e.world_learner("init", {})["status"] == "ok"
    assert e.world_learner("create", {})["status"] == "ok"
    assert e.world_learner("entity", {"category": "a"})["status"] == "ok"
    assert e.world_learner("run", {"n": 2})["status"] == "ok"


# ---------------------------------------------------------------- G4

def test_2424_ce_facade_bad_inputs_error_dict():
    e = _e()
    assert e.curiosity_explorer("create", {"trees": "x"}).get("status") == "error"
    assert e.curiosity_explorer("entity", {"category": "p", "pos": [1, 2]}).get("status") == "error"
    assert e.curiosity_explorer("path", {"path_id": "p", "points": [[1, 2]]}).get("status") == "error"


def test_2424_ce_facade_legal_ok():
    e = _e()
    assert e.curiosity_explorer("create", {})["status"] == "ok"
    assert e.curiosity_explorer("entity", {"category": "a"})["status"] == "ok"


# ---------------------------------------------------------------- G5

def test_2424_errors_are_readable_not_raw_traceback_text():
    e = _e()
    r = e.scene_simulator("entity", {"category": "p", "pos": [1, 2]})
    msg = r.get("error", "")
    assert "三元组" in msg, "应为下游闸的可读中文原因：%r" % (msg,)
    r2 = e.voxel_world("simulate", {"steps": float("nan")})
    assert "必须是整数" in r2.get("error", ""), r2


# ---------------------------------------------------------------- G8（#2424 续·扫尾 7 站点）

def test_2424_wl_predict_horizon_rejected():
    e = _e()
    e.world_learner("init", {}); e.world_learner("create", {})
    e.world_learner("entity", {"category": "a"})
    r = e.world_learner("predict", {"horizon": float("nan")})
    assert r.get("status") == "error", "#2424续 predict horizon=nan 仍裸穿：%r" % (r,)


def test_2424_wl_evaluate_train_ticks_rejected():
    e = _e()
    e.world_learner("init", {}); e.world_learner("create", {})
    r = e.world_learner("evaluate", {"train_ticks": "x"})
    assert r.get("status") == "error", r


def test_2424_wl_curve_epochs_rejected():
    e = _e()
    e.world_learner("init", {}); e.world_learner("create", {})
    r = e.world_learner("curve", {"epochs": float("nan")})
    assert r.get("status") == "error", r


def test_2424_wm_run_n_rejected():
    e = _e()
    e.world_model("init", {}); e.world_model("create", {})
    r = e.world_model("run", {"n": "x"})
    assert r.get("status") == "error", "#2424续 wm run n=\"x\" 仍裸穿：%r" % (r,)


def test_2424_wm_counterfactual_bad_horizon_error_dict():
    e = _e()
    e.world_model("init", {}); e.world_model("create", {})
    r = e.world_model("counterfactual", {"horizon": 0})
    assert r.get("status") == "error" and "1..64" in r.get("error", ""), (
        "#2424续 counterfactual horizon=0 的既有闸 ValueError 改为可读 error dict：%r" % (r,))


def test_2424_wm_history_limit_rejected():
    e = _e()
    e.world_model("init", {}); e.world_model("create", {})
    r = e.world_model("history", {"limit": "x"})
    assert r.get("status") == "error", "#2424续 wm history limit=\"x\" 仍裸穿：%r" % (r,)


def test_2424_wm_history_reachable():
    """附修复：world_model 的 history 动作此前被 graph 分支 return 短路成
    不可达死代码（docstring 与未知动作提示均声明可用，实测恒 未知动作）。
    恢复后合法调用返回 ok。"""
    e = _e()
    e.world_model("init", {}); e.world_model("create", {})
    r = e.world_model("history", {})
    assert r.get("status") == "ok" and "history" in r, r


# ---------------------------------------------------------------- G9（#2424 续·perceive_visual 同契约）

def test_2424_perceive_visual_bad_tick_error_dict():
    """perceive_visual 与 perceive 同契约：tick 闸的可读 ValueError 在门面
    转 error dict（修前裸穿实测，两处——observations 读取与 perceive 步骤）。"""
    e = _e()
    e.world3d("add", {"category": "box", "bbox": [380, 280, 420, 320]})
    e.world_model("init", {}); e.world_model("create", {})
    r = e.world_model("perceive_visual", {"tick": "x"})
    assert r.get("status") == "error" and "tick" in r.get("error", ""), (
        "#2424续 perceive_visual tick=\"x\" 仍裸穿：%r" % (r,))


def test_2424_perceive_visual_legal_ok():
    e = _e()
    e.world3d("add", {"category": "box", "bbox": [380, 280, 420, 320]})
    r = e.world_model("perceive_visual", {})
    assert r.get("status") == "ok", r
