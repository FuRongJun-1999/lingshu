# -*- coding: utf-8 -*-
"""test_issue159_predict_horizon · 回归守卫：horizon 必须真正驱动预测/生成步数
============================================================================
缺陷（本组缺陷单 #159 · world/world_learner.py predict(horizon) /
world/world_model.py generate(horizon)）：两处签名都带 horizon 参数、返回值也
回填 horizon，但函数体从不解引用它——predict(horizon=N)/generate(horizon=N)
与 horizon=1 逐位相同。「N 步后的预测」这个公开契约（core.py 的 world_model /
world_learner 端口把 horizon 直达此处）根本不成立。

守卫断言组：
  A 组 WorldLearner.predict：确定性实体随步数前推（h=3 比 h=1 沿运动方向恰多
    走 (3-1) 步），且 horizon 字段 = 请求步数、步数越大越远。
  B 组 UnifiedWorldModel.generate：同型断言。
  C 组 反例（防过度修复）：随机/可达域实体（预测=当前位置）不因步数改变。
  D 组 兼容：h=1 仍等于「当前位置 + 1 步」（既有单步调用行为不变）。

运行（lingshu 仓根）：
    python -X utf8 tests/test_issue159_predict_horizon.py     # 脚本式，退出码 0/1
    python -X utf8 -m pytest tests/test_issue159_predict_horizon.py
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.world.scene_simulator import SceneSimulator       # noqa: E402
from lingshu.world.world_learner import WorldLearner           # noqa: E402
from lingshu.world.world_model import UnifiedWorldModel        # noqa: E402

_PASS, _FAIL = [], []


def ok(name, cond, detail=""):
    (_PASS if cond else _FAIL).append(name)
    print(("  PASS " if cond else "  FAIL ") + name
          + (("  <- " + str(detail)) if detail else ""))


# ---------- 确定性直线运动场景（follow 巡逻，无关系 → persistence=1.0） ----------

def _add_runner(sc):
    sc.create_scene(trees=0, water=False)
    eid = sc.add_entity("runner", behavior="follow", pos=(2, 1.5, 2),
                        speed=1.0, goal="p1")
    sc.add_path("p1", [(2, 1.5, 2), (6, 1.5, 2), (10, 1.5, 2),
                       (14, 1.5, 2), (18, 1.5, 2)])
    return eid


def _learner():
    sc = SceneSimulator(size=24, seed=42)
    eid = _add_runner(sc)
    lw = WorldLearner(world=sc, size=24)
    lw.run(n=4)
    lw.learn()
    return lw, eid


def _wmodel():
    sc = SceneSimulator(size=24, seed=42)
    eid = _add_runner(sc)
    wm = UnifiedWorldModel(world=sc, size=24)
    for _ in range(4):
        sc.step(n=1)
        wm.perceive()
    return wm, eid


def _stationary_learner():
    """speed=0 的 wander 实体：位移恒为零 → 可达域（bounded_stochastic）。"""
    sc = SceneSimulator(size=24, seed=7)
    sc.create_scene(trees=0, water=False)
    eid = sc.add_entity("rock", behavior="wander", pos=(5, 1.5, 5), speed=0.0)
    lw = WorldLearner(world=sc, size=24)
    lw.run(n=6)
    lw.learn()
    return lw, eid


def _stationary_wmodel():
    sc = SceneSimulator(size=24, seed=7)
    sc.create_scene(trees=0, water=False)
    eid = sc.add_entity("rock", behavior="wander", pos=(5, 1.5, 5), speed=0.0)
    wm = UnifiedWorldModel(world=sc, size=24)
    for _ in range(6):
        sc.step(n=1)
        wm.perceive()
    wm.infer_patterns(force=True)
    return wm, eid


# ---------- A 组：WorldLearner.predict ----------

def group_a_learner_horizon():
    lw, eid = _learner()
    per = lw.model["per_entity"][eid]
    ok("A0 复现前提：实体确定性直线运动（persistence=1.0）",
       per["persistence"] == 1.0, per)
    p1 = lw.predict(horizon=1)["predictions"][eid]
    p3 = lw.predict(horizon=3)["predictions"][eid]
    p5 = lw.predict(horizon=5)["predictions"][eid]
    print("     reading:", p1["predicted"], p3["predicted"], p5["predicted"])
    ok("A1 horizon=3 与 horizon=1 的预测不同（本条回归点）",
       p1["predicted"] != p3["predicted"],
       "%r == %r" % (p1["predicted"], p3["predicted"]))
    dx = p3["predicted"][0] - p1["predicted"][0]
    ok("A2 horizon=3 恰好多前推 2 步（Δx≈(3-1)*speed=2.0）",
       abs(dx - 2.0) <= 0.15, "Δx=%r" % dx)
    ok("A3 步数越大越远（单调）",
       p5["predicted"][0] > p3["predicted"][0] > p1["predicted"][0],
       (p1["predicted"][0], p3["predicted"][0], p5["predicted"][0]))
    ok("A4 返回 horizon 字段 = 请求步数",
       lw.predict(horizon=5)["horizon"] == 5)


# ---------- B 组：UnifiedWorldModel.generate ----------

def group_b_wmodel_horizon():
    wm, eid = _wmodel()
    g1 = wm.generate(horizon=1)["predictions"][eid]
    g3 = wm.generate(horizon=3)["predictions"][eid]
    print("     reading:", g1["predicted"], g3["predicted"])
    ok("B0 复现前提：实体走确定性直线外推（bounded_noisy）",
       g1["mode"] == "bounded_noisy", g1["mode"])
    ok("B1 generate(horizon=3) 与 horizon=1 的候选不同（本条回归点）",
       g1["predicted"] != g3["predicted"],
       "%r == %r" % (g1["predicted"], g3["predicted"]))
    dx = g3["predicted"][0] - g1["predicted"][0]
    ok("B2 horizon=3 恰好多外推 2 步（Δx≈2.0）", abs(dx - 2.0) <= 0.15, "Δx=%r" % dx)
    ok("B3 返回 horizon 字段 = 请求步数",
       wm.generate(horizon=5)["horizon"] == 5)


# ---------- C 组：可达域实体不因步数改变（防过度修复） ----------

def group_c_stochastic_unchanged():
    lw, eid = _stationary_learner()
    m1 = lw.predict(horizon=1)["predictions"][eid]
    m5 = lw.predict(horizon=5)["predictions"][eid]
    print("     learner reading:", m1["mode"], m1["predicted"], m1["bound"])
    ok("C1 复现前提：静止实体走可达域模式 bounded_stochastic",
       m1["mode"] == "bounded_stochastic", m1["mode"])
    ok("C2 可达域实体预测/边界不随步数改变（预测=当前位置）",
       m1["predicted"] == m5["predicted"] and m1["bound"] == m5["bound"],
       "%r vs %r" % (m1, m5))

    wm, eid2 = _stationary_wmodel()
    w1 = wm.generate(horizon=1)["predictions"][eid2]
    w5 = wm.generate(horizon=5)["predictions"][eid2]
    print("     model reading:", w1["mode"], w1["predicted"], w1["bound"])
    ok("C3 world_model 同型：可达域候选不随步数改变",
       w1["mode"] == "bounded_stochastic"
       and w1["predicted"] == w5["predicted"] and w1["bound"] == w5["bound"],
       "%r vs %r" % (w1, w5))


# ---------- D 组：h=1 兼容（单步预测 = 当前位置 + 1 步） ----------

def group_d_single_step_unchanged():
    lw, eid = _learner()
    cur = tuple(lw.nodes[eid].pos)
    p1 = lw.predict(horizon=1)["predictions"][eid]
    print("     reading: cur=%r h1=%r" % (cur, p1["predicted"]))
    ok("D1 h=1 预测 = 当前位置沿运动方向 1 步（既有单步口径不变）",
       abs(p1["predicted"][0] - (cur[0] + 1.0)) <= 0.02
       and p1["predicted"][2] == cur[2],
       "cur=%r h1=%r" % (cur, p1["predicted"]))


def run_checks():
    _PASS.clear(); _FAIL.clear()
    print("== A 组：WorldLearner.predict 随 horizon 前推 ==")
    group_a_learner_horizon()
    print("== B 组：UnifiedWorldModel.generate 随 horizon 外推 ==")
    group_b_wmodel_horizon()
    print("== C 组：可达域实体不随步数改变 ==")
    group_c_stochastic_unchanged()
    print("== D 组：h=1 兼容 ==")
    group_d_single_step_unchanged()
    print()
    print("===== SUMMARY %d/%d 通过 =====" % (len(_PASS), len(_PASS) + len(_FAIL)))
    if _FAIL:
        print("FAILS:", _FAIL)
        return 1
    print("VERDICT=PASS（horizon 真正驱动预测/生成步数；随机实体不被误推）")
    return 0


def test_issue159_predict_horizon():
    """pytest 入口（脚本/pytest 双模式）。"""
    run_checks()
    assert not _FAIL, f"{len(_FAIL)} checks failed: {_FAIL}"


if __name__ == "__main__":
    sys.exit(run_checks())
