# -*- coding: utf-8 -*-
"""test_world_learner_oracle_rate · 回归守卫：oracle 命中率必须用 oracle 自己的覆盖面
============================================================================
缺陷（基线 421deb4 · world_learner.py:371-379）：oracle（真模型上界）的命中数
和 learned/naive 的命中数共用**同一个分母** total，而 total 只在学得模型给出
预测的实体上累加（:355）。学习者覆盖面 < 世界实体数时，oracle_rate 可以 > 1，
gap_to_oracle 相应 > 1。

断言组：
  A 组：学习者只见 1 个实体、世界有 6 个 ⇒ oracle_rate 不得 > 1（本条回归点）
  B 组：oracle 命中数用独立分母（oracle_outcomes 与 outcomes 各自成对）
  C 组：_oracle_predict() 静默失败（返回 {}）⇒ 不得报 gap_to_oracle == 0
        "已追平上界"；须显式标 oracle_unavailable
  D 组（兼容性）：对等场景（学习者认识全部实体）三类率仍在 [0,1] 且 gap>=0

运行（lingshu 仓根）——脚本/pytest 双模式：
    python -X utf8 tests/test_world_learner_oracle_rate.py     # 脚本式，退出码 0/1
    python -X utf8 -m pytest tests/test_world_learner_oracle_rate.py
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.world.scene_simulator import SceneSimulator   # noqa: E402
from lingshu.world.world_learner import WorldLearner       # noqa: E402

_PASS, _FAIL = [], []


def ok(name, cond, detail=""):
    (_PASS if cond else _FAIL).append(name)
    print(("  PASS " if cond else "  FAIL ") + name
          + (("  <- " + str(detail)) if detail else ""))


def _learner_sees_one_of_six():
    """学习者只观测 1 个实体；世界里随后再有 5 个（只用公开 API）。"""
    sc = SceneSimulator(size=24)
    sc.create_scene(trees=0, water=False)
    sc.add_entity("a", behavior="wander", pos=(2, 1.5, 2), speed=0.5)
    lw = WorldLearner(world=sc, size=24)
    lw.observe()                    # 此刻只认识 a
    for i in range(5):
        sc.add_entity("s%d" % i, behavior="wander",
                      pos=(10 + i, 1.5, 10), speed=0.2)
    return lw


# ---------- A 组：oracle 率不得越界（本条回归点） ----------

def group_a_oracle_rate_bounded():
    lw = _learner_sees_one_of_six()
    r = lw.eval_phase(eval_ticks=1)
    print("     reading:", {k: r.get(k) for k in
                            ("outcomes", "oracle_outcomes", "learned_rate",
                             "naive_rate", "oracle_rate", "gap_to_oracle")})
    ok("A1 学习者覆盖面 1 < 世界实体 6（复现前提）",
       r["outcomes"] == 1, r["outcomes"])
    ok("A2 oracle_rate 不得 > 1.0（本条回归点）",
       r.get("oracle_rate") is None or r["oracle_rate"] <= 1.0,
       "oracle_rate=%r" % r.get("oracle_rate"))


# ---------- B 组：分母独立 ----------

def group_b_independent_denominators():
    lw = _learner_sees_one_of_six()
    r = lw.eval_phase(eval_ticks=1)
    ok("B1 记录里带 oracle 自己的分母 oracle_outcomes",
       "oracle_outcomes" in r, sorted(r))
    ok("B2 oracle_outcomes >= oracle 命中实体数、且与 outcomes 独立",
       r.get("oracle_outcomes", 0) >= 1 and r.get("oracle_outcomes") != r.get("outcomes"),
       "%r vs %r" % (r.get("oracle_outcomes"), r.get("outcomes")))


# ---------- C 组：oracle 静默失败不得报"无差距" ----------

def group_c_oracle_silent_failure():
    lw = _learner_sees_one_of_six()
    lw.run(n=3)
    lw.learn()
    lw._oracle_predict = lambda: {}      # 模拟 _oracle_predict 的内部 except: return {}
    r = lw.eval_phase(eval_ticks=1)
    print("     reading:", {k: r.get(k) for k in
                            ("outcomes", "oracle_rate", "oracle_unavailable",
                             "gap_to_oracle")})
    ok("C1 oracle 不可用时显式置 oracle_unavailable=True",
       r.get("oracle_unavailable") is True, r.get("oracle_unavailable"))
    ok("C2 oracle 不可用时 gap_to_oracle 不得为 0.0（伪装成已追平）",
       r.get("gap_to_oracle") != 0.0, "gap=%r" % r.get("gap_to_oracle"))
    ok("C3 oracle 不可用时 oracle_rate 不得为 0.0（伪装成上界全错）",
       r.get("oracle_rate") != 0.0, "oracle_rate=%r" % r.get("oracle_rate"))


# ---------- D 组：对等场景不退化 ----------

def group_d_parity_unchanged():
    sc = SceneSimulator(size=24)
    sc.create_scene(trees=0, water=False)
    sc.add_entity("a", behavior="wander", pos=(2, 1.5, 2), speed=0.5)
    sc.add_entity("b", behavior="wander", pos=(8, 1.5, 8), speed=0.3)
    lw = WorldLearner(world=sc, size=24)
    lw.run(n=8)
    lw.learn()
    r = lw.eval_phase(eval_ticks=3)
    print("     reading:", {k: r.get(k) for k in
                            ("outcomes", "oracle_outcomes", "learned_rate",
                             "naive_rate", "oracle_rate", "gap_to_oracle")})
    for k in ("learned_rate", "naive_rate", "oracle_rate"):
        v = r.get(k)
        ok("D1 %s 在 [0,1]" % k, v is None or 0.0 <= v <= 1.0, "%s=%r" % (k, v))
    g = r.get("gap_to_oracle")
    ok("D2 gap_to_oracle >= 0", g is None or g >= 0.0, "gap=%r" % g)
    ok("D3 对等场景 oracle 可用（oracle_unavailable=False）",
       r.get("oracle_unavailable") is False, r.get("oracle_unavailable"))
    # 兼容性：原有公开字段仍在
    for k in ("tick", "eval_ticks", "outcomes", "learned_rate", "naive_rate",
              "oracle_rate", "gap_to_oracle"):
        ok("D4 公开字段保留 %s" % k, k in r)


def run_checks():
    _PASS.clear(); _FAIL.clear()
    print("== A 组：oracle_rate 不得 > 1 ==")
    group_a_oracle_rate_bounded()
    print("== B 组：分母独立 ==")
    group_b_independent_denominators()
    print("== C 组：oracle 静默失败不得报 gap=0 ==")
    group_c_oracle_silent_failure()
    print("== D 组：对等场景兼容性 ==")
    group_d_parity_unchanged()
    print()
    print("===== SUMMARY %d/%d 通过 =====" % (len(_PASS), len(_PASS) + len(_FAIL)))
    if _FAIL:
        print("FAILS:", _FAIL)
        return 1
    print("VERDICT=PASS（oracle 命中率用各自覆盖面；oracle 不可用须显式标记）")
    return 0


def test_world_learner_oracle_rate():
    """pytest 入口（与 tests/test_hex_composite.py 同惯例：脚本/pytest 双模式）。"""
    run_checks()
    assert not _FAIL, f"{len(_FAIL)} checks failed: {_FAIL}"


if __name__ == "__main__":
    sys.exit(run_checks())
