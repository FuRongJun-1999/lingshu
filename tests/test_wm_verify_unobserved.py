# -*- coding: utf-8 -*-
"""test_wm_verify_unobserved · issue #1 守卫：verify() 未观测预测不得计为命中
============================================================================
缺陷（lingshu issue #1 · 报告人 rinDBeans）：verify() 的 actual 来源曾是模型
自己的节点（`obs = {eid: list(n.pos) for eid, n in self.nodes.items()}`）——
未观测的拓扑假设拿自己的位置当「实际观测」⇒ distance=0、hit=True 自证命中，
污染 hit_rate。修法：verify() 只以最近一次 perceive() 留存的本轮真实观测快照
（`self._obs_snapshot`）为 actual 来源；快照里没有的预测标 pending
（只进 details、不计入 hits/total）；快照缺失 ⇒ total=0 + no_observation，
不回退到「从 nodes 重建」。

断言组：
  A 组（报告人场景）：真实 actor 未命中照旧计分；假设项为 pending 而非 hit；
      hits/total 只计已验证（total=1, hits=0, hit_rate=0.0）
  B 组（坑 b）：此前见过、后来被遮蔽的真实实体 → 预测 pending、不参与计分
  C 组：假设与陈旧记忆仍在 nodes/conditions/edges（不删不改）
  D 组（兼容性）：从未 perceive ⇒ total=0 + no_observation；空观测轮 ⇒ 全 pending

运行（lingshu 仓根）：python -X utf8 tests/test_wm_verify_unobserved.py
退出码：0 = 全过；1 = 有断言失败
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.world.scene_simulator import SceneSimulator  # noqa: E402
from lingshu.world.world_model import UnifiedWorldModel  # noqa: E402
from lingshu.world.wm_simloop import SimulationLoop  # noqa: E402

_PASS = []
_FAIL = []


def ok(cond, msg, extra=""):
    (_PASS if cond else _FAIL).append(msg)
    print(("  PASS " if cond else "  FAIL ") + msg
          + (("  ← " + str(extra)) if (extra and not cond) else ""))


def detail_of(result, eid):
    return next(d for d in result["details"] if d["entity"] == eid)


# ---------- A 组：报告人场景（逐字照抄复现步骤，只调公开 API） ----------

def group_a_reporter_scenario():
    scene = SceneSimulator(size=24)
    actor = scene.add_entity("actor", pos=(2, 1.5, 2), speed=0)
    wm = UnifiedWorldModel(world=scene)
    wm.perceive()
    loop = SimulationLoop(world_model=wm, scene=scene, clock=lambda: 0)

    def motion(world, tick):
        world.entities[actor].pos = (
            2 + tick * 0.25, 1.5, 2 + (tick % 2) * 0.35)

    loop.step(n=4, external=motion)
    hypotheses = [eid for eid, n in wm.nodes.items()
                  if n.attrs.get("hypothesis")]
    ok(len(hypotheses) == 1, "A1 真实观测自然触发生长（假设数=1）", hypotheses)
    if not hypotheses:
        return
    hypothesis = hypotheses[0]
    ok(hypothesis not in scene.entities, "A2 假设不是物理实体（从未被观测）")

    def accelerated_motion(world, tick):
        x, y, z = world.entities[actor].pos
        world.entities[actor].pos = (x + 2, y, z)

    loop.step(n=1, external=accelerated_motion)
    result = wm.verify()

    real = detail_of(result, actor)
    ok(real.get("status") == "verified" and real["hit"] is False
       and real["distance"] == 2.0,
       "A3 真实 actor 照旧计分且未命中（hit=False distance=2.0）", real)
    syn = detail_of(result, hypothesis)
    ok(syn.get("status") == "pending" and syn.get("hit") is None
       and syn.get("distance") is None,
       "A4 未观测假设标 pending 且 hit 不为 True（不再自证命中）", syn)
    ok(result["pending"] == 1 and result["total"] == 1
       and result["hits"] == 0 and result["hit_rate"] == 0.0,
       "A5 计分只含已验证：total=1 hits=0 hit_rate=0.0（pending=1）", result)

    # C 组（同一场景）：假设与陈旧记忆仍在图中（不删不改）
    ok(hypothesis in wm.nodes and wm.nodes[hypothesis].attrs.get("hypothesis"),
       "C1 假设节点仍在 nodes 里（未被删）")
    ok(hypothesis in wm._conditions, "C2 假设的条件空间仍在 conditions 里")
    ok(any(e.source == actor and e.target == hypothesis for e in wm.edges),
       "C3 seek(A→H) 关系边仍在 edges 里")


# ---------- B 组：坑 b——此前见过、后来被遮蔽的真实实体 ----------

def group_b_stale_masked_entity():
    scene = SceneSimulator(size=24)
    a = scene.add_entity("actor", pos=(2, 1.5, 2), speed=0)
    b = scene.add_entity("rabbit", pos=(10, 1.5, 10), speed=0)
    wm = UnifiedWorldModel(world=scene)
    wm.perceive()                       # 两者都进图（b「此前见过」）
    ok(b in wm.nodes, "B1 遮蔽前真实实体已进图", sorted(wm.nodes))
    b_pos = tuple(wm.nodes[b].pos)

    loop = SimulationLoop(world_model=wm, scene=scene, clock=lambda: 0)
    loop._mask = {b}                    # b 从此被遮蔽（模型看不见）
    loop.step(n=1)                      # generate 仍为 b 出预测（陈旧模型位置）
    result = wm.verify()

    bd = detail_of(result, b)
    ok(bd.get("status") == "pending" and bd.get("hit") is None
       and bd.get("distance") is None,
       "B2 此前见过后被遮蔽者 → pending，不参与计分（坑 b）", bd)
    ok(result["total"] == 1 and result["pending"] == 1,
       "B3 计分项只含本轮真观测者",
       {"total": result["total"], "pending": result["pending"]})
    ok(b in wm.nodes and tuple(wm.nodes[b].pos) == b_pos,
       "B4 陈旧记忆留在图中且位置未被改写（不删不改）")
    ok(wm.nodes[b].last_seen < wm.tick,
       "B5 其 last_seen 停在被遮蔽前（故 last_seen==tick 不可作判据）",
       wm.nodes[b].last_seen)
    ok(detail_of(result, a)["hit"] is True,
       "B6 对照组：仍被观测者（a）照旧计分命中", detail_of(result, a))


# ---------- D 组：兼容性（快照缺失 / 空观测轮） ----------

def group_d_snapshot_missing():
    scene = SceneSimulator(size=24)
    scene.add_entity("actor", pos=(2, 1.5, 2), speed=0)
    wm = UnifiedWorldModel(world=scene)
    r0 = wm.verify()                    # 从未 perceive、无预测
    ok(r0["total"] == 0 and r0.get("no_observation") is True,
       "D1 从未 perceive：total=0 + no_observation（不回退 nodes）", r0)

    wm.generate()                       # 无节点 ⇒ 无预测
    r1 = wm.verify()
    ok(r1["total"] == 0 and r1.get("no_observation") is True
       and r1["details"] == [],
       "D2 无节点无预测：total=0 + no_observation + 空 details", r1)

    wm2 = UnifiedWorldModel(world=SceneSimulator(size=24))
    wm2.world.add_entity("actor", pos=(2, 1.5, 2), speed=0)
    wm2.perceive()                      # 有节点、有快照
    wm2.generate()
    wm2.perceive(observations=[])       # 空观测轮：本轮什么都没看见
    r2 = wm2.verify()
    ok(r2["total"] == 0 and r2["pending"] == len(r2["details"])
       and "no_observation" not in r2,
       "D3 空观测轮：全部 pending、total=0、快照在场（不标 no_observation）", r2)


def main():
    print("== A 组：报告人场景（未观测假设不计分）+ C 组（假设留图） ==")
    group_a_reporter_scenario()
    print("== B 组：坑 b（此前见过、后来被遮蔽的真实实体） ==")
    group_b_stale_masked_entity()
    print("== D 组：兼容性（快照缺失 / 空观测轮） ==")
    group_d_snapshot_missing()
    print()
    print(f"===== SUMMARY {len(_PASS)}/{len(_PASS) + len(_FAIL)} 通过 =====")
    if _FAIL:
        print("FAILS:", _FAIL)
        return 1
    print("VERDICT=PASS（issue #1 守卫：未观测预测不计分；假设/陈旧记忆留图）")
    return 0


if __name__ == "__main__":
    sys.exit(main())


def test_wm_verify_unobserved_guard():
    """issue #1 守卫的 pytest 入口（A–D 断言组全跑；失败经 ok()/main() 计入退出码 1）。"""
    assert main() == 0
