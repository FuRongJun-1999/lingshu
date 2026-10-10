"""wm_simloop 缺陷修复守卫（P1 批次：#186 / #192 / #323 / #399）。

每个测试钉住一个缺陷的**不存在**（而非「代码能跑」）：
  - #186 _ray_intersect_2d 必须拒绝「只在一条射线前方」的直线交点
  - #192 生长通道②的持续性阈值必须落在随机游走 null 分布之上
  - #323 假设节点摘除后不得残留指向它的边（快照/WAL 亦不留）
  - #399 flush_payloads → parse_seed_entity 往返不得丢字段/错位

判据来源：见各测试 docstring（经验标定 = 本文件所跑命令的实测读数）。
"""
from __future__ import annotations

import math

from lingshu.world.wm_simloop import SimulationLoop, _ray_intersect_2d


# ==================== #186 ====================

def test_ray_intersect_rejects_intersection_behind_second_ray():
    """#186：交点须同时在两条射线前方。

    q1=(0,0) 沿 +x；q2=(2,1) 沿 +y。两直线交于 (2,0)：在射线 1 前方
    （s=2），却在射线 2 后方（t=-1）。旧实现只查 s ⇒ 误报交点。
    """
    assert _ray_intersect_2d((0.0, 0.0), (1.0, 0.0), (2.0, 1.0), (0.0, 1.0)) is None
    # 对称：在射线 1 后方（s=-2）、射线 2 前方（t=1）⇒ 同样无效
    assert _ray_intersect_2d((0.0, 0.0), (-1.0, 0.0), (2.0, 1.0), (0.0, -1.0)) is None


def test_ray_intersect_accepts_forward_intersection():
    """#186 正例：两条射线正向前方相交 → 返回交点（修复不得把有效交点也删掉）。"""
    assert _ray_intersect_2d((0.0, 0.0), (1.0, 0.0), (2.0, 1.0), (0.0, -1.0)) == (2.0, 0.0)


# ==================== #192 ====================

def _random_walk_loop(seed: int) -> SimulationLoop:
    loop = SimulationLoop(seed=seed)
    loop.scene.create_scene(trees=0, water=False)
    loop.scene.add_entity("rabbit", behavior="wander", pos=(6, 1.5, 13), speed=0.3)
    loop.wm.perceive(observations=loop._observe())
    return loop


def test_persistence_requires_full_window():
    """#192：不足整窗观测 ⇒ 不给持续性估计（旧实现 len<4 即给值，小样本尾巴很宽）。

    经验标定（本仓，窗口=10）：随机游走 null 的 p99≈0.76、max≈0.92，而阈值
    0.45 落在这条尾巴内部 ⇒ 纯随机游走即可越过。
    """
    loop = _random_walk_loop(seed=0)
    eid = next(iter(loop.scene.entities))
    loop.persistence_window = 80
    for _ in range(40):
        loop.step(n=1)
    assert loop._persistence(eid) is None          # 40 < 80 ⇒ 拒绝出估计
    for _ in range(45):
        loop.step(n=1)
    assert loop._persistence(eid) is not None      # 85 ≥ 80 ⇒ 可出估计


def test_random_walker_never_triggers_channel2_growth():
    """#192：纯随机游走不得触发通道②（宽边界未解释结构）生长。

    经验标定（2026-10-10，本仓）：窗口=80 时随机游走 null 的 p99≈0.25、
    max≈0.37（84400 样本），阈值 0.45 落在 null 之上 ⇒ 60 seed×300 tick
    通道②生长 0 次。旧窗口=10 同口径实测 488 次（60/60 seed 均触发）。
    """
    channel2 = 0
    for seed in range(40):
        loop = _random_walk_loop(seed=seed)
        loop.step(n=250)
        channel2 += sum(1 for ev in loop._growth_log
                        if ev["event"] in ("growth", "re-grow")
                        and ev.get("trigger", "").startswith("stochastic_mask"))
    assert channel2 == 0, f"纯随机游走触发了 {channel2} 次通道②生长"


def test_channel2_still_detects_sustained_structure():
    """#192 正对照：修复不得把通道②的判别力削到零。

    带抖动的定向运动（每步方向 ~N(0,60°)，speed=0.12）：局部看近似随机
    （cons(8)<0.7 ⇒ 走通道②），整窗看有净漂移（per≈0.6>0.45）。W=80 时
    实测第 83 tick 触发通道②（旧 W=10 同场景第 10 tick 即触发）。
    """
    import random

    loop = SimulationLoop(seed=7)
    loop.scene.create_scene(trees=0, water=False)
    walker = loop.scene.add_entity("walker", behavior="wander",
                                   pos=(4, 1.5, 4), speed=0.0)
    loop.wm.perceive(observations=loop._observe())
    rng = random.Random(99)
    sigma = math.radians(60.0)
    pos = [4.0, 4.0]

    def jittered(scene, tick):
        th = rng.gauss(0.0, sigma)
        pos[0] = max(0.5, min(23.5, pos[0] + math.cos(th) * 0.12))
        pos[1] = max(0.5, min(23.5, pos[1] + math.sin(th) * 0.12))
        scene.entities[walker].pos = (round(pos[0], 2), 1.5, round(pos[1], 2))

    loop.step(n=150, external=jittered)
    ch2 = [ev for ev in loop._growth_log
           if ev["event"] in ("growth", "re-grow")
           and ev.get("trigger", "").startswith("stochastic_mask")]
    assert ch2, "真实持续结构未触发通道②（判别力被削到零）"


def test_channel2_confirms_a_sustained_chase():
    """#192 正对照之二：真实持续结构应能走完通道② → 验证窗口 → 固化。

    遮蔽目标沿 +x 慢漂移（0.3/tick）叠加正弦，追逐者 speed=0.5：模型按
    随机行为预测（cons<0.7）但整窗持续性 per≈0.63>0.45 ⇒ 通道②生长，
    并在验证窗口内趋向占比达标 → confirm。场地 48 是为容纳整窗 80 tick
    的持续移动（24 场地会先顶到边界）。
    """
    loop = SimulationLoop(size=48)
    loop.scene.create_scene(trees=0, water=False)
    target = loop.scene.add_entity("rabbit", behavior="wander",
                                   pos=(6, 1.5, 13), speed=0.0)
    loop.scene.add_entity("wolf", behavior="seek", goal=target,
                          pos=(2, 1.5, 13), speed=0.5)
    loop._mask = {target}
    loop.wm.perceive(observations=loop._observe())

    def drifting(scene, tick):
        scene.entities[target].pos = (3.0 + 0.3 * tick
                                      + 2.5 * math.sin(tick * 0.45), 1.5, 13.0)

    loop.step(n=140, external=drifting)
    ch2 = [ev for ev in loop._growth_log
           if ev["event"] in ("growth", "re-grow")
           and ev.get("trigger", "").startswith("stochastic_mask")]
    assert ch2, "真实持续结构未触发通道②"
    assert any(ev["event"] == "confirm" for ev in loop._growth_log), \
        "持续结构应能通过验证窗口固化"


# ==================== #323 ====================

def _dangling(loop):
    """源或目标已不在 nodes 的边（悬空边）。"""
    ids = set(loop.wm.nodes)
    return [(e.source, e.relation, e.target) for e in loop.wm.edges
            if e.source not in ids or e.target not in ids]


def _loop_with_hypothesis(seed=1):
    from lingshu.world.world_model import WMEdge  # noqa: F401  (供调用方构造边)

    loop = SimulationLoop(seed=seed)
    loop.scene.create_scene(trees=0, water=False)
    subject = loop.scene.add_entity("actor", pos=(2, 1.5, 2), speed=0.0)
    loop.wm.perceive(observations=loop._observe())
    loop._grow(subject, (20.0, 1.5, 2.0), 0.9, trigger="t1")
    return loop, subject, loop._hypotheses[subject]["node"]


def test_revert_removes_hypothesis_outgoing_edges():
    """#323：摘除假设节点后，与该假设关联的边（入边 + 出边）必须一并清除。

    旧实现只删 `target == hid`，假设节点自身的出边残留为悬空边（源已不在
    nodes），graph()/WAL 快照仍会导出它。出边的真实来源：`world_model.
    infer_patterns` 把假设节点当普通节点推关系边（HEAD 的 perceive 把全部
    节点写进 history ⇒ 假设节点有位移轨迹，遂被推出一条出边）。本守卫直接
    构造该形态的出边（不依赖 world_model 的并发改动），断言摘除后无悬空边。
    """
    from lingshu.world.world_model import WMEdge

    loop, subject, hid = _loop_with_hypothesis()
    loop.wm.edges.append(WMEdge(source=hid, relation="flee", target=subject,
                                confidence=0.5, evidence="inferred"))
    assert any(e.source == hid for e in loop.wm.edges)

    loop._revert(loop._hypotheses[subject], "test")
    assert hid not in loop.wm.nodes
    assert _dangling(loop) == []
    live = set(loop.wm.graph()["nodes"])
    assert all(e["source"] in live and e["target"] in live
               for e in loop._snapshot_edges())


def test_regrow_removes_old_hypothesis_outgoing_edges():
    """#323 re-grow 分支：重新生长时摘除旧假设，旧假设的出边不得残留。"""
    from lingshu.world.world_model import WMEdge

    loop, subject, old = _loop_with_hypothesis()
    loop.wm.edges.append(WMEdge(source=old, relation="flee", target=subject,
                                confidence=0.5, evidence="inferred"))
    loop.wm.tick += 1                     # 使新假设 eid 与旧的区分
    loop._grow(subject, (21.0, 1.5, 2.0), 0.9, trigger="t2")   # re-grow
    assert old not in loop.wm.nodes
    assert _dangling(loop) == []


# ==================== #399 ====================

def test_parse_seed_entity_handles_multiword_category():
    """#399：类别含空格时不得被截成首词。

    缺陷形态：`category = seg.split()[0]` 把 `large cart` 解析成 `large`。
    """
    line = "- entity: large cart id=scene_ab12 pos=(2.0, 1.5, 2.0) confidence=0.5"
    got = SimulationLoop.parse_seed_entity(line)
    assert got == {"category": "large cart", "eid": "scene_ab12",
                   "pos": (2.0, 1.5, 2.0)}
    # 负坐标 / 单字类别 / 非实体行 亦须正确
    assert SimulationLoop.parse_seed_entity(
        "- entity: rabbit id=x1 pos=(-2.5, 1.5, 2.0) confidence=0.5"
    )["pos"] == (-2.5, 1.5, 2.0)
    assert SimulationLoop.parse_seed_entity("- hypothesis: hidden_target id=h1") is None


def test_flush_then_parse_round_trips_entity_fields():
    """#399：flush 写出的实体行必须能被 parse_seed_entity 无损读回。

    flush 写出的类别是裸文本（无引号无转义），类别可含空格。本守卫用多词
    类别跑真实 flush → parse 往返，逐字段比对；旧实现把 `large cart` 读成
    `large`（写读不对称 = 持久化往返损坏）。
    """
    loop = SimulationLoop(seed=2)
    loop.scene.create_scene(trees=0, water=False)
    loop.scene.add_entity("large cart", pos=(2, 1.5, 2), speed=0.0)
    loop.scene.add_entity("rabbit", pos=(9, 1.5, 9), speed=0.0)
    loop.wm.perceive(observations=loop._observe())

    payload = loop.flush_payloads()["payload"]
    lines = [ln for ln in payload["body_md"].splitlines()
             if ln.strip().startswith("- entity:")]
    assert lines, "flush 未写出任何 entity 行"
    parsed = [SimulationLoop.parse_seed_entity(ln) for ln in lines]
    assert all(p is not None for p in parsed)
    live = loop.wm.graph()["nodes"]
    assert {p["eid"] for p in parsed} == set(live)
    for p in parsed:
        assert p["category"] == live[p["eid"]]["category"]
        assert tuple(p["pos"]) == tuple(live[p["eid"]]["pos"])
    assert any(p["category"] == "large cart" for p in parsed)


def test_flush_to_disk_then_load_priors_round_trips(tmp_path, monkeypatch):
    """#399 端到端：flush 落盘 → 新进程 load_priors 读回，类别/位置须一致。

    这是「持久化往返」的完整形态：写（flush_payloads）与读（load_priors →
    parse_seed_entity）跨实例。旧实现在这里把 `large cart` 还原成 `large`。
    """
    monkeypatch.delenv("MDCG_ROOT", raising=False)
    loop = SimulationLoop(seed=2)
    loop.scene.create_scene(trees=0, water=False)
    loop.scene.add_entity("large cart", pos=(2, 1.5, 2), speed=0.0)
    loop.scene.add_entity("rabbit", pos=(9, 1.5, 9), speed=0.0)
    loop.wm.perceive(observations=loop._observe())
    loop.flush_payloads(out_path=str(tmp_path / "contextual" / "run.md"))

    resumed = SimulationLoop()
    loaded = resumed.load_priors(str(tmp_path))
    assert loaded["seeded"] == 2
    by_cat = {e["category"]: tuple(e["pos"]) for e in loaded["entities"]}
    assert by_cat == {"large cart": (2.0, 1.5, 2.0),
                      "rabbit": (9.0, 1.5, 9.0)}
