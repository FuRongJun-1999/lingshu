#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_issue282_345_396_world_guard —— world 三件 P1 缺陷守卫（本组件号 282/345/396）。

逐件钉住「缺陷不再存在」，而非「代码能跑」：

#282（world/scene_model · 「记忆即世界持久层」不成立）
    症状：`load_world_from_memory` 逐条 `wm.add_entity(name, ...)`，同名后写覆盖
    前写；而身体侧 store 返回序是 `importance DESC, last_access DESC`
    （core.py:1557，**最新在前**），于是最后写入的是**最旧**观测，新观测被丢弃、
    重建世界回落到陈旧坐标。
    修法：按实体名归并，取时新度（temporal_coordinate）最大者。
    守卫 G1：同一实体两次观测（新坐标在后）⇒ 重建坐标＝**最新**观测，且 ≠ 旧观测。
    变异（改回逐条 add_entity）⇒ G1 读到旧坐标 ⇒ 红。

#345（world/curiosity_explorer · compare_policies 不透传实例配置）
    症状：`_build_world` 硬编码 `CuriosityExplorer(size=24)`（seed=42），
    `compare_policies` 全程只用内置世界——非缺省实例（如 size=48, seed=7）的
    比较结果与该实例无关。
    修法：`_build_world` 取 `self.size/self.ground_level/self.seed/self.window`。
    守卫 G2：非缺省实例 `_build_world()` 建出的世界配置＝本实例配置
    （size/seed/window/ground_level 逐项相等，且随机流＝seed 对应流）。
    变异（改回 `CuriosityExplorer(size=24)`）⇒ G2 读到 24/42 ⇒ 红。

#396（world/seven_layer_loop · 七层闭环从不调用异常计数）
    症状：`step()` 内 `observe(entities=chosen)` 之后没有 `_count_anomalies`，
    七层宣称的「预测-观测不一致 → 异常加成」反馈形同虚设（对照
    `curiosity_explorer.explore_tick` 会调用）。
    修法：`step()` 在 observe 后调用 `_count_anomalies(chosen)`，并把计数
    写进 L1 留痕。
    守卫 G3：`step()` 必须以本轮 `chosen` 调用一次 `_count_anomalies`，且
    L1 留痕带 `anomalies` 整数字段。
    变异（删掉该调用）⇒ G3 观察到 0 次调用 ⇒ 红。

运行（lingshu 仓根）：python -X utf8 -m pytest tests/test_issue282_345_396_world_guard.py -q --no-header
退出码：0 = 全过；非 0 = 有断言失败。
"""
from __future__ import annotations

import os
import random
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.core.core import SpacetimeMemoryEngine            # noqa: E402
from lingshu.world.curiosity_explorer import CuriosityExplorer  # noqa: E402
from lingshu.world.scene_model import (                         # noqa: E402
    ingest_scene, load_world_from_memory)
from lingshu.world.seven_layer_loop import SevenLayerLoop       # noqa: E402


class _Agent:
    """scene_model 消费面：同时提供 `.store` 与 `.engine`。"""

    def __init__(self, engine):
        self.engine = engine
        self.store = engine.store


# ---------------------------------------------------------------------------
# #282 —— 世界重建取最新观测
# ---------------------------------------------------------------------------

def test_282_rebuild_uses_latest_observation_of_same_entity():
    """同一实体两次观测 ⇒ 重建坐标＝最新观测（改前回落到最旧观测）。"""
    e = SpacetimeMemoryEngine(":memory:")
    agent = _Agent(e)
    # 先写旧坐标，隔开时间戳再写新坐标（同实体、同状态，仅坐标移动）
    old_pos, new_pos = (0.0, 0.85, 5.0), (3.5, 0.85, 5.0)
    ingest_scene(agent, "肥鱼|fatfish|0,0.85,5|shy")
    time.sleep(0.03)
    ingest_scene(agent, "肥鱼|fatfish|3.5,0.85,5|shy")

    # 前置：两条观测都真落库（否则断言可能因「只有一条」而空过）
    nodes = e.store.get_nodes_by_tag("spatial", limit=200)
    assert len(nodes) == 2, [n.content for n in nodes]

    wm = load_world_from_memory(agent.store)
    pos = wm.entities["肥鱼"].pos
    assert pos == new_pos, ("#282 世界重建须取最新观测坐标；"
                            "读到 %r（旧坐标 %r）" % (pos, old_pos))
    assert pos != old_pos, "#282 不得回落到最旧观测"


# ---------------------------------------------------------------------------
# #345 —— compare_policies 的世界配置透传本实例
# ---------------------------------------------------------------------------

def test_345_build_world_inherits_instance_config():
    """非缺省实例 `_build_world()` 建出的世界＝本实例配置（改前硬编码 24/42）。"""
    ex = CuriosityExplorer(size=48, ground_level=1, seed=7, window=8)
    w = ex._build_world()
    assert w.size == 48, "#345 世界尺寸须透传 self.size（改前恒为 24）"
    assert w.seed == 7, "#345 随机种子须透传 self.seed（改前恒为 42）"
    assert w.window == 8, "#345 观测窗口须透传 self.window（改前恒为 6）"
    assert w.ground_level == 1, "#345 地面层须透传 self.ground_level"
    assert w.world.world.size == 48, "#345 物理世界尺寸同步（非仅学习器字段）"
    # 随机流必须＝seed=7 对应的流（证明 seed 真进了物理世界）
    assert w._rng.random() == random.Random(7).random(), "#345 seed 未生效"


def test_345_compare_policies_on_nondefault_instance_does_not_crash():
    """对照：非缺省实例的 compare_policies 仍可跑通（本修不破坏调用面）。"""
    ex = CuriosityExplorer(size=32, seed=3, window=4)
    res = ex.compare_policies(budget=2, explore_ticks=3, probe_ticks=2)
    assert set(res["results"].keys()) == {"curiosity", "random", "round_robin"}


# ---------------------------------------------------------------------------
# #396 —— 七层闭环调用异常计数并留痕
# ---------------------------------------------------------------------------

def test_396_step_invokes_anomaly_counting():
    """`step()` 必须以本轮 chosen 调用 `_count_anomalies` 一次，并写入 L1 留痕。"""
    sll = SevenLayerLoop(size=24, seed=42)
    sll.create_scene(trees=2)
    sll.add_entity("player", behavior="wander", pos=(2, 1.5, 2), speed=0.5)
    sll.add_entity("wolf", behavior="seek", pos=(15, 1.5, 15), speed=0.6)

    calls = []
    orig = sll.explorer._count_anomalies

    def _spy(chosen):
        calls.append(list(chosen))
        return orig(chosen)

    sll.explorer._count_anomalies = _spy
    rec = sll.step()

    assert len(calls) == 1, ("#396 step() 必须调用 _count_anomalies 恰一次"
                             "（改前 0 次）——实际 %d 次" % len(calls))
    assert calls[0] == rec["L7_decision"]["chosen"], (
        "#396 异常计数须针对本轮 chosen 实体", (calls[0], rec["L7_decision"]["chosen"]))
    assert "anomalies" in rec["L1_perception"], rec["L1_perception"]
    assert isinstance(rec["L1_perception"]["anomalies"], int), rec["L1_perception"]


def test_396_count_anomalies_is_live_positive_control():
    """正对照：`_count_anomalies` 本身确实会计数（证明接线非空壳）。"""
    ex = CuriosityExplorer(size=24, seed=42)
    ex.world.create_scene(trees=1)
    eid = ex.world.add_entity("player", behavior="wander", pos=(2, 1.5, 2), speed=0.5)
    ex.observe()
    # 注入一个远离当前坐标的「上一预测」→ 预测-观测不一致
    ex._last_prediction = {eid: {"predicted": [0.0, 0.0, 0.0], "bound": 0.01}}
    n = ex._count_anomalies([eid])
    assert n == 1, "#396 异常计数正对照：远离的预测须计为 1 个异常"
    assert ex._anomaly_counts.get(eid) == 1, ex._anomaly_counts


if __name__ == "__main__":
    import traceback

    failed = 0
    for _name in sorted(n for n in list(globals()) if n.startswith("test_")):
        try:
            globals()[_name]()
            print("  PASS " + _name)
        except Exception as exc:                                   # noqa: BLE001
            failed += 1
            print("  FAIL %s: %s: %s" % (_name, type(exc).__name__, exc))
            traceback.print_exc()
    print("\n%s — %d failed" % ("FAILED" if failed else "OK", failed))
    sys.exit(1 if failed else 0)
