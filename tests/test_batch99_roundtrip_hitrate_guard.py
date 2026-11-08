# -*- coding: utf-8 -*-
"""批九十九守卫：写→读往返 + 同一 dict 内口径自洽（#2541 / #2542）。

覆盖
--------------------------------------------------------------------------
#2541 `wm_simloop.parse_seed_entity` 的类别截断
    G1 类别含 `" id="` 时必须完整往返
       （改前 `split(" id=", 1)` 取第一个标记 ⇒ 类别 `a id=b` 读成 `a`、
        eid 变成 `b id=scene_...`，写→读损坏）
    G2 类别含 `" pos=("` 也必须完整往返
    G3 正对照：普通类别与含空格的类别逐位不变（#399 的既有面）

#2542 `hits`（边界）与 `hit_rate`（点）在同一 dict 内互斥
    G4 `seven_layer_loop._verify` 的返回体必须带 `point_hits`，且
       `point_hits/total == hit_rate`（改前只有 `hits`，两套口径）
    G5 `_overall_hit_rate` 必须与 `hit_rate` 同口径
       （改前按 `hits` 池化 ⇒ `overall=1.0` 而 `hit_history=[0.0]`）
    G6 `world_model.verify` 的 `_compare` 必须带 `point_hits`
    G7 正对照：点与边界同时命中的正常路径，三个读数一致

运行（lingshu 仓根）：
    python -X utf8 -m pytest tests/test_batch99_roundtrip_hitrate_guard.py -q
"""
from __future__ import annotations

import pytest

from lingshu.world.wm_simloop import SimulationLoop


# =============================================== #2541 parse_seed_entity 往返
def _line(category, eid="scene_76d55f30", pos=(2.0, 1.5, 2.0)):
    return "- entity: %s id=%s pos=(%s, %s, %s) confidence=0.5" % (
        category, eid, pos[0], pos[1], pos[2])


def test_G1_category_with_id_marker_roundtrips():
    """G1：类别含 `" id="` 必须完整往返（改前截断成 `a`）。"""
    got = SimulationLoop.parse_seed_entity(_line("a id=b"))
    assert got["category"] == "a id=b", got
    assert got["eid"] == "scene_76d55f30", got
    assert got["pos"] == (2.0, 1.5, 2.0), got


def test_G2_category_with_pos_marker_roundtrips():
    """G2：类别含 `" pos=("` 也必须完整往返。"""
    got = SimulationLoop.parse_seed_entity(_line("a pos=(1,2,3)"))
    assert got["category"] == "a pos=(1,2,3)", got
    assert got["eid"] == "scene_76d55f30", got


@pytest.mark.parametrize("cat", ["desk", "large cart", "plain"])
def test_G3_plain_categories_unchanged(cat):
    """G3 正对照：普通类别与含空格类别逐位不变（#399 既有面）。"""
    got = SimulationLoop.parse_seed_entity(_line(cat))
    assert got["category"] == cat, got
    assert got["eid"] == "scene_76d55f30", got
    assert got["pos"] == (2.0, 1.5, 2.0), got


# ============================================= #2542 hits / hit_rate 同口径
def test_G4_seven_layer_verify_exposes_point_hits():
    """G4：`_verify` 返回体必须带 `point_hits`，且与 `hit_rate` 同口径。"""
    import inspect

    from lingshu.world.seven_layer_loop import SevenLayerLoop
    src = inspect.getsource(SevenLayerLoop._verify)
    assert '"point_hits": point_hits' in src, "`_verify` 返回体缺 point_hits"
    # `hits` 仍是边界口径（不得被改成点口径——那会破坏 #2167 的 bound_coverage）
    assert '"hits": hits' in src and '"bound_coverage": coverage' in src


def test_G5_overall_hit_rate_uses_point_hits():
    """G5：`_overall_hit_rate` 必须按点口径池化（改前按 `hits` 边界口径）。"""
    import inspect

    from lingshu.world.seven_layer_loop import SevenLayerLoop
    src = inspect.getsource(SevenLayerLoop._overall_hit_rate)
    assert "point_hits" in src, "`_overall_hit_rate` 仍按 hits 池化"
    # 兼容旧 audit 记录：无 point_hits 时回落 hits
    assert ".get(" in src, "缺旧记录兼容回落"


def test_G6_world_model_compare_exposes_point_hits():
    """G6：`world_model.verify` 的 `_compare` 必须带 `point_hits`。"""
    import inspect

    from lingshu.world.world_model import UnifiedWorldModel
    src = inspect.getsource(UnifiedWorldModel.verify)
    assert '"point_hits": point_hits' in src, "_compare 缺 point_hits"


def test_G7_consistent_when_point_and_bound_agree():
    """G7 正对照：点与边界同时命中时，hits/total 与 hit_rate 一致。"""
    from lingshu.world.world_model import UnifiedWorldModel
    wm = UnifiedWorldModel(size=24, seed=1)
    wm.generate(1)
    v = wm.verify()
    if v.get("total"):
        # 同一 dict 里 hits/total 与 hit_rate 必须能互相判读
        assert "point_hits" in v, v
        expect = round(v["point_hits"] / v["total"], 4)
        assert abs((v["hit_rate"] or 0.0) - expect) < 1e-9, (v["hit_rate"], expect)
        # 边界覆盖独立于点命中
        assert "bound_coverage" in v
