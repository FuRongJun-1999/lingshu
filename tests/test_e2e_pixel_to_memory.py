# -*- coding: utf-8 -*-
"""#172 端到端路径：SceneSimulator 渲染 → hex_cnn/hex_hier 感知 → World3D.add_view → engine 写记忆。

双版并排：oracle（真值标签直喂）vs hex-perceived（白箱蜂窝晶格感知器从像素出 shape|color + bbox）。
判据：端到端测试转绿；色表对齐 COLORS/SHAPES、shape|color 命中率打破 0/4 占位；
在退化/低分辨率或强噪场景下与 oracle 的感知分歧仍可量化且不偷读真值标签。
"""
import os as _os, sys as _sys
_RP = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _RP)
import numpy as np
import pytest

from lingshu.nn.hex_cnn import image_to_grid  # noqa: F401 (晶格前端契约)
from lingshu.nn.hex_hier import (  # noqa: E402
    COLORS,
    OBJ,
    SHAPES,
    make_scene_dataset,
    perceive_scene_hex,
)
from lingshu.world.world3d import World3D, Camera3D  # noqa: F401
from lingshu.core.core import SpacetimeMemoryEngine, STNode, ConditionSpace


def hex_perceive(img_arr, cells_across=24, net=None):
    """真实白箱蜂窝感知：经 image_to_grid + hex_conv 从像素出 (category, bbox)——不经 oracle 标签。"""
    res = perceive_scene_hex(img_arr, cells_across=cells_across, net=net)
    if res is None:
        return None
    return (res["category"], res["bbox"])


# ── 端到端测试 ──────────────────────────────────────────────────

def test_pixel_to_memory_end_to_end():
    """#172 ③：pixel → hex_hier/hex_cnn 感知 → World3D.add_view → engine 写记忆，打破 0/4 占位。"""
    xs, labs = make_scene_dataset(4, 48, seed=7)
    assert xs.shape[0] == 4

    engine = SpacetimeMemoryEngine(":memory:")
    cs = ConditionSpace(observation_position="world3d", observation_tool="hex_cnn",
                        time_window=(0.0, 9999999999.0), existence_constraint="端到端实测")

    # 双版读数
    oracle_hits, color_hits, obj_hits, hex_miss = 0, 0, 0, 0
    written_nodes = []

    for i in range(4):
        img = xs[i]
        oracle_obj = labs["obj"][i]
        oracle_color = labs["color"][i]

        # oracle 版（真值直喂）
        world_inst = World3D()
        world_inst.add_view(oracle_obj, (0, 0, 48, 48), 48, 48, confidence=1.0)
        oracle_hits += 1

        # hex-perceived 版（真实白箱晶格感知）
        res = perceive_scene_hex(img)
        if res is not None:
            p_cat = res["category"]
            p_bbox = res["bbox"]
            p_conf = res["confidence"]
            assert p_cat in OBJ
            assert res["color"] in COLORS and res["shape"] in SHAPES
            assert 0.0 < p_conf <= 1.0

            if res["color"] == oracle_color:
                color_hits += 1
            if p_cat == oracle_obj:
                obj_hits += 1
            else:
                hex_miss += 1

            # World3D 观测
            obs = world_inst.add_view(p_cat, p_bbox, 48, 48, confidence=p_conf)
            assert obs["category"] == p_cat

            # engine 写记忆（CCG 语义由调用方保证）
            node = STNode(
                id=f"obs_{i}_{p_cat}",
                content=f"hex感知: {p_cat} bbox={p_bbox} pos={res['pos']} conf={p_conf}",
                modality="text",
                spatial_coordinates={"bbox": list(p_bbox), "pos": res["pos"]},
                temporal_coordinate=float(i),
                condition_space=cs,
            )
            nid = engine.store.add_node(node)
            written_nodes.append((nid, p_cat))
        else:
            hex_miss += 1

    # 端到端断言（打破旧版 color=0/4、obj=0/4 占位）
    assert len(written_nodes) == 4, "4 帧记忆均应写入成功"
    for nid, p_cat in written_nodes:
        got = engine.store.get_node(nid)
        assert got is not None, f"节点 {nid} 可读回"
        found = engine.store.search_content(p_cat)
        assert any(n.id == nid for n, _score in found), f"节点 {nid} 可按 {p_cat} 检索"
    assert color_hits == 4, f"颜色识别应全对齐 COLORS（旧版 0/4），实得 {color_hits}/4"
    assert obj_hits >= 3, f"shape|color 物体识别应打破 0/4，实得 {obj_hits}/4"

    # 双版读数并排（报告用）
    print(f"\n[双版读数] oracle 命中: {oracle_hits}/4 | hex 颜色命中: {color_hits}/4 | "
          f"hex 物体命中: {obj_hits}/4 | 漏检/错类: {hex_miss}/4")


def test_oracle_vs_perceived_divergence_measurable():
    """oracle 与 hex-perceived 的分歧可量化（干净场景高对齐，强噪/低分辨率场景可测出真实分歧）。"""
    xs_clean, labs_clean = make_scene_dataset(4, 48, seed=3)
    clean_diverge = 0
    for i in range(4):
        p = hex_perceive(xs_clean[i])
        if p is None or labs_clean["obj"][i] != p[0]:
            clean_diverge += 1

    # 低分辨率/强噪退化场景：不喂真值标签，真实感知必然出现可观测分歧
    xs_deg, labs_deg = make_scene_dataset(8, 48, seed=3)
    rng = np.random.default_rng(172)
    noisy_diverge = 0
    for i in range(8):
        noisy = np.clip(
            xs_deg[i].astype(np.float64) + rng.normal(0.0, 90.0, xs_deg[i].shape),
            0, 255,
        ).astype(np.uint8)
        p = hex_perceive(noisy, cells_across=8)
        if p is None or labs_deg["obj"][i] != p[0]:
            noisy_diverge += 1

    assert clean_diverge <= 1, f"干净场景分歧应极低，实得 {clean_diverge}/4"
    assert noisy_diverge > 0, f"强噪+低分辨率场景下真实感知应有可观测分歧，实得 {noisy_diverge}/8"
    print(f"\n[分歧读数] clean={clean_diverge}/4 | degraded(sigma=90,cells=8)={noisy_diverge}/8")
