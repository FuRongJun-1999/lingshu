# -*- coding: utf-8 -*-
"""守卫 · #172 ③：pixel → hex_cnn/hex_hier → World3D → engine 真实蜂窝感知器升级
====================================================================================
背景：
  旧版 `tests/test_e2e_pixel_to_memory.py` 仅用极简占位器 `_hue_to_color`（返回中文
  "红/绿/蓝" 与仓内 `COLORS=["red","green","blue"]` 失配）+ 硬编码 `square|{color}`，
  导致 `hex_hits = 0/4`、形状全错。

本守卫覆盖：
  1. 真实蜂窝晶格链路：`perceive_scene_hex` 经 `image_to_grid` + `hex_conv` 计算
     六方向边缘/各向同性 Laplacian/Center-Surround 与亚晶格实心度特征；
  2. 多形状/多色/多尺寸准确率（n=96 @ size=48, n=36 @ size=64）：打破 0/4 占位；
  3. 可选 `HexHierNet` 模态支持；
  4. 零真值偷读证明（签名不含 oracle/labels、RGB 通道置换翻转颜色、纯噪声返回 None、
     强噪退化出现真实分歧）；
  5. 非法输入 fail-loud `ValueError`（空数组/坏维度/NaN/Inf/bool 参数）；
  6. 端到端 `World3D.add_view`（含双视角三角化落地 `Object3D`）+ `SpacetimeMemoryEngine`；
  7. `--mutate` 定点变异自证（破坏色表或晶格特征 ⇒ 精确点红，还原全绿）。

运行：
  pytest tests/test_issue172_hex_perceive_upgrade.py -v
  python -X utf8 tests/test_issue172_hex_perceive_upgrade.py
  python -X utf8 tests/test_issue172_hex_perceive_upgrade.py --mutate
"""
from __future__ import annotations

import hashlib
import inspect
import os
import subprocess
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import lingshu.nn.hex_hier as hex_hier_mod  # noqa: E402
from lingshu.core.core import ConditionSpace, STNode, SpacetimeMemoryEngine  # noqa: E402
from lingshu.nn.hex_hier import (  # noqa: E402
    COLORS,
    HexHierNet,
    OBJ,
    POS,
    SHAPES,
    make_scene_dataset,
    perceive_scene_hex,
)
from lingshu.world.world3d import Camera3D, World3D  # noqa: E402

HEX_HIER_PATH = os.path.join(REPO, "lingshu", "nn", "hex_hier.py")


def test_hex_perceive_uses_real_hex_lattice_and_conv(monkeypatch):
    """P1：perceive_scene_hex 必须真实调用 image_to_grid 与 hex_conv，返回完整白箱字段。"""
    calls = {"grid": 0, "conv": 0}
    orig_grid = hex_hier_mod.image_to_grid
    orig_conv = hex_hier_mod.hex_conv

    def spy_grid(*args, **kwargs):
        calls["grid"] += 1
        return orig_grid(*args, **kwargs)

    def spy_conv(*args, **kwargs):
        calls["conv"] += 1
        return orig_conv(*args, **kwargs)

    monkeypatch.setattr(hex_hier_mod, "image_to_grid", spy_grid)
    monkeypatch.setattr(hex_hier_mod, "hex_conv", spy_conv)

    xs, labs = make_scene_dataset(3, 48, seed=7)
    res = perceive_scene_hex(xs[0])
    assert res is not None
    assert calls["grid"] >= 1, "必须经过 image_to_grid 转换六边形晶格"
    assert calls["conv"] >= 8, "必须经过 hex_conv 计算六方向边缘与各向同性响应"

    assert res["category"] in OBJ
    assert res["shape"] in SHAPES and res["color"] in COLORS
    assert res["pos"] in POS
    assert 0.0 < res["confidence"] <= 1.0
    x1, y1, x2, y2 = res["bbox"]
    assert 0 <= x1 <= x2 < 48 and 0 <= y1 <= y2 < 48
    feat = res["features"]
    assert len(feat["dir_energies"]) == 6
    assert all(v >= 0.0 for v in feat["dir_energies"])
    assert feat["lap_ratio"] > 0.0 and feat["solidity"] > 0.0
    assert res["category"] == labs["obj"][0]


def test_benchmark_accuracy_across_seeds_and_sizes():
    """P2：在 n=96 (size=48, seed=7) 与 n=36 (size=64, seed=13) 上实测三类形状与三色准确率。"""
    for n, size, seed in ((96, 48, 7), (36, 64, 13)):
        xs, labs = make_scene_dataset(n, size, seed=seed)
        color_ok = shape_ok = obj_ok = 0
        per_shape_tot = {s: 0 for s in SHAPES}
        per_shape_ok = {s: 0 for s in SHAPES}
        per_color_tot = {c: 0 for c in COLORS}
        per_color_ok = {c: 0 for c in COLORS}

        for i in range(n):
            res = perceive_scene_hex(xs[i])
            assert res is not None, f"样本 {i} 不应漏检"
            gt_s, gt_c, gt_o = labs["shape"][i], labs["color"][i], labs["obj"][i]
            per_shape_tot[gt_s] += 1
            per_color_tot[gt_c] += 1
            if res["color"] == gt_c:
                color_ok += 1
                per_color_ok[gt_c] += 1
            if res["shape"] == gt_s:
                shape_ok += 1
                per_shape_ok[gt_s] += 1
            if res["category"] == gt_o:
                obj_ok += 1

        c_acc = color_ok / n
        s_acc = shape_ok / n
        o_acc = obj_ok / n
        print(f"\n[白箱评测 n={n} size={size} seed={seed}] "
              f"color={c_acc:.1%} shape={s_acc:.1%} obj={o_acc:.1%}")
        assert c_acc >= 0.98, f"颜色准确率过低: {c_acc:.3f}"
        assert s_acc >= 0.95, f"形状准确率过低: {s_acc:.3f}"
        assert o_acc >= 0.95, f"物体联合准确率过低: {o_acc:.3f}"
        for s in SHAPES:
            assert per_shape_ok[s] / max(1, per_shape_tot[s]) >= 0.90, f"形状 {s} 坍缩"
        for c in COLORS:
            assert per_color_ok[c] / max(1, per_color_tot[c]) >= 0.95, f"颜色 {c} 坍缩"


def test_hex_hier_net_mode_supported():
    """P3：传入 HexHierNet 实例时可经 L2/L3 推理出有效类别与概率置信度。"""
    xs, _ = make_scene_dataset(4, 48, seed=7)
    net = HexHierNet(seed=7)
    res = perceive_scene_hex(xs[0], cells_across=16, net=net)
    assert res is not None
    assert res["category"] in OBJ
    assert res["shape"] in SHAPES and res["color"] in COLORS
    assert 0.0 < res["confidence"] <= 1.0
    assert set(res["shape_scores"]) == set(OBJ)


def test_perceive_does_not_leak_oracle_labels():
    """P4：反作弊——签名不接真值标签；通道置换翻转颜色；纯噪返回 None。"""
    sig = inspect.signature(perceive_scene_hex)
    bad_params = {"labels", "labs", "oracle", "truth", "target", "y"}
    assert not (set(sig.parameters) & bad_params)

    # 寻找一个红色样本，置换 R/B 通道应转为蓝色而形状保持一致
    xs, labs = make_scene_dataset(12, 48, seed=7)
    red_idx = int(np.where(labs["color"] == "red")[0][0])
    red_img = xs[red_idx]
    res_orig = perceive_scene_hex(red_img)
    assert res_orig is not None and res_orig["color"] == "red"

    swapped = red_img[..., [2, 1, 0]].copy()
    res_swap = perceive_scene_hex(swapped)
    assert res_swap is not None
    assert res_swap["color"] == "blue", f"R/B 置换后应判为 blue，实得 {res_swap['color']}"
    assert res_swap["shape"] == res_orig["shape"], "颜色置换不应改变几何形状识别"

    # 纯背景低噪图与全零黑图应返回 None（不编造类别）
    rng = np.random.default_rng(42)
    bg_only = (rng.random((48, 48, 3)) * 40).astype(np.uint8)
    assert perceive_scene_hex(bg_only) is None
    assert perceive_scene_hex(np.zeros((48, 48, 3), dtype=np.uint8)) is None


@pytest.mark.parametrize("bad_img", [
    None,
    True,
    np.zeros((0, 48, 3), dtype=np.uint8),
    np.zeros((48, 48), dtype=np.uint8),
    np.zeros((48, 48, 2), dtype=np.uint8),
    np.array([[["r", "g", "b"]]]),
    np.array([[[True, False, True]]], dtype=bool),
    np.full((16, 16, 3), np.nan),
    np.full((16, 16, 3), np.inf),
])
def test_fail_loud_invalid_image_inputs(bad_img):
    """P5a：非法图像入参（空/坏维/字符/bool/NaN/Inf）必须 fail-loud 抛 ValueError。"""
    with pytest.raises(ValueError):
        perceive_scene_hex(bad_img)


@pytest.mark.parametrize("kwargs", [
    {"cells_across": True},
    {"cells_across": 2},
    {"cells_across": 24.5},
    {"min_contrast": True},
    {"min_contrast": -1.0},
    {"min_contrast": float("nan")},
    {"net": "not_a_net"},
])
def test_fail_loud_invalid_scalar_params(kwargs):
    """P5b：非法标量/模型参数（bool/越界/NaN/错误类型）必须 fail-loud 抛 ValueError。"""
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        perceive_scene_hex(img, **kwargs)


def test_e2e_world3d_triangulation_and_memory_roundtrip():
    """P6：端到端验证——三形状全覆盖 + World3D 双视角三角化落出 Object3D + engine 读写检索。"""
    xs, labs = make_scene_dataset(36, 48, seed=7)
    chosen = []
    for s in SHAPES:
        idx = int(np.where(labs["shape"] == s)[0][0])
        chosen.append((idx, xs[idx], labs["obj"][idx]))

    world = World3D()
    cam1 = Camera3D(cx=0.0, cy=1.6, cz=-5.0, yaw=0.0)
    cam2 = Camera3D(cx=1.2, cy=1.6, cz=-5.0, yaw=-0.2)
    engine = SpacetimeMemoryEngine(":memory:")
    cs = ConditionSpace(
        observation_position="world3d",
        observation_tool="hex_hier",
        time_window=(0.0, 9999999999.0),
        existence_constraint="issue172-e2e",
    )

    for step, (idx, img, gt_obj) in enumerate(chosen):
        res = perceive_scene_hex(img)
        assert res is not None and res["category"] == gt_obj
        v1 = world.add_view(res["category"], res["bbox"], 48, 48,
                            camera=cam1, confidence=res["confidence"], eid=f"obj_{step}")
        v2 = world.add_view(res["category"], res["bbox"], 48, 48,
                            camera=cam2, confidence=res["confidence"], eid=f"obj_{step}")
        assert v1["triangulated"] is False
        assert v2["triangulated"] is True and v2["center_3d"] is not None

        node = STNode(
            id=f"hex_e2e_{step}_{res['category']}",
            content=f"感知 {res['category']} center={v2['center_3d']} pos={res['pos']}",
            modality="text",
            spatial_coordinates={"center_3d": list(v2["center_3d"]), "bbox": list(res["bbox"])},
            temporal_coordinate=float(step),
            condition_space=cs,
        )
        nid = engine.store.add_node(node)
        assert engine.store.get_node(nid) is not None
        hits = engine.store.search_content(res["category"])
        assert any(n.id == nid for n, _score in hits)

    assert len(world.objects) == 3
    assert {o.category for o in world.objects} == {gt for _, _, gt in chosen}


# ── 定点变异自证 (--mutate) ───────────────────────────────────

_MUTATIONS = [
    (
        "M1_chinese_color_mismatch",
        "    return COLORS[color_idx], float(np.clip(margin, 0.05, 0.99))",
        '    return ["红", "绿", "蓝"][color_idx], float(np.clip(margin, 0.05, 0.99))',
        {
            "test_hex_perceive_uses_real_hex_lattice_and_conv",
            "test_benchmark_accuracy_across_seeds_and_sizes",
            "test_perceive_does_not_leak_oracle_labels",
            "test_e2e_world3d_triangulation_and_memory_roundtrip",
        },
    ),
    (
        "M2_disable_lattice_shape_feats",
        '        "solidity": float(lat_peak / (px_peak + 1e-6)),\n'
        '        "lap_ratio": float(lap_sum / mass_sum),',
        '        "solidity": 1.0,\n'
        '        "lap_ratio": 0.0,',
        {
            "test_hex_perceive_uses_real_hex_lattice_and_conv",
            "test_benchmark_accuracy_across_seeds_and_sizes",
            "test_e2e_world3d_triangulation_and_memory_roundtrip",
        },
    ),
    (
        "M3_baseline_intact",
        None,
        None,
        set(),
    ),
]


def _run_pytest_collect_failures():
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "pytest", __file__, "-q", "--no-header"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    failed = set()
    for line in out.splitlines():
        if line.startswith("FAILED ") and "::" in line:
            failed.add(line.split("::", 1)[1].split()[0])
    summary = [l for l in out.splitlines() if " passed" in l or " failed" in l]
    return proc.returncode, failed, (summary[-1] if summary else "")


def _mutate():
    with open(HEX_HIER_PATH, "rb") as f:
        orig_bytes = f.read()
    orig_sha = hashlib.sha256(orig_bytes).hexdigest()
    orig_text = orig_bytes.decode("utf-8")
    # 规范为 LF 匹配目标片段，还原时按字节写回 orig_bytes
    bad = 0
    for name, target, repl, expect_fail in _MUTATIONS:
        try:
            if target is not None:
                norm_text = orig_text.replace("\r\n", "\n")
                assert target in norm_text, f"变异锚点未找到: {target!r}"
                mutated = norm_text.replace(target, repl, 1)
                with open(HEX_HIER_PATH, "w", encoding="utf-8", newline="\n") as wf:
                    wf.write(mutated)
            rc, failed, summary = _run_pytest_collect_failures()
            print(f"  {name:<32s} rc={rc} 期望红={len(expect_fail)} 实际红={len(failed)} {summary}")
            ok = (failed == expect_fail) if expect_fail else (rc == 0 and not failed)
            if not ok:
                bad += 1
                print(f"    VERDICT=FAIL 漏红={sorted(expect_fail - failed)} 多红={sorted(failed - expect_fail)}")
            else:
                print("    VERDICT=PASS")
        finally:
            with open(HEX_HIER_PATH, "wb") as wf:
                wf.write(orig_bytes)
            with open(HEX_HIER_PATH, "rb") as rf:
                assert hashlib.sha256(rf.read()).hexdigest() == orig_sha, "hex_hier.py 未原样还原"
    print(f"  OVERALL={'PASS' if bad == 0 else 'FAIL'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    if "--mutate" in sys.argv:
        sys.exit(_mutate())
    sys.exit(pytest.main([__file__, "-v"]))
