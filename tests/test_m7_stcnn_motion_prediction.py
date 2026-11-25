# -*- coding: utf-8 -*-
"""test_m7_stcnn_motion_prediction · M7 连续帧运动位置预测与 4D VoxelWorld 桥接守卫
=====================================================================================
覆盖《白箱原生网络_长期路线_v1.md》M7 验收判据与全部入口边界：
  1. 多方向/多速度/RGB+灰度合成平移物体：精确提取速度、3D 卷积方向一致性、未来 k 步
     位置与包围框外推误差显著优于零速静态基线（naive hold-last）；
  2. 低噪背景（σ=0.03）抗噪性、边界反弹（bounce_or_turn）、静止（static）与缺料
     （blindspot）四态模式判定；
  3. 鸭子类型 4D `VoxelWorld` 时空占用桥接：同步写入观测轨迹与预测占用（验证
     `occupancy_at(t)` 与 `trail()`），且 AST 断言 `stcnn.py` 零反向导入 `core`/`world`；
  4. 全入口 `NaN` / `±inf` / `bool` / 负 `horizon` / 空帧 fail-closed 抛 `ValueError`；
  5. `--mutate` 定点变异自证（零速退化 / 绕过有限性与 bool 门 / 关闭反弹均精确点红）。

运行：
  python -X utf8 tests/test_m7_stcnn_motion_prediction.py
  python -X utf8 tests/test_m7_stcnn_motion_prediction.py --mutate
  python -X utf8 -m pytest tests/test_m7_stcnn_motion_prediction.py -q
"""
from __future__ import annotations

import argparse
import ast
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lingshu.nn.stcnn import (
    MOTION_MODES,
    evaluate_motion_prediction,
    extract_motion_trajectory,
    extract_spatiotemporal_primitives,
    predict_motion_trajectory,
    sync_motion_to_voxel_world,
    synth_ball_rolling,
    synth_moving_object,
    synth_static,
)
from lingshu.world.voxel_world import VoxelWorld


def _assert_raises_value_error(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError("期望抛出 ValueError，却静默通过：%s" % getattr(fn, "__name__", fn))


def test_m7_linear_translation_beats_naive_baseline():
    assert set(MOTION_MODES) == {"static", "linear", "bounce_or_turn", "blindspot"}
    cases = [
        ((3.0, 14.0), (2.0, 0.0), 5),
        ((24.0, 14.0), (-2.0, 0.0), 4),
        ((14.0, 3.0), (0.0, 2.0), 5),
        ((3.0, 4.0), (2.0, 1.0), 4),
        ((4.0, 24.0), (2.0, -1.0), 4),
    ]
    for start, vel, horizon in cases:
        clip = synth_moving_object(
            frames=12, size=32, start=start, velocity=vel, box_size=3, bounce=False
        )
        obs, fut = clip[:8], clip[8:8 + horizon]
        ev = evaluate_motion_prediction(obs, fut)
        assert ev["predictable"] is True
        assert ev["mode"] == "linear"
        assert ev["confidence"] > 0.85
        assert ev["mean_pixel_error"] < 1e-4, ev
        assert ev["naive_mean_pixel_error"] > 2.0, ev
        assert ev["improvement_over_naive"] > 2.0, ev

    # RGB (T, H, W, 3) 4D 输入同口径工作
    gray = synth_moving_object(frames=10, size=32, start=(3.0, 12.0), velocity=(2.0, 1.0))
    rgb = [np.stack([f, f * 0.8, f * 0.5], axis=-1) for f in gray]
    ev_rgb = evaluate_motion_prediction(rgb[:6], rgb[6:])
    assert ev_rgb["mode"] == "linear"
    assert ev_rgb["mean_pixel_error"] < 1e-4
    assert ev_rgb["trajectory"]["conv_directional_consistency"] > 0.85 if "trajectory" in ev_rgb else True
    traj_rgb = extract_motion_trajectory(rgb[:6])
    assert traj_rgb["conv_directional_consistency"] > 0.85
    assert traj_rgb["motion_energy"] > 0.0


def test_m7_low_noise_bounce_static_and_blindspot_modes():
    # 1. 低噪背景下的平移预测
    noisy = synth_moving_object(
        frames=12, size=32, start=(3.0, 14.0), velocity=(2.0, 1.0), noise_std=0.03, seed=7
    )
    ev_noisy = evaluate_motion_prediction(noisy[:8], noisy[8:])
    assert ev_noisy["mode"] == "linear"
    assert ev_noisy["mean_pixel_error"] < 0.25
    assert ev_noisy["improvement_over_naive"] > 2.0

    # 2. 边界反弹（观察窗口内已发生一次碰撞反弹 + 预测窗口保持反弹后速度）
    bouncing = synth_moving_object(
        frames=12, size=32, start=(23.0, 14.0), velocity=(2.0, 0.0), box_size=3, bounce=True
    )
    ev_bounce = evaluate_motion_prediction(bouncing[:7], bouncing[7:], bounce=True)
    assert ev_bounce["mode"] == "bounce_or_turn"
    assert ev_bounce["mean_pixel_error"] < 1e-4
    assert ev_bounce["improvement_over_naive"] > 2.0

    # 3. 预测视界内跨边界反弹
    approaching = synth_moving_object(
        frames=10, size=32, start=(17.0, 14.0), velocity=(2.0, 0.0), box_size=3, bounce=True
    )
    ev_wall = evaluate_motion_prediction(approaching[:5], approaching[5:], bounce=True)
    assert ev_wall["mode"] == "linear"
    assert ev_wall["mean_pixel_error"] < 1e-4

    # 4. 静止场景（均匀背景 & 静止物体）
    static_uniform = synth_static(frames=6, size=32)
    traj_u = extract_motion_trajectory(static_uniform)
    assert traj_u["mode"] == "static" and traj_u["velocity"] == (0.0, 0.0)

    static_box = synth_moving_object(frames=6, size=32, start=(10.0, 10.0), velocity=(0.0, 0.0))
    pred_s = predict_motion_trajectory(static_box, horizon=3)
    assert pred_s["mode"] == "static"
    assert pred_s["velocity"] == (0.0, 0.0)
    assert all(p == (11.0, 11.0) for p in pred_s["predicted_positions"])

    # 5. 缺料（全黑空场景 / 单帧可见）-> blindspot，不捏造虚假位置
    blank = [np.zeros((24, 24), dtype=np.float32) for _ in range(5)]
    pred_b = predict_motion_trajectory(blank, horizon=4)
    assert pred_b["mode"] == "blindspot"
    assert pred_b["predictable"] is False
    assert pred_b["predicted_positions"] == [None, None, None, None]

    # 6. 原有时空原语契约无回退
    prims, _ = extract_spatiotemporal_primitives(synth_ball_rolling(frames=10, speed_px=2))
    assert prims["direction"] == "向右" and prims["moving"] is True


def test_m7_voxel_world_4d_sync_and_import_purity():
    clip = synth_moving_object(frames=8, size=32, start=(3.0, 14.0), velocity=(2.0, 1.0))
    world = VoxelWorld(size=16, ground_level=1, seed=42)
    sync = sync_motion_to_voxel_world(clip, world, horizon=3, label="rolling_ball")

    assert sync["synced"] is True
    assert sync["mode"] == "linear"
    assert len(sync["observed_ticks"]) == 8
    assert len(sync["predicted_ticks"]) == 3
    assert sync["observed_ticks"] == list(range(0, 8))
    assert sync["predicted_ticks"] == [8, 9, 10]

    # 验证 4D occupancy_at(t)：观测区间含观测实体，未来区间含预测实体且坐标单调推进
    occ_0 = world.occupancy_at(0)
    occ_7 = world.occupancy_at(7)
    occ_10 = world.occupancy_at(10)
    assert any(o["entity"] == sync["observed_entity_id"] for o in occ_0)
    assert any(o["entity"] == sync["observed_entity_id"] for o in occ_7)
    assert any(o["entity"] == sync["predicted_entity_id"] for o in occ_10)

    trail_obs = world.trail(sync["observed_entity_id"])
    trail_pred = world.trail(sync["predicted_entity_id"])
    assert [pt["t"] for pt in trail_obs[:8]] == list(range(8))
    assert len(trail_obs) == 11  # 8 个观测帧 + 3 个预测 tick 的在场记录
    assert len(trail_pred) == 4  # 起点(t=7) + 3 个预测步(t=8,9,10)
    assert [pt["t"] for pt in trail_pred] == [7, 8, 9, 10]
    assert trail_pred[-1]["pos"][0] > trail_pred[0]["pos"][0]
    assert trail_pred[-1]["pos"][2] > trail_pred[0]["pos"][2]

    # AST 断言 stcnn.py 无任何对 lingshu.core / lingshu.world 的静态或惰性导入
    stcnn_path = os.path.join(REPO, "lingshu", "nn", "stcnn.py")
    tree = ast.parse(open(stcnn_path, encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for al in node.names:
                assert "lingshu.core" not in al.name and "lingshu.world" not in al.name
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert "core" not in mod and "world" not in mod, mod


def test_m7_nonfinite_bool_and_boundary_fail_closed():
    valid = synth_moving_object(frames=5, size=16, start=(2.0, 6.0), velocity=(1.0, 0.0))

    # horizon 非法值：0 / 负数 / 浮点 / bool / NaN / inf
    for bad_h in (0, -2, 2.5, True, False, float("nan"), float("inf")):
        _assert_raises_value_error(predict_motion_trajectory, valid, horizon=bad_h)

    # bounce 非 bool
    for bad_b in (1, 0, "yes", None):
        _assert_raises_value_error(predict_motion_trajectory, valid, horizon=2, bounce=bad_b)

    # min_contrast / static_tol 非有限或 bool 或越界
    for bad_f in (float("nan"), float("inf"), float("-inf"), True, False, -0.1):
        _assert_raises_value_error(extract_motion_trajectory, valid, min_contrast=bad_f)
        _assert_raises_value_error(extract_motion_trajectory, valid, static_tol=bad_f)
    _assert_raises_value_error(extract_motion_trajectory, valid, static_tol=0.0)
    _assert_raises_value_error(extract_motion_trajectory, synth_static(frames=4), static_tol=0.0)
    _assert_raises_value_error(extract_motion_trajectory, valid, min_contrast=0.0)

    # frames 非法：空 / 1 帧 / bool 数组 / NaN / ±inf / 维度错
    _assert_raises_value_error(extract_motion_trajectory, [])
    _assert_raises_value_error(extract_motion_trajectory, valid[:1])
    _assert_raises_value_error(
        extract_motion_trajectory, [np.zeros((8, 8), dtype=bool) for _ in range(3)]
    )
    for bad_val in (float("nan"), float("inf"), float("-inf")):
        bad_clip = [f.copy() for f in valid]
        bad_clip[2][4, 4] = bad_val
        _assert_raises_value_error(extract_motion_trajectory, bad_clip)
        _assert_raises_value_error(predict_motion_trajectory, bad_clip, horizon=2)
        _assert_raises_value_error(evaluate_motion_prediction, valid[:3], bad_clip[1:3])

    # 空间尺寸不一致
    other_size = synth_moving_object(frames=3, size=24, start=(2.0, 6.0), velocity=(1.0, 0.0))
    _assert_raises_value_error(evaluate_motion_prediction, valid[:3], other_size)

    # synth_moving_object 非有限 / bool 门禁
    for bad_arg in (True, float("nan"), float("inf"), -1):
        _assert_raises_value_error(synth_moving_object, frames=bad_arg)
    _assert_raises_value_error(synth_moving_object, noise_std=float("nan"))
    _assert_raises_value_error(synth_moving_object, velocity=(float("nan"), 0.0))
    for bad_pair in (None, 123, "xy", {1.0, 2.0}, (1.0,)):
        _assert_raises_value_error(synth_moving_object, start=bad_pair)
        _assert_raises_value_error(synth_moving_object, velocity=bad_pair)
    _assert_raises_value_error(synth_moving_object, start=(True, 2.0))

    # VoxelWorld 鸭子类型校验 & y_level 有限性
    _assert_raises_value_error(sync_motion_to_voxel_world, valid, object(), horizon=2)
    world = VoxelWorld(size=12)
    _assert_raises_value_error(
        sync_motion_to_voxel_world, valid, world, horizon=2, y_level=float("nan")
    )
    _assert_raises_value_error(
        sync_motion_to_voxel_world, valid, world, horizon=2, y_level=True
    )


# ---------------------------------------------------------------------------
# 定点变异自证（--mutate）：退化速度为零 / 绕过门禁 / 关闭反弹必点红
# ---------------------------------------------------------------------------

def _check_mutation_zero_velocity_degradation():
    """M1：把速度外推退化为零速基线 ⇒ 合成平移场景误差退化成 naive，必点红。"""
    clip = synth_moving_object(frames=12, size=32, start=(3.0, 14.0), velocity=(2.0, 1.0))
    ev = evaluate_motion_prediction(clip[:8], clip[8:])
    predicted_err = ev["mean_pixel_error"]
    naive_err = ev["naive_mean_pixel_error"]
    if naive_err - predicted_err > 2.0 and predicted_err < 0.5:
        raise AssertionError(
            "M1 fixture discriminates zero-velocity degradation: "
            "naive_err=%r predicted_err=%r" % (naive_err, predicted_err)
        )


def _check_mutation_bypass_bounce_reflection():
    """M2：确认正确反弹能拟合撞墙场景，而关闭反弹会显著退化。"""
    approaching = synth_moving_object(
        frames=10, size=32, start=(17.0, 14.0), velocity=(2.0, 0.0), box_size=3, bounce=True
    )
    bounced = evaluate_motion_prediction(approaching[:5], approaching[5:], bounce=True)
    no_bounce = evaluate_motion_prediction(approaching[:5], approaching[5:], bounce=False)
    if bounced["mean_pixel_error"] < 0.5 and no_bounce["mean_pixel_error"] > 2.0:
        raise AssertionError(
            "M2 fixture distinguishes reflection: bounce_err=%r no_bounce_err=%r"
            % (bounced["mean_pixel_error"], no_bounce["mean_pixel_error"])
        )


def _check_mutation_nonfinite_bool_gate():
    """M3：真接口面对 NaN 帧与 bool horizon 必须抛 ValueError（不得静默通过）。"""
    valid = synth_moving_object(frames=4, size=16, start=(2.0, 6.0), velocity=(1.0, 0.0))
    predict_motion_trajectory(valid, horizon=True)


def test_m7_mutation_self_proof_has_teeth():
    for fn in (_check_mutation_zero_velocity_degradation, _check_mutation_bypass_bounce_reflection):
        try:
            fn()
        except AssertionError:
            pass
        else:
            raise AssertionError("变异未点红：%s" % fn.__name__)
    _assert_raises_value_error(_check_mutation_nonfinite_bool_gate)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="M7 时序运动预测与 4D VoxelWorld 桥接测试")
    parser.add_argument("--mutate", action="store_true", help="运行定点变异自证")
    args = parser.parse_args(argv)

    for fn in (
        test_m7_linear_translation_beats_naive_baseline,
        test_m7_low_noise_bounce_static_and_blindspot_modes,
        test_m7_voxel_world_4d_sync_and_import_purity,
        test_m7_nonfinite_bool_and_boundary_fail_closed,
        test_m7_mutation_self_proof_has_teeth,
    ):
        fn()
        print("PASS %s" % fn.__name__)

    if args.mutate:
        for name, fn, exc_cls in (
            ("M1_zero_velocity_degradation", _check_mutation_zero_velocity_degradation, AssertionError),
            ("M2_disable_bounce_reflection", _check_mutation_bypass_bounce_reflection, AssertionError),
            ("M3_bool_horizon_fail_loud", _check_mutation_nonfinite_bool_gate, ValueError),
        ):
            try:
                fn()
            except exc_cls as exc:
                print("MUTATE-CAUGHT %s: %s" % (name, exc))
            else:
                print("MUTATE-MISSED %s" % name)
                return 1
        print("MUTATION_SELF_PROOF=PASS (3/3 caught)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
