#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_issue172_perception_noise_curve · #172 端到端感知噪声退化曲线守卫（纯标准库）
====================================================================================
承接 PR #1406（tests/test_e2e_pixel_to_memory.py）的 #172 后续第 ① 项：
    SceneSimulator 真值 → Camera3D 投影 → 三轴可控感知噪声注入
    （misclass_rate / drop_rate / bbox_jitter & pos_jitter）
    → World3D.add_vprim → SpacetimeMemoryEngine 写入与召回闭环。

断言组：
    N1  同 seed 逐位确定性（run_episode 与 run_default_curves 两次运行完全一致）；
    N2  零噪声基线（rates=0, jitters=0）逐位等价于 clean oracle（100% 命中，0.0m 误差）；
        drop_rate=1.0 退化至 0% 命中；misclass_rate=1.0 语义准确率与端到端命中率退化至 0%；
    N3  单轴扫描（misclass_rate / drop_rate / bbox_jitter / pos_jitter）与组合网格
        的端到端命中率随噪声增大单调不增，且首末档存在显著退化（jitter 轴平均 XZ 误差严格递增）；
    N4  参数有限性与边界 fail-closed（bool / NaN / ±inf / 负值 / 越界 rate>1.0 / 非数值均抛 ValueError）；
    N5  定点变异自证（--mutate）：短路任意噪声轴或去掉参数有限性闸 ⇒ 对应断言精确点红，还原全绿。

运行方式（仓根）：
    python -X utf8 tests/test_issue172_perception_noise_curve.py
    python -X utf8 tests/test_issue172_perception_noise_curve.py --mutate
    python -X utf8 -m pytest tests/test_issue172_perception_noise_curve.py -q --no-header
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import bench.perception_noise_curve as pnc  # noqa: E402

BENCH_PATH = REPO / "bench" / "perception_noise_curve.py"
RUN_TIMEOUT = 120


def test_n1_same_seed_bitwise_deterministic() -> None:
    """N1：同 seed 与同配置下，单轮与全扫描均逐位确定性可复算。"""
    cfg = pnc.NoiseConfig(misclass_rate=0.25, drop_rate=0.15, bbox_jitter=8.0, pos_jitter=30.0)
    ep1 = pnc.run_episode(cfg, seed=2, num_ticks=5)
    ep2 = pnc.run_episode(cfg, seed=2, num_ticks=5)
    assert ep1 == ep2, "同 seed 单轮读数漂移：%r vs %r" % (ep1, ep2)

    rep1 = pnc.run_default_curves(seeds=(0, 1, 2), num_ticks=4)
    rep2 = pnc.run_default_curves(seeds=(0, 1, 2), num_ticks=4)
    assert json.dumps(rep1, sort_keys=True) == json.dumps(rep2, sort_keys=True), (
        "同参数全扫描 JSON 不一致"
    )


def test_n2_clean_oracle_and_extreme_degradation() -> None:
    """N2：零噪声等价 clean oracle（100% 命中）；drop=1.0 与 misclass=1.0 端到端归零。"""
    clean = pnc.evaluate_noise_point(pnc.NoiseConfig(0.0, 0.0, 0.0, 0.0), seeds=(0, 1, 2), num_ticks=5)
    assert clean["detection_rate"] == 1.0, clean
    assert clean["semantic_accuracy"] == 1.0, clean
    assert clean["spatial_hit_rate"] == 1.0, clean
    assert clean["e2e_hit_rate"] == 1.0, clean
    assert clean["mean_xz_error"] == 0.0, clean

    dropped = pnc.evaluate_noise_point(pnc.NoiseConfig(drop_rate=1.0), seeds=(0, 1, 2), num_ticks=5)
    assert dropped["detection_rate"] == 0.0, dropped
    assert dropped["semantic_accuracy"] == 0.0, dropped
    assert dropped["spatial_hit_rate"] == 0.0, dropped
    assert dropped["e2e_hit_rate"] == 0.0, dropped
    assert dropped["mean_xz_error"] is None, dropped

    mis = pnc.evaluate_noise_point(pnc.NoiseConfig(misclass_rate=1.0), seeds=(0, 1, 2), num_ticks=5)
    assert mis["detection_rate"] == 1.0, mis
    assert mis["semantic_accuracy"] == 0.0, mis
    assert mis["e2e_hit_rate"] == 0.0, mis


def test_n3_monotonic_degradation_across_all_axes() -> None:
    """N3：三类单轴扫描与组合网格的端到端命中率随噪声单调不增，且真有退化落差。"""
    rep = pnc.run_default_curves(seeds=(0, 1, 2, 3, 4), num_ticks=6)
    for axis in ("misclass_rate", "drop_rate", "bbox_jitter", "pos_jitter"):
        rows = rep["sweeps"][axis]
        rates = [r["e2e_hit_rate"] for r in rows]
        for i in range(len(rates) - 1):
            assert rates[i] >= rates[i + 1], (
                "轴 %s 在索引 %d→%d 未单调不增：%r" % (axis, i, i + 1, rates)
            )
        assert rates[0] == 1.0 and rates[0] > rates[-1], (
            "轴 %s 未呈现预期退化：%r" % (axis, rates)
        )
        if axis in ("bbox_jitter", "pos_jitter"):
            errs = [r["mean_xz_error"] for r in rows]
            assert all(e is not None for e in errs), errs
            for i in range(len(errs) - 1):
                assert errs[i] < errs[i + 1], (
                    "抖动轴 %s 平均 XZ 误差未严格递增：%r" % (axis, errs)
                )

    comb_rates = [r["e2e_hit_rate"] for r in rep["combined"]]
    for i in range(len(comb_rates) - 1):
        assert comb_rates[i] >= comb_rates[i + 1], (
            "组合网格未单调不增：%r" % (comb_rates,)
        )
    assert comb_rates[0] == 1.0 and comb_rates[-1] == 0.0, comb_rates


def test_n4_invalid_params_fail_closed() -> None:
    """N4：bool / NaN / ±inf / 越界值 / 非数值均严格抛 ValueError。"""
    bad_rates = (True, False, float("nan"), float("inf"), float("-inf"), -0.01, 1.01, "abc", None)
    for field in ("misclass_rate", "drop_rate"):
        for val in bad_rates:
            try:
                pnc.NoiseConfig(**{field: val})  # type: ignore[arg-type]
            except ValueError:
                pass
            else:
                raise AssertionError("%s=%r 必须抛 ValueError" % (field, val))

    bad_jitters = (True, False, float("nan"), float("inf"), float("-inf"), -1.0, "xyz", None)
    for field in ("bbox_jitter", "pos_jitter"):
        for val in bad_jitters:
            try:
                pnc.NoiseConfig(**{field: val})  # type: ignore[arg-type]
            except ValueError:
                pass
            else:
                raise AssertionError("%s=%r 必须抛 ValueError" % (field, val))

    for bad_tol in (True, False, 0.0, -0.5, float("nan"), float("inf")):
        try:
            pnc.run_episode(pnc.NoiseConfig(), seed=0, num_ticks=2, spatial_tol=bad_tol)  # type: ignore[arg-type]
        except ValueError:
            pass
        else:
            raise AssertionError("spatial_tol=%r 必须抛 ValueError" % (bad_tol,))

    for bad_seed in (True, False, -1, 1.5):
        try:
            pnc.run_episode(pnc.NoiseConfig(), seed=bad_seed, num_ticks=2)  # type: ignore[arg-type]
        except ValueError:
            pass
        else:
            raise AssertionError("seed=%r 必须抛 ValueError" % (bad_seed,))


def _run_all_assertions() -> int:
    tests = (
        test_n1_same_seed_bitwise_deterministic,
        test_n2_clean_oracle_and_extreme_degradation,
        test_n3_monotonic_degradation_across_all_axes,
        test_n4_invalid_params_fail_closed,
    )
    fails = 0
    for fn in tests:
        try:
            fn()
            print("  PASS %s" % fn.__name__)
        except AssertionError as exc:
            fails += 1
            print("  FAIL %s: %s" % (fn.__name__, exc))
        except Exception as exc:
            fails += 1
            print("  FAIL %s: %s: %s" % (fn.__name__, type(exc).__name__, exc))
    print("SUMMARY pass=%d fail=%d" % (len(tests) - fails, fails))
    return 0 if fails == 0 else 1


_MUTATION_RUNNER = r'''
import importlib.util, os, sys
repo = os.getcwd()
if repo not in sys.path:
    sys.path.insert(0, repo)
mut_path, test_path = sys.argv[1], sys.argv[2]
import bench  # noqa: F401
spec = importlib.util.spec_from_file_location("bench.perception_noise_curve", mut_path)
mod = importlib.util.module_from_spec(spec)
sys.modules["bench.perception_noise_curve"] = mod
bench.perception_noise_curve = mod
spec.loader.exec_module(mod)
tspec = importlib.util.spec_from_file_location("test_issue172_pnc", test_path)
tgt = importlib.util.module_from_spec(tspec)
tspec.loader.exec_module(tgt)
sys.exit(tgt._run_all_assertions())
'''


def _mutate_selftest() -> int:
    """定点变异自证：对临时副本注入 4 类失效变异，校验精确点红与还原全绿。"""
    orig_src = BENCH_PATH.read_text(encoding="utf-8")

    arms = [
        (
            "M0_clean_baseline",
            None,
            0,
            set(),
        ),
        (
            "M1_short_circuit_misclass",
            ("    if cfg.misclass_rate > 0.0 and rng.random() < cfg.misclass_rate:",
             "    if False and cfg.misclass_rate > 0.0 and rng.random() < cfg.misclass_rate:"),
            1,
            {"test_n2_clean_oracle_and_extreme_degradation", "test_n3_monotonic_degradation_across_all_axes"},
        ),
        (
            "M2_short_circuit_drop",
            ("    if cfg.drop_rate > 0.0 and rng.random() < cfg.drop_rate:",
             "    if False and cfg.drop_rate > 0.0 and rng.random() < cfg.drop_rate:"),
            1,
            {"test_n2_clean_oracle_and_extreme_degradation", "test_n3_monotonic_degradation_across_all_axes"},
        ),
        (
            "M3_short_circuit_jitters",
            ("    if cfg.pos_jitter > 0.0:\n        cx += rng.uniform(-cfg.pos_jitter, cfg.pos_jitter)\n        cy += rng.uniform(-cfg.pos_jitter, cfg.pos_jitter)\n    if cfg.bbox_jitter > 0.0:",
             "    if False and cfg.pos_jitter > 0.0:\n        cx += 0.0\n        cy += 0.0\n    if False and cfg.bbox_jitter > 0.0:"),
            1,
            {"test_n3_monotonic_degradation_across_all_axes"},
        ),
        (
            "M4_disable_bool_nan_gate",
            ("    if isinstance(value, bool):\n        raise ValueError(f\"{name} 不得为 bool，收到 {value!r}\")",
             "    if False and isinstance(value, bool):\n        raise ValueError(f\"{name} 不得为 bool，收到 {value!r}\")"),
            1,
            {"test_n4_invalid_params_fail_closed"},
        ),
        (
            "M5_restored_baseline",
            None,
            0,
            set(),
        ),
    ]

    bad = 0
    with tempfile.TemporaryDirectory(prefix="pnc_mutate_") as tmp:
        mut_file = Path(tmp) / "perception_noise_curve_mut.py"
        for name, patch_spec, expect_rc, expect_fails in arms:
            src = orig_src
            if patch_spec is not None:
                old, new = patch_spec
                cnt = src.count(old)
                assert cnt == 1, "%s 锚点计数异常=%d：%r" % (name, cnt, old)
                src = src.replace(old, new, 1)
            mut_file.write_text(src, encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, "-X", "utf8", "-c", _MUTATION_RUNNER, str(mut_file), os.path.abspath(__file__)],
                cwd=str(REPO),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=RUN_TIMEOUT,
            )
            out = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
            got_fails = set(re.findall(r"(?m)^\s*FAIL\s+(\S+?)(?::|\s|$)", out))
            matched = (proc.returncode == expect_rc) and (got_fails == expect_fails)
            if not matched:
                bad += 1
            print(
                "  %-28s rc=%d(expect=%d) failed=%s(expect=%s) VERDICT=%s"
                % (name, proc.returncode, expect_rc, sorted(got_fails), sorted(expect_fails),
                   "PASS" if matched else "FAIL")
            )
    assert BENCH_PATH.read_text(encoding="utf-8") == orig_src, "主文件被意外改动"
    print("OVERALL=%s (repo_files_untouched=True)" % ("PASS" if bad == 0 else "FAIL"))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    if "--mutate" in sys.argv[1:]:
        sys.exit(_mutate_selftest())
    sys.exit(_run_all_assertions())
