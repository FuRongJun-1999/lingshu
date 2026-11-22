# -*- coding: utf-8 -*-
"""test_hex_train_stratified_and_grad_diag · 路线图 §三 风险对策守卫（梯度自检 + 分层轮换）
========================================================================================
对应《docs/theory/白箱原生网络_长期路线_v1.md》第 65–66 行：
  · 风险 1：符号更新随参数增长变慢 → `train_infogap(..., sampling="stratified")`
    按参数段平滑赤字轮换无放回遍历全量坐标，ceil(D/k) 步内 100% 覆盖且不饿死小层；
  · 风险 2：第二层 dead 区复发 → `diagnose_gradient_health(net, xs, ys)`
    训练前有限差分逐段自检梯度量级与冻结率，给出四态判定（ACCEPT/DEFER/REJECT/BLINDSPOT）。

运行（设 PYTHONPATH 为仓库根）：
    python -X utf8 -m pytest tests/test_hex_train_stratified_and_grad_diag.py -v
    python -X utf8 tests/test_hex_train_stratified_and_grad_diag.py
    python -X utf8 tests/test_hex_train_stratified_and_grad_diag.py --mutate
"""
from __future__ import annotations

import math
import os
import subprocess
import sys

_RP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RP not in sys.path:
    sys.path.insert(0, _RP)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lingshu.nn.hex_train import (  # noqa: E402
    HexNet,
    StratifiedCoordSampler,
    diagnose_gradient_health,
    images_to_lattices,
    normalize_lattices,
    train_infogap,
)

_HEX_TRAIN_PATH = os.path.join(_RP, "lingshu", "nn", "hex_train.py")


def _toy_lat(n=16, seed=3):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for i in range(n):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        if i % 2 == 0:
            img[:, ::4] = 255
        else:
            img[::4, :] = 255
        img += rng.integers(0, 30, img.shape).astype(np.uint8)
        xs.append(img)
        ys.append(i % 2)
    lat = normalize_lattices(images_to_lattices(np.stack(xs)))
    return lat, np.array(ys)


class _DuckMultiLayerNet:
    """含 conv / conv2 / head 三段的多层鸭子类型网络（验证泛化分段自检）。"""

    def __init__(self, seed=5, zero_conv2=False):
        rng = np.random.default_rng(seed)
        self.conv = rng.normal(0.0, 0.4, (2, 7))
        self.conv2 = np.zeros((2, 4), dtype=np.float64) if zero_conv2 else rng.normal(0.0, 0.4, (2, 4))
        self.head = rng.normal(0.0, 0.4, (2, 2))
        self.stacked = True

    def get_vec(self):
        return np.concatenate([self.conv.ravel(), self.conv2.ravel(), self.head.ravel()])

    def set_vec(self, vec):
        v = np.asarray(vec, dtype=np.float64)
        n1, n2 = self.conv.size, self.conv2.size
        self.conv = v[:n1].reshape(self.conv.shape).copy()
        self.conv2 = v[n1:n1 + n2].reshape(self.conv2.shape).copy()
        self.head = v[n1 + n2:].reshape(self.head.shape).copy()

    def loss(self, x, y):
        # 使用 conv 全部 7 维核参数，避免部分坐标人为失活
        flat = x.reshape(len(x), -1)[:, :7]                     # (B, 7)
        feat1 = np.tanh(flat @ self.conv.T)                      # (B, 2)
        h1 = np.concatenate([feat1, feat1 * 0.5], axis=1)        # (B, 4)
        h2 = np.where(h1 @ self.conv2.T > 0, h1 @ self.conv2.T, 0.05 * (h1 @ self.conv2.T))
        logits = h2 @ self.head.T
        z = logits - logits.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return float(-logp[np.arange(len(y)), y].mean())


def test_diagnose_gradient_health_healthy_accept():
    """健康网络训练前自检：各段梯度非零，判定 ACCEPT，且不改写网络参数。"""
    lat, ys = _toy_lat(12, seed=3)
    net = HexNet(n_kernels=4, n_mix=4, n_class=2, seed=7)
    snap = net.get_vec().copy()
    rep = diagnose_gradient_health(net, lat, ys, lr=0.05)
    assert np.array_equal(net.get_vec(), snap), "自检不得改写网络权重"
    assert rep["verdict"] == "ACCEPT", rep
    assert rep["has_dead_layer"] is False and rep["dead_layers"] == []
    assert set(rep["segments"]) == {"conv", "mix", "fc"}
    for name, seg in rep["segments"].items():
        assert seg["probed"] == seg["n_params"] > 0, (name, seg)
        assert seg["grad_abs_mean"] > 0.0, (name, seg)
        assert seg["is_dead_layer"] is False


def test_diagnose_gradient_health_zero_dead_layer_reject():
    """人为构造全零权重 dead 层：dead_zone_ratio == 1.0 且判定 REJECT。"""
    lat, ys = _toy_lat(12, seed=3)

    net_conv = HexNet(n_kernels=4, n_mix=4, n_class=2, seed=7)
    net_conv.conv[:] = 0.0
    rep_conv = diagnose_gradient_health(net_conv, lat, ys, lr=0.05)
    assert rep_conv["verdict"] == "REJECT", rep_conv
    assert rep_conv["has_dead_layer"] is True and "mix" in rep_conv["dead_layers"]
    assert rep_conv["segments"]["mix"]["dead_zone_ratio"] == 1.0

    net = HexNet(n_kernels=4, n_mix=4, n_class=2, seed=7)
    net.mix[:] = 0.0
    rep = diagnose_gradient_health(net, lat, ys, lr=0.05)
    assert rep["verdict"] == "REJECT", rep
    assert rep["has_dead_layer"] is True and len(rep["dead_layers"]) >= 1
    assert rep["segments"]["conv"]["dead_zone_ratio"] == 1.0
    assert rep["segments"]["fc"]["dead_zone_ratio"] == 1.0

    net_fc = HexNet(n_kernels=4, n_mix=4, n_class=2, seed=7)
    net_fc.fc[:] = 0.0
    rep_fc = diagnose_gradient_health(net_fc, lat, ys, lr=0.05)
    assert rep_fc["verdict"] == "REJECT", rep_fc
    assert "conv" in rep_fc["dead_layers"] and "mix" in rep_fc["dead_layers"]


def test_diagnose_gradient_health_duck_type_multilayer_and_explicit_segments():
    """鸭子类型多层网络（含 conv2/head）与显式 segments 均可正确分段自检。"""
    rng = np.random.default_rng(11)
    x = rng.normal(0.0, 1.0, (8, 3, 3, 1))
    y = np.array([0, 1] * 4)

    ok_net = _DuckMultiLayerNet(seed=5, zero_conv2=False)
    rep_ok = diagnose_gradient_health(ok_net, x, y, lr=0.05)
    assert set(rep_ok["segments"]) == {"conv", "conv2", "head"}
    assert rep_ok["verdict"] == "ACCEPT", rep_ok

    dead_net = _DuckMultiLayerNet(seed=5, zero_conv2=True)
    rep_dead = diagnose_gradient_health(dead_net, x, y, lr=0.05)
    assert rep_dead["verdict"] == "REJECT", rep_dead
    assert "conv" in rep_dead["dead_layers"] or "conv2" in rep_dead["dead_layers"]

    total = len(ok_net.get_vec())
    rep_exp = diagnose_gradient_health(
        ok_net, x, y, segments=[("front", 0, 10), ("back", 10, total)],
    )
    assert set(rep_exp["segments"]) == {"front", "back"}


def test_diagnose_gradient_health_partial_freeze_defer_and_empty_blindspot():
    """空样本诚实判 BLINDSPOT；部分冻结偏高（非全死）判 DEFER。"""
    lat, ys = _toy_lat(12, seed=3)
    net = HexNet(n_kernels=4, n_mix=4, n_class=2, seed=7)
    rep_empty = diagnose_gradient_health(net, lat[:0], ys[:0])
    assert rep_empty["verdict"] == "BLINDSPOT" and rep_empty["reason"] == "empty_samples"

    # 取某段梯度中位数作为 eps_freeze 上界，制造部分冻结但非全死
    base_rep = diagnose_gradient_health(net, lat, ys, lr=0.05)
    med_g = base_rep["segments"]["mix"]["grad_abs_median"]
    rep_defer = diagnose_gradient_health(
        net, lat, ys, lr=0.05,
        eps_freeze=max(1e-5, 0.05 * med_g * 1.01),
        defer_freeze_ratio=0.3,
    )
    assert rep_defer["verdict"] in ("DEFER", "REJECT")
    if not rep_defer["has_dead_layer"]:
        assert rep_defer["verdict"] == "DEFER"


def test_stratified_sampler_100_percent_coverage_and_proportional_quota():
    """分层轮换采样器：每步按段配额不饿死小层，ceil(D/budget) 步内覆盖率精确 100%。"""
    segs = [("conv", 0, 28), ("mix", 28, 60), ("fc", 60, 76)]
    d = 76
    budget = 16
    sampler = StratifiedCoordSampler(segs, seed=7)
    steps_for_full = math.ceil(d / budget)
    covered = set()
    for step in range(steps_for_full):
        batch = sampler.next_batch(budget)
        assert len(batch) == budget and len(set(batch.tolist())) == budget
        # 每步三段都必须分到探测配额（小段不被大段饿死）
        assert np.any((0 <= batch) & (batch < 28)), f"step {step} 漏掉 conv"
        assert np.any((28 <= batch) & (batch < 60)), f"step {step} 漏掉 mix"
        assert np.any((60 <= batch) & (batch < 76)), f"step {step} 漏掉 fc"
        covered.update(batch.tolist())
    assert len(covered) == d, f"应在 {steps_for_full} 步内 100% 覆盖 {d} 参数，实得 {len(covered)}"


def test_train_infogap_stratified_coverage_and_loss_drop():
    """train_infogap(..., sampling='stratified') 在 ceil(D/k) 步覆盖 100% 且损失稳定下降。"""
    lat, ys = _toy_lat(16, seed=3)
    net_cov = HexNet(n_kernels=4, n_mix=6, n_class=2, seed=7)
    d = net_cov.n_params()
    k = 16
    steps_full = math.ceil(d / k)
    rep_cov = train_infogap(
        net_cov, lat, ys, lr=0.05, samples_per_step=k,
        dead_zone_ratio=0.0, max_growth=0, max_steps=steps_full,
        seed=7, sampling="stratified",
    )
    assert rep_cov["sampling"] == "stratified"
    assert rep_cov["coord_coverage"] == 1.0, (
        f"stratified 应在 ceil({d}/{k})={steps_full} 步达到 100% 覆盖: {rep_cov}")

    net_tr = HexNet(n_kernels=4, n_mix=6, n_class=2, seed=7)
    init_d = net_tr.loss(lat, ys)
    rep_tr = train_infogap(
        net_tr, lat, ys, lr=0.1, samples_per_step=16,
        dead_zone_ratio=0.0, max_steps=60, seed=7, sampling="stratified",
    )
    assert rep_tr["final_D"] < init_d, f"分层轮换训练损失未下降: {init_d}->{rep_tr['final_D']}"
    assert net_tr.accuracy(lat, ys) >= 0.9


def test_train_infogap_default_uniform_backward_compatible():
    """默认 sampling='uniform' 与显式传 sampling='uniform' 逐位等价。"""
    lat, ys = _toy_lat(12, seed=3)
    net_a = HexNet(n_kernels=3, n_mix=4, n_class=2, seed=7)
    net_b = HexNet(n_kernels=3, n_mix=4, n_class=2, seed=7)
    ra = train_infogap(net_a, lat, ys, max_steps=10, samples_per_step=12, seed=5)
    rb = train_infogap(net_b, lat, ys, max_steps=10, samples_per_step=12, seed=5, sampling="uniform")
    assert np.array_equal(net_a.get_vec(), net_b.get_vec())
    assert ra["final_D"] == rb["final_D"] and ra["history"] == rb["history"]


def test_invalid_params_fail_loud():
    """非法 sampling 名称、NaN/±inf/bool/越界参数一律 fail-loud 抛 ValueError。"""
    lat, ys = _toy_lat(8, seed=3)
    net = HexNet(n_kernels=2, n_mix=3, n_class=2, seed=7)

    with pytest.raises(ValueError):
        train_infogap(net, lat, ys, sampling="invalid_mode")
    for kw in (
        {"lr": float("nan")}, {"lr": float("inf")}, {"lr": -0.1}, {"lr": True},
        {"eps": 0.0}, {"eps": float("nan")}, {"eps": True},
        {"samples_per_step": 0}, {"samples_per_step": True}, {"samples_per_step": 2.5},
        {"max_steps": True}, {"max_steps": 1.5}, {"batch": 0}, {"seed": -1},
    ):
        with pytest.raises(ValueError):
            train_infogap(net, lat, ys, **kw)

    for kw in (
        {"lr": 0.0}, {"lr": float("nan")}, {"lr": True},
        {"eps": float("inf")}, {"eps_freeze": 0.0}, {"eps_freeze": True},
        {"defer_freeze_ratio": 0.0}, {"defer_freeze_ratio": 1.5}, {"defer_freeze_ratio": float("nan")},
        {"max_probes_per_segment": 0}, {"max_probes_per_segment": True},
    ):
        with pytest.raises(ValueError):
            diagnose_gradient_health(net, lat, ys, **kw)

    bad_lat = lat.copy()
    bad_lat[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        diagnose_gradient_health(net, bad_lat, ys)
    with pytest.raises(ValueError):
        diagnose_gradient_health(net, lat, ys[:-1])


# ==================== --mutate 定点变异自证 ====================

_MUTATIONS = [
    (
        "M1_bypass_dead_layer_reject",
        "    if dead_layers:\n"
        '        verdict = "REJECT"\n'
        '        reason = "dead_layer_detected:" + ",".join(dead_layers)',
        "    if False and dead_layers:\n"
        '        verdict = "REJECT"\n'
        '        reason = "dead_layer_detected:" + ",".join(dead_layers)',
        ("test_diagnose_gradient_health_zero_dead_layer_reject",
         "test_diagnose_gradient_health_duck_type_multilayer_and_explicit_segments"),
    ),
    (
        "M2_bypass_stratified_coverage",
        "        if sampler is not None:\n"
        "            if len(vec) != sampler.total_params:\n"
        "                sampler = StratifiedCoordSampler(\n"
        "                    _infer_param_segments(net, len(vec)), seed=seed + 10007 + step)\n"
        "            sample = sampler.next_batch(samples_per_step)",
        "        if False and sampler is not None:\n"
        "            sample = sampler.next_batch(samples_per_step)",
        ("test_train_infogap_stratified_coverage_and_loss_drop",),
    ),
    (
        "M3_bypass_sampling_mode_gate",
        "    if sampling not in _VALID_SAMPLING_MODES:\n"
        "        raise ValueError(\n"
        '            f"sampling 须为 {_VALID_SAMPLING_MODES!r} 之一，实得 {sampling!r}")',
        "    if sampling is None:\n"
        '        raise ValueError("none")',
        ("test_invalid_params_fail_loud",),
    ),
]


def _run_plain():
    return pytest.main([os.path.abspath(__file__), "-q"])


def _run_mutate():
    with open(_HEX_TRAIN_PATH, "rb") as fh:
        raw_orig = fh.read()
    has_crlf = b"\r\n" in raw_orig
    src_lf = raw_orig.decode("utf-8").replace("\r\n", "\n")

    rc = 0
    for name, old_s, new_s, expect_tests in _MUTATIONS:
        hits = src_lf.count(old_s)
        if hits != 1:
            print(f"[ANCHOR-MISS] {name}  命中 {hits} 处（应为 1）")
            return 1
        mutated_lf = src_lf.replace(old_s, new_s)
        mutated_bytes = (mutated_lf.replace("\n", "\r\n") if has_crlf else mutated_lf).encode("utf-8")
        try:
            with open(_HEX_TRAIN_PATH, "wb") as fh:
                fh.write(mutated_bytes)
            proc = subprocess.run(
                [sys.executable, "-X", "utf8", "-m", "pytest",
                 os.path.abspath(__file__), "-q"],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", cwd=_RP,
            )
            out = (proc.stdout or "") + "\n" + (proc.stderr or "")
            red = proc.returncode != 0
        finally:
            with open(_HEX_TRAIN_PATH, "wb") as fh:
                fh.write(raw_orig)
        if not red:
            print(f"[MUTATE-FAIL] {name}  变异后测试仍绿——守卫无牙")
            rc = 1
        else:
            hit_names = [t for t in expect_tests if t in out]
            if not hit_names:
                print(f"[MUTATE-FAIL] {name}  虽转红但未点名预期用例 {expect_tests}:\n{out}")
                rc = 1
            else:
                print(f"[MUTATE-OK] {name}  精准点红：{','.join(hit_names)}")
    return rc


if __name__ == "__main__":
    if "--mutate" in sys.argv:
        sys.exit(_run_mutate())
    sys.exit(_run_plain())
