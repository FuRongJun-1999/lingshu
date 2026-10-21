# -*- coding: utf-8 -*-
"""守卫 · #294：`rust_bridge.make_plan` 的支路采样上限必须随 k1/k2 走

缺陷（main 8eff4c7，`lingshu/nn/rust_bridge.py::make_plan`）
    原实现把参数量上限硬编码成 `6*7 + 6*6*7 + 9*(6+3)` = 375，只在 `K1=K2=6`
    时与真实参数量巧合相等（预审卡记作 `K6=PASS_BY_COINCIDENCE`）。
    `K1/K2` 取其它值时 `plan_sample` 会产出越界下标，下游 `py_executor_par`
    的 `v[pi]` 直接 `IndexError` —— **硬崩，不是静默**。

正确口径（与 `hex_hier.HexHierNet.n_params()` 对齐）
    `n_params = K1*7 + K2*K1*7 + 9*(K2+3)`
    其中 `9 = len(OBJ)`、`K2+3 = head_in`（`hex_hier.py` head_in = n_kernels2+3）。
    **注意最容易写错的一格**：head 项用 `k2`，不是 `k1` —— `9*(k1+3)` 在
    `k1==k2` 时同样得 375，缺陷会原样藏回去（M2 臂专抓这一格）。

断言组
    P1 三组 (k1,k2) 下 `make_plan` 的采样上限 == `HexHierNet.n_params()-1`，且不越界
    P2 真实跑 `py_executor_par` 不抛 `IndexError`（复现原始崩溃路径）
    P3 五个调用点**都**把 k1/k2 传进去（防「签名接了、值没传」的假修复）
    P4 默认配置 (6,6) 行为不变（375 / 374）

变异四臂（本文件自带，非仓库原生机制）
    python -X utf8 tests/test_issue294_plan_nparams_k1k2_guard.py --mutate
    M1 调用点漏传（bench_rust 内不带 k1/k2）  ⇒ 仅 P3 红
    M2 head 项写回 k1（`9*(k1+3)`）           ⇒ P1[4-5]、P1[8-3] 红；P1[6-6]、P2、P4 绿
       （实测修正：该式**低估**参数量 231<240，采样上限反而更小 ⇒ 不越界，P2 合理绿；
         高估方向才由 P2 抓，低估方向由 P1 的精确等值抓。原预注册多写了 P2。）
    M3 上限多算 1（`n_params + 1`）            ⇒ 全部 P1、P2、P4 红
    每臂跑完按 sha256 校验 rust_bridge.py 原样还原。

运行（仓根）：python -X utf8 -m pytest tests/test_issue294_plan_nparams_k1k2_guard.py -q --no-header
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import lingshu.nn.rust_bridge as rb                      # noqa: E402
from lingshu.nn.hex_hier import HexHierNet               # noqa: E402

RB_PATH = os.path.join(REPO, "lingshu", "nn", "rust_bridge.py")
RUN_TIMEOUT = 300

CONFIGS = ((6, 6), (4, 5), (8, 3))


def _fixture(k1, k2):
    """(net, lat, obj_idx, shape_idx)：lat 为 (B, rows, cols, C=3)。"""
    net = HexHierNet(n_kernels=k1, n_kernels2=k2, seed=1)
    lat = np.zeros((4, 3, 4, 3))
    obj = np.zeros(4, dtype=np.int64)
    shp = np.zeros(4, dtype=np.int64)
    return net, lat, obj, shp


def _plan(k1, k2, samples):
    """旧签名兜底：锚点树没有 k1/k2 形参时退回写死 375 的那版。"""
    try:
        return rb.make_plan(4, 1, 2, samples, 7, k1=k1, k2=k2)
    except TypeError:
        return rb.make_plan(4, 1, 2, samples, 7)


@pytest.mark.parametrize("k1,k2", CONFIGS, ids=["%d-%d" % c for c in CONFIGS])
def test_p1_plan_bound_matches_net_nparams(k1, k2):
    """P1：采样上限必须等于真实参数量减一（不越界、也不保守到浪费）。"""
    net, _lat, _obj, _shp = _fixture(k1, k2)
    real = net.n_params()
    plan = _plan(k1, k2, samples=real + 10)
    plan_max = max(max(s) for s in plan["plan_sample"])
    assert plan_max == real - 1, (
        "k1=%d k2=%d：plan 采样上限 %d != 真实参数量-1 %d（#294 上限口径未随 k1/k2 走）"
        % (k1, k2, plan_max, real - 1))
    assert plan_max < real, (
        "k1=%d k2=%d：plan 采样上限 %d 越界（真实参数量 %d）" % (k1, k2, plan_max, real))


def test_p2_py_executor_no_index_error():
    """P2：真实跑 py_executor_par 不得因越界下标抛 IndexError（原始崩溃路径）。"""
    net, lat, obj, shp = _fixture(4, 5)
    real = net.n_params()
    plan = _plan(4, 5, samples=real + 10)
    try:
        rb.py_executor_par(lat, obj, shp, net.get_vec().copy(), plan, net)
    except IndexError as exc:
        pytest.fail("k1=4 k2=5 下 py_executor_par 越界（#294 未修）：%s" % exc)


def test_p3_all_five_call_sites_forward_k1_k2(monkeypatch):
    """P3：五个调用点都必须把 k1/k2 传进 make_plan（防假修复）。"""
    seen = []
    curve = np.zeros(2)

    def fake_plan(n, steps, batch, samples, seed, k1=6, k2=6):
        seen.append((k1, k2))
        return {"plan_batch": [[0] * batch for _ in range(steps)],
                "plan_sample": [[0] * samples for _ in range(steps)]}

    monkeypatch.setattr(rb, "make_plan", fake_plan)
    monkeypatch.setattr(rb, "py_executor_par", lambda *a, **k: curve)
    monkeypatch.setattr(rb, "py_executor", lambda *a, **k: curve)
    monkeypatch.setattr(rb, "run_rust_train", lambda *a, **k: (curve, None, 0.0))

    net, lat, obj, shp = _fixture(4, 5)
    vec0 = net.get_vec().copy()
    calls = {
        "equivalence_par": lambda: rb.equivalence_par(net, lat, obj, shp, steps=1),
        "equivalence_check": lambda: rb.equivalence_check(net, lat, obj, shp, steps=1),
        "determinism_check": lambda: rb.determinism_check(
            lat, obj, shp, vec0, steps=1, threads_list=(1,), k1=4, k2=5),
        "bench_threads": lambda: rb.bench_threads(
            lat, obj, shp, vec0, steps=1, threads_list=(1,), k1=4, k2=5),
        "bench_rust": lambda: rb.bench_rust(
            lat, obj, shp, vec0, steps=1, batch=2, samples=3, k1=4, k2=5),
    }
    bad = []
    for name, fn in calls.items():
        seen.clear()
        try:
            fn()
        except TypeError as exc:
            bad.append("%s: 不接受 k1/k2 形参（%s）" % (name, exc))
            continue
        if not seen:
            bad.append("%s: 未调用 make_plan" % name)
        elif seen[-1] != (4, 5):
            bad.append("%s: make_plan 收到 %r，期望 (4, 5)" % (name, seen[-1]))
    assert not bad, "调用点未透传 k1/k2（假修复）：\n  " + "\n  ".join(bad)


def test_p4_default_config_unchanged():
    """P4：默认 (6,6) 行为不变——375 参数、上限 374。"""
    net, _lat, _obj, _shp = _fixture(6, 6)
    assert net.n_params() == 375, "默认配置参数量变了：%d" % net.n_params()
    plan = _plan(6, 6, samples=400)
    plan_max = max(max(s) for s in plan["plan_sample"])
    assert plan_max == 374, "默认配置 plan 上限变了：%d（应 374）" % plan_max


# ---------------------------------------------------------------- --mutate

_MUTATIONS = (
    ("M1_drop_forward", "bench_rust 漏传 k1/k2",
     (("""               seed: int = 11, k1: int = 6, k2: int = 6) -> Dict:
    plan = make_plan(len(lat), steps, batch, samples, seed, k1=k1, k2=k2)""",
       """               seed: int = 11, k1: int = 6, k2: int = 6) -> Dict:
    plan = make_plan(len(lat), steps, batch, samples, seed)"""),),
     {"test_p3_all_five_call_sites_forward_k1_k2"}),

    ("M2_head_uses_k1", "head 项写回 k1",
     (("n_params = k1 * 7 + k1 * k2 * 7 + 9 * (k2 + 3)",
       "n_params = k1 * 7 + k1 * k2 * 7 + 9 * (k1 + 3)"),),
     {"test_p1_plan_bound_matches_net_nparams[4-5]",
      "test_p1_plan_bound_matches_net_nparams[8-3]"}),
     # 实测修正：head 项用 k1 是「低估」（4,5 → 231 < 240），不会越界，P2 保持绿。
     # 越界只在「高估」方向出现（M3）；低估由 P1 的精确等值抓。

    ("M3_off_by_one", "上限多算 1",
     (("n_params = k1 * 7 + k1 * k2 * 7 + 9 * (k2 + 3)",
       "n_params = k1 * 7 + k1 * k2 * 7 + 9 * (k2 + 3) + 1"),),
     {"test_p1_plan_bound_matches_net_nparams[6-6]",
      "test_p1_plan_bound_matches_net_nparams[4-5]",
      "test_p1_plan_bound_matches_net_nparams[8-3]",
      "test_p2_py_executor_no_index_error",
      "test_p4_default_config_unchanged"}),
)


def _run_pytest():
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    cmd = [sys.executable, "-X", "utf8", "-m", "pytest", os.path.abspath(__file__),
           "-q", "--no-header", "-p", "no:cacheprovider", "-rf"]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=RUN_TIMEOUT,
                          env=env)
    out = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
    red = set()
    for line in out.splitlines():
        m = re.match(r"^(?:FAILED|ERROR) .*::(\S+)", line)
        if m:
            red.add(m.group(1))
    summary = [l for l in out.splitlines() if " passed" in l or " failed" in l]
    return red, (summary[-1] if summary else "")


def _mutate():
    orig = open(RB_PATH, "rb").read()
    digest = hashlib.sha256(orig).hexdigest()
    text = orig.decode("utf-8")
    bad = 0
    for name, desc, repls, expect in _MUTATIONS:
        try:
            mutated = text
            for old, new in repls:
                hits = mutated.count(old)
                assert hits == 1, "%s 变异锚点命中 %d 次（应 1）：%r" % (name, hits, old[:60])
                mutated = mutated.replace(old, new, 1)
            open(RB_PATH, "wb").write(mutated.encode("utf-8"))
            red, summary = _run_pytest()
            print("  %-16s %s" % (name, desc))
            print("    期望点红=%d 实际点红=%d %s" % (len(expect), len(red), summary))
            if red != expect:
                bad += 1
                print("    VERDICT=FAIL 多红=%r 漏红=%r"
                      % (sorted(red - expect), sorted(expect - red)))
            else:
                print("    VERDICT=PASS 红集=%r" % (sorted(red),))
        finally:
            open(RB_PATH, "wb").write(orig)
            now = hashlib.sha256(open(RB_PATH, "rb").read()).hexdigest()
            assert now == digest, "rust_bridge.py 未原样还原"
    print("  OVERALL=%s" % ("PASS" if bad == 0 else "FAIL",))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    if "--mutate" in sys.argv:
        sys.exit(_mutate())
    sys.exit(0)
