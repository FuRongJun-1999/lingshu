# -*- coding: utf-8 -*-
"""#197 守卫：参数向量「读-改-写」必须经原子接口 apply_delta，且该接口必须真的互斥。

缺陷（lingshu issue #197，判定 REPRODUCES）：`get_vec`/`set_vec` 是两次独立调用，
两线程各自取快照、各自 +1、各自写回 ⇒ 后写者覆盖先写者，**静默丢一次更新**（不抛异常）。

★★ 判别力设计（第一版守卫在这里失败过，见 mutation.txt 的 M2）：
   只在 `apply_delta` **之前**放 Barrier 是不够的 —— 两线程的 get→set 窗口极小，
   即使把锁摘掉也几乎撞不上，变异后守卫照样全绿（**假绿**）。
   正确做法是把 Barrier 放进 `fn` **内部**（即 RMW 的正中间）：

     · 有锁：一个线程持锁进入 fn 并卡在 Barrier，其余线程进不来 ⇒ Barrier 超时
       （BrokenBarrierError，**这是"锁生效"的预期信号**）⇒ 各线程顺序完成，增量 = 线程数。
     · 无锁：全部线程都已各自 `get_vec()` 拿到同一份旧值后才进入 fn，Barrier 放行 ⇒
       所有线程写回同一份「旧值+1」⇒ 增量 = 1（丢更新）⇒ 守卫红。

   两种情形都是**确定性**的，不依赖时序运气。

★ 探针下标按类选取：`UnitHeadNet.set_vec` 对 gain 段做 `np.clip(v[:k], 0.0, GAIN_BOUND)`
  **截断**，对它累加会饱和到上界 —— 那是合法的有界声明，不是原子性失败。
  故 UnitHeadNet 用**未被截断**的 theta 段作探针。

运行（仓根）：python -X utf8 -m pytest tests/test_issue197_vec_rmw_atomic_guard.py -q --no-header
"""
from __future__ import annotations

import os
import sys
import threading

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.nn.hex_train import HexNet          # noqa: E402
from lingshu.nn.hex_hier import HexHierNet       # noqa: E402
from lingshu.nn.hex_unit import UnitHeadNet      # noqa: E402
from lingshu.nn.hex_cnn import DEFAULT_KERNELS   # noqa: E402

CLASSES = ("HexNet", "HexHierNet", "UnitHeadNet")
BARRIER_TIMEOUT = 1.0


def _mk(cls_name):
    """构造被测实例。UnitHeadNet 的前置用真实 HexHierNet（与仓内既有用法一致：
    tests/test_unit_direction.py / tests/test_hex_train.py 都是 `UnitHeadNet(HexHierNet())`）。
    """
    if cls_name == "HexNet":
        return HexNet(n_kernels=4, n_mix=8, seed=7)
    if cls_name == "HexHierNet":
        return HexHierNet(n_kernels=4, n_kernels2=4, seed=7)
    if cls_name == "UnitHeadNet":
        return UnitHeadNet(HexHierNet(n_kernels=4, n_kernels2=4, seed=7), top_k=2)
    raise ValueError(cls_name)


def _probe_index(net, cls_name):
    """返回一个**不被 set_vec 截断**的探针下标。"""
    if cls_name in ("HexNet", "HexHierNet"):
        return 0
    return int(net.mask.sum())          # UnitHeadNet 的 theta 段起点


def _rmw_with_inner_barrier(idx, barrier, counters):
    """返回一个 fn：在 RMW 正中间卡 Barrier，从而确定性放大交错窗口。"""

    def _fn(v):
        try:
            barrier.wait(timeout=BARRIER_TIMEOUT)
            counters["released"] += 1
        except threading.BrokenBarrierError:
            # 有锁时必然走到这里：其余线程被锁挡在临界区外 ⇒ Barrier 凑不齐
            counters["timeout"] += 1
        v = v.copy()
        v[idx] = v[idx] + 1.0
        return v

    return _fn


def _run_concurrent(cls_name, n_threads):
    net = _mk(cls_name)
    idx = _probe_index(net, cls_name)
    v0 = float(net.get_vec()[idx])
    barrier = threading.Barrier(n_threads)
    counters = {"released": 0, "timeout": 0}

    def worker():
        net.apply_delta(_rmw_with_inner_barrier(idx, barrier, counters))

    ts = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    delta = float(net.get_vec()[idx]) - v0
    return delta, counters


# ---------------- J1：判别力核心 ----------------
def _j1(cls_name, repeats=2):
    for r in range(repeats):
        delta, counters = _run_concurrent(cls_name, 2)
        assert abs(delta - 2.0) < 1e-9, (
            f"{cls_name} 第 {r+1} 次：2 线程 apply_delta(+1)（Barrier 在 RMW 中间）"
            f"增量 = {delta}，期望 2.0 —— 丢更新 ⇒ 临界区未互斥"
            f"（counters={counters}）"
        )


def test_j1_two_threads_apply_delta_is_atomic_hexnet():
    _j1("HexNet")


def test_j1_two_threads_apply_delta_is_atomic_hexhiernet():
    _j1("HexHierNet")


def test_j1_two_threads_apply_delta_is_atomic_unitheadnet():
    _j1("UnitHeadNet")


# ---------------- J2：规模留存 ----------------
def test_j2_many_threads_no_update_lost():
    n_threads = 8
    for name in CLASSES:
        delta, counters = _run_concurrent(name, n_threads)
        assert abs(delta - float(n_threads)) < 1e-9, (
            f"{name}：{n_threads} 线程各 1 次 apply_delta，增量 = {delta}，"
            f"期望 {n_threads}（counters={counters}）"
        )


# ---------------- J3：声明可检查 ----------------
def test_j3_declarations_are_present_and_checkable():
    for name in CLASSES:
        net = _mk(name)
        assert hasattr(net, "apply_delta"), f"{name} 缺原子接口 apply_delta"
        gd = net.get_vec.__doc__ or ""
        ad = net.apply_delta.__doc__ or ""
        assert "非原子" in gd, f"{name}.get_vec 未声明「非原子」"
        assert "apply_delta" in gd, f"{name}.get_vec 未指向原子接口 apply_delta"
        assert "apply_delta" in ad and "原子" in ad, f"{name}.apply_delta 未声明原子性"
        assert ("非原子" in ad) or ("成对" in ad), (
            f"{name}.apply_delta 未如实声明「裸用 get/set 仍非原子」的边界"
        )


# ---------------- J4：单线程语义不变 / 不别名 ----------------
def test_j4_single_thread_semantics_unchanged_and_no_alias():
    for name in CLASSES:
        a = _mk(name)
        b = _mk(name)
        assert np.array_equal(a.get_vec(), b.get_vec()), f"{name}: 同种子初值不逐位一致"

        idx = _probe_index(a, name)
        ref = a.get_vec()
        ref[idx] += 1.0
        a.set_vec(ref)

        delta_vec = np.zeros(b.get_vec().size, dtype=np.float64)
        delta_vec[idx] = 1.0
        b.apply_delta(delta_vec)
        assert np.array_equal(a.get_vec(), b.get_vec()), (
            f"{name}: apply_delta(ndarray) 与 get/+=/set 不等价"
        )

        # 不别名调用方数组：外部改 delta 不得影响模型
        c = _mk(name)
        cidx = _probe_index(c, name)
        dvec = np.zeros(c.get_vec().size, dtype=np.float64)
        c.apply_delta(dvec)
        dvec[cidx] = 999.0
        assert abs(float(c.get_vec()[cidx]) - 999.0) > 1e-9, f"{name}: apply_delta 别名了调用方数组"

        # 可调用对象返回 None（原地改传入副本）也生效
        d = _mk(name)
        didx = _probe_index(d, name)
        d0 = float(d.get_vec()[didx])

        def _none_fn(v):
            v[didx] = v[didx] + 2.0
            return None

        d.apply_delta(_none_fn)
        assert abs(float(d.get_vec()[didx]) - (d0 + 2.0)) < 1e-9, (
            f"{name}: fn 返回 None 未生效"
        )


def test_j5_handwritten_family_constant_untouched():
    """非目标面不受影响：DEFAULT_KERNELS 仍是 5 个。"""
    assert len(DEFAULT_KERNELS) == 5