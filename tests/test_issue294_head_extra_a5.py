# -*- coding: utf-8 -*-
"""守卫 · #294 症状③：`deep_norm and amp_sep` 时 head_in 多一列，采样空间必须跟上

## 缺陷（剩余的那一项；症状①②已由前序修复收口）

`make_plan` 的参数量算式 `k1*7 + k1*k2*7 + 9*(k2+3)` 假设 `head_in = k2+3`。
但 `hex_hier.HexHierNet` 在 `deep_norm and amp_sep` 时 `head_in = n_kernels2+3+1`
（`hex_hier.py:134`）⇒ 真实 `n_params = 375+9 = 384`，而 plan 采样上界仍是 374：
**尾部 9 个参数（索引 375..383）永远不被采样**，训练 30 步 head 变化 0/99。

issue 原文读数（main `cc3888b`）：
    {deep_norm, amp_sep}  n_params=384  采样上界=374  永不采样参数=9
预期（按修复建议）：
    {deep_norm, amp_sep}  n_params=384  采样上界=383  永不采样参数=0

## 修法（最小，沿用本文件既有的 k1/k2 口径）

`make_plan(..., n_params: Optional[int] = None)`：显式给出时以其为准（权威规模），
`None` 时退回旧算式 ⇒ 既有调用与 `K1=K2=6` 的 Rust 等价基线**逐位不变**。
两个拿得到 net 的调用点（`equivalence_par` / `equivalence_check`）改传 `net.n_params()`。

## 断言组

| ID | 判据 |
|---|---|
| H1 | `deep_norm+amp_sep`：采样上界 == `net.n_params()-1`（384-1=383），且不越界 |
| H2 | 该配置下采样并集能覆盖**全部** 384 个参数（不再有永不采样的尾部 9 个） |
| H3 | 默认 (6,6)：仍是 375 / 374 —— 防过修（旧口径逐位不变） |
| H4 | `stacked=False` 同病：上界必须 == `net.n_params()-1`（算式同样不覆盖） |
| H5 | 两个拿 net 的调用点确实把 `n_params` 传进去（防「签名接了、值没传」假修复） |
| H6 | 非法 `n_params`（0 / 负数）显式拒绝，不静默采样空集 |

## 变异（自带，非仓库原生机制）

    python -X utf8 tests/test_issue294_head_extra_a5.py --mutate

把 `n_params` 形参摘掉（回滚到「只有 k1/k2 算式」的缺陷形态），预期 H1/H2/H4/H5/H6 红、
H3 绿（默认配置本就 375/374，缺陷形态下同样成立 ⇒ 直接证明判别力落在新增那一格）。
跑完按原文还原并校验 sha256。

运行（仓根）：python -X utf8 -m pytest tests/test_issue294_head_extra_a5.py -q --no-header
"""
from __future__ import annotations

import hashlib
import os
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


def _plan_for(net, n, samples, **kw):
    """优先用权威规模；旧签名（无 n_params 形参）退回缺陷形态以便断言给出红。"""
    try:
        return rb.make_plan(n, 1, 2, samples, 7,
                            k1=net.K, k2=net.K2, n_params=net.n_params(), **kw)
    except TypeError:
        return rb.make_plan(n, 1, 2, samples, 7, k1=net.K, k2=net.K2, **kw)


def test_h1_deep_norm_amp_sep_upper_bound():
    """H1：deep_norm+amp_sep ⇒ 采样上界 == 真实参数量-1（384-1=383）。"""
    net = HexHierNet(n_kernels=6, n_kernels2=6, deep_norm=True, amp_sep=True, seed=1)
    real = net.n_params()
    assert real == 384, "夹具前提变了：该配置参数量应为 384，实测 %d" % real
    plan = _plan_for(net, 4, samples=real + 10)
    plan_max = max(max(s) for s in plan["plan_sample"])
    assert plan_max == real - 1, (
        "deep_norm+amp_sep：采样上界 %d != 真实参数量-1 %d（#294 症状③未修："
        "make_plan 的 head 项没算 deep_norm+amp_sep 的 +1 列）" % (plan_max, real - 1))
    assert plan_max < real, "越界：上界 %d >= 参数量 %d" % (plan_max, real)


def test_h2_no_never_sampled_tail():
    """H2：该配置下采样并集覆盖全部参数（issue 实测：尾部 9 个永不采样）。"""
    net = HexHierNet(n_kernels=6, n_kernels2=6, deep_norm=True, amp_sep=True, seed=1)
    real = net.n_params()
    plan = _plan_for(net, 4, samples=real)          # 一次取满 ⇒ 并集应等于全集
    idx = set(i for s in plan["plan_sample"] for i in s)
    never = sorted(set(range(real)) - idx)
    assert not never, "仍有 %d 个参数永不采样：%r（#294 症状③）" % (len(never), never[:12])
    assert not [i for i in idx if i >= real], "plan 给出越界下标"


def test_h3_default_unchanged():
    """H3：默认 (6,6) 仍是 375/374 —— 防过修。"""
    net = HexHierNet(n_kernels=6, n_kernels2=6, seed=1)
    assert net.n_params() == 375, "默认参数量变了：%d" % net.n_params()
    plan = _plan_for(net, 4, samples=400)
    assert max(max(s) for s in plan["plan_sample"]) == 374, "默认 plan 上界变了（应 374）"
    # 不传 n_params 时也必须仍是旧口径
    plan2 = rb.make_plan(4, 1, 2, 400, 7, k1=6, k2=6)
    assert max(max(s) for s in plan2["plan_sample"]) == 374, "n_params=None 时旧口径被破坏"


def test_h4_unstacked_same_disease():
    """H4：stacked=False 时 head_in=K1+3，算式同样不覆盖 ⇒ 必须也走权威值。"""
    net = HexHierNet(n_kernels=6, n_kernels2=6, stacked=False, seed=1)
    real = net.n_params()
    plan = _plan_for(net, 4, samples=real + 10)
    plan_max = max(max(s) for s in plan["plan_sample"])
    assert plan_max == real - 1, (
        "stacked=False：采样上界 %d != 真实参数量-1 %d（#294 同病另一格）" % (plan_max, real - 1))


def test_h5_call_sites_pass_authoritative_nparams(monkeypatch):
    """H5：两个拿得到 net 的调用点必须传 n_params（防「签名接了、值没传」）。"""
    seen = []

    def fake_plan(n, steps, batch, samples, seed, k1=6, k2=6, n_params=None):
        seen.append((k1, k2, n_params))
        return {"plan_batch": [[0] * batch for _ in range(steps)],
                "plan_sample": [[0] * samples for _ in range(steps)]}

    curve = np.zeros(2)
    monkeypatch.setattr(rb, "make_plan", fake_plan)
    monkeypatch.setattr(rb, "py_executor_par", lambda *a, **k: curve)
    monkeypatch.setattr(rb, "py_executor", lambda *a, **k: curve)
    monkeypatch.setattr(rb, "run_rust_train", lambda *a, **k: (curve, None, 0.0))

    net = HexHierNet(n_kernels=6, n_kernels2=6, deep_norm=True, amp_sep=True, seed=1)
    real = net.n_params()
    lat = np.zeros((4, 3, 4, 3))
    obj = np.zeros(4, dtype=np.int64)
    shp = np.zeros(4, dtype=np.int64)

    bad = []
    for name, fn in (("equivalence_par", lambda: rb.equivalence_par(net, lat, obj, shp, steps=1)),
                     ("equivalence_check", lambda: rb.equivalence_check(net, lat, obj, shp, steps=1))):
        seen.clear()
        fn()
        if not seen:
            bad.append("%s: 未调用 make_plan" % name)
        elif seen[-1][2] != real:
            bad.append("%s: make_plan 收到 n_params=%r，期望 %d（真实参数量）"
                       % (name, seen[-1][2], real))
    assert not bad, "调用点未透传权威规模（假修复）：\n  " + "\n  ".join(bad)


def test_h6_invalid_nparams_rejected():
    """H6：非法 n_params 显式拒绝，不静默采空集。"""
    for bad in (0, -1):
        with pytest.raises(ValueError):
            rb.make_plan(4, 1, 2, 8, 7, k1=6, k2=6, n_params=bad)


# ---------------------------------------------------------------- --mutate

def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _revert_source():
    """把 n_params 形参与分支摘掉 —— 回到「只有 k1/k2 算式」的缺陷形态。"""
    orig = open(RB_PATH, encoding="utf-8", newline="").read()
    eol = "\r\n" if "\r\n" in orig else "\n"
    lines = orig.split(eol)
    i = next(k for k, l in enumerate(lines) if l.strip().startswith("if n_params is None:"))
    # 摘掉 6 行分支（if / 赋值 / if <=0 / raise / 赋值），换成原式
    j = next(k for k, l in enumerate(lines) if l.strip().startswith("n_params = int(n_params)"))
    mut = lines[:i] + [
        '    n_params = k1 * 7 + k1 * k2 * 7 + 9 * (k2 + 3)',
    ] + lines[j + 1:]
    # 签名去掉 n_params 形参
    sig = next(k for k, l in enumerate(mut) if "n_params: Optional[int] = None) -> Dict:" in l)
    mut[sig - 1] = mut[sig - 1].replace("k1: int = 6, k2: int = 6,", "k1: int = 6, k2: int = 6) -> Dict:")
    del mut[sig]
    open(RB_PATH, "w", encoding="utf-8", newline="").write(eol.join(mut))

    def restore():
        open(RB_PATH, "w", encoding="utf-8", newline="").write(orig)
    return restore


def main_mutate():
    before = _sha(RB_PATH)
    restore = _revert_source()
    try:
        p = subprocess.run([sys.executable, "-X", "utf8", "-m", "pytest",
                            os.path.abspath(__file__), "-q", "--tb=no", "-p", "no:cacheprovider"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace",
                           cwd=REPO, timeout=300)
        out = (p.stdout or "") + (p.stderr or "")
        fails = sorted(set(l.split("::")[-1].split(" ")[0]
                           for l in out.splitlines() if l.startswith("FAILED")))
        print("MUTANT_FAILED=" + repr(fails))
        print("MUTANT_RC=" + str(p.returncode))
        ok = (p.returncode != 0 and "test_h3_default_unchanged" not in fails
              and len(fails) >= 4)
        print("MUTATION_DISCRIMINATIVE=" + ("YES" if ok else "NO"))
        return 0 if ok else 1
    finally:
        restore()
        print("RESTORED_SHA_OK=" + str(_sha(RB_PATH) == before))


if __name__ == "__main__":
    if "--mutate" in sys.argv:
        sys.exit(main_mutate())
    print("use: --mutate")
