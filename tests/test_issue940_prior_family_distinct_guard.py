# -*- coding: utf-8 -*-
"""test_issue940_prior_family_distinct_guard · #940 先验核族互异性守卫
============================================================================
缺陷（lingshu issue #940 · P3 · F23-01）：
`lingshu/nn/hex_cnn.py` 的 `extended_prior_family` docstring 声称
「n 个**互异**白箱先验核(每核唯一,无对称重复)」，但扩容变体的旧实现是
    base = fam[i % 10].copy() + (i + 1) * 1e-3
    fam.append(base / (np.abs(base).sum() + 1e-9))
对**全正全等的均匀基核**（`smooth` = 1/7 均匀向量）加一个正常数后仍是均匀向量，
L1 归一恒得同一向量、与常数大小无关 ⇒ `extended_prior_family(37)` 中
idx 6/16/26/36 逐位相同（4 个同形、6 对重复）。#339 想消灭的「重复核」以变种复活。
`n_kernels ≥ 17` / `K > 16` 可达，两参数均无上界校验（默认 K≤7 不可达）。

修法：变体改为**非均匀、随序号变化的确定性微扰**——图案 `(-1)**(k+i)*(k+1)`
（正负混合、相位随 i 轮转，对均匀基核叠加后必然非均匀），幅度 `1e-3*(1+i//10)`
每 10 个变体递增一档（同一基核的后续变体靠幅度区分）；末尾再做**互异性
fail-closed 校验**，真出现同形核直接 `ValueError`，不再静默返回重复核。
判据来源：公开 issue #940 正文（机制与位置）＋ 判定方 def-impl-1b 的复现读数。

守卫断言组（**只回滚 `hex_cnn.py` 的 `extended_prior_family` 函数体 ⇒ A/B 必红**）：
  A 组（核心判据 · 有牙）：`extended_prior_family(37)` 任意两核不得逐位相同
     —— 旧实现 idx 6/16/26/36 同形 ⇒ 6 对重复，直接红。
  B 组（扩容面 · 有牙）：n = 12 / 17 / 37 / 50 全族互异且族大小 == n
     —— 覆盖 issue 指出的 n≥17 可达区间与更远的扩容面。
  C 组（语义保留 · 防过度修）：前 10 项构成不变——idx 0..5 是六方向 edge、
     idx 6 是均匀 smooth（全等且和为 1）、idx 7 粗平滑、idx 8 laplacian、
     idx 9 center-surround；变体归一后仍与基核同向（余弦 > 0.99）但不相等。
  D 组（fail-closed · 契约）：函数**自身**必须能检出重复并抛错——用
     monkeypatch 让变体构造退化为旧行为，断言 `ValueError` 而非静默返回。
     （旧实现没有这段校验 ⇒ 本组在回滚后同样红。）

与既有测试的覆盖区分：
  · `tests/test_hex_ortho.py` 用 `extended_prior_family(12)` 只断言
    Gram-Schmidt 对**同轴 edge**（d 与 d+3 互为相反数）判冗余，属线性相关面，
    **不检查逐位相同**，且 n=12 不含坍缩点 16/26/36 ⇒ 无本机制判别力。
  · `tests/test_param_wiring.py` 的 `test_grow_kernel_family_never_repeats_...`
    只长到 K=10，未触到 idx 16 ⇒ 同样不含判别力。
  本守卫 A/B 组是唯一覆盖 idx ≥ 16 坍缩面的断言。

运行（lingshu 仓根）：
    python -X utf8 tests/test_issue940_prior_family_distinct_guard.py
    python -X utf8 -m pytest tests/test_issue940_prior_family_distinct_guard.py
"""
from __future__ import annotations

import os
import sys
import warnings

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

from lingshu.nn import hex_cnn  # noqa: E402
from lingshu.nn.hex_cnn import extended_prior_family  # noqa: E402

_PASS, _FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> bool:
    (_PASS if cond else _FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'} {name}" + (f"  <- {detail}" if detail else ""))
    return bool(cond)


def dups_of(fam):
    """返回所有逐位相同的核下标对。"""
    n = len(fam)
    return [(i, j) for i in range(n) for j in range(i + 1, n)
            if np.array_equal(fam[i], fam[j])]



def main() -> int:
    # ==================== A 组：核心判据（issue 指名的那一档） ====================
    print("== A 组：#940 核心 —— extended_prior_family(37) 任意两核不得逐位相同 ==")
    fam37 = extended_prior_family(37)
    _d = dups_of(fam37)
    check("A1 族大小 == 37", len(fam37) == 37, f"<- {len(fam37)}")
    check("A2 37 族内无重复对（旧实现 6 对）", not _d, f"<- {_d}")
    _collapsed = [i for i in (6, 16, 26, 36) if np.array_equal(fam37[i], fam37[6])]
    check("A3 issue 指名的 idx 6/16/26/36 不再同形", _collapsed == [6], f"<- 与 idx6 同形的: {_collapsed}")

    # ==================== B 组：扩容面（n≥17 可达区间 + 更远） ====================
    print("== B 组：扩容面 —— n = 12 / 17 / 37 / 50 全族互异 ==")
    for n in (12, 17, 37, 50):
        fam = extended_prior_family(n)
        dd = dups_of(fam)
        check(f"B{n} n={n} 族大小正确", len(fam) == n, f"<- {len(fam)}")
        check(f"B{n} n={n} 无重复对", not dd, f"<- {dd}")

    # ==================== C 组：语义保留（防过度修） ====================
    print("== C 组：前 10 项构成不变（不得为了互异改掉白箱语义） ==")
    check("C1 idx6 是均匀 smooth（全等）",
          np.allclose(fam37[6], fam37[6][0], atol=1e-6),
          f"<- {np.round(fam37[6], 6).tolist()}")
    check("C2 idx6 归一后和为 1", abs(float(fam37[6].sum()) - 1.0) < 1e-5,
          f"<- {float(fam37[6].sum()):.8f}")
    # 六方向 edge：idx 0..5 各含 +1 与 -1，其余 0
    edge_ok = True
    for d in range(6):
        k = fam37[d]
        if not (abs(float(k[1 + d]) + 1.0) < 1e-6 and abs(float(k[1 + (d + 3) % 6]) - 1.0) < 1e-6):
            edge_ok = False
    check("C3 idx0..5 仍是六方向 edge(+1/-1 对)", edge_ok)
    # 变体与基核同向但不等（微扰语义未变）
    base, var = fam37[6], fam37[16]
    cos = float(np.dot(base, var) / (np.linalg.norm(base) * np.linalg.norm(var)))
    check("C4 idx16 仍以 idx6 为基（余弦 > 0.99）", cos > 0.99, f"<- cos={cos:.6f}")
    check("C5 idx16 与 idx6 不相等（微扰生效）", not np.array_equal(base, var))

    # ==================== D 组：fail-closed 契约 ====================
    print("== D 组：函数自身必须检出重复并抛错（不能静默返回） ==")
    # 做法：把两个「原始核」的构造函数都换成都返回同一个均匀核，制造 idx6 == idx8
    # 的退化族。修好的实现末尾有互异性校验 ⇒ 必抛 ValueError；旧实现无校验 ⇒ 静默返回。
    _old_smooth = hex_cnn.kernel_smooth
    _old_lap = hex_cnn.kernel_laplacian


    def _fake_uniform() -> np.ndarray:
        return np.full(7, 1.0 / 7.0, dtype=np.float32)


    try:
        hex_cnn.kernel_smooth = _fake_uniform
        hex_cnn.kernel_laplacian = _fake_uniform
        _raised = None
        try:
            _fam_deg = extended_prior_family(37)
        except ValueError as e:
            _raised = str(e)
        check("D1 原始核重复时抛 ValueError（fail-closed）", _raised is not None,
              f"<- {(_raised or '未抛错，静默返回了重复族')[:70]}")
        check("D2 抛错信息点名了下标", bool(_raised) and "idx" in _raised,
              f"<- {(_raised or '')[:70]}")
    finally:
        hex_cnn.kernel_smooth = _old_smooth
        hex_cnn.kernel_laplacian = _old_lap

    # 复原后必须恢复正常
    _fam_back = extended_prior_family(37)
    check("D3 复原后 37 族仍无重复", not dups_of(_fam_back), f"<- {dups_of(_fam_back)}")

    print()
    print("===== SUMMARY %d/%d 通过 =====" % (len(_PASS), len(_PASS) + len(_FAIL)))
    if _FAIL:
        print("失败项:", _FAIL)
        return 1
    print("VERDICT=PASS（#940 先验核族互异：37 族无重复、扩容面互异、"
          "前 10 项语义保留、重复时 fail-closed）")
    return 0


# ── pytest 腿入口 ────────────────────────────────────────────────────────────
# 为什么保留这一层：本件是**脚本式**（模块级即执行全部断言），pytest 收不到。
# 仓内两条守卫（test_gate_coverage / test_pytest_collection_guard）对此的要求是
# 「走脚本腿（登记进 gate.yml SCRIPT_TESTS）**或** 走 pytest 腿（可被收集）」。
# 原方案是改 .github/workflows/gate.yml 登记 —— 但 fork PR 修改 workflow 文件
# 在 GitHub 侧属于需要维护者批准的敏感改动，实测该 PR 的 workflow 因此**根本
# 没被触发**（commits/<sha>/check-runs = 0）。改成给一个 pytest 入口，两腿都通，
# 且不需要碰 CI 配置 —— 更小的改动面，同样的覆盖。
def test_issue940_prior_family_distinct():
    """pytest 腿：跑完整断言组，任一失败即红（main() 返回 0 = 全过）。"""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
