#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_issue180_cred_step_semantics_guard —— cred_step / cred_factor 参数语义冲突守卫（lingshu #180）

缺陷（triage 判「成立」，取证 `lingshu/core/time_core.py:20-26` 与 `:35`）：
    `cred_factor(gamma, dt)` 返回 **保持率** r = e^(-γ·dt)（乘性：x ← x·r）；
    `cred_step(x, factor)` 的第二形参**也叫 `factor`**，但语义是**每周期衰减量**
    d（减性：x ← x·(1-d)）。两个同名量语义相反、互不通用：
        cred_factor(0.05, 46) = 0.1003
        cred_step(1.0, 0.1003) = 0.8997   ← 既非 1.0·r，也不报错
    调用方按模块自己的词汇把 `cred_factor(...)` 结果直送 `cred_step` 会**静默**
    得到「衰减量恰等于保持率」的数——单步就把 x=1.0 打到 0.9，而本意是 0.1。

修复（最小、只正名不改行为）：
    · `cred_step` 第二形参 `factor` → `decay`（与其语义「每周期衰减量」同名）；
    · 两函数 docstring 各自显式写明「与另一个核的形参/返回值互不通用」；
    · 行为**逐字不变**（`x·(1-decay)`，与旧 `x·(1-factor)` 同一表达式）；
    · 双副本 `lingshu/world/time_core.py` 同步（纪律 6）。

判据来源：
  · 保持率/衰减量的语义分工 —— `lingshu/core/time_core.py` 头部「核形状唯一」
    与 `cred_factor`/`cred_blend` 的既有 docstring（`cred_blend(..., retain)`：
    「保持率 retain = 1-λ」；`cred_factor`：「归一化保持因子」）。
  · 实参语义取既有调用 —— `lingshu/core/core.py:1449 decay_cycle(factor=0.02)` →
    `:1481/:1501/:1527 cred_step(..., factor)`：0.02 是**每周期衰减 2%**（x·0.98），
    不是保持 2%。该调用点在三处，全为**位置实参**（无 `cred_step(..., factor=...)`），
    故正名不触及任何调用方。
  · 「理论章节」追不到（本仓无 `docs/概念钉死批_GPT四点评审_v0.1.md`，模块头部引用
    的那份不在本仓）⇒ 判据属**经验标定**（既有调用点 + 既有守卫
    `tests/test_issue29_memory_pollution_guard.py` W4 的 50 轮衰减读数）。

断言组（抽掉修复即红）：
  G1 形参名不再与 `cred_factor` 的返回值撞名：第二形参名 == 'decay'（旧 == 'factor' ⇒ 红）。
  G2 两核的语义分工在 docstring 里显式点明「互不通用」（旧 docstring 无此表述 ⇒ 红）。
  G3 行为契约不变（防「正名顺手改了行为」）：
     cred_step(x, d) == x·(1-d)；cred_factor(γ,dt) == e^(-γ·dt)（乘性）；
     且 cred_step(x, cred_factor(γ,dt)) != x·cred_factor(γ,dt)（两者确非同一量）。
  G4 既有调用方语义不变（端到端）：引擎 `decay_cycle()` 默认参数下，知识层
     confidence 每轮 ×0.98（50 轮后仍 >0 且严格下降）——与 #29 W4 同形。
  G5 副本唯一（#176 后）：`lingshu/core/time_core.py` 为唯一真源；
     `lingshu/world/time_core.py` 已被删除且不得重新出现（纪律 6 的终点是合一，不是长存双份）。

运行（仓根）：python -X utf8 -m pytest tests/test_issue180_cred_step_semantics_guard.py -q --no-header
             / python -X utf8 tests/test_issue180_cred_step_semantics_guard.py
"""
from __future__ import annotations

import inspect
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.core import time_core as T  # noqa: E402

_CORE_TC = os.path.join(REPO, "lingshu", "core", "time_core.py")
_WORLD_TC = os.path.join(REPO, "lingshu", "world", "time_core.py")


def test_g1_second_param_is_not_named_factor():
    """G1：cred_step 第二形参不得与 cred_factor 的返回值同名（旧 == 'factor'）。"""
    params = list(inspect.signature(T.cred_step).parameters)
    assert params == ["x", "decay"], (
        "cred_step 第二形参仍与 cred_factor 的返回值（保持率 factor）撞名：%r" % (params,))
    assert "factor" not in params, params
    # cred_factor 的返回值确实是「保持率」这一量——它是撞名的另一方
    assert list(inspect.signature(T.cred_factor).parameters)[:2] == ["gamma", "dt"]


def test_g2_docstrings_state_the_two_are_not_interchangeable():
    """G2：两函数 docstring 必须显式声明「互不通用」（旧 docstring 无此表述）。"""
    for fn in (T.cred_step, T.cred_factor):
        doc = fn.__doc__ or ""
        assert "互不通用" in doc, (
            "%s 的 docstring 未声明与另一核的形参/返回值互不通用：%r"
            % (fn.__name__, doc))
    assert "保持" in (T.cred_factor.__doc__ or "")
    assert "衰减量" in (T.cred_step.__doc__ or "")


def test_g3_behaviour_contract_unchanged():
    """G3：两核行为逐字不变，且确为两个不同的量（防「正名顺手改行为」）。"""
    # 减性：每周期衰减量 d
    assert T.cred_step(1.0, 0.02) == 1.0 * (1.0 - 0.02)
    assert abs(T.cred_step(1.0, 0.02) - 0.98) < 1e-12
    # n 步 = (1-d)^n
    x = 1.0
    for _ in range(50):
        x = T.cred_step(x, 0.02)
    assert abs(x - 0.98 ** 50) < 1e-12, x
    # 乘性：保持率 r = e^(-γ·dt)
    assert abs(T.cred_factor(0.05, 46.0) - math.exp(-0.05 * 46.0)) < 1e-12
    # 两者确非同一量（同名陷阱的实证）：cred_step(x, r) == x·(1-r) != x·r
    r = T.cred_factor(0.05, 46.0)
    assert abs(r - 0.1002588437228037) < 1e-12
    assert abs(T.cred_step(1.0, r) - (1.0 - r)) < 1e-12
    assert abs(T.cred_step(1.0, r) - 1.0 * r) > 0.5, (
        "cred_step 与 cred_factor 的量纲意外重合——同名陷阱的论证前提被破坏")


def test_g4_existing_caller_semantics_unchanged():
    """G4：既有调用方（core.decay_cycle 默认 factor=0.02）语义不变。"""
    from lingshu.core.core import SpacetimeMemoryEngine
    e = SpacetimeMemoryEngine(":memory:")
    k = e.add_perception("时间核正名回归探针", skip_dedup=True)
    before = e.store.get_node(k.id).confidence
    e.decay_cycle()
    one = e.store.get_node(k.id).confidence
    assert abs(one - before * 0.98) < 1e-9, (before, one)
    for _ in range(49):
        e.decay_cycle()
    after = e.store.get_node(k.id).confidence
    assert after < before and after > 0.0, (before, after)


def test_g5_dual_copies_have_been_merged():
    """G5（#176 后）：副本已合一 —— world/ 下不得再有 time_core.py。

    改前本断言是「两份逐字节相同」（纪律 6）。#176 建议3 要求「改为从唯一位置
    导入」，纪律 6 的终点即合一；故改为断言世界侧不再有该模块，且真源仍在。
    """
    assert os.path.isfile(_CORE_TC), "唯一真源 lingshu/core/time_core.py 缺失"
    assert not os.path.exists(_WORLD_TC), "副本复现：lingshu/world/time_core.py 不得存在（#176）"


def main():
    fns = [test_g1_second_param_is_not_named_factor,
           test_g2_docstrings_state_the_two_are_not_interchangeable,
           test_g3_behaviour_contract_unchanged,
           test_g4_existing_caller_semantics_unchanged,
           test_g5_dual_copies_have_been_merged]
    failed = 0
    for fn in fns:
        try:
            fn()
            print("PASS %s" % fn.__name__)
        except AssertionError as exc:
            failed += 1
            print("FAIL %s | %s" % (fn.__name__, exc))
    print("-- %d passed, %d failed --" % (len(fns) - failed, failed))
    print("VERDICT=%s" % ("PASS" if not failed else "FAIL"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
