# -*- coding: utf-8 -*-
"""统一时间核（钉死批条款 3：核形状唯一）

    X_i(t) = X_i,∞ + (X_i,0 - X_i,∞)·e^(-γ_i·t)

全库一切时间演化必须经本模块。核形状唯一（指数族），状态 X 与
衰减率 γ 按对象配置；任何在本模块之外出现的衰减核实现
（exp(-t/τ)、×(1-factor)、EMA 保持率）都是 bug——
由 tools/time_core_lint.py 按 E5 口径机械化审计。
规范依据：docs/概念钉死批_GPT四点评审_v0.1.md 钉死 3（该文档不在本仓，属上游/私域件）。
"""
import math


def _as_floats(name, *values):
    """把入口参数统一转为 float；非数值抛可读 TypeError（核形状不变）。"""
    out = []
    for v in values:
        try:
            out.append(float(v))
        except (TypeError, ValueError) as exc:
            raise TypeError("%s 的参数必须是实数，收到 %r" % (name, v)) from exc
    return tuple(out)


def cred(x0: float, x_inf: float, gamma: float, dt: float) -> float:
    """连续指数核：状态从 x0 向终值 x_inf 衰减，速率 γ，历时 dt。"""
    # 自主发现（参数类型）：入口此前无类型转换——传非数值（如 dt="x"）时在
    # math.exp 处抛不可读的 TypeError（"can't multiply sequence by non-int"）。
    # 现统一转 float 并把失败转成可读 TypeError；数值语义不变。
    x0, x_inf, gamma, dt = _as_floats("cred", x0, x_inf, gamma, dt)
    return x_inf + (x0 - x_inf) * math.exp(-gamma * dt)


def cred_factor(gamma: float, dt: float, floor: float = 0.0, ceil: float = 1.0) -> float:
    """归一化**保持因子** r = e^(-γ·dt)，clamp 到 [floor, ceil]。

    适用于 x0=1、x_inf=0 的新近度加权形态（如知识飞轮 recency factor）——
    **乘性**使用：x ← x·r。
    #180：本返回值与 `cred_step` 的第二形参**语义相反、互不通用**——
    前者是保持率（乘），后者是每周期衰减量（减）。`cred_step(x, r) == x·(1-r)`，
    不是 x·r；把本函数结果直送 `cred_step` 会得到「衰减量恰等于保持率」的数。
    #457：非有限结果（NaN/±Inf）fail-closed 归 `floor`——CPython 的
    `min(1.0, nan) == 1.0` 会把 NaN 抬成满额保持（完全不衰减），与 #196/#402
    「非有限不得给出对己有利结论」相反；此处与 #402 同向取最保守端。
    """
    gamma, dt = _as_floats("cred_factor", gamma, dt)
    f = math.exp(-gamma * dt)
    if not math.isfinite(f):
        return floor
    return max(floor, min(ceil, f))


def cred_step(x: float, decay: float) -> float:
    """离散指数核单步：x ← x·(1-decay)。

    `decay` 是**每周期衰减量**（∈[0,1]），不是保持因子（保持因子 = 1-decay）。
    #180：形参旧名为 `factor`，与 `cred_factor` 的返回值同名却语义相反——
    `cred_step(x, cred_factor(γ,dt))` 静默得到 x·(1-r)（r 为保持率），
    既不等于 x·r 也不报错。现正名为 `decay` 并写明互不通用，消除同名陷阱。
    **只正名、不改行为**：调用方（`core.py:1449 decay_cycle(factor=0.02)`）的
    实参语义本就是「每周期衰减 2%」（`x·(1-0.02)=0.98x`），与旧实现一致。
    判据来源：本模块头部「核形状唯一」+ `cred_factor` 的保持因子语义（:20-28）；
    实参语义取 `lingshu/core/core.py:1449-1501` 的既有调用（decay_cycle 默认
    `factor=0.02`，50 轮后知识层 confidence 仍 >0，见 `tests/test_issue29_memory_pollution_guard.py` W4）。

    decay_cycle 逐周期形态；n 步后 x·(1-decay)^n = x·e^(n·ln(1-decay))，
    与连续核同族——衰减率 γ = -ln(1-decay) 每周期。
    """
    return x * (1.0 - decay)


def cred_blend(last: float, incoming: float, retain: float) -> float:
    """指数平滑：last ← last·retain + incoming。

    gap/信息差等累积观测的离散指数核形态（保持率 retain = 1-λ）。
    """
    return last * retain + incoming
