# -*- coding: utf-8 -*-
"""time_core_lint · E5 衰减核机械化审计器（PR-2 新增）

兑现 lingshu/core/time_core.py:9 与 lingshu/world/time_core.py:9 的承诺：

    任何在本模块之外出现的衰减核实现（exp(-t/τ)、×(1-factor)、EMA 保持率）都是 bug
    ——由 tools/time_core_lint.py 按 E5 口径机械化审计。

用法
----
    python tools/time_core_lint.py [路径...] [--json] [--baseline FILE]
                                   [--update-baseline] [--quiet] [--repo-root PATH]
    默认扫描路径 = <仓库根>/lingshu/**/*.py
    默认 baseline = <仓库根>/tools/time_core_lint_baseline.txt
    仓库根 = 本脚本上两级目录（由 __file__ 推导，**不依赖 cwd**）
    `--baseline` 的相对路径**按仓库根解析**（不看 cwd）；绝对路径原样使用。

    两点运维须知：
    · `--baseline FILE --update-baseline` 而 FILE **不存在** → 退出码 2 且**不创建**该文件
      （防误写：写错路径时不会凭空造出一个 baseline）。只有**默认** baseline
      （不给 `--baseline`）不存在时才允许 `--update-baseline` 创建它。
    · 输出以 `{file, line, rule}` 三元组为键：同一逻辑语句同时命中 R2 与 R3 时**输出 2 行**，
      baseline 也必须写 2 行（`--update-baseline` 会自动写 2 行，不是漏了）。

规则表（E5 三类，取自 time_core 自身 docstring）
------------------------------------------------
R1 exp_decay  连续指数核手写：`math.exp(-…` / `np.exp(-…` / `<别名>.exp(-…` / 裸 `exp(-…`
              （口径：`exp(` 的括号内**以负号开头**）
R2 keep_rate  离散保持率改写（三类形态，任一命中即 R2）：
              (a) `X * (1 - f)`（f 为**纯名字**表达式）
              (b) `retain = 1 - f` / `keep = 1 - f` / `keep_rate = 1 - f` 等保持率命名的赋值
              (c) 括号包裹的独立 `(1 - <decay|factor|gamma|rate>|…)` 表达式
R3 ema        EMA/保持率出现在**乘性更新**里：
              (a) 自指乘性更新 `x = … x * …`（x 以 `self.x` 形态亦可）
              (b) 增量乘性赋值 `x *= …`
              (c) 赋值 / return 的表达式中出现 `* <retain|keep|keep_*|momentum|ema|decay*|forget*|…>`

豁免表（写进代码；下表顺序即优先级，先命中者生效）
--------------------------------------------------
EX0 整文件豁免：`lingshu/core/time_core.py`、`lingshu/world/time_core.py`
    —— 核的唯一定义处，本工具的口径来源；其余一切实现都在审计范围内。
EX1 sigmoid：语句含 `1.0 + exp(-…` / `1 + exp(-…`（exp 的负指数出现在 `1 +` 的分母里）
    → 豁免 R1（σ(x)=1/(1+e^-x) 是门控，不是衰减核）
EX2 softmax / 归一化：`exp(-…)` 紧接 `/` 且语句内含 `sum(` / `.sum(`；
    或上一语句是 `e = exp(-…)` 而本语句出现 `e / …sum(…)`（跨语句分子/分母形态）
    → 豁免 R1（softmax 分子分母同式，非时间演化）
EX3 方差 / 标准误：语句含 `sqrt(` 或 `** 0.5` → 豁免该语句的 R2
    （`p*(1-p)/n` 是伯努利方差惯用式；优先于 EX4 判定）
EX4 互补权重线性插值 / alpha 合成：R2 / R3 的因子名 ∈ INTERP_VOCAB（见常量），
    且同一逻辑语句内存在互补的乘性使用（`* f` 或 `f *`）→ 豁免
    （覆盖 `mm[y,x]*(1-fy)*(1-fx)+…` 双线性插值与 `obj*(1-pc)+dark*pc` 合成）
EX5 规范化调用转交：**命中位置落在 `cred(` / `cred_factor(` / `cred_step(` / `cred_blend(` 的
    **实参括号内** → 豁免 R2 / R3（把保持率作为参数交给核，是 time_core 自己写明的参数形态
    `retain = 1-λ`）。注意是**位置级**判定：同一语句里若在 `cred*(…)` **之外**另手写衰减核
    （如 `x = cred_step(a, b) + x * (1 - gamma)`），照样报红。
EX6 baseline 已登记：`tools/time_core_lint_baseline.txt` 中的条目 → 记为「既有」，不计新违规

退出码（冻结）
--------------
0  无新违规（含"只有已登记的既有违规"）
1  存在不在 baseline 的违规
2  用法 / 路径错误（路径不存在、baseline 缺失或格式错、`--update-baseline` 与 `--json` 同用；
   argparse 自身的用法错误同样返回 2）

已知局限（正则口径的假阴 / 假阳边界，如实登记）
------------------------------------------------
L1 只做**逐逻辑语句正则**，不做数据流/语义分析：判定不可避免含名字启发式。
L2 R1 只认 `exp(` 括号内以 `-` 开头：`np.exp(-g*dt)` 命中；先把指数算进变量再 `exp(d)`
   的写法漏报（假阴）；`exp(x)`（无负号）不命中（这是口径，不是遗漏）。
L3 R2(a) 的 factor 必须是**纯名字**表达式（允许 `self.` / 属性链 / 下标，不允许算术运算符，
   不允许以数字字面量开头）：因此 `x * (1 - i*0.18)`、`2.0*(1.0 - 1.5*TOL/d_obj)`、
   `x * (1 - 0.02)`、`x * (1 - gamma*dt)` 一律不命中（假阴，故意为之：算术表达式不是
   "核的参数"）。⚠ 其中 `x * (1 - gamma*dt)` 是真实世界**最常见**的衰减改写形态，这条假阴
   值得维护者知情（想覆盖它需要表达式归因，不是加个正则能收的）。
L4 EX4 是**名字白名单**（INTERP_VOCAB），是本工具最弱的一环：把保持率命名成 `fx/pc/cov/…`
   会漏报；真插值权重若不在白名单（`alpha`、`t`、`u`、`a` 等任何名字）会误报。
   白名单刻意保持小且保守。
L4b R3 的**数值形状**只认 `0 < literal < 1`（`0.9`、`.5`、`0.98`；`1.5`、`2.0`、`100`
   一概不认）：故 `v = 0.9*v + 0.1*g` 命中，而 `x = 2.0 * x`（纯放大，非衰减）不命中。
   代价是 `x = x * 1.5` 也不报——宁可漏判"放大"，不把放大误判成衰减。
L5 注释与字符串字面量在扫描前被**等长掩码成空格**（行号列号不变），故文档里的
   `exp(-t/τ)`、`×(1-factor)` 示例不产生命中；反之 f-string 表达式内部的衰减核也一并被
   掩码（假阴，罕见）。
L6 逻辑语句按括号深度与行尾反斜杠合并续行；**同一物理行上深度 0 的 `;` 会切成独立语句**
   分别判定（避免 `a = a*(1-pc); b = z*pc` 这类同行复合被 EX4 整体洗白）。仍有的边界：
   R3(a) 的赋值目标必须**位于语句开头**（允许 `a, b = …` 的元组前缀、`self.` 前缀与
   `x[i] = …` 下标目标）；`if c: y = y * retain` 这类**同行复合语句**（冒号后内联）里的更新
   会漏报（假阴），换行写成正常语句即可命中。EX1–EX4 的豁免仍是**语句级**的。
L7 仓内文件一律输出**仓内相对路径**；库外路径（如测试 fixture）按传入路径规范化输出，
   此时可能含绝对路径（测试确定性对比只用仓内路径）。
L8 `--quiet` 只打印新违规（既有命中与摘要都不打印；无新违规时完全静默，便于进 CI）。
   `--repo-root` 是便于测试/多仓复用的附加参数，只影响相对路径与规范模块豁免。
L9 已判定为"不按违规处理"的既有候选（每条都带豁免回归测试，见 tests/test_time_core_lint.py）：
   · lingshu/world/gap_dual.py:103 `retain=1.0 - self.decay` —— 该值作为**关键字实参**落在
     规范核 `cred_blend(` 的实参括号内（EX5 位置级豁免）。`lingshu/world/time_core.py:38-43`
     的 `cred_blend(last, incoming, retain) = last * retain + incoming` 乘法本体确实在核内，
     且核 docstring 自己写「保持率 retain = 1-λ」——所以**参数派生留在调用方是核认可的形态**，
     判为不违规；但要说清：这不是"推荐写法"（同模块的 `cred_step(x, factor) = x*(1-factor)`
     语义上可直接用，`cred_step(last, self.decay) + growth` 就能免掉手写 `1-λ`）。
   · lingshu/nn/hex_train.py:289 `d_after <= d_before * retain_gain` —— 出现在**阈值比较**里，
     不是乘性更新：没有状态改写、没有自指。retain_gain 是"生长验证"的判据比值，
     与时间演化无关（R3 只认更新，故不匹配）。
   · lingshu/world/channel_credibility.py:106 `evidence = 1.0 - alpha` —— **跨语句且名字未登记**，
     故按逐语句正则口径不匹配（R2(b) 要求目标名 ∈ 保持率词表、R2(c) 要求括号包裹；
     下游 `* 20.0` 在下一语句）。★ 措辞上说清楚：这是"机制上抓不到"，**不是"性质上不是保持率"**
     ——`1 - α` 当证据保留权重、下游乘性使用，离 E5 精神很近，值得维护者人工看一眼。
   · lingshu/gen/hexgen_self_source.py:236/240/411/413、gen/hexgen_c1_real.py:464-465
     —— 互补权重线性插值 / alpha 合成（EX4）。
   · lingshu/core/core.py:1773/1782/1783 `math.sqrt(p * (1 - p) / n)` —— 伯努利标准误（EX3）。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

LINT_VERSION = 1

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCAN_DIR = "lingshu"
DEFAULT_BASELINE_REL = "tools/time_core_lint_baseline.txt"

CANONICAL_MODULES: Tuple[str, ...] = (
    "lingshu/core/time_core.py",
    "lingshu/world/time_core.py",
)

RULE_NAMES: Dict[str, str] = {
    "R1": "exp_decay",
    "R2": "keep_rate",
    "R3": "ema",
}
RULE_NAME_TO_ID: Dict[str, str] = {v: k for k, v in RULE_NAMES.items()}

# 线性插值 / alpha 合成的权重名白名单（EX4；见局限 L4）
INTERP_VOCAB: Tuple[str, ...] = (
    "fx", "fy", "ix", "iy", "wx", "wy", "w0", "w1",
    "lerp", "frac", "pc", "cov", "blend", "mix", "ratio",
)

MAX_SNIPPET = 160

# ---------------------------------------------------------------- 规则正则
# 纯名字表达式：允许 self./属性链/下标，不允许算术运算符，不允许数字字面量开头
_NAMEONLY = r"[A-Za-z_][A-Za-z0-9_\.\[\]',\s]*"

RULE_R1 = re.compile(r"(?:[A-Za-z_]\w*\.)?\bexp\s*\(\s*-")

RULE_R2A = re.compile(r"\*\s*\(\s*1(?:\.0)?\s*-\s*(" + _NAMEONLY + r")\)")

_KEEPNAME = (r"(?:retain|keep_(?:rate|factor|ratio|prob)|keep|momentum)\w*")
RULE_R2B = re.compile(
    r"(?<![<>=!\w])" + _KEEPNAME + r"\s*=(?!=)\s*1(?:\.0)?\s*-\s*(" + _NAMEONLY + r")")

RULE_R2C = re.compile(
    r"\(\s*1(?:\.0)?\s*-\s*"
    r"(?:decay\w*|factor\w*|gamma|lambda|rate\w*|halflife|half_life)\s*\)")

RULE_R3B = re.compile(r"\*=")

# 赋值语句：目标必须**位于语句开头**（允许 `a, b = ` 元组前缀与 `self.` 前缀与下标目标），
# 以免把关键字实参（`fill=(...)`）或调用参数里的 `=` 当成更新。
_ASSIGN = re.compile(
    r"^\s*(?:[A-Za-z_]\w*\s*,\s*)*(?:self\.)?([A-Za-z_]\w*)"
    r"\s*(?:\[[^\[\]]*\])?\s*=(?!=)\s*(.+)", re.S)

# EMA / 保持率命名（裸 `keep` 也算——否则 `x*(1-keep)` 报而 `x*keep` 不报，口径自相矛盾；
# `keep_pm`/`keepdims` 这类前缀相同的名字因 `keep\b` 要求词边界而不会误伤）
_KEEP_WORD = (
    r"(?:retain\w*"
    r"|keep_(?:rate|factor|ratio|prob)\w*"
    r"|keep\b"
    r"|momentum\w*"
    r"|ema(?:_(?:rate|alpha|factor))?\w*"
    r"|decay(?:_(?:rate|factor|coef|gamma))?\b"
    r"|forget(?:_rate)?\b"
    r"|halflife|half_life|smoothing\w*)"
)
RULE_R3C = re.compile(r"\*\s*(?:self\.|np\.)?(?:[A-Za-z_]\w*\.)*" + _KEEP_WORD)

# 保持率形状的因子：`(1 - f)`、EMA/保持率命名、`0.x`/`.x` 字面量。
# 数值分支带 `(?<![\d.])` 前视，只认 `0 < literal < 1`：`2.0`/`1.5`/`100` 一律不认
# （见头部 L4b：宁可漏判"放大"，不把纯放大误判成衰减）。
_KEEP_SHAPE = (
    r"(?:" + _KEEP_WORD
    + r"|\(\s*1(?:\.0)?\s*-[^()]*\)"
    + r"|(?<![\d.])0?\.\d+)"
)

_SIGMOID = re.compile(r"1(?:\.0)?\s*\+\s*[\w\.]*exp\s*\(\s*-")
_SOFTMAX = re.compile(r"exp\s*\(\s*-[^()]*\)\s*/\s*")
_SQRT = re.compile(r"sqrt\s*\(|\*\*\s*0?\.5\b")
_CANON_CALL = re.compile(r"\bcred(?:_factor|_step|_blend)?\s*\(")


# ---------------------------------------------------------------- 源码掩码
def mask_source(text: str) -> List[str]:
    """把注释与字符串字面量等长掩码成空格；返回逐行列表（行号/列号不变）。"""
    out: List[str] = []
    state: Optional[str] = None           # None | "'''" | '"""' | "'" | '"'
    for raw in text.split("\n"):
        buf = list(raw)
        n = len(raw)
        i = 0
        while i < n:
            ch = raw[i]
            if state in ("'''", '"""'):
                if raw.startswith(state, i):
                    for j in range(i, i + 3):
                        buf[j] = " "
                    i += 3
                    state = None
                    continue
                buf[i] = " "
                i += 1
                continue
            if state in ("'", '"'):
                if ch == "\\":
                    buf[i] = " "
                    if i + 1 < n:
                        buf[i + 1] = " "
                    i += 2
                    continue
                buf[i] = " "
                if ch == state:
                    state = None
                i += 1
                continue
            if ch == "#":
                for j in range(i, n):
                    buf[j] = " "
                break
            if ch in ("'", '"'):
                triple = ch * 3
                if raw.startswith(triple, i):
                    for j in range(i, i + 3):
                        buf[j] = " "
                    i += 3
                    state = triple
                    continue
                buf[i] = " "
                i += 1
                state = ch
                continue
            i += 1
        out.append("".join(buf))
        if state in ("'", '"'):           # 单行字符串不应跨物理行（保守复位）
            state = None
    return out


def _depth(line: str) -> int:
    return (line.count("(") + line.count("[") + line.count("{")
            - line.count(")") - line.count("]") - line.count("}"))


class Statement(object):
    """一条逻辑语句：连续物理行（括号未闭合或行尾反斜杠）合并，带偏移→行号映射。

    lines 为各 part 对应的 **1 基**物理行号（同一物理行上深度 0 的 `;` 会切成多条语句，
    故行号可能重复/不连续；`start_line`/`end_line` 仅供阅读与调试）。
    """

    __slots__ = ("lines", "text", "_offsets")

    def __init__(self, lines: Sequence[int], parts: Sequence[str]):
        self.lines = tuple(lines)
        self.text = " ".join(parts)
        offs: List[int] = []
        pos = 0
        for p in parts:
            offs.append(pos)
            pos += len(p) + 1
        self._offsets = offs

    @property
    def start_line(self) -> int:
        return self.lines[0]

    @property
    def end_line(self) -> int:
        return self.lines[-1]

    def line_of(self, offset: int) -> int:
        idx = 0
        for k, off in enumerate(self._offsets):
            if off <= offset:
                idx = k
            else:
                break
        return self.lines[idx]


def split_fragments(masked_lines: Sequence[str]) -> List[Tuple[int, str]]:
    """按**全局括号深度 0** 的 `;` 切分，返回 [(1 基行号, 片段文本), ...]。

    每个物理行至少产出一个片段（行号即该行）；深度 >0 的 `;`（字典/调用实参里的分号）
    不切；注释与字符串已掩码，不会误切。
    """
    frags: List[Tuple[int, str]] = []
    depth = 0
    for idx, line in enumerate(masked_lines, start=1):
        cur: List[str] = []
        for ch in line:
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth = max(0, depth - 1)
            elif ch == ";" and depth == 0:
                frags.append((idx, "".join(cur)))
                cur = []
                continue
            cur.append(ch)
        frags.append((idx, "".join(cur)))
    return frags


def build_statements(masked_lines: Sequence[str]) -> List[Statement]:
    frags = split_fragments(masked_lines)
    stmts: List[Statement] = []
    i = 0
    n = len(frags)
    while i < n:
        lines = [frags[i][0]]
        parts = [frags[i][1]]
        depth = _depth(frags[i][1])
        cont = frags[i][1].rstrip().endswith("\\")
        i += 1
        while i < n and (depth > 0 or cont):
            lines.append(frags[i][0])
            parts.append(frags[i][1])
            depth += _depth(frags[i][1])
            cont = frags[i][1].rstrip().endswith("\\")
            i += 1
        stmts.append(Statement(lines, parts))
    return stmts


# ---------------------------------------------------------------- 豁免判定
def _factor_base(factor: str) -> str:
    m = re.match(r"\s*(?:self\.)?([A-Za-z_]\w*)", factor)
    return m.group(1) if m else ""


def _is_depth0(s: str, pos: int) -> bool:
    """pos 处是否位于括号深度 0（顶层表达式）。"""
    d = 0
    for ch in s[:pos]:
        if ch in "([{":
            d += 1
        elif ch in ")]}":
            d = max(0, d - 1)
    return d == 0


def _self_multiplicative_pos(rhs: str, name: str) -> Optional[int]:
    """R3(a)：RHS 顶层出现 `NAME * 保持率形状` 或 `保持率形状 * NAME` 时返回该乘法位置。

    深度 >0 的乘法（如 `(t * w).sum()` 正交化、`*fill` 解包）不算"乘性更新"。
    """
    esc = re.escape(name)
    name_pat = r"(?:self\.)?" + esc + r"\b"
    for m in re.finditer(name_pat + r"\s*\*\s*" + _KEEP_SHAPE, rhs):
        if _is_depth0(rhs, m.start()):
            return m.start()
    for m in re.finditer(_KEEP_SHAPE + r"\s*\*\s*" + name_pat, rhs):
        if _is_depth0(rhs, m.start()):
            return m.start()
    return None


def _canon_arg_spans(stmt: str) -> List[Tuple[int, int]]:
    """每个 `cred*(…)` 调用的实参区间 `[实参起点, 调用结束)`（按括号配对找右括号）。"""
    spans: List[Tuple[int, int]] = []
    for m in _CANON_CALL.finditer(stmt):
        depth = 0
        end = len(stmt)
        for i in range(m.end() - 1, len(stmt)):
            ch = stmt[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        spans.append((m.end(), end))
    return spans


def _in_canon_args(stmt: str, pos: int) -> bool:
    """EX5（**位置级**）：命中位置是否落在某个 `cred*(…)` 的实参括号内。

    pos < 0（未给出位置）一律返回 False —— 失败方向朝"报红"，不朝"洗白"。
    """
    if pos < 0:
        return False
    return any(start < pos < end for start, end in _canon_arg_spans(stmt))


def _complementary_weight(stmt: str, base: str) -> bool:
    if not base:
        return False
    esc = re.escape(base)
    if re.search(r"\*\s*(?:self\.)?" + esc + r"\b", stmt):
        return True
    if re.search(r"\b" + esc + r"\b\s*\*", stmt):
        return True
    return False


def _interp_exempt(stmt: str) -> bool:
    """EX4：语句内同时出现 `(1 - w)` 与互补的乘性使用 `* w` / `w *`，且 w ∈ INTERP_VOCAB。"""
    for base in INTERP_VOCAB:
        if not _complementary_weight(stmt, base):
            continue
        if re.search(r"\(\s*1(?:\.0)?\s*-\s*(?:self\.)?" + re.escape(base) + r"\b", stmt):
            return True
    return False


def _exempt_reason(stmt: str, rule: str, factor_base: str = "",
                   pos: int = -1) -> Optional[str]:
    """返回豁免理由（EX 编号），不豁免返回 None。顺序 = 优先级。

    `pos` = 命中在语句内的偏移（EX5 位置级判定用；R1 忽略）。
    """
    if rule == "R1":
        if _SIGMOID.search(stmt):
            return "EX1 sigmoid 门控"
        if _SOFTMAX.search(stmt) and ("sum(" in stmt or ".sum(" in stmt):
            return "EX2 softmax/归一化"
        return None
    # R2 / R3
    if rule == "R2" and _SQRT.search(stmt):
        return "EX3 方差/标准误"
    if (factor_base and factor_base in INTERP_VOCAB and _complementary_weight(stmt, factor_base)) \
            or _interp_exempt(stmt):
        return "EX4 互补权重线性插值/alpha 合成"
    if _in_canon_args(stmt, pos):
        return "EX5 规范化调用转交（命中在 cred*(…) 实参内）"
    return None


# ---------------------------------------------------------------- 单文件审计
def lint_source(text: str, rel_path: str, raw_lines: Optional[Sequence[str]] = None
                ) -> List[Dict]:
    """审计一段源码，返回命中列表（不含 baseline 判定）。"""
    masked = mask_source(text)
    if raw_lines is None:
        raw_lines = text.split("\n")
    hits: Dict[Tuple[int, str], Dict] = {}
    stmts = build_statements(masked)

    # EX2 跨语句 softmax：`e = np.exp(-z)` 紧接 `return e / e.sum()` 形态
    softmax_numerator: Set[int] = set()
    for idx in range(len(stmts) - 1):
        cur = stmts[idx].text
        if not RULE_R1.search(cur):
            continue
        if _SOFTMAX.search(cur) and ("sum(" in cur or ".sum(" in cur):
            continue
        am = _ASSIGN.search(cur)
        if not am:
            continue
        tail = stmts[idx + 1].text
        if re.search(r"\b" + re.escape(am.group(1)) + r"\b\s*/", tail) \
                and ("sum(" in tail or ".sum(" in tail):
            softmax_numerator.add(idx)

    def add(line: int, rule: str, reason: Optional[str]) -> None:
        if reason is not None:
            return
        key = (line, rule)
        if key in hits:
            return
        snippet = raw_lines[line - 1].strip() if 0 < line <= len(raw_lines) else ""
        if len(snippet) > MAX_SNIPPET:
            snippet = snippet[:MAX_SNIPPET - 3] + "..."
        hits[key] = {
            "file": rel_path,
            "line": line,
            "rule": rule,
            "rule_name": RULE_NAMES[rule],
            "code_snippet": snippet,
        }

    for idx, stmt in enumerate(stmts):
        stmt_text = stmt.text

        for m in RULE_R1.finditer(stmt_text):
            reason = _exempt_reason(stmt_text, "R1")
            if reason is None and idx in softmax_numerator:
                reason = "EX2 softmax/归一化（跨语句分子/分母）"
            add(stmt.line_of(m.start()), "R1", reason)

        for m in RULE_R2A.finditer(stmt_text):
            base = _factor_base(m.group(1))
            add(stmt.line_of(m.start()), "R2",
                _exempt_reason(stmt_text, "R2", base, m.start()))
        for m in RULE_R2B.finditer(stmt_text):
            base = _factor_base(m.group(1))
            add(stmt.line_of(m.start()), "R2",
                _exempt_reason(stmt_text, "R2", base, m.start()))
        for m in RULE_R2C.finditer(stmt_text):
            add(stmt.line_of(m.start()), "R2",
                _exempt_reason(stmt_text, "R2", "", m.start()))

        # R3(a) 自指乘性更新 -------------------------------------------------
        am = _ASSIGN.search(stmt_text)
        if am:
            name = am.group(1)
            rhs = am.group(2)
            spos = _self_multiplicative_pos(rhs, name)
            if spos is not None:
                pos = am.start(2) + spos
                add(stmt.line_of(pos), "R3",
                    _exempt_reason(stmt_text, "R3", "", pos))
            # R3(c) 乘性更新里出现 EMA/保持率命名
            cm = RULE_R3C.search(rhs)
            if cm:
                pos = am.start(2) + cm.start()
                add(stmt.line_of(pos), "R3",
                    _exempt_reason(stmt_text, "R3", "", pos))
        elif stmt_text.lstrip().startswith("return "):
            cm = RULE_R3C.search(stmt_text)
            if cm:
                add(stmt.line_of(cm.start()), "R3",
                    _exempt_reason(stmt_text, "R3", "", cm.start()))

        # R3(b) 增量乘性赋值 -----------------------------------------------
        for m in RULE_R3B.finditer(stmt_text):
            add(stmt.line_of(m.start()), "R3",
                _exempt_reason(stmt_text, "R3", "", m.start()))

    return sorted(hits.values(), key=lambda h: (h["file"], h["line"], h["rule"]))


# ---------------------------------------------------------------- baseline
def parse_baseline(text: str) -> Tuple[Set[Tuple[str, int, str]], List[str]]:
    """解析 baseline；返回 (条目集合, 错误列表)。行格式 <相对路径>:<行>:<规则>，`#` 注释。"""
    entries: Set[Tuple[str, int, str]] = set()
    errors: List[str] = []
    for no, raw in enumerate(text.split("\n"), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.rsplit(":", 2)
        if len(parts) != 3:
            errors.append("第 %d 行无法解析（期望 <路径>:<行号>:<规则>）：%s" % (no, raw.strip()))
            continue
        path, lineno_s, rule_s = parts
        rule = rule_s.strip().upper()
        if rule not in RULE_NAMES:                       # 也允许写规则名（exp_decay/keep_rate/ema）
            rule = RULE_NAME_TO_ID.get(rule_s.strip().lower(), rule)
        if not re.fullmatch(r"[0-9]+", lineno_s.strip()):
            errors.append("第 %d 行行号非整数：%s" % (no, raw.strip()))
            continue
        if rule not in RULE_NAMES:
            errors.append("第 %d 行规则名未知（应为 R1/R2/R3 或 exp_decay/keep_rate/ema）：%s"
                          % (no, raw.strip()))
            continue
        if not path.strip():
            errors.append("第 %d 行路径为空：%s" % (no, raw.strip()))
            continue
        entries.add((path.strip().replace("\\", "/"), int(lineno_s.strip()), rule))
    return entries, errors


def render_baseline(entries: Iterable[Tuple[str, int, str]]) -> str:
    lines = [
        "# time_core_lint baseline —— 已登记的既有违规（E5 衰减核审计）",
        "# 格式：<仓内相对路径>:<行号>:<规则id>    （R1=exp_decay / R2=keep_rate / R3=ema）",
        "# `#` 起为注释；本文件由人工复核，--update-baseline 只做机械重写。",
        "# 规则表 / 豁免表 / 退出码 / 已知局限见 tools/time_core_lint.py 头部。",
        "",
    ]
    for path, line, rule in sorted(entries, key=lambda e: (e[0], e[1], e[2])):
        lines.append("%s:%d:%s    # 既有：由 --update-baseline 机械登记，性质待人工复核"
                     % (path, line, rule))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- 扫描
def _read_source(path: Path) -> str:
    data = path.read_bytes()
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace")


def normalize_report_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return Path(os.path.abspath(str(path))).as_posix()


def collect_targets(paths: Sequence[str], root: Path) -> Tuple[List[Path], List[str]]:
    targets: List[Path] = []
    errors: List[str] = []
    for raw in paths:
        p = Path(raw)
        if not p.exists():
            alt = root / raw
            if alt.exists():
                p = alt
            else:
                errors.append("路径不存在：%s" % raw)
                continue
        if p.is_dir():
            targets.extend(sorted((q for q in p.rglob("*.py")), key=lambda q: str(q)))
        else:
            targets.append(p)
    uniq: List[Path] = []
    seen: Set[str] = set()
    for t in targets:
        key = str(t.resolve())
        if key not in seen:
            seen.add(key)
            uniq.append(t)
    return uniq, errors


def repository_scan(root: Path, paths: Sequence[str]) -> Tuple[Dict, List[str]]:
    targets, errors = collect_targets(paths, root)
    if errors:
        return {}, errors

    raw_violations: List[Dict] = []
    for path in targets:
        rel = normalize_report_path(path, root)
        if rel in CANONICAL_MODULES:
            continue
        try:
            text = _read_source(path)
        except OSError as exc:
            errors.append("无法读取 %s：%s" % (rel, exc))
            continue
        raw_violations.extend(lint_source(text, rel))

    raw_violations.sort(key=lambda h: (h["file"], h["line"], h["rule"]))

    baseline_path = root / DEFAULT_BASELINE_REL
    return {
        "files_scanned": len([p for p in targets
                              if normalize_report_path(p, root) not in CANONICAL_MODULES]),
        "violations": raw_violations,
        "default_baseline_path": baseline_path,
    }, errors


# ---------------------------------------------------------------- 主流程
def _load_baseline_file(path: Path, explicit: bool
                        ) -> Tuple[Set[Tuple[str, int, str]], List[str], bool]:
    """返回 (条目集合, 错误列表, 文件是否存在)。显式给出但不存在 = 用法错误（退出 2）。"""
    if not path.exists():
        if explicit:
            return set(), ["baseline 文件不存在：%s" % path], False
        return set(), [], False
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    entries, errors = parse_baseline(text)
    return entries, errors, True


def run(argv: Optional[Sequence[str]] = None) -> Tuple[int, str, Dict]:
    parser = argparse.ArgumentParser(
        prog="time_core_lint.py",
        description="E5 衰减核机械化审计器（R1 exp_decay / R2 keep_rate / R3 ema）")
    parser.add_argument("paths", nargs="*", default=None,
                        help="要扫描的文件或目录（默认 <仓库根>/%s/**/*.py）" % DEFAULT_SCAN_DIR)
    parser.add_argument("--json", action="store_true", help="输出确定性 JSON（sort_keys=True）")
    parser.add_argument("--baseline", metavar="FILE", default=None,
                        help="baseline 文件（默认 <仓库根>/%s；"
                             "相对路径一律按仓库根解析，不看 cwd）" % DEFAULT_BASELINE_REL)
    parser.add_argument("--update-baseline", action="store_true",
                        help="把当前全部违规写回 baseline（不修改源码）")
    parser.add_argument("--quiet", action="store_true",
                        help="只打印新违规（既有命中与摘要都不打印）")
    parser.add_argument("--repo-root", metavar="PATH", default=str(REPO_ROOT),
                        help="仓库根（默认由 __file__ 推导；仅影响相对路径与规范模块豁免）")
    args = parser.parse_args(argv)

    root = Path(args.repo_root).resolve()
    if not root.is_dir():
        return 2, "仓库根不存在：%s\n" % root, {"exit_code": 2}

    scan_paths = list(args.paths) if args.paths else [str(root / DEFAULT_SCAN_DIR)]
    report, errors = repository_scan(root, scan_paths)
    if errors:
        return 2, "\n".join(errors) + "\n", {"exit_code": 2}

    # baseline：相对路径**一律按仓库根解析**（不看 cwd）——避免"在别的目录跑一次
    # --update-baseline 就改掉了无关的同名文件"这类事故；绝对路径原样使用。
    if args.baseline is None:
        baseline_path = root / DEFAULT_BASELINE_REL
        explicit_baseline = False
    else:
        baseline_arg_path = Path(args.baseline)
        baseline_path = (baseline_arg_path if baseline_arg_path.is_absolute()
                         else root / baseline_arg_path)
        explicit_baseline = True
    baseline_entries, baseline_errors, baseline_found = _load_baseline_file(
        baseline_path, explicit_baseline)
    if baseline_errors:
        return 2, "\n".join(baseline_errors) + "\n", {"exit_code": 2}

    violations: List[Dict] = report["violations"]
    registered: List[Dict] = []
    new: List[Dict] = []
    for hit in violations:
        key = (hit["file"], hit["line"], hit["rule"])
        rec = dict(hit)
        rec["baseline"] = key in baseline_entries
        (registered if rec["baseline"] else new).append(rec)

    stale = sorted(
        "%s:%d:%s" % e for e in baseline_entries
        if e not in {(h["file"], h["line"], h["rule"]) for h in violations})

    payload = {
        "baseline_entries": len(baseline_entries),
        "baseline_file": normalize_report_path(baseline_path, root),
        "baseline_found": baseline_found,
        "canonical_modules": list(CANONICAL_MODULES),
        "exit_code": 1 if new else 0,
        "files_scanned": report["files_scanned"],
        "new_violations": new,
        "registered_violations": registered,
        "rule_names": dict(RULE_NAMES),
        "stale_baseline_entries": stale,
        "summary": {
            "new": len(new),
            "registered": len(registered),
            "files": report["files_scanned"],
            "total": len(violations),
        },
        "version": LINT_VERSION,
    }

    if args.update_baseline:
        if args.json:
            return 2, "--update-baseline 不能与 --json 同用\n", {"exit_code": 2}
        text = render_baseline(e for e in
                               ((h["file"], h["line"], h["rule"]) for h in violations))
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(text, encoding="utf-8")
        msg = ""
        if not args.quiet:
            msg = "已写入 %d 条既有违规 → %s\n" % (len(violations), baseline_path)
        return 0, msg, payload

    if args.json:
        return payload["exit_code"], json.dumps(payload, sort_keys=True,
                                                ensure_ascii=False, indent=2) + "\n", payload

    lines: List[str] = []
    if not args.quiet:
        if not baseline_found:
            lines.append("提示：baseline 不存在（%s），本次按空 baseline 运行——所有命中都算新违规"
                         % normalize_report_path(baseline_path, root))
        for hit in registered:
            lines.append("%s:%d:%s  [既有]  %s"
                         % (hit["file"], hit["line"], hit["rule"], hit["code_snippet"]))
    for hit in new:
        lines.append("%s:%d:%s  [新]    %s"
                     % (hit["file"], hit["line"], hit["rule"], hit["code_snippet"]))
    if not args.quiet:
        lines.append("")
        lines.append("扫描 %d 个文件；命中 %d 条（既有 %d / 新增 %d）；baseline 条目 %d"
                     % (payload["files_scanned"], len(violations), len(registered),
                        len(new), len(baseline_entries)))
        if stale:
            lines.append("baseline 中已失效条目 %d 条（可清理）：%s"
                         % (len(stale), ", ".join(stale)))
        lines.append("退出码 %d（%s）" % (payload["exit_code"],
                                       "存在不在 baseline 的违规" if new else "无新违规"))
    return payload["exit_code"], "\n".join(lines) + ("\n" if lines else ""), payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            # newline="\n"：关掉 Windows 的 CRLF 文本翻译，保证输出跨平台逐字节一致
            sys.stdout.reconfigure(encoding="utf-8", newline="\n")
        except Exception:      # pragma: no cover - 平台相关
            pass
    code, out, _payload = run(argv)
    if out:
        sys.stdout.write(out)
    return code


if __name__ == "__main__":
    sys.exit(main())
