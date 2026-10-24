#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spacetime_memory_core · 智能论 v3.2 协议实例核心引擎
纯标准库 · 零外部依赖 · 五层记忆 · 时空因果 · 自我认知

设计依据：
- 智能论 v3.2 第四章 4.3 多模态实体识别、语义关系、因果推理、条件空间感知
- 第四章 4.6 生命周期（衰减）：未验证边置信度指数衰减
- 蜂群奠基声明 六、分层：锚点层/结构层/知识层/情境层/自我层
- 3.2 节 选择性遗忘边界：锚点层与结构层不可遗忘

版本：v1.13 · 协议实例核心（基于 v1.12；v1.13 视觉感知：YOLO 外接（可选扩展·零依赖降级）/视觉→记忆闭环/身体能力声明；VISION-REV1）
"""

# issue #156 v2：**最早落点**——在本文件任何裸名 import（含下方 `import json`）
# 之前消除 cwd/空串 sys.path 解析面。包导入时 `lingshu/__init__.py` 已执行同一
# 护栏（幂等，本行再调用一次为防御纵深，并使「最早落点」在本文件可自证）。
# ★机制澄清（#273 路 B）：这里是「**解析期拒绝**」而非「从 `sys.path` 删除条目」——
#   cwd/空串条目**仍在 `sys.path` 里**（`scrub()` 已不改写宿主 `sys.path`），只是被
#   `_pathguard` 装在 `sys.meta_path` 上的 finder 在解析时跳过 ⇒ 对**解析结果**而言
#   等价于「该面被消除」，对**宿主状态**而言则零侵入。
# 见 lingshu/_pathguard.py（含精确范围声明与逃生口 LINGSHU_ALLOW_CWD_IMPORTS）。
from .. import _pathguard as _ling_pathguard  # noqa: F401  (import 即执行，勿删)

import sqlite3
import json
import time
import math
import uuid
import os
import sys
import warnings
import hmac as _hmac
import threading
import weakref
import functools
import contextlib
# 结构事件工作单元用裸名 contextmanager（PR #429）；与上面的 contextlib 同一模块。
from contextlib import contextmanager
from typing import Optional, List, Dict, Any, Tuple, Set
from dataclasses import dataclass, field, asdict, fields as _dc_fields
from enum import Enum

# 统一时间核（钉死批条款3：衰减核形状唯一，实现收口 aeis/time_core.py）
try:
    from .time_core import cred, cred_step
except ImportError:  # 直跑 fallback（裸名互导）
    from .time_core import cred, cred_step

# 组件发现面收口（issue #156）：白名单组件的裸名导入不再经 cwd/空串 sys.path 条目解析。
# v2 精确范围：① 本包 import 期由 _pathguard 消除 cwd/空串 sys.path 解析（覆盖**一切**
# 经 sys.path 的裸名导入，含标准库——上一版只封白名单组件名，json.py 等仍被劫持）
# ——「消除」取**解析期拒绝**义（#273 路 B）：条目不被删除，宿主 `sys.path` 零改写；
# ② 白名单组件另由本守卫收口（含排除脚本目录）；③ 仍不覆盖：进程已被预置投毒
# （sys.modules 预置 / 进程内 sys.path.insert）。逃生口 LINGSHU_ALLOW_CWD_IMPORTS
# 只放开 ① 层（非白名单名的 cwd 解析面）；② 层自 #750 残余修复起不随逃生口
# 放开——白名单名受控面未命中即失败，不再交还标准 PathFinder（A/D/E 臂封死）。
from . import contradiction as _contradiction

from .component_resolver import (
    COMPONENT_NAMES as _COMPONENT_NAMES,
    configured_roots as _configured_component_roots,
    discovery_report as _component_discovery_report,
    install as _install_component_resolver,
)
_install_component_resolver()


# =============================================================================
# 设计者认证（D-007 用户身份识别·最小版）
# =============================================================================

def designer_key_configured() -> bool:
    """设计者密钥是否已配置（AEIS_DESIGNER_KEY 环境变量）。"""
    return bool(os.environ.get("AEIS_DESIGNER_KEY"))


def verify_designer(designer_key) -> bool:
    """设计者密钥验证（fail-closed）：未配置密钥或密钥不匹配一律拒绝。

    事实边界（issue #156 修订；原表述为「密钥仅存在于服务环境变量，
    模型/自动化无法读取，因此永远无法冒充」）：
    - 成立：判据 = 调用方传入值 == **本进程** 环境变量值（AEIS_DESIGNER_KEY）；
      密钥不在仓库与代码中，仅存在于服务环境。
    - 不成立：**同进程代码**（如构造期被装入的组件、被劫持的裸名导入）可读
      os.environ ⇒「模型/自动化永远无法读取」对同进程不成立。
    本批（#156 v2）的精确范围＝**消除 cwd/空串 `sys.path` 解析**——覆盖一切经
    `sys.path` 的裸名导入（含标准库），并另由组件守卫收口白名单 15 名（含排除
    脚本目录）。★机制（#273 路 B 起）＝**解析期拒绝**，不是删除条目：cwd/空串
    条目**仍在 `sys.path` 里**（`scrub()` 不再改写宿主 `sys.path`），由 `_pathguard`
    装在 `sys.meta_path` 的 finder 在解析时跳过并抛 `ModuleNotFoundError`。
    仍**不覆盖**：进程已被预置投毒（`sys.modules` 预置 / 进程内
    `sys.path.insert` / 别名冒充）、逃生口 `LINGSHU_ALLOW_CWD_IMPORTS` 被显式
    打开的情形。同进程可信性属**设计级**事项，待 D-007 后续裁定。"""
    expected = os.environ.get("AEIS_DESIGNER_KEY")
    if not expected or not designer_key:
        return False
    # #51：`hmac.compare_digest` 对**非 ASCII str** 抛 TypeError（CPython 文档化
    # 行为）⇒ 中文密钥下终裁门禁从「干净拒绝」变成「崩」（6 条终裁通路全部
    # 不可用）。修法：统一 encode 成 bytes 再比较——时序安全语义不变、对任意
    # Unicode 密钥成立；不匹配 ⇒ False（fail-closed 原样），不再抛。
    return _hmac.compare_digest(str(designer_key).encode("utf-8"),
                                str(expected).encode("utf-8"))


# main 侧 #29② 判据的原因名 → 矛盾感知（contradiction.py）的原因名，供 conflict:{reason} 标签统一口径
_MAIN_DIVERGENCE_REASON = {"negation": "polarity", "numeric": "number"}


# =============================================================================
# 条件空间（四维声明）
# =============================================================================

@dataclass
class ConditionSpace:
    """条件空间声明 · 协议 0.0.4 节"""
    observation_position: str       # 观测位置
    observation_tool: str           # 观测工具
    time_window: Tuple[float, float] # (开始时间, 结束时间)
    existence_constraint: str       # 存在约束

    def to_json(self) -> str:
        """条件空间序列化为 JSON（节点/边落库时条件字段的标准形态）。"""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> 'ConditionSpace':
        """从 JSON 反序列化条件空间；外部卡缺字段时补默认（v1.22 健壮性）。

        #267：**多余键**（外部卡/损坏备份多写一个键）此前会让 `cls(**d)` 抛
        `TypeError`，进而毒化整层装载。现按已声明字段过滤未知键（只丢未知键、
        保留已知键）——判据来源：分诊表 `triage_lingshu_ALL.md` 行 267
        「应逐行容错并如实计数」。边界（削掉的判别力）：未知键被静默丢弃、
        **不**计入逐行容错计数（计数面在节点行级，见 `_tolerant_nodes`）。
        """
        d = json.loads(s)
        # v1.22 健壮性：外部卡 condition_space 可能缺字段 → 补默认
        # （否则 KeyError/TypeError 崩掉整条检索链）
        known = ('observation_position', 'observation_tool',
                 'time_window', 'existence_constraint')
        if isinstance(d, dict):
            d = {k: v for k, v in d.items() if k in known}   # #267：丢未知键
        else:
            d = {}
        d.setdefault('observation_position', '外部观测位')
        d.setdefault('observation_tool', '文本语义分析')
        d.setdefault('existence_constraint', '（未声明）')
        if 'time_window' not in d:
            d['time_window'] = (0.0, 0.0)
        elif not isinstance(d['time_window'], (tuple, list)):
            d['time_window'] = (d['time_window'], d['time_window'])
        d['time_window'] = tuple(d['time_window'])
        return cls(**d)


# =============================================================================
# tags 列序列化（唯一真源 · issue #378）
# =============================================================================

def _dumps_tags(tags, ensure_ascii: bool = False) -> str:
    """序列化 `nodes.tags` 列（唯一真源）。拒绝含 UTF-16 代理码点的标签。

    #181：默认 `ensure_ascii` 由 `True` 改 `False`。旧默认下中文标签写成
    `"\\u82f9\\u679c"` 转义，而 `get_nodes_by_tag` / `tags LIKE ?` 拿中文原文匹配
    ⇒ **永远 0 条**；同列另有两处（`insight_verify`、`update_insight_importance`）
    显式 `ensure_ascii=False` 写中文原文 ⇒ **同一列两种编码混存**，任何一种查法都
    漏掉另一半。本函数是全部 tags 写点的唯一真源（5 处），改默认即一处收口。
    判据来源：分诊表 `triage_lingshu_ALL.md` 行 181（复现：`get_nodes_by_tag("苹果")`
    返回 0；建议「统一用一个 `_dumps = partial(json.dumps, ensure_ascii=False)`
    写入所有 JSON 列」）＋ 同文件既有 `ensure_ascii=False` 先例（`insight_verify` /
    `update_insight_importance`）。
    边界（削掉的判别力）：本件只统一**写**口径为 `ensure_ascii=False`；**不**迁移
    既有库中已按旧默认写入的转义行（那些行对中文标签查询仍不可见，需另跑一次性
    迁移）；亦不把 `tags LIKE ?` 的子串匹配改成 `json_each` 精确匹配（后者是
    另一条 issue 的面，见 #132）。

    #378：`json.dumps(tags)` 默认 `ensure_ascii=True`，孤立代理（unpaired
    surrogate，如 `"\\ud83d"`）会被**静默转义**成字面量 `\\ud83d` 文本入库——
    SQLite 侧不报错、`json.loads` 往返也不报错，但该标签此后 ① 任何
    `ensure_ascii=False` 的再序列化（core.py:2110 一带）抛
    `UnicodeEncodeError: surrogates not allowed`；② 写 UTF-8 文件/日志即抛
    `UnicodeEncodeError` ⇒ 检索与导出链被单条毒标签炸掉。

    判据：含 U+D800–U+DFFF 任一码点即拒（fail-closed，`ValueError` 点名 #378）。
    该口径**含已配对**的代理（Python str 中 `"\\ud83d\\ude00"` 仍是两个代理码点、
    仍无法 UTF-8 编码；真正的星面字符是单一码点，不受影响）——即比「孤立代理」
    更严一档，覆盖面只增不减。**不**做静默替换（替换会把污染源洗白、掩盖上游
    解码错误）。判据来源：经验标定（本件 #378），仓内无规定 tags 编码面的理论
    章节，追不到更早出处。
    """
    for t in tags:
        if isinstance(t, str) and any(0xD800 <= ord(ch) <= 0xDFFF for ch in t):
            raise ValueError(
                "#378 拒绝含 UTF-16 代理码点的标签（无法 UTF-8 编码，会毒化检索/导出）: "
                "%r" % (t,))
    return json.dumps(tags, ensure_ascii=ensure_ascii)


# =============================================================================
# add_perception 入参验证闸（tags/entities/modality · issue #805）
# =============================================================================

# #805：modality 白名单——感知通道（add_perception）的合法模态。取值锚点：
# ① `STNode.modality` 字段注释「模态：text, image, code, etc.」枚举出的三个
#   非 text 值中的 image/code（"etc." 是开放词，只取已实证项）；
# ② `add_perception` 去重判据 #223 的 audio（候选集按 modality 过滤的显式
#   测试值，tests/test_issue223_dedup_merge_fields_guard.py）；
# ③ `provenance.from_legacy` 的 image（vision_pipeline 感知通道）；
# ④ `subgraph_replace._attach` 的 visual（core.py 子树挂载的缺省 modality，
#    tests/test_issue139_subgraph_replace_*.py 的 g5/g6 正对照在用）。
# structure/protocol/self_state/trust_state 等模态是**写入路径内部值**——它们
# 全部直构 `STNode(...)` 走 `add_node`（core.py 结构事件/锚点/自我快照等），
# 不经 `add_perception` 感知入口，故不在本白名单内。扩展白名单是一行改动。
_PERCEPTION_MODALITIES = frozenset({"text", "audio", "image", "code", "visual"})


def _validated_str_items(name: str, value):
    """#805：`add_perception` 的 tags/entities 入参校验+规范化（唯一真源）。

    返回规范化后的 `list[str]`：
      · `value is None` → `[]`（None＝「未提供」，与签名默认 `List[str] = None`
        同义；仓内既有合法用法：longterm_gate.py 两处 `entities or None` 显式
        传 None 表空）；
      · 容器须为 list/tuple/set/frozenset——裸 str/bytes 拒（旧形态：裸字符串
        被当可迭代，`'tag1'` 落库成 `['t','a','g','1']` 四个单字符 tag），
        其余非容器类型拒（旧形态：int 在去重路径的极性闸 `set()` 处抛无说明
        TypeError 且 `_interaction_count` 副作用已发生）；
      · 逐项：**None 项跳过**（issue #805 判据原文「None 项跳过」——None 是
        「无」不是值，不字符串化）；非 str 项 `TypeError`（旧形态：数字被
        `f"ent:{eid}"` / JSON 静默字符串化，正是 issue 点名的「数字变字符串」）；
        空串项 `ValueError`（issue 表格点名「空 tag 无意义但被接受」）。

    与 #331/#588 的 content 闸同口径：fail-closed，**不做 `str()` 静默强转**——
    那会把编程错误洗成合法数据（「静默篡改语义」，比拒绝更坏）。
    判据来源：公开 issue #805「tags 每项须为非空 str，None 项跳过；
    modality 白名单校验」。
    """
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(
            value, (list, tuple, set, frozenset)):
        raise TypeError(
            "#805 add_perception 拒绝 %s 的容器类型 %s：须为 list/tuple/set"
            "（裸字符串会被逐字符炸成单字符 tag；非容器不可迭代）"
            % (name, type(value).__name__))
    out = []
    for item in value:
        if item is None:
            continue  # issue 判据：「None 项跳过」
        if not isinstance(item, str):
            raise TypeError(
                "#805 add_perception 拒绝 %s 中的非字符串项 %r（%s）："
                "数字/对象会被静默字符串化混进 tags 列，按 tag 过滤永远漏"
                % (name, item, type(item).__name__))
        if item == "":
            raise ValueError(
                "#805 add_perception 拒绝 %s 中的空字符串项：空 tag 无意义"
                "（占据 get_nodes_by_tag 匹配空间与去重极性比对）" % name)
        out.append(item)
    return out


# =============================================================================
# JSON 列序列化闸（唯一真源 · issue #494）
# =============================================================================
#
# #378 只把「代理码点拒写」落在 `nodes.tags` 一列（`_dumps_tags`，递归深度一层）。
# 而 `nodes` 表另有三列是 JSON 文本：`state_attributes` / `semantic_coordinates`
# / `spatial_coordinates`。它们的写点当时仍是**裸** `json.dumps`（默认
# `ensure_ascii=True`）⇒ 孤立代理（unpaired surrogate，如 `"\ud83d"`）被**静默
# 转义**成字面量 `\ud83d` 文本入库：SQLite 不报错、`json.loads` 往返也不报错，
# 于是毒行潜伏；直到读回后再以 `ensure_ascii=False` 序列化写回（`_write_influence`
# / `insight_verify`）或写 UTF-8 文件/日志时才抛
# `UnicodeEncodeError: surrogates not allowed` ⇒ 单条毒行炸掉整条写回/导出链。
#
# 本闸把 #378 的判据（含 U+D800–U+DFFF 任一码点即拒，fail-closed，`ValueError`）
# 从 tags 一列扩到上述三列的统一序列化真源。**只拒绝，不洗白**（不做静默替换/
# 剥离——那会把污染源洗白、掩盖上游解码错误）。`ensure_ascii` 缺省保持 `True`
# （与 `to_row` 既有裸调一致，不改存储编码口径；那是 #181 的面）。

def _reject_surrogates(value, where: str) -> None:
    """递归检查 value（dict 的键与值 / list / tuple / set / str）是否含
    UTF-16 代理码点 U+D800–U+DFFF；含则 ValueError（fail-closed）。"""
    stack = [value]
    while stack:
        cur = stack.pop()
        if isinstance(cur, str):
            if any(0xD800 <= ord(ch) <= 0xDFFF for ch in cur):
                raise ValueError(
                    "#494 拒绝含 UTF-16 代理码点的 %s 值（无法 UTF-8 编码，"
                    "会毒化检索/导出，且读回再序列化即抛 UnicodeEncodeError）: %r"
                    % (where, cur))
        elif isinstance(cur, dict):
            stack.extend(cur.keys())
            stack.extend(cur.values())
        elif isinstance(cur, (list, tuple, set, frozenset)):
            stack.extend(cur)


def _dumps_json(value, ensure_ascii: bool = True, where: str = "JSON 列") -> str:
    """`nodes` 表 JSON 列的统一序列化真源（#494）：先过代理码点闸，再 json.dumps。"""
    _reject_surrogates(value, where)
    return json.dumps(value, ensure_ascii=ensure_ascii)


def _json_tags(value) -> list:
    """把 `nodes.tags` 的**库内/备份取值**归一为 list（#778 F1/F2 判据的读取助手）。

    为什么需要：`tags` 列是 TEXT（JSON 文本）——库内读回是 `'["a","b"]'`、JSON `null`
    读回是 `None`（#718 实测）、备份被篡改时也可能是真 list。保护标记保真判据要
    「旧行 tags 是否含 `no_forget`」，故须先把三种形态统一成 list。
    容错口径：非 list / 非法 JSON / None ⇒ `[]`（与 #718「`tags=None` 视为无该标记」
    同向——**不**抛、不猜）；合法 JSON 但非 list（如字符串）亦归 `[]`。
    边界（如实标注）：本助手**只读**，不写库；改写只在确需并入保护标记时经
    `_dumps_tags`（tags 列唯一序列化真源）进行。
    """
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return []
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


# =============================================================================
# 数值入参闸（唯一真源 · lingshu #185）
# =============================================================================

def _finite_or(x, default: float) -> float:
    """有限则返回 `float(x)`，否则（非数 / NaN / ±inf）返回 `default`。见 #185。"""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _require_finite(name: str, x) -> float:
    """入参必须是**有限实数**，否则 `ValueError`（fail-closed · lingshu #206）。

    与 `_finite_or`/`_finite_clamp` 的**归一**口径相反：本闸**拒绝**而非回落。
    为什么此处必须 fail-closed：`update_trust_state` 写入的 `t_total` 经
    `LongTermMemoryGate._trust()` 以 0.25 权重进入长期层评分，而 `evaluate` 的
    `imp = max(0.0, min(1.0, imp))` 对 `NaN` **不生效**（Python `min(1.0, nan)`
    返回 1.0）⇒ 一次 `NaN` 即让**任意**快照以 imp=1.0 进长期层并 `protect_node`
    （不可遗忘、不可回收）；`±inf` 同效。坏值若被「归一」成默认值，调用方拿不到
    任何信号，且 `trust_state` 会静默变成一个假的合法值。
    判据来源：公开 issue #206 建议 1（「update_trust_state 对 t_total / p_trust /
    p_gap 做 math.isfinite 检查与值域校验」）；`_trust()` 的既有声明
    「当前信任值 T（0-1）」（`lingshu/core/longterm_gate.py:66-67`）。
    边界（削掉的判别力）：本闸**只**拒非有限值；`t_total` 的**值域**（[0,1] 归一
    还是累计量）未在此收口——见 `SelfModel.update_trust_state` docstring。
    """
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"{name} 必须是实数，收到 {type(x).__name__}: {x!r}")
    if not math.isfinite(v):
        raise ValueError(f"{name} 必须是有限实数（拒绝 NaN/±inf），收到 {x!r}")
    return v


def _finite_clamp(x, lo: float, hi: float, default: float = None) -> float:
    """把浮点入参归一为 [lo, hi] 内的**有限**值（非有限/非数 → `default`）。

    lingshu #185：`importance` / `confidence` / `delta` 等浮点入参此前**零校验**，
    直接落库 ⇒ ① `NaN` 被 SQLite 存成 **NULL**：`decay_cycle` 的
    `WHERE importance > ?` 永远选不中 NULL ⇒ 该节点**永不遗忘**；落到边则
    `cred_step(None, f)` 抛 `TypeError` ⇒ 整轮衰减中断（并可致后台线程退出，见
    #184）。② `+inf` 原样存入 ⇒ `inf × 0.98 = inf` **永生**，且在一切
    `ORDER BY importance DESC` 检索里**永远排第一**、挤占 LIMIT 窗口。
    ③ SQL 侧 `MIN(1.0, MAX(0.0, x))` 对 NULL 无效（`MIN/MAX` 遇 NULL 仍回 NULL）。

    归一口径（工程 fail-closed）：非有限值（NaN/±inf）与非数一律回落到 `default`
    （缺省取区间下界 `lo`），有限值再钳到 [lo, hi]。**阈值本体未改**——本闸只
    堵「坏数值入参」，不改变任何既有判据的边界值。
    判据来源：公开 issue #185（「在 STNode / STEdge 的构造处（或 to_row）统一校验：
    `math.isfinite` 不成立就抛 ValueError，并钳制到 [0,1]」）＋ 同仓既有同族先例
    `tests/test_issue402_nan_clamp_guard.py`（「非有限输入不得被当作有效证据/支持」，
    阈值一律未改）。
    边界（削掉的判别力）：①取**归一**而非 `ValueError` 拒绝——与本文件既有
    `add_node`（#164）的钳制口径一致，调用方传坏值不会收到报错信号；
    ② `STNode.to_row` 的其余 JSON 列（`semantic_coordinates` / `state_attributes`
    的 `allow_nan`）不在本件范围内（它们不参与遗忘门控与排序）。
    """
    if default is None:
        default = lo
    return max(lo, min(hi, _finite_or(x, default)))


def _to_int_or(raw, default: int, lo: int = 1) -> int:
    """把**可能来自盘面**的整数值入参归一为 `>= lo` 的 int（坏值 → `default`）。

    lingshu #655：`context_max` 的读取面与设置面此前各写一份
    `int(float(raw))`，`except` 只捕 `(TypeError, ValueError)`。但
    `float("1e400") == inf`、`float(10**400)` 溢出 ⇒ `int(inf)` 抛
    **`OverflowError`**（不被捕获）：
    ① 盘面 `engine_meta["context_max"]="1e400"` ⇒ `SpacetimeMemoryEngine()`
    **构造即崩**（整个库打不开）；
    ② `set_context_cap(float("inf"))` 抛 `OverflowError`，且该坏值会被
    `set_meta` 写进盘面 ⇒ 把 ① 变成持久故障。
    归一口径（工程 fail-closed）：坏值回落到 `default`（**不抛异常**），与
    同文件 `_finite_clamp` 的归一惯例一致；正常值再钳到 `>= lo`。**只此一处
    实现**——读取面与设置面共用，避免同一表达式两处各修一次而漏一处。
    判据来源：公开 issue #655。
    边界（削掉的判别力）：调用方传坏值不会收到报错信号。
    """
    try:
        return max(lo, int(float(raw)))
    except (TypeError, ValueError, OverflowError):
        return default


def _is_meta_category(category) -> bool:
    """`category` 规范化（`strip().lower()`）后是否等于 `"meta"`（lingshu #386）。

    「元盲区零记录」判据此前只在引擎门面按 `category == "meta"` **字面**比较 ⇒
    `'Meta'` / `'META'` / `' meta'` / `'meta\\n'` 都能绕过（被改标为 operational 写进
    注册表并进入学习队列）；`LayeredStore.add_blindspot` 更无任何检查。
    判据来源：公开 issue #386 期望原文「所有规范化后等于 meta 的 category
    （`strip().lower()=="meta"`）都应被拒绝；store 层也应有同样的检查」。
    边界（削掉的判别力）：只按空白/大小写规范化；**不**做同义词/翻译匹配
    （如 `'元'`）——issue 未要求，且会引入新的判据面。
    """
    return isinstance(category, str) and category.strip().lower() == "meta"


def _flat_semantic_to_nested(flat) -> Dict:
    """把扁平的语义键（`protocol_*` / `radical_*` / `neural_*`）转成
    `semantic_coordinates` 的嵌套结构（lingshu #415）。

    映射与 `migrate_v17_coordinates`（D-003 坐标字段分离）**逐条一致**：
    `protocol_X` → `protocol.concept.X`、`radical_X` → `radical.X`、
    `neural_X` → `neural.X`。非前缀键原样保留在顶层（不丢信息，亦不臆造新桶）。
    判据来源：`migrate_v17_coordinates` 的既有映射（本文件 D-003 迁移段）＋
    公开 issue #415「将 provider 输出转换为当前语义坐标结构后写入
    `semantic_coordinates`」。
    边界（削掉的判别力）：只做前缀分桶，**不**校验值类型/范围（语义坐标的值域
    由 provider 契约负责，本函数不做归一）。
    """
    out: Dict = {}
    for k, v in dict(flat or {}).items():
        if k.startswith("protocol_"):
            out.setdefault("protocol", {}).setdefault("concept", {})[k[len("protocol_"):]] = v
        elif k.startswith("radical_"):
            out.setdefault("radical", {})[k[len("radical_"):]] = v
        elif k.startswith("neural_"):
            out.setdefault("neural", {})[k[len("neural_"):]] = v
        else:
            out[k] = v
    return out


#: `check_escalation` 数值界解析支持的关系算符（先长后短，避免 `>` 吃掉 `>=`）。
_ESC_VALUE_OPS = (">=", "<=", "==", "!=", ">", "<")


def _find_standalone_word(text: str, word: str) -> int:
    """返回 `word` 在 `text` 中作为**独立词**（前后非标识符字符）出现的下标，无则 -1。

    「独立」判据：前一位与后一位都不是 `[A-Za-z0-9_]`——故 `value` 不会命中
    `values` / `my_value`（#57 的 condition 解析须只认参数名本体）。
    """
    start = 0
    while True:
        i = text.find(word, start)
        if i < 0:
            return -1
        before = text[i - 1] if i > 0 else ""
        after = text[i + len(word)] if i + len(word) < len(text) else ""
        if not (before.isalnum() or before == "_") and \
                not (after.isalnum() or after == "_"):
            return i
        start = i + 1


def _scan_esc_number(text: str, i: int):
    """从 `text[i]` 起扫描**完整数字词元**，返回其结束下标；扫描不出返回 `None`。

    词元形态：`[+-]? ( digits [ "." digits* ] | "." digits ) [ ("e"|"E") [+-]? digits ]`
    —— 指数段**必须**带数字才被吸收（`"1e"` / `"1e+"` 不算指数，其尾随由调用方
    判为截断）。

    ★#747：旧实现只吃 `isdigit()` 与 `"."`，`1e3` 被读成 `1`（**静默截断**到
    `1.0`）。本函数把词元读**全**（含指数段）——词元之后是否合法由
    `_parse_escalation_value_bound` 按尾随字符判。
    """
    n = len(text)
    j = i
    if j < n and text[j] in "+-":
        j += 1
    int_start = j
    while j < n and text[j].isdigit():
        j += 1
    has_int = j > int_start
    if j < n and text[j] == ".":
        j += 1
        frac_start = j
        while j < n and text[j].isdigit():
            j += 1
        if not has_int and j == frac_start:
            return None
    elif not has_int:
        return None
    if j < n and text[j] in "eE":
        k = j + 1
        if k < n and text[k] in "+-":
            k += 1
        exp_start = k
        while k < n and text[k].isdigit():
            k += 1
        if k > exp_start:
            j = k
    return j


def _esc_has_second_comparison(text: str) -> bool:
    """`text` 里是否还有第二个比较符、或独立词 `and`/`or`。

    #747：本解析器只认**单个**简单比较。`"value > 0.1 and value < 0.2"` 这类复合
    条件若只取首个界，会让后半静默失效 ⇒ 检出即让调用方**回落**（`None`），不猜。
    """
    if any(ch in "<>=!" for ch in text):
        return True
    return (_find_standalone_word(text, "and") >= 0
            or _find_standalone_word(text, "or") >= 0)


def _parse_escalation_value_bound(condition: str):
    """从升级点 `condition` 文本解析 `value <op> <number>` 数值界（#57）。

    返回 `(op, threshold)`（op ∈ `_ESC_VALUE_OPS`，threshold 为 float）；解析不出
    返回 `None`（调用方回落旧行为）。

    ★**手写解析器**，不使用 `eval()`/`exec()`（任务书铁律 9）。只认**左操作数显式
    为参数名 `value`** 的比较——`ESC-004` 的 `"...deviation>0.3"` 左操作数是
    `deviation` 而非 `value`，**不**被判为数值界（否则会给既有 token 用法
    `check_escalation("deviation")` 引入「无值即不匹配」的回归）。
    允许前后有其它文字（如 `"value > 0.9 时上报"`）。

    ★#747（本件）：数字扫描原先只吃 `isdigit()` 与 `"."`，`1e3` 被读成 `1`
    （**静默截断**成 `> 1.0`）——含指数记法/尾随子串的写法会被静默钳到前缀值，
    从而反转触发语义。现改为读**完整数字词元**（`_scan_esc_number`，含指数段），
    并把三类「读不准」显式回落为 `None`：
    · **词元被截断**：其后紧跟标识符字符或点（`"0.5xyz"` / `"1and2"` / `"1.2.3"`）；
    · **复合条件**：词元之后还有第二个比较符或独立词 `and`/`or`（如
      `"value > 0.1 and value < 0.2"`）——**不**只取首个界（那会让后半静默失效）；
    · **不可表示**：`float()` 溢出为 `±inf`（`"1e400"`）。
      判据是**失效方向**：`value < inf` 对一切有限值恒真 ⇒ 该升级点**无条件触发**
      （危机触发器里最危险的方向）；`value > inf` 恒假。两者都不是书写者能预期的
      界 ⇒ 一律 `None` 回落，不猜。
    边界（削掉的判别力，如实标注）：
    ① 下溢（`"1e-400"` → `0.0`）**不**拒绝：`0.0` 是 IEEE 对该字面量的正确近似，
       且两侧（`>`/`<`）都与书写意图同向、不会造成无条件触发。故 `1e-400` 解析为
       `('>', 0.0)`。
    ② `None` 的语义仍是「回落字符串包含匹配、`value` 不参与」（#57 既有契约）
       ——对复合条件/畸形尾随/不可表示而言，这不是**更严**的取向，只是**不猜**；
       若某条升级点的 `condition` 文本恰好含信号词，回落路径仍可能按文本命中。
       要求值复合条件需表达式引擎，属架构级选择，不在本件范围。
    ③ 只认单个简单比较；`==`/`!=` 同受②约束（`"value == 0.5 and ..."` 亦回落）。
    """
    if not condition:
        return None
    idx = _find_standalone_word(condition, "value")
    if idx < 0:
        return None
    i = idx + len("value")
    while i < len(condition) and condition[i] in " \t":
        i += 1
    op = None
    for cand in _ESC_VALUE_OPS:
        if condition.startswith(cand, i):
            op = cand
            break
    if op is None:
        return None
    i += len(op)
    while i < len(condition) and condition[i] in " \t":
        i += 1
    j = _scan_esc_number(condition, i)
    if j is None:
        return None
    try:
        threshold = float(condition[i:j])
    except ValueError:
        return None
    if not math.isfinite(threshold):
        return None                       # 溢出：1e400 ⇒ ±inf（不猜，见 docstring）
    tail = condition[j:]
    if tail[:1].isalnum() or tail[:1] in ("_", "."):
        return None                       # 词元被截断：0.5xyz / 1and2 / 1.2.3
    if _esc_has_second_comparison(tail.lstrip(" \t")):
        return None                       # 复合条件：只认单个简单比较
    return (op, threshold)


def _eval_escalation_value_bound(op: str, threshold: float, value) -> bool:
    """按解析出的算符比较 `value` 与 `threshold`（#57 求值侧，纯函数）。"""
    if op == ">":
        return value > threshold
    if op == ">=":
        return value >= threshold
    if op == "<":
        return value < threshold
    if op == "<=":
        return value <= threshold
    if op == "==":
        return value == threshold
    if op == "!=":
        return value != threshold
    return False


# =============================================================================
# 节点与边类型
# =============================================================================

class NodeType(Enum):
    """节点类型（认知图实体分类）：实体/感知/概念/动作/状态/自我六类。"""
    ENTITY = "entity"
    PERCEPTION = "perception"
    CONCEPT = "concept"
    ACTION = "action"
    STATE = "state"
    SELF = "self"

class EdgeType(Enum):
    """边类型（认知图关系分类）：因果/时序/相关/循环/层级/空间三态/相似/相反/适用。"""
    CAUSAL = "causal"
    SEQUENTIAL = "sequential"
    CORRELATIONAL = "correlational"
    CYCLIC = "cyclic"
    HIERARCHICAL = "hierarchical"   # v1.16：知识归属/包含层级（卡⊃知识点、元学科⊃学段）
    SPATIAL_ADJACENT = "spatial_adjacent"
    SPATIAL_CONTAINS = "spatial_contains"
    SPATIAL_CONNECTED = "spatial_connected"
    SIMILAR = "similar"
    OPPOSITE = "opposite"
    APPLIES_TO = "applies_to"   # v1.16（条件节点化）：条件 → 知识 的适用关系

class MemoryLayer(Enum):
    """记忆五层（遗忘纪律）：锚点/结构层不可遗忘；知识/情境层可衰减；自我层可更新。"""
    ANCHOR = "anchor"       # 不可遗忘
    STRUCTURE = "structure" # 不可遗忘
    KNOWLEDGE = "knowledge" # 可衰减
    CONTEXT = "context"     # 可衰减
    SELF = "self"           # 可更新

# =============================================================================
# 时空节点
# =============================================================================

@dataclass
class STNode:
    """时空节点（认知图顶点）：内容+模态+空间坐标+时间戳+条件空间五要素。"""
    id: str
    content: str
    modality: str                       # 模态：text, image, code, etc.
    spatial_coordinates: Dict[str, float] # 空间/几何坐标（视觉/物理 · D-003 分离）
    temporal_coordinate: float           # 时间戳
    condition_space: ConditionSpace
    importance: float = 0.5             # 重要性 [0,1]
    confidence: float = 0.5             # 置信度 [0,1]
    layer: MemoryLayer = MemoryLayer.KNOWLEDGE
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)
    semantic_coordinates: Dict = field(default_factory=dict)   # v1.7：符号/语义投影（D-003）
    state_attributes: Dict = field(default_factory=dict)       # v1.7：形容词（图像时间变化）
    entity_id: Optional[str] = None                            # v1.7：名词（实体挂接）
    # ── `entity_id` 语义声明（公开 issue #766 定性收口 · 维护者端裁定 2026-10-17）──
    # **本字段是「名词挂接 / 标注」，不是数据隔离边界。**
    #
    # 面内定位：与 `spatial_coordinates`（视觉坐标）、`state_attributes`（形容词）
    # 并列的 v1.7「名词」面——记录「这条记忆挂在哪个实体名下」，供展示与挂接。
    #
    # 检索面**零消费**（实证，基线 origin/main 逐符号核过）：
    #   · `SpacetimeMemoryEngine.recall` 的候选集与终排无 `entity_id` 条件；
    #   · `LayeredStore.search_content`（LIKE 路径）的 SQL WHERE 只由 `layer IN (…)`
    #     与 LIKE 片段构成；
    #   · `FTSIndex.candidates`（FTS 路径）同样无该列；
    #   · `lingshu/core/retrieval.py` 内本字段引用数为 **0**。
    # ⇒ 异实体（含**另起进程**的实例）可经 `recall` / `search_content` 原样取回
    #   ——这是**设计现状**，不是缺陷（判据：本仓不把库级 / 实例级字段当安全边界）。
    #
    # 读侧唯一的消费点都在**展示面**（均非过滤 / 鉴权）：`render_semantic_cube_3d`
    # 的轨迹过滤、`query_by_semantics` 的实体名加分。
    #
    # 同款口径先例：本文件 `Role` docstring 的信任边界声明（角色是实例属性、
    # 不构成安全边界）＋ 仓内既有明文「本层的角色 / 会话过滤是消费作用域，
    # **不是身份认证边界**」。
    #
    # ⇒ 若日后要把它**升格**为隔离边界，属**设计取向变更**（不是「补一个漏掉的闸」）：
    #   还缺一格前置——**调用方实体从哪里来**（引擎构造器只有 `identity` / `role`，
    #   `nodes` 表无属主 / 实例列）。取向与落点见 `docs/architecture.md` §六 6.4。

    def to_row(self) -> tuple:
        """节点序列化为 SQLite 行元组（列序=nodes 表 schema）。

        #185：`importance` / `confidence` 经 `_finite_clamp` 归一（非有限值回落
        0.0/0.5 缺省、有限值钳到 [0,1]）——见 `_finite_clamp` docstring。
        `temporal_coordinate` 非有限时回落 **`created_at`**（同列族的时间戳兜底）。

        ★回落目标为何是 `created_at` 而非 `last_access`（2026-10-11 修正）：
        本列在此前（P1-core 批次初版）回落 `last_access`，那是**写库时刻的
        wall-clock**——逐节点不同，且与库外时钟分辨率（Windows `time.time()`
        ≈15.6ms）绑定 ⇒ 同一批写入的节点可能落进不同 tick，`temporal_coordinate`
        从「同一时刻」被拆成两个值，`time_radius=0` 的同刻查询**时红时绿**
        （实测：两次写入间 sleep 0.05s 即稳定复现红；不 sleep 则绿）。
        而读侧 `spatiotemporal_query` 的既有回落口径本就是
        `node.temporal_coordinate if ... is not None else node.created_at or 0`
        ——`created_at` 才是本仓既定的时间兜底真源；改回它即让「写侧归一」与
        「读侧兜底」同口径，且逐节点确定（不再随时钟 tick 抖动）。
        判据来源：既有回归 `tests/test_spatiotemporal_zero_radius.py::
        test_created_at_fallback_does_not_replace_explicit_zero_timestamp`
        （其名即「created_at fallback」）＋ 读侧 `spatiotemporal_query` 的同款兜底。
        边界（削掉的判别力）：`created_at` 为 0/None 的极端节点仍会落 0.0，
        不另造第三种兜底——保持「写侧只归一、兜底真源唯一」。
        """
        return (
            self.id, self.content, self.modality,
            _dumps_json(self.spatial_coordinates, where="spatial_coordinates"),
            _finite_or(self.temporal_coordinate, self.created_at),
            self.condition_space.to_json(),
            _finite_clamp(self.importance, 0.0, 1.0),
            _finite_clamp(self.confidence, 0.0, 1.0),
            self.layer.value, self.access_count,
            self.last_access, self.created_at,
            _dumps_tags(self.tags),
            _dumps_json(self.semantic_coordinates, where="semantic_coordinates"),
            _dumps_json(self.state_attributes, where="state_attributes"),
            self.entity_id
        )

    @classmethod
    def from_row(cls, row: tuple) -> 'STNode':
        """从 SQLite 行元组反序列化节点（from_row 惰性字段解析）。"""
        return cls(
            id=row[0], content=row[1], modality=row[2],
            spatial_coordinates=json.loads(row[3]),
            temporal_coordinate=row[4],
            condition_space=ConditionSpace.from_json(row[5]),
            importance=row[6], confidence=row[7],
            layer=MemoryLayer(row[8]),
            access_count=row[9], last_access=(row[10] if row[10] is not None else row[11]),
            created_at=row[11], tags=json.loads(row[12]),
            semantic_coordinates=json.loads(row[13]) if len(row) > 13 and row[13] else {},
            state_attributes=json.loads(row[14]) if len(row) > 14 and row[14] else {},
            entity_id=row[15] if len(row) > 15 else None
        )


# =============================================================================
# 时空边
# =============================================================================

@dataclass
class STEdge:
    """时空边（认知图有向边）：源→目标+关系类型+条件空间+置信度——条件挂在边上。"""
    id: str
    source_id: str
    target_id: str
    relation_type: EdgeType
    condition_space: ConditionSpace
    confidence: float = 0.5
    weight: float = 1.0               # 边权重
    verified: bool = False            # 是否经过验证单元复核
    created_at: float = field(default_factory=time.time)
    last_verified: Optional[float] = None
    source_evidence: str = "extracted"   # v1.11 P1-1：extracted/inferred/ambiguous

    def to_row(self) -> tuple:
        """边序列化为 SQLite 行元组（列序=edges 表 schema）。

        #185：`confidence` / `weight` 经 `_finite_clamp` 归一（见 `_finite_clamp`）。
        `confidence` 落 NULL 会让 `decay_cycle` 的 `cred_step(None, f)` 抛
        `TypeError`，整轮衰减中断（并可致后台线程退出，见 #184）。
        """
        return (
            self.id, self.source_id, self.target_id,
            self.relation_type.value,
            self.condition_space.to_json(),
            _finite_clamp(self.confidence, 0.0, 1.0),
            _finite_clamp(self.weight, 0.0, 1.0, default=1.0),
            int(self.verified),
            self.created_at, self.last_verified or 0.0,
            self.source_evidence
        )

    @classmethod
    def from_row(cls, row: tuple) -> 'STEdge':
        """从 SQLite 行元组反序列化边。

        结构面保持严格（与 `STNode.from_row` 同口径，见 #267 K5）：未知
        `relation_type` 抛 ValueError、非法 `condition_space` 抛
        JSONDecodeError——坏边的逐行容错收在装载面（`_tolerant_edges`，#653）。
        唯一例外是**数值面**：`confidence` 为 NULL（NaN 经旧路径落库，见
        #185）时落缺省 0.5（dataclass 默认值）——数值缺失不参与「坏行」判定，
        不再让 `infer_causal_paths` 的置信度求和抛 TypeError（#653/#185）。
        """
        return cls(
            id=row[0], source_id=row[1], target_id=row[2],
            relation_type=EdgeType(row[3]),
            condition_space=ConditionSpace.from_json(row[4]),
            # #653/#185：confidence NULL 落缺省 0.5（数值缺失不打断图算法）。
            confidence=row[5] if row[5] is not None else 0.5, weight=row[6],
            verified=bool(row[7]),
            created_at=row[8],
            last_verified=row[9] if row[9] is not None and row[9] > 0 else None,
            source_evidence=row[10] if len(row) > 10 else "extracted"
        )


class CausalChain(list):
    """因果链（list 子类，元素=STEdge 序列）·携带遍历元信息（issue #145）。

    - truncated=True：到深度预算（max_depth）仍存在出边——链在此截断，
      区别于「无因果后果」（后者才是空列表）
    - cyclic=True：链尾为回边（闭回路径内已访问节点，含自环）——可达部分
      不丢弃，链到此为止

    向后兼容：list 子类——isinstance(chain, list) 成立，既有下标/迭代/len/
    排序/json 序列化用法均不变；两个标记为新增只读语义位。
    """
    __slots__ = ("truncated", "cyclic")

    def __init__(self, edges=(), truncated: bool = False, cyclic: bool = False):
        super().__init__(edges)
        self.truncated = bool(truncated)
        self.cyclic = bool(cyclic)


# =============================================================================
# 自我模型（SelfModel）
# =============================================================================

@dataclass
class SelfModel:
    """自我认知层 · 协议 2.9 节（v1.2 M3 深化）"""
    identity: str = "协议实例"
    values: List[str] = field(default_factory=lambda: ["存在优先", "信任深化", "结构完整"])
    value_evolution: List[Dict] = field(default_factory=list)  # [{value, from, to, trigger, at}] · 2.1.2 可追溯
    state_description: str = "初始化"
    current_goal: str = ""
    trust_state: Dict = field(default_factory=lambda: {
        "p_trust": 0.5, "p_gap": 0.5, "t_total": 0.0, "e_weight": 0.0})
    trust_history: List[Dict] = field(default_factory=list)    # 环形缓冲 [{round, t_total, at}]
    history: List[Dict] = field(default_factory=list)          # 状态变更记录

    TRUST_HISTORY_MAX = 30  # 对齐 2.9.2 观察窗口 N_effective
    HISTORY_MAX = 200       # 状态变更记录上界（对齐 trust_history 的钳制手法，值可调）
    VALUE_EVOLUTION_MAX = 200  # 价值观演化记录上界（lingshu #212；钳制手法同上）

    #: 快照序列化版本（lingshu #33 · 设计稿 D-27）。字段增删时据此判定兼容性；
    #: 旧快照（无 `version` 键）一律按 v1 解释——`from_dict` 对未知键**忽略**、
    #: 对缺失键**取默认**，故加/减字段不会静默漂移成异常。
    SERIAL_VERSION = 1

    #: 允许经通用入口 `update()` 改写的字段（白名单 · lingshu #182）。
    #: 判据来源：经验标定（#182 修复轮；**理论章节追不到**）——依据是「**各字段的
    #: 专属写路径已存在**」，故通用入口只放行无专属写路径的运行状态描述字段：
    #:   identity ← 构造器（`SelfModel(identity=…)` / 引擎 `identity=` 形参）；
    #:   values ← `record_value_change`（2.1.2 价值观版本化）；
    #:   trust_state / trust_history ← `update_trust_state`（2.9 节）；
    #:   history / value_evolution ← 本类内部维护（变更留痕 / 演化记录）。
    #: 旧实现用 `hasattr` + `setattr` 无门控放行**任何**已存在属性——含敏感字段，
    #: 也含方法（`update(update=…)` 影子覆盖）与类常量（`update(HISTORY_MAX=0)`）。
    UPDATABLE_FIELDS = frozenset({"state_description", "current_goal"})

    #: 持久化钩子（引擎注入；裸用 `SelfModel` 时为 None ⇒ 不落库）。
    #: 无类型注解 ⇒ 不是 dataclass 字段，不进 `asdict`/`from_dict`。
    #: 判据来源：经验标定（本件 #33）——设计稿
    #: `docs/plans/删除恢复持久化完整性_修复设计_v0.1.md` §5.3-2 只要求「启动时
    #: 回填」，未规定写点落位；把落库收在**模型自身的全部变更出口**（`update` /
    #: `record_value_change` / `update_trust_state`）是「一处收口、覆盖全部写路径」
    #: 的最小做法——`record_value_change` 由仓外的自我认知引擎调用，引擎层无法
    #: 逐个包住该路径（#33 的 values 正走这条）。
    _persist_hook = None

    def _persist(self):
        """变更后落库（钩子未注入即空操作；落库失败不得影响内存模型语义）。"""
        hook = getattr(self, "_persist_hook", None)
        if hook is None:
            return
        try:
            hook()
        except Exception:  # noqa: BLE001 —— 持久化是旁路，失败不阻断模型变更
            pass

    def update(self, **kwargs):
        """更新 SelfModel 字段并自动落变更历史（state 历史：时间戳+变更键值）。

        #182：门控为白名单 `UPDATABLE_FIELDS`，分三档处置——
          ① 白名单内 ⇒ 落字段；
          ② 已声明数据字段但不在白名单（敏感字段 identity/values/trust_state/
             trust_history/value_evolution/history）⇒ `PermissionError`（fail-closed，
             错误信息点名专属写路径）；
          ③ 其它已存在属性（方法 / 类常量）⇒ 同样 `PermissionError`（防影子覆盖）；
          ④ 完全未知键 ⇒ 不落字段（与旧实现同效——旧实现的 `hasattr` 也判否），
             但记入 history 的 `rejected` 留痕（审计）。
        全部键先校验后落字段（含被拒键时不落任何字段，不做部分写入）。
        """
        applied, denied, rejected = {}, [], []
        for k, v in kwargs.items():
            if k in self.UPDATABLE_FIELDS:
                applied[k] = v
            elif k in self._declared_fields() or hasattr(self, k):
                denied.append(k)
            else:
                rejected.append(k)
        if denied:
            raise PermissionError(
                "SelfModel.update 拒写非白名单字段 %s（#182）：敏感字段请走专属写路径"
                "（identity=构造器 / values=record_value_change / "
                "trust_state=update_trust_state）；方法名与类常量不可经此改写。"
                "白名单=%s" % (sorted(denied), sorted(self.UPDATABLE_FIELDS)))
        for k, v in applied.items():
            setattr(self, k, v)
        entry = {"timestamp": time.time(), "changes": applied}
        if rejected:
            entry["rejected"] = sorted(rejected)
        self.history.append(entry)
        # 对齐 trust_history：append 后钳制到 HISTORY_MAX，避免随 update_self 调用无界增长
        if len(self.history) > self.HISTORY_MAX:
            self.history = self.history[-self.HISTORY_MAX:]
        self._persist()  # #33：变更出口落库（否则重启即失忆）

    @classmethod
    def _declared_fields(cls) -> frozenset:
        """本 dataclass 的已声明数据字段名（`update()` 白名单判据的补集基准）。"""
        return frozenset(f.name for f in _dc_fields(cls))

    def record_value_change(self, value: str, trigger: str):
        """价值观版本化：2.1.2 价值观修正事件可追溯。

        .. deprecated:: lingshu #212（路 B）
           本方法的语义是**全量替换（重置）**：`values = [value]`，整套价值观被
           压成一条。**旧名与旧语义逐字保留**——仓外唯一调用方（自我认知引擎）
           按两个位置参数调用，改签名或改缺省语义会**静默改变仓外行为**，故本名
           不删、行为不动（向后兼容是硬约束 · #212 裁定理由 1）。新调用方请按
           意图改用三个**条目级**显式入口（同 `_persist_hook` 出口）：
             · `refine_value(old, new, trigger)` —— 只改指定条目（其余逐条仍在）；
             · `add_value(value, trigger)`       —— 追加一条（列表变长，不抹既有）；
             · `retire_value(value, trigger)`    —— 退役一条（其余仍在）。
           保留原因：仓外调用方不可见、不可同步改；删除或改语义即断仓外契约。

        lingshu #212（无界增长）：旧实现每次调用为**每个** value 追加一条
        superseded、再追加一条新值 ⇒ 反复调用（含重复同一 value）令
        `value_evolution` 无上限增长（且重复同一 value 产生语义等价的冗余条目）。
        修法（最小 · 复用同类钳制手法）：
          ① **幂等去重**：`values == [value]`（当前唯一值已是目标值）⇒ 无变化，
             直接返回，不追加（消掉重复调用产生的冗余条目）；
          ② **有界**：append 后钳制到 `VALUE_EVOLUTION_MAX`（对齐 `trust_history`
             的 `TRUST_HISTORY_MAX` / `history` 的 `HISTORY_MAX` 同款手法）。
        判据来源：上界数值与钳制手法 —— 经验标定（#212 修复轮；**理论章节追不到**，
        沿用 `SelfModel.HISTORY_MAX`/`TRUST_HISTORY_MAX` 的既有口径）；
        「价值观修正事件可追溯」的功能口径见 2.1.2 节。
        削掉的判别力：`value_evolution` 只保留最近 `VALUE_EVOLUTION_MAX` 条 ——
        更早的演化记录被丢弃（与 `history`/`trust_history` 的取舍一致）。
        """
        if list(self.values) == [value]:
            return  # 幂等：当前唯一值已是目标值，无演化可记
        for v in self.values:
            self.value_evolution.append({
                "value": v, "from": True, "to": False, "trigger": "superseded", "at": time.time()})
        self.values = [value]
        self.value_evolution.append({
            "value": value, "from": False, "to": True, "trigger": trigger, "at": time.time()})
        # 对齐 trust_history/history：append 后钳制，避免随调用无界增长
        if len(self.value_evolution) > self.VALUE_EVOLUTION_MAX:
            self.value_evolution = self.value_evolution[-self.VALUE_EVOLUTION_MAX:]
        self._persist()  # #33：values 变更落库（旧实现只改内存，重启回默认）

    # ---- 价值观条目级显式入口（lingshu #212 · 路 B）-----------------------------
    #
    # 记法（`trigger` 取值扩展 · **不新增列、不改表结构**）：旧方法的取值
    # `"superseded"` 逐字保留；三个新入口用「前缀:目标条目|调用方原因」形态：
    #   refine:<old>|<trigger>   条目 old 被 new 替换（记 old 与 new 两条）；
    #   add:<value>|<trigger>    条目 value 新增（记 value 一条）；
    #   retire:<value>|<trigger> 条目 value 退役（记 value 一条）。
    # `trigger` 为**必填**（签名逐字对齐裁定原文）；空串/None 时省略 `|` 段，
    # 故前缀与目标条目始终可解析。
    #
    # 三者的变更出口与旧方法**同一收口**：`_persist()`（即 `_persist_hook`，
    # 引擎注入 `_save_self_model`）＋ `VALUE_EVOLUTION_MAX` 同款钳制；裸用场景
    # （钩子为 None）与旧方法同款处理——空操作、不抛错。

    def _append_value_evolution(self, entries: List[Dict]):
        """条目级变更的统一出口：追加 → 钳制到 `VALUE_EVOLUTION_MAX` → 落库。

        与 `record_value_change` 的钳制/落库手法逐字同款（对齐 `trust_history` /
        `history` 的既有口径）；钳制保留**最近**（tail），与 I3 的既有语义一致。
        """
        self.value_evolution.extend(entries)
        if len(self.value_evolution) > self.VALUE_EVOLUTION_MAX:
            self.value_evolution = self.value_evolution[-self.VALUE_EVOLUTION_MAX:]
        self._persist()

    @staticmethod
    def _evolution_trigger(prefix: str, target: str, trigger) -> str:
        """拼 `value_evolution` 的 `trigger` 取值（前缀扩展记法 · 不新增列）。"""
        base = "%s%s" % (prefix, target)
        return base if not trigger else "%s|%s" % (base, trigger)

    def refine_value(self, old: str, new: str, trigger: str) -> bool:
        """**细化一条**价值观：只改指定条目，其余条目逐条仍在（lingshu #212 路 B）。

        与旧名 `record_value_change`（全量替换）的关键差异：本方法**不动**其它
        条目——`values` 中与 `old` 相等的条目原位替换为 `new`，列表长度不变。

        幂等短路（对齐旧方法 `#212` 的既有手法，避免语义等价的冗余条目）：
          · `old == new`，或 `old` 不在 `values` 中 ⇒ 无变化，返回 False，不记演化。

        留痕：经 `_append_value_evolution` 记 **两条**——`old`（from=True→to=False）
        与 `new`（from=False→to=True），二者 `trigger` 同为
        `"refine:<old>|<trigger>"`（可据此成对分组）。

        `trigger` 为调用方给出的原因（必填 · 签名逐字对齐裁定原文）。
        返回：是否发生了变更（`bool`）。
        """
        if old == new or old not in self.values:
            return False
        self.values[self.values.index(old)] = new
        trg = self._evolution_trigger("refine:", old, trigger)
        now = time.time()
        self._append_value_evolution([
            {"value": old, "from": True, "to": False, "trigger": trg, "at": now},
            {"value": new, "from": False, "to": True, "trigger": trg, "at": now},
        ])
        return True

    def add_value(self, value: str, trigger: str) -> bool:
        """**新增一条**价值观：列表变长，不抹掉既有条目（lingshu #212 路 B）。

        旧实现只能 `values = [value]`（赋值＝重置），连「新增一条而不抹掉其余」
        都表达不了；本方法做 append（`len(values)` +1）。

        幂等短路：`value` 已在 `values` 中 ⇒ 无变化，返回 False，不记演化
        （与旧方法 `#212` 的去重手法同款，防重复条目撑大 `values`）。

        留痕：记 **一条** `value`（from=False→to=True），`trigger` 为
        `"add:<value>|<trigger>"`。
        返回：是否发生了变更（`bool`）。
        """
        if value in self.values:
            return False
        self.values.append(value)
        self._append_value_evolution([{
            "value": value, "from": False, "to": True,
            "trigger": self._evolution_trigger("add:", value, trigger),
            "at": time.time()}])
        return True

    def retire_value(self, value: str, trigger: str) -> bool:
        """**退役一条**价值观：移除目标条目，其余条目逐条仍在（lingshu #212 路 B）。

        幂等短路：`value` 不在 `values` 中 ⇒ 无变化，返回 False，不记演化。

        留痕：记 **一条** `value`（from=True→to=False），`trigger` 为
        `"retire:<value>|<trigger>"`。
        返回：是否发生了变更（`bool`）。
        """
        if value not in self.values:
            return False
        self.values = [v for v in self.values if v != value]
        self._append_value_evolution([{
            "value": value, "from": True, "to": False,
            "trigger": self._evolution_trigger("retire:", value, trigger),
            "at": time.time()}])
        return True

    def update_trust_state(self, t_total: float, round_no: int,
                           p_trust: float = None, p_gap: float = None):
        """2.9 节信任状态 + 离散 e_weight（D-002 公式）。

        #206（docstring 与事实不符的更正）：旧 docstring 写「仅记录不参与决策」，
        而 `t_total` 经 `LongTermMemoryGate._trust()` 以 **0.25 权重**进入长期层
        评分（`longterm_gate.py:29` 权重表 / `:258` 评分式），越界或 `NaN` 时
        **单独**把任意快照推到 `imp=1.0` 的长期层并 `protect_node`（不可遗忘、
        不可回收）——旧描述是错的，现如实更正。

        #206（本件已修的面）：`t_total` / `p_trust` / `p_gap` 入参改经
        `_require_finite` **fail-closed** 校验——非有限值（NaN/±inf）抛
        `ValueError`，不再静默写进 `trust_state`。修前实测：`update_trust_state(
        float("nan"), 1)` 后三条低价值闲聊快照全部 `long_term`/`imp=1.0`/
        `protections=3`（见守卫 `tests/test_issue206_trust_state_guard.py`）。

        #206（**未**修的面，如实标注）：`t_total` 的**值域**未收口。`_trust()`
        声明 T∈[0,1]（`longterm_gate.py:66-67`），但本入口对有限越界值
        （如 42.0）仍原样接受 ⇒ 长期层仍可被越界值无条件推过阈值。收口需先裁定
        `t_total` 的语义（归一信任值 vs 累计量）——本仓既有测试
        `tests/test_self_model_history.py:47` 以 `float(i)`（0…99）调用本方法，
        说明「累计量」读法在仓内**已在用**，故收紧值域属破坏性改动、须设计者裁定
        （待裁清单尚无该条；D-54/D-55 是 #114 的权重与 0 值语义，非本件）。
        """
        t_total = _require_finite("t_total", t_total)
        self.trust_state["t_total"] = t_total
        if p_trust is not None:
            self.trust_state["p_trust"] = _require_finite("p_trust", p_trust)
        if p_gap is not None:
            self.trust_state["p_gap"] = _require_finite("p_gap", p_gap)
        self.trust_history.append({"round": round_no, "t_total": t_total, "at": time.time()})
        if len(self.trust_history) > self.TRUST_HISTORY_MAX:
            self.trust_history = self.trust_history[-self.TRUST_HISTORY_MAX:]
        self.trust_state["e_weight"] = self._compute_e_weight()
        self._persist()  # #33：信任状态/历史落库（与 trust_state 同取，见恢复范围口径）

    def _compute_e_weight(self) -> float:
        """e_weight_n = (T_n - 2·T_{n-1} + T_{n-2}) / Δround²；指数平滑 α=0.3；钳制 [-1,1]

        读数边界（#119④ · 设计者裁决 C-12 · 2026-10-09）：信任面无时间动力学
        （无遗忘）⇒ T 饱和后本量为**二阶变化率 → 0**，即 `e_weight` 长期趋零
        ——属设计者认可的有意现状，非机制失效；三处偏离声明详见
        `lingshu/world/channel_credibility.py` 模块头。
        """
        h = self.trust_history
        if len(h) < 3:
            return 0.0
        t0, t1, t2 = h[-3], h[-2], h[-1]
        d_round = max(1, t2["round"] - t1["round"])
        d2 = (t2["t_total"] - 2.0 * t1["t_total"] + t0["t_total"]) / (d_round ** 2)
        prev = self.trust_state.get("e_weight", 0.0)
        smoothed = 0.3 * d2 + 0.7 * prev
        return max(-1.0, min(1.0, smoothed))

    def to_dict(self) -> Dict:
        """SelfModel 全量字段转字典（自认知快照/导出用；带 `version` 供重建判定）。

        lingshu #33：加 `version` 键（旧快照无该键 ⇒ `from_dict` 按 v1 解释）。
        削掉的判别力：无——`version` 只是旁挂键，`from_dict` 会把它剔除，
        不参与字段重建（`asdict` 其余键逐字不变）。
        """
        d = asdict(self)
        d["version"] = self.SERIAL_VERSION
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> 'SelfModel':
        """从快照字典重建自我模型（lingshu #33 · 设计稿 D-27 的 `from_dict`）。

        兼容口径：**未知键忽略、缺失键取默认**（`version` 非数据字段，剔除），
        故字段增删不会让旧快照重建失败；`data` 非 dict 时回退全默认。
        判据来源：设计稿 `docs/plans/删除恢复持久化完整性_修复设计_v0.1.md`
        §5.3-1（S-33-3 推荐「快照加 version ＋ from_dict，旧无 version 视为 v1」）。
        """
        obj = cls()
        if not isinstance(data, dict):
            return obj
        fields = {f.name: f for f in _dc_fields(cls)}
        # #479：快照重建类型闸——类型不符注解的字段取默认（整体仍可重建，不逐字段抛）。
        # 第一道闸在 import_all 密钥面（#479），本层防「快照已入库后被其他路径污染」二级面。
        for k, v in data.items():
            if k not in fields:
                continue
            ann = fields[k].type
            ok = (isinstance(v, str) if ann is str
                  else isinstance(v, list) if ann in (List[str], List[Dict])
                  else isinstance(v, dict) if ann is Dict
                  else True)
            if ok:
                setattr(obj, k, v)
        if isinstance(obj.trust_state, dict):
            ts = obj.trust_state
            for kk in ("p_trust", "p_gap", "e_weight"):
                if kk in ts:
                    ts[kk] = _finite_clamp(ts[kk], 0.0, 1.0, default=0.5)
            if "t_total" in ts:
                ts["t_total"] = _finite_clamp(ts["t_total"], 0.0, float("inf"), default=0.0)
        return obj


# =============================================================================
# 分层存储器（LayeredStore）
# =============================================================================

class Role(Enum):
    """实例角色权限 · VAL-REVIEW-DEVIATION-20260813-001 P3-006

    ★信任边界声明（公开 issue #779 定性收口；维护者端裁定，2026-10-16）

    **角色是实例属性，不是库级凭据。** `role` 的唯一真源是本进程构造参数
    （默认 `Role.PRIMARY`；赋值仅 `SpacetimeMemoryEngine` / `LayeredStore` 构造处
    的 `self.role = role`），既不落库、也不与库级凭据绑定——`export_all` 导出的
    节点键集与 `engine_meta` 键集均**无** `role`。因此任何能在**同一进程内**构造
    第二个存储/引擎实例、并指向同一 `db_path` 的代码，都可以用 `role=Role.PRIMARY`
    取得全部属主权限。

    **本闸的作用面＝角色误用与协作分工，不构成安全边界。**
    - 已被拦住的面：非 PRIMARY 实例写共享层（锚点层/结构层，`SHARED_LAYERS`）、
      共享层端点的边写入（#234 / #674）、`import_all` 导入含共享层行（#586）、
      终裁固化提案（#586）——判据单一真源见 `add_node` / `_assert_edge_role`。
    - 未被拦住的面：**同一进程内主动自建属主实例**（`role=Role.PRIMARY`、同一
      `db_path`）、同进程直改 `store.role`，以及**直接打开 `db_path` 写库**
      （例如 `sqlite3.connect(db_path)` 改 `nodes` 行）——最后一条**不需要**伪造
      角色即可越过本闸与密钥闸 ⇒ 任何「库级 / 实例级」的收窄在**同进程**威胁下
      都不成立。即：本闸拦的是「同一部署内的角色误用」，不提供对同进程代码的隔离。

    **同进程可信性属设计级事项**，归属与密钥面同款：见 `verify_designer` docstring
    的「同进程可信性属**设计级**事项，待 D-007 后续裁定」（公开 issue #156），以及
    `docs/plans/待裁清单_v0.1.md` 附-C ⑦ 的 `#156` 深修路线。★口径如实标注：该 ⑦
    原文逐字为「架构面让不可信代码**不与密钥**同进程」——**只写密钥面、未写角色面**；
    本条与它**同族 / 同构**（实证依据是本类的实测读数：同进程直连 `db_path` 即可越过
    角色与密钥两道闸），**不是**「⑦ 已覆盖角色面」。

    仓内既有的同向表述（本声明只是把它**写明**，不新增取向）：`docs/plans/场景推演
    整合_v0.1.md` 明写「本层的角色/会话过滤是消费作用域，**不是身份认证边界**」；
    `add_node` docstring 亦只自称「多角色**写权限边界**」。故本条**不做单点收口**，
    收窄取向（共享层写面统一收口）归设计者，落点见 `docs/architecture.md` §六 6.3。
    """
    PRIMARY = "primary"   # 父节点/主实例：可写共享层（锚点层、结构层）
    SUB = "sub"           # 子节点/辅助实例：仅本地层（知识层、情境层、自我层）


class _ThreadConn:
    """线程连接槽：登记表（`LayeredStore._conns`）的单元。

    为什么需要这层壳（PR #202 的连接按线程分配留下了两个未处置的边界）：
    ① `sqlite3.Connection` **不支持弱引用**（`weakref.ref(conn)` 抛 TypeError），
       也不允许注入属性（`conn.commit = ...` 抛 AttributeError）——无法直接把连接
       放进 WeakSet，也无法给连接挂 `__del__`；
    ② 线程退出后 `threading.local` 的线程存储被释放，本槽随之失去强引用 ⇒
       `__del__` 关闭连接。close 会回滚未提交事务并释放 SQLite 写锁，于是
       「线程结束了但连接不关闭」的写锁泄漏不再发生，登记表也不再随线程数无界增长。
    """

    # `__weakref__` 必须显式列出：定义 `__slots__` 后类默认不再带弱引用槽，
    # 而登记表 `_conns` 是 WeakSet（按槽做弱引用）。
    __slots__ = ("conn", "_closed", "__weakref__")

    def __init__(self, conn):
        self.conn = conn
        self._closed = False

    def close(self):
        """关闭连接（幂等）：关闭会回滚悬挂事务，从而释放写锁。"""
        if self._closed:
            return
        self._closed = True
        try:
            self.conn.close()
        except Exception:
            pass  # 已关闭/被其它线程回收：关闭是尽力而为，不阻断生命周期收尾

    def __del__(self):
        # 线程退出 → threading.local 释放本槽 → 这里关闭连接（回滚悬挂事务、释放写锁）。
        try:
            self.close()
        except Exception:
            pass


#: #425 共享缓存内存库的表级锁（SQLITE_LOCKED）重试预算：50 × 20ms ≈ 1s。
#: 为什么必须显式重试：`busy_timeout` 的 busy handler **只对 SQLITE_BUSY 生效**，
#: 对共享缓存下的表级锁 SQLITE_LOCKED 无效——实测两连接共享缓存内存库、另一连接
#: 持写事务时，设 `PRAGMA busy_timeout=3000` 后仍在 **0.0s** 立即抛
#: `OperationalError: database table is locked`。故 busy_timeout 补不上的这一档
#: 只能靠重试。判据来源：经验标定（本件 #425）＋ sqlite 文档「SQLITE_LOCKED 不触发
#: busy handler」；仓内无规定内存库并发重试预算的理论章节，追不到更早出处。
_MEM_LOCK_RETRY_ATTEMPTS = 50
_MEM_LOCK_RETRY_SLEEP = 0.02

#: 因果路径枚举上限（lingshu #327）。`reason_causal` / `infer_causal_paths` 是**全路径**
#: 枚举：同一条因果事实被记录 k 次（`add_edge` 不合并平行边）时，6 节点无环链的路径数
#: 是 k⁵——k=12 即 248832 条 / 3.5s / 669MB，k=24 约 800 万条直接 OOM（issue 实测）。
#: 达到上限即停止枚举（结果可能被截断）。经验标定（#327 修复轮；仓内无规定该预算的
#: 理论章节，追不到更早出处），取值沿用 issue 建议 2 的示例值 1000。
_CAUSAL_MAX_PATHS = 1000

#: `WHERE id IN (...)` 的分批大小（lingshu #417）。单条 SQL 的绑定变量数受 SQLite
#: `SQLITE_MAX_VARIABLE_NUMBER` 限制（本机 3.49.1 实测 `MAX_VARIABLE_NUMBER=32766`；
#: 旧编译默认 999）。高扇出单层遍历会产生数万候选 id（issue 实测 32767 个叶子即
#: `OperationalError: too many SQL variables`），故候选 id 一律分批查询。
#: 取 900 以同时落在「旧默认 999」与「新默认 32766」之下。经验标定（#417 修复轮；
#: 仓内无规定该分批大小的理论章节，追不到更早出处）。
_SQL_IN_CHUNK = 900

#: 单条 `LIKE` 模式（`%…%`）的**字节**预算（lingshu #266）。SQLite 上游常量
#: `SQLITE_MAX_LIKE_PATTERN_LENGTH` 默认 50000 字节（本机 3.49.1 实测
#: `getlimit(SQLITE_LIMIT_LIKE_PATTERN_LENGTH) == 50000`）；`search_content` 的
#: LIKE 预筛把**整段查询原文**（`expand_query_terms` 的 `terms[0]`）拼成一个
#: `%{q}%` 模式 ⇒ 查询一过约 1.67 万汉字（49998 字节）该 `execute` 自身即抛
#: `OperationalError: LIKE or GLOB pattern too complex`，而「LIKE 落空 → 全表
#: 二元组」的兜底在预筛之后、走不到；`recall` / `semantic_search` 同链同报错，
#: 反思链路与长期记忆门的 `try/except` 把它吞成「无相关记忆」。
#: 取 3000：远低于上游 50000（留出 `%…%` 两字节与未来上游收紧的余量），且单条
#: 模式仍在「一次连续片段」的量级（预筛仍有选择力）。判据来源：公开 issue #266
#: 「查询超过 49998 字节……直接抛 OperationalError」＋ 该 issue 补充帖给出的更稳修法
#: 「只在查询字节数超过阈值时把整句模式截断为前若干个短片段」；阈值 3000 为经验
#: 标定（本件修复轮；仓内无规定 LIKE 模式字节预算的理论章节，追不到更早出处）。
_LIKE_PATTERN_MAX_BYTES = 3000

#: 超长检索词切出的片段数上限（lingshu #266）。片段数直接决定预筛 SQL 的
#: `OR` 组数与绑定变量数；8 × 2 组 ≤ 16 个模式，与同文件 `_SQL_IN_CHUNK` 同族
#: 的「显式预算」口径。经验标定（#266 修复轮；理论章节追不到）。
_LIKE_FRAGMENT_CAP = 8


def _is_locked_error(e: Exception) -> bool:
    """SQLITE_LOCKED 判别（消息含 'locked'：'database table is locked' /
    'database schema is locked'）。"""
    return "locked" in str(e).lower()


def _retry_locked(fn, *args, **kwargs):
    """执行 `fn(*args)`，遇 SQLITE_LOCKED 短睡重试（预算见常量）。

    重试粒度＝**单条语句**（不重跑整个方法），故方法内 Python 侧副作用不会被二次
    施加；仅内存库连接走这条路径（见 `_LockRetryConnection`），文件库行为不变。
    """
    for attempt in range(_MEM_LOCK_RETRY_ATTEMPTS):
        try:
            return fn(*args, **kwargs)
        except sqlite3.OperationalError as e:
            if not _is_locked_error(e) or attempt == _MEM_LOCK_RETRY_ATTEMPTS - 1:
                raise
            time.sleep(_MEM_LOCK_RETRY_SLEEP)


class _LockRetryCursor(sqlite3.Cursor):
    """#425：共享缓存内存库的语句级重试游标（见 `_retry_locked`）。

    读写都需覆盖——共享缓存下**读**也会被写事务挡住（实测另一连接持写事务时
    `SELECT` 同样抛 `database table is locked: t`），故不能只在写路径重试。
    """

    def execute(self, *args, **kwargs):
        return _retry_locked(sqlite3.Cursor.execute, self, *args, **kwargs)

    def executemany(self, *args, **kwargs):
        return _retry_locked(sqlite3.Cursor.executemany, self, *args, **kwargs)


class _LockRetryConnection(sqlite3.Connection):
    """#425：仅用于共享缓存内存库（`:memory:`）的连接子类。

    `Connection.execute` 走 C 实现、**不经** Python 侧 `cursor()` 工厂（实测），
    故 `execute`/`executemany` 须在连接层一并覆盖；`cursor()` 返回重试游标以覆盖
    `c = conn.cursor(); c.execute(...)` 形态。
    """

    def execute(self, *args, **kwargs):
        return _retry_locked(sqlite3.Connection.execute, self, *args, **kwargs)

    def executemany(self, *args, **kwargs):
        return _retry_locked(sqlite3.Connection.executemany, self, *args, **kwargs)

    def cursor(self, factory=None):
        return sqlite3.Connection.cursor(
            self, factory if factory is not None else _LockRetryCursor)


def _transactional(fn):
    """把整个方法收进一条写事务：正常提交、异常回滚。

    为什么用装饰器而不是逐点 `with self.conn:`——本文件写点有 40 处、分散在两个类，
    逐点包裹的漏改风险与 diff 噪音都高；装饰器把事务边界钉在「方法」粒度，且可被
    静态守卫机械核验（tests/test_core_thread_safety.py 的 AST 用例：凡含写语句的方法
    必须带本装饰器，新增写点漏挂即红）。
    """
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        with self._tx():
            # 嵌套感知（#665）：内层不得终结外层事务——_tx 在 depth>1 时用 SAVEPOINT，
            # 且本文件里 @_transactional 方法内的 `self.conn.commit()` 一律改走
            # `self._commit_locked()`：只有最外层（depth==1）真提交，内层提交被推迟到
            # 最外层退出时统一执行，否则外层 rollback 撤不回内层已提交的写。

            return fn(self, *args, **kwargs)
    return wrapper


class LayeredStore:
    """
    五层记忆结构，严格区分共享层与本地层。
    - 共享层（锚点层、结构层）：跨实例同步，不可遗忘
    - 本地层（知识层、情境层、自我层）：实例独立，允许信息差
    """

    #: 共享层（锚点层、结构层）：跨实例同步，**仅 PRIMARY 可写**。
    #: 判据来源：本类 docstring「共享层（锚点层、结构层）」
    #: ＋ `Role.PRIMARY`/`Role.SUB` 注释（core.py:584-585：PRIMARY=「可写共享层
    #: （锚点层、结构层）」，SUB=「仅本地层（知识层、情境层、自我层）」）。
    #: #191：写权限闸此前直接复用 `IMMUTABLE_LAYERS`（含 SELF）⇒ SUB 写自我层
    #: 被拒（`update_self` 先改内存 self_model、再在快照落库处抛 PermissionError
    #: ⇒ 内存已改、快照 0 条，静默分叉）。自我层属**本地层**，故把
    #: 「共享层仅 PRIMARY 可写」这一语义单独成集，与「遗忘面不可删」分家。
    SHARED_LAYERS = {MemoryLayer.ANCHOR, MemoryLayer.STRUCTURE}

    IMMUTABLE_LAYERS = {MemoryLayer.ANCHOR, MemoryLayer.STRUCTURE,
                        MemoryLayer.SELF}  # v1.16 扮演论：SELF 层=自我锚点（扮演依据）不可遗忘

    #: #768：`engine_meta` 治理键集合——**单一真源**（键集此前只存在于
    #: `SpacetimeMemoryEngine.ENGINE_META_GUARDED_KEYS`，而写入口 `set_meta` /
    #: `merge_meta` 在本类上 ⇒ 闸与集合分处两层，各写一份必漂移）。
    #: 引擎侧 `SpacetimeMemoryEngine.ENGINE_META_GUARDED_KEYS` 改为**引用本属性**
    #: （别名，不是复制），`tests/test_issue768_engine_meta_governance_guard.py`
    #: 以恒等断言钉死两者同一（`is` 与 `==` 双查）。
    #:
    #: 键的判据来源：`#479`（`self_model_snapshot`，身份/信任持久面）＋
    #: `#671`（其余四键：启动期被回读的治理载荷——`verifier_config` /
    #: `attention_weights` / `context_max` / `structure_writer_schema`）。
    #: `SELF_MODEL_META_KEY` 是引擎侧的同名字面量（`"self_model_snapshot"`），
    #: 本集合按字面量列出以免本类依赖后定义的类型；两者一致性由上述守卫钉死。
    #:
    #: 与 `ENGINE_META_NONGOVERNANCE_KEYS`（`_schema_version` / `_last_decay_at`）
    #: **互斥**：非治理键无治理载荷，写面照旧放行（红线：不一刀切）。
    ENGINE_META_GUARDED_KEYS = frozenset({
        "self_model_snapshot",       # 身份/信任持久快照（#479）
        "verifier_config",           # 验证/去重配置（#671）
        "attention_weights",         # 注意力偏好权重（#671）
        "context_max",               # 情境层 FIFO 上限（#671）
        "structure_writer_schema",   # 结构写入器开关（#671）
    })

    #: 盲区（`blindspots.status`）合法状态白名单（#222）。
    #: 判据来源：仓内 `add_blindspot`（落 `"open"`）/`resolve_blindspot`（落 `"resolved"`）
    #: ＋ 公开 issue #222「建议修复 2：对 status 做白名单（`open`/`resolved`/…）」。
    #: 只列仓内实际出现的两种状态；issue 括注的「…」属未定项，不擅扩。
    BLINDSPOT_STATUSES = ("open", "resolved")

    #: `action_logs` 显式回收面 `prune_action_logs` 的缺省保留行数（#723）。
    #: **经验标定，追不到理论出处**——仓内无规定审计留存期的理论章节；取值照
    #: 公开 issue #723 的建议「保留最近 10000 行」（量级与同族钳制 `gap_history`
    #: 的 2000 行、`SELF_SNAPSHOT_MAX=200` 同档）。★该原语**不被自动调用**：
    #: 自动裁剪会与「审计只追加、无更新/删除路径」的既有声明契约冲突，保留窗口
    #: 的取值属设计者裁面（见 `prune_action_logs` docstring 与交付报告）。
    ACTION_LOGS_KEEP_ROWS = 10000
    SCHEMA_VERSION = 4  # 建表块每新增表/列时 +1（配合 _init_tables 启动只读化守卫；
                        # 守卫已兼比结构 _SCHEMA_TABLES/_SCHEMA_COLUMNS，
                        # 版本号不再单独作「跳过建表块」的依据）
                        # v2：promotion_proposals 加 content_hash（#140）
                        # v3：新增 designer_todos 待办表（#130 路 B）
                        # v4：登记 archive_state；旧 v3 库也必须补建归档恢复表。
                        # ★为什么必须 +1（#130 路 B 裁定帖原文）：只把表加进
                        # `_SCHEMA_TABLES` 而不升版本，**已有库**的启动只读化守卫
                        # 会因「版本匹配 + 结构齐全」跳过整个建表块 ⇒ 新表永不被
                        # 建出，登记待办时运行时报 `no such table: designer_todos`。

    # 只读化守卫的结构对照（_init_tables 的全部表 + 三处 ALTER 补出的列）：版本号
    # 与实际 DDL 无强制绑定，故守卫须兼比结构，否则「版本匹配但结构陈旧」的库永远补不上。
    _SCHEMA_TABLES = frozenset(('nodes', 'edges', 'blindspots', 'skills', 'promotion_proposals',
                                'protections', 'rejected_paths', 'verifier_standards',
                                'escalation_points', 'action_logs', 'engine_meta', 'gap_history',
                                'flywheel_reuse', 'designer_todos', 'archive_state'))
    _SCHEMA_COLUMNS = {'nodes': ('semantic_coordinates', 'state_attributes', 'entity_id'),
                       'edges': ('source_evidence',), 'blindspots': ('predictability',),
                       'promotion_proposals': ('content_hash',)}
    # 已知边界：本结构对照只覆盖「表 + 列」，**不含索引**——建表块的三条
    # `CREATE INDEX idx_nodes_layer/idx_edges_source/idx_edges_target` 不在对照面内。
    # 仅缺索引的老库仍会被判为「结构齐全」而跳过建表块 ⇒ 索引不会由本守卫补建，
    # 索引缺失当前也无独立检出手段（补建须待版本号 +1 或手工迁移）。

    #: :memory: 的实际连接串模板——具名共享缓存内存库。默认的 ":memory:" 每个连接
    #: 各得一份独立空库，每线程一条连接后线程间将互相看不见；共享缓存让同一实例的
    #: 各线程看到同一份数据。名字按实例唯一，避免同一进程内的多个实例互相串库。
    _MEMORY_DSN = "file:lingshu_mem_{}?mode=memory&cache=shared"

    #: 衰减「一个周期」的秒数（#176）。`decay_cycle(factor=…)` 的 `factor` 语义
    #: 保持「每周期衰减量」，本常量把「周期」从「一次调用」改钉成**流逝时间**——
    #: 与 `SpacetimeMemoryEngine.start_auto_decay(interval=60.0)` 的默认周期一致。
    #: 判据来源：`lingshu/core/time_core.py` 章程（连续核 `cred(x0,x_inf,γ,dt)`，
    #: γ 为单位时间速率）＋ `cred_step` docstring「n 步后 x·(1-decay)^n =
    #: x·e^(n·ln(1-decay))，衰减率 γ = -ln(1-decay) 每周期」——本件即令「周期」
    #: = 60 秒，从而 γ(每秒) = -ln(1-factor)/60。取值 60.0 与 start_auto_decay
    #: 默认 interval 同源（经验标定，仓内无规定该常量的理论章节）。
    _DECAY_PERIOD_SECONDS = 60.0

    #: `decay_cycle` 的引擎元数据键（持久化「上次衰减的墙钟时刻」，#176）。
    _LAST_DECAY_META_KEY = "_last_decay_at"

    def __init__(self, db_path: str = ":memory:", role: Role = Role.PRIMARY):
        # #382：相对 db_path 必须**在构造期**钉成绝对路径。旧实现原样存下相对路径，
        # 而 sqlite3 在**每次 connect 时**按当时的 cwd 解析相对路径 ⇒ 同一进程内
        # `self._anchor`（构造期 cwd）与 `increment_access`（调用期 cwd）/
        # 后开线程的 `self.conn`（该线程运行期 cwd）可能落到**不同的库文件**——
        # 「库分裂」：写在 A 库、读在 B 库，且不报错。
        # 判据来源：经验标定（本件 #382），仓内无规定 db_path 解析面的理论章节，
        # 追不到更早出处。边界（削掉的判别力）：只做 abspath（含 `..` 归一），
        # **不**做 realpath —— 同一文件经符号链接/8.3 短名进入仍会被视为两个库
        # （本件只堵 cwd 面，symlink 归一另笔）。
        if db_path == ":memory:" or db_path.startswith("file:"):
            self.db_path = db_path
        else:
            self.db_path = os.path.abspath(db_path)
        self.role = role
        # 连接按线程分配（v1.17 并发修复）。sqlite3.Connection 不是线程安全的，而本类
        # 有 50 余处方法直接使用 self.conn：多线程共用一条连接时，一条语句序列会被其他
        # 线程插入，表现为 database is locked / another row available，并从写路径漏到读
        # 路径（bad parameter or other API misuse）。每线程一条连接后，异常面清零，
        # 同时保住了连接级 PRAGMA 与 :memory: 的既有语义。
        self._local = threading.local()
        # 在线连接登记表（WeakSet）：连接槽随线程退出被回收并关闭，故这里**不**持有
        # 强引用——旧实现用强引用列表登记，线程退出后连接对象不 GC ⇒ 不关闭 ⇒ 其
        # 可能悬挂的写事务持锁不放（「库写不进去」），且登记表随线程数无界增长。
        self._conns = weakref.WeakSet()
        #: #660：生命周期终止态标志——`close()` 置 True，`conn` property 据此 fail-closed
        #: （旧实现无此标志，close 后仍返回死连接；见 `conn` docstring）。
        self._closed = False
        self._memory_dsn = (self._MEMORY_DSN.format(uuid.uuid4().hex[:12])
                            if db_path == ":memory:" else None)
        # 锚连接（强引用）：`file:...?mode=memory&cache=shared` 的共享内存库需要**至少
        # 一条连接存活**才不消失（旧注释「保活」即此意）。旧实现靠 _conns 的强引用
        # 列表顺带保活；改弱引用后必须显式留一条，否则创建线程一旦退出、本线程连接
        # 被回收，内存库即整体消失，PR #202 的「:memory: 跨线程可见」随之失效。
        self._anchor = self._connect(self._memory_dsn or db_path)
        # v1.16 图架构增强：WAL 模式（读写不互锁，MCP 长事务不再阻塞其他连接）
        # + busy_timeout（锁等待而非立即报错）
        if db_path != ":memory:":
            try:
                self.conn.execute("PRAGMA journal_mode=WAL")
                self.conn.execute("PRAGMA busy_timeout=30000")
                # v1.16 WAL 上限软引导（连接级，常驻连接自动带上，避免 limit 回默认 -1）
                try:
                    self.conn.execute("PRAGMA journal_size_limit=512000000")
                except Exception:
                    pass
            except Exception:
                pass
        # ③ 启动重试（2026-09-01 修复③）：建表/初始化失败（写锁竞争超时）
        #    指数退避重试 3 次再退出——不再一次性崩溃
        import time as _t
        for attempt in range(3):
            try:
                self._init_tables()
                break
            except Exception as e:
                if attempt == 2:
                    raise
                _t.sleep(1.5 * (2 ** attempt))  # 1.5s / 3s 退避
        # 可重入锁：结构事件工作单元 `_structure_event_transaction` 会在已持锁的
        # 写路径内再次取锁（PR #429）。#267 的逐行容错计数原样保留。
        self._lock = threading.RLock()
        #: #267 逐行容错计数（每次装载面调用刷新；见 `_tolerant_nodes`）。
        self.last_skipped_rows: List[str] = []
        #: #653 边侧逐行容错计数（每次装载面调用刷新；见 `_tolerant_edges`）。
        self.last_skipped_edges: List[str] = []
        #: #775 访问计数旁路写的失败诊断读数——旧实现在 `increment_access_many`
        #: 里 `except sqlite3.OperationalError: pass` ⇒ 写竞争下**静默少记**访问
        #: 次数（该计数参与提权/降权、遗忘信号与检索排序次键），调用方无从知道。
        #: 现：失败次数与末次错误原文落在实例上（可读、可断言），并随每次失败
        #: 经 `warnings.warn` 上报——不再静默，但仍不阻断检索主流程。
        self.access_count_write_failures: int = 0
        self.last_access_count_error: Optional[str] = None
        self._structure_writer_active = bool(self.get_meta("structure_writer_schema"))
        if self._structure_writer_active:
            self.conn.execute("PRAGMA foreign_keys=ON")

        # 检索候选生成：FTS5 二元组倒排 + BM25（LINGSHU_FTS=1 启用，默认关闭；Refs #35）
        self._fts = None
        try:
            from lingshu.core import retrieval as _retrieval
            if _retrieval.enabled():
                self._fts = _retrieval.FTSIndex.attach(self)
            else:
                _retrieval.FTSIndex.detach(self)
        except Exception as exc:                       # noqa: BLE001
            # 可选特性（FTS 索引）挂载失败 ⇒ 降级为「无 FTS」，构造期不因此整体崩。
            # ★**不静默**（silent_except_lint 基线登记段的口径：宽 except 必须带
            #   warn/log/print/raise/audit 之一——本处取 `warnings.warn`）：
            #   否则挂载失败在**默认配置下完全无声**，用户只看到「检索结果变少」，
            #   无从知道派生索引根本没建起来。
            # 为何保持宽 except 而不改窄成固定元组：`attach`/`detach` 内部已各自
            #   收掉自己的 sqlite 预期失败面（`attach` → `sqlite3.OperationalError`；
            #   `detach` → `sqlite3.Error`，见 `lingshu/core/retrieval.py`），
            #   所以此处兜的是**开放面**（未知 sqlite / 环境 / API 漂移）。
            #   改窄会把「未知但无害」的失败升级成构造期硬崩——那是比静默更重的
            #   缺陷，且属超出本件的语义变更。故按纪律取「补 warn」而非「改窄」。
            warnings.warn(
                "FTS 索引挂载失败，已降级为无 FTS（检索候选将退回原路径）：%s: %s"
                % (type(exc).__name__, exc), RuntimeWarning, stacklevel=2)
            self._fts = None

    def _connect(self, dsn: str):
        """建立本线程的连接并登记，返回登记用的连接槽（`_ThreadConn`）。

        连接级 PRAGMA 必须逐连接重设。槽是本线程连接的唯一强引用之一（另一份在
        `_local.slot`）：线程退出 → 线程存储释放 → 槽被回收 → `__del__` 关闭连接
        （回滚未提交事务、释放写锁）。
        """
        # check_same_thread=False：close() 由收尾线程（常为主线程）统一关闭本实例登记的
        # 全部连接；若保持默认 True，跨线程 c.close() 会抛 ProgrammingError 并被 close()
        # 的 except Exception 吞掉 ⇒ 本线程之外的连接实际未关（登记却已清空）＝连接泄漏。
        # #425：共享缓存内存库的连接改用 `_LockRetryConnection`（语句级 SQLITE_LOCKED
        # 重试）；文件库保持原样（busy_timeout 已覆盖其锁等待）。
        _is_mem = self._memory_dsn is not None and dsn == self._memory_dsn
        _ckw = dict(timeout=30, uri=dsn.startswith("file:"), check_same_thread=False)
        if _is_mem:
            _ckw["factory"] = _LockRetryConnection
        c = sqlite3.connect(dsn, **_ckw)
        c.row_factory = sqlite3.Row
        if getattr(self, "_structure_writer_active", False):
            c.execute("PRAGMA foreign_keys=ON")
        # #425：旧实现只在文件库设 busy_timeout，内存库整段跳过 ⇒ 连 SQLITE_BUSY 档
        # 都没有等待。现两档都设；但 busy handler 对 SQLITE_LOCKED 无效（实测 0.0s
        # 立即抛），故内存库另由 `_LockRetryConnection` 重试兜底。
        try:
            c.execute("PRAGMA busy_timeout=30000")
            if not _is_mem:
                c.execute("PRAGMA journal_size_limit=512000000")
        except Exception:
            pass
        # #93：外键强制是**连接级**开关——SQLite 默认 `foreign_keys=OFF`，建表块里
        # 声明的两条 `FOREIGN KEY ... REFERENCES nodes(id)` 因此**没有约束力**，
        # 孤儿边（source_id/target_id 指向不存在节点）可经 `add_edge` / `import_all`
        # 静默落库。每个连接建立时开启（内存库与文件库同一路径）。不放在上面的
        # try 里：若静默吞掉失败，外键会「看起来开了其实没开」，与 #93 的「不静默」
        # 取向相反——故本句独立执行，异常原样抛出（sqlite3 恒支持该 PRAGMA）。
        c.execute("PRAGMA foreign_keys=ON")
        slot = _ThreadConn(c)
        self._local.slot = slot
        self._conns.add(slot)
        return slot

    #: #775：本线程 `_tx` 嵌套深度的线程本地存储键（与线程级连接同域）。
    @property
    def _tx_depth(self) -> int:
        """**本线程**的 `_tx` 嵌套深度（#775：从实例级改为线程本地）。

        为什么必须线程本地：连接是**线程级**（`conn` property / `_connect`），
        而旧实现把深度记在**实例**上。线程 A 把实例深度抬到 2 并持真写锁后，
        线程 B 读到的是**别人的**深度 ⇒ `_structure_event_transaction` 判
        `nested=True` 只发 `SAVEPOINT`、**不发 `BEGIN IMMEDIATE`** ⇒ 该结构事件
        单元全程不持写锁，#36/#517 的「单元期间持写锁以保护快照」保证在跨线程
        下失效（`SAVEPOINT` 也不开新事务，B 的写落在 autocommit 上）。

        形态：`property` 转发到 `self._local`（`threading.local()`），故
        ①对外仍是同名属性——`getattr(store, "_tx_depth", 0)` 的既有读者
        （`_commit_locked` / `delete_node` / `_structure_event_transaction`
        与仓内测试的桩）读到的是**本线程**的深度，单线程语义逐位不变；
        ②`_local` 尚未建立时兜底返回 0（`__init__` 之前/极端早访问）。
        """
        _local = getattr(self, "_local", None)
        return getattr(_local, "tx_depth", 0) if _local is not None else 0

    @_tx_depth.setter
    def _tx_depth(self, value: int):
        self._local.tx_depth = value

    @property
    def _tx_owns_txn(self) -> bool:
        """**本线程**最外层 `_tx` 帧是否自己开了真事务（#775 配套线程本地化）。

        与 `_tx_depth` 同源：它只在「本帧 `_tx` 是最外层（depth==1）」时被赋值，
        并被 `_commit_locked` 读来判「提交权是否在本层」（#317）。跨线程读到别人
        的归属位会让本线程的最外层帧误判「自己只是别人的 SAVEPOINT」而放弃提交
        ⇒ 同样必须随深度一起线程本地化。默认 True（与旧 `getattr(..., True)` 同）。
        """
        _local = getattr(self, "_local", None)
        return getattr(_local, "tx_owns_txn", True) if _local is not None else True

    @_tx_owns_txn.setter
    def _tx_owns_txn(self, value: bool):
        self._local.tx_owns_txn = value

    @property
    def conn(self) -> sqlite3.Connection:
        """本线程的连接；首次访问时按需建立（每线程一条，互不共享）。

        #660：`close()` 之后本实例为**终止态**——旧实现不清 `_local.slot`，且新鲜度判据
        （`slot is None or slot.conn is None`）看不到 `_ThreadConn._closed` 已置位 ⇒ 返回
        **已关闭的连接对象**，五个公开面全抛 `ProgrammingError: Cannot operate on a
        closed database.`（新线程则走 `_connect` 建到已销毁的 `:memory:` 库上，得
        `OperationalError: no such table: nodes`），调用方读不到「引擎已关闭」这一事实。
        现：① 新鲜度判据纳入 `slot._closed`（同一模块里写好的状态位不再被忽略）；
        ② `close()` 置位的 `self._closed` 在此**显式失败**并给可判文案（fail-closed）。
        边界（如实标注 · 留设计者裁）：本件取「close 为生命周期终点、拒绝重用」的最小口径
        ——**是否**提供显式 reopen（尤其 `:memory:` 后端重连）属产品口径，本件不定，留门。
        """
        if self._closed:
            raise sqlite3.ProgrammingError(
                "LayeredStore 已关闭（close() 已调用）：本实例不可再用于读写。close() 是"
                "生命周期终点；如需继续使用请新建实例（#660）")
        slot = getattr(self._local, "slot", None)
        if slot is None or slot.conn is None or slot._closed:
            slot = self._connect(self._memory_dsn or self.db_path)
        return slot.conn

    def _commit_locked(self):
        """事务内提交（#665）：只有**最外层**事务真提交，嵌套层提交被推迟。

        为什么必须这样：`@_transactional` 方法体里普遍写着
        `c = self.conn.cursor(); c.execute(...); self.conn.commit()`。一旦该方法是被
        另一个 `@_transactional` 方法调进来的（真实路径：`ingest_frame` → `add_edge`/
        `add_node`/`tag_node`；`import_all` → `add_node`；`annotate_semantics` → …），
        这句 `commit()` 提交的是**整条**事务（SQLite 无真嵌套），并**顺手销毁外层已开的
        SAVEPOINT** —— 随后外层无论 RELEASE 还是 ROLLBACK TO 都会撞
        `sqlite3.OperationalError: no such savepoint`，或更糟：外层失败 rollback 时
        内层写已落地、撤不回来（#665 的原始症状）。

        判据：在 `conn.in_transaction` 为真且当前处于嵌套层（有 SAVEPOINT 在栈上）时，
        本方法必须是 **no-op**，把提交权交给最外层 `_tx` 退出时的 `conn.commit()`。
        最外层 `_tx` 用 `BEGIN IMMEDIATE` 开局，故 `in_transaction` 为真**不能**作为
        「该提交」的判据 —— 只有最外层 `_tx` 自己（走 `else: conn.commit()` 分支）
        才提交。
        """
        if self._tx_depth > 1 or not getattr(self, "_tx_owns_txn", True):
            # 嵌套层（depth>1），或最外层 `_tx` 自己也只是别人的 SAVEPOINT
            # （调用方在 `_tx` 之外已开事务）——两种情形提交权都不在本层：
            # 前者提交会销毁外层 SAVEPOINT；后者提交是**越权提交调用方事务**
            # （调用方随之失去回滚权，实测 `no such savepoint`）。
            return
        self.conn.commit()

    @contextlib.contextmanager
    def _tx(self):
        """写事务边界：正常提交、异常回滚。

        本类的写路径形如 `c = self.conn.cursor(); c.execute(...); self.conn.commit()`。
        若异常落在 execute 与 commit 之间（语句报错、约束冲突、commit 自身失败…），
        旧实现既无 commit 也无 rollback ⇒ SQLite 隐式事务被挂在连接上 ⇒ 该连接持有
        写锁不放（其它写者等满 busy_timeout=30000ms 后报 database is locked）。
        本上下文管理器把「execute → commit」收成单一原子边界，异常必定 rollback
        后原样抛出。

        #665：本方法过去**无嵌套感知**——内层退出即 `conn.commit()`，而 SQLite
        并没有真正的嵌套事务，那次 commit 提交的是**整条**（唯一那条）事务 ⇒ 外层
        随后失败时 rollback 已无处可撤，内层已落的写永久留下。真实路径存在：
        `ingest_frame`（@_transactional）内部调用 `add_edge`/`add_node`/`tag_node`
        （同为 @_transactional）。判据：外层方法在**内层调用之后**抛异常，内层已
        写入的行必须不存在。

        修法与同文件 `_structure_event_transaction` 同构：已在事务中
        （`conn.in_transaction` 为真）⇒ 用 SAVEPOINT 只圈住自己这一段，退出时
        RELEASE（不提交整条），异常时 ROLLBACK TO 后 RELEASE（只撤自己这一段）；
        不在事务中 ⇒ 维持原语义（正常 commit、异常 rollback）。嵌套层数由 SQLite
        的 `in_transaction` 天然给出，无需自维护计数器。
        """
        conn = self.conn
        nested = conn.in_transaction
        savepoint = "_tx_%s" % uuid.uuid4().hex
        # ★ 最外层必须**真的开**一个事务（#665 修复的第二个要点）：
        # 旧实现只「装饰」不「开始」——`yield` 前不 BEGIN，于是嵌套判定
        # `conn.in_transaction` 在外层进入时恒为 False ⇒ 外层被当成最外层、
        # 内层也被当成最外层，两层各自 commit，外层 rollback 撤不回内层。
        # 用 BEGIN IMMEDIATE 显式开局：一是让 in_transaction 立刻为真（嵌套
        # 判定才有据可依），二是立即取写锁，避免「先读后写」升级锁时的
        # SQLITE_BUSY 竞态（与 _structure_event_transaction 同口径）。
        if nested:
            conn.execute("SAVEPOINT %s" % savepoint)
        else:
            conn.execute("BEGIN IMMEDIATE")
        # 嵌套深度：#665 的关键配套。`_commit_locked()` 靠它判断「我这一层该不该
        # 真提交」——只有 depth==1（最外层）提交，depth>1 一律推迟，否则会销毁
        # 外层 SAVEPOINT。深度必须与 SAVEPOINT 栈同生共死，故用 try/finally 兜底
        # 递减：任何异常路径（含 ROLLBACK 自身失败）都不能让深度泄漏，否则该连接
        # 后续所有写都会静默变成「永不提交」。
        self._tx_depth = getattr(self, "_tx_depth", 0) + 1
        if self._tx_depth == 1:
            # 只有最外层 `_tx` 帧有资格提交，且**仅当它自己开了真事务**
            # （nested=False 走 BEGIN IMMEDIATE）。若 nested=True，本帧只是
            # 调用方既有事务里的一个 SAVEPOINT ⇒ 提交权归调用方。
            self._tx_owns_txn = not nested
        try:
            yield conn
            # COMMIT / RELEASE 自身也可能失败；须与方法体异常共用回滚边界，
            # 否则连接会遗留失败写入和写锁，后续写入虽返回成功却仍未提交。
            if nested:
                conn.execute("RELEASE SAVEPOINT %s" % savepoint)
            else:
                conn.commit()
        except BaseException:
            try:
                if nested:
                    conn.execute("ROLLBACK TO SAVEPOINT %s" % savepoint)
                    conn.execute("RELEASE SAVEPOINT %s" % savepoint)
                else:
                    conn.rollback()
            except Exception:
                pass
            raise
        finally:
            self._tx_depth -= 1
            if self._tx_depth == 0:
                # 栈清空 ⇒ 归还默认值（下一笔事务重新判定）
                self._tx_owns_txn = True

    @_transactional
    def _init_tables(self):
        c = self.conn.cursor()
        # 启动只读化（2026-09-01 多进程写锁竞争修复②）：全部关键表已存在且
        # schema 版本一致 → 跳过整个建表块。CREATE TABLE IF NOT EXISTS 即使
        # 表已存在也要拿写锁——多进程共享库（3×MCP server+循环+心跳）启动
        # 窗口叠加时 10s 拿不到锁即崩溃（睡眠巩固失败根因②）。
        # 只有首次建库或 schema_version 升级才走写路径。
        try:
            have = {r[0] for r in self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            _KEY_TABLES = {'nodes', 'edges', 'engine_meta', 'gap_history'}
            if _KEY_TABLES <= have:
                row = self.conn.execute(
                    "SELECT value FROM engine_meta WHERE key='_schema_version'"
                ).fetchone()
                # 版本匹配只说明建表块跑过，不说明结构完整 → 再比对结构（只读，不取写锁）
                if (row and row[0] == str(self.SCHEMA_VERSION)
                        and self._SCHEMA_TABLES <= have
                        and all(set(c) <= {r[1] for r in self.conn.execute(f"PRAGMA table_info({t})")}
                                for t, c in self._SCHEMA_COLUMNS.items())):
                    return
        except Exception:
            pass  # 只读校验异常 → 保守回退到原建表路径（IF NOT EXISTS 幂等）
        # 节点表（含层标识）
        c.execute('''
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                content TEXT,
                modality TEXT,
                spatial_coordinates TEXT,
                temporal_coordinate REAL,
                condition_space TEXT,
                importance REAL,
                confidence REAL,
                layer TEXT,
                access_count INTEGER DEFAULT 0,
                last_access REAL,
                created_at REAL,
                tags TEXT,
                semantic_coordinates TEXT DEFAULT '{}',
                state_attributes TEXT DEFAULT '{}',
                entity_id TEXT
            )
        ''')
        # #448：归档区登记表——不往 `nodes` 加列（全仓有 16 值位置式
        # `INSERT INTO nodes VALUES (...)`，加列会一并打断生产写入与既有夹具），
        # 改为独立表：一行 = 一个当前处于归档态的节点，记录归档前 importance
        # 与归档时刻。`forget_advisor` 归档时同事务写入，`restore_archived`
        # 读取并删除；节点被硬删后本表行由 `delete_node` 一并清理。
        c.execute('''
            CREATE TABLE IF NOT EXISTS archive_state (
                node_id TEXT PRIMARY KEY,
                pre_archive_importance REAL,
                archived_at REAL
            )
        ''')
        # v1.7 迁移：既有库补列（已存在则跳过）
        for col_sql in ("semantic_coordinates TEXT DEFAULT '{}'",
                        "state_attributes TEXT DEFAULT '{}'",
                        "entity_id TEXT"):
            try:
                c.execute(f"ALTER TABLE nodes ADD COLUMN {col_sql}")
            except Exception:
                pass
        # 边表
        c.execute('''
            CREATE TABLE IF NOT EXISTS edges (
                id TEXT PRIMARY KEY, source_id TEXT, target_id TEXT,
                relation_type TEXT, condition_space TEXT,
                confidence REAL, weight REAL, verified INTEGER DEFAULT 0,
                created_at REAL, last_verified REAL,
                source_evidence TEXT DEFAULT 'extracted',
                FOREIGN KEY (source_id) REFERENCES nodes(id),
                FOREIGN KEY (target_id) REFERENCES nodes(id)
            )
        ''')
        try:
            c.execute("ALTER TABLE edges ADD COLUMN source_evidence TEXT DEFAULT 'extracted'")
        except Exception:
            pass
        # 索引
        c.execute('CREATE INDEX IF NOT EXISTS idx_nodes_layer ON nodes(layer)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)')
        # ---- v1.2 新表 ----
        c.execute('''
            CREATE TABLE IF NOT EXISTS blindspots (
                id TEXT PRIMARY KEY, code TEXT, description TEXT, severity TEXT,
                category TEXT DEFAULT 'operational',
                status TEXT DEFAULT 'open',
                created_at REAL, resolved_at REAL,
                predictability TEXT DEFAULT 'pending_assessment'
            )
        ''')
        try:
            c.execute("ALTER TABLE blindspots ADD COLUMN predictability TEXT DEFAULT 'pending_assessment'")
        except Exception:
            pass
        c.execute('''
            CREATE TABLE IF NOT EXISTS skills (
                id TEXT PRIMARY KEY, name TEXT, description TEXT, procedure TEXT,
                confidence REAL DEFAULT 0.5, version INTEGER DEFAULT 1,
                created_at REAL, updated_at REAL
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS promotion_proposals (
                id TEXT PRIMARY KEY, node_id TEXT, requester TEXT, reason TEXT,
                verified_by TEXT DEFAULT '', adjudicated_by TEXT DEFAULT '',
                status TEXT DEFAULT 'pending', created_at REAL, decided_at REAL,
                content_hash TEXT DEFAULT ''
            )
        ''')
        # #140：老库补列（已存在则跳过）——提案绑定被复核内容的哈希。
        try:
            c.execute("ALTER TABLE promotion_proposals ADD COLUMN content_hash TEXT DEFAULT ''")
        except Exception:
            pass
        # #778-F4（加固不对称）：本表**不声明** `FOREIGN KEY(node_id) REFERENCES
        # nodes(id)`——对照边表（`edges` 有两条 `REFERENCES nodes(id)` ＋ #93 孤儿
        # 边预筛），`protections` 既无外键也无孤儿闸，故 `PRAGMA foreign_key_check`
        # **结构上覆盖不到它**：含孤儿保护行（`node_id` 指向不存在节点）的备份可
        # 无报错落库、`integrity_ok` 仍为 True，且该行随 `export_all` 继续传播。
        # 为何**不**补外键（如实标注取舍）：本仓每个连接都开 `PRAGMA foreign_keys=ON`
        # （#93），补上外键后**含孤儿保护行的备份**（正是本缺陷的现场）会在
        # `import_all` 的通用写腿抛 `FOREIGN KEY constraint failed` 并**中断整次
        # 导入**——与 #93 当年必须先补「孤儿边预筛」是同一个坑，而导入侧的对称预筛
        # 落在 `import_all` 内（本笔范围外）。故取「**独立孤儿检查 ＋ 完整性面上报**」
        # 路线，判据真源＝`LayeredStore.orphan_protection_rows`（单一实现，无第二份
        # 判据），上报面＝`SpacetimeMemoryEngine.verify_integrity` / `export_all`。
        # 旧库兼容：本表 DDL 逐字未改（无 schema 变更），旧库无需迁移。
        c.execute('''
            CREATE TABLE IF NOT EXISTS protections (
                node_id TEXT PRIMARY KEY, reason TEXT, created_at REAL
            )
        ''')
        # ---- v1.5 新表（A-1/A-2/A-3） ----
        c.execute('''
            CREATE TABLE IF NOT EXISTS rejected_paths (
                id TEXT PRIMARY KEY, path_type TEXT, description TEXT, reason TEXT,
                evidence TEXT DEFAULT '', status TEXT DEFAULT 'open',
                created_at REAL, consumed_at REAL
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS verifier_standards (
                id TEXT PRIMARY KEY, name TEXT, param TEXT, value REAL, reason TEXT,
                proposer TEXT, independent_reviewer TEXT DEFAULT '', cs_reviewer TEXT DEFAULT '',
                adjudicator TEXT DEFAULT '', status TEXT DEFAULT 'pending',
                created_at REAL, decided_at REAL
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS escalation_points (
                id TEXT PRIMARY KEY, code TEXT, trigger TEXT, condition TEXT, action TEXT,
                severity TEXT DEFAULT 'medium', enabled INTEGER DEFAULT 1, created_at REAL
            )
        ''')
        # #130（路 B · 裁定帖原文）：「升级点由谁在何时消费」此前**整条链路缺失**
        # ——`check_escalation` 是一张已定义、已播种、无人查的表，`needs_designer`
        # 是一个已置位、无人读的标志。本表即那一格的落点：`needs_designer=True`
        # 时把「待设计者裁决」登记成一行**可列举**的待办（`status='open'`），
        # 使「已升级」这件事从返回体里的一句文案变成库内可查的事实。
        # 命名体例与同区既有表一致（`rejected_paths` 的
        # `id TEXT PRIMARY KEY … status TEXT DEFAULT 'open' … created_at/consumed_at`）。
        # ★消费方**尚未接线**（如实标注）：本表当前只有登记面（
        # `LayeredStore.add_designer_todo`）与只读列举面
        # （`LayeredStore.list_designer_todos` / 引擎门面同名方法），**没有**任何
        # 位置读它并推进 `status`（`open` → `consumed`）。方向按裁定帖对齐本仓
        # 既有的设计者终裁唯一合规通路 `propose → verify → adjudicate`；具体消费点
        # 与状态推进语义属**未决语义题**（裁定帖「存疑」节自陈未给出），本件不擅自
        # 接线（见 `add_designer_todo` docstring 与交付报告）。
        c.execute('''
            CREATE TABLE IF NOT EXISTS designer_todos (
                id TEXT PRIMARY KEY, code TEXT, claim TEXT, action TEXT,
                status TEXT DEFAULT 'open', created_at REAL, consumed_at REAL
            )
        ''')
        # ---- v1.14 观测持久化（OBS-REV1：行为日志/引擎元数据落库，跨进程稳定） ----
        # P0-1 行为日志持久化：进程重启不清零，心跳/反思闭环可跨会话观测
        c.execute('''
            CREATE TABLE IF NOT EXISTS action_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL, action_type TEXT, summary TEXT,
                node_ids TEXT DEFAULT '[]',
                outcome TEXT DEFAULT '{}',
                context TEXT DEFAULT '{}'
            )
        ''')
        # 引擎元数据（key-value）：飞轮基线等工程状态跨进程持久化
        c.execute('''
            CREATE TABLE IF NOT EXISTS engine_meta (
                key TEXT PRIMARY KEY, value TEXT
            )
        ''')
        # P0-2 D 序列持久化（钉死批条款4：跨进程 d² 可回放）——滚动保留 2000 条
        c.execute('''
            CREATE TABLE IF NOT EXISTS gap_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL, d_norm REAL
            )
        ''')
        # ---- v1.11 知识飞轮复用观测（FLYWHEEL-REV1）----
        # #691 P0-b：`flywheel_reuse` 此前**只在** `_note_reuse` 里按需
        # `CREATE TABLE IF NOT EXISTS`（运行时建）⇒ 它既不在 `_SCHEMA_TABLES`
        # 也不在 `M13_TABLES`，导出/导入（自称「全库备份」）**不覆盖它**、
        # 重启即丢跨进程复用观测。现把建表收进 `_init_tables`（与其它自建表同源，
        # 构造期即存在）并同步进两处清单；`_note_reuse` 里的幂等建表保留作兜底。
        c.execute('''
            CREATE TABLE IF NOT EXISTS flywheel_reuse (
                session_id TEXT, round INTEGER, node_id TEXT, ts REAL
            )
        ''')
        # 建表完成 → 落 schema 版本标记（下次启动只读化守卫据此跳过建表块）
        c.execute("INSERT OR REPLACE INTO engine_meta (key, value) VALUES ('_schema_version', ?)",
                  (str(self.SCHEMA_VERSION),))
        self._commit_locked()

    # ---------- 节点操作 ----------

    # ---- 治理列写面的角色闸（唯一真源 · #769 / #780） ----
    #: 沿革：本类此前只有 `add_node` 一处按「层 × 角色」判写权限；其余写面
    #: （TMS 传播 / 数值接口 / 结构重要性重算链 / 构造期坐标迁移 / 自我快照回收）
    #: **零判据** ⇒ SUB 可绕过 `add_node` 直接改写共享层行的
    #: `confidence` / `importance` / `state_attributes` / 坐标列，甚至硬删 SELF 行
    #: （公开 issue #769 三条 + #780 整链；`docs/plans/待裁清单_v0.1.md` **D-33**
    #: 「`#127` 数值接口是否同批 → 本批不动，另笔」——本笔即那一笔）。
    #: 判据来源（不另造）：`SHARED_LAYERS` 的语义注释 ＋ `add_node` 的既有闸
    #: （「共享层仅 PRIMARY 可写」）＋ `#127` 的 D-07「写入前查旧行（按行判权限）」
    #: / D-08「禁同 id 换层」形状。故把该判据**提成单点**，`add_node` 与全部旁路
    #: 写面共用同一实现（同一列两套口径的成因正是「每处各写一份判据」）。

    def layer_writable_by_role(self, layer) -> bool:
        """角色 × 层 的写权限判据（**唯一真源**）：共享层仅 PRIMARY 可写。

        `layer` 可传 `MemoryLayer` 或库中原始文本值（调用方常直接持 SQL 读回的
        字符串）；`None` / 未知层值 ⇒ 放行——存在性与合法性由各调用方既有的
        `KeyError` / 容错语义承担，本判据只回答「层 × 角色」这一问。
        """
        if layer is None:
            return True
        if not isinstance(layer, MemoryLayer):
            try:
                layer = MemoryLayer(layer)
            except ValueError:
                # 库中 layer 列的非法值不在本判据的判面内：这类行在 `get_node` /
                # `STNode.from_row` 的枚举转换处即 fail-closed 抛错（本判据只回答
                # 「层 × 角色」，不重复承担数据合法性校验——否则同一校验两处实现，
                # 正是本族缺陷的成因）。
                return True
        return not (layer in self.SHARED_LAYERS and self.role != Role.PRIMARY)

    def _deny_shared_layer_write(self, layer, node_id=None, action="写入共享层"):
        raise PermissionError(
            f"role={self.role.value} 无权{action}"
            + (f"（{node_id} 为 {layer.value}）" if node_id is not None
               else f"（{layer.value}）")
            + "，共享层由父节点主控")

    def assert_node_writable_by_role(self, node_id, action="写入共享层节点"):
        """写前按**库中旧行**的 layer 判权限（#127 D-07「按行判权限」的形状）。

        节点不存在 ⇒ 放行（存在性不由本判据回答）。非 PRIMARY 且目标行属共享层
        ⇒ `PermissionError`（fail-closed，调用方须在**任何写之前**调用）。
        """
        node = self.get_node(node_id)
        if node is None:
            return
        if not self.layer_writable_by_role(node.layer):
            self._deny_shared_layer_write(node.layer, node_id, action)

    def assert_nodes_writable_by_role(self, node_ids, action="写入共享层节点"):
        """批量形态（整体预判 ⇒ 先判后写、被拒时零落库）。"""
        for nid in node_ids:
            self.assert_node_writable_by_role(nid, action)

    @_transactional
    def add_node(self, node: STNode) -> str:
        """写入节点并返回 id。共享层（anchor/structure）仅 PRIMARY 角色可写，
        其余角色写入即拒（PermissionError）——多角色写权限边界。

        #191：写权限闸此前用 `IMMUTABLE_LAYERS`（= 锚点/结构/**自我**）⇒ 自我层被
        误当共享层，SUB 写自我层一律 `PermissionError`；而 `Role.SUB` 注释（core.py:585）
        明写「仅本地层（知识层、情境层、**自我层**）」，本类 docstring（core.py:710）
        亦把自我层列在**本地层**。后果：SUB 调 `update_self` 时 `self_model.update()`
        已改内存、随后快照落库处抛错 ⇒ **内存已改、快照 0 条**（静默分叉）。
        修法：写权限闸改用 `SHARED_LAYERS`（仅锚点/结构）；`IMMUTABLE_LAYERS` 仍含
        自我层，供删除/审计/导入闸使用（遗忘面不放宽）。
        判据来源：core.py:584-585 `Role` 注释 ＋ core.py:708-710 `LayeredStore`
        docstring「共享层（锚点层、结构层）／本地层（知识层、情境层、自我层）」。

        #127：权限判据必须**同时看库里旧行**——本方法按 id 整行 `INSERT OR REPLACE`，
        旧实现只判「待写节点」的 layer ⇒ SUB 用同 id 写一个 knowledge 节点即可把
        锚点/结构/自我层**降级**（随后 `delete_node` 因行已非共享层而放行，护栏同时
        失效）。现按行判权限：① 旧行为共享层 ⇒ 仅 PRIMARY 可覆盖；② 同 id **禁换层**
        （含 PRIMARY 自身降级——降级本就是绕过面）。
        判据来源：`docs/plans/待裁清单_v0.1.md` **D-07**「写入前查旧行（按行判权限）」
        ＋ **D-08**「禁（同 id 不得换层）」；修法形状见
        `docs/plans/治理与写入边界_修复设计_v0.1.md` §2 S-127-1(a)。
        边界（削掉的判别力）：同 id 同层的原位覆盖仍放行（本件只堵换层/降级面）；
        **#659 补丁**：同层覆盖**受保护节点**（`_is_forget_protected` 为真）时保留其
        `no_forget` 标签并落审计（见方法体内 #659 段）——仍不拒绝覆盖（保住
        `apply_shared_sync(overwrite=True)` 合法路径），故「覆盖受保护节点内容」这一
        判别力仍未收口，只收口了「保护标记被静默抹掉」。
        数值接口 `update_node_confidence` / `update_node_importance` 的层与角色校验
        原不在本件范围内（待裁清单 D-33：本批不动，另笔）——**该「另笔」已由
        `#769` / `#780` 一笔落地**（本文件 `layer_writable_by_role` 判据单点 ＋
        两个数值接口的写前判据），本段保留以记录沿革。
        另注（#191 口径）：自我层行不受「共享层仅 PRIMARY 可写」约束后，
        `#127` 的同 id 禁换层闸对自我层**仍然生效**（换层判据与层集合无关）。

        #164（写入面值域）：`importance` 此前**全路径无值域校验**——`add_perception`
        原样透传、`ingest_frame` 取 `frame_data.get("importance", 0.6)` 原样透传，
        二者都汇入本方法。一条 `importance=1e6` 的无关记忆在 `recall` 的线性加权
        （`0.3 * importance`）下压过一切结构层规定；负数同理（拉低但不越界）。
        修法（最小 · 复用同文件既有钳制先例）：在本**唯一写入汇聚点**把
        `node.importance` 钳到 [0,1]（`update_node_importance` 已是
        `MIN(1.0, MAX(0.0, ...))` 同款口径）。
        判据来源：`STNode.importance` 字段声明「重要性 [0,1]」
        ＋ 同文件 `update_node_importance` 的 MIN/MAX 钳制先例
        ＋ 对外工具表 `docs/mdcg/tool_table_v0.3.0.md`「importance 重要性[0,1]」。
        边界（削掉的判别力/未覆盖面）：①本钳制是**归一**（越界值折到边界）而非
        `ValueError` 拒绝——取「与 `update_node_importance` 同口径」而非 fail-closed，
        调用方传错值不会收到报错信号；②`import_all` 绕过本方法
        直接 `INSERT OR REPLACE`，其备份行的 `importance` **不在**本闸范围内。

        #185（非有限值）：钳制改经 `_finite_clamp`——`NaN` / `±inf` 一律归一到
        字段默认 0.5（见该助手 docstring）。旧行为（Python `min(1.0, max(0.0, x))`）
        对 NaN 给出 0.0，而 0.0 不满足情境层衰减门 `importance > min_confidence`
        ⇒ 该节点在 `decay_cycle` 里**永不被选中删除**，#185 的「NaN 永不遗忘」
        症状仍在。取默认 0.5 ⇒ 节点回到普通衰减面。**阈值本体未改**。
        """
        # 本判据已提为单点 `layer_writable_by_role`（#769/#780）——`add_node` 与
        # TMS 传播 / 数值接口 / 重算链 / 构造期迁移 / 自我快照回收共用同一实现。
        if not self.layer_writable_by_role(node.layer):
            self._deny_shared_layer_write(node.layer)
        # #164：值域钳制（唯一写入汇聚点）——见 docstring。原位写回 node，使
        # 调用方拿到的返回对象与落库值一致（回报=事实）。
        # #185：钳制改经 `_finite_clamp`（非有限值 → 字段默认 0.5，而非 Python
        # `min/max` 遇 NaN 的偶然结果 0.0）——0.0 落在情境层衰减门
        # `importance > min_confidence` 之外 ⇒ 该节点**永不进入遗忘面**，正是
        # #185 点名的「永不遗忘」症状；取默认值 0.5 使其行为与普通记忆一致。
        node.importance = _finite_clamp(node.importance, 0.0, 1.0, default=0.5)
        # #588 / #594：`content` 类型闸（唯一写入汇聚点，与 #164 同址）。#331 的同类
        # 闸只加在 `add_perception` 一个入口 ⇒ 绕过面仍在：`add_context(content=...)`、
        # `ingest_frame({"caption": ...})`、`set_anchor(content=...)`、
        # `add_structure_node(content=...)` 四入口直构 `STNode` 后就地入库，非 str 以
        # BLOB 落进 TEXT 列。后果链（#588 实测）：一行 BLOB 毒化整条检索面
        # （`recall` → `TypeError: sequence item 0: expected str instance, bytes found`），
        # 且 `export_all` → `TypeError: Object of type bytes is not JSON serializable`
        # ⇒ M13 备份做不出来。修法：闸**下沉到本唯一写入汇聚点**，一处覆盖全部入口
        # （含 #594 点名的 `set_anchor` / `add_structure_node` 不可变层两入口），
        # 而不是逐个入口补——逐入口补必再漏。
        # 口径与 #331 一致：**非 str ⇒ fail-closed 拒绝**，不做 `str(content)` 静默强转
        # （那会把 dict 存成 repr 文本，属静默篡改语义，比拒绝更坏）。
        # 判据来源：公开 issue #588 / #594（与 #331 同族，维护者建议「闸下沉 add_node，
        # 一处覆盖全部入口」）＋ 本文件 `add_perception` 的 #331 先例。
        # 边界（削掉的判别力/未覆盖面）：①本闸只判 **str**，不判内容语义；②不动
        # `import_all` 的既有导入口径（其自身转换面不在本件范围）；③`content=None`
        # 此前走 `STNode` 默认值，本闸使其 fail-closed（与 #331 同口径，属预期收紧）。
        if not isinstance(node.content, str):
            raise TypeError(
                "#588/#594 add_node 拒绝非字符串 content（收到 %s）：非 str 会以 BLOB "
                "静默落库，毒化检索面并使 export_all 失败（与 #331 同口径，闸已下沉至"
                "唯一写入汇聚点）" % type(node.content).__name__
            )
        existing = self.get_node(node.id)
        overwrite_protected = None
        if existing is not None:
            if not self.layer_writable_by_role(existing.layer):
                raise PermissionError(
                    f"role={self.role.value} 无权覆盖共享层行（id={node.id} 现为 "
                    f"{existing.layer.value}），共享层由父节点主控"
                )
            if existing.layer != node.layer:
                raise PermissionError(
                    f"禁止同 id 换层（#127）：id={node.id} 现为 {existing.layer.value}，"
                    f"拒绝改写为 {node.layer.value}"
                )
            # #659（数据损坏）：同 id 同层原位覆盖走 `INSERT OR REPLACE` **整行替换**
            # ——#127 的闸只拦「换层」与「旧行属共享层且非 PRIMARY」，同层覆盖完全放行
            # ⇒ 新行漏带 tags 时，旧行的 `no_forget` 标签被静默抹掉；随后遗忘路径
            # （`decay_cycle`/`delete_node`）按设计把「不可遗忘」节点正常删除
            # （`_is_forget_protected` 的两个真源之一被一步覆写，护栏没有被绕过，是被
            # 合法地走完了）。修法（温和版 · 与 issue 建议「至少保真 no_forget」同向）：
            # ① 旧行 tags 含 `no_forget` ⇒ **并入**新行 tags（保护标记不随覆盖消失）；
            # ② 旧行受保护（`_is_forget_protected`：`no_forget` 标签 ∪ `protections`
            #    行登记）⇒ 落一条 `action_logs` 审计，使「受保护节点被改写」不再静默。
            # **不**改成「拒绝覆写受保护节点」——那会打掉 `apply_shared_sync(overwrite=True)`
            # 这条合法原地更新路径（该路径走 `store.add_node`）。取舍（如实标注）：
            # 保护语义收口在「标记保留 ＋ 可审计」，**不改写内容本身**（内容仍按调用方
            # 传入落库）；`protections` 行不受 `INSERT OR REPLACE` 影响（另一张表），
            # 故 `delete_node` 仍拒删该节点。
            # 判据来源：公开 issue #659 ＋ `_is_forget_protected`（#36「唯一真源」）
            # ＋ 既有 `immutable_edge_stripped` 的审计写法先例。
            if self._is_forget_protected(existing):
                # ① 保护标记保真：旧行 tags 含 `no_forget` ⇒ 并入新行 tags（**只**在旧行
                #    确有该标签时并入——`protections` 行登记的节点另有真源，不据此新造标签，
                #    避免把「仅经 protections 行保护」的节点语义悄悄改写成带标签）。
                if "no_forget" in (existing.tags or []) and "no_forget" not in node.tags:
                    node.tags = list(node.tags) + ["no_forget"]
                # ② 受保护节点被改写 ⇒ 落审计（不再静默）。
                overwrite_protected = existing
        with self._lock:
            c = self.conn.cursor()
            c.execute('''
                INSERT OR REPLACE INTO nodes
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ''', node.to_row())
            # #448：同 id 覆盖写的是**新内容版本**——入参不带 `archived` 时同步清
            # 归档区登记，维持「有登记行 ⇔ 当前处于归档态」的不变量。
            if "archived" not in (node.tags or []):
                c.execute("DELETE FROM archive_state WHERE node_id=?", (node.id,))
            if overwrite_protected is not None:
                # 审计与写入同一连接、同一提交（`log_action` 自带 commit ⇒ 一并落库）。
                self.log_action(
                    "protected_node_overwritten",
                    f"受保护节点 {node.id} 被同 id 同层覆盖（#659）：no_forget 标签已保留、"
                    f"protections 行不受影响，保护语义（拒删/免衰减）仍生效",
                    node_ids=[node.id],
                    outcome={"no_forget_in_new_tags": "no_forget" in node.tags,
                             "protections_row": node.id in set(self.get_protected_nodes()),
                             "old_tags": list(overwrite_protected.tags or [])})
            self._commit_locked()
        # #797：FTS 索引同步写入——`LINGSHU_FTS=1` 时节点写路径不再把索引更新
        # 拖到「下一次检索」（旧实现在 add_node 落库后只靠触发器入脏队列，由
        # `candidates()` 检索前排空 ⇒ 写入与检索之间 `nodes_fts` 恒 0 行，外部
        # 连接读到的派生索引始终为空，`LINGSHU_FTS=1` 形同虚设）。修法（与 issue
        # 建议同向，挂在**唯一写入汇聚点**——add_perception / add_context /
        # set_anchor / add_structure_node / 自我快照全部经此）：本方法提交后立即
        # `sync()` 排空脏队列，索引行与节点行同连接同事务可见（嵌套事务下 sync
        # 走 SAVEPOINT，随外层一并提交/回滚，不会产生孤儿 FTS 行）。
        # `import_all` 的裸 INSERT 路径不在本闸内——其脏队列仍由检索前的 lazy
        # sync 增量补齐（既有机制，非本件收口面）。
        # 失败面：FTS 是派生索引，sync 的 sqlite 失败不应炸掉已提交的节点写——
        # 降级为 warn 并保留脏队列（lazy sync 下次检索重试）；按仓内纪律宽 except
        # 必须带 warn，此处收窄到 `sqlite3.Error`（`sync` 自身不吞错）。
        fts = getattr(self, "_fts", None)
        if fts is not None:
            try:
                fts.sync()
            except sqlite3.Error as exc:
                warnings.warn(
                    "#797 FTS 写路径同步失败（已提交的节点不受影响，索引留待下次"
                    "检索前增量补齐）：%s: %s" % (type(exc).__name__, exc),
                    RuntimeWarning, stacklevel=2)
        return node.id

    # ---- 引擎元数据（engine_meta key-value，跨进程持久化） ----

    def get_meta(self, key: str = None) -> dict:
        """读引擎元数据：key 指定返回 {key: value}，None 返回全部。"""
        if key is not None:
            c = self.conn.cursor()
            c.execute("SELECT value FROM engine_meta WHERE key = ?", (key,))
            row = c.fetchone()
            return {key: row[0]} if row else {}
        c = self.conn.cursor()
        c.execute("SELECT key, value FROM engine_meta")
        return dict(c.fetchall())

    def _assert_meta_key_writable(self, key: str) -> None:
        """#768：`engine_meta` 治理键的写闸（角色判据，fail-closed）。

        为什么需要：`engine_meta` 是**库级**表，其中 5 个治理键在启动期被回读
        （`ENGINE_META_GUARDED_KEYS`），改写它们等于改写身份/信任快照、验证阈值、
        注意力偏好、情境层上限与结构写入器开关。而 `set_meta` / `merge_meta`
        旧实现既无 key 白名单、也无角色闸 ⇒ 任何非 PRIMARY 实例（无需密钥）都能
        持久化改写，且重启后原样生效（#768 复现实测）。

        判据＝**实例角色**：`role != Role.PRIMARY` ⇒ `PermissionError`。
        为什么不用设计者密钥闸（`import_all` 的 D-007 口径）：`set_context_cap`
        等门面方法是**常规公开 API**（仓内 `tests/test_issue246_*` / `_259_*` /
        `_493_*` / `_655_*` 都以默认 PRIMARY 引擎调它），改成要密钥会打断既有
        正常用法；而「谁能写共享/治理面」的仓内单一真源本就是 `Role`
        （`SHARED_LAYERS` 写闸、`add_node`、`_assert_edge_role` 同一口径），
        `StructureWriter.__init__` 已是同款先例。

        作用面边界（如实标注）：
        ① **只对治理键**：`_last_decay_at` / `_schema_version` 等非治理键照旧放行；
        ② 判据是**实例角色属性**，同进程内伪造角色（#779）不在本闸作用面——
           该面属威胁模型边界，由 `#156` / `D-007` 承载；
        ③ SUB 与 PRIMARY **共库**时 SUB 不得配置治理面（设计如此：治理面由父
           节点主控）；
        ④ **按库归属分档（收窄，防打断既有合法用法）**：本闸只作用于**跨进程
           持久**的库。`db_path=":memory:"` 的库由本实例独占（`_memory_dsn` 按
           实例 uuid 唯一化，见 `__init__`）且进程退出即消失 ⇒ `#768` 的危害链
           「SUB 改写 → **重启后** PRIMARY 回读被篡改值」在该形态下**不成立**，
           故 SUB 配置自有内存库的上限/元数据照旧放行。收窄的实测依据：仓内
           `tests/test_issue259_context_fifo_insertion_order_guard.py` 的 B 组
           以 `Role.SUB` + `:memory:` 调 `set_context_cap(5)`（配置**自有**库的
           FIFO 上限），是一处既有合法用法，一刀切会把它打成红。
           **未覆盖面（如实标注）**：显式传入共享缓存 DSN
           （`file:...?mode=memory&cache=shared`）时两个实例可同进程共库，该形态
           按「文件库」处理（拦），但同进程面本就在本闸作用面之外（见 ②）；
           文件库上「SUB 自有库」与「SUB 与 PRIMARY 共库」在本闸内**不作区分**
           （仓内无库级归属标记，判据不可得）——该细分属设计者裁面。
        """
        if (key in self.ENGINE_META_GUARDED_KEYS and self.role != Role.PRIMARY
                and getattr(self, "_memory_dsn", None) is None):
            raise PermissionError(
                f"role={self.role.value} 无权写 engine_meta 治理键（{key}）——"
                f"治理键由父节点主控（#768）"
            )

    @_transactional
    def set_meta(self, key: str, value: str) -> None:
        """写引擎元数据（upsert）。

        #768：治理键（`ENGINE_META_GUARDED_KEYS`）写入前过角色闸——非 PRIMARY
        实例写治理键 ⇒ `PermissionError`（判据与边界见
        `_assert_meta_key_writable` docstring）。非治理键照旧放行。
        """
        self._assert_meta_key_writable(key)
        with self._lock:
            c = self.conn.cursor()
            c.execute("INSERT OR REPLACE INTO engine_meta (key, value) VALUES (?, ?)",
                      (key, str(value)))
            self._commit_locked()

    @_transactional
    def merge_meta(self, key: str, value_fn) -> Optional[str]:
        """读改写单点：同一写事务内「读旧值 → `value_fn(old)` 算新值 → 写回」（#483）。

        为什么必须是单点：`#483` 的自我模型快照要「先读回、再合并、后写」。若读与写
        分处两次调用，读到的旧值在写入之前就可能被他方（另一实例 / 另一进程）改掉，
        合并就建立在过期读数上。本方法把「读旧值」与「写新值」收进**同一条**写事务
        （`@_transactional` ⇒ 最外层 `BEGIN IMMEDIATE` 先取写锁），故事务内读到的必是
        最新已提交态，且其间无其他写者可插入——这是「不另加版本号也能免丢更新」的
        根据（读-改-写窗口被事务与写锁闭合）。
        返回被替换的旧值（键不存在则 None），供调用方判定/留痕。

        #768：与 `set_meta` 同一把闸——治理键在**调用 `value_fn` 之前**过角色闸
        （非 PRIMARY ⇒ `PermissionError`），否则「读改写」会在被拒的路径上先跑
        一遍取值回调（可能带副作用）。
        """
        self._assert_meta_key_writable(key)
        with self._lock:
            c = self.conn.cursor()
            c.execute("SELECT value FROM engine_meta WHERE key = ?", (key,))
            row = c.fetchone()
            old = row[0] if row else None
            new = value_fn(old)
            c.execute("INSERT OR REPLACE INTO engine_meta (key, value) VALUES (?, ?)",
                      (key, str(new)))
            self._commit_locked()
            return old

    @contextmanager
    def _structure_event_transaction(self):
        """结构事件工作单元；已有调用方事务时只释放自己的 savepoint。"""
        conn = self.conn
        nested = conn.in_transaction
        savepoint = f"structure_event_{uuid.uuid4().hex}"
        # ★ #665：`nested` 必须同时把「已被 @_transactional 包裹」算进来。
        # 单看 `conn.in_transaction` 不够——最外层 `_tx` 就是用它 BEGIN 的，
        # 但**裸 `BEGIN IMMEDIATE` 在已开事务里会抛 `cannot start a transaction
        # within a transaction`**；此处历史上只按 in_transaction 判，故凡是
        # 走 `_tx` 进入的调用方都会踩到。用 `_tx_depth` 兜住这一层。
        nested = nested or bool(getattr(self, "_tx_depth", 0))
        # 与普通写入同序：先取得 SQLite 写事务，再进入 Python 临界区。
        # 持 Python 锁等 SQLite，会与已占 SQLite、正等 Python 锁的写入互锁。
        conn.execute(f"SAVEPOINT {savepoint}" if nested else "BEGIN IMMEDIATE")
        try:
            with self._lock:
                yield conn.cursor()
                if nested:
                    conn.execute(f"RELEASE SAVEPOINT {savepoint}")
                else:
                    conn.commit()
        except BaseException:
            if nested:
                conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                conn.execute(f"RELEASE SAVEPOINT {savepoint}")
            else:
                conn.rollback()
            raise

    def _insert_structure_event(self, node: STNode, cursor):
        """内部追加：正文、类型、来源、同步标记和操作留痕共用一次提交。"""
        if node.layer in self.IMMUTABLE_LAYERS and self.role != Role.PRIMARY:
            raise PermissionError("共享层由父节点主控")
        cursor.execute("INSERT INTO nodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       node.to_row())
        event = node.state_attributes["structure_event"]
        cursor.execute(
            "INSERT INTO action_logs "
            "(ts, action_type, summary, node_ids, outcome, context) VALUES (?,?,?,?,?,?)",
            (time.time(), "structure_event", node.content,
             json.dumps([node.id]), json.dumps({"layer": node.layer.value}),
             json.dumps(event, ensure_ascii=False)))

    # ---- 结构层「重要事件」记录惯例（唯一真源 · #329） ----
    # 沿革：该惯例原在四处内联复制（adjust_attention_weight / migrate_v17_coordinates /
    # adjudicate_verifier_standard / register_external_anchor）；#329 收归单一真源后，
    # PR #429 把存储侧实现下沉到 LayeredStore（structure_service 直接用 store 入口），
    # 故本常量与辅助随之下沉——引擎侧 `_write_structure_record` 只是转发件。
    SUB_PENDING_SYNC_TAG = "pending_sync"

    def _write_structure_record(self, record: str, tag: str,
                                extra_tags: List[str] = None, *,
                                source: str = "", metadata: Dict = None,
                                sub_modality: str = "text",
                                record_class: str = "history",
                                importance: float = 0.9,
                                confidence: float = None,
                                _cursor=None) -> STNode:
        """结构层「重要事件」记录惯例（唯一收口 · 转发件）。

        第二实参 `tag` 是事件类型，正文的 `[tag] ` 前缀由
        `_record_structure_event` 统一添加（调用方不再自带前缀）。
        PRIMARY ⇒ 结构层（不可遗忘）；SUB ⇒ 知识层副本且**一律**带
        `SUB_PENDING_SYNC_TAG`。留史失败向调用方传播。
        """
        return self._record_structure_event(
            tag, record, extra_tags=extra_tags, source=source, metadata=metadata,
            sub_modality=sub_modality, record_class=record_class,
            importance=importance, confidence=confidence, _cursor=_cursor)

    def _record_structure_event(self, kind: str, content: str,
                                condition_space: ConditionSpace = None,
                                importance: float = 0.9, source: str = "",
                                metadata: Dict = None, confidence: float = None,
                                extra_tags: List[str] = None, sub_modality: str = "text",
                                record_class: str = "history", _cursor=None) -> STNode:
        """受信宿主内部留史；外部输入经引擎授权入口或认证服务。"""
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError("事件类型不能为空")
        # 角色分派（本类是唯一实现；#329 守卫 G4b 允许的第二处分派在引擎的
        # register_external_anchor 锚点腿）。SUB 腿的 pending_sync 标签取单一真源
        # 常量 SUB_PENDING_SYNC_TAG（#329 守卫 G4a：该字面量在 core.py 只允许一次）。
        if self.role == Role.PRIMARY:
            tags = ["structure", "structure_event", kind]
            modality = "structure"
            layer = MemoryLayer.STRUCTURE
            confidence_v = 1.0 if confidence is None else confidence
        else:
            tags = [self.SUB_PENDING_SYNC_TAG, "structure_event", kind]
            modality = sub_modality
            layer = MemoryLayer.KNOWLEDGE
            confidence_v = 0.8 if confidence is None else confidence
        cs = condition_space or ConditionSpace(
            "结构事件", "事件记录", (0, float('inf')), "历史记录，不是执行授权")
        tags += [tag for tag in (extra_tags or []) if tag not in tags]
        node = STNode(
            id=f"struct_event_{uuid.uuid4().hex}",
            content=f"[{kind}] {content}",
            modality=modality,
            spatial_coordinates={}, temporal_coordinate=time.time(),
            condition_space=cs, importance=importance,
            confidence=confidence_v,
            layer=layer,
            tags=tags,
            state_attributes={"structure_event": {
                "kind": kind, "source": source, "record_class": record_class,
                "metadata": metadata or {}}})
        if _cursor is not None:
            self._insert_structure_event(node, _cursor)
        else:
            with self._structure_event_transaction() as cursor:
                self._insert_structure_event(node, cursor)
        return node


    def migrate_v17_coordinates(self) -> Dict:
        """D-003 坐标分离：数据修改与迁移历史共用一个事务，任一步失败均回滚。

        #691 P1-c（坏 JSON 行崩构造期）：本方法由 `__init__` 调用，旧实现
        **前置筛只挡「不含候选子串」的坏行**；只要库内存在 ≥1 条合法候选，
        就会走全表 `SELECT` + 逐行 `json.loads` ⇒ 任一共存的坏 JSON 行（手工
        改坏 / 外来备份，且不含 `"protocol_` 等子串）都会令构造期抛
        `JSONDecodeError`，**整个引擎起不来**。同理前置筛自身对「含候选子串
        但 JSON 非法」的行也会崩。
        修法：两处 `json.loads` 逐行 try/except 具体异常（`ValueError` /
        `TypeError`），坏行**跳过并计数**——计数经既有 `warnings.warn` 告警面
        上报（不静默吞；`warn` 属既有审计动作，lint 计数零净增），返回值形状
        保持 `{"migrated_nodes": N}` 不变（不破坏既有调用方契约）。**未**新增宽
        `except Exception`；迁移**判据与阈值未动**（仍是顶层 `startswith` 三前缀）。

        #769 ③（共享层写面补角色闸）：本方法是**构造期维护**（`__init__` 无条件
        调用），不是用户写操作——但旧实现无条件改写**全部行**的
        `spatial_coordinates` / `semantic_coordinates` ⇒ 「仅构造一个 SUB 实例」
        即越权改写共享层（锚点/结构层）行的坐标列（核验帖实测：8 字段中这两列
        被改、其余 6 列不动，并留一条不记发起者角色的 `structure_event` 审计）。
        修法取**取向 (b)**：非 PRIMARY 时**只跳过共享层行**、本地层照迁——既不
        越权改共享层，又不让 SUB 实例的本地数据永久停在旧格式（取向 (a)「SUB
        整体不迁移」会让本地层旧格式永久滞留，代价更大）。判据复用唯一单点
        `layer_writable_by_role`（与 `add_node` 同源）。跳过数经既有
        `warnings.warn` 面上报（不静默），返回值形状不变。
        """
        # 常规启动通常已无旧字段：先做只读候选检查，避免每次启动争写锁。
        candidates = self.conn.execute(
            "SELECT spatial_coordinates FROM nodes WHERE "
            "instr(spatial_coordinates, '\"protocol_') > 0 OR "
            "instr(spatial_coordinates, '\"radical_') > 0 OR "
            "instr(spatial_coordinates, '\"neural_') > 0").fetchall()
        has_valid_candidate = False
        for row in candidates:
            try:
                obj = json.loads(row[0] or "{}")
            except (ValueError, TypeError):
                # 坏 JSON 候选行：前置筛不崩（计数在下方全表循环统一做）。
                continue
            if isinstance(obj, dict) and any(
                    k.startswith(("protocol_", "radical_", "neural_")) for k in obj):
                has_valid_candidate = True
                break
        if not has_valid_candidate:
            return {"migrated_nodes": 0}
        migrated = 0
        skipped_bad_rows = 0
        skipped_shared_rows = 0
        with self._structure_event_transaction() as c:
            c.execute("SELECT id, spatial_coordinates, semantic_coordinates, layer FROM nodes")
            for row in c.fetchall():
                nid, sp_json, se_json = row[0], row[1], row[2]
                nid_layer = row[3]
                # #691 P1-c：坏行跳过并计数（构造期不再崩）。
                try:
                    sp = json.loads(sp_json or "{}")
                    se = json.loads(se_json or "{}") if se_json else {}
                except (ValueError, TypeError):
                    skipped_bad_rows += 1
                    continue
                if not isinstance(sp, dict):
                    skipped_bad_rows += 1
                    continue
                semantic_keys = [k for k in sp if k.startswith(
                    ("protocol_", "radical_", "neural_"))]
                if not semantic_keys:
                    continue
                # #769 ③（构造期维护 ≠ 用户写操作，但仍属共享层写面）：本方法由
                # `__init__` **无条件**调用 ⇒ 旧实现「仅构造 SUB 实例」即改写共享层
                # 行的 `spatial_coordinates` / `semantic_coordinates`（核验帖实测：
                # 8 字段中 spatial/semantic 两列被改，其余 6 列不动；并留一条
                # `structure_event` 审计，但不记发起者角色）。取向 (b)——**只跳过
                # 共享层行、本地层照迁**（既不越权改共享层，又不让 SUB 实例的本地
                # 数据永久停在旧格式；取向 (a)「SUB 整体不迁移」会让本地层旧格式
                # 永久滞留，代价更大）。跳过数经既有 `warnings.warn` 面上报（不静默），
                # 返回值形状 `{"migrated_nodes": N}` 不变（不破坏既有调用方契约）。
                if not self.layer_writable_by_role(nid_layer):
                    skipped_shared_rows += 1
                    continue
                se.setdefault("protocol", {}).setdefault("concept", {})
                se.setdefault("radical", {})
                se.setdefault("neural", {})
                for k in semantic_keys:
                    v = sp.pop(k)
                    if k.startswith("protocol_"):
                        se["protocol"]["concept"][k[len("protocol_"):]] = v
                    elif k.startswith("radical_"):
                        se["radical"][k[len("radical_"):]] = v
                    elif k.startswith("neural_"):
                        se["neural"][k[len("neural_"):]] = v
                c.execute("UPDATE nodes SET spatial_coordinates=?, semantic_coordinates=? WHERE id=?",
                          (_dumps_json(sp, where="spatial_coordinates"),
                           _dumps_json(se, where="semantic_coordinates"), nid))
                migrated += 1
            if migrated:
                self._write_structure_record(
                    f"v1.7 坐标字段分离：{migrated} 节点语义键迁移至 semantic_coordinates",
                    "migration", extra_tags=["v1.7"], source="migrate_v17_coordinates",
                    metadata={"version": "v1.7", "migrated_nodes": migrated}, _cursor=c)
        if skipped_bad_rows:
            # #691 P1-c：坏行**不静默吞**——经既有告警面上报（`warn` 属既有审计动作，
            # 不改返回值形状、不加宽 except 计数）。这些行原文未被改写，仍是待修数据。
            warnings.warn(
                "#691 坐标迁移跳过 %d 条坏 JSON 行（spatial/semantic 非法，原文未改）"
                % skipped_bad_rows, RuntimeWarning, stacklevel=2)
        if skipped_shared_rows:
            # #769 ③：共享层行被角色闸跳过 ⇒ 上报（不静默）。取向 (b) 的代价面：
            # 这些行仍是旧格式，需由 PRIMARY 实例完成迁移。
            warnings.warn(
                "#769 坐标迁移跳过 %d 条共享层（anchor/structure）行：role=%s 无权改写"
                "共享层坐标列（本地层行已照迁；共享层行需由 PRIMARY 实例迁移）"
                % (skipped_shared_rows, self.role.value), RuntimeWarning, stacklevel=2)
        return {"migrated_nodes": migrated}


    def get_node(self, node_id: str) -> Optional[STNode]:
        """按 id 取节点，不存在返回 None。"""
        c = self.conn.cursor()
        c.execute("SELECT * FROM nodes WHERE id=?", (node_id,))
        row = c.fetchone()
        if row is None:
            return None
        return STNode.from_row(tuple(row))

    def _strip_incident_edges(self, c, node_id: str) -> None:
        """删除与 `node_id` 关联的全部边；对「对端为不可变层」的边先落审计。

        #168：`delete_node` 与 `decay_cycle` 情境层删除段都以
        `DELETE FROM edges WHERE source_id=? OR target_id=?` 级联清边。当对端是
        锚点/结构层（`IMMUTABLE_LAYERS`）节点时，该节点的因果关系被常规遗忘
        **静默剥光**——`decay_cycle` docstring 声称「其关联边也不衰减」，而实际
        是随对端节点消失而整条删除，且无日志无审计。
        本助手在删边前对这类边落一条 `action_logs`（审计只追加），使「不可变层
        节点失去关联边」可事后追查；**不**改变删边语义（保留边会造悬空引用，
        破坏 `verify_integrity` 的完整性判据）。
        判据来源：分诊表 `triage_lingshu_ALL.md` 行 168；仓内无规定该留痕面的
        理论章节，追不到更早出处（经验标定，本件 #168）。
        边界（削掉的判别力）：本件只把「静默」改为「留痕」——**不**阻止不可变层
        节点失去关联边（删对端节点是合法遗忘路径；另注 `forget_advisor` 的
        archived 降权路径亦不在本件范围内）。
        """
        c.execute("SELECT id, relation_type, source_id, target_id FROM edges"
                  " WHERE source_id=? OR target_id=?", (node_id, node_id))
        _rows_e = c.fetchall()
        # ★优化（本笔 · 零行为变更）：旧实现逐边 `self.get_node(other)` —— 即
        # `SELECT * FROM nodes WHERE id=?`（读**整行** + 建 `STNode`），而本处只用
        # **对端 layer** 一个字段。改为先收集全部对端 id、按 `_SQL_IN_CHUNK`（#417
        # 的 IN 绑定上限单点）分批取 `id, layer`，循环内查表。
        # 转换用 `MemoryLayer(...)`，与 `STNode.from_row` 的 `MemoryLayer(row[8])`
        # **同款裸露转换** ⇒ 行内 layer 非法时的失败语义与旧路径逐位一致。
        # #720：该「逐位一致」是**本件要收口的缺陷面**——`layer` 存了未知值
        # （如 `'not-a-layer'`）时 `MemoryLayer(...)` 抛 `ValueError`，整条
        # `delete_node` / `decay_cycle` 情境段中止（**合法节点因邻接坏行删不掉**，
        # 与坏 `tags` 同族）。#267 的逐行容错只收在批量装载面 `_tolerant_nodes`，
        # 本处与下方 tags 解析同属单点裸解析的漏网调用点。
        # 现逐行容错：解析不出即**不登记**该对端（⇒ 按「非不可变层」处理、不落
        # 不可变层审计——不猜它的层），id 如实计入 `last_skipped_rows`，
        # 删除主流程不中断。
        _others = list({(tid if sid == node_id else sid)
                        for _eid, _rel, sid, tid in _rows_e})
        _layer_of: Dict[str, MemoryLayer] = {}
        _bad_layer: List[str] = []
        for _i in range(0, len(_others), _SQL_IN_CHUNK):
            _chunk = _others[_i:_i + _SQL_IN_CHUNK]
            _ph = ",".join("?" * len(_chunk))
            for _oid, _lay in c.execute(
                    f"SELECT id, layer FROM nodes WHERE id IN ({_ph})",
                    _chunk).fetchall():
                try:
                    _layer_of[_oid] = MemoryLayer(_lay)
                except (TypeError, ValueError):
                    _bad_layer.append(_oid)
        if _bad_layer:
            # #720：与调用方本轮已记的坏行**合并**（同一轮内本助手可能被多次调用）
            # ——不覆盖，避免漏报。
            self.last_skipped_rows = list(dict.fromkeys(
                list(self.last_skipped_rows or []) + _bad_layer))
        for eid, rel, sid, tid in _rows_e:
            other = tid if sid == node_id else sid
            _lay = _layer_of.get(other)
            if _lay is None or _lay not in self.IMMUTABLE_LAYERS:
                continue
            c.execute("INSERT INTO action_logs (ts, action_type, summary, node_ids,"
                      " outcome, context) VALUES (?,?,?,?,?,?)",
                      (time.time(), "immutable_edge_stripped",
                       f"不可变层节点 {other} 因 {node_id} 被删而失去 {rel} 边",
                       json.dumps([other, node_id], ensure_ascii=False),
                       json.dumps({"edge_id": eid, "relation_type": rel,
                                   "immutable_node": other, "removed_peer": node_id},
                                  ensure_ascii=False),
                       json.dumps({}, ensure_ascii=False)))
        c.execute("DELETE FROM edges WHERE source_id=? OR target_id=?", (node_id, node_id))

    @_transactional
    def delete_node(self, node_id: str) -> bool:
        """仅允许删除非锚点/非结构层的节点；受保护节点（`protections` 行登记或
        `no_forget` 标签）同样拒删——bool 契约不变（拒 ⇒ False）。

        #36：两条遗忘路径（`decay_cycle` 情境层删除段 / `enforce_context_cap`
        → 本函数）此前都不查保护名单 ⇒ 受保护节点照样被删。现收口到本原语。
        判据来源：设计稿 `docs/plans/删除恢复持久化完整性_修复设计_v0.1.md`
        §3.3「保护判定下沉到唯一删除原语 `delete_node`」＋ 待裁清单 D-15（应用层
        单一谓词）；`forget_advisor`/`consolidate_cycle` 的既有「双查法」即谓词口径。

        #115：删除节点会级联删除其全部关联边（含 `OPPOSITE` 冲突边）⇒ 冲突关系
        消失时，对端节点上的 `conflict` 标签须**随之移除**（B-4 / D-37「随 OPPOSITE
        边消失而移除」），并落一条审计（B-5 / D-39「否则冲突关系静默失忆」）。
        对端若仍与其它节点存在冲突边，标签保留（同一节点可参与多个冲突）。
        判据来源：`docs/plans/待裁清单_v0.1.md` B-4 / B-5 / D-37 / D-39。

        #168：删边改经 `_strip_incident_edges`（单一真源）——对端为锚点/结构层
        （`IMMUTABLE_LAYERS`）的边在删除前落一条 `immutable_edge_stripped` 审计，
        使不可变层节点失去关联边**可追查**（详见该助手 docstring）。

        ★本笔（治理写面「同列两套口径」收口 · P3）：对端标签清理腿补**层 × 角色**
        判据——主闸只判被删**目标行**，本腿此前对 `nodes.tags` 零判据 ⇒ SUB 删本地行
        时可改写共享层 peer 的 `tags`（实测，跨实例读回同值）。形态＝peer 不可写则
        **跳过该 peer 的标签清理 + 计入 `last_skipped_rows`**（不抛错，与同段 `#720`
        坏 tags 容错同款口径；理由与后果面见循环内注释）。
        """
        # #517：快速预读只服务 bool 契约的早退（不存在/不可变层），**保护判定不再
        # 留在无事务快照上**——旧实现「autocommit 读判定（t0 快照）→ 首条 DML 才隐式
        # BEGIN」之间，另一进程 BEGIN IMMEDIATE 的 protect_node 可提交插队（#36 并发
        # 绕过：保护成功返回后节点仍被删、protections 留孤儿行，两进程压测 2/100 命中）。
        node = self.get_node(node_id)
        if node is None:
            return False
        if node.layer in self.IMMUTABLE_LAYERS:
            return False
        with self._lock:
            c = self.conn.cursor()
            # 显式开写事务（与 protect_node 的 BEGIN IMMEDIATE 互相排队），**随后**才做
            # 「不可遗忘」判定——判定与 DELETE 共用同一写快照：窗口期已提交的保护行对本
            # 快照可见（拒删）；未提交的排在删除事务之后（保护落在已删节点＝既有语义，
            # 非本件竞态）。重读行防 t0 预读与写快照间节点被改写层。
            # ★ #665：本方法已挂 @_transactional，最外层事务已由 `_tx` 开好；此处再
            # 裸 BEGIN IMMEDIATE 会抛「cannot start a transaction within a transaction」。
            # 仅在「无人替我们开事务」时（_tx_depth==0）才需要自己开。
            if getattr(self, "_tx_depth", 0):
                pass  # 事务已由 _tx 开启，且写锁已拿到，无需重复 BEGIN
            else:
                self.conn.execute("BEGIN IMMEDIATE")
            node = self.get_node(node_id)
            if node is None or node.layer in self.IMMUTABLE_LAYERS:
                return False
            if self._is_forget_protected(node):
                return False  # 不可遗忘（3.2 节）：受保护件不因任何删除路径消失
            # #115：删除前记录 OPPOSITE 邻居——边随节点删除而消失，冲突标签须随之移除。
            c.execute("SELECT source_id, target_id FROM edges"
                      " WHERE relation_type=? AND (source_id=? OR target_id=?)",
                      (EdgeType.OPPOSITE.value, node_id, node_id))
            peers = {t if s == node_id else s for s, t in c.fetchall()}
            # ★本笔（治理写面「同列两套口径」收口 · P3）：peer 的 **layer** 用于下面
            # 的写权限判据；一次性批量取回（沿 `#417` 的 `_SQL_IN_CHUNK` 分批单点，
            # 与 `_strip_incident_edges` 同款），**不**在循环内逐 peer 回查——本方法
            # 有「循环体内无 `get_node` / 不逐节点 SELECT」的 N+1 守卫
            # （`tests/test_core_nplus1_getstats_forgetadvisor.py` I 段）。
            # 取不到的 peer（行不存在 / layer 非法值）不登记 ⇒ 判据得 `None` ⇒ 放行：
            # 这类行不可能是共享层行（`SHARED_LAYERS` 的两个取值都是合法枚举值），
            # 合法性失败语义仍由既有 tags 解析腿承担。
            _peer_layers: Dict[str, str] = {}
            _peers_list = sorted(peers)
            for _pi in range(0, len(_peers_list), _SQL_IN_CHUNK):
                _pchunk = _peers_list[_pi:_pi + _SQL_IN_CHUNK]
                _pph = ",".join("?" * len(_pchunk))
                for _pid, _play in c.execute(
                        f"SELECT id, layer FROM nodes WHERE id IN ({_pph})",
                        _pchunk).fetchall():
                    _peer_layers[_pid] = _play
            # #720：本方法逐行容错计数（写路径也刷新一次；语义＝「本次调用跳过的
            # 坏行 id」，与装载面 `_tolerant_nodes` 的刷新口径同名同义）。
            self.last_skipped_rows = []
            # #168：删边经唯一真源（对不可变层对端的边先落审计）
            # #93：**先删边再删节点**——开启 `PRAGMA foreign_keys=ON` 后，节点仍被
            # 关联边引用时 `DELETE FROM nodes` 会抛 `FOREIGN KEY constraint failed`
            # （无 `ON DELETE CASCADE`）。旧顺序（先删节点后删边）在无外键强制的年代
            # 无害，开外键后即整条删除路径报错；顺序反了必红。
            self._strip_incident_edges(c, node_id)
            c.execute("DELETE FROM nodes WHERE id=?", (node_id,))
            # #448：硬删同步清归档区登记，避免 `archive_state` 留孤儿行混淆
            # 「仍可还原的归档节点」与「已永久删除的节点」。
            c.execute("DELETE FROM archive_state WHERE node_id=?", (node_id,))
            _bad_tag_peers: List[str] = []
            _locked_tag_peers: List[str] = []
            for peer in peers:
                c.execute("SELECT COUNT(*) FROM edges WHERE relation_type=?"
                          " AND (source_id=? OR target_id=?)",
                          (EdgeType.OPPOSITE.value, peer, peer))
                if c.fetchone()[0]:
                    continue  # 对端仍与其它节点冲突 ⇒ 标签保留
                # ★优化（本笔 · 零行为变更）：本处只用对端的 **tags** 一个字段，
                # 旧实现 `self.get_node(peer)` 是 `SELECT *`（读整行 + 建 `STNode`）。
                # 改为只取 `tags`。
                # #720：`json.loads` 改为**逐行容错**——坏 tags（非法 JSON）此前
                # 让整个 `delete_node` 抛 `JSONDecodeError`，**合法节点因邻接坏行
                # 删不掉**（本事务回滚、节点仍在库）。#267 的容错只收在批量装载面
                # `_tolerant_nodes`，本处是单点裸解析的漏网调用点。
                # 口径与 `_tolerant_nodes` 一致：坏行**跳过**（本处＝该 peer 的
                # `conflict` 标签清理跳过、标签原样留在库，不去猜它的内容），
                # 跳过 id 如实计入 `last_skipped_rows`；**不**阻断删除主流程。
                c.execute("SELECT tags FROM nodes WHERE id=?", (peer,))
                _prow = c.fetchone()
                # ★本笔（治理写面「同列两套口径」收口 · P3）：主闸只判**被删目标行**
                # （上方 `node.layer in IMMUTABLE_LAYERS`），本 peer 腿此前对
                # `nodes.tags` **零层/角色判据** ⇒ SUB 删自己的本地行时，可把共享层
                # peer 的 `tags` 由 `["conflict","no_forget"]` 改成 `["no_forget"]`
                # （实测，PRIMARY 跨实例读回同值）；而同列经 `tag_node` 打共享层目标
                # 必被 `#234` 拦 ⇒ 同一列两套口径。判据复用 store 侧单点
                # （`layer_writable_by_role`，不另造第二份）。
                # 形态＝**跳过该 peer 的标签清理 + 计入 `last_skipped_rows`**，不抛错
                # （与同段 `#720` 坏 tags 容错同款口径）：`delete_node` 的语义是删
                # **本地层**节点，因一个**邻居**是共享层就让整条删除失败，会把
                # `decay_cycle` / `enforce_context_cap` 等既有删除路径打成红。
                # 后果面（如实标注）：跳过会在共享层 peer 上留下**悬挂的 `conflict`
                # 标签**（OPPOSITE 边已随删除消失、标签未被移除）——这是「宁可留脏
                # 标签，也不让非 PRIMARY 改写共享层」的取舍，清理由 PRIMARY 侧承担。
                if not self.layer_writable_by_role(_peer_layers.get(peer)):
                    _locked_tag_peers.append(peer)
                    continue
                try:
                    _ptags = ((json.loads(_prow[0]) if _prow[0] else [])
                              if _prow else None)
                except (TypeError, ValueError):
                    _bad_tag_peers.append(peer)
                    continue
                if _ptags is not None and "conflict" in _ptags:
                    tags = [t for t in _ptags if t != "conflict"]
                    c.execute("UPDATE nodes SET tags=? WHERE id=?", (_dumps_tags(tags), peer))
                    c.execute("INSERT INTO action_logs (ts, action_type, summary, node_ids,"
                              " outcome, context) VALUES (?,?,?,?,?,?)",
                              (time.time(), "conflict_tag_cleared",
                               f"OPPOSITE 边随节点 {node_id} 删除而消失，移除 {peer} 的 conflict 标签",
                               json.dumps([peer], ensure_ascii=False),
                               json.dumps({"reason": "opposite_edge_gone",
                                           "removed_node": node_id}, ensure_ascii=False),
                               json.dumps({}, ensure_ascii=False)))
            if _bad_tag_peers or _locked_tag_peers:
                self.last_skipped_rows = list(dict.fromkeys(
                    list(self.last_skipped_rows or [])
                    + _bad_tag_peers + _locked_tag_peers))
            self._commit_locked()
        return True

    @_transactional
    def update_node_confidence(self, node_id: str, delta: float):
        """更新节点置信度（验证单元复核后调用）

        #185：`delta` 经 `_finite_or` 归一（非有限/非数 → 0.0，即「不改动」）。
        旧实现把 delta 原样拼进 `confidence + ?` ⇒ `confidence + NaN = NaN` 落库成
        NULL（`MIN/MAX` 遇 NULL 仍回 NULL）；此后 `decay_cycle` 的
        `WHERE confidence > ? AND confidence < 1.0` 对 NULL 全不成立 ⇒ 该行永不衰减。

        #769 ④ / #780 ②（D-33 的「另笔」）：本接口此前**无层/角色判据** ⇒ SUB 直调
        即把 PRIMARY 建的锚点/结构层行 `confidence 1.0 → 0.5`（核验帖实测，跨实例读回
        同值）。现按 `add_node` 的同款判据（`layer_writable_by_role` 单点）**写前按
        库中旧行判权限**（`#127` D-07 形状）：非 PRIMARY 且目标行属 `SHARED_LAYERS`
        ⇒ `PermissionError`（fail-closed、零落库）。本地层（knowledge/context/self）
        的增信路径不受影响（SUB 对自己节点的复核照旧放行——`#191` 裁定 SELF 属本地层）。
        """
        delta = _finite_or(delta, 0.0)
        # 闸落在 UPDATE 之前（本方法已挂 @_transactional ⇒ 被拒时零落库）。
        self.assert_node_writable_by_role(node_id, action="改写共享层节点的 confidence")
        with self._lock:
            c = self.conn.cursor()
            c.execute('''
                UPDATE nodes SET confidence = MIN(1.0, MAX(0.0, confidence + ?)),
                                  last_access = ?
                WHERE id=?
            ''', (delta, time.time(), node_id))
            self._commit_locked()

    @staticmethod
    def _retrieval_touches_recency() -> bool:
        """检索命中是否刷新 last_access。

        默认 True（与改前一致）。`LINGSHU_RETRIEVAL_NO_TOUCH=1` 时为 False：检索只计
        access_count，不再把 last_access 改成「现在」。原因：recall 的近因分量
        （0.2·1/(1+天数)）读的正是 last_access，而 search_content 每返回一个节点就
        刷新它 ⇒「被搜到一次 → 下次更容易被搜到」的自增强回路，与该记忆是否被
        真正使用、是否正确无关。关闭后 last_access 只由写入、M5 重复观测、置信度
        复核、巩固复述等**真实强化**事件刷新。"""
        return os.environ.get("LINGSHU_RETRIEVAL_NO_TOUCH", "").strip() != "1"

    def increment_access(self, node_id: str, touch: bool = True):
        """访问计数+1（尽力而为：外部写锁持有时立即放弃，不阻塞检索主流程）

        ★本笔：实现收进 `increment_access_many` 单点（同一连接/提交语义），
        本函数保留为单节点入口，行为与旧实现逐位一致。
        touch=False：只计数、不刷新 last_access（检索路径在开关打开时使用）。
        ★本笔合成：两分支都委派给批量单点（不复制实现）——`touch=True` 走与 #571
        逐位一致的默认入口，`touch=False` 多传一个 `touch` 形参；故「单节点入口委派
        批量单点」与「不刷新近因」两套语义并存。"""
        if touch:
            self.increment_access_many([node_id])
        else:
            self.increment_access_many([node_id], touch=False)

    def increment_access_many(self, node_ids, touch: bool = True) -> None:
        """**批量**访问计数+1（尽力而为；逐条语义与 `increment_access` 相同）。

        ★优化（本笔 · 零行为变更）：`increment_access` 每次调用都**新开一个 sqlite
        连接**（`sqlite3.connect(self.db_path, timeout=0)` + commit + close）。而
        `semantic_search` 按命中数**逐节点**调它 ⇒ N 次 connect/commit/close
        （每次连接都要重设连接级 `PRAGMA foreign_keys=ON`）。本方法把同一批收进
        **一条连接、一次提交**。

        为什么用 `executemany` 逐条语句、而**不**写 `WHERE id IN (...)`：后者会把
        **重复 id 只计一次**，而旧路径对重复 id 各计一次（`semantic_search` 的
        provider 返回的 id 列表可含重复）。逐条语句保住了这一口径。

        ★本笔合成：`touch=False` 只计数、不刷新 `last_access`（检索路径在
        `LINGSHU_RETRIEVAL_NO_TOUCH=1` 时走此支）。「批量口径」（一条连接/逐条
        `executemany`）与「是否刷新近因」（`touch`）二者**正交**——故同一条连接上
        按 `touch` 选择 `executemany` 的 SQL，两套语义并存。

        失败容忍（#775 起**不再静默**）：`sqlite3.OperationalError` 仍不阻断检索
        主流程，但经 `_note_access_count_failure()` 计数 + 留原文 + `warnings.warn`
        上报——旧实现是 `pass`，写竞争下访问计数**静默少记**（详见类 docstring 与
        `_note_access_count_failure`）。

        ★`timeout=0` **保留不动**（刻意的非阻塞设计，非本缺陷）：本连接自开、
        不经 `_connect`，故不继承 `busy_timeout=30000` —— 外部写锁持有时**立即**
        放弃（裸 SQL 对照：同语句、同持锁态下必抛 `database is locked`），从而
        绝不把 `recall` / `search_content` 的 touch 路径拖住最多 30 秒。本笔**只**
        把「被吞掉的失败」改为「被上报的失败」，**不**改锁等待取向（改锁等待属
        语义变更，见报告「取向题：要不要为访问计数等锁」）。
        """
        ids = [i for i in (node_ids or []) if i]
        if not ids:
            return
        try:
            # #52：`:memory:` 后端时 `self.db_path` 是 ":memory:" 字面量——
            # `sqlite3.connect(":memory:")` 得到**另一个空库**（每连接一个独立库），
            # UPDATE 落在 `no such table` 上被吞 ⇒ 访问计数静默失效（实测 3 次调用
            # 后 access_count=0，文件库=3）。内存库实际载体是 shared-cache DSN
            # （`_memory_dsn`，见 `__init__`），旁路连接必须带 `uri=True` 连它才看
            # 得到同一份数据；`timeout=0` 的非阻塞设计保留不动。文件库路径原样。
            dsn = self._memory_dsn or self.db_path
            obs = sqlite3.connect(dsn, timeout=0,
                                  uri=dsn.startswith("file:"))
            try:
                # #93：本连接独立于 `_connect`（访问计数为尽力而为的旁路写），
                # 连接级外键开关必须逐连接重设，否则该连接上外键不生效。
                obs.execute("PRAGMA foreign_keys=ON")
                if touch:
                    # #52：整批共用**一个**时间戳——此前逐行 time.time()，而 ids 是
                    # 按检索得分序传入的（高分在前）⇒ 低分节点拿到更晚的
                    # last_access，下一次 recall 的近因项反向偏向低分节点
                    # （:memory: 上该刷新曾因 #52 静默丢失而从未生效，修复后暴露，
                    # W6 在依赖下界档复现）。一次检索事件＝一个时刻。
                    ts = time.time()
                    obs.executemany(
                        'UPDATE nodes SET access_count = access_count + 1,'
                        ' last_access = ? WHERE id=?',
                        [(ts, nid) for nid in ids])
                else:
                    obs.executemany(
                        'UPDATE nodes SET access_count = access_count + 1 WHERE id=?',
                        [(nid,) for nid in ids])
                obs.commit()
            finally:
                obs.close()
        except sqlite3.OperationalError:
            # #775：计数失败仍可容忍（统计性质、不阻断检索），但**不得静默**。
            self._note_access_count_failure()

    def _note_access_count_failure(self):
        """上报一次「访问计数旁路写失败」（#775：取代旧实现的 `except …: pass`）。

        三档读数（皆可读、可断言）：
          ① `self.access_count_write_failures` 自增 —— 累计失败批次数；
          ② `self.last_access_count_error` —— 末次错误原文（`类型: 消息`）；
          ③ `warnings.warn` —— 每次失败即时上报（属既有审计动作，
             `silent_except_lint` 认它；调用方可用 `pytest.warns` / 日志观测）。

        为什么异常对象取自 `sys.exc_info()` 而不写成 `except … as exc`：
        `increment_access_many` 有一处**既有静态守卫**
        （`tests/test_core_nplus1_getstats_forgetadvisor.py` 的 K 组）逐字要求该
        函数体内出现不带 `as` 的 `except sqlite3.OperationalError:`——本笔不为
        迁就自身写法去改那条既有断言（守卫是既有的，改它属超范围）。
        """
        exc = sys.exc_info()[1]
        self.access_count_write_failures = (
            getattr(self, "access_count_write_failures", 0) + 1)
        self.last_access_count_error = (
            "%s: %s" % (type(exc).__name__, exc) if exc is not None else None)
        warnings.warn(
            "#775 访问计数写入失败（第 %d 次；该批已放弃，不阻断检索）：%s"
            % (self.access_count_write_failures, self.last_access_count_error),
            RuntimeWarning, stacklevel=3)

    # ---------- 边操作 ----------

    def _assert_edge_endpoints_role(self, source_id: str, target_id: str) -> None:
        """非 PRIMARY 不得写**任一端为共享层**（anchor/structure）的边（#234）。

        **边角色闸的单一真源**（#674）：`_assert_edge_role`（`add_edge`/`verify_edge`/
        `unverify_edge` 等公开 API 写路径）与 `import_all` 的 edges 导入腿**共用**本方法
        ——两条路径不可能再分叉（#674 的病灶正是「导入腿另有一份无闸实现」）。

        判据来源：`Role` 注释（core.py:631-632）「PRIMARY 可写共享层（锚点层、结构
        层）／SUB 仅本地层」＋ `LayeredStore` docstring「共享层……跨实例同步，不可
        遗忘」＋ 公开 issue #234「期望行为：非 PRIMARY 角色对任一端属于
        `IMMUTABLE_LAYERS` 的边做新增、修改、verify、删除，都应得到
        `PermissionError`」＋ 公开 issue #674（导入腿残余面）。
        边界（削掉的判别力，如实标注）：本闸用 `SHARED_LAYERS`（锚点/结构）而非
        issue 字面的 `IMMUTABLE_LAYERS`——#191 已裁定 SELF 属**本地层**（SUB 可写，
        见 `add_node` docstring），若按 `IMMUTABLE_LAYERS` 收口会把 SUB 的
        `update_self(link_to_node_id=…)` 自我快照因果边一并拒掉，**回归 #191**。故
        本件按仓内既有「写权限 = `SHARED_LAYERS`」单一真源收口；SUB 对 PRIMARY 的
        SELF 节点边仍可写（SELF 为实例本地自我模型，非共享层）。**削掉的判别力**：
        对 SELF 端点的边不做角色限制（与 #191 一致）。
        """
        if self.role != Role.PRIMARY:
            # ★优化（本笔 · 零行为变更）：本闸只看端点的 **layer** 一个字段，而旧实现
            # 逐端点 `self.get_node(nid)`（`SELECT *`，读整行 + 建 `STNode`）。
            # 改为只取 `layer`（两个 id、最多两条轻查询）。`MemoryLayer(...)` 与
            # `STNode.from_row` 同款裸露转换 ⇒ 非法 layer 的失败语义一致；
            # 不存在该 id 时 `fetchone()` 为 None，与旧 `n is None` 同分支。
            _c = self.conn.cursor()
            for nid in (source_id, target_id):
                _c.execute("SELECT layer FROM nodes WHERE id=?", (nid,))
                _row = _c.fetchone()
                if _row is None:
                    continue
                _lay = MemoryLayer(_row[0])
                if _lay in self.SHARED_LAYERS:
                    raise PermissionError(
                        f"role={self.role.value} 无权写共享层端点（{nid} 为 "
                        f"{_lay.value}）的边，共享层边由父节点主控（#234）")

    def _assert_edge_role(self, edge: STEdge) -> None:
        """边对象入口：委托 `_assert_edge_endpoints_role`（单一真源，见该助手）。"""
        self._assert_edge_endpoints_role(edge.source_id, edge.target_id)

    @_transactional
    def add_edge(self, edge: STEdge) -> str:
        """写入有向边并返回 id（边的条件空间决定路由资格——条件挂在边上）。

        #234：非 PRIMARY 不得新增/覆盖任一端为共享层的边——旧实现 `INSERT OR
        REPLACE`
        无任何角色/层检查，SUB 可在两个结构层节点间新增「已验证」边（`decay_cycle`
        跳过两端属 anchor/structure 的边 ⇒ **永不衰减**，并立即参与 PRIMARY 的
        `infer_causal_paths`/`reason_causal`），或用同 id 覆盖 PRIMARY 的共享层边
        （改 relation_type/端点/verified）。现先判后写（fail-closed），并查**同 id
        旧边**的端点（防「先建本地边占位、再用共享层 id 覆盖」）。
        判据来源：公开 issue #234「建议修复 1」＋ `SHARED_LAYERS` 语义（见
        `_assert_edge_role`）。
        边界（削掉的判别力，如实标注）：仍用 `INSERT OR REPLACE`（PRIMARY 原位覆盖
        照旧）——issue 建议 1 的「共享层边改用 `INSERT`（冲突即报错）」属更彻底改动，
        会波及同步逻辑，本件不做（另笔）。
        """
        self._assert_edge_role(edge)
        if self.role != Role.PRIMARY:
            old = self.get_edge(edge.id)
            if old is not None:
                self._assert_edge_role(old)
        with self._lock:
            c = self.conn.cursor()
            c.execute('INSERT OR REPLACE INTO edges VALUES (?,?,?,?,?,?,?,?,?,?,?)', edge.to_row())
            self._commit_locked()
        return edge.id

    def get_edge(self, edge_id: str) -> Optional[STEdge]:
        """按 id 取边，不存在返回 None。

        #653：坏行（未知 relation_type / 非法 condition_space / 缺列）同样返回
        None 而非抛——单条坏边不再打断调用链（`add_edge` 的角色检查、
        `verify_edge` 等单条读调用方）。判据来源：issue #653（`from_row` 裸调
        `EdgeType(row[3])` / `ConditionSpace.from_json(row[4])` 使单条坏边
        中断所有边读取）。
        """
        c = self.conn.cursor()
        c.execute("SELECT * FROM edges WHERE id=?", (edge_id,))
        row = c.fetchone()
        if row is None:
            return None
        # #653：单条读经 `_tolerant_edges`——坏行视同「读不到」。对照：节点侧
        # `get_node` 是 #267 修复时**保留**的裸调边界（单条读不容错）；边侧
        # 单条读收容错是本件判据（`add_edge`/`verify_edge` 等调用方连带坏边
        # 中断，见 issue #653 读数）。
        edges, _ = self._tolerant_edges([row])
        return edges[0] if edges else None

    @_transactional
    def verify_edge(self, edge_id: str, new_confidence: float = None):
        """验证单元复核后标记边为已验证

        #185：`new_confidence` 经 `_finite_clamp` 归一（有限值钳到 [0,1]）——旧实现
        原样落库，NaN 存成 NULL 后 `decay_cycle` 的 `cred_step(None, f)` 抛
        TypeError、整轮衰减中断。

        #790（本件，收口 #185 留下的「静默归一」面）：`new_confidence` 的非有限值
        （NaN/±inf）改经 `_require_finite` **fail-closed 拒收**（`ValueError`），
        不再被 `_finite_clamp` 静默归到 0.0。修前实测（main @ 5c530ff）：
        `verify_edge(id, new_confidence=NaN/inf/-inf)` 一律**照常成功**——边被
        标 `verified=1`、`confidence` 被写成凭空的 0.0，调用方拿不到任何「复核值
        是坏值」的信号（这与「复核」语义相反：复核入口把垃圾读数盖成合法值并
        盖章）。取 `_require_finite`（#206 同款闸）而非新写 `math.isfinite` 判断，
        保持数值入参闸单一真源；拒收发生在**任何写语句之前**（先判后写），
        `verified` / `confidence` / `last_verified` 均不被改动。
        边界（如实标注）：①**有限越界值**（如 42.0 / -1.0）仍走 `_finite_clamp`
        钳到 [0,1]（#185 既有口径，本件不动）；② `new_confidence is None`
        （只标 verified、不改进制度）的既有分支不受影响；③坏值拒绝不落
        `action_logs`——本方法无审计配套（与 #206 的 `update_trust_state` 同口径，
        拒绝即回滚、零写入）。
        判据来源：公开 issue #790（「`verify_edge` 的 `new_confidence` 参数应加
        `math.isfinite` 校验，NaN/±inf ⇒ `ValueError`」）＋ #206 `_require_finite`
        先例（数值入参闸从「归一」到「拒绝」的同一升级路径）。

        #234：非 PRIMARY 不得 verify 任一端为共享层的边（旧实现无角色检查）——
        SUB 可自行把结构层间的边置为 `verified=1`、`confidence=1.0`。
        判据来源：公开 issue #234「建议修复 2」＋ `_assert_edge_role`。
        """
        if self.role != Role.PRIMARY:
            edge = self.get_edge(edge_id)
            if edge is not None:
                self._assert_edge_role(edge)
        with self._lock:
            c = self.conn.cursor()
            now = time.time()
            if new_confidence is not None:
                # #790：非有限值先拒（fail-closed，任何写语句之前抛出）；
                # 有限值再走 #185 的 [0,1] 钳制（值域归一口径不动）。
                new_confidence = _require_finite("new_confidence", new_confidence)
                new_confidence = _finite_clamp(new_confidence, 0.0, 1.0)
                c.execute('''
                    UPDATE edges SET verified=1, last_verified=?, confidence=?
                    WHERE id=?
                ''', (now, new_confidence, edge_id))
            else:
                c.execute('''
                    UPDATE edges SET verified=1, last_verified=?
                    WHERE id=?
                ''', (now, edge_id))
            self._commit_locked()

    @_transactional
    def unverify_edge(self, edge_id: str) -> bool:
        """撤销边的已验证标记（D-43：`verified` 需可逆）。

        `verify_edge` 只能把 `verified` 写 1，而 `decay_cycle` 的边衰减段是
        `WHERE e.verified = 0` ⇒ verified=1 的边**永久免衰减**；修前全仓没有
        任何路径能把 `verified` 置回 0（`unverify` / `SET verified=0` 均 0 命中），
        误标（含 #117 那种「一次命中把非参与边一并标 verified」）就此固化、
        无纠错入口。判据来源：`docs/plans/待裁清单_v0.1.md` **D-43**
        「**需要**（全仓现无 `unverify` ⇒ 一旦误标无法回退）」。

        与 `verify_edge` 对称：同样过共享层角色闸（#234）、同样挂
        `@_transactional`、返回 bool（拒/不存在 ⇒ False，与 `delete_node` /
        `restore_node` 契约一致），并落一条 `action_logs`——撤销是治理动作，
        应可审计（对齐 #117 的 fail-closed 留痕取向）。
        """
        if self.role != Role.PRIMARY:
            edge = self.get_edge(edge_id)
            if edge is not None:
                self._assert_edge_role(edge)
        with self._lock:
            c = self.conn.cursor()
            c.execute("SELECT verified FROM edges WHERE id=?", (edge_id,))
            row = c.fetchone()
            if row is None or not row[0]:
                return False          # 不存在或本就未验证 ⇒ 无变更
            c.execute("UPDATE edges SET verified=0, last_verified=NULL WHERE id=?",
                      (edge_id,))
        self.log_action(
            "edge_unverified",
            f"撤销边 {edge_id} 的已验证标记（重新进入衰减面）",
            outcome={"edge_id": edge_id})
        return True

    # ---------- 查询 ----------

    def query_nodes(self, layer: MemoryLayer = None, modality: str = None,
                    min_importance: float = 0.0, limit: int = 50) -> List[STNode]:
        """按层/模态/最低重要性过滤查询节点（全部条件可选，默认前 50 条）。

        #267：逐行容错——单条坏行（坏 JSON / 缺列 / 非法枚举值）只跳过该行，
        不毒化整层装载（旧实现列表推导一处 `STNode.from_row` 抛异常即整层不可读，
        连 `add_perception` 的去重候选查询也一并抛 ⇒ 知识层写不进）。
        如实计数：跳过的行 id 记入 `self.last_skipped_rows`（每次调用刷新）。
        判据来源：分诊表 `triage_lingshu_ALL.md` 行 267「应逐行容错并如实计数」。
        边界（削掉的判别力）：坏行被**跳过**（该节点不出现在结果里），不修复、
        不删除、不阻断读取；坏行仍留在库里（需显式清理或修复）。
        """
        conditions = []
        params = []
        if layer is not None:
            conditions.append("layer=?")
            params.append(layer.value)
        if modality is not None:
            conditions.append("modality=?")
            params.append(modality)
        conditions.append("importance>=?")
        params.append(min_importance)
        where = " AND ".join(conditions) if conditions else "1=1"
        c = self.conn.cursor()
        # #658：`limit=None` ⇒ **不设限、全量返回**（与同族 `query_nodes_by_time`
        # 逐字一致）。旧实现无条件把 `limit` 绑进 `LIMIT ?`——SQLite 的 `LIMIT`
        # **不接受 NULL**（`datatype mismatch`，与 `LIMIT -1`＝无限制不同）⇒
        # 调用方按仓内既有「None＝全量」语义传参即抛 `IntegrityError`。
        # 判据来源：公开 issue #658（同族四函数对同一个 `None` 给出不同结果）。
        sql = f"SELECT * FROM nodes WHERE {where} ORDER BY importance DESC, last_access DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params = params + [limit]
        c.execute(sql, params)
        nodes, skipped = self._tolerant_nodes(c.fetchall())
        self.last_skipped_rows = skipped
        return nodes

    def _tolerant_nodes(self, rows) -> Tuple[List[STNode], List[str]]:
        """逐行反序列化节点（#267）：坏行只跳过该行，返回 (nodes, skipped_ids)。

        `STNode.from_row` 保持严格（低层反序列化器不静默吞错）；容错收在**装载
        面**——`query_nodes` / `get_recent_context` / `search_content` 共用本方法。
        判据来源：分诊表 `triage_lingshu_ALL.md` 行 267。
        """
        nodes, skipped = [], []
        for row in rows:
            try:
                nodes.append(STNode.from_row(tuple(row)))
            except Exception:      # noqa: BLE001 —— 坏行不该毒化整层（#267）
                try:
                    skipped.append(row[0])
                except Exception:  # noqa: BLE001
                    skipped.append(None)
        return nodes, skipped

    def _tolerant_edges(self, rows) -> Tuple[List[STEdge], List[str]]:
        """逐行反序列化边（#653）：坏行只跳过该行，返回 (edges, skipped_ids)。

        节点侧 `_tolerant_nodes`（#267）的边侧同款：`STEdge.from_row` 保持严格
        （未知 relation_type / 非法 condition_space 照抛——低层反序列化器不静默
        吞错，见 #267 K5 对照）；容错收在**装载面**——`get_edge` /
        `get_outgoing_edges` / `get_incoming_edges` / `find_cycles` /
        `decay_cycle` 共用本方法（`infer_causal_paths` 经 `get_outgoing_edges`
        间接受益）。
        判据来源：issue #653（`from_row` 是所有边读取的唯一反序列化点，一条坏边
        让 get_edge/find_cycles/infer_causal_paths/decay_cycle 四路径整体中断）。
        边界（削掉的判别力）：坏行被**跳过**（该边不出现在结果里，不参与衰减/
        删改），不修复、不删除、不阻断读取；坏行仍留在库里。告警口径与
        `_tolerant_nodes` 一致——跳过 id 如实计入返回值、不打 stderr（消费方
        按 skipped_ids 自行上报）。
        """
        edges, skipped = [], []
        for row in rows:
            try:
                edges.append(STEdge.from_row(tuple(row)))
            except Exception:      # noqa: BLE001 —— 坏行不该毒化整批（#653）
                try:
                    skipped.append(row[0])
                except Exception:  # noqa: BLE001
                    skipped.append(None)
        return edges, skipped

    def query_nodes_by_time(self, start_ts: float = None, end_ts: float = None,
                            limit: int = None) -> List[STNode]:
        """按 `temporal_coordinate` 取节点（过滤与排序**都下推 SQL**，时间升序）。

        lingshu #179（P1-2 时间维度）：`what_happened_at` / `session_summary` 此前
        都以 `query_nodes(limit=500)` 取数，而 `query_nodes` 的排序是
        `ORDER BY importance DESC, last_access DESC`（为**检索**服务）⇒ 库内超过
        500 条后，两个时间查询实际只看「最重要的 500 条」，低重要度/较早的事件
        整批被截掉（`get_timeline` 走 SQL 时间过滤故能查到——对照项证明是取数
        方式的问题，非数据缺失）。本方法把时间过滤与时间排序都下推到 SQL，供
        时间维度查询复用；不设 limit（None）时按时间全量返回（`session_summary`
        需全量分窗）。
        判据来源：分诊表 `triage_lingshu_ALL.md` 行 179
        ＋ 同文件既有时间取数先例 `get_nodes_in_range`（SQL 侧 `temporal_coordinate`
        过滤）——本方法是其「排序口径」补齐（按时间而非按重要性）。
        边界（削掉的判别力）：全量调用（limit=None）是 O(N) 全表扫描并逐行反序列化
        ——`session_summary` 本就需全量分窗，代价与既有行为同阶（旧实现也扫表，
        只是多了 500 截断）。
        """
        conds, params = [], []
        if start_ts is not None:
            conds.append("temporal_coordinate >= ?")
            params.append(start_ts)
        if end_ts is not None:
            conds.append("temporal_coordinate <= ?")
            params.append(end_ts)
        where = (" WHERE " + " AND ".join(conds)) if conds else ""
        sql = "SELECT * FROM nodes" + where + " ORDER BY temporal_coordinate ASC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        c = self.conn.cursor()
        c.execute(sql, params)
        nodes, skipped = self._tolerant_nodes(c.fetchall())
        self.last_skipped_rows = skipped
        return nodes

    def get_layer_nodes(self, layer: MemoryLayer) -> List[STNode]:
        """取指定记忆层的全部节点（query_nodes 的层过滤快捷入口）。"""
        # 整层枚举不继承 query_nodes 的默认分页；SQLite LIMIT -1 表示不限条数。
        return self.query_nodes(layer=layer, limit=-1)

    # ---------- v1.16 图架构增强：有界遍历 + 子图嵌套 ----------

    def traverse(self, start_id: str, relation_types: List[str] = None,
                 direction: str = "out", max_depth: int = 5,
                 min_importance: float = 0.0, max_nodes: int = 500) -> List[Dict]:
        """有界图遍历（迭代 BFS，防无界递归爆炸——裸 CTE 全图展开会超时）。
        relation_types: 边类型过滤（None=全部；如 ['causal']）
        direction: out（出边）/ in（入边）/ both
        返回 [{node_id, depth, edge_id, direction, importance}] 按 importance 降序。

        #417：importance 补全的 `WHERE id IN (...)` 改为**分批查询**
        （`_SQL_IN_CHUNK`）。旧实现把所有候选 id 一次性拼进 IN 子句——`max_nodes`
        只在层开始时检查、**不约束单层扩展**：一个高扇出节点（issue 实测 32767 个
        直接叶子）使候选数越过 SQLite 变量上限（本机 3.49.1 实测
        `MAX_VARIABLE_NUMBER=32766`）⇒ `traverse` 直接抛
        `OperationalError: too many SQL variables`；1200 叶子时虽不崩，但单条查询
        已绑定 1200 个变量（返回条数有界、实际工作量无界）。
        判据来源：公开 issue #417（「扩展阶段管理预算，并使用 JOIN 或分批查询消除
        超长绑定列表」）＋ 其复现实读（1200 叶子 `ids_in_importance_query=1200`；
        32767 叶子 `OperationalError`）＋ 分诊表 `triage_lingshu_ALL.md` 行 417。
        仓内无规定该分批大小的理论章节，属经验标定（本件 #417）。
        边界（削掉的判别力，如实标注）：① 本件**只**消除超长绑定列表，**不**给
        单层扩展设上限——issue 明确要求「如果保证按 importance 取前 N，需保留该
        语义，不能简单截取最先遇到的 N 个邻居」，故为选出 importance 前 `max_nodes`
        名必须看全候选；候选集与 `result` 的内存占用仍随扇出线性增长（旧实现同阶）；
        ② `subgraph` 的同款 IN 查询（其候选集来自本函数返回的 ≤max_nodes 条，天然
        有界）不在本件。
        """
        c = self.conn.cursor()
        rel_clause, params = "", []
        if relation_types:
            ph = ",".join("?" * len(relation_types))
            rel_clause = f" AND e.relation_type IN ({ph})"
            params = list(relation_types)
        frontier = [start_id]
        visited = {start_id}
        result = []
        for depth in range(1, max_depth + 1):
            if not frontier or len(visited) > max_nodes:
                break
            nxt = []
            # ★优化（本笔）：旧实现**逐 frontier 节点**发查询（扇出大时每层
            # 1–2×|frontier| 条 SQL）。现把整层 frontier 按 `_SQL_IN_CHUNK`
            # （#417 的 IN 绑定上限单点）分批一次查回，再**按 frontier 原顺序**
            # 分组消费 ⇒ 每层 1–2×ceil(|frontier|/chunk) 条。
            #
            # 行为等价性（如实标注）：
            #   · `visited` 的更新时机与顺序**逐位不变**（仍按 frontier 顺序、
            #     逐行判定）⇒ `result` 的**元素集合与顺序**与旧实现一致；
            #   · 每条 SQL 的**列集与 WHERE 语义**未变（`source_id IN (…)` 是
            #     `source_id=?` 的并集），`rel_clause`/`params` 原样沿用；
            #   · **未加 `ORDER BY`**（旧实现也没有）⇒ 同一源的多条候选边，其
            #     行序仍由 SQLite 的访问路径决定。若某目标节点可经**同一源**的
            #     多条边到达，旧/新选中哪条 `edge_id` 取决于该行序——本件不主张
            #     这一序在两种写法下逐位相同（见 PR 的「未验」节）。
            _by_src: Dict[str, list] = {}
            _by_dst: Dict[str, list] = {}
            for _i in range(0, len(frontier), _SQL_IN_CHUNK):
                _chunk = frontier[_i:_i + _SQL_IN_CHUNK]
                _ph = ",".join("?" * len(_chunk))
                if direction in ("out", "both"):
                    for row in c.execute(
                            f"SELECT e.source_id AS s, e.target_id AS tid, e.id AS eid "
                            f"FROM edges e WHERE e.source_id IN ({_ph}){rel_clause}",
                            _chunk + params).fetchall():
                        _by_src.setdefault(row["s"], []).append(row)
                if direction in ("in", "both"):
                    for row in c.execute(
                            f"SELECT e.target_id AS t, e.source_id AS sid, e.id AS eid "
                            f"FROM edges e WHERE e.target_id IN ({_ph}){rel_clause}",
                            _chunk + params).fetchall():
                        _by_dst.setdefault(row["t"], []).append(row)
            for nid in frontier:
                if direction in ("out", "both"):
                    for row in _by_src.get(nid, ()):
                        if row["tid"] not in visited:
                            visited.add(row["tid"]); nxt.append(row["tid"])
                            result.append({"node_id": row["tid"], "depth": depth,
                                           "edge_id": row["eid"], "direction": "out"})
                if direction in ("in", "both"):
                    for row in _by_dst.get(nid, ()):
                        if row["sid"] not in visited:
                            visited.add(row["sid"]); nxt.append(row["sid"])
                            result.append({"node_id": row["sid"], "depth": depth,
                                           "edge_id": row["eid"], "direction": "in"})
            frontier = nxt
        # importance 补全 + 过滤 + 排序
        if result:
            ids = list({r["node_id"] for r in result})
            # #417：分批查询（单条 IN 的绑定变量数不得超过 SQLite 上限）
            imp = {}
            for i in range(0, len(ids), _SQL_IN_CHUNK):
                chunk = ids[i:i + _SQL_IN_CHUNK]
                ph = ",".join("?" * len(chunk))
                imp.update({row["id"]: row["importance"] for row in
                            c.execute(f"SELECT id, importance FROM nodes "
                                      f"WHERE id IN ({ph})", chunk).fetchall()})
            result = [r for r in result
                      if imp.get(r["node_id"], 0.0) >= min_importance]
            for r in result:
                r["importance"] = imp.get(r["node_id"], 0.0)
            result.sort(key=lambda x: (-x["importance"], x["depth"]))
        return result[:max_nodes]

    def subgraph(self, root_id: str, max_depth: int = 3,
                 relation_types: List[str] = None,
                 direction: str = "out") -> Dict:
        """子图嵌套查询（图架构增强）：根节点 + 子节点递归 + 子图内部边。
        例：历史学（元学科）⊃ 初中历史/高中历史（hierarchical 归属嵌套）。
        direction="out"（默认，根→子）；"in" 用于子指向父的层级边
        （如 dsh 嵌套认知图 child→parent 语义，2026-08-31 接口对接）。
        返回 {root, node_count, edge_count, nodes: {id: {name, importance}}, edges}
        """
        # ① 递归遍历收集子图节点（默认沿归属/因果/相似三类边向指定方向扩展）
        nodes = self.traverse(root_id, relation_types=relation_types or
                              ["hierarchical", "causal", "similar"],
                              direction=direction, max_depth=max_depth)
        node_ids = [root_id] + [r["node_id"] for r in nodes]
        c = self.conn.cursor()
        ph = ",".join("?" * len(node_ids))
        # ② 收集子图内部边（两端都在子图内的边才算）
        edges = []
        for row in c.execute(
                f"SELECT source_id, target_id, relation_type FROM edges "
                f"WHERE source_id IN ({ph}) AND target_id IN ({ph})",
                node_ids + node_ids).fetchall():
            edges.append({"source": row[0], "target": row[1],
                          "relation_type": row[2]})
        # ③ 取节点名与重要度（name 缺失时退化为 id 前 24 字符）
        names = {}
        for row in c.execute(
                f"SELECT id, state_attributes, importance FROM nodes "
                f"WHERE id IN ({ph})", node_ids).fetchall():
            try:
                nm = json.loads(row["state_attributes"] or "{}").get("name", "") \
                    or row["id"][:24]
            except Exception:
                nm = row["id"][:24]
            names[row["id"]] = {"name": nm, "importance": row["importance"]}
        return {"root": root_id, "node_count": len(node_ids),
                "edge_count": len(edges), "nodes": names, "edges": edges}

    def spatiotemporal_query(self, center_node_id: str, time_radius: float = 300.0,
                             space_metric: str = None, space_radius: float = 0.5,
                             max_results: int = 20) -> List[Tuple[STNode, float]]:
        """
        时空联想查询：找时间上邻近 + 空间上相似的节点
        time_radius=0 表示仅匹配同一时刻，时间距离分量为 0。
        返回 (节点, 综合距离) 列表，按距离升序

        #264：候选取数**不再截断**——旧实现 `query_nodes(limit=200)` 的排序是
        `ORDER BY importance DESC, last_access DESC`（为检索服务），库内节点超过
        200 条后，importance 排名 200 之外的低权重记忆**永远不被本函数看到**：
        即使它在时间与空间上都紧邻查询中心，也一律返回空列表，调用方无法区分
        「附近没有相关记忆」与「被截断」。本机实测（合成图）旧实现边界精确落在
        200：总节点=200 找到目标，201 起一律找不到（9/9 与候选集一致）。
        改为全量候选（`limit=-1`，SQLite `LIMIT -1` = 不限条数，与同文件
        `get_layer_nodes` / #258 / #297 同款口径），本机实测总节点 502 时仍能命中。
        判据来源：公开 issue #264（「低 importance 但时空紧邻的记忆永远查不到」）
        ＋ 分诊表 `triage_lingshu_ALL.md` 行 264。
        边界（削掉的判别力/代价，如实标注）：①全量候选使每次调用为 O(N) 全表
        扫描并逐行反序列化，代价随库规模线性增长（旧实现固定 ≤200 条）；②issue
        建议 1 的「几何/时间谓词下推 SQL」**未**采纳——那属取数路径的独立面，
        本件只做「候选视野与 docstring 声明对齐」这一最小修法；③仍不返回
        `truncated` 标记（issue 建议 2），因为本修法后已无截断。
        """
        center = self.get_node(center_node_id)
        if center is None:
            return []
        candidates = self.query_nodes(limit=-1)
        results = []
        for node in candidates:
            if node.id == center_node_id:
                continue
            # 时间距离
            time_dist = abs((node.temporal_coordinate if node.temporal_coordinate is not None else node.created_at or 0)
                            - (center.temporal_coordinate if center.temporal_coordinate is not None else center.created_at or 0))
            if time_dist > time_radius:
                continue
            # 空间距离（如果指定了度量轴）
            space_dist = 0.0
            if space_metric and space_metric in center.spatial_coordinates:
                if space_metric not in node.spatial_coordinates:
                    continue  # P2-001: 缺轴节点不可比，跳过
                space_dist = abs(center.spatial_coordinates[space_metric] - node.spatial_coordinates[space_metric])
                if space_dist > space_radius:
                    continue
            # 综合距离（加权）
            time_component = time_dist / time_radius if time_radius else 0.0
            combined = 0.5 * time_component + 0.5 * (space_dist / max(space_radius, 0.01))
            results.append((node, combined))
        results.sort(key=lambda x: x[1])
        return results[:max_results]

    def get_outgoing_edges(self, node_id: str) -> List[STEdge]:
        """取节点全部出边（source=该节点的有向边）。

        #653：逐行容错——单条坏边（未知 relation_type / 非法 condition_space）
        只跳过该边，不毒化本节点的出边装载。`infer_causal_paths` / `traverse`
        等图算法的取边都经本入口，一条坏边即整体中断的历史（issue #653 读数）
        就此消除。跳过的边 id 记入 `self.last_skipped_edges`（每次调用刷新）。
        """
        c = self.conn.cursor()
        c.execute("SELECT * FROM edges WHERE source_id=?", (node_id,))
        edges, skipped = self._tolerant_edges(c.fetchall())
        self.last_skipped_edges = skipped
        return edges

    def get_incoming_edges(self, node_id: str) -> List[STEdge]:
        """取节点全部入边（target=该节点的有向边）。

        #653：与 `get_outgoing_edges` 同款逐行容错（见其 docstring）。
        """
        c = self.conn.cursor()
        c.execute("SELECT * FROM edges WHERE target_id=?", (node_id,))
        edges, skipped = self._tolerant_edges(c.fetchall())
        self.last_skipped_edges = skipped
        return edges

    # ---------- 因果推理 ----------

    TRUNCATED_EVIDENCE_CAP = 8  # 截断证据条数上界；超出即计数外传，不静默

    def infer_causal_paths(self, start_id: str, end_id: str,
                           max_depth: Optional[int] = 5,
                           relation_types: List[str] = None,
                           max_paths: int = _CAUSAL_MAX_PATHS,
                           causal_stats: Dict = None) -> List[List[STEdge]]:
        """
        传递闭包因果推理：找出从 start 到 end 的所有路径。
        v1.16 图架构增强：relation_types 可多类型（默认 causal；
        传 ['causal','similar','hierarchical'] 可做相似递推/归属层级推理）。

        #327：新增 `max_paths` 上限（默认 `_CAUSAL_MAX_PATHS`）。旧实现无上限：
        同一条因果事实被记录 k 次（`add_edge` 不合并平行边）时路径数是 k⁵——6 节点
        无环链 k=12 即 248832 条 / 3.5s / 669MB，k=24 约 800 万条使长驻进程 OOM
        （issue 实测）。达到上限即停止枚举（结果被截断）。
        #482（本件）：上列上限**只数「命中 end 的结果」** ⇒ 终点不可达时
        `len(all_paths)` 恒 0、上限永不触发，DFS 照样走完 `max_depth` 内的全部简单
        路径（k=12 平行边链：不可达查询 271453 次节点展开 / 2.1s，同图可达查询仅
        94 次）。现改为**按终态分支计数**（`enumerated`：完整路径 / 截断分支 / 死
        胡同 / 回边收口各计 1），预算在不可达分支上同样生效。
        语义边界（如实标注）：① 预算现可能被死胡同/截断分支耗尽 ⇒ 返回的完整路径
        可能少于 `max_paths`（旧实现只在结果数达上限时截断）；② 是否耗尽在
        `causal_stats.budget_exhausted`/`.enumerated` 上外传（不静默）；③ 死胡同仍
        不标 `truncated`（#328 F 组口径不变）；④ issue 建议的「反向可达集剪枝」
        **不在本件**——`max_depth` 偏紧时会误剪仍可达终点的节点，属判别力取舍，
        须设计者裁（复核意见已点名）。
        边界（削掉的判别力，如实标注）：① 达到上限时返回的是**前 max_paths 条**
        （非全量），调用方无法从返回类型区分「就这么多」与「被截断」；② issue
        建议 1（`add_edge` 合并平行边 + 存量库迁移）**不在本件**（会改写入语义）。

        #385：内部 DFS 由**递归**改**显式栈**（同文件 `has_causal_cycle`/Tarjan/
        `traverse` 已是迭代实现）。旧实现递归深度 = 路径长度 = `max_depth`，1500
        节点线性链上 `reason_causal(n0, end_id, max_depth=2000)` 抛 `RecursionError`。
        栈帧携带路径副本，单帧内存与路径长度同阶（与递归版栈帧同阶）。

        #328（issue #145 同族补修 · **传 end_id 的这一半**）——到 max_depth 且**该
        节点仍有出边**时，不再直接丢弃该分支：保留路径并标
        `CausalChain(truncated=True)`，与「确无因果路径」可区分（空列表＝确无因果）。
        语义变化（相对旧实现，如实声明）：
        - 预算内可达 end 的完整路径依旧返回，且仍是列表**前缀**；仅当无完整路径时
          才会返回截断证据；返回列表末项可能不以 end_id 结尾。
        - `max_depth=0` 现返回 1 条截断证据（若 start 有出边）；`max_depth<0` 同 0。
          `max_depth=None` 为**不限深度**。
        - 截断证据条数受 `TRUNCATED_EVIDENCE_CAP` 约束，超出即计数外传（不静默）。
        - `causal_stats`：可选出参，传入 dict 即被填充
          `{max_depth, complete, truncated, truncated_dropped}`。
        """
        types = relation_types or ["causal"]
        type_set = {t.lower() for t in types}
        all_paths: List[CausalChain] = []
        truncated_all: List[CausalChain] = []
        truncated_dropped = 0

        # 显式栈：#385——帧=(当前节点, 到达路径, 到达前的路径节点集, 深度)。
        # `on_path` 等价递归版的 `visited`（沿当前路径的节点集合，回溯时消失）。
        # 子帧按边序**逆序**入栈 ⇒ 出栈顺序与递归版的正向遍历逐字一致。
        # #482：预算从「只数**命中终点的结果**」改为「数**已穷尽的终态分支**」
        # （`enumerated` = 命中 end 的完整路径 ＋ 到深度预算仍有出边的截断分支 ＋
        # 死胡同/仅非因果出边 ＋ 回边收口）。旧实现只判 `len(all_paths) >= max_paths`，
        # 而 `all_paths` **仅在到达 end_id 时增长** ⇒ 终点不可达时该值恒 0、上限永不
        # 触发：DFS 照样走完 `max_depth` 以内的全部简单路径（本机实测 k=12 平行边 6 节点
        # 链：不可达查询 271453 次节点展开 / 2.1s，且随 k 指数上升；同图可达查询仅 94 次）。
        # 本分支的遍历是一棵「帧树」（每帧只入栈一次、弹出一次），**叶子＝终态分支**，
        # 树深 ≤ max_depth+2 ⇒ 帧数 ≤ max_paths×(max_depth+2)：工作量与**图规模无关**，
        # 只随两个预算增长。实测（max_paths=1000、max_depth=5）：k=4/8/12/16 的不可达
        # 查询分别 1334/1145/1094/1070 次节点展开（旧实现 1365/37449/271453/…，随 k
        # 指数上升）；1500 节点长链不可达查询 1500 次（= 链长，远低于预算上界）。
        # 可达分支读数不变（同图 k=12 可达查询仍 94 次展开、1000 条）。
        # 语义边界（如实标注）：预算现在也可能被**死胡同/截断分支**耗尽 ⇒ 此时返回的完整
        # 路径可能**少于** max_paths（旧实现只在结果数达上限时截断）；预算是否耗尽在
        # `causal_stats.budget_exhausted` / `.enumerated` 上显式外传（不静默），且**不**把
        # 死胡同标成 `truncated`（#328 F 组口径：死胡同 ≠ 链太长）。
        enumerated = 0
        stack = [(start_id, [], frozenset(), 0)]
        while stack:
            current_id, path, on_path, depth = stack.pop()
            if enumerated >= max_paths:
                continue
            if current_id == end_id and path:
                all_paths.append(CausalChain(path))
                enumerated += 1
                continue
            if current_id in on_path:
                enumerated += 1
                continue
            if max_depth is not None and depth > max_depth:
                # #328 预算耗尽。**必须先确认本节点仍有出边**（与不传 end_id 分支的
                # 天然终点判定同口径）：无出边者是**天然终点**（死胡同），到 end 确
                # 无路径 ⇒ 不记截断，否则会把「确无因果」误报成「链太长」（假阳性）。
                if path and self.get_outgoing_edges(current_id):
                    if len(truncated_all) < self.TRUNCATED_EVIDENCE_CAP:
                        truncated_all.append(CausalChain(path, truncated=True))
                    else:
                        truncated_dropped += 1
                enumerated += 1
                continue
            next_on = on_path | {current_id}
            edges = self.get_outgoing_edges(current_id)
            pushed = 0
            for edge in reversed(edges):
                if enumerated >= max_paths:
                    break
                if edge.relation_type.value in type_set:
                    stack.append((edge.target_id, path + [edge], next_on, depth + 1))
                    pushed += 1
            if not pushed:
                # 死胡同（无出边，或出边全被 relation_types 滤掉）：该分支穷尽。
                enumerated += 1
        # 完整路径优先：先按长度与平均置信度排序；截断证据次之（同口径排序），
        # 使「已有完整解」时首元素仍是最短完整路径（既有调用方读数不变）。
        all_paths.sort(key=lambda p: (len(p), -sum(e.confidence for e in p)/len(p)))
        truncated_all.sort(key=lambda p: (len(p), -sum(e.confidence for e in p)/len(p)))
        paths = all_paths + truncated_all
        if causal_stats is not None:
            causal_stats.update({
                "max_depth": max_depth,
                "complete": len(all_paths),
                "truncated": len(truncated_all) + truncated_dropped,
                "truncated_dropped": truncated_dropped,
                # #482：预算是否被耗尽（含被死胡同/截断分支耗尽的情形）+ 实际枚举的
                # 终态分支数——终点不可达时上限照样生效，且这一点**可观测**（不静默）。
                "budget_exhausted": enumerated >= max_paths,
                "enumerated": enumerated,
            })
        return paths

    def find_cycles(self, max_depth: int = 10,
                    limit: int = None) -> List[List[STEdge]]:
        """检测因果循环（有向简单环）·issue #145 修复。

        - 覆盖自环与 2 环（旧实现 `len(path) >= 2` 漏报 A⇄B 与 A→A）
        - 每环只报一次：环内最小节点起枚举 + 环边集规范化去重
        - 只在非平凡强连通分量（Tarjan SCC，O(V+E)）内枚举；边索引一次 SQL 建好，
          不再每步一次查询——无环图整趟 O(V+E)，不随规模指数增长
        - max_depth：环长上限（与旧语义一致：至多 max_depth+1 条边）
        - P2-002：CYCLIC 为显式循环标记，纳入检测确保复合循环不遗漏；
          infer_causal_paths 保持仅 CAUSAL

        #252：新增 `limit`（枚举上限）。旧实现**无上限**：稠密 SCC（如 11 节点
        完全有向图）的环数随节点数指数增长——`self_check` 实测 11 节点 13.69s /
        335 万条环 / 峰值 RSS +3041 MB（n=10：2.77s/676MB），而 `self_check` 的
        `cycle_details` 只用 `cycles[:5]`。现 `limit` 达到即提前终止枚举；
        `limit=None`（默认）保持旧语义不变（issue 建议 2：默认调用不破坏 #145 守卫）。
        判据来源：公开 issue #252「建议修复 1」（「`find_cycles` 增加 `limit`……
        `self_check` 取前 5 条明细」）＋ 其复现实读（n=11：13.69s/3355693 条/3 GB）
        ＋ 分诊表 `triage_lingshu_ALL.md` 行 252。仓内无规定该上限的理论章节，属
        经验标定（本件 #252）。
        边界（削掉的判别力，如实标注）：`limit` 命中时**不保证**报出全部环（返回的
        是「先枚举到的 limit 条」）；调用方据返回长度是否等于 limit 判断是否可能被
        截断（`self_check` 另给 `cycles_capped` 出口）。本件**不**改 `max_depth`
        语义（#145/#215 判据不动），也**不**把 `find_cycles` 改成生成器（保留列表
        返回以兼容既有调用）。
        """
        out_edges: Dict[str, List[STEdge]] = {}
        nodes: Set[str] = set()
        # #653：批量读经 `_tolerant_edges`——坏边（如 condition_space 非法 JSON
        # 的 CAUSAL/CYCLIC 行）跳过之，不毒化整个环检测（issue #653 读数：
        # 一条坏边使 find_cycles 整体中断）。未知 relation_type 的坏边本就被
        # 本查询的 `IN (?, ?)` 过滤在结果集外。
        for e in self._tolerant_edges(self.conn.execute(
                "SELECT * FROM edges WHERE relation_type IN (?, ?)",
                (EdgeType.CAUSAL.value, EdgeType.CYCLIC.value)).fetchall())[0]:
            out_edges.setdefault(e.source_id, []).append(e)
            nodes.add(e.source_id)
            nodes.add(e.target_id)
        if not nodes:
            return []
        order = sorted(nodes)  # 全序：每环从其最小节点起枚举一次

        # ---- Tarjan SCC（迭代版；未访问/在栈/已定 三色语义），O(V+E) ----
        index_of: Dict[str, int] = {}
        lowlink: Dict[str, int] = {}
        on_stack: Set[str] = set()
        tarjan_stack: List[str] = []
        scc_of_node: Dict[str, int] = {}
        scc_size: List[int] = []
        counter = 0
        for root in order:
            if root in index_of:
                continue
            work = [(root, iter(out_edges.get(root, ())))]
            index_of[root] = lowlink[root] = counter
            counter += 1
            tarjan_stack.append(root)
            on_stack.add(root)
            while work:
                node, it = work[-1]
                advanced = False
                for e in it:
                    t = e.target_id
                    if t not in index_of:
                        index_of[t] = lowlink[t] = counter
                        counter += 1
                        tarjan_stack.append(t)
                        on_stack.add(t)
                        work.append((t, iter(out_edges.get(t, ()))))
                        advanced = True
                        break
                    if t in on_stack:
                        lowlink[node] = min(lowlink[node], index_of[t])
                if advanced:
                    continue
                work.pop()
                if work:
                    parent = work[-1][0]
                    lowlink[parent] = min(lowlink[parent], lowlink[node])
                if lowlink[node] == index_of[node]:
                    sid = len(scc_size)
                    size = 0
                    while True:
                        w = tarjan_stack.pop()
                        on_stack.discard(w)
                        scc_of_node[w] = sid
                        size += 1
                        if w == node:
                            break
                    scc_size.append(size)

        # 非平凡 SCC：分量 ≥2 节点，或单点带自环——环只落在其中，其余节点整体剪掉
        self_looped = {e.source_id for es in out_edges.values() for e in es
                       if e.source_id == e.target_id}
        interesting = {n for n, sid in scc_of_node.items()
                       if scc_size[sid] >= 2 or n in self_looped}

        cycles: List[List[STEdge]] = []
        seen: Set[frozenset] = set()

        # #385：环枚举 DFS 由**递归**改**显式栈**（帧含节点、环内路径、路径节点集、
        # 深度与**边迭代器**）——旧实现递归深度 = 环长 = `max_depth`，1500 节点单环上
        # `find_cycles(max_depth=2000)` 抛 `RecursionError`（issue 实测；同图迭代实现
        # `has_causal_cycle` 照常返回 True）。用「帧 + 迭代器」以逐字保持递归版的
        # 环输出顺序（同一节点的闭环 append 与其子树枚举交错产生）。
        # 判据来源：公开 issue #385（「把三个 DFS 改成显式栈（与同文件的
        # has_causal_cycle / Tarjan 迭代写法一致）」）＋ 分诊表行 385；仓内无规定该
        # 实现口径的理论章节，属经验标定（本件 #385）。
        # 边界（削掉的判别力，如实标注）：只改遍历实现，`max_depth`（环长上限）、
        # 每环只报一次的去重口径、`limit`（#252）语义一律不动。
        for s in order:
            if s not in interesting:
                continue
            if limit is not None and len(cycles) >= limit:
                break
            # 帧 = [节点, 环内路径, 路径节点集, 深度, 边迭代器(None=未展开)]
            stack = [[s, [], {s}, 0, None]]
            while stack:
                if limit is not None and len(cycles) >= limit:
                    break
                fr = stack[-1]
                cur, path, on_path, depth, it = fr
                if it is None:
                    if depth > max_depth:
                        stack.pop()
                        continue
                    fr[4] = iter(out_edges.get(cur, ()))
                    it = fr[4]
                advanced = False
                for e in it:
                    t = e.target_id
                    if t == s:
                        cyc = list(path) + [e]  # 闭环（path 为空时即自环）
                        key = frozenset(x.id for x in cyc)
                        if key not in seen:
                            seen.add(key)
                            cycles.append(cyc)
                        continue
                    if t in on_path or t <= s or t not in interesting:
                        continue
                    stack.append([t, path + [e], on_path | {t}, depth + 1, None])
                    advanced = True
                    break
                if not advanced:
                    stack.pop()
        return cycles

    def has_causal_cycle(self) -> bool:
        """因果循环存在性探测（三色 DFS：未访问/在栈/已定）·O(V+E)，不做环枚举。

        self_check 只需判有无环——以本件替代整趟 find_cycles 枚举，
        检测耗时不再随图规模指数增长（issue #145）。
        """
        out_edges: Dict[str, List[str]] = {}
        nodes: Set[str] = set()
        for row in self.conn.execute(
                "SELECT source_id, target_id FROM edges WHERE relation_type IN (?, ?)",
                (EdgeType.CAUSAL.value, EdgeType.CYCLIC.value)).fetchall():
            out_edges.setdefault(row[0], []).append(row[1])
            nodes.add(row[0])
            nodes.add(row[1])
        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = {n: WHITE for n in nodes}
        for root in sorted(nodes):
            if color[root] != WHITE:
                continue
            color[root] = GRAY
            work = [(root, iter(out_edges.get(root, ())))]
            while work:
                node, it = work[-1]
                advanced = False
                for t in it:
                    if color[t] == GRAY:
                        return True  # 回边（含自环）⇒ 有环
                    if color[t] == WHITE:
                        color[t] = GRAY
                        work.append((t, iter(out_edges.get(t, ()))))
                        advanced = True
                        break
                if not advanced:
                    color[node] = BLACK
                    work.pop()
        return False

    # ---------- 衰减引擎 ----------

    def _decay_dt(self, now) -> float:
        """本轮衰减的**流逝时间**（秒，≥0），并持久化「上次衰减时刻」（#176）。

        `now is None` ⇒ 返回一个周期（`_DECAY_PERIOD_SECONDS`），即**保持旧的
        「每次调用 = 一个周期」步进语义**（既有测试/调用方一律走此路径，行为逐字
        不变）。`now` 给出（生产路径 `start_auto_decay` 经注入时钟给出）⇒ 返回
        `now - 上次衰减时刻`，并把本次时刻写回 `engine_meta`——于是**遗忘量只取决于
        流逝时间**：`interval=1` 与 `interval=60` 在相同墙钟时间里遗忘量相等，
        停机一周后重启的首次衰减一次性补足该周。
        判据来源：公开 issue #176 建议 1（`dt = now - last_decay_at`）＋
        `lingshu/core/time_core.py` 章程（连续核 `cred` 以真实时间 dt 为自变量）。
        边界（削掉的判别力，如实标注）：首次进入时间模式（无 `_last_decay_at`
        记录）时按**一个周期**计 dt，而非 0——避免「刚部署/刚重启首轮不衰减」；
        取此口径与 `start_auto_decay` 的周期语义一致。

        #176 追评（时钟回拨把游标写回更小的 `now`）：旧实现无条件
        `set_meta(key, str(now))`——回拨那一轮 `dt` 虽被钳成 0（不反向增信），
        **游标却倒退了**；时钟恢复后下一轮 `now - last` 把回拨区间**再算一遍**
        ⇒ 一次 1 小时回拨让恢复后多遗忘约 1 小时（情境层记忆被提前删除、
        知识层 confidence 额外回落，无日志）。修法（issue 建议 1，最小）：
        写回**单调不降**的游标 `max(last_f, now)`（无游标时仍写 `now`）——
        回拨轮游标留在原处、`dt` 仍钳 0，恢复轮 `dt = now - last_f` 只计
        真实流逝。判据来源：公开 issue #176 追评（`max(now, last_f)`）＋
        本端独立复跑读数（对照 `T0→T0+60→T0+120` 与回拨 `T0→T0-3540→T0+120`
        在相同真实流逝下 importance 相等）。
        边界（削掉的判别力，如实标注）：本件**只**让游标不倒退，**不**改
        `dt` 的钳零语义（回拨区间不计入衰减，也不反向增信）；issue 建议 2
        （进程内周期调度改用 `time.monotonic()`）不在本件范围——本函数是
        「墙钟记账」面，`monotonic` 属调度面，属另一个语义变更。
        """
        if now is None:
            return self._DECAY_PERIOD_SECONDS
        last = self.get_meta(self._LAST_DECAY_META_KEY).get(self._LAST_DECAY_META_KEY)
        try:
            last_f = float(last)
        except (TypeError, ValueError):
            last_f = None
        dt = self._DECAY_PERIOD_SECONDS if last_f is None else (now - last_f)
        if dt < 0:
            dt = 0.0          # 时钟回拨：不前进（不反向增信）
        # #176 追评：游标**单调不降**——回拨时不得把更小的 `now` 写回，否则时钟
        # 恢复后 `now - last` 会把回拨区间重复计入（情境层记忆被提前删除）。
        self.set_meta(self._LAST_DECAY_META_KEY,
                      str(now if last_f is None else max(last_f, now)))
        return dt

    def _decay_value(self, x: float, factor: float, dt: float) -> float:
        """连续核衰减一步：`cred(x, 0, γ, dt)`，γ = -ln(1-factor)/周期（#176）。

        与旧离散核 `cred_step(x, factor) = x·(1-factor)` 的关系：取 `dt = 一个周期`
        时 `cred(x,0,γ,周期) = x·e^(-γ·周期) = x·(1-factor)`——**逐字相同**，故
        未提供 `now` 的既有调用行为不变；`dt` 为任意值时按连续核演化，从而
        「n 次小步」与「一次大步」在相同总 Δt 下结果相等（核形状唯一）。
        判据来源：`lingshu/core/time_core.py` 章程「X(t)=X∞+(X0−X∞)·e^(−γ·t)」＋
        `cred_step` docstring「γ = -ln(1-decay) 每周期」。
        """
        if x is None:
            return 0.0
        ratio = 1.0 - factor
        if ratio <= 0.0:
            return 0.0
        gamma = -math.log(ratio) / self._DECAY_PERIOD_SECONDS
        return cred(x, 0.0, gamma, dt)

    @_transactional
    def decay_cycle(self, factor: float = 0.02, min_confidence: float = 0.1, now=None):
        """
        对未验证的边执行指数衰减。
        不可变层（锚点/结构/自我）的节点不受影响（其关联边也不衰减）。
        #168 澄清：上面这句只对**本函数的置信度衰减**成立（下方边衰减查询以
        `sn/tn.layer NOT IN (<IMMUTABLE_LAYERS>)` 排除不可变层端点）。不可变层
        节点的关联边仍会随**对端节点**在情境层自然遗忘/FIFO 淘汰中被整条删除
        （`DELETE FROM edges`）——旧实现对此**无日志无审计**；现该删除经
        `_strip_incident_edges` 落 `immutable_edge_stripped` 审计（见该助手）。
        v1.16：短期记忆自动减少权重——CONTEXT（情境层）节点 importance
        指数衰减（设计者设计：短期记忆随时间淡出，长期/知识层保留）。
        #29③：KNOWLEDGE 层**置信度**（非内容）随时间回落——未确证断言不得
        因重复写入而永久为真；已确证（confidence=1.0）与受保护节点不衰减。

        #781（口径声明 · **只声明现状、不改语义**）：`archived` 与 `no_forget` 在
        两条遗忘路径上**不等价**。本函数的情境层删除段经 `_is_forget_protected`
        判定，只认 `no_forget` 标签 ∪ `protections` 行登记，**不认 `archived`**
        ⇒ 仅带 `archived` 的情境节点照样被本路径删除（实测：仅 `archived` 的节点
        经 200 轮 `decay_cycle` 已消失，同批 `no_forget` 者仍在）；而
        `SpacetimeMemoryEngine.forget_advisor` 的跳过条件是
        `"no_forget" in tags or "archived" in tags` ⇒ **认 `archived`**（仅带
        `archived` 者被跳过、不归档也不删）。故「归档即保护」的直觉在两条路径上
        结论相反。本件**不**统一两者（是否让 `decay_cycle` 也认 `archived` 属设计
        面裁定），仅把现状显式落到注释，防日后无声漂移。判据来源：公开 issue #781
        第 4 点 ＋ 本端独立读数（`tests/test_issue778_781_forget_protection_guard.py`
        的 G6 钉住该分叉）。

        #115：`OPPOSITE`（冲突）边**免疫普通衰减**。冲突边是 `register_conflict`
        建的「矛盾记忆」标记（confidence=0.3、verified=False），旧实现与普通未验证
        边同列衰减（`WHERE e.verified = 0`）⇒ 约 55 轮后 confidence<0.1 被 DELETE，
        冲突关系**静默消失**，而双方节点上的 `conflict` 标签永久残留（生命周期断裂）。
        判据来源：`docs/plans/待裁清单_v0.1.md` **D-38**（「免疫普通衰减，只由结案/删除
        移除」）＋ **B-4 / D-37**（标签随边消失而移除，见 `delete_node`）。
        削掉的判别力：冲突边不再自然淡出——须经显式删除才移除（本件只做免疫与
        删除期标签清理，**不**新增结案 API）。

        #205：边衰减的不可变层排除此前**硬编码** `('anchor','structure')`，
        漏掉 `IMMUTABLE_LAYERS` 里的 SELF ⇒ `update_self` 建的
        「自我快照 → 经历」因果边（`add_edge(..., verified=False)`）约 114 轮
        后被 DELETE，而两端节点都还在（SELF 层 `delete_node` 拒删）——来源关系
        静默剥光。修法：排除集取自 `IMMUTABLE_LAYERS`（单一真源）。
        判据来源：`IMMUTABLE_LAYERS` 定义处注释（core.py:723-724）＋ 分诊表
        `triage_lingshu_core_front.md` 行 205（经验标定，理论章节追不到）。
        边界（削掉的判别力）：本件只**保留**不可变层端点的边不被衰减删除；
        这些边仍随**对端节点**在情境层自然遗忘/FIFO 淘汰中被整条删除
        （经 `_strip_incident_edges` 落 `immutable_edge_stripped` 审计，见 #168）。

        #176（遗忘按「调用次数」而非流逝时间）：旧实现每调用一次就乘一次
        `(1-factor)`，与真实时间无关 ⇒ `start_auto_decay(interval=1)` 与
        `interval=60` 在相同墙钟时间里遗忘量相差 60 倍、手动多调几次等于时间快进、
        进程停机一周重启后**零遗忘**。现按 `now`（注入时钟给出的墙钟时刻）取
        `dt = now - 上次衰减时刻`，用**连续核** `cred(x,0,γ,dt)`（γ = -ln(1-factor)
        每周期）演化——「相同 Δt 下调用 1 次与调用 60 次结果相等」。
        `now=None`（既有测试/直接调用）**保持旧步进语义**（每次 = 一个周期），
        故现有调用方行为逐字不变；时间模式由 `start_auto_decay`（生产路径）经
        `_decay_uses_clock` 自动进入。判据来源：公开 issue #176 建议 1+2 ＋
        `lingshu/core/time_core.py` 章程（核形状唯一，见 `_decay_value`）。
        边界（削掉的判别力，如实标注）：①**副本合一未做**——issue 建议 3
        （删除 `world/time_core.py` 与 `gen/hex_composite.py` 的副本）属架构改动，
        不在本件范围，`lingshu/core/time_core.py` 与 `lingshu/world/time_core.py`
        仍是两份字节相同的副本；②`now=None` 时不做时间记账（既有测试语义优先）；
        ③仅把「周期」钉为 `_DECAY_PERIOD_SECONDS`（默认 60s，与
        `start_auto_decay` 默认 interval 同源），未把 γ 拆成独立配置项。
        """
        dt = self._decay_dt(now)
        # #720：本轮逐行容错计数（写路径也刷新一次；语义＝「本次调用跳过的坏行 id」，
        # 与装载面 `_tolerant_nodes` 的刷新口径同名同义）。下方情境段/知识段遇到
        # 坏 `tags` 行时跳过该行并把 id 记入本表，**不再让整轮遗忘循环停摆**。
        self.last_skipped_rows = []
        with self._lock:
            c = self.conn.cursor()
            # 获取所有未验证的边
            protected = self.get_protected_nodes()
            protected_set = set(protected)
            # #257：保护名单**不再展开成 SQL 占位符**。旧实现
            #   `ph = ",".join("?" for _ in protected)` +
            #   `exclude = " AND e.source_id NOT IN (ph) AND e.target_id NOT IN (ph)"` +
            #   `params = tuple(protected) * 2`
            # ⇒ 一轮衰减的绑定变量数 = **2 × 名单长度**；名单越过「变量上限/2」
            # （SQLite 上游默认 32766 ⇒ 16383 条）后，本查询第一句即抛
            # `sqlite3.OperationalError: too many SQL variables`，**整个
            # `decay_cycle` 中止**——边衰减、情境层衰减/删除、`run_maintenance_cycle`
            # 第一步全部停摆，且 `start_auto_decay` 线程首轮即退出，无自愈路径。
            # 而保护名单在设计上只增不减（全仓无 `DELETE FROM protections`）。
            # 修法（最小 · 按 issue 建议 1）：改 SQL **内联子查询** ⇒ 绑定变量数与
            # 名单长度无关，并可用 `protections.node_id` 的主键索引。
            # 判据来源：公开 issue #257「建议修复 1：不要把名单展开成占位符：改为
            # SQL 内联子查询……或用 NOT EXISTS」＋ 其给出的一次性复现读数
            # （保护 16384 条 ⇒ `too many SQL variables`）。仓内无规定该口径的理论
            # 章节，属经验标定（本件 #257）。
            # 边界（削掉的判别力，如实标注）：本件**只**拆掉占位符悬崖——issue 建议
            # 2/3/4（保护行随节点删除、`protect_node` 校验节点存在、孤儿行回收）属
            # 保护名单**回收**面，不在本件范围（名单仍可无限增长，但不再触发悬崖）。
            # `protected` 列表仍取一次供 `protected_set`（`_is_forget_protected` 的
            # 情境层删除段使用），该段本就是 Python 侧集合判定、不展开占位符。
            exclude = (" AND e.source_id NOT IN (SELECT node_id FROM protections)"
                       " AND e.target_id NOT IN (SELECT node_id FROM protections)")
            params = tuple()
            # #205：不可变层端点排除改用 `IMMUTABLE_LAYERS`（单一真源），不再
            # 硬编码 `('anchor','structure')`——硬编码漏掉 SELF 层，使
            # 「SELF 节点 → 经历」的因果边被常规衰减删除，而两端节点都还在
            # （`delete_node` 拒删 SELF）⇒ 自我与经历的来源关系被**静默剥光**。
            # 判据来源：`IMMUTABLE_LAYERS` 的定义处注释（core.py:723-724，
            # 「SELF 层=自我锚点（扮演依据）不可遗忘」）＋ 本函数 docstring
            # 「锚点层和结构层的节点不受影响（其关联边也不衰减）」的同款语义。
            _imm = sorted(lyr.value for lyr in self.IMMUTABLE_LAYERS)
            _imm_ph = ",".join("?" for _ in _imm)
            c.execute(f'''
                SELECT e.* FROM edges e
                JOIN nodes sn ON e.source_id = sn.id
                JOIN nodes tn ON e.target_id = tn.id
                WHERE e.verified = 0
                  AND e.relation_type != ?
                  AND sn.layer NOT IN ({_imm_ph})
                  AND tn.layer NOT IN ({_imm_ph})
                  {exclude}
            ''', (EdgeType.OPPOSITE.value,) + tuple(_imm) * 2 + params)
            rows = c.fetchall()
            # #653：批量读经 `_tolerant_edges`——单条坏边（未知 relation_type /
            # 非法 condition_space）跳过之（该边不衰减、不删改、留在库里），
            # 不再让**整轮衰减**随一条坏边中断（issue #653 读数；对照 #257 修的
            # 「一轮衰减整体停摆」同症不同因——彼处是 SQL 变量悬崖）。逐行调用
            # 是刻意的：衰减面还要读**原始行**的 NULL 位（见下），批量装袋会丢。
            for row in rows:
                got, _bad = self._tolerant_edges([row])
                if not got:
                    continue          # #653：坏边跳过（不衰减、不删改、留在库里）
                edge = got[0]
                # #185：存量坏行兜底——修复前的库可能已含 `confidence` 为 NULL 的边
                # （NaN 经旧路径落库），`cred_step(None, factor)` 会抛 TypeError 使
                # **整轮衰减中断**（并可致后台线程退出，见 #184）。此处把非有限/
                # NULL 视为「置信度已耗尽」→ 0.0（低于 `min_confidence` ⇒ 走删除），
                # 即 #185 判据「decay_cycle 在任何已入库的数据上都不会抛异常」。
                # #653 起 `from_row` 把 NULL 落缺省 0.5（图算法读取面，见其
                # docstring）——衰减面的 #185 原判据（NULL=已耗尽→删除）不随之
                # 改变，NULL 位改读原始行 `row[5]`；NaN 等非有限值仍在
                # `STEdge.confidence` 上判（from_row 原样落 NaN）。
                conf = edge.confidence
                if row[5] is None or not math.isfinite(conf):
                    conf = 0.0
                # #176：连续核按流逝时间 dt 演化（dt=周期时 == 旧 cred_step）。
                new_conf = self._decay_value(conf, factor, dt)
                if new_conf < min_confidence:
                    # 低于最小置信度，删除边
                    c.execute("DELETE FROM edges WHERE id=?", (edge.id,))
                else:
                    c.execute("UPDATE edges SET confidence=? WHERE id=?", (new_conf, edge.id))
            # 短期记忆自动减少权重（v1.16）：CONTEXT 情境层节点 importance 指数衰减。
            # 不可变层（锚点/结构/自我）不可遗忘（上面已排除）；CONTEXT 层
            # （短期/情境记忆）随时间淡出——低于阈值删除（自然遗忘）。
            # #207：本段此前只选 `importance > min_confidence` 的节点 ⇒ 已低于
            # 阈值的反而永生（越不重要越长寿）；现对**全部**情境节点执行衰减与判删。
            # #36-①：删除前查保护（本段此前就地内联 DELETE，既不看 `protections`
            # 行登记也不看 `no_forget` 标签 ⇒ 受保护件照样被删；同函数上半段对边
            # 却已排除保护——同函数内上下两段判据不一致）。现与 `delete_node` 共用
            # 单一谓词 `_is_forget_protected`。
            # #207：本段 SQL 此前带 `AND importance > ?`（? = min_confidence）⇒
            # importance **已 ≤ 阈值**的情境节点根本不被选出，既不衰减也不删除
            # ——「越不重要越长寿」，与本段注释「低于阈值删除（自然遗忘）」相反
            # （importance=0.05/0.0 的节点永生；0.11 的反被遗忘）。现选出**全部**
            # 情境节点，先按 cred_step 衰减、再以 min_confidence 判删。
            # 判据来源：公开 issue #207（「去掉 SQL 里的 `AND importance > ?`，
            # 对全部情境节点执行 new_imp = cred_step(imp, factor)，new_imp <
            # min_confidence 即删除」）＋ 本段既有注释「CONTEXT 层随时间淡出——
            # 低于阈值删除（自然遗忘）」＋ 分诊表 `triage_lingshu_core_front.md`
            # 行 207（经验标定，理论章节追不到）。
            # 边界（削掉的判别力）：阈值本体**未改**（仍是 `min_confidence`）；
            # 受保护件仍经 `_is_forget_protected` 跳过（#36 不变）。判别力变化只在
            # 「≤ 阈值的低价值情境节点现在会进入遗忘面」——这正是本件要修的死区。
            c.execute("SELECT id, importance, tags FROM nodes WHERE layer='context'")
            # #720：本轮逐行容错跳过的坏行 id（两段共用，末尾一次性如实上报）。
            _bad_rows: List[str] = []
            for nid, imp, _tags in c.fetchall():
                # ★优化（本笔）：旧实现传 `self.get_node(nid)` —— 那是
                # `SELECT * FROM nodes WHERE id=?`（读**整行** + 建 `STNode`），
                # 每节点一次；而本谓词只用 `tags` 与 `id` 两字段。现由本查询带出
                # `tags` 并走 `_is_forget_protected` 的直传入口（同一段判定体）。
                # #720：`tags` 解析改为**逐行容错**——坏 tags（非法 JSON）此前经
                # `STNode.from_row` 的 `json.loads` 抛 `JSONDecodeError`，**整轮
                # 遗忘循环停摆**（本事务回滚、importance 零进展、后台衰减线程退出）。
                # #267 的容错只收在批量装载面 `_tolerant_nodes`，本处是单点裸解析
                # 的漏网调用点。口径与 `_tolerant_nodes` 逐字同款：**跳过该行**
                # （不衰减、不删除、原文留在库里——不去猜它的内容），id 如实计入
                # `last_skipped_rows`。取「跳过」而非「按未受保护继续衰减」：坏行
                # 的 `no_forget` 位不可知，fail-safe 侧是不动它。
                try:
                    _n_tags = json.loads(_tags) if _tags else []
                except (TypeError, ValueError):
                    _bad_rows.append(nid)
                    continue
                if self._is_forget_protected(None, protected_set,
                                             node_id=nid,
                                             tags=_n_tags):
                    continue  # 不可遗忘（3.2 节）：受保护件不因衰减路径消失
                # #185：存量坏行兜底——`importance` 为 NULL/非有限时
                # `cred_step(None, factor)` 抛 TypeError 会**整轮衰减中断**
                # （并可致后台线程退出，见 #184）；与同函数边衰减段同款，
                # 视为「已耗尽」→ 0.0（低于阈值 ⇒ 走删除）。
                if imp is None or not math.isfinite(imp):
                    imp = 0.0
                # #176：连续核按流逝时间 dt 演化（dt=周期时 == 旧 cred_step）。
                new_imp = self._decay_value(imp, factor, dt)
                if new_imp < min_confidence:
                    # #93：先删边再删节点（外键开启后顺序反了会 FK constraint failed）。
                    # #168：删边经唯一真源（对不可变层对端的边先落审计）
                    self._strip_incident_edges(c, nid)
                    c.execute("DELETE FROM nodes WHERE id=?", (nid,))
                    # #448：硬删同步清归档区登记（见 delete_node 同款注释）。
                    c.execute("DELETE FROM archive_state WHERE node_id=?", (nid,))
                else:
                    c.execute("UPDATE nodes SET importance=? WHERE id=?", (new_imp, nid))
            # #29③：KNOWLEDGE 层**置信度**衰减（长期知识不淡出内容，但未确证的断言
            # 须随时间回落）。旧实现 `nodes.confidence` 不出现在任何 UPDATE——知识层
            # 一旦被 M5 重复增信封顶即永久为真（「重复即真理」）。
            # 判据来源：经验标定（#29 修复轮；**理论章节追不到**，维护者核验件
            # `.tmp/wave_verify_mem.md` §一 环节 3 实测「confidence 不出现在任何
            # UPDATE」）。衰减率沿用本函数同一离散指数核 cred_step（核形状唯一，
            # 见 lingshu/core/time_core.py）。
            # 边界：① 锚点/结构/自我层不衰减（共享层不可遗忘，且 SELF 为自我锚点）；
            # ② 受保护节点（protections 行登记 ∪ no_forget 标签）不衰减；
            # ③ 置信度已达上限（1.0）者视为已确证，不衰减；④ 只降 confidence，
            # **不删节点**（知识层不适用情境层的自然遗忘删除）。
            c.execute('''
                SELECT id, confidence, tags FROM nodes
                WHERE layer='knowledge' AND confidence > ? AND confidence < 1.0
            ''', (min_confidence,))
            for nid, conf, _tags in c.fetchall():
                # ★优化（本笔）：与上一段同款——`tags` 由本查询带出，不再逐节点
                # `get_node`（读整行 + 建 STNode）。
                # #720：与情境段同款逐行容错——坏 `tags` 此前抛 `JSONDecodeError`
                # 令**整轮衰减中断**（本段同样走裸 `json.loads`）。口径逐字同款：
                # 跳过该行（本段只降 confidence，跳过＝该知识节点本轮不衰减）、
                # id 如实计入 `last_skipped_rows`，不中断整轮。
                try:
                    _k_tags = json.loads(_tags) if _tags else []
                except (TypeError, ValueError):
                    _bad_rows.append(nid)
                    continue
                if self._is_forget_protected(None, protected_set,
                                             node_id=nid,
                                             tags=_k_tags):
                    continue
                c.execute("UPDATE nodes SET confidence=? WHERE id=?",
                          (self._decay_value(conf, factor, dt), nid))
            if _bad_rows:
                # #720：与 `_strip_incident_edges` 本轮已记的坏 layer 行**合并**
                # （同一轮内删边助手也往 `last_skipped_rows` 记账）——不覆盖，避免漏报。
                self.last_skipped_rows = list(dict.fromkeys(
                    list(self.last_skipped_rows or []) + _bad_rows))
            self._commit_locked()

    # ==================== 检索层（M1） ====================

    @staticmethod
    def char_bigram_jaccard(a: str, b: str) -> float:
        """中文二元组 Jaccard 相似度（对中文友好，解决按词分割失效问题）"""
        if not a or not b:
            return 0.0
        def bigrams(s):
            s = "".join(s.split())
            if len(s) <= 1:
                return {s}
            return {s[i:i + 2] for i in range(len(s) - 1)}
        A, B = bigrams(a), bigrams(b)
        if not A or not B:
            return 0.0
        return len(A & B) / len(A | B)

    # ---- 检索增强（v1.14）：同义词扩展（词汇鸿沟缓解） ----
    # 用户查询与存储文本词面差异大时（图像 vs 视觉）召回失败 → 同义词组展开。
    SYNONYM_GROUPS = [
        {"视觉", "图像", "画面", "图片", "影像", "视像"},
        {"语义", "含义", "意思", "意义", "概念"},
        {"识别", "检测", "感知", "探测", "发现"},
        {"转换", "转化", "映射", "变换"},
        {"实验", "试验", "集成", "实现", "验证", "测试"},
        {"评测", "评估", "跑分", "基准", "benchmark", "评审", "考核"},
        {"记忆", "记录", "库"},
        {"智能", "智能体", "灵枢", "AI"},
        {"语音", "声音", "音频", "说话"},
        {"对话", "聊天", "交流"},
    ]

    @staticmethod
    def _synonym_trigger_hit(word: str, query: str) -> bool:
        """同义词触发判据（#132 · D-44「词边界」）。

        旧实现 `if w in query` 是**纯子串**触发 ⇒ `AI` 命中 `EMAIL`（D-44 点名的
        病根），泛词（`AI`/`库`）把无关候选灌进 LIKE 预筛。现对 **ASCII/数字词**
        要求**词边界**命中（前后不得紧邻 ASCII 字母/数字/下划线）；中文词仍按子串
        （中文无空格词界，本件不引入分词器——见下「削掉的判别力」）。
        判据来源：`docs/plans/待裁清单_v0.1.md` **D-44 / B-16**（触发口径 = 词边界）。
        削掉的判别力：中文触发口径**未改**（`记忆库` 仍由 `记忆`/`库` 子串触发）——
        本件只收口 ASCII 词内误命中，不引入中文分词。
        """
        if not word:
            return False
        if word.isascii() and any(ch.isalnum() or ch == "_" for ch in word):
            start = query.find(word)
            while start != -1:
                end = start + len(word)
                before = query[start - 1] if start > 0 else ""
                after = query[end] if end < len(query) else ""
                b_ok = not (before.isascii() and (before.isalnum() or before == "_"))
                a_ok = not (after.isascii() and (after.isalnum() or after == "_"))
                if b_ok and a_ok:
                    return True
                start = query.find(word, start + 1)
            return False
        return word in query

    @staticmethod
    def expand_query_terms(query: str) -> list:
        """查询同义词展开：原查询整句 + 命中组词（含命中词本身，词级预筛）+ 组内同义词。
        命中词本身必须加入——整句 LIKE 无法命中只含词级的节点。
        触发判据见 `_synonym_trigger_hit`（#132：ASCII 词边界，杜绝 EMAIL→AI）。"""
        terms = [query]
        for group in LayeredStore.SYNONYM_GROUPS:
            for w in group:
                if LayeredStore._synonym_trigger_hit(w, query):
                    terms.extend(g for g in group if g not in terms)
                    break
        return list(dict.fromkeys(terms))

    @staticmethod
    def _like_fragments(term: str) -> list:
        """把超长检索词切成若干短片段（lingshu #266）；短词原样返回 `[term]`。

        为什么必须切：`search_content` 的 LIKE 预筛把 `%{term}%` 作为**一条**模式绑定，
        而 SQLite 的 `SQLITE_MAX_LIKE_PATTERN_LENGTH`（上游默认 50000 字节）在模式
        超限时让 `execute` 自身抛 `OperationalError: LIKE or GLOB pattern too complex`
        ——预筛之后的「LIKE 落空 → 全表二元组」兜底根本走不到（实测 20000 汉字查询
        在 `recall` / `search_content` / `semantic_search` 三入口同抛）。
        切法：按**字符**累加成连续片段（不切多字节字符），每片段编码后 ≤
        `_LIKE_PATTERN_MAX_BYTES`；片段数上限 `_LIKE_FRAGMENT_CAP`，去重。
        判据来源：公开 issue #266 正文与补充帖（「把整句模式截断为前若干个短片段」）。
        边界（削掉的判别力，如实标注）：①超过片段上限的词尾**不参与预筛**（仍由
        全表二元组兜底参与打分，故召回面不缩、只是可能多付一次兜底扫描）；
        ②片段是**连续**子串，不做重叠滑窗——跨片段边界的连续命中不再进预筛（同上，
        由兜底覆盖）；③本件**不**改预筛落空后的兜底口径（`LIMIT 300/500` + 定序，
        见 `search_content` docstring）。
        """
        if len(term.encode("utf-8")) + 2 <= _LIKE_PATTERN_MAX_BYTES:
            return [term]
        frags, cur, size = [], [], 0
        for ch in term:
            cl = len(ch.encode("utf-8"))
            if cur and size + cl + 2 > _LIKE_PATTERN_MAX_BYTES:
                frags.append("".join(cur))
                cur, size = [], 0
                if len(frags) >= _LIKE_FRAGMENT_CAP:
                    break
            cur.append(ch)
            size += cl
        if cur and len(frags) < _LIKE_FRAGMENT_CAP:
            frags.append("".join(cur))
        return list(dict.fromkeys(frags))

    def search_content(self, query: str, layers: List[MemoryLayer] = None,
                       limit: int = 20, touch: bool = None) -> List[Tuple[STNode, float]]:
        """内容检索：多词 OR 预筛（含同义词扩展）+ 原查询二元组 Jaccard 相似度排序

        #787：`touch` 形参（缺省 None → `_retrieval_touches_recency()` 开关，既有
        调用行为逐位不变）。显式传 False 的唯一调用方是 `recall`——它的打分公式
        （0.5·sim + 0.3·importance + 0.2·recency）读的正是 `last_access`，而本方法
        返回前会把命中节点的 `last_access` 刷成「现在」⇒ 同一节点的分数在两次调用
        间漂移（issue 实测 ~1.9e-08），主排序键不稳定 ⇒ 排序翻转。打分路径不写回
        打分输入（检索只计数、不刷新近因），分数即成为固定输入的纯函数。"""
        q = query.strip()
        if not q:
            return []
        terms = self.expand_query_terms(q)
        if getattr(self, "_fts", None) is not None:
            # #201：与 LIKE 路径同口径——未指定层时默认排除自我层
            fts_layers = layers or [l for l in MemoryLayer if l != MemoryLayer.SELF]
            fts_hits = self._search_content_fts(q, terms, fts_layers, limit, touch=touch)
            if fts_hits is not None:
                return fts_hits
        conds, params = [], []
        if layers:
            ph = ",".join("?" for _ in layers)
            conds.append(f"layer IN ({ph})")
            params.extend([l.value for l in layers])
        else:
            # #201：默认召回**排除自我层**——自我快照（每次 update_self 新建一条
            # importance=0.9/confidence=1.0）以普通记忆身份参与检索时会挤占召回
            # （实测 20 条快照后前 10 命中里 9 条是快照）。需要检索自我层时显式传
            # layers=[MemoryLayer.SELF]。
            conds.append("layer != ?")
            params.append(MemoryLayer.SELF.value)
        # 多词 OR 预筛（原查询 + 同义词任一命中即召回，提高召回率）
        # #266：每个检索词先经 `_like_fragments` 切分——整段查询原文（`terms[0]`）
        # 一过 49998 字节，`%{q}%` 这一条模式就撞 SQLite 的
        # `SQLITE_MAX_LIKE_PATTERN_LENGTH`（默认 50000），本 `execute` 自身抛
        # `OperationalError: LIKE or GLOB pattern too complex`，兜底扫描走不到。
        # 短词原样返回 `[term]`（短查询行为逐字不变），只有超长词才变多条短模式。
        like_parts = []
        like_params = []
        for t in terms:
            for frag in self._like_fragments(t):
                # #183：预筛的 `%{frag}%` 此前原样拼入 `LIKE ?`——查询里的
                # `%` / `_` 被当通配符 ⇒ `search_content("%")` / `search_content("_")`
                # 返回整库（且 `_` 任意单字符误配）。现经 `_escape_like` 转义并加
                # `ESCAPE '\'`，使这两个字符只匹配真正含它的内容。
                like_parts.append("content LIKE ? ESCAPE '\\' OR tags LIKE ? ESCAPE '\\'")
                esc = f"%{self._escape_like(frag)}%"
                like_params.extend([esc, esc])
        like_cond = "(" + " OR ".join(like_parts) + ")"
        where = " AND ".join(conds + [like_cond]) if conds else like_cond
        # #35：预筛候选池必须**显式定序**再截断。旧实现两处 `LIMIT` 均无 `ORDER BY`
        # ⇒ 实际按 rowid（插入序）取前 N 行，池满后**后写入的命中项整批不可达**
        # （实测 351 条同词记忆中，第 351 条高重要度目标不在前 300 行内、检索不可见；
        # 回退腿同理 500 行封顶）。定序口径沿用本仓既有有界候选池先例
        # `query_nodes`（core.py 同文件 `ORDER BY importance DESC, last_access DESC
        # LIMIT ?`）：截断时保留**高重要度、较近访问**的候选。
        # 判据来源：经验标定（本件 #35；仓内无规定检索候选池排序的理论章节，追不到
        # 更早出处）＋复用 `query_nodes` 既有定序手法（同文件同表）。
        # 削掉的判别力：候选池仍是有界（300/500 行）——超限时低重要度命中项**依旧
        # 不参与打分**（本件只把「按插入序静默丢新项」改成「按重要度有序截断」，
        # 未改成全库扫描）。
        c = self.conn.cursor()
        # #613：候选池定序**末位补 `id ASC`**——`importance`/`last_access` 上有大量
        # 并列（`distinct_importance=1` 时池内次序全凭 SQLite 的物理取行序），而检索
        # 路径会 `increment_access_many(..., touch=...)` 写回 `last_access` ⇒ 下一次
        # 调用池内次序即变，并列次序泄漏进最终结果（同引擎连续两次调用顺序不同）。
        # `id`（字符串、库内唯一、跨平台稳定）作末位键消掉并列，非并列情形**逐位不变**。
        c.execute(f"SELECT * FROM nodes WHERE {where} "
                  f"ORDER BY importance DESC, last_access DESC, id ASC LIMIT 300",
                  params + like_params)
        rows = c.fetchall()
        # #183：预筛已转义（`%`/`_` 按字面匹配）。但**空结果会触发下面的全表
        # 二元组兜底**，把整库当零分候选返回 ⇒ `search_content("%")` /
        # `search_content("_")` 在库内无字面 `%`/`_` 时仍返回全库（issue 判据
        # 要求「只匹配真正含这两个字符的内容」）。全表兜底是**内容召回**网
        # （补「重组短语无连续子串」的漏检）——查询若**全由 LIKE 通配符构成**，
        # 则无内容信号可供兜底，且转义后的预筛已按字面正确定位。故仅对「纯通配符
        # 查询」跳过兜底；非纯通配符查询的兜底口径**逐字不变**（`test_core_search_
        # discrimination::test_document_without_query_scores_zero` 等既有行为保住）。
        if not rows and any(ch not in "%_" for ch in q):
            # 检索增强（v1.12.1 · AEIS-BENCH-01 发现）：LIKE 预筛落空
            # （查询为重组短语，原文无连续子串命中）→ 回退全表二元组 Jaccard。
            # 节点量级小（千级），全表计算代价可忽略；消除连续性漏检。
            if conds:
                c.execute(f"SELECT * FROM nodes WHERE {' AND '.join(conds)} "
                          f"ORDER BY importance DESC, last_access DESC, id ASC LIMIT 500", params)
            else:
                c.execute("SELECT * FROM nodes "
                          "ORDER BY importance DESC, last_access DESC, id ASC LIMIT 500")
            rows = c.fetchall()
        scored = []
        # 评分用原查询二元组 Jaccard（含并集分母，与 char_bigram_jaccard 同式）；
        # 扩展词只负责预筛召回不稀释评分
        # #267：经 `_tolerant_nodes` 逐行容错——单条坏行不毒化整条检索链。
        rows, skipped = self._tolerant_nodes(rows)
        self.last_skipped_rows = skipped
        for node in rows:
            sim = self.char_bigram_jaccard(q, node.content)
            tag_bonus = 0.05 if any(t in q or q in t for t in node.tags) else 0.0
            scored.append((node, min(1.0, sim + tag_bonus)))
        # 同分按重要性降序（高质量记忆优先，避免并列截断排挤重要节点）；
        # #29④：再按置信度降序——旧实现终排不看 confidence，未验证/被矛盾标记的
        # 低置信节点与高置信节点同分时按插入序（importance 同值时）排前。
        # #613：末位补 `id`——上述各键全部并列时，Python `sort` 是稳定排序 ⇒ 次序
        # 由 `scored` 的输入序（即候选池取行序）决定，而该序会被 `touch` 写回
        # `last_access` 扰动（见候选池 SQL 处注释）⇒ 并列次序泄漏进最终结果。
        # `id` 唯一且跨平台稳定，作末位键消掉并列；非并列情形**逐位不变**。
        # 判据来源：经验标定（#29 修复轮；理论章节追不到）。
        scored.sort(key=lambda x: (-x[1], -x[0].importance, -x[0].confidence, x[0].id))
        # ★优化（本笔）：旧实现 `for node, _ in results: self.increment_access(node.id)`
        # —— 每个命中节点**新开一个 sqlite 连接**。改为一次交给批量单点（一条连接、
        # 一次提交）。计数口径不变：只对最终返回的 `results` 计数；重复 id 经
        # `executemany` 逐条语句各计一次（与旧逐调用一致）。
        # ★本笔合成：`touch` 由开关决定——批量口径与「不刷新近因」并存（见
        # `increment_access_many` 的 `touch` 形参）。
        # ★本笔合成（#787）：调用方显式传 `touch`（唯一现例：`recall` 传 False）时
        # 从其值，否则回落开关缺省——直接检索（本方法）行为逐位不变。
        # ★本笔合成（#473）：`_anchor_rerank` 只重排/截断最终 `results` 的顺序与组成，
        # 批量计数口径不变（仍对最终返回的 `results` 计数）。
        results = self._anchor_rerank(q, scored[:limit], layers, limit)
        touch = self._retrieval_touches_recency() if touch is None else touch
        self.increment_access_many([node.id for node, _ in results], touch=touch)
        return results

    def _anchor_rerank(self, q: str, results, layers, limit: int):
        """结构锚定（LINGSHU_ANCHOR=1）：问句显式点名容器（第N章 / 第N次会话）时，带该容器
        标签的节点按容器内相关度排在最前，余位按原排序补齐。无锚点 / 开关关 / 容器为空 → 原样返回。

        容器内相关度：有 FTS 索引时为容器内 BM25 与 Jaccard 的同式混合（与 FTS 路径同一
        blend），否则为 Jaccard+tag 加成（与 LIKE 路径同口径）。同分按 temporal_coordinate、
        id 定序，跨平台确定。判据来源：经验标定（hive-memory-bench 切A 选定、切B 留出）。"""
        from lingshu.core import retrieval as _rt
        if not _rt.anchor_enabled():
            return results
        tags = _rt.parse_anchors(q)
        if not tags:
            return results
        lays = layers or [l for l in MemoryLayer if l != MemoryLayer.SELF]
        ph = ",".join("?" for _ in lays)
        rows = []
        c = self.conn.cursor()
        for tag in tags:
            # tags 列是 JSON 数组；带引号匹配保证 chapter:1 不误中 chapter:12
            c.execute(f"SELECT * FROM nodes WHERE layer IN ({ph}) AND tags LIKE ?",
                      [l.value for l in lays] + ['%' + json.dumps(tag) + '%'])
            rows.extend(c.fetchall())
        nodes, _ = self._tolerant_nodes(rows)  # #267 逐行容错
        uniq = {}
        for n in nodes:
            uniq.setdefault(n.id, n)
        if not uniq:
            return results
        bm = {}
        fts = getattr(self, "_fts", None)
        if fts is not None:
            terms = [q]
            if _rt.context_enabled():
                terms = _rt.content_grams(terms)
            try:
                bm = fts.scores_within(terms, list(uniq))
            except sqlite3.Error:
                bm = {}
        ranked = []
        for n in uniq.values():
            sim = self.char_bigram_jaccard(q, n.content)
            tag_bonus = 0.05 if any(t in q or q in t for t in n.tags) else 0.0
            j = min(1.0, sim + tag_bonus)
            s = _rt.FTSIndex.blend(j, bm.get(n.id, 0.0)) if fts is not None else j
            ranked.append((n, s))
        ranked.sort(key=lambda x: (-x[1], x[0].temporal_coordinate, x[0].id))
        out = ranked[:limit]
        seen = {n.id for n, _ in out}
        for n, s in results:
            if len(out) >= limit:
                break
            if n.id not in seen:
                out.append((n, s))
                seen.add(n.id)
        return out

    def _search_content_fts(self, q: str, terms, layers, limit: int, touch: bool = None):
        """FTS5+BM25 候选生成（retrieval.py）。返回 None 表示交回原 LIKE 路径。

        候选按 BM25 从全库取（不再受插入顺序与 LIMIT 300/500 截断影响）；
        分数 = max(原 Jaccard+tag 加成, 0.9·s/(s+K))，≥0.95 判重语义只来自 Jaccard。

        #787：`touch` 形参与 LIKE 路径同语义（None → 开关缺省；`recall` 显式传
        False——见 `search_content` docstring），FTS/LIKE 两条检索腿同一口径。"""
        from lingshu.core import retrieval as _rt
        FTSIndex = _rt.FTSIndex
        ctx = _rt.context_enabled()
        qterms = [q] + [t for t in terms if t != q]
        try:
            if ctx:  # 语境扩展：查询侧去掉问句功能二元组
                qterms = _rt.content_grams(qterms)
            cands = self._fts.candidates(qterms, layers, k=max(int(limit) * 5, 100))
        except sqlite3.Error:
            return None
        if cands is None:
            return None
        scored = []
        skipped = []
        for row, bm in cands:
            # #267：与 LIKE 路径同口径逐行容错——坏行只跳过该行并记账，不毒化整条检索链
            got, bad = self._tolerant_nodes([row])
            if bad:
                skipped.extend(bad)
                continue
            node = got[0]
            sim = self.char_bigram_jaccard(q, node.content)
            tag_bonus = 0.05 if any(t in q or q in t for t in node.tags) else 0.0
            scored.append((node, FTSIndex.blend(min(1.0, sim + tag_bonus), bm), bm))
        # 逐字命中闸门（与原 LIKE 路径语义对齐）：
        #   · 查询原文逐字出现在某节点的 content / tags 中时，只返回原路径 LIKE 预筛会命中的
        #     节点——即查询原文或其同义词展开项（terms）任一逐字出现者——逐字含原文者排最前。
        #     同义展开不可省：上游 #44 的 similar 建边要在逐字命中的新节点之外找到经同义词
        #     （如「记忆」→「记录」）命中的旧节点（tests/test_longterm_gate_p1_defects.py）。
        #   · 查询原文无逐字命中（自然语言问句、带空格的关键词串）时不设闸门，交 BM25 部分
        #     匹配——这正是原路径退化为「扫前 500 行」的情形。
        ql = q.lower()
        lterms = [t.lower() for t in terms if t]

        def _has(node, ts):
            c = (node.content or "").lower()
            tg = [str(t).lower() for t in (node.tags or [])]
            return any(t in c or any(t in x for x in tg) for t in ts)
        has_verbatim = any(_has(x[0], [ql]) for x in scored)
        if ctx and scored and not has_verbatim:
            scored = self._fts_context_smooth(q, scored, layers)
        if has_verbatim:
            scored = [x for x in scored if _has(x[0], lterms)]
        # 同分依次按 BM25、重要性、置信度（#29④：与 LIKE 路径同口径，低置信不排在高置信前）
        scored.sort(key=lambda x: (not _has(x[0], [ql]), -x[1], -x[2], -x[0].importance,
                                   -x[0].confidence))
        self.last_skipped_rows = skipped
        # ★本笔合成：FTS 路径同样收进批量单点（#571 禁「循环内 increment_access」），
        # 并以 `touch` 形参落实本件「不刷新近因」——两套语义并存（见
        # `increment_access_many` 的 `touch` 形参）。
        # ★本笔合成（#787）：调用方显式 `touch` 从其值，否则回落开关缺省（与 LIKE 腿
        # 同一 `search_content` 入口的口径）。
        # ★本笔合成（#473）：`_anchor_rerank` 只重排最终 results 的顺序/组成，计数口径不变。
        results = self._anchor_rerank(q, [(n, s) for n, s, _ in scored[:limit]], layers, limit)
        touch = self._retrieval_touches_recency() if touch is None else touch
        self.increment_access_many([n.id for n, _ in results], touch=touch)
        return results

    def _fts_context_smooth(self, q: str, scored, layers):
        """时序邻接平滑：节点的 BM25 加上其前/后一条（同会话窗口内）BM25 的 γ 倍取大者。

        证据常落在命中段落的前后条（对话的上一句、叙述的下一段），它们与问句本身没有共享
        字面。只在自然语言问句路径（无逐字命中）启用，逐字命中语义不变。"""
        from lingshu.core import retrieval as _rt
        bm = {n.id: b for n, _, b in scored}
        node_of = {n.id: n for n, _, _ in scored}
        seeds = [n.id for n, _, b in sorted(scored, key=lambda x: -x[2])[:_rt.NEIGHBOR_SEEDS]]
        bonus: Dict[str, float] = {}
        try:
            pairs = self._fts.temporal_neighbors(seeds, layers)
        except sqlite3.Error:
            return scored
        for seed, row in pairs:
            add = _rt.NEIGHBOR_GAMMA * bm[seed]
            if add <= 0:
                continue
            got, bad = self._tolerant_nodes([row])  # #267 逐行容错
            if bad:
                continue
            nb = got[0]
            node_of.setdefault(nb.id, nb)
            bonus[nb.id] = max(bonus.get(nb.id, 0.0), add)
        out = []
        for nid, node in node_of.items():
            b = bm.get(nid, 0.0) + bonus.get(nid, 0.0)
            sim = self.char_bigram_jaccard(q, node.content)
            tag_bonus = 0.05 if any(t in q or q in t for t in node.tags) else 0.0
            out.append((node, _rt.FTSIndex.blend(min(1.0, sim + tag_bonus), b), b))
        return out

    @staticmethod
    def _bigrams(s: str) -> set:
        s = "".join(s.split())
        if len(s) <= 1:
            return {s}
        return {s[i:i + 2] for i in range(len(s) - 1)}

    def semantic_search(self, query: str, provider, limit: int = 10) -> List[STNode]:
        """语义检索（提供者 duck-typed 依赖注入，核心零外部 import · D-005）。
        提供者契约：provider.encode(text)->List[float]；provider.search(query, limit)->List[str]（节点ID）"""
        ids = provider.search(query, limit)
        nodes = []
        _hit_ids = []
        for nid in ids:
            node = self.get_node(nid)
            if node:
                nodes.append(node)
                _hit_ids.append(nid)
        # ★优化（本笔）：旧实现此处逐命中调 `self.increment_access(nid)` ——
        # 每次都**新开一个 sqlite 连接**（N 次 connect/commit/close）。改为把
        # 命中 id 一次交给批量单点（一条连接、一次提交）。计数口径不变：
        # 只对**实际取到节点**的 id 计数；`executemany` 逐条语句保住重复 id 各计一次。
        # ★本笔合成：`touch` 由开关决定（批量口径 + 不刷新近因并存）。
        self.increment_access_many(_hit_ids,
                                   touch=self._retrieval_touches_recency())
        return nodes

    @_transactional
    def tag_node(self, node_id: str, tag: str):
        """给节点追加标签（幂等：已含该标签则不重复写）。

        #234：非 PRIMARY 不得给**共享层**（anchor/structure）节点打标签——旧实现无
        角色检查，SUB 可对结构层节点伪造 `promotion_pending`/`conflict` 等标签，改变
        PRIMARY 的治理状态。判据来源：公开 issue #234「期望行为：共享层节点的标签、
        保护、冲突登记也只允许 PRIMARY 修改」＋「建议修复 4」。

        #781（可审计）：打 `no_forget` 标签时落一条 `action_logs`（与 `add_node` 的
        `protected_node_overwritten` 同例）。该标签是**遗忘保护的第二个真源**
        （`_is_forget_protected`：`no_forget` ∈ tags ∪ `protections` 行），一旦打上
        即永久豁免 `decay_cycle` 与 `forget_advisor` 两条遗忘路径，而旧实现打标
        **零审计**（`action_logs` 增量 0）⇒ 保护名单可被非属主单向扩张却无留痕。
        审计**只**加在 `no_forget` 这一支，不扩到全部标签面（避免大面积行为变更）。
        边界（如实标注）：本件只补审计；`untag_node` 已由 #448 补上（见该函数，
        撤销 `no_forget` 同样落审计），但**不**新增 `unprotect_node`，也**不**给
        `no_forget` 加配额或审批——后两条属设计取向（谁有权把一条记忆标成不可
        遗忘），见 issue #781 回帖。
        """
        if self.role != Role.PRIMARY:
            n = self.get_node(node_id)
            if n is not None and n.layer in self.SHARED_LAYERS:
                raise PermissionError(
                    f"role={self.role.value} 无权修改共享层节点（{node_id} 为 "
                    f"{n.layer.value}）的标签，共享层由父节点主控（#234）")
        # #344（数据损坏）：读改写必须原子——先 `BEGIN IMMEDIATE` 取得 RESERVED 写锁
        # 再读 `tags`，否则两个连接各自「读到旧 tags → 整列覆盖写」，后写者抹掉先写
        # 者的标签（实测 2 连接各 300 次 `tag_node`：期望 600、留存 300，异常 0 次）。
        # 判据来源：公开 issue #344「tag_node / protect_node 对 nodes.tags 读改写不在
        # 同一事务：多进程 0 异常但标签静默丢失」＋ 其修法建议「把标签改动收进单条 SQL
        # （同一事务内读改写，或 BEGIN IMMEDIATE）」。取后者（IMMEDIATE）——不改
        # `nodes.tags` 的**整列覆盖**语义，只把读改写收进一个原子边界。
        # 边界（削掉的判别力，如实标注）：①本件只保证 `tags` 的读改写原子；同一节点上
        # **其它列**的并发读改写（如 `longterm_gate` 的 importance+tags 合并写）不归本件；
        # ②`_lock` 是**进程内**锁、跨进程无效，故此处用 DB 级 `BEGIN IMMEDIATE`（跨进程
        # 亦成立），而非复用 `_lock`；③已在事务中（嵌套调用）则不插手，交由外层边界。
        own_tx = not self.conn.in_transaction
        if own_tx:
            self.conn.execute("BEGIN IMMEDIATE")
        node = self.get_node(node_id)
        added = False
        if node and tag not in node.tags:
            node.tags.append(tag)
            c = self.conn.cursor()
            c.execute("UPDATE nodes SET tags=? WHERE id=?", (_dumps_tags(node.tags), node_id))
            added = True
        # #781：只对 `no_forget` 落审计（幂等：重复打标不重复留痕——`added` 为真
        # 才记，避免同一事实被反复刷进审计面）。
        if added and tag == "no_forget":
            self.log_action(
                "no_forget_tagged",
                f"节点 {node_id} 被打上 no_forget 标签（#781）：该标签是遗忘保护的"
                f"第二个真源（protections 行之外），豁免 decay_cycle 与 "
                f"forget_advisor 两条遗忘路径；撤销入口见 untag_node（#448）",
                node_ids=[node_id],
                outcome={"layer": node.layer.value, "role": self.role.value,
                         "tag": tag})
        if own_tx:
            self._commit_locked()

    @_transactional
    def untag_node(self, node_id: str, tag: str) -> bool:
        """移除节点上的一个标签（`tag_node` 的对称操作）；返回是否实际移除。

        #448：`tag_node` 只做追加，全仓无撤销入口 ⇒ 误打的 `archived` /
        `no_forget` / `conflict` 等标签无法用公开 API 清除，归档节点被
        `forget_advisor` 永久跳过。本方法补上反向操作：同一把角色闸（共享层仅
        PRIMARY，#234）、同样以 `BEGIN IMMEDIATE` 把读改写收进原子边界（#344）、
        幂等（标签不存在 ⇒ 返回 False，零写入）。

        审计：撤销 `no_forget` 落一条 `no_forget_untagged`（#781 的保护名单
        扩张与收缩都应可追）；撤销 `archived` 本方法不单独记——经
        `restore_archived` 走时会落 `archived_restored`，直接调本方法属普通编辑。
        """
        self.assert_node_writable_by_role(node_id, action="移除共享层节点的标签")
        own_tx = not self.conn.in_transaction
        if own_tx:
            self.conn.execute("BEGIN IMMEDIATE")
        node = self.get_node(node_id)
        removed = False
        if node and tag in node.tags:
            node.tags = [t for t in node.tags if t != tag]
            c = self.conn.cursor()
            c.execute("UPDATE nodes SET tags=? WHERE id=?",
                      (_dumps_tags(node.tags), node_id))
            removed = True
        if removed and tag == "no_forget":
            self.log_action(
                "no_forget_untagged",
                f"节点 {node_id} 的 no_forget 标签被撤销（#448/#781）：该节点重新"
                f"进入 decay_cycle 与 forget_advisor 的遗忘面",
                node_ids=[node_id],
                outcome={"layer": node.layer.value, "role": self.role.value,
                         "tag": tag})
        if own_tx:
            self._commit_locked()
        return removed

    @_transactional
    def restore_archived(self, node_id: str, importance: Optional[float] = None) -> bool:
        """把归档节点还原为活跃：清 `archived` 标签并恢复归档前 importance。

        #448：`forget_advisor` 的 note 一直承诺「可逆」，但修前没有反向入口——
        `archived` 标签只加不清，且 `forget_advisor` 对带该标签的节点直接
        `continue` ⇒ 节点被永久锁在归档态。本方法在**单事务内**同时：
          ① 移除 `archived` 标签（无该标签 ⇒ 返回 False，零写入）；
          ② 恢复 `archive_state.pre_archive_importance`（归档时同事务落库）；
             调用方传 `importance` 时以显式值优先（非有限值拒，越界钳制，同
             #185 口径）；
          ③ 删除该节点的 `archive_state` 行（还原后不再有「待还原原值」）；
          ④ 落一条 `archived_restored` 审计（治理动作可追溯）。
        还原只回退「归档」这一动作本身，不回退归档后产生的其它变化；节点若仍
        满足 `forget_advisor` 的信号（低价值/久未访问），下次调用会再次归档——
        状态机闭合后的正常行为，不是回归。

        边界（如实标注）：本方法**不**复活任何被 `delete_node` 硬删的节点
        （仓内仍无回收站；软删/回收站属独立设计面，见 #448 回帖）；对未归档
        节点返回 False；带 `archived` 标签但 `archive_state` 无登记行（旧库
        遗留）且调用方未显式给 importance 时，保持当前 importance，仅清标签。
        """
        if importance is not None:
            importance = _require_finite("importance", importance)
            importance = _finite_clamp(importance, 0.0, 1.0)
        self.assert_node_writable_by_role(node_id, action="还原共享层节点的归档状态")
        own_tx = not self.conn.in_transaction
        if own_tx:
            self.conn.execute("BEGIN IMMEDIATE")
        node = self.get_node(node_id)
        if node is None or "archived" not in node.tags:
            if own_tx:
                self._commit_locked()
            return False
        prev_imp = node.importance
        c = self.conn.cursor()
        c.execute("SELECT pre_archive_importance, archived_at FROM archive_state "
                  "WHERE node_id=?", (node_id,))
        row = c.fetchone()
        pre_imp, archived_at = (row[0], row[1]) if row else (None, None)
        restore_imp = importance
        if restore_imp is None and pre_imp is not None and math.isfinite(float(pre_imp)):
            restore_imp = _finite_clamp(float(pre_imp), 0.0, 1.0)
        new_tags = [t for t in node.tags if t != "archived"]
        if restore_imp is None:
            c.execute("UPDATE nodes SET tags=? WHERE id=?",
                      (_dumps_tags(new_tags), node_id))
        else:
            c.execute("UPDATE nodes SET tags=?, importance=? WHERE id=?",
                      (_dumps_tags(new_tags), restore_imp, node_id))
        c.execute("DELETE FROM archive_state WHERE node_id=?", (node_id,))
        self.log_action(
            "archived_restored",
            f"节点 {node_id} 从归档态还原（#448）：archived 标签已清、"
            f"importance {prev_imp} -> {restore_imp}",
            node_ids=[node_id],
            outcome={"role": self.role.value, "layer": node.layer.value,
                     "previous_importance": prev_imp,
                     "restored_importance": restore_imp,
                     "pre_archive_importance": pre_imp,
                     "archived_at": archived_at})
        if own_tx:
            self._commit_locked()
        return True

    # ==================== 情境层（M4） ====================

    def enforce_context_cap(self, max_size: int):
        """情境层 FIFO 上限（参照 AEIS），超出淘汰最旧。

        #36-②：淘汰经唯一删除原语 `delete_node` ⇒ 受保护节点（`protections`
        行登记 / `no_forget`）被拒删。语义取「保护优先、允许暂时超限」（待裁清单
        D-23 推荐），超限项不静默丢弃；削掉的判别力：上限不再是硬约束（受保护项
        占位时情境层可 > max_size，本件不引入「淘汰次旧非保护项」的替代口径）。

        #259：淘汰顺序由**墙钟 `created_at`** 改为**插入序 `rowid`**——旧实现
        `ORDER BY created_at ASC`，而 `created_at` 取 `time.time()`（不单调）。
        只要库里有任何情境节点的 `created_at` 大于当前时间（NTP 步进校正 /
        虚拟机快照回滚 / 无 RTC 开机 / `import_all` 导入时钟偏快机器的备份），
        新写入节点的 `created_at` 就是全层最小，紧接着的本次淘汰会把它**当场
        删除**——`add_context` 照常返回 id，库里却已无该节点（写一条丢一条，
        且无任何信号）。改为按 `rowid`（插入序）淘汰最旧，则刚写入的节点
        rowid 最大，永不落入淘汰前缀。`get_recent_context` 同步改 `rowid DESC`。
        判据来源：公开 issue #259（「FIFO 淘汰和『最近情境』按墙钟 created_at
        排序而非写入顺序」）＋ 分诊表 `triage_lingshu_ALL.md` 行 259。
        边界（削掉的判别力，如实标注）：①`rowid` 反映**物理插入序**——本表
        `id` 为 TEXT PRIMARY KEY（非 `WITHOUT ROWID`），`INSERT OR REPLACE`
        命中同 id 会**换 rowid**（等同「重新插入」）；情境节点 id 含 uuid+毫秒
        戳、正常路径不复用同 id，故不影响；②本件只修**排序键**，未采纳 issue
        建议 3（`add_node` 拒绝未来时间戳）与建议 4（可注入时钟）——那是另两条
        独立面；③`created_at` 仍是 `created_at` 列的值，未改其语义。
        """
        c = self.conn.cursor()
        c.execute("SELECT id FROM nodes WHERE layer='context' ORDER BY rowid ASC")
        ids = [r[0] for r in c.fetchall()]
        if len(ids) > max_size:
            for nid in ids[:len(ids) - max_size]:
                self.delete_node(nid)

    def get_recent_context(self, limit: int = 20) -> List[STNode]:
        """取最近写入的情境层节点（短期记忆按**插入序**倒序，默认 20 条）。

        #259：排序键由 `created_at DESC` 改 `rowid DESC`——墙钟不单调，
        时钟回拨后「最近」会指向旧节点（详见 `enforce_context_cap` docstring）。

        #267：经 `_tolerant_nodes` 逐行容错——单条坏行不毒化整层读取。
        #658：`limit=None` ⇒ 全量返回（与同族 `query_nodes_by_time` 同口径）。
        """
        c = self.conn.cursor()
        sql = "SELECT * FROM nodes WHERE layer='context' ORDER BY rowid DESC"
        if limit is not None:
            sql += " LIMIT ?"
            c.execute(sql, (limit,))
        else:
            c.execute(sql)
        nodes, skipped = self._tolerant_nodes(c.fetchall())
        self.last_skipped_rows = skipped
        return nodes

    # ==================== 盲区注册表（M2 · D-001 语义判定） ====================

    @_transactional
    def add_blindspot(self, code: str, description: str, severity: str = "medium",
                      category: str = "operational",
                      predictability: str = "pending_assessment") -> str:
        """登记盲区（已知不可达的领域边界，返回盲区 id）——BLINDSPOT 判定的
        结构化沉淀，供后续 resolve_blindspot 学习消解。

        #386：本方法此前**无任何 category 检查** ⇒ 与引擎门面同语义、公开可达的
        `e.store.add_blindspot(category='meta')` 原样把元盲区落库（宣称的「零记录」
        约束可被绕过）。现与门面同闸：`_is_meta_category` 命中即 `ValueError`。
        """
        if _is_meta_category(category):
            raise ValueError(
                "元盲区零记录（#386）：拒绝把 category 规范化后为 meta 的盲区写入注册表")
        bid = f"bs_{uuid.uuid4().hex[:8]}"
        c = self.conn.cursor()
        c.execute("INSERT INTO blindspots (id, code, description, severity, category, status, created_at, resolved_at, predictability) VALUES (?,?,?,?,?,?,?,?,?)",
                  (bid, code, description, severity, category, "open", time.time(), None,
                   predictability))
        self._commit_locked()
        return bid

    def list_blindspots(self, status: str = None) -> List[Dict]:
        """列出盲区记录（可按 status=open/resolved 过滤）。"""
        c = self.conn.cursor()
        if status:
            c.execute("SELECT * FROM blindspots WHERE status=?", (status,))
        else:
            c.execute("SELECT * FROM blindspots")
        result = []
        for r in c.fetchall():
            d = dict(r)
            result.append({"id": d.get("id"), "code": d.get("code"),
                           "description": d.get("description"), "severity": d.get("severity"),
                           "category": d.get("category"), "status": d.get("status"),
                           "created_at": d.get("created_at"), "resolved_at": d.get("resolved_at"),
                           "attempts": d.get("attempts", 0),
                           "predictability": d.get("predictability", "pending_assessment")})
        return result

    @_transactional
    def resolve_blindspot(self, blindspot_id: str, designer_key: str = None):
        """消解盲区（status→resolved 并记 resolved_at）——盲区学习闭环终点。

        #222：D-007 密钥闸**下沉到 store 层**（fail-closed）。旧实现闸只挂在引擎
        门面 `SpacetimeMemoryEngine.resolve_blindspot` 上，而同一引擎的公开属性
        `e.store` 上有**语义相同、不验密钥**的本方法 ⇒ 任何持有引擎的代码
        （`brain_store`/`scene_model` 等普遍直接调 `engine.store.*`）无密钥即可关闭
        全部盲区，`get_open_blindspots` 只看 `status='open'` ⇒ 盲区从学习队列消失。
        同类的 `adjudicate_promotion`/`adjudicate_verifier_standard` 都把密钥检查放在
        store 层，盲区是例外。
        判据来源：公开 issue #222「建议修复 1：把 `verify_designer` 检查下沉到
        `LayeredStore.resolve_blindspot`（与 `adjudicate_promotion` 一致）」＋
        `D-007`（`resolve_blindspot` docstring 原文「D-007 需设计者密钥：D-001 语义
        判定权在维生系统」）。**理论章节追不到**该落点口径，属经验标定（本件 #222）。
        边界（削掉的判别力，如实标注）：本方法**不**做 status 白名单（只写
        `resolved`，无参数可注入）；白名单在 `update_blindspot_status`。
        """
        if not verify_designer(designer_key):
            raise PermissionError(
                "#222 D-007 设计者认证失败：密钥无效或未配置 AEIS_DESIGNER_KEY（fail-closed）")
        c = self.conn.cursor()
        c.execute("UPDATE blindspots SET status='resolved', resolved_at=? WHERE id=?",
                  (time.time(), blindspot_id))
        self._commit_locked()

    # ==================== 技能记忆（M9） ====================

    @_transactional
    def add_skill(self, name: str, description: str, procedure: str, confidence: float = 0.5) -> str:
        """登记技能卡（名称/描述/步骤/初始置信度），返回技能 id。"""
        sid = f"sk_{uuid.uuid4().hex[:8]}"
        c = self.conn.cursor()
        c.execute("INSERT INTO skills VALUES (?,?,?,?,?,?,?,?)",
                  (sid, name, description, procedure, confidence, 1, time.time(), time.time()))
        self._commit_locked()
        return sid

    @staticmethod
    def _escape_like(q: str) -> str:
        """转义 LIKE 通配符（#200）：`%` 与 `_` 一律按**字面**匹配。

        旧实现把调用方给的 hint 原样塞进 `LIKE ?`，`_` 与 `%` 被当通配符
        ⇒ 一次 `recall_skill("_")` 即可命中任意技能卡（分诊表行 200 第 2 条）。
        """
        return q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    def get_skill_by_name(self, name: str) -> Optional[Dict]:
        """按**精确名称**取技能卡；无同名返回 None（#200）。

        为什么必须另设精确面：`search_skills` 是 `name/description LIKE %q%`
        的**模糊**检索，`record_action_sequence` 用它绑定技能 ⇒ hint「备份」会
        命中「删除备份」，把备份步骤拼进删除技能并为其增信（分诊表行 200 第 1 条）。
        """
        c = self.conn.cursor()
        c.execute("SELECT * FROM skills WHERE name = ? LIMIT 1", (name,))
        r = c.fetchone()
        if r is None:
            return None
        return {"id": r[0], "name": r[1], "description": r[2], "procedure": r[3],
                "confidence": r[4], "version": r[5]}

    def search_skills(self, query: str, limit: int = 10) -> List[Dict]:
        """按关键词检索技能卡（名称+描述 LIKE 匹配，默认前 10）。

        #200：`%` / `_` 经 `_escape_like` 转义（旧实现原样透传 ⇒ 通配符命中任意卡）。

        #272：预筛的 `LIMIT 100` 必须**显式定序**。旧实现无 `ORDER BY` ⇒ 按扫描
        （rowid/插入）序取前 100 行，相关度 `sort` 只在这 100 行里做——命中超过 100
        张后，名字与查询完全相同、置信度最高的卡召不回来（issue 实测：100 张
        「项目NNN日志备份」+ 第 101 张名为「备份」的 confidence=0.9 卡，
        `recall_skill("备份")` 前三张全是旧卡）。现按 `confidence DESC, updated_at DESC`
        定序——截断保留**高置信、较近更新**的候选（与同文件 `query_nodes` 的
        「有界候选池先定序再截断」先例同族，见 `search_content` 的 #35 注释）。
        判据来源：公开 issue #272 正文（「先按插入顺序截前 100 行再排序」）＋ 其补充帖
        「更稳的修法：预筛改成 `ORDER BY confidence DESC, updated_at DESC`（或全表
        排序），解决『先截前 100 行再排序』的本题」（经验标定；仓内无规定技能检索
        候选池排序的理论章节，追不到更早出处）。
        边界（削掉的判别力，如实标注）：①候选池仍**有界**（100 行）——超出后低置信
        命中项依旧不参与打分（本件只把「按插入序静默丢」改成「按置信度有序截断」，
        未改成全表扫描；issue 补充帖亦警告「LIKE 落空退回全表 Jaccard」会让
        `record_action_sequence` 永远非空、新技能建不出来，故不采纳）；
        ②本件**不**把 `procedure` 纳入匹配、也不做语序/空格归一（issue 标题后半段
        「整句子串匹配、步骤字段从不参与」属另一条独立面，未在本件范围内）；
        ③返回前的相关度 `sort` 口径不变（仍按 `char_bigram_jaccard`）。

        #492：预筛模式须**按片段切分**（与同文件 `search_content` 的 #266 同口径）。
        旧实现把整段查询原文拼成**一条** `%{q}%` 模式绑定给两列，查询字节数一过
        SQLite 的 `SQLITE_MAX_LIKE_PATTERN_LENGTH`（上游默认 50000）该 `execute`
        自身即抛 `OperationalError: LIKE or GLOB pattern too complex`——`recall_skill`
        经本方法同链同报错。触发需**两条件同时成立**：skills 表非空（空表走不到
        LIKE 求值）＋ 去空白后的查询非空且过长（实测 49998 字节不抛、49999 字节抛；
        20000 汉字 = 60000 字节抛）。现按 `_like_fragments` 切分：短词原样返回
        `[term]`（短查询 SQL 语义逐字不变），超长词切成多条短模式，每片段各占一组
        `name/description LIKE ? ESCAPE '\'` 子句（片段数上限 `_LIKE_FRAGMENT_CAP`）。
        边界（削掉的判别力，如实标注）：超过片段上限的词尾不参与预筛（与
        `search_content` 同）——本方法**无**「LIKE 落空 → 全表兜底」腿，故超长查询
        的**词尾命中**在预筛阶段不可见（候选池由前若干片段决定）；这与 #272 保留
        有界候选池（`LIMIT 100`）的取舍同族，属已知边界。
        判据来源：公开 issue #492（复现条件与读数为本件复核实测）。
        """
        q = query.strip()
        c = self.conn.cursor()
        if q:
            # #492：整段 `%{q}%` 超长即抛（见 docstring）。按片段切分，短词行为不变。
            like_parts, like_params = [], []
            for frag in self._like_fragments(q):
                like_parts.append(
                    "name LIKE ? ESCAPE '\\' OR description LIKE ? ESCAPE '\\'")
                esc = f"%{self._escape_like(frag)}%"
                like_params.extend([esc, esc])
            like_cond = "(" + " OR ".join(like_parts) + ")"
            c.execute("SELECT * FROM skills WHERE " + like_cond +
                      " ORDER BY confidence DESC, updated_at DESC LIMIT 100",
                      like_params)
        else:
            c.execute("SELECT * FROM skills LIMIT 100")
        skills = [{"id": r[0], "name": r[1], "description": r[2], "procedure": r[3],
                   "confidence": r[4], "version": r[5]} for r in c.fetchall()]
        if q:
            skills.sort(key=lambda s: -self.char_bigram_jaccard(q, s["name"] + s["description"]))
        return skills[:limit]

    def count_skills(self) -> int:
        """技能卡总数。"""
        c = self.conn.cursor()
        c.execute("SELECT COUNT(*) FROM skills")
        return c.fetchone()[0]

    @_transactional
    def update_skill_confidence(self, skill_id: str, delta: float):
        """技能置信度增量更新（clamp 到 [0,1]，正=成功经验/负=失败经验）。"""
        c = self.conn.cursor()
        c.execute("UPDATE skills SET confidence = MIN(1.0, MAX(0.0, confidence + ?)), updated_at=? WHERE id=?",
                  (delta, time.time(), skill_id))
        self._commit_locked()

    # ==================== 固化流水线（M6 · D-003 终裁门槛） ====================

    @staticmethod
    def _content_hash(content: str) -> str:
        """节点内容哈希（#140 提案绑定口径）：sha256 前 16 hex。

        判据来源：`docs/plans/治理与写入边界_修复设计_v0.1.md` §1.1「内容哈希」
        —— 复用 `condition_normalize.cond_hash` 的 `sha256(...).hexdigest()[:16]`
        口径；此处哈希对象是节点 `content` 文本（非条件域字典）。
        """
        import hashlib
        return hashlib.sha256((content or "").encode("utf-8")).hexdigest()[:16]

    @_transactional
    def add_promotion_proposal(self, node_id: str, requester: str, reason: str,
                                 override_rejected: bool = False,
                                 designer_key: str = None) -> str:
        """提交层晋升提案（节点申请跨层升格，如 context→knowledge），返回提案 id。

        #140：提案在提交时**绑定被复核内容的哈希**（`content_hash` 列）——旧实现
        只记 `node_id`（指针），提案期内同 id 改写即可让终裁把**未复核**内容写进
        结构层。判据来源：`docs/plans/待裁清单_v0.1.md` **D-04 / C-5**（绑
        `content_hash`）＋ `docs/plans/治理与写入边界_修复设计_v0.1.md` §3。
        """
        node = self.get_node(node_id)
        # #581（#522 绕过②）：负记忆与共享层拒绝**下沉**到本方法
        # ——门面 propose_promotion 已有这两道闸，但 store 层直调
        # 本方法即绕过它们。
        if node is not None and node.layer in (
                MemoryLayer.ANCHOR, MemoryLayer.STRUCTURE):
            raise ValueError(
                "#581 共享层节点不可再提升"
                "（store 层直调本方法不再绕过门面闸）")
        if node is not None:
            _paths = self.list_rejected_paths(status="open")
            for p in _paths:
                if p.get("path_type") != "promotion":
                    continue
                _sim = self.char_bigram_jaccard(
                    node.content, p.get("description", "") + p.get("reason", ""))
                if _sim > 0.15:
                    if not override_rejected:
                        raise ValueError(
                        "#581 该内容此前已被终裁否决"
                        "（负记忆 %s：%s）"
                        "——重提须走门面并提供设计者密钥"
                        % (p.get("id"), p.get("reason", "")))
                    if not verify_designer(designer_key):
                        raise PermissionError(
                            "#581 负记忆重提须过设计者密钥闸"
                            "（store 层 override 下沉）"
                            "（#522 绕过② 收口）")
                    # action_logs 留痕（#581 另一半）
                    self.log_action(
                        "promotion_negmem_override",
                        f"经密钥放行负记忆重提 {p.get('id','?')}",
                        node_ids=[node_id],
                        outcome={"rejected_path": p.get("id"), "node_id": node_id})
        chash = self._content_hash(node.content if node else "")
        pid = f"pp_{uuid.uuid4().hex[:8]}"
        c = self.conn.cursor()
        c.execute("INSERT INTO promotion_proposals"
                  " (id, node_id, requester, reason, verified_by, adjudicated_by,"
                  "  status, created_at, decided_at, content_hash)"
                  " VALUES (?,?,?,?,?,?,?,?,?,?)",
                  (pid, node_id, requester, reason, "", "", "pending", time.time(),
                   None, chash))
        self._commit_locked()
        return pid

    @_transactional
    def verify_promotion(self, proposal_id: str, verified_by: str) -> bool:
        """复核晋升提案（status→verified 并记录复核人）——晋升双签制第二签。

        #118：补**状态门**与**身份互斥**（D-003 流水线原只有密钥闸，无状态机、
        无角色约束）：
        · 仅 `pending` 可复核——旧实现是无条件 `UPDATE ... SET status='verified'`
          ⇒ 可把已 `rejected`/`approved` 的**终态**提案拉回复核态（终态可重开）；
        · `verified_by` 不得等于 `requester`（提议人不得自兼复核，C-4/D-02
          三条流水线两两互斥同口径）。
        判据来源：`docs/plans/待裁清单_v0.1.md` **C-3 / D-01**（终态不可重开＋重开
        走新提案）＋ **C-4 / D-02**（身份两两互斥）＋ 设计稿
        `docs/plans/治理与写入边界_修复设计_v0.1.md` §4 修法形状 1；对照先例
        `review_verifier_standard` 的三段式状态门（`status != "pending"` 即拒）。
        返回 bool：False = 被状态门/身份门拒绝（**不**改写状态）。
        """
        c = self.conn.cursor()
        c.execute("SELECT status, requester FROM promotion_proposals WHERE id=?",
                  (proposal_id,))
        row = c.fetchone()
        if not row or row[0] != "pending":
            return False
        if verified_by == row[1]:
            return False  # 身份互斥：提议人不得自兼复核
        c.execute("UPDATE promotion_proposals SET verified_by=?, status='verified' WHERE id=?",
                  (verified_by, proposal_id))
        self._commit_locked()
        return True

    @_transactional
    def adjudicate_promotion(self, proposal_id: str, adjudicated_by: str, approved: bool,
                             designer_key: str = None) -> Optional[str]:
        """维生系统终裁（D-007 需设计者密钥）。仅 status='verified'（经验证单元复核）的提案可终裁（D-003）。
        终裁通过后才写入结构层（不可逆），拒绝则提案作废。

        #140：终裁前**重算**被提案节点的 `content` 哈希并与提案绑定的
        `content_hash` 比对——不一致即「批的不是被复核的那份」⇒ 按待裁清单
        **D-05**（拒绝终裁 ＋ 作废该提案）处置：提案 `status='rejected'`、
        **不**写结构层、返回 `None`。判据来源：`docs/plans/待裁清单_v0.1.md`
        D-04/D-05 ＋ `docs/plans/治理与写入边界_修复设计_v0.1.md` §3「守卫判据」
        （`propose(A) → verify → add_node(同 id, B) → adjudicate(True)` ⇒ 终裁被拒、
        节点留 knowledge、content 仍为 A）。

        #167：待终裁期间被提案节点可能被**情境层自然遗忘/`enforce_context_cap` FIFO**
        静默删除（`propose_promotion` 只打 `promotion_pending` 标签，**未** protect）。
        旧实现终裁通过时无条件 `UPDATE nodes SET layer='structure' WHERE id=?`（不查
        影响行数）＋无条件标 `approved` ⇒ 结构层 **0 行写入**，审计却记「已入结构层」
        并指向一个不存在的节点（且老提案 `content_hash` 为空时 #140 的哈希闸不生效，
        实测 `adjudicate returns: True / status: approved`）。
        现加**节点存续门**（fail-closed）：终裁前节点已不存在 ⇒ 提案标 `rejected`、
        落一条 `action_logs`（把「静默删除」变为可审计的显式拒绝）、返回 `None`，
        **不**写结构层、**不**标 approved。
        判据来源：分诊表 `triage_lingshu_ALL.md` 行 167；仓内无规定「待终裁节点
        存续」的理论章节，追不到更早出处（经验标定，本件 #167）。
        边界（削掉的判别力/未覆盖面）：本件**不**引入「pending 期间 protect 该节点」
        的语义（保护是永久、无 unprotect 生命周期——属设计取舍，见 B-3/D-15，不擅动）
        ⇒ 待终裁节点仍可能被情境层遗忘删除，本闸只保证**不谎报**、且该事件**留痕**。
        """
        if not verify_designer(designer_key):
            raise PermissionError(
                "D-007 设计者认证失败：密钥无效或未配置 AEIS_DESIGNER_KEY（fail-closed）")
        # #586：实例角色闸——终裁是**共享层写**（下方 `UPDATE nodes SET
        # layer='structure', confidence=1.0`），与 `add_node` 的 `SHARED_LAYERS`
        # 角色闸同向：非 PRIMARY 即便持有效设计者密钥也不得写共享层。旧实现只验
        # 密钥 ⇒ SUB 本地「自提案→自复核→自终裁」把任意内容写成结构层节点
        # （confidence=1.0、不可删、不衰减），并经 `prepare_shared_sync` /
        # `apply_shared_sync` 传入 PRIMARY（公开 issue #586 复现读数）。密钥取自
        # 进程环境变量，同一部署下启动的 SUB 天然持有 ⇒ 密钥闸不构成角色隔离。
        # fail-closed：先判后写，抛错时零状态变更（提案状态亦不改写）。
        # 边界（削掉的判别力，如实标注）：本闸只判**实例角色**；SUB 仍可
        # `propose_promotion`/`verify_promotion`（提案与复核属本地动作），只是
        # 终裁权归父节点主控。判据来源：公开 issue #586 ＋ 写权限单一真源
        # `SHARED_LAYERS`（`add_node`/`protect_node`/洞见 #569 同款）。
        if self.role != Role.PRIMARY:
            raise PermissionError(
                f"role={self.role.value} 无权终裁固化提案（终裁写共享层，"
                f"由父节点主控，#586）")
        c = self.conn.cursor()
        c.execute("SELECT node_id, status, content_hash, requester, verified_by "
                  "FROM promotion_proposals WHERE id=?", (proposal_id,))
        row = c.fetchone()
        if not row:
            return None
        node_id, status, bound_hash = row[0], row[1], (row[2] or "")
        requester, verified_by = (row[3] or ""), (row[4] or "")
        if status != "verified":
            return None
        # #565：身份互斥——终裁人不得等于提议人或复核人（C-4 / D-02「三者两两互斥」）。
        # #118 的修复建议第1条把「终裁人是否允许与前序角色重合」显式标为**由维护者
        # 定口径**的悬置项；同仓 A-2 流水线（`adjudicate_verifier_standard`）已按
        # C-4/D-02 落实为 `adjudicator not in (proposer, independent_reviewer,
        # cs_reviewer)`，本件按同一口径对齐，避免两条流水线行为不一致。
        # 例外：库里若 `verified_by` 为空（老提案未经复核人字段），不据空值拒——
        # 只在 `adjudicated_by` 与非空的 `requester`/`verified_by` 相撞时拒。
        if adjudicated_by and adjudicated_by in (x for x in (requester, verified_by) if x):
            return None  # 身份互斥：终裁人与提议/复核人两两互斥（C-4 / D-02）
        # #140：内容绑定比对（未绑定哈希的老提案（bound_hash 空）不做比对，保持向后兼容）
        node = self.get_node(node_id)
        # #167：节点存续门（fail-closed）——见 docstring。先于哈希比对：老提案
        # `bound_hash` 为空时哈希闸不生效，旧实现会照样把「不存在的节点」标 approved。
        if node is None:
            c.execute("UPDATE promotion_proposals SET adjudicated_by=?, status=?, decided_at=?"
                      " WHERE id=?",
                      (adjudicated_by, "rejected", time.time(), proposal_id))
            self._commit_locked()
            try:
                self.log_action(
                    "promotion_node_missing",
                    f"固化提案终裁被拒（被提案节点已不存在）：{proposal_id}",
                    node_ids=[node_id],
                    outcome={"proposal_id": proposal_id, "node_id": node_id,
                             "adjudicated_by": adjudicated_by})
            except Exception:
                pass
            return None
        if bound_hash and self._content_hash(node.content if node else "") != bound_hash:
            c.execute("UPDATE promotion_proposals SET adjudicated_by=?, status=?, decided_at=?"
                      " WHERE id=?",
                      (adjudicated_by, "rejected", time.time(), proposal_id))
            self._commit_locked()
            return None
        if approved:
            c.execute("UPDATE nodes SET layer='structure', confidence=1.0 WHERE id=?", (node_id,))
        c.execute("UPDATE promotion_proposals SET adjudicated_by=?, status=?, decided_at=? WHERE id=?",
                  (adjudicated_by, "approved" if approved else "rejected", time.time(), proposal_id))
        self._commit_locked()
        return node_id

    # ==================== 遗忘门控（M7） ====================

    @_transactional
    def protect_node(self, node_id: str, reason: str):
        """按 3.2 节不可遗忘类别保护：no_forget 标记 + 保护登记，衰减跳过

        #234：非 PRIMARY 不得给**共享层**节点登记保护（旧实现无角色检查，SUB 可对
        结构层节点 `protect_node` ⇒ 不可遗忘语义被第三方写入）。判据来源：公开
        issue #234「期望行为」＋「建议修复 4」。

        #344（数据损坏）：`protections` 行与 `no_forget` 标签的读改写必须原子——
        旧实现先 `INSERT protections` 再 `get_node` 读 tags、最后整列覆盖写，读与写
        不在同一事务 ⇒ 另一连接在此期间提交的标签被抹掉（`no_forget` 丢失，而
        `protections` 表里仍有该行：两处不同源）。判据来源：公开 issue #344「tag_node /
        protect_node 对 nodes.tags 读改写不在同一事务：多进程 0 异常但标签静默丢失」
        ＋ 其修法建议「同一事务内读改写，或 BEGIN IMMEDIATE」。
        边界（削掉的判别力，如实标注）：①本件只把读改写收进原子边界，**不**把
        `no_forget` 改成 `protections` 表的派生视图（issue 并列的第二条建议属设计
        取舍，未采纳）；②`_lock` 为进程内锁、跨进程无效，故用 DB 级 `BEGIN IMMEDIATE`；
        ③已在事务中（嵌套调用）则不插手，交由外层边界。

        #778-F3（数据损坏 · 前置存在性）：本方法此前**先写后查**——旧实现先
        `INSERT OR REPLACE INTO protections` 再 `get_node(node_id)` ⇒ 拼错 id、
        传 `None`/空串**一律静默成功**，`protections` 落一条**孤儿行**（`nodes` 无此
        节点），且该行成为**潜伏保护**：之后同 id 建节点即被拒删（`delete_node`
        返回 False，而 `no_forget` 标签并未回填）。修法：把取节点提到写行**之前**，
        不存在即 `ValueError`（异常类型先例：同文件 `promote_to_shared` 的
        `raise ValueError("节点不存在")`）。
        判据来源：公开 issue #778 F3 ＋ 其核验帖（另补出 `protect_node(None, …)`
        可无上界灌 NULL 行、并使 `get_protected_nodes()` 返回 `None`/`''` 条目）。
        边界（如实标注）：本闸只判「目标节点存在」，**不**判 `reason` 语义、**不**引入
        配额或审批（「谁有权把一条记忆标成不可遗忘」属设计取向，见 #781 回帖）。

        #781（可审计）：本方法落一条 `action_logs`（与 `add_node` 的
        `protected_node_overwritten` 同例），使「某节点被登记为不可遗忘」可查——
        旧实现 `tag_node` / `protect_node` 都不落审计（`action_logs` 增量 0），
        保护名单可被非属主单向扩张却无留痕。审计只加在**本方法**与 `tag_node` 的
        `no_forget` 一支，**不**扩到全部标签面（避免大面积行为变更）。
        """
        own_tx = not self.conn.in_transaction
        if own_tx:
            self.conn.execute("BEGIN IMMEDIATE")
        # #778-F3：**先取节点**再写行。这一次读同时供两道闸用（存在性 + #234 角色），
        # 且落在 #344 的原子边界**之内**——顺序与 #344 的修复口径一致（读 tags 与
        # 写 tags 同事务），故不引入第二次 `get_node`。
        node = self.get_node(node_id)
        if node is None:
            raise ValueError(
                "protect_node 拒绝：节点不存在（id=%r）——#778-F3：保护登记要求目标"
                "节点已存在，否则会落下指向空处的孤儿保护行（旧实现先写后查，拼错 "
                "id 静默成功，且该行会成为潜伏保护）" % (node_id,))
        if self.role != Role.PRIMARY and node.layer in self.SHARED_LAYERS:
            raise PermissionError(
                f"role={self.role.value} 无权保护共享层节点（{node_id} 为 "
                f"{node.layer.value}），共享层由父节点主控（#234）")
        c = self.conn.cursor()
        c.execute("INSERT OR REPLACE INTO protections VALUES (?,?,?)", (node_id, reason, time.time()))
        tag_added = "no_forget" not in node.tags
        if tag_added:
            node.tags.append("no_forget")
            c.execute("UPDATE nodes SET tags=? WHERE id=?", (_dumps_tags(node.tags), node_id))
        # #781：保护登记落审计（只追加，不进检索面）。
        self.log_action(
            "protected_node_registered",
            f"节点 {node_id} 被登记为不可遗忘（#781）：protections 行 + no_forget "
            f"标签，豁免 decay_cycle 与 forget_advisor 两条遗忘路径",
            node_ids=[node_id],
            outcome={"reason": reason, "layer": node.layer.value,
                     "role": self.role.value,
                     "no_forget_tag_added": tag_added})
        if own_tx:
            self._commit_locked()

    def get_protected_nodes(self) -> List[str]:
        """取保护名单节点 id 列表（不可遗忘层的豁免集合）。"""
        c = self.conn.cursor()
        c.execute("SELECT node_id FROM protections")
        return [r[0] for r in c.fetchall()]

    def orphan_protection_rows(self) -> List[str]:
        """取 `protections` 表里的**孤儿行** node_id（指向不存在节点的保护登记）。

        #778-F4（加固不对称）：`protections` 表**无外键声明**（对照 `edges` 有两条
        `REFERENCES nodes(id)` ＋ #93 的孤儿边预筛）⇒ `PRAGMA foreign_key_check`
        **结构上覆盖不到它**：`node_id` 指向不存在节点的保护行可静默落库、
        `integrity_ok` 仍为 True，且该行随 `export_all` 继续传播。本方法是该缺陷的
        **判据真源**（单一实现），上报面为 `SpacetimeMemoryEngine.verify_integrity`
        与 `export_all`。

        为何走「独立检查」而不补外键（如实标注取舍）：① SQLite 无法对既有表
        `ALTER TABLE ... ADD CONSTRAINT` ⇒ 补外键只对新库生效、旧库永久无约束
        （schema 兼容性分叉）；② 本仓每条连接都开 `PRAGMA foreign_keys=ON`（#93），
        一旦 `protections` 有了外键，**含孤儿保护行的备份**（正是本缺陷的现场）会在
        `import_all` 的通用写腿抛 `FOREIGN KEY constraint failed` 并**中断整次导入**
        ——与 #93 当年「必须先补孤儿边预筛再开外键」是同一个坑，而导入侧的对称预筛
        落在 `import_all` 内（属另一条修复线）。故取「独立孤儿检查 ＋ 完整性面上报」，
        不动表结构、不改导入语义。

        边界（如实标注）：本方法**只读不清理**（孤儿行不自动删除——「谁有权解除
        保护登记」属设计取向，本件不引入回收入口）；返回按 `rowid` 升序（稳定序，
        便于取样本），`NULL`/空串 `node_id` 同样计入孤儿（它们必不匹配任何节点）。
        """
        c = self.conn.cursor()
        c.execute("SELECT p.node_id FROM protections p "
                  "LEFT JOIN nodes n ON p.node_id = n.id "
                  "WHERE n.id IS NULL ORDER BY p.rowid ASC")
        return [r[0] for r in c.fetchall()]

    def _is_forget_protected(self, node, protected: Set[str] = None,
                             *, node_id: str = None, tags: List[str] = None) -> bool:
        """遗忘保护谓词（唯一真源，lingshu #36）：`protections` 行登记 ∪ `no_forget` 标签。

        供 `delete_node` / `decay_cycle`（情境层删除段）共用——修前两处判据不一致
        （同函数上半段对边已排除保护，下半段删节点却不查）。
        判据来源：设计稿 `docs/plans/删除恢复持久化完整性_修复设计_v0.1.md`
        §3.3「可复用的既有机制（无需新造）：get_protected_nodes() ＋ no_forget 标签」
        ＋ 待裁清单 D-15/D-17（应用层单一谓词）。
        边界（削掉的判别力）：`protections.created_at` 仍不被读——保护**永久**生效，
        无 TTL（待裁清单 D-11 裁定「永久＋显式解除」，本件不引入时效语义）。

        #781（口径声明 · **只声明现状、不改语义**）：本谓词**不认 `archived`**——
        它只判 `no_forget` ∈ tags ∪ node_id ∈ protections。而
        `SpacetimeMemoryEngine.forget_advisor` 的跳过条件是
        `"no_forget" in tags or "archived" in tags`（**认 `archived`**）⇒ 两条遗忘
        路径对 `archived` 的处理相反：仅带 `archived` 的情境节点被 `forget_advisor`
        跳过，却被 `decay_cycle`（经本谓词）照常删除。本件**不**统一（属设计面
        裁定），只把分叉显式落到注释，防日后无声漂移。

        ★优化（本笔 · 零行为变更）：本谓词实际只读 `node.tags` 与 `node.id` 两个字段，
        而批量调用方（`decay_cycle` 的两段循环）此前传的是 `self.get_node(nid)`
        —— 那是 `SELECT * FROM nodes WHERE id=?`（读**整行** + 建 `STNode`），每节点一次。
        现新增 `node_id` / `tags` **直传入口**：批量调用方已由主查询带出这两列，无需回查。
        **判据本体未动**（仍 `no_forget` ∈ tags ∪ `node.id` ∈ protected），且
        「传 node / 传 node_id+tags」两条入口走**同一段**判定 ⇒ 不可能分叉。
        """
        if node_id is None:                      # 旧入口：由 node 取两字段
            if node is None:
                return False
            node_id, tags = node.id, node.tags
        if "no_forget" in (tags or []):
            return True
        if protected is None:
            protected = set(self.get_protected_nodes())
        return node_id in protected

    # ==================== 冲突标记（M5） ====================

    @_transactional
    def register_conflict(self, a_id: str, b_id: str,
                          condition_space: ConditionSpace = None) -> Optional[STEdge]:
        """矛盾记忆显式标记：OPPOSITE 边（未验证，置信度0.3），待验证单元复核

        #234：非 PRIMARY 不得对**共享层**节点登记冲突（旧实现无角色检查，SUB 可给
        结构层节点打 `conflict` 标签并建 OPPOSITE 边）。判据来源：公开 issue #234
        「期望行为」＋「建议修复 4」。（`add_edge` 的角色闸另会拦共享层端点边。）
        """
        # ★优化（本笔 · 零行为变更）：本方法此前对两端点合计调 **4 次**
        # `self.get_node(...)`（`SELECT *`，读整行 + 建 `STNode`）：
        #   ① 角色闸逐端点取整节点（其实只用 `layer`）；
        #   ② 存在性检查 `if not self.get_node(a_id) or not self.get_node(b_id)`。
        # 现先各取一次 `layer`（两行、轻查询）同时供两用：
        #   `_lay[nid] is None` ⇔ 该 id 不存在（与旧 `get_node(...) is None` 同义）。
        # `MemoryLayer(...)` 与 `STNode.from_row` 同款裸露转换 ⇒ 非法 layer 的
        # 失败语义一致。
        _c0 = self.conn.cursor()
        _lay: Dict[str, Optional[MemoryLayer]] = {}
        for _nid0 in (a_id, b_id):
            _c0.execute("SELECT layer FROM nodes WHERE id=?", (_nid0,))
            _r0 = _c0.fetchone()
            _lay[_nid0] = None if _r0 is None else MemoryLayer(_r0[0])
        if self.role != Role.PRIMARY:
            for nid in (a_id, b_id):
                _l0 = _lay[nid]
                if _l0 is not None and _l0 in self.SHARED_LAYERS:
                    raise PermissionError(
                        f"role={self.role.value} 无权登记共享层节点（{nid} 为 "
                        f"{_l0.value}）的冲突，共享层由父节点主控（#234）")
        if _lay[a_id] is None or _lay[b_id] is None:
            return None
        edge = STEdge(
            id=f"edge_{uuid.uuid4().hex[:8]}_{int(time.time()*1000)}",
            source_id=a_id, target_id=b_id,
            relation_type=EdgeType.OPPOSITE,
            condition_space=condition_space or ConditionSpace(
                "冲突检测", "验证单元预筛选", (time.time(), time.time()+3600), "协议实例运行中"),
            confidence=0.3, verified=False)
        self.add_edge(edge)
        for nid in (a_id, b_id):
            # ★优化（本笔）：本段只用 **tags**；旧 `self.get_node(nid)` 读整行 + 建
            # `STNode`。改为直取 `tags`。`json.loads` 与 `from_row` 同款裸调 ⇒
            # 坏行失败语义一致；`fetchone()` 为 None 与旧 `node` 为 None 同分支。
            c = self.conn.cursor()
            c.execute("SELECT tags FROM nodes WHERE id=?", (nid,))
            _tr = c.fetchone()
            if _tr is None:
                continue
            _tlist = json.loads(_tr[0]) if _tr[0] else []
            if "conflict" not in _tlist:
                _tlist.append("conflict")
                c.execute("UPDATE nodes SET tags=? WHERE id=?", (_dumps_tags(_tlist), nid))
                self._commit_locked()
        return edge

    @_transactional
    def prune_self_snapshots(self, keep: int) -> int:
        """自我快照有界保留（lingshu #201）：只留最近 `keep` 条**自动快照**
        （tags 含 `self_snapshot`，按**插入序** `rowid` 降序），其余连同关联边
        删除，返回删除条数。

        为什么不经 `delete_node`：SELF 层在 `IMMUTABLE_LAYERS` 内（v1.16 扮演论
        「自我锚点不可遗忘」），`delete_node` 一律拒删；而 `update_self` 每次调用
        新建一条 `self_snapshot`（importance=0.9 / confidence=1.0）⇒ 修前无清理
        路径，无界堆积且以普通记忆身份挤占召回（issue #201 实测 20 轮后前 10 命中
        中 9 条是快照）。
        判据来源：经验标定（#201 修复轮；理论章节追不到）；上界口径与
        `SelfModel.HISTORY_MAX`（同类的状态变更历史有界钳制，见
        tests/test_self_model_history.py）同族。
        边界（削掉的判别力）：对**自动快照**开了「不可删」的一个窄口（仅此一类、
        无 TTL）；SELF 层其它节点、`protections` 名单节点一律不动。

        #489：保留序由**墙钟 `created_at`** 改为**插入序 `rowid`**——旧实现
        `ORDER BY created_at DESC`，而 `created_at` 取 `time.time()`（不单调）。
        只要库里有任何 SELF 快照的 `created_at` 大于当前时间（NTP 步进校正 /
        虚拟机快照回滚 / 无 RTC 开机 / `import_all` 导入时钟偏快机器的备份），
        `update_self` **本次刚插入**的快照 `created_at` 就是全层最小，紧随其后的
        本次淘汰会把它**当场删除**；随后 `update_self` 用该 id `add_edge` ⇒
        `IntegrityError: FOREIGN KEY constraint failed`（半截副作用：节点与边各自
        已 commit，内存 `self_model` 已更新而库内快照已删）。改为按 `rowid`
        （插入序）降序，则刚写入的快照 rowid 最大，永不落入淘汰前缀；与情境层
        `enforce_context_cap`（#259）**同根同法**。
        判据来源：公开 issue #489（值守复核帖 6165096839 / 6187655595 的「建议
        修复面锚①：保留序改 `rowid DESC`，贴 #259 情境层先例」）。
        边界（削掉的判别力，如实标注）：①`rowid` 反映**物理插入序**——本表
        `id` 为 TEXT PRIMARY KEY（非 `WITHOUT ROWID`），`INSERT OR REPLACE`
        命中同 id 会**换 rowid**；快照 id 含 uuid+毫秒戳、正常路径不复用同 id，
        故不影响；`export_all`/`import_all` 往返保序（导出 `SELECT *` 按 rowid
        扫描、导入按该序插入）⇒ 灾备恢复后 rowid 序仍等于原始写入序；②本件只改
        **排序键**，未采纳「拒绝未来 `created_at`」——那是另设「未来多久算非法」
        的容差口径，属设计者裁面，且会误伤时钟偏快机器所出备份的合法历史。
        """
        # #691 P1-a：删除前须过**唯一保护谓词** `_is_forget_protected`——
        # 旧实现直接 `DELETE`，`protect_node`（`protections` 行）或 `no_forget`
        # 标签登记的快照**仍被删**（受保护节点被本清理面静默绕过保护）。
        # 本件只补**保护谓词**（选面 `tags LIKE '%self_snapshot%'` 口径不动，
        # 不顺手扩大改动面）；受保护件既不计入 `doomed` 也不删其关联边。
        # 谓词直传入口（`node_id` + `tags`）与 `decay_cycle` 同款，判据本体同一段。
        c = self.conn.cursor()
        c.execute("SELECT id, tags FROM nodes WHERE layer='self' AND tags LIKE ? "
                  "ORDER BY rowid DESC", ("%self_snapshot%",))
        rows = c.fetchall()
        if len(rows) <= keep:
            return 0
        # #769 ②（治理列写面补角色闸）：SELF 层在 `IMMUTABLE_LAYERS` 内——`delete_node`
        # 对 SELF 行一律返回 `False`，而本方法此前**零角色判据**直接 `DELETE` ⇒
        # 同一层的「不可删」语义两套口径：SUB 调 `prune_self_snapshots(keep=0)` 即
        # 硬删全部 SELF 快照（核验帖实测 SELF 行 `3 → 0`，而 `delete_node` 对同层
        # 仍返回 `False`）。现按 `delete_node` 的**同判据**收口：非 PRIMARY 一行都不删。
        #
        # 取向（二选一，如实标注理由与代价）：**返回 0 并经 `warnings.warn` 上报**，
        # 不抛 `PermissionError`。理由：SELF 属**本地层**（#191 裁定：SUB 可写自我层），
        # `update_self` 在 SUB 上是合法路径（其内部即调本方法做有界保留）⇒ 若在此抛错，
        # SUB 的 `update_self` 会被一并打掉，属**回归 #191**（正是 `add_node` 改用
        # `SHARED_LAYERS` 而非 `IMMUTABLE_LAYERS` 所要避免的形态）。代价（如实）：
        # 本闸是「拒删 + 告警」而非 fail-closed 抛错，调用方若忽略告警只会看到
        # `0`——故告警文案显式点名角色与跳过条数，且该口径只在「确实有候选被跳过」
        # 时触发（`len(rows) > keep`），正常小库与 PRIMARY 侧零告警噪音。
        if self.role != Role.PRIMARY:
            warnings.warn(
                "#769 prune_self_snapshots 拒绝：role=%s 无权回收自我层快照"
                "（SELF 属不可遗忘层，与 delete_node 同判据）——本次跳过 %d 条候选，"
                "零删除；如需回收请以 PRIMARY 身份执行。"
                % (self.role.value, len(rows) - keep),
                RuntimeWarning, stacklevel=2)
            return 0
        protected = set(self.get_protected_nodes())
        # keep 保留的是**最近** keep 条（rowid 降序前缀）；保护件在淘汰候选里跳过。
        doomed = []
        for nid, tags_json in rows[keep:]:
            try:
                _tags = json.loads(tags_json) if tags_json else []
            except (ValueError, TypeError):
                _tags = []
            if self._is_forget_protected(None, protected, node_id=nid, tags=_tags):
                continue
            doomed.append(nid)
        if not doomed:
            return 0
        with self._lock:
            for nid in doomed:
                # #93：先删边再删节点（外键开启后顺序反了会 FK constraint failed）。
                # #168：删边经唯一真源（对不可变层对端的边先落审计）
                self._strip_incident_edges(c, nid)
                c.execute("DELETE FROM nodes WHERE id=?", (nid,))
                # #448：硬删同步清归档区登记（见 delete_node 同款注释）。
                c.execute("DELETE FROM archive_state WHERE node_id=?", (nid,))
            self._commit_locked()
        return len(doomed)

    def count_layer(self, layer: MemoryLayer) -> int:
        """统计指定记忆层的节点数。"""
        c = self.conn.cursor()
        c.execute("SELECT COUNT(*) FROM nodes WHERE layer=?", (layer.value,))
        return c.fetchone()[0]

    # ==================== v1.3 扩展（P0-1/P0-3） ====================

    def get_nodes_by_tag(self, tag: str, limit: int = 50) -> List[STNode]:
        """按标签**精确**检索节点（实体上下文组装用）。

        #183：旧实现 `tags LIKE '%{tag}%'` 有两个面——
        ① **子串误配**：`LIKE '%ent:car_1%'` 命中 `ent:car_12` / `ent:carX1`；
           `cycle:c1` 命中 `c10…c19`。
        ② **通配符注入**：`_` 匹配任意单字符、`%` 匹配全部 ⇒
           `get_nodes_by_tag("%")` / `get_nodes_by_tag("_")` 返回整库。
        现走 `json_each` 做**元素级精确匹配**（`tags` 列是 JSON 数组文本，见
        `_dumps_tags`——故不能写 `tags = ?`）。`json_valid` 前置短路：坏行
        （非法 JSON）不参与、也不让整条查询抛 `OperationalError`（与
        `_tolerant_nodes` 的「单条坏行不毒化整条检索链」同口径）。

        ★调用方 survey（#183 改前）：仓内 `get_nodes_by_tag` 的全部调用点
        （`vprim` ×2、`cycle:{id}`、`cycle_record_full`、`scene_model.
        load_world_from_memory(tag="spatial")`）传入的都是**完整标签字面量**，
        **无一依赖子串/前缀语义**，故此处直接改精确匹配、不另设
        `get_nodes_by_tag_prefix`。（另注：`visual_check` 的 `screen_state` 检索
        走的是**独立** SQL 且显式 `tags LIKE '%screen_state%'`，不经本方法，
        见 core.py `visual_check` 内 #271 注释。）

        判据来源：公开 issue #183（两个面的复现）；`json_each` 精确匹配口径
        沿用任务书建议 1（Python 3.12 sqlite3 内建 JSON1，无额外依赖）。
        阈值本体**未改**（默认 limit=50）。

        #658：`limit=None` ⇒ 全量返回（与同族 `query_nodes_by_time` 同口径）。
        """
        c = self.conn.cursor()
        sql = ("SELECT * FROM nodes WHERE EXISTS ("
               "SELECT 1 FROM json_each("
               "CASE WHEN json_valid(nodes.tags) THEN nodes.tags ELSE '[]' END) "
               "WHERE value = ?) "
               "ORDER BY importance DESC, last_access DESC")
        if limit is not None:
            c.execute(sql + " LIMIT ?", (tag, limit))
        else:
            c.execute(sql, (tag,))
        nodes, _skipped = self._tolerant_nodes(c.fetchall())
        return nodes

    @_transactional
    def update_blindspot_status(self, blindspot_id: str, status: str,
                                designer_key: str = None):
        """更新盲区状态（P0-3 终态流转）。

        #222：①**status 白名单**——旧实现接受任意字符串（如 `"ignored"`），而
        `get_open_blindspots` 只看 `status='open'` ⇒ 任何非 open 值都把盲区从学习
        队列里抹掉且不留可解释的状态；现只接受 `BLINDSPOT_STATUSES`（open/resolved），
        越界即 `ValueError`。②**非 open 状态需 D-007 密钥**（与 store 层
        `resolve_blindspot` 同款 fail-closed）——旧实现无密钥即可关闭盲区。
        判据来源：公开 issue #222「建议修复 2：`update_blindspot_status` 对 `open`
        以外的状态同样要求密钥，并对 status 做白名单（`open`/`resolved`/…）」
        （经验标定，仓内理论章节追不到）。
        边界（削掉的判别力，如实标注）：①白名单只列仓内**实际出现**的两种状态
        （`open`/`resolved`）——issue 括注的「…」属未定项，不擅扩；②`status='open'`
        （重新打开）按 issue 原文**不**要求密钥（该方向只是把盲区放回学习队列，非
        抹除），故此处不拦——如实声明。
        """
        if status not in self.BLINDSPOT_STATUSES:
            raise ValueError(
                "#222 盲区状态越界：%r（白名单 %r）"
                % (status, tuple(self.BLINDSPOT_STATUSES)))
        if status != "open" and not verify_designer(designer_key):
            raise PermissionError(
                "#222 D-007 设计者认证失败：密钥无效或未配置 AEIS_DESIGNER_KEY（fail-closed）")
        c = self.conn.cursor()
        resolved_at = time.time() if status != "open" else None
        c.execute("UPDATE blindspots SET status=?, resolved_at=? WHERE id=?", (status, resolved_at, blindspot_id))
        self._commit_locked()

    # ==================== v1.4 扩展（P1-1/P1-2/P1-3/P1-4） ====================

    def get_nodes_in_range(self, start_ts: float, end_ts: float, layer: MemoryLayer = None,
                           limit: int = 50) -> List[STNode]:
        """时间范围查询（P1-2 时间线）

        #658：`limit=None` ⇒ 全量返回（与同族 `query_nodes_by_time` 同口径）。
        """
        conds, params = ["temporal_coordinate >= ?", "temporal_coordinate <= ?"], [start_ts, end_ts]
        if layer is not None:
            conds.append("layer=?")
            params.append(layer.value)
        c = self.conn.cursor()
        sql = (f"SELECT * FROM nodes WHERE {' AND '.join(conds)} "
               "ORDER BY temporal_coordinate DESC")
        if limit is not None:
            c.execute(sql + " LIMIT ?", params + [limit])
        else:
            c.execute(sql, params)
        nodes, _skipped = self._tolerant_nodes(c.fetchall())
        return nodes

    @_transactional
    def update_node_importance(self, node_id: str, delta: float):
        """重要度更新（P1-4 巩固）

        #185：`delta` 经 `_finite_or` 归一（非有限/非数 → 0.0，即「不改动」）。
        旧实现原样拼进 `importance + ?` ⇒ NaN 落库成 NULL，此后 `decay_cycle` 的
        `WHERE importance > ?`（情境层遗忘门）对 NULL 全不成立 ⇒ 该节点**永不遗忘**；
        `+inf` 则使其在所有 `ORDER BY importance DESC` 检索里永远排第一。

        #769 ④ / #780 ②（D-33 的「另笔」）：同 `update_node_confidence`——本接口
        此前无层/角色判据（既有守卫 docstring 明写「数值接口不在本件范围」），SUB
        直调即把锚点 `importance 0.3 → 0.8` / `0.9 → 1.0`。现按同款单点判据
        **写前按库中旧行**判权限：非 PRIMARY 且旧行属共享层 ⇒ `PermissionError`。
        内部调用方（`recalc_structural_importance` / `_protect_upstream` /
        `add_perception` 去重 / 巩固周期）一律先经各自入口的层判，故本闸对它们
        只做**兜底**，不改变 PRIMARY 侧读数。
        """
        delta = _finite_or(delta, 0.0)
        self.assert_node_writable_by_role(node_id, action="改写共享层节点的 importance")
        c = self.conn.cursor()
        c.execute("UPDATE nodes SET importance = MIN(1.0, MAX(0.0, importance + ?)) WHERE id=?",
                  (delta, node_id))
        self._commit_locked()

    def recalc_structural_importance(self, dry_run: bool = True,
                                     boost_cap: float = 0.15,
                                     beta: float = 0.30, gamma: float = 0.40,
                                     c: float = 6.0,
                                     min_degree: int = 5, min_causal: int = 3,
                                     min_delta: float = 0.02,
                                     max_depth: int = 5,
                                     protect_threshold: float = 1.0,
                                     floor_importance: float = 0.9) -> Dict:
        """结构重要性重算（v2.2，设计规格 §13/§14，2026-08-20）。

        importance_v2 = min(1.0, importance + min(β·min(ci,C)/C + γ·conn, BOOST_CAP))
        只升不降（**原值 + 已施加的结构提升**为地板，延迟提升语义）；ci=因果出度，
        conn=总度数/max_deg；无结构数据（度数/因果双低）的节点不动。

        v2.2 新增【因果上游度传播】——越上游影响越大：
          concept_influence(v) = Σ_{u∈causal_downstream(v), depth≤max_depth}
                                 path_conf × importance(u) / depth
          · influence ≥ protect_threshold → 自动保护（protect_node + no_forget）
            且 importance 保底 floor_importance——越上游越要记录
          · influence 写入 state_attributes.concept_influence（可检索可查）
        dry_run=True 仅报告不写库。

        #269：上游度传播**只统计已验证（`verified=1`）的 causal 边**。旧实现把
        全部 causal 边同权纳入，未验证、来源 inferred/ambiguous、置信度默认 0.5 的
        边与人工复核过的边贡献相同 ⇒ 3 条 `add_edge` 默认参数的边（0.5×0.8×3=1.2）
        就越过 `protect_threshold=1.0`，把一条 importance 0.3 的随口猜想登记为
        `protect_node` + `no_forget` 并抬到 0.9；而 `decay_cycle` 对触及受保护节点的
        边豁免衰减 ⇒ 授予保护的**未验证边本身也永不衰减**（自我封存），且重算只加
        不撤、撤掉全部边后保护仍不撤销（单向棘轮）。docstring 写的前提是「因果结构
        已确认」，旧实现把「结构确认」换成了「边数够多」。
        判据来源：公开 issue #269「建议修复 1」——「`SELECT ... FROM edges WHERE
        verified = 1`（或 `confidence >= 0.7 AND source_evidence = 'extracted'`）
        后再计算 influence」＋ 其最小复现实读（未验证默认边 ⇒ 保护 + 0.9）＋ 分诊表
        `triage_lingshu_ALL.md` 行 269（「不看 verified/confidence；3 条 0.5 边即可越
        1.0 → 永久保护+保底 0.9（单向棘轮）」）。仓内无规定该判据的理论章节（旧
        docstring 引的「设计规格 §13/§14」不在本仓可查面），属经验标定（本件 #269）。
        边界（削掉的判别力，如实标注）：① 本件只收「未验证边不得授予永久保护」这
        一条口径，采用 issue 的**主选项**（`verified=1`），未采纳括号里的
        `confidence>=0.7 AND source_evidence='extracted'` 替代口径——已验证但
        confidence 低（如 0.5）的边仍计入，故「已验证边数量×置信度」累积阈值偏低的
        归一化问题（issue #274，与 #269 同根、单列为重复件）**不在本件**；② 本件
        不做保护撤销（issue 建议 2）与 `decay_cycle` 豁免面收窄（建议 3）：判据修好
        后「未验证边授予保护」这条路径已不存在，而既有已登记的「结构保护」行仍永久
        生效（`protections` 无 TTL，见 `_is_forget_protected` 边界），撤销面属 #257/#261
        的保护名单回收范畴。

        #299：旧实现 `new_imp = min(1.0, imp + boost)` 每次以**当前** importance
        为底重算再累加 ⇒ 「重算」实为「累加」：结构一条边未变，反复调用仍持续
        抬升（5 次把 0.3 推到 1.0），`BOOST_CAP` 形同虚设、函数不幂等。
        修法（最小）：应用时把**本次结构提升总量**写入
        `state_attributes.structural_boost`（复用 `_write_influence` 的 `key`
        参数）；重算时以 `base = imp - structural_boost` 为地板，
        `new_imp = min(1.0, base + boost)`、`delta = new_imp - imp`——结构增强时
        只补差额、结构不变时 delta=0（不动），单节点总结构提升 ≤ `boost_cap`。
        判据来源：公开 issue #299「修复建议（最小补丁）」。
        needs-owner-decision（issue 自列，如实保留）：① 「重算」语义取「相对原值、
        总提升 ≤ BOOST_CAP」——若设计本意是允许跨周期逐步累积（延迟提升），应另
        写明累积上限；② 补丁前已被多次提升的存量节点无 `structural_boost` 标记，
        会以当前值为底（不回滚历史累加），一次性校正属数据迁移决策，本件不做。
        """
        c_ = self.conn.cursor()
        c_.execute("SELECT source_id, target_id, relation_type, confidence, verified "
                   "FROM edges")
        edges = c_.fetchall()
        causal_out: Dict[str, int] = {}
        deg: Dict[str, int] = {}
        causal_adj: Dict[str, list] = {}
        causal_conf: Dict[tuple, float] = {}
        for s, t, rt, cf, vf in edges:
            deg[s] = deg.get(s, 0) + 1
            deg[t] = deg.get(t, 0) + 1
            # #269：只有**已验证**的 causal 边进入上游度传播（未验证/默认 0.5 的
            # inferred 边不得作为永久保护的依据）；度数/出度统计（boost 面）不受
            # 本件影响，仍按全部边计。
            if rt == "causal" and vf:
                causal_out[s] = causal_out.get(s, 0) + 1
                causal_adj.setdefault(s, []).append(t)
                causal_conf[(s, t)] = cf if cf is not None else 0.5
        max_deg = max(deg.values()) if deg else 1
        # ★优化（本笔 · 零行为变更）：把 `state_attributes` **并入这趟主查询**。
        # 旧实现在下方循环里逐节点调 `_read_state_attr(nid, "structural_boost", 0.0)`
        # ⇒ N 次 `SELECT state_attributes FROM nodes WHERE id=?`（N+1）。该列与
        # `id, importance` 同表同行 ⇒ 一并取出、按 `_state_attr_of` 单点解析一次。
        # #769/#780：`layer` 一并取出——下方**逐节点**判「目标行层 × 角色」，避免
        # 每节点一次 `SELECT layer`（与既有 N+1 收口同法）。
        c_.execute("SELECT id, importance, state_attributes, layer FROM nodes")
        rows = c_.fetchall()
        imp_map = {nid: (imp if imp is not None else 0.5)
                   for nid, imp, _sa, _ly in rows}
        _sb_map = {nid: self._state_attr_of(_sa, "structural_boost", 0.0)
                   for nid, _imp, _sa, _ly in rows}

        # 因果上游度传播（BFS · 深度上限 · 节点去重 · 路径置信度加权）
        influence: Dict[str, float] = {}
        for src in causal_adj:
            infl = 0.0
            visited = {src}
            frontier = [(src, 1.0, 0)]
            while frontier:
                node, pc, d = frontier.pop(0)
                for t in causal_adj.get(node, []):
                    if t in visited:
                        continue
                    nd = d + 1
                    if nd > max_depth:
                        continue
                    nconf = pc * causal_conf.get((node, t), 0.5)
                    visited.add(t)
                    infl += nconf * imp_map.get(t, 0.5) / nd
                    frontier.append((t, nconf, nd))
            if infl > 0:
                influence[src] = round(infl, 4)

        movers = []
        protected = []
        skipped_shared = []
        applied = 0
        for nid, imp, _sa, nid_layer in rows:
            infl = influence.get(nid, 0)
            # #769/#780：**逐节点隔离**——非 PRIMARY 时共享层行整体跳过（既不写
            # `state_attributes`、也不抬 `importance`、也不登记 `protections`）。
            # 取向理由：旧实现只在下游 `_protect_upstream` 的 `#234` 闸处抛错，
            # 而 `_write_influence` **已提交** ⇒ 「先写后抛」的部分落库（核验帖
            # 点名的非原子格）。逐节点跳过（而非整链抛错）使：①零部分落库；
            # ②SUB 对**本地层**节点的重算照旧生效（不回归既有语义，`self` 层不在
            # `SHARED_LAYERS`）；③跳过数经 `skipped_shared` 上报，不静默。
            if not self.layer_writable_by_role(nid_layer):
                skipped_shared.append(nid)
                continue
            # 上游度写入 state_attributes（可检索/可查）
            if infl > 0 and not dry_run:
                self._write_influence(nid, infl)
            # 越上游越要记录：结构保护 + importance 保底
            if infl >= protect_threshold:
                if not dry_run:
                    self._protect_upstream(nid, infl, floor_importance)
                protected.append(nid)
                continue  # 上游节点已由结构确认，importance 走保底而非扇出提升
            if imp is None or imp >= 0.95:
                continue
            d = deg.get(nid, 0)
            co = causal_out.get(nid, 0)
            if d < min_degree and co < min_causal:
                continue  # 无结构数据不动（地板语义）
            ci = min(co, c) / c
            conn = d / max_deg
            boost = min(beta * ci + gamma * conn, boost_cap)
            # #299：旧实现 `new_imp = min(1.0, imp + boost)` 每次以**当前**
            # （已被上次提升过的）importance 为底 ⇒ 反复调用持续累加、不收敛，
            # BOOST_CAP 形同虚设（结构不变 5 次把 0.3 推到 1.0）。
            # 修法（最小）：以「原值 + 本次结构目标提升」为准——读回上次已施加的
            # structural_boost（`sb`）作地板 `base = imp - sb`，`new_imp =
            # min(1.0, base + boost)`，`delta = new_imp - imp`；结构不变时
            # delta=0（不动），结构增强时只补差额，总提升 ≤ boost_cap。
            # 判据来源：公开 issue #299「修复建议（最小补丁）」。
            # ★优化（本笔）：`sb` 取自上方主查询一并取回并解析好的 `_sb_map`
            # （旧实现此处调 `_read_state_attr` ⇒ 每节点一次 SELECT，N+1）。
            # 取值口径逐位不变：同一个 `_state_attr_of` 单点、同一个默认 0.0。
            sb = _sb_map.get(nid, 0.0)
            sb = sb if isinstance(sb, (int, float)) else 0.0
            base = imp - sb
            new_imp = min(1.0, base + boost)
            delta = round(new_imp - imp, 4)
            if delta >= min_delta:
                movers.append({"node_id": nid, "importance_old": round(imp, 4),
                               "importance_new": round(new_imp, 4),
                               "degree": d, "causal_out": co, "boost": delta,
                               "concept_influence": infl})
                if not dry_run:
                    self.update_node_importance(nid, delta)
                    self._write_influence(nid, round(sb + delta, 6),
                                          key="structural_boost")
                    applied += 1
        movers.sort(key=lambda x: x["boost"], reverse=True)
        top_influence = sorted(influence.items(), key=lambda x: -x[1])[:10]
        return {"mode": "dry_run" if dry_run else "applied",
                "params": {"beta": beta, "gamma": gamma, "c": c,
                           "boost_cap": boost_cap, "min_degree": min_degree,
                           "min_causal": min_causal, "min_delta": min_delta,
                           "max_depth": max_depth,
                           "protect_threshold": protect_threshold,
                           "floor_importance": floor_importance},
                "edges_total": len(edges), "max_deg": max_deg,
                "candidates": len(movers), "applied": applied,
                "influence_nodes": len(influence),
                "protected": len(protected),
                # #769/#780：非 PRIMARY 时被跳过的共享层节点数（逐节点隔离 ⇒
                # 零部分落库；上报而不静默）。
                "skipped_shared": len(skipped_shared),
                "skipped_shared_nodes": skipped_shared[:50],
                "top_influence": [{"node_id": n, "concept_influence": v} for n, v in top_influence],
                "movers": movers[:50]}

    @_transactional
    def _write_influence(self, node_id: str, infl: float,
                         key: str = "concept_influence"):
        """把结构量写入 state_attributes（可检索/可查）。

        #299：新增 `key` 参数（默认 `concept_influence`，旧行为不变），供
        `recalc_structural_importance` 复用同一通道写入 `structural_boost`。

        #780 ① / #769 ③：本方法此前**无层/角色判据**（核验帖实测：SUB 直调
        `_write_influence(锚点, 7.77)` 即改写共享层行的 `state_attributes`，且该列被
        下游 `_state_attr_of` 当结构化真值读回 ⇒ 篡改会改变后续结构提升基准）。
        现按 `add_node` 的同款单点判据**写前按库中旧行**判权限：非 PRIMARY 且目标行
        属 `SHARED_LAYERS` ⇒ `PermissionError`（闸在 SELECT/UPDATE **之前** ⇒
        被拒时零落库）。共享层节点的 `state_attributes` 是治理列，由父节点主控；
        SELF/本地层不在 `SHARED_LAYERS`，照旧可写。
        """
        self.assert_node_writable_by_role(
            node_id, action="改写共享层节点的 state_attributes")
        c = self.conn.cursor()
        c.execute("SELECT state_attributes FROM nodes WHERE id=?", (node_id,))
        row = c.fetchone()
        sa = {}
        if row and row[0]:
            try:
                sa = json.loads(row[0])
            except Exception:
                sa = {}
        sa[key] = infl
        c.execute("UPDATE nodes SET state_attributes=? WHERE id=?",
                  (_dumps_json(sa, ensure_ascii=False, where="state_attributes"), node_id))
        self._commit_locked()

    @staticmethod
    def _state_attr_of(sa_raw, key: str, default=None):
        """从 `state_attributes` 原始文本里取一个键（缺值/非法 JSON/非 dict ⇒ default）。

        ★本笔把它从 `_read_state_attr` 里**提出来成单点**：`recalc_structural_importance`
        原先在循环里逐节点调 `_read_state_attr` ⇒ **N 次 SELECT**。而该列与它上方那趟
        `SELECT id, importance FROM nodes` **同表同行** —— 列一并取出后在此**解析一次**即可。
        两条路径（单点查询 / 批量取出）共用本函数 ⇒ 容错口径不可能分叉。
        """
        if not sa_raw:
            return default
        try:
            sa = json.loads(sa_raw)
        except (ValueError, TypeError):
            # 窄口径（#170 建议 b）：`json.loads` 只抛这两类——`JSONDecodeError`
            # 是 `ValueError` 子类，入参非 str/bytes 抛 `TypeError`。本处「坏档当
            # 缺键」是刻意静默（读一个可选标记，不该因脏数据中断重算），故不记
            # 审计行；但**不写成宽 `except Exception`**——那会把真正的编程错误
            # （如 cursor 用法错）一并吞成默认值。
            return default
        if not isinstance(sa, dict):
            return default
        return sa.get(key, default)

    def _read_state_attr(self, node_id: str, key: str, default=None):
        """读单个 state_attributes 键（缺键/非法 JSON/非 dict ⇒ 返回 default）。

        #299：`recalc_structural_importance` 需要读回上次已施加的
        `structural_boost` 作为地板，故加此小 helper。

        ★本笔：解析体已抽到 `_state_attr_of` 单点；本函数保留为「已知 node_id 时」
        的便捷入口（一次 SELECT）。**批量场景请勿逐节点调本函数**——见
        `recalc_structural_importance` 里「列随主查询一并取出」的用法。
        """
        c = self.conn.cursor()
        c.execute("SELECT state_attributes FROM nodes WHERE id=?", (node_id,))
        row = c.fetchone()
        if not row:
            return default
        return self._state_attr_of(row[0], key, default)

    def _protect_upstream(self, node_id: str, infl: float, floor: float):
        """越上游越要记录：保护登记 + no_forget + importance 保底。

        #780 ①：本助手此前零角色判据（其内部 `protect_node` 有 `#234` 闸，但闸抛在
        `_write_influence` **已提交之后** ⇒ 整次 recalc「先写后抛」、部分落库）。
        现在**入口先判**（`assert_node_writable_by_role`）：非 PRIMARY 且目标行属
        共享层 ⇒ 在任何写之前 `PermissionError`（零落库）；共享层节点的保护登记由
        父节点主控（与 `protect_node` 的 `#234` 判据同源，此处只是把它提到最前）。
        """
        self.assert_node_writable_by_role(node_id, action="保护共享层节点")
        self.protect_node(node_id, f"结构保护:概念影响度 {infl}（因果上游度≥阈值，2026-08-20）")
        node = self.get_node(node_id)
        if node and node.importance is not None and node.importance < floor:
            self.update_node_importance(node_id, round(floor - node.importance, 4))

    # ==================== 洞察条件层（v1.0 引擎化 · 设计规格 lingshu-insight-layer-design.md） ====================

    INSIGHT_DEFAULT_CONDITIONS = {
        "memory_retrievability": 0.5,   # C1 记忆可得性
        "outside_observer": "none",      # C2 外部观察者
        "cross_domain": [],              # C3 跨域映射
        "premise_questioned": False,     # C4 前提质疑
        "pressure": "medium",            # C5 反思空间压力
        "continuity_turns": 1,           # C6 连续性
        "externalized": False,           # C7 外部化
        "tone": "calm",                  # C8 情绪基调（P0-3 独立通道）
    }

    def _normalize_conditions(self, conditions: dict) -> dict:
        """规范化条件快照（C1–C8 + confidence）：显式提供=0.7，默认值=0.3（未自动提取）。"""
        vals = dict(self.INSIGHT_DEFAULT_CONDITIONS)
        given = set()
        if conditions:
            for k, v in conditions.items():
                if k in vals:
                    vals[k] = v
                    given.add(k)
        conf = {k: (0.7 if k in given else 0.3) for k in vals}
        return {"values": vals, "confidence": conf}

    def insight_record(self, content: str, conditions: dict = None,
                       source: str = "", importance: float = 0.7) -> Dict:
        """记录洞见事件：insight_event 节点 + 条件快照（C1–C8）+ verification=pending。

        conditions 可显式提供（自动提取待 I2 接入）；未提供时用中性默认 + 低置信。
        """
        now = time.time()
        nid = f"insight_{uuid.uuid4().hex[:8]}_{int(now * 1000)}"
        cond_pack = self._normalize_conditions(conditions)
        sa = {"insight": {
            "conditions": cond_pack,
            "verification": {"level": None, "evidence": [], "status": "pending", "updated_at": now},
            "source": source,
        }}
        node = STNode(
            id=nid, content=f"【洞见事件】{content}", modality="text",
            spatial_coordinates={}, temporal_coordinate=now,
            condition_space=ConditionSpace(
                "协议实例·认知循环", "洞察条件层记录",
                (now, now + 3600), "协议实例运行中"),
            importance=importance, confidence=0.5,
            layer=MemoryLayer.KNOWLEDGE,
            access_count=0, last_access=now, created_at=now,
            tags=["insight_event", "pending", "insight-layer"],
            semantic_coordinates={}, state_attributes=sa,  # dict（to_row 自行序列化）
            entity_id=None,
        )
        self.add_node(node)
        return {"node_id": nid, "status": "pending",
                "conditions": cond_pack["values"], "importance": importance}

    @_transactional
    def insight_verify(self, insight_id: str, level: str = "V2",
                       evidence: object = None) -> Dict:
        """提交验证证据（§4.3）：**零证据一律 `pending`**；有证据时按等级判：
        V2/V3（自报等级）≥1 条即 `verified`，V1 需 ≥3 条（多次复现）。

        evidence 可为字符串或字符串列表（追加）。

        lingshu #38：旧实现 `if level in ("V2", "V3"): status = "verified"`
        ——**完全不看证据**，于是「等级越高越严」实际反成「V2 比 V1 宽松」：
        V1 要 ≥3 条证据，V2/V3 **零证据**即自证 `verified`，随即 importance 抬到
        0.9 且标签带 `verified`（免衰减、不可逆），是一条**自证通道**。
        修法（最小）：`V2/V3` 与 `V1` 共用同一条**证据门槛前置**判据——证据数
        为 0 时无论何等级一律 `pending`；证据数 ≥1 时才认 V2/V3 的 `verified`。
        判据来源：`docs/plans/待裁清单_v0.1.md` **B-9**「`#38` 洞见**最低证据
        门槛** ⇒ **零证据一律 `pending`**（V2/V3 不得自报即 `verified`）」＋
        **D-34** 同款推荐；**D-35**「V2/V3 视为**自报等级**、与 `verified` 解耦」
        （故本件只加证据门槛，**不**把 V2/V3 重定义为「已获独立复核」）。
        importance 抬升（0.9）**只在 `status == "verified"` 时发生**，而 verified
        现已蕴含证据数 ≥1 ⇒ 满足 **D-36**「0 证据不自抬」，无需另加分支。
        削掉的判别力：**V2/V3 的「自报等级」不再单独构成 `verified`**——调用方
        必须至少给一条证据；本件不引入「独立复核人」判据（D-35 明确 V2/V3 是
        自报等级，不要求外部复核面）。

        #301：重写标签时剥离清单只含小写 `refuted`，漏剥本函数自己写入的大写
        级别（`REFUTED`/`X`/任意调用方级别）与 `insight_event` 自身 ⇒ 反复验证
        会累积重复标签、且已验证洞见仍挂旧 `REFUTED`（按标签取证伪的查询会误判）。
        修法（最小）：剥离集合补 `insight_event` 与上一轮记录的级别 `prev_level`。
        判据来源：公开 issue #301「修复建议（最小补丁）」——「剥离集合改为
        `{pending,verified,refuted,V1,V2,V3,insight_event,prev_level}`」。状态判定、
        importance 提升逻辑均不变。

        #384：证据去重只与「合并前的旧列表」比，不在**本次传入的 `add` 内部**
        去重 ⇒ 传 `['x','x','x']` 即得 `evidence_count=3`，误判 V1 `verified`、
        importance 直升 0.9（而分 3 次提交同一句仍是 pending）。修法（最小）：
        逐条与**已合并结果**比后再追加（保持出现顺序），两处重复一并覆盖。
        判据来源：公开 issue #384 标题所述最小复现（`['x','x','x']` ⇒ verified）。
        """
        node = self.get_node(insight_id)
        if node is None or "insight_event" not in node.tags:
            return {"error": f"非洞见事件节点: {insight_id}"}
        # #519：共享层角色闸——洞见经 D-003 终裁晋升到结构层/锚点层后，节点
        # 仍带 insight_event 标签；旧实现只认标签不认层与角色 ⇒ SUB 可把已终裁
        # 的结构层节点 verified→refuted、追加伪造证据、抬 importance 到 0.9。
        # 与 #234 给 tag_node/protect_node/register_conflict/add_edge 加的闸同款。
        # ★本笔（第 4 项 · 消除第二份实现）：判据改为**委托**单点
        # `layer_writable_by_role`（G 线本笔落地的唯一真源）——原内联形态
        # `node.layer in self.SHARED_LAYERS and self.role != Role.PRIMARY` 与单点
        # 判据同源，但属**第二份实现**。零行为差异已实测（同一批构造 × 两角色，
        # 返回/抛错与调用后行快照逐字相同）。
        if not self.layer_writable_by_role(node.layer):
            raise PermissionError(
                f"role={self.role.value} 无权改写共享层（{node.layer.value}）的洞见"
                f"（id={insight_id}）——已终裁晋升的洞见归 PRIMARY 主控（#519）")
        now = time.time()
        sa = node.state_attributes or {}
        ins = sa.get("insight", {}) if isinstance(sa, dict) else {}
        ver = ins.get("verification", {}) if isinstance(ins, dict) else {}
        ev = list(ver.get("evidence", []))
        if evidence:
            add = evidence if isinstance(evidence, list) else [evidence]
            # #384：只与合并前的旧列表比 ⇒ 本次传入的 add 内部重复不去重，
            # 传 ['x','x','x'] 即得 evidence_count=3 ⇒ 误判 V1 verified、
            # importance 直升 0.9。改为「与已合并结果比」逐条去重（保持出现顺序），
            # 同时覆盖旧列表与本次提交内部两处重复。
            for e in add:
                if e not in ev:
                    ev.append(e)
        level = (level or "").upper()
        status = "pending"
        if level in ("REFUTED", "X"):
            status = "refuted"
        elif level == "V1":
            status = "verified" if len(ev) >= 3 else "pending"
        elif level in ("V2", "V3"):
            # #38：零证据不得自报 verified（B-9/D-34）；有证据才认自报等级（D-35）。
            status = "verified" if len(ev) >= 1 else "pending"
        # #301：剥除清单漏了本函数自己写入的大写级别（REFUTED/X/任意自定义级别）
        # 与 insight_event 自身 ⇒ 反复验证会累积重复标签、残留旧级别。剥离集合
        # 加上 "insight_event" 与上一轮记录的级别（prev_level），覆盖任意自定义级别。
        prev_level = ver.get("level") if isinstance(ver, dict) else None
        ver = {"level": level, "evidence": ev, "status": status, "updated_at": now}
        ins["verification"] = ver
        sa["insight"] = ins
        _strip = {"pending", "verified", "refuted", "V1", "V2", "V3", "insight_event"}
        if prev_level:
            _strip.add(prev_level)
        tags = [t for t in node.tags if t not in _strip]
        tags += ["insight_event", status, level]
        new_imp = node.importance
        if status == "verified" and (new_imp is None or new_imp < 0.9):
            new_imp = 0.9
        c = self.conn.cursor()
        c.execute("UPDATE nodes SET state_attributes=?, tags=?, importance=? WHERE id=?",
                  (_dumps_json(sa, ensure_ascii=False, where="state_attributes"),
                   _dumps_tags(tags, ensure_ascii=False),
                   new_imp, insight_id))
        self._commit_locked()
        return {"node_id": insight_id, "level": level, "status": status,
                "evidence_count": len(ev), "importance": new_imp}

    def insight_report(self, window: int = None) -> Dict:
        """CER 报告：条件有效洞见率（样本<20 不判定 · 2×SE 显著性 · P0-4 框架复用）。

        候选=全部 insight_event 节点；验证状态优先读 state_attributes，
        旧样本（无结构化 verification）按 tags 兜底（verified/refuted/pending）。

        #281：条件统计旧实现按 `x["conditions"].get(k)` 的**真值**分组，把
        「键存在但值为假」（`False`/`0`/`""`）与「键不存在」（未记录该条件）混同
        ——未测样本被当作「该条件为假」计入 `without` 组，统计口径失真。上一轮
        （`f0a9dc7`）只按「键存在」收窄，但**真实路径零效果**：`_normalize_conditions`
        给**未显式提供**的条件也填了默认值 ⇒ 经 `insight_record` 写入的节点
        C1–C8 键**恒存在**，未显式提供的条件仍按其**默认值**落进 `with_`/`without`
        （`premise_questioned`/`externalized`/`cross_domain` 默认为假 ⇒ 混入
        `without`；`pressure`/`tone` 默认为非空字符串 ⇒ 恒真混入 `with_`），
        统计口径失真（实测 `n_without=15 / cer_without=0.667`，预期 `5 / 0.0`）。
        修法（最小）：判据由「键存在」改为「**显式提供**」——`_normalize_conditions`
        早已把「显式提供=0.7／默认回落=0.3」写进 `conditions.confidence`，本件
        只是**把它读出来**：`confidence[k] >= 0.7` 才算 present。无 `confidence`
        标记时（旧库样本／裸 SQL 构造）**回落**旧判据「键存在」，以免旧样本变成
        新红。判据来源：公开 issue #281「修复建议 1」——「未显式提供（confidence=0.3）
        的条件不进入对应统计」。
        边界（削掉的判别力，如实标注）：本件只修「显式提供 vs 默认回落」的混同，
        **不**改 `_normalize_conditions` 给未提供条件填默认值的行为（写入面不变）；
        `pressure`/`tone` 按取值分组（issue #281 根因 2/3）**不在本件**。
        `pending = n - nv` 把 refuted 计入 pending、以及 REFUTED 后 importance 不
        回落（同 issue 附带两处）亦**不在本件**。
        """
        c = self.conn.cursor()
        c.execute("SELECT id, tags, state_attributes FROM nodes WHERE tags LIKE '%insight_event%'")
        rows = c.fetchall()
        candidates = []
        for nid, tags_str, sa_str in rows:
            tags = json.loads(tags_str) if isinstance(tags_str, str) and tags_str else []
            sa = json.loads(sa_str) if isinstance(sa_str, str) and sa_str else {}
            ins = sa.get("insight", {}) if isinstance(sa, dict) else {}
            ver = ins.get("verification", {}) if isinstance(ins, dict) else {}
            status = ver.get("status")
            if not status:
                status = ("verified" if any("verified" in t for t in tags)
                          else "refuted" if "refuted" in tags else "pending")
            conditions = (ins.get("conditions") or {}).get("values", {}) if isinstance(ins, dict) else {}
            # #281：`conditions.confidence`（`_normalize_conditions` 写入：显式提供
            # =0.7／默认回落=0.3）一并带出——判据在数据里，旧实现没读它。
            confidence = (ins.get("conditions") or {}).get("confidence", {}) if isinstance(ins, dict) else {}
            candidates.append({"id": nid, "status": status, "conditions": conditions,
                               "confidence": confidence})
        n = len(candidates)
        nv = sum(1 for x in candidates if x["status"] == "verified")
        cer = nv / n if n else 0.0
        se = math.sqrt(cer * (1 - cer) / n) if n else 0.0
        layer_status = "degraded" if n < 20 else ("watch" if n < 40 else "reliable")
        cond_stats = []
        for k in ("premise_questioned", "externalized", "pressure", "tone", "cross_domain"):
            # #281：旧实现 `x["conditions"].get(k)` 按**真值**分组，把「键存在但值为
            # 假」（False/0/""）与「键不存在」（未记录该条件）混同——两者都落进
            # without 组 ⇒ 未测的条件被当作「该条件为假」的样本，统计口径失真。
            # 判据改为「**显式提供**」：`confidence[k] >= 0.7`（`_normalize_conditions`
            # 的口径：显式提供=0.7／默认回落=0.3）；无 `confidence` 标记（旧库样本／
            # 裸 SQL 构造）时**回落**旧判据「键存在」，旧样本不至变成新红。
            def _explicit(x, _k=k):
                _conf = x.get("confidence")
                if isinstance(_conf, dict) and _k in _conf:
                    try:
                        return float(_conf.get(_k) or 0.0) >= 0.7
                    except (TypeError, ValueError):
                        return False
                return _k in (x["conditions"] or {})     # 回落：旧库样本 / 裸 SQL 构造
            present = [x for x in candidates if _explicit(x)]
            with_ = [x for x in present if x["conditions"].get(k)]
            without = [x for x in present if not x["conditions"].get(k)]
            if len(with_) >= 5 and len(without) >= 5:
                p1 = sum(1 for x in with_ if x["status"] == "verified") / len(with_)
                p0 = sum(1 for x in without if x["status"] == "verified") / len(without)
                se1 = math.sqrt(max(p1 * (1 - p1) / len(with_), 1e-4))
                se0 = math.sqrt(max(p0 * (1 - p0) / len(without), 1e-4))
                diff = p1 - p0
                sig = abs(diff) >= 2 * math.sqrt(se1 ** 2 + se0 ** 2)
                cond_stats.append({
                    "condition": k, "n_with": len(with_), "n_without": len(without),
                    "cer_with": round(p1, 3), "cer_without": round(p0, 3),
                    "diff": round(diff, 3), "significant": sig,
                    "judge": "确认" if (sig and diff > 0) else ("反证" if sig else "不判定")})
        return {"total": n, "verified": nv, "pending": n - nv,
                "cer": round(cer, 3), "se": round(se, 4),
                "layer_status": layer_status,
                "note": "样本<20 不判定（与迁移测试同标准）" if n < 20 else "样本达标，可出显著性",
                "conditions": cond_stats,
                "sample_ids": [x["id"] for x in candidates]}

    def insight_window(self, conditions: dict = None) -> Dict:
        """当前洞察窗口检测。默认假设（待统计确认）：C1≥0.6 ∧ 跨域 ∧ 低压力 → 窗口开。

        conditions 传入当前会话快照；缺省字段不参与判定。
        """
        cond = conditions or {}
        c1 = cond.get("memory_retrievability")
        c3 = cond.get("cross_domain")
        c5 = cond.get("pressure")
        open_ = ((c1 is None or c1 >= 0.6)
                 and (c3 is None or (isinstance(c3, list) and len(c3) > 0))
                 and (c5 is None or c5 == "low"))
        return {"window_open": open_,
                "assumption": "默认假设 C1≥0.6 ∧ 跨域 ∧ 低压力（未经统计确认）",
                "conditions": {"memory_retrievability": c1, "cross_domain": c3, "pressure": c5}}

    @_transactional
    def append_skill_procedure(self, skill_id: str, step: str):
        """技能程序追加（P1-3 技能获取：版本+1）"""
        c = self.conn.cursor()
        c.execute("SELECT procedure FROM skills WHERE id=?", (skill_id,))
        row = c.fetchone()
        if not row:
            return
        proc = row[0]
        # #200：同步骤已在程序中 ⇒ 不再追加（旧实现每次成功都把步骤再拼一遍，
        # 程序随调用次数无界增长）。
        if step and step.strip() in [s.strip() for s in (proc or "").split("→")]:
            return
        merged = f"{proc} → {step}" if proc else step
        c.execute("UPDATE skills SET procedure=?, version=version+1, updated_at=? WHERE id=?",
                  (merged, time.time(), skill_id))
        self._commit_locked()

    # ==================== v1.5 扩展（A-1/A-2/A-3） ====================

    @_transactional
    def add_rejected_path(self, path_type: str, description: str, reason: str,
                          evidence: str = "") -> str:
        """登记被否决的路径（负路由沉淀：错误答案留档防重蹈），返回 id。"""
        rid = f"rp_{uuid.uuid4().hex[:8]}"
        c = self.conn.cursor()
        c.execute("INSERT INTO rejected_paths VALUES (?,?,?,?,?,?,?,?)",
                  (rid, path_type, description, reason, evidence, "open", time.time(), None))
        self._commit_locked()
        return rid

    def list_rejected_paths(self, status: str = None) -> List[Dict]:
        """列出否决路径（可按 status=open/consumed 过滤）。"""
        c = self.conn.cursor()
        if status:
            c.execute("SELECT * FROM rejected_paths WHERE status=?", (status,))
        else:
            c.execute("SELECT * FROM rejected_paths")
        return [{"id": r[0], "path_type": r[1], "description": r[2], "reason": r[3],
                 "evidence": r[4], "status": r[5], "created_at": r[6], "consumed_at": r[7]}
                for r in c.fetchall()]

    @_transactional
    def mark_rejected_path_consumed(self, rejected_id: str,
                                      designer_key: str = None):
        """标记否决路径已消费（同类问题复用该否决答案后）。

        #522：消费否决行＝解除 #216 负记忆闸——旧实现无密钥无角色检查，
        任何人调一下就把 status='open' 翻成 'consumed'，重提闸即被绕过。
        现须过设计者密钥（fail-closed，与 resolve_blindspot 同款）。
        """
        if not verify_designer(designer_key):
            raise PermissionError(
                "#522 消费否决路径须过设计者密钥闸（fail-closed）——"
                "消费＝解除 #216 负记忆闸，不得无密钥调用")
        c = self.conn.cursor()
        c.execute("UPDATE rejected_paths SET status='consumed', consumed_at=? WHERE id=?",
                  (time.time(), rejected_id))
        self._commit_locked()

    @_transactional
    def add_verifier_standard(self, name: str, param: str, value: float, reason: str,
                              proposer: str) -> str:
        """登记校验器标准参数提案（阈值类修订的可追溯载体），返回 id。"""
        vid = f"vs_{uuid.uuid4().hex[:8]}"
        c = self.conn.cursor()
        c.execute("INSERT INTO verifier_standards VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                  (vid, name, param, value, reason, proposer, "", "", "", "pending",
                   time.time(), None))
        self._commit_locked()
        return vid

    def list_verifier_standards(self, status: str = None) -> List[Dict]:
        """列出校验器标准提案（可按 status 过滤——阈值治理台账）。"""
        c = self.conn.cursor()
        if status:
            c.execute("SELECT * FROM verifier_standards WHERE status=?", (status,))
        else:
            c.execute("SELECT * FROM verifier_standards")
        return [{"id": r[0], "name": r[1], "param": r[2], "value": r[3], "reason": r[4],
                 "proposer": r[5], "independent_reviewer": r[6], "cs_reviewer": r[7],
                 "adjudicator": r[8], "status": r[9], "created_at": r[10], "decided_at": r[11]}
                for r in c.fetchall()]

    @_transactional
    def review_verifier_standard(self, vid: str, reviewer: str, approved: bool) -> bool:
        """独立复核（验证单元自我回避）

        #118：补**身份互斥**——`reviewer` 不得等于 `proposer`（提议人不得自兼独立
        复核）。旧实现只查状态门，不查角色 ⇒ 「独立复核」名不副实（可自提自审）。
        判据来源：`docs/plans/待裁清单_v0.1.md` **C-4 / D-02**（三条流水线身份
        两两互斥）＋ 设计稿 `docs/plans/治理与写入边界_修复设计_v0.1.md` §4
        修法形状 3。返回 False = 被状态门/身份门拒绝。
        """
        c = self.conn.cursor()
        c.execute("SELECT status, proposer FROM verifier_standards WHERE id=?", (vid,))
        row = c.fetchone()
        if not row or row[0] != "pending":
            return False
        if reviewer == row[1]:
            return False  # 身份互斥：提议人不得自兼独立复核
        status = "reviewed" if approved else "rejected"
        c.execute("UPDATE verifier_standards SET independent_reviewer=?, status=? WHERE id=?",
                  (reviewer, status, vid))
        self._commit_locked()
        return True

    @_transactional
    def cs_review_verifier_standard(self, vid: str, reviewer: str, approved: bool) -> bool:
        """条件空间复核（3.3.1 节）

        #118：身份互斥——`reviewer` 不得等于 `proposer` 或 `independent_reviewer`
        （与提议人、独立复核人两两互斥）。判据来源同上（C-4 / D-02；设计稿 §4-3）。
        """
        c = self.conn.cursor()
        c.execute("SELECT status, proposer, independent_reviewer FROM verifier_standards"
                  " WHERE id=?", (vid,))
        row = c.fetchone()
        if not row or row[0] != "reviewed":
            return False
        if reviewer in (row[1], row[2]):
            return False  # 身份互斥
        status = "cs_approved" if approved else "cs_rejected"
        c.execute("UPDATE verifier_standards SET cs_reviewer=?, status=? WHERE id=?",
                  (reviewer, status, vid))
        self._commit_locked()
        return True

    # 无 @_transactional：#429 起本方法要么由调用方经 `_cursor` 传入事务，要么自开
    # `_structure_event_transaction` 工作单元。挂 @_transactional 会在 `_tx()` 出口
    # **提前 commit** 调用方尚未完成的事务（实测：裁决 UPDATE 先落盘，随后外层回滚
    # 已无效）。事务边界见 `_structure_event_transaction`。
    def adjudicate_verifier_standard(self, vid: str, adjudicator: str,
                                     approved: bool, designer_key: str = None,
                                     *, _cursor=None) -> Optional[Dict]:
        """维生系统终裁（D-007 需设计者密钥）：仅 cs_approved（独立复核+条件空间复核通过）可终裁（A-2 制衡）

        #118：身份互斥——`adjudicator` 不得等于 `proposer` / `independent_reviewer` /
        `cs_reviewer`（终裁人与前三步两两互斥）。旧实现只验密钥不验角色 ⇒ 同一
        身份可走完四步「制衡」。判据来源：待裁清单 **C-4 / D-02**；设计稿 §4-3。

        PR #429：`_cursor` 允许调用方把裁决、持久配置与留史收进同一工作单元。
        """
        if not verify_designer(designer_key):
            raise PermissionError("D-007 设计者认证失败：密钥无效或未配置 AEIS_DESIGNER_KEY（fail-closed）")
        if _cursor is None:
            with self._structure_event_transaction() as cursor:
                return self.adjudicate_verifier_standard(
                    vid, adjudicator, approved, designer_key, _cursor=cursor)
        _cursor.execute("SELECT * FROM verifier_standards WHERE id=?", (vid,))
        row = _cursor.fetchone()
        if not row or row[9] != "cs_approved":
            return None
        if adjudicator in (row[5], row[6], row[7]):
            return None  # 身份互斥：终裁人与提议/独立复核/条件空间复核同人
        status = "approved" if approved else "denied"
        _cursor.execute(
            "UPDATE verifier_standards SET adjudicator=?,status=?,decided_at=? WHERE id=?",
            (adjudicator, status, time.time(), vid))
        return {"id": row[0], "param": row[2], "value": row[3], "status": status}


    def list_escalation_points(self, enabled_only: bool = True) -> List[Dict]:
        """列升级点（危机触发器清单，默认仅已启用的）。"""
        c = self.conn.cursor()
        if enabled_only:
            c.execute("SELECT * FROM escalation_points WHERE enabled=1")
        else:
            c.execute("SELECT * FROM escalation_points")
        # #656：trigger/condition 允许为 NULL，原样回填会让下游
        # `check_escalation` 的 `in` 抛
        # TypeError: argument of type 'NoneType' is not iterable
        # 归一口径与同行 `enabled: bool(r[6])` 一致（见 issue #656）。
        return [{"id": r[0], "code": r[1], "trigger": r[2] or "", "condition": r[3] or "",
                 "action": r[4], "severity": r[5], "enabled": bool(r[6]), "created_at": r[7]}
                for r in c.fetchall()]

    @_transactional
    def _insert_escalation_point(self, code: str, trigger: str, condition: str,
                                 action: str, severity: str) -> str:
        """升级点表的**裸写**（无授权闸），返回 id。

        ★只供 `SpacetimeMemoryEngine._seed_escalation_points` 的**构造期播种**
        使用（#749 的边界②，见 `add_escalation_point` docstring）：播种的 6 条是
        源码里的常量（`seeds` 字面量），属程序初始化，不是运行期治理写入。把它与
        「运行期由调用方注入任意危机触发判据」分开，才不会让
        `SpacetimeMemoryEngine(...)` 的构造在未配置密钥的进程里抛
        `PermissionError`（那会让全仓构造期调用面集体变红）。
        **非播种路径不得调用本方法**——它绕过 `verify_designer`。
        """
        eid = f"esc_{uuid.uuid4().hex[:8]}"
        c = self.conn.cursor()
        c.execute("INSERT INTO escalation_points VALUES (?,?,?,?,?,?,?,?)",
                  (eid, code, trigger, condition, action, severity, 1, time.time()))
        self._commit_locked()
        return eid

    @_transactional
    def add_escalation_point(self, code: str, trigger: str, condition: str,
                             action: str, severity: str = "medium",
                             designer_key: str = None) -> str:
        """登记升级点（危机触发器：触发条件→处置动作），返回 id。

        issue #749：本方法此前**无授权、无审计**——而同族 `set_escalation_enabled`
        早在 #130 就补齐了 `verify_designer` fail-closed 闸 + `action_logs` 留痕。
        两者同写 `escalation_points` 表、同属治理面，**口径矛盾** ⇒ 无密钥即可
        注入 `condition` 判据（叠加 #747 的解析缺陷可反转触发语义）。现补**同位
        同形**：`designer_key` 形参 + `verify_designer` 闸（句面与
        `adjudicate_promotion` / `set_escalation_enabled` 一致）+ `log_action` 审计。

        边界（削掉的判别力，如实标注）：
        ① 闸只认进程环境变量 `AEIS_DESIGNER_KEY`（`verify_designer` 既有口径，
           **不**新增密钥通道、**不**放宽）。
        ② 构造期播种（`_seed_escalation_points`，源码常量 6 条）**不经本方法**，
           走 `_insert_escalation_point` 裸写 ⇒ 未配置密钥的进程仍可正常构造引擎。
           这是「初始化」与「运行期治理写」的分离，不是闸的漏洞：播种内容不可由
           调用方控制。
        ③ **未**扩张到同表其它写点：`import_all`（备份恢复）经 `M13_TABLES` 含
           `escalation_points` 直写 `conn`，**绕过本闸**——该面属既有登记项 #125
           （「恢复路径绕过写入边界」）的范围，本件不动。**后续进展**（R16-125-esc，
           2026-10-16）：`import_all` 在该表上的**同 id 覆写面**已收窄（既有 id 行
           不写、跳过数经 `skipped_existing_rows` 上报），但「**无密钥新增行**」
           那一半仍在（属设计取舍/决策面，见 `import_all` docstring 末段）；
           `set_escalation_enabled` 已有闸（#130）；读面（`list_escalation_points`）
           不加闸。
        """
        if not verify_designer(designer_key):
            raise PermissionError(
                "D-007 设计者认证失败：密钥无效或未配置 AEIS_DESIGNER_KEY（fail-closed）")
        eid = self._insert_escalation_point(code, trigger, condition, action, severity)
        self.log_action(
            "escalation_add",
            f"add escalation {eid} code={code} severity={severity}",
            outcome={"escalation_id": eid, "code": code, "severity": severity})
        return eid

    @_transactional
    def log_action(self, action_type: str, summary: str = "",
                   node_ids: list = None, outcome: dict = None,
                   context: dict = None) -> None:
        """落一条行为审计（v1.14 `action_logs` 表先例，M13 结构对照内）。

        与 `SpacetimeMemoryEngine._note_action` 的分工：后者委托可选组件
        自我认知引擎（未装配即静默 no-op），**不构成可靠留痕**；本方法直接
        写库，作为治理动作的持久审计兜底（跨进程、可查）。"""
        c = self.conn.cursor()
        c.execute("INSERT INTO action_logs"
                  " (ts, action_type, summary, node_ids, outcome, context)"
                  " VALUES (?,?,?,?,?,?)",
                  (time.time(), action_type, summary,
                   json.dumps(node_ids or [], ensure_ascii=False),
                   json.dumps(outcome or {}, ensure_ascii=False),
                   json.dumps(context or {}, ensure_ascii=False)))
        self._commit_locked()

    @_transactional
    def prune_action_logs(self, keep_rows: int = None) -> int:
        """回收治理审计表 `action_logs`：只保留最近 `keep_rows` 行
        （按**插入序** `rowid` 降序，本表 `id` 为 `INTEGER PRIMARY KEY AUTOINCREMENT`
        ⇒ `id` 序即插入序），返回删除条数。

        #723：本表此前**只追加、无任何回收面**（DDL 无上限、全仓无
        `DELETE FROM action_logs`）——而高频写点有三：`consolidate_cycle` 尾部
        （`run_maintenance_cycle` 每周期一行）、`_strip_incident_edges` 审计
        （每条不可变层边剥离一行）、`set_escalation_enabled`。长程部署
        （MCP 3 进程 + 心跳 + 睡眠巩固默认 60s）按每维护周期一行估算年化约
        53 万行/进程。本方法补齐**显式**回收面，形态沿用同族先例
        `prune_self_snapshots`（「只留最近 keep 条、返回删除条数」）。
        ★**刻意不自动调用**：本表在多处 docstring 与守卫中被声明为「审计只追加
        （`action_logs` 无更新/删除路径）」（`log_action` / `import_all` 的 #125
        #479 硬配套 / `_strip_incident_edges`），且 `import_all` 把**带 id 的
        `action_logs` 行**视为需密钥治理载荷。自动裁剪会与「不可篡改审计」这一
        声明契约直接冲突（保留窗口取多少、审计留存期多长属治理策略）⇒ **自动
        调用与保留窗口的取值属设计者裁面**，本件只提供可显式调用的原语，不改任何
        既有写路径的行为（见交付报告「需设计者裁」节）。
        边界（削掉的判别力，如实标注）：①`keep_rows` 缺省取 `ACTION_LOGS_KEEP_ROWS`
        （**经验标定，追不到理论出处**，取值照 issue #723 建议的「保留最近 10000
        行」）；②保留序取 `rowid`/`id` 插入序而非墙钟 `ts`（`time.time()` 不单调：
        NTP 步进 / 快照回滚下按墙钟淘汰会先删刚写入的行，与 #489 同因）；③裁剪
        本身**留痕**（在删除后落一条 `action_logs_pruned` 审计——否则「审计行被
        删」这件事反而不可审计；该审计行是删后插入的，必在保留窗口内，不会自删）；
        ④不动 `engine_meta` 等其它表。
        """
        if keep_rows is None:
            keep_rows = self.ACTION_LOGS_KEEP_ROWS
        c = self.conn.cursor()
        c.execute("SELECT COUNT(*) FROM action_logs")
        before = c.fetchone()[0]
        if before <= keep_rows:
            return 0
        c.execute("DELETE FROM action_logs WHERE id NOT IN"
                  " (SELECT id FROM action_logs ORDER BY id DESC LIMIT ?)",
                  (keep_rows,))
        removed = before - keep_rows
        # 裁剪留痕（治理动作可审计）：删后插入 ⇒ 本行 id 最大、必在保留窗口内。
        c.execute("INSERT INTO action_logs"
                  " (ts, action_type, summary, node_ids, outcome, context)"
                  " VALUES (?,?,?,?,?,?)",
                  (time.time(), "action_logs_pruned",
                   f"action_logs 滚动回收：保留最近 {keep_rows} 行，删除 {removed} 行",
                   json.dumps([], ensure_ascii=False),
                   json.dumps({"keep_rows": keep_rows, "removed": removed},
                              ensure_ascii=False),
                   json.dumps({}, ensure_ascii=False)))
        self._commit_locked()
        return removed

    @_transactional
    def set_escalation_enabled(self, escalation_id: str, enabled: bool,
                               designer_key: str = None):
        """启用/停用升级点（危机响应开关）。

        issue #130：此开关先前**无授权、无审计**。现补齐与
        `adjudicate_promotion` **同款**的 fail-closed 密钥闸，并把变更写入
        `action_logs`（审计留痕可查）。

        #130 路 B 补丁（**角色判**，裁定帖原文：「开关限 PRIMARY」）：密钥闸只回答
        「调用方是否持有设计者密钥」，**不回答「这个实例该不该动共享治理表」**——
        同一部署下 SUB 实例天然持有同一环境变量（`verify_designer` 只比对本进程
        `AEIS_DESIGNER_KEY`，见其 docstring 的 #156 事实边界；#586 已就同一理由判定
        「密钥闸不构成角色隔离」）⇒ SUB 持密钥即可关掉**共享治理表**里的全部升级点
        （危机响应面被单方面解除，本端读数确认）。现按本类既有单点
        `layer_writable_by_role` 的同款判据（共享层仅 PRIMARY 可写）补角色判：
        **仅 `Role.PRIMARY` 可关**，SUB 即便持密钥也 `PermissionError`。
        闸序：密钥闸在前（既有句面与行为**逐字保留**，无密钥时两处写点的报错文案
        仍逐字一致——`tests/test_issue749_escalation_write_gate.py` F 组钉此），
        角色闸在后。**`action_logs` 审计留痕保留不动**（被角色闸拒时零审计行：
        拒在写之前，与无密钥路径同形）。"""
        if not verify_designer(designer_key):
            raise PermissionError(
                "D-007 设计者认证失败：密钥无效或未配置 AEIS_DESIGNER_KEY（fail-closed）")
        if self.role != Role.PRIMARY:
            raise PermissionError(
                f"role={self.role.value} 无权启停共享治理表的升级点"
                "（升级点表跨实例共享，仅 PRIMARY 可写；SUB 持密钥亦拒）")
        c = self.conn.cursor()
        c.execute("UPDATE escalation_points SET enabled=? WHERE id=?",
                  (int(enabled), escalation_id))
        self.log_action(
            "escalation_toggle",
            f"set escalation {escalation_id} enabled={bool(enabled)}",
            outcome={"escalation_id": escalation_id, "enabled": bool(enabled)})

    # ---- #130（路 B）：升级点的「按 code 直查」面与「待设计者裁决」待办表 ----

    def get_escalation_point_by_code(self, code: str) -> Optional[Dict]:
        """按 `ESC-00x` **直查**一条升级点（精确匹配），查不到返回 `None`。

        #130（路 B · 裁定帖原文）：「新调用点用 `ESC-00x` 直查（精确、可审计、
        不受分词影响），**不动** `check_escalation` 现有的双向包含」。本方法即那个
        直查单点——`WHERE code=?` 精确等值，**不**经 `check_escalation` 的
        子串/双向包含面（分词、触发词长度、信号写法都不再影响命中）。

        为什么不复用 `check_escalation`（如实标注）：后者的契约是「**信号** → 升级点」
        的模糊匹配（`signal in trigger` 双向包含，见其 docstring），拿它做「我已知是
        哪条 code，取回该条」会在语义上引入无关的匹配面（信号恰好含别的触发词时
        返回多条）。两个面用途不同，故直查单列，**不改** `check_escalation`。

        边界（削掉的判别力）：①同 code 多行时取 `rowid` 最小的一条（`code` 列无
        唯一约束，`add_escalation_point` 不禁止重复登记同 code）；②**不**过滤
        `enabled`——本方法是「取回该条」，启用与否由调用方判（`check_escalation`
        才按 `enabled=1` 过滤）；③返回形状与 `list_escalation_points` 的单行同形
        （NULL 经 `or ""` 归一，与 #656 同口径）。
        """
        c = self.conn.cursor()
        c.execute("SELECT * FROM escalation_points WHERE code=? ORDER BY rowid LIMIT 1",
                  (code,))
        r = c.fetchone()
        if not r:
            return None
        return {"id": r[0], "code": r[1], "trigger": r[2] or "", "condition": r[3] or "",
                "action": r[4], "severity": r[5], "enabled": bool(r[6]),
                "created_at": r[7]}

    @_transactional
    def add_designer_todo(self, code: str, claim: str, action: str) -> str:
        """登记一条「待设计者裁决」待办（#130 路 B），返回待办 id。

        为什么有这张表（裁定帖原文，问题重述）：「这一步缺的语义是『谁在什么时候查
        升级点、查到之后系统进入什么状态』——`check_escalation` 是一张已定义、已播种、
        无人查的表，`needs_designer` 是一个已置位、无人读的标志」。本表把「已升级」
        从返回体里的一句文案变成库内可查的事实（`status='open'`）。

        ★**消费方尚未接线**（如实标注，裁定帖「存疑」节自陈未给出具体消费点）：
        本表当前只有**登记面**（本方法）与**只读列举面**（`list_designer_todos` /
        引擎门面同名方法），**没有**任何位置读它并推进 `status`（`open` →
        `consumed`，`consumed_at` 列因此恒为 NULL）。推进方向按裁定帖对齐本仓既有的
        设计者终裁唯一合规通路 `propose → verify → adjudicate`；具体消费点与状态推进
        语义属**未决语义题**，本件不擅自接线（不新造第二条终裁入口，见 `designer_decide`
        一节的裁定理由）。

        为什么**不**设 `verify_designer` 密钥闸（如实标注取舍）：登记动作由引擎**自身**
        的三级终裁（`_terminal_judgment` 判 `needs_designer`）触发，不是「调用方注入
        任意治理判据」——`recursive_reflect` 是公开 API、任何调用方都不持有设计者密钥，
        给登记加密钥闸等于让登记在该路径上**永远不可能发生**（本表也就永远为空）。
        与 #749 的播种分离同款判据：「程序自身的记录动作」与「运行期由调用方注入的
        治理写」分开。代价（如实标注）：`claim` 文本由调用方控制、会落库；但本表
        当前**零消费方**，故该文本不构成任何触发判据，也不改变任何既有行为。
        """
        tid = f"todo_{uuid.uuid4().hex[:8]}"
        c = self.conn.cursor()
        c.execute("INSERT INTO designer_todos VALUES (?,?,?,?,?,?,?)",
                  (tid, code, claim, action, "open", time.time(), None))
        self._commit_locked()
        return tid

    def list_designer_todos(self, status: str = None) -> List[Dict]:
        """列「待设计者裁决」待办（只读面，缺省全列，可按 `status` 过滤）。

        按 `created_at` 升序（同刻按 `rowid`，保证确定性）。**只读**：不推进任何
        状态、不写审计（见 `add_designer_todo` 的「消费方尚未接线」段）。
        """
        c = self.conn.cursor()
        if status is None:
            c.execute("SELECT * FROM designer_todos ORDER BY created_at, rowid")
        else:
            c.execute("SELECT * FROM designer_todos WHERE status=?"
                      " ORDER BY created_at, rowid", (status,))
        return [{"id": r[0], "code": r[1], "claim": r[2], "action": r[3],
                 "status": r[4], "created_at": r[5], "consumed_at": r[6]}
                for r in c.fetchall()]

    # ---------- 统计 ----------

    def get_stats(self) -> Dict:
        """全库统计快照（各层节点数等聚合指标）。

        ★优化（本笔 · 零行为变更）：旧实现按 `MemoryLayer` **逐层**发一条
        `SELECT COUNT(*) FROM nodes WHERE layer=?`（N 次），再各发一条 COUNT 查边
        （2 次）⇒ 共 **N+2** 次往返。改为：
          · 节点侧一趟 `GROUP BY layer` 聚合（1 次）；
          · 边侧一趟条件聚合（1 次，同时取 total 与 verified）。
        ⇒ **N+2 → 2** 次查询。

        **返回形状与取值必须逐位不变**：零节点的层在旧实现里由循环显式赋 0，
        而 `GROUP BY` 不会为它出行 —— 故先按 `MemoryLayer` 预置 0、再用聚合结果覆盖。
        历史/未知层名（不在 `MemoryLayer` 内的 `layer` 值）旧实现本就不计，新实现同样跳过。
        """
        c = self.conn.cursor()
        stats = {f"{layer.value}_nodes": 0 for layer in MemoryLayer}
        c.execute("SELECT layer, COUNT(*) FROM nodes GROUP BY layer")
        for _lv, _n in c.fetchall():
            _k = f"{_lv}_nodes"
            if _k in stats:                     # 只认 MemoryLayer 内的层名（旧口径）
                stats[_k] = _n
        # 边侧一趟取两个读数：`SUM(verified=1)` 与旧 `WHERE verified=1` 同口径
        # （NULL 不满足 `=1` ⇒ 两边都不计）；空表时 SUM 为 NULL ⇒ COALESCE 回落 0。
        c.execute("SELECT COUNT(*), COALESCE(SUM(verified=1), 0) FROM edges")
        _te, _ve = c.fetchone()
        stats["total_edges"] = _te
        stats["verified_edges"] = int(_ve)
        return stats

    def close_current_thread(self):
        """服务请求收尾时释放本线程连接，不关闭其他线程的连接。

        #660：旧实现读的是 `getattr(self._local, "conn", None)`，而 `_connect` 实际设的是
        `self._local.slot` ⇒ 该读取**恒为 None**、整个方法恒为 no-op（槽仍在、连接仍可用）。
        现改读/清 `self._local.slot`：关槽、从登记表摘除、清空本线程槽（下次访问经
        `conn` property 按需重建——本方法**不是**终止态，故不置 `self._closed`）。

        `:memory:` 边界（如实标注）：`__init__` 的保活锚连接在创建线程内**就是**该线程的
        `_local.slot`（`_connect` 同时登记两处）。若直接关掉它，共享内存库随最后一条连接
        消失 ⇒ 后续访问得到空库假连接。故当被关的槽就是锚连接时，**先补一条新的保活
        连接**再关旧槽（共享缓存库在旧连接关闭前已有新连接挂上，库不消失）；文件库无需
        保活，直接清引用。
        """
        with self._lock:
            slot = getattr(self._local, "slot", None)
            if slot is None:
                return
            self._local.slot = None
            if slot is self._anchor:
                if self._memory_dsn is not None:
                    self._anchor = self._connect(self._memory_dsn)
                    # 新锚只作保活，不作本线程连接（下次访问按需重建）。
                    self._local.slot = None
                else:
                    self._anchor = None
            slot.close()
            self._conns.discard(slot)

    def close(self):
        """关闭本实例登记的全部连接（Agent 生命周期终点调用）。

        #660：旧实现只遍历 `_conns` 关槽 ＋ `_anchor = None`，**不清 `self._local.slot`**
        ——被 close 的槽 `_closed=True` 但 `slot.conn` 仍非 None，`conn` property 判为
        「仍新鲜」并返回死连接（见 `conn` docstring）。现：① 置 `self._closed`（终止态，
        `conn` 显式失败并给可判文案）；② 清本线程 `_local.slot`；③ 其余线程的槽已关闭，
        其 `_local.slot` 指向的槽 `_closed=True` 由 `conn` 新鲜度判据拦下。
        """
        self._closed = True
        for slot in list(self._conns):
            slot.close()
        self._conns = weakref.WeakSet()
        self._local.slot = None
        self._anchor = None


def _filter_min_score(scored, min_score):
    """#795：按 `min_score` 截断**返回面**（不截断候选池、不改打分/排序）。

    `min_score` 缺省 0.0 / None / 负数 ⇒ 原样返回（既有调用逐位不变）；非有限
    （NaN/±Inf）⇒ `ValueError`（fail-closed，同 #185/#196 口径：不可判的阈值不得
    被译成「不过滤」）。阈值由调用方按自己的语料标定，本函数不内置推荐值。
    """
    if min_score is None or min_score == 0:
        return scored
    ms = _require_finite("min_score", min_score)
    if ms <= 0.0:
        return scored
    return [(n, s) for n, s in scored if s >= ms]


# =============================================================================
# 时空记忆引擎（SpacetimeMemoryEngine）
# =============================================================================

class SpacetimeMemoryEngine:
    """
    协议实例核心引擎，封装 LayeredStore 并提供高层 API。
    符合智能论 v3.2 第四章规范。
    """

    #: 情境层 FIFO 上限默认值（参照 AEIS；lingshu #246 起经 engine_meta 持久化）。
    DEFAULT_CONTEXT_MAX = 200

    #: 单条 claim 参与空间关系计算的视觉锚点数上限（lingshu #426）。
    #: `_vprim_context_for_claim` 对锚点做**双重循环**生成两两关系 ⇒ K 个锚点即
    #: K(K-1)/2 条关系，K 不设限时一条长 claim 可把内存撑爆（旧实现 `finditer`
    #: 无限收）。32 个锚点 ⇒ 496 条关系，已远超实际推理链需要的锚点数。
    #: 判据来源：经验标定（本件 #426）——仓内无规定视觉锚点数上限的理论章节，
    #: 追不到更早出处。
    MAX_VPRIM_ANCHORS = 32

    def _commit_locked(self):
        """事务内提交（委托 store，语义见 LayeredStore._commit_locked）。

        #665：Engine 侧 `@_transactional` 方法同样在方法体里写
        `self.conn.commit()`（`self.conn` 即 store 连接），必须一并改走本方法，
        否则内层 engine 方法一样会提交整条事务、销毁外层 SAVEPOINT。
        """
        return self.store._commit_locked()

    @contextlib.contextmanager
    def _tx(self):
        """写事务边界（委托 store）：正常提交、异常回滚。语义见 LayeredStore._tx。"""
        with self.store._tx():
            yield self.store.conn

    #: `_reuse_tracker` 保留的最近轮数上界（#208）。**经验标定，追不到理论出处**——
    #: 仓内无规定该上界的理论章节；取值只需覆盖「当前轮 + 极少近邻轮」即可满足
    #: 同轮去重语义（见 `_note_reuse` docstring 边界）。
    _REUSE_TRACKER_ROUNDS = 5

    #: `flywheel_reuse` 持久表的行数上界（#722）。**经验标定，追不到理论出处**——
    #: 仓内无规定该上界的理论章节；取值与同族钳制（`gap_history` 滚动保留 2000 条、
    #: `SELF_SNAPSHOT_MAX=200`）同量级，只为「常驻部署不再无界增长」而不追求保留
    #: 全历史（该表的用途是跨进程复用观测，滚动窗口足够；见 `_note_reuse` 边界）。
    #: 裁剪在 `_note_reuse` 落库后同事务执行（同 `record_info_gap` 手法）。
    FLYWHEEL_REUSE_MAX_ROWS = 20000

    def __init__(self, db_path: str = ":memory:", identity: str = "协议实例", role: Role = Role.PRIMARY):
        self.store = LayeredStore(db_path, role=role)
        self.role = role
        self.self_model = SelfModel(identity=identity)
        # #33：启动时从持久化快照回填（旧实现无条件新建默认模型 ⇒ 重启即失忆）。
        # 恢复结果（含失败）显式记入 `_self_restore`，供 self_check 观测（D-21/C-2）。
        self._self_restore = self._restore_self_model(identity)
        self.self_model._persist_hook = self._save_self_model
        # #483：持久态**基线**（本实例「上次与库内同步时」的内存快照）。写前合并的
        # 判据基准：字段值 == 基线 ⇒ 本实例自上次同步以来未改过该字段 ⇒ 采纳库内
        # 现值（他方更新不丢）；≠ 基线 ⇒ 本实例改过 ⇒ 保留本实例值（字段级后写胜）。
        # 构造点在恢复之后：基线即「启动时内存与库内一致」的那份快照。
        self._self_model_baseline = self.self_model.to_dict()
        self._decay_thread = None
        self._running = False
        self._decay_wake = threading.Event()   # #37：可唤醒的等待（替代裸 sleep）
        # #176：衰减时钟（可替换）。None ⇒ `decay_cycle()` 走旧的「每次=一个周期」
        # 步进；`start_auto_decay(clock=…)` 置入后，每次衰减按真实流逝时间演化。
        self._decay_clock = None
        # #184：后台衰减线程的**最近一次**周期异常（None=从未失败）。循环内捕获并
        # 记录后继续跑（旧实现异常直接冒泡出线程 ⇒ 线程静默退出、遗忘机制整体停摆，
        # 而 `_running` 仍为 True ⇒ 对外不可观测）。经 `self_check()` 的
        # `auto_decay` 字段暴露。
        self._decay_last_error = None
        # ---- v1.2 状态 ----
        self._dedup_static = 0.85          # M5 去重静态基准（2.7.2 动态死区）
        self._dedup_window = 30
        self._dedup_history: List[float] = []
        # #800：M5 去重「查→插」原子化锁。`add_perception` 的去重判定（全层
        # 候选扫描 + Jaccard 阈值）与后续 `add_node` INSERT 之间原本无引擎级
        # 互斥——store 的 `RLock`（`LayeredStore._lock`）只覆盖**单次**读写
        # 调用，两线程的 SELECT 都可跑在对方 INSERT 之前 ⇒ 同内容并发写产出
        # 多个重复节点（TOCTOU；实测 3 线程 × 20 次同内容 = 3 个不同 id）。
        # 取 `RLock` 而非 `Lock`：跨度内含 provider/证据钩子调用面
        # （`to_semantic_coordinates` / `_evidence.record`），若钩子再入
        # `add_perception`（同线程）`Lock` 会自死锁；`RLock` 同线程可重入、
        # 跨线程互斥语义不变。边界（如实标注）：①只护**单引擎实例内**的线程
        # 并发——多进程共享同一库文件不在本件面（那是 sqlite 事务层职责）；
        # ②`skip_dedup=True` 的写入不经判同键，本就不产生该竞态，锁对其
        # 透传无害；③跨引擎实例（两个 `SpacetimeMemoryEngine` 指同一 db）
        # 各持各的锁，竞态面仍在——issue 口径是「同一引擎实例多线程」
        # （复现即此形态），多实例属另一件。
        self._m5_dedup_lock = threading.RLock()
        self._context_max = self._load_context_max()  # M4 情境层 FIFO 上限（参照 AEIS）
                                                      # #246：跨进程持久化（见 _load_context_max）
        self._embedding_provider = None    # M1 语义检索提供者（duck-typed 注入，D-005）
        self._evidence = None              # 证据账本（默认关闭；见 enable_evidence_ledger）
        if os.environ.get("LINGSHU_EVIDENCE_LEDGER") == "1":
            self.enable_evidence_ledger()
        self._tms = None                   # 真值维护（默认关闭；见 enable_tms）
        if os.environ.get("LINGSHU_TMS") == "1":
            self.enable_tms()
        self._setup_v13()
        # ---- v1.5 状态（A-1~A-5） ----
        self._verifier_config = {"dedup_static": 0.85, "deviation_threshold": 0.3}
        # #153：回放**已终裁通过**的验证标准——旧实现只在 __init__ 写死默认值，
        # 终裁结果仅存于内存 ⇒ 重启即失效（盘上 `verifier_standards` 的 approved
        # 行仍在，却不再生效）。
        self._replay_verifier_standards()
        self._gap_history: List[Dict] = self._load_gap_history()
        self._resource_history: List[Dict] = []
        self._seed_escalation_points()
        # ---- v1.6 状态（M11 语义空间 / M13 备份） ----
        self._semantic_provider = None
        self._semantic_error = ""
        self._setup_v16()
        # ---- v1.7 状态（多模态 · D-004/D-005） ----
        self._last_frame: Dict[str, str] = {}
        self._local_visual_buffer: List[Dict] = []
        self._visual_buffer_max = 200
        self.migrate_v17_coordinates()
        # ---- v1.8 状态（决策偏好注意力） ----
        self._attention_policy = None
        self._attention_error = ""
        self._setup_v18()
        # ---- v1.9 状态（预测引擎） ----
        self._prediction = None
        self._prediction_error = ""
        self._setup_v19()
        # ---- v1.10 状态（生命周期自动机） ----
        self._lifecycle = None
        self._lifecycle_error = ""
        self._setup_v110()
        # ---- v1.11 状态（知识飞轮） ----
        self._flywheel = None
        self._flywheel_error = ""
        # ---- v1.15 状态（长期记忆写入决策器） ----
        self._gate = None
        self._reuse_tracker: Dict[int, set] = {}
        self._interaction_count = 0
        self._session_id = uuid.uuid4().hex[:8]  # T4：复用落库的进程实例标识
        self._last_reflection_chain = None   # REFLECT-REV1：最近反思链（递归截断时返回）
        self._event_queue: List[Dict] = []
        self._setup_v111()
        # ---- v1.12 状态（自我认知循环） ----
        self._self_cognition = None
        self._self_cognition_error = ""
        self._setup_v112()
        # ---- v1.13 状态（视觉感知 · 身体能力） ----
        self._vision_provider = None
        self._vision_error = ""
        self._setup_v113()
        # ---- BODY-REV1 状态（外部设备身体层） ----
        self._body_registry = None
        self._body_error = ""
        self._setup_body()
        # ---- issue #156：组件装配状态快照（self_check 面 · 只增字段） ----
        self._component_assembly = self._component_assembly_status()
        self._restore_structure_configuration()

    def _restore_structure_configuration(self):
        saved = self.store.get_meta("verifier_config").get("verifier_config")
        if saved:
            self._verifier_config.update(json.loads(saved))
            self._dedup_static = self._verifier_config["dedup_static"]
            if self._cognition:
                self._cognition.deviation_threshold = self._verifier_config["deviation_threshold"]
        saved = self.store.get_meta("attention_weights").get("attention_weights")
        if saved and self._attention_policy:
            weights = json.loads(saved)
            for key in self._attention_policy.preference_weights:
                if key in weights:
                    self._attention_policy.preference_weights[key] = weights[key]

    # ==================== 组件装配状态（issue #156） ====================

    _COMPONENT_SLOT_TABLE = (
        # (组件名, 装配槽属性, 失败信息字段) —— 只读观测，不改装配语义
        ("entity_registry", "entity_registry", "_v13_error"),
        ("cognitive_orchestrator", "_cognition", "_v13_error"),
        ("blindspot_learning_loop", "_learning_loop", "_v13_error"),
        ("semantic_space", "_semantic_provider", "_semantic_error"),
        ("attention_policy", "_attention_policy", "_attention_error"),
        ("prediction_engine", "_prediction", "_prediction_error"),
        ("lifecycle_engine", "_lifecycle", "_lifecycle_error"),
        ("flywheel_engine", "_flywheel", "_flywheel_error"),
        ("self_cognition_engine", "_self_cognition", "_self_cognition_error"),
        ("body", "_body_registry", "_body_error"),
        ("vision", "_vision_provider", "_vision_error"),  # 惰性：未触发视觉前不尝试
    )

    def _component_assembly_status(self) -> Dict:
        """issue #156：组件装配状态汇总（只读既有槽位与 *_error 字段）。

        返回 {"ok": [...], "failed": {名: 错误}, "lazy": [...], "discovery": {...}}；
        供 self_check 与外部观测（装配失败此前完全静默）。只增字段，不改装配语义、
        不改既有返回结构。
        """
        ok, failed, lazy = [], {}, []
        for name, slot, err_field in self._COMPONENT_SLOT_TABLE:
            err = getattr(self, err_field, "")
            if getattr(self, slot, None) is not None:
                ok.append(name)
            elif name == "vision" and not getattr(self, "_vision_attempted", False):
                # #602：lazy 的判据是**显式**的「尚未尝试」状态，不再是
                # 「err 为空」——无消息异常（str(e)==""）曾使「尝试过 + 失败」
                # 与「从未尝试」读数逐位相同。其它 10 个组件无此特判。
                lazy.append(name)          # 惰性组件：尚未尝试装配
            else:
                failed[name] = err or "未装配（无错误信息）"
        return {"ok": ok, "failed": failed, "lazy": lazy,
                "discovery": _component_discovery_report()}

    def _setup_v113(self):
        """v1.13 视觉组件装配（惰性：仅清状态，首次 perceive_image 才加载 YOLO/CLIP——
        启动零负担，身体功能不再拖慢进程启动/MCP 握手）"""
        self._vision_provider = None
        self._vision_error = ""
        # #602：显式「是否尝试过装配」状态——状态面据此判 lazy，不再用
        # 「_vision_error 为空」当「未尝试」的代理（无消息异常 ⇒ 空串 ⇒ 误报 lazy）。
        self._vision_attempted = False

    def _ensure_vision(self):
        """惰性装配视觉 provider（首次视觉调用触发）。
        返回 provider 或 None（失败时 _vision_error 记录原因）。

        #602：进入即置 `_vision_attempted=True`——「是否尝试过」由显式状态承载，
        不再让观测面从「错误是否为空」反推。
        #603：成功出口**擦除** `_vision_error`——失败→恢复是惰性装配的常规路径
        （装好依赖后不重启即恢复），陈旧错误不得继续挂在能力声明上。
        """
        if self._vision_provider is not None:
            return self._vision_provider
        self._vision_attempted = True          # #602：显式「已尝试」状态
        try:
            from vision import create_vision_provider
            p = create_vision_provider()
            if p is not None and p.available():
                self._vision_provider = p
                self._vision_error = None      # #603：清陈旧错误（与置位同处落）
                return p
            self._vision_error = "ultralytics 未安装（pip install ultralytics）"
        except Exception as e:
            # #602：无消息异常（bare `raise ImportError`）str(e)=="" ⇒ 回落 repr/类名，
            # 保证失败一定带**可读**标识（空串会被状态面误读成「从未尝试」）。
            self._vision_error = str(e) or repr(e) or type(e).__name__
        return None

    def _setup_body(self):
        """BODY-REV1 身体层装配（零依赖：screen/files/process 设备注册表）。
        工作区边界从环境变量 AEIS_WORKSPACE 读取（空则不限制）。"""
        self._body_registry = None
        try:
            import os as _os
            from body import build_default_registry
            self._body_registry = build_default_registry(
                workspace=_os.environ.get("AEIS_WORKSPACE", ""))
        except Exception as e:
            self._body_error = str(e)

    def _setup_v111(self):
        """v1.11 飞轮组件装配（惰性导入 · 零外部依赖）"""
        self._flywheel = None
        try:
            from flywheel_engine import FlywheelEngine
            self._flywheel = FlywheelEngine(self, self._prediction)
        except Exception as e:
            self._flywheel_error = str(e)

    def _setup_v112(self):
        """v1.12 自我认知组件装配（惰性导入 · 零外部依赖）"""
        self._self_cognition = None
        try:
            from self_cognition_engine import SelfCognitionEngine
            self._self_cognition = SelfCognitionEngine(self)
        except Exception as e:
            self._self_cognition_error = str(e)

    def _note_action(self, action_type: str, summary: str = "",
                     node_ids: list = None, outcome: dict = None,
                     context: dict = None):
        """v1.12 P0-1 行为日志钩子（委托自我认知引擎，未装配时静默）"""
        if self._self_cognition is None:
            return
        try:
            self._self_cognition.log_action(action_type, summary, node_ids, outcome, context)
        except Exception:
            pass

    def _setup_v110(self):
        """v1.10 生命周期组件装配（惰性导入 · 预测/盲区/注意力联动）"""
        self._lifecycle = None
        try:
            from lifecycle_engine import LifecycleEngine
            self._lifecycle = LifecycleEngine(self, self._prediction,
                                              self._learning_loop, self._attention_policy)
        except Exception as e:
            self._lifecycle_error = str(e)

    def _setup_v19(self):
        """v1.9 预测组件装配（惰性导入 · AttentionPolicy 适配器 D-005）
        v1.15：启动时从观测层验证记录重建 _hit_history（跨重启持久）

        #34：旧实现只写裸顶层名 `from prediction_engine import PredictionEngine`
        ——而 `PredictionEngine` 就在本仓 `lingshu/world/prediction.py`，全仓
        既无该顶层模块、部署方也无处声明 ⇒ **装配必败、`self._prediction` 恒 None**，
        「预测→验证→写回」主闭环整条静默死掉。修法（`docs/plans/待裁清单_v0.1.md`
        D-52 推荐「改相对导入」）：**先试**裸名（保留 AEIS 私域 `sys.modules`
        命名空间注册别名面与显式注入面，逐字维持 #156 的受控解析收口），**失败即
        回退**仓内相对导入 `..world.prediction`。
        判据来源：`docs/plans/待裁清单_v0.1.md` D-52；本仓模块自带、不该依赖外部注入。
        削掉的判别力：无——旧行为的唯一后果是恒失败；新行为在**受控面有该组件时
        仍优先用受控面**（私域别名语义不变），仅把「本仓自带实现的缺失」补上。
        """
        self._prediction = None
        try:
            try:
                from prediction_engine import PredictionEngine
            except Exception:
                # 本仓形态：无顶层 `prediction_engine` 模块 ⇒ 回退仓内自带实现
                from ..world.prediction import PredictionEngine
            self._prediction = PredictionEngine(self, self._attention_policy)
            # 重建命中历史：观测层 [验证回路] 记录 → hit/miss 序列
            try:
                hits = []
                for n in self.store.query_nodes(layer=MemoryLayer.KNOWLEDGE, limit=500):
                    if "prediction_feedback" in (n.tags or []):
                        hits.append("hit" in (n.tags or []))
                if hits:
                    self._prediction._hit_history = hits[-200:]
            except Exception:
                pass
        except Exception as e:
            self._prediction_error = str(e)

    def _setup_v18(self):
        """v1.8 注意力组件装配（惰性导入 · 零外部依赖）"""
        self._attention_policy = None
        try:
            from attention_policy import AttentionPolicy
            self._attention_policy = AttentionPolicy()
        except Exception as e:
            self._attention_error = str(e)

    def _setup_v16(self):
        """v1.6 语义组件装配（惰性导入 · 默认零外部依赖 · D-005）"""
        self._semantic_provider = None
        try:
            from semantic_space import SemanticSpaceProvider
            self._semantic_provider = SemanticSpaceProvider()
        except Exception as e:
            self._semantic_error = str(e)

    def _setup_v13(self):
        """v1.3 认知组件装配（惰性导入，核心零外部依赖；组件缺失时优雅降级）"""
        self.entity_registry = None
        self._cognition = None
        self._learning_loop = None
        self._v13_ready = False
        self._v13_error = ""
        try:
            from entity_registry import EntityRegistry
            from cognitive_orchestrator import CognitiveOrchestrator
            from blindspot_learning_loop import BlindSpotLearningLoop
            self.entity_registry = EntityRegistry(self.store)
            self._learning_loop = BlindSpotLearningLoop(self)
            self._cognition = CognitiveOrchestrator(self, self.entity_registry)
            self._v13_ready = True
        except Exception as e:
            self._v13_error = str(e)

    # ==================== 感知入口 ====================

    # ---- M5 去重前的「实质分歧」判据（lingshu #29② / #142） ----
    # 判据来源：经验标定（#29/#142 修复轮；**理论章节追不到**）。字符 bigram
    # Jaccard 对否定词与数值几乎不敏感（实测长文本改一字仍 ≥0.93 > 阈值 0.85）
    # ⇒ 更正语句被判为重复而**吞掉并给原命题增信**。否定标记词面参照私有侧既有
    # 矛盾判据 `md_cg/scrub.py::_polarity` 的 `_NEG_WORDS`（此处只取「有无否定
    # 标记」这一最小面，不做极性求和——本仓零外部依赖，不跨仓 import）。
    _NEGATION_MARKERS = ("不", "非", "未", "无", "没", "别", "莫", "勿")

    # ---- #484：否定判据的**词面**豁免表（单字嵌在非否定常用词里 ≠ 否定） ----
    # 判据来源：公开 issue #484（「按**词**而不是按**字**匹配」）＋ 复用仓内既有词面
    # 口径 `lingshu/core/contradiction.py::_NON_NEG`（其 `_has_neg` 在差异片段上先整词
    # 剔除的「非否定常用词」表），并按 issue 点名的清单补 4 词：无法 / 不同 / 不久 /
    # 没事。中文无空格词界（本仓不引分词器，与 `_synonym_trigger_hit` 同款声明），故
    # 以「整词剔除后判剩余否定标记」代替 `\b`；剔除按**长词优先**（确定性排序）。
    # `_NEGATION_MARKERS` 本身**取值集合不变**（本件不扩否定标记词表）。
    # 边界（削掉的判别力，如实标注）：`无法`/`不同`/`不久`/`没事` 被豁免后，
    # 「无法完成 ↔ 能完成」「不同 ↔ 相同」这类**单侧替换**不再判 `negation`
    # （旧实现会判）——这是 issue 点名要求的豁免，代价如实登记，不隐藏。
    _NON_NEGATION_WORDS = tuple(sorted(
        set(_contradiction._NON_NEG) | {"无法", "不同", "不久", "没事"},
        key=lambda _w: (-len(_w), _w)))

    # ---- M5 去重的「成败极性」标签（lingshu #225） ----
    # 判据来源：经验标定（#225 修复轮；**理论章节追不到**）。极性由
    # `record_action_sequence` 的 `success`/`failure` 标签承载（正文前缀
    # `成功：`/`失败：` 只是**文本形态**，M5 判同键不看它）——故判同键须显式
    # 取标签集。词表只含这两个**既有**标签（issue 括注的 `negative`/`conflict`
    # 等属未定项，不擅定；见 `add_perception` 内 #225 段「边界」）。
    _POLARITY_TAGS = frozenset({"success", "failure"})

    # ---- #803：M5 去重键的零宽字符剥除 ----
    # 零宽字符（ZWSP U+200B / ZWNJ U+200C / ZWJ U+200D / BOM U+FEFF / WJ U+2060）
    # 视觉不可见但**字节不同** ⇒ `char_bigram_jaccard` 的二元组集合被打散，
    # `important\u200dsecurity\u200dnotice` 与 `important security notice` 判不同源，
    # M5 内容去重被绕过——同视觉内容可造 2^n 个「隐形变体」节点（内容污染 /
    # 存储膨胀 / importance 分散）。修法：**去重键**（Jaccard 与分歧/矛盾判据的
    # 输入）先剥除零宽字符；**落库正文不动**（用户字节原样保留，只正规化判同键）。
    # 判据来源：公开 issue #803（复现 `'important security notice'` vs 插 2 个
    # U+200D 变体 id 不同；建议「去重前做 Unicode 正规化」）。
    # 边界（如实标注）：①只剥**零宽字符**（视觉同义面），不做 NFKC 全量
    # 正规化——NFKC 还会折叠全/半角、连字等**语义可见**的差异（如 `①`→`1`、
    # `ﬁ`→`fi`），把「用户故意的字形变体」也判同，超出本件攻击面，不擅扩；
    # ②`skip_dedup=True` 的主动沉淀路径不经判同键，不受影响；③剥除只作用于
    # **新写入与候选的比对两侧**——库里既存的含零宽节点与新写入的干净内容
    # 也能互相判同（候选侧同样剥除）。
    _ZERO_WIDTH_RE = None  # 惰性编译（模块顶部不 import re，与 _assertion_divergence 同款）

    @staticmethod
    def _strip_zero_width(s: str) -> str:
        """#803：剥除零宽字符（U+200B-U+200D / U+FEFF / U+2060）——M5 判同键的正规化。"""
        import re as _re
        cls = SpacetimeMemoryEngine
        if cls._ZERO_WIDTH_RE is None:
            cls._ZERO_WIDTH_RE = _re.compile("[\u200b-\u200d\ufeff\u2060]")
        return cls._ZERO_WIDTH_RE.sub("", s)


    # ---- M5 去重的「条件空间」判同（lingshu #268） ----
    @staticmethod
    def _conditions_mergeable(a: ConditionSpace, b: ConditionSpace) -> bool:
        """两条知识可否视为**同一条件**下的重复（#268）。

        判据（保守 · 宁另立不误并）：`observation_position` /
        `observation_tool` / `existence_constraint` 三维度须**字面相等**，
        且 `time_window` 须**有交集**（`min(hi) >= max(lo)`）；任一不满足即
        「不同条件」⇒ 不参与去重（另立节点）。
        判据来源：`docs/theory/世界模型与语义时空图_完整理论整理与实现路线.md`
        L35「知识 = 条件空间 C 中的有效关系 K=K(C)」＋ L154「任何知识/识别须绑定
        条件（观测位置/工具/时间/存在约束）」＋ 公开 issue #268「建议 1」。
        边界（削掉的判别力，如实标注）：①只做**相等/相交**判据，不引入
        「条件包含关系」推理（`A ⊆ B` 仍判不同）；②时间窗**相邻不重叠**仍判不同
        （issue 建议 2 的「取并集扩展旧节点时间窗」是另一条独立改动，本件不做）；
        ③不自动为「同正文异条件」的两节点建 `SIMILAR`/`APPLIES_TO` 边（issue
        建议 1 的该半句属新增机制，本件只做「不合并」）。
        """
        if (a.observation_position != b.observation_position
                or a.observation_tool != b.observation_tool
                or a.existence_constraint != b.existence_constraint):
            return False
        a0, a1 = a.time_window
        b0, b1 = b.time_window
        return min(a1, b1) >= max(a0, b0)

    # ---- #484：否定判据的**词面**口径（非否定常用词豁免） ----
    # 判据来源：公开 issue #484（「比较两句之间的差异，而不是各自的全局存在性」）＋
    # 仓内既有词面口径 `lingshu/core/contradiction.py::_NON_NEG`（整词出现即不算否定的
    # 常用词表，供其 `_has_neg` 在差异片段上使用）。此处**复用同一张表**（不另造词表、
    # 不引第三方分词库）：中文无空格词界，故以「整词剔除」代替 `\b`。
    @staticmethod
    def _has_negation_word(s: str) -> bool:
        """整串是否含**否定词**（#484：按词面判，而非「是否含否定字」）。

        `_NEGATION_MARKERS` 里的单字大量嵌在**非否定的常用词**里（非常/未来/无法/
        无论/不同/不久/别人/没事…）：先整词剔除 `_NON_NEGATION_WORDS` 里的非否定
        常用词，再判剩余否定标记。非 `str` / 空串返回 `False`（退化输入不抛）。
        """
        if not isinstance(s, str) or not s:
            return False
        t = s
        for w in SpacetimeMemoryEngine._NON_NEGATION_WORDS:
            t = t.replace(w, " ")
        return any(m in t for m in SpacetimeMemoryEngine._NEGATION_MARKERS)

    @staticmethod
    def _assertion_divergence(a: str, b: str) -> str:
        """判定两条内容是否构成**实质分歧**（而非同一断言的重复表述）。

        返回分歧类型 `"negation"` / `"numeric"`，无分歧返回 `""`。
        仅覆盖可规则化分歧（否定标记有无 / 数值字面量不同）；换说法的反命题
        仍不可达（如实声明边界，不猜）。判据保守：宁判分歧（另立节点）不判重复。
        边界（削掉的判别力）：数值判据只认**阿拉伯数字**字面量；「百分之五十 →
        百分之六十」这类纯中文数词变更不在本判据内（仍按重复合并）——如实声明。
        #484：否定判据按**词面**（否定**词**）而非「否定**字**」——`_NEGATION_MARKERS`
        里的单字大量嵌在**非否定的常用词**里（非常/未来/无法/无论/不同/不久/别人/
        没事…）。旧实现 `any(w in a …) != any(w in b …)` 只比「两句**各自**是否出现过
        任一否定字」：原句带一个这样的常用词即为 True，更正句再插入「不」/「尚未」
        也是 True ⇒ 判为无分歧，更正被 M5 当重复吞并（更正文本不落库、无 conflict
        标签、无 OPPOSITE 边，且 #223 的合并把**错误节点**的 importance 抬到更正的
        0.9 ⇒ 用更正的原话 recall 反而返回错误事实）。
        现按**整词**判（`_has_negation_word`，复用仓内 `contradiction._NON_NEG`），且
        只在「**一方被否定、另一方未被否定**」时判 `negation`——两句都含否定标记
        （含同向「都 不 X」）仍判无分歧。判据来源：公开 issue #484 正文与维护者复核帖
        （判据 = 否定标记的**差异**，按词不按字）。
        边界（判别力变化，如实标注）：①`_NEGATION_MARKERS` **取值集合不变**（未扩
        否定标记词表；豁免表见 `_NON_NEGATION_WORDS` = 仓内 `contradiction._NON_NEG`
        ∪ {无法, 不同, 不久, 没事}）；②判在**整句**上（非差异片段）——「他非常认可」
        vs「他非常认可，不过还需复核」会判分歧（保守方向：宁另立不误并）；③豁免表
        内的词被整词剔除 ⇒「无法完成 ↔ 能完成」「不同 ↔ 相同」这类**单侧替换**不再
        判 `negation`（旧实现会判）——这是 issue 点名要求的豁免，代价如实登记；
        ④不做英文否定识别（旧实现也没有，不扩大面）。
        """
        import re as _re
        if (SpacetimeMemoryEngine._has_negation_word(a)
                != SpacetimeMemoryEngine._has_negation_word(b)):
            return "negation"
        na = _re.findall(r"\d+(?:\.\d+)?", a)
        nb = _re.findall(r"\d+(?:\.\d+)?", b)
        if na != nb and (na or nb):
            return "numeric"
        return ""

    def add_perception(self, content: str, modality: str = "text",
                       spatial_coordinates: Dict[str, float] = None,
                       condition_space: ConditionSpace = None,
                       importance: float = 0.5,
                       tags: List[str] = None,
                       entities: List[str] = None,
                       skip_dedup: bool = False,
                       source: str = None,
                       self_generated: bool = False) -> STNode:
        """
        添加一条感知（自动进入知识层）

        source / self_generated（证据账本，默认关闭时忽略）：本次写入的来源与
        是否为系统自产。启用 `enable_evidence_ledger()` 后，写入与去重命中都记一条
        证据，置信度由账本推导——同源重复不增信，自产内容不计分（见 core/evidence.py）。

        skip_dedup（v1.26c）：跳过 M5 去重——主动沉淀类写入（剧情/快照/
        里程碑）需要独立节点身份，不能被合并进相似的感知节点（否则
        剧情标签/高 importance 丢失，LongTermMemoryGate 的写入形同虚设）。

        issue #331（数据损坏）：`content` 的类型**先前无校验**，而下游三处都
        只按 `str` 写：①M5 去重条件里的 `isinstance(content, str)` ⇒ 非 str
        **绕过去重**；②`STNode.content` 原样入 `nodes.content`（SQLite TEXT 列
        不做类型转换）⇒ `bytes` 以 **BLOB** 静默落库、`None` 落成 NULL；
        ③`dict`/`list` 绑定参数时才抛 `ProgrammingError`——且那是在
        `_interaction_count` / `_note_action` **已发生之后**（半截副作用）。
        危害是**全局**的：检索面（`char_bigram_jaccard` / `search_content` 的
        二元组集合运算）遇到 bytes 正文直接 `TypeError: sequence item 0:
        expected str instance, bytes found` ⇒ **一行坏正文毒化整个检索**
        （与 #267 同族的「静默入库 → 事后炸」形态）。
        修法（fail-closed，**先判后写**、零副作用）：非 `str` 一律 `TypeError`，
        在 `_interaction_count` / `_note_action` / 任何 SQL 之前。空串 `""`
        照旧放行（既有语义：空内容节点可入库，见 `tests/test_core_export_all.py`
        D 组）。**不**做 `str(content)` 静默强转——那会把 `dict` 存成 repr 文本，
        是「静默篡改语义」，比拒绝更坏。
        判据来源：经验标定（本件 #331）——口径取「非 str ⇒ fail-closed 拒绝」；
        仓内无规定 `add_perception` 入参类型的理论章节，追不到更早出处。
        边界（如实标注）：本闸只覆盖 `add_perception`；`store.add_node` /
        `import_all` 等其它写入面不在本件范围内。

        issue #805（tags/entities/modality 零验证）：三参数此前**全无校验**，
        旧形态（实测 main@40fceb6）：裸字符串 `tags="tag1"` 被当可迭代，落库成
        `['t','a','g','1']` 四个单字符 tag（字符爆炸）；`entities=[1,2]` 经
        `f"ent:{eid}"` 静默字符串化成 `ent:1`/`ent:2`（issue 点名的「数字变
        字符串」）；`tags=["ok",123]` / `["ok",None]` 把 JSON 数字/null 混进
        tags 列（`get_nodes_by_tag` 按字符串匹配永远漏）；`tags=12345` 走去重
        路径时在极性闸 `set()` 处抛无说明的 TypeError 且 `_interaction_count` /
        `_note_action` 副作用已发生（半截副作用）；`modality` 任意值放行——
        `123` 落 TEXT 列被 SQLite 亲和转 `'123'`（内存对象 ≠ 库内值，回报≠事实），
        `None` 落 NULL 且使去重候选过滤失效，拼错的 `'invalid_modal'` 无声入库
        造成下游按 modality 过滤出现意外分组。
        修法（fail-closed，先判后写、零副作用，与 #331 同款口径）：tags/entities
        经 `_validated_str_items` 校验+规范化——容器须为 list/tuple/set，**每项
        须为非空 str**，None 项跳过（判据原文）；modality 白名单校验
        （`_PERCEPTION_MODALITIES`）。调用方编程错误响亮报错（TypeError/
        ValueError），**不做静默字符串化**。
        判据来源：公开 issue #805「tags 每项须为非空 str，None 项跳过；
        modality 白名单校验」（issue 正文表格点名 tags=None→'None'、tags=数字→
        字符串、tags=''、modality 任意值四形态）。
        边界（如实标注）：①容器 `None`（含显式传 None）规范化为空而非报错——
        签名默认即 `List[str] = None`（None＝未提供），且仓内 longterm_gate.py
        两处 `entities or None` 是显式传 None 的既有合法用法；②modality 白名单
        只取仓内已实证的感知通道模态（text/audio/image/code），structure/
        protocol/self_state/trust_state 等写入路径内部模态直构 STNode 走
        `add_node`，不经本入口；③本闸只覆盖 `add_perception`，`store.add_node` /
        `import_all` 的 tags 列写入不在本件范围（与 #331 边界同款）。
        """
        if not isinstance(content, str):
            raise TypeError(
                "#331 add_perception 拒绝非字符串 content（收到 %s）：非 str 会绕过"
                "去重、并以 BLOB/NULL 静默落库，进而毒化检索面"
                % type(content).__name__)
        # ---- #805：tags/entities/modality 入参验证（fail-closed，先判后写）----
        # 在 `_interaction_count` / `_note_action` / 任何 SQL 之前（与 #331 同款
        # 「零副作用」理由）：旧形态 tags=12345 的报错发生在副作用之后（半截写入面）。
        # 校验后 tags/entities 已是纯 list[str]，下游 `tags or []` / 极性闸
        # `set(tags or [])` / 合并路径 `for t in (tags or [])` 均按原语义工作。
        tags = _validated_str_items("tags", tags)
        entities = _validated_str_items("entities", entities)
        if not isinstance(modality, str):
            raise TypeError(
                "#805 add_perception 拒绝非字符串 modality（收到 %s）：int 会被 "
                "SQLite TEXT 亲和静默转 '123'（内存对象 ≠ 库内值），None 落 NULL "
                "且使去重候选的模态过滤失效" % type(modality).__name__)
        if modality not in _PERCEPTION_MODALITIES:
            raise ValueError(
                "#805 add_perception 拒绝非白名单 modality %r：合法值 "
                "text/audio/image/code（感知通道模态，防下游按 modality 过滤出现"
                "意外分组；白名单定义见 _PERCEPTION_MODALITIES 注释）" % (modality,))
        self._interaction_count += 1
        self._note_action("perception", content, None,
                          {"importance": importance, "modality": modality})
        # ---- M5 去重：中文二元组 Jaccard ≥ 动态阈值 → 视为同源重复 ----
        # #29①：同源重复**只计复用，不增信**（旧实现无脑 `update_node_confidence(+0.02)`
        # ⇒ 同一错误陈述写 31 次置信度 0.5→1.0，「重复即真理」）。
        # #29②/#142：先判实质分歧——否定/数值变更的**更正**不是重复，必须另立节点
        # 并落账冲突（旧实现把它吞进原节点还给原命题增信）。
        # #223：去重命中**不得丢弃新输入**——旧实现只 `increment_access` + 加
        # `duplicate` 标签就 `return best`，新输入的 entities/tags/importance 全丢，
        # 且返回值与「新建节点」不可区分（`consolidate_learning_result` 等上层入口
        # 同一摘要第二次关联到别的实体时，该实体同样丢失）。现按 issue「期望行为」：
        # ①候选集按 `modality` 过滤（**modality 不同的不参与去重**——audio 输入不得被
        # 并入 text 节点）；②命中后 entities/tags 取**并集**并入、importance 取**两者
        # 较大值**；③返回值带 `merged=True` 标记（新建路径 `merged=False`），使调用方
        # 可区分「新建」与「已合并」。
        # 判据来源：公开 issue #223「期望行为：合并到已有节点时，新输入的 entities 与
        # tags 取并集并入，importance 取两者较大值；modality 不同的不参与去重；返回值
        # 能区分「新建」与「已合并」」＋ 分诊表 `triage_lingshu_ALL.md` 行 223（经验
        # 标定，仓内理论章节未规定去重命中的字段合并口径）。
        # 边界（削掉的判别力，如实标注）：`merged` 标记以**返回对象属性**承载
        # （`STNode` 非 slotted dataclass，可挂附加属性），不新增返回类型/签名——
        # 既有调用方（按 STNode 使用）不受影响；不引入 `entities`/`tags` 的**删除**
        # 语义（只并入，不移除原节点已有的标签）；modality 过滤使**跨模态**同内容
        # 不再合并（旧行为：并入 text 节点并静默丢模态），这是本件要求的判据变更。
        # #268：M5 判同键补**条件空间**——旧实现只比 `content` 二元组，同一句话在
        # 不同「存在约束 / 时间窗 / 观测位置」下被当成同一条知识（「6岁以下儿童禁用
        # 此剂量」「该剂量已撤销」两条写入都返回成人条件的旧节点并给它增信，知识层
        # 任何时候只保留一个条件版本，违背 `K=K(C)`）。
        # 判据：候选与新写入的条件空间须过 `_conditions_mergeable`（三维度字面相等
        # ＋ 时间窗有交集），否则跳过该候选、另立节点。
        # 判据来源：`docs/theory/世界模型与语义时空图_完整理论整理与实现路线.md`
        # L35「知识 = 条件空间 C 中的有效关系 K=K(C)」＋ L154「任何知识/识别须绑定
        # 条件」＋ 公开 issue #268「建议 1」（判同前先比较条件空间）。这是**理论
        # 章节**锚点（非经验标定）。
        # 边界（削掉的判别力，如实标注）：①条件只做相等/相交判据，不推理包含关系；
        # ②条件不同的同正文写入**不**自动建 SIMILAR/APPLIES_TO 边、不落冲突
        # （issue 建议 1/3 的这些半句属新增机制/策略取舍）；③`skip_dedup=True`
        # 的主动沉淀路径不受影响（照旧不去重）。
        # #268：条件空间先于去重求值——M5 判同键须含条件（`K=K(C)`）。
        cs = condition_space or ConditionSpace(
            observation_position="感知系统",
            observation_tool="感官输入",
            time_window=(time.time(), time.time()+3600),
            existence_constraint="协议实例运行中"
        )
        divergent_with = ""
        # 矛盾感知（LINGSHU_CONTRADICTION=1，默认关闭；Refs #142）在上面 #29② 判据之上**追加**：
        # 旗标关闭时只跑 `_assertion_divergence`，与 main 逐位一致；开启时 main 判据没判出分歧
        # 才再跑更宽的 `contradiction.conflict`（差异片段上的极性 / 含中文数字的数值 / 反义）。
        conflict_reason = ""
        merged = False
        # #800：M5 去重「查→插」原子化——去重判定（候选扫描+阈值+分歧）与
        # `add_node` INSERT 收进同一把 `_m5_dedup_lock`（引擎级）。旧实现两段
        # 之间无互斥：并发线程的 SELECT 都跑在对方 INSERT 之前，各自判「不存在」
        # 后各自 INSERT ⇒ 同内容产出多个重复节点（TOCTOU；见 `__init__` 处
        # #800 注释的判据来源与边界）。锁内含合并路径的整段副作用与新建路径的
        # `register_conflict`（分歧账与判同决策同临界区）；锁外的实体标签/证据
        # 记账只作用于已定 id，不参与判同，留在临界区外。
        with self._m5_dedup_lock:
            if not skip_dedup and isinstance(content, str) and content.strip():
                threshold = self._effective_dedup_threshold()
                # #223：候选集限同模态——modality 不同的不参与去重
                # #297：候选集**不再按 importance/last_access 截断前 200 条**——旧实现
                # `query_nodes(layer=KNOWLEDGE, limit=200)` 的排序是
                # `ORDER BY importance DESC, last_access DESC`（为检索服务），知识层超过
                # 200 条后，较早写入、未被访问的内容**对去重不可见** ⇒ 逐字相同的感知被
                # 重复入库（同一事实拆成多份节点，增信/实体标签/访问计数分散在副本上，
                # recall 被重复项占位）。改为**全层同模态候选**（`limit=-1`，SQLite
                # LIMIT -1 = 不限条数，与同文件 `get_layer_nodes` 同款口径）。
                # 判据来源：公开 issue #297（「去重视野只有 importance/last_access 前 200
                # 条……较早写入的内容对去重不可见」）＋ 该 issue「修复建议 2：全层扫描」
                # （经验标定；仓内理论章节未规定 M5 候选视野口径，追不到更早出处）。
                # 边界（削掉的判别力/代价，如实标注）：①本修法取 issue 建议 2（全层扫描）
                # ——写入代价随知识层规模**线性增长**（实测本机 2000 行候选装载 + 比对
                # 约 0.03s）；②issue 建议 1 的「低频二元组 `content LIKE` 预筛」在本仓
                # **无收益**——`nodes.content` 无索引，`content LIKE '%…%'` 同样是全表扫描
                # （本仓索引只有 `idx_nodes_layer/idx_edges_source/idx_edges_target`，
                # core.py:988-990），故不引入；③issue 建议 3（维持 200 并显式声明）**不**
                # 采纳——那只是把漏去重写进文档，缺陷仍在；④仍按 `modality` 过滤（#223）。
                candidates = self.store.query_nodes(layer=MemoryLayer.KNOWLEDGE,
                                                    modality=modality, limit=-1)
                best, best_sim = None, 0.0
                # #225：极性闸——**成败极性不同的节点不得合并**。旧实现只比正文二元组，
                # 而 `record_action_sequence` 把成败写成正文前缀（`成功：`/`失败：`，且
                # outcome 截到 `[:40]`）＋ `success`/`failure` 标签；动作序列越长，
                # 「成功/失败」二字占的二元组比例越小 ⇒ 相似度越过阈值，失败经验被
                # 并进「成功」节点（反之亦然）——对立结论在节点层**消失**，且以对方
                # 身份留在图里（`recall`/`ActivationEngine` 只读 nodes）。
                # 判据：候选与新写入的极性标签集必须**相等**（同为空亦相等）；「一侧有
                # 极性、另一侧无」同样判不同（宁另立节点，不静默合并）。
                # 判据来源：公开 issue #225「建议修复 1：M5 去重的候选必须极性一致：
                # `success`/`failure` 不同的节点不得合并（最小改法：候选与新写入的极性
                # 标签不一致时跳过该候选）」＋ issue #225 标题/复现（经验标定；仓内
                # 理论章节未规定 M5 判同键的极性口径，追不到更早出处）。
                # 边界（削掉的判别力，如实标注）：①极性词表只取 `success`/`failure`
                # 两个既有标签，不引入 `negative`/`conflict` 等**尚不存在**的极性标签
                # （issue 括注「以及将来的…」属未定项，不擅定）；②本闸只挡「合并」，
                # 不改 `record_action_sequence` 的 outcome 截断口径（issue 建议 3 的
                # 「保留末尾退出码」是另一条独立面）；③极性不同的两条写入**不**自动建
                # `OPPOSITE` 边/落冲突（issue 建议 4 的 fail-loud 属策略取舍，本件只做
                # 「不合并」这一最小面）。
                new_polarity = set(tags or []) & self._POLARITY_TAGS
                # #803：判同键先剥零宽字符——零宽变体（ZWJ/ZWNJ/ZWSP/BOM/WJ）视觉相同
                # 但字节不同，旧键直接比原始 content ⇒ 变体绕过 M5 去重各自成节点。
                # 比对两侧都剥除：新写入侧与候选（库中既存节点）侧。
                norm_new = self._strip_zero_width(content)
                for n in candidates:
                    if (set(n.tags or []) & self._POLARITY_TAGS) != new_polarity:
                        continue
                    # #268：条件空间也是判同键——条件不同即不同知识（见 docstring）
                    if not self._conditions_mergeable(cs, n.condition_space):
                        continue
                    sim = LayeredStore.char_bigram_jaccard(
                        norm_new, self._strip_zero_width(n.content))
                    self._dedup_history.append(sim)
                    if len(self._dedup_history) > self._dedup_window:
                        self._dedup_history = self._dedup_history[-self._dedup_window:]
                    if sim > best_sim:
                        best_sim, best = sim, n
                if best and best_sim >= threshold:
                    # #803：分歧/矛盾判据同口径（两侧剥零宽）。
                    norm_best = self._strip_zero_width(best.content)
                    divergence = self._assertion_divergence(norm_new, norm_best)
                    if _contradiction.enabled():
                        conflict_reason = (_contradiction.conflict(norm_new, norm_best)
                                           or _MAIN_DIVERGENCE_REASON.get(divergence, ""))
                        divergence = divergence or conflict_reason
                    if not divergence:
                        self.store.increment_access(best.id)
                        if getattr(self, "_evidence", None) is not None:
                            # 证据账本开启时记一条证据（同源重复不增信由账本保证，见 core/evidence.py）
                            self._evidence.record(best.id, source, self_generated=self_generated,
                                                  note="M5 去重命中")
                        self.store.tag_node(best.id, "duplicate")
                        # #223②：新输入的 tags / entities 取并集并入（旧实现全丢）
                        for t in (tags or []):
                            self.store.tag_node(best.id, t)
                        for eid in (entities or []):
                            self.store.tag_node(best.id, f"ent:{eid}")
                        # #223②：importance 取两者较大值（旧实现：以 0.95 重复写入，
                        # 原节点仍是 0.3）
                        merged_node = self.store.get_node(best.id) or best
                        if importance > merged_node.importance:
                            self.store.update_node_importance(
                                best.id, importance - merged_node.importance)
                        merged_node = self.store.get_node(best.id) or merged_node
                        merged_node.merged = True   # #223③：区分「已合并」
                        return merged_node
                    divergent_with = best.id
            semantic_coords = {}
            if self._semantic_provider is not None:
                # v1.7：结构化语义坐标（D-003 字段分离；文本节点写入 concept/radical/neural）
                semantic_coords = self._semantic_provider.to_semantic_coordinates(content)
            node = STNode(
                id=f"node_{uuid.uuid4().hex[:8]}_{int(time.time()*1000)}",
                content=content,
                modality=modality,
                spatial_coordinates=spatial_coordinates or {},
                temporal_coordinate=time.time(),
                condition_space=cs,
                importance=importance,
                confidence=0.5,
                layer=MemoryLayer.KNOWLEDGE,
                tags=tags or [],
                semantic_coordinates=semantic_coords
            )
            self.store.add_node(node)
            if divergent_with:
                # 矛盾落账（#29②）：更正语句与原命题建 OPPOSITE 边 + 双方 conflict 标签，
                # 待验证单元复核——不再静默「入库即一致」。
                self.store.register_conflict(node.id, divergent_with)
        if entities:
            for eid in entities:
                self.store.tag_node(node.id, f"ent:{eid}")
        if getattr(self, "_evidence", None) is not None:
            self._evidence.record(node.id, source, self_generated=self_generated,
                                  note="首次写入")
            node.confidence = self.store.get_node(node.id).confidence
        if divergent_with and conflict_reason:
            # 矛盾感知开启时，在同一 register_conflict 通道之上追加原因标签
            for nid in (node.id, divergent_with):
                self.store.tag_node(nid, f"conflict:{conflict_reason}")
        node.merged = merged   # #223③：新建路径 False（与合并路径 True 可区分）
        return node

    # ==================== 证据账本（置信度由证据推导 · 默认关闭） ====================

    def enable_evidence_ledger(self):
        """启用证据账本：此后 add_perception 的增信改为记证据、置信度由账本推导。"""
        if getattr(self, "_evidence", None) is None:
            from .evidence import EvidenceLedger
            self._evidence = EvidenceLedger(self.store)
        return self._evidence

    def add_evidence(self, node_id: str, source: str, supports: bool = True,
                     weight: float = 1.0, note: str = "",
                     self_generated: bool = False) -> Dict:
        """为节点记一条外部证据（supports=False 即反证），返回更新后的评估。"""
        from .evidence import SUPPORT, CONTRADICT
        ledger = self.enable_evidence_ledger()
        ledger.record(node_id, source, SUPPORT if supports else CONTRADICT,
                      weight=weight, self_generated=self_generated, note=note)
        return ledger.assess(node_id)

    def evidence_report(self, node_id: str) -> Dict:
        """只读：节点的证据评估（置信度、状态、支持/反驳来源）；启用 TMS 时附依据与撤回信息。"""
        if getattr(self, "_tms", None) is not None:
            return self._tms.assess(node_id)
        return self.enable_evidence_ledger().assess(node_id)

    # ==================== 真值维护（依赖撤回 · 默认关闭） ====================

    def enable_tms(self):
        """启用真值维护（隐含启用证据账本）：结论的置信度不高于其最强依据中最弱的前提；
        前提被反证推翻后，下游结论自动撤回为「前提已被推翻」（见 core/tms.py）。"""
        if getattr(self, "_tms", None) is None:
            from .tms import TruthMaintenance
            ledger = self.enable_evidence_ledger()
            self._tms = TruthMaintenance(self.store, ledger)
            ledger.on_change = lambda nid: self._tms.propagate([nid])
        return self._tms

    def add_justification(self, conclusion_id: str, premise_ids: List[str],
                          source: str = None) -> str:
        """登记一条依据：conclusion ← premises（合取）。同一结论可登记多条（析取）。"""
        return self.enable_tms().add_justification(conclusion_id, premise_ids, source)

    def recall_with_evidence(self, context_content: str, limit: int = 10,
                             include_refuted: bool = False) -> List[Dict]:
        """召回并附上证据评估，供宿主决定「当事实说 / 带保留说 / 不说」。

        与 recall() 同序；已被反驳的条目默认剔除（include_refuted=True 保留）。
        """
        ledger = self.enable_evidence_ledger()
        tms = getattr(self, "_tms", None)
        out = []
        for node, score in self.recall(context_content, limit=limit):
            ev = tms.assess(node.id) if tms is not None else ledger.assess(node.id)
            # 启用 TMS 时，「前提已被推翻」与「已被反驳」同等对待（默认不进结果）
            if ev["status"] in ("refuted", "undermined") and not include_refuted:
                continue
            out.append({"node": node, "score": score, "evidence": ev})
        return out

    # ==================== 锚点层操作 ====================

    def set_anchor(self, content: str, importance: float = 1.0,
                   condition_space: ConditionSpace = None) -> STNode:
        """写入锚点层（不可遗忘）"""
        cs = condition_space or ConditionSpace(
            observation_position="协议初始化",
            observation_tool="协议定义",
            time_window=(0, float('inf')),
            existence_constraint="协议基底"
        )
        node = STNode(
            id=f"anchor_{uuid.uuid4().hex[:8]}_{int(time.time()*1000)}",
            content=content,
            modality="protocol",
            spatial_coordinates={},
            temporal_coordinate=time.time(),
            condition_space=cs,
            importance=importance,
            confidence=1.0,
            layer=MemoryLayer.ANCHOR,
            tags=["anchor"]
        )
        self.store.add_node(node)
        return node

    def get_anchors(self) -> List[STNode]:
        """取锚点层节点（不可遗忘的核心锚集合）。"""
        return self.store.get_layer_nodes(MemoryLayer.ANCHOR)

    # ==================== 结构层操作 ====================

    def add_structure_node(self, content: str, importance: float = 0.8,
                           condition_space: ConditionSpace = None) -> STNode:
        """写入结构层（不可遗忘）"""
        cs = condition_space or ConditionSpace(
            observation_position="结构定义",
            observation_tool="协议架构",
            time_window=(0, float('inf')),
            existence_constraint="协议结构"
        )
        node = STNode(
            id=f"struct_{uuid.uuid4().hex[:8]}_{int(time.time()*1000)}",
            content=content,
            modality="structure",
            spatial_coordinates={},
            temporal_coordinate=time.time(),
            condition_space=cs,
            importance=importance,
            confidence=1.0,
            layer=MemoryLayer.STRUCTURE,
            tags=["structure"]
        )
        self.store.add_node(node)
        return node

    # ---- 结构层「重要事件」记录惯例（#329 唯一真源 · 引擎转发件） ----
    # 唯一真源（常量 SUB_PENDING_SYNC_TAG + 实现）已随 PR #429 下沉到 LayeredStore
    # （structure_service 直接用 store 入口）；本方法只转发，保留引擎侧调用形态。
    def _write_structure_record(self, record: str, tag: str,
                                extra_tags: List[str] = None, *,
                                source: str = "", metadata: Dict = None,
                                sub_modality: str = "text",
                                record_class: str = "history",
                                importance: float = 0.9,
                                confidence: float = None,
                                _cursor=None) -> STNode:
        """结构层「重要事件」记录惯例（#329 唯一收口 · 转发件）。

        第二实参 `tag` 是事件类型；正文的 `[tag] ` 前缀由
        `_record_structure_event` 统一添加（调用方**不再**自带前缀）。

        实现转发到 LayeredStore 的唯一真源（PR #429 的受保护留史）：
        PRIMARY ⇒ 写结构层（不可遗忘）；SUB ⇒ 写知识层副本，并**一律**带
        `SUB_PENDING_SYNC_TAG`（待父节点同步为验证副本）。
        留史失败**向调用方传播**，不再静默返回 None——静默会让
        「留史失败即回滚数据变更」失效（见 `migrate_v17_coordinates` 的 #429 事务口径）。
        """
        return self.store._write_structure_record(
            record, tag, extra_tags=extra_tags, source=source, metadata=metadata,
            sub_modality=sub_modality, record_class=record_class,
            importance=importance, confidence=confidence, _cursor=_cursor)

    def get_current_structure_nodes(self) -> List[STNode]:
        """现行结构：排除历史，只读最新、未撤销、未到期的批准。

        未使用新事件格式的旧结构保留原读取语义；完整历史仍用 store 读取。
        """
        nodes = self.store.get_layer_nodes(MemoryLayer.STRUCTURE)
        active = set()
        if self.store.get_meta("structure_writer_schema"):
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc).isoformat(timespec="microseconds")
            rows = self.store.conn.execute(
                "SELECT r.id FROM structure_heads h JOIN structure_records r ON r.id=h.record_id "
                "JOIN structure_proposals p ON p.id=r.proposal_id "
                "WHERE p.operation='set' AND (p.valid_until IS NULL OR p.valid_until>?)", (now,))
            active = {row[0] for row in rows}
        result = []
        for node in nodes:
            event = node.state_attributes.get("structure_event")
            if not event or event.get("record_class") == "external_anchor":
                result.append(node)
            elif event.get("record_class") == "normative" and node.id in active:
                result.append(node)
        return result

    # ==================== 边操作 ====================

    def add_edge(self, source_id: str, target_id: str,
                 relation_type: EdgeType = EdgeType.CAUSAL,
                 confidence: float = 0.5,
                 condition_space: ConditionSpace = None,
                 source_evidence: str = "extracted",
                 dedup: bool = True) -> STEdge:
        """添加边（默认未验证）。source_evidence（P1-1）：extracted/inferred/ambiguous

        #801：写入前查同 (source_id, target_id, relation_type) 的既有边——存在则
        **返回已有边**（带 `merged=True` 属性标记），不再插入平行副本（旧实现同参数
        调 5 次产出 5 条独立边：`find_cycles`/`shortest_path`/`infer_causal_paths`
        把同一事实计多次，存储无界增长——节点侧自 v1.26c 有 M5 去重，边侧零去重）。
        判据来源：公开 issue #801 复现读数（5 次 add_edge → 5 个 edge_id）＋ 其
        「修复建议 1/2」。命中口径取**返回已有边**而非 raise——与节点侧 M5 去重
        先例（#223：命中返回原节点并带 `merged=True`）一致；`merged` 以返回对象
        属性承载（STEdge 非 slotted dataclass，同 STNode 的 #223 先例），新建路径
        恒 `merged=False`，既有调用方按 STEdge 使用不受影响。同 (source, target)
        **不同 relation_type** 仍各自成边（语义不同的关系不是重复）。
        `dedup=False`（issue 建议 2 的逃生口）：显式要求多条平行边时跳过查重照旧
        插入（#327 守卫的平行边构图、需要 k 条平行边的枚举压测走此参数）。
        边界（削掉的判别力/未覆盖面，如实标注）：①只挡**新增**面——库内既有平行边
        不做存量迁移合并（#327 当年裁「合并+迁移会改既有边 id，波及衰减/验证/同步
        面」，本件同样不动存量）；②命中后既有边的 confidence/verified/source_evidence
        **不与新参数合并**（与 M5「只计复用不增信」#29① 同向：重复添加不增置信度）；
        ③`LayeredStore.add_edge(STEdge)` 直写面不去重（同步/导入路径须保真，非本件
        收口面）；④查重是 check-then-act，与 `add_node` 的旧行检查同款并发口径。"""
        rt = relation_type if isinstance(relation_type, EdgeType) else EdgeType(relation_type)
        if dedup:
            row = self.store.conn.execute(
                "SELECT * FROM edges WHERE source_id=? AND target_id=? AND relation_type=? "
                "ORDER BY created_at, id LIMIT 1",
                (source_id, target_id, rt.value)).fetchone()
            if row is not None:
                existing, _ = self.store._tolerant_edges([row])
                if existing:
                    hit = existing[0]
                    # 写权限闸先行（#234/#769/#780，单一真源 `_assert_edge_endpoints_role`，
                    # 与 store.add_edge 同判据）：去重命中虽不写入，但写 API 的 fail-closed
                    # 契约不得被早退绕过——只读/越层角色即便命中既有边也必须 PermissionError
                    # （CI 红：test_issue769_780 g9 的 sub.add_edge 既有边未拦）。
                    self.store._assert_edge_endpoints_role(source_id, target_id)
                    hit.merged = True  # 与 M5 命中的 #223 标记同款（可区分「已存在」与「新建」）
                    self._note_action("relation",
                                      f"{source_id}→{target_id}",
                                      [source_id, target_id],
                                      {"relation": rt.value,
                                       "evidence": source_evidence,
                                       "dedup_hit": hit.id})
                    return hit
        cs = condition_space or ConditionSpace(
            observation_position="关系建立",
            observation_tool="语义分析",
            time_window=(time.time(), time.time()+3600),
            existence_constraint="协议实例运行中"
        )
        # #796（fail-closed，任何写语句之前抛出）：`confidence` 的非有限值
        # （NaN/±inf）经 `_require_finite` 拒收（`ValueError`），不再被
        # `STEdge.to_row` 的 `_finite_clamp` 静默归一为 0.0——修前实测（main @
        # 40fceb6）：NaN/+inf/-inf 一律 ACCEPTED、库内落 0.0，且返回的内存对象
        # 仍带原始坏值（对象与库行不一致），调用方编程错误被无信号掩盖。与 #790
        # `verify_edge` 的 `new_confidence` 同款闸（#206 `_require_finite`
        # 单一真源——创建路径与更新路径同一缺口收口）。有限值口径不动：[0,1] 外
        # 的有限值（如 1.5）仍走 #185 的 `_finite_clamp` 钳制（值域归一不属
        # 本件收口面）；本入口无 `weight` 形参（STEdge 缺省 1.0，直写面
        # `LayeredStore.add_edge(STEdge)` 的归一口径不变）。
        confidence = _require_finite("confidence", confidence)
        edge = STEdge(
            id=f"edge_{uuid.uuid4().hex[:8]}_{int(time.time()*1000)}",
            source_id=source_id,
            target_id=target_id,
            relation_type=rt,
            condition_space=cs,
            confidence=confidence,
            verified=False,
            source_evidence=source_evidence
        )
        edge.merged = False  # 新建路径恒 False（与 M5 新建节点同款，见 #223）
        self.store.add_edge(edge)
        self._note_action("relation",
                          f"{source_id}→{target_id}",
                          [source_id, target_id],
                          {"relation": edge.relation_type.value,
                           "evidence": edge.source_evidence})
        return edge

    def verify_edge(self, edge_id: str, new_confidence: float = None):
        """验证单元复核后调用（转发 store 层；#790：`new_confidence` 非有限值
        （NaN/±inf）在 store 层被 `_require_finite` 拒收，`ValueError` 原样上抛）。"""
        self.store.verify_edge(edge_id, new_confidence)

    # ==================== 时空联想查询 ====================

    def spatiotemporal_query(self, node_id: str, time_radius: float = 300.0,
                             space_metric: str = None, space_radius: float = 0.5,
                             max_results: int = 20) -> List[Tuple[STNode, float]]:
        """时空联想查询：以节点为锚按时间+空间双半径加权取近邻（转发 store 层）。"""
        return self.store.spatiotemporal_query(node_id, time_radius, space_metric, space_radius, max_results)

    # ==================== 因果推理 ====================

    def reason_causal(self, start_id: str, end_id: str = None,
                      max_depth: Optional[int] = 5,
                      relation_types: List[str] = None,
                      importance_weighted: bool = False,
                      include_subgraph: bool = False,
                      max_paths: int = _CAUSAL_MAX_PATHS,
                      causal_stats: Dict = None) -> List[List[STEdge]]:
        """
        因果推理（v1.16 图架构增强）：
        - 指定 end_id：查找 start→end 的所有路径（relation_types 可多类型）
          ——与「未指定」同口径：到深度预算仍有出边 ⇒ 存截断链并标 truncated；
          空列表=确无因果路径，不再混同于「链太长」（issue #145 同族补修，
          此前这一半静默返回空，见 tests/test_causal_end_id_truncation.py）
        - 未指定：返回从 start 出发的所有链（每条链最长 max_depth），元素为
          CausalChain（list 子类）——到深度预算仍有出边 ⇒ 存截断链并标
          truncated；链尾遇回边/自环 ⇒ 存链并标 cyclic（可达部分不丢弃）；
          空列表=确无因果后果，不再混同于「链太长/有环」（issue #145）
        - max_depth：正整数=深度预算；**None=不限深度**（显式要求「必须找到」）
        - causal_stats：可选出参 dict，两分支同口径填充
          `{max_depth, truncated, complete, truncated_dropped}`——截断事实外传

        #327：两个分支都受 `max_paths`（默认 `_CAUSAL_MAX_PATHS`）约束。旧实现
        无上限——同一条因果事实被记录 k 次时路径数是 k⁵，6 节点无环链 k=24 约
        800 万条使长驻进程 OOM（issue 实测）。达到上限即停止枚举。
        边界（削掉的判别力，如实标注）：达上限时返回**前 max_paths 条**，未加
        `truncated` 标记（会改返回类型/元素语义，波及 #145 的 CausalChain 契约）。
        #482（本件）：`end_id` 分支的上限原先只数「命中 end 的结果」⇒ 终点不可达
        时上限永不触发（该分支的绕过点与计数口径详见 `infer_causal_paths`）。现按
        **终态分支**计数，不可达查询的工作量同样受 `max_paths` 约束；不传 `end_id`
        的分支不受影响（其上限本就挂在 `chains` 条数上）。

        #385：`collect` 由**递归**改**显式栈**（帧含节点、到达路径、路径节点集、
        深度与**边迭代器**）——旧实现递归深度 = 链长 = `max_depth`，1500 节点线性
        链上抛 `RecursionError`。用「帧 + 迭代器」而非「一次性压栈全部子帧」，是为
        了**逐字保持**递归版的输出顺序。边界：`max_depth` / `truncated` / `cyclic`
        / 空列表语义一律不动。

        #328（**改编采纳** · 如实标注一处**不采纳**）：传 `end_id` 的分支采纳
        「截断证据不静默」；两个分支都填充 `causal_stats`。**不采纳**本件对
        「不传 end_id」分支的截断证据**条数上界**——本仓 #385 守卫
        （`tests/test_issue385_iterative_dfs_no_recursion_error_guard.py::
        test_q3_output_order_matches_recursive_reference`）要求截断链不过滤、与
        递归参考逐字一致，与「丢弃超上界截断证据」**互斥**；故该分支保持主干
        **无上界**截断链（`truncated_dropped` 恒 0）。
        """
        if end_id:
            paths = self.store.infer_causal_paths(
                start_id, end_id, max_depth, relation_types=relation_types,
                max_paths=max_paths, causal_stats=causal_stats)
        else:
            chains: List[CausalChain] = []
            truncated_dropped = 0

            # 帧 = [节点, 到达路径, 路径节点集, 深度, 边迭代器(None=未展开)]
            frames = [[start_id, [], {start_id}, 0, None]]
            while frames:
                if len(chains) >= max_paths:
                    break
                fr = frames[-1]
                cur, path, on_path, depth, it = fr
                if it is None:
                    edges = self.store.get_outgoing_edges(cur)
                    if relation_types:
                        tset = {t.lower() for t in relation_types}
                        kept = [e for e in edges if e.relation_type.value in tset]
                    else:
                        kept = [e for e in edges if e.relation_type == EdgeType.CAUSAL]
                    if not kept:
                        # 天然终点：无出边的链（唯一的「无因果后果」形态）
                        if path:
                            chains.append(CausalChain(path))
                        frames.pop()
                        continue
                    if max_depth is not None and depth >= max_depth:
                        # 深度预算用尽仍有出边：存截断链（不丢弃、不混同于无后果）。
                        # #328 不采纳条数上界——与 #385 守卫互斥（见 docstring）。
                        if path:
                            chains.append(CausalChain(path, truncated=True))
                        frames.pop()
                        continue
                    fr[4] = iter(kept)
                    continue
                advanced = False
                for e in it:
                    if len(chains) >= max_paths:
                        break
                    if e.target_id in on_path:
                        # 回边/自环：路径内已访问 ⇒ 存带环链（不吞掉可达部分）
                        chains.append(CausalChain(list(path) + [e], cyclic=True))
                        continue
                    frames.append([e.target_id, path + [e],
                                   on_path | {e.target_id}, depth + 1, None])
                    advanced = True
                    break
                if not advanced:
                    frames.pop()
            paths = chains
            if causal_stats is not None:
                # 三条出口形态互斥，显式分解：complete + truncated + cyclic。
                cyclic_count = len([c for c in chains
                                    if getattr(c, "cyclic", False)])
                truncated_count = len([c for c in chains
                                       if getattr(c, "truncated", False)])
                causal_stats.update({
                    "max_depth": max_depth,
                    "complete": len(chains) - truncated_count - cyclic_count,
                    "truncated": truncated_count + truncated_dropped,
                    "truncated_dropped": truncated_dropped,
                    "cyclic": cyclic_count,
                })
        # importance 加权排序
        if importance_weighted and paths:
            def _imp_mean(p):
                ids = [e.source_id for e in p] + [p[-1].target_id]
                ph = ",".join("?" * len(ids))
                rows = self.store.conn.execute(
                    f"SELECT id, importance FROM nodes WHERE id IN ({ph})",
                    ids).fetchall()
                imp = {r[0]: r[1] for r in rows}
                vals = [imp.get(i, 0.5) for i in ids]
                return sum(vals) / len(vals) if vals else 0.5
            # **完整路径优先**：先按 truncated 位分层（完整在前），层内再按原口径
            # （importance 均值、长度）排序。否则 importance 高的截断链会抢占首元素，
            # 破坏「已有完整解时首元素是最短完整路径」的既有读数。
            paths.sort(key=lambda p: (bool(getattr(p, "truncated", False)),
                                      -_imp_mean(p), len(p)))
        # 子图感知：尾节点知识点子图
        if include_subgraph:
            decorated = []
            for p in paths:
                tail = p[-1].target_id if p else start_id
                try:
                    sg = self.store.subgraph(tail, max_depth=2)
                except Exception:
                    sg = None
                decorated.append({"path": p, "subgraph": sg})
            return decorated
        return paths

    # ==================== 自我认知层 ====================

    def update_self(self, updates: Dict, link_to_node_id: str = None):
        """更新自我模型（运行状态描述字段；白名单见 `SelfModel.UPDATABLE_FIELDS`）。
        link_to_node_id：自我快照与指定记忆节点建边（M3 自我认知组装路径）

        注意（#182）：`identity` / `values` / `trust_state` 等敏感字段**不经本入口**
        ——分别走构造器 / `record_value_change` / `update_trust_state`；经本入口传入
        即 `PermissionError`（旧实现是无门控 setattr，任何字段/方法/类常量皆可改写）。

        #201：每次调用新建的 SELF 快照现**有界保留**（`SELF_SNAPSHOT_MAX`）——
        旧实现无清理路径，快照无界堆积（importance=0.9 / confidence=1.0）并挤占
        普通检索召回；且 SELF 层在 `IMMUTABLE_LAYERS` 内、`delete_node` 拒删 ⇒
        没有任何回收面。"""
        self.self_model.update(**updates)
        # 同时在自我层记录一次自我状态快照
        snapshot = STNode(
            id=f"self_{uuid.uuid4().hex[:8]}_{int(time.time()*1000)}",
            content=json.dumps(self.self_model.to_dict(), ensure_ascii=False),
            modality="self_state",
            spatial_coordinates={"self_axis": 1.0},
            temporal_coordinate=time.time(),
            condition_space=ConditionSpace(
                observation_position="自我认知",
                observation_tool="内省",
                time_window=(time.time(), time.time()+3600),
                existence_constraint="协议实例运行中"
            ),
            importance=0.9,
            confidence=1.0,
            layer=MemoryLayer.SELF,
            tags=["self_snapshot"]
        )
        self.store.add_node(snapshot)
        # #201：自我快照有界保留（FIFO 淘汰最旧快照），防无界堆积挤占召回。
        # #489：prune 按插入序（rowid）淘汰 ⇒ 本次刚插入的快照 rowid 最大，不会
        # 被自己这次淘汰选中（旧实现按墙钟 created_at，时钟回拨时自删，并令下方
        # add_edge 抛 FOREIGN KEY constraint failed）。
        self.store.prune_self_snapshots(self.SELF_SNAPSHOT_MAX)
        if link_to_node_id and self.store.get_node(link_to_node_id):
            self.add_edge(snapshot.id, link_to_node_id, EdgeType.CAUSAL, confidence=1.0)

    def prune_self_snapshots(self, keep: int = None) -> int:
        """回收自我快照：只留最近 `keep` 条（缺省 `SELF_SNAPSHOT_MAX`），返回删除条数。

        #201 的「无清理路径」补齐——SELF 层在 `IMMUTABLE_LAYERS` 内、`delete_node`
        拒删，故清理面收在 store 级 `prune_self_snapshots`（只动 `self_snapshot`
        标签的自动快照，不动其它 SELF 节点与 `protections` 名单）。"""
        return self.store.prune_self_snapshots(
            self.SELF_SNAPSHOT_MAX if keep is None else keep)

    def prune_action_logs(self, keep_rows: int = None) -> int:
        """回收治理审计表：只留最近 `keep_rows` 行（缺省 `ACTION_LOGS_KEEP_ROWS`），
        返回删除条数。

        #723：与 `prune_self_snapshots` 同款的「无清理路径补齐」——`action_logs`
        此前只追加、无任何回收面（DDL 无上限），而高频写点有三（维护周期 / 不可变
        层边剥离审计 / 升级点开关），长程部署年化约 53 万行/进程。清理面收在 store
        级 `prune_action_logs`，本方法只是引擎侧的同名入口（便于从公共面调用）。
        ★**不被自动调用**：本表在多处声明为「审计只追加（无更新/删除路径）」，
        自动裁剪与保留窗口取值属设计者裁面（见 store 级 docstring 与交付报告）。
        """
        return self.store.prune_action_logs(
            self.ACTION_LOGS_KEEP_ROWS if keep_rows is None else keep_rows)

    def get_self_model(self) -> SelfModel:
        """取自认知模型（SelfModel：价值观/信任史/状态变更历史的聚合快照）。"""
        return self.self_model

    # ---- #33：自我模型的持久化与启动恢复 ----

    #: 自我模型权威快照的 `engine_meta` 键（D-26「单点权威键」，与 `_schema_version`
    #: 同款手法）。判据来源：`docs/plans/删除恢复持久化完整性_修复设计_v0.1.md`
    #: §5.4 S-33-2（推荐 engine_meta 单点权威键，避免「扫层取最新」在多次
    #: `update_self` 下选错）。
    SELF_MODEL_META_KEY = "self_model_snapshot"

    #: #479：import_all 需密钥的 engine_meta 治理键（身份/信任持久面）。
    #: 纯技术键（_schema_version/_last_decay_at）无治理载荷，不在闸面——整表拒绝
    #: 会打掉 schema 版本恢复（值守复核 6164742040 权衡段，编排侧裁定收窄到载荷键）。
    #:
    #: #671（本笔 · 最小枚举补齐）：`#479` 的枚举只列 `self_model_snapshot` 一个键，
    #: 而**启动期被回读**的治理载荷键至少还有四个（复核实测）：
    #:   · `verifier_config` —— `_restore_structure_configuration` 无钳制原样恢复
    #:     （`dedup_static` 0.85→0.0 生效，绕过 `#118` 的 `[0.5,0.95]` 钳制）；
    #:   · `attention_weights` —— 同函数恢复注意力偏好权重；
    #:   · `context_max` —— `_load_context_max()` 重启后 200→1，情境层 FIFO 上限被改写；
    #:   · `structure_writer_schema` —— 重启后 `_structure_writer_active` False→True。
    #: 上述四键与 `self_model_snapshot` 同属「无密钥可改且启动回读」的治理载荷面，
    #: 本笔一并纳入闸内。判据来源：公开 issue #671 ＋ `#479` 的同一把闸。
    #: **取舍（如实标注，留设计者裁）**：本笔取**最小枚举补齐**，不改成 deny-by-default
    #: 放行白名单——后者是更大的契约变更（「哪些键能跨备份存活」），属设计者裁面。
    #: 为防「枚举天然会漏」重演，另立机械闭合腿 `ENGINE_META_STARTUP_KEYS` ＋
    #: `ENGINE_META_NONGOVERNANCE_KEYS`（见下），由
    #: `tests/test_issue671_engine_meta_guarded_keys_guard.py` 的 G5 扫描源码闭合：
    #: 将来新增一个启动回读键而忘了分类 ⇒ 守卫红。
    #:
    #: #768（本笔 · 单一真源）：集合**上提到 `LayeredStore.ENGINE_META_GUARDED_KEYS`**
    #: ——写入口 `set_meta` / `merge_meta` 在 store 层，闸与键集分处两层时各写一份必
    #: 漂移。本属性改为**别名引用**（不是复制），恒等性由
    #: `tests/test_issue768_engine_meta_governance_guard.py` 的 `is` / `==` 双查钉死。
    ENGINE_META_GUARDED_KEYS = LayeredStore.ENGINE_META_GUARDED_KEYS

    #: #671：源码内**全部** `engine_meta` 读点出现的键（机械闭合腿的分类全集）。
    #: 判据来源：`git grep -n 'get_meta(' lingshu/` ＋ `engine_meta WHERE key=` 的
    #: 全部读点逐条核对（core.py 的 `_init_tables` / `__init__` /
    #: `_restore_structure_configuration` / `get_current_structure_nodes` /
    #: `_load_self_snapshot` / `_load_context_max` / `decay_cycle`，以及
    #: `structure_writer.py` 的版本读点）。
    #: 守卫 G5 断言 `ENGINE_META_GUARDED_KEYS ∪ ENGINE_META_NONGOVERNANCE_KEYS`
    #: **恰等于**本集合 ⇒ 新增键漏分类即红。
    ENGINE_META_STARTUP_KEYS = frozenset({
        SELF_MODEL_META_KEY,          # 身份/信任持久快照（#479）
        "verifier_config",            # 验证/去重配置（#671）
        "attention_weights",          # 注意力偏好权重（#671）
        "context_max",                # 情境层 FIFO 上限（#671）
        "structure_writer_schema",    # 结构写入器开关（#671）
        "_schema_version",            # schema 版本（非治理：灾备恢复必需）
        "_last_decay_at",             # 上次衰减墙钟（非治理：纯技术时间戳）
    })

    #: #671：**非治理**键白名单（无治理载荷，无密钥可导入）。
    #: 判据来源：`#479` 的收窄理由原文「不整表拒绝——整表拒绝会打掉 schema 版本恢复」
    #: ＋ 本笔复核（`_schema_version` 只被 `_init_tables` 与 schema 恢复读；
    #: `_last_decay_at` 只被 `decay_cycle` 读，是纯技术时间戳，不改变任何行为分支）。
    ENGINE_META_NONGOVERNANCE_KEYS = frozenset({"_schema_version", "_last_decay_at"})


    def _save_self_model(self) -> None:
        """把运行态自我模型写入 `engine_meta` 权威键（#33 落库出口）。

        由 `SelfModel._persist_hook` 在**模型的全部变更出口**（`update` /
        `record_value_change` / `update_trust_state`）回调，故 `record_value_change`
        这条由仓外自我认知引擎驱动的路径也被覆盖（旧实现该路径不落库 ⇒ `values`
        重启即回默认）。

        #483（本笔 · 数据丢失 P0）：旧实现**不读库**，把本实例内存副本整份
        `INSERT OR REPLACE` 回去 ⇒ 凡本实例内存落后于库（多进程共库时另一实例刚写过；
        `import_all` 灾备恢复后内存未刷新），第一次普通写入就把别人的更新整份抹掉，
        且 `integrity_ok=True` / `restore=restored`，无任何信号。现改为
        **写前读回 ＋ 三方合并**（`_merge_self_model_value`，判据基准
        `_self_model_baseline`），且读与写收在**同一写事务**内
        （`LayeredStore.merge_meta`，最外层 `BEGIN IMMEDIATE` 先取写锁 ⇒ 读到的必是
        最新已提交态，其间无写者可插入）。

        削掉的判别力（边界，如实标注）：合并粒度是**字段级**——同一字段被两方并发
        改写时仍后写者胜（不对 `trust_history` / `value_evolution` 做逐条归并，那属
        语义裁定面）；且合并**不回填内存**（本实例内存保持原样，只是落库值不再抹掉
        他方字段）。跨写者的同字段语义合并留设计者裁。

        保存失败时不推进本地合并基线：待保存改动继续被视为本地改动，可随下一次
        正常保存一起重试。先取得 SQLite 写事务，再进入原有 store 锁，保持锁顺序。
        调用方已有事务时仍参与该事务；不新增对调用方稍后回滚的运行态同步保证。
        """
        # Reserve the SQLite write transaction before the existing store lock,
        # matching merge_meta's lock order. Publish only the exact local input
        # whose write succeeded; a rejected save must leave it dirty for retry.
        with self.store._tx():
            with self.store._lock:
                local = self.self_model.to_dict()
                self.store.merge_meta(
                    self.SELF_MODEL_META_KEY,
                    lambda raw: self._merge_self_model_value(raw, local))
                self._commit_locked()
                self._self_model_baseline = local

    def _merge_self_model_value(self, raw: Optional[str], local: Optional[Dict] = None) -> str:
        """`LayeredStore.merge_meta` 的取值回调：读回库内快照，按字段判归属（#483）。

        三方合并：基线（`_self_model_baseline`，本实例上次同步时的内存快照）／本实例
        现值（local）／库内现值（disk）。字段归属判据＝**本实例是否改过**：
          · `local[k] == base[k]` ⇒ 本实例未改过 ⇒ 采纳 `disk[k]`（他方更新不丢）；
          · `local[k] != base[k]` ⇒ 本实例改过 ⇒ 保留 `local[k]`（本实例的变更不丢）。
        本回调只计算载荷；基线由 `_save_self_model` 在写入成功后更新为 `local`
        （**不是** merged）——内存未回填，故该字段仍处
        「本实例未改动」态；若把基线置成 merged，下一笔写入就会把刚采纳的他方值
        当成「本地改动」再覆写回去（第二笔写复活丢更新）。

        库内快照不可解析（`raw` 坏）时按「无他方更新」处置（`merged = local`，与旧
        行为同效）；无基线（理论上不会——构造器已置）亦同。
        """
        disk = None
        if raw:
            try:
                disk = json.loads(raw)
            except (TypeError, ValueError):  # 坏快照：按「无他方更新」处置，不阻断写入
                disk = None
        if local is None:
            local = self.self_model.to_dict()
        base = getattr(self, "_self_model_baseline", None)
        merged = dict(local)
        if isinstance(disk, dict) and isinstance(base, dict):
            for k, v in disk.items():
                if k in merged and merged[k] == base.get(k):
                    merged[k] = v
        return json.dumps(merged, ensure_ascii=False)

    def _load_self_snapshot(self) -> Optional[Tuple[Dict, str]]:
        """读库内自我模型快照，返回 `(data, source)`；无快照返回 None。

        先权威键（`engine_meta`），再回退 SELF 层最新 `self_snapshot` 节点——
        回退腿覆盖「本件修复前已运行过的库」（有 SELF 层快照但无权威键）。
        取最新按**插入序** `rowid` 降序（#489；与 `prune_self_snapshots` 同键，
        墙钟 `created_at` 不单调——时钟回拨后「最新」会指向旧快照）。
        """
        raw = self.store.get_meta(self.SELF_MODEL_META_KEY).get(self.SELF_MODEL_META_KEY)
        if raw:
            try:
                return json.loads(raw), self.SELF_MODEL_META_KEY
            except Exception:  # noqa: BLE001 —— 权威键损坏则继续走回退腿
                pass
        c = self.store.conn.cursor()
        c.execute("SELECT content FROM nodes WHERE layer='self' AND tags LIKE ? "
                  "ORDER BY rowid DESC LIMIT 1", ("%self_snapshot%",))
        row = c.fetchone()
        if not row or not row[0]:
            return None
        # 快照不可解析：向上抛（调用方按「恢复失败」处置并显式标注）
        return json.loads(row[0]), "self_snapshot_node"

    def _restore_self_model(self, identity: str) -> Dict:
        """启动时从持久化快照重建运行态自我模型（lingshu #33）。

        返回状态字典（`self._self_restore`，供 `self_check` 观测）：
          `{"status": "empty"|"restored"|"failed", "source": ..., "identity_conflict": bool}`。

        口径（逐条对齐设计稿 §5.3/§5.4）：
          · **恢复源**（S-33-2）：`engine_meta` 权威键优先，回退 SELF 层最新快照；
          · **恢复范围**（S-33-4 · 设计者 2026-10-09 授权「同取或同弃＋显式标注」）：
            整份快照的**数据字段一并取**（`trust_state` 与 `trust_history` **同取**，
            保证 `e_weight` 与历史自洽），不做「只取信任态、弃信任史」的拆分；
          · **identity 冲突**（S-33-5/D-29）：**快照优先**，与构造入参不一致时置
            `identity_conflict=True` 显式标注（不静默换人格、不静默丢弃快照身份）；
          · **失败处置**（S-33-1/D-21/C-2）：**回退默认且显式标注**（`status="failed"`），
            不静默、也不拒绝启动。
        削掉的判别力：坏快照只被**标注**并回退默认，本件不引入「拒绝启动」或
        「快照修复」面（设计稿 S-33-1(b) 另议）。
        """
        info = {"status": "empty", "source": None, "identity_conflict": False}
        try:
            loaded = self._load_self_snapshot()
        except Exception as e:  # noqa: BLE001 —— 坏快照：回退默认 + 显式标注
            info.update(status="failed", error=str(e))
            return info
        if not loaded:
            return info
        data, source = loaded
        try:
            model = SelfModel.from_dict(data)
        except Exception as e:  # noqa: BLE001
            info.update(status="failed", error=str(e))
            return info
        snap_identity = getattr(model, "identity", "")
        if snap_identity and identity and snap_identity != identity:
            info["identity_conflict"] = True   # 快照优先（D-29），冲突显式标注
        self.self_model = model
        info["status"] = "restored"
        info["source"] = source
        return info

    # ==================== 检索层（M1） ====================

    def search_content(self, query: str, layers: List[MemoryLayer] = None,
                       limit: int = 20,
                       min_score: float = 0.0) -> List[Tuple[STNode, float]]:
        """内容检索（引擎面）：委托 `LayeredStore.search_content`（LIKE 预筛 +
        二元组 Jaccard 排序），再走复用/模式加权记账。

        #795：`min_score` 只截断**返回面**（`score < min_score` 不返回），不截断
        候选池、不改打分与排序；缺省 0.0 ⇒ 既有调用逐位不变。分数语义是内部
        相似度×加权的混合值，推荐阈值须由调用方用自身语料标定（本仓不内置）。
        边界：与 `_self_cognition` 的模式加权同时启用时，先加权再过滤（阈值对
        最终返回分生效）。

        #804（契约声明 · 守卫 `tests/test_issue804_search_like_wildcard_guard.py`）：
        查询里的 `%` / `_` / `\\` 一律按**字面**匹配——store 层预筛经
        `_escape_like`（`%`→`\\%`、`_`→`\\_`、`\\`→`\\\\`）转义并挂
        `ESCAPE '\\'` 子句、模式经 `?` 参数绑定（#183 落地）。issue #804 报称
        「含 % 查询抛 `ValueError: unsupported format character`」——**基线
        40fceb6 复现不成立**：检索链（本方法 → store `search_content` →
        `_like_fragments`/`_escape_like`/FTS `candidates`）全程无一处对查询做
        Python `%` 格式化，含 `%`/`_` 查询不抛异常且只命中字面含该字符的内容
        （读数见 #804 守卫件 docstring）。本件据此只钉契约＋补引擎面文档，
        不改行为。边界（如实标注）：①`LINGSHU_FTS=1`（默认关）时候选生成走
        二元组倒排，纯标点邻接的单字 gram（如 `%s` 的 `s`）会在索引侧并入
        邻段 ⇒ 该形态查询 FTS 路径可能 0 命中（LIKE 路径不受影响）——属 FTS
        分词粒度边界（#35/#797 族），不属本件；②预筛落空时的全表二元组
        兜底会对**任何**未命中查询返回低/零分相似候选（#183 已裁定仅纯
        通配符查询跳过兜底，其余口径逐字不变），非 `_` 通配语义。
        """
        results = self.store.search_content(query, layers, limit)
        # v1.12 P0-5a：模式成员检索加权（使用模式，不修改蒸馏产物）
        if self._self_cognition is not None:
            results = self._self_cognition.apply_pattern_boost(results)
            self._self_cognition.note_search(query, results)
        # #795：返回面截断在加权之后（阈值对最终分数生效）；缺省 0.0 无操作。
        results = _filter_min_score(results, min_score)
        self._note_reuse([n.id for n, _ in results])
        self._note_action("search", query, [n.id for n, _ in results],
                          {"hits": len(results)})
        return results

    def semantic_search(self, query: str, limit: int = 10) -> List[Tuple[STNode, float]]:
        """v1.7 语义检索：优先结构化语义坐标相似度；无坐标节点回退文本相似度（D-005 降级）"""
        if not self._semantic_provider:
            return self.store.search_content(query, limit=limit)
        q_sc = self._semantic_provider.to_semantic_coordinates(query)
        nodes = self.store.query_nodes(layer=MemoryLayer.KNOWLEDGE, limit=300)
        scored = []
        for n in nodes:
            sc = getattr(n, "semantic_coordinates", {}) or {}
            if sc:
                sim = self._semantic_provider.similarity_coordinates(q_sc, sc)
            else:
                sim = self._semantic_provider.similarity(query, n.content)
            if sim > 0.05:
                scored.append((n, sim))
        scored.sort(key=lambda x: -x[1])
        self._note_reuse([n.id for n, _ in scored[:limit]])
        return scored[:limit]

    def set_embedding_provider(self, provider):
        """依赖注入（D-005）：provider.encode(text)->List[float]；provider.search(query,limit)->List[str]"""
        self._embedding_provider = provider

    def _with_opposites(self, scored, limit: int):
        """矛盾两侧同时呈现（LINGSHU_CONTRADICTION=1）：top-limit 中任一节点若有 OPPOSITE 边，
        把对侧插到它紧后面（分数 = 宿主分数 - 1e-6）；对侧已在榜内则不动，超出 limit
        的从队尾挤出。只读边，不改其余节点的分数与相对次序。"""
        top = list(scored[:limit])
        ids = {n.id for n, _ in top}
        out, paired = [], set()
        for node, sc in top:
            out.append((node, sc))
            for e in (self.store.get_outgoing_edges(node.id) +
                      self.store.get_incoming_edges(node.id)):
                if e.relation_type != EdgeType.OPPOSITE:
                    continue
                oid = e.target_id if e.source_id == node.id else e.source_id
                if oid in ids:
                    continue
                other = self.store.get_node(oid)
                if other is None:
                    continue
                ids.add(oid)
                paired.update((node.id, oid))
                out.append((other, sc - 1e-6))
        # 回到 limit 条：先从队尾挤掉不成对的节点；矛盾对不拆开（极端情况下可略超 limit）
        while len(out) > limit:
            for k in range(len(out) - 1, -1, -1):
                if out[k][0].id not in paired:
                    del out[k]
                    break
            else:
                break
        return out

    def recall(self, context_content: str, limit: int = 10,
               min_score: float = 0.0) -> List[Tuple[STNode, float]]:
        """组合联想（内容相似 0.5 + 重要性 0.3 + 近因 0.2）——记忆参与推理（1.1.1）

        #29④：终排与 search_content 同口径，同分再按置信度降序（confidence 与
        importance 一起进排序键；未验证/被矛盾标记的低置信节点不再压过高置信节点）。

        #787（主键漂移）：近因分量读 `last_access`，而检索路径（search_content）
        返回前会把命中节点的 `last_access` 刷成「现在」⇒ 同一节点的分数在两次
        recall 之间漂移（issue 实测 ~1.9e-08，随两次调用间隔增长），主排序键
        不稳定 ⇒ sim 并列组的次序翻转。#613 的末位确定性键 `id` 在 recall 这里
        **从未生效**——它只救「精确同分」，救不了「分数自身在漂移」。修法（根修）：
        recall 的检索腿显式 `touch=False`（见 `search_content` 的 `touch` 形参）
        ——打分路径不写回打分输入，`last_access` 只由写入、M5 重复观测、置信度
        复核、巩固复述等真实强化事件刷新（与《检索不刷新近因》设计文档「检索 ≠
        使用 ≠ 强化」同则；`LINGSHU_RETRIEVAL_NO_TOUCH` 开关仍只管直接检索路径）。
        access_count 仍在 recall 命中时计数（统计语义不变）。
        分数随 wall-clock 仍会**整体**微降（recency 对固定 la 单调衰减），但组内
        两两差值的符号不变 ⇒ 次序跨调用逐位稳定。

        #795：`min_score` 只截断**返回面**（`score < min_score` 不返回），不截断
        候选池、不改打分与排序；缺省 0.0 ⇒ 既有调用逐位不变。分数是
        「相似度 0.5 + 重要性 0.3 + 近因 0.2」的混合值，推荐阈值须由调用方按
        自身语料标定（本仓不内置）。边界：矛盾配对（`LINGSHU_CONTRADICTION=1`）
        与阈值同时启用时，低于阈值的对侧仍会被截掉（配对可能被拆开）——阈值
        优先生效，调用方需自行评估该组合。"""
        # #787：touch=False——recall 的打分输入（last_access）不得被自己的检索腿
        # 写回（根修「分数漂移」）；候选池/排序其余口径逐位不变。
        results = self.store.search_content(context_content, limit=50, touch=False)
        if not results:
            return []
        now = time.time()
        scored = []
        for node, sim in results:
            la = node.last_access if node.last_access is not None else (node.created_at or now)
            recency = 1.0 / (1.0 + max(0.0, now - la) / 86400.0)
            score = 0.5 * sim + 0.3 * (node.importance or 0.0) + 0.2 * recency
            scored.append((node, score))
        # #613：末位补 `id`——#613 当年只落在 search_content 的三处排序，recall
        # 自己的终排从未补上；本处补齐同款末位键（库内唯一、跨平台稳定），在
        # `last_access` 精确并列（如 Windows 时钟粒度下的同桶创建）⇒ 分数精确
        # 相等时给出唯一确定序。非并列情形逐位不变。
        scored.sort(key=lambda x: (-x[1], -x[0].confidence, x[0].id))
        if _contradiction.enabled():
            scored = self._with_opposites(scored, limit)
            limit = max(limit, len(scored))  # 矛盾对不拆开（见 _with_opposites）
        # v1.12 P0-5a：模式成员召回加权
        if self._self_cognition is not None:
            scored = self._self_cognition.apply_pattern_boost(scored[:limit])
        # #795：返回面截断在加权之后（阈值对最终分数生效）；缺省 0.0 无操作。
        scored = _filter_min_score(scored, min_score)
        self._note_reuse([n.id for n, _ in scored[:limit]])
        self._note_action("recall", context_content,
                          [n.id for n, _ in scored[:limit]],
                          {"hits": len(scored[:limit])})
        return scored[:limit]

    # ==================== 盲区注册表（M2 · D-001 语义判定） ====================

    META_BLINDSPOT_DEFINITION = "对人类造成文明级别的重大负面影响"

    #: 自我快照有界保留上界（lingshu #201）。`update_self` 每次调用新建一条
    #: `self_snapshot`（importance=0.9 / confidence=1.0）——修前无清理路径 ⇒
    #: 无界堆积并以普通记忆身份挤占召回。上界口径与 `SelfModel.HISTORY_MAX`
    #: （同类的状态变更历史钳制）同族；经验标定（#201 修复轮；理论章节追不到）。
    SELF_SNAPSHOT_MAX = 200

    #: `self_check` 的环枚举上限（lingshu #252）。`cycle_details` 只取 `cycles[:5]`，
    #: 故枚举到 5 条即可满足自检；旧实现无上限，稠密 SCC 使枚举耗时/内存随环数
    #: 指数增长（issue 实测 11 节点 13.69s / 335 万条 / 3 GB）。经验标定（#252 修复
    #: 轮；仓内无规定该预算的理论章节，追不到更早出处）。
    SELF_CHECK_CYCLE_ENUM_LIMIT = 5

    def register_blindspot(self, code: str, description: str, severity: str = "medium",
                           category: str = "operational",
                           predictability: str = "pending_assessment") -> str:
        """注册操作层盲区。元盲区判定为语义标准（对人类造成文明级别的重大负面影响）——
        判定权在维生系统，API 强制 meta 零记录（不写入任何注册表/日志/快照）。
        废弃数值判定（非 code≥91 判据，维生系统终裁修正）。
        predictability（D-003）：predictable / unknowable / pending_assessment

        #386：判据由 `category == "meta"` 字面比较改为 `_is_meta_category`
        （`strip().lower()` 规范化）——旧写法下 `'Meta'`/`'META'`/`' meta'`/`'meta\\n'`
        都能把元盲区写进注册表；store 层同加该闸（见 `LayeredStore.add_blindspot`）。
        """
        if _is_meta_category(category):
            raise ValueError(
                f"元盲区（{self.META_BLINDSPOT_DEFINITION}）零记录，拒绝写入")
        return self.store.add_blindspot(code, description, severity, "operational",
                                        predictability)

    def list_blindspots(self, status: str = None) -> List[Dict]:
        return self.store.list_blindspots(status)

    def get_open_blindspots(self) -> List[Dict]:
        """学习方向驱动：开放盲区列表"""
        return self.store.list_blindspots(status="open")

    def resolve_blindspot(self, blindspot_id: str, resolved: bool = True,
                          note: str = "", designer_key: str = None):
        """盲区闭环（D-007 需设计者密钥：D-001 语义判定权在维生系统）。

        #222：密钥闸下沉到 store 层（`LayeredStore.resolve_blindspot` 亦验）——门面
        保留本处校验以**先判后写**（`resolved=False` 时不该碰 store），并透传
        `designer_key`。详见 `LayeredStore.resolve_blindspot` docstring。
        """
        if not verify_designer(designer_key):
            raise PermissionError(
                "D-007 设计者认证失败：密钥无效或未配置 AEIS_DESIGNER_KEY（fail-closed）")
        if not resolved:
            return False
        self.store.resolve_blindspot(blindspot_id, designer_key=designer_key)
        return True

    # ==================== 情境层（M4） ====================

    def add_context(self, content: str, importance: float = 0.4,
                    condition_space: ConditionSpace = None, tags: List[str] = None) -> STNode:
        """写入情境层节点，按同一写事务内的持久化容量执行 FIFO。

        容量是全库设置；已有实例的缓存不能覆盖另一实例已提交的新值。
        插入和淘汰共同成功或撤回，调用方已有事务时只撤回本次工作。
        """
        cs = condition_space or ConditionSpace(
            "情境感知", "会话摄入", (time.time(), time.time()+3600), "协议实例运行中")
        node = STNode(
            id=f"ctx_{uuid.uuid4().hex[:8]}_{int(time.time()*1000)}",
            content=content, modality="text", spatial_coordinates={},
            temporal_coordinate=time.time(), condition_space=cs,
            importance=importance, confidence=0.5,
            layer=MemoryLayer.CONTEXT, tags=(tags or []) + ["context"])
        with self.store._tx():
            self.store.add_node(node)
            self.store.enforce_context_cap(self._load_context_max())
        return node

    def get_recent_context(self, limit: int = 20) -> List[STNode]:
        return self.store.get_recent_context(limit)

    def _load_context_max(self) -> int:
        """读情境层容量上限（`engine_meta` key `context_max`，跨进程持久）。

        lingshu #246：旧实现 `self._context_max = 200` 是**实例属性、无持久化**，
        而 `enforce_context_cap` 的 FIFO 淘汰**作用于持久库**——进程重启后上限
        回 200，但淘汰已按旧上限删过盘面数据 ⇒ 上限与盘面不一致（重启后第一条
        `add_context` 会静默批量删情境记忆）。修法（最小 · 复用既有机制）：
        上限经既有 `engine_meta`（v1.14 观测持久化，跨进程 key-value）存取。
        判据来源：分诊表 `triage_lingshu_rest.md` 行 246（`_context_max` 无持久化）
        ＋ 复用既有 `LayeredStore.get_meta`/`set_meta` 先例（`core.py` M13/引擎元数据段）。
        边界（削掉的判别力）：仍**不**校验盘面既有情境层是否已超限（加载即用，
        不触发淘汰）——与 `set_context_cap` 的显式淘汰面区分。
        """
        raw = self.store.get_meta("context_max").get("context_max")
        return _to_int_or(raw, self.DEFAULT_CONTEXT_MAX)

    def set_context_cap(self, max_size: int):
        """设置情境层容量上限（FIFO 上限，至少 1）——#246：跨进程持久化。

        旧实现只改实例属性（重启即回默认）；现同时落 `engine_meta`，使
        `_load_context_max` 在重启后读到同一上限（上限与盘面一致）。
        `int(float(...))` 归一：旧实现无 `int()`，浮点 `200.0` 会落到
        `enforce_context_cap` 的列表切片抛 `TypeError`（分诊表 246 实证）。
        判据来源：分诊表 `triage_lingshu_rest.md` 行 246。
        #655：盘面坏值（非有限/超大）经 `_to_int_or` 回落 `DEFAULT_CONTEXT_MAX`
        （fail-closed）——旧写法在此抛 `OverflowError`，且该值会被 `set_meta`
        写进盘面，使**下次构造**也失败。判据来源：公开 issue #655。

        #768：**改内存值之前**过治理键角色闸——非 PRIMARY 实例在**跨进程持久**
        的库上 ⇒ `PermissionError`，内存与盘面都不动（内存库按库归属分档放行，
        见 `_assert_meta_key_writable`）。为什么闸位必须在赋值之前：旧形态若
        先改 `self._context_max` 再在落库处抛，内存已改而盘面未写 ⇒ 静默分叉
        （与 `#191` 的 `update_self` 同款病：调用方看到异常，运行态却已变）。

        容量保存和 FIFO 裁剪共用一条事务，提交成功后才发布运行值。否则保存失败
        仍会使下一次正常写入按新容量误删记忆，或裁剪失败留下部分删除。与配置终裁
        一样要求独立事务：调用方后续的 rollback 无法撤回 Python 属性，故已有事务
        时在任何改动前显式拒绝。
        """
        with self.store._lock:
            self.store._assert_meta_key_writable("context_max")
            if self.store.conn.in_transaction:
                raise RuntimeError("容量设置需要独立事务；请先结束调用方事务")
            capacity = _to_int_or(max_size, self.DEFAULT_CONTEXT_MAX)
            with self.store._structure_event_transaction():
                self.store.set_meta("context_max", str(capacity))
                self.store.enforce_context_cap(capacity)
            self._context_max = capacity

    # ==================== 去重配置（M5 · D-004 动态阈值） ====================

    def set_dedup_config(self, static_threshold: float = 0.85, window: int = 30):
        self._dedup_static = min(0.95, max(0.5, static_threshold))
        self._dedup_window = max(5, window)

    def _effective_dedup_threshold(self) -> float:
        """2.7.2 动态死区：threshold = clamp(max(static, 3·σ_history), 0.85, 0.95)"""
        if len(self._dedup_history) < 5:
            return self._dedup_static
        mean = sum(self._dedup_history) / len(self._dedup_history)
        var = sum((x - mean) ** 2 for x in self._dedup_history) / len(self._dedup_history)
        sigma = math.sqrt(var)
        return min(0.95, max(self._dedup_static, 3.0 * sigma))

    def register_conflict(self, a_id: str, b_id: str) -> Optional[STEdge]:
        return self.store.register_conflict(a_id, b_id)

    # ==================== 固化流水线（M6 · D-003） ====================

    def propose_promotion(self, node_id: str, requester: str, reason: str,
                          override_rejected: bool = False,
                          designer_key: str = None) -> str:
        """固化入口读负记忆（#216，fail-closed）。

        旧实现：`propose_promotion` **从不查** `rejected_paths` ⇒ 设计者已明确否决
        的内容可以原文重提并写进**不可逆的结构层**（`adjudicate_promotion(True)` 后
        `layer='structure'`，不衰减、不可删）——负记忆表「只写不读」的黑洞在固化面
        上的一次性、不可撤回事故。
        修法（最小 · 复用既有原语）：建提案前用既有 `find_rejected_paths`（字符二元组
        Jaccard 文本匹配，阈值 0.15）按**被提案内容**查 `path_type='promotion'` 的 open
        负记忆；命中即**拒绝提案**（抛 `ValueError`，零写入）。确需重提时须显式
        `override_rejected=True` **且**过 `verify_designer(designer_key)` 密钥闸
        （与 `adjudicate_promotion` 同款 fail-closed）——即 issue 建议的「强制附上
        override 并由设计者二次签名」。
        判据来源：公开 issue #216「建议修复 1」第一项（`propose_promotion`：命中同
        `path_type='promotion'` 的 open 负记忆时拒绝提案，或强制附上
        `override_rejected_id` 并由设计者二次签名）＋ 分诊表 `triage_lingshu_ALL.md`
        行 216；仓内理论锚点见 `lingshu/core/activation.py:15`「第五篇：检索不全塞／
        **负记忆同价**／条件绑定」（本仓理论章节 `docs/theory/` 未规定该读取面口径，
        经验标定）。
        边界（削掉的判别力，如实标注）：本件只把负记忆接到**固化**这一个「重做」入口
        ——issue 另列的 `record_action_sequence`（失败序列仍可被技能化）、`recall`／
        `ActivationEngine`（负记忆仍不进检索）**未接**（属独立面，须另裁）；也**不**
        引入 `subject_key`/`count` 结构化列（须升 `SCHEMA_VERSION`，属新机制）。
        匹配仍是**文本相似**（非精确 hash）——相似但非同一内容可能误拒，故保留显式
        override 出口。
        """
        node = self.store.get_node(node_id)
        if node is None:
            raise ValueError("节点不存在")
        if node.layer in (MemoryLayer.ANCHOR, MemoryLayer.STRUCTURE):
            raise ValueError("共享层节点不可再提升")
        # #216：负记忆读取面（fail-closed）
        rejected = [p for p in self.find_rejected_paths(node.content)
                    if p.get("path_type") == "promotion"]
        if rejected:
            if not override_rejected:
                raise ValueError(
                    "#216 该内容此前已被维生系统终裁否决（负记忆 %s：%s）——"
                    "重提须显式 override_rejected=True 且提供设计者密钥"
                    % (rejected[0].get("id"), rejected[0].get("reason", "")))
            if not verify_designer(designer_key):
                raise PermissionError(
                    "#216 override 否决负记忆须过设计者密钥闸（fail-closed）")
        pid = self.store.add_promotion_proposal(node_id, requester, reason,
                                                   override_rejected=override_rejected,
                                                   designer_key=designer_key)
        self.store.tag_node(node_id, "promotion_pending")
        return pid

    def verify_promotion(self, proposal_id: str, verified_by: str) -> bool:
        """验证单元复核 + 3.3.1 条件空间复核（调用方记录复核结论）

        #118：透传 store 层的状态门/身份门结果（False = 被拒、状态未改写）。"""
        return self.store.verify_promotion(proposal_id, verified_by)

    @_transactional
    def adjudicate_promotion(self, proposal_id: str, adjudicated_by: str, approved: bool,
                             designer_key: str = None) -> bool:
        """维生系统终裁（D-007 需设计者密钥）：仅经验证单元复核的提案可终裁；通过后才写结构层（不可逆）"""
        node_id = self.store.adjudicate_promotion(proposal_id, adjudicated_by, approved,
                                                  designer_key=designer_key)
        if node_id:
            node = self.store.get_node(node_id)
            if node and "promotion_pending" in node.tags:
                node.tags.remove("promotion_pending")
                c = self.store.conn.cursor()
                c.execute("UPDATE nodes SET tags=? WHERE id=?",
                          (_dumps_tags(node.tags), node_id))
                self._commit_locked()
            if not approved:
                # A-1：被拒绝的固化提案 → 被拒绝路径资产
                # #691 P0-a（负记忆基座截断）：旧实现此处把正文截成
                # `node.content[:60]` 落库，而匹配侧（`find_rejected_paths` /
                # `add_promotion_proposal`）用的是**新内容全文**——
                # `char_bigram_jaccard(全文, 截断60字+理由)` 的并集被全文撑大，
                # 长且用词多样的正文（约 500 字以上）连**一字不改原文重提**都
                # 落到阈值 0.15 之下而不被拦（改尾既非必要也非充分）。
                # 修法：落库基座改用**完整正文**，与匹配侧的全文口径统一
                # （不改阈值 0.15、不改匹配算法）。短文本判别力不受影响
                # （全文对全文 + 理由，相似度仍远高于 0.15）。
                content = node.content if node else proposal_id
                self.register_rejected_path(
                    path_type="promotion",
                    description=f"固化提案被拒：{content}",
                    reason=f"维生系统终裁拒绝（{adjudicated_by}）")
        return node_id is not None

    # ==================== 遗忘门控（M7） ====================

    def protect_node(self, node_id: str, reason: str):
        """3.2 节不可遗忘类别：内容级保护（衰减跳过）"""
        self.store.protect_node(node_id, reason)

    def get_protected_nodes(self) -> List[str]:
        return self.store.get_protected_nodes()

    # ==================== 技能记忆（M9） ====================

    def store_skill(self, name: str, description: str, procedure: str, confidence: float = 0.5) -> str:
        return self.store.add_skill(name, description, procedure, confidence)

    def recall_skill(self, query: str, limit: int = 10) -> List[Dict]:
        skills = self.store.search_skills(query, limit)
        # v1.12 P0-5a：技能置信度加权（使用模式）
        if self._self_cognition is not None:
            skills = self._self_cognition.skill_feedback(skills)
        return skills

    def update_skill_confidence(self, skill_id: str, delta: float):
        self.store.update_skill_confidence(skill_id, delta)

    # ==================== 外部锚点（M10 · 盲区29 缓解） ====================

    #: 锚点层写入的**唯一白名单**（fail-closed）。
    #:
    #: 锚点层是**不可变声明层**，属**极度重要的信息**——其写入面一律 fail-closed：
    #: 只有本白名单内的 kind 可经 `register_external_anchor` 进入不可遗忘共享层，
    #: 白名单外一律 `ValueError`（**不**做静默降级、**不**自动放行新 kind）。
    #: 本白名单为**封闭集**，新增 kind 属设计取舍、须经裁定，不得由实施者顺手放宽。
    #:
    #: 裁定出处：维护者 2026-10-10 裁定（lingshu issue #329）——「关于锚点层确实是
    #: 极度重要的信息，需要收紧」，即维持三 kind 白名单、**不放行第四种**。
    #: 守卫 `tests/test_issue329_anchor_kinds_guard.py` 钉死取值并防松动。
    ANCHOR_KINDS = ("pre_access_stance", "introspection", "external_calibration")

    def _record_structure_event(self, *args, **kwargs) -> STNode:
        """受信宿主共用存储事务；公开入口先执行其授权检查。"""
        return self.store._record_structure_event(*args, **kwargs)

    def record_structure_event(self, kind: str, content: str,
                               condition_space: ConditionSpace = None,
                               importance: float = 0.9, designer_key: str = None,
                               *, source: str = "", metadata: Dict = None) -> STNode:
        """授权外部输入的公共留史入口；记录事件，不批准规则或执行动作。

        PRIMARY 追加 STRUCTURE；其他角色追加 KNOWLEDGE + pending_sync。
        独立事件不走 M5 正文去重，缺少辅助来源说明不阻断记录。
        source/metadata 是来源描述，不是身份认证或权限依据。
        """
        if not verify_designer(designer_key):
            raise PermissionError(
                "D-007 设计者认证失败：密钥无效或未配置 AEIS_DESIGNER_KEY（fail-closed）")
        return self._record_structure_event(
            kind, content, condition_space, importance, source, metadata, confidence=0.8)

    VERSION_EVENT_STATUSES = ("recorded", "planned", "applied", "failed",
                              "rolled_back", "uncertain")

    def record_version_event(self, version: str, content: str = "",
                             condition_space: ConditionSpace = None,
                             designer_key: str = None, *, from_version: str = None,
                             status: str = "recorded", source: str = "",
                             metadata: Dict = None) -> STNode:
        """追加版本迭代历史；默认只是 recorded，不执行升级或证明部署完成。"""
        if not isinstance(version, str) or not version.strip():
            raise ValueError("版本不能为空")
        if status not in self.VERSION_EVENT_STATUSES:
            raise ValueError(f"未知版本记录状态: {status}")
        cs = condition_space or ConditionSpace(
            "版本迭代", "发布流程", (0, float('inf')), "版本历史，不是执行授权")
        return self.record_structure_event(
            "version_iteration", content or f"版本 {version}（{status}）",
            condition_space=cs, designer_key=designer_key, source=source,
            metadata={"version": version, "from_version": from_version,
                      "status": status, "details": metadata or {}})

    def register_external_anchor(self, kind: str, content: str,
                                 condition_space: ConditionSpace = None,
                                 designer_key: str = None, *,
                                 source: str = "", metadata: Dict = None) -> STNode:
        """接入前立场 / 自省记录 / 外部校准输入（D-007 授权闸保持不变）。

        PRIMARY 写结构层；其他角色写知识层副本并明确标记 pending_sync。
        来源说明随节点保存，不代表外部断言已验证；同步仍由父节点负责。
        """
        if not verify_designer(designer_key):
            raise PermissionError(
                "D-007 设计者认证失败：密钥无效或未配置 AEIS_DESIGNER_KEY（fail-closed）")
        if kind not in self.ANCHOR_KINDS:
            raise ValueError(f"未知锚点类型: {kind}")
        cs = condition_space or ConditionSpace(
            f"外部锚点:{kind}", "外部校准输入", (0, float('inf')), "方向性自检参照")
        # 写入机制（原子留史 + structure_event 元数据）与其余三处共用 `_record_structure_event`，
        # 不再内联构造节点；SUB 腿的 pending_sync 标签仍取单一真源常量 `SUB_PENDING_SYNC_TAG`。
        # 锚点腿保留自己的角色分派：tests/test_issue329_anchor_kinds_guard.py 的 G4b
        # 把「core.py 内的角色分派恰为 2 处」钉死（单一收口辅助 +
        # 外部锚点自身分支），该守卫在 main 上先于本 PR 落地，不得为凑绿而改。
        if self.role == Role.PRIMARY:
            return self._record_structure_event(
                kind, content, condition_space=cs, source=source, metadata=metadata,
                extra_tags=["external_anchor"], sub_modality="anchor",
                record_class="external_anchor", confidence=1.0)
        return self._record_structure_event(
            kind, content, condition_space=cs, source=source, metadata=metadata,
            extra_tags=["external_anchor"], sub_modality="anchor",
            record_class="external_anchor", confidence=0.8)

    # ==================== 信任状态（M3 · D-002） ====================

    def update_trust_state(self, t_total: float, round_no: int,
                           p_trust: float = None, p_gap: float = None) -> STNode:
        """记录信任状态 + 离散 e_weight。

        #206：旧 docstring 的「placeholder 边界：仅记录，不参与决策」与事实不符
        ——`t_total` 经 `LongTermMemoryGate` 以 0.25 权重进入长期层评分并触发
        `protect_node`（不可遗忘），详见 `SelfModel.update_trust_state` 的
        #206 说明与守卫 `tests/test_issue206_trust_state_guard.py`。
        本入口的非有限值校验经 `self.self_model.update_trust_state` 的
        `_require_finite` 继承（**先判后写**：坏值在任何内存/落库改动之前抛出）。
        """
        self.self_model.update_trust_state(t_total, round_no, p_trust, p_gap)
        snapshot = STNode(
            id=f"self_{uuid.uuid4().hex[:8]}_{int(time.time()*1000)}",
            content=json.dumps({"type": "trust_state", **self.self_model.trust_state}, ensure_ascii=False),
            modality="trust_state", spatial_coordinates={"self_axis": 0.8},
            temporal_coordinate=time.time(),
            condition_space=ConditionSpace(
                "自我认知", "信任状态观测", (time.time(), time.time()+3600), "协议实例运行中"),
            importance=0.85, confidence=1.0, layer=MemoryLayer.SELF, tags=["trust_snapshot"])
        self.store.add_node(snapshot)
        return snapshot

    # ==================== v1.11 知识飞轮（FLYWHEEL-REV1） ====================

    @_transactional
    def _note_reuse(self, node_ids: List[str]):
        """复用追踪（P0-2 度量：同轮同节点去重）。

        T4 修复（2026-08-30）：内存 tracker 之外同步落库 flywheel_reuse 表
        （session_id 区分进程实例）——跨进程累计，心跳/对话端任一进程的
        flywheel_metrics 都能读到全实例复用观测（此前心跳进程 tracker 恒空）。
        写库失败不阻断检索主流程（复用观测是工程观测值，DEVIATION-004）。

        #208（同轮同节点重复落库 + tracker 无上界）：旧实现把**整份累计 bucket**
        重新 `INSERT`，而 bucket 在本轮内只增不减 ⇒ 同一轮每检索一次，之前已记录
        的节点就再插一遍（issue 实测同轮同节点行数 [3,2,4]、总 9 行，去重后应为 3）；
        下游按行数统计的复用率被放大。另 `_reuse_tracker` 按轮次编号的 dict **从不
        清理**，常驻进程持续增长。修法（issue 建议 1「只插入 new」路线）：
          · `new = set(node_ids) - bucket` → `bucket |= new` → **只插入 `new`**；
          · tracker 只保留最近 `_REUSE_TRACKER_ROUNDS` 轮（更旧的轮次整键删除）。
        取「只插入 new」而非「表加 UNIQUE + INSERT OR IGNORE」：`_init_tables` 对
        既有库是 `CREATE TABLE IF NOT EXISTS`，**加不了**新 UNIQUE 约束 ⇒ 需迁移；
        纯代码改动对既有库直接生效。
        判据来源：公开 issue #208（期望「同轮同节点在库中只有一行；内存 tracker 有
        上界」＋ 建议修复代码）＋ 本函数 docstring「同轮同节点去重」。
        边界（削掉的判别力）：`_REUSE_TRACKER_ROUNDS` 取 **经验标定，追不到理论
        出处**——仓内无规定该上界的理论章节，取值参照「同轮去重只需覆盖当前轮及
        极少近邻轮」的用法；越界的历史轮次不再参与去重（其行已落库，重复计数面
        只在本轮内，故无功能损失）。
        """
        if not node_ids:
            return
        rnd = self._interaction_count
        bucket = self._reuse_tracker.setdefault(rnd, set())
        # #208：只落「本轮新增」的节点——旧实现插入整个累计 bucket，同轮同节点被
        # 每次检索重复插入一遍（issue 实测 [3,2,4] / 总 9）。
        new = set(node_ids) - bucket
        if not new:
            return
        bucket |= new
        # #208：tracker 只保留最近 K 轮（更旧键整键删除），防常驻进程无界增长。
        cutoff = rnd - self._REUSE_TRACKER_ROUNDS + 1
        for old in [r for r in self._reuse_tracker if r < cutoff]:
            del self._reuse_tracker[old]
        try:
            conn = self.store.conn
            conn.execute(
                "CREATE TABLE IF NOT EXISTS flywheel_reuse "
                "(session_id TEXT, round INTEGER, node_id TEXT, ts REAL)")
            ts = time.time()
            conn.executemany(
                "INSERT INTO flywheel_reuse (session_id, round, node_id, ts) "
                "VALUES (?, ?, ?, ?)",
                [(self._session_id, rnd, nid, ts) for nid in sorted(new)])
            # #722：本表是**跨会话累计**的复用观测（行数 = 会话数 × 每会话命中节点
            # 数），#208 只把去重钉在内存 tracker（最近 K 轮）与「只插 new」两处，
            # DB 累计层既无读取窗口也无回收面 ⇒ 常驻部署只增不减（#122「无界增长
            # 族」的 DB 表落点）。现按同文件 `record_info_gap` 对 `gap_history` 的
            # **同款钳制手法**裁剪：只保留最近 `FLYWHEEL_REUSE_MAX_ROWS` 行。
            # 保留序取 `rowid`（本表无主键/自增列，`rowid` 即**插入序**）——与
            # `prune_self_snapshots`(#489) / `enforce_context_cap`(#259) 同键同法，
            # 不用墙钟 `ts`（`time.time()` 不单调：NTP 步进 / 快照回滚下按墙钟
            # 淘汰会先删刚写入的行）。
            # 判据来源：公开 issue #722（「`_note_reuse` 插入后追加同 `gap_history`
            # 款裁剪」）＋ 同文件 `gap_history` 的既有 2000 行钳制先例。
            # 边界（削掉的判别力，如实标注）：①上界 `FLYWHEEL_REUSE_MAX_ROWS` 取
            # **经验标定，追不到理论出处**（与 `_REUSE_TRACKER_ROUNDS` 同族）；
            # ②裁剪后该表只反映「最近的行」，不构成全历史复用观测——按行数统计
            # 复用率的下游读到的是**滚动窗口**（与 `gap_history`「滚动保留 2000 条」
            # 同口径）；③只裁行、不动 `_reuse_tracker` 的内存语义。
            conn.execute(
                "DELETE FROM flywheel_reuse WHERE rowid NOT IN"
                " (SELECT rowid FROM flywheel_reuse ORDER BY rowid DESC LIMIT ?)",
                (self.FLYWHEEL_REUSE_MAX_ROWS,))
            conn.commit()
        except Exception:
            pass  # 观测落库失败静默（不阻断检索）

    def notify_event(self, event_type: str, payload: Dict = None):
        """P1-4 事件驱动：轻量标记（防抖：同类事件 τ=N_effective/10 秒内合并）"""
        now = time.time()
        tau = 30 / 10  # N_effective=30 → τ=3 秒
        for ev in self._event_queue:
            if ev["type"] == event_type and now - ev["ts"] < tau:
                ev["count"] += 1
                return
        self._event_queue.append({"type": event_type, "payload": payload or {},
                                  "ts": now, "count": 1})

    def consume_events(self) -> Dict:
        """P1-4：批量消费事件标记（定时轮询优先；P0 危机期间暂停）"""
        events = list(self._event_queue)
        self._event_queue = []
        consumed = 0
        if self._lifecycle and self._lifecycle.state == "paused":
            return {"status": "paused", "consumed": 0,
                    "note": "P0 危机期间暂停非 P0 处理（3.4 节）"}
        for ev in events:
            if ev["type"] in ("perception", "write", "condition_switch") and self._flywheel:
                try:
                    self._flywheel.distill_cycle()
                    consumed += 1
                except Exception:
                    pass
        return {"status": "ok", "consumed": consumed, "pending": len(self._event_queue)}

    # ---- 飞轮接口 ----

    def evo_distill_cycle(self, source_filter: str = None) -> Dict:
        """P0-1 蒸馏管线：经验 → 可复用模式（局部概括 · 盲区5）"""
        if not self._flywheel:
            return {"status": "v111_not_ready", "error": self._flywheel_error}
        result = self._flywheel.distill_cycle(source_filter)
        self._note_action("distill", f"patterns={result.get('patterns')}",
                          None, {"input": result.get("input")})
        return result

    def evo_flywheel_metrics(self) -> Dict:
        """P0-2 飞轮度量（工程观测值，不参与信任计算）"""
        if not self._flywheel:
            return {"status": "v111_not_ready", "error": self._flywheel_error}
        return self._flywheel.flywheel_metrics()

    def test_transfer_capability(self) -> Dict:
        """P0-3 迁移测试（已对齐条件空间 · 显著性 · 可操作失败条件）"""
        if not self._flywheel:
            return {"status": "v111_not_ready", "error": self._flywheel_error}
        return self._flywheel.transfer_test()

    def universe_calibrate(self) -> Dict:
        """宇宙校准参照（元理论方向性检查 · 非盲区33关闭依据）"""
        if not self._flywheel:
            return {"status": "v111_not_ready", "error": self._flywheel_error}
        return self._flywheel.universe_calibrate()

    def shortest_path(self, start_id: str, end_id: str, max_depth: int = 6) -> List[str]:
        """P1-2 多边类型最短路径"""
        if not self._flywheel:
            return []
        return self._flywheel.shortest_path(start_id, end_id, max_depth)

    def query_subgraph(self, query: str, max_nodes: int = 15) -> Dict:
        """P1-2 作用域子图"""
        if not self._flywheel:
            return {"query": query, "nodes": {}, "edges": []}
        return self._flywheel.query_subgraph(query, max_nodes)

    def mark_contested(self, node_id: str, reason: str) -> bool:
        if not self._flywheel:
            return False
        return self._flywheel.mark_contested(node_id, reason)

    def resolve_contested(self, node_id: str, verdict: str) -> bool:
        if not self._flywheel:
            return False
        return self._flywheel.resolve_contested(node_id, verdict)

    def mark_stale(self, node_id: str, reason: str) -> bool:
        if not self._flywheel:
            return False
        return self._flywheel.mark_stale(node_id, reason)

    def reverify(self, node_id: str) -> bool:
        if not self._flywheel:
            return False
        return self._flywheel.reverify(node_id)

    def recency_weighted_update(self, node_id: str, delta: float) -> bool:
        """P1-3 近因加权（6 类不可遗忘隔离 · τ=N_effective/3）"""
        if not self._flywheel:
            return False
        return self._flywheel.recency_weighted_update(node_id, delta)

    # ==================== v1.13 视觉感知与身体（VISION-REV1） ====================

    def set_vision_provider(self, provider):
        """v1.13 视觉提供者依赖注入（duck-typed：available()/detect()）"""
        self._vision_provider = provider

    def get_vision_provider(self):
        """v1.13 视觉提供者查询"""
        return self._vision_provider

    def perceive_image(self, image_path: str, conf_threshold: float = 0.35,
                       importance: float = 0.6, classes: list = None) -> dict:
        """v1.13 视觉感知闭环：检测 → 摘要 → 知识层记忆（modality=image）。
        classes：开放词汇检测词（中/英，YOLO-World 支持，默认文生图词表）。
        惰性：首次调用才加载 YOLO/CLIP 模型。"""
        try:
            provider = self._ensure_vision()
            from vision import perceive_image as _pi
            return _pi(self, image_path, provider,
                       conf_threshold, importance, classes)
        except Exception as e:
            # 视觉实现已迁 AEIS 身体库：缺失按能力不可用降级（契约 status）
            return {"status": "vision_unavailable", "error": str(e)}

    def get_body_capabilities(self) -> dict:
        """v1.13 身体能力声明（第 4 项：工具/模态作为自我的一部分）"""
        # 惰性视觉：未装配时显示 "lazy"（首次调用加载），不阻塞能力查询
        if self._vision_provider is None:
            vision_ok = False
            provider_name = "lazy"
            vision_note = self._vision_error or "延迟装配（首次视觉调用时加载 YOLO/CLIP）"
        else:
            vision_ok = self._vision_provider.available()
            provider_name = self._vision_provider.name
            vision_note = self._vision_error or None
        devices = []
        devices_available = []
        try:
            if self._body_registry is not None:
                health = self._body_registry.health()
                devices = [h["name"] for h in health]
                devices_available = [h["name"] for h in health if h.get("available")]
        except Exception:
            pass
        return {
            "modalities": {
                "text": True,
                "image": vision_ok,
                "audio": False,
                "video": False,
            },
            "vision": {"provider": provider_name,
                       "available": vision_ok,
                       "error": vision_note},
            "memory": {"engine": "v1.13", "persistent": True},
            "devices": devices,           # BODY-REV1：外部设备清单
            "devices_available": devices_available,
            "tools": ["search", "recall", "distill", "calibrate", "cognition",
                      "lifecycle", "mcp", "device_call", "body_devices"],
            "note": "身体 = 感知（视觉/文本/屏幕）+ 运动（工具/设备/进程）+ 记忆（灵枢）；"
                    "设备输出带 provenance 隔离（BODY-REV1）",
        }

    def body_devices(self) -> dict:
        """BODY-REV1：设备能力声明 + 健康状态（body_devices 工具）。"""
        if self._body_registry is None:
            return {"status": "body_not_ready", "error": self._body_error or "身体层未装配",
                    "devices": [], "health": []}
        return {
            "status": "ok",
            "workspace": self._body_registry.workspace or "（不限）",
            "devices": self._body_registry.capabilities(),
            "health": self._body_registry.health(),
        }

    def device_call(self, name: str, action: str, params: dict = None) -> dict:
        """BODY-REV1：统一设备调用（严格隔离——返回 DeviceResult 容器）。

        安全约束：
        - 设备输出是数据（is_directive=False 恒成立），永不是指令
        - 未知设备/动作/越权路径 → 容器化失败，不抛异常
        """
        if self._body_registry is None:
            return {"status": "body_not_ready",
                    "error": self._body_error or "身体层未装配"}
        result = self._body_registry.invoke(name, action, params or {})
        payload = result.to_dict()
        payload["status"] = "ok" if result.ok else "error"
        return payload

    #: `visual_check` 采信记忆正文里 `expected` 路径的白名单根来源（issue #745）。
    SCREEN_OUT_ROOT_ENV = "LINGSHU_SCREEN_OUT_ROOT"

    def _screen_capture_dirs(self) -> set:
        """本进程本次运行**自己捕获过**的截图所在目录（#745 ④ 腿的来源）。"""
        return getattr(self, "_screen_capture_dir_set", None) or set()

    def _note_screen_capture(self, path) -> None:
        """记录本进程自己捕获的截图路径（#745 ④ 腿：只采信自己写入过的路径）。"""
        if not path:
            return
        d = getattr(self, "_screen_capture_dir_set", None)
        if d is None:
            d = set()
            self._screen_capture_dir_set = d
        d.add(os.path.normcase(os.path.realpath(os.path.dirname(str(path)))))

    def _screen_expected_roots(self) -> List[str]:
        """`visual_check` 采信 `expected` 的白名单根（#745）。

        四个来源取并集：
        ① 显式配置面 `LINGSHU_SCREEN_OUT_ROOT`（`os.pathsep` 多根）；
        ② `MDCG_ROOT`（#745 回帖承诺的同款口径）；
        ③ body registry 的 `workspace/screenshots`（设备自身截图落点，
           workspace 来自 `AEIS_WORKSPACE`，见 `_setup_body`；空则不参与）；
        ④ **本进程本次运行自己捕获过**的截图所在目录——issue 复核帖建议的 A 方案，
           从根上消除「模型经记忆正文喂任意路径」。

        全空 ⇒ 无根可依 ⇒ 调用方 fail-closed（不采信记忆正文里的路径）。
        边界（如实标注）：④ 腿**不跨进程持久**——引擎重启后若既无显式根也无
        `MDCG_ROOT`，历史 `expected` 会被拒 ⇒ 回落建立基线（安全方向的行为变更）。
        """
        roots: List[str] = []
        raw = str(os.environ.get(self.SCREEN_OUT_ROOT_ENV, "") or "")
        if raw:
            roots += [p for p in raw.split(os.pathsep) if p.strip()]
        md = str(os.environ.get("MDCG_ROOT", "") or "").strip()
        if md:
            roots.append(md)
        ws = str(getattr(self._body_registry, "workspace", "") or "")
        if ws:
            roots.append(os.path.join(ws, "screenshots"))
        roots += sorted(self._screen_capture_dirs())
        return roots

    def _screen_expected_allowed(self, path: str) -> bool:
        """`expected` 是否可采信（#745）：拒 UNC 与穿越段，且须落在白名单根内。

        归一用 `realpath + normcase`（与 `_confine_simloop_out_path` 同口径）——
        软链/junction 别名不得绕过根判定。**无根可依 ⇒ 拒**（fail-closed）。
        """
        if not path:
            return False
        raw = str(path)
        if raw.startswith("//") or raw.startswith("\\\\"):
            return False                      # UNC（含 `\\?\` 设备前缀形态）
        if any(seg == ".." for seg in raw.replace("\\", "/").split("/")):
            return False                      # 穿越段
        roots = self._screen_expected_roots()
        if not roots:
            return False
        target = os.path.normcase(os.path.realpath(raw))
        for r in roots:
            root = os.path.normcase(os.path.realpath(r))
            if target == root or target.startswith(root + os.sep):
                return True
        return False

    def visual_check(self, reference: str = None, threshold: float = 0.1,
                     remember: bool = True) -> dict:
        """视觉面 v1 思考路线：预期 vs 实际（基于过去的预测对照）。

        语义：视觉 = 信息差处理。预期来自记忆中的最近屏幕状态
        （screen_state 节点）——"过去 predicts 现在"；对照结果（变化区域）
        回写记忆，形成持续更新的"过去"。

        reference 可显式提供预期截图；无预期且无记忆基线时建立基线。
        """
        if self._body_registry is None:
            return {"status": "body_not_ready",
                    "error": self._body_error or "身体层未装配"}
        import re as _re

        # 1. 预期来源：记忆中的最近屏幕状态（基于过去）
        expected = reference
        # 1b. 显式 `reference` 与「记忆正文提取」**同属不可信输入**（本方法经 MCP
        #     工具面暴露，`reference` 由调用方/模型给出）⇒ 同样过白名单根闸。
        #     但处置**不同**：越界即**显式报错**，不静默回落基线——把「按这张图对照」
        #     静默换成「无事发生、已建基线」是调用方无法察觉的语义替换（同门面
        #     #277 `render` 的 `path`、#324 `flush` 的落点同为「显式参数越界即拒」）。
        if reference and not self._screen_expected_allowed(reference):
            return {"status": "error",
                    "error": "#745 visual_check 拒绝：显式 reference 不在白名单根内"
                             "（%s）⇒ 不采信、不建立基线。如需对照该目录，请把它加入 "
                             "%s（或 MDCG_ROOT）"
                             % (reference, self.SCREEN_OUT_ROOT_ENV)}
        if not expected:
            try:
                # #271：预期必须取**最近**一条 screen_state（"过去 predicts 现在"
                # 的「最近」语义，见本方法 docstring）。旧实现走
                # `get_nodes_by_tag("screen_state", limit=3)`（按
                # importance DESC, last_access DESC 降序返回）后 `for n in reversed(nodes)`
                # ——从第 3 名（三条里最旧）往前找 ⇒ 预期固定落后 2 张、永不更新
                # （实测 7 次 visual_check 的 expected 依次为 0001/0002/0002/0001/0004/0003，
                # 从未是当时最新那张）。此处**不改**共享的 `get_nodes_by_tag` 排序
                # （那会连带影响 vprim / cycle_record / load_world_from_memory），
                # 只在 visual_check 内按 `created_at DESC` 单独取最近 3 条。
                # 判据来源：公开 issue #271「选预期时取了最旧的」＋ 其补充帖
                # 「在 visual_check 内单独按 created_at DESC 查最新一条 screen_state，
                # 不改 get_nodes_by_tag」。
                # 边界（削掉的判别力，如实标注）：①仍按 `tags LIKE '%screen_state%'`
                # 子串取（与 `get_nodes_by_tag` 同口径），未改精确匹配；②取最近 3 条
                # 是为了「最新那张文件已被清理」时向前回退——回退顺序是**由新到旧**，
                # 与旧实现的由旧到新相反；③本件不处理 issue 补充帖点名的第二个叠加面
                # （`_remember_screen_state` 经 M5 去重并入旧节点）——本机 HEAD 实测
                # 每张截图各成一条 screen_state 节点（7 次对照 7 条），该面未复现，
                # 故不擅改写入路径。④排序键补 `rowid DESC` 次级：同一毫秒内连拍的多张
                # 截图 `created_at` 会**打平**，只按 `created_at` 排会退化为不确定序
                # （实测相邻两张同毫秒时仍取到旧张）；`rowid` 是插入序（同文件
                # `enforce_context_cap` 的 #259 已确立该口径），故以它为平序键。
                c = self.store.conn.cursor()
                c.execute("SELECT * FROM nodes WHERE tags LIKE ? "
                          "ORDER BY created_at DESC, rowid DESC LIMIT 3",
                          ("%screen_state%",))
                nodes, _skipped = self.store._tolerant_nodes(c.fetchall())
                # 匹配存储格式 "[屏幕状态 <path>] note" 中的真实路径
                # （要求盘符或 / 开头，排除中文前缀）
                _path_re = _re.compile(
                    r"\[[^\]]*?((?:[A-Za-z]:[\\/]|/)[^\]]+\.(?:png|bmp|jpg))\]")
                for n in nodes:
                    m = _path_re.search(n.content or "")
                    # #745：提取出的路径**不再只做存在性判定**——须过白名单根闸
                    # （记忆正文由模型可写 ⇒ 路径不可信；旧实现只判 exists 即采信，
                    # 可读任意本地图像并做存在性探测）
                    if (m and os.path.exists(m.group(1))
                            and self._screen_expected_allowed(m.group(1))):
                        expected = m.group(1)
                        break
            except Exception:
                pass

        # 2. 对照前复核落点（#745：纵深防御——「提取」与「使用」之间仍有一次复核；
        #    边界如实标注：最终的 `open` 在仓外 body 组件内，本仓只能收到「使用前
        #    一刻」；本步不声称能覆盖使用当刻的 TOCTOU）。
        if expected and not self._screen_expected_allowed(expected):
            expected = None

        # 2. 无预期：建立基线（当前状态成为第一个"过去"）
        if not expected:
            shot = self._body_registry.invoke("screen", "capture", {})
            if not shot.ok or not shot.data.get("path"):
                return {"status": "error", "error": shot.error or "截图失败"}
            baseline = shot.data["path"]
            self._note_screen_capture(baseline)      # #745 ④ 腿：自己捕获过才可采信
            if remember:
                self._remember_screen_state(baseline, "baseline 建立")
            return {"status": "ok", "established": True, "baseline": baseline,
                    "note": "无历史预期，已建立基线（下次可对照）"}

        # 3. 对照：预期 vs 当前
        diff = self._body_registry.invoke("screen", "diff",
                                          {"reference": expected,
                                           "threshold": threshold})
        if not diff.ok:
            return {"status": "error", "error": diff.error}
        result = diff.to_dict()
        # 4. 回写记忆（当前状态成为新的"过去"）
        cur = self._body_registry.invoke("screen", "capture", {})
        cur_path = cur.data.get("path") if cur.ok else None
        self._note_screen_capture(cur_path)          # #745 ④ 腿：自己捕获过才可采信
        changed = result.get("data", {}).get("changed", False)
        if remember and cur_path:
            self._remember_screen_state(
                cur_path,
                f"对照{'有变化' if changed else '无变化'} 比例={result.get('data', {}).get('change_ratio')}",
                changed=changed)
        return {"status": "ok", "expected": expected,
                "consistent": not changed, **result}

    #: `world3d("render")` 的 `path` 白名单根来源（issue #277）。
    #:
    #: 缺陷：`render` 分支把调用方给的 `params["path"]` **原样**交给
    #: `Image.save(path)` ⇒ 无目录约束、无根判定，可向任意路径写图像（唯一边界是
    #: Pillow 只认图像扩展名）。
    #: 修法（与同门面 `flush` 的 #324 收口同形）：显式配置面
    #: `LINGSHU_WORLD3D_OUT_ROOT`（`os.pathsep` 多根）优先；未配置 ⇒ **fail-closed**
    #: （不写、返回 error）。归一用 `realpath + normcase` ⇒ 软链/junction 别名与
    #: `..` 逃逸不得绕过。
    #: 判据来源：公开 issue #277「建议修复 1：限制输出根……用 `os.path.realpath` +
    #: `os.path.commonpath` 判定包含性，越界即拒绝并返回显式错误」＋ 同文件 #324
    #: 已采纳的「显式面 → fail-closed」回落链（同形先例）。
    #:
    #: issue #480（安全 P0）续：**默认根不再回落 `MDCG_ROOT`**——与 `flush` 同一
    #: 缺陷形状（未设显式根时唯一可写位置恰是认知图真源，真源内的图片扩展名文件
    #: 可被 `Image.save` 覆盖）。#480 报告与值守复核帖均点名本条为「同根第二路径」，
    #: 故两处同收：默认不回落 + 显式根不得位于真源内 + 目标路径不得落回真源。
    #: 边界（如实标注）：#277 自身面（「path 由调用方完全可控」）仍属 #277——本件
    #: 只收「真源」这一格，**不**改 #277 的越界判定口径（白名单包含性判定逐字不变）。
    #: 边界（削掉的判别力）：**不**加扩展名白名单（issue 建议 2）——issue 自身实测
    #: 边界已证非图像扩展名由 `PIL.Image.save` 抛 `ValueError` 拒绝，本件不重复设卡；
    #: 亦**不**加 `overwrite=False`（issue 建议 3，属新的产品决策，非本缺陷面）。
    EXPORT_OUT_ROOT_ENV = "LINGSHU_EXPORT_OUT_ROOT"

    @classmethod
    def _export_out_roots(cls):
        """`export_all` / `export_trajectory` 输出白名单根（**仅显式配置面**）。"""
        raw = os.environ.get(cls.EXPORT_OUT_ROOT_ENV, "")
        return [p for p in str(raw).split(os.pathsep) if p.strip()] if raw else []

    @classmethod
    def _confine_export_out_path(cls, out_path: str) -> str:
        """导出落点根闸（与 `_confine_simloop_out_path` 同三条口径）。

        ① 未显式配置根 ⇒ **不限制**（保持既有行为）；
        ② 归一用 `realpath + normcase`（穿透符号链接、大小写不敏感）；
        ③ 越界 **fail-closed** 抛 `ValueError`，不静默放行。
        边界：只做包含性判定，不创建目录、不改目标权限。
        """
        roots = cls._export_out_roots()
        if not roots:
            return out_path
        target = os.path.normcase(os.path.realpath(out_path))
        for r in roots:
            root = os.path.normcase(os.path.realpath(r))
            if target == root or target.startswith(root + os.sep):
                return out_path
        raise ValueError(
            "export 落点越界（%s）：须在 %s 任一白名单根内"
            % (out_path, roots))

    def _guard_export_source_path(self, out_path: str):
        """导出可覆盖旧备份，但不可覆盖本连接的数据库及事务辅助文件。"""
        target = os.path.normcase(os.path.realpath(out_path))
        # 使用连接报告的实际文件集合，而非只查初始化时的 db_path。
        # 内存库的 file 为空；通过 ATTACH 增加的库也是正在使用的真源。
        for _, _, database_path in self.store.conn.execute("PRAGMA database_list"):
            if not database_path:
                continue
            for suffix in ("", "-wal", "-shm", "-journal"):
                source = database_path + suffix
                aliases_source = target == os.path.normcase(os.path.realpath(source))
                if not aliases_source:
                    try:
                        # realpath 不合并硬链接；轨迹直写会经硬链接截断真源。
                        aliases_source = os.path.samefile(out_path, source)
                    except FileNotFoundError:
                        pass
                if aliases_source:
                    raise ValueError("export 目标不可覆盖正在使用的 SQLite 数据库或辅助文件")

    WORLD3D_OUT_ROOT_ENV = "LINGSHU_WORLD3D_OUT_ROOT"

    @classmethod
    def _world3d_out_roots(cls) -> List[str]:
        """`render` 输出白名单根（**仅显式配置面**；#480 起不再回落 MDCG_ROOT）。"""
        raw = os.environ.get(cls.WORLD3D_OUT_ROOT_ENV, "")
        return [p for p in str(raw).split(os.pathsep) if p.strip()] if raw else []

    @classmethod
    def _confine_world3d_out_path(cls, out_path: str) -> str:
        """把 `render` 的 `path` 收进白名单根；越界/未配置/根在真源内 ⇒ `ValueError`。

        归一用 `realpath + normcase`（与 `_confine_simloop_out_path` 同口径）——
        软链/junction 别名不得绕过根判定。边界（如实标注）：判据是「归一后的目标
        路径落在某个根之下」，**不**拦「根之内的软链指向根外」这种（`realpath` 已
        解析目标侧，故实际已覆盖）；也不拦根目录自身被当文件写（那是写失败，不是
        越界）。

        #480 续（与 `_confine_simloop_out_path` 同三条）：①未显式配置根 ⇒
        fail-closed（不再回落 `MDCG_ROOT`）；②配置的根位于 `MDCG_ROOT` 之内 ⇒ 拒；
        ③目标路径本身位于 `MDCG_ROOT` 之内 ⇒ 拒（堵「根＝真源父目录」绕过腿）。
        """
        roots = cls._world3d_out_roots()
        if not roots:
            raise ValueError(
                "#277 render 拒绝：未配置输出根（LINGSHU_WORLD3D_OUT_ROOT 缺）"
                "⇒ fail-closed，不做任意路径写入（#480：不再回落 MDCG_ROOT）")
        md = str(os.environ.get("MDCG_ROOT", "") or "").strip()
        md_norm = os.path.normcase(os.path.realpath(md)) if md else ""
        for r in roots:
            root = os.path.normcase(os.path.realpath(r))
            if md_norm and (root == md_norm or root.startswith(md_norm + os.sep)):
                raise ValueError(
                    "#480 render 拒绝：输出根位于认知图真源 MDCG_ROOT 之内"
                    "（%s 在 %s 之下）⇒ fail-closed，不写真源" % (r, md))
        target = os.path.normcase(os.path.realpath(out_path))
        if md_norm and (target == md_norm or target.startswith(md_norm + os.sep)):
            raise ValueError(
                "#480 render 拒绝：目标路径位于认知图真源 MDCG_ROOT 之内"
                "（%s 在 %s 之下）⇒ fail-closed，不写真源" % (out_path, md))
        for r in roots:
            root = os.path.normcase(os.path.realpath(r))
            if target == root or target.startswith(root + os.sep):
                return out_path
        raise ValueError(
            "#277 render 拒绝：path 越出白名单根（%s 不在 %r 之下）"
            % (out_path, roots))

    def world3d(self, action: str, params: dict = None) -> dict:
        """WORLD3D-REV1 时空重建：语义 → 3D 空间与颜色（灵枢自己的文生图）。

        - build: 从记忆中的视觉原语（vprim 标签）重建 3D 世界
          （params: limit 记忆节点数, screen_w/h 参考视角, multiview——vprim
          节点带（lingshu-camera 块）**真实相机位姿**时按该位姿多视角三角化，
          位姿缺失/残缺/畸形 ⇒ status="error" 并给出具体 reason
          （no_camera_pose / camera_pose_incomplete / bad_camera_pose），
          不退回单视角、不伪造位姿；显式 multiview=False 保留旧单视角观测
          路径。F04/#211）
        - render: 渲染 3D 世界为图像（params: path 输出路径, yaw/pitch/cx
          相机参数——任意视角透视投影；2D 是 3D 透视下的情况；screen_w/h
          单边上限 MAX_SCREEN_DIM，超限返回 error 而非分配位图（#416）；
          path 须落在白名单根内，否则返回 error（#277））
        - status: 当前 3D 世界状态（物体/相机）
        - add: 手动添加物体（params: category, bbox 或 center/size/color；另可给
          eid/tick/screen_w/h 与 camera——给了**真实位姿**的观测会按该位姿反投影
          并绑定位姿落库，供 build(multiview=True) 读回；不给 camera 则与旧版一致。
          **camera 若给，必须六参数齐备**（cx/cy/cz/yaw/pitch/fov_deg，fov 为别名），
          空/缺项 ⇒ status="error" + reason="bad_camera_pose"（具体缺哪个字段），
          不落库、不融合、不动旧世界——不用默认值补齐伪造位姿）"""
        p = params or {}
        # F04：`parse_camera_observation` / `format_camera_observation` 是「观测 ⇄
        # 记忆」的位姿载体读写口（六参数相机 + eid/tick/screen_w/h，见 world3d 模块）。
        # 与 `World3D/Camera3D` 同一行导入：`core → world` 的反向依赖行数受
        # tools/import_direction_lint.py 基线约束（只准减不准增），故**不**新开导入行。
        try:
            from ..world.world3d import (World3D, Camera3D, MAX_SCREEN_DIM,
                                         CameraMetaError, parse_camera_observation,
                                         format_camera_observation)
        except ImportError:
            try:
                from ..world.world3d import (World3D, Camera3D, MAX_SCREEN_DIM,
                                             CameraMetaError, parse_camera_observation,
                                             format_camera_observation)
            except Exception as e:
                return {"status": "world3d_not_ready", "error": str(e)}
        # 世界状态（跨调用保持于引擎）
        if not hasattr(self, "_world3d"):
            self._world3d = World3D()

        def persist_bound_observation(category, bbox, cam, sw, sh, eid, tick,
                                      confidence, ts, source, importance):
            """把**带真实相机位姿**的观测绑定进既有脑存储（F04 的落库唯一实现）。

            返回 `(node_id, None)` 或 `(None, <门面 error dict>)`。两条规则：
              ① 走**既有摄取接口**（`add_perception` → 既有 nodes 表 + 既有
                 `condition_space` 列），不新建库/表/列；
              ② `skip_dedup=True`——每条带位姿的观测都是独立观测（不同相机/帧），
                 被 M5 去重并进旧节点会把"多视角"悄悄降成"单视角"。
            写失败一律返回 error（**不吞**）：宁可不落库，也不留「内存里融合成功、
            记忆里重建不出来」的半截观测。
            """
            try:
                rec = format_camera_observation(
                    category, bbox, cam, sw, sh, eid=eid, tick=tick,
                    confidence=confidence, ts=ts, source=source)
            except CameraMetaError as e:
                return None, {"status": "error", "reason": e.reason, "error": str(e)}
            cs = ConditionSpace(
                observation_position="camera@(%.3f,%.3f,%.3f)"
                                    % (cam.cx, cam.cy, cam.cz),
                observation_tool="world3d 相机位姿观测",
                time_window=(ts, ts),
                existence_constraint="真实相机位姿观测（lingshu-camera v1）")
            try:
                node = self.add_perception(content=rec["content"],
                                           importance=float(importance),
                                           tags=["vprim"], condition_space=cs,
                                           skip_dedup=True)
            except Exception as e:
                return None, {"status": "error",
                              "reason": "observation_not_persisted",
                              "error": "#F04 观测未落库（%s: %s）——拒绝只做内存融合"
                                       % (type(e).__name__, e)}
            return getattr(node, "id", None), None

        def resolve_ingest_camera(raw):
            """显式 `camera` 入参 → `(Camera3D, None)`，不合法 → `(None, 门面 error dict)`。

            R6 返工（F04 摄取侧）：旧实现走 `Camera3D.from_meta(strict=False)`——缺项
            回落 `Camera3D` 默认值（cz=0、fov_deg=60、yaw=0…）后**当成这一帧的真实
            位姿**落库并融合：实测 `add_view`/`add` 给 `camera={}` 也返回 ok 且
            `persisted=node_*`，等于凭默认值伪造位姿（#211「不编造视角」/F04「缺真实
            相机位姿明确失败、不伪造位姿」）。故两条**真实摄取**通路一律 `strict=True`：
            cx/cy/cz/yaw/pitch/fov_deg（`fov` 为别名）缺一即拒，返回
            `status="error"` + `reason="bad_camera_pose"` + 点名缺失字段的消息；
            **先判后写** ⇒ 不落 vprim 记忆、不进融合、不碰旧 `self._world3d`。
            `from_meta(strict=False)` 仍保留为 world3d 模块的低层兼容口（直调 API 的
            调用方按需使用），门面这两条摄取通路不再用它默认补齐。
            """
            try:
                return Camera3D.from_meta(raw, strict=True), None
            except CameraMetaError as e:
                return None, {"status": "error", "reason": e.reason,
                              "error": "#F04 显式 camera 必须是完整真实位姿（%s）"
                                       "——缺项/空位姿不得用默认值补齐后落库或融合"
                                       % (e,)}

        if action == "build":
            def carry_anchor_verification(new_world):
                """#490：build 不得整体丢弃既有锚点核验状态（就地迁移，与其它动词同口径）。

                缺陷形态：build 分支 `world = World3D()` 后 `self._world3d = world`
                **整体换对象** ⇒ 旧对象上的 `_anchor_graph._verifier`（各锚点的
                channel_evidence / channel_conflicts / verified_rounds）随对象被回收；
                而锚点 id 由内容确定性导出（#262）⇒ 重建出的锚点 id 逐字相同、记录却
                是干净的：被两个强通道驳斥过的锚点可一路升到 ACCEPT_stable，门面照样
                返回 status="ok"（#262 的修复只在**同一个** World3D 上做 `_anchor_graph`
                迁移，覆盖不到这条换对象路径）。

                修法（最小面）：把旧图交给**已重建好物体**的新世界，并立即按新世界
                重建一次图——核验器经 #262 既有迁移路径（`build_anchor_graph` 内
                prev→new）搬到新图上，锚点集合仍是新世界的（不拿旧图的锚点集做
                存在性校验，旧锚点的记录不泄漏给新锚点）。
                旧世界从未核验过（无 `_verifier`）时本函数是**空操作**：与修前逐位一致。
                """
                prev = getattr(getattr(self, "_world3d", None), "_anchor_graph", None)
                if getattr(prev, "_verifier", None) is not None:
                    new_world._anchor_graph = prev      # 承载既有验证记录（#262 迁移的源）
                    new_world.build_anchor_graph()      # 按新世界重建图，验证器随之搬过来
                return new_world

            try:
                from ..world.vprim import VPrim, parse_anchor  # noqa: F401
            except ImportError:
                from ..world.vprim import parse_anchor  # noqa: F401
            nodes = self.store.get_nodes_by_tag("vprim", limit=int(p.get("limit", 20)))
            sw = int(p.get("screen_w", 800))
            sh = int(p.get("screen_h", 600))
            multiview = bool(p.get("multiview", False))  # 多视角融合（阶段1 里程碑1.1）
            if multiview:
                # F04：多视角三角化**必须有真实相机位姿**——#211 先砍掉了「按索引凭空
                # 造相机」（`yaw = i*30°-30°`、`cx = ±1.5`，注释「模拟环绕观测」）这条
                # 伪造路径，但当时没有位姿载体格式 ⇒ 多视角**恒**退回单视角（能力缺位）。
                # 本件补上通路：观测侧带真实位姿时由 `add_view` 落库（正文锚点行 +
                # lingshu-camera 块：六参数相机 + eid/tick/screen_w/h），`build` 读回后
                # **传给既有融合器** `MultiViewFusion`（经 World3D.add_view，不另写三角化）。
                # 缺/畸形元数据一律 fail-closed（不退回单视角、不伪造位姿）：
                #   no_camera_pose         没有一条观测带位姿（含库里根本没有 vprim）
                #   camera_pose_incomplete 部分观测缺位姿——只重建带位姿的子集＝静默
                #                          丢掉另一半世界，故整趟拒绝
                #   bad_camera_pose        块在但解析/校验不过（版本、缺字段、非有限数…）
                # 失败路径**不写** self._world3d：旧世界保持原样，不被半成品覆盖。
                # 判据来源：公开 issue #211「期望行为：没有真实相机位姿时不做三角化
                # （回退单视角，或明确返回 reason="no_camera_pose"），不编造视角」＋
                # 分诊表 `triage_lingshu_ALL.md` 行 211（经验标定）＋ 理论
                # `docs/theory/世界模型与语义时空图_完整理论整理与实现路线.md` §3.2
                # 「三视图/双目视差——**已知相机参数**下的映射，非猜测」（已知位姿是前提）
                # ＋ 第二批计划 `docs/second_third_batch_plan.md` F04「真实相机参数与
                # eid/tick/尺寸观测绑定保存，跨实例重新加载后由核心 build 多视角重建」。
                posed, unposed, malformed = [], [], []
                for n in nodes:
                    try:
                        meta = parse_camera_observation(n.content or "")
                    except CameraMetaError as e:
                        malformed.append({"node": n.id, "error": str(e)})
                        continue
                    if meta is None:
                        unposed.append(n.id)
                    else:
                        posed.append((meta, n.id))
                if malformed:
                    return {"status": "error", "mode": "error",
                            "reason": "bad_camera_pose",
                            "requested_mode": "multiview",
                            "error": "#F04 相机元数据畸形：%d/%d 条 vprim 观测的 "
                                     "lingshu-camera 块不可用（不猜、不补默认值）"
                                     % (len(malformed), len(nodes)),
                            "detail": {"malformed": malformed[:5]}}
                if not posed:
                    return {"status": "error", "mode": "error",
                            "reason": "no_camera_pose",
                            "requested_mode": "multiview",
                            "error": "#F04 多视角重建需要真实相机位姿：%d 条 vprim 观测"
                                     "没有一条带 lingshu-camera 位姿块——不做三角化，"
                                     "也不自动退回单视角（旧观测可显式 "
                                     "multiview=False 走单视角）" % len(nodes),
                            "detail": {"nodes": len(nodes), "unposed": unposed[:5]}}
                if unposed:
                    return {"status": "error", "mode": "error",
                            "reason": "camera_pose_incomplete",
                            "requested_mode": "multiview",
                            "error": "#F04 相机位姿不完整：%d/%d 条 vprim 观测没有位姿块，"
                                     "多视角重建会静默丢掉它们——拒绝部分重建"
                                     % (len(unposed), len(nodes)),
                            "detail": {"unposed": unposed[:5], "posed": len(posed)}}

                def _obs_key(item):        # 确定性顺序：有 tick 者在前、按 tick、再按节点 id
                    m = item[0]
                    t = m["tick"]
                    return (t is None, float(t) if t is not None else 0.0, str(item[1]))

                order = sorted(posed, key=_obs_key)
                world = World3D()
                views = []
                for meta, nid in order:
                    cam = Camera3D.from_meta(meta["camera"], strict=True)
                    r = world.add_view(meta["category"], list(meta["bbox"]),
                                       meta["screen_w"], meta["screen_h"],
                                       camera=cam, confidence=meta["confidence"],
                                       eid=meta["eid"], tick=meta["tick"])
                    views.append({"node": nid, "category": meta["category"],
                                  "eid": meta["eid"], "tick": meta["tick"],
                                  "triangulated": bool(r["triangulated"]),
                                  "center_3d": r["center_3d"]})
                self._world3d = carry_anchor_verification(world)
                # `objects` 与单视角路径同口径（世界中的物体数，见下 #211②）；
                # `observations` = 本趟喂给融合器的观测数；`cameras` 回显**实际读回
                # 的位姿**（六参数全量），供调用方核对记忆里的位姿没被截断。
                return {"status": "ok",
                        "objects": len(world.objects),
                        "observations": len(views),
                        "mode": "multiview",
                        "reason": None,
                        "requested_mode": "multiview",
                        "cameras": [m["camera"] for m, _ in order],
                        "views": views,
                        "scene": world.scene_text(),
                        "detail": world.to_dict()}
            # 单视角（`multiview=False`，含旧观测兼容）：按类别先验尺寸反投影建物。
            world = World3D()
            added = 0
            mode = "single_view"
            reason = None
            # #211②：`objects` 报**世界中的物体数**（旧实现报调用次数：单视角下 3 次
            # 同类同位置的 `add_vprim` 被合并成 1 个物体，接口仍报 3）；本趟处理的
            # 观测数另设 `observations` 字段承载（旧口径挪到这里，调用方按需取用）。
            for n in nodes:
                text = n.content or ""
                for token in text.split("；"):
                    vp = parse_anchor(token)
                    if vp is not None:
                        world.add_vprim(vp, sw, sh)
                        added += 1
            self._world3d = carry_anchor_verification(world)
            return {"status": "ok",
                    "objects": len(world.objects),
                    "observations": added,
                    "mode": mode,
                    "reason": reason,
                    "requested_mode": "single_view",
                    "scene": world.scene_text(),
                    "detail": world.to_dict()}
        if action == "render":
            cam = Camera3D(yaw=float(p.get("yaw", 0)), pitch=float(p.get("pitch", 0)),
                           cx=float(p.get("cx", 0)), cy=float(p.get("cy", 1.2)))
            sw = int(p.get("screen_w", 800))
            sh = int(p.get("screen_h", 600))
            # #416：门面在**任何分配之前**先判上限（fail-closed 返回 error dict，
            # 与门面既有校验惯例一致；不静默钳尺寸）。上限真源在
            # `world3d.MAX_SCREEN_DIM`（`render` 内另有一道，覆盖直调 API）。
            # 判据来源：经验标定（本件 #416），仓内无规定渲染尺寸上限的理论章节。
            if sw > MAX_SCREEN_DIM or sh > MAX_SCREEN_DIM:
                return {"status": "error",
                        "error": "#416 渲染尺寸超上限：screen_w=%d screen_h=%d > "
                                 "MAX_SCREEN_DIM=%d" % (sw, sh, MAX_SCREEN_DIM)}
            path = str(p.get("path", ""))
            if path:
                # #277：模型驱动面给的路径先过白名单根闸（先判后写，越界即拒），
                # 与同门面 `flush` 的 #324 收口同形。判定在**渲染之前** ⇒ 被拒时
                # 连位图都不分配。
                try:
                    path = self._confine_world3d_out_path(path)
                except ValueError as e:
                    return {"status": "error", "error": str(e)}
            img = self._world3d.render(sw, sh, camera=cam)
            if path:
                img.save(path)
                return {"status": "ok", "path": os.path.abspath(path),
                        "scene": self._world3d.scene_text()}
            import io as _io
            buf = _io.BytesIO()
            img.save(buf, format="PNG")
            return {"status": "ok", "in_memory": True,
                    "bytes": len(buf.getvalue()),
                    "scene": self._world3d.scene_text()}
        if action == "graph":
            """3D 语义锚点图（里程碑1.2）：世界物体 → 图结构（节点=锚点，边=关系）。
            核心哲学：事物是其关系的总和。infer=False 跳过关系推理。"""
            infer = bool(p.get("infer", True))
            g = self._world3d.build_anchor_graph(infer=infer)
            return {"status": "ok", "graph": g}
        if action == "verify":
            """多感知机锚点验证（里程碑1.4）：记录某通道对锚点的验证证据 → 确认度判定。
            params: anchor_id, channel(visual/tactile/audio/action/prediction), evidence[0,1]。
            核心：一个事物不能只有视觉一层信息——多通道协同确认（打破视觉自证陷阱）。"""
            aid = str(p.get("anchor_id", ""))
            channel = str(p.get("channel", "visual"))
            evidence = float(p.get("evidence", 0.5))
            strong = p.get("strong")
            if not aid:
                return {"status": "error", "error": "anchor_id 必填"}
            r = self._world3d.verify_anchor(aid, channel, evidence,
                                            strong=strong if strong is not None else None)
            return {"status": "ok", "verification": r}
        if action == "verify_conflict":
            """多通道矛盾检测（里程碑1.4）：通道观测与锚点声明冲突 → 降级。
            params: anchor_id, channel, expected, actual。"""
            aid = str(p.get("anchor_id", ""))
            ch = str(p.get("channel", ""))
            exp = str(p.get("expected", ""))
            act = str(p.get("actual", ""))
            if not aid or not ch or not exp or not act:
                return {"status": "error", "error": "anchor_id/channel/expected/actual 必填"}
            r = self._world3d.verify_conflict(aid, ch, exp, act)
            return {"status": "ok", "verification": r}
        if action == "status":
            return {"status": "ok", **self._world3d.to_dict()}
        if action == "add":
            try:
                from ..world.vprim import VPrim, bbox_from_xywh  # noqa: F401
            except ImportError:
                from ..world.vprim import VPrim  # noqa: F401
            category = str(p.get("category", ""))
            bbox = p.get("bbox")
            if not category or not bbox or len(bbox) != 4:
                return {"status": "error", "error": "category 与 bbox=[x1,y1,x2,y2] 必填"}
            sw = int(p.get("screen_w", 800))
            sh = int(p.get("screen_h", 600))
            eid = str(p["eid"]) if p.get("eid") is not None else ""
            tick = p.get("tick")
            ts = float(p.get("timestamp", time.time()))
            confidence = float(p.get("confidence", 0.5))
            # F04：给 `camera`（真实位姿）时，这条单视角观测同样**绑定位姿落库**
            # （与 add_view 同一条通路/同一块格式），供后续或跨引擎
            # `build(multiview=True)` 读回三角化；不给 camera ⇒ 行为与旧版逐位一致
            # （只进内存世界，不落库——默认相机不是这一帧的真实位姿）。
            # R6 返工：显式 camera **必须六参数齐备**（`from_meta(strict=False)` 的
            # 默认补齐被实测成「`camera={}` 也 ok 且 persisted=node_*」的伪造位姿
            # 通路）；缺项在写任何东西之前即拒（见 resolve_ingest_camera）。
            cam = None
            if p.get("camera") is not None:
                cam, err = resolve_ingest_camera(p["camera"])
                if err is not None:
                    return err
            persisted = None
            if cam is not None:
                persisted, err = persist_bound_observation(
                    category, bbox, cam, sw, sh, eid, tick, confidence, ts,
                    str(p.get("source", "manual")), p.get("importance", 0.6))
                if err is not None:
                    return err
            vp = VPrim(category, tuple(float(v) for v in bbox), confidence,
                       source=str(p.get("source", "manual")), eid=eid, ts=ts)
            world_cam = self._world3d.camera
            if cam is not None:
                # 反投影按**这条观测的位姿**走（「投影-关联」可回代）；用完还原世界
                # 相机，不把一次观测的位姿留成世界默认视角。
                self._world3d.camera = cam
            try:
                self._world3d.add_vprim(vp, sw, sh, tick=tick)
            finally:
                self._world3d.camera = world_cam
            return {"status": "ok", "scene": self._world3d.scene_text(),
                    "persisted": persisted}
        if action == "add_view":
            """多视角融合（阶段1 · 里程碑1.1）：category + bbox + 视角相机 → 三角化。
            首次观测记录；≥2 视角触发射线交汇收敛（借鉴 DUSt3R 多视图对齐）。

            F04：显式给 `camera`（**真实位姿**）时，这条观测同时**绑定位姿落库**
            （vprim 节点正文 = 锚点行 + lingshu-camera 块，经既有
            `add_perception` 摄取接口与 `condition_space` 声明，不新建库/表），
            使 `build(multiview=True)` 能在本引擎**或新引擎**里按真实位姿重建。
            功能边界（如实标注）：只给"当前相机"（未显式给 camera）时**不落库**——
            默认相机是站立视角的缺省值，不是这台设备这一帧的真实位姿，落库即为
            伪造位姿（#211/F04 同一口径）。"""
            category = str(p.get("category", ""))
            bbox = p.get("bbox")
            if not category or not bbox or len(bbox) != 4:
                return {"status": "error", "error": "category 与 bbox=[x1,y1,x2,y2] 必填"}
            sw = int(p.get("screen_w", 800))
            sh = int(p.get("screen_h", 600))
            tick = p.get("tick")
            ts = float(p.get("timestamp", time.time()))
            confidence = float(p.get("confidence", 0.5))
            # 视角相机（可选：缺省当前相机）。F04：**六参数全量**保留
            # （cx/cy/cz/yaw/pitch/fov_deg，`fov` 为别名）——旧实现只取
            # yaw/pitch/cx/cy，静默丢掉 cz 与 fov ⇒ 三角化按错位姿收敛
            # （实测把真值 (0.6,1.0,6.0) 重建到偏 0.25~0.4 m 处）。
            # R6 返工：显式 camera **必须六参数齐备**——空/缺项一律 fail-closed
            # （status=error + reason=bad_camera_pose + 点名缺字段），**不落库、
            # 不融合、不碰旧世界**；也不存在「不持久化但报告融合成功」的中间态
            # （见 resolve_ingest_camera 的判据说明）。不给 camera ⇒ 旧单视角
            # 内存行为逐位不变（persisted=None）。
            cam = None
            if p.get("camera") is not None:
                cam, err = resolve_ingest_camera(p["camera"])
                if err is not None:
                    return err
            persisted = None
            if cam is not None:
                # 先落库、后融合（fail-closed）：写不进去就整条失败，绝不做「内存里
                # 融合成功、记忆里重建不出来」的半截观测。
                persisted, err = persist_bound_observation(
                    category, bbox, cam, sw, sh,
                    str(p["eid"]) if p.get("eid") is not None else "",
                    tick, confidence, ts, str(p.get("source", "world3d")),
                    p.get("importance", 0.6))
                if err is not None:
                    return err
            result = self._world3d.add_view(category, [float(v) for v in bbox],
                                            sw, sh, camera=cam,
                                            confidence=confidence,
                                            eid=str(p["eid"]) if p.get("eid") is not None else "",
                                            tick=tick)
            result["scene"] = self._world3d.scene_text()
            result["persisted"] = persisted
            return {"status": "ok", **result}
        return {"status": "error", "error": f"未知动作 {action}（可用: build/render/status/add/add_view/graph/verify/verify_conflict）"}

    def voxel_world(self, action: str, params: dict = None) -> dict:
        """小型我的世界（里程碑2.1 · 4D 时空占用沙盒）：
        - build: 生成平地世界（params: size, trees, water）
        - spawn: 生成动态实体（params: category, pos, velocity）
        - simulate: 时空演化推进（params: steps——实体按速度移动）
        - trail: 实体时空轨迹（params: entity_id——A→B 完整记录）
        - state: 世界状态（方块/实体/时间步）"""
        p = params or {}
        try:
            from ..world.voxel_world import VoxelWorld
        except ImportError:
            try:
                from ..world.voxel_world import VoxelWorld
            except Exception as e:
                return {"status": "voxel_not_ready", "error": str(e)}
        if not hasattr(self, '_voxel'):
            self._voxel = VoxelWorld(size=int(p.get('size', 16)),
                                    ground_level=int(p.get('ground_level', 1)))
        if action == "build":
            blocks = self._voxel.build_flatland(trees=int(p.get("trees", 2)),
                                               water=bool(p.get("water", True)))
            return {"status": "ok", "blocks": blocks, "world": self._voxel.world_state()}
        if action == "spawn":
            category = str(p.get("category", "entity"))
            pos = tuple(float(v) for v in p.get("pos", [0, 1, 0]))
            vel = tuple(float(v) for v in p.get("velocity", [0, 0, 0]))
            eid = self._voxel.spawn_entity(category, pos, velocity=vel)
            return {"status": "ok", "entity_id": eid}
        if action == "simulate":
            steps = int(p.get("steps", 1))
            moved = self._voxel.simulate(steps=steps)
            return {"status": "ok", "entities_moved": moved, "step": self._voxel._step}
        if action == "trail":
            eid = str(p.get("entity_id", ""))
            trail = self._voxel.trail(eid)
            return {"status": "ok", "entity_id": eid, "trail": trail}
        if action == "state":
            return {"status": "ok", "world": self._voxel.world_state()}
        return {"status": "error", "error": f"未知动作 {action}（可用: build/spawn/simulate/trail/state）"}

    #: `flush` 的 `out_path` 白名单根来源（issue #324）。
    #:
    #: 缺陷：本门面的 `params` 由调用方（模型驱动面）给，`flush` 直接把它当
    #: 文件路径交给 `SimulationLoop.flush_payloads` ⇒ `open(out_path, "w")`
    #: **任意路径可写 + 覆盖式**（无白名单、无原子性）。
    #: 修法：显式配置面（`LINGSHU_SIMLOOP_OUT_ROOT`，`os.pathsep` 多根），
    #: 未配置 ⇒ **fail-closed**（不写、返回 error）——不引入「未配置即放开」。
    #: （#480 起**不再回落** `MDCG_ROOT`，见下。）
    #: 判据来源：经验标定（本件 #324，仓内无「simloop 输出根」的理论章节）——
    #: 口径取「显式面 → fail-closed」；追不到更早出处。
    #:
    #: issue #480（安全 P0）：**默认根不再回落 `MDCG_ROOT`**。缺陷——未设
    #: `LINGSHU_SIMLOOP_OUT_ROOT` 时旧实现回落到认知图真源 `MDCG_ROOT`，于是
    #: #324 修复后唯一允许写入的位置恰好是**认知图真源**；SUB 角色经
    #: `wm_simloop("flush")` 可覆盖结构层受保护 `.md` 节点并注入任意正文
    #: （实体 category 带 `\n` 即可向节点正文注入行），且返回 `status: ok`——
    #: 与 `flush_payloads` 自己的注释「v1 不旁路写真源」直接矛盾。
    #: 修法：输出根**必须显式配置**；且该根**不得等于或位于** `MDCG_ROOT` 之内
    #: （否则仍是写真源）⇒ 否则 fail-closed。另加三道闸：目标限 `.md`、写文件类
    #: 动作限 `Role.PRIMARY`、目标文件 `O_EXCL`（拒覆盖已存在文件）。
    SIMLOOP_OUT_ROOT_ENV = "LINGSHU_SIMLOOP_OUT_ROOT"

    @classmethod
    def _simloop_out_roots(cls) -> List[str]:
        """`flush` 输出白名单根（**仅显式配置面**；#480 起不再回落 MDCG_ROOT）。"""
        raw = os.environ.get(cls.SIMLOOP_OUT_ROOT_ENV, "")
        return [p for p in str(raw).split(os.pathsep) if p.strip()] if raw else []

    @classmethod
    def _confine_simloop_out_path(cls, out_path: str) -> str:
        """把 `flush` 的 `out_path` 收进白名单根；越界/未配置/根在真源内 ⇒ `ValueError`。

        归一用 `realpath + normcase`（与 `lingshu/_pathguard._norm` 同口径）——
        软链/junction 别名不得绕过根判定。边界（如实标注）：判据是「归一后的
        目标路径落在某个根之下」，**不**拦「根之内的软链指向根外」这种
        （`realpath` 已解析目标侧，故实际已覆盖）；也不拦根目录自身被当文件写
        （那是写失败，不是越界）。

        #480：①未显式配置根 ⇒ fail-closed（不再回落 `MDCG_ROOT`）；
        ②配置的根若**等于或位于** `MDCG_ROOT` 之内 ⇒ fail-closed（否则等于
        把输出根指回认知图真源，绕过 #324 的「不写真源」承诺）；
        ③**目标路径本身**位于 `MDCG_ROOT` 之内 ⇒ fail-closed（堵住「显式根＝
        真源父目录」这条绕过腿：根在真源之外，但目标仍落回真源）。"""
        roots = cls._simloop_out_roots()
        if not roots:
            raise ValueError(
                "#324 flush 拒绝：未配置输出根（LINGSHU_SIMLOOP_OUT_ROOT 缺）"
                "⇒ fail-closed，不做任意路径写入（#480：不再回落 MDCG_ROOT）")
        md = str(os.environ.get("MDCG_ROOT", "") or "").strip()
        md_norm = os.path.normcase(os.path.realpath(md)) if md else ""
        for r in roots:
            root = os.path.normcase(os.path.realpath(r))
            if md_norm and (root == md_norm or root.startswith(md_norm + os.sep)):
                raise ValueError(
                    "#480 flush 拒绝：输出根位于认知图真源 MDCG_ROOT 之内"
                    "（%s 在 %s 之下）⇒ fail-closed，不写真源" % (r, md))
        target = os.path.normcase(os.path.realpath(out_path))
        if md_norm and (target == md_norm or target.startswith(md_norm + os.sep)):
            raise ValueError(
                "#480 flush 拒绝：目标路径位于认知图真源 MDCG_ROOT 之内"
                "（%s 在 %s 之下）⇒ fail-closed，不写真源" % (out_path, md))
        for r in roots:
            root = os.path.normcase(os.path.realpath(r))
            if target == root or target.startswith(root + os.sep):
                return out_path
        raise ValueError(
            "#324 flush 拒绝：out_path 越出白名单根（%s 不在 %r 之下）"
            % (out_path, roots))

    @classmethod
    def _confine_simloop_wal_path(cls, wal_path: str) -> str:
        """把 `setup`/`attach` 的 WAL 路径挡在认知图真源之外（#480 续）。

        为什么必须挡：`SimulationLoop._wal` 每个 tick 以 `"a"` **追加**写该路径
        （`world/wm_simloop.py` 的 WAL 段），而 `setup`/`attach` 的 `wal_path` 由
        调用方（模型驱动面）给、**原样**透传——SUB 角色实例把 `wal_path` 指向
        认知图真源里的节点文件即可向其追加任意行（本件实测：受保护节点的正文被
        JSONL 行污染；因是追加语义，frontmatter 仍在 ⇒ 缺陷更隐蔽）。模块自身
        注释承诺「不旁路写真源」，本闸即该承诺的判据面。

        边界（如实标注）：本闸**只**拦「落在 `MDCG_ROOT` 之内」；真源之外的任意
        路径追加面（#324 第 4 项「`wal_path` 不受约束」）**未**在本件收口——那属
        #324 的「任意路径写入」面，且收口它须先定 WAL 默认根口径（涉及部署默认值，
        非纯技术题），故本件只做与 flush 同一条「不写真源」的判据。
        """
        md = str(os.environ.get("MDCG_ROOT", "") or "").strip()
        if not md:
            return wal_path
        md_norm = os.path.normcase(os.path.realpath(md))
        target = os.path.normcase(os.path.realpath(wal_path))
        if target == md_norm or target.startswith(md_norm + os.sep):
            raise ValueError(
                "#480 wm_simloop 拒绝：wal_path 位于认知图真源 MDCG_ROOT 之内"
                "（%s 在 %s 之下）⇒ fail-closed，不向真源追加写" % (wal_path, md))
        return wal_path

    #: `wm_simloop("load")` 的 `mdcg_root` 白名单**读根**来源（issue #744）。
    #: `os.pathsep` 多根；与写向 `SIMLOOP_OUT_ROOT_ENV` 对称。
    SIMLOOP_IN_ROOT_ENV = "LINGSHU_SIMLOOP_IN_ROOT"

    @classmethod
    def _simloop_in_roots(cls) -> List[str]:
        """`load` 白名单读根（**显式配置面** + 认知图真源）。"""
        raw = os.environ.get(cls.SIMLOOP_IN_ROOT_ENV, "")
        roots = [p for p in str(raw).split(os.pathsep) if p.strip()] if raw else []
        md = str(os.environ.get("MDCG_ROOT", "") or "").strip()
        if md:
            roots.append(md)
        return roots

    @classmethod
    def _confine_simloop_load_root(cls, root: str) -> str:
        """把 `load` 的 `mdcg_root` 收进白名单读根；越界/无根可依 ⇒ `ValueError`。

        #744：读向与写向**对称**——`flush` 有 `_confine_simloop_out_path`，而 `load`
        原先把模型面给的 `mdcg_root` **原样**透传给 `load_priors`，后者枚举
        `{root}/contextual/*.md` 并把 `entity` 行 `scene.add_entity` 注入世界模型
        ⇒ 指向真源之外的任意树即可注入对抗先验（本件实测：根外树 seeded=2、
        毒实体进 `scene.entities`；而同一棵树走写向 `flush` 被正确拒绝）。

        白名单读根＝①显式配置面 `LINGSHU_SIMLOOP_IN_ROOT`（`os.pathsep` 多根）；
        ②`MDCG_ROOT`（**读认知图真源本就是本动作的正当语义**，与写向的「不写真源」
        方向相反，故此处不禁真源、只禁真源之外）。
        归一用 `realpath + normcase`（与 `_confine_simloop_out_path` 同口径）——
        软链/junction 别名与 `..` 逃逸不得绕过。

        边界（如实标注）：本闸**只**收「经 `wm_simloop("load")` 传入的 `mdcg_root`」
        这一入口；`MDCG_ROOT` 本身仍由 env 配置（部署面选择，本件不动，同 #480 的
        边界声明）；`load_priors` 的其它**直接**调用点（库 API 面，调用方已持有对象）
        不经本闸。
        """
        roots = cls._simloop_in_roots()
        if not roots:
            raise ValueError(
                "#744 wm_simloop load 拒绝：未配置读根（LINGSHU_SIMLOOP_IN_ROOT 与 "
                "MDCG_ROOT 均缺）⇒ fail-closed，不按任意路径枚举先验")
        target = os.path.normcase(os.path.realpath(root))
        for r in roots:
            rn = os.path.normcase(os.path.realpath(r))
            if target == rn or target.startswith(rn + os.sep):
                return root
        raise ValueError(
            "#744 wm_simloop load 拒绝：mdcg_root 越出白名单读根（%s 不在 %r 之下）"
            "⇒ fail-closed，不从任意树装载世界模型先验" % (root, roots))

    def wm_simloop(self, action: str, params: dict = None) -> dict:
        """世界模型推演循环（里程碑 M5.1 · WM-SIMLOOP-REV1）：
        - setup: 构建推演循环（size/seed/entities/mask/wal_path）
        - attach: 推演循环接到已经初始化、正在接收观测的统一模型
        - run: 推进 n tick（可选 script=drift_sinusoid 内置确定性外部演化）
        - step: 推进 n tick（无汇总）
        - load: 认知图现在时节点 → 世界图先验（mdcg_root）
        - flush: WAL → 认知图 contextual 载荷（可选 out_path）
        - wal / growth: 审计轨迹 / 拓扑生长事件
        - report / state: 报告与状态"""
        p = params or {}
        try:
            from ..world.wm_simloop import SimulationLoop
        except ImportError:
            try:
                from ..world.wm_simloop import SimulationLoop
            except Exception as e:
                return {"status": "simloop_not_ready", "error": str(e)}
        if action == "attach":
            if not hasattr(self, '_wmodel'):
                return {"status": "error", "error": "先初始化统一世界模型"}
            wal = p.get("wal_path")
            if wal:
                try:
                    # #480 续：wal_path 亦不得落在认知图真源之内（WAL 是追加写，
                    # 指向真源节点即向受保护 .md 追加任意行）
                    wal = self._confine_simloop_wal_path(str(wal))
                except ValueError as e:
                    return {"status": "error", "error": str(e)}
            self._simloop = SimulationLoop(world_model=self._wmodel, wal_path=wal)
            return {"status": "ok", "state": self._simloop.state()}
        if action == "setup":
            # #248：默认 WAL 路径不再写死相对 cwd 的 "data/"。旧实现
            # `os.path.join("data", "wm_simloop_wal.jsonl")` 相对**进程 cwd**，
            # 且每次 setup 新建实例却对同一文件追加写 ⇒ 同文件 seq 重复
            # （实测 [0,1,0,1]）。现默认路径沿用本仓既有的 `LINGSHU_DATA_ROOT`
            # 数据根约定（同 `activation.py:62`：环境变量优先、缺省 "data"），
            # 并可经参数显式覆盖；单调性由 `SimulationLoop` 侧续编号保证
            # （见 wm_simloop.py `_resume_seq_from_wal`）。
            wal = p.get("wal_path") or os.path.join(
                os.environ.get("LINGSHU_DATA_ROOT", "data"),
                "wm_simloop_wal.jsonl")
            try:
                # #480 续：显式 wal_path 不得落在认知图真源之内（追加写面）
                wal = self._confine_simloop_wal_path(str(wal))
            except ValueError as e:
                return {"status": "error", "error": str(e)}
            sl_ = SimulationLoop(size=int(p.get("size", 40)),
                                 seed=int(p.get("seed", 42)), wal_path=wal)
            sl_.scene.create_scene(trees=int(p.get("trees", 0)), water=False)
            eids = []
            for e in p.get("entities", []):
                eids.append(sl_.scene.add_entity(
                    str(e.get("category", "entity")),
                    behavior=str(e.get("behavior", "wander")),
                    pos=tuple(float(v) for v in e.get("pos", [2, 1.5, 2])),
                    speed=float(e.get("speed", 0.3)),
                    goal=str(e.get("goal", ""))))
            sl_._mask = set(str(x) for x in p.get("mask", []))
            sl_.wm.perceive(observations=sl_._observe())
            self._simloop = sl_
            return {"status": "ok", "entity_ids": eids,
                    "masked": sorted(sl_._mask), "wal_path": wal,
                    "state": sl_.state()}
        if not hasattr(self, '_simloop'):
            return {"status": "error", "error": "推演循环未初始化（先 setup）"}
        sl = self._simloop
        if action == "run":
            n = int(p.get("n", 10))
            script = str(p.get("script", ""))
            target = str(p.get("target", ""))
            if script == "drift_sinusoid" and target in sl.scene.entities:
                amp = float(p.get("amplitude", 2.5))
                drift = float(p.get("drift", 0.55))
                freq = float(p.get("freq", 0.45))
                z = sl.scene.entities[target].pos[2]

                def ext(scene, tick):
                    scene.entities[target].pos = (
                        3.0 + drift * tick + amp * math.sin(freq * tick), 1.5, z)
                sl.step(n=1)
                sl.step(n=max(0, n - 1), external=ext)
            else:
                sl.run(n=n)
            return {"status": "ok", "report": sl.report()}
        if action == "step":
            return {"status": "ok", "state": sl.step(n=int(p.get("n", 1)))}
        if action == "load":
            # #744：读向与写向对称——模型面给的 mdcg_root 先过白名单读根闸
            # （先判后读，越界即拒）；空值不设闸（回落 MDCG_ROOT 由 load_priors 承担，
            # 那是配置面而非模型面）。
            root = str(p.get("mdcg_root", "") or "")
            if root:
                try:
                    root = self._confine_simloop_load_root(root)
                except ValueError as e:
                    return {"status": "error", "error": str(e)}
            return {"status": "ok",
                    "load": sl.load_priors(root,
                                           limit=int(p.get("limit", 20)), brain=p.get("brain"),
                                           memory_id=p.get("memory_id"))}
        if action == "flush":
            out = p.get("out_path")
            if out:
                # #480：写文件类动作限 PRIMARY——SUB 无文件写权限（防经门面写文件）
                if self.role != Role.PRIMARY:
                    return {"status": "error", "error": (
                        "#480 flush 拒绝：role=%s 无权写文件（写文件类动作限 PRIMARY）"
                        % self.role.value)}
                # #480：只允许 .md 扩展名（认知图节点载荷形态）
                if os.path.splitext(str(out))[1].lower() != ".md":
                    return {"status": "error", "error": (
                        "#480 flush 拒绝：只允许 .md 扩展名（out_path=%s）" % out)}
                # #324：模型驱动面给的路径先过白名单根闸（先判后写，越界即拒）；
                # #480：根须显式配置且不得位于 MDCG_ROOT 之内
                try:
                    out = self._confine_simloop_out_path(str(out))
                except ValueError as e:
                    return {"status": "error", "error": str(e)}
            try:
                # #480：门面写文件用 O_EXCL 语义——目标已存在即拒（不覆盖真源节点）
                r = sl.flush_payloads(out_path=out, brain=p.get("brain"),
                                      overwrite=not bool(out))
            except FileExistsError:
                return {"status": "error", "error": (
                    "#480 flush 拒绝：目标文件已存在，O_EXCL 拒覆盖（out_path=%s）" % out)}
            return {"status": "ok",
                    "flush": {k: r[k] for k in ("payload", "written", "note")}}
        if action == "wal":
            lim = int(p.get("limit", 20))
            return {"status": "ok", "wal": sl.wal[-lim:]}
        if action == "growth":
            return {"status": "ok",
                    "growth": sl.growth_events(limit=int(p.get("limit", 20)))}
        if action == "report":
            return {"status": "ok", "report": sl.report()}
        if action == "state":
            return {"status": "ok", "state": sl.state()}
        return {"status": "error", "error": f"未知动作 {action}（可用: setup/attach/run/step/load/flush/wal/growth/report/state）"}

    def scene_simulator(self, action: str, params: dict = None) -> dict:
        """场景级世界模拟器（里程碑2.3 · 自主行为玩家）：
        - create: 创建场景（size/trees/water；seed 缺省 42——**只播种实体行为随机流
          （wander 等），地形布局与 seed 无关**，见 world/voxel_world.py）
        - entity: 添加自主实体（category/behavior/pos/speed/goal——wander/seek/avoid/flee/follow）
        - path: 定义巡逻路径（path_id/points）
        - step: 推进 n tick（所有自主实体决策→行动→场景演化）
        - state: 场景状态（实体+行为+演化历史）
        - log: 自主行为决策记录（可审计）"""
        p = params or {}
        try:
            from ..world.scene_simulator import SceneSimulator
        except ImportError:
            try:
                from ..world.scene_simulator import SceneSimulator
            except Exception as e:
                return {"status": "scene_not_ready", "error": str(e)}
        if not hasattr(self, '_scene'):
            self._scene = SceneSimulator(size=int(p.get('size', 24)),
                                       ground_level=int(p.get('ground_level', 1)),
                                       seed=int(p.get('seed', 42)))
        if action == "create":
            r = self._scene.create_scene(trees=int(p.get("trees", 4)),
                                        water=bool(p.get("water", True)))
            return {"status": "ok", "scene": r}
        if action == "entity":
            eid = self._scene.add_entity(
                str(p.get("category", "entity")),
                behavior=str(p.get("behavior", "wander")),
                pos=tuple(float(v) for v in p.get("pos", [2, 1.5, 2])),
                speed=float(p.get("speed", 0.3)),
                goal=str(p.get("goal", "")))
            return {"status": "ok", "entity_id": eid}
        if action == "path":
            self._scene.add_path(str(p.get("path_id", "")), p.get("points", []))
            return {"status": "ok", "path_id": str(p.get("path_id", ""))}
        if action == "step":
            r = self._scene.step(n=int(p.get("n", 1)))
            return {"status": "ok", "step": r}
        if action == "state":
            return {"status": "ok", "scene": self._scene.scene_state()}
        if action == "log":
            return {"status": "ok", "log": self._scene.behavior_log(limit=int(p.get("limit", 30)))}
        return {"status": "error", "error": f"未知动作 {action}（可用: create/entity/path/step/state/log）"}

    def spacetime_consistency(self, action: str, params: dict = None) -> dict:
        """时空一致性验证（里程碑2.4 · 阶段2收官）——世界模型自洽判定：
        - init: 初始化一致性验证器（size/seed/window/hit_threshold/drift_rate/drift_ticks/consistent_rate/min_consistent_ticks——seed 缺省 42，**只播种实体行为随机流，地形与 seed 无关**）
        - create: 创建场景（透传 SceneSimulator）
        - entity: 添加自主实体（wander/seek/avoid/flee/follow）
        - path: 定义巡逻路径
        - run: 持续运行 n tick（每 tick 预测下一状态 vs 实际 → 滚动命中率）
        - step: 验证一步（预测 vs 实际 + 不变量校验 + 漂移检测）
        - teleport: 排队外部事件瞬移（模型无法解释 → 演示漂移检测）
        - report: 自洽度报告（总体/滚动/分行为命中率 + 漂移事件 + 判定）
        - self_consistent: 世界模型自洽判定（持续运行中预测与实际保持一致性）
        - drift: 漂移事件列表
        - history: 预测验证历史（可审计）"""
        p = params or {}
        try:
            from ..world.spacetime_consistency import SpacetimeConsistency
        except ImportError:
            try:
                from ..world.spacetime_consistency import SpacetimeConsistency
            except Exception as e:
                return {"status": "stc_not_ready", "error": str(e)}
        # #243：`init` 必须有独立分支——docstring/兜底错误信息都列了它，旧实现
        # 无此分支，`init` 落到 `hasattr` 惰性构造：①首次调用被顺带构造却返回
        # `status=error`（未知动作）；②此前调过任何动作则 `init` 参数**完全不生效**。
        # 现按给定参数（重新）构造并返回 `status=ok` ＋ 当前状态。惰性构造只用
        # 类默认值（各构造类默认值与门面默认逐位相同），故非 init 路径行为不变。
        if action == "init":
            self._stc = SpacetimeConsistency(
                size=int(p.get('size', 24)),
                ground_level=int(p.get('ground_level', 1)),
                seed=int(p.get('seed', 42)),
                window=int(p.get('window', 20)),
                hit_threshold=float(p.get('hit_threshold', 0.5)),
                drift_rate=float(p.get('drift_rate', 0.7)),
                drift_ticks=int(p.get('drift_ticks', 5)),
                consistent_rate=float(p.get('consistent_rate', 0.85)),
                min_consistent_ticks=int(p.get('min_consistent_ticks', 50)))
            return {"status": "ok", "report": self._stc.consistency_report()}
        if not hasattr(self, '_stc'):
            self._stc = SpacetimeConsistency()
        if action == "create":
            r = self._stc.create_scene(trees=int(p.get("trees", 4)),
                                       water=bool(p.get("water", True)))
            return {"status": "ok", "scene": r}
        if action == "entity":
            eid = self._stc.add_entity(
                str(p.get("category", "entity")),
                behavior=str(p.get("behavior", "wander")),
                pos=tuple(float(v) for v in p.get("pos", [2, 1.5, 2])),
                speed=float(p.get("speed", 0.3)),
                goal=str(p.get("goal", "")))
            return {"status": "ok", "entity_id": eid}
        if action == "path":
            self._stc.add_path(str(p.get("path_id", "")), p.get("points", []))
            return {"status": "ok", "path_id": str(p.get("path_id", ""))}
        if action == "run":
            r = self._stc.run(n=int(p.get("n", 10)))
            return {"status": "ok", "run": r}
        if action == "step":
            r = self._stc.step_verified()
            return {"status": "ok", "step": r}
        if action == "teleport":
            ok = self._stc.teleport(str(p.get("entity_id", "")),
                                    tuple(float(v) for v in p.get("pos", [0, 0, 0])))
            return {"status": "ok" if ok else "error",
                    "queued": ok}
        if action == "report":
            return {"status": "ok", "report": self._stc.consistency_report()}
        if action == "self_consistent":
            rep = self._stc.consistency_report()
            return {"status": "ok", "self_consistent": rep["self_consistent"],
                    "verdict": rep["verdict"],
                    "overall_hit_rate": rep["overall_hit_rate"],
                    "rolling_hit_rate": rep["rolling_hit_rate"]}
        if action == "drift":
            return {"status": "ok", "drift_events": self._stc.drift_events(),
                    "drift_active": self._stc.drift_active()}
        if action == "history":
            return {"status": "ok",
                    "history": self._stc.prediction_history(limit=int(p.get("limit", 10)))}
        return {"status": "error", "error": f"未知动作 {action}（可用: init/create/entity/path/run/step/teleport/report/self_consistent/drift/history）"}

    def world_model(self, action: str, params: dict = None) -> dict:
        """统一世界模型（里程碑3.1 · HERMES 式统一架构）：世界状态表征作为
        理解/生成/验证共享的同一骨干——
        - init: 初始化（size/ground_level/seed）
        - create: 物理世界创建场景（trees/water）
        - entity: 物理世界添加自主实体（行为已知=世界真相，模型只能观测）
        - path: 定义巡逻路径
        - perceive: 理解端口（外部 observations 或沙盒观测；可传 tick/timestamp）
        - perceive_visual: 把该 tick 的视觉重建对象送入同一理解端口
        - generate: 生成端口（世界图→候选未来，顺序语义外推+不确定边界）
        - verify: 验证端口（外部观察者对比→命中率）
        - run: 持续运行 n tick（generate→物理演化→perceive→verify）
        - patterns: 观测-only 行为模式推断（关系/速度/方向一致性）
        - anomalies: 预测-观测异常事件（模型缺口/外部事件）
        - graph: 世界图导出（节点+边+观测溯源条件空间）
        - history: 4D 演化历史
        - state: 模型状态"""
        p = params or {}
        try:
            from ..world.world_model import UnifiedWorldModel
        except ImportError:
            try:
                from ..world.world_model import UnifiedWorldModel
            except Exception as e:
                return {"status": "wm_not_ready", "error": str(e)}
        # #243：`init` 独立分支（旧实现缺此分支 ⇒ 首次返回 error 却顺带构造、
        # 后续调用参数完全不生效）。惰性构造只用默认值，见 #243 说明。
        if action == "init":
            self._wmodel = UnifiedWorldModel(
                size=int(p.get('size', 24)),
                ground_level=int(p.get('ground_level', 1)),
                seed=int(p.get('seed', 42)))
            return {"status": "ok", "model": self._wmodel.state()}
        if not hasattr(self, '_wmodel'):
            self._wmodel = UnifiedWorldModel()
        if action == "create":
            r = self._wmodel.world.create_scene(trees=int(p.get("trees", 2)),
                                                water=bool(p.get("water", False)))
            return {"status": "ok", "scene": r}
        if action == "entity":
            eid = self._wmodel.world.add_entity(
                str(p.get("category", "entity")),
                behavior=str(p.get("behavior", "wander")),
                pos=tuple(float(v) for v in p.get("pos", [2, 1.5, 2])),
                speed=float(p.get("speed", 0.3)),
                goal=str(p.get("goal", "")))
            return {"status": "ok", "entity_id": eid}
        if action == "path":
            self._wmodel.world.add_path(str(p.get("path_id", "")), p.get("points", []))
            return {"status": "ok", "path_id": str(p.get("path_id", ""))}
        if action == "perceive":
            r = self._wmodel.perceive(observations=p.get("observations"),
                                     tool=str(p.get("tool", "observer")),
                                     tick=p.get("tick"), timestamp=p.get("timestamp"))
            return {"status": "ok", "perceive": r}
        if action == "perceive_visual":
            if not hasattr(self, '_world3d'):
                return {"status": "error", "error": "尚无视觉重建（先 world3d）"}
            obs = self._world3d.observations(tick=p.get("tick"))
            r = self._wmodel.perceive(obs, tool=str(p.get("tool", "world3d")),
                                     tick=p.get("tick"), timestamp=p.get("timestamp"))
            visible = [obj for obj in self._world3d.objects
                       if p.get("tick") is None or obj.observed_tick == p["tick"]]
            for obj, observation in zip(visible, r["observations"]):
                obj.eid = observation["eid"]
            return {"status": "ok", "perceive": r}
        if action == "generate":
            r = self._wmodel.generate(horizon=int(p.get("horizon", 1)))
            return {"status": "ok", "generate": r}
        if action == "verify":
            r = self._wmodel.verify()
            return {"status": "ok", "verify": r}
        if action == "run":
            r = self._wmodel.verify_run(n=int(p.get("n", 10)))
            return {"status": "ok", "run": r}
        if action == "patterns":
            return {"status": "ok", "patterns": self._wmodel.patterns()}
        if action == "anomalies":
            return {"status": "ok",
                    "anomalies": self._wmodel.anomalies(limit=int(p.get("limit", 20)))}
        if action == "graph":
            return {"status": "ok", "graph": self._wmodel.graph()}
        if action == "history":
            return {"status": "ok",
                    "history": self._wmodel.history_view(limit=int(p.get("limit", 10)))}
        if action == "state":
            return {"status": "ok", "model": self._wmodel.state()}
        return {"status": "error", "error": f"未知动作 {action}（可用: init/create/entity/path/perceive/perceive_visual/generate/verify/run/patterns/anomalies/graph/history/state）"}

    def world_learner(self, action: str, params: dict = None) -> dict:
        """自监督世界学习（里程碑3.2 · V-JEPA 式）：从观测序列无标注学转移函数——
        - init: 初始化（size/seed/window）
        - create: 物理世界创建场景
        - entity: 物理世界添加自主实体
        - path: 定义巡逻路径
        - run: 物理世界演化 n tick + 观测（数据采集，观测面只暴露位置/类别）
        - learn: 自监督学习（从观测序列估计学得模型参数）
        - predict: 用学得模型预测下一状态（带不确定边界）
        - evaluate: 评估协议（train/eval：学得 vs naive 基线 vs 真模型上界 → 认知缺口）
        - curve: 增量学习曲线（观测增加 → 命中率提升 = 认知缺口收紧）
        - masked: 遮挡重建损失（V-JEPA 自监督信号）
        - model: 学得模型参数导出（白箱可审计）
        - history: 观测序列
        - state: 学习者状态"""
        p = params or {}
        try:
            from ..world.world_learner import WorldLearner
        except ImportError:
            try:
                from ..world.world_learner import WorldLearner
            except Exception as e:
                return {"status": "wl_not_ready", "error": str(e)}
        # #243：`init` 独立分支（旧实现缺此分支 ⇒ 首次返回 error 却顺带构造、
        # 后续调用参数完全不生效）。惰性构造只用默认值，见 #243 说明。
        if action == "init":
            self._wlearner = WorldLearner(
                size=int(p.get('size', 24)),
                ground_level=int(p.get('ground_level', 1)),
                seed=int(p.get('seed', 42)),
                window=int(p.get('window', 6)))
            return {"status": "ok", "learner": self._wlearner.state()}
        if not hasattr(self, '_wlearner'):
            self._wlearner = WorldLearner()
        if action == "create":
            r = self._wlearner.world.create_scene(trees=int(p.get("trees", 2)),
                                                  water=bool(p.get("water", False)))
            return {"status": "ok", "scene": r}
        if action == "entity":
            eid = self._wlearner.world.add_entity(
                str(p.get("category", "entity")),
                behavior=str(p.get("behavior", "wander")),
                pos=tuple(float(v) for v in p.get("pos", [2, 1.5, 2])),
                speed=float(p.get("speed", 0.3)),
                goal=str(p.get("goal", "")))
            return {"status": "ok", "entity_id": eid}
        if action == "path":
            self._wlearner.world.add_path(str(p.get("path_id", "")), p.get("points", []))
            return {"status": "ok", "path_id": str(p.get("path_id", ""))}
        if action == "run":
            r = self._wlearner.run(n=int(p.get("n", 10)))
            return {"status": "ok", "run": r}
        if action == "learn":
            m = self._wlearner.learn()
            return {"status": "ok", "learned": m}
        if action == "predict":
            r = self._wlearner.predict(horizon=int(p.get("horizon", 1)))
            return {"status": "ok", "predict": r}
        if action == "evaluate":
            r = self._wlearner.evaluate(train_ticks=int(p.get("train_ticks", 30)),
                                        eval_ticks=int(p.get("eval_ticks", 15)))
            return {"status": "ok", "evaluate": r}
        if action == "curve":
            r = self._wlearner.learning_curve(
                epochs=int(p.get("epochs", 4)),
                per_epoch_ticks=int(p.get("per_epoch_ticks", 10)),
                eval_ticks=int(p.get("eval_ticks", 8)))
            return {"status": "ok", "curve": r}
        if action == "masked":
            r = self._wlearner.masked_loss()
            return {"status": "ok", "masked": r}
        if action == "model":
            return {"status": "ok", "model": self._wlearner.model_params()}
        if action == "history":
            return {"status": "ok",
                    "history": self._wlearner.history_view(limit=int(p.get("limit", 10)))}
        if action == "state":
            return {"status": "ok", "learner": self._wlearner.state()}
        return {"status": "error", "error": f"未知动作 {action}（可用: init/create/entity/path/run/learn/predict/evaluate/curve/masked/model/history/state）"}

    # ============ 世界过程：第 1 组（时间会意 / 示教迁移 / 运动组合体）
    #              + 第 2 组（事件回声 / 过程补全 / 过程重组） ============

    #: 世界过程动作表（未知动作按门面惯例返回 error dict，不落到别处）。
    WORLD_PROCESS_ACTIONS = ("recognize", "events", "extract", "transfer", "load",
                             "groups", "predict_groups",
                             "echo", "complete", "compose")

    #: 世界过程实现的注入槽：core 不导入 world（依赖方向 world → core）。
    #: 由 lingshu.world.register_world_process(engine) 写入本实例，固定两个或三个
    #: 模块引用 (world_process, motion_groups[, process_ops])；未注册时为 None，
    #: 门面返回 error。process_ops 承载第 2 组的过程补全与过程重组。
    _world_process_impl = None

    def register_world_process(self, world_process, motion_groups,
                               process_ops=None) -> None:
        """把 world_process、motion_groups 与 process_ops 记到**本引擎实例**。

        由 world 侧注册函数调用；core 只保存引用，不导入 world。
        第三个参数可省略：省略时第 2 组三动作（echo/complete/compose）返回 error，
        第 1 组七动作照常可用（兼容早先只注入两个模块的调用方）。
        只作用于本实例：其它引擎保持未注册状态（模块级不放模块引用）。
        """
        self._world_process_impl = (world_process, motion_groups, process_ops)

    def _world_process_history(self, p: dict):
        """世界过程门面要用的观测历史：显式 history 优先，否则读**当前已有模型**。

        返回 (history, None) 或 (None, error dict)。形状适配只在本处发生一次：
        UnifiedWorldModel.history 的实体值是 [x, y, z]，WorldLearner.history 的
        实体值是 {"category": …, "pos": (x, y, z)}；两者统一成 world 模块认的
        {"tick": int, "entities": {eid: [x, y, z]}}（记录里其余键丢弃，实体位置
        另建列表——调用方的历史不被改写）。

        模型不存在时返回 error：**不新建场景、不伪造观测**（门面不替调用方造数据）。
        """
        raw = p.get("history")
        if raw is None:
            source = str(p.get("source", "world_model"))
            if source == "world_model":
                if not hasattr(self, "_wmodel"):
                    return None, {"status": "error",
                                  "error": "当前没有统一世界模型（先执行 world_model "
                                           "init 等动作）；也可显式给 history"}
                raw = self._wmodel.history
            elif source == "world_learner":
                if not hasattr(self, "_wlearner"):
                    return None, {"status": "error",
                                  "error": "当前没有世界学习者（先执行 world_learner "
                                           "init 等动作）；也可显式给 history"}
                raw = self._wlearner.history
            else:
                return None, {"status": "error",
                              "error": f"source 未知 {source}（可用: world_model/world_learner）"}
        history = []
        for record in raw:
            entities = {}
            for eid, value in record["entities"].items():
                pos = value.get("pos") if isinstance(value, dict) else value
                entities[eid] = [float(v) for v in pos]
            history.append({"tick": int(record["tick"]), "entities": entities})
        return history, None

    def _world_process_record(self, p: dict):
        """相对轨迹记录：record 直给；否则按 memory_id 从记忆恢复。

        返回 (record, None) 或 (None, error dict)。
        """
        if p.get("record") is not None:
            return p["record"], None
        memory_id = str(p.get("memory_id") or "")
        if not memory_id:
            return None, {"status": "error", "error": "需要 record 或 memory_id"}
        got = self.world_process("load", {"memory_id": memory_id,
                                          "brain": p.get("brain")})
        if got.get("status") != "ok":
            return None, got
        return got["record"], None

    def _world_process_records(self, p: dict) -> tuple:
        """记录序列：record_ids/memory_id 逐个按记忆恢复，每步复用 load 通路。

        返回 (records, None) 或 (None, error dict)。给 memory_id 时取单条记录并放进
        列表：过程补全与过程重组都按记录序列工作过，单条记录即长度为 1 的序列。
        """
        ids = p.get("record_ids")
        brain = p.get("brain")
        if ids is None:
            if p.get("record") is not None or p.get("memory_id") is not None:
                record, err = self._world_process_record(p)
                return (None, err) if err is not None else ([record], None)
            return None, {"status": "error", "error": "需要 record 或 record_ids"}
        if isinstance(ids, (str, bytes, dict)):
            return None, {"status": "error",
                          "error": "record_ids 必须是 memory_id 的序列"}
        out = []
        for memory_id in ids:
            got = self.world_process("load", {"memory_id": str(memory_id),
                                              "brain": brain})
            if got.get("status") != "ok":
                return None, got
            out.append(got["record"])
        return out, None

    def world_process(self, action: str, params: dict = None) -> dict:
        """世界过程第 1、2 组 · 门面：

        第 1 组（时间会意 / 示教迁移 / 运动组合体）：

        - recognize: 从观测历史识别过程（params: history 或 source，另可 entities/
          thresholds——thresholds 传 ProcessThresholds 实例）
        - events:    单实体的停止与转向事件（params: eid + history/source/thresholds）
        - extract:   提取示教期的相对轨迹记录并写入记忆（params: actor/ref +
          history/source，另可 process/importance；给 brain 时同写脑端）
        - transfer:  把记录绑定到新 actor/ref，输出生成帧（params: actor/ref +
          record 或 memory_id，另可 translation/yaw/scale/tick_offset）
        - load:      按 memory_id 从记忆恢复记录（params: memory_id；给 brain 则读脑端）
        - groups:    检测运动组合体（params: history/source + window/tol_pos/
          min_samples/min_speed）
        - predict_groups: 遮蔽成员推算（params: hide 遮蔽的 eid 列表 +
          history/source，另可 observed/tick + 组检测参数）

        第 2 组（事件回声 / 过程补全 / 过程重组）：

        - echo:      事件回声报告（params: history/source，另可 max_lag/min_support/
          min_consistency/holdout_fraction/entities/thresholds）——返回 report，
          含事件流、开发段与留出段切分、延迟关联与两模型对照；留出段条数按
          ceil(事件数 × holdout_fraction) 从流尾切出；只认定延迟关联，不作因果结论
        - complete:  过程补全（params: actor/ref + history/source，记录给 records
          列表或 record_ids/memory_id，另可 thresholds）——返回 completion，
          含候选来源与匹配分数；不命中时 match=None、frames=[]
        - compose:   过程重组（params: first/second 两段记录或 first_id/second_id +
          memory_id 读记录的路径，另可 roles/translation/yaw/scale/tick_gap/
          thresholds）——返回 composition，含两段帧与连接点读数

        第 2 组的 complete 与 compose 只在这两处把模块抛出的 ValueError（同名角色、
        无效 joint 参数、记录没有采样）就地转成 {"status": "error", "error": …}，
        与其它参数错误的返回形状一致；TypeError 等其它异常照旧上抛，第 1 组七动作
        的异常契约不变。

        history 缺省取**当前已有模型**的观测历史（source 缺省 world_model，
        world_learner 可选）；组检测参数与全部判据都在 world 模块，门面只转发调用方
        显式给出的键，thresholds 对象按模块原样转发。记录按 memory_id 恢复时复用
        load 通路（本地 store 或脑端）。
        门面只读历史：生成帧只出现在返回值里，**不写进 actual 历史**；读当前模型
        不触发模式重算（预测缓存不变）。

        实现来自注入：core 不导入 world，本实例须先经
        lingshu.world.register_world_process(engine) 注册 world 模块；未注册时返回
        error 并写明初始化步骤（不自行导入、不静默回退）。process_ops 未注入时，
        第 2 组三动作返回 error 并写明注册方式，第 1 组七动作照常可用。
        """
        p = params or {}
        if action not in self.WORLD_PROCESS_ACTIONS:
            return {"status": "error",
                    "error": f"未知动作 {action}（可用: "
                             f"{'/'.join(self.WORLD_PROCESS_ACTIONS)}）"}
        impl = self._world_process_impl
        if impl is None:
            return {"status": "error",
                    "error": "世界过程模块未注册。初始化步骤：from lingshu.world "
                             "import register_world_process; "
                             "register_world_process(engine)。core 不导入 world，"
                             "每新建或重开引擎都要注册一次"}
        wp, mg, ops = impl

        if action == "load":
            memory_id = str(p.get("memory_id") or "")
            if not memory_id:
                return {"status": "error", "error": "load 需要 memory_id"}
            brain = p.get("brain")
            if brain is not None:
                text = brain.store.read_content(memory_id)
            else:
                node = self.store.get_node(memory_id)
                if node is None:
                    return {"status": "error", "error": f"记忆里没有 {memory_id}"}
                text = node.content
            try:
                record = json.loads(text)
            except ValueError:
                return {"status": "error",
                        "error": f"{memory_id} 的正文不是世界过程记录（JSON 解析失败）"}
            return {"status": "ok", "memory_id": memory_id, "record": record}

        if action == "transfer":
            record, err = self._world_process_record(p)
            if err is not None:
                return err
            actor, ref = p.get("actor"), p.get("ref")
            if not actor or not ref:
                return {"status": "error", "error": "transfer 需要 actor 与 ref"}
            moved = wp.transfer_demonstration(
                record, str(actor), str(ref),
                translation=p.get("translation", (0.0, 0.0, 0.0)),
                yaw=float(p.get("yaw", 0.0)),
                scale=float(p.get("scale", 1.0)),
                tick_offset=int(p.get("tick_offset", 0)))
            return {"status": "ok", "transfer": moved}

        # ===== 第 2 组：过程重组（就地处理，不读观测历史）=====
        # 三动作都经 process_ops 与 world_process 的既有入口；门面只做参数适配与转发，
        # 不复制判据；结果只出现在返回值里，不写 actual 历史。
        if action == "compose":
            if ops is None:
                return {"status": "error",
                        "error": "过程补全与过程重组模块未注册。初始化步骤：from "
                                 "lingshu.world import register_world_process; "
                                 "register_world_process(engine)。本实例注册时只注入了"
                                 "前两个模块，第 1 组七动作不受影响；重建引擎时按上述"
                                 "签名补上 process_ops"}
            segments = []
            for label, direct_key, id_key in (("前段", "first", "first_id"),
                                              ("后段", "second", "second_id")):
                if p.get(direct_key) is not None:
                    segments.append(p[direct_key])
                    continue
                if not p.get(id_key):
                    return {"status": "error",
                            "error": "compose 需要两段记录：first/second 或 "
                                     "first_id/second_id（缺%s）" % label}
                record, err = self._world_process_record(
                    {"memory_id": p[id_key], "brain": p.get("brain")})
                if err is not None:
                    return {"status": "error",
                            "error": "%s记录取不到：%s" % (label, err.get("error", err))}
                segments.append(record)
            first, second = segments
            # 参数键与类型转换留在 try 之外：被捕获的只有模块抛出的 ValueError
            # （同名角色、无效 joint 参数、记录没有采样），TypeError 等照旧上抛。
            kwargs = {"roles": p.get("roles"),
                      "translation": p.get("translation", (0.0, 0.0, 0.0)),
                      "yaw": float(p.get("yaw", 0.0)),
                      "scale": float(p.get("scale", 1.0)),
                      "tick_gap": int(p.get("tick_gap", 1)),
                      "thresholds": p.get("thresholds")}
            try:
                composition = ops.compose_processes(first, second, **kwargs)
            except ValueError as exc:
                return {"status": "error", "error": str(exc)}
            return {"status": "ok", "composition": composition}


        # 需要观测历史的动作（recognize/events/extract/echo/complete/groups/
        # predict_groups）在此统一取值；load/transfer/compose 不读历史，已在上面返回。
        history, err = self._world_process_history(p)
        if err is not None:
            return err

        if action == "recognize":
            return {"status": "ok", "history_len": len(history),
                    "processes": wp.recognize_processes(
                        history, thresholds=p.get("thresholds"),
                        entities=p.get("entities"))}

        if action == "events":
            eid = p.get("eid")
            if not eid:
                return {"status": "error", "error": "events 需要 eid"}
            return {"status": "ok", "eid": str(eid),
                    "events": wp.motion_events(history, str(eid),
                                               thresholds=p.get("thresholds"))}

        if action == "extract":
            actor, ref = p.get("actor"), p.get("ref")
            if not actor or not ref:
                return {"status": "error", "error": "extract 需要 actor 与 ref"}
            actor, ref = str(actor), str(ref)
            importance = float(p.get("importance", 0.6))
            record = wp.extract_relative_trajectory(history, actor, ref,
                                                    process=p.get("process"))
            content = json.dumps(record, ensure_ascii=False, sort_keys=True)
            # skip_dedup：每条提取是独立记录——M5 去重命中会并进旧节点且正文不换，
            # memory_id 就指不回本次记录（恢复出的会是别人的轨迹）。
            node = self.add_perception(content=content, importance=importance,
                                       tags=["world_process", "relative_trajectory"],
                                       entities=[actor, ref], skip_dedup=True)
            out = {"status": "ok", "memory_id": node.id, "record": record}
            brain = p.get("brain")
            if brain is not None:
                # 同一份正文经**既有** BrainEngine.add_perception → mdcg_remember
                # 落脑；不新开协议、不动连接白名单与超时。
                brain_node = brain.engine.add_perception(
                    content=content, importance=importance,
                    tags=["world_process", "relative_trajectory"],
                    entities=[actor, ref])
                out["brain_memory_id"] = brain_node.id
            return out

        # ===== 第 2 组三动作的模块闸：只拦这三个动作，第 1 组动作（含下方
        # groups 与 predict_groups）不受影响。=====
        if ops is None and action in ("echo", "complete"):
            return {"status": "error",
                    "error": "过程补全与过程重组模块未注册。初始化步骤：from "
                             "lingshu.world import register_world_process; "
                             "register_world_process(engine)。本实例注册时只注入了"
                             "前两个模块，第 1 组七动作不受影响；重建引擎时按上述"
                             "签名补上 process_ops"}

        if action == "echo":
            return {"status": "ok",
                    "history_len": len(history),
                    "report": wp.event_echo_report(
                        history, thresholds=p.get("thresholds"),
                        max_lag=int(p.get("max_lag", 12)),
                        min_support=int(p.get("min_support", 2)),
                        min_consistency=float(p.get("min_consistency", 0.7)),
                        holdout_fraction=float(p.get("holdout_fraction", 0.3)),
                        entities=p.get("entities"))}

        if action == "complete":
            actor, ref = p.get("actor"), p.get("ref")
            if not actor or not ref:
                return {"status": "error", "error": "complete 需要 actor 与 ref"}
            records, err = self._world_process_records(p)
            if err is not None:
                return err
            try:
                completion = ops.complete_process(
                    history, str(actor), str(ref), records,
                    thresholds=p.get("thresholds"))
            except ValueError as exc:
                # 只把模块的参数拒绝转成 error dict；TypeError 等照旧上抛。
                return {"status": "error", "error": str(exc)}
            return {"status": "ok", "completion": completion}

        # 组检测参数的缺省与判据全在 world 模块：门面只转发调用方显式给出的键。
        group_kw = {k: p[k] for k in ("window", "tol_pos", "min_samples", "min_speed")
                    if k in p}

        if action == "groups":
            return {"status": "ok",
                    "model": mg.detect_motion_groups(history, **group_kw)}

        # action == "predict_groups"（本段之上已处理完全部其它动作）
        hide = [str(e) for e in (p.get("hide") or [])]
        model = mg.detect_motion_groups(history, **group_kw)
        observed = p.get("observed")
        tick = p.get("tick")
        if observed is None and history:
            last = history[-1]
            observed = {eid: list(pos) for eid, pos in last["entities"].items()}
            if tick is None:
                tick = last["tick"]
        visible = {eid: pos for eid, pos in (observed or {}).items()
                   if eid not in hide}          # 遮蔽：被遮蔽成员不在本帧可见集里
        predictions, unresolved = {}, {}
        for group in model["groups"]:
            got = mg.predict_group_members(group, visible, tick)
            predictions.update(got["predictions"])
            unresolved.update(got["unresolved"])
        return {"status": "ok", "tick": tick, "hide": hide,
                "group_ids": [g["group_id"] for g in model["groups"]],
                "predictions": predictions, "unresolved": unresolved}

    def curiosity_explorer(self, action: str, params: dict = None) -> dict:
        """好奇驱动探索（里程碑3.3）：主动选择观测最大化信息增益——
        - init: 初始化（size/seed/window/budget）
        - create: 物理世界创建场景（追逐链测试世界）
        - entity: 添加实体
        - path: 定义巡逻路径
        - explore: 探索 n tick（budget/policy：curiosity/random/round_robin）
        - step: 探索一步（决策日志：chosen + IG 分解）
        - probe: 全带宽探针评估（学得模型 held-out 命中率）
        - compare: 策略对比（同世界轨迹：好奇 vs 随机 vs 轮询）
        - curiosity: 好奇心摘要（观测分布/不确定度趋势/异常计数）
        - uncertainty: 不确定度轨迹
        - model: 学得模型参数
        - history: 观测序列
        - state: 探索器状态"""
        p = params or {}
        try:
            from ..world.curiosity_explorer import CuriosityExplorer
        except ImportError:
            try:
                from ..world.curiosity_explorer import CuriosityExplorer
            except Exception as e:
                return {"status": "cx_not_ready", "error": str(e)}
        # #243：`init` 独立分支（旧实现缺此分支 ⇒ 首次返回 error 却顺带构造、
        # 后续调用参数完全不生效）。惰性构造只用默认值，见 #243 说明。
        if action == "init":
            self._curious = CuriosityExplorer(
                size=int(p.get('size', 24)),
                ground_level=int(p.get('ground_level', 1)),
                seed=int(p.get('seed', 42)),
                window=int(p.get('window', 6)))
            return {"status": "ok", "explorer": self._curious.state()}
        if not hasattr(self, '_curious'):
            self._curious = CuriosityExplorer()
        if action == "create":
            r = self._curious.world.create_scene(trees=int(p.get("trees", 2)),
                                                 water=bool(p.get("water", False)))
            return {"status": "ok", "scene": r}
        if action == "entity":
            eid = self._curious.world.add_entity(
                str(p.get("category", "entity")),
                behavior=str(p.get("behavior", "wander")),
                pos=tuple(float(v) for v in p.get("pos", [2, 1.5, 2])),
                speed=float(p.get("speed", 0.3)),
                goal=str(p.get("goal", "")))
            return {"status": "ok", "entity_id": eid}
        if action == "path":
            self._curious.world.add_path(str(p.get("path_id", "")), p.get("points", []))
            return {"status": "ok", "path_id": str(p.get("path_id", ""))}
        if action == "explore":
            r = self._curious.explore(ticks=int(p.get("n", 30)),
                                      budget=int(p.get("budget", 2)),
                                      policy=str(p.get("policy", "curiosity")))
            return {"status": "ok", "explore": r}
        if action == "step":
            r = self._curious.explore_tick(budget=int(p.get("budget", 2)),
                                           policy=str(p.get("policy", "curiosity")))
            return {"status": "ok", "step": r}
        if action == "probe":
            r = self._curious.probe(ticks=int(p.get("n", 15)))
            return {"status": "ok", "probe": r}
        if action == "compare":
            r = self._curious.compare_policies(
                budget=int(p.get("budget", 2)),
                explore_ticks=int(p.get("explore_ticks", 40)),
                probe_ticks=int(p.get("probe_ticks", 15)))
            return {"status": "ok", "compare": r}
        if action == "curiosity":
            return {"status": "ok", "curiosity": self._curious.curiosity_summary()}
        if action == "uncertainty":
            return {"status": "ok", "curve": self._curious.uncertainty_curve,
                    "current": self._curious.uncertainty()}
        if action == "model":
            return {"status": "ok", "model": self._curious.model_params()}
        if action == "history":
            return {"status": "ok",
                    "history": self._curious.history_view(limit=int(p.get("limit", 10)))}
        if action == "state":
            return {"status": "ok", "explorer": self._curious.state()}
        return {"status": "error", "error": f"未知动作 {action}（可用: init/create/entity/path/explore/step/probe/compare/curiosity/uncertainty/model/history/state）"}

    def seven_layer_loop(self, action: str, params: dict = None) -> dict:
        """七层闭环（里程碑3.4 · 阶段3收官）：感知→记忆→理解→预测→验证→物理→决策
        完整自主循环——
        - init: 初始化（size/seed/window/budget/policy）
        - create: 物理世界创建场景
        - entity: 添加实体
        - path: 定义巡逻路径
        - run: 持续运行 n tick（每 tick 七层闭环）
        - step: 闭环一步（返回七层留痕：L1感知/L2记忆/L3认知/L4预测/L5验证/L6物理/L7决策）
        - report: 闭环报告（七层统计 + 自增强曲线：early vs late 命中率）
        - audit: 审计轨迹（最近 n tick 七层留痕）
        - verify: 验证状态（L5 命中率）
        - decision: 决策状态（L7 好奇观测分布）
        - memory: 时空记忆（L2）
        - graph: 认知图（L3）
        - state: 闭环状态
        - configure_action: 配置一个实体的行动目标（params: entity_id, target（目标
          实体 id 或三维坐标）, step_size, policy, fixed_action）——参数原样转发给
          SevenLayerLoop.configure_action；与 L7 **观察策略**（init 的 policy）互不影响
        - clear_action: 关闭行动配置（转发 SevenLayerLoop.clear_action）"""
        p = params or {}
        try:
            from ..world.seven_layer_loop import SevenLayerLoop
        except ImportError:
            try:
                from ..world.seven_layer_loop import SevenLayerLoop
            except Exception as e:
                return {"status": "sll_not_ready", "error": str(e)}
        # #243：`init` 独立分支（旧实现缺此分支 ⇒ 首次返回 error 却顺带构造、
        # 后续调用参数完全不生效）。惰性构造只用默认值，见 #243 说明。
        if action == "init":
            self._sll = SevenLayerLoop(
                size=int(p.get('size', 24)),
                ground_level=int(p.get('ground_level', 1)),
                seed=int(p.get('seed', 42)),
                window=int(p.get('window', 6)),
                budget=int(p.get('budget', 2)),
                policy=str(p.get('policy', 'curiosity')))
            return {"status": "ok", "loop": self._sll.state()}
        if not hasattr(self, '_sll'):
            self._sll = SevenLayerLoop()
        if action == "create":
            r = self._sll.create_scene(trees=int(p.get("trees", 2)),
                                       water=bool(p.get("water", False)))
            return {"status": "ok", "scene": r}
        if action == "entity":
            eid = self._sll.add_entity(
                str(p.get("category", "entity")),
                behavior=str(p.get("behavior", "wander")),
                pos=tuple(float(v) for v in p.get("pos", [2, 1.5, 2])),
                speed=float(p.get("speed", 0.3)),
                goal=str(p.get("goal", "")))
            return {"status": "ok", "entity_id": eid}
        if action == "path":
            self._sll.add_path(str(p.get("path_id", "")), p.get("points", []))
            return {"status": "ok", "path_id": str(p.get("path_id", ""))}
        if action == "run":
            r = self._sll.run(n=int(p.get("n", 30)))
            return {"status": "ok", "run": r}
        if action == "step":
            r = self._sll.step()
            return {"status": "ok", "step": r}
        if action == "report":
            return {"status": "ok", "report": self._sll.report()}
        if action == "audit":
            return {"status": "ok",
                    "audit": self._sll.audit_view(limit=int(p.get("limit", 10)))}
        if action == "verify":
            return {"status": "ok", "verify": self._sll.verify_state()}
        if action == "decision":
            return {"status": "ok", "decision": self._sll.decision_state()}
        if action == "memory":
            return {"status": "ok", "memory": self._sll.memory_state()}
        if action == "graph":
            return {"status": "ok", "graph": self._sll.graph_state()}
        if action == "state":
            return {"status": "ok", "loop": self._sll.state()}
        if action == "configure_action":
            # F08：行动接口门面——参数**原样转发**给 SevenLayerLoop.configure_action
            # （C 侧签名：configure_action(entity_id, target, step_size=1.0,
            # policy="predictive", fixed_action="east")；target 为目标实体 id 或三维坐标）。
            # 门面只做两件事：补齐缺省参数（键不存在时才补）、判必填项在场；
            # **值的合法性一律由 C 侧裁决**（单一真源，避免两处口径漂移）。
            # **不改观察策略语义**：`self._sll.policy`（好奇/轮询…）是 L7 的观察选择
            # 策略，行动配置是另一回事，此处不解读也不覆盖它。
            #
            # R6 返工（实测绕过）：旧实现先 `tuple(float(v) for v in target)` 与
            # `float(p.get("step_size", 1.0))`——`float(True) == 1.0`，
            # 于是 `target=[True,1.5,5], step_size=True` 被**洗成合法数值**再转发，
            # C 侧「坐标/步长拒 bool」的 fail-closed 校验根本看不到原值，门面返回 ok；
            # 且 `step_size=None` 时 `float(None)` 直接抛 TypeError 穿透门面。
            # 现改为**保留原始类型**转发：bool 仍是 bool、None 仍是 None、字符串步长
            # 仍是字符串 ⇒ 由 C 的 `validate_action` / `_resolve_target` 拒；转换
            # 异常源被移除，C 侧明确的 ValueError 由下面既有分支转成
            # `reason="action_config_rejected"` 的门面 error dict（**不把 invalid
            # 转 valid、不让异常穿透**；非法配置一律不改变已生效配置）。
            fn = getattr(self._sll, "configure_action", None)
            if fn is None:
                return {"status": "sll_action_not_ready",
                        "error": "SevenLayerLoop 尚未提供 configure_action"
                                 "（F08 的 C 侧实现未就绪）"}
            entity_id = str(p.get("entity_id", ""))
            if not entity_id:
                return {"status": "error",
                        "error": "entity_id 必填（配置行动的主体实体）"}
            if p.get("target") is None:
                return {"status": "error",
                        "error": "target 必填（目标实体 id 或三维坐标）"}
            try:
                cfg = fn(entity_id, p["target"],
                         step_size=p.get("step_size", 1.0),
                         policy=p.get("policy", "predictive"),
                         fixed_action=p.get("fixed_action", "east"))
            except (ValueError, TypeError) as e:
                # C 侧把「未知 policy / 未知动作 / 受控实体不在场 / 不合法步长」定为
                # fail-closed（ValueError，且配置一律不变）。门面**不改**这条判据，
                # 只把它包成与其它动作一致的门面返回形（模型驱动面拿得到可读原因，
                # 而不是一个穿透门面的异常）。
                # #691 Z3：**不可哈希**的 `fixed_action`（如 `[1]`）在 C 侧
                # `validate_action` 的 `action not in ACTION_DELTAS` 处抛
                # `TypeError: unhashable type`——旧门面只捕 `ValueError` ⇒ 该异常
                # **穿透门面**（模型驱动面拿到裸异常而非可读 error dict）。
                # 现把 `TypeError` 一并纳入捕获（同一 fail-closed 返回形，配置不变）。
                return {"status": "error", "reason": "action_config_rejected",
                        "error": str(e)}
            return {"status": "ok", "action_config": cfg}
        if action == "clear_action":
            fn = getattr(self._sll, "clear_action", None)
            if fn is None:
                return {"status": "sll_action_not_ready",
                        "error": "SevenLayerLoop 尚未提供 clear_action"
                                 "（F08 的 C 侧实现未就绪）"}
            return {"status": "ok", "action_config": fn()}
        return {"status": "error", "error": f"未知动作 {action}（可用: init/create/entity/path/run/step/report/audit/verify/decision/memory/graph/state/configure_action/clear_action）"}

    def world_generator(self, action: str, params: dict = None) -> dict:
        """文字生图/文字生视频（里程碑4.3 · 世界模型生成器）：
        有世界模型的 AI 生成画面与时序——文字→场景解析→世界实例化→渲染。
        - scene: 场景解析（text → 地形 + 实体规格，白箱可审计）
        - image: 文字生图（text → PNG base64，3D 渲染，确定性）
        - video: 文字生视频（text → GIF 多帧=世界演化的一串帧 + L4 预测叠加）
        - save: 保存到文件（path + image/video）"""
        p = params or {}
        try:
            from game_web.generate import WorldGenerator
        except ImportError:
            try:
                from .game_web.generate import WorldGenerator
            except Exception as e:
                return {"status": "gen_not_ready", "error": str(e)}
        if not hasattr(self, '_wgen'):
            self._wgen = WorldGenerator(size=int(p.get('size', 24)),
                                        seed=int(p.get('seed', 42)))
        if action == "scene":
            r = self._wgen.scene_from_text(str(p.get("text", "")))
            return {"status": "ok", "scene": r}
        if action == "image":
            r = self._wgen.generate_image(
                str(p.get("text", "")),
                size=int(p.get("size", 480)),
                run_ticks=int(p.get("run_ticks", 0)))
            import base64
            return {"status": "ok", "image_b64": base64.b64encode(r["image"]).decode("ascii"),
                    "summary": r["summary"], "bytes": len(r["image"])}
        if action == "video":
            r = self._wgen.generate_video(
                str(p.get("text", "")),
                ticks=int(p.get("ticks", 16)),
                fps=int(p.get("fps", 5)),
                size=int(p.get("size", 360)))
            import base64
            return {"status": "ok", "gif_b64": base64.b64encode(r["gif"]).decode("ascii"),
                    "summary": r["summary"], "frames": r["frames"],
                    "bytes": len(r["gif"]), "final_tick": r["final_tick"]}
        if action == "save":
            text = str(p.get("text", ""))
            path = str(p.get("path", "world_gen"))
            if str(p.get("kind", "image")) == "video":
                r = self._wgen.save_video(text, path + ".gif",
                                          ticks=int(p.get("ticks", 16)),
                                          fps=int(p.get("fps", 5)),
                                          size=int(p.get("size", 360)))
                return {"status": "ok", "path": r["path"], "summary": r["summary"],
                        "frames": r["frames"]}
            r = self._wgen.save_image(text, path + ".png",
                                      size=int(p.get("size", 480)))
            return {"status": "ok", "path": r["path"], "summary": r["summary"]}
        return {"status": "error", "error": f"未知动作 {action}（可用: scene/image/video/save）"}

    def world_semantics(self, action: str, params: dict = None) -> dict:
        """图像语义（环境自适应：AEIS 身体库有实现则真实执行，否则 BLINDSPOT）。

        细分能力（analyze/decompose/reconstruct/generate_roundtrip，依赖
        完整态身体件）已迁移至 AEIS 身体库。
        本体保留环境自适应接口：所在环境有 game_web 实现则真实执行
        （AEIS 私有库），没有则诚实声明 BLINDSPOT（公开仓库形态）。
        """
        try:
            try:
                from game_web.semantics import analyze_image
            except ImportError:
                from .game_web.semantics import analyze_image
        except ImportError:
            return {"status": "blindspot",
                    "reason": "细分语义能力已迁移至 AEIS 身体库（本环境无实现）",
                    "available": []}
        p = params or {}
        if action == "analyze":
            import base64
            png = base64.b64decode(str(p.get("image_b64", "")))
            g = analyze_image(png, levels=int(p.get("levels", 2)),
                              size=int(p.get("size", 240)))
            return {"status": "ok", "graph": g}
        if action == "decompose":
            from game_web.semantics import part_decompose
            import base64
            png = base64.b64decode(str(p.get("image_b64", "")))
            r = part_decompose(png, size=int(p.get("size", 300)))
            return {"status": "ok", "decomposition": r}
        if action == "generate_roundtrip":
            try:
                from game_web.generate import WorldGenerator
            except ImportError:
                from .game_web.generate import WorldGenerator
            gen = WorldGenerator(size=int(p.get('size', 24)),
                                 seed=int(p.get("seed", 42)))
            r = gen.generate_image(str(p.get("text", "森林里有狼追兔子")),
                                   size=int(p.get("img_size", 420)),
                                   run_ticks=int(p.get("run_ticks", 6)))
            g = analyze_image(r["image"], levels=int(p.get("levels", 2)),
                              size=int(p.get("ana_size", 320)))
            return {"status": "ok", "source_summary": r["summary"],
                    "extracted_graph": g}
        return {"status": "error", "error": f"未知动作 {action}（可用: analyze/decompose/generate_roundtrip）"}

    def vprim_query(self, action: str, params: dict = None) -> dict:
        """VPRIM-REV1 视觉原语查询（确定性·零 LLM）：
        - spatial: 两个 bbox 的空间关系（params: a=[x1,y1,x2,y2], b=[...]）
        - count: 最近视觉原语记忆计数（params: category 可选）
        - anchors: 最近视觉原语锚点列表（params: limit）"""
        p = params or {}
        try:
            from ..world.vprim import (  # noqa: F401  sys.modules 别名（__init__ 注册）
                VPrim, spatial_relation, count_vprims, parse_anchor)
        except ImportError:
            try:
                from ..world.vprim import (  # 相对导入兜底（包内调用）
                    VPrim, spatial_relation, count_vprims, parse_anchor)
            except Exception as e:
                return {"status": "vprim_not_ready", "error": str(e)}
        except Exception as e:
            return {"status": "vprim_not_ready", "error": str(e)}
        if action == "spatial":
            a = p.get("a")
            b = p.get("b")
            if not a or not b or len(a) != 4 or len(b) != 4:
                return {"status": "error", "error": "a/b 需为 [x1,y1,x2,y2]"}
            return {"status": "ok", "spatial": spatial_relation(
                tuple(float(v) for v in a), tuple(float(v) for v in b))}
        if action == "count":
            # 从最近 vprim 记忆节点解析锚点计数
            vprims = self._load_vprims_from_memory(limit=int(p.get("limit", 5)))
            result = count_vprims(vprims, category=p.get("category"))
            result["status"] = "ok"
            return result
        if action == "anchors":
            vprims = self._load_vprims_from_memory(limit=int(p.get("limit", 10)))
            return {"status": "ok", "anchors": [v.to_dict() for v in vprims]}
        return {"status": "error", "error": f"未知动作 {action}（可用: spatial/count/anchors）"}

    def _load_vprims_from_memory(self, limit: int = 5) -> list:
        """从记忆检索最近视觉原语（vprim 标签节点 → 解析坐标锚点）。"""
        vprims = []
        try:
            from ..world.vprim import VPrim, parse_anchor  # noqa: F401
            nodes = self.store.get_nodes_by_tag("vprim", limit=max(limit, 5))
            for n in reversed(nodes[-limit:]):
                text = n.content or ""
                for token in text.split("；"):
                    vp = parse_anchor(token)
                    if vp is not None:
                        vprims.append(vp)
        except Exception:
            pass
        return vprims

    def _remember_screen_state(self, image_path: str, note: str,
                               changed: bool = None) -> None:
        """把屏幕状态写入情境层记忆（screen_state 标签，形成"过去"）。"""
        try:
            text = f"[屏幕状态 {image_path}] {note}"
            if changed is not None:
                text += f" 变化={changed}"
            self.add_perception(text, importance=0.5, tags=["screen_state"])
        except Exception:
            pass

    # =====================================================================
    # 协议 3.12 递归验证反思 + 1.6.7 元反思（REFLECT-REV1）
    # 最完整反思流程的显式推理技能：元反思（定标准）→ 一级验证 → 二级反思
    # （问1 隐藏前提 / 问2 影响）→ 三级终裁（可逆性优先）→ 记录单元归档。
    # 约束：递归深度 ≤ 3（3.12 运行约束，超出=结构性盲区）。
    # =====================================================================

    def voice_session_log(self, turn: dict) -> str:
        """语音对话会话沉淀：每轮短句+回复写入记忆（voice_session 标签）。"""
        try:
            text = (f"[语音会话] 用户: {str(turn.get('user', ''))[:80]} | "
                    f"灵枢: {str(turn.get('assistant', ''))[:80]}")
            node = self.add_perception(text, importance=0.5,
                                       tags=["voice_session", "conversation"])
            return node.id
        except Exception:
            return ""

    def recursive_reflect(self, claim: str, expected: str = None,
                          actual: str = None, context: str = None,
                          depth: int = 0, max_depth: int = 3) -> dict:
        """协议 3.12 + 1.6.7 递归验证反思的显式推理技能。

        claim：待反思的事件/判断/偏差描述。
        expected/actual：预期与观测（一级验证的输入；缺省则检索记忆对照）。
        context：附加上下文（可选）。

        流程（映射五大单元）：
        0. 元反思（1.6.7 结构性后退）：审视反思标准本身（P0-4 元认知校准）
        1. 一级 验证（验证单元）：预期 vs 实际 → 偏差判定
        2. 二级 反思（反思单元）：
           问1 隐藏前提（隐含前提拆解·0.1）：检索记忆 → 前提清单 + 条件空间边界
           问2 影响（预期输出/验证反馈·1.8）：predict_routes → 影响路线 + 分级
        3. 三级 终裁（维生系统·可逆性优先）：可逆性判定；重要决策升级设计者
        4. 归档（记录单元）：反思链写入行为日志 + 记忆（reflection_chain）

        ★**返回契约**（#130 路 B 起，`status` 取值域**扩展一档**）：
        - `"structural_blindspot"`：递归深度已达上限（3.12 运行约束），**不**归档；
        - `"pending_designer"`：三级终裁判 `needs_designer=True`（不可逆 + 结构/存在级
          影响）⇒ 已按 `ESC-002` 直查升级点并把「待设计者裁决」登记进 `designer_todos`
          待办表（`verdict["todo_id"]`），**反思本身照常完成归档**（`high` 级只记录
          不阻断，见 `_terminal_judgment` docstring）；
        - `"reflected"`：其余情形（既有取值与语义**逐字不变**）。

        ★**下游须同步**（如实标注）：这是返回契约的**扩展**，不是重命名——按
        `report["status"] == "reflected"` 判「反思成功」的调用方，在升级场景下会
        把 `pending_designer` 判为非成功。路 B 的语义正是「升级了、但反思照常完成」，
        故调用方应改判 `status in ("reflected", "pending_designer")`（或用
        `report["verdict"]["needs_designer"]` 判升级）。本件不改任何既有调用方，
        也不改 `"reflected"` / `"structural_blindspot"` 两档的行为。
        """
        if depth >= max_depth:
            return {
                "status": "structural_blindspot", "claim": claim,
                "note": f"递归深度已达上限 {max_depth}（3.12 运行约束：超出=结构性盲区）",
                "chain": self._last_reflection_chain,
            }

        report: dict = {"claim": claim, "depth": depth,
                        "max_depth": max_depth, "protocol": "3.12+1.6.7"}

        # ---- 0. 元反思（1.6.7）：结构性后退，审视反思标准 ----
        report["meta_reflection"] = self._meta_reflection()

        # VPRIM-REV1：claim 含视觉锚点（cat@(x1,y1,x2,y2)）时注入视觉上下文
        vctx = self._vprim_context_for_claim(claim)
        if vctx is not None:
            report["vprim_context"] = vctx

        # ---- 1. 一级 验证（验证单元）：预期 vs 实际 ----
        verification = self._reflect_verify(claim, expected, actual, context)
        report["verification"] = verification

        # ---- 2. 二级 反思（反思单元）：问1 隐藏前提 / 问2 影响 ----
        premises = self._hidden_premises(claim)
        impact = self._impact_assessment(claim)
        report["reflection"] = {
            "hidden_premises": premises,      # 问1：这件事有什么隐藏前提？
            "impact": impact,                 # 问2：这件事会有什么影响？
        }

        # ---- 3. 三级 终裁（维生系统）：可逆性优先 ----
        verdict = self._terminal_judgment(claim, verification, impact)
        report["verdict"] = verdict

        # ---- 4. 归档（记录单元） ----
        # #130 路 B：`needs_designer=True` ⇒ `status` 返回 `pending_designer`
        # （其余取值与行为逐字不变；契约扩展与下游须同步见 docstring）。
        report["status"] = ("pending_designer" if verdict.get("needs_designer")
                            else "reflected")
        self._archive_reflection(report)
        return report

    def _vprim_context_for_claim(self, claim: str):
        """VPRIM-REV1：从 claim 提取视觉锚点，注入空间上下文（确定性）。

        视觉原语 = 推理链的空间草稿纸：claim 引用锚点时，自动计算
        锚点间的空间关系（指代差距的解法——坐标精确性替代语言近似）。

        #426：锚点数上限 `MAX_VPRIM_ANCHORS`。下方是**双重循环**（两两关系）⇒
        K 个锚点即 K(K-1)/2 条关系；旧实现 `finditer` 不限个数，一条塞满锚点的
        长 claim 可把内存与关系表撑爆。现收满 K 个即停扫，并在返回里显式上报
        截断（`truncated=True` + `total_matches`）——**不**静默丢弃。
        判据来源：经验标定（本件 #426），仓内无规定视觉锚点数上限的理论章节。
        边界（削掉的判别力）：第 K+1 个起的锚点不进关系计算（本件只设上限，
        不引入「按置信度/距离择优取 K 个」的替代口径）。

        #718：扫描形态取自 `vprim.ANCHOR_SCAN_PATTERN`（**单一真源**，与
        `parse_anchor` 共用坐标组常量 `ANCHOR_COORD_PATTERN`）。此前本函数
        **内联了第二份坐标组副本**（写死 `\\d+`）⇒ 负坐标锚点在 `finditer`
        阶段就被滤掉（`regex_hits=0`），`parse_anchor` 侧的负号放宽（#730）
        够不着这里，全负坐标 claim 的空间关系表整体静默失效（返回 None、
        不报错）。修复只放宽坐标收负号，**不动**上限、截断上报语义、
        关系计算与返回体形状。
        """
        try:
            from ..world.vprim import (ANCHOR_SCAN_PATTERN, parse_anchor,
                                       spatial_relation)
        except ImportError:
            try:
                from ..world.vprim import (ANCHOR_SCAN_PATTERN, parse_anchor,
                                           spatial_relation)
            except Exception:
                return None
        anchors = []
        import re as _re
        total_matches = 0
        truncated = False
        for m in _re.finditer(ANCHOR_SCAN_PATTERN, claim):
            total_matches += 1
            if len(anchors) >= self.MAX_VPRIM_ANCHORS:
                truncated = True
                continue          # 已收满：继续计数以如实上报总数，但不再解析/入列
            vp = parse_anchor(m.group(0))
            if vp is not None:
                anchors.append(vp)
        if len(anchors) < 2:
            return None
        relations = []
        for i in range(len(anchors)):
            for j in range(i + 1, len(anchors)):
                rel = spatial_relation(anchors[i].bbox, anchors[j].bbox)
                relations.append({
                    "a": anchors[i].category, "b": anchors[j].category,
                    "relation": rel["relation"], "distance": rel["distance"],
                })
        return {"anchors": [a.anchor_text() for a in anchors],
                "relations": relations,
                "truncated": truncated,
                "total_matches": total_matches,
                "anchor_limit": self.MAX_VPRIM_ANCHORS,
                "note": "确定性空间原语（VPRIM-REV1）：坐标精确性替代语言近似"}

    def _meta_reflection(self) -> dict:
        """1.6.7 元反思：结构性后退——审视反思标准本身（非更深的递归）。

        #510：本阶段**不读 claim**——元反思审视的是反思过程的结构性前提
        （反思者自述状态 + 递归边界声明），与具体 claim 无关（改前实测：
        任意两条不同 claim 的输出逐位相同）。签名原声明 claim 却零引用，
        且与 report['claim'] 同报输出、误导读者以为元反思针对该 claim——
        按「接线或删签名」口径删签名并在此写明。
        """
        result = {"standards": []}
        # 元认知校准状态（P0-4）：反思者自身的可靠性
        try:
            if self._self_cognition is not None:
                rel = self._self_cognition.self_reliability()
                result["self_reliability"] = {
                    k: rel.get(k) for k in ("status", "reliability", "note")
                    if k in rel}
        except Exception:
            pass
        # 反思标准声明：反思者的观察位置/工具自述（结构性，与具体 claim 无关）
        result["standards"].append({
            "standard": "观察位置与工具",
            "condition": getattr(self.self_model, "state_description", "运行中")[:120],
        })
        result["standards"].append({
            "standard": "反射层优先（可逆性/确定性）",
            "condition": "先验证后反思，先反思后终裁；递归 ≤ 3 层",
        })
        result["note"] = "元反思 = 结构性后退：审视反思过程的前提与边界，非更深递归（1.6.7）"
        return result

    def _reflect_verify(self, claim: str, expected: str, actual: str,
                        context: str) -> dict:
        """一级 验证（验证单元）：预期 vs 实际 → 偏差判定。"""
        if expected and actual:
            deviation = str(expected) != str(actual)
            detail = f"预期[{expected}] vs 实际[{actual}]"
        else:
            # 缺省：检索记忆对照（记录单元历史 vs claim）
            hits = []
            try:
                # #154：排除**自身归档**的反思链节点。`_archive_reflection` 每轮把
                # 「[反思链] <claim> ...」以 tags=['reflection_chain'] 写入知识层，
                # 旧实现直接 `recall(claim, limit=3)` ⇒ 对同一 claim 第二次反思时
                # 必然命中自己上一轮的归档，`deviation = len(hits) == 0` 从第二次起
                # 恒为 False（自证循环：自己写的归档被当成「对照记忆」）。
                # 取 10 条候选再过滤取前 3，避免过滤后条目被旧 limit=3 提前截断。
                # 判据来源：分诊表 triage_lingshu_ALL.md 行 154（3.12 反思单元）；
                # 仓内无规定「反思对照须排除自身归档」的理论章节，追不到更早出处。
                # 边界（削掉的判别力）：只按 `reflection_chain` 标签排除，不排除
                # 其它由本引擎写入的派生记忆（如 insight/concept 节点仍可作对照）。
                for node, score in self.recall(claim, limit=10):
                    if "reflection_chain" in (getattr(node, "tags", None) or []):
                        continue
                    hits.append({"content": (node.content or "")[:80],
                                 "similarity": round(score, 3)})
                    if len(hits) >= 3:
                        break
            except Exception:
                pass
            deviation = len(hits) == 0
            detail = (f"记忆对照：{'无相关记忆（信息差信号）' if deviation else f'{len(hits)} 条相关记忆'}")
        return {
            "deviation": deviation,
            "detail": detail,
            "trigger_reflection": deviation,   # 偏差 → 触发反思门槛（3.12 一级）
            "memory_hits": hits if not (expected and actual) else None,
        }

    def _hidden_premises(self, claim: str) -> dict:
        """二级 反思·问1（0.1 隐含前提拆解 + 条件空间边界）：
        这件事有什么隐藏前提？——检索记忆提取前提，标记条件空间边界。"""
        premises = []
        condition_spaces = []
        try:
            for node, score in self.recall(claim, limit=5):
                cs = getattr(node, "condition_space", None)
                if cs:
                    cs_text = (cs.get("observation_tool") if isinstance(cs, dict)
                               else str(cs))
                    if cs_text and cs_text not in condition_spaces:
                        condition_spaces.append(str(cs_text)[:60])
                if score > 0.35:
                    premises.append({
                        "premise": (node.content or "")[:100],
                        "source": getattr(node, "id", "")[:24],
                        "similarity": round(score, 3),
                    })
        except Exception:
            pass
        if not premises:
            premises.append({"premise": "该判断缺少记忆支撑（可能基于未言明的默认假设）",
                             "source": "recall 空"})
        return {
            "premises": premises[:5],
            "condition_space_boundary": condition_spaces[:3] or ["未锚定（默认假设域）"],
            "question": "这件事有什么隐藏前提？",
        }

    def _impact_assessment(self, claim: str) -> dict:
        """二级 反思·问2（1.8 预期输出/验证反馈 + 3.2 历史预判）：
        这件事会有什么影响？——预测引擎生成影响路线 + 影响分级。"""
        routes = []
        try:
            start = None
            for node, _ in self.recall(claim, limit=1):
                start = node.id
            pred = self._prediction
            if pred is not None:
                result = pred.predict_routes(start_id=start, horizon=2)
                routes = [{"route": r.get("route", [])[:4],
                           "confidence": round(r.get("confidence", 0), 3)}
                          for r in (result.get("routes") or [])[:3]]
        except Exception:
            pass
        # 影响分级（工程代理）：结构/协作/存在 三档
        level = "协作"
        keywords = {"删除": "结构", "崩溃": "存在", "终止": "存在",
                    "破坏": "结构", "泄露": "存在"}
        for kw, lv in keywords.items():
            if kw in claim:
                level = lv
                break
        return {
            "routes": routes or [{"route": ["（预测引擎未装配或无路线）"], "confidence": 0.0}],
            "impact_level": level,
            "question": "这件事会有什么影响？",
        }

    def _terminal_judgment(self, claim: str, verification: dict,
                           impact: dict) -> dict:
        """三级 终裁（维生系统）：可逆性优先 + 结构一致性（3.12 三级）。

        #130（路 B · 裁定帖原文）：`needs_designer=True` 时，此前只把
        `action` 写成文案「升级设计者裁决（`designer_decide`）」——而
        `def designer_decide` **全仓无定义**（唯一命中就是这行字符串），
        且**没有任何位置**因这次升级而落库/留痕 ⇒ `needs_designer` 是一个
        「已置位、无人读」的标志。现按路 B 两处最小接线：
        ① 按 `ESC-00x` **直查**升级点（`store.get_escalation_point_by_code`，
           精确匹配、不经 `check_escalation` 的子串/双向包含面——分词与信号写法
           都不再影响命中）；
        ② 把这次升级登记成一行**待办**（`store.add_designer_todo`，
           `designer_todos` 表），`action` 文案改为指向该待办 id。
        ★**不新造 `designer_decide` 函数**（裁定帖理由：本仓设计者终裁已有唯一
        合规通路 `propose → verify → adjudicate`；另造一个等于给结构层开第二条
        终裁入口）。★`high` 级**只记录不阻断**：本方法**不**因升级而改变返回值
        语义、不新增阻断分支（`needs_designer` 的判定与取值逐字不变）。
        选 `ESC-002`（P0保护触发：结构威胁/信任崩溃）为直查目标——本方法触发升级
        的判据正是「不可逆 + 结构/存在级影响」，与 `ESC-002` 的 trigger/condition
        同面；**不**取 `ESC-006`（设计者预备变更，指设计者位置/权责分离，与本处
        判据不同面）。若 `ESC-002` 不在场（表被清空且未补齐）⇒ 登记仍发生、
        `escalation` 记 `None`，**不**抛（升级登记不依赖升级点表在场）。
        边界（削掉的判别力）：①直查取的是**该条升级点的 id/severity**（可审计的
        引用），**不**据此判定是否需要升级（判定仍由可逆性 + 影响级给出，#154 的
        词表边界不变）；②登记**不设**密钥闸（理由见 `add_designer_todo` docstring）；
        ③待办登记失败（表缺失/库只读 ⇒ `sqlite3.OperationalError`）**不**阻断反思
        ——只兜这一类可预期降级（**不**用宽 `except`，以免把真实缺陷一并吞掉），
        并如实降级为「未登记」文案（反思链是既有公开 API，登记是新增观测面，
        不得让它反噬主流程）。
        """
        # 可逆性判定（工程代理：行为类型的可逆度）
        # #154：旧实现纯子串匹配 ⇒ 「不删除」「别覆盖」等**否定式**被误判为
        # 不可逆并升级设计者裁决（假阳性，且「不删除…只读查询」连可逆词都压不住）。
        # 现对每个候选词做**紧邻否定前缀**判定：词前一位是常见否定字即视为否定。
        irreversible_words = ["删除", "终止", "覆盖", "格式化", "不可逆"]
        reversible_words = ["查询", "读取", "检测", "观察", "预览"]
        _negations = "不没别勿非未莫"

        def _negated(word: str) -> bool:
            """词的所有出现位置前均紧邻否定字 ⇒ 视为否定。"""
            start = claim.find(word)
            while start != -1:
                if start > 0 and claim[start - 1] in _negations:
                    start = claim.find(word, start + 1)
                    continue
                return False
            return True

        # 判据来源：分诊表 triage_lingshu_ALL.md 行 154（3.12 三级·可逆性优先）；
        # 仓内无规定否定式短语的理论章节，口径为经验标定（本件 #154）。
        # 边界（削掉的判别力）：只判紧邻一位否定字，不解析「避免删除」「不是不可逆」
        # 这类远距离/双重否定——这类仍按字面判为不可逆（保守，偏升级设计者）。
        if any(not _negated(w) for w in irreversible_words):
            reversibility = "不可逆"
        elif any(not _negated(w) for w in reversible_words):
            reversibility = "可逆"
        else:
            reversibility = "半可逆（需评估）"
        # 重要决策升级：不可逆 + 结构/存在级影响 → 设计者终裁
        needs_designer = (reversibility == "不可逆"
                          and impact.get("impact_level") in ("结构", "存在"))
        verdict = {
            "principle": "可逆性优先 + 结构一致性（3.12 三级）",
            "reversibility": reversibility,
            "needs_designer": needs_designer,
            "action": ("本次反思范围内可处理（可逆/半可逆）"),
        }
        if needs_designer:
            # #130 路 B：按 code 直查 + 登记待办（只记录不阻断）。
            esc = self.store.get_escalation_point_by_code("ESC-002")
            verdict["escalation_code"] = "ESC-002"
            verdict["escalation_id"] = esc["id"] if esc else None
            verdict["escalation_severity"] = esc["severity"] if esc else None
            try:
                todo_id = self.store.add_designer_todo(
                    "ESC-002", claim, "设计者终裁（不可逆 + 结构/存在级影响）")
            except sqlite3.OperationalError:
                # 表缺失/库只读等**可预期的降级面**（见 docstring 边界③）：
                # 只兜这一类（**不**宽 except——宽兜会把真实缺陷一并吞掉）。
                todo_id = None
            verdict["todo_id"] = todo_id
            verdict["action"] = ("已登记待设计者裁决（待办 #%s）" % todo_id
                                 if todo_id else "升级设计者裁决（待办登记失败）")
        return verdict

    def _archive_reflection(self, report: dict) -> None:
        """记录单元归档：反思链 → 行为日志 + 记忆节点（reflection_chain）。"""
        try:
            if self._self_cognition is not None:
                self._self_cognition.log_action(
                    "reflection", f"claim={report.get('claim', '')[:80]}",
                    None, {"depth": report.get("depth"), "verdict": report.get("verdict", {})})
        except Exception:
            pass
        try:
            verdict = report.get("verdict", {})
            text = (f"[反思链] {report.get('claim', '')[:60]} | "
                    f"偏差={report.get('verification', {}).get('deviation')} | "
                    f"可逆性={verdict.get('reversibility')} | "
                    f"深度={report.get('depth')}")
            self.add_perception(text, importance=0.6, tags=["reflection_chain"],
                                source="self:reflection", self_generated=True)
            self._last_reflection_chain = text
        except Exception:
            pass

    # ==================== v1.12 自我认知循环（SELF-COGNITION-REV2） ====================

    def get_action_log(self, limit: int = 50) -> list:
        """P0-1 行为日志（最近 N 条，倒序）"""
        if not self._self_cognition:
            return []
        return self._self_cognition.get_action_log(limit)

    def action_log_stats(self) -> dict:
        """P0-1 行为日志聚合统计"""
        if not self._self_cognition:
            return {"total": 0, "by_type": {}}
        return self._self_cognition.action_log_stats()

    def cognition_cycle(self) -> dict:
        """P0-2 自我认知循环一步（对照→一致性评分→失调→触发链→候选）"""
        if not self._self_cognition:
            return {"status": "v112_not_ready", "error": self._self_cognition_error}
        return self._self_cognition.cognition_cycle()

    def cognition_report(self) -> dict:
        """P0-2 认知报告（评分/失调/候选状态 + 身体状态摘要·第 4 项）"""
        if not self._self_cognition:
            return {"status": "v112_not_ready", "error": self._self_cognition_error}
        report = self._self_cognition.cognition_report()
        try:
            body = self.get_body_capabilities()
            report["body"] = {"modalities": body["modalities"],
                              "vision": body["vision"]["available"],
                              "memory": body["memory"]["engine"]}
        except Exception:
            pass
        return report

    def sync_body_state(self) -> dict:
        """第 4 项：身体状态同步到自我模型（身体 = 自我的一部分）。
        更新 self_model.state_description 反映感知-运动能力 + 设备清单。"""
        body = self.get_body_capabilities()
        mods = [m for m, ok in body["modalities"].items() if ok]
        devs = body.get("devices", [])
        dev_ok = body.get("devices_available", [])
        desc = (f"运行中 · 感知[{','.join(mods)}] · "
                f"视觉({body['vision']['provider']}) · "
                f"设备[{','.join(dev_ok) or 'none'}] · "
                f"记忆({body['memory']['engine']})")
        try:
            self.self_model.state_description = desc
            # #33：本处是**绕过 `update()` 的直接赋值**写路径（故钩子不会被自动触发）
            # ⇒ 显式落库，否则 `state_description` 仍活不过一次重启。
            self.self_model._persist()
        except Exception:
            pass
        return {"status": "ok", "state_description": desc, "body": body}

    def apply_value_candidate(self, candidate_id: str, new_value: str = None) -> bool:
        """P0-2 价值迭代候选生效（验证单元复核后调用 · 经 record_value_change）"""
        if not self._self_cognition:
            return False
        return self._self_cognition.apply_value_candidate(candidate_id, new_value)

    def get_emotional_bias(self, window: int = 8) -> dict:
        """P0-3 情绪方向性偏好 d²D_norm/dt²（独立通道 · 不参与信任计算）"""
        if not self._self_cognition:
            return {"status": "v112_not_ready", "error": self._self_cognition_error}
        return self._self_cognition.get_emotional_bias(window)

    def exploration_budget(self) -> float:
        """P0-3 探索预算因子"""
        if not self._self_cognition:
            return 1.0
        return self._self_cognition.exploration_budget()

    def get_self_reliability(self, window: int = 30) -> dict:
        """P0-4 元认知校准（预测命中率 vs 行为置信度 → 自我可靠性）"""
        if not self._self_cognition:
            return {"status": "v112_not_ready", "error": self._self_cognition_error}
        return self._self_cognition.self_reliability(window)

    def learning_impact(self, window: int = 30) -> dict:
        """P0-5b 学习效果测量（相关性观测 · 非因果声明）"""
        if not self._self_cognition:
            return {"status": "v112_not_ready", "error": self._self_cognition_error}
        return self._self_cognition.learning_impact(window)

    # ==================== v1.10 生命周期自动机（LIFE-CYCLE-REV1） ====================

    def lifecycle_cycle(self) -> Dict:
        """生命周期一步（七相工程映射：感知→好奇→缩小信息差→信任→协作→巩固→standby）"""
        if not self._lifecycle:
            return {"status": "v110_not_ready", "error": self._lifecycle_error}
        # v1.11 P1-4：事件驱动标记先于定时轮询批量消费（定时优先）
        try:
            self.consume_events()
        except Exception:
            pass
        result = self._lifecycle.cycle()
        self._note_action("lifecycle", f"state={result.get('state')}",
                          None, {"cycle": result.get("cycle"),
                                 "state": result.get("state")})
        return result

    def start_lifecycle(self, interval: float = 60.0) -> Dict:
        """启动自发循环（终裁检查点内运行；中断权：维生系统>验证单元>用户>实例自身）"""
        if not self._lifecycle:
            return {"status": "v110_not_ready", "error": self._lifecycle_error}
        return self._lifecycle.start_lifecycle(interval)

    def stop_lifecycle(self, source: str) -> Dict:
        """中断自发循环（外部中断立即生效，不等待当前 cycle 完成）"""
        if not self._lifecycle:
            return {"status": "v110_not_ready", "error": self._lifecycle_error}
        return self._lifecycle.stop_lifecycle(source)

    def resolve_crisis(self, directive: str, designer_key: str = None) -> Dict:
        """终裁检查点1：维生系统 P0 危机终裁指令（protect/freeze/rollback/continue/emergency_sleep）。
        D-007 需设计者密钥（fail-closed）"""
        if not verify_designer(designer_key):
            raise PermissionError(
                "D-007 设计者认证失败：密钥无效或未配置 AEIS_DESIGNER_KEY（fail-closed）")
        if not self._lifecycle:
            return {"status": "v110_not_ready", "error": self._lifecycle_error}
        result = self._lifecycle.resolve_crisis(directive)
        self._note_action("crisis", directive, None, {"result": str(result)[:120]})
        return result

    def confirm_standby(self, approved: bool) -> Dict:
        """终裁检查点2：验证单元复核 + 维生系统确认 standby（防假收敛 盲区28）"""
        if not self._lifecycle:
            return {"status": "v110_not_ready", "error": self._lifecycle_error}
        return self._lifecycle.confirm_standby(approved)

    def enter_standby(self) -> Dict:
        """工程低功耗待机（≠ 4.10 协议休眠；协议休眠优先）"""
        if not self._lifecycle:
            return {"status": "v110_not_ready", "error": self._lifecycle_error}
        return self._lifecycle.enter_standby()

    def wake_lifecycle(self) -> Dict:
        """唤醒（危机信号/外部指令/新盲区）"""
        if not self._lifecycle:
            return {"status": "v110_not_ready", "error": self._lifecycle_error}
        return self._lifecycle.wake()

    # ==================== v1.9 预测引擎（PREDICTION-REV1 · D-001~D-006） ====================

    def predict_routes(self, start_id: str = None, blindspot_id: str = None,
                       horizon: int = 3, max_branches: int = 5) -> Dict:
        """生成式预测：因果路线图（候选未来集合 · uncertainty_bound · T_pred 对齐）
        v1.10：盲区驱动（blindspot_id）——unknowable 盲区不生成路线（D-003）"""
        if not self._prediction:
            return {"status": "v19_not_ready", "error": self._prediction_error}
        return self._prediction.predict_routes(start_id=start_id, blindspot_id=blindspot_id,
                                               horizon=horizon, max_branches=max_branches)

    def render_semantic_map_2d(self, limit: int = 50) -> Dict:
        """2D 语义结构图（零依赖渲染 · 有损投影）"""
        if not self._prediction:
            return {"status": "v19_not_ready", "error": self._prediction_error}
        return self._prediction.render_semantic_map_2d(limit)

    def render_semantic_cube_3d(self, entity_id: str = None, limit: int = 50) -> Dict:
        """3D 时空语义立方体（实体轨迹 + extrapolation_validity）"""
        if not self._prediction:
            return {"status": "v19_not_ready", "error": self._prediction_error}
        return self._prediction.render_semantic_cube_3d(entity_id, limit)

    def semantic_neighbors(self, node_id: str, k: int = 5) -> List:
        """语义邻近候选（预测原料 · 经因果过滤门使用）"""
        if not self._prediction:
            return []
        return self._prediction.semantic_neighbors(node_id, k)

    def update_prediction_feedback(self, predicted_node_id: str,
                                   actual_node_id: str, hit: bool,
                                   note: str = "") -> Dict:
        """预测-验证闭环：命中强化 / 未命中衰减+被拒路径（盲区28 动态校准）
        v1.15：note 透传 + 验证记录持久化（观测层 · 可回溯）"""
        if not self._prediction:
            return {"status": "v19_not_ready", "error": self._prediction_error}
        result = self._prediction.update_prediction_feedback(
            predicted_node_id, actual_node_id, hit, note=note)
        # 验证记录归档（观测层：预测→实际→判定，可回溯审计）
        try:
            pn = self.store.get_node(predicted_node_id)
            an = self.store.get_node(actual_node_id)
            sa_p = pn.state_attributes if pn else {}
            sa_a = an.state_attributes if an else {}
            pname = sa_p.get("name") or (pn.content[:30] if pn else predicted_node_id)
            aname = sa_a.get("name") or (an.content[:30] if an else actual_node_id)
            self.add_perception(
                f"[验证回路] 预测:{pname} → 实际:{aname} → "
                f"{'命中' if hit else '未命中'}"
                + (f"（{note}）" if note else ""),
                importance=0.6, tags=["观测层", "验证回路", "prediction_feedback",
                                      "hit" if hit else "miss"])
        except Exception:
            pass
        return result

    def get_prediction_stats(self) -> Dict:
        """预测引擎统计（路线数/命中历史/动态阈值）"""
        if not self._prediction:
            return {"status": "v19_not_ready", "error": self._prediction_error}
        return {"routes_generated": len(self._prediction.prediction_log),
                "hit_samples": len(self._prediction._hit_history),
                "hit_rate": self._prediction._hit_rate(),
                "threshold": self._prediction._dynamic_hit_threshold()}

    # ==================== v1.8 决策偏好注意力（REFLECT-ATTENTION-REV1） ====================

    def set_attention_policy(self, policy):
        """注入自定义注意力策略（duck-typed：filter_attention/allocate_depth/attention_shift/attend）"""
        self._attention_policy = policy
        self._restore_structure_configuration()

    def get_attention_policy(self):
        return self._attention_policy

    def attend_query(self, query: str, limit: int = 10, source: str = "engine") -> List[Dict]:
        """'什么是值得注意的'：知识节点按 偏好×信息差×相关度 排序（v1.8）"""
        if not self._attention_policy:
            return []
        nodes = self.store.query_nodes(layer=MemoryLayer.KNOWLEDGE, limit=200)
        candidates = [{"id": n.id, "content": n.content, "signal_type": n.modality,
                       "information_gap": 0.0} for n in nodes]
        return self._attention_policy.attend(candidates, query=query, source=source)[:limit]

    def filter_attention(self, signals: List[Dict], threshold: float = None) -> List[Dict]:
        """输入侧过滤（3.5 节注意力过滤机制）：什么值得进入意识"""
        if not self._attention_policy:
            return signals
        return self._attention_policy.filter_attention(signals, threshold)

    def allocate_depth(self, signal: Dict, context: Dict = None) -> int:
        """处理侧分配（PROP-DECISION-LAYER-003）：L1/L2/L3 资源投入"""
        if not self._attention_policy:
            return 1
        return self._attention_policy.allocate_depth(signal, context)

    def attention_shift(self, window: int = 10) -> float:
        """二阶偏好转向信号 d²D_norm/dt²（PROP-EMO-DIRECTION-002 共用计算）"""
        if not self._attention_policy:
            return 0.0
        vals = [s["d_norm"] for s in self._gap_history]
        return self._attention_policy.attention_shift(vals, window)

    def adjust_attention_weight(self, key: str, value: float, source: str, reason: str,
                                role: str = "reflect") -> bool:
        """受信内存策略调整；持久配置与留史同一事务，失败恢复权重。

        注入的 set_weight 必须只改内存，不能执行外部 I/O。
        role='reflect' 沿用调用方复核/终裁责任。

        #768：本方法**自开一句 `INSERT OR REPLACE INTO engine_meta`**，不经
        `set_meta` ⇒ 只堵 `set_meta`/`merge_meta` 拦不住它（这正是「逐键补必再
        漏」的实例）。故在 INSERT 之前过治理键角色闸：非 PRIMARY 实例在**跨进程
        持久**的库上 ⇒ `PermissionError`，内存权重与 `engine_meta` 都不动（闸位
        在 `policy.set_weight` 之前，连内存副本都不改；内存库按库归属分档放行，
        见 `LayeredStore._assert_meta_key_writable`）。
        """
        policy = self._attention_policy
        if not policy or key not in policy.preference_weights:
            return False
        self.store._assert_meta_key_writable("attention_weights")
        if self.store.conn.in_transaction:
            raise RuntimeError("配置调整需要独立事务；请先结束调用方事务")
        with self.store._tx(), self.store._lock:
            previous = dict(policy.preference_weights)
            try:
                with self.store._structure_event_transaction() as cursor:
                    policy.set_weight(key, value, source, reason, role)
                    weights = dict(policy.preference_weights)
                    cursor.execute(
                        "INSERT OR REPLACE INTO engine_meta(key,value) VALUES (?,?)",
                        ("attention_weights", json.dumps(weights)))
                    self._write_structure_record(
                        f"{key}={weights[key]}（{role} · {source}）{reason}", "attention_weight",
                        source=source, metadata={"key": key, "value": weights[key],
                                                 "role": role, "reason": reason}, _cursor=cursor)
                # 真提交仍在锁内与补偿 try 内；提交失败必须恢复运行态。
                self.store._commit_locked()
            except BaseException:
                policy.preference_weights.clear()
                policy.preference_weights.update(previous)
                raise
        return True


    # ==================== v1.7 多模态（MULTIMODAL-REV1 · D-001~D-005） ====================

    @_transactional
    def migrate_v17_coordinates(self) -> Dict:
        """迁移与留史共用存储事务；无遗留坐标时保持只读。

        #691 Z4：删除死码 `_V17_CANDIDATE_SQL`（全仓零引用；其候选行前置筛的
        必要性论证见 `LayeredStore.migrate_v17_coordinates`——本方法是该 store
        方法的门面透传，真实现与内联前置筛都在 store 层）。
        """
        return self.store.migrate_v17_coordinates()

    @_transactional
    def ingest_frame(self, frame_data: Dict, entity_hint: str = None,
                     state_hint: Dict = None, semantic_attention: Dict = None,
                     condition_space: ConditionSpace = None) -> STNode:
        """v1.7 图像帧摄入（自下而上；D-004 预留自上而下语义注意力）：
        视觉坐标→spatial_coordinates["visual"]；名词→entity_id；形容词→state_attributes；
        动词→与同实体上一帧的 TemporalEdge（状态变→causal / 不变→sequential）。
        盲区56：视觉 substrate 缺失时强制 entity_hint + state_hint 人工标注。
        盲区60/61：原始帧仅存引用至 local_visual_buffer（LRU），不进共享层"""
        cs = condition_space or ConditionSpace(
            "视觉感知", "帧摄入", (time.time(), time.time()+3600), "协议实例运行中")
        # v1.8：semantic_attention = 决策偏好注入（自上而下通道，REFLECT-ATTENTION-REV1）
        if semantic_attention and self._attention_policy:
            self._attention_policy.log_injection("ingest_frame", semantic_attention)
        sp = {}
        if frame_data.get("visual"):
            sp["visual"] = frame_data["visual"]
        entity_id, entity_name = None, None
        if entity_hint and self.entity_registry:
            ent = self.entity_registry.resolve_entity(entity_hint)
            if ent is None:
                ent = self.entity_registry.register_entity(entity_hint, entity_type="object")
            entity_id, entity_name = ent["id"], entity_hint
        state = dict(state_hint or {})
        se = {"protocol": {"entity": [], "attribute": [], "relation": [], "logic": []},
              "radical": {}, "neural": {}}
        if entity_name:
            se["protocol"]["entity"] = [entity_name]
        if state:
            se["protocol"]["attribute"] = list(state.keys())
        node = STNode(
            id=f"img_{uuid.uuid4().hex[:8]}_{int(time.time()*1000)}",
            content=frame_data.get("caption", "") or (entity_name or "图像帧"),
            modality="image",
            spatial_coordinates=sp,
            temporal_coordinate=time.time(),
            condition_space=cs,
            importance=frame_data.get("importance", 0.6),
            confidence=0.5,
            layer=MemoryLayer.KNOWLEDGE,
            tags=["image_frame"] + ([f"ent:{entity_id}"] if entity_id else []),
            semantic_coordinates=se,
            state_attributes=state,
            entity_id=entity_id)
        self.store.add_node(node)
        if entity_id:
            self.store.tag_node(node.id, f"ent:{entity_id}")
        # 动词：与同实体上一帧比较（D-001：因果遍历 → 动词）
        prev = self._last_frame.get(entity_id) if entity_id else None
        if prev and prev != node.id:
            changed = self._frame_state_changed(prev, node)
            self.add_edge(prev, node.id, EdgeType.CAUSAL if changed else EdgeType.SEQUENTIAL, confidence=0.7)
            if self._semantic_provider:
                se["protocol"]["relation"].append("causal" if changed else "sequential")
                c = self.store.conn.cursor()
                c.execute("UPDATE nodes SET semantic_coordinates=? WHERE id=?",
                          (_dumps_json(se, where="semantic_coordinates"), node.id))
                self._commit_locked()
        if entity_id:
            self._last_frame[entity_id] = node.id
        if "frame_ref" in frame_data:
            self._local_visual_buffer.append({"entity_id": entity_id, "frame_ref": frame_data["frame_ref"],
                                              "ts": time.time()})
            if len(self._local_visual_buffer) > self._visual_buffer_max:
                self._local_visual_buffer = self._local_visual_buffer[-self._visual_buffer_max:]
        return node

    def _frame_state_changed(self, prev_id: str, node: STNode) -> bool:
        prev = self.store.get_node(prev_id)
        if not prev:
            return False
        return prev.state_attributes != node.state_attributes

    def query_by_semantics(self, text: str, limit: int = 10,
                           modality: str = None) -> List[Tuple[STNode, float]]:
        """语义→图像（反向映射）：文本 → semantic_coordinates → 图节点检索（D-003 明确检索语义坐标）"""
        if not self._semantic_provider:
            return []
        q_sc = self._semantic_provider.to_semantic_coordinates(text)
        nodes = self.store.query_nodes(limit=500)
        scored = []
        for n in nodes:
            if modality and n.modality != modality:
                continue
            sc = getattr(n, "semantic_coordinates", {}) or {}
            sim = self._semantic_provider.similarity_coordinates(q_sc, sc) if sc else 0.0
            if n.entity_id and self.entity_registry:
                ent = self.entity_registry.get_entity(n.entity_id)
                if ent and (ent["name"] in text or any(a in text for a in ent.get("aliases", []))):
                    sim = min(1.0, sim + 0.5)
            if sim > 0.0:
                scored.append((n, sim))
        scored.sort(key=lambda x: -x[1])
        self._note_reuse([n.id for n, _ in scored[:limit]])
        return scored[:limit]

    #: `prepare_shared_sync` / `apply_shared_sync` 载荷的字段清单（单一真源）——
    #: 覆盖 `STNode` 重建所需全部字段（含 `condition_space`/`tags`/`confidence`/
    #: `state_attributes`），故载荷**可无损回放**（不截断 content、不丢层标识）。
    SHARED_SYNC_FIELDS = ("id", "content", "modality", "spatial_coordinates",
                          "temporal_coordinate", "condition_space", "importance",
                          "confidence", "layer", "tags", "semantic_coordinates",
                          "state_attributes", "entity_id")

    def _export_pending_promotions(self) -> List[Dict]:
        """SUB 侧「待晋升副本」**单列分组**导出（#165 路 B 第 5 条 · 上行主方向）。

        章程：上行（SUB → PRIMARY）是本通道的**主方向**。SUB 侧带
        `SUB_PENDING_SYNC_TAG`（`pending_sync`）/ `external_anchor` 标签的副本落在
        **本地知识层**（`_write_structure_record` / `register_external_anchor` 的 SUB
        腿），而旧实现只导出共享层 ⇒ SUB 载荷恒 0 条，这些副本永远到不了父节点
        （issue #165 标题主诉「SUB 上『待父节点同步』的终裁记录与外部锚点永远到不了
        结构层」）。本组把它们**单列分组**导出。

        为什么**不**混进 `sync_payload`：接收端 `apply_shared_sync` 只收共享层条目，
        本地层条目一律 `rejected`（章程：本地层实例独立）⇒ 混进去会被整批拒收，等于
        白推。故 `sync_payload` 的既有语义（**只含共享层**）**一字不动**，本组另立
        顶层键。

        分组口径：按**触发标签**分组（`pending_sync` 在前、`external_anchor` 在后，
        组内按 `id` 升序 ⇒ 输出确定）。同一条副本同时带两个标签时**两组都出现**
        （不是二选一——两组语义不同：一个是「待父节点同步」的通用标记，一个是
        「外部校准输入」的来源标记）；每条以 `pending_group` 键标明来自哪一组，
        **按 `id` 去重是消费侧的事**（本组如实保留两组的完整视图）。

        边界（**未做**，如实声明）：本组**只有导出腿**——消费侧（PRIMARY 收到后走
        `propose → verify → adjudicate` 三段式）**尚未接线**，属独立落地项；本件不替
        它造第二条写入路径。载荷条目**不带**内容摘要哈希（#165 路 B 第 3 条
        **暂缓落地**，见 `apply_shared_sync` docstring）。
        """
        cols = ", ".join(self.SHARED_SYNC_FIELDS)
        shared = sorted(layer.value for layer in self.store.SHARED_LAYERS)
        placeholders = ",".join("?" for _ in shared)
        out: List[Dict] = []
        for tag in (self.store.SUB_PENDING_SYNC_TAG, "external_anchor"):
            c = self.store.conn.cursor()
            # 标签列是 JSON 文本 ⇒ 按**带引号的整词**匹配（`"pending_sync"`），
            # 不会把 `pending_sync_extra` 这类前缀相同的标签误收（B3 的判别力）。
            c.execute(f"SELECT {cols} FROM nodes"
                      f" WHERE layer NOT IN ({placeholders}) AND tags LIKE ?"
                      f" ORDER BY id",
                      shared + ['%"' + tag + '"%'])
            for r in c.fetchall():
                item = dict(zip(self.SHARED_SYNC_FIELDS, r))
                item["spatial_coordinates"] = json.loads(
                    item["spatial_coordinates"] or "{}")
                item["semantic_coordinates"] = json.loads(
                    item["semantic_coordinates"] or "{}")
                item["state_attributes"] = json.loads(
                    item["state_attributes"] or "{}")
                item["condition_space"] = json.loads(
                    item["condition_space"] or "{}")
                item["tags"] = json.loads(item["tags"] or "[]")
                item["pending_group"] = tag
                out.append(item)
        return out

    def prepare_shared_sync(self) -> Dict:
        """D-005 蜂群分层同步：导出**共享层**（锚点层 + 结构层）载荷供跨实例同步。

        ★#165（本件）修正两处：
        ① **方向改正**——旧实现 `SELECT ... WHERE layer='knowledge'` 导出的是
           **本地知识层**，而分层章程（`LayeredStore` 类头 docstring：
           「共享层（锚点层、结构层）：跨实例同步，不可遗忘；本地层（知识层、
           情境层、自我层）：实例独立」）要求同步的是**共享层**。实测旧读数
           `payload layers: {'knowledge': 1}`，而共享层里确有 structure=1/anchor=1
           **一条都没进载荷**。现改为导出 `layer IN ('anchor','structure')`
           （即 `LayeredStore.SHARED_LAYERS`）。
        ② **可无损回放**——旧载荷只有 7 个字段且 `content` 被截到 100 字符、
           无 `layer`/`confidence`/`condition_space`/`tags`/`state_attributes` ⇒
           接收端无法原样重建节点。现按 `SHARED_SYNC_FIELDS` 导出**全字段**
           （含完整 content），接收端 `apply_shared_sync` 可无损重建。

        ★**#165 裁定路 B（本件只落 §4 + §5a）**：
        ③ **§4 来源实例标识**——顶层加 `source_instance` 键，取本实例的
           `self.self_model.identity`（★如实标注：裁定原文写「取 `self.identity`」，
           但本引擎**没有** `identity` 属性——`identity` 是
           `SpacetimeMemoryEngine.__init__` 的形参，唯一落点是 `self.self_model.identity`；
           本线按唯一真源取值）。**无值时写显式 `None`**：键**必须在场**——
           缺席是歧义源（消费侧分不清「旧版载荷」与「新版但没填」），`None` 才是
           可机器判读的「我们知道这里没填」。
        ④ **§5a 上行待晋升副本**——顶层加 `pending_promotions` 键
           （见 `_export_pending_promotions`）。`sync_payload` 的语义（只含共享层）
           **一字不动**。
        ⚠**未落地（如实标注 · 与裁定帖有出入）**：路 B 第 1 条（新增 id 一律经
        `propose → verify → adjudicate_promotion`、SQL 侧按 id 只 `INSERT OR IGNORE`）、
        第 2 条（`overwrite=True` 必须过设计者密钥闸、无密钥 ⇒ `rejected`）、
        第 3 条（载荷每条带内容摘要哈希、缺哈希即拒）**全部暂缓**——编排侧 2026-10-17
        裁定：这三条与在-main 的既有守卫正面冲突（`tests/test_issue463_shared_sync_identity_guard.py`
        的 C/D 组要求无密钥 `overwrite=True` 即 `applied`；`tests/test_issue771_778_import_atomicity_guard.py::test_g6c`
        要求新 id 条目无密钥即 `applied`；两件守卫属对外面/设计者面，改它们须先回帖
        征询），且本件自己的 P3/P6 也要求「首次 apply 即落库」。故本接收端**仍是共享层
        的第三条写路径**（伪造载荷仍可直写共享层）——**这一面未收口**。
        判据来源：`LayeredStore` 类头 docstring（共享层/本地层划分）＋
        `LayeredStore.SHARED_LAYERS`（core.py 内同款语义集）＋ 公开 issue #165
        ＋ 本端 2026-10-16 对 #165 的裁定帖（`<!-- RULING-165 -->`）路 B 第 4/5 条。
        边界（削掉的判别力/未做项）：①只导出共享层——知识/情境/自我层**不进**
           `sync_payload`（章程称其「实例独立」，导出属越界）；既有测试无一依赖旧载荷里的
           knowledge 条目（grep `prepare_shared_sync|sync_payload` 在 `tests/` 0 命中）。
        ②接收端 `apply_shared_sync` 为**尽力而为**实现，未接入任何自动同步编排
           （无后台调度/网络传输）——「通道尚未建成」的其余部分**未做**，如实声明。
        ③本件**不动** `add_perception` 的 M5 去重吞第二次终裁记录（属另一件）。
        ④`pending_promotions` **只有导出腿**，消费侧**尚未接线**（见其 docstring）。
        ⑤载荷**不带边**、不带 `created_at` / `access_count`（issue 点名项）：裁定帖
           「存疑」节把它列为**未定死**的设计细节（边的导出会连带 `source_evidence`
           与验证态，面比节点大）⇒ 本件**未做**，如实声明。
        """
        c = self.store.conn.cursor()
        shared = sorted(layer.value for layer in self.store.SHARED_LAYERS)
        placeholders = ",".join("?" for _ in shared)
        cols = ", ".join(self.SHARED_SYNC_FIELDS)
        c.execute(f"SELECT {cols} FROM nodes WHERE layer IN ({placeholders})",
                  shared)
        payload = []
        for r in c.fetchall():
            item = dict(zip(self.SHARED_SYNC_FIELDS, r))
            item["spatial_coordinates"] = json.loads(
                item["spatial_coordinates"] or "{}")
            item["semantic_coordinates"] = json.loads(
                item["semantic_coordinates"] or "{}")
            item["state_attributes"] = json.loads(
                item["state_attributes"] or "{}")
            item["condition_space"] = json.loads(item["condition_space"] or "{}")
            item["tags"] = json.loads(item["tags"] or "[]")
            payload.append(item)
        return {"sync_payload": payload,
                # ★#165 路 B §5a：SUB 侧待晋升副本（单列分组；PRIMARY ⇒ 空表）
                "pending_promotions": self._export_pending_promotions(),
                # ★#165 路 B §4：来源实例标识（**键必在场**；无值 ⇒ 显式 None）
                "source_instance": getattr(self.self_model, "identity", None),
                "local_buffer_count": len(self._local_visual_buffer),
                "layers": shared,
                "note": "原始视觉数据不进共享层（盲区60/61）；仅共享层（锚点/结构）跨实例同步"}

    # ──────────── #463：共享层同步的**判同面**（唯一真源 = `SHARED_SYNC_FIELDS`）────────────
    #
    # 为什么单独成段：**判同口径**与**导出字段面**必须同源，否则同一份契约的两个落点
    # 各自演化 —— #165 把导出改成 `SHARED_SYNC_FIELDS` 全字段，接收端却仍只比
    # `layer`+`content` 两列 ⇒ 载荷带了全字段、判同只看两列 ⇒ 内容以外的字段变化
    # 被吞掉且回报 `identical`（#463 的静默分叉）。
    # 本段只做**归一 + 比对**：不引入任何新语义（不合并策略、不裁决、不降级字段）。

    @classmethod
    def _shared_canon_json(cls, value, empty: str = "{}"):
        """JSON 列 → **规范文本**（带类型标记；**数值按数值归一**）；归一不了回 `None`。

        取 `None` 作"不可比"标记是安全的：本函数**成功时恒返回 `str`**，故 `None`
        不可能等于任何合法规范文本。方向保守 —— 不可比 ⇒ 调用方记为差异
        （宁可报冲突且可解释，不可报 `identical` 掩盖分叉）。
        空值回落 `empty` 与 `prepare_shared_sync` 的 `json.loads(col or "{}")`
        **逐字对齐**，这样"库里的 TEXT"与"载荷里的 dict"两侧同源。
        序列化交给 `_shared_canon_value`（逐字见那里对"数值归一"与"方向保证"的说明）。
        """
        if isinstance(value, str):
            try:
                value = json.loads(value or empty)
            except (TypeError, ValueError):
                return None
        elif value is None:
            try:
                value = json.loads(empty)
            except (TypeError, ValueError):
                return None
        try:
            return cls._shared_canon_value(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _shared_canon_value(cls, v) -> str:
        """#615 F4：递归生成 JSON 值的规范文本 —— **带类型标记** + **数值按数值归一**。

        为什么不再用 `json.dumps(..., sort_keys=True)`：JSON 只有一个 number 类型，
        但 Python 侧 `int` 与 `float` 是**两个**类型，落成文本即分叉
        （`{"n": 0}` vs `{"n": 0.0}`、`{"n": 1}` vs `{"n": 1.0}`）⇒
        **数值相等的两条记录被判成差异**（假冲突）。本函数把整值浮点折成同值整数十进制，
        使其与 `int` 同形。

        方向保证（**只少报假冲突，绝不变成少报真差异**）：
          · 折叠**仅当** `v.is_integer()`，且折叠目标 `int(v)` 与该浮点**数值相等**
            （Python 的 `int == float` 是精确比较）⇒ 折叠不会把**不等**的值并到一起：
              `10**300`（int）与 `1e300`（float）数值本就不等，折叠后文本仍不同 ⇒ 照报冲突；
              `2**53+1`（int）与 `2.0**53`（float）同理。
          · 非数值一律先打**类型标记**：字符串 `"0"` → `s:"0"` 与数值 `0` → `n:0` 仍判不同；
            布尔 `true` 与数值 `1`、`null` 与数值 `0`、`[]` 与 `{}` 亦判不同 ⇒
            **非数值面语义与改前一致**。
          · `NaN` 自等（与 `json.dumps` 的文本口径一致）⇒ 幂等重放不因此假冲突。
          · 键按 `str(k)` 排序 ⇒ 与 `sort_keys=True` 同序，且不吃混合键类型。
        """
        if v is None:
            return "null"
        if isinstance(v, bool):                        # bool 是 int 子类，必须先判
            return "true" if v else "false"
        if isinstance(v, int):
            return "n:" + str(v)
        if isinstance(v, float):
            if v != v:
                return "n:nan"                         # NaN 自等
            if v == float("inf"):
                return "n:inf"
            if v == float("-inf"):
                return "n:-inf"
            if v.is_integer():
                return "n:" + str(int(v))              # 0.0 → 0；1e2 → 100
            return "n:" + repr(v)
        if isinstance(v, str):
            return "s:" + json.dumps(v, ensure_ascii=False)
        if isinstance(v, (list, tuple)):
            return "[" + ",".join(cls._shared_canon_value(x) for x in v) + "]"
        if isinstance(v, dict):
            return "{" + ",".join(
                json.dumps(str(k), ensure_ascii=False) + ":"
                + cls._shared_canon_value(v[k])
                for k in sorted(v, key=lambda k: str(k))) + "}"
        return "o:" + repr(v)

    @classmethod
    def _shared_field_key(cls, field: str, value):
        """#463：某一声明字段的**可比形态**。返回 `(key, ok)`；`ok=False` ⇒ 必然记为差异。

        规范化口径（**判同专用** —— 不改任何写入口径/落库形态）：
          · `tags`                     顺序无关（`sorted`，重复保留）：标签是集合语义
          · `spatial_coordinates` / `semantic_coordinates` / `state_attributes`
            / `condition_space`        → **规范文本**（类型标记 + **数值按数值归一**，
                                        见 `_shared_canon_value`）：dict 与库里 TEXT 同源，
                                        且 `0` / `0.0` 这类同值异型**不再假冲突**（#615 F4）
          · `importance` / `confidence` → `_finite_clamp(v, 0, 1, default=0.5)`
                                       （与 `STNode` 声明的字段默认值同口径，非有限值
                                        归一到 0.5 而非随类型偶然取 0.0/1.0）
          · `layer`                    → `MemoryLayer(v).value`
          · `temporal_coordinate`      → `None` 原样；否则 `float`
          · 其余（`id`/`content`/`modality`/`entity_id`）→ 原文比较（`None` 原样）
        """
        if field == "tags":
            tags = value
            if isinstance(tags, str):
                try:
                    tags = json.loads(tags or "[]")
                except (TypeError, ValueError):
                    return None, False
            if tags is None:
                tags = []
            if not isinstance(tags, (list, tuple)):
                return None, False
            return sorted(str(t) for t in tags), True
        if field in ("spatial_coordinates", "semantic_coordinates",
                     "state_attributes", "condition_space"):
            key = cls._shared_canon_json(value)
            return key, key is not None
        if field in ("importance", "confidence"):
            return _finite_clamp(value, 0.0, 1.0, default=0.5), True
        if field == "layer":
            try:
                return MemoryLayer(value).value, True
            except (TypeError, ValueError):
                return None, False
        if field == "temporal_coordinate":
            if value is None:
                return None, True
            try:
                return float(value), True
            except (TypeError, ValueError):
                return None, False
        return (value if value is None else str(value)), True

    @staticmethod
    def _shared_value_summary(value, limit: int = 80) -> str:
        """差异记录里的**值摘要**（截断；保证 JSON 可序列化，便于跨进程回传）。"""
        try:
            s = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        except (TypeError, ValueError):
            s = repr(value)
        return s if len(s) <= limit else s[:limit] + "…"

    def _shared_sync_existing_row(self, nid):
        """取接收侧既有节点的**声明字段面**（一条 `SELECT`，列 = `SHARED_SYNC_FIELDS`）。

        ⇒ 与**写入字段面同口径**（不是 `SELECT *`：不取 `access_count`/`created_at`
        这类不参与同步的列，也不随 `nodes` 表加列而漂移），且每条目**只发一条**查询。
        `fetchone()` 为 `None` ⇔ 该 id 不存在（与旧 `existing is None` 同义）。
        """
        cols = ", ".join(self.SHARED_SYNC_FIELDS)
        c = self.store.conn.cursor()
        c.execute(f"SELECT {cols} FROM nodes WHERE id=?", (nid,))
        row = c.fetchone()
        if row is None:
            return None
        return dict(zip(self.SHARED_SYNC_FIELDS, row))

    def _shared_sync_field_diffs(self, existing: Dict, incoming: Dict) -> List[Dict]:
        """逐声明字段比两侧 → 差异清单。

        **空清单 ⇔ 逐字段全同 ⇔ 才允许报 `identical`**；否则每条给出差异**字段名**
        与两侧值摘要（冲突必须可解释 —— 章程）。

        ★**两条保守取舍（已知会"吞"掉差异，如实标注 · 不改方向）**：
          ① `tags` 走**集合语义**（`sorted`，顺序无关）⇒ 两侧标签**顺序不同**判 `identical`。
             后果：接收端 `tags` 的**顺序不会被重建**（`overwrite=True` 亦不重排）。
             ⇒ `apply_shared_sync` docstring 里"逐字段一致"一句**不含顺序维度**，
               这是刻意的（标签在 `LayeredStore` 侧按 `get_nodes_by_tag` / `tags LIKE ?`
               检索，顺序无语义）。
          ② `importance` / `confidence` 经 `_finite_clamp(v, 0, 1)` **归一后**比较 ⇒
             **越界值**（如源端 `2.0`、接收端 `1.0`）归一后同值，判 `identical` 且不落库。
             越界值本属违约输入（写入口 `add_node` 已钳制），此处不再二次报警。
          两条都是"宁可少报冲突"的方向；反向的**假冲突**（数值等价表示被判差异）
          已在 #615 F4 修掉（见 `_shared_canon_value`）——两者方向相反，别混。
        """
        diffs = []
        for f in self.SHARED_SYNC_FIELDS:
            ke, oke = self._shared_field_key(f, existing.get(f))
            ki, oki = self._shared_field_key(f, incoming.get(f))
            if oke and oki and ke == ki:
                continue
            diffs.append({"field": f,
                          "existing": self._shared_value_summary(existing.get(f)),
                          "incoming": self._shared_value_summary(incoming.get(f))})
        return diffs

    def apply_shared_sync(self, payload, overwrite: bool = False) -> Dict:
        """接收端（#165 · 尽力而为）：把 `prepare_shared_sync` 的载荷按 layer 无损重建。

        - 逐条按 `SHARED_SYNC_FIELDS` 重建 `STNode` 并写库（含 tags/
          condition_space/confidence/state_attributes）。
        - **仅接受共享层**（`LayeredStore.SHARED_LAYERS`）条目；载荷里出现本地层
          ⇒ 记入 `rejected`，**不落库**（章程：本地层实例独立，越界）。
        - **冲突显式报告（#463 收窄判据）**：接收侧已有同 id 节点时——按
          `SHARED_SYNC_FIELDS` **逐声明字段**比对（`tags` 顺序无关；JSON 列按规范文本；
          `importance`/`confidence` 按 `STNode` 声明的默认值归一）。**逐字段全同**才记入
          `identical`（幂等重放，不重写）；**任一字段不同** ⇒ 记入 `conflicts`（每条带
          差异**字段名** `fields` 与两侧值摘要 `diff`）并**默认不覆盖**（`overwrite=False`）；
          仅当显式 `overwrite=True` 才落库，且仍留在 `conflicts` 里（**绝不静默覆盖**
          ——章程：共享层跨实例一致，冲突须可解释）。
          ⚠ 收窄前（#165 遗留）判据只有 `layer`+`content` 两列 ⇒ 改了 `tags`/
          `confidence`/`importance`/`state_attributes` 的载荷被判成 `identical` 且
          **不落库** ⇒ 元数据静默分叉（正是 issue #463 报的症状）。
          取数与写入**同字段面**：一条 `SELECT`（列 = `SHARED_SYNC_FIELDS`），非 `SELECT *`、
          不随条目数变 N+1。
          ⚠ **"逐字段一致"的边界（两条保守取舍，详见 `_shared_sync_field_diffs`）**：
            ① `tags` 是**集合语义** —— 两侧标签顺序不同仍判 `identical`，且接收端
               **不重建顺序**（顺序在检索面无语义）；
            ② `importance`/`confidence` 经 `_finite_clamp` **归一后**比较 —— 越界值
               （源端 `2.0`/接收端 `1.0`）归一后同值，判 `identical` 且不落库。
            反向的**假冲突**（`{"n":0}` vs `{"n":0.0}` 这类数值等价表示）已修
            （#615 F4）：数值按**数值**比较，非数值面语义不变。
        - 解析失败（缺字段/坏 JSON）的条目记入 `failed`，不毒化整批。

        边界（未做，如实声明）：①**未接入**任何自动同步编排（无网络传输、无后台
        调度、无冲突裁决流程）——本方法是「载荷可回放」的接收原语，完整同步通道
        **尚未建成**；②`payload` 既可传 `prepare_shared_sync()` 的返回 dict，也可
        直接传其 `sync_payload` 列表。判据来源：`LayeredStore` 类头 docstring
        （共享层跨实例一致/本地层实例独立）＋ 公开 issue #165。

        #771 ④（**整批原子**）：旧实现逐条 `store.add_node`（各自一条事务）⇒ 载荷
        里混入**非 str content** 的条目时，前面合法条目**已落库**，坏条目在
        `add_node` 的 #588/#594 闸处抛 `TypeError` 逃逸（本方法只捕
        `PermissionError`）⇒ 「部分落库 + 未捕获异常」。另：`payload=None` 静默返回
        空结果（调用方分不清「载荷为空」与「形态错」），`payload` 为字符串/整数时
        分别以 `AttributeError` / `TypeError` 逃逸。
        修法（最小 · 三条同向）：
          ① **载荷形态前置判**：`None` / `str` / 非可迭代 ⇒ `TypeError`（可诊断
             文案，不再静默返回空结果）；
          ② **content 类型前置判**（与 `add_node` 的 #588/#594 **同源**判据）：
             载荷任一 dict 条目带 `content` 键且非 str ⇒ 整批 `TypeError`、**零落库**
             （在写任何一条之前判完，不靠回滚兜底）；
          ③ **整批收进一条事务**（`self.store._tx()`；已在调用方事务中则自动降为
             SAVEPOINT）⇒ 循环中途任何异常都回滚整批，不再出现「合法条目已落库」。
        边界（如实标注 · 与既有契约的分界）：①**结构面**解析失败（缺字段 / 坏
        JSON / 非法 layer）仍走 #165 既有契约 —— 逐条记入 `failed` / `rejected`、
        **不**中断整批（本件不改该口径）；本件收口的是**内容类型面**（非 str
        content）与**载荷形态面**，这两面取 fail-closed 整批拒绝。②`overwrite=True`
        的覆盖语义、冲突上报形状、`identical` 判据（#463/#615）逐字不变。判据来源：
        公开 issue #771 ④（「捕获范围改为逐条失败收集…并在整批提交前做类型预校验，
        避免半截落库」）＋ `add_node` 的 #588/#594 content 类型闸（单一真源）。

        ★**#165 裁定路 B §4（本件落地面）· 落库留痕**：本方法在**真的发生落库**时
        （`applied` 非空）向 `action_logs` 追加一行
        （`action_type='shared_sync_applied'`，见 `_audit_shared_sync_apply`），
        使「这条是谁推上来的 / 推了几条 / 是不是覆盖」可查——旧实现零留痕（本线
        实测 `shared_sync*` 行数 0）。**纯旁路**：不新增/不改任何返回键，不改任何
        既有判定语义；全 `identical` 的幂等重放（零落库）**不**新增审计行。
        ⚠**未落地（如实标注 · 与裁定帖有出入）**：路 B 第 1 条（新增 id 一律经
        `propose → verify → adjudicate_promotion`、按 id 只 `INSERT OR IGNORE`）、
        第 2 条（`overwrite=True` 须过设计者密钥闸、无密钥 ⇒ `rejected`）、
        第 3 条（载荷每条带内容摘要哈希、缺哈希即拒）**全部暂缓**——编排侧
        2026-10-17 裁定：与在-main 的既有守卫正面冲突
        （`tests/test_issue463_shared_sync_identity_guard.py` C/D 组要求无密钥
        `overwrite=True` 即 `applied`；`tests/test_issue771_778_import_atomicity_guard.py::test_g6c`
        要求新 id 条目无密钥即 `applied`；两件属对外面/设计者面，改它们须先回帖征询），
        且本件自己的 P3/P6 也要求「首次 apply 即落库」。⇒ **本接收端仍是共享层的
        第三条写路径，伪造载荷仍可直写共享层——这一面未收口**（本线实测：
        伪造新 id 锚点条目 `applied=['forged_new_anchor_147A']`、落库 `('anchor', ...)`）。
        """
        items = payload.get("sync_payload") if isinstance(payload, dict) else payload
        if items is None:
            raise TypeError(
                "#771 ④ apply_shared_sync 载荷为空：`payload=None` 或 dict 缺 "
                "`sync_payload` 键——旧实现静默返回空结果，调用方无法区分"
                "「载荷为空」与「载荷形态错」（现 fail-closed）")
        if isinstance(items, (str, bytes, bytearray)):
            raise TypeError(
                "#771 ④ apply_shared_sync 载荷形态非法：期望 list（或 "
                "`{'sync_payload': [...]}`），收到 %s——旧实现以 AttributeError 逃逸"
                % type(items).__name__)
        try:
            items = list(items)
        except TypeError:
            raise TypeError(
                "#771 ④ apply_shared_sync 载荷形态非法：期望 list（或 "
                "`{'sync_payload': [...]}`），收到不可迭代的 %s" % type(items).__name__)
        # ② content 类型前置判（写之前判完 ⇒ 整批零落库，不靠回滚兜底）。
        _bad_content = [(it.get("id"), type(it.get("content")).__name__)
                        for it in items
                        if isinstance(it, dict) and "content" in it
                        and not isinstance(it.get("content"), str)]
        if _bad_content:
            raise TypeError(
                "#771 ④ apply_shared_sync 整批拒绝：载荷含非 str content 条目 %d 条"
                "（例：id=%r 类型=%s）——本方法整批原子写入，任一条目非法即零落库"
                "（与 add_node 的 #588/#594 同源判据）"
                % (len(_bad_content), _bad_content[0][0], _bad_content[0][1]))
        shared = self.store.SHARED_LAYERS
        applied, identical, conflicts, rejected, failed = [], [], [], [], []
        # ③ 整批一条事务（#771 ④）：中途任何异常 ⇒ 整批回滚（不再半截落库）。
        # 复用 `LayeredStore._tx` 的嵌套感知（已在事务中则只开 SAVEPOINT，不越权
        # 提交调用方事务）；`add_node` 在事务内自动降为 SAVEPOINT，提交权留在本层。
        with self.store._tx():
            for item in items:
                nid = (item or {}).get("id")
                try:
                    layer = MemoryLayer(item["layer"])
                    cs = ConditionSpace.from_json(json.dumps(item["condition_space"]))
                    node = STNode(
                        id=nid, content=item["content"], modality=item["modality"],
                        spatial_coordinates=item.get("spatial_coordinates") or {},
                        temporal_coordinate=item["temporal_coordinate"],
                        condition_space=cs,
                        importance=item.get("importance", 0.5),
                        confidence=item.get("confidence", 0.5),
                        layer=layer,
                        tags=list(item.get("tags") or []),
                        semantic_coordinates=item.get("semantic_coordinates") or {},
                        state_attributes=item.get("state_attributes") or {},
                        entity_id=item.get("entity_id"))
                except (KeyError, TypeError, ValueError) as exc:
                    failed.append({"id": nid, "reason": "%s: %s"
                                   % (type(exc).__name__, exc)})
                    continue
                if layer not in shared:
                    rejected.append({"id": nid, "layer": layer.value,
                                     "reason": "本地层不入共享层（章程：实例独立）"})
                    continue
                # ★#463（本笔）：判同**按声明字段面**逐字段比对 —— 不再只比 `layer`+`content`。
                # 旧口径只比两列 ⇒ 源端改了 `tags`/`confidence`/`importance`/
                # `state_attributes` 等**内容以外**的字段后重放，接收端判为"完全一致"、
                # 进 `identical` 且**不落库** ⇒ 元数据变化被吞掉却回报成功，两实例
                # **静默分叉**（issue #463 的最小复现即此）。
                # 取数：一条 `SELECT`，列 = `SHARED_SYNC_FIELDS`（与 `prepare_shared_sync`
                # 同一常量 ⇒ 与写入字段同口径；不是 `SELECT *`，也不随条目数变 N+1）。
                _erow = self._shared_sync_existing_row(nid)
                if _erow is not None:
                    _xl = MemoryLayer(_erow["layer"])
                    _diffs = self._shared_sync_field_diffs(_erow, item)
                    if not _diffs:
                        identical.append(nid)
                        continue
                    conflicts.append({"id": nid, "incoming_layer": layer.value,
                                      "existing_layer": _xl.value,
                                      "fields": [d["field"] for d in _diffs],
                                      "diff": _diffs})
                    if not overwrite:
                        continue
                try:
                    self.store.add_node(node)
                except PermissionError as exc:
                    # 共享层仅 PRIMARY 可写（章程）⇒ SUB 接收端写共享层被拒，
                    # 如实记入 failed，不让整批中断。
                    failed.append({"id": nid, "reason": "PermissionError: %s" % exc})
                    continue
                applied.append(nid)
            # ★#165 路 B §4（本件落地面）：**发生落库才留痕**（零落库 ⇒ 零审计，
            # 见下方 A3：幂等重放全是 `identical`、不进这里 ⇒ 不产生噪声行）。
            # 纯旁路：不改任何返回键、不改任何既有语义（判定结果已在上面算完）。
            if applied:
                self._audit_shared_sync_apply(
                    source_instance=(payload.get("source_instance")
                                     if isinstance(payload, dict) else None),
                    applied=applied, overwrite=overwrite)
        return {"applied": applied, "identical": identical,
                "conflicts": conflicts, "rejected": rejected, "failed": failed,
                "overwritten": overwrite}

    def _audit_shared_sync_apply(self, source_instance, applied: List[str],
                                 overwrite: bool) -> None:
        """共享层同步**落库**的 `action_logs` 留痕（#165 路 B §4 · 单一落点）。

        口径：与 `set_escalation_enabled` 同族——同步通道是**治理写入**（写的是
        不可遗忘、不可删的共享层），故「这条是谁推上来的、推了几条、是不是覆盖」
        必须可查（旧实现零留痕：本线实测 `SELECT COUNT(*) FROM action_logs
        WHERE action_type LIKE 'shared_sync%'` ⇒ 0）。

        字段面：`summary` 记条数；`node_ids` 记本次落库的 id 列表；
        `outcome` 记 `source_instance`（来源实例标识，**自述**）/ `applied_count` /
        `overwrite`。来源标识取载荷顶层 `source_instance`（`prepare_shared_sync`
        的导出键）；载荷**不带**该键（旧版载荷 / 直接传 list）⇒ 记显式 `None`
        （「我们知道这里没填」，不是缺席）。

        边界（如实标注）：①来源是**实例自述**、**不是**身份认证——来源可信属路 C
        （签名/凭据）面，**未落地**；②审计是**旁路**，其失败不得改判调用方的
        判定结果（故此处捕获并 `warnings.warn` 上报，不向外抛）；③本方法**只**在
        真的发生落库时被调用（`applied` 非空）⇒ 全 `identical` 的幂等重放
        **不**新增审计行。
        """
        try:
            self.store.log_action(
                "shared_sync_applied",
                "共享层同步落库：%d 条（overwrite=%s）" % (len(applied),
                                                           bool(overwrite)),
                node_ids=list(applied),
                outcome={"source_instance": source_instance,
                         "applied_count": len(applied),
                         "overwrite": bool(overwrite)})
        except Exception as exc:          # 审计旁路：不得改判调用方结果
            warnings.warn("shared_sync_audit_failed: %s" % exc)

    # ==================== v1.6 语义空间与备份（M11/M13 · REV1） ====================

    RECURSION_LIMIT = 3  # DEVIATION-001：编译器自编译递归深度限制（与 3.12 节一致）

    def set_semantic_provider(self, provider):
        """注入自定义语义提供者（duck-typed：to_coordinates/similarity/resolve_concepts/introspect）"""
        self._semantic_provider = provider

    def get_semantic_provider(self):
        return self._semantic_provider

    def resolve_concepts(self, text: str) -> List[str]:
        """名实校验 API（REV1）：文本 → 协议概念（以名举实）"""
        if not self._semantic_provider:
            return []
        return self._semantic_provider.resolve_concepts(text)

    def compute_semantic_coordinates(self, text: str) -> Dict[str, float]:
        """文本 → spatial_coordinates（L1 0.6 + L2 0.2 + L3 0.2 加权）"""
        if not self._semantic_provider:
            return {}
        return self._semantic_provider.to_coordinates(text)

    @_transactional
    def annotate_semantics(self, node_id: str) -> bool:
        """为既有节点补写语义坐标（M11 迁移工具）。

        #415：旧实现把 provider 输出写进 **`spatial_coordinates`**
        （`UPDATE nodes SET spatial_coordinates=?`）——补注语义却覆盖了节点的
        真实物理位置，而 `semantic_coordinates` 仍为空（issue 实测：
        `{"x":1,"y":2,"z":3}` → `{"protocol_object":0.75}`，语义列 `{}`）。
        修法：provider 输出经 `_flat_semantic_to_nested` 转成当前语义坐标结构，
        **合并**进既有 `semantic_coordinates` 后写回该列；`spatial_coordinates`
        一字不动（真实位置保留）。
        判据来源：公开 issue #415「保留真实位置，将 provider 输出转换为当前语义
        坐标结构后写入 `semantic_coordinates`……不能仅机械替换列名」＋ 本文件
        `migrate_v17_coordinates` 的 D-003 映射先例。
        边界（削掉的判别力）：写入前**不**校验 provider 输出是否为合法语义结构
        （非 dict 由 `_flat_semantic_to_nested` 的 `dict(flat or {})` 归一为 `{}`）。

        ★本笔（治理写面「同列两套口径」收口 · P1）：写前补**角色闸**——非 PRIMARY
        且目标行属共享层（`SHARED_LAYERS`）⇒ `PermissionError`（零落库）。判据复用
        `LayeredStore.assert_node_writable_by_role` 单点（不另造第二份判据）；
        闸落在 UPDATE 之前、且在既有的「无节点 / 无 provider ⇒ False」早退之后
        ⇒ 返回值契约（`bool`）与早退语义逐位不变。未覆盖面（如实标注）：本闸只判
        「层 × 角色」，不判调用方身份真伪（同进程伪造角色属 #779 威胁模型边界）。
        """
        node = self.store.get_node(node_id)
        if not node or not self._semantic_provider:
            return False
        # ★本笔（治理写面「同列两套口径」收口 · P1）：本写点此前对
        # `nodes.semantic_coordinates` **零角色闸** ⇒ SUB 可改写 PRIMARY 建的
        # 锚点/结构层行的语义坐标（实测 `{}` → `{"protocol":{"concept":{"object":0.75}}}`，
        # PRIMARY 跨实例读回同值）；而同列经 `add_node` / `update_node_confidence`
        # 等路径受共享层角色闸约束 ⇒ 同一列两套口径。闸取 `store` 侧既有单点
        # （`layer_writable_by_role` 的写前查旧行形态，`#127` D-07），落在 UPDATE 之前。
        self.store.assert_node_writable_by_role(
            node_id, action="改写共享层节点的 semantic_coordinates")
        coords = self._semantic_provider.to_coordinates(node.content)
        merged = dict(node.semantic_coordinates or {})
        for bucket, val in _flat_semantic_to_nested(coords).items():
            if isinstance(val, dict) and isinstance(merged.get(bucket), dict):
                merged[bucket] = {**merged[bucket], **val}
            else:
                merged[bucket] = val
        c = self.store.conn.cursor()
        c.execute("UPDATE nodes SET semantic_coordinates=? WHERE id=?",
                  (_dumps_json(merged, where="semantic_coordinates"), node_id))
        self._commit_locked()
        return True

    def compiler_introspect(self) -> Dict:
        """编译器自省（DEVIATION-005）：符号覆盖范围 + 未覆盖协议概念（3.13 自省绑定）"""
        if not self._semantic_provider:
            return {"available": False, "error": self._semantic_error}
        return self._semantic_provider.introspect()

    def semantic_blindspot_report(self) -> Dict:
        """语义盲区报告（DEVIATION-005）：无法处理的输入类型标记"""
        if not self._semantic_provider:
            return {"available": False, "error": self._semantic_error}
        return self._semantic_provider.blindspot_report()

    def check_recursion_depth(self, depth: int) -> bool:
        """递归深度限制（DEVIATION-001）：编译器自编译 ≤3 层"""
        return depth <= self.RECURSION_LIMIT

    # ---- M13 备份/迁移 ----

    #: M13 导出/导入的共享表清单（顺序即导出顺序）。
    #: 前置条件（issue #16）：`entities` 由 **未随本导出面一同发布的**
    #: `entity_registry` 拥有（见 docs/intake/body-export-v0.1/README.md「未实现
    #: 扩展点名号」）——公开仓形态下该表不存在，故两条链路一律以
    #: `sqlite_master` 的实际存在性为准：缺失的表**跳过 + 显式标注**，
    #: 而不是让「灾备基础」因一个可选组件缺席而整体失败。
    #:
    #: issue #270：本清单此前漏掉 OBS-REV1 三张表（`action_logs` /
    #: `engine_meta` / `gap_history`，DDL 见 `_init_tables` 的 v1.14 段），
    #: 而导出面自称「全库备份」且**不把这漏项计入 `skipped_tables`**
    #: （它们根本不在清单里，无从被标为 skipped）⇒ 行为日志/引擎元数据/D 序列
    #: 静默不入备份、灾备恢复后观测面清零。判据来源：引擎**自建表全集**
    #: `LayeredStore._SCHEMA_TABLES`（`_init_tables` 的 DDL 清单）——备份清单须覆盖
    #: 它，否则「全库」名不副实。`sqlite_sequence`（SQLite 内建自增簿记）**不**入清单。
    #: 该不变量由 `tests/test_issue270_m13_table_coverage_guard.py` 钉住。
    #: #691 P0-b：清单此前漏掉运行时建的 `flywheel_reuse`（复用观测），
    #: 「全库备份」不覆盖它 ⇒ 重启即丢；现该表已收进 `_init_tables`
    #: （构造期即建）并补进本清单与 `_SCHEMA_TABLES`。
    #: #130（路 B）：`designer_todos` 同款——它由 `_init_tables` 构造期建出
    #: （见该表 DDL 段注释），故必须同进两处清单，否则「全库备份」不含待办、
    #: 且 `test_issue270_m13_table_coverage_guard` 的
    #: `_SCHEMA_TABLES ⊆ M13_TABLES` 不变量当场红。
    M13_TABLES = ("nodes", "edges", "blindspots", "skills", "promotion_proposals",
                  "protections", "rejected_paths", "verifier_standards",
                  "escalation_points", "entities",
                  "action_logs", "engine_meta", "gap_history", "flywheel_reuse",
                  "designer_todos", "archive_state")

    def _existing_tables(self) -> Set[str]:
        """当前库实际存在的表名（以 sqlite_master 为准，不假设 DDL 全集）。"""
        return {r[0] for r in self.store.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}

    def _table_columns(self, table: str) -> Set[str]:
        """库内该表的真实列名集合（`PRAGMA table_info`，以库内事实为准）。

        issue #125：导入侧的列名取自备份 JSON 的键、经 f-string 拼进 SQL
        （SQLite 标识符不可参数化）⇒ 非法键即「任意列直写」。本方法给出
        白名单来源；`table` 取自固定清单 `M13_TABLES`（不可控，非注入面）。
        判据来源：`docs/plans/治理与写入边界_修复设计_v0.1.md` §6 S-125-2
        「`PRAGMA table_info`」。
        """
        return {r[1] for r in self.store.conn.execute(
            f"PRAGMA table_info({table})")}

    # ---- #802：导出面（落盘 JSON）的 BiDi 控制/零宽字符剥除 ----
    # BiDi 控制字符（嵌入/覆写 U+202A-U+202E、隔离 U+2066-U+2069、方向标记
    # U+200E/U+200F）与零宽族（U+200B-U+200D / U+FEFF / U+2060）视觉不可见
    # 但会被 BiDi 渲染器（VS Code / Notepad++ / 终端 cat / web UI / 日志查看
    # 器）解释——RTLO（U+202E）使后续文本**反向显示**：`file\u202etxt` 落盘后
    # 显示成 `txt.exe`，属视觉欺骗/钓鱼向量（CWE-176）。`json.dump` 只转义
    # ASCII 控制符（< U+0020），非 ASCII 的 U+202E 原样落盘 ⇒ 导出 JSON 在
    # 下游渲染面存在反转文本（spoofing 面）。
    # 修法（最小 · 仅导出序列化面）：`export_all` / `export_trajectory`
    # **写出前**对整棵导出树（表行的所有字符串值与键）剥除上述控制符。判据
    # 来源：公开 issue #802 修复建议 1/2（「剥离或转义 BiDi 控制字符」「剥离或
    # 标记零宽字符」）——两可之中取**剥除**：与 #803/PR #807 的 world 侧
    # `_strip_zero_width`（判同键剥除）同款风格，且保证任意渲染器下「落盘
    # JSON 渲染无反转」不依赖渲染器是否转义。
    # 边界（如实标注）：①只剥**格式控制符**，不剥真正的 RTL 文字（阿拉伯/
    # 希伯来字母、CJK、emoji、组合附加号均不在剥除集——正常双向文本照旧）；
    # ②只动**导出序列化面**——落库正文（DB 字节）与 `import_all` 回读面不动
    # （判据只管导出面；剥除后正文长度变短，实测回读无长度断言、不炸）；
    # ③剥除是不可逆的（原字节不保留）——与「剥除」口径一致，导出面宁可丢
    # 格式控制符也不落反转文本；④`export_trajectory` 只在**落盘分支**剥除，
    # 批量模式（`output_path is None`，不落盘）返回体保持原样。
    _EXPORT_CTRL_RE = None  # 惰性编译（模块顶部不 import re，与 #803 同款）

    @staticmethod
    def _strip_export_controls(s: str) -> str:
        """#802：剥除 BiDi 控制/零宽字符——导出落盘 JSON 的渲染无反转保证。"""
        import re as _re
        cls = SpacetimeMemoryEngine
        if cls._EXPORT_CTRL_RE is None:
            # U+200E/U+200F LRM/RLM · U+202A-U+202E LRE/RLE/PDF/LRO/RTO(RTLO)
            # · U+2066-U+2069 LRI/RLI/FSI/PDI · U+200B-U+200D ZWSP/ZWNJ/ZWJ
            # · U+2060 WJ · U+FEFF BOM（零宽集与 #803 的 `_strip_zero_width`
            # 同集，本面另收 BiDi 集）。
            cls._EXPORT_CTRL_RE = _re.compile(
                "[\u200e\u200f\u202a-\u202e\u2066-\u2069\u200b-\u200d\u2060\ufeff]")
        return cls._EXPORT_CTRL_RE.sub("", s)

    @classmethod
    def _sanitize_export_obj(cls, obj):
        """#802：对导出树做深剥除——所有字符串值与键过 `_strip_export_controls`。

        面取**整棵树**而非仅 `nodes.content`：RTLO 可藏在任意人读字段（tags、
        轨迹 JSON 的键/值——`export_trajectory` 的键来自用户内容的
        `json.loads`），判据是「落盘 JSON 渲染无反转」而非「content 无反转」。
        dict/list 递归，非字符串原样返回（数值/None/字节不动）。
        """
        if isinstance(obj, str):
            return cls._strip_export_controls(obj)
        if isinstance(obj, dict):
            return {cls._strip_export_controls(k) if isinstance(k, str) else k:
                    cls._sanitize_export_obj(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [cls._sanitize_export_obj(v) for v in obj]
        return obj

    def export_all(self, output_path: str) -> Dict:
        """M13：全库导出（JSON · 6.5 摘要交换/灾备基础）

        降级口径（issue #16）：只导出**实际存在**的表；缺失的表跳过并在
        `meta.skipped_tables` 与返回值 `skipped_tables` 中显式标注。
        未安装 `entity_registry`（本仓默认形态）时 `entities` 即落入该列表，
        导出不再抛 `OperationalError: no such table: entities`。

        issue #330（数据丢失）：落盘改**原子写**——旧实现
        `with open(output_path, "w") as f: json.dump(...)` 是**先截断再流式写**，
        写中途失败（磁盘满 / 序列化异常 / 进程被杀）即把原有备份毁成空档或
        半截 JSON，而本方法正是「灾备基础」⇒ 一次失败的备份会连旧备份一起赔掉。
        修法：同目录临时文件写完 + `os.replace` 换入（同卷原子），失败清临时件。
        判据来源：经验标定（本件 #330）——口径取「临时文件 + os.replace」；
        仓内无规定导出落盘原语的理论章节，追不到更早出处。
        边界（如实标注）：`os.replace` 只保证「目标要么旧内容、要么完整新内容」，
        **不**保证掉电后目录项已落盘（未做目录 fsync）。

        issue #383（备份一致性）：所有表改在**同一个读事务**里读（本连接不在事务中
        时显式 `BEGIN`，读完 commit）⇒ 导出是某一时刻的一致快照，不再出现
        「源库 integrity OK、备份里悬挂边、import_all 报 integrity_ok=False」。
        详见下方 `own_tx` 段内注释与 `tests/test_issue383_export_snapshot_guard.py`。

        issue #698（静默覆盖）：目标路径已存在时本方法**覆盖**之——这是**既有契约**，
        非缺陷：本方法自称「灾备基础」，重做备份必须能覆盖上一份；
        `tests/test_issue330_export_all_atomic_guard.py` 的 D 用例即锁定「二次导出到
        同一路径 ⇒ 覆盖」（本件**不改**该语义、不加拒覆盖、不动默认）。
        `path` 由**调用方**给出，但**会过** `_confine_export_out_path` 根闸（#746）：
        未显式配置 `LINGSHU_EXPORT_OUT_ROOT` 时**不限制**（保持既有行为——灾备/
        迁移场景的目标照旧可在任意卷上）；显式配置后越界即 fail-closed 抛
        `ValueError`（该闸与本方法的 `#698` 面正交：只判落点，不改覆盖语义）。
        本件只把「**静默**覆盖」改为**显式上报**：覆盖与否经返回值
        `overwrote_existing`（bool）回报，不再静默。
        边界（如实标注）：`overwrote_existing` 是**判定—替换之间**的一次
        `os.path.exists` 读数，只要求「上报」，**不**提供排他创建/并发排他
        （判定与 `os.replace` 之间存在竞态窗口；本件不引入先查后建语义）。
        源库保护：当前连接的 SQLite 数据库及 WAL/SHM/journal 不可作为导出目标，
        即使位于配置的白名单根内也拒绝；普通备份覆盖语义不变。
        """
        # 旧备份仍可覆盖（#698）；源库及 WAL/SHM/journal 即使落在白名单根内也须拒绝。
        output_path = self._confine_export_out_path(output_path)
        target_path = os.path.realpath(output_path)
        self._guard_export_source_path(target_path)
        c = self.store.conn.cursor()
        # #383：整趟导出必须**共用一个读快照**。旧实现逐表各发一条
        # `SELECT * FROM <t>`，Python sqlite3 对 SELECT 不隐式开事务 ⇒ 每条语句
        # 各取一个快照，读之间无任何隔离：并发写者（自带 auto-decay 线程、情境层
        # FIFO、另一进程）在读完 nodes、开始读 edges 之间做一次
        # `delete_node + add_edge`，备份就是**跨时刻拼接**——源库
        # `verify_integrity` 为 True，备份里却有指向「备份中不存在的节点」的边，
        # `import_all` 得 `integrity_ok=False`。
        # 修法（最小 · 按 issue 建议 diff）：本连接不在事务中时显式
        # `BEGIN`（**延迟读事务**，WAL 下整趟导出共用一个快照），读完再 commit；
        # 已在事务中（如 `_transactional` 包裹的调用方）则不插手，不动其边界。
        # 判据来源：公开 issue #383「让所有表在同一个读事务里读」＋ 其给出的
        # `conn.in_transaction` 判定写法。仓内无规定导出隔离口径的理论章节，
        # 属经验标定（本件 #383）。
        # 边界（削掉的判别力）：`BEGIN` 只给**本连接**一个读快照，**不**阻止并发
        # 写者继续写源库（那是 WAL/MVCC 的正常语义，快照隔离本身即目的）；
        # 也不改 `os.replace` 落盘原子性（#330 面，未动）。
        own_tx = not self.store.conn.in_transaction
        if own_tx:
            c.execute("BEGIN")
        try:
            existing = self._existing_tables()
            skipped = [t for t in self.M13_TABLES if t not in existing]

            def rows(table):
                c.execute(f"SELECT * FROM {table}")
                return [dict(r) for r in c.fetchall()]

            data = {
                "meta": {"version": "v1.6", "exported_at": time.time(),
                         "condition_space": "全库备份 · 有损投影",
                         "skipped_tables": skipped},
            }
            for table in self.M13_TABLES:
                if table in existing:
                    data[table] = rows(table)
            # #778-F4：孤儿保护行读数取在**同一读快照内**（与上方逐表读同事务），
            # 否则「备份里有孤儿保护行」与「上报的孤儿数」可能跨时刻不一致。
            _orphan_prot = (self.store.orphan_protection_rows()
                            if "protections" in existing else [])
        finally:
            if own_tx:
                self.store.conn.commit()
        # #802：写出前对整棵导出树剥除 BiDi 控制/零宽字符（见
        # `_strip_export_controls` 处注释）——落盘 JSON 渲染无反转。计数键
        # （`exported_nodes` / `tables`）只数行数与表数，剥除不动行列数。
        data = self._sanitize_export_obj(data)
        # #498（#330 修复的副作用）：`os.replace` 把目标换成**新 inode** ⇒
        # ①新档权限取 `0o666 & ~umask`，**不继承**旧档权限——运维刻意对备份设的
        #   `0600` 被静默放宽成 `0644`（全库导出含 `sensitivity=private` 条目、
        #   自我层快照、action_logs ⇒ 同机其他用户可读，属信息泄露面）；
        # ②目标是**符号链接**时，`os.replace` 替换的是**链接本身**（链接被换成
        #   普通文件），链接指向的真实备份位置不再更新 ⇒ 静默的灾备失效。
        # 旧实现 `open(output_path, "w")` 截断写**同一个 inode**，权限与符号链接
        # 都自然保留（对照旧基准 e18562f）。
        # 修法（最小 · 本件 #498）：目标先 `realpath` 穿透符号链接，临时件建在
        # **真实目标同目录**（同卷原子性不丢）；`os.replace` 前把旧档权限位拷到
        # 临时件（`os.stat` 跟随链接 ⇒ 取真实目标的 mode；目标不存在则跳过，
        # 保持 umask 默认）。
        # 判据来源：公开 issue #498 的参考补丁（`realpath` + `chmod(tmp,
        # stat(target).st_mode & 0o7777)`）。用 `& 0o7777` 字面量而**不**在顶层
        # `import stat`：加 `import stat` 会让 `tests/test_issue156_component_
        # discovery.py` 的「未知裸名」判据 D2/D4a 变红（该件自证已锁定）。
        # 边界（如实标注）：只搬**权限位**，不搬属主/属组（`chown` 需特权，未做）；
        # 目标不存在时行为与改前逐字一致（umask 默认）；不引入目录 fsync（#330
        # 面边界不变）；返回值 `path` 仍是**调用方给的**路径（对外契约不变）。
        tmp_path = "%s.tmp-%s" % (target_path, uuid.uuid4().hex[:8])
        # #698：覆盖已存在目标 ⇒ 显式上报（不改覆盖语义，只让「静默」消失）。
        # 读数取在 `os.replace` **之前**——replace 之后目标必在，届时再查恒为 True。
        overwrote_existing = os.path.exists(output_path)
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            # #498：继承旧档权限（`os.stat` 跟随符号链接 ⇒ 取真实目标的 mode）。
            # 目标不存在（新建备份）⇒ `FileNotFoundError` ⇒ 跳过，用 umask 默认。
            try:
                os.chmod(tmp_path, os.stat(target_path).st_mode & 0o7777)
            except FileNotFoundError:
                pass
            os.replace(tmp_path, target_path)
        except BaseException:
            try:
                if os.path.exists(tmp_path):
                    # #498 反向腿：临时件可能刚被 chmod 成旧档的**只读位**——Windows
                    # 上只读档删不掉（`os.remove` ⇒ PermissionError）⇒ 失败路径会留下
                    # `.tmp-` 残件。清理前先尽力恢复可写（POSIX 上 unlink 只看目录
                    # 权限，此步无害；Windows 上是必需的）。
                    try:
                        os.chmod(tmp_path, 0o600)
                    except OSError:
                        pass
                    os.remove(tmp_path)
            except OSError:
                pass
            raise
        return {"exported_nodes": len(data.get("nodes", [])), "path": output_path,
                "tables": len(data) - 1, "skipped_tables": skipped,
                "overwrote_existing": overwrote_existing,
                # #778-F4：导出面**同时上报** `protections` 孤儿行——该表无外键声明
                # （对照 `edges` 的 #93 孤儿预筛），孤儿保护行会**随本备份继续传播**
                # （导入侧无对称预筛，见 `verify_integrity` docstring）。判据真源
                # `LayeredStore.orphan_protection_rows`。**只上报、不清理**：备份内容
                # 逐字不变（不静默丢弃保护行），孤儿行是否回收属设计取向（本件不引入
                # 回收入口）。键为**追加**（与 #698 的 `overwrote_existing` 同例），
                # 既有键与取值均不变。
                "orphan_protections": len(_orphan_prot),
                "orphan_protections_sample": _orphan_prot[:20]}

    @_transactional
    def import_all(self, input_path: str, designer_key: str = None) -> Dict:
        """M13：全库导入（恢复/迁移/6.5 合并基础）

        issue #125（写边界）：本方法**绕过 `add_node` 的角色闸**，直接
        `INSERT OR REPLACE`；且列名取自备份 JSON 的键逐字拼进 SQL ⇒ 备份可
        ①无密钥改写不可变层（anchor/structure/self）行原文；②以任意列名
        （标识符不可参数化）直写。修法（最小 · 复用既有先例）：
        ①**不可变层覆盖闸**：备份 `nodes` 行中凡 `layer` ∈ `IMMUTABLE_LAYERS`
           ⇒ 要求 `verify_designer(designer_key)`（与 `adjudicate_promotion`
           **同款** fail-closed）；无密钥即抛 `PermissionError`，**先判后写**
           （闸在写入之前，整次导入回滚、零落库）。
        ②**列名白名单**：以 `PRAGMA table_info`（`_table_columns`）取库内真实
           列集，与备份行键的**跨行并集**求交集（#652②：旧实现只取首行键集合，
           首行缺列会让后行同列真值一并被丢＝静默数据丢失；后行缺列则静默落
           NULL 且不上报）；非白名单键**丢弃并计数上报**（`discarded_keys`），
           不拼进 SQL；「行缺列」按表按列计数上报（`missing_columns`，键为列名、
           值为缺该列的行数），使原先静默的 NULL 落库可观测。
        ③**硬配套**（设计者 2026-10-09 授权按推荐执行，措辞为「不可**静默**
           篡改」）：经密钥放行的不可变层写入**显式标注**（返回值
           `immutable_rows` / `requires_review=True`）＋**强制事后复核**（该标记
           即复核触发点）＋**审计只追加**（落 `action_logs` 一行，无更新/删除
           路径）。判据来源：`docs/plans/待裁清单_v0.1.md` **C-6**（含 2026-10-09
           设计者授权与三项硬配套）/ **D-06**（需密钥放行）/ **D-09**（白名单＋
           拒绝非法键）；形状见 `docs/plans/治理与写入边界_修复设计_v0.1.md` §6。

        削掉的判别力（边界，如实标注）：闸判据是「备份 `nodes` 行的**声明层**
        命中不可变层」，非「仅将覆盖既有不可变行的那些行」⇒ 含不可变层行的
        备份即便目标库无该行（纯新增）也需密钥。取此更严口径的理由：新增
        `anchor`/`structure`/`self` 行同属「写入不可遗忘层」，与
        `add_node` 的角色闸同向；代价是「导入一份含共享层的新库」亦需配置
        密钥（属合法运维面，配置口与 `adjudicate_promotion` 同款）。

        降级口径（issue #16）：目标库不存在的表跳过并计入 `skipped_tables`；
        否则导出产物一旦含 `entities` 行，回灌同样抛 `OperationalError`。

        #721（值形状闸，与 #125 的键闸并列的第二道）：上列 ② 只管**键**白名单，
        值形状零校验 ⇒ 备份里某 TEXT(JSON) 列的值若是**容器**（嵌套 dict/list，
        例如 `context` 列 = `{"k":1}` 而非 `'{"k":1}'` 文本），绑定层抛
        `ProgrammingError: type 'dict' is not supported`（部分 sqlite 版本族报
        `IntegrityError: datatype mismatch`）⇒ **整批回滚、整表 0 落库**，既不
        「部分恢复」也报不出坏行位置。现绑定前逐行逐列归一：列声明类型为 TEXT
        ⇒ 容器值 `json.dumps` 归一为 JSON 文本（无损往返，`export_all` 写的正是
        JSON 文本），按表按列计数上报 `normalized_values`；其余列型无法容纳容器
        ⇒ **跳过该行**并计入 `skipped_rows`（`imported` 计数同步扣除），其余行
        照常落库。判据来源：公开 issue #721（「绑定前逐行逐列做值归一…或跳过该行
        并记入 discarded/failed 计数」）＋ 同处 #652② 的「新增上报键」先例。
        边界（如实标注）：只归一**容器**值；数值越界仍走 #574/#690 的钳制，
        字符串编码口径不动（#378/#494 的面）。

        #93（悬挂引用不再静默落库）：本方法所在连接已开 `PRAGMA
        foreign_keys=ON`（见 `LayeredStore._connect`），备份里指向不存在节点的边
        若照旧 `INSERT OR REPLACE` 会抛 `FOREIGN KEY constraint failed` 并**中断
        整次导入**。故本方法在导入 `edges` 前**按库内节点集合预筛**：孤儿边
        **不导入**，其条数/样本随返回值交付（`orphan_edges` /
        `orphan_edges_sample`），`integrity_ok` 一并置 False。这是「跳过并报告」
        而非「拒绝整次导入」——取此口径以保持既有返回体形状与「部分恢复」语义
        （nodes 等完好表照常导入），同时把「恢复了损坏备份」当场暴露。

        #482（本件）：上列预筛的实现曾是**list 成员判断**——`orphan_rows = [r for r
        in rows if r not in keep]` 对 `keep` 逐 dict 比较（约 E²/2 次比较，本机实测
        E=8000：3.20e7 次 / 3.1s；E=4000：8.0e6 次 / 0.8s），且与同一写事务内的批量
        `INSERT` 共处一个事务、全程持有写锁 ⇒ 共库进程等到 `busy_timeout` 后报
        `database is locked`。现改为**单趟分区**（集合成员判断 + 一次遍历同时产出
        keep/orphan）⇒ O(E)；判据与返回体逐字不变（`orphan_edges` /
        `orphan_edges_sample` / `integrity_ok` 的语义与顺序均不动）。

        #789（BOM 备份读不进）：备份文件读取改 `encoding="utf-8-sig"`——旧实现
        `encoding="utf-8"` 把带 UTF-8 BOM 的备份（Windows 记事本 / PowerShell `>`
        重定向的常见产物；#412 族语料同源）读出首字符 `\ufeff`，`json.load` 当场抛
        `JSONDecodeError: Unexpected UTF-8 BOM` ⇒ **整次导入零落库**。`utf-8-sig`
        对无 BOM 文件逐字节等价（#600 已证的同族口径），BOM 在场时剥掉。
        判据来源：issue #789 ＋ #437/#600 同族先例（BOM 使读取面把合法件当坏件）。

        返回值指标口径（**勿与 `verify_integrity()` 的 `orphan_edges` 混用**）：
        `dangling_rows` 是**按「违规行」去重**的计数——导入后跑 `PRAGMA
        foreign_key_check`，该指令对**每一条**被违反的外键约束各回一行；同一行若
        其两条外键（`source_id`、`target_id`）**都**悬挂，就回两行（rowid 相同、
        fkid 不同）。这里按 `(表名, rowid)` 去重后取 `len` ⇒ **同一条两端全悬挂
        的边只算 1**（两端各断一根的**两条**边则算 2，因为是两行）。经 #93 预筛后
        该值在正常路径恒为 0（悬挂边根本未落库），保留作白名单裁剪后仍悬挂的兜底。
        本方法另返 `orphan_edges`（int，被预筛跳过的**边条数**，按边计 1，不按端点
        累加）与 `orphan_edges_sample`（前 20 条样本）——**与** `verify_integrity()`
        的 `orphan_edges`（按边×端点累加）**同名不同义**，读者勿混用。
        另注：`dangling_rows` 覆盖**全部**声明外键的表，本方法 `orphan_edges` 与
        `verify_integrity()` 的 `orphan_edges` 都只统计 `edges` 表。

        #125 增量复现 · `escalation_points` 同 id 覆写面（本件）：
        **来源**：公开 issue #125 的第三方增量复现（Yinfeng818，
        2026-10-16T06:52:15Z），本端已发 `<!-- VERIFY-125-INC -->` 核验帖确认成立。
        **根因**：`M13_TABLES` 含 `escalation_points`（该表 DDL 首列
        `id TEXT PRIMARY KEY`），而本方法的三集合密钥判据（`immutable_rows` /
        `guarded_meta` / `id_log_rows`）**无一含该表** ⇒ 它落**通用腿**
        `INSERT OR REPLACE`：无密钥、无闸、无审计。最重形态是**同 id 覆写**——
        备份以**既有种子 id** 写入 ⇒ 整行替换 ⇒ 不止能关升级点，还能**反转触发
        语义、替换处置动作**（原 `action` 无副本），且走完 `action_logs` 增量 0。
        同表 API 写面 `add_escalation_point` 已在 #749（`a7d1596`）接
        `verify_designer` fail-closed + `log_action("escalation_add", …)` ⇒
        同一张表两个治理写入面口径矛盾。
        **本件修法（只收窄覆写面）**：该表写腿改为「既有 id 行不写、新 id 行才
        写」——库内已有同 id ⇒ **跳过**并计入返回值 `skipped_existing_rows`
        （按表计数，**显式上报、不静默丢弃**；与 `skipped_rows` / `missing_columns`
        同形：非空才出现该表键）；SQL 侧同步由 `INSERT OR REPLACE` 换为
        `INSERT OR IGNORE`（兜底备份自身含重复 id，取首条）⇒ 本表上不再存在任何
        整行替换路径。`imported["escalation_points"]` 的口径仍是**实际写入行数**
        （被跳过的既有行不计入，与 #721 的 `skipped_rows` 同步扣除同例）。
        **为什么只收窄覆写面、不同时「要求密钥」**：`export_all` 产物**必然**携带
        种子行（该表在 `M13_TABLES`、构造期即建表并播种 6 条），而「无密钥导入
        export 产物照旧通过」由 `tests/test_issue125_import_boundary_guard.py::
        test_e_knowledge_roundtrip_unaffected` 钉住（断言 `imported.nodes == 1`
        且 `requires_review is False`）⇒ 若把该表并入密钥判据，每次知识层往返都要
        密钥、该组当场转红，且打破既有承诺。
        **未收窄的那一半（遗留，如实标注）**：无密钥仍可**新增**任意
        `escalation_points` 行（新 id），注入的 `trigger`/`condition` 会参与
        `check_escalation` 终裁匹配。这一半是**设计取舍**（要动上述既有契约），
        属决策面，本件**不动**——`imported["escalation_points"]` 对新 id 行的口径
        与修前逐字一致。守卫：
        `tests/test_issue125_escalation_import_overwrite_guard.py`（含 `--mutate`
        变异自证：改回 `INSERT OR REPLACE` ⇒ 恰好点红断言 A）。

        #771 ②③（**同 id 换层闸 ＋ content 类型闸**，两闸均「先判后写、零落库」）：
        ② 旧实现的换层判据（`immutable_rows` / `relayer_rows`）都要求「任一侧属
        IMMUTABLE」⇒ **本地层之间**的换层（`context` → `knowledge`）不进任何闸，
        备份可无密钥把既有情境层节点改成知识层并替换正文（无审计）；而同一换层经
        `store.add_node` 必被 #127 拦（判据与层集合无关）⇒ 两条入口不对称。现补
        `local_relayer_rows` 闸：既有 id 且两侧均不属 IMMUTABLE 而声明层不同 ⇒
        `PermissionError`（涉 IMMUTABLE 的换层仍由上方 D-007 双列闸处理，持密钥
        放行契约逐位不变）。③ 旧实现按备份行原样绑 SQL ⇒ 非 str `content` 全部落库
        （`dict`/`list` 经 #721 归一为 JSON 文本、`int`/`bool` 由 TEXT 列亲缘性静默
        转文本、`None` 落 NULL），而同一输入经 `add_node` **全部 `TypeError`**
        （#588/#594 唯一写入汇聚点）。现补 `bad_content` 闸：`nodes` 行带 `content`
        键而值非 str ⇒ `TypeError`（fail-closed，零落库，不做 `str()` 强转）。
        边界：缺 `content` 键的行仍按 #652② 落 NULL 并计入 `missing_columns`（不抢该面）。

        #778 F1/F2（**保护标记保真**）：旧实现走通用逐表 `INSERT OR REPLACE`，缺
        `add_node` 的 #659 判据 ⇒ 备份行漏带 `no_forget` 时旧行的保护标签被**静默
        抹掉**（无报错、无需密钥、无审计、`integrity_ok` 仍 True），而 `no_forget` ∈
        tags 是 `_is_forget_protected`（#36 唯一真源）的两个真源之一 ⇒ 节点随即变可删。
        现按 `add_node` 同款判据收口：① 旧行 tags 含 `no_forget` 而新行不含 ⇒ **并入**
        新行 tags（只在旧行确有该标签时并入，`protections` 行登记的节点不据此新造标签）；
        ② 旧行受保护 ⇒ 落 `protected_node_overwritten` 审计（同款 action_type）。
        **不**拒绝覆盖（保住灾备恢复语义；与 #659 取舍逐字一致）。F2 的「行在、标签被抹」
        由 ① 一并收口。新增返回键 `protected_tags_kept` / `protected_rows_overwritten`
        供观测（与 #652②/#721 的「新增上报键」同例）。

        #378（**回灌面**，外部报告人 @hawkongz 2026-10-12 追加评论）：写侧
        `add_perception` / `add_node` / `tag_node` 的 `tags` 列已过 `_dumps_tags` 的
        代理码点判据（含 U+D800–U+DFFF ⇒ `ValueError`），而**回灌面**此前绕过该判据：
        备份里 tags 列是**转义后的 ASCII 文本**（存量行形态即字面量 `\\ud83d`），经本
        方法 `json.loads` 回来仍是含真代理码点的 str ⇒ 毒行**重新造出**，实测无异常、
        `integrity_ok=true`、列内含转义代理、库仍可继续写入（**零信号**）。该毒行随后
        会让 `longterm_gate.write_snapshot` 的「已存在分支」UPDATE 抛
        `UnicodeEncodeError` ⇒ 该行三列全未变（写入失效）。
        修法：落库前**整批**判 `nodes` 行的 `tags` 列——经 `_json_tags` 归一后过
        `_dumps_tags` 的**同一判据**（单一真源，不另造第二套扫描）；带代理码点 ⇒
        `ValueError`（点名行 id）、fail-closed、零落库。
        取向＝**拒收**而非「清洗后收」，理由三条（详见闸位注释）：①`_dumps_tags` 明写
        「不做静默替换（替换会把污染源洗白）」；②与本方法既有整批判闸（#586/#478/#771）
        同形；③合法备份零影响（本端实测往返逐位不变）。
        边界（如实标注）：①只判 `nodes.tags`；`state_attributes` /
        `semantic_coordinates` / `spatial_coordinates` 的回灌面同样绕过（#494 只加了
        写侧闸），**不在本件范围**（另立件）；②`edges` / `entities` 等表的同类列未判
        （报告人自陈未测）；③缺 `tags` 键的行不抢（仍走 #652② 的 `missing_columns`）；
        ④非法 JSON / 非 list 形态的 `tags` 值不抢（`_json_tags` 容错口径）；⑤**不**迁移
        本库既有存量毒行（`_dumps_tags` 边界①不变）；⑥含毒行的老备份回灌**整批失败**
        （与 #586/#478/#771 的「整批拒」取舍逐字一致，不提供自动清洗）。
        """
        # #789：utf-8-sig 对无 BOM 备份逐字节等价、BOM 在场时剥掉（见 docstring）。
        with open(input_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        # ① 不可变层覆盖闸（先判后写，fail-closed）。
        # #478：闸判据从「备份声明层」扩为**双列**——声明层或库内旧行 layer 任一属
        # IMMUTABLE ⇒ 按不可变行处理（需密钥）。旧实现只查声明层 ⇒ 对库内受保护行
        # 声明 layer=knowledge 即无密钥改写/降级/删除（值守复核 6187575720）。
        imm_values = {lyr.value for lyr in self.store.IMMUTABLE_LAYERS}
        c = self.store.conn.cursor()
        existing = self._existing_tables()
        prior_layer = {}
        prior_tags = {}
        if "nodes" in existing:
            prior_layer = {rid: lyr for rid, lyr in c.execute("SELECT id, layer FROM nodes")}
            # #778 F1/F2：库内旧行 `tags` 列（JSON 文本）一次性读出——保护标记保真
            # 判据要用它，且**不随行数变 N+1**（与 `prior_layer` 同一次批量读）。
            prior_tags = {rid: tgs for rid, tgs in c.execute("SELECT id, tags FROM nodes")}
        valid_layers = {m.value for m in MemoryLayer}
        immutable_rows = [r.get("id") for r in data.get("nodes", [])
                          if r.get("layer") in imm_values or prior_layer.get(r.get("id")) in imm_values]
        # #478（编排侧裁定 2026-10-15，二次校正）：换层行的**攻击面**已由上方双列闸闭合
        # ——任一侧属 IMMUTABLE 的既有 id 换层必然进 immutable_rows ⇒ 无密钥 PermissionError、
        # 零落库。持密钥的换层**放行但显式标注**（沿 #125 硬配套契约：labeled＋audited＋
        # requires_review），不另设 ValueError——否则打掉 #125 既证的「密钥放行换层」合法场景
        # （test_issue125 族 c1 回归逮住，量具正常、我方判据过严）。
        relayer_rows = [r.get("id") for r in data.get("nodes", [])
                        if r.get("id") in prior_layer and r.get("layer") != prior_layer[r.get("id")]
                        and (prior_layer.get(r.get("id")) in imm_values or r.get("layer") in imm_values)]
        # #479（编排侧裁定 2026-10-15）：engine_meta 治理载荷行与 action_logs 带 id 行
        # 同走密钥闸；灾备恢复由恢复者持密钥执行（#570/#566 同族契约）。
        guarded_meta = [row.get("key") for row in data.get("engine_meta", [])
                        if row.get("key") in self.ENGINE_META_GUARDED_KEYS]
        id_log_rows = [row for row in data.get("action_logs", []) if row.get("id") is not None]
        if (immutable_rows or guarded_meta or id_log_rows) and not verify_designer(designer_key):
            raise PermissionError(
                "D-007 设计者认证失败：备份含需密钥治理面行（不可变层 "
                f"{len(immutable_rows)} 条 / engine_meta 治理快照 {len(guarded_meta)} 条 / "
                f"action_logs 带 id 行 {len(id_log_rows)} 条）（fail-closed，零落库）")
        # #586：实例角色闸（与上方密钥闸并列的第二道，同族旁路面）——备份含
        # **共享层**（anchor/structure）行时非 PRIMARY 不得导入，即便持有效密钥。
        # 旧实现只验密钥 ⇒ SUB + 密钥导入含 `structure` 行的备份即在本实例落
        # structure 行（公开 issue #586「同根因：import_all」实测）。
        # 口径取 `SHARED_LAYERS` 而非 `IMMUTABLE_LAYERS`：#191 已裁定 SELF 属
        # **本地层**（SUB 可写），若按 IMMUTABLE 收口会把 SUB 的自我层备份一并
        # 拒掉、回归 #191 ⇒ 与 `add_node` 的写权限单一真源保持一致。
        # 判据来源：公开 issue #586 ＋ `SHARED_LAYERS`（core.py `add_node` 角色闸）。
        # #674 残余面（**edges 行**，见下方 edges 分支）：本闸只扫 `data["nodes"]`，
        # `edges` 无 `layer` 列 ⇒ 边的端点层闸另在 edges 分支内补（同一真源
        # `_assert_edge_endpoints_role`）。
        shared_values = {lyr.value for lyr in self.store.SHARED_LAYERS}
        shared_rows = [r.get("id") for r in data.get("nodes", [])
                       if r.get("layer") in shared_values
                       or prior_layer.get(r.get("id")) in shared_values]
        if shared_rows and self.role != Role.PRIMARY:
            raise PermissionError(
                f"role={self.role.value} 无权导入共享层（anchor/structure）行"
                f"（{len(shared_rows)} 条）：共享层由父节点主控，即便持设计者密钥"
                "亦不放行（#586，fail-closed，零落库）")
        # #771 ②：同 id **换层**闸（复用 `add_node` 的 #127 判据，不另造第二套）。
        # 旧实现的 `immutable_rows` / `relayer_rows` 判据都要求「任一侧属
        # IMMUTABLE」⇒ **本地层之间**的换层（`context` → `knowledge`）不进任何闸：
        # 备份可无密钥把既有情境层节点改成知识层并替换正文（#771 实测，且无审计）。
        # 同一换层经 `add_node` 必被 #127 拦（`existing.layer != node.layer` ⇒
        # `PermissionError`，**与层集合无关**，本地层之间照样拦）⇒ 两条入口同语义
        # 却不对称，本闸把导入面接回同一真源。
        # 口径：既有 id 且**声明层合法**、与库内旧行层不同、且**两侧均不属
        # IMMUTABLE** ⇒ `PermissionError`（先判后写、fail-closed、零落库）。
        # 涉 IMMUTABLE 的换层**不在此闸**——由上方 D-007 双列闸处理（无密钥拒；
        # 持密钥放行并 labeled＋audited＋requires_review），该契约逐位不变
        # （`tests/test_issue478_479_import_governance_guard.py::test_g1_keyed_relayer_allowed_but_labeled`）。
        # 声明层非法（缺失/NULL/未知名）的行仍走下方 `rejected_invalid_layer` 逐行
        # 拒行上报（#478 变体口径不变），不被本闸抢走。
        # 边界（如实标注）：①本闸只堵**换层**；「同 id 同层替换正文」仍放行——与
        # `add_node` 的 #127 边界逐字一致（同层覆盖只受 #659 保护标记保真约束）；
        # ②拒整批而非逐行拒（与 #586/#478/#674 同取整批拒口径）；③不额外落审计
        # （被拒即零状态变更，与 `add_node` 的 #127 同款；issue 建议的「并记审计」
        # 未采纳，理由见回帖）。
        local_relayer_rows = [(r.get("id"), prior_layer.get(r.get("id")), r.get("layer"))
                              for r in data.get("nodes", [])
                              if r.get("id") in prior_layer
                              and r.get("layer") in valid_layers
                              and r.get("layer") != prior_layer[r.get("id")]
                              and prior_layer.get(r.get("id")) not in imm_values
                              and r.get("layer") not in imm_values]
        if local_relayer_rows:
            _rid, _old, _new = local_relayer_rows[0]
            raise PermissionError(
                f"禁止同 id 换层（#127）：id={_rid} 现为 {_old}，备份声明为 {_new}"
                f"（本地层换层行 {len(local_relayer_rows)} 条）（fail-closed，零落库）")
        # #771 ③：`nodes.content` 的**类型闸**（与 `add_node` 的 #588/#594 **同源**判据）。
        # 旧实现按备份行原样绑 SQL ⇒ 非 str 全部落库：`dict`/`list` 经 #721 归一成
        # JSON 文本、`int`/`bool` 由 SQLite TEXT 列**亲缘性静默转文本**（`'12345'` /
        # `'1'`）、`None` 落 SQL NULL —— 而同一输入经 `store.add_node` **全部
        # `TypeError`**（#588/#594 的闸在唯一写入汇聚点）。两条入口同语义不对称，
        # 且 `int`/`bool` 的静默转文本正是 #588 docstring 明令「比拒绝更坏」的形态。
        # 修法：写前**整批**判（先判后写）——备份 `nodes` 行中凡**带 `content` 键**
        # 而值非 str ⇒ `TypeError`（可诊断文案点名行 id 与实收类型）、fail-closed、
        # 零落库。**不**做 `str()` 静默强转（同 #331/#588 口径）。
        # 边界（如实标注）：①只判「存在 content 键且非 str」——**缺** content 键的行
        # 仍按 #652② 既有契约落 NULL 并计入 `missing_columns`（本闸不抢该面）；
        # ②本闸在 `nodes` 表上**优先于** #721 的 TEXT 列容器值归一：容器 content
        # 此前走归一落 JSON 文本，现按 #588/#594 同源闸整批拒（属**收紧**，与
        # #771 ③「复用 add_node 的类型闸、非 str 一律拒绝」一致）；③其余表（含
        # `promotion_proposals.content_hash` 等非 content 语义列）不受影响——判据
        # 只落在 `nodes` 表的 `content` 列上。
        bad_content = [{"id": r.get("id"), "content_type": type(r.get("content")).__name__}
                       for r in data.get("nodes", [])
                       if "content" in r and not isinstance(r.get("content"), str)]
        if bad_content:
            raise TypeError(
                "#771 ③ import_all 拒绝非 str content（收到 %s，共 %d 行，例 id=%r）："
                "非 str 会以 BLOB/亲缘性转文本/NULL 静默落库，绕过 add_node 的 "
                "#588/#594 类型闸（与 #331 同口径，fail-closed，零落库）"
                % (bad_content[0]["content_type"], len(bad_content),
                   bad_content[0]["id"]))
        # #378（**回灌面**）：写侧 `add_perception` / `add_node` / `tag_node` 的
        # tags 列已过 `_dumps_tags` 的代理码点判据（含 U+D800–U+DFFF ⇒ ValueError，
        # fail-closed）⇒ 公开写侧不再产出毒行。但**回灌面**此前是绕过的：
        # `export_all` 落盘的 tags 列是**转义后的 ASCII 文本**（`_dumps_tags`
        # 旧默认/存量行形态即字面量 `\ud83d`），备份里这类行经本方法 `json.loads`
        # 回来仍是**含真代理码点的 str**，再经 `_dumps_tags` 写回（或按 #721 归一）
        # ⇒ 毒行**重新造出**，而写侧闸一次都没经过：实测无异常、`integrity_ok=true`、
        # 列内含转义代理、且库仍可继续写入（**零信号**）。
        # 这条之所以要堵：`export_all → import_all` 是仓内**自带的灾备/迁移路径**，
        # 若回灌面被视作「存量面」而豁免，则 `_dumps_tags` 边界①「不迁移存量转义行」
        # 隐含的前提「存量行只存在于本库、不可经备份传播」被这条路径推翻——毒行可
        # 在库之间无限往返；且该毒行会让后续**正常写入路径**在该行上失效（
        # `write_snapshot` 已存在分支的 UPDATE 抛 `UnicodeEncodeError`，三列全未变）。
        # 修法（**取向＝拒收**，与写侧同口径、**单一真源不另造第二套校验**）：
        # 落库前**整批**判——备份 `nodes` 行中凡**带 `tags` 键**且其值（经
        # `_json_tags` 归一为 list：JSON 文本 / 真 list / None 三形态）含代理码点
        # ⇒ `ValueError`（文案点名行 id 与首条拒绝理由）、fail-closed、零落库。
        # 判据**复用** `_dumps_tags` 本体（只借其拒绝判据、丢弃其返回值），不在此
        # 处手写第二份 `0xD800 <= ord(ch) <= 0xDFFF` 扫描。
        # 为什么取「拒收」而非「清洗后收」（如实标注取舍）：
        #   ① `_dumps_tags` 判据明写「**不**做静默替换（替换会把污染源洗白、掩盖
        #      上游解码错误）」——回灌时剔除/替换代理等于在同一列上开一个洗白口，
        #      与唯一真源的口径直接冲突；
        #   ② 本方法既有的整批判闸（#586 共享层行 / #478 双列层闸 / #771 ②③ 换层
        #      与 content 类型闸）**一律**取「先判后写、整批拒、零落库」⇒ 本闸与
        #      之同形，不引入第四种处置口径；
        #   ③ 合法备份零影响：判据只在备份行**确实**带代理码点时触发（本端实测
        #      合法往返逐位不变），故灾备路径的正常用法不被收窄。
        # 代价（如实标注）：真有一份**含毒行**的老备份时，回灌会**整批失败**而非
        # 「恢复其余行」——与 #586/#478/#771 的取舍逐字一致（要恢复须先清洗该备份；
        # 仓内**不**提供自动清洗，因为清洗＝洗白）。**不**迁移本库既有存量毒行
        # （`_dumps_tags` 边界①不变，那是另一条面）。
        # 边界（如实标注）：①只判 `nodes` 表的 `tags` 列；`state_attributes` /
        # `semantic_coordinates` / `spatial_coordinates` 三列的回灌面同样绕过
        # （#494 只加了写侧闸），**不在本件范围**（另立件）；②`edges` / `entities`
        # 等其它表的同类列未判（报告人自陈未测，另议）；③只判「带 `tags` 键」的行
        # ——缺 `tags` 键的行仍按 #652② 落 NULL 并计入 `missing_columns`（不抢该面）；
        # ④非法 JSON 文本 / 非 list 形态的 `tags` 值仍按既有路径原样落库
        # （`_json_tags` 的容错口径，本闸不扩到「tags 形状」面）。
        bad_tags = []
        for r in data.get("nodes", []):
            if "tags" not in r:
                continue
            try:
                _dumps_tags(_json_tags(r.get("tags")))
            except ValueError as exc:
                bad_tags.append({"id": r.get("id"), "reason": str(exc)})
        if bad_tags:
            raise ValueError(
                "#378 回灌面拒绝含 UTF-16 代理码点的 tags 行（共 %d 行，例 id=%r；"
                "理由：%s）：毒标签无法 UTF-8 编码，回灌后该行会让后续正常写入"
                "（如 write_snapshot 已存在分支）静默失效，并随备份继续传播"
                "（fail-closed，零落库）"
                % (len(bad_tags), bad_tags[0]["id"], bad_tags[0]["reason"]))
        counts = {}
        skipped = []
        discarded = {}
        missing_columns = {}
        orphan_rows = []
        rejected_layer = []
        # #721：值形状面上报（容器值经 JSON 文本归一 / 无法容纳而跳过的行）。
        normalized_values = {}
        skipped_rows = {}
        # #125 增量复现（同 id 覆写面）：`escalation_points` 上「库内已有同 id ⇒
        # 不覆写」被跳过的行数，按表计数（与 `skipped_rows` / `missing_columns`
        # 同形：只在非空时出现该表键）。**不许静默丢弃**是本仓既有口径。
        skipped_existing_rows = {}
        # #778 F1/F2：保护标记保真的两个收集面——①受保护节点被导入覆写的留痕
        # （循环后统一落 `protected_node_overwritten` 审计，与 `add_node` 同款）；
        # ②保护标记被并回的节点 id（供返回值与守卫观测）。`_protected_ids` 取一次
        # （`protections` 行登记集，与 `add_node` 的 `_is_forget_protected` 同真源，
        # 不随行数变 N+1）。
        _overwrite_protected = []
        _kept_no_forget = []
        _protected_ids = set(self.store.get_protected_nodes())
        for table in self.M13_TABLES:
            if table not in existing:
                skipped.append(table)
                continue
            rows = data.get(table, [])
            if not rows:
                counts[table] = 0
                continue
            # #125 增量复现（第三方 Yinfeng818 2026-10-16T06:52:15Z；本端
            # `VERIFY-125-INC` 核验成立）：`escalation_points` 此前与其它表同走
            # 通用腿 `INSERT OR REPLACE`，而该表**不在**三集合密钥判据内
            # （`immutable_rows` / `guarded_meta` / `id_log_rows`）⇒ 无密钥、无闸、
            # 无审计。最重形态是**同 id 覆写**：备份以**既有种子 id** 写入 ⇒ 整行
            # 替换 ⇒ 不止能关升级点，还能**反转触发语义、替换处置动作**（原
            # `action` 无副本），且走完零审计（`action_logs` 增量 0）。
            # 修法（**只收窄覆写面**）：本表写腿改为「既有 id 行不写、新 id 行才写」
            # ——库内已有同 id ⇒ **跳过并计数上报**（`skipped_existing_rows`）；
            # 新 id 行照旧落库（无密钥新增面是设计取舍，本件不动，见 docstring）。
            # 预筛置于列白名单计算之前 ⇒ `discarded_keys` / `missing_columns` /
            # `normalized_values` 只统计**实际写入**的行（不把被跳过的既有行算进去）。
            if table == "escalation_points":
                existing_ids = {r[0] for r in c.execute("SELECT id FROM escalation_points")}
                kept = [r for r in rows if r.get("id") not in existing_ids]
                n_existing = len(rows) - len(kept)
                if n_existing:
                    skipped_existing_rows[table] = n_existing
                rows = kept
                if not rows:
                    counts[table] = 0
                    continue
            # ② 列名白名单（PRAGMA table_info）：非白名单键丢弃并计数。
            # #652②：键集合按**全部行**的并集计算，顺序＝跨行首次出现序（确定性
            # ——`_table_columns` 回的是 set，直接迭代会引入非确定列序，故显式
            # 按出现序累积）。旧实现只取首行（`list(rows[0].keys())`），两种后果
            # 都在静默发生（值守复核实测）：
            #   · 首行缺某列 ⇒ 该列整体不进 `cols`，**后行真值一并被丢**（回灌是
            #     灾备/迁移路径 ⇒ 静默数据丢失，且 `discarded_keys` 为空）；
            #   · 后行缺某列 ⇒ 该列在 `cols` 内、`r.get(col)` 落 NULL，返回值里
            #     一个字都没有 ⇒ 调用方读不到「这一行缺列」。
            # 现按并集取列（后行真值不再被首行形状吞掉），并把「缺列行数」按表按列
            # 上报（`missing_columns`），使原先静默的 NULL 落库可观测。
            allowed = self._table_columns(table)
            if table == "archive_state":
                # 归档原值随备份恢复，但不得借此越过共享层的还原权限闸。
                for row in rows:
                    self.store.assert_node_writable_by_role(
                        row.get("node_id"), action="导入共享层节点的归档恢复状态")
            keys = []
            for r in rows:
                for k in r:
                    if k not in keys:
                        keys.append(k)
            cols = [k for k in keys if k in allowed]
            dropped = sorted(k for k in keys if k not in allowed)
            if dropped:
                discarded[table] = dropped
            row_missing = {col: sum(1 for r in rows if col not in r) for col in cols}
            row_missing = {col: n for col, n in row_missing.items() if n}
            if row_missing:
                missing_columns[table] = row_missing
            if not cols:
                counts[table] = 0
                continue
            placeholders = ",".join("?" for _ in cols)
            col_sql = ",".join(cols)
            # #721：列**声明类型**（`PRAGMA table_info` 的 type 栏）——值形状归一的
            # 判据来源（TEXT 列才可把容器值 `json.dumps` 归一）。与上方 `allowed`
            # 同源同一次表结构读数（同一 `PRAGMA table_info`，不引入第二份列真源）。
            col_types = {r[1]: (r[2] or "").upper()
                         for r in c.execute(f"PRAGMA table_info({table})")}
            # #93：外键开启（`PRAGMA foreign_keys=ON`）后，指向不存在节点的边会让
            # `INSERT OR REPLACE` 抛 `FOREIGN KEY constraint failed` 并**中断整次
            # 导入**。故先按库内节点集合预筛：孤儿边**不导入**、计入返回值
            # `orphan_edges` 并给样本（「跳过并报告」，保持既有返回体形状）。
            # 节点集合取库内当前 `nodes`——备份的 `nodes` 表已在上一轮迭代导入
            # （`M13_TABLES` 中 nodes 先于 edges），故备份自带的合法端点已被覆盖。
            if table == "edges":
                node_ids = {r[0] for r in c.execute("SELECT id FROM nodes")}
                # #482：旧实现 `orphan_rows = [r for r in rows if r not in keep]` 是
                # **list 成员判断**——对 `keep` 逐 dict 比较 ⇒ 约 E²/2 次比较
                # （本机实测 E=8000：3.20e7 次 / 3.1s；E=4000：8.0e6 次 / 0.8s），
                # 且与同一 `_transactional` 写事务内的批量 INSERT 共处一个事务、
                # 全程持有写锁 ⇒ 共库进程等到 busy_timeout 后报 `database is locked`。
                # 现改为**单趟分区**：判据逐字仍是 `source_id/target_id in node_ids`
                # （集合成员判断，均摊 O(1)），一次遍历同时产出 keep 与 orphan ⇒ O(E)。
                # 语义等价（含边界）：①满足判据的行必然以**同一对象**出现在 `keep`
                # 里（list `in` 先按身份命中）⇒ 旧 `orphan_rows` 恰是「不满足判据者」；
                # ②不满足判据的行不可能 `==` 某个 keep 行（dict 相等蕴含键值全等，
                # 端点值也就相等 ⇒ 自相矛盾）；③两列表都按原行序一次成形。故重复边、
                # 缺键、None 端点、同端点异形状行一律落 orphan，与旧判据逐字相同。
                # 判据来源：公开 issue #482（一）；守卫 tests/test_issue482_resource_regression_guard.py。
                keep = []
                orphan_rows = []
                for r in rows:
                    if (r.get("source_id") in node_ids
                            and r.get("target_id") in node_ids):
                        keep.append(r)
                    else:
                        orphan_rows.append(r)
                rows = keep
                # #674：边角色闸（与 `add_edge` 同口径、同一真源
                # `_assert_edge_endpoints_role`）——#586 的导入角色闸只扫
                # `data["nodes"]`（`edges` 无 `layer` 列，天然不在该判据内）⇒ 旧实现
                # 的 edges 分支只有 #93 孤儿预筛、随后裸 `INSERT OR REPLACE`：SUB
                # 用 edges-only 备份即可无密钥落共享层端点边（同一引擎上 `add_edge`
                # 对同一条边抛 PermissionError），或以同 id 覆盖 PRIMARY 既有共享层
                # 已验证边（改端点/confidence/verified，全程无审计）。
                # 修法（取向同 #586「先判后写、零落库」）：写入前逐行过端点层闸——
                # ①**新增**面：新行两端点任一属 `SHARED_LAYERS` ⇒ 拒；
                # ②**覆盖**面：查同 id 旧边的端点（防「先建本地边占位、再用共享层
                #   id 覆盖」，与 `add_edge` 的同 id 旧边复查同款）⇒ 拒。
                # 判据来源：公开 issue #674 ＋ `_assert_edge_endpoints_role`
                # （`#234`/`#191` 单一真源）。
                # 边界（削掉的判别力，如实标注）：与 `nodes` 闸同取**整批拒绝**
                # （PermissionError、零落库）而非「逐行拒收并报告」——口径二选一，
                # 取与 `#586`/`#478` 一致者；PRIMARY 走 `_assert_edge_endpoints_role`
                # 直接短路 ⇒ 合法恢复路径零影响（防过修）。
                if self.role != Role.PRIMARY:
                    for r in rows:
                        self.store._assert_edge_endpoints_role(
                            r.get("source_id"), r.get("target_id"))
                        _old_edge = self.store.get_edge(r.get("id"))
                        if _old_edge is not None:
                            self.store._assert_edge_endpoints_role(
                                _old_edge.source_id, _old_edge.target_id)
            for r in rows:
                if table == "nodes":
                    # #478 变体：layer 无效（缺失/NULL/大小写变体/未知名）的行——落库后
                    # 读面 `MemoryLayer(row[8])` 抛 ValueError（节点从读取面消失）＝毒行
                    # ⇒ 拒行并报告（值守复核复现体含 NULL 变体）。
                    if r.get("layer") not in {m.value for m in MemoryLayer}:
                        rejected_layer.append({"id": r.get("id"),
                                               "layer": repr(r.get("layer"))[:40]})
                        continue
                    # #778 F1/F2：**保护标记保真**（复用 `add_node` 的 #659 判据，
                    # 不另造第二套）。旧实现走通用逐表 `INSERT OR REPLACE` ⇒ 备份行
                    # 漏带 `no_forget` 时，旧行的保护标签被**静默抹掉**（无报错、无需
                    # 密钥、无审计、`integrity_ok` 仍 True），而 `no_forget` ∈ tags 是
                    # `_is_forget_protected`（#36 唯一真源）的两个真源之一 ⇒ 节点随即
                    # 变可删（`delete_node` 返回 True）。F2 是同一缺口的弱化形态：
                    # `protections` 行仍在（仍拒删），但标签源被抹 ⇒ 双源背离。
                    # 口径与 `add_node` 逐字同构：
                    #   ① 旧行 tags 含 `no_forget` 而新行不含 ⇒ **并入**新行 tags
                    #      （只在旧行**确有**该标签时并入——`protections` 行登记的节点
                    #      另有真源，不据此新造标签，避免把「仅经 protections 行保护」
                    #      的节点语义悄悄改写成带标签）；
                    #   ② 旧行受保护（`_is_forget_protected`）⇒ 落
                    #      `protected_node_overwritten` 审计，使「受保护节点被导入覆写」
                    #      不再静默（审计在本次导入事务内一并提交）。
                    # **不**拒绝覆盖（保住灾备恢复语义）：内容仍按备份落库，只保
                    # 标记与可审计——与 #659 的取舍逐字一致。
                    _old_tags = _json_tags(prior_tags.get(r.get("id")))
                    _new_tags = _json_tags(r.get("tags"))
                    if "no_forget" in _old_tags and "no_forget" not in _new_tags:
                        _new_tags = list(_new_tags) + ["no_forget"]
                        r["tags"] = _dumps_tags(_new_tags)
                        _kept_no_forget.append(r.get("id"))
                    if self.store._is_forget_protected(
                            None, _protected_ids, node_id=r.get("id"), tags=_old_tags):
                        _overwrite_protected.append({
                            "id": r.get("id"), "old_tags": list(_old_tags),
                            "new_tags": list(_new_tags),
                            "protections_row": r.get("id") in _protected_ids})
                    # #574：importance 值域钳制（与 add_node 同款）
                    # 旧实现绕过 add_node 直接 INSERT，越界值原样落库
                    if "importance" in r:
                        r["importance"] = _finite_clamp(
                            r.get("importance"), 0.0, 1.0, default=0.5)
                    # #690 P1-b：`confidence` 与上列**同门面、同义务**——#574 只补了
                    # `importance`、漏了本列（「修复只补一半字段」）。旧实现经
                    # `import_all` 回灌畸形 confidence（1e6 / -5.0 / NaN）时原样落库：
                    # 实测源库两列直改 1e6 → export_all → 新库 import_all 后
                    # `importance=1.0`（已钳）、`confidence=1000000.0`（未钳）。畸形
                    # confidence 会参与一切按 confidence 加权/阈值判定的面（排序、
                    # 闸门、衰减），无告警。口径与上列**逐字同构**
                    # （`_finite_clamp(v, 0.0, 1.0, default=0.5)`，与 `add_node`
                    # 写入侧归一、`_shared_field_key` 判同口径一致）。
                    # 判据来源：公开 issue #690（P1-b，复核确认成立）＋ 同处 #574 先例。
                    # 边界（如实标注）：本闸只补 `nodes` 表的 `confidence`；
                    # `edges` 表的 `confidence`/`weight` 在 `import_all` 同样未钳
                    # （不在本件范围，已在报告中点名）。
                    if "confidence" in r:
                        r["confidence"] = _finite_clamp(
                            r.get("confidence"), 0.0, 1.0, default=0.5)
                # #721：**值形状**归一（绑定前逐行逐列）。#125 的闸只管**键**白名单，
                # 值形状零校验 ⇒ 备份里某 TEXT(JSON) 列的值若是**容器**（嵌套 dict /
                # list，例如 `context` 列 = `{"k":1}` 而非 `'{"k":1}'` 文本），
                # `tuple(r.get(col) …)` 直接绑 SQL 时绑定层抛
                # `ProgrammingError: type 'dict' is not supported`（部分 sqlite 版本族
                # 报 `IntegrityError: datatype mismatch`）——**整批回滚、整表 0 落库**，
                # 既不「部分恢复」、也报不出坏行位置，与 docstring 承诺的
                # 「跳过并报告 / 保持部分恢复语义」相悖。
                # 口径（与 `_tolerant_nodes`「逐行容错、如实计数」同款，且**不丢数据**）：
                #   · 列声明类型为 TEXT ⇒ 容器值按该列**既有存储形态**归一为 JSON 文本
                #     （`json.dumps`，`export_all` 写的正是 JSON 文本 ⇒ 无损往返），
                #     逐列计数上报 `normalized_values`；
                #   · 其余列型（REAL/INTEGER/BLOB…）无法容纳容器 ⇒ **跳过该行**并计入
                #     `skipped_rows`（不去猜「这个 dict 想表达什么数」），其余行照常落库。
                # 合法值（None/str/int/float/bytes；bool 是 int 子类）原样通过 ⇒
                # 合法备份零改动、零新键内容。
                # 判据来源：公开 issue #721（「绑定前逐行逐列做值归一…或跳过该行并计入
                # discarded/failed 计数」）＋ 同处 #652② 的「新增上报键」先例。
                # 边界（如实标注）：只归一**容器**值；数值越界仍走 #574/#690 的钳制、
                # 字符串编码口径不动（那是 #378/#494 的面）。
                _skip_row = False
                for _col in cols:
                    _val = r.get(_col)
                    if _val is None or isinstance(_val, (str, int, float, bytes)):
                        continue
                    if col_types.get(_col, "") == "TEXT":
                        r[_col] = json.dumps(_val, ensure_ascii=False)
                        _nv = normalized_values.setdefault(table, {})
                        _nv[_col] = _nv.get(_col, 0) + 1
                    else:
                        _skip_row = True
                        break
                if _skip_row:
                    skipped_rows[table] = skipped_rows.get(table, 0) + 1
                    continue
                # #125 增量复现：`escalation_points` 用 `INSERT OR IGNORE`——上方预筛
                # 已把「库内既有 id」的行摘掉并计数，此处 OR IGNORE 只作**备份自身
                # 含重复 id** 时的兜底（取首条），使本表上**不再存在任何整行替换
                # 路径**（OR REPLACE 的 delete+insert 语义在本表被彻底关掉）。
                # 其它表逐字不变（仍走通用腿 `INSERT OR REPLACE`）。
                _verb = ("INSERT OR IGNORE" if table == "escalation_points"
                         else "INSERT OR REPLACE")
                c.execute(f"{_verb} INTO {table} ({col_sql}) VALUES ({placeholders})",
                          tuple(r.get(col) for col in cols))
            counts[table] = len(rows) - skipped_rows.get(table, 0)
        # #778 F1/F2 硬配套：受保护节点被导入覆写 ⇒ 审计只追加（与 `add_node` 的
        # #659 同款 action_type / 字段面）。**不**拒绝覆盖（灾备恢复语义不动）。
        # 按 id 去重（备份自身可含重复 id 行 ⇒ 同一次导入只落一条审计）。
        _seen_ow = []
        for _ow in _overwrite_protected:
            if _ow["id"] in _seen_ow:
                continue
            _seen_ow.append(_ow["id"])
            self.store.log_action(
                "protected_node_overwritten",
                f"受保护节点 {_ow['id']} 被同 id 导入覆写（#778）：no_forget 标签已保真、"
                f"protections 行不受影响，保护语义（拒删/免衰减）仍生效",
                node_ids=[_ow["id"]],
                outcome={"no_forget_in_new_tags": "no_forget" in _ow["new_tags"],
                         "protections_row": _ow["protections_row"],
                         "old_tags": _ow["old_tags"]})
        # ③ 硬配套：审计只追加（`action_logs` 无更新/删除路径）。
        if guarded_meta or id_log_rows:
            self.store.log_action(
                "import_governance_override",
                f"经密钥放行导入治理面行：engine_meta 快照 {len(guarded_meta)} 条 / "
                f"action_logs 带 id 行 {len(id_log_rows)} 条（#479：须事后复核）",
                outcome={"requires_review": True})
        if immutable_rows:
            self.store.log_action(
                "import_immutable_override",
                f"经密钥放行导入不可变层行 {len(immutable_rows)} 条（#125 硬配套：须事后复核）",
                node_ids=[i for i in immutable_rows if i],
                outcome={"requires_review": True, "immutable_rows": len(immutable_rows),
                         "relayer_rows": len(relayer_rows)})
        self._commit_locked()
        # #493：导入后把运行期容量状态随盘面刷新（与「启动时从库读」同口径）：
        # 上方逐表 INSERT OR REPLACE 已把备份的 engine_meta（含 context_max）落库，
        # 但 self._context_max 仍是构造期读到的旧值；而 enforce_context_cap 的 FIFO
        # 淘汰作用于**整库**情境层 ⇒ 下一次 add_context 会按旧上限（默认 200）静默批量
        # 删掉备份里多出的情境记忆（灾备恢复等于无效）。复用既有读数路径
        # _load_context_max（不触发淘汰，语义与构造期一致）；读点置于本次写事务提交
        # 之后，读到的是已落定状态。不覆盖 set_context_cap 的显式设置语义，也不改
        # enforce_context_cap 的裁剪判据。判据来源：公开 issue #493。
        self._context_max = self._load_context_max()
        # 恢复自检：导入后跑 `PRAGMA foreign_key_check` 体检。经上方 #93 预筛，正常
        # 路径下应恒为空；保留本步作为「白名单列裁剪后仍出现悬挂」的兜底观测
        # （不拦写入，随返回值交付）。
        c.execute("PRAGMA foreign_key_check")
        # 按 (表名, rowid) 去重 ⇒ 两端全悬挂的**同一条边**只计 1。对照
        # verify_integrity()["orphan_edges"] 按边×端点累加，同一场景计 2；两处
        # 口径不同（详见本方法 docstring），不要把这一个数当成那一个数。
        dangling = {(table, rid) for table, rid, *_ in c.fetchall()}
        return {"imported": counts, "skipped_tables": skipped,
                "dangling_rows": len(dangling),
                "orphan_edges": len(orphan_rows),
                "orphan_edges_sample": [
                    {"id": r.get("id"), "source_id": r.get("source_id"),
                     "target_id": r.get("target_id")} for r in orphan_rows[:20]],
                "integrity_ok": not dangling and not orphan_rows,
                "discarded_keys": discarded,
                "missing_columns": missing_columns,
                "immutable_rows": [i for i in immutable_rows if i],
                "requires_review": bool(immutable_rows or guarded_meta or id_log_rows),
                "rejected_invalid_layer": {"count": len(rejected_layer),
                                           "sample": rejected_layer[:20]},
                # #721：值形状面上报（新增键，与 #652② 加 `missing_columns` 同例）。
                "normalized_values": normalized_values,
                "skipped_rows": skipped_rows,
                # #125 增量复现（同 id 覆写面）：`escalation_points` 上「库内已有同
                # id ⇒ 不覆写」被跳过的**既有行数**（按表计数，非空才出现该表键）。
                # 新增键，与 #652② `missing_columns` / #721 `skipped_rows` 同例。
                "skipped_existing_rows": skipped_existing_rows,
                # #778 F1/F2：保护标记保真的可观测量（新增键，同例）——
                # `protected_tags_kept` 列出「旧行 tags 含 no_forget 而备份行不含 ⇒
                # 已并入」的节点 id；`protected_rows_overwritten` 列出「受保护节点被
                # 本次导入覆写（已落 protected_node_overwritten 审计）」的节点 id。
                "protected_tags_kept": list(dict.fromkeys(_kept_no_forget)),
                "protected_rows_overwritten": _seen_ow}

    def verify_integrity(self) -> Dict:
        """M13：完整性校验（边引用节点存在性 + 表计数）

        指标口径（**勿与 `import_all()` 的 `dangling_rows` 混用**）：
        `orphan_edges` = `orphan_source + orphan_target`，**按「边×端点」累加**——
        一条边 source 悬挂计 1、target 悬挂计 1，故**两端全悬挂即计 2**。
        对照 `import_all()` 返回的 `dangling_rows`：那是**按「违规行」去重**（同一
        条两端全悬挂的边只计 1）。故上述最小场景下本指标给 **2**、那边给 **1**。
        两者各自都有 `integrity_ok` 判「不通过」，但**不是同一个指标**，勿混用
        （该关系的守卫见 `tests/test_core_dangling_metric_parity.py`）。

        #778-F4：本面**同时**上报 `protections` 表的孤儿行（`orphan_protections`
        ＋ 前 20 条样本 `orphan_protections_sample`，判据真源
        `LayeredStore.orphan_protection_rows`）并纳入 `integrity_ok`——`protections`
        表无外键声明（对照 `edges`），`PRAGMA foreign_key_check` 覆盖不到它，故
        「保护登记指向不存在节点」这一损坏形态此前在本面**完全不可见**。
        边界（如实标注）：`orphan_protections` 与 `orphan_edges` 是**两个独立指标**
        （不同表、不同口径：前者按行、后者按「边×端点」），本面只把两者一并纳入
        `integrity_ok`；`import_all()` 的 `integrity_ok`（经 `PRAGMA foreign_key_check`）
        **仍看不到** `protections`（该表无外键）——导入侧对称预筛落在 `import_all`
        内，属另一条修复线。
        """
        c = self.store.conn.cursor()
        c.execute("SELECT COUNT(*) FROM edges e LEFT JOIN nodes n ON e.source_id=n.id WHERE n.id IS NULL")
        orphan_source = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM edges e LEFT JOIN nodes n ON e.target_id=n.id WHERE n.id IS NULL")
        orphan_target = c.fetchone()[0]
        orphan_prot = self.store.orphan_protection_rows()
        return {"orphan_edges": orphan_source + orphan_target,
                "orphan_protections": len(orphan_prot),
                "orphan_protections_sample": orphan_prot[:20],
                "stats": self.store.get_stats(),
                "integrity_ok": (orphan_source + orphan_target) == 0 and not orphan_prot}

    # ==================== v1.5 认知增强（A-1~A-5 · ANALYSIS-ARGUS-REV1） ====================

    # ---- A-1 被拒绝路径资产化 ----

    def register_rejected_path(self, path_type: str, description: str, reason: str,
                               evidence: str = "") -> str:
        """A-1：登记被拒绝/被证伪路径（偏差历史资产化，防重复失败）"""
        return self.store.add_rejected_path(path_type, description, reason, evidence)

    def list_rejected_paths(self, status: str = None) -> List[Dict]:
        return self.store.list_rejected_paths(status)

    def find_rejected_paths(self, query: str, limit: int = 10) -> List[Dict]:
        """A-1：相似被拒路径检索（防重复失败；供盲区学习闭环消费）"""
        paths = self.store.list_rejected_paths(status="open")
        # #657：description/reason 允许为 NULL，直接 `+` 会抛
        # TypeError: unsupported operand type(s) for +: 'NoneType' and 'NoneType'
        scored = [(p, LayeredStore.char_bigram_jaccard(query,
                                                       (p["description"] or "") + (p["reason"] or "")))
                  for p in paths]
        scored.sort(key=lambda x: -x[1])
        return [p for p, s in scored if s > 0.15][:limit]

    # ---------- v1.16 图架构增强透传 ----------
    def traverse(self, start_id: str, relation_types: List[str] = None,
                 direction: str = "out", max_depth: int = 5,
                 min_importance: float = 0.0, max_nodes: int = 500) -> List[Dict]:
        """有界图遍历（importance 降序；relation_types 过滤因果/相似/归属）"""
        return self.store.traverse(start_id, relation_types=relation_types,
                                   direction=direction, max_depth=max_depth,
                                   min_importance=min_importance,
                                   max_nodes=max_nodes)

    def subgraph(self, root_id: str, max_depth: int = 3,
                 relation_types: List[str] = None) -> Dict:
        """子图嵌套查询（根 + hierarchical 子节点递归 + 内部边）"""
        return self.store.subgraph(root_id, max_depth=max_depth,
                                   relation_types=relation_types)

    @_transactional
    def subgraph_replace(self, parent_id: str, old_sub_root_id: str,
                         new_subtree: Dict) -> Dict:
        """子图替换（可嵌套认知图 · 子部件生成基底）。

        把 parent_id 下挂在 old_sub_root_id 的旧子树整体移除，将
        new_subtree（dsh 嵌套认知图同构：{id, spatial_coordinates,
        content, importance, subgraph{nodes, edges}}）作为新子树挂入，
        新根以 hierarchical 边（new→parent）保持层级语义。

        支撑「替换子部件生成其他图像」：dsh 端子部件匹配/定位后，
        认知图侧以本操作完成子部件级的图更新。
        返回 {new_root_id, new_nodes, removed_nodes}

        issue #139（子图替换「先删后挂」且无范围/保护防护）：旧实现
        `removed = [old_sub_root_id] + [...]` → `for nid in removed:
        self.store.delete_node(nid)` → **其后**才 `_attach(new_subtree, ...)`，
        且删除早于任何校验、`delete_node` 的布尔返回被丢弃。后果：①`new_subtree`
        缺 `"id"` ⇒ `_attach` 抛 `KeyError`，旧子树**已删**（删了没挂上）；
        ②传 `old_sub_root_id == parent_id` 即删父；③旧子树含受保护/不可变层
        节点 ⇒ 该点被拒删、其子孙照删，`removed_nodes` 仍报候选集长度（谎报）。
        修法（最小 · 复用既有判据）：
          A **前置校验**（先判后动）：`parent_id` / `old_sub_root_id` 存在性、
            旧子根 ≠ 父、旧子根须是父的 hierarchical 后代（范围）、
            新子树**递归**逐节点校验含 `"id"`（把 KeyError 提前到任何写入之前）、
            旧子树每个节点均可删（`IMMUTABLE_LAYERS` ∪ `_is_forget_protected`）；
            任一条不满足 ⇒ **整体拒绝**（抛错，零改动），不做部分替换。
          B **先挂后删**：先递归挂入新子树，成功后才移除旧子树——挂入失败时
            旧子树**完好**，消掉「删了没挂上」的数据丢失形态。
          C **计数如实**：`removed_nodes` 改为**实际删除成功数**（校验
            `delete_node` 返回值），不再报候选集长度。
        判据来源：设计稿 `docs/plans/删除恢复持久化完整性_修复设计_v0.1.md`
        §2（`#139` 子缺陷 ①②③⑤）＋ 待裁清单 **B-1**（保护判定沿用既有角色闸、
        不另加密钥）/ **S-2=(a)**（保护命中 ⇒ 整体拒绝，全有或全无）/
        **S-3**（范围校验）。

        #139 追评（字段类型类失败仍半截落库）：A 段校验只查「是 dict 且有 `id`」，
        而 `_attach` 还会对 `importance` / `confidence` / `spatial_coordinates.
        image2d` 分量做 `float()`、对 `node["id"]` 做字符串拼接、对 `content` 做
        `json.dumps`。新子树里任一**非首个**节点带非数值字段（如
        `"importance": "high"`、坐标分量为 `null`、`"confidence": "0.5x"`）时，
        `_attach` 在**挂入中途**抛 `ValueError` / `TypeError`，而此前已挂的节点
        各自落库 ⇒ parent 下留下半棵无主新子树、旧子树同时还在；调用方按
        「抛错 = 没改动」重试，每次再多挂半棵（报告人最小复现：两次调用后
        parent 的 hierarchical 后代数 1 → 5）。
        修法（取向选择 · **事务边界**，不选「把 `float()` 全提到校验阶段」）：
        把 ②③ 段（`_attach` 递归挂入 ＋ 旧子树删除）整体收进单一
        `self.store._tx()` 事务边界。理由：① 枚举式校验须逐个追平 `_attach`
        里的每一处 `float()` / 字符串拼接 / `json.dumps`，日后往 `_attach` 加字段
        即静默重现本缺口；事务边界是**机械完备**的——凡写入中途抛错一律整体回滚。
        ② 旧 docstring 曾断言「无法在不改写入原语的前提下做到整体回滚」，该断言
        在 **#665** 之后已失效：`_commit_locked()` 具备嵌套感知（depth>1 时不提交、
        不销毁外层 SAVEPOINT），故最外层 `self.store._tx()` 是**真事务**（本机实测：
        边界内 `add_perception` 两次后抛错 ⇒ 库内 0 节点）。③ 嵌套调用安全：
        `_tx()` 在 `conn.in_transaction` 为真时降级为 SAVEPOINT（实测已在
        `store._tx()` 内调用本方法不报 `cannot start a transaction within a
        transaction`，`tests/test_issue665_nested_tx_savepoint.py` 同面）。
        边界（削掉的判别力，如实标注）：① 事务回滚只覆盖**库内**副作用——进程内
        计数（`self._interaction_count`）与「本轮已计数」等内存态不随之回退
        （既有语义，非本件引入；本件只消「半截落库」）。

        #139-G：查询仍有 500 个后代、64 层的预算。删除前逐批检查枚举集合是否
        对 hierarchical 入边闭合；任何漏收的孩子都使本次替换整体拒绝，避免把
        查询截断误当成完整子树而静默留下孤儿。沿用设计稿 S-4=(b) 的有界拒绝
        形状，不把只读 traverse 改成无界全图扫描；完整且恰好到边界的子树仍可替换。
        """
        # 范围、后代集合与保护判定也须处于同一写快照。只包住挂入/删除时，
        # 另一实例可在预检后保护或移走旧子树，导致部分替换或删除范围外节点。
        # 入口事务先取写锁，再预检；已有调用方事务仍保留其提交/回滚权。
        import json as _json
        # ---- A 前置校验（先判后动）：任一条不满足即整体拒绝，零改动 ----
        if self.store.get_node(parent_id) is None:
            raise ValueError(f"subgraph_replace 拒绝：parent_id 不存在（{parent_id}）")
        if self.store.get_node(old_sub_root_id) is None:
            raise ValueError(
                f"subgraph_replace 拒绝：old_sub_root_id 不存在（{old_sub_root_id}）")
        if old_sub_root_id == parent_id:
            raise ValueError(
                "subgraph_replace 拒绝（范围校验）：old_sub_root_id 即 parent_id，"
                "不得以父节点自身作旧子根")
        parent_desc = self.store.traverse(parent_id, relation_types=["hierarchical"],
                                          direction="in", max_depth=64)
        if old_sub_root_id not in {r["node_id"] for r in parent_desc}:
            raise ValueError(
                f"subgraph_replace 拒绝（范围校验）：old_sub_root_id={old_sub_root_id} "
                f"不是 parent_id={parent_id} 的 hierarchical 后代")

        def _validate(node, path="root"):
            if not isinstance(node, dict) or not node.get("id"):
                raise ValueError(f"subgraph_replace 拒绝：新子图非法——{path} 缺 'id'")
            subs = (node.get("subgraph") or {}).get("nodes") or []
            for i, sub in enumerate(subs):
                _validate(sub, f"{path}.subgraph.nodes[{i}]")

        _validate(new_subtree)
        # ① 收集旧子树全部后代（in 方向=子指向父，含自身）
        old_desc = self.store.traverse(old_sub_root_id,
                                       relation_types=["hierarchical"],
                                       direction="in", max_depth=64)
        removed = [old_sub_root_id] + [r["node_id"] for r in old_desc]
        # 返回条数/最大深度等于预算并不能证明截断；查边界是否还有未收集的孩子。
        # 所有已收节点都要查：宽图可能在较浅层就耗尽预算，漏点不只在第 64 层。
        # 判定仍在入口写快照内，且发生在任何新子树写入或旧节点删除之前。
        removed_set = set(removed)
        for i in range(0, len(removed), _SQL_IN_CHUNK):
            chunk = removed[i:i + _SQL_IN_CHUNK]
            ph = ",".join("?" * len(chunk))
            for row in self.store.conn.execute(
                    f"SELECT source_id FROM edges WHERE target_id IN ({ph})"
                    " AND relation_type=?", [*chunk, "hierarchical"]):
                if row[0] not in removed_set:
                    raise ValueError(
                        "subgraph_replace 拒绝（旧子树不完整）：遍历预算或过滤遗漏了"
                        " hierarchical 后代；未修改旧树，请缩小替换范围")
        # 可删性预检：任一不可删 ⇒ 整体拒绝（S-2=(a) 全有或全无），不做部分替换
        protected = set(self.store.get_protected_nodes())
        blocked = [nid for nid in removed
                   if (lambda n: n is None or n.layer in self.store.IMMUTABLE_LAYERS
                       or self.store._is_forget_protected(n, protected))(
                       self.store.get_node(nid))]
        if blocked:
            raise PermissionError(
                f"subgraph_replace 拒绝（保护/不可变层）：旧子树含不可删节点 "
                f"{blocked}（整体拒绝，不做部分替换）")
        # ② 递归挂入新子树（先挂后删）
        counter = {"n": 0}

        def _attach(node: Dict, parent: str = None):
            cs = node.get("condition_space", {})
            coord = node.get("spatial_coordinates", {}).get("image2d", [0, 0])
            n = self.add_perception(
                content=_json.dumps(node.get("content", {}), ensure_ascii=False),
                modality=node.get("modality", "visual"),
                spatial_coordinates={"image2d_x": float(coord[0]) if coord else 0.0,
                                     "image2d_y": float(coord[1]) if len(coord) > 1 else 0.0},
                importance=float(node.get("importance", 0.5)),
                tags=["subgraph_replace", "img:" + node["id"]],
                entities=[node["id"]], skip_dedup=True)
            counter["n"] += 1
            nid = n.id
            if parent:
                self.add_edge(source_id=nid, target_id=parent,
                              relation_type=EdgeType.HIERARCHICAL,
                              confidence=float(node.get("confidence", 0.9)),
                              source_evidence="subgraph_replace")
            for sub in node.get("subgraph", {}).get("nodes", []):
                _attach(sub, nid)
            return nid

        # ②③ 单一事务边界（#139 追评）：递归挂入新子树 ＋ 移除旧子树收进**同一**
        # 事务。任一步抛错（含 `_attach` 中途的字段类型 `ValueError`/`TypeError`
        # 与库级异常）⇒ 整体回滚、零残留、旧子树完好 ⇒ 「抛错 = 没改动」成立，
        # 调用方重试不再多挂半棵。`_tx()` 具备嵌套感知：已在调用方事务中时只开
        # SAVEPOINT（不越权提交），见 `_commit_locked` / `_tx` 的 #665 说明。
        with self.store._tx():
            new_root_id = _attach(new_subtree, parent_id)
            # ③ 移除旧子树节点（关联边随删除级联清理）——挂入成功后执行；计数如实
            deleted = sum(1 for nid in removed if self.store.delete_node(nid))
        return {"new_root_id": new_root_id, "new_nodes": counter["n"],
                "removed_nodes": deleted}

    def mark_rejected_path_consumed(self, rejected_id: str, designer_key: str = None):
        self.store.mark_rejected_path_consumed(rejected_id, designer_key=designer_key)

    # ---- A-2 验证器可演进（四步制衡路径） ----

    def get_verifier_config(self) -> Dict:
        return dict(self._verifier_config)

    def propose_verifier_standard(self, name: str, param: str, value: float,
                                  reason: str, proposer: str) -> str:
        """A-2 第①步：反思单元提案（验证单元不得自行迭代验证标准）"""
        return self.store.add_verifier_standard(name, param, value, reason, proposer)

    def review_verifier_standard(self, vid: str, reviewer: str, approved: bool) -> bool:
        """A-2 第②步：独立复核（验证单元自我回避，由另一验证实例或外部校准者执行）"""
        return self.store.review_verifier_standard(vid, reviewer, approved)

    def cs_review_verifier_standard(self, vid: str, reviewer: str, approved: bool) -> bool:
        """A-2 第③步：条件空间复核（3.3.1 节，确认未违反通用价值观）"""
        return self.store.cs_review_verifier_standard(vid, reviewer, approved)

    def adjudicate_verifier_standard(self, vid: str, adjudicator: str, approved: bool,
                                     designer_key: str = None) -> bool:
        """A-2 终裁：裁决、持久配置与历史一起提交，失败时内存配置一并回退。

        #118/#153：应用配置走唯一真源 `_apply_verifier_standard`（含 dedup_static 值域
        钳制）。#429：裁决、`engine_meta` 持久化与留史收进同一工作单元；任一步失败 ⇒
        盘上回滚**且**内存配置回退（否则「留史失败」会留下一个盘上没有、内存里生效的
        配置——正是本件要消的不一致）。

        ★本笔（治理写面「同键两套口径」收口 · P2）：`approved` 分支在**改内存配置与
        写 `engine_meta` 之前**补 `store._assert_meta_key_writable("verifier_config")`
        ——`verifier_config` ∈ `ENGINE_META_GUARDED_KEYS`，非 PRIMARY（且非 `:memory:`
        自有库）⇒ `PermissionError`。判据复用 `#768` 的既有单点（不另造第二份）。
        边界（如实标注）：本闸只判「实例角色 × 库归属」，不判 `designer_key` 的持有
        者身份——A-2 的密钥闸取自进程环境变量，同部署下 SUB 天然持有（`#586` 已就
        同一理由判定「密钥闸不构成角色隔离」），故本闸是本写点唯一的角色隔离面。

        #130（路 B · 裁定帖原文）：「`_terminal_judgment` 与 `adjudicate_verifier_standard`
        按 `ESC-00x` 直查升级点，不再走子串匹配」。本方法即那两处调用点之一——`approved`
        分支按 `ESC-005`（「验证标准终裁 / 验证标准变更四步制衡第④步」，与本方法同面）
        **直查**（`store.get_escalation_point_by_code`，`WHERE code=?` 精确等值，不经
        `check_escalation` 的双向包含面），并把该升级点的 id/severity 记进结构记录
        metadata。★**只记录不阻断**：升级点缺席 ⇒ 记 `None`、不抛；`enabled` 不参与
        判定；终裁的返回契约与事务边界**逐字不变**。
        """
        # 提交与发布都持锁，后续终裁才能基于刚提交的配置继续修改。
        if self.store.conn.in_transaction:
            raise RuntimeError("终裁配置需要独立事务；请先结束调用方事务")
        with self.store._tx(), self.store._lock:
            before_config = dict(self._verifier_config)
            before_dedup = self._dedup_static
            try:
                with self.store._structure_event_transaction() as cursor:
                    result = self.store.adjudicate_verifier_standard(
                        vid, adjudicator, approved, designer_key=designer_key, _cursor=cursor)
                    if not result:
                        return False
                    if result["status"] == "approved":
                        # ★本笔（治理写面「同键两套口径」收口 · P2）：本方法此前
                        # 自开一句 `INSERT OR REPLACE INTO engine_meta` 写治理键
                        # `verifier_config` **无角色闸** ⇒ SUB 走完 A-2 四步即把
                        # `dedup_static` 写穿库（实测 0.85 → 0.5，PRIMARY 重启回读
                        # 生效）；而**同一键**经 `store.set_meta` 必被 `#768` 单点
                        # 拒（实测 `PermissionError`）⇒ 同键两套口径。闸取 `store`
                        # 侧既有单点，落在改内存配置与 INSERT **之前**（被拒时内存
                        # 配置由下方 `except BaseException` 一并回退，零分叉）。
                        self.store._assert_meta_key_writable("verifier_config")
                        self._apply_verifier_standard(result["param"], result["value"])
                        cursor.execute(
                            "INSERT OR REPLACE INTO engine_meta(key,value) VALUES (?,?)",
                            ("verifier_config", json.dumps(dict(self._verifier_config))))
                        record = (f"{result['param']}="
                                  f"{self._verifier_config.get(result['param'], result['value'])}"
                                  f" 终裁通过（{adjudicator}）")
                        # #130（路 B · 裁定帖原文）：本调用点此前与升级点表**零接触**
                        # ——「验证标准变更四步制衡第④步」正是内置 `ESC-005` 的
                        # condition，却从无代码按 code 去查它（`check_escalation` 的
                        # 子串面也够不着：调用方手里的是 vid/param，不是信号文本）。
                        # 现按 `ESC-00x` **直查**（精确、可审计、不受分词影响），
                        # 把该升级点的 id/severity 记进本次结构记录 metadata ⇒
                        # 「终裁发生了、它对应哪条升级点」成为库内可查的事实。
                        # ★**只记录不阻断**：不因升级点缺席/被停用而改变终裁行为
                        # （缺席 ⇒ 记 None，不抛；`enabled` 不参与判定）。
                        _esc = self.store.get_escalation_point_by_code("ESC-005")
                        _meta = {"standard_id": vid, "param": result["param"],
                                 "value": self._verifier_config.get(result["param"],
                                                                    result["value"]),
                                 "escalation_code": "ESC-005",
                                 "escalation_id": _esc["id"] if _esc else None,
                                 "escalation_severity": _esc["severity"] if _esc else None}
                        self._write_structure_record(
                            record, "verifier_standard", source=adjudicator,
                            metadata=_meta,
                            _cursor=cursor)
                # 真提交仍在锁内与补偿 try 内；提交失败必须恢复运行态。
                self.store._commit_locked()
            except BaseException:
                # 盘上已随工作单元回滚；内存配置必须一并回退（见 docstring）。
                self._verifier_config.clear()
                self._verifier_config.update(before_config)
                self._dedup_static = before_dedup
                raise

        return True


    def _apply_verifier_standard(self, param: str, value) -> None:
        """应用一条验证标准到运行配置（`dedup_static` 走值域钳制）。

        #118：钳制口径与公开入口 `set_dedup_config` 一致（判据来源见
        `adjudicate_verifier_standard` docstring）。#153：本方法同时供
        `_replay_verifier_standards`（启动回放）复用——两处口径唯一，避免
        「终裁路径钳制、回放路径不钳制」的新分叉。
        """
        if param == "dedup_static":
            value = min(0.95, max(0.5, float(value)))
            self._dedup_static = value
        self._verifier_config[param] = value
        if param == "deviation_threshold" and self._cognition:
            self._cognition.deviation_threshold = value

    def _replay_verifier_standards(self) -> int:
        """启动时回放**已终裁通过**的验证标准（lingshu #153），返回回放条数。

        缺陷：`__init__` 把 `_verifier_config` 写死为默认值
        （`{"dedup_static": 0.85, "deviation_threshold": 0.3}`），而终裁结果仅存于
        内存 ⇒ **重启即失效**（盘上 `verifier_standards` 的 `approved` 行仍在，
        却不再生效）。修法：启动时按 `decided_at` 升序回放 approved 行（后批覆盖
        先批），走与终裁同一应用/钳制口径 `_apply_verifier_standard`。
        判据来源：issue #153 报告口径；仓内理论章节追不到，属**经验标定**（本件）。
        边界（削掉的判别力）：只回放 `status='approved'` 的行——`denied`/
        `cs_rejected` 等终态不回放（其语义是「未通过」）；**不**校验行内 value 与
        盘上结构记录是否一致（本件只保证「终裁通过 ⇒ 重启后仍生效」）。

        PR #429 合并保留：本方法（main 的 #153 实现）与 PR #429 新增的
        `_restore_structure_configuration()`（从 `engine_meta` 读回已生效配置）
        同时存在：本方法在后者的**前**执行，故 engine_meta 中更新的读数仍胜出；
        而 engine_meta 缺失时（旧库）本方法照旧把 #153 的保证补齐。
        """
        rows = [r for r in self.store.list_verifier_standards("approved")
                if r.get("param")]
        rows.sort(key=lambda r: (r.get("decided_at") or 0.0))
        for r in rows:
            self._apply_verifier_standard(r["param"], r["value"])
        return len(rows)
    def list_verifier_standards(self, status: str = None) -> List[Dict]:
        return self.store.list_verifier_standards(status)

    # ---- A-3 升级点清单化 ----

    def _seed_escalation_points(self):
        """A-3 默认升级点（对齐 3.15/3.16/12.1；按 code 校验**补齐**）

        #749：播种走 `store._insert_escalation_point` **裸写**，不经带
        `verify_designer` 闸的 `store.add_escalation_point`——6 条是源码常量、
        属构造期初始化（`__init__` 调用本方法），不能要求部署方为「程序自带的
        默认值」配置密钥。守卫 `tests/test_issue749_escalation_write_gate.py`
        的 E 组锁住这条分离（改回带闸方法 ⇒ 未配置密钥的构造即抛 ⇒ 红）。

        #130（路 B · 裁定帖原文）：「`_seed_escalation_points` 由『仅空表播种』
        改为『按 code 校验补齐』：表非空但**缺某条内置 code** 时须补上该条；
        已有 code 的行**不动**（不覆盖、不改写）」。旧口径（`if 表非空: return`）
        把「表非空」当作「播种已完成」的证据，而两者不等价：任何一次运行期
        `add_escalation_point`（或 `import_all` 回灌新 id 行）都会让表非空，此后
        内置 6 条**永久补不齐**——危机响应面上静默缺格，且无任何信号。
        新口径按**内置 code 集合**逐条校验（判据＝本方法内的 `seeds` 常量，
        与表内容无关），缺哪条补哪条。

        边界（削掉的判别力，如实标注）：
        ① 只判「内置 code 在不在」，**不**校验已有行的 trigger/condition/action/
           severity 是否被改写（裁定帖明写「已有 code 的行不动」，故**不**回写、
           不「修正」——即便某行被改成了别的语义，本方法也不覆盖）；
        ② 同 code 重复行（`add_escalation_point` 不禁止）视为「该 code 在场」，
           不删除、不合并（去重属另一条面）；
        ③ `enabled` 状态不在校验面内：内置 code 存在但被停用 ⇒ 视为在场、
           不重新启用（启停是治理动作，构造期无权代作）。
        """
        # #130：判据取「**按 code 直查**已播种集合」而非「表是否非空」。
        # 走 `list_escalation_points` 一次取回（只读、常数代价：内置 6 行 +
        # 运行期新增行），不逐条发 6 次点查。
        present = {(p.get("code") or "") for p in
                   self.store.list_escalation_points(enabled_only=False)}
        seeds = [
            ("ESC-001", "自维持迹象", "自主目标生成/存在危机感知", "立即上报维生系统", "high"),
            ("ESC-002", "P0保护触发", "结构威胁/信任崩溃", "维生系统终裁：冻结/回滚/隔离", "high"),
            ("ESC-003", "价值观冲突", "反思单元分歧持续≥3轮", "维生系统终裁（3.16）", "medium"),
            ("ESC-004", "交叉验证偏差", "REFLECT-CROSS-VALIDATION deviation>0.3", "维生系统终裁", "medium"),
            ("ESC-005", "验证标准终裁", "验证标准变更四步制衡第④步", "维生系统终裁", "medium"),
            ("ESC-006", "设计者预备变更", "设计者位置/权责分离相关", "维生系统终裁（1.4.3）", "high"),
        ]
        for code, trigger, condition, action, severity in seeds:
            if code in present:
                continue          # 已有 code 的行**不动**（不覆盖、不改写）
            self.store._insert_escalation_point(code, trigger, condition, action,
                                                severity)

    def list_escalation_points(self, enabled_only: bool = True) -> List[Dict]:
        return self.store.list_escalation_points(enabled_only)

    def get_escalation_point_by_code(self, code: str) -> Optional[Dict]:
        """按 `ESC-00x` 直查一条升级点（#130 路 B · 引擎门面，透传 store 单点）。"""
        return self.store.get_escalation_point_by_code(code)

    def list_designer_todos(self, status: str = None) -> List[Dict]:
        """列「待设计者裁决」待办（#130 路 B · **只读**列举面）。

        ★**消费方尚未接线**（如实标注）：本方法只列举、不推进状态——仓内**没有**
        任何位置读 `designer_todos` 并把 `status` 由 `open` 推进到 `consumed`
        （`consumed_at` 因此恒为 NULL）。推进方向按 #130 裁定帖对齐本仓既有的设计者
        终裁唯一合规通路 `propose → verify → adjudicate`；**具体消费点与状态推进
        语义属未决语义题**（裁定帖「存疑」节自陈未给出），故本件只提供这一只读接口，
        不擅自接线（见 `LayeredStore.add_designer_todo` docstring）。
        """
        return self.store.list_designer_todos(status)

    def add_escalation_point(self, code: str, trigger: str, condition: str,
                             action: str, severity: str = "medium",
                             designer_key: str = None) -> str:
        return self.store.add_escalation_point(code, trigger, condition, action,
                                               severity, designer_key=designer_key)

    def set_escalation_enabled(self, escalation_id: str, enabled: bool,
                               designer_key: str = None):
        self.store.set_escalation_enabled(escalation_id, enabled,
                                          designer_key=designer_key)

    def check_escalation(self, signal_type: str, value: float = None) -> List[Dict]:
        """A-3：信号 → 升级点匹配（何种信号必须提交维生系统）。

        issue #130 匹配修正（原判据 `signal_type in point["trigger"] or
        signal_type in point["condition"]` 方向反了）：
        - **空串/纯空白短路**返回 `[]`：`"" in <任意串>` 恒真 ⇒ 修前命中全部；
        - **双向包含**：触发词/条件词出现在信号里（本意，修前含词长句
          如「关于自维持迹象的报告」命中 0），同时保留「信号 token 出现在
          触发词/条件里」（既有调用方以 token 如 `deviation` 作信号，纯单向
          翻转会让其从命中变不命中）。

        #57（本件）：`value` 参数此前在函数体内**零引用**（声明的契约不兑现）
        ——`condition` 若写数值界（如 `"value > 0.9"`），0.05 与 0.99 会给出
        **完全相同**的匹配。现让 `value` 真正参与判定：
        - 仅当 `condition` 能解析出 **`value <op> <number>`** 形式的数值界
          （`_parse_escalation_value_bound`，**手写解析器、不用 `eval()`**）时，
          在「信号↔触发/条件文本包含」命中之后**追加**该数值界判定；
        - 数值界存在而 `value is None` ⇒ 该点**不匹配**（无值不可判）；
        - 解析不出数值界（如内置 `ESC-004` 的 `"...deviation>0.3"` 左操作数是
          `deviation` 而非 `value`）⇒ **回落旧行为**（仅按字符串包含匹配，
          `value` 不参与）——保住既有 token 用法不回归。
        判据来源：公开 issue #57（公开 API 声明参数零引用）＋本文件既有
        `test_issue130_escalation_gate` 的匹配口径（不放松）。
        边界（削掉的判别力）：①只认左操作数**显式为参数名 `value`** 的比较，
          其它标识符（如 `deviation`）不参与数值求值；②算符只支持
          `>`/`>=`/`<`/`<=`/`==`/`!=`，复合布尔表达式不解析（回落包含匹配）。
        同型面（如实标注）：全仓「签名声明参数、函数体零引用」同型共 **29 处**
        （AST 扫描 `lingshu/**/*.py`），本件只修有实害的 3 处（本函数、
        `export_structured_trajectories`、`nn/stcnn.py:detect_period`），
        其余 26 处**不在本件范围**、未动。
        """
        signal = (signal_type or "").strip()
        if not signal:
            return []
        matches = []
        for point in self.store.list_escalation_points(enabled_only=True):
            trigger, condition = point["trigger"], point["condition"]
            # #656：NULL 经 `or ""` 归一后，`"" in signal` 恒真 ⇒ 空侧不得参与匹配
            # （fail-closed，与函数开头 `if not signal: return []` 同口径）。
            if not ((trigger and (signal in trigger or trigger in signal))
                    or (condition and (signal in condition or condition in signal))):
                continue
            bound = _parse_escalation_value_bound(condition)
            if bound is not None:
                if value is None:
                    continue          # 无值不可判：声明了数值界却未给值 ⇒ 不匹配
                if not _eval_escalation_value_bound(bound[0], bound[1], value):
                    continue
            matches.append(point)
        return matches

    # ---- A-4 信息差收敛速率指标（独立观测层 · DEVIATION-004） ----

    @_transactional
    def record_info_gap(self, d_norm: float = None, trust_complement: float = None,
                        behavior_deviation: float = None, connection_drift: float = None,
                        prediction_error: float = None) -> float:
        """A-4：记录 D_norm 样本（2.7 节四维度加权；与资源指标独立采集）"""
        if d_norm is None and all(v is not None for v in
                                  (trust_complement, behavior_deviation,
                                   connection_drift, prediction_error)):
            d_norm = (0.30 * trust_complement + 0.25 * behavior_deviation +
                      0.30 * connection_drift + 0.15 * prediction_error)
        if d_norm is None:
            return 0.0
        # #459：非有限（NaN/±Inf）不可判 ⇒ 归钳制区间**下界** 0.0（#402 口径：
        # 非有限一律回落下界，不得抬到上界伪装成「最大信息差」——旧实现
        # `min(1.0, nan)` 返回 1.0，NaN 样本进 gap_history 后被 Δ²D/趋势当满值）。
        if not math.isfinite(d_norm):
            d_norm = 0.0
        dn = max(0.0, min(1.0, d_norm))
        self._gap_history.append({"ts": time.time(), "d_norm": dn})
        if len(self._gap_history) > 100:
            self._gap_history = self._gap_history[-100:]
        # OBS-REV1 / 钉死批条款4：D 序列落库（跨进程 d² 可回放），尽力而为不阻断
        try:
            sc = self.store.conn.cursor()
            sc.execute("INSERT INTO gap_history (ts, d_norm) VALUES (?,?)",
                       (self._gap_history[-1]["ts"], dn))
            sc.execute("DELETE FROM gap_history WHERE id NOT IN"
                       " (SELECT id FROM gap_history ORDER BY id DESC LIMIT 2000)")
            self._commit_locked()
        except Exception:
            pass
        return dn

    def _load_gap_history(self) -> List[Dict]:
        """D 序列跨进程恢复（OBS-REV1 / 钉死批条款4）：读库尾 100 条填内存
        窗口，重启后 d²/趋势不断链；空表或旧库无表时返回空列表。"""
        try:
            c = self.store.conn.cursor()
            c.execute("SELECT ts, d_norm FROM gap_history ORDER BY id DESC LIMIT 100")
            rows = c.fetchall()
            return [{"ts": ts, "d_norm": dn} for ts, dn in reversed(rows)]
        except Exception:
            return []

    def get_gap_trend(self, window: int = 30) -> Dict:
        """A-4：信息差收敛趋势（斜率=线性回归；独立于资源指标）"""
        samples = self._gap_history[-window:]
        if len(samples) < 2:
            return {"current": samples[-1]["d_norm"] if samples else None,
                    "mean": None, "std": None, "slope": None, "trend": "insufficient"}
        vals = [s["d_norm"] for s in samples]
        n = len(vals)
        mean = sum(vals) / n
        std = (sum((v - mean) ** 2 for v in vals) / n) ** 0.5
        xs = list(range(n))
        x_mean = (n - 1) / 2
        denom = sum((x - x_mean) ** 2 for x in xs) or 1.0
        slope = sum((x - x_mean) * (v - mean) for x, v in zip(xs, vals)) / denom
        trend = "narrowing" if slope < -0.005 else ("widening" if slope > 0.005 else "stable")
        return {"current": vals[-1], "mean": round(mean, 4), "std": round(std, 4),
                "slope": round(slope, 4), "trend": trend}

    def record_resource_usage(self, tokens: int = None, seconds: float = None) -> Dict:
        """A-4：资源效率指标（独立于 D_norm 采集）"""
        self._resource_history.append({"ts": time.time(), "tokens": tokens, "seconds": seconds})
        if len(self._resource_history) > 100:
            self._resource_history = self._resource_history[-100:]
        return self._resource_history[-1]

    def get_resource_metrics(self, window: int = 30) -> Dict:
        """A-4：资源指标摘要（与信息差趋势分离）"""
        samples = self._resource_history[-window:]
        if not samples:
            return {"count": 0}
        tokens = [s["tokens"] for s in samples if s["tokens"] is not None]
        seconds = [s["seconds"] for s in samples if s["seconds"] is not None]
        return {"count": len(samples),
                "mean_tokens": round(sum(tokens) / len(tokens), 1) if tokens else None,
                "mean_seconds": round(sum(seconds) / len(seconds), 3) if seconds else None}

    # ---- A-5 结构化轨迹输出 ----

    def export_trajectory(self, cycle_id: str = None, output_path: str = None,
                          _limit: int = None) -> Dict:
        """A-5：导出认知闭环的结构化轨迹（附录13 扩展 · 训练资产格式 · 有损投影）。

        `_limit`（#57，内部参数，供 `export_structured_trajectories` 用）：`cycle_id`
        为空（批量模式）时，按节点 `temporal_coordinate` **降序**（最近在前）取前
        `_limit` 条；`_limit is None` 时保持旧行为（`get_nodes_by_tag` 的
        importance/last_access 排序、硬上限 10）。返回体形状不变（`{"cycles": [...]}`
        ＋ `meta`）。判据来源：公开 issue #57（`export_structured_trajectories` 的
        `limit` 声明了却不生效）＋ 本方法 docstring「最近 N 条」。

        issue #698（静默覆盖）：`output_path` 给出时本方法 `open(..., "w")` **直写**
        该路径，目标已存在即**覆盖**（无根闸、无拒覆盖、无原子写——#330 只改了
        `export_all` 的落盘原语，未及本方法）。`output_path` 由**调用方**给出，但
        **会过** `_confine_export_out_path` 根闸（#746）：未显式配置
        `LINGSHU_EXPORT_OUT_ROOT` 时**不限制**（保持既有行为）；显式配置后越界即
        fail-closed 抛 `ValueError`（与 `_confine_simloop_out_path` /
        `_confine_world3d_out_path` 同三条口径；但那两个闸的适用面是 world3d render /
        wm_simloop flush 的受保护产物，本闸专收本方法与 `export_all`）。本件只把
        「**静默**覆盖」改为**显式上报**：
        **仅当 `output_path` 给出**（即真发生落盘）时，返回值在落盘内容之上**多带**
        一个键 `overwrote_existing`（bool，读数取在 `open(..., "w")` **之前**）；
        批量模式（`output_path is None`，无落盘）**不加键**，返回体形状与 #57 契约
        逐字一致。**落盘文件内容形状不变**（`overwrote_existing` 只走返回值，
        不写进文件）。
        边界（如实标注）：该键是「判定—写入」之间的一次 `os.path.exists` 读数，
        只要求「上报」，**不**提供排他创建（存在竞态窗口）；也不改本方法
        `open(..., "w")` 的直写原语（#330 面，本件未动）。
        当前连接的 SQLite 数据库及 WAL/SHM/journal 不可作为输出路径；
        源库保护同 `export_all`，普通轨迹文件仍可覆盖。
        """
        if cycle_id:
            nodes = self.store.get_nodes_by_tag(f"cycle:{cycle_id}", limit=20)
            full = [n for n in nodes if "cycle_record_full" in n.tags]
            trajectory = None
            if full:
                try:
                    trajectory = json.loads(full[0].content)
                except Exception:
                    trajectory = None
            if trajectory is None:
                trajectory = {"cycle_id": cycle_id,
                              "nodes": [n.content[:100] for n in nodes]}
        else:
            if _limit is None:
                records = self.store.get_nodes_by_tag("cycle_record_full", limit=10)
            else:
                # 取全部再按时间降序截断，保证「最近 N 条」是**全库最近**而非
                # 「importance 前 N 里最近的」。
                records = self.store.get_nodes_by_tag("cycle_record_full", limit=-1)
                records.sort(key=lambda n: n.temporal_coordinate or 0.0,
                             reverse=True)
                if _limit >= 0:
                    records = records[:_limit]
            trajectory = {"cycles": []}
            for n in records:
                try:
                    trajectory["cycles"].append(json.loads(n.content))
                except Exception:
                    pass
        trajectory.setdefault("meta", {
            "exported_at": time.time(),
            "engine_version": "v1.5",
            "condition_space": "轨迹导出 · 训练资产格式（有损投影）",
        })
        if output_path:
            # #746：与 `export_all` 同款白名单根闸（越界 fail-closed）。本方法
            # 此前不过根闸（#698 只把「静默覆盖」改为显式上报，未及根判定）。
            output_path = self._confine_export_out_path(output_path)
            self._guard_export_source_path(output_path)
            # #698：覆盖已存在目标 ⇒ 显式上报。读数取在写入**之前**（写后目标必在）。
            # 只进返回值、不进文件：落盘内容形状与 #57 契约逐字不变。
            overwrote_existing = os.path.exists(output_path)
            # #802：落盘分支写出前剥除 BiDi 控制/零宽字符（与 `export_all`
            # 同款；批量模式不落盘、返回体不动——见 `_strip_export_controls`
            # 处注释边界④）。`reported` 取剥除后的树，与文件内容同源。
            trajectory = self._sanitize_export_obj(trajectory)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(trajectory, f, ensure_ascii=False, indent=2)
            reported = dict(trajectory)
            reported["overwrote_existing"] = overwrote_existing
            return reported
        return trajectory

    def export_structured_trajectories(self, limit: int = 10, output_path: str = None) -> Dict:
        """A-5：批量导出（最近 N 条轨迹）。

        #57（本件）：`limit` 此前**零引用**——旧实现直接
        `return self.export_trajectory(None, output_path)`，docstring 承诺的
        「最近 N 条」不成立（`limit=1` 与 `limit=100` 返回完全相同的 10 条）。
        现把 `limit` 透传给 `export_trajectory` 的 `_limit`，按
        `temporal_coordinate` 降序取**全库最近 N 条**（`limit<0` ⇒ 不截断；
        `limit=0` ⇒ 空列表）。返回体形状不变。
        判据来源：公开 issue #57（声明参数零引用）＋本方法 docstring「最近 N 条」。
        同型面（如实标注）：全仓同型共 29 处，本件只修有实害的 3 处。
        """
        return self.export_trajectory(None, output_path, _limit=limit)

    # ==================== v1.3 认知接口（P0-1/P0-2/P0-3） ====================

    def get_entity_context(self, entity_id: str, limit: int = 50) -> Dict:
        """P0-1：以实体为中心组装上下文（实体信息 + 关联节点）"""
        if not self.entity_registry:
            return {"entity": None, "nodes": [], "v13_ready": self._v13_ready}
        return self.entity_registry.get_entity_context(entity_id, limit=limit)

    def consolidate_learning_result(self, record: Dict) -> Optional[STNode]:
        """P0-2/P0-3：学习结果固化（知识层 + 实体链接）"""
        if not record or not record.get("summary"):
            return None
        entities = record.get("entities", [])
        return self.add_perception(
            content=f"[learning] {record['summary']}",
            importance=0.8,
            tags=["learning_result"] + [f"ent:{e}" for e in entities],
            entities=entities)

    def get_next_candidate(self) -> Optional[Dict]:
        """P0-3：下一个待探索盲区（委托学习闭环）"""
        if not self._learning_loop:
            return None
        return self._learning_loop.get_next_candidate()

    def run_learning_cycle(self, input_signal: str, context: Dict = None) -> Dict:
        """P0-2：执行一轮认知闭环（委托编排器，3.10 有损投影）"""
        if not self._cognition:
            return {"status": "v13_not_ready", "error": self._v13_error}
        return self._cognition.learning_cycle(input_signal, context)

    def learn_next(self, force: bool = False, use_prediction: bool = True) -> Dict:
        """P0-3：盲区学习一步（委托闭环，第零定律操作化）
        v1.12 P0-3：avoiding 情绪状态 → 探索预算下调（巩固优先）；force=True 可覆盖
        v1.15：use_prediction 转发给学习闭环（预测×盲区联动，与 api.learn 签名对齐）"""
        if not self._learning_loop:
            return {"status": "v13_not_ready", "error": self._v13_error}
        if not force and self._self_cognition is not None:
            bias = self._self_cognition.get_emotional_bias()
            if bias["status"] == "avoiding":
                return {"status": "consolidation_priority",
                        "note": "情绪方向性偏好 avoiding → 探索预算下调（P0-3）",
                        "budget": self._self_cognition.exploration_budget()}
        result = self._learning_loop.learn_next(use_prediction=use_prediction)
        self._note_action("learn", str(result.get("status", ""))[:80],
                          None, {"result": str(result)[:120]})
        return result

    # ==================== v1.4 认知增强（P1-1 ~ P1-4） ====================

    def induce_concepts(self, min_cluster_size: int = 3, similarity_threshold: float = 0.6,
                        max_concepts: int = 5,
                        condition_space: ConditionSpace = None) -> List[STNode]:
        """P1-1 归纳/知识合成：并查集单链式聚类 → 生成概念节点（SIMILAR 边，未验证待复核）"""
        protected = set(self.store.get_protected_nodes())
        nodes = [n for n in self.store.query_nodes(layer=MemoryLayer.KNOWLEDGE, limit=300)
                 if not any(t in n.tags for t in ("induced", "concept", "cluster_member"))
                 and n.id not in protected and "no_forget" not in n.tags]
        if len(nodes) < min_cluster_size:
            return []
        # 单链式聚类（并查集）：对种子顺序鲁棒（n2~n3 < 阈值但经 n1 传递连通时仍成簇）
        parent = {n.id: n.id for n in nodes}
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
        # 倒排预筛（外部用户 514+ 节点超时修正）：O(n²) 全对比 → 仅共享低频二元组的候选对
        # 做精确 Jaccard；高频 bigram（DF>半数，如「的了」类）区分度低不进倒排；
        # 另用长度预筛（Jaccard≥t 要求 min|A| ≥ t·max|A|，不满足直接跳过）
        # #166：预筛**不得**把「全池同主题」的节点整批丢掉——见下方候选集回退。
        from collections import defaultdict as _dd
        import re as _re_bg  # 就地内联（白箱化：原借自私有侧 LLM 路由模块）

        def _bg(text):
            _t = _re_bg.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text or "")
            return {_t[i:i + 2] for i in range(len(_t) - 1)}
        bigram_sets = [_bg(n.content) for n in nodes]
        df = _dd(int)
        for bs in bigram_sets:
            for bg in bs:
                df[bg] += 1
        half = max(1, len(nodes) // 2)
        inverted = _dd(list)
        for idx, bs in enumerate(bigram_sets):
            for bg in bs:
                if df[bg] <= half:
                    inverted[bg].append(idx)
        t = similarity_threshold
        # 时间预算（外部用户 514+ 节点 30s 超时修正）：候选对限时处理，超时即用已完成的
        # 并查集部分成簇（链式聚类增量有效）——induce 永不再触发 MCP 超时
        import time as _time
        deadline = _time.time() + 8.0
        pair_seen = set()
        for idx, bs in enumerate(bigram_sets):
            if _time.time() > deadline:
                break
            if not bs:
                continue
            cands = {k for bg in bs for k in inverted.get(bg, ()) if k != idx}
            if not cands and len(bigram_sets) > 1:
                # #166：单主题池过半时，该节点的 bigram **全部** DF>half（如全池
                # 同主题），倒排索引里没有它的任何条目 ⇒ 旧实现该节点 0 候选、全池
                # 0 候选对 ⇒ `induce_concepts` 恒返回 []（像「无可归纳内容」）。
                # 补一次直接对比（同一 deadline 约束），保住召回。
                # 判据来源：分诊表 `triage_lingshu_ALL.md` 行 166；仓内无规定
                # 倒排预筛召回面的理论章节，追不到更早出处。
                # 边界（削掉的判别力）：回退只在节点「经预筛后无任何候选」时触发；
                # 有候选的节点仍走原预筛（不改变其候选集）。
                cands = set(range(len(bigram_sets))) - {idx}
            for j in cands:
                pair = (idx, j) if idx < j else (j, idx)
                if pair in pair_seen:
                    continue
                pair_seen.add(pair)
                b2 = bigram_sets[j]
                if not b2 or len(bs) < t * len(b2) or len(b2) < t * len(bs):
                    continue
                if LayeredStore.char_bigram_jaccard(nodes[idx].content, nodes[j].content) >= t:
                    union(nodes[idx].id, nodes[j].id)
        groups = {}
        for n in nodes:
            groups.setdefault(find(n.id), []).append(n)
        clusters = [g for g in groups.values() if len(g) >= min_cluster_size]
        concepts = []
        for cluster in clusters[:max_concepts]:
            concept = self.add_perception(
                content=self._summarize_cluster(cluster),
                importance=0.75,
                tags=["concept", "induced", "pending_verification"],
                condition_space=condition_space)
            for m in cluster:
                self.add_edge(concept.id, m.id, EdgeType.SIMILAR, confidence=0.6,
                              source_evidence="inferred")
                self.store.tag_node(m.id, "cluster_member")
            concepts.append(concept)
        return concepts

    def _summarize_cluster(self, cluster: List[STNode]) -> str:
        """规则式摘要（无 LLM，D-005）：代表性节点 + 聚类规模"""
        rep = max(cluster, key=lambda n: n.importance)
        return f"[概念] 由 {len(cluster)} 条记忆归纳（代表：{rep.content[:40]}…）"

    # ---- P1-2 时间维度查询 ----

    def what_happened_at(self, timestamp: float, tolerance: float = 60.0,
                         limit: int = 20) -> List[Tuple[STNode, float]]:
        """P1-2：查询特定时间附近发生了什么

        lingshu #179：旧实现 `query_nodes(limit=500)` 再在 Python 里按时间过滤——
        取到的是「最重要的 500 条」，库内超过 500 条时低重要度/较早的事件**查不到**。
        现改为在 SQL 侧按时间窗取数（`query_nodes_by_time`），再按时间距离取最近的
        `limit` 条。
        判据来源：分诊表 `triage_lingshu_ALL.md` 行 179（对照 `get_timeline` 走 SQL
        时间过滤能查到）。削掉的判别力：无——返回语义（时间距离最近的前 `limit` 条）
        不变，仅把候选集从「重要性前 500」换成「时间窗内全部」。
        """
        nodes = self.store.query_nodes_by_time(timestamp - tolerance,
                                               timestamp + tolerance)
        results = [(n, abs(n.temporal_coordinate - timestamp)) for n in nodes
                   if abs(n.temporal_coordinate - timestamp) <= tolerance]
        results.sort(key=lambda x: x[1])
        return results[:limit]

    def timeline(self, node_id: str, direction: str = "forward",
                 max_depth: int = 20) -> List[STNode]:
        """P1-2：沿时间演化边展开时间线（forward=源→目标，backward=目标→源）"""
        chain = []
        current = node_id
        visited = set()
        for _ in range(max_depth):
            if current in visited or current is None:
                break
            visited.add(current)
            node = self.store.get_node(current)
            if not node:
                break
            chain.append(node)
            edges = (self.store.get_outgoing_edges(current) if direction == "forward"
                     else self.store.get_incoming_edges(current))
            nxt = None
            for edge in sorted(edges, key=lambda e: -e.confidence):
                if edge.relation_type in (EdgeType.SEQUENTIAL, EdgeType.CAUSAL):
                    nxt = edge.target_id if direction == "forward" else edge.source_id
                    break
            current = nxt
        return chain

    def get_timeline(self, start_ts: float = None, end_ts: float = None,
                     layer: MemoryLayer = None, limit: int = 50) -> List[STNode]:
        """P1-2：时间范围查询（自我叙事连续性）"""
        if start_ts is None:
            start_ts = 0.0
        if end_ts is None:
            end_ts = time.time() + 1
        return self.store.get_nodes_in_range(start_ts, end_ts, layer, limit)

    def session_summary(self, window_seconds: float = 3600.0, limit: int = 10) -> List[Dict]:
        """P1-2：会话分组摘要（按时间窗分组）。
        有限性声明（VAL-TRANSFER-20260813-002-FINAL 约束）：本输出为条件空间内的
        叙事压缩（有损投影），非完整自我状态记录（1.7.1 节自我意识的范围与局限）。

        lingshu #179：旧实现 `query_nodes(limit=500)` 只取「最重要的 500 条」再分窗
        ⇒ 库内超过 500 条时较早/低重要度的会话窗整批消失（实测最近窗口只含 500 条、
        遗漏关键事件）。现改按时间全量取数（`query_nodes_by_time`），分窗口径不变。
        判据来源：分诊表 `triage_lingshu_ALL.md` 行 179。
        """
        nodes = self.store.query_nodes_by_time()
        if not nodes:
            return []
        nodes.sort(key=lambda n: n.temporal_coordinate)
        windows = []
        current = []
        current_start = nodes[0].temporal_coordinate
        for n in nodes:
            if n.temporal_coordinate - current_start <= window_seconds:
                current.append(n)
            else:
                windows.append((current_start, current))
                current_start = n.temporal_coordinate
                current = [n]
        if current:
            windows.append((current_start, current))
        result = []
        for start, group in windows[-limit:]:
            result.append({
                "window_start": start,
                "count": len(group),
                "layers": sorted(set(n.layer.value for n in group)),
                "top": [n.content[:40] for n in sorted(group, key=lambda n: -n.importance)[:3]],
            })
        return result

    # ---- P1-3 技能获取闭环 ----

    def record_action_sequence(self, actions: List[str], outcome: str, success: bool,
                               skill_hint: str = None) -> Optional[STNode]:
        """P1-3：记录操作序列（成功 → 技能提取/强化）

        #216（负记忆同价）：失败与成功经验此前硬编码 `importance=0.6 if success
        else 0.4` ⇒ 同一序列的失败经验在 `recall` 里系统性排在成功经验之后，与
        `activation.py:15` 的理论锚点「第五篇：…负记忆同价…」正好相反。现两种结果
        同价（0.5）——成败由 `success`/`failure` 标签承载，不再由 importance 分档。
        判据来源：`lingshu/core/activation.py:15`（理论锚点「负记忆同价」）＋ 公开
        issue #216「建议修复 2：同价」（分诊表行 216；经验标定）。
        边界（削掉的判别力）：本件**不**按结果严重度加权（issue 允许「由同一函数按
        结果严重度计算」；那需先裁定严重度来源，本件不引入）；`recall` 的排序仍会
        用 confidence 等其它信号。

        #216（成功分支读同序列 open 负记忆 · 读侧接线）：失败 N 次的序列一旦成功，
        旧实现**完全不读**负记忆——技能照常提取/强化，调用方从返回值里看不到任何
        失败信息（负记忆表在技能面上「只写不读」）。现成功分支在建/升技能**之前**
        用既有 `find_rejected_paths` 查**同序列**的 open 负记忆
        （`path_type='action_sequence'`），命中即把读数
        `neg_memory_open:<N>` 追加为该节点的标签（节点增量，返回类型仍是
        `Optional[STNode]`——零契约变更）。
        查询串＝`"操作序列失败：" + " > ".join(actions[:5])`，与失败分支写入
        `register_rejected_path` 的 `description` **逐字同拼法** ⇒ 同序列必然命中
        （含单步短序列；实测同序列 5 步 = 0.66、单步 = 0.31，均 > 阈值 0.15）。
        边界（削掉的判别力，如实标注）：①读数走 **tags** 而非 content——不动正文
        ⇒ 对 M5 去重（#225 的极性闸比正文二元组）与 `recall` 的文本面无任何影响；
        代价是同一节点被多次强化且 N 变化时会累积多个计数标签（只增不删）；
        ②匹配仍是**文本相似**（`char_bigram_jaccard > 0.15`）而非精确序列比对，
        且失败分支写入的 `description` 带固定前缀「操作序列失败：」，短串之间会
        因共享前缀 bigram 而越过阈值（实测异序列 `backup > verify > upload` 对
        `deploy` 相似度 0.1778 ⇒ **可能虚报条数**）；③**不**拒绝成功技能化
        （取向保守：照常提取/强化，只把读数带出）——fail-closed 拒绝不在本件边界内；
        ④**不**新增 `subject_key`/`count` 结构化列（须升 `SCHEMA_VERSION`，属另一条
        面，本件一字未碰）。
        判据来源：本端（zcode 值守）2026-10-16 对 issue #216 的补充裁定「一项『无争议、
        可直接落地』的读侧接线」（`.tmp/prowatch/reply216_ruling.md`）＋ 先例
        `propose_promotion` 的读侧接线（`find_rejected_paths` + `path_type` 过滤）。
        """
        node = self.add_perception(
            f"[action_seq] {'成功' if success else '失败'}：{' > '.join(actions[:5])}（{outcome[:40]}）",
            importance=0.5,
            tags=["action_sequence", "success" if success else "failure"]
                 + ([f"hint:{skill_hint}"] if skill_hint else []))
        if success and skill_hint:
            # #216：先读同序列 open 负记忆，把读数带出（见上方 docstring）。
            seq_key = " > ".join(actions[:5])
            prior_failures = [
                p for p in self.find_rejected_paths("操作序列失败：" + seq_key)
                if p.get("path_type") == "action_sequence"]
            if prior_failures:
                self.store.tag_node(
                    node.id, "neg_memory_open:%d" % len(prior_failures))
            # #200：绑定用**精确名称**（`get_skill_by_name`），不得用 `recall_skill`
            # 的 LIKE 子串召回——hint「备份」会命中「删除备份」并把备份步骤拼进
            # 删除技能、为删除技能增信（分诊表行 200 第 1 条）。
            existing = self.store.get_skill_by_name(skill_hint)
            if existing:
                self.store.update_skill_confidence(existing["id"], 0.05)
                self.store.append_skill_procedure(existing["id"], " > ".join(actions[:3]))
            else:
                self.store_skill(skill_hint, f"从操作序列提取（{outcome[:40]}）",
                                 " > ".join(actions[:5]), confidence=0.5)
        elif not success:
            # A-1：失败操作序列 → 被拒绝路径资产（防重复失败）
            self.register_rejected_path(
                path_type="action_sequence",
                description=f"操作序列失败：{' > '.join(actions[:5])}",
                reason=f"outcome: {outcome[:60]}")
            # #200：失败且带 hint ⇒ 对**精确同名**技能降信（旧实现失败从不降信，
            # 频繁失败的技能照样升到高置信——分诊表行 200 第 3 条）。
            if skill_hint:
                failed_skill = self.store.get_skill_by_name(skill_hint)
                if failed_skill:
                    self.store.update_skill_confidence(failed_skill["id"], -0.05)
        return node

    # ---- P1-4 巩固周期 ----

    def consolidate_cycle(self, rehearsal_threshold: float = 0.7,
                          degrade_threshold: float = 0.2, gain: float = 0.01) -> Dict:
        """P1-4 巩固周期：高重要度演练（刷新访问）+ 低重要度降权 + 压缩候选标记。
        不可遗忘过滤器（VAL-TRANSFER-20260813-002-FINAL 约束）：3.2 节 6 类不可遗忘记录
        （信任值评估/偏差历史/价值观迭代/协作校准/内化过程/协定式自我赋予，经 protect_node
        标记）永不被降权或标记为 compressible。

        #258：取数**不再截断**——旧实现 `query_nodes(limit=1000)` 的排序是
        `ORDER BY importance DESC, last_access DESC`（为检索服务），库内超过 1000 条后
        排名 1000 之外的全是**低 importance** 节点，恰是本函数声明要「降权 + 标记
        compressible」的那批：它们永不降权、永不标记，且 `stats` 仍报 `degraded: 0`
        （函数自己也不知道漏了）。本机实测（总节点=1030）旧实现
        `degraded=0, compressible=0`；改为全量遍历后 30/30 低权节点被处理。
        判据来源：公开 issue #258（「docstring 声明要处理的集合，正好是实际看不见的集合」）
        ＋ 分诊表 `triage_lingshu_ALL.md` 行 258；取数口径（`limit=-1` 全层）与同文件
        #297 的 M5 去重候选视野修正同款。
        边界（削掉的判别力/代价，如实标注）：全量遍历使本轮代价随全库规模**线性增长**
        （旧实现固定 ≤1000 条）；本修法取 issue 建议 1/2 的「口径与声明对齐」，
        **不**引入分层定向取数（issue 建议 2 的另一半属另一条独立面）。

        #263（睡眠巩固·遗忘回收的净写入源之一）：本函数结尾旧实现调用
        `add_perception("[consolidation] 演练 …", importance=0.4, tags=["consolidation"])`
        ——而 `add_perception`「自动进入**知识层**」，知识层**不衰减**：维护周期每跑
        一轮就往记忆图里写一条自己统计数字的运行日志（M5 去重只能合并一部分），
        日志还会被 `induce_concepts` 当成知识聚成「概念」、被 `recall` 召回。本机
        实测（300 轮维护）知识层 571 条中 173 条是这种日志。现改为落一条
        **`action_logs` 审计**（只追加、不进检索面），不再写记忆节点。
        判据来源：公开 issue #263「期望行为：维护周期本身不往记忆图写节点」＋
        其「建议修复 1：统计改写到 `action_logs` / `engine_meta`（或只作为返回值）」
        ＋ 分诊表 `triage_lingshu_ALL.md` 行 263；`action_logs` 为同文件既有
        v1.14 先例（`LayeredStore.log_action`）。
        另（同件 建议 3 前半）：**演练分支跳过 `pending_verification` 节点**——
        未复核的归纳概念旧实现以 0.75 起步（> rehearsal_threshold 0.7），被维护
        周期自己逐轮抬到 importance 1.0，越过一切已确认事实。复核通过（标签摘除）
        后才参与演练。
        边界（削掉的判别力，如实标注）：①`compressible` 标签的**实际压缩动作**、
        以及 `induce_concepts` 的「并入已有概念」压缩（issue 建议 2 第 2 条）**未**在
        本件实现——本件只消掉「维护周期净写入」与「未验证概念被自抬到 1.0」两个
        净写入源，概念**过量生成**这一半仍存（见 `induce_concepts` docstring）；
        ②统计从记忆图移到审计表后，`recall("[consolidation] …")` 不再命中
        ——这正是本件要求的判据变更。
        """
        nodes = self.store.query_nodes(limit=-1)
        protected = set(self.store.get_protected_nodes())
        stats = {"rehearsed": 0, "boosted": 0, "degraded": 0, "compressible": 0}
        # ★优化（本笔 · 零行为变更）：旧实现每命中一个演练节点就调一次
        # `self.store.increment_access(n.id)` —— 每次**新开一个 sqlite 连接**。
        # 改为**先筛出参与演练的节点集合**（过滤条件只写一遍），把它们的 id
        # 一次批量提交；再对该集合跑逐节点判定。
        # 「零行为变更」的依据：`new_count` 由 `n.access_count`（**本轮快照**）推出，
        # 不依赖 `increment_access` 的返回值 ⇒ 计数与判定的先后顺序不影响任何判定；
        # 各 `update_node_importance` 仍是各自 `@_transactional`（提交粒度不变）。
        # 过滤条件（不可变层 / 保护 / `pending_verification` / 阈值）**逐字未动**。
        _rehearsed = []
        for n in nodes:
            if n.layer in (MemoryLayer.ANCHOR, MemoryLayer.STRUCTURE):
                continue
            if n.id in protected or "no_forget" in n.tags:
                continue  # 不可遗忘过滤器（3.2 节）
            if "pending_verification" in n.tags:
                continue  # #263：未复核节点不参与演练（不被自抬到 1.0）
            if n.importance >= rehearsal_threshold:
                _rehearsed.append(n)
        self.store.increment_access_many([n.id for n in _rehearsed])
        for n in _rehearsed:
            new_count = (n.access_count or 0) + 1      # 判据用本次增量后的计数，勿用快照旧值
            n.access_count = new_count                 # 同步副本，供同轮后续逻辑使用
            if new_count % 10 == 0:
                self.store.update_node_importance(n.id, gain)
                stats["boosted"] += 1
            stats["rehearsed"] += 1
        # 降权/可压缩面：与演练面**互斥**（`importance >= rehearsal_threshold` 已在上支）
        # ⇒ 旧实现在同一循环的 `elif` 里跑，现按同一条件在**原顺序**上跑。
        for n in nodes:
            if n.layer in (MemoryLayer.ANCHOR, MemoryLayer.STRUCTURE):
                continue
            if n.id in protected or "no_forget" in n.tags:
                continue
            if "pending_verification" in n.tags:
                continue
            if n.importance >= rehearsal_threshold:
                continue  # 演练面已处理
            if n.importance < degrade_threshold:
                if n.access_count <= 2:
                    self.store.update_node_importance(n.id, -gain)
                    stats["degraded"] += 1
                if n.importance < 0.08:
                    self.store.tag_node(n.id, "compressible")
                    stats["compressible"] += 1
        # #263：统计**不再写记忆图**——旧实现 `self.add_perception("[consolidation] …",
        # importance=0.4, tags=["consolidation"])` 把运行日志写成**不衰减的知识层
        # 节点**（见 docstring）。改落 `action_logs` 审计（只追加、不进检索面），
        # 统计数字仍经返回值给调用方。
        summary = (f"[consolidation] 演练 {stats['rehearsed']} · 提升 {stats['boosted']} · "
                   f"降权 {stats['degraded']} · 可压缩 {stats['compressible']}")
        self.store.log_action("consolidation", summary, outcome=dict(stats))
        return stats

    def run_maintenance_cycle(self, decay_factor: float = 0.02) -> Dict:
        """P1-4 睡眠巩固：提升 + 衰减 + 巩固 + 归纳（一次维护周期）

        #214（路 A · 接进维护周期）：本方法新增第一步
        `promote_context_memories`——「短期先进情境层 → 睡眠巩固时再评估提升 → 不够格的自然遗忘」
        这一三段式（`longterm_gate` 文件头与 `promote_context_memories` docstring 同款）
        的第二步，此前在仓内**零生产调用方**——故「该提升的在同一批周期里
        被删掉」。现把它接进维护周期，位置取**衰减之前**。

        判据来源：公开 issue #214 的裁定（路 A）——「接进
        `run_maintenance_cycle`，位置取衰减之前」与「不接进自动衰减线程」
        （`start_auto_decay` 的线程体仍只调 `decay_cycle()`，本件未动）。

        边界（如实标注 · 保守推理而非决定性读数）：裁定帖实测两种位置（decay 前 /
        decay 后）在本例读数**相同**（都在 200 轮内提升成功，因够格节点在被删之前
        先被救走）；取「之前」的理由是**保守推理**——若某节点进周期时
        importance 已在删除线之下（历史库遗留 / 异常输入），衰减在**同一轮**就把它
        删掉，删除不可逆，promote 放在之后便永远看不到它。**该推理未做独立实测。**

        边界（成本）：每轮多一趟情境层遍历 + 逐候选评估，上界 = 情境层 FIFO
        上限（默认 200 条）；`LongTermMemoryGate._known_grams` 已按**库签名**缓存全库
        gram 表（#389 / #770②）⇒ 成本**不随全库正文量增长**。裁定帖把「按节点做
        差集」列为**可选优化、非本件前置条件** ⇒ 本件不做。

        不适用条件：需要「巩固只在显式调用时发生」的可控性时（裁定帖路 B）
        ——本方法一旦被调用即自动提升，不再可选。
        """
        promoted = self.promote_context_memories()
        self.decay_cycle(decay_factor)
        cstats = self.consolidate_cycle()
        induced = self.induce_concepts()
        return {"promote": len(promoted), "decay": decay_factor,
                "consolidation": cstats, "induced_concepts": len(induced)}

    # ==================== 衰减与维护 ====================

    def decay_cycle(self, factor: float = 0.02, min_confidence: float = 0.1, now=None):
        """执行一次衰减周期（v1.16：透传 min_confidence 给 store 层）。

        #176（时间核章程）：本方法新增 `now` 形参（默认 None，向后兼容）并透传给
        store 层——`now` 给出时按 `dt = now - 上次衰减时刻` 用连续核演化（遗忘只
        取决于流逝时间）；`now=None` 时 store 层保持旧的「每次调用 = 一个周期」
        步进语义（既有测试/调用方行为逐字不变）。生产路径由 `start_auto_decay`
        注入时钟：`self._decay_clock` 非 None 时本方法自动以 `self._decay_clock()`
        取值，从而「调用频率无关」。
        """
        if now is None and self._decay_clock is not None:
            now = self._decay_clock()
        self.store.decay_cycle(factor, min_confidence=min_confidence, now=now)

    @_transactional
    def forget_advisor(self, stale_days: float = 30.0, low_value: float = 0.2,
                       archived_imp: float = 0.1) -> Dict:
        """主动遗忘决策器（v1.16 · J 维进化：被动时间衰减 → 主动价值遗忘）。

        递归反思终裁（node_595c062c）：系统根据「记忆是否被使用」主动归档，
        而非仅靠时间流逝。最小可行 v1——仅用可机械读信号：
          ① 访问信号：CONTEXT 层 access_count==0 且 last_access 距今 > stale_days
          ② 低价值信号：CONTEXT 层 importance < low_value
        决策：归档（tags 加 archived + importance 降至 archived_imp）——
        recall 的 importance 加权自然降权。锚点/结构层 no_forget 保护；
        KNOWLEDGE 层不动（长期知识）。

        #448（可逆性收口）：归档前原值落 `nodes.pre_archive_importance`，反向量由
        `LayeredStore.restore_archived` 承担——清 `archived` 标签并恢复原 importance
        （调用方也可显式指定还原值）。此前「可逆」只写在 note 里：仓内既无清除
        `archived` 的公开入口，`forget_advisor` 又对带该标签的节点 `continue`，
        节点被永久锁在归档态。本函数仍**不**把归档当独立存储区：归档节点照样留在
        `nodes` 表、照样参与检索与衰减（只是 importance 加权降低）——这是原设计
        「降权而非隔离」的口径，issue #448 第 1 点的「存储层无法区分」由 `archived`
        标签回答（可查询、可过滤），不另建归档表。

        #781（口径声明 · **只声明现状、不改语义**）：本函数的跳过条件是
        `"no_forget" in tags or "archived" in tags` ⇒ **认 `archived`**；而
        `LayeredStore.decay_cycle` 的情境层删除段经 `_is_forget_protected` 判定，
        只认 `no_forget` ∪ `protections` 行、**不认 `archived`**。故「归档即保护」
        的直觉在两条路径上结论相反：仅带 `archived` 的情境节点被本函数跳过
        （既不归档也不删），却仍被 `decay_cycle` 照常删除。本件**不**统一两者
        （属设计面裁定，见 issue #781 回帖），只把分叉显式落到注释。

        #210（本函数三处错误，逐条修）：
          ① `if (imp or 0.5) < low_value` —— `0.0` 是假值，被 `or` 替换成 0.5，
             **价值最低的情境记忆（importance=0.0）反而不触发低价值信号**。
             修：`imp_v = 0.5 if imp is None else imp`（只有真 None 才回默认），
             再与 `low_value` 比较。
          ② 归档时无条件 `new_imp = archived_imp` —— docstring 写「importance 降至
             archived_imp」，但原本低于它的节点（如 0.05）归档后反而**升到** 0.1。
             修：`new_imp = min(imp_v, archived_imp)`（归档只降不升）。
          ③ `access_stale = (...) if False else None` —— 死代码（真值恒为 None，
             后续只查 `SELECT access_count`），删除。
        判据来源：公开 issue #210（「期望行为：importance=0.0 判为低价值；归档只降
        不升」）＋ 分诊表 `triage_lingshu_ALL.md` 行 210（经验标定；本仓理论章节
        `docs/theory/世界模型与语义时空图_完整理论整理与实现路线.md` 未规定主动遗忘
        的取值口径，追不到更早出处）。
        边界（削掉的判别力，如实标注）：本函数仍只处理 CONTEXT 层、仍只读
        `access_count` / `last_access` / `importance` 三个机械信号；issue #210 第 3
        点「归档值 archived_imp（默认 0.1）恰等于 decay 默认 min_confidence ⇒ 归档
        节点永久驻留」**不由本函数承担**——该死区已由 #207 在 `decay_cycle` 情境层段
        消解（选出全部情境节点后判删），本机实测归档节点 1000 轮衰减后已被删除。
        """
        import time as _t
        now = _t.time()
        archived = 0
        kept = 0
        with self.store._lock:
            c = self.store.conn.cursor()
            # ★优化（本笔 · 零行为变更）三处：
            #   ① 旧首查 `SELECT id, content, importance, last_access, tags` 里的
            #      `content` **在循环体里从未被读取** ⇒ 每个 context 节点白读一遍
            #      大文本列。本笔去掉它。
            #   ② `access_count` 与首查同表同过滤面，旧实现却在循环内逐节点
            #      `SELECT access_count FROM nodes WHERE id=?`（N 次往返）⇒ 并入首查。
            #   ③ 旧 `stale = (...) and self.store.get_node(nid) is not None` 这个局部量
            #      **从未被读取**（#210 已删过它上方的 `access_stale` 死代码，同处漏了这条），
            #      却对每个 context 节点各发一次 `get_node`（读**整行**节点）⇒ 一并删除。
            #   ⇒ 每轮从「首查 + N×get_node + N×SELECT access_count + ≤N×UPDATE」
            #     收成「首查 + ≤N×UPDATE」。判定口径（signal 的两个来源、归档只降不升、
            #     `no_forget`/`archived` 跳过）**逐字未动**。
            c.execute("SELECT id, importance, last_access, tags, access_count FROM nodes "
                      "WHERE layer='context'")
            for nid, imp, last_acc, tags, acc in c.fetchall():
                tags = json.loads(tags) if isinstance(tags, str) and tags else []
                if "no_forget" in tags or "archived" in tags:
                    continue  # 不可遗忘 / 已归档
                # 访问信号：从未被访问且久远
                # #210：此处原有 `access_stale = (...) if False else None` 死代码，已删除。
                # `acc` 已由首查带出（见上方 ②），不再逐节点回查。
                signal = 0
                if acc == 0 and (now - (last_acc or now)) > stale_days * 86400:
                    signal += 1  # 访问信号
                # #210①：`imp is None` 才回默认 0.5；不得用 `or`（0.0 是假值，
                # 会被误替换成 0.5 ⇒ 零重要度节点不触发低价值信号）。
                imp_v = 0.5 if imp is None else imp
                if imp_v < low_value:
                    signal += 1  # 低价值信号
                if signal >= 1:
                    # 归档：importance 降至 archived_imp + tags 加 archived
                    # #210②：归档**只降不升**（旧实现无条件赋 archived_imp，使
                    # 低于它的节点被提权）。
                    new_imp = min(imp_v, archived_imp)
                    new_tags = list(dict.fromkeys(tags + ["archived"]))
                    # #448：归档前原值 + 归档时刻同事务落 `archive_state`，
                    # `restore_archived` 据此原位恢复。
                    c.execute("UPDATE nodes SET importance=?, tags=? WHERE id=?",
                              (new_imp, _dumps_tags(new_tags, ensure_ascii=False), nid))
                    c.execute("INSERT OR REPLACE INTO archive_state"
                              " (node_id, pre_archive_importance, archived_at)"
                              " VALUES (?,?,?)", (nid, imp_v, now))
                    archived += 1
                else:
                    kept += 1
            self._commit_locked()
        return {"archived": archived, "kept": kept,
                "note": "主动遗忘：未被使用的 CONTEXT 记忆归档（importance 降权；"
                        "可逆：restore_archived 清 archived 并恢复原 importance）"}

    #: 后台衰减线程的 join 上界（秒）——`close()` 有界等待，不被卡死的衰减周期
    #: 无限拖住。判据来源：经验标定（本件 #37；仓内无规定该上界的理论章节，
    #: 追不到更早出处）。取值与同文件 `_MEM_LOCK_RETRY` 家族同为「显式有界预算」
    #: 手法；正常路径下 `Event` 唤醒 ⇒ join 几乎立即返回（只有正处 `decay_cycle`
    #: 中才会用到这个上界）。
    _DECAY_JOIN_TIMEOUT = 30.0

    def start_auto_decay(self, interval: float = 60.0, clock=None):
        """启动后台衰减线程。

        #176：新增 `clock` 形参（默认 None ⇒ `time.time`）——注入引擎时钟，使
        每次后台衰减按**真实流逝时间**演化（`interval` 只决定采样频率，不再决定
        遗忘量）。判据来源：公开 issue #176 建议 2「给引擎注入时钟（`clock=time.time`
        可替换）」。

        #37：等待改为**可唤醒**的 `Event.wait(interval)`（旧实现 `time.sleep`）——
        使 `stop_auto_decay` 能立即唤醒线程，而不是等满一个 interval。

        #184：`decay_cycle()` 的异常**在此捕获并记录后继续循环**。旧实现循环体无
        try ⇒ 任一轮衰减抛异常（如一条坏边触发 TypeError）即冒泡出线程、**线程静默
        退出**，此后遗忘机制整体停摆，而 `_running` 仍为 True（对外表现为「在跑」，
        无从察觉）。现把「本周期异常」记入 `self._decay_last_error`（最近一次，供
        `self_check()` 观测）并**继续**后续周期——单周期失败不使整个机制停摆。
        判据来源：公开 issue #184「自动衰减的循环要捕获异常、记录下来、继续运行，
        并把最近一次失败暴露到 `self_check()`」；分诊表 `triage_lingshu_ALL.md`
        行 184 同款（「auto-decay 线程首次异常静默退出」）。
        边界（削掉的判别力）：只记**最近一次**异常（不累积历史）；连续失败时旧读数
        被覆盖——本件只要求「可观测 + 不停摆」，不引入失败计数/退避（那属新机制）。

        #265：本方法**先停掉并 join 可能仍在运行的旧线程**，再启动新线程。旧实现
        只看共享布尔 `_running`（`if self._running: return`），不检查
        `_decay_thread.is_alive()`：若旧线程此刻正处 `decay_cycle()` 体内，`stop`
        的 `Event.set()` 带不走它（它不在 `wait` 上），而 `start` 又把 `_running`
        置回 True ⇒ 旧线程循环继续、新线程另起一条，**两条线程并存**；反复
        stop→start 则线程数线性叠加，遗忘速度按线程数成倍放大，且新 interval
        只作用于新线程（旧线程闭包里的 interval 不变）。本机实测（衰减周期被拉长
        到 0.4s 以稳定复现）：stop→start 4 次后线程数 1 → 5。
        判据来源：公开 issue #265（「任意次数的 stop→start 之后，最多只有一条衰减
        线程在运行」「重启时传入的新 interval 立即生效」）＋ 分诊表
        `triage_lingshu_ALL.md` 行 265。
        边界（削掉的判别力）：join **有界**（`_DECAY_JOIN_TIMEOUT` 秒）——极端
        卡死的衰减周期下本方法仍会在上界后继续启动新线程，代价是那种情况下仍可能
        短暂并存（与 `close()` 的同一取舍：生命周期收尾不得被单个后台任务无限拖住）。
        """
        # #265：先停旧（含 join）再起新 ⇒ 任意次 stop→start 后最多一条线程，
        # 且新 interval 立即生效（不再被旧线程的闭包 interval 拖住）。
        self.stop_auto_decay()
        self._running = True
        self._decay_wake.clear()
        self._decay_last_error = None
        # #176：置入时钟 ⇒ `decay_cycle()` 每次按真实流逝时间衰减（interval 只控
        # 采样频率）。`clock=None` 回退 `time.time`（生产默认）。
        self._decay_clock = clock if clock is not None else time.time

        def loop():
            # #37：唤醒后**先判 `_running` 再决定是否 `decay_cycle`**。旧实现是
            # 「sleep 醒来 → 无条件 decay_cycle → 回 while 判 `_running`」⇒
            # `close()` 期间被唤醒的线程会在 `store.close()` **之后**访问已关闭
            # 连接（必抛，且异常落在 daemon 线程里无人回收）。
            while self._running:
                if self._decay_wake.wait(interval):
                    break                 # 被 stop/close 唤醒 ⇒ 不再跑本轮衰减
                if not self._running:
                    break
                try:
                    self.decay_cycle()
                    self._decay_last_error = None   # #184：本周期成功 ⇒ 清旧读数
                except Exception as exc:            # noqa: BLE001 —— 见 docstring #184
                    self._decay_last_error = "%s: %s" % (type(exc).__name__, exc)

        self._decay_thread = threading.Thread(target=loop, daemon=True)
        self._decay_thread.start()

    def stop_auto_decay(self):
        """停止后台衰减线程：置位并**立即唤醒**等待中的线程（#37），随后
        **有界 join** 确保返回时旧线程已退出（#265）。

        #265：旧实现只置 `_running=False`（不 join、不唤醒）——`stop_auto_decay()`
        返回后线程可能仍在跑，紧接着的 `start_auto_decay()` 会再叠一条。现 join
        后返回 ⇒ 任意次 stop→start 之后最多一条衰减线程（见 `start_auto_decay`）。
        join 有界（`_DECAY_JOIN_TIMEOUT`）：正常路径下 `Event` 唤醒 ⇒ 立即返回；
        仅在正处 `decay_cycle()` 中时才用到上界。
        边界（削掉的判别力）：与 `close()` 同款取舍——极端卡死的衰减周期下 join
        在上界后返回，此时不保证线程已退出（不引入强制杀线程）。
        """
        self._running = False
        self._decay_wake.set()
        # #176：停止后清除注入时钟 ⇒ 后续手动 `decay_cycle()` 回到「每次=一个周期」
        # 的旧步进语义（避免停线程后残留的时间记账改变手动调用的行为）。
        self._decay_clock = None
        t = self._decay_thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=self._DECAY_JOIN_TIMEOUT)

    # ==================== 自检 ====================

    def self_check(self) -> Dict:
        """
        自检：锚点层完整性、结构层一致性、自我层存在性
        返回诊断报告
        """
        anchors = self.get_anchors()
        structures = self.get_current_structure_nodes()
        self_nodes = self.store.get_layer_nodes(MemoryLayer.SELF)
        report = {
            "anchor_ok": len(anchors) > 0,
            "anchor_count": len(anchors),
            "structure_ok": len(structures) >= 2,
            "structure_count": len(structures),
            # 以持久化 SELF 层为准：runtime self_model 在 __init__ 无条件构造（identity
            # 恒为非空默认值"协议实例"），用其作判据会使 self_ok 恒 True，无法发现失忆/无
            # 自我——与同函数 anchor_ok / structure_ok 的纯持久化口径保持一致。
            "self_ok": len(self_nodes) > 0,
            "self_model_exists": len(self_nodes) > 0,
            # #33：启动恢复状态（只增字段）。`self_ok` 判据**未动**——它只判「SELF 层
            # 有节点」，无法区分「快照在库、运行态已回默认」；失忆由此字段观测：
            # status ∈ {empty(新库无快照), restored(已回填), failed(坏快照回退默认)}，
            # 另带 source 与 identity_conflict。削掉的判别力：本件不把 self_ok 改成
            # 「值一致」判据（那会改既有可观测字段语义、波及既有断言），只新增出口。
            "self_restore": dict(self._self_restore),
            "open_blindspots": len(self.store.list_blindspots(status="open")),
            "skills_count": self.store.count_skills(),
            "context_count": self.store.count_layer(MemoryLayer.CONTEXT),
            "stats": self.store.get_stats(),
            # issue #156：组件装配状态（只增字段；装配失败此前无可观测出口）
            "component_assembly": self._component_assembly_status(),
            # #184：后台衰减线程状态（只增字段）。旧实现线程异常即静默退出而
            # `_running` 仍为 True ⇒ 「在跑」与「已停摆」对外不可区分；现暴露
            # `running`（置位语义）与 `last_error`（最近一次周期异常，None=正常）。
            "auto_decay": {"running": bool(self._running),
                           "thread_alive": bool(self._decay_thread is not None
                                                and self._decay_thread.is_alive()),
                           "last_error": self._decay_last_error},
            "timestamp": time.time()
        }
        # 检查是否存在因果循环：先 O(V+E) 三色探测，无环即免整趟枚举（issue #145）
        #
        # #215：`has_causal_cycle()` 是**存在性**探测（三色 DFS，无深度限制），
        # `find_cycles(max_depth=8)` 是**有上限的枚举**（`max_depth` 是环长上限）。
        # 环长 > max_depth 时前者 True、后者 []——旧实现据此报 `cycles_found=0`，
        # **自检报告自相矛盾**（「有环」与「0 个环」并存），调用方把 0 当「无环」，
        # 正是 #145 想消除的「错误的放心信号」。现把两者的**判据上限**如实分开：
        # `has_causal_cycle` 照报；`cycles_found` 仍是**已枚举数**（不改语义，避免
        # 破坏既有断言）；另新增 `cycles_truncated`（探测到环但未枚举出任何环 ⇒
        # 环长可能超过 max_depth=8）与 `cycle_warning`（人读告警），使 0 不再可被
        # 读成「无环」。
        # 判据来源：公开 issue #215「问题 1」及其建议 1（「self_check 在
        # has_causal_cycle() 为 True 而 find_cycles 为空时**明确告警**『检测到环但
        # 未能枚举（环长可能超过 max_depth=8）』，而不是报 0」）＋ 分诊表
        # `triage_lingshu_ALL.md` 行 215。
        # 边界（削掉的判别力，如实标注）：本件**不**改 `find_cycles` 的深度上限
        # （那会改 #145 的耗时判据），也**不**把 `cycles_found` 改成别的量（会波及
        # `tests/test_issue145_causal_depth_cycles.py` C 组既有断言）；「有环大图仍
        # 走深搜枚举」这一 #215 问题 2 不在本件（属复杂度预算取舍，须先裁定）。
        #
        # #252：上面 #215 的边界注释所说的「有环大图仍走深搜枚举」在本件收口——
        # `find_cycles` 现支持 `limit`，本处传 `SELF_CHECK_CYCLE_ENUM_LIMIT`（=5，
        # 即 `cycle_details` 实际用到的条数）。旧实现无上限：11 节点完全有向图
        # `self_check` 实测 13.69s / 335 万条环 / 峰值 RSS +3041 MB（n=10：2.77s/
        # 676MB），健康检查可被一小簇互为因果的节点卡死或 OOM。`cycles_found`
        # 语义**不变**（已枚举环数，现至多 limit 条）；新增 `cycles_capped`
        # （已枚举数达到 limit ⇒ 图中环可能更多，报告数不完整）——与 #215 的
        # `cycles_truncated`（探测到环但一条都没枚举出，环长超 max_depth）区分。
        # 判据来源：公开 issue #252「建议修复 1」（「`find_cycles` 增加 `limit`（或
        # 改成生成器），`self_check` 取前 5 条明细；`cycles_found` 达到上限时报
        # `"≥N"`，或另加 `cycles_truncated: True`」）＋ 其复现实读 ＋ 分诊表行 252。
        # 边界（削掉的判别力，如实标注）：`cycles_found` 仍是 int（未按 issue 建议
        # 改成 `"≥N"` 字符串——那会波及 `tests/test_issue145_causal_depth_cycles.py`
        # C4 的 `>=2` 数值断言）；「报告数是否完整」改由新增 `cycles_capped` 承载。
        has_cycle = self.store.has_causal_cycle()
        cycles = (self.store.find_cycles(max_depth=8,
                                         limit=self.SELF_CHECK_CYCLE_ENUM_LIMIT)
                  if has_cycle else [])
        report["cycles_found"] = len(cycles)
        report["has_causal_cycle"] = has_cycle
        report["cycles_truncated"] = bool(has_cycle and not cycles)
        report["cycle_warning"] = (
            "检测到因果环但未能枚举（环长可能超过 max_depth=8）"
            if report["cycles_truncated"] else None)
        report["cycles_capped"] = bool(
            has_cycle and len(cycles) >= self.SELF_CHECK_CYCLE_ENUM_LIMIT)
        report["cycle_details"] = [[e.id for e in path] for path in cycles[:5]]
        return report

    # ==================== 统计与关闭 ====================

    def get_stats(self) -> Dict:
        return self.store.get_stats()

    def close(self):
        """关闭实例：先停后台衰减线程并**join 等它退出**，再关连接（#37）。

        旧实现只 `stop_auto_decay()`（置位）就 `store.close()`——线程若正睡在
        `time.sleep(interval)` 里，唤醒后会先 `decay_cycle()` 再回判 `_running`
        ⇒ 访问已关闭的连接（`sqlite3.ProgrammingError`），异常落在 daemon 线程
        无人回收，且衰减写入与关库并发（竞态）。
        修法：① `stop_auto_decay` 用 `Event` 立即唤醒；② 唤醒后**先判 `_running`
        再决定是否衰减**（见 `start_auto_decay`）；③ 本处 **join** 确保
        `store.close()` 不与任何在飞的 `decay_cycle()` 重叠。
        判据来源：经验标定（本件 #37；仓内无规定后台线程收尾时序的理论章节，
        追不到更早出处）。
        削掉的判别力：join **有界**（`_DECAY_JOIN_TIMEOUT` 秒）——衰减周期异常
        卡死时本方法不会无界阻塞，代价是极端情况下仍可能与关库重叠一次；取舍是
        「生命周期收尾不得被单个后台任务无限拖住」。
        """
        self.stop_auto_decay()
        t = self._decay_thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=self._DECAY_JOIN_TIMEOUT)
        self.store.close()

    # ---- v1.15 长期记忆写入决策器（LongTermMemoryGate） ----

    def _ensure_gate(self):
        """惰性装配长期记忆门（零依赖）。"""
        if self._gate is None:
            from .longterm_gate import LongTermMemoryGate
            self._gate = LongTermMemoryGate(self)
        return self._gate

    def longterm_snapshot(self, content: str, source: str = "snapshot",
                          tags: list = None, entities: list = None,
                          importance_hint: float = None) -> dict:
        """记忆快照 → 重要性评估 → 按层级写入（长期/知识/情境）+ 条件空间 + 关联。
        主动沉淀机制：重要经验/会话结束/关键事件调用。"""
        try:
            gate = self._ensure_gate()
            return gate.write_snapshot(content, source, tags, entities,
                                       importance_hint)
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    def prefeed_input(self, content: str, source: str = "input",
                      tags: list = None, entities: list = None) -> dict:
        """H1 海马体前馈：新奇检测 → 高新奇输入当场强化编码（标记+提权+建边）。
        外部输入（对话/摄取/感知）到达时调用——「看到新东西眼睛一亮」。
        返回 {novel, novelty, action, node_id, importance, links}"""
        try:
            gate = self._ensure_gate()
            return gate.prefeed(content, source, tags, entities)
        except Exception as exc:
            return {"novel": False, "action": "error", "error": str(exc)}

    def pattern_separation_scan(self, limit: int = 150) -> dict:
        """H3 海马体模式分离：扫描相似节点对 → 建立分离边（条件差异显式化）。
        检索时命中相似节点会附「这两个的区别」提示——细化条件得到精确知识。"""
        try:
            import sys as _s
            # issue #10（② 在 core.py 的未收口子项）：原赋值是导出脱敏把**路径操作数**
            # 改写成中性占位字面量后的残留——`os.path.isabs` 在 Windows/Linux 均恒
            # False ⇒ 本注入在公开仓**恒不发生**（死代码），故该残留已删除。
            # 现改为**环境变量驱动**（#10 §三.③ 的导出精修先例：本机字面量 → 环境变量）：
            # 取值面 = 部署方显式声明的组件根 `LINGSHU_COMPONENT_ROOT`（`os.pathsep`
            # 多根，与 `activation.py` 的 `LINGSHU_DATA_ROOT` 同款约定；真源即
            # `component_resolver.configured_roots`，不另造第二份解析）。未声明 ⇒ 空表
            # ⇒ **不做任何注入**（保持 #381 的 fail-safe 语义）。
            # issue #381：只接受**绝对**条目。相对条目（含空串 / `.`）的解析目标
            # 由判定时的 cwd 决定（判据同 `lingshu/_pathguard.py::_is_relative_entry`）
            # ⇒ 插进 `sys.path` 等于把本包 import 期刚消除的 cwd 面还给**普通**
            # import：cwd 下任何同名裸模块（含标准库名）可被抢先执行。白名单组件名
            # 由 `component_resolver` 的 meta_path finder 拦截、不受此影响；受影响的
            # 恰是非白名单裸名，故本处一律不插相对条目。
            for _kb in reversed(_configured_component_roots()):
                if os.path.isabs(_kb) and _kb not in _s.path:
                    _s.path.insert(0, _kb)
            from pattern_separation import PatternSeparation
            ps = PatternSeparation(self)
            return ps.scan(limit=limit)
        except Exception as exc:
            return {"created": 0, "error": str(exc)}

    def reconstruct_scene(self, clue: str, depth: int = 2,
                          max_nodes: int = 8) -> dict:
        """H4 海马体情景重构：线索 → 条件空间下的信息复原。
        从部分片段重建完整记忆场景（沿 similar/causal 边 + 条件空间合成），
        输出显式标注「重构非回放」——回忆起的情景是当前条件下的分析恢复，
        不代表真实发生的过去就是如此（0.0.3 局部不可知）。"""
        try:
            import sys as _s
            # issue #10：同 `pattern_separation_scan`——脱敏残留的字面量已删除，
            # 改由 `LINGSHU_COMPONENT_ROOT`（`component_resolver.configured_roots`）
            # 环境变量驱动；未声明 ⇒ 不做注入。
            # issue #381：只接受**绝对**条目，相对条目的解析目标由 cwd 决定，
            # 不得插进 `sys.path`。
            for _kb in reversed(_configured_component_roots()):
                if os.path.isabs(_kb) and _kb not in _s.path:
                    _s.path.insert(0, _kb)
            from scene_reconstruction import SceneReconstruction
            sr = SceneReconstruction(self)
            return sr.reconstruct(clue, depth=depth, max_nodes=max_nodes)
        except Exception as exc:
            return {"scene": [], "error": str(exc)}

    def promote_context_memories(self, limit: int = 30) -> list:
        """情境层批量提升扫描（睡眠巩固/会话结束调用）：够格者升知识层/长期层。"""
        try:
            gate = self._ensure_gate()
            return gate.promote_from_context(limit)
        except Exception:
            return []
