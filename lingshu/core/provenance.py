# -*- coding: utf-8 -*-
"""provenance · 记忆写入的「来源与验证」契约（权威实现）
============================================================================
动机（荣 2026-09-09 标注复检结论）：
  session_other / other 里混进了**用户本人的真实对话** —— 因为写入时没有记录
  「谁 / 在什么工具下 / 哪个会话 / 什么时间 / 验证过没有」，事后只能靠
  `tags LIKE '%dsh%'` 这类字符串猜类别，猜错就归错桶。

契约（每条记忆必须能回答 6 个问题）：
  1. 谁说的        → source           （user / assistant / system / tool / fixture）
  2. 在哪观测      → observation_position
  3. 用什么工具    → observation_tool
  4. 哪个会话      → session_id
  5. 什么时候      → time_window      （+ existence_constraint：这条记忆在什么条件下成立）
  6. 验证过没有    → verify_status + verify_methods + verify_ref（可复现凭证）

与协议的关系：
  · 前四项中的 observation_position / observation_tool / time_window /
    existence_constraint 即协议「条件空间 ConditionSpace」四栏；
  · **四栏必须原样进 condition_space**（aeis.core.ConditionSpace.from_json 用
    cls(**d)，多一个键就 TypeError），其余字段（source / session_id / 验证信息）
    落到结构化 tag，不要塞进 condition_space。

本文件是契约的**唯一权威实现**（协议内 · 纯标准库 · 零依赖）：
  · 写入方：`lingshu_runtime.py` / `voice_daemon.py` 为保持自包含进程，内联同等
    规则，但取值必须落在本文件取值域内；
  · 工具链（上游仓 / 未随本仓导出）：`AEIS/tools/provenance.py` 是薄壳 re-export；
    `AEIS/tools/audit_provenance.py` 用本契约对全库做合规普查（两者均不在本仓）。
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# =============================================================================
# 取值域（收敛口径，避免每个写入方自创一套）
# =============================================================================
POSITIONS = {
    "电脑端·DSH", "灵枢·语音", "感知·摄像头", "感知·屏幕",
    "云端·知识入库", "测试台", "协议基底", "未知观测位",
}
TOOLS = {
    "dsh", "lingshu_voice", "vision_pipeline", "screen_capture",
    "knowledge_ingest", "consolidation_engine", "bench_fixture",
    "protocol_core", "unknown",
}
SOURCES = {"user", "assistant", "system", "tool", "external", "fixture", "unknown"}

# 验证状态：强度递增（对齐 anchored_verification 的弱/强锚定）
VERIFY_STATUS = ("unverified", "partial", "verified", "anchored")
# 验证方式
VERIFY_METHODS = {
    "none",            # 没验证
    "llm_verify",      # 模型自审（弱）
    "cross_channel",   # 感知通道互裁（弱）
    "whitebox_code",   # 白箱：代码/脚本真跑（强）
    "action_world",    # 行动 / 世界裁决（强）
    "human_review",    # 人工复核（强）
}
# 强验证方式（对齐 aeis.anchored_verification.STRONG_SOURCES）
STRONG_METHODS = {"whitebox_code", "action_world", "human_review"}

# 协议四栏
CS_FIELDS = ("observation_position", "observation_tool",
             "time_window", "existence_constraint")

# 引擎默认值（= 没有信息量的四栏，回填脚本与审计脚本据此识别）
DEFAULT_CS = {
    "observation_position": "感知系统",
    "observation_tool": "感官输入",
    "existence_constraint": "协议实例运行中",
}
DEFAULT_WINDOW = 3600.0


# =============================================================================
# 契约对象
# =============================================================================
@dataclass
class Provenance:
    """一条记忆的来源与验证声明。"""

    # —— 谁 / 在哪 / 用什么 / 哪个会话 ——
    source: str = "unknown"
    observation_position: str = "未知观测位"
    observation_tool: str = "unknown"
    session_id: Optional[str] = None

    # —— 什么时候 / 在什么条件下成立 ——
    time_window: Tuple[float, float] = (0.0, 0.0)
    existence_constraint: str = "（未声明）"

    # —— 验证过没有 / 怎么验的 / 凭证 ——
    verify_status: str = "unverified"
    verify_methods: List[str] = field(default_factory=lambda: ["none"])
    verify_ref: str = ""          # 可复现凭证：脚本路径 / 命令 / 测试 id / 日志行号

    # —— 元信息：哪些字段是推断或回填的（不是原始观测） ——
    synthetic: List[str] = field(default_factory=list)

    # -------------------------------------------------------------------------
    def to_condition_space(self) -> Dict[str, Any]:
        """协议四栏（严格四键，不可多加）。"""
        return {
            "observation_position": self.observation_position,
            "observation_tool": self.observation_tool,
            "time_window": list(self.time_window),
            "existence_constraint": self.existence_constraint,
        }

    def to_tags(self, extra: Optional[List[str]] = None) -> List[str]:
        """来源/会话/验证信息 → 结构化 tag（口径与既有标签一致）。

        默认不写 `unverified`（省略即未验证）；一旦有验证声明则连同方式与
        凭证一起落地，避免出现「声称验证过但没有凭证」的不可复现状态。
        """
        out: List[str] = []

        def add(t: Optional[str]):
            if t and t not in out:
                out.append(t)

        for t in (extra or []):
            add(t)
        if self.session_id:
            add("session:" + str(self.session_id))
        if self.source and self.source != "unknown":
            add(self.source)
        if self.verify_status != "unverified":
            add(self.verify_status)
            for m in self.verify_methods:
                if m and m != "none":
                    add(m)
            if self.verify_ref:
                add("vref:" + self.verify_ref)
        return out

    def to_sql(self, extra_tags: Optional[List[str]] = None) -> Tuple[str, str]:
        """返回 (condition_space_json, tags_json)，供裸 sqlite 写入方使用。"""
        return (json.dumps(self.to_condition_space(), ensure_ascii=False),
                json.dumps(self.to_tags(extra_tags), ensure_ascii=False))

    def to_json(self) -> str:
        d = asdict(self)
        d["time_window"] = list(self.time_window)
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "Provenance":
        """从 JSON 反序列化；**形状异常的行按契约落 unknown**（#342）。

        既有库里可能有形状不对的 provenance 列（`null` / 数字 / 数组 /
        `time_window=[null,null]`——core 自己的 ConditionSpace 规范化后会产出
        后者）。此前这些行会让 `d.get` / `float(...)` 直接抛异常并毒化整条读链；
        现在一律回落到字段默认值（source=unknown、time_window=(0,0)），不抛。
        削掉的判别力：形状异常行与「合法但全默认」的行在本函数出口上不再可分
        （二者的 Provenance 相同）；形状本身的信息不进 `synthetic`（本函数是
        反序列化而非推断，无 provenance 上下文可记账）。
        """
        try:
            d = json.loads(s or "{}")
        except Exception:
            d = {}
        if not isinstance(d, dict):
            d = {}
        tw = d.get("time_window") or [0.0, 0.0]
        d["time_window"] = _safe_window(tw) or (0.0, 0.0)
        for k in ("verify_methods", "synthetic"):
            v = d.get(k)
            d[k] = list(v) if isinstance(v, (list, tuple)) else ([] if k == "synthetic" else ["none"])
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    # -------------------------------------------------------------------------
    def strength(self) -> str:
        """验证强度：strong / weak / none。"""
        if self.verify_status == "unverified":
            return "none"
        if any(m in STRONG_METHODS for m in self.verify_methods):
            return "strong"
        return "weak"

    def is_traceable(self) -> bool:
        """能否回答「谁、在哪个会话、什么时候说的」。

        人类/AI 的对话必须可回溯到具体会话；系统日志与夹具不必；
        来源 unknown 一律不可回溯——连「谁说的」都答不上来。

        #20-B 口径决策（豁免）：**强验证记录**（`strength() == "strong"`，即含
        whitebox_code / action_world / human_review）的可信链锚在**可复现凭证**上
        （脚本 / 命令 / 测试 id / 行动裁决），不经过对话会话——故这类记录即便
        没有 session_id 也判为可回溯。此前「白箱校验」记录 `strength()=="strong"`
        而 `is_traceable()==False`，同一对象上两个公开判定给出相反的可信度结论，
        下游读任一都会系统性偏乐观或偏保守。
        削掉的判别力：一条声称强验证（且 verify_status != unverified）的记录，
        即便查不到会话、连来源都是 unknown，也判为可回溯——本判定不再能区分
        「强验证 + 有会话」与「强验证 + 无任何会话/来源」。判据来源：issue #20
        第二节与「建议判据」第 2 条（口径决策落纸，非风格意见）。
        """
        if self.source not in ("user", "assistant", "unknown"):
            return True            # 系统日志 / 夹具：不必回溯到会话
        if self.source != "unknown" and bool(self.session_id) \
                and self.observation_tool != "unknown" \
                and _window_start_is_positive(self.time_window):
            return True
        return self.strength() == "strong"   # #20-B 强验证豁免

    def extensions(self) -> List[str]:
        """哪些字段用了取值域之外的扩展值。

        扩展值**不是**错误——它是有信息量的自定义声明（如知识卡自述的观测位置、
        学科卡拆分工具），只是未收敛到统一取值域。回填时必须原样保留。
        """
        out: List[str] = []
        if self.observation_position not in POSITIONS:
            out.append("observation_position=" + str(self.observation_position))
        if self.observation_tool not in TOOLS:
            out.append("observation_tool=" + str(self.observation_tool))
        return out

    def validate(self) -> List[str]:
        """返回问题列表（空 = 合规）。"""
        p: List[str] = []
        if self.source not in SOURCES:
            p.append(f"source 取值非法: {self.source}")
        if self.observation_position not in POSITIONS:
            if not self.observation_position or self.observation_position in DEFAULT_CS.values():
                p.append(f"observation_position 无信息量: {self.observation_position!r}")
            else:
                p.append(f"observation_position 取值域外（扩展值）: {self.observation_position}")
        if self.observation_tool not in TOOLS:
            if not self.observation_tool or self.observation_tool in DEFAULT_CS.values():
                p.append(f"observation_tool 无信息量: {self.observation_tool!r}")
            else:
                p.append(f"observation_tool 取值域外（扩展值）: {self.observation_tool}")
        tw = self.time_window
        if not (isinstance(tw, (tuple, list)) and len(tw) == 2):
            p.append("time_window 必须是 [start, end]")
        elif not _is_numeric_window(tw):
            p.append(f"time_window 元素必须是数值: {tw}")
        elif tw[0] > tw[1]:
            p.append(f"time_window 起点晚于终点: {tw}")
        elif tw[0] <= 0:
            p.append("time_window 缺观测时刻（start <= 0）")
        if self.verify_status not in VERIFY_STATUS:
            p.append(f"verify_status 取值非法: {self.verify_status}")
        bad = [m for m in self.verify_methods if m not in VERIFY_METHODS]
        if bad:
            p.append(f"verify_methods 取值非法: {bad}")
        # 一致性约束
        if self.source in ("user", "assistant") and not self.session_id:
            p.append("source 为 user/assistant 但缺 session_id —— 无法回溯到具体会话")
        if self.verify_status != "unverified":
            if self.verify_methods == ["none"] or not self.verify_methods:
                p.append("声称已验证但未声明 verify_methods")
            if not self.verify_ref:
                p.append("声称已验证但缺 verify_ref（不可复现的验证等于没验证）")
        if self.synthetic:
            p.append("含推断/回填字段: " + ",".join(self.synthetic) + "（非原始观测）")
        return p


# =============================================================================
# 写入侧便捷入口
# =============================================================================
def new_provenance(position: str, tool: str, source: str,
                   session_id: Optional[str] = None,
                   existence_constraint: str = "协议实例运行中",
                   verify_status: str = "unverified",
                   verify_methods: Optional[List[str]] = None,
                   verify_ref: str = "",
                   observed_at: Optional[float] = None,
                   window: float = DEFAULT_WINDOW) -> Provenance:
    """写入方构造一条合规 provenance 的便捷入口。"""
    t0 = float(observed_at if observed_at is not None else time.time())
    return Provenance(
        source=source, observation_position=position, observation_tool=tool,
        session_id=session_id, time_window=(t0, t0 + window),
        existence_constraint=existence_constraint,
        verify_status=verify_status,
        verify_methods=list(verify_methods or ["none"]),
        verify_ref=verify_ref,
    )


# =============================================================================
# 从既有数据推断（迁移 / 回填 / 审计用）
# =============================================================================
def _tags(tags: Any) -> List[str]:
    """tags 列 → 字符串标签列表；形状异常一律回落到空表（= unknown，#342）。

    既有库里 tags 列可能不是 JSON 数组：`'5'`（数字）、`'"dsh"'`（字符串标量）、
    `null`、对象……。此前非列表会 `TypeError: 'int' object is not iterable`，
    字符串标量会被**按字符拆**成 `['d','s','h']`。现在：JSON 字符串标量按
    **单个标签**处理（不是拆字），其余非列表形状回落到空表。
    削掉的判别力：形状异常行的原始形状（数字 / 对象）不再可辨——一律等同「无标签」。
    """
    if isinstance(tags, str):
        try:
            tags = json.loads(tags)
        except Exception:
            return []
    if isinstance(tags, str):
        tags = [tags]                     # JSON 字符串标量 → 单标签，勿拆字符
    if not isinstance(tags, (list, tuple)):
        return []
    return [str(t).strip() for t in tags]


def _safe_window(tw: Any) -> Optional[Tuple[float, float]]:
    """把候选 time_window 规整为 (start, end)；形状/数值不合法或 start<=0 → None。

    #342：`float(tw[0])` 此前对 `[null,null]`（core 的 ConditionSpace 规范化
    产物）抛 TypeError、对 `['2026-10-01', ...]` 抛 ValueError，直接毒化读链。
    现在不合法一律返回 None，交由调用方回落到 unknown / created_at。
    """
    if not isinstance(tw, (list, tuple)) or len(tw) != 2:
        return None
    try:
        start, end = float(tw[0]), float(tw[1])
    except (TypeError, ValueError):
        return None
    if start <= 0:
        return None
    return (start, end)


def _is_number(v: Any) -> bool:
    """真数值（bool 是 int 的子类，但对时间窗不算数值）。"""
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_numeric_window(tw: Any) -> bool:
    """time_window 是否为「两个元素都是真数值」的形状（#1962）。"""
    return (isinstance(tw, (tuple, list)) and len(tw) == 2
            and _is_number(tw[0]) and _is_number(tw[1]))


def _window_start_is_positive(tw: Any) -> bool:
    """time_window[0] 是否为正数值——取不到数值就返回 False，不抛（#1962）。

    与 `_safe_window` 同口径：形状/类型不合法一律落到「无观测时刻」。
    只判定**起点**：终点是坏值时 `is_traceable()` 的既有口径（只看起点）
    不在 #1962 范围内，不在此收紧。
    """
    if not (isinstance(tw, (tuple, list)) and len(tw) == 2):
        return False
    return _is_number(tw[0]) and tw[0] > 0


def from_legacy(tags: Any, created_at: Optional[float] = None,
                condition_space: Any = None, modality: str = "text") -> Provenance:
    """从既有行（tags / created_at / condition_space）反推 provenance。

    只做**可证据的**推断；推不出的留 unknown，并在 synthetic 里记账。
    """
    ts = _tags(tags)
    low = {t.lower() for t in ts}
    p = Provenance()

    # 会话 id：tags 里的 session:*
    for t in ts:
        if t.startswith("session:"):
            p.session_id = t.split(":", 1)[1]
            break
    # 独立会话域：role:<id> → role-session-<id>（会话边界即记忆库边界）
    if not p.session_id:
        for t in ts:
            if t.startswith("role:"):
                p.session_id = "role-session-" + t.split(":", 1)[1]
                p.synthetic.append("session_id")
                break

    # 观测工具 / 位置
    if "dsh" in low:
        p.observation_tool, p.observation_position = "dsh", "电脑端·DSH"
    elif "voice" in low:
        p.observation_tool, p.observation_position = "lingshu_voice", "灵枢·语音"
    elif modality == "image":
        p.observation_tool, p.observation_position = "vision_pipeline", "感知·摄像头"
    elif "knowledge_ingest" in low:
        p.observation_tool, p.observation_position = "knowledge_ingest", "云端·知识入库"
    elif "consolidation" in low:
        p.observation_tool, p.observation_position = "consolidation_engine", "测试台"
    elif "novel_prefeed" in low or "gate" in low:
        p.observation_tool, p.observation_position = "bench_fixture", "测试台"

    # 来源：原始声明优先；gate / novel_prefeed 描述处理路径，不能覆盖「谁说的」。
    for source in ("user", "assistant", "external", "system", "tool", "fixture"):
        if source in low:
            p.source = source
            break
    else:
        # 没有来源声明时，保留既有数据的推断规则。
        if p.observation_tool == "bench_fixture":
            p.source = "fixture"
        elif "consolidation" in low:
            p.source = "system"
        elif "白箱校验" in ts or "llm_verify" in low:
            p.source = "assistant"

    # 时间窗 / 存在约束：优先取 condition_space，识别默认值
    cs = condition_space
    if isinstance(cs, str):
        try:
            cs = json.loads(cs or "{}")
        except Exception:
            cs = {}
    if not isinstance(cs, dict):      # #342：JSON 数组/标量 → 视同未声明
        cs = {}
    tw = _safe_window(cs.get("time_window"))
    if tw is not None:
        p.time_window = tw
    else:
        t0 = None
        if created_at:
            try:
                t0 = float(created_at)
            except (TypeError, ValueError):
                t0 = None
        if t0 is not None:
            p.time_window = (t0, t0 + DEFAULT_WINDOW)
            p.synthetic.append("time_window")
    for k, v in DEFAULT_CS.items():
        cur = cs.get(k)
        if cur and cur != v:
            setattr(p, k, cur)
        else:
            p.synthetic.append(k)
    if not cs.get("existence_constraint"):
        p.existence_constraint = "（未声明）"

    # 验证状态 / 方式（按强度递减取首个命中；VERIFY_STATUS 为强度递增序：
    # unverified < partial < verified < anchored）
    for s in ("anchored", "verified", "partial", "unverified"):
        if s in low:
            p.verify_status = s
            break
    # 验证方式（#146）：既要认历史中文标签（白箱校验），也要读回本模块
    # `to_tags` 自己写出的**规范英文方式名**——否则 to_tags → from_legacy
    # 往返会把 strong 记录降成 weak，并让 validate 反报「未声明 verify_methods」。
    # 同时使 VERIFY_METHODS 的每一项都有 legacy 入口（#20-A：此前 6 项仅 2 项
    # 可反推，human_review 等静默丢失）。
    methods: List[str] = []
    if "白箱校验" in ts:                       # 历史中文标签
        methods.append("whitebox_code")
    for m in sorted(VERIFY_METHODS - {"none"}):
        if m in low and m not in methods:      # 规范英文名（大小写不敏感）
            methods.append(m)
    p.verify_methods = methods or ["none"]
    # 可复现凭证（#146）：读回 to_tags 写出的 `vref:<ref>`（ref 内可含 ':'）
    for t in ts:
        if t.startswith("vref:"):
            p.verify_ref = t[len("vref:"):]
            break

    if not p.session_id and p.source in ("user", "assistant"):
        p.synthetic.append("session_id")
    return p


def is_default_cs(cs: Any) -> bool:
    """condition_space 是否为引擎默认值（= 四栏没有信息量）。

    #342：形状异常（JSON 数组 / 标量 / 解析失败）不再抛 AttributeError——
    一律视同「未声明」并返回 True。
    """
    if isinstance(cs, str):
        try:
            cs = json.loads(cs or "{}")
        except Exception:
            return True
    if not isinstance(cs, dict):
        return True
    # 缺键 = 未声明 → 同样算「没有信息量」（灵枢/语音写入方曾直接写 "{}"）
    return (cs.get("observation_position", DEFAULT_CS["observation_position"]) == DEFAULT_CS["observation_position"]
            and cs.get("observation_tool", DEFAULT_CS["observation_tool"]) == DEFAULT_CS["observation_tool"]
            and cs.get("existence_constraint", DEFAULT_CS["existence_constraint"]) == DEFAULT_CS["existence_constraint"])


if __name__ == "__main__":   # 自检
    cases = [
        # 电脑端 DSH 的用户口述：应可回溯
        (["dsh", "user", "session:dsh-1"], 1786728904.0,
         {"time_window": [1786728904.0, 1786732504.0]}, "text"),
        # 白箱校验记录：声称验证过，但没有可复现凭证
        (["白箱校验", "llm_verify", "partial"], 1787180575.0, {}, "text"),
        # 灵枢语音：内联规则写入后应能通过校验
        (["voice", "dialogue", "user", "session:lingshu-1"], 1787022564.0, {}, "text"),
    ]
    for tags, ts, cs, mod in cases:
        p = from_legacy(tags, ts, cs, mod)
        print(json.dumps(json.loads(p.to_json()), ensure_ascii=False, indent=1))
        print("  strength:", p.strength(), "| traceable:", p.is_traceable())
        print("  tags:", p.to_tags(["voice", "dialogue"]))
        for x in p.validate():
            print("  ⚠", x)
        print("-" * 70)
