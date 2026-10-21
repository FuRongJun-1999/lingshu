# -*- coding: utf-8 -*-
"""evidence · 证据账本（置信度由证据推导，而非裸浮点随手加减）
============================================================================
问题（同一根因，多件 issue 的共同症状）：
    节点置信度是 `nodes.confidence` 一个裸浮点，各入口自行 `+delta` / 直写 1.0：
      · M5 去重命中即 `+0.02`——同一来源把同一句话重复写 25 次就到 1.0
        （#29 / #142）；失败经验并进成功节点也照样增信（#225）；
      · 系统自产内容（反思链归档、洞见自报）回流成「证据」（#154 / #38）；
      · 矛盾/反证无处入账，置信度只升不降（#29 / #103）；
      · 召回终排不看置信度，也不告诉调用方「这条有没有佐证」（#103）。
    结果：宿主 AI 读到的记忆里，「被重复说过的话」与「被独立证实的事」不可区分——
    这是记忆层把幻觉喂回模型的通道。

本件做什么（只新增，不改既有默认行为）：
    1. `evidence` 表：每条证据 = (节点, 来源, 极性 支持/反驳, 权重, 是否自产, 备注, 时间)。
    2. 置信度**由账本推导**（Beta(1,1) 先验的后验均值）：
         conf = (1 + S) / (2 + S + C)
       S / C = 按**来源去重**后的支持 / 反驳权重和（同一来源只取其最大一条）。
       ⇒ 同一来源重复 N 次 = 1 次；自产证据记账但**不计分**；反证真实扣分。
    3. 状态（白箱、阈值显式）：
         unverified    无任何计分支持
         single_source 仅 1 个独立来源支持、无反驳
         corroborated  ≥ CORROBORATION_MIN 个独立来源支持、无反驳
         contested     支持与反驳并存
         refuted       仅有反驳（或 conf < REFUTED_BELOW）
    4. 只读报告 `assess()`：调用方（宿主 AI / MCP 工具）可据此决定
       「当事实说 / 带保留说 / 不说」。

边界（如实）：
    - 本件**不**判定两段文本是否语义矛盾（#142 的否定词/数值盲区仍在 M5 匹配器里）；
      它只保证：一旦有人把反证记进来，置信度会降，且不会被重复写入「养」回去。
    - 「来源是否真的独立」由调用方声明的 source 字符串决定；本件只做规范化
      （去首尾空白、小写），不做身份核验。
    - 默认关闭：引擎侧需显式 `enable_evidence_ledger()` 或设
      `LINGSHU_EVIDENCE_LEDGER=1`，否则一切路径与改前逐位一致。
"""
from __future__ import annotations

import math
import time
import uuid
from typing import Dict, List, Optional

SUPPORT = 1
CONTRADICT = -1

CORROBORATION_MIN = 2      # 「已佐证」所需的最少独立支持来源数
REFUTED_BELOW = 0.3        # 推导置信度低于此值且无支持 ⇒ refuted
PRIOR_A = 1.0              # Beta 先验（支持侧）
PRIOR_B = 1.0              # Beta 先验（反驳侧）

STATUS_LABEL_ZH = {
    "unverified": "未证实",
    "single_source": "单一来源",
    "corroborated": "已佐证",
    "contested": "有争议",
    "refuted": "已被反驳",
}

_DDL = """
CREATE TABLE IF NOT EXISTS evidence (
    id TEXT PRIMARY KEY,
    node_id TEXT NOT NULL,
    source TEXT NOT NULL,
    polarity INTEGER NOT NULL,
    weight REAL NOT NULL,
    self_generated INTEGER NOT NULL DEFAULT 0,
    note TEXT,
    created_at REAL NOT NULL
)
"""
_IDX = "CREATE INDEX IF NOT EXISTS idx_evidence_node ON evidence(node_id)"


#: `Role.PRIMARY` 的枚举**值**（`lingshu/core/core.py`：`PRIMARY = "primary"`）。
#: 这里只记值、不 import `Role`：`core.py` 是在函数内惰性 import 本模块的
#: （`enable_evidence_ledger`），本模块若在顶层 `from .core import Role`
#: 会形成**循环导入**。层与角色的**单一真源**仍取 `self.store`
#: （`LayeredStore.SHARED_LAYERS` / `LayeredStore.role`），与 `add_node` 同源。
_PRIMARY_ROLE_VALUE = "primary"


def _value_of(x):
    """枚举取 `.value`，非枚举原样返回（容忍把 role 直接设成字符串的用法）。"""
    return getattr(x, "value", x)


def normalize_source(source: Optional[str]) -> str:
    s = (source or "").strip().lower()
    return s or "unattributed"


def _check_weight(weight) -> float:
    try:
        w = float(weight)
    except (TypeError, ValueError):
        raise ValueError(f"evidence weight 必须是数值，收到 {weight!r}")
    if not math.isfinite(w) or w <= 0.0 or w > 1.0:
        raise ValueError(f"evidence weight 须在 (0, 1] 且有限，收到 {weight!r}")
    return w


class EvidenceLedger:
    """挂在 LayeredStore 上的证据账本。复用 store 的连接与事务边界。

    首次初始化需在调用方事务之外完成；建表成功后才由引擎缓存本对象。
    后续写入可参与调用方事务，证据、置信度同步与传播共同提交或回滚。
    """

    def __init__(self, store):
        self.store = store
        self.on_change = None  # 证据变化回调（TMS 用于下游传播；None = 无）
        with store._lock:
            if store.conn.in_transaction:
                raise RuntimeError("首次启用证据账本需要独立事务；请先结束调用方事务")
            with store._structure_event_transaction() as c:
                c.execute(_DDL)
                c.execute(_IDX)

    # ------------------------------------------------------------ 写权限闸
    def _assert_shared_layer_writable(self, node_id: str) -> None:
        """共享层写权限闸（#765 / #774）：非 PRIMARY 不得经证据账本写共享层。

        缺陷形态（修前 `origin/main`）：`EvidenceLedger` 的写面（`record` / `sync`）
        **没有任何层/角色闸** ⇒ SUB 实例拿到账本对象后直调 `record(...,
        polarity=CONTRADICT)` 即可把 PRIMARY 建的**锚点/结构层**节点
        `confidence 1.000000 → 0.333333`（`sync` 单独调则 → 0.5），且另一
        PRIMARY 实例读回一致——引擎侧 `add_evidence` 的角色闸（若有）拦不住
        「绕过引擎、直调账本对象」这条路。

        判据（与 `LayeredStore.add_node` 的共享层闸**同款、同源**）：
        节点所在层 ∈ `self.store.SHARED_LAYERS` 且 `self.store.role != PRIMARY`
        ⇒ `PermissionError`。层与角色都从 `self.store` 上取（单一真源），
        本模块**不** import `core`（避免循环导入，见 `_PRIMARY_ROLE_VALUE` 注）。

        ★本笔（#769 / #780）**不再自持判据本体**：优先**委托** store 的唯一单点
        `assert_node_writable_by_role`（`core.py` 的 `layer_writable_by_role`）。
        原实现是本模块内联一份「层 ∈ SHARED_LAYERS 且 role != PRIMARY」——同一
        判据两份实现正是本族缺陷（#769 三条旁路 + #780 整链）的成因，故收归单点。
        对**鸭子类型** store（无该方法的测试替身 / 旧宿主）保留原内联兜底，避免
        本闸因宿主换形而静默失效（兜底判据与原实现逐字一致，行为不变）。

        边界（如实）：
          · 本闸是**角色/层判据**，不是身份核验——`role` 仍是进程内构造参数
            （该面属 issue #779，不在本条）。
          · 闸拦的是「写共享层」，**不拦「写本地层」**：SUB 对自己的
            context / knowledge / self 层节点记证据是合法用法，照旧放行。
          · 只覆盖 `EvidenceLedger` 自己的写面；引擎侧 `annotate_semantics` /
            `decay_cycle` 两条（#774 ③④，落在 `core.py`）本轮**未覆盖**，已登记。
          · 闸只认**节点在库中的层**；节点不存在时交由调用方既有的
            `KeyError` 语义处理（`record` 已先判存在性）。
        """
        # ★本笔（#769/#780）：判据收归 store 的唯一单点（存在性也由该单点处理：
        # 节点不存在 ⇒ 放行，交由调用方既有语义）。
        single = getattr(self.store, "assert_node_writable_by_role", None)
        if callable(single):
            single(node_id, "经证据账本写入共享层节点的 confidence")
            return
        # 兜底（鸭子类型 store）：与原实现逐字一致的判据，行为不变。
        node = self.store.get_node(node_id)
        if node is None:
            return
        layer = getattr(node, "layer", None)
        if layer is None:
            return
        role = getattr(self.store, "role", None)
        if _value_of(role) == _PRIMARY_ROLE_VALUE:
            return
        shared = getattr(self.store, "SHARED_LAYERS", None) or ()
        if layer in shared:
            raise PermissionError(
                f"role={_value_of(role)} 无权写入共享层（{_value_of(layer)}），"
                "共享层由父节点主控"
            )

    # ------------------------------------------------------------ 写
    def record(self, node_id: str, source: Optional[str], polarity: int = SUPPORT,
               weight: float = 1.0, self_generated: bool = False,
               note: str = "") -> str:
        if polarity not in (SUPPORT, CONTRADICT):
            raise ValueError(f"polarity 只能是 SUPPORT(1) / CONTRADICT(-1)，收到 {polarity!r}")
        w = _check_weight(weight)
        eid = uuid.uuid4().hex
        with self.store._structure_event_transaction() as c:
            if not node_id or self.store.get_node(node_id) is None:
                raise KeyError(f"节点不存在：{node_id!r}")
            # #774：闸必须在 INSERT **之前**——只在 `sync` 落闸会留下孤儿证据行
            # （record 先 INSERT 再调 sync；SUB 被拒时 evidence 表已多一行）。
            self._assert_shared_layer_writable(node_id)
            c.execute(
                "INSERT INTO evidence (id, node_id, source, polarity, weight, "
                "self_generated, note, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (eid, node_id, normalize_source(source), int(polarity), w,
                 1 if self_generated else 0, (note or "")[:500], time.time()))
            self.sync(node_id)
            if self.on_change is not None:
                self.on_change(node_id)
        return eid

    def sync(self, node_id: str) -> float:
        """把推导出的置信度写回 nodes.confidence，使既有读路径看到同一个值。

        #765 / #774：本方法**直接 UPDATE 共享层节点的 confidence**（0.5 那格），
        故同样要先过 `_assert_shared_layer_writable`——闸落在 UPDATE 之前，
        被拒时零状态变更。
        """
        with self.store._structure_event_transaction() as c:
            self._assert_shared_layer_writable(node_id)
            conf = self.assess(node_id)["confidence"]
            c.execute("UPDATE nodes SET confidence=? WHERE id=?", (conf, node_id))
        return conf

    # ------------------------------------------------------------ 读
    def entries(self, node_id: str) -> List[Dict]:
        rows = self.store.conn.execute(
            "SELECT id, source, polarity, weight, self_generated, note, created_at "
            "FROM evidence WHERE node_id=? ORDER BY created_at, rowid", (node_id,)).fetchall()
        return [{"id": r[0], "source": r[1], "polarity": r[2], "weight": r[3],
                 "self_generated": bool(r[4]), "note": r[5], "created_at": r[6]}
                for r in rows]

    def assess(self, node_id: str) -> Dict:
        support: Dict[str, float] = {}
        contra: Dict[str, float] = {}
        self_count = 0
        rows = self.entries(node_id)
        for e in rows:
            if e["self_generated"]:
                self_count += 1
                continue
            bucket = support if e["polarity"] == SUPPORT else contra
            bucket[e["source"]] = max(bucket.get(e["source"], 0.0), e["weight"])
        S, C = sum(support.values()), sum(contra.values())
        conf = (PRIOR_A + S) / (PRIOR_A + PRIOR_B + S + C)
        if support and contra:
            status = "contested"
        elif contra or (not support and conf < REFUTED_BELOW):
            status = "refuted"
        elif len(support) >= CORROBORATION_MIN:
            status = "corroborated"
        elif len(support) == 1:
            status = "single_source"
        else:
            status = "unverified"
        return {
            "node_id": node_id,
            "confidence": round(conf, 6),
            "status": status,
            "status_zh": STATUS_LABEL_ZH[status],
            "support_sources": sorted(support),
            "contradict_sources": sorted(contra),
            "self_generated_ignored": self_count,
            "entries": len(rows),
        }
