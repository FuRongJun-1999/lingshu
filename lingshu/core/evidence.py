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
    """挂在 LayeredStore 上的证据账本。复用 store 的连接与写锁。"""

    def __init__(self, store):
        self.store = store
        with store._lock:
            c = store.conn
            c.execute(_DDL)
            c.execute(_IDX)
            c.commit()

    # ------------------------------------------------------------ 写
    def record(self, node_id: str, source: Optional[str], polarity: int = SUPPORT,
               weight: float = 1.0, self_generated: bool = False,
               note: str = "") -> str:
        if polarity not in (SUPPORT, CONTRADICT):
            raise ValueError(f"polarity 只能是 SUPPORT(1) / CONTRADICT(-1)，收到 {polarity!r}")
        w = _check_weight(weight)
        if not node_id or self.store.get_node(node_id) is None:
            raise KeyError(f"节点不存在：{node_id!r}")
        eid = uuid.uuid4().hex
        with self.store._lock:
            c = self.store.conn
            c.execute(
                "INSERT INTO evidence (id, node_id, source, polarity, weight, "
                "self_generated, note, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (eid, node_id, normalize_source(source), int(polarity), w,
                 1 if self_generated else 0, (note or "")[:500], time.time()))
            c.commit()
        self.sync(node_id)
        return eid

    def sync(self, node_id: str) -> float:
        """把推导出的置信度写回 nodes.confidence，使既有读路径看到同一个值。"""
        conf = self.assess(node_id)["confidence"]
        with self.store._lock:
            c = self.store.conn
            c.execute("UPDATE nodes SET confidence=? WHERE id=?", (conf, node_id))
            c.commit()
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
