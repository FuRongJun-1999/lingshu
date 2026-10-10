"""真值维护（JTMS 式依赖撤回），叠在证据账本之上（默认关闭）。

动机：
  证据账本（evidence.py）让**单个**命题的置信度由它自己的证据推导；但推理产物——
  「因为 P1、P2，所以 C」——在 P1 被反证推翻后依旧保持原置信度，宿主照样把 C 当事实说。
  这是幻觉的另一条来路：前提死了，结论还活着。

模型（白箱、确定性、零 LLM，D-005）：
  - 依据（justification）：结论 c ← 前提集合 {p1..pk}（合取）。一个结论可有多条依据（析取）。
    存于 `justifications(jid, conclusion, source, created_at)` 与 `justification_premises(jid, premise)`。
  - 信念：
        own(n)      = 证据账本对 n 自身证据推导的置信度与状态
        jsup(c)     = max_j  min_{p∈j} belief(p)          （无依据则不定义）
        belief(c)   = own(c)                               若 c 无依据，或 own 状态为「已佐证」
                    = min(own(c), jsup(c))                 否则
    即：TMS **只撤回、不放大**——前提再可靠也不替结论增信（推导不是证据）；结论若有
    ≥2 个独立外部来源佐证，则不受前提影响（它已被独立证实）。
  - 撤回标记：jsup(c) < REFUTED_BELOW 且 c 未被独立佐证 ⇒ 状态 `undermined`（前提已被推翻），
    并列出推翻它的前提。
  - 传播：任一节点的证据变化 → 收集其下游闭包 → 以各自 own 为初值做 Gauss-Seidel 迭代至不动点
    （值域是有限个 own 值的 min/max 组合，至多 |闭包|+1 轮收敛；环可处理）→ 把 belief 写回
    nodes.confidence，使既有读路径（recall 的排序、阈值）看到同一个值。
"""
from __future__ import annotations

import json
import time
import uuid
from collections import deque
from typing import Dict, Iterable, List, Optional

from .evidence import REFUTED_BELOW

_DDL = (
    "CREATE TABLE IF NOT EXISTS justifications ("
    " jid TEXT PRIMARY KEY, conclusion TEXT NOT NULL, source TEXT, created_at REAL)",
    "CREATE TABLE IF NOT EXISTS justification_premises ("
    " jid TEXT NOT NULL, premise TEXT NOT NULL, PRIMARY KEY (jid, premise))",
    "CREATE INDEX IF NOT EXISTS idx_just_concl ON justifications(conclusion)",
    "CREATE INDEX IF NOT EXISTS idx_just_prem ON justification_premises(premise)",
)


class TruthMaintenance:
    def __init__(self, store, ledger):
        self.store = store
        self.ledger = ledger
        with store._lock:
            c = store.conn
            for sql in _DDL:
                c.execute(sql)
            c.commit()

    # ------------------------------------------------------------ 写
    def add_justification(self, conclusion: str, premises: Iterable[str],
                          source: Optional[str] = None) -> str:
        prem = sorted({p for p in premises if p})
        if not prem:
            raise ValueError("依据至少需要一个前提")
        if conclusion in prem:
            raise ValueError("结论不能作为自己的前提")
        for nid in [conclusion] + prem:
            if self.store.get_node(nid) is None:
                raise KeyError(f"节点不存在：{nid!r}")
        jid = uuid.uuid4().hex
        with self.store._lock:
            c = self.store.conn
            c.execute("INSERT INTO justifications VALUES (?,?,?,?)",
                      (jid, conclusion, source or "", time.time()))
            c.executemany("INSERT INTO justification_premises VALUES (?,?)",
                          [(jid, p) for p in prem])
            c.commit()
        self.propagate([conclusion])
        return jid

    # ------------------------------------------------------------ 读
    def justifications_of(self, node_id: str) -> List[List[str]]:
        c = self.store.conn
        out = []
        for (jid,) in c.execute("SELECT jid FROM justifications WHERE conclusion=? "
                                "ORDER BY created_at, rowid", (node_id,)).fetchall():
            out.append([r[0] for r in c.execute(
                "SELECT premise FROM justification_premises WHERE jid=? ORDER BY premise",
                (jid,))])
        return out

    def dependents_of(self, node_id: str) -> List[str]:
        return [r[0] for r in self.store.conn.execute(
            "SELECT DISTINCT j.conclusion FROM justification_premises p "
            "JOIN justifications j ON j.jid = p.jid WHERE p.premise=?", (node_id,))]

    def _closure(self, roots: Iterable[str]) -> List[str]:
        seen, order, q = set(), [], deque(roots)
        while q:
            n = q.popleft()
            if n in seen:
                continue
            seen.add(n)
            order.append(n)
            q.extend(self.dependents_of(n))
        return order

    def _belief(self, n: str, own: Dict[str, Dict], belief: Dict[str, float]) -> float:
        o = own[n]
        js = self._just_cache.get(n)
        if not js or o["status"] == "corroborated":
            return o["confidence"]

        def b(p):  # 闭包内取本轮迭代值；闭包外取已维护的 nodes.confidence
            return belief[p] if p in belief else self._external_belief(p)

        jsup = max(min(b(p) for p in j) for j in js)
        return min(o["confidence"], jsup)

    def _external_belief(self, p: str) -> float:
        node = self.store.get_node(p)
        return float(node.confidence) if node is not None else 0.0

    # ------------------------------------------------------------ 传播
    def propagate(self, roots: Iterable[str]) -> Dict[str, float]:
        """重算 roots 及其下游闭包的信念并写回 nodes.confidence；返回 {节点: 新信念}。"""
        nodes = self._closure(roots)
        if not nodes:
            return {}
        own = {n: self.ledger.assess(n) for n in nodes}
        self._just_cache = {n: self.justifications_of(n) for n in nodes}
        belief = {n: own[n]["confidence"] for n in nodes}
        for _ in range(len(nodes) + 1):
            changed = False
            for n in nodes:
                b = self._belief(n, own, belief)
                if abs(b - belief[n]) > 1e-12:
                    belief[n] = b
                    changed = True
            if not changed:
                break
        with self.store._lock:
            c = self.store.conn
            c.executemany("UPDATE nodes SET confidence=? WHERE id=?",
                          [(round(b, 6), n) for n, b in belief.items()])
            c.commit()
        self._just_cache = {}
        return belief

    def assess(self, node_id: str) -> Dict:
        """账本评估 + TMS 叠加：belief、是否 undermined、推翻它的前提。"""
        ev = dict(self.ledger.assess(node_id))
        js = self.justifications_of(node_id)
        ev["justifications"] = js
        node = self.store.get_node(node_id)
        belief = float(node.confidence) if node is not None else ev["confidence"]
        ev["belief"] = round(belief, 6)
        ev["undermined_by"] = []
        if js and ev["status"] != "corroborated":
            jsup = max(min(self._external_belief(p) for p in j) for j in js)
            if jsup < REFUTED_BELOW:
                ev["undermined_by"] = sorted({p for j in js for p in j
                                              if self._external_belief(p) < REFUTED_BELOW})
                ev["status"] = "undermined"
                ev["status_zh"] = "前提已被推翻"
        return ev
