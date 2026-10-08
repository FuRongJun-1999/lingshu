# -*- coding: utf-8 -*-
"""activation · 激活引擎（ACTIVATION-REV1 · 存算一体认知图的上下文层）
====================================================================================
荣 2026-09-06 裁定（grill 访谈四条）：
  1. 四件缺口一体交付——激活子图(状态容器)/CSPMN 并行查询(读取)/条件衰减(传播)/
     重构路径审计(留痕)是一个激活引擎的四个面,分里程碑不拆模块;
  2. 激活状态独立于内容记忆(工作记忆/长时记忆分离)——独立激活表+快照可序列化;
  3. CSPMN 语义用 numpy 全图并行实现(同步询问→投票→扩散),蜂群分布式为规模选项(接口预留);
  4. 效果验证按第五篇第九节:消融实验,重复犯错次数说了算(见 tests/test_activation.py 消融段)。

理论锚点:
  - 白箱认知记忆设计 v1 §8:记忆=连接锚点+重构(非内容存储);读取=并行同步查询
    (CSPMN:同步询问→评估关联→投票→候选→沿关联边扩散→回忆=同步构建)——存算一体;
  - 上下文=当前条件空间+激活子图(工作态,Retained Reasoning),非窗口文本;
  - 第五篇:检索不全塞/负记忆同价/条件绑定。

核心公式:
  扩散: activation(v, hop+1) = max over u∈N(v): activation(u,hop) × decay(u→v)
  衰减: decay(u→v) = base(边类型) × cond_match(两端条件空间重合率)
  自条件(ELF §3.3 同构补充,2026-09-10):
        act₀ = max(SELF_CONDITION_WEIGHT × act_prev(上一步工作态), seeds(query))
        ——迭代系统每步以自身上一时刻的输出为条件;本版先验只增不减(不引入噪声/流)。
  审计: 每次激活全程留痕(种子/每步激活/工作态/自条件)→JSONL(完全确认 2.9.3a 数据基础)

纯 numpy+scipy.sparse+标准库(视觉/图域依赖惯例,D-005 零 LLM)。
"""
from __future__ import annotations
import json
import os
import sqlite3
import time
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np
try:
    from scipy.sparse import csr_matrix
except ImportError:  # scipy 缺失时降级为 edge-list 传播(声明局限)
    csr_matrix = None

ALGO = "activation-0.1"
# 数据根：环境变量优先（不写本机路径字面量）
_DATA_ROOT = os.environ.get("LINGSHU_DATA_ROOT", "data")
DB = os.path.join(_DATA_ROOT, "aeis_memory.db")
AUDIT_PATH = os.path.join(_DATA_ROOT, "activation_audit.jsonl")

# 条件衰减基准(边类型,工程默认值——条件智能:不同关系的信息传递保真度不同)
EDGE_BASE_DECAY: Dict[str, float] = {
    "causal": 0.85,        # 因果:强传导
    "similar": 0.75,       # 相似:语义联想
    "counterpart": 0.75,   # 对应
    "hierarchical": 0.70,  # 层级:上下位
    "sequential": 0.60,    # 时序:邻近
    "spatial": 0.50,       # 空间:位置关联(最弱)
}
DEFAULT_DECAY = 0.5     # 未注册边类型

# self-conditioning 回注权重(工程默认值,与 EDGE_BASE_DECAY 同类;可调用级覆盖)。
# 0.0 = 关闭(与 ACTIVATION-REV1 行为严格等价);0.5 = 推荐值(先验与种子同权)。
SELF_CONDITION_WEIGHT = 0.5


def cond_match(cs_a: Dict, cs_b: Dict) -> float:
    """条件空间重合率(4 维:观测位置/工具/时间窗/存在约束——逐维比对)。"""
    if not cs_a or not cs_b:
        return 0.5  # 单边无条件声明→中性(不奖励也不惩罚)
    keys = ("observation_position", "observation_tool", "existence_constraint")
    hits = sum(1 for k in keys if cs_a.get(k) == cs_b.get(k))
    ta, tb = cs_a.get("time_window"), cs_b.get("time_window")
    if ta and tb and isinstance(ta, list) and isinstance(tb, list) and len(ta) == 2 and len(tb) == 2:
        lo, hi = max(ta[0], tb[0]), min(ta[1], tb[1])
        if hi > lo:  # 时间窗有交叠
            hits += 1
        elif max(0.0, lo - hi) < 86400:  # 相邻窗(24h 内)算半匹配
            hits += 0.5
    return hits / 4.0


class ActivationEngine:
    """激活引擎:在认知图(aeis_memory.db)上维护工作态(激活子图)。

    用法:
      eng = ActivationEngine()
      sub = eng.activate("查询文本", conditions={...}, workset="sess_A")   # 构建/更新工作态
      # 工作态演化(自条件,ELF §3.3 同构):上一步输出作下一步条件
      eng.activate("下一轮查询", workset="sess_A",
                   self_condition=SELF_CONDITION_WEIGHT)                 # 继承上一时刻工作态
      eng.carry_vector("sess_A")                                           # 显式读"上一步输出"
      snap = eng.export_workset("sess_A")                                  # 快照可序列化
      eng.import_workset(snap)                                             # 跨进程恢复
      state = eng.workset_state("sess_A")                                  # 当前工作态(执行层可见)
    """

    def __init__(self, db_path: str = DB, audit_path: str = AUDIT_PATH):
        self.db_path = db_path
        self.audit_path = audit_path
        self._ensure_tables()
        self._adj = None            # 缓存稀疏邻接(node_idx 矩阵)
        self._node_index = None     # node_id → idx
        self._edge_meta = None      # (src_idx, dst_idx, edge_type) 列表

    # ---------------- 表结构(独立激活表,与内容记忆分离) ----------------
    def _ensure_tables(self):
        con = sqlite3.connect(self.db_path, timeout=30)
        con.execute("""CREATE TABLE IF NOT EXISTS activation_nodes (
            id TEXT PRIMARY KEY, workset TEXT NOT NULL, node_id TEXT NOT NULL,
            activation REAL NOT NULL, source TEXT NOT NULL, hop INTEGER NOT NULL, ts REAL NOT NULL)""")
        con.execute("CREATE INDEX IF NOT EXISTS idx_act_ws ON activation_nodes(workset)")
        con.commit(); con.close()

    def _con(self):
        return sqlite3.connect(self.db_path, timeout=30)

    # ---------------- 图缓存(numpy 全图矩阵化) ----------------
    def _load_graph(self):
        """全图→(稀疏邻接矩阵, 节点索引, 边元数据)——CSPMN 全图并行的底物。缓存。"""
        if self._adj is not None:
            return
        con = self._con()
        cur = con.cursor()
        cur.execute("SELECT id FROM nodes")
        node_ids = [r[0] for r in cur.fetchall()]
        cur.execute("SELECT source_id, target_id, relation_type, condition_space FROM edges")
        edges = cur.fetchall()
        con.close()
        idx = {nid: i for i, nid in enumerate(node_ids)}
        n = len(node_ids)
        rows, cols, decays = [], [], []
        for src, dst, rtype, cs_json in edges:
            si, di = idx.get(src), idx.get(dst)
            if si is None or di is None:
                continue
            try:
                cs_e = json.loads(cs_json) if cs_json else {}
            except (json.JSONDecodeError, TypeError):
                cs_e = {}
            d = EDGE_BASE_DECAY.get(rtype, DEFAULT_DECAY)
            # 边条件空间与图整体条件基准的重合(边级衰减修正)——保守用 1.0(边级 cond_match 在传播时按节点对算)
            rows.append(si); cols.append(di); decays.append(d)
        if csr_matrix is not None and rows:
            adj = csr_matrix((np.array(decays, dtype=np.float32), (rows, cols)),
                             shape=(n, n))
        else:
            adj = None  # 降级:edge-list 传播
        self._adj = adj
        self._node_index = idx
        self._edge_meta = (rows, cols, decays)
        self._node_ids = node_ids
        # 节点条件空间矩阵化(节点级 cond_match 用)
        cur_cs = {}
        con = self._con()
        cur = con.cursor()
        cur.execute("SELECT id, condition_space FROM nodes")
        for nid, cs_json in cur.fetchall():
            try:
                cur_cs[nid] = json.loads(cs_json) if cs_json else {}
            except (json.JSONDecodeError, TypeError):
                cur_cs[nid] = {}
        con.close()
        self._node_cond = cur_cs

    # ---------------- 种子生成(同步询问→投票) ----------------
    def _seed_by_text(self, query: str, top_k: int = 12) -> List[Tuple[str, float]]:
        """种子:中文二元组倒排——同步询问全图(所有含查询词元的节点),计数投票。"""
        toks = {query[i:i + 2] for i in range(len(query) - 1)} | {query}
        con = self._con()
        cur = con.cursor()
        cur.execute("SELECT id, content, tags FROM nodes")
        scores = []
        for nid, content, tags_json in cur.fetchall():
            hay = content or ""
            try:
                hay += " " + " ".join(json.loads(tags_json)) if tags_json else ""
            except (json.JSONDecodeError, TypeError):
                pass
            s = sum(1 for t in toks if t in hay) / max(1, len(toks))
            if s > 0:
                scores.append((nid, s))
        con.close()
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    # ---------------- 四件之一+之二+之三:activate(查询→并行扩散→工作态) ----------------
    def activate(self, query: str, conditions: Optional[Dict] = None,
                 workset: str = "default", top_k: int = 12,
                 hops: int = 2, act_floor: float = 0.15,
                 self_condition: float = 0.0,
                 prior_workset: Optional[str] = None) -> Dict:
        """激活一次:种子(同步询问投票)→沿边扩散(条件衰减)→写入激活表+审计。

        空图返回空工作态，并保留各步的零激活审计记录。

        self-conditioning(ELF §3.3 同构 · 显式接口约定):
          ELF:上一步预测 x̂′ 作为下一步的条件输入;本引擎同构——"上一步工作态"
          作为本步先验激活,与种子一起扩散(不额外增加查询路径):
              act₀ = max(SELF_CONDITION_WEIGHT × act_prev, seeds(query))
          差别:我们不引入噪声/连续流,先验是**上一时刻的显式工作态**(白箱可审计),
          且回注的节点以 source="self_condition" 落表(演化可追溯)。
          参数:
            self_condition: 回注权重;0.0=关闭(默认,与 REV1 行为严格等价);
                            推荐 SELF_CONDITION_WEIGHT(0.5)。
            prior_workset:  先验来源;None 且开启 → 取本 workset 上一步(正牌自条件);
                            显式给名 → 跨工作态继承(会话迁移/上下文接力)。
          声明局限(独立落点,本版未实现):ELF 的 SDE "噪声重注"纠错
          (周期性激活回退+重扩散,防激活路径锁死)——本版先验只增不减、不回退。
        """
        self._load_graph()
        t0 = time.time()
        seeds = self._seed_by_text(query, top_k)
        idx = self._node_index
        n = len(self._node_ids)
        act = np.zeros(n, dtype=np.float32)
        seed_ids = []
        for nid, s in seeds:
            i = idx.get(nid)
            if i is not None:
                act[i] = max(act[i], s)
                seed_ids.append(nid)
        # self-conditioning:上一步工作态作本步先验(先验∩图域内节点才有效)
        prior_name = prior_workset or workset
        carried_ids: List[str] = []
        if self_condition > 0.0:
            for nid, a in self.carry_vector(prior_name).items():
                i = idx.get(nid)
                if i is None:
                    continue
                carried_ids.append(nid)
                act[i] = max(act[i], float(a) * self_condition)
        path = [{"step": 0, "phase": "seed", "activated": seed_ids,
                 "scores": {nid: round(s, 4) for nid, s in seeds}}]
        if self_condition > 0.0:
            path.append({"step": 0, "phase": "self_condition",
                         "prior_workset": prior_name,
                         "weight": self_condition, "carried": len(carried_ids)})
        # 逐跳扩散(条件衰减)
        for hop in range(1, hops + 1):
            prop = np.zeros(n, dtype=np.float32)
            if self._adj is not None:
                # activation × 邻接(带衰减)——稀疏矩阵乘即全图并行传播
                prop = self._adj.T @ act    # (dst ← src):prop[dst]=Σ act[src]·decay
            else:  # edge-list 降级
                rows, cols, decays = self._edge_meta
                for si, di, d in zip(rows, cols, decays):
                    if act[si] > 0:
                        prop[di] = max(prop[di], act[si] * d)
            # 节点级条件衰减修正(边 base × 节点对条件重合率)——按激活邻域逐点算
            nz = np.nonzero(prop)[0]
            for di in nz:
                pass  # V0:节点对级 cond_match 在稀疏边级计算成本高,以边类型 base 为准(声明局限)
            act = np.maximum(act, prop)
            newly = [self._node_ids[i] for i in np.nonzero(act >= act_floor)[0]]
            path.append({"step": hop, "phase": "propagate",
                         "activated_count": len(newly),
                         "max_act": round(float(act.max()), 4) if n else 0.0})
        # 阈值截断→工作态
        keep = np.nonzero(act >= act_floor)[0]
        members = [(self._node_ids[i], round(float(act[i]), 4)) for i in keep]
        members.sort(key=lambda x: -x[1])
        sources = {nid: "self_condition" for nid in carried_ids}
        self._write_workset(workset, members, query, sources)
        # 四件之四:重构路径审计(全程留痕)
        audit = {"algo": ALGO, "ts": time.time(), "workset": workset,
                 "query": query, "conditions": conditions or {},
                 "seeds": len(seed_ids), "steps": path,
                 "self_condition": {"weight": self_condition,
                                    "prior_workset": prior_name if self_condition > 0.0 else None,
                                    "carried": len(carried_ids)},
                 "workset_size": len(members),
                 "top": [m[0] for m in members[:10]],
                 "latency_ms": round((time.time() - t0) * 1000, 1)}
        self._append_audit(audit)
        return {"status": "ok", "workset": workset, "size": len(members),
                "top": members[:10], "seeds": len(seed_ids), "hops": hops,
                "carried": len(carried_ids),
                "latency_ms": audit["latency_ms"]}

    def _write_workset(self, workset: str, members: List[Tuple[str, float]], query: str,
                       sources: Optional[Dict[str, str]] = None):
        """落表:members=(node_id, 激活强度);sources 标注来源(propagate/self_condition)。"""
        con = self._con()
        con.execute("DELETE FROM activation_nodes WHERE workset=?", (workset,))
        ts = time.time()
        src_map = sources or {}
        for nid, a in members:
            con.execute(
                "INSERT OR REPLACE INTO activation_nodes VALUES (?,?,?,?,?,?,?)",
                ("act_%s_%d" % (uuid.uuid4().hex[:8], int(ts * 1000)),
                 workset, nid, a, src_map.get(nid, "propagate"), 0, ts))
        con.commit(); con.close()

    def _append_audit(self, audit: Dict):
        os.makedirs(os.path.dirname(self.audit_path) or ".", exist_ok=True)
        with open(self.audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(audit, ensure_ascii=False) + "\n")

    # ---------------- 工作态读写(执行层接口) ----------------
    def carry_vector(self, workset: str = "default") -> Dict[str, float]:
        """上一步输出(显式接口 · ELF §3.3):工作态 node_id→激活强度,即自条件的先验向量。

        纯查询(不写库):既供 activate 内部回注,也供调用方/测试直接读取
        "上一时刻的工作态"——把原本隐含的演化关系显式化为可观测契约。
        """
        con = self._con()
        cur = con.cursor()
        cur.execute("SELECT node_id, activation FROM activation_nodes WHERE workset=?",
                    (workset,))
        rows = cur.fetchall()
        con.close()
        return {r[0]: float(r[1]) for r in rows}

    def workset_state(self, workset: str = "default", limit: int = 20) -> Dict:
        """当前工作态:激活节点+强度+内容摘要(执行层上下文的来源)。"""
        con = self._con()
        cur = con.cursor()
        cur.execute("SELECT node_id, activation FROM activation_nodes WHERE workset=? "
                    "ORDER BY activation DESC LIMIT ?", (workset, limit))
        rows = cur.fetchall()
        con.close()
        members = []
        if rows:
            con = self._con(); cur = con.cursor()
            for nid, a in rows:
                try:
                    cur.execute("SELECT content, layer, importance FROM nodes WHERE id=?", (nid,))
                    r = cur.fetchone()
                except sqlite3.OperationalError:
                    r = None  # 恢复库无内容表(独立激活库)——仅返回激活值
                members.append({"node_id": nid, "activation": a,
                                "content": (r[0] or "")[:80] if r else "",
                                "layer": r[1] if r else None,
                                "importance": r[2] if r else None})
            con.close()
        return {"workset": workset, "size": len(rows), "members": members}

    def export_workset(self, workset: str) -> Dict:
        """快照(可序列化——跨进程恢复,Retained Reasoning 的持久形态)。"""
        con = self._con()
        cur = con.cursor()
        cur.execute("SELECT node_id, activation, source, hop, ts FROM activation_nodes WHERE workset=?",
                    (workset,))
        rows = cur.fetchall()
        con.close()
        return {"algo": ALGO, "workset": workset, "ts": time.time(),
                "members": [{"node_id": r[0], "activation": r[1],
                             "source": r[2], "hop": r[3], "ts": r[4]} for r in rows]}

    def import_workset(self, snapshot: Dict, workset: Optional[str] = None) -> Dict:
        """恢复快照(激活状态重建——进程重启后工作态不失)。"""
        ws = workset or snapshot["workset"]
        con = self._con()
        con.execute("DELETE FROM activation_nodes WHERE workset=?", (ws,))
        ts = time.time()
        for m in snapshot["members"]:
            con.execute("INSERT OR REPLACE INTO activation_nodes VALUES (?,?,?,?,?,?,?)",
                        ("act_%s_%d" % (uuid.uuid4().hex[:8], int(ts * 1000)),
                         ws, m["node_id"], m["activation"], "restored", m.get("hop", 0), ts))
        con.commit(); con.close()
        return {"status": "ok", "workset": ws, "restored": len(snapshot["members"])}


if __name__ == "__main__":
    eng = ActivationEngine()
    r1 = eng.activate("灵枢 认知图 上下文", workset="demo")
    r2 = eng.activate("激活 工作态 演化", workset="demo",
                      self_condition=SELF_CONDITION_WEIGHT)
    print(json.dumps({"first": r1, "second": r2}, ensure_ascii=False, indent=1)[:600])
    st = eng.workset_state("demo", 3)
    print(json.dumps(st, ensure_ascii=False)[:400])
