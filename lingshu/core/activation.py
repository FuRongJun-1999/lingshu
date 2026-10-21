# -*- coding: utf-8 -*-
"""activation · 激活引擎（ACTIVATION-REV1 · 存算一体认知图的上下文层）
====================================================================================
荣 2026-09-06 裁定（grill 访谈四条）：
  1. 四件缺口一体交付——激活子图(状态容器)/CSPMN 并行查询(读取)/条件衰减(传播)/
     重构路径审计(留痕)是一个激活引擎的四个面,分里程碑不拆模块;
  2. 激活状态独立于内容记忆(工作记忆/长时记忆分离)——独立激活表+快照可序列化;
  3. CSPMN 语义用 numpy 全图并行实现(同步询问→投票→扩散),蜂群分布式为规模选项(接口预留);
  4. 效果验证按第五篇第九节:消融实验,重复犯错次数说了算(本仓暂无该模块的消融验证面——`tests/test_activation.py` 不在本仓;现有定点验证面见 tests/test_issue7_161_293_314_53_activation_contract.py 与 tests/test_issue229_280_activation_suppression_bounded.py)。

理论锚点:
  - 白箱认知记忆设计 v1 §8:记忆=连接锚点+重构(非内容存储);读取=并行同步查询
    (CSPMN:同步询问→评估关联→投票→候选→沿关联边扩散→回忆=同步构建)——存算一体;
  - 上下文=当前条件空间+激活子图(工作态,Retained Reasoning),非窗口文本;
  - 第五篇:检索不全塞/负记忆同价/条件绑定。

核心公式:
  扩散: activation(v, hop+1) = max over u∈N(v): activation(u,hop) × decay(u→v)
        ——max-plus 口径,**不跨跳累加**(scipy 稀疏与 edge-list 降级同口径);
        故 act 有界(≤ max(种子, 上一步先验)),不随跳数/轮数发散(issue #280)。
  抑制(负记忆同价,第五篇;issue #229):decay(u→v) < 0 的边(opposite)走抑制通路——
        supp(v) = max over u∈N(v): act(u)·|decay(u→v)|;act(v) ← max(act(v) − supp(v), 0),
        即相反记忆扣减目标激活(取最强抑制,不叠加),扣减后钳制到 0。
        判据:边衰减符号承自本文件 :15 锚点「第五篇:…负记忆同价…」;幅值 base=-1.0 为
        经验标定(满激活源 1.0 恰抵消目标满激活)——追不到理论常数。
  衰减: decay(u→v) = base(边类型) × cond_match(两端条件空间重合率)
        ——#53:此前 cond_match 定义后**零调用**(装载期是空循环),公式与实现不一致;
        现于 `_load_graph` 装载边时按节点对算一次(重合率静态,与逐跳算等价),
        稀疏与 edge-list 两后端同口径(条件绑定,第五篇定则)。
  自条件(ELF §3.3 同构补充,2026-09-10):
        act₀ = max(SELF_CONDITION_WEIGHT × act_prev(上一步工作态), seeds(query))
        ——迭代系统每步以自身上一时刻的输出为条件;本版先验只增不减(不引入噪声/流)。
  审计: 每次激活全程留痕(种子/每步激活/工作态/自条件)→JSONL(完全确认 2.9.3a 数据基础);
        落表元数据 source(seed/propagate/self_condition)与 hop(首次激活跳号,种子=0)
        一并留痕,快照可还原扩散路径(issue #314)。

纯 numpy+scipy.sparse+标准库(视觉/图域依赖惯例,D-005 零 LLM)。
#7 轻核承诺:core 不装任何 extras 也可**导入**(README「轻核：core 纯标准库，零依赖可跑」)
——numpy 缺失时本模块仍可 import,`ActivationEngine.activate()` 返回
`{"status": "activation_not_ready", ...}` 显式降级(与 core.py 各 `*_not_ready` 同口径),
不再以裸 `import numpy` 击穿整包导入;scipy 仍为可选稀疏后端(缺失降级 edge-list)。
"""
from __future__ import annotations
from contextlib import contextmanager
import json
import math
import os
import sqlite3
import time
import uuid
import warnings
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:  # #7:缺失时模块可导入,激活面由 activate() 显式降级
    np = None
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
# 符号语义(v0.2,issue #229):正=兴奋传导(见核心公式「扩散」);负=抑制传导,
#   绝对值=抑制强度,走独立抑制通路(见核心公式「抑制」)。未注册边类型默认兴奋(0.5)。
EDGE_BASE_DECAY: Dict[str, float] = {
    "causal": 0.85,        # 因果:强传导
    "similar": 0.75,       # 相似:语义联想
    "counterpart": 0.75,   # 对应
    "hierarchical": 0.70,  # 层级:上下位
    "sequential": 0.60,    # 时序:邻近
    "spatial": 0.50,       # 空间:位置关联(最弱)
    "opposite": -1.0,      # 相反/矛盾(负记忆同价,第五篇):抑制传导,幅值=经验标定
}
DEFAULT_DECAY = 0.5     # 未注册边类型

# 条件空间重合率的中性值(不奖励也不惩罚)——`cond_match` 的既有回落值。
# #709:同一常量也用作 `_load_graph` 里该函数意外异常时的兜底衰减因子(两处同源)。
COND_MATCH_NEUTRAL = 0.5

# self-conditioning 回注权重(工程默认值,与 EDGE_BASE_DECAY 同类;可调用级覆盖)。
# 0.0 = 关闭(与 ACTIVATION-REV1 行为严格等价);0.5 = 推荐值(先验与种子同权)。
SELF_CONDITION_WEIGHT = 0.5


def cond_match(cs_a: Dict, cs_b: Dict) -> float:
    """条件空间重合率(4 维:观测位置/工具/时间窗/存在约束——逐维比对)。

    #709:入参与 `time_window` **逐层**校验——守卫不能只查外层容器。库里的
    `condition_space` 是外部卡/灾备恢复产物（`Provenance.from_json` docstring 已
    逐字记载「core 自己的 ConditionSpace 规范化后会产出」`time_window=[null,null]`），
    故下列形状**都会出现**：`null` / `5` / `[1,2]`（整列非 dict）、
    `[null,null]` / `["a","b"]`（元素非有限数值）。旧实现下 `if ta and tb` 对
    `[None,None]` 为真（列表非空）⇒ 直接进 `max(ta[0], tb[0])` ⇒
    `TypeError: '>' not supported between instances of 'NoneType' and 'float'`。
    现一律回落 docstring 声明的中性值 `0.5`（「单边无条件声明→中性(不奖励也不惩罚)」），
    与 `null` / 非法 JSON 的既有回落口径一致。
    """
    if not isinstance(cs_a, dict) or not isinstance(cs_b, dict):
        return COND_MATCH_NEUTRAL  # #709:非 dict 入参(JSON 里的 5 / [1,2] / null)→中性
    if not cs_a or not cs_b:
        return COND_MATCH_NEUTRAL  # 单边无条件声明→中性(不奖励也不惩罚)
    keys = ("observation_position", "observation_tool", "existence_constraint")
    hits = sum(1 for k in keys if cs_a.get(k) == cs_b.get(k))

    def _finite_pair(v) -> bool:
        """#709:`time_window` 必须是「两个有限数值」——元素级判据（bool 不算数值）。"""
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            return False
        for x in v:
            if isinstance(x, bool) or not isinstance(x, (int, float)):
                return False
            if isinstance(x, float) and not math.isfinite(x):
                return False
        return True

    ta, tb = cs_a.get("time_window"), cs_b.get("time_window")
    if _finite_pair(ta) and _finite_pair(tb):
        lo, hi = max(ta[0], tb[0]), min(ta[1], tb[1])
        if hi > lo:  # 时间窗有交叠
            hits += 1
        elif max(0.0, lo - hi) < 86400:  # 相邻窗(24h 内)算半匹配
            hits += 0.5
    return hits / 4.0


class _BorrowedConnection:
    """借用调用方连接（#293）：`close()` 空转，其余方法逐字转发。

    引擎内部各处按「取连接 → 用完 close()」书写；`store=` / `conn=` 复用引擎连接
    （如 `:memory:`）时，真 `close()` 会把调用方的库一起关掉——本包装只让 `close()`
    空转，不改变 `execute` / `commit` / `cursor` 的语义。事务由激活引擎的
    `_transaction` 管理，借用已有事务时只操作自己的 SAVEPOINT。
    """

    def __init__(self, conn):
        self._conn = conn

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def close(self):
        pass


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

    def __init__(self, db_path: str = DB, audit_path: str = AUDIT_PATH,
                 conn=None, store=None):
        """构造激活引擎。

        #293：`db_path=":memory:"` 时每次 `sqlite3.connect` 都是**新空库**
        （建表/读 `nodes`/写工作态各落在不同临时库 ⇒ `no such table: nodes`），
        故必须复用同一连接：
          · `store=`（如 `SpacetimeMemoryEngine().store`）或 `conn=` → 借用其连接
            （`_BorrowedConnection` 保证 `close()` 不关调用方的库）；
          · 只给 `db_path=":memory:"` → 显式 `ValueError`（不再静默拿空库）；
          · 文件路径 → 父目录不存在时自动创建（旧实现在干净目录下构造即
            `unable to open database file`）。
        默认 `db_path` 仍相对 cwd（是否改为包内/用户目录属设计者裁，本件不动）。
        判据来源：本件 #293（分诊表 `triage_lingshu_ALL.md` 行 293 判定成立）。
        """
        if store is not None:
            self.db_path = store.db_path
            self._conn = _BorrowedConnection(store.conn)
        elif conn is not None:
            self.db_path = db_path
            self._conn = _BorrowedConnection(conn)
        elif db_path == ":memory:":
            raise ValueError(
                "ActivationEngine: ':memory:' 每次连接都是新空库,请传 conn= 或 "
                "store=(如 engine.store) 共享同一连接")
        else:
            self.db_path = db_path
            self._conn = None
            parent = os.path.dirname(os.path.abspath(db_path))
            if parent:
                os.makedirs(parent, exist_ok=True)
        self.audit_path = audit_path
        self._ensure_tables()
        self._warn_if_content_tables_absent()
        self._adj = None            # 本轮稀疏邻接(node_idx 矩阵,仅兴奋边)
        self._adj_inh = None        # 本轮抑制邻接(仅 opposite 边,存 |decay|)
        self._node_index = None     # node_id → idx
        self._edge_meta = None      # (src_idx, dst_idx, decay) 兴奋边列表
        self._inh_meta = None       # (src_idx, dst_idx, |decay|) 抑制边列表

    # ---------------- 表结构(独立激活表,与内容记忆分离) ----------------
    def _ensure_tables(self):
        with self._transaction() as con:
            con.execute("""CREATE TABLE IF NOT EXISTS activation_nodes (
                id TEXT PRIMARY KEY, workset TEXT NOT NULL, node_id TEXT NOT NULL,
                activation REAL NOT NULL, source TEXT NOT NULL, hop INTEGER NOT NULL, ts REAL NOT NULL)""")
            con.execute("CREATE INDEX IF NOT EXISTS idx_act_ws ON activation_nodes(workset)")

    def _con(self):
        """取连接：借用的共享连接优先（#293），否则按路径新开。"""
        if self._conn is not None:
            return self._conn
        return sqlite3.connect(self.db_path, timeout=30)

    @contextmanager
    def _transaction(self):
        """Write atomically without committing a borrowed caller transaction.

        Initialization participates in the same boundary. If its first schema
        creation is rolled back by the caller, reconstruct the activation engine.
        Commit/RELEASE failures use the same rollback path as statement failures.
        """
        con = self._con()
        nested = con.in_transaction
        savepoint = "activation_%s" % uuid.uuid4().hex
        try:
            if nested:
                con.execute("SAVEPOINT %s" % savepoint)
            else:
                con.execute("BEGIN IMMEDIATE")
            try:
                yield con
                if nested:
                    con.execute("RELEASE SAVEPOINT %s" % savepoint)
                else:
                    con.commit()
            except BaseException as error:
                try:
                    if con.in_transaction:
                        if nested:
                            con.execute("ROLLBACK TO SAVEPOINT %s" % savepoint)
                            con.execute("RELEASE SAVEPOINT %s" % savepoint)
                        else:
                            con.rollback()
                except sqlite3.Error as rollback_error:
                    error.add_note("Activation rollback failed: %s" % rollback_error)
                raise
        finally:
            con.close()

    def _warn_if_content_tables_absent(self):
        """#714：构造期给出一条明确 warning——`nodes` 表缺席时激活面是**空工作态**。

        构造成功不等于「图已就绪」：本引擎的表只有 `activation_nodes`，内容表由
        `core.py` 的 `LayeredStore` 建。缺 `nodes` 时 `activate()` 仍返回
        `status=ok, size=0`（与 `workset_state` 的降级口径一致），故调用方须有
        一条可检索的提示，避免把「空图」误读成「查询无命中」。
        这里只探测表名（`sqlite_master` 在空库上恒存在），不触碰内容表本身。
        """
        con = self._con()
        try:
            have = {r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
        except sqlite3.OperationalError:  # 表名都探不到 ⇒ 不臆断，读图腿各自降级
            return
        finally:
            con.close()
        if "nodes" not in have:
            warnings.warn(
                "ActivationEngine: 本库无内容表 nodes（独立激活库/新目录），"
                "activate() 将按空图返回 status=ok, size=0；如需内容记忆请指向 "
                "LayeredStore 的 db_path（#714）", RuntimeWarning, stacklevel=2)

    # ---------------- 本轮图读取(numpy 全图矩阵化) ----------------
    def _load_graph(self):
        """每轮重读全图，保证已提交的节点/边变更进入索引与传播底物。

        与 edge-list 路径一致，不跨轮复用缺乏失效机制的稀疏矩阵。

        #714：内容表（`nodes` / `edges`）**不存在**是受支持场景——独立激活库 /
        灾备恢复出只含激活表的库 / `LINGSHU_DATA_ROOT` 指向新目录（同文件
        `workset_state` 对同一情形已显式 `except sqlite3.OperationalError` 降级，
        注释：「恢复库无内容表(独立激活库)——仅返回激活值」）。故两条 SELECT 各自把
        `no such table` 视作**空图**（`nodes=[]` / `edges=[]`），让 `activate()` 落到
        既有的空工作态返回路径（`status=ok, size=0`），而不是整轮抛穿。
        两条腿分开 catch：缺 `edges` 而 `nodes` 在场时仍保留节点域。
        """
        con = self._con()
        cur = con.cursor()
        try:
            cur.execute("SELECT id, condition_space FROM nodes")
            node_rows = cur.fetchall()
        except sqlite3.OperationalError:  # #714:无内容表 ⇒ 空图
            node_rows = []
        node_ids = [r[0] for r in node_rows]
        try:
            cur.execute("SELECT source_id, target_id, relation_type, condition_space FROM edges")
            edges = cur.fetchall()
        except sqlite3.OperationalError:  # #714:无内容表 ⇒ 空图
            edges = []
        con.close()
        idx = {nid: i for i, nid in enumerate(node_ids)}
        n = len(node_ids)
        cur_cs = {}
        for nid, cs_json in node_rows:
            try:
                cur_cs[nid] = json.loads(cs_json) if cs_json else {}
            except (json.JSONDecodeError, TypeError):
                cur_cs[nid] = {}
        rows, cols, decays = [], [], []
        inh_rows, inh_cols, inh_mags = [], [], []
        for src, dst, rtype, cs_json in edges:
            si, di = idx.get(src), idx.get(dst)
            if si is None or di is None:
                continue
            try:
                cs_e = json.loads(cs_json) if cs_json else {}
            except (json.JSONDecodeError, TypeError):
                cs_e = {}
            d = EDGE_BASE_DECAY.get(rtype, DEFAULT_DECAY)
            # #53:节点对级条件衰减修正——decay = base(边类型) × cond_match(两端条件空间)
            #   （模块 docstring 核心公式 :26）。重合率由两端节点静态决定、与跳数无关，
            #   故装载期算一次即等价于逐跳算（稀疏与 edge-list 两后端同口径）。
            # #709：`cond_match` 自身已对非 dict 入参 / 非有限数值 `time_window` 回落中性；
            #   此处再兜一层——未枚举到的坏条件空间只让**该条边**回落中性衰减，
            #   不得让一条形状异常记录把整轮 `_load_graph`（进而 `activate()`）抛穿
            #   （`_load_graph` 每轮全图重读，坏节点是否参与本次查询都无关）。
            try:
                d *= cond_match(cur_cs.get(src), cur_cs.get(dst))
            except (TypeError, ValueError):
                d = d * COND_MATCH_NEUTRAL
            if d < 0:  # 负记忆同价(第五篇):相反边走独立抑制通路,不进兴奋邻接(issue #229)
                inh_rows.append(si); inh_cols.append(di); inh_mags.append(-d)
            else:
                rows.append(si); cols.append(di); decays.append(d)
        if csr_matrix is not None and rows:
            adj = csr_matrix((np.array(decays, dtype=np.float32), (rows, cols)),
                             shape=(n, n))
        else:
            adj = None  # 降级:edge-list 传播
        if csr_matrix is not None and inh_rows:
            adj_inh = csr_matrix((np.array(inh_mags, dtype=np.float32), (inh_rows, inh_cols)),
                                 shape=(n, n))
        else:
            adj_inh = None  # 降级:edge-list 抑制
        self._adj = adj
        self._adj_inh = adj_inh
        self._node_index = idx
        self._edge_meta = (rows, cols, decays)
        self._inh_meta = (inh_rows, inh_cols, inh_mags)
        self._node_ids = node_ids
        # 节点条件空间(节点级 cond_match 用;装载期已消费,留作审计/调试)
        self._node_cond = cur_cs

    @staticmethod
    def _max_plus(adj, act: np.ndarray) -> np.ndarray:
        """max-plus 稀疏传播:out[dst] = max over src (act[src]·w[src,dst])。

        scipy 的 `@` 是 Σ 半环(多源/多跳累加),自条件回注下会跨轮指数发散
        (issue #280);本函数改用 max 半环,与 edge-list 降级路径口径一致。
        w 非负(抑制边存 |decay| 且走独立通路),故只需最大值,不需处理负权。
        向量化:把 CSR 的每个非零元展平成 (src, dst, w),一次 `np.maximum.at`
        按 dst 归约——重复 dst 取最大值,与逐行写法同口径但不随行数循环。
        """
        n = act.shape[0]
        out = np.zeros(n, dtype=np.float32)
        indptr, indices, data = adj.indptr, adj.indices, adj.data
        if data.size == 0:
            return out
        src = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
        np.maximum.at(out, indices, act[src] * data)
        return out

    # ---------------- 种子生成(同步询问→投票) ----------------
    def _seed_by_text(self, query: str, top_k: int = 12) -> List[Tuple[str, float]]:
        """种子:中文二元组倒排——同步询问全图(所有含查询词元的节点),计数投票。

        #161:空查询 / 纯空白查询**无可投票词元**——旧实现下 `toks` 含 `""`(或 `" "`),
        而 `"" in hay` 恒真,于是按表序前 top_k 个节点全以 1.0 种入、覆盖该 workset。
        现直接返回空种子(activate 由此得到 seeds=0 的空工作态)。
        判据来源:本件 #161(分诊表 `triage_lingshu_ALL.md` 行 161 判定成立)。
        """
        if not (query or "").strip():
            return []
        toks = {query[i:i + 2] for i in range(len(query) - 1)} | {query}
        con = self._con()
        cur = con.cursor()
        try:
            cur.execute("SELECT id, content, tags FROM nodes")
            rows = cur.fetchall()
        except sqlite3.OperationalError:  # #714:无内容表(独立激活库)⇒ 无可投票词元
            rows = []
        con.close()
        scores = []
        for nid, content, tags_json in rows:
            hay = content or ""
            try:
                hay += " " + " ".join(json.loads(tags_json)) if tags_json else ""
            except (json.JSONDecodeError, TypeError):
                pass
            s = sum(1 for t in toks if t in hay) / max(1, len(toks))
            if s > 0:
                scores.append((nid, s))
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

        #7 轻核降级：numpy 缺失时**不抛异常**，返回
        `{"status": "activation_not_ready", ...}`（与 core.py 各 `*_not_ready` 同口径），
        使「core 纯标准库可导入」的契约在无 extras 环境下成立。
        判据来源：README「轻核：core 纯标准库，零依赖可跑」＋ pyproject.toml:12
        `dependencies = []`；分诊表 `triage_lingshu_ALL.md` 行 7 判定成立。

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
        if np is None:  # #7:轻核降级——模块可导入,激活面显式 not_ready
            return {"status": "activation_not_ready",
                    "error": "ActivationEngine 需要 numpy（pip install 'lingshu[world]'）",
                    "workset": workset, "size": 0, "top": [], "seeds": 0,
                    "hops": hops, "carried": 0, "latency_ms": 0.0}
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
        # #314:重构路径审计的落表元数据——种子/自条件回注 = hop 0,每跳新激活的节点
        #   记该跳号(旧实现 hop 列恒 0、种子 source 被兜底记成 propagate)。
        first_hop: Dict[str, int] = {nid: 0 for nid in seed_ids}
        for nid in carried_ids:
            first_hop[nid] = 0
        # 逐跳扩散(条件衰减)——max-plus 口径(issue #280):
        #   prop[dst] = max over src (act[src]·decay),不跨跳/跨源累加,
        #   故 act 有界(≤ max(种子,上一步先验)),不随跳数/轮数指数发散。
        for hop in range(1, hops + 1):
            prop = np.zeros(n, dtype=np.float32)
            if self._adj is not None:
                # 稀疏图上的 max-plus 传播(scipy matvec 是 Σ 半环,会累加→发散,故不用 @)
                prop = self._max_plus(self._adj, act)
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
            # 抑制通路(负记忆同价,第五篇;issue #229):opposite 边扣减目标激活,
            #   supp[dst] = max over src (act[src]·|decay|),act[dst] ← max(act[dst]−supp,0)。
            supp = np.zeros(n, dtype=np.float32)
            if self._adj_inh is not None:
                supp = self._max_plus(self._adj_inh, act)
            else:  # edge-list 降级
                inh_rows, inh_cols, inh_mags = self._inh_meta
                for si, di, m in zip(inh_rows, inh_cols, inh_mags):
                    if act[si] > 0:
                        supp[di] = max(supp[di], act[si] * m)
            if supp.any():
                act = np.maximum(act - supp, 0.0)
            for i in np.nonzero(act >= act_floor)[0]:
                nid = self._node_ids[i]
                if nid not in first_hop:  # #314:首次进入工作态的跳号
                    first_hop[nid] = hop
            newly = [self._node_ids[i] for i in np.nonzero(act >= act_floor)[0]]
            path.append({"step": hop, "phase": "propagate",
                         "activated_count": len(newly),
                         "max_act": round(float(act.max()), 4) if n else 0.0})
        # 阈值截断→工作态
        keep = np.nonzero(act >= act_floor)[0]
        members = [(self._node_ids[i], round(float(act[i]), 4)) for i in keep]
        members.sort(key=lambda x: -x[1])
        sources = {nid: "seed" for nid in seed_ids}
        for nid in carried_ids:  # 自条件回注覆盖种子标签(既有行为不变)
            sources[nid] = "self_condition"
        self._write_workset(workset, members, query, sources, first_hop)
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
                       sources: Optional[Dict[str, str]] = None,
                       hops: Optional[Dict[str, int]] = None):
        """落表:members=(node_id, 激活强度);sources 标注来源(seed/propagate/self_condition)。

        #314：`hops` 给每个 node_id 的**首次激活跳号**（种子/自条件 = 0）；缺省仍写 0
        （签名向后兼容——不传即旧行为）。
        """
        with self._transaction() as con:
            con.execute("DELETE FROM activation_nodes WHERE workset=?", (workset,))
            ts = time.time()
            src_map = sources or {}
            hop_map = hops or {}
            for nid, a in members:
                con.execute(
                    "INSERT OR REPLACE INTO activation_nodes VALUES (?,?,?,?,?,?,?)",
                    ("act_%s_%d" % (uuid.uuid4().hex[:8], int(ts * 1000)),
                     workset, nid, a, src_map.get(nid, "propagate"),
                     hop_map.get(nid, 0), ts))

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
        """恢复快照(激活状态重建——进程重启后工作态不失)。

        #771 ①（原子性）：旧实现**先 `DELETE` 目标 workset、再迭代 members**，全程无
        事务保护、无 `except`/`rollback` ⇒ members 畸形（`None` / 非列表 / member 缺
        `node_id` / `node_id=None` / activation 为 dict…）时异常逃逸，而**目标
        workset 已被清空且无回滚**。借用连接形态（`store=` / `conn=`，即常驻部署
        形态）下尤其重：`DELETE` 挂在调用方连接上，任一次后续提交即把它**固化**。
        修法（最小 · 只补原子性，**不改错误语义**）：
          ① 先把 `members` 物化成 `list` —— 非可迭代形态（`None` 等）在**任何写之前**
             即抛（异常类型与旧实现逐字相同，仍是 `TypeError`）；
          ② `DELETE` 与全部 `INSERT` 收进**同一事务**（已在外层事务中则用 SAVEPOINT，
             不越权提交调用方事务），异常一律回滚后原样抛出 ⇒ 畸形输入下目标 workset
             **逐行不变**（零变更）。
        边界（如实标注）：①畸形 members 仍**抛异常**，不改判「静默返回错误结构」
        （调用方契约原本就是抛）；②「目标 workset 不存在」不报错（旧语义：`DELETE`
        影响 0 行、照常写入），本件不动；③本件只加「可迭代性」这一层前置校验，
        member 内部的 schema 仍由既有异常面承担（`KeyError` / `ProgrammingError` /
        `IntegrityError`），错误类型不重定义。判据来源：公开 issue #771 ①
        （「校验通过后再在同一事务里 DELETE + INSERT；异常路径回滚」）。
        """
        ws = workset or snapshot["workset"]
        # ① 写前物化（非可迭代 ⇒ 在任何 DELETE 之前抛出，目标零变更）。
        members = list(snapshot["members"])
        with self._transaction() as con:
            ts = time.time()
            con.execute("DELETE FROM activation_nodes WHERE workset=?", (ws,))
            for m in members:
                con.execute("INSERT OR REPLACE INTO activation_nodes VALUES (?,?,?,?,?,?,?)",
                            ("act_%s_%d" % (uuid.uuid4().hex[:8], int(ts * 1000)),
                             ws, m["node_id"], m["activation"], "restored", m.get("hop", 0), ts))
        return {"status": "ok", "workset": ws, "restored": len(members)}


if __name__ == "__main__":
    eng = ActivationEngine()
    r1 = eng.activate("灵枢 认知图 上下文", workset="demo")
    r2 = eng.activate("激活 工作态 演化", workset="demo",
                      self_condition=SELF_CONDITION_WEIGHT)
    print(json.dumps({"first": r1, "second": r2}, ensure_ascii=False, indent=1)[:600])
    st = eng.workset_state("demo", 3)
    print(json.dumps(st, ensure_ascii=False)[:400])
