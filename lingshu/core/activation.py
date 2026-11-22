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
import hashlib
import json
import math
import os
import sqlite3
import struct
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
        相对 `db_path` 仍以构造时 cwd 选择原位置，实例随后固定该文件；进程 chdir
        不得让同一实例的后续重连改写另一同名库。不改默认数据根的选择。
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
            self.db_path = os.path.abspath(db_path)
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
        self._inh_meta = None       # (src_idx, dst_idx, |decay|) 抑制边列表（单向）
        self._adj_inh_unv = None    # 低置信(<0.5)抑制邻接（双向补充，issue #521）
        self._inh_unv_meta = None   # (src_idx, dst_idx, |decay|×confidence) 低置信边列表
        self._reset_graph_cache()

    # ---------------- 表结构(独立激活表,与内容记忆分离) ----------------
    def _ensure_tables(self):
        with self._transaction() as con:
            con.execute("""CREATE TABLE IF NOT EXISTS activation_nodes (
                id TEXT PRIMARY KEY, workset TEXT NOT NULL, node_id TEXT NOT NULL,
                activation REAL NOT NULL, source TEXT NOT NULL, hop INTEGER NOT NULL, ts REAL NOT NULL,
                query TEXT)""")
            # #505：`query` 列是工作集的可回溯输入（与 sources/hops 同签名、同落库面）。
            # 既有库（旧 7 列结构）在这里补列；已存在则跳过（幂等）。
            try:
                con.execute("ALTER TABLE activation_nodes ADD COLUMN query TEXT")
            except sqlite3.OperationalError:
                pass
            con.execute("CREATE INDEX IF NOT EXISTS idx_act_ws ON activation_nodes(workset)")

    def _con(self):
        """取连接：借用的共享连接优先（#293），否则按路径新开。"""
        if self._conn is not None:
            return self._conn
        return sqlite3.connect(self.db_path, timeout=30)

    @contextmanager
    def _transaction(self, _on_commit=None):
        """Write atomically without committing a borrowed caller transaction.

        Initialization participates in the same boundary. If its first schema
        creation is rolled back by the caller, reconstruct the activation engine.
        Commit/RELEASE failures use the same rollback path as statement failures.
        """
        con = self._con()
        nested = con.in_transaction
        savepoint = "activation_%s" % uuid.uuid4().hex
        committed = False
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
                committed = True
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
            try:
                if committed and _on_commit is not None:
                    _on_commit(con, nested)
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
        node_rows, edges, cache_sig, node_hays = self._fetch_graph_rows()
        node_ids = [r[0] for r in node_rows]
        idx = {nid: i for i, nid in enumerate(node_ids)}
        n = len(node_ids)
        cur_cs = {}
        raw_cs_list = []
        _cs_json_cache = {}
        for nid, cs_json in node_rows:
            raw_cs_list.append(cs_json)
            if not cs_json:
                cur_cs[nid] = {}
            elif cs_json in _cs_json_cache:
                cur_cs[nid] = _cs_json_cache[cs_json]
            else:
                try:
                    parsed = json.loads(cs_json)
                except (json.JSONDecodeError, TypeError):
                    parsed = {}
                _cs_json_cache[cs_json] = parsed
                cur_cs[nid] = parsed
        rows, cols, decays = [], [], []
        inh_rows, inh_cols, inh_mags = [], [], []
        unv_rows, unv_cols, unv_mags = [], [], []
        _use_cm_cache, _cm_cache = (cond_match is _ORIG_COND_MATCH), {}
        for src, dst, rtype, edge_conf in edges:
            si, di = idx.get(src), idx.get(dst)
            if si is None or di is None:
                continue
            base = EDGE_BASE_DECAY.get(rtype, DEFAULT_DECAY)
            if _use_cm_cache:
                _pair = (raw_cs_list[si], raw_cs_list[di])
                _cm = _cm_cache.get(_pair)
                if _cm is None:
                    try:
                        _cm = _ORIG_COND_MATCH(cur_cs.get(src), cur_cs.get(dst))
                    except (TypeError, ValueError):
                        _cm = COND_MATCH_NEUTRAL
                    _cm_cache[_pair] = _cm
                d = base * _cm
            else:
                d = base
                try:
                    d *= cond_match(cur_cs.get(src), cur_cs.get(dst))
                except (TypeError, ValueError):
                    d = base * COND_MATCH_NEUTRAL
            if d < 0:  # 负记忆同价(第五篇):相反边走独立抑制通路,不进兴奋邻接(issue #229)
                # issue #521：**低置信**（< 默认 0.5，含 register_conflict 的 0.3）
                # 的 opposite 边按置信缩放、并进双向补充面（contested）；默认 0.5
                # 及以上的边保持既有「满额单向」语义（#229 的既有契约，勿回归）。
                try:
                    _c = float(edge_conf) if edge_conf is not None else 0.5
                except (TypeError, ValueError):
                    _c = 0.5
                if _c != _c:  # NaN
                    _c = 0.5
                _c = min(1.0, max(0.0, _c))
                if _c < 0.5:
                    inh_rows.append(si); inh_cols.append(di); inh_mags.append(-d * _c)
                    unv_rows.append(si); unv_cols.append(di); unv_mags.append(-d * _c)
                else:
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
        if csr_matrix is not None and unv_rows:
            _unv_data = np.array(unv_mags, dtype=np.float32)
            if _FastTransposeCSR is not None and csr_matrix is _ORIG_CSR_MATRIX:
                adj_inh_unv = _FastTransposeCSR((_unv_data, (unv_rows, unv_cols)), shape=(n, n))
                adj_inh_unv._t_csr = csr_matrix((_unv_data, (unv_cols, unv_rows)), shape=(n, n))
            else:
                adj_inh_unv = csr_matrix((_unv_data, (unv_rows, unv_cols)), shape=(n, n))
        else:
            adj_inh_unv = None
        self._adj = adj
        self._adj_inh = adj_inh
        self._adj_inh_unv = adj_inh_unv
        self._node_index = idx
        self._edge_meta = (rows, cols, decays)
        self._inh_meta = (inh_rows, inh_cols, inh_mags)
        self._inh_unv_meta = (unv_rows, unv_cols, unv_mags)
        self._node_ids = node_ids
        # 节点条件空间(节点级 cond_match 用;装载期已消费,留作审计/调试)
        self._node_cond = cur_cs
        self._node_hays = node_hays
        self._build_frontier_index(n)
        self._graph_cache_sig = cache_sig
        self._graph_cache_env = self._cache_env_key() if cache_sig is not None else None
        self._graph_load_count += 1

    def _reset_graph_cache(self):
        """初始化图缓存与稀疏前沿索引状态（零跨实例共享）。"""
        self._graph_cache_sig = None
        self._graph_cache_env = None
        self._graph_load_count = 0
        self._node_hays = None
        self._exc_indptr = None
        self._exc_indices = None
        self._exc_data = None
        self._exc_src = None
        self._exc_wmax = 0.0
        self._exc_canonical = False
        self._inh_csr = None
        self._inh_unv_t_csr = None
        self._inh_src_arr = None
        self._inh_dst_mask = None
        self._rev_exc_csr = None
        self._inh_dist = None
        self._inh_dist_max_depth = -1
        self._inh_bfs_frontier = None
        self._inh_exc_cache = {}
        self._last_frontier_stats = {"edges_visited": 0, "edges_pruned": 0, "cache_hit": False}

    @staticmethod
    def _cache_env_key():
        return (
            csr_matrix is not None,
            cond_match,
            tuple(sorted(EDGE_BASE_DECAY.items())),
            DEFAULT_DECAY,
            COND_MATCH_NEUTRAL,
        )

    def _conn_sig(self, con, allow_in_tx: bool = False) -> Optional[Tuple]:
        """Borrowed-connection O(1) version signature.

        Uses PRAGMA data_version (cross-connection commits in WAL/DELETE),
        PRAGMA schema_version + temp.schema_version (DDL), and con.total_changes
        (same-connection DML). Refuses caching inside uncommitted caller
        transactions so a later ROLLBACK can never expose stale state.
        """
        if not allow_in_tx and con.in_transaction:
            return None
        try:
            raw_con = con._conn if isinstance(con, _BorrowedConnection) else con
            dv = int(con.execute("PRAGMA data_version").fetchone()[0])
            sv = int(con.execute("PRAGMA schema_version").fetchone()[0])
            tsv = int(con.execute("PRAGMA temp.schema_version").fetchone()[0])
            return ("conn", id(raw_con), dv, sv, tsv, int(con.total_changes))
        except sqlite3.Error:
            return None

    def _file_fast_sig(self, con) -> Optional[Tuple]:
        """File-backed O(1) sound signature across separate connections.

        - Rollback journal modes: main DB file header bytes 24..27 increment on
          every commit (verified for DELETE/TRUNCATE/PERSIST/MEMORY/OFF).
        - WAL mode with live WAL (mxFrame > 0, isInit == 1, duplicated 48-byte
          WalIndexHdr copies equal): shm header captures iChange, mxFrame,
          cumulative frame checksum, and 64-bit random WAL salt.
        - Clean/checkpointed WAL (no -wal/-shm or mxFrame == 0): returns None
          so callers fall back to exact table content digest (WAL checkpoints
          do not bump the main DB header change counter).
        """
        try:
            jm = str(con.execute("PRAGMA journal_mode").fetchone()[0]).lower()
            sv = int(con.execute("PRAGMA schema_version").fetchone()[0])
            st = os.stat(self.db_path)
            with open(self.db_path, "rb", buffering=0) as f:
                hdr = f.read(100)
            if len(hdr) < 100:
                return None
            main_part = (st.st_dev, st.st_ino, st.st_size,
                         hdr[24:32], hdr[40:44], hdr[92:96])
            if jm == "wal":
                wal_path = self.db_path + "-wal"
                shm_path = self.db_path + "-shm"
                if not os.path.exists(wal_path) or not os.path.exists(shm_path):
                    return None
                # Unbuffered 96-byte read avoids overlapping SQLite's Win32
                # mandatory byte-range lock on -shm at offsets 120..128.
                with open(shm_path, "rb", buffering=0) as f:
                    shm = f.read(96)
                if len(shm) < 96 or shm[:48] != shm[48:96]:
                    return None
                is_init = shm[12]
                mx_frame = struct.unpack("=I", shm[16:20])[0]
                if is_init != 1 or mx_frame == 0:
                    return None
                return ("file-wal", sv, main_part, shm[:48])
            if jm in ("delete", "truncate", "persist", "memory", "off"):
                return ("file-rb", jm, sv, main_part)
            return None
        except (OSError, sqlite3.Error, struct.error, TypeError, ValueError):
            return None

    @staticmethod
    def _table_content_digest(con) -> Optional[Tuple]:
        """Exact read-only BLAKE2b digest of nodes & edges (clean-WAL fallback).

        Fails closed (returns None) on any sqlite3.Error so transient locks or
        schema errors can never produce two equal sentinel digests. Absent
        tables (#714) return empty PRAGMA table_info cleanly (n_dig/e_dig=None).
        """
        try:
            sv = int(con.execute("PRAGMA schema_version").fetchone()[0])
            cur = con.cursor()
            ncols = {r[1] for r in cur.execute("PRAGMA table_info(nodes)")}
            if ncols:
                parts = ["quote(id)",
                         "quote(condition_space)" if "condition_space" in ncols else "'NULL'"]
                if "content" in ncols:
                    parts.append("quote(content)")
                if "tags" in ncols:
                    parts.append("quote(tags)")
                expr = "||','||".join(parts)
                cnt, blob = cur.execute(
                    "SELECT COUNT(*), CAST(GROUP_CONCAT(" + expr
                    + ", char(10)) AS BLOB) FROM nodes").fetchone()
                n_dig = (int(cnt), hashlib.blake2b(blob or b"", digest_size=16).digest())
            else:
                n_dig = None
            ecols = {r[1] for r in cur.execute("PRAGMA table_info(edges)")}
            if ecols:
                conf_q = "quote(confidence)" if "confidence" in ecols else "'0.3'"
                e_expr = ("quote(source_id)||','||quote(target_id)||','||"
                          "quote(relation_type)||','||" + conf_q)
                cnt, blob = cur.execute(
                    "SELECT COUNT(*), CAST(GROUP_CONCAT(" + e_expr
                    + ", char(10)) AS BLOB) FROM edges").fetchone()
                e_dig = (int(cnt), hashlib.blake2b(blob or b"", digest_size=16).digest())
            else:
                e_dig = None
            return ("file-digest", sv, n_dig, e_dig)
        except sqlite3.Error:
            return None

    def _is_graph_cache_valid(self) -> bool:
        """Check whether the cached graph matches the current DB state soundly."""
        sig = self._graph_cache_sig
        if sig is None or self._graph_cache_env != self._cache_env_key():
            return False
        con = self._con()
        try:
            if self._conn is not None:
                if sig[0] != "conn":
                    return False
                return self._conn_sig(con) == sig
            if sig[0] in ("file-rb", "file-wal"):
                # Start a brief read snapshot so SQLite syncs WAL-index shm
                con.execute("BEGIN")
                try:
                    con.execute("PRAGMA schema_version").fetchone()
                    return self._file_fast_sig(con) == sig
                finally:
                    if con.in_transaction:
                        con.rollback()
            if sig[0] == "file-digest":
                con.execute("BEGIN")
                try:
                    return self._table_content_digest(con) == sig
                finally:
                    if con.in_transaction:
                        con.rollback()
            return False
        except sqlite3.Error:
            return False
        finally:
            con.close()

    def _ensure_graph(self) -> bool:
        """Load graph only on cache miss; returns True on cache hit."""
        if self._is_graph_cache_valid():
            self._last_frontier_stats["cache_hit"] = True
            return True
        self._load_graph()
        self._last_frontier_stats["cache_hit"] = False
        return False

    @staticmethod
    def _build_hay(content, tags_json) -> str:
        hay = content or ""
        if tags_json:
            if tags_json == "[]":
                hay += " "
            else:
                try:
                    hay += " " + " ".join(json.loads(tags_json))
                except (json.JSONDecodeError, TypeError):
                    pass
        return hay

    def _fetch_graph_rows(self):
        """Read nodes (plus text/tags when present) and edges with a sound snapshot sig."""
        con = self._con()
        cache_sig = None
        started_tx = False
        try:
            if self._conn is not None:
                pre_sig = self._conn_sig(con)
            else:
                pre_fast = self._file_fast_sig(con)
                con.execute("BEGIN")
                started_tx = True
            cur = con.cursor()
            try:
                _node_cols = {r[1] for r in cur.execute("PRAGMA table_info(nodes)")}
            except sqlite3.OperationalError:
                _node_cols = set()
            has_text = ("content" in _node_cols and "tags" in _node_cols)
            node_hays = None
            try:
                if has_text:
                    cur.execute("SELECT id, condition_space, content, tags FROM nodes")
                    raw_nodes = cur.fetchall()
                    node_rows = [(r[0], r[1]) for r in raw_nodes]
                    _bh = self._build_hay
                    node_hays = [_bh(r[2], r[3]) for r in raw_nodes]
                else:
                    cur.execute("SELECT id, condition_space FROM nodes")
                    node_rows = cur.fetchall()
            except sqlite3.OperationalError:  # #714:无内容表 ⇒ 空图
                node_rows = []
                node_hays = []
            # issue #521：抑制幅度要按边的 confidence 缩放（未复核 0.3 的 opposite
            # 边不得满额否决）。旧 schema 可能没有 confidence 列——用 PRAGMA 探测，
            # 缺列时按 0.3 兜底，避免把「列不存在」误判成「无内容表」而清空整图。
            try:
                _edge_cols = {r[1] for r in cur.execute("PRAGMA table_info(edges)")}
            except sqlite3.OperationalError:
                _edge_cols = set()
            _conf_expr = "confidence" if "confidence" in _edge_cols else "0.3 AS confidence"
            try:
                cur.execute("SELECT source_id, target_id, relation_type, "
                            + _conf_expr + " FROM edges")
                edges = cur.fetchall()
            except sqlite3.OperationalError:  # #714:无内容表 ⇒ 空图
                edges = []
            if self._conn is not None:
                if pre_sig is not None and self._conn_sig(con) == pre_sig:
                    cache_sig = pre_sig
            else:
                if pre_fast is None:
                    # Clean/checkpointed WAL: digest inside the exact same read snapshot
                    cache_sig = self._table_content_digest(con)
                if started_tx and con.in_transaction:
                    con.rollback()
                    started_tx = False
                if pre_fast is not None:
                    # Seqlock verification: cache only if no commit occurred during load
                    if self._file_fast_sig(con) == pre_fast:
                        cache_sig = pre_fast
            return node_rows, edges, cache_sig, node_hays
        finally:
            if started_tx and con.in_transaction:
                try:
                    con.rollback()
                except sqlite3.Error:
                    pass
            con.close()

    @staticmethod
    def _build_numpy_csr(n: int, rows, cols, weights, sort_desc: bool = False):
        """Build pure-NumPy CSR arrays (works identically with or without SciPy)."""
        if len(rows) == 0:
            return (np.zeros(n + 1, dtype=np.intp),
                    np.empty(0, dtype=np.int32),
                    np.empty(0, dtype=np.float32),
                    np.empty(0, dtype=np.int32))
        s = np.asarray(rows, dtype=np.int32)
        c = np.asarray(cols, dtype=np.int32)
        w = np.asarray(weights, dtype=np.float32)
        mask = w > 0
        if not mask.all():
            s, c, w = s[mask], c[mask], w[mask]
        if s.size == 0:
            return (np.zeros(n + 1, dtype=np.intp),
                    np.empty(0, dtype=np.int32),
                    np.empty(0, dtype=np.float32),
                    np.empty(0, dtype=np.int32))
        if sort_desc:
            order = np.lexsort((-w, s))
        else:
            order = np.argsort(s, kind="stable")
        s_sorted = s[order]
        c_sorted = c[order]
        w_sorted = w[order]
        indptr = np.zeros(n + 1, dtype=np.intp)
        np.cumsum(np.bincount(s_sorted, minlength=n), out=indptr[1:])
        return indptr, c_sorted, w_sorted, s_sorted

    def _build_frontier_index(self, n: int):
        """Build descending-weight excitatory CSR & inhibitory endpoint structures."""
        if np is None:
            return
        # Reset lazy backward-BFS caches for the new graph
        self._rev_exc_csr = None
        self._inh_dist = None
        self._inh_dist_max_depth = -1
        self._inh_bfs_frontier = None
        self._inh_exc_cache = {}
        # 1. Excitatory outgoing CSR sorted by weight descending
        if self._adj is not None and self._adj.data.size:
            raw_src = np.repeat(np.arange(n, dtype=np.int32), np.diff(self._adj.indptr))
            indptr, indices, data, src = self._build_numpy_csr(
                n, raw_src, self._adj.indices, self._adj.data, sort_desc=True)
            self._exc_canonical = bool(getattr(self._adj, "has_canonical_format", False))
        else:
            rows, cols, decays = self._edge_meta
            indptr, indices, data, src = self._build_numpy_csr(
                n, rows, cols, decays, sort_desc=True)
            self._exc_canonical = False
        self._exc_indptr = indptr
        self._exc_indices = indices
        self._exc_data = data
        self._exc_src = src
        self._exc_wmax = float(data.max()) if data.size else 0.0
        # 2. Edge-list fallback NumPy CSR for inhibitory edges (active-support speed without SciPy)
        inh_rows, inh_cols, inh_mags = self._inh_meta
        unv_rows, unv_cols, unv_mags = self._inh_unv_meta
        if self._adj_inh is None and inh_rows:
            i_ip, i_idx, i_dat, _ = self._build_numpy_csr(n, inh_rows, inh_cols, inh_mags)
            self._inh_csr = (i_ip, i_idx, i_dat) if i_dat.size else None
        else:
            self._inh_csr = None
        if self._adj_inh is None and unv_rows:
            u_ip, u_idx, u_dat, _ = self._build_numpy_csr(n, unv_cols, unv_rows, unv_mags)
            self._inh_unv_t_csr = (u_ip, u_idx, u_dat) if u_dat.size else None
        else:
            self._inh_unv_t_csr = None
        # 3. Inhibitory source set (I_src) and target mask (I_dst) for exact pruning
        src_parts = []
        dst_parts = []
        if self._adj_inh is not None and self._adj_inh.data.size:
            pos = self._adj_inh.data > 0
            if pos.any():
                raw_is = np.repeat(np.arange(n, dtype=np.int32), np.diff(self._adj_inh.indptr))
                src_parts.append(raw_is[pos])
                dst_parts.append(self._adj_inh.indices[pos].astype(np.int32, copy=False))
        elif inh_rows:
            im = np.asarray(inh_mags, dtype=np.float32)
            pos = im > 0
            if pos.any():
                src_parts.append(np.asarray(inh_rows, dtype=np.int32)[pos])
                dst_parts.append(np.asarray(inh_cols, dtype=np.int32)[pos])
        if self._adj_inh_unv is not None and self._adj_inh_unv.data.size:
            pos = self._adj_inh_unv.data > 0
            if pos.any():
                # Reverse edge di -> si: source is col (di), target is row (si)
                raw_us = np.repeat(np.arange(n, dtype=np.int32), np.diff(self._adj_inh_unv.indptr))
                src_parts.append(self._adj_inh_unv.indices[pos].astype(np.int32, copy=False))
                dst_parts.append(raw_us[pos])
        elif unv_rows:
            um = np.asarray(unv_mags, dtype=np.float32)
            pos = um > 0
            if pos.any():
                src_parts.append(np.asarray(unv_cols, dtype=np.int32)[pos])
                dst_parts.append(np.asarray(unv_rows, dtype=np.int32)[pos])
        if src_parts:
            self._inh_src_arr = np.unique(np.concatenate(src_parts))
            dst_mask = np.zeros(n, dtype=bool)
            dst_mask[np.concatenate(dst_parts)] = True
            self._inh_dst_mask = dst_mask
        else:
            self._inh_src_arr = None
            self._inh_dst_mask = None

    def _ensure_inh_dist(self, max_rem: int):
        """Lazily compute backward shortest excitatory hop distance to I_src up to max_rem."""
        if self._inh_src_arr is None or max_rem < 0:
            return None
        n = len(self._node_ids)
        if self._inh_dist is None:
            dist = np.full(n, -1, dtype=np.int32)
            dist[self._inh_src_arr] = 0
            self._inh_dist = dist
            self._inh_dist_max_depth = 0
            self._inh_bfs_frontier = self._inh_src_arr
        dist = self._inh_dist
        if self._inh_dist_max_depth >= max_rem or self._inh_bfs_frontier.size == 0:
            return dist
        if self._rev_exc_csr is None:
            # Build reverse excitatory adjacency once lazily (dst -> src)
            if self._exc_indices.size:
                order = np.argsort(self._exc_indices, kind="stable")
                rev_dst = self._exc_indices[order]
                rev_src = self._exc_src[order]
                rev_indptr = np.zeros(n + 1, dtype=np.intp)
                np.cumsum(np.bincount(rev_dst, minlength=n), out=rev_indptr[1:])
                self._rev_exc_csr = (rev_indptr, rev_src)
            else:
                self._rev_exc_csr = (np.zeros(n + 1, dtype=np.intp), np.empty(0, dtype=np.int32))
        rev_indptr, rev_src = self._rev_exc_csr
        while self._inh_dist_max_depth < max_rem and self._inh_bfs_frontier.size > 0:
            next_d = self._inh_dist_max_depth + 1
            nxt = []
            for v in self._inh_bfs_frontier.tolist():
                s = int(rev_indptr[v])
                e = int(rev_indptr[v + 1])
                if s < e:
                    for u in rev_src[s:e].tolist():
                        if dist[u] == -1:
                            dist[u] = next_d
                            nxt.append(u)
            self._inh_bfs_frontier = np.array(nxt, dtype=np.int32)
            self._inh_dist_max_depth = next_d
        return dist

    def _get_inh_exc_csr(self, rem: int):
        """Return cached sub-CSR of excitatory edges landing in B_inh(rem), None, or 'ALL'."""
        if self._inh_src_arr is None or self._exc_data.size == 0:
            return None
        cached = self._inh_exc_cache.get(rem, False)
        if cached is not False:
            return cached
        dist = self._ensure_inh_dist(rem)
        d_dst = dist[self._exc_indices]
        mask = (d_dst >= 0) & (d_dst <= rem)
        hit_cnt = int(mask.sum())
        if hit_cnt == 0:
            res = None
        elif hit_cnt * 10 >= mask.size * 9:
            res = "ALL"
        else:
            n = len(self._node_ids)
            s_b = self._exc_src[mask]
            c_b = self._exc_indices[mask]
            w_b = self._exc_data[mask]
            b_indptr = np.zeros(n + 1, dtype=np.intp)
            np.cumsum(np.bincount(s_b, minlength=n), out=b_indptr[1:])
            res = (b_indptr, c_b, w_b)
        self._inh_exc_cache[rem] = res
        return res

    @staticmethod
    def _max_plus_csr_arrays(indptr, indices, data, act: np.ndarray,
                             active: np.ndarray, out: np.ndarray,
                             canonical: bool = False) -> np.ndarray:
        """Active-support sparse frontier max-plus over CSR arrays: O(|S_h| + |E(S_h)|)."""
        k = active.size
        if k == 0 or data.size == 0:
            return out
        if k <= 64:
            if canonical:
                for u in active.tolist():
                    s = int(indptr[u])
                    e = int(indptr[u + 1])
                    if s < e:
                        dst = indices[s:e]
                        out[dst] = np.maximum(out[dst], act[u] * data[s:e])
            else:
                for u in active.tolist():
                    s = int(indptr[u])
                    e = int(indptr[u + 1])
                    if s < e:
                        np.maximum.at(out, indices[s:e], act[u] * data[s:e])
            return out
        starts = indptr[active]
        counts = indptr[active + 1] - starts
        nz = counts > 0
        if not nz.any():
            return out
        act_u = active[nz]
        s_u = starts[nz].astype(np.intp, copy=False)
        c_u = counts[nz].astype(np.intp, copy=False)
        total = int(c_u.sum())
        offsets = np.cumsum(c_u)
        base = np.repeat(s_u - (offsets - c_u), c_u)
        edge_idx = np.arange(total, dtype=np.intp) + base
        src_act = np.repeat(act[act_u], c_u)
        np.maximum.at(out, indices[edge_idx], src_act * data[edge_idx])
        return out

    @staticmethod
    def _max_plus(adj, act: np.ndarray,
                  active: Optional[np.ndarray] = None) -> np.ndarray:
        """max-plus 稀疏传播:out[dst] = max over src (act[src]·w[src,dst])。

        Active-support frontier propagation: only visits outgoing edges of active
        nodes (act[u] > 0), O(|S_h| + |E(S_h)|) instead of O(|V| + |E|).
        Handles CSC inputs (e.g. csr_matrix.T for #521 bidirectional inhibition)
        by converting to CSR so row=src, col=dst matches the edge-list backend.
        """
        n = act.shape[0]
        out = np.zeros(n, dtype=np.float32)
        if adj.data.size == 0:
            return out
        if active is None:
            active = np.nonzero(act > 0)[0]
        if active.size == 0:
            return out
        if getattr(adj, "format", "csr") == "csc":
            adj = adj.tocsr()
        return ActivationEngine._max_plus_csr_arrays(
            adj.indptr, adj.indices, adj.data, act, active, out,
            canonical=bool(getattr(adj, "has_canonical_format", False)))

    def _propagate_excitatory(self, act: np.ndarray, active: np.ndarray,
                              prune_floor: float, rem: int) -> Tuple[np.ndarray, int, int]:
        """Exact inhibitory-safe descending-edge early-stopping propagation.

        Prunes an outgoing excitatory edge (u -> v, w) ONLY when:
          1. self._exc_wmax <= 1.0 (guaranteed non-amplifying downstream),
          2. float(act[u] * w) < prune_floor <= min(act_floor, M_safe), AND
          3. v not in B_inh(rem) (cannot reach any inhibitory source within rem hops).
        """
        n = act.shape[0]
        out = np.zeros(n, dtype=np.float32)
        indptr = self._exc_indptr
        indices = self._exc_indices
        data = self._exc_data
        if active.size == 0 or data is None or data.size == 0:
            return out, 0, 0
        total_active_edges = int((indptr[active + 1] - indptr[active]).sum())
        if total_active_edges == 0:
            return out, 0, 0
        if prune_floor <= 0.0:
            self._max_plus_csr_arrays(
                indptr, indices, data, act, active, out, canonical=self._exc_canonical)
            return out, total_active_edges, 0
        inh_exc = self._get_inh_exc_csr(rem)
        if inh_exc == "ALL":
            self._max_plus_csr_arrays(
                indptr, indices, data, act, active, out, canonical=self._exc_canonical)
            return out, total_active_edges, 0
        # Pass 1: descending-weight early stop for edges with act[u] * w >= prune_floor
        visited = 0
        strong = active[act[active] >= prune_floor]
        canonical = self._exc_canonical
        if strong.size:
            for u in strong.tolist():
                s = int(indptr[u])
                e = int(indptr[u + 1])
                if s == e:
                    continue
                a_u = act[u]
                if float(a_u * data[s]) < prune_floor:
                    continue
                if float(a_u * data[e - 1]) >= prune_floor:
                    cut = e
                else:
                    lo, hi = s + 1, e - 1
                    while lo < hi:
                        mid = (lo + hi) >> 1
                        if float(a_u * data[mid]) >= prune_floor:
                            lo = mid + 1
                        else:
                            hi = mid
                    cut = lo
                visited += cut - s
                if canonical:
                    dst = indices[s:cut]
                    out[dst] = np.maximum(out[dst], a_u * data[s:cut])
                else:
                    np.maximum.at(out, indices[s:cut], a_u * data[s:cut])
        # Pass 2: inhibitory-safe tail for edges landing in B_inh(rem)
        if inh_exc is not None:
            b_ip, b_idx, b_dat = inh_exc
            b_cnt = int((b_ip[active + 1] - b_ip[active]).sum())
            if b_cnt:
                visited += b_cnt
                self._max_plus_csr_arrays(
                    b_ip, b_idx, b_dat, act, active, out, canonical=canonical)
        pruned = max(0, total_active_edges - visited)
        return out, visited, pruned

    @staticmethod
    def _has_activation_triggers(con) -> bool:
        for master in ("sqlite_master", "sqlite_temp_master"):
            try:
                if con.execute(
                    "SELECT 1 FROM %s WHERE type='trigger' AND tbl_name='activation_nodes' LIMIT 1"
                    % master).fetchone() is not None:
                    return True
            except sqlite3.OperationalError:
                pass
        return False

    # ---------------- 种子生成(同步询问→投票) ----------------
    def _seed_by_text(self, query: str, top_k: int = 12,
                      _graph_ready: bool = False) -> List[Tuple[str, float]]:
        """种子:中文二元组倒排——同步询问全图(所有含查询词元的节点),计数投票。

        #161:空查询 / 纯空白查询**无可投票词元**——旧实现下 `toks` 含 `""`(或 `" "`),
        而 `"" in hay` 恒真,于是按表序前 top_k 个节点全以 1.0 种入、覆盖该 workset。
        现直接返回空种子(activate 由此得到 seeds=0 的空工作态)。
        判据来源:本件 #161(分诊表 `triage_lingshu_ALL.md` 行 161 判定成立)。
        """
        if not (query or "").strip():
            return []
        toks = tuple({query[i:i + 2] for i in range(len(query) - 1)} | {query})
        n_toks = max(1, len(toks))
        use_cached = (_graph_ready or self._is_graph_cache_valid()) and self._node_hays is not None
        scores = []
        if use_cached:
            for nid, hay in zip(self._node_ids, self._node_hays):
                cnt = 0
                for t in toks:
                    if t in hay:
                        cnt += 1
                if cnt:
                    scores.append((nid, cnt / n_toks))
        else:
            con = self._con()
            cur = con.cursor()
            try:
                cur.execute("SELECT id, content, tags FROM nodes")
                rows = cur.fetchall()
            except sqlite3.OperationalError:  # #714:无内容表(独立激活库)⇒ 无可投票词元
                rows = []
            con.close()
            _bh = self._build_hay
            for nid, content, tags_json in rows:
                hay = _bh(content, tags_json)
                cnt = 0
                for t in toks:
                    if t in hay:
                        cnt += 1
                if cnt:
                    scores.append((nid, cnt / n_toks))
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
        self._ensure_graph()
        t0 = time.time()
        seeds = self._seed_by_text(query, top_k, _graph_ready=True)
        idx = self._node_index
        n = len(self._node_ids)
        act = np.zeros(n, dtype=np.float32)
        seed_ids = []
        for nid, s in seeds:
            i = idx.get(nid)
            if i is not None:
                act[i] = max(act[i], s)
                seed_ids.append(nid)
        # #2165：self_condition 的 [0,1] 权重域显式闸（docstring 定义的语义就是
        # 「回注**权重**」）——修前只判 `> 0.0`，w>1 时 act0 = max(w·act_prev, seeds)
        # 成为几何放大算子（1.5^30 ≈ 127834，实测吻合），且会污染库内历史值被
        # 后续合法调用继承。非数值/非有限/越界一律 fail-closed。
        if isinstance(self_condition, bool) or not isinstance(self_condition,
                                                          (int, float)):
            raise ValueError("self_condition 必须是数值，收到 %r" % (self_condition,))
        if not math.isfinite(self_condition) or not (0.0 <= self_condition <= 1.0):
            raise ValueError("self_condition 必须落在 [0,1]，收到 %r"
                             % (self_condition,))
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
        total_edges_visited = 0
        total_edges_pruned = 0
        for hop in range(1, hops + 1):
            active = np.nonzero(act > 0)[0]
            if active.size:
                if act_floor > 0.0 and self._exc_wmax <= 1.0:
                    if self._inh_dst_mask is None:
                        m_safe = float(act[active].max())
                    else:
                        uninh = active[~self._inh_dst_mask[active]]
                        m_safe = float(act[uninh].max()) if uninh.size else 0.0
                    prune_floor = min(float(act_floor), m_safe)
                else:
                    prune_floor = 0.0
                prop, vis, pru = self._propagate_excitatory(
                    act, active, prune_floor, hops - hop)
                total_edges_visited += vis
                total_edges_pruned += pru
                act = np.maximum(act, prop)
                # 抑制通路(负记忆同价,第五篇;issue #229):opposite 边扣减目标激活,
                #   supp[dst] = max over src (act[src]·|decay|),act[dst] ← max(act[dst]−supp,0)。
                supp = np.zeros(n, dtype=np.float32)
                if self._adj_inh is not None:
                    supp = self._max_plus(self._adj_inh, act)
                    # issue #521：**低置信**边（<0.5）额外做反向抑制（双向），
                    # 未复核 0.3 边不得只按存储方向让后写者单方面否决先写者；
                    # 默认 0.5+ 的边保持既有单向（#229）。
                    if self._adj_inh_unv is not None:
                        supp = np.maximum(supp, self._max_plus(self._adj_inh_unv.T, act))
                else:  # edge-list 降级（低置信边双向同口径）
                    if self._inh_csr is not None or self._inh_unv_t_csr is not None:
                        active_exc = np.nonzero(act > 0)[0]
                        if self._inh_csr is not None:
                            self._max_plus_csr_arrays(
                                self._inh_csr[0], self._inh_csr[1], self._inh_csr[2],
                                act, active_exc, supp, canonical=False)
                        if self._inh_unv_t_csr is not None:
                            self._max_plus_csr_arrays(
                                self._inh_unv_t_csr[0], self._inh_unv_t_csr[1],
                                self._inh_unv_t_csr[2], act, active_exc, supp,
                                canonical=False)
                    else:
                        inh_rows, inh_cols, inh_mags = self._inh_meta
                        for si, di, m in zip(inh_rows, inh_cols, inh_mags):
                            if act[si] > 0:
                                supp[di] = max(supp[di], act[si] * m)
                        unv_rows, unv_cols, unv_mags = self._inh_unv_meta
                        for si, di, m in zip(unv_rows, unv_cols, unv_mags):
                            if act[di] > 0:
                                supp[si] = max(supp[si], act[di] * m)
                if supp.any():
                    act = np.maximum(act - supp, 0.0)
            above = np.nonzero(act >= act_floor)[0]
            for i in above.tolist():
                nid = self._node_ids[i]
                if nid not in first_hop:  # #314:首次进入工作态的跳号
                    first_hop[nid] = hop
            path.append({"step": hop, "phase": "propagate",
                         "activated_count": int(above.size),
                         "max_act": round(float(act.max()), 4) if n else 0.0})
        self._last_frontier_stats["edges_visited"] = total_edges_visited
        self._last_frontier_stats["edges_pruned"] = total_edges_pruned
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
        #505：`query` 与 sources/hops 同签名、同落库面——工作集必须能回溯到产生它的
        查询（旧实现收下即弃、表里也无列可存）。实际查询文本随每个成员行同写
        （同一 workset 的查询相同；按行冗余存储换取无需额外表与 join）。
        """
        state = {"track": False, "dv": None, "sv": None, "tsv": None, "mode": None}

        def _after_commit(con, was_nested):
            if not state["track"] or was_nested:
                if self._graph_cache_sig is not None and self._graph_cache_sig[0] != "file-digest":
                    self._graph_cache_sig = None
                return
            try:
                if state["mode"] == "conn":
                    dv = int(con.execute("PRAGMA data_version").fetchone()[0])
                    sv = int(con.execute("PRAGMA schema_version").fetchone()[0])
                    tsv = int(con.execute("PRAGMA temp.schema_version").fetchone()[0])
                    if (dv == state["dv"] and sv == state["sv"]
                            and tsv == state["tsv"] and not con.in_transaction):
                        raw_con = con._conn if isinstance(con, _BorrowedConnection) else con
                        self._graph_cache_sig = (
                            "conn", id(raw_con), dv, sv, tsv, int(con.total_changes))
                    else:
                        self._graph_cache_sig = None
                elif state["mode"] in ("file-rb", "file-wal"):
                    sig_after = self._file_fast_sig(con)
                    dv = int(con.execute("PRAGMA data_version").fetchone()[0])
                    sv = int(con.execute("PRAGMA schema_version").fetchone()[0])
                    if sig_after is not None and dv == state["dv"] and sv == state["sv"]:
                        self._graph_cache_sig = sig_after
                    else:
                        self._graph_cache_sig = None
            except Exception:
                self._graph_cache_sig = None

        with self._transaction(_on_commit=_after_commit) as con:
            sig = self._graph_cache_sig
            if sig is not None and sig[0] != "file-digest":
                if not self._has_activation_triggers(con):
                    if self._conn is not None and sig[0] == "conn":
                        if self._conn_sig(con, allow_in_tx=True) == sig:
                            state["track"] = True
                            state["mode"] = "conn"
                            state["dv"] = sig[2]
                            state["sv"] = sig[3]
                            state["tsv"] = sig[4]
                    elif self._conn is None and sig[0] in ("file-rb", "file-wal"):
                        if self._file_fast_sig(con) == sig:
                            state["track"] = True
                            state["mode"] = sig[0]
                            state["dv"] = int(con.execute("PRAGMA data_version").fetchone()[0])
                            state["sv"] = int(con.execute("PRAGMA schema_version").fetchone()[0])
            con.execute("DELETE FROM activation_nodes WHERE workset=?", (workset,))
            ts = time.time()
            src_map = sources or {}
            hop_map = hops or {}
            if members:
                con.executemany(
                    "INSERT OR REPLACE INTO activation_nodes"
                    " (id, workset, node_id, activation, source, hop, ts, query)"
                    " VALUES (?,?,?,?,?,?,?,?)",
                    [("act_%s_%d" % (uuid.uuid4().hex[:8], int(ts * 1000)),
                      workset, nid, a, src_map.get(nid, "propagate"),
                      hop_map.get(nid, 0), ts, query) for nid, a in members])

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
        # #2544：`size` 此前是「本页条数」——同一 workset 有 25 个成员时
        # `activate()["size"]=25` / `export_workset()` 25 条，而本方法默认
        # `limit=20` 返回 `size=20`（且无 `truncated`），`limit=0` 时返回
        # `size=0, query=None`（与「空工作集」不可区分）。`limit` 是**摘要页大小**，
        # 不是工作集大小的定义 ⇒ 用全量 COUNT 作 `size`，另报 `returned`/`truncated`。
        cur.execute("SELECT COUNT(*) FROM activation_nodes WHERE workset=?", (workset,))
        _total = int(cur.fetchone()[0])
        cur.execute("SELECT node_id, activation, query FROM activation_nodes WHERE workset=? "
                    "ORDER BY activation DESC LIMIT ?", (workset, limit))
        rows = cur.fetchall()
        # #505：工作态回读带出产生它的查询（同一 workset 的查询相同；空工作态为 None）。
        # #2544：`limit=0` 时 rows 为空但工作集非空 ⇒ query 必须独立于 limit 读取。
        query = rows[0][2] if rows else None
        if query is None and _total:
            cur.execute("SELECT query FROM activation_nodes WHERE workset=? LIMIT 1", (workset,))
            _q = cur.fetchone()
            query = _q[0] if _q else None
        con.close()
        members = []
        if rows:
            con = self._con(); cur = con.cursor()
            for nid, a, _q in rows:
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
        return {"workset": workset, "query": query, "size": _total,
                "returned": len(rows), "truncated": len(rows) < _total,
                "members": members}

    def export_workset(self, workset: str) -> Dict:
        """快照(可序列化——跨进程恢复,Retained Reasoning 的持久形态)。

        #505：快照顶层带出 `query`（产生该工作集的查询文本）——恢复后仍可回溯输入。
        """
        con = self._con()
        cur = con.cursor()
        cur.execute("SELECT node_id, activation, source, hop, ts, query"
                    " FROM activation_nodes WHERE workset=?", (workset,))
        rows = cur.fetchall()
        con.close()
        query = rows[0][5] if rows else None
        return {"algo": ALGO, "workset": workset, "ts": time.time(), "query": query,
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
        # #505：旧版快照没有 `query` 键 ⇒ None（向后兼容）；新版按快照顶层恢复。
        query = snapshot.get("query")
        with self._transaction() as con:
            ts = time.time()
            con.execute("DELETE FROM activation_nodes WHERE workset=?", (ws,))
            for m in members:
                con.execute("INSERT OR REPLACE INTO activation_nodes"
                            " (id, workset, node_id, activation, source, hop, ts, query)"
                            " VALUES (?,?,?,?,?,?,?,?)",
                            ("act_%s_%d" % (uuid.uuid4().hex[:8], int(ts * 1000)),
                             ws, m["node_id"], m["activation"], "restored",
                             m.get("hop", 0), ts, query))
        return {"status": "ok", "workset": ws, "restored": len(members)}


_ORIG_COND_MATCH = cond_match
_ORIG_CSR_MATRIX = csr_matrix
if csr_matrix is not None:
    class _FastTransposeCSR(csr_matrix):
        """csr_matrix holding a precomputed CSR transpose so `.T` is O(1) CSR."""

        _t_csr = None

        def transpose(self, axes=None, copy=False):
            if axes is None and not copy and self._t_csr is not None:
                return self._t_csr
            return super().transpose(axes=axes, copy=copy)
else:
    _FastTransposeCSR = None


if __name__ == "__main__":
    eng = ActivationEngine()
    r1 = eng.activate("灵枢 认知图 上下文", workset="demo")
    r2 = eng.activate("激活 工作态 演化", workset="demo",
                      self_condition=SELF_CONDITION_WEIGHT)
    print(json.dumps({"first": r1, "second": r2}, ensure_ascii=False, indent=1)[:600])
    st = eng.workset_state("demo", 3)
    print(json.dumps(st, ensure_ascii=False)[:400])
