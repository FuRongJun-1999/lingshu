"""检索候选生成：FTS5 字符二元组倒排索引 + BM25 排序（默认关闭）。

动机（Refs #35 / #82 / #132 / #266）：
  现行 ``LayeredStore.search_content`` 先用 ``LIKE`` 子串预筛（``LIMIT 300``、无 ORDER BY），
  落空即回退扫「前 500 行」再做二元组 Jaccard。自然语言问句、带空格的关键词串几乎
  命不中连续子串，于是结果只取决于**插入顺序**；库一过 500 条，后写入的记忆在这类
  查询下永久不可达。瓶颈在**候选生成**而非打分公式。

本模块：
  - 维护一张 FTS5 表，文档 = 节点 content + tags 的字符二元组（汉字/字母数字连续段内
    取相邻两字；单字段保留单字），空格分隔交给 ``unicode61`` 切词——全确定、白箱、可溯源，
    零新依赖（标准库 sqlite3 自带 FTS5；不可用时整体回退原路径）。
  - 同步不靠在触发器里切词（那会要求每条连接都注册 Python 函数）：触发器只把变更节点
    id 写进脏队列 ``nodes_fts_dirty``（纯 SQL，任何连接、任何写入路径——包括绕过
    ``add_node`` 的裸 SQL——都会入队），检索前由 Python 增量排空。
  - ``INSERT OR REPLACE`` 会换 rowid，故 FTS 行与节点经 ``nodes_fts_map(nid→frid)``
    以 id 关联；内容哈希不变的重复写入（如只改置信度）跳过重建。

打分契约（调用方有 ``>=0.95`` 判重、``>=0.25`` 建边等阈值，必须保持语义）：
  score = max(原 Jaccard(+tag 加成), 0.9 · s/(s+K))，s 为 BM25 正分。
  - 近重复判定（≥0.95）只可能来自 Jaccard——BM25 分量封顶 0.9，不会把「包含查询的长文」
    误判为重复；
  - 排序主要由 BM25 决定（IDF 抑制「什么/为什」等高频二元组，长度归一化温和）。

开关：环境变量 ``LINGSHU_FTS=1``（默认关闭）。关闭时若库里残留本模块的触发器，
启动即移除（派生数据，可随时重建），保证默认写路径与原实现逐字节一致。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from typing import List, Optional, Sequence, Tuple

ENV_FLAG = "LINGSHU_FTS"
BM25_K = 10.0          # s/(s+K) 的半饱和点
BM25_CAP = 0.9         # BM25 分量封顶，保证 ≥0.95 只来自 Jaccard
MAX_QUERY_GRAMS = 256  # 超长查询（#266）截断 OR 项数，防 FTS 表达式膨胀
_SEG = re.compile(r"[0-9a-z\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")

_TRIGGERS = ("nodes_fts_ai", "nodes_fts_au", "nodes_fts_ad")


def enabled() -> bool:
    return os.environ.get(ENV_FLAG, "").strip() == "1"


# ---------- 语境扩展（LINGSHU_FTS_CONTEXT=1，须同时开 LINGSHU_FTS；默认关闭） ----------
CONTEXT_FLAG = "LINGSHU_FTS_CONTEXT"
NEIGHBOR_GAMMA = 0.2        # 时序邻居继承的 BM25 份额（只在切A 上选定）
NEIGHBOR_WINDOW_S = 3600.0  # 只把 1 小时内的前后条视为同一语境
NEIGHBOR_SEEDS = 30         # 只从 BM25 前 N 个候选向外扩

# 问句/指令功能二元组：出题模板里的「请说明」「哪些」「这部作品」「人物」等，几乎与任何
# 叙述块都共现，却不指向证据。手写通用表（不依赖任何语料统计），只作用于查询侧；
# 全部被滤掉时保留原查询，避免空查询。
QUERY_FUNCTION_GRAMS = frozenset("""
请说 说明 请指 指出 请结 结合 分析 请分 简述 请简 概括 请概 描述 请描 为什 什么 怎样 怎么
如何 哪些 哪个 哪里 哪儿 是否 是不 有没 没有 至少 两处 一处 几处 各自 作用 方式 关系 以及
这部 部作 作品 这篇 篇小 小说 故事 事里 品里 里的 中的 的人 人物 是个 个怎 样的 的作 的身
身份 立场 体现 表现 具体 例子 举例 原因 为何 何种 哪种 么样 么要 么方 用什 出来 了出 讲了
做了 上做 安排 设计 并说 明各 明他 他的 她的 它的 们的 这个 那个 这些 那些 一个 时候 之间
前后 后来 最后 开始
""".split())


def context_enabled() -> bool:
    return enabled() and os.environ.get(CONTEXT_FLAG, "").strip() == "1"


def content_grams(terms: Sequence[str]) -> List[str]:
    """查询二元组去掉问句功能词；全被去掉时原样返回。"""
    toks, seen = [], set()
    for t in terms:
        for g in grams(t):
            if g not in seen:
                seen.add(g)
                toks.append(g)
    kept = [g for g in toks if g not in QUERY_FUNCTION_GRAMS]
    return kept or toks


def grams(text: str) -> List[str]:
    """字符二元组（段内），单字段保留单字；保序去重。

    先折叠空白（与 char_bigram_jaccard 的 `"".join(s.split())` 同口径，跨空格的二元组
    同样成立），再按标点等非字词字符切段。"""
    out, seen = [], set()
    for seg in _SEG.findall("".join((text or "").lower().split())):
        toks = [seg] if len(seg) == 1 else [seg[i:i + 2] for i in range(len(seg) - 1)]
        for t in toks:
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out


def _doc(content: str, tags_json: str) -> str:
    try:
        tags = json.loads(tags_json) if tags_json else []
    except Exception:
        tags = []
    text = (content or "") + " " + " ".join(str(t) for t in tags if t is not None)
    return " ".join(grams(text))


class FTSIndex:
    """挂在 LayeredStore 上的派生索引；所有方法失败都只让调用方回退原路径。"""

    def __init__(self, store):
        self.store = store

    # ---------- 建立 / 拆除 ----------
    @classmethod
    def attach(cls, store) -> Optional["FTSIndex"]:
        try:
            with store._tx():
                c = store.conn
                c.execute("CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts "
                          "USING fts5(toks, tokenize='unicode61')")
                c.execute("CREATE TABLE IF NOT EXISTS nodes_fts_map "
                          "(nid TEXT PRIMARY KEY, frid INTEGER, h TEXT)")
                c.execute("CREATE TABLE IF NOT EXISTS nodes_fts_dirty (nid TEXT PRIMARY KEY)")
                # 语境扩展按 temporal_coordinate 取前后条；索引只服务检索，属派生结构
                c.execute("CREATE INDEX IF NOT EXISTS idx_nodes_fts_tc ON nodes(temporal_coordinate)")
                c.execute("CREATE TRIGGER IF NOT EXISTS nodes_fts_ai AFTER INSERT ON nodes "
                          "BEGIN INSERT OR IGNORE INTO nodes_fts_dirty VALUES (new.id); END")
                c.execute("CREATE TRIGGER IF NOT EXISTS nodes_fts_au AFTER UPDATE OF id, content, tags "
                          "ON nodes BEGIN INSERT OR IGNORE INTO nodes_fts_dirty VALUES (old.id); "
                          "INSERT OR IGNORE INTO nodes_fts_dirty VALUES (new.id); END")
                c.execute("CREATE TRIGGER IF NOT EXISTS nodes_fts_ad AFTER DELETE ON nodes "
                          "BEGIN INSERT OR IGNORE INTO nodes_fts_dirty VALUES (old.id); END")
                # 首次挂载 / 曾被拆除：全量入队（只入队 map 里没有的，幂等）
                c.execute("INSERT OR IGNORE INTO nodes_fts_dirty "
                          "SELECT id FROM nodes WHERE id NOT IN (SELECT nid FROM nodes_fts_map)")
            idx = cls(store)
            idx.sync()
            return idx
        except sqlite3.OperationalError:
            return None  # 编译时未带 FTS5 / 只读库 → 回退原路径

    @staticmethod
    def detach(store) -> None:
        """开关关闭时移除触发器（保持默认写路径不变）；索引表留作派生缓存，重开时增量补齐。"""
        try:
            c = store.conn
            have = {r[0] for r in c.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE 'nodes_fts_%'")}
            if not have:
                return
            with store._tx():
                for t in _TRIGGERS:
                    c.execute(f"DROP TRIGGER IF EXISTS {t}")
                # 触发器拆除后索引不再同步：清空映射，重开时全量重建，杜绝陈旧命中
                c.execute("DELETE FROM nodes_fts_map")
                c.execute("DELETE FROM nodes_fts")
                c.execute("DELETE FROM nodes_fts_dirty")
        except sqlite3.Error:
            pass

    # ---------- 增量同步 ----------
    def sync(self, batch: int = 2000) -> int:
        store = self.store
        done = 0
        while True:
            c = store.conn
            ids = [r[0] for r in c.execute("SELECT nid FROM nodes_fts_dirty LIMIT ?", (batch,))]
            if not ids:
                return done
            with store._tx():
                for nid in ids:
                    row = c.execute("SELECT content, tags FROM nodes WHERE id=?", (nid,)).fetchone()
                    old = c.execute("SELECT frid, h FROM nodes_fts_map WHERE nid=?", (nid,)).fetchone()
                    if row is None:
                        if old is not None:
                            c.execute("DELETE FROM nodes_fts WHERE rowid=?", (old[0],))
                            c.execute("DELETE FROM nodes_fts_map WHERE nid=?", (nid,))
                    else:
                        doc = _doc(row[0] if isinstance(row[0], str) else str(row[0] or ""), row[1])
                        h = hashlib.blake2b(doc.encode("utf-8", "surrogatepass"),
                                            digest_size=12).hexdigest()
                        if old is None or old[1] != h:
                            if old is not None:
                                c.execute("DELETE FROM nodes_fts WHERE rowid=?", (old[0],))
                            cur = c.execute("INSERT INTO nodes_fts(toks) VALUES (?)", (doc,))
                            c.execute("INSERT OR REPLACE INTO nodes_fts_map VALUES (?,?,?)",
                                      (nid, cur.lastrowid, h))
                    c.execute("DELETE FROM nodes_fts_dirty WHERE nid=?", (nid,))
            done += len(ids)

    # ---------- 检索 ----------
    def candidates(self, query_terms: Sequence[str], layers=None,
                   k: int = 100) -> Optional[List[Tuple[tuple, float]]]:
        """返回 [(nodes 行, BM25 正分)]，按 BM25 降序；无可用二元组时返回 None（交回原路径）。"""
        toks: List[str] = []
        seen = set()
        for t in query_terms:
            for g in grams(t):
                if g not in seen:
                    seen.add(g)
                    toks.append(g)
        if not toks:
            return None
        toks = toks[:MAX_QUERY_GRAMS]
        self.sync()
        match = " OR ".join('"' + g + '"' for g in toks)
        # 先在 FTS 内部按 rank 截断（rank 即 bm25，FTS5 对 ORDER BY rank LIMIT 走 top-k 优化），
        # 再回表取节点——若先 JOIN 再排序，高频二元组会让全部命中行回表，50k 时退化到秒级。
        conn = self.store.conn
        want = int(k)
        inner = want if not layers else want * 4
        while True:
            sql = ("SELECT n.*, f.b AS _b FROM (SELECT rowid AS rid, rank AS b FROM nodes_fts "
                   "WHERE nodes_fts MATCH ? ORDER BY rank LIMIT ?) f "
                   "JOIN nodes_fts_map m ON m.frid = f.rid JOIN nodes n ON n.id = m.nid")
            params: list = [match, inner]
            if layers:
                sql += f" WHERE n.layer IN ({','.join('?' for _ in layers)})"
                params.extend(l.value for l in layers)
            sql += " ORDER BY f.b"
            rows = conn.execute(sql, params).fetchall()
            # 层过滤后不足且内层已截满 → 放大内层再取（有界：最多到全部命中）
            if not layers or len(rows) >= want or inner >= 1 << 20:
                break
            n_inner = len(conn.execute("SELECT rowid FROM nodes_fts WHERE nodes_fts MATCH ? "
                                       "LIMIT ?", (match, inner + 1)).fetchall())
            if n_inner <= inner:
                break
            inner = inner << 2  # 内层放大 4 倍
        return [(tuple(r)[:-1], -float(r["_b"])) for r in rows[:want]]

    def temporal_neighbors(self, nids: Sequence[str], layers=None,
                           window: float = NEIGHBOR_WINDOW_S) -> List[Tuple[str, tuple]]:
        """[(种子 nid, 邻居 nodes 行)]：每个种子取 temporal_coordinate 紧邻的前一条与后一条，
        间隔超过 window 秒的不算（会话边界）。同层过滤与 search_content 一致。"""
        conn = self.store.conn
        lay, lp = "", []
        if layers:
            lay = f" AND layer IN ({','.join('?' for _ in layers)})"
            lp = [l.value for l in layers]
        out = []
        for nid in nids:
            r = conn.execute("SELECT temporal_coordinate FROM nodes WHERE id=?", (nid,)).fetchone()
            if not r or r[0] is None:
                continue
            tc = float(r[0])
            for op, order in (("<", "DESC"), (">", "ASC")):
                row = conn.execute(
                    f"SELECT * FROM nodes WHERE temporal_coordinate {op} ? AND id != ?{lay} "
                    f"ORDER BY temporal_coordinate {order} LIMIT 1", [tc, nid] + lp).fetchone()
                if row is not None and abs(float(row["temporal_coordinate"]) - tc) <= window:
                    out.append((nid, tuple(row)))
        return out

    @staticmethod
    def blend(jaccard: float, bm25_pos: float) -> float:
        r = bm25_pos / (bm25_pos + BM25_K) if bm25_pos > 0 else 0.0
        return min(1.0, max(jaccard, BM25_CAP * r))
