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
    ``add_node`` 的裸 SQL——都会入队），检索前由 Python 增量排空；#797 起
    ``LayeredStore.add_node`` 写路径提交后**立即**排空一次（节点行与索引行同事务
    可见），绕过 ``add_node`` 的路径仍靠检索前排空兜底。
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
import math
import numbers
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


# ---------- 结构锚定（LINGSHU_ANCHOR=1；默认关闭，与 LINGSHU_FTS 独立） ----------
# 问句点名一个「容器」（第七章 / 第3次会话 / chapter 7）时，容器本身就是最强的检索信号：
# 容器里的内容不会复述「第七章」「在讲什么」这些字，字面检索因此系统性落空。
# 契约：写入方用标签 `<kind>:<n>` 标记容器（chapter:7、session:3）；查询侧解析锚点 →
# 取同标签节点 → 按容器内相关度排在最前，不足 limit 再用原排序补齐。无锚点 → 原路径逐位不变。
ANCHOR_FLAG = "LINGSHU_ANCHOR"
_CN_DIGIT = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
             "六": 6, "七": 7, "八": 8, "九": 9}
_NUM = r"([0-9]{1,4}|[零〇一二两三四五六七八九十百]{1,6})"
# (正则, 容器种类)。只认「第N+量词」与英文 chapter/session N 两种显式写法，不猜隐式指代。
ANCHOR_PATTERNS = (
    (re.compile("第" + _NUM + r"\s*[章回]"), "chapter"),
    (re.compile("第" + _NUM + r"\s*[次场轮]?\s*(?:会话|对话|聊天)"), "session"),
    (re.compile(r"\bchapter\s+([0-9]{1,4})\b", re.I), "chapter"),
    (re.compile(r"\bsession\s+([0-9]{1,4})\b", re.I), "session"),
)


CONTAINER_TAG = re.compile(r"^(?:chapter|session):[0-9]+$")


def anchor_enabled() -> bool:
    return os.environ.get(ANCHOR_FLAG, "").strip() == "1"


def cn_number(s: str) -> Optional[int]:
    """阿拉伯数字或 1–999 的中文数字（十五、二十、一百零三）；无法解析返回 None。"""
    if s.isdigit():
        return int(s)
    if not s or any(ch not in _CN_DIGIT and ch not in "十百" for ch in s):
        return None
    total, cur = 0, None
    for ch in s:
        if ch in _CN_DIGIT:
            cur = _CN_DIGIT[ch]
        else:
            unit = 10 if ch == "十" else 100
            total += (1 if cur is None else cur) * unit
            cur = None
    total += cur or 0
    return total or None


def parse_anchors(query: str) -> List[str]:
    """问句中显式点名的容器 → 标签列表（保序去重），如「第十五章」→ ['chapter:15']。"""
    out: List[str] = []
    for pat, kind in ANCHOR_PATTERNS:
        for m in pat.finditer(query or ""):
            n = cn_number(m.group(1))
            if n is not None:
                tag = f"{kind}:{n}"
                if tag not in out:
                    out.append(tag)
    return out


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
    except (TypeError, ValueError):
        #: ⚠ 收窄到 json 解析的**真实失败面**（`json.JSONDecodeError` ⊂ `ValueError`；
        #: `tags_json` 非字符串则 `TypeError`）。原为宽 `except Exception` 且静默
        #: ⇒ 既让 `silent_except_lint` 超基线，也会把「库读坏」这类真错一并吞掉。
        tags = []
    #: #707：反序列化**成功但形状不符**（`null` / `5` / `true` / 对象）此前落在 try 之外，
    #: `for t in tags` 抛 `TypeError: 'int' object is not iterable` 并穿透 `sync()` /
    #: `attach()`（两者都不接 TypeError）⇒ 一条坏行毒化整库检索（`candidates()` 每次先
    #: `sync()`）。口径与姊妹函数 `provenance._tags` 对齐：JSON 字符串标量按**单个标签**
    #: 处理（勿按字符拆），其余非 list/tuple 形状一律回落到 `[]`。
    if isinstance(tags, str):
        tags = [tags]
    if not isinstance(tags, (list, tuple)):
        tags = []
    # 结构容器标签（chapter:7 / session:3）是元数据，不进全文索引：否则「chapter」的拉丁二元组
    # 与单个数字会成为与内容无关的可匹配词项，打标签本身就会改变排序。
    text = (content or "") + " " + " ".join(str(t) for t in tags
                                            if t is not None and not CONTAINER_TAG.match(str(t)))
    return " ".join(grams(text))


_SQLITE_MAX_LIMIT = (1 << 62) - 1


def _normalize_candidates_k(k: Optional[int]) -> Optional[int]:
    """校验并归一 ``FTSIndex.candidates`` 的上限参数 ``k``（对齐 #658/#1795/#1474 口径）。

    - ``bool`` 与非整数且非 ``None`` 入参（如 ``True``/``False``/``1.5``/``nan``/``"x"``）
      → fail-loud ``TypeError``（``bool`` 是 ``int`` 子类，须显式拦截，避免 ``int(True)==1``）；
    - ``None`` 或负数（如 ``-1``/``-5``）→ 返回 ``None``（表示全量返回，与 SQLite
      ``LIMIT -1`` 及 ``core._normalize_limit`` 同口径，杜绝负 ``LIMIT`` 全量取回后
      再遭 Python 负切片 ``rows[:-1]`` 静默丢尾）；
    - ``0`` → 返回 ``0``（空列表）；
    - 正整数 → 原样返回。
    """
    if k is not None and (isinstance(k, bool) or not isinstance(k, int)):
        raise TypeError("candidates 的 k 必须是整数或 None，收到 %r" % (k,))
    if k is None or k < 0:
        return None
    return k


def _require_finite_real(value, name: str) -> float:
    """校验有限实数且拒绝 ``bool``（#196 第十三批）。"""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} 必须是有限实数（不能为 bool），收到 {value!r}")
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} 必须是有限实数，收到 {value!r}") from exc
    if not math.isfinite(f):
        raise ValueError(f"{name} 必须有限（不能为 NaN/±inf），收到 {value!r}")
    return f


def _require_positive_int(value, name: str) -> int:
    """校验正整数（>= 1）且拒绝 ``bool``（#196 第十三批）。"""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} 必须是正整数（>= 1，不能为 bool），收到 {value!r}")
    return value


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

    def _tracking_epoch(self) -> int:
        """派生索引失去跟踪时交回原检索路径，不把空缓存当成无记忆。

        默认配置的另一实例会 detach 全库触发器；存活对象不能仅凭存在就
        继续相信索引。schema_version 同时用于识别一次读取期间的拆除/重挂。
        """
        conn = self.store.conn
        epoch = conn.execute("PRAGMA schema_version").fetchone()[0]
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND name IN (?,?,?)",
            _TRIGGERS)}
        if names != set(_TRIGGERS):
            raise sqlite3.OperationalError("FTS 跟踪触发器不完整，回退原检索路径")
        return epoch

    def _require_tracking_epoch(self, epoch: int) -> None:
        if self.store.conn.execute("PRAGMA schema_version").fetchone()[0] != epoch:
            raise sqlite3.OperationalError("FTS 读取期间派生结构已变更，回退原检索路径")

    # ---------- 增量同步 ----------
    def sync(self, batch: int = 2000) -> int:
        batch = _require_positive_int(batch, "batch")
        sql_limit = min(batch, _SQLITE_MAX_LIMIT)
        store = self.store
        done = 0
        epoch = self._tracking_epoch()
        while True:
            c = store.conn
            ids = [r[0] for r in c.execute("SELECT nid FROM nodes_fts_dirty LIMIT ?", (sql_limit,))]
            if not ids:
                self._require_tracking_epoch(epoch)
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
                   k: Optional[int] = 100) -> Optional[List[Tuple[tuple, float]]]:
        """返回 [(nodes 行, BM25 正分)]，按 BM25 降序；无可用二元组时返回 None（交回原路径）。

        上限 ``k`` 口径（对齐 #658 / #1795 / #1474 与 ``core._normalize_limit``）：
          - 正整数：最多返回 ``k`` 条候选；
          - ``0``：返回空列表 ``[]``（若 ``query_terms`` 无可用二元组仍先返回 ``None``）；
          - ``None`` 或负整数：返回全部命中候选（SQLite ``LIMIT -1``，不做 Python 负切片）；
          - ``bool`` 或其它非整数类型：fail-loud 抛出 ``TypeError``。

        #797：``query_terms`` 收到**裸字符串**（如 ``candidates('full text search')``）时，
        ``str`` 本身是 ``Sequence[str]``——按字符迭代会把查询拆成单字，而索引文档是
        二元组 token，单字 MATCH 恒零命中 ⇒ 恒空列表（#797 复现读数之二）。此处把
        裸字符串按**一个词条**处理（与 ``candidates(['full text search'])`` 等价，
        ``grams`` 自带空白折叠，跨词二元组同样成立）。内部调用方（``_search_content_fts``
        传词条列表）行为不变。"""
        want = _normalize_candidates_k(k)
        if isinstance(query_terms, str):
            query_terms = [query_terms]
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
        epoch = self._tracking_epoch()
        self.sync()
        if want == 0:
            self._require_tracking_epoch(epoch)
            return []
        match = " OR ".join('"' + g + '"' for g in toks)
        # 先在 FTS 内部按 rank 截断（rank 即 bm25，FTS5 对 ORDER BY rank LIMIT 走 top-k 优化），
        # 再回表取节点——若先 JOIN 再排序，高频二元组会让全部命中行回表，50k 时退化到秒级。
        conn = self.store.conn
        if want is None:
            inner = -1
        else:
            raw_inner = want if not layers else want * 4
            inner = -1 if raw_inner > _SQLITE_MAX_LIMIT else raw_inner
        while True:
            sql = ("SELECT n.*, f.b AS _b FROM (SELECT rowid AS rid, rank AS b FROM nodes_fts "
                   "WHERE nodes_fts MATCH ? ORDER BY rank LIMIT ?) f "
                   "JOIN nodes_fts_map m ON m.frid = f.rid JOIN nodes n ON n.id = m.nid")
            params: list = [match, inner]
            if layers:
                sql += f" WHERE n.layer IN ({','.join('?' for _ in layers)})"
                params.extend(l.value for l in layers)
            sql += " ORDER BY f.b, f.rid"  # 同分按 FTS rowid（写入顺序）定序，跨平台确定
            rows = conn.execute(sql, params).fetchall()
            # 层过滤后不足且内层已截满 → 放大内层再取（有界：最多到全部命中）
            if want is None or inner < 0 or not layers or len(rows) >= want or inner >= 1 << 20:
                break
            n_inner = len(conn.execute("SELECT rowid FROM nodes_fts WHERE nodes_fts MATCH ? "
                                       "LIMIT ?", (match, inner + 1)).fetchall())
            if n_inner <= inner:
                break
            inner = inner << 2  # 内层放大 4 倍
        self._require_tracking_epoch(epoch)
        selected = rows if want is None else rows[:want]
        return [(tuple(r)[:-1], -float(r["_b"])) for r in selected]

    def scores_within(self, query_terms: Sequence[str], nids: Sequence[str]) -> dict:
        """{nid: BM25 正分}：只在给定节点集合内打分（结构锚定的容器内排序）；未命中者不出现。

        #797：与 ``candidates`` 同款裸字符串容错（见该方法 docstring）。"""
        if isinstance(query_terms, str):
            query_terms = [query_terms]
        toks: List[str] = []
        seen = set()
        for t in query_terms:
            for g in grams(t):
                if g not in seen:
                    seen.add(g)
                    toks.append(g)
        if not toks or not nids:
            return {}
        epoch = self._tracking_epoch()
        self.sync()
        match = " OR ".join('"' + g + '"' for g in toks[:MAX_QUERY_GRAMS])
        conn = self.store.conn
        out = {}
        ids = list(nids)
        for i in range(0, len(ids), 500):  # 分批，避开 SQLite 变量上限
            part = ids[i:i + 500]
            sql = ("SELECT m.nid AS nid, f.rank AS b FROM nodes_fts f JOIN nodes_fts_map m "
                   "ON m.frid = f.rowid WHERE nodes_fts MATCH ? AND m.nid IN (%s)"
                   % ",".join("?" for _ in part))
            for r in conn.execute(sql, [match] + part).fetchall():
                out[r["nid"]] = -float(r["b"])
        self._require_tracking_epoch(epoch)
        return out

    def temporal_neighbors(self, nids: Sequence[str], layers=None,
                           window: float = NEIGHBOR_WINDOW_S) -> List[Tuple[str, tuple]]:
        """[(种子 nid, 邻居 nodes 行)]：每个种子取 temporal_coordinate 紧邻的前一条与后一条，
        间隔超过 window 秒的不算（会话边界）。同层过滤与 search_content 一致。"""
        win = _require_finite_real(window, "window")
        if win < 0.0:
            raise ValueError(f"window 必须是非负有限实数，收到 {window!r}")
        conn = self.store.conn
        lay, lp = "", []
        if layers:
            # `+layer`：一元加号让 SQLite 不用 layer 索引，保住 (tc, rowid) 索引的有序扫描
            lay = f" AND +layer IN ({','.join('?' for _ in layers)})"
            lp = [l.value for l in layers]
        out = []
        for nid in nids:
            r = conn.execute("SELECT temporal_coordinate, rowid FROM nodes WHERE id=?",
                             (nid,)).fetchone()
            if not r or r[0] is None:
                continue
            tc, rid = float(r[0]), int(r[1])
            # 全序 (temporal_coordinate, rowid)：时钟粗粒度（Windows time.time() ≈15.6 ms）下
            # 同一 tick 写入的多条 tc 完全相同，只按 tc 取前/后一条会在并列里任意选、且跳过同
            # tick 的真正前后条。以写入顺序 rowid 作显式 tiebreak ⇒ 邻居唯一、跨平台可复现。
            # 两条都走 idx_nodes_fts_tc（索引键 = (tc, rowid)）：先找同 tick 内的紧邻，没有再跨 tick。
            # 合成一条 `tc<? OR (tc=? AND rowid<?)` 会让 SQLite 放弃索引退化为全表扫描。
            for op, order in (("<", "DESC"), (">", "ASC")):
                row = conn.execute(
                    f"SELECT * FROM nodes WHERE temporal_coordinate = ? AND rowid {op} ? "
                    f"AND id != ?{lay} ORDER BY rowid {order} LIMIT 1",
                    [tc, rid, nid] + lp).fetchone()
                if row is None:
                    row = conn.execute(
                        f"SELECT * FROM nodes WHERE temporal_coordinate {op} ? AND id != ?{lay} "
                        f"ORDER BY temporal_coordinate {order}, rowid {order} LIMIT 1",
                        [tc, nid] + lp).fetchone()
                if row is not None and abs(float(row["temporal_coordinate"]) - tc) <= win:
                    out.append((nid, tuple(row)))
        return out

    @staticmethod
    def blend(jaccard: float, bm25_pos: float) -> float:
        j = _require_finite_real(jaccard, "jaccard")
        b = _require_finite_real(bm25_pos, "bm25_pos")
        r = b / (b + BM25_K) if b > 0 else 0.0
        return min(1.0, max(j, BM25_CAP * r))
