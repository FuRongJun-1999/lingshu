# -*- coding: utf-8 -*-
"""longterm_gate · 长期记忆写入决策器（LongTermMemoryGate）
================================================
机制：记忆快照 → 重要性评估（信息差/信任/二阶变化/提及次数）→
决策写入层级（长期层/知识层/情境层）+ 条件空间 + 关联边。

特征来源（复用已有机制，零新增依赖）：
- 信息差 D_norm     → engine._gap_history / get_gap_trend（A-4）
- 信任 T            → self_model.trust_state["t_total"]（协议信任状态）
- Δ²D 信息差二阶     → gap_history 二阶差分（加速/减速）
- Δ²T 信任二阶      → trust_history 二阶差分（信任跃升）
- 提及次数 N        → 候选节点 access_count / 本轮提及

评分：imp = w1·新信息度 + w2·T + w3·Δ²D + w4·Δ²T + w5·log(1+N)
决策：
- imp ≥ 0.7  → 长期层：高 importance + protect_node（不可遗忘保护）+ 关联边
- 0.4 ≤ imp < 0.7 → 知识层常规写入（importance = imp）
- imp < 0.4  → 情境层（短期，睡眠巩固时再评估提升）
"""
import hashlib
import logging
import time

_log = logging.getLogger(__name__)


class LongTermMemoryGate:
    """长期记忆写入决策器（挂载于引擎，纯标准库）。"""

    # 默认权重（可配；场景化权重调整留作后续）
    DEFAULT_WEIGHTS = {
        "novelty": 0.30,   # 新信息度（核心词新颖比例；参考集 = 全库既有知识）
        "trust": 0.25,     # 信任（#501 如实标注：取 self_model.trust_state 的
                           #   会话级全局信任，**非**本次快照的来源维度——
                           #   evaluate 的 source 形参当前未参与评分）
        "d2": 0.15,        # 信息差二阶变化（加速=新领域涌现）
        "t2": 0.15,        # 信任二阶变化（信任跃升=里程碑/校准锚点）
        "mention": 0.15,   # 提及次数（重复=重要性信号）
    }
    LONG_TERM_THRESHOLD = 0.70
    KNOWLEDGE_THRESHOLD = 0.40

    def __init__(self, engine, weights: dict = None):
        self.engine = engine
        self.weights = dict(self.DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        # 全库核心词表缓存（lingshu #389）：按 (节点数, max rowid) 签名失效。
        self._known_cache = None
        self._known_sig = None
        # #770②：正文总长另存一份——签名第三项已改为**内容指纹**，而增量分支的
        # 第三条门槛仍是「正文总长增加」（口径同改前的 `SUM(LENGTH(content))`，
        # 只对 str 正文计数），故长度单独保留。
        self._known_total_len = 0

    # ---- 特征提取 ----

    def _d_norm(self) -> float:
        """当前信息差 D_norm（0-1；无样本 0.5 中性）。"""
        hist = getattr(self.engine, "_gap_history", None) or []
        return hist[-1]["d_norm"] if hist else 0.5

    def _d_second(self) -> float:
        """信息差二阶变化 Δ²D：**最近一点**的二阶差分（需 ≥3 样本）。
        正=信息差加速扩大（新领域涌现信号）；负=收敛。

        #387 修复：旧实现 `d1[-1] - d1[0]` 是**窗口两端一阶差之差**
        （跨整窗的趋势量），不是二阶差分——与文件头 `:10`「gap_history 二阶差分」
        及本函数 docstring「二阶变化」不符。后果：只要窗口内 slope 曾变过，
        该量就按**窗口跨度**取满，`max(-0.5, …)` 常被钳到 −0.5 档（与最近一步的
        真实加速度无关）。现取 d1[-1] - d1[-2]（= vals[-1] - 2·vals[-2] + vals[-3]），
        即最近一点的二阶差分。
        判据来源：本文件头 `:10-11` 的声明（「Δ²D 信息差二阶 → gap_history
        二阶差分（加速/减速）」）＋ 二阶差分的标准定义（Δ²x_n = x_n − 2x_{n−1}
        + x_{n−2}）；理论章节追不到（本仓无 longterm_gate 独立理论稿）⇒ 属
        文件内声明与实现的自洽性修复。
        """
        hist = getattr(self.engine, "_gap_history", None) or []
        if len(hist) < 3:
            return 0.0
        vals = [h["d_norm"] for h in hist[-5:]]
        d1 = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
        if len(d1) < 2:
            return 0.0
        return max(-0.5, min(0.5, d1[-1] - d1[-2]))  # 归一化钳制

    def _trust(self) -> float:
        """当前信任值 T（0-1）。

        未初始化 → 中性 0.5：SelfModel.trust_state 初值为 `t_total: 0.0`
        （core.py:290），而全仓无内部调用者更新它（update_trust_state 仅定义）。
        旧实现 `ts.get("t_total", 0.5)` 的 fallback 对**存在但为初值 0.0** 的键
        永不生效，于是默认态 T=0.0，评分上限被压到 0.4725 < 0.70，长期层
        结构性不可达、全新内容快照一律 discarded（lingshu #114）。
        判据来源：lingshu #114 §六·建议1；`:66` 既有 fallback 0.5 的原本语义。
        真实 0.0（已 update_trust_state 且 trust_history 非空）仍如实返回 0.0。
        """
        sm = getattr(self.engine, "self_model", None)
        ts = getattr(sm, "trust_state", None) or {}
        th = getattr(sm, "trust_history", None) or []
        if "t_total" not in ts:
            return 0.5          # 无该键：沿用原 fallback
        t = ts.get("t_total")
        if t is None or (not th and float(t) == 0.0):
            return 0.5          # 初值 0.0 且无历史 → 未初始化，取中性
        return float(t)

    def _trust_second(self) -> float:
        """信任二阶变化 Δ²T：trust_history 的**最近一点**二阶差分（信任跃升信号）。

        #387 修复：与 `_d_second` 同根——旧实现 `d1[-1] - d1[0]` 是窗口两端一阶差
        之差（跨整窗趋势量），不是二阶差分；现取 `d1[-1] - d1[-2]`。
        判据来源同 `_d_second`（文件头 `:11` 声明 ＋ 二阶差分标准定义）。
        """
        th = getattr(getattr(self.engine, "self_model", None),
                     "trust_history", None) or []
        if len(th) < 3:
            return 0.0
        vals = [h["t_total"] for h in th[-6:]]
        d1 = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
        if len(d1) < 2:
            return 0.0
        return max(-0.5, min(0.5, d1[-1] - d1[-2]))

    # 非 ASCII 文字面：CJK 基本区/扩展A/兼容表意 + 假名 + 谚文 + 西里尔 + 拉丁扩展 + 希腊。
    # 旧实现只留 \u4e00-\u9fff，假名/谚文/西里尔等一律被删（lingshu #213）。
    _SCRIPT_CHARS = (r'\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff'
                     r'\u3040-\u309f\u30a0-\u30ff\uff66-\uff9f'
                     r'\u1100-\u11ff\u3130-\u318f\uac00-\ud7af'
                     r'\u0400-\u04ff\u0370-\u03ff\u00c0-\u024f')
    # ASCII/数字标识符整体 token（含 . - : / @ _ 等路径/域名/版本号字符）。
    _ASCII_TOKEN = None  # 惰性编译，见 _core_grams

    @staticmethod
    def _core_grams(text: str) -> set:
        """核心词：CJK/假名/谚文/西里尔等按 3/4 字片段；ASCII 标识符按整体 token。

        lingshu #213：旧实现两道过滤（`[^\\u4e00-\\u9fffA-Za-z0-9]` 删除 +
        `^[\\dA-Za-z_]+$` 丢弃纯 ASCII 片段）把英文/代码/数字/非汉字文字全部删光，
        grams 为空 → novelty 恒 0.5（NOVEL_TRIGGER=0.75 ⇒ prefeed 永不触发），
        且中文句里只改数字/标识符时该变更不可见。现：①第一道过滤保留更多文字面
        ②纯 ASCII 片段不再整片丢弃，而是按「标识符整体」作为 token 参与。
        判据来源：lingshu #213 §六·建议1。
        未修：模板相同、仅 ASCII token 不同的输入仍按比例计新（不足 0.75 触发）
        ——建议2 的模板比对属独立判据，未纳入本次最小修复。
        """
        import re as _re
        s = text or ""
        grams = set()
        t = _re.sub('[^%sA-Za-z0-9]' % LongTermMemoryGate._SCRIPT_CHARS, '', s)
        for n in (4, 3):
            for i in range(len(t) - n + 1):
                g = t[i:i + n]
                if len(g) == n and not _re.match(r'^[\dA-Za-z_]+$', g):
                    grams.add(g)
        if LongTermMemoryGate._ASCII_TOKEN is None:
            LongTermMemoryGate._ASCII_TOKEN = _re.compile(
                r'[A-Za-z0-9][A-Za-z0-9_.\-:/@]*')
        for m in LongTermMemoryGate._ASCII_TOKEN.findall(s):
            if len(m) >= 2:      # 单字符 token（a/I/1）是噪音
                grams.add(m)
        return grams

    def _known_grams(self, exclude_id: str = None) -> set:
        """已有知识的核心词表——覆盖**全库**节点（v1.27 修；v1.28 加缓存）。

        旧实现用 `store.query_nodes(limit=80)` 取参考集，而该查询按
        `importance DESC, last_access DESC` 排序（core.py:639），
        参考集于是只是「最重要的 80 条」；库里更老的既有知识对它不可见，
        同一句已存在的内容会被判为新知识（重复入库、importance 倒挂）。
        参考集是硬判依据（NOVEL_TRIGGER=0.75），不能做排名采样。

        lingshu #389：v1.27 起每次调用都把全库正文重新切成 3/4 字片段
        （prefeed 一次调用算两遍：_novelty + evaluate→_novelty），库里有一篇
        长文档后单条短输入耗时数秒、峰值内存数百 MB。现按库签名
        （节点数 / max(rowid) / 正文内容指纹，见 `_known_snapshot`）缓存
        「每节点 gram 集合 + 词频 + 并集」，
        签名未变则直接复用，不再重切。**判别力不变**：仍是全库参考集、仍是
        硬判阈值；缓存的只是解析结果。**内存未改善**：缓存仍持有全库 gram
        （#389 期望的「成本与库总正文量无关」需倒排索引/增量表，属更大改动，
        本次不做）。

        exclude_id：排除被评估节点自身——让「已有节点重新评估」
        （promote_from_context / 重复快照）衡量的是相对**其它**节点的新信息。
        """
        union, counts, per_node, _ = self._known_snapshot()
        if exclude_id is None:
            return union
        skip = str(exclude_id)
        xg = per_node.get(skip)
        if not xg:
            return union
        # 仅剔除「只出现在被排除节点」的 gram（仍出现在其它节点的不算新）
        return {g for g in union if g not in xg or counts.get(g, 0) > 1}

    @staticmethod
    def _row_digest(content) -> int:
        """单行正文摘要（8 字节 blake2b → 整数）——#770② 的「内容维」。

        旧签名第三项是 `SUM(LENGTH(content))`：**等长原地改写**（`nodes.content`
        被原地 UPDATE 成**同样长度**的另一段正文）下三项逐位不变 ⇒ 缓存被判有效、
        复用陈旧 gram 表（新正文 0/21 进并集、旧正文 21/21 留存，实测签名
        `(1,1,13)`→`(1,1,13)`）。长度这一维无法区分「同长度的两段不同正文」，
        故补逐行**内容摘要**。
        非 str 正文（历史坏行 / 非文本列）按 `str(...)` 归一后取摘要——与
        `add_perception` 的 str 写入面口径一致，且不让一行坏正文打断缓存重建。
        """
        text = content if isinstance(content, str) else str(content or "")
        return int.from_bytes(
            hashlib.blake2b(text.encode("utf-8", "replace"),
                            digest_size=8).digest(), "big")

    def _known_snapshot(self):
        """按库签名缓存全库核心词表，返回 (union, counts, per_node, sig)。

        签名 = (节点数, max(rowid), 正文内容指纹)。签名未变 → 直接复用；
        **仅追加**（旧行逐行摘要逐位不变、且新增行 rowid 全大于旧 max）→ 只切
        新增行并增量并入（#389 的实际工作负载是「每条外部输入都写一个节点」，
        全量重建会让缓存形同虚设）；其余情形（删除/改写/清空）→ 全量重建。

        #770②：签名第三项由 `SUM(LENGTH(content))` 改为**正文内容指纹**
        （逐行 `(rowid, 摘要)` 折叠成 8 字节整数）——长度对**等长原地改写**不
        敏感（改前实测 `(1,1,13)`→`(1,1,13)` 逐位不变 ⇒ 直接命中「签名相等 ⇒
        复用缓存」这一支），内容指纹对任何正文字节变化都变。返回值形状仍是
        `(int, int, int)` 三元组（调用方与既有测试按三元组解包，形制不变）。
        #770② 把「签名相等」的有效性判据换成了内容指纹；增量分支的三条「更多」
        门槛（节点数增 ∧ max(rowid) 增 ∧ 正文总长增，总长另存于
        `_known_total_len`）逐字保留。

        #462（增量分支补「无删除」证明）：三条门槛只证明总量变大，**不证明没有
        删除**——删 1 增 2 时三项全增，旧实现直接增量并入，被删节点的 core grams
        永远留在参考集 ⇒ 被遗忘内容再出现时 `_novelty` 算 1.0→0.0、`evaluate`
        从 knowledge 掉到 context。现要求两道证明同时成立才增量：
          ① 计数平衡：`old_cnt + len(rowid > old_max 的行) == 现节点数`。同 id
             覆盖写（INSERT OR REPLACE 换 rowid）也会使该式不成立 ⇒ 全量重建；
          ② 旧 id 一个不少：`per_node` 的键集合 ⊆ 本次扫描的存活 id 集合。删除
             最大 rowid 的行后 SQLite 会复用该 rowid，①可被误判通过，②兜住。
        任一不成立 ⇒ 落到下方全量重建（缓存被整条替换，不留陈旧 gram）。

        已知残留（如实标注 · 未覆盖面）：**等长/变长的原地新增式改写 ∧ 同期追加
        节点**（`UPDATE nodes SET content=...` 保持同 rowid，同时插入新行把三项
        门槛抬高）——①计数平衡与②id 存在性都成立，仍会走增量、复用被改写行的
        陈旧 gram。检测它需要逐行比对旧行摘要（旧缓存未存每行摘要），属另一次
        改动面；本件不夹带。
        代价（如实标注）：判缓存有效性需逐行读 `rowid, id, content`（游标**惰性
        迭代**，不 fetchall）并逐行取摘要 —— 相比旧实现的一条聚合 SQL，每次
        调用多一趟 O(全库正文) 顺序读 + 摘要；省下的是 `_core_grams` 的正则切分
        与 gram 集合构造（#389 的主要成本，实测单条短输入数秒）。这是「签名须对
        等长改写敏感」的必然代价：SQLite 无内容哈希聚合函数，不读正文就无法
        区分同长度的两段不同正文。
        """
        h = hashlib.blake2b(digest_size=8)
        total_len = 0
        cnt = 0
        max_rid = 0
        # #462：同趟顺带收集存活 id 集合——增量分支用它证明「旧节点一个没少」
        # （见下方 ②）。不额外扫表，不改变签名口径。
        live_ids = set()
        for rid, nid, content in self.engine.store.conn.execute(
                "SELECT rowid, id, content FROM nodes ORDER BY rowid"):
            rid = int(rid)
            cnt += 1
            max_rid = rid
            live_ids.add(str(nid))
            total_len += len(content) if isinstance(content, str) else 0
            h.update(b"%d:%d;" % (rid, self._row_digest(content)))
        sig = (cnt, max_rid, int.from_bytes(h.digest(), "big"))
        cache = self._known_cache
        if cache is not None and self._known_sig == sig:
            return cache["union"], cache["counts"], cache["per_node"], sig
        if cache is not None:
            # 增量判定：三条「更多」门槛（节点数增 ∧ max(rowid) 增 ∧ 正文总长增）
            # ＋ #462 的「无删除」证明才可增量。
            old_cnt, old_max = self._known_sig[0], self._known_sig[1]
            old_len = self._known_total_len
            if sig[0] > old_cnt and sig[1] > old_max and total_len > old_len:
                rows = self.engine.store.conn.execute(
                    "SELECT id, content FROM nodes WHERE rowid > ?", (old_max,)
                ).fetchall()
                # #462：三项「更多」只证明总量变大，**不证明没有删除**——删 1 增 ≥2
                # 时节点数照样增加，被删节点不在 `rowid > old_max` 的新行里，增量
                # 并入会把它的核心词永久留在参考集（新奇度算 0，写入层级掉档）。
                # 两道证明都通过才允许增量：
                #   ① 计数平衡：旧节点数 + 新增行数 == 现节点数。同 id 覆盖写
                #      （INSERT OR REPLACE 换 rowid）会使该式不成立 ⇒ 全量重建；
                #   ② 旧 id 一个不少：`per_node` 的键集合 ⊆ 现存 id。删除最大
                #      rowid 的行后，SQLite 会复用该 rowid，①会误判通过，②兜住。
                if (old_cnt + len(rows) == sig[0]
                        and set(cache["per_node"]) <= live_ids):
                    union, counts, per_node = (cache["union"], cache["counts"],
                                               cache["per_node"])
                    for nid, content in rows:
                        gs = self._core_grams(content)
                        per_node[str(nid)] = gs
                        union |= gs
                        for g in gs:
                            counts[g] = counts.get(g, 0) + 1
                    self._known_sig = sig
                    self._known_total_len = total_len
                    return union, counts, per_node, sig
        rows = self.engine.store.conn.execute(
            "SELECT id, content FROM nodes").fetchall()
        union, counts, per_node = set(), {}, {}
        for nid, content in rows:
            gs = self._core_grams(content)
            per_node[str(nid)] = gs
            union |= gs
            for g in gs:
                counts[g] = counts.get(g, 0) + 1
        self._known_cache = {"union": union, "counts": counts, "per_node": per_node}
        self._known_sig = sig
        self._known_total_len = total_len
        return union, counts, per_node, sig

    @staticmethod
    def _template_of(text: str):
        """把 ASCII token 抹成占位符 ⇒ (句型模板, token 集合)。#573 / #213 建议2。

        模板 = 正文里每个「ASCII 标识符整体」（数字/主机名/域名/IP/CVE 号/账号…）
        被替换成 `\\x00` 之后的字符串；token 集合 = 被抹掉的那些 token。

        「同模板、不同 token」⇒ 内容确实变了（高新奇）；「同模板、同 token」⇒ 重复。
        这是 `_core_grams` 比例判据的补充：比例判据里新增/消失的 gram 会相互抵消，
        句越长上限越低（实测 25 字句上限仅 6/25=0.24，远低于 0.75 阈值）。
        """
        import re as _re
        if LongTermMemoryGate._ASCII_TOKEN is None:
            LongTermMemoryGate._ASCII_TOKEN = _re.compile(
                r'[A-Za-z0-9][A-Za-z0-9_.\-:/@]*')
        toks = []

        def _sub(m):
            tok = m.group(0)
            if len(tok) >= 2:        # 与 _core_grams 同口径：单字符 token 是噪音
                toks.append(tok)
                return "\x00"
            return tok

        tpl = LongTermMemoryGate._ASCII_TOKEN.sub(_sub, text or "")
        return tpl, frozenset(toks)

    def _known_templates(self, exclude_id: str = None) -> dict:
        """已知内容的「句型模板 → token 集合的集合」索引（带签名缓存）。"""
        sig = self._known_sig
        cache = getattr(self, "_tpl_cache", None)
        if (cache is not None and sig is not None
                and getattr(self, "_tpl_sig", None) == sig):
            idx = cache
        else:
            rows = self.engine.store.conn.execute(
                "SELECT id, content FROM nodes").fetchall()
            idx = {}
            for nid, content in rows:
                tpl, toks = LongTermMemoryGate._template_of(content)
                idx.setdefault(tpl, set()).add(toks)
            self._tpl_cache = idx
            self._tpl_sig = sig
        if exclude_id:
            # 排除自身节点：其模板/token 不应让「自己」显得已知
            node = self.engine.store.get_node(exclude_id)
            if node is not None:
                tpl, toks = LongTermMemoryGate._template_of(node.content)
                seen = idx.get(tpl)
                if seen is not None and toks in seen:
                    idx = dict(idx)
                    rest = set(seen)
                    rest.discard(toks)
                    if rest:
                        idx[tpl] = rest
                    else:
                        idx.pop(tpl, None)
        return idx

    def _novelty(self, content: str, existing_id: str = None) -> float:
        """新信息度（v1.27 改：参考集覆盖全库，并真正使用 existing_id）。

        海马体识别的是「新信息成分」——句子里有多少**核心词**是库里没见过的。
        v1.15 起用核心词新颖比例（非整句相似度），但参考集取的是排名前 80 条，
        「库里的老知识」对它不可见：同一内容只改写入先后/importance，
        新奇度就在 1.0 与 0.0 两端跳变。现按全库统计，且 existing_id 生效。

        #573（#213 建议2 落地）：**比例判据在中等长度句子上有系统性上限**——
        同模板只换一个 ASCII token 时，新增 gram 与消失 gram 数量相等，
        比例上相互抵消，且上限随句长下降（实测 25 字句仅 6/25=0.24）。
        故在比例之外**并列一条模板判据**（任一命中即判变更）：
          · 模板命中已知模板，但 **token 集合不同** ⇒ 内容确实变了 ⇒ 取 1.0；
          · 模板命中且 token 集合相同 ⇒ 真重复 ⇒ 保持比例值（会被判低）；
          · 模板未命中（新句型）⇒ 保持比例值（比例判据本就够用）。
        这样既修好「换数字/账号/域名/IP」这一族，又不动「真重复」与「全新句型」
        的既有语义（比例判据仍是这两类的唯一依据）。
        判据来源：issue #573 正文 + `_core_grams` docstring 的「未修」条 + #213 §六·建议2。
        削掉的判别力：模板相同而 token 不同时**不再看比例**（一律 1.0）——
        这是有意的：该族的比例上限由句长决定、与改动重要性无关，继续用比例就是错的。
        """
        try:
            grams = self._core_grams(content)
            if not grams:
                return 0.5
            known = self._known_grams(exclude_id=existing_id)
            if not known:
                return 0.5  # 无参照：中性
            novel_grams = sum(1 for g in grams if g not in known)
            ratio = novel_grams / max(1, len(grams))
            ratio = max(0.0, min(1.0, ratio))
            # #573：模板判据（同模板换 token ⇒ 变更）
            try:
                tpl, toks = LongTermMemoryGate._template_of(content)
                seen = self._known_templates(exclude_id=existing_id).get(tpl)
                if seen is not None and toks not in seen:
                    return 1.0
            except Exception:
                pass                      # 模板判据失败不得影响主判据
            return ratio
        except Exception:
            pass
        return 0.5  # 无参照：中性

    def _mention(self, existing_id: str = None) -> int:
        """提及次数：已有节点的 access_count；新节点 0。"""
        if not existing_id:
            return 0
        node = self.engine.store.get_node(existing_id)
        return node.access_count if node else 0

    # ---- 评估与决策 ----

    def evaluate(self, content: str, source: str = "snapshot",
                 tags=None, existing_id: str = None,
                 novelty: float = None) -> dict:
        """快照评估：特征 → 评分 → 层级决策。

        novelty：调用方已算出的新奇度（如 prefeed 已算过）。给了就复用，
        避免对同一输入重复全库扫描（lingshu #389）。语义与内部计算一致。

        #501 如实标注（审计面，判据不变）：`source` 与 `tags` 形参当前**未参与
        评分**（保留以维持调用方契约）；`trust` 分量取 `self_model.trust_state`
        的**会话级全局信任**，不是本次快照来源（user/assistant/external/…）的
        可信度。把来源真正接进评分属设计裁定（改变落层行为），不在本标注内。
        """
        if novelty is None:
            novelty = self._novelty(content, existing_id)
        trust = self._trust()
        d2 = self._d_second()
        t2 = self._trust_second()
        n = self._mention(existing_id)
        w = self.weights
        imp = (w["novelty"] * novelty + w["trust"] * trust
               + w["d2"] * d2 + w["t2"] * t2
               + w["mention"] * min(1.0, 0.15 * (n + 1) / (n + 2)))  # log 平滑近似
        imp = max(0.0, min(1.0, imp))
        if imp >= self.LONG_TERM_THRESHOLD:
            layer = "long_term"
        elif imp >= self.KNOWLEDGE_THRESHOLD:
            layer = "knowledge"
        else:
            layer = "context"
        return {
            "importance": round(imp, 3),
            "layer": layer,
            "features": {
                "novelty": round(novelty, 3),
                "trust": round(trust, 3),
                "d2": round(d2, 3),
                "t2": round(t2, 3),
                "mention": n,
            },
            "decision": (f"长期记忆（imp={imp:.2f}≥{self.LONG_TERM_THRESHOLD}）"
                         if layer == "long_term" else
                         f"知识层（imp={imp:.2f}）" if layer == "knowledge" else
                         f"情境层（imp={imp:.2f}，可提升）"),
        }

    # 来源类保留标签（lingshu #217）：来源只能由创建时的可信路径声明，
    # 不得由后续快照并入**既有**节点（否则 external 知识被洗成 user）。
    SOURCE_RESERVED_TAGS = frozenset(
        {"user", "assistant", "external", "system", "tool", "fixture"})
    SOURCE_RESERVED_PREFIXES = ("session:", "role:")

    @classmethod
    def _strip_source_tags(cls, tags) -> list:
        """剔除来源类保留标签（用于「已存在 → 更新」分支的标签合并）。"""
        out = []
        for t in (tags or []):
            low = str(t).lower()
            if low in cls.SOURCE_RESERVED_TAGS or \
                    low.startswith(cls.SOURCE_RESERVED_PREFIXES):
                continue
            out.append(t)
        return out

    def write_snapshot(self, content: str, source: str = "snapshot",
                       tags: list = None, entities: list = None,
                       importance_hint: float = None) -> dict:
        """快照写入：评估 → 按层级写入（含条件空间/关联/保护）。

        返回的 `layer` = **实际落库层**（读回 DB），`evaluated_layer` = 决策层；
        二者不一致的旧行为已修（lingshu #224）。

        #292 追加变体：新增结果键 `write_failed`（成功路径 `False`）与
        `write_error`（成功路径 `None`）——「已存在」腿的层级/重要度 `UPDATE`
        失败时不再与成功路径同值（旧实现 try 尾与 except 各写一遍
        `node_id = existing_id`，异常在返回值上不可观测）。既有键的语义与取值
        形态一字未改。

        #770①③（并发面）：把「存在性检查 → 读既有节点 → 写」这一段收进**单一写
        事务**（复用仓内既有 `LayeredStore._tx()`，与 #344 的 `tag_node` /
        #517 的 `delete_node` 同一原语，不另造第二套）——`BEGIN IMMEDIATE` 先取
        RESERVED 写锁，故：
        ① 「已存在」腿读到的 `node.tags` 与随后的整列 `UPDATE tags` 在同一写锁下
           （改前两进程各自读到旧 `tags`、后写者用旧值覆盖先写者 ⇒ 整组 worker 的
           标签静默消失；4 进程×6 次实测丢 18/12/12，串行对照 24/24）；
        ③ 存在性判定与 `skip_dedup=True` 新建腿**互斥**（同一把写锁串行化
           「判定 → 新建」）——改前 40 轮×4 进程并发首写同一内容，40/40 轮出重复行。
        ★事务**只覆盖判定复核与写**（`_snapshot_write`）；`search_content` 的存在性
        预判与 `evaluate()` 留在事务**之外**（`_snapshot_probe`，只读）。为什么必须
        这样切：`search_content` 的命中计数腿走 `increment_access_many`——它**自开
        一条独立连接**（`sqlite3.connect(..., timeout=0)`）写回 `access_count`；
        若把它放进本事务，那条旁路写与本次 `BEGIN IMMEDIATE` **自冲突**，并把
        「另一进程 busy 等待持 PENDING」与「本进程 commit 需 EXCLUSIVE」的互等窗口
        放大到 `busy_timeout`（实测 4 进程×6 次出现 30s 级
        `OperationalError: database is locked`；把该旁路置 no-op 后同形态 1.26s
        全绿）。故存在性判定分两段：事务外**预判**（判据与改前逐字相同，且保住
        「检索命中刷新访问计数」这一既有副作用），事务内**复核**（纯读、零旁路写）。
        边界（如实标注）：①本件只保证**本门控这条路径**的读改写原子性——其它路径
        对同一节点的并发读改写（如 `promote_from_context`、引擎侧裸 SQL）不归本件
        （`core.py` 的 AST 事务守卫亦**只扫 `core.py`**，本文件方法不在其扫描面内）；
        ②事务边界由 `_tx()` 的嵌套感知兜住（`_tx_depth>1` 时发 SAVEPOINT、提交权归
        最外层），故被已在事务内的调用方调进来时不会踩
        `cannot start a transaction within a transaction`；
        ③事务内的复核判据是「预判命中的节点仍在、且正文 `char_bigram_jaccard` ≥ 0.95」
        ＋「正文**精确等值**兜底」——前者是 LIKE 路径的同款相似度口径（预判已判过
        一次，复核只确认没被并发改写）；若预判是**靠 tag_bonus 加成**越阈（jaccard
        在 0.90~0.95 之间）或走 FTS 混合分路径，复核会判否并退回精确等值 ⇒ 极端
        情形下从「提升既有」退化为「新建」，此差异已记入未覆盖面；
        ④长期层的相似边建边（`_snapshot_link`）移到**事务提交之后**——它同样调
        `search_content`（同一旁路写），留在事务内会重现上面的自冲突。
        """
        probe = self._snapshot_probe(content, source, tags)
        store = self.engine.store
        with store._tx():
            result = self._snapshot_write(content, source, tags, entities,
                                          importance_hint, probe)
        if result.pop("_need_links", False):
            # 事务外建边（见边界④）；失败与改前同值（`links=0`），不抛出。
            result["links"] = self._snapshot_link(result.get("node_id"), content)
        divergent_id = result.pop("_divergent_id", None)
        if divergent_id:
            # issue #481：与判为实质分歧的既有节点登记 OPPOSITE 边（未复核
            # 0.3）——放在事务外，避免 `add_edge` 的 `_commit_locked()` 提前
            # 终结外层写事务（与 `_snapshot_link` 同一原因，见边界④）。
            try:
                self.engine.register_conflict(result.get("node_id"), divergent_id)
                result["conflict_registered"] = True
            except Exception as exc:
                result["conflict_registered"] = False
                _log.warning(
                    "write_snapshot 分歧边登记失败（更正节点已落库，冲突待复核"
                    " 面缺失）：%s: %s", type(exc).__name__, exc)
        return result

    def _snapshot_probe(self, content: str, source: str, tags: list) -> dict:
        """事务外**只读**预判：存在性检查（`search_content`）+ 评估（`evaluate`）。

        判据与改前 `write_snapshot` 逐字相同（含「检索命中刷新访问计数」的既有
        副作用）；本函数**不**写库、**不**开事务。事务内由 `_recheck_existing`
        复核其结论（见 `write_snapshot` 边界③）。
        """
        engine = self.engine
        from .core import LayeredStore, MemoryLayer
        existing_id = None
        divergent_id = None
        try:
            hits = engine.store.search_content(
                content, layers=[MemoryLayer.KNOWLEDGE, MemoryLayer.CONTEXT], limit=1)
            if hits:
                cand = hits[0][0]
                # issue #481：存在性判据按**正文**（不计 tag_bonus）复核，且
                # 要求无实质分歧（否定/数值）——旧实现只看 search_content 的
                # 合成分 ≥0.95（含 0.05 tag_bonus）：差一个「不」或一个数字
                # 即被判为同一内容 ⇒ 更正被丢弃、错误节点被提权到 0.9 并
                # protect+no_forget（永久置顶召回）。有分歧另立节点并挂
                # OPPOSITE 边待复核，而不是把更正并进错误节点。
                sim = LayeredStore.char_bigram_jaccard(content, cand.content or "")
                if sim >= 0.95:
                    if engine._assertion_divergence(content, cand.content or ""):
                        divergent_id = cand.id
                    else:
                        existing_id = cand.id
        except Exception:
            pass
        return {"existing_id": existing_id, "divergent_id": divergent_id,
                "ev": self.evaluate(content, source, tags, existing_id)}

    def _recheck_existing(self, candidate: str, content: str):
        """事务内复核预判结论（**纯读**，零 `search_content`／零旁路写）。

        ① 预判命中的 `candidate` 仍在、且正文仍近乎相同（`char_bigram_jaccard`
           ≥ 0.95）⇒ 沿用——保证「读 tags → 写 tags」落在同一写锁下（#770①）；
        ② 否则退回「正文**精确等值**」检查（限调用方可写层，与 #217 同口径）——
           并发首写同一内容时两进程正文逐字相同 ⇒ 后进事务者必命中先写者的行
           （#770③ 的互斥面）。
        """
        from .core import LayeredStore, MemoryLayer
        store = self.engine.store
        if candidate:
            node = store.get_node(candidate)
            if node is not None and LayeredStore.char_bigram_jaccard(
                    content, node.content or "") >= 0.95:
                # issue #481：预判到事务之间候选可能被并发改写成实质分歧
                # （否定/数值），此时不得沿用——否则更正仍会被并进错误节点。
                if not self.engine._assertion_divergence(content, node.content or ""):
                    return candidate
        row = store.conn.execute(
            "SELECT id FROM nodes WHERE content=? AND layer IN (?,?) "
            "ORDER BY rowid LIMIT 1",
            (content, MemoryLayer.KNOWLEDGE.value,
             MemoryLayer.CONTEXT.value)).fetchone()
        return row[0] if row else None

    def _snapshot_link(self, node_id, content: str) -> int:
        """长期层关联边（事务外调用）：与最相似知识节点建 similar 边，返回建边数。

        改前该段内联在 `write_snapshot` 里；#770 把它移出写事务（见 `write_snapshot`
        边界④）。异常语义与改前同值：整体失败 ⇒ `0`，不抛出。
        """
        try:
            from .core import EdgeType
            links = self.engine.store.search_content(content, limit=3)
            for other, sim in links:
                if other.id != node_id and sim >= 0.25:
                    self.engine.add_edge(node_id, other.id,
                                         relation_type=EdgeType.SIMILAR,
                                         source_evidence="inferred")
            return len([x for x in links if x[0].id != node_id])
        except Exception:
            return 0

    def _snapshot_write(self, content: str, source: str, tags: list,
                        entities: list, importance_hint: float,
                        probe: dict) -> dict:
        """`write_snapshot` 的实现体（调用方已持有写事务，见 `write_snapshot`）。

        本体内**不得**再开事务，也不得直接 `conn.commit()`——提交一律走
        `store._commit_locked()`（#665：嵌套层是 no-op，只有最外层真提交）；
        也不得调用 `search_content`（其旁路写会与本事务自冲突，见 `write_snapshot`）。
        """
        engine = self.engine
        from .core import MemoryLayer, _finite_clamp
        ev = probe["ev"]
        # 已存在性检查（同内容重复快照 → 提升而非新建）。
        # #217：限定在调用方可写的本地层——不限层时会命中锚点/结构/自我层，
        # 后续裸 SQL 更新绕过 add_node 的 IMMUTABLE_LAYERS + Role 校验，
        # 使 SUB 只凭一段内容片段即可改写共享层节点（并登记永久保护）。
        # #770③：事务内复核（预判命中的节点仍在 + 精确等值兜底）。
        existing_id = self._recheck_existing(probe["existing_id"], content)
        # #292（复核者 123WP-a 追加的同族变体）：`except` 分支**不得**与成功路径
        # 同值——旧实现里层级/重要度提升的 `UPDATE` 失败被吞，`node_id` 仍赋
        # `existing_id`（与上一行完全相同的赋值），异常在返回值上**不可观测**。
        # 现加结果键 `write_failed` / `write_error`：成功路径 `False` / `None`。
        # 既有键（status / layer / importance / node_id …）的语义与取值形态不变。
        write_failed = False
        write_error = None
        # #587：值域闸**单点收口**——本门控有两处**裸 SQL**写 importance
        # （下方「已存在」分支与 `promote_from_context`），不经 `STNode.to_row`
        # 的 `_finite_clamp`（#164/#185 的钳制只挂在 `add_*` 腿）。旧实现
        # `importance_hint=1e6/inf` 原样落库 ⇒ `inf × decay = inf` 永生、
        # `ORDER BY importance DESC` 检索恒占首位、挤占 LIMIT 窗口，且
        # `imp ≥ LONG_TERM_THRESHOLD` 直接触发 `protect_node` 永久保护。
        # 现按 #185 口径在**入口**归一（与 `add_node` 的
        # `_finite_clamp(x, 0.0, 1.0, default=0.5)` 同向）：有限值钳到 [0,1]，
        # 非有限（NaN/±inf）回落 `default`；此处 default 取**评估值**
        # `ev["importance"]`——提示值坏掉时退回本门控自己的评分，而不是凭空
        # 造一个 0.0/0.5。阈值本体未改。
        imp = importance_hint if importance_hint is not None else ev["importance"]
        imp = _finite_clamp(imp, 0.0, 1.0, default=ev["importance"])
        # 层级决策与最终 importance 统一（显式提示同样参与层级判定）
        if imp >= self.LONG_TERM_THRESHOLD:
            decided = "long_term"
        elif imp >= self.KNOWLEDGE_THRESHOLD:
            decided = "knowledge"
        else:
            decided = "context"

        # 情境层（<0.4）且无显式提示：快照不落库（返回评估，防低价值快照污染）。
        if decided == "context" and importance_hint is None:
            return {"status": "discarded", "importance": round(imp, 3),
                    "layer": "context", "features": ev["features"],
                    "decision": ev["decision"],
                    "write_failed": False, "write_error": None}

        node_id = None
        if existing_id:
            node = engine.store.get_node(existing_id)
            if node is None or node.layer in engine.store.IMMUTABLE_LAYERS:
                existing_id = None      # 不可写共享层 → 改为新建（#217）
            else:
                try:
                    import json as _json
                    tags_merged = list(dict.fromkeys(
                        (node.tags or []) + self._strip_source_tags(tags)))
                    # #224：命中情境层节点且判定为知识/长期层 → 同步提升层级，
                    # 否则节点留在情境层被 decay 删除，而返回值却报 long_term。
                    new_layer = (MemoryLayer.KNOWLEDGE.value
                                 if decided in ("knowledge", "long_term")
                                 else node.layer.value)
                    engine.store.conn.execute(
                        "UPDATE nodes SET importance=?, tags=?, confidence=?, layer=? "
                        "WHERE id=?",
                        # #672：`node.importance` 是**库内读回值**（存量行可能越界，
                        # 如 1e6/inf）——`max` 会让它原样写回，绕过上面入口处的钳制。
                        # 与姊妹腿 `promote_from_context` 对称，落库前过同一口径。
                        # default 取 `imp`（入口已归一），以保住 #587 守卫③
                        # 「合法 hint 的 max 合并语义不动」。
                    (_finite_clamp(max(node.importance, imp), 0.0, 1.0,
                                   default=imp),
                         _json.dumps(tags_merged, ensure_ascii=False),
                         node.confidence, new_layer, existing_id))
                    # #770①：提交走 `_commit_locked()`（不是裸 `conn.commit()`）
                    # ——本腿现由 `write_snapshot` 包在 `_tx()` 里，裸提交会提前
                    # 终结外层事务（#665 同款：嵌套层提交权不归本层）。
                    engine.store._commit_locked()
                    node_id = existing_id
                except Exception as exc:
                    node_id = existing_id
                    write_failed = True
                    write_error = type(exc).__name__
                    _log.warning(
                        "write_snapshot 层级/重要度更新失败（既有节点未更新，"
                        "回报仍指向既有 node_id=%s）：%s: %s",
                        existing_id, type(exc).__name__, exc)

        if node_id is None:
            if decided == "context":
                # #214 入口 / #224：判为情境层 → 写入情境层（对齐文件头
                # 「imp < 0.4 → 情境层（短期，睡眠巩固时再评估提升）」）。
                # 旧实现走 add_perception 落 KNOWLEDGE 层，返回值却报 context。
                node = engine.add_context(content, importance=imp,
                                          tags=(tags or []) + ["gate"])
                node_id = getattr(node, "id", None) or node
            else:
                # v1.26c（外部测试 v3-P2）：skip_dedup——主动沉淀（剧情/快照）
                # 必须独立成节点。之前 add_perception 的 M5 去重把剧情内容合并进
                # 刚写入的相似对话节点（只加 duplicate 标签），剧情标签/高
                # importance 丢失 → 剧情连续性失效（plot 节点写了个寂寞）。
                node = engine.add_perception(
                    content, importance=imp, tags=(tags or []) + ["gate"],
                    entities=entities or None, skip_dedup=True)
                node_id = getattr(node, "id", None) or node

        # 回报 = 事实：layer 取回读的实写层（#224）
        stored = engine.store.get_node(node_id)
        stored_layer = stored.layer.value if stored is not None else decided
        result = {"node_id": node_id, "importance": round(imp, 3),
                  "layer": stored_layer, "evaluated_layer": decided,
                  "features": ev["features"],
                  "write_failed": write_failed, "write_error": write_error}
        # issue #481：新建腿若预判到「实质分歧」的既有节点，把其 id 透传给
        # `write_snapshot`，由事务**提交后**登记 OPPOSITE 边（add_edge 自带
        # commit，不能留在本事务内，见 write_snapshot 边界④）。
        if existing_id is None and probe.get("divergent_id"):
            result["_divergent_id"] = probe["divergent_id"]
        if decided == "long_term":
            try:
                engine.protect_node(node_id, f"LongTermMemoryGate:{source}")
                result["protected"] = True
            except Exception:
                result["protected"] = False
            # #770：关联边（与最相似知识节点建 similar 边）移到**事务提交之后**
            # ——该段调 `search_content`，其命中计数腿自开独立连接写回，留在本事务
            # 内会与 `BEGIN IMMEDIATE` 自冲突（见 `write_snapshot` 边界④）。
            # `_need_links` 由 `write_snapshot` 消费后从结果里弹出。
            result["_need_links"] = True
        return result

    # ---- 前馈新奇检测（H1 · 海马体学习：新颖→当场强化编码） ----

    NOVEL_TRIGGER = 0.75   # 新奇度阈值：1-相似度 ≥ 0.75 → 判定「新东西」
    NOVEL_BOOST = 0.15     # 新奇输入 importance 提升
    NOVEL_EDGE_SIM = 0.25  # 与相关知识的建边最低相似度

    def prefeed(self, content: str, source: str = "input",
                tags: list = None, entities: list = None) -> dict:
        """海马体式前馈：输入到来时先检测新奇度，高新奇 → 当场强化编码。

        返回 {novel, novelty, action, node_id, importance, links, degraded}
        - novel=True / action='prefeed_boost'：触发了强化编码（标记 novel_prefeed
          + importance 提升 + 建边）；`degraded` 列出**实际失败**的降级面
          （'links' / 'protect'），无降级时为 `[]`。
        - novel=False / action='routine'：常规路径（novelty 未达阈值，不干预）。
        - novel=False / action='prefeed_failed'：主写入失败（fail-closed，附
          `error` = 异常类名）；`node_id=None`、`importance=0.0`、`links=0`
          ——**不**回报一个并未生效的成功语义（#292 追加变体）。
        """
        engine = self.engine
        try:
            novelty = self._novelty(content)
        except Exception:
            novelty = 0.5
        if novelty < self.NOVEL_TRIGGER:
            # #292：`degraded` 键**恒在场**（本路径无降级面 ⇒ `[]`），使 docstring
            # 承诺的键集在三条返回路径上一致——调用方读 `r["degraded"]` 不会
            # 因走了哪条路径而 KeyError。
            return {"novel": False, "novelty": round(novelty, 3),
                    "action": "routine", "degraded": []}

        # 高新奇 → 强化编码：importance 提升 + 标签 + 建边
        # novelty 复用上面已算出的值（#389：避免对同一输入重复全库扫描）
        base_imp = self.evaluate(content, source, tags, novelty=novelty).get(
            "importance", 0.5)
        imp = min(1.0, base_imp + self.NOVEL_BOOST)
        tags_all = list(dict.fromkeys((tags or []) + ["novel_prefeed", "gate"]))
        node_id = None
        links = 0
        actual_imp = None
        # #292（复核者 123WP-a 追加的同族变体）：**失败必须在返回值上可观测**。
        # 旧实现主写入（add_perception）抛异常被宽 `except: pass` 吞掉后，
        # `node_id` 停在 None 却仍回报 `novel=True / action='prefeed_boost'`
        # ——0 个节点写入却回报成功（与同文件 `write_snapshot` 的
        # 「回报 ≠ 事实」纪律同族，前一处按该纪律修过、后一处漏改）。
        # 现：主写入失败 ⇒ fail-closed 回报 `prefeed_failed`；降级面（建边 /
        # protect_node）失败 ⇒ 节点已写入仍算成功，但失败面逐项登记进
        # `degraded`（#170 口径：宽 except 必须留痕或收窄）。
        degraded = []
        try:
            # #292：主动沉淀（前馈强化编码）必须独立成节点——不带 skip_dedup 时
            # M5 去重会把新输入并入相似旧节点（只加 duplicate 标签），回执却报
            # prefeed_boost/importance，新正文根本没入库（与 write_snapshot
            # v1.26c 的同一失败模式，此处漏改）。
            node = engine.add_perception(
                content, importance=imp, tags=tags_all, entities=entities or None,
                skip_dedup=True)
            node_id = getattr(node, "id", None)
            # 去重命中时 add_perception 返回的是**既有**节点：回报实际落库值，
            # 不要回报一个并未生效的计算值（"回报 ≠ 事实"）。
            actual_imp = getattr(node, "importance", None)
        except Exception as exc:
            _log.warning("prefeed 主写入失败（fail-closed，不报 boost）：%s: %s",
                         type(exc).__name__, exc)
            return {"novel": False, "novelty": round(novelty, 3),
                    "action": "prefeed_failed", "node_id": None,
                    "importance": 0.0, "links": 0,
                    "error": type(exc).__name__, "degraded": degraded}
        # 与相关知识建边（信息差驱动的关联）——降级面：失败留痕，不吞。
        try:
            from .core import EdgeType
            rel = engine.store.search_content(content, limit=3)
            for other, sim in rel:
                if other.id != node_id and sim >= self.NOVEL_EDGE_SIM:
                    engine.add_edge(node_id, other.id,
                                    relation_type=EdgeType.SIMILAR,
                                    source_evidence="inferred")
                    links += 1
        except Exception as exc:
            degraded.append("links")
            _log.warning("prefeed 建边降级（节点已写入）：%s: %s",
                         type(exc).__name__, exc)
        # 长期层 → 保护——同为降级面：失败留痕。
        if imp >= self.LONG_TERM_THRESHOLD:
            try:
                engine.protect_node(node_id, f"Prefeed:{source}")
            except Exception as exc:
                degraded.append("protect")
                _log.warning("prefeed 保护登记降级（节点已写入）：%s: %s",
                             type(exc).__name__, exc)
        return {"novel": True, "novelty": round(novelty, 3),
                "action": "prefeed_boost", "node_id": node_id,
                "importance": round(float(actual_imp if actual_imp is not None
                                          else imp), 3),
                "links": links, "degraded": degraded}

    def promote_from_context(self, limit: int = 30) -> list:
        """情境层批量提升扫描（睡眠巩固/会话结束时调用）：
        情境节点重新评估，够格者提升到知识层。

        lingshu #388：旧实现 `query_nodes(layer=CONTEXT, limit=limit)` 只取
        importance 前 limit 条，而该查询 ORDER BY importance DESC——落选节点
        不改状态、不打标记，窗口恒定：只要有 ≥limit 条 importance 更高但
        不够格的节点占位，排在后面的够格新内容扫多少轮都不会被评估。
        现改为**全量评估、按提升数计 limit**（limit 是「本轮最多提升几条」，
        不是「最多看几条」），饥饿消除。判据来源：lingshu #388 §修复建议。
        成本上界：情境层 FIFO 上限默认 200 条（core.py:2116）。
        """
        engine = self.engine
        promoted = []
        try:
            from .core import MemoryLayer, _finite_clamp
            nodes = engine.store.get_layer_nodes(MemoryLayer.CONTEXT)
            # dsh-bounty #1794（#665 纪律漏网点）：旧实现循环体内裸
            # `engine.store.conn.commit()`——它不是 `@_transactional` 方法、体内
            # 无 `_tx` 包装，`_commit_locked()` 在调用方外层事务（depth==1）内
            # 仍会真提交 ⇒ 调用方「先 promote 再失败回滚」时提升被固化成
            # 「回滚了但已生效」（假原子性）。修法对齐 `write_snapshot`：整段
            # 写入包进 `with store._tx():`——独立调用时 depth==1 真提交（行为
            # 不变）；被外层事务调进来时 depth==2 走 SAVEPOINT，提交权归外层。
            with engine.store._tx():
              for node in nodes:
                if len(promoted) >= limit:
                    break
                ev = self.evaluate(node.content, "promote", node.tags, node.id)
                if ev["layer"] in ("long_term", "knowledge"):
                    import json as _json
                    # #587：本腿同为**裸 SQL**——`ev["importance"]` 经 `evaluate`
                    # 已在 [0,1]，但 `node.importance` 是**库里读回值**，修复前
                    # 落库的越界行（1e6/inf）会经 `max` 再次写回 ⇒ 一并过同一
                    # 钳制（同 `write_snapshot` 入口口径，阈值未改）。
                    new_imp = _finite_clamp(
                        max(node.importance, ev["importance"]), 0.0, 1.0,
                        default=ev["importance"])
                    tags_new = list(dict.fromkeys((node.tags or []) + ["promoted"]))
                    engine.store.conn.execute(
                        "UPDATE nodes SET importance=?, layer=?, tags=? WHERE id=?",
                        (new_imp, MemoryLayer.KNOWLEDGE.value,
                         _json.dumps(tags_new, ensure_ascii=False), node.id))
                    # #665/#770①：提交走 `_commit_locked()`（本块已包在 `_tx()`
                    # 内：嵌套时 no-op、推迟到外层；独立调用时真提交）。
                    engine.store._commit_locked()
                    if ev["layer"] == "long_term":
                        try:
                            engine.protect_node(node.id, "LongTermMemoryGate:promote")
                        except Exception:
                            pass
                    promoted.append({"node_id": node.id,
                                     "importance": ev["importance"],
                                     "layer": ev["layer"]})
        except Exception:
            pass
        return promoted
