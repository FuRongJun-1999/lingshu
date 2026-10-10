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
import time


class LongTermMemoryGate:
    """长期记忆写入决策器（挂载于引擎，纯标准库）。"""

    # 默认权重（可配；场景化权重调整留作后续）
    DEFAULT_WEIGHTS = {
        "novelty": 0.30,   # 新信息度（核心词新颖比例；参考集 = 全库既有知识）
        "trust": 0.25,     # 信任（来源可信度）
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

    # ---- 特征提取 ----

    def _d_norm(self) -> float:
        """当前信息差 D_norm（0-1；无样本 0.5 中性）。"""
        hist = getattr(self.engine, "_gap_history", None) or []
        return hist[-1]["d_norm"] if hist else 0.5

    def _d_second(self) -> float:
        """信息差二阶变化 Δ²D：slope 的变化趋势（需 ≥3 样本）。
        正=信息差加速扩大（新领域涌现信号）；负=收敛。"""
        hist = getattr(self.engine, "_gap_history", None) or []
        if len(hist) < 3:
            return 0.0
        vals = [h["d_norm"] for h in hist[-5:]]
        d1 = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
        if not d1:
            return 0.0
        return max(-0.5, min(0.5, d1[-1] - d1[0]))  # 归一化钳制

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
        """信任二阶变化 Δ²T：trust_history 二阶差分（信任跃升信号）。"""
        th = getattr(getattr(self.engine, "self_model", None),
                     "trust_history", None) or []
        if len(th) < 3:
            return 0.0
        vals = [h["t_total"] for h in th[-6:]]
        d1 = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
        if not d1:
            return 0.0
        return max(-0.5, min(0.5, d1[-1] - d1[0]))

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
        （节点数 / max(rowid) / 正文总长）缓存「每节点 gram 集合 + 词频 + 并集」，
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

    def _known_snapshot(self):
        """按库签名缓存全库核心词表，返回 (union, counts, per_node, sig)。

        签名 = (节点数, max(rowid), 正文总长)。签名未变 → 直接复用；
        **仅追加**（节点数与总长增加、max(rowid) 增加）→ 只切新增行并增量并入
        （#389 的实际工作负载是「每条外部输入都写一个节点」，全量重建会让缓存
        形同虚设）；其余情形（删除/改写/清空）→ 全量重建。
        """
        row = self.engine.store.conn.execute(
            "SELECT count(*), COALESCE(MAX(rowid),0), "
            "COALESCE(SUM(LENGTH(content)),0) FROM nodes").fetchone()
        sig = (int(row[0]), int(row[1]), int(row[2]))
        cache = self._known_cache
        if cache is not None and self._known_sig == sig:
            return cache["union"], cache["counts"], cache["per_node"], sig
        if cache is not None:
            old_cnt, old_max, old_len = self._known_sig
            if sig[0] > old_cnt and sig[1] > old_max and sig[2] > old_len:
                rows = self.engine.store.conn.execute(
                    "SELECT id, content FROM nodes WHERE rowid > ?", (old_max,)
                ).fetchall()
                union, counts, per_node = (cache["union"], cache["counts"],
                                           cache["per_node"])
                for nid, content in rows:
                    gs = self._core_grams(content)
                    per_node[str(nid)] = gs
                    union |= gs
                    for g in gs:
                        counts[g] = counts.get(g, 0) + 1
                self._known_sig = sig
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
        return union, counts, per_node, sig

    def _novelty(self, content: str, existing_id: str = None) -> float:
        """新信息度（v1.27 改：参考集覆盖全库，并真正使用 existing_id）。

        海马体识别的是「新信息成分」——句子里有多少**核心词**是库里没见过的。
        v1.15 起用核心词新颖比例（非整句相似度），但参考集取的是排名前 80 条，
        「库里的老知识」对它不可见：同一内容只改写入先后/importance，
        新奇度就在 1.0 与 0.0 两端跳变。现按全库统计，且 existing_id 生效。
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
            return max(0.0, min(1.0, ratio))
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
        """
        engine = self.engine
        from .core import MemoryLayer
        # 已存在性检查（同内容重复快照 → 提升而非新建）。
        # #217：限定在调用方可写的本地层——不限层时会命中锚点/结构/自我层，
        # 后续裸 SQL 更新绕过 add_node 的 IMMUTABLE_LAYERS + Role 校验，
        # 使 SUB 只凭一段内容片段即可改写共享层节点（并登记永久保护）。
        existing_id = None
        try:
            hits = engine.store.search_content(
                content, layers=[MemoryLayer.KNOWLEDGE, MemoryLayer.CONTEXT], limit=1)
            if hits and hits[0][1] >= 0.95:
                existing_id = hits[0][0].id
        except Exception:
            pass

        ev = self.evaluate(content, source, tags, existing_id)
        imp = importance_hint if importance_hint is not None else ev["importance"]
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
                    "decision": ev["decision"]}

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
                        (max(node.importance, imp),
                         _json.dumps(tags_merged, ensure_ascii=False),
                         node.confidence, new_layer, existing_id))
                    engine.store.conn.commit()
                    node_id = existing_id
                except Exception:
                    node_id = existing_id

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
                  "features": ev["features"]}
        if decided == "long_term":
            try:
                engine.protect_node(node_id, f"LongTermMemoryGate:{source}")
                result["protected"] = True
            except Exception:
                result["protected"] = False
            # 关联边：与最相似知识节点建 similar 边（信息差驱动的关联）
            try:
                from .core import EdgeType
                links = engine.store.search_content(content, limit=3)
                for other, sim in links:
                    if other.id != node_id and sim >= 0.25:
                        engine.add_edge(node_id, other.id,
                                        relation_type=EdgeType.SIMILAR,
                                        source_evidence="inferred")
                result["links"] = len([x for x in links if x[0].id != node_id])
            except Exception:
                result["links"] = 0
        return result

    # ---- 前馈新奇检测（H1 · 海马体学习：新颖→当场强化编码） ----

    NOVEL_TRIGGER = 0.75   # 新奇度阈值：1-相似度 ≥ 0.75 → 判定「新东西」
    NOVEL_BOOST = 0.15     # 新奇输入 importance 提升
    NOVEL_EDGE_SIM = 0.25  # 与相关知识的建边最低相似度

    def prefeed(self, content: str, source: str = "input",
                tags: list = None, entities: list = None) -> dict:
        """海马体式前馈：输入到来时先检测新奇度，高新奇 → 当场强化编码。

        返回 {novel, novelty, action, node_id, importance, links}
        - novel=True：触发了强化编码（标记 novel_prefeed + importance 提升 + 建边）
        - novel=False：常规路径（novelty 未达阈值，不干预）
        """
        engine = self.engine
        try:
            novelty = self._novelty(content)
        except Exception:
            novelty = 0.5
        if novelty < self.NOVEL_TRIGGER:
            return {"novel": False, "novelty": round(novelty, 3),
                    "action": "routine"}

        # 高新奇 → 强化编码：importance 提升 + 标签 + 建边
        # novelty 复用上面已算出的值（#389：避免对同一输入重复全库扫描）
        base_imp = self.evaluate(content, source, tags, novelty=novelty).get(
            "importance", 0.5)
        imp = min(1.0, base_imp + self.NOVEL_BOOST)
        tags_all = list(dict.fromkeys((tags or []) + ["novel_prefeed", "gate"]))
        node_id = None
        links = 0
        actual_imp = None
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
            # 与相关知识建边（信息差驱动的关联）
            try:
                from .core import EdgeType
                rel = engine.store.search_content(content, limit=3)
                for other, sim in rel:
                    if other.id != node_id and sim >= self.NOVEL_EDGE_SIM:
                        engine.add_edge(node_id, other.id,
                                        relation_type=EdgeType.SIMILAR,
                                        source_evidence="inferred")
                        links += 1
            except Exception:
                pass
            # 长期层 → 保护
            if imp >= self.LONG_TERM_THRESHOLD:
                try:
                    engine.protect_node(node_id, f"Prefeed:{source}")
                except Exception:
                    pass
        except Exception:
            pass
        return {"novel": True, "novelty": round(novelty, 3),
                "action": "prefeed_boost", "node_id": node_id,
                "importance": round(float(actual_imp if actual_imp is not None
                                          else imp), 3),
                "links": links}

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
            from .core import MemoryLayer
            nodes = engine.store.get_layer_nodes(MemoryLayer.CONTEXT)
            for node in nodes:
                if len(promoted) >= limit:
                    break
                ev = self.evaluate(node.content, "promote", node.tags, node.id)
                if ev["layer"] in ("long_term", "knowledge"):
                    import json as _json
                    new_imp = max(node.importance, ev["importance"])
                    tags_new = list(dict.fromkeys((node.tags or []) + ["promoted"]))
                    engine.store.conn.execute(
                        "UPDATE nodes SET importance=?, layer=?, tags=? WHERE id=?",
                        (new_imp, MemoryLayer.KNOWLEDGE.value,
                         _json.dumps(tags_new, ensure_ascii=False), node.id))
                    engine.store.conn.commit()
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
