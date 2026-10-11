#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prediction_engine · 预测能力补全（v1.9）
四通道预测引擎（PREDICTION-COMPLETION-PLAN-REV1-20260813-001）：
  通道3 生成式：predict_routes（因果路线图）
  通道4 语义式：2D/3D 语义结构图（经因果过滤门）
修正要点：
  D-001 局部路径生成 + uncertainty_bound（候选未来，非必然未来）
  D-002 语义邻近过滤门（伪因果防护）
  D-003 3D 轨迹局部线性近似 + extrapolation_validity（smooth/jump/unknown）
  D-004 评分与 2.10 节 T_pred 四维度对齐
  D-005 AttentionPolicy 适配器 + 降级路径（边置信度排序）
  D-006 命中率动态校准（MIN_SAMPLES=50 · 2.7.2 动态死区）
纯标准库 · 零外部依赖
"""

import time
from typing import Dict, List, Optional

# #40：语义邻近回退（中文二元组 Jaccard）的实现来源。
# 私域（AEIS）以裸名 `spacetime_memory_core` 命名空间注册**同一** LayeredStore；
# 本仓形态无该别名——旧实现在此 `return 0.0`，致通道4（语义邻近）在本仓
# **整体静默失效**（`semantic_neighbors` 恒空）。补仓内回落：
try:  # 私域别名面（component_resolver 白名单内，勿删）
    from spacetime_memory_core import LayeredStore as _LayeredStore
except Exception:  # 本仓形态：回落仓内 core 的同名实现（char_bigram_jaccard 逐字同式）
    from ..core.core import LayeredStore as _LayeredStore


class PredictionEngine:
    """预测引擎：'图结构的过去 + 结构 → 候选未来集合'（非确定性输出）"""

    MIN_SAMPLES = 50          # D-006：最低样本量（置信区间收敛）
    BASE_HIT_RATE = 0.40      # D-006：基线阈值（工程初值，非协议承诺）
    # #395：路线图生成的总量预算（资源上界）。
    # 判据来源＝**经验标定，追不到理论出处**（本仓理论稿
    # docs/theory/世界模型与语义时空图_完整理论整理与实现路线.md 未给路线规模口径）；
    # 取同仓 `core.py` HISTORY_MAX「常量 + 超限截断」既有手法。默认参数
    # （horizon=3 · max_branches=5 ⇒ ≤155 条；core.py 调用 horizon=2）远低于此，
    # 仅在显式大 horizon 的病态输入上生效。
    MAX_ROUTES = 500

    def __init__(self, engine, attention_policy=None):
        self.engine = engine
        self.attention_policy = attention_policy   # D-005 适配器（duck-typed get_weights()）
        self.prediction_log: List[Dict] = []
        self._hit_history: List[bool] = []          # 验证闭环历史（D-006）
        # #117：最近一次 _generate_routes 中**实际参与**的边（edge 级参与记录）——
        # 目标节点 → 进入它的因果/时序边 id 集合。供命中反馈只强化参与边（D-40）。
        self._last_participation: Dict[str, set] = {}

    # ==================== 语义邻近（通道4原料 · 经过滤门） ====================

    def semantic_neighbors(self, node_id: str, k: int = 5) -> List:
        """语义邻近候选（语义坐标余弦相似度；无坐标回退文本相似度）"""
        center = self.engine.store.get_node(node_id)
        if not center:
            return []
        nodes = self.engine.store.query_nodes(limit=300)
        scored = []
        for n in nodes:
            if n.id == node_id:
                continue
            sim = self._similarity(center, n)
            if sim > 0.05:
                scored.append((n, sim))
        scored.sort(key=lambda x: -x[1])
        return [n for n, _ in scored[:k]]

    def _similarity(self, a, b) -> float:
        """语义坐标相似度（优先）或中文二元组 Jaccard（回退）"""
        sc_a = getattr(a, "semantic_coordinates", {}) or {}
        sc_b = getattr(b, "semantic_coordinates", {}) or {}
        if sc_a and sc_b:
            try:
                from semantic_space import SemanticSpaceProvider
                return SemanticSpaceProvider.similarity_coordinates(sc_a, sc_b)
            except Exception:
                pass
        return _LayeredStore.char_bigram_jaccard(a.content, b.content)

    # ==================== 过滤门（D-002 伪因果防护） ====================

    def has_causal_link(self, a_id: str, b_id: str) -> bool:
        """直接因果/时序边"""
        for e in self.engine.store.get_outgoing_edges(a_id):
            if e.target_id == b_id and e.relation_type.value in ("causal", "sequential"):
                return True
        return False

    def has_structural_pattern(self, a_id: str, b_id: str) -> bool:
        """结构模式：共同父节点（间接关联）"""
        a_parents = {e.source_id for e in self.engine.store.get_incoming_edges(a_id)}
        b_parents = {e.source_id for e in self.engine.store.get_incoming_edges(b_id)}
        return bool(a_parents & b_parents)

    def _preference_weight(self, content: str) -> float:
        """D-005 适配器：AttentionPolicy 偏好权重；降级返回 0.0（回退边置信度排序）"""
        if self.attention_policy is None:
            return 0.0
        try:
            w = self.attention_policy.get_weights()
            score = 0.0
            if "存在" in content or "威胁" in content:
                score += w.get("existence", 1.0)
            if "信任" in content:
                score += w.get("trust", 0.8)
            if "信息差" in content or "盲区" in content:
                score += w.get("gap", 0.6)
            return score
        except Exception:
            return 0.0

    def _branch_candidates(self, start_id: str) -> List:
        """分支候选：因果边（直通）+ 语义邻近（经过滤门 D-002）
        v1.16（GPT 审查·可信度分层）：不同来源具有不同认知地位——
          causal（显式因果边）→ 高可信主路径（边置信度）
          structural（共同父结构模式）→ 中可信探索路径（0.6）
          preference（注意力偏好）→ 低可信假设路径（0.3）
        不禁止低可信候选（CNN/探索是感知机构，禁止会掐死探索）——只分层。"""
        candidates = []
        seen = set()
        for e in self.engine.store.get_outgoing_edges(start_id):
            if e.relation_type.value in ("causal", "sequential"):
                candidates.append((e.target_id, e.confidence, "causal"))
                seen.add(e.target_id)
        for n in self.semantic_neighbors(start_id, k=5):
            if n.id in seen:
                continue
            if self.has_causal_link(start_id, n.id):
                candidates.append((n.id, 0.6, "structural_causal"))
            elif self.has_structural_pattern(start_id, n.id):
                candidates.append((n.id, 0.6, "structural_pattern"))
            elif self._preference_weight(n.content) > 0.8:
                candidates.append((n.id, 0.3, "semantic_induced"))
        return candidates

    # ==================== 生成式预测：因果路线图（D-001/D-004） ====================

    def predict_routes(self, start_id: str = None, blindspot_id: str = None,
                       horizon: int = 3, max_branches: int = 5) -> Dict:
        """生成式预测：候选未来路径集合（非必然未来 · uncertainty_bound）
        v1.10：盲区驱动（blindspot_id）——unknowable 盲区不生成路线（D-003）
        #117：入口即清空参与边记录——本轮未生成的路线不留旧记录（防跨轮误强化）"""
        self._last_participation = {}
        if blindspot_id is not None:
            bs = self._find_blindspot(blindspot_id)
            if bs is None:
                return {"status": "blindspot_not_found", "routes": []}
            if bs.get("predictability") == "unknowable":
                return {"status": "unpredictable", "reason": "structural_unknowability",
                        "routes": []}
            anchor = self._anchor_from_description(bs.get("description", ""))
            if anchor is None:
                return {"status": "no_anchor", "routes": []}
            result = self._generate_routes(anchor, horizon, max_branches)
            result["meta"]["blindspot_id"] = blindspot_id
            return result
        if start_id is None:
            return {"status": "no_start", "routes": []}
        return self._generate_routes(start_id, horizon, max_branches)

    def _generate_routes(self, start_id: str, horizon: int, max_branches: int) -> Dict:
        """路线图生成（原 predict_routes 主体）
        v1.15 H2：预演规划——每条路线附带「条件空间序列层」，
        每个路径节点标注该步成立的预测条件（来自边/节点条件空间的存在约束）。

        #395 两道上界（旧实现只按 `depth >= horizon` 截断）：
          ① **简单路径**：`nid in path` 即跳过——因果路线不得重访节点（环/自环
             不再生成带重复节点的伪路线，旧实现会产出 a→b→a→b… 这类"判错"路线）。
          ② **总量预算** `MAX_ROUTES`：路线条数达上限即停止展开——旧实现按
             `max_branches ** horizon` 指数增长（无预算/去重）。
        削掉的判别力（如实声明）：环状因果结构不再产出"绕环一圈"的路线；大 horizon
        下的候选未来被截到 MAX_ROUTES 条（尾部低分路线不再出现）。两道上界只在
        路线数**将要超过** MAX_ROUTES 时咬合（默认 core.py 调用 horizon=2）；
        注意 #40 修复使语义邻近通道在本仓恢复，同一图上的候选边会比修复前更多
        （通道4 由恒空变为可用），故"修复前后路线条数逐字相同"**不成立**。"""
        routes = []

        def _cs_label(node_id: str, edge_cs=None) -> str:
            """提取条件标签：边条件空间 → 节点条件空间 → 待定。
            优先取存在约束（existence_constraint），截断至 40 字。"""
            if edge_cs is not None:
                try:
                    import json as _json
                    d = _json.loads(edge_cs.to_json())
                    ec = str(d.get("existence_constraint", "")).strip()
                    if ec:
                        return ec[:40]
                except Exception:
                    pass
            try:
                n = self.engine.store.get_node(node_id)
                if n is not None and n.condition_space is not None:
                    import json as _json
                    d = _json.loads(n.condition_space.to_json())
                    ec = str(d.get("existence_constraint", "")).strip()
                    if ec:
                        return ec[:40]
            except Exception:
                pass
            return "待定（条件空间未声明）"

        def dfs(current: str, path: List[str], conditions: List[str],
                depth: int, conf: float):
            if depth >= horizon:
                return
            for nid, ec, src in self._branch_candidates(current)[:max_branches]:
                if len(routes) >= self.MAX_ROUTES:   # #395②：总量预算（逐条严格封顶）
                    return
                if nid in path:      # #395①：简单路径——不重访（环/自环不成路线）
                    continue
                new_path = path + [nid]
                # 该步条件：优先取 current→nid 边的条件空间
                edge_cs = None
                edge_id = None
                try:
                    for e in self.engine.store.get_outgoing_edges(current):
                        if e.target_id == nid and e.relation_type.value in ("causal", "sequential"):
                            edge_cs = e.condition_space
                            edge_id = e.id
                            break
                except Exception:
                    pass
                cond = _cs_label(nid, edge_cs)
                new_conds = conditions + [cond]
                if edge_id is not None:   # #117：记录进入 nid 的参与边（edge 级）
                    self._last_participation.setdefault(nid, set()).add(edge_id)
                routes.append({"path": new_path, "conf": round(conf * ec, 4),
                               "source": src, "conditions": new_conds})
                dfs(nid, new_path, new_conds, depth + 1, conf * ec)

        dfs(start_id, [start_id], ["起点（观测条件）"], 0, 1.0)
        scored = []
        for r in routes:
            s = self._score_route(r)
            scored.append({**r, "score": s,
                           "uncertainty_bound": self._uncertainty(r["conf"])})
        scored.sort(key=lambda r: -r["score"]["composite"])
        self.prediction_log.append({"type": "predict_routes", "start": start_id,
                                    "routes": len(scored), "ts": time.time()})
        return {"routes": scored,
                "meta": {"horizon": horizon, "start": start_id,
                         "note": "候选未来集合，非必然未来（0.0.3 局部不可知）；"
                                 "每条路线含条件空间序列（预演规划 H2）"}}

    def _find_blindspot(self, blindspot_id: str) -> Optional[Dict]:
        try:
            for b in self.engine.list_blindspots():
                if b["id"] == blindspot_id:
                    return b
        except Exception:
            pass
        return None

    def _anchor_from_description(self, description: str) -> Optional[str]:
        """盲区描述的语义锚点：LIKE 检索优先，语义坐标相似度回退"""
        try:
            hits = self.engine.search_content(description, limit=3)
            if hits:
                return hits[0][0].id
        except Exception:
            pass
        try:
            from semantic_space import SemanticSpaceProvider
            q = SemanticSpaceProvider().to_semantic_coordinates(description)
            best, best_sim = None, 0.0
            best_with_routes, best_routes_sim = None, 0.0
            for n in self.engine.store.query_nodes(limit=300):
                sc = getattr(n, "semantic_coordinates", {}) or {}
                if not sc:
                    continue
                sim = SemanticSpaceProvider.similarity_coordinates(q, sc)
                if sim > best_sim:
                    best, best_sim = n.id, sim
                if sim > best_routes_sim and self.engine.store.get_outgoing_edges(n.id):
                    best_with_routes, best_routes_sim = n.id, sim
            if best_with_routes and best_routes_sim > 0.05:
                return best_with_routes   # 优先有因果延续的锚点（预测需要路线）
            if best and best_sim > 0.05:
                return best
        except Exception:
            pass
        return None

    def _score_route(self, route: Dict) -> Dict:
        """T_pred 四维度对齐（D-004 · 2.10 节）：
        trend(D₁ 边置信度) · boundary(D₂ 可信边界一致性) · verification(D₃ 命中率) · balance(D₄ 分支多样性)"""
        trend = route["conf"]
        boundary = self._boundary_consistency(route["path"])
        verification = self._hit_rate()
        balance = self._branch_diversity(route["path"])
        composite = round(0.40 * trend + 0.20 * boundary + 0.25 * verification + 0.15 * balance, 4)
        return {"trend": round(trend, 4), "boundary": round(boundary, 4),
                "verification": round(verification, 4), "balance": round(balance, 4),
                "composite": composite}

    def _boundary_consistency(self, path: List[str]) -> float:
        """D₂：路径节点是否均有可信边界声明（boundary 标记/不确定声明）"""
        if not path:
            return 0.0
        ok = 0
        for nid in path:
            n = self.engine.store.get_node(nid)
            if n and ("boundary" in n.tags or "不确定" in n.content or "边界" in n.content):
                ok += 1
        return round(ok / len(path), 4)

    def _hit_rate(self) -> float:
        """D₃：预测-验证闭环历史命中率"""
        if not self._hit_history:
            return 0.0
        return round(sum(1 for h in self._hit_history if h) / len(self._hit_history), 4)

    def _branch_diversity(self, path: List[str]) -> float:
        """D₄：路径覆盖语义子空间维度数（防单一偏好主导 · 盲区47）"""
        dims = set()
        for nid in path:
            n = self.engine.store.get_node(nid)
            if n:
                sc = n.semantic_coordinates or {}
                for k in sc.get("protocol", {}):
                    dims.add(k)
        return round(min(1.0, len(dims) / 4.0), 4)

    def _uncertainty(self, conf: float) -> Dict:
        """D-001：基于局部不可知原理的置信区间估计"""
        base = 1.0 - conf
        return {"lower": round(max(0.0, conf - base * 0.5), 4),
                "upper": round(min(1.0, conf + base * 0.5), 4)}

    # ==================== 验证闭环（盲区28 · D-006 动态校准） ====================

    def update_prediction_feedback(self, predicted_node_id: str,
                                   actual_node_id: str, hit: bool,
                                   note: str = "") -> Dict:
        """命中：路径强化（边置信度 +0.05）/ 未命中：衰减 + 被拒路径登记
        v1.15：note 记录到验证条目（可审计）
        v1.16（GPT 审查·自动条件化）：未命中 → 除登记被拒路径外，**自动发现
        缺失条件并写入条件候选节点**（错误 → 新条件 → 新结构，不等待飞轮触发）

        #117 修法（判据来源＝设计者裁定 D-40/D-41，见 docs/plans/待裁清单_v0.1.md
        §附-E「#117 参与边记录粒度=edge 级／未命中只不增信＋记录」）：
          命中只强化**本轮实际参与**的边（`_last_participation` 里记录的 edge id），
          且 `verify_edge` 只收**未验证**的边——旧实现遍历目标节点**全部**因果入边、
          无条件 `verified=1` ⇒ 一次命中即把**非参与**边一并永久标 verified
          （`decay_cycle` 的 `WHERE e.verified = 0` 从此豁免它们）。无参与记录
          （从未经 `predict_routes` 走到该目标节点）时 **fail-closed**：不强化任何边。
        削掉的判别力（如实声明）：未经 `predict_routes` 的**直接反馈**不再强化
          「目标节点的全部因果入边」——它不再产生任何增信；若目标节点的参与边早已
          verified，本次命中同样不再增信（旧实现会重复 +0.05）。"""
        self._hit_history.append(hit)
        if len(self._hit_history) > 200:
            self._hit_history = self._hit_history[-200:]
        if hit and predicted_node_id == actual_node_id:
            participated = self._last_participation.get(predicted_node_id) or set()
            for e in self.engine.store.get_incoming_edges(predicted_node_id):
                if e.relation_type.value not in ("causal", "sequential"):
                    continue
                if e.verified:          # #117：已验证边不重复增信（幂等）
                    continue
                if e.id not in participated:
                    continue            # #117：非本轮参与边不强化（fail-closed）
                self.engine.store.verify_edge(e.id, min(1.0, e.confidence + 0.05))
        elif not hit:
            try:
                self.engine.register_rejected_path(
                    path_type="prediction",
                    description=f"预测未命中：{predicted_node_id}"
                                + (f"（{note}）" if note else ""),
                    reason=f"实际节点：{actual_node_id}")
            except Exception:
                pass
            # 自动条件化：比较预测/实际节点条件空间 → 缺失条件 → 条件候选节点
            try:
                self._auto_conditionize(predicted_node_id, actual_node_id)
            except Exception:
                pass
        return self._dynamic_hit_threshold()

    def _auto_conditionize(self, predicted_id: str, actual_id: str) -> List[str]:
        """预测误差自动条件化（GPT 审查·第一优先级）：
        错误 → 发现缺失条件 → 条件候选节点（条件成为可学习的认知对象）。
        机制：比较预测节点 vs 实际节点的条件空间——实际有的存在约束/观测
        位置预测没有 → 即「遗漏条件」，写入条件候选节点（可验证/可继承）。"""
        try:
            pn = self.engine.store.get_node(predicted_id)
            an = self.engine.store.get_node(actual_id)
        except Exception:
            return []
        if not pn or not an:
            return []
        missing = []
        try:
            import json as _json
            p_cs = _json.loads(pn.condition_space.to_json()) if pn.condition_space else {}
            a_cs = _json.loads(an.condition_space.to_json()) if an.condition_space else {}
        except Exception:
            return []
        p_ec = str(p_cs.get("existence_constraint", "")).strip()
        a_ec = str(a_cs.get("existence_constraint", "")).strip()
        if a_ec and a_ec != p_ec:
            missing.append(f"缺失条件：{a_ec}")
        a_pos = str(a_cs.get("observation_position", "")).strip()
        p_pos = str(p_cs.get("observation_position", "")).strip()
        if a_pos and a_pos != p_pos and "外部" not in a_pos:
            missing.append(f"缺失条件：观测位置[{a_pos}]")
        created = []
        for cond in missing:
            try:
                node = self.engine.add_perception(
                    content=f"[条件候选] {cond}（预测误差自动条件化：{predicted_id}→{actual_id}）",
                    importance=0.7,
                    tags=["condition_candidate", "auto_conditionized", "prediction_error"])
                created.append(node.id)
            except Exception:
                pass
        if created:
            self.prediction_log.append({"type": "auto_conditionize",
                                        "predicted": predicted_id,
                                        "actual": actual_id,
                                        "conditions": missing,
                                        "created": created, "ts": time.time()})
        return created

    def _dynamic_hit_threshold(self) -> Dict:
        """D-006：动态阈值 max(BASE, mean-2σ)；样本 < MIN_SAMPLES 不触发反思"""
        n = len(self._hit_history)
        if n < self.MIN_SAMPLES:
            return {"threshold": self.BASE_HIT_RATE, "samples": n,
                    "reflect": False, "note": "样本不足（<50），不触发反思"}
        mean = sum(1 for h in self._hit_history if h) / n
        var = sum(((1.0 if h else 0.0) - mean) ** 2 for h in self._hit_history) / n
        std = var ** 0.5
        threshold = max(self.BASE_HIT_RATE, mean - 2 * std)
        return {"threshold": round(threshold, 4), "samples": n,
                "reflect": mean < threshold, "mean": round(mean, 4)}

    # ==================== 2D 语义地图（通道4 · 零依赖渲染） ====================

    def render_semantic_map_2d(self, limit: int = 50) -> Dict:
        """2D 语义结构图：语义坐标 → 2D 投影（最高频两语义轴 · 有损投影盲区25）"""
        nodes = self.engine.store.query_nodes(limit=limit)
        axes = self._top_axes(nodes, 2)
        positions = {}
        for n in nodes:
            concept = (n.semantic_coordinates or {}).get("protocol", {}).get("concept", {})
            x = concept.get(axes[0], 0.0) if len(axes) > 0 else 0.0
            y = concept.get(axes[1], 0.0) if len(axes) > 1 else 0.0
            positions[n.id] = {"x": round(x, 4), "y": round(y, 4), "content": n.content[:12]}
        return {"axes": axes, "positions": positions, "count": len(positions),
                "note": "ND 语义空间 2D 有损投影（盲区25）"}

    # ==================== 3D 时空语义立方体（D-003） ====================

    def render_semantic_cube_3d(self, entity_id: str = None, limit: int = 50) -> Dict:
        """3D 时空语义立方体：时间轴 + 双语义轴；实体轨迹 + extrapolation_validity"""
        nodes = self.engine.store.query_nodes(limit=limit)
        if entity_id:
            nodes = [n for n in nodes if n.entity_id == entity_id
                     or (n.tags and f"ent:{entity_id}" in n.tags)]
        if not nodes:
            return {"entity_id": entity_id, "trajectory": [], "note": "无轨迹数据"}
        axes = self._top_axes(nodes, 2)
        trajectory = []
        prev_pt = None
        for n in sorted(nodes, key=lambda n: n.temporal_coordinate):
            concept = (n.semantic_coordinates or {}).get("protocol", {}).get("concept", {})
            x = concept.get(axes[0], 0.0) if len(axes) > 0 else 0.0
            y = concept.get(axes[1], 0.0) if len(axes) > 1 else 0.0
            pt = {"id": n.id, "t": round(n.temporal_coordinate, 4),
                  "x": round(x, 4), "y": round(y, 4)}
            if prev_pt is not None:
                delta = abs(pt["x"] - prev_pt["x"]) + abs(pt["y"] - prev_pt["y"])
                if delta < 0.05:
                    pt["extrapolation_validity"] = "smooth"      # 局部线性近似有效
                elif delta >= 0.3:
                    pt["extrapolation_validity"] = "jump"        # 外推失效，须依赖因果边
                else:
                    pt["extrapolation_validity"] = "unknown"     # 数据不足，不外推
            prev_pt = pt
            trajectory.append(pt)
        return {"entity_id": entity_id, "axes": axes, "trajectory": trajectory,
                "note": "外推仅在 smooth 区间有效（D-003）"}

    def _top_axes(self, nodes: List, n: int) -> List[str]:
        axis_freq = {}
        for node in nodes:
            concept = (node.semantic_coordinates or {}).get("protocol", {}).get("concept", {})
            for k in concept:
                axis_freq[k] = axis_freq.get(k, 0) + 1
        axes = sorted(axis_freq, key=axis_freq.get, reverse=True)[:n]
        return axes
