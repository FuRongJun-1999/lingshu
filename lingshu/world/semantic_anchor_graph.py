# -*- coding: utf-8 -*-
"""semantic_anchor_graph · 3D 语义锚点图（世界模型阶段1 · 里程碑1.2）
============================================================================
核心哲学（荣）：「事物是其关系的总和」——一个物体的身份由其与世界中
其他物体的关系定义，而非孤立的属性列表。

图结构（可扩展）：
  节点 = 3D 语义锚点（类别 + 空间坐标 + provenance + 可信度）
  边   = 关系（空间：位于/相邻/支撑；语义：属于/类似；时间：先于/伴随）

设计参考：
  - 游戏场景图（ECS/Scene Graph）：节点层级 + 关系边
  - 智能论3.4 第四章空间关系边（adjacent/contains/connected/similar/opposite）
  - LangSplat/S3Gaussian（3d-world）：语义-几何绑定
  - 3D 场景图（3D Scene Graph）：物体-关系-物体三元组

可扩展性：
  - 关系类型开放注册（register_relation_type）——任意新关系即插即用
  - 节点属性开放（attrs 字典）——新语义维度随时添加

纯标准库 · 零外部依赖（D-005）
"""
from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 关系类型注册表（开放扩展——任意新关系类型即插即用）
# ---------------------------------------------------------------------------

# 默认关系类型（空间/语义/时间三维）
RELATION_TYPES: Dict[str, Dict] = {
    # 空间关系（3D 场景图核心）
    "位于":     {"dim": "spatial", "desc": "A 位于 B 之内/之上（如杯子在桌上）"},
    "相邻":     {"dim": "spatial", "desc": "A 与 B 空间相邻（如椅子邻桌）"},
    "支撑":     {"dim": "spatial", "desc": "A 支撑 B（如桌子支撑杯子）"},
    "朝向":     {"dim": "spatial", "desc": "A 面向 B（如椅子朝向桌子）"},
    "包含":     {"dim": "spatial", "desc": "A 包含 B（如房间包含桌子）"},
    # 语义关系
    "属于":     {"dim": "semantic", "desc": "A 属于类别/集合 B"},
    "类似":     {"dim": "semantic", "desc": "A 与 B 语义类似"},
    "部分是":   {"dim": "semantic", "desc": "A 是 B 的组成部分"},
    # 时间关系
    "先于":     {"dim": "temporal", "desc": "A 在时间上先于 B"},
    "伴随":     {"dim": "temporal", "desc": "A 与 B 同时出现/运动"},
}


def register_relation_type(name: str, dim: str = "custom", desc: str = "") -> None:
    """注册新关系类型（可扩展核心——任意新关系即插即用）。"""
    RELATION_TYPES[name] = {"dim": dim, "desc": desc}


# ---------------------------------------------------------------------------
# 节点：3D 语义锚点
# ---------------------------------------------------------------------------


@dataclass
class SemanticAnchor:
    """3D 语义锚点节点：类别 + 空间坐标 + provenance + 可信度。

    attrs: 开放属性字典（任意新语义维度）
    provenance: 观测来源（哪个视角/时刻/工具观测到——缸中之脑边界）
    """
    category: str
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)   # 世界坐标 (x,y,z)
    size: Tuple[float, float, float] = (1.0, 1.0, 1.0)      # (w,h,d) 米
    confidence: float = 0.5
    provenance: str = "world3d"      # 观测来源（视角/时刻/工具）
    ts: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: "anchor_" + uuid.uuid4().hex[:10])
    attrs: Dict = field(default_factory=dict)   # 开放属性

    def to_dict(self) -> Dict:
        return asdict(self)

    def __repr__(self):
        c = tuple(round(v, 2) for v in self.center)
        return f"<Anchor {self.category}@{c} conf={self.confidence:.2f}>"


# ---------------------------------------------------------------------------
# 边：关系（事物是其关系的总和）
# ---------------------------------------------------------------------------


@dataclass
class RelationEdge:
    """关系边：源节点 -关系-> 目标节点。

    relation: 关系类型名（来自 RELATION_TYPES 注册表）
    weight: 关系强度/置信度
    attrs: 开放属性（如 spatial_distance、time_delta）
    """
    source: str
    target: str
    relation: str = "相邻"
    weight: float = 0.5
    attrs: Dict = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# 3D 语义锚点图（图结构 · 事物是其关系的总和）
# ---------------------------------------------------------------------------


class SemanticAnchorGraph:
    """3D 语义锚点图：节点（锚点）+ 边（关系）。

    核心操作：
      - add_anchor(anchor)：添加节点
      - relate(a_id, b_id, relation, weight, attrs)：建立关系边
      - relations_of(anchor_id)：某节点的全部关系（"事物的总和"）
      - neighbors(anchor_id, relation=None)：邻居（可按关系过滤）
      - query(category=None, region=None)：按类别/区域查询
      - infer_relations()：关系推理（空间邻近 → 相邻/支撑）
      - scene_text()：场景语义描述
    """

    def __init__(self):
        self.anchors: Dict[str, SemanticAnchor] = {}
        self.edges: List[RelationEdge] = []

    # ---- 节点操作 ----

    def add_anchor(self, anchor: SemanticAnchor) -> str:
        self.anchors[anchor.id] = anchor
        return anchor.id

    def add(self, category: str, center: Tuple[float, float, float],
            size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
            confidence: float = 0.5, provenance: str = "world3d",
            attrs: Optional[Dict] = None) -> str:
        """便捷添加锚点。"""
        a = SemanticAnchor(category, center, size, confidence, provenance,
                           attrs=attrs or {})
        return self.add_anchor(a)

    def get(self, anchor_id: str) -> Optional[SemanticAnchor]:
        return self.anchors.get(anchor_id)

    # ---- 边操作（关系）----

    def relate(self, source: str, target: str, relation: str = "相邻",
               weight: float = 0.5, attrs: Optional[Dict] = None) -> Optional[RelationEdge]:
        """建立关系边（source -relation-> target）。"""
        if source not in self.anchors or target not in self.anchors:
            return None
        if relation not in RELATION_TYPES:
            register_relation_type(relation, desc="自定义关系")  # 自动注册
        e = RelationEdge(source, target, relation, weight, attrs or {})
        self.edges.append(e)
        return e

    # ---- 查询（事物是其关系的总和）----

    def relations_of(self, anchor_id: str) -> List[Dict]:
        """某节点的全部关系——"事物的总和"（出边+入边）。"""
        out = []
        for e in self.edges:
            if e.source == anchor_id:
                out.append({"direction": "out", "other": e.target,
                            "relation": e.relation, "weight": e.weight,
                            "attrs": e.attrs})
            elif e.target == anchor_id:
                out.append({"direction": "in", "other": e.source,
                            "relation": e.relation, "weight": e.weight,
                            "attrs": e.attrs})
        return out

    def neighbors(self, anchor_id: str, relation: Optional[str] = None) -> List[Dict]:
        """邻居（可按关系类型过滤）。"""
        out = []
        for e in self.edges:
            if relation and e.relation != relation:
                continue
            if e.source == anchor_id:
                out.append({"id": e.target, "relation": e.relation,
                            "weight": e.weight})
            elif e.target == anchor_id:
                out.append({"id": e.source, "relation": e.relation,
                            "weight": e.weight})
        return out

    def query(self, category: Optional[str] = None,
              region: Optional[Tuple] = None) -> List[SemanticAnchor]:
        """按类别/区域查询锚点。region=(xmin,ymin,zmin,xmax,ymax,zmax)。"""
        out = []
        for a in self.anchors.values():
            if category and a.category != category:
                continue
            if region:
                x, y, z = a.center
                if not (region[0] <= x <= region[3] and region[1] <= y <= region[4]
                        and region[2] <= z <= region[5]):
                    continue
            out.append(a)
        return out

    # ---- 关系推理（确定性 · 零 LLM）----

    def infer_relations(self, distance_threshold: float = 1.5) -> int:
        """空间邻近推理：近距物体 → 相邻边；上下叠放 → 支撑边。

        借鉴 3D 场景图（3D Scene Graph）的物体-关系-物体三元组。
        返回新增边数。"""
        added = 0
        ids = list(self.anchors.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = self.anchors[ids[i]], self.anchors[ids[j]]
                dist = math.dist(a.center, b.center)
                # 是否已有边
                has_edge = any((e.source == a.id and e.target == b.id) or
                               (e.source == b.id and e.target == a.id)
                               for e in self.edges)
                if has_edge:
                    continue
                # 水平距离（忽略高度差）
                h_dist = math.hypot(a.center[0] - b.center[0], a.center[2] - b.center[2])
                v_gap = abs(a.center[1] - b.center[1])
                if h_dist < distance_threshold and v_gap < 0.2:
                    # 几乎同一位置不同高度 → 支撑（下撑上）
                    lower, upper = (a, b) if a.center[1] < b.center[1] else (b, a)
                    self.relate(lower.id, upper.id, "支撑", 0.7,
                                attrs={"v_gap": round(v_gap, 2)})
                    added += 1
                elif dist < distance_threshold:
                    self.relate(a.id, b.id, "相邻", 0.5,
                                attrs={"distance": round(dist, 2)})
                    added += 1
        return added

    # ---- 场景描述 ----

    def scene_text(self) -> str:
        """场景语义描述（图结构文本形态——锚点 + 关系）。"""
        parts = []
        for a in self.anchors.values():
            x, y, z = [round(v, 1) for v in a.center]
            parts.append(f"{a.category}@({x},{y},{z})")
        rels = []
        for e in self.edges:
            rels.append(f"{self.anchors.get(e.source).category if e.source in self.anchors else '?'}"
                        f"{e.relation}{self.anchors.get(e.target).category if e.target in self.anchors else '?'}")
        scene = "；".join(parts) if parts else "（空）"
        rel_text = "；".join(rels) if rels else "（无关系）"
        return f"锚点: {scene} | 关系: {rel_text}"

    def to_dict(self) -> Dict:
        return {
            "anchors": [a.to_dict() for a in self.anchors.values()],
            "edges": [e.to_dict() for e in self.edges],
            "relation_types": list(RELATION_TYPES.keys()),
        }

    # ---- 多感知机锚点验证（里程碑1.4）----

    def verify(self, anchor_id: str, channel: str, evidence: float,
               strong: Optional[bool] = None) -> Dict:
        """多感知机验证：记录某通道证据 → 确认度判定。

        核心（荣）：一个事物不能只有视觉一层信息——触觉/听觉/行动等
        多通道协同才能确认锚点（打破视觉自证陷阱）。"""
        try:
            from .anchor_verify import AnchorVerification
        except ImportError:
            from .anchor_verify import AnchorVerification
        if not hasattr(self, '_verifier'):
            self._verifier = AnchorVerification(graph=self)
        self._verifier.add_channel_evidence(anchor_id, channel, evidence, strong)
        return self._verifier.verify_anchor(anchor_id)

    def verify_conflict(self, anchor_id: str, channel: str,
                        expected: str, actual: str) -> Dict:
        """多通道矛盾检测：通道观测与锚点声明不符 → 冲突记录 + 降级。"""
        try:
            from .anchor_verify import AnchorVerification
        except ImportError:
            from .anchor_verify import AnchorVerification
        if not hasattr(self, '_verifier'):
            self._verifier = AnchorVerification(graph=self)
        return self._verifier.channel_conflict_detect(anchor_id, channel, expected, actual)
