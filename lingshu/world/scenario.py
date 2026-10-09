# Copyright 2026 灵枢 (Lingshu) · MIT
"""同一语义时空图的场景消费视图；不写脑、不生成或验证新知识。

沿用脑端 roleplay / role:<id> / session:<id> 与子知识约定。资格裁决
保留脑端原值；本层只检查声明的作用域并组装材料，不替代 COND-ANALYSIS。
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict

from ..core.provenance import SOURCES, VERIFY_STATUS

MODES = ("roleplay", "world", "story", "history", "causal")
STATES = frozenset({"ACCEPT", "REJECT", "DEFER", "BLINDSPOT"})
CONDITION_FIELDS = ("observation_position", "observation_tool", "time_window",
                    "existence_constraint")


def _copy(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _tags_with(tags, prefix):
    return [tag[len(prefix):] for tag in tags if tag.startswith(prefix)]


def _declared(cs):
    """只查四栏是否声明，不猜测自由文本条件的含义。"""
    if not isinstance(cs, dict):
        return False
    for key in ("observation_position", "observation_tool", "existence_constraint"):
        if not isinstance(cs.get(key), str) or not cs[key].strip() or cs[key] in {
            "unknown", "未知观测位", "外部观测位", "文本语义分析", "（未声明）"
        }:
            return False
    tw = cs.get("time_window")
    return (isinstance(tw, (list, tuple)) and len(tw) == 2
            and all(isinstance(t, (int, float)) and not isinstance(t, bool)
                    and math.isfinite(t) for t in tw) and tw[0] < tw[1])


def _provenance(tags):
    # 显式来源优先；不把 gate/novel_prefeed 当成“谁说的”。
    source = next((s for s in ("user", "assistant", "external", "system", "tool", "fixture")
                   if s in tags), "unknown")
    status = next((s for s in reversed(VERIFY_STATUS) if s in tags), "unverified")
    return {"source": source, "verify_status": status,
            "references": _tags_with(tags, "vref:"),
            "session_ids": _tags_with(tags, "session:")}


def _node(node, qualification="BLINDSPOT"):
    fm = node.get("frontmatter") or {}
    tags = list(fm.get("tags") or [])
    cs = fm.get("condition_space") or {}
    return {"id": str(node.get("id") or fm.get("id") or ""),
            "content": str(node.get("content") or ""), "tags": tags,
            "layer": fm.get("layer", "knowledge"),
            "conditions": {key: cs.get(key) for key in CONDITION_FIELDS},
            "provenance": _provenance(tags),
            "qualification": qualification if qualification in STATES else "BLINDSPOT",
            "spatial": (fm.get("spatial") or {}).get("coords3d") or {},
            "attributes": fm.get("state_attributes") or {},
            "record_role": fm.get("role"),
            "body_window": {key: node.get(key) for key in (
                "truncated", "content_lines", "content_bytes", "offset", "next_offset")}}


class ScenarioContext:
    """显式选取/召回的只读快照。每次 view 返回独立 JSON，不形成第二个记忆库。

    local store 读取沿用 get_node/get_outgoing_edges；脑端沿用已存在的
    cg(op=read)。没有脑端资格声明的本地记录保留 BLINDSPOT，不伪造 ACCEPT。
    """

    def __init__(self, nodes, edges=(), *, role_id=None, session_id=None,
                 gaps=(), origin="selected", max_nodes=256, max_edges=1024):
        if not isinstance(max_nodes, int) or isinstance(max_nodes, bool) or max_nodes < 1:
            raise ValueError("max_nodes must be a positive integer")
        if not isinstance(max_edges, int) or isinstance(max_edges, bool) or max_edges < 1:
            raise ValueError("max_edges must be a positive integer")
        for value in (role_id, session_id):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError("role_id/session_id must be nonempty strings or None")
        nodes, edges = list(nodes), list(edges)
        if len(nodes) > max_nodes:
            raise ValueError("selected nodes exceed max_nodes; select a smaller context")
        self.role_id, self.session_id, self.origin = role_id, session_id, origin
        self._nodes = {n["id"]: _copy(n) for n in nodes}
        if "" in self._nodes or len(self._nodes) != len(nodes):
            raise ValueError("nodes must have unique nonempty ids")
        self._gaps = list(gaps)
        edges = [e for e in edges if e["source_id"] in self._nodes
                 and e["target_id"] in self._nodes]
        edges.sort(key=lambda e: (e["source_id"], e["target_id"], e["id"]))
        if len(edges) > max_edges:
            self._gaps.append("edge_budget_exhausted")
        self._edges = _copy(edges[:max_edges])

    @classmethod
    def from_store(cls, store, node_ids, **kwargs):
        """按显式 id 读取既有本地图；不召回、不更新访问统计、不自动扩图。"""
        if isinstance(node_ids, str):
            raise ValueError("node_ids must be an iterable of nonempty node ids, not a string")
        ids = list(dict.fromkeys(node_ids))
        if any(not isinstance(nid, str) or not nid for nid in ids):
            raise ValueError("node_ids must contain nonempty strings")
        if len(ids) > kwargs.get("max_nodes", 256):
            raise ValueError("selected nodes exceed max_nodes")
        nodes, edges, gaps = [], [], []
        for nid in ids:
            n = store.get_node(nid)
            if n is None:
                gaps.append("missing_node:" + nid)
                continue
            nodes.append(_node({"id": n.id, "content": n.content, "frontmatter": {
                "tags": n.tags, "layer": n.layer.value,
                "condition_space": asdict(n.condition_space),
                "spatial": {"coords3d": n.spatial_coordinates},
                "state_attributes": n.state_attributes}}))
            for e in store.get_outgoing_edges(nid):
                edges.append({"id": e.id, "source_id": e.source_id,
                              "target_id": e.target_id, "relation_type": e.relation_type.value,
                              "conditions": asdict(e.condition_space), "verified": e.verified,
                              "source_evidence": e.source_evidence})
        return cls(nodes, edges, gaps=gaps, origin="selected", **kwargs)

    @classmethod
    def from_brain(cls, client, query, *, k=64, **kwargs):
        """使用现有 MCPClient；候选集是有界召回，不声称全库或完整世界。"""
        if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= kwargs.get("max_nodes", 256):
            raise ValueError("k must be an integer in [1, max_nodes]")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a nonempty string")
        args = {"op": "read", "query": query, "k": k}
        if kwargs.get("session_id"):
            args["session"] = kwargs["session_id"]
        response = client.call("cg", args)
        if response.get("ok") is False or response.get("error"):
            raise RuntimeError("brain read failed: " + str(response.get("error")))
        if not isinstance(response.get("results"), list):
            raise RuntimeError("brain read response has no results list")
        if len(response["results"]) > k:
            raise RuntimeError("brain read returned more candidates than requested k")
        nodes, edges = [], []
        for item in response.get("results", []):
            raw = item.get("node") or {}
            n = _node(raw, item.get("state"))
            nodes.append(n)
            for index, edge in enumerate((raw.get("frontmatter") or {}).get("edges") or []):
                edges.append({"id": str(edge.get("id") or f"{n['id']}:edge:{index}"),
                              "source_id": n["id"], "target_id": str(edge.get("target") or ""),
                              "relation_type": edge.get("relation_type"),
                              "conditions": edge.get("condition_space") or {},
                              "verified": edge.get("verified") is True,
                              "source_evidence": edge.get("source_evidence", "ambiguous")})
        gaps = ["recall_limit_reached"] if len(nodes) >= k else []
        real_spatial = [n for n in nodes if n["spatial"] and not (
            _tags_with(n["tags"], "role:") or "roleplay" in n["tags"] or "sub-knowledge" in n["tags"])]
        if real_spatial:
            # 沿用身体×脑 M1：事件为源，状态槽为投影。仅实体名匹配的现实记录可消费
            # 既有无角色命名空间的槽；角色同名实体不继承宿主状态。
            state_response = client.call("stg", {"op": "state_chain", "limit": 500})
            if not isinstance(state_response.get("items"), list):
                raise RuntimeError("brain state_chain response has no items list")
            if len(state_response["items"]) >= 500:
                gaps.append("state_projection_limit_reached")
            states = {}
            for event in state_response["items"]:
                if event.get("slot") == "状态" and event.get("state") == "active":
                    states.setdefault(str(event.get("subject")), []).append(event)
            for n in real_spatial:
                names = _tags_with(n["tags"], "ent:")
                events = states.get(names[0], []) if len(names) == 1 else []
                if len(events) == 1:
                    n["attributes"] = {"state": events[0].get("value"),
                                       "state_evidence": _copy(events[0])}
                elif len(events) > 1:
                    gaps.append("ambiguous_state_projection:" + n["id"])
        return cls(nodes, edges, gaps=gaps, origin="brain_recall", **kwargs)

    def _visible(self, node, mode):
        tags = node["tags"]
        roles = _tags_with(tags, "role:")
        role_bound = bool(roles) or "roleplay" in tags or "sub-knowledge" in tags
        if mode == "history" and role_bound:
            return False
        if role_bound and (not self.role_id or roles != [self.role_id]):
            return False
        if mode == "roleplay" and not role_bound:
            return False  # 不给角色注入未声明属于其视角的全知材料。
        sessions = node["provenance"]["session_ids"]
        if sessions and (not self.session_id or sessions != [self.session_id]):
            return False
        return node["qualification"] != "REJECT"

    @staticmethod
    def _domain(node):
        return "role" if (_tags_with(node["tags"], "role:") or "roleplay" in node["tags"]
                          or "sub-knowledge" in node["tags"]) else "real"

    def view(self, mode, *, start_id=None, end_id=None, max_depth=16,
             max_paths=32, max_steps=4096):
        """五种消费任务共用作用域/来源/条件；输出仅是材料与候选推演。"""
        if mode not in MODES:
            raise ValueError("unknown scenario mode: " + str(mode))
        facts = [n for n in self._nodes.values() if self._visible(n, mode)]
        facts.sort(key=lambda n: (self._time(n), n["id"]))
        ids = {n["id"] for n in facts}
        visible_nodes = {n["id"]: n for n in facts}
        edges, cross_domain = [], []
        for edge in self._edges:
            if edge["source_id"] not in ids or edge["target_id"] not in ids:
                continue
            if self._domain(visible_nodes[edge["source_id"]]) != self._domain(visible_nodes[edge["target_id"]]):
                cross_domain.append("cross_domain_edge:" + edge["id"])
            else:
                edges.append(edge)
        gaps = list(self._gaps) + cross_domain
        for n in facts:
            if n["provenance"]["source"] == "unknown":
                gaps.append("source_undeclared:" + n["id"])
            if not _declared(n["conditions"]):
                gaps.append("undeclared_conditions:" + n["id"])
            if n["body_window"].get("truncated"):
                gaps.append("body_window_truncated:" + n["id"])
            if n["qualification"] != "ACCEPT":
                gaps.append("qualification_pending:" + n["id"])
        domains = sorted({self._domain(n) for n in facts})
        domain = ("mixed" if len(domains) > 1 else domains[0] if domains else
                  "role" if mode != "history" and self.role_id else "real")
        report = {"mode": mode, "state": "ACCEPT", "domain": domain, "domains": domains, "scope": {
            "role_id": self.role_id, "session_id": self.session_id, "origin": self.origin},
            "facts": facts, "edges": edges, "gaps": gaps, "payload": {},
            "boundary": "消费视图，不验证新事实、不生成正文、不写入或升格记忆"}
        if mode == "roleplay":
            if not self.role_id:
                gaps.append("role_id_required")
            identity = [n for n in facts if "roleplay:role" in n["tags"]]
            if not identity:
                gaps.append("role_definition_missing")
            report["payload"] = {"identity": identity,
                                 "lore": [n for n in facts if "sub-knowledge" in n["tags"]],
                                 "memory": [n for n in facts if "roleplay:memory" in n["tags"]],
                                 "anchors": [n for n in facts if "roleplay:anchors" in n["tags"]],
                                 "values": [n for n in facts if "roleplay:values" in n["tags"]],
                                 "turns": [n for n in facts if _tags_with(n["tags"], "turn:")]}
        elif mode == "world":
            latest, domains = {}, {}
            for n in facts:
                names = _tags_with(n["tags"], "ent:")
                if not names or not n["spatial"]:
                    continue
                if len(names) != 1:
                    gaps.append("ambiguous_entity_name:" + n["id"])
                    continue
                sp = n["spatial"]
                if not all(isinstance(sp.get(axis), (int, float))
                           and not isinstance(sp[axis], bool) and math.isfinite(sp[axis])
                           for axis in ("x", "y", "z")):
                    gaps.append("invalid_coords:" + n["id"])
                    continue
                state = n["attributes"].get("state")
                if not isinstance(state, str) or not state.strip():
                    state = None
                    gaps.append("entity_state_unknown:" + n["id"])
                domains.setdefault(names[0], set()).add(self._domain(n))
                latest[names[0]] = {"name": names[0], "node_id": n["id"],
                                    "pos": [sp[axis] for axis in ("x", "y", "z")],
                                    "category": next(iter(_tags_with(n["tags"], "cat:")), "object"),
                                    "state": state}
            for name in list(latest):
                if len(domains[name]) > 1:
                    gaps.append("ambiguous_entity_domain:" + name)
                    del latest[name]  # 不让同名虚构实体静默顶替现实实体或反向顶替。
            report["payload"] = {"entities": list(latest.values())}
            if not latest:
                gaps.append("spatial_entities_missing")
        elif mode == "story":
            outline, cyclic = self._outline(facts, edges)
            report["payload"] = {"outline": outline,
                                 "fiction": None if domain == "mixed" else domain == "role",
                                 "background": [n for n in facts if self._background(n)],
                                 "cyclic": cyclic}
            if cyclic:
                gaps.append("event_order_cycle")
        elif mode == "history":
            timeline, unresolved = [], []
            for n in facts:
                p = n["provenance"]
                # 保留待核材料，不用“有引用”冒充“史实为真”。
                if (p["source"] not in SOURCES or p["source"] in {"unknown", "fixture"}
                        or not p["references"] or n["qualification"] != "ACCEPT"
                        or not _declared(n["conditions"]) or n["body_window"].get("truncated")):
                    unresolved.append(n)
                else:
                    timeline.append(n)
            report["payload"] = {"evidence_timeline": timeline, "unresolved": unresolved,
                                 "time_basis": "观测时间窗；不是历史事件年代"}
            if unresolved:
                gaps.append("historical_evidence_pending")
        else:
            payload = self._causal(facts, edges, start_id, end_id, max_depth, max_paths, max_steps)
            report["payload"] = payload
            if payload["truncated"]:
                gaps.append("causal_budget_exhausted")
            if payload["endpoint_outside_scope"]:
                report["state"] = ("REJECT" if start_id in self._nodes and end_id in self._nodes
                                   else "BLINDSPOT")
            elif not payload["paths"]:
                gaps.append("causal_path_not_found")
            elif any(p["hypothesis"] for p in payload["paths"]):
                gaps.append("causal_evidence_pending")
        if report["state"] not in {"REJECT", "BLINDSPOT"}:
            report["state"] = "BLINDSPOT" if not facts else ("DEFER" if gaps else "ACCEPT")
        return _copy(report)

    @staticmethod
    def _time(node):
        tw = node["conditions"].get("time_window")
        return tw[0] if _declared(node["conditions"]) else 0

    @staticmethod
    def _background(node):
        return (bool(set(node["tags"]).intersection({"roleplay:role", "sub-knowledge",
                    "roleplay:memory", "roleplay:anchors", "roleplay:values"}))
                and "event" not in node["tags"])

    @staticmethod
    def _outline(facts, edges):
        """因果/时序约束决定提纲顺序；相关性不能充当因果线。"""
        remaining = {n["id"]: n for n in facts if not ScenarioContext._background(n)}
        parents = {nid: set() for nid in remaining}
        for edge in edges:
            if (edge["relation_type"] in {"causal", "sequential"}
                    and edge["source_id"] in remaining and edge["target_id"] in remaining):
                parents[edge["target_id"]].add(edge["source_id"])
        outline = []
        while remaining:
            ready = [nid for nid in remaining if not parents[nid].intersection(remaining)]
            if not ready:
                return outline, sorted(remaining)
            ready.sort(key=lambda nid: (ScenarioContext._time(remaining[nid]), nid))
            for nid in ready:
                n = remaining.pop(nid)
                outline.append({"node_id": nid, "text": n["content"],
                                "domain": ScenarioContext._domain(n),
                                "depends_on": sorted(parents[nid])})
        return outline, []

    @staticmethod
    def _causal(facts, edges, start, end, depth, paths_limit, steps_limit):
        for value in (depth, paths_limit, steps_limit):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError("causal budgets must be positive integers")
        if not isinstance(start, str) or not start or not isinstance(end, str) or not end:
            raise ValueError("causal view requires nonempty start_id and end_id")
        nodes = {n["id"]: n for n in facts}
        result = {"paths": [], "truncated": False, "cyclic": False,
                  "endpoint_outside_scope": start not in nodes or end not in nodes,
                  "steps": 0}
        if result["endpoint_outside_scope"]:
            return result
        outgoing = {}
        for edge in edges:
            if edge["relation_type"] == "causal":
                outgoing.setdefault(edge["source_id"], []).append(edge)
        stack = [(start, (start,), ())]
        while stack:
            if result["steps"] >= steps_limit or len(result["paths"]) >= paths_limit:
                result["truncated"] = True
                break
            current, seen, path = stack.pop()
            result["steps"] += 1
            if current == end:
                result["paths"].append({"node_ids": list(seen), "edges": list(path),
                    "hypothesis": any(not e["verified"] or e["source_evidence"] != "extracted"
                                      or not _declared(e["conditions"]) for e in path)
                                      or any(nodes[nid]["qualification"] != "ACCEPT"
                                             or not _declared(nodes[nid]["conditions"])
                                             or nodes[nid]["body_window"].get("truncated")
                                             or nodes[nid]["provenance"]["source"] == "unknown"
                                             for nid in seen)})
                continue
            successors = outgoing.get(current, [])
            if len(path) >= depth:
                result["truncated"] |= any(e["target_id"] not in seen for e in successors)
                result["cyclic"] |= any(e["target_id"] in seen for e in successors)
                continue
            for edge in reversed(successors):
                target = edge["target_id"]
                if target in seen:
                    result["cyclic"] = True
                else:
                    stack.append((target, seen + (target,), path + (edge,)))
        return result

    def to_world_model(self):
        """按同一 world 视图构造已有 WorldModel；world extras 只在此处加载。"""
        from .scene_model import WorldModel

        report = self.view("world")
        model = WorldModel()
        for entity in report["payload"]["entities"]:
            model.add_entity(entity["name"], entity["category"], tuple(entity["pos"]),
                             state=entity["state"] if entity["state"] is not None else "neutral")
        model.build()
        return model
