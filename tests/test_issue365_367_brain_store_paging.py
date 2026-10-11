# -*- coding: utf-8 -*-
"""test_issue365_367_brain_store_paging · lingshu issue #365 / #367 守卫（P1）
============================================================================
两件都出在 `lingshu/world/brain_store.py` 的 M1 读向（适配器侧），且都是
「一页读全 → 静默丢数据」——守卫用**桩客户端**复刻脑端真实返回体形态，不需要
真脑端即可无条件运行（真脑端端到端见 `tests/test_brain_store.py`，缺前置则 SKIP）。

## #365 · `_state_map` 固定读 `state_chain(limit=500)` 且丢弃 `truncated`
脑端 `stg(op=state_chain)` 的返回体是 `items = units[:limit]`、
`truncated = count > len(items)`（`md_cg/stg.py:state_chain`，投影按
`(subject, slot)` 字典序稳定排序，`md_cg/state_slots.py`）。改前 `_state_map`
固定 `limit=500`、不看 `truncated`：全库状态槽位超过一页时，排在 500 名之后的
主体被静默读成「无值」→ `BrainNode.state_attributes=None` → 世界重建回落
neutral（issue 报告实测：`count=503 items=500 truncated=True`，目标主体
「肥鱼」不在 items 中）。
修法：`slot` 下推缩小命中集；`truncated` 为真时按**调用方给出的 subject 清单**
逐个精确补读（`subject`+`slot` 过滤后每 `(subject, slot)` 至多一个单元，不受
全局截断影响）；无主体清单可补读时抛 `BrainError`（「没读到」不得当作「没有值」）。

## #367 · `get_nodes_by_tag` 先取全库 top-k 再按 tag 过滤
脑端 `cg(op=read)` 回的是词法召回 top-k（`k` 生效，`md_cg/mcp_server.py` 的
read 分支），候选面截断上限是脑端 `GLOBAL_CAP = 500`（`md_cg/mdcg.py:94`）；
截断**前**的候选总数在返回体 `meta.pre_cap`（`md_cg/mdcg.py:4149,4638`）。
改前只取一页（`k=limit`）再在适配器侧按 tag 过滤：别的场景把这一页占满，本场景
返回 0 → `load_world_from_memory` 重建出空世界（issue 报告实测：roomA 实有 5 个，
`limit=5/10/25` 全回 0，`limit=30` 才回 5）。
修法：按 `meta.pre_cap` 逐轮加大 `k` 直到覆盖整个候选面再过滤；候选面本身被脑端
上限截断、`limit` 取不满时抛 `BrainError`——「无匹配」与「被截断」必须可分辨。

断言组（桩件，逐条钉住「缺陷不再存在」而非「代码能跑」）：
  A1 #365 修复后：目标主体排在投影第 501 位、全量页被截断时，其状态仍取回
     （`state_attributes == {"state": ...}`）。
  A2 #365 有牙自证：同桩件走**改前算法**（固定一页、不看 truncated）必得 None
     ——证明桩件确实复刻了缺陷形态，守卫不是空转。
  A3 #367 修复后：另一场景占满 top-k 时，本场景实体仍全部返回。
  A4 #367 有牙自证：同桩件走**改前算法**（一页 + 过滤）必得 0。
  A5 #367 候选面被脑端上限截断、`limit` 取不满 ⇒ 抛 `BrainError`（不静默返回
     残缺世界）。
  A6 #367 反向腿（防修死）：候选面已取尽且确实不足 `limit` ⇒ 正常返回，不抛。

运行（lingshu 仓根）：python -X utf8 tests/test_issue365_367_brain_store_paging.py
                    / python -X utf8 -m pytest tests/test_issue365_367_brain_store_paging.py
退出码：0 = 全过；1 = 有断言失败
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lingshu.world.brain_store import (  # noqa: E402
    BrainError, BrainStore, STATE_SLOT, _STATE_PAGE)


# ---------------------------------------------------------------------------
# 桩客户端：复刻脑端 stg(op=state_chain) / cg(op=read) 的返回体形态
# ---------------------------------------------------------------------------


def _unit(subject, value, slot=STATE_SLOT):
    return {"subject": subject, "slot": slot, "value": value, "state": "active"}


def _node(i, tag, ent):
    return {"state": "ACCEPT",
            "node": {"id": i, "content": "场景实体 %s" % ent,
                     "frontmatter": {"id": i, "tags": ["spatial", tag, "ent:%s" % ent],
                                     "spatial": {"coords3d": {"x": 0.0, "y": 0.0,
                                                              "z": 1.0}}}}}


class _StateChainClient:
    """桩：`stg(op=state_chain)` 按页截断；`cg(op=read)` 只回目标节点。

    `units` 的**顺序**即脑端投影序（`(subject, slot)` 字典序）；目标主体置于末位，
    使其落在第一页（`_STATE_PAGE`）之外——正是 #365 的触发形态。
    """

    def __init__(self, units, read_nodes):
        self.units = list(units)
        self.read_nodes = list(read_nodes)
        self.calls = []

    def call(self, name, arguments):
        self.calls.append((name, dict(arguments)))
        if name == "stg":
            subject = arguments.get("subject")
            slot = arguments.get("slot")
            us = [u for u in self.units
                  if (subject is None or u["subject"] == subject)
                  and (slot is None or u["slot"] == slot)]
            page = int(arguments.get("limit") or 50)
            items = us[:page]
            return {"count": len(us), "items": items, "kept": len(items),
                    "truncated": len(us) > len(items), "limit": page}
        if name == "cg" and arguments.get("op") == "read":
            k = max(0, int(arguments.get("k") or 0))
            return {"meta": {"pre_cap": len(self.read_nodes), "cap": 500},
                    "results": self.read_nodes[:k]}
        return {}


class _CrowdingClient:
    """桩：`cg(op=read)` 按 `k` 回候选前缀，`meta.pre_cap` 报截断前候选总数。

    `candidates` 顺序＝脑端候选序（词法相关度序）；把**别场景**的节点排在前、
    本场景的排在后，即 #367 的触发形态。
    """

    def __init__(self, candidates):
        self.candidates = list(candidates)
        self.calls = []

    def call(self, name, arguments):
        self.calls.append((name, dict(arguments)))
        if name == "cg" and arguments.get("op") == "read":
            k = max(0, int(arguments.get("k") or 0))
            return {"meta": {"pre_cap": len(self.candidates), "cap": 500},
                    "results": self.candidates[:k]}
        if name == "stg":
            return {"count": 0, "items": [], "kept": 0, "truncated": False}
        return {}


#: 改前算法（issue #365 形态）：固定一页、不看 `truncated`。
def _legacy_state_map(client):
    resp = client.call("stg", {"op": "state_chain", "limit": 500})
    m = {}
    for u in (resp.get("items") or []):
        if u.get("slot") == STATE_SLOT and u.get("state") == "active":
            m[str(u.get("subject"))] = u.get("value")
    return m


#: 改前算法（issue #367 形态）：一页 top-k 再按 tag 过滤。
def _legacy_get_nodes_by_tag(client, tag, limit):
    resp = client.call("cg", {"op": "read", "query": "场景实体",
                              "k": max(1, int(limit))})
    out = []
    for item in (resp.get("results") or resp.get("items") or []):
        node = item.get("node") or item
        fm = node.get("frontmatter") or item.get("frontmatter") or {}
        if tag in list(fm.get("tags") or []):
            out.append(str(node.get("id") or fm.get("id") or ""))
    return out


# ---------------------------------------------------------------------------
# A1/A2 · #365
# ---------------------------------------------------------------------------

#: 500 个无关主体占满第一页 + 末位一个目标主体（第 501 位，超出 `_STATE_PAGE`）。
_TARGET_ID = "led_target"
_TARGET_STATE = "shy"


def _overflow_client():
    units = [_unit("led_%03d" % i, "v%d" % i) for i in range(_STATE_PAGE)]
    units.append(_unit(_TARGET_ID, _TARGET_STATE))
    return _StateChainClient(units, [_node(_TARGET_ID, "kitchen", "肥鱼")])


def test_a1_state_map_recovers_subject_beyond_first_page():
    """A1 · #365 修复后：目标主体排在投影第 501 位、全量页被截断时仍取回其状态。"""
    client = _overflow_client()
    # 桩件确实处于截断形态（否则本条不构成 #365 场景）
    full = client.call("stg", {"op": "state_chain", "slot": STATE_SLOT,
                               "limit": _STATE_PAGE})
    assert full["truncated"] is True and full["count"] == _STATE_PAGE + 1, full
    assert all(u["subject"] != _TARGET_ID for u in full["items"]), \
        "目标主体不该出现在第一页（否则本桩件不构成 #365 场景）"

    nodes = BrainStore(client).get_nodes_by_tag("kitchen", limit=5)
    assert len(nodes) == 1 and nodes[0].id == _TARGET_ID, nodes
    assert nodes[0].state_attributes == {"state": _TARGET_STATE}, nodes[0]


def test_a2_legacy_state_map_has_teeth():
    """A2 · #365 有牙自证：同桩件走改前算法必得 None（守卫不是空转）。"""
    client = _overflow_client()
    m = _legacy_state_map(client)
    assert _TARGET_ID not in m, \
        "改前算法竟取回了第 501 位主体——桩件未复刻 #365 形态（守卫无牙）"
    assert len(m) == _STATE_PAGE, len(m)


# ---------------------------------------------------------------------------
# A3/A4/A5/A6 · #367
# ---------------------------------------------------------------------------


def _crowding_client(n_other=25, n_mine=5, cap_overflow=False):
    cands = [_node("other%02d" % i, "roomB", "box%d" % i) for i in range(n_other)]
    cands += [_node("mine%d" % i, "roomA", "apple%d" % i) for i in range(n_mine)]
    if cap_overflow:
        # 本场景节点被挤到脑端候选上限（500）之外：另一场景 >500 条。
        cands = [_node("other%03d" % i, "roomB", "box%d" % i) for i in range(600)]
        cands += [_node("mine0", "roomA", "apple0")]
    return _CrowdingClient(cands)


def test_a3_crowded_scene_still_returns_all_mine():
    """A3 · #367 修复后：另一场景占满 top-k 时本场景实体仍全部返回。"""
    client = _crowding_client()
    got = [n.id for n in BrainStore(client).get_nodes_by_tag("roomA", limit=5)]
    assert got == ["mine%d" % i for i in range(5)], got


def test_a4_legacy_get_nodes_has_teeth():
    """A4 · #367 有牙自证：同桩件走改前算法（一页 + 过滤）必得 0。"""
    client = _crowding_client()
    got = _legacy_get_nodes_by_tag(client, "roomA", 5)
    assert got == [], "改前算法竟取回 roomA——桩件未复刻 #367 形态（守卫无牙）"


def test_a5_candidate_face_truncated_raises_not_silent():
    """A5 · #367 候选面被脑端上限截断、limit 取不满 ⇒ 抛 BrainError（不静默）。"""
    client = _crowding_client(cap_overflow=True)
    try:
        BrainStore(client).get_nodes_by_tag("roomA", limit=5)
    except BrainError as exc:
        assert "截断" in str(exc), exc
    else:
        raise AssertionError(
            "#367：候选面被脑端上限截断、limit 取不满却静默返回残缺世界")


def test_a6_candidate_face_exhausted_returns_without_raising():
    """A6 · 反向腿（防修死）：候选面已取尽且确实不足 limit ⇒ 正常返回、不抛。"""
    client = _crowding_client(n_other=1, n_mine=2)
    got = [n.id for n in BrainStore(client).get_nodes_by_tag("roomA", limit=5)]
    assert got == ["mine0", "mine1"], got


# ---------------------------------------------------------------------------
# 手动入口
# ---------------------------------------------------------------------------

_PASS: list = []
_FAIL: list = []


def main() -> int:
    tests = [
        test_a1_state_map_recovers_subject_beyond_first_page,
        test_a2_legacy_state_map_has_teeth,
        test_a3_crowded_scene_still_returns_all_mine,
        test_a4_legacy_get_nodes_has_teeth,
        test_a5_candidate_face_truncated_raises_not_silent,
        test_a6_candidate_face_exhausted_returns_without_raising,
    ]
    for t in tests:
        try:
            t()
            _PASS.append(t.__name__)
            print("  PASS " + t.__name__)
        except AssertionError as exc:
            _FAIL.append((t.__name__, str(exc)))
            print("  FAIL %s ← %s" % (t.__name__, exc))
        except Exception as exc:  # noqa: BLE001
            _FAIL.append((t.__name__, repr(exc)))
            print("  ERROR %s ← %r" % (t.__name__, exc))
    print()
    print(f"===== SUMMARY {len(_PASS)}/{len(_PASS) + len(_FAIL)} 通过 =====")
    if _FAIL:
        print("FAILS:", _FAIL)
        return 1
    print("VERDICT=PASS（#365 状态投影超页仍取回 + #367 候选面取全再过滤；"
          "两件各有改前算法有牙自证）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
