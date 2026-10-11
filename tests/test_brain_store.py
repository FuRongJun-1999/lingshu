#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_brain_store —— M1/M2 最小闭环端到端（身侧视角，隔离库）。

对端：`dsh-memory`（脑）经 MCP stdio；本件跑的是 `lingshu.world.scene_model`
**零改动**对接面（`ingest_scene` / `load_world_from_memory`）。

前置（章程六：本仓不写本机路径字面量）：
  环境变量 `MDCG_BRAIN_PYTHONPATH` = 脑包所在目录（须含 `md_cg/` 与仓根 `utf8_boot.py`）。

运行（lingshu 仓根）：python -X utf8 tests/test_brain_store.py
退出码：0 = 全过；1 = 有断言失败；缺前置 = SKIP（0）。
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.world.brain_store import (  # noqa: E402
    BrainStore, connect, ingest_scene_to_brain, load_world_from_brain)

_PASS = []
_FAIL = []


def ok(cond, msg, extra=""):
    (_PASS if cond else _FAIL).append(msg)
    print(("  PASS " if cond else "  FAIL ") + msg
          + (("  ← " + str(extra)) if (extra and not cond) else ""))


#: 隔离身份三开关（脑侧 legacy 形态；非真实令牌——与对端 body_e2e_smoke 同款）
_LEGACY_ID = {"MDCG_LEGACY_ENV_AUTH": "1", "MDCG_CAN_ADMIN": "1",
              "MDCG_LEGACY_ENV_ADMIN": "1"}

_DESC = ("肥鱼|fatfish|0,0.85,5|shy\n"
         "桌子|table|1.5,0.45,6|neutral\n"
         "树|tree|-2,1,7|neutral")


def main() -> int:
    pythonpath = os.environ.get("MDCG_BRAIN_PYTHONPATH")
    if not pythonpath:
        print("SKIP：未设 MDCG_BRAIN_PYTHONPATH（脑包目录）——见文件头前置说明")
        return 0

    root = tempfile.mkdtemp(prefix="lingshu_brain_")
    assert root.startswith(tempfile.gettempdir()), "隔离根必须是临时目录（fail-closed）"
    agent = connect(root=root, pythonpath=pythonpath, extra_env=dict(_LEGACY_ID))
    try:
        # ① 握手
        ok(bool((agent.store.client.server_info or {}).get("version")),
           "S1 脑侧握手 serverInfo", agent.store.client.server_info)

        # ② M2 写向：ingest_scene 零改动（场景节点 + 状态经 conn 垫片入账）
        ids = ingest_scene_to_brain(agent, _DESC)
        ok(len(ids) == 3, "S2 ingest_scene 写入 3 实体", ids)

        # ③ M1 读向：世界重建（实体/类别/坐标/状态）
        wm = load_world_from_brain(agent)
        ok(set(wm.entities.keys()) == {"肥鱼", "桌子", "树"},
           "S3 世界重建实体集合", sorted(wm.entities.keys()))
        ok(wm.entities["肥鱼"].category == "fatfish",
           "S4 类别经 cat: 标签取回", wm.entities["肥鱼"].category)
        p = wm.entities["肥鱼"].pos
        ok(abs(p[0] - 0.0) < 1e-9 and abs(p[1] - 0.85) < 1e-9 and abs(p[2] - 5.0) < 1e-9,
           "S5 坐标经 spatial.coords3d 直存取回", p)
        ok(wm.entities["肥鱼"].state == "shy",
           "S6 状态经槽位投影取回（事件是源）", wm.entities["肥鱼"].state)
        ok(wm.entities["桌子"].state == "neutral", "S7 第二实体状态", wm.entities["桌子"].state)

        # ④ 状态更新 → 重载现值（投影查询时现算，无缓存）
        ingest_scene_to_brain(agent, "肥鱼|fatfish|0,0.85,5|happy")
        wm2 = load_world_from_brain(agent)
        ok(wm2.entities["肥鱼"].state == "happy",
           "S8 状态更新后重载取新值", wm2.entities["肥鱼"].state)

        # ⑤ tag 过滤在适配器侧（裁定三）：无 spatial 标签的同型内容不进世界
        r = agent.store.client.call("mdcg_remember", {
            "content": "场景实体 幽灵（ghost）位于 (0,0,0)，状态 neutral",
            "layer": "contextual", "gated": False,
            "tags": ["world_model", "ent:幽灵"],
            "spatial": {"coords3d": {"x": 0.0, "y": 0.0, "z": 0.0}}})
        ok(bool(r.get("ok")), "S9a 幽灵节点已写入（对照组）", r)
        nodes = agent.store.get_nodes_by_tag("spatial", limit=50)
        ents = {str(t)[4:] for n in nodes for t in n.tags if str(t).startswith("ent:")}
        ok("幽灵" not in ents, "S9b 无 spatial 标签者被适配器侧过滤", sorted(ents))

        # ⑥ issue #2 回归守卫：读取条数契约——`limit` 须译为脑端条数参数 `k`。
        #    改前读数（真实 MCP stdio；21 实体场景）：limit=200 只回 20、limit=1 也回
        #    20（脑端 `_int_arg(a, "k", 20)` 缺省静默回落），世界缺最后一个实体。
        ents_before = {str(t)[4:]
                       for n in agent.store.get_nodes_by_tag("spatial", limit=200)
                       for t in n.tags if str(t).startswith("ent:")}
        want21 = {"chair%d" % i for i in range(21)}
        desc21 = "\n".join("chair%d|chair|%.1f,0.45,5.0|neutral" % (i, i * 0.1)
                           for i in range(21))
        ids21 = ingest_scene_to_brain(agent, desc21)
        ok(len(ids21) == 21, "S10a 21 实体写入成功（issue #2 场景，对照组）", len(ids21))
        nodes21 = agent.store.get_nodes_by_tag("spatial", limit=200)
        ents21 = {str(t)[4:] for n in nodes21
                  for t in n.tags if str(t).startswith("ent:")}
        ok(ents21 - ents_before == want21,
           "S10b limit=200 ⇒ 新增候选恰为写入的 21 个（脑端 k 生效；改前只回 20）",
           sorted(want21 - ents21))
        got21 = set(load_world_from_brain(agent).entities.keys())
        ok(got21 - ents_before == want21,
           "S10c 世界重建实体集合＝写入集合（增量口径；改前缺末位实体）",
           sorted((want21 | ents_before) - got21))
        n1 = agent.store.get_nodes_by_tag("spatial", limit=1)
        ok(len(n1) <= 1, "S11 limit=1 ⇒ 至多 1 个候选（改前回落 k=20 ⇒ 20 个）", len(n1))

        # ⑦ S12 守卫（PR #3 部位②）：脑端资格判据判 REJECT 的带 tag 节点**不进世界重建**。
        #    构造：六要素齐全 + `# 不适用条件：场景实体`（命中读取情境 seed_query）
        #    ⇒ 脑端 `cg(op=read)` 对该节点回 `state=REJECT`；正文刻意把「场景」「实体」
        #    拆开（「实体坐标登记到场景台账」）以绕过脑端一致性自否定闸，同时保留召回。
        #    只踢 REJECT（不清空）：场景实体节点缺六要素 ⇒ 实测恒为 BLINDSPOT，须保留。
        #    变异鉴别：去掉本仓过滤 ⇒ S12b 红；改回 `(item.get("state") or "")` 的
        #    静默剔除写法 ⇒ S12f 红（缺 state 者被误剔）。
        rj = agent.store.client.call("mdcg_remember", {
            "content": "# 功能名：G7 标定物坐标台账\n"
                       "# 生效条件：任意时刻\n# 子功能：无\n"
                       "# 执行：把标定物 G7 的实体坐标登记到场景台账\n"
                       "# 验证方式：对端复核台账（measurement）\n"
                       "# 不适用条件：场景实体\n",
            "layer": "contextual", "gated": False,
            "tags": ["spatial", "cat:marker", "ent:G7_REJECT"],
            "spatial": {"coords3d": {"x": 9.0, "y": 0.0, "z": 9.0}},
            "verification_basis": "measurement",
            "non_applicable_conditions": ["场景实体"], "consistency": False})
        ok(bool(rj.get("ok")), "S12a REJECT 标定物已写入（对照组）", rj)
        wm_rj = load_world_from_brain(agent)
        ok("G7_REJECT" not in wm_rj.entities,
           "S12b REJECT 资格节点不注入世界重建（PR#3 部位②）",
           sorted(wm_rj.entities.keys()))
        ok("肥鱼" in wm_rj.entities,
           "S12c BLINDSPOT 实体保留（只踢 REJECT，不清空）",
           sorted(wm_rj.entities.keys()))
        ents_off = {str(t)[4:] for n in agent.store.get_nodes_by_tag(
            "spatial", limit=200, accept_states=set())
            for t in n.tags if str(t).startswith("ent:")}
        ok("G7_REJECT" in ents_off,
           "S12d accept_states=set() 关闭过滤 ⇒ REJECT 回候选（旧行为兼容口）",
           sorted(ents_off))
        ents_tight = {str(t)[4:] for n in agent.store.get_nodes_by_tag(
            "spatial", limit=200, accept_states={"ACCEPT"})
            for t in n.tags if str(t).startswith("ent:")}
        ok("G7_REJECT" not in ents_tight and "肥鱼" not in ents_tight,
           "S12e accept_states={'ACCEPT'} 收紧 ⇒ REJECT 与 BLINDSPOT 均出局",
           sorted(ents_tight))

        class _NoStateClient:
            """桩客户端：`cg(op=read)` 回一条 tags 含 spatial 但**无 state 键**的条目。"""

            def call(self, name, arguments):      # noqa: D102 —— 桩件
                if name == "cg":
                    return {"results": [{
                        "reason": "stub", "ref": "mem_stub",
                        "node": {"id": "mem_stub", "content": "无态节点",
                                 "frontmatter": {
                                     "id": "mem_stub",
                                     "tags": ["spatial", "ent:S12_STUB"],
                                     "spatial": {"coords3d": {"x": 1.0,
                                                              "y": 0.0,
                                                              "z": 1.0}}}}}]}
                if name == "stg":
                    return {"items": []}
                return {}

        stub_ids = [n.id for n in BrainStore(_NoStateClient()).get_nodes_by_tag(
            "spatial", limit=5)]
        ok("mem_stub" in stub_ids,
           "S12f 缺 state 者 fail-open 保留（未知≠否决；PR 原写法会静默剔除）",
           sorted(stub_ids))

        # ⑧ S13（#366）跨场景同名实体：两条 ent: 同名、节点 id 不同的观测，状态
        #    各记各的（改前以**实体名**作台账 subject ⇒ 后写盖先写，两条都读到同一个值）。
        #    走真实 legacy 通路（engine.add_perception(entities=[名]) → conn 垫片记账），
        #    以便变异回「实体名作 subject」时确实变红。
        n13a = agent.engine.add_perception(
            "场景实体 同名（S13）位于 (0,0,0)，状态 a", importance=0.6,
            spatial_coordinates={"x": 0.0, "y": 0.0, "z": 0.0},
            tags=["spatial", "cat:object", "ent:S13同名", "world_model"],
            entities=["S13同名"], skip_dedup=True)
        n13b = agent.engine.add_perception(
            "场景实体 同名（S13）位于 (0,0,1)，状态 b", importance=0.6,
            spatial_coordinates={"x": 0.0, "y": 0.0, "z": 1.0},
            tags=["spatial", "cat:object", "ent:S13同名", "world_model"],
            entities=["S13同名"], skip_dedup=True)
        ok(bool(n13a.id) and bool(n13b.id) and n13a.id != n13b.id,
           "S13a 两个同名实体节点已写入（对照组）", (n13a.id, n13b.id))
        agent.store.record_state(n13a.id, "s13_a")
        agent.store.record_state(n13b.id, "s13_b")
        n13 = {n.id: n for n in agent.store.get_nodes_by_tag("spatial", limit=200)}
        ok(n13[n13a.id].state_attributes == {"state": "s13_a"},
           "S13b 同实体名·节点 id A 的状态＝自己记的 s13_a（不被 B 覆盖）",
           n13[n13a.id].state_attributes)
        ok(n13[n13b.id].state_attributes == {"state": "s13_b"},
           "S13c 同实体名·节点 id B 的状态＝自己记的 s13_b（不被 A 覆盖）",
           n13[n13b.id].state_attributes)
    finally:
        try:
            agent.store.client.close()
        finally:
            shutil.rmtree(root, ignore_errors=True)

    print()
    print(f"===== SUMMARY {len(_PASS)}/{len(_PASS) + len(_FAIL)} 通过 =====")
    if _FAIL:
        print("FAILS:", _FAIL)
        return 1
    print("VERDICT=PASS（M1 读向 + M2 写向 最小闭环全过；隔离库）")
    return 0


if __name__ == "__main__":
    sys.exit(main())


def test_brain_store_m1_m2_loop():
    """M1/M2 最小闭环的 pytest 入口（缺前置 MDCG_BRAIN_PYTHONPATH 时 main() 以 SKIP 退出 0）。"""
    assert main() == 0


# ---------------------------------------------------------------------------
# #364 / #366 守卫（纯桩件，**不需要真实脑端**，故无条件运行）
# ---------------------------------------------------------------------------


class _CaptureClient:
    """记录 `tools/call` 入参的桩客户端；只回守卫需要的形态。"""

    def __init__(self, read_items=None, chain_items=None):
        self.calls = []
        self._read_items = read_items or []
        self._chain_items = chain_items or []

    def call(self, name, arguments):
        self.calls.append((name, dict(arguments)))
        if name == "mdcg_remember":
            return {"ok": True, "id": "mem_%d" % len(self.calls)}
        if name == "cg" and arguments.get("op") == "read":
            return {"results": self._read_items}
        if name == "cg" and arguments.get("op") == "state_event":
            return {"ok": True}
        if name == "stg":
            return {"items": self._chain_items}
        return {}


def _node(i, ent, z):
    return {"state": "ACCEPT",
            "node": {"id": i, "content": "场景实体 %s" % ent,
                     "frontmatter": {"id": i, "tags": ["spatial", "ent:%s" % ent],
                                     "spatial": {"coords3d": {"x": 0.0, "y": 0.0,
                                                              "z": z}}}}}


def test_364_add_perception_passes_sensitivity_role_condition_space():
    """#364：密级/来源/条件空间必须原样进 mdcg_remember args（不得被 **_kw 静默吞）。

    判据来源：issue #364 分诊（`.tmp/triage_lingshu_world_back.md:30,68`）——
    改前 `add_perception(..., **_kw)` 把 sensitivity/condition_space 吞掉，
    声明 private 的观测按脑端缺省 internal 落盘。变异：把三参数移回 **_kw ⇒ 红。
    """
    from lingshu.world.brain_store import BrainEngine, BrainStore

    c = _CaptureClient()
    eng = BrainEngine(BrainStore(c))
    eng.add_perception("观测", sensitivity="private", role="user",
                       condition_space={"time_window": ["2026-01-01", "2026-12-31"]})
    args = [a for n, a in c.calls if n == "mdcg_remember"][0]
    assert args.get("sensitivity") == "private", args
    assert args.get("role") == "user", args
    assert args.get("condition_space") == {"time_window": ["2026-01-01",
                                                           "2026-12-31"]}, args


def test_364_add_perception_rejects_unknown_kwargs():
    """#364：未具名的关键字必须 fail-closed 抛错，不得静默丢弃（「也未拒绝」）。

    变异：去掉 `if _kw: raise TypeError` ⇒ 本断言红（拼错的参数被静默吞掉）。
    `skip_dedup` 是 legacy 调用形（scene_model.ingest_scene）透传的名字，须被显式
    接收（脑侧直写不做去重，故为如实记录的无操作）——不得误伤。
    """
    from lingshu.world.brain_store import BrainEngine, BrainStore

    c = _CaptureClient()
    eng = BrainEngine(BrainStore(c))
    eng.add_perception("观测", skip_dedup=True)          # legacy 兼容：不抛
    try:
        eng.add_perception("观测", sensitivty="private")  # 拼错的名字
    except TypeError:
        pass
    else:
        raise AssertionError("未具名关键字被静默吞掉（#364：须 fail-closed 抛 TypeError）")


def test_366_state_ledger_keyed_by_node_id_not_entity_name():
    """#366：状态台账 subject＝节点 id，不是实体名（跨场景同名实体互不覆盖）。

    判据来源：issue #366 分诊（`.tmp/triage_lingshu_world_back.md:32,69`）——
    改前 `record_state` 以实体名作 subject（`_node_meta` 登记的名字）、
    `_state_map`/`get_nodes_by_tag` 以实体名作键，另一场景的同名实体会改掉本
    场景的状态。变异：把 subject 改回实体名 ⇒ 本断言红。
    """
    from lingshu.world.brain_store import BrainEngine, BrainStore

    # 两条**同 ent: 名、不同节点 id** 的观测（走 legacy 通路：engine.add_perception
    # 带 entities=[名]），台账里各记各的；另掺一条以实体名（「肥鱼」）作 subject
    # 的旧形态行——它**不得**被当作节点键。
    c = _CaptureClient(
        read_items=[_node("mem_1", "肥鱼", 5.0), _node("mem_2", "肥鱼", 6.0)],
        chain_items=[{"slot": "状态", "state": "active", "subject": "mem_1",
                      "value": "shy"},
                     {"slot": "状态", "state": "active", "subject": "mem_2",
                      "value": "happy"},
                     {"slot": "状态", "state": "active", "subject": "肥鱼",
                      "value": "angry"}])
    store = BrainStore(c)
    eng = BrainEngine(store)
    n1 = eng.add_perception("观测 A", tags=["spatial", "ent:肥鱼"], entities=["肥鱼"])
    n2 = eng.add_perception("观测 B", tags=["spatial", "ent:肥鱼"], entities=["肥鱼"])
    assert n1.id == "mem_1" and n2.id == "mem_2", (n1, n2)

    nodes = {n.id: n for n in store.get_nodes_by_tag("spatial", limit=10)}
    assert nodes["mem_1"].state_attributes == {"state": "shy"}, nodes["mem_1"]
    assert nodes["mem_2"].state_attributes == {"state": "happy"}, nodes["mem_2"]

    # 写向：record_state 的 subject 必须是节点 id（不是实体名）。
    store.record_state(n1.id, "happy")
    se = [a for n, a in c.calls if n == "cg" and a.get("op") == "state_event"]
    assert se and se[0]["subject"] == "mem_1", se
    assert se[0]["slot"] == "状态", se


def test_366_entity_name_only_ledger_does_not_leak_into_nodes():
    """#366 负向：台账里只有以实体名作 subject 的行时，节点不得据此取到状态。

    （钉住「键已从实体名改为节点 id」——旧形态行不再被认作任何节点的状态。）
    """
    from lingshu.world.brain_store import BrainStore

    c = _CaptureClient(
        read_items=[_node("mem_A", "肥鱼", 5.0)],
        chain_items=[{"slot": "状态", "state": "active", "subject": "肥鱼",
                      "value": "angry"}])
    nodes = {n.id: n for n in BrainStore(c).get_nodes_by_tag("spatial", limit=10)}
    assert nodes["mem_A"].state_attributes is None, nodes["mem_A"]
