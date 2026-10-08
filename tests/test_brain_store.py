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
    connect, ingest_scene_to_brain, load_world_from_brain)

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

        # ⑥ ★ 读取条数契约（此前无断言 —— 正是 issue #2 长期潜伏的原因）
        #    脑端 cg(op=read) 的条数参数是 k，不是 limit。适配器若发错键名，
        #    所有上限会静默回落到脑端默认 20 ⇒ >20 实体时静默丢实体。
        #    这里显式写入 N(=25) > 20 个实体，验证「要多少给多少」。
        n_big = 25
        big_desc = "\n".join(f"pad{i}|pad|{i},0,9|neutral" for i in range(n_big))
        big_ids = ingest_scene_to_brain(agent, big_desc)
        got_big = len(agent.store.get_nodes_by_tag("spatial", limit=200))
        ok(len(big_ids) == n_big, "S10a 写入 25 个实体", len(big_ids))
        ok(got_big >= n_big,
           "S10b 读取条数契约：limit=200 应至少返回 25（>默认 20）", got_big)

        # 上限必须被遵守：小 limit 不能仍返回一大堆
        got_one = len(agent.store.get_nodes_by_tag("spatial", limit=1))
        ok(got_one == 1, "S10c 读取上限被遵守：limit=1 应恰返回 1", got_one)

        # 世界重建不应因上限静默丢实体
        wm_big = load_world_from_brain(agent)
        missing_big = [f"pad{i}" for i in range(n_big)
                       if f"pad{i}" not in wm_big.entities]
        ok(not missing_big,
           "S10d 世界重建不丢实体（无 limit 截断）", missing_big)
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
