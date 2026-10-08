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
import sys
import tempfile

import pytest

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
    _PASS.clear()
    _FAIL.clear()
    pythonpath = os.environ.get("MDCG_BRAIN_PYTHONPATH")
    if not pythonpath:
        print("SKIP：未设 MDCG_BRAIN_PYTHONPATH（脑包目录）——见文件头前置说明")
        return 0

    # Keep memory, identity and auxiliary data inside the same disposable root.
    with tempfile.TemporaryDirectory(prefix="lingshu_brain_") as root:
        extra_env = {
            **_LEGACY_ID,
            "MDCG_AUX_ROOT": os.path.join(root, "identity"),
            "MDCG_DATA_ROOT": os.path.join(root, "data"),
            "MDCG_STATE_ROOT": os.path.join(root, "state"),
            "MDCG_SUSTAIN": "0",
        }
        agent = connect(root=os.path.join(root, "memory"),
                        pythonpath=pythonpath, extra_env=extra_env)
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
        finally:
            agent.store.client.close()

    print()
    print(f"===== SUMMARY {len(_PASS)}/{len(_PASS) + len(_FAIL)} 通过 =====")
    if _FAIL:
        print("FAILS:", _FAIL)
        return 1
    print("VERDICT=PASS（M1 读向 + M2 写向 最小闭环全过；隔离库）")
    return 0


@pytest.mark.integration
def test_scene_memory_roundtrip():
    """Run the existing MCP roundtrip with an explicit external dependency."""
    if not os.environ.get("MDCG_BRAIN_PYTHONPATH"):
        pytest.skip("set MDCG_BRAIN_PYTHONPATH to the dsh-memory package directory")
    assert main() == 0, f"Failed checks: {_FAIL}"


if __name__ == "__main__":
    sys.exit(main())
