#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_state_write_observable —— 状态写入失败必须可观测（回归，隔离库）。

背景：`ingest_scene` 曾把状态写入整条包在 `except Exception: pass` 里，
`_ConnShim.execute` 又丢弃 `record_state` 的返回值。于是"节点正文写 shy、
重建世界给 neutral"可以全程无异常、无日志（lingshu#28）。

本件断言修复后的三条行为：
  T1 正常路径：状态落账 1 行、重建 shy，且**不**发警告（不误报）。
  T2 台账不可写：ingest 仍不抛（向后兼容），但**发 RuntimeWarning**，
     台账 0 行、重建 neutral —— 失败可见。
  T3 空状态（`名称|类别|坐标|`）：属"无状态可写"，不报警、重建 neutral。

前置：环境变量 `MDCG_BRAIN_PYTHONPATH` = 脑包所在目录（含 `md_cg/`）。
运行（lingshu 仓根）：python -X utf8 tests/test_state_write_observable.py
退出码：0 = 全过；1 = 有断言失败；缺前置 = SKIP（0）。
"""
from __future__ import annotations

import os
import shutil
import stat
import sys
import tempfile
import warnings

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.world.brain_store import connect  # noqa: E402
from lingshu.world.scene_model import (  # noqa: E402
    ingest_scene, load_world_from_memory)

_PASS = []
_FAIL = []

#: 隔离身份三开关（脑侧 legacy 形态；非真实令牌）
_LEGACY_ID = {"MDCG_LEGACY_ENV_AUTH": "1", "MDCG_CAN_ADMIN": "1",
              "MDCG_LEGACY_ENV_ADMIN": "1"}

_LEDGER = "_state_events.jsonl"


def ok(cond, msg, extra=""):
    (_PASS if cond else _FAIL).append(msg)
    print(("  PASS " if cond else "  FAIL ") + msg
          + (("  ← " + str(extra)) if (extra and not cond) else ""))


def _ledger_rows(path: str) -> int:
    if not os.path.isfile(path):
        return 0
    with open(path, encoding="utf-8") as fh:
        return sum(1 for ln in fh if ln.strip())


def _scenario(desc: str, pythonpath: str, readonly_ledger: bool = False) -> dict:
    """跑一条场景描述，返回 {raised, ids, warnings, rows, state, content}。"""
    root = tempfile.mkdtemp(prefix="lingshu_state_obs_")
    assert root.startswith(tempfile.gettempdir()), "隔离根必须是临时目录（fail-closed）"
    ledger = os.path.join(root, _LEDGER)
    if readonly_ledger:
        open(ledger, "w", encoding="utf-8").close()
        os.chmod(ledger, stat.S_IREAD)
    agent = connect(root=root, pythonpath=pythonpath, extra_env=dict(_LEGACY_ID))
    out = {"raised": None, "ids": [], "warnings": [], "rows": 0,
           "state": None, "content": ""}
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                out["ids"] = [str(i) for i in ingest_scene(agent, desc)]
            except Exception as exc:                              # noqa: BLE001
                out["raised"] = "%s: %s" % (type(exc).__name__, exc)
        out["warnings"] = [str(w.message) for w in caught
                           if issubclass(w.category, RuntimeWarning)]
        out["rows"] = _ledger_rows(ledger)
        wm = load_world_from_memory(agent.store)
        out["state"] = wm.entities["肥鱼"].state if "肥鱼" in wm.entities else None
        if out["ids"]:
            r = agent.store.client.call("cg", {"op": "read",
                                              "query": "场景实体", "limit": 50})
            for item in (r.get("results") or r.get("items") or []):
                node = item.get("node") or item
                nid = str(node.get("id") or (node.get("frontmatter") or {}).get("id") or "")
                if nid == out["ids"][0]:
                    out["content"] = str(node.get("content") or "")
                    break
    finally:
        try:
            agent.store.client.close()
        finally:
            if os.path.isfile(ledger):
                os.chmod(ledger, stat.S_IWRITE | stat.S_IREAD)
            shutil.rmtree(root, ignore_errors=True)
    return out


def main() -> int:
    pythonpath = os.environ.get("MDCG_BRAIN_PYTHONPATH")
    if not pythonpath:
        print("SKIP：未设 MDCG_BRAIN_PYTHONPATH（脑包目录）——见文件头前置说明")
        return 0

    healthy = _scenario("肥鱼|fatfish|0,0.85,5|shy", pythonpath)
    broken = _scenario("肥鱼|fatfish|0,0.85,5|shy", pythonpath, readonly_ledger=True)
    empty = _scenario("肥鱼|fatfish|0,0.85,5|", pythonpath)

    ok(healthy["raised"] is None and len(healthy["ids"]) == 1,
       "T1a 正常路径 ingest 无异常", healthy["raised"])
    ok(healthy["rows"] == 1 and healthy["state"] == "shy",
       "T1b 正常路径状态落账且重建为 shy",
       (healthy["rows"], healthy["state"]))
    ok(not healthy["warnings"],
       "T1c 正常路径不误报警告", healthy["warnings"])

    ok(broken["raised"] is None and len(broken["ids"]) == 1,
       "T2a 台账不可写时 ingest 仍不抛（向后兼容）", broken["raised"])
    ok(any("未写入脑侧台账" in w for w in broken["warnings"]),
       "T2b 台账不可写时发出 RuntimeWarning", broken["warnings"])
    ok(broken["rows"] == 0 and broken["state"] == "neutral",
       "T2c 台账 0 行、重建回落 neutral（失败已可见）",
       (broken["rows"], broken["state"]))
    ok("状态 shy" in broken["content"],
       "T2d 节点正文仍写 shy —— 正是需要被暴露的那处矛盾", broken["content"])

    ok(empty["raised"] is None and not empty["warnings"],
       "T3a 空状态不报警（无状态可写，非写失败）", empty["warnings"])
    ok(empty["rows"] == 0 and empty["state"] == "neutral",
       "T3b 空状态重建 neutral", (empty["rows"], empty["state"]))

    print()
    print(f"===== SUMMARY {len(_PASS)}/{len(_PASS) + len(_FAIL)} 通过 =====")
    if _FAIL:
        print("FAILS:", _FAIL)
        return 1
    print("VERDICT=PASS（状态写入失败可观测；正常/空状态不误报）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
