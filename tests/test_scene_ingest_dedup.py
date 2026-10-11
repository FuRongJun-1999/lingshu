#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_scene_ingest_dedup —— 场景导入的新观测不得被 M5 去重吞掉（回归，隔离库）。

缺陷（lingshu#326）：`scene_model.ingest_scene` 走
`engine.add_perception(content, ...)` 时**未传** `skip_dedup`，于是 M5 去重
（中文二元组 Jaccard ≥ 阈值 0.85）命中相似旧节点后直接 `return best`
（core.py:2367~2371）——不新建节点、只提升旧节点。调用方 `ingest_scene`
却按返回的节点 id 继续记账（scene_model.py:321 的
`UPDATE nodes SET state_attributes=? WHERE id=node.id`），结果：

  - 同一实体的连续观测（如移动跟踪：坐标由 (0,0.85,5) 变到 (0.1,0.85,5)，
    状态不变；两行内容 Jaccard = 0.9143 ≥ 阈值 0.85）被短路成同一条旧节点；
  - `written` 列表里出现**重复 id**，新观测的坐标静默丢失；
  - 世界重建回落到旧坐标——观测被吞，无异常、无日志。

（注：若两次观测**状态也不同**，内容 Jaccard 降至 0.816 < 0.85，反而不触发
去重——所以缺陷只在「坐标变、状态不变」的相似观测上发作，更隐蔽。）

修法：场景导入是「主动沉淀类写入」，每行是带独立空间锚点/实体身份的观测，
须保留自己的节点身份 —— `ingest_scene` 显式传 `skip_dedup=True`
（判据来源：core.py:2348 `skip_dedup` 的引入理由 v1.26c「主动沉淀类写入需要
独立节点身份」；同款先例 core.py:4925 `subgraph_replace` 亦传 skip_dedup=True）。

断言全部落在**确定性**事实上（`:memory:` 隔离库，不落盘）；不依赖
`load_world_from_memory` 对同 importance 节点的排序（那受 last_access 并列影响，
是另一处独立的排序脆弱性，不在本件范围）：
  G1 同一实体两次相似观测 ⇒ 返回 2 个**互异** id（改前：两个相同 id）。
  G2 两次观测都真落库 ⇒ 知识层 2 条节点（改前：1 条）。
  G3 两条观测内容并存、各保各的坐标（改前：新观测整条缺失，只剩旧坐标）。
  G4 每个返回 id 都能在库中取到节点（改前：两个 id 指向同一节点）。
  G5 对照组：不同实体的场景行仍各成节点（非相似内容，改前也过——防口径误判）。
  G6 范围控制：`engine.add_perception` **不传** skip_dedup 时 M5 去重仍生效
     （证明本修只放开场景导入这条路，未全局关掉去重）。

运行（lingshu 仓根）：python -X utf8 tests/test_scene_ingest_dedup.py
退出码：0 = 全过；1 = 有断言失败。
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.core.core import MemoryLayer, SpacetimeMemoryEngine  # noqa: E402
from lingshu.world.scene_model import ingest_scene  # noqa: E402

_PASS = []
_FAIL = []


def ok(cond, msg, extra=""):
    (_PASS if cond else _FAIL).append(msg)
    print(("  PASS " if cond else "  FAIL ") + msg
          + (("  ← " + str(extra)) if (extra and not cond) else ""))
    return bool(cond)


class _Agent:
    """scene_model 消费面：同时提供 `.store` 与 `.engine`。"""

    def __init__(self, engine):
        self.engine = engine
        self.store = engine.store


def _nodes(engine):
    return engine.store.query_nodes(layer=MemoryLayer.KNOWLEDGE, limit=50)


def main() -> int:
    print("== test_scene_ingest_dedup ==")

    # 同一实体连续观测：坐标移动、状态不变；两行内容 Jaccard=0.9143 ≥ 0.85
    e = SpacetimeMemoryEngine(":memory:")
    agent = _Agent(e)
    desc = ("肥鱼|fatfish|0,0.85,5|shy\n"
            "肥鱼|fatfish|0.1,0.85,5|shy")
    ids = ingest_scene(agent, desc)

    print("[G1] 同一实体两次相似观测 ⇒ 两个互异节点 id")
    ok(len(ids) == 2, "G1a ingest 返回 2 个 id", ids)
    ok(len(set(ids)) == 2, "G1b 两个 id 互异（改前：去重短路成同一 id）", ids)

    print("[G2] 两次观测都真落库")
    nodes = _nodes(e)
    ok(len(nodes) == 2, "G2 知识层节点数 = 2（改前：1）", [n.id for n in nodes])

    print("[G3] 两条观测内容并存、各保各的坐标")
    contents = sorted(n.content for n in nodes)
    ok(any("(0.0,0.85,5.0)" in c for c in contents),
       "G3a 旧观测 (0.0,0.85,5.0) 仍在库中", contents)
    ok(any("(0.1,0.85,5.0)" in c for c in contents),
       "G3b 新观测 (0.1,0.85,5.0) 独立成节点（改前整条缺失）", contents)

    print("[G4] 每个返回 id 都能在库中解析到节点（无幽灵 id）")
    got = {n.id for n in nodes}
    ok(set(ids) <= got,
       "G4 返回 id 均可解析（改前：重复 id 只解析到同 1 节点）",
       (ids, sorted(got)))

    print("[G5] 对照组：不同实体各成节点")
    e2 = SpacetimeMemoryEngine(":memory:")
    ids2 = ingest_scene(_Agent(e2),
                        "肥鱼|fatfish|0,0.85,5|shy\n桌子|table|1.5,0.45,6|neutral")
    ok(len(set(ids2)) == 2, "G5 不同实体 ⇒ 2 个互异 id", ids2)

    print("[G6] 范围控制：不传 skip_dedup 时 M5 去重仍生效")
    e3 = SpacetimeMemoryEngine(":memory:")
    # 夹具对必须避开 #29②/#142 的「实质分歧」判据（否定标记有无 / 阿拉伯数字
    # 字面量不同 ⇒ 另立节点）。原夹具用的是坐标 (0.0,…) vs (0.1,…) —— 恰是
    # **数值更正**，在 #29 落地后被**刻意判为分歧并另立节点**；继续拿它测
    # 「去重仍生效」就变成在测「#29 不生效」，与本组判据（只放开场景导入这一
    # 条路、未全局关去重）无关。故改用「措辞不同但数字与否定标记全同」的一对
    # （实测 Jaccard 0.9143 ≥ 阈值 0.85、_assertion_divergence 为空 ⇒ 仍判重复）。
    a = e3.add_perception("场景实体 肥鱼（fatfish）位于 (0.0,0.85,5.0)，状态 shy",
                          importance=0.6)
    b = e3.add_perception("场景实体 肥鱼（fatfish）位于 (0.0,0.85,5.0)，状态为 shy",
                          importance=0.6)
    ok(a.id == b.id and len(_nodes(e3)) == 1,
       "G6 缺省去重路径仍短路（未全局关去重）",
       (a.id, b.id, [n.id for n in _nodes(e3)]))

    print("[G7] skip_dedup=True 自身必须被独立锁住（不得被 #29 分歧判据代偿）")
    # 覆盖重叠的现场：G1–G5 用「坐标变了」的观测对——#29② 落地后，**数值更正
    # 本身就会被判为实质分歧并另立节点**，于是 G1–G5 即使把 scene_model 的
    # `skip_dedup=True` 抽掉也照样全绿（实测：skip_dedup=False 单变量 ⇒ 8/8 通过）。
    # ⇒ 那几组对 `skip_dedup=True` 已**不再有判别力**（两套机制任一生效即通过）。
    # 本组用**逐字相同**的场景行导入两次：内容全同 ⇒ 无分歧可判 ⇒ 只有
    # `skip_dedup=True` 能让两行各成节点；抽掉它必红（实测单变量即 1 failed）。
    e4 = SpacetimeMemoryEngine(":memory:")
    same_line = "肥鱼|fatfish|0,0.85,5|shy"
    ids4 = ingest_scene(_Agent(e4), same_line + "\n" + same_line)
    ok(len(set(ids4)) == 2 and len(_nodes(e4)) == 2,
       "G7 逐字相同的两行场景导入仍各成节点（skip_dedup=True 生效，未被分歧判据代偿）",
       (ids4, [n.id for n in _nodes(e4)]))

    print()
    print("-- %d passed, %d failed --" % (len(_PASS), len(_FAIL)))
    print("VERDICT=%s" % ("PASS" if not _FAIL else "FAIL"))
    return 1 if _FAIL else 0


def test_scene_ingest_dedup():
    """pytest 入口：场景导入新观测不被 M5 去重吞掉。"""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
