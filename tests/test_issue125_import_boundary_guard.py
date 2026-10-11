# -*- coding: utf-8 -*-
"""#125 import_all 是无密钥的全权写入通道、JSON 键未校验 —— 守卫（lingshu/core/core.py）

缺陷（lingshu issue #125；分诊表 `triage_lingshu_core_front.md` 行 125 判定成立）：
  `SpacetimeMemoryEngine.import_all` 绕过 `LayeredStore.add_node`（直接
  `INSERT OR REPLACE`）⇒ `add_node` 的角色闸、`adjudicate_promotion` 的密钥闸
  全被跳过；且列名取自备份 JSON 的键逐字、经 f-string 拼进 SQL
  （`cols = list(rows[0].keys())` / `col_sql = ",".join(cols)`）⇒
  ①无密钥即可改写不可变层（anchor/structure/self）行原文；
  ②以任意列名（SQLite 标识符不可参数化）直写。
  事实边界（`docs/plans/治理与写入边界_修复设计_v0.1.md` §6）：`table` 取自
  固定清单 `M13_TABLES`（**非**注入面）；注入面是**列名**。

判据来源（修法）：
  · `docs/plans/待裁清单_v0.1.md` **C-6**（`#125` 不可变层覆盖 → **需密钥放行**；
    ★设计者 2026-10-09 授权按推荐执行，硬配套＝显式标注＋强制事后复核＋审计只追加，
    措辞改「不可**静默**篡改」）/ **D-06**（需密钥放行）/ **D-09**（`PRAGMA
    table_info` 白名单＋**拒绝**非法键）。
  · 形状见 `docs/plans/治理与写入边界_修复设计_v0.1.md` §6「修法形状」
    （列名 `PRAGMA table_info` 求交；授权闸同 `adjudicate_promotion` 款 fail-closed）。

断言组（抽掉任一修复即红）：
  A1 篡改备份（把既有锚点行改写为 structure 层）无密钥 ⇒ `PermissionError`。
  A2 被拒后**零落库**：目标库旧行 layer/content 逐字不变、节点数不变。
  A3 新增（库里无该 id）不可变层行、无密钥 ⇒ 同样拒（取更严口径，见 docstring）。
  B  错误密钥 ⇒ 拒。
  C1 正确密钥 ⇒ 照旧写 structure 层（行为未被废掉）。
  C2 显式标注：返回值 `requires_review is True` 且 `immutable_rows` 含该 id。
  C3 审计只追加：`action_logs` 新增一行（action_type=import_immutable_override）。
  D1 非法列名键（标识符注入形状）被**丢弃**、计数上报 `discarded_keys`，不拼进 SQL。
  D2 注入串未被执行（`nodes` 表仍在、库结构未变）。
  E  防误杀：知识层备份无密钥往返照旧通过（`requires_review is False`、无丢弃）。

运行（仓根）：python -X utf8 -m pytest tests/test_issue125_import_boundary_guard.py -q --no-header
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.core.core import (  # noqa: E402
    ConditionSpace, MemoryLayer, SpacetimeMemoryEngine,
)

_CS = ConditionSpace("守卫", "守卫", (time.time(), time.time() + 3600), "运行中").to_json()


def _node_row(nid, layer, content):
    return {"id": nid, "content": content, "modality": "text", "tags": "[]",
            "spatial_coordinates": "{}", "temporal_coordinate": time.time(),
            "condition_space": _CS, "importance": 0.5, "confidence": 0.5,
            "layer": layer, "access_count": 0, "last_access": None,
            "created_at": time.time(), "semantic_coordinates": "{}",
            "state_attributes": "{}", "entity_id": None}


def _write_backup(nodes):
    path = os.path.join(tempfile.mkdtemp(), "backup.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"meta": {"version": "v1.6"}, "nodes": nodes}, f, ensure_ascii=False)
    return path


def _set_key(monkeypatch, val):
    if val is None:
        monkeypatch.delenv("AEIS_DESIGNER_KEY", raising=False)
    else:
        monkeypatch.setenv("AEIS_DESIGNER_KEY", val)


def test_a1_a2_a3_keyless_immutable_overwrite_rejected_without_write(monkeypatch):
    """A：无密钥改写/新增不可变层行 ⇒ 拒，且零落库。"""
    _set_key(monkeypatch, None)
    e = SpacetimeMemoryEngine(":memory:")
    anchor = e.set_anchor("锚点原文不可静默篡改")
    before = e.store.get_stats()

    # A1：篡改——同 id 声明为 structure、内容改写成「被篡改」
    tampered = _write_backup([_node_row(anchor.id, "structure", "被篡改")])
    try:
        e.import_all(tampered)
        raise AssertionError("A1 无密钥导入不可变层行未被拒（旧缺陷行为）")
    except PermissionError:
        pass

    # A2：零落库（旧行逐字不变、节点数不变）
    got = e.store.get_node(anchor.id)
    assert got is not None and got.layer is MemoryLayer.ANCHOR, \
        "A2 旧行 layer 被改：%s" % (got.layer.value if got else None)
    assert got.content == "锚点原文不可静默篡改", "A2 旧行内容被改写：%r" % (got.content,)
    assert e.store.get_stats() == before, \
        "A2 被拒后仍有落库：%r -> %r" % (before, e.store.get_stats())

    # A3：新增不可变层行（库里无该 id）同样拒
    fresh = _write_backup([_node_row("struct_new_1", "structure", "新结构行")])
    try:
        e.import_all(fresh)
        raise AssertionError("A3 无密钥新增不可变层行未被拒")
    except PermissionError:
        pass
    assert e.store.get_node("struct_new_1") is None, "A3 被拒行仍落库"


def test_b_wrong_key_rejected(monkeypatch):
    """B：错误密钥 ⇒ 拒（fail-closed）。"""
    _set_key(monkeypatch, "RIGHT-KEY")
    e = SpacetimeMemoryEngine(":memory:")
    anchor = e.set_anchor("锚点")
    tampered = _write_backup([_node_row(anchor.id, "structure", "被篡改")])
    try:
        e.import_all(tampered, designer_key="WRONG-KEY")
        raise AssertionError("B 错误密钥未被拒")
    except PermissionError:
        pass
    assert e.store.get_node(anchor.id).content == "锚点", "B 被拒后旧行仍被改"


def test_c1_c2_c3_keyed_import_allowed_labeled_and_audited(monkeypatch):
    """C：正确密钥 ⇒ 照旧生效 + 显式标注 + 审计只追加。"""
    _set_key(monkeypatch, "RIGHT-KEY")
    e = SpacetimeMemoryEngine(":memory:")
    anchor = e.set_anchor("锚点原文")
    logs_before = e.store.conn.execute("SELECT COUNT(*) FROM action_logs").fetchone()[0]

    tampered = _write_backup([_node_row(anchor.id, "structure", "经密钥恢复的内容")])
    ret = e.import_all(tampered, designer_key="RIGHT-KEY")

    # C1：写入照旧生效（未把灾备面废掉）
    got = e.store.get_node(anchor.id)
    assert got.layer is MemoryLayer.STRUCTURE, "C1 未写入：%s" % got.layer.value
    assert got.content == "经密钥恢复的内容", "C1 内容未恢复：%r" % (got.content,)

    # C2：显式标注
    assert ret.get("requires_review") is True, "C2 未显式标注需复核：%r" % (ret,)
    assert anchor.id in ret.get("immutable_rows", []), \
        "C2 immutable_rows 未列出被写行：%r" % (ret.get("immutable_rows"),)

    # C3：审计只追加
    logs_after = e.store.conn.execute("SELECT COUNT(*) FROM action_logs").fetchone()[0]
    assert logs_after == logs_before + 1, \
        "C3 审计未新增一行：%r -> %r" % (logs_before, logs_after)
    row = e.store.conn.execute(
        "SELECT action_type FROM action_logs ORDER BY id DESC LIMIT 1").fetchone()
    assert row[0] == "import_immutable_override", "C3 审计类型不符：%r" % (row[0],)


def test_d_illegal_column_key_discarded(monkeypatch):
    """D：非法列名键被丢弃（不拼进 SQL），注入串不执行。"""
    _set_key(monkeypatch, None)
    e = SpacetimeMemoryEngine(":memory:")
    evil = "id) VALUES (NULL); DROP TABLE nodes; --"
    row = _node_row("kn_1", "knowledge", "正常知识行")
    row[evil] = "x"                       # 标识符注入形状的非法键
    path = _write_backup([row])

    ret = e.import_all(path)              # 知识层 ⇒ 无需密钥，不抛

    # D1：非法键被丢弃并计数上报
    assert evil in ret.get("discarded_keys", {}).get("nodes", []), \
        "D1 非法键未被丢弃上报：%r" % (ret.get("discarded_keys"),)
    # D2：注入串未执行（nodes 表仍在、库结构未变）
    names = {r[0] for r in e.store.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert "nodes" in names, "D2 注入串被执行（nodes 表消失）"
    # 合法列照旧导入
    got = e.store.get_node("kn_1")
    assert got is not None and got.content == "正常知识行", "D1 合法列未导入"


def test_e_knowledge_roundtrip_unaffected(monkeypatch):
    """E：防误杀——知识层备份无密钥往返照旧通过。"""
    _set_key(monkeypatch, None)
    src = SpacetimeMemoryEngine(":memory:")
    src.add_perception("往返测试记忆", skip_dedup=True)
    path = os.path.join(tempfile.mkdtemp(), "kb.json")
    src.export_all(path)

    dst = SpacetimeMemoryEngine(":memory:")
    ret = dst.import_all(path)
    assert ret["imported"]["nodes"] == 1, "E 知识层往返节点数不符：%r" % (ret["imported"],)
    assert ret.get("requires_review") is False, "E 知识层被误标需复核：%r" % (ret,)
    assert ret.get("discarded_keys") == {}, "E 合法备份被误报丢弃键：%r" % (ret,)
