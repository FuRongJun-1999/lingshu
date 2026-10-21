# -*- coding: utf-8 -*-
"""test_issue789_import_all_bom_guard · import_all 对 BOM 备份的读入守卫（#789）
====================================================================================
缺陷（#789，成立）：``SpacetimeMemoryEngine.import_all`` 用 ``encoding="utf-8"`` 读
备份文件——带 UTF-8 BOM 的备份（Windows 记事本 / PowerShell ``>`` 重定向的常见产物；
#412 族语料同源）首字符读成 ``\ufeff``，``json.load`` 当场抛 ``JSONDecodeError:
Unexpected UTF-8 BOM (decode using utf-8-sig): line 1 column 1 (char 0)`` ⇒ **整次
导入零落库**——本方法自称「恢复/迁移/灾备基础」，恢复入口却被一个字节序标记挡死。

改前读数（本机，Windows / py3.12，main @ 5c530ff）：同一份知识层备份写两份
（``plain.json`` 无 BOM、``bom.json`` 带 BOM），``import_all`` 一成一败——
plain：``imported={'nodes': 1, ...}`` 照常；bom：``JSONDecodeError`` 零落库。

修法：读取改 ``encoding="utf-8-sig"``（无 BOM 文件逐字节等价，BOM 在场时剥掉）。
与 #437（coggraph 四处读取点）、#600（simloop load_priors 第五处）同口径，本处为
灾备读入面。

守卫判据（注入式断言，回退修复即红）：
  G1  BOM 备份与无 BOM 备份**同等导入**（节点逐字落库、导入计数一致）；
  G2  无 BOM 路径零回归（既有行为不变：内容/计数/`requires_review` 逐字）。

判据来源：issue #789 取证 + #437/#600/#412 同族先例（「BOM 使读取面把合法件当
坏件」的失效形态表）；理论章节不涉及编码口径。
运行：python -X utf8 -m pytest tests/test_issue789_import_all_bom_guard.py -q --no-header
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


def _write_backup(path, encoding):
    """同一份备份内容（2 节点 + 1 边）按指定编码落盘；BOM 件首 3 字节须为 EF BB BF。"""
    payload = {"meta": {"version": "v1.6"},
               "nodes": [_node_row("n-bom-k1", "knowledge", "BOM 备份知识节点"),
                         _node_row("n-bom-k2", "knowledge", "BOM 备份知识节点二")],
               "edges": [{"id": "e-bom-1", "source_id": "n-bom-k1",
                          "target_id": "n-bom-k2", "relation_type": "causal",
                          "condition_space": _CS, "confidence": 0.6,
                          "verified": 0, "weight": 1.0, "created_at": time.time(),
                          "last_verified": None, "source_evidence": "extracted"}]}
    with open(path, "w", encoding=encoding, newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False)
    with open(path, "rb") as f:
        head = f.read(3)
    if encoding == "utf-8-sig":
        assert head == b"\xef\xbb\xbf", head
    else:
        assert head != b"\xef\xbb\xbf", head


def _make_pair():
    """同一份备份内容写两份：plain（无 BOM）与 bom（带 BOM）。返回 (plain, bom)。"""
    tmp = tempfile.mkdtemp(prefix="issue789_")
    plain = os.path.join(tmp, "plain.json")
    bom = os.path.join(tmp, "bom.json")
    _write_backup(plain, "utf-8")
    _write_backup(bom, "utf-8-sig")
    return plain, bom


def _import_into_fresh(path):
    """新库导入一份备份 ⇒ (返回体, 库内节点 (id, layer, content) 列表, 边数)。"""
    e = SpacetimeMemoryEngine(":memory:")
    try:
        r = e.import_all(path)
        nodes = [(n.id, n.layer.value, n.content)
                 for n in e.store.query_nodes(limit=10)]
        n_edges = e.store.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        return r, nodes, n_edges
    finally:
        e.close()


def test_g1_bom_backup_imports_equally():
    """G1：BOM 备份必须与无 BOM 备份同等导入（#789：不得 JSONDecodeError 零落库）。"""
    plain, bom = _make_pair()
    r_plain, nodes_plain, edges_plain = _import_into_fresh(plain)
    r_bom, nodes_bom, edges_bom = _import_into_fresh(bom)
    assert nodes_bom == nodes_plain == [
        ("n-bom-k1", "knowledge", "BOM 备份知识节点"),
        ("n-bom-k2", "knowledge", "BOM 备份知识节点二")], (
        "BOM 备份未与无 BOM 备份同等落库（#789）：plain=%r bom=%r——"
        "import_all 备份读取须用 utf-8-sig（BOM 剥除），与 #437/#600 同口径。"
        % (nodes_plain, nodes_bom))
    assert edges_bom == edges_plain == 1, (edges_plain, edges_bom)
    assert r_bom["imported"]["nodes"] == r_plain["imported"]["nodes"] == 2, (
        r_plain["imported"], r_bom["imported"])
    assert r_bom["imported"]["edges"] == r_plain["imported"]["edges"] == 1, (
        r_plain["imported"], r_bom["imported"])


def test_g2_plain_backup_zero_regression():
    """G2（防过修）：无 BOM 备份导入路径零回归。"""
    plain, _bom = _make_pair()
    r, nodes, n_edges = _import_into_fresh(plain)
    assert nodes == [("n-bom-k1", "knowledge", "BOM 备份知识节点"),
                     ("n-bom-k2", "knowledge", "BOM 备份知识节点二")], nodes
    assert n_edges == 1, n_edges
    assert r["imported"]["nodes"] == 2 and r["imported"]["edges"] == 1, r["imported"]
    assert r["requires_review"] is False, r
    assert MemoryLayer.KNOWLEDGE is not None  # layer 值经 query_nodes 还原（正对照）


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q", "--no-header"]))
