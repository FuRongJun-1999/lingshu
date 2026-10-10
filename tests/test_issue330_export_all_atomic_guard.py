# -*- coding: utf-8 -*-
"""#330 export_all 先 "w" 截断再流式写 —— 守卫（lingshu/core/core.py）

缺陷（lingshu issue #330 · [core/M13 导出][数据丢失]）：
  `SpacetimeMemoryEngine.export_all` 旧实现
  `with open(output_path, "w", encoding="utf-8") as f: json.dump(data, f, ...)`
  是**先截断再流式写**（"w" 打开即把目标清零，再逐块序列化）⇒ 写中途失败
  （磁盘满 / 序列化异常 / 进程被杀）留下**空档或半截 JSON**。本方法自称
  「灾备基础」⇒ 一次失败的备份会把**上一次的好备份**一起赔掉（先毁后写）。

修法：同目录临时文件写完 + `os.replace` 换入（同卷原子），失败即清临时件。

判据来源：经验标定（本件 #330）——口径取「临时文件 + os.replace」；仓内无规定
导出落盘原语的理论章节，追不到更早出处。
边界（如实标注）：`os.replace` 只保证「目标要么旧内容、要么完整新内容」，
**不**保证掉电后目录项已落盘（未做目录 fsync）。

断言组（抽掉修复＝还原为 `open(output_path, "w")` 直写 ⇒ A/B 红）：
  A 序列化中途失败 ⇒ 既有旧备份**逐字不变**（旧实现＝已被截断成空/半截）
  B 失败不留 `.tmp-` 残件（清理到位）
  C 防误杀：正常导出照旧落盘、JSON 可解析、返回值口径不变、无残件
  D 覆盖式语义保留：第二次导出替换第一次（不是拒绝覆盖）

运行（仓根）：python -X utf8 -m pytest tests/test_issue330_export_all_atomic_guard.py -q --no-header
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.core import core as core_mod  # noqa: E402
from lingshu.core.core import SpacetimeMemoryEngine  # noqa: E402

OLD = "旧备份：这份内容不可被一次失败的导出毁掉"


def _old_backup(td, name="backup.json"):
    path = os.path.join(td, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(OLD)
    return path


class _BoomDump:
    """把 `json.dump` 换成「先写一段再炸」——模拟流式序列化中途失败。"""

    def __init__(self, real):
        self._real = real

    def __call__(self, obj, fp, **kw):
        fp.write('{"meta": {"partial": ')      # 已经往目标写了一段
        fp.flush()
        raise RuntimeError("模拟导出中途失败（#330 探针）")


def test_a_写中途失败不毁旧档(monkeypatch):
    """A：序列化中途失败 ⇒ 旧备份逐字不变（旧实现＝先截断成空）。"""
    td = tempfile.mkdtemp()
    path = _old_backup(td)
    eng = SpacetimeMemoryEngine(":memory:")
    eng.add_perception("导出探针", skip_dedup=True)

    monkeypatch.setattr(core_mod.json, "dump", _BoomDump(json.dump))
    raised = None
    try:
        eng.export_all(path)
    except RuntimeError as e:
        raised = e
    monkeypatch.undo()

    assert raised is not None, "A 探针无效：中途失败未抛出"
    with open(path, encoding="utf-8") as f:
        got = f.read()
    assert got == OLD, (
        "#330 失败导出毁了旧备份：%r（旧实现＝open(...,'w') 先截断再写）" % (got,))


def test_b_失败不留临时件(monkeypatch):
    """B：失败后同目录不留 `.tmp-` 残件。"""
    td = tempfile.mkdtemp()
    path = _old_backup(td)
    eng = SpacetimeMemoryEngine(":memory:")
    monkeypatch.setattr(core_mod.json, "dump", _BoomDump(json.dump))
    try:
        eng.export_all(path)
    except RuntimeError:
        pass
    monkeypatch.undo()
    leftovers = [n for n in os.listdir(td) if ".tmp-" in n]
    assert leftovers == [], "B 临时文件未清理：%r" % (leftovers,)


def test_c_正常导出照旧(monkeypatch):
    """C：防误杀——正常导出落盘、JSON 可解析、口径不变、无残件。"""
    td = tempfile.mkdtemp()
    path = os.path.join(td, "fresh.json")
    eng = SpacetimeMemoryEngine(":memory:")
    eng.add_perception("正常导出探针", skip_dedup=True)

    res = eng.export_all(path)
    assert os.path.exists(path), "C 产物未落盘"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)                  # 必须是**完整可解析**的 JSON
    assert res["exported_nodes"] == 1, "C exported_nodes 口径变了：%r" % (res,)
    assert res["path"] == path and res["skipped_tables"] == ["entities"], \
        "C 返回值口径变了：%r" % (res,)
    assert "nodes" in data and len(data["nodes"]) == 1, "C 内容不对"
    assert [n for n in os.listdir(td) if ".tmp-" in n] == [], "C 残留临时件"


def test_d_覆盖式语义保留():
    """D：第二次导出替换第一次（原子写不得把覆盖语义变成拒绝/追加）。"""
    td = tempfile.mkdtemp()
    path = os.path.join(td, "over.json")
    eng = SpacetimeMemoryEngine(":memory:")
    eng.add_perception("第一条", skip_dedup=True)
    eng.export_all(path)
    eng.add_perception("第二条", skip_dedup=True)
    res2 = eng.export_all(path)

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert res2["exported_nodes"] == 2, "D 二次导出计数不对：%r" % (res2,)
    assert len(data["nodes"]) == 2, "D 覆盖式语义丢失（内容为 %d 条）" % len(data["nodes"])
    assert [n for n in os.listdir(td) if ".tmp-" in n] == [], "D 残留临时件"
