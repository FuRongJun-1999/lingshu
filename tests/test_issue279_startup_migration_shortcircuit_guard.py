# -*- coding: utf-8 -*-
"""#279 每次构造都对全表跑 migrate_v17_coordinates —— 守卫（lingshu/core/core.py）

缺陷（lingshu issue #279 · [core/启动·迁移]）：
  `SpacetimeMemoryEngine.__init__`（core.py:2426）**每次构造**都调用
  `migrate_v17_coordinates()`，而旧实现无条件
  `SELECT id, spatial_coordinates, semantic_coordinates FROM nodes` +
  `fetchall()`，再对**每一行**做 `json.loads(spatial_coordinates)`。
  v1.7 之后的库常态是「一行待迁移都没有」，于是每次进程启动（多进程共享库
  下就是每次实例化）都要付一次**全表物化 + 全表 JSON 解析**。

修法：候选行下推到 SQL（`_V17_CANDIDATE_SQL`：`spatial_coordinates LIKE
'%"protocol_%' OR ...`），候选集为空即**零 Python 侧解析**（幂等短路）。
迁移**判据**未动：仍是 Python 侧 `startswith(("protocol_","radical_","neural_"))`；
SQL 只是超集筛（键必以 `"` 起头 ⇒ 假阴性不可能；假阳性只多判一次、不改结果集）。

判据来源：经验标定（本件 #279）——口径取「无候选行 ⇒ 零解析」，无理论章节规定
启动期迁移的短路口径；追不到更早出处。

断言组（抽掉修复＝把 SQL 还原为无条件全表 SELECT ⇒ A/C/D 红）：
  A 短路：50 行干净节点（无候选）⇒ `migrate_v17_coordinates()` 期间对
    `spatial_coordinates` 的 JSON 解析次数 **0**，返回 migrated_nodes == 0
  B 不误跳：真候选行照旧迁移，字段搬移逐字正确、spatial 里的语义键被摘除
  C 构造期：在含 200 行干净节点的**文件库**上构造引擎，无一行坐标被解析
  D 迁完幂等：候选行迁过一轮后，再调用 0 迁移、0 解析、库逐字不变
  E 超集筛边界：嵌套语义键（旧实现也不迁）仍不迁，库不变
  F 防误杀：候选行在场时构造期照旧完成迁移（不是「永久不迁」）
  G 削掉的判别力（如实钉住）：非 JSON 坏行由「构造期崩」改为「静默跳过」

运行（仓根）：python -X utf8 -m pytest tests/test_issue279_startup_migration_shortcircuit_guard.py -q --no-header
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


def _insert_raw(eng, nid, spatial, semantic="{}", content="x"):
    eng.store.conn.execute(
        "INSERT INTO nodes (id, content, layer, spatial_coordinates, semantic_coordinates) "
        "VALUES (?,?,?,?,?)",
        (nid, content, "knowledge", spatial, semantic))
    eng.store.conn.commit()


class _LoadsSpy:
    """包装 `json.loads`：记录每次被解析的实参（用于判「这一行到底被解析了没」）。"""

    def __init__(self):
        self.args = []
        self._real = json.loads

    def __call__(self, s, *a, **k):
        self.args.append(s)
        return self._real(s, *a, **k)

    def count_in(self, values):
        vs = set(values)
        return sum(1 for x in self.args if x in vs)


def test_a_无候选行时零解析(monkeypatch):
    """A：50 行干净节点 ⇒ 迁移期间一次坐标 JSON 解析都不该发生。"""
    eng = SpacetimeMemoryEngine(":memory:")
    clean = []
    for i in range(50):
        sp = json.dumps({"visual": [i, i + 1], "gps": i})
        clean.append(sp)
        _insert_raw(eng, "clean_%d" % i, sp)

    spy = _LoadsSpy()
    monkeypatch.setattr(core_mod.json, "loads", spy)
    ret = eng.migrate_v17_coordinates()
    monkeypatch.undo()

    assert ret["migrated_nodes"] == 0, "A 干净库不该迁移任何行：%r" % (ret,)
    hit = spy.count_in(clean)
    assert hit == 0, (
        "#279 无候选行仍逐行解析坐标：50 行干净节点里 %d 行被 json.loads"
        "（旧缺陷＝全表 fetchall + 逐行解析）" % hit)
    assert spy.args == [], (
        "#279 迁移期间仍有 %d 次 json.loads（短路未生效）" % len(spy.args))


def test_b_候选行照旧迁移():
    """B：真候选行照旧迁移（SQL 只是超集筛，不得改变结果集）。"""
    eng = SpacetimeMemoryEngine(":memory:")
    _insert_raw(eng, "legacy_1", json.dumps(
        {"protocol_alpha": 1, "radical_beta": 2, "neural_gamma": 3, "visual": [1, 2]}))

    ret = eng.migrate_v17_coordinates()
    assert ret["migrated_nodes"] == 1, "B 候选行未被迁移：%r" % (ret,)
    sp = json.loads(eng.store.conn.execute(
        "SELECT spatial_coordinates FROM nodes WHERE id='legacy_1'").fetchone()[0])
    se = json.loads(eng.store.conn.execute(
        "SELECT semantic_coordinates FROM nodes WHERE id='legacy_1'").fetchone()[0])
    assert sp == {"visual": [1, 2]}, "B 语义键未从 spatial 摘除：%r" % (sp,)
    assert se["protocol"]["concept"] == {"alpha": 1}, "B protocol 搬移错：%r" % (se,)
    assert se["radical"] == {"beta": 2}, "B radical 搬移错：%r" % (se,)
    assert se["neural"] == {"gamma": 3}, "B neural 搬移错：%r" % (se,)


def test_c_构造期不再解析每行坐标(monkeypatch):
    """C：含 200 行干净节点的文件库上**构造引擎**，无一行坐标被解析。"""
    td = tempfile.mkdtemp()
    path = os.path.join(td, "startup.db")
    eng = SpacetimeMemoryEngine(path)
    clean = []
    for i in range(200):
        sp = json.dumps({"visual": [i], "n": i})
        clean.append(sp)
        _insert_raw(eng, "row_%d" % i, sp)

    spy = _LoadsSpy()
    monkeypatch.setattr(core_mod.json, "loads", spy)
    SpacetimeMemoryEngine(path)          # 复现「每次构造都跑迁移」
    monkeypatch.undo()

    hit = spy.count_in(clean)
    assert hit == 0, (
        "#279 构造期仍解析节点坐标：200 行干净节点里 %d 行被 json.loads"
        "（旧缺陷＝每次构造全表解析）" % hit)


def test_d_迁移完成后幂等且零解析(monkeypatch):
    """D：迁完一轮后，再调用必须 0 迁移、0 解析、库逐字不变。"""
    eng = SpacetimeMemoryEngine(":memory:")
    _insert_raw(eng, "legacy_3", json.dumps({"protocol_eps": 4, "visual": [1]}))
    first = eng.migrate_v17_coordinates()
    assert first["migrated_nodes"] == 1, "D 前置：候选行未被迁移：%r" % (first,)
    before = [tuple(r) for r in eng.store.conn.execute(
        "SELECT id, spatial_coordinates, semantic_coordinates FROM nodes ORDER BY id")]

    spy = _LoadsSpy()
    monkeypatch.setattr(core_mod.json, "loads", spy)
    second = eng.migrate_v17_coordinates()
    monkeypatch.undo()

    assert second["migrated_nodes"] == 0, "D 非幂等：%r" % (second,)
    assert spy.args == [], (
        "#279 迁完仍逐行解析：%d 次" % len(spy.args))
    after = [tuple(r) for r in eng.store.conn.execute(
        "SELECT id, spatial_coordinates, semantic_coordinates FROM nodes ORDER BY id")]
    assert before == after, "D 二次调用改动了库：%r -> %r" % (before, after)


def test_e_嵌套语义键仍不迁(monkeypatch):
    """E：SQL 只是超集筛——嵌套语义键（旧实现也**不**迁）仍不迁，语义不变。

    该行含 `"protocol_` 子串 ⇒ SQL 会把它当候选（**已知假阳性**，见
    `_V17_CANDIDATE_SQL` 的必要性论证）；真判据仍是 Python 侧顶层
    `startswith`，故结果与旧实现一致：不迁移、库不变。
    """
    eng = SpacetimeMemoryEngine(":memory:")
    nested = json.dumps({"visual": {"protocol_x": 1}})
    _insert_raw(eng, "nested_1", nested)
    ret = eng.migrate_v17_coordinates()
    assert ret["migrated_nodes"] == 0, "E 嵌套键被误迁（改变旧语义）：%r" % (ret,)
    got = eng.store.conn.execute(
        "SELECT spatial_coordinates FROM nodes WHERE id='nested_1'").fetchone()[0]
    assert json.loads(got) == {"visual": {"protocol_x": 1}}, \
        "E 嵌套键行被改写：%r" % (got,)


def test_f_候选行在场时构造期照旧迁移():
    """F：防误杀——候选行在场时，构造期迁移照旧完成（不是永久不迁）。"""
    td = tempfile.mkdtemp()
    path = os.path.join(td, "legacy.db")
    eng = SpacetimeMemoryEngine(path)
    _insert_raw(eng, "legacy_2", json.dumps({"protocol_delta": 9}))

    fresh = SpacetimeMemoryEngine(path)      # 构造期即迁移
    row = fresh.store.conn.execute(
        "SELECT spatial_coordinates, semantic_coordinates FROM nodes WHERE id='legacy_2'"
    ).fetchone()
    assert json.loads(row[0]) == {}, "F 构造期未摘除语义键：%r" % (row[0],)
    assert json.loads(row[1])["protocol"]["concept"] == {"delta": 9}, \
        "F 构造期未完成迁移：%r" % (row[1],)


def test_g_非JSON坏行的新边界_构造不再崩():
    """G：**如实钉住削掉的判别力**（见 `_V17_CANDIDATE_SQL` 第二处边界）。

    旧实现无条件全表 `json.loads` ⇒ 非 JSON 正文行（如 `{'protocol_x': 1}`
    单引号形态）令 `migrate_v17_coordinates` 抛 `JSONDecodeError`，经
    `__init__` 变成**构造期崩溃**（坏行被当场点名，但整个引擎起不来）。
    新实现该行不含 `"protocol_` 子串 ⇒ 静默跳过，**构造不再崩、该行也不再被
    当场点名**。本断言钉住这一取舍（取「构造可用」优先），防止日后被误读成
    「迁移一定覆盖全库坏行」。
    """
    td = tempfile.mkdtemp()
    path = os.path.join(td, "bad.db")
    eng = SpacetimeMemoryEngine(path)
    _insert_raw(eng, "bad_json_1", "{'protocol_x': 1}")   # 非 JSON（单引号）
    eng.store.close()

    fresh = SpacetimeMemoryEngine(path)                  # 旧实现：此处 JSONDecodeError
    assert fresh.migrate_v17_coordinates()["migrated_nodes"] == 0, \
        "G 坏行被当成候选迁移了（应静默跳过）"
    got = fresh.store.conn.execute(
        "SELECT spatial_coordinates FROM nodes WHERE id='bad_json_1'").fetchone()[0]
    assert got == "{'protocol_x': 1}", "G 坏行被改写了：%r" % (got,)
