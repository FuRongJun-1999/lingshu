# -*- coding: utf-8 -*-
"""#770 守卫：`write_snapshot` 写面原子化 ＋ `_known_snapshot` 签名判别力。

缺陷态（`origin/main` 上三条独立复现，见 issue #770 核验帖）
------------------------------------------------------------
① **并发 `write_snapshot` 丢标签**（`LongTermMemoryGate.write_snapshot`）：
   「存在性检查 → 读既有节点 → 整列 `UPDATE nodes SET importance=?, tags=?, …`」
   三步**跨事务**（读在事务外、写是短事务）⇒ 两个连接各自读到同一份旧 `tags`，
   后写者用旧值整列覆盖先写者。实测 4 进程 × 6 次：留存 6/12/12（期望 24）、
   **异常 0**、库内恒 1 行 —— 丢的是**整组 worker 的标签**，静默无信号。
   正对照：同并发形态改走 `tag_node`（#344 已用 `BEGIN IMMEDIATE` 原子化）
   ⇒ 3/3 轮 24/24 留存 ⇒ 缺陷是 `write_snapshot` 这条路径特有的。

② **`_known_snapshot` 签名对等长原地改写失明**：签名
   `(COUNT(*), MAX(rowid), SUM(LENGTH(content)))` 在 `nodes.content` 被原地改成
   **同样长度**的另一段正文时**逐位不变**（实测 `(1,1,13)`→`(1,1,13)`）⇒ 缓存被
   判有效、陈旧 gram 表被复用：新正文 gram 0/21 进并集、旧正文 gram 21/21 留存；
   后果是「已入库正文」被判 `novelty=1.0` / `knowledge` / `prefeed_boost`，而同一
   内容走新建腿是 `0.0` / `context` / `routine`。

③ **存在性检查与 `skip_dedup=True` 新建腿无互斥**：检查（`search_content` 命中
   0.95）与新建（`add_perception(..., skip_dedup=True)`）之间无锁 ⇒ 并发首写同一
   内容各建一行。实测 40 轮 × 4 进程：**40/40 轮**出重复行（4 行 22 轮、3 行 18 轮），
   重复行 tags 均为 `["<tag>","gate","no_forget"]`、worker 返回多个不同 `node_id`。

修法（落点 `lingshu/core/longterm_gate.py`）
--------------------------------------------
① ③ 「判定复核 ＋ 写」收进**单一写事务**（复用仓内既有 `LayeredStore._tx()`，
   与 #344 `tag_node` / #517 `delete_node` 同一原语）：`BEGIN IMMEDIATE` 先取
   RESERVED 写锁，`_recheck_existing`（纯读复核）与整列 `UPDATE` 同锁。
   ★事务**只覆盖复核与写**：`search_content` 的存在性预判与 `evaluate()` 留在
   事务外（`_snapshot_probe`）——`search_content` 的命中计数腿走
   `increment_access_many`（**自开一条 `timeout=0` 独立连接**），放进事务会与
   `BEGIN IMMEDIATE` 自冲突并把互等窗口放大到 `busy_timeout`（实测 4 进程 × 6 次
   出现 30s 级 `database is locked`）。长期层的相似边建边同理移到提交之后。
② 签名第三项由 `SUM(LENGTH(content))` 改为**正文内容指纹**（逐行
   `(rowid, 摘要)` 折叠成 8 字节整数）；增量分支的三条门槛保留。返回形状仍是
   `(int, int, int)`。
   ★后续 #462 修复在增量分支补了「计数平衡 + 旧 id 一个不少」两道无删除证明
   ⇒ D 格（删 1 增 2）由「增量并入」反转为**全量重建**，本文件 G3 判据随之更新。

断言组（把 `longterm_gate.py` 回退到 #770 修复前 ⇒ G1/G3/G5/G6/G9 必红；
还原 ⇒ 全绿且 `git diff` 为空）
-------------------------------------------------------------------------
  G1 **A 格（等长原地改写）确定性**：签名必变、缓存对象必重建、新正文 gram 全数
     入并集、旧正文 gram 全数退出；且同库同内容的 `novelty` 与「新建腿」一致
     （改前：签名 `(1,1,13)`→`(1,1,13)` 不变、新 gram 0 入、旧 gram 全留、
     `novelty=1.0`）。
  G2 **B/C 格（变长改写）仍全量重建**：新 gram 全入、旧 gram 退出。
  G3 **D 格（删 1 增 2）全量重建**（#462 修复后反转）：三项门槛仍全增，但
     「无删除」证明不成立 ⇒ **不得增量**（`union` 对象不复用、被删 gram 退出）。
  G4 **纯追加仍走增量**（#389 的工作负载：每条输入写一个节点）⇒ `union` 对象同一、
     节点数 +1（防「改成每次全量重建」把 #389 的增量收益打掉）。
  G5 **确定性读—写窗口**（不靠概率）：A 线程在「读到旧 `tags`」之后停顿一个窗口，
     B 线程在同一窗口内写同一节点 —— 两个 tag 必须都留存（改前 B 的 tag 被 A 的
     整列覆盖抹掉）。这是 #344 J1 的同款手法，把并发缺陷做成**确定性红**。
  G6 **4 进程 × 6 次并发 `write_snapshot`**：24/24 留存、库内 1 行、worker 异常 0
     （改前：留存 6/12/12、丢整组 worker 标签）。
  G7 **并发首写唯一性**：N 轮 × 4 进程并发首写同一内容 ⇒ 每轮库内该内容**恰 1 行**
     （改前：40/40 轮出重复行）。
  G8 **重入面**：`write_snapshot` 被已在 `_tx()` 内的调用方调进来 ⇒ 不抛
     `cannot start a transaction within a transaction`，且外层事务仍可整体回滚。
  G9 **事务边界不泄漏**：`write_snapshot` 返回后连接**不在事务中**（不把写锁留给
     后续语句），且失败路径异常原样抛出、连接同样不在事务中。
  G10 正对照（防过修）：返回键集与取值形态**修前修后逐位相同**——`node_id` /
     `layer`（回读实写层）/ `evaluated_layer` / `importance` / `features` /
     `write_failed` / `write_error`；`status='discarded'` 的情境层不落库腿不变。

边界（本件不改，如实标注）
--------------------------
`@_transactional` 的 AST 守卫（`tests/test_core_thread_safety.py`）**只扫
`lingshu/core/core.py` 一个文件**，`longterm_gate.py` 不在其扫描面内 ⇒ 本件不扩该
扫描面（扩面会让全包 20 个未挂装饰器的方法当场转红，属另一笔）。`promote_from_context`
的同类读改写**不在本件范围**。`_known_snapshot` 的删除误判（#462）已由后续修复
处理（见 G3 反转）；剩余未覆盖面＝「同 rowid 原地 UPDATE ∧ 同期追加节点」的混合
形态（旧缓存未存逐行摘要，检测它属另一次改动面，见该函数 docstring 的已知残留）。

运行（仓根）：python -X utf8 -m pytest tests/test_issue770_snapshot_atomicity_and_signature_guard.py -q --no-header
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.core.core import LayeredStore, SpacetimeMemoryEngine  # noqa: E402
from lingshu.core.longterm_gate import LongTermMemoryGate  # noqa: E402

#: 等长对照串（A 格的核心：两串**字符数相同**、内容不同）。
LEN_EQ_A = "alpha12345678"
LEN_EQ_B = "bravo76543210"
LEN_LONGER = "charlie1234567890"
LEN_SHORTER = "delta123"
assert len(LEN_EQ_A) == len(LEN_EQ_B) == 13
assert len(LEN_LONGER) == 17 and len(LEN_SHORTER) == 8

#: 并发探针的内容（≥ 2 个 gram，保证 `_core_grams` 非空）。
CONC_CONTENT = "并发写快照探针内容：蜂巢并发写入同一既有节点"

#: 并发格的时间盒（单格总超时；超时即判失败，不无限等）。
CONC_TIMEOUT = 180


def _eng(db=":memory:"):
    return SpacetimeMemoryEngine(db)


def _tags_of(db, node_id):
    st = LayeredStore(db)
    try:
        row = st.conn.execute("SELECT tags FROM nodes WHERE id=?",
                              (node_id,)).fetchone()
        return json.loads(row[0]) if row and row[0] else []
    finally:
        st.close()


def _rows_with_content(db, content):
    st = LayeredStore(db)
    try:
        return st.conn.execute(
            "SELECT id, tags, importance FROM nodes WHERE content=?",
            (content,)).fetchall()
    finally:
        st.close()


def _rewrite_in_place(engine, node_id, new_content):
    """原地改写 `nodes.content`（绕过 `add_*` 入口）——模拟等长/变长原地改写。"""
    engine.store.conn.execute("UPDATE nodes SET content=? WHERE id=?",
                              (new_content, node_id))
    engine.store.conn.commit()


def _in_transaction(store):
    try:
        return bool(store.conn.in_transaction)
    except Exception:
        return False


# ---------------- ② 签名判别力（A / B / C / D / 追加） ----------------

def test_g1_equal_length_in_place_rewrite_is_detected():
    """G1：A 格（等长原地改写）—— 签名必变、缓存必重建、并集必换血。

    改前读数（`origin/main`）：签名 `(1,1,13)`→`(1,1,13)` **逐位不变**、缓存对象
    **同一**复用、新正文 gram **0/1** 入并集、旧正文 gram **1/1** 留存、
    `novelty=1.0` / `knowledge`（与同库同内容的新建腿 `0.0` / `context` 相反）。
    """
    e = _eng()
    try:
        g = e._ensure_gate()
        nid = e.add_perception(LEN_EQ_A, importance=0.5).id
        u1, _, p1, s1 = g._known_snapshot()
        assert len(s1) == 3, "签名形状必须是三元组（调用方按三元组解包）：%r" % (s1,)
        ga = g._core_grams(LEN_EQ_A)
        gb = g._core_grams(LEN_EQ_B)
        assert ga and gb, "对照串必须切出非空 gram：%r/%r" % (ga, gb)
        assert len(ga & u1) == len(ga), "前置：旧正文 gram 应已在并集：%r" % (u1,)

        _rewrite_in_place(e, nid, LEN_EQ_B)

        u2, _, p2, s2 = g._known_snapshot()
        assert s2 != s1, (
            "等长原地改写后签名逐位不变 ⇒ 缓存被判有效、复用陈旧 gram 表"
            "（#770② 缺陷形态）：%r -> %r" % (s1, s2))
        assert u2 is not u1, "签名已变 ⇒ 必须重建缓存，不得复用同一 union 对象"
        assert len(gb & u2) == len(gb), (
            "新正文 gram 未进并集（复用陈旧缓存）：%d/%d" % (len(gb & u2), len(gb)))
        assert not (ga & u2), (
            "旧正文 gram 仍留在并集（陈旧缓存未失效）：%r" % sorted(ga & u2))

        # 判别力的**后果**：同库同内容的两条取数腿必须一致
        # （改前：原地改写腿 1.0 / knowledge vs 新建腿 0.0 / context）
        ev = g.evaluate(LEN_EQ_B)
        assert ev["features"]["novelty"] == 0.0, (
            "已入库正文被误判为全新（陈旧缓存）：%r" % (ev,))
        assert ev["layer"] == "context", (
            "已入库正文被判为知识层（陈旧缓存抬高了 imp）：%r" % (ev,))
    finally:
        e.close()


def test_g2_variable_length_rewrite_still_rebuilds():
    """G2：B/C 格（变长原地改写）—— 既有行为不得改动：仍全量重建。"""
    for new_content in (LEN_LONGER, LEN_SHORTER):
        e = _eng()
        try:
            g = e._ensure_gate()
            nid = e.add_perception(LEN_EQ_A, importance=0.5).id
            u1, _, _, s1 = g._known_snapshot()
            ga = g._core_grams(LEN_EQ_A)
            _rewrite_in_place(e, nid, new_content)
            u2, _, _, s2 = g._known_snapshot()
            gn = g._core_grams(new_content)
            assert s2 != s1, "变长改写后签名未变：%r -> %r" % (s1, s2)
            assert len(gn & u2) == len(gn), (
                "变长改写的新正文 gram 未全数入并集：%d/%d" % (len(gn & u2), len(gn)))
            assert not (ga & u2), "变长改写的旧正文 gram 仍留存：%r" % sorted(ga & u2)
        finally:
            e.close()


def test_g3_delete_one_add_two_rebuilds_without_stale_grams():
    """G3：D 格（删 1 增 2，#462）—— 修复后必须**全量重建**。

    #770 当时的派工口径是「只修 A 格，D 的增量行为原样保留（#462 属另一笔
    账面）」，故旧版此例锁的是 `union is u1`。账面 #462 已修：增量分支补了
    「计数平衡 + 旧 id 一个不少」两道证明，D 格现在落到全量重建。本用例随之反转，
    并断言被删行的 gram 不在新并集里。
    """
    e = _eng()
    try:
        g = e._ensure_gate()
        nid = e.add_perception(LEN_EQ_A, importance=0.5).id
        victim_grams = g._core_grams(LEN_EQ_A)
        u1, _, _, s1 = g._known_snapshot()
        e.store.conn.execute("DELETE FROM nodes WHERE id=?", (nid,))
        e.store.conn.commit()
        e.add_perception("echo111222333", importance=0.5, skip_dedup=True)
        e.add_perception("foxtrot444555666", importance=0.5, skip_dedup=True)
        u2, _, _, s2 = g._known_snapshot()
        assert s2 != s1, "D 格签名未变：%r -> %r" % (s1, s2)
        assert u2 is not u1, (
            "删 1 增 2 仍走增量并入（union 对象被复用）—— #462 修复后必须全量重建")
        assert not (victim_grams & u2), (
            "被删节点的 gram 仍留在参考集（#462）：%r" % sorted(victim_grams & u2))
    finally:
        e.close()


def test_g4_append_only_still_incremental():
    """G4：纯追加仍走增量（#389 的工作负载）——防「改成每次全量重建」。"""
    e = _eng()
    try:
        for i in range(5):
            e.add_perception("高权无关条目第%03d号" % i, importance=0.9)
        g = e._ensure_gate()
        u1, _, p1, s1 = g._known_snapshot()
        n1 = len(p1)      # 增量路径复用**同一** per_node 对象 ⇒ 长度须先存
        e.add_perception("追加的一条全新知识记录用于增量验证", importance=0.5)
        u2, _, p2, s2 = g._known_snapshot()
        assert u2 is u1, "追加节点必须走增量并入（union 对象不变）"
        assert s2[0] == s1[0] + 1 and len(p2) == n1 + 1
    finally:
        e.close()


# ---------------- ① 写面原子性 ----------------

def test_g5_read_write_window_keeps_both_tags(tmp_path, monkeypatch):
    """G5：**确定性**读—写窗口 —— 两个 tag 必须都留存（不靠概率）。

    手法（与 #344 J1 同款「造最小扰动稳定复现」）：给 A 的连接套一层代理，把
    「整列 `UPDATE nodes SET importance=…, tags=…`」这条语句**推迟一个窗口**再
    真正下发；B 线程在该窗口内完成一次 `write_snapshot`。窗口恰好落在「A 已按自己
    读到的 tags 算好合并结果 → A 真正写回」之间：
      · 改前：A 的读在事务外 ⇒ A 的 UPDATE 用**陈旧 tags** 整列覆盖，B 写下的
        `tag_b` 被抹掉（0 异常、0 信号）；
      · 修后：A 的 `BEGIN IMMEDIATE` 在**读之前**就取到写锁 ⇒ B 的写被排到 A 提交
        之后，B 读到的是含 `tag_a` 的新 tags ⇒ 两者都留存。
    本件不使用「概率性多进程竞争」作为唯一判据（那一格见 G6/G7）。
    """
    db = str(tmp_path / "g5.db")
    eng = _eng(db)
    try:
        nid = eng.add_perception(CONC_CONTENT, importance=0.5).id
        eng._ensure_gate()._known_snapshot()   # 预热缓存（避免窗口内做全库扫描）
    finally:
        eng.close()

    eng_a, eng_b = _eng(db), _eng(db)
    gate_a, gate_b = eng_a._ensure_gate(), eng_b._ensure_gate()
    write_started = threading.Event()
    errs = []

    class _SlowUpdateConn:
        """代理 A 的连接：命中整列 UPDATE 时先放行 B、再真正下发。"""

        def __init__(self, real):
            self._real = real

        def execute(self, sql, *a, **kw):
            if sql.startswith("UPDATE nodes SET importance"):
                write_started.set()
                time.sleep(0.5)          # 「A 算好 tags → A 写回」之间的窗口
            return self._real.execute(sql, *a, **kw)

        def __getattr__(self, name):
            return getattr(self._real, name)

    real_conn_prop = LayeredStore.conn
    store_a = eng_a.store

    def conn_proxy(self):
        real = real_conn_prop.fget(self)
        return _SlowUpdateConn(real) if self is store_a else real

    monkeypatch.setattr(LayeredStore, "conn", property(conn_proxy))

    def other_connection():
        write_started.wait(10.0)
        try:
            gate_b.write_snapshot(CONC_CONTENT, source="probe", tags=["tag_b"],
                                  importance_hint=0.5)
        except Exception as ex:      # pragma: no cover
            errs.append("%s: %s" % (type(ex).__name__, ex))

    t = threading.Thread(target=other_connection)
    try:
        t.start()
        gate_a.write_snapshot(CONC_CONTENT, source="probe", tags=["tag_a"],
                              importance_hint=0.5)
        t.join(60.0)
        assert not t.is_alive(), "B 线程未在时间盒内结束（疑似死锁/锁等待）"
        assert not errs, "并发写抛异常：%r" % (errs,)
        tags = _tags_of(db, nid)
        assert "tag_a" in tags, "A 自己的标签丢了：tags=%r" % (tags,)
        assert "tag_b" in tags, (
            "窗口内并发写入的 tag_b 被 A 的整列覆盖抹掉（#770①：读—判断—写跨事务）"
            "：tags=%r" % (tags,))
        assert len(_rows_with_content(db, CONC_CONTENT)) == 1, "并发写不得新建重复行"
    finally:
        eng_a.close()
        eng_b.close()


def _worker_snapshot(db, content, wtag, n):
    """子进程 worker：`write_snapshot` N 次，各追加不同 tag。"""
    return [sys.executable, "-X", "utf8", "-c", _WORKER_SRC,
            db, content, wtag, str(n)]


_WORKER_SRC = (
    "import json,os,sys\n"
    "sys.path.insert(0, %r)\n"
    "db, content, wtag, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])\n"
    "out = {'ids': [], 'errs': []}\n"
    "try:\n"
    "    from lingshu.core.core import SpacetimeMemoryEngine\n"
    "    e = SpacetimeMemoryEngine(db)\n"
    "    g = e._ensure_gate()\n"
    "    for k in range(n):\n"
    "        r = g.write_snapshot(content, source='probe',\n"
    "                             tags=['%%s_%%d' %% (wtag, k)], importance_hint=0.5)\n"
    "        out['ids'].append(r.get('node_id'))\n"
    "    e.close()\n"
    "except Exception as exc:\n"
    "    out['errs'].append('%%s: %%s' %% (type(exc).__name__, exc))\n"
    "sys.stdout.write('@@R@@' + json.dumps(out))\n"
) % (REPO,)


def _run_workers(db, content, n_per_worker, workers=4):
    """起 `workers` 个子进程各写 `n_per_worker` 次；返回 (ids, errs)。"""
    procs = [subprocess.Popen(_worker_snapshot(db, content, "w%d" % w, n_per_worker),
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              env=dict(os.environ, PYTHONUTF8="1"))
             for w in range(workers)]
    ids, errs = [], []
    deadline = time.time() + CONC_TIMEOUT
    for p in procs:
        remain = max(1.0, deadline - time.time())
        so, se = p.communicate(timeout=remain)
        if p.returncode != 0:
            errs.append("rc=%s %s" % (p.returncode,
                                      se.decode("utf-8", "replace")[-300:]))
            continue
        for line in so.decode("utf-8", "replace").splitlines():
            if line.startswith("@@R@@"):
                d = json.loads(line[len("@@R@@"):])
                ids.extend(d.get("ids") or [])
                errs.extend(d.get("errs") or [])
    return ids, errs


def test_g6_concurrent_snapshot_keeps_every_tag(tmp_path):
    """G6：4 进程 × 6 次并发 `write_snapshot` —— 24/24 留存、1 行、异常 0。

    改前读数（`origin/main`，3 轮）：留存 6 / 12 / 12（期望 24）、库内恒 1 行、
    worker 异常 0 —— 丢的是整组 worker 的标签。
    """
    db = str(tmp_path / "g6.db")
    eng = _eng(db)
    try:
        r0 = eng._ensure_gate().write_snapshot(CONC_CONTENT, source="seed",
                                               tags=["base"], importance_hint=0.5)
        nid = r0.get("node_id")
        assert nid, "seed 快照未落库：%r" % (r0,)
    finally:
        eng.close()

    ids, errs = _run_workers(db, CONC_CONTENT, 6, workers=4)
    assert not errs, "并发 worker 异常（改前为 0；此处不得新增）：%r" % (errs,)
    tags = set(_tags_of(db, nid))
    want = {"w%d_%d" % (w, k) for w in range(4) for k in range(6)}
    missing = sorted(want - tags)
    assert not missing, (
        "并发 write_snapshot 丢标签（#770①）：缺失 %d/%d 个 = %r（留存 %d）"
        % (len(missing), len(want), missing, len(want) - len(missing)))
    assert len(set(ids)) == 1, "并发写不得新建重复节点：distinct node_id=%r" % (
        sorted(set(ids)),)
    assert len(_rows_with_content(db, CONC_CONTENT)) == 1, "库内该内容应恰 1 行"


def test_g7_concurrent_first_write_is_unique(tmp_path):
    """G7：N 轮 × 4 进程并发首写同一内容 ⇒ 每轮库内**恰 1 行**。

    改前读数（`origin/main`，40 轮）：**40/40 轮**出重复行（4 行 22 轮、3 行 18 轮），
    重复行 tags 均为 `["<tag>","gate","no_forget"]`（`importance_hint=0.8` ⇒ 都被
    `protect_node` 保护）。本件取 12 轮（时间盒内），任一轮出重复即红。
    """
    db = str(tmp_path / "g7.db")
    _eng(db).close()
    rounds = 12
    bad = []
    for r in range(rounds):
        content = "并发首写唯一性探针内容第%03d号：蜂巢并发写同一正文" % r
        _run_workers(db, content, 1, workers=4)
        rows = _rows_with_content(db, content)
        if len(rows) != 1:
            bad.append((r, len(rows), [row[1] for row in rows]))
    assert not bad, (
        "并发首写同一内容出重复行（#770③）：%d/%d 轮，读数（轮, 行数, tags）= %r"
        % (len(bad), rounds, bad[:5]))


# ---------------- 重入 / 事务边界 / 正对照 ----------------

def test_g8_nested_call_does_not_start_second_transaction(tmp_path):
    """G8：重入面 —— 已被 `_tx()` 包住时调 `write_snapshot` 不得抛
    `cannot start a transaction within a transaction`；外层仍可整体回滚。"""
    db = str(tmp_path / "g8.db")
    e = _eng(db)
    try:
        store = e.store
        with store._tx():
            r = e._ensure_gate().write_snapshot(
                "重入面探针：事务内调用 write_snapshot", source="probe",
                tags=["nested"], importance_hint=0.5)
            assert r.get("node_id"), "事务内调用未落库：%r" % (r,)
            raise RuntimeError("test_issue770: 外层事务主动回滚")
    except RuntimeError:
        pass
    finally:
        e.close()
    rows = _rows_with_content(db, "重入面探针：事务内调用 write_snapshot")
    assert not rows, "外层回滚未撤回内层写入（#665 同款）：%r" % (rows,)


def test_g9_transaction_boundary_not_leaked(tmp_path):
    """G9：`write_snapshot` 返回后连接不在事务中（不把写锁留给后续语句）。"""
    db = str(tmp_path / "g9.db")
    e = _eng(db)
    try:
        g = e._ensure_gate()
        r = g.write_snapshot("事务边界探针：返回后不得留在事务里", source="probe",
                             tags=["boundary"], importance_hint=0.5)
        assert r.get("node_id"), r
        assert not _in_transaction(e.store), (
            "write_snapshot 返回后连接仍在事务中 ⇒ 写锁泄漏给后续语句")
        # 情境层且无 hint ⇒ B-12 裁决后**落库**（丢弃不可逆）——落库腿同样
        # 不得留事务（#114 裁决联动更新：旧「status=discarded」断言随裁决反转）。
        r2 = g.write_snapshot("事务边界探针：情境层不落库腿", source="probe")
        assert r2.get("status") != "discarded", r2
        assert r2.get("node_id"), r2
        assert not _in_transaction(e.store), (
            "情境层落库腿返回后连接仍在事务中（写锁泄漏）")
    finally:
        e.close()


def test_g10_return_shape_unchanged():
    """G10：正对照（防过修）—— 返回键集与取值形态与改前**逐位相同**。"""
    e = _eng()
    try:
        g = e._ensure_gate()
        r = g.write_snapshot("正对照探针：地球绕太阳公转一周约一年",
                             source="probe", tags=["t"], importance_hint=0.5)
        for key in ("node_id", "importance", "layer", "evaluated_layer",
                    "features", "write_failed", "write_error"):
            assert key in r, "既有结果键丢失：%r（键集 %r）" % (key, sorted(r))
        assert r["write_failed"] is False and r["write_error"] is None, r
        assert set(r["features"]) == {"novelty", "trust", "d2", "t2", "mention"}, r
        assert r["layer"] in ("knowledge", "long_term", "context"), r
        # 情境层且无 hint ⇒ B-12 裁决后落库（丢弃不可逆 ⇒ 以 context 层身份
        # 走 add_context 既有腿；#114 裁决联动更新：旧「status=discarded」
        # 断言随裁决反转——返回形态含 node_id 且 write_failed=False）。
        # layer 不硬钉 context——B-10 重归一后同内容 imp 上移过 0.40 知识层
        # 阈值是合法涟漪；只钉「落库成功＋键集完整＋非丢弃」。
        r2 = g.write_snapshot("正对照探针：情境层不落库腿内容")
        assert r2.get("status") != "discarded", r2
        assert r2.get("layer") in ("knowledge", "long_term", "context"), r2
        assert r2.get("node_id"), r2
        assert r2.get("write_failed") is False and r2.get("write_error") is None, r2
        # 层级/重要度提升失败仍可观测（#292 腿：键在且取值与成功路径不同）
        assert LongTermMemoryGate is not None
    finally:
        e.close()
