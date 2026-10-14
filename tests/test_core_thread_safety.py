# -*- coding: utf-8 -*-
"""test_core_thread_safety · 进程内多线程共享连接的并发回归 + 连接槽生命周期守卫

LayeredStore 只建一条 sqlite3.Connection 并设 check_same_thread=False 让所有线程共用；
而 sqlite3.Connection 不是线程安全的——多线程并发时一条语句序列会被其他线程插入，
表现为 database is locked / another row available，并从写路径漏到读路径。

2026-10-09 连接槽生命周期修复（PR #202 的遗留边界）后，本件扩为三面：
  ① 并发回归（既有 4 条，外部可观察行为）；
  ② 连接槽生命周期：线程退出 ⇒ 连接被 `_ThreadConn.__del__` 关闭并回收；close() 后清空；
  ③ AST 静态守卫：凡含写语句的方法必须带 @_transactional（防新增写点漏挂装饰器）。
②③ 须读内部字段（`_conns` / `_anchor`），是对「连接是否真被关闭」的直接判据，
不再局限「只看外部行为」。

运行（仓根）：python -X utf8 -m pytest tests/test_core_thread_safety.py -v
"""
import ast
import collections
import gc
import io
import os
import re
import sqlite3
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingshu.core.core import SpacetimeMemoryEngine  # noqa: E402

TAG, SEED, PER = "并发回归填充", 40, 80

# =============================================================================
# AST 静态守卫：凡含写语句的方法必须带 @_transactional
# =============================================================================
# 设计（2026-10-09 连接槽生命周期修复）：写事务边界钉在「方法」粒度——@_transactional
# 的 wrapper 用 `with self._tx():` 包住整个方法体（正常 commit / 异常 rollback）。
# 写点分散在两个类共 40 处，逐点 `with` 包裹漏改风险高；故以静态守卫机械核验：
# 凡方法体内出现「写语句」而**未**挂本装饰器，即红并点名 `类.方法:行号`。
#
# 判据（写语句）：① `execute/executemany/executescript` 且 SQL 字面量含
#   INSERT/UPDATE/DELETE/REPLACE；② 任意 `.commit()` 调用。
# 已知边界：SQL 由变量拼装（非字面量）时本守卫**看不见**——本文件写点均为字面量
# SQL（`_note_reuse` 的 CREATE TABLE + executemany(INSERT) 亦为字面量）；新增动态
# SQL 写点须人工复核，或先落成字面量再挂装饰器。
_WRITE_KEYWORDS = re.compile(r"\b(INSERT|UPDATE|DELETE|REPLACE)\b", re.I)

# 显式白名单（**必须写明理由**，不许硬编码绕过；条目须真实存在，否则守卫报「陈旧白名单」）。
_WRITE_GUARD_WHITELIST = {
    ("LayeredStore", "_tx"):
        "装饰器实现本体：@_transactional 的 wrapper 调用的就是它——若给它再套装饰器即"
        "无限递归。其自身即事务边界（正常 commit / 异常 rollback 后原样抛）。",
    ("LayeredStore", "increment_access"):
        "另开独立连接 `sqlite3.connect(self.db_path, timeout=0)` 并自带 try/finally: "
        "obs.close()，**不共用** self.conn；PR #202 明确列为「另一议题」（访问计数为统计"
        "性质，拿不到锁即放弃，不阻断检索主流程）。其写不落在 self.conn 的事务面上，"
        "故不适用本装饰器。",
}


def _literal_sql(node):
    """尽力取回 SQL 字面量字符串；非字面量（变量/f-string 拼装）返回 None。"""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _literal_sql(node.left), _literal_sql(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def _iter_calls_flat(node):
    """深度优先遍历调用节点，**不进入**嵌套函数/lambda 体（只算本方法自身的语句）。"""
    stack = [node]
    while stack:
        cur = stack.pop()
        if isinstance(cur, ast.Call):
            yield cur
        for child in ast.iter_child_nodes(cur):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            stack.append(child)


def _write_hits(fn_node):
    """返回方法体内命中的写语句调用节点列表（空列表 = 无写语句）。"""
    hits = []
    for call in _iter_calls_flat(fn_node):
        f = call.func
        if not isinstance(f, ast.Attribute):
            continue
        if f.attr in ("execute", "executemany", "executescript"):
            if call.args:
                sql = _literal_sql(call.args[0])
                if sql and _WRITE_KEYWORDS.search(sql):
                    hits.append(call)
        elif f.attr == "commit":
            hits.append(call)
    return hits


def _decorator_names(fn_node):
    names = set()
    for d in fn_node.decorator_list:
        if isinstance(d, ast.Name):
            names.add(d.id)
        elif isinstance(d, ast.Attribute):
            names.add(d.attr)
        elif isinstance(d, ast.Call):
            target = d.func
            if isinstance(target, ast.Name):
                names.add(target.id)
            elif isinstance(target, ast.Attribute):
                names.add(target.attr)
    return names



def _concurrent(n_writers, n_readers):
    """返回 (异常总数, {类型: 次数})。文件库——:memory: 不涉及跨连接写锁。"""
    engine = SpacetimeMemoryEngine(os.path.join(tempfile.mkdtemp(), "conc.db"))
    try:
        for i in range(SEED):
            engine.add_perception("%s %d" % (TAG, i), skip_dedup=True)
        errs, stop = collections.Counter(), threading.Event()

        def writer(tid):
            for i in range(SEED + tid * PER, SEED + tid * PER + PER):
                try:
                    engine.add_perception("%s %d" % (TAG, i), skip_dedup=True)
                except Exception as ex:
                    errs[type(ex).__name__] += 1

        def reader():
            while not stop.is_set():
                try:
                    engine.search_content(TAG)
                except Exception as ex:
                    errs[type(ex).__name__] += 1

        ts = [threading.Thread(target=writer, args=(i,)) for i in range(n_writers)]
        ts += [threading.Thread(target=reader) for _ in range(n_readers)]
        for t in ts:
            t.start()
        for t in ts[:n_writers]:
            t.join()
        stop.set()
        for t in ts[n_writers:]:
            t.join()
        return sum(errs.values()), dict(errs)
    finally:
        engine.close()


def test_two_writers_two_readers_raise_nothing():
    """2 写 + 2 读并发：不得出现任何异常。"""
    total, detail = _concurrent(2, 2)
    assert total == 0, "并发异常 %d 次：%s" % (total, detail)


def test_four_writers_four_readers_raise_nothing():
    """4 写 + 4 读并发：不得出现任何异常（基线在此约 3~4×10² 次）。"""
    total, detail = _concurrent(4, 4)
    assert total == 0, "并发异常 %d 次：%s" % (total, detail)


def test_memory_backend_shares_data_across_threads():
    """对照项：:memory: 后端不得因并发改造而在线程间互相看不见。"""
    engine = SpacetimeMemoryEngine(":memory:")
    try:
        engine.add_perception("主线程写入", skip_dedup=True)
        seen = {}

        def other():
            try:
                seen["hits"] = len(engine.search_content("主线程写入"))
                engine.add_perception("子线程写入", skip_dedup=True)
            except Exception as ex:
                seen["err"] = "%s: %s" % (type(ex).__name__, ex)

        t = threading.Thread(target=other)
        t.start()
        t.join()
        assert "err" not in seen, seen
        assert seen["hits"] == 1, "子线程看不到主线程写入的记忆：%r" % (seen,)
        assert len(engine.search_content("子线程写入")) == 1, "主线程看不到子线程写入的记忆"
    finally:
        engine.close()


def test_connections_are_per_thread_but_stable_within_one():
    """对照项：同线程重复取 conn 必须是同一条；不同线程不得是同一条。"""
    engine = SpacetimeMemoryEngine(os.path.join(tempfile.mkdtemp(), "c.db"))
    try:
        first = engine.store.conn
        assert engine.store.conn is first, "同线程重复访问 conn 每次都在新建连接"
        other = {}
        t = threading.Thread(target=lambda: other.update(conn=engine.store.conn))
        t.start()
        t.join()
        assert other["conn"] is not first, "不同线程拿到了同一条连接（仍是共享连接）"
    finally:
        engine.close()


def test_close_closes_every_per_thread_connection():
    """N 线程各建一条连接；线程退出后其连接被回收关闭，close() 后登记清空。

    返工补正（#202）：`_connect` 若漏掉 `check_same_thread=False`，主线程在 `close()`
    里对本线程之外的连接调 `c.close()` 会抛 `ProgrammingError: SQLite objects created
    in a thread can only be used in that same thread`，被 `except Exception: pass` 吞掉
    ⇒ 那些连接**实际未关**（`_conns` 虽被清空，连接却泄漏），与 docstring「关闭本实例
    登记的全部连接」不符。本件以「关闭后再对连接执行语句」为判据：已关连接抛
    `... closed database`，未关的跨线程连接抛线程归属错（不含 closed）。

    断言更迭（2026-10-09 连接槽生命周期修复，**性质＝旧断言锁住了要改的行为**）：
    旧断言 `len(engine.store._conns) == n + 1`（8 线程各建一条 + 主线程一条 = 9）锁住的
    是**旧语义**——连接登记后永不移除、登记表随线程数无界增长、线程退出后连接不关闭
    （其可能悬挂的写事务持锁不放）。本次修复把登记表改为 `weakref.WeakSet` + `_ThreadConn`
    槽的 `__del__`：线程退出 ⇒ threading.local 释放槽 ⇒ 连接关闭并从登记表回收。
    故线程 join 之后 `_conns` 只剩**锚连接** 1 条——这正是修复要的效果，而非回归。
    本件因此保留原始意图（close() 后登记清空、且被登记的连接确实已关），把「登记数恒
    等于线程数」这条过时断言换成新语义下的正确断言：线程 join 后只剩锚连接 1 条；
    且线程内连接**确已关闭**（不只从登记表消失）。
    """
    engine = SpacetimeMemoryEngine(os.path.join(tempfile.mkdtemp(), "close.db"))
    n = 8
    conns, errs = {}, []
    gate = threading.Barrier(n)

    def make(tid):
        try:
            gate.wait()
            conns[tid] = engine.store.conn
        except Exception as ex:  # pragma: no cover
            errs.append("%s: %s" % (type(ex).__name__, ex))

    ts = [threading.Thread(target=make, args=(i,)) for i in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    assert not errs, errs
    assert len(conns) == n, conns
    assert len({id(c) for c in conns.values()}) == n, "不同线程拿到了同一条连接"
    # 新语义：线程退出 ⇒ 其连接槽被回收（WeakSet 不再登记）⇒ 只剩锚连接 1 条。
    gc.collect()
    assert len(engine.store._conns) == 1, (
        "线程退出后登记表应只剩锚连接 1 条（旧行为＝n+1 条，锁住的是修复前的语义）；"
        "实际 %d 条" % len(engine.store._conns))

    # 关键：线程内的连接**确实已关闭**（不只是从登记表消失）——__del__ 未触发即在此红。
    def closed(c):
        try:
            c.execute("SELECT 1")
        except Exception as ex:
            return "closed" in str(ex).lower()
        return False

    leaked = [tid for tid, c in conns.items() if not closed(c)]
    assert not leaked, (
        "线程退出后其连接仍可执行语句（线程 %r）——__del__ 未触发，连接未关闭" % (leaked,))

    engine.close()
    assert len(engine.store._conns) == 0, "close() 后 _conns 登记未清空"
    assert engine.store._anchor is None, "close() 后锚连接未释放"

    still_open = [tid for tid, c in conns.items() if not closed(c)]
    assert not still_open, (
        "close() 后仍有未关闭的连接（线程 %r）——常见根因：_connect 缺 "
        "check_same_thread=False，主线程 close() 跨线程关连接抛 ProgrammingError 被吞"
        % (still_open,))


def test_no_untracked_write_method_misses_transactional():
    """AST 静态守卫：凡含写语句的方法必须带 @_transactional，漏挂即点名 `类.方法:行号`。

    `_transactional` docstring 承诺本守卫存在（防新增写点漏挂装饰器）；本件即其落地。
    白名单见 `_WRITE_GUARD_WHITELIST`（每条附理由），并校验白名单未陈旧。
    """
    src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "lingshu", "core", "core.py")
    with io.open(src_path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=src_path)

    seen_whitelist, offenders = set(), []
    for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
        for fn in cls.body:
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not _write_hits(fn):
                continue  # 无写语句：不要求装饰器
            key = (cls.name, fn.name)
            seen_whitelist.add(key)
            if "_transactional" in _decorator_names(fn):
                continue
            if key in _WRITE_GUARD_WHITELIST:
                continue
            offenders.append("%s.%s:%d" % (cls.name, fn.name, fn.lineno))

    assert not offenders, (
        "以下含写语句的方法未挂 @_transactional（新增写点漏挂装饰器即在此红）：\n  "
        + "\n  ".join(offenders)
        + "\n如确为设计豁免，请加入 _WRITE_GUARD_WHITELIST 并写明理由。")

    stale = sorted(set(_WRITE_GUARD_WHITELIST) - seen_whitelist)
    assert not stale, "白名单条目已不存在（陈旧白名单应删除）：%r" % (stale,)


def test_memory_backend_shares_data_across_threads_repeated():
    """:memory: 语义在「多线程反复创建/退出」后仍成立——锚连接保活共享内存库。

    修复把 `_conns` 由强引用列表改 WeakSet 后，必须显式留一条**强引用**锚连接
    （`_anchor`），否则创建线程退出、其连接被回收，`file:...?mode=memory&cache=shared`
    的内存库整体消失，跨线程可见性失效。本件以「线程反复进出后仍互相可见」为判据。
    """
    engine = SpacetimeMemoryEngine(":memory:")
    try:
        engine.add_perception("锚写入", skip_dedup=True)
        for rnd in range(5):
            seen = {}

            def worker(r=rnd):
                try:
                    seen["hits"] = len(engine.search_content("锚写入"))
                    engine.add_perception("轮次写入 %d" % r, skip_dedup=True)
                except Exception as ex:
                    seen["err"] = "%s: %s" % (type(ex).__name__, ex)

            t = threading.Thread(target=worker)
            t.start()
            t.join()
            gc.collect()
            assert "err" not in seen, "轮 %d 子线程异常：%r" % (rnd, seen)
            assert seen["hits"] == 1, "轮 %d 子线程看不到锚连接写入：%r" % (rnd, seen)
            assert len(engine.search_content("轮次写入 %d" % rnd)) == 1, (
                "轮 %d 主线程看不到子线程写入（内存库可能已随线程退出消失）" % rnd)
    finally:
        engine.close()


def test_memory_backend_survives_creating_thread_exit():
    """:memory: 锚连接保活：**创建线程退出后**共享内存库不得消失。

    本件直接针对 `_anchor` 强引用：引擎在子线程内构造（建表 + 写入），子线程 join
    后其 threading.local 槽被释放——若没有 `_anchor` 这条强引用，该槽被 GC ⇒ `__del__`
    关闭连接 ⇒ `file:...?mode=memory&cache=shared` 的内存库整体消失，主线程再连就是空库。
    判据：主线程必须仍看得见子线程写入的数据。
    """
    box = {}

    def create():
        eng = SpacetimeMemoryEngine(":memory:")
        eng.add_perception("创建线程写入", skip_dedup=True)
        box["engine"] = eng

    t = threading.Thread(target=create)
    t.start()
    t.join()
    gc.collect()
    engine = box["engine"]
    try:
        hits = len(engine.search_content("创建线程写入"))
        assert hits == 1, (
            "创建线程退出后主线程看不到其写入（%d 条）——锚连接未保活，内存库已消失" % hits)
    finally:
        engine.close()


def test_exception_does_not_leave_transaction_dangling():
    """异常后事务不悬挂：写中途抛异常 ⇒ 同库另一连接能**立即**写（不等 30 秒）。

    设计者报告的现象：旧实现写路径 execute 与 commit 之间抛异常时既无 commit 也无
    rollback ⇒ SQLite 隐式事务挂在连接上持写锁不放 ⇒ 其它写者等满 busy_timeout=30000ms
    后报 database is locked。修复用 `_tx()` 保证异常必定 rollback。本件直接断言：
    注入一次写异常，随后**另一个连接**写同一库必须秒级成功。
    """
    db = os.path.join(tempfile.mkdtemp(), "dangling.db")
    engine = SpacetimeMemoryEngine(db)
    try:
        engine.add_perception("前置写入", skip_dedup=True)

        class _Boom(Exception):
            pass

        # 制造「execute 成功但方法中途抛异常」：在 _tx() 边界内真写一行，随后抛异常。
        # 旧实现（无 rollback）会让隐式事务挂在连接上持写锁不放；修复后 _tx() 必回滚。
        with engine.store._tx():
            engine.store.conn.execute(
                "INSERT INTO nodes (id, content, modality, layer, tags) "
                "VALUES ('dangling_probe','x','text','knowledge','[]')")
            raise _Boom("写后抛异常，事务必须回滚")

        raise AssertionError("上面的 _Boom 未被抛出")  # pragma: no cover
    except _Boom:
        pass

    # 同库另一条独立连接：必须能立即写成功（不等 busy_timeout=30s）。
    t0 = time.time()
    other = sqlite3.connect(db, timeout=30)
    try:
        other.execute("INSERT INTO nodes (id, content, modality, layer, tags) "
                      "VALUES ('probe1','x','text','knowledge','[]')")
        other.commit()
    finally:
        other.close()
    elapsed = time.time() - t0
    assert elapsed < 5.0, (
        "异常后另一连接写库耗时 %.1fs（>5s）——写锁仍被悬挂事务持有，rollback 未生效"
        % elapsed)
    # 被回滚的探针行不得残留（证明是真回滚，不是仅释放锁）。
    c = engine.store.conn.cursor()
    c.execute("SELECT COUNT(*) FROM nodes WHERE id='dangling_probe'")
    assert c.fetchone()[0] == 0, "异常路径的写入未回滚，dangling_probe 残留"
    engine.close()


def test_normal_write_path_unaffected_and_concurrent():
    """正对照：正常写路径不受装饰器影响（含多线程并发写）。"""
    db = os.path.join(tempfile.mkdtemp(), "normal.db")
    engine = SpacetimeMemoryEngine(db)
    try:
        engine.add_perception("正常写入", skip_dedup=True)
        assert len(engine.search_content("正常写入")) == 1

        errs = collections.Counter()

        def writer(tid):
            for i in range(20):
                try:
                    engine.add_perception("并发 %d %d" % (tid, i), skip_dedup=True)
                except Exception as ex:
                    errs[type(ex).__name__] += 1

        ts = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert not errs, "并发写异常：%r" % (dict(errs),)
    finally:
        engine.close()
