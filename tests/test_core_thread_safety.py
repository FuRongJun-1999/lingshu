# -*- coding: utf-8 -*-
"""test_core_thread_safety · 进程内多线程共享连接的并发回归

LayeredStore 只建一条 sqlite3.Connection 并设 check_same_thread=False 让所有线程共用；
而 sqlite3.Connection 不是线程安全的——多线程并发时一条语句序列会被其他线程插入，
表现为 database is locked / another row available，并从写路径漏到读路径。
本件断言外部可观察行为，不读内部字段。

运行（仓根）：python -X utf8 -m pytest tests/test_core_thread_safety.py -v
"""
import collections
import os
import sys
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingshu.core.core import SpacetimeMemoryEngine  # noqa: E402

TAG, SEED, PER = "并发回归填充", 40, 80


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