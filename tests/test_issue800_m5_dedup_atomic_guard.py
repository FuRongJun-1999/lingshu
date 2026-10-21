# -*- coding: utf-8 -*-
"""#800 守卫：M5 去重「查→插」临界区原子性——同内容并发写只产出一个节点。

现象（issue #800）
    `add_perception` 的 M5 去重（SELECT 候选 → 无同源 ⇒ INSERT）无引擎级互斥：
    多线程同时写**同一内容**时，各线程的 SELECT 都跑在对方 INSERT 之前，
    各自判「不存在」后各自 INSERT。实测 3 线程 × 20 次同内容 ⇒ **3 个不同 id**
    （期望 1）。lifecycle 后台线程与用户代码并发、多线程共享同一引擎实例时
    必然触发；后果＝recall 返回重复项 / importance 与 last_access 分散 / 存储
    与检索名额被占。

改前读数（基线 40fceb6 · Windows 11 · Python 3.12 · 本 worktree 实测）
    3 线程 × 20 次同内容写入：total=60，unique ids=3（`node_046f…` /
    `node_2c3f…` / `node_48bd…`）——TOCTOU 命中；
    单线程顺序写同内容 ×5：unique=1（去重本体正常，缺陷只在并发窗口）。
    修后（本件 `_m5_dedup_lock` 包住「去重判定→add_node→register_conflict」）：
    3×20 ⇒ unique=1；8×10 ⇒ unique=1；顺序写、合并路径、另立路径逐位不变。

断言组（回退 `_m5_dedup_lock`（换成 no-op 上下文）⇒ A/B 大概率红——并发窗口
真实存在时 SELECT 必然互相穿过；C/D/E 为行为不变面）：
  A issue 原复现：3 线程 × 20 次同内容 ⇒ 全部 60 个返回 id 相同
  B 高并发：8 线程 × 10 次同内容 ⇒ 80 个返回 id 相同（且 merged=True）
  C 节点数守恒：并发写后库里该内容**只有一个**知识层节点
  D 无竞态路径不变：顺序重复写仍合并、不同内容仍另立（锁定行为不回归）
  E 锁可重入：同线程在持锁路径内再入 add_perception 不自死锁
    （`_m5_dedup_lock` 为 RLock；以嵌套调用烟雾测试钉住）

运行（仓根）：python -X utf8 -m pytest tests/test_issue800_m5_dedup_atomic_guard.py -q --no-header
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lingshu.core.core import SpacetimeMemoryEngine, MemoryLayer  # noqa: E402


def _engine():
    d = tempfile.mkdtemp(prefix="lingshu800_")
    return SpacetimeMemoryEngine(os.path.join(d, "race.db"))


def _run_writers(e, n_threads, per_thread, content):
    got, lk = [], threading.Lock()

    def writer(_tid):
        for _ in range(per_thread):
            n = e.add_perception(content)
            with lk:
                got.append(n.id)

    ts = [threading.Thread(target=writer, args=(i,)) for i in range(n_threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return got


def test_a_issue_repro_three_writers_one_id():
    """A：issue 原复现——3 线程 × 20 次同内容，60 个 id 全同。"""
    e = _engine()
    got = _run_writers(e, 3, 20, "shared content from writers")
    assert len(got) == 60
    assert len(set(got)) == 1, (
        "并发同内容写产出 %d 个不同节点（TOCTOU 未封死）：%r"
        % (len(set(got)), sorted(set(got))))


def test_b_high_concurrency_single_id_and_merged_flag():
    """B：8 线程 × 10 次同内容——80 个 id 全同，非首个返回 merged=True。"""
    e = _engine()
    got = _run_writers(e, 8, 10, "hot content")
    assert len(got) == 80
    assert len(set(got)) == 1, "高并发下仍分裂：%d 个 id" % (len(set(got)),)
    again = e.add_perception("hot content")
    assert again.id == got[0]
    assert getattr(again, "merged", False), "合并路径 merged 标记丢失"


def test_c_store_holds_single_node_for_content():
    """C：并发写后，知识层同模态下该内容只有一个节点。"""
    e = _engine()
    _run_writers(e, 4, 15, "counted content")
    nodes = e.store.query_nodes(layer=MemoryLayer.KNOWLEDGE,
                                modality="text", limit=-1)
    same = [n for n in nodes if n.content == "counted content"]
    assert len(same) == 1, (
        "库内同内容节点数 = %d（期望 1）：%r" % (len(same), [n.id for n in same]))


def test_d_sequential_behavior_unchanged():
    """D：顺序写不回归——重复合并、不同内容另立。"""
    e = _engine()
    a = e.add_perception("sequential same content")
    b = e.add_perception("sequential same content")
    c = e.add_perception("totally different content")
    assert b.id == a.id, "顺序重复写不再合并（非竞态面被误改）"
    assert c.id != a.id, "不同内容被误并（判同键被误改）"


def test_e_reentrant_lock_no_self_deadlock():
    """E：RLock 同线程可重入——语义 provider 在判同临界区内再入不挂死。

    以「持锁后同线程再调 add_perception」直接钉住可重入性（若有人把 RLock
    降级为 Lock，此处超时死亡/死锁异常即红）。
    """
    e = _engine()
    done = threading.Event()

    def reentrant():
        with e._m5_dedup_lock:
            n = e.add_perception("reentrant write inside lock")
            assert n.id
        done.set()

    t = threading.Thread(target=reentrant)
    t.daemon = True
    t.start()
    joined = done.wait(timeout=10.0)
    t.join(timeout=5.0)
    assert joined, "同线程持锁再入 add_perception 死锁（锁不可重入）"
