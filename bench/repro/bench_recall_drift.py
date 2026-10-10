# -*- coding: utf-8 -*-
"""recall 端到端 + 近因自增强漂移（PR 的「检索不刷新近因」部分）。

语料写入后把每个节点的 created_at/last_access 均匀打散到过去 1–365 天（固定种子）；每轮把全部题目
打乱后逐题调用 engine.recall(q, limit=10)。第 1 轮是冷启动，之后各轮暴露 search→last_access→recall
的自增强。ORDER=seq（默认）按语料顺序写入。用法：[开关…] python bench_recall_drift.py <A|B> <rounds>
"""
import sys
from common import build_set, evaluate, engine, emit, pct, time, random, os

cut, R = sys.argv[1], int(sys.argv[2])
chunks, qs = build_set(8 if cut == "A" else 16)
e = engine()
idx = list(range(len(chunks)))
if os.environ.get("ORDER", "seq") == "shuffle":
    random.Random(11).shuffle(idx)
m = {}
for i in idx:
    m[e.add_perception(chunks[i][2], importance=0.5, skip_dedup=True).id] = i
rnd = random.Random(5)
now = time.time()
e.store.conn.executemany("UPDATE nodes SET created_at=?, last_access=? WHERE id=?",
                         [(now - rnd.uniform(1, 365) * 86400,) * 2 + (nid,) for nid in m])
e.store.conn.commit()
rounds = []
for r in range(R):
    q2 = list(qs)
    random.Random(100 + r).shuffle(q2)
    lat = []

    def fn(q):
        t = time.perf_counter()
        res = e.recall(q, limit=10)
        lat.append(time.perf_counter() - t)
        return [m[n.id] for n, _ in res if n.id in m]
    x = evaluate(chunks, q2, fn)
    rounds.append(dict(round=r + 1, 证据保留=x["证据保留"], 另一侧覆盖=x["另一侧覆盖"], 块级命中=x["块级命中"],
                       p50_ms=round(1000 * pct(lat, .5), 2)))
emit(dict(suite=f"recall_{cut}", rounds=rounds))
