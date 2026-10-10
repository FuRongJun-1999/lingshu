# -*- coding: utf-8 -*-
"""检索层：store.search_content top-10（PR 的 FTS5 / 语境扩展部分）。

用法：[LINGSHU_FTS=1] [LINGSHU_FTS_CONTEXT=1] [ORDER=seq|shuffle] python bench_search.py <suite> [arg]
  hmb A | hmb B          hive-memory-bench 切A / 切B
  hmb B_full             切B 题目，全书 23 章入库（同领域干扰）
  hmbscale N             切B + N 条 KdConv 跨领域干扰
  locomo N               locomo-zh-500（关键词串查询）+ N 条 KdConv 干扰
ORDER=seq（默认）按语料顺序写入，等价于对话/叙述按时间到达；shuffle 打乱写入顺序。
"""
import sys
from common import build_set, evaluate, kdconv, locomo, engine, emit, pct, time, random, os


def ingest(e, texts, seed):
    idx = list(range(len(texts)))
    if os.environ.get("ORDER", "seq") == "shuffle":
        random.Random(seed).shuffle(idx)
    m = {}
    for i in idx:
        m[e.add_perception(texts[i], importance=0.5, skip_dedup=True).id] = i
    return m


def hmb(chunks, qs, extra, name):
    e = engine()
    m = ingest(e, [t for _, _, t in chunks] + extra, 11)
    lat = []

    def fn(q):
        t = time.perf_counter()
        r = [m[n.id] for n, _ in e.store.search_content(q, limit=10)]
        lat.append(time.perf_counter() - t)
        return [i for i in r if i < len(chunks)]
    e.store.search_content("预热", limit=1)
    r = evaluate(chunks, qs, fn)
    r.update(suite=name, pool=len(chunks) + len(extra), p50_ms=round(1000 * pct(lat, .5), 2))
    return r


def main():
    s = sys.argv[1]
    if s == "hmb":
        cut = sys.argv[2]
        if cut == "B_full":
            chunks, qs = build_set(16)
            full, _ = build_set(99)
            r = hmb(chunks, qs, [t for (_, no, t) in full if no > 16], "hmb_B_full")
        else:
            chunks, qs = build_set(8 if cut == "A" else 16)
            r = hmb(chunks, qs, [], f"hmb_{cut}")
    elif s == "hmbscale":
        n = int(sys.argv[2])
        chunks, qs = build_set(16)
        r = hmb(chunks, qs, kdconv(n), f"hmbscale_{n}")
    elif s == "locomo":
        n = int(sys.argv[2])
        corpus, qs = locomo()
        texts = [c["zh"] for c in corpus] + kdconv(n)
        ids = [c["id"] for c in corpus] + [None] * n
        e = engine()
        os.environ.setdefault("ORDER", "shuffle")  # locomo 语料本无叙述顺序
        m = ingest(e, texts, n)
        h1 = h10 = 0
        mrr = 0.0
        lat = []
        for q in qs:
            g = set(q["evidence_turns"])
            t = time.perf_counter()
            rk = [ids[m[x.id]] for x, _ in e.store.search_content(q["question"], limit=10)]
            lat.append(time.perf_counter() - t)
            p = next((i for i, d in enumerate(rk) if d in g), None)
            if p is not None:
                h10 += 1
                h1 += p == 0
                mrr += 1 / (p + 1)
        N = len(qs)
        r = dict(suite=f"locomo_{n}", pool=len(texts), hit1=round(h1 / N, 3), hit10=round(h10 / N, 3),
                 mrr10=round(mrr / N, 3), p50_ms=round(1000 * pct(lat, .5), 2))
    r["order"] = os.environ.get("ORDER", "seq")
    emit(r)


main()
