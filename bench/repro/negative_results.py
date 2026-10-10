# -*- coding: utf-8 -*-
"""试过但没有并入的方案（负结果），供维护者复核。

在离线参考实现上比较（纯 Python BM25，字符二元组；不走引擎，数值与 bench_search 不同口径，只做同框架内对比）：
  base        BM25 + 问句功能词过滤 + 同章时序邻接平滑 γ=0.2（＝本 PR 语境扩展的离线等价物）
  entity      实体桥：前 k 块里查询外的高 IDF 二元组作第二路查询，补进后 4 位
  ppr         图桥：同章相邻 + 共享≥3 个高 IDF 二元组建图，以 base 分数做种子跑个性化 PageRank 后混合
  docexp      文档扩展：每块并入前后块的高 IDF 二元组（权重 0.15）
  dense_rrf   base ⊕ bge-small-zh-v1.5 稠密检索，RRF（需 pip install fastembed，未安装则跳过）
所有参数只在切A 上选，切B 为留出集。用法：python negative_results.py
"""
import collections, math, os, sys
import numpy as np
from common import build_set, evaluate, emit, bigrams

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from lingshu.core.retrieval import QUERY_FUNCTION_GRAMS as QSTOP  # noqa: E402


def qgrams(q):
    out = []
    for g in bigrams(q):
        if g not in out:
            out.append(g)
    return [g for g in out if g not in QSTOP] or out


class Ref:
    def __init__(self, chunks):
        self.ch = chunks
        self.N = N = len(chunks)
        self.tf = [collections.Counter(bigrams(t)) for _, _, t in chunks]
        self.bg = [set(c) for c in self.tf]
        df = collections.Counter(g for s in self.bg for g in s)
        self.idf = {g: math.log((N - d + .5) / (d + .5) + 1) for g, d in df.items()}
        self.dl = np.array([sum(c.values()) for c in self.tf])
        self.prev = [i - 1 if i > 0 and chunks[i - 1][0] == chunks[i][0] else -1 for i in range(N)]
        self.next = [i + 1 if i < N - 1 and chunks[i + 1][0] == chunks[i][0] else -1 for i in range(N)]
        self.set_docs(self.tf)
        A = np.zeros((N, N))
        for i in range(N):
            if self.next[i] >= 0:
                A[i, i + 1] = A[i + 1, i] = 1.0
        rare = [{g for g in s if self.idf[g] > math.log(N / 6)} for s in self.bg]
        for i in range(N):
            for j in range(i + 1, N):
                w = len(rare[i] & rare[j])
                if w >= 3:
                    A[i, j] = A[j, i] = A[i, j] + min(1.0, w / 10)
        d = A.sum(1)
        d[d == 0] = 1
        self.P = A / d[:, None]

    def set_docs(self, tf):
        self.post = collections.defaultdict(list)
        for i, c in enumerate(tf):
            for g, f in c.items():
                self.post[g].append((i, f))
        self.dlx = np.array([sum(c.values()) for c in tf])
        self.avg = self.dlx.mean()

    def bm(self, terms, k1=1.2, b=0.75):
        sc = np.zeros(self.N)
        for t in set(terms):
            for i, f in self.post.get(t, ()):
                sc[i] += self.idf[t] * f * (k1 + 1) / (f + k1 * (1 - b + b * self.dlx[i] / self.avg))
        return sc

    def smooth(self, sc, gam=0.2):
        nb = np.array([max(sc[p] if p >= 0 else 0, sc[n] if n >= 0 else 0) for p, n in zip(self.prev, self.next)])
        return sc + gam * nb

    @staticmethod
    def top(s, k=10):
        return [int(i) for i in np.argsort(-s, kind="stable")[:k] if s[i] > 0]

    def base(self, q):
        return self.top(self.smooth(self.bm(qgrams(q))))

    def entity(self, q, main=6, seeds=3, nterm=8, thr=5):
        t = qgrams(q)
        s = self.smooth(self.bm(t))
        order = self.top(s, self.N)
        tf = collections.Counter()
        for r, i in enumerate(order[:seeds]):
            for g in self.bg[i] - set(t):
                if self.idf[g] > math.log(self.N / thr):
                    tf[g] += self.idf[g] / (1 + r)
        s2 = self.bm([g for g, _ in tf.most_common(nterm)])
        out = order[:main]
        for i in self.top(s2, self.N):
            if len(out) >= 10:
                break
            if i not in out:
                out.append(i)
        return (out + [i for i in order if i not in out])[:10]

    def ppr(self, q, alpha=0.3, w=0.4, k=30):
        s = self.smooth(self.bm(qgrams(q)))
        if s.max() <= 0:
            return []
        v = np.zeros(self.N)
        idx = np.argsort(-s)[:k]
        v[idx] = s[idx]
        v = v / v.sum()
        x = v.copy()
        for _ in range(20):
            x = alpha * v + (1 - alpha) * x @ self.P
        return self.top(s / s.max() + w * x / x.max())

    def docexp(self, q, wn=0.15, thr=10):
        tf2 = []
        for i in range(self.N):
            c = collections.Counter({g: float(f) for g, f in self.tf[i].items()})
            for j in (self.prev[i], self.next[i]):
                if j >= 0:
                    for g, f in self.tf[j].items():
                        if self.idf[g] > math.log(self.N / thr):
                            c[g] += wn * f
            tf2.append(c)
        self.set_docs(tf2)
        try:
            return self.base(q)
        finally:
            self.set_docs(self.tf)


def main():
    try:
        from fastembed import TextEmbedding
        M = TextEmbedding("BAAI/bge-small-zh-v1.5")
    except Exception:
        M = None
    for cut, m in (("A", 8), ("B", 16)):
        chunks, qs = build_set(m)
        R = Ref(chunks)
        for name in ("base", "entity", "ppr", "docexp"):
            r = evaluate(chunks, qs, getattr(R, name))
            r.update(suite=f"negative_{cut}", method=name)
            emit(r)
        if M is not None:
            E = np.array(list(M.embed([t for _, _, t in chunks])))
            E /= np.linalg.norm(E, axis=1, keepdims=True)

            def dense_rrf(q, wd=1.0, k=60):
                v = np.array(list(M.embed(["为这个句子生成表示以用于检索相关文章：" + q])))[0]
                d = E @ (v / np.linalg.norm(v))
                sc = collections.defaultdict(float)
                for rank, i in enumerate(R.top(R.smooth(R.bm(qgrams(q))), k)):
                    sc[i] += 1 / (60 + rank)
                for rank, i in enumerate(np.argsort(-d)[:k]):
                    sc[int(i)] += wd / (60 + rank)
                return sorted(sc, key=lambda i: -sc[i])[:10]
            r = evaluate(chunks, qs, dense_rrf)
            r.update(suite=f"negative_{cut}", method="dense_rrf")
            emit(r)


main()
