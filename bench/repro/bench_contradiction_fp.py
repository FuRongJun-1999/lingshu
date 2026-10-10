# -*- coding: utf-8 -*-
"""用法：python bench_contradiction_fp.py（检测器本身是纯函数，不需要开关）
真实语料上的误报：在 hmb 全书块、locomo-zh-567、KdConv 前 2 万句里找 Jaccard≥0.85 的近重复对（即 M5 会合并的对），
统计检测器判为冲突的比例，并打印样例供人工核对。"""
import sys,json,os,itertools,collections
from common import build_set, locomo, kdconv, DATA
from lingshu.core.contradiction import conflict
from lingshu.core.core import LayeredStore
J=LayeredStore.char_bigram_jaccard
def bg(s): s="".join(s.split()); return {s[i:i+2] for i in range(len(s)-1)}
def pairs(texts):
    inv=collections.defaultdict(list); B=[bg(t) for t in texts]
    for i,b in enumerate(B):
        for g in b: inv[g].append(i)
    out=set()
    for i,b in enumerate(B):
        cnt=collections.Counter()
        for g in b:
            if len(inv[g])<200: cnt.update(j for j in inv[g] if j>i)
        for j,c in cnt.items():
            if c>=0.8*min(len(b),len(B[j])) and J(texts[i],texts[j])>=0.85: out.add((i,j))
    return out
_kd=[json.loads(l)['t'] for l in open(os.path.join(DATA,'kdconv_utts.jsonl'),encoding='utf-8')]
sets={'hmb':[t for _,_,t in build_set(99)[0]],'locomo':[c['zh'] for c in locomo()[0]],
      'kdconv20k':_kd[:20000],'kdconv_heldout':_kd[20000:]}
for k,T in sets.items():
    P=pairs(T); f=[(T[i],T[j],conflict(T[i],T[j])) for i,j in P]; hit=[x for x in f if x[2]]
    print(k,'近重复对',len(P),'判冲突',len(hit))
    for a,b,rs in hit[:8]: print('  ',rs,'|',a[:40],'||',b[:40])
