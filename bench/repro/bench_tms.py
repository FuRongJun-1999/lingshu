# -*- coding: utf-8 -*-
"""TMS 依赖撤回实测（构造场景，如实）：随机生成推理 DAG（含析取与少量环），推翻一部分基础事实，
对照「仅证据账本」与「账本+TMS」：应撤回结论的召回率 / 误撤率，以及 recall_with_evidence 是否还把它们当事实给出。
真值：结论应撤回 ⇔ 它的每条依据都含一个「应撤回」的前提（对基础事实：被推翻）；按不动点求。
用法：python bench_tms.py n_facts n_concl refute_frac use_tms(0/1)
"""
import sys,os,json,random,time,tempfile
from common import SpacetimeMemoryEngine, emit
NF,NC,FR,T=int(sys.argv[1]),int(sys.argv[2]),float(sys.argv[3]),sys.argv[4]=='1'
r=random.Random(21)
e=SpacetimeMemoryEngine(os.path.join(tempfile.mkdtemp(),'t.db')); e.enable_evidence_ledger()
if T: e.enable_tms()
facts=[e.add_perception(f"事实{i}：观测记录编号{i}的内容成立",source=f"src:{i}",skip_dedup=True).id for i in range(NF)]
for i,f in enumerate(facts): e.add_evidence(f,f"src2:{i}")          # 每条基础事实两源佐证
concl=[e.add_perception(f"结论{i}：由若干观测推出的判断{i}",source="self:reason",self_generated=True,skip_dedup=True).id for i in range(NC)]
just={}
pool=list(facts)
t0=time.perf_counter()
for i,c in enumerate(concl):
    js=[]
    for _ in range(1 if r.random()<0.7 else 2):                       # 30% 析取
        k=r.randint(1,3); js.append(r.sample(pool,k))
    if i>10 and r.random()<0.02: js.append([r.choice(concl[i+1:] or concl[:1])])  # 少量前向/环
    just[c]=js
    for j in js:
        if T: e.add_justification(c,j)
    pool.append(c)
tj=time.perf_counter()-t0
refuted=set(r.sample(facts,int(FR*NF)))
t0=time.perf_counter()
for f in refuted:
    for s in ["x%d"%k for k in range(8)]: e.add_evidence(f,s,supports=False)
tr=time.perf_counter()-t0
# 真值不动点
bad=set(refuted)
while True:
    nb={c for c,js in just.items() if c not in bad and all(any(p in bad for p in j) for j in js)}
    if not nb: break
    bad|=nb
should={c for c in concl if c in bad}; ok={c for c in concl if c not in bad}
def und(c):
    rep=e.evidence_report(c); return rep['status'] in ('undermined','refuted') or rep['confidence']<0.3
hit=sum(und(c) for c in should); fp=sum(und(c) for c in ok)
emit(dict(suite='tms',tms=int(T),基础事实=NF,结论=NC,推翻比例=FR,应撤回=len(should),撤回召回=round(hit/max(1,len(should)),3),误撤=f"{fp}/{len(ok)}",
   登记依据耗时s=round(tj,2),推翻传播耗时ms_每次=round(1000*tr/max(1,len(refuted)*8),2)))
