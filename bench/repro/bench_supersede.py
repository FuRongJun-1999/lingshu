# -*- coding: utf-8 -*-
"""知识更新（新事实取代旧事实）场景：检验 search→last_access→recall 近因 的自增强。
构造场景（如实）：M 个「实体·属性」，旧值写于 200 天前；更新前先用查询流把旧事实检索 K 次
（真实使用里，旧事实在更新前常被反复问到）；然后写入新值（当前时刻），再问「现在的X是什么」。
判据：recall top-1 是新值的比例；新值排在旧值之前的比例。新旧句式从同一模板池独立抽取，无系统偏置。
写入用 skip_dedup=True（M5 会把只差数字的新旧事实合并，见 #142，那是另一个问题）。
用法：[开关…] python bench_supersede.py M K distractorsN
"""
import sys,os,json,time,random,tempfile
from common import SpacetimeMemoryEngine, kdconv, emit
M,K,N=int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3])
r=random.Random(3)
sur="赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜"; giv="明华强伟芳娜敏静丽军磊洋勇艳杰涛超秀英霞平刚桂兰"
attrs={"住址":lambda:f"{r.choice(['东城','西城','南湖','北山','江岸','新港'])}区{r.randint(1,99)}号","电话":lambda:f"1{r.randint(3,9)}{r.randint(100000000,999999999)}",
 "职位":lambda:r.choice(['工程师','主管','经理','研究员','顾问','总监','分析师']),"负责的项目":lambda:r.choice(['蜂巢','灯塔','北极星','青鸟','磐石','潮汐','远航'])+"计划",
 "车牌":lambda:f"京{r.choice('ABCDEFG')}{r.randint(10000,99999)}","常去的咖啡馆":lambda:r.choice(['蓝瓶','角落','半岛','山野','白塔','慢时光'])+"咖啡"}
tpl=["{e}的{a}是{v}","记录：{e}的{a}为{v}","{e}告诉我，{e}的{a}是{v}","关于{e}：{a}={v}","据了解，{e}目前的{a}是{v}"]
facts=[];seen=set()
while len(facts)<M:
    e=r.choice(sur)+r.choice(giv)+r.choice(giv); a=r.choice(list(attrs))
    if (e,a) in seen: continue
    seen.add((e,a)); v1=attrs[a](); v2=attrs[a]()
    while v2==v1: v2=attrs[a]()
    facts.append((e,a,v1,v2))
e_=SpacetimeMemoryEngine(os.path.join(tempfile.mkdtemp(),'s.db')); st=e_.store; now=time.time()
old={};new={}
for (e,a,v1,v2) in facts:
    n=e_.add_perception(r.choice(tpl).format(e=e,a=a,v=v1),importance=0.5,skip_dedup=True); old[(e,a)]=n.id
if N:
    for t in kdconv(N): e_.add_perception(t,importance=0.5,skip_dedup=True)
t200=now-200*86400
st.conn.execute("UPDATE nodes SET created_at=?, last_access=?",(t200,t200)); st.conn.commit()
for _ in range(K):                       # 更新前：旧事实被问到 K 次
    for (e,a,_,_) in facts: e_.recall(f"{e}的{a}是什么？",limit=10)
for (e,a,v1,v2) in facts:                # 写入更新
    new[(e,a)]=e_.add_perception(r.choice(tpl).format(e=e,a=a,v=v2),importance=0.5,skip_dedup=True).id
top1=ahead=0
for (e,a,_,_) in facts:
    res=[n.id for n,_ in e_.recall(f"{e}现在的{a}是什么？",limit=10)]
    top1+= bool(res) and res[0]==new[(e,a)]
    pn=res.index(new[(e,a)]) if new[(e,a)] in res else 99; po=res.index(old[(e,a)]) if old[(e,a)] in res else 99
    ahead+= pn<po
emit(dict(suite='supersede',M=M,K=K,N=N,新值top1=round(top1/M,3),新值在旧值前=round(ahead/M,3)))
