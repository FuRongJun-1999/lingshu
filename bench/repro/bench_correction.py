# -*- coding: utf-8 -*-
"""更正 / 矛盾场景（Refs #142）：M5 去重是否吞掉更正，矛盾两侧能否同时召回，同义复述是否被误拆。
LINGSHU_CONTRADICTION 由环境给定。用法：[LINGSHU_CONTRADICTION=1] [LINGSHU_FTS=1] python bench_correction.py n_per_type K
构造数据（如实）：6 类事实模板 × n 条；每条事实写 1 次，再写其更正 K 次（极性 / 数值 / 反义）；
另写同义复述（标点、语气词、互换豁免词，且都不改变真值），应被 M5 合并。
"""
import sys,os,json,random,tempfile
from common import SpacetimeMemoryEngine, emit
n,K=int(sys.argv[1]),int(sys.argv[2]); r=random.Random(9)
P=[a+b for a in "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨" for b in ["明华","建国","思远","雨桐","子涵","浩然","欣怡","一鸣"]]
D=["青霉素","阿莫西林","头孢","布洛芬","阿司匹林","磺胺类药物"]
def gen(t):
    p=r.choice(P);d=r.choice(D);x=r.randint(2,60);y=x+r.choice([1,2,5,10,-1]);city=r.choice(["华东","华南","西北","东北","西南"])
    if t==0: return (f"{p}的体检报告显示，{p}对{d}严重过敏，开药时需要特别注意", f"{p}的体检报告显示，{p}对{d}不过敏，开药时需要特别注意","polarity",f"{p}对{d}过敏吗")
    if t==1: return (f"{p}每天服用{d}，剂量为{x}毫克，饭后半小时服用，连续两周", f"{p}每天服用{d}，剂量为{y}毫克，饭后半小时服用，连续两周","number",f"{p}的{d}剂量是多少")
    if t==2: return (f"编号{x}{p[0]}的安全漏洞已经修复，补丁在上周随版本发布", f"编号{x}{p[0]}的安全漏洞尚未修复，补丁在上周随版本发布","polarity",f"编号{x}{p[0]}的漏洞修复了吗")
    if t==3: return (f"{p}提交的年度预算方案在部门会议上获得通过，下个月开始执行", f"{p}提交的年度预算方案在部门会议上被否决，下个月开始执行","polarity",f"{p}的预算方案结果如何")
    if t==4: return (f"{city}{x}号订单服务器目前处于在线状态，负责处理{city}地区的订单", f"{city}{x}号订单服务器目前处于离线状态，负责处理{city}地区的订单","antonym",f"{city}{x}号服务器状态")
    if t==5: return (f"{p}乘坐的航班定于{x%12+1}点起飞，登机口在{x}号，需要提前到达", f"{p}乘坐的航班定于{y%12+1}点起飞，登机口在{x}号，需要提前到达","number",f"{p}的航班几点起飞")
def para(s):
    k=r.randint(0,2)
    if k==0: return s+"。"
    if k==1: return s.replace("，",",",1)
    return s.replace("特别注意","非常注意") if "特别注意" in s else s+"啊"
e=SpacetimeMemoryEngine(os.path.join(tempfile.mkdtemp(),'c.db'))
items=[];_seen=set()
while len(items)<6*n:
    it=gen(len(items)//n)
    if it[0] in _seen or it[1] in _seen: continue
    _seen.update(it[:2]); items.append(it)
kept=both=infl=0; ids=[]; byt={}
for (a,b,typ,q) in items:
    na=e.add_perception(a,importance=0.5)
    for _ in range(K): nb=e.add_perception(b,importance=0.5)
    ok=nb.id!=na.id; kept+=ok
    conf=e.store.get_node(na.id).confidence; infl+= conf>0.5+1e-9
    res=[x.id for x,_ in e.recall(q,limit=5)]
    bb= ok and na.id in res and nb.id in res; both+=bb
    t=byt.setdefault(typ,[0,0,0]); t[0]+=1; t[1]+=ok; t[2]+=bb
# 同义复述（标点 / 语气词 / 豁免词互换，真值不变）：检测器直接判一遍，看误报
from lingshu.core.contradiction import conflict as _C
split=merged_base=0
for (a,b,typ,q) in items:
    for _ in range(3):
        s2=para(a); merged_base+=1; split+= _C(a,s2) is not None
N=len(items)
emit(dict(suite='correction',事实数=N,更正保留=round(kept/N,3),原事实被更正增信=round(infl/N,3),矛盾两侧同时召回=round(both/N,3),
  分类型={k:f"{v[1]}/{v[0]} 保留, {v[2]}/{v[0]} 两侧" for k,v in byt.items()},同义复述误拆=f"{split}/{merged_base}"))
