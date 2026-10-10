#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bench_retrieval.py —— 检索与召回的自包含基准（零外部语料，固定种子，可复现）。

两组场景，按当前环境开关运行（LINGSHU_FTS / LINGSHU_RETRIEVAL_NO_TOUCH / LINGSHU_CONTRADICTION）：

  reach      可达性：N 条合成干扰 + M 条目标事实（随机插入位置），用自然语言问句检索，
             报告 search_content / recall 的 hit@1、hit@10、p50 延迟。
  supersede  知识更新：M 个「实体·属性」旧值写于 200 天前，先被问到 K 次，再写入新值，
             问「现在的X是什么」，报告 recall top-1 为新值的比例。

用法：
    python bench/bench_retrieval.py reach 5000
    python bench/bench_retrieval.py supersede 200
    LINGSHU_FTS=1 LINGSHU_RETRIEVAL_NO_TOUCH=1 python bench/bench_retrieval.py supersede 200

边界（如实）：合成语料比真实对话容易，读数用于开关前后对照，不代表绝对水平。
"""
import json
import os
import random
import sys
import tempfile
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lingshu.core.core import SpacetimeMemoryEngine  # noqa: E402

SUR = "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜"
GIV = "明华强伟芳娜敏静丽军磊洋勇艳杰涛超秀英霞平刚桂兰"
TOPIC = ["天气", "午饭", "电影", "球赛", "地铁", "快递", "健身", "会议", "咖啡", "周末"]
ATTRS = {
    "住址": lambda r: f"{r.choice(['东城', '西城', '南湖', '北山', '江岸', '新港'])}区{r.randint(1, 99)}号",
    "电话": lambda r: f"1{r.randint(3, 9)}{r.randint(100000000, 999999999)}",
    "职位": lambda r: r.choice(["工程师", "主管", "经理", "研究员", "顾问", "总监", "分析师"]),
    "负责的项目": lambda r: r.choice(["蜂巢", "灯塔", "北极星", "青鸟", "磐石", "潮汐", "远航"]) + "计划",
    "车牌": lambda r: f"京{r.choice('ABCDEFG')}{r.randint(10000, 99999)}",
}
TPL = ["{e}的{a}是{v}", "记录：{e}的{a}为{v}", "关于{e}：{a}={v}", "据了解，{e}目前的{a}是{v}"]
DAY = 86400.0


def _name(r):
    return r.choice(SUR) + r.choice(GIV) + r.choice(GIV)


def _filler(r):
    return (f"{_name(r)}说今天的{r.choice(TOPIC)}{r.choice(['还不错', '有点糟', '一般般', '挺意外'])}，"
            f"顺便提到{r.choice(TOPIC)}和{r.choice(TOPIC)}的事")


def _engine():
    return SpacetimeMemoryEngine(os.path.join(tempfile.mkdtemp(prefix="bench_ret_"), "b.db"))


def _flags():
    return {k: os.environ.get(k, "0") for k in
            ("LINGSHU_FTS", "LINGSHU_RETRIEVAL_NO_TOUCH", "LINGSHU_CONTRADICTION")}


def _facts(r, m):
    out, seen = [], set()
    while len(out) < m:
        e, a = _name(r), r.choice(list(ATTRS))
        if (e, a) in seen:
            continue
        seen.add((e, a))
        v1 = ATTRS[a](r)
        v2 = ATTRS[a](r)
        while v2 == v1:
            v2 = ATTRS[a](r)
        out.append((e, a, v1, v2))
    return out


def reach(n_fill, m=100, seed=1):
    r = random.Random(seed)
    facts = _facts(r, m)
    docs = [(None, _filler(r)) for _ in range(n_fill)]
    for i, (e, a, v, _) in enumerate(facts):
        docs.insert(r.randint(0, len(docs)), (i, r.choice(TPL).format(e=e, a=a, v=v)))
    eng = _engine()
    gold = {}
    for i, text in docs:
        nid = eng.add_perception(text, importance=0.5, skip_dedup=True).id
        if i is not None:
            gold[i] = nid
    res = {}
    for name, fn in (("search_content", lambda q: eng.store.search_content(q, limit=10)),
                     ("recall", lambda q: eng.recall(q, limit=10))):
        h1 = h10 = 0
        lat = []
        for i, (e, a, _, _) in enumerate(facts):
            t = time.perf_counter()
            ids = [n.id for n, _ in fn(f"{e}的{a}是什么？")]
            lat.append(time.perf_counter() - t)
            h1 += bool(ids) and ids[0] == gold[i]
            h10 += gold[i] in ids
        lat.sort()
        res[name] = {"hit@1": round(h1 / m, 3), "hit@10": round(h10 / m, 3),
                     "p50_ms": round(1000 * lat[len(lat) // 2], 2)}
    return {"suite": "reach", "pool": len(docs), **_flags(), **res}


def supersede(m, k=1, n_fill=1000, seed=3):
    r = random.Random(seed)
    facts = _facts(r, m)
    eng = _engine()
    st = eng.store
    old = {}
    for e, a, v1, _ in facts:
        old[(e, a)] = eng.add_perception(r.choice(TPL).format(e=e, a=a, v=v1),
                                         importance=0.5, skip_dedup=True).id
    for _ in range(n_fill):
        eng.add_perception(_filler(r), importance=0.5, skip_dedup=True)
    t200 = time.time() - 200 * DAY
    st.conn.execute("UPDATE nodes SET created_at=?, last_access=?", (t200, t200))
    st.conn.commit()
    for _ in range(k):  # 更新前，旧事实被问到 k 次
        for e, a, _, _ in facts:
            eng.recall(f"{e}的{a}是什么？", limit=10)
    new = {(e, a): eng.add_perception(r.choice(TPL).format(e=e, a=a, v=v2),
                                      importance=0.5, skip_dedup=True).id
           for e, a, _, v2 in facts}
    top1 = 0
    for e, a, _, _ in facts:
        ids = [n.id for n, _ in eng.recall(f"{e}现在的{a}是什么？", limit=10)]
        top1 += bool(ids) and ids[0] == new[(e, a)]
    return {"suite": "supersede", "facts": m, "asked_before_update": k, **_flags(),
            "new_value_top1": round(top1 / m, 3)}


if __name__ == "__main__":
    suite = sys.argv[1] if len(sys.argv) > 1 else "reach"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else (2000 if suite == "reach" else 200)
    out = reach(n) if suite == "reach" else supersede(n)
    print(json.dumps(out, ensure_ascii=False))
