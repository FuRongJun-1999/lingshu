# -*- coding: utf-8 -*-
"""把 results/raw.jsonl 汇总成 markdown 表。"""
import json, sys

rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
out = []
P = out.append
CFG = {"base": "改前", "fts": "FTS5", "fts_ctx": "FTS5+语境扩展", "no_touch": "不刷新近因", "fts_no_touch": "FTS5+不刷新近因",
       "contra": "矛盾感知", "fts_contra": "FTS5+矛盾感知", "all": "全开"}
P("## 1. 检索层 search_content top-10\n")
P("| 套件 | 配置 | 证据保留 | 另一侧覆盖 | 块级命中 | hit@10 | MRR@10 | p50 ms |\n|---|---|---|---|---|---|---|---|")
for r in rows:
    if "suite" in r and r["suite"].startswith(("hmb", "locomo")) and "rounds" not in r:
        P(f"| {r['suite']}{'（打乱写入）' if r.get('order')=='shuffle' and r['suite'].startswith('hmb') else ''} | {CFG[r['cfg']]} | {r.get('证据保留','')} | {r.get('另一侧覆盖','')} | {r.get('块级命中','')} | "
          f"{r.get('hit10','')} | {r.get('mrr10','')} | {r['p50_ms']} |")
P("\n## 2. recall 端到端（每轮全部题目各查一次）\n")
P("| 切 | 配置 | 轮 | 证据保留 | 另一侧覆盖 | 块级命中 | p50 ms |\n|---|---|---|---|---|---|---|")
for r in rows:
    if r.get("suite", "").startswith("recall_"):
        for x in r["rounds"]:
            P(f"| {r['suite'][-1]} | {CFG[r['cfg']]} | {x['round']} | {x['证据保留']} | {x['另一侧覆盖']} | {x['块级命中']} | {x['p50_ms']} |")
P("\n## 3. 知识更新（200 个实体·属性，旧值 200 天前写入并被查询 1 次）\n")
P("| 配置 | 新值 top-1 | 新值排在旧值前 |\n|---|---|---|")
for r in rows:
    if r.get("suite") == "supersede":
        P(f"| {CFG[r['cfg']]} | {r['新值top1']} | {r['新值在旧值前']} |")
P("\n## 4. 更正 / 矛盾（6 类 × 30 条事实，每条写 5 次更正）\n")
P("| 配置 | 更正保留 | 原事实被更正增信 | 矛盾两侧同时召回 | 同义复述误拆 | 分类型 |\n|---|---|---|---|---|---|")
for r in rows:
    if r.get("suite") == "correction":
        P(f"| {CFG[r['cfg']]} | {r['更正保留']} | {r['原事实被更正增信']} | {r['矛盾两侧同时召回']} | {r['同义复述误拆']} | "
          + "；".join(f"{k} {v}" for k, v in r["分类型"].items()) + " |")
P("\n## 5. 真值维护（随机推理 DAG，推翻 10% 基础事实）\n")
P("| 规模 | TMS | 应撤回 | 撤回召回 | 误撤 | 传播 ms/条证据 |\n|---|---|---|---|---|---|")
for r in rows:
    if r.get("suite") == "tms":
        P(f"| {r['基础事实']}/{r['结论']} | {'开' if r['tms'] else '关（仅账本）'} | {r['应撤回']} | {r['撤回召回']} | {r['误撤']} | {r['推翻传播耗时ms_每次']} |")
print("\n".join(out))
