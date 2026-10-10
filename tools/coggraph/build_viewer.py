#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""认知图可视化 · 构建器（graph.json + derived_edges.json → viewer_graph.json + index.html）

布局在 Python 端确定性预计算（桶网格 + 黄金角散布），前端只负责渲染与过滤。
cytoscape.min.js 首次运行时自动下载到输出目录（离线可用）。
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import math
import os
import sys
import time
import urllib.request

CYTO_URL = "https://cdn.jsdelivr.net/npm/cytoscape@3.30.2/dist/cytoscape.min.js"

LAYER_COLOR = {
    "knowledge": "#4C8DFF", "contextual": "#F5A623", "structural": "#9B59B6",
    "self": "#2ECC71", "anchor": "#E74C3C", "goals": "#1ABC9C",
    "rejected": "#7F8C8D", "unresolved": "#E67E22", "hub": "#BDC3C7",
}
RULE_COLOR = {
    "explicit": "#5D6D7E", "R1_same_bucket": "#85C1E9", "R1_same_bucket_pair": "#5DADE2",
    "R2_same_source": "#48C9B0", "R3_tag_jaccard": "#F7DC6F", "R4_same_day": "#D7BDE2",
    "part_of": "#E59866", "similar": "#A569BD", "hierarchical": "#7FB3D5",
    "causal": "#EC7063", "derived_from": "#F1948A",
    "sequential": "#FFB4A2", "R6_session_chain": "#FFB4A2", "R6_session_hub": "#FFD6A5",
    "in_session": "#FFD6A5", "same_source": "#48C9B0", "in_bucket": "#85C1E9",
    "tag_similar": "#F7DC6F", "same_day": "#D7BDE2", "body_crossref": "#82E0AA",
}
HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>灵枢认知图</title>
<style>
 html,body{margin:0;width:100%;height:100%;font:13px/1.5 "PingFang SC","Microsoft YaHei",system-ui,sans-serif;background:#0f1115;color:#dfe3ea}
 body{display:flex}
 #bar{width:280px;flex:none;background:#151922;border-right:1px solid #232936;padding:12px;overflow-y:auto;box-sizing:border-box}
 #bar h1{font-size:15px;margin:0 0 8px} #bar h2{font-size:12px;margin:14px 0 6px;color:#8fa0b8;text-transform:uppercase;letter-spacing:.05em}
 label{display:block;margin:3px 0;cursor:pointer} input[type=text]{width:100%;box-sizing:border-box;background:#0f1115;color:#dfe3ea;border:1px solid #2a3242;border-radius:6px;padding:6px 8px}
 #cy{flex:1} #stat{font-size:11px;color:#7c8aa0;margin:8px 0}
 .chip{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}
 #detail{position:absolute;right:12px;top:12px;width:340px;max-height:70vh;overflow:auto;background:#151922ee;border:1px solid #2a3242;border-radius:10px;padding:12px;display:none}
 #detail h3{margin:0 0 6px;font-size:14px;color:#9fc2ff} #detail .m{color:#8fa0b8;font-size:11px;margin:2px 0}
 #detail pre{white-space:pre-wrap;word-break:break-word;background:#0f1115;border:1px solid #232936;border-radius:8px;padding:8px;max-height:300px;overflow:auto;font-size:12px}
 .sw{display:inline-block;width:26px;height:12px;border-radius:2px;margin-right:4px;vertical-align:-2px}
</style></head><body>
<div id="bar">
 <h1>🧠 灵枢认知图</h1>
 <input id="q" type="text" placeholder="搜索标题…">
 <div id="stat"></div>
 <h2>层</h2><div id="layers"></div>
 <h2>边类型</h2><div id="etypes"></div>
 <h2>显示</h2>
 <label><input type="checkbox" id="iso"> 孤立节点</label>
 <label><input type="checkbox" id="dashed" checked> 推导边（虚线）</label>
 <label><input type="checkbox" id="hubn" checked> 枢纽/桶节点</label>
 <label><input type="checkbox" id="anch" checked> 只看已锚定（隐藏隔离区）</label>
</div>
<div id="cy"></div>
<div id="detail"></div>
<script src="cytoscape.min.js"></script>
<script>
let G = null, cy = null;
const layerColor = %%LAYER_COLOR%%;
const ruleColor = %%RULE_COLOR%%;
// 层中文描述（显示用；过滤/逻辑仍用英文 key，未知类型兜底显示原文）
const layerNameZh = {"knowledge":"知识","contextual":"情境","structural":"结构","self":"自我","anchor":"锚点","goals":"目标","rejected":"已拒绝","unresolved":"未解决","hub":"枢纽"};
// 边类型中文描述（显示用；过滤/逻辑仍用英文 key，未知类型兜底显示原文）
const edgeNameZh = {"explicit":"显式关系","part_of":"组成关系","counterpart":"对应关系","above":"上位关系","untyped":"未标注关系","hierarchical":"层级关系","causal":"因果关系","similar":"相似关系","sequential":"时序先后","derived_from":"派生自","tag_similar":"标签相似","same_bucket":"同桶","same_source":"同源引用","in_bucket":"同桶成员","same_day":"同日创建","in_session":"会话内","R1_same_bucket":"同桶推导(R1)","R1_same_bucket_pair":"同桶成对(R1)","R2_same_source":"同源推导(R2)","R3_tag_jaccard":"标签相似推导(R3)","R4_same_day":"同日推导(R4)","R6_session_chain":"会话先后链(R6)","R6_session_hub":"会话枢纽(R6)","body_crossref":"正文互引"};
const DEFAULT_EDGE_ON = new Set(['hierarchical','similar','part_of','causal','counterpart','above','untyped','body_crossref','sequential','derived_from']);
function chip(c){return '<span class="chip" style="background:'+c+'"></span>';}
function esc(s){return String(s).replace(/[&<>]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}

async function load(){
  G = await (await fetch('viewer_graph.json')).json();
  document.getElementById('layers').innerHTML = Object.keys(G.layer_count).map(l=>{
    const c = layerColor[l]||'#888';
    return '<label title="'+l+'"><input type="checkbox" class="lf" value="'+l+'" checked>'+chip(c)+(layerNameZh[l]||l)+' ('+G.layer_count[l]+')</label>';
  }).join('');
  document.getElementById('etypes').innerHTML = Object.keys(G.edge_type_count).map(t=>{
    const c = ruleColor[t]||'#666'; const on = DEFAULT_EDGE_ON.has(t); const zh = edgeNameZh[t]||t;
    return '<label title="'+t+'"><input type="checkbox" class="ef" value="'+t+'"'+(on?' checked':'')+'>' + '<span class="sw" style="background:'+c+'"></span>'+zh+' ('+G.edge_type_count[t]+')</label>';
  }).join('');
  build();
  bind();
}
function build(){
  const q = document.getElementById('q').value.trim().toLowerCase();
  const lay = new Set([...document.querySelectorAll('.lf:checked')].map(x=>x.value));
  const ety = new Set([...document.querySelectorAll('.ef:checked')].map(x=>x.value));
  const showIso = document.getElementById('iso').checked;
  const showDash = document.getElementById('dashed').checked;
  const showHub = document.getElementById('hubn').checked;
  const onlyAnch = document.getElementById('anch').checked;
  const els = [];
  const keep = new Set();
  for(const n of G.nodes){
    if(!lay.has(n.layer)) continue;
    if(n.hub && !showHub) continue;
    if(onlyAnch && n.ncls === '隔离区') continue;
    if(n.degree===0 && !showIso) continue;
    if(q && !(n.title||'').toLowerCase().includes(q) && !n.id.toLowerCase().includes(q)) continue;
    keep.add(n.id);
    els.push({data:{id:n.id, label:(n.title||n.id).slice(0,26), layer:n.layer, degree:n.degree, hub:n.hub||false, imp:n.importance}, position:{x:n.x,y:n.y}, classes:(n.hub?'hub ':'')+(n.degree===0?'iso':'')});
  }
  for(const e of G.edges){
    if(!ety.has(e.type)) continue;
    if(e.derived && !showDash) continue;
    if(!keep.has(e.s) || !keep.has(e.t)) continue;
    els.push({data:{id:e.s+'|'+e.t+'|'+e.type+'|'+(e.derived?1:0), source:e.s, target:e.t}, classes:(e.derived?'derived ':'')+('r-'+e.type)});
  }
  if(cy) cy.destroy();
  cy = cytoscape({container:document.getElementById('cy'), elements:els,
    style:[{selector:'node',style:{'background-color':'data(bg)','label':'data(label)','width':'mapData(degree,0,60,6,26)','height':'mapData(degree,0,60,6,26)','font-size':5,'color':'#c9d4e4','text-valign':'bottom','text-margin-y':2,'min-zoomed-font-size':6}},
      {selector:'node.hub',style:{'shape':'round-rectangle','width':30,'height':14,'font-size':5}},
      {selector:'node.iso',style:{'opacity':.55}},
      {selector:'edge',style:{'width':.6,'line-color':'data(c)','line-style':'data(ls)','curve-style':'haystack','haystack-radius':.4,'opacity':.5}},
      {selector:'edge.derived',style:{'opacity':.32}}],
    layout:{name:'preset', fit:true, padding:30}, wheelSensitivity:.15, minZoom:.05, maxZoom:3});
  cy.nodes().forEach(n=>{ n.style('background-color', layerColor[n.data('layer')]||'#888'); });
  // 容器初始化时可能尚未完成 flex 布局（0x0），必须显式 resize + fit，否则画布空白
  cy.resize();
  cy.fit(undefined, 30);
  window.addEventListener('resize', ()=>{ cy.resize(); cy.fit(undefined, 30); });
  cy.edges().forEach(e=>{ const d=e.data(); const t=d.id.split('|')[2];
    e.style('line-color', ruleColor[t]||'#666'); e.style('line-style', d.id.endsWith('|1')?'dashed':'solid'); });
  const shown = cy.nodes().length, se = cy.edges().length;
  document.getElementById('stat').innerHTML = '显示 '+shown+' 节点 / '+se+' 边<br>全库 '+G.meta.counts.nodes+' 节点 / '+(G.meta.counts.edges+G.meta.derived)+' 边'+(G.merged_aliases?('（已归并 '+G.merged_aliases+' 个重复/同义节点）'):'')+'<br>真源指纹 '+G.meta.source_sha;
  cy.on('tap','node', async evt=>{
    const id = evt.target.id(); const box = document.getElementById('detail');
    box.style.display='block'; box.innerHTML = '<h3>'+esc(id)+'</h3><div class="m">加载中…</div>';
    try{ const r = await fetch('/api/node?id='+encodeURIComponent(id)); const j = await r.json();
      box.innerHTML = '<h3>'+esc(j.title||id)+'</h3>'
        + '<div class="m">层 '+esc(j.layer)+' ｜ 桶 '+esc(j.bucket||'-')+' ｜ 重要度 '+esc(j.importance)+'</div>'
        + '<div class="m">tags: '+esc((j.tags||[]).join(', ')||'-')+'</div>'
        + (j.ncls?('<div class="m">命名判定: '+esc(j.ncls)+'</div>'):'')
        + (j.path?('<div class="m">path: '+esc(j.path)+'</div>'):'')
        + '<pre>'+esc(j.content||'(无正文)')+'</pre>';
    }catch(err){ box.innerHTML='<div class="m">加载失败</div>'; }
  });
  cy.on('tap', e=>{ if(e.target===cy) document.getElementById('detail').style.display='none'; });
}
function bind(){
  document.getElementById('q').addEventListener('input', ()=>build());
  document.querySelectorAll('.lf,.ef').forEach(x=>x.addEventListener('change', ()=>build()));
  ['iso','dashed','hubn','anch'].forEach(id=>document.getElementById(id).addEventListener('change', ()=>build()));
}
load();
</script></body></html>
"""

with io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "viewer_serve.tmpl"), encoding="utf-8") as _f:
    SERVE = _f.read()


def build_layout(nodes, hubs_by_id):
    # 桶网格 + 黄金角散布；枢纽放簇中心偏移
    buckets = collections.defaultdict(list)
    for n in nodes:
        buckets[n.get("bucket") or ("__layer__" + n["layer"])].append(n)
    keys = sorted(buckets.keys())
    cols = max(1, int(math.ceil(math.sqrt(len(keys)))))
    cell = 320.0
    pos = {}
    for bi, key in enumerate(keys):
        cx = (bi % cols) * cell + cell / 2
        cy = (bi // cols) * cell + cell / 2
        members = buckets[key]
        for k, n in enumerate(members):
            ang = k * 2.399963229728653
            r = 14.0 * math.sqrt(k)
            pos[n["id"]] = (round(cx + r * math.cos(ang), 1), round(cy + r * math.sin(ang), 1))
        hid = "bucket:" + key
        if hid in hubs_by_id:
            pos[hid] = (round(cx, 1), round(cy, 1))
    for hid in hubs_by_id:
        if hid not in pos:
            pos[hid] = (0.0, 0.0)
    return pos


def center_hubs(pos, hubs_by_id, all_edges, ndigits=1):
    """枢纽坐标移到成员质心（否则全部 src:/day:/session: 枢纽挤在原点，形成巨型扇形射线）。

    必须在把 pos 拷进 vnodes 之前调用 —— 拷贝之后改 pos 不会影响已写入的坐标（issue #45 第二处）。
    """
    hub_neigh = collections.defaultdict(list)
    for e in all_edges:
        for a, b in ((e["s"], e["t"]), (e["t"], e["s"])):
            if a in hubs_by_id and b not in hubs_by_id:
                hub_neigh[a].append(b)
    for hid, neigh in hub_neigh.items():
        pts = [pos.get(x) for x in neigh if x in pos]
        if pts:
            pos[hid] = (round(sum(p[0] for p in pts) / len(pts), ndigits),
                        round(sum(p[1] for p in pts) / len(pts), ndigits))
    return pos


def fetch_cytoscape(out_dir):
    p = os.path.join(out_dir, "cytoscape.min.js")
    if os.path.isfile(p) and os.path.getsize(p) > 100000:
        return p, "已有"
    for attempt in range(3):
        try:
            req = urllib.request.Request(CYTO_URL, headers={"User-Agent": "lingshu-viewer"})
            with urllib.request.urlopen(req, timeout=120) as r, open(p, "wb") as f:
                f.write(r.read())
            return p, "已下载"
        except Exception as e:
            last = str(e)
            time.sleep(3)
    raise SystemExit("cytoscape 下载失败（离线环境请手工放置 cytoscape.min.js 到 %s）：%s" % (out_dir, last))


def main(argv=None):
    ap = argparse.ArgumentParser(description="认知图可视化构建器")
    ap.add_argument("--graph", required=True)
    ap.add_argument("--derived", action="append", default=[], help="derived_edges.json（可多个：规则推导/会话链…）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mdcg-root", default=os.environ.get("MDCG_ROOT", ""))
    ap.add_argument("--naming", default="", help="naming_report.json（可选，提供命名锚定分类）")
    ap.add_argument("--synonyms", default="", help="synonym_groups.json（可选，做同义/重复归并）")
    args = ap.parse_args()

    with io.open(args.graph, encoding="utf-8") as f:
        g = json.load(f)
    d_hubs, d_edges = {}, []
    for dpath in args.derived:
        with io.open(dpath, encoding="utf-8") as f:
            dd = json.load(f)
        for h in dd.get("hubs", []):
            if h["id"] in d_hubs:
                d_hubs[h["id"]]["count"] += h.get("count", 0)
            else:
                d_hubs[h["id"]] = dict(h)
        d_edges.extend(dd.get("edges", []))
    hubs_list = list(d_hubs.values())
    d_edge_count = len(d_edges)

    # ---- 同义/重复归并：别名节点重映射到 canonical（仅视图层，不改真源）
    alias_to_canon = {}
    merged_alias_count = 0
    if args.synonyms and os.path.isfile(args.synonyms):
        with io.open(args.synonyms, encoding="utf-8") as f:
            syn = json.load(f)
        for m in syn.get("groups", []):
            cid = m.get("canonical_id")
            if not cid:
                continue
            for mem in m.get("members", []):
                if mem["id"] != cid:
                    alias_to_canon[mem["id"]] = cid
        merged_alias_count = len(alias_to_canon)
        print("同义归并：别名节点 %d 个 → canonical" % merged_alias_count)

    def canon(x):
        return alias_to_canon.get(x, x)

    all_edges, seen_edge = [], set()
    for e in g["edges"] + d_edges:
        s, t = canon(e["s"]), canon(e["t"])
        if s == t:
            continue
        k = (s, t, e["type"], bool(e.get("derived")))
        if k in seen_edge:
            continue
        seen_edge.add(k)
        all_edges.append({"s": s, "t": t, "type": e["type"], "derived": bool(e.get("derived"))})

    deg = collections.Counter()
    for e in all_edges:
        if not e["s"].startswith(("bucket:", "src:", "day:", "session:")):
            deg[e["s"]] += 1
        if not e["t"].startswith(("bucket:", "src:", "day:", "session:")):
            deg[e["t"]] += 1

    hub_ids = {h["id"] for h in hubs_list}
    hubs_by_id = {h["id"]: h for h in hubs_list}

    title2cls = {}
    if args.naming and os.path.isfile(args.naming):
        with io.open(args.naming, encoding="utf-8") as f:
            for row in json.load(f).get("rows", []):
                title2cls[row["title"]] = row["class"]

    vnodes = []
    for n in g["nodes"]:
        if n["id"] in alias_to_canon:
            continue
        vnodes.append({"id": n["id"], "layer": n["layer"], "title": n["title"],
                       "bucket": n.get("bucket", ""), "tags": n.get("tags", []),
                       "importance": n.get("importance", 0), "degree": deg.get(n["id"], 0),
                       "path": n.get("path", ""),
                       "ncls": title2cls.get((n.get("title") or "").strip(), ""),
                       "hub": n["id"] in hub_ids})
    for h in hubs_list:
        vnodes.append({"id": h["id"], "layer": "hub", "title": h["label"],
                       "bucket": "", "tags": [], "importance": 0,
                       "degree": h["count"], "hub": True})

    # 先算布局，再把枢纽移到成员质心（必须在拷进 vnodes 之前；否则拷贝到的是未居中的坐标）
    pos = build_layout(g["nodes"], hubs_by_id)
    center_hubs(pos, hubs_by_id, all_edges)

    for n in vnodes:
        x, y = pos.get(n["id"], (0.0, 0.0))
        n["x"], n["y"] = x, y
        n["bg"] = LAYER_COLOR.get(n["layer"], "#888888")
        n["ncls"] = title2cls.get((n.get("title") or "").strip(), "")

    vedges = []
    et_count = collections.Counter()
    for e in all_edges:
        vedges.append({"s": e["s"], "t": e["t"], "type": e["type"], "derived": e["derived"]})
        et_count[e["type"]] += 1

    layer_count = collections.Counter(n["layer"] for n in vnodes)
    doc = (HTML.replace("%%LAYER_COLOR%%", json.dumps(LAYER_COLOR))
               .replace("%%RULE_COLOR%%", json.dumps(RULE_COLOR)))
    # meta.derived：前端 stat 行读 G.meta.counts.edges + G.meta.derived（#413）——
    # 旧版 derived 只在顶层，G.meta.derived 是 undefined ⇒ 边数显示 NaN。
    meta = dict(g["meta"])
    meta["derived"] = d_edge_count
    vgraph = {"meta": meta, "nodes": vnodes, "edges": vedges,
              "layer_count": dict(layer_count), "edge_type_count": dict(et_count),
              "derived": d_edge_count, "merged_aliases": merged_alias_count}

    os.makedirs(args.out, exist_ok=True)
    with io.open(os.path.join(args.out, "index.html"), "w", encoding="utf-8", newline="\n") as f:
        f.write(doc)
    with io.open(os.path.join(args.out, "viewer_graph.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(vgraph, f, ensure_ascii=False)
    with io.open(os.path.join(args.out, "serve.py"), "w", encoding="utf-8", newline="\n") as f:
        f.write(SERVE)
    _, note = fetch_cytoscape(args.out)
    print("产物：%s" % args.out)
    print("  index.html / viewer_graph.json(%d 节点 %d 边) / serve.py / cytoscape.min.js(%s)"
          % (len(vnodes), len(vedges), note))


if __name__ == "__main__":
    raise SystemExit(main())
