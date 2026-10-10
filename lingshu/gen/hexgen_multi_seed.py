# -*- coding: utf-8 -*-
"""hexgen_multi_seed · 「同 prompt 多次渲染」语料 + 件内方差量测（R293 · 荣 2026-09-29 裁决①）
====================================================================================
背景与动机（全部来自台账实测，不是设想）：
  · 形状列自 R282 达最好后**五次**「在标定集上再拟合判别结构」的尝试全败（R278/R281/R283/R285/R286），
    共同机制被量到**数据层**：标定集每类只有 2~7 件 ⇒ 件内方差与类间差同量级。
  · 但从没有人**直接量过**「同一个 prompt 多次渲染」的件内散布——现有语料每个 prompt 只有
    **一张**图，件内方差只能靠「不同 prompt 的同形状件」间接猜。
  · 荣 2026-09-29 裁决①：「扩『同 prompt 多次渲染』语料（同一 prompt 渲多张，例如每 prompt ≥5 张、
    按现有 37 条 prompt 的子集起步）」⇒ 本模块先建该语料，再量三件事。

三件事（monotone，每件都有数）：
  ① **读出稳定性**（同 prompt S 张之间）：形状/颜色/花纹的**多重集**是否一致、数量是否一致；
  ② **真值兑现率**：同一 prompt 的多张里，读出与 prompt 真值一致的比例（此前只能用单张近似，
     现在可以给「语料级」的定义：**多次渲染中兑现的比例**）；
  ③ **件内方差 vs 类间差**：逐特征算「同 prompt 内散布（按形状类 pooled）」（分子）与
     「类间散布（类中心 sd）」（分母），给出比值 ⇒ 这是形状列能否再提升的判据。
     类间差 ≈ 0 而件内 > 0 的维不可分（比值 = +∞）：记进 `indistinguishable_dims`，
     该维比值输出 null、排序时排最前，且**不参与** `median_ratio`——不写成 0.0
     （0 = 「件内比类间还稳」= 最优，正好把最坏情形藏成最好）。

红线：白箱（零 LLM、纯 stdlib+numpy+PIL）；读图/量测**只读复用** `hexgen_c1_real` 的确定性实现；
      生成侧只用**本地 Qwen-Image-2.1 管线**（即被研究的黑箱本体，与既有语料同源同参数）。
命令行：
    python experiments/hexgen_multi_seed.py --gen            # 只生成
    python experiments/hexgen_multi_seed.py --measure         # 只量测（需已有语料）
    python experiments/hexgen_multi_seed.py                  # 两步都做
    --prompts 8 --seeds 5     # 子集大小与每 prompt 张数
"""
import argparse
import hashlib
import io
import json
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, ".")
sys.path.insert(0, "experiments")
from . import hexgen_c1_real as C          # 只读复用：解析器/读出链/度量（旁路，不改它）

QWEN = (r"<local-path>"
        r"\snapshots\master")
LIB = "data/vision/qwen_controlled/connection_library.json"
OUTDIR = "data/vision/qwen_multi"
OUT = "data/hexgen_multi_seed.json"
STEPS, SIZE = 20, 512
SEEDS = (11, 22, 33, 44, 55)         # 每 prompt 的渲染 seed（固定 ⇒ 可复跑）


def pick_prompts(n, lib_path=LIB):
    """确定性选子集：库记录里筛出**受控几何域**（`vs_s*` 且 prompt 能被 `C.parse_prompt`
    解析成五列真值）的记录，去重 prompt 文本后按 `name` 升序取前 n 条。
    选法只依赖库文件内容（无随机、无 LLM），故可复跑；`sid` 用库记录名，便于溯源。

    ⚠ 实现事故（R293 首跑）：初版只按 `name` 排序去重取前 n ⇒ 排在前面的 `ls_*`（**风景域**）
    记录被选中，29 张 meadow/waterfall/forest 渲染全部作废（已移出语料）。**故判据必须是
    「能通过受控模板解析」而不是「名字排序靠前」**——这正是本模块守门测试钉的那一条。"""
    with io.open(lib_path, encoding="utf-8") as f:
        lib = json.load(f)
    seen, out = set(), []
    for r in sorted(lib, key=lambda x: x["name"]):
        if not str(r.get("name", "")).startswith("vs_s"):
            continue                       # 只取受控几何域（C 线语料同域）
        p = (r.get("tags") or {}).get("prompt")
        if not p or p in seen or C.parse_prompt(p) is None:
            continue                       # 必须能被受控模板解析出五列真值
        seen.add(p)
        out.append({"sid": r["name"], "prompt": p})
        if len(out) >= n:
            break
    return out


def gen(prompts, seeds=SEEDS, outdir=OUTDIR):
    """黑箱多 seed 渲染。**已在则跳过**（可断点续跑）；管道按既有语料同款参数加载。"""
    import torch
    from diffusers import QwenImage21Pipeline
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    pipe = QwenImage21Pipeline.from_pretrained(QWEN, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    t_load = time.time() - t0
    rows, t_gen0, n_new = [], time.time(), 0
    for it in prompts:
        d = os.path.join(outdir, it["sid"])
        os.makedirs(d, exist_ok=True)
        for s in seeds:
            p = os.path.join(d, f"s{s}.png")
            if not os.path.isfile(p):
                g = torch.Generator("cpu").manual_seed(s)
                im = pipe(prompt=it["prompt"], height=SIZE, width=SIZE,
                          num_inference_steps=STEPS, generator=g).images[0]
                im.save(p)
                n_new += 1
                print(f"  生成 {it['sid']} seed={s} -> {p}", flush=True)
            else:
                print(f"  复用 {it['sid']} seed={s}", flush=True)
            rows.append({"sid": it["sid"], "prompt": it["prompt"], "seed": s, "path": p})
    return {"load_s": round(t_load, 1), "gen_s": round(time.time() - t_gen0, 1),
            "n_new": n_new, "n_img": len(rows), "steps": STEPS, "size": SIZE,
            "rows": rows}


def _stable_id(sid, seed) -> int:
    """确定性整数 id：`blake2b` 摘要前 8 字节 % 10**9。

    #198：原式 `hash((r["sid"], r["seed"])) % 10 ** 9` 用**内建 `hash()`**——
    `str` 的哈希受 `PYTHONHASHSEED` 随机化（Python 3 默认逐进程随机）⇒ 同一份语料
    在两个进程里给出**不同 id**，报告不可复现、跨进程对不上。改用
    `hashlib.blake2b`（摘要稳定、与进程无关；同仓 `semantic_anchor_graph.py:90`
    亦用哈希摘要造 id）。数值域与旧式一致（`% 10**9`），下游只当不透明标识用。
    """
    raw = ("%s|%s" % (sid, seed)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big") % 10 ** 9


def load_multi(rows):
    """把多 seed 语料读成 `hexgen_c1_real` 的 item 形状（真值由**同一**确定性解析器给出）。"""
    items = []
    for r in rows:
        truth = C.parse_prompt(r["prompt"])
        #   与 `hexgen_c1_real.load_items` 同一口径：先 `.convert("RGB")` 再取数组
        #   （`[..., :3]` 对 L/P 模式图会静默取到亮度值/调色板索引）。
        img = np.asarray(Image.open(r["path"]).convert("RGB")).astype(np.float64)
        items.append({"file": r["path"], "id": _stable_id(r["sid"], r["seed"]),
                      "shape": truth["shape"] if truth else None,
                      "color": truth["color"] if truth else None,
                      "img": img, "truth": truth, "prompt": r["prompt"],
                      "sid": r["sid"], "seed": r["seed"]})
    return items


def measure(items, d=4):
    """逐图读出（用**冻结的**既有读者：标定集取自原语料，口径与 C 线一致）。
    返回 [{sid, seed, n_read, shapes, colors, patterns, cells, feats}]。"""
    base = [it for it in C.load_items(C.CORPUS) if it["truth"]]
    cal, _ = C.split_ids(base)
    ids_c = {it["id"] for it in cal}
    m0 = {"items": base, "by_id": {it["id"]: C.measure_item(it, tol=C.TOL, d=d)[0]
                                   for it in base}, "d": d}
    fit0 = C.fit(m0, ids_c)                    # 冻结读者（R282 口径：中位原型 + Fisher 加权）
    out = []
    for it in items:
        objs = C.measure_item(it, tol=C.TOL, d=d)[0]
        preds = [C.predict(o, fit0) for o in objs]
        out.append({"sid": it["sid"], "seed": it["seed"], "n_read": len(objs),
                    "shapes": sorted(p["shape"] for p in preds),
                    "colors": sorted(p["color"] for p in preds),
                    "patterns": sorted(p["pattern"] for p in preds),
                    "cells": sorted(p["cell"] for p in preds),
                    "feats": [C._feat_vec(o["feat"]) for o in objs]})
    return out, fit0


def summarize(reads, items):
    """三件事的数字：①读出稳定性 ②真值兑现率 ③件内方差 vs 类间差。"""
    by_sid = {}
    for r in reads:
        by_sid.setdefault(r["sid"], []).append(r)
    truth_of = {it["sid"]: it["truth"] for it in items}

    #   ① 同 prompt 多张之间的读出稳定性（多重集一致性 + 数量一致性）
    stab = {"shape": [], "color": [], "pattern": [], "count": [], "cell": []}
    for sid, rs in by_sid.items():
        if len(rs) < 2:
            continue
        for k in ("shapes", "colors", "patterns", "cells"):
            stab["shape" if k == "shapes" else k[:-1]].append(
                sum(1 for a, b in zip(rs, rs[1:]) if a[k] == b[k]) / (len(rs) - 1))
        stab["count"].append(sum(1 for a, b in zip(rs, rs[1:])
                                 if a["n_read"] == b["n_read"]) / (len(rs) - 1))

    #   ② 真值兑现率：读出多重集与 prompt 真值的对齐（每图一次，再按 prompt 平均）
    hon = {"shape": [], "color": [], "pattern": [], "count": []}
    for sid, rs in by_sid.items():
        t = truth_of.get(sid)
        if not t:
            continue
        #   花纹列读者只出**二档** `plain`/`patterned`（`predict` 的 `pattern`，见
        #   hexgen_c1_real.py:958），而 prompt 真值是**三档**（`plain`/`striped`/`dotted`，
        #   `parse_prompt`，hexgen_c1_real.py:120）。两者必须先降到同一档再比：判据取
        #   C 线既有口径 `evaluate` 的 `tpat`（hexgen_c1_real.py:1074-1075，「花纹（二档
        #   plain vs patterned）」列）。直接拿二档读出比三档真值，会让所有 striped/dotted
        #   件恒判不兑现（issue #390）。代价：二档比较**削掉了 striped vs dotted 的判别力**
        #   —— 该列现在只能判「有无花纹」；三档读出的判别力由 C 线 `pattern3_rate` 承担。
        tpat = "plain" if t["pattern"] == "plain" else "patterned"
        for r in rs:
            hon["shape"].append(float(r["shapes"] == sorted([t["shape"]] * t["n"])))
            hon["color"].append(float(r["colors"] == sorted([t["color"]] * t["n"])))
            hon["pattern"].append(float(r["patterns"] == sorted([tpat] * t["n"])))
            hon["count"].append(float(r["n_read"] == t["n"]))

    #   ③ 件内方差 vs 类间差（逐特征）
    #   分子 = **同 prompt 内**散布：先算每个 prompt 自己的 per-image sd，再按形状类 pooled
    #          （sqrt(Σ(n_i−1)·s_i² / Σ(n_i−1))，n_i = 该 prompt 的图数）。「件内」= 同一
    #          prompt 的多次渲染之间，**不是**同形状类的不同 prompt 之间。
    #   分母 = 类间散布（各类中心的 sd）。
    cls_of = {it["sid"]: it["shape"] for it in items}
    vecs = {}                                   # 类 → [[每图的特征行], ...]（类中心用）
    per_prompt_sd = {}                          # 类 → [(n_i, sd_i), ...]（**每 prompt 一项**）
    for sid, rs in by_sid.items():
        k = cls_of.get(sid)
        if not k:
            continue
        vecs.setdefault(k, []).extend([r["feats"] for r in rs])
        with_f = [r["feats"] for r in rs if r["feats"]]
        #   「件内」只在同一 prompt 有 ≥2 张时才有定义：先按图求均值（消除「一张图里多个
        #   物体」的干扰），再对图求 sd；单张件对分子没有贡献。
        if len(with_f) >= 2:
            img_means = np.stack([np.stack(im).mean(axis=0) for im in with_f])
            per_prompt_sd.setdefault(k, []).append(
                (len(img_means), img_means.std(axis=0, ddof=1)))
    dims = list(C.FEATS)
    within, between, n_cls = [], [], 0
    means = []
    n_empty = 0
    for k, per_img in vecs.items():
        #   **空读件必须单独计数而不是参与统计**（R293 首跑即崩在这里：有渲染件一个物体都读不出，
        #   `np.stack([])` 直接抛错）。空读率本身是「真值兑现」的第一手证据。
        with_obj = [im for im in per_img if im]
        n_empty += len(per_img) - len(with_obj)
        if len(with_obj) < 2:
            continue
        means.append(np.stack([np.stack(im).mean(axis=0) for im in with_obj]).mean(axis=0))
        n_cls += 1
    #   件内：按类 pooled —— 每个 prompt 先出 per-image sd，再按自由度加权合并
    for k, entries in per_prompt_sd.items():
        den = sum(n - 1 for n, _ in entries)
        if den <= 0:
            continue
        within.append(np.sqrt(sum((n - 1) * (s ** 2) for n, s in entries) / den))
    if len(means) > 1:
        between = np.stack(means).std(axis=0, ddof=1)
    else:
        between = np.zeros(len(dims))
    within_m = np.median(np.stack(within), axis=0) if within else np.zeros(len(dims))
    #   类间差 ≈ 0 而件内散布 > 0 的维：比值在数学上无定义（= +∞）。它既不能落进 0.0
    #   （0 = 「件内比类间还稳」= 最优，恰好把最坏情形写成最好），也不该悄悄混进
    #   median_ratio；故单独点名 indistinguishable_dims、该维比值输出 null、排序时排最前。
    indistinguishable = (between <= 1e-9) & (within_m > 0)
    defined = ~indistinguishable
    ratio = np.zeros(len(dims))
    ratio[defined] = within_m[defined] / np.maximum(between[defined], 1e-9)
    order = np.argsort(-np.where(indistinguishable, np.inf, ratio))
    med = float(np.median(ratio[defined])) if bool(defined.any()) else float("nan")
    return {
        "n_prompt": len(by_sid), "n_img": len(reads),
        "n_img_empty": n_empty,
        "empty_rate": round(n_empty / max(1, len(reads)), 4),
        "stability": {k: round(float(np.mean(v)), 4) if v else None
                      for k, v in stab.items()},
        "honored": {k: round(float(np.mean(v)), 4) if v else None
                    for k, v in hon.items()},
        "within_vs_between": {
            "dims": dims,
            "within_prompt_sd": [round(float(x), 5) for x in within_m],
            "between_class_sd": [round(float(x), 5) for x in between],
            "ratio_within_over_between": [None if indistinguishable[i] else round(float(ratio[i]), 3)
                                          for i in range(len(dims))],
            "indistinguishable_dims": [dims[i] for i in np.where(indistinguishable)[0]],
            "median_ratio": round(med, 3) if med == med else None,
            "worst_dims": [dims[i] for i in order[:5]],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--gen", action="store_true")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    seeds = SEEDS[:a.seeds]
    prompts = pick_prompts(a.prompts)
    print(f"[多 seed 语料] prompt 子集 {len(prompts)} 条 × {len(seeds)} seed = "
          f"{len(prompts) * len(seeds)} 张（seed={list(seeds)}）")
    for it in prompts:
        print(f"  {it['sid']}: {it['prompt'][:78]}…")
    rep = {"prompts": prompts, "seeds": list(seeds), "params":
           {"steps": STEPS, "size": SIZE, "outdir": OUTDIR, "d": 4}}
    do_both = not (a.gen or a.measure)
    if a.gen or do_both:
        g = gen(prompts, seeds)
        rep["generation"] = {k: v for k, v in g.items() if k != "rows"}
        rep["rows"] = g["rows"]
        print(f"  生成：新增 {g['n_new']}/{g['n_img']} 张（加载 {g['load_s']}s、生成 {g['gen_s']}s）")
    if "rows" not in rep:
        #   只跑 --measure 时从**目录**回收已有渲染件（缺图则如实标记，不静默跳过）
        rep["rows"] = [{"sid": it["sid"], "prompt": it["prompt"], "seed": s,
                        "path": os.path.join(OUTDIR, it["sid"], f"s{s}.png")}
                       for it in prompts for s in seeds]
        miss = [r["path"] for r in rep["rows"] if not os.path.isfile(r["path"])]
        if miss:
            print(f"  ⚠ 缺 {len(miss)} 张渲染件，量测只覆盖存在的（首例 {miss[0]}）")
            rep["rows"] = [r for r in rep["rows"] if os.path.isfile(r["path"])]
    if a.measure or do_both:
        items = load_multi(rep["rows"])
        reads, _fit0 = measure(items)
        rep["summary"] = summarize(reads, items)
        s = rep["summary"]
        print(f"  量测：{s['n_img']} 张 / {s['n_prompt']} prompt")
        print(f"    ① 同 prompt 多张**读出稳定性**（相邻张一致率）：{s['stability']}")
        print(f"    ② **真值兑现率**（读出多重集＝prompt 真值）：{s['honored']}")
        w = s["within_vs_between"]
        print(f"    ③ 件内方差/类间差 中位比 **{w['median_ratio']}**"
              f"（最差 5 维 {w['worst_dims']}）")
        print(f"    ④ 空读件 {s['n_img_empty']}/{s['n_img']}（{s['empty_rate']:.1%}）")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with io.open(a.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1,
                  default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"报告: {a.out}")


if __name__ == "__main__":
    main()
