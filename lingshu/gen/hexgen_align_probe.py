# -*- coding: utf-8 -*-
"""hexgen_align_probe · 跨源对齐探针（R297 · 荣裁决②第四步）
====================================================================================
R296 把「对齐外部语义」拆成三类：**已对齐**（circle 8/8）、**命名口径差**（stripe→square）、
**对方语义 ≠ 几何**（triangle→diamond，而我们的三角解析正确）。本轮把第三类**量到底**：
黑箱标定的读者为什么把「解析正确的三角」读成 `diamond`？

三问（每问都要数）：
  ① **黑箱读者的形状原型长什么样**：`triangle` / `diamond` 两个原型的逐维值（`n_peaks`/`kdom`/
     `solidity`/`extent`/`peak_prom`/谐波）——用来判断「它的 triangle 是不是真三角」。
  ② **我们的三角到两个原型的距离与逐维贡献**：谁在把判决推向 `diamond`（Top-5 维 + 贡献占比）。
  ③ **交叉对照**：黑箱语料里**它自己**的真三角件到 `triangle` 原型是什么距离分布——若我们的件
     落在同一分布内，则说明「我们的三角与它的三角同类，是判决而非几何有问题」；若显著更远，
     则说明两源几何确实不同。

另登记**比较口径**：跨源比对时 `stripe ↔ square` 是**命名差**（本词表 `stripe`＝方块；真实词表
＝`square`）⇒ 给出「应用命名映射后的跨源命中率」，与未映射的读数并列（纪律 ④）。

红线：白箱（零 LLM、纯 stdlib+numpy）；只读复用 `hexgen_c1_real`（读者/度量）与
`hexgen_self_source`（自家 v2 渲染）；不改它们。
命令行：python experiments/hexgen_align_probe.py
"""
import argparse
import io
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "experiments")
from . import hexgen_c1_real as C
from . import hexgen_self_source as S

OUT = "data/hexgen_align_probe.json"

#   **跨源命名映射（比较口径）**：自家词表名 → 真实词表名（仅用于跨源比对，不改自家词表）
NAME_MAP_SELF2REAL = {"circle": "circle", "triangle": "triangle", "stripe": "square"}


def blackbox_reader():
    """黑箱标定的读者（C 线口径：标定集来自黑箱语料）。返回 (fit, measured)。"""
    base = [it for it in C.load_items(C.CORPUS) if it["truth"]]
    cal, _ = C.split_ids(base)
    m = {"items": base, "by_id": {it["id"]: C.measure_item(it, tol=C.TOL, d=4)[0]
                                  for it in base}, "d": 4}
    return C.fit(m, {it["id"] for it in cal}), m


def proto_report(fit):
    """① 黑箱读者的形状原型（挑出关键维，便于人读）。"""
    out = {}
    for cls in ("triangle", "diamond", "circle", "square"):
        p = fit["shape_protos"].get(cls)
        if p is None:
            continue
        f = {k: round(float(p[C.FEATS.index(k)]), 4) for k in
             ("n_peaks", "solidity", "extent", "peak_prom", "compactness")}
        f["kdom"] = C._proto_kdom(p)
        f["n_calib_obj"] = int(fit["n_shape"].get(cls, 0))
        out[cls] = f
    return out


def distance_contrib(q, p, w, sc):
    """逐维贡献（加权 L1），返回 (总距离, [(维, 贡献), ...] 降序)。"""
    per = np.abs(q - p) / sc * w
    tot = float(per.sum())
    pairs = sorted(zip(C.FEATS, per.tolist()), key=lambda x: -x[1])
    return tot, [(k, round(float(v), 4)) for k, v in pairs]


def probe_self_shape(shape="triangle", pattern="solid", fit=None, measured=None):
    """② 自家 v2 件（单物体在 r4）到各原型的距离与逐维贡献。"""
    img = S.render_self((1, shape, "red", pattern, ("r4",)))
    objs = C.measure_item({"img": img.astype(np.float64)}, tol=C.TOL, d=4)[0]
    assert objs, f"自家 {shape} 件抠不出物体"
    o = objs[0]
    q = C._feat_vec(o["feat"])
    w = np.asarray(fit["feat_weight"], float)
    sc = np.asarray(fit["feat_scale"], float)
    out = {"self_shape": shape, "own_metrics": {
        k: round(float(o["feat"][k]), 4) for k in
        ("n_peaks", "solidity", "extent", "peak_prom", "compactness")},
        "kdom": int(o["kdom"]), "hp_rel": round(float(o["hp_rel"]), 5), "dist": {},
        "verdict": C.predict(o, fit)["shape"]}
    for cls, p in fit["shape_protos"].items():
        tot, top = distance_contrib(q, p, w, sc)
        out["dist"][cls] = {"total": round(tot, 4), "top5": top[:5]}
    out["ranked"] = sorted(out["dist"], key=lambda k: out["dist"][k]["total"])[:3]
    return out


def blackbox_own_distribution(fit, measured):
    """③ 黑箱语料里**它自己**的真三角件到 `triangle` 原型的距离分布（作对照）。"""
    w = np.asarray(fit["feat_weight"], float)
    sc = np.asarray(fit["feat_scale"], float)
    ds = []
    for it in measured["items"]:
        if it["shape"] != "triangle":
            continue
        for o in measured["by_id"][it["id"]]:
            q = C._feat_vec(o["feat"])
            tot, _ = distance_contrib(q, fit["shape_protos"]["triangle"], w, sc)
            ds.append(tot)
    ds = sorted(ds)
    if not ds:
        return None
    return {"n": len(ds), "min": round(ds[0], 4), "median": round(float(np.median(ds)), 4),
            "max": round(ds[-1], 4), "all": [round(x, 4) for x in ds]}


def cross_source_hits(with_name_map=True):
    """跨源命中（黑箱标定读者读自家 v2 留出件）；`with_name_map` 时按比较口径映射自家标签。"""
    items = S.build_enum_items(repeats=1)
    cal, hold = S.split_self(items)
    m = {"items": items, "by_id": {it["id"]: C.measure_item(it, tol=C.TOL, d=4)[0]
                                   for it in items}, "d": 4}
    fit, _ = blackbox_reader()
    conf, hit, tot = {}, 0, 0
    for it in hold:
        truth = NAME_MAP_SELF2REAL[it["shape"]] if with_name_map else it["shape"]
        for o in m["by_id"][it["id"]]:
            pred = C.predict(o, fit)["shape"]
            conf[f"{it['shape']}→{pred}"] = conf.get(f"{it['shape']}→{pred}", 0) + 1
            hit += int(pred == truth)
            tot += 1
    return {"hit": hit, "n": tot, "rate": round(hit / max(1, tot), 4),
            "with_name_map": with_name_map, "confusion": dict(sorted(conf.items()))}


# ==================== R298 · 同题面对分（词表扩容后） ====================
def head_to_head(palette_mode=None, ids_subset=None, pattern_mode=None):
    """**同题面对分**：同一批题面（黑箱语料 37 条 prompt 的真值五列）——一侧是黑箱自己的渲染件，
    另一侧是自家 v3 渲染件（8 形状 × 10 色）；**用同一个读者**（黑箱标定）分别读，逐列并列。
    这是「源端对分」：读者固定、题面固定，**只有成像源不同**。"""
    fit, m_box = blackbox_reader()
    box_items = [it for it in m_box["items"] if it["truth"]]
    ids = {it["id"] for it in box_items}
    if ids_subset is not None:                      # 「判定在留出集」用
        ids = ids & set(ids_subset)
    pal = S.palette(palette_mode) if palette_mode else None
    keep_pm = S.PATTERN_MODE
    if pattern_mode:
        S.PATTERN_MODE = pattern_mode
    try:
        #   R304：把**冻结的方位标定**传给渲染器——`centroid` 口径的约束就是读者自己的格判定
        #   （约束与判决量同量纲）；不得用自家判定，否则等于拿判决量当选型量。
        self_items = [it for it in S.self_prompt_items(pal=pal, zone_calib=fit.get("zone"))
                      if it["id"] in ids]
    finally:
        S.PATTERN_MODE = keep_pm
    m_self = {"items": self_items,
              "by_id": {it["id"]: C.measure_item(it, tol=C.TOL, d=4)[0] for it in self_items},
              "d": 4}
    out = {}
    for tag, m in (("blackbox", m_box), ("self_v3", m_self)):
        r = C.evaluate(m, fit, ids, tag=tag)
        out[tag] = {"n_obj": r["n_obj"], "shape": r["rate"]["shape"],
                    "color": r["rate"]["color"], "pattern": r["rate"]["pattern"],
                    "pattern3": r["pattern3_rate"], "zone1": r["rate"]["zone"],
                    "zone3": r["zone3_rate"], "count": r["count_rate"],
                    "count_within1": r["count_rate_within1"],
                    "n_img": r["n_img"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    fit, measured = blackbox_reader()
    rep = {"name_map_self2real": NAME_MAP_SELF2REAL,
           "blackbox_prototypes": proto_report(fit)}
    print("① 黑箱读者的形状原型：")
    for cls, f in rep["blackbox_prototypes"].items():
        print(f"   {cls:9s} n_peaks={f['n_peaks']:.0f} kdom={f['kdom']} solidity={f['solidity']:.3f} "
              f"extent={f['extent']:.3f} peak_prom={f['peak_prom']:.3f}（标定物件 {f['n_calib_obj']}）")
    rep["self_probes"] = {}
    print("② 自家 v2 件到各原型的距离与 Top-5 贡献：")
    for shape in ("circle", "triangle", "stripe"):
        pr = probe_self_shape(shape, fit=fit)
        rep["self_probes"][shape] = pr
        d = pr["dist"]
        print(f"   {shape:9s} 自家指标 n_peaks={pr['own_metrics']['n_peaks']:.0f} kdom={pr['kdom']} "
              f"solidity={pr['own_metrics']['solidity']:.3f} extent={pr['own_metrics']['extent']:.3f}"
              f" ⇒ 读出 `{pr['verdict']}`；最近三原型 {pr['ranked']}")
        for cls in pr["ranked"]:
            print(f"        {cls:9s} 距离 {d[cls]['total']:8.3f} | Top5 {d[cls]['top5']}")
    rep["blackbox_triangle_own_distance"] = blackbox_own_distribution(fit, measured)
    bt = rep["blackbox_triangle_own_distance"]
    if bt:
        print(f"③ 黑箱**自己**的真三角件到 triangle 原型：n={bt['n']} 距离 中位 {bt['median']} "
              f"区间 [{bt['min']}, {bt['max']}]")
    rep["cross_source"] = {"raw": cross_source_hits(False), "mapped": cross_source_hits(True)}
    for tag in ("raw", "mapped"):
        r = rep["cross_source"][tag]
        print(f"④ 跨源命中（{tag}，命名映射={'无' if tag == 'raw' else '有'}）："
              f"{r['hit']}/{r['n']} = {r['rate']:.1%} | 混淆 {r['confusion']}")
    hold_ids = {it["id"] for it in C.split_ids([it for it in C.load_items(C.CORPUS)
                                                 if it["truth"]])[1]}
    rep["head_to_head"] = head_to_head()
    rep["palette_fallback"] = S.palette_fallback_names()
    rep["head_to_head_by_palette"] = {
        "canonical_hold": head_to_head("canonical", hold_ids),
        "aligned_hold": head_to_head("aligned", hold_ids),
        "canonical_all": head_to_head("canonical"),
        "aligned_all": head_to_head("aligned"),
    }
    rep["r300_variants"] = {
        f"{p}_{m}_{t}": head_to_head(p, hold_ids if t == "hold" else None, m)
        for p in ("canonical", "aligned") for m in ("darken", "symmetric")
        for t in ("all", "hold")
    }
    print("⑤ **同题面对分**（同一读者、同一批 37 条题面，只有成像源不同）：")
    cols = ("n_obj", "shape", "color", "pattern", "pattern3", "zone1", "zone3", "count", "count_within1")
    for tag in ("blackbox", "self_v3"):
        r = rep["head_to_head"][tag]
        print(f"   {tag:9s} 物件 {r['n_obj']:3d} | " +
              " ".join(f"{k}={r[k] if k == 'n_obj' else format(r[k], '.1%')}" for k in cols[1:]))
    print(f"⑥ d1 调色板对齐（回退色名：{rep['palette_fallback']}）：")
    for tag, hh in rep["head_to_head_by_palette"].items():
        b, sf = hh["blackbox"], hh["self_v3"]
        print(f"   {tag:15s} 颜色 黑箱 {b['color']:.1%} → 自家 {sf['color']:.1%}"
              f"（物件 {sf['n_obj']}）| 形状 {sf['shape']:.1%} 花纹 {sf['pattern']:.1%} "
              f"方位@1 {sf['zone1']:.1%} 数量精确 {sf['count']:.1%}")
    print("⑦ R300 四口径（调色板 × 花纹调制；all=37 题面 / hold 子集）：")
    for tag, hh in sorted(rep["r300_variants"].items()):
        sf, b = hh["self_v3"], hh["blackbox"]
        print(f"   {tag:22s} 颜色 {sf['color']:.1%}（黑箱 {b['color']:.1%}）| 花纹 {sf['pattern']:.1%} "
              f"| 形状 {sf['shape']:.1%} | 方位@1 {sf['zone1']:.1%} | 数量精确 {sf['count']:.1%}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with io.open(a.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1,
                  default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"报告: {a.out}")


if __name__ == "__main__":
    main()
