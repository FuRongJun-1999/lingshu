# -*- coding: utf-8 -*-
"""test_hex_composite · M4.3/M4.3j 复合体会意单元测试
（v1 会意匹配 + v2 属性维度扩展 + 证据密度门——全部零网络纯函数；
脚本/pytest 双模式：python tests/test_hex_composite.py 或 -m pytest）"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from lingshu.nn.hex_composite import (
    COMPOSITES, COMPOSITES_V2, COMP_NAMES, COMP_NAMES_V2, _ZONES,
    make_composite_dataset, make_composite_dataset_v2,
    compose_match, compose_match_v2,
    evidence_density, evidence_density_v2,
    extract_attributes, calibrate_attributes, judge_density)

_results = [0, 0]


def check(name, cond, detail=""):
    if cond:
        _results[0] += 1
        print(f"  PASS {name}")
    else:
        _results[1] += 1
        print(f"  FAIL {name} {detail}")


def run_checks():
    # ---------- v1 兼容性（旧 API 不回归） ----------
    imgs, metas = make_composite_dataset(12, seed=3)
    check("v1 数据集生成", imgs.shape == (12, 96, 96, 3)
          and all(m["composite"] in COMP_NAMES for m in metas))
    m = compose_match([{"obj": "stripe|red", "pos": "r1", "conf": 0.9},
                       {"obj": "circle|red", "pos": "r6", "conf": 0.8},
                       {"obj": "circle|red", "pos": "r8", "conf": 0.7}])
    check("v1 会意匹配 car 完整", m[0]["composite"] == "car"
          and m[0]["score"] >= 0.999, str(m[0]))
    check("v1 证据密度", abs(evidence_density(
        [{"conf": 0.9}, {"conf": 0.8}]) - 1.7) < 1e-9)

    # ---------- v2 场景与配方 ----------
    check("v2 配方数 8", len(COMPOSITES_V2) == 8, str(COMP_NAMES_V2))
    check("v2 配方部件可解析", all(
        all(set(p) >= {"shape", "zone"} and p["zone"] in _ZONES
            for p in spec) for spec in COMPOSITES_V2.values()))
    sig = {n: tuple(sorted((p["shape"], p["zone"]) for p in s))
           for n, s in COMPOSITES_V2.items()}
    check("v2 配方方位签名互异", len(set(sig.values())) == 8, str(sig))
    v2i, v2m = make_composite_dataset_v2(32, seed=7)
    check("v2 数据集生成", v2i.shape == (32, 96, 96, 3)
          and sorted({m["composite"] for m in v2m}) == sorted(COMP_NAMES_V2))
    check("v2 属性落盘", all(
        p["pattern"] in ("solid", "striped", "dotted")
        and p["size"] in ("small", "large", "medium")
        for m in v2m for p in m["parts"]))

    # ---------- 属性提取（确定性 + 标定 + 精度） ----------
    calib = calibrate_attributes(v2i, v2m)
    check("属性标定三阈值齐", all(k in calib for k in
          ("area_cut", "tx_solid", "vert_striped")), str(calib))
    ok_p = n_p = ok_s = n_s = 0
    for img, meta in zip(v2i, v2m):
        for p in meta["parts"]:
            r = extract_attributes(img, p["pos"], meta["body_color"], calib,
                                   shape_hint=p["shape"])
            if p["pattern"] in ("solid", "striped", "dotted") \
                    and r["pattern"] is not None:
                n_p += 1
                ok_p += int(r["pattern"] == p["pattern"])
            if p["size"] in ("small", "large"):
                n_s += 1
                ok_s += int(r["size"] == p["size"])
    check("花纹提取精度 ≥95%", ok_p / max(1, n_p) >= 0.95, f"{ok_p}/{n_p}")
    check("尺寸提取精度 ≥95%", ok_s / max(1, n_s) >= 0.95, f"{ok_s}/{n_s}")
    check("尺寸只在圆上判定", extract_attributes(
        v2i[0], "r4", v2m[0]["body_color"], calib,
        shape_hint="triangle")["size"] is None)
    check("提取确定性（两次调用一致）", extract_attributes(
        v2i[0], "r4", v2m[0]["body_color"], calib, "circle") == extract_attributes(
        v2i[0], "r4", v2m[0]["body_color"], calib, "circle"))

    # ---------- v2 会意匹配 + 属性消融 ----------
    det_snow = [{"obj": "circle|r", "pos": "r1", "conf": 0.9, "size": "small"},
                {"obj": "circle|r", "pos": "r7", "conf": 0.8, "size": "large"}]
    m2 = compose_match_v2(det_snow)
    check("v2 雪人（尺寸属性命中）", m2[0]["composite"] == "snowman"
          and m2[0]["score"] >= 0.999, str(m2[0]))
    det_badsize = [{**det_snow[0], "size": "large"}, {**det_snow[1], "size": "small"}]
    m3 = compose_match_v2(det_badsize)
    check("v2 尺寸不符不匹配 snowman", m3[0]["composite"] != "snowman"
          or m3[0]["score"] < 0.999, str(m3[0]))
    m4 = compose_match_v2(det_badsize, use_attrs=False)
    check("属性消融恢复形状级匹配", m4[0]["composite"] == "snowman"
          and m4[0]["score"] >= 0.999, str(m4[0]))
    det_car = [{"obj": "stripe|r", "pos": "r1", "conf": 0.9, "pattern": "striped"},
               {"obj": "circle|r", "pos": "r6", "conf": 0.8, "pattern": "solid"},
               {"obj": "circle|r", "pos": "r8", "conf": 0.7, "pattern": "solid"}]
    m5 = compose_match_v2(det_car)
    check("v2 车（花纹必要属性）", m5[0]["composite"] == "car"
          and m5[0]["score"] >= 0.999, str(m5[0]))
    det_car_badpat = [{**det_car[0], "pattern": "dotted"}, *det_car[1:]]
    m6 = compose_match_v2(det_car_badpat)
    check("v2 车身花纹不符不匹配", m6[0]["composite"] != "car"
          or m6[0]["score"] < 0.999, str(m6[0]))

    # ---------- 证据密度 v2 + 密度门（四态） ----------
    d2 = evidence_density_v2(det_car)
    check("密度 v2 属性计入", abs(d2 - (0.9 + 0.8 + 0.7) * 2) < 1e-9, str(d2))
    check("密度 v2 无属性回退 v1", abs(evidence_density_v2(
        [{"conf": 0.5, "pattern": None, "size": None}]) - 0.5) < 1e-9)
    j1 = judge_density({"composite": "car", "score": 1.0}, 5.0, 3.0)
    check("密度门 ACCEPT", j1["state"] == "ACCEPT" and j1["density"] == 5.0)
    j2 = judge_density({"composite": "car", "score": 1.0}, 1.0, 3.0)
    check("密度门 DEFER(依据不足)", j2["state"] == "DEFER"
          and j2["reason"] == "证据密度不足")
    j3 = judge_density({"composite": "car", "score": 0.667}, 9.0, 3.0)
    check("密度门 DEFER(配方不完整优先)", j3["state"] == "DEFER"
          and j3["reason"] == "配方不完整")
    j4 = judge_density(None, 9.0, 3.0)
    check("无匹配 DEFER", j4["state"] == "DEFER")
    return _results


def test_composite_v1_v2():
    p, f = run_checks()
    assert f == 0, f"{f} checks failed"


if __name__ == "__main__":
    run_checks()
    total = _results[0] + _results[1]
    print(f"\n===== HEX-COMPOSITE v2 回归: {_results[0]}/{total} 通过 =====")
    sys.exit(1 if _results[1] else 0)
