# -*- coding: utf-8 -*-
"""test_hex_gen_constructive_independence · issue #295 守卫：构造性验证的判据必须是渲染像素
====================================================================================
缺陷（lingshu issue #295）：`hex_gen.verify_constructive(parts, log)` 拿**渲染器自己
产出的落格日志**当证据，把 `log[i]["zone_landed"]/shape/color/pattern/size` 与期望
`parts[i]` 比。日志是渲染器写的账，用它核对自己 ⇒ 恒等比对：只要 `render_relations`
把账写对（它总是写对），兑现率恒为 1.0，验证形同虚设（pixels 里部件画没画、画在哪，
判据一概不看）。

修法：`verify_constructive(parts, img)` 只读**渲染像素**——在期望 zone 的像素窗口里
数该部件色像素，存在 ≥1 即该部件 zone+color 条件兑现；与背景不可分的色计
`unverifiable` 并从分母剔除（诚实落账，不记为兑现）。

断言组（每条都钉「缺陷不再存在」，不是「代码能跑」）：
  A 组（像素是判据 · 直击缺陷）：期望 zone 与像素一致 ⇒ rate=1.0；**把图上该部件的
     色像素抹成背景色**（日志原封不动、仍写 zone_landed==zone）⇒ rate 必须为 0.0。
     旧口径读日志 ⇒ 恒 1.0 ⇒ 本组必红。
  B 组（夹具辨别力自证）：本地按**旧口径**（日志字段比对）对 A 组的同一夹具重算，
     必须得 1.0 —— 证明 A 组的 0.0 是新判据读出来的，不是夹具本身太软。
  C 组（期望与像素不一致）：同一张图，把期望 zone 改到别处 ⇒ matched=0、rate=0.0。
  D 组（不可分色诚实落账）：黑圆画在噪声底上（噪声中位≈21，黑=25，逐通道差 12）
     ⇒ 计 unverifiable、scored=0、rate=0.0（不把不可判读的部件算成兑现）。
  E 组（D 不是把所有黑都排除）：同样的黑圆画在灰底(208)上 ⇒ 色可分 ⇒ rate=1.0。
  F 组（端到端走像素）：generate_from_text 的 constructive 与手算像素口径一致，
     且图里无该色像素时期望值随之掉下来（旧口径下恒 1.0）。

运行（lingshu 仓根）：python -X utf8 tests/test_hex_gen_constructive_independence.py
退出码：0 = 全过；1 = 有断言失败
"""
from __future__ import annotations

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.nn.hex_gen import (generate_from_text, render_relations,  # noqa: E402
                                verify_constructive, zone_rc, _OPEN_RGB)
from lingshu.nn.hex_composite import _COLORS_RGB  # noqa: E402

TOL = 120  # 与 hex_gen._COLOR_TOL / hex_composite.extract_attributes 同口径

# 本文件自备的色表与背景估计（**不 import 修复新增的 _part_rgb/_bg_level**）：
# 缺陷形态下这些辅助函数不存在，若 import 它们，守卫会死在 import 期而不是
# 死在断言上——红要落在「判据读不读像素」这条断言上，才算钉住缺陷。
_RGB = {**_COLORS_RGB, **_OPEN_RGB}


def part_rgb(color):
    return _RGB.get(color, (220, 40, 40))


def bg_level(img):
    return np.median(img.reshape(-1, 3).astype(int), axis=0)

_PASS, _FAIL = [], []


def ok(cond, msg, extra=""):
    (_PASS if cond else _FAIL).append(msg)
    print(("  PASS " if cond else "  FAIL ") + msg + (("  " + str(extra)) if extra else ""))


def color_pixels(img, color):
    """独立色掩码（本文件自算，不调 hex_gen 判据）。"""
    ref = np.array(part_rgb(color))
    return np.abs(img.astype(int) - ref).sum(axis=2) < TOL


def zone_window_count(img, color, zone):
    size = img.shape[0]
    q = size // 3
    r, c = zone_rc(zone)
    roi = img[r * q:(r + 1) * q, c * q:(c + 1) * q]
    return int(color_pixels(roi, color).sum())


def old_style_verdict(parts, log):
    """旧口径（缺陷形态）本地复算：日志字段恒等比对。仅用于 B 组自证。"""
    matched = 0
    for i, p in enumerate(parts):
        e = next((x for x in log if x["part"] == i), None)
        matched += int(e is not None and e["zone_landed"] == p["zone"]
                       and e["shape"] == p["shape"] and e["color"] == p["color"]
                       and e.get("pattern") == p.get("pattern", "solid")
                       and e.get("size") == p.get("size", "medium"))
    return matched / max(1, len(parts))


def run_checks():
    # ---------- A 组：像素是判据（直击 #295）----------
    parts = [{"shape": "circle", "color": "red", "zone": "r0"}]
    img, log = render_relations(parts, size=96, seed=3)
    # 前置：像素确实画在 r0（用本文件的独立色掩码确认，不借 hex_gen 判据）
    ok(zone_window_count(img, "red", "r0") > 0, "A0 夹具：红圆像素确实在 r0",
       f"px={zone_window_count(img, 'red', 'r0')}")
    v = verify_constructive(parts, img)
    ok(v["rate"] == 1.0 and v["matched"] == 1, "A1 期望与像素一致 ⇒ rate=1.0", str(v))

    # 抹掉图上的部件色像素（日志不动）⇒ 判据必须掉到 0
    erased = img.copy()
    m = color_pixels(erased, "red")
    erased[m] = np.array(bg_level(img), dtype=np.uint8)
    ok(int(color_pixels(erased, "red").sum()) == 0, "A2 夹具：红像素已抹净")
    v2 = verify_constructive(parts, erased)
    ok(v2["rate"] == 0.0 and v2["matched"] == 0,
       "A3 图上无该色像素 ⇒ rate=0.0（旧日志口径下为 1.0）", str(v2))

    # ---------- B 组：夹具辨别力自证（旧口径在 A3 夹具上仍报 1.0）----------
    old = old_style_verdict(parts, log)
    ok(old == 1.0,
       "B1 旧口径（日志恒等比对）在同一夹具上仍报 1.0 ⇒ A3 的 0.0 是新判据读像素得来",
       f"old_style={old}")

    # ---------- C 组：期望 zone 与像素不一致 ----------
    v3 = verify_constructive([{"shape": "circle", "color": "red", "zone": "r8"}], img)
    ok(v3["rate"] == 0.0 and v3["matched"] == 0,
       "C1 期望 zone=r8 而像素在 r0 ⇒ matched=0/rate=0.0", str(v3))

    # ---------- D 组：不可分色诚实落账（黑画在噪声底）----------
    bparts = [{"shape": "circle", "color": "black", "zone": "r4"}]
    bimg, _ = render_relations(bparts, size=96, seed=3)      # 默认噪声底
    bg = bg_level(bimg)
    dist = float(np.abs(np.array(part_rgb("black")) - bg).sum())
    ok(dist < TOL, "D0 夹具：黑与噪声底逐通道差 < 阈值（确实不可分）",
       f"dist={dist} bg={bg.tolist()}")
    v4 = verify_constructive(bparts, bimg)
    ok(v4["unverifiable"] == 1 and v4["scored"] == 0 and v4["rate"] == 0.0
       and v4["matched"] == 0,
       "D1 不可分色 ⇒ unverifiable=1、scored=0、rate=0.0（不记为兑现）", str(v4))

    # ---------- E 组：D 不是把所有黑都排除 ----------
    gimg, _ = render_relations(bparts, size=96, seed=3, background="gray")
    gbg = bg_level(gimg)
    gdist = float(np.abs(np.array(part_rgb("black")) - gbg).sum())
    v5 = verify_constructive(bparts, gimg)
    ok(gdist >= TOL and v5["rate"] == 1.0 and v5["unverifiable"] == 0,
       "E1 黑画在灰底上 ⇒ 色可分 ⇒ rate=1.0（不是一刀切排除黑）",
       f"dist={gdist} {v5}")

    # ---------- F 组：端到端 constructive 走像素 ----------
    g = generate_from_text("左上有红色圆形,右下有绿色三角", size=96, seed=3)
    manual = sum(1 for p in g["parts"]
                 if zone_window_count(g["image"], p["color"], p["zone"]) > 0)
    ok(g["constructive"]["rate"] == manual / len(g["parts"]) and manual == 2,
       "F1 generate_from_text 的 constructive 与手算像素口径一致", str(g["constructive"]))
    # 抹掉图上红/绿像素 ⇒ 端到端读数必须跟着掉（旧口径下不变）
    erased2 = g["image"].copy()
    for col in ("red", "green"):
        erased2[color_pixels(erased2, col)] = np.array(bg_level(g["image"]), dtype=np.uint8)
    v6 = verify_constructive(g["parts"], erased2)
    ok(v6["rate"] == 0.0, "F2 抹净部件色像素后端到端读数掉到 0.0", str(v6))

    return _PASS, _FAIL


def test_hex_gen_constructive_independence():
    _, fail = run_checks()
    assert not fail, f"{len(fail)} checks failed: {fail}"


if __name__ == "__main__":
    p, f = run_checks()
    print(f"\n===== #295 守卫: {len(p)}/{len(p) + len(f)} 通过 =====")
    sys.exit(1 if f else 0)
