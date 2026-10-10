# -*- coding: utf-8 -*-
"""test_hexgen_multi_seed_honored_pattern · issue #390 守卫：② 号「花纹兑现率」必须先降到
同一档再比
============================================================================
缺陷（lingshu issue #390）：`hexgen_multi_seed.summarize()` 的 `honored["pattern"]` 拿
读者 `predict` 的**二档**输出（`plain`/`patterned`，hexgen_c1_real.py:958）去比
`parse_prompt` 的**三档**真值（`plain`/`striped`/`dotted`，hexgen_c1_real.py:120）：

    hon["pattern"].append(float(r["patterns"] == sorted([t["pattern"]] * t["n"])))

⇒ 所有 `striped`/`dotted` 件恒判不兑现，花纹兑现率被系统性算成 0。

修法：判据取 C 线既有口径 `evaluate` 的 `tpat`（hexgen_c1_real.py:1074-1075，「花纹
（二档 plain vs patterned）」列）——真值也先降到二档再比。**代价**：二档比较削掉了
`striped` vs `dotted` 的判别力（该列只能判「有无花纹」）；三档读出的判别力由 C 线
`pattern3_rate` 承担。

断言组（都钉「缺陷不再存在」，不是「代码能跑」）：
  1) 真值 `striped`（三档）+ 读出 `patterned`（二档）⇒ honored["pattern"] == 1.0
     （缺陷形态下恒 0.0）
  2) 真值 `dotted`（三档）+ 读出 `patterned`（二档）⇒ == 1.0（同上）
  3) 真值 `plain` + 读出 `plain` ⇒ == 1.0
  4) 真值 `striped` + 读出 `plain` ⇒ == 0.0（**反向**：不是恒 1 的假修复）
  5) 真值 `plain` + 读出 `patterned` ⇒ == 0.0（同上）
  6) 多张件里掺一张不兑现 ⇒ 兑现率 = 兑现张数 / 总张数（不是全有全无）

运行：python -X utf8 tests/test_hexgen_multi_seed_honored_pattern.py
      python -X utf8 -m pytest tests/test_hexgen_multi_seed_honored_pattern.py -q --no-header
退出码：0 = 全过；1 = 有断言失败
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.gen import hexgen_multi_seed as M       # noqa: E402


def mk_reads(sid, patterns):
    """构造 summarize() 期望形状的读数：每张图 n=len(patterns)、四列多重集。"""
    return [{"sid": sid, "seed": i, "n_read": len(pat),
             "shapes": ["circle"] * len(pat), "colors": ["red"] * len(pat),
             "patterns": sorted(pat), "cells": ["r4"] * len(pat), "feats": []}
            for i, pat in enumerate(patterns)]


def mk_item(sid, truth_pattern, n=2):
    """item 形状：真值用**三档**词表（与 parse_prompt 同 schema）。"""
    return {"sid": sid, "shape": "circle",
            "truth": {"n": n, "color": "red", "pattern": truth_pattern,
                      "shape": "circle", "cells": ["r4"], "zone_text": None}}


def honored_for(truth_pattern, read_pattern, n=2):
    sid = "p1"
    reads = mk_reads(sid, [sorted([read_pattern] * n)])
    rep = M.summarize(reads, [mk_item(sid, truth_pattern, n)])
    return rep["honored"]["pattern"]


def test_striped_truth_vs_patterned_read_is_honored():
    """三档真值 striped + 二档读出 patterned ⇒ 兑现（缺陷：0.0）。"""
    got = honored_for("striped", "patterned")
    assert got == 1.0, f"honored['pattern']={got}（期望 1.0；缺陷形态下恒 0.0）"


def test_dotted_truth_vs_patterned_read_is_honored():
    """三档真值 dotted + 二档读出 patterned ⇒ 兑现（缺陷：0.0）。"""
    got = honored_for("dotted", "patterned")
    assert got == 1.0, f"honored['pattern']={got}（期望 1.0；缺陷形态下恒 0.0）"


def test_plain_truth_vs_plain_read_is_honored():
    got = honored_for("plain", "plain")
    assert got == 1.0, f"honored['pattern']={got}"


def test_patterned_read_on_striped_truth_is_not_auto_honored():
    """反向守卫：真值 striped 但读出 plain ⇒ 不兑现（挡住「恒 1」的假修复）。"""
    got = honored_for("striped", "plain")
    assert got == 0.0, f"honored['pattern']={got}（期望 0.0，否则该列被削成恒真）"


def test_patterned_read_on_plain_truth_is_not_honored():
    got = honored_for("plain", "patterned")
    assert got == 0.0, f"honored['pattern']={got}"


def test_rate_is_fraction_over_images_not_all_or_nothing():
    """多张件：3 张里 1 张读 plain、2 张读 patterned（真值 striped）⇒ 2/3。"""
    sid = "p1"
    reads = mk_reads(sid, [["plain", "plain"], ["patterned", "patterned"],
                           ["patterned", "patterned"]])
    rep = M.summarize(reads, [mk_item(sid, "striped", 2)])
    got = rep["honored"]["pattern"]
    #   summarize 对报告值做 4 位四舍五入 ⇒ 2/3 → 0.6667（不是全有全无的 0/1）
    assert got == round(2 / 3, 4), f"honored['pattern']={got}（期望 {round(2 / 3, 4)}）"
    assert 0.0 < got < 1.0, f"honored['pattern']={got}（必须是分数而不是 0/1）"


def _main():
    fails = []
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS {name}")
            except AssertionError as e:                       # noqa: BLE001
                fails.append((name, str(e)))
                print(f"  FAIL {name}: {e}")
    print()
    if fails:
        print(f"===== FAIL {len(fails)} 项 =====")
        return 1
    print("===== PASS（issue #390 守卫：花纹兑现率二档对二档）=====")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
