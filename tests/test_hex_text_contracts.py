# -*- coding: utf-8 -*-
"""test_hex_text_contracts · 图文一致性判分与噪声描述的口径守卫（被晶格越界遮蔽的两处）
============================================================================
晶格铺满画幅之后（见 tests/test_hex_grid_extent.py），spatial_detect 的位置基本都对了，
test_hex_text 的「噪声描述应大量 REJECT」随即 0/12。原因不在模型，在下面两处——
此前位置常错，REJECT 全靠「位置对不上」凑出来，它们一直没露面：

缺陷 1 · fuse_consistency 与自己的 docstring 不一致：docstring 写「位置命中物卡的
  形状/颜色匹配度（1 全符 / 0.5 半符 / 0 冲突）」，实现却是 0.5 位置底分 + 0.25×形状
  + 0.25×颜色 ⇒ 全符 1.0 / 半符 0.75 / 冲突 0.5。位置对、形状颜色全错的子句恒为
  DEFER，永远到不了 REJECT（< 0.3）。另：子句缺形状/颜色词（如「左上有东西」）且
  位置不中时，`None in str` 抛 TypeError。
  修法：每个属性相符 0.5 / 冲突 0 / 子句未陈述 0.25（沿用原分值，不改未陈述子句的判定）；
  「位置异但物体同」只在子句同时点名形状与颜色时成立。
缺陷 2 · corrupt_description 级联改写：逐词条在「已改过的串」上查找替换，换进去的
  「三角形」会被「三角」词条再改一次，rate=1.0 也能改回原义（「红」→「绿」→「红」同理）。
  test_hex_text 的 12 条「重度噪声」里只有 3 条两个属性都真被改了。
  修法：每个原文词只抽签一次、最后一次性替换；被长词覆盖的短词不另抽签。

断言组：
  A 组（判分口径）：同一检测（右下 circle|red）下，全符 / 半符 / 冲突 = 1.0 / 0.5 / 0.0；
      未陈述属性的子句判定不变；缺词子句不崩
  B 组（噪声描述）：rate=1.0 时 9 种物体 × 20 个种子，形状与颜色都与原文不同（180/180），
      且仍解析为同一位置的单个子句；rate=0 恒等；rate=0.5 时各属性改写频率 ≈ 0.5
      （「三角形」按词条「三角」替换会留下「圆形形」这类词面，解析语义正确，属既有词表口径，本守卫不涉）

运行（lingshu 仓根）：python -X utf8 tests/test_hex_text_contracts.py
退出码：0 = 全过；1 = 有断言失败
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

from lingshu.nn.hex_text import (COLOR_CN, COLORS, SHAPE_CN, SHAPES,  # noqa: E402
                                 corrupt_description, fuse_consistency,
                                 parse_description)

_PASS = []
_FAIL = []

DET = [{"pos": "r8", "obj": "circle|red", "conf": 0.5, "share": 0.5}]   # 右下 红圆


def ok(cond, msg, extra=""):
    (_PASS if cond else _FAIL).append(msg)
    print(("  PASS " if cond else "  FAIL ") + msg
          + (("  ← " + str(extra)) if (extra and not cond) else ""))


def _fuse(text):
    try:
        return fuse_consistency(parse_description(text), DET)
    except Exception as e:                                # noqa: BLE001
        return {"verdict": f"EXC {type(e).__name__}", "score": None}


def group_a_scoring():
    cases = [
        ("A1", "右下有红色圆形", 1.0, "ACCEPT", "全符"),
        ("A2", "右下有红色三角形", 0.5, "DEFER", "半符"),
        ("A3", "右下有绿色三角形", 0.0, "REJECT", "冲突（位置对、形状颜色全错）"),
        ("A4", "右下有东西", 0.5, "DEFER", "未陈述形状颜色：判定不变"),
        ("A5", "右下有红色", 0.75, "DEFER", "只陈述颜色且相符：判定不变"),
        ("A6", "右下有绿色", 0.25, "REJECT", "只陈述颜色且冲突"),
        ("A7", "左上有红色圆形", 0.4, "DEFER", "位置异但物体同：判定不变"),
        ("A8", "左上有东西", 0.0, "REJECT", "缺词子句且位置不中：不崩"),
        ("A9", "左上有红色", 0.0, "REJECT", "只陈述颜色且位置不中：不崩"),
    ]
    for tag, text, score, verdict, why in cases:
        r = _fuse(text)
        ok(r["score"] == score and r["verdict"] == verdict,
           f"{tag} 「{text}」→ {score} {verdict}（{why}）", r)
    r = fuse_consistency([], [])
    ok(r["verdict"] == "BLINDSPOT", "A10 双路无证据 → BLINDSPOT（不变）", r)


def _descs():
    return [(s, c, f"左上有{COLOR_CN[c]}{SHAPE_CN[s]}") for s in SHAPES for c in COLORS]


def group_b_corrupt():
    both, parsed, total, worst = 0, 0, 0, []
    for s, c, desc in _descs():
        for seed in range(20):
            out = corrupt_description(desc, np.random.default_rng(seed), 1.0)
            cl = parse_description(out)
            total += 1
            if len(cl) == 1 and cl[0]["pos"] == "r0" and cl[0]["shape"] and cl[0]["color"]:
                parsed += 1
            if cl and cl[0]["shape"] != s and cl[0]["color"] != c:
                both += 1
            elif len(worst) < 3:
                worst.append(f"{desc}→{out}")
    ok(both == total, f"B1 rate=1.0：形状与颜色都被改掉（{total}/{total}）",
       f"{both}/{total}，例：{worst}")
    ok(parsed == total, f"B2 rate=1.0：改后仍解析为同一位置的单个完整子句（{total}/{total}）",
       f"{parsed}/{total}")

    same = all(corrupt_description(d, np.random.default_rng(k), 0.0) == d
               for k, (_, _, d) in enumerate(_descs()))
    ok(same, "B3 rate=0：恒等")

    rng = np.random.default_rng(0)
    n, sh, co = 0, 0, 0
    for _ in range(40):
        for s, c, desc in _descs():
            cl = parse_description(corrupt_description(desc, rng, 0.5))[0]
            n += 1
            sh += cl["shape"] != s
            co += cl["color"] != c
    ok(0.45 <= sh / n <= 0.55 and 0.45 <= co / n <= 0.55,
       f"B4 rate=0.5：每属性改写频率 ≈ 0.5（{n} 次）",
       f"形状 {sh / n:.3f} / 颜色 {co / n:.3f}")


def main():
    print("== A 组：fuse_consistency 判分口径（1 全符 / 0.5 半符 / 0 冲突） ==")
    group_a_scoring()
    print("== B 组：corrupt_description 不级联 ==")
    group_b_corrupt()
    print()
    print(f"===== SUMMARY {len(_PASS)}/{len(_PASS) + len(_FAIL)} 通过 =====")
    if _FAIL:
        print("FAILS:", _FAIL)
        return 1
    print("VERDICT=PASS（冲突子句可被 REJECT；噪声描述真改原义且不级联）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
