# -*- coding: utf-8 -*-
"""test_hex_text_contracts · 图文一致性判分口径守卫（被晶格越界遮蔽的缺陷）
============================================================================
晶格铺满画幅之后（见 tests/test_hex_grid_extent.py），spatial_detect 的位置基本都对了，
test_hex_text 的「噪声描述应大量 REJECT」随即 0/12。原因不在模型——此前位置常错，
REJECT 全靠「位置对不上」凑出来，下面这处一直没露面：

缺陷 · fuse_consistency 与自己的 docstring 不一致：docstring 写「位置命中物卡的
  形状/颜色匹配度（1 全符 / 0.5 半符 / 0 冲突）」，实现却是 0.5 位置底分 + 0.25×形状
  + 0.25×颜色 ⇒ 全符 1.0 / 半符 0.75 / 冲突 0.5。位置对、形状颜色全错的子句恒为
  DEFER，永远到不了 REJECT（< 0.3）。另：子句缺形状/颜色词（如「左上有东西」）且
  位置不中时，`None in str` 抛 TypeError。
  修法：每个属性相符 0.5 / 冲突 0 / 子句未陈述 0.25（沿用原分值，不改未陈述子句的判定）；
  「位置异但物体同」只在子句同时点名形状与颜色时成立。

断言组：
  A 组（判分口径）：同一检测（右下 circle|red）下，全符 / 半符 / 冲突 = 1.0 / 0.5 / 0.0；
      未陈述属性的子句判定不变；缺词子句不崩

运行（lingshu 仓根）：python -X utf8 tests/test_hex_text_contracts.py
退出码：0 = 全过；1 = 有断言失败
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.nn.hex_text import fuse_consistency, parse_description  # noqa: E402

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


def main():
    print("== A 组：fuse_consistency 判分口径（1 全符 / 0.5 半符 / 0 冲突） ==")
    group_a_scoring()
    print()
    print(f"===== SUMMARY {len(_PASS)}/{len(_PASS) + len(_FAIL)} 通过 =====")
    if _FAIL:
        print("FAILS:", _FAIL)
        return 1
    print("VERDICT=PASS（冲突子句可被 REJECT；缺词子句不崩）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
