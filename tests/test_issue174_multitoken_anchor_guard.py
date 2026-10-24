# -*- coding: utf-8 -*-
"""守卫 · parse_anchor 必须读回自家 anchor_text/describe 的多词类别（issue #174）

缺陷（锚定 50ee03b2，`lingshu/world/vprim.py`）：
    写侧 `anchor_text()`/`describe()` 直接拼 `f"{category}@(...)"`，类别是**裸文本**
    （可含空格）；读侧 `parse_anchor` 旧实现「紧邻 `@(` 的非空白串」⇒ 多词类别被截断：
      `'cell phone@(...)'  -> 'phone'`、`'traffic light@(...)' -> 'light'`（实测 5/5 全错）。
    且扫描侧 `ANCHOR_SCAN_PATTERN` 用 `[\\w\\-]+`、core 传的是**匹配片段** `m.group(0)`
    ⇒ 即便读回侧修好，扫描入口仍会先截断（与 #718 同形态：扫描先于解析收窄格式）。

修法：
  · `parse_anchor` 类别起点改按**强分隔符**定位（`；`/`;`/换行/`|`/`] `/`)`），
    沿用仓内 #399 `parse_seed_entity` 的同一思路（按标记定位，不按空格裸切）；
  · `ANCHOR_SCAN_PATTERN` 类别段放宽到 `[^@()；;|\\n\\r]+`，并把尾部 `conf=` 段纳入。

五条：
  A 多词类别：`parse_anchor(v.describe())` 与 `parse_anchor(v.anchor_text())` 读回原类别
  B 扫描贯通：`ANCHOR_SCAN_PATTERN` 的 `m.group(0)` 再喂 `parse_anchor` 也读回原类别
    （core 就是这么用的 —— 只测 `parse_anchor(整串)` 会漏掉这条）
  C 场景文本：`vprims_to_scene_text` 全串逐条读回 category/bbox/conf 全对
  D 既有口径不回归：单词类别 / 非 `\\w` 类别（`obj.2`/`a/b`）/ 负坐标 / conf 照旧正确
  E 已知限制如实钉住：自由文本里类别前有别的词时仍会多吞（格式固有限制，非本件范围）
    —— 这条**故意断言「会多吞」**，防止有人把它当已修而误删 docstring 的诚实边界。

运行（仓根）：
  PYTHONUTF8=1 python -X utf8 -m pytest tests/test_issue174_multitoken_anchor_guard.py -q --no-header
"""
from __future__ import annotations

import os
import re
import sys

_THIS = os.path.abspath(__file__)
REPO = os.environ.get("LINGSHU_ROOT") or os.path.dirname(os.path.dirname(_THIS))
sys.path.insert(0, REPO)

from lingshu.world.vprim import (  # noqa: E402
    ANCHOR_SCAN_PATTERN, VPrim, parse_anchor, vprims_to_scene_text)

MULTI = ["cell phone", "traffic light", "teddy bear", "hot dog", "dining table"]


def _vp(cat, bbox=(120, 40, 180, 90), conf=0.90):
    return VPrim(category=cat, bbox=bbox, confidence=conf, source="t")


# ---------------- A 多词类别：describe / anchor_text 往返 ----------------

def test_a_multitoken_category_round_trips():
    """A 多词类别经 describe()/anchor_text() 写出后必须原样读回。"""
    bad = []
    for cat in MULTI:
        v = _vp(cat)
        for label, text in (("describe", v.describe()), ("anchor_text", v.anchor_text())):
            p = parse_anchor(text)
            got = p.category if p else None
            if got != cat:
                bad.append((cat, label, text, got))
    assert not bad, (
        "A 多词类别读不回（前 3 例 %r）—— parse_anchor 仍在按空白截断类别：见 #174"
        % (bad[:3],))


# ---------------- B 扫描贯通（core 的用法） ----------------

def test_b_scan_fragment_round_trips():
    """B 扫描片段的 `m.group(0)` 再解析也必须读回原类别（core 传的是片段，不是整串）。"""
    bad = []
    for cat in MULTI:
        v = _vp(cat)
        for text in (v.describe(), "[视觉原语] " + v.describe()):
            frags = [m.group(0) for m in re.finditer(ANCHOR_SCAN_PATTERN, text)]
            assert frags, "B 扫描形态未匹配到锚点：%r" % (text,)
            got = [parse_anchor(f).category if parse_anchor(f) else None for f in frags]
            if got != [cat]:
                bad.append((cat, text, frags, got))
    assert not bad, (
        "B 扫描片段读不回（前 2 例 %r）—— 扫描侧仍在截断类别，读回侧的修复被架空"
        % (bad[:2],))


# ---------------- C 场景文本全串 ----------------

def test_c_scene_text_round_trips():
    """C vprims_to_scene_text 全串逐条读回，category/bbox/conf 全对。"""
    vs = [_vp("cell phone", (120, 40, 180, 90), 0.90),
          _vp("traffic light", (10, 5, 30, 60), 0.90),
          _vp("person", (-4, 0, 60, 200), 0.52)]
    scene = vprims_to_scene_text(vs)
    got = []
    for m in re.finditer(ANCHOR_SCAN_PATTERN, scene):
        p = parse_anchor(m.group(0))
        if p:
            got.append((p.category, p.bbox, p.confidence))
    expect = [(v.category, v.bbox, v.confidence) for v in vs]
    assert got == expect, "C 场景文本读回不一致：\n got=%r\n exp=%r" % (got, expect)


# ---------------- D 既有口径不回归 ----------------

def test_d_existing_cases_still_work():
    """D 单词类别 / 非 \\w 类别 / 负坐标 / conf 必须照旧正确。"""
    for cat in ("cat", "moon", "person-1", "obj.2", "a/b"):
        v = _vp(cat)
        p = parse_anchor(v.anchor_text())
        assert p is not None and p.category == cat, (
            "D 单词类别 %r 读回 %r —— 修复把既有口径改坏了" % (cat, p.category if p else None))
    # 负坐标（#730）
    v = _vp("cat", (-3, 20, 50, 80))
    p = parse_anchor(v.describe())
    assert p is not None and p.bbox == (-3, 20, 50, 80), "D 负坐标读回坏了：%r" % (p.bbox if p else None,)
    # conf（#753 之后的具名组）
    for conf in (0.90, 0.52, 0.07):
        v = _vp("cat", (10, 5, 30, 60), conf)
        p = parse_anchor(v.describe())
        assert p is not None and abs(p.confidence - conf) < 1e-9, (
            "D conf=%.2f 读回 %r" % (conf, p.confidence if p else None))


# ---------------- E 已知限制如实钉住 ----------------

def test_e_free_text_prefix_keeps_known_limit():
    """E 已知限制如实钉住：**裸文本**（不加引号）的多词类别仍会被截成末词。

    本件修法 = 写侧对含空白的类别**加引号**（`_encode_category`）。因此：
      · `anchor_text()`/`describe()` 的产物（经写侧编码）⇒ 多词类别完整读回（A/B/C 已钉）；
      · 但若调用方**绕过写侧**、手写裸文本 `cell phone@(...)`，读回仍是 `phone`
        —— 因为裸文本里「类别内空格」与「类别前词间空格」信息论上不可区分。
    本守卫**故意断言仍会截断**，防止有人误以为「任意文本都能读多词类别」而删掉
    docstring 的诚实边界说明。若将来改为可识别无引号多词类别（须有边界约定），
    请同步更新 docstring 与本守卫。
    """
    p = parse_anchor("我看到 cell phone@(1,2,3,4) 在那里")
    assert p is not None
    assert p.category == "phone", (
        "E 裸文本多词类别的表现变了（现读回 %r）—— 若已能识别无引号多词类别，"
        "请同步更新 docstring 的诚实边界段与本守卫" % (p.category,))
    # 对照：经写侧编码（加引号）后同一条能完整读回
    v = _vp("cell phone", (1, 2, 3, 4))
    assert parse_anchor(v.anchor_text()).category == "cell phone"
