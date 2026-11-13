# -*- coding: utf-8 -*-
"""#138 守卫：非法值 / 首尾带空格的合法值不得被静默折叠成 __UNASSERTED__。

缺陷（50ee03b 修复前）：`canonical_condition` 对「不在枚举里的值」与「首尾带空格的合法值」
一律改写成 UNASSERTED —— 与「根本没声明该域」得到同一个 canonical dict，
因此 cond_hash 相同（写错的条件被静默路由进「未声明」的桶），
且 canonical_sourced / validate_edges 都无从报出。

issue 判据（逐字）：
  · " 动漫 " 与 "动漫" 应得到同一个 hash；
  · 非法值的 hash 应不同于未声明，或者显式报错/告警。

判别力：把 canonical_condition 的 strip / INVALID 分支回退（变异）后，本守卫必红。
仓库根：优先 LINGSHU_ROOT，其次 __file__ 上溯（`<repo>/tests/` 差一层），不写死沙箱路径。
"""
from __future__ import annotations

import os
import sys

REPO = os.environ.get("LINGSHU_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
print("ROOT=", REPO)
sys.path.insert(0, REPO)

from lingshu.core.condition_normalize import (  # noqa: E402
    INVALID, UNASSERTED, canonical_condition, canonical_sourced,
    cond_hash, invalid_domains, validate_edges,
)

BASE = {"subject": "人物", "style": "动漫"}


def test_a_invalid_differs_from_unasserted():
    """A：非法值的 hash 必须不同于「未声明」（修前两者相同 ⇒ 串桶）。"""
    assert cond_hash({"subject": "人物"}) != cond_hash({**BASE, "style": "油画"}), (
        "非法值与未声明同 hash —— 条件被静默折叠进「未声明」的桶")


def test_b_whitespace_padded_value_equals_plain():
    """B：' 动漫 ' 与 '动漫' 必须同 hash（先 strip 再查同义词）。"""
    assert cond_hash({**BASE, "style": "动漫 "}) == cond_hash(BASE)
    assert cond_hash({**BASE, "style": "  动漫  "}) == cond_hash(BASE)
    assert canonical_condition({**BASE, "style": " 动漫 "})["style"] == "动漫"


def test_c_invalid_is_reported_not_silent():
    """C：非法值必须可被调用方看见（canonical_sourced 的 invalid 字段）。"""
    cs = canonical_sourced({**BASE, "style": "油画"})
    assert cs["style"]["value"] == INVALID, cs["style"]
    assert cs["style"]["invalid"] == "油画", cs["style"]
    assert invalid_domains({**BASE, "style": "油画"}) == {"style": "油画"}


def test_d_validate_edges_surfaces_invalid():
    """D：validate_edges 必须报出非法值（修前完全看不见，返回 consistent）。"""
    r = validate_edges({**BASE, "style": "油画"})
    assert r["state"] == "DEFER_REVIEW", r
    assert "E0_INVALID" in [v["edge"] for v in r["violations"]], r
    # 反例：合法输入不得触发 E0
    ok = validate_edges(BASE)
    assert "E0_INVALID" not in [v["edge"] for v in ok["violations"]], ok


def test_e_unasserted_semantics_unchanged():
    """E：回归面 —— 「缺键」仍编码为 UNASSERTED，与 INVALID 互不相同。"""
    c = canonical_condition({"subject": "人物"})
    assert c["style"] == UNASSERTED, c
    assert UNASSERTED != INVALID
    assert c["style"] != INVALID


def test_f_invalid_occlusion_is_not_treated_as_occluded():
    """F：**归一后**的非法 occlusion（INVALID）不得被判「有遮挡」。

    边界（如实标注）：`assess` 的契约是收 `image_domains`，正常调用方传的是
    `canonical_condition(...)` 的结果；未归一的原始串不在本件范围（#138 修的是
    归一化层把非法值折叠成 UNASSERTED，以及随之而来的 INVALID 语义）。
    """
    from lingshu.core.verdict import assess

    raw = {"occlusion": "半遮"}                  # 不在枚举内
    canon = canonical_condition(raw)
    assert canon["occlusion"] == INVALID, canon  # 前提：归一后是 INVALID

    r = assess({"type": "head"}, canon, part_fg_px=9000, part_area=10000)
    assert r["verdict"] == "ACCEPT", r
    assert r.get("occluded") is not True, r

    # 对照：显式「部分遮挡」仍应 DEFER（不因本件放宽判据）
    r2 = assess({"type": "head"}, canonical_condition({"occlusion": "部分遮挡"}),
                part_fg_px=9000, part_area=10000)
    assert r2["verdict"] == "DEFER", r2
    assert r2.get("occluded") is True, r2