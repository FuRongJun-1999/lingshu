"""#1063 守卫：condition_normalize 对自身输出封闭 + 类型错误落 INVALID。

缺陷（修复前实测，锚定 ee4da17）：
  ① 二次归一：`__UNASSERTED__` 不在 allowed ⇒ 判 INVALID（幂等破坏、cond_hash 漂移、
     validate_edges 把 21 个「未声明」域全报 E0_INVALID 并判 DEFER_REVIEW）
  ② 非法类型（list/dict/set）：`SYNONYMS.get(v, v)` unhashable TypeError 崩溃，
     而文件头第 39 行承诺「类型不对」属非法值（应落 INVALID）
本文件是「回退即红」对照。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingshu.core.condition_normalize import (  # noqa: E402
    INVALID,
    UNASSERTED,
    canonical_condition,
    cond_hash,
    invalid_domains,
    validate_edges,
)


def test_g1_canonical_is_idempotent():
    raw = {"subject": "人物", "style": "动漫"}
    c1 = canonical_condition(raw)
    c2 = canonical_condition(c1)
    assert c1 == c2
    assert cond_hash(c1) == cond_hash(c2)


def test_g2_unasserted_stays_unasserted_and_does_not_raise_e0():
    c1 = canonical_condition({"subject": "人物"})       # 其余 21 域未声明
    assert c1["style"] == UNASSERTED
    assert invalid_domains(c1) == {}                     # 未声明 ≠ 非法
    assert validate_edges(c1)["state"] == "consistent"   # 修复前 DEFER_REVIEW + 21 个 E0


@pytest.mark.parametrize("bad", [["动漫"], {"k": "v"}, {"动漫"}])
def test_g3_unhashable_values_land_in_invalid(bad):
    for fn in (canonical_condition, cond_hash, validate_edges, invalid_domains):
        fn({"style": bad})                               # 修复前 TypeError
    assert canonical_condition({"style": bad})["style"] == INVALID


def test_g4_legal_paths_unchanged():
    """防过修正向：strip/同义词/域内合法值/域外值逐字不变。"""
    assert canonical_condition({"style": " 动漫 "})["style"] == "动漫"   # strip + 合法
    assert canonical_condition({"background": "纯白"})["background"] == "白"  # 同义词
    assert canonical_condition({"style": "纯白"})["style"] == INVALID   # 域外（确定性）
    assert canonical_condition({"style": "3D"})["style"] == "3D"
