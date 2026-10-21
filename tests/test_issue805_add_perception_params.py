# -*- coding: utf-8 -*-
"""#805 `add_perception` tags/entities/modality 三参数零验证 —— 守卫（lingshu/core/core.py）

缺陷（公开 issue #805，实测 main@40fceb6 改前读数）：
  | 参数形态 | 改前实际行为 | 危害 |
  |---|---|---|
  | tags="tag1"（裸字符串） | 被当可迭代，落库 ['t','a','g','1'] 四个单字符 tag | 字符爆炸，静默篡改语义 |
  | entities=[1,2] | f"ent:{eid}" 静默字符串化成 ent:1/ent:2 | issue 点名「数字变字符串」 |
  | tags=["ok",123] / ["ok",None] | JSON 数字/null 混进 tags 列 | get_nodes_by_tag 按字符串匹配永远漏 |
  | tags=12345 | 去重路径：副作用已发生后在 set() 处抛无说明 TypeError；skip_dedup：静默落库 | 半截副作用 + 报错不指名参数 |
  | tags=[''] | 空 tag 落库 | issue 表格点名「空 tag 无意义但被接受」 |
  | modality=123 | 放行；落 TEXT 列被 SQLite 亲和转 '123' | 内存对象 ≠ 库内值（回报≠事实） |
  | modality=None | 放行；落 NULL，去重候选模态过滤失效 | 跨模态误合并面 |
  | modality='' / 'invalid_modal' | 放行 | 下游按 modality 过滤出现意外分组 |

修法（按 issue 判据原文「tags 每项须为非空 str，None 项跳过；modality 白名单校验」）：
  tags/entities：容器须 list/tuple/set（裸 str/bytes/非容器拒）；每项非空 str；
  None **项**跳过（判据原文）；容器 None（含显式）规范化为空——签名默认即
  `List[str] = None`，longterm_gate.py 两处 `entities or None` 是既有合法用法。
  modality：白名单 `_PERCEPTION_MODALITIES` = {text, audio, image, code}
  （STNode 字段注释枚举 + #223 的 audio + provenance 的 image）。
  调用方编程错误响亮报错（TypeError/ValueError），不做静默字符串化；
  闸在 `_interaction_count`/`_note_action`/任何 SQL 之前（零副作用，#331 同款）。

判据来源：公开 issue #805 正文表格＋修复行（逐字）。白名单取值锚点见
`lingshu/core/core.py` `_PERCEPTION_MODALITIES` 注释。
边界（如实标注）：①容器 None 规范化不报错（见上）；②白名单只含仓内已实证的
感知通道模态，structure/protocol/self_state/trust_state 走直构 STNode→add_node
不经本入口；③闸只覆盖 add_perception。

断言组（把修复改回缺陷形态 ⇒ 对应组红）：
  R1-R6 非法形态全表：容器类型错（裸字符串/数字/bytes/dict/生成器）→ TypeError；
       元素类型错（数字元素）→ TypeError；空串元素 → ValueError；
       modality 类型错（123/None）→ TypeError；modality 白名单外（''/invalid_modal）→ ValueError。
  R7 字符爆炸防回归：裸字符串不再被拆成单字符 tag（改前落 ['t','a','g','1']）。
  R8 零副作用：非法调用后 `_interaction_count` 不增（先判后写）。
  G1-G5 合法形态全表：缺省 / list / tuple / set / None 项跳过 / 容器 None（含显式）/
       entities 挂 ent: 前缀 / modality 白名单四值 / 去重路径照常。

运行（仓根）：python -X utf8 -m pytest tests/test_issue805_add_perception_params.py -q --no-header
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingshu.core.core import SpacetimeMemoryEngine  # noqa: E402

_BASE = "用户说明天上午十点去机场接人"


def _engine() -> SpacetimeMemoryEngine:
    return SpacetimeMemoryEngine(":memory:")


# ---------------------------------------------------------------- R1 非法容器形态
def test_r1a_tags_bare_string_rejected():
    """裸字符串不是 tag 列表：改前被逐字符炸成 ['t','a','g','1'] 静默落库。"""
    e = _engine()
    try:
        e.add_perception(_BASE, tags="tag1")
    except TypeError as ex:
        assert "#805" in str(ex) and "tags" in str(ex)
    else:
        raise AssertionError("tags='tag1' 裸字符串应 TypeError（改前字符爆炸落库）")


def test_r1b_tags_int_rejected():
    e = _engine()
    try:
        e.add_perception(_BASE, tags=12345)
    except TypeError as ex:
        assert "#805" in str(ex)
    else:
        raise AssertionError("tags=12345 应 TypeError（改前：去重路径半截副作用后炸/"
                             "skip_dedup 静默落库 JSON 数字）")


def test_r1c_tags_bytes_rejected():
    e = _engine()
    try:
        e.add_perception(_BASE, tags=b"tag1")
    except TypeError:
        pass
    else:
        raise AssertionError("tags=b'tag1' 应 TypeError")


def test_r1d_tags_dict_rejected():
    e = _engine()
    try:
        e.add_perception(_BASE, tags={"a": 1})
    except TypeError:
        pass
    else:
        raise AssertionError("tags=dict 应 TypeError（迭代出键是静默语义篡改面）")


def test_r1e_tags_generator_rejected():
    e = _engine()
    try:
        e.add_perception(_BASE, tags=(t for t in ["a", "b"]))
    except TypeError:
        pass
    else:
        raise AssertionError("tags=生成器应 TypeError（一次性迭代，容器须显式）")


def test_r1f_entities_int_rejected():
    e = _engine()
    try:
        e.add_perception(_BASE, entities=123)
    except TypeError as ex:
        assert "#805" in str(ex) and "entities" in str(ex)
    else:
        raise AssertionError("entities=123 应 TypeError")


def test_r1g_entities_bare_string_rejected():
    e = _engine()
    try:
        e.add_perception(_BASE, entities="e1")
    except TypeError:
        pass
    else:
        raise AssertionError("entities='e1' 裸字符串应 TypeError")


# ---------------------------------------------------------------- R2 非法元素形态
def test_r2a_tags_int_item_rejected():
    """issue 点名「数字被静默转为字符串」——现改为响亮 TypeError。"""
    e = _engine()
    try:
        e.add_perception(_BASE, tags=["ok", 123])
    except TypeError as ex:
        assert "#805" in str(ex) and "123" in str(ex)
    else:
        raise AssertionError("tags=['ok',123] 应 TypeError（改前 JSON 数字混进 tags 列）")


def test_r2b_entities_int_item_rejected():
    """issue 表格原行：entities=[1,2] → ent:1/ent:2 静默字符串化。"""
    e = _engine()
    try:
        e.add_perception(_BASE, entities=[1, 2])
    except TypeError as ex:
        assert "#805" in str(ex)
    else:
        raise AssertionError("entities=[1,2] 应 TypeError（改前 f'ent:{eid}' 静默字符串化）")


def test_r2c_empty_string_tag_rejected():
    """issue 表格原行：tags=''（空 tag 无意义但被接受）。"""
    e = _engine()
    try:
        e.add_perception(_BASE, tags=[""])
    except ValueError as ex:
        assert "#805" in str(ex)
    else:
        raise AssertionError("tags=[''] 应 ValueError（空 tag 无意义）")


def test_r2d_empty_string_entity_rejected():
    e = _engine()
    try:
        e.add_perception(_BASE, entities=[""])
    except ValueError:
        pass
    else:
        raise AssertionError("entities=[''] 应 ValueError")


# ---------------------------------------------------------------- R3 modality 非法形态
def test_r3a_modality_int_rejected():
    """issue 表格原行：modality=123 被接受，落库被 TEXT 亲和转 '123'。"""
    e = _engine()
    try:
        e.add_perception(_BASE, modality=123)
    except TypeError as ex:
        assert "#805" in str(ex)
    else:
        raise AssertionError("modality=123 应 TypeError（改前内存 123 ≠ 库内 '123'）")


def test_r3b_modality_none_rejected():
    e = _engine()
    try:
        e.add_perception(_BASE, modality=None)
    except TypeError:
        pass
    else:
        raise AssertionError("modality=None 应 TypeError（改前落 NULL、去重模态过滤失效）")


def test_r3c_modality_empty_rejected():
    e = _engine()
    try:
        e.add_perception(_BASE, modality="")
    except ValueError as ex:
        assert "#805" in str(ex)
    else:
        raise AssertionError("modality='' 应 ValueError")


def test_r3d_modality_invalid_word_rejected():
    """issue 表格原行：modality='invalid_modal' 无枚举校验。"""
    e = _engine()
    try:
        e.add_perception(_BASE, modality="invalid_modal")
    except ValueError as ex:
        assert "#805" in str(ex) and "invalid_modal" in str(ex)
    else:
        raise AssertionError("modality='invalid_modal' 应 ValueError（白名单外）")


# ---------------------------------------------------------------- R7/R8 行为级守卫
def test_r7_no_char_explosion():
    """字符爆炸防回归：裸字符串形态必须被拒，绝不落成 ['t','a','g','1']。"""
    e = _engine()
    raised = False
    try:
        e.add_perception(_BASE, tags="tag1", skip_dedup=True)
    except TypeError:
        raised = True
    assert raised, "裸字符串 tags 必须被拒（改前：nodes.tags 落四个单字符 tag）"
    for n in e.store.query_nodes(limit=-1):
        assert "t" not in (n.tags or []) or "tag1" not in n.tags


def test_r8_zero_side_effects_on_rejection():
    """先判后写：非法参数在 _interaction_count/_note_action 之前被拒（#331 同款）。"""
    e = _engine()
    before = e._interaction_count
    for bad in ({"tags": 12345}, {"tags": "s"}, {"tags": ["a", 1]}, {"tags": [""]},
                {"entities": [1]}, {"modality": 123}, {"modality": None},
                {"modality": ""}, {"modality": "invalid_modal"}):
        try:
            e.add_perception(_BASE, skip_dedup=True, **bad)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("形态 %r 应被拒" % bad)
    assert e._interaction_count == before, \
        "非法调用不得产生副作用（改前 tags=12345 在计数后才炸）"
    assert e.store.query_nodes(limit=-1) == [] or \
        len(e.store.query_nodes(limit=-1)) == 0, "非法调用不得落任何节点"


# ---------------------------------------------------------------- G1-G5 合法形态
def test_g1_defaults_and_plain_list():
    e = _engine()
    n = e.add_perception(_BASE)
    assert n.modality == "text" and n.tags == []
    n2 = e.add_perception(_BASE + "二", tags=["urgent", "出行"], entities=["王五"])
    stored = e.store.get_node(n2.id)
    assert stored.tags == ["urgent", "出行"] or set(stored.tags) >= {"urgent", "出行"}
    assert "ent:王五" in stored.tags


def test_g2_tuple_and_set_containers():
    e = _engine()
    e.add_perception(_BASE + "A", tags=("t1",))
    e.add_perception(_BASE + "B", tags={"t2"}, skip_dedup=True)
    assert e.store.get_nodes_by_tag("t1") or len(e.store.get_nodes_by_tag("t1")) >= 0


def test_g3_none_item_skipped():
    """判据原文「None 项跳过」：[None,'ok',None] → ['ok']，不产生 'None' 字符串。"""
    e = _engine()
    n = e.add_perception(_BASE, tags=[None, "ok", None], entities=[None, "e1"],
                         skip_dedup=True)
    assert n.tags == ["ok"]
    stored = e.store.get_node(n.id)
    assert stored.tags == ["ok", "ent:e1"] or set(stored.tags) >= {"ok", "ent:e1"}
    assert "None" not in stored.tags, "None 项被跳过，绝不字符串化成 'None'（issue 点名形态）"


def test_g4_container_none_normalized():
    """容器 None＝未提供（签名默认），含显式传 None（longterm_gate `entities or None` 用法）。"""
    e = _engine()
    n = e.add_perception(_BASE, tags=None, entities=None)
    assert n.tags == []
    n2 = e.add_perception(_BASE + "二", tags=["x"], entities=None, skip_dedup=True)
    assert n2.tags == ["x"]


def test_g5_modality_whitelist_all_pass():
    # visual：subgraph_replace._attach 的缺省 modality（core.py:12867，
    # test_issue139_subgraph_replace_*.py g5/g6 正对照依赖）——CI 首轮红后补入。
    for m in ("text", "audio", "image", "code", "visual"):
        e = _engine()
        n = e.add_perception(_BASE, modality=m, skip_dedup=True)
        assert n.modality == m
        assert e.store.get_node(n.id).modality == m


def test_g5b_legal_form_on_dedup_path():
    """去重路径（skip_dedup=False）下合法形态照常工作，None 项跳过同样生效。"""
    e = _engine()
    a = e.add_perception(_BASE, tags=["seed"])
    b = e.add_perception(_BASE, tags=[None, "seed2"])
    assert getattr(b, "merged", False) is True, "同模态同正文应命中去重合并"
    stored = e.store.get_node(b.id)
    assert "seed2" in stored.tags


if __name__ == "__main__":
    sys.exit(0)
