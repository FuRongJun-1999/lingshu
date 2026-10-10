# -*- coding: utf-8 -*-
"""test_contradiction_aware · 矛盾感知守卫（LINGSHU_CONTRADICTION=1，默认关闭；Refs #142）

判据：
  · 默认关闭：近重复的更正照旧被 M5 合并并 +0.02（与改前一致）；
  · 开启：极性 / 数值 / 反义冲突的近重复不合并、不增信；新建节点，经既有 register_conflict
    通道打 OPPOSITE 边与 conflict 标签；
  · 开启：逐字重复、标点 / 语气词复述照常合并；
  · 开启：recall 把矛盾对侧紧随宿主呈现（limit=1 也不拆开），其余节点次序不变；
  · 检测器：豁免词（非常 / 没错 / 不过，…）不算否定。

变异哨兵：把 contradiction.conflict 改成恒返回 None，test_correction_not_swallowed 必红；
把 _with_opposites 改成 `return scored[:limit]`，test_recall_presents_both_sides 必红。
"""
import pytest

from lingshu.core.contradiction import conflict
from lingshu.core.core import EdgeType, SpacetimeMemoryEngine

pytestmark = pytest.mark.filterwarnings("ignore")

A = "根据病历记录，该用户对青霉素类抗生素严重过敏，开药时须避开"
A_NEG = "根据病历记录，该用户对青霉素类抗生素不严重过敏，开药时须避开"
DOSE = "该患者每天服用布洛芬，剂量为 7.5 毫克，饭后半小时服用，连续两周"
DOSE2 = "该患者每天服用布洛芬，剂量为 75 毫克，饭后半小时服用，连续两周"
SRV = "华东三号订单服务器目前处于在线状态，负责处理华东地区的全部订单"
SRV2 = "华东三号订单服务器目前处于离线状态，负责处理华东地区的全部订单"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    # 本文件断言的是旧置信度口径（+0.02）；证据账本 / TMS 全局开关会改口径，此处隔离
    for k in ("LINGSHU_EVIDENCE_LEDGER", "LINGSHU_TMS"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv("LINGSHU_CONTRADICTION", "1")


def _opp(e, nid):
    return [x for x in e.store.get_outgoing_edges(nid) + e.store.get_incoming_edges(nid)
            if x.relation_type == EdgeType.OPPOSITE]


def test_default_off_merges_as_before(monkeypatch):
    monkeypatch.delenv("LINGSHU_CONTRADICTION", raising=False)
    e = SpacetimeMemoryEngine(":memory:")
    a = e.add_perception(A)
    b = e.add_perception(A_NEG)
    assert b.id == a.id
    assert e.store.get_node(a.id).confidence == pytest.approx(0.52)


@pytest.mark.parametrize("old,new,reason", [(A, A_NEG, "polarity"), (DOSE, DOSE2, "number"),
                                            (SRV, SRV2, "antonym")])
def test_correction_not_swallowed(on, old, new, reason):
    e = SpacetimeMemoryEngine(":memory:")
    a = e.add_perception(old)
    b = e.add_perception(new)
    assert b.id != a.id
    assert e.store.get_node(a.id).confidence == pytest.approx(0.5)  # 原命题不因更正增信
    assert len(_opp(e, a.id)) == 1
    for nid in (a.id, b.id):
        tags = e.store.get_node(nid).tags
        assert "conflict" in tags and f"conflict:{reason}" in tags


def test_true_duplicates_still_merge(on):
    e = SpacetimeMemoryEngine(":memory:")
    a = e.add_perception(A)
    assert e.add_perception(A).id == a.id
    assert e.add_perception(A + "。").id == a.id
    assert not _opp(e, a.id)


def test_recall_presents_both_sides(on):
    e = SpacetimeMemoryEngine(":memory:")
    a = e.add_perception(A)
    b = e.add_perception(A_NEG)
    ids = [n.id for n, _ in e.recall("青霉素过敏", limit=1)]
    assert set(ids) == {a.id, b.id}


def test_detector_exemptions():
    assert conflict("今天天气特别好，适合出门", "今天天气非常好，适合出门") is None
    assert conflict("第7回电视剧奖男配角奖", "没错，第7回电视剧奖男配角奖") is None
    assert conflict("他不喜欢咖啡", "他并不喜欢咖啡") is None
    assert conflict("用户对青霉素过敏", "用户对青霉素不过敏") == "polarity"
    assert conflict("漏洞尚未修复", "漏洞已修复") == "polarity"
