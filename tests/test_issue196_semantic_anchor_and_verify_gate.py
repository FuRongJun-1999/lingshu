# -*- coding: utf-8 -*-
"""#196 census 第七批：semantic_anchor_graph.verify(evidence) ＋ anchored_verification.verify(conf) 非有限闸

三处收口：
- anchored_verification.verify(conf)：NaN/inf/bool → ValueError（不再静默写入信任状态）
- semantic_anchor_graph.verify(evidence)：NaN/inf/bool → ValueError（不再静默写入锚点确认）
- semantic_anchor_graph.verify(str)（evidence）→ TypeError

变异自证：摘除任一闸 → 守卫红 → 还原绿。
"""
import math
import pytest


def test_av_verify_nan_conf_raises():
    from lingshu.world.anchored_verification import AnchoredVerification
    v = AnchoredVerification()
    for bad in (float('nan'), float('inf'), float('-inf')):
        with pytest.raises(ValueError, match="conf 必须落在"):
            v.verify("visual", "action", conf=bad)


def test_av_verify_bool_conf_raises():
    from lingshu.world.anchored_verification import AnchoredVerification
    v = AnchoredVerification()
    with pytest.raises(TypeError, match="conf 必须是数值"):
        v.verify("visual", "action", conf=True)


def test_av_verify_valid_conf_passes():
    from lingshu.world.anchored_verification import AnchoredVerification
    v = AnchoredVerification()
    r = v.verify("visual", "action", conf=0.8)
    assert r["conf"] == 0.8


def test_sag_verify_nan_evidence_raises():
    from lingshu.world.semantic_anchor_graph import SemanticAnchorGraph
    g = SemanticAnchorGraph()
    # 构造一个锚点（触发闸前锚点必须存在）
    from lingshu.world.semantic_anchor_graph import SemanticAnchor
    # 直接往 dict 里放（避开构造器耦合）
    class _FakeAnchor:
        id = "a1"
        def to_dict(self):
            return {"id": "a1"}
    g.anchors["a1"] = _FakeAnchor()
    for bad in (float('nan'), float('inf'), float('-inf')):
        with pytest.raises(ValueError, match="evidence 必须落在"):
            g.verify("a1", "visual", evidence=bad, strong=True)


def test_sag_verify_bool_evidence_raises():
    from lingshu.world.semantic_anchor_graph import SemanticAnchorGraph
    g = SemanticAnchorGraph()
    class _FakeAnchor:
        id = "a1"
        def to_dict(self):
            return {"id": "a1"}
    g.anchors["a1"] = _FakeAnchor()
    with pytest.raises(TypeError, match="evidence 必须是数值"):
        g.verify("a1", "visual", evidence=True, strong=True)


def test_sag_verify_valid_evidence_passes():
    from lingshu.world.semantic_anchor_graph import SemanticAnchorGraph
    g = SemanticAnchorGraph()
    class _FakeAnchor:
        id = "a1"
        def to_dict(self):
            return {"id": "a1"}
    g.anchors["a1"] = _FakeAnchor()
    # 合法 evidence 不被闸拦（可能因锚点验证流程返回其他结果但不抛 TypeError/ValueError）
    try:
        g.verify("a1", "visual", evidence=0.8, strong=True)
    except (ValueError, TypeError):
        pytest.fail("合法 evidence=0.8 不应被类型/值域闸拦截")
