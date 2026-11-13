# -*- coding: utf-8 -*-
"""#501 守卫：权重表与 evaluate docstring 对 `trust`/`source` 的口径**如实**。

缺陷面（公开 issue #501，intake 申报）：权重表把 25% 的 `trust` 自述为
「来源可信度」，但 `evaluate` 的 `source` 形参零引用——`trust` 实取
`self_model.trust_state` 的会话级全局信任。本件**只动标注不动行为**：
把来源真正接进评分属设计裁定（改变落层行为），不在本守卫范围。

守卫锁：
  ① 五来源（user/assistant/external/fixture/tool）读数逐位相同——
     本件是「如实标注」，若未来有人接了 source，此断言会红，
     届时应同步更新 docstring 的「未参与评分」声明（双向如实）；
  ② 权重表 trust 注释含「#501 如实标注」；
  ③ evaluate docstring 声明「source 与 tags 形参当前未参与评分」。

运行（lingshu 仓根）：python -X utf8 -m pytest tests/test_issue501_trust_doc_honest_guard.py -q
"""
import inspect
import io
import sys

import pytest

from lingshu.core.longterm_gate import LongTermMemoryGate


def test_501_sources_readings_identical():
    """五来源读数逐位相同（当前行为锁定；接 source 时此断言必须连同声明一起更新）。"""
    from lingshu.core.core import SpacetimeMemoryEngine
    rows = []
    for src in ("user", "assistant", "external", "fixture", "tool"):
        e = SpacetimeMemoryEngine(":memory:")
        e.add_perception("锚点知识：六边形蜂窝网格等距邻居编码记录", importance=0.5)
        gate = e._ensure_gate()
        r = gate.evaluate("记忆：六边形蜂窝网格等距邻居编码方式与对称群阶数", source=src)
        rows.append((r["importance"], r["layer"], r["features"].get("trust")))
    assert len({tuple(r) for r in rows}) == 1, (
        "五来源读数出现分异 ⇒ source 已参与评分——必须同步更新 evaluate docstring "
        "与权重表注释的「未参与评分」声明（#501 双向如实）")


def test_501_weights_comment_honest():
    src = inspect.getsource(LongTermMemoryGate)
    assert "#501 如实标注" in src, "权重表缺 #501 如实标注"
    assert "非**本次快照的来源维度" in src or "非本次快照的来源维度" in src, (
        "权重表 trust 注释未声明「非本次快照来源维度」")


def test_501_evaluate_docstring_honest():
    doc = inspect.getdoc(LongTermMemoryGate.evaluate) or ""
    assert "#501 如实标注" in doc, "evaluate docstring 缺 #501 声明"
    assert "未参与" in doc and "source" in doc, (
        "evaluate docstring 未声明 source 未参与评分")
