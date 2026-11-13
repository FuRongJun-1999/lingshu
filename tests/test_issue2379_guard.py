# -*- coding: utf-8 -*-
"""test_issue2379_guard · longterm_gate.evaluate(novelty) 值域闸回归钉
============================================================================
钉 PR fix/gapvalue-batch12 一处修复的**行为**（修前形态必红）：

  · #2379 core/longterm_gate.evaluate：调用方直传的 `novelty` 加 [0,1] 值域
    显式闸——修前零校验：`novelty=nan` 经 `min(1.0, nan)=1.0` 的语义陷阱被
    译成 imp=1.0 **长期层满分**（与 #2124/#457/#402 的「非有限不得给出对己
    有利结论」相反）、features.novelty=nan 使返回体严格 JSON 不可序列化；
    5.0/-1.0/inf 同样直通。内部自算路径 `_novelty` 有 [0,1] 钳制
    （longterm_gate.py:403-404），调用方路径对齐同款值域（fail-closed）。

运行（仓根）：python -X utf8 -m pytest tests/test_issue2379_guard.py
"""
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def _gate():
    from lingshu.core.core import SpacetimeMemoryEngine
    from lingshu.core.longterm_gate import LongTermMemoryGate
    e = SpacetimeMemoryEngine(db_path=":memory:")
    return LongTermMemoryGate(e)


def test_2379_nonfinite_novelty_rejected():
    g = _gate()
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            g.evaluate("测试快照内容甲乙丙丁", novelty=bad)


def test_2379_out_of_range_novelty_rejected():
    g = _gate()
    for bad in (5.0, -1.0, 1.01, -0.01):
        with pytest.raises(ValueError):
            g.evaluate("测试快照内容甲乙丙丁", novelty=bad)


def test_2379_non_numeric_novelty_rejected():
    g = _gate()
    for bad in ("0.5", True, None if False else [0.5]):
        with pytest.raises(ValueError):
            g.evaluate("测试快照内容甲乙丙丁", novelty=bad)


def test_2379_valid_novelty_unchanged():
    g = _gate()
    r = g.evaluate("测试快照内容甲乙丙丁", novelty=0.5)
    assert r["features"]["novelty"] == 0.5, "合法值路径行为不变"
    assert r["layer"] == "context" and r["importance"] < 1.0
    json.dumps(r, allow_nan=False)  # 返回体严格 JSON 可序列化
