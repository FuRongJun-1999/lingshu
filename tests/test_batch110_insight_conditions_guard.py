# -*- coding: utf-8 -*-
"""批一百一十守卫：#2591 insight_record 条件快照零校验（NaN 落库 + 未知键静默丢失）。

覆盖
--------------------------------------------------------------------------
#2591 `_normalize_conditions` 对调用方 conditions **零校验**：
    · 未知键静默丢弃 ⇒ 调用方以为条件已记录，实则无痕丢失（同方法对
      content/importance 均 fail-loud，唯独条件面裸）；
    · NaN/±inf 落库成**非标准 JSON 字面量**（DB 行实测
      `"memory_retrievability": NaN`，strict 解析必崩）；
    · NaN 按显式提供拿 **0.7 高置信度**——NaN 冒充已测可靠条件。
    G1 行为级：合法键下 NaN ⇒ ValueError（M1 撤闸 ⇒ 落库成功 ⇒ 红）
    G2 行为级：合法键下 inf ⇒ ValueError
    G3 行为级：未知键 ⇒ ValueError 且列出合法键（M1 撤闸 ⇒ 静默落库 ⇒ 红）
    G4 正对照：合法混合条件照常记录、given=0.7 / 缺省=0.3 口径不变
    G5 源码级：未知键 fail-loud 与非有限 fail-loud 两闸在位

运行（lingshu 仓根）：
    python -X utf8 -m pytest tests/test_batch110_insight_conditions_guard.py -q
"""
from __future__ import annotations

import math
import sqlite3
import warnings

import pytest

import lingshu.core.core as _core
from lingshu.core.core import SpacetimeMemoryEngine


@pytest.fixture()
def engine(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eng = SpacetimeMemoryEngine(str(tmp_path / "svc.db"))
    return eng


def test_G1_nan_condition_rejected(engine):
    """G1：合法键下 NaN ⇒ ValueError（M1 撤闸 ⇒ NaN 落库成非标准 JSON ⇒ 红）。"""
    with pytest.raises(ValueError) as ei:
        engine.store.insight_record(
            "洞见", conditions={"memory_retrievability": float("nan")})
    assert "memory_retrievability" in str(ei.value)


def test_G2_inf_condition_rejected(engine):
    """G2：合法键下 ±inf ⇒ ValueError（同闸）。"""
    with pytest.raises(ValueError):
        engine.store.insight_record(
            "洞见", conditions={"memory_retrievability": float("inf")})
    with pytest.raises(ValueError):
        engine.store.insight_record(
            "洞见", conditions={"continuity_turns": float("-inf")})


def test_G3_unknown_condition_key_rejected(engine):
    """G3：未知键 ⇒ ValueError 且列出合法键（M1 撤闸 ⇒ 静默丢弃 ⇒ 红）。"""
    with pytest.raises(ValueError) as ei:
        engine.store.insight_record("洞见", conditions={"unknown_key_x": 1.0})
    assert "unknown_key_x" in str(ei.value)
    # 注释里的 C1 标签不是键名——误用 C1 也要拒（不留静默丢数据的口子）
    with pytest.raises(ValueError) as ei2:
        engine.store.insight_record("洞见", conditions={"C1": 0.7})
    assert "C1" in str(ei2.value)


def test_G4_valid_conditions_record_normally(engine):
    """G4 正对照：合法混合条件照常记录；given=0.7 / 缺省=0.3 口径不变。"""
    r = engine.store.insight_record(
        "正常洞见", conditions={"pressure": "low", "tone": "calm"})
    n = engine.store.get_node(r["node_id"])
    cond = n.state_attributes["insight"]["conditions"]
    assert cond["values"]["pressure"] == "low"
    assert cond["confidence"]["pressure"] == 0.7      # 显式提供
    assert cond["confidence"]["tone"] == 0.7
    assert cond["confidence"]["memory_retrievability"] == 0.3  # 缺省
    assert r["importance"] == 0.7


def test_G5_source_pins():
    """G5：源码锚——未知键 fail-loud 与非有限 fail-loud 两闸在位。"""
    import inspect
    src = inspect.getsource(_core)
    assert "insight conditions 未知键" in src, "缺未知键 fail-loud"
    assert "须为有限值，收到" in src, "缺非有限 fail-loud"
    # 旧静默丢弃形态必须消失
    body = src.split("def _normalize_conditions")[1].split("\n    def ")[0]
    assert "if k in vals:" not in body, "静默丢弃形态仍在"
