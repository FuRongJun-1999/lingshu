# -*- coding: utf-8 -*-
"""#182 SelfModel.update 是无门控的 setattr —— 守卫（lingshu/core/core.py）

缺陷（lingshu issue #182；分诊表 `triage_lingshu_core_front.md` 行 182 判定成立）：
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
  任何调用方都能改写**任何**已存在属性——含敏感字段（`identity` / `values` /
  `trust_state`）、含方法（`update(update=…)` 影子覆盖）、含类常量
  （`update(HISTORY_MAX=0)` 令 history 钳制失效）；无白名单、无拒绝留痕。

判据来源：
  · 白名单取值 —— 经验标定（#182 修复轮），依据是「**各字段的专属写路径已存在**」：
    `identity` ← 构造器（`core.py` `SelfModel(identity=…)` / 引擎 `identity=` 形参）；
    `values` ← `SelfModel.record_value_change`（2.1.2 价值观版本化）；
    `trust_state`/`trust_history` ← `SelfModel.update_trust_state`（2.9 节）；
    `history`/`value_evolution` ← 本类内部维护。
    ⇒ 通用入口只放行无专属写路径的运行状态描述字段（`state_description` /
    `current_goal`）。
  · 拒写形态 —— 对齐仓内既有 fail-closed 取向（`add_node` 越权 ⇒ `PermissionError`）。

断言组（抽掉白名单即红）：
  S1 白名单字段照旧可写（防误杀）。
  S2 敏感字段（identity / values / trust_state）经 `update()` ⇒ `PermissionError`，
     且原值不变。
  S3 方法名 / 类常量经 `update()` ⇒ `PermissionError`（防影子覆盖）；`HISTORY_MAX`
     钳制仍生效、`update` 仍可调用。
  S4 未知键仍不落字段（与旧实现同效，不制造新失败面），但留痕于 history 的
     `rejected`（审计）。
  S5 端到端：`engine.update_self({"identity": ...})` 同样被拒（公开入口不绕过门控），
     且**零 SELF 快照落库**（拒写先于落盘，无部分写入）。
  S6 `history` 钳制与 `changes` 语义不回退（`current_goal` 变更仍记入 `changes`）。

运行（仓根）：python -X utf8 -m pytest tests/test_issue182_self_model_gate_guard.py -q --no-header
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from lingshu.core.core import SpacetimeMemoryEngine, SelfModel, MemoryLayer  # noqa: E402


def test_s1_whitelisted_fields_still_writable():
    """S1：白名单字段（运行状态描述）照旧可写。"""
    e = SpacetimeMemoryEngine(":memory:")
    e.update_self({"current_goal": "目标A", "state_description": "描述A"})
    assert e.self_model.current_goal == "目标A"
    assert e.self_model.state_description == "描述A"


@pytest.mark.parametrize("key,value", [
    ("identity", "被改写身份"),
    ("values", ["只剩一条"]),
    ("trust_state", {"p_trust": 0.0}),
])
def test_s2_sensitive_fields_denied(key, value):
    """S2：敏感字段经通用入口 ⇒ PermissionError 且原值不变。"""
    e = SpacetimeMemoryEngine(":memory:")
    before = getattr(e.self_model, key)
    with pytest.raises(PermissionError):
        e.self_model.update(**{key: value})
    assert getattr(e.self_model, key) == before, "%s 被改写" % key


@pytest.mark.parametrize("key,value", [
    ("update", lambda *a, **k: None),
    ("HISTORY_MAX", 0),
    ("TRUST_HISTORY_MAX", 0),
])
def test_s3_methods_and_class_constants_not_shadowable(key, value):
    """S3：方法名 / 类常量经通用入口 ⇒ PermissionError（旧实现可影子覆盖）。"""
    e = SpacetimeMemoryEngine(":memory:")
    with pytest.raises(PermissionError):
        e.self_model.update(**{key: value})
    assert SelfModel.HISTORY_MAX == 200, "类常量被改写"
    assert callable(e.self_model.update), "方法被影子覆盖"


def test_s4_unknown_keys_ignored_but_audited():
    """S4：未知键不落字段（同旧实现），但留痕 `rejected`。"""
    e = SpacetimeMemoryEngine(":memory:")
    e.self_model.update(完全未知键=1)
    assert not hasattr(e.self_model, "完全未知键")
    assert e.self_model.history[-1].get("rejected") == ["完全未知键"]


def test_s5_engine_entry_denies_and_writes_no_snapshot():
    """S5：引擎公开入口同样被拒，且零快照落库（无部分写入）。"""
    e = SpacetimeMemoryEngine(":memory:")
    with pytest.raises(PermissionError):
        e.update_self({"identity": "被改写身份"})
    assert e.self_model.identity == "协议实例"
    assert e.store.count_layer(MemoryLayer.SELF) == 0, "拒写后仍落了快照"


def test_s6_history_changes_semantics_hold():
    """S6：`changes` 仍记已落字段；history 钳制不回退。"""
    e = SpacetimeMemoryEngine(":memory:")
    e.update_self({"current_goal": "g1"})
    assert e.self_model.history[-1]["changes"] == {"current_goal": "g1"}
    for i in range(SelfModel.HISTORY_MAX + 20):
        e.update_self({"current_goal": "g%d" % i})
    assert len(e.self_model.history) == SelfModel.HISTORY_MAX
