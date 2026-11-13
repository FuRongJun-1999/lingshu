# -*- coding: utf-8 -*-
"""#942 / F23-06 守卫：`_meta_*` 审计改道命名空间不得被 meta 自带键覆盖。

缺陷面（R23 清单 F23-06）：meta 自带 `_meta_kernels` 时走普通写入，把
「meta 曾带 kernels」的审计回显覆盖成攻击者/调用方值。修法：`_meta_*`
前缀键改道二级命名空间（`__meta_*` 形态的 `_" + mk`），原值保留、真回显不动。

运行（lingshu 仓根）：python -X utf8 -m pytest tests/test_issue942_meta_namespace_guard.py -q
"""
import json
import os

import numpy as np
import pytest

from lingshu.nn.hex_cnn import load_consolidated, save_consolidated


def _kernels():
    return {"k1": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32)}


def test_f23_06_meta_key_cannot_override_audit_echo(tmp_path):
    """形态一：meta 只带 `_meta_kernels`（伪造审计位）——伪造不得落地。"""
    p = str(tmp_path / "t.json")
    save_consolidated(_kernels(), {"epochs": 8, "_meta_kernels": {"evil": True}}, p)
    raw = json.load(open(p, encoding="utf-8"))
    assert "_meta_kernels" not in raw, (
        "meta 自带 _meta_kernels 被普通写入＝伪造「meta 曾带 kernels」审计位（F23-06）")
    # 调用方原值保留（审计不删数据）：改道到二级命名空间
    assert raw.get("__meta_kernels") == {"evil": True}, (
        "meta 自带 _meta_* 键的原值应改道保留而非丢弃")


def test_f23_06_echo_not_overridden_when_both_present(tmp_path):
    """形态二：meta 同时带 `kernels`（触发真回显）与 `_meta_kernels`（覆盖攻击）
    ——真回显（setdefault 先落）不得被后续普通写入覆盖。"""
    p = str(tmp_path / "t2.json")
    save_consolidated(_kernels(), {"kernels": {"fake": 1}, "_meta_kernels": {"evil": True}}, p)
    raw = json.load(open(p, encoding="utf-8"))
    assert raw["_meta_kernels"] == {"fake": 1}, (
        "真回显（meta 带过 kernels）被 _meta_kernels 键覆盖（F23-06 覆盖形态）")
    assert raw.get("__meta_kernels") is not None, (
        "覆盖攻击的原值应改道保留而非丢弃")


def test_f23_06_normal_meta_unaffected(tmp_path):
    p = str(tmp_path / "t.json")
    save_consolidated(_kernels(), {"epochs": 8}, p)
    back = load_consolidated(p)
    assert "k1" in back
    raw = json.load(open(p, encoding="utf-8"))
    assert raw["epochs"] == 8 and "_meta_kernels" not in raw


def test_f23_06_load_still_roundtrip(tmp_path):
    p = str(tmp_path / "t.json")
    save_consolidated(_kernels(), {"_meta_kernels": {"evil": True}}, p)
    back = load_consolidated(p)
    assert np.allclose(back["k1"], _kernels()["k1"])
