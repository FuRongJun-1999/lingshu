# -*- coding: utf-8 -*-
"""#8 后半守卫：world3d 的类别先验表与默认回落必须显式登记（只登记，不改算法）。

缺陷面（#8 报告）：`add_vprim` 的 X/Y/Z 全部由 `DEFAULT_VISUAL_SPECS` 的
`spec.size[0]` 反投影推导；未知类别经 `default_spec` 静默回落 1m 立方体，
下游无法分辨这个深度是「量出来的」还是「猜出来的」。本件只做口径登记：
表头与 `default_spec` docstring 明写「类别先验 / 未登记先验 / 不是观测」，
不改变任何数值与算法。
"""
from pathlib import Path

from lingshu.world.world3d import DEFAULT_VISUAL_SPECS, default_spec

REPO = Path(__file__).resolve().parents[1]


def test_unknown_category_falls_back_to_1m_prior():
    spec = default_spec("zzz_unknown")
    assert spec.size == (1.0, 1.0, 1.0)
    assert "zzz_unknown" not in DEFAULT_VISUAL_SPECS


def test_known_category_reuses_table_entry():
    assert default_spec("person") is DEFAULT_VISUAL_SPECS["person"]


def test_prior_table_and_fallback_are_registered_as_prior():
    src = (REPO / "lingshu/world/world3d.py").read_text(encoding="utf-8")
    assert "类别先验" in src
    assert "未登记先验" in src
    assert "不是观测" in src
    doc = default_spec.__doc__ or ""
    assert "未登记先验" in doc
    assert "不是观测" in doc
