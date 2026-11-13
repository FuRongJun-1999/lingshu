# -*- coding: utf-8 -*-
"""#459 守卫：`record_info_gap` 的非有限 d_norm 不得被钳位抬到上界 1.0。

缺陷面（公开 issue #459 普查点名的 longterm 入口）：
`dn = max(0.0, min(1.0, d_norm))` —— Python `min(1.0, nan)` 返回 1.0（nan<1.0 为
False），NaN 被**抬到上界**伪装成「最大信息差」，进 gap_history 后被 Δ²D/趋势当
满值读数（#402 口径：非有限不得被当作有效证据，非有限一律回落钳制区间**下界**）。

本件锁：
  ① NaN/±Inf ⇒ 归 0.0（下界），不抛、不进上界；
  ② 正常值 0..1 原样；
  ③ 越界有限值钳到 [0,1]（既有口径不动）；
  ④ 返回值是 dn（钳后口径），不再返回原始未钳输入；
  ⑤ 四维加权推导路径遇任一 NaN 分量 ⇒ 整体按非有限归 0.0。
定点变异：去掉 isfinite 归零 ⇒ ① 红。
"""
import math

import pytest

from lingshu.core.core import SpacetimeMemoryEngine


@pytest.fixture
def e():
    return SpacetimeMemoryEngine(":memory:")


# ---------------------------------------------------------------- ① 非有限归下界

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_d_norm_falls_to_lower_bound(e, bad):
    got = e.record_info_gap(d_norm=bad)
    assert got == 0.0, "非有限 d_norm 应归下界 0.0，实得 %r" % (got,)
    row = e.store.conn.execute(
        "SELECT d_norm FROM gap_history ORDER BY id DESC LIMIT 1").fetchone()
    assert row and row[0] == 0.0, "gap_history 落库值也应为 0.0：%r" % (row,)


# ---------------------------------------------------------------- ② 正常值原样

def test_normal_value_passes_through(e):
    assert e.record_info_gap(d_norm=0.5) == 0.5
    assert e.record_info_gap(d_norm=0.0) == 0.0
    assert e.record_info_gap(d_norm=1.0) == 1.0


# ---------------------------------------------------------------- ③ 越界钳制

def test_out_of_range_finite_clamped(e):
    assert e.record_info_gap(d_norm=2.0) == 1.0
    assert e.record_info_gap(d_norm=-3.0) == 0.0


# ---------------------------------------------------------------- ④ 返回口径

def test_return_is_clamped_not_raw(e):
    got = e.record_info_gap(d_norm=7.5)
    assert got == 1.0, "返回值应为钳后 dn（而非原始输入）：%r" % (got,)


# ---------------------------------------------------------------- ⑤ 四维推导路径

def test_derived_path_with_nan_component_falls_to_zero(e):
    got = e.record_info_gap(trust_complement=float("nan"),
                            behavior_deviation=0.5,
                            connection_drift=0.5,
                            prediction_error=0.5)
    assert got == 0.0, "任一分量 NaN ⇒ 整体按非有限归 0.0：%r" % (got,)
