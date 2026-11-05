# -*- coding: utf-8 -*-
"""#457 守卫：`cred_factor` 对非有限输入必须 fail-closed 归 floor，不得抬成 1.0。

缺陷面：`max(floor, min(ceil, f))` 里 CPython 的 `min(1.0, nan) == 1.0`，
NaN 被夹到上界＝满额保持（完全不衰减），与 #196/#402「非有限不得给出对己
有利结论」相反。本件按报告推荐路 A：非有限结果返回 `floor`。
两份实现（core / world）同源，必须同时守住。
"""
import math

import pytest

from lingshu.core import time_core as core_tc

# #176：world/time_core.py 副本已删，唯一真源为 lingshu/core/time_core.py
MODULES = [core_tc]


@pytest.mark.parametrize("tc", MODULES)
@pytest.mark.parametrize("gamma,dt", [
    (float("nan"), 1.0),
    (1.0, float("nan")),
    (float("-inf"), 1.0),
    (float("inf"), 1.0),
    (1.0, float("inf")),
])
def test_nonfinite_inputs_fail_closed_to_floor(tc, gamma, dt):
    assert tc.cred_factor(gamma, dt) == 0.0


@pytest.mark.parametrize("tc", MODULES)
def test_nonfinite_uses_given_floor(tc):
    assert tc.cred_factor(float("nan"), 1.0, floor=0.2, ceil=0.5) == 0.2


@pytest.mark.parametrize("tc", MODULES)
def test_normal_values_unchanged(tc):
    assert tc.cred_factor(1.0, 1.0) == pytest.approx(math.exp(-1.0))
    assert tc.cred_factor(1e9, 1.0) == 0.0
    assert tc.cred_factor(0.0, 0.0) == 1.0


def test_cred_still_propagates_nan():
    assert math.isnan(core_tc.cred(1.0, 0.0, float("nan"), 1.0))
