# -*- coding: utf-8 -*-
"""
Regression unit tests for Issue #2218:
lingshu world/confirmation: kl_binary and gain_task bare float() OverflowError.
Verifies fail-closed behavior (returns 0.0) on huge integers (10**309, 10**400, 2**2000),
bad strings, None, and maintains mathematical consistency on normal inputs.
"""
import math
import pytest
from lingshu.world.confirmation import kl_binary, gain_task, _sigmoid, value


def test_kl_binary_overflow_huge_int_p_after():
    # 10**309 and 10**400 exceed IEEE 754 float64 max (~1.79e308)
    # Must fail-closed returning 0.0 instead of raising bare OverflowError
    assert kl_binary(10**309, 0.5) == 0.0
    assert kl_binary(10**400, 0.5) == 0.0
    assert kl_binary(2**2000, 0.5) == 0.0


def test_kl_binary_overflow_huge_int_p_before():
    assert kl_binary(0.5, 10**309) == 0.0
    assert kl_binary(0.5, 10**400) == 0.0
    assert kl_binary(0.5, 2**2000) == 0.0


def test_kl_binary_type_error_value_error():
    assert kl_binary("abc", 0.5) == 0.0
    assert kl_binary(0.5, "abc") == 0.0
    assert kl_binary(None, 0.5) == 0.0
    assert kl_binary(0.5, None) == 0.0


def test_kl_binary_finite_and_symmetry():
    # Normal inputs should maintain exact mathematical properties
    assert kl_binary(0.5, 0.5) == 0.0
    diff1 = kl_binary(0.9, 0.5)
    diff2 = kl_binary(0.1, 0.5)
    assert math.isclose(diff1, diff2, rel_tol=1e-9)
    assert diff1 > 0.3


def test_gain_task_overflow_huge_int_task_relevance():
    assert gain_task(0.9, 0.5, task_relevance=10**400) == 0.0
    assert gain_task(0.9, 0.5, task_relevance=10**309) == 0.0
    assert gain_task(0.9, 0.5, task_relevance=2**2000) == 0.0


def test_gain_task_overflow_propagation_from_p():
    assert gain_task(10**400, 0.5, task_relevance=1.0) == 0.0
    assert gain_task(0.5, 10**400, task_relevance=1.0) == 0.0


def test_gain_task_type_and_value_error():
    assert gain_task(0.9, 0.5, task_relevance="invalid") == 0.0
    assert gain_task(0.9, 0.5, task_relevance=None) == 0.0


def test_gain_task_normal_operation():
    g = gain_task(0.9, 0.5, task_relevance=0.5)
    assert g > 0.0
    assert math.isclose(g, 0.5 * kl_binary(0.9, 0.5), rel_tol=1e-9)
