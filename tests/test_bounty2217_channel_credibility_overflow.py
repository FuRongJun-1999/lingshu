# -*- coding: utf-8 -*-
"""
Regression tests for FuRongJun-1999/dsh-bounty Issue #2217:
lingshu world/channel_credibility: _update bare float(conf) OverflowError.
Verifies fail-closed behavior (conf clamped to 0.0, zero n_eff, posteriors unchanged)
when receiving huge integers (10**309, 10**400, 2**2000), None, or invalid strings.
"""
import pytest
from lingshu.world.channel_credibility import ChannelCredibilityRegistry, PRIOR_A, PRIOR_B


def test_channel_credibility_overflow_huge_int_hit():
    registry = ChannelCredibilityRegistry()
    # Initial state
    st_init = registry.channel_state("visual")
    assert st_init["a"] == PRIOR_A
    assert st_init["b"] == PRIOR_B

    # Pass huge int 10**309, 10**400, 2**2000 as conf
    # Must fail-closed without raising OverflowError; n_eff should be 0.0 so state remains unchanged
    st_after = registry.record_hit("visual", 10**309)
    assert st_after["a"] == PRIOR_A
    assert st_after["b"] == PRIOR_B

    st_after_400 = registry.record_hit("visual", 10**400)
    assert st_after_400["a"] == PRIOR_A

    st_after_2000 = registry.record_hit("visual", 2**2000)
    assert st_after_2000["a"] == PRIOR_A


def test_channel_credibility_overflow_huge_int_miss():
    registry = ChannelCredibilityRegistry()
    st_init = registry.channel_state("tactile")
    assert st_init["b"] == PRIOR_B

    # Pass huge int 10**400 as refutation conf in record_miss
    st_after = registry.record_miss("tactile", 10**400)
    assert st_after["a"] == PRIOR_A
    assert st_after["b"] == PRIOR_B


def test_channel_credibility_type_error_and_value_error():
    registry = ChannelCredibilityRegistry()
    # Invalid string or None
    st_str = registry.record_hit("visual", "not_a_number")
    assert st_str["a"] == PRIOR_A

    st_none = registry.record_hit("visual", None)
    assert st_none["a"] == PRIOR_A


def test_channel_credibility_normal_hit_updates_posterior():
    registry = ChannelCredibilityRegistry()
    st_after = registry.record_hit("visual", 0.9, strong=True)
    # Valid hit should update posterior 'a' above prior
    assert st_after["a"] > PRIOR_A
    assert st_after["hits"] > 0.0
