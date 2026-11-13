"""声明的抑制连接在初始化、显式载入和真实训练后都不得变为兴奋。"""
import numpy as np

from lingshu.nn.hex_hier import HexHierNet
from lingshu.nn.hex_unit import UnitHeadNet, finetune_units


def _unit():
    return UnitHeadNet(HexHierNet(seed=7), top_k=4)


def test_initial_connection_directions_match_the_real_base_network():
    unit = _unit()
    selected = unit.mask > 0
    expected = np.sign(unit.base.head[selected])
    actual = np.sign((unit.dir * unit.gain)[selected])
    assert np.array_equal(actual, expected), "initial inhibitory connections changed direction"


def test_inhibitory_feature_lowers_its_card_logit():
    unit = _unit()
    card, feature = np.argwhere((unit.mask > 0) & (unit.dir < 0))[0]
    inputs = np.zeros((2, unit.n_feat))
    inputs[1, feature] = 1.0
    logits = unit.logits(inputs)
    assert logits[1, card] < logits[0, card], "declared inhibitory feature raised its card logit"


def test_loading_negative_gain_cannot_flip_a_declared_direction():
    unit = _unit()
    vec = unit.get_vec()
    count = int(unit.mask.sum())
    vec[:count] = -1.0
    vec[count:] = np.arange(unit.n_card) - 4
    unit.set_vec(vec)
    assert np.all(unit.gain[unit.mask > 0] >= 0), "negative loaded gain reversed a declared connection"
    # #2168：theta 与 gain 同口径有界（±THETA_BOUND）——修前无界原样回读的
    # 断言与新契约冲突，同步为有界回读（方向不受影响的断言不变）。
    np.testing.assert_array_equal(
        unit.theta, np.clip(vec[count:], -unit.THETA_BOUND, unit.THETA_BOUND))
    vec[:count] = 100
    unit.set_vec(vec)
    assert np.all(unit.gain[unit.mask > 0] <= unit.GAIN_BOUND)


def test_finite_difference_training_preserves_the_frozen_connection_directions():
    unit = _unit()
    rng = np.random.default_rng(8)
    crops = rng.normal(size=(3, 3, 4, 3))
    before = unit.base.get_vec().copy()
    mask, direction = unit.mask.copy(), unit.dir.copy()
    curve = finetune_units(unit, crops, np.array([0, 1, 2]), steps=2, batch=3, lr=2.0)
    assert np.isfinite(curve).all()
    assert np.array_equal(unit.mask, mask) and np.array_equal(unit.dir, direction)
    assert np.all(unit.gain[mask > 0] >= 0), "training reversed a frozen connection direction"
    np.testing.assert_array_equal(unit.base.get_vec(), before)


def test_no_selected_connection_leaves_only_the_card_bias():
    unit = UnitHeadNet(HexHierNet(seed=3), top_k=0)
    unit.set_vec(np.arange(unit.n_card, dtype=float))
    inputs = np.ones((2, unit.n_feat))
    # #2168：theta 被 clip 到 ±THETA_BOUND ⇒ logits 的 bias 项随之钳制
    # （top_k=0 时 logits = -theta；theta=clip(0..8, ±2) = [0,1,2,2,2,2,2,2,2]）
    want = -np.clip(np.arange(unit.n_card), -2, 2)
    np.testing.assert_array_equal(unit.logits(inputs), np.tile(want, (2, 1)))


def test_loading_bias_owns_its_buffer_and_does_not_change_the_frontend():
    unit = _unit()
    before = unit.base.get_vec().copy()
    vec = unit.get_vec()
    unit.set_vec(vec)
    saved = unit.theta.copy()
    vec[:] = 100
    np.testing.assert_array_equal(unit.theta, saved)
    np.testing.assert_array_equal(unit.base.get_vec(), before)
