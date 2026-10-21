import math
import numpy as np
from lingshu.nn.hex_train import HexNet, train_infogap


def fixture():
    rng = np.random.default_rng(0)
    x = rng.random((4, 3, 3, 7)).astype(np.float32)
    y = np.array([0, 1, 0, 1])
    return HexNet(n_kernels=2, n_mix=4, seed=1), x, y


def assert_shape(res, net):
    required = {'algo','dead_zone','final_D','steps','growths','reverts','final_K','n_params','frozen_total','seconds','history'}
    assert required <= set(res), sorted(set(res))
    assert isinstance(res['history'], list)
    assert all(math.isfinite(float(res[k])) for k in ('dead_zone','final_D','seconds'))
    assert res['seconds'] >= 0
    assert res['final_K'] == net.K and res['n_params'] == net.n_params()


def test_z0_zero_steps_returns_initial_d0():
    net, x, y = fixture(); res = train_infogap(net, x, y, max_steps=0)
    assert_shape(res, net)
    assert res['steps'] == 0 and res['history'] == []
    assert res['final_D'] == 2.31565
    assert res['growths'] == res['reverts'] == res['frozen_total'] == 0


def test_zn_negative_one_preserves_empty_loop_contract():
    net, x, y = fixture(); res = train_infogap(net, x, y, max_steps=-1)
    assert_shape(res, net)
    assert res['steps'] == 0 and res['history'] == [] and res['final_D'] == 2.31565
    assert res['growths'] == res['reverts'] == res['frozen_total'] == 0


def test_t3_trains_and_returns_last_record():
    net, x, y = fixture(); res = train_infogap(net, x, y, max_steps=3, samples_per_step=4)
    assert_shape(res, net)
    assert 0 < res['steps'] == len(res['history']) <= 3
    assert res['final_D'] == res['history'][-1]['D']
    assert res['final_D'] != 2.31565


def test_c1_early_convergence_preserved():
    net, x, y = fixture(); res = train_infogap(net, x, y, max_steps=3, samples_per_step=4, dead_zone_ratio=2.0)
    assert_shape(res, net)
    assert res['steps'] == len(res['history']) == 1
    assert res['history'][0]['step'] == 0
    assert res['history'][0]['verdict'] == 'CONVERGED'
    assert res['final_D'] == res['history'][-1]['D'] == 2.31565
