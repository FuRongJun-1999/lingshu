"""#339 守卫测试：grow_kernel 连续生长 6 次，新核必须与所有已有核互异。

修复前失败（第 2 次生长起新核与已有核逐位重复），修复后通过。
判据：新核与每个已有核的 L2 距离 > 0；且 mix 新列为零（新分支初始不影响输出）。
"""
import numpy as np

from lingshu.nn.hex_train import HexNet


def test_grow_kernel_no_duplicate_over_six_growths():
    net = HexNet(n_kernels=4, seed=7)
    assert net.conv.shape[0] == 4

    for g in range(1, 7):
        before = net.conv.copy()
        mix_cols_before = net.mix.shape[1]
        net.grow_kernel()

        new_k = net.conv[-1]
        dists = [float(np.linalg.norm(new_k - e)) for e in before]
        dups = [i for i, d in enumerate(dists) if d <= 0.0]
        assert not dups, f"第 {g} 次生长：新核与已有核 {dups} 重复（L2=0）"
        assert np.all(net.mix[:, mix_cols_before] == 0.0), f"第 {g} 次生长：mix 新列非零"
        assert net.K == net.conv.shape[0], f"第 {g} 次生长：K 与真实核数不同步"


def test_grow_kernel_uses_whole_handwritten_family_first():
    """手写核族 5 个，4 核起步：第 1 次生长必须补上第 5 个（不重复）。"""
    from lingshu.nn.hex_cnn import DEFAULT_KERNELS

    names = list(DEFAULT_KERNELS)
    net = HexNet(n_kernels=4, seed=7)
    net.grow_kernel()
    assert np.array_equal(net.conv[-1], DEFAULT_KERNELS[names[4]])


def test_grow_kernel_beyond_family_is_still_distinct():
    """用尽手写族后转扩容族，仍须互异（回归 #339 的回绕形态）。"""
    net = HexNet(n_kernels=4, seed=7)
    for _ in range(10):
        before = net.conv.copy()
        net.grow_kernel()
        dists = [float(np.linalg.norm(net.conv[-1] - e)) for e in before]
        assert min(dists) > 0.0, "生长后出现与已有核重复的新核"
    assert net.K == 14
