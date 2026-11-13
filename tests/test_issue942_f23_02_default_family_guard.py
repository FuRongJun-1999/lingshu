# -*- coding: utf-8 -*-
"""#942 / F23-02 守卫：HexNet 默认初始化集钉住＋族文档对称对如实。

R23 清单 F23-02：#339 顺带把默认初始化集从 DEFAULT_KERNELS 具名核改写为
`extended_prior_family(...)[0..3]`——基准漂移且无测试钉住；族 docstring
「无对称重复」对基族三对反向对不成立。

运行（lingshu 仓根）：python -X utf8 -m pytest tests/test_issue942_f23_02_default_family_guard.py -q
"""
import numpy as np

from lingshu.nn.hex_cnn import extended_prior_family
from lingshu.nn.hex_train import HexNet


def test_f23_02_default_init_is_family_head():
    """默认初始化集钉住：HexNet(n_kernels=4) 的核 == family[0..3]（防再漂移）。"""
    fam = [np.asarray(k).ravel() for k in extended_prior_family(len(__import__(
        "lingshu.nn.hex_cnn", fromlist=["DEFAULT_KERNELS"]).DEFAULT_KERNELS) + 32)]
    net = HexNet(n_kernels=4, seed=7)
    for i in range(4):
        row = np.asarray(net.conv[i]).ravel()
        assert np.array_equal(row, fam[i]), (
            "HexNet 初始核 %d 与 family[%d] 不一致——默认初始化集漂移（F23-02）" % (i, i))


def test_f23_02_antipodal_pairs_declared():
    """基族三对反向对实测在位（docstring 如实声明的事实基底）。"""
    fam = [np.asarray(k).ravel() for k in extended_prior_family(37)]
    for a, b in ((0, 3), (1, 4), (2, 5)):
        assert np.array_equal(fam[b], -fam[a]), (
            "fam[%d] != -fam[%d]——反向对事实变化，docstring 标注须同步" % (b, a))


def test_f23_02_docstring_honest():
    doc = extended_prior_family.__doc__ or ""
    assert "F23-02" in doc and "反向对" in doc, (
        "extended_prior_family docstring 缺 F23-02 对称对如实标注")
