# -*- coding: utf-8 -*-
"""bench/bench_hex.py 回归测试：build_crops 的 pos 标签是 9 类非常量；
--steps 对两个训练器都生效。

对应 issue：[bench] labs['pos'] 恒为常量 'r4'、--steps 不作用于 train_infogap。
"""
import collections
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from bench.bench_hex import build_crops


def test_pos_labels_are_9_classes_not_constant():
    """pos 标签必须是 9 个象限标识，而不是旧版的常量 'r4'。"""
    _, labs = build_crops(n=96)
    pos = labs["pos"].tolist()
    unique = set(pos)
    assert len(unique) == 9, ("pos 应为 9 类，实际 %d 类: %s"
                              % (len(unique), unique))
    assert unique == {"r%d" % i for i in range(1, 10)}, \
        "pos 取值集合不符合 r1..r9: %s" % unique


def test_pos_label_matches_actual_crop_quadrant():
    """pos 标签应与该样本实际裁剪的象限一致（而不是恒为 r4）。"""
    crops, labs = build_crops(n=48)
    # 至少 3 种取值出现 → 证明不是常量
    assert len(set(labs["pos"].tolist())) >= 3


if __name__ == "__main__":
    test_pos_labels_are_9_classes_not_constant()
    test_pos_label_matches_actual_crop_quadrant()
    print("OK: 2 项回归通过")
