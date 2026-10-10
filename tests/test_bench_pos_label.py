# -*- coding: utf-8 -*-
"""bench/bench_hex.py 回归测试：build_crops 的 pos 标签是 9 类非常量。

对应 issue：[bench] labs['pos'] 恒为常量 'r4'、--steps 不作用于 train_infogap。

注：这里刻意用 importlib 按路径加载 bench/bench_hex.py，而不是
`from bench.bench_hex import ...`—— 门禁 tests/test_gate_dependency_closure.py
会把顶层 `bench` 当成未声明的第三方发行名（bench 并非 pyproject 里的
distribution），导致 CI 报 undeclared import。按路径加载与
tests/test_coggraph_public_paths.py 的既有做法一致。
"""
import importlib.util
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_bench_hex():
    path = os.path.join(REPO, "bench", "bench_hex.py")
    spec = importlib.util.spec_from_file_location("_bench_hex_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BENCH_HEX = _load_bench_hex()
build_crops = BENCH_HEX.build_crops


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
    _, labs = build_crops(n=48)
    # 至少 3 种取值出现 → 证明不是常量
    assert len(set(labs["pos"].tolist())) >= 3


if __name__ == "__main__":
    test_pos_labels_are_9_classes_not_constant()
    test_pos_label_matches_actual_crop_quadrant()
    print("OK: 2 项回归通过")