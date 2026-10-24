# -*- coding: utf-8 -*-
"""#393 残余守卫：save_consolidated 的 meta 值经 `_json_safe` 透明转换。

主修复（#393）解决的是「先截断再 dump」的静默丢失（原子落盘）；本守卫钉住的是
**残余面**：meta 里的 numpy 标量/数组原样进 payload ⇒ `json.dump` 抛 TypeError。
该形态（`{"epochs": np.int64(8)}`）属常见调用面，旧代码 fail-fast 拒绝了本可
透明转换的输入。修复后：np.generic → `.item()`、ndarray → `tolist()`、嵌套容器
递归；kernels/consolidated/algo 结构字段保护（#349/#895）不受影响。

判别力（--mutate）：把 `hex_cnn._json_safe` 就地替换为恒等函数（等价于回退
本修复）⇒ 两条 meta 断言必红（TypeError）。

运行：
  python -X utf8 -m pytest tests/test_issue393_meta_json_safe.py -q
  python -X utf8 tests/test_issue393_meta_json_safe.py            # 脚本式全跑
  python -X utf8 tests/test_issue393_meta_json_safe.py --mutate   # 变异自证
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np

_RP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _RP)

from lingshu.nn.hex_cnn import save_consolidated  # noqa: E402

PASS = FAIL = 0


def _save(meta) -> tuple:
    d = tempfile.mkdtemp(prefix="i393_")
    p = os.path.join(d, "k.json")
    kernels = {"a": np.array([.1, .2, .1, .1, .2, .2, .1], np.float32)}
    save_consolidated(kernels, meta, p)
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def test_meta_numpy_scalars_are_transparent():
    d = _save({"source": "v2", "epochs": np.int64(8), "loss": np.float32(0.5)})
    assert d["epochs"] == 8 and type(d["epochs"]) is int
    assert abs(d["loss"] - 0.5) < 1e-6 and type(d["loss"]) is float
    assert d["kernels"]["a"] == [round(x, 6) for x in (.1, .2, .1, .1, .2, .2, .1)]


def test_meta_nested_containers_are_transparent():
    d = _save({"ctx": {"arr": np.array([1, 2]), "g": np.float64(1.25)},
               "rows": [np.int32(3), np.int32(4)]})
    assert d["ctx"]["arr"] == [1, 2]
    assert d["ctx"]["g"] == 1.25
    assert d["rows"] == [3, 4]


def test_structural_fields_still_protected():
    # #349/#895：meta 同名键不得覆盖结构字段
    d = _save({"kernels": "evil", "algo": "evil"})
    assert d["kernels"] != "evil" and d["algo"] != "evil"
    assert d["_meta_kernels"] == "evil" and d["_meta_algo"] == "evil"


def _run_all() -> int:
    checks = [test_meta_numpy_scalars_are_transparent,
              test_meta_nested_containers_are_transparent,
              test_structural_fields_still_protected]
    bad = 0
    for c in checks:
        try:
            c()
            print(f"[✓] {c.__name__}")
        except AssertionError as ex:
            bad += 1
            print(f"[✘] {c.__name__}: {ex}")
    return bad


def main(argv) -> int:
    if "--mutate" in argv:
        # 定点变异：`_json_safe` → 恒等（等价回退本修复）。
        # 变异点 = lingshu/nn/hex_cnn.py 中 meta 赋值处的 `_json_safe(mv)` 调用；
        # 进程内替换模块属性与源级替换该调用语义一致（该点唯一调用面）。
        import lingshu.nn.hex_cnn as hc
        assert hc._json_safe(np.int64(8)) == 8  # 修复在位
        hc._json_safe = lambda v: v
        bad = _run_all()
        print(f"#393 残余守卫（变异态）：{3 - bad} 通过 / {bad} 失败")
        print("判据：变异 ⇒ meta 两条必红（TypeError→AssertionError 路径）")
        return 0 if bad >= 2 else 1
    bad = _run_all()
    print(f"#393 残余守卫：{3 - bad} 通过 / {bad} 失败")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
