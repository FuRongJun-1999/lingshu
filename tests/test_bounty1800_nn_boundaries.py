"""dsh-bounty #1800 守卫：nn 三项边界。

缺陷（修复前实测，锚定 ed8700d）：
  ① `query_match` 对空连接库（公开 API 合法状态：`ConnectionLibrary().save()`
     落盘 `[]`、`load` 读回 0 条）在 `results[0]` 处 IndexError——四态契约
     （docstring 承诺 BLINDSPOT＝「库中无相关配方」）被崩溃打破。
  ② `HexHierNet` 的 `amp_log` 在 `stacked=False`（make_plan 认可配置）下
     被静默忽略——非 stacked 分支漏了 stacked 分支的 log 中心化，无界原始
     幅度直接进 head（M4.4 注释声明要防的形态）。
  ③ `images_to_lattices(空批次)` 返回 None——违反声明的返回类型
     np.ndarray，错误延迟到下游（normalize_lattices 抛 AttributeError）。

判据来源：dsh-bounty #1800（含逐字读数）。本文件是「回退即红」对照。
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingshu.nn.hex_cnn import image_to_grid  # noqa: E402
from lingshu.nn.hex_hier import HexHierNet, make_scene_dataset  # noqa: E402
from lingshu.nn.hex_query import ConnectionLibrary, query_match  # noqa: E402
from lingshu.nn.hex_train import images_to_lattices, normalize_lattices  # noqa: E402


# ------------------------------------------------------ ① 空库 BLINDSPOT

def test_i_empty_library_returns_blindspot_not_crash():
    d = Path(tempfile.mkdtemp())
    p = d / "lib.json"
    ConnectionLibrary().save(str(p))          # 落盘 "[]"（公开 API）
    lib = ConnectionLibrary.load(str(p))      # 读回 0 条
    assert len(lib) == 0
    out = query_match([{"obj": "circle|red", "pos": "r4"}], lib)
    assert isinstance(out, list) and out, out
    assert out[0]["verdict"] == "BLINDSPOT"   # 修复前：IndexError


# ------------------------------------------------ ② amp_log 在两配置下同效

def _amp_channel(stacked, amp_log):
    imgs, _ = make_scene_dataset(4, 32, seed=7)
    lat = normalize_lattices(np.stack([image_to_grid(im, 16)[0] for im in imgs]))
    net = HexHierNet(n_kernels=4, n_kernels2=4, stacked=stacked,
                     deep_norm=True, amp_sep=True, amp_log=amp_log, seed=7)
    sf, _ = net.l2_features(lat)
    return sf[:, -1].copy()


def test_ii_amp_log_effective_in_non_stacked():
    off = _amp_channel(False, False)
    on = _amp_channel(False, True)
    assert not np.array_equal(off, on), "amp_log 在 stacked=False 下必须生效"
    # 与 stacked 分支同效（对照组）
    assert not np.array_equal(_amp_channel(True, False), _amp_channel(True, True))


# ------------------------------------------- ③ 空批次返回数组不返回 None

def test_iii_empty_batch_returns_array():
    out = images_to_lattices(np.zeros((0, 8, 8, 3), dtype=np.uint8))
    assert isinstance(out, np.ndarray)        # 修复前：None
    assert out.shape[0] == 0 and out.dtype == np.float32


def test_iii_b_nonempty_batch_unchanged():
    """防过修正向：非空批次行为不变（形状/数值与 docstring 一致）。"""
    imgs, _ = make_scene_dataset(2, 32, seed=3)
    arr = np.stack(imgs)
    out = images_to_lattices(arr)
    assert out.shape[0] == 2 and out.shape[3] == 1
    assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0
