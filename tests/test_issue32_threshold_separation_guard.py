# -*- coding: utf-8 -*-
"""test_issue32_threshold_separation_guard · lingshu issue #32 守卫（**改声明**口径）

本件的结论（与 issue 的推论相反，故必须钉住，避免被再次误读）：
  · issue #32 的 12 组读数**数字可信**（本机逐项复算一致）；
  · 但「阈值 0.2611 落在未训练波动带内 ⇒ 通过不构成证据」这一**核心推论**
    **只在 `stacked=False`（M4.1）口径下成立**；测试 `test_hier_training_learns_object`
    的真实执行口径是 **`stacked=True`**（`HexHierNet(seed=7)` 未传 stacked，
    而 `hex_hier.py:101` 默认 True），该口径下未训练带 = [0.0833, 0.1562]、**0/6 越阈**，
    测试所用 data_seed=7 的未训练值 = 0.1250（距阈值 0.1361）⇒ **断言确实能区分**
    「未训练」与「训练后」。
  · 仍成立的那一半：仓内阈值 0.2611 与路线图 M4.2 目标 `>= 0.60` 相差 **33.9pp**
    ⇒ 已在断言处**如实声明**（本件不改数值行为、不改阈值）。

断言组（**只回滚 `tests/test_hex_hier.py` 的声明 ⇒ G1 变红**；只回滚源码即可判别）：
  G1 声明存在：断言块须写明「口径绑定」+「与路线图 0.60 相差 33.9pp」+「#32」。
  G2 真实执行口径的**分离性**：未训练(stacked=True, data_seed=7) < 阈值（这才是断言携带信息的判据）。
  G3 反例口径**仍越阈**（如实保留 issue 成立的那一半）：stacked=False 且 data_seed∈{23,13} 均 > 阈值。
  G4 口径前提：`HexHierNet` 的 `stacked` 默认必须是 True（若被改成 False，本件的口径分析全部失效 ⇒ 红）。

★ 不硬编码仓库路径：根由 `__file__` 上溯两层，可用 LINGSHU_ROOT 覆盖。
运行（仓根）：python -X utf8 -m pytest tests/test_issue32_threshold_separation_guard.py -q --no-header
"""
from __future__ import annotations

import inspect
import os
import sys
import warnings

ROOT = os.environ.get("LINGSHU_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

from lingshu.nn.hex_cnn import image_to_grid  # noqa: E402
from lingshu.nn.hex_hier import (  # noqa: E402
    HexHierNet, hier_report, make_scene_dataset)
from lingshu.nn.hex_train import normalize_lattices  # noqa: E402

N, SIZE, NET_SEED = 96, 48, 7
THRESHOLD = 1 / 9 + 0.15
TEST_FILE = os.path.join(ROOT, "tests", "test_hex_hier.py")


def _untrained(data_seed, stacked):
    xs, labs = make_scene_dataset(N, SIZE, data_seed)
    feats = [image_to_grid(xs[i], cells_across=16)[0] / 255.0 for i in range(N)]
    lat = normalize_lattices(np.stack(feats))
    return float(hier_report(HexHierNet(seed=NET_SEED, stacked=stacked), lat, labs)["acc_obj"])


def test_g1_declaration_present_at_assertion():
    """G1 声明存在（回滚本件的声明改动 ⇒ 本项红）。"""
    src = open(TEST_FILE, encoding="utf-8").read()
    assert "#32" in src, "断言块未标注 #32"
    assert "33.9 个百分点" in src, "未声明与路线图 0.60 的 33.9pp 差距"
    assert "口径绑定" in src, "未声明阈值判别力的口径绑定性质"
    assert "stacked 取默认 True" in src, "未写明测试真实执行口径"
    assert "0.1361" in src and "0.1250" in src, "未写明真实口径下的未训练读数与余量"


def test_g2_separation_holds_in_executed_config():
    """G2 真实执行口径(stacked=True, data_seed=7)下未训练 < 阈值 ⇒ 断言携带信息。"""
    v = _untrained(7, True)
    assert v < THRESHOLD, (
        "真实执行口径下未训练 acc_obj=%.4f 已 >= 阈值 %.4f ⇒ 该断言失去判别力"
        "（此时 issue #32 的推论就成立了，须重新裁定）" % (v, THRESHOLD))


def test_g3_counter_config_still_exceeds():
    """G3 如实保留 issue 成立的那一半：stacked=False 的 23/13 两 seed 仍越阈。"""
    for ds in (23, 13):
        v = _untrained(ds, False)
        assert v > THRESHOLD, (
            "stacked=False/data_seed=%d 未训练 acc_obj=%.4f 不再越阈 ⇒ 本件"
            "「推论仅在 stacked=False 成立」的表述需更新" % (ds, v))


def test_g4_premise_stacked_default_is_true():
    """G4 口径前提：stacked 默认必须是 True（被改则本件的口径分析失效）。"""
    sig = inspect.signature(HexHierNet.__init__)
    d = sig.parameters["stacked"].default
    assert d is True, "HexHierNet.stacked 默认已变为 %r ⇒ #32 的口径分析须重做" % (d,)