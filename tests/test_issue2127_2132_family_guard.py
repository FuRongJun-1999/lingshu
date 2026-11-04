# -*- coding: utf-8 -*-
"""test_issue2127_2132_family_guard · nn/gen/world 族五处修复回归钉
============================================================================
本件钉 PR fix/gen-nn-batch5 五处修复的**行为**（每条都是修前形态必红）：

  · #2127 nn.hex_train：`four_state_accuracy` / `HexNet.accuracy` 对空评估集
    不再除零（修前 ZeroDivisionError；同函数 acc_total 的 max(1,…) 有防护、
    len(x)=0 没有——同行防护不一致即遗漏）。空集按「无样本不显示为满分/零分」
    返回 None（accuracy）/ accept_precision=None + n=0（four_state）。
  · #2129 world.seven_layer_loop.predict_self：y 直通不取整（与执行侧逐位对齐
    ——修前受控体 y 带小数时 self_error 每 tick 恒等于 |y−round(y,2)|，校验
    字段永久误报）。
  · #2130 gen.hexgen_self_source._solve_placement：set 入参先 sorted 再按序
    访问（R311 同函数先例 :760/:964 同款）——修前 spread/zone/hybrid 三档对
    set 入参直接 'set' object is not subscriptable。
  · #2131 gen.hexgen_pack_certify.support_vec：入口 `_v3_shape` 归一（与生产侧
    paint_support 同源）——修前 stripe 落各向同性回退 41.0、生产面 57.57，
    可行方向判决不健全。
  · #2132 nn.hex_ortho：晶格/核含非有限值 ⇒ fail-closed 返回 stop_reason=
    "invalid_input"（修前两个语义停因被 NaN 比较恒 False 绕过，剥到核族耗尽并
    误标良性 card_exhausted、报告整体 NaN 化）；gram_schmidt_kernels 的 ratio
    非有限保持 None 不再把损坏核静默判成「完全重复解释」。

运行（仓根）：python -X utf8 -m pytest tests/test_issue2127_2132_family_guard.py
"""
import math
import sys

import numpy as np
import pytest

REPO = __import__("os").path.dirname(__import__("os").path.dirname(
    __import__("os").path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


# ---------------------------------------------------------------- #2127

def test_2127_empty_eval_set_no_division():
    from lingshu.nn import hex_train as _HT
    net = _HT.HexNet()
    cards = [{"class": c, "margin_threshold": 2.0, "p_threshold": 0.6}
             for c in range(10)]
    r = _HT.four_state_accuracy(net, np.zeros((0, 8, 8, 1)), np.array([]), cards)
    assert r["n"] == 0 and r["accept_coverage"] is None and \
        r["accept_precision"] is None, "空集不得除零，指标返回 None：%r" % (r,)
    assert net.accuracy(np.zeros((0,)), np.zeros((0,))) is None


def test_2127_nonempty_paths_unchanged():
    from lingshu.nn import hex_train as _HT
    net = _HT.HexNet()
    cards = [{"class": c, "margin_threshold": 2.0, "p_threshold": 0.6}
             for c in range(10)]
    x = np.zeros((2, 8, 8, 1), dtype=float)
    y = np.array([0, 1])
    r = _HT.four_state_accuracy(net, x, y, cards)
    assert r["n"] == 2 and r["accept_coverage"] is not None, (
        "非空集路径行为不变（修前口径），实测 %r" % (r,))
    a = net.accuracy(x, y)
    assert a is not None and math.isfinite(float(a)), "非空 accuracy 正常返回"


# ---------------------------------------------------------------- #2129

def test_2129_predict_self_y_passthrough():
    from lingshu.world.seven_layer_loop import ActionDecision
    ps = ActionDecision.predict_self((2.0, 1.234, 2.0), "east", 1.0, 64)
    assert ps[1] == 1.234, (
        "y 直通（与执行侧逐位对齐；修前 round 成 1.23 制造 self_error 偏影）：%r"
        % (ps,))
    assert ps[0] == 3.0 and ps[2] == 2.0, "x/z 取整口径不变"


# ---------------------------------------------------------------- #2130

def test_2130_set_cells_all_modes():
    from lingshu.gen import hexgen_self_source as _S
    cells = {(c, r) for c in range(6) for r in range(6)}
    for mode in ("spread", "hybrid", "zone"):
        _S.PLACE_MODE = mode
        try:
            pts, _r = _S.place_objects(cells, 3, 1.0)
            assert len(pts) == 3, "%s 档应产出 3 点，实测 %r" % (mode, pts)
        finally:
            _S.PLACE_MODE = "centroid"


# ---------------------------------------------------------------- #2131

def test_2131_support_vec_matches_production_on_stripe():
    from lingshu.gen import hexgen_pack_certify as _PC
    from lingshu.gen import hexgen_self_source as _S
    sv = float(_PC.support_vec("stripe", 40, 1, 1, math.sqrt(2)))
    pv = float(_S.paint_support("stripe", 40, 0.7071, 0.7071))
    assert abs(sv - pv) < 1e-3, (
        "证书与生产面须同值（修前 41.0 vs 57.57，可行方向判决不健全）："
        "cert=%r prod=%r" % (sv, pv))
    # 正名 square 不受影响
    sv2 = float(_PC.support_vec("square", 40, 1, 1, math.sqrt(2)))
    assert abs(sv2 - 57.5685424949238) < 1e-9


# ---------------------------------------------------------------- #2132

def test_2132_nonfinite_input_fails_closed():
    from lingshu.nn import hex_ortho as _HO
    lat = np.zeros((24, 24, 3), dtype=float)
    lat[12, 12, 0] = 1.0
    lat[5, 5, 0] = float("nan")
    r = _HO.residual_pyramid(lat, [np.full(7, 0.3), np.full(7, -0.2)])
    assert r["stop_reason"] == "invalid_input", (
        "NaN 晶格不得误标良性 card_exhausted（修前形态）：%r" % (r.get("stop_reason"),))
    assert r["residual_share"] is None
    r2 = _HO.residual_pyramid(np.ones((24, 24, 3)) * 0.1,
                              [np.full(7, 0.3), np.full(7, float("nan"))])
    assert r2["stop_reason"] == "invalid_input", "NaN 核同样拦截"


def test_2132_clean_input_semantic_stops_intact():
    from lingshu.nn import hex_ortho as _HO
    r = _HO.residual_pyramid(np.ones((24, 24, 3)) * 0.1,
                             [np.full(7, 0.3), np.full(7, -0.2)])
    assert r["stop_reason"] in ("dead_zone", "orthogonal_exhausted",
                                "card_exhausted", "max_layers")
    assert r["stop_reason"] == "dead_zone", (
        "干净输入的正常语义停因不得被新闸改变，实测 %r" % (r.get("stop_reason"),))
    assert math.isfinite(float(r["residual_share"])), "报告值域可审计"


def test_2132_gram_schmidt_nan_not_silent_duplicate():
    from lingshu.nn import hex_ortho as _HO
    k = np.full(7, float("nan"))
    r = _HO.gram_schmidt_kernels([k])
    entry = r["report"][0] if isinstance(r, dict) and "report" in r else \
        (r[0] if isinstance(r, list) else None)
    assert entry is not None, "返回形态未识别 %r" % (type(r),)
    if isinstance(entry, dict) and "ratio" in entry:
        assert entry["ratio"] is None or not entry.get("redundant"), (
            "NaN 核不得被静默判成「完全重复解释」：%r" % (entry,))
