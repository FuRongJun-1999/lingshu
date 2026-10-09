# -*- coding: utf-8 -*-
"""query_match 顺序无关性回归。

_match_score 原为贪心 first-fit（取第一个匹配的未用检测即 break），当同类型 conn
的匹配集有包含关系（通配 zone="*" ⊃ 具体 zone="top"）时，通配 conn 会抢占具体 conn
需要的检测，命中数依赖 detections 顺序 → verdict 在 ACCEPT/DEFER 间翻转。修复后
用增广路求最大匹配，命中数与 detections 顺序无关。
"""

import itertools


def _lib():
    from lingshu.nn.hex_query import ConnectionLibrary
    lib = ConnectionLibrary()
    # 通配 conn ⊃ 具体 conn，制造贪心抢占
    lib.add_record("gizmo", [
        {"type": "circle", "zone": "*", "color": "*"},
        {"type": "circle", "zone": "top", "color": "*"},
    ])
    lib.add_record("dummy", [{"type": "stripe", "zone": "*", "color": "*"}])
    return lib


def test_match_score_order_independent():
    """同 detections 集合打乱顺序 → _match_score 必须相同（修复前 0.5 vs 1.0）。"""
    from lingshu.nn.hex_query import _match_score

    rec = _lib().records[0]
    dets = [
        {"obj": "circle|red", "pos": "r0"},   # r0 ∈ top
        {"obj": "circle|red", "pos": "r7"},   # r7 ∈ bottom(非 top)
    ]
    scores = {_match_score(rec, list(p)) for p in itertools.permutations(dets)}
    assert len(scores) == 1, f"_match_score 应与 detections 顺序无关，实际得到 {scores}"


def test_query_match_verdict_stable_across_orders():
    """query_match 的 verdict 应对 detections 顺序稳定（修复前 DEFER/ACCEPT 翻转）。"""
    from lingshu.nn.hex_query import query_match

    lib = _lib()
    dets = [
        {"obj": "circle|red", "pos": "r0"},
        {"obj": "circle|red", "pos": "r7"},
    ]
    verdicts = {query_match(list(p), lib)[0]["verdict"]
                for p in itertools.permutations(dets)}
    assert len(verdicts) == 1, f"verdict 应稳定，实际 {verdicts}"


def test_max_matching_optimal_hit_count():
    """贪心可能漏命中；最大匹配应取到最优（此处 2 conn 都应被满足 → score=1.0）。"""
    from lingshu.nn.hex_query import _match_score

    rec = _lib().records[0]
    # 通配 conn 可匹配任一 circle，具体 top conn 匹配 r0；两者都能被满足
    dets = [
        {"obj": "circle|red", "pos": "r0"},   # top
        {"obj": "circle|red", "pos": "r7"},   # bottom
    ]
    # 最大匹配：通配→r7，具体top→r0，两 conn 都命中
    assert _match_score(rec, dets) == 1.0, "最大匹配应把两个 conn 都匹配到（score=1.0）"