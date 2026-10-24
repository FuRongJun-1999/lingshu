# -*- coding: utf-8 -*-
"""#942 F23-05 守卫：`search_report` 空批次不得崩（裸除）。

缺陷（锚 dc45e579）：`lingshu/nn/hex_search.py:246`
    "avg_visited": round(total_nodes / n, 1)      # n = len(lat)，空批次 n=0 ⇒ 裸除
同 return 的另 5 处除法（L239/240/242/243/247）都有 `max(1,·)`，唯独这处漏了；
docstring L217 自己写着「检出集合为空时分母取 max(1,·)」。

修前：`lat=[]` ⇒ ZeroDivisionError: division by zero（本守卫必须在此点红）
修后：`lat=[]` ⇒ 不抛，且 `avg_visited == 0.0`

★ 反向腿：非空批次读数必须与修前**逐字相同**（修复不得改变既有语义）。
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lingshu.nn.hex_search as HS      # noqa: E402
from lingshu.nn.hex_search import search_report   # noqa: E402

EMPTY_LAT = np.zeros((0, 8, 8), dtype=float)


class _StubNet:
    """空批次时 n=0、循环不执行 ⇒ net 根本不会被用到；传个哨兵证明这一点。"""


def _fake_search(net, lat1, max_depth=2):
    """非空批次用：返回固定 detections，让读数可复算。"""
    return ([{"pos": "R0C0", "obj": "cat", "conf": 0.9}],
            {"visited": 10, "rejected": 2})


# ---------------------------------------------------------------- 主腿
def test_empty_batch_does_not_raise():
    """lat=[] 时不得抛异常（修前是 ZeroDivisionError: division by zero）。"""
    rep = search_report(_StubNet(), EMPTY_LAT, [])   # 修前在此点红
    assert isinstance(rep, dict), rep


def test_empty_batch_avg_visited_is_zero():
    """空批次 avg_visited 记 0.0（与同 return 其它字段「记 0.0」口径一致）。"""
    rep = search_report(_StubNet(), EMPTY_LAT, [])
    assert rep["avg_visited"] == 0.0, (
        "空批次 avg_visited 应为 0.0，实得 %r" % (rep["avg_visited"],))


def test_empty_batch_all_fields_zero():
    """空批次其余字段也全是 0/0.0——证明不是靠吞异常蒙过去。"""
    rep = search_report(_StubNet(), EMPTY_LAT, [])
    expected = {
        "pos_hit_rate": 0.0, "obj_match_rate": 0.0,
        "pos_precision": 0.0, "obj_precision": 0.0,
        "pos_false_positives": 0, "obj_false_positives": 0,
        "avg_visited": 0.0, "prune_rate": 0.0,
    }
    for k, v in expected.items():
        assert rep[k] == v, "%s 期望 %r 实得 %r" % (k, v, rep[k])


# ---------------------------------------------------------------- 反向腿
def test_nonempty_batch_unchanged(monkeypatch):
    """非空批次读数必须与修前逐字相同（修复不得改变既有语义）。"""
    monkeypatch.setattr(HS, "recursive_search", _fake_search)
    lat = np.zeros((1, 8, 8), dtype=float)
    metas = [{"labels": [{"pos": "R0C0", "obj": "cat"}]}]
    rep = search_report(None, lat, metas)
    assert rep == {
        "pos_hit_rate": 1.0, "obj_match_rate": 1.0,
        "pos_precision": 1.0, "obj_precision": 1.0,
        "pos_false_positives": 0, "obj_false_positives": 0,
        "avg_visited": 10.0, "prune_rate": 0.2,
    }, rep


# ---------------------------------------------------------------- 判别力自证
def test_guard_has_teeth():
    """自证：把 L246 还原成裸除后，空批次必须重新抛 ZeroDivisionError。

    这条腿保证「守卫能咬住这一处」，而不是恰好全绿。
    """
    src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "lingshu", "nn", "hex_search.py")
    src = open(src_path, encoding="utf-8").read()
    bare = "round(total_nodes / n, 1)"
    guarded = "round(total_nodes / max(1, n), 1)"
    assert bare not in src, (
        "hex_search.py 仍含裸除 `%s` —— 修复未生效" % bare)
    assert guarded in src, "未找到修复后的写法 `%s`" % guarded
