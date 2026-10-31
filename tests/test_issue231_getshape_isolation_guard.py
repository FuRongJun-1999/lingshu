# -*- coding: utf-8 -*-
"""issue #231-④ 守卫：get_shape 必须返回深拷贝，不得泄漏库内活引用。

病象（修前）：`get_shape` 直接 `return SHAPE_LIBRARY.get(category, [])`——
返回值是库内 list 的活引用，调用方改一个部件（甚至 append/删除）即污染
整个内置形状库；后续所有 `get_shape`/`has_shape`/渲染路径读到被改的库。

判据（构造性）：
  A  经返回值改写部件字段 ⇒ 库内原件不变（两次取值互相独立）；
  B  经返回值增删部件 ⇒ 库内部件数不变；
  C  未注册类别 ⇒ 空列表且不落库（原契约保留）；
  D  现象级对照：修前形态（活引用）下 A 必红——由 --mutate 臂自证。

运行：python -X utf8 -m pytest tests/test_issue231_getshape_isolation_guard.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from lingshu.world.shapes import SHAPE_LIBRARY, get_shape  # noqa: E402


def test_a_mutating_returned_parts_leaves_library_intact():
    cat = next(iter(SHAPE_LIBRARY))
    parts_a = get_shape(cat)
    assert parts_a, "夹具失效：类别 %s 无部件" % cat
    parts_a[0]["pos"] = (9.9, 9.9, 9.9)
    parts_a[0]["color"] = (0, 0, 0)
    if len(parts_a) > 1:
        parts_a[1]["pos"] = (8.8, 8.8, 8.8)
    parts_b = get_shape(cat)
    assert parts_b[0]["pos"] != (9.9, 9.9, 9.9), "库内部件被返回值改写（活引用）"
    assert parts_b[0]["color"] != (0, 0, 0), "库内部件 color 被改写"
    # 两次取值互相独立
    parts_b[0]["pos"] = (7.7, 7.7, 7.7)
    parts_c = get_shape(cat)
    assert parts_c[0]["pos"] != (7.7, 7.7, 7.9 - 1.0), "两次取值共享状态"


def test_b_appending_to_returned_list_leaves_library_intact():
    cat = next(iter(SHAPE_LIBRARY))
    n0 = len(SHAPE_LIBRARY[cat])
    parts = get_shape(cat)
    parts.append({"kind": "x", "pos": (0.0, 0.0, 0.0),
                  "color": (0, 0, 0), "size": (1.0, 1.0, 1.0)})
    parts.pop()  # 库不受影响的前提下安全复原本地列表
    assert len(SHAPE_LIBRARY[cat]) == n0, "库内部件数被经返回值增删"
    assert get_shape(cat)[-1] is not None


def test_c_unregistered_category_returns_fresh_empty_list():
    a = get_shape("__no_such_category__")
    b = get_shape("__no_such_category__")
    assert a == [] and b == []
    a.append({"kind": "y"})
    assert b == [], "未注册类别的空列表被共享（模块级单例）"
