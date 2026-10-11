# -*- coding: utf-8 -*-
"""守卫 · `lingshu/gen/hexgen_self_source.py` 三件 P1 缺陷（#5 / #218 / #391）
====================================================================================
本文件只钉「缺陷不再存在」，不是「代码能跑」。三条各自带**变异可判红**的判据：

#5（部分成立）· 自家 v1/v2 词表名 `stripe`（＝方块）走 v3 侧入口 `KeyError`
    判据来源：`hexgen_align_probe.py:16-17`／`:39`（`NAME_MAP_SELF2REAL`，
    R297 已登记「本词表 `stripe`＝方块」的比较口径）＋ `hexgen_self_source.py:42-51`
    （`SELF_TUPLES` 用了 `stripe`）与 `:1128-1138`（`enumerate_tuples` 形状取自
    `hex_composite.SHAPES`，含 `stripe`），而 `EXTENT_FRAC` 按真实词表 `SHAPES8` 建键。
    断言：`stripe` 与 `square` 在 `paint_extent`/`paint_support`/`shape_mask_v3` 上**逐位一致**，
    且 `place_objects`/`render_prompt_v3` 以 `stripe` 入参不再抛 `KeyError`。
    抽掉修复（去掉 `_v3_shape` 归一）⇒ 本组首条即 `KeyError`。

#218（成立）· `_solve_centroid._anchor` 把 `legal_win` 的 (x0,x1,y0,y1) 按 (x0,y0,x1,y1) 解包
    判据来源：`hexgen_self_source.py:813-816` `legal_win` docstring 明写返回 (x0,x1,y0,y1)；
    本文件 :1064-1066（R311 判据「只要存在合法摆位就必须达到标称半径」）。
    断言：**标称半径下的合法窗口非空**（⇒ 该题面在标称半径上确有合法摆位，构造性存在性）
    ⇒ `_solve_centroid` 必须返回标称半径。抽掉修复 ⇒ 66 例掉到 35（见变异读数）。

#391（部分成立）· `split_self` 按「件」交替 ⇒ repeats≥2 时留出集是标定集的逐字节副本
    判据来源：`hexgen_self_source.py:1177-1187`（`build_enum_items` 同配方多 rep）与
    自家渲染器的**确定性**（`:73-90` docstring「同样入参 ⇒ 同样字节」）。
    断言：以配方（sid）为切分单位 ⇒ 两集 sid 不相交、同一 sid 的重复件不跨集、
    **两集之间无逐字节相同的图**。抽掉修复 ⇒ 该断言在 24 例上判红。

运行（lingshu 仓根）：python -X utf8 -m pytest tests/test_hexgen_self_source_guards.py -q --no-header
"""
import hashlib

import numpy as np
import pytest

from lingshu.gen import hexgen_self_source as S

RAD = 51
SIZE = 512

#   非对称切点（真实冻结切点两轴都偏；这批切点让 `judged_rect` 中心出画布内缩 ⇒ 触发 `_anchor`）
CUT_PAIRS = [
    ([0.15, 0.85], [0.2, 0.3]),
    ([0.2, 0.3], [0.15, 0.85]),
    ([0.1, 0.9], [0.2, 0.3]),
    ([0.3, 0.7], [0.2, 0.3]),
    ([0.5236, 0.6484], [0.264, 0.6234]),      # docstring :796-797 的实测冻结切点
]
CELLS = [f"r{i}" for i in range(9)]


def _zc(rc, cc):
    return {"row": {"cuts": list(rc)}, "col": {"cuts": list(cc)}}


# ==================== #5 · stripe 词表归一 ====================

def test_self_vocab_shape_names_are_all_known_to_extent_frac():
    """自家词表（SELF_TUPLES / enumerate_tuples / hex_composite.SHAPES）里的每个形状名，
    经 `_v3_shape` 归一后都必须能查到 `EXTENT_FRAC`（否则 `paint_extent` 抛 KeyError）。"""
    from lingshu.gen.hex_composite import SHAPES as SELF_SHAPES
    names = set(SELF_SHAPES) | {t[1] for t in S.SELF_TUPLES} | {t[1] for t in S.enumerate_tuples()}
    names |= set(S.SHAPES8)
    for sh in sorted(names):
        key = S._v3_shape(sh)
        assert key in S.EXTENT_FRAC, f"{sh!r}→{key!r} 不在 EXTENT_FRAC"
        assert S.paint_extent(sh, RAD) > 0


def test_stripe_is_the_same_primitive_as_square():
    """自家名 `stripe` ≡ 真实名 `square`（R297 已登记命名差）⇒ 三条几何函数逐位一致。"""
    assert S.paint_extent("stripe", RAD) == S.paint_extent("square", RAD)
    for ux, uy in ((1.0, 0.0), (0.0, 1.0), (0.7071, 0.7071)):
        assert S.paint_support("stripe", RAD, ux, uy) == S.paint_support("square", RAD, ux, uy)
    a = S.shape_mask_v3("stripe", 128, 64, 64, 20)
    b = S.shape_mask_v3("square", 128, 64, 64, 20)
    assert np.array_equal(a, b)


def test_stripe_shape_reaches_placement_and_prompt_rendering():
    """以 `stripe` 入参走生产入口不再抛 KeyError，且真的画出非背景像素。"""
    pts, r = S.place_objects(("r4",), 1, RAD, SIZE, shape="stripe")
    assert r == RAD and len(pts) == 1
    img = S.render_prompt_v3((3, "stripe", "blue", "dotted", ("r0", "r4", "r8")),
                             size=128, pal={"blue": (40, 70, 220)})
    assert (img < 250).any(), "stripe 件渲染后不应是纯白底"


def test_every_self_tuple_renders():
    """SELF_TUPLES 与 enumerate_tuples 的每条配方都必须能渲染（stripe 在内）。"""
    for tup in S.SELF_TUPLES:
        assert S.render_self(tup).shape == (SIZE, SIZE, 3)
    for tup in S.enumerate_tuples():
        S.render_self(tup)


# ==================== #218 · legal_win 解包序 ====================

@pytest.mark.parametrize("rc,cc", CUT_PAIRS)
def test_nominal_radius_reachable_when_legal_window_nonempty(rc, cc):
    """标称半径下合法窗口非空 ⇒ 存在性成立 ⇒ 求解器必须达到标称半径。

    （窗口非空即构造性见证：`legal_win` 已把画布内缩写进两端，故窗口内任一点都合法。）
    缺陷形态（`_anchor` 轴对调）会把起点推到非法格 ⇒ 下降卡死 ⇒ 半径掉到 35。
    """
    zc = _zc(rc, cc)
    checked = 0
    for c in CELLS:
        w = S.legal_win(c, "circle", RAD, SIZE, zc)
        if not (w[0] <= w[1] and w[2] <= w[3]):
            continue                                   # 该格标称半径放不下：不构成存在性
        checked += 1
        r, pts = S._solve_centroid((c,), 1, RAD, SIZE, "circle", zc)
        assert r == RAD, f"cell={c} cuts={rc}/{cc}：窗口非空却只解出 r={r}"
        # 位置也必须落在该格（判决量口径）
        assert S._cell_of_pt(pts[0][0], pts[0][1], SIZE, zc) == c
    assert checked > 0, "本组切点未覆盖任何非空窗口，判据空转"


def test_anchor_window_axis_order_is_not_swapped():
    """锚点必须落在 `legal_win` 内：取一个 `judged_rect` 中心出画布、被迫走 `_anchor` 的题面，
    标称半径下返回点须在窗口内（轴对调会把 x 送进 y 区间）。"""
    rc, cc = [0.15, 0.85], [0.2, 0.3]
    zc = _zc(rc, cc)
    w = S.legal_win("r0", "circle", RAD, SIZE, zc)
    r, pts = S._solve_centroid(("r0",), 1, RAD, SIZE, "circle", zc)
    x, y = pts[0]
    assert w[0] <= x <= w[1] and w[2] <= y <= w[3], f"({x},{y}) 不在 legal_win={w} 内"
    assert r == RAD


# ==================== #391 · 标定/留出无副本泄漏 ====================

def test_split_self_keeps_replicas_of_a_recipe_on_one_side():
    """`repeats≥2` 时同一配方的重复件必须整组进同一侧（切分单位＝配方 sid）。"""
    items = S.build_enum_items(repeats=2)
    cal, hold = S.split_self(items)
    side = {}
    for it in cal:
        side.setdefault(it["sid"], set()).add("cal")
    for it in hold:
        side.setdefault(it["sid"], set()).add("hold")
    straddle = {sid: s for sid, s in side.items() if len(s) > 1}
    assert not straddle, f"同一配方的重复件跨集：{straddle}"
    assert {it["sid"] for it in cal}.isdisjoint({it["sid"] for it in hold})


def test_split_self_has_no_byte_identical_leak_across_sides():
    """缺陷形态下（按件交替）留出集是标定集的逐字节副本 ⇒ 两集之间不得有相同的图。"""
    items = S.build_enum_items(repeats=2)
    cal, hold = S.split_self(items)

    def sig(it):
        return hashlib.sha256(np.ascontiguousarray(it["img"]).tobytes()).hexdigest()

    cal_sigs = {sig(it) for it in cal}
    shared = [it["sid"] for it in hold if sig(it) in cal_sigs]
    assert not shared, f"留出集有 {len(shared)} 件与标定集逐字节相同（评估泄漏）：{shared[:6]}"


def test_split_self_single_repeat_still_splits_both_sides():
    """`repeats=1` 的既有行为不回退：两集都非空、互不相交、且与旧口径「按 id 升序交替」逐位一致。"""
    items = S.build_enum_items(repeats=1)
    cal, hold = S.split_self(items)
    assert cal and hold
    assert {it["id"] for it in cal}.isdisjoint({it["id"] for it in hold})
    by_shape = {}
    for it in items:
        by_shape.setdefault(it["shape"], []).append(it)
    old_c, old_h = [], []
    for sh in sorted(by_shape):
        rows = sorted(by_shape[sh], key=lambda x: x["id"])
        old_c += rows[0::2]
        old_h += rows[1::2]
    assert [it["id"] for it in cal] == [it["id"] for it in old_c]
    assert [it["id"] for it in hold] == [it["id"] for it in old_h]
