# -*- coding: utf-8 -*-
"""test_mesh_gen_selfmodel_guard · 世界模型「自生成建模」守卫（mesh_gen）
============================================================================
缺陷（这批要防的是什么）
------------------------
用户口径：「Blender 只是建模导入，最终要靠**世界模型自己生成建模**」。
本轮之前世界模型**没有"语义 → 网格"的生成面**：`shapes.py` 只会摆 4 种图元
（box/sphere/pillar/pyramid），椅子的腿是方块、树冠是多边形近似的球；而
Blender 侧真值是多面数网格（wmtest 三件实测 2028 / 344 / 2044 面）。
`mesh_gen.py` 补上生成端（图元 7 种 + 放样/车削/扫掠/并置算子 + 12 个语义配方
+ 与真值对拍的度量）。本件守它的**静默错**——生成端错了最坏情况是"给出一张
看起来是模型、其实算错的网格"，不报错、不崩，只是悄悄错。

修法（本件钉住的 8 处）
----------------------
1. `ensure_outward`：算子内部只保证"同网格各面**同向**"，全局符号按**有符号
   体积**定（体积<0 ⇒ 全部反向）。不这么做就得手推每个算子的环绕向（球极点 /
   放样端盖 / 旋转体端盖 / 扫掠标架各不同），极容易错——而**面朝外是背面剔除
   的硬前提**，反了模型会渲成内翻空壳。
2. `_revolve_core` 的退化闸：剖面两端都落在轴上（全 r=0）⇒ 必须 raise，不许
   返回一条 0 面的"空网格"。
3. `loft_layers` 的等点数闸：截面点数不等必须 raise（旧形态会索引越界成
   IndexError，或静默错配）。
4. `triangulate_polygon`：**耳切**替代扇形三角化。凹多边形上"从 0 号点拉扇"
   会切出多边形**之外**的三角形（面积对不上、渲染多一块），且完全静默。
5. `soup_bounds`：旧写法把"第 k 个顶点的三轴极值"当成"第 k 轴极值" ⇒ **一切
   AABB 退化成立方体**，连锁导致 `transform_to_bbox` 把模型拉成立方、体素/剪影
   栅格按立方分配 ⇒ 相似度读数全部失真（实测：arm 自生成 bbox 报
   [1.529,1.529,1.529]，逐轴实读应为 [1.2292,0.6727,1.529]）。
6. `part_to_mesh`：未知 kind 必须 raise，**不许退化成盒子**——静默退化正是
   "只能呈现方块"的病根。
7. `_face_idx`：度量/自证函数同时吃 `MeshData`（裸索引三元组）与
   `mesh3d.MeshGeometry`（`Face.idx` 对象）⇒ 自生成网格与载入的真值能同尺对拍。
8. `to_obj_text` 的 `# normalize:` 头：`mesh3d.parse_obj` 是 **fail-closed**
   的（解析不到 normalize 就 raise）⇒ 生成物不带头就**读不回来**，往返自证失效。

判据来源
--------
· 闭合/定向：有向边恰出现一次 + 反向边存在（流形无边界）；外法向 = 有符号体积 > 0
  （散度定理）——生成端可机械判定的最小充分条件。
· 背面剔除判据：`mesh3d.prepare_triangles` 用世界空间 `normal·(eye−face_center)>0`，
  故"面朝外"是硬前提；world3d 图元通道同款判据（#311）。
· 凹多边形三角化：耳切定理（简单多边形必可耳切；面积守恒是必要条件）。
· 证据读数见 `_evidence/wm-selfmodel/`（compare_truth.json / res_sweep.json /
  sheet_*.png / hero_scene.png）。

断言组（抽掉修复 ⇒ 哪组红）
--------------------------
  A 组 闭合 + 外法向 + 尺寸口径 ── 抽掉 `ensure_outward` ⇒ **A2 红**
     （实测：`link_between` 与 `sweep_path` 的自然环绕向是反的，stats 里
      `flipped_outward=1`；不翻面时 signed_volume 为负）
  B 组 车削(revolve) 退化闸 ────── 抽掉"全极点"闸 ⇒ **B1 红**（返回 0 面空网格）
  C 组 放样/扫掠入参闸 ─────────── 抽掉等点数闸 ⇒ **C1 红**（IndexError 而非 ValueError）
  D 组 凹多边形耳切 ────────────── 换回扇形三角化 ⇒ **D1/D2/D3 红**
     （实测 U 形：耳切面积 7.0 = 多边形面积；扇形 11.0，多出 4.0）
  E 组 度量口径 ───────────────── 回退 `soup_bounds` 旧写法 ⇒ **E1/E2/E3 红**
     （实测：非立方用例 dims 由 (2,1,0) 变 (0,2,1)；arm 生成 bbox 由
      [1.2292,0.6727,1.529] 变 [1.529,1.529,1.529]）
  F 组 形状库向后兼容 + 新图元校验 ── 把新 kind 从 PART_KINDS 去掉 ⇒ **F3 红**
  G 组 部件组合 → 网格 ─────────── 让 `part_to_mesh` 对未知 kind 回退 box ⇒ **F5/G 红**
  H 组 OBJ 往返 ───────────────── 去掉 `# normalize:` 头 ⇒ **H1/H2 红**（载入 raise）
  I 组 确定性/无副作用 ────────── 让 build_mesh 写全局 MESH_LIBRARY ⇒ **I2 红**
  J 组 mesh3d/world3d 互操作 ──── 法线用错环绕向 ⇒ **J2 红**；渲染不出 ⇒ **J3 红**

运行命令
--------
（系统 python3 未装 pytest；本机 pytest 9.1.1 在 _pr-prep venv 内）
    cd <repo>  # wm-verify/lingshu
    /home/sixtyseven67/DSH/测试/测试/_pr-prep/.venv/bin/python -m pytest \
        tests/test_mesh_gen_selfmodel_guard.py -q --no-header
脚本模式（不依赖 pytest，本文件自带 _main）：
    python3 -X utf8 tests/test_mesh_gen_selfmodel_guard.py
"""
from __future__ import annotations

import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lingshu.world import mesh_gen as MG                          # noqa: E402
from lingshu.world import shapes as SH                             # noqa: E402


# --------------------------------------------------------------------------- #
# 夹具
# --------------------------------------------------------------------------- #
def _primitives():
    """每次现构造（不放模块级全局，避免测试之间互相污染）。"""
    return {
        "box": MG.prim_box((1.0, 2.0, 3.0)),
        "sphere": MG.prim_sphere((1.0, 1.0, 1.0), seg_u=16, seg_v=10),
        "cylinder": MG.prim_cylinder((1.0, 2.0, 1.0), seg=20),
        "cone": MG.prim_cone((1.0, 2.0, 1.0), top=0.0, seg=20),
        "frustum": MG.prim_cone((1.0, 2.0, 1.0), top=0.5, seg=20),
        "disk": MG.prim_disk((1.0, 0.2, 1.0), seg=20),
        "torus": MG.prim_torus((1.0, 0.25, 1.0), seg_u=20, seg_v=10),
        "pyramid": MG.prim_pyramid((1.0, 1.0, 2.0)),
    }


def _operators():
    return {
        "revolve": MG.revolve([(0.0, 0.0), (0.5, 0.0), (0.5, 0.4), (0.3, 0.8),
                               (0.0, 1.0)], seg=20),
        "loft": MG.loft_rects([(0.0, 1.0, 1.0), (0.5, 1.4, 1.2, 0.1, 0.0),
                               (1.0, 0.8, 0.6)]),
        "link": MG.link_between((0.0, 0.0, 0.0), (0.5, 1.0, 0.25), 0.2, 0.15),
        "sweep": MG.sweep_path([(0.0, 0.0, 0.0), (0.3, 0.5, 0.0),
                                (0.7, 0.6, 0.2)], 0.1, 0.1),
    }


def _tmpdir():
    """脚本模式下的临时目录（刻意**不用 tempfile**：#156 的裸名白名单里没有它）。"""
    base = os.path.join(os.environ.get("TMPDIR") or "/tmp",
                        "meshgen_guard_%d" % os.getpid())
    os.makedirs(base, exist_ok=True)
    return base


def _tri_area(tri):
    (a, b, c) = tri
    u = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    v = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    cx = u[1] * v[2] - u[2] * v[1]
    cy = u[2] * v[0] - u[0] * v[2]
    cz = u[0] * v[1] - u[1] * v[0]
    return 0.5 * math.sqrt(cx * cx + cy * cy + cz * cz)


def _raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return True
    except BaseException as e:                                    # noqa: BLE001
        raise AssertionError("期望 %s，实际 %s: %s" % (exc.__name__,
                                                     type(e).__name__, e))
    return False


# =========================================================================== #
# A 组：闭合 + 外法向 + 尺寸口径
# =========================================================================== #
def test_A1_all_primitives_and_operators_watertight():
    """每个图元/算子都是闭合可定向流形（有向边恰出现一次 + 反向边存在）。"""
    bad = []
    for name, m in list(_primitives().items()) + list(_operators().items()):
        if not MG.is_watertight(m):
            bad.append(name)
    assert not bad, "非闭合（边界边/重复有向边）：%r" % (bad,)


def test_A2_all_primitives_and_operators_outward_positive_volume():
    """外法向判据：有符号体积 > 0。

    抽掉 `ensure_outward` ⇒ link / sweep 的自然环绕向是反的（stats 里
    flipped_outward 记录了这一点），体积变负 ⇒ 本组红。
    """
    bad = []
    for name, m in list(_primitives().items()) + list(_operators().items()):
        v = MG.signed_volume(m)
        if not v > 0.0:
            bad.append((name, round(v, 9)))
    assert not bad, "有符号体积 <= 0（面朝内，渲出来是内翻空壳）：%r" % (bad,)
    # 回退即红的实测：这两个算子**必须**靠 ensure_outward 翻正
    flipped = [n for n, m in _operators().items()
               if m.stats.get("flipped_outward")]
    assert set(flipped) == {"link", "sweep"}, (
        "ensure_outward 的生效面变了（实测 link/sweep 需要翻面）：%r" % (flipped,))


def test_A3_primitive_bbox_matches_declared_size():
    """图元的 AABB 必须等于声明尺寸（尺寸口径是"包围盒边长"，不是半径）。"""
    lo, hi = MG.mesh_bounds(MG.prim_box((1.0, 2.0, 3.0)))
    assert [round(hi[k] - lo[k], 9) for k in range(3)] == [1.0, 2.0, 3.0]
    for name, size in (("cylinder", (1.0, 2.0, 0.6)), ("cone", (1.0, 2.0, 0.6)),
                       ("disk", (1.0, 0.2, 1.0))):
        m = getattr(MG, "prim_" + name)(size, seg=24)
        lo, hi = MG.mesh_bounds(m)
        got = [hi[k] - lo[k] for k in range(3)]
        assert max(abs(got[k] - size[k]) for k in range(3)) < 1e-9, \
            "%s 的 bbox %r != 声明 %r" % (name, got, size)
    # 四棱锥的底面必须**恰为**声明的轴对齐矩形（不是内接菱形）
    p = MG.prim_pyramid((1.0, 1.0, 2.0))
    lo, hi = MG.mesh_bounds(p)
    got = [hi[k] - lo[k] for k in range(3)]
    assert max(abs(got[k] - (1.0, 1.0, 2.0)[k]) for k in range(3)) < 1e-9, got
    assert p.tri_count == 8, p.tri_count              # 4 侧面 + 底面 2 个三角
    # 球/环是分段多面体：三轴都不得超出声明（内切近似），且不得小于 0.9 倍
    for name, m in (("sphere", MG.prim_sphere((1.0, 1.0, 1.0), 24, 14)),
                    ("torus", MG.prim_torus((1.0, 0.25, 1.0), 32, 16))):
        lo, hi = MG.mesh_bounds(m)
        got = [hi[k] - lo[k] for k in range(3)]
        want = (1.0, 1.0, 1.0) if name == "sphere" else (1.0, 0.25, 1.0)
        for k in range(3):
            assert got[k] <= want[k] + 1e-9, "%s bbox 超出声明：%r" % (name, got)
            assert got[k] >= 0.9 * want[k], "%s 过分缩水：%r" % (name, got)


def test_A4_all_recipes_are_valid_solid_meshes():
    """12 个语义配方全部：面数够（多面数，不是 4 个盒子）、闭合、外法向。"""
    for cat in sorted(MG.MESH_RECIPES):
        m = MG.build_mesh(cat)
        assert m is not None, "配方 %r 建不出来" % cat
        assert m.tri_count >= 24, "配方 %r 面数 %d 太少（不像多面数建模）" \
            % (cat, m.tri_count)
        assert MG.is_watertight(m), "配方 %r 非闭合" % cat
        assert MG.signed_volume(m) > 0.0, "配方 %r 面朝内" % cat
        assert m.stats.get("degenerate_dropped", 0) == 0, \
            "配方 %r 产生退化面 %d 个" % (cat, m.stats.get("degenerate_dropped"))


def test_A5_no_zero_area_triangles_anywhere():
    """零面积三角形是"看着是面、其实占不到像素"的静默坑：一个都不许有。"""
    for group in (_primitives(), _operators()):
        for name, m in group.items():
            for tri in MG.tri_soup(m):
                assert _tri_area(tri) > 1e-12, "%s 存在零面积三角形" % name
    for cat in sorted(MG.MESH_RECIPES):
        m = MG.build_mesh(cat)
        for tri in MG.tri_soup(m):
            assert _tri_area(tri) > 1e-12, "配方 %s 存在零面积三角形" % cat


# =========================================================================== #
# B 组：车削 revolve 的退化闸
# =========================================================================== #
def test_B1_revolve_all_pole_profile_raises():
    """剖面两端都落在轴上 ⇒ 必须 raise（不许静默返回 0 面空网格）。"""
    assert _raises(ValueError, MG.revolve, [(0.0, 0.0), (0.0, 1.0)], seg=8)


def test_B2_revolve_negative_radius_raises():
    assert _raises(ValueError, MG.revolve, [(0.5, 0.0), (-0.5, 0.5)], seg=8)


def test_B3_revolve_duplicate_point_raises():
    assert _raises(ValueError, MG.revolve, [(0.5, 0.0), (0.5, 0.0), (0.0, 1.0)],
                   seg=8)


def test_B4_revolve_bad_seg_raises():
    assert _raises(ValueError, MG.revolve, [(0.0, 0.0), (0.5, 0.5)], seg=2)


def test_B5_revolve_hollow_profile_is_watertight_outward():
    """中空容器（外壁上行 → 过口沿 → 内壁下行；y 非单调）仍须闭合且外法向。

    这是"杯/瓶/碗"的表达基础：剖面折回时环绕向必须依旧一致，否则内壁会翻。
    """
    mug = MG.revolve([(0.0, 0.0), (0.2, 0.0), (0.21, 0.02), (0.22, 0.30),
                      (0.21, 0.32), (0.17, 0.32), (0.155, 0.30), (0.15, 0.05),
                      (0.0, 0.05)], seg=24)
    assert MG.is_watertight(mug), "中空杯非闭合（内壁翻面）"
    assert MG.signed_volume(mug) > 0.0, "中空杯面朝内"
    # 杯壁是壳：体积必须**明显小于**同外径实心柱（否则内壁没被"挖"出来），
    # 但也不能小到接近 0（那样是空壳/翻面）。实测：0.0260 vs 实心 0.0481 = 0.54。
    solid = MG.prim_cylinder((0.44, 0.32, 0.44), seg=24)
    ratio = MG.signed_volume(mug) / MG.signed_volume(solid)
    assert 0.15 < ratio < 0.75, \
        "中空杯体积比 %.3f 不在 (0.15, 0.75)：剖面折回段没形成内壁或成了空壳" % ratio


# =========================================================================== #
# C 组：放样 / 扫掠入参闸
# =========================================================================== #
def test_C1_loft_point_count_mismatch_raises():
    """截面点数不等 ⇒ ValueError（抽掉闸门会变成 IndexError/静默错配）。"""
    assert _raises(ValueError, MG.loft_layers,
                   [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
                    [(0.0, 0.0, 1.0), (1.0, 0.0, 1.0)]])


def test_C2_loft_y_must_increase():
    assert _raises(ValueError, MG.loft_rects,
                   [(0.0, 1.0, 1.0), (0.5, 1.0, 1.0), (0.4, 1.0, 1.0)])


def test_C3_loft_needs_two_sections():
    assert _raises(ValueError, MG.loft_rects, [(0.0, 1.0, 1.0)])


def test_C4_sweep_needs_two_path_points():
    assert _raises(ValueError, MG.sweep_path, [(0.0, 0.0, 0.0)], 0.1)


def test_C5_link_same_endpoints_raises():
    assert _raises(ValueError, MG.link_between, (0, 0, 0), (0, 0, 0), 0.1)


def test_C6_link_degenerate_frame_is_counted_not_silent():
    """轴向 ∥ 参考 up 时改用 X 轴参照，并把退化的次数**记进 stats**（不静默）。"""
    m = MG.link_between((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.2)
    assert m.stats.get("frame_degenerate") == 1
    assert MG.is_watertight(m) and MG.signed_volume(m) > 0.0


def _mean_edge(mesh):
    v = mesh.verts
    tot = n = 0.0
    for (a, b, c) in mesh.faces:
        for (i, j) in ((a, b), (b, c), (c, a)):
            p, q = v[i], v[j]
            tot += math.dist(p, q)
            n += 1
    return tot / n


def test_C7_loft_aligns_ring_starts_and_removes_twist():
    """放样起点对齐：同点数两环起点差半圈时，对齐必须让带更"不拧"。

    抽掉 `_align_layer_rings`（`align=False` 的形态）⇒ 本组红：环绕着拧过去，
    平均三角边长显著变大（实测 1.3359 vs 1.2147，本用例形状；不同环形差异更大）。
    """
    ring1 = [(math.cos(2 * math.pi * k / 6), 0.0, math.sin(2 * math.pi * k / 6))
             for k in range(6)]
    ring2 = [(math.cos(math.pi / 6 + 2 * math.pi * k / 6), 1.0,
              math.sin(math.pi / 6 + 2 * math.pi * k / 6)) for k in range(6)]
    raw = MG.loft_layers([ring1, ring2], align=False)
    ali = MG.loft_layers([ring1, ring2], align=True)
    assert ali.tri_count == raw.tri_count                 # 对齐只换起点，不改拓扑
    assert _mean_edge(ali) < _mean_edge(raw) - 1e-9, \
        "起点对齐没生效：%.4f vs %.4f" % (_mean_edge(ali), _mean_edge(raw))
    assert MG.is_watertight(ali) and MG.signed_volume(ali) > 0.0


def test_C8_ensure_outward_is_global_only_documented_boundary():
    """`ensure_outward` 只按**全局**有符号体积翻整体，**不修局部绕向不一致**。

    这是本件的**已知边界**（审计给出：打乱后 9 条有向边重复，ensure_outward
    不消除）：生成器的职责是"产出一致绕向"，`is_watertight` 是它的判据。
    """
    m = MG.prim_box((1.0, 1.0, 1.0))
    f = list(m.faces)
    f[3] = (f[3][0], f[3][2], f[3][1])                    # 单面反向
    bad = MG.MeshData(verts=m.verts, faces=f)
    assert not MG.is_watertight(bad), "局部反向没被检出"
    MG.ensure_outward(bad)
    assert not MG.is_watertight(bad), "ensure_outward 竟修好了局部不一致（口径变了）"


# =========================================================================== #
# D 组：凹多边形耳切（扇形三角化会切到多边形外）
# =========================================================================== #
#: U 形（8 点）：扇形从 0 号点拉扇会横跨中间那道凹槽 ⇒ 面积虚增 4.0
U_SHAPE = [(0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (3.0, 3.0, 0.0), (2.0, 3.0, 0.0),
           (2.0, 1.0, 0.0), (1.0, 1.0, 0.0), (1.0, 3.0, 0.0), (0.0, 3.0, 0.0)]


def test_D1_earclip_triangle_area_equals_polygon_area():
    """耳切三角化的面积和必须等于多边形面积（U 形 = 7.0）。"""
    tris = MG.triangulate_polygon(U_SHAPE)
    assert len(tris) == len(U_SHAPE) - 2
    total = sum(_tri_area([U_SHAPE[i] for i in t]) for t in tris)
    assert abs(total - MG.polygon_area(U_SHAPE)) < 1e-9
    assert abs(MG.polygon_area(U_SHAPE) - 7.0) < 1e-9


def test_D2_fan_triangulation_would_be_wrong():
    """回退即红的**对照读数**：扇形三角化在同一 U 形上面积 = 11.0（多 4.0）。"""
    fan = [(0, k, k + 1) for k in range(1, len(U_SHAPE) - 1)]
    total = sum(_tri_area([U_SHAPE[i] for i in t]) for t in fan)
    assert abs(total - 11.0) < 1e-9, "扇形三角化的对照读数变了：%r" % total
    assert abs(total - MG.polygon_area(U_SHAPE)) > 3.9, \
        "若两者相等 ⇒ 本判据失去区分力（扇形不再错，或耳切也错）"


def test_D3_add_poly_uses_earclip_not_fan():
    """`MeshData.add_poly` 必须走耳切（面积=7.0；扇形会给 11.0）。"""
    m = MG.MeshData(verts=U_SHAPE, faces=[tuple(range(len(U_SHAPE)))])
    total = sum(_tri_area([U_SHAPE[i] for i in t]) for t in m.faces)
    assert abs(total - 7.0) < 1e-9, "add_poly 用了扇形（面积 %r）" % total
    assert len(m.faces) == 6


def test_D4_degenerate_polygon_raises():
    """自交/共线多边形找不到耳 ⇒ raise，不许静默给一张错网格。"""
    assert _raises(ValueError, MG.triangulate_polygon,
                   [(0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0)])
    assert _raises(ValueError, MG.triangulate_polygon,
                   [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0),
                    (3.0, 0.0, 0.0)])
    assert _raises(ValueError, MG.triangulate_polygon, [(0.0, 0.0, 0.0)])


def test_D5_earclip_winding_follows_input_order():
    """环绕向口径：输入顺时针 ⇒ 输出三角形也是顺时针（放样端盖靠它保闭合）。"""
    ccw = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
    cw = list(reversed(ccw))
    for poly, sign in ((ccw, 1.0), (cw, -1.0)):
        for (a, b, c) in MG.triangulate_polygon(poly):
            u = (poly[b][0] - poly[a][0], poly[b][1] - poly[a][1])
            v = (poly[c][0] - poly[a][0], poly[c][1] - poly[a][1])
            assert (u[0] * v[1] - u[1] * v[0]) * sign > 0.0


# =========================================================================== #
# E 组：度量口径（对拍读数的可信度前提）
# =========================================================================== #
def test_E1_soup_bounds_does_not_collapse_axes():
    """非立方三角汤的 AABB 必须逐轴正确（回退旧写法 ⇒ 得到 (0,2,1)）。"""
    soup = [((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 1.0, 0.0))]
    lo, hi = MG.soup_bounds(soup)
    assert lo == (0.0, 0.0, 0.0) and hi == (2.0, 1.0, 0.0), (lo, hi)
    assert MG.bbox_dims(soup) == (2.0, 1.0, 0.0), MG.bbox_dims(soup)


def test_E2_soup_bounds_matches_mesh3d_bounds():
    """与 `MeshGeometry.bounds()` 逐位一致（跨来源同尺对拍的前提）。"""
    from lingshu.world.mesh3d import load_mtl, load_obj_mesh, MESH_LIBRARY
    path = "/mnt/d/DSH/blender/out/wmtest/arm/mesh.obj"
    if not os.path.isfile(path):
        return                                    # 真值不在本机 ⇒ 不假绿（跳过式返回）
    mtl = path[:-4] + ".mtl"
    load_obj_mesh(path, "__guard_arm__",
                  load_mtl(mtl) if os.path.isfile(mtl) else None, axis_swap=True)
    mesh = MESH_LIBRARY.pop("__guard_arm__")
    lo, hi = mesh.bounds()
    slo, shi = MG.soup_bounds(MG.tri_soup(mesh))
    assert max(abs(lo[k] - slo[k]) for k in range(3)) < 1e-12
    assert max(abs(hi[k] - shi[k]) for k in range(3)) < 1e-12
    dims = MG.bbox_dims(MG.tri_soup(mesh))
    assert abs(dims[0] - 1.2292) < 1e-3 and abs(dims[2] - 1.5290) < 1e-3, dims
    assert abs(dims[0] - dims[2]) > 0.2, "三轴被压成同一值（旧 bug 的形态）：%r" % (dims,)


def test_E3_transform_to_bbox_is_per_axis():
    """按观测 bbox 定比例必须**逐轴**精确（回退 ⇒ 被拉成立方）。"""
    m = MG.prim_box((1.0, 1.0, 1.0))
    out = MG.transform_to_bbox(m, (2.0, 0.5, 3.0))
    lo, hi = MG.mesh_bounds(out)
    got = [hi[k] - lo[k] for k in range(3)]
    assert max(abs(got[k] - (2.0, 0.5, 3.0)[k]) for k in range(3)) < 1e-9, got
    assert _raises(ValueError, MG.transform_to_bbox, MG.prim_box((1, 1, 1)),
                   (2.0, 0.0, 3.0))


def test_E4_similarity_is_translation_invariant():
    a = MG.prim_box((1.0, 2.0, 1.0))
    b = MG.translate_mesh(MG.prim_box((1.0, 2.0, 1.0)), (7.0, -3.0, 2.5))
    sim = MG.shape_similarity(MG.tri_soup(a), MG.tri_soup(b), res=16)
    assert sim["voxel_iou"] > 0.98, "平移后不相似（度量在比位姿，不是在比形状）：%r" % sim
    assert sim["dims_ratio"] == [1.0, 1.0, 1.0], sim["dims_ratio"]


def test_E5_similarity_self_is_one():
    m = MG.build_mesh("vase")
    sim = MG.shape_similarity(MG.tri_soup(m), MG.tri_soup(m), res=24)
    assert sim["voxel_iou"] > 0.999 and abs(sim["score"] - 1.0) < 0.01, sim


def test_E6_voxel_iou_on_disjoint_shapes_is_zero():
    a = MG.prim_box((0.5, 0.5, 0.5))
    b = MG.translate_mesh(MG.prim_box((0.5, 0.5, 0.5)), (10.0, 0.0, 0.0))
    assert MG.voxel_iou(MG.tri_soup(a), MG.tri_soup(b), res=12) == 0.0


def test_E7_similarity_is_deterministic():
    """同一对输入两次算出的读数逐位相同（读数要能复现才算证据）。"""
    a = MG.tri_soup(MG.build_mesh("chair"))
    b = MG.tri_soup(MG.build_mesh("table"))
    s1 = MG.shape_similarity(a, b, res=16)
    s2 = MG.shape_similarity(a, b, res=16)
    assert json.dumps(s1, sort_keys=True) == json.dumps(s2, sort_keys=True)


# =========================================================================== #
# F 组：形状库向后兼容 + 新图元校验
# =========================================================================== #
def test_F1_builtin_categories_use_no_mesh_only_kind():
    """既有 12 个内置类别的部件只用旧图元通道认得的 kind（像素零变化的证据）。"""
    bad = {}
    for cat in sorted(SH.SHAPE_LIBRARY):
        d = SH.legacy_degraded_kinds(SH.get_shape(cat))
        if d:
            bad[cat] = d
    assert not bad, "内置类别出现「旧通道会退化成盒子」的 kind：%r" % (bad,)


def test_F2_part_kinds_is_additive():
    """新 kind 只许**追加**（前 5 项与旧版逐位相同 ⇒ 旧判别语义不变）。"""
    assert SH.PART_KINDS[:5] == ("box", "sphere", "pillar", "disk", "pyramid")
    for k in ("cylinder", "cone", "torus"):
        assert k in SH.PART_KINDS
    assert set(SH.MESH_ONLY_KINDS) <= set(SH.PART_KINDS)
    assert set(SH.PART_MESH_KINDS) >= set(SH.PART_KINDS)


def test_F3_normalize_part_accepts_new_kinds_and_validates():
    p = SH.normalize_part(SH._cone(1.0, 1.2, 1.0, (200, 80, 60), top=0.35, seg=24))
    assert p["kind"] == "cone" and p["seg"] == 24 and abs(p["top"] - 0.35) < 1e-12
    assert SH.normalize_part(SH._torus(1.0, 0.2, (1, 2, 3)))["kind"] == "torus"
    # 越界必须拒收（fail-closed，不静默钳位）
    assert _raises(ValueError, SH.normalize_part,
                   SH._cone(1.0, 1.0, 1.0, (1, 2, 3), top=1.0))
    assert _raises(ValueError, SH.normalize_part,
                   SH._cone(1.0, 1.0, 1.0, (1, 2, 3), seg=2))
    assert _raises(ValueError, SH.normalize_part, {"kind": "banana",
                                                   "size": (1, 1, 1)})


def test_F4_shapes_dump_load_roundtrip_keeps_new_fields(tmp_path=None):
    cats = ["__guard_cone__", "__guard_torus__"]
    SH.register_shape(cats[0], [SH._cone(1.0, 2.0, 1.0, (10, 20, 30), top=0.4,
                                        seg=18)],
                      source="guard", recenter=False)
    SH.register_shape(cats[1], [SH._torus(1.0, 0.3, (30, 20, 10), seg=12)],
                      source="guard", recenter=False)
    tmp_path = tmp_path or _tmpdir()
    path = os.path.join(str(tmp_path), "shapes_guard.json")
    try:
        n = SH.dump_shapes(path, source="guard", only=cats)
        assert n == 2
        SH.SHAPE_LIBRARY.pop(cats[0], None)
        SH.SHAPE_LIBRARY.pop(cats[1], None)
        n_cat, n_part = SH.load_shapes(path, source="guard")
        assert (n_cat, n_part) == (2, 2)
        cone = SH.get_shape(cats[0])[0]
        assert cone["seg"] == 18 and abs(cone["top"] - 0.4) < 1e-12, cone
    finally:
        for c in cats:
            SH.SHAPE_LIBRARY.pop(c, None)


def test_F5_part_mesh_unknown_kind_raises_not_box():
    """未知 kind **不许**静默退化成盒子（"只能呈现方块"的病根）。"""
    assert _raises(ValueError, MG.part_to_mesh,
                   {"kind": "banana", "pos": (0, 0, 0), "size": (1, 1, 1)})
    assert _raises(ValueError, SH.part_mesh,
                   {"kind": "banana", "pos": (0, 0, 0), "size": (1, 1, 1)})


# =========================================================================== #
# G 组：部件组合 → 网格（形状库升级路径）
# =========================================================================== #
def test_G1_all_builtin_categories_upgrade_to_valid_mesh():
    for cat in sorted(SH.SHAPE_LIBRARY):
        m = SH.parts_mesh(SH.get_shape(cat), seg=12, name=cat)
        assert MG.is_watertight(m), "%s 的部件组合网格非闭合" % cat
        assert MG.signed_volume(m) > 0.0, "%s 的部件组合网格面朝内" % cat
        assert m.tri_count >= 12 * len(SH.get_shape(cat)) // 2


def test_G2_parts_mesh_keeps_cylinder_legs_round():
    """升级的意义：pillar 部件必须变成**圆**柱（分段数决定面数），不是方柱。"""
    one = SH.parts_mesh([SH._pillar(0.1, 0.5, 0.1, (1, 2, 3))], seg=16)
    assert MG.is_watertight(one)
    lo, hi = MG.mesh_bounds(one)
    d = [hi[k] - lo[k] for k in range(3)]
    assert abs(d[1] - 0.5) < 1e-9, d
    # 16 段圆柱 = 16 侧面×2 + 上下盖各 16 = 64 面
    assert one.tri_count == 64, one.tri_count
    assert abs(MG.signed_volume(one) - math.pi * 0.05 ** 2 * 0.5) < 0.002


def test_G3_part_to_mesh_applies_rot_and_pos():
    p = SH.normalize_part({"kind": "box", "pos": (1.0, 2.0, 3.0),
                           "size": (0.2, 0.4, 0.6),
                           "rot": (0.0, math.pi / 2.0, 0.0)})
    m = MG.part_to_mesh(p)
    lo, hi = MG.mesh_bounds(m)
    c = [(lo[k] + hi[k]) / 2.0 for k in range(3)]
    assert max(abs(c[k] - (1.0, 2.0, 3.0)[k]) for k in range(3)) < 1e-9, c
    d = [hi[k] - lo[k] for k in range(3)]
    assert max(abs(d[k] - (0.6, 0.4, 0.2)[k]) for k in range(3)) < 1e-9, d


# =========================================================================== #
# H 组：OBJ 往返（生成物必须能被世界模型自己的载入面读回）
# =========================================================================== #
def test_H1_obj_roundtrip_preserves_faces(tmp_path=None):
    m = MG.build_mesh("chair")
    tmp_path = tmp_path or _tmpdir()
    path = os.path.join(str(tmp_path), "chair.obj")
    MG.write_obj(m, path, name="chair")
    from lingshu.world.mesh3d import MESH_LIBRARY, load_mtl, load_obj_mesh
    info = load_obj_mesh(path, "__guard_chair__",
                         load_mtl(path[:-4] + ".mtl"), axis_swap=False)
    try:
        assert info["tris"] == m.tri_count, info
        assert info["verts"] == m.vert_count, info
        assert info["dropped"] == 0, info
        assert info["materials"] == len(set(m.colors)), info
    finally:
        MESH_LIBRARY.pop("__guard_chair__", None)


def test_H2_obj_header_carries_normalize_and_units():
    """`mesh3d.parse_obj` 是 fail-closed 的：没有 normalize 头就读不回来。"""
    text = MG.to_obj_text(MG.prim_box((1, 1, 1)), name="b")
    assert "# normalize:" in text and "scale=" in text and "center=(" in text
    assert text.startswith("# mesh_gen")
    assert "\nmtllib b.mtl\n" in text


def test_H3_obj_text_is_byte_identical_across_runs():
    a = MG.to_obj_text(MG.build_mesh("car"), name="car")
    b = MG.to_obj_text(MG.build_mesh("car"), name="car")
    assert a == b, "同一配方两次生成不逐字节相同 ⇒ 读数不可复现"


# =========================================================================== #
# I 组：纯度与确定性
# =========================================================================== #
def test_I1_build_mesh_is_pure_no_global_registration():
    """`build_mesh` 不许有副作用（登记必须显式走 `install`）。"""
    from lingshu.world.mesh3d import MESH_LIBRARY
    before = set(MESH_LIBRARY)
    MG.build_mesh("turret")
    MG.build_mesh("vase")
    assert set(MESH_LIBRARY) == before, "build_mesh 偷偷写了全局 MESH_LIBRARY"


def test_I2_install_registers_and_respects_overwrite():
    from lingshu.world.mesh3d import MESH_LIBRARY
    cat = "__guard_install__"
    MG.MESH_RECIPES[cat] = lambda cfg: MG.prim_box((1.0, 1.0, 1.0))
    MG._RECIPE_SOURCES[cat] = "guard"
    try:
        out = MG.install([cat])
        assert out == {cat: 12}, out
        assert cat in MESH_LIBRARY
        MESH_LIBRARY[cat] = "sentinel"
        assert MG.install([cat]) == {}, "overwrite=False 不该顶掉已登记件"
        assert MESH_LIBRARY[cat] == "sentinel"
    finally:
        MG.MESH_RECIPES.pop(cat, None)
        MG._RECIPE_SOURCES.pop(cat, None)
        MESH_LIBRARY.pop(cat, None)


def test_I3_fit_recipe_is_deterministic_and_bounded():
    target = MG.tri_soup(MG.build_mesh("chair"))
    grid = {"yaw": [0.0, 0.3, 0.6], "pitch": [-0.4, 0.0]}
    r1 = MG.fit_recipe("arm", target, grid, res=16)
    r2 = MG.fit_recipe("arm", target, grid, res=16)
    assert r1["config"] == r2["config"] and r1["mask_score"] == r2["mask_score"]
    assert r1["candidates"] == 6 and r1["evaluated"] == 6
    assert _raises(ValueError, MG.fit_recipe, "arm", target,
                   {"yaw": list(range(40)), "pitch": list(range(40))}, 16)


# =========================================================================== #
# J 组：mesh3d / world3d 互操作（"世界模型自己造的模型渲得出"）
# =========================================================================== #
def test_J1_to_mesh_geometry_conserves_faces_and_bounds():
    m = MG.build_mesh("tree")
    gm = MG.to_mesh_geometry(m, name="tree")
    assert len(gm.faces) == m.tri_count
    assert len(gm.verts) == m.vert_count
    lo1, hi1 = MG.mesh_bounds(m)
    lo2, hi2 = gm.bounds()
    assert max(abs(lo1[k] - lo2[k]) for k in range(3)) < 1e-12
    assert max(abs(hi1[k] - hi2[k]) for k in range(3)) < 1e-12


def test_J2_face_normals_unit_and_outward():
    """面法线 = 环绕向右手定则，且**朝外**（背面剔除的硬前提）。

    判据：法线与"面心→远点"同侧——用质心版本：normal·(face_center − mesh_center) > 0
    对凸网格成立；本件用凸体（棱锥）避免凹陷反例。
    """
    m = MG.prim_pyramid((1.0, 1.0, 2.0))
    gm = MG.to_mesh_geometry(m)
    for f in gm.faces:
        n = f.normal
        L = math.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
        assert abs(L - 1.0) < 1e-9, "面法线未归一化：%r" % (n,)
        a, b, c = (gm.verts[i] for i in f.idx)
        cen = ((a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0,
               (a[2] + b[2] + c[2]) / 3.0)
        assert n[0] * cen[0] + n[1] * cen[1] + n[2] * cen[2] > 0.0, \
            "面法线朝内（渲出来会内翻）：%r" % (f.idx,)


def test_J3_world3d_renders_procedural_mesh_untouched():
    """不改 world3d 一行，登记进 MESH_LIBRARY 就能渲出（像素 > 0）。"""
    try:
        from PIL import Image                                      # noqa: F401
    except ImportError:                                            # pragma: no cover
        return
    from lingshu.world.mesh3d import MESH_LIBRARY
    from lingshu.world.world3d import Camera3D, Object3D, World3D
    cat = "__guard_render__"
    md = MG.build_mesh("chair")
    lo, hi = MG.mesh_bounds(md)
    dims = [hi[k] - lo[k] for k in range(3)]
    saved = MESH_LIBRARY.get(cat)
    MESH_LIBRARY[cat] = MG.to_mesh_geometry(md, name=cat)
    try:
        cam = Camera3D.look_at((1.6, 1.4, 2.2), (0.0, dims[1] / 2.0, 0.0),
                               fov_deg=45.0)
        w = World3D(camera=cam)
        w.objects.append(Object3D(category=cat,
                                  center=(0.0, dims[1] / 2.0, 0.0),
                                  size=tuple(dims), color=(170, 170, 175),
                                  shape="box", source="guard"))
        img = w.render(240, 200, camera=cam)
        bg = (20, 24, 40)
        px = img.load()
        painted = sum(1 for y in range(200) for x in range(240)
                      if px[x, y][:3] != bg)
        assert painted > 300, "自生成网格渲染出的前景像素只有 %d" % painted
    finally:
        if saved is None:
            MESH_LIBRARY.pop(cat, None)
        else:
            MESH_LIBRARY[cat] = saved


def test_J4_metrics_accept_mesh3d_geometry_objects():
    """度量层同时吃 MeshData 与 MeshGeometry（跨来源对拍）。"""
    gm = MG.to_mesh_geometry(MG.build_mesh("vase"))
    assert MG.is_watertight(gm)
    assert MG.signed_volume(gm) > 0.0
    assert MG.geometry_report(gm)["watertight"] is True


def test_guard_is_collectable():
    assert os.path.basename(__file__).startswith("test_")
    assert MG is not None and SH is not None


def _main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print("FAIL %-58s %s" % (t.__name__, e))
        else:
            passed += 1
            print("ok   %s" % t.__name__)
    print("\n%d/%d passed" % (passed, len(tests)))
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(_main())
