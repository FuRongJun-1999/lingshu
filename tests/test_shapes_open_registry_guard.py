# -*- coding: utf-8 -*-
"""形状库开放注册面 —— 守卫

缺陷（本件 · [world/shapes][宣称 vs 实现]）：
  `shapes.py` 的文件头 docstring 从里程碑 1.3 起就写着「可扩展：`register_shape(category,
  parts)` 任意新类别即插即用」，但 **`register_shape` 从未实现** —— 形状库是 10 个手写
  类别、外部无法注册；`Part` 是裸 dict、**无任何校验**；也没有任何可交换的描述格式。
  后果：语义时空图里的实体只能用内置词表，**身体侧（真实引擎）的几何进不来**；
  且坏值（负尺寸 / NaN / 未知 kind）会被静默接受，一路带到绘制分支才炸或者画错。

修法（兑现 docstring 承诺，全部零外部依赖）：
  ① `normalize_part()` —— `Part` 单点校验（kind 白名单 / 有限数 / 正尺寸 / 颜色域），
     坏值一律 `ValueError`（**fail-closed，不静默降级、不给默认值**）；
  ② `register_shape()` / `load_shapes()` / `dump_shapes()` —— 开放注册面 + 带 schema
     的可交换格式 `lingshu.shapes/1`；`load_shapes` **先全量校验再落库**（不留半截登记）；
  ③ `shape_sources()` —— 来源可追溯（内置 / 外部文件），`replace=False` 时拒绝被
     不同来源静默顶掉。
  ⚠ `pos_origin` 是**必需**字段（`bbox_center` | `anchor`）：内置库口径把部件包围盒
    归零到物体中心；而身体侧导出的 `pos` 是**相对关节枢轴**的，归零会把枢轴锚点挪走
    （实测 `[0.246,0.244,-0.260] → [0.071,0.025,0.030]`，机构随之整体错位）。

判据来源：docstring 自述的可扩展承诺（宣称 vs 实现）+ 身体侧枢轴口径的实测锚点。

断言组（抽掉修复 ⇒ 各组按标注转红）：
  A 注册生效：新类别进库、`has_shape` 为真、部件被规范化（抽 ② ⇒ 红：无此函数）
  B fail-closed：坏 kind / 负尺寸 / NaN / 非映射 / 空序列 ⇒ `ValueError`，且
    **库未被改动**（抽 ① ⇒ 红：坏值被静默接受）
  C 载入原子性：schema 不符或任一件非法 ⇒ raise 且**一个类别都没落库**（抽 ② ⇒ 红）
  D 枢轴口径：`pos_origin="anchor"` 原样保留、`"bbox_center"` 归零；非法取值 raise
  E 往返：`dump_shapes` → `load_shapes` 几何逐位一致（抽 ③ ⇒ 红）
  F 防误杀：内置 10 类在所有上述操作后**仍在且件数不变**

运行（仓根）：python -X utf8 -m pytest tests/test_shapes_open_registry_guard.py -q --no-header
"""
from __future__ import annotations

import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.world import shapes as S  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_library():
    """每个用例前后快照/还原全局库 —— 免得污染同仓其它测试。"""
    saved = {k: [dict(p) for p in v] for k, v in S.SHAPE_LIBRARY.items()}
    saved_src = dict(S._SHAPE_SOURCES)
    yield
    S.SHAPE_LIBRARY.clear()
    S.SHAPE_LIBRARY.update(saved)
    S._SHAPE_SOURCES.clear()
    S._SHAPE_SOURCES.update(saved_src)


BOX = {"kind": "box", "pos": (0.0, 0.0, 0.0), "size": (1.0, 2.0, 3.0)}


def test_a_注册生效并规范化():
    """A：`register_shape` 真能注册；部件被规范化（元组化 + color 补齐）。"""
    n = S.register_shape("__probe_a", [BOX], source="test")
    assert n == 1
    assert S.has_shape("__probe_a") is True
    p = S.SHAPE_LIBRARY["__probe_a"][0]
    assert p["kind"] == "box"
    assert tuple(p["pos"]) == (0.0, 0.0, 0.0)
    assert len(p["color"]) == 3, "color 未补齐 ⇒ 绘制分支会拿到长度不对的颜色"
    assert S.shape_sources()["__probe_a"] == "test", "来源未登记 ⇒ 不可审计"


@pytest.mark.parametrize("bad", [
    # ⚠ 2026-10-14：原样例是 `"torus"`，而本 PR 已把 torus/cylinder/cone
    #   并入 `PART_KINDS`（自生成建模扩容）⇒ 它不再"不在白名单"。改用**真正非法**的名字，
    #   判据不变（仍是"kind 不在白名单 ⇒ 拒收"）。
    {"kind": "no-such-kind", "size": (1, 1, 1)},            # kind 不在白名单
    {"kind": "box", "size": (-1, 1, 1)},                     # 尺寸非正
    {"kind": "box", "size": (1, 0, 1)},                      # 尺寸为 0
    {"kind": "box", "size": (float("nan"), 1, 1)},           # 非有限
    {"kind": "box", "size": (1, 1)},                         # 分量个数不对
    {"kind": "box"},                                         # 缺 size
    "not-a-mapping",                                         # 根本不是映射
    {"kind": "box", "size": (1, 1, 1), "color": (1, 2)},     # 颜色分量个数不对
], ids=["kind", "size-neg", "size-zero", "size-nan", "size-arity", "size-missing",
        "not-a-mapping", "color-arity"])
def test_b_坏值一律拒收且库不动(bad):
    """B：坏值 ⇒ `ValueError`，且**库未被改动**（不留半截登记）。"""
    before = dict(S.SHAPE_LIBRARY)
    with pytest.raises(ValueError):
        S.register_shape("__probe_b", [bad], source="test")
    assert "__probe_b" not in S.SHAPE_LIBRARY, "坏值被静默登记 ⇒ fail-open"
    assert set(S.SHAPE_LIBRARY) == set(before), "库被改动了"


def test_b2_空序列等于没注册须拒收():
    """B2：空部件序列 = 空形状，必须拒收（而非静默注册一个空类别）。"""
    with pytest.raises(ValueError):
        S.register_shape("__probe_b2", [], source="test")
    assert "__probe_b2" not in S.SHAPE_LIBRARY


def test_b3_跨来源覆盖须显式同意():
    """B3：`replace=False` 时，不同来源不许静默顶掉已有类别。"""
    S.register_shape("__probe_b3", [BOX], source="blender/HEX01")
    with pytest.raises(ValueError):
        S.register_shape("__probe_b3", [BOX], source="other/src", replace=False)
    assert S.shape_sources()["__probe_b3"] == "blender/HEX01", "来源被静默改了"


def test_c_载入先全量校验再落库(tmp_path):
    """C：schema 不符 / 任一件非法 ⇒ raise，且**一个类别都没落库**。"""
    good = {"schema": S.SHAPES_SCHEMA, "source": "t", "shapes": {"__ok": [dict(BOX)]}}
    bad_schema = {"schema": "lingshu.shapes/999", "shapes": {"__bad": [dict(BOX)]}}
    bad_part = {"schema": S.SHAPES_SCHEMA,
                "shapes": {"__ok2": [dict(BOX)], "__bad2": [{"kind": "box", "size": (-1, 1, 1)}]}}
    for i, doc in enumerate((bad_schema, bad_part)):
        p = tmp_path / ("d%d.json" % i)
        p.write_text(json.dumps(doc), encoding="utf-8")
        before = set(S.SHAPE_LIBRARY)
        with pytest.raises(ValueError):
            S.load_shapes(str(p))
        assert set(S.SHAPE_LIBRARY) == before, "载入失败却落了库 ⇒ 半截登记"
    p = tmp_path / "ok.json"
    p.write_text(json.dumps(good), encoding="utf-8")
    n_cat, n_part = S.load_shapes(str(p))
    assert (n_cat, n_part) == (1, 1)


def test_d_枢轴口径不得被静默归零(tmp_path):
    """D：`pos_origin="anchor"` 必须**原样保留** —— 这是身体侧几何不错位的前提。"""
    pos = (0.246, 0.244, -0.260)
    doc = {"schema": S.SHAPES_SCHEMA, "source": "blender/HEX01", "pos_origin": "anchor",
           "shapes": {"__ax": [{"kind": "box", "pos": pos, "size": (0.1, 0.1, 0.1)}]}}
    p = tmp_path / "anchor.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    S.load_shapes(str(p))
    got = tuple(S.SHAPE_LIBRARY["__ax"][0]["pos"])
    assert got == pos, "anchor 口径被归零 ⇒ 关节枢轴锚点被挪走、机构整体错位"

    doc["pos_origin"] = "bbox_center"
    p2 = tmp_path / "bc.json"
    p2.write_text(json.dumps(doc), encoding="utf-8")
    S.load_shapes(str(p2))
    assert tuple(S.SHAPE_LIBRARY["__ax"][0]["pos"]) == (0.0, 0.0, 0.0), "bbox_center 未归零"

    doc["pos_origin"] = "guess"
    p3 = tmp_path / "bad.json"
    p3.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError):
        S.load_shapes(str(p3))


def test_e_往返几何逐位一致(tmp_path):
    """E：`dump_shapes` → `load_shapes` 往返后，几何与来源逐位一致。"""
    S.register_shape("__rt", [BOX, {"kind": "sphere", "pos": (1, 2, 3), "size": (0.5, 0.5, 0.5)}],
                     source="round/trip")
    before = {c: [dict(p) for p in S.SHAPE_LIBRARY[c]] for c in ("__rt",)}
    src = {c: S.shape_sources()[c] for c in ("__rt",)}
    out = tmp_path / "rt.json"
    assert S.dump_shapes(str(out), source="round/trip", only=["__rt"]) == 1
    S.SHAPE_LIBRARY.pop("__rt")
    n_cat, n_part = S.load_shapes(str(out))
    assert (n_cat, n_part) == (1, 2)
    assert {c: [dict(p) for p in S.SHAPE_LIBRARY[c]] for c in ("__rt",)} == before
    assert S.shape_sources()["__rt"] == src["__rt"]


#: 内置 10 类及其件数（2026-10-12 实测；件数是**防误杀**的硬数，动了就该有人来看）
BUILTIN = {"chair": 6, "table": 5, "bed": 3, "cat": 5, "dog": 6,
           "rabbit": 4, "bird": 4, "person": 6, "tree": 2, "mountain": 1}


def test_f_内置库不被误伤():
    """F（防误杀）：内置 10 类始终在、件数不变、来源回落 builtin。"""
    assert set(S.SHAPE_LIBRARY) >= set(BUILTIN), "内置类别缺失"
    for c, n in BUILTIN.items():
        assert len(S.SHAPE_LIBRARY[c]) == n, "内置类别 %s 件数 %d ≠ %d" % (
            c, len(S.SHAPE_LIBRARY[c]), n)
    assert S.shape_sources()["chair"] == "builtin", "内置件来源应回落 builtin"
    # 图元白名单必须覆盖内置库实际用到的 kind（否则内置件自己就画不出来）
    used = {p["kind"] for parts in S.SHAPE_LIBRARY.values() for p in parts}
    assert used <= set(S.PART_KINDS), "内置库用了白名单外的图元：%s" % (used - set(S.PART_KINDS))
