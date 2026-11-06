# -*- coding: utf-8 -*-
"""test_issue2403_mesh_addquad_atomic_guard —— add_quad 原子性守卫。

缺陷（dsh-bounty #2403）
============================================================================
`MeshData.add_quad` 直接串联两次 `add_tri`（各查各自）——第二个三角越界抛
`ValueError` 时第一个已 `faces.append`/`colors.append` ⇒ 调用方按常识重试会拿到
「多一块面片」的半成品 mesh（渲染可见多余面片、无告警）。同文件 add_tri（:274）
与 add_poly（:286）都是「先全量校验后写入」，三条写面对失败态给出两种语义。

实现侧口径（本守卫的判据）
--------------------------------------------------------------------------
前置段只做「int 转换 + 四索引越界检查」，通过后照原样执行两次 add_tri——
合法输入结果**逐位不变**（含退化三角的 degenerate_dropped 计数语义）。

守卫用例
--------------------------------------------------------------------------
G1 缺陷态：d 越界 ⇒ ValueError 且 faces/colors **零残留**（改前留 1 面）。
G2 缺陷态：a/b/c 任一越界同样零残留。
G3 正对照：四合法索引 ⇒ 恰 2 个三角，顶点索引与颜色照旧。
G4 语义保持：退化三角（a==b）仍只计一次 degenerate_dropped、只落 1 面。
G5 与非 int 输入兼容：字符串数字索引照旧可转（int("1")），非法串仍报错且零残留。

运行（lingshu 仓根）：
    python -X utf8 -m pytest tests/test_issue2403_mesh_addquad_atomic_guard.py -q
"""
from __future__ import annotations

import pytest

from lingshu.world.mesh_gen import MeshData


def _quad_mesh():
    m = MeshData()
    for v in [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]:
        m.add_vert(v)
    return m


# ---------------------------------------------------------------- G1

def test_2403_bad_d_leaves_no_residue():
    m = _quad_mesh()
    with pytest.raises(ValueError):
        m.add_quad(0, 1, 2, 99)
    assert m.faces == [], "改前会残留三角 (0,1,2)：%r" % (m.faces,)
    assert m.colors == [], m.colors


# ---------------------------------------------------------------- G2

@pytest.mark.parametrize("idx", [(99, 1, 2, 3), (0, 99, 2, 3), (0, 1, 99, 3),
                                 (-1, 1, 2, 3)])
def test_2403_bad_any_index_leaves_no_residue(idx):
    m = _quad_mesh()
    with pytest.raises(ValueError):
        m.add_quad(*idx)
    assert m.faces == [], "索引 %r 失败后残留：%r" % (idx, m.faces)
    assert m.colors == []


# ---------------------------------------------------------------- G3

def test_2403_legal_quad_unchanged():
    m = _quad_mesh()
    m.add_quad(0, 1, 2, 3)
    assert m.faces == [(0, 1, 2), (0, 2, 3)], m.faces
    assert len(m.colors) == 2


# ---------------------------------------------------------------- G4

def test_2403_degenerate_semantics_preserved():
    m = _quad_mesh()
    m.add_quad(0, 0, 2, 3)          # 第一个三角退化（a==b）
    assert len(m.faces) == 1, m.faces        # 只有 (0,2,3)
    assert m.stats.get("degenerate_dropped") == 1, m.stats


# ---------------------------------------------------------------- G5

def test_2403_string_indices_and_invalid_string():
    m = _quad_mesh()
    m.add_quad("0", "1", "2", "3")           # int() 转换照旧
    assert len(m.faces) == 2, m.faces
    m2 = _quad_mesh()
    with pytest.raises((ValueError, TypeError)):
        m2.add_quad("x", 1, 2, 3)
    assert m2.faces == [], "转换失败后也须零残留：%r" % (m2.faces,)
