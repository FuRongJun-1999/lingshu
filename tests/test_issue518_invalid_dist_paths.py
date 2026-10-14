# -*- coding: utf-8 -*-
"""test_issue518_invalid_dist_paths · 无效 dist 路径不参选 compute_d_task（#518）
==================================================================================
缺陷（#518，改前实测三例）：#124 的修复把选路键改为 ``(cond_gap, dist)`` 字典序
并把 ``d_task=∞`` 判 BLINDSPOT，但只覆盖「唯一可用路径缺 dist」——多路径下：

  A  缺 dist 但条件齐全（cond_gap=0）的路径压过有限距离路径 ⇒
     ``{'d_task': inf, 'blindspot': True, 'best_path': 'full'}``——明明存在
     有限距离的可用路径却报盲区（docstring 的 ``D_task = min_{p 可用} …`` 不成立）。
  B  NaN dist 参与比较恒 False ⇒ 结果随 dict 插入顺序漂移（B1 报 nan、B2 报 1.0）。
  C  dist=None 直接 ``TypeError: '<' not supported …``。

修法：选路面＝可用且 dist 为**有限数**的路径（缺失/NaN/None 一律不参选）；
该面为空 ⇒ ∞ + BLINDSPOT（与 #124 单路径口径一致，推广到多路径）。
#124 的字典序（先 cond_gap 后 dist）在有效路径间**原样保留**。

判据（本件）：
  G1 A 例：有限距离路径胜出——d_task=1.0、blindspot=False、best_path='close'；
  G2 B 例：NaN 不参选且结果与插入顺序无关（B1=B2=1.0/'y'）；
  G3 C 例：None 不再 TypeError，同 G1 语义（1.0/'y'）；
  G4 全无效面：所有可用路径 dist 均缺失/非有限 ⇒ ∞ + blindspot=True
     （#124 单路径口径的推广，既有守卫 test_gap_dual_d_meta_guard 亦钉此面）；
  G5 防过修：#124 的字典序不回归——p_close(0.5,{a}) vs p_far(2.0,{a,b})、
     任务 {a,b} ⇒ 仍选 p_far（条件覆盖完整者优先）。
  G6 定点变异：把选路面从 valid 还原为 usable ⇒ G1/G2/G3 必红（判据不空转）。

运行（仓根）：python -X utf8 -m pytest tests/test_issue518_invalid_dist_paths.py -q
"""
from __future__ import annotations

from pathlib import Path

from lingshu.world.gap_dual import GapDual

SRC = Path(__file__).resolve().parents[1] / "lingshu" / "world" / "gap_dual.py"
NAN = float("nan")


def _show(r):
    return {k: r.get(k) for k in ("d_task", "blindspot", "best_path", "cond_gap")}


def test_a_finite_path_beats_gap0_nodist_path():
    """G1：改前 {'d_task': inf, 'blindspot': True, 'best_path': 'full'}——有限路径被压。"""
    g = GapDual()
    paths = {"full": {"available": True, "cond_set": {"a", "b"}},
             "close": {"available": True, "dist": 1.0, "cond_set": {"a"}}}
    r = g.compute_d_task(paths, {"a", "b"})
    assert r["d_task"] == 1.0 and r["blindspot"] is False and r["best_path"] == "close", _show(r)


def test_b_nan_excluded_order_independent():
    """G2：NaN 不参选、与插入顺序无关（改前 B1=nan/x、B2=1.0/y）。"""
    g = GapDual()
    b1 = {"x": {"available": True, "dist": NAN}, "y": {"available": True, "dist": 1.0}}
    b2 = {"y": {"available": True, "dist": 1.0}, "x": {"available": True, "dist": NAN}}
    r1, r2 = g.compute_d_task(b1), g.compute_d_task(b2)
    assert _show(r1) == _show(r2), (_show(r1), _show(r2))
    assert r1["d_task"] == 1.0 and r1["best_path"] == "y" and r1["blindspot"] is False, _show(r1)


def test_c_none_no_typeerror():
    """G3：dist=None 不再 TypeError（改前 '<' not supported …）。"""
    g = GapDual()
    paths = {"x": {"available": True, "dist": None}, "y": {"available": True, "dist": 1.0}}
    r = g.compute_d_task(paths)
    assert r["d_task"] == 1.0 and r["best_path"] == "y" and r["blindspot"] is False, _show(r)


def test_d_all_invalid_dist_is_blindspot():
    """G4：全无效面 ⇒ ∞ + BLINDSPOT（#124 单路径口径的多路径推广）。"""
    g = GapDual()
    paths = {"x": {"available": True},                       # 缺 dist
             "y": {"available": True, "dist": NAN},           # NaN
             "z": {"available": True, "dist": None}}          # None
    r = g.compute_d_task(paths)
    assert r["d_task"] == float("inf") and r["blindspot"] is True, _show(r)


def test_e_124_lexicographic_order_kept():
    """G5（防过修）：#124 字典序保留——条件覆盖完整的 p_far 仍胜出。"""
    g = GapDual()
    paths = {"p_close": {"available": True, "dist": 0.5, "cond_set": {"a"}},
             "p_far": {"available": True, "dist": 2.0, "cond_set": {"a", "b"}}}
    r = g.compute_d_task(paths, {"a", "b"})
    assert r["best_path"] == "p_far" and r["d_task"] == 2.0, _show(r)


def test_mutation_source_anchor_valid_selection():
    """G6（形态锚·定点变异自证）：选路面必须是 `valid`（不得还原 `usable`）。"""
    src = SRC.read_text(encoding="utf-8")
    assert "valid, key=lambda kv: (_cond_gap(kv[1])," in src, (
        "#518：选路面应作用于 valid（有限 dist 路径）——还原为 usable 即回潮："
        "A/B/C 三例在 test_gap_dual_d_meta_guard 之外将再次复现")


if __name__ == "__main__":
    import subprocess
    import sys
    raise SystemExit(subprocess.call(
        [sys.executable, "-X", "utf8", "-m", "pytest", __file__, "-q", "--no-header"]))
