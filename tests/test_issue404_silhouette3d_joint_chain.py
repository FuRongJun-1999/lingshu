# -*- coding: utf-8 -*-
"""#404 守卫：silhouette3d 关节必须有父子链——父关节旋转带动整条子链。

缺陷态（取证 · 修复前 HEAD `abb52d0`）：
  · `SilhouettePart` 只有 `pivot`（`silhouette3d.py:41`）没有 `parent` 字段；
  · `rotated_points` 只绕**本部件自己的** pivot 旋转（`:46 if self.pivot is None or
    self.angle == 0.0: return self.points` / `:48 cx, cy = self.pivot`）；
  · `apply_pose` 却按**骨骼链语义**设值——`setj(s, 0.9, "shoulder_r", "upper_r")` /
    `setj(s, -1.2, "elbow_r", "lower_r")`（`:878-879`）——上臂转了，肘及下臂纹丝
    不动 ⇒ 手臂在肩关节处断开。

判据：子部件（`lower_*`/`elbow_*`/`calf_*`/`tail2..4`）的 `angle` 是**相对父部件**
的角度；渲染点 = 先绕自身 pivot 施加自身 angle，再依次绕父、祖父…的 pivot 施加其
angle（`Silhouette3D.transformed_points`）。故父关节一动，子链各点必须跟着动。

来源：关节链语义为本件报告的判据（报告 #404），非理论章节引用——属**经验标定，
追不到理论出处**；`parent` 字段与 `transformed_points` 为本件新增的判据实现。

变异点：把 `render` 里的 `self.transformed_points(part)` 改回
`part.rotated_points()`（或删掉各部件的 `parent=`）⇒ 本文件 test_chain_* 必红。

运行（lingshu 仓根）：python -X utf8 -m pytest tests/test_issue404_silhouette3d_joint_chain.py -q
"""
from __future__ import annotations

import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.world import silhouette3d as S  # noqa: E402


def _part(s, name):
    return next((p for p in s.parts if p.name == name), None)


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ── 量具自检 ────────────────────────────────────────────────────────────────
def test_gauge_is_alive():
    """量具自检：`parent` 字段与 `transformed_points` 真在场。"""
    p = S.SilhouettePart("x", [(0.0, 0.0)])
    assert hasattr(p, "parent") and p.parent is None, "SilhouettePart 缺 `parent` 字段"
    assert callable(getattr(S.Silhouette3D, "transformed_points", None)), (
        "Silhouette3D 缺 `transformed_points` ⇒ 父子链未实现")


# ── 链结构（两个模板都要有）────────────────────────────────────────────────
def test_chain_structure_declared():
    """两个模板的四肢/尾巴部件必须声明父子链（缺陷态：parent 全为 None）。"""
    for maker in (S.fatfish_skinned, S.fatfish_skeleton):
        s = maker()
        if maker is S.fatfish_skinned:
            expected = {"lower_l": "upper_l", "lower_r": "upper_r",
                        "calf_l": "thigh_l", "calf_r": "thigh_r",
                        "tail2": "tail1", "tail3": "tail2", "tail4": "tail3"}
        else:
            expected = {"elbow_l": "shoulder_l", "elbow_r": "shoulder_r",
                        "tail2": "tail1", "tail3": "tail2"}
        for child, parent in expected.items():
            cp = _part(s, child)
            assert cp is not None, "%s 缺部件 %s" % (maker.__name__, child)
            assert cp.parent == parent, (
                "%s：%s.parent=%r，应为 %r（缺陷态=各转各的断链）"
                % (maker.__name__, child, cp.parent, parent))


def test_no_dangling_or_cyclic_parent():
    """parent 必须指向同角色内真实存在的部件，且不得成环。"""
    for maker in (S.fatfish_skinned, S.fatfish_skeleton):
        s = maker()
        names = {p.name for p in s.parts}
        for p in s.parts:
            if p.parent is not None:
                assert p.parent in names, (
                    "%s：%s.parent=%r 指向不存在的部件" % (maker.__name__, p.name, p.parent))
                # 沿链走一圈不得回到自身
                seen, cur = set(), p
                by = {q.name: q for q in s.parts}
                while cur is not None:
                    assert cur.name not in seen, (
                        "%s：parent 链成环于 %s" % (maker.__name__, cur.name))
                    seen.add(cur.name)
                    cur = by.get(cur.parent) if cur.parent else None


# ── 行为：父关节旋转带动子链 ───────────────────────────────────────────────
def test_parent_rotation_moves_child_chain():
    """父关节（肩）旋转 ⇒ 子部件（肘）顶点必须跟着动，哪怕肘自身 angle=0。

    缺陷态（修复前）：肘点只绕自身 pivot 转 ⇒ 肩转动对它零影响，本断言红。
    """
    for maker in (S.fatfish_skinned, S.fatfish_skeleton):
        s = maker()
        up = _part(s, "upper_r" if maker is S.fatfish_skinned else "shoulder_r")
        lo = _part(s, "lower_r" if maker is S.fatfish_skinned else "elbow_r")
        assert up is not None and lo is not None
        lo.angle = 0.0                     # 肘自身不转，只动肩
        before = list(s.transformed_points(lo))
        up.angle = 0.9
        after = list(s.transformed_points(lo))
        moved = max(_dist(b, a) for b, a in zip(before, after))
        assert moved > 1e-3, (
            "%s：父关节 %s 转 0.9 rad，子部件 %s 顶点位移仅 %.6f ⇒ 链断"
            % (maker.__name__, up.name, lo.name, moved))
        # 断链对照：只绕自身 pivot 时子部件纹丝不动（正是缺陷态读数）
        assert list(lo.rotated_points()) == before, (
            "子部件自身 rotated_points 不该受父关节影响（父链只在 transformed_points 施加）")


def test_child_angle_composes_with_parent():
    """肘自身角度须叠加在肩角度之上（链式合成），不是互相覆盖。"""
    s = S.fatfish_skeleton()
    sh = _part(s, "shoulder_r")
    el = _part(s, "elbow_r")
    sh.angle, el.angle = 0.0, 0.0
    base = list(s.transformed_points(el))
    sh.angle = 0.5
    sh_only = list(s.transformed_points(el))
    el.angle = 0.5
    both = list(s.transformed_points(el))
    assert max(_dist(a, b) for a, b in zip(base, sh_only)) > 1e-3
    assert max(_dist(a, b) for a, b in zip(sh_only, both)) > 1e-3, (
        "肘自身角度被父链吞掉 ⇒ 未合成")


def test_tail_chain_propagates():
    """尾巴 3 节链：tail1 摆动必须带动 tail2/tail3（`_set_tail_wave` 的级联语义）。"""
    for maker in (S.fatfish_skinned, S.fatfish_skeleton):
        s = maker()
        t2 = _part(s, "tail2")
        t2.angle = 0.0
        before = list(s.transformed_points(t2))
        S._set_tail_wave(s, 0.5)
        after = list(s.transformed_points(t2))
        moved = max(_dist(b, a) for b, a in zip(before, after))
        assert moved > 1e-3, (
            "%s：摆尾（tail1 转）未带动 tail2，位移 %.6f ⇒ 尾链断"
            % (maker.__name__, moved))


def test_pose_happy_moves_forearm_with_shoulder():
    """端到端：`apply_pose(s,"happy")` 后前臂必须因肩角而离开自然位（不只自身转）。"""
    for maker in (S.fatfish_skinned, S.fatfish_skeleton):
        s = maker()
        S.apply_pose(s, "neutral")
        lo = _part(s, "lower_r" if maker is S.fatfish_skinned else "elbow_r")
        before = list(s.transformed_points(lo))
        S.apply_pose(s, "happy")
        after = list(s.transformed_points(lo))
        moved = max(_dist(b, a) for b, a in zip(before, after))
        assert moved > 1e-3, (
            "%s：happy 姿态未使前臂随肩移动（位移 %.6f）" % (maker.__name__, moved))


def test_render_path_goes_through_chain(monkeypatch):
    """渲染管线必须走 `transformed_points`（缺陷态：render 用 `rotated_points`）。

    变异点：把 `render` 里的 `self.transformed_points(part)` 改回
    `part.rotated_points()` ⇒ 本断言红（spy 收不到子部件调用）。
    """
    s = S.fatfish_skeleton()
    calls = []
    orig = S.Silhouette3D.transformed_points

    def spy(self, part):
        calls.append(part.name)
        return orig(self, part)

    monkeypatch.setattr(S.Silhouette3D, "transformed_points", spy)
    s.render(screen_w=64, screen_h=64)
    assert "elbow_r" in calls and "tail2" in calls, (
        "render 未对子部件走父子链 ⇒ 姿态渲染仍是断链（calls=%r）" % (sorted(calls),))


if __name__ == "__main__":
    raise SystemExit(__import__("pytest").main([__file__, "-q"]))
