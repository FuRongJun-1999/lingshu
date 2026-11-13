# -*- coding: utf-8 -*-
"""issue #231-②③ 守卫：部件组合不得有脱离件（DETACHED），且朝向语义正确。

病象（修前）：`bird` 的喙在头的反侧（z=−0.2，头 z=+0.05）且与身体脱离 0.03；
`cat` 的尾是竖直短柱且与身体脱离 0.075；`person` 头与躯干脱离 0.04、双臂与
躯干 x 向脱离 0.01；`chair` 四腿与座面各脱离 0.02。

判据（构造性，AABB 相交即视为相连）：
  A  全库不变量：每类（≥2 部件）每部件与至少一个其它部件相交（单部件类豁免）；
  B  专项：bird 喙在头前方（z > 头心 z）；cat 尾沿 z 向后（dz 尺寸 > dy 尺寸）；
  C  变异臂：bird 喙回退到修前坐标 ⇒ A 红（--mutate 自证）。
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from lingshu.world.shapes import SHAPE_LIBRARY  # noqa: E402

FAILS = []


def _aabb(p):
    dx, dy, dz = p["size"]
    x, y, z = p["pos"]
    return (x - dx / 2, x + dx / 2, y - dy / 2, y + dy / 2, z - dz / 2, z + dz / 2)


def _overlap(a, b):
    return (a[0] <= b[1] and b[0] <= a[1] and a[2] <= b[3] and b[2] <= a[3]
            and a[4] <= b[5] and b[4] <= a[5])


def test_a_every_part_attaches_to_another():
    for cat, parts in SHAPE_LIBRARY.items():
        if len(parts) < 2:
            continue  # 单部件类（如 mountain）豁免
        boxes = [_aabb(p) for p in parts]
        for i, p in enumerate(parts):
            attached = any(_overlap(boxes[i], boxes[j])
                           for j in range(len(parts)) if j != i)
            assert attached, "DETACHED：%s[%d] %s pos=%r 无任何相交部件" % (
                cat, i, p["kind"], p["pos"])


def test_b_semantic_orientation_pins():
    bird = SHAPE_LIBRARY["bird"]
    head = next(p for p in bird if p["kind"] == "sphere" and p["size"][0] < 0.2)
    beak = next(p for p in bird if p["kind"] == "pyramid")
    assert beak["pos"][2] > head["pos"][2], "喙仍 在头的反侧（z 应大于头心 z）"
    cat = SHAPE_LIBRARY["cat"]
    tail = next(p for p in cat if p["kind"] in ("pillar", "box")
                and p["color"] == (180, 130, 90))
    assert tail["size"][2] > tail["size"][1], "猫尾仍竖直（应沿 z 向后）"


def _run_assertions():
    del FAILS[:]
    for fn in (test_a_every_part_attaches_to_another,
               test_b_semantic_orientation_pins):
        try:
            fn()
        except AssertionError as exc:
            FAILS.append(str(exc))
    return len(FAILS)


if __name__ == "__main__":
    n = _run_assertions()
    if "--mutate" in sys.argv:
        # 写盘后必须**子进程**重跑——本进程 import 的库是变异前的旧值。
        import re as _re
        import subprocess as _sp
        target = _REPO / "lingshu" / "world" / "shapes.py"
        orig = target.read_bytes()
        old = b"(220, 150, 60), 0, 0.3, 0.15)"
        assert orig.count(old) == 1, orig.count(old)
        mut = orig.replace(old, b"(220, 150, 60), 0, 0.05, -0.2)")
        target.write_bytes(mut)
        red = False
        try:
            r = _sp.run([sys.executable, "-X", "utf8", __file__],
                        capture_output=True, text=True, encoding="utf-8",
                        errors="replace", cwd=str(_REPO), timeout=120)
            red = (r.returncode != 0
                   or bool(_re.findall(r"DETACHED", r.stdout + r.stderr)))
            print("[MUTATE-%s] 喙回退修前坐标 ⇒ 子进程判红=%s" % (
                "OK" if red else "FAIL（变异未生效）", bool(red)))
        finally:
            target.write_bytes(orig)
        sys.exit(0 if red else 1)
    print("通过 %d 组断言" % (2 - n))
    sys.exit(1 if n else 0)
