# -*- coding: utf-8 -*-
"""坐标边界语义「只有一份实现」—— 守卫

缺陷（[world/*][重复实现 · 静默失败]）：
  「把实体中心钳在世界内（留 0.5 米余量）」这条规则在仓里被**手写了五份**：
    `lingshu/world/world_model.py`（本体 `_apply_move`）、`scene_simulator.py`、
    `world_learner.py`（自带一份 `_apply_move`）、`spacetime_consistency.py`、
    `voxel_world.py`。五份的**量化位数、零方向短路范围、非有限坐标处置各不相同**，
    且都不带判据说明 —— 任何一处被「顺手改一下」（换成 `min(max(...))`、先量化再钳、
    或把 NaN 静默抬成边界值）都会**改掉域内 trace**，而没有任何断言会红。
    `#347` 就是这条路的实例：`max(lo, min(hi, nan))` 在 Python 下**返回 `hi`**
    （`min` 在比较为假时返回第一个参数）⇒ NaN 被**静默**抬成上边界。

修法（`lingshu/world/coord_contract.py`，本次新增）：
  ① 算式收成**一个**真源 `clamp_axis` / `clamp_xz` / `apply_move`；
  ② 四个站点改路由（`from .coord_contract import …`），算法逐字不变；
  ③ 口径差异**显式命名**而不是藏在函数体里：
     `ZERO_ROUND_ALL | ZERO_XZ_ONLY`、`NF_NONE | NF_REJECT_XZ_ONLY | NF_REJECT_ROUND_ALL`；
  ④ 附带 `scan_clamp_copies()`（AST 档）把「又抄一份」变成**可断言**的事。

判据来源：本仓 `bench/bench_world.py` 的域内 `trace_sha256`（改前改后须逐位一致）。

断言组（抽掉修复 ⇒ 各组按标注转红）：
  A **逐位等价**：`clamp_xz` / `apply_move`（各口径）与**原算式**在系统性输入网格上
    `repr` 级相同（含 ±0.0 / NaN / ±inf / 边界值）——抽 ① ⇒ 红（模块不存在）
  B 口径**不混**：`ZERO_XZ_ONLY` 只量化 x/z、`ZERO_ROUND_ALL` 三分量都量化；
    `NF_NONE` 复现旧的静默抬界、`NF_REJECT_*` 拒绝写入 —— 抽 ③ ⇒ 红
  C 四站点**已改路由**：都 `import` 共享模块，且文件内**不再有**字面 `max(0.5, min(…-0.5,…))`
    —— 抽 ② ⇒ 红
  D 扫描器**能抓新拷贝**（往临时目录注入一份 ⇒ 必须被发现）—— 抽 ④ ⇒ 红

运行（仓根）：python -X utf8 -m pytest tests/test_coord_contract_single_source_guard.py -q --no-header
"""
from __future__ import annotations

import math
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.world import coord_contract as C  # noqa: E402

WORLD = os.path.join(REPO, "lingshu", "world")
SITES = ("scene_simulator.py", "world_learner.py",
         "spacetime_consistency.py", "voxel_world.py")

NAN = float("nan")
INF = float("inf")
#: 系统性输入：边界值、刚好越界、±0.0、极小、NaN、±inf
GRID = (0.0, -0.0, 0.5, 0.4999999999, 0.5000000001, 1.0, 3.5, 9.5, 9.5000000001,
        10.0, -1.0, 1e-12, -1e-12, 1e12, NAN, INF, -INF)
SIZES = (4.0, 10.0, 1.0, 0.5)


def _same(a, b):
    """`repr` 级相同（区分 `0.0` 与 `-0.0`；NaN 与自身也算相同 —— 见 §同族坑）。"""
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return repr(a) == repr(b)


# ---------------------------------------------------------------- 原算式（oracle）
def _orig_clamp_xz(x, z, size):
    """`scene_simulator` / `voxel_world` 改前的字面算式（逐字抄自 main）。"""
    return (max(0.5, min(size - 0.5, x)), max(0.5, min(size - 0.5, z)))


def _orig_apply_move(pos, speed, d, size):
    """`world_learner._apply_move` 改前的字面算式（逐字抄自 main）。"""
    if d == (0.0, 0.0, 0.0):
        return tuple(round(v, 2) for v in pos)
    nx = max(0.5, min(size - 0.5, pos[0] + d[0] * speed))
    nz = max(0.5, min(size - 0.5, pos[2] + d[2] * speed))
    return (round(nx, 2), pos[1], round(nz, 2))


def _orig_apply_move_capped(pos, speed, d, size, dist):
    """`spacetime_consistency` 改前的字面算式（含步长封顶与 #347 非有限拒绝）。"""
    if d == (0.0, 0.0, 0.0):
        return tuple(round(v, 2) for v in pos)
    step_len = speed if dist is None else min(speed, dist)
    nx = pos[0] + d[0] * step_len
    nz = pos[2] + d[2] * step_len
    if not (math.isfinite(nx) and math.isfinite(nz)):
        return tuple(round(v, 2) for v in pos)
    nx = max(0.5, min(size - 0.5, nx))
    nz = max(0.5, min(size - 0.5, nz))
    return (round(nx, 2), pos[1], round(nz, 2))


def test_a_clamp_xz_与原算式逐位一致():
    """A1：`clamp_xz` 与改前字面算式在整个网格上 `repr` 级相同。"""
    n = 0
    for size in SIZES:
        for x in GRID:
            for z in GRID:
                got = C.clamp_xz(x, z, size)
                want = _orig_clamp_xz(x, z, size)
                assert _same(got, want), (
                    "clamp_xz(%r, %r, %r) = %r ≠ 原算式 %r" % (x, z, size, got, want))
                n += 1
    assert n >= 1000, "网格太小（%d），证不了" % n


def test_a2_apply_move_默认口径与原算式逐位一致():
    """A2：`apply_move` 默认参数 = `world_learner._apply_move` 改前口径。"""
    dirs = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.6, 0.0, 0.8),
            (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (NAN, 0.0, 0.0), (INF, 0.0, 0.0))
    for size in SIZES:
        for x in GRID:
            for z in GRID:
                pos = (x, 7.0, z)
                for d in dirs:
                    for speed in (0.0, 0.25, 1.0, 1e6, NAN, INF):
                        got = C.apply_move(pos, speed, d, size)
                        want = _orig_apply_move(pos, speed, d, size)
                        assert _same(got, want), (
                            "apply_move(%r,%r,%r,%r) = %r ≠ 原算式 %r"
                            % (pos, speed, d, size, got, want))


def test_a3_封顶与拒绝口径与原算式逐位一致():
    """A3：`step_cap` + `NF_REJECT_ROUND_ALL` = `spacetime_consistency` 改前口径。"""
    dirs = ((1.0, 0.0, 0.0), (0.6, 0.0, 0.8), (-1.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    for size in SIZES:
        for x in GRID:
            for z in GRID:
                pos = (x, 3.0, z)
                for d in dirs:
                    for speed, dist in ((1.0, None), (1.0, 0.3), (NAN, None),
                                        (INF, 0.5), (0.0, 0.0)):
                        got = C.apply_move(pos, speed, d, size, step_cap=dist,
                                           on_nonfinite=C.NF_REJECT_ROUND_ALL)
                        want = _orig_apply_move_capped(pos, speed, d, size, dist)
                        assert _same(got, want), (
                            "capped apply_move(%r,%r,%r,%r,cap=%r) = %r ≠ 原算式 %r"
                            % (pos, speed, d, size, dist, got, want))


def test_b_口径不得混用():
    """B：零方向与非有限的**两种**口径各自成立，且互不串味。"""
    pos = (1.234, 5.678, 9.012)
    zero = (0.0, 0.0, 0.0)
    all_q = C.apply_move(pos, 1.0, zero, 10.0, zero_shortcut=C.ZERO_ROUND_ALL)
    xz_q = C.apply_move(pos, 1.0, zero, 10.0, zero_shortcut=C.ZERO_XZ_ONLY)
    assert repr(all_q[1]) == repr(round(5.678, 2)), "ZERO_ROUND_ALL 应量化 y"
    assert repr(xz_q[1]) == repr(5.678), "ZERO_XZ_ONLY 不应量化 y"
    assert _same(all_q[0], xz_q[0]) and _same(all_q[2], xz_q[2])

    # `#347` 的真实形态：**起点有限、结果非有限**（方向含 inf/nan 或速度非有限）。
    #   · NF_NONE —— 旧口径：`max(0.5, min(9.5, inf))` ⇒ 静默抬成上边界 **9.5**；
    #     这是**既有**行为，此处**如实保留并断言出来**（改了它域内 trace 就会变）。
    #   · NF_REJECT_* —— 拒绝写入，回原位置（量化后仍有限）⇒ 不静默给一个假坐标。
    pos = (1.234, 5.678, 9.012)
    d_bad = (INF, 0.0, INF)
    silent = C.apply_move(pos, 1.0, d_bad, 10.0, on_nonfinite=C.NF_NONE)
    assert _same(silent, _orig_apply_move(pos, 1.0, d_bad, 10.0)), \
        "NF_NONE 必须复现旧口径 —— 若这里变了，域内 trace_sha256 会变"
    assert silent[0] == 9.5 and silent[2] == 9.5, \
        "旧口径下非有限结果应被静默抬到上边界 9.5（#347 的缺陷形态，如实保留）"
    rej = C.apply_move(pos, 1.0, d_bad, 10.0, on_nonfinite=C.NF_REJECT_ROUND_ALL)
    assert _same(rej, (1.23, 5.68, 9.01)), \
        "NF_REJECT_ROUND_ALL 应拒绝写入并回原位置（量化）：%r" % (rej,)
    assert all(math.isfinite(v) for v in rej), "拒绝写入后仍应是有限坐标"
    rej_xz = C.apply_move(pos, 1.0, d_bad, 10.0, on_nonfinite=C.NF_REJECT_XZ_ONLY)
    assert _same(rej_xz, (1.23, 5.678, 9.01)), \
        "NF_REJECT_XZ_ONLY 只量化 x/z、y 原样：%r" % (rej_xz,)


def test_c_四站点已改路由且不再自带字面算式():
    """C：四站点都 import 共享模块，且文件内不再出现字面 margin clamp。"""
    for fn in SITES:
        p = os.path.join(WORLD, fn)
        src = open(p, encoding="utf-8").read()
        assert "from .coord_contract import" in src, "%s 未改路由" % fn
        # 只看**代码行**（去掉注释），避免把我们写的说明注释当成实现
        code = "\n".join(ln for ln in src.splitlines()
                         if not ln.lstrip().startswith("#"))
        assert "max(0.5, min(" not in code, "%s 仍留有字面 margin clamp" % fn


def test_d_扫描器能抓新拷贝():
    """D：往临时目录注入一份**新拷贝** ⇒ `scan_clamp_copies` 必须发现。"""
    with tempfile.TemporaryDirectory() as td:
        sub = os.path.join(td, "pkg")
        os.makedirs(sub)
        with open(os.path.join(sub, "new_site.py"), "w", encoding="utf-8") as f:
            f.write("def step(pos, size):\n"
                    "    nx = max(0.5, min(size - 0.5, pos[0]))\n"
                    "    return nx\n")
        hits = [h for h in C.scan_clamp_copies(td)
                if h["tier"] in ("literal", "variable")]
        assert len(hits) == 1, "注入的拷贝没被扫出来：%r" % hits
    # 真实树：四个已改路由的站点必须**零**字面 clamp
    real = C.scan_clamp_copies(os.path.join(REPO, "lingshu"))
    for h in real:
        assert h["file"].split("/")[-1] not in SITES, \
            "%s 仍有 clamp 拷贝：%r" % (h["file"], h)
