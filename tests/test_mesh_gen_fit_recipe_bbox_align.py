# -*- coding: utf-8 -*-
"""test_mesh_gen_fit_recipe_bbox_align · fit_recipe 剪影掩码共享包围盒对齐守卫
================================================================================
缺陷（改前实测，origin/main @ 4f0d55f）：
  `lingshu/world/mesh_gen.py::fit_recipe()` 在候选循环前按目标自身包围盒
  `(t_lo, t_hi)` 预投影 `t_masks`，循环中却将候选 `soup` 投影到目标与候选的
  并集包围盒 `bnd` 后，直接与 `t_masks[ax]` 计算 `mask_iou`——两张栅格掩码
  不在同一坐标系下：
  - 目标 `1×1×1` 盒，候选 `s=0.9`（真三轴 `profile_iou@32=0.765625`）vs `s=4.0`
    （真三轴 `profile_iou@32=0.0625`），`fit_recipe(..., fit_bbox=False)` 错选
    `s=4.0` 并谎报 `mask_score=1.0`，与其自身 `similarity["profile_iou"]`
    （`0.0625`）直接自相矛盾。
  - 位置偏置候选（与目标在 Y/Z 投影零重叠的 `disjoint_twins`、`dx=3.0` 偏置
    复合体）同样获得虚高 `mask_score`（`0.666667` / `0.528646` vs 真值
    `0.333333`），导致错选。

修法：
  当并集包围盒 `bnd != (t_lo, t_hi)` 时，在同一并集包围盒 `bnd` 下重新投影
  `target` 再与 `soup` 算三轴 `mask_iou`；当 `bnd == (t_lo, t_hi)` 时安全复用
  预投影缓存 `t_masks`，不引入额外投影开销。

运行：
  pytest:     python -X utf8 -m pytest tests/test_mesh_gen_fit_recipe_bbox_align.py -q
  直跑:       python -X utf8 tests/test_mesh_gen_fit_recipe_bbox_align.py
  变异自证:   python -X utf8 tests/test_mesh_gen_fit_recipe_bbox_align.py --mutate
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import re
import subprocess
import sys

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from lingshu.world import mesh_gen as MG  # noqa: E402


@contextmanager
def _temp_recipe(name: str, fn):
    """临时注册测试配方并在退出时彻底清理，避免污染全局配方表。"""
    had_recipe = name in MG.MESH_RECIPES
    old_recipe = MG.MESH_RECIPES.get(name)
    had_source = name in MG._RECIPE_SOURCES
    old_source = MG._RECIPE_SOURCES.get(name)
    MG.register_recipe(name, fn, source="test_guard", replace=True)
    try:
        yield name
    finally:
        if had_recipe:
            MG.MESH_RECIPES[name] = old_recipe
        else:
            MG.MESH_RECIPES.pop(name, None)
        if had_source:
            MG._RECIPE_SOURCES[name] = old_source
        else:
            MG._RECIPE_SOURCES.pop(name, None)


def _unit_box_soup():
    return MG.tri_soup(MG.prim_box((1.0, 1.0, 1.0)))


def _mean_profile_iou(sim_dict: dict) -> float:
    vals = sim_dict["profile_iou"].values()
    return sum(vals) / 3.0


def _ref_mean_profile_iou(cand_mesh, target_soup, res: int) -> float:
    a = MG.soup_recenter(MG.tri_soup(cand_mesh))
    b = MG.soup_recenter(target_soup)
    return round(sum(MG.profile_iou(a, b, ax, res=res) for ax in MG.AXES) / 3.0, 6)


def _make_disjoint_twins():
    """两个位于 x=-1.5 与 x=+1.5 的 1x1x1 盒，与居中 1x1x1 目标在 Y/Z 投影零重叠。"""
    return MG._compose("disjoint_twins", [
        (MG.prim_box((1.0, 1.0, 1.0)), (-1.5, 0.0, 0.0), None),
        (MG.prim_box((1.0, 1.0, 1.0)), (1.5, 0.0, 0.0), None),
    ])


def _make_asym_shifted(dx: float):
    """主 1x1x1 盒 + 沿 X 偏置 dx 的 0.2x0.2x0.2 附属块（偏置大时归心后主盒偏离原点）。"""
    return MG._compose("asym_shifted", [
        (MG.prim_box((1.0, 1.0, 1.0)), (0.0, 0.0, 0.0), None),
        (MG.prim_box((0.2, 0.2, 0.2)), (dx, 0.0, 0.0), None),
    ])


# ===========================================================================
# 守卫测试函数（供 pytest 发现，也供脚本直跑与 --mutate 复用）
# ===========================================================================

def test_a01_size_mismatch_selects_closer_scale():
    """A01：尺寸不匹配（各向同性 s=0.9 vs s=4.0 及单轴拉长），不得错选超大候选。"""
    target = _unit_box_soup()

    with _temp_recipe("_guard_scale_box",
                      lambda cfg: MG.prim_box((cfg["s"], cfg["s"], cfg["s"]))) as cat:
        res_fit = MG.fit_recipe(cat, target, grid={"s": [0.9, 4.0]},
                                res=32, fit_bbox=False)
        assert res_fit["config"] == {"s": 0.9}, (
            "s=0.9 应胜过 s=4.0，实得 %r" % (res_fit,))
        expected_09 = _ref_mean_profile_iou(MG.prim_box((0.9, 0.9, 0.9)), target, 32)
        assert res_fit["mask_score"] == expected_09, (
            "s=0.9 mask_score 期望 %s，实得 %s" % (expected_09, res_fit["mask_score"]))
        assert 0.70 < res_fit["mask_score"] < 0.85, res_fit["mask_score"]

        # 单测 s=4.0：不得谎报 1.0，必须与真实三轴 profile_iou=0.0625 一致
        res_big = MG.fit_recipe(cat, target, grid={"s": [4.0]},
                                res=32, fit_bbox=False)
        assert abs(res_big["mask_score"] - 0.0625) < 1e-6, (
            "s=4.0 真实 mask_score 应为 0.0625，实得 %s" % res_big["mask_score"])
        assert abs(_mean_profile_iou(res_big["similarity"]) - 0.0625) < 1e-6, (
            res_big["similarity"])

    # 单轴尺寸不匹配（仅 X 轴伸出 target 包围盒）
    with _temp_recipe("_guard_aniso_box",
                      lambda cfg: MG.prim_box((cfg["sx"], 1.0, 1.0))) as cat:
        res_aniso = MG.fit_recipe(cat, target, grid={"sx": [0.9, 4.0]},
                                  res=32, fit_bbox=False)
        assert res_aniso["config"] == {"sx": 0.9}, res_aniso
        res_stretch = MG.fit_recipe(cat, target, grid={"sx": [4.0]},
                                    res=32, fit_bbox=False)
        # X 投影=1.0，Y/Z 投影=0.25 ⇒ 均值 0.5（修前误报 1.0）
        assert abs(res_stretch["mask_score"] - 0.5) < 1e-6, (
            "sx=4.0 真实 mask_score 应为 0.5，实得 %s" % res_stretch["mask_score"])


def test_a02_position_offset_mismatch_rejects_disjoint_candidate():
    """A02：位置平移偏置候选（与目标在 Y/Z 投影零重叠）不得借跨坐标系假高分胜出。"""
    target = _unit_box_soup()

    def _twins_or_center(cfg):
        if cfg["mode"] == "centered_07":
            return MG.prim_box((0.7, 0.7, 0.7))
        return _make_disjoint_twins()

    with _temp_recipe("_guard_shift_twins", _twins_or_center) as cat:
        res_cmp = MG.fit_recipe(cat, target,
                                grid={"mode": ["centered_07", "disjoint_twins"]},
                                res=32, fit_bbox=False)
        assert res_cmp["config"] == {"mode": "centered_07"}, (
            "居中 0.7 盒应胜过零重叠双盒，实得 %r" % (res_cmp,))
        expected_07 = _ref_mean_profile_iou(MG.prim_box((0.7, 0.7, 0.7)), target, 32)
        assert res_cmp["mask_score"] == expected_07, res_cmp

        res_twins = MG.fit_recipe(cat, target, grid={"mode": ["disjoint_twins"]},
                                  res=32, fit_bbox=False)
        # 仅 X 投影重叠 (1.0)，Y/Z 投影零重叠 (0.0) ⇒ 均值 1/3
        assert abs(res_twins["mask_score"] - (1.0 / 3.0)) < 1e-5, (
            "disjoint_twins mask_score 应为 0.333333，实得 %s" % res_twins["mask_score"])
        assert res_twins["similarity"]["profile_iou"] == {"x": 1.0, "y": 0.0, "z": 0.0}

    def _asym_or_center(cfg):
        if cfg["mode"] == "centered_07":
            return MG.prim_box((0.7, 0.7, 0.7))
        return _make_asym_shifted(3.0)

    with _temp_recipe("_guard_shift_asym", _asym_or_center) as cat:
        res_asym_cmp = MG.fit_recipe(cat, target,
                                     grid={"mode": ["centered_07", "asym_dx3"]},
                                     res=32, fit_bbox=False)
        assert res_asym_cmp["config"] == {"mode": "centered_07"}, res_asym_cmp
        res_asym = MG.fit_recipe(cat, target, grid={"mode": ["asym_dx3"]},
                                 res=32, fit_bbox=False)
        assert abs(res_asym["mask_score"] - (1.0 / 3.0)) < 1e-5, (
            "asym dx=3.0 mask_score 应为 0.333333，实得 %s" % res_asym["mask_score"])


def test_a03_fit_bbox_true_vs_false_contrast():
    """A03：fit_bbox=True 与 fit_bbox=False 语义对照（含世界坐标平移输入）。"""
    # 将目标盒平移到世界坐标非原点，验证 soup_recenter 与 fit_bbox 开关协同工作
    shifted_target = MG.tri_soup(
        MG._translated(MG.prim_box((1.0, 1.0, 1.0)), (5.0, -3.0, 2.0))
    )

    with _temp_recipe("_guard_fit_bbox_contrast",
                      lambda cfg: MG._translated(
                          MG.prim_box((cfg["s"], cfg["s"], cfg["s"])),
                          (10.0, 0.0, -4.0),
                      )) as cat:
        r_true = MG.fit_recipe(cat, shifted_target, grid={"s": [0.9, 4.0]},
                               res=32, fit_bbox=True)
        assert r_true["fit_bbox"] is True
        assert abs(r_true["mask_score"] - 1.0) < 1e-6, (
            "fit_bbox=True 归一化后应满配 1.0，实得 %s" % r_true["mask_score"])
        assert r_true["similarity"]["profile_iou"] == {"x": 1.0, "y": 1.0, "z": 1.0}

        r_false = MG.fit_recipe(cat, shifted_target, grid={"s": [0.9, 4.0]},
                                res=32, fit_bbox=False)
        assert r_false["fit_bbox"] is False
        assert r_false["config"] == {"s": 0.9}, r_false
        assert r_false["mask_score"] < 1.0, r_false

    # 几何形状差异下 fit_bbox=True 仍正确区分形状（盒 vs 球，匹配目标盒）
    with _temp_recipe("_guard_shape_kind",
                      lambda cfg: MG.prim_box((2.0, 2.0, 2.0))
                      if cfg["kind"] == "box"
                      else MG.prim_sphere((2.0, 2.0, 2.0), seg_u=16, seg_v=10)) as cat:
        r_shape = MG.fit_recipe(cat, shifted_target,
                                grid={"kind": ["sphere", "box"]},
                                res=24, fit_bbox=True)
        assert r_shape["config"] == {"kind": "box"}, r_shape
        assert abs(r_shape["mask_score"] - 1.0) < 1e-6, r_shape


def test_a04_mask_score_matches_shape_similarity_same_res():
    """A04：fit_recipe 的 mask_score 与 shape_similarity.profile_iou 同分辨率 100% 一致。"""
    target = _unit_box_soup()
    cases = [
        ("s09", MG.prim_box((0.9, 0.9, 0.9))),
        ("s40", MG.prim_box((4.0, 4.0, 4.0))),
        ("aniso", MG.prim_box((4.0, 1.0, 1.0))),
        ("twins", _make_disjoint_twins()),
        ("asym06", _make_asym_shifted(0.6)),
        ("asym30", _make_asym_shifted(3.0)),
    ]
    mesh_map = dict(cases)
    with _temp_recipe("_guard_sim_align", lambda cfg: mesh_map[cfg["tag"]]) as cat:
        for tag, cand_mesh in cases:
            cand_soup = MG.tri_soup(cand_mesh)
            for res in (16, 32, 48):
                fit_out = MG.fit_recipe(cat, target, grid={"tag": [tag]},
                                        res=res, fit_bbox=False)
                # 1) 与未逐轴提前四舍五入的 profile_iou 均值精确相等
                ref_exact = _ref_mean_profile_iou(cand_mesh, target, res)
                assert fit_out["mask_score"] == ref_exact, (
                    "tag=%s res=%d: mask_score=%s != ref=%s"
                    % (tag, res, fit_out["mask_score"], ref_exact))
                # 2) 与 shape_similarity(mask_res=res) 三轴均值一致（允许逐轴 6 位舍入差 <=1e-6）
                sim_same = MG.shape_similarity(cand_soup, target,
                                               res=min(res, 24), mask_res=res)
                sim_mean = _mean_profile_iou(sim_same)
                assert abs(fit_out["mask_score"] - sim_mean) <= 1e-6, (
                    "tag=%s res=%d: mask_score=%s != shape_similarity mean=%s"
                    % (tag, res, fit_out["mask_score"], sim_mean))
                # 3) res=48 时与 fit_recipe 自带 similarity（默认 mask_res=48）直接一致
                if res == 48:
                    embedded_mean = _mean_profile_iou(fit_out["similarity"])
                    assert abs(fit_out["mask_score"] - embedded_mean) <= 1e-6, (
                        "tag=%s res=48: mask_score=%s != fit_out['similarity'] mean=%s"
                        % (tag, fit_out["mask_score"], embedded_mean))


def test_a05_same_bbox_reuses_cached_target_masks():
    """A05：当 bnd == target_bnd 时复用预计算 t_masks，不得重复投影 target。"""
    target = _unit_box_soup()
    orig_project = MG.project_mask
    calls = []

    def _counting_project(soup, axis="y", res=48, bounds=None):
        calls.append((axis, int(res), bounds))
        return orig_project(soup, axis, res, bounds)

    # 使用 res=20（区别于 shape_similarity 默认 mask_res=48），精确隔离 fit_recipe 阶段调用
    test_res = 20
    with _temp_recipe("_guard_cache_reuse",
                      lambda cfg: MG.prim_box((cfg["s"], cfg["s"], cfg["s"]))) as cat:
        MG.project_mask = _counting_project
        try:
            # 场景 1：fit_bbox=False 且所有候选落在 target 包围盒内（4 个候选均 bnd == t_bnd）
            calls.clear()
            r1 = MG.fit_recipe(cat, target, grid={"s": [0.5, 0.7, 0.9, 1.0]},
                               res=test_res, fit_bbox=False)
            assert r1["config"] == {"s": 1.0} and abs(r1["mask_score"] - 1.0) < 1e-6
            n_fit1 = sum(1 for _ax, r, _b in calls if r == test_res)
            # 期望 = 3（预投影 target）+ 4 * 3（4 个候选）= 15；若未复用则为 27
            assert n_fit1 == 15, (
                "bnd==t_bnd 应复用 t_masks（期望 15 次 project_mask，实测 %d）" % n_fit1)

            # 场景 2：fit_bbox=True（3 个不同尺度候选均归一到 target 包围盒，均 bnd == t_bnd）
            calls.clear()
            r2 = MG.fit_recipe(cat, target, grid={"s": [0.5, 2.0, 4.0]},
                               res=test_res, fit_bbox=True)
            assert abs(r2["mask_score"] - 1.0) < 1e-6
            n_fit2 = sum(1 for _ax, r, _b in calls if r == test_res)
            # 期望 = 3 + 3 * 3 = 12；若未复用则为 21
            assert n_fit2 == 12, (
                "fit_bbox=True 应复用 t_masks（期望 12 次 project_mask，实测 %d）" % n_fit2)
        finally:
            MG.project_mask = orig_project


def test_a06_mixed_bbox_reprojects_only_expanded_candidates():
    """A06：混合网格（1 个内缩 + 1 个越界）仅对越界候选在共享 bnd 下重投影 target。"""
    target = _unit_box_soup()
    orig_project = MG.project_mask
    calls = []

    def _counting_project(soup, axis="y", res=48, bounds=None):
        calls.append((axis, int(res), bounds))
        return orig_project(soup, axis, res, bounds)

    test_res = 20
    with _temp_recipe("_guard_mixed_reuse",
                      lambda cfg: MG.prim_box((cfg["s"], cfg["s"], cfg["s"]))) as cat:
        MG.project_mask = _counting_project
        try:
            calls.clear()
            r = MG.fit_recipe(cat, target, grid={"s": [0.9, 4.0]},
                              res=test_res, fit_bbox=False)
            assert r["config"] == {"s": 0.9}, r
            n_fit = sum(1 for _ax, res_val, _b in calls if res_val == test_res)
            # 期望 = 3（预投影）+ 3（s=0.9 候选，复用 target）+ 6（s=4.0 候选与 target 同 bnd）= 12
            # 未重投影（旧缺陷）= 9；未复用缓存 = 15
            assert n_fit == 12, (
                "混合网格应调用 12 次 project_mask（实测 %d）" % n_fit)
        finally:
            MG.project_mask = orig_project


# ===========================================================================
# 脚本直跑 & --mutate 定点变异自证
# ===========================================================================

_ORDERED_CHECKS = (
    ("A01", "尺寸不匹配选近尺度且拒绝超大盒谎报", test_a01_size_mismatch_selects_closer_scale),
    ("A02", "位置平移偏置候选拒绝零重叠假高分", test_a02_position_offset_mismatch_rejects_disjoint_candidate),
    ("A03", "fit_bbox=True/False 对照与世界坐标平移", test_a03_fit_bbox_true_vs_false_contrast),
    ("A04", "mask_score 与 shape_similarity 同分辨率一致", test_a04_mask_score_matches_shape_similarity_same_res),
    ("A05", "bnd == t_bnd 快速复用缓存不回归", test_a05_same_bbox_reuses_cached_target_masks),
    ("A06", "混合网格仅对越界候选在共享 bnd 重投影", test_a06_mixed_bbox_reprojects_only_expanded_candidates),
)


def _run_assertions() -> int:
    fails = []
    for cid, desc, fn in _ORDERED_CHECKS:
        try:
            fn()
        except AssertionError as ex:
            fails.append(cid)
            print("  FAIL %s %s  %s" % (cid, desc, ex))
        except Exception as ex:  # noqa: BLE001 - 守卫直跑兜底记录意外异常
            fails.append(cid)
            print("  FAIL %s %s  %s: %s" % (cid, desc, type(ex).__name__, ex))
        else:
            print("  PASS %s %s" % (cid, desc))
    print("通过 %d / 失败 %d" % (len(_ORDERED_CHECKS) - len(fails), len(fails)))
    return len(fails)


_ARMS = {
    # M1：回退到旧缺陷（无论 bnd 是否扩展，始终复用 t_masks）
    "M1": (
        "lingshu/world/mesh_gen.py",
        [(
            "        t_cur = t_masks if bnd == t_bnd else {\n"
            "            ax: project_mask(target, ax, res, bnd) for ax in AXES\n"
            "        }",
            "        t_cur = t_masks",
        )],
        {"A01", "A02", "A03", "A04", "A06"},
    ),
    # M2：破坏 bnd == t_bnd 缓存复用分支（每次都重投影 target）
    "M2": (
        "lingshu/world/mesh_gen.py",
        [(
            "        t_cur = t_masks if bnd == t_bnd else {",
            "        t_cur = t_masks if False else {",
        )],
        {"A05", "A06"},
    ),
    # M3：候选投影丢弃共享 bnd（退回各自独立坐标系）
    "M3": (
        "lingshu/world/mesh_gen.py",
        [(
            "            score += mask_iou(project_mask(soup, ax, res, bnd), t_cur[ax])",
            "            score += mask_iou(project_mask(soup, ax, res), t_cur[ax])",
        )],
        {"A01", "A02", "A03", "A04", "A05"},
    ),
}


def _subprocess_fail_names():
    for cache in _REPO.rglob("__pycache__"):
        for pyc in cache.glob("mesh_gen.*.pyc"):
            pyc.unlink(missing_ok=True)
    r = subprocess.run(
        [sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--assert-only"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(_REPO),
        timeout=300,
    )
    out = (r.stdout or "") + (r.stderr or "")
    return set(re.findall(r"(?m)^\s*FAIL\s+(A\d+)", out)), out


def _mutate(argv) -> int:
    want = ""
    for arg in argv:
        if arg == "--mutate":
            want = want or "ALL"
        elif arg.startswith("--mutate="):
            want = arg.split("=", 1)[1]
    if not want:
        return 0
    ids = sorted(_ARMS) if want == "ALL" else [want]
    red = 0
    for mid in ids:
        if mid not in _ARMS:
            print("[ANCHOR-MISS] %s" % mid)
            return 2
        rel, edits, expect = _ARMS[mid]
        target = _REPO / rel
        orig = target.read_bytes()
        try:
            text = orig.decode("utf-8")
            nl = "\r\n" if "\r\n" in text else "\n"
            norm = text.replace("\r\n", "\n")
            for old, new in edits:
                cnt = norm.count(old)
                if cnt != 1:
                    print("[ANCHOR-MISS] %s: count=%d" % (mid, cnt))
                    return 2
                norm = norm.replace(old, new)
            target.write_bytes(norm.replace("\n", nl).encode("utf-8"))
            actual, out = _subprocess_fail_names()
            if actual == expect:
                print("[MUTATE-OK]   %s: %s" % (mid, sorted(actual)))
            else:
                print("[MUTATE-FAIL] %s: 期望 %s 实际 %s"
                      % (mid, sorted(expect), sorted(actual)))
                print(out)
                red += 1
        finally:
            target.write_bytes(orig)
            for cache in _REPO.rglob("__pycache__"):
                for pyc in cache.glob("mesh_gen.*.pyc"):
                    pyc.unlink(missing_ok=True)
    return 1 if red else 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--assert-only" in argv:
        return 1 if _run_assertions() else 0
    if any(a == "--mutate" or a.startswith("--mutate=") for a in argv):
        return _mutate(argv)
    return 1 if _run_assertions() else 0


if __name__ == "__main__":
    raise SystemExit(main())
