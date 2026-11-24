# -*- coding: utf-8 -*-
"""守卫 · R322 逐主体 8 域白箱求解器（lingshu/gen/hexgen_subject_split.py）
===========================================================================
覆盖 #11 声明阻塞在主体求解器上的全部 8 域：
  1. count / composition / pose / view_distance / occlusion / subject_priority
     六项源端可测域的定量读数与 ACCEPT / DEFER / BLINDSPOT 四态诚实判定；
  2. pose 各向同性退化（圆/方）必须诚实标 BLINDSPOT（orientation_deg=None），
     拉长形必须给出准确主轴朝向角且尺度无关；
  3. subject_priority 对称平局必须诚实降 DEFER，主次明确时准确排序；
  4. view_dir / dof 在 2D 平面合成图上诚实标 BLINDSPOT（不编造三维视角）；
  5. 非法输入（NaN/±inf/bool/坏图像维度）fail-loud 抛 ValueError；
  6. `--mutate` 定点变异自证（破坏各向同性门禁 / 优先级公式 / 2D BLINDSPOT ⇒ 必红）。

运行：
  pytest tests/test_hexgen_subject_domains.py -v
  python -X utf8 tests/test_hexgen_subject_domains.py
  python -X utf8 tests/test_hexgen_subject_domains.py --mutate
"""
from __future__ import annotations

import hashlib
import math
import os
import subprocess
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lingshu.gen.hexgen_subject_split import DOMAINS_8, analyze_subject_domains  # noqa: E402

SPLIT_PATH = os.path.join(REPO, "lingshu", "gen", "hexgen_subject_split.py")


def _canvas(h=256, w=256, bg=(20, 20, 20)):
    img = np.zeros((h, w, 3), dtype=np.float64)
    img[:, :] = bg
    return img


def _draw_circle(img, cy, cx, r, col):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    img[(xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2] = col


def _draw_ellipse(img, cy, cx, rx, ry, angle_deg, col):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    th = math.radians(angle_deg)
    dx, dy = xx - cx, yy - cy
    u = dx * math.cos(th) + dy * math.sin(th)
    v = -dx * math.sin(th) + dy * math.cos(th)
    img[(u / float(rx)) ** 2 + (v / float(ry)) ** 2 <= 1.0] = col


def test_eight_domains_complete_and_2d_unidentifiable_honest():
    """D1：输出完整覆盖 8 域；2D 平面图上 view_dir 与 dof 诚实标 BLINDSPOT。"""
    img = _canvas(256, 256)
    _draw_ellipse(img, 128, 128, 48, 16, 25.0, (220, 80, 60))
    res = analyze_subject_domains(img)
    assert set(res["domains"]) == set(DOMAINS_8)
    assert res["n_instances"] == 1
    assert res["domains"]["view_dir"]["verdict"] == "BLINDSPOT"
    assert res["domains"]["view_dir"]["measurable"] is False
    assert res["domains"]["view_dir"]["value"] is None
    assert res["domains"]["view_dir"]["reason"]
    assert res["domains"]["dof"]["verdict"] == "BLINDSPOT"
    assert res["domains"]["dof"]["measurable"] is False
    assert res["domains"]["dof"]["reason"]


def test_pose_elongated_orientation_vs_isotropic_blindspot():
    """D2：拉长形主轴角准确且尺度无关；圆/方各向同性诚实标 BLINDSPOT。"""
    # 30° 与 135° 拉长椭圆（256x256）
    for target_deg in (30.0, 135.0):
        img = _canvas(256, 256)
        _draw_ellipse(img, 128, 128, 54, 18, target_deg, (60, 200, 80))
        res = analyze_subject_domains(img)
        pose = res["domains"]["pose"]
        assert pose["verdict"] == "ACCEPT"
        sub = res["subjects"][0]
        assert sub["pose_verdict"] == "ACCEPT"
        assert sub["eccentricity"] >= 0.85
        diff = abs(sub["orientation_deg"] - target_deg)
        diff = min(diff, 180.0 - diff)
        assert diff <= 3.0, f"期望角度 {target_deg}°，实得 {sub['orientation_deg']}°"

    # 尺度无关：512x512 同比例 30° 椭圆主轴角与景别应一致
    img_big = _canvas(512, 512)
    _draw_ellipse(img_big, 256, 256, 108, 36, 30.0, (60, 200, 80))
    res_big = analyze_subject_domains(img_big)
    assert abs(res_big["subjects"][0]["orientation_deg"] - 30.0) <= 3.0
    assert res_big["domains"]["view_distance"]["shot_size"] == "medium"

    # 圆形与正方形各向同性 ⇒ pose 必须诚实 BLINDSPOT（orientation_deg=None）
    im_circle = _canvas(256, 256)
    _draw_circle(im_circle, 128, 128, 40, (220, 60, 60))
    res_c = analyze_subject_domains(im_circle)
    assert res_c["domains"]["pose"]["verdict"] == "BLINDSPOT"
    assert res_c["subjects"][0]["orientation_deg"] is None
    assert res_c["subjects"][0]["eccentricity"] < 0.25

    im_sq = _canvas(256, 256)
    im_sq[98:158, 98:158] = (60, 80, 220)
    res_sq = analyze_subject_domains(im_sq)
    assert res_sq["domains"]["pose"]["verdict"] == "BLINDSPOT"
    assert res_sq["subjects"][0]["orientation_deg"] is None


def test_composition_view_distance_and_subject_priority():
    """D3：景别分档、构图质心/左右平衡、主次优先级排序与对称平局 DEFER。"""
    # close / medium / far 景别（256x256，half_min_dim=128）
    for rad, want_shot in ((48, "close"), (22, "medium"), (10, "far")):
        im = _canvas(256, 256)
        _draw_circle(im, 127.5, 127.5, rad, (220, 120, 60))
        res = analyze_subject_domains(im)
        assert res["domains"]["view_distance"]["shot_size"] == want_shot
        assert res["domains"]["composition"]["lr_balance"] >= 0.95

    # 主次悬殊：中央大亮圆（index 0/1 中面积最大）应列第一主角且 ACCEPT
    im_prio = _canvas(256, 256)
    _draw_circle(im_prio, 128, 128, 42, (240, 220, 60))  # 大且居中高亮
    _draw_circle(im_prio, 40, 40, 14, (70, 90, 110))     # 小且角落低对比
    res_p = analyze_subject_domains(im_prio)
    assert res_p["n_instances"] == 2
    prio = res_p["domains"]["subject_priority"]
    assert prio["verdict"] == "ACCEPT"
    primary_sub = next(s for s in res_p["subjects"] if s["index"] == prio["primary_index"])
    assert primary_sub["area"] > 3000
    assert primary_sub["priority_rank"] == 1

    # 左右对称等大等对比双圆 ⇒ 平局，subject_priority 必须诚实降 DEFER
    im_tie = _canvas(256, 256)
    _draw_circle(im_tie, 128, 68, 28, (220, 70, 70))
    _draw_circle(im_tie, 128, 188, 28, (220, 70, 70))
    res_tie = analyze_subject_domains(im_tie)
    assert res_tie["n_instances"] == 2
    assert res_tie["domains"]["composition"]["lr_balance"] >= 0.95
    assert res_tie["domains"]["subject_priority"]["verdict"] == "DEFER"
    assert res_tie["domains"]["subject_priority"]["top_margin"] < 0.08


def test_occlusion_clean_vs_clipped_and_contacting():
    """D4：独立物体无遮挡（ACCEPT）；边缘截断与实例邻接接触诚实判疑似遮挡（DEFER）。"""
    im_clean = _canvas(256, 256)
    _draw_circle(im_clean, 128, 128, 36, (80, 200, 120))
    res_clean = analyze_subject_domains(im_clean)
    assert res_clean["domains"]["occlusion"]["verdict"] == "ACCEPT"
    assert res_clean["domains"]["occlusion"]["occluded_suspected"] is False

    # 画幅左缘截断
    im_clip = _canvas(256, 256)
    _draw_circle(im_clip, 128, 12, 35, (220, 160, 40))
    res_clip = analyze_subject_domains(im_clip)
    assert res_clip["domains"]["occlusion"]["verdict"] == "DEFER"
    assert res_clip["domains"]["occlusion"]["occluded_suspected"] is True
    assert res_clip["domains"]["occlusion"]["max_edge_clip_ratio"] >= 0.05

    # 双物体紧邻接触带（中心距 88，半径 50 与 36 -> 间隙 2px，落入膨胀接触带）
    im_contact = _canvas(256, 256)
    _draw_circle(im_contact, 128, 80, 50, (220, 60, 60))
    _draw_circle(im_contact, 128, 168, 36, (60, 90, 220))
    res_contact = analyze_subject_domains(im_contact)
    assert res_contact["n_instances"] == 2
    assert res_contact["domains"]["occlusion"]["verdict"] == "DEFER"
    assert res_contact["domains"]["occlusion"]["max_contact_ratio"] > 0.0


def test_zero_objects_blank_image_honest():
    """D5：纯背景无主体时不崩溃，count=0 且主体依赖域诚实标 BLINDSPOT。"""
    blank = _canvas(128, 128)
    res = analyze_subject_domains(blank)
    assert res["n_instances"] == 0
    assert res["domains"]["count"]["verdict"] == "ACCEPT"
    assert res["domains"]["count"]["n_instances"] == 0
    for dname in ("composition", "pose", "view_dir", "view_distance",
                  "occlusion", "dof", "subject_priority"):
        assert res["domains"][dname]["verdict"] == "BLINDSPOT"


@pytest.mark.parametrize("bad_img", [
    None,
    True,
    np.zeros((64, 64), dtype=np.float64),
    np.zeros((4, 64, 3), dtype=np.float64),
    np.zeros((64, 64, 4), dtype=np.float64),
    np.zeros((64, 64, 3), dtype=bool),
    np.full((32, 32, 3), np.nan),
    np.full((32, 32, 3), np.inf),
])
def test_fail_loud_invalid_images(bad_img):
    """D6a：坏图像维度/NaN/Inf/bool 必须 fail-loud 抛 ValueError。"""
    with pytest.raises(ValueError):
        analyze_subject_domains(bad_img)


@pytest.mark.parametrize("kwargs", [
    {"tol": True},
    {"tol": 0.0},
    {"tol": float("nan")},
    {"d": False},
    {"d": 0},
    {"min_px": True},
    {"iso_ecc_tol": 1.0},
    {"iso_ecc_tol": float("nan")},
    {"priority_tie_margin": -0.1},
    {"edge_clip_tol": 1.5},
    {"dof_grad_ratio_tol": 1.0},
])
def test_fail_loud_invalid_scalar_params(kwargs):
    """D6b：NaN/±inf/bool/越界标量参数必须 fail-loud 抛 ValueError。"""
    img = _canvas(64, 64)
    with pytest.raises(ValueError):
        analyze_subject_domains(img, **kwargs)


# ── 定点变异自证 (--mutate) ───────────────────────────────────

_MUTATIONS = [
    (
        "M1_strip_isotropic_blindspot_gate",
        "    if ecc < iso_ecc_tol:",
        "    if False:",
        {"test_pose_elongated_orientation_vs_isotropic_blindspot"},
    ),
    (
        "M2_corrupt_priority_formula",
        "        prio_score = 0.45 * area_norm + 0.35 * center_prox + 0.20 * contrast_norm",
        "        prio_score = 0.5",
        {"test_composition_view_distance_and_subject_priority"},
    ),
    (
        "M3_fabricate_2d_view_dir",
        '    dom_view_dir = {\n'
        '        "verdict": "BLINDSPOT",',
        '    dom_view_dir = {\n'
        '        "verdict": "ACCEPT",',
        {"test_eight_domains_complete_and_2d_unidentifiable_honest"},
    ),
    (
        "M4_baseline_intact",
        None,
        None,
        set(),
    ),
]


def _run_pytest_collect_failures():
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "pytest", __file__, "-q", "--no-header"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    failed = set()
    for line in out.splitlines():
        if line.startswith("FAILED ") and "::" in line:
            failed.add(line.split("::", 1)[1].split()[0])
    summary = [l for l in out.splitlines() if " passed" in l or " failed" in l]
    return proc.returncode, failed, (summary[-1] if summary else "")


def _mutate():
    with open(SPLIT_PATH, "rb") as f:
        orig_bytes = f.read()
    orig_sha = hashlib.sha256(orig_bytes).hexdigest()
    orig_text = orig_bytes.decode("utf-8")
    bad = 0
    for name, target, repl, expect_fail in _MUTATIONS:
        try:
            if target is not None:
                norm_text = orig_text.replace("\r\n", "\n")
                assert target in norm_text, f"变异锚点未找到: {target!r}"
                mutated = norm_text.replace(target, repl, 1)
                with open(SPLIT_PATH, "w", encoding="utf-8", newline="\n") as wf:
                    wf.write(mutated)
            rc, failed, summary = _run_pytest_collect_failures()
            print(f"  {name:<36s} rc={rc} 期望红={len(expect_fail)} 实际红={len(failed)} {summary}")
            ok = expect_fail.issubset(failed) if expect_fail else (rc == 0 and not failed)
            if not ok:
                bad += 1
                print(f"    VERDICT=FAIL 漏红={sorted(expect_fail - failed)} 实际={sorted(failed)}")
            else:
                print("    VERDICT=PASS")
        finally:
            with open(SPLIT_PATH, "wb") as wf:
                wf.write(orig_bytes)
            with open(SPLIT_PATH, "rb") as rf:
                assert hashlib.sha256(rf.read()).hexdigest() == orig_sha, "hexgen_subject_split.py 未原样还原"
    print(f"  OVERALL={'PASS' if bad == 0 else 'FAIL'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    if "--mutate" in sys.argv:
        sys.exit(_mutate())
    sys.exit(pytest.main([__file__, "-v"]))
