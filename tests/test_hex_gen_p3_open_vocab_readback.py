# -*- coding: utf-8 -*-
"""test_hex_gen_p3_open_vocab_readback · HEX-GEN P3 开放词汇像素独立回读与四态诚实闭环守卫。

覆盖范围：
  1. 既有契约保留：READBACK_COLORS == ("red", "green", "blue") 且默认 open_vocab=False
     行为逐位不变（orange 等开放色默认仍不进回读、verifiable=False、before/after=None）；
  2. 像素独立回读（11 色 × 9 形状 × 3 花纹 × 3 尺寸 @ 96px）：
     - extract_open_attributes 在 zone 窗口内按 _bg_level(img) 与目标色距离分割前景，
       并在连通包络内测量 pattern 与跨度归一化 size，准确率均 >= 85%；
     - 抹掉像素后读数跌为 area=0 / None（不偷看 parts/log），远端噪点不污染连通包络；
  3. 四态边界诚实（色差不可分绝不伪造 0 差异）：
     - clean_white 上 white、dark 上 black、默认噪声底上 black、128 灰底上 gray
       均记入 uncovered（含不可分原因与距离读数），全不可分时 verifiable=False、
       before/after=None、verify_readback 返回 None；混合场景如实披露部分覆盖；
  4. 入口参数闸：open_vocab 非 bool、size/rounds/supersample 混入 bool 或 NaN/±inf
     等非法输入一律 fail-loud 抛 ValueError；
  5. 自证变异（--mutate）：至少 4 条定点变异（不可分拦截 / 开放花纹回读 / open_vocab
     类型闸 / int 参数防 bool 闸），变异必红、还原后 sha256 逐字节一致且全绿。

运行：
  pytest tests/test_hex_gen_p3_open_vocab_readback.py -q --no-header
  python -X utf8 tests/test_hex_gen_p3_open_vocab_readback.py
  python -X utf8 tests/test_hex_gen_p3_open_vocab_readback.py --mutate
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lingshu.nn import hex_gen as G  # noqa: E402

HEX_GEN_PATH = REPO / "lingshu" / "nn" / "hex_gen.py"
RUN_TIMEOUT = 300


# ==================== 1. 既有契约保留 ====================

def test_p3_constants_and_default_legacy_contract_preserved():
    assert G.READBACK_COLORS == ("red", "green", "blue")
    assert G.OPEN_READBACK_COLORS == tuple(G.OPEN_COLORS.values())
    assert len(G.OPEN_READBACK_COLORS) == 11
    assert set(G.OPEN_SHAPES.values()) == {
        "circle", "triangle", "stripe", "square", "rectangle",
        "star", "heart", "hexagon", "diamond",
    }

    parts = [{"shape": "triangle", "color": "orange", "zone": "r4",
              "pattern": "dotted", "size": "medium"}]
    img, log = G.render_relations(parts, size=96, seed=0)
    vr = G.verify_readback(img, parts, log)
    assert vr["n_pattern"] == 0 and vr["pattern_acc"] is None
    assert vr["n_size"] == 0 and vr["size_acc"] is None

    rr = G.refine_render(parts, size=96, seed=0)
    assert rr["verifiable"] is False
    assert rr["before"] is None and rr["after"] is None
    assert rr["coverage"]["colors"] == ["red", "green", "blue"]
    assert rr["coverage"]["size_shape"] == "circle"


# ==================== 2. 像素独立回读（11 色 × 9 形状）====================

def test_p3_open_vocab_96px_accuracy_across_11_colors_and_9_shapes():
    pat_ok = pat_n = sz_ok = sz_n = 0
    for color in G.OPEN_READBACK_COLORS:
        bg = "clean_white" if color == "black" else "white"
        for shape in G.OPEN_SHAPES.values():
            for size_attr in ("small", "medium", "large"):
                for pat in ("solid", "striped", "dotted"):
                    parts = [{"shape": shape, "color": color, "zone": "r4",
                              "pattern": pat, "size": size_attr}]
                    img, log = G.render_relations(parts, size=96, seed=7, background=bg)
                    vr = G.verify_readback(img, parts, log, open_vocab=True)
                    pat_n += vr["n_pattern"]
                    if vr["pattern_acc"] is not None:
                        pat_ok += int(round(vr["pattern_acc"] * vr["n_pattern"]))
                    if vr["n_size"] > 0:
                        sz_n += vr["n_size"]
                        sz_ok += int(round(vr["size_acc"] * vr["n_size"]))
    pat_rate = pat_ok / pat_n
    sz_rate = sz_ok / sz_n
    assert pat_n == 11 * 9 * 3 * 3
    assert sz_n == 11 * 9 * 2 * 3
    assert pat_rate >= 0.85, f"开放词汇花纹回读准确率 {pat_rate:.3f} < 0.85"
    assert sz_rate >= 0.85, f"开放词汇尺寸回读准确率 {sz_rate:.3f} < 0.85"


def test_p3_pixel_independence_and_connected_envelope_isolation():
    parts = [{"shape": "hexagon", "color": "purple", "zone": "r4",
              "pattern": "striped", "size": "large"}]
    img, log = G.render_relations(parts, size=96, seed=7, noise=False)
    zone = log[0]["zone_landed"]
    read_real = G.extract_open_attributes(img, zone, "purple", shape_hint="hexagon")
    assert read_real["pattern"] == "striped"
    assert read_real["size"] == "large"
    assert read_real["separable"] is True

    # 抹掉全部前景像素 ⇒ 读数必须归零（绝不偷看 parts/log）
    erased = img.copy()
    bg = G._bg_level(img).astype(np.uint8)
    ref = np.array(G._part_rgb("purple"), dtype=int)
    mask = np.abs(erased.astype(int) - ref).sum(axis=2) < G._COLOR_TOL
    erased[mask] = bg
    read_erased = G.extract_open_attributes(erased, zone, "purple", shape_hint="hexagon")
    assert read_erased["area"] == 0
    assert read_erased["pattern"] is None and read_erased["size"] is None

    # 在 zone 远端角放入孤立同色噪点：连通包络必须将其剔除，不扭曲 span 与 size
    noisy = img.copy()
    qy, qx = G.zone_rc(zone)
    h = img.shape[0] // 3
    noisy[qy * h, qx * h] = np.array(G._part_rgb("purple"), dtype=np.uint8)
    read_noisy = G.extract_open_attributes(noisy, zone, "purple", shape_hint="hexagon")
    assert read_noisy["span"] == read_real["span"]
    assert read_noisy["pattern"] == "striped" and read_noisy["size"] == "large"


def test_p3_nondefault_triangle_pattern_preserves_default_outline():
    size, cx, cy, radius, period = 48, 24, 24, 7, 5
    base = np.zeros((size, size, 3), dtype=np.uint8)
    G._paint(base, "triangle", cx, cy, radius, (180, 20, 40), "solid")

    for pattern in ("striped", "dotted"):
        actual = np.zeros_like(base)
        G._paint_base_patterned(
            actual, "triangle", cx, cy, radius, (180, 20, 40), pattern, period)
        yy, xx = np.mgrid[0:size, 0:size]
        expected_mask = base[..., 0] != 0
        if pattern == "striped":
            expected_mask &= xx % period == 0
        else:
            expected_mask &= (xx % period == 0) & (yy % period == 0)
        expected = np.zeros_like(base)
        expected[expected_mask] = (180, 20, 40)
        assert np.array_equal(actual, expected), pattern


def test_p3_large_valid_target_zone_remains_separable():
    img = np.full((96, 96, 3), 255, dtype=np.uint8)
    ref = np.asarray(G._part_rgb("purple"), dtype=np.uint8)
    img[32:64, 32:64] = ref

    bg = G._bg_level(img)
    separable, distance = G._zone_color_separable(img, "r4", "purple", bg)
    assert separable is True and distance >= G._COLOR_TOL

    readback = G.extract_open_attributes(img, "r4", "purple", shape_hint="hexagon")
    assert readback["separable"] is True
    assert readback["area"] > 0.80 * 32 * 32


def test_p3_shape_span_normalization_covers_all_9_open_shapes():
    for shape in G.OPEN_SHAPES.values():
        for sz in ("small", "large"):
            parts = [{"shape": shape, "color": "yellow", "zone": "r4",
                      "pattern": "solid", "size": sz}]
            img, log = G.render_relations(parts, size=96, seed=7)
            res = G.extract_open_attributes(img, log[0]["zone_landed"], "yellow",
                                            shape_hint=shape)
            assert res["pattern"] == "solid", (shape, sz, res)
            assert res["size"] == sz, (shape, sz, res)


# ==================== 3. 四态边界诚实（色差不可分）====================

@pytest.mark.parametrize("bg_mode,color", [
    ("clean_white", "white"),
    ("dark", "black"),
    ("white", "black"),
    ("mid_gray_canvas", "gray"),
])
def test_p3_inseparable_background_is_honestly_uncovered_never_zero_diff(bg_mode, color):
    parts = [{"shape": "circle", "color": color, "zone": "r4",
              "pattern": "solid", "size": "medium"}]
    if bg_mode == "mid_gray_canvas":
        # 人工构造纯 128 灰底画布与同色灰圆（_color_separable == 0 < _COLOR_TOL）
        img = np.full((96, 96, 3), 128, dtype=np.uint8)
        log = [{"part": 0, "shape": "circle", "color": "gray",
                "zone_intended": "r4", "zone_landed": "r4",
                "pattern": "solid", "size": "medium",
                "pattern_period": 3, "center": (48, 48)}]
    else:
        img, log = G.render_relations(parts, size=96, seed=7, background=bg_mode)

    ea = G.extract_open_attributes(img, "r4", color, shape_hint="circle")
    assert ea["separable"] is False
    assert ea["pattern"] is None and ea["size"] is None
    assert ea["color_bg_dist"] < G._COLOR_TOL

    vr = G.verify_readback(img, parts, log, open_vocab=True)
    assert vr["n_pattern"] == 0 and vr["pattern_acc"] is None
    assert vr["n_size"] == 0 and vr["size_acc"] is None

    mm = G.readback_mismatches(img, parts, log, 96, open_vocab=True)
    assert mm["verifiable"] is False and mm["total"] is None
    assert mm["coverage"]["covered"] == []
    assert len(mm["coverage"]["uncovered"]) == 1
    u0 = mm["coverage"]["uncovered"][0]
    assert u0["part"] == 0 and u0["color"] == color
    assert "不可分" in u0["reason"]
    assert "color_bg_dist" in u0 and u0["color_bg_dist"] < G._COLOR_TOL

    if bg_mode != "mid_gray_canvas":
        rr = G.refine_render(parts, size=96, seed=7, open_vocab=True, background=bg_mode)
        assert rr["verifiable"] is False
        assert rr["before"] is None and rr["after"] is None
        assert rr["improved"] is False and rr["overrides"] == []
        assert np.array_equal(rr["image"], img)


def test_p3_partial_coverage_separable_plus_inseparable_discloses_and_refines():
    parts = [
        {"shape": "heart", "color": "orange", "zone": "r0",
         "pattern": "dotted", "size": "small"},
        {"shape": "circle", "color": "black", "zone": "r4",
         "pattern": "solid", "size": "medium"},
    ]
    off_img, _ = G.render_relations(parts, size=96, seed=7)
    rr = G.refine_render(parts, size=96, seed=7, open_vocab=True)
    assert rr["verifiable"] is True
    assert rr["before"] == 2 and rr["after"] == 0 and rr["improved"] is True
    assert rr["overrides"] == [{"part": 0, "render_params": {"pattern_period": 2}}]
    assert not np.array_equal(rr["image"], off_img)

    cov = rr["coverage"]
    assert cov["covered"] == [0] and cov["judged"] == [0]
    assert len(cov["uncovered"]) == 1
    assert cov["uncovered"][0]["part"] == 1 and cov["uncovered"][0]["color"] == "black"
    assert "不可分" in cov["uncovered"][0]["reason"]
    assert "不得外推" in cov["reason"]


def test_p3_end_to_end_generate_from_text_open_vocab_refine():
    text = "左上有橙色小点纹心形,右下有紫色大条纹六边形"
    on = G.generate_from_text(text, size=96, seed=7, open_vocab=True)
    off = G.generate_from_text(text, size=96, seed=7, open_vocab=True, refine_rounds=0)
    assert on["text"] == text
    assert on["parts"] == off["parts"]
    assert on["refine"]["verifiable"] is True
    assert on["refine"]["before"] == 2 and on["refine"]["after"] == 0
    assert on["refine"]["improved"] is True
    assert on["readback"]["pattern_acc"] == 1.0 and on["readback"]["n_pattern"] == 2
    assert on["readback"]["size_acc"] == 1.0 and on["readback"]["n_size"] == 2
    assert on["refine"]["coverage"]["colors"] == list(G.OPEN_READBACK_COLORS)

    # 48px 与 supersample!=1 在 open_vocab=True 下仍诚实拒绝修正
    r48 = G.generate_from_text(text, size=48, seed=7, open_vocab=True)
    assert r48["refine"]["verifiable"] is False
    assert r48["refine"]["before"] is None and r48["refine"]["after"] is None
    rss = G.generate_from_text(text, size=96, seed=7, supersample=2, open_vocab=True)
    assert rss["refine"]["verifiable"] is False
    assert rss["refine"]["before"] is None and rss["refine"]["after"] is None


# ==================== 4. 入口参数闸（fail-loud ValueError）====================

def test_p3_entry_parameter_gates_fail_loud():
    parts = [{"shape": "circle", "color": "orange", "zone": "r4",
              "pattern": "solid", "size": "medium"}]
    img, log = G.render_relations(parts, size=96, seed=7)

    # open_vocab 非 bool 一律 ValueError
    for bad_flag in (1, 0, "true", None):
        with pytest.raises(ValueError):
            G.verify_readback(img, parts, log, open_vocab=bad_flag)
        with pytest.raises(ValueError):
            G.readback_mismatches(img, parts, log, 96, open_vocab=bad_flag)
        with pytest.raises(ValueError):
            G.refine_render(parts, size=96, open_vocab=bad_flag)
        with pytest.raises(ValueError):
            G.generate_from_text("中间有橙色圆", size=96, open_vocab=bad_flag)

    # size / rounds / supersample 混入 bool 或非有限值 / 越界值
    for bad_pos in (True, False, float("nan"), float("inf"), float("-inf"), 0, -96, 96.5):
        with pytest.raises(ValueError):
            G.render_relations(parts, size=bad_pos)
        with pytest.raises(ValueError):
            G.refine_render(parts, size=bad_pos)
        with pytest.raises(ValueError):
            G.generate_from_text("中间有橙色圆", size=bad_pos)
        with pytest.raises(ValueError):
            G.refine_render(parts, size=96, supersample=bad_pos)
        with pytest.raises(ValueError):
            G.generate_from_text("中间有橙色圆", size=96, supersample=bad_pos)
        with pytest.raises(ValueError):
            G.readback_mismatches(img, parts, log, bad_pos)

    for bad_rounds in (True, False, float("nan"), float("inf"), -1, 1.5):
        with pytest.raises(ValueError):
            G.refine_render(parts, size=96, rounds=bad_rounds)
        with pytest.raises(ValueError):
            G.generate_from_text("中间有橙色圆", size=96, refine_rounds=bad_rounds)

    # extract_open_attributes 参数闸
    with pytest.raises(ValueError):
        G.extract_open_attributes(img, "r9", "orange")
    with pytest.raises(ValueError):
        G.extract_open_attributes(img, "r4", "cyan")
    with pytest.raises(ValueError):
        G.extract_open_attributes(img, "r4", "orange", calib={"tx_solid": True})
    with pytest.raises(ValueError):
        G.extract_open_attributes(img, "r4", "orange", calib={"area_cut": float("nan")})
    bad_img = img.astype(np.float64)
    bad_img[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        G.extract_open_attributes(bad_img, "r4", "orange")


# ==================== 5. --mutate 定点变异自证 ====================

_MUTATIONS = (
    (
        "M1_break_inseparable_guard",
        ("            if not is_sep:", "            if False:"),
        {
            "test_p3_inseparable_background_is_honestly_uncovered_never_zero_diff[clean_white-white]",
            "test_p3_inseparable_background_is_honestly_uncovered_never_zero_diff[dark-black]",
            "test_p3_inseparable_background_is_honestly_uncovered_never_zero_diff[white-black]",
            "test_p3_inseparable_background_is_honestly_uncovered_never_zero_diff[mid_gray_canvas-gray]",
            "test_p3_partial_coverage_separable_plus_inseparable_discloses_and_refines",
        },
    ),
    (
        "M2_break_open_vocab_pattern_readback",
        ("    if tx <= tx_solid:\n        pattern = \"solid\"\n",
         "    if tx <= 999.0:\n        pattern = \"solid\"\n"),
        {
            "test_p3_open_vocab_96px_accuracy_across_11_colors_and_9_shapes",
            "test_p3_pixel_independence_and_connected_envelope_isolation",
            "test_p3_partial_coverage_separable_plus_inseparable_discloses_and_refines",
            "test_p3_end_to_end_generate_from_text_open_vocab_refine",
        },
    ),
    (
        "M3_break_open_vocab_bool_gate",
        ("def _require_bool(val, name: str) -> bool:\n"
         "    \"\"\"入口 bool 参数闸：拒绝 int/str/None 等非 bool 类型（fail-loud）。\"\"\"\n"
         "    if not isinstance(val, bool):\n"
         "        raise ValueError(f\"{name} 必须是 bool，收到 {type(val).__name__}: {val!r}\")\n"
         "    return val",
         "def _require_bool(val, name: str) -> bool:\n"
         "    return bool(val)"),
        {
            "test_p3_entry_parameter_gates_fail_loud",
        },
    ),
    (
        "M4_break_int_param_bool_gate",
        ("    if isinstance(val, bool):\n"
         "        raise ValueError(f\"{name} 不能是 bool，收到 {val!r}\")",
         "    if False:\n"
         "        raise ValueError(f\"{name} 不能是 bool，收到 {val!r}\")"),
        {
            "test_p3_entry_parameter_gates_fail_loud",
        },
    ),
)


def _run_pytest_subprocess():
    cmd = [
        sys.executable, "-X", "utf8", "-m", "pytest",
        os.path.abspath(__file__), "-q", "--no-header", "-p", "no:cacheprovider",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=RUN_TIMEOUT,
    )
    out = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
    failed = set()
    for line in out.splitlines():
        m = re.match(r"^(?:FAILED|ERROR) .*::(\S+)", line)
        if m:
            failed.add(m.group(1))
    summary = [ln for ln in out.splitlines() if " passed" in ln or " failed" in ln]
    return proc.returncode, failed, (summary[-1] if summary else out)


def _mutate():
    orig = HEX_GEN_PATH.read_bytes()
    orig_sha = hashlib.sha256(orig).hexdigest()
    bad = 0
    for name, (old, new), expect in _MUTATIONS:
        try:
            text = orig.decode("utf-8")
            nl = "\r\n" if "\r\n" in text else "\n"
            norm = text.replace("\r\n", "\n")
            cnt = norm.count(old)
            if cnt != 1:
                print(f"[ANCHOR-MISS] {name}: expected 1 match, got {cnt}")
                bad += 1
                continue
            mutated = norm.replace(old, new, 1).replace("\n", nl).encode("utf-8")
            HEX_GEN_PATH.write_bytes(mutated)
            rc, failed, summary = _run_pytest_subprocess()
            hit = (rc != 0) and expect.issubset(failed)
            print(f"  {name:<40s} rc={rc} failed={len(failed)} expect_sub={len(expect)} ({summary})")
            if not hit:
                bad += 1
                print(f"    VERDICT=FAIL missing={sorted(expect - failed)}")
            else:
                print("    VERDICT=PASS")
        finally:
            HEX_GEN_PATH.write_bytes(orig)
            assert hashlib.sha256(HEX_GEN_PATH.read_bytes()).hexdigest() == orig_sha, (
                "hex_gen.py 未原样还原"
            )
    # 还原后复跑一次确认全绿
    rc, failed, summary = _run_pytest_subprocess()
    if rc != 0 or failed:
        bad += 1
        print(f"  RESTORE_CHECK VERDICT=FAIL rc={rc} failed={sorted(failed)}")
    else:
        print(f"  RESTORE_CHECK VERDICT=PASS ({summary})")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    if "--mutate" in sys.argv[1:]:
        sys.exit(_mutate())
    sys.exit(pytest.main([os.path.abspath(__file__), "-q", "--no-header"]))
