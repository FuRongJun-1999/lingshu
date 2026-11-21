# -*- coding: utf-8 -*-
"""test_m43_multi_object_occlusion · 路线图 M4.3 多物体与遮挡测试（含 --mutate 定点变异自证）
========================================================================================
对应《docs/theory/白箱原生网络_长期路线_v1.md》第 32–34 行 M4.3 里程碑与判据：
  · 2–4 物体 + 可控部分遮挡 / 子部件嵌套对抗场景；
  · 3 物体场景双位置+类别检出率稳定 ≥ 80%；
  · 重度遮挡与对抗遮挡块下四态诚实降级（DEFER / BLINDSPOT，或预期冲突时 REJECT），
    绝不把重度残缺/嵌套伪影高置信度虚报为错误 ACCEPT；
  · 全部新增入口对 NaN / ±inf / bool / 越界参数 fail-loud 抛 ValueError。

运行（设 PYTHONPATH 为仓库根）：
    python -X utf8 -m pytest tests/test_m43_multi_object_occlusion.py -v
    python -X utf8 tests/test_m43_multi_object_occlusion.py
    python -X utf8 tests/test_m43_multi_object_occlusion.py --mutate
"""
from __future__ import annotations

import os
import subprocess
import sys

_RP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RP not in sys.path:
    sys.path.insert(0, _RP)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lingshu.nn.hex_cnn import image_to_grid  # noqa: E402
from lingshu.nn.hex_hier import COLORS, OBJ, POS, SHAPES, HexHierNet, train_hier  # noqa: E402
from lingshu.nn.hex_search import (  # noqa: E402
    detect_multi_objects_m43,
    evaluate_m43_occlusion,
    make_multi_object_scene,
)
from lingshu.nn.hex_text import make_multimodal_scene  # noqa: E402

_HEX_SEARCH_PATH = os.path.join(_RP, "lingshu", "nn", "hex_search.py")
_CELLS_ACROSS = 21
_SCENE_SIZE = 96


def _encode_scenes(images, mu, sd):
    feats = np.stack([image_to_grid(im, _CELLS_ACROSS)[0] / 255.0 for im in images])
    return ((feats - mu) / sd).astype(np.float64)


@pytest.fixture(scope="module")
def trained_m43_bundle():
    """单物体象限裁剪训练一次（训练分布 = 象限推理分布），供全部 M4.3 场景测试复用。"""
    rng = np.random.default_rng(11)
    xs, metas = [], []
    for _ in range(72):
        im, meta = make_multimodal_scene(rng, _SCENE_SIZE, 1)
        xs.append(im)
        metas.append(meta)
    feats = np.stack([image_to_grid(im, _CELLS_ACROSS)[0] / 255.0 for im in xs])
    mu = float(feats.mean())
    sd = float(feats.std()) + 1e-8
    lat = ((feats - mu) / sd).astype(np.float64)

    r3, c3 = lat.shape[1] // 3, lat.shape[2] // 3
    crops, labs = [], {"shape": [], "color": [], "obj": [], "pos": []}
    for b, m in enumerate(metas):
        lb = m["labels"][0]
        qy, qx = divmod(int(lb["pos"][1]), 3)
        crops.append(lat[b, qy * r3:(qy + 1) * r3, qx * c3:(qx + 1) * c3])
        labs["shape"].append(lb["obj"].split("|")[0])
        labs["color"].append(lb["obj"].split("|")[1])
        labs["obj"].append(lb["obj"])
        labs["pos"].append("r4")

    net = HexHierNet(
        n_kernels=6, n_kernels2=6, stacked=True,
        deep_norm=True, amp_sep=True, amp_log=True, seed=7,
    )
    train_hier(
        net, np.stack(crops), {k: np.array(v) for k, v in labs.items()},
        steps=150, samples_per_step=20, lr=0.1, batch=72, seed=7,
    )
    return net, mu, sd


def _build_m43_batch(n_scenes, n_objects, seed, occlusion_ratio=0.0,
                     adversarial_occluder=False, mu=0.0, sd=1.0):
    rng = np.random.default_rng(seed)
    xs, metas = [], []
    for _ in range(n_scenes):
        im, meta = make_multi_object_scene(
            rng, n_objects=n_objects, size=_SCENE_SIZE,
            occlusion_ratio=occlusion_ratio,
            adversarial_occluder=adversarial_occluder,
        )
        xs.append(im)
        metas.append(meta)
    lat = _encode_scenes(xs, mu, sd)
    return lat, metas


def test_m43_scene_generator_2_to_4_objects_and_occlusion():
    """多物体（2–4）+ 可控遮挡/对抗遮挡生成器契约与确定性自检。"""
    for n_obj in (2, 3, 4):
        im0, m0 = make_multi_object_scene(
            np.random.default_rng(100 + n_obj), n_objects=n_obj,
            size=_SCENE_SIZE, occlusion_ratio=0.0,
        )
        im1, m1 = make_multi_object_scene(
            np.random.default_rng(100 + n_obj), n_objects=n_obj,
            size=_SCENE_SIZE, occlusion_ratio=0.0,
        )
        assert np.array_equal(im0, im1), "同种子生成结果必须确定性一致"
        assert m0 == m1
        assert im0.shape == (_SCENE_SIZE, _SCENE_SIZE, 3) and im0.dtype == np.uint8
        assert len(m0["labels"]) == n_obj
        assert len({lb["pos"] for lb in m0["labels"]}) == n_obj
        for lb in m0["labels"]:
            assert lb["shape"] in SHAPES and lb["color"] in COLORS
            assert lb["obj"] in OBJ and lb["pos"] in POS
            assert lb["visible_ratio"] == 1.0 and lb["occluded"] is False

    _, m_part = make_multi_object_scene(
        np.random.default_rng(7), n_objects=3, size=_SCENE_SIZE,
        occlusion_ratio=0.75, adversarial_occluder=True,
    )
    occ_lbs = [lb for lb in m_part["labels"] if lb["occluded"]]
    unocc_lbs = [lb for lb in m_part["labels"] if not lb["occluded"]]
    assert occ_lbs and unocc_lbs
    assert all(0.0 <= lb["visible_ratio"] < 0.55 for lb in occ_lbs)
    assert all(lb["visible_ratio"] == 1.0 for lb in unocc_lbs)

    _, m_full = make_multi_object_scene(
        np.random.default_rng(9), n_objects=2, size=_SCENE_SIZE,
        occlusion_ratio=1.0,
    )
    assert any(lb["occluded"] and lb["visible_ratio"] == 0.0 for lb in m_full["labels"])


def test_m43_three_object_detection_above_80_percent(trained_m43_bundle):
    """M4.3 硬指标：未遮挡 3 物体合成集双位置+类别检出率稳定 ≥ 80%，空槽零误报。"""
    net, mu, sd = trained_m43_bundle
    lat, metas = _build_m43_batch(
        n_scenes=30, n_objects=3, seed=42, occlusion_ratio=0.0, mu=mu, sd=sd,
    )
    rep = evaluate_m43_occlusion(net, lat, metas)
    assert rep["unoccluded_total"] == 90
    assert rep["unoccluded_pos_hit_rate"] >= 0.80, f"位置检出率不足 80%: {rep}"
    assert rep["unoccluded_obj_match_rate"] >= 0.80, f"位置+类别检出率不足 80%: {rep}"
    assert rep["empty_false_accepts"] == 0, f"空象限不得虚报 ACCEPT: {rep}"


def test_m43_splitting_stress_2_to_4_objects(trained_m43_bundle):
    """拆分管线压力测试：2、3、4 物体场景均保持高检出且空槽零误报。"""
    net, mu, sd = trained_m43_bundle
    for n_obj in (2, 3, 4):
        lat, metas = _build_m43_batch(
            n_scenes=20, n_objects=n_obj, seed=50 + n_obj,
            occlusion_ratio=0.0, mu=mu, sd=sd,
        )
        rep = evaluate_m43_occlusion(net, lat, metas)
        assert rep["unoccluded_obj_match_rate"] >= 0.80, (
            f"{n_obj} 物体场景检出率不足 80%: {rep}")
        assert rep["empty_false_accepts"] == 0, f"{n_obj} 物体空象限误报: {rep}"


def test_m43_heavy_neutral_occlusion_honest_four_state(trained_m43_bundle):
    """重度中性遮挡（0.75 / 0.90）：遮挡区域四态诚实降级，零错误 ACCEPT 虚报。"""
    net, mu, sd = trained_m43_bundle
    for occ in (0.75, 0.90):
        lat, metas = _build_m43_batch(
            n_scenes=30, n_objects=3, seed=42, occlusion_ratio=occ,
            adversarial_occluder=False, mu=mu, sd=sd,
        )
        rep = evaluate_m43_occlusion(net, lat, metas)
        assert rep["occluded_total"] > 0
        assert rep["occluded_false_accepts"] == 0, (
            f"中性遮挡 {occ} 存在错误 ACCEPT 虚报: {rep}")
        assert rep["occluded_honest_rate"] == 1.0, rep
        ov = rep["occluded_verdict_counts"]
        assert ov["DEFER"] + ov["BLINDSPOT"] > 0, f"重度遮挡应降级为 DEFER/BLINDSPOT: {rep}"
        assert rep["unoccluded_obj_match_rate"] >= 0.80, (
            f"同场景未遮挡物体检出受损: {rep}")
        assert rep["empty_false_accepts"] == 0, rep


def test_m43_adversarial_nested_occluder_no_false_accept(trained_m43_bundle):
    """对抗验证（遮挡=子部件嵌套）：反差色+内嵌子条纹遮挡块零 ACCEPT 虚报。"""
    net, mu, sd = trained_m43_bundle
    for occ in (0.75, 0.90):
        lat, metas = _build_m43_batch(
            n_scenes=30, n_objects=3, seed=42, occlusion_ratio=occ,
            adversarial_occluder=True, mu=mu, sd=sd,
        )
        rep = evaluate_m43_occlusion(net, lat, metas)
        assert rep["occluded_total"] > 0
        assert rep["occluded_accepts"] == 0, (
            f"子部件嵌套对抗遮挡 {occ} 出现虚报 ACCEPT: {rep}")
        assert rep["occluded_false_accepts"] == 0, rep
        assert rep["occluded_verdict_counts"]["DEFER"] == rep["occluded_total"], rep
        assert rep["empty_false_accepts"] == 0, rep
        assert rep["unoccluded_obj_match_rate"] >= 0.80, rep


def test_m43_expected_conflict_rejects_honestly(trained_m43_bundle):
    """与预期冲突时应诚实判 REJECT，而非强行 ACCEPT。"""
    net, mu, sd = trained_m43_bundle
    lat, metas = _build_m43_batch(
        n_scenes=8, n_objects=3, seed=42, occlusion_ratio=0.0, mu=mu, sd=sd,
    )
    # 构造与真值相反的 expected，验证预期冲突走 REJECT
    flipped_metas = []
    for m in metas:
        wrong_exp = {}
        new_lbs = []
        for lb in m["labels"]:
            other = next(o for o in OBJ if o != lb["obj"])
            wrong_exp[lb["pos"]] = other
            nlb = dict(lb)
            nlb["obj"] = other
            new_lbs.append(nlb)
        flipped_metas.append({"labels": new_lbs})

    res = detect_multi_objects_m43(net, lat[0:1], expected=wrong_exp)
    assert res["verdict"] == "REJECT"
    assert any(s["reason"] == "expected_conflict" for s in res["rejected"])

    rep = evaluate_m43_occlusion(net, lat, flipped_metas, use_expected=True)
    assert rep["unoccluded_obj_match_rate"] == 0.0
    assert rep["verdict_counts"]["REJECT"] > 0


def test_m43_empty_batch_and_invalid_inputs_fail_loud(trained_m43_bundle):
    """空批次安全返回；全部 NaN / ±inf / bool / 越界参数 fail-loud 抛 ValueError。"""
    net, mu, sd = trained_m43_bundle
    empty_lat = np.zeros((0, 24, 21, 3), dtype=np.float64)
    rep_empty = evaluate_m43_occlusion(net, empty_lat, [])
    assert rep_empty["n_scenes"] == 0 and rep_empty["occluded_false_accepts"] == 0

    rng = np.random.default_rng(1)
    for bad_n in (0, -1, 1, 5, True, False, 2.5, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            make_multi_object_scene(rng, n_objects=bad_n)

    for bad_occ in (-0.01, 1.01, True, False, float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            make_multi_object_scene(rng, n_objects=3, occlusion_ratio=bad_occ)

    for bad_sz in (10, 23, True, 96.0, float("nan")):
        with pytest.raises(ValueError):
            make_multi_object_scene(rng, n_objects=3, size=bad_sz)

    with pytest.raises(ValueError):
        make_multi_object_scene(None, n_objects=3)
    with pytest.raises(ValueError):
        make_multi_object_scene(rng, n_objects=3, adversarial_occluder="yes")

    lat_ok, metas_ok = _build_m43_batch(2, 3, seed=42, mu=mu, sd=sd)
    bad_lat = lat_ok[0:1].copy()
    bad_lat[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        detect_multi_objects_m43(net, bad_lat)

    inf_lat = lat_ok[0:1].copy()
    inf_lat[0, 0, 0, 0] = np.inf
    with pytest.raises(ValueError):
        detect_multi_objects_m43(net, inf_lat)

    with pytest.raises(ValueError):
        evaluate_m43_occlusion(net, lat_ok, metas_ok[:-1])
    with pytest.raises(ValueError):
        evaluate_m43_occlusion(net, np.where(np.isfinite(lat_ok), lat_ok, 0.0) * np.nan, metas_ok)
    with pytest.raises(ValueError):
        detect_multi_objects_m43(net, lat_ok[0:1], th_conf=float("nan"))
    with pytest.raises(ValueError):
        detect_multi_objects_m43(net, lat_ok[0:1], min_color_purity=True)
    with pytest.raises(ValueError):
        detect_multi_objects_m43(net, lat_ok[0:1], min_chroma_frac=0.8, max_chroma_frac=0.2)
    with pytest.raises(ValueError):
        detect_multi_objects_m43(None, lat_ok[0:1])


# ==================== --mutate 定点变异自证 ====================

_MUTATIONS = [
    (
        "M1_bypass_occlusion_honesty_gate",
        "elif (chroma_frac < mcf or chroma_frac > xcf\n"
        "              or color_purity < mcp or second_ratio > msr\n"
        "              or centroid_shift > mcs\n"
        "              or (mps is not None and proto_sim is not None and proto_sim < mps)\n"
        "              or conf < tc or margin < tm):",
        "elif chroma_frac < -1.0:",
        ("test_m43_heavy_neutral_occlusion_honest_four_state",
         "test_m43_adversarial_nested_occluder_no_false_accept"),
    ),
    (
        "M2_bypass_occlusion_ratio_gate",
        'occ_r = _validate_real(occlusion_ratio, "occlusion_ratio", lo=0.0, hi=1.0)',
        "occ_r = float(occlusion_ratio)",
        ("test_m43_empty_batch_and_invalid_inputs_fail_loud",),
    ),
]


def _run_plain():
    return pytest.main([os.path.abspath(__file__), "-q"])


def _run_mutate():
    with open(_HEX_SEARCH_PATH, "rb") as fh:
        raw_orig = fh.read()
    has_crlf = b"\r\n" in raw_orig
    src_lf = raw_orig.decode("utf-8").replace("\r\n", "\n")

    rc = 0
    for name, old_s, new_s, expect_tests in _MUTATIONS:
        hits = src_lf.count(old_s)
        if hits != 1:
            print(f"[ANCHOR-MISS] {name}  命中 {hits} 处（应为 1）")
            return 1
        mutated_lf = src_lf.replace(old_s, new_s)
        mutated_bytes = (mutated_lf.replace("\n", "\r\n") if has_crlf else mutated_lf).encode("utf-8")
        try:
            with open(_HEX_SEARCH_PATH, "wb") as fh:
                fh.write(mutated_bytes)
            proc = subprocess.run(
                [sys.executable, "-X", "utf8", "-m", "pytest",
                 os.path.abspath(__file__), "-q"],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", cwd=_RP,
            )
            out = (proc.stdout or "") + "\n" + (proc.stderr or "")
            red = proc.returncode != 0
        finally:
            with open(_HEX_SEARCH_PATH, "wb") as fh:
                fh.write(raw_orig)
        if not red:
            print(f"[MUTATE-FAIL] {name}  变异后测试仍绿——守卫无牙")
            rc = 1
        else:
            hit_names = [t for t in expect_tests if t in out]
            if not hit_names:
                print(f"[MUTATE-FAIL] {name}  虽转红但未点名预期用例 {expect_tests}:\n{out}")
                rc = 1
            else:
                print(f"[MUTATE-OK] {name}  精准点红：{','.join(hit_names)}")
    return rc


if __name__ == "__main__":
    if "--mutate" in sys.argv:
        sys.exit(_run_mutate())
    sys.exit(_run_plain())
