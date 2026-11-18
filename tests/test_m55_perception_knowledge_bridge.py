# -*- coding: utf-8 -*-
"""守卫 · M5.5 感知-知识桥（条件卡灌入 / 图文四态回写 / 统一词典 / 白箱视觉问答）
====================================================================================
对照《docs/theory/白箱原生网络_长期路线_v1.md》M5.5 判据：
  1. 感知条件卡（p_threshold / margin）以四栏 ConditionSpace 灌入灵枢认知图；
  2. 图文融合四态判定（ACCEPT / DEFER / REJECT / BLINDSPOT）回写认知图，
     REJECT / BLINDSPOT 不得伪装成已确认事实；
  3. 知识库词典与感知词典统一（semantic_card_table 单一真源，零反向 import core）；
  4. 桥上白箱问答只读认知图节点并用中文感知词答出「图像里有什么」与位置/存在性查询；
  5. 非法输入（NaN/±inf/bool/空串/坏维度/非法 store）fail-loud 抛 ValueError；
  6. `--mutate` 定点变异自证（破坏四态过滤或词表映射 ⇒ 精确点红，还原全绿）。

运行：
  pytest tests/test_m55_perception_knowledge_bridge.py -v
  python -X utf8 tests/test_m55_perception_knowledge_bridge.py
  python -X utf8 tests/test_m55_perception_knowledge_bridge.py --mutate
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import os
import subprocess
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lingshu.core.core import MemoryLayer, SpacetimeMemoryEngine  # noqa: E402
from lingshu.nn.hex_cnn import image_to_grid  # noqa: E402
from lingshu.nn.hex_hier import (  # noqa: E402
    COLORS,
    HexHierNet,
    OBJ,
    POS,
    SHAPES,
    calibrate_hier_cards,
    train_hier,
)
from lingshu.nn.hex_text import (  # noqa: E402
    COLOR_CN,
    COLOR_WORDS,
    POS_CN,
    POS_WORDS,
    SHAPE_CN,
    SHAPE_WORDS,
    bridge_perceive_and_record,
    extract_perception_cards,
    ingest_perception_cards,
    make_multimodal_scene,
    query_visual_memory,
    record_multimodal_verdict,
    semantic_card_table,
)
from lingshu.nn.hex_train import normalize_lattices  # noqa: E402

HEX_TEXT_PATH = os.path.join(REPO, "lingshu", "nn", "hex_text.py")


def _build_trained_net_and_cards():
    """象限裁剪训练一个轻量 HexHierNet 并校准条件卡。"""
    rng = np.random.default_rng(11)
    xs, metas = [], []
    for _ in range(48):
        im, m = make_multimodal_scene(rng, 48, n_objects=1)
        xs.append(im)
        metas.append(m)
    feats = [image_to_grid(im, cells_across=16)[0] / 255.0 for im in xs]
    lat = normalize_lattices(np.stack(feats))
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
    crop_arr = np.stack(crops)
    lab_arr = {k: np.array(v) for k, v in labs.items()}
    net = HexHierNet(seed=7)
    train_hier(net, crop_arr, lab_arr, steps=45, samples_per_step=16, lr=0.1)
    cal = calibrate_hier_cards(net, crop_arr, lab_arr)
    return net, cal


@pytest.fixture(scope="module")
def bridge_fixture():
    return _build_trained_net_and_cards()


def test_semantic_card_table_unified_and_no_core_import():
    """M5.5a：感知与知识词典单一真源对齐，且 hex_text.py 零反向导入 core。"""
    table = semantic_card_table()
    assert set(table["shapes"]) == set(SHAPES)
    assert set(table["colors"]) == set(COLORS)
    assert set(table["positions"]) == set(POS)
    assert set(table["objects"]) == set(OBJ)
    for s in SHAPES:
        assert table["shapes"][s]["cn"] == SHAPE_CN[s]
        for w in table["shapes"][s]["words"]:
            assert SHAPE_WORDS[w] == s
    for c in COLORS:
        assert table["colors"][c]["cn"] == COLOR_CN[c]
        for w in table["colors"][c]["words"]:
            assert COLOR_WORDS[w] == c
    for p in POS:
        assert table["positions"][p]["cn"] == POS_CN[p]
        for w in table["positions"][p]["words"]:
            assert POS_WORDS[w] == p

    # AST 核验 nn/hex_text.py 不含任何对 lingshu.core 的静态导入
    with open(HEX_TEXT_PATH, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=HEX_TEXT_PATH)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "core" not in alias.name.split(".")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert "core" not in mod.split(".")


def test_ingest_perception_cards_with_calibrated_thresholds(bridge_fixture):
    """M5.5b：感知条件卡（p_threshold/margin）经 ConditionSpace 灌入认知图并可复读。"""
    net, cal = bridge_fixture
    rng = np.random.default_rng(21)
    im, _ = make_multimodal_scene(rng, 48, n_objects=1)
    lat = normalize_lattices((image_to_grid(im, cells_across=16)[0] / 255.0)[None, ...])

    engine = SpacetimeMemoryEngine(":memory:")
    res = ingest_perception_cards(
        engine, net, lat, image_id="scene_cal",
        p_threshold=0.20, min_margin=0.02,
        time_window=(10.0, 100.0), timestamp=50.0,
        calibrated_cards=cal,
    )
    assert len(res["node_ids"]) >= 1
    for nid in res["node_ids"]:
        node = engine.store.get_node(nid)
        assert node is not None
        assert node.condition_space.observation_tool == "hex_hier+hex_text.m55"
        assert node.condition_space.time_window == (10.0, 100.0)
        assert "p_threshold=" in node.condition_space.existence_constraint
        assert "margin=" in node.condition_space.existence_constraint
        assert "p_threshold" in node.state_attributes
        assert "margin" in node.state_attributes

    # 高门槛 p_threshold=0.99 ⇒ 退化为 BLINDSPOT，只能进 CONTEXT 不能混入 KNOWLEDGE
    engine_strict = SpacetimeMemoryEngine(":memory:")
    strict_res = ingest_perception_cards(
        engine_strict, net, lat, image_id="scene_strict",
        p_threshold=0.99, min_margin=0.50,
    )
    assert all(c["card_verdict"] == "BLINDSPOT" and not c["qualified"]
               for c in strict_res["cards"])
    for nid in strict_res["node_ids"]:
        assert engine_strict.store.get_node(nid).layer == MemoryLayer.CONTEXT
    q_strict = query_visual_memory(engine_strict, "图像里有什么", image_id="scene_strict")
    assert q_strict["found"] is False
    assert q_strict["rejected_filtered"] >= 1


def test_multimodal_verdict_writeback_and_reject_filtering(bridge_fixture):
    """M5.5c：图文四态回写认知图，REJECT 冲突描述进 CONTEXT 且绝不被问答当作事实。"""
    net, _ = bridge_fixture
    rng = np.random.default_rng(7)
    im, _ = make_multimodal_scene(rng, 48, n_objects=1)
    lat = normalize_lattices((image_to_grid(im, cells_across=16)[0] / 255.0)[None, ...])

    cards = extract_perception_cards(net, lat, p_threshold=0.20, min_margin=0.01)[0]
    top = cards[0]
    true_desc = f"{top['pos_cn']}有{top['color_cn']}{top['shape_cn']}"

    # 构造一个与感知结果完全冲突的描述（必判 REJECT）
    wrong_pos = next(p for p in POS if p != top["pos"])
    wrong_shape = next(s for s in SHAPES if s != top["shape"])
    wrong_color = next(c for c in COLORS if c != top["color"])
    false_desc = f"{POS_CN[wrong_pos]}有{COLOR_CN[wrong_color]}{SHAPE_CN[wrong_shape]}"

    engine = SpacetimeMemoryEngine(":memory:")
    out = bridge_perceive_and_record(
        engine, net, lat, text=true_desc, image_id="img_mix",
        p_threshold=0.20, min_margin=0.01,
    )
    assert out["fusion"]["fused"]["verdict"] in ("ACCEPT", "DEFER")
    assert out["fusion"]["qualified"] is True
    assert engine.store.get_node(out["fusion"]["node_id"]).layer == MemoryLayer.KNOWLEDGE

    rej = record_multimodal_verdict(engine, net, lat, false_desc, image_id="img_mix")
    assert rej["fused"]["verdict"] == "REJECT"
    assert rej["confirmed"] is False and rej["qualified"] is False
    rej_node = engine.store.get_node(rej["node_id"])
    assert rej_node.layer == MemoryLayer.CONTEXT
    assert "[图文冲突·REJECT·未确认]" in rej_node.content

    # 桥上白箱问答：答出真实感知，不把 REJECT 冲突声明当事实答出
    ans_all = query_visual_memory(engine, "图像里有什么", image_id="img_mix")
    assert ans_all["found"] is True
    assert top["shape_cn"] in ans_all["answer"]
    assert top["color_cn"] in ans_all["answer"]
    assert ans_all["rejected_filtered"] >= 1
    assert f"{COLOR_CN[wrong_color]}{SHAPE_CN[wrong_shape]}" not in ans_all["answer"]

    ans_false = query_visual_memory(engine, f"{false_desc}吗", image_id="img_mix")
    assert ans_false["found"] is False
    assert ans_false["rejected_filtered"] >= 1


def test_whitebox_vqa_queries_and_no_pixel_peeking(bridge_fixture):
    """M5.5d：支持多类中文白箱查询（有什么/在哪里/有没有/时间窗/直连 store）且不偷读像素。"""
    sig = inspect.signature(query_visual_memory)
    forbidden = {"img", "img_arr", "lat", "net", "labels", "labs", "oracle"}
    assert not (set(sig.parameters) & forbidden)

    net, _ = bridge_fixture
    rng = np.random.default_rng(31)
    im, _ = make_multimodal_scene(rng, 48, n_objects=2)
    lat = normalize_lattices((image_to_grid(im, cells_across=16)[0] / 255.0)[None, ...])

    engine = SpacetimeMemoryEngine(":memory:")
    # 直接传入 LayeredStore 也必须通过鸭子类型工作
    res = ingest_perception_cards(
        engine.store, net, lat, image_id="img_vqa",
        p_threshold=0.20, min_margin=0.01, time_window=(0.0, 50.0),
    )
    top = res["cards"][0]

    q_what = query_visual_memory(engine.store, "图像里有什么", image_id="img_vqa", time_at=25.0)
    assert q_what["found"] is True
    assert top["pos_cn"] in q_what["answer"]

    q_pos = query_visual_memory(engine, f"{top['pos_cn']}有什么", image_id="img_vqa")
    assert q_pos["found"] is True and top["shape_cn"] in q_pos["answer"]

    q_where = query_visual_memory(
        engine, f"{top['color_cn']}{top['shape_cn']}在哪里", image_id="img_vqa"
    )
    assert q_where["found"] is True and top["pos_cn"] in q_where["answer"]
    assert q_where["query_parsed"]["shape"] == top["shape"]

    q_exists = query_visual_memory(
        engine, f"有没有{top['color_cn']}{top['shape_cn']}", image_id="img_vqa"
    )
    assert q_exists["found"] is True and q_exists["answer"].startswith("有：")

    # 负例形状查询：同色异形绝不能把 top 错当命中（校验 SHAPE_WORDS 过滤生效）
    other_shapes = [s for s in SHAPES if all(c["shape"] != s for c in res["cards"])]
    if other_shapes:
        q_neg = query_visual_memory(
            engine, f"有没有{top['color_cn']}{SHAPE_CN[other_shapes[0]]}", image_id="img_vqa"
        )
        assert q_neg["found"] is False

    # 超出条件时间窗 ⇒ 不命中
    q_oob = query_visual_memory(engine, "图像里有什么", image_id="img_vqa", time_at=999.0)
    assert q_oob["found"] is False


def test_m55_fail_loud_invalid_inputs(bridge_fixture):
    """M5.5e：NaN/±inf/bool/空串/坏维度/非法 store 一律 fail-loud 抛 ValueError。"""
    net, _ = bridge_fixture
    engine = SpacetimeMemoryEngine(":memory:")
    good_lat = np.zeros((1, 9, 9, 3), dtype=np.float64)

    for bad_store in (None, True, 123, "bad_store", object()):
        with pytest.raises(ValueError):
            ingest_perception_cards(bad_store, net, good_lat)
        with pytest.raises(ValueError):
            query_visual_memory(bad_store, "图像里有什么")

    for bad_lat in (
        None,
        True,
        np.zeros((9, 9), dtype=np.float64),
        np.zeros((1, 2, 9, 3), dtype=np.float64),
        np.full((1, 9, 9, 3), np.nan),
        np.full((1, 9, 9, 3), np.inf),
        np.zeros((2, 9, 9, 3), dtype=np.float64),
    ):
        with pytest.raises(ValueError):
            ingest_perception_cards(engine, net, bad_lat)

    for kw in (
        {"p_threshold": True},
        {"p_threshold": float("nan")},
        {"p_threshold": 1.5},
        {"min_margin": -0.1},
        {"exist_mult": 0.0},
        {"image_id": ""},
        {"time_window": (10.0, 5.0)},
        {"timestamp": float("inf")},
    ):
        with pytest.raises(ValueError):
            ingest_perception_cards(engine, net, good_lat, **kw)

    with pytest.raises(ValueError):
        record_multimodal_verdict(engine, net, good_lat, "")
    with pytest.raises(ValueError):
        record_multimodal_verdict(engine, net, good_lat, "左上有红圆形", n_expected=True)
    with pytest.raises(ValueError):
        query_visual_memory(engine, "")
    with pytest.raises(ValueError):
        query_visual_memory(engine, "图像里有什么", allow_unconfirmed="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        query_visual_memory(engine, "图像里有什么", time_at=float("nan"))


# ── 定点变异自证 (--mutate) ───────────────────────────────────

_MUTATIONS = [
    (
        "M1_bypass_four_state_filter",
        "        if not qualified and not allow_unconfirmed:\n"
        "            rejected_filtered += 1\n"
        "            continue",
        "        if False:\n"
        "            rejected_filtered += 1\n"
        "            continue",
        {
            "test_ingest_perception_cards_with_calibrated_thresholds",
        },
    ),
    (
        "M2_bypass_fuse_reject_count",
        "        if not qualified and not allow_unconfirmed:\n"
        "            rejected_filtered += 1\n"
        "        verdicts.append({",
        "        if False:\n"
        "            rejected_filtered += 1\n"
        "        verdicts.append({",
        {
            "test_multimodal_verdict_writeback_and_reject_filtering",
        },
    ),
    (
        "M3_break_card_vocab_mapping",
        '    shape = next((SHAPE_WORDS[w] for w in SHAPE_WORDS if w in question), None)',
        '    shape = None',
        {
            "test_whitebox_vqa_queries_and_no_pixel_peeking",
        },
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
    with open(HEX_TEXT_PATH, "rb") as f:
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
                with open(HEX_TEXT_PATH, "w", encoding="utf-8", newline="\n") as wf:
                    wf.write(mutated)
            rc, failed, summary = _run_pytest_collect_failures()
            print(f"  {name:<30s} rc={rc} 期望红={len(expect_fail)} 实际红={len(failed)} {summary}")
            ok = expect_fail.issubset(failed) if expect_fail else (rc == 0 and not failed)
            if not ok:
                bad += 1
                print(f"    VERDICT=FAIL 漏红={sorted(expect_fail - failed)} 实际={sorted(failed)}")
            else:
                print("    VERDICT=PASS")
        finally:
            with open(HEX_TEXT_PATH, "wb") as wf:
                wf.write(orig_bytes)
            with open(HEX_TEXT_PATH, "rb") as rf:
                assert hashlib.sha256(rf.read()).hexdigest() == orig_sha, "hex_text.py 未原样还原"
    print(f"  OVERALL={'PASS' if bad == 0 else 'FAIL'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    if "--mutate" in sys.argv:
        sys.exit(_mutate())
    sys.exit(pytest.main([__file__, "-v"]))
