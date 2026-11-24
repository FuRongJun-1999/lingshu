# -*- coding: utf-8 -*-
"""test_m5_hanzi_radical_composition · M5 汉字六书部件条件组合卡闭环守卫与变异自证
=====================================================================================
覆盖《白箱原生网络_长期路线_v1.md》第 36–40 行 M5 验收判据：
  1. 零样本「休=人+木」六书组合：仅在 19 个独体部件上校准蜂窝晶格 HexConv 条件卡
     （`trained_on_compounds=False`），对未见合体字「休」按左右槽位拆出 `(⿰, 人, 木) -> 休`；
  2. 24+ 双部件样本字（含带平移/笔画粗细/缩放扰动的留出集）实测部件判定率与整字判定率
     均稳定 `>= 70%`；
  3. 四态诚实：未登记假合体字（`⿰火十`）与全黑空图诚实报 `BLINDSPOT`，置信/边距不足
     报 `DEFER`，图文/声明冲突报 `REJECT`，命中报 `ACCEPT`；
  4. 全入口 `NaN` / `±inf` / `bool` / 非法 IDS 表达式 / 坏图像尺寸 fail-loud 抛 `ValueError`；
  5. `--mutate` 定点变异自证（破坏槽位方向、破坏部件特征卡或绕过参数门禁均精确点红）。

运行：
  python -X utf8 tests/test_m5_hanzi_radical_composition.py
  python -X utf8 tests/test_m5_hanzi_radical_composition.py --mutate
  python -X utf8 -m pytest tests/test_m5_hanzi_radical_composition.py -q
"""
from __future__ import annotations

import argparse
import ast
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lingshu.nn.hex_unit import (
    HANZI_IDS_DICT,
    HANZI_VERDICTS,
    HanziRadicalCardBank,
    RADICAL_VOCAB,
    evaluate_hanzi_m5,
    parse_ids,
    recognize_hanzi,
    render_hanzi,
    render_ids,
    render_radical,
)


def _assert_raises_value_error(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError("期望抛出 ValueError，却静默通过：%s" % getattr(fn, "__name__", fn))


def test_m5_zero_shot_xiu_and_24plus_sample_accuracy():
    assert len(RADICAL_VOCAB) >= 12
    assert len(HANZI_IDS_DICT) >= 24
    assert set(HANZI_VERDICTS) == {"ACCEPT", "DEFER", "BLINDSPOT", "REJECT"}

    bank = HanziRadicalCardBank()
    assert bank.trained_on_compounds is False

    # 1. 「休=人+木」零样本六书组合原生实证
    xiu_img = render_hanzi("休", size=64)
    res_xiu = recognize_hanzi(xiu_img, bank=bank)
    assert res_xiu["trained_on_compounds"] is False
    assert res_xiu["verdict"] == "ACCEPT"
    assert res_xiu["op"] == "⿰"
    assert res_xiu["radicals"] == ("人", "木")
    assert res_xiu["ids"] == "⿰人木"
    assert res_xiu["hanzi"] == "休"

    # 2. 上下结构正例「昌=⿱日日」「思=⿱田心」
    res_chang = recognize_hanzi(render_hanzi("昌", size=64), bank=bank)
    assert res_chang["verdict"] == "ACCEPT" and res_chang["hanzi"] == "昌"
    assert res_chang["op"] == "⿱" and res_chang["radicals"] == ("日", "日")

    res_si = recognize_hanzi(render_hanzi("思", size=64), bank=bank)
    assert res_si["verdict"] == "ACCEPT" and res_si["hanzi"] == "思"
    assert res_si["op"] == "⿱" and res_si["radicals"] == ("田", "心")

    # 3. 34 个双部件样本字（标准 + 轻微平移/粗细/缩放扰动留出集）均 >= 70%
    clean_eval = evaluate_hanzi_m5(bank)
    assert clean_eval["total_chars"] >= 24
    assert clean_eval["radical_accuracy"] >= 0.70, clean_eval
    assert clean_eval["char_accuracy"] >= 0.70, clean_eval

    held_out = list(HANZI_IDS_DICT.keys())[:24]
    pert_eval = evaluate_hanzi_m5(
        bank, characters=held_out, shift=(1.5, -1.0), stroke_width=2.8, scale=0.92
    )
    assert pert_eval["total_chars"] == 24
    assert pert_eval["radical_accuracy"] >= 0.70, pert_eval
    assert pert_eval["char_accuracy"] >= 0.70, pert_eval


def test_m5_four_state_honesty_and_import_purity():
    bank = HanziRadicalCardBank()

    # BLINDSPOT：未登记的假合体字（部品本身合法，但组合不在 IDS 词典中）
    fake_img = render_ids("⿰火十", size=64)
    res_fake = recognize_hanzi(fake_img, bank=bank)
    assert res_fake["verdict"] == "BLINDSPOT"
    assert res_fake["hanzi"] is None
    assert res_fake["ids"] == "⿰火十"

    # BLINDSPOT：全黑空图像
    blank = np.zeros((64, 64, 3), dtype=np.uint8)
    res_blank = recognize_hanzi(blank, bank=bank)
    assert res_blank["verdict"] == "BLINDSPOT"

    # REJECT：声明整字与图上部件组合冲突
    res_conflict = recognize_hanzi(render_hanzi("明", size=64), bank=bank, declared_hanzi="休")
    assert res_conflict["verdict"] == "REJECT"
    assert res_conflict["reason"] == "declared_hanzi_conflict"

    # DEFER：高边距/高置信要求下诚实降级为待定（不虚报 ACCEPT）
    res_defer = recognize_hanzi(
        render_hanzi("休", size=64), bank=bank, p_threshold=0.9999, margin_threshold=0.5
    )
    assert res_defer["verdict"] == "DEFER"

    # AST 断言 hex_unit.py 零反向导入 lingshu.core / lingshu.world
    unit_path = os.path.join(REPO, "lingshu", "nn", "hex_unit.py")
    tree = ast.parse(open(unit_path, encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for al in node.names:
                assert "core" not in al.name and "world" not in al.name
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert "core" not in mod and "world" not in mod, mod


def test_m5_nonfinite_bool_and_ids_fail_closed():
    # 非法 IDS 表达式
    for bad_expr in ("", "⿰", "⿱人", "⿻人木", "⿰人？", 123, None):
        _assert_raises_value_error(parse_ids, bad_expr)
        _assert_raises_value_error(render_ids, bad_expr)

    for bad_hz in ("", "我", "⿰人木", None, 42):
        _assert_raises_value_error(render_hanzi, bad_hz)

    for bad_rad in ("", "水", "休", None):
        _assert_raises_value_error(render_radical, bad_rad)

    # 非有限 / bool 参数门禁
    for bad_num in (float("nan"), float("inf"), float("-inf"), True, False, -1.0, 0.0):
        _assert_raises_value_error(render_radical, "人", stroke_width=bad_num)
        _assert_raises_value_error(render_radical, "人", scale=bad_num)
        _assert_raises_value_error(HanziRadicalCardBank, p_threshold=bad_num)
    for bad_int in (True, False, 16.5, 4):
        _assert_raises_value_error(render_radical, "人", size=bad_int)
        _assert_raises_value_error(HanziRadicalCardBank, cells_across=bad_int)
    _assert_raises_value_error(render_radical, "人", shift=(float("nan"), 0.0))
    _assert_raises_value_error(render_radical, "人", shift=(True, 0.0))

    # 坏图像输入
    bank = HanziRadicalCardBank()
    _assert_raises_value_error(recognize_hanzi, None, bank=bank)
    _assert_raises_value_error(recognize_hanzi, np.zeros((32, 32), dtype=bool), bank=bank)
    _assert_raises_value_error(recognize_hanzi, np.zeros((8, 8, 3), dtype=np.uint8), bank=bank)
    bad_img = render_hanzi("休", size=64).astype(np.float32)
    bad_img[10, 10, 0] = float("nan")
    _assert_raises_value_error(recognize_hanzi, bad_img, bank=bank)
    bad_img[10, 10, 0] = float("inf")
    _assert_raises_value_error(recognize_hanzi, bad_img, bank=bank)
    _assert_raises_value_error(recognize_hanzi, render_hanzi("休"), bank=bank, op="⿻")
    _assert_raises_value_error(recognize_hanzi, render_hanzi("休"), bank=bank, declared_hanzi="我")
    _assert_raises_value_error(recognize_hanzi, render_hanzi("休"), bank=bank, p_threshold=True)


# ---------------------------------------------------------------------------
# 定点变异自证（--mutate）：破坏槽位方向 / 破坏部件特征 / 绕过门禁必点红
# ---------------------------------------------------------------------------

def _check_mutation_wrong_slot_orientation():
    """M1：强制将左右结构「休」按上下槽位（⿱）切分 ⇒ 必无法正确识别「休」。"""
    bank = HanziRadicalCardBank()
    res = recognize_hanzi(render_hanzi("休", size=64), bank=bank, op="⿱")
    assert res["verdict"] == "ACCEPT" and res["hanzi"] == "休", (
        "M1 槽位方向变异点红：op=⿱ 下输出 verdict=%s hanzi=%r ids=%r"
        % (res["verdict"], res["hanzi"], res["ids"])
    )


def _check_mutation_corrupt_radical_card():
    """M2：破坏独体部件条件卡特征（把「木」「人」「口」「日」原型清零/反号）⇒ 整字判定率必跌破 70%。"""
    bank = HanziRadicalCardBank()
    for r in ("木", "人", "口", "日", "月"):
        bank.slot_cards["⿰"][r] = -bank.slot_cards["⿰"][r]
        bank.slot_cards["⿱"][r] = -bank.slot_cards["⿱"][r]
    ev = evaluate_hanzi_m5(bank)
    assert ev["char_accuracy"] >= 0.70, (
        "M2 破坏部件条件卡点红：char_accuracy=%.4f < 0.70" % ev["char_accuracy"]
    )


def _check_mutation_bool_threshold_gate():
    """M3：真接口面对 bool 阈值必须 fail-loud 抛 ValueError（不得将 True 当 1.0）。"""
    bank = HanziRadicalCardBank()
    recognize_hanzi(render_hanzi("休", size=64), bank=bank, p_threshold=True)


def test_m5_mutation_self_proof_has_teeth():
    for fn in (_check_mutation_wrong_slot_orientation, _check_mutation_corrupt_radical_card):
        try:
            fn()
        except AssertionError:
            pass
        else:
            raise AssertionError("变异未点红：%s" % fn.__name__)
    _assert_raises_value_error(_check_mutation_bool_threshold_gate)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="M5 汉字部件条件组合卡测试与变异自证")
    parser.add_argument("--mutate", action="store_true", help="运行定点变异自证")
    args = parser.parse_args(argv)

    for fn in (
        test_m5_zero_shot_xiu_and_24plus_sample_accuracy,
        test_m5_four_state_honesty_and_import_purity,
        test_m5_nonfinite_bool_and_ids_fail_closed,
        test_m5_mutation_self_proof_has_teeth,
    ):
        fn()
        print("PASS %s" % fn.__name__)

    if args.mutate:
        for name, fn, exc_cls in (
            ("M1_wrong_slot_orientation", _check_mutation_wrong_slot_orientation, AssertionError),
            ("M2_corrupt_radical_cards", _check_mutation_corrupt_radical_card, AssertionError),
            ("M3_bool_threshold_fail_loud", _check_mutation_bool_threshold_gate, ValueError),
        ):
            try:
                fn()
            except exc_cls as exc:
                print("MUTATE-CAUGHT %s: %s" % (name, exc))
            else:
                print("MUTATE-MISSED %s" % name)
                return 1
        print("MUTATION_SELF_PROOF=PASS (3/3 caught)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
