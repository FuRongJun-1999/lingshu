# -*- coding: utf-8 -*-
"""影视层 P7 负记忆迭代闭环守卫 —— wmfilm.iteration/1 + _det.iter_loop

理论依据：
    docs/plans/影视级视听语言_理论架构_v0.1.md
      · §一 命题 6（负记忆驱动收敛）：现象 → 根因 → 新判据/新不适用条件 → 归档 → 下次自动拦住
      · §4.1 三级门（G1 只收 standard/consensus；G2 收 empirical 告警不拦；G3 收 style 只 DEFER）
      · §七 P7（每次迭代必须产出 >=1 条新判据或新不适用条件）
      · §7.2.1 核心不变式（缺料必产出 BLINDSPOT，绝不静默变成 ACCEPT）

运行方式：
    python -X utf8 -m pytest tests/test_film_p7_iter_loop.py -v    # Py3.13 pytest
    python -X utf8 tests/test_film_p7_iter_loop.py                 # Py3.12 纯标准库直跑
    python -X utf8 tests/test_film_p7_iter_loop.py --mutate        # 4 条变异点红自证
"""
from __future__ import annotations

import copy
import os
import re
import subprocess
import sys

try:
    import pytest  # noqa: E402
except ImportError:
    pytest = None

_HERE = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.dirname(_HERE)
_PKG_ROOT = os.path.join(_ROOT, "bridges", "blender")
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from wm_bridge.factory.film import iteration as IT                       # noqa: E402
from wm_bridge.factory.film import oracle as OR                          # noqa: E402
from wm_bridge.factory.film._det import iter_loop as IL                  # noqa: E402
from wm_bridge.factory.film._det import scene as S                       # noqa: E402

_D_BEFORE = "1" * 64
_D_AFTER = "2" * 64


def _assert_raises(exc_cls, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_cls:
        return
    raise AssertionError("期望抛出 %s 但未抛出" % exc_cls.__name__)


def _criterion_c8():
    return {
        "name": "C8",
        "gate": "G1",
        "nature": "consensus",
        "metric": "silhouette_delta_L",
        "op": "ge",
        "threshold": 0.15,
        "source": "架构文档 §4.2 C8：主体轮廓与背景亮度差 >= 0.15",
        "description": "主次体剪影亮度对比度硬门",
    }


def _criterion_c1():
    return {
        "name": "C1",
        "gate": "G1",
        "nature": "consensus",
        "metric": "coverage_actual",
        "op": "between",
        "range": [0.60, 0.75],
        "source": "架构文档 §4.2 C1：主体覆盖率 0.60-0.75",
        "description": "主体覆盖率有效区间",
    }


def _criterion_a2_g2():
    return {
        "name": "A2",
        "gate": "G2",
        "nature": "empirical",
        "metric": "color_temp_est_k",
        "op": "between",
        "range": [3200.0, 5600.0],
        "source": "架构文档 §4.2 A2：主辅色温经验区间",
        "description": "色温舒适经验软门",
    }


def _criterion_c3_g3():
    return {
        "name": "C3",
        "gate": "G3",
        "nature": "style",
        "metric": "coverage_actual",
        "op": "le",
        "threshold": 0.70,
        "source": "架构文档 §4.2 C3：视觉平衡属风格取向（G3）",
    }


def _iter_doc(iter_id="iter_001", criteria=None, negatives=None, **kw):
    doc = {
        "schema": "wmfilm.iteration/1",
        "id": iter_id,
        "phenomenon": {
            "shot_id": "shot_03",
            "frame_range": [48, 96],
            "observed_readings": {"silhouette_delta_L": 0.06},
            "description": "逆光下主体剪影与暗背景混成一团",
        },
        "root_cause": {
            "layer": "shot",
            "component": "light.rim",
            "mechanism": "轮廓光能量不足且背景天光过暗，导致轮廓亮度差低于 0.15",
        },
        "outcomes": {
            "new_criteria": [_criterion_c8()] if criteria is None else criteria,
            "new_negatives": [] if negatives is None else negatives,
        },
        "verification": {
            "before_oracle_ref": {"name": "oracle_before", "digest": _D_BEFORE},
            "after_oracle_ref": {"name": "oracle_after", "digest": _D_AFTER},
            "before_verdict": "REJECT",
            "after_verdict": "ACCEPT",
        },
    }
    doc.update(kw)
    return doc


def _scene_measured(shot_id="shot_03", frame=72, **measured):
    return S.build_scene(truth=[{
        "shot_id": shot_id,
        "frame": frame,
        "measured": measured,
    }])


# ══════════════════════════════════════════════════════ [A] 契约规范化与反例拒收
def test_a1_valid_iteration_normalizes_and_summary():
    raw = _iter_doc(negatives=[{
        "target": "C8",
        "condition": "纯剪影黑场转场帧不要求轮廓反差",
        "reason": "黑场转场本身用于遮黑切换",
        "fallback": "E4",
    }])
    snap = copy.deepcopy(raw)
    norm = IT.normalize_iteration(raw)
    assert raw == snap, "normalize_iteration 不得修改入参"
    assert IT.validate_iteration(raw) == (True, "")
    assert norm["outcomes"]["new_criteria"][0]["family"] == "C"
    assert norm["outcomes"]["new_criteria"][0]["threshold_name"] == "C8_silhouette_delta_L"

    rev = {k: raw[k] for k in reversed(list(raw.keys()))}
    d1 = IT.iteration_digest(raw)
    d2 = IT.iteration_digest(rev)
    assert len(d1) == 64 and d1 == d2

    sm = IT.iteration_summary(raw)
    assert sm["id"] == "iter_001"
    assert sm["layer"] == "shot"
    assert sm["criteria_count"] == 1
    assert sm["negatives_count"] == 1
    assert sm["digest"] == d1


def test_a2_rejects_wrong_schema_and_unknown_keys():
    bad_schema = _iter_doc(schema="wmfilm.iteration/2")
    assert IT.validate_iteration(bad_schema)[0] is False

    cases = []
    c1 = _iter_doc()
    c1["extra_top"] = 1
    cases.append(c1)

    c2 = _iter_doc()
    c2["phenomenon"] = dict(c2["phenomenon"], unknown_ph=1)
    cases.append(c2)

    c3 = _iter_doc()
    c3["root_cause"] = dict(c3["root_cause"], unknown_rc=1)
    cases.append(c3)

    c4 = _iter_doc()
    c4["outcomes"] = dict(c4["outcomes"], unknown_oc=1)
    cases.append(c4)

    c5 = _iter_doc(criteria=[dict(_criterion_c8(), unknown_crit=1)])
    cases.append(c5)

    c6 = _iter_doc()
    c6["verification"] = dict(c6["verification"], unknown_vr=1)
    cases.append(c6)

    c7 = _iter_doc()
    c7["verification"]["before_oracle_ref"] = {
        "name": "o", "digest": _D_BEFORE, "extra": 1
    }
    cases.append(c7)

    for bad in cases:
        ok, why = IT.validate_iteration(bad)
        assert ok is False and why
        _assert_raises(IT.IterationError, IT.normalize_iteration, bad)


def test_a3_rejects_invalid_layer_and_family_name():
    bad_layer = _iter_doc()
    bad_layer["root_cause"] = {"layer": "render_magic", "mechanism": "m"}
    assert IT.validate_iteration(bad_layer)[0] is False

    for bad_name in ("T1", "Z9", "invalid_name", "C"):
        bad_crit = _iter_doc(criteria=[dict(_criterion_c8(), name=bad_name)])
        assert IT.validate_iteration(bad_crit)[0] is False


def test_a4_rejects_bool_nan_inf_and_bad_sha256():
    for bad_val in (True, False, float("nan"), float("inf"), float("-inf")):
        b1 = _iter_doc()
        b1["phenomenon"]["observed_readings"] = {"silhouette_delta_L": bad_val}
        assert IT.validate_iteration(b1)[0] is False

        b2 = _iter_doc(criteria=[dict(_criterion_c8(), threshold=bad_val)])
        assert IT.validate_iteration(b2)[0] is False

    b_fr = _iter_doc()
    b_fr["phenomenon"]["frame_range"] = [True, 10]
    assert IT.validate_iteration(b_fr)[0] is False

    b_fr2 = _iter_doc()
    b_fr2["phenomenon"]["frame_range"] = [50, 10]
    assert IT.validate_iteration(b_fr2)[0] is False

    for bad_hex in ("abc", "a" * 63, "g" * 64, "<placeholder_sha256>"):
        b_ref = _iter_doc()
        b_ref["verification"]["before_oracle_ref"] = {"name": "b", "digest": bad_hex}
        assert IT.validate_iteration(b_ref)[0] is False

    for junk in (None, [], "str", 42, {"schema": "wrong"}):
        ok, why = IT.validate_iteration(junk)
        assert ok is False and isinstance(why, str) and why


# ══════════════════════════════════════════════════════ [B] P7 硬判据与 §4.1 三级门
def test_b1_p7_zero_outcome_must_reject():
    """P7 硬验收判据：new_criteria 与 new_negatives 皆空 ⇒ fail-closed 拒收。"""
    empty_out = _iter_doc(criteria=[], negatives=[])
    ok, why = IT.validate_iteration(empty_out)
    assert ok is False
    assert "P7" in why or "new_criteria" in why
    _assert_raises(IT.IterationError, IT.normalize_iteration, empty_out)


def test_b2_p7_negatives_only_outcome_is_accepted():
    neg_only = _iter_doc(criteria=[], negatives=[{
        "target": "K4",
        "condition": "cut-in 机位不变但景别跨级",
        "reason": "K4 纯角度判据会误杀 cut-in",
        "fallback": "K5",
    }])
    assert IT.validate_iteration(neg_only) == (True, "")


def test_b3_g1_rejects_style_and_empirical():
    """§4.1 三级门：严禁把 style（或 empirical）塞进 G1 硬门拦创作。"""
    style_in_g1 = _iter_doc(criteria=[{
        "name": "C3",
        "gate": "G1",
        "nature": "style",
        "metric": "coverage_actual",
        "op": "le",
        "threshold": 0.70,
        "source": "试图把个人风格塞进 G1 硬门",
    }])
    ok, why = IT.validate_iteration(style_in_g1)
    assert ok is False
    assert "G1" in why and "style" in why
    _assert_raises(IT.IterationError, IT.normalize_iteration, style_in_g1)

    emp_in_g1 = _iter_doc(criteria=[dict(_criterion_a2_g2(), gate="G1")])
    assert IT.validate_iteration(emp_in_g1)[0] is False

    style_in_g2 = _iter_doc(criteria=[dict(_criterion_c3_g3(), gate="G2")])
    assert IT.validate_iteration(style_in_g2)[0] is False

    cons_in_g3 = _iter_doc(criteria=[dict(_criterion_c8(), gate="G3")])
    assert IT.validate_iteration(cons_in_g3)[0] is False


# ══════════════════════════════════════════════════════ [C] 声明式求值与四态语义
def test_c1_g1_scalar_interval_and_nested_metric():
    c8 = _criterion_c8()
    bad_sc = _scene_measured(silhouette_delta_L=0.06)
    good_sc = _scene_measured(silhouette_delta_L=0.22)
    r_bad = IL.evaluate_custom_criterion(c8, bad_sc)
    r_good = IL.evaluate_custom_criterion(c8, good_sc)
    assert r_bad["verdict"] == "REJECT"
    assert r_bad["measured"] == [0.06]
    assert r_good["verdict"] == "ACCEPT"
    assert r_good["measured"] == [0.22]

    c1 = _criterion_c1()
    assert IL.evaluate_custom_criterion(c1, _scene_measured(coverage_actual=0.40))["verdict"] == "REJECT"
    assert IL.evaluate_custom_criterion(c1, _scene_measured(coverage_actual=0.68))["verdict"] == "ACCEPT"

    # outside 算子 + 点路径 histogram.p50
    c_out = {
        "name": "A3", "gate": "G1", "nature": "standard",
        "metric": "histogram.p50", "op": "outside", "range": [0.45, 0.55],
        "source": "避免灰阶均值死锁在 0.50±0.05",
    }
    assert IL.evaluate_custom_criterion(c_out, _scene_measured(histogram={"p50": 0.50}))["verdict"] == "REJECT"
    assert IL.evaluate_custom_criterion(c_out, _scene_measured(histogram={"p50": 0.30}))["verdict"] == "ACCEPT"


def test_c2_g2_soft_gate_warns_without_reject():
    a2 = _criterion_a2_g2()
    warn_chk = IL.evaluate_custom_criterion(a2, _scene_measured(color_temp_est_k=6500.0))
    ok_chk = IL.evaluate_custom_criterion(a2, _scene_measured(color_temp_est_k=4200.0))
    assert warn_chk["verdict"] == "ACCEPT"
    assert IL.is_soft_warning(warn_chk) is True
    assert ok_chk["verdict"] == "ACCEPT"
    assert IL.is_soft_warning(ok_chk) is False


def test_c3_g3_style_always_defers_with_empty_measured():
    c3 = _criterion_c3_g3()
    chk = IL.evaluate_custom_criterion(c3, _scene_measured(coverage_actual=0.95))
    assert chk["verdict"] == "DEFER"
    assert chk["measured"] == []
    assert chk["reason"].strip()


def test_c4_missing_truth_or_metric_always_blindspot_never_accept():
    """§7.2.1 核心不变式：缺实测字段必产 BLINDSPOT，绝不静默变成 ACCEPT。"""
    c8 = _criterion_c8()
    for sc in (
        S.build_scene(),
        _scene_measured(shot_id="other_shot", silhouette_delta_L=0.25) if False else _scene_measured(coverage_actual=0.68),
        _scene_measured(silhouette_delta_L=None),
        _scene_measured(silhouette_delta_L=float("nan")),
    ):
        chk = IL.evaluate_custom_criterion(c8, sc)
        assert chk["verdict"] == "BLINDSPOT", "缺料竟未判 BLINDSPOT：%r" % (chk,)
        assert chk["verdict"] != "ACCEPT"
        assert "silhouette_delta_L" in chk["reason"]

    # 指定 shot_id 但 truth 里只有别的镜头 ⇒ BLINDSPOT
    c8_shot = dict(c8, shot_id="shot_03")
    chk_miss_shot = IL.evaluate_custom_criterion(
        c8_shot, _scene_measured(shot_id="shot_99", silhouette_delta_L=0.30))
    assert chk_miss_shot["verdict"] == "BLINDSPOT"


# ══════════════════════════════════════════════════════ [D] 判别力自证 + 防篡改哈希链 + 闭环重放
def test_d1_discrimination_rejects_useless_criteria():
    # 1. 阈值过松：bad_scene 上竟然 ACCEPT ⇒ 无判别力，拒绝归档
    loose = _iter_doc(criteria=[dict(_criterion_c8(), threshold=0.01)])
    bad_sc = _scene_measured(silhouette_delta_L=0.06)
    fixed_sc = _scene_measured(silhouette_delta_L=0.22)
    _assert_raises(IL.IterLoopError, IL.verify_iteration_discrimination,
                   loose, bad_sc, fixed_sc)
    _assert_raises(IL.IterLoopError, IL.append_iteration_ledger,
                   [], loose, bad_sc, fixed_sc)

    # 2. 阈值过严：fixed_scene 上也 REJECT ⇒ 误拦修复态，拒绝归档
    too_strict = _iter_doc(criteria=[dict(_criterion_c8(), threshold=0.50)])
    _assert_raises(IL.IterLoopError, IL.verify_iteration_discrimination,
                   too_strict, bad_sc, fixed_sc)

    # 3. 缺料 BLINDSPOT ⇒ 拒绝归档
    _assert_raises(IL.IterLoopError, IL.verify_iteration_discrimination,
                   _iter_doc(), S.build_scene(), fixed_sc)

    # 4. G2 软门：bad_scene 上无告警 ⇒ 拒绝归档；fixed_scene 上仍告警 ⇒ 拒绝归档
    g2_doc = _iter_doc(iter_id="iter_g2", criteria=[_criterion_a2_g2()])
    _assert_raises(IL.IterLoopError, IL.verify_iteration_discrimination,
                   g2_doc, _scene_measured(color_temp_est_k=4500.0),
                   _scene_measured(color_temp_est_k=4500.0))
    _assert_raises(IL.IterLoopError, IL.verify_iteration_discrimination,
                   g2_doc, _scene_measured(color_temp_est_k=6500.0),
                   _scene_measured(color_temp_est_k=6500.0))


def test_d2_end_to_end_negative_memory_convergence_loop():
    """端到端负记忆收敛闭环：两轮迭代归档 → 历史缺陷快照自动 REJECT → 修复快照放行。"""
    it1 = _iter_doc(iter_id="iter_001", criteria=[_criterion_c8()])
    bad1 = _scene_measured(silhouette_delta_L=0.06)
    fix1 = _scene_measured(silhouette_delta_L=0.22)

    it2 = _iter_doc(
        iter_id="iter_002",
        criteria=[_criterion_c1(), _criterion_a2_g2()],
    )
    it2["phenomenon"]["observed_readings"] = {"coverage_actual": 0.35, "color_temp_est_k": 6800.0}
    it2["phenomenon"]["description"] = "主体过小且冷暖失控"
    bad2 = _scene_measured(coverage_actual=0.35, color_temp_est_k=6800.0)
    fix2 = _scene_measured(coverage_actual=0.68, color_temp_est_k=4300.0)

    ledger = IL.append_iteration_ledger(None, it1, bad1, fix1)
    snap_l1 = copy.deepcopy(ledger)
    ledger = IL.append_iteration_ledger(ledger, it2, bad2, fix2)
    assert ledger[:1] == snap_l1, "append 不得修改已有台账列表"
    assert len(ledger) == 2
    assert ledger[0]["prev_digest"] == IL.GENESIS_DIGEST
    assert ledger[1]["prev_digest"] == ledger[0]["entry_digest"]
    assert IL.verify_ledger_chain(ledger) == (True, "")

    # 重放历史缺陷场景：同时踩中 C8 与 C1 ⇒ 自动拦住（wmfilm.oracle/1 校验通过）
    hist_bad = _scene_measured(silhouette_delta_L=0.06, coverage_actual=0.35, color_temp_est_k=6800.0)
    or_bad = IL.replay_ledger_against_scene(ledger, hist_bad, oracle_id="or_bad")
    assert OR.validate_oracle(or_bad) == (True, "")
    ok_bad, blocking = OR.gates_passed(or_bad)
    assert ok_bad is False
    assert set(blocking) == {"C8", "C1"}

    # 重放修复后场景：全部转绿放行
    hist_fix = _scene_measured(silhouette_delta_L=0.22, coverage_actual=0.68, color_temp_est_k=4300.0)
    or_fix = IL.replay_ledger_against_scene(ledger, hist_fix, oracle_id="or_fix")
    assert OR.validate_oracle(or_fix) == (True, "")
    assert OR.gates_passed(or_fix) == (True, [])


def test_d3_ledger_hash_chain_tamper_detection():
    it1 = _iter_doc(iter_id="iter_001", criteria=[_criterion_c8()])
    it2 = _iter_doc(iter_id="iter_002", criteria=[_criterion_c1()])
    bad1 = _scene_measured(silhouette_delta_L=0.06)
    fix1 = _scene_measured(silhouette_delta_L=0.22)
    bad2 = _scene_measured(coverage_actual=0.35)
    fix2 = _scene_measured(coverage_actual=0.68)

    ledger = IL.append_iteration_ledger([], it1, bad1, fix1)
    ledger = IL.append_iteration_ledger(ledger, it2, bad2, fix2)

    # 1. 篡改 iteration 内容
    t1 = copy.deepcopy(ledger)
    t1[0]["iteration"]["phenomenon"]["description"] = "篡改后的描述"
    assert IL.verify_ledger_chain(t1)[0] is False
    _assert_raises(IL.IterLoopError, IL.replay_ledger_against_scene, t1, fix2)

    # 2. 篡改 prev_digest（断链，即使连带重算 entry_digest 也必须被 prev_digest 链校验拦住）
    t2 = copy.deepcopy(ledger)
    t2[1]["prev_digest"] = "f" * 64
    t2[1]["entry_digest"] = IL._compute_entry_digest(
        t2[1]["seq"], t2[1]["prev_digest"], t2[1]["iteration_digest"])
    assert IL.verify_ledger_chain(t2)[0] is False

    # 3. 篡改 entry_digest
    t3 = copy.deepcopy(ledger)
    t3[0]["entry_digest"] = "e" * 64
    assert IL.verify_ledger_chain(t3)[0] is False

    # 4. 删掉首条导致 seq / prev_digest 失配
    t4 = copy.deepcopy(ledger[1:])
    assert IL.verify_ledger_chain(t4)[0] is False

    # 5. 重复迭代 id 拒收
    _assert_raises(IL.IterLoopError, IL.append_iteration_ledger, ledger, it1, bad1, fix1)


# ══════════════════════════════════════════════════════ [E] B4 纯标准库阻断自证
def test_e1_b4_pure_stdlib_metapath_blocker():
    code = (
        "import sys\n"
        "BLOCK = {'bpy', 'numpy', 'PIL', 'mathutils'}\n"
        "class _Block:\n"
        "    def find_module(self, name, path=None):\n"
        "        return self.find_spec(name, path)\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in BLOCK:\n"
        "            raise ImportError('B4 违反：引入了 ' + name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, _Block())\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from wm_bridge.factory.film import iteration as IT, oracle as OR\n"
        "from wm_bridge.factory.film._det import iter_loop as IL, scene as S\n"
        "doc = {\n"
        "  'schema': 'wmfilm.iteration/1', 'id': 'iter_b4',\n"
        "  'phenomenon': {'shot_id': 's1', 'frame_range': [0, 24],\n"
        "                 'observed_readings': {'silhouette_delta_L': 0.05},\n"
        "                 'description': 'b4 smoke'},\n"
        "  'root_cause': {'layer': 'shot', 'mechanism': 'low contrast'},\n"
        "  'outcomes': {'new_criteria': [{\n"
        "      'name': 'C8', 'gate': 'G1', 'nature': 'consensus',\n"
        "      'metric': 'silhouette_delta_L', 'op': 'ge', 'threshold': 0.15,\n"
        "      'source': '§4.2 C8'}],\n"
        "      'new_negatives': []},\n"
        "  'verification': {\n"
        "      'before_oracle_ref': {'name': 'b', 'digest': '1'*64},\n"
        "      'after_oracle_ref': {'name': 'a', 'digest': '2'*64}},\n"
        "}\n"
        "bad = S.build_scene(truth=[{'shot_id': 's1', 'frame': 0, 'measured': {'silhouette_delta_L': 0.05}}])\n"
        "fix = S.build_scene(truth=[{'shot_id': 's1', 'frame': 0, 'measured': {'silhouette_delta_L': 0.25}}])\n"
        "led = IL.append_iteration_ledger([], doc, bad, fix)\n"
        "or_doc = IL.replay_ledger_against_scene(led, bad)\n"
        "assert OR.validate_oracle(or_doc) == (True, '')\n"
        "assert OR.gates_passed(or_doc)[0] is False\n"
        "print('B4_P7_OK')\n"
    )
    p = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", code, _PKG_ROOT],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    assert p.returncode == 0, "B4 纯标准库执行失败：\n%s\n%s" % (p.stdout, p.stderr)
    assert "B4_P7_OK" in p.stdout


def test_e2_sources_do_not_import_heavy_modules():
    pat = re.compile(r"^\s*(?:import|from)\s+(bpy|numpy|PIL|mathutils)\b", re.MULTILINE)
    for rel in (
        os.path.join("bridges", "blender", "wm_bridge", "factory", "film", "iteration.py"),
        os.path.join("bridges", "blender", "wm_bridge", "factory", "film", "_det", "iter_loop.py"),
    ):
        with open(os.path.join(_ROOT, rel), encoding="utf-8") as fh:
            src = fh.read()
        assert not pat.search(src), "%s 引入了禁止的重依赖" % rel


# ══════════════════════════════════════════════════════ --mutate 变异点红自证
_MUTATIONS = (
    (
        "M1_drop_p7_zero_outcome_gate",
        os.path.join("bridges", "blender", "wm_bridge", "factory", "film", "iteration.py"),
        "if not raw_criteria and not raw_negatives:",
        "if False:",
    ),
    (
        "M2_allow_style_in_g1",
        os.path.join("bridges", "blender", "wm_bridge", "factory", "film", "iteration.py"),
        '"G1": ("standard", "consensus"),',
        '"G1": ("standard", "consensus", "style"),',
    ),
    (
        "M3_blindspot_silently_accept",
        os.path.join("bridges", "blender", "wm_bridge", "factory", "film", "_det", "iter_loop.py"),
        'if missing_count > 0 or not readings:\n        return {\n            "name": name,\n            "verdict": "BLINDSPOT",',
        'if missing_count > 0 or not readings:\n        return {\n            "name": name,\n            "verdict": "ACCEPT",',
    ),
    (
        "M4_disable_prev_digest_chain_check",
        os.path.join("bridges", "blender", "wm_bridge", "factory", "film", "_det", "iter_loop.py"),
        "if prev_d != expected_prev:",
        "if False:",
    ),
)


def _run_mutations():
    all_ok = True
    for label, rel, target, repl in _MUTATIONS:
        fpath = os.path.join(_ROOT, rel)
        with open(fpath, encoding="utf-8") as fh:
            orig = fh.read()
        cnt = orig.count(target)
        if cnt != 1:
            print("[ANCHOR-MISS] %s: %s 命中 %d 处" % (label, rel, cnt))
            all_ok = False
            continue
        try:
            with open(fpath, "w", encoding="utf-8", newline="") as fh:
                fh.write(orig.replace(target, repl, 1))
            proc = subprocess.run(
                [sys.executable, "-X", "utf8", os.path.abspath(__file__)],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                cwd=_ROOT,
            )
            red = proc.returncode != 0
            out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        finally:
            with open(fpath, "w", encoding="utf-8", newline="") as fh:
                fh.write(orig)
        if red:
            failed_line = [ln.strip() for ln in out.splitlines() if "failed:" in ln]
            print("[MUTATE-OK] %s -> 变红 (%s)" % (label, failed_line[0] if failed_line else "rc!=0"))
        else:
            print("[MUTATE-FAIL] %s -> 变异后仍绿，守卫失明！" % label)
            all_ok = False
    return 0 if all_ok else 1


def _run_all():
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad, ok = [], 0
    for fn in fns:
        try:
            fn()
            ok += 1
        except Exception:
            bad.append(fn.__name__)
            print("FAIL", fn.__name__)
            traceback.print_exc()
    print("REPORT film_p7_iter_loop: %d passed / %d failed" % (ok, len(bad)))
    if bad:
        print("  failed:", ", ".join(bad))
    print("VERDICT=%s" % ("PASS" if not bad else "FAIL"))
    return 1 if bad else 0


if __name__ == "__main__":
    if "--mutate" in sys.argv:
        sys.exit(_run_mutations())
    sys.exit(_run_all())
