# -*- coding: utf-8 -*-
"""影视层契约族守卫 —— wm_bridge.factory.film 七契约（CI 可收集版）

运行方式：
    python -X utf8 -m pytest tests/test_film_contract_guard.py -v
    python -X utf8 tests/test_film_contract_guard.py          # 无 pytest 时直跑

缺陷（本守卫防的是什么）：
    影视层七份契约（§3.2）落在 bridges/blender/wm_bridge/factory/film/，
    但 CI 只跑仓根 tests/，**不收集 factory/tests/** —— 于是七契约的校验器
    可以在本地全绿、在 CI 上无人过问，直到下游实现踩到「契约静默放过非法输入」。
    本文件把七契约的门禁搬进 CI 收集面，并额外钉死 B4（纯标准库）。

修法：
    1. 本文件置于仓根 tests/（CI 的 pytest 收集面），用 _PKG_ROOT 把
       bridges/blender 插进 sys.path 后 import wm_bridge.factory.film.*；
    2. 每个契约都做「正例过 / 反例拒」双向断言 —— 只测正例的守卫等于没测；
    3. B4 用 sys.modules 差分钉死：import 前后不得出现 bpy / numpy / PIL。

判据来源：
    架构文档 docs/plans/影视级视听语言_理论架构_v0.1.md（内部版本 v0.2）
    §3.2 契约表 · §3.4 wmfilm.shot/1 · §3.5 依赖分层 · §4.2 判据族
    §七 P0（七契约）· §八 不适用条件
    安全区口径 = EBU R95（0.80 / 0.90）；CPS 口径 = Netflix Timed Text Style Guide（<= 20）

断言组（抽掉修复 ⇒ 哪组红）：
    [A] 七契约齐备（模块 / schema 标识 / 关键导出）     抽掉模块 ⇒ A 红
    [B] 正例可规范化 + digest 幂等且与键序无关          抽掉规范化 ⇒ B 红
    [C] 反例必须被拒（每契约至少 2 条）                 抽掉任一条校验 ⇒ C 红
    [D] 跨契约一致性（闭集唯一权威 / digest 绑定）      改成分散定义 ⇒ D 红
    [E] B4 纯标准库（不引入 bpy / numpy / PIL）         误加重依赖 ⇒ E 红
    [F] 口径数值（16:9 安全区 / CPS 上限）              改口径 ⇒ F 红
    [G] #825 双门（CJK limits 判据落到内容 / 硬门读     拆判据或重开旁路 ⇒ G 红
        text 正文、cps_of 读数与正文一致）
"""
import os
import sys

_HERE = os.path.abspath(os.path.dirname(__file__))
_PKG_ROOT = os.path.abspath(os.path.join(_HERE, "..", "bridges", "blender"))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

import pytest  # noqa: E402

from wm_bridge.factory import film as F                                   # noqa: E402
from wm_bridge.factory.film import shot as SH                             # noqa: E402
from wm_bridge.factory.film import perform as PF                          # noqa: E402
from wm_bridge.factory.film import score as SC                            # noqa: E402
from wm_bridge.factory.film import bind as BD                             # noqa: E402
from wm_bridge.factory.film import edit as ED                             # noqa: E402
from wm_bridge.factory.film import text as TX                             # noqa: E402
from wm_bridge.factory.film import oracle as OR                           # noqa: E402

_Digest = "a" * 64
_REF = {"name": "ref_01", "digest": _Digest}


def _shot(**kw):
    d = {
        "schema": "wmfilm.shot/1", "id": "shot_01", "spec_ref": dict(_REF),
        "frame_base": {"fps": 24, "frame_start": 0, "frame_end": 47},
        "camera": {"path": [
            {"t": 0.0, "pos": [0.0, 0.0, 1.0], "aim": [0.0, 0.0, 0.0],
             "focal_35eq": 35.0},
            {"t": 1.0, "pos": [1.0, 0.0, 1.0], "aim": [0.0, 0.0, 0.0],
             "focal_35eq": 50.0}],
            "move_kind": "dolly_in"},
        "compose": {"aspect_ratio": 1.778, "subject_anchor": "center",
                    "headroom_ratio": 0.08, "lookroom_ratio": 0.18,
                    "horizon_v": 0.5, "depth_layers_min": 3,
                    "coverage_target": 0.68},
        "light": {"key_fill_ratio": 4.0, "mood": "neutral",
                  "color_temp_k": {"key": 5600, "fill": 6500},
                  "practicals": [],
                  "atmosphere": {"fog_density": 0.0, "volumetric_gain": 0.0}},
        "transition_out": {"kind": "cut", "quantize_beats": 1,
                           "tolerance_frames": 1},
    }
    d.update(kw)
    return d


def _perform(**kw):
    d = {
        "schema": "wmfilm.perform/1", "id": "perf_01", "spec_ref": dict(_REF),
        "frame_base": {"fps": 24, "frame_start": 0, "frame_end": 48},
        "phases": {"anticipation": {"start": 0, "end": 8},
                   "action": {"start": 8, "end": 32},
                   "follow_through": {"start": 32, "end": 48}},
        "overlap": [{"joint": "hand_R", "delay_frames": 3}],
        "joint_overrides": [{"joint": "spine", "kind": "ease", "axis": "x",
                             "keys": [[0.0, 0.0], [0.5, 12.0], [1.0, 0.0]],
                             "unit": "deg", "easing": "linear"}],
        "principle_refs": ["anticipation", "arc"],
    }
    d.update(kw)
    return d


def _score(**kw):
    d = {
        "schema": "wmmv.score/1", "id": "score_01",
        "source": {"path": "a.wav", "digest": "b" * 64, "duration_s": 2.0,
                   "sample_rate": 48000, "channels": 2},
        "tempo": {"bpm": 120.0, "meter": [4, 4], "offset_s": 0.0},
        "beats": [0.0, 0.5, 1.0, 1.5], "downbeats": [0.0, 1.0],
        "sections": [{"kind": "intro", "start_s": 0.0, "end_s": 1.0,
                      "energy": 0.3},
                     {"kind": "chorus", "start_s": 1.0, "end_s": 2.0,
                      "energy": 0.9}],
        "energy": {"hop_s": 0.5, "metric": "rms",
                   "values": [0.1, 0.4, 0.9, 0.7, 0.2]},
        "key": {"tonic": "A", "mode": "minor", "confidence": 0.8},
        "loudness": {"integrated_lufs": -14.0, "true_peak_db": -1.0},
    }
    d.update(kw)
    return d


def _bind(**kw):
    d = {
        "schema": "wmmv.bind/1", "id": "bind_01",
        "score_ref": {"name": "score_01", "digest": "b" * 64},
        "cut_rule": {"on": "downbeat", "quantize_beats": 1,
                     "tolerance_frames": 1},
        "energy_map": {"metric": "rms", "bands": [
            {"lo": 0.0, "hi": 0.4, "asl_frames": 48, "shot_scale": "wide",
             "move_intensity": 0.2},
            {"lo": 0.4, "hi": 1.0, "asl_frames": 12, "shot_scale": "close",
             "move_intensity": 0.9}]},
        "timbre_map": [{"timbre": "bright", "hue_deg": 45.0, "saturation": 0.7,
                        "value": 0.9}],
        "section_map": [{"section": "intro", "shot_scale": "wide",
                         "move_bias": "crane", "mood": "low_key"}],
    }
    d.update(kw)
    return d


def _edit(**kw):
    d = {
        "schema": "wmmv.edit/1", "id": "edit_01",
        "bind_ref": dict(_REF), "score_ref": dict(_REF),
        "frame_base": {"fps": 24, "frame_start": 0, "frame_end": 95},
        "shots": [
            {"id": "s1", "shot_ref": dict(_REF), "start_frame": 0,
             "end_frame": 47, "transition_out": {"kind": "cut"}},
            {"id": "s2", "shot_ref": dict(_REF), "start_frame": 48,
             "end_frame": 95, "transition_in": {"kind": "dissolve"}}],
        "alignment": [{"at_frame": 48, "beat_s": 2.0, "offset_frames": 0}],
    }
    d.update(kw)
    return d


def _text(**kw):
    d = {
        "schema": "wmmv.text/1", "id": "text_01",
        "frame_base": {"fps": 24, "frame_start": 0, "frame_end": 95},
        "cues": [{"kind": "lyric", "start_frame": 0, "end_frame": 47,
                  "text": "Hold the light\nbefore it fades"}],
    }
    d.update(kw)
    return d


def _oracle(**kw):
    d = {
        "schema": "wmfilm.oracle/1", "id": "oracle_01",
        "subject_ref": dict(_REF),
        "thresholds": [{"name": "K1_arc_ratio", "value": 0.85,
                        "source": "架构文档 §4.2 K1"},
                       {"name": "E1_tol", "value": 1,
                        "source": "架构文档 §4.2 E1"}],
        "checks": [{"name": "K1", "verdict": "ACCEPT", "measured": [0.91],
                    "threshold": "K1_arc_ratio"},
                   {"name": "K3", "verdict": "DEFER", "reason": "首批未实现"},
                   {"name": "E1", "verdict": "ACCEPT", "measured": [4.0, 4.0, 0.0],
                    "threshold": "E1_tol"}],
        "generated_by": "wm_bridge.factory.film._det",
    }
    d.update(kw)
    return d


# ══════════════════════════════════════════════════════ [A] 七契约齐备
def test_a1_seven_contract_modules_exist():
    for m in ("shot", "perform", "score", "bind", "edit", "text", "oracle"):
        assert os.path.isfile(os.path.join(os.path.dirname(F.__file__), m + ".py"))


def test_a2_schema_identifiers_match_registry():
    assert F.SCHEMA_PREFIX == "wmfilm."
    assert F.SCHEMA_MV_PREFIX == "wmmv."
    assert SH.SHOT_SCHEMA == "wmfilm.shot/1"
    assert PF.PERFORM_SCHEMA == "wmfilm.perform/1"
    assert SC.SCORE_SCHEMA == "wmmv.score/1"
    assert BD.BIND_SCHEMA == "wmmv.bind/1"
    assert ED.EDIT_SCHEMA == "wmmv.edit/1"
    assert TX.TEXT_SCHEMA == "wmmv.text/1"
    assert OR.ORACLE_SCHEMA == "wmfilm.oracle/1"


def test_a3_two_faces_exist_per_contract():
    for mod, pre in ((SH, "shot"), (PF, "perform"), (SC, "score"), (BD, "bind"),
                     (ED, "edit"), (TX, "text"), (OR, "oracle")):
        assert callable(getattr(mod, "normalize_" + pre))
        assert callable(getattr(mod, "validate_" + pre))
        assert callable(getattr(mod, pre + "_digest"))
        assert callable(getattr(mod, pre + "_summary"))


def test_a4_wrong_schema_is_rejected():
    bad = _shot()
    bad["schema"] = "wmfilm.shot/2"
    assert SH.validate_shot(bad)[0] is False
    bad2 = _oracle()
    bad2["schema"] = "wmmv.oracle/1"
    assert OR.validate_oracle(bad2)[0] is False


# ══════════════════════════════════════════════════════ [B] 正例 + digest
def test_b1_all_contracts_accept_valid_input():
    assert SH.validate_shot(_shot()) == (True, "")
    assert PF.validate_perform(_perform()) == (True, "")
    assert SC.validate_score(_score()) == (True, "")
    assert BD.validate_bind(_bind()) == (True, "")
    assert ED.validate_edit(_edit()) == (True, "")
    assert TX.validate_text(_text()) == (True, "")
    assert OR.validate_oracle(_oracle()) == (True, "")


def test_b2_normalize_does_not_mutate_input():
    import copy
    for fn, obj in ((SH.normalize_shot, _shot()), (PF.normalize_perform, _perform()),
                    (SC.normalize_score, _score()), (BD.normalize_bind, _bind()),
                    (ED.normalize_edit, _edit()), (TX.normalize_text, _text()),
                    (OR.normalize_oracle, _oracle())):
        snap = copy.deepcopy(obj)
        fn(obj)
        assert obj == snap


def test_b3_digest_is_stable_and_key_order_independent():
    a = _shot()
    b = {k: a[k] for k in reversed(list(a.keys()))}
    assert SH.shot_digest(a) == SH.shot_digest(b)
    assert SH.shot_digest(a) == SH.shot_digest(a)
    assert len(SH.shot_digest(a)) == 64


def test_b4_digest_changes_when_content_changes():
    a, b = _shot(), _shot()
    b["camera"] = dict(a["camera"])
    b["camera"]["path"] = [dict(a["camera"]["path"][0]),
                           dict(a["camera"]["path"][1], focal_35eq=85.0)]
    assert SH.shot_digest(a) != SH.shot_digest(b)


def test_b5_defaults_are_filled_not_invented():
    minimal = {"schema": "wmfilm.shot/1", "id": "s",
               "spec_ref": dict(_REF),
               "frame_base": {"fps": 24, "frame_start": 0, "frame_end": 47},
               "camera": {"path": [
                   {"t": 0.0, "pos": [0, 0, 0], "aim": [0, 0, 1], "focal_35eq": 35},
                   {"t": 1.0, "pos": [0, 0, 0], "aim": [0, 0, 1],
                    "focal_35eq": 35}]},
               "compose": {}, "light": {}}
    n = SH.normalize_shot(minimal)
    assert n["camera"]["move_kind"] == "static"
    assert n["camera"]["shutter_angle"] == 180.0
    assert n["compose"]["aspect_ratio"] == F.ASPECT_16_9
    assert n["compose"]["depth_layers_min"] == 3
    assert n["light"]["mood"] == "neutral"
    assert n["transition_out"]["kind"] == "cut"


# ══════════════════════════════════════════════════════ [C] 反例必拒
def test_c1_shot_rejections():
    b = _shot()
    b["camra"] = 1                                     # 未知键
    assert SH.validate_shot(b)[0] is False
    b2 = _shot()
    b2["camera"] = dict(_shot()["camera"])
    b2["camera"]["path"] = [b2["camera"]["path"][0]]   # 单点（非时间函数）
    assert SH.validate_shot(b2)[0] is False
    b3 = _shot()
    b3["camera"] = dict(_shot()["camera"])
    b3["camera"]["move_kind"] = "teleport"             # 闭集外
    assert SH.validate_shot(b3)[0] is False
    b4 = _shot()
    b4["camera"] = dict(_shot()["camera"])
    p = [dict(_shot()["camera"]["path"][0]), dict(_shot()["camera"]["path"][1])]
    p[0]["pos"] = [float("nan"), 0.0, 0.0]             # NaN
    b4["camera"]["path"] = p
    assert SH.validate_shot(b4)[0] is False
    b5 = _shot()
    b5["spec_ref"] = {"name": "x", "digest": "deadbeef"}   # 半截 hex
    assert SH.validate_shot(b5)[0] is False


def test_c2_perform_rejections():
    b = _perform()
    b["phases"] = {"anticipation": {"start": 0, "end": 8},
                   "action": {"start": 9, "end": 32},
                   "follow_through": {"start": 32, "end": 48}}   # 留缝
    assert PF.validate_perform(b)[0] is False
    b2 = _perform()
    b2["phases"] = {"anticipation": {"start": 0, "end": 8},
                    "action": {"start": 8, "end": 32},
                    "follow_through": {"start": 32, "end": 40}}  # 未覆盖到 end
    assert PF.validate_perform(b2)[0] is False
    b3 = _perform()
    b3["overlap"] = [{"joint": "a", "delay_frames": 0}]         # 0 帧非重叠
    assert PF.validate_perform(b3)[0] is False
    b4 = _perform()
    b4["principle_refs"] = ["squash"]                           # 闭集外原理
    assert PF.validate_perform(b4)[0] is False


def test_c3_score_rejections():
    b = _score()
    b["downbeats"] = [0.0, 0.7]                    # 非拍点
    assert SC.validate_score(b)[0] is False
    b2 = _score()
    b2["energy"] = {"hop_s": 0.5, "values": [0.1, 0.2]}     # 长度不自洽
    assert SC.validate_score(b2)[0] is False
    b3 = _score()
    b3["sections"] = [{"kind": "intro", "start_s": 0.0, "end_s": 1.0},
                      {"kind": "chorus", "start_s": 1.2, "end_s": 2.0}]  # 留缝
    assert SC.validate_score(b3)[0] is False
    b4 = _score()
    b4["beats"] = [0.5, 0.5, 1.0]                  # 非严格递增
    assert SC.validate_score(b4)[0] is False
    b5 = _score()
    b5["tempo"] = {"bpm": 120.0, "meter": [4, 5]}  # 闭集外拍号
    assert SC.validate_score(b5)[0] is False


def test_c4_bind_rejections():
    b = _bind()
    b["energy_map"] = {"metric": "rms", "bands": [
        {"lo": 0.0, "hi": 0.6, "asl_frames": 48, "shot_scale": "wide",
         "move_intensity": 0.2},
        {"lo": 0.4, "hi": 1.0, "asl_frames": 12, "shot_scale": "close",
         "move_intensity": 0.9}]}                  # 分档重叠
    assert BD.validate_bind(b)[0] is False
    b2 = _bind()
    b2["timbre_map"] = [{"timbre": "bright", "hue_deg": 400.0,
                         "saturation": 0.7, "value": 0.9}]     # hue 越界
    assert BD.validate_bind(b2)[0] is False
    b3 = _bind()
    b3["section_map"] = [{"section": "intro", "shot_scale": "wide",
                          "move_bias": "teleport", "mood": "low_key"}]
    assert BD.validate_bind(b3)[0] is False       # 运镜闭集来自 shot（跨契约）
    b4 = _bind()
    # 近失值：连字符写法 "pre-chorus" 不在 SECTION_KINDS（闭集内是 "pre_chorus"）。
    # 用近失值而非乱码，是因为闭集要防的正是这种「看起来对」的写法分裂。
    b4["section_map"] = [{"section": "pre-chorus", "shot_scale": "wide",
                          "move_bias": "crane", "mood": "low_key"}]
    assert BD.validate_bind(b4)[0] is False       # 段落闭集（近失值）


def test_c5_edit_rejections():
    b = _edit()
    b["shots"] = [dict(_edit()["shots"][0]),
                  dict(_edit()["shots"][1], start_frame=50)]   # 留缝
    assert ED.validate_edit(b)[0] is False
    b2 = _edit()
    b2["shots"] = [dict(_edit()["shots"][0]),
                   dict(_edit()["shots"][1], start_frame=47)]  # 重叠
    assert ED.validate_edit(b2)[0] is False
    b3 = _edit()
    b3["shots"] = [dict(_edit()["shots"][0]), dict(_edit()["shots"][0])]  # 重 id
    assert ED.validate_edit(b3)[0] is False
    b4 = _edit()
    b4["alignment"] = [{"at_frame": 10, "beat_s": 1.0}]   # 非切点
    assert ED.validate_edit(b4)[0] is False


def test_c6_text_rejections():
    b = _text()
    b["cues"] = [dict(_text()["cues"][0], end_frame=200)]      # 越界
    assert TX.validate_text(b)[0] is False
    b2 = _text()
    b2["cues"] = [{"kind": "lyric", "start_frame": 0, "end_frame": 2,
                   "text": "This line is far too long to read in two frames"}]
    assert TX.validate_text(b2)[0] is False                    # CPS 超限
    b3 = _text()
    b3["cues"] = [{"kind": "lyric", "start_frame": 0, "end_frame": 47,
                   "text": "一二三四五六七八九十", "script": "cjk"}]
    assert TX.validate_text(b3)[0] is False                    # CJK 未声明 limits
    b4 = _text()
    b4["cues"] = [dict(_text()["cues"][0]),
                  dict(_text()["cues"][0], start_frame=10, end_frame=60)]
    assert TX.validate_text(b4)[0] is False                    # 条目重叠


def test_c7_oracle_rejections():
    b = _oracle()
    b["checks"] = [{"name": "E1", "verdict": "DEFER", "reason": "x",
                    "measured": [1.0]}]                        # DEFER 带实测值
    assert OR.validate_oracle(b)[0] is False
    b2 = _oracle()
    b2["checks"] = [{"name": "K1", "verdict": "ACCEPT",
                     "threshold": "K1_arc_ratio"}]             # ACCEPT 无实测值
    assert OR.validate_oracle(b2)[0] is False
    b3 = _oracle()
    b3["checks"] = [{"name": "K1", "verdict": "ACCEPT", "measured": [0.9],
                     "threshold": "no_such"}]                  # 未登记阈值
    assert OR.validate_oracle(b3)[0] is False
    b4 = _oracle()
    b4["thresholds"] = [{"name": "x", "value": 1.0, "source": "  "}]  # 无来源
    assert OR.validate_oracle(b4)[0] is False
    b5 = _oracle()
    b5["checks"] = [{"name": "Z9", "verdict": "ACCEPT", "measured": [1.0]}]
    assert OR.validate_oracle(b5)[0] is False                  # 判据名不合规
    b6 = _oracle()
    b6["checks"] = [{"name": "K2", "verdict": "DEFER",
                     "measured": [0.0], "reason": "x"}]
    assert OR.validate_oracle(b6)[0] is False                  # DEFER 不得带读数


def test_c8_validate_never_raises_on_garbage():
    """查询面必须 fail-closed 成 (False, reason)，**不得抛异常**。"""
    for fn in (SH.validate_shot, PF.validate_perform, SC.validate_score,
               BD.validate_bind, ED.validate_edit, TX.validate_text,
               OR.validate_oracle):
        for junk in (None, [], "x", 0, {"schema": "wrong"}):
            ok, why = fn(junk)
            assert ok is False
            assert isinstance(why, str) and why


# ══════════════════════════════════════════════════════ [D] 跨契约一致性
def test_d1_closed_sets_have_single_authority():
    """运镜 / 转场 / 段落 / 能量度量 的闭集只有一个定义处。"""
    assert BD.CUT_ONS and SH.MOVE_KINDS and SC.SECTION_KINDS
    assert BD.scale_rank("close") > BD.scale_rank("wide")
    assert ED.transition_kinds() == SH.TRANSITION_KINDS
    assert set(BD._consts()[0]) == set(SC.ENERGY_METRICS)
    assert set(BD._consts()[1]) == set(SC.SECTION_KINDS)
    assert set(BD._consts()[2]) == set(SH.MOVE_KINDS)
    assert set(BD._consts()[3]) == set(SH.MOODS)


def test_d2_cross_contract_digest_binding_is_rejected_when_short():
    for maker, key in ((_bind, "score_ref"), (_edit, "bind_ref")):
        d = maker()
        d[key] = {"name": "x", "digest": "abc"}
        mod = BD if key == "score_ref" else ED
        fn = BD.validate_bind if key == "score_ref" else ED.validate_edit
        assert fn(d)[0] is False


def test_d3_shot_camera_is_a_time_function():
    s = _shot()
    n = SH.normalize_shot(s)
    assert len(n["camera"]["path"]) >= 2
    mid = SH.camera_at(s, 0.5)
    assert mid["pos"] == pytest.approx((0.5, 0.0, 1.0))
    assert mid["focal_35eq"] == pytest.approx(42.5)
    with pytest.raises(Exception):
        SH.camera_at(s, 1.5)                       # 越界不夹取


def test_d4_oracle_gates_block_reject_and_blindspot():
    assert OR.gates_passed(_oracle()) == (True, [])
    b = _oracle()
    b["checks"] = [{"name": "K1", "verdict": "REJECT", "measured": [0.5],
                    "threshold": "K1_arc_ratio"}]
    assert OR.gates_passed(b)[0] is False
    b2 = _oracle()
    b2["checks"] = [{"name": "K2", "verdict": "BLINDSPOT", "measured": [0.0],
                     "reason": "契约缺字段"}]
    assert OR.gates_passed(b2) == (False, ["K2"])


def test_d5_score_digest_round_trips_into_bind():
    sc = _score()
    dg = SC.score_digest(sc)
    assert len(dg) == 64
    b = _bind()
    b["score_ref"] = {"name": "score_01", "digest": dg}
    assert BD.validate_bind(b) == (True, "")


# ══════════════════════════════════════════════════════ [E] B4 纯标准库
def test_e1_no_heavy_deps_pulled_in_by_film_package():
    """B4：本包不得把 bpy / numpy / PIL 带进来（world/ 与轻核共用此保证）。

    ★ 为什么用**子进程 + import 阻断器**而不是查 sys.modules：
      本文件与仓内其它测试同进程收集，若别的测试先 import 了 bpy，
      sys.modules 里就会留下它 —— 断言会变成「看收集顺序」的伪测试。
      子进程里装一个 meta_path finder，重依赖一旦被 import 立即抛错，
      于是这条断言只与 film 包自身有关，与顺序无关。
    """
    import subprocess
    code = (
        "import sys\n"
        "BLOCK = {'bpy', 'numpy', 'PIL', 'mathutils'}\n"
        "class _Block:\n"
        "    def find_module(self, name, path=None):\n"
        "        return self.find_spec(name, path)\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in BLOCK:\n"
        "            raise ImportError('B4 违反：film 包引入了 ' + name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, _Block())\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from wm_bridge.factory.film import shot, perform, score, bind, edit,"
        " text, oracle\n"
        "print('B4_OK')\n")
    p = subprocess.run([sys.executable, "-X", "utf8", "-c", code, _PKG_ROOT],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    assert p.returncode == 0, (
        "film 包在纯标准库环境下 import 失败（B4 违反）：\n%s\n%s"
        % (p.stdout, p.stderr))
    assert "B4_OK" in p.stdout


def test_e2_film_sources_do_not_import_heavy_modules():
    import re
    d = os.path.dirname(F.__file__)
    pat = re.compile(r"^\s*(?:import|from)\s+(bpy|numpy|PIL|mathutils)\b",
                     re.MULTILINE)
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".py"):
            continue
        with open(os.path.join(d, fn), encoding="utf-8") as fh:
            src = fh.read()
        assert not pat.search(src), "%s 顶层 import 了重依赖" % fn


# ══════════════════════════════════════════════════════ [F] 口径数值
def test_f1_safe_area_matches_ebu_r95():
    assert F.TITLE_SAFE == 0.80
    assert F.ACTION_SAFE == 0.90
    px = TX.safe_area_px(_text(), 1920, 1080)
    assert px["title_safe"] == (1536, 864)
    assert px["action_safe"] == (1728, 972)


def test_f2_aspect_is_16_9_and_vertical_out_of_scope():
    assert F.ASPECT_16_9 == pytest.approx(1.778, abs=1e-3)
    assert SH.aspect_is_16_9(_shot()) is True
    b = _shot()
    b["compose"] = dict(_shot()["compose"])
    b["compose"]["aspect_ratio"] = 0.5625          # 9:16
    assert SH.aspect_is_16_9(b) is False


def test_f3_cps_ceiling_is_twenty():
    assert TX.LATIN_LIMITS["max_cps"] == 20.0
    assert TX.LATIN_LIMITS["max_chars_per_line"] == 42
    assert TX.LATIN_LIMITS["max_lines"] == 2
    n = TX.normalize_text(_text())
    assert TX.cps_of(n["cues"][0], 24.0) <= 20.0


def test_f4_oracle_verdict_and_family_closed_sets():
    assert OR.VERDICTS == ("ACCEPT", "REJECT", "DEFER", "BLINDSPOT")
    assert OR.family_of("K1") == "K"
    assert OR.family_of("E1") == "E"
    assert OR.family_of("P2a") == "P"
    with pytest.raises(Exception):
        OR.family_of("X1")


# ══════════════════════════════════════════════════════ [G] #825 双门
_CJK_CUE = {"kind": "lyric", "start_frame": 0, "end_frame": 95,
            "text": "一二三四五六七八九十", "script": "cjk"}
_CJK_LIMITS = {"max_cps": 8.0, "max_chars_per_line": 16, "max_lines": 2}


def _cjk_text(**kw):
    d = _text()
    d["cues"] = [dict(_CJK_CUE)]
    d.update(kw)
    return d


def test_g1_cjk_limits_declaration_must_be_content_complete():
    """#825 ①：「显式声明」落在内容上 —— 空对象 / null / 部分给出都拒。

    模块头宣称 CJK「不给默认值、必须显式声明」—— 若判据只认键存在，
    limits={} 即可让 LATIN_LIMITS 冒名顶替使用者的 CJK 声明。
    """
    for bad in ({}, None, {"max_cps": 8.0},
                {"max_cps": 8.0, "max_chars_per_line": 16},
                {"max_cps": 8.0, "max_chars_per_line": None, "max_lines": 2}):
        d = _cjk_text(limits=bad)
        assert TX.validate_text(d)[0] is False, "limits=%r 须拒" % (bad,)


def test_g2_cjk_full_limits_accepted_and_user_values_applied():
    """三项全给才放行，且生效的是使用者声明的值（非拉丁默认）。"""
    d = _cjk_text(limits=dict(_CJK_LIMITS))
    assert TX.validate_text(d) == (True, "")
    n = TX.normalize_text(d)
    assert n["limits"] == {"max_cps": 8.0, "max_chars_per_line": 16,
                           "max_lines": 2}
    assert n["limits"] != dict(TX.LATIN_LIMITS)      # 不得冒名顶替
    lat = _text()
    lat["limits"] = {}                               # 拉丁侧默认是设计内行为
    assert TX.validate_text(lat) == (True, "")


def test_g3_lines_cannot_bypass_line_width_gate():
    """#825 ②：47 字符 text 配短 lines —— 旁路封死（两形态都拒）。"""
    long_text = "This line is far too long to read in two frames"   # 47 > 42
    a = _text()
    a["cues"] = [{"kind": "lyric", "start_frame": 0, "end_frame": 2,
                  "text": long_text}]
    b = _text()
    b["cues"] = [{"kind": "lyric", "start_frame": 0, "end_frame": 2,
                  "text": long_text, "lines": ["ok"]}]
    assert TX.validate_text(a)[0] is False           # 只给 text：行宽门
    ok, why = TX.validate_text(b)
    assert ok is False and "不一致" in why           # 加 lines 不再翻盘


def test_g4_lines_cannot_bypass_line_count_gate():
    a = _text()
    a["cues"] = [{"kind": "lyric", "start_frame": 0, "end_frame": 95,
                  "text": "aa\nbb\ncc"}]
    b = _text()
    b["cues"] = [{"kind": "lyric", "start_frame": 0, "end_frame": 95,
                  "text": "aa\nbb\ncc", "lines": ["aa", "bb"]}]
    assert TX.validate_text(a)[0] is False           # 3 行 > max_lines=2
    assert TX.validate_text(b)[0] is False           # 旁路封死


def test_g5_consistent_explicit_lines_still_accepted():
    """对照：与 text 逐行一致的 lines（含每行带尾随空格）不误杀。"""
    d = _text()
    d["cues"] = [dict(_text()["cues"][0],
                      lines=["Hold the light ", "before it fades"])]
    assert TX.validate_text(d) == (True, "")
    n = TX.normalize_text(d)
    assert n["cues"][0]["lines"] == ["Hold the light", "before it fades"]


def test_g6_cps_of_and_summary_read_the_text_body():
    """#825 ②读数一致：cps_of/text_summary 钉在 text 正文上，不读 lines。"""
    n = TX.normalize_text(_text())
    c = n["cues"][0]
    expect = sum(len(x) for x in c["text"].split("\n")) / (48 / 24.0)
    assert TX.cps_of(c, 24.0) == pytest.approx(expect)
    assert TX.text_summary(_text())["max_cps"] == pytest.approx(expect)
    hand = {"start_frame": 0, "end_frame": 2, "text": "x" * 47, "lines": ["ok"]}
    assert TX.cps_of(hand, 24.0) == pytest.approx(376.0)   # 47 / 0.125s


# ══════════════════════════════════════════════════════ REPORT
if __name__ == "__main__":
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
    print("REPORT film_contract_guard: %d passed / %d failed" % (ok, len(bad)))
    if bad:
        print("  failed:", ", ".join(bad))
    print("VERDICT=%s" % ("PASS" if not bad else "FAIL"))
    sys.exit(1 if bad else 0)
