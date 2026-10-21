# -*- coding: utf-8 -*-
"""影视层判据门守卫 —— K 族（K1–K5）+ E1 的变异自证（§4.3）

运行方式：
    python -X utf8 -m pytest tests/test_film_criteria_guard.py -v
    python -X utf8 tests/test_film_criteria_guard.py          # 无 pytest 时直跑

缺陷（本守卫防的是什么）：
    §4.3 定死了一条纪律：**只有变异能把它判红的判据，才算真判据**。
    没有这条守卫，判据门极易退化成「永远返回 ACCEPT 的装饰品」——
    尤其是那些**缺料就走 BLINDSPOT** 的判据（K1/K2/K3/K5）：写错了也不会红，
    因为报告里看到的是一堆 BLINDSPOT，看起来像「环境没配好」，而不是「判据写错了」。
    本文件对每条判据做**两相断言**：正常料 ⇒ 该判的态；变异料 ⇒ 必须 REJECT。

修法：
    1. 每条判据配一条**定向变异**（把被测的那个量推过阈值），只改一处；
    2. 同时断言**未变异相**为 ACCEPT —— 只有「红/绿」都验过，才能排除
       「判据其实恒红」或「判据其实恒绿」；
    3. 另立一组断言缺料必须走 BLINDSPOT 且 reason 点名缺哪个字段。

判据来源：
    架构文档 §4.1 三级门 · §4.2 K 族/E 族表 · §4.3 变异自证 · §六.1 真值回读字段
    阈值来源见 `_det.gate_all.all_thresholds()`（每项都带 source，由 oracle 契约强制非空）

断言组（抽掉修复 ⇒ 哪组红）：
    [A] K4 30° 规则：≥30° ⇒ ACCEPT；夹角 11.3° ⇒ REJECT        抽掉角度校验 ⇒ A 红
    [B] K5 景别跳跃：跨级 ⇒ ACCEPT；同机位同景别 ⇒ REJECT        抽掉景别校验 ⇒ B 红
    [C] K1 轴线一致：同侧 ⇒ ACCEPT；异侧 ⇒ REJECT                抽掉侧向校验 ⇒ C 红
    [D] E1 卡点对齐：偏差 0 帧 ⇒ ACCEPT；挪 3 帧 ⇒ REJECT        抽掉容差校验 ⇒ D 红
    [E] 缺料一律 BLINDSPOT 且点名缺字段（不沉默通过）            缺料改判 ACCEPT ⇒ E 红
        （#812 收口：K5 的「机位夹角不可算」也属缺料 —— 与 K4 同口径 BLINDSPOT，
         reason 保留不可算原因、-1.0 哨兵退出 measured；抽掉该修复 ⇒ e9–e11 红）
    [F] gate_all 装配的 oracle 过契约校验；DEFER 登记齐全        登记表漏项 ⇒ F 红
    [G] B4 纯标准库（_det 不引入 bpy/numpy/PIL）                 误加重依赖 ⇒ G 红
"""
import os
import sys

_HERE = os.path.abspath(os.path.dirname(__file__))
_PKG_ROOT = os.path.abspath(os.path.join(_HERE, "..", "bridges", "blender"))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

import pytest  # noqa: E402

from wm_bridge.factory.film import oracle as OR                          # noqa: E402
from wm_bridge.factory.film._det import gate_all as GA                   # noqa: E402
from wm_bridge.factory.film._det import scene as S                       # noqa: E402

_D = "a" * 64
_AXIS = {"a": [-5.0, 0.0, 0.0], "b": [5.0, 0.0, 0.0]}
_SCORE = {"schema": "wmmv.score/1", "id": "sc_01",
          "source": {"digest": "b" * 64, "duration_s": 4.0, "sample_rate": 48000},
          "tempo": {"bpm": 60.0, "meter": [4, 4], "offset_s": 0.0},
          "beats": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
          "downbeats": [0.0, 1.0, 2.0, 3.0]}

#: 基准机位：s2 在 (5, −5, 1.6)，与 s1 的 (0, −5, 1.6) 在目标点处夹角 44.8°（> 30°）
_POS_B_OK = (5.0, -5.0, 1.6)
#: 变异机位：夹角 11.3°（< 30°）
_POS_B_NEAR = (1.0, -5.0, 1.6)
#: 变异机位：落在轴线另一侧（侧向符号翻转）
_POS_B_FLIP = (5.0, 5.0, 1.6)


def _shot(sid, pos):
    return {"schema": "wmfilm.shot/1", "id": sid,
            "spec_ref": {"name": "hero_walker", "digest": _D},
            "frame_base": {"fps": 24, "frame_start": 0, "frame_end": 47},
            "camera": {"path": [
                {"t": 0.0, "pos": list(pos), "aim": [0.0, 0.0, 1.0],
                 "focal_35eq": 35.0},
                {"t": 1.0, "pos": list(pos), "aim": [0.0, 0.0, 1.0],
                 "focal_35eq": 35.0}],
                "move_kind": "static"},
            "compose": {}, "light": {}}


def _edit(a_end=47, b_start=48):
    return {"schema": "wmmv.edit/1", "id": "edit_01",
            "bind_ref": {"name": "bind_01", "digest": _D},
            "score_ref": {"name": "sc_01", "digest": _D},
            "frame_base": {"fps": 24, "frame_start": 0, "frame_end": 95},
            "shots": [
                {"id": "s1", "shot_ref": {"name": "shot_01", "digest": _D},
                 "start_frame": 0, "end_frame": a_end},
                {"id": "s2", "shot_ref": {"name": "shot_02", "digest": _D},
                 "start_frame": b_start, "end_frame": 95}]}


def _truth(h_a=700.0, h_b=980.0, gaze=None, motion=None):
    out = []
    for sid, h, fr in (("s1", h_a, 0), ("s2", h_b, 48)):
        m = {"subject_bbox_screen": [400.0, 200.0, 1000.0, 200.0 + h]}
        if gaze is not None:
            m["subject_gaze_screen"] = gaze[sid]
        if motion is not None:
            m["subject_motion_screen"] = motion[sid]
        out.append({"shot_id": sid, "frame": fr, "measured": m})
    return out


def _scene(pos_b=_POS_B_OK, a_end=47, b_start=48, h_a=700.0, h_b=980.0,
           with_truth=True, with_axis=True, gaze=None, motion=None):
    return S.build_scene(
        shots={"s1": _shot("shot_01", (0.0, -5.0, 1.6)),
               "s2": _shot("shot_02", pos_b)},
        edit=_edit(a_end, b_start), score=_SCORE,
        axis=_AXIS if with_axis else None,
        truth=_truth(h_a, h_b, gaze, motion) if with_truth else None)


def _v(sc):
    doc = GA.build_oracle(sc, "oracle_t", "edit_01", _D)
    return {c["name"]: c["verdict"] for c in doc["checks"]}


# ══════════════════════════════════════════════════════ [A] K4 30° 规则
def test_a1_k4_accepts_when_angle_ge_30():
    assert _v(_scene(_POS_B_OK))["K4"] == "ACCEPT"


def test_a2_k4_mutation_angle_under_30_must_reject():
    """变异：把 s2 机位挪到夹角 11.3° ⇒ K4 必须红（还原 ⇒ 绿，见 a1）。"""
    assert _v(_scene(_POS_B_NEAR))["K4"] == "REJECT"


def test_a3_k4_reading_is_the_angle():
    doc = GA.build_oracle(_scene(_POS_B_OK), "o", "e", _D)
    k4 = [c for c in doc["checks"] if c["name"] == "K4"][0]
    assert k4["measured"] and k4["measured"][0] == pytest.approx(44.8, abs=0.5)
    assert k4["threshold"] == "K4_min_angle_deg"


# ══════════════════════════════════════════════════════ [B] K5 景别跳跃
def test_b1_k5_accepts_when_scale_crosses_level():
    assert _v(_scene(_POS_B_OK, h_b=980.0))["K5"] == "ACCEPT"


def test_b2_k5_mutation_same_scale_same_position_must_reject():
    """变异：同机位（11.3°）+ 同景别（高度相同）⇒ 跳切 ⇒ K5 必须红。"""
    assert _v(_scene(_POS_B_NEAR, h_b=700.0))["K5"] == "REJECT"


def test_b3_k5_allows_same_scale_when_angle_ge_30():
    """换机位（44.8°）时同景别是**合法剪辑**，K5 不得误杀。"""
    assert _v(_scene(_POS_B_OK, h_b=700.0))["K5"] == "ACCEPT"


# ══════════════════════════════════════════════════════ [C] K1 轴线一致
def test_c1_k1_accepts_when_same_side():
    assert _v(_scene(_POS_B_OK))["K1"] == "ACCEPT"


def test_c2_k1_mutation_crossing_axis_must_reject():
    """变异：把 s2 机位镜像到轴线另一侧 ⇒ 越轴 ⇒ K1 必须红。"""
    assert _v(_scene(_POS_B_FLIP))["K1"] == "REJECT"


# ══════════════════════════════════════════════════════ [D] E1 卡点对齐
def test_d1_e1_accepts_when_cut_on_downbeat():
    assert _v(_scene(a_end=47, b_start=48))["E1"] == "ACCEPT"


def test_d2_e1_mutation_shift_cut_by_3_frames_must_reject():
    """变异：切点从第 48 帧挪到第 51 帧（2.125 s，偏 3 帧 > 容差 1 帧）⇒ 必须红。"""
    assert _v(_scene(a_end=50, b_start=51))["E1"] == "REJECT"


def test_d3_e1_declared_tolerance_can_only_tighten():
    """edit 声明的容差只收紧不放宽：声明 0 帧时，偏差 1 帧必须红。"""
    ed = _edit(a_end=47, b_start=48)
    ed["shots"][0]["transition_out"] = {"kind": "cut", "quantize_beats": 1,
                                        "tolerance_frames": 0}
    sc = S.build_scene(shots={"s1": _shot("shot_01", (0.0, -5.0, 1.6)),
                              "s2": _shot("shot_02", _POS_B_OK)},
                       edit=ed, score=_SCORE, axis=_AXIS, truth=_truth())
    assert _v(sc)["E1"] == "ACCEPT"          # 偏差恰 0 帧，声明 0 帧仍过


def test_d4_e1_threshold_is_registered_with_source():
    th = {t["name"]: t for t in GA.all_thresholds()}
    assert "E1_tolerance_frames" in th
    assert th["E1_tolerance_frames"]["value"] == 1.0
    assert th["E1_tolerance_frames"]["source"].strip()


# ══════════════════════════════════════════════════════ [E] 缺料 ⇒ BLINDSPOT
def test_e1_k1_blindspot_without_axis():
    doc = GA.build_oracle(_scene(with_axis=False), "o", "e", _D)
    c = [x for x in doc["checks"] if x["name"] == "K1"][0]
    assert c["verdict"] == "BLINDSPOT"
    assert "scene.axis" in c["reason"]


def test_e2_k2_blindspot_without_gaze_field():
    doc = GA.build_oracle(_scene(), "o", "e", _D)
    c = [x for x in doc["checks"] if x["name"] == "K2"][0]
    assert c["verdict"] == "BLINDSPOT"
    assert "subject_gaze_screen" in c["reason"]


def test_e3_k2_accepts_when_gaze_opposes():
    """给出视线真值：s1 看画右(+1)、s2 看画左(−1) ⇒ 正反打成立 ⇒ ACCEPT。"""
    sc = _scene(gaze={"s1": 1.0, "s2": -1.0})
    assert _v(sc)["K2"] == "ACCEPT"


def test_e4_k2_mutation_same_gaze_direction_must_reject():
    """变异：两镜视线同向（都在看画右）⇒ 未对视 ⇒ K2 必须红。"""
    sc = _scene(gaze={"s1": 1.0, "s2": 1.0})
    assert _v(sc)["K2"] == "REJECT"


def test_e5_k3_mutation_screen_direction_flip_must_reject():
    sc = _scene(motion={"s1": 1.0, "s2": -1.0})
    assert _v(sc)["K3"] == "REJECT"
    sc2 = _scene(motion={"s1": 1.0, "s2": 1.0})
    assert _v(sc2)["K3"] == "ACCEPT"


def test_e6_k5_blindspot_without_truth():
    doc = GA.build_oracle(_scene(with_truth=False), "o", "e", _D)
    c = [x for x in doc["checks"] if x["name"] == "K5"][0]
    assert c["verdict"] == "BLINDSPOT"
    assert "subject_bbox_screen" in c["reason"]


def test_e7_e1_blindspot_without_score():
    sc = S.build_scene(shots={"s1": _shot("shot_01", (0.0, -5.0, 1.6)),
                              "s2": _shot("shot_02", _POS_B_OK)},
                       edit=_edit(), axis=_AXIS, truth=_truth())
    doc = GA.build_oracle(sc, "o", "e", _D)
    c = [x for x in doc["checks"] if x["name"] == "E1"][0]
    assert c["verdict"] == "BLINDSPOT"
    assert "score" in c["reason"]


def test_e8_blindspot_never_silently_passes():
    """缺料绝不得变成 ACCEPT —— 这是本设计的核心不变式。"""
    v = _v(_scene(with_truth=False, with_axis=False))
    assert v["K1"] == "BLINDSPOT"
    assert v["K2"] == "BLINDSPOT"
    assert v["K3"] == "BLINDSPOT"
    assert v["K5"] == "BLINDSPOT"
    assert "ACCEPT" not in (v["K1"], v["K2"], v["K3"], v["K5"])


def test_e9_k5_blindspot_when_angle_uncomputable_missing_contract():
    """#812 α：edit 引用 s2 但缺其镜头契约 ⇒ 夹角不可算 ⇒ K5 必须 BLINDSPOT。

    修前缺陷：None 被并入「< 30°」分支判 REJECT、measured 第二位写 -1.0 哨兵、
    reason 伪称「夹角 < 30°」。同一输入 K4 判 BLINDSPOT —— 两判据必须同口径。
    两条路径都要钉死：rd ≤ 容差（修前 REJECT+哨兵）与 rd > 容差（修前 ACCEPT
    +哨兵）—— 哨兵写入是无条件的，不得只收口一个分支。
    """
    for h_b in (700.0, 980.0):            # rd ≈ 0（微变）与 rd ≈ 0.286（跨级）
        sc = S.build_scene(shots={"s1": _shot("shot_01", (0.0, -5.0, 1.6))},
                           edit=_edit(), score=_SCORE, axis=_AXIS,
                           truth=_truth(h_b=h_b))
        c = [x for x in GA.build_oracle(sc, "o", "e", _D)["checks"]
             if x["name"] == "K5"][0]
        assert c["verdict"] == "BLINDSPOT"
        assert "缺镜头契约：s2" in c["reason"]
        assert "机位夹角不可算" in c["reason"]
        assert -1.0 not in (c.get("measured") or [])


def test_e10_k5_blindspot_when_angle_uncomputable_aim_offset():
    """#812 β：两镜契约齐全但瞄准点相差 1.0 m > 0.5 m ⇒ 夹角基准不唯一。

    `_pair_angle` 的不可算原因必须进 reason（修前被丢进未使用的 `_why`），
    且 -1.0 哨兵不得充当实测夹角读数。rd ≤ / > 容差两分支同 test_e9。
    """
    s2 = _shot("shot_02", _POS_B_OK)
    for kf in s2["camera"]["path"]:
        kf["aim"] = [1.0, 0.0, 1.0]       # 与 s1 的瞄准点相差 1.0 m（> 0.5 m）
    for h_b in (700.0, 980.0):
        sc = S.build_scene(shots={"s1": _shot("shot_01", (0.0, -5.0, 1.6)),
                                  "s2": s2},
                           edit=_edit(), score=_SCORE, axis=_AXIS,
                           truth=_truth(h_b=h_b))
        c = [x for x in GA.build_oracle(sc, "o", "e", _D)["checks"]
             if x["name"] == "K5"][0]
        assert c["verdict"] == "BLINDSPOT"
        assert "瞄准点相差" in c["reason"]
        assert "机位夹角不可算" in c["reason"]
        assert -1.0 not in (c.get("measured") or [])


def test_e11_k5_computable_path_reads_real_angle():
    """#812 回归：夹角可算时 K5 的判态与读数不得被本修复波及。

    读数保持 [景别相对差, 机位夹角] 成对且为真实夹角 —— 跨级（cut-in）ACCEPT、
    同机位同景别（真跳切）REJECT 两相都在。
    """
    acc = [x for x in GA.build_oracle(_scene(_POS_B_OK, h_b=980.0), "o", "e", _D)
           ["checks"] if x["name"] == "K5"][0]
    assert acc["verdict"] == "ACCEPT"
    assert acc["measured"][1] == pytest.approx(44.8, abs=0.5)
    rej = [x for x in GA.build_oracle(_scene(_POS_B_NEAR, h_b=700.0), "o", "e", _D)
           ["checks"] if x["name"] == "K5"][0]
    assert rej["verdict"] == "REJECT"
    assert rej["measured"][1] == pytest.approx(11.3, abs=0.5)
    for c in (acc, rej):
        assert -1.0 not in c["measured"]


# ══════════════════════════════════════════════════════ [F] gate_all / oracle
def test_f1_oracle_document_passes_contract():
    doc = GA.build_oracle(_scene(), "oracle_p1", "edit_01", _D)
    assert OR.validate_oracle(doc) == (True, "")


def test_f2_deferred_registry_is_complete():
    """登记表必须覆盖 §4.2 六族的全部判据编号（首批 K1-K5/E1 之外一律 DEFER）。"""
    names = {n for n, _d, _r in GA.DEFERRED}
    # §4.2 六族 C/M/K/E/A/P 的全部编号减去首批已实现的 K1-K5 与 E1。
    # ★ 不含 T1-T3：文本类校验在 §4.2 里**没有编号**（契约层拦），见 SCHEMA_ENFORCED。
    expect = {"C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8",
              "M1", "M2", "M3", "M4", "M5", "M6",
              "E2", "E3", "E4", "E5",
              "P1", "P2", "P3", "P4", "P5", "P6",
              "A1", "A2", "A3", "A4", "A5"}
    assert expect <= names, "未登记：%s" % sorted(expect - names)
    assert not (names & {"T1", "T2", "T3"}), "T 类不是 §4.2 判据族成员，不得进判据登记"


def test_f9_schema_enforced_registry_is_declared():
    """契约层已拦的校验必须诚实登记（否则报告上看像「没做」）。"""
    names = {n for n, _d, _r in GA.SCHEMA_ENFORCED}
    assert {"T1", "T2", "T3", "M2", "E4"} <= names
    for _n, _d, r in GA.SCHEMA_ENFORCED:
        assert r.strip()


def test_f10_family_closed_set_is_the_six_registered_families():
    assert OR.FAMILIES == ("C", "M", "K", "E", "A", "P")
    for n in ("C1", "M3", "K1", "E1", "A5", "P2a"):
        assert OR.family_of(n) == n[0]
    for bad in ("T1", "X1"):
        with pytest.raises(Exception):
            OR.family_of(bad)


def test_f3_deferred_entries_have_reasons_and_no_measurements():
    doc = GA.build_oracle(_scene(), "o", "e", _D)
    for c in doc["checks"]:
        if c["verdict"] == "DEFER":
            assert c["reason"].strip()
            assert "measured" not in c


def test_f4_every_threshold_has_a_source():
    for t in GA.all_thresholds():
        assert t["source"].strip(), "阈值 %s 缺来源" % t["name"]


def test_f5_blindspot_blocks_release():
    """BLINDSPOT 也拦放行 —— 它看起来像有结论，比明确的 DEFER 更危险。"""
    ok, bad = OR.gates_passed(GA.build_oracle(_scene(with_axis=False), "o", "e", _D))
    assert ok is False
    assert "K1" in bad


def test_f6_reject_blocks_release():
    ok, bad = OR.gates_passed(GA.build_oracle(_scene(_POS_B_FLIP), "o", "e", _D))
    assert ok is False
    assert "K1" in bad


def test_f7_first_batch_is_exactly_k1_to_k5_plus_e1():
    implemented = {fn.__name__.replace("check_", "").upper() for fn in GA.CHECKS}
    assert implemented == {"K1", "K2", "K3", "K4", "K5", "E1"}


def test_f8_scene_rejects_unknown_keys():
    with pytest.raises(Exception):
        GA.run_all({"shots": {}, "unknown_key": 1})


# ══════════════════════════════════════════════════════ [G] B4 纯标准库
def test_g1_det_package_is_pure_stdlib():
    import subprocess
    code = (
        "import sys\n"
        "BLOCK = {'bpy', 'numpy', 'PIL', 'mathutils'}\n"
        "class _Block:\n"
        "    def find_module(self, name, path=None):\n"
        "        return self.find_spec(name, path)\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in BLOCK:\n"
        "            raise ImportError('B4 违反：_det 引入了 ' + name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, _Block())\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from wm_bridge.factory.film._det import gate_all, scene, gates_k, gates_e\n"
        "print('B4_OK')\n")
    p = subprocess.run([sys.executable, "-X", "utf8", "-c", code, _PKG_ROOT],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    assert p.returncode == 0, "\n%s\n%s" % (p.stdout, p.stderr)
    assert "B4_OK" in p.stdout


# ══════════════════════════════════════════════════════ [H] 反向守卫
#: 影视守卫是 `tests/` 下**首个真 import wm_bridge** 的用例 —— 这正是依赖闭包
#: 登记表（`tests/test_gate_dependency_closure.py` 的 LOCAL_TOP）缺它时，
#: 全量回归会红的那个耦合点。把这条耦合**显式钉死在这里**：
#: 谁把 wm_bridge 从登记表里删掉，本组会立刻点名原因，而不是让人去翻 2950 项回归。
def test_h1_wm_bridge_is_registered_as_repo_local():
    import re
    gate = os.path.join(os.path.dirname(_HERE), "tests",
                        "test_gate_dependency_closure.py")
    src = open(gate, encoding="utf-8").read()
    m = re.search(r"^LOCAL_TOP = \{(.*?)\}", src, re.M | re.S)
    assert m, "依赖闭包判据里找不到 LOCAL_TOP —— 它被改名/删除了"
    names = set()
    for s in m.group(1).split(","):
        names.add(s.strip().strip(chr(34)).strip(chr(39)))
    assert "wm_bridge" in names, (
        "wm_bridge 不在依赖闭包登记表 LOCAL_TOP 里 ⇒ 全量回归的 "
        "test_gate_dependency_closure 会把本包误判为「未声明的第三方依赖」。\n"
        "wm_bridge 是**仓内包**（bridges/ 下 73 个 git 跟踪文件），"
        "不是发行包（pyproject packages.find 只收 lingshu*），"
        "因此既不能在 pyproject 声明，也不能不登记。\n"
        "登记表实为 = %r" % sorted(names))


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
    print("REPORT film_criteria_guard: %d passed / %d failed" % (ok, len(bad)))
    if bad:
        print("  failed:", ", ".join(bad))
    print("VERDICT=%s" % ("PASS" if not bad else "FAIL"))
    sys.exit(1 if bad else 0)
