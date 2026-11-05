# -*- coding: utf-8 -*-
"""test_issue2106_2118_family_guard · bridges 族七处修复回归钉
============================================================================
本件钉 PR fix/bridges-spec-batch10 七处修复的**行为**（每条都是修前形态必红）：

  · #2106 timeline.cut_alignment_report：切点取 `end_frame + 1`——修前取
    `end_frame`（闭区间末帧 = downbeat−1）⇒ 完美时间线每刀 offset=-1、
    tolerance=0 下 100% 误报违例，与 build_edit 的「切点 = frame_end + 1」矛盾。
  · #2107 timeline.stub_score：包络长度 `int(round(dur/hop)) + 1`（含端点）——
    修前 ceil(dur/hop) 与 score 契约 `int(dur/hop)+1` 不自洽，整秒时长（8.0 等）
    下自产 ScoreError，127/200 个时长两式不一致。
  · #2108 timeline.plan_shot_table：细分循环段内无可用 downbeat ⇒ TimelineError
    ——修前 `_even_cuts([], 2)[0]` 裸 IndexError（无消息、违背模块 fail-closed
    错误面承诺）。
  · #2109 geo.spec.normalize_spec：`material` 显式 falsy（0/""/[]）不再被
    `or {}` 吞成默认材质 —— 修前与「省略 material」产出逐字节相同的 spec。
  · #2114 shade.material_plan：spec 声明的 emission/emission_strength 随条目
    透传并参与去重 key —— 修前两个不同发光声明产出完全相同的 recipe digest。
  · #2115 vec.bpy_vec._role_from：painted_metal 优先于 metal —— 修前裸子串匹配
    把 `painted_metal*` 认成 metal（两产线角色漂移）。
  · #2118 vec.bpy_vec.build_document：rim 死条件 `0.0 if not X else 0.0` 按意图
    恢复为 `0.0 if emissive else 0.22` —— 修前边缘光在静态产线永不生效，且与
    动画路径 `_object_fill`（默认 0.22）口径分裂。

运行（仓根）：python -X utf8 -m pytest tests/test_issue2106_2118_family_guard.py
"""
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BRIDGES = os.path.join(REPO, "bridges", "blender")
for p in (REPO, _BRIDGES):
    if p not in sys.path:
        sys.path.insert(0, p)


def _timeline():
    from wm_bridge.factory.film.mv._det import timeline as TL
    return TL


def _brief():
    from wm_bridge.factory.film import mv
    with open(mv.DEFAULT_BRIEF, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------- #2106

def test_2106_cut_report_agrees_with_build_edit():
    TL = _timeline()
    brief = _brief()
    sc = TL.stub_score(brief)
    bind = TL.build_bind(brief, sc)
    shots = TL.plan_shot_table(brief, sc, bind)
    edit = TL.build_edit(brief, shots, sc, bind)
    rep = TL.cut_alignment_report(edit, sc, tolerance_frames=0)
    # 末镜头的 end 是片尾不是切点（票内明写豁免）：报告 cuts 比 alignment 多末条
    r_off = [c["offset_frames"] for c in rep["cuts"]]
    assert all(o == 0 for o in r_off), (
        "完美卡点时间线在 tolerance=0 下不得误报违例（修前每刀 -1）：%r" % (r_off,))
    e_off = [a["offset_frames"] for a in edit["alignment"]]
    assert e_off[:len(edit["alignment"])] == r_off[:len(e_off)], (
        "前 n-1 个切点的报告偏移须与 build_edit 的 align 读数一致：%r vs %r"
        % (e_off, r_off[:len(e_off)]))


# ---------------------------------------------------------------- #2107

def test_2107_stub_score_envelope_matches_contract():
    TL = _timeline()
    brief = _brief()
    sc = TL.stub_score(brief)
    hop = sc["energy"]["hop_s"]
    dur = float(brief["duration_s"])
    want = int(round(dur / hop)) + 1
    assert len(sc["energy"]["values"]) == want, (
        "包络长度须含端点 round+1（修前 ceil 差一）：want %d got %d"
        % (want, len(sc["energy"]["values"])))


def test_2107_stub_score_whole_second_legal():
    """整秒时长（8.0s）：修前 ceil(8.0/0.05)=160 ≠ 161 ⇒ 自产 ScoreError。"""
    TL = _timeline()
    for dur in (1.0, 2.0, 3.0, 8.0):
        brief = {"fps": 24, "duration_s": dur,
                 "tempo": {"bpm": 120.0, "meter": [4, 4], "offset_s": 0.0},
                 "sections": [{"kind": "chorus", "start_s": 0.0,
                               "end_s": dur, "energy": 0.5}]}
        sc = TL.stub_score(brief)
        hop = sc["energy"]["hop_s"]
        assert len(sc["energy"]["values"]) == int(round(dur / hop)) + 1, (
            "dur=%r 的包络长度须与契约一致" % (dur,))


# ---------------------------------------------------------------- #2108

def test_2108_empty_candidate_is_timeline_error():
    """细分循环段内无 downbeat ⇒ TimelineError（修前裸 IndexError）。

    白盒：直接复现细分循环的取候选式（与源码同式），确认 _even_cuts 空输入
    返回空、并由调用点转成 TimelineError。行为面由源码结构保证（见 timeline.py
    细分循环的 `if not _mid_cands: raise TimelineError`）。
    """
    TL = _timeline()
    assert TL._even_cuts([], 2) == [], "空候选返回空列表（前置条件）"
    import inspect
    src = inspect.getsource(TL.plan_shot_table)
    assert "_mid_cands" in src and "TimelineError" in src, (
        "细分循环必须先取候选并显式 fail-closed 为 TimelineError")
    idx = src.find("_mid_cands")
    seg = src[idx:idx + 600]
    assert "raise TimelineError" in seg, (
        "空候选分支须 raise TimelineError（修前 `_even_cuts(...)[0]` 裸 IndexError）")


# ---------------------------------------------------------------- #2109

def _spec_base(TL_material=None):
    return {"schema": "wmfactory.spec/1", "name": "场景甲",
            "scene": {"world_size": 100.0}}


def _spec_with_material(mat):
    base = _spec_base()
    part = {"name": "p1", "kind": "box", "size": [1.0, 1.0, 1.0]}
    if mat is not None:
        part["material"] = mat
    base["links"] = [{"name": "base", "anchor": [0, 0, 0], "parts": [part]}]
    return base


def test_2109_explicit_falsy_material_rejected():
    from wm_bridge.factory.geo import spec as SP
    for bad in (0, "", [], False):
        with pytest.raises(SP.SpecError):
            SP.normalize_spec(_spec_with_material(bad))


def test_2109_omitted_material_still_defaults():
    from wm_bridge.factory.geo import spec as SP
    n = SP.normalize_spec(_spec_with_material(None))
    assert n["links"][0]["parts"][0]["material"]["role"] == "painted_metal"


# ---------------------------------------------------------------- #2114

def test_2114_emission_reaches_plan_entries():
    from wm_bridge.factory.shade import material as M
    spec = _spec_base()
    spec["links"] = [{"name": "base", "anchor": [0, 0, 0], "parts": [
        {"name": "glow_a", "kind": "box", "size": [1.0, 1.0, 1.0],
         "material": {"role": "emissive", "color": [1.0, 0.5, 0.06],
                      "emission": [1.0, 0.5, 0.06], "emission_strength": 5.0}},
        {"name": "glow_b", "kind": "box", "size": [1.0, 1.0, 1.0],
         "material": {"role": "emissive", "color": [1.0, 0.5, 0.06],
                      "emission": [0.0, 1.0, 0.0], "emission_strength": 2.0}}]}]
    plan = M.material_plan(spec)
    a, b = plan["entries"][0], plan["entries"][1]
    assert tuple(a["emission"]) == (1.0, 0.5, 0.06) and a["emission_strength"] == 5.0, (
        "spec 声明的 emission 必须透传到计划条目（修前全丢）：%r"
        % (a["emission"],))
    assert tuple(b["emission"]) == (0.0, 1.0, 0.0) and b["emission_strength"] == 2.0
    assert plan["distinct_recipes"] == 2, (
        "两个不同发光声明不得合并成同一配方：%r" % (plan["distinct_recipes"],))


# ---------------------------------------------------------------- #2115

def test_2115_painted_metal_wins():
    from wm_bridge.factory.vec import bpy_vec as BV

    class _Mat:
        def __init__(self, name):
            self.name = name
    assert BV._role_from(_Mat("painted_metal_hatch"), "x") == "painted_metal", (
        "painted_metal* 不得被裸子串 metal 抢走（修前 -> metal）")
    assert BV._role_from(_Mat("SHADE_pbr_painted_metal"), "x") == "painted_metal"
    assert BV._role_from(_Mat("bare_metal_plate"), "x") == "metal", (
        "真正的 metal 件仍判 metal（防过度收紧）")


# ---------------------------------------------------------------- #2118

def test_2118_rim_no_longer_dead_expression():
    import inspect
    from wm_bridge.factory.vec import bpy_vec as BV
    # 只看**代码行**（注释里留着修前串作历史记录，不算违规）
    code = "\n".join(ln for ln in inspect.getsource(BV.build_document).splitlines()
                     if not ln.lstrip().startswith("#"))
    assert "0.0 if not ob.emissive else 0.0" not in code, (
        "rim 两侧恒等的死表达式必须消失（修前静态产线边缘光永不生效）")
    assert "rim=0.0 if ob.emissive else 0.22" in code, (
        "rim 须按意图恢复（非发光件取默认 0.22）")


def test_2118_rim_capability_effective():
    from wm_bridge.factory.vec import bpy_vec as BV
    c0 = BV.shade_band((0.5, 0.5, 0.5), (0, 0, 1), (0, 0, 1),
                       view_dir=(0, 1, 0), rim=0.0)
    c1 = BV.shade_band((0.5, 0.5, 0.5), (0, 0, 1), (0, 0, 1),
                       view_dir=(0, 1, 0), rim=0.22)
    assert c0 != c1, "rim 能力本身有效（对照组：0 与 0.22 输出不同）"
