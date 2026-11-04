# -*- coding: utf-8 -*-
"""test_process_ops · 过程补全与过程重组判据（世界过程第 2 组）
============================================================================
被测模块：lingshu/world/process_ops.py

判据组：
  A 过程补全：观测前缀命中已记录过程，补全轨迹由记录后续采样生成（平移量取观测末帧
    ref 与已匹配记录末帧 ref 之差，首帧 tick 接在前缀末 tick 之后）；形状不同的前缀
    不命中，返回 match=None 与空 frames；逐帧容差边界（差 0.6 在默认 0.5 下不命中、
    阈值放到 0.7 即命中）；对比帧数不足与无采样记录都不抛异常；候选来源、几何与时序
    读数与匹配分数逐条给出并按分数降序；前缀 tick 平移后补全 tick 跟着前缀走，不照抄
    记录 tick；历史与记录不被改写，重复调用同输出。
  A2 时序判据：ref 移动时补全尾部逐帧还原记录的后续帧，观测场景整体平移后等式仍成立；
    elapsed tick 不同的记录不命中（逐帧几何相同也不行），绝对 tick 整体平移仍命中；
    非均匀正 dt 与缺帧照常处理，少于 min_match_samples 仍走原逻辑；同名角色被拒绝。
  B 过程重组：显式 roles 重绑定角色后帧里只有新角色名；每帧 actor − ref 等于该段记录
    的相对偏移；连接点把后段 ref 锚定到前段末帧 ref（ref_gap ≈ 0），actor 衔接距离写进 joint；
    joint 另给 ref_y_gap 与 actor_y_gap 原始高度差，continuous 仍只按水平 x/z 判定；
    偏移差 4.0 的后段判为 continuous=False 且仍生成新轨迹，阈值放宽到 5.0 即判连续；
    relations 给出两段的类型、首末阶段与角色关系；输入不被改写、重复调用同输出；
    yaw 与 scale 沿用 transfer_demonstration 的整体场景变换；
    前置条件不满足（记录无采样、roles 缺键、解析后 actor 与 ref 同名、tick_gap < 1）
    抛 ValueError。
  C 定点变异：M1 匹配分数下限失效、M2 接续 tick 不加该采样 dt、M3 连接点判据恒真、
    M4 后段不锚定前段末帧 ref、M5 角色重绑定失效、M6 候选排序反向、M7 尾部平移退回
    “首条未观测采样放到观测末位置”、M8 命中帧不再要求 elapsed tick 相同、M9 同名角色
    校验失效，九项各自在隔离副本里让目标断言变红；工作树源码 sha256 不变。
  D 边界：模块不调用 perceive/generate/execute，也不写 actual。

运行（仓根）：
    python -X utf8 -m pytest tests/test_process_ops.py -q -p no:cacheprovider
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import json
import math
import os
import shutil
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.world.process_ops import (ProcessOpsThresholds,      # noqa: E402
                                       complete_process,
                                       compose_processes)
from lingshu.world.world_model import UnifiedWorldModel           # noqa: E402
from lingshu.world.world_process import (extract_relative_trajectory,  # noqa: E402
                                        recognize_processes)

MODULE_PATH = os.path.join(REPO, "lingshu", "world", "process_ops.py")
GENERATED = "generated"


# ================= 夹具 =================

def real_history(tracks):
    """tracks = {eid: [[x, y, z], ...]} → UnifiedWorldModel 的真实观测 history。"""
    wm = UnifiedWorldModel(size=24, seed=7)
    for i in range(len(next(iter(tracks.values())))):
        wm.perceive([{"eid": eid, "category": "c", "pos": tuple(track[i])}
                     for eid, track in tracks.items()], tick=i + 1)
    return wm.history


def approach_history():
    """接近 4 tick → 停留 4 tick → 离开 4 tick；参照物 landmark 不动。"""
    return real_history({
        "landmark": [[10, 1, 10]] * 12,
        "walker": [[4, 1, 10], [5, 1, 10], [6, 1, 10], [7, 1, 10], [8, 1, 10], [8.2, 1, 10],
                   [8, 1, 10], [8.1, 1, 10], [7, 1, 10], [5, 1, 10], [3, 1, 10], [1, 1, 10]]})


def partial_history():
    """前 6 帧里后 2 帧相对偏移比 approach_history 多 1.0：匹配分数 4/6。"""
    return real_history({
        "landmark": [[10, 1, 10]] * 12,
        "walker": [[4, 1, 10], [5, 1, 10], [6, 1, 10], [7, 1, 10], [9, 1, 10], [9.2, 1, 10],
                   [8, 1, 10], [8.1, 1, 10], [7, 1, 10], [5, 1, 10], [3, 1, 10], [1, 1, 10]]})


def stray_history():
    """形状与记录不同的前缀：逐帧偏移都差 20 以上。"""
    return real_history({
        "landmark": [[10, 1, 10]] * 6,
        "walker": [[30, 1, 10], [28, 1, 10], [26, 1, 10], [24, 1, 10], [22, 1, 10],
                   [20, 1, 10]]})


def shifted_prefix_history():
    """逐帧偏移整体差 0.6 的前缀：默认容差 0.5 不命中，0.7 命中。"""
    return real_history({
        "landmark": [[10, 1, 10]] * 6,
        "walker": [[4.6, 1, 10], [5.6, 1, 10], [6.6, 1, 10], [7.6, 1, 10], [8.6, 1, 10],
                   [8.8, 1, 10]]})


def retick(history, offset):
    """同形 history 的 tick 整体平移，返回新记录（入参不改写）。"""
    return [{"tick": rec["tick"] + offset, "entities": dict(rec["entities"])}
            for rec in history]


def offsets_of(history, actor, ref):
    """测试件另算一遍：history 里 actor 相对 ref 的水平偏移序列。"""
    out = []
    for rec in history:
        a, b = rec["entities"].get(actor), rec["entities"].get(ref)
        if a is None or b is None:
            continue
        out.append((a[0] - b[0], a[2] - b[2]))
    return out


def offsets_of_record(record):
    """测试件另算一遍：记录采样的水平相对偏移序列。"""
    return [(s["rel"][0], s["rel"][2]) for s in record["samples"]]


def frame_offset(frame, actor, ref):
    """帧内 actor 相对 ref 的三维偏移；角色名缺失时当场断言失败。"""
    assert actor in frame["entities"] and ref in frame["entities"], \
        "重组帧必须只带新绑定的角色名，实为 %r" % sorted(frame["entities"])
    return [frame["entities"][actor][k] - frame["entities"][ref][k] for k in range(3)]


def seg_a_history():
    """前段：rider 从离 beacon 4.0 走到 2.0。"""
    return real_history({
        "beacon": [[0, 1, 0]] * 5,
        "rider": [[4, 1, 0], [3.5, 1, 0], [3, 1, 0], [2.5, 1, 0], [2, 1, 0]]})


def seg_b_history():
    """后段：rider 从离 beacon 2.0 起走，起点偏移等于前段终点偏移。"""
    return real_history({
        "beacon": [[0, 1, 0]] * 5,
        "rider": [[2, 1, 0], [1.5, 1, 0.5], [1, 1, 1], [0.5, 1, 1.5], [0, 1, 2]]})


def seg_c_history():
    """后段（不衔接）：起点偏移 6.0，与前段终点偏移差 4.0。"""
    return real_history({
        "beacon": [[0, 1, 0]] * 5,
        "rider": [[6, 1, 0], [5, 1, 0.5], [4, 1, 1], [3, 1, 1.5], [2, 1, 2]]})


def seg_record(history, process=None):
    """一段观测 → 已记录过程（extract_relative_trajectory 的记录结构）。"""
    return extract_relative_trajectory(history, "rider", "beacon", process=process)


def sparse_history(tracks, ticks=None):
    """按 tick 列表构造观测：track[i] 为 None 表示该帧缺该实体。

    ticks 省略时用 1..n。采样落在给定 tick 上，缺观测处不补帧、不插值。
    """
    length = len(next(iter(tracks.values())))
    ticks = list(range(1, length + 1)) if ticks is None else list(ticks)
    wm = UnifiedWorldModel(size=64, seed=7)
    for i, tick in enumerate(ticks):
        frame = [{"eid": eid, "category": "c", "pos": tuple(track[i])}
                 for eid, track in tracks.items() if track[i] is not None]
        wm.perceive(frame, tick=tick)
    return wm.history


def sparse_from_record(record, ticks):
    """按记录的 ref 与 rel 造同形观测，采样落在给定 ticks 上。"""
    return sparse_history(
        {"leader": [list(s["ref"]) for s in record["samples"]],
         "walker": [[s["ref"][k] + s["rel"][k] for k in range(3)]
                    for s in record["samples"]]}, ticks)


def shift_history(history, shift):
    """整个观测场景平移：每帧每个实体的位置加上同一个向量。"""
    return [{"tick": rec["tick"],
             "entities": {eid: [pos[k] + shift[k] for k in range(3)]
                          for eid, pos in rec["entities"].items()}}
            for rec in history]


def retick_record(record, offset):
    """记录整体 tick 平移，返回新记录（入参不改写）。"""
    return {"kind": record.get("kind"), "actor": record.get("actor"),
            "ref": record.get("ref"), "process": record.get("process"),
            "ticks": [t + offset for t in record["ticks"]],
            "samples": [{"tick": int(s["tick"]) + offset, "dt": int(s["dt"]),
                         "ref": list(s["ref"]), "rel": list(s["rel"])}
                        for s in record["samples"]]}


def moving_ref_history():
    """参照物每 tick 走 +1.0，执行者相对偏移逐帧收窄 0.25。"""
    leader = [[10.0 + i, 1.0, 10.0] for i in range(12)]
    walker = [[10.0 + i + 6.0 - 0.25 * i, 1.0, 10.0] for i in range(12)]
    return real_history({"leader": leader, "walker": walker})


def gap_tracks():
    """leader 每 tick 走 0.5，walker 相对偏移逐帧收窄 0.1。"""
    leader = [[20.0 + 0.5 * i, 1.0, 5.0] for i in range(18)]
    walker = [[20.0 + 0.5 * i + 3.0 - 0.1 * i, 1.0, 5.0] for i in range(18)]
    return {"leader": leader, "walker": walker}


def gapped_history():
    """walker 在第 4、11 tick 缺观测：成对采样 16 条，dt 为 1 与 2 混合。"""
    tracks = gap_tracks()
    tracks["walker"][3] = None
    tracks["walker"][10] = None
    return sparse_history(tracks)


def uniform_history():
    """同一轨迹不缺口：成对采样 18 条，dt 全为 1。"""
    return sparse_history(gap_tracks())


def described_record():
    """带过程标注的记录：approach_stay_leave（actor=walker，ref=landmark）。"""
    history = approach_history()
    processes = recognize_processes(history)
    assert processes and processes[0]["type"] == "approach_stay_leave", processes
    return extract_relative_trajectory(history, "walker", "landmark",
                                       process=processes[0])


# ================= A 组：过程补全 =================

def test_completion_hits_recorded_tail_from_observed_prefix():
    """前缀命中：补全轨迹取记录后续采样，相对偏移与 tick 都能对上。"""
    full = approach_history()
    record = extract_relative_trajectory(full, "walker", "landmark")
    got = complete_process(full[:6], "walker", "landmark", [record])

    assert got["kind"] == "process_completion"
    assert got["source"] == GENERATED, "结果必须标生成帧来源"
    assert got["actor"] == "walker" and got["ref"] == "landmark"
    assert got["prefix"]["ticks"] == [1, 6] and got["prefix"]["samples"] == 6

    candidate = got["candidates"][0]
    assert candidate["matched"] is True and candidate["source"]["index"] == 0
    assert candidate["matched_samples"] == 6 and candidate["tail_samples"] == 6
    assert candidate["hit_frames"] == 6 and candidate["match_score"] == 1.0
    assert candidate["max_frame_error"] == 0.0 and candidate["mean_frame_error"] == 0.0
    assert candidate["reason"] == ""
    assert got["match"]["source"]["index"] == 0
    assert got["match"]["matched"] is True

    # 测试件另算一遍逐帧误差：前缀偏移与记录偏移逐条相同。
    errors = [math.hypot(a[0] - b[0], a[1] - b[1])
              for a, b in zip(offsets_of(full[:6], "walker", "landmark"),
                              offsets_of_record(record)[:6])]
    assert max(errors) == 0.0

    frames = got["frames"]
    assert len(frames) == 6 and got["ticks"] == [7, 12]
    anchor = full[5]["entities"]["landmark"]
    for frame, sample in zip(frames, record["samples"][6:]):
        assert frame["source"] == GENERATED
        assert frame["dt"] == sample["dt"]
        assert frame_offset(frame, "walker", "landmark") == pytest.approx(sample["rel"])
        assert frame["entities"]["landmark"] == pytest.approx(list(anchor))
    assert [f["tick"] for f in frames] == [7, 8, 9, 10, 11, 12]
    # 接缝读数：前缀末帧 actor 与补全首帧 actor 的水平距离 = 记录在已匹配末帧
    # 与后续首帧之间的位移（walker 从离 1.8 走到 2.0）。
    last_observed = full[5]["entities"]["walker"]
    first_generated = frames[0]["entities"]["walker"]
    assert got["seam"]["actor_gap"] == pytest.approx(
        math.hypot(last_observed[0] - first_generated[0],
                   last_observed[2] - first_generated[2]), abs=1e-12)
    assert got["seam"]["actor_gap"] == pytest.approx(0.2, abs=1e-9)
    assert got["seam"]["first_tick"] == 7 and got["seam"]["prefix_tick"] == 6


def test_completion_miss_returns_no_frames():
    """前缀形状不同：不命中，返回空轨迹，候选仍给出分数与原因。"""
    record = extract_relative_trajectory(approach_history(), "walker", "landmark")
    got = complete_process(stray_history(), "walker", "landmark", [record])

    assert got["match"] is None
    assert got["frames"] == [] and got["ticks"] == [] and got["seam"] is None
    candidate = got["candidates"][0]
    assert candidate["matched"] is False
    assert candidate["match_score"] == 0.0 and candidate["hit_frames"] == 0
    assert candidate["max_frame_error"] > 0.5
    assert "低于下限" in candidate["reason"]
    assert candidate["source"]["index"] == 0 and candidate["source"]["samples"] == 12


def test_completion_tolerance_boundary_decides_hit():
    """逐帧差 0.6：默认容差 0.5 不命中，容差 0.7 命中。"""
    record = extract_relative_trajectory(approach_history(), "walker", "landmark")
    prefix = shifted_prefix_history()

    tight = complete_process(prefix, "walker", "landmark", [record])
    assert tight["match"] is None and tight["frames"] == []
    assert tight["candidates"][0]["max_frame_error"] == pytest.approx(0.6, abs=1e-9)

    loose = complete_process(prefix, "walker", "landmark", [record],
                             thresholds=ProcessOpsThresholds(match_tol=0.7))
    assert loose["match"] is not None
    assert loose["candidates"][0]["match_score"] == 1.0
    assert len(loose["frames"]) == 6


def test_completion_prefix_too_short_or_record_empty_is_not_a_hit():
    """对比帧数不足、记录没有采样：不命中，不抛异常。"""
    record = extract_relative_trajectory(approach_history(), "walker", "landmark")
    short = complete_process(approach_history()[:2], "walker", "landmark", [record])
    assert short["prefix"]["samples"] == 2
    assert short["candidates"][0]["matched"] is False
    assert short["candidates"][0]["matched_samples"] == 2
    assert "对比帧数" in short["candidates"][0]["reason"]
    assert short["match"] is None and short["frames"] == []

    empty = {"kind": "relative_trajectory", "actor": "walker", "ref": "landmark",
             "process": None, "ticks": [], "samples": []}
    absent = complete_process(approach_history()[:6], "walker", "landmark", [empty])
    assert absent["candidates"][0]["reason"] == "前缀与记录没有可比采样"
    assert absent["match"] is None and absent["frames"] == []


def test_completion_candidates_carry_source_and_score():
    """候选逐条给出来源与匹配分数，按分数降序；换顺序不改命中的记录。"""
    full = approach_history()
    record = extract_relative_trajectory(full, "walker", "landmark")
    partial = extract_relative_trajectory(partial_history(), "walker", "landmark")
    stray = extract_relative_trajectory(stray_history(), "walker", "landmark")
    th = ProcessOpsThresholds(min_hit_ratio=0.5)
    got = complete_process(full[:6], "walker", "landmark", [stray, partial, record],
                           thresholds=th)

    candidates = got["candidates"]
    assert len(candidates) == 3, "候选必须逐条给出，不因不命中而省略"
    assert [c["source"]["index"] for c in candidates] == [2, 1, 0]
    assert [c["matched"] for c in candidates] == [True, True, False]
    assert candidates[0]["match_score"] == 1.0
    assert candidates[1]["match_score"] == pytest.approx(4.0 / 6.0, abs=1e-6)
    assert candidates[2]["match_score"] == 0.0
    assert candidates[0]["match_score"] > candidates[1]["match_score"], \
        "命中候选必须按匹配分数降序给出"
    for candidate, source_record in zip(candidates, (record, partial, stray)):
        assert set(candidate["source"]) == {"index", "kind", "actor", "ref",
                                            "ticks", "samples"}
        assert candidate["source"]["kind"] == "relative_trajectory"
        assert candidate["source"]["actor"] == "walker"
        assert candidate["source"]["ref"] == "landmark"
        assert candidate["source"]["ticks"] == source_record["ticks"]
        assert candidate["source"]["samples"] == len(source_record["samples"])
        assert candidate["hit_frames"] == round(candidate["match_score"]
                                                * candidate["matched_samples"])
    assert candidates[2]["source"]["samples"] == 6, "候选来源按各自记录给出"

    # 测试件另算一遍分数：命中帧 = 逐帧偏移差不超过容差。
    errors = [math.hypot(a[0] - b[0], a[1] - b[1])
              for a, b in zip(offsets_of(full[:6], "walker", "landmark"),
                              offsets_of_record(partial)[:6])]
    assert sum(1 for e in errors if e <= th.match_tol) == 4
    assert got["match"]["source"]["index"] == 2
    assert got["match"]["source"]["ticks"] == [1, 12]

    reordered = complete_process(full[:6], "walker", "landmark", [record, stray, partial],
                                 thresholds=th)
    assert reordered["match"]["source"]["index"] == 0
    assert reordered["match"]["source"]["ticks"] == got["match"]["source"]["ticks"]
    assert reordered["ticks"] == got["ticks"]


def test_completion_tick_continues_observed_prefix():
    """前缀 tick 平移到 101 起：补全 tick 跟前缀走，不照抄记录 tick。"""
    record = extract_relative_trajectory(approach_history(), "walker", "landmark")
    late = retick(approach_history()[:6], 100)
    got = complete_process(late, "walker", "landmark", [record])

    assert got["prefix"]["ticks"] == [101, 106]
    assert got["match"]["source"]["ticks"] == [1, 12], "记录 tick 原样上报"
    assert [f["tick"] for f in got["frames"]] == [107, 108, 109, 110, 111, 112]
    assert got["ticks"] == [107, 112]
    assert got["seam"]["prefix_tick"] == 106 and got["seam"]["first_tick"] == 107
    assert got["seam"]["step"] == 1
    assert all(f["dt"] == 1 and f["source"] == GENERATED for f in got["frames"])
    assert record["samples"][0]["tick"] == 1, "记录自身 tick 不变"


def test_completion_inputs_unchanged_and_deterministic():
    """输入不变量：前缀与记录不改写、结果与输入不共享对象、重复调用同输出。"""
    full = approach_history()
    prefix = full[:6]
    records = [extract_relative_trajectory(full, "walker", "landmark"),
               extract_relative_trajectory(stray_history(), "walker", "landmark")]
    prefix_snapshot = copy.deepcopy(prefix)
    records_snapshot = copy.deepcopy(records)

    first = complete_process(prefix, "walker", "landmark", records)
    second = complete_process(prefix, "walker", "landmark", records)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert prefix == prefix_snapshot and records == records_snapshot

    first["frames"][0]["entities"]["walker"][0] += 100.0
    first["candidates"][0]["source"]["ticks"].append(999)
    first["match"]["source"]["ticks"].append(999)
    assert records == records_snapshot, "结果与记录不得共享可变对象"
    assert prefix == prefix_snapshot


def test_completion_restores_moving_ref_tail_and_follows_scene_shift():
    """ref 移动：前缀等于记录前 6 帧时，尾部逐帧还原；整体平移后等式仍成立。"""
    full = moving_ref_history()
    record = extract_relative_trajectory(full, "walker", "leader")
    prefix_snapshot = copy.deepcopy(full[:6])
    got = complete_process(full[:6], "walker", "leader", [record])

    assert got["match"] is not None
    candidate = got["candidates"][0]
    assert candidate["matched"] is True and candidate["timing"]["matched"] is True
    assert candidate["geometry"]["score"] == 1.0
    assert candidate["tail_samples"] == 6

    frames = got["frames"]
    assert len(frames) == 6 and got["ticks"] == [7, 12]
    for frame, observed in zip(frames, full[6:]):
        assert frame["tick"] == observed["tick"]
        assert frame["entities"]["leader"] == pytest.approx(
            observed["entities"]["leader"]), "移动 ref 的补全尾部必须逐帧还原观测"
        assert frame["entities"]["walker"] == pytest.approx(
            observed["entities"]["walker"])
    # 观测与记录同源：平移量 = 观测末帧 ref − 已匹配记录末帧 ref = 0。
    assert got["match"]["translation"] == pytest.approx([0.0, 0.0, 0.0])
    assert got["match"]["matched_record_index"] == 5
    assert got["seam"]["matched_record_tick"] == 6
    assert got["seam"]["ref_anchor"] == pytest.approx(full[5]["entities"]["leader"])

    # 整个观测场景平移、记录不动：平移量等于场景位移，尾部等于观测的后续帧。
    shift = (100.0, 5.0, -30.0)
    moved = shift_history(full, shift)
    moved_got = complete_process(moved[:6], "walker", "leader", [record])
    assert moved_got["match"] is not None
    assert moved_got["match"]["translation"] == pytest.approx(list(shift))
    for frame, observed in zip(moved_got["frames"], moved[6:]):
        assert frame["tick"] == observed["tick"]
        assert frame["entities"]["leader"] == pytest.approx(observed["entities"]["leader"])
        assert frame["entities"]["walker"] == pytest.approx(observed["entities"]["walker"])
    assert full[:6] == prefix_snapshot, "补全不得改写观测前缀"


def test_completion_requires_matching_elapsed_ticks():
    """采样节奏：elapsed tick 不同的记录不命中，绝对 tick 整体平移仍命中。"""
    full = moving_ref_history()
    record = extract_relative_trajectory(full, "walker", "leader")
    # 与记录逐帧同形、4 tick 一帧的观测：几何全中，采样节奏差 4 倍。
    sparse = sparse_from_record(record, [1, 5, 9, 13, 17, 21])

    tight = complete_process(sparse, "walker", "leader", [record])
    loose = complete_process(sparse, "walker", "leader", [record],
                             thresholds=ProcessOpsThresholds(min_hit_ratio=0.5))
    for got in (tight, loose):
        candidate = got["candidates"][0]
        assert candidate["matched"] is False
        assert candidate["timing"]["matched"] is False
        assert candidate["timing"]["hit_frames"] == 1
        assert candidate["timing"]["score"] == pytest.approx(1.0 / 6.0, abs=1e-6)
        assert candidate["timing"]["deltas"] == [0, 3, 6, 9, 12, 15]
        assert candidate["timing"]["max_abs_delta_ticks"] == 15
        assert candidate["geometry"]["hit_frames"] == 6
        assert candidate["geometry"]["score"] == 1.0
        assert candidate["match_score"] == pytest.approx(1.0 / 6.0, abs=1e-6)
        assert "采样节奏" in candidate["reason"]
        assert got["match"] is None
        assert got["frames"] == [] and got["ticks"] == [] and got["seam"] is None

    # 等 dt：观测整体 +100 仍命中，输出 tick 跟前缀走。
    late = retick(full[:6], 100)
    late_got = complete_process(late, "walker", "leader", [record])
    assert late_got["match"] is not None
    assert late_got["candidates"][0]["timing"]["matched"] is True
    assert [f["tick"] for f in late_got["frames"]] == [107, 108, 109, 110, 111, 112]

    # 等 dt：记录整体 +500 仍命中，输出 tick 不用记录的绝对 tick。
    moved = retick_record(record, 500)
    moved_got = complete_process(full[:6], "walker", "leader", [moved])
    assert moved_got["match"] is not None
    assert moved_got["candidates"][0]["timing"]["matched"] is True
    assert moved_got["match"]["source"]["ticks"] == [501, 512]
    assert [f["tick"] for f in moved_got["frames"]] == [7, 8, 9, 10, 11, 12]
    assert moved_got["seam"]["matched_record_tick"] == 506


def test_completion_handles_irregular_dt_and_missing_frames():
    """非均匀正 dt 与缺帧：同节奏的记录照常命中，缺帧处不插值补帧。"""
    full = gapped_history()
    record = extract_relative_trajectory(full, "walker", "leader")
    dts = [s["dt"] for s in record["samples"]]
    assert len(record["samples"]) == 16
    assert min(dts[1:]) == 1 and max(dts) == 2, "成对采样的 dt 非均匀"

    got = complete_process(full[:7], "walker", "leader", [record])
    assert got["prefix"]["ticks"] == [1, 7] and got["prefix"]["samples"] == 6
    candidate = got["candidates"][0]
    assert candidate["matched"] is True
    assert candidate["timing"]["matched"] is True
    assert candidate["timing"]["deltas"] == [0] * 6
    assert candidate["matched_samples"] == 6 and candidate["tail_samples"] == 10
    frames = got["frames"]
    assert [f["tick"] for f in frames] == [8, 9, 10, 12, 13, 14, 15, 16, 17, 18]
    assert got["ticks"] == [8, 18]
    assert 4 not in [f["tick"] for f in frames], "缺帧处不得插值补帧"
    assert 11 not in [f["tick"] for f in frames], "缺帧处不得插值补帧"
    expected = {rec["tick"]: rec for rec in full}
    for frame in frames:
        observed = expected[frame["tick"]]
        assert frame["entities"]["walker"] == pytest.approx(
            observed["entities"]["walker"])
        assert frame["entities"]["leader"] == pytest.approx(
            observed["entities"]["leader"])

    # 缺帧的观测配均匀采样的记录：elapsed 不同，不命中。
    miss = complete_process(full[:7], "walker", "leader",
                            [extract_relative_trajectory(uniform_history(),
                                                         "walker", "leader")])
    assert miss["match"] is None and miss["frames"] == []
    assert miss["candidates"][0]["timing"]["matched"] is False
    assert "采样节奏" in miss["candidates"][0]["reason"]

    # 少于 min_match_samples 仍走原逻辑：对比帧数不足，不报节奏问题。
    short = complete_process(gapped_history()[:2], "walker", "leader", [record])
    assert short["prefix"]["samples"] == 2
    assert short["candidates"][0]["matched"] is False
    assert "对比帧数" in short["candidates"][0]["reason"]


def test_same_named_roles_raise_before_output():
    """actor 与 ref 同名时两个入口都抛 ValueError，避免帧 entities 互相覆写。"""
    full = approach_history()
    record = extract_relative_trajectory(full, "walker", "landmark")
    with pytest.raises(ValueError, match="actor 与 ref"):
        complete_process(full[:6], "walker", "walker", [record])
    with pytest.raises(ValueError, match="actor 与 ref"):
        complete_process(full[:6], "landmark", "landmark", [record])

    first, second = seg_record(seg_a_history()), seg_record(seg_b_history())
    with pytest.raises(ValueError, match="actor 与 ref"):
        compose_processes(first, second, roles={"actor": "hero", "ref": "hero"})
    same_named = {"kind": "relative_trajectory", "actor": "hero", "ref": "hero",
                  "process": None, "ticks": [1, 2],
                  "samples": [{"tick": 1, "dt": 0, "ref": [0.0, 0.0, 0.0],
                               "rel": [0.0, 0.0, 0.0]},
                              {"tick": 2, "dt": 1, "ref": [0.0, 0.0, 0.0],
                               "rel": [1.0, 0.0, 0.0]}]}
    with pytest.raises(ValueError, match="actor 与 ref"):
        compose_processes(same_named, second)
    # 角色不同名时照常产出帧，帧里两个角色都在。
    ok = compose_processes(first, second, roles={"actor": "hero", "ref": "mark"})
    assert set(ok["frames"][0]["entities"]) == {"hero", "mark"}


def test_new_completion_paths_keep_inputs_unchanged():
    """新判据路径不改写输入：前缀、记录与稀疏观测调用前后逐位相同，重复调用同输出。"""
    full = moving_ref_history()
    record = extract_relative_trajectory(full, "walker", "leader")
    prefix = full[:6]
    sparse = sparse_from_record(record, [1, 5, 9, 13, 17, 21])
    snapshots = [copy.deepcopy(prefix), copy.deepcopy(record), copy.deepcopy(sparse)]

    hit_one = complete_process(prefix, "walker", "leader", [record])
    hit_two = complete_process(prefix, "walker", "leader", [record])
    miss_one = complete_process(sparse, "walker", "leader", [record])
    miss_two = complete_process(sparse, "walker", "leader", [record])
    assert json.dumps(hit_one, sort_keys=True) == json.dumps(hit_two, sort_keys=True)
    assert json.dumps(miss_one, sort_keys=True) == json.dumps(miss_two, sort_keys=True)
    assert [prefix, record, sparse] == snapshots

    hit_one["frames"][0]["entities"]["leader"][0] += 1000.0
    hit_one["candidates"][0]["timing"]["deltas"].append(999)
    hit_one["match"]["source"]["ticks"].append(999)
    miss_one["candidates"][0]["geometry"]["hit_frames"] = 0
    assert [prefix, record, sparse] == snapshots, "结果与输入不得共享可变对象"


# ================= B 组：过程重组 =================

def test_composition_binds_roles_and_keeps_segment_geometry():
    """角色重绑定：帧里只有新角色名，逐段相对偏移等于记录原值。"""
    first, second = seg_record(seg_a_history()), seg_record(seg_b_history())
    got = compose_processes(first, second, roles={"actor": "hero", "ref": "mark"},
                            translation=(10.0, 0.0, 5.0))

    assert got["kind"] == "process_composition" and got["source"] == GENERATED
    assert got["actor"] == "hero" and got["ref"] == "mark"
    assert got["relations"]["roles_rebound"] is True
    assert len(got["frames"]) == 10
    for frame in got["frames"]:
        assert set(frame["entities"]) == {"hero", "mark"}, \
            "重组帧必须只带新绑定的角色名"
        assert frame["source"] == GENERATED

    for segment, record in zip(got["segments"], (first, second)):
        assert len(segment["frames"]) == len(record["samples"]) == 5
        assert segment["roles"] == {"actor": "rider", "ref": "beacon"}
        for frame, sample in zip(segment["frames"], record["samples"]):
            assert frame_offset(frame, "hero", "mark") == pytest.approx(sample["rel"],
                                                                       abs=1e-9)
            assert frame["dt"] == sample["dt"]
    assert got["transform"]["translation"] == [10.0, 0.0, 5.0]
    assert got["transform"]["tick_gap"] == 1


def test_composition_joint_aligns_ref_and_checks_seam():
    """连接点：后段 ref 锚定前段末帧 ref，tick 相接，衔接偏移为 0 判连续。"""
    first, second = seg_record(seg_a_history()), seg_record(seg_b_history())
    got = compose_processes(first, second, roles={"actor": "hero", "ref": "mark"},
                            translation=(10.0, 0.0, 5.0))
    head, tail = got["segments"][0]["frames"], got["segments"][1]["frames"]
    joint = got["joint"]

    assert all(set(f["entities"]) == {"hero", "mark"} for f in got["frames"])
    assert joint["on"] == "ref"
    assert joint["tick"] == head[-1]["tick"] == 5
    assert joint["ref"] == pytest.approx(head[-1]["entities"]["mark"])
    assert joint["ref_gap"] <= 1e-9
    assert tail[0]["entities"]["mark"] == pytest.approx(head[-1]["entities"]["mark"],
                                                        abs=1e-9)
    assert joint["actor_gap"] == pytest.approx(0.0, abs=1e-9)
    assert joint["offset_end_first"] == joint["offset_start_second"] == [2.0, 0.0]
    assert joint["joint_tol"] == 0.5 and joint["continuous"] is True
    assert [f["tick"] for f in got["frames"]] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert got["ticks"] == [1, 10]
    # 连接点两侧的 actor 位置：前段终点 (12,1,5) 与后段起点 (12,1,5)。
    assert head[-1]["entities"]["hero"] == pytest.approx([12.0, 1.0, 5.0])
    assert tail[0]["entities"]["hero"] == pytest.approx([12.0, 1.0, 5.0])


def test_composition_reports_discontinuous_joint():
    """衔接不达标：continuous=False 并给出读数，轨迹照常生成。"""
    first, broken = seg_record(seg_a_history()), seg_record(seg_c_history())
    got = compose_processes(first, broken, roles={"actor": "hero", "ref": "mark"},
                            translation=(10.0, 0.0, 5.0))
    joint = got["joint"]

    assert joint["offset_end_first"] == [2.0, 0.0]
    assert joint["offset_start_second"] == [6.0, 0.0]
    assert joint["ref_gap"] <= 1e-9, "连接点仍锚定到前段末帧 ref"
    assert joint["actor_gap"] == pytest.approx(4.0)
    assert joint["actor_gap"] > joint["joint_tol"]
    assert joint["continuous"] is False
    assert len(got["frames"]) == 10, "连接点不达标仍生成新轨迹"

    loose = compose_processes(first, broken, roles={"actor": "hero", "ref": "mark"},
                              translation=(10.0, 0.0, 5.0),
                              thresholds=ProcessOpsThresholds(joint_tol=5.0))
    assert loose["joint"]["continuous"] is True
    assert loose["joint"]["actor_gap"] == pytest.approx(4.0)


def test_composition_reports_process_relations():
    """过程关系：两段的类型、首末阶段与角色关系逐项给出。"""
    plain = compose_processes(seg_record(seg_a_history()), seg_record(seg_b_history()),
                              roles={"actor": "hero", "ref": "mark"})
    relations = plain["relations"]
    assert relations["shared_actor"] is True and relations["shared_ref"] is True
    assert relations["first"]["type"] is None and relations["second"]["type"] is None
    assert relations["same_type"] is None, "记录没有过程标注时不得猜类型关系"
    assert relations["first"]["start_phase"] is None
    assert relations["second"]["end_phase"] is None
    assert relations["first"]["roles"] == {}

    described = described_record()
    same = compose_processes(described, described)
    srel = same["relations"]
    assert srel["same_type"] is True
    assert srel["first"]["type"] == srel["second"]["type"] == "approach_stay_leave"
    assert srel["first"]["start_phase"] == "approach"
    assert srel["first"]["end_phase"] == "leave"
    assert srel["second"]["start_phase"] == "approach"
    assert srel["roles_rebound"] is False, "省略 roles 时沿用前段角色名"
    assert srel["actor"] == "walker" and srel["ref"] == "landmark"
    assert srel["first"]["roles"] == {"actor": "walker", "ref": "landmark"}
    assert srel["shared_actor"] is True and srel["shared_ref"] is True


def test_composition_inputs_unchanged_and_deterministic():
    """输入不变量：两段记录不改写、结果与记录不共享对象、重复调用同输出。"""
    first, second = seg_record(seg_a_history()), seg_record(seg_b_history())
    snapshot_first, snapshot_second = copy.deepcopy(first), copy.deepcopy(second)

    one = compose_processes(first, second, roles={"actor": "hero", "ref": "mark"},
                            translation=(10.0, 0.0, 5.0))
    two = compose_processes(first, second, roles={"actor": "hero", "ref": "mark"},
                            translation=(10.0, 0.0, 5.0))
    assert json.dumps(one, sort_keys=True) == json.dumps(two, sort_keys=True)
    assert first == snapshot_first and second == snapshot_second

    assert set(one["frames"][0]["entities"]) == {"hero", "mark"}
    one["frames"][0]["entities"]["hero"][0] += 50.0
    one["segments"][0]["frames"][0]["entities"]["mark"][1] += 1.0
    one["segments"][1]["source"]["ticks"].append(999)
    assert first == snapshot_first and second == snapshot_second


def test_composition_reuses_transfer_transform_with_yaw_and_scale():
    """yaw 与 scale 沿用 transfer_demonstration 的整体场景变换。"""
    first, second = seg_record(seg_a_history()), seg_record(seg_b_history())
    yaw, scale = math.pi / 2, 2.0
    got = compose_processes(first, second, roles={"actor": "hero", "ref": "mark"},
                            yaw=yaw, scale=scale)
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)

    for segment, record in zip(got["segments"], (first, second)):
        for frame, sample in zip(segment["frames"], record["samples"]):
            dx, dy, dz = (v * scale for v in sample["rel"])
            want = [dx * cos_y + dz * sin_y, dy, -dx * sin_y + dz * cos_y]
            assert frame_offset(frame, "hero", "mark") == pytest.approx(want, abs=1e-9)
    assert got["transform"]["yaw"] == yaw and got["transform"]["scale"] == scale
    assert got["joint"]["continuous"] is True, "整景变换不改衔接判据"
    assert got["relations"]["roles_rebound"] is True


def test_composition_preconditions_raise():
    """前置条件不满足时抛 ValueError，不做静默回退。"""
    first, second = seg_record(seg_a_history()), seg_record(seg_b_history())
    with pytest.raises(ValueError, match="roles 必须同时给出"):
        compose_processes(first, second, roles={"actor": "hero"})
    with pytest.raises(ValueError, match="tick_gap"):
        compose_processes(first, second, tick_gap=0)
    with pytest.raises(ValueError, match="second 记录没有采样"):
        compose_processes(first, {"kind": "relative_trajectory", "actor": "rider",
                                  "ref": "beacon", "ticks": [], "samples": []})
    with pytest.raises(ValueError, match="records 必须是已记录过程的序列"):
        complete_process(approach_history()[:6], "walker", "landmark",
                         {"kind": "relative_trajectory"})


def test_composition_reports_vertical_gap_keeps_horizontal_judgement():
    """joint 补高度读数：连续仍按水平 x/z 判定，竖直高度差单独上报。"""
    first = seg_record(seg_a_history())
    tall = seg_record(seg_b_history())
    for sample in tall["samples"]:
        sample["rel"][1] += 50.0
    got = compose_processes(first, tall, roles={"actor": "hero", "ref": "mark"},
                            translation=(10.0, 0.0, 5.0))
    joint = got["joint"]

    assert joint["actor_gap"] == pytest.approx(0.0, abs=1e-9)
    assert joint["ref_gap"] == pytest.approx(0.0, abs=1e-9)
    assert joint["actor_y_gap"] == pytest.approx(50.0, abs=1e-9)
    assert joint["ref_y_gap"] == pytest.approx(0.0, abs=1e-9)
    assert joint["continuous"] is True, "连续判据仍只按水平 x/z，不改三维判据"
    assert len(got["frames"]) == 10


# ================= C 组：定点变异（隔离副本） =================

def _mutation_scenario(module):
    """定点变异的判别场景：任一语义被破坏即 AssertionError。"""
    full = approach_history()
    record = extract_relative_trajectory(full, "walker", "landmark")
    th = module.ProcessOpsThresholds(min_hit_ratio=0.5)

    hit = module.complete_process(full[:6], "walker", "landmark", [record], thresholds=th)
    assert hit["match"] is not None, "同形前缀必须命中"
    assert hit["frames"][0]["tick"] == 7, "补全首帧 tick 必须接在前缀末 tick 之后"
    assert [f["tick"] for f in hit["frames"]] == [7, 8, 9, 10, 11, 12], \
        "补全轨迹 tick 必须逐 tick 递增"

    miss = module.complete_process(stray_history(), "walker", "landmark", [record],
                                   thresholds=th)
    assert miss["match"] is None and miss["frames"] == [], "形状不同的前缀不得命中"

    partial = extract_relative_trajectory(partial_history(), "walker", "landmark")
    ordered = module.complete_process(full[:6], "walker", "landmark", [partial, record],
                                      thresholds=th)
    scores = [c["match_score"] for c in ordered["candidates"]]
    assert [c["matched"] for c in ordered["candidates"]] == [True, True]
    assert scores == sorted(scores, reverse=True), "候选必须按匹配分数降序给出"
    assert ordered["candidates"][0]["source"]["index"] == 1, "候选必须按匹配分数降序给出"

    first = extract_relative_trajectory(seg_a_history(), "rider", "beacon")
    second = extract_relative_trajectory(seg_b_history(), "rider", "beacon")
    composed = module.compose_processes(first, second, roles={"actor": "hero", "ref": "mark"},
                                        translation=(10.0, 0.0, 5.0))
    assert all(set(f["entities"]) == {"hero", "mark"} for f in composed["frames"]), \
        "重组帧必须只带新绑定的角色名"
    assert composed["joint"]["ref_gap"] <= 1e-9, "后段 ref 必须锚定到前段末帧 ref"
    assert composed["joint"]["continuous"] is True, "偏移衔接的接点必须判为连续"

    broken = module.compose_processes(first,
                                      extract_relative_trajectory(seg_c_history(),
                                                                  "rider", "beacon"),
                                      roles={"actor": "hero", "ref": "mark"},
                                      translation=(10.0, 0.0, 5.0))
    assert broken["joint"]["continuous"] is False, "偏移差超容差的连接点必须判为不连续"

    # 目标语义 1：ref 移动时尾部逐帧还原记录的后续帧，平移量取已匹配末帧的偏差。
    moving = moving_ref_history()
    moving_record = extract_relative_trajectory(moving, "walker", "leader")
    restored = module.complete_process(moving[:6], "walker", "leader", [moving_record])
    assert restored["match"] is not None, "移动 ref 的同形前缀必须命中"
    for frame, observed in zip(restored["frames"], moving[6:]):
        assert frame["entities"]["leader"] == pytest.approx(
            observed["entities"]["leader"]), \
            "移动 ref 的补全尾部必须逐帧还原记录的后续帧"
    assert restored["match"]["translation"] == pytest.approx([0.0, 0.0, 0.0])

    # 目标语义 2：采样节奏不同的记录不得命中。
    sparse = sparse_from_record(moving_record, [1, 5, 9, 13, 17, 21])
    offbeat = module.complete_process(sparse, "walker", "leader", [moving_record])
    assert offbeat["match"] is None, "elapsed tick 不同的记录不得命中"
    assert offbeat["candidates"][0]["timing"]["matched"] is False, \
        "elapsed tick 不同的候选必须报 timing 不匹配"

    # 目标语义 3：同名角色必须被拒绝。
    try:
        module.compose_processes(first, second, roles={"actor": "hero", "ref": "hero"})
    except ValueError:
        pass
    else:
        raise AssertionError("同名角色的重组必须被拒绝")


_MUTATIONS = [
    ("M1", "补全：匹配分数下限失效（候选一律命中）",
     "    elif hits / compared < th.min_hit_ratio:\n",
     "    elif False:  # 定点变异 M1\n",
     "形状不同的前缀不得命中"),
    ("M2", "补全：接续 tick 不加该采样的 dt",
     '    tick_offset = (last_prefix["tick"] + step) - int(tail[0]["tick"])\n',
     '    tick_offset = last_prefix["tick"] - int(tail[0]["tick"])  # 定点变异 M2\n',
     "补全首帧 tick 必须接在前缀末 tick 之后"),
    ("M3", "重组：连接点判据恒真",
     "    continuous = actor_gap <= th.joint_tol\n",
     "    continuous = True  # 定点变异 M3\n",
     "偏移差超容差的连接点必须判为不连续"),
    ("M4", "重组：后段不锚定前段末帧 ref",
     "    second_shift = [float(joint_ref[k]) - second_ref0[k] for k in range(3)]\n",
     "    second_shift = [0.0, 0.0, 0.0]  # 定点变异 M4\n",
     "后段 ref 必须锚定到前段末帧 ref"),
    ("M5", "重组：角色重绑定失效（沿用前段原角色名）",
     '        new_actor, new_ref = roles["actor"], roles["ref"]\n',
     '        new_actor, new_ref = first["actor"], first["ref"]  # 定点变异 M5\n',
     "重组帧必须只带新绑定的角色名"),
    ("M6", "补全：候选排序不再按匹配分数降序",
     '            -cand["match_score"],\n',
     '            cand["match_score"],  # 定点变异 M6\n',
     "候选必须按匹配分数降序给出"),
    ("M7", "补全：尾部平移退回“首条未观测采样放到观测末位置”",
     '    translation = [float(anchor[k]) - float(matched_ref[k]) for k in range(3)]\n',
     '    translation = [float(anchor[k]) - float(tail[0]["ref"][k])'
     ' for k in range(3)]  # 定点变异 M7\n',
     "移动 ref 的补全尾部必须逐帧还原记录的后续帧"),
    ("M8", "补全：命中帧不再要求 elapsed tick 相同",
     "    hits = sum(1 for i in range(compared) if errors[i] <= th.match_tol"
     " and timing_hits[i])\n",
     "    hits = sum(1 for i in range(compared) if errors[i] <= th.match_tol)"
     "  # 定点变异 M8\n",
     "elapsed tick 不同的记录不得命中"),
    ("M9", "重组：同名角色校验失效",
     '    if new_actor == new_ref:\n'
     '        raise ValueError("actor 与 ref 不得同名：帧里的 entities 会互相覆写")\n',
     '    if False:  # 定点变异 M9\n'
     '        raise ValueError("actor 与 ref 不得同名：帧里的 entities 会互相覆写")\n',
     "同名角色的重组必须被拒绝"),
]


def _load_mutant(source, tmp_path, index, *, disable_transfer_role_guard=False):
    """把变异源码放进临时包（连带 world_process 副本）导入；工作树不被改写。"""
    name = "process_ops_mutant_pkg_%d" % index
    root = tmp_path / ("mutant_%d" % index)
    package = root / name
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    with open(os.path.join(REPO, "lingshu", "world", "world_process.py"),
              encoding="utf-8") as handle:
        transfer_source = handle.read()
    if disable_transfer_role_guard:
        # M9 必须撤掉调用链上两处同名校验。否则 transfer 新增的入口闸会
        # 兜住 compose 的变异，让“去掉同名保护”仍被 ValueError 拒绝。
        # 仅改隔离副本；其它变异沿用完整的真实 transfer 实现。
        gate = ('    if actor == ref:\n'
                '        raise ValueError("actor 与 ref 不得同名：帧里的 entities 会互相覆写")\n')
        assert transfer_source.count(gate) == 1
        transfer_source = transfer_source.replace(gate, '', 1)
    (package / "world_process.py").write_text(transfer_source, encoding="utf-8")
    (package / "process_ops.py").write_text(source, encoding="utf-8")
    sys.path.insert(0, str(root))
    try:
        importlib.invalidate_caches()
        return importlib.import_module(name + ".process_ops")
    finally:
        sys.path.remove(str(root))


def _digest(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def test_c_named_mutations_turn_isolation_red_without_touching_working_tree(tmp_path):
    """M1 至 M9 各自在隔离副本里让目标断言变红；工作树源码不被改写。"""
    import lingshu.world.process_ops as module

    before = _digest(MODULE_PATH)
    with open(MODULE_PATH, encoding="utf-8") as handle:
        source = handle.read()
    _mutation_scenario(module)

    for index, (name, title, old, new, expected) in enumerate(_MUTATIONS):
        assert source.count(old) == 1, "变异点 %s 在源码里不唯一：%r" % (name, old)
        mutant = _load_mutant(source.replace(old, new), tmp_path, index,
                              disable_transfer_role_guard=(name == "M9"))
        with pytest.raises(AssertionError, match=expected):
            _mutation_scenario(mutant)

    assert _digest(MODULE_PATH) == before, "定点变异不得改动工作树源码"


# ================= D 组：边界 =================

def test_module_does_not_call_simulator_or_write_actual():
    """边界：模块不调用 perceive/generate/execute，也不写 actual。"""
    with open(MODULE_PATH, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    banned = {"perceive", "generate", "execute"}
    assert not (names | attrs) & banned, "过程补全与重组模块不得调用模拟器入口"
    strings = {node.value for node in ast.walk(tree)
               if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    assert "actual" not in strings, "模块不得写 actual"
