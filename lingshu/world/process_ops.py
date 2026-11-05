# -*- coding: utf-8 -*-
"""process_ops · 过程补全与过程重组（世界过程第 2 组）
============================================================================
复用第 1 组的示教记录结构（lingshu/world/world_process.py 的
extract_relative_trajectory 返回值）与 transfer_demonstration 的变换。
本模块不新增记录格式、动作语言、调度框架与渲染模块。

过程补全（process completion）
    输入是观测前缀（world_model.perceive 记录的 history 形状）与已记录过程列表。
    判据逐帧比较前缀与记录的前 N 条采样，N = 前缀成对采样数：
      · 几何帧 = actor 相对 ref 的水平偏移差 ≤ match_tol；
      · 时序帧 = 相对首帧的 elapsed tick 相同（真实 dt，不比采样序号）；
      · 命中帧 = 同时满足几何帧与时序帧；
      · 匹配分数 match_score = 命中帧数 ÷ 对比帧数；
      · 命中要求 match_score ≥ min_hit_ratio，且对比帧数 ≥ min_match_samples，
        且记录还有 ≥ min_tail_samples 条采样未被前缀覆盖。
    首版不做重采样、时间缩放与插值：采样节奏不同的记录不能以 match_score 1.0 命中。
    候选逐条给出 geometry 与 timing 两组读数，dt 按记录原值上报，不换算。
    命中的候选按匹配分数降序、最大逐帧误差升序、记录下标升序取第一条。
    补全轨迹由 transfer_demonstration 生成：平移量 = 观测末帧 ref − 已匹配的记录末帧 ref，
    首帧 tick = 前缀末 tick + 该采样真实 dt。
    不命中返回 match=None 与空 frames，不抛异常。候选列表始终给出全部记录的来源与
    匹配分数。

过程重组（process composition）
    首版只连接 2 个已记录过程，不做多路串联。
    角色重绑定由调用方显式给出：roles={"actor": …eid…, "ref": …eid…}；省略时沿用
    前段的角色名。两段共用同一次 yaw 与 scale（transfer_demonstration 的整体场景
    变换），后段再按连接点平移。
    连接点取参照物：后段的 ref 锚定到前段末帧的 ref 位置。
      · joint.ref_gap   两段在连接点的 ref 水平距离（锚定残差）；
      · joint.actor_gap 前段末帧 actor 与后段首帧 actor 的水平距离；
      · joint.ref_y_gap / joint.actor_y_gap 两端原始的竖直高度差，只上报不参与判定；
      · joint.continuous = actor_gap ≤ joint_tol，只覆盖水平面。
    连接点不达标时仍生成新轨迹，判据与读数写进结果。
    relations 只报两段的过程类型、起止阶段与角色是否共用，不判断动作可执行性，
    也不判断因果兼容。记录没有过程标注时 same_type 为 None，表示未知，不猜测。

边界与不变量
    · 两个入口都是纯函数：不改写 history、records 与其嵌套列表，只返回新对象。
    · 不调用 perceive()/generate()/execute()，不读模拟器，不写 actual。
    · 匹配与连接判据只用水平 x/z；生成帧的 y 取记录原值，平移量按已匹配记录末帧给出。
    · 同一输入重复调用结果相同；候选与命中共分时按记录下标定序。
    · 前置条件不满足时抛 ValueError：records 不是序列、actor 与 ref 同名、roles 缺键、
      解析后 actor 与 ref 同名、tick_gap 小于 1、记录没有采样。不做静默回退。

运行（仓根）：
    python -X utf8 -m pytest tests/test_process_ops.py -q -p no:cacheprovider
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .world_process import GENERATED, pair_samples, transfer_demonstration


@dataclass(frozen=True)
class ProcessOpsThresholds:
    """过程补全与过程重组的判据阈值（长度单位 = 世界单位，只量水平面距离）。"""

    match_tol: float = 0.5        # 过程补全：逐帧偏移差容差
    min_hit_ratio: float = 1.0    # 过程补全：匹配分数下限（1.0 = 逐帧全在容差内）
    min_match_samples: int = 3    # 过程补全：前缀与记录的最少对比帧数
    min_tail_samples: int = 1     # 过程补全：未被前缀覆盖的最少后续采样数
    joint_tol: float = 0.5        # 过程重组：连接点 actor 衔接距离上限


# ================= 采样读数（只读，不返回原对象） =================

def _planar(p: Sequence[float], q: Sequence[float]) -> float:
    """两点的水平距离（入参是完整三维位置）；与第 1 组的判据口径一致。"""
    return math.hypot(float(p[0]) - float(q[0]), float(p[2]) - float(q[2]))


def _offset_gap(p: Sequence[float], q: Sequence[float]) -> float:
    """两条水平相对偏移 (x, z) 的距离。"""
    return math.hypot(float(p[0]) - float(q[0]), float(p[1]) - float(q[1]))


def _prefix_samples(history: Sequence[Dict], actor: str, ref: str) -> List[Dict]:
    """观测前缀的成对真实采样：tick、dt、水平相对偏移、两端位置。

    用 world_process.pair_samples 取采样：只有 actor 与 ref 同帧有观测才算一条。
    """
    out: List[Dict] = []
    for sample in pair_samples(history, actor, ref):
        rel = [float(sample["a"][k]) - float(sample["b"][k]) for k in range(3)]
        out.append({"tick": int(sample["tick"]),
                    "dt": int(sample["dt"]),
                    "offset": (rel[0], rel[2]),
                    "rel": rel,
                    "actor": [float(v) for v in sample["a"]],
                    "ref": [float(v) for v in sample["b"]]})
    return out


def _record_samples(record: Dict) -> List[Dict]:
    """已记录过程的采样读数：tick、dt、水平相对偏移、ref 位置。

    记录格式 = extract_relative_trajectory 的返回。缺 samples 键按空列表处理。
    """
    out: List[Dict] = []
    for sample in record.get("samples") or []:
        rel = [float(v) for v in sample["rel"]]
        out.append({"tick": int(sample["tick"]),
                    "dt": int(sample["dt"]),
                    "offset": (rel[0], rel[2]),
                    "rel": rel,
                    "ref": [float(v) for v in sample["ref"]]})
    return out


def _source_of(record: Dict, index: int, samples: int) -> Dict:
    """候选来源：记录在入参列表里的下标与记录自报身份。"""
    return {"index": index,
            "kind": record.get("kind"),
            "actor": record.get("actor"),
            "ref": record.get("ref"),
            "ticks": [int(v) for v in (record.get("ticks") or [])],
            "samples": samples}


# ================= 过程补全 =================

def _timing_readings(prefix: Sequence[Dict], rec: Sequence[Dict],
                     compared: int) -> Dict:
    """采样节奏读数：相对首帧的 elapsed tick 向量逐帧比较。

    elapsed = 本采样 tick − 首条采样 tick，即真实 dt 之和。只比相对量，因此记录 tick
    整体平移仍算同一节奏；不重采样、不缩放，也不改用采样序号。
    """
    deltas = [int(prefix[i]["tick"]) - int(prefix[0]["tick"])
              - (int(rec[i]["tick"]) - int(rec[0]["tick"]))
              for i in range(compared)]
    hit_frames = sum(1 for d in deltas if d == 0)
    return {"matched": hit_frames == compared,
            "hit_frames": hit_frames,
            "score": round(hit_frames / compared, 6),
            "max_abs_delta_ticks": max(abs(d) for d in deltas),
            "deltas": deltas}


def _candidate(record: Dict, index: int, prefix: Sequence[Dict],
               th: ProcessOpsThresholds) -> Dict:
    """一条记录的候选读数：来源、几何与时序两组读数、匹配分数与不命中原因。"""
    rec = _record_samples(record)
    compared = min(len(prefix), len(rec))
    cand = {"source": _source_of(record, index, len(rec)),
            "matched_samples": compared,
            "hit_frames": 0,
            "match_score": 0.0,
            "max_frame_error": None,
            "mean_frame_error": None,
            "tail_samples": max(0, len(rec) - compared),
            "geometry": {"hit_frames": 0, "score": 0.0},
            "timing": {"matched": False,
                       "hit_frames": 0,
                       "score": 0.0,
                       "max_abs_delta_ticks": None,
                       "deltas": []},
            "matched": False,
            "reason": ""}
    if compared == 0:
        cand["reason"] = "前缀与记录没有可比采样"
        return cand
    errors = [_offset_gap(prefix[i]["offset"], rec[i]["offset"]) for i in range(compared)]
    # #2159：非有限误差先过滤并显式上报丢弃数——修前同一条误差列表被两个聚合器
    # 给出矛盾读数：`max(errors)` 从左折叠、NaN 只在首位才传播（在其它位置被
    # `>` 比较比下去 ⇒ NaN 被静默丢弃返回 0.0），`sum(errors)` 必被任一 NaN
    # 传染成 NaN ⇒ max=0.0 宣称「全部在容差内」、mean=nan 宣称「算不出来」，
    # 两读数指相反方向（实测 max=0.0 / mean=nan / hit_frames=4 / matched=5）。
    # 口径与 `motion.py` 的读噪声处理同款：显式丢弃 + 上报。
    finite_errors = [e for e in errors if math.isfinite(e)]
    dropped = len(errors) - len(finite_errors)
    geometry_hits = sum(1 for e in finite_errors if e <= th.match_tol)
    timing = _timing_readings(prefix, rec, compared)
    timing_hits = [d == 0 for d in timing["deltas"]]
    hits = sum(1 for i in range(compared) if errors[i] <= th.match_tol and timing_hits[i])
    cand["geometry"] = {"hit_frames": geometry_hits,
                        "score": round(geometry_hits / compared, 6)}
    cand["timing"] = timing
    cand["hit_frames"] = hits
    cand["match_score"] = round(hits / compared, 6)
    cand["dropped_nonfinite"] = dropped
    if finite_errors:
        cand["max_frame_error"] = round(max(finite_errors), 6)
        cand["mean_frame_error"] = round(sum(finite_errors) / len(finite_errors), 6)
    else:
        cand["max_frame_error"] = None
        cand["mean_frame_error"] = None
    if compared < max(1, int(th.min_match_samples)):
        cand["reason"] = "对比帧数 %d 少于下限 %d" % (compared, th.min_match_samples)
    elif hits / compared < th.min_hit_ratio:
        if timing["matched"]:
            cand["reason"] = "匹配分数 %.3f 低于下限 %.3f" % (
                hits / compared, th.min_hit_ratio)
        elif geometry_hits == compared:
            cand["reason"] = ("采样节奏不匹配：elapsed tick 有 %d/%d 帧与记录不同"
                              % (compared - timing["hit_frames"], compared))
        else:
            cand["reason"] = ("匹配分数 %.3f 低于下限 %.3f；采样节奏有 %d/%d 帧不同"
                              % (hits / compared, th.min_hit_ratio,
                                 compared - timing["hit_frames"], compared))
    elif cand["tail_samples"] < int(th.min_tail_samples):
        cand["reason"] = "后续采样 %d 少于下限 %d" % (cand["tail_samples"],
                                                  th.min_tail_samples)
    else:
        cand["matched"] = True
    return cand


def _candidate_key(cand: Dict) -> Tuple:
    """候选排序键：命中优先、匹配分数降序、最大逐帧误差升序、记录下标升序。"""
    error = cand["max_frame_error"]
    return (0 if cand["matched"] else 1,
            -cand["match_score"],
            math.inf if error is None else error,
            cand["source"]["index"])


def _tail_frames(record: Dict, matched: int, last_prefix: Dict, actor: str,
                 ref: str) -> Tuple[List[Dict], Dict]:
    """用 transfer_demonstration 生成记录第 matched 条起的后续轨迹。

    matched 是已匹配的采样条数，最后一条已匹配采样是 samples[matched - 1]。
    平移量 = 观测末帧 ref − 已匹配记录末帧 ref：整段后续轨迹按两侧在已匹配末帧上的
    偏差平移。不得把首条未观测采样的 ref 放到观测末帧 ref 的位置，否则 ref 移动时尾部慢一帧。
    tick 接在前缀之后：首帧 tick = 前缀末 tick + 已匹配末帧到下一采样的真实 tick 差
    （差非正时取 1）。记录的采样逐条复制成新列表，入参记录不被改写。
    """
    samples = record["samples"]
    tail = samples[matched:]
    if not tail:
        raise ValueError("记录没有可用的后续采样，无法补全")
    anchor = last_prefix["ref"]
    matched_ref = [float(v) for v in samples[matched - 1]["ref"]]
    translation = [float(anchor[k]) - float(matched_ref[k]) for k in range(3)]
    step = int(tail[0]["tick"]) - int(samples[matched - 1]["tick"])
    if step <= 0:
        step = 1
    tick_offset = (last_prefix["tick"] + step) - int(tail[0]["tick"])
    tail_record = {"kind": record.get("kind", "relative_trajectory"),
                   "actor": record.get("actor"),
                   "ref": record.get("ref"),
                   "process": record.get("process"),
                   "ticks": [int(tail[0]["tick"]), int(tail[-1]["tick"])],
                   "samples": [{"tick": int(s["tick"]), "dt": int(s["dt"]),
                                "ref": [float(v) for v in s["ref"]],
                                "rel": [float(v) for v in s["rel"]]} for s in tail]}
    transfer = transfer_demonstration(tail_record, actor, ref,
                                      translation=translation, yaw=0.0, scale=1.0,
                                      tick_offset=tick_offset)
    return transfer["frames"], {"translation": translation,
                                "tick_offset": tick_offset,
                                "step": step,
                                "matched_record_index": matched - 1,
                                "matched_record_tick": int(samples[matched - 1]["tick"])}


def _thresholds_of(th: ProcessOpsThresholds) -> Dict:
    """判据读数的快照：调用方能在结果里核对用的是哪一档阈值。"""
    return {"match_tol": th.match_tol,
            "min_hit_ratio": th.min_hit_ratio,
            "min_match_samples": th.min_match_samples,
            "min_tail_samples": th.min_tail_samples,
            "joint_tol": th.joint_tol}


def complete_process(history: Sequence[Dict], actor: str, ref: str,
                     records: Sequence[Dict], *,
                     thresholds: Optional[ProcessOpsThresholds] = None) -> Dict:
    """过程补全：观测前缀匹配已记录过程，输出匹配过程的后续轨迹。

    history 是观测前缀（按 tick 递增），actor 与 ref 是前缀里的不同实体标识。
    records 是 extract_relative_trajectory 的记录序列；顺序只影响候选下标。
    返回：
        kind / source（生成帧来源标记）/ actor / ref / thresholds /
        prefix（前缀 tick 区间、成对采样数、末帧相对偏移与 ref 位置）/
        candidates（每条记录的来源、几何与时序读数与匹配分数，按 _candidate_key 排序）/
        match（命中候选、平移量、tick 偏移与已匹配记录末帧下标，未命中为 None）/
        frames（后续轨迹帧）/ ticks（补全轨迹 tick 区间）/
        seam（首帧与前缀末帧的衔接读数）。
    未命中时 match=None、frames=[]、ticks=[]、seam=None，不抛异常。
    actor 与 ref 同名时抛 ValueError：同名会让帧里的 entities 互相覆写。
    """
    th = thresholds or ProcessOpsThresholds()
    if actor == ref:
        raise ValueError("actor 与 ref 不得同名：帧里的 entities 会互相覆写")
    if records is None or isinstance(records, (dict, str, bytes)):
        raise ValueError("records 必须是已记录过程的序列")
    seq = list(records)
    prefix = _prefix_samples(history, actor, ref)
    candidates = [_candidate(record, index, prefix, th)
                  for index, record in enumerate(seq)]
    candidates.sort(key=_candidate_key)
    result = {"kind": "process_completion",
              "source": GENERATED,
              "actor": actor,
              "ref": ref,
              "thresholds": _thresholds_of(th),
              "prefix": {"ticks": [prefix[0]["tick"], prefix[-1]["tick"]] if prefix else [],
                         "samples": len(prefix),
                         "last_rel": list(prefix[-1]["rel"]) if prefix else None,
                         "last_ref": list(prefix[-1]["ref"]) if prefix else None},
              "candidates": candidates,
              "match": None,
              "frames": [],
              "ticks": [],
              "seam": None}
    hit = next((c for c in candidates if c["matched"]), None)
    if hit is None or not prefix:
        return result
    record = seq[hit["source"]["index"]]
    frames, place = _tail_frames(record, hit["matched_samples"], prefix[-1], actor, ref)
    match = dict(hit)
    match.update({"translation": place["translation"],
                  "tick_offset": place["tick_offset"],
                  "step": place["step"],
                  "matched_record_index": place["matched_record_index"],
                  "matched_record_tick": place["matched_record_tick"]})
    result["match"] = match
    result["frames"] = frames
    result["ticks"] = [frames[0]["tick"], frames[-1]["tick"]]
    result["seam"] = {"actor_gap": round(_planar(prefix[-1]["actor"],
                                                 frames[0]["entities"][actor]), 6),
                      "prefix_tick": prefix[-1]["tick"],
                      "first_tick": frames[0]["tick"],
                      "step": place["step"],
                      "ref_anchor": [round(float(v), 6) for v in prefix[-1]["ref"]],
                      "matched_record_index": place["matched_record_index"],
                      "matched_record_tick": place["matched_record_tick"]}
    return result


# ================= 过程重组 =================

def _process_facts(record: Dict) -> Dict:
    """记录里的过程标注（类型与首末阶段）；没有标注时各值为 None。"""
    process = record.get("process") or {}
    phases = process.get("phases") or []
    return {"type": process.get("type"),
            "start_phase": phases[0].get("phase") if phases else None,
            "end_phase": phases[-1].get("phase") if phases else None,
            "roles": dict(process.get("roles") or {})}


def _segments_frame(record: Dict, index: int) -> Dict:
    """一段记录的身份读数：来源、原角色与采样数。"""
    return {"source": _source_of(record, index, len(record.get("samples") or [])),
            "roles": {"actor": record.get("actor"), "ref": record.get("ref")},
            "process": _process_facts(record)}


def compose_processes(first: Dict, second: Dict, *,
                      roles: Optional[Dict[str, str]] = None,
                      translation: Sequence[float] = (0.0, 0.0, 0.0),
                      yaw: float = 0.0,
                      scale: float = 1.0,
                      tick_gap: int = 1,
                      thresholds: Optional[ProcessOpsThresholds] = None) -> Dict:
    """过程重组：连接 2 个已记录过程，重新绑定角色并检查连接点与过程关系。

    first 与 second 是 extract_relative_trajectory 的记录，首版只连接两段。
    roles 显式给出新角色名 {"actor": …, "ref": …}；省略时沿用前段的角色名。
    translation / yaw / scale 作用在前段（transfer_demonstration 的整体场景变换），
    后段共用同一 yaw 与 scale，再按连接点平移到前段末帧的 ref 位置。
    tick_gap 是连接点两侧的 tick 间隔（不小于 1）。
    返回：
        kind / source / actor / ref / transform / segments（两段身份、帧与变换）/
        joint（连接点 tick、ref 位置、ref_gap、actor_gap、ref_y_gap、actor_y_gap、
        两侧偏移、continuous）/ relations（两段过程类型、首末阶段与角色是否共用）/
        ticks / frames。
    连接点不连续时 frames 照常给出，joint.continuous 记 False。continuous 按水平 x/z
    判定；ref_y_gap 与 actor_y_gap 只上报原始高度差，不参与判定。
    记录没有采样、roles 缺键、解析后 actor 与 ref 同名、tick_gap 小于 1 时抛 ValueError。
    """
    th = thresholds or ProcessOpsThresholds()
    for label, record in (("first", first), ("second", second)):
        if not (record.get("samples") or []):
            raise ValueError("%s 记录没有采样，无法重组" % label)
    if roles is None:
        new_actor, new_ref = first["actor"], first["ref"]
    else:
        if "actor" not in roles or "ref" not in roles:
            raise ValueError("roles 必须同时给出 actor 与 ref")
        new_actor, new_ref = roles["actor"], roles["ref"]
    if new_actor == new_ref:
        raise ValueError("actor 与 ref 不得同名：帧里的 entities 会互相覆写")
    if isinstance(tick_gap, bool) or int(tick_gap) < 1:
        raise ValueError("tick_gap 必须是不小于 1 的整数")
    gap = int(tick_gap)

    head = transfer_demonstration(first, new_actor, new_ref,
                                  translation=translation, yaw=yaw, scale=scale,
                                  tick_offset=0)
    head_frames = head["frames"]
    joint_ref = head_frames[-1]["entities"][new_ref]
    second_ref0 = [float(v) for v in second["samples"][0]["ref"]]
    second_shift = [float(joint_ref[k]) - second_ref0[k] for k in range(3)]
    second_offset = (head_frames[-1]["tick"] + gap) - int(second["samples"][0]["tick"])
    tail = transfer_demonstration(second, new_actor, new_ref,
                                  translation=second_shift, yaw=yaw, scale=scale,
                                  tick_offset=second_offset)
    tail_frames = tail["frames"]
    # #2158：**顶层合并帧**的跨段首帧 `dt` 修前继承源记录首帧的 dt（源首帧恒 0
    # ——它是自己那段的时间轴起点）⇒ 「跨段 tick 差 = gap（≥1）而该帧 dt = 0」
    # 直接矛盾（tick 与 dt 是同一份时间轴的两套记法，tick 被 tick_offset 平移、
    # dt 没跟着对齐）。连接点两侧的真实间隔正是 tick_gap ⇒ 顶层帧改写为 gap。
    # 段级 `segments[*].frames` 保留源 dt 不动——那是各段自身时间轴的证据面
    # （test_process_ops 的段级 `frame["dt"] == sample["dt"]` 契约不变），
    # 故合并时浅拷贝 tail 帧、只改顶层副本。
    frames = head_frames + [dict(f) for f in tail_frames]
    if tail_frames:
        frames[len(head_frames)]["dt"] = gap

    # 连接点读数：ref 锚定残差与 actor 衔接距离（含两侧相对偏移）。
    first_end_ref = head_frames[-1]["entities"][new_ref]
    first_end_actor = head_frames[-1]["entities"][new_actor]
    second_start_ref = tail_frames[0]["entities"][new_ref]
    second_start_actor = tail_frames[0]["entities"][new_actor]
    ref_gap = _planar(first_end_ref, second_start_ref)
    actor_gap = _planar(first_end_actor, second_start_actor)
    ref_y_gap = abs(float(first_end_ref[1]) - float(second_start_ref[1]))
    actor_y_gap = abs(float(first_end_actor[1]) - float(second_start_actor[1]))
    continuous = actor_gap <= th.joint_tol
    joint = {"on": "ref",
             "tick": head_frames[-1]["tick"],
             "ref": [round(float(v), 6) for v in first_end_ref],
             "ref_gap": round(ref_gap, 6),
             "actor_gap": round(actor_gap, 6),
             "ref_y_gap": round(ref_y_gap, 6),
             "actor_y_gap": round(actor_y_gap, 6),
             "offset_end_first": [round(float(first_end_actor[k] - first_end_ref[k]), 6)
                                  for k in (0, 2)],
             "offset_start_second": [round(float(second_start_actor[k] - second_start_ref[k]), 6)
                                     for k in (0, 2)],
             "joint_tol": th.joint_tol,
             "continuous": continuous}

    facts_first, facts_second = _process_facts(first), _process_facts(second)
    if facts_first["type"] is None or facts_second["type"] is None:
        same_type: Optional[bool] = None
    else:
        same_type = facts_first["type"] == facts_second["type"]
    relations = {"actor": new_actor,
                 "ref": new_ref,
                 "roles_rebound": bool(new_actor != first.get("actor")
                                       or new_ref != first.get("ref")),
                 "shared_actor": first.get("actor") == second.get("actor"),
                 "shared_ref": first.get("ref") == second.get("ref"),
                 "same_type": same_type,
                 "first": facts_first,
                 "second": facts_second}

    segments = [dict(_segments_frame(first, 0),
                     frames=head_frames, ticks=head["ticks"], transform=head["transform"]),
                dict(_segments_frame(second, 1),
                     frames=tail_frames, ticks=tail["ticks"], transform=tail["transform"])]
    return {"kind": "process_composition",
            "source": GENERATED,
            "actor": new_actor,
            "ref": new_ref,
            "thresholds": _thresholds_of(th),
            "transform": {"translation": [float(v) for v in translation],
                          "yaw": float(yaw),
                          "scale": float(scale),
                          "tick_gap": gap},
            "segments": segments,
            "joint": joint,
            "relations": relations,
            "ticks": [frames[0]["tick"], frames[-1]["tick"]],
            "frames": frames}
