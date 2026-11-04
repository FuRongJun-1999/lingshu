"""观测历史中的运动组合体：观察窗口内共同平移的实体集合。

输入
----
history 取统一世界模型的真实观测记录，每条形如
``{"tick": int, "entities": {eid: [x, y, z]}}``。实体键是 entities。
位移判据只看水平 x/z。y 不参与判据，组模型存成员相对锚点的已观测高度差，
预测按该高度差给出未观测成员的高度。

入口
----
detect_motion_groups(history, ...)      纯函数，从观测历史检测运动组合体。
get_motion_group(model, eid)            取含 eid 的组模型，取不到返回 None。
predict_group_members(group, observed)  用组模型和已观测锚点预测未观测成员。
predict_with_groups(history, ...)       检测后一次给出预测，供 core 调用。

范围
----
首版只支持共同平移：成员两两相对位置在窗口内稳定，且窗口内确有位移。
“窗口内相对位置稳定”的判据是：以两名成员首个共同观测 tick 的相对偏移为基准，
其后任一共同观测 tick 的相对偏移偏离基准不超过 tol_pos。
累计缓慢分离与来回漂移都会超界，只看首尾净值的判据会漏掉这两种情形，本版不用。
组模型描述运动关系，不认定成员之间存在机械连接。
同地点静止的实体集合没有位移证据，不判成共同运动。
竖直方向的共同运动不判据、不建模。
锚点取成员里字典序最小者，成员偏移相对锚点给出。
成员集合按 eid 字典序首次适配合并：成员之间两两兼容时分组结果与改名无关；
兼容关系非传递时，首次适配按字典序选中的极大团可能随改名改变，本版不承诺改名不变。

预测
----
预测值取已观测锚点的当前位置加组偏移，来源记为 motion_group，
actual 恒为 False，不写回观测历史。高度取已观测锚点的 y 加组内相对高度差，
不把成员拉平到锚点高度。组内没有可见成员时不给坐标，只在 unresolved 里给原因。
组速度只描述窗口内的共同运动，不用于外推。
本帧多个可见成员的相对位置若违背组内偏移，则缺失成员记为 unresolved，
不任选一个锚点给出相互矛盾的坐标；判据沿用组模型的 tol_pos。
"""
from __future__ import annotations

import hashlib
import math


def _records(history, window: int) -> list:
    """取窗口内最近 window 条记录，并校验形状。公开入口各调用一次。"""
    if not isinstance(history, (list, tuple)):
        raise ValueError("history 必须是观测记录序列")
    if isinstance(window, bool) or not isinstance(window, int) or window < 2:
        raise ValueError("window 必须是不小于 2 的整数")
    records = list(history[max(0, len(history) - window):])
    for _record in records:
        if not isinstance(_record, dict) or "tick" not in _record or "entities" not in _record:
            raise ValueError("history 记录必须含 tick 与 entities")
        if isinstance(_record["tick"], bool) or not isinstance(_record["tick"], int):
            raise ValueError("history 记录的 tick 必须是整数")
        entities = _record["entities"]
        if not isinstance(entities, dict):
            raise ValueError("history 记录的 entities 必须是 eid 到位置的映射")
        for eid, pos in entities.items():
            if not isinstance(eid, str) or not isinstance(pos, (list, tuple)) or len(pos) != 3:
                raise ValueError("entities 的成员必须是 eid 到三元位置的映射")
    return records


def _frames(records) -> dict:
    """把记录展成 tick 到 {eid: (x, y, z)} 的观测帧。只收真实出现的观测。"""
    frames = {}
    for _record in records:
        frame = frames.setdefault(_record["tick"], {})
        for eid, pos in _record["entities"].items():
            frame[eid] = (float(pos[0]), float(pos[1]), float(pos[2]))
    return frames


def _group_id(members: list) -> str:
    """组标识只由成员集合决定，与观测顺序、窗口无关。"""
    digest = hashlib.sha1("|".join(members).encode("utf-8")).hexdigest()
    return "mg_" + digest[:12]


def _pair_evidence(frames, ticks, a, b, tol_pos, min_samples, min_speed):
    """实体 a、b 的共同平移证据；不满足判据时返回 None。

    样本取 a、b 都被观测到的真实 tick，dt 取相邻两个样本的 tick 差。
    缺帧不补样本：成员缺席的区间整段跳过，所以 dt 是实际采样间隔。
    判据只有一条：两者相对偏移相对首个共同观测的偏离不超过 tol_pos。
    任一共同观测 tick 违反即返回 None。
    """
    base = None
    prev_a = None
    prev_tick = None
    samples = 0
    moving = 0
    for tick in ticks:
        frame = frames[tick]
        pos_a = frame.get(a)
        pos_b = frame.get(b)
        if pos_a is None or pos_b is None:
            continue
        offset = (pos_b[0] - pos_a[0], pos_b[2] - pos_a[2])
        if base is not None:
            drift = math.hypot(offset[0] - base[0], offset[1] - base[1])
            if drift > tol_pos:
                return None
            if math.hypot(pos_a[0] - prev_a[0], pos_a[2] - prev_a[2]) / (tick - prev_tick) > min_speed:
                moving += 1
            samples += 1
        else:
            base = offset
        prev_tick = tick
        prev_a = pos_a
    if samples < min_samples or moving == 0:
        return None
    return True


def _clusters(eids, evidence) -> list:
    """按 eid 字典序首次适配合并，只保留两两有证据的成员集合。

    首次适配是贪心极大团，合并顺序由 eid 字典序固定。
    成员之间两两兼容时，结果与改名、插入序无关；
    兼容关系非传递时，字典序不同会选到不同极大团，结果可能随改名改变。
    """
    clusters = []
    for eid in eids:
        for cluster in clusters:
            keys = [(min(eid, other), max(eid, other)) for other in cluster]
            if all(key in evidence for key in keys):
                cluster.append(eid)
                break
        else:
            clusters.append([eid])
    return [cluster for cluster in clusters if len(cluster) >= 2]


def _fit_group(frames, ticks, members, tol_pos, min_samples, min_speed):
    """按成员的共同观测 tick 拟合组模型；共同采样不足时返回 None。

    speed 定义为净位移除以总 dt：锚点在首个与末个共同观测 tick 之间的位移，
    除以两者的 tick 差。采样间隔不均时，区间速度的算术均值会偏向 dt 小的区间，
    本版不取算术均值。判据在 _pair_evidence 里施加，以两名成员各自的首个共同观测为基准；
    evidence 里的 max_offset_drift 是读数，以组内首个共同观测 tick 为基准。
    """
    anchor = members[0]
    common = [tick for tick in ticks if all(eid in frames[tick] for eid in members)]
    if len(common) < 2:
        return None
    first, last = common[0], common[-1]
    step_x = frames[last][anchor][0] - frames[first][anchor][0]
    step_z = frames[last][anchor][2] - frames[first][anchor][2]
    moving = 0
    max_step_drift = 0.0
    for prev_tick, tick in zip(common, common[1:]):
        prev_frame = frames[prev_tick]
        frame = frames[tick]
        step_ax = frame[anchor][0] - prev_frame[anchor][0]
        step_az = frame[anchor][2] - prev_frame[anchor][2]
        if math.hypot(step_ax, step_az) / (tick - prev_tick) > min_speed:
            moving += 1
        for eid in members:
            drift = math.hypot((frame[eid][0] - prev_frame[eid][0]) - step_ax,
                               (frame[eid][2] - prev_frame[eid][2]) - step_az)
            if drift > max_step_drift:
                max_step_drift = drift
    samples = len(common) - 1
    if samples < min_samples or moving == 0:
        return None
    offsets = {}
    y_offsets = {}
    max_offset_drift = 0.0
    for eid in members[1:]:
        first_chord = (frames[first][eid][0] - frames[first][anchor][0],
                       frames[first][eid][2] - frames[first][anchor][2])
        for tick in common:
            frame = frames[tick]
            drift = math.hypot((frame[eid][0] - frame[anchor][0]) - first_chord[0],
                               (frame[eid][2] - frame[anchor][2]) - first_chord[1])
            if drift > max_offset_drift:
                max_offset_drift = drift
        offsets[eid] = [round(frames[last][eid][0] - frames[last][anchor][0], 6),
                        round(frames[last][eid][2] - frames[last][anchor][2], 6)]
        y_offsets[eid] = round(frames[last][eid][1] - frames[last][anchor][1], 6)
    elapsed = last - first
    vx = step_x / elapsed
    vz = step_z / elapsed
    return {
        "group_id": _group_id(members),
        "kind": "common_translation",
        "mechanical_link": False,
        "members": list(members),
        "anchor": anchor,
        "offsets": offsets,
        "y_offsets": y_offsets,
        "speed": [round(vx, 6), round(vz, 6)],
        "speed_magnitude": round(math.hypot(vx, vz), 6),
        "displacement": [round(step_x, 6), round(step_z, 6)],
        "window": {"start_tick": first, "end_tick": last,
                   "frames": len(common), "ticks": list(common)},
        "samples": samples,
        "evidence": {"max_step_drift": round(max_step_drift, 6),
                     "max_offset_drift": round(max_offset_drift, 6),
                     "dts": [float(tick - prev) for prev, tick in zip(common, common[1:])],
                     "moving_samples": moving, "tol_pos": tol_pos,
                     "min_samples": min_samples},
    }


def _detect(records, window, tol_pos, min_samples, min_speed) -> dict:
    """检测主体：帧 → 成对证据 → 首次适配分组 → 组拟合。"""
    frames = _frames(records)
    ticks = sorted(frames)
    seen = {}
    for tick in ticks:
        for eid in frames[tick]:
            seen[eid] = seen.get(eid, 0) + 1
    eids = sorted(eid for eid, count in seen.items() if count >= 2)
    evidence = {}
    for index, first in enumerate(eids):
        for second in eids[index + 1:]:
            if _pair_evidence(frames, ticks, first, second,
                              tol_pos, min_samples, min_speed):
                evidence[(first, second)] = True
    groups = []
    rejected = []
    for cluster in _clusters(eids, evidence):
        group = _fit_group(frames, ticks, cluster, tol_pos, min_samples, min_speed)
        if group is None:
            rejected.append({"members": cluster, "reason": "insufficient_common_samples"})
        else:
            groups.append(group)
    groups.sort(key=lambda item: item["group_id"])
    return {"kind": "motion_groups", "window": window,
            "ticks": [ticks[0], ticks[-1]] if ticks else [],
            "groups": groups, "rejected": rejected}


def detect_motion_groups(history, window: int = 8, tol_pos: float = 0.05,
                         min_samples: int = 2, min_speed: float = 1e-6) -> dict:
    """从观测历史检测运动组合体，返回可直接存 JSON 的结果。

    history     统一世界模型的真实观测 history；不改动入参。
    window      检测窗口，取最近 window 条记录。
    tol_pos     窗口内相对位置稳定容差：相对首个共同观测偏移的最大误差上限。
    min_samples 判成共同平移所需的最少真实采样区间数。
    min_speed   记为“有位移”的速度下限，位置单位每 tick。

    返回 {"kind", "window", "ticks", "groups", "rejected"}；组字段见
    _fit_group，rejected 记下成员两两通过但共同采样不足的集合。
    """
    records = _records(history, window)
    return _detect(records, window, tol_pos, min_samples, min_speed)


def get_motion_group(model: dict, eid: str):
    """取含 eid 的组模型；该实体不在任何组内时返回 None。"""
    if not isinstance(model, dict):
        raise ValueError("model 必须是 detect_motion_groups 的返回结果")
    for group in model.get("groups", ()):
        if eid in group["members"]:
            return group
    return None


def _offset_between(group, base, reference, eid):
    """成员 eid 相对 reference 的 x/z 偏移与相对高度；组偏移以 base 为参照存放。"""
    def relative(member):
        if member == base:
            return (0.0, 0.0, 0.0)
        offset = group["offsets"][member]
        return (float(offset[0]), float(group["y_offsets"][member]), float(offset[1]))

    ref_offset = relative(reference)
    target_offset = relative(eid)
    return (target_offset[0] - ref_offset[0],
            target_offset[1] - ref_offset[1],
            target_offset[2] - ref_offset[2])


def predict_group_members(group: dict, observed: dict, tick=None) -> dict:
    """用组模型和本帧已观测成员，预测组内未观测成员的落点。

    group     detect_motion_groups 给出的组模型。
    observed  本帧真实观测 {"eid": [x, y, z]}；只把这里出现的成员算可见。
    tick      本帧 tick；用于标注预测时刻与组窗口末 tick 的间隔。

    x/z 取已观测锚点的当前位置加组内偏移；y 取已观测锚点的 y 加组内相对高度差，
    所以异高成员不会被拉平到锚点高度。
    返回 {"tick", "group_id", "predictions", "unresolved"}。
    predictions 每项含 source="motion_group" 与 actual=False。
    多个可见成员先两两核对水平偏移；矛盾时不给缺失成员坐标。
    """
    if not isinstance(group, dict) or "members" not in group or "offsets" not in group:
        raise ValueError("group 必须是 detect_motion_groups 给出的组模型")
    if not isinstance(observed, dict):
        raise ValueError("observed 必须是 eid 到三元位置的映射")
    for pos in observed.values():
        if not isinstance(pos, (list, tuple)) or len(pos) != 3:
            raise ValueError("observed 的成员必须是 eid 到三元位置的映射")
    base = group["anchor"]
    members = list(group["members"])
    visible = [eid for eid in members if eid in observed]
    missing = [eid for eid in members if eid not in observed]
    predictions = {}
    unresolved = {}
    if not visible:
        for eid in missing:
            unresolved[eid] = {"reason": "no_observed_anchor",
                               "group_id": group["group_id"]}
        return {"tick": tick, "group_id": group["group_id"],
                "predictions": predictions, "unresolved": unresolved}
    tol_pos = float(group["evidence"]["tol_pos"])
    # _fit_group 的偏移按六位小数保存；两成员各一份偏移相减，每轴至多
    # 1e-6 的舍入差，二维范数至多 sqrt(2)*1e-6。此预算只补偿序列化精度。
    limit = tol_pos + math.sqrt(2.0) * 1e-6
    for index, first in enumerate(visible):
        for second in visible[index + 1:]:
            expected = _offset_between(group, base, first, second)
            actual = (float(observed[second][0]) - float(observed[first][0]),
                      float(observed[second][2]) - float(observed[first][2]))
            drift = math.hypot(actual[0] - expected[0], actual[1] - expected[2])
            if drift > limit:
                for eid in missing:
                    unresolved[eid] = {
                        "reason": "conflicting_observed_anchors",
                        "group_id": group["group_id"],
                        "observed_pair": [first, second],
                        "offset_error": round(drift, 6), "tol_pos": tol_pos,
                    }
                return {"tick": tick, "group_id": group["group_id"],
                        "predictions": predictions, "unresolved": unresolved}
    anchor = base if base in observed else visible[0]
    anchor_pos = [float(value) for value in observed[anchor]]
    offset_tick = group["window"]["end_tick"]
    gap = tick - offset_tick if isinstance(tick, int) and not isinstance(tick, bool) else None
    for eid in missing:
        offset = _offset_between(group, base, anchor, eid)
        predictions[eid] = {
            "pos": [round(anchor_pos[0] + offset[0], 6),
                    round(anchor_pos[1] + offset[1], 6),
                    round(anchor_pos[2] + offset[2], 6)],
            "source": "motion_group",
            "actual": False,
            "group_id": group["group_id"],
            "anchor": anchor,
            "offset": [round(offset[0], 6), round(offset[2], 6)],
            "y_offset": round(offset[1], 6),
            "offset_tick": offset_tick,
            "gap": gap,
        }
    return {"tick": tick, "group_id": group["group_id"],
            "predictions": predictions, "unresolved": unresolved}


def predict_with_groups(history, window: int = 8, tol_pos: float = 0.05,
                        min_samples: int = 2, min_speed: float = 1e-6,
                        tick=None) -> dict:
    """检测运动组合体，并把最后一条观测记录当本帧给出预测。

    参数同 detect_motion_groups；tick 缺省取最后一条记录的 tick。
    返回 {"kind", "tick", "groups", "predictions", "unresolved"}。
    本帧已观测的成员不出现在 predictions 里，也不写回 history。
    """
    records = _records(history, window)
    model = _detect(records, window, tol_pos, min_samples, min_speed)
    observed = {}
    if records:
        last = records[-1]
        observed = {eid: [float(value) for value in pos]
                    for eid, pos in last["entities"].items()}
        if tick is None:
            tick = last["tick"]
    predictions = {}
    unresolved = {}
    for group in model["groups"]:
        result = predict_group_members(group, observed, tick)
        predictions.update(result["predictions"])
        unresolved.update(result["unresolved"])
    return {"kind": "motion_group_predictions", "tick": tick,
            "groups": model["groups"], "predictions": predictions,
            "unresolved": unresolved}
