# -*- coding: utf-8 -*-
"""world_process · 时间会意、示教迁移与事件回声（世界过程第 1、2 组）
============================================================================
输入合同：UnifiedWorldModel 的真实观测 history——
    [{"tick": int, "entities": {eid: [x, y, z]}}, ...]
（world_model.py perceive() 第 243-246 行追加的形状）。本模块只读该列表。

时间会意（过程识别）从实体间关系变化识别 3 类过程：
  · approach_stay_leave 接近→停留→离开
  · detour_return       绕行后返回
  · meet_escort         汇合后同行
判据只用位置：一对实体的水平距离序列、各自水平路程、相对偏移、actor 绕 ref 的方位角、
actor 自身绝对轨迹的闭合回路形状（回路等效半径），以及相对轨迹到 ref 的最近距离
（绕行不能把参照物当路点穿过）。
detour_return 由三项几何证据共同判定：actor 绝对轨迹的闭合回路、actor 绕 ref 的方位角
扫描、相对轨迹不穿越参照物。
绝对回路判据不依赖 ref 是否移动：参照物移动只改变相对形状，不改 actor 自身轨迹的回路。
本模块不读 category、不读 behavior_inferred、不读模拟器行为日志。

示教迁移把示教期的相对轨迹存成 JSON 记录，再绑定新 actor/ref，
支持平移、水平旋转（绕 +y 轴）与显式尺度 3 种变换，输出带 tick 的帧轨迹，
每帧标 source="generated"。yaw 与 scale 是整体场景变换：以记录首帧 ref 位置为中心，
ref 位移与 rel 一起缩放、旋转；ref 静止时与只变换 rel 等价。

边界与不变量：
  · 本模块不调用 perceive()/generate()/execute()，不读模拟器，不写 actual。
  · 缺失帧不插值：只对有观测的 tick 计算，dt 取相邻真实采样的 tick 差。
  · y 轴按原值保留：识别只用水平 x/z；迁移对 ref 位移与相对偏移的 y 分量同样缩放与平移。
  · 同一对实体在观察窗口内至多报 1 条过程，判据优先级见 PROCESS_TYPES。
  · 未识别到过程时返回空列表，不猜测。
  · 回路等效半径的周长含末点回到首点的闭合边；12 个既有场景下去掉该边不改变判定。
  · 已声明限制：路径贴近参照物（0.05 以上）经过、而回路实际在远处的对抗构造可能误报。
    穿越容差 0.05 是绝对量，不随参照物尺寸变化。
  · 已声明限制（漏报方向）：参照物在窗口内去回振幅超过 actor 回路半径约 18 倍时，
    合法绕行可能漏报（复审实测翻转点：ref 去回振幅 55.1、actor 回路半径 2.97）。

事件回声从 motion_events 的停止/转向事件学习延迟关联：在 1..max_lag tick 的
有限延迟范围内按固定签名对 (source_eid, source_event, target_eid, target_event)
统计 lag，取次数最多、次数相同则 lag 最小的一条。留出段条数取
ceil(事件数 × holdout_fraction)，从流尾切出，同一 tick 的事件整组进同一段。
留出段上比较延迟模型（预测 tick = 当前 tick + lag）与无延迟模型（预测 tick =
当前 tick），两者的触发集合与目标签名相同。命中要求目标 eid、event、tick 三者
与固定目标签名在源事件之后最早的一条响应全等。关联只表示事件在时间上的先后与
间隔，不表示因果作用。

运行（仓根）：
    python -X utf8 -m pytest tests/test_world_process.py tests/test_event_echo.py -q
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

PROCESS_APPROACH_STAY_LEAVE = "approach_stay_leave"
PROCESS_DETOUR_RETURN = "detour_return"
PROCESS_MEET_ESCORT = "meet_escort"

# 元组顺序 = 判据优先级：绕行判据最严，先判；同一对实体只报第 1 条命中的过程。
PROCESS_TYPES: Tuple[str, ...] = (PROCESS_DETOUR_RETURN,
                                  PROCESS_APPROACH_STAY_LEAVE,
                                  PROCESS_MEET_ESCORT)

GENERATED = "generated"          # 生成帧的来源标记；观测帧没有该键

PHASE_APPROACH = "approach"
PHASE_STAY = "stay"
PHASE_LEAVE = "leave"
PHASE_DETOUR = "detour"
PHASE_RETURN = "return"
PHASE_CONVERGE = "converge"
PHASE_ESCORT = "escort"


@dataclass(frozen=True)
class ProcessThresholds:
    """过程识别判据的阈值（长度单位 = 世界单位，只量水平面距离）。"""

    min_samples: int = 4          # 一对实体的成对真实采样数下限
    approach_drop: float = 1.0    # 接近阶段：距离相对起点的净缩短下限
    leave_rise: float = 1.0       # 离开阶段：距离相对最近点的净拉长下限
    stay_band: float = 0.5        # 停留阶段：距离偏离最近点的波动上限
    stay_move: float = 0.5        # 停留阶段：actor 相对停留窗口中心样本的水平偏离上限
    hold_min_samples: int = 3     # 持续阶段（停留/同行）的最少真实采样数
    hold_min_ticks: int = 3       # 持续阶段（停留/同行）的最少覆盖 tick 数
    detour_sweep_deg: float = 90.0   # 绕行阶段：actor 绕 ref 的方位角总变化下限（度）
    detour_loop_radius: float = 0.5  # 绕行回路等效半径下限：2 × 回路面积 ÷ 折线周长
    detour_crossing_tol: float = 0.05  # 穿越参照物的近距容差：actor 相对轨迹到 ref 的
                                       # 最近距离须大于该值；真穿越的读数为 0，
                                       # 容差只覆盖浮点与采样表述误差（语义是穿越，
                                       # 不是最小分离）
    detour_return_tol: float = 0.5   # 返回容差：距离与 actor 自身位置
    detour_path_min: float = 3.0  # 绕行的水平路程下限
    meet_gap: float = 1.5         # 汇合判据：距离降到该值以下算汇合
    escort_offset_tol: float = 0.5   # 同行阶段：相对偏移偏离汇合时偏移的上限
    escort_move: float = 0.5      # 同行阶段：双方各自的最小水平位移
    stop_speed: float = 0.2       # 事件提取：水平速度不超过该值记为停止（世界单位每 tick）
    turn_deg: float = 45.0        # 事件提取：方向变化不小于该角度记为转向


# ================= 采样：只取真实观测 =================

def _planar(pos: Sequence[float]) -> Tuple[float, float]:
    """观测位置 → 水平坐标 (x, z)。识别不用 y 轴。"""
    return float(pos[0]), float(pos[2])


def _planar_distance(p: Sequence[float], q: Sequence[float]) -> float:
    """两点的水平距离；也用于相邻采样的水平位移。"""
    return math.hypot(float(p[0]) - float(q[0]), float(p[2]) - float(q[2]))


def _path_length(positions: Sequence[Sequence[float]]) -> float:
    """相邻真实采样之间的水平路程之和。"""
    return sum(_planar_distance(p, q) for p, q in zip(positions, positions[1:]))


def track_samples(history: Sequence[Dict], eid: str) -> List[Dict]:
    """单个实体的真实采样：[{"tick", "dt", "pos"}]。

    history 里没有该 eid 的帧直接跳过：缺帧不补采样、不插值。
    dt = 本采样 tick 与上一真实采样 tick 之差；首个采样记 0。
    pos 是新列表，调用方的 history 不被改写。
    """
    out: List[Dict] = []
    prev_tick: Optional[int] = None
    for rec in history:
        pos = rec["entities"].get(eid)
        if pos is None:
            continue
        tick = int(rec["tick"])
        out.append({"tick": tick,
                    "dt": 0 if prev_tick is None else tick - prev_tick,
                    "pos": [float(v) for v in pos]})
        prev_tick = tick
    return out


def pair_samples(history: Sequence[Dict], a: str, b: str) -> List[Dict]:
    """一对实体的成对真实采样：[{"tick", "dt", "distance", "a", "b"}]。

    只有 a 与 b 在同一帧都有观测才记一条；dt 取相邻成对采样的 tick 差。
    """
    out: List[Dict] = []
    prev_tick: Optional[int] = None
    for rec in history:
        entities = rec["entities"]
        pa, pb = entities.get(a), entities.get(b)
        if pa is None or pb is None:
            continue
        tick = int(rec["tick"])
        out.append({"tick": tick,
                    "dt": 0 if prev_tick is None else tick - prev_tick,
                    "distance": _planar_distance(pa, pb),
                    "a": [float(v) for v in pa],
                    "b": [float(v) for v in pb]})
        prev_tick = tick
    return out


def motion_events(history: Sequence[Dict], eid: str, *,
                  thresholds: Optional[ProcessThresholds] = None) -> List[Dict]:
    """单个实体的最小停止/转向事件，复用 track_samples 的采样与 dt。

    停止：相邻两个真实采样的水平速度（水平位移 ÷ dt）≤ stop_speed。
    转向：速度大于 stop_speed 的采样对，与前一个运动方向的变化 ≥ turn_deg。
    停止的采样对不更新参考方向；恢复运动后与停止前的运动方向比较。
    事件挂在后一个采样的 tick 上；缺帧处没有采样，因此不产生事件。
    速度按真实 dt 归一化：4 tick 间隔里漂移 0.4 与逐 tick 移动 0.4 不等价。

    dt <= 0 的采样对**不判事件**（相邻两帧同 tick / tick 回退都没有时间间隔，
    速度在数学上无定义）：既不判停止也不判转向，参考方向也不更新——该采样对
    不构成运动证据。此处**显式拒绝**该采样对，不用 `max(dt, eps)` 之类把物理量
    静默改成"看起来合理"的值。口径与 `motion.py::_estimate_detail` 的
    `if dt <= 0: continue` 逐字一致（同仓既有的位移对口径）。`track_samples`
    仍如实返回 dt=0 / dt<0 的采样读数（那是真实观测的 tick 差，不是待删的坏点）。
    """
    th = thresholds or ProcessThresholds()
    samples = track_samples(history, eid)
    events: List[Dict] = []
    last_dir: Optional[Tuple[float, float]] = None
    for prev, cur in zip(samples, samples[1:]):
        if cur["dt"] <= 0:
            continue
        dx = cur["pos"][0] - prev["pos"][0]
        dz = cur["pos"][2] - prev["pos"][2]
        moved = math.hypot(dx, dz)
        speed = moved / cur["dt"]
        if speed <= th.stop_speed:
            events.append({"tick": cur["tick"], "dt": cur["dt"], "event": "stop",
                           "moved": round(moved, 6), "speed": round(speed, 6)})
            continue
        direction = (dx / moved, dz / moved)
        if last_dir is not None:
            dot = max(-1.0, min(1.0, direction[0] * last_dir[0] + direction[1] * last_dir[1]))
            angle = math.degrees(math.acos(dot))
            if angle >= th.turn_deg:
                events.append({"tick": cur["tick"], "dt": cur["dt"],
                               "event": "turn", "angle_deg": round(angle, 3)})
        last_dir = direction
    return events


# ================= 时间会意：3 类过程 =================

def _hold_window(dists: Sequence[float], positions: Sequence[Sequence[float]], center: int,
                band: float, move: float) -> Tuple[int, int]:
    """包含 center 的最长连续样本区间：距离不超过 dists[center]+band，且 actor
    相对 center 样本的水平偏离不超过 move。停留是原地停留，因此等距绕行
    （距离不变、位置在动）不构成停留窗口。"""
    ref = dists[center]
    anchor = positions[center]
    n = len(dists)

    def holds(i: int) -> bool:
        return (dists[i] <= ref + band
                and _planar_distance(positions[i], anchor) <= move)

    lo = center
    while lo > 0 and holds(lo - 1):
        lo -= 1
    hi = center
    while hi < n - 1 and holds(hi + 1):
        hi += 1
    return lo, hi


def _offset(sample: Dict) -> Tuple[float, float]:
    """actor 相对 ref 的水平偏移。"""
    return (sample["a"][0] - sample["b"][0], sample["a"][2] - sample["b"][2])


def _bearing_sweep_deg(samples: Sequence[Dict]) -> float:
    """actor 绕 ref 的水平方位角总绝对变化量（度）。

    方位角 = atan2(actor.z − ref.z, actor.x − ref.x)。相邻采样的角差归一化到
    (-180°, 180°] 后取绝对值累加：等半径绕一圈约 360°，直线往返约 0°。
    actor 与 ref 重合时方位角无定义，该样本不参与累加。
    """
    total = 0.0
    prev: Optional[float] = None
    for s in samples:
        dx = s["a"][0] - s["b"][0]
        dz = s["a"][2] - s["b"][2]
        if dx == 0.0 and dz == 0.0:
            continue
        cur = math.atan2(dz, dx)
        if prev is not None:
            total += abs(math.atan2(math.sin(cur - prev), math.cos(cur - prev)))
        prev = cur
    return math.degrees(total)


def _loop_radius(points: Sequence[Sequence[float]]) -> float:
    """actor 自身绝对水平轨迹的闭合回路等效半径 = 2 × 回路面积 ÷ 折线周长。

    输入是 actor 自己的水平采样点 (x, z)，不是相对 ref 的偏移。
    面积取 shoelace 有向面积，周长含末点回到首点的闭合边。沿同一条直线往返时，
    往程与回程经过的三角形符号相反、面积相消，等效半径恒为 0；
    绕开参照物走一圈时回路面积为正，等效半径约等于回路的平均半径。
    参照物在窗口内移动不会改 actor 自己的轨迹：绝对回路判据不依赖 ref 是否移动。
    传入相对轨迹（actor − ref）时得到的是相对形状的等效半径，不是本判据用的量。
    """
    area2 = 0.0
    length = 0.0
    n = len(points)
    for i in range(n):
        x1, z1 = points[i]
        x2, z2 = points[(i + 1) % n]
        area2 += x1 * z2 - x2 * z1
        length += math.hypot(x2 - x1, z2 - z1)
    return abs(area2) / length if length > 0.0 else 0.0


def _ref_clearance(rel: Sequence[Sequence[float]]) -> float:
    """相对轨迹到参照物（相对坐标原点）的最小水平距离（世界单位）。

    逐条采样线段取到原点的最近距离，即 evidence 里的 ref_min_distance。
    该读数只作穿越判定：直线穿过参照物的轨迹读数为 0；
    绕行把参照物保持在回路之外，读数大于 0，但不要求最小分离达到某个距离。
    """
    best = math.inf
    for p, q in zip(rel, rel[1:]):
        dx, dz = q[0] - p[0], q[1] - p[1]
        seg2 = dx * dx + dz * dz
        if seg2 == 0.0:
            dist = math.hypot(p[0], p[1])
        else:
            t = max(0.0, min(1.0, -(p[0] * dx + p[1] * dz) / seg2))
            dist = math.hypot(p[0] + t * dx, p[1] + t * dz)
        if dist < best:
            best = dist
    return best


def _result(ptype: str, actor: str, ref: str, ticks: List[int],
            phases: List[Dict], evidence: Dict, n_samples: int) -> Dict:
    """过程结果：类型 / 角色 / 时间范围 / 阶段 / 数值依据。"""
    return {"type": ptype,
            "roles": {"actor": actor, "ref": ref},
            "ticks": [ticks[0], ticks[-1]],
            "phases": phases,
            "evidence": evidence,
            "samples": n_samples}


def _recognize_pair(a: str, b: str, samples: Sequence[Dict],
                    th: ProcessThresholds) -> Optional[Dict]:
    """判定一对实体的过程；a < b 由调用方保证，平手时取 a 作 actor。"""
    ticks = [s["tick"] for s in samples]
    dists = [s["distance"] for s in samples]
    a_pos = [s["a"] for s in samples]
    b_pos = [s["b"] for s in samples]
    n = len(samples)
    start_d, end_d = dists[0], dists[-1]

    # 角色：actor = 窗口内水平路程更长的实体；平手时取 a，保证确定性。
    path_a, path_b = _path_length(a_pos), _path_length(b_pos)
    actor, ref = (a, b) if path_a >= path_b else (b, a)
    actor_pos = a_pos if actor == a else b_pos
    ref_pos = b_pos if actor == a else a_pos
    actor_path = path_a if actor == a else path_b

    # ① 绕行后返回：actor 绕开 ref 后回到起点。三项几何证据：
    #    ① actor 自身绝对轨迹构成闭合回路（回路等效半径 ≥ detour_loop_radius）；
    #    ② actor 绕 ref 的方位角总变化 ≥ detour_sweep_deg（水平角度证据）；
    #    ③ actor 相对 ref 的轨迹不穿越参照物（到 ref 的最近距离 > detour_crossing_tol）。
    #    另有返回与路程：actor 回到起点 ±tol；距离回到起点距离 ±tol；路程 ≥ path_min。
    #    径向抬升不是判据：等半径绕行的抬升为 0，直线往返的抬升、actor 回位、
    #    水平路程都与绕行相同。直线穿过参照物再原路返回时，方位角在参照物两侧
    #    各跳 180°、累计 360°，但绝对轨迹是直线、往程与回程的面积相消、等效半径为 0。
    #    参照物移动时其自身位移会把相对轨迹撑出面积（相对等效半径可达 1.0）；
    #    绝对回路判据不看相对形状，因此不依赖 ref 是否移动。
    #    三项证据一起区分“绕开参照物的回路”与“把参照物当路点穿过的直线”。
    apex = max(range(n), key=lambda i: _planar_distance(actor_pos[0], actor_pos[i]))
    sweep_deg = _bearing_sweep_deg(samples)
    rel = [(p[0] - q[0], p[2] - q[2]) for p, q in zip(actor_pos, ref_pos)]
    loop_radius = _loop_radius([_planar(p) for p in actor_pos])
    ref_min_distance = _ref_clearance(rel)
    actor_return = _planar_distance(actor_pos[0], actor_pos[-1])
    dmax = max(dists)
    if (abs(end_d - start_d) <= th.detour_return_tol
            and actor_path >= th.detour_path_min
            and sweep_deg >= th.detour_sweep_deg
            and loop_radius >= th.detour_loop_radius
            and ref_min_distance > th.detour_crossing_tol
            and 0 < apex < n - 1
            and actor_return <= th.detour_return_tol):
        evidence = {"distance_start": round(start_d, 6),
                    "distance_peak": round(dmax, 6),
                    "distance_end": round(end_d, 6),
                    "rise": round(dmax - start_d, 6),
                    "return_gap": round(abs(end_d - start_d), 6),
                    "return_tol": th.detour_return_tol,
                    "actor_path": round(actor_path, 6),
                    "actor_path_min": th.detour_path_min,
                    "actor_return": round(actor_return, 6),
                    "loop_radius": round(loop_radius, 6),
                    "loop_radius_min": th.detour_loop_radius,
                    "ref_min_distance": round(ref_min_distance, 6),
                    "crossing_tol": th.detour_crossing_tol,
                    "bearing_sweep_deg": round(sweep_deg, 6),
                    "bearing_sweep_min_deg": th.detour_sweep_deg}
        phases = [{"phase": PHASE_DETOUR, "ticks": [ticks[0], ticks[apex]]},
                  {"phase": PHASE_RETURN, "ticks": [ticks[apex], ticks[-1]]}]
        return _result(PROCESS_DETOUR_RETURN, actor, ref, ticks, phases, evidence, n)

    # ② 接近→停留→离开：距离先净缩短 ≥ approach_drop，在最近点附近保持 band
    #    内、且 actor 相对最近点偏离 ≤ stay_move 的窗口里，有 hold_min_samples
    #    个真实采样且覆盖 hold_min_ticks 个 tick，再相对最近点净拉长 ≥ leave_rise。
    imin = min(range(n), key=lambda i: dists[i])
    lo, hi = _hold_window(dists, actor_pos, imin, th.stay_band, th.stay_move)
    drop = start_d - dists[imin]
    rise = dists[-1] - dists[imin]
    held_samples, held_ticks = hi - lo + 1, ticks[hi] - ticks[lo]
    if (drop >= th.approach_drop and rise >= th.leave_rise
            and held_samples >= th.hold_min_samples and held_ticks >= th.hold_min_ticks):
        evidence = {"distance_start": round(start_d, 6),
                    "distance_min": round(dists[imin], 6),
                    "distance_end": round(end_d, 6),
                    "approach_drop": round(drop, 6),
                    "leave_rise": round(rise, 6),
                    "stay_band": th.stay_band,
                    "stay_move": th.stay_move,
                    "stay_samples": held_samples,
                    "stay_held_ticks": held_ticks,
                    "actor_path": round(actor_path, 6)}
        phases = [{"phase": PHASE_APPROACH, "ticks": [ticks[0], ticks[imin]]},
                  {"phase": PHASE_STAY, "ticks": [ticks[lo], ticks[hi]]},
                  {"phase": PHASE_LEAVE, "ticks": [ticks[hi], ticks[-1]]}]
        return _result(PROCESS_APPROACH_STAY_LEAVE, actor, ref, ticks, phases, evidence, n)

    # ③ 汇合后同行：距离从 meet_gap 以上降到 meet_gap 以下（汇合），
    #    其后相对偏移偏离汇合时偏移不超过 escort_offset_tol，且双方都在移动。
    imeet = next((i for i in range(n) if dists[i] <= th.meet_gap), None)
    if imeet is not None and imeet > 0:
        base = _offset(samples[imeet])
        dev = max(math.hypot(ox - base[0], oz - base[1])
                  for ox, oz in (_offset(s) for s in samples[imeet:]))
        # 角色：actor = 汇合阶段位移更大的一方。
        conv_a = _planar_distance(a_pos[0], a_pos[imeet])
        conv_b = _planar_distance(b_pos[0], b_pos[imeet])
        actor, ref = (a, b) if conv_a >= conv_b else (b, a)
        actor_pos = a_pos if actor == a else b_pos
        ref_pos = b_pos if actor == a else a_pos
        move_actor = _planar_distance(actor_pos[imeet], actor_pos[-1])
        move_ref = _planar_distance(ref_pos[imeet], ref_pos[-1])
        esc_samples = n - imeet
        esc_ticks = ticks[-1] - ticks[imeet]
        if (dev <= th.escort_offset_tol and move_actor >= th.escort_move
                and move_ref >= th.escort_move
                and esc_samples >= th.hold_min_samples and esc_ticks >= th.hold_min_ticks):
            evidence = {"distance_start": round(start_d, 6),
                        "distance_meet": round(dists[imeet], 6),
                        "meet_gap": th.meet_gap,
                        "offset_deviation": round(dev, 6),
                        "offset_tol": th.escort_offset_tol,
                        "escort_samples": esc_samples,
                        "escort_ticks": esc_ticks,
                        "actor_move": round(move_actor, 6),
                        "ref_move": round(move_ref, 6)}
            phases = [{"phase": PHASE_CONVERGE, "ticks": [ticks[0], ticks[imeet]]},
                      {"phase": PHASE_ESCORT, "ticks": [ticks[imeet], ticks[-1]]}]
            return _result(PROCESS_MEET_ESCORT, actor, ref, ticks, phases, evidence, n)

    return None


def recognize_processes(history: Sequence[Dict], *,
                        thresholds: Optional[ProcessThresholds] = None,
                        entities: Optional[Sequence[str]] = None) -> List[Dict]:
    """识别观察窗口内的过程，返回过程结果列表（无过程时为空列表）。

    history 按 tick 递增（perceive 保证）；本函数不改写 history。
    entities 限定参与判定的实体；省略时取 history 中出现过的全部实体。
    结果按 (起始 tick, 类型, actor, ref) 排序，同一输入重复调用结果相同。
    """
    th = thresholds or ProcessThresholds()
    if entities is None:
        seen = set()
        for rec in history:
            seen.update(rec["entities"])
        eids = sorted(seen)
    else:
        eids = sorted(set(str(e) for e in entities))
    out: List[Dict] = []
    for index, a in enumerate(eids):
        for b in eids[index + 1:]:
            samples = pair_samples(history, a, b)
            if len(samples) < th.min_samples:
                continue
            proc = _recognize_pair(a, b, samples, th)
            if proc is not None:
                out.append(proc)
    out.sort(key=lambda p: (p["ticks"][0], p["type"], p["roles"]["actor"],
                            p["roles"]["ref"]))
    return out


# ================= 示教迁移 =================

def extract_relative_trajectory(history: Sequence[Dict], actor: str, ref: str, *,
                                process: Optional[Dict] = None) -> Dict:
    """示教期的相对轨迹记录（JSON 可存）。

    rel = actor 位置 − ref 位置（含 y），tick 与 dt 保留原采样值，不重采样。
    process 参数保存示教原意（recognize_processes 的某条结果，可为 None）。
    """
    samples = pair_samples(history, actor, ref)
    return {"kind": "relative_trajectory",
            "actor": actor,
            "ref": ref,
            "process": process,
            "ticks": [samples[0]["tick"], samples[-1]["tick"]] if samples else [],
            "samples": [{"tick": s["tick"], "dt": s["dt"],
                         "ref": list(s["b"]),
                         "rel": [s["a"][k] - s["b"][k] for k in range(3)]}
                        for s in samples]}


def transfer_demonstration(record: Dict, actor: str, ref: str, *,
                           translation: Sequence[float] = (0.0, 0.0, 0.0),
                           yaw: float = 0.0,
                           scale: float = 1.0,
                           tick_offset: int = 0) -> Dict:
    """把相对轨迹记录绑定到新 actor/ref，输出带 tick 的生成帧轨迹。

    yaw 与 scale 是整体场景变换，锚点是记录首帧的 ref 位置 ref0：
        ref 帧位置 = ref0 + translation + 旋转(scale × (记录 ref 位置 − ref0))
        执行者帧位置 = ref 帧位置 + 旋转(scale × rel)
    变换顺序：刻度 scale → 绕 +y 轴右手旋转 yaw → 平移 translation。
    ref 静止时 ref 位移为 0，此时只有 rel 随 yaw 与 scale 变化。
    角色变换只改帧里的 eid：记录本身不改写。
    translation 取 (0,0,0)、yaw 取 0、scale 取 1 时即原坐标回放。
    scale 必须为正数；record 必须含至少 1 条采样；两条约束在此处只查一次。
    """
    samples = record.get("samples") or []
    if not samples:
        raise ValueError("相对轨迹记录没有采样，无法迁移")
    # #1136：`nan <= 0.0` 与 `inf <= 0.0` 均为 False ⇒ NaN/inf 曾穿透此门，
    # 产出 transform.scale=NaN/inf 的「生成帧」原样回填（下游无法察觉失败）。
    # 与本仓同类数值门（Environment 的 math.isfinite 族）同口径收口。
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("尺度必须为正的有限数，收到 %r" % (scale,))
    tx, ty, tz = (float(v) for v in translation)
    cos_y, sin_y = math.cos(float(yaw)), math.sin(float(yaw))
    ox0, oy0, oz0 = (float(v) for v in samples[0]["ref"])
    frames: List[Dict] = []
    for s in samples:
        rx, ry, rz = (float(v) for v in s["ref"])
        # ref 位移与 rel 走同一条变换：先刻度 scale，再绕 +y 轴旋转 yaw。
        mx, my, mz = (rx - ox0) * scale, (ry - oy0) * scale, (rz - oz0) * scale
        ref_pos = [ox0 + tx + mx * cos_y + mz * sin_y,
                   oy0 + ty + my,
                   oz0 + tz - mx * sin_y + mz * cos_y]
        dx, dy, dz = (float(v) * scale for v in s["rel"])
        frames.append({"tick": int(s["tick"]) + int(tick_offset),
                       "dt": int(s["dt"]),
                       "entities": {ref: ref_pos,
                                    actor: [ref_pos[0] + dx * cos_y + dz * sin_y,
                                            ref_pos[1] + dy,
                                            ref_pos[2] - dx * sin_y + dz * cos_y]},
                       "source": GENERATED})
    return {"kind": "demonstration_transfer",
            "source": GENERATED,
            "actor": actor,
            "ref": ref,
            "transform": {"translation": [tx, ty, tz],
                          "yaw": float(yaw),
                          "scale": float(scale),
                          "tick_offset": int(tick_offset)},
            "ticks": [frames[0]["tick"], frames[-1]["tick"]],
            "frames": frames}

# ================= 事件回声：事件间的延迟关联 =================
# 事件源复用上文的 motion_events：只取停止与转向两类事件。
# 本段只学“事件 A 发生后 lag 个 tick 后事件 B 发生”这一时间关联，
# 关联不代表 A 引起 B，结果字段与注释都不使用因果表述。
# 配对口径（本版修正）：
#   · 统计键保留 source 与 target 两侧的 eid 与 event，即四元身份
#     (source_eid, source_event, target_eid, target_event)；同一事件类型组合下
#     同实体对与跨实体对分开计数，跨实体的固定签名对仍可学。
#   · 候选对按事件流位置枚举：对每个源事件取该事件之后 lag 在 1..max_lag 内的每个事件，
#     同 tick 上的其它事件既是别人的候选，也仍是本源的候选，
#     所以同 tick 簇不打断同一签名对的相邻事件对。
#   · 事件流里的位置间隔不参与统计，插入无关事件不改变同一签名对的 lag 计数。

def _echo_stream(events: Sequence[Dict]) -> List[Dict]:
    """事件流按 (tick, eid, event) 严格升序；只读输入，不改写。"""
    return sorted(events, key=lambda e: (int(e["tick"]), str(e["eid"]),
                                         str(e["event"])))


def _echo_key(event: Dict) -> Tuple[int, str, str]:
    """事件身份：tick、eid、event 三者一起。"""
    return (int(event["tick"]), str(event["eid"]), str(event["event"]))


def _echo_rank(assoc: Dict) -> Tuple[int, int, str, str]:
    """关联排序键：count 降序、lag 升序、target_event 升序、target_eid 升序。"""
    return (-int(assoc["count"]), int(assoc["lag"]),
            str(assoc["target_event"]), str(assoc["target_eid"]))


def _echo_pick_lag(lags: Dict[int, int]) -> int:
    """同一签名对多个候选 lag 时取次数最多的一条；次数相同取 lag 最小的一条。"""
    keys = [int(lag) for lag in lags]
    return min(keys, key=lambda lag: (-lags[lag], lag))


def _echo_source_of(assoc: Dict) -> Tuple[str, str]:
    """关联键的源侧：eid 与 event。"""
    return (str(assoc["source_eid"]), str(assoc["source_event"]))


def _echo_prediction_of(event: Dict, lag: int,
                        associations: Sequence[Dict]) -> Optional[Dict]:
    """按签名对的触发数取一条关联，再给出预测。

    触发集合只由当前源事件（eid + event）与传入的关联决定；本函数不读实际下一事件，
    也不读留出段的未来事件。返回 {"assoc", "predicted_eid", "predicted_event",
    "predicted_tick", "lag"}；当前源事件没有可用关联时返回 None。
    lag 由调用方给出：延迟模型用关联学到的 lag，无延迟模型用 0。
    """
    source = (str(event["eid"]), str(event["event"]))
    ranked = sorted([a for a in associations if _echo_source_of(a) == source],
                    key=_echo_rank)
    if not ranked:
        return None
    assoc = ranked[0]
    return {"assoc": assoc,
            "predicted_eid": str(assoc["target_eid"]),
            "predicted_event": str(assoc["target_event"]),
            "predicted_tick": int(event["tick"]) + int(lag),
            "lag": int(lag)}


def _echo_predicted_key(predicted: Optional[Dict]) -> Optional[Tuple[int, str, str]]:
    """预测的三键身份：预测 tick、目标 eid、目标 event。"""
    if predicted is None:
        return None
    return (int(predicted["predicted_tick"]), str(predicted["predicted_eid"]),
            str(predicted["predicted_event"]))


def _echo_matches_signature(item: Dict, target: Tuple[str, str]) -> bool:
    """事件是否属于固定目标签名；签名只含 eid 与 event 两侧（tick 不参与筛选）。"""
    return (str(item["eid"]), str(item["event"])) == target


def _echo_response_for(stream: Sequence[Dict], row: Dict,
                       predicted: Optional[Dict]) -> Optional[Dict]:
    """固定目标签名在 row 之后最早的一条响应事件；row 自身与排在 row 前面的事件都不算。

    目标签名取预测的 target_eid 与 target_event。无关实体的插入或删除不改变
    同一目标签名的最早响应：别的实体的事件既不是候选，也不会挤掉候选。
    事件流按 (tick, eid, event) 严格升序，身份键大于 row 的键即严格在 row 之后，
    所以源事件自己不会成为自己的响应。
    本函数只用事件流与预测，不读留出段的实际下一事件，也不读入参 actual_next。
    """
    if predicted is None:
        return None
    here = _echo_key(row)
    target = (str(predicted["predicted_eid"]), str(predicted["predicted_event"]))
    for item in stream:
        if _echo_key(item) <= here:
            continue
        if _echo_matches_signature(item, target):
            return item
    return None


def _echo_beyond_stream(stream: Sequence[Dict], predicted: Optional[Dict]) -> bool:
    """预测 tick 是否超出已观察到的事件流尾部。

    观察窗口止于事件流最后一条事件的 tick；预测 tick 大于该 tick 时窗口已经结束，
    本行属于右侧删失，不给命中也不给偏差，另按 prediction_beyond_window 计数。
    """
    if predicted is None or not stream:
        return False
    return int(predicted["predicted_tick"]) > int(stream[-1]["tick"])


def _echo_score(stream: Sequence[Dict], row: Dict,
                predicted: Optional[Dict]) -> Dict:
    """唯一的命中判据：目标 eid、event、tick 三者全部吻合才算命中。

    真实响应取固定目标签名在 row 之后最早的一条事件；row 自身不参与比对，
    所以“预测落在本事件自己的 tick 上”不会靠本事件顶替成命中。
    实际三键全部取自同一个响应对象，不混用入参与事件流两处来源。
    end_state 取四者之一：没有预测记 no_prediction；命中记 hit；
    预测 tick 超出观察尾部记 beyond_window；其余落空记 missed。
    scored = end_state 为 hit 或 missed，即判据真正分辨出结果的行。
    """
    response = _echo_response_for(stream, row, predicted)
    predicted_key = _echo_predicted_key(predicted)
    actual_key = None if response is None else _echo_key(response)
    beyond = _echo_beyond_stream(stream, predicted)
    hit = bool(actual_key is not None and predicted_key is not None
               and actual_key == predicted_key)
    if predicted is None:
        end_state = "no_prediction"
    elif hit:
        end_state = "hit"
    elif beyond:
        end_state = "beyond_window"
    else:
        end_state = "missed"
    return {"response": response, "actual_key": actual_key,
            "predicted_key": predicted_key, "hit": hit,
            "beyond_window": bool(beyond), "end_state": end_state}


def _echo_row(stream: Sequence[Dict], event: Dict, lag: int,
              associations: Sequence[Dict]) -> Dict:
    """一条预测读数：预测、命中、评分状态与偏差都在这里落定。

    actual_tick、actual_eid、actual_event 三个键全部取自同一个实际响应对象，
    即固定目标签名在源事件之后最早的一条；没有响应时三个键一起为 None。
    预测超出观察尾部时 end_state 记 beyond_window，偏差为 None 且不参与 MAE。
    """
    predicted = _echo_prediction_of(event, lag, associations)
    scored = _echo_score(stream, event, predicted)
    response = scored["response"]
    predicted_tick = None if predicted is None else int(predicted["predicted_tick"])
    return {"tick": int(event["tick"]),
            "eid": str(event["eid"]),
            "event": str(event["event"]),
            "after": str(event["event"]),
            "actual_tick": None if response is None else int(response["tick"]),
            "actual_eid": None if response is None else str(response["eid"]),
            "actual_event": None if response is None else str(response["event"]),
            "predicted_tick": predicted_tick,
            "predicted_event": (None if predicted is None
                                else str(predicted["predicted_event"])),
            "predicted_eid": (None if predicted is None
                              else str(predicted["predicted_eid"])),
            "lag": None if predicted is None else int(predicted["lag"]),
            "hit": bool(scored["hit"]),
            "scored": scored["end_state"] in ("hit", "missed"),
            "end_state": scored["end_state"],
            "abs_tick_error": (
                None if response is None or predicted is None
                or scored["beyond_window"]
                else abs(predicted_tick - int(response["tick"])))}


def _echo_walk(stream: Sequence[Dict], lag: Optional[int],
               associations: Sequence[Dict]) -> List[Dict]:
    """逐条填预测与命中；延迟模型与无延迟模型走同一条路径。

    两个模型的触发集合（有该签名对的源事件）、目标签名与命中判据全部相同，
    差别只有 lag：lag=None 时用每条关联学到的 lag，lag=0 时预测与当前事件同 tick。
    """
    rows: List[Dict] = []
    for event in stream:
        predicted = _echo_prediction_of(event, 0, associations)
        use = 0 if predicted is None else (0 if lag == 0
                                           else int(predicted["assoc"]["lag"]))
        rows.append(_echo_row(stream, event, use, associations))
    return rows


def _echo_metrics(rows: Sequence[Dict]) -> Dict:
    """一个模型的读数：预测、评分、未决尾部三类数量与误差。

    predictions = 给出预测 tick 的行数，no_prediction = 没有预测的行数，
    两者相加等于 events。scored = 判据真正分辨出结果的行数（命中与落空都算），
    unscored = 未评分行数，两者相加等于 events。
    prediction_beyond_window = 预测 tick 超出已观察尾部、判据分辨不出结果的行数，
    单列计数，不与普通落空混在一起；unresolved_window_end 与该键同值，保留旧键名。
    三类构成 events 的一个划分：scored + no_prediction + prediction_beyond_window
    = events。未预测与超窗的事件都不删，计数留在 events 与 misses 里。
    hit_rate = 命中数 ÷ 事件数；coverage = predictions ÷ 事件数，即给出预测的比例；
    scored_rate = scored ÷ 事件数。
    misses = 非命中行数 = events − hits，含无预测、超窗与真实落空三类。
    偏差只统计判据分辨出结果、且响应落在观察窗口内的预测；
    超出观察尾部的预测不计入偏差，也不进 scored。
    """
    total = len(rows)
    hits = sum(1 for r in rows if r["hit"])
    predictions = sum(1 for r in rows if r["predicted_tick"] is not None)
    scored = [r for r in rows if r["scored"]]
    beyond = sum(1 for r in rows if r["end_state"] == "beyond_window")
    errors = [int(r["abs_tick_error"]) for r in rows
              if r["abs_tick_error"] is not None]
    return {"events": total,
            "predictions": predictions,
            "hits": int(hits),
            "misses": total - int(hits),
            "scored": len(scored),
            "no_prediction": total - predictions,
            "unscored": total - len(scored),
            "prediction_beyond_window": beyond,
            "unresolved_window_end": beyond,
            "hit_rate": round(int(hits) / total, 6) if total else 0.0,
            "coverage": round(predictions / total, 6) if total else 0.0,
            "scored_rate": round(len(scored) / total, 6) if total else 0.0,
            "mean_abs_tick_error": (round(sum(errors) / len(errors), 6)
                                    if errors else None),
            "abs_tick_errors": errors}


def event_echo_readings(history: Sequence[Dict], *,
                        thresholds: Optional[ProcessThresholds] = None,
                        entities: Optional[Sequence[str]] = None) -> List[Dict]:
    """把所有实体的停止/转向事件合成一条事件流（复用 motion_events）。

    history 只读；返回按 (tick, eid, event) 排序的新列表，元素形如
    {"tick", "eid", "event", "dt", "lag_from_prev"}。
    lag_from_prev = 本事件 tick − 前一事件 tick（同 tick 记 0），按真实 tick 计。
    省略 entities 时取 history 中出现过的全部实体。
    同一 tick 可以有多条事件：不同实体各一条，本函数逐条保留。
    """
    if entities is None:
        seen = set()
        for rec in history:
            seen.update(rec["entities"])
        eids = sorted(seen)
    else:
        eids = sorted(set(str(e) for e in entities))
    events: List[Dict] = []
    for eid in eids:
        for item in motion_events(history, eid, thresholds=thresholds):
            events.append({"tick": int(item["tick"]),
                           "eid": eid,
                           "event": item["event"],
                           "dt": item["dt"],
                           "lag_from_prev": 0})
    events.sort(key=lambda e: (e["tick"], e["eid"], e["event"]))
    prev_tick: Optional[int] = None
    for e in events:
        e["lag_from_prev"] = 0 if prev_tick is None else e["tick"] - prev_tick
        prev_tick = e["tick"]
    return events


def echo_lag_associations(events: Sequence[Dict], *,
                          max_lag: int = 12, min_support: int = 2,
                          min_consistency: float = 0.7) -> List[Dict]:
    """从事件流学习固定事件签名对的延迟关联。

    统计口径：签名对 = (source_eid, source_event, target_eid, target_event)。
    计数口径：对每个源事件，按 (target_eid, target_event) 分桶，每桶只记该桶里
    最早的一条响应（最接近源事件的那条），桶内更晚的响应不重复计数。
    事件流里的位置间隔不参与统计，因此插入无关事件不改变同一签名对测到的 lag，
    同 tick 簇也不打断同一签名对的配对；同一 tick 上的多条事件按整组处理。
    lag=0（同 tick 的两条事件）不进入延迟统计。

    输出条件（两条都要满足）：
      · count = 该签名对在主 lag 上的计数 ≥ min_support；
      · consistency = 主 lag 计数 ÷ 该源签名在 1..max_lag 内有任意响应的触发数
        ≥ min_consistency。

    consistency 的分子与分母口径：
      · 分子 count 只数这一个四元签名对在主 lag 上的次数，含目标选择成分；
      · 分母 sources_with_pair（= triggers）数该源签名（eid + event）在 1..max_lag
        内至少有一条候选目标的源事件条数，候选不限目标签名，含 lag 集中成分。
    所以 consistency 同时混合目标覆盖与 lag 集中两个成分，不单独衡量其中任何一项。
    sources_total 是该源签名在事件流里出现的总条数；响应覆盖率另按
    sources_with_pair ÷ sources_total 读，不由 consistency 承担。
    lag_consistency = count ÷ 该签名对全部 lag 计数之和，只作原始读数，
    不参与门槛，也没有做检验。
    min_support 过滤绝对次数不足的签名对，min_consistency 过滤主 lag 占比不足的
    签名对。本段不做统计检验，也不保证随机序列不出现偶然候选。

    返回按 (after, before, target_eid) 排序的新列表，元素：
    {"after", "before", "source_eid", "source_event", "target_eid", "target_event",
     "lag", "count", "lag_counts", "triggers", "sources_total", "sources_with_pair",
     "consistency", "lag_consistency"}。
    after/before 沿用源与目标的事件类型，供既有调用方读取。
    无延迟关联的序列返回空列表；关联只表示两个事件在时间上的先后与间隔，不表示因果作用。
    """
    stream = _echo_stream(events)
    lags: Dict[Tuple[str, str, str, str], Dict[int, int]] = {}
    responded: set = set()
    for src in stream:
        src_key = (str(src["eid"]), str(src["event"]))
        first: Dict[Tuple[str, str], int] = {}
        for tgt in stream:
            target = (str(tgt["eid"]), str(tgt["event"]))
            if target in first:
                continue            # 同一签名只记最早的一条响应，不重复计数
            lag = int(tgt["tick"]) - int(src["tick"])
            if lag < 1:
                continue
            if lag > int(max_lag):
                break               # 事件流按 tick 升序，后面的更远
            first[target] = lag
        for target in sorted(first):
            sig = (src_key[0], src_key[1], target[0], target[1])
            lags.setdefault(sig, {})
            lag = first[target]
            lags[sig][lag] = lags[sig].get(lag, 0) + 1
            responded.add(_echo_key(src))
    with_pair: Dict[Tuple[str, str], int] = {}
    for item in stream:
        if _echo_key(item) in responded:
            key = (str(item["eid"]), str(item["event"]))
            with_pair[key] = with_pair.get(key, 0) + 1
    sources_total: Dict[Tuple[str, str], int] = {}
    for e in stream:
        src_key = (str(e["eid"]), str(e["event"]))
        sources_total[src_key] = sources_total.get(src_key, 0) + 1
    out: List[Dict] = []
    for sig in sorted(lags):
        lag = _echo_pick_lag(lags[sig])
        count = lags[sig][lag]
        triggers = int(with_pair.get((sig[0], sig[1]), 0))
        if count < int(min_support) or triggers <= 0:
            continue
        consistency = round(count / triggers, 6)
        if consistency < float(min_consistency):
            continue
        pair_total = sum(int(v) for v in lags[sig].values())
        out.append({"after": sig[1], "before": sig[3],
                    "source_eid": sig[0], "source_event": sig[1],
                    "target_eid": sig[2], "target_event": sig[3],
                    "lag": int(lag), "count": int(count),
                    "lag_counts": {k: lags[sig][k] for k in sorted(lags[sig])},
                    "triggers": triggers,
                    "sources_total": int(sources_total.get((sig[0], sig[1]), 0)),
                    "sources_with_pair": triggers,
                    "consistency": consistency,
                    "lag_consistency": round(count / pair_total, 6)})
    out.sort(key=lambda a: (a["after"], a["before"], a["target_eid"]))
    return out


def _echo_inside_tick_group(stream: Sequence[Dict], index: int) -> bool:
    """下标 index 处是否落在同一 tick 组的内部。

    只比较 index 与前一条：前一条同 tick 时，index 位于该 tick 组中间，
    边界必须向前移到该组首。后一条同 tick 只说明 index 在组首或组内，
    不构成“落在组内部”的判据，因此不得据 index+1 继续多退一组。
    """
    return (0 < index < len(stream)
            and int(stream[index - 1]["tick"]) == int(stream[index]["tick"]))


def echo_split(events: Sequence[Dict], holdout_fraction: float = 0.3) -> Dict:
    """把事件流按 tick 切成开发段与留出段；同一 tick 的事件整组进同一段。

    留出段条数 holdout_count = ceil(事件数 × holdout_fraction)，
    切分下标 split_index = 事件数 − holdout_count：留出段是流的最后 holdout_count 条。
    holdout_fraction=0 时留出段为空，=1 时全部进留出段。
    切分边界落在同一 tick 组内部（前一条与本条同 tick）时，边界向前移到该组首，
    整组进留出段；因此只要两段都非空，开发段末尾 tick 严格小于留出段首个 tick。
    同一 tick 组大于按比例算出的留出条数时，留出段条数会多于比例值。
    两段不重叠，任一条事件只出现在一段里；输入列表不改写。
    返回 {"events", "holdout_fraction", "holdout_events", "dev_events",
    "boundary_tick", "dev_last_tick", "holdout_first_tick", "same_tick_grouped",
    "dev", "holdout"}。
    """
    stream = _echo_stream(events)
    total = len(stream)
    holdout_count = max(0, min(total, int(math.ceil(total * float(holdout_fraction)))))
    index = total - holdout_count
    # 边界落在同 tick 组内部时整组进留出段：向前移到该组首。
    while 0 < index < total and _echo_inside_tick_group(stream, index):
        index -= 1
    dev = stream[:index]
    holdout = stream[index:]
    return {"events": total,
            "holdout_fraction": float(holdout_fraction),
            "holdout_events": len(holdout),
            "dev_events": len(dev),
            "boundary_tick": None if not holdout else int(holdout[0]["tick"]),
            "dev_last_tick": None if not dev else int(dev[-1]["tick"]),
            "holdout_first_tick": None if not holdout else int(holdout[0]["tick"]),
            "same_tick_grouped": True,
            "dev": dev,
            "holdout": holdout}


def echo_lag_prediction(event: Dict, actual_next: Optional[Dict], *,
                        associations: Sequence[Dict],
                        events: Sequence[Dict]) -> Optional[Dict]:
    """用延迟关联预测 event 之后的下一个事件，并与实际下一事件对照。

    events 是查找响应的事件流（留出段）的唯一来源。预测的目标签名（eid + event）
    与 lag 只由 associations 与当前源事件决定：目标签名取该源事件签名对里触发数
    最多的一条关联，预测 tick = 当前 tick + 该关联的 lag。
    actual_next 只作入参兼容保留：不参与选择关联，也不参与 actual 三键与偏差的
    计算。actual_tick、actual_eid、actual_event 三个键统一取同一个响应对象，即固定
    目标签名在 events 里严格晚于本事件的最早一条；本函数不读留出段的未来决定模型。

    命中判据与 evaluate_event_echo 共用 _echo_score：目标 eid、event、tick 三者
    全部吻合才算命中；本事件自身不能顶替成命中；预测 tick 超出 events 尾部时
    end_state 记 beyond_window，偏差为 None 且不参与统计。
    返回 {"tick", "after", "source_eid", "predicted_eid", "predicted_event", "lag",
    "predicted_tick", "actual_tick", "actual_eid", "actual_event", "hit", "scored",
    "end_state", "abs_tick_error"}；没有可用关联时返回 None。
    """
    predicted = _echo_prediction_of(event, 0, associations)
    if predicted is None:
        return None
    assoc = predicted["assoc"]
    predicted_tick = int(event["tick"]) + int(assoc["lag"])
    stream = _echo_stream(events)
    row = {"tick": int(event["tick"]), "eid": str(event["eid"]),
           "event": str(event["event"])}
    scored = _echo_score(stream, row, {"predicted_tick": predicted_tick,
                                       "predicted_eid": str(assoc["target_eid"]),
                                       "predicted_event": str(assoc["target_event"])})
    response = scored["response"]
    return {"tick": int(event["tick"]),
            "after": str(event["event"]),
            "source_eid": str(event["eid"]),
            "predicted_eid": str(assoc["target_eid"]),
            "predicted_event": str(assoc["target_event"]),
            "lag": int(assoc["lag"]),
            "predicted_tick": predicted_tick,
            "actual_tick": None if response is None else int(response["tick"]),
            "actual_eid": None if response is None else str(response["eid"]),
            "actual_event": None if response is None else str(response["event"]),
            "hit": bool(scored["hit"]),
            "scored": scored["end_state"] in ("hit", "missed"),
            "end_state": scored["end_state"],
            "abs_tick_error": (None if response is None
                               or scored["beyond_window"]
                               else abs(predicted_tick - int(response["tick"])))}


def evaluate_event_echo(dev_events: Sequence[Dict],
                        holdout_events: Sequence[Dict], *,
                        max_lag: int = 12, min_support: int = 2,
                        min_consistency: float = 0.7) -> Dict:
    """在留出片段上比较延迟模型与无延迟模型。

    延迟模型：用开发段学到的关联预测下一个事件，预测 tick = 当前 tick + 关联 lag，
    目标签名取该关联的 target_eid + target_event。
    无延迟模型：触发集合、目标签名与命中判据与延迟模型完全相同，只把 lag 记成 0，
    即预测下一个事件与当前事件同 tick。
    两个模型共用 _echo_prediction_of 与 _echo_score 这一条路径；评分只在一处。
    留出段的实际事件不参与关联学习与图对统计，也不参与模型或目标的选择。

    读数含四类数量：predictions / no_prediction / scored / prediction_beyond_window；
    coverage = predictions ÷ events，scored_rate = scored ÷ events。未预测与超窗的
    事件都不删，计数留在 events 与 misses 里。
    两个模型共用同一真实响应对象（固定目标签名在源事件之后最早的一条），
    所以触发集合、目标签名与偏差口径逐条可比；只有预测 tick 不同，
    因此 lag 模型的预测可能先越过观察尾部而记 beyond_window。
    同一输入重复调用返回相同结果，输入列表不改写。
    """
    associations = echo_lag_associations(dev_events, max_lag=max_lag,
                                         min_support=min_support,
                                         min_consistency=min_consistency)
    stream = _echo_stream(holdout_events)
    lag_rows = _echo_walk(stream, None, associations)
    base_rows = _echo_walk(stream, 0, associations)
    return {"kind": "event_echo_evaluation",
            "models": ["lag", "no_delay"],
            "max_lag": int(max_lag),
            "min_support": int(min_support),
            "min_consistency": float(min_consistency),
            "dev_events": len(dev_events),
            "holdout_events": len(stream),
            "associations": associations,
            "lag_model": _echo_metrics(lag_rows),
            "no_delay_model": _echo_metrics(base_rows),
            "lag_readings": lag_rows,
            "no_delay_readings": base_rows}


def event_echo_report(history: Sequence[Dict], *,
                      thresholds: Optional[ProcessThresholds] = None,
                      max_lag: int = 12, min_support: int = 2,
                      holdout_fraction: float = 0.3,
                      min_consistency: float = 0.7,
                      entities: Optional[Sequence[str]] = None) -> Dict:
    """事件回声的完整读数：事件流 + 开发/留出切分 + 两模型对照。

    按 thresholds 取事件流，按 holdout_fraction 与 tick 边界切分，在开发段学延迟
    关联（min_support 与 min_consistency 两道门槛），在留出段比较延迟模型与无延迟
    模型。返回 {"kind", "thresholds", "events", "split", "associations", "evaluation"}；
    core 与演示只读这些键，新增读数挂在 thresholds、split、associations 与 evaluation
    内部，主要键仍按原名返回。全部读数由延迟关联与事件时间得出，不含因果结论。
    """
    th = thresholds or ProcessThresholds()
    events = event_echo_readings(history, thresholds=thresholds, entities=entities)
    split = echo_split(events, holdout_fraction)
    evaluation = evaluate_event_echo(split["dev"], split["holdout"],
                                     max_lag=max_lag, min_support=min_support,
                                     min_consistency=min_consistency)
    return {"kind": "event_echo_report",
            "thresholds": {"max_lag": int(max_lag), "min_support": int(min_support),
                           "min_consistency": float(min_consistency),
                           "holdout_fraction": float(holdout_fraction),
                           "stop_speed": th.stop_speed, "turn_deg": th.turn_deg},
            "events": events,
            "split": {"events": split["events"],
                      "dev_events": split["dev_events"],
                      "holdout_events": split["holdout_events"],
                      "boundary_tick": split["boundary_tick"],
                      "dev_last_tick": split["dev_last_tick"],
                      "holdout_first_tick": split["holdout_first_tick"],
                      "same_tick_grouped": split["same_tick_grouped"]},
            "associations": evaluation["associations"],
            "evaluation": evaluation}
