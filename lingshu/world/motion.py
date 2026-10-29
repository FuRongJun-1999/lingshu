"""Motion statistics from real, timestamped observations only.

issue #418（有限带宽稀疏观测下必须真的学到运动速度）
----------------------------------------------------
缺陷：稀疏观测下 `ObservationWindow.motion()` 会**丢掉可用的真实样本**。旧回填
条件是 `if points and self.start and len(points) < len(self.frames)`，两处漏：
  · 窗口内**一个**观测都没有时（长期遮挡 / 观测周期 > 观测窗口）整段跳过回填，
    估计返回 None ⇒ 下游回落默认 `persistence=0.0, speed_est=0.3`——**上一次真实
    观测连同它的时间戳一起被丢**（"不是样本不足，是可用样本被丢"）；
  · 即便触发也只回填**一个**点 ⇒ 窗口内 0 点时仍不足 2 点，构不成一次位移。

修法（照 issue 给的方向，允许重写估计路径）：
  1. **按需回填**：窗口内 0 点时取更早历史里最近的**两个**真实观测配对（一次位移
     至少 2 点）；窗口内 ≥1 点但不满窗时沿用既有的"补一个前置点、给首个位移供
     真实 dt"语义（该路径行为逐位不变）；
  2. **不补造帧**：只搬运真实观测 `(tick, pos)`，缺失帧既不生成点、也不当作静止
     （#39 的口径不变）；未观测信念不进历史；
  3. **有效期 / 长间隔**：配对时间差 `dt > interval_limit(window)` 时按 `limit/dt`
     **降权**（不丢弃、不冒充高置信），并把 `confidence` / `max_obs_dt` /
     `long_interval_pairs` / `stale_ticks` / `evidence` 做成**可读读数**
     （`motion_detail()`）⇒ 长间隔不是无脑外推，而是"有读数的低置信采纳"；
  4. `motion()` 的四元组契约 `(persistence, speed, samples, direction)` **不变**；
     `samples` 仍只数**窗口内有真实样本参与**的位移对（#148 的"新证据计数"口径：
     窗口内没有该实体时 `samples` 仍为 0，而速度取自"上一次真实观测对"）。

口径写成数据（不写死在调用点）：`MOTION_SCHEMA` / `INTERVAL_LIMIT_FACTOR` /
`MIN_INTERVAL_WINDOW` / `interval_limit()` 都可读回，`motion_detail()` 里
`interval_limit` 与 `confidence` 成对出现。
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

#: 运动读数口径标识（下游据此判读 `motion_detail()` 的字段语义）
MOTION_SCHEMA = "wm.motion-confidence/1"
#: 有效期因子：`dt > interval_limit` 的配对按 `limit/dt` 降权（不丢证据，只降置信）
INTERVAL_LIMIT_FACTOR = 3
#: 有效期窗口下限（tick）：窗口再小也不把正常间隔误判成"长间隔"
MIN_INTERVAL_WINDOW = 4


def interval_limit(window) -> int:
    """配对时间差的**有效期**（tick）= `max(window, MIN_INTERVAL_WINDOW) × FACTOR`。

    超过它不算"无效证据"：仍计入均值，但按 `limit/dt` 降权，并在
    `motion_detail()["confidence"]` 里如实报出。
    """
    return max(int(window or 0), MIN_INTERVAL_WINDOW) * INTERVAL_LIMIT_FACTOR


def _finite_pos(pos) -> bool:
    """观测坐标是否**三轴有限**（#418-F3：NaN/±Inf 一律不算可用证据）。

    为何在这里挡：非有限坐标会经 `math.hypot` 传染成 NaN/Inf 速度，并被下游
    `learn()` 原样写进 `per_entity["speed_est"]` 与 `_motion_estimates`；
    本笔新增的 `confidence` / `max_obs_dt` 读数同样不能把它报成合法读数。
    非法/不可解包的输入 ⇒ False（**拒绝**，不是静默放过）。
    """
    try:
        values = list(pos)
        return len(values) == 3 and all(math.isfinite(float(v)) for v in values)
    except (TypeError, ValueError, IndexError, OverflowError):
        return False


def _nonfinite_points(points) -> int:
    """`(tick, pos)` 序列里坐标非有限的观测**点数**（可读读数用，不静默丢弃）。"""
    return sum(1 for (_tick, pos) in points if not _finite_pos(pos))


def _nonfinite_pairs(points) -> int:
    """因端点为非有限坐标而被**拒绝**的位移对数（可读读数用）。"""
    count = 0
    for (t0, p0), (t1, p1) in zip(points, points[1:]):
        if t1 - t0 <= 0:
            continue
        if not (_finite_pos(p0) and _finite_pos(p1)):
            count += 1
    return count


def observed_positions(history, eid, window=None, nested=False):
    """Return actual observations; include one context point before a tick window.

    Missing frames are not zero-velocity measurements. The preceding real
    observation supplies dt for the first displacement in the window.

    #418-F1：回填判据补一条"**先保证能构成一对位移**"——旧式只在"窗口缺帧"时回填，
    于是 `window=1` 且该帧正好观测到时（`points=1`、窗口也是 1 帧）不回填，
    只有 1 个点 ⇒ 构不成位移对 ⇒ 估计为 None（公开 helper `estimate_motion(...,1)`
    受影响）。窗口内 0 个观测时保持原样（**不**回填）——那条路径由
    `ObservationWindow._context_points` 负责，且 `_traj` 等消费方依赖"窗口内观测"
    的原义，不在此处改变语义。
    #1116：`window<=0` 曾直接 `IndexError`（`start = len(history) - window` 越界，
    `history[start]` 崩）——与 `ObservationWindow.__init__` 同一套语义的两个入口，
    一个有保护一个没有。现所有入口统一 `max(1, int(window))` 钳制（`window=None`
    保留「全历史」语义不变）。
    """
    if window is not None:
        window = max(1, int(window))
    start = max(0, len(history) - window) if window is not None else 0
    points = []
    for rec in history[start:]:
        cur = rec["entities"].get(eid)
        if cur is not None:
            points.append((rec["tick"], cur["pos"] if nested else cur))
    need = 1 if (points and start and len(points) < len(history) - start) else 0
    if points and start:
        need = max(need, 2 - len(points))          # 保证能构成一对位移（F1）
    kept = []
    for index in range(start - 1, -1, -1):
        if len(kept) >= need:
            break
        rec = history[index]
        cur = rec["entities"].get(eid)
        if cur is not None:
            kept.append((rec["tick"], cur["pos"] if nested else cur))
    return kept[::-1] + points


def estimate_motion(history, eid, window, nested=False):
    """Return (persistence, speed, samples, last moving direction), or None."""
    points = observed_positions(history, eid, window, nested)
    if window is not None:
        window = max(1, int(window))           # #1116：与 observed_positions 同钳制
    start = max(0, len(history) - window) if window is not None else 0
    # #1116：start 越界（钳制前形态）或 history 空时 w_start=None——窗口起点未知，
    # `_motion_rows` 不再把窗口外位移对计入 samples（见该函数 #1116 注）。
    w_start = history[start]["tick"] if (history and start < len(history)) else None
    return _estimate_points(points, w_start, interval_limit(window))


def _motion_rows(points, window_start=None, limit=None):
    """真实 `(tick, pos)` 序列 → 运动读数 dict；无可用位移 ⇒ None。

    · `w = 1` 若 `dt <= limit`，否则 `limit/dt`（**长间隔降权**，不丢弃证据）；
    · `speed` = Σ(w·|v|)/Σw、`persistence` = |Σ(w·û)|/Σw —— `w ≡ 1` 时与旧式
      （未加权的算术平均）**逐位相同**，故所有"间隔都在有效期内"的既有路径行为不变；
    · `samples` = **后端点落在窗口内**（`t1 >= window_start`）的位移对数；窗口内
      没有该实体时仍为 0（#148 新证据计数口径），此时速度来自"上一次真实观测对"；
    · `direction` = 最后一个非零位移的单位方向（旧口径不变）；
    · **非有限坐标（NaN/±Inf）的位移对一律拒绝**并计数（`nonfinite_pairs_dropped`）
      —— 否则 `math.hypot` 会把 NaN/Inf 传染进 `speed`/`persistence`/`confidence`
      （#418-F3）；全部被拒 ⇒ 返回 None（下游回落已学估计/默认，不写坏值）。
    """
    lim = float(limit) if limit else None
    rows = []
    nonfinite = 0
    arithmetic = 0
    for (t0, p0), (t1, p1) in zip(points, points[1:]):
        dt = t1 - t0
        if dt <= 0:
            continue
        if not (_finite_pos(p0) and _finite_pos(p1)):
            nonfinite += 1
            continue
        try:
            dx = (p1[0] - p0[0]) / dt
            dz = (p1[2] - p0[2]) / dt
            # 差先溢出但速度仍可表示时，先除真实间隔再相减。
            if not math.isfinite(dx):
                dx = p1[0] / dt - p0[0] / dt
            if not math.isfinite(dz):
                dz = p1[2] / dt - p0[2] / dt
            length = math.hypot(dx, dz)
            weight = 1.0 if (lim is None or dt <= lim) else (lim / dt)
        except (TypeError, ValueError, IndexError, OverflowError):
            arithmetic += 1
            continue
        if not all(math.isfinite(v) for v in (dx, dz, length, weight)) or weight <= 0.0:
            arithmetic += 1
            continue
        rows.append((dt, dx, dz, length, weight,
                     # #1116：window_start 未知（None）时窗口外/内无从判定 ⇒
                     # 不计 samples（旧实现 `window_start is None or` 全放行，
                     # 窗口外位移对混进 samples 计数）。
                     window_start is not None and t1 >= window_start))
    return rows, nonfinite, arithmetic


def _estimate_detail(points, window_start=None, limit=None) -> Optional[Dict]:
    """保持三参估计契约；行构造另返算术拒绝计数，不改变无估计时的 None。"""
    rows, nonfinite, arithmetic = _motion_rows(points, window_start, limit)
    lim = float(limit) if limit else None
    if not rows:
        return None
    den = sum(r[4] for r in rows)
    speed = sum(r[4] * r[3] for r in rows) / den
    if not math.isfinite(speed):
        peak = max(r[3] for r in rows)
        ratio = math.fsum(r[4] * (r[3] / peak) for r in rows) / den
        speed = peak * min(1.0, ratio)
    ux = uz = uw = 0.0
    direction = None
    for (_dt, dx, dz, length, weight, _in_window) in rows:
        if length <= 1e-6:
            continue
        ux += weight * dx / length
        uz += weight * dz / length
        uw += weight
        direction = (dx / length, 0.0, dz / length)
    return {
        "persistence": round((math.hypot(ux, uz) / uw) if uw > 0 else 0.0, 3),
        "speed": round(speed, 3),
        "samples": sum(1 for r in rows if r[5]),
        "direction": direction,
        "dts": [r[0] for r in rows],
        "max_obs_dt": max(r[0] for r in rows),
        "long_interval_pairs": sum(1 for r in rows if lim is not None and r[0] > lim),
        "confidence": round(den / len(rows), 6),
        "interval_limit": lim,
        "nonfinite_pairs_dropped": nonfinite,
        "arithmetic_pairs_dropped": arithmetic,
    }


def _estimate_points(points, window_start=None, limit=None):
    """Return (persistence, speed, samples, last moving direction), or None."""
    detail = _estimate_detail(points, window_start, limit)
    if detail is None:
        return None
    return (detail["persistence"], detail["speed"], detail["samples"],
            detail["direction"])


class ObservationWindow:
    """A shared, one-pass view of the real observations used by inference.

    Normalize only the active window. Observation masks reject pairs without
    joint evidence before consulting older history. Each remaining pair
    shares one context lookup and traversal for both directions. This view
    lives for one learn/infer_patterns call; there is no cross-tick cache.
    """

    def __init__(self, history, window, nested=False):
        self.history = history
        # #1116：与 estimate_motion 同钳制（window<=0 曾使 start 越界 IndexError）
        window = max(1, int(window)) if window is not None else None
        self.start = max(0, len(history) - window) if window is not None else 0
        self.nested = nested
        self.frames = []
        self.points = {}
        self.masks = {}
        self._estimates = {}
        self._motions = {}
        self._details = {}
        self._pairs = {}
        #: 窗口首帧 tick（判"新证据"用）与最后一帧 tick（判陈旧度用）
        self.window_start = (history[self.start]['tick']
                             if self.start < len(history) else None)
        self.last_tick = history[-1]['tick'] if history else None
        #: 配对有效期（长间隔按 limit/dt 降权，读数见 motion_detail）
        self.interval_limit = interval_limit(window)
        for index, rec in enumerate(history[self.start:]):
            frame = {eid: cur['pos'] if nested else cur
                     for eid, cur in rec['entities'].items()}
            self.frames.append(frame)
            for eid, pos in frame.items():
                self.points.setdefault(eid, []).append((rec['tick'], pos))
                self.masks[eid] = self.masks.get(eid, 0) | (1 << index)

    def _context_points(self, eid, points):
        """按需从**更早历史**取真实观测补齐（返回 `(points, 回填点数)`）。

        #418：旧实现 `if points and self.start and len(points) < len(self.frames)`
        只在一个条件下回填、且只回填 1 个点：
          · 窗口内 0 个观测（长期遮挡 / 观测周期 > 窗口）⇒ 整段跳过 ⇒ 返回 None
            ⇒ 下游回落默认 speed=0.3（丢弃上一次真实观测——本件要修的主症状）；
          · 只回填 1 点 ⇒ 0 点场景下仍不足 2 点，构不成位移。
        现按需回填，判据分两层：
          ① **先保证能构成一对位移**：`need = max(0, 2 - len(points))`。
             ★这是复核 F1 的靶：旧判据按"窗口缺了几帧"判，`window=1` 且该帧正好
             观测到时 `len(points)=1`、`len(frames)=1` ⇒ 旧式 `1 < 1` 为假 ⇒ 不回填
             ⇒ 只有 1 点 ⇒ 构不成位移对 ⇒ 回落默认 0.3——而历史里明明还有更早的真实观测；
          ② 再沿用既有语义：窗口内 ≥1 点但**不满窗** ⇒ 至少补 1 个前置点，给窗口内
             首个位移供真实 dt（该路径行为与旧实现逐位相同）。
        只搬真实观测，不补造帧；窗内 0 点时最多补 2 个（构成一对）。
        """
        if not self.start:
            return points, 0
        # #1117：need 按窗口内**有限**点数算——非有限读数不是可用证据
        # （docstring「不进估计、也不静默」）；旧实现 len(points) 把含 NaN
        # 的观测也算数 ⇒ 该补的上下文不补、evidence/window_points 读数失真。
        n_finite = sum(1 for _t, p in points if _finite_pos(p))
        need = max(0, 2 - n_finite)
        if n_finite and n_finite < len(self.frames):
            need = max(need, 1)
        if need <= 0:
            return points, 0
        kept = []
        for index in range(self.start - 1, -1, -1):
            if len(kept) >= need:
                break
            cur = self.history[index]['entities'].get(eid)
            if cur is not None:
                kept.append((self.history[index]['tick'],
                             cur['pos'] if self.nested else cur))
        return kept[::-1] + list(points), len(kept)

    def _motion_estimate(self, eid):
        """Resolve real context and numeric estimates once within this view."""
        if eid not in self._estimates:
            points, n_context = self._context_points(eid, self.points.get(eid, []))
            detail = _estimate_detail(points, self.window_start, self.interval_limit)
            self._estimates[eid] = points, n_context, detail
        return self._estimates[eid]

    def motion(self, eid):
        if eid not in self._motions:
            _points, _context, detail = self._motion_estimate(eid)
            self._motions[eid] = (None if detail is None else
                                 (detail["persistence"], detail["speed"],
                                  detail["samples"], detail["direction"]))
        return self._motions[eid]

    def motion_detail(self, eid) -> Dict:
        """#418：估计的**可读读数**（证据来源 / 间隔 / 置信，全在数据里，不静默）。

        字段：`schema` 口径标识 · `estimate`（(persistence, speed, samples,
        direction) 或 None）· `window_points` / `context_points`（证据来源计数）
        · `evidence` ∈ {window, window+context, context_only} · `dts`/`max_obs_dt`
        · `long_interval_pairs` · `confidence`（加权占比 ∈(0,1]，长间隔越大越小）
        · `interval_limit` · `last_obs_tick` / `stale_ticks`（陈旧度）· `adopted`
        · `nonfinite_points` / `nonfinite_pairs_dropped`（#418-F3：NaN/±Inf 观测点数
          与被拒的位移对数——非有限读数**不进估计、也不静默**）
        · `reason`（估计不出来时说明为什么：`nonfinite_observation` /
          `insufficient_real_observations`）。
        """
        if eid not in self._details:
            window_points = self.points.get(eid, [])
            # #1117：`window_points` 读数与证据判定同改**有限口径**——非有限
            # 读数不是可用证据（与 need 的有限口径一致；`nonfinite_points`
            # 字段照旧把坏读数如实带出，不静默）。
            window_finite = sum(1 for _t, p in window_points if _finite_pos(p))
            points, n_context, detail = self._motion_estimate(eid)
            nonfinite = _nonfinite_points(points)
            last_obs = points[-1][0] if points else None
            stale = (self.last_tick - last_obs
                     if (last_obs is not None and self.last_tick is not None)
                     else None)
            out = {
                "schema": MOTION_SCHEMA, "eid": eid,
                "window_points": window_finite,
                "context_points": n_context,
                "interval_limit": self.interval_limit,
                "last_obs_tick": last_obs, "stale_ticks": stale,
                "estimate": None, "dts": [], "max_obs_dt": None,
                "long_interval_pairs": 0, "confidence": 0.0, "adopted": False,
                "evidence": "none",
                "nonfinite_points": nonfinite,
                "nonfinite_pairs_dropped": _nonfinite_pairs(points),
            }
            arithmetic = (detail.get("arithmetic_pairs_dropped", 0) if detail is not None
                          else _motion_rows(points, self.window_start, self.interval_limit)[2])
            if arithmetic:
                out["arithmetic_pairs_dropped"] = arithmetic
            if detail is None:
                # 全无可采纳位移：区分"读数坏了"与"真实样本不足"（不静默、可判读）
                out["reason"] = ("invalid_motion_arithmetic" if arithmetic else
                                 "nonfinite_observation" if nonfinite
                                 else "insufficient_real_observations")
            else:
                out.update({
                    "estimate": (detail["persistence"], detail["speed"],
                                 detail["samples"], detail["direction"]),
                    "dts": detail["dts"], "max_obs_dt": detail["max_obs_dt"],
                    "long_interval_pairs": detail["long_interval_pairs"],
                    "confidence": detail["confidence"],
                    "adopted": True,
                })
                if window_finite >= 2:
                    out["evidence"] = "window"
                elif window_finite:
                    out["evidence"] = "window+context"
                else:
                    out["evidence"] = "context_only"
            # Both accessors use one estimate and context lookup per view.
            # Keep the tuple separate from the public mutable diagnostic dict:
            # callers changing that dict must not change motion()'s result.
            self._motions[eid] = out["estimate"]
            self._details[eid] = out
        return self._details[eid]

    def pair(self, a, b):
        """Return (mean cosine, directional samples, observed intervals).

        Stationary/coincident observations are measured intervals, but do not
        support a direction. Missing observations contribute neither count.
        """
        common = self.masks.get(a, 0) & self.masks.get(b, 0)
        if not common:
            return 0.0, 0, 0
        if (a, b) not in self._pairs:
            previous = None
            if self.start and common != (1 << len(self.frames)) - 1:
                for index in range(self.start - 1, -1, -1):
                    entities = self.history[index]['entities']
                    if a in entities and b in entities:
                        previous = ((entities[a]['pos'], entities[b]['pos']) if self.nested
                                    else (entities[a], entities[b]))
                        break
            scores_a, scores_b, intervals = [], [], 0
            for index, frame in enumerate(self.frames):
                if not (common & (1 << index)):
                    continue
                pa, pb = frame[a], frame[b]
                if previous is not None:
                    # #1115：每一对坐标先过 `_finite_pos`（与 `_motion_rows`
                    # 同口径——非有限/不可解包/**超出 float 的超大 int** 一律
                    # 拒绝；裸 `math.isfinite(v)` 对 10**400 会自己抛
                    # OverflowError）——不计 intervals、不进 scores，且**不
                    # 更新链**（与「缺观测不计」一致：后续有限对的位移仍以
                    # 上一有限帧为参照，NaN 不污染整条链）。
                    if not (_finite_pos(pa) and _finite_pos(pb)):
                        continue
                    intervals += 1
                    tx, tz = pb[0] - pa[0], pb[2] - pa[2]
                    length = math.hypot(tx, tz)
                    if length > 1e-6:
                        for pos, old, dx_target, dz_target, scores in (
                                (pa, previous[0], tx, tz, scores_a),
                                (pb, previous[1], -tx, -tz, scores_b)):
                            dx, dz = pos[0] - old[0], pos[2] - old[2]
                            movement = math.hypot(dx, dz)
                            if movement > 1e-6:
                                scores.append((dx * dx_target + dz * dz_target) / (movement * length))
                previous = pa, pb
            for key, scores in (((a, b), scores_a), ((b, a), scores_b)):
                mean = round(sum(scores) / len(scores), 3) if scores else 0.0
                self._pairs[key] = mean, len(scores), intervals
        return self._pairs[a, b]


def recent_direction(history, eid, nested=False) -> Optional[Tuple[float, float, float]]:
    """Latest nonzero displacement, without fabricating missing observations.

    #1117：坏读数（非有限坐标）按「**跳过该帧**」处理（与 missing 帧同路径，
    不重置 `newer`）——docstring 的「不补造缺失帧」是说缺失帧不补造，不是「坏
    读数清空历史」：改前形态一次坏读数丢掉两侧有限观测构成的真实方向（NaN 夹
    心场景 `None`，而同构的缺失帧场景正常返回方向）。`_motion_rows` 对坏读数
    同为「只丢这一对，其余对照常」——两处口径就此一致。
    """
    newer = None
    for rec in reversed(history):
        cur = rec["entities"].get(eid)
        if cur is None:
            continue
        pos = cur["pos"] if nested else cur
        if not _finite_pos(pos):
            continue                     # #1117：跳过，不重置 newer
        if newer is not None:
            try:
                dx, dz = newer[0] - pos[0], newer[2] - pos[2]
                length = math.hypot(dx, dz)
            except (TypeError, ValueError, IndexError, OverflowError):
                newer = pos
                continue
            if math.isfinite(length) and length > 1e-6:
                return dx / length, 0.0, dz / length
        newer = pos
    return None
