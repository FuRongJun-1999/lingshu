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
    """
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


def _clamp_window(window, history):
    """把 `window` 钳成 `None`（无窗口）或 `1..len(history)`（#1116）。

    为什么需要：`window` 是公开入参（`WorldLearner.learn` / `infer_patterns`
    的透传面），`window=0` 与负值是合法整数；旧实现里
    `start = max(0, len(history) - window)` 在 `window<=0` 时得到
    `start >= len(history)`，随后 `history[start]` 越界（IndexError）。
    另外 `window=0` 会使 `window_start=None`，让「窗口外」的位移对
    满足 `t1 >= window_start` 恒真 ⇒ 被记成 `samples=1`，绕过
    「不得用上下文证据覆盖已学估计」的闸（world_model.py:338-341）。
    这里统一成：`None` 原样，其余钳到 `[0, len(history)]` 且 `0 -> None`。
    """
    if window is None:
        return None
    try:
        n = int(window)
    except (TypeError, ValueError):
        return 0                         # 非法类型 ⇒ 空窗口（不静默取全历史）
    if n <= 0:
        # 0/负 ⇒ **空窗口**（不是「无窗口」）：窗口内一帧都看不见 ⇒ samples=0
        # ⇒ 按 world_model.py:338-341 保留已学估计，而不是用全历史当新证据
        # 覆盖它（旧实现 window=0 时 window_start=None 使「窗口外」位移对它
        # 恒真，speed_est 从 1.0 被覆盖成 10.0）。
        return 0
    return min(n, len(history))


def estimate_motion(history, eid, window, nested=False):
    """Return (persistence, speed, samples, last moving direction), or None.

    #1116：`window <= 0` 是合法整数入参（`world_learner.learn(window=)`
    / `infer_patterns(window=)` 直接透传），旧实现 `start = max(0, len-window)`
    在 window=0/负值时 `start >= len(history)` ⇒ `history[start]` IndexError。
    钳到 `[0, len(history)]`——0 视为「无窗口」（与 `window=None` 同路径）。
    """
    limit = _clamp_window(window, history)
    points = observed_positions(history, eid, limit, nested)
    if limit is None:
        start, w_start = 0, (history[0]["tick"] if history else None)
    elif limit <= 0:
        # 空窗口：window_start 取最后一帧的下一 tick，使任何位移对都不满足
        # `t1 >= window_start` ⇒ samples=0（保留已学估计）。
        w_start = history[-1]["tick"] + 1 if history else None
        return _estimate_points(points, w_start, interval_limit(limit))
    else:
        start = max(0, len(history) - limit)
        w_start = history[start]["tick"] if history else None
    return _estimate_points(points, w_start, interval_limit(limit))


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
                     window_start is None or t1 >= window_start))
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
        # #1116：同 `_clamp_window`——`window<=0` 会让 `start >= len(history)`
        # 并在 `_context_points` 的 `self.history[index]` 处 IndexError；
        # 且 `window_start=None` 会把窗口外位移对记成 samples=1。
        _w = _clamp_window(window, history)
        if _w is None:
            self.start = 0
        elif _w <= 0:
            self.start = len(history)          # 空窗口：无帧可看
        else:
            self.start = max(0, len(history) - _w)
        self.nested = nested
        self.frames = []
        self.points = {}
        self.masks = {}
        self._estimates = {}
        self._motions = {}
        self._details = {}
        self._pairs = {}
        #: 窗口首帧 tick（判"新证据"用）与最后一帧 tick（判陈旧度用）
        # #1116：空窗口时 `window_start` 不能是 None——`_motion_rows` 的
        # `window_start is None or t1 >= window_start` 会因前半为真而放行全部
        # 位移对（把全历史当新证据）。取「最后一帧 + 1」使判据恒假。
        if _w is not None and _w <= 0:
            self.window_start = history[-1]['tick'] + 1 if history else None
        else:
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
        # #1117②：`need` 只按**有限**点数算——旧实现把窗口内的非有限观测
        # 也算进 `points`，于是「窗口内 [有限, NaN] + 更早还有真实观测」时
        # 认为已够 2 点、不回填（该补不补），且 `window_points`/`evidence`
        # 把它报成「窗口内自足」。与 `motion_detail` docstring「非有限读数
        # 不进估计、也不静默」及 `_nonfinite_pairs` 的口径一致。
        finite_n = sum(1 for _t, p in points if _finite_pos(p))
        need = max(0, 2 - finite_n)
        if finite_n and len(points) < len(self.frames):
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
            points, n_context, detail = self._motion_estimate(eid)
            nonfinite = _nonfinite_points(points)
            last_obs = points[-1][0] if points else None
            stale = (self.last_tick - last_obs
                     if (last_obs is not None and self.last_tick is not None)
                     else None)
            out = {
                "schema": MOTION_SCHEMA, "eid": eid,
                "window_points": len(window_points),
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
                if len(window_points) >= 2:
                    out["evidence"] = "window"
                elif window_points:
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
                # #1115：非有限/不可解包坐标按 `_finite_pos` 同口径**拒绝**——
                # 旧实现把它们当作「已测位移」计入 `intervals`/`samples`：
                #   · 双实体场景下 samples>=3 且 directional==0 成立，下游
                #     `world_model.infer_patterns` / `world_learner.learn` 去索引
                #     `motion(a)[3]`（此时 motion() 正确返回 None）⇒ TypeError；
                #   · 1e154 量级有限坐标使 `dx*dx_t + dz*dz_t` 溢出成 inf、再除
                #     得 NaN ⇒ `directional>=3` 仍成立、`sufficient` 为真，但
                #     `abs(nan) > abs(best)` 恒 False ⇒ best=None ⇒ 已学关系被
                #     **静默移除**（违反 world_model「只有测得静止才能撤回关系」）。
                # 与 `_motion_rows`（:145）和 `recent_direction`（:412）的既有
                # 闸同款：坏读数既不进 samples，也不算「测得静止」。
                if not (_finite_pos(pa) and _finite_pos(pb)):
                    self._nonfinite_pairs = getattr(self, "_nonfinite_pairs", 0) + 1
                    previous = None
                    continue
                if previous is not None and not (_finite_pos(previous[0])
                                                 and _finite_pos(previous[1])):
                    previous = pa, pb
                    continue
                if previous is not None:
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
                                dot = dx * dx_target + dz * dz_target
                                denom = movement * length
                                # #1115：乘积溢出得 inf 时 dot/denom 为 nan 或
                                # inf——NaN 分数会静默抹掉已学关系（见上）。按
                                # `_motion_rows` 的同款手法先归一化再点乘。
                                if not math.isfinite(dot) or not math.isfinite(denom):
                                    ux, uz = dx / movement, dz / movement
                                    vx, vz = dx_target / length, dz_target / length
                                    dot, denom = ux * vx + uz * vz, 1.0
                                scores.append(dot / denom)
                previous = pa, pb
            for key, scores in (((a, b), scores_a), ((b, a), scores_b)):
                mean = round(sum(scores) / len(scores), 3) if scores else 0.0
                self._pairs[key] = mean, len(scores), intervals
        return self._pairs[a, b]


def recent_direction(history, eid, nested=False) -> Optional[Tuple[float, float, float]]:
    """Latest nonzero displacement, without fabricating missing observations."""
    newer = None
    for rec in reversed(history):
        cur = rec["entities"].get(eid)
        if cur is None:
            continue
        pos = cur["pos"] if nested else cur
        if not _finite_pos(pos):
            # #1117③：坏读数按「该帧没有可用观测」处理（与 `cur is None` 同路径）
            # ——旧实现把 `newer` 重置成 None，于是 `[有限, NaN, 有限]` 三帧里
            # 两侧真实有限观测永远构不成配对，方向回落 None（实测）；同模块
            # `_motion_rows` 的口径是「只丢这一对，其余对照常」。
            continue
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
