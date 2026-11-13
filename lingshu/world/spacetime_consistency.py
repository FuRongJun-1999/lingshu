# -*- coding: utf-8 -*-
"""spacetime_consistency · 时空一致性验证（世界模型阶段2 · 里程碑2.4 · 阶段2收官）
============================================================================
核心（荣）：时空验证就是持续运行，保持一致性验证。达到这一点，那么实际上
世界模型就是自洽的了。

SpacetimeConsistency = SceneSimulator + 时空一致性验证闭环：
  - 持续运行：run(n) 每 tick 验证一步（预测下一状态 vs 实际 → 命中判定）
  - 滚动命中率：窗口内预测命中率跟踪（rolling hit rate，可调 window）
  - 一致性漂移检测：滚动命中率跌破阈值持续 K tick → 漂移事件（可审计）
  - 世界状态不变量：边界/落地/实体存在性每 tick 校验（保持一致性验证）
  - 自洽度报告：总体/滚动/分行为命中率 + 漂移事件 + 世界模型自洽判定

预测模型（白箱 · 确定性 · 零 LLM · D-005）：
  - seek/avoid/follow（确定性行为）：重放行为决策 → 精确位置预测（exact）
  - wander/flee（随机行为）：可达域预测（模型知道其随机性——预测实体
    遵守运动规则 speed/边界 → bounded 命中域 = speed*factor + pad）
  - 不变量校验：实体始终在界内且不下穿地面（世界保持自身一致性）

设计参考：
  - world_server.verify_prediction（预测 vs 实际 → 命中判定，阈值 0.5）
  - 智能论 3.4：2.10.1 第六感预测 · D-006 动态校准（预测回填 → 校准）
  - 时空验证 = 持续运行 + 一致性验证（达到此点 → 世界模型自洽）

纯标准库 · 零外部依赖（D-005）
"""
from __future__ import annotations

import logging as _logging

_logger = _logging.getLogger(__name__)

import math
from collections import deque
from typing import Dict, List, Optional, Tuple

from .coord_contract import (apply_move,                     # noqa: E402
                             NF_REJECT_ROUND_ALL)             # 第 4 轮 #4

#: 每 tick 验证记录（预测历史 `_results`）的环形缓冲长度（#427：此前 `_results`
#: 每 tick 追加、永不裁剪，`overall_hit_rate`/`per_behavior_rates` 又每次全量
#: 遍历它 ⇒ 长期运行内存与单次报告耗时都随 tick 线性增长）。
#: 上限值经验标定（追不到理论章节；与 `scene_simulator.HISTORY_MAXLEN` 同量级）：
#: 取 5000，使 5 实体场景下常驻记录 ≈ 5000 × 5 × 400 B ≈ 10 MB 以内。
HISTORY_MAXLEN = 5000

# 3D 场景模拟器实现已迁移至 AEIS——大脑保留接口，缺失时 SceneSimulator=None
try:
    from .scene_simulator import SceneSimulator
except Exception as _e:
    _logger.debug("#170 3D 场景模拟器已迁移 AEIS，缺失回落 SceneSimulator=None：%s", _e)
    try:
        from .scene_simulator import SceneSimulator
    except Exception as _e2:
        _logger.debug("#170 二次回落仍缺失，SceneSimulator=None：%s", _e2)
        SceneSimulator = None


class SpacetimeConsistency:
    """时空一致性验证器：封装 SceneSimulator，持续运行 + 一致性验证。

    能力：
      - create_scene/add_entity/add_path：场景构建（透传 SceneSimulator）
      - run(n)/step_verified()：持续运行——每步预测下一状态 vs 实际
      - rolling_hit_rate()：滚动命中率跟踪
      - drift_events()/drift_active()：一致性漂移检测
      - consistency_report()/self_consistent()：自洽度报告与世界模型自洽判定
      - teleport()：外部事件瞬移（注入不一致，验证漂移检测——世界模型
        无法解释的变更会被识别为一致性破坏）
      - prediction_history()：预测验证历史（可审计）
    """

    def __init__(self, size: int = 24, ground_level: int = 1,
                 seed: int = 42,
                 window: int = 20, hit_threshold: float = 0.5,
                 drift_rate: float = 0.7, drift_ticks: int = 5,
                 consistent_rate: float = 0.85,
                 min_consistent_ticks: int = 50,
                 wander_bound_factor: float = 1.5,
                 wander_bound_pad: float = 0.2,
                 max_results: int = HISTORY_MAXLEN):
        # #25：seed 轴透传（此前 __init__ 无 seed 参数，构造 SceneSimulator 时
        # 丢弃种子 ⇒ 场景演化在结构上不可做 seed 维度消融）。缺省 42 = 与
        # `scene_simulator.py` 的历史硬编码逐位一致，默认行为不变。本类自身
        # 不产生随机数（随机性全部来自 `self.scene._rng`），故一个 seed 参数
        # 透传即覆盖整条随机轴；影子预测（`_predict_next`）不消耗 RNG。
        self.scene = SceneSimulator(size=size, ground_level=ground_level,
                                    seed=seed)
        # #2419：三个 rate 参数此前裸 float() 零值域闸——`hit_threshold=nan`
        # 使逐 tick `rate >= threshold` 恒 False ⇒ 同种子同场景 overall
        # 1.0→0.0、假漂移告警（实测）；±inf 同理，负阈值则恒 True（假自洽）。
        # 同段的离散参数已有 max(1, int(…)) 闸，rate 族补齐同一维口径。
        for _name, _raw in (("hit_threshold", hit_threshold),
                            ("drift_rate", drift_rate),
                            ("consistent_rate", consistent_rate)):
            _f = float(_raw)
            if not (0.0 <= _f <= 1.0):
                raise ValueError("%s 需为 [0,1] 内的有限数值，收到 %r"
                                 % (_name, _raw))

        # #2427：#2419 只收了 rate 三参数，同段的 wander_bound_* 与 ground_level
        # 仍裸 float()——`wander_bound_factor=inf` 使 `_reach_bound` 为 inf ⇒
        # 任意偏差都判命中（23 m/tick 瞬移被判 self_consistent，漂移检测恒不
        # 触发）；`nan` 被 `max(0.5, nan)` 吃成 0.5 ⇒ 正常 wander 全判失败（假
        # 漂移）。`ground_level=nan/-inf` 使 `y < gnd + 0.4` 恒假 ⇒ below_ground
        # 不变量整条变死代码（地下 100 m 的世界判 self_consistent）。
        for _name, _raw in (("wander_bound_factor", wander_bound_factor),
                            ("wander_bound_pad", wander_bound_pad)):
            if isinstance(_raw, bool) or not isinstance(_raw, (int, float))                     or not math.isfinite(float(_raw)) or float(_raw) < 0.0:
                raise ValueError("%s 需为有限非负数，收到 %r" % (_name, _raw))
        if isinstance(ground_level, bool) or not isinstance(ground_level, (int, float))                 or not math.isfinite(float(ground_level)):
            raise ValueError("ground_level 需为有限数值，收到 %r" % (ground_level,))
        self.window = max(1, int(window))
        self.hit_threshold = float(hit_threshold)
        self.drift_rate = float(drift_rate)
        self.drift_ticks = max(1, int(drift_ticks))
        self.consistent_rate = float(consistent_rate)
        self.min_consistent_ticks = max(1, int(min_consistent_ticks))
        self.wander_bound_factor = float(wander_bound_factor)
        self.wander_bound_pad = float(wander_bound_pad)
        self.max_results = max(1, int(max_results))
        # 验证状态
        self._results: deque = deque(maxlen=self.max_results)  # 每 tick 验证记录（有界）
        self._rolling: deque = deque(maxlen=self.window)  # 每 tick 命中率窗口（None=无样本）
        # 终身累积聚合（#427：`_results` 裁剪后仍保持总体/分行为口径不变）
        self._agg_hits = 0
        self._agg_outcomes = 0
        self._agg_behavior: Dict[str, Dict[str, int]] = {}
        self._agg_mode: Dict[str, Dict[str, int]] = {
            "exact": {"outcomes": 0, "hits": 0},
            "stochastic": {"outcomes": 0, "hits": 0}}
        # 逐实体窗口命中（#9：全体平均看不到「部分实体长期漂移」）
        self._entity_hits: Dict[str, deque] = {}
        self._drift_events: List[Dict] = []
        self._active_drift: Optional[Dict] = None
        self._low_streak = 0                  # 连续低于漂移阈值的 tick 数
        #: #716：不变量违例的**当前状态**窗口（每 tick 追加 0/1）。
        #: 与 `_rolling` / `_entity_hits` 共用 `self.window` 口径 ⇒ 三者都读
        #: 「最近 window 个 tick」。判定读它；累计值 `_invariant_violations`
        #: 只作审计（见 `consistency_report` docstring）。
        self._inv_window: deque = deque(maxlen=self.window)
        self._invariant_violations = 0
        self._pending_events: List[Tuple[str, Tuple[float, float, float]]] = []
        self.tick_count = 0

    # ---- 场景构建（透传 SceneSimulator）----

    def create_scene(self, trees: int = 4, water: bool = True) -> Dict:
        return self.scene.create_scene(trees=trees, water=water)

    def add_entity(self, category: str, behavior: str = "wander",
                   pos: Tuple[float, float, float] = (2, 1.5, 2),
                   speed: float = 0.3, goal: str = "") -> str:
        return self.scene.add_entity(category=category, behavior=behavior,
                                     pos=pos, speed=speed, goal=goal)

    def add_path(self, path_id: str,
                 points: List[Tuple[float, float, float]]) -> None:
        self.scene.add_path(path_id, points)

    def scene_state(self) -> Dict:
        return self.scene.scene_state()

    def behavior_log(self, limit: int = 30) -> List[Dict]:
        return self.scene.behavior_log(limit=limit)

    def evolution(self) -> List[Dict]:
        return self.scene.evolution()

    # ---- 预测模型（白箱 · 确定性）----

    def _is_deterministic(self, e) -> bool:
        """行为是否可精确预测（不消耗 RNG 的确定性分支）。

        #398：follow 要求**路径非空**——世界侧 `_decide` 的 `if path:` 对空路径
        不成立（退化为随机游走），此前这里只判 `goal in paths` ⇒ 一个随机游走
        实体被当作 exact 判分（deterministic_rate 失真 + 假漂移）。
        """
        if e.behavior == "seek" and e.goal in self.scene.entities:
            return True
        if e.behavior == "avoid" and e.goal in self.scene.entities:
            return True
        if e.behavior == "follow" and self.scene.paths.get(e.goal):
            return True
        return False

    def _apply_move_at(self, pos: Tuple[float, float, float], speed: float,
                       d: Tuple[float, float, float],
                       dist: Optional[float] = None) -> Tuple[float, float, float]:
        """按行为方向从指定位置移动（与 SceneSimulator.step 完全同式：clamp+round2）。

        #249：`dist`（到目标的剩余距离）非 None 时步长按 `min(speed, dist)` 封顶
        ——与世界侧 seek/follow 的到达封顶同式，否则到达邻域预测会系统性地
        越过目标（影子与物理世界分叉）。
        #347：非有限坐标拒绝写入（clamp 表达式遇 NaN 会静默抬成边界值）。
        """
        # ⚠ 第 4 轮 #4：语义收归 `coord_contract.apply_move`，本站点的**两处差异**
        #   一律显式命名，不再藏在函数体里：
        #     · `step_cap=dist`      —— #249 步长按到目标的剩余距离封顶
        #     · `on_nonfinite=...`   —— #347 非有限坐标必须**拒绝**，不许被 clamp 静默抬成边界值
        return apply_move(pos, speed, d, self.scene.world.size,
                          step_cap=dist, on_nonfinite=NF_REJECT_ROUND_ALL)

    def _shadow_decide(self, e, shadow: Dict[str, Tuple[float, float, float]],
                       follow_target: Optional[Dict[str, int]] = None
                       ) -> Tuple[float, float, float]:
        """影子状态下的行为决策（与 SceneSimulator._decide 同式，读取影子位置）。

        世界 step 按实体插入序依次决策/行动——后决策的实体看到的是前序实体
        已移动后的位置。预测必须复刻该顺序语义，否则追逐链会产生 ~speed 的
        系统性偏差（世界比模型多一阶）。不消耗 RNG（仅确定性分支调用）。

        `follow_target`：follow 的"已提交目标路径点"表，由 `_predict_next` 以
        世界侧 `scene._follow_target` 的当前值初始化，并在影子重放中同样推进。
        否则影子每 tick 重算"最近点"，推进一格后会弹回原点，与世界实际巡逻
        轨迹分叉（follow 属确定性行为、走 exact 分支，必须逐 tick 对齐）。
        """
        bx, by, bz = shadow.get(e.id, e.pos)
        if e.behavior == "seek":
            target = self.scene.entities.get(e.goal)
            if target:
                t = shadow.get(e.goal, target.pos)
                dx, dz = t[0] - bx, t[2] - bz
                return self.scene._normalize(dx, 0, dz)
        elif e.behavior == "avoid":
            target = self.scene.entities.get(e.goal)
            if target:
                t = shadow.get(e.goal, target.pos)
                dx, dz = bx - t[0], bz - t[2]
                return self.scene._normalize(dx, 0, dz)
        elif e.behavior == "follow":
            path = self.scene.paths.get(e.goal)
            if path:
                fi = follow_target if follow_target is not None else {}
                idx = fi.get(e.id)
                if idx is None or not (0 <= idx < len(path)):
                    idx = min(range(len(path)),
                              key=lambda i: math.hypot(path[i][0] - bx,
                                                       path[i][2] - bz))
                tgt = path[idx]
                if math.hypot(tgt[0] - bx, tgt[2] - bz) <= max(e.speed, 1e-9):
                    idx = (idx + 1) % len(path)
                fi[e.id] = idx
                tgt = path[idx]
                return self.scene._normalize(tgt[0] - bx, 0, tgt[2] - bz)
        return (0.0, 0.0, 0.0)

    def _shadow_approach_dist(self, e, shadow: Dict[str, Tuple[float, float, float]],
                              follow_target: Dict[str, int]) -> Optional[float]:
        """影子状态下的剩余平面距离（#249：与 SceneSimulator._approach_distance 同式）。

        必须在 `_shadow_decide` **之后**调用（follow 的已提交目标点由它推进）。
        """
        bx, _, bz = shadow.get(e.id, e.pos)
        if e.behavior == "seek":
            t = shadow.get(e.goal)
            if t is not None:
                return math.hypot(t[0] - bx, t[2] - bz)
        elif e.behavior == "follow":
            path = self.scene.paths.get(e.goal)
            if path:
                idx = follow_target.get(e.id)
                if idx is not None and 0 <= idx < len(path):
                    tgt = path[idx]
                    return math.hypot(tgt[0] - bx, tgt[2] - bz)
        return None

    def _predict_next(self) -> Dict[str, Tuple[Tuple[float, float, float], str,
                                               str, str, float]]:
        """预测下一 tick 各实体位置（复刻世界顺序语义）。

        返回 {entity_id: (预测位置, mode, category, behavior, bound)}：
          - exact：确定性行为（seek/avoid/follow）——按实体插入序重放决策，
            影子位置随移动更新（后决策实体看到前序实体已移动）→ 精确命中
          - bounded：随机行为（wander/flee）——可达域预测（预测=当前位置，
            命中域=speed*factor+pad，模型知道自己随机性边界）
        注意：确定性分支不消耗 RNG（与 _decide 的随机分支隔离），不扰动实际演化。
        """
        shadow = {eid: tuple(e.pos) for eid, e in self.scene.entities.items()}
        # follow 的已提交目标索引以世界侧当前值为种子，影子重放中同步推进
        # ——否则影子每 tick 重算"最近点"，会与世界的巡逻轨迹分叉。
        follow_target: Dict[str, int] = dict(
            getattr(self.scene, "_follow_target", {}) or {})
        pred = {}
        for eid, e in self.scene.entities.items():
            exact, bound = self._exactness(e, pred)
            if exact:
                d = self._shadow_decide(e, shadow, follow_target)
                dist = self._shadow_approach_dist(e, shadow, follow_target)
                np_ = self._apply_move_at(shadow[eid], e.speed, d, dist)
                shadow[eid] = np_              # 已移动 → 后续实体决策可见
                pred[eid] = (np_, "exact", e.category, e.behavior, 0.0)
            else:
                pred[eid] = (shadow[eid], "bounded",
                             e.category, e.behavior, bound)
        return pred

    def _exactness(self, e, pred: Dict) -> Tuple[bool, float]:
        """精确性判定 + 不确定边界（依赖链传播）。

        确定性行为可精确预测，除非其追逐目标（seek/avoid 的 goal）在实体
        序中更早且是随机行为——目标位置不确定（≤目标可达域）→ 追逐结果
        同样不确定，降级为 bounded，命中域 = max(自身可达域, 目标可达域)
        + hit_threshold。follow（静态路径）始终精确。
        """
        if not self._is_deterministic(e):
            return False, self._reach_bound(e)
        if e.behavior in ("seek", "avoid"):
            g = self.scene.entities.get(e.goal)
            if g is not None and g.id in pred and pred[g.id][1] == "bounded":
                return False, max(self._reach_bound(e),
                                  self._reach_bound(g)) + self.hit_threshold
        return True, 0.0

    def _reach_bound(self, e) -> float:
        """随机行为可达域（模型知道自己的随机性边界）。"""
        return max(self.hit_threshold,
                   e.speed * self.wander_bound_factor + self.wander_bound_pad)

    # ---- 世界状态不变量校验 ----

    def _check_invariants(self) -> Tuple[bool, List[str]]:
        """世界状态不变量：实体在界内 · 不下穿地面 · 位置为有限数。"""
        issues = []
        size = self.scene.world.size
        gnd = float(self.scene.world.ground_level)
        if not math.isfinite(gnd):
            # #2427 纵深：直接改 `scene.world.ground_level` 也要拦——
            # 否则 `y < nan + 0.4` 恒假会让 below_ground 判据静默失效。
            issues.append("bad_ground_level:%r" % (gnd,))
        for eid, e in self.scene.entities.items():
            x, y, z = e.pos
            if not (0.5 <= x <= size - 0.5 and 0.5 <= z <= size - 0.5):
                issues.append("out_of_bounds:" + eid)
            if y < gnd + 0.4:
                issues.append("below_ground:" + eid)
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                issues.append("non_finite:" + eid)
        return (len(issues) == 0, issues)

    # ---- 持续运行 · 一致性验证闭环 ----

    def step_verified(self) -> Dict:
        """验证一步：预测下一状态 → 实际 step → 逐实体命中判定 → 更新统计。

        外部事件队列（teleport 排队）在预测之后、实际演化之前生效——
        世界发生模型无法解释的变更 → 预测与实际偏离 → 一致性漂移被检测。
        """
        pred = self._predict_next()
        # 应用排队的外部事件（模型未知，发生在 tick 内）
        for eid, pos in self._pending_events:
            e = self.scene.entities.get(eid)
            if e is not None:
                e.pos = pos
        self._pending_events.clear()
        self.scene.step(n=1)
        self.tick_count = self.scene.tick_count
        hits = 0
        outcomes = []
        for eid, (pp, mode, category, behavior, bound) in pred.items():
            e = self.scene.entities.get(eid)
            if e is None:
                # 实体消失（外部事件删除）→ 预测但未观测到 = 不一致
                outcomes.append({"entity": eid, "category": category,
                                 "behavior": behavior, "mode": mode,
                                 "predicted": list(pp), "actual": None,
                                 "distance": None, "hit": False,
                                 "missing": True})
                continue
            ap = tuple(round(v, 2) for v in e.pos)
            dist = math.sqrt(sum((pp[i] - ap[i]) ** 2 for i in range(3)))
            if mode == "exact":
                hit = dist < self.hit_threshold
            else:
                hit = dist < (bound if bound > 0.0 else self._reach_bound(e))
            if hit:
                hits += 1
            outcomes.append({"entity": eid, "category": e.category,
                             "behavior": e.behavior, "mode": mode,
                             "predicted": list(pp), "actual": list(ap),
                             "distance": round(dist, 4), "hit": hit,
                             "missing": False})
        total = len(outcomes)
        # #22：分母为 0（本 tick 无任何可比实体）＝无样本，不得回数值满分 1.0
        rate = round(hits / total, 4) if total else None
        self._rolling.append(rate)
        rolling = self._rolling_rate()
        # #427：终身累积聚合（_results 有界裁剪后，总体/分行为口径不随裁剪而变）
        self._agg_hits += hits
        self._agg_outcomes += total
        for o in outcomes:
            b = o["behavior"]
            s = self._agg_behavior.setdefault(b, {"outcomes": 0, "hits": 0})
            s["outcomes"] += 1
            if o["hit"]:
                s["hits"] += 1
            m = "exact" if o["mode"] == "exact" else "stochastic"
            self._agg_mode[m]["outcomes"] += 1
            if o["hit"]:
                self._agg_mode[m]["hits"] += 1
        # #9：逐实体窗口命中（漂移检测改看「最差实体」而非全体平均）
        for o in outcomes:
            d = self._entity_hits.setdefault(o["entity"], deque(maxlen=self.window))
            d.append(1 if o["hit"] else 0)
        for eid in list(self._entity_hits):      # 已离场实体不再参与最差判定
            if eid not in pred:
                del self._entity_hits[eid]
        # 不变量校验（世界保持自身一致）
        # #716：窗口记录「本 tick 是否违规」（0/1）——判定读窗口，累计值只作审计。
        #   旧实现只有终身累计计数，判定直接读它 ⇒ 一次瞬时违规（teleport /
        #   瞬时抖动 / 外部写入）之后世界即便完全自愈，`verdict` 也**永久**停在
        #   `invariant_violated`、`self_consistent()` 永久 False —— 「持续运行」
        #   框架下最严重的语义倒退（本条即报此）。
        inv_ok, inv_issues = self._check_invariants()
        self._inv_window.append(0 if inv_ok else 1)
        if not inv_ok:
            self._invariant_violations += 1
        # 一致性漂移检测（#9：以最差实体的窗口命中率为判据）
        worst = self._worst_entity_rate()
        self._update_drift(worst)
        record = {"tick": self.tick_count, "hits": hits, "total": total,
                  "rate": rate, "rolling": rolling,
                  "worst_entity_rate": worst,
                  "scored_rounds": sum(1 for r in self._rolling if r is not None),
                  "total_rounds": len(self._rolling),
                  "outcomes": outcomes,
                  "invariants_ok": inv_ok, "invariant_issues": inv_issues,
                  "drift_active": self._active_drift is not None}
        self._results.append(record)
        return record

    def _rolling_rate(self) -> Optional[float]:
        """窗口滚动命中率（只聚合已计分 tick；无已计分 tick ⇒ None，非 1.0）。"""
        scored = [r for r in self._rolling if r is not None]
        return round(sum(scored) / len(scored), 4) if scored else None

    def _worst_entity_rate(self) -> Optional[float]:
        """窗口内各实体命中率的最小值（#9）。

        判据来源：triage #9（本仓 `world/spacetime_consistency.py`，判定成立）。
        原判据用「全体实体平均命中率」对比固定 `drift_rate`——N≥4 且有 1 个
        实体永久预测失败时 rolling=(N-1)/N>drift_rate，**永不报警**；该检测器
        只能捕捉「全体同时崩塌」。改为逐实体窗口命中率取 min：任一实体在窗口
        内长期低于阈值即触发。单实体场景下与旧判据逐位等价。
        """
        rates = [sum(d) / len(d) for d in self._entity_hits.values() if len(d)]
        return round(min(rates), 4) if rates else None

    def _update_drift(self, detect_rate: Optional[float]) -> None:
        if detect_rate is None:
            return                    # 无样本：不判漂移（#22：不得把「没测到」当高一致）
        if detect_rate < self.drift_rate:
            self._low_streak += 1
            if self._low_streak >= self.drift_ticks:
                if self._active_drift is None:
                    self._active_drift = {"start_tick": self.tick_count,
                                          "min_rate": detect_rate,
                                          "ticks": self._low_streak}
                else:
                    self._active_drift["ticks"] = self._low_streak
                    self._active_drift["min_rate"] = min(
                        self._active_drift["min_rate"], detect_rate)
        else:
            if self._active_drift is not None:
                self._active_drift["end_tick"] = self.tick_count - 1
                self._drift_events.append(self._active_drift)
                self._active_drift = None
            self._low_streak = 0

    def run(self, n: int = 1) -> Dict:
        """持续运行 n tick（每 tick 预测 vs 实际 → 一致性验证）。"""
        for _ in range(max(0, int(n))):
            self.step_verified()
        return {"status": "ok", "ticks": int(n), "tick": self.tick_count,
                "rolling_hit_rate": self._rolling_rate(),
                "drift_active": self._active_drift is not None}

    # ---- 外部事件（验证漂移检测）----

    def teleport(self, entity_id: str,
                 pos: Tuple[float, float, float]) -> bool:
        """外部事件：排队一次「世界模型无法解释」的瞬移。

        事件在下一验证 tick 的预测之后、实际演化之前生效（模型不知道
        它会发生 → 预测与实际偏离 → 滚动命中率下降 → 漂移事件被检测）。
        用于验证漂移检测：持续运行中注入不一致 → 世界模型自洽破坏被如实报告。
        实体不存在返回 False（不排队）。
        """
        if entity_id not in self.scene.entities:
            return False
        self._pending_events.append((entity_id, tuple(float(v) for v in pos)))
        return True

    # ---- 统计与报告 ----

    def rolling_hit_rate(self, window: Optional[int] = None) -> Optional[float]:
        """滚动命中率（默认当前窗口；无已计分 tick ⇒ None，非 1.0）。"""
        if window is not None and 0 < window < len(self._rolling):
            recent = [r for r in list(self._rolling)[-int(window):] if r is not None]
            return round(sum(recent) / len(recent), 4) if recent else None
        return self._rolling_rate()

    def overall_hit_rate(self) -> Optional[float]:
        """终身累积总体命中率（#427：`_results` 有界裁剪后仍覆盖全部 tick）。

        无任何已计分样本 ⇒ None（#22：不得回数值满分 1.0）。
        """
        return (round(self._agg_hits / self._agg_outcomes, 4)
                if self._agg_outcomes else None)

    def per_behavior_rates(self) -> Dict:
        """分行为命中率（终身累积）：确定性（exact）与随机（bounded）分别统计。

        #427：改由终身累积计数器聚合（`_results` 已改为有界环形缓冲，遍历它
        会随裁剪丢掉早期样本）。无样本桶的 rate 为 None（#22）。
        """
        stats = {b: dict(v) for b, v in self._agg_behavior.items()}
        det = dict(self._agg_mode["exact"])
        sto = dict(self._agg_mode["stochastic"])
        for s in list(stats.values()) + [det, sto]:
            s["rate"] = round(s["hits"] / s["outcomes"], 4) if s["outcomes"] else None
        return {"per_behavior": stats,
                "deterministic": det, "stochastic": sto}

    def drift_events(self) -> List[Dict]:
        """已结束的漂移事件列表（start_tick/min_rate/ticks/end_tick）。"""
        return list(self._drift_events)

    def drift_active(self) -> bool:
        return self._active_drift is not None

    def prediction_history(self, limit: int = 10) -> List[Dict]:
        """预测验证历史（可审计）：每 tick 预测 vs 实际 + 命中。

        #427：`_results` 为有界环形缓冲（最多 `max_results` 条），故此处返回的
        是「最近 limit 条」而非「全部历史」——保留历史条数上限见 `HISTORY_MAXLEN`。
        """
        return list(self._results)[-max(1, int(limit)):]

    def consistency_report(self) -> Dict:
        """自洽度报告：持续运行 + 一致性验证的汇总判定。

        verdict:
          - no_observation：无任何已计分观测（#22 前置独立判定——不得落入
            `self_consistent`；这是「空世界跑满 tick 被判自洽」的判决级假阳性修法）
          - drift_detected：一致性漂移（预测与实际偏离，世界模型不自洽）
          - invariant_violated：世界状态不变量被破坏（#303——此前
            `_invariant_violations` 只进报告、不进判定）
          - inconsistent：总体命中率低于自洽阈值
          - running：持续运行时长不足（继续验证中）
          - self_consistent：持续运行足够且预测与实际保持一致性（自洽）

        判据来源：`docs/plans/待裁清单_v0.1.md` D-65（无样本在数值比较**之前**
        独立判定）／C-11；triage #303（不变量只作数字输出、不进 verdict）。

        #716（本件）：不变量违例判据从**终身累计**改为**窗口内当前状态**
        —— 字段名 `_invariant_violations` 与本节 docstring 的 `invariant_violated`
        都在描述「世界状态**被破坏**」，而 `_check_invariants()` 是**每 tick 的
        当前态校验**（不变量本身是状态谓词，不是事件计数）⇒ 本条判为「当前状态」
        语义。旧实现读终身累计 ⇒ 一次瞬时违规后世界即便完全自愈，`verdict` 也
        永久停在 `invariant_violated`、`self_consistent()` 永久 False。
        修法：新增 `self._inv_window = deque(maxlen=self.window)`，每 tick 追加
        0/1，判定改读 `any(self._inv_window)`；累计值 `_invariant_violations`
        **保留**（审计口径不变，`report["invariant_violations"]` 仍是终身计数）。
        判据来源：triage #716（判定成立）；#303 的修复建议已明写该取舍
        （「如果 owner 倾向『一次违规永久不自洽』，只需把窗口改成全程」——
        本条报的正是代码落到了它标注为**待决**的那一侧）。
        告警行为变化（如实标注）：世界在窗口（`self.window` 个 tick，缺省 20）内
        自愈后，`verdict` 由 `invariant_violated` 回落为按命中率判定；
        `invariant_violations` 累计值**不变**，故「曾经违规」仍可审计。
        """
        overall = self.overall_hit_rate()
        rolling = self._rolling_rate()
        sustained = self.tick_count >= self.min_consistent_ticks
        active = self._active_drift is not None
        if overall is None:
            verdict = "no_observation"
        elif active:
            verdict = "drift_detected"
        elif any(self._inv_window):
            verdict = "invariant_violated"
        elif overall < self.consistent_rate:
            verdict = "inconsistent"
        elif not sustained:
            verdict = "running"
        else:
            verdict = "self_consistent"
        pbr = self.per_behavior_rates()
        return {
            "component": "spacetime_consistency",
            "tick": self.tick_count,
            "sustained": sustained,
            "min_consistent_ticks": self.min_consistent_ticks,
            "window": self.window,
            "overall_hit_rate": overall,
            "rolling_hit_rate": rolling,
            "scored_rounds": sum(1 for r in self._rolling if r is not None),
            "total_rounds": len(self._rolling),
            "consistent_rate": self.consistent_rate,
            "per_behavior": pbr["per_behavior"],
            "deterministic_rate": pbr["deterministic"]["rate"],
            "stochastic_rate": pbr["stochastic"]["rate"],
            "drift_events": list(self._drift_events),
            "drift_active": active,
            "drift_rate": self.drift_rate,
            "invariant_violations": self._invariant_violations,
            "verdict": verdict,
            "self_consistent": verdict == "self_consistent",
        }

    def self_consistent(self) -> bool:
        """世界模型自洽判定：持续运行中预测与实际保持一致性。"""
        return self.consistency_report()["self_consistent"]
