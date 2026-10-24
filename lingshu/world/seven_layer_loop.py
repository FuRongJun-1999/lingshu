# -*- coding: utf-8 -*-
"""seven_layer_loop · 七层闭环（世界模型阶段3 · 里程碑3.4 · 阶段3收官）
============================================================================
核心（荣）：感知→记忆→理解→预测→验证→物理→决策 的完整自主循环——
把前三个里程碑（统一骨干/自监督学习/好奇探索）串成一条持续运行、可审计的闭环。

七层映射（智能论 3.4 七层架构）：
  L1 感知机：好奇选定的实体观测（缸中之脑——只读位置/类别，带宽受限）
  L2 时空记忆图：观测序列 + 世界图（4D 时空记忆）
  L3 认知图：实体/关系/行为推断（理解——谁是谁、谁依赖谁）
  L4 第六感预测：学得模型生成候选未来（带不确定边界）
  L5 锚定验证：外部观察者全实体对比预测 vs 实际 → 命中率（有世界访问权）
  L6 CSPMN物理：物理世界演化（tick）
  L7 决策目标函数：好奇驱动——选择信息增益最大的观测（下一轮 L1）

每 tick 七层留痕（audit trail），闭环报告汇总七层统计 + 自增强曲线
（运行越久 → 认知越准 → 预测越准 → 好奇越聚焦 → 闭环越稳）。

F08 · 可选预测行动（configure_action / clear_action · 默认关闭）：
  受控实体在沙盒里用离散动作（east/west/north/south/stay）接近目标。决策链：
    L7 选观测（保证受控体与目标的必要观测）→ L4 学得模型预测 →
    L7 行动决策：对**每个**离散动作预测下一步自身位置，用学得模型预测目标
    下一步位置，评分＝两者距离 → 确定性择优（tie-break＝动作表序）→
    L6 由 SceneSimulator **真实执行**一步 → L1 新鲜观测 → 计算目标预测误差
    并修正下一 tick 的目标预测（EWMA）→ 下一 tick 决策随之变化。
    误差反馈的诚实口径：只有**本轮新观测到**的目标位置才算新证据；目标本轮未被
    观测（陈旧读数）时显式标注为未验证，不写 correction / error_history
    （见 observe_outcome）。
  信息源硬约束：候选评分**只**读观测（explorer.nodes）与学得模型
  （explorer.predict），不读 SceneSimulator 里的真值目标位置、也不读任何
  behavior/goal 隐藏行为规则；只有执行 actuator 才接触受控实体实际位置。
  未配置（或 clear_action() 之后）时本模块行为与历史逐位一致。

设计参考：
  - 智能论 3.4：七层架构（感知机/时空记忆图/认知图/第六感/锚定验证/CSPMN物理/决策）
  - 第六章：P2 观察者验证（L5）/ P3 系统无需意识 / D1 可达域传播（L4/L7）
  - 里程碑 3.1 world_model（统一骨干）/ 3.2 world_learner（自监督学习）/
    3.3 curiosity_explorer（好奇探索）

纯标准库 · 零外部依赖（D-005）
"""
from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Tuple
from .time_core import cred_blend

# 3D 场景模拟器实现已迁移至 AEIS——大脑保留接口，缺失时 SceneSimulator=None
# （F08 起把原先重复两次的同一条导入回退合并为一处：静默宽 except 计数
#   2→1，已由调度者在 tools/silent_except_lint_baseline.txt 登记 114→113；
#   行为不变，异常一律退回"无模拟器"的降级形态。）
try:
    # F08：离散动作表与真实执行（actuator）同源，避免两处方向定义漂移。
    from .scene_simulator import SceneSimulator, ACTION_DELTAS, ACTION_NAMES
except Exception:
    SceneSimulator = None
    ACTION_DELTAS: Dict[str, Tuple[float, float]] = {}
    ACTION_NAMES: Tuple[str, ...] = ()

try:
    from .curiosity_explorer import CuriosityExplorer
except ImportError:
    from .curiosity_explorer import CuriosityExplorer


#: F08 合法行动策略（fail-closed 白名单；未知策略名在 configure_action 即拒）
ACTION_POLICIES: Tuple[str, ...] = ("predictive", "fixed")
FEEDBACK_POLICIES: Tuple[str, ...] = ("legacy", "qualified")

#: F08 目标误差结算状态（诚实口径）——只有 "verified" 允许写 correction /
#: error_history 并返回 corrected=True；其余状态一律只标注、不记账。
TARGET_ERROR_STATUSES: Tuple[str, ...] = (
    "verified",                  # 本轮新观测到目标 ⇒ 误差是新证据，可反馈
    "static_coordinate",         # 坐标目标：预测即配置常量，误差按构造恒 0
    "stale_target_observation",  # 目标本轮未被观测（陈旧读数）⇒ 不作为新误差反馈
    "target_unobserved",         # 本轮完全没有目标观测 ⇒ 不编造误差
    "no_forecast",               # 本 tick 留痕里没有目标预测 ⇒ 无误差可结算
    "feedback_deferred",         # qualified：等待匹配时刻/新鲜证据，不消费修订
    "feedback_rejected",         # qualified：行动上下文或数值不合法
)


class ActionDecision:
    """F08 行动决策器：观测 + 学得模型 → 离散动作的候选未来与确定性择优。

    信息源（硬约束，判据见 tests/test_third_batch_actions.py）：
      · 受控体 / 目标当前位置 ＝ 学习者观测 explorer.nodes[eid].pos（缸中之脑
        只读位置/类别）；
      · 目标下一 tick 位置 ＝ 学得模型 explorer.predict(horizon=1) 的预测，
        再叠加上一 tick 的观测误差修正（EWMA，见 observe_outcome）——修正**只**
        由本轮新观测到的目标误差驱动，陈旧/缺失观测不参与反馈；
      · 受控体下一 tick 位置 ＝ 动作模型（动作表方向 × step_size + 边界钳制 +
        坐标取整 2 位），与 SceneSimulator 的真实执行口径逐位一致——
        executed.self_error 即该一致性的每 tick 校验。
    本类不持有、也不读 SceneSimulator.entities（真值目标位置）与实体的
    behavior/goal（隐藏行为规则）；真实位移一律由 SceneSimulator 执行。
    """

    #: 误差修正系数（EWMA）：新观测误差权重 α，旧修正权重 1-α。
    CORRECTION_ALPHA = 0.5

    def __init__(self, entity_id: str, target, target_kind: str,
                 step_size: float, policy: str, fixed_action: str,
                 feedback_policy: str = "legacy"):
        if feedback_policy not in FEEDBACK_POLICIES:
            raise ValueError("未知 feedback_policy（可用：legacy/qualified）")
        self.feedback_policy = feedback_policy
        self._last_feedback_tick = None
        self.entity_id = entity_id
        self.target = target
        self.target_kind = target_kind
        self.step_size = float(step_size)
        self.policy = policy
        self.fixed_action = fixed_action
        #: 目标预测的观测误差修正（x/z 面偏移，EWMA）——下一 tick 决策直接使用
        self.correction: Tuple[float, float] = (0.0, 0.0)
        self.decisions = 0
        self.cold_starts = 0
        self.error_history: List[float] = []
        self.last_trace: Optional[Dict] = None

    # ---- 配置面 ----

    def target_ref(self):
        """目标的外部表示：实体 id（str）或 [x, y, z]。"""
        return (list(self.target) if self.target_kind == "coordinate"
                else self.target)

    def config(self) -> Dict:
        """本决策器的当前配置与统计（白箱可审计）。"""
        out = {"entity": self.entity_id, "target": self.target_ref(),
                "target_kind": self.target_kind, "step_size": self.step_size,
                "policy": self.policy, "fixed_action": self.fixed_action,
                "correction": [round(self.correction[0], 4),
                               round(self.correction[1], 4)],
                "decisions": self.decisions, "cold_starts": self.cold_starts,
                "mean_abs_target_error": (
                    round(sum(self.error_history) / len(self.error_history), 4)
                    if self.error_history else None),
                "error_samples": len(self.error_history)}
        if self.feedback_policy == "qualified":
            out.update(feedback_policy=self.feedback_policy,
                       last_feedback_tick=self._last_feedback_tick)
        return out

    # ---- 预测面（全部来自观测与学得模型）----

    @staticmethod
    def predict_self(pos, action: str, step_size: float, size: int) -> List[float]:
        """动作模型：受控体执行 action 后的位置（与 simulator 执行口径一致）。"""
        dx, dz = ACTION_DELTAS[action]
        nx = max(0.5, min(size - 0.5, pos[0] + dx * step_size))
        nz = max(0.5, min(size - 0.5, pos[2] + dz * step_size))
        return [round(nx, 2), round(pos[1], 2), round(nz, 2)]

    def forecast_target(self, explorer, predictions: Optional[Dict] = None
                        ) -> Optional[Dict]:
        """目标下一 tick 位置：学得模型预测 + 观测误差修正（坐标目标 ⇒ 静态）。

        predictions 为空时自行调用 explorer.predict(horizon=1)（standalone 可测）；
        step() 传 L4 同一份预测，避免重复外推。目标从未被观测 ⇒ None（显式失败）。
        """
        if self.target_kind == "coordinate":
            fixed = [round(float(v), 4) for v in self.target]
            return {"predicted": list(fixed), "observed": list(fixed),
                    "mode": "static_coordinate", "bound": 0.0, "samples": None,
                    "model_ready": True, "extrapolated": False,
                    "correction_applied": [0.0, 0.0],
                    "source": "configured_coordinate", "target_tick": None,
                    "observation_age": None}
        node = explorer.nodes.get(self.target)
        if node is None:
            return None
        preds = predictions
        if preds is None:
            preds = explorer.predict(horizon=1)["predictions"]
        pred = preds.get(self.target)
        base = list(pred["predicted"]) if pred else [float(v) for v in node.pos]
        params = (explorer.model or {}).get("per_entity", {}).get(self.target, {})
        samples = params.get("samples")
        corrected = [round(base[0] + self.correction[0], 4), round(base[1], 4),
                     round(base[2] + self.correction[1], 4)]
        observed = [round(float(v), 4) for v in node.pos]
        return {"predicted": corrected, "observed": observed,
                "mode": (pred or {}).get("mode", "no_prediction"),
                "bound": (pred or {}).get("bound"), "samples": samples,
                "model_ready": bool(samples),
                # 外推证据：预测位置 != 观测位置 ⇒ 学得速度/误差修正参与了打分
                "extrapolated": (abs(corrected[0] - node.pos[0]) > 1e-9
                                 or abs(corrected[2] - node.pos[2]) > 1e-9),
                "correction_applied": [round(self.correction[0], 4),
                                       round(self.correction[1], 4)],
                "source": "explorer.predict(horizon=1)+observed_error_correction",
                "target_tick": (pred or {}).get("target_tick"),
                "observation_age": (pred or {}).get("observation_age")}

    # ---- 决策面 ----

    def decide(self, explorer, predictions: Optional[Dict] = None,
               cold_start: bool = False) -> Dict:
        """候选未来 → 评分 → 确定性择优；返回本 tick 行动留痕（未执行）。"""
        trace: Dict = {"enabled": True, "status": "ok",
                       "entity": self.entity_id, "target": self.target_ref(),
                       "target_kind": self.target_kind, "policy": self.policy,
                       "fixed_action": self.fixed_action,
                       "step_size": self.step_size,
                       "cold_start": bool(cold_start),
                       "information_source": "观测 explorer.nodes + 学得模型 explorer.predict",
                       "decision_rule": "argmin(score) over ACTION_NAMES（tie-break＝动作表序）",
                       "candidates": [], "model_choice": None,
                       "greedy_current_choice": None, "chosen_action": None,
                       "forecast": None, "observation": None,
                       "executed": None, "outcome": None}
        if self.feedback_policy == "qualified":
            trace["feedback_context"] = {
                "entity": self.entity_id, "target": self.target_ref(),
                "target_kind": self.target_kind, "step_size": self.step_size,
                "policy": self.policy, "fixed_action": self.fixed_action,
                "decision_tick": explorer.tick, "expected_observation_tick": explorer.tick + 1}
        node = explorer.nodes.get(self.entity_id)
        if node is None:
            trace["status"] = "controlled_unobserved"
            self.last_trace = trace
            return trace
        forecast = self.forecast_target(explorer, predictions)
        if forecast is None:
            trace["status"] = "target_unobserved"
            self.last_trace = trace
            return trace
        trace["forecast"] = forecast
        trace["model_ready"] = bool(forecast["model_ready"])
        if not forecast["model_ready"]:
            # 冷启动/观测不足**明确标注**：学得模型还估不出目标速度时，候选
            # 评分退化为「按最新观测位置贪心」，审计里不再声称在做外推。
            trace["degraded"] = "insufficient_target_observations"
        trace["observation"] = {
            "self": [round(float(v), 4) for v in node.pos],
            "target": list(forecast["observed"]),
            "self_tick": node.last_seen,
            "target_tick": (explorer.nodes[self.target].last_seen
                            if self.target_kind == "entity" else None),
            "self_age": max(0, explorer.tick - node.last_seen),
            "target_age": (max(0, explorer.tick - explorer.nodes[self.target].last_seen)
                           if self.target_kind == "entity" else 0),
            "source": "explorer.nodes（观测面，非物理真值）"}
        base_self = list(trace["observation"]["self"])
        candidates = []
        for name in ACTION_NAMES:
            ps = self.predict_self(base_self, name, self.step_size, explorer.size)
            candidates.append({"action": name, "predicted_self": ps,
                               "score": round(math.dist(
                                   ps, forecast["predicted"]), 4)})
        trace["candidates"] = candidates
        scores = {c["action"]: c["score"] for c in candidates}
        # 不外推对照（同一观测面）：以目标**当前观测位置**评分的最优动作——
        # 记录它是为了给出「预测真的参与了选择」的可复算证据，不参与执行。
        greedy_scores = {c["action"]: round(math.dist(c["predicted_self"],
                                                       forecast["observed"]), 4)
                         for c in candidates}
        trace["greedy_scores"] = greedy_scores
        trace["greedy_current_choice"] = min(
            (c["action"] for c in candidates),
            key=lambda a: (greedy_scores[a], ACTION_NAMES.index(a)))
        model_choice = min((c["action"] for c in candidates),
                           key=lambda a: (scores[a], ACTION_NAMES.index(a)))
        trace["model_choice"] = model_choice
        chosen = self.fixed_action if self.policy == "fixed" else model_choice
        chosen_candidate = next(c for c in candidates if c["action"] == chosen)
        trace["chosen_action"] = chosen
        trace["executed"] = {"action": chosen,
                             "predicted_self": chosen_candidate["predicted_self"],
                             "source": "SceneSimulator.set_command（真实执行侧）",
                             "observed_to": None, "self_error": None,
                             "distance_before": round(math.dist(
                                 trace["observation"]["self"],
                                 trace["observation"]["target"]), 4),
                             "distance_after": None}
        self.decisions += 1
        if cold_start:
            self.cold_starts += 1
        self.last_trace = trace
        return trace

    def observe_outcome(self, trace: Dict, observed_self, observed_target,
                        target_fresh: bool, *, self_fresh: Optional[bool] = None,
                        feedback_tick: Optional[int] = None) -> Dict:
        """用**新观测**结算上一预测：自/目标误差 + 距离变化 + 修正下一步。

        诚实口径（F08 本次返工钉死）：只有 target_fresh=True 且本轮**确实观测到**
        目标时，该目标位置才是新证据。陈旧读数（target_fresh=False：本轮未观测到
        目标，用的是上一 tick 或更早的观测）在决策时已经作为预测输入用过，再与预测
        比一次得到的是「预测 vs 自身输入」的循环读数——据此更新 EWMA correction /
        error_history 等于把同一份观测重复计入反馈回路，会把修正量推向一个没有
        新证据支持的方向。

        故 target_fresh=False（含此前已持有非零 correction 的对象）：不写 correction、
        不写 error_history、不返回 corrected=True；outcome 以
        target_error_status="stale_target_observation" + target_error_reason 写明
        理由，target_error / target_error_norm 保持 None（不宣称 target 误差已验证）。
        缺观测（observed_target 为空）与无目标预测同样只标状态、不记账；坐标目标是
        配置常量（误差按构造恒 0），单列 static_coordinate。只有新观测走既有更新路径：
        误差入 error_history、entity 目标的 correction 按 EWMA 更新、corrected=True
        （已知误差 1→0.5→0 主路径不变）。

        executed.distance_after 仍按观测面几何照算——它只是「两个被观测位置的距离」
        读数，不带「目标误差已验证」含义。
        """
        if trace.get("status") != "ok" or trace.get("chosen_action") is None:
            return trace
        if self.feedback_policy == "qualified":
            return self._qualified_outcome(trace, observed_self, observed_target,
                                           target_fresh, self_fresh, feedback_tick)
        executed = trace.get("executed") or {}
        forecast = trace.get("forecast") or {}
        predicted_self = executed.get("predicted_self")
        predicted_target = forecast.get("predicted")
        if observed_self is not None and predicted_self is not None:
            executed["observed_to"] = observed_self
            executed["self_error"] = round(math.dist(predicted_self, observed_self), 4)
        if observed_self is not None and observed_target is not None:
            executed["distance_after"] = round(
                math.dist(observed_self, observed_target), 4)
        trace["executed"] = executed
        outcome: Dict = {"observed_self": observed_self,
                         "observed_target": observed_target,
                         "predicted_target": predicted_target,
                         "target_observation_fresh": bool(target_fresh),
                         "target_error": None, "target_error_norm": None,
                         "target_error_status": None, "target_error_reason": None,
                         "corrected": False,
                         "next_correction": [round(self.correction[0], 4),
                                             round(self.correction[1], 4)]}
        if predicted_target is None:
            outcome["target_error_status"] = "no_forecast"
            outcome["target_error_reason"] = (
                "本 tick 留痕没有目标预测（forecast.predicted 为空）⇒ 无误差可结算，"
                "不写 correction / error_history")
        elif observed_target is None:
            outcome["target_error_status"] = "target_unobserved"
            outcome["target_error_reason"] = (
                "本轮没有任何目标观测（observed_target 为空）⇒ 不编造目标误差，"
                "不写 correction / error_history")
        elif not target_fresh:
            outcome["target_error_status"] = "stale_target_observation"
            outcome["target_error_reason"] = (
                "目标本轮未被观测（target_fresh=False）：该位置在决策时已作为预测"
                "输入用过，陈旧读数不是新误差反馈 ⇒ 不写 correction / error_history，"
                "不宣称 target 误差已验证")
        else:
            err = [round(observed_target[0] - predicted_target[0], 4),
                   round(observed_target[2] - predicted_target[2], 4)]
            outcome["target_error"] = err
            outcome["target_error_norm"] = round(math.hypot(err[0], err[1]), 4)
            self.error_history.append(outcome["target_error_norm"])
            if self.target_kind == "entity":
                a = self.CORRECTION_ALPHA
                self.correction = (
                    round(a * err[0] + (1 - a) * self.correction[0], 4),
                    round(a * err[1] + (1 - a) * self.correction[1], 4))
                outcome["corrected"] = True
                outcome["next_correction"] = [self.correction[0],
                                              self.correction[1]]
                outcome["target_error_status"] = "verified"
            else:
                # 坐标目标：预测位置＝配置常量，误差按构造恒 0，不是对目标观测的
                # 验证；既有统计口径（0.0 样本入 error_history、correction 不动）未改。
                outcome["target_error_status"] = "static_coordinate"
                outcome["target_error_reason"] = (
                    "坐标目标是配置常量（预测位置即该常量）⇒ 误差按构造恒为 0，"
                    "不构成对目标观测的验证；既有统计口径未改")
        trace["outcome"] = outcome
        return trace

    @staticmethod
    def _feedback_component(predicted, observed, fresh: bool) -> Dict:
        """分量残差只来自新鲜观测；不是对执行器/目标行为的因果裁决。"""
        row = {"state": "DEFER", "reason": "no_forecast", "predicted": None,
               "observed": None, "residual_xyz": None, "norm": None}
        if predicted is None:
            return row
        if observed is None:
            row["reason"] = "unobserved"
            return row
        if fresh is not True:
            row["reason"] = "stale_observation"
            return row
        for vector in (predicted, observed):
            try:
                invalid = (not isinstance(vector, (list, tuple)) or len(vector) != 3
                           or any(isinstance(v, bool) or not isinstance(v, (int, float))
                                  or not math.isfinite(v) for v in vector))
            except OverflowError:
                invalid = True
            if invalid:
                row.update(state="REJECT", reason="invalid_vector")
                return row
        delta = [round(observed[i] - predicted[i], 4) for i in range(3)]
        norm = round(math.dist(predicted, observed), 4)
        if not all(math.isfinite(v) for v in [*delta, norm]):
            row.update(state="REJECT", reason="nonfinite_residual")
            return row
        row.update(state="ACCEPT", reason="fresh_observation",
                   predicted=list(predicted), observed=list(observed),
                   residual_xyz=delta, norm=norm)
        return row

    def _qualified_outcome(self, trace, observed_self, observed_target,
                           target_fresh, self_fresh, feedback_tick) -> Dict:
        """W1：分别诊断残差，只修订本行动目标的 x/z 预测偏移。"""
        ctx = trace.get("feedback_context") or {}
        expected = {"entity": self.entity_id, "target": self.target_ref(),
                    "target_kind": self.target_kind, "step_size": self.step_size,
                    "policy": self.policy, "fixed_action": self.fixed_action}
        reason = None
        if not isinstance(ctx, dict) or any(ctx.get(k) != v for k, v in expected.items()):
            reason = ("goal_context_changed" if isinstance(ctx, dict)
                      and ctx.get("target") != self.target_ref() else "action_context_changed")
        elif (isinstance(feedback_tick, bool) or not isinstance(feedback_tick, int)
              or feedback_tick != ctx.get("expected_observation_tick")
              or ctx.get("decision_tick") != feedback_tick - 1):
            reason = "feedback_tick_mismatch"
        elif self._last_feedback_tick is not None and feedback_tick <= self._last_feedback_tick:
            reason = "duplicate_or_out_of_order_feedback"
        executed = dict(trace.get("executed") or {})
        forecast = trace.get("forecast") or {}
        execution = self._feedback_component(executed.get("predicted_self"), observed_self, self_fresh)
        target = self._feedback_component(forecast.get("predicted"), observed_target, target_fresh)
        if self.target_kind == "coordinate":
            target = {"state": "BLINDSPOT", "reason": "configured_coordinate",
                      "predicted": self.target_ref(), "observed": None,
                      "residual_xyz": None, "norm": None}
        revision = {"state": "DEFER", "target": "target_forecast_offset_xz",
                    "reason": "target_evidence_unqualified", "changed": False,
                    "before": list(self.correction), "after": list(self.correction),
                    "base_forecast_residual_xz": None}
        if reason is not None:
            state = "DEFER" if reason == "duplicate_or_out_of_order_feedback" else "REJECT"
            revision.update(state=state, reason=reason)
            for component in (execution, target):
                component.update(state=state, reason=reason, residual_xyz=None, norm=None)
        elif target["state"] == "ACCEPT":
            applied = forecast.get("correction_applied")
            if (not isinstance(applied, (list, tuple)) or list(applied) != list(self.correction)
                    or any(isinstance(v, bool) or not isinstance(v, (int, float))
                           or not math.isfinite(v) for v in applied)):
                revision.update(state="REJECT", reason="revision_context_changed")
            else:
                # 残差针对已加偏移的预测；先还原对未修正模型的偏差，再交统一时间核。
                err = [target["residual_xyz"][0], target["residual_xyz"][2]]
                innovation = [err[i] + applied[i] for i in range(2)]
                a = self.CORRECTION_ALPHA
                proposed = tuple(round(cred_blend(self.correction[i], a * innovation[i], 1 - a), 4)
                                 for i in range(2))
                if not all(math.isfinite(v) for v in [*innovation, *proposed]):
                    revision.update(state="REJECT", reason="nonfinite_revision")
                else:
                    revision.update(state="ACCEPT", reason="fresh_target_residual",
                                    changed=proposed != self.correction, after=list(proposed),
                                    base_forecast_residual_xz=innovation)
                    self.correction = proposed
                    self.error_history.append(round(math.hypot(*err), 4))
                    self._last_feedback_tick = feedback_tick
        elif target["state"] == "REJECT":
            revision.update(state="REJECT", reason=target["reason"])
        elif self.target_kind == "coordinate":
            revision.update(state="BLINDSPOT", reason="configured_coordinate")
        else:
            revision["reason"] = target["reason"]
        execution_changed = execution["norm"] is not None and execution["norm"] > 0
        target_changed = target["norm"] is not None and target["norm"] > 0
        classification = (reason or ("execution_and_target_residual" if execution_changed and target_changed
            else "execution_residual" if execution_changed else "target_forecast_residual" if target_changed
            else "no_measured_residual" if execution["state"] == target["state"] == "ACCEPT"
            else "partial_or_missing_evidence"))
        executed.update(observed_to=execution["observed"], self_error=execution["norm"], distance_after=None)
        distance_target = target["observed"] if self.target_kind == "entity" else self.target_ref()
        if reason is None and execution["state"] == "ACCEPT" and distance_target is not None:
            executed["distance_after"] = round(math.dist(execution["observed"], distance_target), 4)
        target_accepted = target["state"] == "ACCEPT" and reason is None
        status = ("feedback_rejected" if revision["state"] == "REJECT" else "verified" if target_accepted
                  else "static_coordinate" if self.target_kind == "coordinate" and reason is None else "feedback_deferred")
        outcome = {"observed_self": execution["observed"], "observed_target": target["observed"],
                   "predicted_target": target["predicted"], "target_observation_fresh": target_fresh is True,
                   "target_error": ([target["residual_xyz"][i] for i in (0, 2)] if target_accepted else None),
                   "target_error_norm": (round(math.hypot(target["residual_xyz"][0], target["residual_xyz"][2]), 4)
                                         if target_accepted else None),
                   "target_error_status": status,
                   "target_error_reason": (revision["reason"] if status.startswith("feedback_") else target["reason"]),
                   "corrected": revision["changed"], "next_correction": list(self.correction),
                   "diagnosis": {"feedback_policy": self.feedback_policy, "feedback_tick": feedback_tick,
                       "classification": classification, "cause": "unknown",
                       "cause_note": "分量残差不能单独证明目标意图变化、执行器故障或身份/观测错误",
                       "execution": execution, "target_forecast": target, "revision": revision,
                       "unsupported_correction_axes": ["y"]}}
        # 拒绝旧留痕时不改写之前已入审计的对象；原始成功结算仍与 audit 共用同一留痕。
        out = dict(trace) if reason is not None else trace
        out.update(executed=executed, outcome=outcome)
        return out


class SevenLayerLoop:
    """七层闭环：感知→记忆→理解→预测→验证→物理→决策 的持续自主循环。

    每 tick 执行完整七层，产出七层留痕（audit）；闭环报告汇总各层统计。
    学习者（CuriosityExplorer）内部封装 WorldLearner 学得模型；物理世界
    （SceneSimulator）在模型之外——L5 外部观察者有世界访问权，模型没有。

    F08（可选，默认关闭）：configure_action(entity_id, target, step_size,
    policy="predictive", fixed_action="east") 让受控实体用离散动作接近目标；
    clear_action() 关闭。配置后每 tick 的候选未来/评分/选择/执行/后续观测/
    误差写入 audit 的 "L7_action" 字段；未配置时 audit 形状与历史行为不变。
    """

    LAYERS = ("L1_perception", "L2_memory", "L3_cognition", "L4_prediction",
              "L5_verification", "L6_physics", "L7_decision")

    def __init__(self, size: int = 24, ground_level: int = 1, seed: int = 42,
                 window: int = 6, budget: int = 2, policy: str = "curiosity",
                 world: Optional[SceneSimulator] = None):
        self.world = world or SceneSimulator(size=size, ground_level=ground_level,
                                             seed=seed)
        self.explorer = CuriosityExplorer(size=size, ground_level=ground_level,
                                          seed=seed, window=window, world=self.world)
        self.budget = max(1, int(budget))
        # issue #233（B）：策略名白名单 fail-closed——未知策略在构造点即拒，
        # 不再「拼错即静默轮询、审计照抄拼错的策略名」。
        if policy not in CuriosityExplorer.POLICY_NAMES:
            raise ValueError(
                "未知 policy %r（可用：%s）——拒绝静默退化为轮询（issue #233）"
                % (policy, "/".join(CuriosityExplorer.POLICY_NAMES)))
        self.policy = policy
        self.audit: List[Dict] = []
        self.hit_history: List[float] = []
        # F08：预测行动配置（None＝未配置 ⇒ step() 走历史路径，零影响）
        self._action: Optional[ActionDecision] = None
        self._action_log: List[Dict] = []
        self.tick = 0

    # ---- 场景构建（透传物理世界）----

    def create_scene(self, trees: int = 2, water: bool = False) -> Dict:
        return self.world.create_scene(trees=trees, water=water)

    def add_entity(self, category: str, behavior: str = "wander",
                   pos: Tuple[float, float, float] = (2, 1.5, 2),
                   speed: float = 0.3, goal: str = "") -> str:
        return self.world.add_entity(category=category, behavior=behavior,
                                     pos=pos, speed=speed, goal=goal)

    def add_path(self, path_id: str,
                 points: List[Tuple[float, float, float]]) -> None:
        self.world.add_path(path_id, points)

    # ---- F08 预测行动（可选；未配置时零影响）----

    def configure_action(self, entity_id, target, step_size: float = 1.0,
                         policy: str = "predictive",
                         fixed_action: str = "east", feedback_policy: str = "legacy") -> Dict:
        """配置受控实体的预测行动（F08）。

        entity_id：受控实体 id（必须在场）；target：目标实体 id，或 (x, y, z)
        坐标；step_size：每个离散动作的位移（有限正数）；policy：predictive
        （用观测 + 学得模型预测每个动作的结果后确定性择优）或 fixed（固定动作
        基线）；fixed_action 是 fixed 策略执行的动作，且**恒**先校验（拼错的
        动作名在同一入口即拒，不静默退化为默认）。

        全部校验先于任何状态写入：任一项不合法 ⇒ ValueError，当前配置一律不变
        （无污染）。configure_action 本身不读世界真值、不改观测面。
        """
        if policy not in ACTION_POLICIES:
            raise ValueError("未知行动 policy %r（可用：%s）"
                             % (policy, "/".join(ACTION_POLICIES)))
        # 动作名 + 步长由动作表所在模块（真实执行侧）裁决，避免两处口径漂移
        step = self.world.validate_action(fixed_action, step_size)
        if entity_id not in self.world.entities:
            raise ValueError("受控实体 %r 不在场景中（现有：%s）"
                             % (entity_id, "/".join(sorted(self.world.entities))))
        target_kind, resolved = self._resolve_target(entity_id, target)
        decision = ActionDecision(entity_id=entity_id, target=resolved,
                                  target_kind=target_kind, step_size=step,
                                  policy=policy, fixed_action=fixed_action,
                                  feedback_policy=feedback_policy)
        # 原子替换：校验全部通过后才落配置；顺带清掉可能残留的待消费动作
        self._action = decision
        self.world.clear_command()
        # 返回值＝**行动配置载荷本身**（core 门面把它放进 action_config 字段，
        # 见 tests/test_third_batch_core_action.py 的转发约定）——不再自套一层
        # status/action_config，避免门面下面出现第二层同名包装。
        return decision.config()

    def clear_action(self) -> Dict:
        """关闭预测行动（幂等）。返回撤销摘要。

        关闭后 step() 的观测选择、预测、验证、留痕与配置前逐位一致（audit 不再
        出现 L7_action）；已产生的行动留痕保留在 _action_log（历史证据不丢）。
        """
        previous = self._action
        self._action = None
        pending = self.world.clear_command()
        return {"cleared": previous is not None,
                "entity": previous.entity_id if previous is not None else None,
                "pending_commands_removed": pending["removed"],
                "action_ticks": len(self._action_log)}

    def action_state(self) -> Dict:
        """预测行动状态（配置 + 累计 + 最近一 tick 完整留痕）。"""
        return {"configured": self._action is not None,
                "config": (self._action.config() if self._action is not None
                           else None),
                "action_ticks": len(self._action_log),
                "latest": self._action_log[-1] if self._action_log else None}

    def action_log_view(self, limit: int = 10) -> List[Dict]:
        """行动留痕轨迹（最近 n tick；与 audit 的 L7_action 同一对象）。"""
        return self._action_log[-max(1, int(limit)):]

    def _resolve_target(self, entity_id, target):
        """解析目标：实体 id（str）或 3D 坐标；不合法一律 ValueError。"""
        if isinstance(target, str):
            if target == entity_id:
                raise ValueError("目标不能是受控实体自身（%r）" % (target,))
            if target not in self.world.entities:
                raise ValueError("目标实体 %r 不在场景中（现有：%s）"
                                 % (target, "/".join(sorted(self.world.entities))))
            return "entity", target
        if isinstance(target, (tuple, list)):
            if len(target) != 3:
                raise ValueError("坐标目标需为 (x, y, z) 三元组，收到 %r"
                                 % (target,))
            vals = []
            for v in target:
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise ValueError("坐标目标需为有限数值，收到 %r" % (target,))
                f = float(v)
                if not math.isfinite(f):
                    raise ValueError("坐标目标需为有限数值，收到 %r" % (target,))
                vals.append(f)
            return "coordinate", tuple(vals)
        raise ValueError("目标 %r 既不是实体 id 也不是 (x, y, z) 坐标"
                         % (target,))

    def _required_observations(self, act: ActionDecision) -> List[str]:
        """必要观测：受控体 + 目标实体（坐标目标无需观测；自身作目标已被拒）。"""
        req = [act.entity_id]
        if act.target_kind == "entity" and act.target != act.entity_id:
            req.append(act.target)
        return [e for e in req if e in self.world.entities]

    def _ensure_action_observations(self, act: ActionDecision
                                    ) -> Tuple[List[str], bool]:
        """冷启动：受控体/目标**从未被观测** ⇒ 决策前先取一次真实观测。

        只在「从未观测」时触发（本循环的首个行动 tick）。此后每 tick 由 L1 的
        观测集合（_merge_required）保证两者的新鲜观测，故不会出现同 tick 重复
        观测——重复观测会在运动窗口里插入 0 位移读数，把学得速度估计拉低。
        """
        req = self._required_observations(act)
        missing = [e for e in req if e not in self.explorer.nodes]
        if not missing:
            return req, False
        self.explorer.observe(entities=missing)
        return req, True

    def _merge_required(self, chosen: List[str], required: List[str]) -> List[str]:
        """把受控体与目标并入本轮观测（必要观测优先，保序去重）。

        budget 语义说明：explorer._select 的选择逻辑与 self.budget 一律未改；
        这里只在预算够时用必要观测顶替 IG 名额（总数仍 = budget），预算不够
        （len(required) > budget）时扩到 len(required) 并在审计里显式标出
        budget_expanded / effective_observations——不静默超预算、不静默丢必要观测。
        """
        out = list(required)
        limit = max(1, int(self.budget))
        for eid in chosen:
            if len(out) >= limit:
                break
            if eid not in out:
                out.append(eid)
        return out

    def _finalize_action(self, act: ActionDecision, trace: Dict) -> Dict:
        """行动结算：用本轮 L1 的**新鲜观测**算误差、修正下一 tick 的目标预测。"""
        self_tick = self.explorer.tick
        self_node = self.explorer.nodes.get(act.entity_id)
        observed_self = ([round(float(v), 4) for v in self_node.pos]
                         if self_node is not None else None)
        if act.target_kind == "entity":
            target_node = self.explorer.nodes.get(act.target)
            observed_target = ([round(float(v), 4) for v in target_node.pos]
                               if target_node is not None else None)
            fresh = bool(target_node is not None
                         and target_node.last_seen == self_tick)
        else:
            observed_target = [round(float(v), 4) for v in act.target]
            fresh = True
        if act.feedback_policy == "qualified":
            return act.observe_outcome(trace, observed_self, observed_target, fresh,
                self_fresh=bool(self_node is not None and self_node.last_seen == self_tick),
                feedback_tick=self_tick)
        return act.observe_outcome(trace, observed_self, observed_target, fresh)

    # ---- 七层一步 ----

    def _verify(self, predictions: Dict) -> Dict:
        """L5 锚定验证：外部观察者全实体对比；无可比项时 hit_rate=None。"""
        hits, total = 0, 0
        point_hits = 0
        distances, bounds = [], []
        details = []
        for eid, p in predictions.items():
            e = self.world.entities.get(eid)
            if e is None:
                continue
            dist = math.dist(p["predicted"], tuple(e.pos))
            distances.append(dist)
            bounds.append(p["bound"])
            point_hits += dist < self.explorer.hit_threshold
            hit = dist < p["bound"]
            if hit:
                hits += 1
            total += 1
            details.append({"entity": eid, "mode": p["mode"], "hit": hit,
                            "distance": round(dist, 4)})
        rate = round(hits / total, 4) if total else None
        return {"hits": hits, "total": total, "hit_rate": rate,
                "bound_coverage": rate,
                "point_hit_rate": round(point_hits / total, 4) if total else None,
                "mean_distance": round(sum(distances) / total, 4) if total else None,
                "mean_bound": round(sum(bounds) / total, 4) if total else None,
                "details": details}

    def step(self) -> Dict:
        """闭环一步（七层顺序执行）：

        L7 决策（选本轮观测）→ L4 预测（基于当前模型）→ L6 物理演化 →
        L1 感知（好奇选定的实体）→ L5 验证（预测 vs 实际）→
        L2 记忆 / L3 认知（更新）。

        F08（可选）：配置了预测行动时，本轮在 L4 之后、L6 之前插入行动决策
        （候选未来/评分/确定性择优），由 SceneSimulator 在 L6 真实执行该动作，
        再用 L1 的新鲜观测结算误差并修正下一 tick 的目标预测；留痕 rec["L7_action"]。
        未配置时本函数不新增任何字段、观测集合与决策路径与历史逐位一致。
        """
        rec: Dict = {}
        act = self._action
        required: List[str] = []
        cold_start = False
        if act is not None:
            # 必要观测（受控体 + 目标）：冷启动首 tick 先补一次真实观测
            required, cold_start = self._ensure_action_observations(act)
        # L7 决策目标函数（好奇：信息增益最大化）
        chosen, scores = self.explorer._select(self.budget, self.policy)
        ig_selected = list(chosen)
        if act is not None:
            chosen = self._merge_required(chosen, required)
        rec["L7_decision"] = {"chosen": list(chosen), "ig_scores": scores,
                              "policy": self.policy, "budget": self.budget}
        if act is not None:
            budget = max(1, int(self.budget))
            expanded = len(chosen) > budget
            note = ("必要观测（受控体+目标）优先：预算 %d 不足以覆盖，本轮扩到 %d 个"
                    "（已显式标注，不静默超预算）" % (budget, len(chosen)))
            rec["L7_decision"].update({
                "required_observations": list(required),
                "ig_selected": ig_selected,
                "effective_observations": len(chosen),
                "budget_expanded": expanded,
                "budget_note": note if expanded else "必要观测已含在预算内，总数不变"})
        # L4 第六感预测（学得模型 + 不确定边界）
        if not self.explorer.model:
            self.explorer.learn()
        pred = self.explorer.predict(horizon=1)
        rec["L4_prediction"] = {"predictions": pred["predictions"]}
        # F08 行动决策（紧随 L4：候选未来复用同一份学得模型预测）
        if act is not None:
            trace = act.decide(self.explorer, predictions=pred["predictions"],
                               cold_start=cold_start)
            rec["L7_action"] = trace
            self._action_log.append(trace)
            if trace.get("status") == "ok":
                # 真实执行交给 simulator（actuator）；决策值只在这里落地
                self.world.set_command(act.entity_id, trace["chosen_action"],
                                       act.step_size)
            else:
                # 决策面拿不到必要观测 ⇒ 显式告警 + 本 tick 不登记动作（不猜、
                # 不静默按旧动作执行）；留痕里的 status 已写明原因。
                warnings.warn(
                    "seven_layer_loop: F08 行动决策未就绪（status=%s）⇒ "
                    "本 tick 不登记受控动作" % trace.get("status"),
                    RuntimeWarning, stacklevel=2)
        # L6 CSPMN 物理（世界演化；受控动作在 simulator 内真实执行）
        self.world.step(n=1)
        rec["L6_physics"] = {"world_tick": self.world.tick_count,
                             "entities": len(self.world.entities)}
        # L1 感知机（好奇选定的实体，缸中之脑）
        self.explorer.observe(entities=chosen)
        anomalies = self.explorer._count_anomalies(chosen)
        rec["L1_perception"] = {"observed": list(chosen),
                                "observed_count": len(chosen), "anomalies": anomalies}
        if act is not None and rec["L7_action"].get("status") == "ok":
            # 后续观测 → 误差 → 修正下一步（只走观测面）
            self._finalize_action(act, rec["L7_action"])
        # L5 锚定验证（外部观察者全实体）
        ver = self._verify(pred["predictions"])
        rec["L5_verification"] = {key: val for key, val in ver.items() if key != "details"}
        if ver["hit_rate"] is not None:
            self.hit_history.append(ver["hit_rate"])
        # L2 时空记忆图（观测序列 + 世界图）
        rec["L2_memory"] = {"history_len": len(self.explorer.history),
                            "entities": {eid: list(n.pos)
                                         for eid, n in self.explorer.nodes.items()}}
        # L3 认知图（关系/行为推断）
        self.explorer.learn()
        rec["L3_cognition"] = {
            "relations": [r for r in self.explorer.model.get("relations", [])],
            "entity_count": len(self.explorer.nodes)}
        self.tick += 1
        rec["tick"] = self.tick
        self.audit.append(rec)
        return rec

    def run(self, n: int = 30) -> Dict:
        """持续运行 n tick（完整七层闭环）。

        自举：若尚无观测，先做一轮初始全带宽感知（闭环启动需要认知种子——
        与单元测试/引擎调用方行为一致，保证闭环自增强度量可比）。

        F08 边界：本自举段是**感知种子阶段**（全带宽观测 window tick），既不做
        行动决策也不登记受控动作——配置了预测行动时，受控实体在这几 tick 仍按
        自身 behavior 演化；此后每个 step() 才逐 tick 参与预测行动闭环。

        issue #233（A）：返回的 `"ticks"` 改为**真实推进的闭环 tick 数**
        （`self.tick` 增量），且 `n <= 0` 不再偷跑 1 个 tick。旧码
        `for _ in range(max(1, int(n)))` 配 `"ticks": int(n)`——`run(0)` 的
        同一个返回 dict 里 `ticks=0` 而 `loop_tick=1`、`run(-5)` 报
        `ticks=-5` 却已推进 1 个 tick（自相矛盾，且与 docstring「运行 n tick」
        不符）。判据＝《PR 审核细则 v0.1》§三.4 判据外置：字段语义须由断言
        钉住（`ticks` == 实际推进数）。

        issue #725（B）：上面那次只修了主体循环，**bootstrap 分支未动**——
        `if not self.explorer.history:` 无条件执行「全带宽感知 + 跑 window tick」，
        故 `run(0)`/`run(-5)` 仍偷跑世界 `window` tick + `window+1` 帧观测，而同一
        返回 dict 报 `ticks=0`/`loop_tick=0`（副作用与返回值自相矛盾）。现把自举
        收进 `n0 > 0`：自举的用途是给**本轮要跑的闭环**备认知种子（`step()` 的
        L7 决策与 L4 预测需要 explorer 已有观测），`n0 == 0` 时闭环一次都不跑
        ⇒ 不需要种子，也就不该改世界状态/观测史。`ticks` 语义未变（仍是闭环
        tick 增量，自举段不推进 `self.tick`）。
        """
        n0 = max(0, int(n))
        if n0 > 0 and not self.explorer.history:
            self.explorer.observe()
            self.explorer.run(n=self.explorer.window)
        start = self.tick
        for _ in range(n0):
            self.step()
        return {"status": "ok", "ticks": self.tick - start, "loop_tick": self.tick,
                "overall_hit_rate": self._overall_hit_rate()}

    def _overall_hit_rate(self) -> Optional[float]:
        total = sum(rec["L5_verification"]["total"] for rec in self.audit)
        hits = sum(rec["L5_verification"]["hits"] for rec in self.audit)
        return round(hits / total, 4) if total else None

    # ---- 闭环报告与审计 ----

    def report(self) -> Dict:
        """闭环报告：七层统计 + 自增强曲线。"""
        recent = self.hit_history[-min(10, len(self.hit_history)):]
        early = self.hit_history[:max(1, len(self.hit_history) // 2)]
        late = self.hit_history[len(self.hit_history) // 2:]
        return {
            "component": "seven_layer_loop",
            "loop_tick": self.tick,
            "L1_perception": {"observations": len(self.explorer.history),
                              "obs_distribution": dict(self.explorer.obs_counts)},
            "L2_memory": {"history_len": len(self.explorer.history),
                          "entities": len(self.explorer.nodes)},
            "L3_cognition": {"relations": len(self.explorer.model.get("relations", [])),
                             "entity_count": len(self.explorer.nodes)},
            "L4_prediction": {"entity_count": len(self.explorer._last_prediction)},
            "L5_verification": {"overall_hit_rate": self._overall_hit_rate(),
                                "verified_ticks": len(self.hit_history),
                                "unverified_ticks": self.tick - len(self.hit_history),
                                "recent": recent},
            "L6_physics": {"world_tick": self.world.tick_count,
                           "entities": len(self.world.entities)},
            "L7_decision": {"policy": self.policy, "budget": self.budget,
                            "obs_distribution": dict(self.explorer.obs_counts)},
            "closed_loop_enhancement": {
                "early_hit_rate": round(sum(early) / len(early), 4) if early else None,
                "late_hit_rate": round(sum(late) / len(late), 4) if late else None,
                "improvement": round(sum(late) / len(late) - sum(early) / len(early), 4)
                if early and late else None},
            "loop_closed": True,
        }

    def audit_view(self, limit: int = 10) -> List[Dict]:
        """审计轨迹：最近 n tick 的七层留痕（白箱可审计）。"""
        return self.audit[-max(1, int(limit)):]

    def verify_state(self) -> Dict:
        return {"hit_rate": self._overall_hit_rate(),
                "hit_history": self.hit_history[-20:]}

    def decision_state(self) -> Dict:
        return {"policy": self.policy, "budget": self.budget,
                "obs_distribution": dict(self.explorer.obs_counts),
                "latest": self.audit[-1]["L7_decision"] if self.audit else {}}

    def memory_state(self) -> Dict:
        return {"history_len": len(self.explorer.history),
                "entities": {eid: list(n.pos) for eid, n in self.explorer.nodes.items()},
                "history_view": self.explorer.history[-5:]}

    def graph_state(self) -> Dict:
        return {"relations": self.explorer.model.get("relations", []),
                "entities": {eid: {"category": n.category, "pos": list(n.pos)}
                             for eid, n in self.explorer.nodes.items()}}

    def state(self) -> Dict:
        """闭环状态。配置了 F08 预测行动时额外带 action 配置（默认无此键）。"""
        state = {"status": "ok", "loop_tick": self.tick,
                 "audit_len": len(self.audit),
                 "overall_hit_rate": self._overall_hit_rate(),
                 "world_tick": self.world.tick_count,
                 "entities": len(self.world.entities),
                 "policy": self.policy, "budget": self.budget}
        if self._action is not None:
            state["action"] = self._action.config()
        return state
