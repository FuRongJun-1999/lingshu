# -*- coding: utf-8 -*-
"""response_probe · 竞争响应假设 × 主动探测 × 四态证据闭环（Issue #110）
===========================================================================
核心定位（与 `WorldLearner` 互补共存，不破坏既有运动相容统计）：
  1. **保留既有运动相容统计**：`WorldLearner.learn()` 的 `confidence`（`abs(mean cos)`）
     衡量“轨迹在运动学上与 seek/flee 方向是否相容”，适合提出/拟合候选，但不能
     区分“A 自主 seek B”与“A、B 同受共同外力平移”这类观察等价机制；
  2. **竞争响应假设集**：从同一被动观测历史拟合多个竞争响应候选（例如 `seek` vs
     `independent_motion`），将 `motion_support`、`hypothesis_weight`、
     `response_evidence`、`evidence_state` 严格分开存储；
  3. **主动探测动作选择**：基于候选的竞争响应预测，在预算约束（`action_budget`、
     `read_budget`）与传感器噪声模型（`SensorModel`）下选择最大化预期分歧 /
     信息增益（EIG）的探测动作与测量实体（随几何旋转自适应，不写死动作模板）；
  4. **四态证据状态机**（对接 Issue #110 与《维生系统功能路线架构 v0.2》）：
       - `observationally_compatible`：仅被动轨迹相容，尚未经主动干预检验；
       - `response_supported`：经主动干预与测量后，在当前封闭候选集与噪声门槛内
         获得唯一支持（审计标签显式标注 `response_supported_within_candidates`，
         不冒充开放世界一般因果确认）；
       - `unresolved`：未探测、预算不足、探测无分歧（如静止/共线/读非响应者）或
         测量噪声过高导致候选未决；
       - `contradicted`：候选预测与实测响应冲突（单个候选被证伪，或全候选失配
         `out_of_model` / 先验冲突 `prior_conflict`，触发候选重开并保留未解释容量）。

纯标准库 · 零外部依赖 · 逐位确定性（D-005）。
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .coord_contract import apply_move
from .world_learner import LNNode, WorldLearner

PARAM_VERSION = "wm.competing-response/1"

#: 四态证据状态机（Issue #110 × 《维生系统功能路线架构 v0.2》）
EVIDENCE_STATE_COMPATIBLE = "observationally_compatible"
EVIDENCE_STATE_SUPPORTED = "response_supported"
EVIDENCE_STATE_UNRESOLVED = "unresolved"
EVIDENCE_STATE_CONTRADICTED = "contradicted"

EVIDENCE_STATES: Tuple[str, ...] = (
    EVIDENCE_STATE_COMPATIBLE,
    EVIDENCE_STATE_SUPPORTED,
    EVIDENCE_STATE_UNRESOLVED,
    EVIDENCE_STATE_CONTRADICTED,
)

#: 闭集作用域说明（#110 第 3 节与 2026-11-13 维护者复核口径）
CANDIDATE_SUPPORTED_SCOPE_LABEL = "response_supported_within_candidates"

#: 标准研究原语动作集（单位：世界坐标位移 (dx, dz)）
DEFAULT_ACTIONS: Dict[str, Tuple[float, float]] = {
    "passive": (0.0, 0.0),
    "x_plus": (6.0, 0.0),
    "x_minus": (-6.0, 0.0),
    "z_plus": (0.0, 6.0),
    "z_minus": (0.0, -6.0),
}

DEFAULT_CANDIDATE_FAMILIES: Tuple[str, ...] = ("seek", "independent_motion")
KNOWN_CANDIDATE_FAMILIES: Tuple[str, ...] = (
    "seek",
    "flee",
    "independent_motion",
    "delayed_seek",
)


# ---------------------------------------------------------------------------
# Fail-closed 参数门禁（严防 NaN / ±Inf / bool 静默通过）
# ---------------------------------------------------------------------------

def _require_int(name: str, value: Any, *, min_value: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("%s 必须是 int（拒绝 bool/非整数），收到 %r" % (name, value))
    if min_value is not None and value < min_value:
        raise ValueError("%s 必须 >= %d，收到 %r" % (name, min_value, value))
    return value


def _require_finite_float(
    name: str,
    value: Any,
    *,
    min_value: Optional[float] = None,
    min_exclusive: Optional[float] = None,
    max_value: Optional[float] = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("%s 必须是有限实数（拒绝 bool/非数值），收到 %r" % (name, value))
    fval = float(value)
    if not math.isfinite(fval):
        raise ValueError("%s 必须是有限数值（拒绝 NaN/Inf），收到 %r" % (name, value))
    if min_value is not None and fval < min_value:
        raise ValueError("%s 必须 >= %r，收到 %r" % (name, min_value, value))
    if min_exclusive is not None and fval <= min_exclusive:
        raise ValueError("%s 必须 > %r，收到 %r" % (name, min_exclusive, value))
    if max_value is not None and fval > max_value:
        raise ValueError("%s 必须 <= %r，收到 %r" % (name, max_value, value))
    return fval


def _require_pos(name: str, pos: Any) -> Tuple[float, float, float]:
    if isinstance(pos, (str, bytes, dict)):
        raise TypeError("%s 必须是三维坐标序列 (x, y, z)，收到 %r" % (name, pos))
    try:
        items = list(pos)
    except TypeError as exc:
        raise TypeError("%s 必须是三维坐标序列 (x, y, z)，收到 %r" % (name, pos)) from exc
    if len(items) != 3:
        raise ValueError("%s 必须是三维坐标 (x, y, z)，收到 %r" % (name, pos))
    return (
        _require_finite_float("%s[0]" % name, items[0]),
        _require_finite_float("%s[1]" % name, items[1]),
        _require_finite_float("%s[2]" % name, items[2]),
    )


def _require_actions(
    allowed_actions: Mapping[str, Sequence[Any]],
) -> Dict[str, Tuple[float, float]]:
    if not isinstance(allowed_actions, Mapping) or not allowed_actions:
        raise ValueError("allowed_actions 必须是非空动作映射")
    out: Dict[str, Tuple[float, float]] = {}
    for name, delta in allowed_actions.items():
        if not isinstance(name, str) or not name.strip():
            raise TypeError("动作名必须是非空字符串，收到 %r" % (name,))
        if isinstance(delta, (str, bytes, dict)):
            raise TypeError("动作 %r 的位移必须是 (dx, dz)，收到 %r" % (name, delta))
        try:
            pair = list(delta)
        except TypeError as exc:
            raise TypeError("动作 %r 的位移必须是 (dx, dz)，收到 %r" % (name, delta)) from exc
        if len(pair) != 2:
            raise ValueError("动作 %r 的位移必须是二元组 (dx, dz)，收到 %r" % (name, delta))
        out[name] = (
            _require_finite_float("action[%s].dx" % name, pair[0]),
            _require_finite_float("action[%s].dz" % name, pair[1]),
        )
    return out


def _require_weights(
    weights: Mapping[str, Any],
    *,
    expected_keys: Optional[Iterable[str]] = None,
    allow_zero_sum: bool = False,
) -> Dict[str, float]:
    if not isinstance(weights, Mapping) or not weights:
        raise ValueError("weights 必须是非空映射")
    parsed: Dict[str, float] = {}
    for key, val in weights.items():
        if not isinstance(key, str) or not key.strip():
            raise TypeError("候选名必须是非空字符串，收到 %r" % (key,))
        parsed[key] = _require_finite_float("weights[%s]" % key, val, min_value=0.0)
    if expected_keys is not None:
        exp = list(expected_keys)
        if set(parsed) != set(exp):
            raise ValueError("weights 键集 %r 与候选集 %r 不一致" % (sorted(parsed), sorted(exp)))
        parsed = {k: parsed[k] for k in exp}
    total = sum(parsed.values())
    if not math.isfinite(total):
        raise ValueError("weights 之和必须有限")
    if total <= 0.0:
        if allow_zero_sum:
            return parsed
        raise ValueError("weights 之和必须 > 0，收到 %r" % (parsed,))
    return {k: v / total for k, v in parsed.items()}


def _safe_dist(a: Sequence[float], b: Sequence[float]) -> float:
    pa = _require_pos("a", a)
    pb = _require_pos("b", b)
    try:
        d = math.dist(pa, pb)
    except OverflowError as exc:
        raise ValueError("坐标差溢出无法计算距离：%r vs %r" % (pa, pb)) from exc
    if not math.isfinite(d):
        raise ValueError("坐标距离非有限：%r" % (d,))
    return d


def advance_position(
    pos: Sequence[float],
    dx: float,
    dz: float,
    size: float = 64.0,
) -> Tuple[float, float, float]:
    """沿平面位移 (dx, dz) 推进一位，经唯一坐标真源 `coord_contract.apply_move` 钳制量化。"""
    p = _require_pos("pos", pos)
    fdx = _require_finite_float("dx", dx)
    fdz = _require_finite_float("dz", dz)
    fsize = _require_finite_float("size", size, min_exclusive=1.0)
    return apply_move(p, 1.0, (fdx, 0.0, fdz), fsize)


# ---------------------------------------------------------------------------
# 核心数据结构
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SensorModel:
    """传感器观测模型（显式声明噪声与判决门槛，不隐式默认零误差）。"""

    sigma: float = 0.0
    tolerance: float = 1e-6
    support_posterior_threshold: float = 0.90
    min_snr: float = 2.0
    contradiction_z: float = 4.0
    support_z_max: float = 3.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "sigma", _require_finite_float("sigma", self.sigma, min_value=0.0)
        )
        object.__setattr__(
            self, "tolerance", _require_finite_float("tolerance", self.tolerance, min_value=0.0)
        )
        object.__setattr__(
            self,
            "support_posterior_threshold",
            _require_finite_float(
                "support_posterior_threshold",
                self.support_posterior_threshold,
                min_exclusive=0.5,
                max_value=1.0,
            ),
        )
        object.__setattr__(
            self, "min_snr", _require_finite_float("min_snr", self.min_snr, min_exclusive=0.0)
        )
        object.__setattr__(
            self,
            "contradiction_z",
            _require_finite_float("contradiction_z", self.contradiction_z, min_exclusive=0.0),
        )
        object.__setattr__(
            self,
            "support_z_max",
            _require_finite_float("support_z_max", self.support_z_max, min_exclusive=0.0),
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "sigma": self.sigma,
            "tolerance": self.tolerance,
            "support_posterior_threshold": self.support_posterior_threshold,
            "min_snr": self.min_snr,
            "contradiction_z": self.contradiction_z,
            "support_z_max": self.support_z_max,
        }


@dataclass
class ResponseCandidate:
    """单个竞争响应假设，严格分离运动相容分、候选权重与响应证据状态。"""

    name: str
    family: str
    actor_id: str
    driver_id: str
    motion_support: float
    hypothesis_weight: float
    fitted_model: Dict[str, Any]
    history_snapshot: List[Dict[str, Any]]
    last_positions: Dict[str, Tuple[float, float, float]]
    categories: Dict[str, str]
    driver_delta: Tuple[float, float]
    world_size: float = 64.0
    evidence_state: str = EVIDENCE_STATE_COMPATIBLE
    response_evidence: Optional[Dict[str, Any]] = None
    param_version: str = PARAM_VERSION


@dataclass
class CandidateProposal:
    """`propose()` 的输出：竞争候选集 + 各自运动相容分 + 初始权重 + 证据状态。"""

    candidates: Dict[str, ResponseCandidate]
    weights: Dict[str, float]
    motion_support: Dict[str, float]
    evidence_state: str
    hypothesis_states: Dict[str, str]
    world_learner_model: Dict[str, Any]
    passive_models_identical: bool
    last_positions: Dict[str, Tuple[float, float, float]]
    driver_delta: Tuple[float, float]
    actor_id: str
    driver_id: str
    world_size: float
    param_version: str = PARAM_VERSION


@dataclass
class ProbePlan:
    """`choose_probe()` 的输出：选定探测动作、待测实体、预期分歧/EIG 与审计表。"""

    action: Optional[str]
    action_delta: Optional[Tuple[float, float]]
    action_cost: float
    read_entity: Optional[str]
    eig_bits: float
    separation: float
    informative: bool
    budget_sufficient: bool
    read_budget: int
    action_budget: int
    predictions: Dict[str, Dict[str, Tuple[float, float, float]]]
    candidates_table: List[Dict[str, Any]]
    weights_snapshot: Dict[str, float]
    execution_conditions: Dict[str, Any]
    state_snapshot: Dict[str, Tuple[float, float, float]]
    candidate_objects: Dict[str, ResponseCandidate] = field(repr=False)
    reason: str = "ok"


@dataclass
class EvidenceUpdate:
    """`update_evidence()` / `reconcile_follow_up()` 的四态证据更新记录。"""

    evidence_state: str
    hypothesis_states: Dict[str, str]
    weights: Dict[str, float]
    motion_support: Dict[str, float]
    response_evidence: Dict[str, Any]
    residuals: Dict[str, float]
    z_scores: Optional[Dict[str, float]]
    within_candidates: bool
    response_scope_label: Optional[str]
    out_of_model: bool
    prior_conflict: bool
    reopen_required: bool
    unexplained_capacity_preserved: bool
    unexplained_residual: Optional[float]
    receipt: Dict[str, Any]


# ---------------------------------------------------------------------------
# 内部：纯观测视图与 WorldLearner 拟合/预测桥接
# ---------------------------------------------------------------------------

class _ObservationNamespace:
    """只暴露 (category, pos) 的实体视图——严格不接触真值 behavior/goal/mechanism。"""

    __slots__ = ("category", "pos")

    def __init__(self, category: str, pos: Tuple[float, float, float]) -> None:
        self.category = category
        self.pos = pos


class _ObservationWorldView:
    """给 `WorldLearner` 注入的纯被动观测视图（缸中之脑姿态）。"""

    def __init__(self, size: int) -> None:
        self.world = type("_WorldExtent", (), {"size": size})()
        self.entities: Dict[str, _ObservationNamespace] = {}

    def set_frame(self, frame: Mapping[str, Mapping[str, Any]]) -> None:
        entities: Dict[str, _ObservationNamespace] = {}
        for eid, info in frame.items():
            if not isinstance(eid, str) or not eid.strip():
                raise TypeError("实体 id 必须是非空字符串，收到 %r" % (eid,))
            if not isinstance(info, Mapping) or "pos" not in info:
                raise TypeError("观测帧实体 %r 缺少 pos 字段：%r" % (eid, info))
            cat = info.get("category", "entity")
            if not isinstance(cat, str) or not cat.strip():
                raise TypeError("观测帧实体 %r 的 category 必须是非空字符串，收到 %r" % (eid, cat))
            entities[eid] = _ObservationNamespace(cat, _require_pos("obs[%s].pos" % eid, info["pos"]))
        self.entities = entities


def _normalize_observation_history(
    observation_history: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Dict[str, Any]]]:
    if not isinstance(observation_history, Sequence) or isinstance(observation_history, (str, bytes)):
        raise TypeError("observation_history 必须是非空观测序列")
    if len(observation_history) < 2:
        raise ValueError("observation_history 至少需要 2 帧以估计位移，收到 %d 帧" % len(observation_history))
    frames: List[Dict[str, Dict[str, Any]]] = []
    for idx, raw in enumerate(observation_history):
        if not isinstance(raw, Mapping) or not raw:
            raise ValueError("第 %d 帧观测为空或非映射" % idx)
        ent_map = raw["entities"] if ("entities" in raw and isinstance(raw["entities"], Mapping)) else raw
        if not ent_map:
            raise ValueError("第 %d 帧实体映射为空" % idx)
        cleaned: Dict[str, Dict[str, Any]] = {}
        for eid, info in ent_map.items():
            if not isinstance(eid, str) or not eid.strip():
                raise TypeError("第 %d 帧实体 id 必须是非空字符串，收到 %r" % (idx, eid))
            if not isinstance(info, Mapping) or "pos" not in info:
                raise TypeError("第 %d 帧实体 %r 缺少 pos" % (idx, eid))
            cat = info.get("category", "entity")
            if not isinstance(cat, str) or not cat.strip():
                raise TypeError("第 %d 帧实体 %r 的 category 必须是非空字符串" % (idx, cat))
            cleaned[eid] = {
                "category": cat,
                "pos": _require_pos("frame[%d][%s].pos" % (idx, eid), info["pos"]),
            }
        frames.append(cleaned)
    return frames


def _fit_world_learner(
    frames: List[Dict[str, Dict[str, Any]]],
    *,
    window: int,
    size: float,
) -> WorldLearner:
    world_size = max(2, int(round(size)))
    view = _ObservationWorldView(size=world_size)
    learner = WorldLearner(size=world_size, window=window, world=view)
    for frame in frames:
        view.set_frame(frame)
        learner.observe()
    learner.learn(window=window)
    return learner


def _entropy_bits(probs: Iterable[float]) -> float:
    return -sum(p * math.log2(p) for p in probs if p > 0.0)


def _exact_noiseless_eig(
    predictions: Mapping[str, Mapping[str, Tuple[float, float, float]]],
    entity: str,
    prior: Mapping[str, float],
    tolerance: float,
) -> float:
    """无噪声（或容差离散分桶）条件下的精确互信息 EIG（单位：bit）。"""
    groups: List[Tuple[Tuple[float, float, float], float]] = []
    for name, w in prior.items():
        if w <= 0.0:
            continue
        pos = predictions[name][entity]
        placed = False
        for idx, (rep, mass) in enumerate(groups):
            if _safe_dist(pos, rep) <= tolerance:
                groups[idx] = (rep, mass + w)
                placed = True
                break
        if not placed:
            groups.append((pos, w))
    if len(groups) <= 1:
        return 0.0
    prior_h = _entropy_bits(prior.values())
    post_h = 0.0
    for rep, mass in groups:
        if mass <= 0.0:
            continue
        bucket = [w / mass for name, w in prior.items() if w > 0.0 and _safe_dist(predictions[name][entity], rep) <= tolerance]
        post_h += mass * _entropy_bits(bucket)
    return max(0.0, prior_h - post_h)


def _gaussian_eig_1d_exact(delta: float, w0: float, w1: float, sigma: float) -> float:
    """二候选在各向同性高斯观测噪声下的精确 1D 投影互信息（确定性求积）。

    沿连接两候选均值向量方向的一维充分统计量完全捕获似然比，
    与维数正交的噪声分量在贝叶斯后验中严格抵消。
    """
    if delta <= 0.0 or w0 <= 0.0 or w1 <= 0.0:
        return 0.0
    total = w0 + w1
    p0, p1 = w0 / total, w1 / total
    prior_h = _entropy_bits((p0, p1))
    lo = min(0.0, delta) - 7.0 * sigma
    hi = max(0.0, delta) + 7.0 * sigma
    steps = 240
    step = (hi - lo) / steps
    inv_var = 1.0 / (2.0 * sigma * sigma)
    norm = 1.0 / (math.sqrt(2.0 * math.pi) * sigma)
    cond_h = 0.0
    for i in range(steps):
        x = lo + (i + 0.5) * step
        # 高斯空间似然密度（非时间衰减核）：先求对数核值再取 exp
        arg0 = -(x * x) * inv_var
        arg1 = -((x - delta) ** 2) * inv_var
        l0 = p0 * norm * math.exp(arg0)
        l1 = p1 * norm * math.exp(arg1)
        px = l0 + l1
        if px <= 1e-300:
            continue
        q0 = l0 / px
        q1 = l1 / px
        cond_h += px * step * _entropy_bits((q0, q1))
    return max(0.0, prior_h - cond_h)


def _gaussian_eig_approx(
    predictions: Mapping[str, Mapping[str, Tuple[float, float, float]]],
    entity: str,
    prior: Mapping[str, float],
    sensor_model: SensorModel,
) -> float:
    active = [k for k, w in prior.items() if w > 0.0]
    if len(active) <= 1:
        return 0.0
    if len(active) == 2:
        n0, n1 = active[0], active[1]
        delta = _safe_dist(predictions[n0][entity], predictions[n1][entity])
        return _gaussian_eig_1d_exact(delta, prior[n0], prior[n1], sensor_model.sigma)
    # 超过 2 个正先验候选：用最小相邻分离与成对高斯通道保守下界估计可辨识信息增益
    # （并由 SNR 缩放，保证无分歧时严格为 0，分离越大单调不减）。
    noiseless = _exact_noiseless_eig(predictions, entity, prior, sensor_model.tolerance)
    if noiseless <= 0.0:
        return 0.0
    pair_eigs = []
    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            a, b = active[i], active[j]
            d = _safe_dist(predictions[a][entity], predictions[b][entity])
            pair_eigs.append(_gaussian_eig_1d_exact(d, prior[a], prior[b], sensor_model.sigma))
    return min(noiseless, max(pair_eigs) if pair_eigs else 0.0)


def gaussian_posterior(
    predictions: Mapping[str, Mapping[str, Sequence[float]]],
    entity: str,
    observed: Sequence[float],
    prior: Mapping[str, Any],
    sigma: float,
) -> Dict[str, float]:
    """已知各向同性高斯观测噪声（σ > 0）下的精确贝叶斯候选后验（log-sum-exp）。"""
    fsigma = _require_finite_float("sigma", sigma, min_exclusive=0.0)
    obs_pos = _require_pos("observed", observed)
    if not isinstance(predictions, Mapping) or not predictions:
        raise ValueError("predictions 必须是非空映射")
    names = list(predictions.keys())
    clean_prior = _require_weights(prior, expected_keys=names)
    inv_2var = 1.0 / (2.0 * fsigma * fsigma)
    logs: Dict[str, float] = {}
    for name in names:
        if entity not in predictions[name]:
            raise KeyError("候选 %r 的预测缺少实体 %r" % (name, entity))
        pred_pos = _require_pos("predictions[%s][%s]" % (name, entity), predictions[name][entity])
        w = clean_prior[name]
        if w <= 0.0:
            logs[name] = -math.inf
        else:
            sq_err = sum((a - b) ** 2 for a, b in zip(pred_pos, obs_pos))
            logs[name] = math.log(w) - sq_err * inv_2var
    max_log = max(logs.values())
    weights = {
        name: (math.exp(val - max_log) if math.isfinite(val) else 0.0)
        for name, val in logs.items()
    }
    z = sum(weights.values())
    if z <= 0.0 or not math.isfinite(z):
        raise ValueError("高斯后验归一化因子异常：%r" % (z,))
    return {name: weights[name] / z for name in names}


# ---------------------------------------------------------------------------
# Issue #110 §4 最小公开接口：propose / forecast / choose_probe / update_evidence
# ---------------------------------------------------------------------------

def propose(
    observation_history: Sequence[Mapping[str, Any]],
    *,
    window: int = 6,
    size: float = 64.0,
    prior: Optional[Mapping[str, Any]] = None,
    candidate_families: Optional[Sequence[str]] = None,
    actor_id: Optional[str] = None,
    driver_id: Optional[str] = None,
) -> CandidateProposal:
    """从纯被动观测提出竞争响应假设集（不篡改 WorldLearner 运动相容统计）。

    每个候选仅共享同一套由真实 `WorldLearner` 从 `(category, pos)` 拟合出的参数，
    初始 `evidence_state` 均为 `observationally_compatible`，`response_evidence=None`。
    """
    w = _require_int("window", window, min_value=2)
    fsize = _require_finite_float("size", size, min_exclusive=1.0)
    frames = _normalize_observation_history(observation_history)
    learner = _fit_world_learner(frames, window=w, size=fsize)
    base_model = copy.deepcopy(learner.model)

    entities = list(frames[-1].keys())
    relations = base_model.get("relations", [])
    if actor_id is not None or driver_id is not None:
        if not isinstance(actor_id, str) or actor_id not in frames[-1]:
            raise ValueError("actor_id %r 不在观测实体中" % (actor_id,))
        if not isinstance(driver_id, str) or driver_id not in frames[-1]:
            raise ValueError("driver_id %r 不在观测实体中" % (driver_id,))
        if actor_id == driver_id:
            raise ValueError("actor_id 与 driver_id 不能相同：%r" % (actor_id,))
        chosen_actor, chosen_driver = actor_id, driver_id
        primary_rel = next(
            (r for r in relations if r["source"] == chosen_actor and r["target"] == chosen_driver),
            None,
        )
    elif relations:
        primary_rel = max(relations, key=lambda r: float(r.get("confidence", 0.0)))
        chosen_actor = primary_rel["source"]
        chosen_driver = primary_rel["target"]
    elif len(entities) >= 2:
        chosen_actor, chosen_driver = entities[0], entities[1]
        primary_rel = None
    else:
        raise ValueError("至少需要 2 个可观测实体以构造竞争响应候选，当前：%r" % (entities,))

    if candidate_families is None:
        rel_family = primary_rel["relation"] if primary_rel is not None else "seek"
        families = [rel_family, "independent_motion"]
    else:
        families = []
        for fam in candidate_families:
            if not isinstance(fam, str) or fam not in KNOWN_CANDIDATE_FAMILIES:
                raise ValueError("未知候选族 %r（可用：%r）" % (fam, KNOWN_CANDIDATE_FAMILIES))
            if fam not in families:
                families.append(fam)
        if not families:
            raise ValueError("candidate_families 不能为空")

    if prior is None:
        equal = 1.0 / len(families)
        weights = {fam: equal for fam in families}
    else:
        weights = _require_weights(prior, expected_keys=families)

    last_positions = {eid: tuple(info["pos"]) for eid, info in frames[-1].items()}
    categories = {eid: str(info["category"]) for eid, info in frames[-1].items()}
    prev_driver = frames[-2].get(chosen_driver, {}).get("pos", last_positions[chosen_driver])
    driver_delta = (
        round(last_positions[chosen_driver][0] - prev_driver[0], 6),
        round(last_positions[chosen_driver][2] - prev_driver[2], 6),
    )

    rel_conf = float(primary_rel.get("confidence", 0.0)) if primary_rel is not None else 0.0
    actor_pers = float(base_model.get("per_entity", {}).get(chosen_actor, {}).get("persistence", 0.0))

    candidates: Dict[str, ResponseCandidate] = {}
    motion_support: Dict[str, float] = {}
    hypothesis_states: Dict[str, str] = {}

    for fam in families:
        cand_model = copy.deepcopy(base_model)
        if fam == "independent_motion":
            cand_model["relations"] = []
            cand_model["stochastic_targets"] = []
            ms = actor_pers
        elif fam in ("seek", "flee", "delayed_seek"):
            if not any(
                r["source"] == chosen_actor and r["target"] == chosen_driver
                for r in cand_model.get("relations", [])
            ):
                cand_model["relations"] = [
                    {
                        "source": chosen_actor,
                        "relation": "flee" if fam == "flee" else "seek",
                        "target": chosen_driver,
                        "confidence": round(max(rel_conf, actor_pers), 3),
                    }
                ]
            ms = rel_conf if primary_rel is not None else actor_pers
        else:
            ms = actor_pers
        cand_name = fam
        candidates[cand_name] = ResponseCandidate(
            name=cand_name,
            family=fam,
            actor_id=chosen_actor,
            driver_id=chosen_driver,
            motion_support=ms,
            hypothesis_weight=weights[cand_name],
            fitted_model=cand_model,
            history_snapshot=copy.deepcopy(learner.history),
            last_positions=dict(last_positions),
            categories=dict(categories),
            driver_delta=driver_delta,
            world_size=fsize,
            evidence_state=EVIDENCE_STATE_COMPATIBLE,
            response_evidence=None,
            param_version=PARAM_VERSION,
        )
        motion_support[cand_name] = ms
        hypothesis_states[cand_name] = EVIDENCE_STATE_COMPATIBLE

    return CandidateProposal(
        candidates=candidates,
        weights=weights,
        motion_support=motion_support,
        evidence_state=EVIDENCE_STATE_COMPATIBLE,
        hypothesis_states=hypothesis_states,
        world_learner_model=base_model,
        passive_models_identical=True,
        last_positions=last_positions,
        driver_delta=driver_delta,
        actor_id=chosen_actor,
        driver_id=chosen_driver,
        world_size=fsize,
        param_version=PARAM_VERSION,
    )


def _learner_from_candidate(
    candidate: ResponseCandidate,
    state: Mapping[str, Sequence[float]],
    driver_override: Optional[Tuple[float, float, float]] = None,
) -> WorldLearner:
    world_size = max(2, int(round(candidate.world_size)))
    view = _ObservationWorldView(size=world_size)
    learner = WorldLearner(size=world_size, window=6, world=view)
    learner.history = copy.deepcopy(candidate.history_snapshot)
    learner.tick = learner.history[-1]["tick"] if learner.history else 0
    learner.model = copy.deepcopy(candidate.fitted_model)
    for eid, pos in state.items():
        p = _require_pos("state[%s]" % eid, pos)
        cat = candidate.categories.get(eid, "entity")
        view.entities[eid] = _ObservationNamespace(cat, p)
        learner.nodes[eid] = LNNode(
            eid=eid, category=cat, pos=p, first_seen=1, last_seen=learner.tick
        )
    if driver_override is not None and candidate.driver_id in learner.nodes:
        learner.nodes[candidate.driver_id].pos = driver_override
    return learner


def forecast(
    candidate: ResponseCandidate,
    state: Mapping[str, Sequence[float]],
    action: str | Sequence[float],
    *,
    allowed_actions: Optional[Mapping[str, Sequence[float]]] = None,
    size: Optional[float] = None,
) -> Dict[str, Tuple[float, float, float]]:
    """根据候选机制与已宣告干预动作预测下一步全部实体的观测位置。"""
    if not isinstance(candidate, ResponseCandidate):
        raise TypeError("candidate 必须是 ResponseCandidate 实例，收到 %r" % (candidate,))
    if not isinstance(state, Mapping) or not state:
        raise ValueError("state 必须是非空实体位置映射")
    fsize = candidate.world_size if size is None else _require_finite_float(
        "size", size, min_exclusive=1.0
    )
    clean_state = {eid: _require_pos("state[%s]" % eid, pos) for eid, pos in state.items()}
    if candidate.driver_id not in clean_state or candidate.actor_id not in clean_state:
        raise ValueError("state 缺少候选所需实体 %r / %r" % (candidate.actor_id, candidate.driver_id))

    if isinstance(action, str):
        action_map = _require_actions(allowed_actions or DEFAULT_ACTIONS)
        if action not in action_map:
            raise ValueError("未知动作 %r（可用：%r）" % (action, sorted(action_map)))
        dx, dz = action_map[action]
    else:
        if isinstance(action, (bytes, dict)):
            raise TypeError("action 必须是动作名或 (dx, dz)，收到 %r" % (action,))
        try:
            pair = list(action)
        except TypeError as exc:
            raise TypeError("action 必须是动作名或 (dx, dz)，收到 %r" % (action,)) from exc
        if len(pair) != 2:
            raise ValueError("action 位移必须是 (dx, dz)，收到 %r" % (action,))
        dx = _require_finite_float("action.dx", pair[0])
        dz = _require_finite_float("action.dz", pair[1])

    driver_pre = clean_state[candidate.driver_id]
    intervened_driver = advance_position(driver_pre, dx, dz, size=fsize)
    driver_next = advance_position(
        intervened_driver,
        candidate.driver_delta[0],
        candidate.driver_delta[1],
        size=fsize,
    )

    if candidate.family == "delayed_seek":
        # 漏真延迟响应：A 在本拍仍读干预前的 driver 位置（下一拍才响应干预后位置）
        driver_seen_by_actor = driver_pre
    else:
        driver_seen_by_actor = intervened_driver

    learner = _learner_from_candidate(candidate, clean_state, driver_override=driver_seen_by_actor)
    pred = learner.predict(horizon=1)["predictions"]

    out: Dict[str, Tuple[float, float, float]] = {}
    for eid in clean_state:
        if eid == candidate.driver_id:
            out[eid] = driver_next
        elif eid in pred:
            out[eid] = _require_pos("pred[%s]" % eid, pred[eid]["predicted"])
        else:
            out[eid] = clean_state[eid]
    return out


def choose_probe(
    candidates: Mapping[str, ResponseCandidate],
    weights: Mapping[str, Any],
    allowed_actions: Mapping[str, Sequence[float]],
    read_budget: int,
    sensor_model: SensorModel,
    *,
    action_budget: int = 1,
    state: Optional[Mapping[str, Sequence[float]]] = None,
) -> ProbePlan:
    """在动作/读取预算与传感器模型下选择最大化候选区分信息的探测计划。"""
    if not isinstance(candidates, Mapping) or not candidates:
        raise ValueError("candidates 必须是非空映射")
    if not isinstance(sensor_model, SensorModel):
        raise TypeError("sensor_model 必须是 SensorModel 实例，收到 %r" % (sensor_model,))
    rb = _require_int("read_budget", read_budget, min_value=0)
    ab = _require_int("action_budget", action_budget, min_value=0)
    actions = _require_actions(allowed_actions)
    cand_names = list(candidates.keys())
    norm_weights = _require_weights(weights, expected_keys=cand_names)

    first_cand = candidates[cand_names[0]]
    cur_state = (
        {eid: _require_pos("state[%s]" % eid, pos) for eid, pos in state.items()}
        if state is not None
        else dict(first_cand.last_positions)
    )
    entities = list(cur_state.keys())

    if rb <= 0:
        return ProbePlan(
            action=None,
            action_delta=None,
            action_cost=0.0,
            read_entity=None,
            eig_bits=0.0,
            separation=0.0,
            informative=False,
            budget_sufficient=False,
            read_budget=rb,
            action_budget=ab,
            predictions={},
            candidates_table=[],
            weights_snapshot=norm_weights,
            execution_conditions={
                "driver_id": first_cand.driver_id,
                "actor_id": first_cand.actor_id,
                "param_version": first_cand.param_version,
                "world_size": first_cand.world_size,
                "read_budget": rb,
                "action_budget": ab,
            },
            state_snapshot=cur_state,
            candidate_objects=dict(candidates),
            reason="insufficient_read_budget",
        )

    usable_actions = {
        aname: delta
        for aname, delta in actions.items()
        if ab > 0 or (delta[0] == 0.0 and delta[1] == 0.0)
    }
    if not usable_actions:
        return ProbePlan(
            action=None,
            action_delta=None,
            action_cost=0.0,
            read_entity=None,
            eig_bits=0.0,
            separation=0.0,
            informative=False,
            budget_sufficient=False,
            read_budget=rb,
            action_budget=ab,
            predictions={},
            candidates_table=[],
            weights_snapshot=norm_weights,
            execution_conditions={
                "driver_id": first_cand.driver_id,
                "actor_id": first_cand.actor_id,
                "param_version": first_cand.param_version,
                "world_size": first_cand.world_size,
                "read_budget": rb,
                "action_budget": ab,
            },
            state_snapshot=cur_state,
            candidate_objects=dict(candidates),
            reason="insufficient_action_budget",
        )

    predictions_by_action: Dict[str, Dict[str, Dict[str, Tuple[float, float, float]]]] = {}
    rows: List[Dict[str, Any]] = []
    for aname, delta in usable_actions.items():
        preds = {
            cname: forecast(
                candidates[cname],
                cur_state,
                aname,
                allowed_actions=usable_actions,
            )
            for cname in cand_names
        }
        predictions_by_action[aname] = preds
        cost = math.hypot(delta[0], delta[1])
        for entity in entities:
            active_preds = [preds[cname][entity] for cname in cand_names if norm_weights[cname] > 0.0]
            sep = 0.0
            for i in range(len(active_preds)):
                for j in range(i + 1, len(active_preds)):
                    sep = max(sep, _safe_dist(active_preds[i], active_preds[j]))
            if sensor_model.sigma == 0.0:
                eig = _exact_noiseless_eig(preds, entity, norm_weights, sensor_model.tolerance)
            else:
                eig = _gaussian_eig_approx(preds, entity, norm_weights, sensor_model)
            rows.append(
                {
                    "action": aname,
                    "action_delta": delta,
                    "read": entity,
                    "EIG_bits": eig,
                    "separation": sep,
                    "action_cost": cost,
                }
            )

    best = max(rows, key=lambda r: (round(r["EIG_bits"], 12), round(r["separation"], 12), -r["action_cost"]))
    informative = best["separation"] > sensor_model.tolerance and best["EIG_bits"] > 1e-12
    chosen_action = best["action"]

    return ProbePlan(
        action=chosen_action,
        action_delta=best["action_delta"],
        action_cost=best["action_cost"],
        read_entity=best["read"],
        eig_bits=best["EIG_bits"],
        separation=best["separation"],
        informative=informative,
        budget_sufficient=True,
        read_budget=rb,
        action_budget=ab,
        predictions=predictions_by_action[chosen_action],
        candidates_table=rows,
        weights_snapshot=norm_weights,
        execution_conditions={
            "driver_id": first_cand.driver_id,
            "actor_id": first_cand.actor_id,
            "param_version": first_cand.param_version,
            "world_size": first_cand.world_size,
            "read_budget": rb,
            "action_budget": ab,
            "independent_actuator_assumed": True,
        },
        state_snapshot=cur_state,
        candidate_objects=dict(candidates),
        reason="ok" if informative else "no_informative_probe",
    )


def _build_receipt(
    plan: ProbePlan,
    observed: Optional[Tuple[float, float, float]],
    sensor_model: SensorModel,
    read_tick: Optional[int],
    action_executed: bool,
) -> Dict[str, Any]:
    return {
        "param_version": plan.execution_conditions.get("param_version", PARAM_VERSION),
        "action": plan.action,
        "action_delta": list(plan.action_delta) if plan.action_delta is not None else None,
        "action_cost": plan.action_cost,
        "action_executed": bool(action_executed),
        "execution_conditions": dict(plan.execution_conditions),
        "read_entity": plan.read_entity,
        "read_tick": read_tick,
        "observed_position": list(observed) if observed is not None else None,
        "sensor_model": sensor_model.to_dict(),
        "read_budget": plan.read_budget,
        "action_budget": plan.action_budget,
        "eig_bits": plan.eig_bits,
        "separation": plan.separation,
    }


def _update_noiseless(
    cand_names: List[str],
    prior: Dict[str, float],
    residuals: Dict[str, float],
    sensor_model: SensorModel,
) -> Tuple[str, Dict[str, str], Dict[str, float], bool, bool, bool, Optional[float]]:
    matching = [c for c in cand_names if residuals[c] <= sensor_model.tolerance]
    if not matching:
        hyp_states = {c: EVIDENCE_STATE_CONTRADICTED for c in cand_names}
        weights = {c: 0.0 for c in cand_names}
        min_res = min(residuals.values())
        return EVIDENCE_STATE_CONTRADICTED, hyp_states, weights, True, False, True, min_res

    mass = sum(prior[c] for c in matching)
    if mass <= 0.0:
        hyp_states = {
            c: (EVIDENCE_STATE_UNRESOLVED if c in matching else EVIDENCE_STATE_CONTRADICTED)
            for c in cand_names
        }
        return (
            EVIDENCE_STATE_CONTRADICTED,
            hyp_states,
            dict(prior),
            False,
            True,
            True,
            min(residuals[c] for c in matching),
        )

    weights = {c: (prior[c] / mass if c in matching else 0.0) for c in cand_names}
    positive_matching = [c for c in matching if prior[c] > 0.0]
    if len(positive_matching) == 1:
        winner = positive_matching[0]
        hyp_states = {
            c: (EVIDENCE_STATE_SUPPORTED if c == winner else EVIDENCE_STATE_CONTRADICTED)
            for c in cand_names
        }
        return EVIDENCE_STATE_SUPPORTED, hyp_states, weights, False, False, False, None

    hyp_states = {
        c: (EVIDENCE_STATE_UNRESOLVED if c in matching else EVIDENCE_STATE_CONTRADICTED)
        for c in cand_names
    }
    return EVIDENCE_STATE_UNRESOLVED, hyp_states, weights, False, False, False, None


def _update_noisy(
    plan: ProbePlan,
    cand_names: List[str],
    prior: Dict[str, float],
    observed: Tuple[float, float, float],
    residuals: Dict[str, float],
    sensor_model: SensorModel,
) -> Tuple[str, Dict[str, str], Dict[str, float], Dict[str, float], bool, bool, Optional[float]]:
    assert plan.read_entity is not None
    z_scores = {c: residuals[c] / sensor_model.sigma for c in cand_names}
    positive_cands = [c for c in cand_names if prior[c] > 0.0]

    if positive_cands and all(z_scores[c] > sensor_model.contradiction_z for c in positive_cands):
        hyp_states = {c: EVIDENCE_STATE_CONTRADICTED for c in cand_names}
        weights = {c: 0.0 for c in cand_names}
        return (
            EVIDENCE_STATE_CONTRADICTED,
            hyp_states,
            weights,
            z_scores,
            True,
            True,
            min(residuals[c] for c in positive_cands),
        )

    post = gaussian_posterior(
        plan.predictions,
        plan.read_entity,
        observed,
        prior,
        sensor_model.sigma,
    )
    snr = plan.separation / sensor_model.sigma
    winner = max(cand_names, key=lambda c: post[c])

    supported = (
        plan.informative
        and snr >= sensor_model.min_snr
        and post[winner] >= sensor_model.support_posterior_threshold
        and z_scores[winner] <= sensor_model.support_z_max
    )
    if supported:
        hyp_states: Dict[str, str] = {}
        for c in cand_names:
            if c == winner:
                hyp_states[c] = EVIDENCE_STATE_SUPPORTED
            elif (
                z_scores[c] > sensor_model.contradiction_z
                or post[c] <= (1.0 - sensor_model.support_posterior_threshold)
            ):
                hyp_states[c] = EVIDENCE_STATE_CONTRADICTED
            else:
                hyp_states[c] = EVIDENCE_STATE_UNRESOLVED
        return EVIDENCE_STATE_SUPPORTED, hyp_states, post, z_scores, False, False, None

    hyp_states = {
        c: (
            EVIDENCE_STATE_CONTRADICTED
            if z_scores[c] > sensor_model.contradiction_z
            else EVIDENCE_STATE_UNRESOLVED
        )
        for c in cand_names
    }
    return EVIDENCE_STATE_UNRESOLVED, hyp_states, post, z_scores, False, False, None


def update_evidence(
    plan: ProbePlan,
    one_observation: Optional[Sequence[float]],
    sensor_model: SensorModel,
    *,
    weights: Optional[Mapping[str, Any]] = None,
    action_executed: bool = True,
    read_tick: Optional[int] = None,
) -> EvidenceUpdate:
    """根据一次探测执行与单实体观测，按四态证据状态机更新候选权重与状态。"""
    if not isinstance(plan, ProbePlan):
        raise TypeError("plan 必须是 ProbePlan 实例，收到 %r" % (plan,))
    if not isinstance(sensor_model, SensorModel):
        raise TypeError("sensor_model 必须是 SensorModel 实例，收到 %r" % (sensor_model,))
    if not isinstance(action_executed, bool):
        raise TypeError("action_executed 必须是 bool，收到 %r" % (action_executed,))
    if read_tick is not None:
        _require_int("read_tick", read_tick, min_value=0)

    cand_names = list(plan.candidate_objects.keys())
    prior = _require_weights(
        weights if weights is not None else plan.weights_snapshot,
        expected_keys=cand_names,
    )
    motion_support = {c: plan.candidate_objects[c].motion_support for c in cand_names}

    observed = (
        _require_pos("one_observation", one_observation)
        if one_observation is not None
        else None
    )
    receipt = _build_receipt(plan, observed, sensor_model, read_tick, action_executed)

    if (
        not plan.budget_sufficient
        or plan.read_budget <= 0
        or observed is None
        or not action_executed
        or plan.read_entity is None
    ):
        hyp_states = {c: EVIDENCE_STATE_UNRESOLVED for c in cand_names}
        resp_ev = {
            "status_reason": (
                plan.reason
                if not plan.budget_sufficient
                else ("action_not_executed" if not action_executed else "no_observation")
            ),
            "informative_probe": False,
        }
        return EvidenceUpdate(
            evidence_state=EVIDENCE_STATE_UNRESOLVED,
            hypothesis_states=hyp_states,
            weights=dict(prior),
            motion_support=motion_support,
            response_evidence=resp_ev,
            residuals={},
            z_scores=None,
            within_candidates=False,
            response_scope_label=None,
            out_of_model=False,
            prior_conflict=False,
            reopen_required=False,
            unexplained_capacity_preserved=True,
            unexplained_residual=None,
            receipt=receipt,
        )

    residuals = {
        c: _safe_dist(plan.predictions[c][plan.read_entity], observed)
        for c in cand_names
    }

    z_scores: Optional[Dict[str, float]]
    if sensor_model.sigma == 0.0:
        (
            overall,
            hyp_states,
            new_weights,
            out_of_model,
            prior_conflict,
            reopen_required,
            unexplained_res,
        ) = _update_noiseless(cand_names, prior, residuals, sensor_model)
        z_scores = None
    else:
        (
            overall,
            hyp_states,
            new_weights,
            z_scores,
            out_of_model,
            reopen_required,
            unexplained_res,
        ) = _update_noisy(plan, cand_names, prior, observed, residuals, sensor_model)
        prior_conflict = False

    within_candidates = overall == EVIDENCE_STATE_SUPPORTED
    scope_label = CANDIDATE_SUPPORTED_SCOPE_LABEL if within_candidates else None
    resp_ev = {
        "action": plan.action,
        "read_entity": plan.read_entity,
        "residuals": dict(residuals),
        "z_scores": dict(z_scores) if z_scores is not None else None,
        "separation": plan.separation,
        "eig_bits": plan.eig_bits,
        "informative_probe": plan.informative,
        "within_candidates": within_candidates,
        "closed_set_disclaimer": (
            "supported_within_closed_candidate_set_only" if within_candidates else None
        ),
    }

    return EvidenceUpdate(
        evidence_state=overall,
        hypothesis_states=hyp_states,
        weights=new_weights,
        motion_support=motion_support,
        response_evidence=resp_ev,
        residuals=residuals,
        z_scores=z_scores,
        within_candidates=within_candidates,
        response_scope_label=scope_label,
        out_of_model=out_of_model,
        prior_conflict=prior_conflict,
        reopen_required=reopen_required,
        unexplained_capacity_preserved=True,
        unexplained_residual=unexplained_res,
        receipt=receipt,
    )


# ---------------------------------------------------------------------------
# 后续矛盾重开（#110 漏真反例闭环）与冻结未来外推
# ---------------------------------------------------------------------------

def reconcile_follow_up(
    previous_update: EvidenceUpdate,
    candidates: Mapping[str, ResponseCandidate],
    next_state: Mapping[str, Sequence[float]],
    next_observation: Sequence[float],
    sensor_model: SensorModel,
    *,
    read_entity: Optional[str] = None,
    action: str = "passive",
    read_tick: Optional[int] = None,
) -> EvidenceUpdate:
    """用后续实测帧复核当前候选（例如揭露漏真延迟响应 `delayed_seek` 并触发重开）。"""
    if not isinstance(previous_update, EvidenceUpdate):
        raise TypeError("previous_update 必须是 EvidenceUpdate 实例")
    if not isinstance(candidates, Mapping) or not candidates:
        raise ValueError("candidates 必须是非空映射")
    if not isinstance(sensor_model, SensorModel):
        raise TypeError("sensor_model 必须是 SensorModel 实例")
    obs = _require_pos("next_observation", next_observation)
    clean_state = {eid: _require_pos("next_state[%s]" % eid, pos) for eid, pos in next_state.items()}
    target_entity = read_entity or previous_update.receipt.get("read_entity")
    if not isinstance(target_entity, str) or not target_entity:
        raise ValueError("必须指定 read_entity")
    if read_tick is not None:
        _require_int("read_tick", read_tick, min_value=0)

    cand_names = list(candidates.keys())
    preds = {
        cname: forecast(candidates[cname], clean_state, action)
        for cname in cand_names
    }
    residuals = {c: _safe_dist(preds[c][target_entity], obs) for c in cand_names}

    prev_states = dict(previous_update.hypothesis_states)
    prev_weights = dict(previous_update.weights)
    hyp_states: Dict[str, str] = {}
    z_scores: Optional[Dict[str, float]] = None

    if sensor_model.sigma == 0.0:
        for c in cand_names:
            if prev_states.get(c) == EVIDENCE_STATE_CONTRADICTED:
                hyp_states[c] = EVIDENCE_STATE_CONTRADICTED
            elif residuals[c] <= sensor_model.tolerance:
                hyp_states[c] = prev_states.get(c, EVIDENCE_STATE_UNRESOLVED)
            else:
                hyp_states[c] = EVIDENCE_STATE_CONTRADICTED
        surviving = [c for c in cand_names if hyp_states[c] != EVIDENCE_STATE_CONTRADICTED]
    else:
        z_scores = {c: residuals[c] / sensor_model.sigma for c in cand_names}
        for c in cand_names:
            if prev_states.get(c) == EVIDENCE_STATE_CONTRADICTED:
                hyp_states[c] = EVIDENCE_STATE_CONTRADICTED
            elif z_scores[c] > sensor_model.contradiction_z:
                hyp_states[c] = EVIDENCE_STATE_CONTRADICTED
            else:
                hyp_states[c] = prev_states.get(c, EVIDENCE_STATE_UNRESOLVED)
        surviving = [c for c in cand_names if hyp_states[c] != EVIDENCE_STATE_CONTRADICTED]

    if not surviving:
        new_weights = {c: 0.0 for c in cand_names}
        overall = EVIDENCE_STATE_CONTRADICTED
        out_of_model = True
        reopen_required = True
        unexplained_res: Optional[float] = min(residuals.values())
    elif len(surviving) == 1 and prev_states.get(surviving[0]) == EVIDENCE_STATE_SUPPORTED:
        w = surviving[0]
        new_weights = {c: (1.0 if c == w else 0.0) for c in cand_names}
        overall = EVIDENCE_STATE_SUPPORTED
        out_of_model = False
        reopen_required = False
        unexplained_res = None
    else:
        mass = sum(prev_weights.get(c, 0.0) for c in surviving)
        if mass > 0.0:
            new_weights = {c: (prev_weights.get(c, 0.0) / mass if c in surviving else 0.0) for c in cand_names}
        else:
            eq = 1.0 / len(surviving)
            new_weights = {c: (eq if c in surviving else 0.0) for c in cand_names}
        overall = EVIDENCE_STATE_UNRESOLVED
        out_of_model = False
        reopen_required = False
        unexplained_res = None

    receipt = dict(previous_update.receipt)
    receipt["follow_up_action"] = action
    receipt["follow_up_read_tick"] = read_tick
    receipt["follow_up_observed"] = list(obs)

    return EvidenceUpdate(
        evidence_state=overall,
        hypothesis_states=hyp_states,
        weights=new_weights,
        motion_support=dict(previous_update.motion_support),
        response_evidence={
            "follow_up_residuals": residuals,
            "follow_up_z_scores": z_scores,
            "surviving_candidates": surviving,
        },
        residuals=residuals,
        z_scores=z_scores,
        within_candidates=overall == EVIDENCE_STATE_SUPPORTED,
        response_scope_label=(
            CANDIDATE_SUPPORTED_SCOPE_LABEL if overall == EVIDENCE_STATE_SUPPORTED else None
        ),
        out_of_model=out_of_model,
        prior_conflict=False,
        reopen_required=reopen_required,
        unexplained_capacity_preserved=True,
        unexplained_residual=unexplained_res,
        receipt=receipt,
    )


def reopen_candidates(
    observation_history: Sequence[Mapping[str, Any]],
    candidate_families: Sequence[str],
    *,
    window: int = 6,
    size: float = 64.0,
    prior: Optional[Mapping[str, Any]] = None,
    actor_id: Optional[str] = None,
    driver_id: Optional[str] = None,
) -> CandidateProposal:
    """在矛盾或漏真触发重开后，以扩展候选族重建未决候选集（不沿用过期封闭确信）。"""
    proposal = propose(
        observation_history,
        window=window,
        size=size,
        prior=prior,
        candidate_families=candidate_families,
        actor_id=actor_id,
        driver_id=driver_id,
    )
    proposal.evidence_state = EVIDENCE_STATE_UNRESOLVED
    for cname, cand in proposal.candidates.items():
        cand.evidence_state = EVIDENCE_STATE_UNRESOLVED
        proposal.hypothesis_states[cname] = EVIDENCE_STATE_UNRESOLVED
    return proposal


def rollout_candidate(
    candidate: ResponseCandidate,
    initial_state: Mapping[str, Sequence[float]],
    *,
    steps: int = 10,
    size: Optional[float] = None,
) -> List[Tuple[float, float, float]]:
    """冻结候选参数从同一起点外推未来 steps 步 actor 轨迹（供辨识后独立评测）。"""
    if not isinstance(candidate, ResponseCandidate):
        raise TypeError("candidate 必须是 ResponseCandidate 实例")
    n_steps = _require_int("steps", steps, min_value=1)
    fsize = candidate.world_size if size is None else _require_finite_float(
        "size", size, min_exclusive=1.0
    )
    state = {eid: _require_pos("initial_state[%s]" % eid, pos) for eid, pos in initial_state.items()}
    learner = _learner_from_candidate(candidate, state)
    frozen_model = copy.deepcopy(learner.model)
    trajectory: List[Tuple[float, float, float]] = []
    for _ in range(n_steps):
        pred = learner.predict(horizon=1)["predictions"]
        next_actor = _require_pos("rollout.pred", pred[candidate.actor_id]["predicted"])
        next_driver = advance_position(
            state[candidate.driver_id],
            candidate.driver_delta[0],
            candidate.driver_delta[1],
            size=fsize,
        )
        state[candidate.actor_id] = next_actor
        state[candidate.driver_id] = next_driver
        for eid, pos in state.items():
            learner.nodes[eid].pos = pos
        trajectory.append(next_actor)
    assert learner.model == frozen_model
    return trajectory


class CompetingResponseProbe:
    """竞争响应主动辨识状态机封装（串联 propose → choose_probe → update_evidence → reconcile/reopen）。"""

    def __init__(
        self,
        *,
        window: int = 6,
        size: float = 64.0,
        sensor_model: Optional[SensorModel] = None,
        allowed_actions: Optional[Mapping[str, Sequence[float]]] = None,
    ) -> None:
        self.window = _require_int("window", window, min_value=2)
        self.size = _require_finite_float("size", size, min_exclusive=1.0)
        self.sensor_model = sensor_model if sensor_model is not None else SensorModel()
        if not isinstance(self.sensor_model, SensorModel):
            raise TypeError("sensor_model 必须是 SensorModel 实例")
        self.allowed_actions = _require_actions(
            allowed_actions if allowed_actions is not None else DEFAULT_ACTIONS
        )
        self.proposal: Optional[CandidateProposal] = None
        self.last_plan: Optional[ProbePlan] = None
        self.last_update: Optional[EvidenceUpdate] = None
        self.history_log: List[Dict[str, Any]] = []

    @property
    def evidence_state(self) -> Optional[str]:
        if self.last_update is not None:
            return self.last_update.evidence_state
        if self.proposal is not None:
            return self.proposal.evidence_state
        return None

    def observe_passive(
        self,
        observation_history: Sequence[Mapping[str, Any]],
        *,
        prior: Optional[Mapping[str, Any]] = None,
        candidate_families: Optional[Sequence[str]] = None,
    ) -> CandidateProposal:
        self.proposal = propose(
            observation_history,
            window=self.window,
            size=self.size,
            prior=prior,
            candidate_families=candidate_families,
        )
        self.last_plan = None
        self.last_update = None
        self.history_log.append(
            {
                "phase": "propose",
                "evidence_state": self.proposal.evidence_state,
                "weights": dict(self.proposal.weights),
                "motion_support": dict(self.proposal.motion_support),
            }
        )
        return self.proposal

    def select_probe(
        self,
        *,
        read_budget: int = 1,
        action_budget: int = 1,
        state: Optional[Mapping[str, Sequence[float]]] = None,
    ) -> ProbePlan:
        if self.proposal is None:
            raise RuntimeError("请先调用 observe_passive() 生成竞争候选")
        weights = (
            self.last_update.weights
            if self.last_update is not None
            else self.proposal.weights
        )
        plan = choose_probe(
            self.proposal.candidates,
            weights,
            self.allowed_actions,
            read_budget,
            self.sensor_model,
            action_budget=action_budget,
            state=state,
        )
        self.last_plan = plan
        return plan

    def record_measurement(
        self,
        one_observation: Optional[Sequence[float]],
        *,
        action_executed: bool = True,
        read_tick: Optional[int] = None,
    ) -> EvidenceUpdate:
        if self.proposal is None or self.last_plan is None:
            raise RuntimeError("请先调用 observe_passive() 与 select_probe()")
        upd = update_evidence(
            self.last_plan,
            one_observation,
            self.sensor_model,
            weights=self.last_plan.weights_snapshot,
            action_executed=action_executed,
            read_tick=read_tick,
        )
        self.last_update = upd
        for cname, st in upd.hypothesis_states.items():
            if cname in self.proposal.candidates:
                self.proposal.candidates[cname].evidence_state = st
                self.proposal.candidates[cname].hypothesis_weight = upd.weights[cname]
                self.proposal.candidates[cname].response_evidence = dict(upd.response_evidence)
        self.proposal.evidence_state = upd.evidence_state
        self.proposal.hypothesis_states = dict(upd.hypothesis_states)
        self.proposal.weights = dict(upd.weights)
        self.history_log.append(
            {
                "phase": "update_evidence",
                "evidence_state": upd.evidence_state,
                "hypothesis_states": dict(upd.hypothesis_states),
                "weights": dict(upd.weights),
                "receipt": dict(upd.receipt),
            }
        )
        return upd

    def follow_up(
        self,
        next_state: Mapping[str, Sequence[float]],
        next_observation: Sequence[float],
        *,
        read_entity: Optional[str] = None,
        action: str = "passive",
        read_tick: Optional[int] = None,
    ) -> EvidenceUpdate:
        if self.proposal is None or self.last_update is None:
            raise RuntimeError("请先完成一次 record_measurement()")
        upd = reconcile_follow_up(
            self.last_update,
            self.proposal.candidates,
            next_state,
            next_observation,
            self.sensor_model,
            read_entity=read_entity,
            action=action,
            read_tick=read_tick,
        )
        self.last_update = upd
        self.proposal.evidence_state = upd.evidence_state
        self.proposal.hypothesis_states = dict(upd.hypothesis_states)
        self.proposal.weights = dict(upd.weights)
        for cname, st in upd.hypothesis_states.items():
            if cname in self.proposal.candidates:
                self.proposal.candidates[cname].evidence_state = st
                self.proposal.candidates[cname].hypothesis_weight = upd.weights[cname]
        self.history_log.append(
            {
                "phase": "follow_up",
                "evidence_state": upd.evidence_state,
                "hypothesis_states": dict(upd.hypothesis_states),
                "reopen_required": upd.reopen_required,
            }
        )
        return upd

    def reopen(
        self,
        observation_history: Sequence[Mapping[str, Any]],
        candidate_families: Sequence[str],
        *,
        prior: Optional[Mapping[str, Any]] = None,
    ) -> CandidateProposal:
        self.proposal = reopen_candidates(
            observation_history,
            candidate_families,
            window=self.window,
            size=self.size,
            prior=prior,
        )
        self.last_plan = None
        self.last_update = None
        self.history_log.append(
            {
                "phase": "reopen",
                "evidence_state": self.proposal.evidence_state,
                "candidates": list(self.proposal.candidates.keys()),
            }
        )
        return self.proposal


__all__ = [
    "CANDIDATE_SUPPORTED_SCOPE_LABEL",
    "CandidateProposal",
    "CompetingResponseProbe",
    "DEFAULT_ACTIONS",
    "DEFAULT_CANDIDATE_FAMILIES",
    "EVIDENCE_STATES",
    "EVIDENCE_STATE_COMPATIBLE",
    "EVIDENCE_STATE_CONTRADICTED",
    "EVIDENCE_STATE_SUPPORTED",
    "EVIDENCE_STATE_UNRESOLVED",
    "EvidenceUpdate",
    "KNOWN_CANDIDATE_FAMILIES",
    "PARAM_VERSION",
    "ProbePlan",
    "ResponseCandidate",
    "SensorModel",
    "advance_position",
    "choose_probe",
    "forecast",
    "gaussian_posterior",
    "propose",
    "reconcile_follow_up",
    "reopen_candidates",
    "rollout_candidate",
    "update_evidence",
]
