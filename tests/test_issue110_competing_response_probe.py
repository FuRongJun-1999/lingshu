# -*- coding: utf-8 -*-
"""test_issue110_competing_response_probe · Issue #110 四态竞争响应闭环守卫与变异自证
=====================================================================================
覆盖范围（纯标准库 · Py3.12 / Py3.13 双通 · 零外部依赖）：
  1. 观察等价反例（SceneSimulator + WorldLearner）：被动 12 帧轨迹逐位一致，保留
     WorldLearner motion confidence=1.0；干预前两候选均为 observationally_compatible，
     经旋转自适应主动探测动作执行后，真机制升格 response_supported、伪机制精确证伪为
     contradicted，且 10 步冻结外推误差 < 1e-9；
  2. 未探测 / 零预算 / 共线 / 读非响应者 / 未执行动作：保持 unresolved，不误判胜负；
  3. 测量噪声门槛与 Monte Carlo 校准：高噪退回 unresolved（防噪声误杀/误确认），
     低噪精准区分，后验 MAP 准确率贴合解析闭式值；
  4. 漏真反例与重开机制：out-of-model 与先验冲突触发 contradicted + reopen_required，
     delayed_seek 在闭集一拍误配后经 follow-up 精确证伪并可带新候选重开；
  5. 参数有限性与防 bool 门禁：NaN / ±Inf / True / False / 负预算全链路 fail-loud；
  6. `--mutate` 定点变异自证：退化主动判据为被动相容、短路噪声门、绕过非有限/bool 门
     均精确点红。

运行：
  python -X utf8 tests/test_issue110_competing_response_probe.py
  python -X utf8 tests/test_issue110_competing_response_probe.py --mutate
  python -X utf8 -m pytest tests/test_issue110_competing_response_probe.py -q
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lingshu.world.curiosity_explorer import CuriosityExplorer
from lingshu.world.response_probe import (
    CANDIDATE_SUPPORTED_SCOPE_LABEL,
    CompetingResponseProbe,
    DEFAULT_ACTIONS,
    EVIDENCE_STATES,
    EVIDENCE_STATE_COMPATIBLE,
    EVIDENCE_STATE_CONTRADICTED,
    EVIDENCE_STATE_SUPPORTED,
    EVIDENCE_STATE_UNRESOLVED,
    ProbePlan,
    SensorModel,
    advance_position,
    choose_probe,
    forecast,
    gaussian_posterior,
    propose,
    reconcile_follow_up,
    reopen_candidates,
    rollout_candidate,
    update_evidence,
)
from lingshu.world.scene_simulator import SceneEntity, SceneSimulator
from lingshu.world.world_learner import WorldLearner

FRAMES = 12
SIZE = 64


class _DualWorldHarness:
    """在真实 SceneSimulator 上构造 Issue #110 的三种真实生成机制（仅评测侧持有机制标签）。"""

    def __init__(self, mechanism: str, speed: float, axis: str = "x") -> None:
        self.mechanism = mechanism
        self.speed = speed
        self.axis = axis
        self.scene = SceneSimulator(size=SIZE, seed=42)
        pos_a, pos_b = (
            ((2.0, 1.5, 8.0), (10.0, 1.5, 8.0))
            if axis == "x"
            else ((8.0, 1.5, 2.0), (8.0, 1.5, 10.0))
        )
        is_seek = mechanism in ("seek", "delayed_seek")
        self.scene.entities["A"] = SceneEntity(
            id="A",
            category="actor",
            pos=pos_a,
            behavior="seek" if is_seek else "wander",
            speed=speed if is_seek else 0.0,
            goal="B" if is_seek else "",
        )
        self.scene.entities["B"] = SceneEntity(
            id="B",
            category="marker",
            pos=pos_b,
            behavior="wander",
            speed=0.0,
            goal="",
        )
        self.previous_b = pos_b
        self.read_calls = 0

    def observation(self) -> dict:
        return {
            eid: {"category": ent.category, "pos": tuple(ent.pos)}
            for eid, ent in self.scene.entities.items()
        }

    def step(self) -> None:
        latest_b = tuple(self.scene.entities["B"].pos)
        if self.mechanism == "delayed_seek":
            self.scene.entities["B"].pos = self.previous_b
        self.scene.step(n=1)
        if self.mechanism == "delayed_seek":
            self.scene.entities["B"].pos = latest_b
        dx, dz = (self.speed, 0.0) if self.axis == "x" else (0.0, self.speed)
        if self.mechanism == "independent_motion":
            ent_a = self.scene.entities["A"]
            ent_a.pos = advance_position(ent_a.pos, dx, dz, size=SIZE)
        ent_b = self.scene.entities["B"]
        ent_b.pos = advance_position(ent_b.pos, dx, dz, size=SIZE)
        self.previous_b = latest_b

    def intervene(self, action: str) -> None:
        dx, dz = DEFAULT_ACTIONS[action]
        ent_b = self.scene.entities["B"]
        ent_b.pos = advance_position(ent_b.pos, dx, dz, size=SIZE)

    def read(self, entity: str) -> tuple:
        self.read_calls += 1
        return tuple(self.scene.entities[entity].pos)


def _collect(mechanism: str, speed: float, axis: str = "x"):
    har = _DualWorldHarness(mechanism, speed, axis=axis)
    obs = [har.observation()]
    for _ in range(FRAMES - 1):
        har.step()
        obs.append(har.observation())
    return har, obs


def _fit_curiosity_proxy_score(obs, action: str) -> str:
    class _View:
        def __init__(self):
            self.world = type("_W", (), {"size": SIZE})()
            self.entities = {}

        def set_obs(self, frame):
            self.entities = {
                k: type("_E", (), {"category": v["category"], "pos": tuple(v["pos"])})()
                for k, v in frame.items()
            }

    view = _View()
    exp = CuriosityExplorer(size=SIZE, window=6, world=view)
    for frame in obs:
        view.set_obs(frame)
        exp.observe(entities=list(view.entities))
    exp.learn()
    exp.nodes["B"].pos = advance_position(obs[-1]["B"]["pos"], *DEFAULT_ACTIONS[action], size=SIZE)
    scores = {e: exp._info_gain(e) for e in ("A", "B")}
    return max(scores, key=scores.get)


def _assert_raises(exc_types, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_types:
        return
    raise AssertionError("期望抛出 %r，却静默通过" % (exc_types,))


# ---------------------------------------------------------------------------
# 1. 观察等价反例 × 旋转自适应主动辨识 × 四态更新 × 独立未来滚动
# ---------------------------------------------------------------------------

def test_observational_equivalence_and_active_discrimination():
    assert set(EVIDENCE_STATES) == {
        EVIDENCE_STATE_COMPATIBLE,
        EVIDENCE_STATE_SUPPORTED,
        EVIDENCE_STATE_UNRESOLVED,
        EVIDENCE_STATE_CONTRADICTED,
    }
    sensor = SensorModel(sigma=0.0, tolerance=1e-6)
    for axis in ("x", "z"):
        for speed in (0.3, 0.6, 0.9):
            _, obs_seek = _collect("seek", speed, axis)
            _, obs_ind = _collect("independent_motion", speed, axis)
            assert obs_seek == obs_ind

            # 1) 被动阶段：WorldLearner 原样拟合出同模型（confidence=1.0），两候选皆仅相容
            prop = propose(obs_seek, window=6, size=SIZE)
            assert prop.passive_models_identical is True
            assert prop.evidence_state == EVIDENCE_STATE_COMPATIBLE
            assert prop.world_learner_model["relations"][0]["confidence"] == 1.0
            assert prop.motion_support["seek"] == 1.0
            assert prop.motion_support["independent_motion"] == 1.0
            assert prop.weights == {"seek": 0.5, "independent_motion": 0.5}
            assert all(c.evidence_state == EVIDENCE_STATE_COMPATIBLE for c in prop.candidates.values())
            assert all(c.response_evidence is None for c in prop.candidates.values())

            # 2) 选择主动探针：随轴旋转自适应横移（x 向运动选 z 向，z 向运动选 x 向），非写死模板
            plan = choose_probe(prop.candidates, prop.weights, DEFAULT_ACTIONS, 1, sensor)
            expected_lateral = ("z_plus", "z_minus") if axis == "x" else ("x_plus", "x_minus")
            assert plan.budget_sufficient is True and plan.informative is True
            assert plan.action in expected_lateral
            assert plan.read_entity == "A"
            assert math.isclose(plan.eig_bits, 1.0, abs_tol=1e-12)
            assert plan.separation > 0.0
            assert _fit_curiosity_proxy_score(obs_seek, plan.action) == "B"

            # 3) 在两座真实 SceneSimulator 上分别执行探测与四态更新
            for truth in ("seek", "independent_motion"):
                har, training = _collect(truth, speed, axis)
                assert training == obs_seek
                har.intervene(plan.action)
                har.step()
                obs_one = har.read(plan.read_entity)
                assert har.read_calls == 1

                upd = update_evidence(plan, obs_one, sensor, read_tick=FRAMES + 1)
                rival = "independent_motion" if truth == "seek" else "seek"
                assert upd.evidence_state == EVIDENCE_STATE_SUPPORTED
                assert upd.hypothesis_states[truth] == EVIDENCE_STATE_SUPPORTED
                assert upd.hypothesis_states[rival] == EVIDENCE_STATE_CONTRADICTED
                assert upd.weights[truth] == 1.0 and upd.weights[rival] == 0.0
                assert upd.within_candidates is True
                assert upd.response_scope_label == CANDIDATE_SUPPORTED_SCOPE_LABEL
                assert upd.out_of_model is False and upd.reopen_required is False
                # motion_support 保留被动相容度，不被回填覆盖
                assert upd.motion_support[rival] == 1.0
                assert upd.receipt["read_entity"] == "A"
                assert upd.receipt["action"] == plan.action
                json.dumps(upd.receipt, allow_nan=False)

                # 4) 辨识后同起点冻结外推 10 步，选中候选零误差
                shared_start = {
                    "A": obs_one,
                    "B": plan.predictions[truth]["B"],
                }
                fc_true = rollout_candidate(prop.candidates[truth], shared_start, steps=10, size=SIZE)
                fc_rival = rollout_candidate(prop.candidates[rival], shared_start, steps=10, size=SIZE)
                actual = []
                for _ in range(10):
                    har.step()
                    actual.append(har.observation()["A"]["pos"])
                assert har.read_calls == 1
                err_true = sum(math.dist(a, b) for a, b in zip(fc_true, actual)) / 10.0
                err_rival = sum(math.dist(a, b) for a, b in zip(fc_rival, actual)) / 10.0
                assert err_true < 1e-9
                assert err_rival > 0.1


# ---------------------------------------------------------------------------
# 2. 预算不足 / 未探测 / 共线对照 / 读非响应者 -> unresolved
# ---------------------------------------------------------------------------

def test_unprobed_budget_and_uninformative_controls_stay_unresolved():
    _, obs = _collect("seek", 0.6, "x")
    prop = propose(obs, window=6, size=SIZE)
    sensor = SensorModel(sigma=0.0, tolerance=1e-6)

    # read_budget == 0 -> budget_sufficient=False -> unresolved
    plan_zero_read = choose_probe(prop.candidates, prop.weights, DEFAULT_ACTIONS, 0, sensor)
    assert plan_zero_read.budget_sufficient is False
    upd_zero = update_evidence(plan_zero_read, None, sensor)
    assert upd_zero.evidence_state == EVIDENCE_STATE_UNRESOLVED
    assert upd_zero.weights == {"seek": 0.5, "independent_motion": 0.5}
    assert all(st == EVIDENCE_STATE_UNRESOLVED for st in upd_zero.hypothesis_states.values())

    # action_budget == 0 (仅允许 passive) -> 无区分力 -> unresolved
    plan_zero_act = choose_probe(
        prop.candidates, prop.weights, DEFAULT_ACTIONS, 1, sensor, action_budget=0
    )
    assert plan_zero_act.action == "passive" and plan_zero_act.informative is False
    har, _ = _collect("seek", 0.6, "x")
    har.intervene("passive")
    har.step()
    upd_passive = update_evidence(plan_zero_act, har.read("A"), sensor)
    assert upd_passive.evidence_state == EVIDENCE_STATE_UNRESOLVED
    assert upd_passive.weights == {"seek": 0.5, "independent_motion": 0.5}

    # 未执行动作或缺失观测 -> unresolved
    active_plan = choose_probe(prop.candidates, prop.weights, DEFAULT_ACTIONS, 1, sensor)
    upd_no_obs = update_evidence(active_plan, None, sensor)
    assert upd_no_obs.evidence_state == EVIDENCE_STATE_UNRESOLVED
    upd_not_exec = update_evidence(active_plan, (2.6, 1.5, 8.0), sensor, action_executed=False)
    assert upd_not_exec.evidence_state == EVIDENCE_STATE_UNRESOLVED

    # 共线移动（x_plus）读 A、或横移（z_plus）读 B -> 无区分力 -> unresolved
    for act, ent in (("x_plus", "A"), ("z_plus", "B")):
        narrowed = choose_probe(
            prop.candidates,
            prop.weights,
            {act: DEFAULT_ACTIONS[act]},
            1,
            sensor,
        )
        ctrl_plan = ProbePlan(
            action=act,
            action_delta=DEFAULT_ACTIONS[act],
            action_cost=math.hypot(*DEFAULT_ACTIONS[act]),
            read_entity=ent,
            eig_bits=0.0,
            separation=0.0,
            informative=False,
            budget_sufficient=True,
            read_budget=1,
            action_budget=1,
            predictions=narrowed.predictions,
            candidates_table=narrowed.candidates_table,
            weights_snapshot=narrowed.weights_snapshot,
            execution_conditions=narrowed.execution_conditions,
            state_snapshot=narrowed.state_snapshot,
            candidate_objects=narrowed.candidate_objects,
        )
        har_c, _ = _collect("seek", 0.6, "x")
        har_c.intervene(act)
        har_c.step()
        upd_c = update_evidence(ctrl_plan, har_c.read(ent), sensor)
        assert upd_c.evidence_state == EVIDENCE_STATE_UNRESOLVED
        assert upd_c.weights == {"seek": 0.5, "independent_motion": 0.5}


# ---------------------------------------------------------------------------
# 3. 高噪退回 unresolved + 低噪分辨 + Monte Carlo 解析校准
# ---------------------------------------------------------------------------

def test_noise_tolerance_gates_and_gaussian_calibration():
    _, obs = _collect("seek", 0.6, "x")
    prop = propose(obs, window=6, size=SIZE)

    means = {}
    for truth in ("seek", "independent_motion"):
        har, _ = _collect(truth, 0.6, "x")
        har.intervene("z_plus")
        har.step()
        means[truth] = har.read("A")
    delta = math.dist(means["seek"], means["independent_motion"])

    # 低噪 (sigma=0.05, SNR ~ 7.59 >= 2.0): 在真均值附近观测可升格 response_supported
    low_noise = SensorModel(sigma=0.05)
    plan_low = choose_probe(
        prop.candidates, prop.weights, {"z_plus": DEFAULT_ACTIONS["z_plus"]}, 1, low_noise
    )
    upd_low = update_evidence(plan_low, means["seek"], low_noise)
    assert upd_low.evidence_state == EVIDENCE_STATE_SUPPORTED
    assert upd_low.hypothesis_states["seek"] == EVIDENCE_STATE_SUPPORTED
    assert upd_low.hypothesis_states["independent_motion"] == EVIDENCE_STATE_CONTRADICTED
    assert upd_low.weights["seek"] > 0.999

    # 高噪 (sigma=0.5 与 sigma=5.0, SNR < 2.0): 保持 unresolved，不误杀真机制
    for high_sigma in (0.5, 5.0):
        high_noise = SensorModel(sigma=high_sigma)
        plan_high = choose_probe(
            prop.candidates, prop.weights, {"z_plus": DEFAULT_ACTIONS["z_plus"]}, 1, high_noise
        )
        assert plan_high.eig_bits < plan_low.eig_bits
        upd_high = update_evidence(plan_high, means["seek"], high_noise)
        assert upd_high.evidence_state == EVIDENCE_STATE_UNRESOLVED
        assert upd_high.hypothesis_states["seek"] == EVIDENCE_STATE_UNRESOLVED
        assert upd_high.hypothesis_states["independent_motion"] == EVIDENCE_STATE_UNRESOLVED

    # Monte Carlo 校准对比解析闭式 Φ(δ / (2σ))
    preds = plan_low.predictions
    for sigma in (0.05, 0.2, 0.5):
        rng = random.Random(1103)
        correct = 0
        total = 2000
        for truth in ("seek", "independent_motion"):
            for _ in range(total // 2):
                noisy = tuple(v + rng.gauss(0.0, sigma) for v in means[truth])
                post = gaussian_posterior(preds, "A", noisy, prop.weights, sigma)
                if max(post, key=post.get) == truth:
                    correct += 1
        emp = correct / total
        analytic = 0.5 * (1.0 + math.erf(delta / (2.0 * sigma * math.sqrt(2.0))))
        assert abs(emp - analytic) < 0.03


# ---------------------------------------------------------------------------
# 4. 漏真反例（out_of_model / prior_conflict / delayed_seek）与重开闭环
# ---------------------------------------------------------------------------

def test_out_of_model_prior_conflict_and_delayed_seek_reopen():
    _, obs = _collect("seek", 0.6, "x")
    probe = CompetingResponseProbe(size=SIZE, sensor_model=SensorModel(sigma=0.0, tolerance=1e-6))
    probe.observe_passive(obs)
    plan = probe.select_probe(read_budget=1, action_budget=1)

    # A. 模型外异常观测 -> 全候选 contradicted + reopen_required + 未解释残差
    expected_seek = plan.predictions["seek"]["A"]
    unexpected = (expected_seek[0] + 3.0, expected_seek[1], expected_seek[2] + 3.0)
    upd_oom = update_evidence(plan, unexpected, probe.sensor_model)
    assert upd_oom.evidence_state == EVIDENCE_STATE_CONTRADICTED
    assert upd_oom.out_of_model is True and upd_oom.reopen_required is True
    assert upd_oom.unexplained_capacity_preserved is True
    assert upd_oom.unexplained_residual > 2.0
    assert all(st == EVIDENCE_STATE_CONTRADICTED for st in upd_oom.hypothesis_states.values())

    # A2. 带噪传感器下的模型外异常观测 -> 同触发 contradicted + out_of_model
    noisy_sensor = SensorModel(sigma=0.05)
    noisy_plan = choose_probe(
        probe.proposal.candidates, probe.proposal.weights, DEFAULT_ACTIONS, 1, noisy_sensor
    )
    upd_noisy_oom = update_evidence(noisy_plan, unexpected, noisy_sensor)
    assert upd_noisy_oom.evidence_state == EVIDENCE_STATE_CONTRADICTED
    assert upd_noisy_oom.out_of_model is True and upd_noisy_oom.reopen_required is True

    # B. 零先验冲突 -> prior_conflict + contradicted + reopen_required（不得把 0 先验硬改 1）
    expected_ind = plan.predictions["independent_motion"]["A"]
    upd_conflict = update_evidence(
        plan,
        expected_ind,
        probe.sensor_model,
        weights={"seek": 1.0, "independent_motion": 0.0},
    )
    assert upd_conflict.evidence_state == EVIDENCE_STATE_CONTRADICTED
    assert upd_conflict.prior_conflict is True and upd_conflict.reopen_required is True
    assert upd_conflict.weights == {"seek": 1.0, "independent_motion": 0.0}

    # C. 漏真延迟响应反例 (delayed_seek)：
    #    第 1 拍闭集暂选 independent_motion (within_candidates)，第 2 拍实测暴露延迟转向 ->
    #    independent_motion 被精确推翻为 contradicted，触发重开并支持 delayed_seek！
    har_d, obs_d = _collect("delayed_seek", 0.6, "x")
    assert obs_d == obs
    probe_d = CompetingResponseProbe(size=SIZE, sensor_model=SensorModel(sigma=0.0, tolerance=1e-6))
    probe_d.observe_passive(obs_d)
    plan_d = probe_d.select_probe(read_budget=1, action_budget=1)

    har_d.intervene(plan_d.action)
    har_d.step()
    obs_step1 = har_d.read("A")
    upd1 = probe_d.record_measurement(obs_step1, read_tick=FRAMES + 1)
    assert upd1.evidence_state == EVIDENCE_STATE_SUPPORTED
    assert upd1.within_candidates is True
    assert upd1.response_scope_label == CANDIDATE_SUPPORTED_SCOPE_LABEL
    assert upd1.hypothesis_states["independent_motion"] == EVIDENCE_STATE_SUPPORTED

    # 第 2 拍到来：A 开始转向被干预后的 B
    state_before_step2 = {"A": obs_step1, "B": plan_d.predictions["independent_motion"]["B"]}
    har_d.step()
    obs_step2 = har_d.observation()["A"]["pos"]
    upd2 = probe_d.follow_up(state_before_step2, obs_step2, read_tick=FRAMES + 2)
    assert upd2.evidence_state == EVIDENCE_STATE_CONTRADICTED
    assert upd2.hypothesis_states["independent_motion"] == EVIDENCE_STATE_CONTRADICTED
    assert upd2.out_of_model is True and upd2.reopen_required is True
    assert upd2.unexplained_capacity_preserved is True

    # 重开候选集引入 delayed_seek，再经主动探测确认 delayed_seek
    reopened = probe_d.reopen(obs_d, ("seek", "independent_motion", "delayed_seek"))
    assert reopened.evidence_state == EVIDENCE_STATE_UNRESOLVED
    plan_re = probe_d.select_probe(read_budget=1, action_budget=1)
    har_d2, _ = _collect("delayed_seek", 0.6, "x")
    har_d2.intervene(plan_re.action)
    har_d2.step()
    # 2-候选时第 1 拍区分为 seek vs {independent_motion, delayed_seek} -> unresolved（保留两幸存候选）
    upd_re1 = probe_d.record_measurement(har_d2.read("A"), read_tick=FRAMES + 1)
    assert upd_re1.evidence_state == EVIDENCE_STATE_UNRESOLVED
    assert upd_re1.hypothesis_states["seek"] == EVIDENCE_STATE_CONTRADICTED
    assert upd_re1.hypothesis_states["delayed_seek"] == EVIDENCE_STATE_UNRESOLVED
    assert upd_re1.hypothesis_states["independent_motion"] == EVIDENCE_STATE_UNRESOLVED


# ---------------------------------------------------------------------------
# 5. 非有限 / bool 门禁（fail-closed）
# ---------------------------------------------------------------------------

def test_nonfinite_and_bool_guards():
    for bad in (float("nan"), float("inf"), float("-inf"), True, False):
        _assert_raises((TypeError, ValueError), SensorModel, sigma=bad)
        _assert_raises((TypeError, ValueError), SensorModel, tolerance=bad)
        _assert_raises((TypeError, ValueError), SensorModel, min_snr=bad)
        _assert_raises((TypeError, ValueError), SensorModel, contradiction_z=bad)
        _assert_raises((TypeError, ValueError), SensorModel, support_z_max=bad)
        _assert_raises((TypeError, ValueError), SensorModel, support_posterior_threshold=bad)

    _, obs = _collect("seek", 0.6, "x")
    for bad_int in (True, False, 1.5, -1, 1):
        if bad_int == 1:
            continue
        _assert_raises((TypeError, ValueError), propose, obs, window=bad_int)
    for bad_size in (float("nan"), float("inf"), True, False, 0.5, -10.0):
        _assert_raises((TypeError, ValueError), propose, obs, size=bad_size)
    for bad_w in (float("nan"), float("inf"), True, -0.1):
        _assert_raises(
            (TypeError, ValueError),
            propose,
            obs,
            prior={"seek": bad_w, "independent_motion": 0.5},
        )

    prop = propose(obs, window=6, size=SIZE)
    sensor = SensorModel()
    for bad_budget in (True, False, 1.0, -1):
        _assert_raises(
            (TypeError, ValueError),
            choose_probe,
            prop.candidates,
            prop.weights,
            DEFAULT_ACTIONS,
            bad_budget,
            sensor,
        )
        _assert_raises(
            (TypeError, ValueError),
            choose_probe,
            prop.candidates,
            prop.weights,
            DEFAULT_ACTIONS,
            1,
            sensor,
            action_budget=bad_budget,
        )
    for bad_act in ((float("nan"), 0.0), (0.0, float("inf")), (True, 0.0)):
        _assert_raises(
            (TypeError, ValueError),
            choose_probe,
            prop.candidates,
            prop.weights,
            {"bad": bad_act},
            1,
            sensor,
        )

    plan = choose_probe(prop.candidates, prop.weights, DEFAULT_ACTIONS, 1, sensor)
    for bad_obs in (
        (float("nan"), 1.5, 8.0),
        (2.0, float("inf"), 8.0),
        (True, 1.5, 8.0),
        (2.0, 1.5),
    ):
        _assert_raises((TypeError, ValueError), update_evidence, plan, bad_obs, sensor)
    _assert_raises((TypeError, ValueError), update_evidence, plan, (2.6, 1.5, 8.0), sensor, action_executed=1)
    _assert_raises((TypeError, ValueError), update_evidence, plan, (2.6, 1.5, 8.0), sensor, read_tick=True)
    _assert_raises(
        (TypeError, ValueError),
        gaussian_posterior,
        plan.predictions,
        "A",
        (2.6, 1.5, 8.0),
        prop.weights,
        0.0,
    )
    _assert_raises(
        (TypeError, ValueError),
        gaussian_posterior,
        plan.predictions,
        "A",
        (2.6, 1.5, 8.0),
        prop.weights,
        float("nan"),
    )


# ---------------------------------------------------------------------------
# 6. 定点变异自证（--mutate）：退化干预判据 / 短路噪声门 / 绕过 bool-NaN 门必红
# ---------------------------------------------------------------------------

def _check_mutation_degrade_to_passive():
    """变异 M1：把主动动作退化回被动相容（只允许 passive）⇒ 必无法区分 seek 与 independent_motion。"""
    _, obs = _collect("seek", 0.6, "x")
    prop = propose(obs, window=6, size=SIZE)
    sensor = SensorModel(sigma=0.0, tolerance=1e-6)
    mutated_plan = choose_probe(
        prop.candidates, prop.weights, {"passive": DEFAULT_ACTIONS["passive"]}, 1, sensor
    )
    har, _ = _collect("seek", 0.6, "x")
    har.intervene(mutated_plan.action)
    har.step()
    upd = update_evidence(mutated_plan, har.read(mutated_plan.read_entity), sensor)
    assert upd.evidence_state == EVIDENCE_STATE_SUPPORTED, (
        "M1 应当在退化为被动探测时点红：实际状态=%s" % upd.evidence_state
    )


def _check_mutation_short_circuit_noise_gate():
    """变异 M2：短路最小信噪比与后验支持门槛 ⇒ 高噪 σ=5.0 会误将猜测升为 response_supported。"""
    _, obs = _collect("seek", 0.6, "x")
    prop = propose(obs, window=6, size=SIZE)
    # 故意把门禁拉垮到 min_snr=1e-6, support_posterior_threshold=0.500001
    broken_sensor = SensorModel(sigma=5.0, support_posterior_threshold=0.500001, min_snr=1e-6)
    plan = choose_probe(
        prop.candidates, prop.weights, {"z_plus": DEFAULT_ACTIONS["z_plus"]}, 1, broken_sensor
    )
    har, _ = _collect("seek", 0.6, "x")
    har.intervene("z_plus")
    har.step()
    upd = update_evidence(plan, har.read("A"), broken_sensor)
    assert upd.evidence_state == EVIDENCE_STATE_UNRESOLVED, (
        "M2 应当在噪声门被短路时点红：实际状态=%s, weights=%r"
        % (upd.evidence_state, upd.weights)
    )


def _check_mutation_bypass_finite_bool_gate():
    """变异 M3：绕过 SensorModel/update_evidence 对 NaN 与 bool 的校验 ⇒ 静默通过而非抛错。"""
    # 模拟未加门禁的朴素比对：dist <= tol 对 NaN 静默返回 False，而非抛 ValueError
    naive_residual = math.dist((2.6, 1.5, 8.0), (float("nan"), 1.5, 8.0))
    if math.isfinite(naive_residual):
        raise AssertionError("预期 NaN 残差非有限")
    # 守卫要求对 NaN 观测必须 fail-loud 抛 ValueError；若退化成朴素 dist <= tol，不会抛错 ⇒ 必抓红
    _, obs = _collect("seek", 0.6, "x")
    prop = propose(obs, window=6, size=SIZE)
    sensor = SensorModel()
    plan = choose_probe(prop.candidates, prop.weights, DEFAULT_ACTIONS, 1, sensor)
    update_evidence(plan, (float("nan"), 1.5, 8.0), sensor)


def test_mutation_self_proof_guards_have_teeth():
    for label, fn in (
        ("M1_degrade_to_passive", _check_mutation_degrade_to_passive),
        ("M2_short_circuit_noise_gate", _check_mutation_short_circuit_noise_gate),
    ):
        try:
            fn()
        except AssertionError:
            pass
        else:
            raise AssertionError("变异 %s 未被守卫捕获！" % label)

    # M3：真接口面对 NaN 必须抛 ValueError（不得静默返回）
    _assert_raises(ValueError, _check_mutation_bypass_finite_bool_gate)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Issue #110 竞争响应四态闭环测试与变异自证")
    parser.add_argument("--mutate", action="store_true", help="运行定点变异自证")
    args = parser.parse_args(argv)

    tests = [
        test_observational_equivalence_and_active_discrimination,
        test_unprobed_budget_and_uninformative_controls_stay_unresolved,
        test_noise_tolerance_gates_and_gaussian_calibration,
        test_out_of_model_prior_conflict_and_delayed_seek_reopening
        if "test_out_of_model_prior_conflict_and_delayed_seek_reopening" in globals()
        else test_out_of_model_prior_conflict_and_delayed_seek_reopen,
        test_nonfinite_and_bool_guards,
        test_mutation_self_proof_guards_have_teeth,
    ]
    for fn in tests:
        fn()
        print("PASS %s" % fn.__name__)

    if args.mutate:
        mutations = (
            ("M1_degrade_to_passive", _check_mutation_degrade_to_passive, AssertionError),
            ("M2_short_circuit_noise_gate", _check_mutation_short_circuit_noise_gate, AssertionError),
            ("M3_nan_observation_fail_loud", _check_mutation_bypass_finite_bool_gate, ValueError),
        )
        for name, fn, expected_exc in mutations:
            try:
                fn()
            except expected_exc as exc:
                print("MUTATE-CAUGHT %s: %s" % (name, exc))
            else:
                print("MUTATE-MISSED %s" % name)
                return 1
        print("MUTATION_SELF_PROOF=PASS (3/3 caught)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
