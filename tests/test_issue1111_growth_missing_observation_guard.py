# -*- coding: utf-8 -*-
"""#1111：默认 `legacy` 策略下主体缺观测时仍推进假设验证/加置信的收口守卫。

契约（`_growth_tick` 缺观测门槛，#1111 起两策略通用）：主体被遮挡、真实观测
快照没有它时，验证窗口与假设置信均**等待新证据**——此时 `verify()` 正确返回
`total=0`，没有外部验证样本：candidate 不假 revert、confirmed 不凭空加置信；
观测恢复后窗口照常推进（门槛是「等待」不是「冻结」）。
"""
import pytest

from lingshu.world.world_model import UnifiedWorldModel
from lingshu.world.wm_simloop import SimulationLoop


def _make(phase):
    model = UnifiedWorldModel(size=24)
    loop = SimulationLoop(world_model=model, verify_window=2)
    loop.scene.create_scene(trees=0, water=False)
    subject = loop.scene.add_entity("actor", pos=(2, 1.5, 2), speed=0.0)
    model.perceive(loop._observe())
    loop._grow(subject, (10, 1.5, 2), 0.9, trigger="guard_fixture")
    h = loop._hypotheses[subject]
    if phase == "confirmed":
        h["phase"] = phase
        model.nodes[h["node"]].confidence = 0.5
    loop._mask.add(subject)
    return loop, model, subject, h


def test_candidate_without_observation_not_reverted():
    loop, model, subject, h = _make("candidate")
    loop.step(n=2)
    assert subject in loop._hypotheses, "缺观测时 candidate 被假 revert（#1111）"
    assert model.nodes[h["node"]].confidence == pytest.approx(0.3)
    assert loop.wal[-1]["payload"]["total"] == 0


def test_confirmed_without_observation_no_confidence_inflation():
    loop, model, subject, h = _make("confirmed")
    loop.step(n=2)
    assert subject in loop._hypotheses
    assert model.nodes[h["node"]].confidence == pytest.approx(0.5), (
        "缺观测时 confirmed 凭空加置信（#1111：0.5→0.6 形态）")


def test_window_resumes_when_observation_returns():
    # 门槛是「等待」不是「冻结」：观测恢复后窗口照常推进（本场景 subject 静止、
    # 从未趋向假设目标 ⇒ 窗口走满后按证据 revert 是正当收敛，不断言节点存活）。
    loop, model, subject, h = _make("candidate")
    loop.step(n=1)                       # 缺观测：窗口暂停
    loop._mask.discard(subject)          # 观测恢复
    loop.step(n=3)
    assert h["ticks"] >= 1, "观测恢复后验证窗口应继续推进"


def test_conservative_policy_unchanged_regression():
    # 回归锚：conservative 的原门槛行为保持（修法是把门槛推广，不是新建）。
    model = UnifiedWorldModel(size=24)
    model.set_identity_policy("conservative")
    loop = SimulationLoop(world_model=model, verify_window=2)
    loop.scene.create_scene(trees=0, water=False)
    subject = loop.scene.add_entity("actor", pos=(2, 1.5, 2), speed=0.0)
    model.perceive(loop._observe())
    loop._grow(subject, (10, 1.5, 2), 0.9, trigger="guard_conservative")
    loop._mask.add(subject)
    loop.step(n=2)
    assert subject in loop._hypotheses


def test_with_observation_still_progresses_regression():
    # 回归锚：有观测时验证窗口照常推进（修法不得把正常路径改停）。
    loop, model, subject, h = _make("candidate")
    loop._mask.discard(subject)          # 从一开始就可见
    loop.step(n=3)
    assert h["ticks"] >= 1, "有观测时窗口应推进"
