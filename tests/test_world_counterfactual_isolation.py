"""W3：真实预测链上的关系条件试演；候选不提交为观测或缓存。"""
from copy import deepcopy
import pickle

import pytest

from lingshu.world.world_model import UnifiedWorldModel


def observed_model():
    model = UnifiedWorldModel(size=80, seed=7)
    for x in range(6, 12):
        model.perceive([{"eid": "actor", "category": "person", "pos": [x, 1.5, 2]},
                        {"eid": "goal", "category": "marker", "pos": [30, 1.5, 2]}])
    return model


def trial(model, **kwargs):
    method = getattr(model, "counterfactual", None)
    assert callable(method), "W3_API_MISSING: 缺少隔离反事实入口"
    return method(**kwargs)


def flee():
    return [{"source": "actor", "relation": "flee", "target": "goal"}]


def frozen(model):
    return pickle.dumps(model.__dict__, protocol=4)


def test_legacy_generate_control():
    model = observed_model()
    out = model.generate()
    assert out["predictions"]["actor"]["predicted"] == [12, 1.5, 2]
    assert model._last_prediction is out["predictions"]


def test_relation_changes_prediction_without_committing_any_model_state():
    model = observed_model()
    before = frozen(model)
    cache = model._last_prediction
    nodes = model.nodes
    out = trial(model, relations=flee())
    assert frozen(model) == before, "W3_LIVE_STATE_CHANGED: 试演污染真实状态"
    assert model._last_prediction is cache and model.nodes is nodes
    assert out["baseline"]["predictions"]["actor"]["predicted"] == [12, 1.5, 2]
    assert out["alternative"]["predictions"]["actor"]["predicted"] == [10, 1.5, 2]
    assert out["deltas"]["actor"] == {"position": [-2.0, 0.0, 0.0], "distance": 2.0}
    assert out["provenance"] == "counterfactual" and out["verification"] == "unverified"
    assert out["committed"] is False, "W3_CANDIDATE_MARKED_COMMITTED"
    assert out["source_tick"] == 6 and out["target_tick"] == 7


def test_no_conditions_matches_normal_generate_on_an_independent_model():
    model = observed_model()
    expected = deepcopy(model).generate(horizon=3)
    before = frozen(model)
    out = trial(model, horizon=3)
    assert out["baseline"] == out["alternative"] == expected
    assert all(d["distance"] == 0 for d in out["deltas"].values())
    assert frozen(model) == before


def test_existing_live_forecast_still_verifies_at_its_original_target_tick():
    model = observed_model()
    forecast = model.generate(horizon=2)
    before = frozen(model)
    out = trial(model, horizon=1, relations=flee())
    assert model._last_prediction is forecast["predictions"], "W3_CACHE_REPLACED: 候选替换真实预测"
    assert frozen(model) == before and out["target_tick"] == 7
    actual = [{"eid": eid, "category": model.nodes[eid].category, "pos": p["predicted"]}
              for eid, p in forecast["predictions"].items()]
    model.perceive(actual, tick=8)
    checked = model.verify()
    assert checked["total"] == 2 and checked["hits"] == 2


def test_outputs_inputs_and_repeated_trials_do_not_alias_the_source():
    model = observed_model()
    relation = flee()
    before = frozen(model)
    first = trial(model, relations=relation)
    assert first == trial(model, relations=relation)
    first["baseline"]["predictions"]["actor"]["predicted"][0] = -999
    first["alternative"]["predictions"]["goal"]["confidence"] = -1
    first["assumptions"][0]["relation"] = "seek"
    assert relation == flee() and frozen(model) == before
    assert trial(model, relations=relation)["assumptions"] == flee()


def test_failed_prediction_only_mutates_discarded_branch(monkeypatch):
    model = observed_model()
    before = frozen(model)

    def fail(branch, **kwargs):
        branch.nodes["actor"].attrs["trial"] = {"nested": [1, 2]}
        branch.history[-1]["entities"]["actor"][0] = -999
        branch._last_prediction["trial"] = {"predicted": [0, 0, 0]}
        branch._rng.random()
        raise RuntimeError("intentional candidate failure")

    monkeypatch.setattr(UnifiedWorldModel, "generate", fail)
    with pytest.raises(RuntimeError, match="intentional candidate failure"):
        trial(model, relations=flee())
    assert frozen(model) == before, "W3_EXCEPTION_LEAK: 失败候选污染真实状态"


def test_simulator_and_its_alias_are_neither_copied_nor_read(monkeypatch):
    model = observed_model()

    class HiddenWorld:
        def __deepcopy__(self, memo):
            raise AssertionError("W3_HIDDEN_WORLD_COPIED: 读取了模拟器隐藏真相")

    hidden = HiddenWorld()
    model.world = hidden
    model._world_alias = hidden
    generate = UnifiedWorldModel.generate

    def observed_only(branch, **kwargs):
        assert branch.world is None and branch._world_alias is None, "W3_WORLD_ALIAS_LEAK"
        return generate(branch, **kwargs)

    monkeypatch.setattr(UnifiedWorldModel, "generate", observed_only)
    out = trial(model, relations=flee())
    assert model.world is hidden and model._world_alias is hidden
    assert out["alternative"]["predictions"]["actor"]["predicted"][0] == 10


def test_core_uses_same_model_without_database_writes():
    from lingshu.core.core import SpacetimeMemoryEngine
    engine = SpacetimeMemoryEngine(":memory:")
    try:
        engine.world_model("init", {"size": 80, "seed": 7})
        model = engine._wmodel = observed_model()
        before = frozen(model)
        writes = engine.store.conn.total_changes
        result = engine.world_model("counterfactual", {"horizon": 2, "relations": flee()})
        assert result["status"] == "ok" and "counterfactual" in result, "W3_CORE_ROUTE_MISSING"
        out = result["counterfactual"]
        assert out["horizon"] == 2 and out["assumptions"] == flee(), "W3_CORE_CONDITIONS_DROPPED"
        assert out["alternative"]["predictions"]["actor"]["predicted"][0] == 9
        assert engine._wmodel is model and frozen(model) == before
        assert engine.store.conn.total_changes == writes
    finally:
        engine.store.close()


@pytest.mark.parametrize("horizon", [True, 0, -1, 1.5, "2", 65],
                         ids=["bool", "zero", "negative", "float", "string", "budget"])
def test_invalid_horizon_rejected_before_generation(horizon, monkeypatch):
    model = observed_model()
    before = frozen(model)

    def forbidden(*args, **kwargs):
        pytest.fail("无效请求不应开始生成")

    monkeypatch.setattr(UnifiedWorldModel, "generate", forbidden)
    with pytest.raises(ValueError, match="horizon"):
        trial(model, horizon=horizon)
    assert frozen(model) == before


@pytest.mark.parametrize("relations", [
    "flee", ["flee"], [{"source": "actor", "relation": "flee"}],
    [{"source": "actor", "relation": "flee", "target": "missing"}],
    [{"source": "actor", "relation": "flee", "target": "actor"}],
    [{"source": "actor", "relation": "teleport", "target": "goal"}],
    [{"source": "actor", "relation": "flee", "target": "goal", "confidence": 1}],
    flee() + flee(),
], ids=["not-list", "not-object", "missing-field", "unknown-id", "self-edge",
        "unsupported-relation", "extra-field", "duplicate-source"])
def test_invalid_relations_are_atomic(relations):
    model = observed_model()
    before = frozen(model)
    with pytest.raises(ValueError, match="relation"):
        trial(model, relations=relations)
    assert frozen(model) == before


def test_explicit_physics_configuration_uses_same_prediction_wrapper(monkeypatch):
    from lingshu.world.physics_predict import wrap_generate
    from lingshu.world.physics import make_environment
    model = observed_model()
    model.physics_env = make_environment()
    monkeypatch.setattr(UnifiedWorldModel, "generate", wrap_generate(UnifiedWorldModel.generate))
    before = frozen(model)
    out = trial(model, relations=flee())
    assert out["baseline"]["physics"]["env_aware"] is True
    assert out["alternative"]["physics"]["env_source"] == ["physics_env"]
    assert frozen(model) == before
