"""Pursuit centers use observed direction without hiding target uncertainty."""
import math

import pytest

from lingshu.world.curiosity_explorer import CuriosityExplorer
from lingshu.world.scene_simulator import SceneSimulator
from lingshu.world.world_learner import WorldLearner
from lingshu.world.world_model import UnifiedWorldModel


def pursuit_scene(seed):
    scene = SceneSimulator(size=80, seed=seed)
    player = scene.add_entity("player", pos=(30, 1.5, 30), speed=0.6)
    wolf = scene.add_entity("wolf", behavior="seek", pos=(10, 1.5, 10), speed=0.5, goal=player)
    scene.add_entity("rabbit", behavior="flee", pos=(45, 1.5, 45), speed=0.4, goal=wolf)
    scene.add_entity("stationary", pos=(60, 1.5, 15), speed=0)
    return scene


def observe(model):
    return model.perceive() if isinstance(model, UnifiedWorldModel) else model.observe()


def predict(model):
    return model.generate()["predictions"] if isinstance(model, UnifiedWorldModel) else model.predict()["predictions"]


def warmup(model, ticks):
    observe(model)
    for _ in range(ticks):
        model.world.step()
        observe(model)
    if isinstance(model, UnifiedWorldModel):
        model.infer_patterns()
    else:
        model.learn()


@pytest.mark.parametrize("model_class", [WorldLearner, UnifiedWorldModel])
def test_chaser_of_random_target_moves_center_with_wide_bound(model_class):
    scene = pursuit_scene(3)
    model = model_class(world=scene)
    warmup(model, 12)
    actor = next(eid for eid, e in scene.entities.items() if e.category == "wolf")
    target = next(eid for eid, e in scene.entities.items() if e.category == "player")
    old_position = model.nodes[actor].pos
    old_target_position = model.nodes[target].pos
    # Prediction must not consult the simulator's behavior, goal, or RNG.
    model.world = object()
    prediction = predict(model)[actor]
    assert prediction["mode"] == "chase_stochastic"
    assert math.dist(prediction["predicted"], old_position) > 0.45
    assert math.dist(prediction["predicted"], old_target_position) < math.dist(old_position, old_target_position)
    assert prediction["bound"] > 1
    assert model.nodes[actor].pos == old_position
    scene.step()
    actual = scene.entities[actor].pos
    assert math.dist(prediction["predicted"], actual) < 0.05
    assert math.dist(prediction["predicted"], actual) < math.dist(old_position, actual)


@pytest.mark.parametrize("model_class", [WorldLearner, UnifiedWorldModel])
def test_accidental_seek_toward_stationary_object_is_not_exact(model_class):
    scene = pursuit_scene(1)
    model = model_class(world=scene)
    warmup(model, 45)
    actor = next(eid for eid, e in scene.entities.items() if e.category == "player")
    position = model.nodes[actor].pos
    prediction = predict(model)[actor]
    assert prediction["mode"] == "bounded_stochastic"
    assert tuple(prediction["predicted"]) == position
    assert prediction["bound"] > model.hit_threshold + 0.05


def test_curiosity_radius_agrees_with_weak_seek_prediction():
    scene = pursuit_scene(1)
    model = CuriosityExplorer(world=scene)
    warmup(model, 45)
    actor = next(eid for eid, e in scene.entities.items() if e.category == "player")
    prediction = predict(model)[actor]
    assert model._prediction_bound(actor) == prediction["bound"]
