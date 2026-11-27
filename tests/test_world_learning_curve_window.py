"""Learning curve rows share the evaluation window and its step budget."""
import pytest

from lingshu.world.scene_simulator import SceneSimulator
from lingshu.world.world_learner import WorldLearner


@pytest.fixture
def walking_learner():
    scene = SceneSimulator(seed=0)
    scene.add_path("square", [(5, 1.5, 5), (7, 1.5, 5), (7, 1.5, 7), (5, 1.5, 7)])
    scene.add_entity("walker", behavior="follow", pos=(5, 1.5, 5), goal="square", speed=0.8)
    return WorldLearner(world=scene, seed=0)


@pytest.mark.parametrize("eval_ticks,observations", [(3, [7, 14]), (0, [4, 8])],
                         ids=["with-evaluation", "zero-evaluation"])
def test_curve_advances_once_per_evaluation(walking_learner, eval_ticks, observations):
    result = walking_learner.learning_curve(epochs=2, per_epoch_ticks=4, eval_ticks=eval_ticks)

    assert walking_learner.world.tick_count == observations[-1], "curve advanced beyond its evaluation budget"
    assert len(walking_learner.history) == observations[-1]
    assert [row["observations"] for row in result["curve"]] == observations
    assert [row["tick"] for row in walking_learner.evals] == observations
    for row in result["curve"]:
        assert (row["learned_rate"] is None) == (eval_ticks == 0)
        assert (row["mean_distance"] is None) == (eval_ticks == 0)


def test_curve_distance_matches_its_evaluation(walking_learner):
    result = walking_learner.learning_curve(epochs=2, per_epoch_ticks=4, eval_ticks=3)
    distances = [row["mean_distance"] for row in walking_learner.evals]

    assert [row["outcomes"] for row in walking_learner.evals] == [3, 3]
    assert [row["mean_distance"] for row in result["curve"]] == distances, "curve mixed evaluation windows"
    assert result["distance_drop"] == round(distances[0] - distances[1], 4)
