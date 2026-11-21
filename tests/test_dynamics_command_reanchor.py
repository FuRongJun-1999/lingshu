"""A finite-duration translation command must become the next physics anchor.

F08 reports rounded command velocity in units/tick. Physics v0 is units/second,
hence uses the actual endpoint difference, without re-rounding, divided by dt.
Stay explicitly stops: its existing coordinate quantization adds no momentum.
Every expected free-flight point below comes from independent equations.
"""
from __future__ import annotations
import copy
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lingshu.world.scene_simulator import Environment, PHYSICS_DYNAMICS, SceneSimulator
from lingshu.world.seven_layer_loop import SevenLayerLoop

DIRECTIONS = {"east": (1., 0.), "west": (-1., 0.), "north": (0., -1.),
              "south": (0., 1.), "stay": (0., 0.)}


def scene(dt=.1, boxes=0, seeded=True, pos=(64., 1000., 64.), v0=(0., 0., 0.)):
    s = SceneSimulator(size=128, seed=42)
    s.set_environment(gravity=10., dt=dt)
    a = s.add_entity("actor", "wander", pos, 0.)
    s.entities[a].attrs["physics"] = PHYSICS_DYNAMICS
    if seeded:
        s.set_entity_physics(a, v0=v0, beta=.4, radius=.3, mass=2.5)
    if boxes:
        s.set_entity_collision_box(a, (.25, .25, .25))
    return s, a


def advance(pos, velocity, elapsed, gravity=10.):
    return (tuple(pos[i] + velocity[i] * elapsed - (gravity * elapsed**2 / 2 if i == 1 else 0.)
                  for i in range(3)),
            tuple(velocity[i] - (gravity * elapsed if i == 1 else 0.) for i in range(3)))


def close(actual, expected, message):
    assert all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9)
               for a, b in zip(actual, expected)), (message, actual, expected)


def command_oracle(old, action, step, size=128):
    dx, dz = DIRECTIONS[action]
    pos = (round(max(.5, min(size - .5, old[0] + dx * step)), 2), old[1],
           round(max(.5, min(size - .5, old[2] + dz * step)), 2))
    displacement = (round(pos[0] - old[0], 2), 0., round(pos[2] - old[2], 2))
    return pos, displacement


@pytest.mark.parametrize("action", tuple(DIRECTIONS))
@pytest.mark.parametrize("dt", (.05, .1, 1.))
@pytest.mark.parametrize("boxes", (0, 1))
def test_command_then_resume_uses_real_endpoint_and_velocity(action, dt, boxes):
    s, a = scene(dt=dt, boxes=boxes)
    s.step(3)
    old = s.entities[a].pos
    expected_pos, displacement = command_oracle(old, action, 2.)
    s.set_command(a, action, 2.)
    s.step()
    e = s.entities[a]
    close(e.pos, expected_pos, "COMMAND_ENDPOINT_CHANGED")
    close(e.velocity, displacement, "COMMAND_PUBLIC_VELOCITY_UNITS_CHANGED")
    physical_v = ((0., 0., 0.) if action == "stay" else
                  tuple((expected_pos[i] - old[i]) / dt for i in range(3)))
    for n in (1, 2, 3):
        s.step()
        p, v = advance(expected_pos, physical_v, n * dt)
        close(e.pos, p, "COMMAND_TRANSLATION_WAS_REPLAYED_FROM_OLD_ANCHOR")
        close(e.velocity, v, "COMMAND_RESUME_VELOCITY_HAS_WRONG_UNITS")


def test_command_reanchors_at_current_time_without_changing_public_velocity():
    s, a = scene(dt=.1)
    s.step(3)
    s.set_command(a, "east", 2.)
    s.step()
    e = s.entities[a]
    assert e.velocity == (2., 0., 0.), "COMMAND_PUBLIC_VELOCITY_UNITS_CHANGED"
    st = e.attrs["physics_state"]
    assert st["p0"] == e.pos, "COMMAND_PHYSICS_POSITION_ANCHOR_STALE"
    assert st["v0"] == (20., 0., 0.), "COMMAND_PHYSICS_VELOCITY_NOT_PER_SECOND"
    assert st["t_ref"] == .4, "COMMAND_PHYSICS_TIME_ANCHOR_STALE"
    assert (st["beta"], st["radius"], st["mass"]) == (.4, .3, 2.5)


def test_first_command_initializes_a_physics_anchor_without_a_blank_resume_tick():
    s, a = scene(dt=.2, seeded=False)
    s.set_command(a, "north", .75)
    s.step()
    e = s.entities[a]
    command_pos = e.pos
    s.step()
    p, v = advance(command_pos, (0., 0., -3.75), .2)
    close(e.pos, p, "COMMAND_FIRST_RESUME_WAS_ONLY_A_SEED")
    close(e.velocity, v, "COMMAND_FIRST_RESUME_VELOCITY_WRONG")


def test_multiple_translation_direction_changes_and_stay_stop():
    s, a = scene(dt=.1)
    s.step(2)
    e = s.entities[a]
    for action, step in (("east", 2.), ("north", .5), ("west", 1.), ("stay", 3.), ("south", .75)):
        old = e.pos
        pos, displacement = command_oracle(e.pos, action, step)
        s.set_command(a, action, step)
        s.step()
        close(e.pos, pos, "COMMAND_SEQUENCE_ENDPOINT_WRONG")
        close(e.velocity, displacement, "COMMAND_SEQUENCE_PUBLIC_VELOCITY_CHANGED")
        v0 = ((0., 0., 0.) if action == "stay" else
              tuple((pos[i] - old[i]) / .1 for i in range(3)))
        p, v = advance(pos, v0, .1)
        s.step()
        close(e.pos, p, "COMMAND_SEQUENCE_RESUME_IGNORED_LATEST_ACTION")
        close(e.velocity, v, "COMMAND_SEQUENCE_RESUME_VELOCITY_WRONG")


def test_stay_stops_prior_velocity_including_the_vertical_component():
    s, a = scene(dt=.1, v0=(2., 3., 1.))
    s.step(3)
    pos = s.entities[a].pos
    s.set_command(a, "stay", 1.)
    s.step()
    assert s.entities[a].pos == pos and s.entities[a].velocity == (0., 0., 0.)
    s.step()
    p, v = advance(pos, (0., 0., 0.), .1)
    close(s.entities[a].pos, p, "COMMAND_STAY_DID_NOT_STOP_PRIOR_MOMENTUM")
    close(s.entities[a].velocity, v, "COMMAND_STAY_RESUMED_OLD_VELOCITY")


def test_clamping_uses_actual_displacement_for_the_physics_velocity():
    s, a = scene(dt=.1, pos=(127.25, 1000., 64.))
    s.set_command(a, "east", 5.)
    s.step()
    e = s.entities[a]
    assert e.pos == (127.5, 1000., 64.) and e.velocity == (.25, 0., 0.)
    assert e.attrs["physics_state"]["v0"] == (2.5, 0., 0.), "COMMAND_REQUESTED_STEP_REPLACED_ACTUAL_MOTION"


def test_quantized_zero_move_stops_instead_of_using_the_requested_step():
    s, a = scene(dt=.1, v0=(1., 2., 0.))
    s.step(2)
    pos = s.entities[a].pos
    s.set_command(a, "east", .001)
    s.step()
    assert s.entities[a].velocity == (0., 0., 0.)
    s.step()
    p, v = advance(pos, (0., 0., 0.), .1)
    close(s.entities[a].pos, p, "COMMAND_QUANTIZED_ZERO_DID_NOT_STOP")
    close(s.entities[a].velocity, v, "COMMAND_QUANTIZED_ZERO_VELOCITY_WRONG")


def test_actual_collision_reseeds_the_postcollision_endpoint_and_keeps_parameters():
    s, a = scene(dt=.25, pos=(10., 10., 10.))
    b = s.add_entity("wall", "wander", (12.5, 10., 10.), 0.)
    s.set_entity_collision_box(a, (.5, .5, .5))
    s.set_entity_collision_box(b, (.5, .5, .5), static=True)
    s.set_command(a, "east", 3.)
    s.step()
    e = s.entities[a]
    assert any(c["hit"] for c in s.collision_report()["contacts"])
    assert e.pos == (11.5, 10., 10.)
    st = e.attrs["physics_state"]
    assert st["p0"] == e.pos and st["v0"] == (6., 0., 0.) and st["t_ref"] == .25
    assert (st["beta"], st["radius"], st["mass"]) == (.4, .3, 2.5)


def test_real_seven_layer_actuator_then_clear_does_not_undo_execution():
    s, a = scene(dt=.1)
    loop = SevenLayerLoop(world=s, size=128, seed=42, budget=1, policy="round_robin")
    for _ in range(3):
        loop.step()
    old = s.entities[a].pos
    loop.configure_action(a, (90., 1000., 64.), step_size=2., policy="fixed", fixed_action="east")
    executed = loop.step()["L7_action"]["executed"]
    assert executed["self_error"] == 0. and executed["observed_to"][0] == old[0] + 2.
    loop.clear_action()
    loop.step()
    assert s.entities[a].pos[0] == old[0] + 4., "SEVEN_LAYER_CLEAR_UNDID_REAL_EXECUTION"


def test_no_command_keeps_the_original_closed_form_state():
    s, a = scene(dt=.1, v0=(.25, .5, -.1))
    original = copy.deepcopy(s.entities[a].attrs["physics_state"])
    s.step(4)
    p, v = advance((64., 1000., 64.), (.25, .5, -.1), .4)
    close(s.entities[a].pos, p, "UNCOMMANDED_PHYSICS_POSITION_CHANGED")
    close(s.entities[a].velocity, v, "UNCOMMANDED_PHYSICS_VELOCITY_CHANGED")
    assert s.entities[a].attrs["physics_state"] == original


def test_kinematic_commands_keep_their_existing_state_and_units():
    s = SceneSimulator(size=128, seed=42)
    s.set_environment(dt=.1)
    a = s.add_entity("actor", "wander", (64., 1000., 64.), 0.)
    s.set_command(a, "east", 2.)
    s.step()
    assert s.entities[a].pos == (66., 1000., 64.) and s.entities[a].velocity == (2., 0., 0.)
    assert "physics_state" not in s.entities[a].attrs, "COMMAND_CHANGED_KINEMATIC_STATE"


def test_unrepresentable_physical_command_velocity_does_not_poison_the_entity():
    s, a = scene(dt=1e-320)
    e = s.entities[a]
    before = copy.deepcopy(e.to_dict())
    s.set_command(a, "east", 1.)
    with pytest.raises(ValueError, match="受控动力学初值"):
        s.step()
    assert e.to_dict() == before, "COMMAND_INVALID_VELOCITY_PARTIALLY_APPLIED"


@pytest.mark.parametrize("representation", ("dict", "environment"))
def test_entity_environment_override_is_used_when_physics_resumes(representation):
    s, a = scene(dt=.1)
    if representation == "dict":
        s.set_entity_environment(a, gravity=7.)
    else:
        s.entities[a].attrs["environment"] = Environment(gravity=7., dt=.1)
    s.step(3)
    s.set_command(a, "east", .5)
    s.step()
    pos = s.entities[a].pos
    s.step(2)
    p, v = advance(pos, (5., 0., 0.), .2, gravity=7.)
    close(s.entities[a].pos, p, "COMMAND_IGNORED_ENTITY_ENVIRONMENT")
    close(s.entities[a].velocity, v, "COMMAND_ENTITY_ENVIRONMENT_VELOCITY_WRONG")


def test_command_preserves_custom_metadata_without_mutating_the_previous_state():
    s, a = scene()
    previous = s.entities[a].attrs["physics_state"]
    previous["audit_marker"] = {"source": "external", "sequence": [1, 2]}
    original = copy.deepcopy(previous)
    s.set_command(a, "east", 2.)
    s.step()
    current = s.entities[a].attrs["physics_state"]
    assert current is not previous and previous == original, "COMMAND_MUTATED_OLD_STATE"
    assert current.get("audit_marker") == original["audit_marker"], "COMMAND_DROPPED_STATE_METADATA"
    assert tuple(current.get(k) for k in ("beta", "radius", "mass")) == (.4, .3, 2.5), "COMMAND_DROPPED_PHYSICAL_PARAMETERS"


def test_unrepresentable_command_reference_time_does_not_poison_the_entity():
    s, a = scene(dt=1e308)
    s.set_command(a, "east", 1.)
    s.step()
    e = s.entities[a]
    before = copy.deepcopy(e.to_dict())
    s.set_command(a, "east", 1.)
    with pytest.raises(ValueError, match="受控动力学初值"):
        s.step()
    assert e.to_dict() == before, "COMMAND_INVALID_TIME_PARTIALLY_APPLIED"


@pytest.mark.parametrize("up", (0, 1, 2))
def test_command_resume_respects_the_gravity_axis(up):
    s, a = scene(dt=.125)
    s.set_environment(up=up, incline_axis=1 if up == 0 else 0)
    s.step(3)
    old = s.entities[a].pos
    s.set_command(a, "north", .5)
    s.step()
    p0 = s.entities[a].pos
    physical_v = tuple((p0[i] - old[i]) / .125 for i in range(3))
    s.step(2)
    t = .25
    expected_p = tuple(p0[i] + physical_v[i] * t - (5. * t*t if i == up else 0.)
                       for i in range(3))
    expected_v = tuple(physical_v[i] - (10. * t if i == up else 0.)
                       for i in range(3))
    close(s.entities[a].pos, expected_p, "COMMAND_RESUME_GRAVITY_AXIS_WRONG")
    close(s.entities[a].velocity, expected_v, "COMMAND_RESUME_GRAVITY_AXIS_VELOCITY_WRONG")


def test_stay_resumes_quadratic_drag_from_rest_instead_of_the_old_momentum():
    s, a = scene(dt=.125)
    s.set_environment(drag_c=.8, mass=2.)
    s.step(3)
    s.set_command(a, "stay", 1.)
    s.step()
    p0 = s.entities[a].pos
    s.step(4)
    t, beta, g = .5, .4, 10.
    fall = math.log(math.cosh(math.sqrt(g * beta) * t)) / beta
    v = -math.sqrt(g / beta) * math.tanh(math.sqrt(g * beta) * t)
    close(s.entities[a].pos, (p0[0], p0[1] - fall, p0[2]), "COMMAND_DRAG_RESTART_POSITION_WRONG")
    close(s.entities[a].velocity, (0., v, 0.), "COMMAND_DRAG_RESTART_VELOCITY_WRONG")
    assert s.environment_readout(a)["readout"]["kind"] == "drag2"


@pytest.mark.parametrize("rolling", (False, True))
def test_stay_resumes_the_existing_slope_model_from_its_new_anchor(rolling):
    s, a = scene(dt=.125)
    s.set_environment(slope_deg=30., mu_k=.1)
    if not rolling:
        s.entities[a].attrs["physics_state"].pop("beta")
    s.step(3)
    s.set_command(a, "stay", 1.)
    s.step()
    p0 = s.entities[a].pos
    s.step(2)
    theta, t = math.pi / 6., .25
    accel = (10. * math.sin(theta) / 1.4 if rolling else
             10. * (math.sin(theta) - .1 * math.cos(theta)))
    direction = (math.cos(theta), -math.sin(theta), 0.)
    p = tuple(p0[i] + .5 * accel * t*t * direction[i] for i in range(3))
    v = tuple(accel * t * direction[i] for i in range(3))
    close(s.entities[a].pos, p, "COMMAND_SLOPE_RESTART_POSITION_WRONG")
    close(s.entities[a].velocity, v, "COMMAND_SLOPE_RESTART_VELOCITY_WRONG")
    assert s.environment_readout(a)["readout"]["kind"] == ("rolling" if rolling else "slope_friction")


@pytest.mark.parametrize("action,step,old_x,expected_x,expected_vx", (
    ("east", 2., 64.003, 66., 19.97),
    ("west", 2., 64.003, 62., -20.03),
    ("east", .001, 64.003, 64., -.03),
    ("west", .001, 64.007, 64.01, .03),
    ("east", 5., 127.499, 127.5, .01),
    ("west", 5., .501, .5, -.01),
))
def test_continuous_position_uses_unrounded_actual_endpoint_displacement(
        action, step, old_x, expected_x, expected_vx):
    s, a = scene(dt=.1, pos=(old_x, 1000.007, 64.))
    s.set_command(a, action, step)
    s.step()
    e = s.entities[a]
    assert e.pos[0] == expected_x
    assert e.velocity[0] == round(expected_x - old_x, 2), "COMMAND_PUBLIC_VELOCITY_UNITS_CHANGED"
    assert e.attrs["physics_state"]["v0"][0] == pytest.approx(expected_vx, abs=1e-12), "COMMAND_REROUNDED_CONTINUOUS_DISPLACEMENT"
    p0 = e.pos
    s.step()
    p, v = advance(p0, (expected_vx, 0., 0.), .1)
    p = (max(.5, min(127.5, p[0])), p[1], p[2])
    close(e.pos, p, "COMMAND_CONTINUOUS_RESUME_POSITION_WRONG")
    close(e.velocity, v, "COMMAND_CONTINUOUS_RESUME_VELOCITY_WRONG")


@pytest.mark.parametrize("action,step,expected_z,expected_vz", (
    ("north", 2., 62.01, -19.97),
    ("south", 2., 66.01, 20.03),
    ("north", .001, 64.01, .03),
))
def test_continuous_z_endpoint_uses_its_actual_displacement(action, step, expected_z, expected_vz):
    s, a = scene(dt=.1, pos=(64., 1000.007, 64.007))
    s.set_command(a, action, step)
    s.step()
    e = s.entities[a]
    assert e.pos[2] == expected_z
    assert e.attrs["physics_state"]["v0"][2] == pytest.approx(expected_vz, abs=1e-12), "COMMAND_REROUNDED_CONTINUOUS_Z_DISPLACEMENT"
    p0 = e.pos
    s.step()
    p, v = advance(p0, (0., 0., expected_vz), .1)
    close(e.pos, p, "COMMAND_CONTINUOUS_Z_RESUME_POSITION_WRONG")
    close(e.velocity, v, "COMMAND_CONTINUOUS_Z_RESUME_VELOCITY_WRONG")


def test_stay_quantizes_continuous_position_without_creating_new_momentum():
    s, a = scene(dt=.1, pos=(64.003, 1000.007, 64.007), v0=(2., 3., 1.))
    s.set_command(a, "stay", 1.)
    s.step()
    e = s.entities[a]
    assert e.pos == (64., 1000.007, 64.01)
    assert e.velocity == (0., 0., 0.)
    assert e.attrs["physics_state"]["v0"] == (0., 0., 0.), "COMMAND_STAY_QUANTIZATION_CREATED_MOMENTUM"
    p0 = e.pos
    s.step()
    p, v = advance(p0, (0., 0., 0.), .1)
    close(e.pos, p, "COMMAND_STAY_CONTINUOUS_POSITION_DRIFTED")
    close(e.velocity, v, "COMMAND_STAY_CONTINUOUS_VELOCITY_WRONG")
