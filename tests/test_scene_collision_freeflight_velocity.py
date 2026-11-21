"""Collision registration must not replace a free-flight endpoint with its mean.

The expected position/velocity below are independent constant-acceleration
equations. Tests use the public scene API and never call physics.simulate for
their oracle. Existing contact displacement/reseeding semantics remain covered.
"""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lingshu.world.scene_simulator import PHYSICS_DYNAMICS, SceneSimulator

I3 = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))


def _free_scene(*, dt=.1, gravity=10., up=1, boxes=2, obb=False,
                p0=(64., 1000., 64.), v0=(.25, .5, -.1)):
    s = SceneSimulator(size=128, seed=42)
    s.set_environment(gravity=gravity, dt=dt, up=up, incline_axis=(up + 1) % 3)
    a = s.add_entity("ball", "wander", p0, 0.)
    b = s.add_entity("wall", "wander", (115., 1000., 115.), 0.)
    s.entities[a].attrs["physics"] = PHYSICS_DYNAMICS
    s.set_entity_physics(a, p0=p0, v0=v0, mass=2.5)
    if boxes:
        box = {"half": (.25, .25, .25), "rot": I3} if obb else (.25, .25, .25)
        s.set_entity_collision_box(a, box)
    if boxes > 1:
        s.set_entity_collision_box(b, (.25, .25, .25), static=True)
    return s, a, p0, v0


def _oracle(p0, v0, t, gravity, up):
    p = tuple(p0[i] + v0[i] * t - (gravity * t * t / 2 if i == up else 0.)
              for i in range(3))
    v = tuple(v0[i] - (gravity * t if i == up else 0.) for i in range(3))
    return p, v


def _assert_close(actual, expected, label):
    assert all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9)
               for a, b in zip(actual, expected)), (label, actual, expected)


@pytest.mark.parametrize("up", (0, 1, 2))
@pytest.mark.parametrize("dt", (.01, .05, .2, .5))
@pytest.mark.parametrize("gravity", (.5, 9.80665))
def test_freeflight_all_axes_and_step_sizes(up, dt, gravity):
    s, a, p0, v0 = _free_scene(dt=dt, gravity=gravity, up=up)
    for tick in range(1, 7):
        s.step()
        e = s.entities[a]
        p, v = _oracle(p0, v0, tick * dt, gravity, up)
        assert not any(c["hit"] for c in s.collision_report()["contacts"])
        _assert_close(e.pos, p, "SCENE_FREEFLIGHT_POSITION")
        _assert_close(e.velocity, v, "SCENE_FREEFLIGHT_ENDPOINT_VELOCITY")


def test_noncontact_keeps_instantaneous_endpoint_velocity():
    s, a, p0, v0 = _free_scene()
    s.step()
    _, expected = _oracle(p0, v0, .1, 10., 1)
    _assert_close(s.entities[a].velocity, expected, "SCENE_ENDPOINT_IS_NOT_MEAN")


def test_noncontact_keeps_original_physics_state():
    s, a, _, _ = _free_scene()
    original = copy.deepcopy(s.entities[a].attrs["physics_state"])
    s.step(4)
    assert s.entities[a].attrs["physics_state"] == original, "SCENE_NO_CONTACT_RESEEDED"


def test_physics_readout_matches_public_noncontact_endpoint():
    s, a, _, _ = _free_scene()
    s.step()
    e = s.entities[a]
    assert e.velocity == e.attrs["physics_readout"]["vel"], "SCENE_READOUT_ENDPOINT_DRIFT"


@pytest.mark.parametrize("boxes", (0, 1, 2))
def test_adding_a_distant_box_keeps_the_same_closed_form_trajectory(boxes):
    s, a, p0, v0 = _free_scene(boxes=boxes)
    s.step(10)
    p, v = _oracle(p0, v0, 1., 10., 1)
    _assert_close(s.entities[a].pos, p, "SCENE_BOX_COUNT_CHANGED_FREEFLIGHT")
    _assert_close(s.entities[a].velocity, v, "SCENE_BOX_COUNT_CHANGED_VELOCITY")


def test_noncontact_obb_keeps_endpoint_and_initial_state():
    s, a, p0, v0 = _free_scene(obb=True)
    original = copy.deepcopy(s.entities[a].attrs["physics_state"])
    s.step(8)
    p, v = _oracle(p0, v0, .8, 10., 1)
    _assert_close(s.entities[a].pos, p, "SCENE_OBB_FREEFLIGHT_POSITION")
    _assert_close(s.entities[a].velocity, v, "SCENE_OBB_FREEFLIGHT_VELOCITY")
    assert s.entities[a].attrs["physics_state"] == original


def _contact_scene(*, overlap=False, extra_free=False, reverse=False):
    s = SceneSimulator(size=128, seed=42)
    dt, g = .25, .5
    s.set_environment(gravity=g, dt=dt)
    ids = {}
    entries = [("mover", (10., 10., 10.)),
               ("wall", ((10.25 if overlap else 12.5), 10., 10.))]
    if extra_free:
        entries.insert(0, ("free", (64., 1000., 64.)))
    for name, pos in (reversed(entries) if reverse else entries):
        ids[name] = s.add_entity(name, "wander", pos, 0.)
    for name, eid in ids.items():
        if name != "wall":
            s.entities[eid].attrs["physics"] = PHYSICS_DYNAMICS
            v0 = ((0. if overlap else 10.), g * dt / 2, 0.) if name == "mover" else (.25, .5, -.1)
            s.set_entity_physics(eid, v0=v0, beta=.4, radius=.3, mass=2.5)
        s.set_entity_collision_box(eid, (.5, .5, .5), static=name == "wall")
    return s, ids, dt, g


def test_real_collision_still_reseeds_at_the_correct_contact_surface():
    s, ids, dt, _ = _contact_scene()
    s.step()
    e = s.entities[ids["mover"]]
    _assert_close(e.pos, (11.5, 10., 10.), "SCENE_CONTACT_WAS_NOT_STOPPED")
    st = e.attrs["physics_state"]
    assert st["p0"] == e.pos and st["v0"] == e.velocity, "SCENE_CONTACT_WAS_NOT_RESEEDED"
    assert st["t_ref"] == dt
    assert (st["beta"], st["radius"], st["mass"]) == (.4, .3, 2.5)
    assert any(c["hit"] for c in s.collision_report()["contacts"])
    s.step(4)
    assert s.entities[ids["mover"]].pos[0] <= 11.5 + 1e-9


def test_initial_overlap_still_reseeds_after_penetration_correction():
    s, ids, dt, _ = _contact_scene(overlap=True)
    s.step()
    e = s.entities[ids["mover"]]
    _assert_close(e.pos, (9.25, 10., 10.), "SCENE_OVERLAP_WAS_NOT_RESOLVED")
    assert e.attrs["physics_state"]["p0"] == e.pos
    assert e.attrs["physics_state"]["t_ref"] == dt
    assert any(c["resolved"] for c in s.collision_report()["contacts"])


@pytest.mark.parametrize("reverse", (False, True))
def test_colliding_pair_does_not_reseed_a_third_free_entity(reverse):
    s, ids, dt, g = _contact_scene(extra_free=True, reverse=reverse)
    free = s.entities[ids["free"]]
    original = copy.deepcopy(free.attrs["physics_state"])
    s.step()
    p, v = _oracle((64., 1000., 64.), (.25, .5, -.1), dt, g, 1)
    _assert_close(free.pos, p, "SCENE_THIRD_ENTITY_POSITION_CHANGED")
    _assert_close(free.velocity, v, "SCENE_THIRD_ENTITY_VELOCITY_CHANGED")
    assert free.attrs["physics_state"] == original, "SCENE_THIRD_ENTITY_RESEEDED"
    mover = s.entities[ids["mover"]]
    assert mover.attrs["physics_state"]["p0"] == mover.pos
    assert mover.pos[0] == pytest.approx(11.5)


def test_kinematic_velocity_still_records_collision_corrected_displacement():
    s = SceneSimulator(size=24, seed=42)
    a = s.add_entity("a", "wander", (5., 1.5, 5.), 0.)
    b = s.add_entity("b", "wander", (8., 1.5, 5.), 0.)
    s.set_entity_collision_box(a, (.5, .5, .5))
    s.set_entity_collision_box(b, (.5, .5, .5), static=True)
    s.set_command(a, "east", 4.)
    s.step()
    assert s.entities[a].pos == (7., 1.5, 5.)
    assert s.entities[a].velocity == (2., 0., 0.), "SCENE_KINEMATIC_DISPLACEMENT_CHANGED"


def _boundary_scene(gap, vx, both_static=False):
    s = SceneSimulator(size=128, seed=42)
    s.set_environment(gravity=.5, dt=.25)
    a = s.add_entity("a", "wander", (10., 1000., 10.), 0.)
    b = s.add_entity("b", "wander", (10. + gap, 1000., 10.), 0.)
    s.entities[a].attrs["physics"] = PHYSICS_DYNAMICS
    # Net vertical displacement is exactly zero, while the endpoint velocity is
    # -.0625. This distinguishes displacement, instantaneous velocity and hits.
    s.set_entity_physics(a, v0=(vx, .0625, 0.))
    s.set_entity_collision_box(a, (.5, .5, .5), static=both_static)
    s.set_entity_collision_box(b, (.5, .5, .5), static=True)
    return s, a


@pytest.mark.parametrize("gap,vx,expected_x", ((1., 10., 10.), (0., 0., 9.),
                                             (.25, -3., 9.25)))
def test_start_contact_keeps_existing_reseed_semantics(gap, vx, expected_x):
    s, a = _boundary_scene(gap, vx)
    s.step()
    e = s.entities[a]
    assert s.collision_report()["contacts"][0]["hit"] is True
    assert e.pos == (expected_x, 1000., 10.)
    assert e.attrs["physics_state"]["t_ref"] == .25, "SCENE_REPORTED_HIT_MUST_RESEED"
    assert e.attrs["physics_state"]["p0"] == e.pos
    assert e.velocity[1] == 0., "SCENE_REPORTED_HIT_VELOCITY_CHANGED"


def test_fully_overlapping_static_boxes_keep_existing_contact_velocity():
    s, a = _boundary_scene(0., 0., both_static=True)
    original = copy.deepcopy(s.entities[a].attrs["physics_state"])
    s.step()
    e = s.entities[a]
    assert s.collision_report()["contacts"][0]["hit"] is True
    assert e.pos == (10., 1000., 10.)
    assert e.velocity == (0., 0., 0.), "SCENE_STATIC_CONTACT_VELOCITY_CHANGED"
    assert e.attrs["physics_state"] == original


def test_end_of_tick_touch_is_not_a_collision_and_keeps_endpoint_velocity():
    s, a = _boundary_scene(3.5, 10.)
    original = copy.deepcopy(s.entities[a].attrs["physics_state"])
    s.step()
    e = s.entities[a]
    assert s.collision_report()["contacts"][0]["hit"] is False
    assert e.pos == (12.5, 1000., 10.)
    assert e.velocity == (10., -.0625, 0.), "SCENE_END_TOUCH_REPLACED_ENDPOINT"
    assert e.attrs["physics_state"] == original
