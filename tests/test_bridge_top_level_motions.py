"""The animation extension must coexist with the frozen geometry spec contract."""
import copy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bridges" / "blender"))
from wm_bridge.factory.anim import eval as E
from wm_bridge.factory.geo import spec as S


def _spec():
    return {"schema": "wmfactory.spec/1", "name": "motion_extension", "scene": {"fps": 24},
            "links": [{"name": "bob", "anchor": [0, 0, 0], "parts": []}], "joints": []}


def _motion(kind="spring"):
    base = {"link": "bob", "kind": kind, "axis": "y"}
    if kind == "spring":
        return dict(base, omega0=2.0, zeta=0.15, amp=30.0, unit="deg")
    if kind == "pendulum":
        return dict(base, omega0=2.0, zeta=0.1, theta0_deg=4.0)
    return dict(base, p0=[1, -2, 3], v0=[0.25, 0, 1], g=[0, 0, -9.80665])


def _invoke(entry, spec, motions=None):
    if entry == "eval_link":
        return E.eval_link(spec, 12, "bob", motions=motions)
    if entry == "eval_frames":
        return E.eval_frames(spec, [0, 6, 12], motions=motions)
    if entry == "to_oracle_records":
        return E.to_oracle_records(spec, 12, motions=motions)
    if entry == "tracks_to_oracle_doc":
        return E.tracks_to_oracle_doc(spec, [0, 6, 12], motions=motions)
    return E.eval_at(spec, 12, motions=motions)


@pytest.mark.parametrize("kind", ["spring", "pendulum", "projectile"])
@pytest.mark.parametrize("entry", ["eval_at", "eval_link", "eval_frames",
                                  "to_oracle_records", "tracks_to_oracle_doc"])
def test_top_level_motions_match_explicit_public_path(kind, entry):
    spec, motions = _spec(), [_motion(kind)]
    expected = _invoke(entry, spec, motions)
    bundled = dict(spec, motions=copy.deepcopy(motions))
    before = copy.deepcopy(bundled)
    try:
        actual = _invoke(entry, bundled)
    except S.SpecError as exc:
        pytest.fail("top-level motions must reach animation evaluation: " + str(exc))
    assert actual == expected, "top-level motion must preserve the explicit path's pose"
    assert bundled == before, "animation must not mutate the caller's spec or motions"


def test_normalize_removes_only_animation_extension():
    spec = _spec()
    expected = S.normalize_spec(spec)
    spec["motions"] = [_motion()]
    actual = E.normalize(spec)
    assert actual == expected, "normalization must pass geometry fields to the frozen validator"
    assert "motions" not in actual
    with pytest.raises(S.SpecError):
        S.normalize_spec(spec)


def test_normalize_does_not_modify_caller_input():
    spec = dict(_spec(), motions=[_motion()])
    before = copy.deepcopy(spec)
    E.normalize(spec)
    assert spec == before, "normalization must not remove motions from caller input"


def test_explicit_motions_override_top_level_declarations():
    spec = dict(_spec(), motions=[_motion("projectile")])
    explicit = [_motion("spring")]
    expected = E.eval_at(_spec(), 12, motions=explicit)
    actual = E.eval_at(spec, 12, motions=explicit)
    assert actual == expected, "explicit motions must take precedence over top-level declarations"


def test_explicit_empty_motions_disable_top_level_motion():
    spec = dict(_spec(), motions=[_motion()])
    actual = E.eval_at(spec, 12, motions=[])
    assert actual == E.eval_at(_spec(), 12), "explicit empty motions must disable top-level fallback"


@pytest.mark.parametrize("value", [None, []], ids=["none", "empty"])
def test_optional_empty_top_level_motion_keeps_static_behavior(value):
    assert E.eval_at(dict(_spec(), motions=value), 12) == E.eval_at(_spec(), 12)


@pytest.mark.parametrize("location", ["root", "scene", "link"])
def test_other_unknown_keys_remain_rejected(location):
    spec = dict(_spec(), motions=[_motion()])
    if location == "root":
        spec["motinos"] = []
    elif location == "scene":
        spec["scene"]["motions"] = []
    else:
        spec["links"][0]["motions"] = []
    with pytest.raises(S.SpecError):
        E.eval_at(spec, 12)


@pytest.mark.parametrize("motions", [[1], [{"link": "bob", "kind": "unknown"}],
                                     [_motion(), _motion()]],
                         ids=["non_object", "unknown_kind", "duplicate_link"])
def test_malformed_top_level_motion_still_raises_anim_error(motions):
    with pytest.raises(E.AnimError):
        E.eval_at(dict(_spec(), motions=motions), 12)


def test_explicit_parameter_can_override_invalid_top_level_motion():
    spec = dict(_spec(), motions=[{"link": "bob", "kind": "unknown"}])
    explicit = [_motion()]
    assert E.eval_at(spec, 12, motions=explicit) == E.eval_at(_spec(), 12, motions=explicit)


def test_normalized_tuple_ease_keys_remain_accepted():
    plain = _spec()
    plain["joints"] = [{"link": "bob", "kind": "ease", "axis": "y", "unit": "deg",
                        "keys": [[0.0, 0.0], [1.0, 45.0]]}]
    normalized = S.normalize_spec(plain)
    assert isinstance(normalized["joints"][0]["keys"], tuple)
    bundled = dict(normalized, motions=[_motion()])
    before = copy.deepcopy(bundled)
    expected = E.eval_at(normalized, 12, motions=[_motion()])
    assert E.eval_at(bundled, 12) == expected
    assert bundled == before


def test_geometry_only_control_keeps_normalized_result():
    spec = _spec()
    assert E.normalize(spec) == S.normalize_spec(spec)
    assert E.eval_at(spec, 0) == E.eval_at(spec, 12)
