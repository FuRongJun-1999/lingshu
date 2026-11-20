"""Standalone bridge exact-motion guards; no Blender or lingshu imports."""
import copy
import math
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "bridges" / "blender"
sys.path.insert(0, str(BRIDGE))
from wm_bridge.factory.anim import eval as E


def _motion(initial=math.radians(60), **updates):
    motion = {"link": "bob", "kind": "pendulum", "axis": "y", "exact": True,
              "theta0_deg": math.degrees(initial), "omega0": 1.0, "zeta": 0.0}
    motion.update(updates)
    return motion


def _spec(fps=120.0):
    return {"schema": "wmfactory.spec/1", "name": "pendulum_guard", "scene": {"fps": fps},
            "links": [{"name": "bob", "anchor": [0, 0, 0], "parts": []}], "joints": []}


def _eval_angle(motion, elapsed):
    spec = _spec(1.0 / abs(elapsed) if elapsed else 120.0)
    frame = (1 if elapsed > 0 else -1) if elapsed else 0
    pose = E.eval_at(spec, frame, motions=[motion])["bob"]
    assert pose["pos"] == (0.0, 0.0, 0.0)
    quat = pose["quat"]
    return 2.0 * math.atan2(quat[2], quat[0])


def _rk4_release(initial, elapsed, dt=1.0 / 4096):
    theta, velocity = initial, 0.0
    steps = max(1, math.ceil(abs(elapsed) / dt))
    step = elapsed / steps
    for _ in range(steps):
        a1, b1 = velocity, -math.sin(theta)
        a2, b2 = velocity + step * b1 / 2, -math.sin(theta + step * a1 / 2)
        a3, b3 = velocity + step * b2 / 2, -math.sin(theta + step * a2 / 2)
        a4, b4 = velocity + step * b3, -math.sin(theta + step * a3)
        theta += step * (a1 + 2 * a2 + 2 * a3 + a4) / 6
        velocity += step * (b1 + 2 * b2 + 2 * b3 + b4) / 6
    return theta


def _quarter_period(initial):
    a, b = 1.0, math.cos(initial / 2)
    for _ in range(32):
        a, b = (a + b) / 2, math.sqrt(a * b)
    return math.pi / (2 * a)


def test_exact_motion_eval_matches_independent_rk4():
    for degrees in (-179.9, -120, -60, -5, 0, 5, 30, 60, 120, 170, 179.9):
        initial = math.radians(degrees)
        for elapsed in (0.25, 1.0, 4.0, 8.0):
            expected = _rk4_release(initial, elapsed)
            for omega in (0.25, 2.0, 10.0):
                motion = _motion(initial, omega0=omega)
                actual = _eval_angle(motion, elapsed / omega)
                assert abs(actual - expected) < 1e-10, "bridge exact motion must match independent RK4"


def test_small_amplitude_exact_motion_uses_release_formula():
    initial = math.radians(5)
    actual = _eval_angle(_motion(initial), 1.0)
    assert abs(actual - _rk4_release(initial, 1.0)) < 1e-11, \
        "bridge release formula must include the dn factor"


def test_near_separatrix_motion_validates_and_keeps_quarter_phase():
    for initial in (math.radians(179.9), math.radians(179.99999), math.nextafter(math.pi, 0.0)):
        quarter = _quarter_period(initial)
        for sign in (-1.0, 1.0):
            motion = _motion(sign * initial)
            ok, reason = E.validate_motion(motion)
            assert ok, "bridge near-separatrix motion must validate: " + reason
            for multiple, expected in ((0, sign * initial), (1, 0.0), (2, -sign * initial),
                                       (3, 0.0), (4, sign * initial)):
                actual = _eval_angle(motion, multiple * quarter)
                assert abs(actual - expected) < 1e-11, "bridge complementary modulus must retain quarter phase"


def test_near_separatrix_motion_matches_frozen_80_digit_reference():
    # Binary64 inputs promoted exactly to mpmath (80 dps), then evaluated as
    # 2*asin(sin(theta0/2)*sn(ellipk(m)-t,m)), m=sin(theta0/2)**2.
    cases = (
        (math.nextafter(math.pi, 0.0), 18.59320050603265, 3.14159261992812691711730431078527),
        (math.pi - 1e-12, 14.85512565734031, 3.14159123922678095171381163396301),
        (math.nextafter(math.pi, 0.0), 35.25, 2.56865494480586864516584127039092),
    )
    for initial, elapsed, expected in cases:
        for sign in (-1.0, 1.0):
            actual = _eval_angle(_motion(sign * initial), elapsed)
            assert abs(actual - sign * expected) < 1e-12, \
                "bridge stable descent must match high-precision reference"


def test_exact_motion_still_rejects_damping():
    for damping in (-0.1, 0.1, float("nan"), float("inf")):
        motion = _motion(zeta=damping)
        assert not E.validate_motion(motion)[0]
        with pytest.raises(E.AnimError):
            _eval_angle(motion, 0.5)


def test_exact_motion_input_errors_remain_anim_errors():
    for index in range(3):
        for value in (float("nan"), float("inf"), -float("inf"), True):
            arguments = [0.5, 1.0, 0.25]
            arguments[index] = value
            with pytest.raises(E.AnimError):
                E.pendulum_exact_angle(*arguments)
    for initial in (-2 * math.pi, -math.pi, math.pi, 2 * math.pi):
        motion = _motion(initial)
        assert not E.validate_motion(motion)[0]
        with pytest.raises(E.AnimError):
            _eval_angle(motion, 0.5)
    for initial in (float("nan"), float("inf"), -float("inf")):
        motion = _motion(theta0_deg=initial)
        assert not E.validate_motion(motion)[0]
        with pytest.raises(E.AnimError):
            _eval_angle(motion, 0.5)
    with pytest.raises(E.AnimError):
        E.pendulum_exact_angle(0.5, 1e308, 1e308)


def test_exact_motion_preserves_input_offset_and_length_path():
    motion = _motion(math.radians(-60), length_m=9.80665, offset_deg=5)
    before = copy.deepcopy(motion)
    expected = _rk4_release(math.radians(-60), 1.0) + math.radians(5)
    assert abs(_eval_angle(motion, 1.0) - expected) < 1e-11
    assert motion == before
    assert E.validate_motion(motion)[0]
    assert motion == before
    spec = _spec()
    spec_before = copy.deepcopy(spec)
    E.eval_at(spec, 120, motions=[motion])
    assert spec == spec_before and motion == before


def test_bridge_exact_motion_imports_without_optional_dependencies():
    code = """
import importlib.abc
import math
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'lingshu', 'bpy', 'numpy'}:
            raise RuntimeError('forbidden bridge dependency: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
sys.path.insert(0, sys.argv[1])
from wm_bridge.factory.anim import eval as E
motion = {'link':'bob', 'kind':'pendulum', 'axis':'y', 'exact':True,
          'theta0_deg':120.0, 'omega0':2.0, 'zeta':0.0}
spec = {'schema':'wmfactory.spec/1', 'name':'stdlib', 'scene':{'fps':24},
        'links':[{'name':'bob', 'anchor':[0,0,0], 'parts':[]}], 'joints':[]}
assert E.validate_motion(motion)[0]
assert all(math.isfinite(v) for v in E.eval_at(spec, 12, [motion])['bob']['quat'])
print('standalone bridge exact motion PASS')
"""
    result = subprocess.run([sys.executable, "-B", "-S", "-X", "utf8", "-c", code, str(BRIDGE)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, "bridge exact motion must remain stdlib-only: " + result.stderr
    assert "standalone bridge exact motion PASS" in result.stdout
