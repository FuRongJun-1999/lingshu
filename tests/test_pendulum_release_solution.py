"""Release-from-rest pendulum truth, boundary and invalid-input regression guards."""
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lingshu.world import physics as P


def _rk4_release(theta0, elapsed, dt=1.0 / 4096):
    """Integrate theta''=-sin(theta), independent of the elliptic implementation."""
    theta, velocity = theta0, 0.0
    steps = max(1, math.ceil(abs(elapsed) / dt))
    step = elapsed / steps
    for _ in range(steps):
        a1, b1 = velocity, -math.sin(theta)
        a2 = velocity + step * b1 / 2
        b2 = -math.sin(theta + step * a1 / 2)
        a3 = velocity + step * b2 / 2
        b3 = -math.sin(theta + step * a2 / 2)
        a4 = velocity + step * b3
        b4 = -math.sin(theta + step * a3)
        theta += step * (a1 + 2 * a2 + 2 * a3 + a4) / 6
        velocity += step * (b1 + 2 * b2 + 2 * b3 + b4) / 6
    return theta


def _quarter_period(theta0):
    a, b = 1.0, math.cos(theta0 / 2)
    for _ in range(32):
        a, b = (a + b) / 2, math.sqrt(a * b)
    return math.pi / (2 * a)


def test_release_solution_matches_independent_rk4():
    for degrees in (-179.9, -120.0, -60.0, -5.0, 0.0, 5.0, 30.0, 60.0, 120.0, 170.0, 179.9):
        theta0 = math.radians(degrees)
        for elapsed in (0.25, 1.0, 4.0, 8.0):
            expected = _rk4_release(theta0, elapsed)
            for omega in (0.25, 2.0, 10.0):
                actual = P.pendulum_exact_angle(theta0, omega, elapsed / omega)
                assert abs(actual - expected) < 1e-10, "release solution must match independent RK4"


def test_explicit_large_domain_uses_the_correct_release_formula():
    theta0 = math.radians(120)
    actual = P.pendulum_exact_angle(theta0, 1.0, 1.0, tol_deg=180.0)
    expected = _rk4_release(theta0, 1.0)
    assert abs(actual - expected) < 1e-11, "release formula must include the dn factor"


def test_release_initial_symmetry_and_near_separatrix_quarter_periods():
    for theta0 in (math.radians(179.9), math.radians(179.99999), math.nextafter(math.pi, 0.0)):
        quarter = _quarter_period(theta0)
        for sign in (-1.0, 1.0):
            initial = sign * theta0
            assert P.pendulum_exact_angle(initial, 1.0, 0.0) == initial
            for multiple, expected in ((1, 0.0), (2, -initial), (3, 0.0), (4, initial)):
                actual = P.pendulum_exact_angle(initial, 1.0, multiple * quarter)
                assert abs(actual - expected) < 1e-11, "complementary modulus must retain near-separatrix precision"
            for elapsed in (0.25, 1.0):
                assert P.pendulum_exact_angle(initial, 1.0, elapsed) == P.pendulum_exact_angle(initial, 1.0, -elapsed)


def test_near_separatrix_matches_frozen_80_digit_reference():
    # mpmath at 80 dps: promote each binary64 input exactly, then evaluate
    # 2*asin(sin(theta0/2)*sn(ellipk(m)-t,m)), m=sin(theta0/2)**2.
    # Fixed constants keep this guard stdlib-only; these non-quarter phases
    # expose the asin descent's loss of precision when c/a rounds to one.
    cases = (
        (math.nextafter(math.pi, 0.0), 18.59320050603265, 3.14159261992812691711730431078527),
        (math.pi - 1e-12, 14.85512565734031, 3.14159123922678095171381163396301),
        (math.nextafter(math.pi, 0.0), 35.25, 2.56865494480586864516584127039092),
    )
    for initial, elapsed, expected in cases:
        for sign in (-1.0, 1.0):
            actual = P.pendulum_exact_angle(sign * initial, 1.0, elapsed)
            assert abs(actual - sign * expected) < 1e-12, "stable descent must match high-precision reference"


def test_truth_period_retains_near_separatrix_complementary_modulus():
    for degrees in (179.99999, math.nextafter(180.0, 0.0)):
        for sign in (-1.0, 1.0):
            truth = P.pendulum_truth({"theta0_deg": sign * degrees, "omega0": 2.0})
            quarter = _quarter_period(truth["theta0_rad"])
            assert abs(truth["period_exact_elliptic"] - 2 * quarter) < 1e-12, "truth period must preserve complementary modulus"


def test_nonfinite_and_out_of_domain_arguments_fail_explicitly():
    for index in range(4):
        for value in (float("nan"), float("inf"), -float("inf"), True):
            args = [0.5, 1.0, 0.25, 180.0]
            args[index] = value
            with pytest.raises(P.PhysicsError):
                P.pendulum_exact_angle(*args)
    for initial in (-2 * math.pi, -math.pi, math.pi, 2 * math.pi):
        with pytest.raises(P.PhysicsError):
            P.pendulum_exact_angle(initial, 1.0, 1.0)
    with pytest.raises(P.PhysicsError):
        P.pendulum_exact_angle(0.5, 1.0, 1.0, tol_deg=720.0)
    with pytest.raises(P.PhysicsError):
        P.pendulum_exact_angle(0.5, 1e308, 1e308)
    with pytest.raises(P.PhysicsError):
        P.pendulum_exact_angle(math.radians(30), 1.0, 1.0, tol_deg=5.0)
