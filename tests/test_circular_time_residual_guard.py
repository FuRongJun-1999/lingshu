"""Circular/helix fits must preserve the time origin and score all three axes.

The trajectories and centered linear-regression oracle are independent of the
production fitters. All three up axes are exercised, including the up=2 control
where the old #786 coordinate-order defect could not occur.
"""
import math

import pytest

from lingshu.world import physics as P


def _state(tau, up, omega=1.5, drift=2.0, acceleration=0.0):
    idx = [axis for axis in range(3) if axis != up]
    pos, vel = [0.0] * 3, [0.0] * 3
    theta = 0.4 + omega * tau
    pos[idx[0]] = 2.0 + 2.0 * math.cos(theta)
    pos[idx[1]] = -1.0 + 2.0 * math.sin(theta)
    pos[up] = 3.0 + drift * tau + 0.5 * acceleration * tau**2
    vel[idx[0]] = -2.0 * omega * math.sin(theta)
    vel[idx[1]] = 2.0 * omega * math.cos(theta)
    vel[up] = drift + acceleration * tau
    return tuple(pos), tuple(vel)


def _samples(up, epoch=0.0, **kw):
    return [(epoch + i * 0.125, _state(i * 0.125, up, **kw)[0])
            for i in range(25)]


def test_shifted_helix_is_anchored_at_first_sample():
    for up in range(3):
        for epoch in (-10.0, 10.0, 1024.0):
            model = P.fit_model("circular", _samples(up, epoch), up=up)
            assert model["t_ref"] == epoch
            assert model["center"][up] == pytest.approx(3.0, abs=1e-10), "helix center must be at t_ref"
            assert model["v_up"] == pytest.approx(2.0, abs=1e-10)
            for tau in (-0.125, 0.0, 0.375, 2.0, 3.5):
                pos, vel = P.predict_at(model, epoch + tau)
                expected_pos, expected_vel = _state(tau, up)
                assert pos == pytest.approx(expected_pos, abs=1e-9)
                assert vel == pytest.approx(expected_vel, abs=1e-9)


def test_epoch_angular_and_vertical_fits_remain_nonsingular():
    for up in range(3):
        for epoch in (-1700000000.0, 1700000000.0):
            try:
                model = P.fit_circular(_samples(up, epoch), up=up)
            except P.PhysicsError as exc:
                pytest.fail("epoch helix must remain nonsingular: " + str(exc))
            for tau in (0.0, 0.625, 3.5):
                pos, vel = P.predict_at(model, epoch + tau)
                expected_pos, expected_vel = _state(tau, up)
                assert pos == pytest.approx(expected_pos, abs=1e-9), "epoch helix position drifted"
                assert vel == pytest.approx(expected_vel, abs=1e-9), "epoch helix velocity drifted"


def test_irregular_time_translation_preserves_parameters():
    times = (0.0, 0.0625, 0.25, 0.625, 1.25, 2.0, 2.875, 3.125)
    for up in range(3):
        for omega in (-1.25, 1.5):
            samples = [(t, _state(t, up, omega=omega, drift=-0.75)[0]) for t in times]
            zero = P.fit_circular(samples, up=up)
            shifted = P.fit_circular([(t + 1700000000.0, pos) for t, pos in samples], up=up)
            for field in ("center", "radius", "omega", "theta0", "v_up", "residual_rms"):
                assert shifted[field] == zero[field], "time translation changed local model parameters"


def test_vertical_fit_residual_matches_centered_regression_oracle():
    for up in range(3):
        for epoch in (0.0, 10.0):
            for acceleration in (-20.0, 2.0, 20.0):
                samples = _samples(up, epoch, acceleration=acceleration)
                ts = [t - epoch for t, _ in samples]
                ys = [pos[up] for _, pos in samples]
                t_mean, y_mean = sum(ts) / len(ts), sum(ys) / len(ys)
                slope = sum((t - t_mean) * (y - y_mean) for t, y in zip(ts, ys)) / sum(
                    (t - t_mean)**2 for t in ts)
                intercept = y_mean - slope * t_mean
                expected_rms = math.sqrt(sum((intercept + slope * t - y)**2
                                            for t, y in zip(ts, ys)) / (3 * len(ts)))
                model = P.fit_circular(samples, up=up)
                assert model["residual_rms"] == pytest.approx(expected_rms, rel=1e-10), "vertical residual must contribute to 3D coordinate RMS"
                assert model["center"][up] == pytest.approx(intercept, abs=1e-9)
                assert model["v_up"] == pytest.approx(slope, abs=1e-9)


def test_planar_noise_uses_same_three_coordinate_rms():
    for up in range(3):
        samples = _samples(up)
        idx = [axis for axis in range(3) if axis != up]
        noisy = []
        for i, (t, pos) in enumerate(samples):
            q = list(pos)
            q[idx[0]] += 0.02 * math.sin(i * 1.7)
            q[idx[1]] += 0.03 * math.cos(i * 0.9)
            noisy.append((t, tuple(q)))
        model = P.fit_circular(noisy, up=up)
        # Recompute the metric from public predictions, without the fitter's
        # internal accumulator, to cover both the plane and the vertical axis.
        rms = math.sqrt(sum(math.dist(P.predict_at(model, t)[0], pos)**2
                            for t, pos in noisy) / (3 * len(noisy)))
        assert rms > 1e-3
        assert model["residual_rms"] == pytest.approx(rms, rel=1e-12), "all fits must use three-coordinate normalization"


def test_classification_scores_vertical_mismatch_before_selecting_circular():
    for up in range(3):
        bad = P.classify(_samples(up, acceleration=20.0), up=up)
        assert bad["kind"] == "accelerated", "vertical acceleration must not masquerade as a perfect helix"
        assert bad["residuals"]["circular"] > bad["residuals"]["accelerated"]
        good = P.classify(_samples(up, 1700000000.0), up=up)
        assert good["kind"] == "circular", "valid epoch helix must remain selectable"
        predicted, velocity = P.predict_at(good["model"], 1700000003.5)
        expected_pos, expected_vel = _state(3.5, up)
        assert predicted == pytest.approx(expected_pos, abs=1e-9)
        assert velocity == pytest.approx(expected_vel, abs=1e-9)
