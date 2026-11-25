# -*- coding: utf-8 -*-
"""Regression and mutation-verified guard for local-time lateral drift in fit_drag2().

Defect (before fix):
  fit_drag2() fitted the vertical quadratic-drag axis in relative time
  `tau = t - t_ref`, while fitting the two lateral drift axes with
  `polyfit(ts, xs, 1)` in absolute time `ts`. That stored the `t = 0` intercept
  in `p0[i]` even though the returned model recorded `t_ref = ts[0]`. Subsequent
  `predict_at(model, t)` evaluated `p0[i] + drift[i] * (t - t_ref)`, shifting
  every horizontal prediction by `-drift[i] * t_ref`, inflating `residual_rms`,
  causing `classify(..., env=medium)` to misclassify non-zero-epoch `drag2`
  trajectories as `accelerated`/`uniform`, and hitting singular matrices at
  Unix-epoch timestamps.

Fix:
  Compute `t0 = ts[0]` and `tau = [t - t0 for t in ts]` at the top of
  `fit_drag2()` (shared by both fixed-beta and scanned-beta paths) and fit
  lateral drift via `polyfit(tau, xs, 1)`.

Run (pytest):  python -X utf8 -m pytest tests/test_physics_fit_drag2_local_time.py -q
Run (stdlib):  python -X utf8 tests/test_physics_fit_drag2_local_time.py
Self-verify:   python -X utf8 tests/test_physics_fit_drag2_local_time.py --mutate
"""
from __future__ import annotations

import ast
import math
import re
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from lingshu.world import physics as P  # noqa: E402


_CHECKS = []
FAILS = []


def _check(name, cond, detail=""):
    if name not in _CHECKS:
        _CHECKS.append(name)
    if cond:
        print("  PASS " + name)
    else:
        print("  FAIL " + name + "  " + str(detail)[:260])
        FAILS.append(name)


def _log_cosh_oracle(u):
    a = abs(float(u))
    return a + math.log1p(math.exp(-2.0 * a)) - math.log(2.0)


def _drag2_state_oracle(tau, p0, drift, beta, up=1, gravity=P.DEFAULT_G, v_down0=0.0):
    """Independent analytic oracle for 1D vertical drag2 + lateral uniform drift."""
    vt = math.sqrt(gravity / beta)
    u0 = math.atanh(v_down0 / vt)
    u = u0 + gravity * tau / vt
    s_down = (vt * vt / gravity) * (_log_cosh_oracle(u) - _log_cosh_oracle(u0))
    v_down = vt * math.tanh(u)
    pos = [0.0, 0.0, 0.0]
    vel = [0.0, 0.0, 0.0]
    for axis in range(3):
        if axis == up:
            pos[axis] = p0[axis] - s_down
            vel[axis] = -v_down
        else:
            pos[axis] = p0[axis] + drift[axis] * tau
            vel[axis] = drift[axis]
    return tuple(pos), tuple(vel)


def _make_samples(t0, p0, drift, beta=0.25, up=1, gravity=P.DEFAULT_G,
                  v_down0=0.0, taus=None):
    if taus is None:
        taus = [k * 0.125 for k in range(33)]
    out = []
    for tau in taus:
        pos, _ = _drag2_state_oracle(
            tau, p0, drift, beta, up=up, gravity=gravity, v_down0=v_down0)
        out.append((t0 + tau, pos))
    return out


def _vec_close(actual, expected, tol=1e-9):
    return len(actual) == len(expected) and all(
        abs(float(a) - float(b)) <= tol for a, b in zip(actual, expected)
    )


def _lateral_for_up(up, h0=3.0, h1=-2.0):
    vals = [0.0, 0.0, 0.0]
    lat = [ax for ax in range(3) if ax != up]
    vals[lat[0]] = h0
    vals[lat[1]] = h1
    return tuple(vals)


def _check_a01_nonzero_tref_fixed_beta():
    """A01: Positive and negative non-zero t_ref with fixed beta & mass/drag_c."""
    p0 = (5.0, 100.0, -4.0)
    drift = (3.0, 0.0, -2.0)
    beta = 0.25
    for v_down0 in (0.0, 1.5, -1.25):
        for t0 in (0.0, 10.0, -15.0, 1024.0):
            samples = _make_samples(t0, p0, drift, beta=beta, up=1, v_down0=v_down0)
            for kwargs in ({"beta": beta}, {"mass": 2.0, "drag_c": 0.5}):
                try:
                    model = P.fit_drag2(samples, up=1, **kwargs)
                except (P.PhysicsError, UnboundLocalError) as exc:
                    return False, "fit_drag2 failed at t0=%r kwargs=%r: %s" % (t0, kwargs, exc)
                if model["t_ref"] != t0:
                    return False, "t_ref mismatch: %r != %r" % (model["t_ref"], t0)
                if not _vec_close(model["p0"], p0, tol=1e-9):
                    return False, "t0=%r p0=%r expected=%r" % (t0, model["p0"], p0)
                if not _vec_close(model["drift"], drift, tol=1e-9):
                    return False, "t0=%r drift=%r expected=%r" % (t0, model["drift"], drift)
                if model["residual_rms"] > 1e-10:
                    return False, "t0=%r rms=%r" % (t0, model["residual_rms"])
                for tau in (-0.25, 0.0, 0.5, 2.5, 5.0):
                    pred_p, pred_v = P.predict_at(model, t0 + tau)
                    exp_p, exp_v = _drag2_state_oracle(
                        tau, p0, drift, beta, up=1, v_down0=v_down0)
                    if not _vec_close(pred_p, exp_p, tol=1e-9):
                        return False, "t0=%r tau=%r pred_p=%r exp_p=%r" % (t0, tau, pred_p, exp_p)
                    if not _vec_close(pred_v, exp_v, tol=1e-9):
                        return False, "t0=%r tau=%r pred_v=%r exp_v=%r" % (t0, tau, pred_v, exp_v)
    return True, ""


def _check_a02_all_gravity_axes():
    """A02: All three up/gravity axes (0, 1, 2) with non-zero lateral drift & t_ref."""
    p0 = (5.0, -7.5, 12.0)
    beta = 0.25
    for up in (0, 1, 2):
        drift = _lateral_for_up(up, h0=3.5, h1=-2.25)
        for t0 in (-15.0, 10.0):
            samples = _make_samples(t0, p0, drift, beta=beta, up=up, v_down0=0.75)
            for model in (
                P.fit_drag2(samples, up=up, beta=beta),
                P.fit_model("drag2", samples, up=up, beta=beta),
            ):
                if model["up"] != up or model["t_ref"] != t0:
                    return False, "up=%d t0=%r header=%r" % (up, t0, (model["up"], model["t_ref"]))
                if not _vec_close(model["p0"], p0, tol=1e-9):
                    return False, "up=%d t0=%r p0=%r expected=%r" % (up, t0, model["p0"], p0)
                if not _vec_close(model["drift"], drift, tol=1e-9):
                    return False, "up=%d t0=%r drift=%r expected=%r" % (up, t0, model["drift"], drift)
                if model["residual_rms"] > 1e-10:
                    return False, "up=%d t0=%r rms=%r" % (up, t0, model["residual_rms"])
                for tau in (0.0, 1.0, 4.0):
                    pred_p, pred_v = P.predict_at(model, t0 + tau)
                    exp_p, exp_v = _drag2_state_oracle(
                        tau, p0, drift, beta, up=up, v_down0=0.75)
                    if not _vec_close(pred_p, exp_p, tol=1e-9) or not _vec_close(pred_v, exp_v, tol=1e-9):
                        return False, "up=%d t0=%r tau=%r mismatch" % (up, t0, tau)
    return True, ""


def _check_a03_beta_scan_path_nonzero_tref():
    """A03: Unknown-beta scan path (beta=None) anchors lateral drift in tau for all axes."""
    p0 = (5.0, 100.0, -4.0)
    beta = 0.25
    for up in (0, 1, 2):
        drift = _lateral_for_up(up, h0=3.0, h1=-2.0)
        for t0 in (-15.0, 10.0):
            samples = _make_samples(t0, p0, drift, beta=beta, up=up)
            try:
                model = P.fit_drag2(samples, up=up)
            except (P.PhysicsError, UnboundLocalError) as exc:
                return False, "beta=None failed up=%d t0=%r: %s" % (up, t0, exc)
            if abs(model["beta"] - beta) > 1e-4:
                return False, "up=%d t0=%r beta=%r" % (up, t0, model["beta"])
            if not _vec_close(model["p0"], p0, tol=1e-4):
                return False, "up=%d t0=%r p0=%r expected=%r" % (up, t0, model["p0"], p0)
            if not _vec_close(model["drift"], drift, tol=1e-9):
                return False, "up=%d t0=%r drift=%r expected=%r" % (up, t0, model["drift"], drift)
            if model["residual_rms"] > 1e-5:
                return False, "up=%d t0=%r rms=%r" % (up, t0, model["residual_rms"])
            pred_p, _ = P.predict_at(model, t0)
            if not _vec_close(pred_p, p0, tol=1e-4):
                return False, "up=%d t0=%r pred(t0)=%r expected=%r" % (up, t0, pred_p, p0)
    return True, ""


def _check_a04_time_translation_invariance_and_epoch():
    """A04: Strict time-translation invariance across irregular times and Unix epochs."""
    p0 = (5.0, 100.0, -4.0)
    beta = 0.25
    taus = (0.0, 0.125, 0.25, 0.5, 1.0, 1.75, 2.5, 3.25, 4.0)
    fields = ("beta", "terminal_v", "p0", "v0", "drift", "u0", "residual_rms")
    for up in (0, 1, 2):
        drift = _lateral_for_up(up, h0=3.0, h1=-2.0)
        base = _make_samples(0.0, p0, drift, beta=beta, up=up, v_down0=0.5, taus=taus)
        for kwargs in ({"beta": beta}, {}):
            try:
                ref = P.fit_drag2(base, up=up, **kwargs)
            except (P.PhysicsError, UnboundLocalError) as exc:
                return False, "base fit failed up=%d kwargs=%r: %s" % (up, kwargs, exc)
            for shift in (-1700000000.0, -15.0, 10.0, 1700000000.0):
                shifted = [(t + shift, pos) for t, pos in base]
                try:
                    got = P.fit_drag2(shifted, up=up, **kwargs)
                except (P.PhysicsError, UnboundLocalError) as exc:
                    return False, "shifted fit failed up=%d shift=%r kwargs=%r: %s" % (
                        up, shift, kwargs, exc)
                if got["t_ref"] != shift:
                    return False, "t_ref=%r != %r" % (got["t_ref"], shift)
                for field in fields:
                    if got[field] != ref[field]:
                        return False, "up=%d shift=%r kwargs=%r field=%s: %r != %r" % (
                            up, shift, kwargs, field, got[field], ref[field])
                pred_ref, vel_ref = P.predict_at(ref, 2.25)
                pred_got, vel_got = P.predict_at(got, shift + 2.25)
                if pred_got != pred_ref or vel_got != vel_ref:
                    return False, "prediction shifted at up=%d shift=%r" % (up, shift)
    return True, ""


def _check_a05_classify_selects_drag2_at_shifted_tref():
    """A05: classify(..., env=medium) selects 'drag2' for shifted trajectories across all axes."""
    p0 = (5.0, 100.0, -4.0)
    env = P.make_environment(medium={"mass": 2.0, "drag_c": 0.5})
    taus = [k * 0.1 for k in range(31)]
    for up in (0, 1, 2):
        drift = _lateral_for_up(up, h0=3.0, h1=-2.0)
        for t0 in (0.0, 10.0, -15.0, 1700000000.0):
            # Use binary-exact steps at Unix epoch to avoid float64 cancellation in t0 + tau.
            cur_taus = [k * 0.125 for k in range(33)] if abs(t0) > 1e6 else taus
            samples = _make_samples(t0, p0, drift, beta=0.25, up=up, taus=cur_taus)
            try:
                res = P.classify(samples, up=up, env=env)
            except (P.PhysicsError, UnboundLocalError) as exc:
                return False, "classify failed up=%d t0=%r: %s" % (up, t0, exc)
            if res["kind"] != "drag2":
                return False, "up=%d t0=%r got kind=%r drag2_rms=%r acc_rms=%r" % (
                    up, t0, res["kind"], res["residuals"].get("drag2"),
                    res["residuals"].get("accelerated"))
            if res["residuals"]["drag2"] > 1e-10:
                return False, "up=%d t0=%r drag2_rms=%r" % (up, t0, res["residuals"]["drag2"])
            pred_p, pred_v = P.predict_at(res["model"], t0 + 1.5)
            exp_p, exp_v = _drag2_state_oracle(1.5, p0, drift, 0.25, up=up)
            if not _vec_close(pred_p, exp_p, tol=1e-9) or not _vec_close(pred_v, exp_v, tol=1e-9):
                return False, "up=%d t0=%r classify model prediction mismatch" % (up, t0)
    return True, ""


def _check_a06_noisy_lateral_centered_ols_oracle():
    """A06: Noisy lateral drift matches independent tau-centered OLS and 3D coordinate RMS."""
    p0 = (5.0, 100.0, -4.0)
    drift = (3.0, 0.0, -2.0)
    beta = 0.25
    taus = [k * 0.125 for k in range(25)]
    for t0 in (-15.0, 10.0):
        base = _make_samples(t0, p0, drift, beta=beta, up=1, taus=taus)
        noisy = []
        for idx, (t, pos) in enumerate(base):
            q = [pos[0] + 0.04 * math.sin(idx * 1.3),
                 pos[1],
                 pos[2] + 0.03 * math.cos(idx * 0.9)]
            noisy.append((t, tuple(q)))
        model = P.fit_drag2(noisy, up=1, beta=beta)
        t_mean = sum(taus) / float(len(taus))
        denom = sum((x - t_mean) ** 2 for x in taus)
        for ax in (0, 2):
            xs = [pos[ax] for _, pos in noisy]
            x_mean = sum(xs) / float(len(xs))
            slope = sum((x - t_mean) * (y - x_mean) for x, y in zip(taus, xs)) / denom
            intercept = x_mean - slope * t_mean
            if abs(model["p0"][ax] - intercept) > 1e-10:
                return False, "t0=%r ax=%d p0=%r expected=%r" % (t0, ax, model["p0"][ax], intercept)
            if abs(model["drift"][ax] - slope) > 1e-10:
                return False, "t0=%r ax=%d drift=%r expected=%r" % (t0, ax, model["drift"][ax], slope)
        rms = math.sqrt(sum(
            math.dist(P.predict_at(model, t)[0], pos) ** 2 for t, pos in noisy
        ) / (3.0 * len(noisy)))
        if rms <= 1e-3 or abs(model["residual_rms"] - rms) > 1e-12:
            return False, "t0=%r rms=%r expected=%r" % (t0, model["residual_rms"], rms)
    return True, ""


def _check_b01_control_zero_tref_zero_drift():
    """B01 (control): t0=0.0 with zero lateral drift remains exact and classifies as drag2."""
    p0 = (0.0, 100.0, 0.0)
    drift = (0.0, 0.0, 0.0)
    samples = _make_samples(0.0, p0, drift, beta=0.25, up=1)
    model = P.fit_drag2(samples, up=1, beta=0.25)
    env = P.make_environment(medium={"mass": 2.0, "drag_c": 0.5})
    cls = P.classify(samples, up=1, env=env)
    ok = (
        model["t_ref"] == 0.0
        and _vec_close(model["p0"], p0, tol=1e-10)
        and _vec_close(model["drift"], drift, tol=1e-10)
        and model["residual_rms"] < 1e-11
        and cls["kind"] == "drag2"
    )
    return ok, (model["p0"], model["drift"], model["residual_rms"], cls["kind"])


def _check_e01_source_anchor_uses_tau():
    """E01: Source AST inside fit_drag2() passes local time `tau` (not `ts`) to polyfit."""
    src = (_REPO / "lingshu" / "world" / "physics.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fit_node = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "fit_drag2":
            fit_node = node
            break
    if fit_node is None:
        return False, "fit_drag2 not found in lingshu/world/physics.py"
    first_args = []
    for sub in ast.walk(fit_node):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id == "polyfit":
            if sub.args and isinstance(sub.args[0], ast.Name):
                first_args.append(sub.args[0].id)
            else:
                first_args.append(None)
    return first_args == ["tau"], "polyfit time args inside fit_drag2: %r" % (first_args,)


_CASES = [
    ("A01", "non-zero t_ref (pos/neg) with fixed beta and mass/drag_c", _check_a01_nonzero_tref_fixed_beta),
    ("A02", "all gravity axes up in (0, 1, 2) with non-zero lateral drift", _check_a02_all_gravity_axes),
    ("A03", "beta=None scan path anchors lateral drift in tau", _check_a03_beta_scan_path_nonzero_tref),
    ("A04", "time-translation invariance across irregular times and Unix epochs", _check_a04_time_translation_invariance_and_epoch),
    ("A05", "classify(..., env=medium) selects drag2 at shifted t_ref", _check_a05_classify_selects_drag2_at_shifted_tref),
    ("A06", "noisy lateral drift matches centered tau OLS and 3D RMS", _check_a06_noisy_lateral_centered_ols_oracle),
    ("B01", "control: t0=0.0 with zero lateral drift", _check_b01_control_zero_tref_zero_drift),
    ("E01", "AST anchor: fit_drag2 calls polyfit(tau, ...)", _check_e01_source_anchor_uses_tau),
]


def _assertions():
    for cid, label, fn in _CASES:
        ok, detail = fn()
        _check("%s %s" % (cid, label), ok, detail)


def _run_assertions():
    global FAILS
    FAILS = []
    del _CHECKS[:]
    _assertions()
    print("passed %d / failed %d" % (len(_CHECKS) - len(FAILS), len(FAILS)))
    return len(FAILS)


_ARMS = {
    "M1": (
        "lingshu/world/physics.py",
        [("        c = polyfit(tau, xs, 1)",
          "        c = polyfit(ts, xs, 1)")],
        {"A01", "A02", "A03", "A04", "A05", "A06", "E01"},
    ),
    "M2": (
        "lingshu/world/physics.py",
        [("    g = _positive_num(gravity, \"gravity\")\n"
          "    t0 = ts[0]\n"
          "    tau = [t - t0 for t in ts]",
          "    g = _positive_num(gravity, \"gravity\")"),
         ("        ys = [-float(s[1][up]) for s in samples]      # 下落坐标（正=向下）\n"
          "        vt = math.sqrt(g / beta)",
          "        ys = [-float(s[1][up]) for s in samples]      # 下落坐标（正=向下）\n"
          "        t0 = ts[0]\n"
          "        tau = [t - t0 for t in ts]\n"
          "        vt = math.sqrt(g / beta)")],
        {"A03", "A04"},
    ),
}


def _subprocess_fail_names():
    for cache in (_REPO / "lingshu" / "world").rglob("__pycache__"):
        for pyc in cache.glob("physics.*.pyc"):
            pyc.unlink(missing_ok=True)
    proc = subprocess.run(
        [sys.executable, "-X", "utf8", __file__, "--assert-only"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(_REPO), timeout=300,
    )
    names = set(re.findall(r"(?m)^\s*FAIL\s+([A-Z]\d+)", proc.stdout or ""))
    return names, (proc.stdout or "") + (proc.stderr or "")


def _mutate(argv):
    want = ""
    for arg in argv:
        if arg == "--mutate":
            want = want or "ALL"
        elif arg.startswith("--mutate="):
            want = arg.split("=", 1)[1]
    if not want:
        return 0
    ids = sorted(_ARMS) if want == "ALL" else [want]
    red = 0
    for mid in ids:
        if mid not in _ARMS:
            print("[ANCHOR-MISS] %s: unknown mutation arm" % mid)
            return 2
        rel, edits, expect = _ARMS[mid]
        target = _REPO / rel
        orig = target.read_bytes()
        try:
            text = orig.decode("utf-8")
            nl = "\r\n" if "\r\n" in text else "\n"
            norm = text.replace("\r\n", "\n")
            for old, new in edits:
                cnt = norm.count(old)
                if cnt != 1:
                    print("[ANCHOR-MISS] %s: anchor hit %d times (expected 1)" % (mid, cnt))
                    return 2
                norm = norm.replace(old, new)
            target.write_bytes(norm.replace("\n", nl).encode("utf-8"))
            actual, out = _subprocess_fail_names()
            if actual == expect:
                print("[MUTATE-OK]   %s: expected failures hit %s" % (mid, sorted(actual)))
            else:
                print("[MUTATE-FAIL] %s: expected %s, actual %s"
                      % (mid, sorted(expect), sorted(actual)))
                for line in out.splitlines():
                    if line.strip().startswith("FAIL"):
                        print("    " + line)
                red += 1
        finally:
            target.write_bytes(orig)
            for cache in (_REPO / "lingshu" / "world").rglob("__pycache__"):
                for pyc in cache.glob("physics.*.pyc"):
                    pyc.unlink(missing_ok=True)
    print("mutation summary: %d arms, %s"
          % (len(ids), "all matched" if red == 0 else "%d mismatched" % red))
    return 1 if red else 0


# pytest entry points (pure stdlib, no pytest import required)
def test_a01_nonzero_tref_fixed_beta():
    ok, detail = _check_a01_nonzero_tref_fixed_beta()
    assert ok, detail


def test_a02_all_gravity_axes():
    ok, detail = _check_a02_all_gravity_axes()
    assert ok, detail


def test_a03_beta_scan_path_nonzero_tref():
    ok, detail = _check_a03_beta_scan_path_nonzero_tref()
    assert ok, detail


def test_a04_time_translation_invariance_and_epoch():
    ok, detail = _check_a04_time_translation_invariance_and_epoch()
    assert ok, detail


def test_a05_classify_selects_drag2_at_shifted_tref():
    ok, detail = _check_a05_classify_selects_drag2_at_shifted_tref()
    assert ok, detail


def test_a06_noisy_lateral_centered_ols_oracle():
    ok, detail = _check_a06_noisy_lateral_centered_ols_oracle()
    assert ok, detail


def test_b01_control_zero_tref_zero_drift():
    ok, detail = _check_b01_control_zero_tref_zero_drift()
    assert ok, detail


def test_e01_source_anchor_uses_tau():
    ok, detail = _check_e01_source_anchor_uses_tau()
    assert ok, detail


if __name__ == "__main__":
    _f = _run_assertions()
    if "--assert-only" in sys.argv:
        sys.exit(1 if _f else 0)
    _m = _mutate(sys.argv[1:])
    sys.exit(1 if (_f or _m) else 0)
