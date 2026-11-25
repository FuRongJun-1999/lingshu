# -*- coding: utf-8 -*-
"""World/motion non-finite anchor, historical backfill, overflow & bool guards.

Covers four correctness defects in ``lingshu/world/motion.py``:
  1. M1 (``ObservationWindow.pair``): a non-finite first joint frame previously
     poisoned ``previous`` before ``_finite_pos`` ran, fabricating ``intervals=1``
     (and bogus directional scores when the other entity moved) on ``[NaN, finite]``.
  2. M2a (``ObservationWindow.pair`` historical backfill): scanning
     ``history[:start]`` previously accepted the first joint frame without
     checking ``_finite_pos``, allowing a non-finite historical frame to shadow
     earlier valid finite anchors or fabricate a bogus interval.
  3. M2b (``ObservationWindow._context_points`` historical backfill): non-finite
     historical frames previously consumed the ``need`` quota and hid earlier
     valid finite observations behind them (while still preserving
     ``nonfinite_observation`` diagnostics when fewer than two finite points exist).
  4. M3 (``ObservationWindow.pair`` subtraction overflow): extreme finite
     coordinates such as ``-1e308 -> +1e308`` (and 2D diagonals where even
     half-scaling overflows ``math.hypot``) previously overflowed subtraction to
     ``inf`` and appended ``NaN`` to ``scores`` (``inf / inf``).
  5. M4 (``_finite_pos``): ``bool`` coordinates (``True``/``False``) previously
     passed ``math.isfinite(float(v))`` as ``1.0``/``0.0``.

Run:
  pytest: python -X utf8 -m pytest tests/test_motion_nonfinite_backfill_and_pair_guard.py -q
  direct: python -X utf8 tests/test_motion_nonfinite_backfill_and_pair_guard.py
  mutate: python -X utf8 tests/test_motion_nonfinite_backfill_and_pair_guard.py --mutate=ALL
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

_REPO = Path(__file__).resolve().parents[1]
_SANDBOX = os.environ.get("LINGSHU_MUTATE_SANDBOX")
if _SANDBOX:
    if _SANDBOX not in sys.path:
        sys.path.insert(0, _SANDBOX)
elif str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from lingshu.world import motion as M  # noqa: E402
from lingshu.world.motion import ObservationWindow  # noqa: E402
from lingshu.world.world_model import UnifiedWorldModel  # noqa: E402


def _frame(tick, nested=False, **entities):
    if nested:
        return {"tick": tick, "entities": {k: {"pos": v} for k, v in entities.items()}}
    return {"tick": tick, "entities": dict(entities)}


# --------------------------------------------------------------------------- #
# Group A: Bug M1 -- pair() first-frame non-finite anchor
# --------------------------------------------------------------------------- #
def _check_a01_pair_first_frame_nonfinite_never_poisons_previous():
    """A01: [non-finite, finite] must yield (0.0, 0, 0) in both directions."""
    for bad in (float("nan"), float("inf"), float("-inf")):
        for nested in (False, True):
            # Entity 'a' non-finite on first frame while 'b' moves on second frame
            h_a = [
                _frame(0, nested, a=(bad, 1.5, 2.0), b=(10.0, 1.5, 2.0)),
                _frame(1, nested, a=(2.0, 1.5, 2.0), b=(12.0, 1.5, 2.0)),
            ]
            ow_a = ObservationWindow(h_a, 2, nested=nested)
            if ow_a.pair("a", "b") != (0.0, 0, 0) or ow_a.pair("b", "a") != (0.0, 0, 0):
                return False, "a-bad(%r,nested=%s): ab=%r ba=%r" % (
                    bad, nested, ow_a.pair("a", "b"), ow_a.pair("b", "a"))

            # Entity 'b' non-finite on first frame while 'a' moves toward 'b'
            h_b = [
                _frame(0, nested, a=(0.0, 1.5, 2.0), b=(bad, 1.5, 2.0)),
                _frame(1, nested, a=(5.0, 1.5, 2.0), b=(20.0, 1.5, 2.0)),
            ]
            ow_b = ObservationWindow(h_b, 2, nested=nested)
            if ow_b.pair("a", "b") != (0.0, 0, 0) or ow_b.pair("b", "a") != (0.0, 0, 0):
                return False, "b-bad(%r,nested=%s): ab=%r ba=%r" % (
                    bad, nested, ow_b.pair("a", "b"), ow_b.pair("b", "a"))
    return True, ""


def _check_a02_pair_skips_leading_nonfinite_and_measures_valid_suffix():
    """A02: [NaN, fin1, fin2] measures only fin1->fin2; [fin0, NaN, fin2] bridges fin0->fin2."""
    h_lead = [
        _frame(0, a=(float("nan"), 1.5, 2.0), b=(20.0, 1.5, 2.0)),
        _frame(1, a=(0.0, 1.5, 2.0), b=(20.0, 1.5, 2.0)),
        _frame(2, a=(5.0, 1.5, 2.0), b=(20.0, 1.5, 2.0)),
    ]
    ow_lead = ObservationWindow(h_lead, 3)
    if ow_lead.pair("a", "b") != (1.0, 1, 1) or ow_lead.pair("b", "a") != (0.0, 0, 1):
        return False, "leading NaN: ab=%r ba=%r" % (
            ow_lead.pair("a", "b"), ow_lead.pair("b", "a"))

    h_mid = [
        _frame(0, a=(0.0, 1.5, 2.0), b=(20.0, 1.5, 2.0)),
        _frame(1, a=(float("inf"), 1.5, 2.0), b=(20.0, 1.5, 2.0)),
        _frame(2, a=(5.0, 1.5, 2.0), b=(20.0, 1.5, 2.0)),
    ]
    ow_mid = ObservationWindow(h_mid, 3)
    if ow_mid.pair("a", "b") != (1.0, 1, 1):
        return False, "mid inf: ab=%r" % (ow_mid.pair("a", "b"),)
    return True, ""


def _check_a03_pair_window_nonfinite_backfill_and_early_exit():
    """A03: All-non-finite window does not scan old history; [fin | NaN, fin] backfills."""
    class GuardOldFrame(dict):
        def __getitem__(self, key):
            raise AssertionError("old history scanned for all-non-finite window")

    h_all_bad = [
        GuardOldFrame(tick=0, entities={"a": (0.0, 1.5, 2.0), "b": (20.0, 1.5, 2.0)}),
        _frame(1, a=(float("nan"), 1.5, 2.0), b=(20.0, 1.5, 2.0)),
        _frame(2, a=(float("inf"), 1.5, 2.0), b=(20.0, 1.5, 2.0)),
    ]
    ow_bad = ObservationWindow(h_all_bad, 2)
    if ow_bad.pair("a", "b") != (0.0, 0, 0) or ow_bad.pair("b", "a") != (0.0, 0, 0):
        return False, "all-bad window: ab=%r" % (ow_bad.pair("a", "b"),)

    # Window has [NaN, fin] with finite frame right before window -> backfill treats NaN like gap
    h_win_nan = [
        _frame(0, a=(0.0, 1.5, 2.0), b=(20.0, 1.5, 2.0)),
        _frame(1, a=(float("nan"), 1.5, 2.0), b=(20.0, 1.5, 2.0)),
        _frame(2, a=(10.0, 1.5, 2.0), b=(20.0, 1.5, 2.0)),
    ]
    ow_win_nan = ObservationWindow(h_win_nan, 2)
    if ow_win_nan.pair("a", "b") != (1.0, 1, 1):
        return False, "window-NaN backfill: ab=%r" % (ow_win_nan.pair("a", "b"),)
    return True, ""


# --------------------------------------------------------------------------- #
# Group B: Bug M2a -- pair() historical backfill skips non-finite frames
# --------------------------------------------------------------------------- #
def _check_b01_pair_backfill_skips_nonfinite_to_reach_real_anchor():
    """B01: [t0=fin, t1=non-finite | win: miss, fin] anchors on t0, not t1."""
    for bad in (float("nan"), float("inf"), float("-inf"), 10**400):
        for nested in (False, True):
            h = [
                _frame(0, nested, a=(0.0, 1.5, 2.0), b=(20.0, 1.5, 2.0)),
                _frame(1, nested, a=(bad, 1.5, 2.0), b=(20.0, 1.5, 2.0)),
                _frame(2, nested),
                _frame(3, nested, a=(10.0, 1.5, 2.0), b=(20.0, 1.5, 2.0)),
            ]
            ow = ObservationWindow(h, 2, nested=nested)
            if ow.pair("a", "b") != (1.0, 1, 1) or ow.pair("b", "a") != (0.0, 0, 1):
                return False, "bad=%r nested=%s: ab=%r ba=%r" % (
                    bad, nested, ow.pair("a", "b"), ow.pair("b", "a"))

    # Multiple non-finite historical frames between t0 and window
    h_multi = [
        _frame(0, a=(0.0, 1.5, 2.0), b=(20.0, 1.5, 2.0)),
        _frame(1, a=(0.0, 1.5, 2.0), b=(float("inf"), 1.5, 2.0)),
        _frame(2, a=(float("nan"), 1.5, 2.0), b=(20.0, 1.5, 2.0)),
        _frame(3),
        _frame(4, a=(10.0, 1.5, 2.0), b=(20.0, 1.5, 2.0)),
    ]
    ow_multi = ObservationWindow(h_multi, 2)
    if ow_multi.pair("a", "b") != (1.0, 1, 1):
        return False, "multi-bad backfill: ab=%r" % (ow_multi.pair("a", "b"),)
    return True, ""


def _check_b02_pair_backfill_only_nonfinite_before_window_gives_zero():
    """B02: [t0=NaN | win: miss, fin] must return (0.0, 0, 0), never intervals=1."""
    for nested in (False, True):
        h = [
            _frame(0, nested, a=(float("nan"), 1.5, 2.0), b=(10.0, 1.5, 2.0)),
            _frame(1, nested),
            _frame(2, nested, a=(5.0, 1.5, 2.0), b=(15.0, 1.5, 2.0)),
        ]
        ow = ObservationWindow(h, 2, nested=nested)
        if ow.pair("a", "b") != (0.0, 0, 0) or ow.pair("b", "a") != (0.0, 0, 0):
            return False, "nan-only-pre(nested=%s): ab=%r ba=%r" % (
                nested, ow.pair("a", "b"), ow.pair("b", "a"))
    return True, ""


# --------------------------------------------------------------------------- #
# Group C: Bug M2b -- _context_points() historical backfill skips non-finite
# --------------------------------------------------------------------------- #
def _check_c01_context_backfill_window1_skips_nonfinite_history():
    """C01: [0, 1, NaN | win(w=1): 3] skips NaN and pairs tick 1 -> tick 3."""
    for bad in (float("nan"), float("inf"), float("-inf")):
        for nested in (False, True):
            h = [
                _frame(0, nested, a=(0.0, 1.5, 2.0)),
                _frame(1, nested, a=(1.0, 1.5, 2.0)),
                _frame(2, nested, a=(bad, 1.5, 2.0)),
                _frame(3, nested, a=(3.0, 1.5, 2.0)),
            ]
            ow = ObservationWindow(h, 1, nested=nested)
            est = ow.motion("a")
            det = ow.motion_detail("a")
            if est != (1.0, 1.0, 1, (1.0, 0.0, 0.0)):
                return False, "bad=%r nested=%s est=%r det=%r" % (bad, nested, est, det)
            if not (det["adopted"] and det["evidence"] == "window+context"
                    and det["window_points"] == 1 and det["context_points"] == 1
                    and det["dts"] == [2]):
                return False, "bad=%r nested=%s det=%r" % (bad, nested, det)
    return True, ""


def _check_c02_context_only_occlusion_skips_nonfinite_history():
    """C02: [0, 1, inf, NaN | win(w=2): miss, miss] recovers finite pair (0 -> 1)."""
    h = [
        _frame(0, a=(0.0, 1.5, 2.0)),
        _frame(1, a=(0.8, 1.5, 2.0)),
        _frame(2, a=(float("inf"), 1.5, 2.0)),
        _frame(3, a=(float("nan"), 1.5, 2.0)),
        _frame(4),
        _frame(5),
    ]
    ow = ObservationWindow(h, 2)
    est = ow.motion("a")
    det = ow.motion_detail("a")
    if est != (1.0, 0.8, 0, (1.0, 0.0, 0.0)):
        return False, "occlusion est=%r det=%r" % (est, det)
    if not (det["adopted"] and det["evidence"] == "context_only"
            and det["window_points"] == 0 and det["context_points"] == 2):
        return False, "occlusion det=%r" % (det,)
    return True, ""


def _check_c03_nonfinite_reason_preserved_when_no_finite_pair_exists():
    """C03: Preserves nonfinite_observation diagnostics when < 2 finite points exist."""
    # Case 1: 1 finite + 1 NaN in pre-window history, empty window (matches #418 G4)
    h1 = [_frame(0, nested=True, e=(0.0, 0.0, 0.0)),
          _frame(1, nested=True),
          _frame(2, nested=True, e=(float("nan"), 0.0, 0.0))]
    h1 += [_frame(t, nested=True) for t in range(3, 10)]
    d1 = ObservationWindow(h1, 6, nested=True).motion_detail("e")
    if not (d1["estimate"] is None and d1["nonfinite_points"] == 1
            and d1["nonfinite_pairs_dropped"] == 1
            and d1["reason"] == "nonfinite_observation"):
        return False, "case1 det=%r" % (d1,)

    # Case 2: only non-finite frames before window, 1 finite frame in window
    h2 = [
        _frame(0, e=(float("nan"), 0.0, 0.0)),
        _frame(1, e=(float("inf"), 0.0, 0.0)),
        _frame(2, e=(3.0, 0.0, 0.0)),
    ]
    d2 = ObservationWindow(h2, 1).motion_detail("e")
    if not (d2["estimate"] is None and d2["nonfinite_points"] >= 1
            and d2["nonfinite_pairs_dropped"] >= 1
            and d2["reason"] == "nonfinite_observation"):
        return False, "case2 det=%r" % (d2,)
    return True, ""


# --------------------------------------------------------------------------- #
# Group D: Bug M3 -- pair() extreme finite subtraction overflow
# --------------------------------------------------------------------------- #
def _check_d01_pair_1d_subtraction_overflow_stays_exact_and_finite():
    """D01: -1e308 -> +1e308 produces exact finite +1.0 / -1.0, never NaN."""
    # Move toward stationary target ahead at +1.5e308
    h_toward = [
        _frame(0, a=(-1e308, 1.5, 2.0), b=(1.5e308, 1.5, 2.0)),
        _frame(1, a=(1e308, 1.5, 2.0), b=(1.5e308, 1.5, 2.0)),
    ]
    ow1 = ObservationWindow(h_toward, 2)
    ab1 = ow1.pair("a", "b")
    ba1 = ow1.pair("b", "a")
    if not (math.isfinite(ab1[0]) and ab1 == (1.0, 1, 1) and ba1 == (0.0, 0, 1)):
        return False, "toward: ab=%r ba=%r" % (ab1, ba1)

    # Move away from target behind at -1.5e308 (both movement AND target delta overflow!)
    h_away = [
        _frame(0, a=(-1e308, 1.5, 2.0), b=(-1.5e308, 1.5, 2.0)),
        _frame(1, a=(1e308, 1.5, 2.0), b=(-1.5e308, 1.5, 2.0)),
    ]
    ow2 = ObservationWindow(h_away, 2)
    ab2 = ow2.pair("a", "b")
    if not (math.isfinite(ab2[0]) and ab2 == (-1.0, 1, 1)):
        return False, "away: ab=%r" % (ab2,)
    return True, ""


def _check_d02_pair_2d_diagonal_hypot_overflow_and_large_ints():
    """D02: 2D diagonal (-1.5e308,-1.5e308)->(+1.5e308,+1.5e308) & +-10**308 ints."""
    # Even 0.5-scaling gives (1.5e308, 1.5e308) whose hypot=2.12e308 > DBL_MAX!
    h_diag = [
        _frame(0, a=(-1.5e308, 0.0, -1.5e308), b=(1.5e308, 0.0, -1.5e308)),
        _frame(1, a=(1.5e308, 0.0, 1.5e308), b=(1.5e308, 0.0, -1.5e308)),
    ]
    ow_diag = ObservationWindow(h_diag, 2)
    ab_diag = ow_diag.pair("a", "b")
    expect = round(-1.0 / math.sqrt(2.0), 3)
    if not (math.isfinite(ab_diag[0]) and ab_diag == (expect, 1, 1)):
        return False, "diag: got=%r expected=(%r, 1, 1)" % (ab_diag, expect)

    # Float-representable huge Python ints whose int difference overflows float()
    h_ints = [
        _frame(0, a=(-(10**308), 0, 0), b=(15 * (10**307), 0, 0)),
        _frame(1, a=(10**308, 0, 0), b=(15 * (10**307), 0, 0)),
    ]
    ow_ints = ObservationWindow(h_ints, 2)
    ab_ints = ow_ints.pair("a", "b")
    if not (math.isfinite(ab_ints[0]) and ab_ints == (1.0, 1, 1)):
        return False, "huge-ints: got=%r" % (ab_ints,)
    return True, ""


def _check_d03_world_model_extreme_pair_stays_finite_json():
    """D03: UnifiedWorldModel keeps finite relations & JSON on -1e308 -> +1e308."""
    m = UnifiedWorldModel(size=80)
    for t in range(6):
        m.perceive([
            {"eid": "a", "category": "a", "pos": [float(t), 1.5, 2.0]},
            {"eid": "b", "category": "b", "pos": [float(t) + 10.0, 1.5, 2.0]},
        ], tick=t)
    m.infer_patterns(window=6, force=True)
    before = {(e.source, e.relation, e.target) for e in m.edges}
    if not before:
        return False, "precondition: expected initial relation"

    # Append extreme finite seeking steps where raw subtraction overflows to inf
    pts = [
        (-1e308, 1.5e308),
        (1e308, 1.5e308),
        (-1e308, -1.5e308),
        (1e308, 1.5e308),
    ]
    for offset, (ax, bx) in enumerate(pts, start=6):
        m.perceive([
            {"eid": "a", "category": "a", "pos": [ax, 1.5, 2.0]},
            {"eid": "b", "category": "b", "pos": [bx, 1.5, 2.0]},
        ], tick=offset)
    pat = m.infer_patterns(window=4, force=True)
    json.dumps(pat, allow_nan=False)
    after = {(e.source, e.relation, e.target) for e in m.edges}
    if ("a", "seek", "b") not in after:
        return False, "extreme coords erased relation: before=%r after=%r" % (before, after)
    return True, ""


# --------------------------------------------------------------------------- #
# Group E: Bug M4 -- _finite_pos rejects bool coordinates on every axis
# --------------------------------------------------------------------------- #
def _check_e01_finite_pos_rejects_bool_and_invalid_accepts_numbers():
    """E01: _finite_pos rejects bool on every axis (plus str/NaN/inf/overflow)."""
    bad_positions = [
        (True, 0.0, 0.0),
        (0.0, False, 0.0),
        (0.0, 0.0, True),
        (True, False, True),
        [False, False, False],
        "123",
        b"123",
        (float("nan"), 0.0, 0.0),
        (0.0, float("inf"), 0.0),
        (0.0, 0.0, float("-inf")),
        (10**400, 0.0, 0.0),
        (1.0, 2.0),
        None,
    ]
    for pos in bad_positions:
        if M._finite_pos(pos) is not False:
            return False, "expected False for %r" % (pos,)

    good_positions = [
        (0, 0, 0),
        (1, -2, 3),
        (-1e308, 1.5, 1e308),
        [-0.0, 0.0, 3.14],
    ]
    for pos in good_positions:
        if M._finite_pos(pos) is not True:
            return False, "expected True for %r" % (pos,)
    return True, ""


def _check_e02_bool_coordinates_rejected_in_pair_and_motion():
    """E02: bool coordinates cannot form pair intervals, motion estimates, or backfill."""
    # pair() with leading bool frame
    h_pair = [
        _frame(0, a=(True, 0.0, 0.0), b=(10.0, 0.0, 0.0)),
        _frame(1, a=(2.0, 0.0, 0.0), b=(10.0, 0.0, 0.0)),
    ]
    ow_p = ObservationWindow(h_pair, 2)
    if ow_p.pair("a", "b") != (0.0, 0, 0):
        return False, "bool pair=%r" % (ow_p.pair("a", "b"),)

    # motion() with bool coordinate
    h_mot = [
        _frame(0, a=(0.0, 0.0, 0.0)),
        _frame(1, a=(True, False, True)),
    ]
    d_mot = ObservationWindow(h_mot, 2).motion_detail("a")
    if not (d_mot["estimate"] is None and d_mot["nonfinite_points"] == 1
            and d_mot["reason"] == "nonfinite_observation"):
        return False, "bool motion det=%r" % (d_mot,)

    # _context_points() skips historical bool frame to reach earlier real observation
    h_bf = [
        _frame(0, a=(0.0, 1.5, 2.0)),
        _frame(1, a=(1.0, 1.5, 2.0)),
        _frame(2, a=(True, 1.5, 2.0)),
        _frame(3, a=(3.0, 1.5, 2.0)),
    ]
    ow_bf = ObservationWindow(h_bf, 1)
    if ow_bf.motion("a") != (1.0, 1.0, 1, (1.0, 0.0, 0.0)):
        return False, "bool backfill est=%r" % (ow_bf.motion("a"),)
    return True, ""


# --------------------------------------------------------------------------- #
# Group F: Positive finite regressions
# --------------------------------------------------------------------------- #
def _check_f01_positive_finite_regressions_unchanged():
    """F01: Normal finite trajectories preserve exact pair, motion, and cache semantics."""
    for nested in (False, True):
        h = [
            _frame(1, nested, a=(2.0, 1.5, 2.0), b=(20.0, 1.5, 2.0)),
            _frame(2, nested),
            _frame(3, nested, a=(4.0, 1.5, 2.0), b=(20.0, 1.5, 2.0)),
            _frame(4, nested, a=(6.0, 1.5, 2.0), b=(20.0, 1.5, 2.0)),
        ]
        ow = ObservationWindow(h, 3, nested=nested)
        if ow.motion("a") != (1.0, 1.5, 2, (1.0, 0.0, 0.0)):
            return False, "nested=%s motion=%r" % (nested, ow.motion("a"))
        if ow.pair("a", "b") != (1.0, 2, 2) or ow.pair("b", "a") != (0.0, 0, 2):
            return False, "nested=%s pair=%r/%r" % (
                nested, ow.pair("a", "b"), ow.pair("b", "a"))

    # Full-window stationary observations must NOT backfill pre-window moving frame
    h_stop = [
        _frame(0, a=(0.0, 1.5, 2.0), b=(20.0, 1.5, 2.0)),
        _frame(1, a=(5.0, 1.5, 2.0), b=(20.0, 1.5, 2.0)),
        _frame(2, a=(5.0, 1.5, 2.0), b=(20.0, 1.5, 2.0)),
        _frame(3, a=(5.0, 1.5, 2.0), b=(20.0, 1.5, 2.0)),
    ]
    ow_stop = ObservationWindow(h_stop, 3)
    if ow_stop.pair("a", "b") != (0.0, 0, 2):
        return False, "full stationary window backfilled unexpectedly: %r" % (
            ow_stop.pair("a", "b"),)
    return True, ""


_SUITE = [
    ("A01", _check_a01_pair_first_frame_nonfinite_never_poisons_previous),
    ("A02", _check_a02_pair_skips_leading_nonfinite_and_measures_valid_suffix),
    ("A03", _check_a03_pair_window_nonfinite_backfill_and_early_exit),
    ("B01", _check_b01_pair_backfill_skips_nonfinite_to_reach_real_anchor),
    ("B02", _check_b02_pair_backfill_only_nonfinite_before_window_gives_zero),
    ("C01", _check_c01_context_backfill_window1_skips_nonfinite_history),
    ("C02", _check_c02_context_only_occlusion_skips_nonfinite_history),
    ("C03", _check_c03_nonfinite_reason_preserved_when_no_finite_pair_exists),
    ("D01", _check_d01_pair_1d_subtraction_overflow_stays_exact_and_finite),
    ("D02", _check_d02_pair_2d_diagonal_hypot_overflow_and_large_ints),
    ("D03", _check_d03_world_model_extreme_pair_stays_finite_json),
    ("E01", _check_e01_finite_pos_rejects_bool_and_invalid_accepts_numbers),
    ("E02", _check_e02_bool_coordinates_rejected_in_pair_and_motion),
    ("F01", _check_f01_positive_finite_regressions_unchanged),
]


# --------------------------------------------------------------------------- #
# pytest entrypoints (no module-level sys.exit)
# --------------------------------------------------------------------------- #
def test_a01_pair_first_frame_nonfinite_never_poisons_previous():
    ok, detail = _check_a01_pair_first_frame_nonfinite_never_poisons_previous()
    assert ok, detail


def test_a02_pair_skips_leading_nonfinite_and_measures_valid_suffix():
    ok, detail = _check_a02_pair_skips_leading_nonfinite_and_measures_valid_suffix()
    assert ok, detail


def test_a03_pair_window_nonfinite_backfill_and_early_exit():
    ok, detail = _check_a03_pair_window_nonfinite_backfill_and_early_exit()
    assert ok, detail


def test_b01_pair_backfill_skips_nonfinite_to_reach_real_anchor():
    ok, detail = _check_b01_pair_backfill_skips_nonfinite_to_reach_real_anchor()
    assert ok, detail


def test_b02_pair_backfill_only_nonfinite_before_window_gives_zero():
    ok, detail = _check_b02_pair_backfill_only_nonfinite_before_window_gives_zero()
    assert ok, detail


def test_c01_context_backfill_window1_skips_nonfinite_history():
    ok, detail = _check_c01_context_backfill_window1_skips_nonfinite_history()
    assert ok, detail


def test_c02_context_only_occlusion_skips_nonfinite_history():
    ok, detail = _check_c02_context_only_occlusion_skips_nonfinite_history()
    assert ok, detail


def test_c03_nonfinite_reason_preserved_when_no_finite_pair_exists():
    ok, detail = _check_c03_nonfinite_reason_preserved_when_no_finite_pair_exists()
    assert ok, detail


def test_d01_pair_1d_subtraction_overflow_stays_exact_and_finite():
    ok, detail = _check_d01_pair_1d_subtraction_overflow_stays_exact_and_finite()
    assert ok, detail


def test_d02_pair_2d_diagonal_hypot_overflow_and_large_ints():
    ok, detail = _check_d02_pair_2d_diagonal_hypot_overflow_and_large_ints()
    assert ok, detail


def test_d03_world_model_extreme_pair_stays_finite_json():
    ok, detail = _check_d03_world_model_extreme_pair_stays_finite_json()
    assert ok, detail


def test_e01_finite_pos_rejects_bool_and_invalid_accepts_numbers():
    ok, detail = _check_e01_finite_pos_rejects_bool_and_invalid_accepts_numbers()
    assert ok, detail


def test_e02_bool_coordinates_rejected_in_pair_and_motion():
    ok, detail = _check_e02_bool_coordinates_rejected_in_pair_and_motion()
    assert ok, detail


def test_f01_positive_finite_regressions_unchanged():
    ok, detail = _check_f01_positive_finite_regressions_unchanged()
    assert ok, detail


# --------------------------------------------------------------------------- #
# Direct execution & isolated sandbox mutation self-verification
# --------------------------------------------------------------------------- #
def _run_assertions() -> int:
    fails = 0
    for cid, fn in _SUITE:
        try:
            ok, detail = fn()
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, "%s: %s" % (type(exc).__name__, exc)
        if ok:
            print("  PASS %s" % cid)
        else:
            print("  FAIL %s  %s" % (cid, detail[:240]))
            fails += 1
    print("passed %d / failed %d" % (len(_SUITE) - fails, fails))
    return fails


_MUTANTS = {
    # M1: Re-introduce first-frame non-finite anchor poisoning in pair()
    "M1": (
        "lingshu/world/motion.py",
        [(
            "            valid_frames = []\n"
            "            for index, frame in enumerate(self.frames):\n"
            "                if not (common & (1 << index)):\n"
            "                    continue\n"
            "                pa, pb = frame[a], frame[b]\n"
            "                # #1115：每一对坐标先过 `_finite_pos`（含首帧），非有限/不可解包/\n"
            "                # bool/超大 int 一律跳过，既不成为 `previous` 锚点，也不计 intervals\n"
            "                # 或进 scores（与「缺观测不计」一致：后续有限对仍以此前最后有限帧为参照）。\n"
            "                if not (_finite_pos(pa) and _finite_pos(pb)):\n"
            "                    continue\n"
            "                valid_frames.append((pa, pb))\n"
            "            if not valid_frames:\n"
            "                self._pairs[(a, b)] = self._pairs[(b, a)] = (0.0, 0, 0)\n"
            "                return 0.0, 0, 0\n"
            "            previous = None\n"
            "            if self.start and len(valid_frames) < len(self.frames):",
            "            valid_frames = [\n"
            "                (frame[a], frame[b])\n"
            "                for index, frame in enumerate(self.frames)\n"
            "                if (common & (1 << index))\n"
            "            ]\n"
            "            previous = None\n"
            "            if self.start and common != (1 << len(self.frames)) - 1:",
        ), (
            "            for pa, pb in valid_frames:\n"
            "                if previous is not None:\n"
            "                    intervals += 1",
            "            for pa, pb in valid_frames:\n"
            "                if previous is not None:\n"
            "                    if not (_finite_pos(pa) and _finite_pos(pb)):\n"
            "                        continue\n"
            "                    intervals += 1",
        )],
        {"A01", "A02", "A03", "E02"},
    ),
    # M2a: Re-introduce unvalidated historical backfill in pair()
    "M2a": (
        "lingshu/world/motion.py",
        [(
            "                        if _finite_pos(cand_a) and _finite_pos(cand_b):\n"
            "                            previous = (cand_a, cand_b)\n"
            "                            break",
            "                        previous = (cand_a, cand_b)\n"
            "                        break",
        )],
        {"B01", "B02"},
    ),
    # M2b: Re-introduce non-finite quota consumption in _context_points()
    "M2b": (
        "lingshu/world/motion.py",
        [(
            "                if _finite_pos(entry[1]):\n"
            "                    kept.append(entry)",
            "                kept.append(entry)",
        )],
        {"C01", "C02", "E02"},
    ),
    # M3: Remove overflow rescaling in _safe_delta_2d()
    "M3": (
        "lingshu/world/motion.py",
        [(
            "    if not (math.isfinite(dx) and math.isfinite(dz) and math.isfinite(length)):\n"
            "        dx = 0.25 * fx1 - 0.25 * fx0\n"
            "        dz = 0.25 * fz1 - 0.25 * fz0\n"
            "        length = math.hypot(dx, dz)",
            "    if False:\n"
            "        dx = 0.25 * fx1 - 0.25 * fx0",
        )],
        {"D01", "D02", "D03"},
    ),
    # M4: Re-introduce bool coordinate acceptance in _finite_pos()
    "M4": (
        "lingshu/world/motion.py",
        [(
            "            and all(not isinstance(v, bool) and math.isfinite(float(v)) for v in values)",
            "            and all(math.isfinite(float(v)) for v in values)",
        )],
        {"E01", "E02"},
    ),
}


def _run_mutant_in_sandbox(rel: str, edits):
    with tempfile.TemporaryDirectory() as td:
        sandbox = Path(td)
        shutil.copytree(
            _REPO / "lingshu", sandbox / "lingshu",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        target = sandbox / rel
        norm = target.read_text(encoding="utf-8").replace("\r\n", "\n")
        for old, new in edits:
            cnt = norm.count(old)
            if cnt != 1:
                return None, "anchor hit %d times (expected 1): %r" % (cnt, old[:80])
            norm = norm.replace(old, new, 1)
        target.write_text(norm, encoding="utf-8", newline="\n")
        env = dict(os.environ)
        env["LINGSHU_MUTATE_SANDBOX"] = str(sandbox)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        r = subprocess.run(
            [sys.executable, "-B", "-X", "utf8", __file__, "--assert-only"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(_REPO), env=env, timeout=180)
        out = (r.stdout or "") + (r.stderr or "")
        failed = set(re.findall(r"(?m)^\s*FAIL\s+([A-Z]\d+)", r.stdout or ""))
        return failed, out


def _mutate(argv) -> int:
    want = ""
    for a in argv:
        if a == "--mutate":
            want = want or "ALL"
        elif a.startswith("--mutate="):
            want = a.split("=", 1)[1]
    if not want:
        return 0
    ids = sorted(_MUTANTS) if want == "ALL" else [want]
    red = 0
    for mid in ids:
        if mid not in _MUTANTS:
            print("[ANCHOR-MISS] %s: unknown mutant" % mid)
            return 2
        rel, edits, expect = _MUTANTS[mid]
        actual, out = _run_mutant_in_sandbox(rel, edits)
        if actual is None:
            print("[ANCHOR-MISS] %s: %s" % (mid, out))
            return 2
        if actual == expect:
            print("[MUTATE-OK]   %s: expected assertions turned red %s" % (mid, sorted(actual)))
        else:
            print("[MUTATE-FAIL] %s: expected %s, actual %s"
                  % (mid, sorted(expect), sorted(actual)))
            for line in out.splitlines():
                if line.strip().startswith("FAIL"):
                    print("    " + line)
            red += 1
    print("mutation summary: %d mutants, %s"
          % (len(ids), "all matched" if red == 0 else "%d mismatched" % red))
    return 1 if red else 0


if __name__ == "__main__":
    _f = _run_assertions()
    if "--assert-only" in sys.argv:
        sys.exit(1 if _f else 0)
    _m = _mutate(sys.argv[1:])
    sys.exit(1 if (_f or _m) else 0)
