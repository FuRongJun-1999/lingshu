# -*- coding: utf-8 -*-
"""Guard test for pixel-center right-edge bound in ``vec.geom2d.DepthBuffer``.

Bug summary
-----------
In ``bridges/blender/wm_bridge/factory/vec/geom2d.py``, ``DepthBuffer._add_one``
samples pixels at pixel centers ``(x + 0.5, y + 0.5)``. A pixel center falls
inside scanline span ``[xa, xb]`` iff ``xa <= x + 0.5 <= xb``, which solves to:

    ceil(xa - 0.5) <= x <= floor(xb - 0.5)

Both the numpy path and the pure-Python fallback erroneously computed the upper
bound as ``floor(xb + 0.5)``:
  1. Every scanline over-filled 1 extra pixel outside the right edge (e.g. a
     triangle spanning ``x in [10, 20]`` wrote pixel ``x = 20`` whose center is
     ``20.5 > 20``).
  2. On narrow acute slivers such as
     ``((20, 10.49, 10), (20.02, 10.5, 1), (20, 30, 10))``, sampling the
     out-of-triangle center ``xc = 20.5`` produced ``tt = 25.0 >> 1.0`` and
     linearly extrapolated ``1/z = 22.6`` (``z = 0.0442``, 22.6x closer than
     the nearest vertex ``z = 1.0``), poisoning ``DepthBuffer`` and downstream
     ``split_hidden`` visibility.

Usage (from repository root):
    pytest:  <py313> -X utf8 -m pytest tests/test_vec_depthbuffer_right_edge.py -q
    direct:  <py313|py312> -X utf8 tests/test_vec_depthbuffer_right_edge.py
    mutate:  <py313|py312> -X utf8 tests/test_vec_depthbuffer_right_edge.py --mutate
"""
from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BRIDGE_ROOT = REPO / "bridges" / "blender"
if str(BRIDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(BRIDGE_ROOT))

from wm_bridge.factory.vec import geom2d as G  # noqa: E402

_ORIG_HAVE_NUMPY = bool(G.HAVE_NUMPY)

#: Acute sliver whose scanline span at y=10 (yc=10.5) is [20.0, 20.02],
#: containing no pixel center; pre-fix wrote 1/z = 22.6 at pixel (20, 10).
SLIVER_TRI = (
    (20.0, 10.49, 10.0),
    (20.02, 10.5, 1.0),
    (20.0, 30.0, 10.0),
)

#: Right triangle spanning x in [10, 20], y in [10, 20] at constant z = 5.0.
FLAT_RIGHT_TRI = (
    (10.0, 10.0, 5.0),
    (20.0, 10.0, 5.0),
    (20.0, 20.0, 5.0),
)

#: Quad [10, 20] x [10, 20] split into two triangles at constant z = 5.0.
FLAT_QUAD_TRIS = (
    ((10.0, 10.0, 5.0), (20.0, 10.0, 5.0), (20.0, 20.0, 5.0)),
    ((10.0, 10.0, 5.0), (20.0, 20.0, 5.0), (10.0, 20.0, 5.0)),
)

#: Steep 1/z triangle that genuinely covers pixel centers x=19 (19.5) and
#: x=20 (20.5) on y=10, while excluding x=21 (21.5).
STEEP_VALID_TRI = (
    (19.5, 10.0, 10.0),
    (20.6, 10.5, 1.0),
    (19.5, 30.0, 10.0),
)


class _SkipNoNumpy(Exception):
    """Raised when a numpy-only check runs in a pure-stdlib runtime."""


def _require_numpy():
    if not _ORIG_HAVE_NUMPY:
        pytest_mod = sys.modules.get("pytest")
        if pytest_mod is not None:
            pytest_mod.skip("numpy unavailable")
        raise _SkipNoNumpy("numpy unavailable")


@contextmanager
def _with_backend(use_numpy: bool):
    """Temporarily select numpy or pure-Python backend on ``geom2d``."""
    if use_numpy and not _ORIG_HAVE_NUMPY:
        raise RuntimeError("numpy backend requested in environment without numpy")
    prev = G.HAVE_NUMPY
    G.HAVE_NUMPY = bool(use_numpy)
    try:
        yield
    finally:
        G.HAVE_NUMPY = prev


def _available_backends():
    """Return available backends in this Python runtime."""
    return (True, False) if _ORIG_HAVE_NUMPY else (False,)


def _nonzero_pixels(zbuf):
    """Return dict {(x, y): invz} for all non-zero cells in ``zbuf``."""
    out = {}
    for y in range(zbuf.height):
        for x in range(zbuf.width):
            v = zbuf.invz_at(x + 0.5, y + 0.5)
            if v != 0.0:
                out[(x, y)] = v
    return out


def _barycentric(px, py, tri):
    """Return exact barycentric coordinates (l0, l1, l2) of (px, py) in ``tri``."""
    (x0, y0, _), (x1, y1, _), (x2, y2, _) = tri
    det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    l0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / det
    l1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / det
    l2 = 1.0 - l0 - l1
    return l0, l1, l2


def _make_camera(width=40, height=40):
    return G.Camera(
        eye=(0.0, -5.0, 0.0),
        look_at=(0.0, 0.0, 0.0),
        up=(0.0, 0.0, 1.0),
        fov_deg=60.0,
        width=width,
        height=height,
    )


def _unproject(cam, sx, sy, zc):
    """Invert ``Camera.project`` for screen pixel ``(sx, sy)`` at depth ``zc``."""
    xc = (sx - cam.width / 2.0) * zc / cam.focal_px
    yc = (cam.height / 2.0 - sy) * zc / cam.focal_px
    return cam._cam_to_world((xc, yc, zc))


# ============================================================ Tests


def test_sliver_no_extrapolation_numpy():
    """Acute sliver must not extrapolate 1/z = 22.6 at (20, 10) (numpy path)."""
    _require_numpy()
    with _with_backend(True):
        zb = G.DepthBuffer(40, 40)
        assert zb.stats()["backend"] == "numpy"
        drawn = zb.add_triangles([SLIVER_TRI])
        assert drawn == 1
        assert zb.invz_at(20.5, 10.5) == 0.0, (
            "numpy path wrote out-of-triangle pixel (20, 10): invz=%r"
            % (zb.invz_at(20.5, 10.5),)
        )
        assert _nonzero_pixels(zb) == {}, (
            "sliver span [20.0, 20.02] contains no pixel centers, got %r"
            % (_nonzero_pixels(zb),)
        )
        assert zb.pixels_written == 0


def test_sliver_no_extrapolation_pure_python():
    """Acute sliver must not extrapolate 1/z = 22.6 at (20, 10) (pure-Python path)."""
    with _with_backend(False):
        zb = G.DepthBuffer(40, 40)
        assert zb.stats()["backend"] == "pure-python"
        drawn = zb.add_triangles([SLIVER_TRI])
        assert drawn == 1
        assert zb.invz_at(20.5, 10.5) == 0.0, (
            "pure-Python path wrote out-of-triangle pixel (20, 10): invz=%r"
            % (zb.invz_at(20.5, 10.5),)
        )
        assert _nonzero_pixels(zb) == {}, (
            "sliver span [20.0, 20.02] contains no pixel centers, got %r"
            % (_nonzero_pixels(zb),)
        )
        assert zb.pixels_written == 0


def _assert_flat_right_edge(use_numpy: bool):
    with _with_backend(use_numpy):
        zb_tri = G.DepthBuffer(40, 40)
        zb_tri.add_triangles([FLAT_RIGHT_TRI])
        x20_hits = [
            y for y in range(40) if zb_tri.invz_at(20.5, y + 0.5) != 0.0
        ]
        assert x20_hits == [], (
            "right edge x=20 must not be written (center 20.5 > 20.0), got y=%r"
            % (x20_hits,)
        )
        # Rows y=10..19 have spans [10.5..20], [11.5..20], ..., [19.5..20]:
        # pixel counts 10 + 9 + ... + 1 = 55.
        assert zb_tri.pixels_written == 55, (
            "expected 55 pixels for right triangle, got %d"
            % (zb_tri.pixels_written,)
        )
        assert abs(zb_tri.invz_at(19.5, 10.5) - 0.2) < 1e-6

        zb_quad = G.DepthBuffer(40, 40)
        zb_quad.add_triangles(FLAT_QUAD_TRIS)
        nz = _nonzero_pixels(zb_quad)
        expected = {(x, y) for y in range(10, 20) for x in range(10, 20)}
        assert set(nz.keys()) == expected, (
            "quad [10,20]x[10,20] must cover exactly {10..19}x{10..19}, "
            "extra=%r missing=%r"
            % (sorted(set(nz.keys()) - expected), sorted(expected - set(nz.keys())))
        )


def test_flat_right_edge_no_1px_overrun_numpy():
    """Flat [10, 20] triangle/quad must not over-fill pixel x=20 (numpy path)."""
    _require_numpy()
    _assert_flat_right_edge(True)


def test_flat_right_edge_no_1px_overrun_pure_python():
    """Flat [10, 20] triangle/quad must not over-fill pixel x=20 (pure-Python path)."""
    _assert_flat_right_edge(False)


def test_invz_convex_hull_and_exact_pixel_center_coverage():
    """Every written pixel must lie inside the triangle with 1/z in vertex convex hull."""
    rng = random.Random(20261122)
    tris = [
        SLIVER_TRI,
        FLAT_RIGHT_TRI,
        STEEP_VALID_TRI,
        # Exact pixel-center apex on y=10 (yc=10.5 -> xa == xb == 20.5, denom == 0)
        ((20.5, 10.5, 2.0), (25.0, 20.0, 4.0), (16.0, 20.0, 8.0)),
        # Integer apex on y=10 (yc=10.5 -> xa == xb == 20.0 -> lo=20 > hi=19, skip)
        ((20.0, 10.5, 2.0), (25.0, 20.0, 4.0), (16.0, 20.0, 8.0)),
        # Horizontal top edge on exact scanline yc=10.5
        ((10.0, 10.5, 2.0), (20.0, 10.5, 1.0), (15.0, 20.5, 5.0)),
        # Horizontal bottom edge on exact scanline yc=19.5
        ((15.0, 10.5, 5.0), (10.0, 19.5, 2.0), (20.0, 19.5, 1.0)),
        # Half-shifted exact pixel-center inclusive quad halves [10.5, 19.5]
        ((10.5, 10.5, 2.0), (19.5, 10.5, 1.0), (19.5, 19.5, 4.0)),
        # Partially off-screen left/right/top/bottom clipping
        ((-3.2, 5.0, 2.0), (4.4, 8.0, 1.0), (0.6, 15.0, 6.0)),
        ((35.2, 34.0, 3.0), (44.0, 36.0, 1.5), (38.0, 43.0, 9.0)),
        # Sub-pixel narrow span straddling a single pixel center x=15.5
        ((15.45, 5.0, 10.0), (15.55, 10.5, 1.0), (15.45, 20.0, 10.0)),
    ]
    for _ in range(30):
        x0, y0 = rng.uniform(-2.0, 42.0), rng.uniform(-2.0, 42.0)
        x1, y1 = rng.uniform(-2.0, 42.0), rng.uniform(-2.0, 42.0)
        x2, y2 = rng.uniform(-2.0, 42.0), rng.uniform(-2.0, 42.0)
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(area) < 1e-6:
            continue
        z0 = rng.uniform(0.5, 20.0)
        z1 = rng.uniform(0.5, 20.0)
        z2 = rng.uniform(0.5, 20.0)
        tris.append(((x0, y0, z0), (x1, y1, z1), (x2, y2, z2)))

    for use_numpy in _available_backends():
        tol_scale = 1e-6 if use_numpy else 1e-11
        with _with_backend(use_numpy):
            # Specific check on steep valid triangle
            zb_steep = G.DepthBuffer(40, 40)
            zb_steep.add_triangles([STEEP_VALID_TRI])
            nz_steep = _nonzero_pixels(zb_steep)
            assert (19, 10) in nz_steep and (20, 10) in nz_steep
            assert (21, 10) not in nz_steep
            expected_w20 = 0.1 + 0.9 * (20.5 - 19.5) / (20.6 - 19.5)
            assert abs(nz_steep[(19, 10)] - 0.1) <= tol_scale
            assert abs(nz_steep[(20, 10)] - expected_w20) <= tol_scale
            for (px, py), val in nz_steep.items():
                assert 0.1 - tol_scale <= val <= 1.0 + tol_scale, (
                    "steep tri pixel (%d,%d) invz=%r outside [0.1, 1.0]"
                    % (px, py, val)
                )

            # Specific check on exact pixel-center apex (denom == 0) vs integer apex
            zb_apex_hit = G.DepthBuffer(40, 40)
            zb_apex_hit.add_triangles([tris[3]])
            assert abs(zb_apex_hit.invz_at(20.5, 10.5) - 0.5) <= tol_scale

            zb_apex_miss = G.DepthBuffer(40, 40)
            zb_apex_miss.add_triangles([tris[4]])
            assert zb_apex_miss.invz_at(20.5, 10.5) == 0.0

            # Full convex-hull + barycentric oracle across all triangles
            for idx, tri in enumerate(tris):
                zb = G.DepthBuffer(40, 40)
                zb.add_triangles([tri])
                ws = (1.0 / tri[0][2], 1.0 / tri[1][2], 1.0 / tri[2][2])
                w_lo, w_hi = min(ws), max(ws)
                w_tol = tol_scale * max(1.0, w_hi)
                for y in range(40):
                    yc = y + 0.5
                    for x in range(40):
                        xc = x + 0.5
                        l0, l1, l2 = _barycentric(xc, yc, tri)
                        val = zb.invz_at(xc, yc)
                        if val != 0.0:
                            assert min(l0, l1, l2) >= -1e-9, (
                                "tri#%d backend=%s wrote outside pixel (%d,%d) bary=(%.3e,%.3e,%.3e)"
                                % (idx, use_numpy, x, y, l0, l1, l2)
                            )
                            assert w_lo - w_tol <= val <= w_hi + w_tol, (
                                "tri#%d backend=%s invz=%r outside [%r, %r] at (%d,%d)"
                                % (idx, use_numpy, val, w_lo, w_hi, x, y)
                            )
                            expected_w = l0 * ws[0] + l1 * ws[1] + l2 * ws[2]
                            assert abs(val - expected_w) <= w_tol, (
                                "tri#%d backend=%s invz=%r != bary %r at (%d,%d)"
                                % (idx, use_numpy, val, expected_w, x, y)
                            )
                        elif min(l0, l1, l2) > 1e-9:
                            raise AssertionError(
                                "tri#%d backend=%s missed interior pixel (%d,%d) bary=(%.3e,%.3e,%.3e)"
                                % (idx, use_numpy, x, y, l0, l1, l2)
                            )


def test_numpy_and_pure_python_exact_parity():
    """Numpy and pure-Python paths must agree pixel-for-pixel on mask and depth."""
    _require_numpy()
    rng = random.Random(8848)
    scene = list(FLAT_QUAD_TRIS) + [
        SLIVER_TRI,
        STEEP_VALID_TRI,
        ((20.5, 10.5, 2.0), (25.0, 20.0, 4.0), (16.0, 20.0, 8.0)),
        ((20.0, 10.5, 2.0), (25.0, 20.0, 4.0), (16.0, 20.0, 8.0)),
        ((-4.0, 2.0, 3.0), (5.5, 12.0, 1.2), (1.0, 18.0, 7.0)),
        ((42.0, 30.0, 2.5), (52.0, 40.0, 1.1), (45.0, 50.0, 6.0)),
    ]
    for _ in range(25):
        tri = tuple(
            (rng.uniform(-3.0, 51.0), rng.uniform(-3.0, 51.0), rng.uniform(0.5, 15.0))
            for _ in range(3)
        )
        scene.append(tri)

    with _with_backend(True):
        zb_np = G.DepthBuffer(48, 48)
        drawn_np = zb_np.add_triangles(scene)
        mask_np = _nonzero_pixels(zb_np)

    with _with_backend(False):
        zb_py = G.DepthBuffer(48, 48)
        drawn_py = zb_py.add_triangles(scene)
        mask_py = _nonzero_pixels(zb_py)

    assert drawn_np == drawn_py
    assert zb_np.tris_drawn == zb_py.tris_drawn
    assert zb_np.tris_skipped == zb_py.tris_skipped
    assert zb_np.pixels_written == zb_py.pixels_written, (
        "pixels_written mismatch: numpy=%d pure-python=%d"
        % (zb_np.pixels_written, zb_py.pixels_written)
    )
    assert set(mask_np.keys()) == set(mask_py.keys()), (
        "coverage mismatch: np_only=%r py_only=%r"
        % (
            sorted(set(mask_np.keys()) - set(mask_py.keys()))[:8],
            sorted(set(mask_py.keys()) - set(mask_np.keys()))[:8],
        )
    )
    for pos, v_py in mask_py.items():
        v_np = mask_np[pos]
        expected_f32 = float(G._np.float32(v_py))
        assert abs(v_np - expected_f32) <= 1e-6 * max(1.0, abs(expected_f32)), (
            "depth mismatch at %r: numpy=%r py=%r (f32=%r)"
            % (pos, v_np, v_py, expected_f32)
        )


def test_split_hidden_not_poisoned_by_sliver_or_right_edge():
    """Downstream ``split_hidden`` must not falsely occlude foreground or right-adjacent lines."""
    cam = _make_camera(40, 40)
    sliver_proj = tuple(cam.project(_unproject(cam, *v)) for v in SLIVER_TRI)
    fg_line = [_unproject(cam, 18.5, 10.5, 0.5), _unproject(cam, 22.5, 10.5, 0.5)]

    occ_quad_proj = [
        tuple(cam.project(_unproject(cam, sx, sy, 2.0)) for (sx, sy, _) in tri)
        for tri in FLAT_QUAD_TRIS
    ]
    right_adjacent_back_line = [
        _unproject(cam, 20.5, 10.5, 5.0),
        _unproject(cam, 20.5, 19.5, 5.0),
    ]

    for use_numpy in _available_backends():
        with _with_backend(use_numpy):
            # Case A: Foreground line at z=0.5 in front of sliver z in [1.0, 10.0]
            zb_sliver = G.DepthBuffer(cam.width, cam.height)
            zb_sliver.add_triangles([sliver_proj])
            runs_fg, st_fg = G.split_hidden(fg_line, cam, zb_sliver, samples_per_px=1.0)
            assert st_fg["hidden"] == 0, (
                "backend=%s foreground line z=0.5 falsely occluded by sliver z>=1.0: %r"
                % (use_numpy, st_fg)
            )
            assert len(runs_fg) == 1 and runs_fg[0][1] is True

            # Case B: Background line at x=20.5 strictly right of [10, 20] occluder
            zb_occ = G.DepthBuffer(cam.width, cam.height)
            zb_occ.add_triangles(occ_quad_proj)
            runs_adj, st_adj = G.split_hidden(
                right_adjacent_back_line, cam, zb_occ, samples_per_px=1.0
            )
            assert st_adj["hidden"] == 0, (
                "backend=%s line outside right edge x=20.5 falsely occluded: %r"
                % (use_numpy, st_adj)
            )
            assert len(runs_adj) == 1 and runs_adj[0][1] is True


def test_interior_coverage_and_true_occlusion_preserved():
    """Positive control: valid rightmost interior pixels and true occlusion remain intact."""
    cam = _make_camera(40, 40)
    occ_quad_proj = [
        tuple(cam.project(_unproject(cam, sx, sy, 2.0)) for (sx, sy, _) in tri)
        for tri in FLAT_QUAD_TRIS
    ]
    back_line_inside = [
        _unproject(cam, 19.5, 11.5, 5.0),
        _unproject(cam, 19.5, 18.5, 5.0),
    ]
    front_line_inside = [
        _unproject(cam, 19.5, 11.5, 1.0),
        _unproject(cam, 19.5, 18.5, 1.0),
    ]
    thin_valid_tri = ((19.0, 10.0, 2.0), (20.0, 10.0, 2.0), (20.0, 20.0, 2.0))

    for use_numpy in _available_backends():
        with _with_backend(use_numpy):
            zb = G.DepthBuffer(cam.width, cam.height)
            zb.add_triangles(occ_quad_proj)
            assert abs(zb.invz_at(10.5, 15.5) - 0.5) < 1e-5
            assert abs(zb.invz_at(19.5, 15.5) - 0.5) < 1e-5

            runs_back, st_back = G.split_hidden(
                back_line_inside, cam, zb, samples_per_px=1.0
            )
            assert st_back["samples"] > 0 and st_back["hidden"] == st_back["samples"], (
                "backend=%s true occlusion at rightmost valid column x=19.5 broken: %r"
                % (use_numpy, st_back)
            )
            assert len(runs_back) == 1 and runs_back[0][1] is False

            runs_front, st_front = G.split_hidden(
                front_line_inside, cam, zb, samples_per_px=1.0
            )
            assert st_front["hidden"] == 0
            assert len(runs_front) == 1 and runs_front[0][1] is True

            zb_thin = G.DepthBuffer(40, 40)
            zb_thin.add_triangles([thin_valid_tri])
            assert abs(zb_thin.invz_at(19.5, 10.5) - 0.5) < 1e-5


_ALL_TESTS = (
    test_sliver_no_extrapolation_numpy,
    test_sliver_no_extrapolation_pure_python,
    test_flat_right_edge_no_1px_overrun_numpy,
    test_flat_right_edge_no_1px_overrun_pure_python,
    test_invz_convex_hull_and_exact_pixel_center_coverage,
    test_numpy_and_pure_python_exact_parity,
    test_split_hidden_not_poisoned_by_sliver_or_right_edge,
    test_interior_coverage_and_true_occlusion_preserved,
)


def _run_all():
    passed = []
    skipped = {}
    failed = {}
    for fn in _ALL_TESTS:
        name = fn.__name__
        try:
            fn()
        except _SkipNoNumpy as exc:
            skipped[name] = str(exc)
        except AssertionError as exc:
            failed[name] = "AssertionError: %s" % (exc,)
        except Exception as exc:  # noqa: BLE001 - report unexpected test error explicitly
            failed[name] = "%s: %s" % (type(exc).__name__, exc)
        else:
            passed.append(name)
    return passed, skipped, failed


# ============================================================ --mutate self-proof

_NP_FIXED = "hi = min(maxx, int(math.floor(xb[k] - 0.5)))"
_NP_BUGGY = "hi = min(maxx, int(math.floor(xb[k] + 0.5)))"
_NP_TIGHT = "hi = min(maxx, int(math.floor(xb[k] - 1.5)))"

_PY_FIXED = "hi = min(maxx, int(math.floor(xb - 0.5)))"
_PY_BUGGY = "hi = min(maxx, int(math.floor(xb + 0.5)))"
_PY_TIGHT = "hi = min(maxx, int(math.floor(xb - 1.5)))"


def _mutation_arms():
    """Return mutation matrix tailored to whether numpy is present in this runtime."""
    if _ORIG_HAVE_NUMPY:
        return [
            (
                "M1_numpy_right_bound_plus_half",
                [(_NP_FIXED, _NP_BUGGY)],
                frozenset({
                    "test_sliver_no_extrapolation_numpy",
                    "test_flat_right_edge_no_1px_overrun_numpy",
                    "test_invz_convex_hull_and_exact_pixel_center_coverage",
                    "test_numpy_and_pure_python_exact_parity",
                    "test_split_hidden_not_poisoned_by_sliver_or_right_edge",
                }),
            ),
            (
                "M2_pure_python_right_bound_plus_half",
                [(_PY_FIXED, _PY_BUGGY)],
                frozenset({
                    "test_sliver_no_extrapolation_pure_python",
                    "test_flat_right_edge_no_1px_overrun_pure_python",
                    "test_invz_convex_hull_and_exact_pixel_center_coverage",
                    "test_numpy_and_pure_python_exact_parity",
                    "test_split_hidden_not_poisoned_by_sliver_or_right_edge",
                }),
            ),
            (
                "M3_both_paths_pre_fix_plus_half",
                [(_NP_FIXED, _NP_BUGGY), (_PY_FIXED, _PY_BUGGY)],
                frozenset({
                    "test_sliver_no_extrapolation_numpy",
                    "test_sliver_no_extrapolation_pure_python",
                    "test_flat_right_edge_no_1px_overrun_numpy",
                    "test_flat_right_edge_no_1px_overrun_pure_python",
                    "test_invz_convex_hull_and_exact_pixel_center_coverage",
                    "test_split_hidden_not_poisoned_by_sliver_or_right_edge",
                }),
            ),
            (
                "M4_both_paths_over_tight_minus_1_5",
                [(_NP_FIXED, _NP_TIGHT), (_PY_FIXED, _PY_TIGHT)],
                frozenset({
                    "test_flat_right_edge_no_1px_overrun_numpy",
                    "test_flat_right_edge_no_1px_overrun_pure_python",
                    "test_invz_convex_hull_and_exact_pixel_center_coverage",
                    "test_interior_coverage_and_true_occlusion_preserved",
                }),
            ),
        ]
    return [
        (
            "M2_pure_python_right_bound_plus_half",
            [(_PY_FIXED, _PY_BUGGY)],
            frozenset({
                "test_sliver_no_extrapolation_pure_python",
                "test_flat_right_edge_no_1px_overrun_pure_python",
                "test_invz_convex_hull_and_exact_pixel_center_coverage",
                "test_split_hidden_not_poisoned_by_sliver_or_right_edge",
            }),
        ),
        (
            "M3_both_paths_pre_fix_plus_half",
            [(_NP_FIXED, _NP_BUGGY), (_PY_FIXED, _PY_BUGGY)],
            frozenset({
                "test_sliver_no_extrapolation_pure_python",
                "test_flat_right_edge_no_1px_overrun_pure_python",
                "test_invz_convex_hull_and_exact_pixel_center_coverage",
                "test_split_hidden_not_poisoned_by_sliver_or_right_edge",
            }),
        ),
        (
            "M4_both_paths_over_tight_minus_1_5",
            [(_NP_FIXED, _NP_TIGHT), (_PY_FIXED, _PY_TIGHT)],
            frozenset({
                "test_flat_right_edge_no_1px_overrun_pure_python",
                "test_invz_convex_hull_and_exact_pixel_center_coverage",
                "test_interior_coverage_and_true_occlusion_preserved",
            }),
        ),
    ]


def _run_mutations() -> int:
    all_names = frozenset(fn.__name__ for fn in _ALL_TESTS)
    geom_rel = Path("bridges/blender/wm_bridge/factory/vec/geom2d.py")
    init_rel = Path("bridges/blender/wm_bridge/__init__.py")
    test_rel = Path("tests") / Path(__file__).name

    overall_ok = True
    arms = _mutation_arms()
    print(
        "[mutate] runtime=%s have_numpy=%s arms=%d"
        % (sys.version.split()[0], _ORIG_HAVE_NUMPY, len(arms))
    )
    for arm_name, replacements, expected_red in arms:
        tmp = Path(tempfile.mkdtemp(prefix="vec_depthbuf_mutate_"))
        try:
            dst = tmp / "repo"
            for rel in (init_rel, geom_rel, test_rel):
                target = dst / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(REPO / rel, target)

            geom_dst = dst / geom_rel
            text = geom_dst.read_text(encoding="utf-8")
            for old, new in replacements:
                hits = text.count(old)
                if hits != 1:
                    print(
                        "[mutate] FAIL %s: anchor count=%d for %r"
                        % (arm_name, hits, old)
                    )
                    return 2
                text = text.replace(old, new, 1)
            geom_dst.write_text(text, encoding="utf-8")

            proc = subprocess.run(
                [sys.executable, "-X", "utf8", str(dst / test_rel), "--run-json"],
                cwd=str(dst),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if proc.returncode not in (0, 1):
                print(
                    "[mutate] FAIL %s: unexpected rc=%d stderr=%s"
                    % (arm_name, proc.returncode, proc.stderr.strip())
                )
                overall_ok = False
                continue
            payload = json.loads(proc.stdout)
            loaded = Path(payload["geom2d_file"]).resolve()
            if loaded != geom_dst.resolve():
                print(
                    "[mutate] FAIL %s: loaded wrong geom2d %s != %s"
                    % (arm_name, loaded, geom_dst.resolve())
                )
                return 2
            red = frozenset(payload["failed"].keys())
            green = frozenset(payload["passed"])
            skipped = frozenset(payload["skipped"].keys())
            expected_skipped = (
                frozenset()
                if _ORIG_HAVE_NUMPY
                else frozenset({
                    "test_sliver_no_extrapolation_numpy",
                    "test_flat_right_edge_no_1px_overrun_numpy",
                    "test_numpy_and_pure_python_exact_parity",
                })
            )
            expected_green = all_names - expected_red - expected_skipped
            matched = (
                (red == expected_red)
                and (green == expected_green)
                and (skipped == expected_skipped)
            )
            print(
                "[mutate] %s %s red=%d/%d green=%d/%d skipped=%d/%d"
                % (
                    "OK" if matched else "MISMATCH",
                    arm_name,
                    len(red),
                    len(expected_red),
                    len(green),
                    len(expected_green),
                    len(skipped),
                    len(expected_skipped),
                )
            )
            if not matched:
                print("  expected_red:    ", sorted(expected_red))
                print("  actual_red:      ", sorted(red))
                print("  expected_skipped:", sorted(expected_skipped))
                print("  actual_skipped:  ", sorted(skipped))
                print("  failures:        ", payload["failed"])
                overall_ok = False
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    return 0 if overall_ok else 1


def main(argv) -> int:
    if "--mutate" in argv:
        return _run_mutations()
    passed, skipped, failed = _run_all()
    if "--run-json" in argv:
        print(
            json.dumps(
                {
                    "have_numpy": _ORIG_HAVE_NUMPY,
                    "geom2d_file": str(Path(G.__file__).resolve()),
                    "passed": passed,
                    "skipped": skipped,
                    "failed": failed,
                },
                ensure_ascii=False,
            )
        )
        return 0 if not failed else 1
    for name in passed:
        print("PASS %s" % (name,))
    for name, msg in skipped.items():
        print("SKIP %s :: %s" % (name, msg))
    for name, msg in failed.items():
        print("FAIL %s :: %s" % (name, msg))
    print(
        "TOTAL %d passed, %d skipped, %d failed (have_numpy=%s)"
        % (len(passed), len(skipped), len(failed), _ORIG_HAVE_NUMPY)
    )
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
