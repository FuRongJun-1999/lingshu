# -*- coding: utf-8 -*-
"""Issue #196 regression & isolated-sandbox mutation guards for anim, shade_paint, and mesh_svg."""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

_REPO = Path(__file__).resolve().parent.parent


def test_anim_size_background_path_traversal_and_no_tmp_leak(tmp_path):
    PIL = pytest.importorskip("PIL.Image")
    from lingshu.world import anim

    class DummyRenderer:
        def render(self, w, h, **_kw):
            return PIL.new("RGB", (w, h), (10, 20, 30))

    for bad_size in ((True, 10), (10, False), (10.5, 10), (float("inf"), 10), (10,), (10, 10, 10), "10x10"):
        with pytest.raises((TypeError, ValueError)):
            anim.frames(DummyRenderer(), 1, size=bad_size)

    img1 = PIL.new("RGB", (8, 8), (10, 20, 30))
    img2 = PIL.new("RGB", (8, 8), (40, 50, 60))

    for bad_bg in ((True, 0, 0), (0, 1.5, 0), (-1, 0, 0), (0, 256, 0), (0, 0), (0, 0, 0, 0)):
        with pytest.raises((TypeError, ValueError)):
            anim.to_strip([img1], background=bad_bg)

    # save_frames rejects path traversal in prefix / ext
    out_dir = tmp_path / "frames"
    for bad_prefix in ("../escape", "sub/frame", r"sub\frame", "..", "", "C:frame"):
        with pytest.raises(ValueError):
            anim.save_frames([img1], out_dir, prefix=bad_prefix)
    for bad_ext in ("png/../x", "png/bad", "p.ng"):
        with pytest.raises(ValueError):
            anim.save_frames([img1], out_dir, ext=bad_ext)

    # to_gif must NOT leak .tmp files when _quantize fails
    gif_dir = tmp_path / "gif_out"
    gif_dir.mkdir()
    with mock.patch.object(anim, "_quantize", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError, match="boom"):
            anim.to_gif([img1, img2], gif_dir / "out.gif")
    assert list(gif_dir.iterdir()) == [], f"leaked temp files: {list(gif_dir.iterdir())}"


def test_shade_paint_nonfinite_overflow_and_bool_guards():
    from lingshu.world import shade_paint as sp

    # shade_rgb handles 0 * inf and inf channel without NaN/OverflowError
    assert sp.shade_rgb((0, 0, 0), float("inf")) == (0, 0, 0)
    assert sp.shade_rgb((255, 255, 255), float("inf")) == (255, 255, 255)
    assert sp.shade_rgb((float("inf"), 0, float("nan")), 1.0) == (255, 0, 0)

    # shade_rgb_legacy handles NaN / inf without crashing while preserving finite values
    assert sp.shade_rgb_legacy((120, 100, 80), 0.5) == (60, 50, 40)
    assert sp.shade_rgb_legacy((100, 100, 100), float("nan")) == (0, 0, 0)
    assert sp.shade_rgb_legacy((0, 100, 100), float("inf")) == (0, 255, 255)

    # shade_studio uses hypot (no overflow on 1e200 normal) and rejects non-finite/bool rim_scale
    v_unit = sp.shade_studio((1.0, 1.0, 1.0), (0.0, 0.0, -1.0))
    v_huge = sp.shade_studio((1e200, 1e200, 1e200), (0.0, 0.0, -1.0))
    assert math.isclose(v_unit, v_huge, rel_tol=1e-9, abs_tol=1e-9)
    for bad_rs in (float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            sp.shade_studio((0, 1, 0), (0, 0, -1), rim_scale=bad_rs)

    # env_tint rejects non-finite/bool gain & ny & base and clamps RGB to 0..255
    for kw in ({"gain": float("nan")}, {"gain": float("inf")}, {"gain": True}):
        with pytest.raises(ValueError):
            sp.env_tint((100, 100, 100), 0.5, "metal", **kw)
    # Degenerate face normal ny=nan returns neutral clamped base (neither ground-biased nor crashing mesh render)
    assert sp.env_tint((100, 100, 100), float("nan"), "metal") == (100, 100, 100)
    with pytest.raises(ValueError):
        sp.env_tint((100, 100, 100), True, "metal")
    clamped = sp.env_tint((250, 10, 10), 1.0, "metal", gain=100.0)
    assert all(0 <= c <= 255 for c in clamped)

    # shade_intensity_mode toon rejects bool / non-int bands
    for bad_bands in (True, False, float("inf"), float("nan"), 2.5):
        with pytest.raises(ValueError):
            sp.shade_intensity_mode((0, 1, 0), (0, 0, -1), "toon", bands=bad_bands)


def test_mesh_svg_project_depth_color_and_style_guards():
    from lingshu.world import mesh_svg, world3d

    cam = world3d.Camera3D()

    # project_point returns None for non-finite coords and rejects bad screen dims
    assert mesh_svg.project_point((float("nan"), 0.0, 5.0), cam, 800, 600) is None
    assert mesh_svg.project_point((0.0, float("inf"), 5.0), cam, 800, 600) is None
    assert mesh_svg.project_point((0.0, 0.0, float("inf")), cam, 800, 600) is None
    for bad_dim in (0, -10, True):
        with pytest.raises(ValueError):
            mesh_svg.project_point((0.0, 0.0, 5.0), cam, bad_dim, 600)

    # _project_edge returns None on inf/nan coords instead of collapsing to center
    assert mesh_svg._project_edge((0.0, 0.0, float("inf")), (1.0, 0.0, 5.0), (0, 0, 0), cam, 800, 600, 0.05) is None
    assert mesh_svg._project_edge((float("nan"), 0.0, 5.0), (1.0, 0.0, 5.0), (0, 0, 0), cam, 800, 600, 0.05) is None

    # LayeredSVGBackend.set_depth rejects nan/inf/bool
    be = mesh_svg.LayeredSVGBackend(100, 100)
    for bad_z in (float("nan"), float("inf"), -float("inf"), True):
        with pytest.raises(ValueError):
            be.set_depth(bad_z)

    # _darken / _hex / _tone reject non-finite / bool rgb channels
    for bad_rgb in ((float("nan"), 0, 0), (float("inf"), 0, 0), (True, 0, 0)):
        with pytest.raises(ValueError):
            mesh_svg._darken(bad_rgb, 0.5)
        with pytest.raises(ValueError):
            mesh_svg._hex(bad_rgb)
        with pytest.raises(ValueError):
            mesh_svg._tone(bad_rgb, "lighten", 0.5)

    # render_mesh_svg rejects non-dict/str style with ValueError (not naked TypeError)
    w = world3d.World3D()
    with pytest.raises(ValueError):
        mesh_svg.render_mesh_svg(w, style=123)


_MUTATION_ARMS = {
    "M1_anim_prefix_traversal": (
        "lingshu/world/anim.py",
        'if any(sep in prefix for sep in ("/", "\\\\", ":", "\\x00")) or os.path.splitdrive(prefix)[0] or ".." in prefix:',
        "if False:",
        "test_anim_size_background_path_traversal_and_no_tmp_leak",
    ),
    "M2_shade_rgb_inf": (
        "lingshu/world/shade_paint.py",
        "        if not math.isfinite(i):\n            v = 0.0 if lin == 0.0 else 1.0",
        "        if False:\n            v = 0.0 if lin == 0.0 else 1.0",
        "test_shade_paint_nonfinite_overflow_and_bool_guards",
    ),
    "M3_shade_studio_hypot": (
        "lingshu/world/shade_paint.py",
        "    nl = math.hypot(nx, ny, nz)",
        "    nl = math.sqrt(nx * nx + ny * ny + nz * nz)",
        "test_shade_paint_nonfinite_overflow_and_bool_guards",
    ),
    "M4_mesh_svg_set_depth": (
        "lingshu/world/mesh_svg.py",
        '        z = _require_finite_range(inv_z, "inv_z")',
        "        z = float(inv_z)",
        "test_mesh_svg_project_depth_color_and_style_guards",
    ),
    "M5_mesh_svg_project_point": (
        "lingshu/world/mesh_svg.py",
        "    if not (all(math.isfinite(float(v)) for v in c) and math.isfinite(x) and math.isfinite(y) and math.isfinite(invz)):\n        return None",
        "    if False:\n        return None",
        "test_mesh_svg_project_depth_color_and_style_guards",
    ),
}


def _run_mutate(want: str) -> int:
    arms = list(_MUTATION_ARMS) if want == "ALL" else [want]
    failed = 0
    for arm in arms:
        if arm not in _MUTATION_ARMS:
            print(f"[ANCHOR-MISS] unknown arm {arm}")
            return 2
        rel, old, new, test_name = _MUTATION_ARMS[arm]
        src_path = _REPO / rel
        orig_bytes = src_path.read_bytes()
        tmp = Path(tempfile.mkdtemp(prefix=f"mut_l3_{arm}_"))
        try:
            shutil.copytree(_REPO / "lingshu", tmp / "lingshu",
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            (tmp / "tests").mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(__file__), tmp / "tests" / Path(__file__).name)
            target = tmp / rel
            text = target.read_bytes().decode("utf-8").replace("\r\n", "\n")
            if text.count(old) != 1:
                print(f"[ANCHOR-MISS] {arm}: count={text.count(old)} in {rel}")
                return 2
            target.write_bytes(text.replace(old, new, 1).encode("utf-8"))
            env = os.environ.copy()
            env["PYTHONPATH"] = str(tmp)
            r = subprocess.run(
                [sys.executable, "-X", "utf8", "-m", "pytest", "-q",
                 str(tmp / "tests" / Path(__file__).name), "-k", test_name],
                cwd=str(tmp), env=env, capture_output=True, text=True,
                encoding="utf-8", errors="replace",
            )
            untouched = src_path.read_bytes() == orig_bytes
            if r.returncode != 0 and untouched:
                print(f"[MUTATE-OK] {arm} killed (rc={r.returncode}, repo_files_untouched={untouched})")
            else:
                print(f"[MUTATE-FAIL] {arm} survived (rc={r.returncode}, repo_files_untouched={untouched})")
                failed += 1
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    want = None
    for a in sys.argv[1:]:
        if a == "--mutate":
            want = "ALL"
        elif a.startswith("--mutate="):
            want = a.split("=", 1)[1]
    if want is not None:
        raise SystemExit(_run_mutate(want))
    raise SystemExit(pytest.main(["-q", __file__]))
