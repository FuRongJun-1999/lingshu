# -*- coding: utf-8 -*-
"""Guard tests for `mesh3d.prepare_triangles(smooth=True)` vertex normal indexing.

Defect:
  `Face.vn` stores per-corner normal indices parsed from OBJ `f v//vn` faces,
  but `prepare_triangles(smooth=True)` ignored `face.vn` and indexed
  `mesh.normals` with vertex indices `(i0, i1, i2)`:
    1. Standard OBJ meshes with `len(normals) < len(verts)` (e.g. 8-vert /
       6-normal cube) crashed with `IndexError: list index out of range`.
    2. Meshes with `len(normals) >= len(verts)` where normal indices differ
       from vertex indices silently read the wrong normals and distorted shading.

Fix rules in `prepare_triangles(smooth=True)`:
  1. If `face.vn` is not None and all 3 indices are within `[0, len(normals))`,
     index `mesh.normals` with `face.vn`.
  2. Otherwise, if `face.vn is None` and `len(mesh.normals) == len(mesh.verts)`
     (procedural per-vertex normal tables), preserve indexing via `(i0, i1, i2)`.
  3. Otherwise (mixed OBJ face without `vn` when `len(normals) != len(verts)`,
     or invalid/out-of-bounds `face.vn`), fall back to face normal `face.normal`.

Run:
  pytest:  python -X utf8 -m pytest tests/test_mesh3d_smooth_vn_index.py -q
  direct:  python -X utf8 tests/test_mesh3d_smooth_vn_index.py
  mutate:  python -X utf8 tests/test_mesh3d_smooth_vn_index.py --mutate
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lingshu.world import mesh3d  # noqa: E402
from lingshu.world.world3d import Camera3D  # noqa: E402

CUBE_OBJ_LINES = (
    "# normalize: scale=1.0 center=(0.0,0.0,0.0)\n"
    "v -1 -1 -1\n"
    "v  1 -1 -1\n"
    "v  1  1 -1\n"
    "v -1  1 -1\n"
    "v -1 -1  1\n"
    "v  1 -1  1\n"
    "v  1  1  1\n"
    "v -1  1  1\n"
    "vn  0  0 -1\n"
    "vn  0  0  1\n"
    "vn  0 -1  0\n"
    "vn  0  1  0\n"
    "vn -1  0  0\n"
    "vn  1  0  0\n"
    "f 1//1 4//1 3//1 2//1\n"
    "f 5//2 6//2 7//2 8//2\n"
    "f 1//3 2//3 6//3 5//3\n"
    "f 4//4 8//4 7//4 3//4\n"
    "f 1//5 5//5 8//5 4//5\n"
    "f 2//6 3//6 7//6 6//6\n"
).splitlines()


def _expected_view(mesh, face, center, cam):
    cx, cy, cz = center
    i0, i1, i2 = face.idx
    a, b, c = mesh.verts[i0], mesh.verts[i1], mesh.verts[i2]
    fx = (a[0] + b[0] + c[0]) / 3.0 + cx
    fy = (a[1] + b[1] + c[1]) / 3.0 + cy
    fz = (a[2] + b[2] + c[2]) / 3.0 + cz
    vx, vy, vz = cam.cx - fx, cam.cy - fy, cam.cz - fz
    vl = math.sqrt(vx * vx + vy * vy + vz * vz) or 1.0
    return (vx / vl, vy / vl, vz / vl)


def _expected_intensities(mesh, face, center, cam, shade, use_normals):
    view = _expected_view(mesh, face, center, cam)
    n = face.normal
    cosang = n[0] * view[0] + n[1] * view[1] + n[2] * view[2]
    role = mesh3d.role_for_material(face.mat)
    if use_normals is not None:
        return tuple(
            mesh3d.shade_intensity(vn, view, shade, cosang, role)
            for vn in use_normals
        )
    iv = mesh3d.shade_intensity(n, view, shade, cosang, role)
    return (iv, iv, iv)


def test_obj_cube_fewer_normals_than_verts_smooth_matches_ground_truth():
    """OBJ cube (8 verts, 6 shared normals) must not raise IndexError and must match face.vn truth."""
    cube = mesh3d.parse_obj(CUBE_OBJ_LINES, path="cube.obj", axis_swap=False, recenter=False)
    assert len(cube.verts) == 8
    assert len(cube.normals) == 6
    assert len(cube.faces) == 12
    assert all(f.vn is not None for f in cube.faces)

    # Looking at (+1,+1,+1) corner exposes vertices 6 and 7 (>= len(normals)=6).
    cam = Camera3D.look_at((3.0, 3.0, 5.0), (0.0, 0.0, 0.0))
    tris, stats = mesh3d.prepare_triangles(
        cube, (0.0, 0.0, 0.0), cam, 320, 240, shade="lambert", cull=True, smooth=True
    )
    assert stats["kept"] == 6, "Expected 3 visible quad faces (6 tris), got %r" % (stats,)

    # Verify every visible triangle's intensities match ground truth from face.vn.
    visible_faces = []
    for face in cube.faces:
        view = _expected_view(cube, face, (0.0, 0.0, 0.0), cam)
        n = face.normal
        if (n[0] * view[0] + n[1] * view[1] + n[2] * view[2]) > 0.0:
            visible_faces.append(face)
    assert len(visible_faces) == len(tris)
    for face, tri in zip(visible_faces, tris):
        got = (tri[3], tri[7], tri[11])
        vn_vecs = tuple(cube.normals[k] for k in face.vn)
        want = _expected_intensities(cube, face, (0.0, 0.0, 0.0), cam, "lambert", vn_vecs)
        for g, w in zip(got, want):
            assert abs(g - w) < 1e-12, "OBJ cube smooth intensity mismatch: got=%r want=%r" % (got, want)

    # Full bitmap render path with smooth=True and cull=False also succeeds without IndexError.
    zb = mesh3d.ZBuffer(160, 120)
    tris_all, stats_all = mesh3d.render_mesh_bitmap(
        cube, (0.0, 0.0, 0.0), cam, 160, 120, zb.depth, zb.color,
        shade="studio", cull=False, smooth=True,
    )
    assert stats_all["kept"] == 12
    assert len(tris_all) == 12
    assert any(d > 0.0 for d in zb.depth)


def test_explicit_face_vn_used_when_normals_ge_verts():
    """When len(normals) >= len(verts), explicit face.vn must be used instead of face.idx."""
    cam = Camera3D(cx=0.0, cy=0.0, cz=-5.0, yaw=0.0, pitch=0.0)
    verts = [(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)]

    # Case A: len(normals)=5 > len(verts)=3; normals[0..2] point down while face.vn=(3,4,3).
    normals = [
        (0.0, -1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0),
    ]
    face = mesh3d.Face(idx=(0, 1, 2), color=(200, 200, 200), normal=(0.0, 0.0, -1.0), vn=(3, 4, 3))
    mesh = mesh3d.MeshGeometry(verts=verts, faces=[face], normals=normals, name="vn_gt_verts")

    tris, stats = mesh3d.prepare_triangles(
        mesh, (0.0, 0.0, 0.0), cam, 320, 240, shade="lambert", cull=True, smooth=True
    )
    assert stats["kept"] == 1
    got = (tris[0][3], tris[0][7], tris[0][11])
    want = _expected_intensities(
        mesh, face, (0.0, 0.0, 0.0), cam, "lambert", (normals[3], normals[4], normals[3])
    )
    wrong = _expected_intensities(
        mesh, face, (0.0, 0.0, 0.0), cam, "lambert", (normals[0], normals[1], normals[2])
    )
    for g, w in zip(got, want):
        assert abs(g - w) < 1e-12, "Expected vn=(3,4,3) shading %r, got %r" % (want, got)
    assert abs(got[0] - wrong[0]) > 0.5, "Shading did not diverge from misread idx normals"

    # Case B: len(normals) == len(verts) == 3, but explicit face.vn=(2, 1, 0) differs from idx=(0, 1, 2).
    normals_eq = [
        (0.0, -1.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    ]
    face_eq = mesh3d.Face(idx=(0, 1, 2), color=(200, 200, 200), normal=(0.0, 0.0, -1.0), vn=(2, 1, 0))
    mesh_eq = mesh3d.MeshGeometry(verts=verts, faces=[face_eq], normals=normals_eq, name="vn_eq_verts")
    tris_eq, _ = mesh3d.prepare_triangles(
        mesh_eq, (0.0, 0.0, 0.0), cam, 320, 240, shade="lambert", cull=True, smooth=True
    )
    got_eq = (tris_eq[0][3], tris_eq[0][7], tris_eq[0][11])
    want_eq = _expected_intensities(
        mesh_eq, face_eq, (0.0, 0.0, 0.0), cam, "lambert", (normals_eq[2], normals_eq[1], normals_eq[0])
    )
    for g, w in zip(got_eq, want_eq):
        assert abs(g - w) < 1e-12, "Explicit face.vn must override vertex order when len(normals)==len(verts)"
    assert abs(got_eq[0] - got_eq[2]) > 0.5


def test_procedural_mesh_vn_none_per_vertex_normals_compat():
    """Procedural meshes (face.vn=None, len(normals)==len(verts)) keep 100% per-vertex smooth shading."""
    cam = Camera3D(cx=0.0, cy=0.0, cz=-5.0, yaw=0.0, pitch=0.0)
    verts = [(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)]
    normals = [
        (0.0, -1.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    ]
    face = mesh3d.Face(idx=(0, 1, 2), color=(180, 190, 200), normal=(0.0, 0.0, -1.0), vn=None)
    mesh = mesh3d.MeshGeometry(verts=verts, faces=[face], normals=normals, name="procedural_smooth")

    tris_smooth, stats_s = mesh3d.prepare_triangles(
        mesh, (0.0, 0.0, 0.0), cam, 320, 240, shade="lambert", cull=True, smooth=True
    )
    tris_flat, stats_f = mesh3d.prepare_triangles(
        mesh, (0.0, 0.0, 0.0), cam, 320, 240, shade="lambert", cull=True, smooth=False
    )
    assert stats_s["kept"] == 1 and stats_f["kept"] == 1
    got = (tris_smooth[0][3], tris_smooth[0][7], tris_smooth[0][11])
    want = _expected_intensities(
        mesh, face, (0.0, 0.0, 0.0), cam, "lambert", (normals[0], normals[1], normals[2])
    )
    flat = (tris_flat[0][3], tris_flat[0][7], tris_flat[0][11])
    for g, w in zip(got, want):
        assert abs(g - w) < 1e-12, "Procedural per-vertex smooth shading mismatch: got=%r want=%r" % (got, want)
    assert got != flat, "Procedural smooth shading must differ from flat shading"


def test_mixed_obj_faces_vn_none_mismatched_normals_falls_back_to_face_normal():
    """Mixed OBJ (some faces with vn, some without, len(normals)!=len(verts)) must not crash or misread."""
    cam = Camera3D(cx=0.0, cy=0.0, cz=-5.0, yaw=0.0, pitch=0.0)

    # Case A: len(normals) < len(verts) (6 verts, 1 normal).
    # Face 1 uses explicit vn=(0,0,0) pointing up; Face 2 has no vn (verts 3,4,5 >= len(normals)=1).
    mixed_fewer = (
        "# normalize: scale=1.0 center=(0.0,0.0,0.0)\n"
        "v -1.5 -0.5 0.0\n"
        "v -0.5 -0.5 0.0\n"
        "v -1.0  0.5 0.0\n"
        "v  0.5 -0.5 0.0\n"
        "v  1.5 -0.5 0.0\n"
        "v  1.0  0.5 0.0\n"
        "vn 0.0 1.0 0.0\n"
        "f 1//1 3//1 2//1\n"
        "f 4 6 5\n"
    ).splitlines()
    m_fewer = mesh3d.parse_obj(mixed_fewer, path="mixed_fewer.obj", axis_swap=False, recenter=False)
    assert len(m_fewer.verts) == 6 and len(m_fewer.normals) == 1
    assert m_fewer.faces[0].vn == (0, 0, 0)
    assert m_fewer.faces[1].vn is None

    tris_a, stats_a = mesh3d.prepare_triangles(
        m_fewer, (0.0, 0.0, 0.0), cam, 320, 240, shade="lambert", cull=True, smooth=True
    )
    assert stats_a["kept"] == 2
    want_a0 = _expected_intensities(
        m_fewer, m_fewer.faces[0], (0.0, 0.0, 0.0), cam, "lambert",
        (m_fewer.normals[0], m_fewer.normals[0], m_fewer.normals[0]),
    )
    want_a1 = _expected_intensities(
        m_fewer, m_fewer.faces[1], (0.0, 0.0, 0.0), cam, "lambert", None
    )
    got_a0 = (tris_a[0][3], tris_a[0][7], tris_a[0][11])
    got_a1 = (tris_a[1][3], tris_a[1][7], tris_a[1][11])
    for g, w in zip(got_a0, want_a0):
        assert abs(g - w) < 1e-12
    for g, w in zip(got_a1, want_a1):
        assert abs(g - w) < 1e-12

    # Case B: len(normals) > len(verts) (3 verts, 4 normals), face has vn=None.
    # Must fall back to face.normal rather than silently misreading normals[0..2].
    verts = [(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)]
    normals_more = [
        (0.0, -1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 1.0, 0.0),
    ]
    face_b = mesh3d.Face(idx=(0, 1, 2), color=(200, 200, 200), normal=(0.0, 0.0, -1.0), vn=None)
    m_more = mesh3d.MeshGeometry(verts=verts, faces=[face_b], normals=normals_more, name="mixed_more")
    tris_b, stats_b = mesh3d.prepare_triangles(
        m_more, (0.0, 0.0, 0.0), cam, 320, 240, shade="lambert", cull=True, smooth=True
    )
    assert stats_b["kept"] == 1
    got_b = (tris_b[0][3], tris_b[0][7], tris_b[0][11])
    want_b = _expected_intensities(m_more, face_b, (0.0, 0.0, 0.0), cam, "lambert", None)
    wrong_b = _expected_intensities(
        m_more, face_b, (0.0, 0.0, 0.0), cam, "lambert", (normals_more[0], normals_more[1], normals_more[2])
    )
    for g, w in zip(got_b, want_b):
        assert abs(g - w) < 1e-12, "Mismatched vn=None must fall back to face.normal, got=%r want=%r" % (got_b, want_b)
    assert abs(got_b[0] - wrong_b[0]) > 0.2


def test_invalid_face_vn_and_rotated_and_near_clip_smooth():
    """Out-of-bounds/negative face.vn falls back safely; rotated() and near-plane clip work with face.vn."""
    cam = Camera3D(cx=0.0, cy=0.0, cz=-5.0, yaw=0.0, pitch=0.0)
    verts = [(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)]
    normals = [(0.0, -1.0, 0.0), (0.0, -1.0, 0.0), (0.0, 1.0, 0.0)]

    # Negative or out-of-range face.vn must NOT wrap via Python negative indexing or fall through to idx.
    for bad_vn in ((-1, 0, 1), (0, 3, 1), (0, 1)):
        face_bad = mesh3d.Face(idx=(0, 1, 2), color=(200, 200, 200), normal=(0.0, 0.0, -1.0), vn=bad_vn)
        mesh_bad = mesh3d.MeshGeometry(verts=verts, faces=[face_bad], normals=normals, name="bad_vn")
        tris_bad, stats_bad = mesh3d.prepare_triangles(
            mesh_bad, (0.0, 0.0, 0.0), cam, 320, 240, shade="lambert", cull=True, smooth=True
        )
        assert stats_bad["kept"] == 1
        got_bad = (tris_bad[0][3], tris_bad[0][7], tris_bad[0][11])
        want_bad = _expected_intensities(mesh_bad, face_bad, (0.0, 0.0, 0.0), cam, "lambert", None)
        for g, w in zip(got_bad, want_bad):
            assert abs(g - w) < 1e-12, "Invalid face.vn=%r must fall back to face.normal" % (bad_vn,)

    # rotated() preserves face.vn and rotates mesh.normals on 8-vert / 6-normal OBJ cube.
    cube = mesh3d.parse_obj(CUBE_OBJ_LINES, path="cube.obj", axis_swap=False, recenter=False)
    rot_cube = cube.rotated((0.4, 0.7, 0.2))
    cam_corner = Camera3D.look_at((3.0, 3.0, 5.0), (0.0, 0.0, 0.0))
    tris_rot, stats_rot = mesh3d.prepare_triangles(
        rot_cube, (0.0, 0.0, 0.0), cam_corner, 320, 240, shade="lambert", cull=True, smooth=True
    )
    assert stats_rot["kept"] > 0
    vis_rot = [
        f for f in rot_cube.faces
        if (f.normal[0] * _expected_view(rot_cube, f, (0.0, 0.0, 0.0), cam_corner)[0]
            + f.normal[1] * _expected_view(rot_cube, f, (0.0, 0.0, 0.0), cam_corner)[1]
            + f.normal[2] * _expected_view(rot_cube, f, (0.0, 0.0, 0.0), cam_corner)[2]) > 0.0
    ]
    assert len(vis_rot) == len(tris_rot)
    for face, tri in zip(vis_rot, tris_rot):
        vn_vecs = tuple(rot_cube.normals[k] for k in face.vn)
        want = _expected_intensities(rot_cube, face, (0.0, 0.0, 0.0), cam_corner, "lambert", vn_vecs)
        for g, w in zip((tri[3], tri[7], tri[11]), want):
            assert abs(g - w) < 1e-12

    # Near-plane clipping on a face using vertex indices >= len(normals) and valid face.vn.
    clip_verts = [
        (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),  # dummy padding verts 0..2
        (-1.0, -1.0, -0.2), (1.0, -1.0, 1.5), (0.0, 1.0, 1.5),
    ]
    clip_face = mesh3d.Face(
        idx=(3, 4, 5), color=(180, 180, 180), normal=(0.0, 0.0, -1.0), vn=(0, 0, 0)
    )
    clip_mesh = mesh3d.MeshGeometry(
        verts=clip_verts, faces=[clip_face], normals=[(0.0, 1.0, 0.0)], name="clip_vn"
    )
    cam_near = Camera3D(cx=0.0, cy=0.0, cz=0.0, yaw=0.0, pitch=0.0)
    tris_clip, stats_clip = mesh3d.prepare_triangles(
        clip_mesh, (0.0, 0.0, 0.0), cam_near, 320, 240, shade="lambert", cull=False, smooth=True
    )
    assert stats_clip["clipped"] == 1
    assert stats_clip["kept"] == 2  # triangle clipped to quad -> 2 fan tris
    want_clip = _expected_intensities(
        clip_mesh, clip_face, (0.0, 0.0, 0.0), cam_near, "lambert",
        (clip_mesh.normals[0], clip_mesh.normals[0], clip_mesh.normals[0]),
    )
    for tri in tris_clip:
        for g in (tri[3], tri[7], tri[11]):
            assert abs(g - want_clip[0]) < 1e-12


ALL_TESTS = (
    test_obj_cube_fewer_normals_than_verts_smooth_matches_ground_truth,
    test_explicit_face_vn_used_when_normals_ge_verts,
    test_procedural_mesh_vn_none_per_vertex_normals_compat,
    test_mixed_obj_faces_vn_none_mismatched_normals_falls_back_to_face_normal,
    test_invalid_face_vn_and_rotated_and_near_clip_smooth,
)


def _run_all_tests(report_json: bool = False) -> int:
    passed = []
    failed = {}
    for fn in ALL_TESTS:
        try:
            fn()
            passed.append(fn.__name__)
        except Exception as exc:
            failed[fn.__name__] = "%s: %s" % (type(exc).__name__, exc)
    if report_json:
        print(json.dumps({"passed": passed, "failed": failed}, ensure_ascii=False))
    else:
        for name in passed:
            print("PASS", name)
        for name, msg in failed.items():
            print("FAIL", name, "->", msg)
        print("Summary: %d passed, %d failed" % (len(passed), len(failed)))
    return 0 if not failed else 1


MUTATION_CASES = (
    (
        "revert_to_vertex_idx",
        "        vn_idx = None\n"
        "        if normals is not None:\n"
        "            fvn = face.vn\n"
        "            if fvn is not None:\n"
        "                if (len(fvn) == 3\n"
        "                        and 0 <= fvn[0] < n_normals\n"
        "                        and 0 <= fvn[1] < n_normals\n"
        "                        and 0 <= fvn[2] < n_normals):\n"
        "                    vn_idx = fvn\n"
        "            elif per_vert_normals and (\n"
        "                    0 <= i0 < n_normals\n"
        "                    and 0 <= i1 < n_normals\n"
        "                    and 0 <= i2 < n_normals):\n"
        "                vn_idx = (i0, i1, i2)",
        "        vn_idx = (i0, i1, i2) if normals is not None else None",
        frozenset({
            "test_obj_cube_fewer_normals_than_verts_smooth_matches_ground_truth",
            "test_explicit_face_vn_used_when_normals_ge_verts",
            "test_mixed_obj_faces_vn_none_mismatched_normals_falls_back_to_face_normal",
            "test_invalid_face_vn_and_rotated_and_near_clip_smooth",
        }),
        frozenset({
            "test_procedural_mesh_vn_none_per_vertex_normals_compat",
        }),
    ),
    (
        "drop_per_vert_compat",
        "            elif per_vert_normals and (\n"
        "                    0 <= i0 < n_normals\n"
        "                    and 0 <= i1 < n_normals\n"
        "                    and 0 <= i2 < n_normals):\n"
        "                vn_idx = (i0, i1, i2)",
        "            elif False:\n"
        "                vn_idx = (i0, i1, i2)",
        frozenset({
            "test_procedural_mesh_vn_none_per_vertex_normals_compat",
        }),
        frozenset({
            "test_obj_cube_fewer_normals_than_verts_smooth_matches_ground_truth",
            "test_explicit_face_vn_used_when_normals_ge_verts",
            "test_mixed_obj_faces_vn_none_mismatched_normals_falls_back_to_face_normal",
            "test_invalid_face_vn_and_rotated_and_near_clip_smooth",
        }),
    ),
)


def _mutate_and_run() -> int:
    """Apply targeted mutants in an isolated temp copy and verify exact RED/GREEN sets."""
    all_ok = True
    for label, old_snippet, new_snippet, expect_red, expect_green in MUTATION_CASES:
        tmp = Path(tempfile.mkdtemp(prefix="mesh3d_vn_mutate_"))
        try:
            dst = tmp / "repo"
            dst.mkdir()
            shutil.copytree(
                REPO / "lingshu", dst / "lingshu",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
            (dst / "tests").mkdir()
            shutil.copy2(
                Path(__file__).resolve(),
                dst / "tests" / "test_mesh3d_smooth_vn_index.py",
            )
            target = dst / "lingshu" / "world" / "mesh3d.py"
            text = target.read_text(encoding="utf-8")
            norm_text = text.replace("\r\n", "\n")
            if norm_text.count(old_snippet) != 1:
                print("[mutate:%s] anchor count != 1 (%d)" % (label, norm_text.count(old_snippet)))
                all_ok = False
                continue
            target.write_text(norm_text.replace(old_snippet, new_snippet), encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, "-X", "utf8",
                 str(dst / "tests" / "test_mesh3d_smooth_vn_index.py"), "--report"],
                cwd=str(dst), capture_output=True, text=True,
                encoding="utf-8", errors="replace",
            )
            payload = json.loads(proc.stdout.strip().splitlines()[-1])
            got_red = frozenset(payload["failed"].keys())
            got_green = frozenset(payload["passed"])
            ok = (got_red == expect_red and got_green == expect_green)
            print("[mutate:%s] rc=%d match=%s red=%s green=%s" % (
                label, proc.returncode, ok, sorted(got_red), sorted(got_green),
            ))
            if not ok:
                print("  expect_red=%s expect_green=%s" % (sorted(expect_red), sorted(expect_green)))
                print("  failed_detail=%r" % (payload["failed"],))
                all_ok = False
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    if "--mutate" in sys.argv:
        sys.exit(_mutate_and_run())
    sys.exit(_run_all_tests(report_json="--report" in sys.argv))
