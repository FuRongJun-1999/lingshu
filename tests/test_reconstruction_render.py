"""Independent analytic checks of the static mesh depth consumer."""
import math
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from lingshu.world.reconstruction_camera import ReconstructionCamera
from lingshu.world.reconstruction_render import render_reconstruction
from lingshu.world.scene_model import WorldModel


_spec = importlib.util.spec_from_file_location(
    "_reconstruction_native_fixture", Path(__file__).resolve().with_name("test_reconstruction_artifacts.py"))
_native = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_native)
write_ply = _native.write_ply


def reconstruction(tmp_path, depths=(2, 4, 8), *, scale=1.0, translation=(0, 0, 0), extra=False):
    camera = ReconstructionCamera("view.jpg", 7, 7, "original.jpg", 0, "PINHOLE",
                                  ((2, 1, 3), (0, 3, 2), (0, 0, 1)),
                                  ((1, 0, 0, translation[0]), (0, 1, 0, translation[1]),
                                   (0, 0, 1, translation[2])), scale)
    vertices = [camera.unproject(u, v, depth) for (u, v), depth in zip(((0, 0), (6, 0), (0, 6)), depths)]
    faces = [(0, 1, 2)]
    if extra:
        vertices.extend(camera.unproject(u, v, 1) for u, v in ((0, 0), (6, 0), (0, 6)))
        faces.append((3, 4, 5))
    path = tmp_path / "mesh.ply"
    write_ply(path, vertices, faces, binary=True)
    return SimpleNamespace(cameras={camera.name: camera}, artifacts={"mesh": path},
                           units="meters" if scale != 1 else "SfM relative units")


def test_skew_anisotropic_intrinsics_and_perspective_depth(tmp_path):
    recon = reconstruction(tmp_path)
    result = render_reconstruction(recon, "view.jpg")
    # Array pixel(1,1) samples continuous(1.5,1.5): weights 1/2,1/4,1/4.
    expected = 1 / (0.5/2 + 0.25/4 + 0.25/8)
    assert result.depth[1*7+1] == pytest.approx(expected, rel=1e-6)
    assert result.depth[1*7+1] != pytest.approx(0.5*2+0.25*4+0.25*8)
    assert result.mask[1*7+1] == 1 and result.mask[5*7+5] == 0
    assert result.depth[5*7+5] == 0
    assert len(result.rgb) == 7*7*3 and len(result.mask) == len(result.depth) == 49


def test_zbuffer_selects_nearer_surface_in_both_face_orders(tmp_path):
    recon = reconstruction(tmp_path, extra=True)
    first = render_reconstruction(recon, "view.jpg")
    from lingshu.world.reconstruction_artifacts import read_ply
    mesh = read_ply(recon.artifacts["mesh"], mesh=True)
    write_ply(recon.artifacts["mesh"], mesh.vertices, list(reversed(mesh.faces)))
    second = render_reconstruction(recon, "view.jpg")
    assert first.depth == second.depth and first.mask == second.mask
    assert first.depth[8] == pytest.approx(1)


@pytest.mark.parametrize("depths,near,far", [((0.25, 2, 2), 1, 3), ((2, 8, 2), 1, 3), ((-1, 2, 2), 0.5, 4)])
def test_triangles_are_really_clipped_at_near_far(tmp_path, depths, near, far):
    if depths[0] < 0:
        recon = reconstruction(tmp_path, (1, 2, 2))
        from lingshu.world.reconstruction_artifacts import read_ply
        mesh = read_ply(recon.artifacts["mesh"], mesh=True)
        vertices = list(mesh.vertices)
        vertices[0] = (vertices[0][0], vertices[0][1], -1)
        write_ply(recon.artifacts["mesh"], vertices, mesh.faces)
    else:
        recon = reconstruction(tmp_path, depths)
    result = render_reconstruction(recon, "view.jpg", near=near, far=far)
    assert result.stats["visible_pixels"] > 0 and result.stats["clipped_faces"] == 1
    assert all(math.isfinite(z) and (z == 0 or near-1e-6 <= z <= far+1e-6) for z in result.depth)


def test_fully_outside_surfaces_are_empty(tmp_path):
    recon = reconstruction(tmp_path, (2, 2, 2))
    result = render_reconstruction(recon, "view.jpg", near=3, far=4)
    assert result.stats["visible_pixels"] == 0 and not any(result.depth)


def test_metric_mesh_is_not_scaled_twice_and_translation_is_scaled(tmp_path):
    recon = reconstruction(tmp_path, (0.5, 0.5, 0.5), scale=0.25, translation=(-4, 2, 1))
    result = render_reconstruction(recon, "view.jpg")
    assert result.depth[8] == pytest.approx(0.5) and result.units == "meters"
    assert result.near == pytest.approx(2.5e-7)


@pytest.mark.parametrize("near,far", [(0, None), (-1, None), (True, None), (float("nan"), None), (1, 1), (1, 0)])
def test_invalid_clipping_range_rejected(tmp_path, near, far):
    with pytest.raises(ValueError):
        render_reconstruction(reconstruction(tmp_path), "view.jpg", near=near, far=far)


def test_world_renderer_requires_attachment_and_does_not_build_entities(tmp_path):
    world = WorldModel().add_entity("existing", "tree", (1, 0, 2))
    with pytest.raises(ValueError, match="no reconstruction"):
        world.render_reconstruction("view.jpg")
    recon = reconstruction(tmp_path)
    world.reconstruction = recon
    recon.render = lambda view, **options: render_reconstruction(recon, view, **options)
    held = world.scene
    result = world.render_reconstruction("view.jpg")
    assert result.stats["visible_pixels"] > 0
    assert list(world.entities) == ["existing"] and world.scene is held


def test_render_executes_without_site_packages(tmp_path):
    reconstruction(tmp_path, (2, 2, 2))
    source = """
import sys
from pathlib import Path
from types import SimpleNamespace
from lingshu.world.reconstruction_camera import ReconstructionCamera
from lingshu.world.reconstruction_render import render_reconstruction
camera = ReconstructionCamera('v',7,7,'source',0,'PINHOLE',((2,1,3),(0,3,2),(0,0,1)),
                              ((1,0,0,0),(0,1,0,0),(0,0,1,0)))
recon = SimpleNamespace(cameras={'v':camera}, artifacts={'mesh':Path(sys.argv[1])}, units='relative')
result = render_reconstruction(recon, 'v')
assert result.stats['visible_pixels'] > 0
assert not {'numpy','cv2','PIL','open3d','torch','pycolmap'} & set(sys.modules)
"""
    result = subprocess.run([sys.executable, "-S", "-c", source, str(tmp_path / "mesh.ply")],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True,
                            text=True, encoding="utf-8", errors="replace")
    assert result.returncode == 0, result.stderr


def test_full_k_half_pixel_render_matches_independent_mesh_raycast(tmp_path):
    np = pytest.importorskip("numpy")
    o3d = pytest.importorskip("open3d")
    # Explicit triangle coordinates and K: no self-project/unproject round trip.
    vertices = np.array([[-1.4, -0.8, 1.57], [1.6, -0.8, 2.32], [-1.4, 1.2, 1.77]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.uint32)
    path = tmp_path / "oracle.ply"
    write_ply(path, vertices.tolist(), faces.tolist(), binary=True)
    width, height = 64, 48
    fx, fy, skew, cx, cy = 45.0, 52.0, 7.0, 29.0, 19.0
    camera = ReconstructionCamera("v", width, height, "explicit", 0, "PINHOLE",
                                  ((fx, skew, cx), (0, fy, cy), (0, 0, 1)),
                                  ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)))
    recon = SimpleNamespace(cameras={"v": camera}, artifacts={"mesh": path}, units="test")
    result = render_reconstruction(recon, "v")
    y, x = np.indices((height, width), dtype=np.float32)
    yn = (y+0.5-cy)/fy
    xn = (x+0.5-cx-skew*yn)/fx
    rays = np.stack((np.zeros_like(x), np.zeros_like(x), np.zeros_like(x), xn, yn, np.ones_like(x)), axis=-1)
    mesh = o3d.t.geometry.TriangleMesh(o3d.core.Tensor(vertices), o3d.core.Tensor(faces))
    oracle = o3d.t.geometry.RaycastingScene()
    oracle.add_triangles(mesh)
    truth = oracle.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    mask = np.isfinite(truth)
    actual_mask = np.frombuffer(result.mask, dtype=np.uint8).reshape(height, width).astype(bool)
    assert np.array_equal(actual_mask, mask)
    depth = np.asarray(result.depth).reshape(height, width)
    assert np.max(np.abs(depth[mask]-truth[mask])) < 1e-5
