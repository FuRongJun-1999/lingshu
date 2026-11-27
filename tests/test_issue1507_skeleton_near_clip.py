"""#1507 LS-05: retain the visible portion of bones crossing the near plane."""
import pytest

pytest.importorskip("PIL")

from PIL import ImageDraw

from lingshu.world.skeleton3d import Skeleton3D
from lingshu.world.world3d import Camera3D


def _cameras():
    return [
        Camera3D(),
        Camera3D.look_at((3, 4, 5), (0, 0, 0), roll=0.45),
        Camera3D.from_look_at((3, 4, 5), (0, 0, 0), up=(0, 1, 0)),
    ]


def _add_bone(sk, name, a, b):
    sk.add_joint(name + "_a", a, radius=0)
    sk.add_joint(name + "_b", tuple(b[i] - a[i] for i in range(3)),
                 parent=name + "_a", radius=0)


@pytest.mark.parametrize("cam", _cameras())
@pytest.mark.parametrize("reverse", [False, True])
def test_crossing_bone_keeps_visible_pixels(cam, reverse):
    a = cam.unproject(300, 200, 0.05, 400, 400)
    b = cam.unproject(200, 200, 1.0, 400, 400)
    sk = Skeleton3D(center=(0, 0, 0))
    _add_bone(sk, "crossing", *((b, a) if reverse else (a, b)))

    img = sk.render(400, 400, camera=cam)

    # The clipped line spans x=200..247.4; this pixel is clear of the joint.
    assert any(img.getpixel((225, y)) == sk._bone_color for y in (199, 200, 201))
    assert img.getpixel((275, 200)) == (24, 28, 46)


@pytest.mark.parametrize("depths", [(0.05, 0.09), (-1.0, -0.5), (0.1, 0.1)])
def test_fully_hidden_bone_is_not_drawn(depths):
    cam = Camera3D()
    sk = Skeleton3D(center=(0, 0, 0))
    a = cam.unproject(100, 200, depths[0], 400, 400)
    b = cam.unproject(300, 200, depths[1], 400, 400)
    _add_bone(sk, "hidden", a, b)

    assert sk.render(400, 400, camera=cam).getcolors(160000) == [
        (160000, (24, 28, 46))]


def test_visible_bone_and_joint_projection_are_preserved(monkeypatch):
    cam = _cameras()[1]
    sk = Skeleton3D(center=(0, 0, 0))
    a = cam.unproject(100, 200, 0.2, 400, 400)
    b = cam.unproject(300, 200, 1.0, 400, 400)
    _add_bone(sk, "visible", a, b)
    calls = []
    line = ImageDraw.ImageDraw.line

    def capture(self, xy, *args, **kwargs):
        calls.append(xy)
        return line(self, xy, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "line", capture)
    img = sk.render(400, 400, camera=cam)
    expected = [cam.project(sk.world_pos(n), 400, 400) for n in sk.joints]
    assert len(calls) == 1
    assert calls[0][0] == pytest.approx(expected[0])
    assert calls[0][1] == pytest.approx(expected[1])
    assert img.getpixel((100, 200)) == sk._joint_color
    assert img.getpixel((300, 200)) == sk._joint_color


def test_draw_order_uses_depth_of_clipped_segment(monkeypatch):
    cam = Camera3D()
    sk = Skeleton3D(center=(0, 0, 0))
    # Original mean depth is -4.5, but visible mean is (0.1+1)/2 = 0.55.
    _add_bone(sk, "crossing", cam.unproject(300, 150, -10, 400, 400),
              cam.unproject(200, 150, 1, 400, 400))
    _add_bone(sk, "nearer", cam.unproject(100, 250, 0.4, 400, 400),
              cam.unproject(300, 250, 0.4, 400, 400))
    calls = []
    monkeypatch.setattr(ImageDraw.ImageDraw, "line",
                        lambda self, xy, *args, **kwargs: calls.append(xy))

    sk.render(400, 400, camera=cam)

    assert len(calls) == 2
    assert calls[0][1][1] == pytest.approx(150)
    assert calls[1][0][1] == pytest.approx(250)
