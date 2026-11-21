"""SAT must use face normals for the supported non-orthogonal box frames."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lingshu.world import physics as P


def _separated_skew_boxes():
    a = {
        "center": (0.5, 0.5, 0.5),
        "half": (0.5, 0.5, 0.5),
        "rot": (
            (-0.554749, 0.360100, -0.750054),
            (-0.407590, -0.897262, 0.169682),
            (-0.506776, 0.320162, 0.800422),
        ),
    }
    b = {
        "center": (-0.861919, 0.556873, -0.312856),
        "half": (0.5, 0.5, 0.5),
        "rot": (
            (-0.515958, -0.760046, -0.395117),
            (-0.261757, 0.593418, -0.761143),
            (-0.994881, 0.072175, -0.070733),
        ),
    }
    return a, b


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _unit(vector):
    length = sum(x * x for x in vector) ** 0.5
    return tuple(x / length for x in vector)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _projection_radius(box, axis):
    edges = tuple(_unit(row) for row in box["rot"])
    return sum(half * abs(_dot(edge, axis))
               for half, edge in zip(box["half"], edges))


def test_supported_skew_boxes_use_face_normal_separating_axis():
    a, b = _separated_skew_boxes()
    face_normal = _unit(_cross(a["rot"][1], a["rot"][2]))
    delta = tuple(b["center"][i] - a["center"][i] for i in range(3))
    gap = (abs(_dot(delta, face_normal))
           - _projection_radius(a, face_normal)
           - _projection_radius(b, face_normal))
    assert gap == pytest.approx(0.0321480993, abs=1e-9)

    assert P.obb_overlap(a, b) is False, "SAT must test each parallelepiped face normal"
    assert P.obb_mtv(a, b) is None


def test_non_spanning_rot_rows_are_rejected():
    singular = {
        "center": (0.0, 0.0, 0.0),
        "half": (1.0, 1.0, 1.0),
        "rot": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0)),
    }
    with pytest.raises(P.PhysicsError, match="线性无关"):
        P.obb_of(aabb={"center": singular["center"], "half": singular["half"]},
                 rot=singular["rot"])
