"""Exact pinhole cameras from an undistorted CPU reconstruction.

Native pixels use the COLMAP/OpenCV convention (right, down, forward).
``Camera3D`` has one focal length and a centered, vertically inverted image
plane.  The explicit virtual-pixel conversion below connects the two without
discarding anisotropic focal lengths, an off-center principal point or skew.
Virtual pixels are not coordinates in the original photograph.

World coordinates remain the reconstruction's local SfM coordinates.  Scaling
them does not establish a gravity direction or align them to a simulation map.
Only the caller can establish that a scale factor actually converts to meters.
This module uses only the standard library and the existing Camera3D class.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .world3d import Camera3D


def _number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _vector(value, size: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{label} must contain {size} numbers")
    return tuple(_number(item, f"{label}[{i}]") for i, item in enumerate(value))


def _matrix(value, rows: int, cols: int, label: str) -> tuple:
    if not isinstance(value, (list, tuple)) or len(value) != rows:
        raise ValueError(f"{label} must have shape {rows}x{cols}")
    return tuple(_vector(row, cols, f"{label}[{i}]") for i, row in enumerate(value))


@dataclass(frozen=True, slots=True)
class ReconstructionCamera:
    """An immutable native camera and its explicit reconstruction-unit scale.

    ``cam_from_world`` retains the original SfM ``[R|t]`` matrix.  Methods accept
    and return world points and camera-Z depths in the scaled coordinates, so
    their transform is ``camera_point = R @ world_point + coordinate_scale*t``.
    Pixel coordinates always refer to the exported undistorted image size.
    """

    name: str
    width: int
    height: int
    source: str
    source_index: int
    model: str
    intrinsics: tuple
    cam_from_world: tuple
    coordinate_scale: float = 1.0

    def __post_init__(self) -> None:
        for label in ("name", "source"):
            value = getattr(self, label)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label} must be a nonempty string")
        for label in ("width", "height", "source_index"):
            value = getattr(self, label)
            minimum = 0 if label == "source_index" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{label} must be an integer >= {minimum}")
        if self.model not in ("PINHOLE", "SIMPLE_PINHOLE"):
            raise ValueError("camera model must be undistorted PINHOLE or SIMPLE_PINHOLE")
        scale = _number(self.coordinate_scale, "coordinate_scale")
        if scale <= 0:
            raise ValueError("coordinate_scale must be positive")
        intrinsic = _matrix(self.intrinsics, 3, 3, "intrinsics")
        extrinsic = _matrix(self.cam_from_world, 3, 4, "cam_from_world")
        if intrinsic[0][0] <= 0 or intrinsic[1][1] <= 0:
            raise ValueError("intrinsics fx and fy must be positive")
        if (intrinsic[1][0] != 0 or intrinsic[2] != (0.0, 0.0, 1.0)):
            raise ValueError("intrinsics must have the pinhole upper-triangular form")
        rotation = tuple(row[:3] for row in extrinsic)
        for i in range(3):
            for j in range(3):
                dot = sum(rotation[i][k] * rotation[j][k] for k in range(3))
                if not math.isclose(dot, float(i == j), abs_tol=1e-6, rel_tol=0):
                    raise ValueError("camera rotation must be orthonormal")
        a, b, c = rotation
        determinant = (a[0] * (b[1]*c[2] - b[2]*c[1])
                       - a[1] * (b[0]*c[2] - b[2]*c[0])
                       + a[2] * (b[0]*c[1] - b[1]*c[0]))
        if not math.isclose(determinant, 1.0, abs_tol=1e-6, rel_tol=0):
            raise ValueError("camera rotation must be proper (determinant +1)")
        # Values too extreme to represent an existing Camera3D must fail here,
        # rather than generating a zero- or 180-degree virtual camera later.
        fov = math.degrees(2.0 * math.atan((self.width / 2.0) / intrinsic[0][0]))
        if not 0 < fov < 180:
            raise ValueError("fx and width must define a representable field of view")
        object.__setattr__(self, "intrinsics", intrinsic)
        object.__setattr__(self, "cam_from_world", extrinsic)
        object.__setattr__(self, "coordinate_scale", scale)
        _vector(self._center(), 3, "scaled camera center")

    @classmethod
    def from_pose(cls, name: str, pose: dict, *, coordinate_scale: float = 1.0):
        """Read one ``poses.json`` entry; missing fields never get fake defaults."""
        if not isinstance(pose, dict):
            raise ValueError("pose must be a dictionary")
        keys = ("width", "height", "source", "source_index", "model",
                "intrinsics", "cam_from_world")
        missing = [key for key in keys if key not in pose]
        if missing:
            raise ValueError("pose is missing fields: " + ", ".join(missing))
        return cls(name=name, coordinate_scale=coordinate_scale,
                   **{key: pose[key] for key in keys})

    def _center(self) -> tuple[float, float, float]:
        matrix = self.cam_from_world
        return tuple(-self.coordinate_scale * sum(matrix[i][j] * matrix[i][3]
                                                  for i in range(3))
                     for j in range(3))

    def _camera_point(self, point) -> tuple[float, float, float]:
        p = _vector(point, 3, "point")
        return _vector(tuple(sum(row[i] * p[i] for i in range(3))
                             + self.coordinate_scale * row[3]
                             for row in self.cam_from_world), 3, "camera point")

    def camera_depth(self, point) -> float:
        """Return camera-Z depth in the scaled reconstruction coordinates."""
        return self._camera_point(point)[2]

    def project(self, point) -> tuple[float, float] | None:
        """Project to native undistorted pixels; nonpositive depth returns None."""
        x, y, z = self._camera_point(point)
        if z <= 0:
            return None
        k = self.intrinsics
        return _vector((k[0][0] * (x / z) + k[0][1] * (y / z) + k[0][2],
                        k[1][1] * (y / z) + k[1][2]), 2, "projected pixel")

    def unproject(self, u: float, v: float, depth: float) -> tuple[float, float, float]:
        """Recover a scaled world point from native pixels and positive camera-Z."""
        u, v = _vector((u, v), 2, "pixel")
        depth = _number(depth, "depth")
        if depth <= 0:
            raise ValueError("depth must be positive")
        k = self.intrinsics
        yn = (v - k[1][2]) / k[1][1]
        xn = (u - k[0][2] - k[0][1] * yn) / k[0][0]
        camera_point = (xn * depth, yn * depth, depth)
        centered = tuple(camera_point[i] - self.coordinate_scale * self.cam_from_world[i][3]
                         for i in range(3))
        world = tuple(sum(self.cam_from_world[i][j] * centered[i] for i in range(3))
                      for j in range(3))
        return _vector(world, 3, "unprojected point")

    def as_camera3d(self) -> Camera3D:
        """Return the existing camera for use exclusively with virtual pixels.

        Its basis is R, without a reflection.  Native down-positive image Y is
        flipped by ``to_virtual_pixel``; both operations are required together.
        """
        x, y, z = self._center()
        fov = math.degrees(2.0 * math.atan((self.width / 2.0) / self.intrinsics[0][0]))
        return Camera3D(fov_deg=fov, cx=x, cy=y, cz=z,
                        basis=tuple(row[:3] for row in self.cam_from_world))

    def to_virtual_pixel(self, u: float, v: float) -> tuple[float, float]:
        """Map native pixels to the centered single-focal Camera3D image plane."""
        u, v = _vector((u, v), 2, "pixel")
        k = self.intrinsics
        yn = (v - k[1][2]) / k[1][1]
        xn = (u - k[0][2] - k[0][1] * yn) / k[0][0]
        return _vector((self.width / 2.0 + k[0][0] * xn,
                        self.height / 2.0 - k[0][0] * yn), 2, "virtual pixel")

    def to_native_pixel(self, u: float, v: float) -> tuple[float, float]:
        """Reverse ``to_virtual_pixel``; virtual pixels are never source evidence."""
        u, v = _vector((u, v), 2, "virtual pixel")
        k = self.intrinsics
        xn = (u - self.width / 2.0) / k[0][0]
        yn = (self.height / 2.0 - v) / k[0][0]
        return _vector((k[0][0] * xn + k[0][1] * yn + k[0][2],
                        k[1][1] * yn + k[1][2]), 2, "native pixel")
