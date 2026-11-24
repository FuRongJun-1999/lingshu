"""Validate and unproject bounded batches of pixel/forward-depth samples."""

import math
from numbers import Real

from .world3d import Camera3D


def _finite_real(value, where):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{where} must be a real number (not bool)")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{where} must be a finite float") from exc
    if not math.isfinite(result):
        raise ValueError(f"{where} must be finite")
    return result


def _validate_camera(camera):
    if not isinstance(camera, Camera3D):
        raise TypeError("camera must be a Camera3D instance")
    for field in ("cx", "cy", "cz", "yaw", "pitch", "roll", "fov_deg"):
        value = _finite_real(getattr(camera, field), f"camera.{field}")
        if field == "fov_deg" and not 0 < value < 180:
            raise ValueError("camera.fov_deg must be between 0 and 180 (exclusive)")
    basis = camera.basis
    if basis is None:
        return
    if not isinstance(basis, (list, tuple)) or len(basis) != 3:
        raise TypeError("camera.basis must be a 3x3 list or tuple")
    rows = []
    for index, row in enumerate(basis):
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise TypeError("camera.basis must have three three-item rows")
        rows.append(tuple(_finite_real(value, f"camera.basis[{index}][{axis}]")
                          for axis, value in enumerate(row)))
    tolerance = 1e-9
    for row in rows:
        if not math.isclose(math.hypot(*row), 1.0, rel_tol=0, abs_tol=tolerance):
            raise ValueError("camera.basis rows must have unit length")
    for first, second in ((0, 1), (0, 2), (1, 2)):
        dot = sum(a * b for a, b in zip(rows[first], rows[second]))
        if not math.isclose(dot, 0.0, rel_tol=0, abs_tol=tolerance):
            raise ValueError("camera.basis rows must be orthogonal")
    (a, b, c), (d, e, f), (g, h, i) = rows
    determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if not math.isclose(determinant, 1.0, rel_tol=0, abs_tol=tolerance):
        raise ValueError("camera.basis must be right-handed (determinant +1)")


def unproject_depth_samples(samples, camera, screen_size, *, max_samples=4096) -> dict:
    """Return JSON-safe world points in input order, without changing inputs.

    ``samples`` is a list/tuple of ``(pixel_x, pixel_y, forward_depth)``
    triples. Pixels must lie inside the positive integer ``(width, height)``
    screen; forward depth must exceed Camera3D.project's 0.1 cutoff.
    Camera pose/FOV must be finite; explicit bases must be right-handed
    orthonormal 3x3 matrices (absolute tolerance 1e-9). Screen dimensions
    must be representable as finite floats. All inputs are validated before
    the first camera.unproject call.
    Type errors raise TypeError; invalid ranges/nonfinite numbers raise
    ValueError, including any nonfinite world coordinate from the camera.
    No projection formulas or persistence are implemented here.
    """
    if isinstance(max_samples, bool) or not isinstance(max_samples, int):
        raise TypeError("max_samples must be an integer (not bool)")
    if not 1 <= max_samples <= 4096:
        raise ValueError("max_samples must be between 1 and 4096")
    if not isinstance(samples, (list, tuple)):
        raise TypeError("samples must be a finite list or tuple")
    if len(samples) > max_samples:
        raise ValueError("sample count exceeds max_samples")
    if not isinstance(screen_size, (list, tuple)) or len(screen_size) != 2:
        raise TypeError("screen_size must be a (width, height) pair")
    width, height = screen_size
    for dimension in (width, height):
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise TypeError("screen dimensions must be integers (not bool)")
        if dimension <= 0:
            raise ValueError("screen dimensions must be positive")
        _finite_real(dimension, "screen dimension")
    _validate_camera(camera)

    # Finish this pass before delegating any geometry to the camera.
    for index, sample in enumerate(samples):
        if not isinstance(sample, (list, tuple)) or len(sample) != 3:
            raise TypeError(f"samples[{index}] must be a three-item list or tuple")
        px, py, depth = (
            _finite_real(value, f"samples[{index}][{axis}]")
            for axis, value in enumerate(sample)
        )
        if not 0 <= px < width or not 0 <= py < height:
            raise ValueError(f"samples[{index}] pixel is outside the screen")
        if depth <= 0.1:
            raise ValueError(f"samples[{index}] forward_depth must exceed 0.1")

    points = []
    for sample in samples:
        px, py, depth = (float(value) for value in sample)
        world = camera.unproject(px, py, depth, width, height)
        try:
            wx, wy, wz = world
            coordinates = [_finite_real(value, "world coordinate")
                           for value in (wx, wy, wz)]
        except (TypeError, ValueError) as exc:
            raise ValueError("camera.unproject must return three finite real coordinates") from exc
        points.append({"pixel": [px, py], "depth": depth, "world": coordinates})
    return {"width": int(width), "height": int(height),
            "sample_count": len(points), "points": points}
