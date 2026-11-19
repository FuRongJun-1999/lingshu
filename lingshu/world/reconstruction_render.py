"""Static reconstruction mesh rendering through Lingshu's CPU z-buffer.

Native pinhole K is preserved, including skew. Depth is camera-Z in the bundle's
declared units; a zero depth/mask means no surface. Array pixel (x,y) is sampled
at COLMAP continuous coordinates (x+0.5,y+0.5), matching the reconstruction/GT
camera contract without modifying K or the continuous camera adapter.
RGB is a face-averaged vertex-color preview using the existing renderer's color
curve, not a textured/photometrically calibrated novel-view reconstruction.
"""
from __future__ import annotations

from array import array
from dataclasses import dataclass
import math


@dataclass
class ReconstructionRender:
    width: int
    height: int
    rgb: bytes
    depth: array
    mask: bytes
    units: str
    view_id: str
    near: float
    far: float | None
    stats: dict

    def to_image(self):
        """Return an RGB Pillow image; requires the existing ``world`` extra."""
        from PIL import Image

        return Image.frombytes("RGB", (self.width, self.height), self.rgb)


def _clip_polygon(polygon, plane):
    """Sutherland-Hodgman clipping in camera space against a linear plane."""
    if not polygon:
        return []
    result = []
    previous = polygon[-1]
    before = plane(previous)
    for current in polygon:
        after = plane(current)
        if (before >= 0) != (after >= 0):
            ratio = before / (before - after)
            result.append(tuple(a + ratio * (b-a) for a, b in zip(previous, current)))
        if after >= 0:
            result.append(current)
        previous, before = current, after
    return result


def render_reconstruction(reconstruction, view_id, *, near=None, far=None, background=(20, 24, 40)):
    """Render one registered native view without semantic entities or time ticks.

    Near defaults to 1e-6 SfM units converted to the output units. Explicit near
    and far are output-unit camera-Z distances. Clipping includes all six view
    planes before projection; the shared rasterizer interpolates inverse Z and
    resolves visibility per pixel. Render again after changing artifact bytes.
    """
    from .mesh3d import ZBuffer, rasterize_triangles
    from .reconstruction_artifacts import read_ply

    if view_id not in reconstruction.cameras:
        raise KeyError(f"unknown registered reconstruction view: {view_id}")
    camera = reconstruction.cameras[view_id]
    if near is None:
        near = 1e-6 * camera.coordinate_scale
    for label, value in (("near", near), ("far", far)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value) or value <= 0):
            raise ValueError(f"{label} must be a finite positive distance")
    if far is not None and far <= near:
        raise ValueError("far must be greater than near")
    if (not isinstance(background, (tuple, list)) or len(background) != 3
            or any(type(value) is not int or not 0 <= value <= 255 for value in background)):
        raise ValueError("background must contain three RGB bytes")
    mesh = read_ply(reconstruction.artifacts["mesh"], mesh=True)
    matrix = camera.cam_from_world
    # Fused mesh vertices are already in output units; scale only original t.
    points = [tuple(sum(row[i]*point[i] for i in range(3)) + camera.coordinate_scale*row[3]
                    for row in matrix) for point in mesh.vertices]
    if any(not all(math.isfinite(value) for value in point) for point in points):
        raise ValueError("non-finite transformed mesh coordinates")
    k = camera.intrinsics
    fx, skew, cx, fy, cy = k[0][0], k[0][1], k[0][2], k[1][1], k[1][2]
    width, height = camera.width, camera.height
    planes = [lambda p: p[2]-near,
              lambda p: fx*p[0]+skew*p[1]+cx*p[2],
              lambda p: (width-cx)*p[2]-fx*p[0]-skew*p[1],
              lambda p: fy*p[1]+cy*p[2],
              lambda p: (height-cy)*p[2]-fy*p[1]]
    if far is not None:
        planes.insert(1, lambda p: far-p[2])
    triangles = []
    clipped_faces = 0
    for indices in mesh.faces:
        original = [points[index] for index in indices]
        polygon = original
        for plane in planes:
            polygon = _clip_polygon(polygon, plane)
            if len(polygon) < 3:
                break
        if polygon != original:
            clipped_faces += 1
        if len(polygon) < 3:
            continue
        color = tuple(round(sum(mesh.colors[index][c] for index in indices)/3) for c in range(3))
        projected = [(fx*(x/z)+skew*(y/z)+cx,
                      fy*(y/z)+cy, 1.0/z, 1.0) for x, y, z in polygon]
        if any(not all(math.isfinite(value) for value in point) for point in projected):
            raise ValueError("non-finite projected mesh coordinates")
        for index in range(1, len(projected)-1):
            triangles.append((*projected[0], *projected[index], *projected[index+1], *color))
    buffer = ZBuffer(width, height, background=background)
    stats = {"input_vertices": len(points), "input_faces": len(mesh.faces),
             "clipped_faces": clipped_faces, "raster_triangles": len(triangles)}
    rasterize_triangles(buffer.depth, buffer.color, width, height, triangles, stats=stats)
    mask = bytes(int(value > 0) for value in buffer.depth)
    depth = array("f", (1.0/value if value > 0 else 0.0 for value in buffer.depth))
    stats["visible_pixels"] = sum(mask)
    return ReconstructionRender(width, height, bytes(buffer.color), depth, mask,
                                reconstruction.units, view_id, float(near),
                                float(far) if far is not None else None, stats)
