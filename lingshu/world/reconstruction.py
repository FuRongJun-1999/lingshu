"""Read-only CPU reconstruction products for the world's geometric layer.

The loader validates metadata, sparse observations and source JSONL before an
attachment is published. Native artifact presence is checked at load; explicit
``validate_artifacts()`` decodes PLY, SQLite, images and NPZ. A successful load
does not establish geometric accuracy or invent semantic entities, metric scale,
gravity, camera poses, timestamps or independent verification evidence.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath

from .reconstruction_camera import ReconstructionCamera


def _number(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite {'positive ' if positive else ''}number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} is too large") from exc
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{label} must be a finite {'positive ' if positive else ''}number")
    return result


def _integer(value, label, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _vector(value, size, label):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{label} must have {size} coordinates")
    return tuple(_number(x, label) for x in value)


def _object(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _text(value, label):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _reject_constant(value):
    raise ValueError(f"non-finite JSON constant {value}")


def _decode(text, label):
    try:
        return json.loads(text, parse_constant=_reject_constant)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"invalid JSON in {label}: {exc}") from exc


def _json(path):
    return _decode(path.read_text(encoding="utf-8"), path)


def _jsonl(path):
    with path.open(encoding="utf-8") as stream:
        for line, text in enumerate(stream, 1):
            if text.strip():
                yield _object(_decode(text, f"{path}:{line}"), f"{path}:{line}")


def _file(root, name, old_root=None):
    """Confine paths, rebasing old absolute producer paths after relocation."""
    raw = _text(name, "artifact path")
    path_type = PureWindowsPath if (PureWindowsPath(raw).drive or "\\" in raw) else PurePosixPath
    pure = path_type(raw)
    if ".." in pure.parts:
        raise ValueError(f"artifact path escapes run: {raw}")
    if pure.is_absolute():
        try:
            pure = pure.relative_to(path_type(_text(old_root, "original output directory")))
        except ValueError as exc:
            raise ValueError(f"artifact path is outside original run: {raw}") from exc
    # Reject drive-relative Windows paths such as C:mesh.ply as well.
    if pure.drive or pure.root:
        raise ValueError(f"artifact path must be relative to run: {raw}")
    resolved = root.joinpath(*pure.parts).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"artifact path escapes run: {raw}")
    if not resolved.is_file():
        raise ValueError(f"missing reconstruction artifact: {resolved}")
    return resolved


class Reconstruction:
    """Static geometry attachment; coordinates retain the SfM world's axes.

    Camera translation and sparse coordinates are converted once on reading.
    Fused artifacts and their provenance already use the report's output units.
    Files must remain unchanged while the attachment is used; iterators re-read
    them and validate each row, rather than retaining large JSONL arrays in RAM.
    """

    @classmethod
    def load(cls, run_directory):
        root = Path(run_directory).resolve()
        report_path = _file(root, "report.json")
        report_bytes = report_path.read_bytes()
        report = _object(_decode(report_bytes.decode("utf-8"), report_path), "report")
        if type(report.get("schema_version")) is not int or report["schema_version"] != 1:
            raise ValueError("unsupported reconstruction report schema_version")
        self = cls()
        self.root = root
        self.run_id = "reconstruction_" + hashlib.sha256(report_bytes).hexdigest()[:20]
        self.report = report
        self.units = _text(report.get("units"), "units")
        self.absolute_metric_scale = report.get("absolute_metric_scale")
        if type(self.absolute_metric_scale) is not bool:
            raise ValueError("absolute_metric_scale must be a boolean")
        stages = _object(report.get("stages"), "stages")
        alignment = _object(stages.get("alignment"), "alignment")
        conversion = _object(alignment.get("external_metric_conversion"), "external metric conversion")
        scale = conversion.get("meters_per_sfm_unit")
        relative_units = {"COLMAP relative units", "SfM relative units"}
        if self.absolute_metric_scale:
            self.coordinate_scale = _number(scale, "meters_per_sfm_unit", positive=True)
            if self.units != "meters":
                raise ValueError("metric reconstruction must declare meters")
        else:
            self.coordinate_scale = 1.0
            if scale is not None or self.units not in relative_units:
                raise ValueError("relative reconstruction must not claim metric conversion")
        if conversion.get("output_units") != self.units:
            raise ValueError("metric conversion output_units differs from report units")

        self.artifacts = {key: _file(root, path) for key, path in {
            "report": "report.json", "manifest": "00_input/input_manifest.json",
            "poses": "02_sparse/poses.json", "tracks": "02_sparse/tracks.jsonl",
            "scale_report": "04_alignment/scale_report.json",
        }.items()}
        scale_report = _object(_json(self.artifacts["scale_report"]), "scale report")
        if scale_report.get("external_metric_conversion") != conversion:
            raise ValueError("scale report and reconstruction report conversions differ")
        self.manifest = _json(self.artifacts["manifest"])
        if not isinstance(self.manifest, list) or len(self.manifest) < 2:
            raise ValueError("input manifest must contain at least two frames")
        self._frames = {}
        indices = set()
        for frame in self.manifest:
            _object(frame, "manifest frame")
            name = _text(frame.get("name"), "frame name")
            if "/" in name or "\\" in name or name in {".", ".."}:
                raise ValueError("frame name must be a filename")
            index = _integer(frame.get("index"), "frame index")
            if name in self._frames or index in indices:
                raise ValueError("duplicate input frame name or index")
            indices.add(index)
            _text(frame.get("source"), "source image reference")
            for size in ("width", "height", "source_width", "source_height"):
                _integer(frame.get(size), f"frame {size}", minimum=1)
            self._frames[name] = frame
        input_stage = _object(stages.get("input"), "input stage")
        if _integer(input_stage.get("images"), "input image count") != len(self.manifest):
            raise ValueError("input frame count differs from manifest")
        poses = _object(_json(self.artifacts["poses"]), "camera poses")
        if len(poses) < 2:
            raise ValueError("reconstruction must contain at least two registered views")
        self.cameras = {}
        for name, pose in poses.items():
            if name not in self._frames:
                raise ValueError(f"camera has no input frame: {name}")
            frame = self._frames[name]
            _object(pose, "camera pose")
            if pose.get("source") != frame["source"] or pose.get("source_index") != frame["index"]:
                raise ValueError(f"camera source mapping differs from manifest: {name}")
            self.cameras[name] = ReconstructionCamera.from_pose(
                name, pose, coordinate_scale=self.coordinate_scale)
            _file(root, f"02_undistorted/images/{name}")
        self.unregistered_images = [frame["source"] for frame in self.manifest
                                    if frame["name"] not in poses]
        sparse = _object(stages.get("sparse"), "sparse stage")
        if _integer(sparse.get("registered_images"), "registered image count") != len(poses):
            raise ValueError("registered image count differs from camera poses")
        if sparse.get("unregistered_images", self.unregistered_images) != self.unregistered_images:
            raise ValueError("unregistered image list differs from manifest")
        depth = _object(stages.get("depth"), "depth stage")
        self.inferred_images = self._frame_list(depth.get("inferred_images"), "inferred_images")
        self.skipped_images = self._frame_list(depth.get("skipped_images"), "skipped_images")
        if (set(self.inferred_images) & set(self.skipped_images)
                or set(self.inferred_images) | set(self.skipped_images) != set(poses)
                or _integer(depth.get("inferred_views"), "inferred view count") != len(self.inferred_images)
                or _integer(depth.get("registered_views"), "depth registered view count") != len(poses)):
            raise ValueError("dense frame coverage differs from registered views")
        if _integer(conversion.get("aligned_frames"), "aligned frame count") != len(self.inferred_images):
            raise ValueError("aligned frame count differs from dense views")
        for name in self.inferred_images:
            stem = Path(name).stem
            _file(root, f"03_depth/{stem}.npz")
            _file(root, f"04_alignment/depth_aligned/{stem}.npz")
        fusion = _object(stages.get("fusion"), "fusion stage")
        if (fusion.get("units") != self.units
                or _integer(fusion.get("integrated_frames"), "integrated frame count") != len(self.inferred_images)):
            raise ValueError("fusion units or frame count differs from depth stage")
        files = _object(fusion.get("files"), "fusion files")
        for key in ("mesh", "cloud", "provenance_jsonl", "provenance_sqlite", "mesh_provenance_jsonl"):
            self.artifacts[key] = _file(root, files.get(key), report.get("output"))

        count = observations = 0
        point_ids = set()
        for track in self.iter_tracks():
            if track["point3D_id"] in point_ids:
                raise ValueError("duplicate sparse point identity")
            point_ids.add(track["point3D_id"])
            count += 1
            observations += len(track["observations"])
        if (_integer(sparse.get("sparse_points"), "sparse point count") != count
                or _integer(sparse.get("observations"), "sparse observation count") != observations):
            raise ValueError("sparse track counts differ from report")
        for key, rows in (("valid_voxels", self.iter_voxel_provenance()),
                          ("mesh_vertices", self.iter_mesh_provenance())):
            if _integer(fusion.get(key), key) != sum(1 for _ in rows):
                raise ValueError(f"{key} provenance count differs from report")
        return self

    def _frame_list(self, value, label):
        if (not isinstance(value, list) or any(not isinstance(v, str) for v in value)
                or len(set(value)) != len(value) or any(v not in self.cameras for v in value)):
            raise ValueError(f"{label} must contain unique registered frame names")
        return list(value)

    def validate_artifacts(self):
        """Explicit full content pass (NumPy/OpenCV lazy; no accuracy claim)."""
        from .reconstruction_artifacts import validate_artifacts

        return validate_artifacts(self)

    def render(self, view_id, **options):
        """Return static native-view RGB, camera-Z depth and visibility mask."""
        from .reconstruction_render import render_reconstruction

        return render_reconstruction(self, view_id, **options)

    def iter_tracks(self):
        """Yield sparse geometry in output units; no category or identity guessing."""
        for row in _jsonl(self.artifacts["tracks"]):
            _integer(row.get("point3D_id"), "point3D_id")
            xyz = _vector(row.get("xyz"), 3, "sparse xyz")
            scaled = [x * self.coordinate_scale for x in xyz]
            if any(not math.isfinite(x) for x in scaled):
                raise ValueError("scaled sparse coordinate is not finite")
            obs = row.get("observations")
            if not isinstance(obs, list) or len(obs) < 2:
                raise ValueError("sparse track needs at least two observations")
            seen = set()
            copied = []
            for original in obs:
                _object(original, "track observation")
                name = _text(original.get("image"), "track observation image")
                if name not in self.cameras:
                    raise ValueError("track observation has unknown camera")
                seen.add(name)
                _vector(original.get("pixel_xy"), 2, "track pixel")
                depth = _number(original.get("camera_z"), "track camera_z", positive=True)
                scaled_depth = depth * self.coordinate_scale
                if not math.isclose(self.cameras[name].camera_depth(scaled), scaled_depth,
                                    rel_tol=1e-7, abs_tol=1e-7 * self.coordinate_scale):
                    raise ValueError("track camera_z disagrees with camera pose")
                copied.append({**original, "camera_z": scaled_depth})
            if len(seen) < 2:
                raise ValueError("sparse track needs observations from two distinct views")
            yield {**row, "xyz": scaled, "observations": copied}

    def triangulate_track(self, point3D_id):
        """Re-estimate one track with Lingshu's existing ray triangulation."""
        from .multiview import ViewObs, _triangulate

        _integer(point3D_id, "point3D_id")
        for track in self.iter_tracks():
            if track["point3D_id"] != point3D_id:
                continue
            views = []
            for obs in track["observations"]:
                camera = self.cameras[obs["image"]]
                pixel = camera.to_virtual_pixel(*obs["pixel_xy"])
                views.append(ViewObs(camera.as_camera3d(), pixel, camera.width, camera.height))
            return _triangulate(views)
        raise KeyError(point3D_id)

    def _sources(self, value):
        if not isinstance(value, list) or not value:
            raise ValueError("voxel provenance must retain source observations")
        for source in value:
            _object(source, "source observation")
            name = _text(source.get("image"), "source image")
            if name not in self.inferred_images:
                raise ValueError("source observation has no inferred depth frame")
            if source.get("frame") != self.cameras[name].source:
                raise ValueError("source frame differs from input image reference")
            pixel = _vector(source.get("pixel_xy"), 2, "source pixel")
            if any(v < 0 for v in pixel):
                raise ValueError("source pixel must be nonnegative")

    def iter_voxel_provenance(self):
        """Yield already-scaled fusion provenance in the producer's depth pixels."""
        for row in _jsonl(self.artifacts["provenance_jsonl"]):
            _vector(row.get("voxel"), 3, "voxel key")
            _vector(row.get("center"), 3, "voxel center")
            _integer(row.get("observations"), "voxel observations", minimum=1)
            if row.get("units") != self.units:
                raise ValueError("voxel units differ from reconstruction units")
            self._sources(row.get("sources"))
            yield row

    def iter_mesh_provenance(self):
        """Yield nearest-observed-voxel mapping, preserving unmatched vertices."""
        for index, row in enumerate(_jsonl(self.artifacts["mesh_provenance_jsonl"])):
            if _integer(row.get("mesh_vertex"), "mesh vertex index") != index:
                raise ValueError("mesh provenance indices must be consecutive")
            _vector(row.get("xyz"), 3, "mesh vertex")
            if row.get("units") != self.units:
                raise ValueError("mesh units differ from reconstruction units")
            if row.get("source") is not None:
                source = _object(row["source"], "mesh source voxel")
                _vector(source.get("voxel"), 3, "mesh source voxel key")
                _integer(source.get("observations"), "mesh source observations", minimum=1)
                if _number(row.get("nearest_distance"), "nearest source distance") < 0:
                    raise ValueError("nearest source distance must be nonnegative")
                self._sources(source.get("sources"))
            elif row.get("nearest_distance") is not None:
                raise ValueError("unmapped mesh vertex must not claim a source distance")
            yield row

    def to_dict(self):
        """Describe the attachment without claiming semantic or GT acceptance."""
        return {
            "schema": "lingshu.reconstruction/1", "run_id": self.run_id,
            "run_directory": str(self.root), "units": self.units,
            "absolute_metric_scale": self.absolute_metric_scale,
            "coordinate_scale": self.coordinate_scale,
            "coordinate_frame": "SfM world axes; gravity and origin are not calibrated",
            "frames": [{"name": f["name"], "source": f["source"],
                        "source_index": f["index"], "registered": f["name"] in self.cameras,
                        "dense": f["name"] in self.inferred_images} for f in self.manifest],
            "unregistered_images": list(self.unregistered_images),
            "skipped_depth_images": list(self.skipped_images),
            "artifacts": {key: str(path) for key, path in self.artifacts.items()},
            "verification": deepcopy(self.report.get("verification", {})),
            "validation": "load checks metadata, sparse tracks and source JSONL; native contents require explicit validate_artifacts()",
        }
