from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def list_images(source: Path) -> list[Path]:
    paths = [source] if source.is_file() else [p for p in source.rglob("*") if p.is_file()]
    return sorted((p.resolve() for p in paths if p.suffix.lower() in IMAGE_EXTENSIONS), key=lambda p: str(p).casefold())


def normalize_images(source: Path, destination: Path, max_image_size: int) -> list[dict]:
    """Normalize RGB and preserve optional alpha as a binary input valid domain.

    For uint8/uint16 alpha, every value > 0 is valid, including semitransparent
    pixels. This is input validity, not a foreground estimate. Opaque alpha and
    ordinary RGB create no mask; black RGB pixels are never treated as invalid.
    """
    paths = list_images(source)
    if len(paths) < 2:
        raise ValueError(f"at least two input images are required; found {len(paths)}")

    destination.mkdir(parents=True, exist_ok=False)
    manifest = []
    for index, path in enumerate(paths):
        data = np.fromfile(path, dtype=np.uint8)
        raw = cv2.imdecode(data, cv2.IMREAD_UNCHANGED) if path.suffix.lower() not in {".jpg", ".jpeg"} else None
        alpha_status, alpha_dtype, mask = "absent", None, None
        if raw is not None and raw.ndim == 3 and raw.shape[2] == 4:
            if raw.dtype not in (np.dtype("uint8"), np.dtype("uint16")):
                raise ValueError(f"alpha must use uint8 or uint16 samples: {path}")
            alpha = raw[:, :, 3]
            if not np.any(alpha > 0):
                raise ValueError(f"fully transparent input has no valid pixels: {path}")
            alpha_dtype = raw.dtype.name
            alpha_status = "opaque" if np.all(alpha == np.iinfo(raw.dtype).max) else "transparent"
            if alpha_status == "transparent":
                mask = (alpha > 0).astype(np.uint8) * 255
            # Color and alpha share the same decoded orientation. For 16-bit
            # samples, use the same high-byte conversion as IMREAD_COLOR.
            image = (raw[:, :, :3] >> 8).astype(np.uint8) if raw.dtype == np.uint16 else raw[:, :, :3].copy()
        else:
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError(f"cannot decode image: {path}")
        height, width = image.shape[:2]
        factor = min(1.0, max_image_size / max(height, width))
        if factor < 1.0:
            size = (max(1, round(width * factor)), max(1, round(height * factor)))
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            if mask is not None:
                mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
                if not np.any(mask):
                    raise ValueError(f"input alpha has no valid pixels after resizing: {path}")
        name = f"{index:06d}.jpg"
        output = destination / name
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RuntimeError(f"failed to normalize image: {path}")
        encoded.tofile(output)
        mask_relative = None
        if mask is not None:
            mask_relative = f"masks/{index:06d}.png"
            _write_mask(destination.parent / mask_relative, mask)
        manifest.append({
            "index": index,
            "name": name,
            "source": str(path),
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "source_width": int(width),
            "source_height": int(height),
            "alpha_status": alpha_status,
            "alpha_dtype": alpha_dtype,
            "valid_mask": mask_relative,
            "valid_mask_rule": "alpha>0" if mask is not None else None,
        })

    (destination.parent / "input_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def _write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", mask)
    if not ok:
        raise RuntimeError(f"failed to encode input valid mask: {path}")
    encoded.tofile(path)


def _confined_relative(root: Path, value: str, *, filename: bool = False) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("manifest paths must be nonempty relative paths")
    windows, posix = PureWindowsPath(value), PurePosixPath(value.replace("\\", "/"))
    if windows.drive or windows.root or posix.root or ".." in posix.parts:
        raise ValueError(f"manifest path must be relative and cannot escape its root: {value}")
    if filename and (len(posix.parts) != 1 or posix.name in {"", ".", ".."}):
        raise ValueError(f"manifest image name must be a confined filename: {value}")
    resolved = (root / Path(*posix.parts)).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"manifest path resolves outside its root (escape): {value}")
    return resolved


def export_feature_masks(manifest: list[dict], input_root: Path, destination: Path) -> Path | None:
    """Pass the declared input domain to native SIFT without inventing masks.

    COLMAP requires ``<full image filename>.png`` for every image when its
    mask directory is enabled. Unmasked RGB frames therefore receive an
    explicitly all-valid bitmap only in a mixed masked/unmasked input set.
    """
    if not any(record.get("valid_mask") is not None for record in manifest):
        return None
    seen = set()
    for record in manifest:
        name = record["name"]
        _confined_relative(input_root, name, filename=True)
        if name in seen:
            raise ValueError(f"duplicate input image name: {name}")
        seen.add(name)
        shape = (record["height"], record["width"])
        relative = record.get("valid_mask")
        if relative is None:
            mask = np.full(shape, 255, dtype=np.uint8)
        else:
            source = _confined_relative(input_root, relative)
            mask = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
            if mask is None or mask.dtype != np.uint8 or mask.ndim != 2:
                raise ValueError(f"cannot decode binary input valid mask: {source}")
            if mask.shape != shape:
                raise ValueError(f"input valid mask dimensions do not match {name}")
            if np.any((mask != 0) & (mask != 255)) or not np.any(mask):
                raise ValueError(f"input valid mask must have a nonempty binary domain: {source}")
        _write_mask(_confined_relative(destination, name + ".png", filename=True), mask)
    return destination


def _remap_mask(mask: np.ndarray, source_camera, target_camera, source_pose: np.ndarray,
                target_pose: np.ndarray) -> np.ndarray:
    source_center = -source_pose[:, :3].T @ source_pose[:, 3]
    target_center = -target_pose[:, :3].T @ target_pose[:, 3]
    if not np.allclose(source_center, target_center, rtol=1e-8, atol=1e-8):
        raise ValueError("cannot map input validity between different camera centers without depth")
    relative_rotation = source_pose[:, :3] @ target_pose[:, :3].T
    height, width = int(target_camera.height), int(target_camera.width)
    result = np.zeros((height, width), dtype=np.uint8)
    for start in range(0, height, 128):
        end = min(start + 128, height)
        ys, xs = np.mgrid[start:end, :width]
        # COLMAP uses pixel centers at (x+.5,y+.5); OpenCV remap uses zero-
        # based center indices. Convert exactly once at each end of the rays.
        pixels = np.column_stack((xs.ravel() + .5, ys.ravel() + .5))
        normalized = np.asarray(target_camera.cam_from_img(pixels), dtype=np.float64)
        rays = np.column_stack((normalized, np.ones(len(pixels)))) @ relative_rotation.T
        valid = np.isfinite(rays).all(axis=1) & (rays[:, 2] > 0)
        projected = np.full((len(pixels), 2), np.nan, dtype=np.float64)
        if np.any(valid):
            projected[valid] = source_camera.img_from_cam(rays[valid])
        valid &= (np.isfinite(projected).all(axis=1)
                  & (projected[:, 0] >= 0) & (projected[:, 0] < source_camera.width)
                  & (projected[:, 1] >= 0) & (projected[:, 1] < source_camera.height))
        map_x = np.where(valid, projected[:, 0] - .5, -1).astype(np.float32).reshape(end-start, width)
        map_y = np.where(valid, projected[:, 1] - .5, -1).astype(np.float32).reshape(end-start, width)
        result[start:end] = cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return result


def undistort_valid_masks(manifest: list[dict], input_root: Path, source_model: Path,
                         rectified_model: Path, destination: Path) -> dict[str, str]:
    """Map alpha validity through native source/rectified camera ray models.

    ``valid_mask`` is relative to ``input_root`` (the 00_input directory).
    Returns registered image names mapped to mask filenames in ``destination``.
    Models must describe the same camera centers; their orientations, dimensions,
    intrinsics and distortion may differ. Unregistered views are left unmapped.
    Only supplied alpha masks are read; RGB, scene colors and GT are not masks.
    """
    masks = []
    names = set()
    for record in manifest:
        relative = record.get("valid_mask")
        if relative is None:
            continue
        name = record["name"]
        _confined_relative(input_root, name, filename=True)
        path = _confined_relative(input_root, relative)
        if name in names:
            raise ValueError(f"duplicate masked image name in manifest: {name}")
        names.add(name)
        masks.append((name, path))
    if not masks:
        return {}

    import pycolmap

    source = pycolmap.Reconstruction(str(source_model))
    target = pycolmap.Reconstruction(str(rectified_model))
    source_images = {image.name: image for image in source.images.values() if image.has_pose}
    target_images = {image.name: image for image in target.images.values() if image.has_pose}
    mapped = {}
    for name, path in masks:
        if name not in target_images:
            continue
        if name not in source_images:
            raise ValueError(f"rectified view has no corresponding native source pose: {name}")
        mask = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        source_image, target_image = source_images[name], target_images[name]
        source_camera = source.camera(source_image.camera_id)
        target_camera = target.camera(target_image.camera_id)
        if mask is None or mask.dtype != np.uint8 or mask.ndim != 2 or not np.isin(mask, [0, 255]).all():
            raise ValueError(f"input valid mask must be a binary uint8 image: {path}")
        if mask.shape != (source_camera.height, source_camera.width):
            raise ValueError(f"mask dimensions do not match native source camera for {name}: {path}")
        source_pose = np.asarray(source_image.cam_from_world().matrix(), dtype=np.float64)
        target_pose = np.asarray(target_image.cam_from_world().matrix(), dtype=np.float64)
        warped = _remap_mask(mask, source_camera, target_camera, source_pose, target_pose)
        filename = f"{Path(name).stem}.png"
        output = _confined_relative(destination, filename, filename=True)
        if filename in mapped.values():
            raise ValueError(f"masked image names collide at output filename: {filename}")
        _write_mask(output, warped)
        mapped[name] = filename
    return mapped


def copy_manifest_image(record: dict, source_dir: Path, target_dir: Path) -> Path:
    source = source_dir / record["name"]
    target = target_dir / record["name"]
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target
