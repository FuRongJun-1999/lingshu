"""Standard-library configuration checks shared by the CLI and producer."""

from __future__ import annotations

import math


def positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def finite_number(value, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value)


def validate_options(*, threads=None, depth_threads=None, max_image_size=1600,
                     voxel_size=None, voxel_fraction=0.006,
                     confidence_percentile=0.0, max_depth_units=None,
                     meters_per_sfm_unit=None, depth_views="all",
                     depth_conv=None, sfm_features="full", matching_mode="auto") -> None:
    for name, value in (("threads", threads), ("depth_threads", depth_threads)):
        if value is not None:
            positive_int(value, name)
    positive_int(max_image_size, "max_image_size")
    for name, value in (("voxel_size", voxel_size), ("max_depth_units", max_depth_units),
                        ("meters_per_sfm_unit", meters_per_sfm_unit)):
        if value is not None:
            finite_number(value, name, positive=True)
    finite_number(voxel_fraction, "voxel_fraction", positive=True)
    percentile = finite_number(confidence_percentile, "confidence_percentile")
    if not 0 <= percentile <= 100:
        raise ValueError("confidence_percentile must be between 0 and 100")
    for name, value, choices in (
        ("depth_views", depth_views, {"auto", "all"}),
        ("depth_conv", depth_conv, {None, "auto", "direct", "winograd", "im2col"}),
        ("sfm_features", sfm_features, {"full", "native"}),
        ("matching_mode", matching_mode, {"auto", "sequential", "batched", "exhaustive"}),
    ):
        if value not in choices:
            raise ValueError(f"unsupported {name}: {value!r}")
