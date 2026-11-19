"""Run CPU GLOMAP + DA3 reconstruction with explicitly supplied local assets."""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import sys

from .config import validate_options


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("images", type=Path, help="overlapping photos ordered by filename")
    result.add_argument("--output", type=Path, help="new run directory; defaults to ./runs/<timestamp>")
    result.add_argument("--resource-root", type=Path,
                        help="asset root; defaults to LINGSHU_RECONSTRUCTION_HOME or ~/.cache/lingshu/reconstruction")
    result.add_argument("--colmap", type=Path, help="CPU COLMAP executable with global_mapper")
    result.add_argument("--model", type=Path, help="DA3 Base q8_0 GGUF model")
    result.add_argument("--depth-library", type=Path, help="depth-anything.cpp shared C API library")
    result.add_argument("--runtime-bin", type=Path, help="optional Windows runtime DLL directory")
    result.add_argument("--threads", type=int,
                        help="CPU thread override; defaults to 2 SfM / up to 4 physical DA3 cores")
    result.add_argument("--depth-threads", type=int, help="separate DA3 thread override")
    result.add_argument("--sfm-features", choices=("full", "native"), default="full",
                        help="full SIFT or faster native resolution with possible track loss")
    result.add_argument("--sfm-matching", choices=("auto", "sequential", "batched", "exhaustive"), default="auto",
                        help="ordered pairs by default; exhaustive explicitly matches all pairs in unordered sets")
    result.add_argument("--depth-conv", choices=("auto", "direct", "winograd", "im2col"),
                        help="DA3 convolution backend; Windows without AVX-512 defaults to direct")
    result.add_argument("--depth-views", choices=("auto", "all"), default="all",
                        help="process every registered view (default); auto explicitly trades dense coverage for speed")
    result.add_argument("--include-da3-pose", action="store_true",
                        help="also export DA3 pose diagnostics; reconstruction always uses GLOMAP")
    result.add_argument("--max-image-size", type=int, default=1600)
    result.add_argument("--voxel-size", type=float, help="TSDF edge in output coordinate units")
    result.add_argument("--voxel-fraction", type=float, default=0.006)
    result.add_argument("--confidence-percentile", type=float, default=0.0,
                        help="optional lossy depth-confidence percentile filter; default keeps the valid input domain")
    result.add_argument("--max-depth", type=float, help="TSDF depth truncation in output units")
    result.add_argument("--meters-per-sfm-unit", type=float,
                        help="trusted external scale factor; otherwise output stays SfM-relative")
    result.add_argument("--camera-model", default="SIMPLE_RADIAL")
    return result


def resource_paths(args: argparse.Namespace) -> dict[str, Path]:
    """Resolve assets without downloading them or relying on the checkout path."""
    root = args.resource_root or Path(os.environ.get(
        "LINGSHU_RECONSTRUCTION_HOME", str(Path.home() / ".cache/lingshu/reconstruction")))
    windows = os.name == "nt"
    build = root / ".tools/depth-anything.cpp/build"
    return {
        "colmap": args.colmap or root / ".tools/colmap/bin" / ("colmap.exe" if windows else "colmap"),
        "model": args.model or root / "models/depth-anything-base-q8_0.gguf",
        "depth_library": args.depth_library or build / ("libdepthanything.dll" if windows else "libdepthanything.so"),
        "runtime_bin": args.runtime_bin or root / ".tools/llvm-mingw/bin",
    }


def main(argv: list[str] | None = None) -> int:
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    options = dict(
        threads=args.threads, depth_threads=args.depth_threads,
        max_image_size=args.max_image_size, voxel_size=args.voxel_size,
        voxel_fraction=args.voxel_fraction, confidence_percentile=args.confidence_percentile,
        max_depth_units=args.max_depth, meters_per_sfm_unit=args.meters_per_sfm_unit,
        depth_views=args.depth_views, depth_conv=args.depth_conv,
        sfm_features=args.sfm_features, matching_mode=args.sfm_matching,
    )
    try:
        validate_options(**options)
    except ValueError as error:
        argument_parser.error(str(error))
    try:
        from .pipeline import ReconstructionPipeline
    except ImportError as error:
        print('CPU reconstruction needs the optional dependencies: pip install "lingshu[world-reconstruct]". '
              f"Missing import: {error}", file=sys.stderr)
        return 1
    try:
        ReconstructionPipeline(
            input_path=args.images,
            output_path=args.output or Path.cwd() / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S"),
            include_da3_pose=args.include_da3_pose, camera_model=args.camera_model,
            **resource_paths(args), **options,
        ).run()
    except (OSError, ValueError, RuntimeError) as error:
        print(f"reconstruction failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
