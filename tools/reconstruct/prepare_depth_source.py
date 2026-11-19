"""Enable DA3's fused depth graph in an explicitly supplied source checkout.

This applies the small C API change required by ``--include-da3-pose`` being
off. It downloads nothing and rejects source with missing or ambiguous anchors.
Use the depth-anything.cpp archive recorded in the reconstruction resource lock.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


_ANCHOR = b"""    } else {
        std::array<float,12> ext; std::array<float,9> intr;
        if (!c->engine->depth_pose_native(img, depth, second, ext, intr, H, W)){"""
_PREFIX = b"""    } else if (!out_ext && !out_intr){
        if (!c->engine->depth_native_image(img, depth, second, H, W) || second.size() != depth.size()){
            c->last_error = "depth_dense: depth-only graph failed"; return -1; }
"""
_MARKER = b'c->last_error = "depth_dense: depth-only graph failed"'


def patch_depth_source(path: Path) -> bool:
    """Patch one C++ file, preserving its bytes outside the unique anchor.

    Return True when changed and False when the exact patch is already present.
    Any other partial/multiple patch state raises before writing.
    """
    path = Path(path)
    source = path.read_bytes()
    anchors = [(eol, _ANCHOR.replace(b"\n", eol)) for eol in (b"\n", b"\r\n")]
    matches = [(eol, anchor) for eol, anchor in anchors if source.count(anchor)]
    if len(matches) != 1 or source.count(matches[0][1]) != 1:
        raise ValueError("Unexpected depth-anything.cpp source: expected one C API anchor; use the locked archive")
    eol, anchor = matches[0]
    replacement = _PREFIX.replace(b"\n", eol) + anchor
    if source.count(replacement) == 1 and source.count(_MARKER) == 1:
        return False
    if _MARKER in source or b"else if (!out_ext && !out_intr)" in source:
        raise ValueError("Unexpected partial depth-only patch; use the locked archive")
    path.write_bytes(source.replace(anchor, replacement, 1))
    return True


def prepare_source(source: Path) -> bool:
    """Prepare an explicitly selected depth-anything.cpp source directory."""
    return patch_depth_source(Path(source) / "src" / "da_capi.cpp")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="unpacked depth-anything.cpp source directory")
    args = parser.parse_args(argv)
    try:
        changed = prepare_source(args.source)
    except (OSError, ValueError) as error:
        print(f"Depth source preparation failed: {error}", file=sys.stderr)
        return 1
    print("Enabled fused depth C API" if changed else "Fused depth C API already enabled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
