from __future__ import annotations

from contextlib import contextmanager
import ctypes
import os
from pathlib import Path

import numpy as np


def convolution_mode(requested: str | None = None) -> str:
    selected = requested or os.environ.get("DA_CONV")
    if selected:
        if selected not in {"auto", "direct", "winograd", "im2col"}:
            raise ValueError(f"unsupported DA_CONV: {selected!r}")
        return selected
    if os.name == "nt" and not ctypes.windll.kernel32.IsProcessorFeaturePresent(41):  # PF_AVX512F
        return "direct"
    return "auto"


@contextmanager
def depth_convolution(mode: str):
    previous = os.environ.get("DA_CONV")
    os.environ["DA_CONV"] = mode
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("DA_CONV", None)
        else:
            os.environ["DA_CONV"] = previous


class DepthEngine:
    def __init__(self, library_path: Path, model_path: Path, runtime_dir: Path | None = None, threads: int = 2):
        self._dll_dirs = []
        if os.name == "nt":
            runtime_dirs = [library_path.parent]
            if runtime_dir is not None:
                runtime_dirs.append(runtime_dir)
            for directory in runtime_dirs:
                if directory.exists():
                    self._dll_dirs.append(os.add_dll_directory(str(directory.resolve())))
        self.lib = ctypes.CDLL(str(library_path.resolve()))
        required = ("da_capi_load", "da_capi_free", "da_capi_free_floats",
                    "da_capi_last_error", "da_capi_depth_dense")
        missing = [name for name in required if not hasattr(self.lib, name)]
        if missing:
            self._dll_dirs.clear()
            raise RuntimeError("DA3 shared library is missing the reconstruction C API: "
                               + ", ".join(missing) + "; rebuild the native adapter")
        self.lib.da_capi_load.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.lib.da_capi_load.restype = ctypes.c_void_p
        self.lib.da_capi_free.argtypes = [ctypes.c_void_p]
        self.lib.da_capi_free.restype = None
        self.lib.da_capi_free_floats.argtypes = [ctypes.c_void_p]
        self.lib.da_capi_free_floats.restype = None
        self.lib.da_capi_last_error.argtypes = [ctypes.c_void_p]
        self.lib.da_capi_last_error.restype = ctypes.c_char_p
        self.lib.da_capi_depth_dense.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int),
        ]
        self.lib.da_capi_depth_dense.restype = ctypes.c_int
        self.ctx = self.lib.da_capi_load(os.fsencode(model_path.resolve()), threads)
        if not self.ctx:
            raise RuntimeError(f"DA3 could not load model: {model_path}")

    def predict(self, image_path: Path, *, include_pose: bool = True) -> dict:
        height = ctypes.c_int()
        width = ctypes.c_int()
        depth_ptr = ctypes.c_void_p()
        conf_ptr = ctypes.c_void_p()
        sky_ptr = ctypes.c_void_p()
        extrinsics = (ctypes.c_float * 12)() if include_pose else None
        intrinsics = (ctypes.c_float * 9)() if include_pose else None
        is_metric = ctypes.c_int()
        status = self.lib.da_capi_depth_dense(
            self.ctx,
            os.fsencode(image_path.resolve()),
            ctypes.byref(height),
            ctypes.byref(width),
            ctypes.byref(depth_ptr),
            ctypes.byref(conf_ptr),
            ctypes.byref(sky_ptr),
            extrinsics,
            intrinsics,
            ctypes.byref(is_metric),
        )
        if status != 0:
            detail = self.lib.da_capi_last_error(self.ctx)
            message = detail.decode("utf-8", errors="replace") if detail else "unknown DA3 error"
            raise RuntimeError(f"DA3 inference failed for {image_path.name}: {message}")
        count = height.value * width.value
        try:
            if not depth_ptr or not conf_ptr:
                raise RuntimeError("the selected DA3 checkpoint did not return depth and confidence")
            depth = np.ctypeslib.as_array(ctypes.cast(depth_ptr, ctypes.POINTER(ctypes.c_float)), shape=(count,)).copy()
            confidence = np.ctypeslib.as_array(ctypes.cast(conf_ptr, ctypes.POINTER(ctypes.c_float)), shape=(count,)).copy()
            return {
                "depth": depth.reshape(height.value, width.value),
                "confidence": confidence.reshape(height.value, width.value),
                "extrinsics": np.ctypeslib.as_array(extrinsics).reshape(3, 4).copy() if include_pose else None,
                "intrinsics": np.ctypeslib.as_array(intrinsics).reshape(3, 3).copy() if include_pose else None,
                "is_metric": bool(is_metric.value),
            }
        finally:
            if depth_ptr:
                self.lib.da_capi_free_floats(depth_ptr)
            if conf_ptr:
                self.lib.da_capi_free_floats(conf_ptr)
            if sky_ptr:
                self.lib.da_capi_free_floats(sky_ptr)

    def close(self) -> None:
        if self.ctx:
            self.lib.da_capi_free(self.ctx)
            self.ctx = None
        self._dll_dirs.clear()

    def __enter__(self) -> "DepthEngine":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
