"""Real-file guards for concurrent consolidated model saves.

The barrier only orders the final renames. Serialization, fsync, rename,
cleanup and model loading all use their real implementations.
"""
import builtins
import json
import os
from pathlib import Path
import stat
import threading
from types import SimpleNamespace
import uuid

import numpy as np
import pytest

from lingshu.nn import hex_cnn


@pytest.mark.parametrize("second_fails", [False, True], ids=["success", "failure"])
def test_parallel_saves_do_not_share_temporary_files(tmp_path, monkeypatch, second_fails):
    target = tmp_path / "model.json"
    first_kernel = np.arange(7, dtype=np.float32)
    hex_cnn.save_consolidated({"k": first_kernel}, {"run": "original"}, str(target))
    original = target.read_bytes()
    ready, second_done = threading.Event(), threading.Event()
    real_replace = os.replace
    outcomes = {}

    def ordered_replace(src, dst):
        if threading.current_thread().name == "first-model-save":
            ready.set()
            assert second_done.wait(10), "second save did not finish"
        return real_replace(src, dst)

    def first_save():
        try:
            outcomes["first"] = hex_cnn.save_consolidated(
                {"k": first_kernel}, {"run": "first"}, str(target))
        except BaseException as exc:
            outcomes["first"] = type(exc).__name__

    monkeypatch.setattr(hex_cnn.os, "replace", ordered_replace)
    worker = threading.Thread(target=first_save, name="first-model-save", daemon=True)
    worker.start()
    try:
        assert ready.wait(10), "first save did not reach its rename"
        assert target.read_bytes() == original, "unpublished save changed the old model"
        second_meta = {"run": "second"}
        if second_fails:
            second_meta["invalid"] = object()
        try:
            outcomes["second"] = hex_cnn.save_consolidated(
                {"k": first_kernel + 10}, second_meta, str(target))
        except BaseException as exc:
            outcomes["second"] = type(exc).__name__
        if second_fails:
            assert outcomes["second"] == "TypeError"
            assert target.read_bytes() == original, "failed save changed the old model"
        else:
            assert outcomes["second"] == str(target), "concurrent valid save was rejected"
            np.testing.assert_array_equal(hex_cnn.load_consolidated(str(target))["k"], first_kernel + 10)
    finally:
        second_done.set()
        worker.join(10)
    assert not worker.is_alive(), "first save remained blocked"
    assert outcomes["first"] == str(target), "another save consumed the first save's temporary file"
    assert json.loads(target.read_text(encoding="utf-8"))["run"] == "first"
    np.testing.assert_array_equal(hex_cnn.load_consolidated(str(target))["k"], first_kernel)
    assert not list(tmp_path.glob("*.tmp-*")), "save left temporary files behind"


def test_temporary_name_collision_keeps_unowned_file(tmp_path, monkeypatch):
    target = tmp_path / "model.json"
    hex_cnn.save_consolidated({"k": np.zeros(7)}, {"run": "original"}, str(target))
    original = target.read_bytes()
    monkeypatch.setattr(uuid, "uuid4", lambda: SimpleNamespace(hex="collision"))
    candidates = [Path(f"{target}.tmp-{os.getpid()}"),
                  tmp_path / f".lingshu-save.tmp-{os.getpid()}-collision"]
    for candidate in candidates:
        candidate.write_bytes(b"owned by another save")
    error = None
    try:
        hex_cnn.save_consolidated({"k": np.ones(7)}, {"run": "new"}, str(target))
    except FileExistsError as exc:
        error = exc
    assert error is not None, "temporary-name collision was not rejected"
    assert all(p.exists() for p in candidates), "another save's temporary file was removed"
    assert all(p.read_bytes() == b"owned by another save" for p in candidates), \
        "another save's temporary file was overwritten"
    assert target.read_bytes() == original, "collision changed the published model"


def test_new_model_keeps_normal_creation_permissions(tmp_path):
    control = tmp_path / "normal.json"
    control.write_text("normal file", encoding="utf-8")
    target = tmp_path / "model.json"
    returned = hex_cnn.save_consolidated({"k": np.arange(7)}, {"run": "new"}, str(target))
    assert returned == str(target)
    assert stat.S_IMODE(target.stat().st_mode) == stat.S_IMODE(control.stat().st_mode), \
        "save changed normal umask-based creation permissions"
    np.testing.assert_array_equal(hex_cnn.load_consolidated(str(target))["k"], np.arange(7))


def test_temporary_basename_is_independent_of_model_name(tmp_path, monkeypatch):
    """Observe real opens before JSON failure, including Windows' path-length limit."""
    names, errors = [], []

    def observed_open(path, *args, **kwargs):
        names.append(Path(path).name)
        return builtins.open(path, *args, **kwargs)

    monkeypatch.setattr(hex_cnn, "open", observed_open, raising=False)
    for basename in ("short.json", "model_" + "x" * 220 + ".json"):
        try:
            hex_cnn.save_consolidated({"k": np.arange(7)}, {"invalid": object()},
                                     str(tmp_path / basename))
        except BaseException as exc:
            errors.append(type(exc))
    assert len(names) == 2, "save did not attempt its two real temporary-file opens"
    assert len(names[0]) == len(names[1]), "temporary basename grew with model filename"
    assert errors == [TypeError, TypeError], "long model name broke temporary-file creation"
    assert not list(tmp_path.glob("*.tmp-*")), "failed serialization left temporary files behind"
