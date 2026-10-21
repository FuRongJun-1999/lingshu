"""#392：真实子进程协议夹具验证桥接文件归属，不冒充 Rust 训练算法验收。"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
import pytest

from lingshu.nn import rust_bridge as rb


@pytest.fixture
def executor(tmp_path, monkeypatch):
    # 夹具仅按现有二进制格式读输入，以样本数/初始向量作为独立的归属见证。
    script = tmp_path / "executor.py"
    script.write_text('''import pathlib, struct, sys
mode, data, plan, vec, out = sys.argv[1:]
if mode == "fail":
    raise SystemExit(17)
n = struct.unpack_from("<Q", pathlib.Path(data).read_bytes(), 8)[0]
steps = struct.unpack_from("<Q", pathlib.Path(plan).read_bytes(), 8)[0]
raw = pathlib.Path(vec).read_bytes()[16:]
vector = list(struct.unpack("<%sd" % (len(raw) // 8), raw))
if mode == "short_curve":
    steps = max(0, steps - 1)
if mode == "short_vector":
    vector = vector[:-1]
values = [float(n)] * steps + vector + [0.0]
pathlib.Path(out).write_bytes(struct.pack("<QQ", 0x4845584E4E4F5554, steps)
    + struct.pack("<%sd" % len(values), *values))
''', encoding="utf-8")
    work = tmp_path / "caller"
    scratch = tmp_path / "scratch"
    work.mkdir()
    scratch.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setenv("AEIS_HEX_NN", sys.executable)
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    original_run = subprocess.run
    launches = []
    launch_lock = threading.Lock()
    barrier = [None]

    def run(command, **kwargs):
        with launch_lock:
            launches.append(tuple(Path(p).resolve() for p in command[2:]))
        if barrier[0] is not None:
            barrier[0].wait(timeout=5)
        return original_run([command[0], "-X", "utf8", str(script), *command[1:]], **kwargs)

    monkeypatch.setattr(rb.subprocess, "run", run)
    return work, scratch, launches, barrier


def train(n=3, marker=11.0, **kwargs):
    plan = rb.make_plan(n, steps=2, batch=2, samples=1, seed=7)
    return rb.run_rust_train(np.zeros((n, 3, 3, 3)), np.zeros(n, dtype=int),
        np.zeros(n, dtype=int), np.full(375, marker), plan, **kwargs)


def test_concurrent_default_calls_keep_their_own_inputs(executor):
    _, scratch, launches, barrier = executor
    barrier[0] = threading.Barrier(2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(train, n, marker) for n, marker in ((3, 11.0), (5, 29.0))]
        results = [future.result(timeout=8) for future in futures]
    for (n, marker), (curve, vector, _) in zip(((3, 11.0), (5, 29.0)), results):
        assert curve.tolist() == [float(n)] * 2, "training result belongs to another call"
        assert vector.tolist() == [marker] * 375, "training vector belongs to another call"
    assert launches[0][0].parent != launches[1][0].parent, "default training workspace is shared"
    assert not list(scratch.iterdir()), "owned training workspace was not cleaned"


def test_default_call_preserves_caller_files(executor):
    work, scratch, _, _ = executor
    existing = work / "data/rust_bridge"
    existing.mkdir(parents=True)
    before = {name: ("caller-owned-" + name).encode() for name in ("data.bin", "plan.bin", "vec.bin", "out.bin")}
    for name, value in before.items():
        (existing / name).write_bytes(value)
    train()
    assert {p.name: p.read_bytes() for p in existing.iterdir()} == before, "default call overwrote caller files"
    assert not list(scratch.iterdir()), "owned training workspace was not cleaned"


def test_failed_child_cleans_only_its_default_workspace(executor):
    work, scratch, _, _ = executor
    with pytest.raises(subprocess.CalledProcessError):
        train(mode="fail")
    assert not (work / "data").exists(), "failed default call left files in caller cwd"
    assert not list(scratch.iterdir()), "failed training workspace was not cleaned"


@pytest.mark.parametrize("mode", ["short_curve", "short_vector"])
def test_output_must_match_this_call_dimensions(executor, mode):
    _, scratch, _, _ = executor
    with pytest.raises(ValueError, match="本次输入"):
        train(mode=mode)
    assert not list(scratch.iterdir()), "invalid-output training workspace was not cleaned"


def test_explicit_workdir_retains_diagnostic_files(executor, tmp_path):
    _, scratch, _, _ = executor
    target = tmp_path / "explicit"
    curve, vector, _ = train(workdir=target)
    assert curve.tolist() == [3.0, 3.0] and vector.size == 375
    assert target.is_dir(), "explicit workdir files were not retained"
    assert {p.name for p in target.iterdir()} == {"data.bin", "plan.bin", "vec.bin", "out.bin"}, "explicit workdir files were not retained"
    assert not list(scratch.iterdir())


def test_explicit_failed_workdir_keeps_inputs(executor, tmp_path):
    target = tmp_path / "explicit-failed"
    with pytest.raises(subprocess.CalledProcessError):
        train(workdir=target, mode="fail")
    assert {p.name for p in target.iterdir()} == {"data.bin", "plan.bin", "vec.bin"}, "explicit failure removed diagnostic inputs"
