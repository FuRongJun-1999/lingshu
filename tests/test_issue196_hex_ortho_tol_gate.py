# -*- coding: utf-8 -*-
"""test_issue196_hex_ortho_tol_gate · hex_ortho 公开入口参数闸与 NaN 核防串扰守卫
============================================================================
缺陷背景（#196 第十一批 · `lingshu/nn/hex_ortho.py`）：
  `residual_pyramid` 与 `pyramid_report` 已在 #2132/#2521 补齐参数与非有限闸，
  但同模块的公开入口 `gram_schmidt_kernels`、`orthogonal_pool`、`kname` 以及
  `residual_pyramid(..., use_orthogonal_pool=...)` 仍存在四处静默绕过：
    1. `gram_schmidt_kernels(kernels, tol=...)` / `orthogonal_pool(kernels, tol=...)`
       完全未校验 `tol`：
         - `tol=nan` / `-inf` / `-0.5` / `False` 使 `ratio < tol` 恒 False，
           12 个核（含 7 个同轴/重复核）静默全收（dropped=0）；
         - `tol=+inf` 使 `ratio < inf` 恒 True，全部 12 个核被无依据判成冗余（dropped=12）；
         - `tol=True` 按 `1.0` 静默比较，正交基核亦被误剔（dropped=8/12）。
    2. `gram_schmidt_kernels([nan_kernel, valid_kernel])` 中 `nan_kernel` 因
       `redundant=False` 进入已保留基底 `kept`，把随后所有合法核的 `new_info_ratio`
       也污染成 `None`，`orthogonal_pool` 亦把下标 0（NaN 核）放进候选池 `[0, 1]`。
    3. `residual_pyramid(..., use_orthogonal_pool="false")` 按 truthy 字符串静默走
       `True` 分支。
    4. `kname(True)` / `kname(-1)` 静默按 `nm[1]` / `nm[-1]` 索引。

断言组（G1–G4）：
  - G1：`gram_schmidt_kernels` / `orthogonal_pool` 对 `tol` 为 `NaN`/`±inf`/`bool`/
    负值抛 `ValueError`，非实数类型（如 `str`/`None`）抛 `TypeError`（含空核列表）。
  - G2：`[nan_kernel, valid_kernel]` 下 `nan_kernel` 保持 `new_info_ratio=None,
    redundant=False`（不回归 #2132），且不污染后续合法核；`orthogonal_pool` 返回
    下标不含 NaN 核。
  - G3：`residual_pyramid(..., use_orthogonal_pool=...)` 对非 `bool` 实参 fail-closed
    返回 `stop_reason="invalid_input"`；`kname` 对 `bool`/负数抛 `ValueError`、非整数抛
    `TypeError`。
  - G4：正对照——`extended_prior_family(12)` 默认 `tol=0.20` 仍精确留 5 删 7，
    `residual_pyramid` 合法调用与 `dead_zone` 停因读数不变。

运行方式：
  - pytest：`python -X utf8 -m pytest -q tests/test_issue196_hex_ortho_tol_gate.py`
  - 直跑  ：`python -X utf8 tests/test_issue196_hex_ortho_tol_gate.py`
  - 变异自证：`python -X utf8 tests/test_issue196_hex_ortho_tol_gate.py --mutate`
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.append(str(_REPO))

from lingshu.nn.hex_cnn import extended_prior_family  # noqa: E402
from lingshu.nn import hex_ortho as HO  # noqa: E402


def _assert_raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type:
        return
    except Exception as exc:  # pragma: no cover
        raise AssertionError(
            "期望抛出 %s，实际抛出 %s: %s" % (exc_type.__name__, type(exc).__name__, exc)
        ) from exc
    raise AssertionError("期望抛出 %s，但调用静默返回" % exc_type.__name__)


def test_g1_gram_schmidt_and_pool_tol_gate():
    """G1：gram_schmidt_kernels / orthogonal_pool 拦截非法 tol（含空核表）。"""
    ks = extended_prior_family(12)
    unit = [np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])]
    bad_values = [
        float("nan"),
        float("inf"),
        float("-inf"),
        -0.5,
        -1e-9,
        True,
        False,
        np.bool_(True),
        np.float64("nan"),
    ]
    for bad in bad_values:
        _assert_raises(ValueError, HO.gram_schmidt_kernels, ks, tol=bad)
        _assert_raises(ValueError, HO.orthogonal_pool, ks, tol=bad)
        _assert_raises(ValueError, HO.gram_schmidt_kernels, [], tol=bad)
        _assert_raises(ValueError, HO.orthogonal_pool, unit, tol=bad)
    for bad_type in ("0.2", None, 1 + 2j, [0.2]):
        _assert_raises(TypeError, HO.gram_schmidt_kernels, ks, tol=bad_type)
        _assert_raises(TypeError, HO.orthogonal_pool, ks, tol=bad_type)


def test_g2_nan_kernel_no_poison_and_pool_exclusion():
    """G2：NaN/inf 核不破坏 #2132 口径、不污染后续合法核，且不进入 orthogonal_pool。"""
    nan_k = np.full(7, float("nan"))
    inf_k = np.array([float("inf"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    valid_a = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    valid_b = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    dup_a = np.array([-2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    rep = HO.gram_schmidt_kernels([nan_k, valid_a, dup_a, inf_k, valid_b])
    assert rep[0]["new_info_ratio"] is None and rep[0]["redundant"] is False
    assert rep[1]["new_info_ratio"] == 1.0 and rep[1]["redundant"] is False
    assert rep[2]["new_info_ratio"] == 0.0 and rep[2]["redundant"] is True
    assert rep[3]["new_info_ratio"] is None and rep[3]["redundant"] is False
    assert rep[4]["new_info_ratio"] == 1.0 and rep[4]["redundant"] is False

    keep, _ = HO.orthogonal_pool([nan_k, valid_a, dup_a, inf_k, valid_b])
    assert keep == [1, 4], "NaN/inf 核与重复核均不得进入正交候选池，实得 %r" % (keep,)


def test_g3_residual_pyramid_and_kname_bool_gate():
    """G3：residual_pyramid.use_orthogonal_pool 与 kname 下标类型/范围闸。"""
    ks = extended_prior_family(12)
    lat = np.ones((24, 24, 3), dtype=float) * 0.1
    for bad_flag in ("false", "true", 1, 0, None, float("nan")):
        r = HO.residual_pyramid(lat, ks, use_orthogonal_pool=bad_flag)
        assert r["stop_reason"] == "invalid_input", (bad_flag, r.get("stop_reason"))
        assert "use_orthogonal_pool" in r.get("error", "")

    for bad_idx in (True, False, np.bool_(True), -1, -5):
        _assert_raises(ValueError, HO.kname, bad_idx)
    for bad_type in (1.5, "0", None):
        _assert_raises(TypeError, HO.kname, bad_type)


def test_g4_valid_paths_unchanged():
    """G4 正对照：合法参数路径与默认读数完全一致。"""
    ks = extended_prior_family(12)
    keep, rep = HO.orthogonal_pool(ks)
    assert len(keep) == 5 and len(rep) == 12
    assert sum(1 for r in rep if r["redundant"]) == 7
    keep_np, _ = HO.orthogonal_pool(ks, tol=np.float32(0.2))
    assert keep_np == keep

    lat = np.ones((24, 24, 3), dtype=float) * 0.1
    r = HO.residual_pyramid(lat, ks, use_orthogonal_pool=True)
    assert r["stop_reason"] == "dead_zone"
    assert r["orthogonality"]["dropped"] == 7
    r_off = HO.residual_pyramid(lat, ks, use_orthogonal_pool=False)
    assert r_off["stop_reason"] == "dead_zone"
    assert r_off["orthogonality"]["dropped"] == 0
    assert HO.kname(0) == "edge_d0" and HO.kname(np.int64(6)) == "smooth"


_TESTS = [
    test_g1_gram_schmidt_and_pool_tol_gate,
    test_g2_nan_kernel_no_poison_and_pool_exclusion,
    test_g3_residual_pyramid_and_kname_bool_gate,
    test_g4_valid_paths_unchanged,
]


def _run_direct() -> int:
    failed = 0
    for fn in _TESTS:
        try:
            fn()
            print("  PASS", fn.__name__)
        except Exception as exc:
            failed += 1
            print("  FAIL", fn.__name__, ":", exc)
    print("SUMMARY pass=%d fail=%d" % (len(_TESTS) - failed, failed))
    return 1 if failed else 0


def _run_subtest(temp_root: Path) -> tuple[int, list[str]]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(temp_root) + os.pathsep + str(_REPO)
    proc = subprocess.run(
        [sys.executable, "-X", "utf8", str(Path(__file__).resolve())],
        cwd=str(temp_root),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    failed = [
        line.split()[1]
        for line in proc.stdout.splitlines()
        if line.strip().startswith("FAIL ")
    ]
    return proc.returncode, failed


def _run_mutate() -> int:
    src_file = _REPO / "lingshu" / "nn" / "hex_ortho.py"
    orig_bytes = src_file.read_bytes()
    orig_sha = hashlib.sha256(orig_bytes).hexdigest()
    orig_text = orig_bytes.decode("utf-8").replace("\r\n", "\n")

    mut_m1 = orig_text.replace("    tol_val = _validate_tol(tol)\n", "    tol_val = tol\n", 1)
    assert mut_m1 != orig_text, "M1 变异点未命中"

    mut_m2 = orig_text.replace(
        "        if not redundant and ratio is not None:\n",
        "        if not redundant:\n",
        1,
    ).replace(
        "    keep = [r[\"idx\"] for r in report\n"
        "            if not r[\"redundant\"] and r[\"new_info_ratio\"] is not None]\n",
        "    keep = [r[\"idx\"] for r in report if not r[\"redundant\"]]\n",
        1,
    ).replace(
        "    if not isinstance(use_orthogonal_pool, (bool, np.bool_)):\n",
        "    if False:\n",
        1,
    )
    assert mut_m2 != orig_text, "M2 变异点未命中"

    with tempfile.TemporaryDirectory(prefix="hex_ortho_mut_") as td:
        tmp_pkg = Path(td) / "lingshu"
        shutil.copytree(
            _REPO / "lingshu",
            tmp_pkg,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "assets"),
        )
        target = tmp_pkg / "nn" / "hex_ortho.py"

        cases = [
            ("M0_clean_baseline", orig_text, 0, []),
            ("M1_strip_tol_gate", mut_m1, 1, ["test_g1_gram_schmidt_and_pool_tol_gate"]),
            (
                "M2_strip_nan_and_pool_guards",
                mut_m2,
                1,
                [
                    "test_g2_nan_kernel_no_poison_and_pool_exclusion",
                    "test_g3_residual_pyramid_and_kname_bool_gate",
                ],
            ),
            ("M3_restored_baseline", orig_text, 0, []),
        ]
        all_ok = True
        for label, code, exp_rc, exp_failed in cases:
            target.write_text(code, encoding="utf-8", newline="\n")
            rc, failed = _run_subtest(Path(td))
            verdict = rc == exp_rc and failed == exp_failed
            all_ok = all_ok and verdict
            print(
                "  %-28s rc=%d(expect=%d) failed=%s(expect=%s) VERDICT=%s"
                % (label, rc, exp_rc, failed, exp_failed, "PASS" if verdict else "FAIL")
            )

    after_sha = hashlib.sha256(src_file.read_bytes()).hexdigest()
    untouched = orig_sha == after_sha
    print("OVERALL=%s (repo_files_untouched=%s)" % ("PASS" if (all_ok and untouched) else "FAIL", untouched))
    return 0 if (all_ok and untouched) else 1


if __name__ == "__main__":
    if "--mutate" in sys.argv[1:]:
        sys.exit(_run_mutate())
    sys.exit(_run_direct())
