# -*- coding: utf-8 -*-
"""test_issue196_camera_calib_gate · #196 第九批：camera_calib 非有限与防 bool 闸（纯标准库）。

背景（issue #196）：
  `lingshu/world/camera_calib.py` 此前对输入只做 `<= 0` / `== 0` 比较，未校验有限性
  与 `bool` 类型：
    - `cross_check(..., claim_focal35_mm=9999.0, tol_frac=float("inf"))` 静默返回 `pass=True`；
    - `cross_check(..., claim_focal35_mm=100.0, tol_frac=True)` 把 `True` 当作 `1.0` 静默通过；
    - `cross_check(..., lens_mm=float("nan"), ...)` / `claim_focal35_mm=nan` 算出并透传 `nan`；
    - `intrinsics_from_spec(..., lens_mm=True, resolution=(True, True))` 构造 1x1/`lens=True` 内参；
    - `crop_factor`/`focal35`/`hfov_deg`/`vfov_deg`/`lens_from_focal35`/`lens_from_hfov` 放行 `nan`/`inf`/`bool`。

守卫内容：
  G1 `cross_check` 的 `tol_frac` 遇 `nan`/`±inf`/`bool`/越 `[0,1]`/非数值 ⇒ `SpecError`（`ValueError` 子类）；
  G2 `cross_check` 的 `lens_mm`/`claim_focal35_mm`/`claim_hfov_deg`/`resolution` 遇 `nan`/`±inf`/`bool` ⇒ `SpecError`；
  G3 `intrinsics_from_spec`、`crop_factor`、`focal35`、`hfov_deg`、`vfov_deg`、`lens_from_focal35`、
     `lens_from_hfov` 与 `CameraIntrinsics.from_dict` 遇 `nan`/`±inf`/`bool` ⇒ `SpecError`；
  G4 合法输入与 `0.0` 物理传感器占位（`from_dict` 兼容 `camera_bridge`）行为不变；
  G5 `--mutate` 定点变异自证：临时副本摘除 `cross_check` 校验 ⇒ 静默 `pass=True` 复现且守卫点红，还原转绿。

运行：
  python -X utf8 -m pytest tests/test_issue196_camera_calib_gate.py -v
  python -X utf8 tests/test_issue196_camera_calib_gate.py
  python -X utf8 tests/test_issue196_camera_calib_gate.py --mutate
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lingshu.world import camera_calib as CC  # noqa: E402


def _assert_spec_error(fn, *args, **kwargs):
    try:
        res = fn(*args, **kwargs)
    except CC.SpecError:
        return
    raise AssertionError(f"期望抛出 SpecError，实际返回 {res!r}")


def test_g1_cross_check_tol_frac_rejects_nonfinite_bool_and_out_of_range():
    """G1：cross_check 的 tol_frac 拒绝 nan / ±inf / bool / 越 [0,1] / 非数值。"""
    for bad_tol in (float("nan"), float("inf"), float("-inf"), True, False, -0.01, 1.01, "0.02"):
        _assert_spec_error(
            CC.cross_check,
            sensor="fullframe",
            lens_mm=50.0,
            resolution=(4000, 3000),
            claim_focal35_mm=9999.0,
            tol_frac=bad_tol,
        )


def test_g2_cross_check_inputs_reject_nonfinite_and_bool():
    """G2：cross_check 的 lens_mm / claim_* / resolution 拒绝 nan / ±inf / bool。"""
    for bad in (float("nan"), float("inf"), float("-inf"), True, False):
        _assert_spec_error(
            CC.cross_check,
            sensor="fullframe",
            lens_mm=bad,
            resolution=(4000, 3000),
            claim_focal35_mm=50.0,
        )
        _assert_spec_error(
            CC.cross_check,
            sensor="fullframe",
            lens_mm=50.0,
            resolution=(4000, 3000),
            claim_focal35_mm=bad,
        )
        _assert_spec_error(
            CC.cross_check,
            sensor="fullframe",
            lens_mm=50.0,
            resolution=(4000, 3000),
            claim_hfov_deg=bad,
        )
    for bad_res in ((True, 100), (100, False), (100.5, 100), (0, 100)):
        _assert_spec_error(
            CC.cross_check,
            sensor="fullframe",
            lens_mm=50.0,
            resolution=bad_res,
            claim_focal35_mm=50.0,
        )


def test_g3_spec_and_conversion_apis_reject_nonfinite_and_bool():
    """G3：全模块规格换算与反序列化公开入口均拒绝 nan / ±inf / bool。"""
    for bad in (float("nan"), float("inf"), float("-inf"), True, False):
        _assert_spec_error(CC.crop_factor, bad, 24.0)
        _assert_spec_error(CC.crop_factor, 36.0, bad)
        _assert_spec_error(CC.focal35, bad, 36.0, 24.0)
        _assert_spec_error(CC.hfov_deg, bad, 36.0)
        _assert_spec_error(CC.vfov_deg, bad, 24.0)
        _assert_spec_error(CC.lens_from_focal35, bad, 36.0, 24.0)
        _assert_spec_error(CC.lens_from_hfov, bad, 36.0)
        _assert_spec_error(
            CC.intrinsics_from_spec,
            sensor="fullframe",
            lens_mm=bad,
            resolution=(4000, 3000),
        )
    _assert_spec_error(
        CC.intrinsics_from_spec,
        sensor="fullframe",
        lens_mm=50.0,
        resolution=(4000, 3000),
        principal_point=(float("nan"), 1500.0),
    )
    _assert_spec_error(
        CC.intrinsics_from_spec,
        sensor="fullframe",
        lens_mm=50.0,
        resolution=(True, 3000),
    )

    valid_doc = CC.intrinsics_from_spec(
        sensor="fullframe", lens_mm=50.0, resolution=(4000, 3000)
    ).to_dict()
    for field_name, bad_val in (
        ("fx", float("nan")),
        ("fy", float("inf")),
        ("cx", True),
        ("cy", float("-inf")),
        ("width", True),
        ("height", False),
        ("lens_mm", float("nan")),
    ):
        poisoned = dict(valid_doc)
        poisoned[field_name] = bad_val
        _assert_spec_error(CC.CameraIntrinsics.from_dict, poisoned)


def test_g4_valid_calls_and_bridge_zero_sensor_placeholder_unchanged():
    """G4：合法规格换算、tol_frac=0.0 边界及 camera_bridge 0.0 传感占位兼容均不变。"""
    ok_rep = CC.cross_check(
        sensor="fullframe",
        lens_mm=50.0,
        resolution=(4000, 3000),
        claim_focal35_mm=50.0,
        tol_frac=0.0,
    )
    assert ok_rep["pass"] is True
    bad_rep = CC.cross_check(
        sensor="fullframe",
        lens_mm=50.0,
        resolution=(4000, 3000),
        claim_hfov_deg=90.0,
    )
    assert bad_rep["pass"] is False
    assert CC.self_check_fullframe()["pass"] is True

    bridge_doc = CC.CameraIntrinsics(
        fx=1000.0, fy=1000.0, cx=500.0, cy=400.0,
        width=1000, height=800, sensor_w_mm=0.0, sensor_h_mm=0.0, lens_mm=0.0,
    ).to_dict()
    restored = CC.CameraIntrinsics.from_dict(bridge_doc)
    assert restored.fx == 1000.0 and restored.lens_mm == 0.0


def run_mutation_self_proof() -> int:
    """G5 `--mutate` 定点变异自证：临时摘除 cross_check 入口有限/bool 校验 ⇒ 静默 pass=True 复现并点红。"""
    src_path = REPO_ROOT / "lingshu" / "world" / "camera_calib.py"
    orig_src = src_path.read_text(encoding="utf-8")

    probe_code = (
        "import sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from lingshu.world import camera_calib as CC\n"
        "for kw in (\n"
        "    {'claim_focal35_mm': 9999.0, 'tol_frac': float('inf')},\n"
        "    {'claim_focal35_mm': 100.0, 'tol_frac': True},\n"
        "    {'claim_focal35_mm': float('nan')},\n"
        "):\n"
        "    try:\n"
        "        rep = CC.cross_check(sensor='fullframe', lens_mm=50.0, resolution=(4000, 3000), **kw)\n"
        "    except CC.SpecError:\n"
        "        continue\n"
        "    print('BYPASS:' + str(kw) + '->pass=' + str(rep['pass']))\n"
        "    raise SystemExit(1)\n"
        "print('ALL_BLOCKED')\n"
    )

    with tempfile.TemporaryDirectory() as tmp:
        pkg_root = Path(tmp) / "repo"
        (pkg_root / "lingshu" / "world").mkdir(parents=True)
        (pkg_root / "lingshu" / "__init__.py").write_text("", encoding="utf-8")
        (pkg_root / "lingshu" / "world" / "__init__.py").write_text("", encoding="utf-8")
        target = pkg_root / "lingshu" / "world" / "camera_calib.py"

        def _run_probe(label: str, expect_rc: int, expect_prefix: str) -> bool:
            proc = subprocess.run(
                [sys.executable, "-X", "utf8", "-c", probe_code, str(pkg_root)],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
            )
            out = (proc.stdout or "").strip()
            err = (proc.stderr or "").strip()
            ok = (proc.returncode == expect_rc and out.startswith(expect_prefix))
            print(f"{label} rc={proc.returncode} out={out!r} VERDICT={'PASS' if ok else 'FAIL'}")
            if not ok and err:
                print(f"  stderr={err}")
            return ok

        target.write_text(orig_src, encoding="utf-8")
        m0 = _run_probe("M0_clean_gated", 0, "ALL_BLOCKED")

        # 定点变异：把 _require_finite 退化为透传（模拟无有限性与防 bool 校验的旧版）
        mutated = re.sub(
            r"def _require_finite\(.*?\n    return f\n",
            "def _require_finite(name: str, val, **_kw) -> float:\n    return float(val) if isinstance(val, bool) else val\n",
            orig_src,
            count=1,
            flags=re.DOTALL,
        )
        assert mutated != orig_src, "变异未命中目标函数"
        target.write_text(mutated, encoding="utf-8")
        m1 = _run_probe("M1_strip_finite_and_bool_gate", 1, "BYPASS:")

        target.write_text(orig_src, encoding="utf-8")
        m2 = _run_probe("M2_restored_gated", 0, "ALL_BLOCKED")

        all_ok = m0 and m1 and m2
        print(f"OVERALL={'PASS' if all_ok else 'FAIL'} (repo_files_untouched=True)")
        return 0 if all_ok else 1


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--mutate" in args:
        return run_mutation_self_proof()
    tests = [
        test_g1_cross_check_tol_frac_rejects_nonfinite_bool_and_out_of_range,
        test_g2_cross_check_inputs_reject_nonfinite_and_bool,
        test_g3_spec_and_conversion_apis_reject_nonfinite_and_bool,
        test_g4_valid_calls_and_bridge_zero_sensor_placeholder_unchanged,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL {fn.__name__}: {exc}")
    print(f"SUMMARY pass={len(tests) - failed} fail={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
