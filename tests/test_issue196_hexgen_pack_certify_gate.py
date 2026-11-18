# -*- coding: utf-8 -*-
"""test_issue196_hexgen_pack_certify_gate · #196 第十四批（lingshu/gen/hexgen_pack_certify.py）
============================================================================================
守卫对象：`lingshu/gen/hexgen_pack_certify.py` 全部公开与辅助入口的有限性、防 bool 与边界门禁：
  1. `legal_mask` / `_rank_ok_vec`：
     - 拒绝 `r` 为 `NaN` / `±inf` / `<= 0` / `bool` / `np.bool_` / 非数值（修前 `r=nan/inf` 静默产出全 0 掩膜，`r=True/-5` 静默产出 29241 像素掩膜）。
     - 拒绝 `size` 为 `<= 0` / `bool` / `np.bool_` / `float` / `NaN`。
     - 拒绝 `zone_calib` 含非有限切点、`bool` 切点、越出 `[0, 1]`、逆序 `[0.8, 0.2]` 或缺失字段（修前 `cuts=[nan, 0.6]` 静默落入第 2 带产出 40810 像素错掩膜）。
     - 拒绝非法 `cells`（空、非 `'r0'..'r8'`、字符串本身）与未知 `shape`。
  2. `support_vec` / `slack_vec`：
     - 拒绝 `r` / `sep` / `pad` 为非有限/`bool`/越界，拒绝 `dx` / `dy` / `dist` 含 `NaN`/`±inf`/`bool`（含标量与数组），拒绝各向异性形状 `dist <= 0`（保留 `test_gen_import_purity` 的 `support_vec('triangle', 10, 0.0, 0.0, 0.0)` 契约）。
  3. `certify_radius` / `certify_p1` / `witness_at`：
     - 拒绝 `r=nan/inf/<=0/True` 等非法输入（修前 `certify_radius(..., r=nan)` 静默产出假「无解证书」`{'feasible': False, 'n_p1': 0}`，`r=True/-10` 静默产出 `feasible=True`）。
     - 拒绝 `n` 为 `bool` 或不在 `{2, 3}`、`max_p1` 为 `0`/`-1`/`True`/`nan`、`verbose` 非 `bool`、`x1/y1` 为 `bool`/`nan`/`float`、含 `NaN`/`inf` 或非 2D 的浮点 `mask`（修前 `np.array([[nan]], bool)` 把 `NaN` 静默转成 `True`）。
  4. `check_witness`：
     - 拒绝 `r=nan/True/-5`、`n=True/0/1`、`len(pts) != n`、`pts` 含 `NaN`/`±inf`/`bool`（修前 `r=nan` 返回 `(False, inf)`，`r=True` 返回 `(True, ...)`）。
  5. 正对照：合法输入（含标称 `r=40`、无解大半径 `r=300` `#353`、`"stripe"` `#2131`、有效 `zone_calib`）结果与改前逐位一致。

运行：
  pytest tests/test_issue196_hexgen_pack_certify_gate.py -q
  python -X utf8 tests/test_issue196_hexgen_pack_certify_gate.py
  python -X utf8 tests/test_issue196_hexgen_pack_certify_gate.py --mutate
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lingshu.gen import hexgen_pack_certify as P  # noqa: E402
from lingshu.gen import hexgen_self_source as S   # noqa: E402


def _assert_raises(exc_type, fn, *args, **kwargs):
    try:
        res = fn(*args, **kwargs)
    except exc_type:
        return
    except Exception as other:  # pragma: no cover - 诊断路径
        raise AssertionError(
            f"expected {exc_type.__name__}, got {type(other).__name__}: {other}"
        ) from other
    raise AssertionError(f"expected {exc_type.__name__}, got return value {res!r}")


_BAD_POS_REALS = (
    float("nan"),
    float("inf"),
    float("-inf"),
    0.0,
    -1.0,
    -10.0,
    True,
    False,
    np.bool_(True),
    np.float64("nan"),
    np.float64("inf"),
    "40",
    None,
)

_BAD_INTS = (
    True,
    False,
    np.bool_(True),
    float("nan"),
    float("inf"),
    1.5,
    "1",
    None,
)


def test_g1_legal_mask_and_rank_ok_gates():
    """legal_mask 与 _rank_ok_vec 对 r / size / cuts / cells / shape 严格门禁。"""
    for bad_r in _BAD_POS_REALS:
        _assert_raises(ValueError, P.legal_mask, ("r4",), "circle", bad_r, None)

    for bad_size in (0, -512, True, False, np.bool_(True), float("nan"), 64.5, "64", None):
        _assert_raises(ValueError, P.legal_mask, ("r4",), "circle", 10, None, bad_size)
        _assert_raises(ValueError, P._rank_ok_vec, None, ("r4",), bad_size)

    bad_calibs = (
        {"row": {"cuts": [float("nan"), 0.6]}, "col": {"cuts": [0.3, 0.6]}},
        {"row": {"cuts": [0.3, float("inf")]}, "col": {"cuts": [0.3, 0.6]}},
        {"row": {"cuts": [True, 0.6]}, "col": {"cuts": [0.3, 0.6]}},
        {"row": {"cuts": [0.7, 0.3]}, "col": {"cuts": [0.3, 0.6]}},
        {"row": {"cuts": [-0.1, 0.6]}, "col": {"cuts": [0.3, 0.6]}},
        {"row": {"cuts": [0.3, 1.1]}, "col": {"cuts": [0.3, 0.6]}},
        {"row": {"cuts": [0.3]}, "col": {"cuts": [0.3, 0.6]}},
        {"row": {"cuts": None}, "col": {"cuts": [0.3, 0.6]}},
        {"row": {"cuts": [0.3, 0.6]}},
        "bad-calib",
    )
    for bad_zc in bad_calibs:
        _assert_raises(ValueError, P.legal_mask, ("r4",), "circle", 40, bad_zc)
        _assert_raises(ValueError, P._rank_ok_vec, bad_zc, ("r4",))

    for bad_cells in ((), "r4", ("r9",), ("r-1",), ("x4",), (True,), None):
        _assert_raises(ValueError, P.legal_mask, bad_cells, "circle", 40, None)

    for bad_shape in ("unknown_shape", "", 123, True, None):
        _assert_raises(ValueError, P.legal_mask, ("r4",), bad_shape, 40, None)


def test_g2_support_vec_and_slack_vec_gates():
    """support_vec 与 slack_vec 拒绝 bool/非有限/负距离/未知形状，保留合法调用契约。"""
    for bad_r in _BAD_POS_REALS:
        _assert_raises(ValueError, P.support_vec, "square", bad_r, 1.0, 1.0, math.sqrt(2.0))
        _assert_raises(ValueError, P.slack_vec, "square", bad_r, np.array([1.0]), np.array([1.0]))

    for bad_val in (float("nan"), float("inf"), float("-inf"), True, False, np.bool_(False)):
        _assert_raises(ValueError, P.support_vec, "square", 20.0, bad_val, 1.0, 1.0)
        _assert_raises(ValueError, P.support_vec, "square", 20.0, 1.0, bad_val, 1.0)
        _assert_raises(ValueError, P.support_vec, "square", 20.0, 1.0, 1.0, bad_val)
        _assert_raises(ValueError, P.slack_vec, "square", 20.0, [1.0, bad_val], [1.0, 1.0])
        _assert_raises(ValueError, P.slack_vec, "square", 20.0, [1.0], [bad_val])

    _assert_raises(ValueError, P.support_vec, "circle", 20.0, 1.0, 1.0, -0.5)
    _assert_raises(ValueError, P.support_vec, "square", 20.0, 0.0, 0.0, 0.0)
    _assert_raises(ValueError, P.support_vec, "rectangle", 20.0, 0.0, 0.0, 0.0)
    _assert_raises(ValueError, P.support_vec, "diamond", 20.0, 0.0, 0.0, 0.0)
    _assert_raises(ValueError, P.support_vec, "square", 20.0, np.array([True]), np.array([1.0]), np.array([1.0]))
    _assert_raises(ValueError, P.slack_vec, "square", 20.0, np.array([True]), np.array([1.0]))

    for bad_sp in (float("nan"), float("inf"), -1.0, True, False):
        _assert_raises(ValueError, P.slack_vec, "square", 20.0, np.array([100.0]), np.array([0.0]), sep=bad_sp)
        _assert_raises(ValueError, P.slack_vec, "square", 20.0, np.array([100.0]), np.array([0.0]), pad=bad_sp)

    # 保留 test_gen_import_purity 既定契约：triangle 在 dist=0.0 可调用且值正确
    tri = P.support_vec("triangle", 10, 0.0, 0.0, 0.0)
    assert math.isclose(float(tri), S.paint_extent("triangle", 10), rel_tol=0.0, abs_tol=1e-12)


def test_g3_certify_and_witness_entry_gates():
    """certify_radius / certify_p1 / witness_at 对非有限、bool、坏 mask 全量 fail-loud。"""
    for bad_r in _BAD_POS_REALS:
        _assert_raises(ValueError, P.certify_radius, ("r4",), 2, "circle", bad_r, None, max_p1=2)
        _assert_raises(ValueError, P.certify_p1, 250, 250, ("r4",), 2, "circle", bad_r, None)
        _assert_raises(ValueError, P.witness_at, ("r4",), 2, "circle", bad_r, None)

    for bad_n in (True, False, np.bool_(True), 0, 1, 4, -2, float("nan"), 2.0, "2"):
        _assert_raises(ValueError, P.certify_radius, ("r4",), bad_n, "circle", 40, None, max_p1=2)
        _assert_raises(ValueError, P.certify_p1, 250, 250, ("r4",), bad_n, "circle", 40, None)
        _assert_raises(ValueError, P.witness_at, ("r4",), bad_n, "circle", 40, None)

    for bad_mp in (0, -1, True, False, np.bool_(True), float("nan"), 1.5, "2"):
        _assert_raises(ValueError, P.certify_radius, ("r4",), 2, "circle", 40, None, max_p1=bad_mp)

    for bad_v in (0, 1, "yes", None):
        _assert_raises(ValueError, P.certify_radius, ("r4",), 2, "circle", 40, None, verbose=bad_v, max_p1=1)

    for bad_xy in _BAD_INTS:
        _assert_raises(ValueError, P.certify_p1, bad_xy, 250, ("r4",), 2, "circle", 40, None)
        _assert_raises(ValueError, P.certify_p1, 250, bad_xy, ("r4",), 2, "circle", 40, None)

    nan_mask = np.array([[float("nan"), 0.0], [0.0, 1.0]], dtype=float)
    bad_val_mask = np.array([[0, 2], [0, 1]], dtype=int)
    bad_dim_mask = np.array([True, False], dtype=bool)
    for m in (nan_mask, bad_val_mask, bad_dim_mask):
        _assert_raises(ValueError, P.certify_radius, ("r4",), 2, "circle", 40, None, mask=m)
        _assert_raises(ValueError, P.certify_p1, 0, 0, ("r4",), 2, "circle", 40, None, mask=m)
        _assert_raises(ValueError, P.witness_at, ("r4",), 2, "circle", 40, None, mask=m)


def test_g4_check_witness_gates():
    """check_witness 拒绝非有限/bool/错长度 pts 与非法 r/n，消除 (False, inf) 假读数。"""
    pts2 = [(220, 220), (320, 220)]
    for bad_r in _BAD_POS_REALS:
        _assert_raises(ValueError, P.check_witness, pts2, ("r4",), 2, "circle", bad_r, None)

    for bad_n in (True, False, 0, 1, 3, 4, float("nan")):
        _assert_raises(ValueError, P.check_witness, pts2, ("r4",), bad_n, "circle", 20, None)

    for bad_pts in (
        [(float("nan"), 250), (260, 250)],
        [(float("inf"), 250), (260, 250)],
        [(True, 250), (260, 250)],
        [(250, False), (260, 250)],
        [(250,), (260, 250)],
        [(250, 250)],
        None,
        "pts",
    ):
        _assert_raises(ValueError, P.check_witness, bad_pts, ("r4",), 2, "circle", 20, None)


def test_g5_valid_inputs_preserve_exact_semantics():
    """合法输入（含大半径空掩膜 #353、stripe #2131、标定 cuts、check_witness）行为逐位保持。"""
    m4 = P.legal_mask(("r4",), "circle", 40, None, 512)
    assert int(m4.sum()) == 171 * 171

    zc = {"row": {"cuts": [0.25, 0.75]}, "col": {"cuts": [0.3, 0.7]}}
    m_zc = P.legal_mask(("r4",), "circle", 40, zc, 512)
    assert int(m_zc.sum()) > 0

    # #353：合法但放不下的大半径 r=300 保持返回「无解」而非抛 ValueError
    r300 = P.certify_radius(("r4",), 3, "circle", 300, None)
    assert r300 == {"feasible": False, "max_slack": float("-inf"), "best_p1": None, "n_p1": 0, "per_p1": []}
    assert P.certify_p1(10, 10, ("r4",), 3, "circle", 300, None) == (float("-inf"), 0)
    assert P.witness_at(("r4",), 3, "circle", 300, None) is None

    # 可行 2 件解 + check_witness 对拍
    got = P.certify_radius(("r4",), 2, "circle", 20, None, max_p1=3)
    assert got["feasible"] is True and got["n_p1"] == 3
    w = P.witness_at(("r4",), 2, "circle", 20, None)
    assert w is not None and len(w) == 2
    ok, sl = P.check_witness(w, ("r4",), 2, "circle", 20, None)
    assert ok is True and math.isfinite(sl) and sl >= 0.0

    # #2131 stripe 归一一致
    sv_stripe = float(P.support_vec("stripe", 40, 1.0, 1.0, math.sqrt(2.0)))
    sv_square = float(P.support_vec("square", 40, 1.0, 1.0, math.sqrt(2.0)))
    assert sv_stripe == sv_square


_TESTS = (
    test_g1_legal_mask_and_rank_ok_gates,
    test_g2_support_vec_and_slack_vec_gates,
    test_g3_certify_and_witness_entry_gates,
    test_g4_check_witness_gates,
    test_g5_valid_inputs_preserve_exact_semantics,
)


def _run_all():
    passed = 0
    for fn in _TESTS:
        fn()
        passed += 1
        print(f"PASS {fn.__name__}")
    print(f"SUMMARY pass={passed} fail=0")
    return 0


def _run_in_temp_repo(tmp_dir: Path, mutator=None):
    dst_lingshu = tmp_dir / "lingshu"
    shutil.copytree(REPO / "lingshu", dst_lingshu)
    target = dst_lingshu / "gen" / "hexgen_pack_certify.py"
    if mutator is not None:
        src = target.read_text(encoding="utf-8")
        mutated = mutator(src)
        assert mutated != src, "mutator produced no diff"
        target.write_text(mutated, encoding="utf-8")
    dst_test = tmp_dir / "test_gate_copy.py"
    shutil.copy2(Path(__file__), dst_test)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tmp_dir)
    env["PYTHONUTF8"] = "1"
    return subprocess.run(
        [sys.executable, "-X", "utf8", str(dst_test)],
        cwd=str(tmp_dir),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )


def _mut_strip_pos_finite(src: str) -> str:
    old = 'def _require_pos_finite(val, name):\n    """要求严格正有限实数（> 0），拒绝 bool / 非有限 / <= 0。"""\n    out = _require_finite_real(val, name)\n    if out <= 0.0:\n        raise ValueError(f"{name} must be > 0, got {val!r}")\n    return out'
    new = 'def _require_pos_finite(val, name):\n    return float(val)'
    assert old in src
    return src.replace(old, new, 1)


def _mut_strip_zone_calib(src: str) -> str:
    target = 'def _validate_zone_calib(zone_calib):\n    """校验方位标定切点（None/空映射回落精确三等分；否则要求有限非 bool 且 0<=c0<=c1<=1）。"""\n    if not zone_calib:\n        return [1 / 3.0, 2 / 3.0], [1 / 3.0, 2 / 3.0]'
    rep = 'def _validate_zone_calib(zone_calib):\n    if not zone_calib:\n        return [1 / 3.0, 2 / 3.0], [1 / 3.0, 2 / 3.0]\n    return list(zone_calib["row"]["cuts"]), list(zone_calib["col"]["cuts"])'
    assert target in src
    return src.replace(target, rep, 1)


def _mut_strip_mask_and_pts(src: str) -> str:
    old = 'def _validate_mask(mask):'
    new = 'def _validate_mask_unused(mask):'
    assert old in src
    src2 = src.replace(old, new, 1).replace(
        "L = legal_mask(cells, shape, r, zone_calib, size) if mask is None else _validate_mask(mask)",
        "L = legal_mask(cells, shape, r, zone_calib, size) if mask is None else np.array(mask, bool)",
    )
    return src2.replace(
        "cells, n, shape, r, size = _validate_cert_inputs(cells, n, shape, r, zone_calib, size)",
        "cells = tuple(cells)",
        1,
    )


def _run_mutate_self_proof() -> int:
    mutations = (
        ("M1_strip_pos_finite_gate", _mut_strip_pos_finite),
        ("M2_strip_zone_calib_gate", _mut_strip_zone_calib),
        ("M3_strip_mask_and_cert_gates", _mut_strip_mask_and_pts),
    )
    with tempfile.TemporaryDirectory(prefix="lingshu-196-mut-") as td:
        root = Path(td)
        base_dir = root / "clean"
        base = _run_in_temp_repo(base_dir)
        assert base.returncode == 0, f"clean baseline failed:\n{base.stdout}\n{base.stderr}"
        print("M0_clean_baseline PASS rc=0")
        for idx, (label, mut) in enumerate(mutations, 1):
            mdir = root / f"mut_{idx}"
            proc = _run_in_temp_repo(mdir, mut)
            assert proc.returncode != 0, f"{label} unexpectedly passed!\n{proc.stdout}"
            print(f"{label} PASS caught rc={proc.returncode}")
        restored = _run_in_temp_repo(root / "restored")
        assert restored.returncode == 0, f"restored baseline failed:\n{restored.stdout}\n{restored.stderr}"
        print("M4_restored_baseline PASS rc=0")
    print("OVERALL=PASS (repo_files_untouched=True)")
    return 0


if __name__ == "__main__":
    if "--mutate" in sys.argv[1:]:
        raise SystemExit(_run_mutate_self_proof())
    raise SystemExit(_run_all())
