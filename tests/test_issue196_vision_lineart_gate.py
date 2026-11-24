#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_issue196_vision_lineart_gate · #196 lineart 非有限阈值与参数类型闸守卫
=============================================================================
缺陷（main @ d78185c）：
    `lingshu/vision/lineart.py` 的 `darkline` / `linework` 计算阈值时执行
    `thr = max(_THR_FLOOR, float(mag.mean()) + float(mag.std()) * float(thr_mult))`：
      · `thr_mult=float('nan')` / `float('-inf')` 时 `max(40.0, nan|'-inf')`
        直接静默返回 `_THR_FLOOR=40.0`，产出看似正常的线稿而不报错；
      · `thr_mult=float('inf')` 静默输出全白图；
      · `thr_mult=True/False`、`thr_mult < 0`、`white_bg="false"` 非 bool、
        非 `PIL.Image.Image` 或 0 尺寸图像均缺乏 fail-loud 校验。

断言组：
    G1 非有限 `thr_mult`（`nan` / `±inf` / `np.nan` / `np.inf`）在 `darkline` 与
       `linework` 均抛 `ValueError`；
    G2 `bool` / `np.bool_` / 负 `thr_mult` 抛 `ValueError`，非实数类型（字符串/None）抛 `TypeError`；
    G3 非 `bool` `white_bg` 与非法/0 尺寸 `img` 在两条入口均 fail-loud；
    G4 正对照：`np.float64(1.5)`、`thr_mult=0.0` 及 `Silhouette3D.render()` 线稿提取
       与既有像素参考公式逐像素一致（行为零回归）；
    G5 `--mutate` 定点变异自证：回退阈值或类型闸 ⇒ 对应断言精确点红，还原全绿。

运行方式（仓根）：
    python -X utf8 tests/test_issue196_vision_lineart_gate.py
    python -X utf8 tests/test_issue196_vision_lineart_gate.py --mutate
    python -X utf8 -m pytest tests/test_issue196_vision_lineart_gate.py -q --no-header
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lingshu.vision.lineart import darkline, linework  # noqa: E402
from lingshu.world.silhouette3d import fatfish_skinned  # noqa: E402

LINEART_PATH = REPO / "lingshu" / "vision" / "lineart.py"
RUN_TIMEOUT = 120


def _sample_disc() -> Image.Image:
    a = np.full((32, 32, 3), (250, 245, 240), np.uint8)
    a[8:24, 8:24] = (40, 60, 120)
    return Image.fromarray(a, "RGB")


def _assert_raises(exc_types, fn, msg: str) -> None:
    try:
        fn()
    except exc_types:
        return
    except Exception as exc:
        raise AssertionError("%s：期望 %r，实得 %s: %s" % (msg, exc_types, type(exc).__name__, exc)) from exc
    raise AssertionError("%s：未抛异常而静默返回" % (msg,))


def test_g1_nonfinite_thr_mult_rejected() -> None:
    """G1：darkline 与 linework 拒绝 NaN / ±inf / np.nan / np.inf 阈值倍数。"""
    im = _sample_disc()
    for bad in (float("nan"), float("inf"), float("-inf"), np.nan, np.inf, -np.inf):
        _assert_raises(ValueError, lambda b=bad: darkline(im, thr_mult=b), "darkline thr_mult=%r" % (bad,))
        _assert_raises(ValueError, lambda b=bad: linework(im, thr_mult=b), "linework thr_mult=%r" % (bad,))


def test_g2_bool_negative_and_bad_type_thr_mult_rejected() -> None:
    """G2：bool / np.bool_ / 负值抛 ValueError，非数值类型抛 TypeError。"""
    im = _sample_disc()
    for bad in (True, False, np.bool_(True), -0.01, -5.0):
        _assert_raises(ValueError, lambda b=bad: darkline(im, thr_mult=b), "darkline thr_mult=%r" % (bad,))
        _assert_raises(ValueError, lambda b=bad: linework(im, thr_mult=b), "linework thr_mult=%r" % (bad,))
    for bad in ("1.5", None, object()):
        _assert_raises(TypeError, lambda b=bad: darkline(im, thr_mult=b), "darkline bad type=%r" % (bad,))  # type: ignore[arg-type]
        _assert_raises(TypeError, lambda b=bad: linework(im, thr_mult=b), "linework bad type=%r" % (bad,))  # type: ignore[arg-type]


def test_g3_invalid_white_bg_and_image_rejected() -> None:
    """G3：非 bool white_bg 与非 PIL.Image.Image / 0 尺寸图像 fail-loud。"""
    im = _sample_disc()
    for bad_bg in ("false", 1, 0, None):
        _assert_raises(TypeError, lambda bg=bad_bg: darkline(im, white_bg=bg), "white_bg=%r" % (bad_bg,))  # type: ignore[arg-type]

    for bad_im in (None, np.zeros((16, 16, 3), dtype=np.uint8), "not_an_image"):
        _assert_raises(TypeError, lambda i=bad_im: darkline(i), "darkline bad img=%r" % (bad_im,))  # type: ignore[arg-type]
        _assert_raises(TypeError, lambda i=bad_im: linework(i), "linework bad img=%r" % (bad_im,))  # type: ignore[arg-type]

    for empty_size in ((0, 16), (16, 0), (0, 0)):
        empty_im = Image.new("RGB", empty_size, (255, 255, 255))
        _assert_raises(ValueError, lambda i=empty_im: darkline(i), "darkline empty=%r" % (empty_size,))
        _assert_raises(ValueError, lambda i=empty_im: linework(i), "linework empty=%r" % (empty_size,))


def test_g4_valid_paths_and_silhouette3d_unchanged() -> None:
    """G4：正对照——合法参数与 Silhouette3D.render() 线稿提取行为逐像素不变。"""
    im = _sample_disc()
    out_py = np.asarray(darkline(im, thr_mult=1.5))
    out_np = np.asarray(darkline(im, thr_mult=np.float64(1.5)))
    assert np.array_equal(out_py, out_np)

    out_zero = np.asarray(darkline(im, thr_mult=0.0))
    assert set(np.unique(out_zero).tolist()) <= {0, 255}
    assert (out_zero == 0).any()

    sil_img = fatfish_skinned().render(120, 140)
    dl = darkline(sil_img, thr_mult=1.5, white_bg=True)
    lw = linework(sil_img, thr_mult=1.5)
    assert dl.mode == "L" and dl.size == (120, 140)
    assert lw.mode == "RGB" and lw.size == (120, 140)
    dl_arr = np.asarray(dl)
    assert set(np.unique(dl_arr).tolist()) <= {0, 255}
    assert (dl_arr == 0).any() and (dl_arr == 255).any()
    assert np.array_equal(np.asarray(lw), np.asarray(dl.convert("RGB")))


def _run_all_assertions() -> int:
    tests = (
        test_g1_nonfinite_thr_mult_rejected,
        test_g2_bool_negative_and_bad_type_thr_mult_rejected,
        test_g3_invalid_white_bg_and_image_rejected,
        test_g4_valid_paths_and_silhouette3d_unchanged,
    )
    fails = 0
    for fn in tests:
        try:
            fn()
            print("  PASS %s" % fn.__name__)
        except AssertionError as exc:
            fails += 1
            print("  FAIL %s: %s" % (fn.__name__, exc))
        except Exception as exc:
            fails += 1
            print("  FAIL %s: %s: %s" % (fn.__name__, type(exc).__name__, exc))
    print("SUMMARY pass=%d fail=%d" % (len(tests) - fails, fails))
    return 0 if fails == 0 else 1


_MUTATION_RUNNER = r'''
import importlib.util, os, sys
repo = os.getcwd()
if repo not in sys.path:
    sys.path.insert(0, repo)
mut_path, test_path = sys.argv[1], sys.argv[2]
import lingshu.vision as vis
spec = importlib.util.spec_from_file_location("lingshu.vision.lineart", mut_path)
mod = importlib.util.module_from_spec(spec)
sys.modules["lingshu.vision.lineart"] = mod
vis.lineart = mod
spec.loader.exec_module(mod)
vis.darkline = mod.darkline
vis.linework = mod.linework
tspec = importlib.util.spec_from_file_location("test_i196_lineart_mut", test_path)
tgt = importlib.util.module_from_spec(tspec)
tspec.loader.exec_module(tgt)
sys.exit(tgt._run_all_assertions())
'''


def _mutate_selftest() -> int:
    """定点变异自证：对临时副本去掉阈值闸或 white_bg 闸 ⇒ 精确点红，还原全绿。"""
    orig_src = LINEART_PATH.read_text(encoding="utf-8")
    orig_sha = hashlib.sha256(LINEART_PATH.read_bytes()).hexdigest()

    arms = [
        (
            "M0_clean_baseline",
            None,
            0,
            set(),
        ),
        (
            "M1_strip_thr_mult_gate",
            ("    mult = _validate_thr_mult(thr_mult)",
             "    mult = float(thr_mult)"),
            1,
            {"test_g1_nonfinite_thr_mult_rejected", "test_g2_bool_negative_and_bad_type_thr_mult_rejected"},
        ),
        (
            "M2_strip_white_bg_gate",
            ("    if not isinstance(white_bg, bool):\n        raise TypeError(\"white_bg 必须是 bool，收到 %s\" % type(white_bg).__name__)",
             "    if False and not isinstance(white_bg, bool):\n        raise TypeError(\"white_bg 必须是 bool，收到 %s\" % type(white_bg).__name__)"),
            1,
            {"test_g3_invalid_white_bg_and_image_rejected"},
        ),
        (
            "M3_restored_baseline",
            None,
            0,
            set(),
        ),
    ]

    bad = 0
    with tempfile.TemporaryDirectory(prefix="lineart_mutate_") as tmp:
        mut_file = Path(tmp) / "lineart_mut.py"
        for name, patch_spec, expect_rc, expect_fails in arms:
            src = orig_src
            if patch_spec is not None:
                old, new = patch_spec
                cnt = src.count(old)
                assert cnt == 1, "%s 锚点计数异常=%d：%r" % (name, cnt, old)
                src = src.replace(old, new, 1)
            mut_file.write_text(src, encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, "-X", "utf8", "-c", _MUTATION_RUNNER, str(mut_file), os.path.abspath(__file__)],
                cwd=str(REPO),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=RUN_TIMEOUT,
            )
            out = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
            got_fails = set(re.findall(r"(?m)^\s*FAIL\s+(\S+?)(?::|\s|$)", out))
            matched = (proc.returncode == expect_rc) and (got_fails == expect_fails)
            if not matched:
                bad += 1
            print(
                "  %-24s rc=%d(expect=%d) failed=%s(expect=%s) VERDICT=%s"
                % (name, proc.returncode, expect_rc, sorted(got_fails), sorted(expect_fails),
                   "PASS" if matched else "FAIL")
            )
    assert hashlib.sha256(LINEART_PATH.read_bytes()).hexdigest() == orig_sha, "lineart.py 被意外改动"
    print("OVERALL=%s (repo_files_untouched=True)" % ("PASS" if bad == 0 else "FAIL"))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    if "--mutate" in sys.argv[1:]:
        sys.exit(_mutate_selftest())
    sys.exit(_run_all_assertions())
