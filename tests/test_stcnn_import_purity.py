# -*- coding: utf-8 -*-
"""test_stcnn_import_purity · `lingshu/nn/stcnn.py` 的导入期纯净性守卫
============================================================================
来历（issue #147，审查基线 `8a2de8f`）：
    `lingshu/nn/stcnn.py` 在**模块顶层**执行

        sys.stdout.reconfigure(encoding='utf-8')

    由此产生两个后果：
      ① stdout 不是真实文件（如被 `contextlib.redirect_stdout(io.StringIO())`
         包住、或宿主替换了 stdout）时，`import lingshu.nn.stcnn` 直接
         `AttributeError`（`_io.StringIO` 没有 `reconfigure`）——该模块的
         `conv3d` / `detect_period` / `SpatiotemporalMemory` 全部不可用；
      ② **导入会改写宿主进程的 stdout 编码**（实测 `PYTHONIOENCODING=gbk`
         导入后变成 utf-8），影响导入方自己的所有输出。

判据（三条）：
  A 重定向 stdout 后仍可导入该模块（导入期不得依赖真实 `TextIOWrapper`）。
  B 导入前后**宿主 stdout 编码不变**（导入不得改动宿主进程状态）。
  C `python -m lingshu.nn.stcnn` 的演示**仍能跑完且判定通过**（修法不得
    损害演示入口）。

为什么用子进程：判据 A/B 都要求"把一个**干净解释器**交给被测导入"，且要让
    `PYTHONIOENCODING` 与 stdout 包装形态可控；在 pytest 进程内做会污染 pytest
    自身且无法重复。判据 C 本身就必须是独立进程（`-m` 形态）。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_TARGET = "lingshu.nn.stcnn"
_TIMEOUT = 300


def _run_child(source: str, extra_env=None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("PYTHONIOENCODING", None)          # 由用例显式指定
    env["PYTHONPATH"] = str(REPO)
    env["PYTHONUTF8"] = "1"
    env.update(extra_env or {})
    return subprocess.run(
        [sys.executable, "-W", "ignore", "-c", source],
        cwd=str(REPO), env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=_TIMEOUT,
    )


def test_import_survives_redirected_stdout():
    """判据 A：stdout 被替换成非文件对象时，导入仍必须成功。"""
    src = (
        "import contextlib, io\n"
        "try:\n"
        "    with contextlib.redirect_stdout(io.StringIO()):\n"
        f"        import {_TARGET}\n"
        "    print('IMPORT_OK')\n"
        "except Exception as exc:\n"
        "    print('IMPORT_FAIL: %s: %s' % (type(exc).__name__, exc))\n"
    )
    proc = _run_child(src)
    assert "IMPORT_OK" in proc.stdout, (
        "stdout 被重定向后，导入 %s 失败（issue #147）：\n"
        f"  stdout={proc.stdout.strip()!r}\n  stderr={proc.stderr.strip()[-400:]!r}\n"
        "成因：模块顶层 `sys.stdout.reconfigure(...)` 依赖真实 TextIOWrapper。\n"
        "修法：把编码调整移入 `if __name__ == \"__main__\":` 并对无 `reconfigure` "
        "的流做 `hasattr` 保护。"
    )


def test_import_does_not_change_host_stdout_encoding():
    """判据 B：导入不得改写宿主进程的 stdout 编码。"""
    src = (
        "import sys\n"
        "before = getattr(sys.stdout, 'encoding', None)\n"
        f"import {_TARGET}\n"
        "after = getattr(sys.stdout, 'encoding', None)\n"
        "print('ENC|%s|%s' % (before, after), file=sys.stderr)\n"
    )
    proc = _run_child(src, {"PYTHONIOENCODING": "gbk"})
    marker = next((ln for ln in proc.stderr.splitlines() if ln.startswith("ENC|")), None)
    assert marker is not None, (
        "未取到编码读数（子进程可能崩）。\n"
        f"  rc={proc.returncode}\n  stderr={proc.stderr.strip()[-400:]!r}"
    )
    _, before, after = marker.split("|")
    assert before == after, (
        "导入 %s 改写了宿主进程的 stdout 编码（issue #147）：\n"
        f"  {before!r} -> {after!r}\n"
        "契约：库的导入不得改动宿主进程的全局状态。"
    )


def test_demo_still_runs_and_passes():
    """判据 C：`python -m lingshu.nn.stcnn` 演示仍退出 0 且判定通过（4/4）。"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO)
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run(
        [sys.executable, "-W", "ignore", "-m", _TARGET],
        cwd=str(REPO), env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=_TIMEOUT,
    )
    assert proc.returncode == 0, (
        "演示入口 `python -m %s` 退出码非 0：\n"
        f"  rc={proc.returncode}\n  stderr={proc.stderr.strip()[-500:]!r}"
    )
    assert "时空识别: 4/4" in proc.stdout, (
        "演示判定行未出现 `时空识别: 4/4`（演示行为被改动？）：\n"
        f"  stdout 末段={proc.stdout.strip()[-500:]!r}"
    )
