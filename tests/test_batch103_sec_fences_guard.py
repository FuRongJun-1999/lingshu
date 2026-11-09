# -*- coding: utf-8 -*-
"""批一百零三守卫：两处安全面收口（#2565 / #2566）。

覆盖
--------------------------------------------------------------------------
#2565 `scenario_cli` 的脑根渲染围栏被 `\\?\` 逐字前缀绕过
    G1 源码级：`brain` 分支先拒 UNC/device 前缀，再做 `is_relative_to`
       （改前只有 `is_relative_to`；`Path("\\\\?\\C:\\...").resolve()` 保留前缀
        ⇒ 恒 False ⇒ 围栏放行）
    G2 行为级：`Path(verbatim).resolve()` 与 `Path(root).resolve()` 的相对判定
       在**未加前缀闸**时确实为 False（复现根因）
    G3 正对照：普通路径的 `is_relative_to` 仍正确

#2566 `realpath` 围栏对硬链接失效
    G4 源码级：`_confine_world3d_out_path` 含 `st_nlink > 1` 拒绝
    G5 行为级：硬链接别名指向受保护文件 ⇒ `render` 返回 error dict，
       且受保护文件**字节数不变**（改前被覆盖成 PNG）
    G6 正对照：正常新文件仍能渲染（`status=ok`）

运行（lingshu 仓根）：
    python -X utf8 -m pytest tests/test_batch103_sec_fences_guard.py -q
"""
from __future__ import annotations

import inspect
import os
import tempfile
from pathlib import Path

import pytest

from lingshu.core.core import SpacetimeMemoryEngine
from lingshu.world import scenario_cli as SC

_VERB = "\\" * 2 + "?" + "\\"


# ============================================ #2565 scenario_cli 围栏
def test_G1_brain_branch_rejects_device_prefix():
    """G1：`brain` 分支必须**行为上**拒 UNC/device 前缀。

    注意：不能只做源码字符串断言——把 `if _raw.startswith(...)` 改成
    `if False:` 时字符串仍在、断言仍绿（实测无判别力）。这里真跑 `main`。
    """
    root = tempfile.mkdtemp(prefix="brainroot_")
    evil = os.path.join(root, "evil.png")
    # `--mode world` 必须显式给（否则 argparse 在 causal/all 分支先退）
    argv = ["brain", "--root", root, "--brain-path", root,
            "--render", _VERB + evil, "--query", "q", "--mode", "world"]
    # 关键：不能只断言 `rc != 0`——围栏之后还有 `brain-path` 校验也会返回 1，
    # 那样变异（`if False:`）后仍能通过。必须**捕获 stderr 文案**确认是围栏本身拒的。
    import contextlib
    import io as _io
    buf = _io.StringIO()
    rc = None
    try:
        with contextlib.redirect_stderr(buf):
            rc = SC.main(argv)
    except SystemExit as e:
        rc = e.code
    except (ValueError, OSError):
        rc = 1
    err = buf.getvalue()
    assert rc != 0, "verbatim 前缀路径未被围栏拒绝（rc=%r）" % (rc,)
    assert "UNC/device" in err, (
        "未由围栏拒绝（可能被后续检查兜住）——stderr=%r" % (err[:200],))


@pytest.mark.skipif(os.name != "nt",
                    reason="verbatim prefix is Win32-only syntax")
def test_G2_verbatim_prefix_breaks_is_relative_to():
    """G2 根因复现（Windows 专属）：`\\?\` 前缀让 `is_relative_to` 恒 False。"""
    root = tempfile.mkdtemp(prefix="brain_")
    target = os.path.join(root, "evil.png")
    root_res = Path(root).resolve()
    assert Path(target).resolve().is_relative_to(root_res) is True
    v = Path(_VERB + target).resolve()
    # 改前：verbatim 版本判 False（漏洞）；这里只钉「前缀确实会破坏该判定」
    assert v.is_relative_to(root_res) is False, \
        "若为 True 说明该平台已归一前缀，本件的根因描述需复核"
    assert os.path.realpath(_VERB + target) == _VERB + target, \
        "realpath 应保留 verbatim 前缀"


def test_G3_plain_path_still_relative():
    """G3 正对照：普通路径的 `is_relative_to` 仍正确。"""
    root = tempfile.mkdtemp(prefix="brain_")
    inside = Path(os.path.join(root, "a.png")).resolve()
    outside = Path(tempfile.mkdtemp()).resolve() / "b.png"
    assert inside.is_relative_to(Path(root).resolve()) is True
    assert outside.is_relative_to(Path(root).resolve()) is False


# ============================================ #2566 硬链接围栏
def test_G4_confine_has_nlink_gate():
    """G4：`_confine_world3d_out_path` 必须含 `st_nlink > 1` 拒绝。"""
    src = inspect.getsource(SpacetimeMemoryEngine._confine_world3d_out_path)
    assert "st_nlink" in src and "> 1" in src, "缺 nlink 闸：%r" % src[-400:]


def test_G5_hardlink_alias_rejected_and_protected_intact():
    """G5：硬链接别名 ⇒ error dict，且受保护文件字节数不变（改前被覆盖成 PNG）。"""
    base = tempfile.mkdtemp(prefix="hl_")
    allowed = os.path.join(base, "allowed")
    os.makedirs(allowed)
    prot = os.path.join(base, "protected.md")
    with open(prot, "w", encoding="utf-8") as f:
        f.write("---\nid: x\n---\nORIGINAL\n")
    alias = os.path.join(allowed, "chart.png")
    os.link(prot, alias)
    assert os.path.samefile(alias, prot)

    old_env = os.environ.get("LINGSHU_WORLD3D_OUT_ROOT")
    os.environ["LINGSHU_WORLD3D_OUT_ROOT"] = allowed
    try:
        before = os.path.getsize(prot)
        e = SpacetimeMemoryEngine(":memory:")
        e.world3d("add", {"category": "box", "bbox": [380, 280, 420, 320]})
        r = e.world3d("render", {"path": alias})
        after = os.path.getsize(prot)
    finally:
        if old_env is None:
            os.environ.pop("LINGSHU_WORLD3D_OUT_ROOT", None)
        else:
            os.environ["LINGSHU_WORLD3D_OUT_ROOT"] = old_env

    assert r.get("status") == "error", r
    assert "硬链接" in r.get("error", "") or "nlink" in r.get("error", ""), r
    assert after == before, "受保护文件被改写：%d -> %d" % (before, after)
    with open(prot, "rb") as f:
        assert f.read(3) == b"---", "frontmatter 被毁"


def test_G6_fresh_file_still_renders():
    """G6 正对照：正常新文件仍能渲染。"""
    base = tempfile.mkdtemp(prefix="ok_")
    allowed = os.path.join(base, "allowed")
    os.makedirs(allowed)
    old_env = os.environ.get("LINGSHU_WORLD3D_OUT_ROOT")
    os.environ["LINGSHU_WORLD3D_OUT_ROOT"] = allowed
    try:
        e = SpacetimeMemoryEngine(":memory:")
        e.world3d("add", {"category": "box", "bbox": [380, 280, 420, 320]})
        fresh = os.path.join(allowed, "fresh.png")
        r = e.world3d("render", {"path": fresh})
    finally:
        if old_env is None:
            os.environ.pop("LINGSHU_WORLD3D_OUT_ROOT", None)
        else:
            os.environ["LINGSHU_WORLD3D_OUT_ROOT"] = old_env
    assert r.get("status") == "ok", r
    assert os.path.exists(fresh)
