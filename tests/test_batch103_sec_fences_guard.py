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
    """G1：`brain` 分支必须先拒 UNC/device 前缀（改前只有 is_relative_to）。"""
    src = inspect.getsource(SC.main)
    # `args.source == "brain"` 在 main 里出现两次（前面还有 --k 校验）；
    # 定位到**含 is_relative_to 的那一处**（渲染围栏）。
    i = src.find("outside the brain root")
    assert i > 0, "未找到脑根围栏文案"
    seg = src[max(0, i - 900):i + 120]
    assert "UNC/device path" in seg, "brain 分支缺 UNC/device 拒绝：%r" % seg[-500:]
    assert "is_relative_to" in seg, "brain 分支缺 is_relative_to 判定"


def test_G2_verbatim_prefix_breaks_is_relative_to():
    """G2 根因复现：`\\?\` 前缀让 `is_relative_to` 恒 False（而两路径同文件）。"""
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
