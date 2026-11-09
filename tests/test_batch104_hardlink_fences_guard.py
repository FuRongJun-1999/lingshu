# -*- coding: utf-8 -*-
"""批一百零四守卫：两处硬链接围栏收口（#2568 / #2569）。

覆盖
--------------------------------------------------------------------------
#2568 WAL 追加面必须拒绝硬链接
    G1 源码级：`_wal()` 含 `st_nlink > 1` 拒绝
    G2 行为级：硬链接别名作 `wal_path` ⇒ `step` 抛 ValueError，
       且受保护文件**字节数不变**（改前 53 → 211 被追加 JSONL）
    G3 正对照：正常新文件仍能写 WAL

#2569 `export_trajectory` 必须拒绝硬链接
    G4 源码级：落盘前含 `st_nlink > 1` 拒绝
    G5 行为级：硬链接别名 ⇒ ValueError，且受保护文件**字节数不变、
       frontmatter 完好**（改前 53 → 189、frontmatter 被毁）
    G6 正对照：正常新文件仍能导出

运行（lingshu 仓根）：
    python -X utf8 -m pytest tests/test_batch104_hardlink_fences_guard.py -q
"""
from __future__ import annotations

import inspect
import os
import tempfile

import pytest

from lingshu.core.core import SpacetimeMemoryEngine
from lingshu.world.wm_simloop import SimulationLoop

_ORIG = "---\nid: x\ntitle: 受保护\n---\nORIGINAL CONTENT\n"


def _hardlinked_pair(alias_name="alias.bin"):
    base = tempfile.mkdtemp(prefix="hl_")
    allowed = os.path.join(base, "allowed")
    os.makedirs(allowed)
    prot = os.path.join(base, "protected.md")
    with open(prot, "w", encoding="utf-8") as f:
        f.write(_ORIG)
    alias = os.path.join(allowed, alias_name)
    os.link(prot, alias)
    assert os.path.samefile(alias, prot)
    return base, allowed, prot, alias


# ============================================ #2568 WAL 硬链接
def test_G1_wal_has_nlink_gate():
    """G1：`_wal()` 必须含 `st_nlink > 1` 拒绝。"""
    src = inspect.getsource(SimulationLoop._wal)
    assert "st_nlink" in src and "> 1" in src, "缺 nlink 闸：%r" % src[-500:]


def test_G2_wal_hardlink_rejected_and_protected_intact():
    """G2：硬链接 `wal_path` ⇒ ValueError，受保护文件字节数不变（改前 53→211）。"""
    base, allowed, prot, alias = _hardlinked_pair("run.jsonl")
    old_md = os.environ.get("MDCG_ROOT")
    old_out = os.environ.get("LINGSHU_SIMLOOP_OUT_ROOT")
    os.environ["MDCG_ROOT"] = os.path.join(base, "mdcg_far")
    os.environ["LINGSHU_SIMLOOP_OUT_ROOT"] = allowed
    try:
        before = os.path.getsize(prot)
        e = SpacetimeMemoryEngine(":memory:")
        e.wm_simloop("setup", {"size": 16, "seed": 1, "wal_path": alias})
        e.wm_simloop("entity", {"eid": "w", "pos": (2.0, 1.5, 2.0), "speed": 0.3})
        with pytest.raises(ValueError) as ei:
            e.wm_simloop("step", {"n": 1})
        after = os.path.getsize(prot)
    finally:
        for k, v in (("MDCG_ROOT", old_md),
                     ("LINGSHU_SIMLOOP_OUT_ROOT", old_out)):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    assert "硬链接" in str(ei.value) or "nlink" in str(ei.value), str(ei.value)
    assert after == before, "受保护文件被改写：%d -> %d" % (before, after)


def test_G3_wal_fresh_file_still_works():
    """G3 正对照：正常新文件仍能写 WAL。"""
    base = tempfile.mkdtemp(prefix="ok_")
    allowed = os.path.join(base, "allowed")
    os.makedirs(allowed)
    fresh = os.path.join(allowed, "run.jsonl")
    old_md = os.environ.get("MDCG_ROOT")
    old_out = os.environ.get("LINGSHU_SIMLOOP_OUT_ROOT")
    os.environ["MDCG_ROOT"] = os.path.join(base, "mdcg_far")
    os.environ["LINGSHU_SIMLOOP_OUT_ROOT"] = allowed
    try:
        e = SpacetimeMemoryEngine(":memory:")
        e.wm_simloop("setup", {"size": 16, "seed": 1, "wal_path": fresh})
        e.wm_simloop("entity", {"eid": "w", "pos": (2.0, 1.5, 2.0), "speed": 0.3})
        r = e.wm_simloop("step", {"n": 1})
    finally:
        if old_md is None:
            os.environ.pop("MDCG_ROOT", None)
        else:
            os.environ["MDCG_ROOT"] = old_md
        if old_out is None:
            os.environ.pop("LINGSHU_SIMLOOP_OUT_ROOT", None)
        else:
            os.environ["LINGSHU_SIMLOOP_OUT_ROOT"] = old_out
    assert r.get("status") == "ok", r
    assert os.path.getsize(fresh) > 0


# ============================================ #2569 export_trajectory 硬链接
def test_G4_export_trajectory_has_nlink_gate():
    """G4：`export_trajectory` 落盘前必须含 `st_nlink > 1` 拒绝。"""
    src = inspect.getsource(SpacetimeMemoryEngine.export_trajectory)
    assert "st_nlink" in src and "> 1" in src, \
        "缺 nlink 闸：%r" % src[-600:]


def test_G5_export_hardlink_rejected_and_protected_intact():
    """G5：硬链接 ⇒ ValueError，受保护文件不变且 frontmatter 完好（改前被截断）。"""
    base, allowed, prot, alias = _hardlinked_pair("traj.json")
    old_out = os.environ.get("LINGSHU_EXPORT_OUT_ROOT")
    os.environ["LINGSHU_EXPORT_OUT_ROOT"] = allowed
    try:
        before = os.path.getsize(prot)
        e = SpacetimeMemoryEngine(":memory:")
        e.add_perception("测试节点ABC", importance=0.6)
        with pytest.raises(ValueError) as ei:
            e.export_trajectory(output_path=alias)
        after = os.path.getsize(prot)
        with open(prot, "rb") as f:
            head = f.read(3)
    finally:
        if old_out is None:
            os.environ.pop("LINGSHU_EXPORT_OUT_ROOT", None)
        else:
            os.environ["LINGSHU_EXPORT_OUT_ROOT"] = old_out
    assert "硬链接" in str(ei.value) or "nlink" in str(ei.value), str(ei.value)
    assert after == before, "受保护文件被截断：%d -> %d" % (before, after)
    assert head == b"---", "frontmatter 被毁"


def test_G6_export_fresh_file_still_works():
    """G6 正对照：正常新文件仍能导出。"""
    base = tempfile.mkdtemp(prefix="ok2_")
    allowed = os.path.join(base, "allowed")
    os.makedirs(allowed)
    fresh = os.path.join(allowed, "traj.json")
    old_out = os.environ.get("LINGSHU_EXPORT_OUT_ROOT")
    os.environ["LINGSHU_EXPORT_OUT_ROOT"] = allowed
    try:
        e = SpacetimeMemoryEngine(":memory:")
        e.add_perception("测试节点DEF", importance=0.6)
        r = e.export_trajectory(output_path=fresh)
    finally:
        if old_out is None:
            os.environ.pop("LINGSHU_EXPORT_OUT_ROOT", None)
        else:
            os.environ["LINGSHU_EXPORT_OUT_ROOT"] = old_out
    assert isinstance(r, dict)
    assert os.path.getsize(fresh) > 0
