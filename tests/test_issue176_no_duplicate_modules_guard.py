# -*- coding: utf-8 -*-
"""守卫 · #176：同名模块在包内必须唯一（副本合一）

缺陷（main 8a2de8f）
    `lingshu/core/time_core.py` 与 `lingshu/world/time_core.py` 逐字节相同；
    `lingshu/nn/hex_composite.py` 与 `lingshu/gen/hex_composite.py` 同样。
    「唯一核」本身有两个来源：改一份忘另一份即静默分叉。

修复（#176 建议3）
    删除 `world/time_core.py` 与 `gen/hex_composite.py`，改为从唯一位置导入；
    在 lint/守卫里禁止同名副本。

断言组
    U1 全包内名为 `time_core.py` 的文件恰好 1 个，且是 `lingshu/core/time_core.py`
    U2 全包内名为 `hex_composite.py` 的文件恰好 1 个，且是 `lingshu/nn/hex_composite.py`
    U3 反例自证：向 `lingshu/world/` 放一个同名空文件 ⇒ U1 必须变红（本文件 --mutate 自带）
    U4 唯一真源可导入且关键符号在位

运行（仓根）
    python -X utf8 -m pytest tests/test_issue176_no_duplicate_modules_guard.py -q --no-header
    python -X utf8 tests/test_issue176_no_duplicate_modules_guard.py --mutate
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(REPO, "lingshu")

CANONICAL = {
    "time_core.py": os.path.join("core", "time_core.py"),
    "hex_composite.py": os.path.join("nn", "hex_composite.py"),
}
# #176 明确删除的两个副本位置
REMOVED = (
    os.path.join("world", "time_core.py"),
    os.path.join("gen", "hex_composite.py"),
)


def _walk_py_basenames(root):
    """返回 {basename: [相对路径, ...]}，只扫包内 .py。"""
    seen = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ("__pycache__", ".git")]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            rel = os.path.relpath(os.path.join(dirpath, fn), root)
            seen.setdefault(fn, []).append(rel)
    return seen


def _duplicates():
    seen = _walk_py_basenames(PKG)
    return {b: paths for b, paths in seen.items() if len(paths) > 1}


def test_u1_time_core_unique():
    """U1：包内 time_core.py 恰好 1 个，且位于 core/。"""
    seen = _walk_py_basenames(PKG)
    paths = seen.get("time_core.py", [])
    assert len(paths) == 1, "time_core.py 必须唯一，实得 %r" % (paths,)
    assert paths[0].replace(os.sep, "/") == "core/time_core.py", paths


def test_u2_hex_composite_unique():
    """U2：包内 hex_composite.py 恰好 1 个，且位于 nn/。"""
    seen = _walk_py_basenames(PKG)
    paths = seen.get("hex_composite.py", [])
    assert len(paths) == 1, "hex_composite.py 必须唯一，实得 %r" % (paths,)
    assert paths[0].replace(os.sep, "/") == "nn/hex_composite.py", paths


def test_u3_removed_paths_stay_removed():
    """U3：#176 明确删除的两处不得重新出现。"""
    for rel in REMOVED:
        p = os.path.join(PKG, rel)
        assert not os.path.exists(p), "副本复活：%s" % rel


WATCH = ("time_core.py", "hex_composite.py")


def test_u4_watched_basenames_unique():
    """U4：本件范围内的同名模块必须唯一。

    只断言本件涉及的 basename（与 #176 直接相关）。包内其它既存同名文件
    属另外的问题，不在此处判红，仅打印供后续排查——否则本守卫会被
    「与 #176 无关的既存重复」长期占红，失去判别力。
    """
    seen = _walk_py_basenames(PKG)
    for name in WATCH:
        paths = seen.get(name, [])
        assert len(paths) == 1, "%s 必须唯一，实得 %r" % (name, paths)
    others = {b: v for b, v in _duplicates().items() if b not in WATCH}
    if others:
        print("[info] 包内还有其它同名文件（非 #176 范围）：%r" % (sorted(others),))


def test_u5_canonical_modules_importable():
    """U5：唯一真源可导入且关键符号在位。"""
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    from lingshu.core import time_core as T
    from lingshu.nn import hex_composite as H
    for name in ("cred", "cred_factor", "cred_step", "cred_blend"):
        assert hasattr(T, name), "time_core 缺 %s" % name
    for name in ("extract_attributes", "judge_density", "_COLORS_RGB", "_paint",
                 "COLORS", "PATTERNS", "SHAPES"):
        assert hasattr(H, name), "hex_composite 缺 %s" % name


def _mutate():
    """U1 的判别力自证：放一个同名副本 ⇒ 必须变红。"""
    victim = os.path.join(PKG, "world", "time_core.py")
    assert not os.path.exists(victim), "预置条件：副本本不该存在"
    os.makedirs(os.path.dirname(victim), exist_ok=True)
    open(victim, "w", encoding="utf-8").write("# 变异注入：同名副本\n")
    bad = 0
    try:
        test_u1_time_core_unique()
        print("  M1 放回 world/time_core.py  期望=红 实得=绿  VERDICT=FAIL（守卫没咬住）")
        bad += 1
    except AssertionError as exc:
        print("  M1 放回 world/time_core.py  期望=红 实得=红  VERDICT=PASS  (%s)" % str(exc)[:60])
    try:
        test_u4_watched_basenames_unique()
        print("  M1 通则 U4                  期望=红 实得=绿  VERDICT=FAIL")
        bad += 1
    except AssertionError:
        print("  M1 通则 U4                  期望=红 实得=红  VERDICT=PASS")
    finally:
        os.unlink(victim)
    assert not os.path.exists(victim), "还原失败"
    test_u1_time_core_unique()
    print("  M2 还原后                   期望=绿 实得=绿  VERDICT=PASS")
    print("-- %s --" % ("PASS" if not bad else "FAIL"))
    return bad


if __name__ == "__main__":
    if "--mutate" in sys.argv:
        sys.exit(1 if _mutate() else 0)
    fns = [test_u1_time_core_unique, test_u2_hex_composite_unique,
           test_u3_removed_paths_stay_removed, test_u4_watched_basenames_unique,
           test_u5_canonical_modules_importable]
    failed = 0
    for fn in fns:
        try:
            fn()
            print("PASS %s" % fn.__name__)
        except AssertionError as exc:
            failed += 1
            print("FAIL %s | %s" % (fn.__name__, exc))
    print("-- %d passed, %d failed --" % (len(fns) - failed, failed))
    print("VERDICT=%s" % ("PASS" if not failed else "FAIL"))
    sys.exit(1 if failed else 0)
