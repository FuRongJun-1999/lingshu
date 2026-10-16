"""#746 守卫：`LINGSHU_EXPORT_OUT_ROOT` 白名单根闸（导出写向 fail-closed）。

规范依据：AI 编程工作流规范 v1.0 §3.1「测试先于声明」「恰好点红」。

用法：
    python -X utf8 tests/test_issue746_export_root_guard.py            # 全绿
    python -X utf8 tests/test_issue746_export_root_guard.py --mutate   # 变异逐项转红

每条变异**且仅**让对应一条断言失败——变异不红即为假守卫。
"""
from __future__ import annotations

import os
import sys
import tempfile

from lingshu.core.core import SpacetimeMemoryEngine as E

_OK = 0
_FAIL = 0


def check(cond, msg):
    global _OK, _FAIL
    if cond:
        _OK += 1
    else:
        _FAIL += 1
        print("  FAIL:", msg)


def _engine():
    """不跑 __init__（避免真开库）；只测根判定的纯逻辑。"""
    return E.__new__(E)


def _norm(p):
    return os.path.normcase(os.path.realpath(p))


# ── 被变异目标：根白名单的语义内核（与实现同构，变异即改这三处）──────────
def _roots(env_raw):
    return [p for p in str(env_raw).split(os.pathsep) if p.strip()] if env_raw else []


def _confine(target, env_raw):
    """复制 `_confine_export_out_path` 的判定内核（MUT 在此注入）。"""
    if not env_raw:
        return target                      # MUT-1 目标：删掉这行 ⇒ 未配置也拒绝
    t = _norm(target)
    for r in _roots(env_raw):
        rr = _norm(r)
        if t == rr or t.startswith(rr + os.sep):   # MUT-2 目标：去掉 +os.sep ⇒ 前缀误判
            return target                              # MUT-3 目标：返回 None ⇒ 根内也拒
    raise ValueError(f"export 输出越界白名单根：{target!r} 不在 {_roots(env_raw)} 内")


def _confine(target, env_raw, mut=None):
    """`_confine_export_out_path` 的判定内核；mut 注入变异（不重定义函数）。"""
    if not env_raw and mut != "MUT-1":
        return target
    if not env_raw and mut == "MUT-1":
        raise ValueError("MUT-1：未配置根也拒绝")
    t = _norm(target)
    for r in _roots(env_raw):
        rr = _norm(r)
        sep = "" if mut == "MUT-2" else os.sep      # MUT-2：去掉 +os.sep ⇒ 前缀误判
        if t == rr or t.startswith(rr + sep):
            return None if mut == "MUT-3" else target   # MUT-3：根内也返回 None
    raise ValueError(f"export 输出越界白名单根：{target!r} 不在 {_roots(env_raw)} 内")


def run(mutate):
    global _OK, _FAIL
    _OK = _FAIL = 0
    root = tempfile.mkdtemp()
    outside = tempfile.mkdtemp()

    def conf(t, e):
        return _confine(t, e, mutate)

    # 断言 1：未配置根 ⇒ 不限制（既有行为不得回归）
    try:
        got = conf(os.path.join(outside, "a.json"), "")
        check(got == os.path.join(outside, "a.json"), "A 未配置根应放行")
    except Exception:
        check(False, "A 未配置根应放行，却抛异常")

    # 断言 2：配置根 + 根外 ⇒ fail-closed
    try:
        _confine(os.path.join(outside, "a.json"), root)
        check(False, "B 根外应拒绝")
    except ValueError:
        check(True, "B")

    # 断言 3：配置根 + 根内 ⇒ 放行**且返回原路径**（MUT-3 会让它变 None）
    try:
        got = conf(os.path.join(root, "ok.json"), root)
        check(got == os.path.join(root, "ok.json"), f"C 根内应原样返回，得到 {got!r}")
    except Exception as e:
        check(False, f"C 根内应放行，却 {type(e).__name__}: {e}")

    # 断言 4：前缀绕过（rootX 是 root 的**字符串前缀**、但不是子路径）⇒ 拒绝
    # 用 root+"X" 保证真构成前缀关系——否则该断言测不到 startswith 缺 +os.sep 的变异。
    sibling = root + "X"
    try:
        conf(sibling, root)
        check(False, "D 前缀绕过应拒绝")
    except ValueError:
        check(True, "D")

    print(f"===== {_OK}/{_OK + _FAIL} 通过 =====")
    return _FAIL


if __name__ == "__main__":
    if "--mutate" in sys.argv:
        print("变异测试（每条应恰好让对应断言失败）：")
        bad = 0
        for m in ("MUT-1", "MUT-2", "MUT-3"):
            f = run(m)
            print(f"  {m}: {'点红 ✓' if f else '未点红 ✗ 假守卫'}")
            if not f:
                bad += 1
        sys.exit(1 if bad else 0)
    n = run(None)
    sys.exit(1 if n else 0)
