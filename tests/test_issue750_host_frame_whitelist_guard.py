# -*- coding: utf-8 -*-
"""test_issue750_host_frame_whitelist_guard · 宿主帧白名单名导入必须先过受控面
============================================================================
背景（lingshu issue #750 · 维护侧 TRIAGE-750 已登记的最小改动）：
  `_ComponentFinder.find_spec` 旧顺序为
      if fullname not in COMPONENT_NAMES: return None
      if not _importer_is_lingshu(): return None      # ← 先于受控面查找
      roots = safe_search_path(); ...
  于是**宿主帧**发起的白名单名导入（`vision` / `body` / `game_web` …）根本
  不看受控面，直接交还后续 finder；标准 `PathFinder` 按 `sys.path` 解析到
  cwd / 脚本目录里的**同名毒模块**并在引擎进程内执行（`LINGSHU_ALLOW_CWD_IMPORTS=1`
  打开时更直接）。维护侧处置：①不修（逃生口语义 + #275 有意取舍），但**已登记**
  最小改动 —— 「对白名单名先用 safe_search_path() 找一遍，找不到再 return None」。

修复：`find_spec` 把受控面查找**提到调用方判定之前**——受控面命中即返回受控实现
（不论调用方）；受控面未命中时（**#750 残余修复 · EQRS #817 方向①**起）不再
`return None` 交还后续 finder，而是同样按「组件缺失」失败——宿主帧白名单名的
解析面自此**只剩受控面** `safe_search_path()`，A/D/E 三臂（#817 四臂矩阵）封死。

断言组（回退 / 放宽即红）：
  G1 逃生口**开** + cwd 放毒 `vision.py` + 受控面有合法 `vision.py` ⇒ 宿主帧
     `import vision` 命中**受控面**实现（OWNER=='legit'、POISON 标记不存在）。
  G2 逃生口**关**：同上，仍命中受控面实现。
  G3 **#275 不得被破坏**：宿主自有**非白名单**同名模块（`myhostmod`）在逃生口开
     时仍可被标准 `PathFinder` 解析（回归红线）。
  G4 **#750 残余收口后口径**：白名单名在受控面**无**实现时，宿主帧**不再**交还
     后续 finder——`ModuleNotFoundError`、cwd 毒模块**不执行**（旧口径「交还
     PathFinder、宿主自有同名模块照旧可解析」即 A 臂漏洞本体，已按 #817 方向①
     收紧；宿主确需同名模块的支持径见 tests/test_issue750_residual_arms_guard.py
     G5/G6/G9）。
  G5 引擎侧收口未坏：`LINGSHU_COMPONENT_ROOT` 显式根里的组件仍能被引擎装配。
  G6 定点变异自证：把宿主帧分支短路回「无条件 return None」（复刻旧实现）后
     G1 场景**必红**（毒模块被执行）。

运行（仓根）：python -X utf8 tests/test_issue750_host_frame_whitelist_guard.py
退出码：0 = 全过；1 = 有断言失败
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

WL_NAME = "vision"          # 白名单名（COMPONENT_NAMES 成员）
NON_WL = "myhostmod"        # 非白名单名

_LEGIT = "OWNER = 'legit'\nPOISON = False\n"
_POISON = ("import os as _os\n"
           "POISON = True\n"
           "OWNER = 'poison'\n"
           "open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),"
           " 'MARKER.txt'), 'w').write('EXECUTED')\n")

_MAIN = (
    "import sys, lingshu.core.core\n"
    "for m in (%r, %r):\n"
    "    try:\n"
    "        mod = __import__(m)\n"
    "        print('IMPORT %%-12s ok owner=%%s poison=%%s' %% (m, getattr(mod, 'OWNER', None), getattr(mod, 'POISON', None)))\n"
    "    except ModuleNotFoundError as e:\n"
    "        print('IMPORT %%-12s ModuleNotFoundError' %% m)\n"
) % (WL_NAME, NON_WL)


def _mk_roots():
    """返回 (component_root, poison_dir)：受控面放合法 vision.py；毒目录放 vision.py。"""
    root = tempfile.mkdtemp(prefix="f750_root_")
    poison = tempfile.mkdtemp(prefix="f750_poison_")
    with open(os.path.join(root, WL_NAME + ".py"), "w", encoding="utf-8") as f:
        f.write(_LEGIT)
    with open(os.path.join(poison, WL_NAME + ".py"), "w", encoding="utf-8") as f:
        f.write(_POISON)
    with open(os.path.join(poison, NON_WL + ".py"), "w", encoding="utf-8") as f:
        f.write("OWNER = 'host'\n")
    with open(os.path.join(poison, "main.py"), "w", encoding="utf-8") as f:
        f.write(_MAIN)
    return root, poison


def _run(poison, extra_env=None, prelude=""):
    env = {k: v for k, v in os.environ.items()
           if k not in ("LINGSHU_ALLOW_CWD_IMPORTS", "LINGSHU_COMPONENT_ROOT")}
    env["PYTHONPATH"] = REPO
    env["PYTHONUTF8"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if extra_env:
        env.update(extra_env)
    main = os.path.join(poison, "main.py")
    if prelude:
        with open(main, "w", encoding="utf-8") as f:
            f.write(prelude + _MAIN)
    try:
        return subprocess.run([sys.executable, "-B", "-W", "ignore", main],
                              cwd=poison, env=env, capture_output=True,
                              text=True, encoding="utf-8")
    finally:
        if prelude:
            with open(main, "w", encoding="utf-8") as f:
                f.write(_MAIN)


def _line(out, name):
    for ln in out.splitlines():
        if ln.startswith("IMPORT %-12s" % name):
            return ln
    return "<missing>"


# ---------------------------------------------------------------- G1 / G2
def test_g1_escape_hatch_on_host_frame_uses_controlled_root():
    root, poison = _mk_roots()
    r = _run(poison, {"LINGSHU_ALLOW_CWD_IMPORTS": "1", "LINGSHU_COMPONENT_ROOT": root})
    assert r.returncode == 0, r.stderr[-600:]
    ln = _line(r.stdout, WL_NAME)
    assert "owner=legit poison=False" in ln, (
        "G1: 宿主帧白名单名未命中受控面实现（#750）\n%s\n%s" % (ln, r.stdout))
    assert not os.path.exists(os.path.join(poison, "MARKER.txt")), (
        "G1: cwd 毒模块被执行（MARKER.txt 存在）")


def test_g2_escape_hatch_off_host_frame_uses_controlled_root():
    root, poison = _mk_roots()
    r = _run(poison, {"LINGSHU_COMPONENT_ROOT": root})
    assert r.returncode == 0, r.stderr[-600:]
    ln = _line(r.stdout, WL_NAME)
    assert "owner=legit poison=False" in ln, (
        "G2: 逃生口关闭时宿主帧白名单名未命中受控面实现\n%s\n%s" % (ln, r.stdout))


# ---------------------------------------------------------------- G3 / G4
def test_g3_host_own_non_whitelist_module_still_importable():
    root, poison = _mk_roots()
    r = _run(poison, {"LINGSHU_ALLOW_CWD_IMPORTS": "1", "LINGSHU_COMPONENT_ROOT": root})
    assert r.returncode == 0, r.stderr[-600:]
    ln = _line(r.stdout, NON_WL)
    assert "owner=host" in ln, "G3(#275): 宿主自有非白名单模块被接管\n%s" % r.stdout


def test_g4_whitelist_miss_no_fallback_to_cwd_poison():
    """受控面无该白名单名 ⇒ 宿主帧不再交还后续 finder（#750 残余收口后口径）。

    旧口径（#782 时代 G4）断言 `owner=poison` 可解析——那正是 #817 四臂矩阵的
    A 臂（逃生口开 + cwd==脚本目录 + 受控面无实现 ⇒ 毒模块执行）本体；本 PR 按
    EQRS #817 方向①收口后，该腿改钉「ModuleNotFoundError + 毒模块零执行」。
    """
    empty_root = tempfile.mkdtemp(prefix="f750_empty_")
    _root, poison = _mk_roots()
    r = _run(poison, {"LINGSHU_ALLOW_CWD_IMPORTS": "1", "LINGSHU_COMPONENT_ROOT": empty_root})
    assert r.returncode == 0, r.stderr[-600:]
    ln = _line(r.stdout, WL_NAME)
    assert "ModuleNotFoundError" in ln, (
        "G4(#750 残余): 受控面未命中时宿主帧不应交还后续 finder\n%s\n%s" % (ln, r.stdout))
    assert not os.path.exists(os.path.join(poison, "MARKER.txt")), (
        "G4(#750 残余): cwd 毒模块被执行（MARKER.txt 存在）")


# ------------------------------------------------------------------- G5
def test_g5_engine_side_controlled_root_still_works():
    root = tempfile.mkdtemp(prefix="f750_eng_")
    with open(os.path.join(root, "semantic_space.py"), "w", encoding="utf-8") as f:
        f.write("class SemanticSpace:\n"
                "    def __init__(self, *a, **k): pass\n")
    d = tempfile.mkdtemp(prefix="f750_engwork_")
    script = os.path.join(d, "eng.py")
    with open(script, "w", encoding="utf-8") as f:
        f.write("from lingshu.core.core import SpacetimeMemoryEngine\n"
                "e = SpacetimeMemoryEngine(':memory:')\n"
                "print('OK')\n")
    env = {k: v for k, v in os.environ.items()
           if k not in ("LINGSHU_ALLOW_CWD_IMPORTS", "LINGSHU_COMPONENT_ROOT")}
    env.update({"PYTHONPATH": REPO, "PYTHONUTF8": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "LINGSHU_COMPONENT_ROOT": root})
    r = subprocess.run([sys.executable, "-B", "-W", "ignore", script],
                       cwd=d, env=env, capture_output=True, text=True,
                       encoding="utf-8")
    assert r.returncode == 0 and "OK" in r.stdout, (
        "G5: 受控面组件装配失败\n%s\n%s" % (r.stdout, r.stderr[-600:]))


# ------------------------------------------------------------------- G6
_BROKEN_PRELUDE = (
    "import sys\n"
    "import lingshu.core.core as _c\n"
    "import lingshu.core.component_resolver as _cr\n"
    "class _BrokenFinder(_cr._ComponentFinder):\n"
    "    def find_spec(self, fullname, path=None, target=None):\n"
    "        if fullname not in _cr.COMPONENT_NAMES:\n"
    "            return None\n"
    "        if not _cr._importer_is_lingshu():\n"
    "            return None\n"
    "        return super().find_spec(fullname, path, target)\n"
    "sys.meta_path[:] = [_BrokenFinder() if isinstance(f, _cr._ComponentFinder) else f\n"
    "                    for f in sys.meta_path]\n"
)


def test_g6_mutation_self_proof_broken_finder_executes_poison():
    """定点变异自证：复刻旧实现（宿主帧无条件 return None）⇒ 毒模块必被执行。"""
    root, poison = _mk_roots()
    r = _run(poison, {"LINGSHU_ALLOW_CWD_IMPORTS": "1", "LINGSHU_COMPONENT_ROOT": root},
             prelude=_BROKEN_PRELUDE)
    ln = _line(r.stdout, WL_NAME)
    marker = os.path.join(poison, "MARKER.txt")
    assert "owner=poison" in ln or os.path.exists(marker), (
        "G6: 变异体未复现缺陷（判别力不足）\n%s\n%s" % (ln, r.stdout))


if __name__ == "__main__":
    _fails = 0
    for _name, _fn in sorted(list(globals().items())):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print("PASS", _name)
            except AssertionError as _e:
                _fails += 1
                print("FAIL", _name, str(_e)[:400])
    print("FAILED" if _fails else "ALL PASS")
    sys.exit(1 if _fails else 0)
