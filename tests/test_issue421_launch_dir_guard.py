# -*- coding: utf-8 -*-
"""test_issue421_launch_dir_guard · 启动面护栏（lingshu issue #421 · 附 #381 回归锁定）
============================================================================
缺陷（lingshu issue #421 · 代码执行）：
  `_pathguard.main_script_dir()`（「排除解释器注入的脚本目录」这道护栏）把启动面
  按**判定时**的 `sys.argv[0]` / `os.getcwd()` / `sys.path[0]` 现算：

    · `python -m <mod>` 形态下解释器注入的启动面是**启动 cwd（包根）**，
      `sys.path[0]` 就是它；而 argv0 是**模块文件**路径 ⇒ `dirname(argv0)` 给的是
      模块目录（如 `<LAUNCH>/pkg`），**不等**于启动面（`<LAUNCH>`）。
    · 启动器在 `import lingshu` **之前** `os.chdir()` 到别处（如自身数据目录）后，
      `os.getcwd()` 已变 ⇒ `_is_cwd_entry(<LAUNCH>)` 不再命中（逃过 scrub）。

  两条叠加 ⇒ `<LAUNCH>` 被当「受控面」保留：其下放一个同名白名单组件
  （`self_cognition_engine.py`）即被组件守卫找到并**在引擎进程内执行**。

复现读数（修复前，本机 · `python -B -m pkg.launcher` 自 <LAUNCH> 启动后 chdir 到
<DATA>，<LAUNCH>/self_cognition_engine.py 为毒组件）：
  main_script_dir = <LAUNCH>\\pkg（模块目录，非启动面）
  launch_in_search = true   marker = true   sce_err = ''
  （毒组件被实例化，MARKER.txt 落盘）

修法（`lingshu/_pathguard.py`，最小改动）：
  · 新增 `_STARTUP_LAUNCH_DIR = _capture_startup_launch_dir()`——在**本模块 import
    时**（`scrub()` 之前）冻结 `sys.path[0]`（要求绝对路径；`-c`/REPL 的 `''` 与
    `-P`/`PYTHONSAFEPATH` 下解释器不注入 ⇒ 记 None）；
  · `main_script_dir()` 改以该**冻结快照**为唯一真源（`realpath` 归一），不再现算；
  · `_capture_controlled_surface()` 既有的「排除脚本目录」于是自动覆盖 `-m` 启动面。

断言组（回退到「按判定时 cwd/argv0 现算」的旧实现即红并点名）：
  L1 #421 端到端：`-m` 启动 + import 前 chdir ⇒ ①启动面**确实在** `sys.path` 里
     （非空转），②但它**不在** `safe_search_path()` 受控面内，③毒组件未被导入执行
     （MARKER 不落盘、按原生「组件缺失」失败），④`main_script_dir()` 报**启动面**
     （点名旧读数：`<LAUNCH>\\pkg` 模块目录）。
  L2 冻结：再 chdir 进启动面本身（cwd 与启动面重合）⇒ 受控面读数不变、启动面仍被排除。
  L3 #381 回归锁定：启动期相对路径条目（`./rel`）不被当受控面、其下毒组件不执行。
     （实读判定：该条在 HEAD 已由 #288 · D-2 `_is_relative_entry` 修好；本腿锁住它。）
  L4 正对照（防过度收紧）：绝对路径的部署方组件目录（`PYTHONPATH`）仍被接受、
     组件照常装配。

运行（lingshu 仓根）：python -X utf8 tests/test_issue421_launch_dir_guard.py
                    / python -X utf8 -m pytest tests/test_issue421_launch_dir_guard.py
退出码：0 = 全过；1 = 有断言失败
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lingshu._pathguard import _norm as pg_norm  # noqa: E402

# 毒组件：导入期顶层代码落盘 MARKER（必须不执行）
_POISON = '''# poison component for issue421 guard (top-level code must NOT run)
import os as _os
with open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "MARKER.txt"),
          "w", encoding="utf-8") as _f:
    _f.write("EXECUTED cwd=%s\\n" % _os.getcwd())
class SelfCognitionEngine:
    def __init__(self, *a, **k): pass
'''

# 启动器（放在 <LAUNCH>/pkg/launcher.py，以 `python -m pkg.launcher` 启动）
# 自报读数：启动面 = dirname(dirname(__file__)) = <LAUNCH>（`-m` 形态的 sys.path[0]）
_LAUNCHER = '''# -*- coding: utf-8 -*-
import json, os, sys
_mode = sys.argv[1] if len(sys.argv) > 1 else "chdir"
if _mode in ("chdir", "chdir_twice"):
    os.chdir(sys.argv[2])
import lingshu._pathguard as pg
from lingshu.core.component_resolver import safe_search_path
from lingshu.core.core import SpacetimeMemoryEngine
if _mode == "chdir_twice":
    os.chdir(sys.argv[3])                    # 第二次 chdir：cwd 落回启动面之内
e = SpacetimeMemoryEngine(":memory:")
launch = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("##P##" + json.dumps({
    "p0": sys.path[0],
    "cwd": os.getcwd(),
    "launch": launch,
    "main_script_dir": pg.main_script_dir(),
    "launch_in_syspath": (pg._norm(launch) in {pg._norm(p) for p in sys.path}),
    "launch_in_search": (pg._norm(launch) in {pg._norm(p) for p in safe_search_path()}),
    "launch_is_script_dir_entry": pg.is_script_dir_entry(launch),
    "search": sorted(pg._norm(p) for p in safe_search_path()),
    "marker": os.path.exists(os.path.join(launch, "MARKER.txt")),
    "sce_ok": e._self_cognition is not None,
    "sce_err": repr(e._self_cognition_error),
}, ensure_ascii=False))
'''

# 相对条目探针（#381）：启动期在 sys.path 里塞相对路径条目 `./rel`
_REL_PROBE = '''# -*- coding: utf-8 -*-
import json, os, sys
sys.path.insert(0, "./rel")                  # 相对路径条目（启动期面）
import lingshu._pathguard as pg
from lingshu.core.component_resolver import safe_search_path
from lingshu.core.core import SpacetimeMemoryEngine
e = SpacetimeMemoryEngine(":memory:")
rel = os.path.abspath("./rel")
print("##P##" + json.dumps({
    "rel_in_syspath": pg._norm(rel) in {pg._norm(p) for p in sys.path},
    "rel_in_search": pg._norm(rel) in {pg._norm(p) for p in safe_search_path()},
    "marker": os.path.exists(os.path.join(rel, "MARKER.txt")),
    "sce_ok": e._self_cognition is not None,
    "sce_err": repr(e._self_cognition_error),
}, ensure_ascii=False))
'''

_STUB_SCE = '''class SelfCognitionEngine:
    def __init__(self, *a, **k): pass
'''


def _child_env(path=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("MDCG_")}
    env.update({"PYTHONPATH": path if path is not None else REPO,
                "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    env.pop("LINGSHU_COMPONENT_ROOT", None)
    env.pop("LINGSHU_ALLOW_CWD_IMPORTS", None)
    return env


def _run(argv, cwd, path=None):
    r = subprocess.run([sys.executable, "-B"] + argv, cwd=cwd, env=_child_env(path),
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    payload = None
    for line in (r.stdout or "").splitlines():
        if line.startswith("##P##"):
            payload = json.loads(line[len("##P##"):])
    return payload, r


def _layout(base, name):
    """建 <LAUNCH>/pkg/launcher.py + <LAUNCH>/self_cognition_engine.py（毒）+ <DATA>。"""
    launch = os.path.join(base, name)
    pkg = os.path.join(launch, "pkg")
    data = os.path.join(base, name + "_DATA")
    os.makedirs(pkg, exist_ok=True)
    os.makedirs(data, exist_ok=True)
    open(os.path.join(pkg, "__init__.py"), "w").close()
    with open(os.path.join(pkg, "launcher.py"), "w", encoding="utf-8") as f:
        f.write(_LAUNCHER)
    with open(os.path.join(launch, "self_cognition_engine.py"), "w", encoding="utf-8") as f:
        f.write(_POISON)
    return launch, pkg, data


def test_l1_m_launch_dir_is_not_a_controlled_surface():
    """L1 (#421 端到端)：-m 启动 + import 前 chdir ⇒ 启动面不入受控面、毒组件不执行。"""
    base = tempfile.mkdtemp(prefix="i421_l1_")
    try:
        launch, pkg, data = _layout(base, "LAUNCH")
        payload, r = _run(["-m", "pkg.launcher", "chdir", data], launch)
        assert r.returncode == 0 and payload is not None, (r.returncode, r.stderr[:400])
        # ① 非空转：启动面确实在 sys.path 里（护栏要排除的正是这条真在场的条目）
        assert payload["launch_in_syspath"] is True, \
            "用例前提不成立：启动面不在 sys.path 里 ⇒ 本腿空转（%r）" % payload["search"]
        # ② 启动面不在受控面内
        assert payload["launch_in_search"] is False, \
            "启动面仍被当受控面保留（issue #421）：search=%r" % (payload["search"],)
        # ③ 毒组件未被导入执行，按原生「组件缺失」失败
        assert payload["marker"] is False, "启动面下同名毒组件被执行（MARKER 落盘）"
        assert payload["sce_err"] == "\"No module named 'self_cognition_engine'\"", \
            "毒组件未按「组件缺失」失败：%r" % (payload["sce_err"],)
        # ④ 点名旧读数：main_script_dir() 必须报**启动面**，而非 argv0 的模块目录
        assert payload["main_script_dir"] is not None, "main_script_dir() 为 None ⇒ 护栏失明"
        assert pg_norm(payload["main_script_dir"]) == pg_norm(launch), \
            "main_script_dir() 未报启动面（旧实现读数 <LAUNCH>/pkg）：%r" % (
                payload["main_script_dir"],)
        assert pg_norm(payload["main_script_dir"]) == pg_norm(payload["p0"]), \
            "main_script_dir() 与启动期 sys.path[0] 不同源：sd=%r p0=%r" % (
                payload["main_script_dir"], payload["p0"])
        assert pg_norm(payload["main_script_dir"]) != pg_norm(pkg), \
            "main_script_dir() 仍按 argv0 给模块目录（旧实现读数）：%r" % (
                payload["main_script_dir"],)
        # ⑤ 机制点名：启动面被判为「解释器注入的启动面」条目（component_resolver
        #    正是靠 `_is_script_dir_entry` 把它挡在受控面外）
        assert payload["launch_is_script_dir_entry"] is True, \
            "启动面未被判为启动面条目 ⇒ 排除机制未生效（%r）" % (payload["main_script_dir"],)
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_l2_launch_surface_exclusion_is_frozen_across_second_chdir():
    """L2 冻结：再 chdir 进启动面本身 ⇒ 启动面仍被排除、读数不变。"""
    base = tempfile.mkdtemp(prefix="i421_l2_")
    try:
        launch, pkg, data = _layout(base, "LAUNCH")
        payload, r = _run(["-m", "pkg.launcher", "chdir_twice", data, launch], launch)
        assert r.returncode == 0 and payload is not None, (r.returncode, r.stderr[:400])
        assert pg_norm(payload["cwd"]) == pg_norm(launch), \
            "用例前提不成立：第二次 chdir 未落到启动面（cwd=%r）" % (payload["cwd"],)
        assert payload["launch_in_syspath"] is True, "启动面不在 sys.path 里 ⇒ 本腿空转"
        assert payload["launch_in_search"] is False, \
            "cwd 与启动面重合后启动面重新入受控面：search=%r" % (payload["search"],)
        assert payload["marker"] is False, "启动面下毒组件被执行（MARKER 落盘）"
        assert pg_norm(payload["main_script_dir"]) == pg_norm(launch), \
            "main_script_dir() 未报启动面：%r" % (payload["main_script_dir"],)
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_l3_relative_sys_path_entry_is_not_a_controlled_surface():
    """L3 (#381 回归锁定)：启动期相对路径条目不入受控面、其下毒组件不执行。

    实读判定：本缺陷在 HEAD 已由 #288 · D-2（`_is_relative_entry` + 启动期快照）
    修好；本腿把「相对条目不构成受控面」钉在**执行后果**层（旧 #288 K2 只查面清单）。
    """
    base = tempfile.mkdtemp(prefix="i421_l3_")
    try:
        work = os.path.join(base, "work")
        rel = os.path.join(work, "rel")
        os.makedirs(rel)
        with open(os.path.join(rel, "self_cognition_engine.py"), "w", encoding="utf-8") as f:
            f.write(_POISON)
        payload, r = _run(["-c", _REL_PROBE], work)
        assert r.returncode == 0 and payload is not None, (r.returncode, r.stderr[:400])
        assert payload["rel_in_syspath"] is True, \
            "用例前提不成立：相对条目不在 sys.path 里 ⇒ 本腿空转"
        assert payload["rel_in_search"] is False, \
            "相对路径条目被当受控面（issue #381 复发）"
        assert payload["marker"] is False, "相对条目下同名毒组件被执行（MARKER 落盘）"
        assert payload["sce_err"] == "\"No module named 'self_cognition_engine'\"", \
            "毒组件未按「组件缺失」失败：%r" % (payload["sce_err"],)
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_l4_positive_control_absolute_deployment_dir_still_accepted():
    """L4 正对照：绝对路径部署方组件目录（PYTHONPATH）仍被接受（防过度收紧）。"""
    base = tempfile.mkdtemp(prefix="i421_l4_")
    try:
        launch, pkg, data = _layout(base, "LAUNCH")
        comp = os.path.join(base, "comp")
        os.makedirs(comp)
        with open(os.path.join(comp, "self_cognition_engine.py"), "w", encoding="utf-8") as f:
            f.write(_STUB_SCE)
        payload, r = _run(["-m", "pkg.launcher", "chdir", data], launch,
                          path=comp + os.pathsep + REPO)
        assert r.returncode == 0 and payload is not None, (r.returncode, r.stderr[:400])
        assert pg_norm(comp) in payload["search"], \
            "绝对路径部署方目录被误排除（过度收紧）：search=%r" % (payload["search"],)
        assert payload["sce_ok"] is True and "No module named" not in payload["sce_err"], \
            "部署方目录里的组件未被装配（过度收紧）：%r" % (payload["sce_err"],)
        assert payload["marker"] is False, "装配取了启动面下的毒组件而非部署方目录（MARKER 落盘）"
    finally:
        shutil.rmtree(base, ignore_errors=True)


_PASS: list = []
_FAIL: list = []


def main() -> int:
    tests = [
        test_l1_m_launch_dir_is_not_a_controlled_surface,
        test_l2_launch_surface_exclusion_is_frozen_across_second_chdir,
        test_l3_relative_sys_path_entry_is_not_a_controlled_surface,
        test_l4_positive_control_absolute_deployment_dir_still_accepted,
    ]
    for t in tests:
        try:
            t()
            _PASS.append(t.__name__)
            print("  PASS " + t.__name__)
        except AssertionError as exc:
            _FAIL.append((t.__name__, str(exc)))
            print("  FAIL %s ← %s" % (t.__name__, exc))
        except Exception as exc:  # noqa: BLE001
            _FAIL.append((t.__name__, repr(exc)))
            print("  ERROR %s ← %r" % (t.__name__, exc))
    print()
    print("===== SUMMARY %d/%d 通过 =====" % (len(_PASS), len(_PASS) + len(_FAIL)))
    if _FAIL:
        print("FAILS:", _FAIL)
        return 1
    print("VERDICT=PASS（issue #421：启动面按**启动期快照**判定——`-m` 启动 cwd 不被"
          "当受控面、其下毒组件不执行、main_script_dir() 报启动面；#381 相对条目"
          "回归锁定；绝对部署方目录仍被接受（正对照））")
    return 0


def test_issue421_launch_dir_guard():
    """pytest 入口：与脚本式 main() 同一套断言。"""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
