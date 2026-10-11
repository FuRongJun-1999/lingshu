# -*- coding: utf-8 -*-
"""test_issue343_script_dir_guard · lingshu issue #343 守卫（P1 · 代码执行）
============================================================================
缺陷（lingshu issue #343）：「排除脚本目录」这道护栏（`_pathguard.main_script_dir`
→ `component_resolver._is_script_dir_entry`）用 `abspath(argv[0])` 求脚本目录，而
判定另一侧 `sys.path[0]` 由解释器**在进程启动期写定**、且**保留 argv0 的未解析
形态**。两者一旦不同步，同目录毒组件即被当「受控面」导入执行：

  · **import 前 chdir**：启动用相对 argv0（launcher 常态），脚本在 `import lingshu`
    之前 `os.chdir()` ⇒ `abspath(argv0)` 相对**新** cwd 解析、`isfile` 落空 ⇒
    旧实现返回 `None` ⇒ 脚本目录不被排除 ⇒ 毒组件执行。
  · **符号链接 / junction 启动**：`sys.path[0]` 是**别名路径**（未解析），而
    `realpath` 给真实路径——只改一侧会把原本相符的两侧改成不符。

复现读数（修复前，本机）：
  chdir 腿：`sys_path0` = <SCRIPT_DIR>，`main_script_dir` = null，
            `marker` = true，`sce_err` = ""（毒组件被实例化）
  别名腿：`sys_path0` = <ALIAS_DIR>，`dirname(realpath(argv0))` = <REAL_DIR>
            （两者字符串不等 ⇒ 单侧 realpath 必红）

修法（两处**同归一**）：
  · `_pathguard.main_script_dir()`：`realpath` + argv0 不可解析时退回启动期
    `sys.path[0]`（脚本目录，与 cwd 无关）；
  · `_pathguard._norm()` 与 `component_resolver._norm()`：`abspath` → `realpath`
    （同一归一，别名与真实路径判等）。

断言组：
  J1 chdir 腿：import 前 chdir + 相对 argv0 ⇒ 同目录毒组件不执行；`main_script_dir()`
     非空且经 `_norm` 等于 `sys.path[0]`（**点名**：旧实现此处为 None）。
  J2 别名腿：经目录 junction/symlink 启动 ⇒ 真实目录里的毒组件不执行；且
     `_norm(别名) == _norm(真实)`（两处同归一才可比）。无别名能力则 SKIP。
  J3 可比值 + 两副本一致：`_pathguard._norm` 与 `component_resolver._norm` 同归一；
     别名存在时 `abspath(别名) != realpath(别名)`（⇒ 单侧改必红，故须两侧同改）。
  J4 受控面读数：子进程报回 `safe_search_path()`，脚本目录（含别名形态）不在其中。

运行（lingshu 仓根）：python -X utf8 tests/test_issue343_script_dir_guard.py
                    / python -X utf8 -m pytest tests/test_issue343_script_dir_guard.py
退出码：0 = 全过（无别名能力时别名腿 SKIP）；1 = 有断言失败
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

from lingshu._pathguard import _norm as pg_norm, main_script_dir  # noqa: E402
from lingshu.core.component_resolver import _norm as cr_norm  # noqa: E402

_POISON = '''# poison component for issue343 guard (top-level code must NOT run)
import os as _os
with open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "MARKER.txt"),
          "w", encoding="utf-8") as _f:
    _f.write("EXECUTED cwd=%s\\n" % _os.getcwd())
class SelfCognitionEngine:
    def __init__(self, *a, **k): pass
'''

_LAUNCHER = '''import json, os, sys
_mode = sys.argv[1] if len(sys.argv) > 1 else "nochdir"
if _mode == "chdir":                  # chdir 腿：import 前切换 cwd
    os.chdir(sys.argv[2])
import lingshu._pathguard as pg
from lingshu.core.component_resolver import safe_search_path
from lingshu.core.core import SpacetimeMemoryEngine
e = SpacetimeMemoryEngine(":memory:")
here = os.path.dirname(os.path.abspath(__file__))
print("##P##" + json.dumps({
    "sys_path0": sys.path[0],
    "main_script_dir": pg.main_script_dir(),
    "norm_sys0": pg._norm(sys.path[0]),
    "norm_sd": pg._norm(pg.main_script_dir()) if pg.main_script_dir() else None,
    "norm_search": sorted(pg._norm(p) for p in safe_search_path()),
    "marker": os.path.exists(os.path.join(here, "MARKER.txt")),
    "sce_err": repr(e._self_cognition_error),
}, ensure_ascii=False))
'''


def _child_env():
    env = {k: v for k, v in os.environ.items() if not k.startswith("MDCG_")}
    env.update({"PYTHONPATH": REPO, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    env.pop("LINGSHU_COMPONENT_ROOT", None)
    env.pop("LINGSHU_ALLOW_CWD_IMPORTS", None)
    return env


def _run(argv, cwd):
    r = subprocess.run([sys.executable, "-B"] + argv, cwd=cwd, env=_child_env(),
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    payload = None
    for line in (r.stdout or "").splitlines():
        if line.startswith("##P##"):
            payload = json.loads(line[len("##P##"):])
    return payload, r


def _make_alias(target, link):
    """造目录别名：OS 符号链接优先，Windows 回落 junction。返回种类或 None。"""
    try:
        os.symlink(target, link, target_is_directory=True)
        return "symlink"
    except Exception:  # noqa: BLE001
        pass
    if os.name == "nt":
        mk = subprocess.run(["cmd", "/c", "mklink", "/J", link, target],
                            capture_output=True, text=True, encoding="utf-8",
                            errors="replace")
        if mk.returncode == 0 and os.path.isdir(link):
            return "junction"
    return None


def test_j1_chdir_before_import_keeps_script_dir_excluded():
    """J1 import 前 chdir + 相对 argv0 ⇒ 同目录毒组件不执行（点名旧实现 None）。"""
    base = tempfile.mkdtemp(prefix="i343_j1_")
    try:
        script_dir = os.path.join(base, "SCRIPT_DIR")
        start = os.path.join(base, "START")
        other = os.path.join(base, "OTHER", "SUB")
        for d in (script_dir, start, other):
            os.makedirs(d)
        with open(os.path.join(script_dir, "self_cognition_engine.py"), "w",
                  encoding="utf-8") as f:
            f.write(_POISON)
        with open(os.path.join(script_dir, "launcher.py"), "w", encoding="utf-8") as f:
            f.write(_LAUNCHER)
        rel = os.path.relpath(os.path.join(script_dir, "launcher.py"), start)
        payload, r = _run([rel, "chdir", other], start)          # 相对 argv0，从 START 启动
        assert r.returncode == 0, (r.returncode, r.stderr[:400])
        assert payload is not None, (r.stdout, r.stderr[:400])
        assert payload["main_script_dir"] is not None, \
            "main_script_dir() 为 None ⇒ 护栏失明（旧实现读数）"
        assert payload["norm_sd"] == payload["norm_sys0"], \
            "脚本目录两侧不同归一：sd=%r path0=%r" % (payload["norm_sd"], payload["norm_sys0"])
        assert payload["marker"] is False, "同目录毒组件被执行（MARKER 落盘）"
        assert "No module named 'self_cognition_engine'" in payload["sce_err"], \
            "毒组件未按「组件缺失」失败：%r" % payload["sce_err"]
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_j2_alias_launch_keeps_script_dir_excluded():
    """J2 经目录别名（symlink/junction）启动 ⇒ 真实目录里的毒组件不执行。"""
    base = tempfile.mkdtemp(prefix="i343_j2_")
    try:
        real = os.path.join(base, "REAL_DIR")
        os.makedirs(real)
        with open(os.path.join(real, "self_cognition_engine.py"), "w",
                  encoding="utf-8") as f:
            f.write(_POISON)
        with open(os.path.join(real, "launcher.py"), "w", encoding="utf-8") as f:
            f.write(_LAUNCHER)
        alias = os.path.join(base, "ALIAS_DIR")
        kind = _make_alias(real, alias)
        if kind is None:
            print("  SKIP J2：本机无法创建目录符号链接/junction")
            return
        payload, r = _run([os.path.join(alias, "launcher.py"), "nochdir"], base)
        assert r.returncode == 0 and payload is not None, (r.returncode, r.stderr[:400])
        # 两处同归一 ⇒ 别名与真实路径判等（单侧 realpath 时此断言必红）
        assert pg_norm(alias) == pg_norm(real), "别名与真实路径未同归一（两处不可比）"
        assert payload["norm_sd"] == payload["norm_sys0"], \
            "别名腿脚本目录两侧不同归一：sd=%r path0=%r" % (payload["norm_sd"], payload["norm_sys0"])
        assert payload["marker"] is False, "经别名启动时真实目录毒组件被执行"
        assert "No module named 'self_cognition_engine'" in payload["sce_err"], \
            "毒组件未按「组件缺失」失败：%r" % payload["sce_err"]
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_j3_norm_copies_agree_and_alias_needs_two_sided_realpath():
    """J3 两副本同归一 + 单侧 realpath 必然失配（可比值自证）。"""
    assert pg_norm(REPO) == cr_norm(REPO), "两份 _norm 归一不一致"
    base = tempfile.mkdtemp(prefix="i343_j3_")
    try:
        real = os.path.join(base, "REAL_DIR")
        os.makedirs(real)
        alias = os.path.join(base, "ALIAS_DIR")
        kind = _make_alias(real, alias)
        if kind is None:
            print("  SKIP J3 别名腿：本机无法创建目录符号链接/junction")
            return
        # 别名与真实路径的 abspath 形态**不等**（⇒ 只把一侧改 realpath 必失配）
        assert os.path.abspath(alias) != os.path.realpath(alias), \
            "别名未被 realpath 解析，本用例前提不成立"
        assert pg_norm(alias) == pg_norm(real), "realpath 归一未生效"
        assert cr_norm(alias) == cr_norm(real), "component_resolver._norm 未同归一"
        assert os.path.abspath(alias) != os.path.realpath(alias)
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_j4_script_dir_absent_from_controlled_surface():
    """J4 受控面读数：脚本目录（别名形态）不在 safe_search_path() 里。"""
    base = tempfile.mkdtemp(prefix="i343_j4_")
    try:
        script_dir = os.path.join(base, "SCRIPT_DIR")
        start = os.path.join(base, "START")
        other = os.path.join(base, "OTHER", "SUB")
        for d in (script_dir, start, other):
            os.makedirs(d)
        with open(os.path.join(script_dir, "launcher.py"), "w", encoding="utf-8") as f:
            f.write(_LAUNCHER)
        rel = os.path.relpath(os.path.join(script_dir, "launcher.py"), start)
        payload, r = _run([rel, "chdir", other], start)
        assert r.returncode == 0 and payload is not None, (r.returncode, r.stderr[:400])
        assert pg_norm(script_dir) not in payload["norm_search"], \
            "脚本目录仍在受控搜索面内：%r" % (payload["norm_search"],)
    finally:
        shutil.rmtree(base, ignore_errors=True)


_PASS: list = []
_FAIL: list = []


def main() -> int:
    tests = [
        test_j1_chdir_before_import_keeps_script_dir_excluded,
        test_j2_alias_launch_keeps_script_dir_excluded,
        test_j3_norm_copies_agree_and_alias_needs_two_sided_realpath,
        test_j4_script_dir_absent_from_controlled_surface,
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
    print(f"===== SUMMARY {len(_PASS)}/{len(_PASS) + len(_FAIL)} 通过 =====")
    if _FAIL:
        print("FAILS:", _FAIL)
        return 1
    print("VERDICT=PASS（issue #343：脚本目录两侧同归一（realpath + 启动期 sys.path[0] 兜底），"
          "chdir 腿与别名腿的同目录毒组件均不执行，且不在受控搜索面内）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
