# -*- coding: utf-8 -*-
"""test_issue288_host_purity_d2 · 导入期宿主纯净性 · D-2 守卫（lingshu issue #288）
============================================================================
背景（lingshu issue #288 · 关联 #273 / #275）：
  白名单组件的「受控面」此前由**判定时**的 `sys.path` ＋ **判定时 cwd** 现算——
  `component_resolver.safe_search_path()` 逐条判 `_is_cwd_entry`（吃 `os.getcwd()`）
  与 `_is_script_dir_entry`。于是同一个 `sys.path` 条目是否被当作受控面，会随
  **判定时刻的 cwd** 变化（相对条目的目标由 cwd 决定）。这是 #273/#275 的根因面。

裁定（维护者 2026-10-10，采 **D-2** · 原 E-1 选项 A「按判定时 cwd 清洗」已否）：
  受控面由**进程启动时刻**确定（或由显式配置 `LINGSHU_COMPONENT_ROOT` 声明），
  **不再随 cwd 变化**；**相对路径条目不得**因为「恰好当前 cwd 在受控目录内」
  就被当作受控。

修法（本仓）：
  · `lingshu/_pathguard.py`：新增启动期快照 `_capture_controlled_surface()` /
    `startup_controlled_entries()`（模块 import 时冻结：绝对、非 cwd、非脚本目录）；
  · `lingshu/core/component_resolver.py`：`safe_search_path()` 改取该快照，
    并显式排除相对路径条目（`_is_relative_entry`，**不看 cwd**）。

断言组（回退到吃 cwd 的旧实现即红）：
  K1 冻结：启动后 `os.chdir()` 到别处 ⇒ 受控面（`safe_search_path()` 与
      `startup_controlled_entries()`）**逐项不变**。
  K2 相对条目：启动期在 `sys.path` 里的**相对路径条目**不被当作受控面
      （无论判定时 cwd 是否恰好在其目录内）。
  K3 **正对照（防过度收紧）**：绝对路径的合法受控目录（`PYTHONPATH` 指向的
      组件目录）**仍被接受**——组件照常装配、无「组件缺失」告警。
  K4 启动后新插入的条目不进受控面（受控面 = 启动时刻确定，非「当前 sys.path」）。

定点变异自证（见 `__main__` 提示）：
  把 `safe_search_path()` 改回「逐条吃 cwd」的旧实现 ⇒ K1/K2 必红并点名；
  恢复后复绿。

运行（仓根）：python -X utf8 tests/test_issue288_host_purity_d2.py
             / python -X utf8 -m pytest tests/test_issue288_host_purity_d2.py
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

_PASS: list = []
_FAIL: list = []


def ok(cond, msg, extra=""):
    (_PASS if cond else _FAIL).append(msg)
    print(("  PASS " if cond else "  FAIL ") + msg
          + (("  ← " + str(extra)) if (extra and not cond) else ""))


# ------------------------------------------------------------------ 子进程夹具
# 启动期在 sys.path 里塞一个**相对条目** `sub`（Python 自身会把 PYTHONPATH 绝对化，
# 故只能由代码在 import lingshu **之前**插入——这恰是「启动期受控面」的定义面）。
_PROBE = '''# -*- coding: utf-8 -*-
import json, os, sys
sys.path.insert(0, "sub")                      # 相对条目：启动期面
import lingshu._pathguard as pg
from lingshu.core.component_resolver import safe_search_path
rel_key = pg._norm("sub")                      # 在当前 cwd 下解析相对条目
def snap():
    return {
        "search": sorted(pg._norm(p) for p in safe_search_path()),
        "entries": [pg._norm(p) for p in pg.startup_controlled_entries()],
        "rel_in_search": rel_key in {pg._norm(p) for p in safe_search_path()},
    }
before = snap()
os.chdir(os.path.join(os.getcwd(), "sub"))     # 启动后 chdir 到别处
after = snap()
# 启动后再插入一个相对条目（不应进受控面：受控面 = 启动时刻确定）
sys.path.insert(0, "sub")
post = snap()
print("##P##" + json.dumps({
    "before": before, "after": after, "post": post,
    "rel_key": rel_key,
    "repo_key": pg._norm(%r),
    "cwd0": os.getcwd(),
}, ensure_ascii=False))
''' % (REPO,)

_ASSEMBLE_PROBE = '''# -*- coding: utf-8 -*-
import json, os, sys
from lingshu.core.core import SpacetimeMemoryEngine
e = SpacetimeMemoryEngine(":memory:")
print("##P##" + json.dumps({
    "sce_ok": e._self_cognition is not None,
    "sce_err": repr(e._self_cognition_error),
    "search_count": len(__import__("lingshu.core.component_resolver",
                                   fromlist=["safe_search_path"]).safe_search_path()),
}, ensure_ascii=False))
'''

_STUB_SCE = '''class SelfCognitionEngine:
    def __init__(self, *a, **k): pass
'''


def _child_env(path=None, extra=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("MDCG_")}
    env["PYTHONPATH"] = path if path is not None else REPO
    env["PYTHONUTF8"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("LINGSHU_COMPONENT_ROOT", None)
    env.pop("LINGSHU_ALLOW_CWD_IMPORTS", None)
    if extra:
        env.update(extra)
    return env


def _run(argv, cwd, path=None, extra=None):
    r = subprocess.run([sys.executable, "-B"] + argv, cwd=cwd,
                       env=_child_env(path, extra), capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    payload = None
    for line in (r.stdout or "").splitlines():
        if line.startswith("##P##"):
            payload = json.loads(line[len("##P##"):])
    return payload, r


# ------------------------------------------------------------------ 断言组

def k1_controlled_surface_is_frozen_across_chdir(base):
    """K1 启动后 chdir ⇒ 受控面逐项不变（冻结）。"""
    print("== K1：启动后 chdir 到别处 ⇒ 受控面判定不变 ==")
    work = os.path.join(base, "k1")
    os.makedirs(os.path.join(work, "sub"), exist_ok=True)
    payload, r = _run(["-c", _PROBE], work)
    ok(r.returncode == 0 and payload is not None,
       "K1a 子进程取到受控面读数", (r.stderr or "")[:400])
    if payload is None:
        return
    ok(payload["before"]["search"] == payload["after"]["search"],
       "K1b chdir 后 safe_search_path() 不变（冻结；旧实现吃 cwd 会变）",
       "before=%r after=%r" % (payload["before"]["search"], payload["after"]["search"]))
    ok(payload["before"]["entries"] == payload["after"]["entries"],
       "K1c chdir 后 startup_controlled_entries() 不变",
       "before=%r after=%r" % (payload["before"]["entries"], payload["after"]["entries"]))


def k2_relative_entry_is_not_controlled(base):
    """K2 相对路径条目不被当作受控面。"""
    print("== K2：相对路径 sys.path 条目不进受控面（#273/#275 核心面） ==")
    work = os.path.join(base, "k2")
    os.makedirs(os.path.join(work, "sub"), exist_ok=True)
    payload, r = _run(["-c", _PROBE], work)
    ok(r.returncode == 0 and payload is not None,
       "K2a 子进程取到读数", (r.stderr or "")[:400])
    if payload is None:
        return
    ok(payload["before"]["rel_in_search"] is False,
       "K2b 相对条目（cwd 下的 `sub`）不在受控面内（启动期）",
       "rel_key=%r search=%r" % (payload["rel_key"], payload["before"]["search"]))
    ok(payload["after"]["rel_in_search"] is False,
       "K2c chdir 进该相对条目目录后，它**仍不**被当作受控面",
       "rel_key=%r search=%r" % (payload["rel_key"], payload["after"]["search"]))


def k4_surface_is_startup_snapshot_not_live_syspath(base):
    """K4 启动后新插入的条目不进受控面（受控面 = 启动时刻，非当前 sys.path）。"""
    print("== K4：启动后 sys.path 变动不改受控面（快照语义） ==")
    work = os.path.join(base, "k4")
    os.makedirs(os.path.join(work, "sub"), exist_ok=True)
    payload, r = _run(["-c", _PROBE], work)
    ok(r.returncode == 0 and payload is not None, "K4a 子进程取到读数",
       (r.stderr or "")[:400])
    if payload is None:
        return
    ok(payload["before"]["search"] == payload["post"]["search"],
       "K4b 启动后再 sys.path.insert 相对条目 ⇒ 受控面不变",
       "before=%r post=%r" % (payload["before"]["search"], payload["post"]["search"]))


def k3_positive_control_legal_absolute_dir_still_accepted(base):
    """K3 正对照：绝对路径的合法受控目录仍被接受（防过度收紧）。"""
    print("== K3：正对照——绝对路径合法受控目录仍被接受（组件照常装配） ==")
    comp = os.path.join(base, "k3_components")
    os.makedirs(comp, exist_ok=True)
    with open(os.path.join(comp, "self_cognition_engine.py"), "w", encoding="utf-8") as f:
        f.write(_STUB_SCE)
    work = os.path.join(base, "k3_work")
    os.makedirs(work, exist_ok=True)
    # 绝对路径组件目录经 PYTHONPATH（部署方显式面）声明
    path = comp + os.pathsep + REPO
    payload, r = _run(["-c", _ASSEMBLE_PROBE], work, path=path)
    ok(r.returncode == 0 and payload is not None,
       "K3a 子进程构造引擎成功", (r.stderr or "")[:400])
    if payload is None:
        return
    ok(payload["sce_ok"] is True and "No module named" not in payload["sce_err"],
       "K3b 绝对路径合法受控目录里的组件**仍被装配**（未过度收紧）",
       payload)
    # 正对照②：LINGSHU_COMPONENT_ROOT 显式根（永不受 cwd/脚本目录判定影响）
    payload2, r2 = _run(["-c", _ASSEMBLE_PROBE], work,
                        path=REPO, extra={"LINGSHU_COMPONENT_ROOT": comp})
    ok(r2.returncode == 0 and payload2 is not None and payload2["sce_ok"] is True,
       "K3c LINGSHU_COMPONENT_ROOT 显式根仍被接受（显式声明面不受 cwd 影响）",
       (r2.stdout or "")[:300])


def k0_guard_self_check():
    """量具自检：受控面快照必须真的非空（否则断言恒真、无判别力）。"""
    from lingshu._pathguard import startup_controlled_entries
    from lingshu.core.component_resolver import safe_search_path
    entries = startup_controlled_entries()
    ok(len(entries) >= 1,
       "K0a 启动期受控面快照非空（否则本守卫恒真空转）", entries)
    ok(len(safe_search_path()) >= 1,
       "K0b safe_search_path() 非空", safe_search_path())


def main() -> int:
    print("# 定点变异自证：把 safe_search_path() 改回逐条吃 cwd 的旧实现 ⇒ K1/K2 必红并点名。")
    base = tempfile.mkdtemp(prefix="i288_d2_")
    print("# guard temp base: %s" % base)
    try:
        k0_guard_self_check()
        k1_controlled_surface_is_frozen_across_chdir(base)
        k2_relative_entry_is_not_controlled(base)
        k4_surface_is_startup_snapshot_not_live_syspath(base)
        k3_positive_control_legal_absolute_dir_still_accepted(base)
    finally:
        shutil.rmtree(base, ignore_errors=True)
    print()
    print("===== SUMMARY %d/%d 通过 =====" % (len(_PASS), len(_PASS) + len(_FAIL)))
    if _FAIL:
        print("FAILS:")
        for m in _FAIL:
            print("  -", m)
        return 1
    print("VERDICT=PASS（issue #288 · D-2：受控面 = 启动期快照（不吃判定时 cwd）；"
          "相对路径条目不被当作受控面；绝对路径合法受控目录仍被接受（正对照））")
    return 0


def test_issue288_host_purity_d2():
    """pytest 入口：与脚本式 main() 同一套断言。"""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
