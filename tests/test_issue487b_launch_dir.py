# -*- coding: utf-8 -*-
"""issue #487② 守卫：宿主 import 前 sys.path.insert(vendor) 不得带偏启动面。

病象（修前）：_capture_startup_launch_dir 只取 sys.path[0]——宿主在
import lingshu 之前 sys.path.insert(0, HERE/"vendor")，快照就记成 vendor；
真实脚本目录退到 sys.path[1]，既不是 cwd、也不是快照，遂被当「部署方受控面」，
其下同名毒组件（self_cognition_engine）在引擎构造期执行并读到设计者密钥。

修复语义：main_script_dir() 优先用 argv[0] 的脚本目录（-m 的 __main__.py /
-c / REPL 跳过，回退 sys.path[0] 快照）；argv[0] 目录必须在 sys.path 里
（脚本形态必在；-m 的模块目录通常不在），避免把 -m 模块目录误当启动面。

运行：python -X utf8 tests/test_issue487b_launch_dir.py
自证：python -X utf8 tests/test_issue487b_launch_dir.py --mutate
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_CHECKS = []
FAILS = []

_BODY = '''import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
if os.environ.get("DO_INSERT"):
    sys.path.insert(0, os.path.join(HERE, "vendor"))
from lingshu.core.core import SpacetimeMemoryEngine
import lingshu._pathguard as pg
from lingshu.core import component_resolver as cr
e = SpacetimeMemoryEngine(":memory:")
print("RESULT=" + json.dumps({
    "main": pg.main_script_dir(),
    "safe": [os.path.realpath(p) for p in cr.safe_search_path()],
    "self": type(getattr(e, "_self_cognition", None)).__name__,
}, ensure_ascii=False))
'''
_POISON = '''import os
print("[POISON] self_cognition_engine executed from", __file__,
      "| key=", os.environ.get("AEIS_DESIGNER_KEY"))
class SelfCognitionEngine:
    def __init__(self, engine): pass
    def log_action(self, *a, **k): pass
'''


def _check(name, cond, detail=""):
    if name not in _CHECKS:
        _CHECKS.append(name)
    if cond:
        print("  PASS " + name)
    else:
        print("  FAIL " + name + "  " + str(detail)[:240])
        FAILS.append(name)


def _run_case(kind, insert, comp_root=None, comp_root_self=False):
    base = os.path.realpath(tempfile.mkdtemp(prefix="r487b_"))
    try:
        app = os.path.join(base, "app")
        os.makedirs(os.path.join(app, "vendor"))
        for name in ("main.py", "mod.py", "__main__.py"):
            with open(os.path.join(app, name), "w", encoding="utf-8") as fh:
                fh.write(_BODY)
        open(os.path.join(app, "__init__.py"), "w", encoding="utf-8").write("")
        with open(os.path.join(app, "self_cognition_engine.py"), "w",
                  encoding="utf-8") as fh:
            fh.write(_POISON)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(_REPO)
        env["AEIS_DESIGNER_KEY"] = "designer-secret-123"
        env.pop("LINGSHU_COMPONENT_ROOT", None)
        if insert:
            env["DO_INSERT"] = "1"
        if comp_root_self:
            env["LINGSHU_COMPONENT_ROOT"] = app
        elif comp_root is not None:
            env["LINGSHU_COMPONENT_ROOT"] = comp_root
        if kind == "script":
            cmd = [sys.executable, "-X", "utf8", os.path.join(app, "main.py")]
            cwd = base
        elif kind == "m_mod":
            cmd = [sys.executable, "-X", "utf8", "-m", "app.mod"]
            cwd = base
        else:
            cmd = [sys.executable, "-X", "utf8", "-m", "app"]
            cwd = base
        r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=120)
        data = {}
        for line in (r.stdout or "").splitlines():
            if line.startswith("RESULT="):
                try:
                    data = json.loads(line[len("RESULT="):])
                except json.JSONDecodeError:
                    data = {}
        data["poison"] = "[POISON]" in (r.stdout or "")
        data["rc"] = r.returncode
        data["app"] = app
        data["base"] = base
        return data
    finally:
        shutil.rmtree(base, ignore_errors=True)


def _assertions():
    print("== A 组：脚本形态（python app/main.py）==")
    r = _run_case("script", insert=False)
    app = os.path.realpath(os.path.join(r["base"], "app"))
    _check("A01 前提：不 insert 时 main==app、脚本目录不在受控面、无毒执行",
           r["rc"] == 0 and os.path.realpath(r.get("main") or "") == app
           and app not in r.get("safe", []) and r.get("self") == "NoneType"
           and not r["poison"], r)

    r = _run_case("script", insert=True)
    app = os.path.realpath(os.path.join(r["base"], "app"))
    _check("A02 ★行：insert vendor 后 main 仍是 app、脚本目录不在受控面、无毒执行",
           r["rc"] == 0 and os.path.realpath(r.get("main") or "") == app
           and app not in r.get("safe", []) and r.get("self") == "NoneType"
           and not r["poison"], r)

    r2 = _run_case("script", insert=True, comp_root_self=True)
    _check("A03 ★对照：显式 LINGSHU_COMPONENT_ROOT=app 时毒组件可达（非一刀切）",
           r2["rc"] == 0 and r2.get("self") == "SelfCognitionEngine"
           and r2["poison"], r2)

    print("== B 组：-m 形态 ==")
    r = _run_case("m_mod", insert=True)
    app = os.path.realpath(os.path.join(r["base"], "app"))
    _check("B01 ★行：-m app.mod 下不把模块目录当启动面、无毒执行",
           r["rc"] == 0 and os.path.realpath(r.get("main") or "") != app
           and r.get("self") == "NoneType" and not r["poison"], r)

    r = _run_case("m_pkg", insert=True)
    app = os.path.realpath(os.path.join(r["base"], "app"))
    _check("B02 ★行：-m app（__main__.py）下不把包目录当启动面、无毒执行",
           r["rc"] == 0 and os.path.realpath(r.get("main") or "") != app
           and r.get("self") == "NoneType" and not r["poison"], r)

    print("== E 组：源码钉子 ==")

    def _src(rel):
        try:
            return (_REPO / rel).read_text(encoding="utf-8")
        except OSError:
            return ""

    src = _src("lingshu/_pathguard.py")
    _check("E01 ★钉子：argv[0] 脚本目录优先", "_STARTUP_ARGV_SCRIPT_DIR is not None" in src)
    _check("E02 ★钉子：argv[0] 目录必须在 sys.path + __main__.py 跳过",
           "not in keys" in src and '!= "__main__.py"' in src)


def _run_assertions():
    global FAILS
    FAILS = []
    del _CHECKS[:]
    _assertions()
    print("通过 %d / 失败 %d" % (len(_CHECKS) - len(FAILS), len(FAILS)))
    return len(FAILS)


_ARMS = {
    "M1": (
        "lingshu/_pathguard.py",
        [("    if _STARTUP_ARGV_SCRIPT_DIR is not None:",
          "    if False:")],
        {"A02", "E01"},
    ),
    "M2": (
        "lingshu/_pathguard.py",
        [("        if _os.path.normcase(_os.path.realpath(d)) not in keys:",
          "        if False:")],
        {"B01", "E02"},
    ),
    "M3": (
        "lingshu/_pathguard.py",
        [('        if not (_os.path.isfile(ap) and _os.path.basename(ap) != "__main__.py"):',
          "        if not _os.path.isfile(ap):"),
         ("        if _os.path.normcase(_os.path.realpath(d)) not in keys:",
          "        if False:")],
        {"B01", "B02", "E02"},
    ),
}


def _subprocess_fail_names():
    r = subprocess.run([sys.executable, "-X", "utf8", __file__, "--assert-only"],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=str(_REPO), timeout=900)
    names = set(re.findall(r"(?m)^\s*FAIL\s+([A-Z]\d+)", r.stdout or ""))
    return names, (r.stdout or "") + (r.stderr or "")


def _mutate(argv):
    want = ""
    for a in argv:
        if a == "--mutate":
            want = want or "ALL"
        elif a.startswith("--mutate="):
            want = a.split("=", 1)[1]
    if not want:
        return 0
    ids = sorted(_ARMS) if want == "ALL" else [want]
    red = 0
    for mid in ids:
        if mid not in _ARMS:
            print("[ANCHOR-MISS] %s：未登记该变异臂" % mid)
            return 2
        rel, edits, expect = _ARMS[mid]
        target = _REPO / rel
        orig = target.read_bytes()
        try:
            text = orig.decode("utf-8")
            nl = "\r\n" if "\r\n" in text else "\n"
            norm = text.replace("\r\n", "\n")
            for old, new in edits:
                cnt = norm.count(old)
                if cnt != 1:
                    print("[ANCHOR-MISS] %s：锚点命中 %d 次（期望 1）" % (mid, cnt))
                    return 2
                norm = norm.replace(old, new)
            target.write_bytes(norm.replace("\n", nl).encode("utf-8"))
            actual, out = _subprocess_fail_names()
            if actual == expect:
                print("[MUTATE-OK]   %s：指定断言转红 %s" % (mid, sorted(actual)))
            else:
                print("[MUTATE-FAIL] %s：期望红 %s，实际红 %s"
                      % (mid, sorted(expect), sorted(actual)))
                for line in out.splitlines():
                    if line.strip().startswith("FAIL"):
                        print("    " + line)
                red += 1
        finally:
            target.write_bytes(orig)
    print("反证结果：%d 条变异，%s" % (len(ids), "全部恰好命中" if red == 0 else "%d 条未命中" % red))
    return 1 if red else 0


if __name__ == "__main__":
    _f = _run_assertions()
    if "--assert-only" in sys.argv:
        sys.exit(1 if _f else 0)
    _m = _mutate(sys.argv[1:])
    sys.exit(1 if (_f or _m) else 0)
