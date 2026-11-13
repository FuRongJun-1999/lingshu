# -*- coding: utf-8 -*-
"""#1094：`LINGSHU_COMPONENT_ROOT` 相对根绕过受控面（任意代码执行）的收口守卫。

契约（模块 docstring 既有承诺，本守卫把配置根拉回同一条承诺）：
  · 受控面不由**判定时**的 cwd 现算——相对条目的目标目录由 cwd 决定
    （import 之后的合法 chdir 即可翻转加载源），与快照条目**同口径剔除**；
  · 部署方声明的**绝对**组件根始终可达（正对照，防过度收紧）；
  · `configured_roots()` 是原始配置读数，不做过滤（过滤面在 `safe_search_path()`）。
"""
import os
import subprocess
import sys
import tempfile

import pytest

from lingshu.core import component_resolver as CR

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------- 单元面：safe_search_path 的相对根剔除（纯函数，不 install） ----------------

@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv(CR.COMPONENT_ROOT_ENV, raising=False)
    return monkeypatch


def test_relative_root_excluded(clean_env):
    clean_env.setenv(CR.COMPONENT_ROOT_ENV, "components")
    sp = CR.safe_search_path()
    assert not any(CR._is_relative_entry(p) for p in sp), (
        "相对配置根不得入受控面（其目标由判定时 cwd 决定）：%r" % (sp,))


@pytest.mark.parametrize("rel", [".", "./", ".\\", "", "  "])
def test_dot_forms_excluded(clean_env, rel):
    clean_env.setenv(CR.COMPONENT_ROOT_ENV, rel)
    assert not any(CR._is_relative_entry(p) for p in CR.safe_search_path())


def test_absolute_root_still_included(clean_env):
    # 正对照：部署方声明的绝对组件根始终可达（防过度收紧）。
    root = tempfile.mkdtemp(prefix="c1094_abs_")
    clean_env.setenv(CR.COMPONENT_ROOT_ENV, root)
    assert CR._norm(root) in [CR._norm(p) for p in CR.safe_search_path()]


def test_mixed_entries_only_absolute_kept(clean_env):
    abs_root = tempfile.mkdtemp(prefix="c1094_mix_")
    clean_env.setenv(CR.COMPONENT_ROOT_ENV,
                     os.pathsep.join(["rel-comp", abs_root, "."]))
    sp = CR.safe_search_path()
    assert not any(CR._is_relative_entry(p) for p in sp), sp
    assert CR._norm(abs_root) in [CR._norm(p) for p in sp]


def test_configured_roots_keeps_raw_reading(clean_env):
    # configured_roots() 是原始配置读数：不过滤（诊断面需要看到部署方写了什么）。
    clean_env.setenv(CR.COMPONENT_ROOT_ENV, "components")
    assert CR.configured_roots() == ["components"]


# ---------------- 端到端（子进程，#275 模式）：相对根 + import 后 chdir ----------------

_HOST_MAIN = """
import os, sys
os.chdir(r"%(start)s")
os.environ["LINGSHU_COMPONENT_ROOT"] = %(root)r
import lingshu.core.component_resolver as CR
CR.install()
os.chdir(r"%(then)s")
try:
    import body
    print("LOADED", body.OWNER)
except ModuleNotFoundError:
    print("REFUSED")
"""


def _make_two_deployments():
    base = tempfile.mkdtemp(prefix="c1094_e2e_")
    dirs = {}
    for tag, owner in (("A", "A-deploy"), ("C", "C-attacker")):
        d = os.path.join(base, tag, "components")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "body.py"), "w", encoding="utf-8") as f:
            f.write("OWNER = %r\n" % owner)
        dirs[tag] = os.path.join(base, tag)
    return dirs


def _run_probe(start, then, root_value):
    dirs = _make_two_deployments()
    start = os.path.join(dirs[start]) if start else dirs["A"]
    then = dirs[then]
    main = os.path.join(dirs["A"], "main_probe.py")
    with open(main, "w", encoding="utf-8") as f:
        f.write(_HOST_MAIN % {"start": dirs["A"], "then": dirs["C"],
                              "root": root_value})
    env = {k: v for k, v in os.environ.items()
           if k not in ("LINGSHU_COMPONENT_ROOT",)}
    env.update({"PYTHONPATH": REPO, "PYTHONUTF8": "1",
                "PYTHONDONTWRITEBYTECODE": "1"})
    return subprocess.run([sys.executable, "-B", "-W", "ignore", main],
                          cwd=tempfile.gettempdir(), env=env,
                          capture_output=True, text=True, encoding="utf-8",
                          timeout=120)


def test_relative_root_does_not_load_attacker_cwd_component():
    r = _run_probe("A", "C", "components")     # 相对根：解析目标随 chdir 翻转
    assert "REFUSED" in r.stdout, (
        "相对根仍从判定时 cwd 解析组件（#1094 未收口）\n%s%s" % (r.stdout, r.stderr))
    assert "C-attacker" not in r.stdout


def test_absolute_root_still_loads_deployer_component():
    # 正对照：绝对根不随 chdir 变——部署组件照常可达。
    dirs = _make_two_deployments()
    r = _run_probe("A", "C", os.path.join(dirs["A"], "components"))
    assert "LOADED A-deploy" in r.stdout, (
        "绝对根可达性被修坏（正对照）\n%s%s" % (r.stdout, r.stderr))
    assert "C-attacker" not in r.stdout
