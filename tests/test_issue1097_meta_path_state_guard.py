# -*- coding: utf-8 -*-
"""#1097：`component_resolver` reload 后状态机失配与 finder 无界堆积的收口守卫。

契约（docstring 既有承诺）：
  · `install()` 幂等、可重复调用——reload 后再 install **不净增** finder；
  · `is_installed()` / `discovery_report()["guard_installed"]` 与**实际拦截行为**
    一致（reload 后旧实例仍在 `sys.meta_path` 拦截 ⇒ 状态必须报 True）。

全部走子进程（reload / meta_path 操作是进程级全局面，不在收集进程内做）。
"""
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_HOST_MAIN = """
import importlib, sys
sys.path.insert(0, r"%(repo)s")
import lingshu.core.component_resolver as CR

def n_finders():
    return sum(1 for f in sys.meta_path if f.__class__.__name__ == "_ComponentFinder")

CR.install()
print("after_install_1", CR.is_installed(), n_finders())
importlib.reload(CR)
print("after_reload", CR.is_installed(), n_finders())
try:
    import vision
    print("import_vision ok")
except BaseException as e:
    print("import_vision", type(e).__name__)
CR.install()
print("after_install_2", CR.is_installed(), n_finders())
for _ in range(5):
    importlib.reload(CR)
    CR.install()
print("after_5_cycles", CR.is_installed(), n_finders())
print("guard_installed_report", CR.discovery_report()["guard_installed"])
"""


def _run_host():
    main = os.path.join(REPO, "tests", "_probe_1097_host.py")
    with open(main, "w", encoding="utf-8") as f:
        f.write(_HOST_MAIN % {"repo": REPO})
    try:
        env = dict(os.environ)
        env.update({"PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"})
        return subprocess.run([sys.executable, "-B", "-W", "ignore", main],
                              cwd=os.path.dirname(main), env=env,
                              capture_output=True, text=True,
                              encoding="utf-8", timeout=180)
    finally:
        try:
            os.remove(main)
        except OSError:
            pass


def test_reload_state_consistency_and_no_finder_accumulation():
    r = _run_host()
    assert r.returncode == 0, "宿主进程异常退出：%s" % r.stderr[-400:]
    out = dict()
    for line in r.stdout.splitlines():
        parts = line.split()
        if parts and parts[0].startswith(("after_", "guard_installed_report")):
            out[parts[0]] = parts[1:]
    # 首次安装：1 个 finder，状态 True
    assert out["after_install_1"] == ["True", "1"], r.stdout
    # reload 后：旧实例仍在拦截 ⇒ is_installed 必须仍为 True（此前假报 False）
    assert out["after_reload"] == ["True", "1"], (
        "reload 后状态查询与实际行为矛盾（#1097 未收口）\n%s" % r.stdout)
    # 拦截仍在：受控裸名仍被守卫接管（此宿主无 vision ⇒ ModuleNotFoundError）
    assert "import_vision ModuleNotFoundError" in r.stdout, r.stdout
    # reload+install 再不净增 finder（此前 1→2→3 无界堆积）
    assert out["after_install_2"] == ["True", "1"], r.stdout
    assert out["after_5_cycles"] == ["True", "1"], r.stdout
    # 诊断面与实际一致（同源失真收口）
    assert out["guard_installed_report"] == ["True"], r.stdout
