"""#1094 守卫：`LINGSHU_COMPONENT_ROOT` 的相对条目不得进入受控解析面。

缺陷（修复前实测，锚定 89a2551）：`_safe_roots()` 只对启动期快照条目调用
`_is_relative_entry()` 过滤，对 `configured_roots()`（环境变量面）直接 `_add`
⇒ `LINGSHU_COMPONENT_ROOT=components` 或 `.` 进入受控面，其解析目标由**判定时
的 cwd** 决定；chdir 后即加载该目录下的组件（实测引擎进程内任意代码执行）。

契约出处：模块 docstring「解释器按当前目录解析这一隐式面，被本模块关闭」；
`safe_search_path()` docstring「相对路径条目一律不计入（其目标由 cwd 决定）」；
`_is_relative_entry()` docstring「一律不计入受控面」。

本文件是「回退即红」对照。
"""
import os
import sys
import textwrap

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESOLVER = "lingshu/core/component_resolver.py"


@pytest.mark.parametrize("root", ["components", ".", "./x", "sub/dir"])
def test_g1_relative_configured_root_excluded(tmp_path, root, monkeypatch):
    """相对根不得出现在受控解析面（修复前 safe_search_path() 首项即相对串）。"""
    # 在子进程里跑：component_resolver 有进程级安装状态
    probe = textwrap.dedent(
        """
        import os, sys
        sys.path.insert(0, sys.argv[1])
        os.environ["LINGSHU_COMPONENT_ROOT"] = sys.argv[2]
        import lingshu.core.component_resolver as CR
        CR.install()
        sp = CR.safe_search_path()
        rel = [p for p in sp if not os.path.isabs(p)]
        print("RELATIVE_PRESENT" if rel else "CLEAN")
        """
    )
    import subprocess
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run([sys.executable, "-X", "utf8", "-c", probe, repo, root],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    assert "RELATIVE_PRESENT" not in r.stdout, (root, r.stdout, r.stderr)
    assert "CLEAN" in r.stdout, (root, r.stdout, r.stderr)


def test_g2_absolute_root_still_works(tmp_path):
    """防过修正向：绝对路径配置根仍正常进入受控面。"""
    comp = tmp_path / "comp"
    comp.mkdir()
    probe = textwrap.dedent(
        """
        import os, sys
        sys.path.insert(0, sys.argv[1])
        os.environ["LINGSHU_COMPONENT_ROOT"] = sys.argv[2]
        import lingshu.core.component_resolver as CR
        CR.install()
        print("FOUND" if sys.argv[2] in CR.safe_search_path() else "MISSING")
        """
    )
    import subprocess
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run([sys.executable, "-X", "utf8", "-c", probe, repo, str(comp)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    assert "FOUND" in r.stdout, (r.stdout, r.stderr)


def test_g3_cwd_poison_not_executed(tmp_path):
    """端到端：相对根 + chdir 后不得加载 cwd 毒组件（修复前 .EXEC 落盘）。"""
    import subprocess
    base = tmp_path
    A, C = base / "A", base / "C"
    for d, owner in ((A, "A-deploy"), (C, "C-attacker")):
        (d / "components").mkdir(parents=True)
        src = "OWNER = %r\n" % owner
        if owner.startswith("C"):
            src += "open(__file__ + '.EXEC', 'w').write('x')\n"
        (d / "components" / "body.py").write_text(src, encoding="utf-8")

    probe = textwrap.dedent(
        """
        import os, sys
        sys.path.insert(0, sys.argv[1])
        os.chdir(sys.argv[2])
        os.environ["LINGSHU_COMPONENT_ROOT"] = "components"
        import lingshu.core.component_resolver as CR
        CR.install()
        os.chdir(sys.argv[3])
        try:
            import body
            print("LOADED", body.OWNER)
        except ModuleNotFoundError:
            print("BLOCKED")
        """
    )
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run([sys.executable, "-X", "utf8", "-c", probe,
                        repo, str(A), str(C)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    assert "BLOCKED" in r.stdout, (r.stdout, r.stderr)
    assert not (C / "components" / "body.py.EXEC").exists()
