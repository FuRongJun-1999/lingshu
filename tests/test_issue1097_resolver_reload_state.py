"""#1097 守卫：component_resolver 的 install/is_installed 按 sys.meta_path 实际实例判定。

缺陷（修复前实测，锚定 89a2551）：`importlib.reload` 把模块级 `_FINDER` 清成
None，但旧实例仍在 sys.meta_path 里继续拦截 ⇒ `is_installed()` 报 False（与实际
行为矛盾，`import vision` 仍 ModuleNotFoundError）；再 `install()` 新建实例插入
⇒ 幂等性破坏、每次 reload+install 净增一个 finder（5 次后 finders=5）。

契约出处：`install()` docstring「幂等，可重复调用」；`is_installed()` docstring
「守卫是否已安装」。本文件是「回退即红」对照。
"""
import os
import subprocess
import sys
import textwrap

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(script: str) -> str:
    r = subprocess.run([sys.executable, "-X", "utf8", "-c", textwrap.dedent(script), REPO],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
    return r.stdout


def test_g1_is_installed_true_after_reload():
    out = _run("""
        import sys, importlib
        sys.path.insert(0, sys.argv[1])
        import lingshu.core.component_resolver as CR
        CR.install()
        before = CR.is_installed()
        importlib.reload(CR)
        after = CR.is_installed()
        print("before=%s after=%s" % (before, after))
    """)
    assert "before=True after=True" in out, out   # 修复前 after=False


def test_g2_install_idempotent_across_reload():
    out = _run("""
        import sys, importlib
        sys.path.insert(0, sys.argv[1])
        import lingshu.core.component_resolver as CR
        def nf():
            return sum(1 for f in sys.meta_path
                       if f.__class__.__name__ == "_ComponentFinder")
        CR.install()
        n0 = nf()
        for _ in range(5):
            importlib.reload(CR)
            CR.install()
        print("n0=%d n5=%d" % (n0, nf()))
    """)
    assert "n0=1 n5=1" in out, out                 # 修复前 n5=6


def test_g3_guard_still_blocks_after_reload():
    """防过修正向：reload 后守卫仍在拦截（不是「关掉守卫换来幂等」）。"""
    out = _run("""
        import sys, importlib
        sys.path.insert(0, sys.argv[1])
        import lingshu.core.component_resolver as CR
        CR.install()
        importlib.reload(CR)
        try:
            import vision
            print("LOADED")
        except ModuleNotFoundError:
            print("BLOCKED")
    """)
    assert "BLOCKED" in out, out


def test_g4_clean_install_single_finder():
    out = _run("""
        import sys
        sys.path.insert(0, sys.argv[1])
        import lingshu.core.component_resolver as CR
        CR.install(); CR.install(); CR.install()
        print(sum(1 for f in sys.meta_path if f.__class__.__name__ == "_ComponentFinder"))
    """)
    assert out.strip().endswith("1"), out
