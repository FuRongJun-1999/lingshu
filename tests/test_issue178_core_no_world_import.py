"""#178 中立守卫：core 层不得 import world（含函数内惰性 import）。

缺陷（锚点 926b7b7e 实测）：
    lingshu/core/core.py 里存在 28 处 `from ..world...`，**全部是函数内惰性 import**
    （模块级 0 处、TYPE_CHECKING 块内 0 处），跨 11 个 world 子模块。

为什么判据是「出现总数 == 0」而不是「模块级 import 数 == 0」：
    28 处全在函数体内，所以「模块级」守卫一建起来就是绿的 —— 点不红，是摆设。
    issue #178 的诉求是「core 不应硬依赖 world 实现」，函数内 import 同样算硬依赖。

为什么用 AST 而不是 grep：
    grep 会把注释、docstring、错误提示串里的 `from lingshu.world import ...` 也算进去
    （core.py:10625/10626/10672/10759 就是这种提示串），产生假阳性。
    AST 只认真实 ImportFrom 节点。

本卡不改 core：本守卫当前就是红的，这是成功态（VERDICT: GUARD_RED）。
    后续 178-B/C/D 逐门面改走注册 seam 后，本守卫转绿。
"""

from __future__ import annotations

import ast
from pathlib import Path

# core 层禁止的 import 前缀
FORBIDDEN_PREFIXES = ("..world", "lingshu.world")

CORE_DIR = Path(__file__).resolve().parents[1] / "lingshu" / "core"


def _display_module(node):
    """还原成源码写法（`..world.vprim`）。ast 把前导点放在 level 字段，module 不含点。"""
    level = getattr(node, "level", 0) or 0
    mod = node.module or ""
    return "." * level + mod


def _is_forbidden(node):
    """ImportFrom 节点是否指向 world 层。

    注意：`from ..world.vprim import X` 在 AST 里是 level=2、module='world.vprim'
    （前导点**不**在 module 里）。所以不能直接拿 '..world' 去前缀匹配 —— 会永远落空。
    """
    mod = node.module or ""
    level = getattr(node, "level", 0) or 0

    # 相对形式：`from ..world...`（level>=2 且第一段是 world）
    if level >= 2 and (mod == "world" or mod.startswith("world.")):
        return True

    # 绝对形式：`from lingshu.world...`
    if mod == "lingshu.world" or mod.startswith("lingshu.world."):
        return True

    return False


class _Collector(ast.NodeVisitor):
    """收集 ImportFrom 节点，并记录所在函数/方法名（模块级为 <module>）。"""

    def __init__(self):
        self.hits = []
        self._stack = []

    def _visit_func(self, node):
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()

    visit_FunctionDef = _visit_func
    visit_AsyncFunctionDef = _visit_func

    def visit_ImportFrom(self, node):
        if _is_forbidden(node):
            owner = ".".join(self._stack) if self._stack else "<module>"
            self.hits.append((node.lineno, _display_module(node), owner))
        self.generic_visit(node)


def collect_world_imports(tree):
    """返回 [(行号, module, 所在函数名)]，按行号升序。"""
    collector = _Collector()
    collector.visit(tree)
    return sorted(collector.hits)


def scan_source(text):
    """扫描一段源码文本（供判别力自检使用）。"""
    return collect_world_imports(ast.parse(text))


def scan_core_dir():
    """扫描 lingshu/core/*.py，返回 [(相对路径, 行号, module, 所在函数名)]。"""
    assert CORE_DIR.is_dir(), "core 目录不存在：%s" % CORE_DIR
    found = []
    for path in sorted(CORE_DIR.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(CORE_DIR.parent.parent).as_posix()
        for lineno, module, owner in scan_source(path.read_text(encoding="utf-8")):
            found.append((rel, lineno, module, owner))
    return found


# ── 测试 ────────────────────────────────────────────────────────────


def test_issue178_guard_detects_sample():
    """判别力自检：守卫必须能检出函数内的 `from ..world import x`。

    这条挂了就说明守卫本身是摆设（AST 遍历漏掉函数体等），
    那么下面「当前树必须红」那条就毫无意义。
    """
    sample = (
        "import os\n"
        "def outer():\n"
        "    def inner():\n"
        "        from ..world.vprim import VPrim\n"
        "        return VPrim\n"
        "    return inner\n"
    )
    hits = scan_source(sample)
    assert len(hits) == 1, "守卫未检出函数内惰性 import，实得 %r" % (hits,)
    lineno, module, owner = hits[0]
    assert module == "..world.vprim", module
    assert owner == "outer.inner", owner

    # 绝对形式同样要检出
    hits2 = scan_source("def f():\n    from lingshu.world import register_world_process\n")
    assert len(hits2) == 1, "守卫未检出绝对形式 import，实得 %r" % (hits2,)

    # 反向对照：非 world 模块不得误报
    hits3 = scan_source("def f():\n    from ..core import thing\n    from ..gen import other\n")
    assert hits3 == [], "守卫误报非 world 模块，实得 %r" % (hits3,)


def test_issue178_core_has_no_world_import():
    """core 层对 world 的 import 总数必须为 0（模块级与函数内都算）。"""
    hits = scan_core_dir()
    if hits:
        detail = "\n".join(
            "  %s:%d:%s  ->  from %s import ..." % (rel, lineno, owner, module)
            for rel, lineno, module, owner in hits
        )
        raise AssertionError(
            "#178 lingshu/core/ 仍有 %d 处对 world 的 import（应为 0；模块级与函数内都计入）：\n%s\n"
            "\n修法：门面改为经 register_world_process() 注册后调用；"
            "world 侧未注册时保持惰性 import 兜底并计数（裁定 E：fail-open，不改公开行为）。"
            % (len(hits), detail)
        )
