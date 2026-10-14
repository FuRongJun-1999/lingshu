# -*- coding: utf-8 -*-
"""test_gate_dependency_closure · 门禁所需依赖必须已在 pyproject 声明
============================================================================
来历（2026-10-09 实测的仓库红灯）：
    `cfd1b22`（修 issue #163）新增了 `tests/test_coggraph_public_paths.py`，
    它会 shell 出去调 `tools/coggraph/export_coggraph.py`；该工具需要 PyYAML，
    写法是：

        try:
            import yaml
        except ImportError:
            print("需要 PyYAML", file=sys.stderr)
            raise            # ★ 重新抛出 ⇒ 这是**硬依赖**，不是软可选

    而 CI 的安装步骤是 `pip install -e ".[full,dev]"`（`full` = numpy+Pillow、
    `dev` = pytest）⇒ **CI 环境里没有 yaml** ⇒ 该测试在 CI 上必红，且因它属
    pytest 腿，**整个 gate 自 2026-10-09 01:56 起对所有人红**。
    本机实测（屏蔽 yaml）：`1 failed, 10 passed`；CI 日志报错点逐字一致
    （`ModuleNotFoundError: No module named 'yaml'`）。

判据：把 `tests/` `tools/` `scripts/` 下的**硬第三方导入**与 `pyproject.toml`
    的 `dependencies` + `[project.optional-dependencies]` 做闭包对比；
    未声明者即红，并点名「谁 import 的、在第几行」。

**"硬/软"的判定规则**（本判据的核心，也是它一开始写错、被自身变异测试抓出来的地方）：
    · 在 `try:` 里但**任一 handler 会 re-raise**（含 `raise`）⇒ **硬**
      —— 缺依赖时该模块**不可用**，门禁必炸；
    · 在 `try:` 里且 handler **吞掉异常**（如 `x = None`）⇒ **软**（可选依赖，不计）；
    · 在 `if TYPE_CHECKING:`（或 `if False:`）里 ⇒ **软**（运行时不执行）；
    · 其余位置的导入（模块顶层、函数内、`__main__` 块内）⇒ **硬**
      —— 门禁会执行这些文件。
    ★ 只按"模块顶层直接子节点"取是**错的**：`import yaml` 在 `try` 里，
      那条规则会漏掉它、判据静默通过——本件自查发现并改正的正是这一处。

量具自检（同族纪律）：① 扫描必须确实读到文件；② 必须扫到 `numpy`
    （已知已声明）与 `yaml`（已知的 try-wrapped 硬依赖）——否则扫描器坏了，
    本判据会**静默通过**。
"""
from __future__ import annotations

import ast
import os
import sys

import pytest

try:
    import tomllib                       # Python ≥3.11
except ModuleNotFoundError:              # pragma: no cover
    tomllib = None

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCAN_DIRS = ("tests", "tools", "scripts")
LOCAL_TOP = {"lingshu", "tools", "scripts", "tests", "experiments"}
DIST_ALIAS = {"pil": "pillow", "yaml": "pyyaml", "pytest": "pytest"}


def _handler_reraises(handler):
    """该 except handler 是否会重新抛出（⇒ 对应 try 里的导入是硬依赖）。"""
    for node in ast.walk(handler):
        if isinstance(node, ast.Raise):
            return True
    return False


def _is_type_checking_guard(node):
    """`if TYPE_CHECKING:` / `if False:` —— 运行时不执行，其内导入为软。"""
    if not isinstance(node, ast.If):
        return False
    src = ast.dump(node.test)
    return ("TYPE_CHECKING" in src) or (isinstance(node.test, ast.Constant) and node.test.value is False)


class _HardImportCollector(ast.NodeVisitor):
    """收集"缺依赖就会炸"的导入；(soft_depth) 记录当前是否处于软上下文。"""

    def __init__(self):
        self.hard = []          # [(module_top_name, lineno)]
        self.soft = []          # 记下被排除的，供诊断
        self._soft_depth = 0

    def _visit_soft(self, node):
        self._soft_depth += 1
        try:
            self.generic_visit(node)
        finally:
            self._soft_depth -= 1

    def visit_Try(self, node):
        # handlers 是否会 re-raise ⇒ 决定 body 里的导入是硬还是软
        reraises = any(_handler_reraises(h) for h in node.handlers) or not node.handlers
        if reraises:
            for st in node.body:
                self.visit(st)
        else:
            for st in node.body:
                self._visit_soft(st)
        for h in node.handlers:
            self.visit(h)
        for st in (node.orelse or []) + (node.finalbody or []):
            self.visit(st)

    def visit_If(self, node):
        if _is_type_checking_guard(node):
            self._visit_soft(node)
            return
        self.generic_visit(node)

    def _record(self, name, lineno):
        top = name.split(".")[0]
        if self._soft_depth:
            self.soft.append((top, lineno))
        else:
            self.hard.append((top, lineno))

    def visit_Import(self, node):
        for a in node.names:
            self._record(a.name, node.lineno)

    def visit_ImportFrom(self, node):
        if node.level == 0 and node.module:
            self._record(node.module, node.lineno)


def _hard_third_party(path):
    """返回该文件里"缺依赖就会炸"的第三方顶层模块名 → [行号]。"""
    with open(path, encoding="utf-8") as fh:
        try:
            tree = ast.parse(fh.read())
        except SyntaxError:
            return {}
    c = _HardImportCollector()
    c.visit(tree)
    out = {}
    for name, lineno in c.hard:
        if (name in sys.stdlib_module_names or name in LOCAL_TOP
                or name.startswith("_")):
            continue
        out.setdefault(name, []).append(lineno)
    return out


def _scan():
    """{第三方名: [相对路径:行, ...]}"""
    found = {}
    for sub in SCAN_DIRS:
        for dp, _dn, fn in os.walk(os.path.join(REPO, sub)):
            for f in fn:
                if not f.endswith(".py"):
                    continue
                p = os.path.join(dp, f)
                rel = os.path.relpath(p, REPO).replace("\\", "/")
                for m, lines in _hard_third_party(p).items():
                    found.setdefault(m, [])
                    found[m].extend("%s:%d" % (rel, ln) for ln in lines)
    return found


def _declared():
    """pyproject 里声明的发行名（小写，去掉版本约束）。"""
    with open(os.path.join(REPO, "pyproject.toml"), "rb") as fh:
        data = tomllib.load(fh)
    proj = data.get("project", {})
    raw = list(proj.get("dependencies") or [])
    for _extra, items in (proj.get("optional-dependencies") or {}).items():
        raw.extend(items or [])
    names = set()
    for spec in raw:
        head = spec.split("[")[0]
        for sep in (">", "=", "<", "!", ";", " "):
            head = head.split(sep)[0]
        names.add(head.strip().lower())
    return names, raw


def test_scanner_is_alive():
    """量具自检：扫描器必须真的读到文件、且必须认出已知的两种形态。"""
    if tomllib is None:                                # pragma: no cover
        pytest.skip("需要 tomllib（Python ≥3.11）解析 pyproject.toml；本仓 requires-python>=3.10")
    found = _scan()
    assert found, "扫描器没扫到任何第三方导入 ⇒ 它坏了（或 SCAN_DIRS 失效），本判据此时无判别力"
    assert "numpy" in found, "没扫到 `numpy`（已知的普通形态导入）⇒ 扫描器坏了"
    assert "yaml" in found, (
        "没扫到 `yaml`——它是 `try: import yaml / except ImportError: … raise` 的**硬依赖**形态。\n"
        "扫不到即说明扫描器退回成了「只看模块顶层直接子节点」的错规则 ⇒ 本判据会**静默通过**。"
    )


def test_every_hard_third_party_import_is_declared():
    """判据：门禁会执行的硬依赖必须已在 pyproject 声明（dependencies 或任一 extra）。"""
    if tomllib is None:                                # pragma: no cover
        pytest.skip("需要 tomllib（Python ≥3.11）解析 pyproject.toml；本仓 requires-python>=3.10")
    found = _scan()
    declared, raw = _declared()
    undeclared = {}
    for mod, places in found.items():
        if DIST_ALIAS.get(mod.lower(), mod.lower()) not in declared:
            undeclared[mod] = places
    assert not undeclared, (
        "以下第三方导入**没有被 pyproject 声明**——CI（`pip install -e \".[full,dev]\"`）不会装它们，"
        "门禁跑到这些位置时必然 `ModuleNotFoundError`：\n"
        + "\n".join("  · %s   ← %s" % (m, ", ".join(sorted(p))) for m, p in sorted(undeclared.items()))
        + "\n\n已声明 = " + repr(sorted(declared))
        + "\n原文 = " + repr(raw)
        + "\n修法：把发行名加进 pyproject 的 `dependencies`（运行时）或对应 extra（如 `dev`）。"
    )
