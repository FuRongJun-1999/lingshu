# -*- coding: utf-8 -*-
"""守卫 · #108（lingshu/gen：3 个模块在 import 期把相对路径写进 sys.path）
====================================================================================
缺陷（分诊表 `p2_groups.json` → `lingshu/gen/hexgen_self_source.py` 行 #108，判定成立）：
  `hexgen_self_source.py:31-32` / `hexgen_multi_seed.py:46-47` / `hexgen_align_probe.py:31-32`
  在**模块顶层**执行 `sys.path.insert(0, ".")` 与 `sys.path.insert(0, "experiments")`。
  相对 `sys.path` 条目的解析目标由**判定时的 cwd** 决定 ⇒ 这是「把 cwd 依赖的解析面插回
  `sys.path`」——而 `lingshu/__init__.py` 在包首次 import 时已由 `_pathguard.scrub()` 把
  cwd/空串条目**消除**（issue #156 v2）；`_pathguard._is_relative_entry`（issue #288·D-2）
  也明写「相对条目目标由 cwd 决定 ⇒ 不可信、不属部署方显式声明面」。故这两行等于把刚被
  消除的解析面又插回来：任何以 cwd 为根的**同名裸模块**（含标准库名）可被抢先解析并在
  本进程内执行。

判据来源：
  · `lingshu/_pathguard.py:301-316` `_is_relative_entry`（相对条目非受控面）；
  · `lingshu/_pathguard.py:185-219` `scrub()`（消除 cwd/空串条目，本模块 import 前已执行）；
  · 本仓 `experiments/` 目录**不存在**（`git ls-files experiments` 为空），且三模块的 import
    全是**包内相对 import**（`from . import ...`）⇒ 这两个条目对它们毫无用处。

断言（每条都钉「缺陷不再存在」，不是「代码能跑」）：
  A 在**干净子进程**里 import `lingshu`（触发 scrub）后再 import 三模块之一，比较前后
    `sys.path`：**新增条目中不得有任何相对条目**，且 `.` / `experiments` 不得出现在
    `sys.path` 中。缺陷形态下新增恰为 `['experiments', '.']`（相对）⇒ 首条即红。
  B 同一子进程里三模块的**生产符号可正常取得**（`legal_win` / `summarize` / `NAME_MAP_SELF2REAL`）
    ——构造性证明「删掉这两个 insert 不影响包内 import」。
  C 三模块**源码**里不得再出现 `sys.path.insert`（形态锚：按片段文本定位，抽掉修复即命中）。
    与 A 互补：A 判运行期效果，C 判源码形态，二者都不可被对方代偿。

运行（lingshu 仓根）：python -X utf8 -m pytest tests/test_issue108_import_path_guard.py -q --no-header
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#   模块 → 该模块里必须仍可取到的生产符号（证明 import 未因删掉 insert 而失效）
TARGETS = {
    "lingshu.gen.hexgen_self_source": "legal_win",
    "lingshu.gen.hexgen_multi_seed": "summarize",
    "lingshu.gen.hexgen_align_probe": "NAME_MAP_SELF2REAL",
}

_SNIPPET = r'''
import importlib, json, os, sys
import lingshu                      # 触发 _pathguard.scrub()：消除 cwd/空串条目
before = list(sys.path)
mod = importlib.import_module(MODNAME)
after = list(sys.path)
added = [e for e in after if e not in before]
rel = [e for e in added if not os.path.isabs(e)]
print(json.dumps({
    "added": added,
    "rel": rel,
    "has_symbol": hasattr(mod, SYMBOL),
    "cwd_rel_entries": [e for e in after if e in (".", "./", ".\\", "experiments", "")],
}))
'''


def _run(modname, symbol):
    code = _SNIPPET.replace("MODNAME", repr(modname)).replace("SYMBOL", repr(symbol))
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO          # 绝对路径，与被测的相对条目无关
    p = subprocess.run([sys.executable, "-X", "utf8", "-c", code],
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       cwd=REPO, env=env)  # #455
    assert p.returncode == 0, f"子进程 import 失败（{modname}）：\n{p.stderr[-2000:]}"
    return json.loads(p.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("modname,symbol", sorted(TARGETS.items()))
def test_a_import_adds_no_relative_sys_path_entry(modname, symbol):
    """A：import 三模块之一后，`sys.path` 不得新增**相对条目**（缺陷形态：新增 `['experiments','.']`）。"""
    got = _run(modname, symbol)
    assert got["rel"] == [], \
        f"{modname} import 后向 sys.path 写入相对条目：{got['rel']}（新增全量 {got['added']}）"
    assert "." not in got["cwd_rel_entries"] and "experiments" not in got["cwd_rel_entries"], \
        f"{modname} import 后 sys.path 含 cwd 依赖条目：{got['cwd_rel_entries']}"


@pytest.mark.parametrize("modname,symbol", sorted(TARGETS.items()))
def test_b_production_symbol_still_importable(modname, symbol):
    """B：删掉 insert 后包内 import 照常可用（构造性证明这两个条目对生产 import 无用）。"""
    got = _run(modname, symbol)
    assert got["has_symbol"] is True, f"{modname} 缺少生产符号 {symbol}"


def _sys_path_inserts(tree):
    """AST 里 `sys.path.insert(...)` 的调用行号（**只认代码**，不误命中注释/字符串里的字样）。"""
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not isinstance(fn, ast.Attribute) or fn.attr != "insert":
            continue
        base = fn.value
        if (isinstance(base, ast.Attribute) and base.attr == "path"
                and isinstance(base.value, ast.Name) and base.value.id == "sys"):
            hits.append(node.lineno)
    return hits


def test_c_source_has_no_sys_path_insert():
    """C：三模块的**代码**里不得再出现 `sys.path.insert`（AST 判据；抽掉修复即命中）。

    用 AST 而非文本匹配：本组修复处留有说明性注释（含 `sys.path.insert` 字样），
    文本匹配会把注释误判为缺陷，反过来逼人删掉判据来源说明。"""
    offenders = []
    for modname in TARGETS:
        rel = os.path.join(*modname.split(".")) + ".py"
        with open(os.path.join(REPO, rel), encoding="utf-8") as f:
            src = f.read()
        for lineno in _sys_path_inserts(ast.parse(src, filename=rel)):
            offenders.append(f"{rel}:{lineno}")
    assert not offenders, "仍有 import 期 sys.path.insert 调用：\n" + "\n".join(offenders)
