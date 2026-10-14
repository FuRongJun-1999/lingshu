# -*- coding: utf-8 -*-
"""test_issue510_meta_reflection_param · _meta_reflection 删 claim 死参（#510）
==================================================================================
缺陷（#510，改前实测）：``_meta_reflection(self, claim: str)`` 函数体**零引用**
claim——返回的 standards 一条取自 ``self.self_model.state_description``、一条是
写死常量；**任意两条不同 claim 的输出逐位相同**（改前实测 True）。而它被写进
三级反思流水线第 0 阶段、与 ``report['claim']`` 同报输出——读者会以为元反思
针对该 claim。签名声明了、调用点传了、docstring 又提了相关语义，读者无从判断
该参数是否参与（「看不出它不用」是最坏形态）。

修法（issue 的「接线或删签名」二选一，取删签名——元反思是结构性标准，
与具体 claim 无关，接线反而改变语义）：签名删 claim、docstring 写明
「本阶段不读 claim」、调用点 ``self._meta_reflection()``。

判据（本件）：
  G1 AST：``_meta_reflection`` 除 self 外**零形参**；生产调用点零实参——
     谁把死参加回来（形参出现而 body 零引用）本组断言即红；
  G2 AST（通用口径）：该函数每个形参都在函数体被引用 ≥1 次
     （与 PR #511 的 tests/test_param_wiring.py 同族判据，钉「声明必参与」）；
  G3 行为：``_meta_reflection()`` 可直接调用且两次调用输出一致（确定性）；
     三级反思入口不因签名收窄破坏（reflective_pipeline 冒烟）。

运行（仓根）：python -X utf8 -m pytest tests/test_issue510_meta_reflection_param.py -q
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from lingshu.core.core import SpacetimeMemoryEngine

REPO = Path(__file__).resolve().parents[1]
CORE = REPO / "lingshu" / "core" / "core.py"


def _func_node():
    tree = ast.parse(CORE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_meta_reflection":
            return node
    pytest.fail("core.py 里找不到 _meta_reflection（被改名/删除？）")


def _body_refs(node, name):
    """name 在函数体（不含 args 声明）里被引用的次数。"""
    refs = 0
    for stmt in node.body:
        for sub in ast.walk(stmt):
            if isinstance(sub, ast.Name) and sub.id == name:
                refs += 1
            elif isinstance(sub, ast.arg) and sub.arg == name:
                refs += 1
    return refs


def test_signature_has_no_dead_param():
    """G1：除 self 外零形参（改前：claim 声明且零引用）。"""
    node = _func_node()
    params = [a.arg for a in node.args.args]
    assert params == ["self"], (
        "#510 _meta_reflection 形参应为 [self]（实测 %r）——元反思与具体 claim "
        "无关，接线请改语义测试，勿重新引入零引用形参" % params)


def test_every_param_is_referenced_in_body():
    """G2：每个形参都在函数体被引用 ≥1 次（声明必参与）。"""
    node = _func_node()
    for a in node.args.args:
        if a.arg == "self":
            continue
        refs = _body_refs(node, a.arg)
        assert refs >= 1, (
            "#510 形参 %r 在函数体零引用——死参回潮（声明必参与，否则删签名）"
            % a.arg)


def test_meta_reflection_deterministic_and_pipeline_smoke():
    """G3：两次调用输出一致；三级反思入口不因签名收窄破坏（冒烟）。"""
    e = SpacetimeMemoryEngine(":memory:")
    try:
        r1 = e._meta_reflection()
        r2 = e._meta_reflection()
        assert r1 == r2, "同一引擎两次元反思输出应一致（确定性）"
        assert isinstance(r1, dict) and "standards" in r1, r1
        # 三级反思入口冒烟（调用点已改为零实参——若签名/调用不一致此处即 TypeError）
        rep = e.recursive_reflect("冒烟主张：观察某物", expected="预期", actual="实际")
        assert "meta_reflection" in rep and "claim" in rep, (
            "三级反思报告应同时含 claim 与 meta_reflection（#510 只删死参，"
            "不改报告结构）：%r" % sorted(rep))
    finally:
        e.close()


def test_mutation_dead_param_would_fail_g2():
    """定点变异自证：把 claim 加回签名（body 不读）⇒ G2 必红。

    不做源码手术——用 AST 直证：改前形态（claim 声明 + body 零引用）正是
    G2 所钉的形状，本断言重放该形状确认判据不空转。
    """
    synthetic = ast.parse("def _meta_reflection(self, claim):\n    return 1\n")
    node = synthetic.body[0]
    a = node.args.args[1]
    assert _body_refs(node, a.arg) == 0, (
        "合成死参应零引用——若不是，G2 判据空转")


if __name__ == "__main__":
    import subprocess
    import sys
    raise SystemExit(subprocess.call(
        [sys.executable, "-X", "utf8", "-m", "pytest",
         __file__, "-q", "--no-header"]))
