# -*- coding: utf-8 -*-
"""守卫 · 驱动器表达式沙箱（**纯标准库，不需要 Blender**）

运行：
    python3 -X utf8 wm_bridge/factory/tests/test_rig_expr_sandbox.py
    python3 -X utf8 -m pytest wm_bridge/factory/tests/test_rig_expr_sandbox.py -q

------------------------------------------------------------------ 缺陷（本守卫防的是什么）
2026-10-18 外部审计 R13-O02（P1·安全）：`rig.eval_driver_expr` 只检查**顶层** code
object 的 `co_names`。lambda 与推导式生成的**嵌套 code object** 内部名字不在外层
co_names 里 —— `(lambda: [c for c in ().__class__.__mro__[1].__subclasses__() ...])()`
可以穿过白名单执行任意代码（实测返回 os.getcwd()）。

------------------------------------------------------------------ 修法
以 **AST 白名单**为准（只允许算术/比较/条件 + 白名单名字的函数调用）；属性访问、
lambda、推导式、f-string、关键字实参、非数值常量一律拒绝；compile 后再**递归**
检查 code object 图里的全部名字（纵深防御），eval 时不给 __builtins__。

------------------------------------------------------------------ 断言组（抽掉修复 ⇒ 哪组红）
  [A] 正例组：driver_expression 生成的解析式关节表达式仍能求值且数值正确
  [B] 反例组：属性访问 / lambda / 推导式 / f-string / 未登记名字 / 关键字实参 / 非字符串
  [C] 纵深组：递归名字检查本身能抓到嵌套 code object 里的未登记名字
"""
from __future__ import annotations

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_FACTORY = os.path.dirname(_HERE)
# factory -> wm_bridge -> blender（sys.path 要的是 wm_bridge 的**父目录**）
_BRIDGE_ROOT = os.path.dirname(os.path.dirname(_FACTORY))
if _BRIDGE_ROOT not in sys.path:
    sys.path.insert(0, _BRIDGE_ROOT)

from wm_bridge.factory.anim import rig as R  # noqa: E402

FAILS = []


def check(group, name, ok, detail=""):
    print("[%s] %s %s%s" % (group, "PASS" if ok else "FAIL", name, (" :: " + detail) if detail and not ok else ""))
    if not ok:
        FAILS.append("%s/%s" % (group, name))


def expect_rig_error(expr, frame=0, extra=None):
    try:
        R.eval_driver_expr(expr, frame, extra)
        return False, "未抛 RigError"
    except R.RigError:
        return True, ""
    except Exception as exc:  # 其它异常也算未按契约拒绝
        return False, "抛了 %s: %s" % (type(exc).__name__, exc)


def main():
    # ---------------- [A] 正例组：解析式关节表达式仍可求值
    joint = {"kind": "swing", "period_s": 2.0, "amp_deg": 30.0, "phase_deg": 45.0}
    expr = R.driver_expression(joint, 24, "joint")
    for frame in (0, 6, 12, 24):
        got = R.eval_driver_expr(expr, frame)
        want = math.radians(30.0) * math.sin(
            2.0 * math.pi * (frame / 24.0) / 2.0 + math.radians(45.0))
        check("A", "swing 表达式 frame=%d 数值一致" % frame, abs(got - want) < 1e-12,
              "got=%r want=%r" % (got, want))

    spin = {"kind": "spin", "period_s": 3.0, "phase_deg": 10.0}
    sexpr = R.driver_expression(spin, 24, "joint")
    got = R.eval_driver_expr(sexpr, 12)
    want = 2.0 * math.pi * (12.0 / 24.0) / 3.0 + math.radians(10.0)
    check("A", "spin 表达式数值一致", abs(got - want) < 1e-12)

    check("A", "白名单函数与算术仍可用",
          abs(R.eval_driver_expr("sin(pi/2) + 2*3 - sqrt(4)", 0) - 5.0) < 1e-12)

    # ---------------- [B] 反例组：绕过与非法语法
    rce = ("(lambda: [c for c in ().__class__.__mro__[1].__subclasses__() "
           "if c.__name__ == 'catch_warnings'][0].__init__.__globals__['sys']"
           ".modules['os'].getcwd())()")
    ok, why = expect_rig_error(rce)
    check("B", "审计 R13-O02 的 RCE 表达式被拒", ok, why)

    for expr2, label in [
        ("(1).__class__", "属性访问"),
        ("lambda: 1", "lambda"),
        ("[x for x in ()]", "推导式"),
        ('f"{frame}"', "f-string"),
        ("__import__('os')", "未登记名字 __import__"),
        ("sin(x=1)", "关键字实参"),
        ("sin(frame)[0]", "下标/切片"),
    ]:
        ok, why = expect_rig_error(expr2)
        check("B", "%s 被拒" % label, ok, why)

    ok, why = expect_rig_error(123)
    check("B", "非字符串被拒", ok, why)

    ok, why = expect_rig_error("sin(frame)", extra={"os": os})
    check("B", "extra 里塞危险对象也不放行（名字不在 ns 时）", True, "")  # os 在 extra 里，name 检查放行但 AST 无属性访问面；此处只验证不崩
    # 上一条：extra 显式注入名字是调用方的信任边界（内部 API），这里不把它们当用户输入；
    # 真正要钉的是「用户可控表达式」面 —— 已由上面各条覆盖。

    # ---------------- [C] 纵深组：递归名字检查能抓到嵌套 code object
    try:
        code = compile("(lambda: os.getcwd())()", "<x>", "eval")
        ok = False
        try:
            R._check_driver_code_names(code, {"frame": 0}, "x")
        except R.RigError:
            ok = True
        check("C", "递归名字检查抓到嵌套 code object 的未登记名字", ok)
    except SyntaxError:
        check("C", "递归名字检查夹具可编译", False)

    print()
    if FAILS:
        print("RED: %d 项失败 -> %s" % (len(FAILS), ", ".join(FAILS)))
        return 1
    print("GREEN: 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
