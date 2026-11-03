# -*- coding: utf-8 -*-
"""test_issue200_skill_failure_confidence_guard · #200 ③ 技能失败降信守卫
============================================================================
钉的语义（lingshu core.py P1-3 技能获取闭环，#200 ③）：
  技能置信度是 **Beta(1,1) 先验下的平滑成功率后验均值** `(ok+1)/(ok+fail+2)`——
  「50 败 10 胜」的技能必须 < 0.5（issue 判据），失败必须真正降信。

#2291（本件修前是**恒绿空壳**）：整份文件只有「subprocess 自跑自身 rc==0」的
pytest 包装、自身无主流程体（无 `__main__`、无检查代码）⇒ 子进程定义完函数即
退出（rc 恒 0、零输出）⇒ assert 恒真——变异 M9 把 #200 增信语义反转成降信后
仍 1 passed。现写入**真实检查体**，三条腿：

  R1 单元语义：`SpacetimeMemoryEngine._smoothed_confidence`——
     (0,0) ⇒ 0.5（无历史建卡默认，与旧断言兼容）；50 败 10 胜 ⇒
     11/62 ≈ 0.177 < 0.5；10 胜 0 败 ⇒ 11/12 ≈ 0.917 > 0.5（正对照）。
  R2 端到端：`:memory:` 库上 50 次 `record_action_sequence(success=False)` +
     10 次 `success=True`（同 hint）⇒ `get_skill_by_name` 的置信度 < 0.5
     （修前「无条件 +0.05」形态 ⇒ 0.5 起步逐胜 +0.05 必 ≥ 0.5，必红）。
  R3 降信支：已存在技能再失败一次 ⇒ 置信度**下降**（旧实现失败从不降信——
     分诊表行 200 第 3 条；删掉 `update_skill_confidence(id, -0.05)` 支 ⇒ 红）。

subprocess 包装保留（直跑同判据），但子进程现在跑的是**真检查**（`__main__`
块调用 `_real_checks()`，失败即非 0 退出）——包装不再是空壳的掩体。

运行（仓根）：python -X utf8 tests/test_issue200_skill_failure_confidence_guard.py
             / python -X utf8 -m pytest tests/test_issue200_skill_failure_confidence_guard.py
退出码：0 = 全过；1 = 有断言失败
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def _real_checks() -> None:
    """真实检查体：R1 单元语义 + R2 端到端 + R3 降信支。失败即抛。"""
    from lingshu.core.core import SpacetimeMemoryEngine

    # ---- R1 单元语义 ----
    sc = SpacetimeMemoryEngine._smoothed_confidence
    v00 = sc(0, 0)
    assert v00 == 0.5, (
        "R1a 无历史 (0,0) 应为建卡默认 0.5，得到 %r" % (v00,))
    v_bad = sc(10, 50)
    assert v_bad < 0.5, (
        "R1b 「50 败 10 胜」平滑成功率应 < 0.5（issue 判据），得到 %r" % (v_bad,))
    assert abs(v_bad - 11.0 / 62.0) < 1e-12, (
        "R1c 平滑成功率公式应为 (ok+1)/(ok+fail+2)=11/62，得到 %r" % (v_bad,))
    v_good = sc(10, 0)
    assert v_good > 0.5, (
        "R1d 正对照：10 胜 0 败应 > 0.5，得到 %r" % (v_good,))

    # ---- R2 端到端：50 败 10 胜 ⇒ 技能置信度 < 0.5 ----
    eng = SpacetimeMemoryEngine(":memory:")
    for _ in range(50):
        eng.record_action_sequence(["步骤一"], "失败", success=False,
                                   skill_hint="演示技能")
    for _ in range(10):
        eng.record_action_sequence(["步骤一"], "成功", success=True,
                                   skill_hint="演示技能")
    skill = eng.store.get_skill_by_name("演示技能")
    assert skill is not None, "R2a 50 败 10 胜后应已建卡（精确同名绑定）"
    conf = float(skill["confidence"])
    assert conf < 0.5, (
        "R2b 「50 败 10 胜」技能置信度应 < 0.5（修前无条件增信形态会 ≥ 0.5），"
        "得到 %r" % (conf,))

    # ---- R3 降信支：失败一次 ⇒ 置信度下降 ----
    eng2 = SpacetimeMemoryEngine(":memory:")
    eng2.record_action_sequence(["步骤二"], "成功", success=True,
                                skill_hint="演示技能二")
    sk0 = eng2.store.get_skill_by_name("演示技能二")
    assert sk0 is not None, "R3a 首次成功应建卡"
    c0 = float(sk0["confidence"])
    eng2.record_action_sequence(["步骤二"], "失败", success=False,
                                skill_hint="演示技能二")
    sk1 = eng2.store.get_skill_by_name("演示技能二")
    c1 = float(sk1["confidence"])
    assert c1 < c0, (
        "R3b 失败后置信度必须下降（旧实现失败从不降信）：%.4f -> %.4f"
        % (c0, c1))


def test_skill_failure_confidence_gate():
    """pytest 收集入口（#200 脚本式守卫的门禁接线）：先**进程内**直跑真实
    检查体（失败点直接见栈），再 subprocess 直跑同判据（rc 非 0 即红）。"""
    _real_checks()

    import subprocess as _sp
    main_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "test_issue200_skill_failure_confidence_guard.py")
    env = dict(os.environ)
    env.update({"PYTHONPATH": REPO, "PYTHONUTF8": "1",
                "PYTHONDONTWRITEBYTECODE": "1"})
    r = _sp.run([sys.executable, "-B", main_path], capture_output=True,
                text=True, encoding="utf-8", env=env, cwd=REPO, timeout=600)
    assert r.returncode == 0, (
        "技能失败降信守卫红：\n%s\n%s" % (r.stdout[-2000:], r.stderr[-500:]))


if __name__ == "__main__":
    _real_checks()
    print("PASS #200 技能失败降信守卫：R1 单元语义 / R2 端到端 50败10胜<0.5 / "
          "R3 失败降信支 全过")
    sys.exit(0)
