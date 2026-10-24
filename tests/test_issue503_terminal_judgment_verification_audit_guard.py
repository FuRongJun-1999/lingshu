# -*- coding: utf-8 -*-
"""#503 守卫：`_terminal_judgment` 的 `verification` 形参不得再是零引用，
且「判据不读一级验证」这条边界必须被如实声明、不被后续误接。

缺陷（lingshu issue #503 · 实读复现见 readings.txt）：
  `_terminal_judgment(claim, verification, impact)` 的 `verification` 在 86 行函数体内
  **零引用**（只出现在签名行）⇒ 签名承诺了一个不存在的接线，终裁产物（verdict）里
  完全没有一级验证的读数；而 `_archive_reflection` 记录的是整个 `verdict`
  ⇒ 行为日志的终裁块同样丢失一级验证读数。

修法（最小 · 只补观测面，不补判别力）：
  把一级验证的 `deviation` / `detail` **如实记入** verdict
  （`verification_deviation` / `verification_detail`）；**判据逐字不变**。

断言组（修前应红 / 修后应绿）：
  A 形参不再零引用：verdict 携带的一级验证读数必须与 report["verification"] 一致，
    且两个偏差取值不同的场景各自如实回显（修前 verdict 无这些键 ⇒ 红）。
  B 边界被如实声明：docstring 必须写明「verification 只做审计、不参与判据」（修前 ⇒ 红）。
  C 反向边界（**修前修后均应绿**）：同一 claim 下 deviation 变化时，
    `reversibility` / `needs_designer` / `action` 种别三者必须逐字不变
    —— 本项守的是「不许把判据接上去」，不随本件修复由红转绿。
  D 直接调用面：绕过 recursive_reflect，直接给 _terminal_judgment 两个只有 deviation
    不同的 verification，返回值除审计字段外必须完全一致。

依赖说明：本守卫依赖 `SpacetimeMemoryEngine` 与 `_terminal_judgment` 的**公开调用面**
（不绑内部实现细节）；若类名或方法名更名，本守卫会红 —— 这是有意的（契约变更须显式处理）。

运行（仓根）：python -X utf8 -m pytest tests/test_issue503_terminal_judgment_verification_audit_guard.py -q --no-header
"""
from __future__ import annotations

import os
import re
import sys
import tempfile

# 仓根定位：<repo>/tests/ 下差一层 ⇒ parents[1] 才是 repo。
# 不写死沙箱路径（可由 LINGSHU_REPO 覆盖），否则换 clone 即假绿。
REPO = os.environ.get("LINGSHU_REPO") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.core.core import SpacetimeMemoryEngine  # noqa: E402

CLAIM = "删除用户数据"
AUDIT_KEYS = ("verification_deviation", "verification_detail")
JUDGMENT_KEYS = ("reversibility", "needs_designer", "action")


def _db(name):
    return os.path.join(tempfile.mkdtemp(prefix="lingshu503_"), name + ".db")


def _kind(action):
    """action 里嵌了每次运行唯一的 todo_id ⇒ 按种别归一后再比。"""
    return re.sub(r"#\S+", "#<id>", str(action))


def _two_scenarios():
    """场景A 无对照记忆（deviation=True）/ 场景B 有对照记忆（deviation=False）。"""
    eA = SpacetimeMemoryEngine(_db("no_hit"))
    rA = eA.recursive_reflect(CLAIM)
    eB = SpacetimeMemoryEngine(_db("hit"))
    eB.add_perception(CLAIM + "，已定位为日志轮转配置错误", importance=0.6)
    rB = eB.recursive_reflect(CLAIM)
    return rA, rB


def test_a_param_is_no_longer_dead():
    """A verdict 必须如实回显一级验证读数（修前 verdict 无这些键）。"""
    rA, rB = _two_scenarios()
    assert rA["verification"]["deviation"] is True, rA["verification"]
    assert rB["verification"]["deviation"] is False, rB["verification"]
    for r in (rA, rB):
        v, ver = r["verdict"], r["verification"]
        missing = [k for k in AUDIT_KEYS if k not in v]
        assert not missing, "终裁产物未回显一级验证读数（verification 仍零引用）：缺 %r，verdict=%r" % (missing, v)
        assert v["verification_deviation"] == ver["deviation"], (
            "回显的偏差与一级验证不符：verdict=%r verification=%r"
            % (v["verification_deviation"], ver["deviation"]))
        assert v["verification_detail"] == ver["detail"], (
            "回显的 detail 与一级验证不符：%r vs %r"
            % (v["verification_detail"], ver["detail"]))


def test_b_boundary_declared_in_docstring():
    """B docstring 必须声明「verification 只做审计、不参与判据」。"""
    import inspect
    doc = inspect.getdoc(SpacetimeMemoryEngine._terminal_judgment) or ""
    assert "#503" in doc, "docstring 未标注 #503 边界"
    assert "零引用" in doc, "docstring 未说明 verification 曾零引用"
    assert "不" in doc and "判据" in doc, "docstring 未声明「不接判据」的边界"


def test_c_judgment_is_invariant_under_deviation():
    """C 反向边界：偏差变化不得改变任何判据字段（修前修后均应绿）。

    ★必须包含 needs_designer=False 的场景：否则「(needs_designer or deviation)」这类
    误接会被 True or ... 掩蔽（本件实测：只用不可逆场景时 M3 变异**不变红**，守卫没咬住）。
    故此处同时覆盖 可逆场景（needs_designer=False）与 不可逆场景（True）。
    """
    cases = [
        ("可逆", "查询日志记录"),          # needs_designer 应为 False
        ("不可逆", CLAIM),                 # needs_designer 应为 True
    ]
    seen = {}
    for tag, claim in cases:
        eA = SpacetimeMemoryEngine(_db("inv_no_" + tag))
        rA = eA.recursive_reflect(claim)
        eB = SpacetimeMemoryEngine(_db("inv_hit_" + tag))
        eB.add_perception(claim + "，已定位为日志轮转配置错误", importance=0.6)
        rB = eB.recursive_reflect(claim)
        jA, jB = rA["verdict"], rB["verdict"]
        seen[tag] = jA["needs_designer"]
        assert rA["verification"]["deviation"] != rB["verification"]["deviation"], (
            "%s：两场景偏差应不同，否则本项无判别力" % tag)
        assert jA["reversibility"] == jB["reversibility"], (tag, jA, jB)
        assert jA["needs_designer"] == jB["needs_designer"], (
            "%s：判据被 verification 影响（%r vs %r）" % (tag, jA["needs_designer"], jB["needs_designer"]))
        assert _kind(jA["action"]) == _kind(jB["action"]), (tag, jA["action"], jB["action"])
        assert rA["status"] == rB["status"], (tag, rA["status"], rB["status"])
    # 判别力自检：两组场景的 needs_designer 必须一真一假，
    # 否则「误接判据」仍可被掩蔽（这条正是本件实测踩到的坑）。
    assert seen["可逆"] is False, "可逆场景应为 needs_designer=False，实得 %r（本项无判别力）" % (seen["可逆"],)
    assert seen["不可逆"] is True, "不可逆场景应为 needs_designer=True，实得 %r" % (seen["不可逆"],)


def test_d_direct_call_ignores_verification_for_judgment():
    """D 直接调用面：只有 deviation 不同的 verification ⇒ 返回值除审计字段外一致。

    同样必须覆盖 needs_designer=False 的可逆场景（理由见 C 组）。
    """
    e = SpacetimeMemoryEngine(_db("direct"))
    for claim in ("查询日志记录", CLAIM):
        impact = e._impact_assessment(claim)
        v_true = e._terminal_judgment(claim, {"deviation": True, "detail": "x"}, impact)
        v_false = e._terminal_judgment(claim, {"deviation": False, "detail": "y"}, impact)
        for k in JUDGMENT_KEYS:
            a, b = v_true.get(k), v_false.get(k)
            assert (a == b) if k != "action" else (_kind(a) == _kind(b)), (
                "claim=%r：判据被 verification 影响（%s：%r vs %r）" % (claim, k, a, b))
        assert v_true["verification_deviation"] is True, v_true
        assert v_false["verification_deviation"] is False, v_false
        assert v_true["verification_detail"] == "x", v_true
        assert v_false["verification_detail"] == "y", v_false