# -*- coding: utf-8 -*-
"""#187 守卫：synonym_merge L2 的序数守卫必须真正生效（不再是死代码）。

缺陷态（修复前 HEAD）`tools/coggraph/synonym_merge.py`：
  · `is_ordinal_diff`（`:38-51`）在等长且差异不止 1 处时直接 `return len(diffs)==0`；
  · L2（`:124`）复用该函数判字谜串：字谜串字符多重集相同 ⇒ 至少 2 处不同 ⇒
    必然返回 False ⇒ `if not is_ordinal_diff(...)` 恒真，「序数守卫」永不生效——
    仅序数互换的标题（如「第一章第二节」/「第二章第一节」）被并入同一 synonym 组。

判据（issue #187 报告 + 验收三条）：
  1. 序数互换对不得归并，且不计入 `L2_语序变体`；
  2. 真正的语序变体（「知识图谱构建」/「构建知识图谱」）仍按 L2 归并；
  3. L3（编辑距离 1）行为不变。
  来源：issue #187（报告人 EricEricEr，2026-10-08）；属**经验标定，追不到理论出处**。

变异点：把 `is_ordinal_diff` 等长分支改回 `if len(diffs) != 1: return len(diffs)==0`
⇒ 本文件 test_ordinal_swap_* / test_l2_guard_is_live 必红。

运行（lingshu 仓根）：python -X utf8 -m pytest tests/test_issue187_synonym_merge_l2_ordinal.py -q
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT = os.path.join(REPO, "tools", "coggraph", "synonym_merge.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("_synonym_merge", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_MOD = _load_module()


def _run_graph(titles):
    """titles: list[str] → 跑 main()，返回 stdout 里那行 JSON 的 dict。"""
    nodes = [{"id": "n%d" % i, "title": t, "bucket": "concept",
              "layer": "knowledge", "importance": 0.5}
             for i, t in enumerate(titles)]
    d = tempfile.mkdtemp(prefix="syn_guard_")
    gp = os.path.join(d, "g.json")
    with open(gp, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"nodes": nodes}, f, ensure_ascii=False)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = _MOD.main(["--graph", gp, "--out", os.path.join(d, "out"), "--write"])
    assert rc == 0
    for ln in buf.getvalue().splitlines():
        if ln.startswith("{"):
            return json.loads(ln)
    raise AssertionError("没拿到 JSON 读数：\n" + buf.getvalue())


# ── 量具自检 ────────────────────────────────────────────────────────────────
def test_gauge_is_alive():
    assert callable(_MOD.is_ordinal_diff)
    assert _MOD.is_ordinal_diff("第一章", "第二章") is True, "编辑距离 1 的序数判定回退"


# ── 判据 1：序数互换对（issue 复现串）──────────────────────────────────────
def test_ordinal_swap_pair_is_ordinal_diff():
    """issue 复现串：等长字谜、差异全为序数对 ⇒ 必须判为序数差异（缺陷态返回 False）。"""
    assert _MOD.is_ordinal_diff("第一章第二节", "第二章第一节") is True, (
        "#187：字谜序数互换未被识别为序数差异 ⇒ L2 守卫仍是死代码")
    # 非序数语序变体不得误判为序数差异
    assert _MOD.is_ordinal_diff("知识图谱构建", "构建知识图谱") is False


def test_ordinal_swap_pair_not_merged():
    """两对序数互换标题的图：归并组 0，且不计入 L2_语序变体。"""
    out = _run_graph(["第一章第二节", "第二章第一节",
                      "第三章第四节", "第四章第三节"])
    assert out["归并组"] == 0, "#187：序数互换对被归并了（读数 %r）" % (out,)
    assert out["stats"].get("L2_语序变体", 0) == 0, (
        "#187：序数互换对计入了 L2_语序变体（读数 %r）" % (out,))


# ── 判据 2：真实语序变体仍归并 ─────────────────────────────────────────────
def test_real_word_order_variant_still_merged():
    out = _run_graph(["知识图谱构建", "构建知识图谱"])
    assert out["归并组"] == 1 and out["stats"].get("L2_语序变体") == 1, (
        "#187：非序数语序变体被误杀（读数 %r）" % (out,))
    assert out["分组"].get("synonym") == 1, "应产出 1 个 synonym 组：%r" % (out,)


# ── 判据 3：L3 行为不变 ────────────────────────────────────────────────────
def test_l3_behavior_unchanged():
    # 编辑距离 1、序数差异 ⇒ 不进待复核
    assert _MOD.is_ordinal_diff("第一章", "第二章") is True
    assert _MOD.edit_dist_le1("第一章", "第二章") is True
    # 编辑距离 1、非序数差异 ⇒ 仍进待复核
    assert _MOD.is_ordinal_diff("学习方法", "学习方式") is False
    assert _MOD.edit_dist_le1("学习方法", "学习方式") is True
    # 长度差 1 的序数判定不变
    assert _MOD.is_ordinal_diff("第一章", "第一章节") is False
    assert _MOD.is_ordinal_diff("第一章", "第一章二") is True


def test_l3_review_pipeline_unchanged():
    """端到端 L3：非序数编辑距离 1 仍进待复核；序数编辑距离 1 不进。"""
    out = _run_graph(["学习方法", "学习方式", "第一章", "第二章"])
    review = out["L3待复核"]
    assert review == 1, "L3 待复核数应为 1（仅非序数对），实得 %r（读数 %r）" % (review, out)


# ── 结构性：L2 守卫必须是「可达」的，不是恒真 ──────────────────────────────
def test_l2_guard_is_live():
    """L2 守卫须对某输入为真（可达），否则就是死代码。"""
    live = _MOD.is_ordinal_diff("第一章第二节", "第二章第一节")
    dead = _MOD.is_ordinal_diff("知识图谱构建", "构建知识图谱")
    assert live is True and dead is False, (
        "L2 序数守卫不是活判据：live=%r dead=%r（live 必须 True、dead 必须 False）"
        % (live, dead))


if __name__ == "__main__":
    raise SystemExit(__import__("pytest").main([__file__, "-q"]))
