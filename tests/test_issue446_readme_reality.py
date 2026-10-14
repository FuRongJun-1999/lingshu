# -*- coding: utf-8 -*-
"""test_issue446_readme_reality · tests/README 声明须与仓内现实一致（#446）
================================================================================
缺陷（#446）：`tests/README.md` 声明了**未落仓**的状态——
① 「CI（.github/workflows/ci.yml）按此分两档」：ci.yml 不存在（实测 workflows
  目录只有 gate.yml / pr-lint.yml / repro-bot.yml / upstream-tracker.yml），
  是全仓**唯一被引用而不存在**的 workflow；真源是 gate.yml。
② 改造注记写「两项已知红加 xfail 标注」「新增 … ci.yml」——PR #87 部分采纳时
  **明确未取** xfail（根因由 9bc4985 真修取代），xfail 从未落仓；文档却记成
  已发生的改动（AST 核对：两个用例 decorator_list 长度均为 0）。

判据（本件）：
  G1 README 引用的每个 `.github/workflows/<name>` 文件都真实存在
     （引用而不存在 = 文档撒谎，改前命中 ci.yml）；
  G2 文档↔代码一致：README 称已知红「已转绿」⇒ 全仓 tests/*.py 的
     xfail 标注数 == 0（两侧互相印证，谁漂移谁红）；
  G3 定点变异：合成「引用不存在的 workflow」文本 ⇒ 扫描器必命中
     （证明 G1 的 0 是「核过且一致」而非「没核」）。

运行（仓根）：python -X utf8 -m pytest tests/test_issue446_readme_reality.py -q
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
README = REPO / "tests" / "README.md"
WF = REPO / ".github" / "workflows"

RE_WF_REF = re.compile(r"\.github/workflows/([\w\-]+\.yml)")


def _dangling_wf_refs(text: str, wf_dir: Path):
    """返回文本里引用了但 wf_dir 下不存在的 workflow 名。"""
    refs = set(RE_WF_REF.findall(text))
    return sorted(r for r in refs if not (wf_dir / r).is_file())


def test_readme_workflow_refs_all_exist():
    """G1：README 引用的 workflow 全部真实存在（改前：ci.yml 悬空）。"""
    dangling = _dangling_wf_refs(README.read_text(encoding="utf-8"), WF)
    assert dangling == [], (
        "#446 tests/README.md 引用了不存在的 workflow：%r——实际存在：%r。"
        "文档不得声明未落仓的 CI 状态（真源是 gate.yml）。"
        % (dangling, sorted(p.name for p in WF.glob('*.yml'))))


def test_no_xfail_marks_while_readme_claims_green():
    """G2：README 已声明已知红转绿 ⇒ 全仓 tests/ 零 xfail **标记**（文档↔代码一致）。

    判据锚标记形态（``pytest.mark.xfail`` / ``@xfail`` / ``pytest.xfail``）而非
    单词「xfail」——文档/注释里的历史性提及不是标记。
    """
    text = README.read_text(encoding="utf-8")
    assert "已转绿" in text, "README 应保留「已知红已转绿」章节（其存在是本判据的前提）"
    mark = re.compile(r"pytest\.mark\.xfail|@xfail\b|pytest\.xfail")
    hits = []
    for p in sorted((REPO / "tests").glob("*.py")):
        if p.name == Path(__file__).name:
            continue        # 守卫自身必然含该模式（正则/文档提及）——排除自引
        for i, ln in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if mark.search(ln):
                hits.append("tests/%s:%d" % (p.name, i))
    assert hits == [], (
        "#446 README 声明已知红已转绿、且 xfail 从未被采纳——但仓内仍有 xfail 标记：%r。"
        "两态必居其一：要么标记是死标记应删，要么 README 的转绿声明失实。" % hits)


def test_mutation_scanner_catches_dangling_ref():
    """G3（定点变异自证）：引用不存在的 workflow ⇒ 扫描器必命中；真引用不命中。"""
    bad = "CI（.github/workflows/ci.yml）按此分两档。"
    assert _dangling_wf_refs(bad, WF) == ["ci.yml"], "合成悬空引用应被命中——否则 G1 判据空转"
    good = "CI 真源是 .github/workflows/gate.yml；标题门在 pr-lint.yml。"
    assert _dangling_wf_refs(good, WF) == [], "真实存在的 workflow 引用不应命中"


if __name__ == "__main__":
    import subprocess
    import sys
    raise SystemExit(subprocess.call(
        [sys.executable, "-X", "utf8", "-m", "pytest", __file__, "-q", "--no-header"]))
