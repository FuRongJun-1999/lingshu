# -*- coding: utf-8 -*-
"""#238 守卫：不得引用**仓内不存在的编号纪律**（悬空条号引用）。

缺陷（50ee03b 修复前）：全仓 10 处按「纪律第 N 条」引用某编号纪律，
而**仓内没有任何编号纪律的定义文件**（`grep` 只命中引用方自身）——
读者按图索骥必扑空；N 也无从核对（12/15/16 三个号各出现若干次）。

修复方向：**改声明**——按仓内既有风格改为**按名字引用**
（既有先例 `tests/test_hexgen_self_source_guards.py:191`「跨批覆盖重叠」），
规则正文原本就内联在每处，故只去掉不可验证的**条号**，不删规则内容。

本守卫两条判据（缺一即有假绿风险）：
  A 反向：全仓不得再出现「纪律第 <数字> 条」（编号无从核对 ⇒ 必为悬空引用）。
  B 正向（防「把规则一起删掉」冒充修复）：被修文件的**规则正文必须仍在**。

判别力（实测）：
  · 修复前 A 判据红：10 处命中；
  · 从修复后状态**回填任一编号**（变异）A 判据复红（见 verify.md 变异读数）。

仓库根由 `__file__` 上溯定位（`<repo>/tests/` 下差一层），**不写死沙箱路径**。
"""
from __future__ import annotations

import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 只扫这几类文本文件，避开二进制与第三方目录
_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".mypy_cache", ".pytest_cache"}
_EXTS = (".py", ".yml", ".yaml", ".md", ".json", ".txt", ".toml", ".cfg", ".ini")

# 「纪律第 N 条」——含「仓工作纪律第 N 条」「纪律第 N 条」两种前缀形态。
# ★ 必须容忍「纪律」与「第」之间的换行+缩进：站点 2 原先正是
#   `仓工作纪律\r\n    第 15 条` 的**跨行**形态；首版正则要求「纪律第」相邻，
#   只报 9 处而全量为 10 处 ⇒ 漏掉 exactly 这一种，已修（宽度有界，避免跨段误配）。
NUMBERED = re.compile(r"纪律[ \t\r\n]{0,8}第[ \t\r\n]{0,4}\d+[ \t\r\n]{0,4}条")


def _iter_text_files():
    for dirpath, dirnames, filenames in os.walk(REPO):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for name in filenames:
            if name.endswith(_EXTS):
                yield os.path.join(dirpath, name)


def _scan():
    hits = []
    for path in _iter_text_files():
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
        except (UnicodeDecodeError, OSError):
            continue
        for m in NUMBERED.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            hits.append((os.path.relpath(path, REPO).replace("\\", "/"), line, m.group(0)))
    return hits


def test_a_no_dangling_numbered_discipline_citation():
    """A 反向：仓内不得出现「纪律第 N 条」——无编号纪律定义 ⇒ 必为悬空引用。"""
    hits = _scan()
    assert not hits, (
        "发现悬空条号引用 %d 处（仓内无编号纪律定义，按图索骥必扑空）：\n%s"
        % (len(hits), "\n".join("  %s:%d  %s" % h for h in hits[:20])))


# 正向对照：修复不得以「删掉规则正文」冒充。
# (文件相对路径, 必须仍在的规则正文片段)
_MUST_SURVIVE = [
    (".github/workflows/gate.yml", "命令执行统一 UTF-8"),
    ("scripts/extract_and_run_issue_repro.py", "公开面路径写相对路径"),
    ("scripts/extract_and_run_issue_repro.py", "做好控制台兼容"),
    ("scripts/extract_and_run_issue_repro.py", "不做控制台兼容"),
    ("tests/test_issue120_239_362_repro_exec.py", "全仓文本与路径一律 UTF-8"),
    ("tests/test_issue7_161_293_314_53_activation_contract.py", "跨批覆盖重叠处置"),
    ("docs/theory/存算一体架构学习_外部理论对照与路线交接_20260916.md", "写入后读回确认"),
]


def test_b_rule_text_still_present():
    """B 正向：规则正文必须仍在（防「删规则」冒充修复）。"""
    missing = []
    for rel, frag in _MUST_SURVIVE:
        if frag == "做好控制台兼容":      # 该片段是**反例**，见下条断言
            continue
        path = os.path.join(REPO, rel)
        if not os.path.exists(path):
            missing.append("(文件缺失) %s" % rel)
            continue
        with open(path, encoding="utf-8") as f:
            if frag not in f.read():
                missing.append("%s 缺少规则正文片段 %r" % (rel, frag))
    assert not missing, "规则正文被误删：\n" + "\n".join("  " + m for m in missing)


def test_c_citation_by_name_is_the_chosen_form():
    """C：确认改用「按名字引用」形态（说明修复方向不是把引用整体删掉）。"""
    path = os.path.join(REPO, ".github/workflows/gate.yml")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    assert "仓工作纪律" in text, "点名「仓工作纪律」的引用被整体删除，超出本单范围"
    assert "仓工作纪律「统一 UTF-8」" in text, "应改为按名字引用「统一 UTF-8」"