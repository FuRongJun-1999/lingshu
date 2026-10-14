# -*- coding: utf-8 -*-
"""test_issue236_fullwidth_semicolon · 分句正则收全角分号「；」（#236）
======================================================================
缺陷（#236，改前实测复现）：`parse_description` 的分句字符类
``re.split(r"[,，、;;。]+", text)`` 里是**两个半角 `;`**——中文句间习惯的
全角分号「；」不在类里：

    parse_description("左上有红圆形；右下有蓝条纹")   → 1 个子句（第二个物体丢失）
    parse_description("左上有红圆形;右下有蓝条纹")     → 2 个子句（半角对照正常）

docstring 自称「子句按逗号/顿号/分号切分」——全角分号本是承诺口径，正则漏收。

修法：字符类补「；」⇒ `[,，、;;；。]+`。

判据（本件）：
  G1 全角分号切分：「左上红圆；右下蓝条纹」⇒ 恰 2 子句，两物体（shape/color/pos）
     逐项可取——第二个物体不得静默丢失；
  G2 半角对照不变；顿号/逗号/句号既有切分不回归；
  G3 定点变异：把「；」从字符类移除（还原旧类）⇒ G1 必红并点名（子句数回到 1）。

运行（仓根）：python -X utf8 -m pytest tests/test_issue236_fullwidth_semicolon.py -q
"""
from __future__ import annotations

import re

from lingshu.nn.hex_text import parse_description

FULLWIDTH = "左上有红圆形；右下有蓝条纹"
HALFWIDTH = "左上有红圆形;右下有蓝条纹"


def test_fullwidth_semicolon_splits_into_two_clauses():
    """G1：全角分号 ⇒ 2 子句、两物体逐项可取（改前：1 子句，第二物体丢失）。"""
    cs = parse_description(FULLWIDTH)
    assert len(cs) == 2, (
        "#236 全角分号未切分：子句数 %d（期望 2）——第二物体被静默丢弃。clauses=%r"
        % (len(cs), cs))
    assert cs[0]["shape"] == "circle" and cs[0]["color"] == "red" and cs[0]["pos"] == "r0", cs[0]
    assert cs[1]["shape"] == "stripe" and cs[1]["color"] == "blue" and cs[1]["pos"] == "r8", cs[1]


def test_halfwidth_and_other_separators_unchanged():
    """G2（防过修）：半角/顿号/逗号/句号既有切分行为不变。"""
    assert len(parse_description(HALFWIDTH)) == 2
    assert len(parse_description("左上有红圆形、右下有蓝条纹")) == 2
    assert len(parse_description("左上有红圆形，右下有蓝条纹")) == 2
    assert len(parse_description("左上有红圆形。右下有蓝条纹")) == 2
    # 无分隔符的单句不受影响
    assert len(parse_description("左上有红圆形")) == 1


def test_mutation_removal_of_fullwidth_turns_red():
    """G3（定点变异自证）：还原旧字符类（去「；」）⇒ 全角句不再切分。

    直接在测试内重建旧/新两种口径的行为对照——证明 G1 的判别力来自「；」
    这一个字符，不是偶然通过。
    """
    text = FULLWIDTH
    old = re.split(r"[,，、;;。]+", text)      # ← 旧字符类（#236 缺陷形态）
    new = re.split(r"[,，、;;；。]+", text)    # ← 修复后字符类
    assert len([s for s in old if s.strip()]) == 1, "旧类应仍复现缺陷（1 子句）——若不复现，G1 判据失效"
    assert len([s for s in new if s.strip()]) == 2, "新类应切出 2 子句"


if __name__ == "__main__":
    import subprocess
    import sys
    raise SystemExit(subprocess.call(
        [sys.executable, "-X", "utf8", "-m", "pytest",
         __file__, "-q", "--no-header"]))
