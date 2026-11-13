"""dsh-bounty #1796 守卫：英文缩约否定（n't）必须被 _has_neg 识别。

缺陷（修复前实测，锚定 ed8700d）：`_EN_NEG = r"\\b(...|n't|...)\\b"` 的
`\\b` 在缩约词 "doesn't" 的 "n" 前永不成立（前一字符 "s" 是词字符）⇒
`n't` 备选不可满足——「the feature doesn't work」判无否定、「does not
work」判有否定，语义等价的两种写法裁决相反（`conflict` 前者 None、
后者 'polarity'）。这正是模块头动机段要防的「否定更正被当重复吞掉」。

判据来源：dsh-bounty #1796（含逐字读数）。本文件是「回退即红」对照：
把 `_EN_NEG` 改回旧式，G1/G2 失败。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingshu.core import contradiction as C  # noqa: E402


# ------------------------------------------------- G1 缩约词逐个被识别

def test_g1_has_neg_recognizes_contractions():
    for w in ["doesn't", "isn't", "wasn't", "aren't", "can't", "couldn't",
              "won't", "didn't", "hasn't", "haven't", "Don't"]:
        assert C._has_neg(w + " work"), w  # 修复前全部 False


def test_g1b_expanded_forms_unchanged():
    """防过修正向：非缩约备选行为逐字不变。"""
    for w in ["not", "no", "never", "none", "cannot", "without"]:
        assert C._has_neg(w + " ok")
    assert not C._has_neg("it works fine")
    assert not C._has_neg("nothing n t")  # 空格隔开的 n t 仍不匹配


# --------------------------------------------- G2 等价写法裁决一致

def test_g2_conflict_contraction_equals_expanded():
    a = "the feature works"
    assert C.conflict(a, "the feature does not work") == "polarity"
    assert C.conflict(a, "the feature doesn't work") == "polarity"  # 修复前 None


def test_g2b_chinese_negation_untouched():
    """中文否定面（_NEG_MARKERS/_NON_NEG）行为不变。"""
    assert C._has_neg("没有完成")
    assert not C._has_neg("非常完成")   # _NON_NEG 豁免仍生效
    assert not C._has_neg("未来完成")
