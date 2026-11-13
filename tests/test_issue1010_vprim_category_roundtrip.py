"""#1010 守卫：vprim 类别 编码→解析 往返逐字相等。

缺陷（修复前实测，锚定 e83529c）：三族输入 anchor_text()→parse_anchor() 往返破坏——
  ① stop-set 字符（| ; ； @ ( )）裸写不引：`a|b` 读回 `b`（diff 给读侧 stop-set
     增补 `；;|` 后的回归——基线 1682d04 stop-set 为 '@()'，同输入读回正确）
  ② 非 ASCII 空白（\u3000/\n/\r）不加引号：模块约定「含空白加引号」，
     旧条件只查 ASCII 空格/tab，isspace() 漏网
  ③ 旧式裸类别以引号结尾（ab"@(1,2,3,4)）：读侧 text[i-1]=='"' 强走引号形态、
     回扫无起始引号返回空 ⇒ parse_anchor 整条 None（合法旧存档静默消失）
本文件是「回退即红」对照。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingshu.world.vprim import VPrim, parse_anchor  # noqa: E402


def _roundtrip(cat):
    v = VPrim(category=cat, bbox=(1, 2, 3, 4), confidence=0.9, source="t")
    p = parse_anchor(v.anchor_text())
    assert p is not None, cat
    assert p.category == cat, (cat, p.category)
    assert p.bbox == (1, 2, 3, 4)


@pytest.mark.parametrize("cat", ["a|b", "a;b", "a；b", "a@b", "p(1)",
                                 "充电\u3000桩", "a\nb", "a\rb",
                                 "multi word", 'q"uote', "back\slash"])
def test_g1_stopset_and_whitespace_categories_roundtrip(cat):
    _roundtrip(cat)


def test_g2_plain_categories_write_verbatim():
    """向后兼容承诺：不含保留字符的类别输出逐字不变。"""
    assert VPrim(category="tree", bbox=(1, 2, 3, 4),
                 confidence=0.9, source="t").anchor_text() == "tree@(1,2,3,4)"


def test_g3_legacy_bare_quote_ending_archive():
    """旧式裸存档 ab\"@(1,2,3,4) 修复前 parse_anchor 返回 None。"""
    p = parse_anchor('ab"@(1,2,3,4)')
    assert p is not None and p.category == 'ab"' and p.bbox == (1, 2, 3, 4)


def test_g4_multi_anchor_pipe_line_still_parses():
    """防过修正向：stop-set 保留 | 的动机（多锚点行）不受影响。"""
    p = parse_anchor("rock@(5,6,7,8)")
    assert p is not None and p.category == "rock"
