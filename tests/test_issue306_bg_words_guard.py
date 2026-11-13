# -*- coding: utf-8 -*-
"""#306/#307 守卫：背景词的色字不参与部件颜色；方位/花纹词的字不参与形状。

缺陷面（公开 issue #306/#307）：
  · #306：BACKGROUND_WORDS（白底/灰底/渐变蓝底/渐变粉底/渐变绿底/深色底）里的
    色字参与部件颜色子串竞争 ⇒「渐变绿底中间有蓝色圆形」画成绿色、「白底中间有
    灰色圆形」画成白色、「渐变蓝底中间有圆形」得 blue 且不落缺省账、纯背景子句
    生成幻影部件。
  · #307 同族：方位词/花纹词里的「心/方/条」参与形状子串竞争 ⇒「中心有蓝色菱形」
    画成心形、「右下方有蓝色菱形」画成方块。

本件锁：
  ① #306 五格：背景词场景的部件颜色取**部件色字**（gray/blue/white/red）；
  ② #306 幻影部件：纯背景子句「渐变蓝底，左上有红色圆」只产 1 个部件（r0 红），
     r4 无幻影；
  ③ #306 缺省落账：「渐变蓝底中间有圆形」无色字 ⇒ 缺省 red 且 defaults_applied
     含 color 账目；
  ④ #307 三格：方位/花纹语境的字不再触发形状（菱形不被心/方/条劫持）；
  ⑤ 反向对照：无背景词的普通句子颜色照旧（红/蓝不受影响）。
定点变异：剥背景词的两处循环删掉 ⇒ ①②③ 红。
"""
import pytest

from lingshu.nn.hex_gen import compile_description as c


# ---------------------------------------------------------------- ① #306 五格

@pytest.mark.parametrize("text,want", [
    ("白底中间有灰色圆形", "gray"),
    ("渐变绿底中间有蓝色圆形", "blue"),
    ("灰底右下有白色方块", "white"),
    ("渐变蓝底中间有圆形", "red"),     # 无色字 ⇒ 缺省 red
    ("中间有灰色圆形", "gray"),          # 无背景词对照
])
def test_background_color_words_do_not_leak(text, want):
    r = c(text)
    assert r["parts"], "%s 应产出部件" % text
    got = r["parts"][0]["color"]
    assert got == want, "%s 颜色被背景词劫持：%r（期望 %r）" % (text, got, want)


# ---------------------------------------------------------------- ② 幻影部件

def test_background_only_clause_makes_no_phantom_part():
    r = c("渐变蓝底，左上有红色圆")
    assert len(r["parts"]) == 1, (
        "纯背景子句不得生成幻影部件：%r" % ([(p["shape"], p["color"], p["zone"])
                                             for p in r["parts"]],))
    assert r["parts"][0]["zone"] == "r0" and r["parts"][0]["color"] == "red"


# ---------------------------------------------------------------- ③ 缺省落账

def test_missing_color_defaults_are_logged():
    r = c("渐变蓝底中间有圆形")
    fields = [d["field"] for d in r["defaults_applied"]]
    assert "color" in fields, "颜色缺省必须落账（旧实现得 blue 且不落账）：%r" % (fields,)


# ---------------------------------------------------------------- ④ #307 现状锁定

@pytest.mark.parametrize("text,got_shape", [
    ("中心有蓝色菱形", "diamond"),    # #307 已修：'心' 不再被方位语境劫持
    ("右下方有蓝色菱形", "diamond"),  # #307 已修：'方' 不再被方位语境劫持
])
def test_position_context_shape_hijack_is_documented(text, got_shape):
    """#307 已修：方位词里的单字不再被当成形状词（原「现状锁定」用例已反转）。

    上游本用例原先把 #307 的劫持行为如实钉住，并在 docstring 写明「修 #307 时把
    这两条的期望反转为 diamond 即可」——PR #909 rebase 到 c8fecc48 时按该指引反转。
    用例名保留（history 可查），语义已从「锁定缺陷」变为「锁定修复」。"""
    r = c(text)
    assert r["parts"][0]["shape"] == got_shape, (
        "%s 的形状行为变了（%r）——若这是 #307 的修复，请同步反转本用例期望"
        % (text, r["parts"][0]["shape"]))


def test_pattern_word_stripe_hijack_documented():
    """'条纹背景上有蓝色菱形' 现得 diamond——'条' 不再劫持形状（#307 已修）。

    与「中心/右下方」同族：上游原先把该劫持行为钉住并注明「#307 修复时一并反转」，
    本 PR rebase 时按指引反转；花纹仍应识别为 striped（'条纹' 仍是花纹词）。"""
    r = c("条纹背景上有蓝色菱形")
    assert r["parts"][0]["shape"] == "diamond", (
        "'条' 仍劫持形状：%r（#307 应得 diamond）" % (r["parts"][0]["shape"],))
    assert r["parts"][0]["pattern"] == "striped", (
        "花纹丢失：'条纹' 仍应识别为 striped，实得 %r" % (r["parts"][0]["pattern"],))


# ---------------------------------------------------------------- ⑤ 反向对照

def test_plain_colors_unchanged():
    # 对照句用逗号分隔（「和」不在本模块子句切分集，此前误当分隔符）
    r = c("中间有红色圆形,右上有蓝色三角")
    colors = [p["color"] for p in r["parts"]]
    assert "red" in colors and "blue" in colors, colors
