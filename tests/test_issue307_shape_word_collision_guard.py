# -*- coding: utf-8 -*-
"""#307 守卫：方位词/花纹词里的单字不得被当成形状词。

缺陷（main ea7a13c 实测）：compile_description 用 `_match_open` 在**整句**上匹配
开放形状词表，而方位词「中心/右下方/下方/中间」含「心/方」、花纹词「条纹/条形」含「条」，
单字碰撞导致 6 句错判：
    中心有红色         → heart  （期望 circle，无形状词应落缺省账）
    中心有蓝色菱形     → heart  （期望 diamond）
    右下方有蓝色菱形   → square （期望 diamond）
    下方有红色星       → square （期望 star）
    中间有红色实心菱形 → heart  （期望 diamond）
    中间有蓝色条纹菱形 → stripe （期望 diamond）

修复：形状匹配前先整词扣掉方位词/花纹词/尺寸词（`_strip_nonshape`），
颜色/花纹/尺寸仍按原句取。本守卫在修复前 3 个用例全红，修复后全绿。

词汇表事实（读码所得，非推断）：
    hex_text.SHAPE_WORDS = {圆形,圆→circle; 三角→triangle; 条纹→stripe}
    hex_gen.OPEN_SHAPES  = {圆,三角,条→stripe; 方→square; 矩形; 星; 心; 六边; 菱}
    hex_gen.PATTERN_WORDS= {实心→solid; 条纹,条形→striped; 点纹,斑点→dotted}
⇒ 「条纹」同时是形状词与花纹词（历史双义），「条形」只是花纹词。
  故「中间有条纹」仍得 stripe 形状（扣掉后回落窄表），「中间有条形」无形状词→落缺省。
"""
from lingshu.nn.hex_gen import compile_description

# —— 6 句原缺陷句：(描述, 期望 shape, 期望 zone, 期望 pattern) ——
DEFECT_CASES = [
    ("中心有红色",           "circle",  "r4", "solid"),
    ("中心有蓝色菱形",       "diamond", "r4", "solid"),
    ("右下方有蓝色菱形",     "diamond", "r8", "solid"),
    ("下方有红色星",         "star",    "r7", "solid"),
    ("中间有红色实心菱形",   "diamond", "r4", "solid"),
    ("中间有蓝色条纹菱形",   "diamond", "r4", "striped"),
]

# —— 无形状词的句子：必须落 shape 缺省账，不得从方位/花纹词里挖形状 ——
DEFAULT_CASES = [
    ("中心有红色",   "circle", "r4", "solid"),
    ("右下方有蓝色", "circle", "r8", "solid"),
    ("下方有红色",   "circle", "r7", "solid"),
    ("中间有条形",   "circle", "r4", "striped"),   # 条形=花纹词，非形状词
]

# —— 真形状词仍须被认出（防剥离过度） ——
REAL_SHAPE_CASES = [
    ("中心有圆形",   "circle",    "r4", "solid"),
    ("右下方有方形", "square",    "r8", "solid"),
    ("中间有条纹",   "stripe",    "r4", "striped"),  # 条纹是窄表形状词
    ("上方有六边形", "hexagon",   "r1", "solid"),
    ("左下有矩形",   "rectangle", "r6", "solid"),
]


def _part(text):
    return compile_description(text)["parts"][0]


def test_307_defect_sentences_get_written_shape():
    """6 句原缺陷句：形状必须等于用户写的形状词（修复前全部点红）。"""
    for text, exp_shape, exp_zone, exp_pattern in DEFECT_CASES:
        got = _part(text)
        assert got["shape"] == exp_shape, (
            "%r shape=%r 期望 %r（方位/花纹词里的单字被误当形状词）"
            % (text, got["shape"], exp_shape))
        assert got["zone"] == exp_zone, "%r zone=%r 期望 %r" % (
            text, got["zone"], exp_zone)
        assert got["pattern"] == exp_pattern, "%r pattern=%r 期望 %r" % (
            text, got["pattern"], exp_pattern)


def test_307_no_shape_word_falls_to_default():
    """无形状词的句子落 shape 缺省账，且缺省必须留痕于 defaults_applied。"""
    for text, exp_shape, exp_zone, exp_pattern in DEFAULT_CASES:
        res = compile_description(text)
        got = res["parts"][0]
        assert got["shape"] == exp_shape, (
            "%r shape=%r 期望缺省 %r" % (text, got["shape"], exp_shape))
        assert got["zone"] == exp_zone, "%r zone=%r 期望 %r" % (
            text, got["zone"], exp_zone)
        fields = [d["field"] for d in res["defaults_applied"]]
        assert "shape" in fields, (
            "%r 落 shape 缺省但未记入 defaults_applied（fields=%r）"
            % (text, fields))


def test_307_real_shape_words_still_matched():
    """真形状词仍须被认出（防剥离过度把真形状也扣掉）。"""
    for text, exp_shape, exp_zone, exp_pattern in REAL_SHAPE_CASES:
        got = _part(text)
        assert got["shape"] == exp_shape, (
            "%r shape=%r 期望 %r（真形状词被误剥离）"
            % (text, got["shape"], exp_shape))
        assert got["zone"] == exp_zone, "%r zone=%r 期望 %r" % (
            text, got["zone"], exp_zone)
# CI nudge 2026-10-24：本行无功能作用，仅重建提交以产生 pull_request/synchronize 事件。
# 起因：7 个分支 80 秒内连推，GitHub 合并/丢弃了后几个的 pull_request 事件；
# close+reopen、同 tree 重推、workflow_dispatch(403 非 admin) 均无效。
