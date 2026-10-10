"""矛盾感知：近重复判定前的极性 / 数值 / 反义冲突检测（默认关闭，Refs #142）。

动机：
  M5 去重只看字符二元组 Jaccard。「用户对青霉素过敏」与「用户对青霉素不过敏」、
  「剂量 7.5 毫克」与「剂量 75 毫克」、「漏洞尚未修复」与「漏洞已修复」的 Jaccard 都很高，
  于是更正被当成重复吞掉，原命题反而 +0.02 增信；更正 N 次后错误事实置信度 → 1.0。
  这不是阈值问题——二元组集合本身就看不见一个「不」字带来的语义翻转。

做法（白箱、确定性、零 LLM，D-005）：
  只在**已经被判为近重复**的一对文本上做差分（difflib 字符级对齐），看差异片段：
  - 极性：差异片段里出现否定标记（不/没/未/无/非/别/勿/莫/否/毫无/并非/not/no/never…），
    且该标记不属于整词出现在片段内的非否定常用词（非常、无论、未来、不过[，]、否则…）；
  - 数值：两侧数字（阿拉伯数字含小数，及连续中文数字）多重集不同且两侧都有数字；
  - 反义：差异片段命中一张小反义表（已/未、成功/失败、允许/禁止、增加/减少…）。
  命中即判「冲突」，返回原因；否则返回 None（照旧去重）。

  只做「近重复对」上的差分，所以误报面被近重复阈值天然限定；同义改写（措辞变化、
  标点、语序微调）不含上述标记，照常合并。
"""
from __future__ import annotations

import difflib
import os
import re
from collections import Counter
from typing import Optional

ENV_FLAG = "LINGSHU_CONTRADICTION"

_NEG_MARKERS = ("并非", "并不", "毫无", "从未", "尚未", "没有", "不是", "不", "没", "未",
                "无", "非", "别", "勿", "莫", "否")
# 片段中**整词**出现时不算否定的常用词（只在整词完整落在差异片段里时豁免）
_NON_NEG = ("非常", "无论", "未来", "否则", "不断", "不仅", "不少", "不过，", "不过,",
            "别人", "别的", "特别", "区别", "分别", "莫名", "无比", "毫无疑问", "未必然",
            "没错", "不错", "没关系", "不客气", "不用谢", "不好意思", "无聊", "无数", "非洲")
_EN_NEG = re.compile(r"\b(not|no|never|none|cannot|n't|without)\b", re.I)
_ANTONYMS = (("已", "未"), ("成功", "失败"), ("允许", "禁止"), ("增加", "减少"),
             ("上升", "下降"), ("通过", "否决"), ("支持", "反对"), ("同意", "拒绝"),
             ("接受", "拒绝"), ("正确", "错误"), ("真", "假"), ("在线", "离线"),
             ("开启", "关闭"), ("存活", "死亡"), ("包含", "排除"), ("可以", "不能"),
             ("喜欢", "讨厌"), ("有效", "无效"), ("合法", "非法"), ("安全", "危险"))
_NUM = re.compile(r"\d+(?:\.\d+)?")
_CN_NUM = re.compile(r"[〇零一二两三四五六七八九十百千万亿]{2,}|[两三四五六七八九十百千万亿]")


def enabled() -> bool:
    return os.environ.get(ENV_FLAG, "").strip() == "1"


def _has_neg(seg: str) -> bool:
    if not seg:
        return False
    if _EN_NEG.search(seg):
        return True
    s = seg
    for w in _NON_NEG:
        s = s.replace(w, " ")
    return any(m in s for m in _NEG_MARKERS)


def _nums(s: str) -> Counter:
    c = Counter(float(x) for x in _NUM.findall(s))
    c.update("cn:" + x for x in _CN_NUM.findall(_NUM.sub(" ", s)))
    return c


def conflict(a: str, b: str) -> Optional[str]:
    """a、b 为近重复文本时，返回冲突原因（'polarity' / 'number' / 'antonym'）或 None。"""
    if not isinstance(a, str) or not isinstance(b, str) or a == b:
        return None
    na, nb = _nums(a), _nums(b)
    if na and nb and na != nb:
        return "number"
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        da, db = a[i1:i2], b[j1:j2]
        if _has_neg(da) != _has_neg(db):
            return "polarity"
        # 反义词可能被对齐切开（在线/离线 只差「在/离」）：在片段两侧各扩 3 字的窗口里找，
        # 且要求该词只出现在一侧（另一侧原文里没有），避免两边都含同一词时误报
        wa, wb = a[max(0, i1 - 3):i2 + 3], b[max(0, j1 - 3):j2 + 3]
        for x, y in _ANTONYMS:
            for p, q in ((x, y), (y, x)):
                if p in wa and q in wb and p not in b and q not in a:
                    return "antonym"
    return None
