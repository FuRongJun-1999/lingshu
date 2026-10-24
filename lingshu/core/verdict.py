# -*- coding: utf-8 -*-
"""verdict · 图像识别与分割方案 M2：四态判定（v0.4 资格判断的工程实现）
给每个部件候选赋 verdict：ACCEPT(条件充分,可作为事实)/REJECT(条件冲突)/DEFER(证据不足,待验)/BLINDSPOT(能力不可判)。
依据：部件前景占比 + 图像大域语义卡(occlusion/clothing/contrast/line_edge) + 遮挡推断。
白箱 · 确定性 · 零LLM(D-005)。

非有限输入闸（#196）：NaN 与任何阈值比较恒为 False，`ratio < 0.10` 不成立即
直落末尾 ACCEPT——损坏/未初始化的数值被静默判为「通过」。故 assess 入口先判
数值有限性，非有限（NaN/±Inf）一律 BLINDSPOT（能力不可判），不给出通过态。
判据来源：四态语义见 docs/theory/自研蜂窝CNN_理论稿_v0.1.md:33「BLINDSPOT(不可判带)」；
「非有限输入 ⇒ 不可判 ⇒ 不给 ACCEPT」这一步理论未规定 NaN 语义（追不到），
属工程 fail-closed 约定，与同仓 tests/test_issue402_nan_clamp_guard.py 头部
「非有限输入不得被当作有效证据/支持」同源。阈值本体（0.10/0.30）未改。

未声明域与零面积（#296）：
  · `occlusion` 取 `__UNASSERTED__`（condition_normalize 对「未声明」的编码）
    与**缺键**同义——均表示「无从断言遮挡」，**不是**「有遮挡」。旧实现只对缺键
    默认「无遮挡」，却把 UNASSERTED 判成 `!= "无遮挡"` ⇒ 同一图归一前后结论相反。
    判据：`condition_normalize.py:67-76`（UNASSERTED 编码）+ 卡 v0.2 P0-001
    「未声明无可校验」；`clothing`/`contrast` 对 UNASSERTED 本已与缺键同结论，无需改。
  · `part_area <= 0` ⇒ REJECT（无有效区域）。旧式 `_fg_ratio` 的 `max(1, part_area)`
    在零面积时把比值退化为前景像素数本身，越过全部阈值判 ACCEPT。零面积是
    整数边界（与 #196 的 NaN 面独立），判据：条件冲突 ⇒ REJECT（四态语义同源）。
"""
import math
from typing import Dict

from .condition_normalize import INVALID, UNASSERTED

# 图像大域 → 部件遮挡映射(常服/裙等覆盖 → 相应部件 occluded)
CLOTH_OCCLUDES = {
    "常服": ["torso", "upper_leg_L", "upper_leg_R", "lower_leg_L", "lower_leg_R"],
    "制服": ["torso", "upper_leg_L", "upper_leg_R"],
    "铠甲": ["torso", "upper_arm_L", "upper_arm_R", "upper_leg_L", "upper_leg_R"],
}


def _fg_ratio(part_fg_px, part_area):
    return part_fg_px / max(1, part_area)


def assess(part: Dict, image_domains: Dict, part_fg_px: int, part_area: int) -> Dict:
    """四态判定。part含type/bbox; image_domains 含 occlusion/clothing/contrast/line_edge。

    数值非有限（NaN/±Inf）→ BLINDSPOT（#196）：此类输入下全部阈值比较恒为
    False，会静默落进 ACCEPT 分支；这里先拦，不把「数值坏了」翻成「质量门通过」。
    """
    name = part["type"]
    if not (math.isfinite(part_fg_px) and math.isfinite(part_area)):
        return {"verdict": "BLINDSPOT", "reason": "non-finite input",
                "fg_ratio": None}
    ratio = _fg_ratio(part_fg_px, part_area)
    clothing = image_domains.get("clothing", "无")
    contrast = image_domains.get("contrast", "清晰")
    occluded = image_domains.get("occlusion", "无遮挡")
    # 未声明（condition_normalize.UNASSERTED）与**缺键**同义：都表示「无从断言遮挡」，
    # 不是「有遮挡」（issue #296）。旧实现在缺键时默认「无遮挡」、却把 UNASSERTED
    # 判成 `!= "无遮挡"` ⇒ 同一张图经 canonical_condition 归一后 4 个可见部件被判
    # DEFER，与未归一输入结论相反。判据：`condition_normalize.py:67-76` 的 UNASSERTED
    # 编码「未声明」，卡 v0.2 P0-001「未声明无可校验」——不得据此判「有遮挡」。
    # #138：非法 occlusion 取值（INVALID）同样「无从断言遮挡」——若只认 UNASSERTED，
    # 一个写错的 occlusion 会掉进下方 `occluded != "无遮挡"` ⇒ 被判「有遮挡」DEFER，
    # 与「未声明」结论相反。二者在此同义（报出非法值的责任在 canonical_sourced/validate_edges）。
    if occluded in (UNASSERTED, INVALID):
        occluded = "无遮挡"
    # 0) 部件区面积为零/负 ⇒ 无有效区域，条件冲突 REJECT（issue #296）。
    # 旧式 `_fg_ratio` 用 `max(1, part_area)`，零面积时比值退化为前景像素数本身
    # （如 5.0）⇒ 越过所有阈值被判 ACCEPT。零面积不是「条件充分」。
    if part_area <= 0:
        return {"verdict": "REJECT", "reason": "part region empty", "fg_ratio": 0.0}
    # 1) 部件区几乎无前景 → 条件冲突 REJECT
    if ratio < 0.10:
        return {"verdict": "REJECT", "reason": "part region empty", "fg_ratio": round(ratio,2)}
    # 2) 前景太稀疏, 难判 → BLINDSPOT
    if ratio < 0.30:
        return {"verdict": "BLINDSPOT", "reason": "ambiguous fg", "fg_ratio": round(ratio,2)}
    # 3) 被服装覆盖 / 遮挡 → 证据不足 DEFER(可见性待验)
    if name in CLOTH_OCCLUDES.get(clothing, []) or occluded != "无遮挡":
        return {"verdict": "DEFER", "reason": "clothing/occlusion covers", "fg_ratio": round(ratio,2),
                "occluded": True}
    # 4) 对比不足 → DEFER
    if contrast in ("模糊", "低对比"):
        return {"verdict": "DEFER", "reason": "low contrast", "fg_ratio": round(ratio,2)}
    # 5) 条件充分 → ACCEPT
    return {"verdict": "ACCEPT", "reason": "condition satisfied", "fg_ratio": round(ratio,2)}


if __name__ == "__main__":
    # img0 演示: 服装覆盖 torso/legs → DEFER; head/arm/hand 可见 → ACCEPT
    D = {"clothing":"常服","occlusion":"无遮挡","contrast":"清晰","line_edge":"淡线稿"}
    for t in ["head","torso","upper_arm_L","hand_L","upper_leg_L","foot_L"]:
        r = assess({"type":t}, D, part_fg_px=9000, part_area=10000)
        print(t, "->", r["verdict"], "|", r["reason"])
