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
"""
import math
from typing import Dict

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
