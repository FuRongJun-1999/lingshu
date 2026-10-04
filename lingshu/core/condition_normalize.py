# -*- coding: utf-8 -*-
"""condition_normalize · 图像识别与分割方案 M1：条件归一化 / cond_hash
解决 cond_key 泛化边界(同一图的描述串不同→hash碎片化)。
用图像大域语义卡(canonical domain)做归一化骨架：
  原始视觉描述 → 同义词归一化 → canonical condition set(枚举值) → cond_hash。
白箱 · 确定性 · 零LLM(D-005)。
"""
import hashlib, json
from typing import Dict, List

# 图像大域语义卡 v0.2 的 canonical 域(枚举取值) —— 归一化骨架
# R10：补齐卡 §二 与实现的两个缺失域（主体优先级/构图——均为求值域）
IMAGE_DOMAINS = {
    "subject":        ["人物","景色","物品","动物"],
    "count":          ["单体","双体","群像"],
    "subject_priority": ["主体","配角","环境"],
    "composition":    ["居中","三分","偏移","留白","对称构图"],
    "era_theme":      ["现代","古风","未来","赛博","奇幻","日常"],
    "pose":           ["静止","站姿","坐姿","躺姿","行走","奔跑","回眸"],
    "view_distance":  ["特写","半身","全身","远景"],
    "view_dir":       ["正视图","侧视","俯视","仰视","背影"],
    "frame":          ["竖幅","横幅","方形","全景","宽幅"],
    "background":     ["白","黑","简单纯色","透明","景色"],
    "dof":            ["浅景深","深景深","前景模糊"],
    "style":          ["动漫","写实","水彩","扁平","3D"],
    "line_edge":      ["粗线稿","淡线稿","无线稿","硬边","描边","软边"],
    "color":          ["黑白","柔和","丰富","多彩","霓虹"],
    "lighting":       ["暗","柔和","明亮","逆光","霓虹光"],
    "contrast":       ["模糊","清晰","高对比","低对比"],
    "material":       ["光滑","粗糙","金属","玻璃","布料","蕾丝"],
    "detail":         ["高清","低清","噪点","精细","极简"],
    "abstraction":    ["符号化","简化","拟真","写实化"],
    "occlusion":      ["无遮挡","部分遮挡","全遮挡"],
    "symmetry":       ["对称","部分对称","不对称","镜像"],
    "clothing":       ["无","泳装","常服","铠甲","制服"],
    "input_modality": ["彩色立绘","线稿","mask","PSD分层"],
}
UNASSERTED = "__UNASSERTED__"

# R10 · source/asserted_by 三字段（条件卡 v0.2 §二 声明-求值二分 P0-001 的实装）
# evaluated=求值器能从像素判定（强验证）；asserted=声明来源（弱验证，可被求值域反驳）
DOMAIN_SOURCE = {
    "subject": "asserted", "count": "evaluated", "subject_priority": "evaluated",
    "composition": "evaluated", "era_theme": "asserted", "pose": "evaluated",
    "view_distance": "evaluated", "view_dir": "evaluated", "frame": "evaluated",
    "background": "evaluated", "dof": "evaluated", "style": "asserted",
    "line_edge": "evaluated", "color": "asserted", "lighting": "asserted",
    "contrast": "evaluated", "material": "asserted", "detail": "evaluated",
    "abstraction": "asserted", "occlusion": "evaluated", "symmetry": "evaluated",
    "clothing": "asserted", "input_modality": "asserted",
}
# 声明域的默认声明来源（按卡 §〇.A：生图参数/人工标注/folder 描述）
ASSERTED_BY_DEFAULT = {
    "style": "人工标注", "era_theme": "人工标注", "material": "人工标注",
    "lighting": "人工标注", "clothing": "人工标注", "color": "枚举档声明",
    "subject": "人工标注", "input_modality": "输入声明",
    "abstraction": "人工标注",
}
# 同义词归一化(描述变体→规范枚举值)—— cond_key 泛化边界的关键
SYNONYMS = {
    "可爱动漫":"动漫","日系动漫":"动漫","二次元":"动漫","少女漫画":"动漫",
    "全身立绘":"全身","立绘":"全身","正面":"正视图","正立":"正视图",
    "纯白":"白","白底":"白","浅色底":"白","平涂":"扁平",
}


def canonical_condition(domains: Dict[str, str]) -> Dict:
    """域取值→同义词归一化→canonical dict(仅合法枚举值,未知→UNASSERTED,键排序)。"""
    canon = {}
    for k, allowed in IMAGE_DOMAINS.items():
        v = domains.get(k, UNASSERTED)
        v = SYNONYMS.get(v, v)
        if v not in allowed:
            v = UNASSERTED
        canon[k] = v
    return dict(sorted(canon.items()))


def canonical_sourced(domains: Dict[str, str],
                      asserted_by: Dict[str, str] = None) -> Dict:
    """R10：canonical + source/asserted_by 三字段（卡 v0.2 §三 KCCS 格式）。

    每域输出 {value, source, asserted_by, validate}：
      source=evaluated → validate=强验证，asserted_by=None；
      source=asserted  → validate=弱验证（可被求值域反驳，见 validate_edges）。
    注意：本结构不参与 cond_hash（hash 只取值层，保持路由稳定）。
    """
    canon = canonical_condition(domains)
    by = dict(ASSERTED_BY_DEFAULT)
    by.update(asserted_by or {})
    out = {}
    for k in sorted(IMAGE_DOMAINS):
        src = DOMAIN_SOURCE[k]
        out[k] = {"value": canon[k], "source": src,
                  "asserted_by": by.get(k) if src == "asserted" else None,
                  "validate": "强验证" if src == "evaluated" else "弱验证"}
    return out


def validate_edges(domains: Dict[str, str]) -> Dict:
    """R10：校验边（声明域×求值域的一致性裁决，卡 v0.2 §〇.A 反驳机制）。

    实作两条范式边（卡 §四：实作样例先行、其余增量补）：
      E1 声明↔求值：input_modality=线稿（声明）但 color 求值为色块档
         → 校验失败 → DEFER 复核（卡原文例：声明=线稿但检测到色块填充）；
      E2 求值↔求值：pose 为运动态（行走/奔跑/回眸）却 symmetry=对称
         → 求值内部矛盾 → DEFER 复核。
    UNASSERTED 域不参与（未声明无可校验——P2-001 语义）。
    """
    c = canonical_condition(domains)
    violations = []
    im, color, pose, sym = (c["input_modality"], c["color"],
                            c["pose"], c["symmetry"])
    if (im == "线稿" and color in ("柔和", "丰富", "多彩", "霓虹")):
        violations.append({"edge": "E1",
                           "note": "声明=线稿 但 求值色彩=色块档",
                           "domains": {"input_modality": im, "color": color}})
    if (pose in ("行走", "奔跑", "回眸") and sym == "对称"):
        violations.append({"edge": "E2",
                           "note": "运动姿态与对称求值矛盾",
                           "domains": {"pose": pose, "symmetry": sym}})
    return {"state": "DEFER_REVIEW" if violations else "consistent",
            "violations": violations}


def cond_hash(domains: Dict[str, str], salt: str = "") -> str:
    """canonical condition set → cond_hash(sha256 前16hex)。只与域取值有关,与描述串无关。"""
    c = canonical_condition(domains)
    payload = json.dumps(c, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256((salt + payload).encode("utf-8")).hexdigest()[:16]


def part_cond_hash(part_type: str, image_hash: str, part_idx: int) -> str:
    """部件级 cond_hash = hash(image_cond_hash + part_type + part_idx)。"""
    return hashlib.sha256(("%s|%s|%d" % (image_hash, part_type, part_idx)).encode("utf-8")).hexdigest()[:16]


def img0_domains() -> Dict[str, str]:
    return {"subject":"人物","count":"单体","pose":"站姿","view_distance":"全身",
            "view_dir":"正视图","frame":"竖幅","background":"白","style":"可爱动漫",
            "line_edge":"淡线稿","color":"柔和","lighting":"柔和","contrast":"清晰",
            "symmetry":"对称","clothing":"常服","input_modality":"彩色立绘"}


if __name__ == "__main__":
    d = img0_domains()
    h = cond_hash(d)
    print("img0 canonical:", json.dumps(canonical_condition(d), ensure_ascii=False)[:200])
    print("img0 cond_hash:", h)
    # 泛化测试: "可爱动漫"→动漫, 与直接"动漫"应同 hash
    d2 = dict(d); d2["style"] = "动漫"; h2 = cond_hash(d2)
    print("同义词一致(可爱动漫==动漫)?", h == h2)
    print("part head cond_hash:", part_cond_hash("head", h, 0))
