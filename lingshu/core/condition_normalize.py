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
# #138：非法值（不在枚举里 / 类型不对）与「未声明」必须区分——旧实现两者都写成
# UNASSERTED，于是「写错的条件」与「根本没声明」同 cond_hash，被静默串进同一个桶，
# 且 canonical_sourced / validate_edges 都无从报出（issue #138）。
INVALID = "__INVALID__"

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
    """域取值→(strip)→同义词归一化→canonical dict（键排序）。

    #138（口径变更，如实登记）：三种情形此前**都**落 UNASSERTED、彼此 cond_hash 相同，
    于是「写错的值」被静默路由进「未声明」的桶（与 P2-001「未声明无可校验」冲突）。
    现明确区分：
      · 缺键            → UNASSERTED（未声明，语义不变）；
      · 合法值（先 strip）→ 规范枚举值；**" 动漫 " 与 "动漫" 同值同 hash**；
      · 非法值          → INVALID（与未声明**不同** hash，且可被上游报出）。
    削掉的判别力：旧口径下「非法值」与「未声明」不可分辨；新口径下二者 hash 不同，
    故依赖「非法值≡未声明」的历史读数会变（仓内无此类调用方，见 tests 回归面）。
    """
    canon = {}
    for k, allowed in IMAGE_DOMAINS.items():
        if k not in domains:
            canon[k] = UNASSERTED          # 未声明：语义不变
            continue
        v = domains[k]
        v = v.strip() if isinstance(v, str) else v   # #138：先剥空白再查同义词
        v = SYNONYMS.get(v, v)
        canon[k] = v if v in allowed else INVALID
    return dict(sorted(canon.items()))


def invalid_domains(domains: Dict[str, str]) -> Dict:
    """#138：返回 {域: 原值}——本次输入里被判为非法（≠未声明）的域。"""
    canon = canonical_condition(domains)
    return {k: domains.get(k) for k, v in canon.items() if v == INVALID}


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
        if canon[k] == INVALID:            # #138：非法值必须可被调用方看见
            out[k]["invalid"] = domains.get(k)
    return out


def validate_edges(domains: Dict[str, str]) -> Dict:
    """R10：校验边（声明域×求值域的一致性裁决，卡 v0.2 §〇.A 反驳机制）。

    实作两条范式边（卡 §四：实作样例先行、其余增量补）：
      E1 声明↔求值：input_modality=线稿（**声明域**，卡 §二·输入模态声明）但
         求值域 line_edge=无线稿（**求值域**，卡 §二 域13 线条/边缘，像素判定）
         → 声明被求值域反驳 → 校验失败 → DEFER 复核
         （卡原文例：声明=线稿但检测到色块填充——"色块填充"在实现里即
          求值域读数 line_edge=无线稿）；
      E2 求值↔求值：pose 为运动态（行走/奔跑/回眸）却 symmetry=对称
         → 求值内部矛盾 → DEFER 复核。
    UNASSERTED 域不参与（未声明无可校验——P2-001 语义）。

    #133 修复（口径变更，如实登记）：
      旧 E1 的求值侧取 `color`，但 `DOMAIN_SOURCE["color"] == "asserted"`
      （卡 §二 域14：色彩 = asserted(枚举档)/evaluated(连续值)，枚举档为声明）
      ——E1 两端皆声明域，**「声明域可被求值域反驳」在实现里根本不成立**
      （声明×声明不构成反驳，且可在零像素证据下触发）。
      现求值侧改为**求值域** `line_edge`（`DOMAIN_SOURCE == "evaluated"`），
      使 E1 成为真正的「声明域 ← 求值域」反驳边。
      **削掉的判别力**：旧 E1 会在「声明=线稿 ∧ 声明色彩∈{柔和,丰富,多彩,霓虹}」
      时判 DEFER；该组合现仅两条弱声明并存，不再触发（两条声明互不构成反驳）。
      **换得的判别力**：只有当真有一条**求值域读数**与声明冲突时才 DEFER——
      不再把「两个声明不一致」误记为「求值反驳声明」。
    """
    c = canonical_condition(domains)
    violations = []
    # #138：非法值优先报出（E0）。它不是「未声明」，也不是「有冲突的读数」——
    # 而是**输入本身写错了**，此前被折叠成 UNASSERTED 后此函数完全看不见。
    bad = {k: v for k, v in c.items() if v == INVALID}
    if bad:
        violations.append({"edge": "E0_INVALID",
                           "note": "条件取值不在枚举内（既非合法值，也不等于未声明）",
                           "domains": {k: domains.get(k) for k in sorted(bad)}})
    im, le, pose, sym = (c["input_modality"], c["line_edge"],
                         c["pose"], c["symmetry"])
    if (im == "线稿" and le == "无线稿"):
        violations.append({"edge": "E1",
                           "note": "声明=线稿 但 求值线条=无线稿（色块填充）",
                           "domains": {"input_modality": im, "line_edge": le}})
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
