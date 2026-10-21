# -*- coding: utf-8 -*-
"""wm_bridge.factory.film._det.gates_k · K 族判据门（连续性 · G1 硬门）

判据定义来源：架构文档 §4.2「K 族 · 连续性（全是 G1 硬门）」表（K1–K5，首批 P1）。
门的准入规则见 §4.1：**G1 只收 [标准] + [共识]**，行为是 REJECT / fail-closed。

★ 本模块的诚实边界（**先读这一节**）
    §六.2 定死了「判据必须吃事实」。但 §3.2 登记的七份契约**没有**承载全部事实：
    契约里只有**声明**（相机在哪、景别目标多少），没有**实测**（主体在画面哪一侧、
    视线朝哪、实际景别多大）。因此本模块对每条判据明确两件事：
      ① 它的**输入**来自声明还是事实；
      ② 缺料时的**行为** —— 一律 BLINDSPOT（跑了但判不了）并写明缺哪个字段，
         **绝不因为缺料而沉默通过**。缺什么由此变成可归档的读数，驱动下一轮补契约。

    | 判据 | 输入 | 缺料行为 |
    |---|---|---|
    | K1 轴线一致 | 声明（机位）+ 场景轴线 | 无轴线 ⇒ BLINDSPOT |
    | K2 视线匹配 | **事实**（`subject_gaze_screen`，§六.1 未列 —— 契约缺口） | 无该字段 ⇒ BLINDSPOT |
    | K3 屏幕方向 | **事实**（`subject_motion_screen` 或相邻帧 bbox 位移） | 无 ⇒ BLINDSPOT |
    | K4 30° 规则 | 声明（相邻镜头首帧机位） | 缺镜头契约 ⇒ BLINDSPOT |
    | K5 景别跳跃 | **事实**（`subject_bbox_screen`）+ 角度 | 无 bbox / 无画幅 / 夹角不可算 ⇒ BLINDSPOT |

★ K4 与 K5 的关系（**不是重复**）
    K4 吃**声明**：相邻同主体镜头首帧机位的夹角 ≥ 30°。它拦得住「同机位同景别」的跳切，
    但它**拦不住也放不过**一类合法剪辑 —— **cut-in**（机位几乎不动、景别猛跨级，
    如从 wide 直接切到 extreme_close）。故 K4 的不适用条件是 cut-in（见 `K4_NEGATIVE`）。
    K5 吃**事实**：实测景别相对差 ≤ 容差即「微变」；微变**且**机位夹角 < 30° 才 REJECT，
    于是 cut-in 由 K5 放行。
    两者对「同机位 + 同景别」这个跳切**判同一结论**；差别在**数据来源与读数**，
    这使二者在只有声明、或只有事实时仍各有一条能跑。
"""
from __future__ import annotations

from .. import _common as C
from . import scene as S

#: 本模块登记的阈值（名字 / 值 / 来源）。来源不得为空 —— 由 oracle 契约强制。
THRESHOLDS = (
    ("K4_min_angle_deg", 30.0,
     "影视语言共识：30° 规则（相邻同主体镜头机位夹角 ≥ 30°，否则观众感到跳）"),
    ("K5_scale_rel_tol", 0.20,
     "本方案 §4.2 K5 实现口径：实测主体高度相对差 ≤ 20% 视为同景别（实现口径，非外部规范）"),
    ("K1_axis_side_eps", 0.0,
     "本方案 §4.2 K1 实现口径：侧向符号由 (轴线方向 × 世界上方向) 点乘确定，容差 0（符号量）"),
)

#: K4 的不适用条件（negative）—— 纪律 4「填补 negative」
K4_NEGATIVE = (
    "cut-in：机位几乎不动（夹角 < 30°）但景别猛跨级（如 wide → extreme_close）时，"
    "K4 的纯角度判据会误判为越轴式跳切，此时应改由 K5 判（K5 吃实测景别，会放行）。",
    "主体在镜头内发生显著位移（走位调度）时，首帧机位夹角不代表全程机位关系，"
    "本版只取首帧，故此时 K4 的读数仅供参考。",
)

#: K5 的不适用条件
K5_NEGATIVE = (
    "真值缺 subject_bbox_screen 时不可判（本版一律 BLINDSPOT，不猜）。",
    "主体被遮挡 / 出画导致 bbox 非完整主体时，高度比例不代表景别。",
    "相邻镜头机位夹角不可算（缺镜头契约 / 瞄准点相差 > 0.5 m）时不可判 —— "
    "本版一律 BLINDSPOT（与 K4 同口径，reason 保留不可算原因），"
    "且不写 -1.0 之类的哨兵读数冒充实测夹角。",
)

#: 两镜头瞄准点差异超过此值（米）时，30° 规则基准不唯一 ⇒ BLINDSPOT
_AIM_TOL_M = 0.5


def _pair_angle(sc, id1, id2):
    """相邻同主体两镜头的首帧机位夹角（度）；不可算 ⇒ (None, 原因)。"""
    shots = S.shots_of(sc)
    s1, s2 = shots.get(id1), shots.get(id2)
    if s1 is None or s2 is None:
        return None, "缺镜头契约：%s" % (id1 if s1 is None else id2)
    p1 = S.first_camera(s1)
    p2 = S.first_camera(s2)
    aim1, aim2 = tuple(p1["aim"]), tuple(p2["aim"])
    if S.dist(aim1, aim2) > _AIM_TOL_M:
        return None, ("两镜头瞄准点相差 %.3f m（> %.2f m），30° 规则的夹角基准不唯一"
                      % (S.dist(aim1, aim2), _AIM_TOL_M))
    target = S.mean_point(aim1, aim2)
    try:
        return S.angle_at(target, tuple(p1["pos"]), tuple(p2["pos"])), None
    except S.SceneError as exc:
        return None, str(exc)


def _adjacent_same_subject(sc):
    """产出 (id1, id2, spec_name) 序列：相邻且**同主体**（spec_ref.name 相同）的镜头对。

    「同主体」由 shot.spec_ref.name 判定 —— 它绑定物与姿态的真源（§3.4 设计要点 1）。
    """
    ed = sc.get("edit")
    if ed is None:
        return []
    from .. import edit as ED
    n = ED.normalize_edit(ed)
    shots = S.shots_of(sc)
    out = []
    for a, b in zip(n["shots"], n["shots"][1:]):
        sa, sb = shots.get(a["id"]), shots.get(b["id"])
        if sa is None or sb is None:
            out.append((a["id"], b["id"], None))
            continue
        na = sa.get("spec_ref", {}).get("name")
        nb = sb.get("spec_ref", {}).get("name")
        if na is not None and na == nb:
            out.append((a["id"], b["id"], na))
    return out


# ══════════════════════════════════════════════════════ K1 轴线一致
def check_k1(sc):
    """K1 轴线一致：相邻镜头相机相对动作轴线的侧向符号必须一致（180° 规则）。"""
    name = "K1"
    if "axis" not in sc:
        return {"name": name, "verdict": "BLINDSPOT", "measured": [],
                "reason": "scene.axis 未声明 —— 180° 规则需要一个动作轴线（两点定一线），"
                          "而 wmfilm.shot/1 **不含**轴线字段（契约缺口）；"
                          "轴线未声明时判据不得猜测，故不可判"}
    pairs = _adjacent_same_subject(sc)
    if not pairs:
        return {"name": name, "verdict": "DEFER", "measured": [],
                "reason": "无可判的相邻同主体镜头对（缺 wmmv.edit/1 或镜头均不同主体）"}
    shots = S.shots_of(sc)
    readings, bad, missing = [], [], []
    for id1, id2, _spec in pairs:
        s1, s2 = shots.get(id1), shots.get(id2)
        if s1 is None or s2 is None:
            missing.append(id1 if s1 is None else id2)
            continue
        try:
            a = S.axis_side(sc["axis"], tuple(S.first_camera(s1)["pos"]))
            b = S.axis_side(sc["axis"], tuple(S.first_camera(s2)["pos"]))
        except S.SceneError as exc:
            return {"name": name, "verdict": "BLINDSPOT", "measured": [],
                    "reason": "轴线不可用：%s" % exc}
        readings.extend([float(a), float(b)])
        if a != 0 and b != 0 and a != b:
            bad.append("%s→%s" % (id1, id2))
    if missing:
        return {"name": name, "verdict": "BLINDSPOT", "measured": readings,
                "reason": "缺镜头契约：%s" % ", ".join(sorted(set(missing)))}
    if bad:
        return {"name": name, "verdict": "REJECT", "measured": readings,
                "threshold": "K1_axis_side_eps",
                "reason": "越轴：%s 的相机落在轴线异侧（侧向符号翻转）" % ", ".join(bad),
                "evidence": "相邻同主体镜头对 %d 组" % len(pairs)}
    return {"name": name, "verdict": "ACCEPT", "measured": readings,
            "threshold": "K1_axis_side_eps",
            "evidence": "相邻同主体镜头对 %d 组，侧向符号一致" % len(pairs)}


# ══════════════════════════════════════════════════════ K2 视线匹配
def check_k2(sc):
    """K2 视线匹配：正反打中 A 看画右 ⇒ B 看画左（视线水平分量符号相反）。"""
    name = "K2"
    pairs = _adjacent_same_subject(sc)
    if not pairs:
        return {"name": name, "verdict": "DEFER", "measured": [],
                "reason": "无可判的相邻同主体镜头对"}
    readings, bad, lack = [], [], []
    for id1, id2, _spec in pairs:
        g1 = _gaze(sc, id1)
        g2 = _gaze(sc, id2)
        if g1 is None or g2 is None:
            lack.append(id1 if g1 is None else id2)
            continue
        readings.extend([g1, g2])
        if g1 * g2 > 0:
            bad.append("%s→%s" % (id1, id2))
    if lack:
        return {"name": name, "verdict": "BLINDSPOT", "measured": readings,
                "reason": "真值缺 subject_gaze_screen（镜头 %s）—— §六.1 未列该回读字段，"
                          "属契约缺口；视线不可测时不得猜测" % ", ".join(sorted(set(lack)))}
    if bad:
        return {"name": name, "verdict": "REJECT", "measured": readings,
                "reason": "视线同向：%s 的两镜视线水平分量同号（正反打未对视）"
                          % ", ".join(bad)}
    return {"name": name, "verdict": "ACCEPT", "measured": readings,
            "evidence": "相邻同主体镜头对 %d 组，视线水平分量异号" % len(pairs)}


def _gaze(sc, shot_id):
    for e in S.truth_at(sc, shot_id):
        v = S.measured(e).get("subject_gaze_screen")
        if v is not None:
            if not C.is_num(v):
                raise S.SceneError("subject_gaze_screen 须为有限数值，实为 %r" % (v,))
            return float(v)
    return None


# ══════════════════════════════════════════════════════ K3 屏幕方向
def check_k3(sc):
    """K3 屏幕方向：同一场景内主体运动方向不得翻转（符号一致）。"""
    name = "K3"
    pairs = _adjacent_same_subject(sc)
    if not pairs:
        return {"name": name, "verdict": "DEFER", "measured": [],
                "reason": "无可判的相邻同主体镜头对"}
    readings, bad, lack = [], [], []
    for id1, id2, _spec in pairs:
        d1 = _motion(sc, id1)
        d2 = _motion(sc, id2)
        if not d1 or not d2:            # None（缺料）与 0.0（无位移）都无从判方向
            lack.append(id1 if not d1 else id2)
            continue
        readings.extend([d1, d2])
        if d1 * d2 < 0:
            bad.append("%s→%s" % (id1, id2))
    if lack:
        return {"name": name, "verdict": "BLINDSPOT", "measured": readings,
                "reason": "真值不足（镜头 %s）：既无 subject_motion_screen，"
                          "也无法由相邻帧 subject_bbox_screen 求位移 —— 缺料不猜"
                          % ", ".join(sorted(set(lack)))}
    if bad:
        return {"name": name, "verdict": "REJECT", "measured": readings,
                "reason": "屏幕方向翻转：%s 的相邻镜头主体运动方向异号" % ", ".join(bad)}
    return {"name": name, "verdict": "ACCEPT", "measured": readings,
            "evidence": "相邻同主体镜头对 %d 组，运动方向符号一致" % len(pairs)}


def _motion(sc, shot_id):
    """主体屏幕运动方向符号：优先取实测字段，否则由首末帧 bbox 中心位移求。"""
    entries = S.truth_at(sc, shot_id)
    for e in entries:
        v = S.measured(e).get("subject_motion_screen")
        if v is not None:
            if not C.is_num(v):
                raise S.SceneError("subject_motion_screen 须为有限数值，实为 %r" % (v,))
            return float(v)
    pts = []
    for e in entries:
        bbox = S.measured(e).get("subject_bbox_screen")
        if bbox is not None:
            pts.append((e.get("frame"), S.centroid(bbox)))
    if len(pts) < 2:
        return None
    pts.sort(key=lambda x: x[0])
    dx = pts[-1][1][0] - pts[0][1][0]
    if abs(dx) <= 1e-9:
        return None                     # 无位移 ⇒ 屏幕方向无从判定（不猜 0）
    return 1.0 if dx > 0 else -1.0


# ══════════════════════════════════════════════════════ K4 30° 规则
def check_k4(sc):
    """K4 30° 规则：相邻同主体镜头首帧机位夹角 ≥ 30°（吃**声明**）。"""
    name = "K4"
    pairs = _adjacent_same_subject(sc)
    if not pairs:
        return {"name": name, "verdict": "DEFER", "measured": [],
                "reason": "无可判的相邻同主体镜头对"}
    readings, bad, lack = [], [], []
    for id1, id2, _spec in pairs:
        ang, why = _pair_angle(sc, id1, id2)
        if ang is None:
            lack.append("%s→%s（%s）" % (id1, id2, why))
            continue
        readings.append(ang)
        if ang < THRESHOLDS[0][1]:
            bad.append("%s→%s=%.2f°" % (id1, id2, ang))
    if lack:
        return {"name": name, "verdict": "BLINDSPOT", "measured": readings,
                "reason": "不可算：" + "；".join(lack)}
    if bad:
        return {"name": name, "verdict": "REJECT", "measured": readings,
                "threshold": "K4_min_angle_deg",
                "reason": "机位夹角不足 30°：%s" % ", ".join(bad),
                "evidence": "不适用条件（cut-in 等）见 K4_NEGATIVE"}
    return {"name": name, "verdict": "ACCEPT", "measured": readings,
            "threshold": "K4_min_angle_deg",
            "evidence": "相邻同主体镜头对 %d 组，夹角均 ≥ 30°" % len(pairs)}


# ══════════════════════════════════════════════════════ K5 景别跳跃
def check_k5(sc):
    """K5 景别跳跃：同主体相邻镜头实测景别**不得微变**（吃**事实**）。

    微变（实测高度相对差 ≤ 容差）且机位夹角 < 30° ⇒ REJECT（既不换景别也不换机位
    = 跳切）；夹角 ≥ 30° ⇒ 放行（换机位可同景别，这是正常剪辑）。
    机位夹角不可算（缺镜头契约 / 瞄准点相差 > 0.5 m）⇒ BLINDSPOT —— 与 K4 同款
    缺料口径：reason 保留 `_pair_angle` 的不可算原因，且不写 -1.0 哨兵冒充读数。
    """
    name = "K5"
    pairs = _adjacent_same_subject(sc)
    if not pairs:
        return {"name": name, "verdict": "DEFER", "measured": [],
                "reason": "无可判的相邻同主体镜头对"}
    try:
        fsz = S.frame_size_of(sc)
    except S.SceneError as exc:
        return {"name": name, "verdict": "BLINDSPOT", "measured": [],
                "reason": "画幅不可用：%s" % exc}
    readings, bad, lack, no_angle = [], [], [], []
    for id1, id2, _spec in pairs:
        h1 = _scale_frac(sc, id1, fsz)
        h2 = _scale_frac(sc, id2, fsz)
        if h1 is None or h2 is None:
            lack.append(id1 if h1 is None else id2)
            continue
        rd = S.rel_diff(h1, h2)
        ang, why = _pair_angle(sc, id1, id2)
        if ang is None:              # 与 K4 同款缺料口径：不可算 ⇒ BLINDSPOT 不猜，
            no_angle.append("%s→%s（%s）" % (id1, id2, why))   # 原因进 reason 可归档
            continue                 # 不写 -1.0 —— 哨兵不是实测夹角（#812）
        readings.extend([rd, ang])
        if rd <= THRESHOLDS[1][1] and ang < THRESHOLDS[0][1]:
            bad.append("%s→%s=%.3f" % (id1, id2, rd))
    if lack:
        return {"name": name, "verdict": "BLINDSPOT", "measured": readings,
                "reason": "真值缺 subject_bbox_screen（镜头 %s）—— 实测景别不可得，不猜"
                          % ", ".join(sorted(set(lack)))}
    if no_angle:
        return {"name": name, "verdict": "BLINDSPOT", "measured": readings,
                "reason": "机位夹角不可算：%s —— 缺料不猜，不写哨兵读数"
                          % "；".join(no_angle)}
    if bad:
        return {"name": name, "verdict": "REJECT", "measured": readings,
                "threshold": "K5_scale_rel_tol",
                "reason": "景别微变且机位夹角 < 30°（跳切）：%s" % ", ".join(bad),
                "evidence": "读数格式：[景别相对差, 机位夹角] 成对"}
    return {"name": name, "verdict": "ACCEPT", "measured": readings,
            "threshold": "K5_scale_rel_tol",
            "evidence": "相邻同主体镜头对 %d 组，景别均跨级或机位夹角 ≥ 30°" % len(pairs)}


def _scale_frac(sc, shot_id, frame_size):
    hs = []
    for e in S.truth_at(sc, shot_id):
        bbox = S.measured(e).get("subject_bbox_screen")
        if bbox is not None:
            hs.append(S.bbox_height_frac(bbox, frame_size))
    if not hs:
        return None
    return sum(hs) / len(hs)


ALL_K = (check_k1, check_k2, check_k3, check_k4, check_k5)
