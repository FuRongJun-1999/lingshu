# -*- coding: utf-8 -*-
"""timeline · 剪辑时间线（`wmmv.bind/1` + `wmmv.edit/1` + `wmmv.text/1` + 全片镜头表）

依据 `docs/plans/影视级视听语言_理论架构_v0.1.md`（文内 v0.2）：
  · §3.4 镜头层 `wmfilm.shot/1`（camera · compose · light）
  · §4.2 E 族 E1「卡点对齐 ≤ 1 帧」= **G1 硬门**（本模块的第一纪律）
  · §七 P4 音画绑定 `wmmv.bind/1` / P5 全片剪辑时间轴 `wmmv.edit/1`

设计口径（每条都有理由，便于复核）：
  1. **卡点由构造保证**：所有切点都是「downbeat 秒 → 帧号四舍五入」得到的帧，
     因此 `offset_frames` 恒为 0（见 `cut_alignment_report` 的独立复算面，不靠自证）。
  2. **段落边界也必须是 downbeat**：边界不在拍网格上（`_snap_boundary`）⇒ 直接抛，
     绝不产出非卡点切点（**E1 硬门优先于 E3 段落边界可检出**，不静默偏移）。
  3. **骨架优先（已裁 D1）**：先铺满全片、≥ 8 镜；段内镜头数 = round(段落小节数 / 能量档位
     目标小节数)，`move_bias ≥ 0.6` 的段落再减 1 小节（高 bias ⇒ 切得更碎）；
     仍不足 8 镜 ⇒ 继续细分「ASL 最长且可再分」的镜头（细分点仍是 downbeat）。
  4. **ASL 由 bind 的 energy_map 驱动**：低能量档 ⇒ 多小节 ⇒ 长 ASL；高能量档 ⇒ 1 小节 ⇒ 短 ASL。
  5. **能量映射单调**：`energy_map.bands` 的 `asl_frames` 递减、景别由远及近，
     覆盖 [0,1] 且无缝（契约层强制）。
  6. **相邻镜头必须变化**：`move_kind` 不同、景别跨级、机位方位角相邻差 ≥ 30°（K4 30° 规则友好）
     且全在轴线同一侧（K1 轴线一致）——`wmfilm.shot/1` 无 shot_scale 字段，
     景别由 `compose.coverage_target` 反查（见 `shot_scale_of`，供 K5 判据复用）。
  7. **文本硬拦在契约层**：每 cue ≤2 行、每行 ≤42 字符、CPS ≤ min(段落 cps_max, 20)；
     按段长装不下 ⇒ 抛 `TimelineError`（不删歌词、不偷改 cap、不静默截断）。
     歌词 ↔ 段落按**出现次序**配对（同名段落可能重复，按名字配会让 cue 重叠）。
  8. **缺字体不假装**：`render_overlay` 取不到字体就返回原图，
     并在 `last_overlay_info()` 里如实标注 `reason="missing_font"`。

只读 `contracts.py` 的闭集（`MOVE_KINDS` / `SHOT_SCALES` / `CUT_ONS` / `TRANSITION_KINDS` /
`SUBJECT_ANCHORS` / `MOODS`），**不另起一份**。
"""
from __future__ import annotations

import hashlib
import json
import math

from . import contracts as C
from . import paths

paths.ensure_lingshu()  # INTERFACES §0：把上游克隆插进 sys.path（幂等）

__all__ = [
    "TimelineError", "stub_score",
    "build_bind", "plan_shot_table", "build_edit", "build_text",
    "write_srt", "render_overlay", "last_overlay_info",
    "cut_alignment_report", "shot_scale_of", "has_cjk",
    "camera_side", "shot_side_report",
    "camera_right", "shot_screen_direction_signs",
]


class TimelineError(Exception):
    """时间线不合法（fail-closed：不猜、不补、不静默降级）。"""


# ---------------------------------------------------------------- 小工具

def _fps(brief: dict) -> int:
    return int(round(float(brief.get("fps", 24))))


def _total_frames(brief: dict) -> int:
    return int(round(float(brief["duration_s"]) * _fps(brief)))


def _sha16(s: str) -> str:
    #: ★ 完整 64 位：P0（`shot.spec_ref` / `<x>_ref.digest`）的 `_digest_str` 规定
    #: "形如十六进制就必须是 64 位 sha256"，截断到 16 位会被当场拒收。
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _brief_digest(brief: dict) -> str:
    """镜头 `spec_ref.digest` 的确定性绑定源。

    正式链路里 `spec_ref` 应指向 scene 的 `spec/1` digest；MV 产线尚未冻结 scene 契约，
    这里退到「brief 的内容摘要」并**显式记录**该口径（不假装绑定了不存在的 spec）。
    """
    return _sha16(C.canonical(brief))


def _sec_frames(score: dict, fps: int):
    """乐谱段落 → 帧区间（闭合 [start, end]；end 取四舍五入后的帧号）。"""
    out = []
    for s in score["sections"]:
        a = int(round(float(s["start_s"]) * fps))
        b = int(round(float(s["end_s"]) * fps))
        out.append({"kind": s["kind"], "start": a, "end": b,
                    "energy": float(s["energy"])})
    return out


def _section_at(secs, frame: int) -> int:
    """frame 落在第几个段落（越界则夹到首/末段）。"""
    for i, s in enumerate(secs):
        if s["start"] <= frame < s["end"]:
            return i
    return 0 if frame < secs[0]["start"] else len(secs) - 1


def _bar_frames(score: dict, fps: int) -> int:
    """一小节的帧数（downbeat 间距）。"""
    tp = score["tempo"]
    meter = tp.get("meter", [4, 4])
    beat_s = 60.0 / float(tp["bpm"])
    return max(1, int(round(int(meter[0]) * beat_s * fps)))


def _downbeat_frames(score: dict, fps: int, total: int):
    fs = sorted({int(round(float(b) * fps)) for b in score["downbeats"]})
    return [f for f in fs if 0 <= f <= total]


def _align_for(frame: int, score: dict, fps: int) -> dict:
    """切点 → 最近 downbeat 的**如实**记录（E1 的数据面）。"""
    best = None
    for b in score["downbeats"]:
        bf = int(round(float(b) * fps))
        d = abs(frame - bf)
        if best is None or d < best[0] or (d == best[0] and float(b) < best[1]):
            best = (d, float(b), bf)
    if best is None:
        raise TimelineError("score.downbeats 为空，无法做卡点对齐（缺料 ⇒ 不猜）")
    return {"at_frame": int(frame), "beat_s": round(best[1], 6),
            "offset_frames": int(frame - best[2])}


# ---------------------------------------------------------------- 探针乐谱

def stub_score(brief: dict) -> dict:
    """**仅供无 audio 交付时的探针/自测**：按 brief 的 bpm/meter/sections/duration 造合法乐谱。

    正式链路以 `audio.make(brief, out)["score"]` 为准（拍点精确可算：`beats[i]=offset+i*60/bpm`）。
    这里刻意保持同一口径（含末拍 == duration_s 时的收尾拍），使卡点复算有意义。
    """
    fps = _fps(brief)
    dur = float(brief["duration_s"])
    tp = brief["tempo"]
    bpm = float(tp["bpm"])
    meter = list(tp.get("meter", [4, 4]))
    offset = float(tp.get("offset_s", 0.0))
    beat_dt = 60.0 / bpm
    beats = []
    i = 0
    while True:
        t = offset + i * beat_dt
        if t > dur + 1e-9 or i > 100000:
            break
        beats.append(round(t, 6))
        i += 1
    if not beats:
        raise TimelineError("brief 的 bpm/duration 造不出任何拍点")
    per_bar = max(1, int(meter[0]))
    downbeats = [b for j, b in enumerate(beats) if j % per_bar == 0]
    secs = []
    for s in brief["sections"]:
        secs.append({"kind": s["kind"], "start_s": round(float(s["start_s"]), 6),
                     "end_s": round(float(s["end_s"]), 6),
                     "energy": float(s["energy"])})
    secs[-1]["end_s"] = round(dur, 6)          # 末段必须 == duration_s（契约强制）
    hop = 0.05
    n = int(math.ceil(dur / hop))
    vals = []
    for k in range(n):
        t = k * hop
        e = secs[-1]["energy"]
        for s in secs:
            if s["start_s"] <= t < s["end_s"]:
                e = s["energy"]
                break
        vals.append(round(float(e), 6))
    doc = {
        "schema": "wmmv.score/1", "id": "score_main",
        "source": {"path": "music.wav", "digest": _sha16("stub:" + C.canonical(brief)),
                   "duration_s": dur, "sample_rate": 44100, "channels": 2},
        "tempo": {"bpm": bpm, "meter": [int(x) for x in meter], "offset_s": offset},
        "beats": beats, "downbeats": downbeats, "sections": secs,
        "energy": {"hop_s": hop, "metric": "rms", "values": vals},
    }
    if "key" in brief and brief["key"]:
        doc["key"] = {"tonic": brief["key"]["tonic"], "mode": brief["key"]["mode"],
                      "confidence": 0.5}
    return C.normalize("score", doc)


# ---------------------------------------------------------------- bind

#: 能量分档（低能量 → 长 ASL + 宽景别；高能量 → 短 ASL + 中近景）
_BANDS = (
    {"lo": 0.00, "hi": 0.30, "asl_frames": 144, "shot_scale": "extreme_wide", "move_intensity": 0.15},
    {"lo": 0.30, "hi": 0.50, "asl_frames": 96, "shot_scale": "wide", "move_intensity": 0.35},
    {"lo": 0.50, "hi": 0.72, "asl_frames": 72, "shot_scale": "medium_wide", "move_intensity": 0.55},
    {"lo": 0.72, "hi": 1.00, "asl_frames": 48, "shot_scale": "medium", "move_intensity": 0.85},
)


def _mood_for_energy(energy: float) -> str:
    if energy < 0.40:
        return "low_key"
    if energy <= 0.72:
        return "neutral"
    return "high_key"


def _band_for_energy(energy: float) -> dict:
    for b in _BANDS:
        if b["lo"] - 1e-9 <= energy <= b["hi"] + 1e-9:
            return b
    return _BANDS[-1] if energy > _BANDS[-1]["hi"] else _BANDS[0]


def build_bind(brief: dict, score: dict) -> dict:
    """`wmmv.bind/1`：拍点→切点 · 能量→ASL/景别/运动量 · 段落→基调。

    切点依据**恒为 downbeat**（E1 硬门），`quantize_beats=1`（每小节一个候选切点）、
    `tolerance_frames=1`（与 §4.2 E1 阈值同口径）。
    """
    score = C.normalize("score", score)
    brief_sections = brief.get("sections", [])

    # section_map 是**kind 级**默认表（契约以 section kind 为键）；同一 kind 多次出现时
    # 取「能量最高的一次」——保证默认表不会低于实际驱动量（如实、可复算）。
    by_kind = {}
    for s in brief_sections:
        k = s["kind"]
        cur = by_kind.get(k)
        if cur is None or float(s["energy"]) > float(cur["energy"]):
            by_kind[k] = s
    section_map = []
    for k in sorted(by_kind, key=lambda x: list(by_kind).index(x)):
        s = by_kind[k]
        #: `wmmv.bind/1` 的 `section_map.move_bias` 是 **MOVE_KINDS 闭集字符串**
        #: （= 该段落偏好的机位运动），不是 0~1 幅度。复用镜头表同一套分档，
        #: 保证"默认表"与"实际选择"一致（否则契约写着 dolly、画面却在 orbit）。
        section_map.append({"section": k, "shot_scale": s["shot_scale"],
                            "move_bias": _move_kind_for(float(s["move_bias"]), 0, None),
                            "mood": _mood_for_energy(float(s["energy"]))})

    key = score.get("key") or {}
    if key.get("mode") == "minor":
        timbre_map = [
            {"timbre": "dark", "hue_deg": 232.0, "saturation": 0.55, "value": 0.34},
            {"timbre": "cold", "hue_deg": 198.0, "saturation": 0.42, "value": 0.62},
            {"timbre": "percussive", "hue_deg": 318.0, "saturation": 0.68, "value": 0.78},
        ]
    else:
        timbre_map = [
            {"timbre": "warm", "hue_deg": 38.0, "saturation": 0.52, "value": 0.70},
            {"timbre": "bright", "hue_deg": 52.0, "saturation": 0.60, "value": 0.82},
        ]

    doc = {
        "schema": "wmmv.bind/1", "id": "bind_main",
        "score_ref": {"name": score["id"], "digest": C.digest("score", score)},
        "cut_rule": {"on": "downbeat", "quantize_beats": 1, "tolerance_frames": 1},
        "energy_map": {"metric": score["energy"]["metric"], "bands": [dict(b) for b in _BANDS]},
        "timbre_map": timbre_map,
        "section_map": section_map,
    }
    return C.normalize("bind", doc)


# ---------------------------------------------------------------- 镜头表

#: move_bias 分档 → 运动闭集子集（低 bias 静观，高 bias 激进；全部 ∈ contracts.MOVE_KINDS）
_MOVE_TIERS = (
    (0.34, ("static", "crane_up", "orbit")),
    (0.62, ("orbit", "truck_left", "tilt_up", "dolly_in")),
    (1.01, ("dolly_in", "whip_pan", "truck_left", "handheld", "dolly_out")),
)

#: 景别 → (机位距离 m, 机位高度 m, 35mm 等效焦距)
#: **Y-up 世界**（地面 y=0；主体胸口 aim_y≈0.9）：机位高度取 1.45~1.8 m 人眼/略高机位，
#: 与 Lead 的 `Camera3D.from_look_at(..., up=(0,1,0))` 同一口径（crane_up 会在镜内抬升）。
_SCALE_RIG = {
    "extreme_wide": (14.0, 1.75, 21.0),
    "wide": (10.0, 1.70, 24.0),
    "medium_wide": (7.0, 1.65, 28.0),
    "medium": (5.2, 1.60, 35.0),
    "medium_close": (3.6, 1.58, 45.0),
    "close": (2.5, 1.55, 60.0),
    "extreme_close": (1.6, 1.50, 85.0),
}

#: 景别 → 主体覆盖率目标（C1 口径：hero 0.68；越近越大，越远越小）
_COVERAGE = {"extreme_wide": 0.35, "wide": 0.45, "medium_wide": 0.55, "medium": 0.62,
             "medium_close": 0.68, "close": 0.72, "extreme_close": 0.78}

#: 景别 → 主体锚点（闭集 SUBJECT_ANCHORS）
_ANCHOR = {"extreme_wide": "center", "wide": "center", "medium_wide": "left_third",
           "medium": "right_third", "medium_close": "left_third",
           "close": "right_third", "extreme_close": "center"}

#: 机位方位角表（度；`atan2(dx, dz)`，相对瞄准点）。
#:
#: **K1 轴线一致是 G1 硬门**：主体沿 +z 行走 ⇒ 侧向符号 = `sign(eye_x − aim_x)` = `sign(sin(az))`。
#: 所以全表**一律取负**（相机恒在轴线 −x 侧），**绝不出现符号交替** ⇒ 全片不越轴、K3 屏幕方向不翻转。
#: （旧表的 `-62,-24,+14,+52,…` 在 ±90° 内但**跨了轴线两侧**，属真实越轴剪辑错误。）
#: 同时保持相邻差 ≥38°（K4 30° 规则留 8° 余量），且各机位离轴线 ≥16°（留镜内摆动余量）。
_AZIMUTHS = (-16.0, -58.0, -20.0, -62.0, -18.0, -60.0,
             -22.0, -64.0, -24.0, -66.0, -26.0, -68.0)


def camera_side(eye, aim, axis) -> int:
    """**K1 同款**侧向符号：`sign( ((aim−eye) × axis) · up )`，up = +y（Y-up 场景）。

    与 `gates.check_k1` 逐字同式（含 `|cross_y| ≤ 1e-9 ⇒ +1` 的兜底），
    保证「我方自证」与「Lead 的判据」是同一把尺子，不各造一份。
    """
    v = [float(aim[k]) - float(eye[k]) for k in range(3)]
    ax = [float(axis[k]) for k in range(3)]
    cross_y = v[2] * ax[0] - v[0] * ax[2]
    side = cross_y if abs(cross_y) > 1e-9 else (1.0 if cross_y >= 0 else -1.0)
    return 1 if side > 0 else -1


def shot_side_report(shots, axis) -> dict:
    """全片逐镜**每个控制点**的侧向符号（比 K1 更严：K1 只看首帧，这里看整条轨迹）。

    返回 `{"signs": {shot_id: [sign,…]}, "all_same": bool, "flips": [[a,b],…]}`。
    """
    signs, flips = {}, []
    prev, prev_id = None, None
    for s in shots:
        ss = [camera_side(p["pos"], p["aim"], axis) for p in s["camera"]["path"]]
        signs[s["id"]] = ss
        cur = ss[0]
        if prev is not None and cur != prev:
            flips.append([prev_id, s["id"]])
        prev, prev_id = cur, s["id"]
    all_same = all(len(set(v)) == 1 for v in signs.values())
    uniq = {v[0] for v in signs.values()}
    return {"signs": signs, "all_same": all_same and len(uniq) <= 1, "flips": flips,
            "axis": [float(x) for x in axis]}


def camera_right(eye, aim) -> tuple:
    """该镜声明机位的**相机右向量**（Y-up：`up=(0,1,0)`），即 `normalize(forward × up)`。"""
    f = [float(aim[k]) - float(eye[k]) for k in range(3)]
    L = math.sqrt(sum(v * v for v in f))
    if L < 1e-12:
        raise TimelineError("camera_right：eye 与 aim 重合，朝向未定义")
    f = [v / L for v in f]
    r = (-f[2], 0.0, f[0])
    rl = math.sqrt(sum(v * v for v in r))
    if rl < 1e-12:
        raise TimelineError("camera_right：视线与 up 共线，右向量退化")
    return tuple(v / rl for v in r)


def shot_screen_direction_signs(shots, axis) -> dict:
    """**K3 的几何口径**（Lead 已采纳为 `gates.check_k3` 的裁决口径）：主体世界位移在该镜
    相机右向量上的投影符号。

    为什么必须换掉「1 帧剪影 bbox 位移」这个像素口径（实测读数；探针产物已删，读数留档于此）：

    · 每镜渲 12 帧测 `subject_bbox_screen` 中心 x，镜内是**振荡**而非趋势：
      `shot_08(static)` = [632,632,638,644,650,642,632,630,640,640,638,636]
      `shot_07(handheld)` = [374,366,362,370,378,368,356,360,368,356,348,358]
      ⇒ **峰峰值 20~24 px、周期 ≈10 帧（= 步态周期）**，而 12 帧**净漂移仅 ±2~8 px** ⇒ 信号 < 噪声。
    · 首帧 1 帧差符号实测 `+1,-1,+1,-1,+1,+1,-1,+1,+1,+1` ⇒ 6 对判红，与轴线侧别**无因果关系**。
    · 对照实验：把跟踪目标从「世界 AABB 中心」换成平滑的 `SceneBuilder.subject_center`（躯干），
      抖动**未消失**（峰峰值 24 px，符号还反了）⇒ 抖动源是**剪影包围盒本身**（摆臂/迈腿改变左右最外沿），
      不是跟踪目标。
    · 结构原因：`CameraRig._compose_aim` 把瞄准点绑到主体 ⇒ 相机**主动抵消主体的净屏幕位移**
      （为避免"主体走动出画"的刻意设计）。主体在屏幕上几乎不动，"屏幕运动方向"在像素里**已无信号**，
      故**调机位方位角修不动 K3**。

    本式：轴线吃**事实**（主体世界位移）、右向量吃**声明**（机位）——同侧机位下**恒同号**，
    不需要逐帧渲染回读，也不受步态抖动影响。实测（真实 scene 轴线 `[-0.0001,0.0361,4.2882]`）：
    `uniform=True, pairs=[], 逐镜 sign 全 +1`。

    返回 `{"signs": {shot_id: ±1}, "uniform": bool, "pairs": [[a,b],…]}`（`pairs` = 判红的相邻对）。
    """
    ax = [float(axis[k]) for k in range(3)]
    signs, pairs = {}, []
    for s in shots:
        p = s["camera"]["path"][0]
        r = camera_right(p["pos"], p["aim"])
        proj = ax[0] * r[0] + ax[1] * r[1] + ax[2] * r[2]
        if abs(proj) < 1e-12:
            raise TimelineError("shot_screen_direction_signs：%s 的位移垂直于屏幕横轴，符号未定义"
                                % s["id"])
        signs[s["id"]] = 1 if proj > 0 else -1
    ids = [s["id"] for s in shots]
    for i in range(len(ids) - 1):
        if signs[ids[i]] != signs[ids[i + 1]]:
            pairs.append([ids[i], ids[i + 1]])
    return {"signs": signs, "uniform": not pairs, "pairs": pairs}


def _move_kind_for(bias: float, shot_index: int, prev: str = None) -> str:
    for hi, moves in _MOVE_TIERS:
        if bias < hi:
            idx = (shot_index + int(round(bias * 10))) % len(moves)
            mk = moves[idx]
            if prev is not None and mk == prev:      # 相邻镜头 move_kind 必须变化（K4/K5 友好）
                mk = moves[(idx + 1) % len(moves)]
            return mk
    return _MOVE_TIERS[-1][1][0]


#: 运动种类 → 机位/视线/焦距的一阶变化率（同镜头内方向自洽）
#: `height_rate` 为**绝对米数**（相对 dist 会在大远景里把机位抬到天上）；`pos_dx` 为 ×dist。
_MOVE_RATES = {
    "static": {}, "handheld": {},
    "orbit": {},                                   # az_sweep 在 _camera_path 里按侧别动态定（见下）
    "crane_up": {"height_rate": 0.90}, "crane_down": {"height_rate": -0.60},
    "crane": {"height_rate": 0.50},
    "dolly_in": {"dist_rate": -0.28, "focal_rate": 0.12},
    "dolly_out": {"dist_rate": 0.35, "focal_rate": -0.10},
    "dolly": {"dist_rate": -0.15},
    "truck_left": {"pos_dx": -0.30}, "truck_right": {"pos_dx": 0.30},
    "truck": {"pos_dx": -0.20},
    "pan_left": {"aim_dx": -1.0}, "pan_right": {"aim_dx": 1.0}, "pan": {"aim_dx": -0.7},
    "tilt_up": {"aim_dy": 0.5}, "tilt_down": {"aim_dy": -0.5}, "tilt": {"aim_dy": 0.35},
    "whip_pan": {"aim_dx": 2.4},
    "zoom_in": {"focal_rate": 0.35}, "zoom_out": {"focal_rate": -0.25},
    "zoom": {"focal_rate": 0.20},
    "dolly_zoom": {"dist_rate": -0.22, "focal_rate": -0.18},
}


def _camera_path(move_kind: str, az_deg: float, dist: float, height: float,
                 aim, focal: float):
    """按 move_kind 生成 ≥2 个控制点（t 严格递增且覆盖 [0,1]；Y-up 世界）。

    `orbit` 的摆动**朝轴线方向但绝不越过轴线**（停在离轴线 5° 处）：
    旧实现固定 `+35°`，当 `az` 靠近 0 时会把相机甩到轴线另一侧 ⇒ **镜内越轴**
    （K1 只看首帧抓不到，但渲染出来的 K3 屏幕方向会中途翻转）。
    """
    r = _MOVE_RATES.get(move_kind, {})
    sweep = r.get("az_sweep", 0.0)
    if move_kind == "orbit":
        toward = 1.0 if az_deg < 0 else -1.0        # 朝 0°（轴线）摆
        sweep = min(35.0, max(0.0, abs(az_deg) - 5.0)) * toward
    pts = []
    for t in (0.0, 0.5, 1.0):
        a = math.radians(az_deg + sweep * t)
        d = dist * (1.0 + r.get("dist_rate", 0.0) * t)
        h = height + r.get("height_rate", 0.0) * t
        px = aim[0] + d * math.sin(a) + dist * r.get("pos_dx", 0.0) * t
        pz = aim[2] + d * math.cos(a)
        ax = aim[0] + r.get("aim_dx", 0.0) * t
        ay = aim[1] + r.get("aim_dy", 0.0) * t
        f = focal * (1.0 + r.get("focal_rate", 0.0) * t)
        pts.append({"t": t,
                    "pos": [round(px, 4), round(h, 4), round(pz, 4)],
                    "aim": [round(ax, 4), round(ay, 4), round(aim[2], 4)],
                    "focal_35eq": round(min(400.0, max(8.0, f)), 4)})
    return pts


def _even_cuts(cands, n: int):
    """从候选 downbeat 帧里**均匀**取 n-1 个切点（n ≤ len(cands)+1）。"""
    m = len(cands)
    need = min(int(n) - 1, m)
    if need <= 0:
        return []
    if need >= m:
        return list(cands)
    if need == 1:
        return [cands[m // 2]]
    return [cands[int(round(j * (m - 1) / (need - 1)))] for j in range(need)]


def _snap_boundary(f: int, dbf, tol: int) -> int:
    """段落边界必须落在 downbeat 网格上（**E1 优先于 E3**）；否则 fail-closed。"""
    near = min(dbf, key=lambda d: (abs(d - f), d))
    if abs(near - f) > tol:
        raise TimelineError(
            "段落边界帧 %d 不在 downbeat 网格上（最近 %d，差 %d 帧 > 容差 %d）："
            "E1 是硬门，请让乐谱段落边界落在 downbeat 上" % (f, near, abs(near - f), tol))
    return near


def shot_scale_of(shot: dict) -> str:
    """从 `compose.coverage_target` 反查景别档位。

    `wmfilm.shot/1` **没有** `shot_scale` 字段，景别只能由 coverage_target 反查——
    供 K5（景别跳跃）等判据复用，避免各方各造一份映射。
    """
    shot = C.normalize("shot", shot)
    cov = float(shot["compose"]["coverage_target"])
    for k, v in _COVERAGE.items():
        if abs(v - cov) < 1e-9:
            return k
    raise TimelineError("coverage_target=%r 不对应任何景别档位（闭集：%s）"
                        % (cov, sorted(_COVERAGE.values())))


def plan_shot_table(brief: dict, score: dict, bind: dict) -> list:
    """全片镜头表（`wmfilm.shot/1` 列表）——**骨架优先**。

    1. 铺满 [0, total_frames]、首尾相接无洞；
    2. **≥ 8 镜**：段内镜头数 = round(段落小节数 / 能量档位目标小节数)，
       `move_bias ≥ 0.6` 的段落目标再减 1 小节（高 bias ⇒ 切得更碎）；
       仍不足 8 镜时，继续细分「ASL 最长且可再分」的镜头（细分点仍是 downbeat）；
    3. 所有切点 ∈ downbeat 帧（E1 硬门；段落边界不在网格上 ⇒ 直接抛，不静默偏移）；
    4. 相邻镜头 `move_kind` 与景别**都必须变化**（K4 30° / K5 景别跳跃友好；
       机位方位角表相邻差 ≥ 30° 且全在同一侧 ⇒ K1 轴线一致）。
    """
    score = C.normalize("score", score)
    bind = C.normalize("bind", bind)
    fps = _fps(brief)
    total = _total_frames(brief)
    secs = _sec_frames(score, fps)
    if abs(secs[-1]["end"] - total) > 1:
        raise TimelineError("score 段落末帧(%d) 与 brief 全片帧数(%d) 不一致"
                            % (secs[-1]["end"], total))
    bar = _bar_frames(score, fps)
    tol = int(bind["cut_rule"]["tolerance_frames"])
    dbf = _downbeat_frames(score, fps, total)
    if len(dbf) < 2:
        raise TimelineError("downbeats 不足以铺出多镜头骨架（得到 %d 个）" % len(dbf))
    # 段落边界吸附到 downbeat 网格；末段 end 固定为全片帧数（片尾不是切点）
    for i, sec in enumerate(secs):
        sec["start"] = _snap_boundary(sec["start"], dbf, tol)
        sec["end"] = _snap_boundary(sec["end"], dbf, tol) if i < len(secs) - 1 else total
    if secs[0]["start"] != 0:
        raise TimelineError("首段起点须为 0（得到 %d）" % secs[0]["start"])

    brief_sections = brief.get("sections", [])
    by_kind = {}
    for s in brief_sections:
        by_kind.setdefault(s["kind"], s)

    def params_for(si: int) -> dict:
        sec = secs[si]
        b = None
        if len(brief_sections) == len(secs):
            cand = brief_sections[si]
            if cand["kind"] == sec["kind"]:
                b = cand
        if b is None:
            b = by_kind.get(sec["kind"])
        if b is None:
            raise TimelineError("brief.sections 缺 %r 段落参数（缺料 ⇒ 不猜）" % sec["kind"])
        return {"shot_scale": b["shot_scale"], "move_bias": float(b["move_bias"]),
                "energy": sec["energy"], "kind": sec["kind"],
                "start": sec["start"], "end": sec["end"]}

    # ---- 段内细分（切点全部取自段内 downbeat）
    segs = []
    for si, sec in enumerate(secs):
        p = params_for(si)
        target_bars = max(1, int(math.ceil(
            _band_for_energy(p["energy"])["asl_frames"] / float(bar))))
        if p["move_bias"] >= 0.60:
            target_bars = max(1, target_bars - 1)
        span = sec["end"] - sec["start"]
        n_want = max(1, int(round((span / float(bar)) / target_bars)))
        cands = [f for f in dbf if sec["start"] < f < sec["end"]]
        n = max(1, min(n_want, len(cands) + 1))
        bounds = [sec["start"]] + _even_cuts(cands, n) + [sec["end"]]
        for k in range(len(bounds) - 1):
            segs.append({"start": bounds[k], "end": bounds[k + 1], "sec": si, "k": k})

    # ---- 骨架优先：不足 8 镜 ⇒ 继续细分最长的可再分镜头
    while len(segs) < 8:
        best, best_span = None, 0
        for i, s in enumerate(segs):
            if s["end"] - s["start"] > bar and s["end"] - s["start"] > best_span:
                best, best_span = i, s["end"] - s["start"]
        if best is None:
            break
        s = segs[best]
        mid = _even_cuts([f for f in dbf if s["start"] < f < s["end"]], 2)[0]
        segs[best:best + 1] = [{"start": s["start"], "end": mid, "sec": s["sec"], "k": 0},
                               {"start": mid, "end": s["end"], "sec": s["sec"], "k": 0}]
    if len(segs) < 8:
        raise TimelineError("镜头数 %d < 8（骨架优先：全片须 ≥ 8 镜；可用 downbeat 太少）"
                            % len(segs))
    cnt = {}
    for s in segs:
        s["k"] = cnt.get(s["sec"], 0)
        cnt[s["sec"]] = s["k"] + 1

    # ---- 逐镜落文档
    shots = []
    spec_ref = {"name": "mv_scene", "digest": _brief_digest(brief)}
    prev_move, prev_scale = None, None
    for i, s in enumerate(segs):
        p = params_for(s["sec"])
        base_idx = C.SHOT_SCALES.index(p["shot_scale"])
        d = 0 if s["k"] == 0 else (1 if s["k"] % 2 == 1 else -1)
        ci = min(len(C.SHOT_SCALES) - 1, max(0, base_idx + d))
        if prev_scale is not None and ci == prev_scale:       # 相邻景别必须跨级（K5）
            ci = next(k for k in sorted(range(len(C.SHOT_SCALES)),
                                        key=lambda k: (abs(k - ci), k)) if k != prev_scale)
        prev_scale = ci
        scale = C.SHOT_SCALES[ci]
        dist, height, focal = _SCALE_RIG[scale]
        az = _AZIMUTHS[i % len(_AZIMUTHS)]
        move_kind = _move_kind_for(p["move_bias"], i, prev_move)
        prev_move = move_kind
        aim = [0.0, 0.9, 0.0]
        focal = focal * (1.0 + 0.08 * ((i % 3) - 1))          # ±8% 抖动：不改档位、避免同构
        handheld = ({"amp_deg": 0.6, "hz": 6.0} if move_kind == "handheld"
                    else {"amp_deg": 0.0, "hz": 0.0})
        mood = _mood_for_energy(p["energy"])
        ctk = {"low_key": (3200, 5600), "neutral": (4300, 5200),
               "high_key": (5600, 6200)}[mood]
        light = {
            "key_fill_ratio": round(6.5 - 4.0 * p["energy"], 3),
            "mood": mood,
            "color_temp_k": {"key": float(ctk[0]), "fill": float(ctk[1])},
            "practicals": ([{"kind": "neon", "color_temp_k": 6500.0, "in_frame": True}]
                           if p["energy"] >= 0.50 else []),
            "atmosphere": {"fog_density": round(min(1.0, max(0.0, float(
                brief.get("style", {}).get("fog_density", 0.0)))), 4),
                "volumetric_gain": round(min(1.0, 0.10 + 0.50 * p["energy"]), 4)},
        }
        # 转场：默认 cut；段落交界且「进入副歌/桥段」的升能处用 dissolve（∈ TRANSITION_KINDS）
        trans = {"kind": "cut", "quantize_beats": 1, "tolerance_frames": tol}
        if s["end"] == p["end"] and s["end"] < total:
            nxt = secs[_section_at(secs, s["end"])]
            if (nxt["kind"] != p["kind"] and nxt["energy"] > p["energy"]
                    and nxt["kind"] in ("chorus", "bridge")):
                trans = {"kind": "dissolve", "quantize_beats": 1, "tolerance_frames": tol}
        shot = {
            "schema": "wmfilm.shot/1", "id": "shot_%02d" % (i + 1),
            "spec_ref": spec_ref,
            #: `wmfilm.shot/1` 的 frame_base 用**闭区间**（[start, end] 含两端）——
            #: 与 `wmmv.edit/1` 的"b.start == a.end + 1"同一口径。执行器内部仍用
            #: 半开区间 [start, end) 计算，只在落契约时就地 −1。
            "frame_base": {"fps": fps, "frame_start": s["start"], "frame_end": s["end"] - 1},
            "camera": {
                "path": _camera_path(move_kind, az, dist, height, aim, focal),
                "move_kind": move_kind,
                "easing": list(C.DEFAULT_EASING),
                "shutter_angle": 90.0 if p["energy"] > 0.72 else 180.0,
                "handheld": handheld,
            },
            "compose": {
                "aspect_ratio": C.ASPECT,
                "subject_anchor": _ANCHOR[scale],
                "headroom_ratio": 0.05 if scale in ("close", "extreme_close") else 0.08,
                "lookroom_ratio": 0.18,
                "horizon_v": round(0.30 + 0.02 * (ci % 4), 4),
                "depth_layers_min": 3,
                "coverage_target": _COVERAGE[scale],
            },
            "light": light,
            "transition_out": trans,
        }
        shots.append(C.normalize("shot", shot))
    return shots


# ---------------------------------------------------------------- edit

def build_edit(brief: dict, shots: list, score: dict, bind: dict) -> dict:
    """`wmmv.edit/1`：首尾相接的镜头表 + 每切点的 downbeat 对齐读数。

    · 内部切点（shot i 的 end，i < n-1）偏差 > tolerance ⇒ **抛**（E1 是硬门，不静默降级）；
    · 末镜头 end 是**片尾不是切点**，但仍如实记录 align（由 `cut_alignment_report` 复核）。
    """
    score = C.normalize("score", score)
    bind = C.normalize("bind", bind)
    fps = _fps(brief)
    tol = int(bind["cut_rule"]["tolerance_frames"])
    shot_docs = [C.normalize("shot", s) for s in shots]
    if not shot_docs:
        raise TimelineError("shots 为空")
    total_incl = shot_docs[-1]["frame_base"]["frame_end"]      # 闭区间末帧

    out_shots, alignment = [], []
    for i, s in enumerate(shot_docs):
        a = s["frame_base"]["frame_start"]
        b = s["frame_base"]["frame_end"]
        #: 闭区间：shot 的 frame_base 已是闭区间 ⇒ 直接沿用；切点是**下一镜首帧**
        #: （= 本镜 frame_end + 1），对齐读数按切点帧算。
        cut = b + 1
        e = {"id": s["id"],
             "shot_ref": {"name": s["id"], "digest": C.digest("shot", s)},
             "start_frame": a, "end_frame": b,
             "transition_in": {"kind": "cut"},
             "transition_out": dict(s["transition_out"])}
        align = _align_for(cut, score, fps)
        if i > 0 and out_shots[-1]["transition_out"]["kind"] != "cut":
            e["transition_in"] = dict(out_shots[-1]["transition_out"])
        if i < len(shot_docs) - 1 and abs(align["offset_frames"]) > tol:
            raise TimelineError(
                "切点 shot_%02d@%d 相对最近 downbeat 偏差 %d 帧 > 容差 %d（E1 硬门）"
                % (i + 1, cut, align["offset_frames"], tol))
        out_shots.append(e)
        if i < len(shot_docs) - 1:                 # 末镜 end 是片尾，不是切点
            alignment.append(align)

    doc = {
        "schema": "wmmv.edit/1", "id": "edit_main",
        "frame_base": {"fps": fps, "frame_start": 0, "frame_end": total_incl},
        "shots": out_shots, "alignment": alignment,
        "bind_ref": {"name": bind["id"], "digest": C.digest("bind", bind)},
        "score_ref": {"name": score["id"], "digest": C.digest("score", score)},
    }
    return C.normalize("edit", doc)


def cut_alignment_report(edit: dict, score: dict, tolerance_frames=None) -> dict:
    """E1 的**独立复算面**：每个切点 → 最近 downbeat 的帧偏差（不读 edit 里的 align，自己算）。"""
    edit = C.normalize("edit", edit)
    score = C.normalize("score", score)
    fps = int(round(float(edit["frame_base"]["fps"])))
    tol = (int(tolerance_frames) if tolerance_frames is not None
           else int(edit["shots"][0].get("transition_out", {}).get("tolerance_frames", 1)))
    cuts = []
    for i, s in enumerate(edit["shots"]):
        f = s["end_frame"]
        al = _align_for(f, score, fps)
        cuts.append({"shot_id": s["id"], "at_frame": f, "beat_s": al["beat_s"],
                     "offset_frames": al["offset_frames"],
                     "is_final": i == len(edit["shots"]) - 1,
                     "ok": abs(al["offset_frames"]) <= tol})
    bad = [c for c in cuts if not c["ok"]]
    return {"fps": fps, "tolerance_frames": tol, "cuts": cuts,
            "max_abs_offset": max(abs(c["offset_frames"]) for c in cuts),
            "violations": bad, "ok": not bad}


# ---------------------------------------------------------------- text

#: CJK 判定区段（含汉字/假名/谚文/全角）
_CJK_RANGES = ((0x3000, 0x303F), (0x3040, 0x30FF), (0x3400, 0x4DBF),
               (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0xAC00, 0xD7AF),
               (0xFF00, 0xFFEF))


def has_cjk(s: str) -> bool:
    return any(any(a <= ord(ch) <= b for a, b in _CJK_RANGES) for ch in s)


def _script_of(lines) -> str:
    return "cjk" if any(has_cjk(x) for x in lines) else "latin"


def _chunks(lines, n):
    return [list(lines[i:i + n]) for i in range(0, len(lines), n)]


def _distribute(needs, total_extra):
    """把富余帧轮转分给各条（确定性：余数给靠前的条）。"""
    alloc = list(needs)
    for k in range(int(total_extra)):
        alloc[k % len(alloc)] += 1
    return alloc


def _cues_in_span(sec_a: int, sec_b: int, lines, cap: float, fps: int):
    """把一段歌词铺进 [sec_a, sec_b]：按 ≤2 行/条分组，每条占满分到的帧；装不下 ⇒ 抛。

    为什么不做「超限就按行拆条」：一条 cue 的最小可用时长是 `ceil(字数*fps/cap)`，
    而按行拆后的**总需求** `Σceil(len_i*fps/cap) ≥ ceil(Σlen_i*fps/cap)`——段落整体装不下时，
    拆条只会更装不下。因此唯一诚实的行为是抛错（不删歌词、不偷改 cap、不静默截断）。
    """
    span = sec_b - sec_a
    if span <= 0:
        raise TimelineError("段落帧区间为空（%d..%d）" % (sec_a, sec_b))
    groups = _chunks(lines, 2)
    needs = [max(1, int(math.ceil(sum(len(l) for l in g) * fps / cap))) for g in groups]
    if sum(needs) > span:
        raise TimelineError(
            "歌词在 CPS≤%.1f 下需 %d 帧 > 段落可用 %d 帧（按段长装不下；不删歌词，请扩段或降字）"
            % (cap, sum(needs), span))
    alloc = _distribute(needs, span - sum(needs))
    cues, cur = [], sec_a
    for g, ga in zip(groups, alloc):
        if ga < max(1, int(math.ceil(sum(len(l) for l in g) * fps / cap))):
            raise TimelineError("分帧不足（内部不变量被破坏）")   # 自证：alloc ≥ needs
        cues.append((cur, cur + ga, g))
        cur += ga
    return cues


def build_text(brief: dict, edit: dict, score: dict) -> dict:
    """`wmmv.text/1`：brief.lyrics → 歌词 cues（按段落起止落帧，不删歌词、不超 CPS）。"""
    edit = C.normalize("edit", edit)
    score = C.normalize("score", score)
    fps = int(round(float(edit["frame_base"]["fps"])))
    secs = _sec_frames(score, fps)
    brief_sections = brief.get("sections", [])
    if len(brief_sections) != len(secs):
        raise TimelineError("brief.sections(%d) 与 score.sections(%d) 数量不一致（缺料 ⇒ 不猜）"
                            % (len(brief_sections), len(secs)))
    for i, (bs, ss) in enumerate(zip(brief_sections, secs)):
        if bs["kind"] != ss["kind"]:
            raise TimelineError("第 %d 段 kind 不一致：brief=%r score=%r"
                                % (i, bs["kind"], ss["kind"]))

    limits_cps = 20.0
    limits_chars = 42
    limits_lines = 2
    # 歌词 ↔ 段落按**出现次序**配对（同名段落可能重复：如 chorus 12-18 与 20-24）。
    # 按名字配对会把两条 chorus 歌词都挂到第一个 chorus 段上 ⇒ cue 重叠（契约硬拦）。
    by_occ = {}
    occ = {}
    for i, bs in enumerate(brief_sections):
        by_occ[(bs["kind"], occ.get(bs["kind"], 0))] = i
        occ[bs["kind"]] = occ.get(bs["kind"], 0) + 1
    seen = {}
    cues = []
    for entry in brief.get("lyrics", []):
        kind = entry["section"]
        n = seen.get(kind, 0)
        idx = by_occ.get((kind, n))
        if idx is None:
            raise TimelineError("歌词第 %d 次出现的段落 %r 在 brief.sections 里没有对应段落"
                                % (n + 1, kind))
        seen[kind] = n + 1
        ss = secs[idx]
        lines = [str(x) for x in entry["lines"]]
        if not lines:
            raise TimelineError("段落 %r 的歌词为空" % kind)
        for ln in lines:
            if len(ln) > limits_chars:
                raise TimelineError("歌词行 %d 字符 > 上限 %d：%r"
                                    % (len(ln), limits_chars, ln))
        cap = min(float(entry.get("cps_max", limits_cps)), limits_cps)
        for a, b, g in _cues_in_span(ss["start"], ss["end"], lines, cap, fps):
            #: `wmmv.text/1` 的 cue 也是**闭区间**（`en > st`，且 `b.start > a.end`
            #: —— 相邻 cue 不得共帧）⇒ 半开区间的 b 落契约时 −1。
            cues.append({"kind": "lyric", "start_frame": int(a), "end_frame": int(b) - 1,
                         "text": "\n".join(g), "lines": list(g),
                         "script": _script_of(g)})
    cues.sort(key=lambda c: (c["start_frame"], c["end_frame"]))

    doc = {
        "schema": "wmmv.text/1", "id": "text_main",
        "frame_base": dict(edit["frame_base"]),
        "aspect_ratio": C.ASPECT,
        "safe_area": {"title_safe": C.TITLE_SAFE, "action_safe": C.ACTION_SAFE},
        "limits": {"max_cps": limits_cps, "max_chars_per_line": limits_chars,
                   "max_lines": limits_lines},
        "cues": cues,
    }
    return C.normalize("text", doc)


# ---------------------------------------------------------------- SRT

def _tc(frame: int, fps: int) -> str:
    ms = int(round(frame * 1000.0 / fps))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return "%02d:%02d:%02d,%03d" % (h, m, s, ms)


def write_srt(text_doc: dict, path: str) -> str:
    """写 SRT（UTF-8、无 BOM、LF），返回文件 sha256(64)。"""
    text_doc = C.normalize("text", text_doc)
    fps = int(round(float(text_doc["frame_base"]["fps"])))
    blocks = []
    for i, c in enumerate(text_doc["cues"], 1):
        blocks.append("%d\n%s --> %s\n%s\n"
                      % (i, _tc(c["start_frame"], fps), _tc(c["end_frame"], fps),
                         "\n".join(c["lines"])))
    data = ("\n".join(blocks)).encode("utf-8")
    d = path.rsplit("/", 1)[0] if "/" in path else ""
    if d:
        import os
        os.makedirs(d, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------- overlay

#: 字幕默认样式（style 里的同名键可覆盖；也接受 style["subtitle"] / style["text"] 嵌套）
DEFAULT_TEXT_STYLE = {
    "text_color": (255, 255, 255, 255),
    "stroke_color": (0, 0, 0, 255),
    "stroke_width_ratio": 0.10,     # × 字号
    "bg_color": (0, 0, 0, 140),     # 半透明底（保证可读）
    "bg_pad_ratio": 0.30,           # × 字号
    "font_px_ratio": 0.052,         # × 画高
    "line_gap_ratio": 0.30,         # × 字号
    "bottom_margin_ratio": 0.03,    # × 画高（自 title safe 下沿内缩）
}

_LAST_OVERLAY_INFO = {"drawn": False, "reason": "not_called"}


def last_overlay_info() -> dict:
    """最近一次 `render_overlay` 的如实读数（缺字体/无 cue/出安全区都能看见）。"""
    return dict(_LAST_OVERLAY_INFO)


def _text_style(style) -> dict:
    st = dict(DEFAULT_TEXT_STYLE)
    style = style or {}
    for src in (style.get("subtitle") if isinstance(style, dict) else None,
                style.get("text") if isinstance(style, dict) else None,
                style):
        if isinstance(src, dict):
            for k in DEFAULT_TEXT_STYLE:
                if k in src and src[k] is not None:
                    st[k] = src[k]
    return st


def _cue_at(text_doc: dict, frame: int):
    for c in text_doc["cues"]:
        if c["start_frame"] <= frame <= c["end_frame"]:
            return c
    return None


def render_overlay(img, text_doc: dict, frame: int, style: dict = None):
    """把 `frame` 时刻的歌词烧进画面：底部居中 + 描边 + 半透明底，且落在 title safe 内。

    返回新图（不改入参）。缺字体/无 cue ⇒ 原图 + `last_overlay_info()` 里如实标注。
    """
    global _LAST_OVERLAY_INFO
    from PIL import Image, ImageDraw, ImageFont  # 渲染层依赖：函数内 import（缺则调用点报错）

    text_doc = C.normalize("text", text_doc)
    st = _text_style(style)
    W, H = img.size
    ts = float(text_doc["safe_area"]["title_safe"])
    sx0, sy0 = (1.0 - ts) / 2.0 * W, (1.0 - ts) / 2.0 * H
    sx1, sy1 = W - sx0, H - sy0
    info = {"frame": int(frame), "drawn": False, "reason": None, "script": None,
            "font_path": "", "font_px": None, "lines": [], "bbox": None,
            "safe_box": [sx0, sy0, sx1, sy1], "inside": None, "image_size": [W, H]}

    cue = _cue_at(text_doc, frame)
    if cue is None:
        info["reason"] = "no_active_cue"
        _LAST_OVERLAY_INFO = info
        return img.copy()
    info["script"] = cue["script"]
    info["lines"] = list(cue["lines"])
    fp = paths.font_path(cue["script"])
    info["font_path"] = fp
    if not fp:
        info["reason"] = "missing_font"   # 缺字体：不画，但如实标注
        _LAST_OVERLAY_INFO = info
        return img.copy()

    lines = list(cue["lines"])
    n = len(lines)
    bottom = sy1 - H * st["bottom_margin_ratio"]
    fs0 = max(8, int(round(H * st["font_px_ratio"])))
    placed = None
    for fs in range(fs0, 7, -1):
        font = ImageFont.truetype(fp, fs)
        sw = max(1, int(round(fs * st["stroke_width_ratio"])))
        probe = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
        boxes = [probe.textbbox((0, 0), ln, font=font, stroke_width=sw) for ln in lines]
        lh = max(b[3] - b[1] for b in boxes)
        gap = int(round(fs * st["line_gap_ratio"]))
        block_h = n * lh + (n - 1) * gap
        if block_h > (sy1 - sy0):
            continue
        top = bottom - block_h
        pos, xs0, ys0, xs1, ys1 = [], None, None, None, None
        ok = True
        for i, (ln, b) in enumerate(zip(lines, boxes)):
            w_ = b[2] - b[0]
            px = (W - w_) / 2.0 - b[0]
            py = top + i * (lh + gap) - b[1]
            pos.append((px, py))
            x0, y0, x1, y1 = px + b[0], py + b[1], px + b[2], py + b[3]
            xs0 = x0 if xs0 is None else min(xs0, x0)
            ys0 = y0 if ys0 is None else min(ys0, y0)
            xs1 = x1 if xs1 is None else max(xs1, x1)
            ys1 = y1 if ys1 is None else max(ys1, y1)
        if xs0 >= sx0 - 0.5 and xs1 <= sx1 + 0.5 and ys0 >= sy0 - 0.5 and ys1 <= sy1 + 0.5:
            placed = {"font": font, "sw": sw, "pos": pos, "boxes": boxes,
                      "bbox": [xs0, ys0, xs1, ys1]}
            break
        ok = False
    if placed is None:
        info["reason"] = "text_out_of_safe_area"   # 缩到最小仍越界 ⇒ 不画（不静默截断）
        _LAST_OVERLAY_INFO = info
        return img.copy()

    base = img.convert("RGBA") if img.mode != "RGBA" else img.copy()
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    bx0, by0, bx1, by1 = placed["bbox"]
    pad = placed["font"].size * st["bg_pad_ratio"]
    d.rectangle([max(sx0, bx0 - pad), max(sy0, by0 - pad),
                 min(sx1, bx1 + pad), min(sy1, by1 + pad)],
                fill=tuple(st["bg_color"]))
    for ln, (px, py) in zip(lines, placed["pos"]):
        d.text((px, py), ln, font=placed["font"], fill=tuple(st["text_color"]),
               stroke_width=placed["sw"], stroke_fill=tuple(st["stroke_color"]))
    out = Image.alpha_composite(base, layer)
    if img.mode != "RGBA":
        out = out.convert(img.mode)

    info.update({"drawn": True, "reason": "drawn", "font_px": placed["font"].size,
                 "bbox": [round(v, 2) for v in placed["bbox"]],
                 "inside": True})
    _LAST_OVERLAY_INFO = info
    return out
