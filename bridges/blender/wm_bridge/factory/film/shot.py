# -*- coding: utf-8 -*-
"""wm_bridge.factory.film.shot · 镜头契约 wmfilm.shot/1（纯标准库 · 零 bpy）

★ 这一份契约要回答的缺口（G1）
    既有四产线能产出「一件东西怎么动」（anim）、「它长什么样」（geo / shade），
    但**没有任何一处声明「摄影机怎么动、画面怎么构、光怎么变」**——
    运镜 / 构图 / 布光此前只存在于人工调参与口头描述里。
    wmfilm.shot/1 把它们变成**可校验、可 diff、可复算**的数据。

★ 三条设计要点（架构文档 §3.4 的 1 / 2 / 3，逐条落地）
    1. spec_ref.digest 绑定物与姿态 —— 防止「镜头用的模型和渲染用的模型不是同一版」；
    2. frame_base 必须与 spec.scene 一致 —— 防止帧基漂移（命题 3）；
    3. camera.path 是**时间函数**不是单点 —— 这是 G1 缺口的正面回答。

★ 画幅（已裁 D4）
    16:9 横屏，compose.aspect_ratio = 1.778（= 16/9，容差 1e-3）。
    基准 1920x1080，可选 3840x2160（同 16:9）。竖屏 9:16 **不在本版范围**。

★ 本模块只立**契约与校验器**（§七 P0），不含渲染实现。
"""
from __future__ import annotations

from . import ASPECT_16_9
from . import _common as C

#: 契约标识（写进存档与所有下游产物）
SHOT_SCHEMA = "wmfilm.shot/1"

#: 运镜类型（闭集 —— 拒收未知值）
#: 运镜类型（闭集 —— 拒收未知值）。
#: 为什么把方向变体也列进闭集：影视语言里「推」与「拉」是**不同**的镜头
#: （推 = 收拢注意，拉 = 揭示环境），M 族判据要按运动类型判缓动与画内保持，
#: 方向不能靠猜。闭集**显式列全**（而非用正则推导）是为了
#: 「一眼看全 + 可变异自证」——正则会把非法值判成合法而不易被发现。
MOVE_KINDS = (
    "static", "orbit", "handheld", "dolly_zoom", "whip_pan",
    "pan", "pan_left", "pan_right",
    "tilt", "tilt_up", "tilt_down",
    "dolly", "dolly_in", "dolly_out",
    "truck", "truck_left", "truck_right",
    "crane", "crane_up", "crane_down",
    "zoom", "zoom_in", "zoom_out",
)

#: 主体锚位（闭集）
SUBJECT_ANCHORS = ("center", "left_third", "right_third", "upper_third", "lower_third")

#: 影调（闭集）
MOODS = ("high_key", "low_key", "neutral", "practical_driven")

#: 转场类型（闭集）
TRANSITION_KINDS = ("cut", "dissolve", "fade", "wipe", "match_cut",
                    "jump_cut", "smash_cut")

#: 实景光源类型（闭集）
PRACTICAL_KINDS = ("neon", "bulb", "screen", "fire", "candle", "window", "other")

#: 默认缓动 = ease-in-out 标准控制点（与 §3.4 示例同值）
DEFAULT_EASING = (0.42, 0.0, 0.58, 1.0)

#: 画幅容差（浮点比较；1.778 与 16/9 之差 < 1e-3）
ASPECT_TOL = 1e-3

#: 快门角上界（度）
SHUTTER_ANGLE_MAX = 360.0

__all__ = [
    "SHOT_SCHEMA", "MOVE_KINDS", "SUBJECT_ANCHORS", "MOODS", "TRANSITION_KINDS",
    "PRACTICAL_KINDS", "DEFAULT_EASING", "ASPECT_TOL", "SHUTTER_ANGLE_MAX",
    "ShotError", "normalize_shot", "validate_shot", "shot_digest", "shot_summary",
    "camera_at", "frame_count", "spec_ref_of", "aspect_is_16_9",
]

#: 各层的**声明键**白名单。违反即拒 —— 这是 fail-closed 的落点。
_KEYS_SHOT = ("schema", "id", "spec_ref", "frame_base", "camera", "compose",
              "light", "transition_out", "notes")
_KEYS_SPEC_REF = ("name", "digest")
_KEYS_FRAME_BASE = ("fps", "frame_start", "frame_end")
_KEYS_CAMERA = ("path", "move_kind", "easing", "shutter_angle", "handheld")
_KEYS_PATH_PT = ("t", "pos", "aim", "focal_35eq")
_KEYS_HANDHELD = ("amp_deg", "hz")
_KEYS_COMPOSE = ("aspect_ratio", "subject_anchor", "headroom_ratio", "lookroom_ratio",
                 "horizon_v", "depth_layers_min", "coverage_target")
_KEYS_LIGHT = ("key_fill_ratio", "mood", "color_temp_k", "practicals", "atmosphere")
_KEYS_COLOR_TEMP = ("key", "fill")
_KEYS_PRACTICAL = ("kind", "color_temp_k", "in_frame")
_KEYS_ATMOS = ("fog_density", "volumetric_gain")
_KEYS_TRANSITION = ("kind", "quantize_beats", "tolerance_frames")


class ShotError(C.ContractError):
    """镜头契约不合法（fail-closed：不猜、不补、不静默降级）。"""


# ─────────────────────────────────────────────────────────── 局部校验助手
def _is_hex(s):
    return bool(s) and all(ch in "0123456789abcdefABCDEF" for ch in s)


def _digest_str(d, where):
    """digest 须为非空字符串；**若形如十六进制则必须恰为 sha256（64 位）**。

    为什么这样判：既有产线用 geo.spec.spec_digest() 的 sha256（64 hex）；而契约文档
    的示例写作 <spec/1 的 sha256> 占位。两者都要能过，但**半截 hex 必须被拒**——
    否则「声称绑定了、其实绑错」会静默通过，而那正是本字段唯一要防的事。
    """
    v = C.req_str(d, "digest", where)
    if _is_hex(v) and len(v) != 64:
        raise ShotError(
            "%s.digest 形如十六进制但非 sha256（须 64 位），实为 %r" % (where, v))
    return v


def _opt_frac(d, key, default, where):
    """可选归一化比值：须落在 [0, 1]。"""
    return C.frac01(C.opt_num(d, key, default, where), "%s.%s" % (where, key))


def _pos_num(d, key, where):
    """必填正数。"""
    return C.positive(C.req_num(d, key, where), "%s.%s" % (where, key))


def _lerp(x, y, k):
    return float(x) + (float(y) - float(x)) * float(k)


# ─────────────────────────────────────────────────────────── 规范化（执行面）
def normalize_shot(shot):
    """校验并补全默认值；返回**新的**规范化 shot（不修改入参）。

    补全的默认值（每一项都是「不写时的确定含义」，不猜）：
      camera.move_kind          -> "static"
      camera.easing             -> (0.42, 0.0, 0.58, 1.0)  （ease-in-out 标准）
      camera.shutter_angle      -> 180.0
      camera.handheld           -> {"amp_deg": 0.0, "hz": 0.0}
      compose.aspect_ratio      -> 16/9（已裁 D4）
      compose.subject_anchor    -> "center"
      compose.horizon_v         -> 0.5
      compose.depth_layers_min  -> 3
      compose.coverage_target   -> 0.68（复用既有 hero 目标，不新造口径）
      light.key_fill_ratio      -> 4.0
      light.mood                -> "neutral"
      light.practicals          -> []
      light.atmosphere          -> {"fog_density": 0.0, "volumetric_gain": 0.0}
      transition_out            -> {"kind": "cut", "quantize_beats": 1,
                                    "tolerance_frames": 1}

    生效条件：合法 ⇒ 返回新 dict；不合法 ⇒ raise ShotError（消息含字段路径）。
    """
    if not isinstance(shot, dict):
        raise ShotError("shot 顶层须为对象，实为 %s" % type(shot).__name__)
    C.check_schema(shot, SHOT_SCHEMA, "shot")
    C.reject_unknown(shot, _KEYS_SHOT, "shot 顶层")

    out = {"schema": SHOT_SCHEMA}
    out["id"] = C.req_str(shot, "id", "shot")

    # ---- spec_ref：绑定物与姿态真源（设计要点 1）
    sr = C.req_map(shot, "spec_ref", "shot")
    C.reject_unknown(sr, _KEYS_SPEC_REF, "shot.spec_ref")
    out["spec_ref"] = {"name": C.req_str(sr, "name", "shot.spec_ref"),
                       "digest": _digest_str(sr, "shot.spec_ref")}

    # ---- frame_base：必须与 spec.scene 一致（设计要点 2）
    out["frame_base"] = _norm_frame_base(C.req_map(shot, "frame_base", "shot"))

    # ---- camera：时间函数，非单点（设计要点 3）
    out["camera"] = _norm_camera(C.req_map(shot, "camera", "shot"))

    out["compose"] = _norm_compose(C.req_map(shot, "compose", "shot"))
    out["light"] = _norm_light(C.req_map(shot, "light", "shot"))
    out["transition_out"] = _norm_transition(shot.get("transition_out"))

    if shot.get("notes") is not None:
        out["notes"] = C.req_str(shot, "notes", "shot")
    return out


def _norm_frame_base(fb):
    C.reject_unknown(fb, _KEYS_FRAME_BASE, "shot.frame_base")
    fps = C.positive(C.req_num(fb, "fps", "shot.frame_base"), "shot.frame_base.fps")
    fs = C.req_int(fb, "frame_start", "shot.frame_base")
    fe = C.req_int(fb, "frame_end", "shot.frame_base")
    if fe < fs:
        raise ShotError(
            "shot.frame_base 的 frame_end(%d) 不得早于 frame_start(%d)" % (fe, fs))
    return {"fps": fps, "frame_start": fs, "frame_end": fe}


def _norm_camera(cam):
    C.reject_unknown(cam, _KEYS_CAMERA, "shot.camera")

    pts_in = C.req_list(cam, "path", "shot.camera")
    if len(pts_in) < 2:
        raise ShotError(
            "shot.camera.path 至少 2 个关键点（它是**时间函数**不是单点），实为 %d 个"
            % len(pts_in))
    pts, prev_t = [], None
    for i, pt in enumerate(pts_in):
        w = "shot.camera.path[%d]" % i
        if not isinstance(pt, dict):
            raise ShotError("%s 须为对象，实为 %s" % (w, type(pt).__name__))
        C.reject_unknown(pt, _KEYS_PATH_PT, w)
        t = C.req_num(pt, "t", w)
        if not (0.0 <= t <= 1.0):
            raise ShotError("%s.t 须落在 [0, 1]（镜头内归一化进度），实为 %r" % (w, t))
        if prev_t is not None and not (t > prev_t):
            raise ShotError(
                "%s.t 须严格递增（时间函数不可回头），实为 %r 不大于前一点 %r"
                % (w, t, prev_t))
        prev_t = t
        pts.append({"t": t,
                    "pos": C.vec3(pt.get("pos"), w + ".pos"),
                    "aim": C.vec3(pt.get("aim"), w + ".aim"),
                    "focal_35eq": _pos_num(pt, "focal_35eq", w)})
    if pts[0]["t"] != 0.0:
        raise ShotError("shot.camera.path 首点的 t 必须为 0.0，实为 %r" % (pts[0]["t"],))
    if pts[-1]["t"] != 1.0:
        raise ShotError("shot.camera.path 末点的 t 必须为 1.0，实为 %r" % (pts[-1]["t"],))

    # #2120：`or` 把「显式 easing: []」吞成默认——「不写时」的默认语义被扩到
    # 「写了但为空」，既有 len != 4 报错分支对 0 永不可达；同仓 _common.opt_str
    # 把同款 or 吞 falsy 模式明文记为历史缺陷族。opt_list 无法区分「没写」与
    # 「显式 []」（都回 []），改按键存在性判默认；显式 [] 落入 len != 4 报错。
    if "easing" in cam and cam["easing"] is not None:
        eas = C.opt_list(cam, "easing", "shot.camera")
    else:
        eas = list(DEFAULT_EASING)
    if len(eas) != 4:
        raise ShotError("shot.camera.easing 须为 4 个控制点，实为 %d 个" % len(eas))
    for i, x in enumerate(eas):
        if not C.is_num(x):
            raise ShotError("shot.camera.easing[%d] 须为有限数值，实为 %r" % (i, x))
        if i in (0, 2) and not (0.0 <= float(x) <= 1.0):
            raise ShotError(
                "shot.camera.easing[%d] 是 Bézier 的 x 控制点，须落在 [0, 1]，实为 %r"
                % (i, x))

    sa = C.opt_num(cam, "shutter_angle", 180.0, "shot.camera")
    if not (0.0 < sa <= SHUTTER_ANGLE_MAX):
        raise ShotError("shot.camera.shutter_angle 须落在 (0, %g]，实为 %r"
                        % (SHUTTER_ANGLE_MAX, sa))

    hh_in = C.opt_map(cam, "handheld", "shot.camera")
    C.reject_unknown(hh_in, _KEYS_HANDHELD, "shot.camera.handheld")
    amp = C.opt_num(hh_in, "amp_deg", 0.0, "shot.camera.handheld")
    hz = C.opt_num(hh_in, "hz", 0.0, "shot.camera.handheld")
    if amp < 0.0 or hz < 0.0:
        raise ShotError("shot.camera.handheld 的 amp_deg / hz 须非负，实为 %r / %r"
                        % (amp, hz))

    return {"path": pts,
            "move_kind": C.opt_enum(cam, "move_kind", "static", "shot.camera",
                                    MOVE_KINDS),
            "easing": [float(x) for x in eas],
            "shutter_angle": sa,
            "handheld": {"amp_deg": amp, "hz": hz}}


def _norm_compose(cp):
    C.reject_unknown(cp, _KEYS_COMPOSE, "shot.compose")
    ar = C.opt_num(cp, "aspect_ratio", ASPECT_16_9, "shot.compose")
    if not (ar > 0.0):
        raise ShotError("shot.compose.aspect_ratio 须为正数，实为 %r" % (ar,))
    dl = C.opt_int(cp, "depth_layers_min", 3, "shot.compose")
    if dl < 1:
        raise ShotError("shot.compose.depth_layers_min 须 >= 1，实为 %r" % (dl,))
    return {"aspect_ratio": ar,
            "subject_anchor": C.opt_enum(cp, "subject_anchor", "center", "shot.compose",
                                         SUBJECT_ANCHORS),
            "headroom_ratio": _opt_frac(cp, "headroom_ratio", 0.0, "shot.compose"),
            "lookroom_ratio": _opt_frac(cp, "lookroom_ratio", 0.0, "shot.compose"),
            "horizon_v": _opt_frac(cp, "horizon_v", 0.5, "shot.compose"),
            "depth_layers_min": dl,
            "coverage_target": _opt_frac(cp, "coverage_target", 0.68, "shot.compose")}


def _norm_light(lt):
    C.reject_unknown(lt, _KEYS_LIGHT, "shot.light")
    kf = C.positive(C.opt_num(lt, "key_fill_ratio", 4.0, "shot.light"),
                    "shot.light.key_fill_ratio")

    ct_in = C.opt_map(lt, "color_temp_k", "shot.light")
    C.reject_unknown(ct_in, _KEYS_COLOR_TEMP, "shot.light.color_temp_k")
    ct = {}
    for k in ("key", "fill"):
        v = C.opt_num(ct_in, k, None, "shot.light.color_temp_k")
        if v is None:
            continue
        ct[k] = C.positive(v, "shot.light.color_temp_k.%s" % k)

    prs = []
    for i, p in enumerate(C.opt_list(lt, "practicals", "shot.light")):
        w = "shot.light.practicals[%d]" % i
        if not isinstance(p, dict):
            raise ShotError("%s 须为对象，实为 %s" % (w, type(p).__name__))
        C.reject_unknown(p, _KEYS_PRACTICAL, w)
        pk = C.enum(C.req_str(p, "kind", w), PRACTICAL_KINDS, w + ".kind")
        prs.append({"kind": pk,
                    "color_temp_k": C.positive(
                        C.opt_num(p, "color_temp_k", 6500.0, w), w + ".color_temp_k"),
                    "in_frame": C.opt_bool(p, "in_frame", True, w)})

    at_in = C.opt_map(lt, "atmosphere", "shot.light")
    C.reject_unknown(at_in, _KEYS_ATMOS, "shot.light.atmosphere")

    return {"key_fill_ratio": kf,
            "mood": C.opt_enum(lt, "mood", "neutral", "shot.light", MOODS),
            "color_temp_k": ct,
            "practicals": prs,
            "atmosphere": {
                "fog_density": _opt_frac(at_in, "fog_density", 0.0,
                                         "shot.light.atmosphere"),
                "volumetric_gain": C.nonneg(
                    C.opt_num(at_in, "volumetric_gain", 0.0,
                              "shot.light.atmosphere"),
                    "shot.light.atmosphere.volumetric_gain")}}


def _norm_transition(tr):
    if tr is None:
        return {"kind": "cut", "quantize_beats": 1, "tolerance_frames": 1}
    if not isinstance(tr, dict):
        raise ShotError("shot.transition_out 须为对象，实为 %s" % type(tr).__name__)
    C.reject_unknown(tr, _KEYS_TRANSITION, "shot.transition_out")
    qb = C.opt_int(tr, "quantize_beats", 1, "shot.transition_out")
    tf = C.opt_int(tr, "tolerance_frames", 1, "shot.transition_out")
    if qb < 1:
        raise ShotError("shot.transition_out.quantize_beats 须 >= 1，实为 %r" % (qb,))
    if tf < 0:
        raise ShotError("shot.transition_out.tolerance_frames 须 >= 0，实为 %r" % (tf,))
    return {"kind": C.opt_enum(tr, "kind", "cut", "shot.transition_out",
                               TRANSITION_KINDS),
            "quantize_beats": qb, "tolerance_frames": tf}


# ─────────────────────────────────────────────────────────── 查询面与摘要
def validate_shot(shot):
    """查询面：返回 (ok: bool, reason: str)。**不抛异常、不改入参。**"""
    return C.validate_with(normalize_shot, shot)


def shot_digest(shot):
    """镜头契约的内容摘要（sha256 hex）。只依赖内容，与路径 / mtime / 时刻无关。"""
    return C.digest_of(normalize_shot(shot))


def frame_count(shot):
    """本镜头的帧数（闭区间计数：frame_end - frame_start + 1）。"""
    fb = normalize_shot(shot)["frame_base"]
    return fb["frame_end"] - fb["frame_start"] + 1


def aspect_is_16_9(shot, tol=ASPECT_TOL):
    """画幅是否为本版裁定的 16:9（已裁 D4）。"""
    return abs(normalize_shot(shot)["compose"]["aspect_ratio"] - ASPECT_16_9) <= tol


def camera_at(shot, t):
    """取镜头内归一化进度 t ∈ [0, 1] 处的相机状态（**契约语义的参考实现**）。

    语义（本函数即该语义的定义；P3 的 CameraRig 必须与它逐点对拍）：
      · t 恰为某关键点的 t ⇒ 返回该点原值（不做插值误差）；
      · t 落在相邻两点之间 ⇒ pos / aim / focal_35eq 各分量**分段线性插值**；
      · 缓动**不在此处施加** —— easing 是「进度 → 时间」的重参数化，属于播放侧（P3）；
        此处只回答「进度 t 时相机在哪」，保持纯几何、可复算、无隐状态。

    越界 t ⇒ raise（**不夹取** —— 静默夹取会让「跑过头」看起来一切正常）。
    """
    n = normalize_shot(shot)
    if not C.is_num(t):
        raise ShotError("camera_at 的 t 须为有限数值，实为 %r" % (t,))
    t = float(t)
    if not (0.0 <= t <= 1.0):
        raise ShotError("camera_at 的 t 须落在 [0, 1]，实为 %r" % (t,))

    pts = n["camera"]["path"]
    for p in pts:
        if p["t"] == t:
            return {"t": t, "pos": p["pos"], "aim": p["aim"],
                    "focal_35eq": p["focal_35eq"]}
    for a, b in zip(pts, pts[1:]):
        if a["t"] < t < b["t"]:
            k = (t - a["t"]) / (b["t"] - a["t"])
            return {"t": t,
                    "pos": tuple(_lerp(a["pos"][i], b["pos"][i], k) for i in range(3)),
                    "aim": tuple(_lerp(a["aim"][i], b["aim"][i], k) for i in range(3)),
                    "focal_35eq": _lerp(a["focal_35eq"], b["focal_35eq"], k)}
    raise ShotError("camera_at 内部错误：t=%r 未落在任何区间" % (t,))


def shot_summary(shot):
    """人读摘要（用于报告与守卫断言，不参与哈希）。"""
    n = normalize_shot(shot)
    return {"schema": n["schema"], "id": n["id"],
            "spec": n["spec_ref"]["name"],
            "fps": n["frame_base"]["fps"], "frames": frame_count(n),
            "move_kind": n["camera"]["move_kind"],
            "keypoints": len(n["camera"]["path"]),
            "aspect_ratio": n["compose"]["aspect_ratio"],
            "is_16_9": aspect_is_16_9(n),
            "mood": n["light"]["mood"],
            "transition": n["transition_out"]["kind"],
            "digest": shot_digest(n)}


def spec_ref_of(spec, name=None):
    """由一份 wmfactory.spec/1 造出 spec_ref（digest 用 geo.spec.spec_digest 复算）。

    ★ 为什么用**函数内 import**：本模块的立身之本是「纯标准库、可独立自测」。
      这里依赖 geo.spec 只是为复用**同一个**摘要算法（不另造一套，否则「同一份 spec」
      会有两个 digest，正是本字段要防的漂移）。延迟导入让本模块在 geo 不可用时
      仍可单独校验结构。
    """
    from wm_bridge.factory.geo import spec as S
    nm = name if name is not None else S.normalize_spec(spec)["name"]
    return {"name": nm, "digest": S.spec_digest(spec)}
