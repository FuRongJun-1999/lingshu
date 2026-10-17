# -*- coding: utf-8 -*-
"""MV 制作插件守卫 —— `wm_bridge.factory.film.mv`（P2–P6 执行器）

运行方式：
    python -X utf8 -m pytest tests/test_film_mv_plugin.py -v
    python -X utf8 tests/test_film_mv_plugin.py          # 无 pytest 时直跑

缺陷（本守卫防的是什么）：
    影视层 P0/P1 已在仓里（七契约 + K 族/E1 判据闸），但**没人能把它们变成一支片子**：
    乐谱从哪来、镜头表怎么铺、相机怎么对准主体、逐帧怎么渲、真值怎么回读、
    母版怎么编码 —— 每一步都成了"看文档自己写"的私人实现。私人实现有三个必然结果：
      · 判据口径各写一套（同一部片子两个分数）；
      · 契约校验各写一套（同一份 schema 两种判定）；
      · 坐标轴各写一套（Y-up 的世界喂给 Z-up 的判据 ⇒ 侧别/夹角全是错的）。

修法（本插件的三条硬约束）：
    1. **契约不复制**：`_det/contracts.py` 只做委托（`film/<kind>.py`），不实现校验；
    2. **判据不重写**：`_det/gates.py` 只装配输入束，裁决交给 `film/_det/gate_all`；
    3. **坐标显式桥**：`_det/gates.to_film_vec` 把 Y-up 世界绕 x 轴 +90° 搬到 P1 的 Z-up 口径，
       保持 `up` 映射、夹角与叉积符号（det=+1），并有往返/符号自证。

断言组（抽掉修复 ⇒ 哪组红）：
    [A] 插件登记与默认简报                   抽掉登记 ⇒ A 红
    [B] 契约委托（含**幂等**）               退回自写校验 ⇒ B 红
    [C] 坐标桥（往返 / up / 叉积符号）       改成随手换轴 ⇒ C 红
    [D] 输入束形状（P1 白名单）              多塞字段 ⇒ D 红
    [E] 判据可跑（含 DEFER 与 BLINDSPOT）    自己下结论 ⇒ E 红
    [F] 执行器小样本（编码/音频确定性）      引入随机/时间 ⇒ F 红
    [G] 取景与 up 轴不变量                   符号写反 / 轴传错 ⇒ G 红
"""
import importlib
import json
import math
import os
import sys

_HERE = os.path.abspath(os.path.dirname(__file__))
_PKG_ROOT = os.path.abspath(os.path.join(_HERE, "..", "bridges", "blender"))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

import pytest  # noqa: E402

from wm_bridge.factory.film import bind as BD                              # noqa: E402
from wm_bridge.factory.film import edit as ED                              # noqa: E402
from wm_bridge.factory.film import mv                                      # noqa: E402
from wm_bridge.factory.film import score as SC                             # noqa: E402
from wm_bridge.factory.film import shot as SH                              # noqa: E402
from wm_bridge.factory.film import text as TX                              # noqa: E402
from wm_bridge.factory.film.mv._det import contracts as MVC                # noqa: E402
from wm_bridge.factory.film.mv._det import gates as MVG                    # noqa: E402

_Dig = "a" * 64
_REF = {"name": "ref_01", "digest": _Dig}


def _need(mod):
    """导入依赖上游 lingshu 的执行器模块；没有上游 ⇒ skip（不假绿）。"""
    try:
        return importlib.import_module("wm_bridge.factory.film.mv._det." + mod)
    except Exception as exc:                                  # noqa: BLE001
        pytest.skip("需要上游 lingshu（WM_LINGSHU_ROOT）：%s: %s" % (type(exc).__name__, exc))


# ---------------------------------------------------------------- 夹具
def _shot(**kw):
    d = {"schema": "wmfilm.shot/1", "id": "shot_01", "spec_ref": dict(_REF),
         "frame_base": {"fps": 24, "frame_start": 0, "frame_end": 23},
         "camera": {"path": [
             {"t": 0.0, "pos": [0.0, 1.6, -3.0], "aim": [0.0, 0.9, 0.0],
              "focal_35eq": 45.0},
             {"t": 1.0, "pos": [1.5, 1.4, -1.4], "aim": [0.0, 0.9, 0.0],
              "focal_35eq": 60.0}],
             "move_kind": "dolly_in"},
         "compose": {"aspect_ratio": 1.778, "subject_anchor": "center",
                     "headroom_ratio": 0.08, "lookroom_ratio": 0.18,
                     "horizon_v": 0.5, "depth_layers_min": 3,
                     "coverage_target": 0.68},
         "light": {"key_fill_ratio": 4.0, "mood": "low_key",
                   "color_temp_k": {"key": 3200, "fill": 5600},
                   "practicals": [],
                   "atmosphere": {"fog_density": 0.0, "volumetric_gain": 0.0}},
         "transition_out": {"kind": "cut", "quantize_beats": 1,
                            "tolerance_frames": 1}}
    d.update(kw)
    return SH.normalize_shot(d)


def _score(**kw):
    d = {"schema": "wmmv.score/1", "id": "score_01",
         "source": {"path": "a.wav", "digest": "b" * 64, "duration_s": 2.0,
                    "sample_rate": 48000, "channels": 2},
         "tempo": {"bpm": 120.0, "meter": [4, 4], "offset_s": 0.0},
         "beats": [0.0, 0.5, 1.0, 1.5], "downbeats": [0.0, 1.0],
         "sections": [{"kind": "intro", "start_s": 0.0, "end_s": 1.0, "energy": 0.3},
                      {"kind": "chorus", "start_s": 1.0, "end_s": 2.0, "energy": 0.9}],
         "energy": {"hop_s": 0.5, "metric": "rms",
                    "values": [0.1, 0.4, 0.9, 0.7, 0.2]}}
    d.update(kw)
    return SC.normalize_score(d)


def _edit(**kw):
    d = {"schema": "wmmv.edit/1", "id": "edit_01",
         "bind_ref": dict(_REF), "score_ref": dict(_REF),
         "frame_base": {"fps": 24, "frame_start": 0, "frame_end": 47},
         "shots": [{"id": "shot_01", "shot_ref": dict(_REF),
                    "start_frame": 0, "end_frame": 23,
                    "transition_out": {"kind": "cut"}},
                   {"id": "shot_02", "shot_ref": dict(_REF),
                    "start_frame": 24, "end_frame": 47,
                    "transition_in": {"kind": "cut"}}],
         "alignment": [{"at_frame": 24, "beat_s": 1.0, "offset_frames": 0}]}
    d.update(kw)
    return ED.normalize_edit(d)


def _bundle(axis=True, truth=True):
    shots = [_shot(),
             _shot(id="shot_02",
                   frame_base={"fps": 24, "frame_start": 24, "frame_end": 47},
                   camera={"path": [
                       {"t": 0.0, "pos": [3.2, 1.7, -2.6], "aim": [0.0, 0.9, 0.0],
                        "focal_35eq": 70.0},
                       {"t": 1.0, "pos": [2.0, 1.5, -3.4], "aim": [0.0, 0.9, 0.0],
                        "focal_35eq": 50.0}],
                       "move_kind": "orbit"})]
    b = {"shots": shots, "edit": _edit(), "score": _score(),
         "frame_size": [1920, 1080]}
    if axis:
        #: 主体沿 **+x 侧向**行进（相机看 +z）⇒ 屏幕行进行向可判；
        #: 若沿视线方向走，横向分量为 0 ⇒ 按设计必须 BLINDSPOT（见 test_d4）。
        b["subject_centers"] = {"shot_01": [(0.0, 0.9, 0.0), (0.9, 0.9, 0.0)],
                                "shot_02": [(0.9, 0.9, 0.0), (1.8, 0.9, 0.0)]}
    if truth:
        b["truth"] = {}
        for sid, base in (("shot_01", 0), ("shot_02", 24)):
            b["truth"][sid] = [{"shot_id": sid, "frame": base + i,
                                "measure_size": [240, 135],
                                "coverage_actual": 0.30,
                                "subject_bbox_screen": [800, 200, 1100, 900],
                                "silhouette_delta_L": 0.4,
                                "histogram": {"p05": 0.1, "p50": 0.4, "p95": 0.8},
                                "color_temp_est_k": 4200,
                                "camera_actual": {"pos": [0.0, 1.6, -3.0],
                                                  "focal_35eq": 45.0, "fov_deg": 50.0}}
                               for i in range(4)]
    return b


# ══════════════════════════════════════════════════════ [A] 插件登记
def test_a1_plugin_registry():
    assert mv.PLUGIN_ID == "wmfilm.mv"
    assert mv.VERSION
    stages = {s[0] for s in mv.PIPELINE}
    assert {"P2", "P3", "P4", "P5", "P6"} <= stages, "P2–P6 必须都有阶段登记"
    contracts = {s[2] for s in mv.PIPELINE if s[2] != "-"}
    for c in ("wmmv.score/1", "wmmv.bind/1", "wmfilm.shot/1", "wmfilm.oracle/1"):
        assert c in contracts, "阶段登记漏了契约 %s" % c


def test_a2_default_brief_is_usable():
    with open(mv.DEFAULT_BRIEF, encoding="utf-8") as f:
        b = json.load(f)
    for k in ("title", "duration_s", "fps", "tempo", "sections", "lyrics", "style"):
        assert k in b, "默认简报缺 %s" % k
    secs = b["sections"]
    assert abs(secs[0]["start_s"]) < 1e-9
    assert abs(secs[-1]["end_s"] - float(b["duration_s"])) < 1e-9, "段落须严丝合缝覆盖全片"
    kinds = {s["kind"] for s in secs}
    assert kinds <= set(SC.SECTION_KINDS), "段落 kind 越出闭集：%s" % (kinds - set(SC.SECTION_KINDS))


# ══════════════════════════════════════════════════════ [B] 契约委托
def test_b1_normalize_delegates_to_p0():
    d = _score()
    assert MVC.normalize("score", d) == SC.normalize_score(d)
    for kind, fn in (("shot", SH.normalize_shot), ("bind", BD.normalize_bind),
                     ("edit", ED.normalize_edit), ("text", TX.normalize_text)):
        assert kind in MVC.KINDS
        assert callable(fn)


def test_b2_digest_is_full_sha256():
    d = SH.normalize_shot(_shot())
    dg = MVC.digest("shot", d)
    assert len(dg) == 64 and all(c in "0123456789abcdef" for c in dg), \
        "P0 的 _digest_str 规定：形如十六进制就必须是 64 位 sha256"
    assert MVC.digest("shot", d) == MVC.digest("shot", dict(reversed(list(d.items())))), \
        "摘要必须只依赖内容（键序无关）"


def test_b3_unknown_kind_raises():
    with pytest.raises(MVC.ContractError):
        MVC.normalize("perform", {"schema": "wmfilm.perform/1"})
    ok, why = MVC.validate("nope", {})
    assert not ok and "未知契约种类" in why


def test_b4_normalize_is_idempotent():
    """**规范化必须幂等** —— 修的是 P0 的一个真缺陷。

    `film/score.py` 原先把 `tempo.meter` 落成 **tuple**，而校验器 `req_list` 只收 list ⇒
    `normalize_score(normalize_score(x))` 第二次必抛。任何"trust but verify"的调用方
    （校验一份已规范化的文档）都会直接崩，且报错点是 `meter` 类型，极难联想到"规范化不幂等"。
    """
    for kind, doc in (("score", _score()), ("shot", _shot()), ("edit", _edit())):
        once = MVC.normalize(kind, doc)
        twice = MVC.normalize(kind, once)
        assert MVC.digest(kind, once) == MVC.digest(kind, twice), \
            "%s 的规范化不幂等（摘要漂移）" % kind


# ══════════════════════════════════════════════════════ [C] 坐标桥
def test_c1_roundtrip():
    for p in ([0.0, 0.0, 0.0], [1.5, 2.5, -3.5], [-7.0, 0.25, 12.0]):
        back = MVG.from_film_vec(MVG.to_film_vec(p))
        assert all(abs(back[i] - p[i]) < 1e-12 for i in range(3)), (p, back)


def test_c2_up_maps_to_p1_up():
    film_up = MVG.to_film_vec(MVG.UP_Y)
    assert all(abs(film_up[i] - MVG.UP_Z_P1[i]) < 1e-12 for i in range(3)), \
        "渲染世界的 +y 必须映射到 P1 的 +z（%r → %r）" % (MVG.UP_Y, MVG.UP_Z_P1)


def test_c3_preserves_angles_and_cross_product_sign():
    """旋转 det=+1 ⇒ 夹角与**叉积符号**都不变；侧别判据（K1）全靠后者。"""
    a, b = (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)
    fa, fb = MVG.to_film_vec(a), MVG.to_film_vec(b)
    dot_p = sum(a[i] * b[i] for i in range(3))
    dot_f = sum(fa[i] * fb[i] for i in range(3))
    assert abs(dot_p - dot_f) < 1e-12
    cr_p = (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
    cr_f = (fa[1] * fb[2] - fa[2] * fb[1], fa[2] * fb[0] - fa[0] * fb[2],
            fa[0] * fb[1] - fa[1] * fb[0])
    # 叉积本身也要被同一旋转搬过去（不是取负）
    mapped = MVG.to_film_vec(cr_p)
    assert all(abs(mapped[i] - cr_f[i]) < 1e-12 for i in range(3)), \
        "叉积符号被翻了 ⇒ 会把手性翻掉，K1 侧别结论反向"


# ══════════════════════════════════════════════════════ [D] 输入束形状
def test_d1_bundle_keys_are_p1_whitelisted():
    from wm_bridge.factory.film._det import scene as SCENE
    sc = MVG.build_film_bundle(_bundle())
    assert set(sc) <= set(SCENE.SCENE_KEYS), set(sc) - set(SCENE.SCENE_KEYS)
    for e in sc["truth"]:
        assert set(e) <= set(SCENE.TRUTH_KEYS), set(e) - set(SCENE.TRUTH_KEYS)
        assert set(e["measured"]) <= set(SCENE.MEASURED_KEYS), \
            "measured 越界（P1 会拒收）：%s" % (set(e["measured"]) - set(SCENE.MEASURED_KEYS))


def test_d2_axis_present_and_rotated():
    sc = MVG.build_film_bundle(_bundle())
    assert "axis" in sc, "轴线必须由主体真实位移给出（否则 K1 只能 BLINDSPOT）"
    a, b = sc["axis"]["a"], sc["axis"]["b"]
    assert all(abs(a[i] - MVG.to_film_vec([0.0, 0.9, 0.0])[i]) < 1e-9 for i in range(3))
    assert all(abs(b[i] - MVG.to_film_vec([1.8, 0.9, 0.0])[i]) < 1e-9 for i in range(3))


def test_d3_motion_sign_is_geometric_not_pixel():
    """`subject_motion_screen` 必须是**几何事实**（±1），不受屏幕抖动影响。"""
    sc = MVG.build_film_bundle(_bundle())
    vals = [e["measured"].get("subject_motion_screen") for e in sc["truth"]]
    assert vals and all(v in (1.0, -1.0) for v in vals), vals
    assert len(set(vals)) == 1, "同一轴线方向下符号必须一致（侧向行进）"


def test_d4_motion_sign_none_when_along_view_axis():
    """主体**沿视线方向**行进 ⇒ 横向分量为 0 ⇒ 符号必须缺省（判据才会 BLINDSPOT）。

    返回 0 或硬凑一个符号都会让 K3 静默通过 —— 这正是"缺料不得变 ACCEPT"。
    """
    b = _bundle()
    b["subject_centers"] = {"shot_01": [(0.0, 0.9, 0.0), (0.0, 0.9, 1.5)],
                            "shot_02": [(0.0, 0.9, 1.5), (0.0, 0.9, 3.0)]}
    sc = MVG.build_film_bundle(b)
    #: 只有**机位在轴线上**的镜头才真退化（shot_01：pos.x == aim.x == 0）；
    #: shot_02 机位偏轴（x=3.2）⇒ 主体沿 +z 走仍有横向分量，符号是合法的。
    deg = [e["measured"].get("subject_motion_screen")
           for e in sc["truth"] if e["shot_id"] == "shot_01"]
    assert deg and all(v is None for v in deg), deg
    ok = [e["measured"].get("subject_motion_screen")
          for e in sc["truth"] if e["shot_id"] == "shot_02"]
    assert ok and all(v in (1.0, -1.0) for v in ok), ok


# ══════════════════════════════════════════════════════ [E] 判据
def test_e1_run_all_includes_defer_registry():
    checks = MVG.run_all(_bundle())
    names = {c["name"] for c in checks}
    for n in ("K1", "K2", "K3", "K4", "K5", "E1"):
        assert n in names, "P1 已实现的判据必须都出现：缺 %s" % n
    assert sum(1 for c in checks if c["verdict"] == "DEFER") > 0, \
        "P1 登记的未实现判据族（C/M/A/P）必须显式 DEFER，不能凭空消失"


def test_e2_clean_bundle_accepts_k1_k4_e1():
    checks = {c["name"]: c for c in MVG.run_all(_bundle())}
    assert checks["K1"]["verdict"] == "ACCEPT", checks["K1"]
    assert checks["K4"]["verdict"] == "ACCEPT", checks["K4"]
    assert checks["E1"]["verdict"] == "ACCEPT", checks["E1"]


def test_e3_k1_blindspot_without_axis():
    checks = {c["name"]: c for c in MVG.run_all(_bundle(axis=False))}
    assert checks["K1"]["verdict"] == "BLINDSPOT", "缺轴线必须 BLINDSPOT，不得猜"
    assert checks["K1"]["measured"] == []


def test_e4_k2_blindspot_without_gaze():
    checks = {c["name"]: c for c in MVG.run_all(_bundle())}
    assert checks["K2"]["verdict"] == "BLINDSPOT", "未回读视线就必须 BLINDSPOT（不猜）"


def test_e5_thresholds_carry_sources():
    ths = MVG.all_thresholds()
    assert ths and all(t["source"].strip() for t in ths), "阈值必须带来源（否则'凭什么定这个数'无据）"


# ══════════════════════════════════════════════════════ [F] 执行器小样本
def test_f1_avi_roundtrip_and_determinism(tmp_path):
    PIL = pytest.importorskip("PIL.Image")
    enc = _need("encode")
    frames = []
    for i in range(4):
        p = str(tmp_path / ("f%d.png" % i))
        PIL.new("RGB", (32, 18), (10 * i, 40, 80)).save(p)
        frames.append(p)
    a = str(tmp_path / "a.avi")
    b = str(tmp_path / "b.avi")
    enc.write_mjpeg_avi(frames, a, 24, (32, 18))
    enc.write_mjpeg_avi(frames, b, 24, (32, 18))
    assert enc.sha256_file(a) == enc.sha256_file(b), "同一帧列表两次写盘必须逐字节相同"
    rb = enc.read_mjpeg_avi(a)
    assert rb["frames"] == 4 and tuple(rb["size"]) == (32, 18) and int(rb["fps"]) == 24


def test_f2_audio_envelope_length_and_determinism(tmp_path):
    audio = _need("audio")
    brief = {"duration_s": 2.0, "fps": 24, "tempo": {"bpm": 120.0, "meter": [4, 4]},
             "sections": [{"kind": "intro", "start_s": 0.0, "end_s": 1.0, "energy": 0.3},
                          {"kind": "chorus", "start_s": 1.0, "end_s": 2.0, "energy": 0.9}]}
    a = audio.make(brief, str(tmp_path / "a"))
    b = audio.make(brief, str(tmp_path / "b"))
    assert a["wav"]["sha256"] == b["wav"]["sha256"], "音频必须逐字节确定"
    s = SC.normalize_score(a["score"])
    hop = s["energy"]["hop_s"]
    want = int(round(float(s["source"]["duration_s"]) / hop)) + 1
    assert len(s["energy"]["values"]) == want, \
        "P0 要求包络长度由 hop_s 与时长唯一决定（含端点）：want %d got %d" % (
            want, len(s["energy"]["values"]))


# ══════════════════════════════════════════════════════ [G] 取景与 up 轴
def test_g1_hfrac_target_monotone_in_coverage():
    cam = _need("camera")
    rig = cam.CameraRig(_shot(), (960, 540))
    vals = [rig.framing_target_hfrac(1.8, 0.6) for _ in (0,)]
    assert 0.30 <= vals[0] <= 1.25
    r2 = cam.CameraRig(SH.normalize_shot(dict(_shot(), compose={
        "aspect_ratio": 1.778, "subject_anchor": "center", "headroom_ratio": 0.08,
        "lookroom_ratio": 0.18, "horizon_v": 0.5, "depth_layers_min": 3,
        "coverage_target": 0.40})), (960, 540))
    assert r2.framing_target_hfrac(1.8, 0.6) < vals[0], \
        "coverage_target 变小 ⇒ 目标高度占比必须变小（否则标定方向反了）"


def test_g2_aim_offset_sign_for_headroom():
    """`headroom_ratio` 的符号：主体占比 < 1 时瞄准点必须在主体中心**之下**。

    写反的后果实测过：8% 的 headroom 变成 **31%**（画面里主体整体下沉、头顶一大片空）。
    """
    cam = _need("camera")
    rig = cam.CameraRig(_shot(), (960, 540))
    assert rig.aim_offset(1.8, 0.60) < 0.0, "t=0.60 时瞄准点应低于主体中心"
    assert rig.aim_offset(1.8, 1.20) > 0.0, "t>1（特写）时瞄准点应高于主体中心"


def test_g3_up_axis_regression():
    """**回归门**：Y-up 世界必须 `up=(0,1,0)`；且门要用**离轴机位**才敏感。

    `from_look_at` 的默认 up=(0,0,1) 会让近水平视线的相机右向量变成 ≈ −y ⇒
    画面滚转约 90°。机位在 x=0 看向 x=0 平面时两种 up 等价（退化）⇒ 门会假绿。
    """
    cam = _need("camera")
    from lingshu.world.world3d import Camera3D
    off = SH.normalize_shot(dict(_shot(), camera={
        "path": [{"t": 0.0, "pos": [3.2, 1.7, -2.6], "aim": [0.0, 0.9, 0.0],
                  "focal_35eq": 45.0},
                 {"t": 1.0, "pos": [3.2, 1.7, -2.6], "aim": [0.0, 0.9, 0.0],
                  "focal_35eq": 45.0}],
        "move_kind": "static"}))
    rig = cam.CameraRig(off, (320, 180), track=lambda f: ((0.0, 0.9, 0.0), 1.8))
    c = rig.camera_at(0)
    head = c.project((0.0, 1.8, 0.0), 320, 180)
    feet = c.project((0.0, 0.0, 0.0), 320, 180)
    assert head[1] < feet[1], "头顶必须在脚上方"
    assert abs(head[0] - feet[0]) < 5.0, "头脚应竖直对齐（横向错开 = 画面被滚转）"
    bad = Camera3D.from_look_at(eye=(3.2, 1.7, -2.6), target=(0.0, 0.9, 0.0),
                                up=(0.0, 0.0, 1.0), fov_deg=50.0)
    assert abs(bad.project((0.0, 1.8, 0.0), 320, 180)[0]
               - bad.project((0.0, 0.0, 0.0), 320, 180)[0]) > 5.0, \
        "注入错误 up 轴后本门必须变红（否则是假绿的门）"


if __name__ == "__main__":                                    # 无 pytest 时直跑
    import tempfile
    _tmp = tempfile.mkdtemp(prefix="mvplug_")

    class _TP:
        def __init__(self, root):
            self.root = root

        def __truediv__(self, name):
            return os.path.join(self.root, name)

    ok = fail = skip = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn(_TP(_tmp)) if fn.__code__.co_argcount else fn()
            ok += 1
            print("PASS", name)
        except BaseException as e:                            # noqa: BLE001
            if type(e).__name__ == "Skipped":
                skip += 1
                print("SKIP", name, e)
            else:
                fail += 1
                print("FAIL", name, type(e).__name__, e)
    print("TALLY ok=%d fail=%d skip=%d" % (ok, fail, skip))
    sys.exit(1 if fail else 0)
