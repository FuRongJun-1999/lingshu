# -*- coding: utf-8 -*-
"""make · 全片装配（架构 §七 P2–P6 的一条命令）

    python3 -X utf8 -m mvpipe.make [--brief brief.json] [--size 960x540]
                                  [--max-shots N] [--probe] [--no-gif] [--svg]

流程（单向、零 LLM）：
    brief ─▶ audio(乐谱+WAV) ─▶ timeline(绑定/镜头表/剪辑/文本)
          ─▶ scene(逐帧世界) ─▶ render(逐帧画面) ─▶ oracle(真值回读)
          ─▶ gates(K1–K5+E1) ─▶ encode(AVI/GIF/SRT/清单)
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time

from . import contracts as C
from . import paths

paths.ensure_lingshu()


def _sha256hex(obj) -> str:
    """内容摘要（完整 64 位）。P0 要求"形如十六进制须 64 位"，故不得截断。"""
    return hashlib.sha256(C.canonical(obj).encode("utf-8")).hexdigest()


#: worker（以及串行路径）的进程内状态：场景 / 主体类别 / 画幅 / ssaa / 风格
_W = {}


def _parse_size(s: str):
    a, b = s.lower().split("x")
    return (int(a), int(b))


def _subject_cats(sb) -> set:
    """主角的类别集合（兼容三种实现：`subject_categories()` 方法 / 属性 / `subject()`）。

    角色是**分部件**建模的（躯干/头/四肢各自一个 category）⇒ 仅取 `subject()` 会漏掉部件。
    故 `subject()` 返回主类别时，把**同前缀**的类别一并纳入（`walker` → `walker_torso` …）。
    """
    cats = set()
    if hasattr(sb, "subject_categories"):
        v = sb.subject_categories
        v = v() if callable(v) else v
        if v:
            cats = set(v)
    if not cats:
        cat, _eid = sb.subject()
        try:
            from lingshu.world.mesh3d import MESH_LIBRARY
            cats = {c for c in MESH_LIBRARY if c == cat or c.startswith(str(cat) + "_")}
        except ImportError:
            cats = set()
        cats.add(cat)
    return cats


def _aabb_of(world, cats):
    """主体部件的合并**世界 AABB** → (center, (sx, sy, sz))。无主体 ⇒ None。"""
    xs, ys, zs = [], [], []
    for o in world.objects:
        if getattr(o, "category", "") in cats:
            for p in o.corners():
                xs.append(p[0])
                ys.append(p[1])
                zs.append(p[2])
    if not ys:
        return None
    return (((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0,
             (min(zs) + max(zs)) / 2.0),
            (float(max(xs) - min(xs)), float(max(ys) - min(ys)), float(max(zs) - min(zs))))


def _bbox_of(world, cats):
    """该世界里主体部件的合并包围盒 → (center, height)。无主体 ⇒ None。"""
    a = _aabb_of(world, cats)
    if a is None:
        return None
    return a[0], a[1][1]


def _projected_height_frac(cam, center, ext, size):
    """主体世界 AABB 的 8 角投影 → 屏幕包围盒的**高度占比**（解析，无需渲染）。"""
    hx, hy, hz = ext[0] / 2.0, ext[1] / 2.0, ext[2] / 2.0
    ys = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                p = cam.project((center[0] + sx * hx, center[1] + sy * hy,
                                 center[2] + sz * hz), int(size[0]), int(size[1]))
                if p is not None:
                    ys.append(p[1])
    if not ys:
        return 0.0
    return (max(ys) - min(ys)) / float(size[1])


def _mask_stats(world, cam, cats, mw, mh):
    """在低分辨率上渲"只有主体"的世界 → (高度占比, 覆盖率, 屏幕包围盒像素)。

    这是**实测**口径：`_aabb_of` 的世界 AABB 会被走姿/摆臂撑大（8 角投影因此高估
    可见身高），所以最终标定必须以**剪影包围盒**为准。
    """
    import numpy as np
    from lingshu.world.world3d import World3D
    from .render import KEY_BG

    sub = World3D(camera=cam)
    sub.objects = [o for o in world.objects if getattr(o, "category", "") in cats]
    if not sub.objects:
        return 0.0, 0.0, None
    img = sub.render(mw, mh, camera=cam, background=KEY_BG, ground_color=KEY_BG, ssaa=1)
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    k = np.all(np.abs(arr - np.array(KEY_BG, dtype=np.float32)[None, None, :]) < 0.5, axis=2)
    mask = ~k
    n = int(mask.sum())
    if not n:
        return 0.0, 0.0, None
    ys, xs = np.nonzero(mask)
    bb = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return (bb[3] - bb[1]) / float(mh), n / float(mw * mh), bb


def _calibrate_rig(rig, get_world, cats, size, target, log=print, sid=""):
    """按 `compose.coverage_target` 标定镜头距离（架构 §3.4）。

    两步：
      ① **解析首猜**：`coverage_target` → 目标**高度占比** `t`
         （`camera.framing_target_hfrac`：面积占比 ≈ t²·F/a）；再用世界 AABB 的 8 角投影
         做不动点迭代（高度占比 ∝ 1/距离）。
      ② **实测精修**：在 1/8 画幅上渲"只有主体"的世界，用**剪影包围盒**的真实高度占比
         校正 3 次（世界 AABB 会被走姿撑大 ⇒ 只靠 ① 会把主体拍小 / 近景切头）。

    返回 (scale, 实测剪影覆盖率)；并把实测高度占比记进日志（可核对读数）。
    """
    f = rig.mid_frame()
    w = get_world(f)
    a = _aabb_of(w, cats)
    if a is None:
        return 1.0, 0.0
    center, ext = a
    t_target = rig.framing_target_hfrac(ext[1], max(ext[0], ext[2]))
    rig.set_h_frac(t_target)
    mw, mh = max(24, int(size[0]) // 8), max(16, int(size[1]) // 8)
    s = 1.0
    for _ in range(6):                                    # ① 解析
        rig.set_distance_scale(s)
        cam = rig.camera_at(f)
        got = _projected_height_frac(cam, center, ext, size)
        if got <= 1e-9:
            break
        s = min(60.0, max(1e-3, s * got / t_target))
    hf, cov = t_target, 0.0
    for _ in range(3):                                    # ② 实测精修
        rig.set_distance_scale(s)
        cam = rig.camera_at(f)
        hf, cov, _bb = _mask_stats(w, cam, cats, mw, mh)
        if hf <= 1e-6:
            break
        s = min(60.0, max(1e-3, s * hf / t_target))
    rig.set_distance_scale(s)
    cam = rig.camera_at(f)
    hf, cov, bb = _mask_stats(w, cam, cats, mw, mh)
    log("      取景标定 %s：scale=%.4f 高度占比 目标%.3f/实测%.3f 剪影覆盖率%.3f "
        "bbox=%s（coverage_target %.3f）"
        % (sid, s, t_target, hf, cov, bb, target))
    return s, cov, t_target


def _init_worker(brief, score, size, ssaa, style):
    """多进程 worker 初始化：**每个进程**建一次场景（网格注册是进程内全局）。"""
    from . import scene as scene_mod
    _W["sb"] = scene_mod.SceneBuilder(brief, score)
    _W["cats"] = _subject_cats(_W["sb"])
    _W["size"] = tuple(size)
    _W["ssaa"] = int(ssaa)
    _W["style"] = style


#: 并行分块粒度（帧）。按整镜头分派会有**长尾**：本片最大镜头 96 帧、单帧 ≈5.7 s
#: ⇒ 该镜头独占 9 分钟，12 进程也只能等它。按 24 帧切块后 24 块均匀铺满 12 进程。
CHUNK_FRAMES = 24


def _render_chunk(job):
    """渲染一个**帧块**（worker 侧）。标定已由主进程完成 ⇒ 这里只做渲染与回读。

    每帧是 `(shot, frame)` 的纯函数 ⇒ 并行不改变任何像素（已用 worker/主进程
    逐像素摘要对照自证）。
    """
    from . import render, oracle, timeline
    from .camera import CameraRig

    sid, shot, f0, f1, text, out, scale, h_frac = job
    sb = _W["sb"]
    cats = _W["cats"]
    size = _W["size"]
    style = _W["style"]
    ssaa = _W["ssaa"]

    def get_world(f, _sid=sid):
        w = sb.world_at(f, _sid)
        w.mesh_shade = str(style.get("shade", "studio"))
        w.mesh_smooth = bool(style.get("smooth", True))
        w.mesh_cull = bool(style.get("cull", True))
        return w

    rig = CameraRig(shot, size, track=lambda f: (_bbox_of(get_world(f), cats)
                                                 or ((0.0, 0.9, 0.0), 1.7)))
    rig.set_distance_scale(scale)
    rig.set_h_frac(h_frac)
    truth, centers, digests = [], [], []
    for f in range(f0, f1 + 1):            # 闭区间
        world = get_world(f)
        cam = rig.camera_at(f)
        img = render.render_frame(world, cam, f, shot, size, style, ssaa=ssaa)
        img = timeline.render_overlay(img, text, f, style)
        p = os.path.join(out, "frames", "%s_%05d.png" % (sid, f))
        img.save(p, "PNG", optimize=False)
        digests.append((f, render.frame_digest(img)))
        truth.append(oracle.readback(img, world, cam, f, sid, cats))
        if hasattr(sb, "subject_center"):
            centers.append(tuple(float(v) for v in sb.subject_center(f)))
    return {"sid": sid, "f0": f0, "f1": f1, "truth": truth, "centers": centers,
            "digests": digests}


def regates(brief: dict, out: str, size=None, log=print) -> dict:
    """**只重跑判据**（不重渲）：从已存帧 + 场景重算真值与裁定，落 `truth.json` / `oracle.json`。

    为什么需要：判据口径会演进（本产线就改过 K3 的度量），而**画面不因判据改变**。
    若判据一改就重渲全片（960×540 约 10 分钟），"读数可重跑"就成了空话。
    本函数把**证据（已存 PNG）**与**裁定（判据）**解耦：像素只读一次，判据可反复重跑。

    代价控制：只有 K5 吃像素（且只需**切点边界帧**：每镜首帧与末帧），
    K1/K3 吃几何（`subject_center`，无需渲染），K4/E1 吃声明。故本函数只对
    每镜首帧 + 末帧做回读（2×镜头数帧），秒级完成。
    """
    from . import scene as scene_mod, render, oracle, gates
    from .camera import CameraRig
    from PIL import Image

    size = tuple(size or (C.read_json(os.path.join(out, "manifest.json")).get("extra", {}).get("size") or (960, 540)))
    score = C.normalize("score", C.read_json(os.path.join(out, "score.json")))
    edit = C.normalize("edit", C.read_json(os.path.join(out, "edit.json")))
    shots = [C.normalize("shot", C.read_json(os.path.join(out, "shots", "%s.json" % e["id"])))
             for e in edit["shots"]]
    shot_by_id = {s["id"]: s for s in shots}
    sb = scene_mod.SceneBuilder(brief, score)
    cats = _subject_cats(sb)
    style = render.make_style(brief)

    truth, centers = {}, {}
    for e in edit["shots"]:
        sid = e["id"]
        shot = shot_by_id[sid]
        _c = {}

        def gw(f, _sid=sid, _cc=_c):
            if f not in _cc:
                w = sb.world_at(f, _sid)
                w.mesh_shade = str(style.get("shade", "studio"))
                w.mesh_smooth = bool(style.get("smooth", True))
                w.mesh_cull = bool(style.get("cull", True))
                _cc[f] = w
            return _cc[f]

        def trk(f, _gw=gw, _ct=cats):
            b = _bbox_of(_gw(f), _ct)
            return b if b else ((0.0, 0.9, 0.0), 1.7)

        rig = CameraRig(shot, size, track=trk)
        scale, _cov, hf_t = _calibrate_rig(rig, gw, cats, size,
                                           rig.coverage_target, lambda *a: None, sid)
        rig.set_distance_scale(scale)
        rig.set_h_frac(hf_t)
        truth[sid], centers[sid] = [], []
        # 判据所需：每镜**首帧**（K5 入点）与**末帧**（K5 出点）
        for f in sorted({e["start_frame"], e["end_frame"]}):   # 闭区间：首帧 / 末帧
            p = os.path.join(out, "frames", "%s_%05d.png" % (sid, f))
            if not os.path.isfile(p):
                raise FileNotFoundError("缺帧文件（无法重跑判据）：%s" % p)
            world = gw(f)
            cam = rig.camera_at(f)
            truth[sid].append(oracle.readback(Image.open(p), world, cam, f, sid, cats))
        for f in range(e["start_frame"], e["end_frame"] + 1):
            centers[sid].append(tuple(float(v) for v in sb.subject_center(f)))
        _c.clear()

    bundle = {"shots": shots, "edit": edit, "score": score, "truth": truth,
              "frame_size": list(size), "subject_centers": centers}
    oracle_doc = gates.build_oracle(bundle, "oracle_main",
                                    str(brief.get("title") or "mv"), _sha256hex(brief))
    checks = oracle_doc["checks"]
    C.write_json(os.path.join(out, "oracle.json"), oracle_doc)
    C.write_json(os.path.join(out, "truth.json"),
                 {"schema": "mvtruth/1", "note": "判据重跑时的边界帧回读（K5 入/出点）",
                  "frames": truth})
    tally = {}
    for c in checks:
        tally[c["verdict"]] = tally.get(c["verdict"], 0) + 1
    # ★ 同步刷新清单：`oracle.json` 内容变了 ⇒ 它的 sha256 与 `oracle_tally` 必须一起更新，
    #   否则清单**自相矛盾**（验收的"逐件复算"会红，而这与画面无关）。
    from . import encode
    try:
        man = C.read_json(os.path.join(out, "manifest.json"))
        artifacts = {}
        for name, rec in (man.get("files") or {}).items():
            p = os.path.join(out, rec["path"])
            if os.path.isfile(p):
                artifacts[name] = p
        artifacts["oracle"] = os.path.join(out, "oracle.json")
        artifacts["truth"] = os.path.join(out, "truth.json")
        extra = dict(man.get("extra") or {})
        extra["oracle_tally"] = tally
        extra["gates_rerun"] = True
        encode.write_manifest(out, artifacts, extra)
    except Exception as e:                                            # noqa: BLE001
        log("      [warn] 清单刷新失败（如实登记）：%s" % e)
    for c in checks:
        log("      %-3s %-10s %s" % (c["name"], c["verdict"], c.get("note", "")[:70]))
    log("      " + "  ".join("%s=%d" % (k, tally[k]) for k in sorted(tally)))
    return {"oracle": oracle_doc, "tally": tally}


def build_all(brief: dict, out: str, size=None, max_shots=None, do_gif=True,
              do_svg=False, ssaa=1, log=print, jobs=1, max_frames=None) -> dict:
    from . import audio, timeline, scene as scene_mod, render, oracle, gates, encode

    t0 = time.time()
    fps = int(brief.get("fps") or 24)
    size = tuple(size or brief.get("size") or (960, 540))
    os.makedirs(os.path.join(out, "frames"), exist_ok=True)

    # ---- ① 乐谱 + 音轨 ----
    log("[1/6] 乐谱与音轨 …")
    a = audio.make(brief, out)
    score = C.normalize("score", a["score"])
    C.write_json(os.path.join(out, "score.json"), score)
    log("      music.wav %s bytes  sha256=%s  时长 %.2fs  beats=%d downbeats=%d"
        % (a["wav"].get("bytes", "?"), str(a["wav"].get("sha256"))[:16],
           float(score["source"]["duration_s"]), len(score["beats"]),
           len(score["downbeats"])))

    # ---- ② 绑定 / 镜头表 / 剪辑 / 文本 ----
    log("[2/6] 绑定 · 镜头表 · 剪辑 · 文本 …")
    bind = C.normalize("bind", timeline.build_bind(brief, score))
    shots = [C.normalize("shot", s) for s in timeline.plan_shot_table(brief, score, bind)]
    edit = C.normalize("edit", timeline.build_edit(brief, shots, score, bind))
    text = C.normalize("text", timeline.build_text(brief, edit, score))
    C.write_json(os.path.join(out, "bind.json"), bind)
    C.write_json(os.path.join(out, "edit.json"), edit)
    C.write_json(os.path.join(out, "text.json"), text)
    for s in shots:
        C.write_json(os.path.join(out, "shots", "%s.json" % s["id"]), s)
    log("      shots=%d total_frames=%d cues=%d" % (len(shots), edit["frame_base"]["frame_end"] + 1,
                                                    len(text["cues"])))

    # ---- ③ 场景 ----
    log("[3/6] 场景构建 …")
    sb = scene_mod.SceneBuilder(brief, score)
    cats = _subject_cats(sb)
    shot_by_id = {s["id"]: s for s in shots}
    style = render.make_style(brief)

    # ---- ④ 逐帧渲染 + 真值回读 ----
    log("[4/6] 逐帧渲染与真值回读（%s，ssaa=%d，jobs=%d，块=%d 帧）…"
        % ("x".join(map(str, size)), ssaa, jobs, CHUNK_FRAMES))
    frames, truth, centers = [], {}, {}
    t_render = time.time()
    shot_sel = []
    for si, e in enumerate(edit["shots"]):
        if max_shots is not None and si >= max_shots:
            break
        shot_sel.append(e)

    # ① 主进程先**标定**每个镜头（掩码渲染不开光滑着色 ⇒ 每次毫秒级），
    #    这样 worker 不必重复标定，且分块可以跨镜头均匀铺满所有进程。
    calib = {}
    from .camera import CameraRig
    for e in shot_sel:
        sid = e["id"]
        shot = shot_by_id[sid]
        _c = {}

        def gw(f, _sid=sid, _cc=_c):
            if f not in _cc:
                _cc[f] = sb.world_at(f, _sid)
            return _cc[f]

        def trk(f, _gw=gw, _ct=cats):
            b = _bbox_of(_gw(f), _ct)
            return b if b else ((0.0, 0.9, 0.0), 1.7)

        rig = CameraRig(shot, size, track=trk)
        scale, cov, hf_t = _calibrate_rig(rig, gw, cats, size,
                                          rig.coverage_target, log, sid)
        calib[sid] = (scale, hf_t, cov)
        _c.clear()

    # ② 按 `CHUNK_FRAMES` 切块（避免"最大镜头独占一个进程"的长尾）
    jobs_list = []
    budget = max_frames
    for e in shot_sel:
        sid = e["id"]
        scale, hf_t, _cov = calib[sid]
        f = e["start_frame"]
        while f <= e["end_frame"]:
            g = min(e["end_frame"], f + CHUNK_FRAMES - 1)
            if budget is not None:
                if budget <= 0:
                    break
                g = min(g, f + budget)
                budget -= (g - f)
            jobs_list.append((sid, shot_by_id[sid], f, g, text, out, scale, hf_t))
            #: ★ 闭区间 ⇒ 下一块从 g+1 起。写成 `f = g` 会让最后一块（f == end）**原地死循环**
            #: 往 jobs_list 里无限追加任务 —— 实测 RSS 爬到 6.5 GB 后被 OOM 杀掉。
            f = g + 1
        if budget is not None and budget <= 0:
            break

    results = []
    if jobs > 1 and len(jobs_list) > 1:
        import multiprocessing as mp
        ctx = mp.get_context("fork")                 # Linux：fork 复用已加载模块
        with ctx.Pool(processes=min(jobs, len(jobs_list)),
                      initializer=_init_worker,
                      initargs=(brief, score, size, ssaa, style)) as pool:
            for i, r in enumerate(pool.imap_unordered(_render_chunk, jobs_list), 1):
                results.append(r)
                if i % 4 == 0 or i == len(jobs_list):
                    log("      块 [%d/%d] 完成（%s [%d,%d)）"
                        % (i, len(jobs_list), r["sid"], r["f0"], r["f1"]))
    else:
        _init_worker(brief, score, size, ssaa, style)
        for i, j in enumerate(jobs_list, 1):
            results.append(_render_chunk(j))
            log("      块 [%d/%d] 完成（%s [%d,%d)）" % (i, len(jobs_list), j[0], j[2], j[3]))

    # 主进程按**镜头顺序 + 帧号**重排（并行不得改变产物顺序与内容）
    order = {e["id"]: i for i, e in enumerate(edit["shots"])}
    results.sort(key=lambda r: (order[r["sid"]], r["f0"]))
    for r in results:
        sid = r["sid"]
        truth.setdefault(sid, []).extend(r["truth"])
        centers.setdefault(sid, []).extend(r["centers"])
        for f, _d in r["digests"]:
            frames.append(os.path.join(out, "frames", "%s_%05d.png" % (sid, f)))
    for e in shot_sel:
        sid = e["id"]
        scale, hf_t, cov = calib[sid]
        log("      %s 帧 [%d,%d) scale=%.4f 高度占比目标 %.3f 剪影覆盖率 %.3f"
            % (sid, e["start_frame"], e["end_frame"] + 1, scale, hf_t, cov))
    log("      渲染合计 %.1fs（%d 帧）" % (time.time() - t_render, len(frames)))

    # ---- ⑤ 判据闸 ----
    log("[5/6] 判据闸 K1–K5 + E1 …")
    bundle = {"shots": shots, "edit": edit, "score": score, "truth": truth,
              "frame_size": list(size), "subject_centers": centers}
    # 判据走 P1（`film/_det/gate_all`）：执行器只负责产出输入束，不自己下结论
    oracle_doc = gates.build_oracle(bundle, "oracle_main",
                                    str(brief.get("title") or "mv"), _sha256hex(brief))
    checks = oracle_doc["checks"]
    C.write_json(os.path.join(out, "oracle.json"), oracle_doc)
    tally = {}
    for c in checks:
        tally[c["verdict"]] = tally.get(c["verdict"], 0) + 1
    log("      " + "  ".join("%s=%d" % (k, tally[k]) for k in sorted(tally)))

    # ---- ⑥ 编码与清单 ----
    log("[6/6] 编码（AVI/GIF/SRT）与清单 …")
    artifacts = {}
    if frames:
        avi = encode.write_mjpeg_avi(frames, os.path.join(out, "mv.avi"), fps, size)
        artifacts["video"] = os.path.join(out, "mv.avi")
        log("      mv.avi %d bytes  %s" % (avi.get("bytes", -1), avi.get("readback", "")))
        if do_gif:
            #: ★ 用**抽帧预览**而不是把全片塞进 GIF：576 帧 @960×540 全存实测 **180 MB**
            #: （GIF 是 8 位调色板 + LZW，全片体积极差），而预览只需看"动起来对不对"。
            #: 抽帧 1/4、宽 640 ⇒ 约 5~8 MB，且与母版同一渲染源（不是另画一版）。
            try:
                from . import preview as _pv
                _g = _pv.make_preview(out, stride=4, width=640)
                artifacts["preview_gif"] = _g["path"]
                log("      mv_preview.gif %d bytes（抽帧 1/4 · 宽 640 · %d 帧）"
                    % (_g["bytes"], _g["frames"]))
            except Exception as _e:                                # noqa: BLE001
                log("      [warn] 预览 GIF 失败（如实登记，不影响母版）：%s" % _e)
    artifacts["audio"] = a["wav"]["path"]
    srt = os.path.join(out, "mv.srt")
    timeline.write_srt(text, srt)
    artifacts["subtitles"] = srt
    for name, path in (("score", "score.json"), ("bind", "bind.json"),
                       ("edit", "edit.json"), ("text", "text.json"),
                       ("oracle", "oracle.json")):
        artifacts[name] = os.path.join(out, path)
    # 可选 MP4（有 ffmpeg 才有；缺则**保留 AVI+WAV 路径**，绝不因此失败）
    if frames and hasattr(encode, "write_mp4"):
        try:
            mp4 = encode.write_mp4(frames, a["wav"]["path"], os.path.join(out, "mv.mp4"),
                                   fps, size)
            if mp4:
                artifacts["video_mp4"] = os.path.join(out, "mv.mp4")
                log("      mv.mp4 %d bytes（H.264+AAC，含音轨；逐字节不确定性已如实标注）"
                    % int(mp4.get("bytes", -1)))
            else:
                log("      [info] 未找到 ffmpeg ⇒ 跳过 MP4（交付形态保留 AVI + WAV）")
        except Exception as e:                                     # noqa: BLE001
            log("      [warn] MP4 封装失败（如实登记，不影响 AVI/WAV 主产物）：%s" % e)
    if do_svg and frames:
        try:
            sid = edit["shots"][0]["id"]
            w0 = sb.world_at(edit["shots"][0]["start_frame"], sid)
            cam0 = __import__("mvpipe.camera", fromlist=["CameraRig"]).CameraRig(
                shot_by_id[sid], size).camera_at(edit["shots"][0]["start_frame"])
            svg = render.render_frame_svg(w0, cam0, shot_by_id[sid], size, style)
            sp = os.path.join(out, "frame0000.svg")
            with open(sp, "w", encoding="utf-8") as f:
                f.write(svg)
            artifacts["vector_frame"] = sp
        except Exception as e:                                     # noqa: BLE001
            log("      [warn] SVG 出口失败（如实登记，不影响主产物）：%s" % e)

    manifest = encode.write_manifest(out, artifacts, {
        "title": brief.get("title"),
        "fps": fps, "size": list(size), "frames": len(frames),
        "duration_s": round(len(frames) / float(fps), 4),
        "shots": len([s for s in edit["shots"]]),
        "brief_sha256": _sha256hex(brief),
        "score_digest": C.digest("score", score),
        "edit_digest": C.digest("edit", edit),
        "text_digest": C.digest("text", text),
        "render_ssaa": int(ssaa),
        "cjk_font": bool(paths.font_path("cjk")),
        "oracle_tally": tally,
    })
    log("      清单 manifest.json  sha256=%s"
        % encode.sha256_file(os.path.join(out, "manifest.json"))[:16])
    log("完成：%d 帧 / %.1fs（总 %.1fs）" % (len(frames), time.time() - t_render,
                                            time.time() - t0))
    return {"frames": frames, "manifest": manifest, "oracle": oracle_doc,
            "edit": edit, "score": score, "text": text, "truth": truth,
            "artifacts": artifacts, "tally": tally}
