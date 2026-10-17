# -*- coding: utf-8 -*-
"""acceptance · Lead 侧**冻结态验收**（复算，不采信转述）

    python3 -X utf8 -m mvpipe.acceptance [--out out] [--brief brief.json] [--sample 6]

七组检查（每组给"命令口径 / 原始读数 / 判定"）：
    A 产物齐备      —— 视频/音轨/字幕/清单/裁定/契约文档/帧数
    B 清单完整性    —— manifest 里每个 artifact 的 sha256 与字节数**逐件复算**
    C 视频自证      —— AVI 回读帧数 == manifest.frames == 帧文件数
    D **确定性**    —— 抽样帧**重新渲染**，与已存 PNG 逐像素比 sha256
    E 判据不变式    —— 无 REJECT；E1 偏差 ≤1 帧；ACCEPT 必带非空 measured；K2 如实 BLINDSPOT
    F 字幕校核      —— SRT 时间↔帧号换算、CPS ≤20、行宽 ≤42、条目数 ≥ 歌词条数
    G 契约复验      —— 五份契约文档重新过 `contracts.normalize`
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

import numpy as np
from PIL import Image

from . import contracts as C
from . import paths

paths.ensure_lingshu()


def _sha_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _pix_digest(img) -> str:
    return hashlib.sha256(np.asarray(img.convert("RGB"), dtype=np.uint8).tobytes()).hexdigest()


class Report:
    def __init__(self):
        self.groups = []

    def add(self, group, name, ok, detail):
        self.groups.append({"group": group, "check": name,
                            "status": "PASS" if ok else "FAIL", "detail": detail})
        return ok

    def tally(self):
        t = {}
        for g in self.groups:
            t[g["status"]] = t.get(g["status"], 0) + 1
        return t

    def ok(self):
        return all(g["status"] == "PASS" for g in self.groups)


def run(out: str, brief: dict, sample: int = 6) -> dict:
    from . import scene as scene_mod, render, timeline
    from .camera import CameraRig
    from . import encode

    r = Report()
    man = C.read_json(os.path.join(out, "manifest.json"))
    oracle = C.read_json(os.path.join(out, "oracle.json"))
    edit = C.read_json(os.path.join(out, "edit.json"))
    text = C.read_json(os.path.join(out, "text.json"))
    score = C.read_json(os.path.join(out, "score.json"))
    ex = man.get("extra") or {}
    size = tuple(ex.get("size") or (0, 0))
    fps = int(ex.get("fps") or 0)
    n_frames = int(ex.get("frames") or 0)

    # ---- A 产物齐备 ----
    frames_dir = os.path.join(out, "frames")
    frame_files = sorted(os.path.join(frames_dir, f) for f in os.listdir(frames_dir)) \
        if os.path.isdir(frames_dir) else []
    need = ["mv.avi", "music.wav", "mv.srt", "manifest.json", "oracle.json",
            "score.json", "bind.json", "edit.json", "text.json"]
    missing = [n for n in need if not os.path.isfile(os.path.join(out, n))]
    r.add("A", "产物齐备", not missing, "缺：%s" % missing if missing else
          "9 件齐备；帧 %d 张" % len(frame_files))
    r.add("A", "帧数==manifest.frames", len(frame_files) == n_frames,
          "帧文件 %d / manifest %d" % (len(frame_files), n_frames))

    # ---- B 清单完整性 ----
    bad = []
    files = man.get("files") or {}
    for name, rec in files.items():
        p = os.path.join(out, rec.get("path", ""))
        if not os.path.isfile(p):
            bad.append((name, "缺文件"))
            continue
        actual = _sha_file(p)
        if rec.get("sha256") != actual:
            bad.append((name, "sha256 不符"))
        if int(rec.get("bytes", -1)) != os.path.getsize(p):
            bad.append((name, "字节数不符"))
    r.add("B", "清单逐件复算", (not bad) and bool(files),
          "不符：%s" % bad if bad else "%d 件 sha256+字节数一致" % len(files))

    # ---- C 视频自证 ----
    try:
        rb = encode.read_mjpeg_avi(os.path.join(out, "mv.avi"))
        ok = (rb["frames"] == n_frames == len(frame_files)
              and tuple(rb["size"]) == size and int(rb["fps"]) == fps)
        r.add("C", "AVI 回读", ok, "回读 frames=%s size=%s fps=%s（manifest %d/%s/%d）"
              % (rb["frames"], rb["size"], rb["fps"], n_frames, size, fps))
    except Exception as e:                                            # noqa: BLE001
        r.add("C", "AVI 回读", False, "%s: %s" % (type(e).__name__, e))

    # ---- D 确定性（重渲抽样帧）----
    # ⚠ 必须**复现同一取景标定**：渲染用的相机是"标定后"的（瞄准绑定主体 + 距离标定），
    #   若这里用未标定的 `CameraRig(shot, size)`，相机回到声明的远机位 ⇒ 像素必然不同，
    #   会把"验收脚本没复现标定"误报成"产线不确定"（本脚本第一版就是这么错的）。
    from . import make as MK
    sb = scene_mod.SceneBuilder(brief, score)
    cats_d = MK._subject_cats(sb)
    style = render.make_style(brief)
    shots = {}
    for s in edit["shots"]:
        shots[s["id"]] = C.normalize("shot", C.read_json(os.path.join(out, "shots", "%s.json" % s["id"])))
    step = max(1, len(frame_files) // max(1, sample))
    picks = frame_files[::step][:sample]
    diffs = []
    for p in picks:
        base = os.path.basename(p)[:-4]
        sid, f = base.rsplit("_", 1)
        f = int(f)
        shot = shots[sid]
        _c = {}

        def gw(fr, _s=sid, _cc=_c):
            if fr not in _cc:
                _cc[fr] = sb.world_at(fr, _s)
            return _cc[fr]

        def trk(fr, _gw=gw, _ct=cats_d):
            b = MK._bbox_of(_gw(fr), _ct)
            return b if b else ((0.0, 0.9, 0.0), 1.7)

        rig = CameraRig(shot, size, track=trk)
        scale, _cov, hf_t = MK._calibrate_rig(rig, gw, cats_d, size,
                                              rig.coverage_target, lambda *a: None, sid)
        rig.set_distance_scale(scale)
        rig.set_h_frac(hf_t)
        world = gw(f)
        world.mesh_shade = str(style.get("shade", "studio"))
        world.mesh_smooth = bool(style.get("smooth", True))
        world.mesh_cull = bool(style.get("cull", True))
        cam = rig.camera_at(f)
        img = render.render_frame(world, cam, f, shot, size, style)
        img = timeline.render_overlay(img, text, f, style)
        a = _pix_digest(img)
        b = _pix_digest(Image.open(p))
        if a != b:
            diffs.append((base, a[:12], b[:12]))
        _c.clear()
    r.add("D", "抽样帧重渲逐像素一致", not diffs,
          "抽样 %d 帧全部一致" % len(picks) if not diffs else "不一致：%s" % diffs)

    # ---- E 判据不变式 ----
    checks = {c["name"]: c for c in oracle["checks"]}
    rej = [c["name"] for c in oracle["checks"] if c["verdict"] == "REJECT"]
    r.add("E", "无 REJECT", not rej, "REJECT：%s" % rej if rej else "0 条 REJECT")
    bad_measured = [c["name"] for c in oracle["checks"]
                    if c["verdict"] in ("ACCEPT", "REJECT") and not c["measured"]]
    r.add("E", "ACCEPT/REJECT 必带 measured", not bad_measured, "违规：%s" % bad_measured)
    e1 = checks.get("E1") or {}
    offs = [abs(m.get("offset_frames", 99)) for m in (e1.get("measured") or [])]
    r.add("E", "E1 卡点偏差 ≤1 帧", bool(offs) and max(offs) <= 1.0 + 1e-9,
          "最大偏差 %.4f 帧（%d 个切点）" % (max(offs) if offs else -1, len(offs)))
    k2 = checks.get("K2") or {}
    r.add("E", "K2 如实 BLINDSPOT", k2.get("verdict") == "BLINDSPOT",
          "K2=%s（%s）" % (k2.get("verdict"), k2.get("note", "")[:40]))
    r.add("E", "判据条数 == 6（K1-K5+E1）", len(oracle["checks"]) == 6,
          "实际 %d 条：%s" % (len(oracle["checks"]), sorted(checks)))

    # ---- F 字幕校核 ----
    srt = open(os.path.join(out, "mv.srt"), encoding="utf-8").read()
    blocks = [b for b in re.split(r"\n\s*\n", srt.strip()) if b.strip()]
    parsed = []
    pat = re.compile(r"(\d\d):(\d\d):(\d\d),(\d\d\d) --> (\d\d):(\d\d):(\d\d),(\d\d\d)")
    for b in blocks:
        lines = b.strip().split("\n")
        m = pat.search(b)
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        t0 = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000.0
        t1 = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000.0
        body = " ".join(lines[2:]).strip()
        parsed.append((t0, t1, body))
    n_lyrics = len(brief.get("lyrics") or [])
    r.add("F", "SRT 条目数 ≥ 歌词条数", len(parsed) >= n_lyrics,
          "SRT %d 条 / brief 歌词 %d 条" % (len(parsed), n_lyrics))
    bad_cps, bad_len = [], []
    for t0, t1, body in parsed:
        dur = max(1e-6, t1 - t0)
        cps = len(body.replace(" ", "")) / dur if body else 0.0
        if cps > 20.0 + 1e-9:
            bad_cps.append((body[:10], round(cps, 2)))
        for ln in body.split("\n"):
            if len(ln) > 42:
                bad_len.append(ln[:20])
    r.add("F", "CPS ≤ 20", not bad_cps, "违规：%s" % bad_cps)
    r.add("F", "行宽 ≤ 42", not bad_len, "违规：%s" % bad_len)
    # 帧号↔时间换算一致性（抽第一条）
    if text["cues"]:
        c0 = text["cues"][0]
        want0 = c0["start_frame"] / float(fps)
        got0 = min(t0 for t0, _, _ in parsed) if parsed else -1
        r.add("F", "SRT 时间↔帧号换算", abs(want0 - got0) < 0.05,
              "首条帧 %d ⇒ 期望 %.3fs，SRT %.3fs" % (c0["start_frame"], want0, got0))

    # ---- G 契约复验 ----
    errs = []
    for kind, name in (("score", "score.json"), ("bind", "bind.json"),
                       ("edit", "edit.json"), ("text", "text.json"),
                       ("oracle", "oracle.json")):
        ok, why = C.validate(kind, C.read_json(os.path.join(out, name)))
        if not ok:
            errs.append((name, why))
    r.add("G", "五份契约复验通过", not errs, "违规：%s" % errs if errs else "5/5 通过")

    # ---- H 渲染开关有效性（**光滑着色对照门**）----
    # `mesh_smooth=True` 只在网格带 `MeshGeometry.normals` 时才生效；否则**静默退化**
    # 为平面着色（"开了等于没开"）。所以不能只看开关值，必须看**像素是否真的变了**。
    try:
        from . import make as MK
        from lingshu.world.mesh3d import MESH_LIBRARY
        have_n = sum(1 for g in MESH_LIBRARY.values() if getattr(g, "normals", None))
        r.add("H", "网格带顶点法线", have_n > 0,
              "%d/%d 件注册网格有 normals" % (have_n, len(MESH_LIBRARY)))
        sid0 = edit["shots"][0]["id"]
        f0 = edit["shots"][0]["start_frame"]
        shot0 = shots[sid0]
        _c = {}

        def gw0(f, _s=sid0, _c=_c):
            if f not in _c:
                _c[f] = sb.world_at(f, _s)
            return _c[f]

        cats0 = MK._subject_cats(sb)

        def trk0(f, _gw=gw0, _c=cats0):
            b = MK._bbox_of(_gw(f), _c)
            return b if b else ((0.0, 0.9, 0.0), 1.7)

        rig0 = CameraRig(shot0, size, track=trk0)
        MK._calibrate_rig(rig0, gw0, cats0, size, rig0.coverage_target,
                          lambda *a: None, sid0)
        cam0 = rig0.camera_at(f0)
        digests = {}
        for flag in (True, False):
            w = gw0(f0)
            w.mesh_shade = str(style.get("shade", "studio"))
            w.mesh_smooth = flag
            digests[flag] = _pix_digest(render.render_frame(w, cam0, f0, shot0, size, style))
        r.add("H", "光滑着色真的生效（开≠关）", digests[True] != digests[False],
              "smooth 开/关像素摘要 %s / %s" % (digests[True][:12], digests[False][:12]))
    except Exception as e:                                            # noqa: BLE001
        r.add("H", "光滑着色对照门", False, "%s: %s" % (type(e).__name__, e))

    return {"out": out, "checks": r.groups, "tally": r.tally(), "ok": r.ok()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="MV 产线冻结态验收")
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out", default=os.path.join(here, "out"))
    ap.add_argument("--brief", default=os.path.join(here, "brief.json"))
    ap.add_argument("--sample", type=int, default=6)
    a = ap.parse_args(argv)
    with open(a.brief, encoding="utf-8") as f:
        brief = json.load(f)
    res = run(a.out, brief, a.sample)
    for c in res["checks"]:
        print("[%s] %-28s %s  %s" % (c["status"], c["check"], c["group"], c["detail"]))
    print("TALLY", res["tally"], "OK" if res["ok"] else "FAILED")
    C.write_json(os.path.join(a.out, "acceptance.json"), res)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
