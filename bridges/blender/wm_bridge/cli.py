# -*- coding: utf-8 -*-
"""wm_bridge.cli · 命令行入口（供 lingshu 侧进程外调用）

用法：
  python -m wm_bridge.cli probe                          # 通道探针（fail-closed，退出码 0/2）
  python -m wm_bridge.cli manifest ROOT OUT [REL ...]    # 建 SHA-256 清单
  python -m wm_bridge.cli verify ROOT MANIFEST [--no-strict]
  python -m wm_bridge.cli validate DOC.json              # 校验一份 oracle 导出
  python -m wm_bridge.cli validate-atta SCENE_DIR        # 校验 Atta deliver/v3 清单与文件

几何 / 部件 / 真值导出（**不需要 Blender**：吃 `bpy_dump` 抽出的 spec，或重出既有产物）：
  python -m wm_bridge.cli export-obj    --spec SPEC.json --out mesh.obj [--mtl mesh.mtl]
  python -m wm_bridge.cli export-shapes --spec SPEC.json --out shapes.json
  python -m wm_bridge.cli export-oracle --spec SPEC.json --out oracle.jsonl
  # 重出既有三件（往返自证 / 口径迁移；旧件必须显式声明帧）
  python -m wm_bridge.cli export-obj    --from-obj    mesh.obj    --out out.obj
  python -m wm_bridge.cli export-shapes --from-shapes shapes.json --out out.json --legacy-frame 0
  python -m wm_bridge.cli export-oracle --from-oracle oracle.jsonl --out out.jsonl --legacy-frame 0

退出码：0 = 通过；2 = 不通过 / 输入错误（fail-closed：不通就必须是 2，不许 0）。

为什么 `export-*` 不直接读 Blender：见 `bpy_dump.py` 的分工说明 —— Blender 只抽
**原始量**（世界顶点/局部 bbox/世界矩阵/逐 tick 位姿），轴换算与 schema 全在桥里，
所以这三件可以在没有 Blender 的机器上复跑与自测。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import manifest as M
from . import probe as P
from . import contract as C
from . import geom as G
from . import oracle as O
from . import atta_receipt as AR


def _emit(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="wm_bridge.cli", description="灵枢世界模型 × Blender 桥")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe", help="通道探针")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--config", default=None)

    m = sub.add_parser("manifest", help="建清单")
    m.add_argument("root")
    m.add_argument("out")
    m.add_argument("rels", nargs="*", default=[])

    v = sub.add_parser("verify", help="核验清单")
    v.add_argument("root")
    v.add_argument("manifest_path")
    v.add_argument("--no-strict", action="store_true")

    d = sub.add_parser("validate", help="校验 oracle 导出")
    d.add_argument("doc")

    va = sub.add_parser("validate-atta", help="校验 Atta deliver/v3 清单与文件")
    va.add_argument("scene_dir")
    va.add_argument("--manifest", default=None)
    va.add_argument("--allow-degraded", action="store_true")
    va.add_argument("--allow-partial", action="store_true",
                    help="接受 Atta status=partial 的交付（存在导出错误）——默认拒绝")

    eo = sub.add_parser("export-obj", help="导出 OBJ+MTL（1 单位=1 m / Z-up 原样 / 不归一化）")
    eo.add_argument("--spec", default=None, help="bpy_dump 抽出的 spec.json")
    eo.add_argument("--from-obj", default=None, help="重出既有 OBJ（往返自证）")
    eo.add_argument("--out", required=True)
    eo.add_argument("--mtl", default=None)

    es = sub.add_parser("export-shapes", help="导出 lingshu.shapes/1 部件描述")
    es.add_argument("--spec", default=None)
    es.add_argument("--from-shapes", default=None)
    es.add_argument("--out", required=True)
    es.add_argument("--legacy-frame", type=int, default=None,
                    help="旧件缺 export_frame 时**必须**显式声明其部件位姿取自哪一 tick")
    es.add_argument("--frame", type=int, default=None,
                    help="--legacy-frame 的别名（与 bpy_dump 的 --frame 同义）")

    eor = sub.add_parser("export-oracle", help="导出 wm_bridge.oracle/1 真值（写入前自校）")
    eor.add_argument("--spec", default=None)
    eor.add_argument("--from-oracle", default=None)
    eor.add_argument("--out", required=True)
    eor.add_argument("--legacy-frame", type=int, default=None)
    eor.add_argument("--frame", type=int, default=None)

    args = ap.parse_args(argv)

    if args.cmd == "probe":
        r = P.probe(host=args.host, config_path=args.config)
        _emit(r)
        return 0 if r["ok"] else 2

    if args.cmd == "manifest":
        try:
            man = M.write(args.root, args.rels, args.out, "wm_bridge.cli")
        except (OSError, FileNotFoundError) as exc:
            _emit({"ok": False, "error": str(exc)})
            return 2
        _emit({"ok": True, "out": args.out, "count": man["count"]})
        return 0

    if args.cmd == "verify":
        try:
            man = M.load(args.manifest_path)
        except (OSError, ValueError) as exc:
            _emit({"ok": False, "error": "清单读不了：%s" % exc})
            return 2
        ok, problems = M.verify(man, args.root, strict=not args.no_strict)
        _emit({"ok": ok, "count": len(problems),
               "problems": [{"kind": k, "rel": r, "detail": d} for k, r, d in problems[:50]]})
        return 0 if ok else 2

    if args.cmd == "validate":
        try:
            doc = json.load(open(args.doc, encoding="utf-8"))
        except (OSError, ValueError) as exc:
            _emit({"ok": False, "error": "文档读不了：%s" % exc})
            return 2
        ok, why = C.validate_doc(doc)
        _emit({"ok": ok, "reason": why,
               "tracks": len(doc.get("tracks") or []) if isinstance(doc, dict) else 0})
        return 0 if ok else 2

    if args.cmd == "validate-atta":
        try:
            report = AR.verify_atta_receipt(
                args.scene_dir, manifest_path=args.manifest,
                allow_degraded=args.allow_degraded,
                allow_partial=args.allow_partial)
        except (AR.AttaEvidenceError, OSError, ValueError) as exc:
            _emit({"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)})
            return 2
        _emit({"ok": True, "report": report})
        return 0

    # ---------------- 几何 / 部件 / 真值导出 ----------------
    if args.cmd in ("export-obj", "export-shapes", "export-oracle"):
        return _export(args)

    return 2


def _load_spec(path):
    """读 spec；坏件 raise（调用方统一转 rc=2）。"""
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    if not isinstance(spec, dict):
        raise G.GeomError("spec 顶层不是对象")
    if spec.get("schema") not in (None, "wm_bridge.dump/1"):
        raise G.GeomError("spec.schema 不是 wm_bridge.dump/1：%r" % (spec.get("schema"),))
    if not isinstance(spec.get("frame"), int) or isinstance(spec.get("frame"), bool):
        raise G.GeomError("spec 缺 frame（导出必须声明取哪一帧的位姿）")
    if not (spec.get("meshes") or spec.get("links")):
        raise G.GeomError("spec 既没有 meshes 也没有 links")
    return spec


def _crosscheck(spec, shapes_doc):
    """spec 内的 tracks × shapes 的位姿核对（有 tracks 就必须过）。"""
    if not (spec.get("tracks") and spec.get("links")):
        return {"ok": True, "checked": False,
                "note": "spec 无 tracks（--no-tracks 抽取）⇒ 跨件核对跳过"}
    doc = O.build_doc(_scene_from_spec(spec),
                      [_track_from_spec(t) for t in spec["tracks"]])
    ok, probs = G.check_pose_consistency(shapes_doc, doc, spec)
    cov = O.frame_coverage_problem(doc)
    if cov:
        probs = list(probs) + [{"link": "-", "kind": "frame_coverage", "detail": cov}]
        ok = False
    return {"ok": ok, "checked": True, "problems": list(probs)[:10],
            "n_problems": len(probs)}


def _scene_from_spec(spec):
    """spec → scene 头（**export_frame 一律用 spec.frame 覆盖**，避免旧 spec 缺帧）。"""
    sc = dict(spec.get("scene") or {})
    sc.setdefault("name", spec.get("name"))
    sc.setdefault("seed", 0)
    sc.setdefault("tick_hz", spec.get("tick_hz"))
    sc.setdefault("world_size", spec.get("world_size"))
    sc.setdefault("units", "m")
    sc["export_frame"] = spec.get("frame")
    return sc


def _track_from_spec(t):
    """spec 的 track（Blender 轴原始量）→ 契约记录（lingshu 轴）。"""
    pos = G.to_lingshu(t["pos"])
    q = G.mat3_to_quat(G.axis_rotate(t["rot3"]))
    return O.track_record(t["entity"], t["tick"], pos, rot_quat=q,
                          category=t.get("category") or "wmtest",
                          link=t.get("link"))


def _export(args):
    """三个 export-* 子命令的共用实现（spec 模式 / from-* 重出模式）。"""
    legacy = args.legacy_frame if getattr(args, "legacy_frame", None) is not None \
        else getattr(args, "frame", None)
    try:
        if args.cmd == "export-obj":
            if bool(args.spec) == bool(args.from_obj):
                raise G.GeomError("必须且只能给 --spec 或 --from-obj 之一")
            if args.spec:
                spec = _load_spec(args.spec)
                summary = G.write_obj(args.mtl or (args.out + ".mtl"), args.out,
                                      G.meshes_from_dump(spec),
                                      translate=tuple(spec.get("translate")
                                                      or (0.0, 0.0, 0.0)))
                _emit({"ok": True, "out": args.out, "mtl": args.mtl or (args.out + ".mtl"),
                       "mode": "spec", "summary": summary})
                return 0
            text = open(args.from_obj, encoding="utf-8").read()
            parsed = G.parse_obj(text)          # 内含 require_normalize（fail-closed）
            # 重出保留原件的 mtllib 声明（除非调用方显式换了 --mtl 名）
            mtl_ref = (os.path.basename(args.mtl) if args.mtl
                       else (parsed.get("mtllib") or "mesh.mtl"))
            new_text, mtl_rows, summary = G.format_obj(
                parsed["meshes"], mtl_name=mtl_ref)
            with open(args.out, "w", encoding="utf-8", newline="\n") as f:
                f.write(new_text)
            if args.mtl:
                with open(args.mtl, "w", encoding="utf-8", newline="\n") as f:
                    f.write("\n".join(mtl_rows) + "\n")
            same = _signed_zero_eq(text, new_text)
            mig = _migration_eq(text, new_text, None)
            _emit({"ok": bool(same), "out": args.out, "mode": "from-obj",
                   "summary": summary,
                   "roundtrip": {"byte_equal": same,
                                 "signed_zero_normalized": same,
                                 "after_frame_migration": mig}})
            return 0 if same else 2

        if args.cmd == "export-shapes":
            if bool(args.spec) == bool(args.from_shapes):
                raise G.GeomError("必须且只能给 --spec 或 --from-shapes 之一")
            if args.spec:
                spec = _load_spec(args.spec)
                summary = G.write_shapes(args.out, G.links_from_dump(spec),
                                         source=spec.get("source") or "blender/wmtest",
                                         export_frame=spec["frame"])
                doc = G.read_shapes(args.out)
                cc = _crosscheck(spec, doc)
                _emit({"ok": bool(cc["ok"]), "out": args.out, "mode": "spec",
                       "summary": summary, "crosscheck": cc})
                return 0 if cc["ok"] else 2
            doc = G.read_shapes(args.from_shapes, legacy_frame=legacy)
            summary = G.emit_shapes_doc(doc, args.out)
            old = open(args.from_shapes, encoding="utf-8").read()
            new = open(args.out, encoding="utf-8").read()
            beq = _signed_zero_eq(old, new)
            meq = _migration_eq(old, new, doc["export_frame"])
            _emit({"ok": bool(beq or meq), "out": args.out, "mode": "from-shapes",
                   "summary": summary, "export_frame": doc["export_frame"],
                   "legacy_frame_declared": bool(doc.get("_legacy_frame_declared")),
                   "roundtrip": {"byte_equal": beq, "after_frame_migration": meq}})
            return 0 if (beq or meq) else 2

        if bool(args.spec) == bool(args.from_oracle):
            raise G.GeomError("必须且只能给 --spec 或 --from-oracle 之一")
        if args.spec:
            spec = _load_spec(args.spec)
            summary = O.write_oracle(args.out, O.doc_from_dump(spec))
            doc = O.read_oracle(args.out)
            cov = O.frame_coverage_problem(doc)
            _emit({"ok": cov is None, "out": args.out, "mode": "spec",
                   "summary": summary, "frame_coverage_problem": cov})
            return 0 if cov is None else 2
        doc = O.read_oracle(args.from_oracle, legacy_frame=(legacy
                                                            if legacy is not None
                                                            else O._MISSING))
        summary = O.write_oracle(args.out, doc)
        old = open(args.from_oracle, encoding="utf-8").read()
        new = open(args.out, encoding="utf-8").read()
        beq = _signed_zero_eq(old, new)
        meq = _migration_eq(old, new, doc["scene"].get("export_frame"))
        _emit({"ok": bool(beq or meq), "out": args.out, "mode": "from-oracle",
               "summary": summary,
               "roundtrip": {"byte_equal": beq, "after_frame_migration": meq}})
        return 0 if (beq or meq) else 2
    except (G.GeomError, O.OracleContractError, OSError, ValueError, KeyError) as exc:
        _emit({"ok": False, "cmd": args.cmd,
               "error": "%s: %s" % (type(exc).__name__, exc)})
        return 2


def _signed_zero_eq(a, b):
    """比较两份文本：把 `-0.000000` 与 `0.000000` 视作同一数值。

    6 位小数是不可逆舍入（原值可能是 −1e-9，写出即 `-0.000000`，读回成 −0.0），
    所以"逐字节往返"必须容许这一个等价类；其余任何差异都算不同。
    """
    return a.replace("-0.000000", "0.000000") == b.replace("-0.000000", "0.000000")


def _migration_eq(old, new, frame):
    """旧件 → 新口径的可复核等价：**除了新增 `"export_frame": N,` 一处**，其余逐字节相同。

    这是本桥唯一允许的"旧件口径迁移"：位姿描述必须记帧，所以重出必然多这一个键。
    任何**别的**差异都不算等价（返回 False）。
    """
    if frame is None:
        return False
    stripped = new.replace('"export_frame": %d, ' % frame, "")
    return _signed_zero_eq(old, stripped)


if __name__ == "__main__":
    raise SystemExit(main())
