# -*- coding: utf-8 -*-
"""CLI · MV 制作插件的命令行入口

    # 产出整支 MV（默认 960x540 / 24fps、单进程；`--jobs` 缺省 1，960×540 建议 8；
    # 产物落 $MV_OUT 或 ./mv_out）
    python3 -m wm_bridge.factory.film.mv.make --jobs 8

    # 快速探针（只渲 8 帧）
    python3 -m wm_bridge.factory.film.mv.make --max-frames 8 --size 320x180 --jobs 4

    # 只重跑判据（不重渲，秒级）；有 REJECT ⇒ 1（与 --accept 同口径）
    python3 -m wm_bridge.factory.film.mv.make --judge

    # 发布放行：REJECT 或 BLINDSPOT 任一存在 ⇒ 1
    python3 -m wm_bridge.factory.film.mv.make --strict

    # --judge / --strict 的 1 也覆盖「输入产物或裁定文档不合契约」：错误写 stderr
    # （`judge error [<类型>]: …`），stdout 保持干净 JSON 或空，属 fail-closed。

    # 冻结态验收（复算清单 / 确定性 / 判据不变式 / 字幕 / 契约）
    python3 -m wm_bridge.factory.film.mv.make --accept

    # 小体积预览 GIF
    python3 -m wm_bridge.factory.film.mv.make --preview

需要 `WM_LINGSHU_ROOT`（或仓库相对候选）指向 lingshu 仓根 —— 见 `factory/_deps.py`。
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _parse_size(s):
    a, b = s.lower().split("x")
    return (int(a), int(b))


def main(argv=None) -> int:
    from . import DEFAULT_BRIEF, accept, judge, make_mv, summarize
    from ._det import paths, preview

    ap = argparse.ArgumentParser(description="wm_bridge.factory.film.mv · MV 制作插件")
    ap.add_argument("--brief", default=DEFAULT_BRIEF, help="创意简报 JSON")
    ap.add_argument("--out", default=None, help="产物目录（缺省 $MV_OUT 或 ./mv_out）")
    ap.add_argument("--size", default=None, help="画幅，如 960x540")
    ap.add_argument("--jobs", type=int, default=1, help="渲染进程数（每帧纯函数 ⇒ 并行不改像素）")
    ap.add_argument("--ssaa", type=int, default=1)
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--max-shots", type=int, default=None, help="只渲前 N 镜（探针）")
    ap.add_argument("--max-frames", type=int, default=None, help="总共只渲前 N 帧（探针）")
    ap.add_argument("--judge", action="store_true",
                    help="只重跑判据（不重渲）；有 REJECT 时退出码 1（与 --accept 同口径）")
    ap.add_argument("--strict", action="store_true",
                    help="与 --judge 同口径，并按发布放行再要求无 BLINDSPOT（REJECT 或 BLINDSPOT 均退出码 1）")
    ap.add_argument("--accept", action="store_true", help="冻结态验收")
    ap.add_argument("--preview", action="store_true", help="生成预览 GIF")
    a = ap.parse_args(argv)

    out = a.out or paths.out_dir()
    size = _parse_size(a.size) if a.size else None

    if a.judge or a.strict:
        from .._common import ContractError
        try:
            r = judge(out=out, brief=a.brief)
            s = summarize(r["oracle"])
        except ContractError as exc:
            # 契约错有两个来源：judge 侧是「输入产物不合契约」（如 score.json/edit.json
            # 坏），summarize 侧是「裁定文档畸形」。两者都 fail-loud、退出码 1，
            # 但报错必须走 stderr（stdout 正常时是 JSON，不能被污染），并点名异常类型，
            # 让人一眼分清是片子没过还是产物坏了。只抓契约错，不抓宽泛 Exception。
            print("judge error [%s]: %s" % (type(exc).__name__, exc), file=sys.stderr)
            return 1
        print(json.dumps({"out": out, "tally": r["tally"],
                          "blocking": s["blocking"],
                          "gates_passed": s["gates_passed"],
                          "release_blocking": s["release_blocking"],
                          "strict": bool(a.strict)},
                         ensure_ascii=False, indent=2))
        if a.strict:
            return 0 if s["gates_passed"] else 1
        return 1 if s["blocking"] else 0
    if a.accept:
        r = accept(out=out, brief=a.brief)
        for c in r["checks"]:
            print("[%s] %-26s %s  %s" % (c["status"], c["check"], c["group"], c["detail"]))
        print("TALLY", r["tally"], "OK" if r["ok"] else "FAILED")
        return 0 if r["ok"] else 1
    if a.preview:
        p = preview.make_preview(out)
        print("预览 GIF：%s  %d 字节  %d 帧" % (p["path"], p["bytes"], p["frames"]))
        return 0

    r = make_mv(a.brief, out=out, size=size, jobs=max(1, a.jobs), ssaa=a.ssaa,
                gif=not a.no_gif, max_shots=a.max_shots, max_frames=a.max_frames)
    print(json.dumps({"out": out, "frames": len(r["frames"]),
                      "tally": r["tally"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
