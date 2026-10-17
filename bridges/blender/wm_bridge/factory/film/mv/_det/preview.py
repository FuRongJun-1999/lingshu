# -*- coding: utf-8 -*-
"""preview · 从成片帧序列生成**小体积预览 GIF**（交付件：让用户一眼看到成片）

    python3 -X utf8 -m mvpipe.preview --out out --stride 4 --width 480

为什么单独做：全片 576 帧 @960×540 直接存 GIF 会有几十 MB（GIF 是 8 位调色板 + LZW，
且 576 帧全存没必要）。预览按 `stride` 抽帧、按 `width` 缩放，体积可控。

确定性：抽帧/缩放/量化参数全部固定 ⇒ 同一输入两次生成的文件逐字节相同。
"""
from __future__ import annotations

import argparse
import os
import sys

from PIL import Image

from . import paths


def make_preview(out: str, stride: int = 4, width: int = 480,
                 max_colors: int = 128, max_frames: int = 96) -> dict:
    """抽帧 → 缩放 → 量化 → 存动画 GIF。返回 {path, bytes, frames, sha256}。"""
    import hashlib

    fdir = os.path.join(out, "frames")
    if not os.path.isdir(fdir):
        raise FileNotFoundError("找不到帧目录：%s（先跑 mvpipe.make）" % fdir)
    files = sorted(f for f in os.listdir(fdir) if f.endswith(".png"))
    if not files:
        raise FileNotFoundError("帧目录为空：%s" % fdir)
    picks = files[::max(1, stride)][:max_frames]
    imgs = []
    for name in picks:
        im = Image.open(os.path.join(fdir, name)).convert("RGB")
        w, h = im.size
        nh = max(2, int(round(h * width / float(w))))
        # 缩放用固定重采样（NEAREST 会锯齿、LANCZOS 最平滑但慢；统一用 BILINEAR）
        imgs.append(im.resize((width, nh), Image.BILINEAR))
    # 统一量化（同一调色板口径 ⇒ 确定性）
    pal = [im.quantize(colors=max_colors, method=Image.MEDIANCUT) for im in imgs]
    path = os.path.join(out, "mv_preview.gif")
    pal[0].save(path, "GIF", save_all=True, append_images=pal[1:],
                duration=int(round(1000.0 * stride / 24.0)), loop=0, optimize=False)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return {"path": path, "bytes": os.path.getsize(path), "frames": len(pal),
            "sha256": h.hexdigest(), "stride": stride, "width": width}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="成片预览 GIF")
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out", default=os.path.join(here, "out"))
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--colors", type=int, default=128)
    ap.add_argument("--max-frames", type=int, default=96)
    a = ap.parse_args(argv)
    r = make_preview(a.out, a.stride, a.width, a.colors, a.max_frames)
    print("预览 GIF：%s  %d 字节  %d 帧  sha256=%s"
          % (r["path"], r["bytes"], r["frames"], r["sha256"][:16]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
