# -*- coding: utf-8 -*-
"""wm_bridge.factory.anim.seq · **帧序列编排 + 真动画出口**（纯标准库；PIL **只在函数内** import）

定位（主线）：这是**世界模型"自己产出动画"的出口能力** —— 由 `eval.py`（关节运动）
或 `physics.py`（物理预测）算出逐帧位姿/轨迹后，落到：
    · **帧序列**（逐字节确定的 PNG —— 自己写的编码器，不依赖 Pillow 的写入行为）
    · **真 GIF**（PIL 合成，写后**回读自校验**）
    · **胶片条**（一张**静态**图；`n_frames == 1` —— 名字必须是胶片条，不是动画）

零依赖纪律（D-005）
------------------
    · 帧序列的**写盘**用本模块自带的 PNG 编码器（`zlib` + `struct` + `binascii.crc32`），
      **不 import PIL**；所以"同输入 ⇒ 同字节"是**结构保证**，不是"但愿 Pillow 不写时间戳"。
    · GIF 合成必须用 PIL ⇒ **只在函数体内 import**（与 `lingshu.world.anim` 同约定）。
    · 全程零 numpy。

⚠ 必须知道的 PIL 行为（上游 `world/anim.py` 已实测，这里照抄判据、不重蹈覆辙）
---------------------------------------------------------------------------
`GifImagePlugin._write_multiple_frames` 在**相邻两帧逐像素相同**时会**跳过**后一帧并把
时长累加到前一帧；只剩 1 帧时(:771)回退成单帧 GIF。⇒ 4 张相同帧 ⇒ `n_frames == 1`。
故本模块的 `to_gif`：
    · **单帧输入直接 raise**（一帧不构成动画；也让"胶片条当动画"这条路彻底堵死）
    · 写完**回读**（帧数/尺寸/时长/loop 全核），不一致就 raise，且先写临时文件、
      校验通过才 `os.replace` ⇒ 目标路径上**永不出现**半成品。
GIF 的 delay 是 16 位**厘秒** ⇒ 毫秒只能表达 10 的整数倍（实测请求 105 读回 100），
故 `duration_ms` 非 10 的倍数 / < 10 / > 655350 一律**显式 raise**（不静默改写时长）。
"""
from __future__ import annotations

import binascii
import hashlib
import os
import struct
import tempfile
import zlib

__all__ = [
    "SeqError", "GIF_MS_STEP", "GIF_MS_MAX", "GIF_LOOP_MAX", "GIF_COLORS",
    "encode_png", "write_png", "save_frames", "frame_hashes", "hash_file",
    "to_gif", "to_strip", "from_paths", "read_gif_info", "load_frames",
    "render_track", "motion_track", "timeline",
]

#: GIF 时长粒度（厘秒）
GIF_MS_STEP = 10
#: 65535 厘秒 —— 超过它 PIL 抛裸 struct.error
GIF_MS_MAX = 65535 * 10
#: GIF loop 字段上限（0 = 无限循环）
GIF_LOOP_MAX = 65535
#: 单帧调色板上限
GIF_COLORS = 256


class SeqError(ValueError):
    """帧序列 / GIF 编排违反契约（fail-closed：不产坏文件）。"""


# ============================================================ 确定性 PNG 编码
def encode_png(width, height, rgb, compress_level=6):
    """纯标准库 PNG 编码器（8 位真彩、无 alpha、**无任何时间戳/文本块**）。

    生效条件：`len(rgb) == width*height*3`；否则 `SeqError`（不写半张图）。
    `compress_level` 写死（默认 6）⇒ 同输入同输出**逐字节相同**（zlib 对同输入同 level
    是确定性的）。这是"帧序列可复现"的**结构保证**：不依赖 Pillow 版本的写入细节。
    """
    if isinstance(width, bool) or isinstance(height, bool) or not isinstance(width, int) \
            or not isinstance(height, int) or width < 1 or height < 1:
        raise SeqError("encode_png：width/height 须为 >=1 的整数，得到 %r/%r" % (width, height))
    if not isinstance(rgb, (bytes, bytearray)):
        raise SeqError("encode_png：rgb 须为 bytes，得到 %s" % type(rgb).__name__)
    want = width * height * 3
    if len(rgb) != want:
        raise SeqError("encode_png：rgb 长度 %d != w*h*3 = %d" % (len(rgb), want))

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", binascii.crc32(tag + data) & 0xFFFFFFFF))

    stride = width * 3
    raw = bytearray()
    for y in range(height):
        raw.append(0)                                   # filter type 0（None）
        raw += rgb[y * stride:(y + 1) * stride]
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(bytes(raw), compress_level))
            + chunk(b"IEND", b""))


def write_png(path, image):
    """把一张图（鸭子类型：需有 `.size` 与 `.tobytes()`，PIL Image 满足）写成确定性 PNG。

    走 `encode_png` ⇒ 与 Pillow 版本无关的逐字节确定性。**不 import PIL。**
    """
    if not hasattr(image, "size") or not hasattr(image, "tobytes"):
        raise SeqError("write_png：需要带 .size/.tobytes() 的图对象，得到 %s"
                       % type(image).__name__)
    w, h = int(image.size[0]), int(image.size[1])
    mode = getattr(image, "mode", "RGB")
    im = image if mode == "RGB" else image.convert("RGB")
    data = im.tobytes()
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as f:
        f.write(encode_png(w, h, data))
    return path


def save_frames(images, out_dir, prefix="frame", ext="png"):
    """逐帧落盘（**确定性**），返回按序路径列表。

    生效条件：frames 非空（0 帧 ⇒ `SeqError`）；`ext` 只支持 `png`
    （其它格式要 Pillow ⇒ 那是 GIF 的活，不在这里假装能做）。
    """
    seq = _as_seq(images, "save_frames()")
    if not seq:
        raise SeqError("save_frames()：frames 为空 —— 不做 0 帧输出")
    ext = str(ext).lstrip(".").lower() or "png"
    if ext != "png":
        raise SeqError("save_frames()：ext 只支持 'png'（要 GIF 请走 to_gif），得到 %r" % ext)
    out_dir = os.fspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, im in enumerate(seq):
        p = os.path.join(out_dir, "%s_%04d.png" % (prefix, i))
        write_png(p, im)
        paths.append(p)
    return paths


def hash_file(path):
    """文件 sha256（分块读，大图不吃内存）。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def frame_hashes(paths):
    """逐帧 sha256 列表（`[{"path","sha256","bytes"}, …]`）。"""
    return [{"path": os.fspath(p), "sha256": hash_file(p),
             "bytes": os.path.getsize(p)} for p in paths]


# ============================================================ 帧序列规整
def _lazy_image():
    from PIL import Image  # noqa: PLC0415 —— 刻意函数内 import（D-005：模块顶层零 PIL）
    return Image


def _as_seq(images, where):
    if not isinstance(images, (list, tuple)):
        try:
            images = list(images)
        except TypeError as exc:
            raise SeqError("%s：frames 须为可迭代的图像序列，得到 %s"
                           % (where, type(images).__name__)) from exc
    return list(images)


def _uniform_size(seq, where):
    if not seq:
        raise SeqError("%s：frames 为空" % where)
    size = tuple(seq[0].size)
    for i, im in enumerate(seq):
        if tuple(im.size) != size:
            raise SeqError("%s：帧尺寸不一致 —— 第 0 帧 %s，第 %d 帧 %s；"
                           "PIL 对不一致尺寸**不报错**、会静默只写 1 帧"
                           % (where, size, i, tuple(im.size)))
    return size


def _check_duration(duration_ms):
    if isinstance(duration_ms, bool) or not isinstance(duration_ms, int):
        raise SeqError("to_gif()：duration_ms 须为整数毫秒，得到 %s"
                       % type(duration_ms).__name__)
    if duration_ms < GIF_MS_STEP:
        raise SeqError("to_gif()：duration_ms=%r 不可表达（GIF 以厘秒存时长，最小 %dms；"
                       "0/负/个位数会被读回成 None/0 或静默失真）"
                       % (duration_ms, GIF_MS_STEP))
    if duration_ms % GIF_MS_STEP:
        raise SeqError("to_gif()：duration_ms=%r 不是 %d 的整数倍 —— GIF 以厘秒存时长，"
                       "读回会变成 %dms；与其静默改写，不如显式拒绝"
                       % (duration_ms, GIF_MS_STEP, duration_ms // GIF_MS_STEP * GIF_MS_STEP))
    if duration_ms > GIF_MS_MAX:
        raise SeqError("to_gif()：duration_ms=%r 超出 GIF 上限 %dms" % (duration_ms, GIF_MS_MAX))
    return duration_ms


def _check_loop(loop):
    if isinstance(loop, bool) or not isinstance(loop, int):
        raise SeqError("to_gif()：loop 须为整数（0 = 无限循环），得到 %s" % type(loop).__name__)
    if loop < 0 or loop > GIF_LOOP_MAX:
        raise SeqError("to_gif()：loop=%r 不可表达（合法 0..%d）" % (loop, GIF_LOOP_MAX))
    return loop


def _flatten_rgba(seq, background, where="to_gif()"):
    """把带 alpha 的帧**如实合成**到背景色，并返回 `(新序列, 被合成的帧数)`。

    ⚠ 这是一条**独立复核者实测抓到的 fail-open**：原实现直接 `im.convert("RGB")`，
    RGBA 输入的 alpha 被**静默丢弃**（半透明像素变成其未预乘的 RGB，画面与输入不符，
    而返回值里看不出发生过任何事）。按本桥纪律：**不许静默丢** —— 要么 raise，
    要么如实合成**并记录**。这里选后者（GIF 本来也不支持 alpha），
    把帧数记进 `to_gif` 的返回体 `alpha_composited` 与 `alpha_background`。
    """
    Image = _lazy_image()
    bg = tuple(int(c) for c in background)
    if len(bg) != 3:
        raise SeqError("%s：background 须为 (r,g,b)，得到 %r" % (where, background))
    out = []
    n = 0
    for i, im in enumerate(seq):
        if "A" in im.getbands():
            n += 1
            base = Image.new("RGB", im.size, bg)
            base.paste(im.convert("RGBA"), (0, 0), im.convert("RGBA"))
            out.append(base)
        else:
            out.append(im.convert("RGB"))
    return out, n


def _quantize(seq):
    """显式钉住调色板（跨帧共享 ⇒ 不闪色且确定性）；拼图超预算则退回逐帧自适应。"""
    Image = _lazy_image()
    rgb = [im.convert("RGB") for im in seq]
    w, h = rgb[0].size
    if w * h * len(rgb) <= 16_000_000:
        sheet = Image.new("RGB", (w, h * len(rgb)))
        for i, im in enumerate(rgb):
            sheet.paste(im, (0, i * h))
        ref = sheet.quantize(colors=GIF_COLORS, method=Image.Quantize.MEDIANCUT,
                             dither=Image.Dither.NONE)
        pal = Image.new("P", (1, 1))
        pal.putpalette(ref.getpalette())
        return [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in rgb]
    return [im.quantize(colors=GIF_COLORS, method=Image.Quantize.MEDIANCUT,
                        dither=Image.Dither.NONE) for im in rgb]


def to_gif(images, path, duration_ms=60, loop=0, min_frames=2, background=(0, 0, 0)):
    """帧序列 → **真动画** GIF；返回**回读到的**观测面。

    `{"path","frames","size","duration_ms","loop","bytes","alpha_composited",
      "alpha_background"}`（前者全部来自 write 之后的回读）。

    **RGBA 输入不再静默丢 alpha**：带 alpha 的帧按 `background`（默认黑）如实合成，
    并把被合成的帧数记进返回体 `alpha_composited`（独立复核者实测的 fail-open，已修）。

    生效条件（任一不满足 ⇒ `SeqError`，且**目标路径上不留半成品**）：
        · `len(images) < min_frames`（默认 2）—— **单帧不是动画**（也堵死"胶片条冒充动画"）
        · 尺寸不一致 / duration 非 10 的倍数或 <=0 / loop 越界
        · **写后回读**帧数、尺寸、时长、loop 与请求不符
    """
    Image = _lazy_image()
    del Image
    seq = _as_seq(images, "to_gif()")
    if len(seq) < int(min_frames):
        raise SeqError("to_gif()：只有 %d 帧（min_frames=%d）—— 单帧图不是动画；"
                       "要静态横排图请用 to_strip()，要逐帧落盘请用 save_frames()"
                       % (len(seq), int(min_frames)))
    size = _uniform_size(seq, "to_gif()")
    duration_ms = _check_duration(duration_ms)
    loop = _check_loop(loop)
    path = os.fspath(path)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp",
                               dir=parent or ".")
    os.close(fd)
    flat, n_alpha = _flatten_rgba(seq, background)
    pal = _quantize(flat)
    try:
        pal[0].save(tmp, format="GIF", save_all=True, append_images=pal[1:],
                    duration=duration_ms, loop=loop, optimize=False, disposal=2)
        obs = _verify_gif(tmp, len(seq), size, duration_ms, loop)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
    obs["path"] = path
    obs["alpha_composited"] = n_alpha            # ⇒ 0 表示输入本来就无 alpha（没有静默丢失）
    obs["alpha_background"] = tuple(int(c) for c in background)
    return obs


def _verify_gif(path, want_frames, want_size, want_duration, want_loop):
    Image = _lazy_image()
    with Image.open(path) as rb:
        got = {"frames": rb.n_frames, "size": tuple(rb.size),
               "duration_ms": rb.info.get("duration"), "loop": rb.info.get("loop")}
    if got["frames"] != want_frames:
        raise SeqError("to_gif()：期望 %d 帧，回读到 %d 帧 —— GifImagePlugin 对"
                       "**与前帧逐像素相同**的帧无条件跳过（并把时长累加到前帧），"
                       "只剩 1 帧时回退单帧；修法：让相邻帧真正不同" % (want_frames, got["frames"]))
    if got["size"] != tuple(want_size):
        raise SeqError("to_gif()：回读尺寸 %s != 请求 %s" % (got["size"], tuple(want_size)))
    if got["duration_ms"] != want_duration:
        raise SeqError("to_gif()：回读时长 %rms != 请求 %dms" % (got["duration_ms"], want_duration))
    if got["loop"] != want_loop:
        raise SeqError("to_gif()：回读 loop=%r != 请求 %d" % (got["loop"], want_loop))
    got["bytes"] = os.path.getsize(path)
    return got


def to_strip(images, cols=None, background=(18, 22, 34)):
    """帧序列 → **胶片条**（一张静态图）。名字必须是胶片条，**不是动画**。

    返回值上 `n_frames == 1`（仅当该属性缺失时补，不覆盖 Pillow 将来可能自带的）
    —— 与 `Image.open(落盘文件).n_frames` **同义**，这是"动画 vs 胶片条"的判据落点。
    """
    Image = _lazy_image()
    seq = _as_seq(images, "to_strip()")
    if not seq:
        raise SeqError("to_strip()：frames 为空 —— 做不出胶片条")
    w, h = _uniform_size(seq, "to_strip()")
    if cols is None:
        cols = len(seq)
    if isinstance(cols, bool) or not isinstance(cols, int) or cols < 1:
        raise SeqError("to_strip()：cols 须为 >=1 的整数，得到 %r" % (cols,))
    bg = tuple(int(c) for c in background)
    if len(bg) != 3:
        raise SeqError("to_strip()：background 须为 (r,g,b)")
    rows = -(-len(seq) // cols)
    sheet = Image.new("RGB", (w * cols, h * rows), bg)
    for i, im in enumerate(seq):
        sheet.paste(im.convert("RGB"), ((i % cols) * w, (i // cols) * h))
    if not hasattr(sheet, "n_frames"):
        sheet.n_frames = 1
    return sheet


def load_frames(paths):
    """读回一组帧文件（PIL 惰性）→ `List[Image]`；缺文件 ⇒ `SeqError`。"""
    Image = _lazy_image()
    out = []
    for p in paths:
        p = os.fspath(p)
        if not os.path.isfile(p):
            raise SeqError("load_frames：帧文件不存在：%s" % p)
        with Image.open(p) as im:
            im.load()
            out.append(im.copy())
    return out


def from_paths(paths, out=None, duration_ms=60, loop=0, min_frames=2):
    """已落盘的帧序列 → GIF（把既有产物接起来）；顺序**按给定顺序**，不排序。"""
    seq = load_frames(paths)
    if out is None:
        out = os.path.join(os.path.dirname(os.fspath(paths[0])), "frames.gif")
    return to_gif(seq, out, duration_ms=duration_ms, loop=loop, min_frames=min_frames)


def read_gif_info(path):
    """回读 GIF 观测面（判据用）：`{"frames","size","duration_ms","loop","bytes"}`。"""
    Image = _lazy_image()
    with Image.open(path) as rb:
        return {"frames": rb.n_frames, "size": tuple(rb.size),
                "duration_ms": rb.info.get("duration"), "loop": rb.info.get("loop"),
                "bytes": os.path.getsize(path)}


def timeline(fps, frame_start, count):
    """帧号时间线：`frame_start + 0..count-1`（`count < 1` ⇒ `SeqError`）。"""
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise SeqError("timeline：count 须 >= 1，得到 %r" % (count,))
    return [int(frame_start) + i for i in range(count)]


# ============================================================ 轨迹 → 逐帧位图（主线出口）
def _axonometric(p, k, ox, oy):
    """等轴测投影（世界 (x,y,z) → 屏幕 (sx,sy)）：+x 右、+y 上、+z 往右下压。"""
    return (ox + (p[0] + 0.45 * p[2]) * k, oy - (p[1] - 0.32 * p[2]) * k)


def render_track(tracks, size=(560, 560), scale=44.0, trail=24, grid=6.0,
                 bg=(14, 18, 30), fg=(120, 235, 255), trail_col=(40, 120, 170),
                 ground=(28, 34, 54)):
    """物理预测轨迹 → 逐帧 PIL 图（**世界模型自己的动画出口**，不经过 Blender）。

    `tracks`：`[(t, (x,y,z), (vx,vy,vz)), …]`（`physics.predict_trajectory` 的输出）。
    · 地面网格（y=0）+ 已走过的轨迹（渐隐）+ 当前位置的球 + 速度矢量。
    · `trail`：拖尾帧数（0 = 不画拖尾）。
    · 生效条件：`tracks` 非空、`size` 为正、`scale > 0`。
    """
    # #2105（同函数第二处残留）：修前 `Image, ImageDraw = _lazy_image(), None` +
    # `del Image` 让后续 `Image.new` 抛 UnboundLocalError（删掉调试行后露出的
    # 下一层）——删 `del Image`，Image/ImageDraw 正常供给画布。
    Image = _lazy_image()
    from PIL import ImageDraw as _ID  # noqa: PLC0415
    ImageDraw = _ID
    tr = list(tracks)
    if not tr:
        raise SeqError("render_track：tracks 为空 —— 不做 0 帧动画")
    w, h = int(size[0]), int(size[1])
    if w < 16 or h < 16:
        raise SeqError("render_track：size 太小（>=16×16），得到 %r" % (size,))
    if scale <= 0:
        raise SeqError("render_track：scale 须 > 0，得到 %r" % (scale,))
    ys = [p[1] for _t, p, _v in tr]
    xs = [p[0] for _t, p, _v in tr]
    zs = [p[2] for _t, p, _v in tr]
    # #2105：残留调试行 `ex = min(xs + zs * 0.0), max(...)` 已删——`zs` 是 list，
    # `list * 0.0` 按语言规范必抛 TypeError（render_track 对任何合法输入 100% 崩，
    # 且 ex 随即 del、无任何消费者）。
    cx = 0.5 * (min(xs) + max(xs))
    k = scale
    ox = w * 0.5 - (cx + 0.45 * 0.5 * (min(zs) + max(zs))) * k
    oy = h * 0.72 + 0.5 * (min(ys) + max(ys)) * k
    frames = []
    for i, (_t, p, v) in enumerate(tr):
        im = Image.new("RGB", (w, h), bg)
        d = ImageDraw.Draw(im)
        # 地面网格
        g = int(grid)
        for gi in range(-g, g + 1):
            a = _axonometric((gi * 1.0, 0.0, -g * 1.0), k, ox, oy)
            b = _axonometric((gi * 1.0, 0.0, g * 1.0), k, ox, oy)
            d.line([a, b], fill=ground, width=1)
            a = _axonometric((-g * 1.0, 0.0, gi * 1.0), k, ox, oy)
            b = _axonometric((g * 1.0, 0.0, gi * 1.0), k, ox, oy)
            d.line([a, b], fill=ground, width=1)
        # 轨迹拖尾
        lo = max(0, i - int(trail)) if trail else i
        pts = [_axonometric(tr[j][1], k, ox, oy) for j in range(lo, i + 1)]
        for j in range(1, len(pts)):
            frac = j / float(max(1, len(pts) - 1))
            col = tuple(int(trail_col[c] * (0.35 + 0.65 * frac)) for c in range(3))
            d.line([pts[j - 1], pts[j]], fill=col, width=2)
        # 位置球 + 速度矢量
        sx, sy = _axonometric(p, k, ox, oy)
        r = 6
        d.ellipse([sx - r, sy - r, sx + r, sy + r], fill=fg)
        va = _axonometric(p, k, ox, oy)
        vb = _axonometric((p[0] + v[0] * 0.25, p[1] + v[1] * 0.25, p[2] + v[2] * 0.25),
                          k, ox, oy)
        d.line([va, vb], fill=(255, 190, 90), width=3)
        frames.append(im)
    return frames


# ============================================================ 运动量化读数
def motion_track(paths, analysis_width=280, threshold=8):
    """逐帧**前景像素质心位移** + 运动面积（"确实在动"的量化证据）。

    前景判据（不猜相机、不猜背景色）：**逐像素跨帧中位数**作为静态背景估计，
    `|pixel − median|` 三通道之和 > `threshold` 记为前景。中位数对"运动物体"
    天然稳健（物体只在一部分帧里盖住该像素）。

    返回：
        {"analysis_size", "frames": [{"i","fg_px","fg_frac","cx","cy","cx_norm","cy_norm"}…],
         "disp_px": [逐帧质心位移（分析分辨率）],
         "max_disp_px", "mean_disp_px", "min_disp_px",
         "temporal_std_moving_frac": 时序标准差超阈的像素占比（运动**面积**证据）,
         "mean_abs_interframe_diff": 相邻帧平均绝对差（0..255）,
         "scale_to_full": 分析分辨率→原图比例的换算系数}

    生效条件：paths 非空、至少 2 帧；否则 `SeqError`。
    """
    Image = _lazy_image()
    from PIL import ImageChops, ImageStat  # noqa: PLC0415
    ps = [os.fspath(p) for p in paths]
    if len(ps) < 2:
        raise SeqError("motion_track：至少需要 2 帧才能谈运动，得到 %d" % len(ps))
    full = None
    small = []
    for p in ps:
        if not os.path.isfile(p):
            raise SeqError("motion_track：帧文件不存在：%s" % p)
        with Image.open(p) as im:
            im = im.convert("RGB")
            if full is None:
                full = im.size
            ratio = float(analysis_width) / im.size[0]
            s = im.resize((max(2, int(im.size[0] * ratio)),
                           max(2, int(im.size[1] * ratio))), Image.BILINEAR)
            small.append(s)
    w, h = small[0].size
    if any(im.size != (w, h) for im in small):
        raise SeqError("motion_track：缩放后尺寸不一致（原图尺寸不齐）")
    data = [im.tobytes() for im in small]
    n = len(data)
    npix = w * h
    med = bytearray(len(data[0]))
    for i in range(len(data[0])):
        vals = sorted(fb[i] for fb in data)
        med[i] = vals[n // 2]
    out_frames = []
    disp = []
    prev = None
    moving = 0
    for fi, fb in enumerate(data):
        sx = sy = 0.0
        cnt = 0
        for ix in range(npix):
            b = ix * 3
            d = abs(fb[b] - med[b]) + abs(fb[b + 1] - med[b + 1]) + abs(fb[b + 2] - med[b + 2])
            if d > threshold:
                sx += ix % w
                sy += ix // w
                cnt += 1
        cx = (sx / cnt) if cnt else float("nan")
        cy = (sy / cnt) if cnt else float("nan")
        out_frames.append({"i": fi, "fg_px": cnt, "fg_frac": cnt / float(npix),
                           "cx": cx, "cy": cy, "cx_norm": cx / w, "cy_norm": cy / h})
        if prev is not None and cnt and out_frames[fi - 1]["fg_px"]:
            disp.append(((cx - prev[0]) ** 2 + (cy - prev[1]) ** 2) ** 0.5)
        prev = (cx, cy)
    # 时序标准差超阈的像素占比（运动面积）
    for ix in range(npix):
        b = ix * 3
        for c in range(3):
            vals = [fb[b + c] for fb in data]
            m = sum(vals) / n
            var = sum((v - m) ** 2 for v in vals) / n
            if var ** 0.5 > threshold / 2.0:
                moving += 1
                break
    diffs = []
    for a, b in zip(small, small[1:]):
        diffs.append(ImageStat.Stat(ImageChops.difference(a, b)).mean)
    mad = sum(sum(x) / 3.0 for x in diffs) / max(1, len(diffs))
    ds = sorted(disp)
    jerk = 0.0
    for i in range(1, len(disp)):
        jerk = max(jerk, abs(disp[i] - disp[i - 1]))
    return {
        "analysis_size": (w, h),
        "full_size": full,
        "frames": out_frames,
        "disp_px": disp,
        "max_disp_px": max(disp) if disp else 0.0,
        "mean_disp_px": (sum(disp) / len(disp)) if disp else 0.0,
        "min_disp_px": ds[0] if ds else 0.0,
        "median_disp_px": ds[len(ds) // 2] if ds else 0.0,
        "max_second_difference_px": jerk,
        "jerk_ratio": (jerk / max(disp)) if disp and max(disp) > 0 else 0.0,
        "temporal_std_moving_frac": moving / float(npix),
        "mean_abs_interframe_diff": mad,
        "scale_to_full": (full[0] / float(w)) if full else 1.0,
    }
