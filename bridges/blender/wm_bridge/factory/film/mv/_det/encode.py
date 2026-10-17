# -*- coding: utf-8 -*-
"""出口编码：纯 Python 手写 MJPEG/AVI 容器 + 动画 GIF + 产物清单（+ 可选 MP4 交付形态）。

本机**PATH 里没有 ffmpeg**（INTERFACES.md §0），所以主产物 AVI 容器全部手写：
    RIFF 'AVI '
      LIST 'hdrl'
        'avih' AVIMAINHEADER
        LIST 'strl'
          'strh' AVISTREAMHEADER（vids / MJPG）
          'strf' BITMAPINFOHEADER（biCompression='MJPG', biBitCount=24）
      LIST 'movi'
        '00dc' <JPEG bytes>   （块长按 2 字节对齐补齐）
      'idx1' 每帧 16 字节（ckid='00dc', flags=0x10, offset, size）

**可选** MP4（H.264 + AAC）交付形态：若探测到 ffmpeg（环境变量 / `imageio_ffmpeg` 自带 /
PATH），`write_mp4` 会额外合成一个带音轨的 MP4；**探测不到 ⇒ 返回 None**，绝不因此让产线失败。
主产物（AVI/GIF/帧序列）**不依赖** ffmpeg，且 MP4 的 H.264 逐字节不确定 ⇒ 返回体如实标注
`deterministic: False`，确定性一律以 AVI / 帧序列为准。

纪律（继承仓内既有）：
  · **fail-closed**：帧缺失 / 尺寸不一致 / 空帧表 / 回读帧数不符 ⇒ 抛 `EncodeError`，
    绝不静默截断、绝不静默降级；
  · **确定性**：头里不写当前时间（`dwMicroSecPerFrame = round(1e6/fps)`，其余保留字段固定），
    JPEG 参数固定（quality / subsampling 显式传入），同一帧列表两次写 ⇒ **sha256 相同**；
  · `write_manifest` 的时间戳**只从入参取**，不读系统时钟。
"""
from __future__ import annotations

import hashlib
import os
import shutil
import struct
import subprocess
import tempfile

from . import contracts

try:  # PIL 是渲染层允许的依赖（INTERFACES.md §0）；缺了也不许静默降级
    from PIL import Image
except Exception as _e:  # pragma: no cover - 环境缺 PIL 时给出可读错误
    Image = None
    _PIL_IMPORT_ERROR = _e
else:
    _PIL_IMPORT_ERROR = None

#: JPEG 编码参数（**固定**，是确定性的前提之一）
JPEG_QUALITY = 90
JPEG_SUBSAMPLING = 0        # 4:4:4，避免色度下采样引入的色偏
#: 清单 schema（本项目命名口径：wmmv.* / wmfilm.*）
MANIFEST_SCHEMA = "wmmv.manifest/1"
#: idx1 偏移基准：`dwChunkOffset` 相对 **'movi' 标签**（= LIST payload 起点）计数，
#: 且指向帧**块头**（'00dc' fourcc）。'movi' 标签之后紧跟第 0 帧块 ⇒ 首条 offset = 4。
_MOVI_DATA_OFFSET = 4
#: 帧块 fourcc：未压缩视频流 0 号 ⇒ '00dc'（'dc' = 压缩帧）
_FRAME_CKID = b"00dc"
#: idx1 条目 flags：0x10 = AVIIF_KEYFRAME（MJPG 每帧都是关键帧）
_AVIIF_KEYFRAME = 0x10


class EncodeError(Exception):
    """出口编码失败（fail-closed：不猜、不补、不静默降级）。"""


# ---------------------------------------------------------------- 小工具

def _require_pil() -> None:
    if Image is None:  # pragma: no cover
        raise EncodeError("需要 PIL（Pillow）才能编码：%s" % _PIL_IMPORT_ERROR)


def _check_fps(fps) -> int:
    if isinstance(fps, bool) or not isinstance(fps, int):
        raise EncodeError("fps 须为正整数，得到 %r" % (fps,))
    if fps <= 0:
        raise EncodeError("fps 须 > 0，得到 %r" % (fps,))
    return int(fps)


def _check_size(size) -> tuple:
    if not isinstance(size, (tuple, list)) or len(size) != 2:
        raise EncodeError("size 须为 (w, h)，得到 %r" % (size,))
    w, h = size
    for v, name in ((w, "w"), (h, "h")):
        if isinstance(v, bool) or not isinstance(v, int):
            raise EncodeError("size.%s 须为整数，得到 %r" % (name, v))
    if w <= 0 or h <= 0:
        raise EncodeError("size 须为正，得到 %r" % (size,))
    return (int(w), int(h))


def _check_frames(frames) -> list:
    if isinstance(frames, (str, bytes)) or not isinstance(frames, (list, tuple)):
        raise EncodeError("frames 须为路径/图像列表，得到 %r" % (type(frames).__name__,))
    frames = list(frames)
    if not frames:
        raise EncodeError("frames 不得为空（0 帧不成片）")
    return frames


def _frame_source(item, idx: int):
    """把一帧入参归一为 (来源说明, 打开函数)。

    支持：`str`/`os.PathLike` 图片路径（契约口径），或 PIL `Image`（便于渲染层直连）。
    """
    if isinstance(item, (str, os.PathLike)):
        p = os.fspath(item)
        if not os.path.isfile(p):
            raise EncodeError("第 %d 帧文件不存在：%s" % (idx, p))
        return (p, lambda: Image.open(p))
    if Image is not None and isinstance(item, Image.Image):
        return ("<PIL.Image #%d>" % idx, lambda: item)
    raise EncodeError("第 %d 帧须为图片路径或 PIL.Image，得到 %r" % (idx, type(item).__name__))


def _load_rgb(item, idx: int, size: tuple):
    """打开一帧并校验尺寸（fail-closed），返回 RGB `Image`（已 `load()`，可脱离文件句柄）。"""
    where, opener = _frame_source(item, idx)
    try:
        with opener() as im:
            im.load()
            if im.size != size:
                raise EncodeError("第 %d 帧 %s 尺寸 %r 与 size %r 不一致"
                                  % (idx, where, tuple(im.size), size))
            return im.convert("RGB")
    except EncodeError:
        raise
    except Exception as e:
        raise EncodeError("第 %d 帧 %s 无法读取：%s: %s" % (idx, where, type(e).__name__, e))


def _encode_jpeg(im, quality: int, subsampling: int) -> bytes:
    import io
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=int(quality), subsampling=int(subsampling),
            optimize=False, progressive=False)
    return buf.getvalue()


def _u32(n: int) -> int:
    if not (0 <= n <= 0xFFFFFFFF):
        raise EncodeError("超出 uint32 范围：%r" % (n,))
    return int(n)


def _chunk(ckid: bytes, payload: bytes) -> bytes:
    """一个 RIFF 块：fourcc + size + payload（payload 长度须为偶数）。"""
    if len(ckid) != 4:
        raise EncodeError("fourcc 须为 4 字节，得到 %r" % (ckid,))
    if len(payload) % 2:
        raise EncodeError("块 %r 的 payload 长度须为偶数（得到 %d）" % (ckid, len(payload)))
    return ckid + struct.pack("<I", _u32(len(payload))) + payload


def _list_chunk(list_type: bytes, payload: bytes) -> bytes:
    """一个 RIFF `LIST`：'LIST' + size + list_type + payload（size 含 list_type）。"""
    if len(list_type) != 4:
        raise EncodeError("LIST 类型须为 4 字节，得到 %r" % (list_type,))
    body = list_type + payload
    if len(body) % 2:
        raise EncodeError("LIST %r 的 body 长度须为偶数（得到 %d）" % (list_type, len(body)))
    return b"LIST" + struct.pack("<I", _u32(len(body))) + body


# ---------------------------------------------------------------- sha256

def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    """文件 sha256（64 hex，流式读取，大文件不爆内存）。"""
    p = os.fspath(path)
    if not os.path.isfile(p):
        raise EncodeError("文件不存在：%s" % p)
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- AVI 写出

def write_mjpeg_avi(frames: list, path: str, fps: int, size: tuple) -> dict:
    """把 PNG 帧序列编成 MJPEG/AVI（手写 RIFF 容器）。

    `frames` = 图片路径列表（契约口径；也接受 PIL.Image 便于直连）。
    返回读数：`{"path","bytes","sha256","frames","size","fps","codec","jpeg_quality",
    "jpeg_subsampling"}`。失败一律抛 `EncodeError`。
    """
    _require_pil()
    fps = _check_fps(fps)
    size = _check_size(size)
    frames = _check_frames(frames)
    w, h = size

    # 先全部编码成 JPEG（内存里），这样 chunk 长度与 idx1 偏移一次算准
    jpegs = []
    for i, item in enumerate(frames):
        im = _load_rgb(item, i, size)
        data = _encode_jpeg(im, JPEG_QUALITY, JPEG_SUBSAMPLING)
        if len(data) < 4 or data[:2] != b"\xff\xd8":
            raise EncodeError("第 %d 帧 JPEG 编码结果非法（非 SOI 开头）" % i)
        jpegs.append(data)

    n = len(jpegs)
    micros = int(round(1e6 / fps))
    if micros <= 0:
        raise EncodeError("fps=%r 过大：dwMicroSecPerFrame 会算成 0" % (fps,))

    # ---- avih：AVIMAINHEADER（14 个 DWORD；不写当前时间） ----
    avih = struct.pack(
        "<14I",
        micros,                 # dwMicroSecPerFrame
        0,                      # dwMaxBytesPerSec（信息性；固定 0 以保证确定性）
        0,                      # dwPaddingGranularity
        _u32(_AVIIF_KEYFRAME),  # dwFlags：AVIF_HASINDEX(0x10)
        n,                      # dwTotalFrames
        0,                      # dwInitialFrames
        1,                      # dwStreams
        0,                      # dwSuggestedBufferSize
        w, h,                   # dwWidth / dwHeight
        0, 0, 0, 0,             # dwReserved[4]
    )

    # ---- strh：AVISTREAMHEADER（vids + MJPG；dwSampleSize=0 = 每帧一个 chunk） ----
    strh = struct.pack(
        "<4s4sIHHIIIIIIIIhhhh",
        b"vids", b"MJPG",
        0,                      # dwFlags
        0, 0,                   # wPriority / wLanguage
        0,                      # dwInitialFrames
        1,                      # dwScale
        fps,                    # dwRate ⇒ fps = dwRate / dwScale
        0,                      # dwStart
        n,                      # dwLength（帧数）
        0,                      # dwSuggestedBufferSize
        0xFFFFFFFF,             # dwQuality（信息性：用默认质量）
        0,                      # dwSampleSize（0 ⇒ 每帧一 chunk）
        0, 0, w, h,             # rcFrame：left/top/right/bottom
    )

    # ---- strf：BITMAPINFOHEADER（biCompression='MJPG', biBitCount=24） ----
    strf = struct.pack(
        "<IiiHH4sIiiII",
        40,                     # biSize
        w, h,                   # biWidth / biHeight
        1, 24,                  # biPlanes / biBitCount
        b"MJPG",                # biCompression
        0,                      # biSizeImage（MJPG 可为 0）
        0, 0,                   # biXPelsPerMeter / biYPelsPerMeter
        0, 0,                   # biClrUsed / biClrImportant
    )

    strl = _list_chunk(b"strl", _chunk(b"strh", strh) + _chunk(b"strf", strf))
    hdrl = _list_chunk(b"hdrl", _chunk(b"avih", avih) + strl)

    # ---- movi：每帧一个 '00dc'（奇数长度补 1 字节 0） ----
    movi_body = bytearray()
    idx_entries = []
    for i, data in enumerate(jpegs):
        # 帧块相对 'movi' 标签的偏移：标签(4) + 已写字节（指向块头 '00dc'）
        offset = _MOVI_DATA_OFFSET + len(movi_body)
        movi_body += _FRAME_CKID
        movi_body += struct.pack("<I", _u32(len(data)))
        movi_body += data
        if len(data) % 2:
            movi_body += b"\x00"
        idx_entries.append(struct.pack("<4sIII", _FRAME_CKID, _AVIIF_KEYFRAME,
                                       _u32(offset), _u32(len(data))))

    movi = _list_chunk(b"movi", bytes(movi_body))
    idx1 = _chunk(b"idx1", b"".join(idx_entries))

    body = hdrl + movi + idx1
    riff = b"RIFF" + struct.pack("<I", _u32(len(body) + 4)) + b"AVI " + body

    d = os.path.dirname(os.path.abspath(os.fspath(path)))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(os.fspath(path), "wb") as f:
        f.write(riff)

    return {"path": os.fspath(path), "bytes": len(riff), "sha256": sha256_file(path),
            "frames": n, "size": size, "fps": fps, "codec": "MJPG",
            "jpeg_quality": JPEG_QUALITY, "jpeg_subsampling": JPEG_SUBSAMPLING}


# ---------------------------------------------------------------- AVI 回读

def _iter_riff(data: bytes, start: int, end: int):
    """在 [start, end) 内按 RIFF 块遍历，产出 (ckid, payload_off, payload_len)。"""
    off = start
    while off + 8 <= end:
        ckid = data[off:off + 4]
        (ln,) = struct.unpack_from("<I", data, off + 4)
        body = off + 8
        if body + ln > end:
            raise EncodeError("RIFF 块 %r 在 %d 处越界（声明 %d 字节，剩余 %d）"
                              % (ckid, off, ln, end - body))
        yield ckid, body, ln
        off = body + ln + (ln % 2)      # 块按 2 字节对齐


def _scan_movi(data: bytes, movi_tag: int, movi_size: int) -> list:
    """扫描 `LIST movi` 的帧块，返回 [(ckid, payload_off, payload_len)]（按文件顺序）。

    `movi_tag` = 'movi' 标签的绝对偏移；`movi_size` = 该 LIST 的 size（含 4 字节标签）。
    """
    body_start = movi_tag + 4
    body_end = movi_tag + movi_size
    if body_end > len(data):
        raise EncodeError("movi 块越界（end=%d > 文件 %d 字节）" % (body_end, len(data)))
    found = []
    for ckid, poff, plen in _iter_riff(data, body_start, body_end):
        if ckid == _FRAME_CKID:
            found.append((ckid, poff, plen))
    return found


def read_mjpeg_avi(path: str) -> dict:
    """**真解析** AVI：从 `hdrl` 取尺寸与 fps，数 `movi` 里的 `00dc` 帧块。

    返回 `{"frames": n, "size": (w, h), "fps": f}`（另附回读自证读数：
    `bytes` / `sha256` / `idx1_frames` / `movi_bytes` / `jpeg_sha256`）。
    帧数与 idx1 不符、帧数为 0、容器畸形 ⇒ 抛 `EncodeError`。
    """
    p = os.fspath(path)
    if not os.path.isfile(p):
        raise EncodeError("AVI 文件不存在：%s" % p)
    with open(p, "rb") as f:
        data = f.read()
    if len(data) < 12 or data[:4] != b"RIFF":
        raise EncodeError("不是 RIFF 容器：%s" % p)
    (riff_size,) = struct.unpack_from("<I", data, 4)
    if data[8:12] != b"AVI ":
        raise EncodeError("RIFF 类型不是 'AVI '：%r" % (data[8:12],))
    if riff_size + 8 > len(data):
        raise EncodeError("RIFF 声明 %d 字节，文件只有 %d" % (riff_size + 8, len(data)))

    width = height = fps = None
    stream_frames = None
    movi_tag = movi_size = None
    idx1_payload = None

    for ckid, poff, plen in _iter_riff(data, 12, riff_size + 8):
        if ckid == b"LIST" and plen >= 4:
            ltype = data[poff:poff + 4]
            if ltype == b"hdrl":
                for c2, p2, l2 in _iter_riff(data, poff + 4, poff + plen):
                    if c2 == b"avih":
                        if l2 < 40:
                            raise EncodeError("avih 过短（%d 字节）" % l2)
                        vals = struct.unpack_from("<14I", data, p2)
                        micros = vals[0]
                        width, height = vals[8], vals[9]
                        if micros <= 0:
                            raise EncodeError("avih.dwMicroSecPerFrame=%d 非法" % micros)
                        fps = 1e6 / micros
                    elif c2 == b"LIST" and l2 >= 4 and data[p2:p2 + 4] == b"strl":
                        for c3, p3, l3 in _iter_riff(data, p2 + 4, p2 + l2):
                            if c3 == b"strh":
                                if l3 < 48:
                                    raise EncodeError("strh 过短（%d 字节）" % l3)
                                fcc_type = data[p3:p3 + 4]
                                fcc_handler = data[p3 + 4:p3 + 8]
                                # strh：+20 dwScale / +24 dwRate / +28 dwStart / +32 dwLength
                                scale, rate, _start, length = struct.unpack_from("<IIII", data, p3 + 20)
                                if fcc_type != b"vids":
                                    raise EncodeError("strh.fccType=%r 非视频流" % (fcc_type,))
                                if fcc_handler != b"MJPG":
                                    raise EncodeError("strh.fccHandler=%r 非 MJPG" % (fcc_handler,))
                                if rate <= 0 or scale <= 0:
                                    raise EncodeError("strh scale/rate 非法：%d/%d" % (scale, rate))
                                fps = rate / scale
                                stream_frames = length
                            elif c3 == b"strf":
                                if l3 < 40:
                                    raise EncodeError("strf 过短（%d 字节）" % l3)
                                (bi_size, bi_w, bi_h, planes, bpp) = struct.unpack_from("<IiiHH", data, p3)
                                compression = data[p3 + 16:p3 + 20]
                                if compression != b"MJPG":
                                    raise EncodeError("strf.biCompression=%r 非 MJPG" % (compression,))
                                if bpp != 24:
                                    raise EncodeError("strf.biBitCount=%d 非 24" % bpp)
                                width, height = bi_w, bi_h
            elif ltype == b"movi":
                movi_tag, movi_size = poff, plen          # poff 即 'movi' 标签的绝对偏移
        elif ckid == b"idx1":
            idx1_payload = (poff, plen)

    if width is None or height is None or fps is None:
        raise EncodeError("容器缺 avih/strf 读数，无法确定尺寸或 fps：%s" % p)
    if movi_tag is None:
        raise EncodeError("容器缺 LIST movi：%s" % p)

    entries = _scan_movi(data, movi_tag, movi_size)
    n = len(entries)
    if n == 0:
        raise EncodeError("movi 里没有 00dc 帧块：%s" % p)
    if stream_frames is not None and stream_frames != n:
        raise EncodeError("strh.dwLength(%d) 与 movi 实际帧数(%d) 不符" % (stream_frames, n))

    idx_n = None
    if idx1_payload is not None:
        ioff, ilen = idx1_payload
        if ilen % 16:
            raise EncodeError("idx1 长度 %d 非 16 的倍数" % ilen)
        idx_n = ilen // 16
        if idx_n != n:
            raise EncodeError("idx1 条目数(%d) 与 movi 帧数(%d) 不符" % (idx_n, n))
        # 交叉校验 idx1 的 offset/size 是否真指向 movi 里的帧块。
        # idx1.offset 相对 'movi' 标签起点；帧块 payload 偏移 = movi标签 + offset + 块头(8)
        for i in range(idx_n):
            ckid, _flags, offset, isize = struct.unpack_from("<4sIII", data, ioff + 16 * i)
            if ckid != _FRAME_CKID:
                raise EncodeError("idx1[%d].ckid=%r 非 00dc" % (i, ckid))
            want_poff = movi_tag + offset + 8
            if want_poff != entries[i][1]:
                raise EncodeError("idx1[%d].offset=%d 未指向对应帧块（期望 payload@%d，实际@%d）"
                                  % (i, offset, want_poff, entries[i][1]))
            if isize != entries[i][2]:
                raise EncodeError("idx1[%d].size=%d 与实际块长 %d 不符"
                                  % (i, isize, entries[i][2]))

    # 逐帧校验 JPEG 魔数 + 全帧摘要（回读自证：内容确实在容器里）
    jh = hashlib.sha256()
    for i, (_ck, poff, plen) in enumerate(entries):
        if plen < 4 or data[poff:poff + 2] != b"\xff\xd8":
            raise EncodeError("第 %d 帧不是 JPEG（缺 SOI）" % i)
        jh.update(data[poff:poff + plen])

    out_fps = fps
    if abs(out_fps - round(out_fps)) < 1e-9:
        out_fps = int(round(out_fps))
    return {"frames": n, "size": (int(width), int(height)), "fps": out_fps,
            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "idx1_frames": idx_n, "movi_bytes": sum(e[2] for e in entries),
            "jpeg_sha256": jh.hexdigest()}


def extract_mjpeg_frame(path: str, index: int = 0) -> bytes:
    """取出 AVI 里第 `index` 帧的 JPEG 字节（供自证/验收把帧解回 PNG 看一眼）。"""
    p = os.fspath(path)
    with open(p, "rb") as f:
        data = f.read()
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"AVI ":
        raise EncodeError("不是 AVI：%s" % p)
    (riff_size,) = struct.unpack_from("<I", data, 4)
    movi_tag = movi_size = None
    for ckid, poff, plen in _iter_riff(data, 12, riff_size + 8):
        if ckid == b"LIST" and plen >= 4 and data[poff:poff + 4] == b"movi":
            movi_tag, movi_size = poff, plen
            break
    if movi_tag is None:
        raise EncodeError("容器缺 LIST movi：%s" % p)
    entries = _scan_movi(data, movi_tag, movi_size)
    if isinstance(index, bool) or not isinstance(index, int) or not (0 <= index < len(entries)):
        raise EncodeError("帧号 %r 越界（共 %d 帧）" % (index, len(entries)))
    _ck, poff, plen = entries[index]
    return data[poff:poff + plen]


# ---------------------------------------------------------------- GIF 写出

def write_gif(frames: list, path: str, fps: int, size: tuple, max_colors: int = 128) -> dict:
    """写动画 GIF（PIL）：逐帧 `duration = round(1000/fps)` ms、`loop=0`、调色板量化。

    返回读数：`{"path","bytes","sha256","frames","size","fps","duration_ms",
    "max_colors","loop"}`。失败抛 `EncodeError`。
    """
    _require_pil()
    fps = _check_fps(fps)
    size = _check_size(size)
    frames = _check_frames(frames)
    if isinstance(max_colors, bool) or not isinstance(max_colors, int) or not (2 <= max_colors <= 256):
        raise EncodeError("max_colors 须为 [2, 256] 的整数，得到 %r" % (max_colors,))

    duration_ms = int(round(1000.0 / fps))
    if duration_ms <= 0:
        raise EncodeError("fps=%r 过大：GIF 帧时长会算成 0 ms" % (fps,))

    imgs = []
    for i, item in enumerate(frames):
        im = _load_rgb(item, i, size)
        # 量化到 max_colors（确定性：PIL 的 median-cut 对同一输入稳定）
        imgs.append(im.quantize(colors=max_colors, method=Image.MEDIANCUT, dither=Image.FLOYDSTEINBERG))

    d = os.path.dirname(os.path.abspath(os.fspath(path)))
    if d:
        os.makedirs(d, exist_ok=True)
    imgs[0].save(os.fspath(path), "GIF", save_all=True, append_images=imgs[1:],
                 duration=duration_ms, loop=0, optimize=False, disposal=2)

    with Image.open(os.fspath(path)) as back:
        n_back = int(getattr(back, "n_frames", 1))
        dur_back = int(back.info.get("duration", 0))
    if n_back != len(frames):
        raise EncodeError("GIF 回读帧数 %d != 输入帧数 %d" % (n_back, len(frames)))

    return {"path": os.fspath(path), "bytes": os.path.getsize(path), "sha256": sha256_file(path),
            "frames": len(frames), "size": size, "fps": fps,
            "duration_ms": duration_ms,
            # GIF 的时间精度是 **10 ms（厘秒）**，PIL 会量化：如实回传盘上真值，不假装无损
            "duration_ms_actual": dur_back,
            "duration_quantized": dur_back != duration_ms,
            "max_colors": int(max_colors), "loop": 0}


# ---------------------------------------------------------------- 清单

def write_manifest(out_dir: str, artifacts: dict, extra: dict) -> dict:
    """写 `<out_dir>/manifest.json`（canonical JSON：键排序、缩进 2、末行换行）。

    `artifacts` = `{名字: 路径}`，逐文件算 sha256 + 字节数。
    **时间戳只从 `extra` 取**（不读系统时钟 ⇒ 可复现）。
    缺文件 ⇒ 抛 `EncodeError`（不静默漏算）。返回写出的 manifest dict。
    """
    if not isinstance(artifacts, dict) or not artifacts:
        raise EncodeError("artifacts 须为非空 {名字: 路径}，得到 %r" % (artifacts,))
    if not isinstance(extra, dict):
        raise EncodeError("extra 须为对象，得到 %r" % (type(extra).__name__,))
    od = os.fspath(out_dir)
    os.makedirs(od, exist_ok=True)

    files = {}
    for name, apath in sorted(artifacts.items()):
        if not isinstance(name, str) or not name.strip():
            raise EncodeError("artifacts 的名字须为非空字符串，得到 %r" % (name,))
        p = os.fspath(apath)
        if not os.path.isfile(p):
            raise EncodeError("artifact %r 的文件不存在：%s" % (name, p))
        rel = os.path.relpath(os.path.abspath(p), os.path.abspath(od))
        files[name] = {"path": rel, "bytes": os.path.getsize(p), "sha256": sha256_file(p)}

    # 时间戳字段**只**来自入参（确定性）
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "generated_at": extra.get("generated_at"),
        "files": files,
        "extra": dict(extra),
    }
    write_json(os.path.join(od, "manifest.json"), manifest)
    return manifest


def write_json(path: str, doc: dict) -> str:
    """canonical JSON（键排序、缩进 2、末行换行），返回 sha256(64)。

    与 `contracts.write_json` 同口径（浮点量化到 `contracts.QUANT`）——
    直接复用契约层实现，避免两份口径漂移。
    """
    return contracts.write_json(path, doc)


# ---------------------------------------------------------------- 可选 MP4（交付形态）

def find_ffmpeg() -> str | None:
    """按序探测 ffmpeg 可执行文件；**都找不到 ⇒ 返回 None，绝不抛**。

    ① 环境变量 `MV_FFMPEG`（绝对路径，或可被 `shutil.which` 解析的名字）
    ② `imageio_ffmpeg.get_ffmpeg_exe()`（自带的真二进制；**不得**变成硬依赖）
    ③ `shutil.which("ffmpeg")`（PATH）

    探测失败一律回落下一级，不抛异常 —— 这样缺 ffmpeg 的机器上产线照样能出 AVI+WAV。
    """
    # ① 环境变量显式指定（用户的意图优先级最高）
    env = os.environ.get("MV_FFMPEG")
    if env:
        if os.path.isfile(env) and os.access(env, os.X_OK):
            return env
        found = shutil.which(env)          # 允许 MV_FFMPEG=ffmpeg 这种裸名字
        if found:
            return found
        # 显式给了但不可用 ⇒ 如实回落到下一级（不静默当成"有"，也不抛）
    # ② imageio_ffmpeg 自带的二进制（ImportError / 任何取用异常都不得逃出去）
    try:
        import imageio_ffmpeg
    except ImportError:
        pass
    else:
        try:
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            exe = None
        if exe and os.path.isfile(exe) and os.access(exe, os.X_OK):
            return exe
    # ③ PATH
    found = shutil.which("ffmpeg")
    if found:
        return found
    return None


def _ffmpeg_version_line(exe: str) -> str:
    """`ffmpeg -version` 的首行（用于登记"用的是哪个 ffmpeg"）。"""
    try:
        cp = subprocess.run([exe, "-version"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        raise EncodeError("无法执行 ffmpeg -version（%s）：%s" % (exe, e))
    first = (cp.stdout or cp.stderr or "").strip().splitlines()
    if cp.returncode != 0 or not first:
        raise EncodeError("ffmpeg -version 失败（exit %d）：%s"
                          % (cp.returncode, (cp.stderr or "").strip()[-400:]))
    return first[0].strip()


def _ffmpeg_probe_streams(exe: str, path: str) -> str:
    """`ffmpeg -i <file>`（无输出）拿到容器信息；ffmpeg 对"无输出"返回 1 是正常的。"""
    try:
        cp = subprocess.run([exe, "-hide_banner", "-i", path],
                            capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError) as e:
        raise EncodeError("无法探测 MP4（%s）：%s" % (path, e))
    info = (cp.stderr or "") + (cp.stdout or "")
    if "Input #0" not in info:
        raise EncodeError("ffmpeg 认不出合成产物：%s" % info.strip()[-600:])
    return info


def write_mp4(frames: list, wav: str, path: str, fps: int, size: tuple) -> dict | None:
    """**可选**出口：把帧序列 + WAV 音轨合成 MP4（H.264 + AAC）。

    - 探测不到 ffmpeg ⇒ 返回 `None`（**不抛**）：调用方保留 AVI + WAV 路径，产线不受影响；
    - 有 ffmpeg 但合成/自证失败 ⇒ 抛 `EncodeError`（不静默降级、不留半个坏文件）；
    - 合成后**自证**：`ffmpeg -v error -i <out> -f null -` 必须 exit 0，且产物必须真带
      H.264 视频流 + AAC 音轨（从 `ffmpeg -i` 的流信息里核对，不靠猜）。

    返回：`{"path","bytes","sha256","frames","size","fps","ffmpeg","has_audio",
    "video_codec","audio_codec","deterministic": False}`。

    **确定性**：H.264 编码不保证逐字节确定（编码器多线程 / 版本差异 / 容器时间戳），
    故 `deterministic` 如实为 `False`；**确定性以 AVI / 帧序列为准，MP4 只是交付形态**。
    这里**不**为求确定性加 `-threads 1` 之类会改变画质的参数。
    """
    exe = find_ffmpeg()
    if exe is None:
        return None                        # 可选出口：缺 ffmpeg 绝不失败

    _require_pil()                         # 校验帧尺寸 / 落 PIL 帧都要它
    fps = _check_fps(fps)
    size = _check_size(size)
    frames = _check_frames(frames)
    w, h = size
    if w % 2 or h % 2:
        # libx264 + yuv420p 的硬性约束：色度 2x2 下采样要求宽高皆偶。宁可报错也不偷偷裁/补。
        raise EncodeError("H.264/yuv420p 要求宽高均为偶数，得到 %r" % (size,))
    wav_p = os.fspath(wav)
    if not os.path.isfile(wav_p):
        raise EncodeError("音轨 WAV 不存在：%s" % wav_p)

    out = os.fspath(path)
    out_dir = os.path.dirname(os.path.abspath(out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    version = _ffmpeg_version_line(exe)

    tmp = tempfile.mkdtemp(prefix="mvmp4_")
    try:
        # 把帧按 image2 序列命名落进临时目录（优先符号链接，避免复制 GB 级 PNG）
        for i, item in enumerate(frames, 1):
            dst = os.path.join(tmp, "%06d.png" % i)
            if isinstance(item, (str, os.PathLike)):
                src = os.path.abspath(os.fspath(item))
                if not os.path.isfile(src):
                    raise EncodeError("第 %d 帧文件不存在：%s" % (i - 1, src))
                with Image.open(src) as im:          # 只读头部，不解码全图
                    if im.size != size:
                        raise EncodeError("第 %d 帧 %s 尺寸 %r 与 size %r 不一致"
                                          % (i - 1, src, tuple(im.size), size))
                try:
                    os.symlink(src, dst)
                except OSError:
                    shutil.copyfile(src, dst)
            else:
                _load_rgb(item, i - 1, size).save(dst, "PNG")

        pattern = os.path.join(tmp, "%06d.png")
        cmd = [exe, "-hide_banner", "-nostdin", "-y",
               "-framerate", str(fps), "-start_number", "1", "-i", pattern,
               "-i", wav_p,
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps),
               "-c:a", "aac", "-b:a", "192k",
               "-shortest", "-movflags", "+faststart",
               out]
        try:
            cp = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        except (OSError, subprocess.SubprocessError) as e:
            raise EncodeError("调用 ffmpeg 合成 MP4 失败：%s" % e)
        if cp.returncode != 0:
            raise EncodeError("ffmpeg 合成 MP4 失败（exit %d）：%s"
                              % (cp.returncode, (cp.stderr or "").strip()[-900:]))
        if not os.path.isfile(out) or os.path.getsize(out) == 0:
            raise EncodeError("ffmpeg 报成功但 MP4 不存在或为空：%s" % out)

        # ---- 自证 1：真解码一遍（-f null -）必须干净退出 ----
        try:
            cp2 = subprocess.run([exe, "-v", "error", "-i", out, "-f", "null", "-"],
                                 capture_output=True, text=True, timeout=1800)
        except (OSError, subprocess.SubprocessError) as e:
            raise EncodeError("MP4 回读自证无法执行：%s" % e)
        if cp2.returncode != 0:
            raise EncodeError("MP4 回读自证失败（exit %d）：%s"
                              % (cp2.returncode, (cp2.stderr or "").strip()[-900:]))

        # ---- 自证 2：产物必须真带 H.264 视频流 + AAC 音轨 ----
        info = _ffmpeg_probe_streams(exe, out)
        if "Video: h264" not in info:
            raise EncodeError("产物视频流不是 H.264：%s" % info.strip()[-600:])
        if "Audio:" not in info:
            raise EncodeError("产物没有音轨（期望 AAC）：%s" % info.strip()[-600:])
        audio_codec = "aac" if "Audio: aac" in info else "unknown"

        return {"path": out, "bytes": os.path.getsize(out), "sha256": sha256_file(out),
                "frames": len(frames), "size": size, "fps": fps,
                "ffmpeg": version, "has_audio": True,
                "video_codec": "h264", "audio_codec": audio_codec,
                # 如实标注：H.264 不保证逐字节确定；确定性以 AVI/帧序列为准
                "deterministic": False}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
