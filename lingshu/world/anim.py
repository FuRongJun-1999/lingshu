# Copyright 2026 灵枢 (Lingshu) · MIT
"""anim —— 帧序列 → **真动画**（GIF）；并把「动画」与「胶片条」在名字上分开。

背景（实测，非推断）
--------------------
本仓此前的"动画"只有一处：`skeleton3d.__main__` 里打印
`walk animation 6 frames → …/skeleton_walk.png`，而该文件是
`Image.new(frame_w * 6, frame_h)` + `paste` 拼出来的**一张横排静态图**。
全仓 `grep` 实测没有 GIF / `save_all` / 帧序列导出 ⇒ 名字与事实不符，且没有
任何可复用的动画出口。本模块补两件事：**真动画出口**（`to_gif` / `from_paths`）
与**把胶片条正名为胶片条**（`to_strip`）。

设计口径（三条硬约束）
----------------------
1. **纯标准库 + PIL，且 PIL 只在函数内 import**：`world` 的位图面依赖 pillow
   extra；模块级 `from PIL import …` 会让 `import lingshu.world.anim` 连带需要
   pillow（与 `world3d.py` / `silhouette3d.py` / `skeleton3d.py` 同口径）。
2. **逐字节确定性**：同一 `frames` 输入 ⇒ 同一 GIF 字节。做法是**显式钉住调色板**
   （把所有帧拼成一张竖排图量化**一次**得到共享调色板，再逐帧按该调色板量化；
   量化参数 `MEDIANCUT` + `dither=NONE` 全部写死），并且 `optimize=False`
   （PIL 实测 `optimize` 会改变输出字节）。不依赖进程随机 / 时间 / 路径。
3. **不静默**：任何"会产出与请求不符的文件"的输入都**显式 raise**，且写完**回读自校验**
   （见下）。绝不因为"出错就少写一帧"而留下坏文件 —— `to_gif` 先写临时文件，
   自校验通过才 `os.replace` 到目标路径。

⚠ 必须知道的 PIL 行为（本模块的存在理由之一，实测于 Pillow 12.3.0）
--------------------------------------------------------------------
`GifImagePlugin._write_multiple_frames`（`:697-704`）在**相邻两帧像素完全相同**时
会**跳过**后一帧，并把它的时长**累加**到前一帧：
```python
delta, bbox = _getbbox(previous_im, im_frame)
if not bbox:                                  # 与前帧逐像素相同
    if encoderinfo.get("duration"):
        im_frames[-1].encoderinfo["duration"] += encoderinfo["duration"]
    continue                                  # ← 无条件 continue，没有 encoderinfo 开关能阻止
```
紧接着（`:771`）`if len(im_frames) == 1: … return False` ⇒ 只剩 1 帧时**回退成单帧 GIF**。
即 4 张相同的帧 → `n_frames == 1`、`duration == 400`：`optimize` / `disposal` /
duration 列表都改变不了这个分支（这是**机制定论**，不是现象观测）。这与契约里
"`frames(renderer, n)` 不给 `mutate` ⇒ 同状态渲 n 次（合法，静止对照）"正面冲突：
返回 `{"frames": 4}` 而文件只有 1 帧 = **观测面与文件不符**。
故 `to_gif` **写完回读**：`n_frames != len(frames)` ⇒ `ValueError`，并在消息里
点明成因与修法（用 `mutate` 让相邻帧真正不同）。

时长粒度：GIF 的 delay 是 **16 位厘秒**字段 ⇒ 毫秒只能表达 **10 的整数倍**，
上限 **655350 ms**。`duration_ms < 10` / **非 10 的倍数** / 超上限一律显式 raise
（0 或 1 会被`读回成` None/0，负数或 ≥655360 会让 PIL 抛裸 `struct.error`；
**实测**：请求 `duration_ms=105` ⇒ PIL 照写、读回 `100` —— 与其静默改写时长，
不如显式 raise，这样 `info["duration"] == duration_ms` 对**所有成功调用恒成立**）。
`loop` 同样是 16 位：取 `0..65535`（`0` = 无限循环）。

用法
----
    from lingshu.world.anim import frames, to_gif, to_strip, save_frames, from_paths

    sk = human_skeleton(center=(0, 0.85, 5), height=1.7)

    def pose(i, sk):                    # mutate：第 i 帧渲染**之前**调用
        sk.joints["thigh_l"].rot = (0.5 * math.sin(i / 6 * 2 * math.pi), 0, 0)

    fr = frames(sk, 6, mutate=pose, size=(240, 240))
    obs = to_gif(fr, "walk.gif", duration_ms=100, loop=0)   # 真动画
    to_strip(fr).save("walk_sheet.png")                     # 胶片条（静态，别叫动画）
    save_frames(fr, "out/", prefix="walk")                  # 逐帧落盘
    from_paths(["out/walk_0000.png", "out/walk_0001.png"], out="back.gif")

通用性：任何提供 `.render(w, h, **kw) -> PIL.Image` 的对象都能用
（`World3D` / `Silhouette3D` / `Skeleton3D` / `SceneModel` 都有这个面）。
"""
from __future__ import annotations

import os
import tempfile
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = ["frames", "to_gif", "to_strip", "save_frames", "from_paths"]

#: GIF 时长字段是 16 位**厘秒** ⇒ 毫秒只表达 10 的整数倍，上限 65535 厘秒。
_GIF_MS_STEP = 10
#: 65535 厘秒 —— 超过它 PIL 会抛裸 `struct.error: 'H' format requires 0 <= number <= 65535`。
_GIF_MS_MAX = 65535 * 10
#: GIF 的 loop 字段同为 16 位无符号（0 = 无限循环）。
_GIF_LOOP_MAX = 65535
#: GIF 单帧调色板上限（格式硬限制）。
_GIF_COLORS = 256
#: 共享调色板「竖排拼图」的像素预算：超出就退回逐帧自适应调色板（仍确定性，
#: 只是没有跨帧共享调色板）。防超大输入（如 2000×2000×60）把内存打爆。
_PALETTE_BUDGET = 16_000_000


# ---------------------------------------------------------------------------
# 内部工具（PIL 一律函数内 import）
# ---------------------------------------------------------------------------


def _lazy_image():
    """惰性取 PIL.Image —— 保证 `import lingshu.world.anim` 不需要 pillow。"""
    from PIL import Image  # noqa: PLC0415 —— 刻意函数内 import（见模块 docstring ①）

    return Image


def _as_frame_list(seq: Any, where: str) -> List[Any]:
    """把入参规整为 PIL.Image 列表；不是帧序列就显式报错（不静默吞）。"""
    Image = _lazy_image()
    if isinstance(seq, Image.Image):
        raise TypeError(
            f"{where}: 传入的是一张图片，不是帧序列 —— 单帧请用 [img]" f"（本函数要的是帧序列）"
        )
    try:
        items = list(seq)
    except TypeError as exc:  # noqa: PERF203 —— 非可迭代入参要给出人话报错
        raise TypeError(f"{where}: frames 必须是 PIL.Image 的可迭代序列（得到 {type(seq).__name__}）") from exc
    out = []
    for i, im in enumerate(items):
        if not isinstance(im, Image.Image):
            raise TypeError(
                f"{where}: 第 {i} 项不是 PIL.Image，而是 {type(im).__name__}（不做隐式转换）"
            )
        out.append(im)
    return out


def _check_uniform_size(seq: List[Any], where: str) -> Tuple[int, int]:
    """所有帧必须同画幅 —— PIL 对不一致尺寸**不报错**，会静默只写 1 帧。"""
    size = seq[0].size
    for i, im in enumerate(seq):
        if im.size != size:
            raise ValueError(
                f"{where}: 帧尺寸不一致 —— 第 0 帧 {size}，第 {i} 帧 {im.size}；"
                f"GIF 只接受同一画幅（PIL 对这种情况不报错、会静默只写 1 帧）"
            )
    return size


def _check_duration(duration_ms: Any) -> int:
    if isinstance(duration_ms, bool) or not isinstance(duration_ms, int):
        raise TypeError(f"to_gif(): duration_ms 必须是整数毫秒，得到 {type(duration_ms).__name__}")
    if duration_ms < _GIF_MS_STEP:
        raise ValueError(
            f"to_gif(): duration_ms={duration_ms} 不可表达 —— GIF 以**厘秒**存时长，"
            f"最小 {_GIF_MS_STEP}ms；0/1 会被读回成 None/0（静默失真），故显式拒绝"
        )
    if duration_ms > _GIF_MS_MAX:
        raise ValueError(
            f"to_gif(): duration_ms={duration_ms} 超出 GIF 上限 {_GIF_MS_MAX}ms"
            f"（16 位厘秒字段，超出 PIL 抛裸 struct.error），故显式拒绝"
        )
    if duration_ms % _GIF_MS_STEP:
        raise ValueError(
            f"to_gif(): duration_ms={duration_ms} 不是 {_GIF_MS_STEP} 的整数倍 —— "
            f"GIF 以厘秒存时长，读回会变成 {duration_ms // _GIF_MS_STEP * _GIF_MS_STEP}ms；"
            f"与其静默改写时长，不如显式拒绝"
        )
    return duration_ms


def _check_loop(loop: Any) -> int:
    if isinstance(loop, bool) or not isinstance(loop, int):
        raise TypeError(f"to_gif(): loop 必须是整数（0 = 无限循环），得到 {type(loop).__name__}")
    if loop < 0 or loop > _GIF_LOOP_MAX:
        raise ValueError(
            f"to_gif(): loop={loop} 不可表达 —— 合法范围 0..{_GIF_LOOP_MAX}（0 = 无限循环）"
        )
    return loop


def _quantize(seq: List[Any]) -> List[Any]:
    """把帧序列显式钉到调色板（确定性）：优先跨帧共享调色板。

    ① 竖排拼图量化一次 → 共享调色板 → 逐帧按该调色板量化（无跨帧闪色）；
    ② 拼图超出 `_PALETTE_BUDGET` ⇒ 退回逐帧自适应（同样写死 MEDIANCUT/dither=NONE，
       同输入仍逐字节确定）。alpha 在 `convert("RGB")` 处丢弃（GIF 无 alpha 通道）。
    """
    Image = _lazy_image()
    rgb = [im.convert("RGB") for im in seq]
    w, h = rgb[0].size
    if w * h * len(rgb) <= _PALETTE_BUDGET:
        sheet = Image.new("RGB", (w, h * len(rgb)))
        for i, im in enumerate(rgb):
            sheet.paste(im, (0, i * h))
        ref = sheet.quantize(
            colors=_GIF_COLORS, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
        )
        pal = Image.new("P", (1, 1))
        pal.putpalette(ref.getpalette())
        return [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in rgb]
    return [
        im.quantize(colors=_GIF_COLORS, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        for im in rgb
    ]


def _verify_gif(path: str, want_frames: int, want_duration: int, want_loop: int) -> Dict[str, Any]:
    """回读自校验：文件必须与请求一致，否则 raise（消息讲清成因）。"""
    Image = _lazy_image()
    with Image.open(path) as rb:
        got_frames = rb.n_frames
        got_size = tuple(rb.size)
        got_duration = rb.info.get("duration")
        got_loop = rb.info.get("loop")
    if got_frames != want_frames:
        raise ValueError(
            f"to_gif(): 期望 {want_frames} 帧，回读到 {got_frames} 帧 —— "
            f"GifImagePlugin._write_multiple_frames(:697-704) 对**与前帧逐像素相同**的帧"
            f"无条件 continue（跳过并把时长累加到前帧），只剩 1 帧时(:771)回退单帧；"
            f"optimize/disposal/duration 列表都挡不住。修法：用 mutate 让相邻帧真正不同，"
            f"或改用 to_strip/save_frames。本条是**写后回读自校验**，不是猜测"
        )
    if got_duration != want_duration:
        raise ValueError(
            f"to_gif(): 回读时长 {got_duration}ms != 请求 {want_duration}ms（GIF 时长以厘秒存）"
        )
    if got_loop != want_loop:
        raise ValueError(f"to_gif(): 回读 loop={got_loop} != 请求 {want_loop}")
    return {
        "path": path,
        "frames": got_frames,
        "size": got_size,
        "duration_ms": got_duration,
        "loop": got_loop,
        "bytes": os.path.getsize(path),
    }


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


# ---------------------------------------------------------------------------
# 公开面（契约：见 _pr-prep/ANIM-BRIEF.md §1）
# ---------------------------------------------------------------------------


def frames(
    renderer: Any,
    n: int,
    mutate: Optional[Callable[[int, Any], None]] = None,
    size: Tuple[int, int] = (240, 240),
    **kw: Any,
) -> List[Any]:
    """逐帧渲染 → `List[PIL.Image]`（**帧序列**，不是动画文件，也不是胶片条）。

    · `n >= 1` 帧；`mutate(i, renderer)` 在第 i 帧渲染**之前**调用（摆姿态 / 推进世界）。
    · 不给 `mutate` ⇒ 同状态渲 n 次。**合法**（用于"静止"对照），但那种帧序列交给
      `to_gif` 时相邻帧完全相同 ⇒ 会被 GIF 编码器合并 ⇒ `to_gif` 回读自校验**显式 raise**
      （不假装成 n 帧动画）。
    · `n < 1` ⇒ `ValueError`（不做 0 帧输出）。
    · renderer 必须提供 `.render(w, h, **kw) -> PIL.Image`；返回非 PIL.Image ⇒ `TypeError`。
    """
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError(f"frames(): n 必须是整数，得到 {type(n).__name__}")
    if n < 1:
        raise ValueError(f"frames(): n 必须 >= 1，得到 {n}（不做 0 帧输出）")
    try:
        w, h = int(size[0]), int(size[1])
    except (TypeError, IndexError, ValueError) as exc:
        raise TypeError(f"frames(): size 必须是 (宽, 高) 二元组，得到 {size!r}") from exc
    if w < 1 or h < 1:
        raise ValueError(f"frames(): size 必须为正，(宽, 高) = {size!r}")
    render = getattr(renderer, "render", None)
    if not callable(render):
        raise TypeError(
            f"frames(): renderer 需要提供 render(w, h, **kw) -> PIL.Image，"
            f"得到 {type(renderer).__name__}"
        )
    Image = _lazy_image()
    out: List[Any] = []
    for i in range(n):
        if mutate is not None:
            mutate(i, renderer)
        img = render(w, h, **kw)
        if not isinstance(img, Image.Image):
            raise TypeError(
                f"frames(): 第 {i} 帧 render() 返回了 {type(img).__name__}，不是 PIL.Image"
            )
        out.append(img)
    return out


def to_gif(
    frames: Sequence[Any],
    path: Any,
    duration_ms: int = 100,
    loop: int = 0,
) -> Dict[str, Any]:
    """帧序列 → **真动画** GIF。返回**回读到的**观测面：

    `{"path", "frames", "size", "duration_ms", "loop", "bytes"}`。

    · `frames` 为空 ⇒ `ValueError`（不做 0 帧文件）；尺寸不一致 ⇒ `ValueError`（PIL 会静默丢帧）。
    · `duration_ms` **必须是 10 的整数倍**且 ≤ 655350；`loop` ∈ 0..65535 —— 越界显式 raise。
      为什么卡 10 的倍数：GIF 的 delay 是**厘秒**字段，PIL 照写、读回会变（实测请求 `105`
      读回 `100`）⇒ 与其静默改写时长，不如拒绝；这样"回读 `info["duration"] == duration_ms`"
      对所有成功调用恒成立。
    · **写完回读自校验**：帧数/尺寸/时长/loop 与请求不符 ⇒ `ValueError`，且先写临时文件、
      校验通过才 `os.replace` ⇒ 目标路径上**永不出现**半成品/坏文件。
    · 确定：同输入两次导出逐字节相同（调色板与 `optimize=False` 都显式钉住）。
    """
    path = os.fspath(path)
    seq = _as_frame_list(frames, "to_gif()")
    if not seq:
        raise ValueError("to_gif(): frames 为空 —— 不做 0 帧文件")
    _check_uniform_size(seq, "to_gif()")
    duration_ms = _check_duration(duration_ms)
    loop = _check_loop(loop)

    _ensure_parent(path)
    # 先写同目录的临时文件，回读自校验通过才 os.replace 到目标 ⇒ 目标路径上永不出现坏文件。
    fd, tmp = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".", suffix=".tmp",
        dir=os.path.dirname(os.path.abspath(path)),
    )
    os.close(fd)
    pal = _quantize(seq)
    try:
        pal[0].save(
            tmp,
            format="GIF",
            save_all=True,
            append_images=pal[1:],
            duration=duration_ms,
            loop=loop,
            optimize=False,  # 显式钉住：PIL 实测 optimize 会改变输出字节
            disposal=2,
        )
        obs = _verify_gif(tmp, len(seq), duration_ms, loop)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:  # pragma: no cover —— 清理失败不该盖住原始异常
                pass
    obs["path"] = path
    return obs


def to_strip(
    frames: Sequence[Any],
    cols: Optional[int] = None,
    background: Tuple[int, int, int] = (24, 28, 46),
) -> Any:
    """帧序列 → **胶片条**（一张静态图，横排/网格）。

    名字必须是**胶片条**，**不是**动画：产物是 `Image.new(...)` + `paste` 拼出来的
    单张 RGB 图 —— 落盘成 PNG/GIF 都是 `n_frames == 1`。

    返回值上的 `n_frames == 1` 是**本函数显式补的**（仅当该属性**不存在**时才补，
    不覆盖 Pillow 将来可能自带的同名属性），它与 `Image.open(<落盘路径>).n_frames`
    **同义** —— 该对象就是单帧位图。之所以要补：Pillow 12 的 `Image.Image` **基类没有**
    `n_frames`（该属性由各插件在 `open()` 时设置，见 `Image.py:2740/2822` 的文档），
    不补的话"胶片条 vs 动画"这条读回判据就只能靠落盘再开才算得出来。
    判据要能把"胶片条"和"动画"分开，见守卫 E 组。
    · `cols=None` ⇒ 单行横排（`cols = len(frames)`）。
    · `frames` 为空 / 尺寸不一致 / `cols < 1` ⇒ `ValueError`。
    """
    Image = _lazy_image()
    seq = _as_frame_list(frames, "to_strip()")
    if not seq:
        raise ValueError("to_strip(): frames 为空 —— 做不出胶片条")
    w, h = _check_uniform_size(seq, "to_strip()")
    if cols is None:
        cols = len(seq)
    if isinstance(cols, bool) or not isinstance(cols, int) or cols < 1:
        raise ValueError(f"to_strip(): cols 必须是 >= 1 的整数，得到 {cols!r}")
    rows = -(-len(seq) // cols)
    try:
        bg = tuple(int(c) for c in background)
        if len(bg) != 3:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise TypeError(f"to_strip(): background 必须是 (r, g, b)，得到 {background!r}") from exc
    sheet = Image.new("RGB", (w * cols, h * rows), bg)
    for i, im in enumerate(seq):
        sheet.paste(im.convert("RGB"), ((i % cols) * w, (i // cols) * h))
    # Pillow 12 的 `Image.Image` 基类**没有** `n_frames`（该属性由插件在 open() 时设置）。
    # 仅当缺失时才补 —— 不覆盖 Pillow 将来可能自带的同名属性。补出来的值与
    # `Image.open(<落盘路径>).n_frames` **同义**：这是一张单帧位图。
    if not hasattr(sheet, "n_frames"):
        sheet.n_frames = 1
    return sheet


def save_frames(
    frames: Sequence[Any],
    out_dir: Any,
    prefix: str = "frame",
    ext: str = "png",
) -> List[str]:
    """逐帧落盘，返回路径列表（**按序**，第 i 项 = 第 i 帧）。

    · 文件名 `<prefix>_<i:04d>.<ext>`；`out_dir` 不存在则创建。
    · `frames` 为空 ⇒ `ValueError`（不做 0 帧输出）。
    · 逐帧原样保存（不强制转 RGB ⇒ RGBA/P 帧的保真度不受影响）。
    """
    _lazy_image()  # 早失败：缺 pillow 时在写目录之前就报错
    seq = _as_frame_list(frames, "save_frames()")
    if not seq:
        raise ValueError("save_frames(): frames 为空 —— 不做 0 帧输出")
    out_dir = os.fspath(out_dir)
    ext = str(ext).lstrip(".").lower() or "png"
    os.makedirs(out_dir, exist_ok=True)
    paths: List[str] = []
    for i, im in enumerate(seq):
        p = os.path.join(out_dir, f"{prefix}_{i:04d}.{ext}")
        im.save(p)
        paths.append(p)
    return paths


def from_paths(
    paths: Sequence[Any],
    duration_ms: int = 100,
    loop: int = 0,
    out: Any = None,
) -> Dict[str, Any]:
    """已落盘的帧（PNG 序列）→ GIF（把既有产物接起来）。

    · 顺序**按给定顺序**（不排序）—— 帧序即动画时序。
    · 每张图取**第 0 帧**（`copy()`），故多帧 GIF 当输入时只取首帧。
    · `out=None` ⇒ 落在 `paths[0]` 同目录的 `frames.gif`。
    · `paths` 为空 ⇒ `ValueError`；文件不存在 ⇒ `FileNotFoundError`。
    · 返回面与 `to_gif` **同形**（同一条回读自校验路径）。
    """
    Image = _lazy_image()
    plist = [os.fspath(p) for p in paths]
    if not plist:
        raise ValueError("from_paths(): paths 为空 —— 没有帧可接")
    for p in plist:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"from_paths(): 帧文件不存在：{p}")
    seq: List[Any] = []
    for p in plist:
        with Image.open(p) as im:
            im.load()
            seq.append(im.copy())
    if out is None:
        out = os.path.join(os.path.dirname(plist[0]), "frames.gif")
    return to_gif(seq, out, duration_ms=duration_ms, loop=loop)
