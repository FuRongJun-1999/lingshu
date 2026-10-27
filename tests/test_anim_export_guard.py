# -*- coding: utf-8 -*-
"""守卫 · 真动画（帧序列 / GIF）vs 胶片条 · `lingshu/world/anim.py` + `skeleton3d` 演示
============================================================================
缺陷 / 能力缺口（取证 = 修复前上游 `wm-verify/lingshu-orig/lingshu/world/`，与
`wm-verify/lingshu/` 逐字相同；`skeleton3d.py` md5 `3315d78cef8f4dd1f2b22cf033ab742e`）：
  ① **没有真动画出口**：全仓 `grep` 实测没有 GIF / `save_all` / 帧序列导出 —— 只有
     `skeleton3d.py:338` 打印 `walk animation 6 frames → …/skeleton_walk.png`，
     而该文件是 `Image.new(frame_w * 6, frame_h)` + `paste` 拼出来的**一张静态横排图**
     ⇒ **名字与事实不符**（"animation"其实是胶片条），且不是可复用 API、还往仓内写文件。
  ② **没有"胶片条"这个正确的名字**：把静态拼图叫动画，调用方无法区分"能动"与"不能动"。
  ③ **PIL 的三个静默陷阱**（机制定论，Pillow 12.3.0 源码，非现象观测）：
     · `GifImagePlugin._write_multiple_frames` `:697-704` 对**与前帧逐像素相同**的帧
       **无条件 `continue`**（跳过该帧 + 把时长累加到前一帧），紧接着 `:771`
       `if len(im_frames) == 1: … return False` ⇒ 只剩 1 帧时**回退成单帧 GIF**。
       `optimize` / `disposal` / 逐帧 duration 列表都改变不了这个分支。
       实测：4 张相同帧 → `n_frames == 1`、`duration == 400`（100 被累加）。
     · **帧尺寸不一致不报错**：PIL 直接只写 1 帧（静默丢帧）。
     · **时长边界静默/裸异常**：`duration_ms` 0/1 → 读回 `None`/0（GIF 以厘秒存）；
       负数或 ≥655360 → 抛**裸 `struct.error: 'H' format requires 0 <= number <= 65535`**。
     一个"写完就自报请求值"的导出器，会**自报 4 帧而文件只有 1 帧**、自报 100ms
     而文件是 400ms ⇒ 观测面与文件不符（正是本件要治的"名不副实"病）。

修法（新增 5 原语 + 演示改名并追加真 GIF）：
  ① `frames(renderer, n, mutate=None, size=(240,240), **kw)`：通用逐帧渲染（任何有
     `.render(w, h, **kw) -> PIL.Image` 的对象）；`mutate(i, renderer)` 在第 i 帧
     **渲染之前**调用；`n < 1` ⇒ `ValueError`。
  ② `to_gif(frames, path, duration_ms=100, loop=0)`：**真动画**出口。显式钉住调色板
     （全部帧竖排拼图量化**一次**得跨帧共享调色板；`MEDIANCUT` + `dither=NONE` 写死）
     与 `optimize=False` ⇒ 同输入**逐字节确定**；**先写临时文件 → 写完回读自校验 →
     `os.replace`** ⇒ 帧数/尺寸/时长/loop 与请求不符就 `ValueError`，目标路径上
     **永不出现**半成品或坏文件；返回 dict 的 `frames`/`duration_ms` 是**回读值**。
  ③ `to_strip(frames, cols=None, background=…)`：**胶片条**（一张静态单帧图；名字与
     事实对齐；返回值显式带 `n_frames == 1`）。
  ④ `save_frames(...)` / `from_paths(...)`：逐帧落盘 + 把既有 PNG 序列接成 GIF。
  ⑤ `skeleton3d.__main__`：只改那句 `print` 的措辞（`walk animation` → `walk filmstrip`、
     局部量 `anim_path` → `strip_path`）+ **追加**真 GIF 块；**胶片条 PNG 像素不变**
     （文件字节随打包的 zlib 实现而变 ⇒ 按**像素**冻结，见文末"覆盖边界"）。

判据来源：冻结契约 `_pr-prep/ANIM-BRIEF.md` §1（5 原语签名 / PIL 函数内惰性 import /
逐字节确定性 / 读回可验 / 不静默）、§2（写面分工）、§3（最小性 / 既有 PNG 字节不变 /
守卫用 `tmp_path`、`coverage_map_lint` 要求 `anim` 被测试引用）、§4（复核口径）；
Lead 裁决（2026-10-13，BRIEF 之外的口径补丁，已并入契约执行）：`to_gif` **必须写后回读
自校验**；`duration_ms >= 10` 且 ≤ 655350（厘秒粒度）；返回 dict 不得自报与文件不符的值；
PIL 机制行号 = Pillow 12.3.0 `GifImagePlugin.py:697-704` 与 `:771`；
基线读数 = `_pr-prep/ANIM-BASELINE.md`（Lead 在任何人动手前实测）。

断言组（把修复抽掉 ⇒ 哪组红；回退实测读数见 `_pr-prep/animwork/REVERT-RED.md`）：
  A [帧序列] `frames()`：n 帧 / 尺寸生效 / `mutate(i, renderer)` **在第 i 帧渲染之前**
    按序调用（用带事件日志的假渲染器证明顺序是「摆姿态 → 渲染」，而非"渲染完再改"）；
    不给 `mutate` ⇒ n 帧同状态（静止对照）；`n<1` / `size` 非法 ⇒ raise；renderer 无
    `.render` 或返回非 `PIL.Image` ⇒ raise。
    ⇒ 把 `frames` 写成"先渲染 n 帧再统一改姿态"、或吞掉 `n<1`：A 红。
  B [GIF 是动画] `to_gif`：`Image.open(path)` 的 `n_frames == len(frames)`、`size` == 请求、
    `info["duration"] == duration_ms`、`info["loop"] == loop`、返回 dict 与回读一致、
    `bytes == os.path.getsize(path)`；逐帧像素 MAE 如实记录（本件线稿帧实测 0.0）。
    ⇒ 改动 `save_all`/调色板/`optimize` 导致少帧或改画幅：B 红。
  C [确定性] 同输入两次 `to_gif` ⇒ sha256 相同；**跨进程**（`PYTHONHASHSEED` 0 与 1）
    两次 ⇒ 与进程内哈希三者相同；`from_paths(save_frames(fr))` 与 `to_gif(fr)` 逐字节相同；
    调色板预算回退路径（monkeypatch `_PALETTE_BUDGET`）仍逐字节相同。
    ⇒ 去掉显式调色板钉法（吃默认 dither/自适应）或让输出吃进程随机：C 红。
  D [不静默] 空帧 / 尺寸不一致 / `duration_ms ∈ {0,1,5,11,105,655360,10**9}` /
    `loop ∈ {-1,65536}` ⇒ 显式 raise，且**目标路径不产生文件**；`duration_ms` 必须**是 10 的
    整数倍**（D8 先实测 PIL 把 `105` 写成并读回 `100`，再断言 `to_gif` 对 105 显式 raise）
    且上限 `655350` 可表达（D9）；连续相同帧（含纯色帧）
    ⇒ `to_gif` 回读自校验 raise、**不留坏文件**；单帧（退化）⇒ 合法且 `n_frames == 1`；
    RGBA 帧 ⇒ 可导出（alpha 在 `convert("RGB")` 处丢弃，如实登记）。
    ⇒ 去掉回读自校验（相同帧静默写成 1 帧、返回 `{"frames": 4}`）或去掉边界校验：D 红。
  E [胶片条 ≠ 动画] 同一条帧序列：`to_strip` ⇒ **单帧静态图**（返回值 `n_frames == 1`、
    落盘再读回 `n_frames == 1`、尺寸 = 网格尺寸），而 `to_gif` ⇒ `n_frames == n > 1`。
    返回值上那个 `n_frames` 是 `to_strip` **显式补的**（仅当缺失时补）——E1 先断言 Pillow 12
    的基类 `Image.Image` **确实没有**该属性（补法的事实依据），再断言返回值与
    `Image.open(落盘路径).n_frames` **同义**。
    ⇒ 判据能**分开**"胶片条"与"动画"；把 `to_strip` 写成返回帧列表/多帧 GIF：E 红。
  F [演示回归] 在 `tmp_path` 的包副本里跑 `python -m lingshu.world.skeleton3d`：
    · 胶片条 `skeleton_walk.png` **像素 == 改前循环的冻结复算**（本文件内嵌的逐字副本）
      且**像素级金标**（解码后逐像素 sha256 == 改前基线）+ PNG 结构（IHDR 维度/位深/
      颜色类型、单 IDAT、IEND）；`skeleton_demo.png` 同理；
    · **同平台复算一致性**：再跑一遍演示，两次的 PNG/GIF 产物逐字节相同
      （替代"文件字节金标"这一格——PNG 文件字节随打包的 zlib 实现而变，
      `PIL.__version__` 抓不住，见文末"覆盖边界"）；
    · 追加的 `skeleton_walk.gif` 是**真动画**（GIF 魔数合法、`n_frames == 12`、
      duration 100、loop 0）；
    · stdout 措辞已对齐（提及 `.png` 的那行是 filmstrip 且不含 animation；提及 `.gif`
      的那行含 animation）；
    · **仓内 `lingshu/data` 不被本守卫触碰**（逐文件 size+mtime 快照前后一致）。
    ⇒ 改演示里胶片条的渲染/顺序、或让两次导出吃进程随机、或把 print 措辞退回
     "胶条也叫 animation"、或让演示写回仓内：F 红。
  G [惰性 PIL] `anim.py` 的 AST 里**没有**模块级 PIL import、且**有**函数级 PIL import；
    子进程里屏蔽 `PIL` 后仍能 `import lingshu.world.anim`（`world` 的位图面依赖 pillow
    extra，模块级 import 会让纯 import 也连带需要 pillow）。
    ⇒ 把 `from PIL import Image` 提到模块级：G 红。

覆盖边界（如实，未验的部分）：
  · **不验**"GIF 在播放器里看起来是否动"（本环境无播放器/浏览器渲染器）⇒ 全部判据是
    PIL 读回读数 + 文件字节/像素读数，不是视觉判据；
  · F 组金标是**两层**（2026-10-15 定案）：**像素层恒验** + **文件字节层按
    `(PIL.__version__, Image.core.zlib_version)` 双档**。
    —— PNG 文件字节由打包进 Pillow 轮的 **zlib 实现**决定（本机 zlib-ng vs 记录基线的
    stock zlib，同一 `PIL.__version__` 下字节仍不同），故**单键（只看 PIL 版本）分档不够**；
    双档之下，同一组合内**逐字节必须相等**（字节级判别力在场），未登记组合**显式 SKIP 并
    打印当前组合**（不静默通过、不误红）。另加「PNG 结构」（IHDR/IDAT/IEND）与
    「同平台两次导出一致性」两格兜底。
    此边界如实登记；**像素一变必红，像素不变而字节变也必红（命中档时）**。
  · 不验超大尺寸（> `_PALETTE_BUDGET` 触发逐帧调色板回退）的**视觉**质量，只验回退路径
    仍确定、仍可读回（C 组）；
  · 不验并发写同一路径、不验 Windows 路径语义。

运行（`lingshu` 仓根；两种都可）：
    python -X utf8 -m pytest tests/test_anim_export_guard.py -q
    python -X utf8 -s tests/test_anim_export_guard.py     # 直跑并打印原始读数
退出码：0 = 全过；1 = 有断言失败。
"""
from __future__ import annotations

import ast
import hashlib
import math
import os
import shutil
import struct
import subprocess
import sys
import textwrap

import pytest
from PIL import Image, __version__ as PIL_VERSION

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lingshu.world import anim  # noqa: E402
from lingshu.world.skeleton3d import cat_skeleton, human_skeleton  # noqa: E402
from lingshu.world.world3d import Camera3D  # noqa: E402

ANIM_PY = os.path.join(REPO, "lingshu", "world", "anim.py")
SKELETON_PY = os.path.join(REPO, "lingshu", "world", "skeleton3d.py")
REPO_DATA = os.path.join(REPO, "lingshu", "data")

# 改前基线（Lead 在任何人动手前实测）——**像素级**金标，与 PIL 的 deflate 后端无关：
#   skeleton_walk.png = 像素 sha256 247c38432b4449ad…（1440×240，1036800 B）
#   skeleton_demo.png = 像素 sha256 0a7b79bf0c83a56c…（1200×400，1440000 B）
#   ★ 2026-10-27 #1066（关节半径改按相机深度）：demo 相机是 look_at 摆位
#     （cz=-2.20/pitch=-8°）、关节世界 z=5.0 而真实深度≈7.2 ⇒ 旧实现把关节
#     画大了约 30%；本条金标随之更新。skeleton_walk.png（默认机位）逐字节
#     不变（247c3843… 前后一致），可作「影响面仅限非默认机位」的证据。
#
# ★为何冻结在**像素**层而不是文件字节层（2026-10-14 机制定因，本机实测）：
#   PNG 文件字节由**打包进 Pillow 轮的 zlib 实现**决定，而 `PIL.__version__` 不反映它 ——
#   本机 `PIL.__version__ == "12.3.0"`、`Image.core.zlib_version == "1.3.1.zlib-ng"`
#   （zlib-ng 构建），而记录基线的环境链的是 **stock zlib 1.3.1**。同一像素 + 同一
#   libpng 过滤流下：zlib-ng 把 IDAT 压成 3346 B，stock zlib（level 6, Z_FILTERED）
#   压成 3408 B ⇒ 文件 sha 不同（本机 c5d21eb5…/3403 vs 金标 9fd99af8…/3465）。
#   反证：把本机产物的 IDAT 抽出、用 stock zlib（level 6, Z_FILTERED）重压、其余 chunk
#   原样拼回 ⇒ **逐字节 == 记录的文件 sha**（walk `9fd99af8…`、demo `59ba2304…`）。
#   ⇒ 文件字节金标**不能只按 PIL 版本分档**（版本串相同而 zlib 后端不同 ⇒ 字节仍变）。
#   本端一度据此把整层降为像素级；**现已改为双键分档恢复字节层**（见下方 `_GOLDEN_PNG`），
#   像素层保留为跨后端不变量恒验，另加「PNG 结构」与「同平台两次导出一致性」兜底。
_GOLDEN_PIXELS = {
    "skeleton_walk.png": (
        "247c38432b4449ad03857c73f6ccb7fa5850ffee7704198f9d928165db174355", 1036800, (1440, 240)),
    "skeleton_demo.png": (
        "0a7b79bf0c83a56c52433a6c2eb3bbe43e9b8ec2454ec398a13b9baf70fbf1b2", 1440000, (1200, 400)),
}

# 文件字节级金标（**作者原设计的那一层，2026-10-15 恢复**）：分档键从 `PIL.__version__`
# 换成 **`(PIL.__version__, Image.core.zlib_version)`** —— PNG 文件字节由打包进 Pillow 轮的
# zlib 实现决定，同一 `PIL.__version__` 下 zlib-ng 与 stock zlib 压出的 IDAT 长度不同。
# ⇒ 同一组合内**逐字节必须相等**（保住字节级判别力）；未登记组合**显式 SKIP 并打印当前组合**
#   （不静默通过、也不误红）。像素层（上方 `_GOLDEN_PIXELS`）在任何组合下都恒验。
_ZLIB_VERSION = getattr(Image.core, "zlib_version", "unknown")
_GOLDEN_PNG = {
    # 记录机（Lead 出值的环境）：stock zlib
    ("12.3.0", "1.3.1"): {
        "skeleton_walk.png": ("9fd99af8acf41522b7687b812712f55e573bb7df2ee95309558f0a9b4c38ad00", 3465),
        "skeleton_demo.png": ("5456cab39aeb60dc9c4929d1e52ce8c6af678a0711483a8c265f6ed6d857e4d5", 4134),
    },
    # 本机（Windows / zlib-ng 构建的 Pillow 轮）：2026-10-15 实测
    ("12.3.0", "1.3.1.zlib-ng"): {
        "skeleton_walk.png": ("c5d21eb5ad9df7237b3b20c7a0a3033f436e58deeba199a55e32d43155db3d45", 3403),
        "skeleton_demo.png": ("5456cab39aeb60dc9c4929d1e52ce8c6af678a0711483a8c265f6ed6d857e4d5", 4134),
    },
}


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


def _sha256(path) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _png_chunks(path):
    """PNG 原始字节的 chunk 序列 `[(type, data), …]`，并先校验签名。"""
    with open(path, "rb") as fh:
        raw = fh.read()
    assert raw[:8] == b"\x89PNG\r\n\x1a\n", f"{path} 不是 PNG（签名不符）"
    chunks, i = [], 8
    while i < len(raw):
        (ln,) = struct.unpack(">I", raw[i:i + 4])
        typ = raw[i + 4:i + 8]
        chunks.append((typ, raw[i + 8:i + 8 + ln]))
        i += 8 + ln + 4
    return chunks


def _png_ihdr(path):
    """`(width, height, bit_depth, color_type)` —— 与 deflate 后端无关的结构面。"""
    typ, data = _png_chunks(path)[0]
    assert typ == b"IHDR", f"{path} 首个 chunk 不是 IHDR: {typ!r}"
    w, h, bd, ct = struct.unpack(">IIBB", data[:10])
    return w, h, bd, ct


def _walk_cam():
    return Camera3D.look_at(eye=(0, 1.2, 1.8), target=(0, 1.0, 5.0), fov_deg=50)


def _walk_pose(i, sk, n: int = 6):
    """走路姿态（相位 (i+0.5)/n）——相邻帧必不相同（GIF 编码器会合并相同相邻帧）。"""
    t = (i + 0.5) / n * 2 * math.pi
    swing = 0.5 * math.sin(t)
    sk.joints["thigh_l"].rot = (swing, 0, 0)
    sk.joints["thigh_r"].rot = (-swing, 0, 0)
    sk.joints["shoulder_l"].rot = (0, 0, -0.4 * math.sin(t + math.pi))
    sk.joints["shoulder_r"].rot = (0, 0, 0.4 * math.sin(t + math.pi))


def _walk_frames(n: int = 6, size=(240, 240)):
    sk = human_skeleton(center=(0, 0.85, 5), height=1.7)
    return anim.frames(sk, n, mutate=lambda i, s: _walk_pose(i, s, n), size=size, camera=_walk_cam())


def _frozen_strip_6():
    """改前 `skeleton3d.__main__` 胶片条代码的**逐字副本**（冻结基线，勿随产品一起改）。

    来源：`wm-verify/lingshu-orig/lingshu/world/skeleton3d.py:322-335`。
    它是 F 组"既有胶片条产物未变"的判据：产品演示若动了这段渲染，像素就会与本复算不符。
    """
    cam_anim = Camera3D.look_at(eye=(0, 1.2, 1.8), target=(0, 1.0, 5.0), fov_deg=50)
    frame_w, frame_h = 240, 240
    strip = Image.new("RGB", (frame_w * 6, frame_h), (24, 28, 46))
    for i in range(6):
        t = i / 6 * 2 * math.pi
        h = human_skeleton(center=(0, 0.85, 5), height=1.7)
        swing = 0.5 * math.sin(t)
        h.joints["thigh_l"].rot = (swing, 0, 0)
        h.joints["thigh_r"].rot = (-swing, 0, 0)
        h.joints["shoulder_l"].rot = (0, 0, -0.4 * math.sin(t + math.pi))
        h.joints["shoulder_r"].rot = (0, 0, 0.4 * math.sin(t + math.pi))
        f = h.render(frame_w, frame_h, camera=cam_anim)
        strip.paste(f, (i * frame_w, 0))
    return strip


def _frozen_demo_grid():
    """改前 `skeleton3d.__main__` 三视图拼图代码的**逐字副本**（冻结基线）。"""
    cam_a = Camera3D.look_at(eye=(0, 1.2, 2.0), target=(0, 1.0, 5.0))
    cam_b = Camera3D.look_at(eye=(2.2, 1.2, 2.0), target=(0, 1.0, 5.0))
    human = human_skeleton(center=(0, 0.85, 5), height=1.7)
    human.joints["thigh_l"].rot = (0.4, 0, 0)
    human.joints["shin_l"].rot = (-0.3, 0, 0)
    human.joints["thigh_r"].rot = (-0.4, 0, 0)
    human.joints["shin_r"].rot = (0.3, 0, 0)
    human.joints["shoulder_l"].rot = (0, 0, -0.5)
    human.joints["shoulder_r"].rot = (0, 0, 0.5)
    grid = Image.new("RGB", (400 * 3, 400), (24, 28, 46))
    for i, cam in enumerate([cam_a, cam_b]):
        grid.paste(human.render(400, 400, camera=cam), (i * 400, 0))
    cat = cat_skeleton(center=(0, 0.4, 5), body_len=0.6)
    cat.joints["tail_tip"].rot = (0.3, 0, 0)
    grid.paste(cat.render(400, 400), (2 * 400, 0))
    return grid


def _snapshot_dir(path):
    """目录快照：{相对文件名: (字节数, mtime_ns)}；目录不存在 ⇒ None。"""
    if not os.path.isdir(path):
        return None
    snap = {}
    for name in sorted(os.listdir(path)):
        full = os.path.join(path, name)
        if os.path.isfile(full):
            st = os.stat(full)
            snap[name] = (st.st_size, st.st_mtime_ns)
    return snap


class _RecordingRenderer:
    """最小 renderer 假件：记录 (改状态 / 渲染) 事件顺序，并按当前状态出图。"""

    def __init__(self):
        self.log = []
        self.state = 0

    def set_state(self, value):
        self.log.append(("set", value))
        self.state = value

    def render(self, w, h, **kw):
        self.log.append(("render", self.state, w, h, dict(kw)))
        return Image.new("RGB", (w, h), (self.state * 20 % 256, 0, 0))


class _NotAnImageRenderer:
    def render(self, w, h, **kw):
        return "not an image"


# ===========================================================================
# A 组 [帧序列] frames()
# ===========================================================================


def test_a1_mutate_runs_before_each_frame_in_order():
    r = _RecordingRenderer()
    out = anim.frames(r, 3, mutate=lambda i, rr: rr.set_state(i), size=(8, 6), tag="T")
    assert r.log == [
        ("set", 0), ("render", 0, 8, 6, {"tag": "T"}),
        ("set", 1), ("render", 1, 8, 6, {"tag": "T"}),
        ("set", 2), ("render", 2, 8, 6, {"tag": "T"}),
    ], f"mutate 不在第 i 帧渲染**之前**按序调用：log={r.log!r}"
    assert len(out) == 3
    assert [im.size for im in out] == [(8, 6)] * 3
    # 每帧真的落在**当帧**姿态上（第 i 帧 = set_state(i) 之后渲染的）
    assert [im.getpixel((0, 0))[0] for im in out] == [0, 20, 40]


def test_a2_without_mutate_is_n_times_same_state():
    """契约：不给 mutate ⇒ 同状态渲 n 次（合法"静止"对照，但不许假装是动画）。"""
    r = _RecordingRenderer()
    out = anim.frames(r, 3, size=(8, 8))
    assert len(out) == 3
    assert r.log == [("render", 0, 8, 8, {})] * 3
    assert all(im.tobytes() == out[0].tobytes() for im in out)


def test_a3_frames_rejects_bad_inputs():
    r = _RecordingRenderer()
    with pytest.raises(ValueError):
        anim.frames(r, 0)
    with pytest.raises(ValueError):
        anim.frames(r, -1)
    with pytest.raises(TypeError):
        anim.frames(r, "3")
    with pytest.raises(ValueError):
        anim.frames(r, 2, size=(0, 4))
    with pytest.raises(TypeError):
        anim.frames(object(), 2)          # 没有 render 面
    with pytest.raises(TypeError):
        anim.frames(_NotAnImageRenderer(), 2)  # render 返回的不是 PIL.Image


def test_a4_frames_size_is_honored_and_kwargs_forwarded():
    r = _RecordingRenderer()
    out = anim.frames(r, 2, size=(11, 13), extra=1)
    assert [im.size for im in out] == [(11, 13), (11, 13)]
    assert r.log[0][4] == {"extra": 1}


# ===========================================================================
# B 组 [GIF 是动画] to_gif 读回
# ===========================================================================


def test_b1_gif_readback_matches_request(tmp_path):
    fr = _walk_frames()
    p = tmp_path / "walk.gif"
    obs = anim.to_gif(fr, str(p), duration_ms=100, loop=0)
    with Image.open(p) as im:
        n_frames, size = im.n_frames, im.size
        duration, loop = im.info.get("duration"), im.info.get("loop")
    print(f"\n[B1] n_frames={n_frames} size={size} duration={duration} loop={loop} bytes={obs['bytes']}")
    assert n_frames == 6 == len(fr)
    assert size == (240, 240) == fr[0].size
    assert duration == 100
    assert loop == 0
    # 返回 dict 必须是**回读值**（Lead 裁决），且 bytes 与真实文件一致
    assert obs["frames"] == n_frames
    assert obs["size"] == size
    assert obs["duration_ms"] == duration
    assert obs["loop"] == loop
    assert obs["bytes"] == os.path.getsize(p)
    assert obs["path"] == str(p)


def test_b2_loop_and_duration_are_what_was_asked(tmp_path):
    fr = _walk_frames(3)
    p = tmp_path / "a.gif"
    obs = anim.to_gif(fr, str(p), duration_ms=40, loop=7)
    with Image.open(p) as im:
        assert im.n_frames == 3
        assert im.info.get("duration") == 40
        assert im.info.get("loop") == 7
    assert obs["duration_ms"] == 40 and obs["loop"] == 7


def test_b3_frame_pixels_survive_quantization_mae(tmp_path):
    """调色板量化误差必须**如实量化**（不假装无损）：本件线稿帧实测 MAE = 0.0。"""
    fr = _walk_frames()
    p = tmp_path / "q.gif"
    anim.to_gif(fr, str(p), duration_ms=100, loop=0)
    maes = []
    with Image.open(p) as im:
        for i in range(im.n_frames):
            im.seek(i)
            a = im.convert("RGB").tobytes()
            b = fr[i].convert("RGB").tobytes()
            maes.append(sum(abs(x - y) for x, y in zip(a, b)) / len(a))
    print(f"[B3] 逐帧 MAE = {[round(m, 6) for m in maes]}")
    assert len(maes) == len(fr)
    assert max(maes) <= 16.0, f"量化误差过大：{maes!r}"


# ===========================================================================
# C 组 [确定性]
# ===========================================================================

_SUBPROC_RENDER_GIF = textwrap.dedent(
    """
    import hashlib, math, sys
    from lingshu.world import anim
    from lingshu.world.skeleton3d import human_skeleton
    from lingshu.world.world3d import Camera3D

    cam = Camera3D.look_at(eye=(0, 1.2, 1.8), target=(0, 1.0, 5.0), fov_deg=50)
    sk = human_skeleton(center=(0, 0.85, 5), height=1.7)

    def pose(i, s):
        t = (i + 0.5) / 6 * 2 * math.pi
        sw = 0.5 * math.sin(t)
        s.joints["thigh_l"].rot = (sw, 0, 0)
        s.joints["thigh_r"].rot = (-sw, 0, 0)
        s.joints["shoulder_l"].rot = (0, 0, -0.4 * math.sin(t + math.pi))
        s.joints["shoulder_r"].rot = (0, 0, 0.4 * math.sin(t + math.pi))

    fr = anim.frames(sk, 6, mutate=pose, size=(240, 240), camera=cam)
    obs = anim.to_gif(fr, sys.argv[1], duration_ms=100, loop=0)
    with open(obs["path"], "rb") as fh:
        print(hashlib.sha256(fh.read()).hexdigest())
    """
)


def test_c1_same_input_byte_identical_in_process(tmp_path):
    fr = _walk_frames()
    a = anim.to_gif(fr, str(tmp_path / "a.gif"), duration_ms=100, loop=0)
    b = anim.to_gif(fr, str(tmp_path / "b.gif"), duration_ms=100, loop=0)
    ha, hb = _sha256(a["path"]), _sha256(b["path"])
    print(f"\n[C1] sha256#1={ha}\n[C1] sha256#2={hb}")
    assert ha == hb


def test_c2_cross_process_and_cross_hashseed_identical(tmp_path):
    """跨进程 + 跨 PYTHONHASHSEED 也必须逐字节相同（防吃进程随机/哈希序）。"""
    hashes = []
    for seed in ("0", "1"):
        env = dict(os.environ, PYTHONPATH=REPO, PYTHONHASHSEED=seed)
        cp = subprocess.run(
            [sys.executable, "-X", "utf8", "-c", _SUBPROC_RENDER_GIF, str(tmp_path / f"s{seed}.gif")],
            cwd=str(tmp_path), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        assert cp.returncode == 0, cp.stderr
        hashes.append(cp.stdout.strip().splitlines()[-1])
    inproc = _sha256(anim.to_gif(_walk_frames(), str(tmp_path / "in.gif"))["path"])
    print(f"[C2] seed0={hashes[0]}\n[C2] seed1={hashes[1]}\n[C2] inproc={inproc}")
    assert hashes[0] == hashes[1] == inproc


def test_c3_from_paths_roundtrip_is_byte_identical(tmp_path):
    fr = _walk_frames()
    direct = anim.to_gif(fr, str(tmp_path / "direct.gif"), duration_ms=100, loop=0)
    paths = anim.save_frames(fr, str(tmp_path / "seq"), prefix="walk")
    assert [os.path.basename(p) for p in paths] == [f"walk_{i:04d}.png" for i in range(6)]
    back = anim.from_paths(paths, out=str(tmp_path / "back.gif"), duration_ms=100, loop=0)
    hd, hb = _sha256(direct["path"]), _sha256(back["path"])
    print(f"[C3] direct={hd}\n[C3] from_paths={hb}")
    assert hb == hd
    assert back["frames"] == 6


def test_c4_palette_budget_fallback_still_deterministic(tmp_path, monkeypatch):
    """超过 `_PALETTE_BUDGET` 的走逐帧调色板回退 —— 仍须确定且可读回。"""
    fr = _walk_frames()
    ref = _sha256(anim.to_gif(fr, str(tmp_path / "ref.gif"))["path"])
    monkeypatch.setattr(anim, "_PALETTE_BUDGET", 1)
    a = anim.to_gif(fr, str(tmp_path / "a.gif"), duration_ms=100, loop=0)
    b = anim.to_gif(fr, str(tmp_path / "b.gif"), duration_ms=100, loop=0)
    ha, hb = _sha256(a["path"]), _sha256(b["path"])
    print(f"[C4] 回退 sha256 两次={ha} {hb}；共享调色板={ref}")
    assert ha == hb
    assert a["frames"] == 6
    with Image.open(a["path"]) as im:
        assert im.n_frames == 6 and im.size == (240, 240)


# ===========================================================================
# D 组 [不静默] 边界与对抗
# ===========================================================================


def test_d1_empty_and_size_mismatch_raise_without_file(tmp_path):
    p = tmp_path / "x.gif"
    with pytest.raises(ValueError):
        anim.to_gif([], str(p))
    assert not p.exists(), "空输入不该留下文件"
    fr = _walk_frames(2)
    with pytest.raises(ValueError):
        anim.to_gif([fr[0], Image.new("RGB", (10, 10), (0, 0, 0))], str(p))
    assert not p.exists(), "尺寸不一致不该留下文件（PIL 会静默只写 1 帧）"
    with pytest.raises(ValueError):
        anim.to_strip([])
    with pytest.raises(ValueError):
        anim.to_strip(fr, cols=0)
    with pytest.raises(ValueError):
        anim.save_frames([], str(tmp_path / "none"))
    with pytest.raises(ValueError):
        anim.from_paths([])
    with pytest.raises(FileNotFoundError):
        anim.from_paths([str(tmp_path / "nope.png")])


@pytest.mark.parametrize("bad", [0, 1, 5, 11, 105, -100, 655360, 10 ** 9])
def test_d2_duration_out_of_domain_raises(tmp_path, bad):
    p = tmp_path / "d.gif"
    with pytest.raises(ValueError):
        anim.to_gif(_walk_frames(2), str(p), duration_ms=bad)
    assert not p.exists()


def test_d3_duration_type_and_loop_domain(tmp_path):
    fr = _walk_frames(2)
    p = tmp_path / "d.gif"
    with pytest.raises(TypeError):
        anim.to_gif(fr, str(p), duration_ms=100.0)
    for bad_loop in (-1, 65536):
        with pytest.raises(ValueError):
            anim.to_gif(fr, str(p), loop=bad_loop)
    with pytest.raises(TypeError):
        anim.to_gif(fr, str(p), loop="3")


def test_d4_identical_neighbour_frames_raise_and_leave_no_file(tmp_path):
    """PIL 会静默合并相邻相同帧 ⇒ 必须回读自校验 raise，且**不留坏文件**。"""
    fr = _walk_frames(6)
    p = tmp_path / "dup.gif"
    dup = [fr[0], fr[1], fr[1], fr[1], fr[2]]   # 连续相同帧
    with pytest.raises(ValueError) as ei:
        anim.to_gif(dup, str(p), duration_ms=100, loop=0)
    print(f"\n[D4] raise: {ei.value}")
    assert "帧" in str(ei.value)
    assert not p.exists(), "自校验失败后目标路径不得留下文件"


def test_d5_pure_colour_frames_merge_and_raise(tmp_path):
    p = tmp_path / "pc.gif"
    with pytest.raises(ValueError):
        anim.to_gif([Image.new("RGB", (16, 16), (7, 7, 7)) for _ in range(4)], str(p))
    assert not p.exists()


def test_d6_single_frame_is_legal_degenerate(tmp_path):
    fr = _walk_frames(1)
    p = tmp_path / "one.gif"
    obs = anim.to_gif(fr, str(p), duration_ms=250, loop=1)
    with Image.open(p) as im:
        assert im.n_frames == 1
        assert im.info.get("duration") == 250
        assert im.info.get("loop") == 1
    assert obs["frames"] == 1


def test_d7_rgba_frames_exportable_alpha_dropped(tmp_path):
    fr = [Image.new("RGBA", (32, 32), (i * 40, 60, 90, 180)) for i in range(3)]
    p = tmp_path / "rgba.gif"
    obs = anim.to_gif(fr, str(p), duration_ms=50, loop=2)
    with Image.open(p) as im:
        assert im.n_frames == 3
        assert im.size == (32, 32)
        assert im.convert("RGB").mode == "RGB"
    assert obs["frames"] == 3


def test_d8_duration_must_be_multiple_of_ten_with_measured_why(tmp_path):
    """为什么 `duration_ms` 必须是 10 的整数倍：GIF 的 delay 是**厘秒**字段。

    先实测 PIL 的裸行为（请求 105ms ⇒ 照写、读回 100ms，**静默改写**），
    再断言 `to_gif` 对 105 显式 raise（不静默），且不留文件。
    """
    raw = tmp_path / "raw105.gif"
    pal = Image.new("P", (8, 8))
    pal.putpalette([0, 0, 0] * 256)
    pal.save(raw, format="GIF", save_all=True, duration=105, loop=3)
    with Image.open(raw) as im:
        got = im.info.get("duration")
    print(f"\n[D8] PIL 裸行为：请求 duration=105ms ⇒ 读回 {got}ms（厘秒粒度，静默改写）")
    assert got == 100, f"前提变了：PIL 对 105ms 的读回是 {got} —— 本判据需重新标定"

    p = tmp_path / "never.gif"
    with pytest.raises(ValueError) as ei:
        anim.to_gif(_walk_frames(2), str(p), duration_ms=105)
    msg = str(ei.value)
    assert "105" in msg and "100" in msg, msg
    assert not p.exists(), "拒绝之后不得留下文件"


def test_d9_duration_representable_boundaries(tmp_path):
    """下限 10ms 与上限 655350ms 都必须**真能表达**（回读 == 请求）。"""
    for d in (10, 655350):
        p = tmp_path / f"b{d}.gif"
        obs = anim.to_gif(_walk_frames(2), str(p), duration_ms=d, loop=0)
        with Image.open(p) as im:
            assert im.info.get("duration") == d
        assert obs["duration_ms"] == d
        print(f"[D9] duration_ms={d} ⇒ 回读 {obs['duration_ms']}ms（可表达）")


# ===========================================================================
# E 组 [胶片条 ≠ 动画] 判据必须能分开两者
# ===========================================================================


def test_e1_strip_is_static_while_gif_is_animated(tmp_path):
    fr = _walk_frames()
    # 补 `n_frames` 的**事实依据**：Pillow 12 的基类 `Image.Image` 没有该属性
    # （属性由各插件在 open() 时设置）。若将来 Pillow 自带，本行会红 ⇒ 请复核 to_strip
    # 的"仅缺失时才补"是否仍然成立（而不是删掉本行）。
    assert not hasattr(Image.new("RGB", (2, 2)), "n_frames"), (
        "前提变了：基类 Image.Image 现在自带 n_frames ⇒ to_strip 的显式补法需重新复核"
    )
    strip = anim.to_strip(fr)
    assert strip.n_frames == 1, "to_strip 的返回值必须是单帧静态图"
    sp = tmp_path / "strip.png"
    strip.save(sp)
    with Image.open(sp) as im:
        assert im.n_frames == 1
        assert im.size == (240 * 6, 240)
    # 返回值上的 n_frames 与"落盘再 open"同义
    assert strip.n_frames == Image.open(sp).n_frames == 1
    gp = tmp_path / "anim.gif"
    obs = anim.to_gif(fr, str(gp), duration_ms=100, loop=0)
    with Image.open(gp) as im:
        gif_frames = im.n_frames
    print(f"\n[E1] 胶片条 n_frames=1 size={(240*6, 240)}；GIF n_frames={gif_frames}")
    assert gif_frames == 6 == obs["frames"]
    assert 1 < gif_frames, "判据必须能把'胶片条(1 帧)'与'动画(n 帧)'分开"


def test_e2_strip_grid_layout(tmp_path):
    fr = _walk_frames(5, size=(20, 10))
    row = anim.to_strip(fr)
    assert row.size == (100, 10)
    grid = anim.to_strip(fr, cols=3, background=(0, 0, 0))
    assert grid.size == (60, 20)          # 2 行 × 3 列，第 6 格空
    assert grid.n_frames == 1
    # 空格的空隙露出背景色 ⇒ background 生效（不是被帧铺满的假网格）
    assert grid.getpixel((41, 11)) == (0, 0, 0)
    assert grid.getpixel((1, 1)) != (0, 0, 0)   # 有帧的格子不是背景色


# ===========================================================================
# F 组 [演示回归] 胶片条字节未变 + 追加真 GIF + 措辞对齐 + 不污染仓
# ===========================================================================


def _run_demo_sandbox(tmp_path):
    root = tmp_path / "sandbox"
    shutil.copytree(
        os.path.join(REPO, "lingshu"),
        root / "lingshu",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "data"),
    )
    env = dict(os.environ, PYTHONPATH=str(root), PYTHONHASHSEED="0")
    cp = subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "lingshu.world.skeleton3d"],
        cwd=str(root), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return root, cp


def test_f1_demo_strip_bytes_frozen_and_gif_appended(tmp_path):
    before = _snapshot_dir(REPO_DATA)
    root, cp = _run_demo_sandbox(tmp_path)
    assert cp.returncode == 0, cp.stdout + cp.stderr
    out = root / "lingshu" / "data"
    strip_png, gif = out / "skeleton_walk.png", out / "skeleton_walk.gif"
    demo_png = out / "skeleton_demo.png"
    assert strip_png.is_file(), cp.stdout
    assert gif.is_file(), f"演示未追加真 GIF。stdout={cp.stdout!r} stderr={cp.stderr!r}"

    # ---- 既有胶片条：单帧静态 + 像素与改前循环逐字复算一致（跨平台强判据）
    with Image.open(strip_png) as im:
        assert im.n_frames == 1, "产物是胶片条（单帧静态图），不是动画"
        got_strip = im.convert("RGB").tobytes()
    assert got_strip == _frozen_strip_6().tobytes(), "胶片条像素被改动了"
    with Image.open(demo_png) as im:
        got_demo = im.convert("RGB").tobytes()
    assert got_demo == _frozen_demo_grid().tobytes(), "三视图拼图像素被改动了"

    # ---- 像素级金标（平台/zlib 无关）+ PNG 结构（压缩后端无关）
    pixel_seen = {
        "skeleton_walk.png": (hashlib.sha256(got_strip).hexdigest(), len(got_strip)),
        "skeleton_demo.png": (hashlib.sha256(got_demo).hexdigest(), len(got_demo)),
    }
    for name, path in (("skeleton_walk.png", strip_png), ("skeleton_demo.png", demo_png)):
        sha, pbytes, size = _GOLDEN_PIXELS[name]
        got_psha, got_pbytes = pixel_seen[name]
        print(f"[F1] {name} pixel_sha256={got_psha} pixels={got_pbytes} "
              f"file_bytes={path.stat().st_size} (Pillow {PIL_VERSION}/"
              f"{Image.core.zlib_version})")
        assert got_psha == sha, f"{name} 像素与改前基线不同（产物被改动了）"
        assert got_pbytes == pbytes
        w, h, bd, ct = _png_ihdr(path)
        assert (w, h) == size and bd == 8 and ct == 2, f"{name} 的 PNG IHDR 不符：{(w, h, bd, ct)}"
        types = [t for t, _ in _png_chunks(path)]
        assert types[0] == b"IHDR" and types[-1] == b"IEND" and b"IDAT" in types, types

    # ---- 文件字节级金标：按 (Pillow 版本, zlib 实现) 双档，命中即**逐字节必须相等**
    combo = (PIL_VERSION, _ZLIB_VERSION)
    tier = _GOLDEN_PNG.get(combo)
    if tier is None:
        print(f"[F1] 字节层 SKIP：未登记组合 {combo}（像素层与 PNG 结构仍已验）")
    else:
        for name, path in (("skeleton_walk.png", strip_png), ("skeleton_demo.png", demo_png)):
            sha, nbytes = tier[name]
            got = _sha256(path); size = path.stat().st_size
            print(f"[F1] {name} 字节档 {combo}: file_sha256={got[:16]}…/{size} "
                  f"金标={sha[:16]}…/{nbytes}")
            assert size == nbytes, f"{name} 文件字节数变了：{size} != {nbytes}"
            assert got == sha, f"{name} 文件字节与金标不同（像素未变而字节变 ⇒ 编码/元数据被动过）"

    # ---- 追加的真 GIF
    assert open(gif, "rb").read(6) in (b"GIF87a", b"GIF89a"), "skeleton_walk.gif 不是合法 GIF"
    with Image.open(gif) as im:
        gif_frames, gif_size = im.n_frames, im.size
        gif_duration, gif_loop = im.info.get("duration"), im.info.get("loop")
    print(f"[F1] skeleton_walk.gif n_frames={gif_frames} size={gif_size} "
          f"duration={gif_duration} loop={gif_loop} bytes={gif.stat().st_size}")
    assert gif_frames == 12 and 1 < gif_frames
    assert gif_size == (240, 240)
    assert gif_duration == 100
    assert gif_loop == 0

    # ---- 同平台复算一致性：再跑一遍演示，两次产物逐字节相同（平台内确定性，
    #      替代跨平台不可复现的文件字节金标这一格）
    root2, cp2 = _run_demo_sandbox(tmp_path / "rerun")
    assert cp2.returncode == 0, cp2.stdout + cp2.stderr
    out2 = root2 / "lingshu" / "data"
    for name in ("skeleton_walk.png", "skeleton_demo.png", "skeleton_walk.gif"):
        a, b = (out / name).read_bytes(), (out2 / name).read_bytes()
        print(f"[F1] rerun {name}: run1={len(a)}B run2={len(b)}B identical={a == b}")
        assert a == b, f"{name} 两次导出字节不同（同平台不确定性 / 吃进程随机）"

    # ---- stdout 措辞与事实对齐
    lines = [ln for ln in cp.stdout.splitlines() if ln.strip()]
    png_line = next((ln for ln in lines if "skeleton_walk.png" in ln), "")
    gif_line = next((ln for ln in lines if "skeleton_walk.gif" in ln), "")
    print(f"[F1] stdout png 行: {png_line!r}\n[F1] stdout gif 行: {gif_line!r}")
    low = png_line.lower()
    assert ("filmstrip" in low or "胶片条" in png_line), png_line
    # 必须**显式否定**自己是动画（不是把"animation"从措辞里藏掉就算数）
    assert ("not an animation" in low or "not animation" in low
            or "非动画" in png_line or "静态" in png_line), f"胶片条仍被当作动画陈述：{png_line!r}"
    assert "animation" in gif_line.lower(), gif_line

    # ---- 演示/守卫都不得写仓内 lingshu/data
    assert _snapshot_dir(REPO_DATA) == before, "本守卫触碰了仓内 lingshu/data（必须只在 tmp_path 落盘）"


def test_f2_demo_source_wording_is_aligned():
    """源码级：`skeleton3d.__main__` 里那段产物的名字不能再叫 animation。"""
    with open(SKELETON_PY, encoding="utf-8") as fh:
        src = fh.read()
    assert "filmstrip" in src, "演示里胶片条的名字仍未与事实对齐"
    assert 'skeleton_walk.png' in src and 'skeleton_walk.gif' in src
    assert "from .anim import" in src, "演示没有用上新模块 ⇒ 追加的 GIF 不是真动画出口产的"


# ===========================================================================
# G 组 [惰性 PIL]
# ===========================================================================


def _top_level_names(node) -> list:
    if isinstance(node, ast.Import):
        return [a.name.split(".")[0] for a in node.names]
    if isinstance(node, ast.ImportFrom):
        if node.level:            # 相对导入（仓内）
            return []
        return [(node.module or "").split(".")[0]]
    return []


def test_g1_anim_has_no_module_level_pil_import():
    with open(ANIM_PY, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    module_level, func_level = [], []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module_level += _top_level_names(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for sub in ast.walk(node):
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    func_level += _top_level_names(sub)
    print(f"\n[G1] 模块级 import={sorted(set(module_level))} 函数级={sorted(set(func_level))}")
    assert "PIL" not in module_level, "PIL 必须在函数内惰性 import（world 的位图面依赖 pillow extra）"
    assert "PIL" in func_level, "anim.py 缺少函数内 PIL import"


def test_g2_import_anim_still_works_with_pil_blocked(tmp_path):
    script = textwrap.dedent(
        """
        import importlib.abc, sys

        class _BlockPIL(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name == "PIL" or name.startswith("PIL."):
                    raise ImportError("PIL blocked by guard")
                return None

        sys.meta_path.insert(0, _BlockPIL())
        for mod in [m for m in list(sys.modules) if m == "PIL" or m.startswith("PIL.")]:
            del sys.modules[mod]
        import lingshu.world.anim as A
        print("OK", len(A.__all__))
        """
    )
    env = dict(os.environ, PYTHONPATH=REPO, PYTHONHASHSEED="0")
    cp = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", script],
        cwd=str(tmp_path), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    assert cp.returncode == 0, cp.stderr
    assert cp.stdout.strip() == "OK 5", f"{cp.stdout!r} / {cp.stderr!r}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


# ================================================ G 真步态（骨架运动逻辑贴合现实）
def _gait_stats(n=96):
    import math as _m
    from lingshu.world.skeleton3d import human_skeleton, apply_walk, WALK_DUTY
    sk = human_skeleton(center=(0, 0.85, 5), height=1.7)
    toe, toe_l, toe_r, hip, knee, hipf = [], [], [], [], [], []
    for i in range(n):
        apply_walk(sk, i / float(n))
        yl, yr = sk.world_pos("foot_l")[1], sk.world_pos("foot_r")[1]
        toe.append(min(yl, yr))
        toe_l.append(yl)
        toe_r.append(yr)
        hip.append(sk.world_pos("hip")[1])
        knee.append(_m.degrees(sk.joints["shin_l"].rot[0]))
        hipf.append(-_m.degrees(sk.joints["thigh_l"].rot[0]))
    return toe, toe_l, toe_r, hip, knee, hipf, n, WALK_DUTY


def test_g1_脚真的踩在地面上_不再悬空():
    """G1：改前踝在 `y≈0.578`（**悬空 0.58 m**）；改后趾最小高度必须 ≈0，
    且摆动相要真的抬起来（否则是"贴地滑行"，同样不是走路）。"""
    toe, toe_l, toe_r, _, _, _, _, _ = _gait_stats()
    assert abs(min(toe)) < 0.006, "不贴地：%r" % (min(toe),)
    #: ⚠ 抬脚要按**单脚**判：`min(左右)` 里总有一脚踩地 ⇒ 恒≈0（本判据第一版就这么红的）
    assert max(toe_l) > 0.05 and max(toe_r) > 0.05, \
        "摆动相没抬脚：左 %.4f 右 %.4f" % (max(toe_l), max(toe_r))


def test_g2_关节活动度落在步态文献区间():
    """G2：膝/髋的 ROM 必须落在步态分析文献的常见区间。
    改前只转大腿、膝恒为 0° ⇒ 这条**必红**（回退即红的判据之一）。"""
    _, _, _, _, knee, hipf, _, _ = _gait_stats()
    assert 50.0 <= max(knee) <= 70.0, "膝摆动峰 %.1f° 不在 55~65 附近" % max(knee)
    assert min(knee) <= 8.0, "膝最小屈曲 %.1f° 应≈5°（不是死直）" % min(knee)
    assert 22.0 <= max(hipf) <= 35.0, "髋屈峰 %.1f° 不在 25~30 附近" % max(hipf)
    assert -18.0 <= min(hipf) <= -5.0, "髋伸峰 %.1f° 不在 −10~−15 附近" % min(hipf)


def test_g3_竖直起伏由运动学导出_步频为每步幅两次():
    """G3：真人步行的髋部竖直起伏 ≈4~5 cm，且**每个步幅出现两次**（步频）。
    这里用 `h(φ) == h(φ+0.5)` 判别"步频 2/步幅"——改前没有起伏（恒 0）⇒ 必红。"""
    _, _, _, hip, _, _, n, _ = _gait_stats()
    amp = max(hip) - min(hip)
    assert 0.025 <= amp <= 0.075, "髋起伏 %.1f cm 不在 4~5 cm 附近" % (amp * 100)
    asym = max(abs(hip[i] - hip[(i + n // 2) % n]) for i in range(n))
    assert asym < 0.002, "半周期不对称 %.4f m ⇒ 步频不是 2/步幅" % asym


def test_g4_回退即红_换回单正弦摆大腿就不成立():
    """G4（**回退即红**）：把运动逻辑换回改前的"单正弦只摆大腿"，G2/G3 必须转红。

    实测读数（本机）：膝 ROM 由 `4.9~58.3°` 变成 `0~0°`；髋起伏由 `5.3 cm` 变成 `0.0 cm`。
    这条证明 G2/G3 有**判别力**，不是恒真断言。
    """
    import math as _m
    from lingshu.world.skeleton3d import human_skeleton
    sk = human_skeleton(center=(0, 0.85, 5), height=1.7)
    knee, hip = [], []
    for i in range(96):
        t = i / 96.0 * 2 * _m.pi
        sw = 0.5 * _m.sin(t)
        sk.joints["thigh_l"].rot = (sw, 0, 0)
        sk.joints["thigh_r"].rot = (-sw, 0, 0)
        knee.append(_m.degrees(sk.joints["shin_l"].rot[0]))
        hip.append(sk.world_pos("hip")[1])
    assert max(knee) - min(knee) == 0.0, "改前逻辑膝应为恒 0（实测）"
    assert max(hip) - min(hip) == 0.0, "改前逻辑无竖直起伏（实测）"
    assert max(knee) < 50.0, "改前逻辑不该满足 G2 的膝峰下界"
