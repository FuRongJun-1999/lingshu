# -*- coding: utf-8 -*-
"""render · 逐帧渲染与"绘画"后处理（架构 §2.1 绘画 / §2.3 可复用件 / §七 P2）

渲染链（每一步都可解释、可关掉做对照）：

    ① `World3D.render()`   —— 上游画家算法 / **真 z-buffer**（场景含网格时自动走后者）
    ② 天空渐变合成          —— 把背景键色像素换成 sky_top→sky_horizon 竖直渐变
    ③ 地平线雾带            —— 靠近地平线的一带向天色过渡（纵深感的低成本做法）
    ④ bloom                 —— 亮部提取 + 高斯模糊 + 屏幕叠加（霓虹光晕）
    ⑤ vignette              —— 四角压暗（把视线收回画面中心）
    ⑥ 胶片颗粒              —— **确定性**噪声（`default_rng(frame)`，同帧同图）
    ⑦ 文字叠层              —— 由 `timeline.render_overlay` 负责（本模块不重复实现）

色彩来源：`lingshu.world.shade_paint`（`role_palette` / `look_curve`），不另造配色表。
"""
from __future__ import annotations

import numpy as np

from . import paths

paths.ensure_lingshu()

from PIL import Image, ImageFilter  # noqa: E402

#: 背景**键色**：渲染时用它铺底，随后被天空渐变整片替换。
#: 取值刻意远离任何实拍色（近黑），避免与物体颜色撞色导致误替换。
KEY_BG = (1, 2, 3)


def make_style(brief: dict) -> dict:
    """从 brief 派生渲染风格（颜色/强度），缺省值在此集中，别散到调用点。"""
    st = dict(brief.get("style") or {})
    return {
        "sky_top": tuple(st.get("sky_top") or (8, 10, 26)),
        "sky_horizon": tuple(st.get("sky_horizon") or (58, 38, 72)),
        "ground": tuple(st.get("ground") or (16, 18, 30)),
        "bg": KEY_BG,
        "bloom": float(st.get("bloom", 0.5)),
        "vignette": float(st.get("vignette", 0.4)),
        "grain": float(st.get("grain", 0.015)),
        "fog_density": float(st.get("fog_density", 0.25)),
        "palette": st.get("palette", "neon_night"),
        # 渲染开关（`World3D` 实例属性）：`studio` 是上游的三点布光光学模型
        "shade": str(st.get("shade", "studio")),
        "smooth": bool(st.get("smooth", True)),
        "cull": bool(st.get("cull", True)),
        # 调色（可缺；见 `_grade`）
        "grade": dict(st.get("grade") or {}),
    }


def _sky_gradient(w: int, h: int, top, horizon) -> np.ndarray:
    """竖直渐变（上暗下亮到地平线色），形状 (h,w,3) float32。"""
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    k = np.clip(y / 0.45, 0.0, 1.0) ** 0.85          # 地平线在 45% 高度
    top = np.array(top, dtype=np.float32)
    hor = np.array(horizon, dtype=np.float32)
    row = top[None, :] * (1.0 - k) + hor[None, :] * k
    return np.repeat(row[:, None, :], w, axis=1)


def _bloom(arr: np.ndarray, amount: float) -> np.ndarray:
    """亮部光晕：阈值提取 → 高斯模糊 → 屏幕叠加。"""
    if amount <= 0.0:
        return arr
    luma = arr @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    mask = np.clip((luma - 0.62) / 0.38, 0.0, 1.0)[:, :, None]
    bright = arr * mask
    img = Image.fromarray(np.clip(bright, 0, 255).astype(np.uint8), "RGB")
    blurred = np.asarray(img.filter(ImageFilter.GaussianBlur(radius=max(2, arr.shape[1] // 120))),
                         dtype=np.float32)
    out = 255.0 - (255.0 - arr) * (255.0 - blurred * amount) / 255.0
    return np.clip(out, 0.0, 255.0)


def _vignette(arr: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0.0:
        return arr
    h, w = arr.shape[:2]
    yy = (np.linspace(-1.0, 1.0, h, dtype=np.float32) ** 2)[:, None]
    xx = (np.linspace(-1.0, 1.0, w, dtype=np.float32) ** 2)[None, :]
    r = np.sqrt(0.6 * yy + xx)
    v = 1.0 - amount * np.clip(r / 1.35, 0.0, 1.0) ** 2.2
    return arr * v[:, :, None]


def _grain(arr: np.ndarray, amount: float, frame: int) -> np.ndarray:
    if amount <= 0.0:
        return arr
    rng = np.random.default_rng(1000003 + int(frame))     # 确定性：同帧同颗粒
    n = rng.standard_normal(arr.shape[:2]).astype(np.float32)[:, :, None] * (amount * 255.0)
    return np.clip(arr + n, 0.0, 255.0)


def _fog_band(arr: np.ndarray, density: float, horizon_col) -> np.ndarray:
    """地平线附近的雾：以 45% 高度为中心、上下各约 8% 画布宽的过渡带。"""
    if density <= 0.0:
        return arr
    h, w = arr.shape[:2]
    y = np.arange(h, dtype=np.float32)[:, None]
    center = 0.45 * h
    width = max(4.0, 0.08 * h)
    k = np.exp(-((y - center) ** 2) / (2.0 * width * width)) * density
    hor = np.array(horizon_col, dtype=np.float32)[None, None, :]
    return arr * (1.0 - k[:, :, None]) + hor * k[:, :, None]


def _grade(arr: np.ndarray, g: dict) -> np.ndarray:
    """调色：对比 / 饱和 / 阴影-高光分离染色（确定性、逐像素、可解释）。

    `g` 键（都可缺）：
      · `contrast`    对比度倍数（以 0.5 为轴）
      · `saturation`  饱和度倍数（以 Rec.709 亮度为轴）
      · `shadow_tint` / `highlight_tint`  按亮度在两者间插值的**乘法染色**
      · `lift`        黑位抬升（0~1，加在阴影上，模拟胶片黑位）
    """
    if not g:
        return arr
    x = arr / 255.0
    c = float(g.get("contrast", 1.0))
    if abs(c - 1.0) > 1e-9:
        x = (x - 0.5) * c + 0.5
    sat = float(g.get("saturation", 1.0))
    if abs(sat - 1.0) > 1e-9:
        luma = (x @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))[:, :, None]
        x = luma + (x - luma) * sat
    st = g.get("shadow_tint")
    ht = g.get("highlight_tint")
    if st or ht:
        st = np.array(st or (1.0, 1.0, 1.0), dtype=np.float32)
        ht = np.array(ht or (1.0, 1.0, 1.0), dtype=np.float32)
        lum = np.clip(x @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32), 0.0, 1.0)
        tint = st[None, None, :] * (1.0 - lum[:, :, None]) + ht[None, None, :] * lum[:, :, None]
        x = x * tint
    lift = float(g.get("lift", 0.0))
    if lift > 0.0:
        x = x + lift * (1.0 - np.clip(x, 0.0, 1.0)) * 0.5
    return np.clip(x * 255.0, 0.0, 255.0)


def render_frame(world, camera, frame: int, shot: dict, size=(960, 540),
                 style: dict = None, ssaa: int = 1) -> "Image.Image":
    """渲染一帧（含后处理）。`size=(w,h)`；纯函数式：同入参 ⇒ 同像素。"""
    w, h = int(size[0]), int(size[1])
    style = style or {}
    img = world.render(w, h, camera=camera,
                       background=tuple(style.get("bg", KEY_BG)),
                       ground_color=tuple(style.get("ground", (16, 18, 30))),
                       ssaa=int(ssaa))
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)

    # ② 天空渐变：只替换**恰好等于键色**的像素（背景区）
    bg = np.array(style.get("bg", KEY_BG), dtype=np.float32)
    key = np.all(np.abs(arr - bg[None, None, :]) < 0.5, axis=2)
    if key.any():
        grad = _sky_gradient(w, h, style.get("sky_top", (8, 10, 26)),
                             style.get("sky_horizon", (58, 38, 72)))
        arr[key] = grad[key]

    # ③ 地平线雾
    arr = _fog_band(arr, float(style.get("fog_density", 0.25)),
                    style.get("sky_horizon", (58, 38, 72)))
    # ③b 调色（冷阴影/暖高光 + 对比/饱和）
    arr = _grade(arr, style.get("grade") or {})
    # ④ bloom ⑤ vignette ⑥ grain
    arr = _bloom(arr, float(style.get("bloom", 0.5)))
    arr = _vignette(arr, float(style.get("vignette", 0.4)))
    arr = _grain(arr, float(style.get("grain", 0.015)), frame)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def render_frame_svg(world, camera, shot: dict, size=(960, 540), style: dict = None) -> str:
    """同参的**矢量出口**（架构 §2.1 矢量产线）——返回 SVG 文档字符串。

    注意：SVG 出口不做后处理（天空渐变/bloom 是位图操作），只证明"同一条绘制序列
    能出矢量"，与位图出口**同源**（都走 `World3D._paint`）。
    """
    w, h = int(size[0]), int(size[1])
    style = style or {}
    return world.render_svg(w, h, camera=camera,
                            background=tuple(style.get("bg", KEY_BG)),
                            ground_color=tuple(style.get("ground", (16, 18, 30))))


def frame_digest(img: "Image.Image") -> str:
    """帧像素的 sha256（前 16 hex）——确定性自证用（不含 PNG 编码器差异）。"""
    import hashlib
    return hashlib.sha256(np.asarray(img.convert("RGB"), dtype=np.uint8).tobytes()).hexdigest()[:16]
