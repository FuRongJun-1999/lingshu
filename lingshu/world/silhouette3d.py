# Copyright 2026 灵枢 (Lingshu) · MIT
"""Silhouette3D — 线条轮廓 + 深度着色（漫画风 **2.5D** 呈现）。

表示层口径（如实声明，#8）：
- 角色 = 一组**轮廓部件**（2D 轮廓点 + 深度 z 偏移 + 基础色），无需网格/贴图。
- **每个部件是零厚度平面多边形**（`SilhouettePart.points` 只有 (x, y)；z 是部件
  级的常数偏移，不是逐顶点深度）。遮挡由**画家算法排序**实现，非深度缓冲。
- 投影后按**相机坐标系深度着色**（见 `render`/`_depth_factor`）：近的部件亮/暖，
  远的暗/冷 —— 颜色即距离。
- 旋转视角时部件产生**视差位移 + 画家算法遮挡**。这是**分层视差（2.5D）**：
  纸片/皮影式表现，**不是**有体积的三维体——侧视 90° 时各平面坍塌成细线。
  据此外推「多视角几何一致性/重建精度」会量到视差而非几何（#8）。

肥鱼（猫娘少女）轮廓模板：猫耳/头/身体(裙)/手臂/腿/尾巴，各部件深度不同。

用法：
    python -m lingshu.world.silhouette3d   # 渲染 3 视角 demo → data/fatfish_*.png
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .svg_backend import SVGBackend
from .world3d import Camera3D

# ---------------------------------------------------------------------------
# 轮廓部件
# ---------------------------------------------------------------------------


@dataclass
class SilhouettePart:
    """一个轮廓部件：2D 轮廓点(相对角色中心，米，y 向上) + 深度 z + 基础色。

    关节支持：pivot（2D 旋转中心）+ angle（弧度）——渲染时先绕 pivot 旋转
    顶点再投影，实现四肢/尾巴摆动（动画与姿态）。

    父子链：`parent` 指向父部件的 name（同一 Silhouette3D 内）。子部件的
    `angle` 是**相对父部件**的角度，父关节旋转会带动整条子链（见
    `Silhouette3D.transformed_points`）——没有 parent 的部件是链根，
    `angle` 即绝对角。
    """
    name: str
    points: List[Tuple[float, float]]    # 轮廓（多边形顶点）
    depth: float = 0.0                   # 部件相对中心的 z 偏移（米，越大越靠角色背面）
    color: Tuple[int, int, int] = (200, 200, 200)
    outline: Tuple[int, int, int] = (40, 30, 35)   # 线条色（漫画描边）
    pivot: Optional[Tuple[float, float]] = None    # 旋转中心（2D，相对角色中心）
    angle: float = 0.0                             # 相对 pivot 的旋转角（弧度，有 parent 时相对父）
    parent: Optional[str] = None                   # 父部件 name（关节链，None=链根）

    def rotated_points(self) -> List[Tuple[float, float]]:
        """返回旋转后的轮廓点（无 pivot 时原样）。"""
        if self.pivot is None or self.angle == 0.0:
            return self.points
        cx, cy = self.pivot
        c, s = math.cos(self.angle), math.sin(self.angle)
        out = []
        for x, y in self.points:
            dx, dy = x - cx, y - cy
            out.append((cx + dx * c - dy * s, cy + dx * s + dy * c))
        return out


class Silhouette3D:
    """轮廓角色：部件集合 + 相机投影 + 深度着色渲染。"""

    def __init__(self, center: Tuple[float, float, float] = (0.0, 0.85, 5.0),
                 depth_span: float = 0.30):
        self.center = center
        self.parts: List[SilhouettePart] = []
        self.depth_span = depth_span       # 深度着色范围（米）：span 内 1.0→0.55 亮度
        # 表情几何基准（#376）：`set_expression` 首次调用时快照眼/嘴/高光的原始
        # 几何，其后每次调用都从这份基准重算 —— 使 set_expression 幂等。
        self._expr_base: Optional[dict] = None
        # 矢量出口的观测面（不静默）：最近一次 `render_svg*` 的 `SVGBackend.stats()`
        # 与最近一次 `animate_svg` 的帧/动画信息；未调用过则 None。
        self.last_svg_stats: Optional[dict] = None
        self.last_anim_info: Optional[dict] = None

    def add_part(self, part: SilhouettePart) -> "Silhouette3D":
        self.parts.append(part)
        return self

    def joint(self, name: str, angle: float) -> "Silhouette3D":
        """设置关节角度（姿态控制）。有 parent 的部件该角度相对父部件。"""
        for p in self.parts:
            if p.name == name:
                p.angle = angle
                break
        return self

    # ---- 关节父子链 ----

    def _chain(self, part: SilhouettePart) -> List[SilhouettePart]:
        """沿 parent 链返回 [本部件, 父, 祖父, …]（环形/悬空 parent 处截断）。"""
        by_name = {p.name: p for p in self.parts}
        chain: List[SilhouettePart] = []
        seen = set()
        cur: Optional[SilhouettePart] = part
        while cur is not None and cur.name not in seen:
            seen.add(cur.name)
            chain.append(cur)
            cur = by_name.get(cur.parent) if cur.parent else None
        return chain

    def transformed_points(self, part: SilhouettePart) -> List[Tuple[float, float]]:
        """部件顶点经「自身旋转 → 父链各关节旋转」后的坐标。

        父关节旋转会带动整条子链（先绕本部件 pivot 施加自身 angle，再依次绕
        父、祖父…的 pivot 施加其 angle）——关节链语义，替代此前各部件各转各的
        断链行为（#404）。
        """
        pts = part.rotated_points()
        for anc in self._chain(part)[1:]:
            if anc.pivot is None or anc.angle == 0.0:
                continue
            cx, cy = anc.pivot
            c, s = math.cos(anc.angle), math.sin(anc.angle)
            pts = [(cx + (x - cx) * c - (y - cy) * s,
                    cy + (x - cx) * s + (y - cy) * c) for x, y in pts]
        return pts

    # ---- 投影 ----

    def _to_3d(self, p2: Tuple[float, float], depth: float) -> Tuple[float, float, float]:
        """轮廓点 → 世界 3D（轮廓在 x-y 平面，z = 中心深度 + 部件深度偏移）。"""
        return (self.center[0] + p2[0], self.center[1] + p2[1],
                self.center[2] + depth)

    def _depth_factor(self, dz: float) -> float:
        """部件相对角色中心的相机前向深度差 → 亮度因子：近（dz 小）亮，远（dz 大）暗。

        `dz` = 部件相机前向深度 − 角色中心相机前向深度（均为
        `Camera3D.depth_of` 读数，与排序同一口径，见 #175）；默认相机下
        `dz` 恰为部件自身的 `depth` 偏移，故默认视角着色与旧口径逐位一致。
        """
        t = max(0.0, min(1.0, dz / max(self.depth_span, 1e-6)))
        return 1.0 - 0.45 * t        # 近 1.0 → 远 0.55

    def _shade(self, base: Tuple[int, int, int], f: float) -> Tuple[int, int, int]:
        return tuple(max(0, min(255, int(c * f))) for c in base)

    # ---- 渲染 ----

    def _default_camera(self) -> Camera3D:
        """默认相机：眼位 = 角色中心 − 3·z，看向角色中心。"""
        return Camera3D.look_at(
            eye=(self.center[0], self.center[1], self.center[2] - 3.0),
            target=self.center)

    def _view(self, screen_w: int, screen_h: int,
              camera: Optional[Camera3D] = None
              ) -> List[Tuple[SilhouettePart, List[Tuple[float, float]], float]]:
        """投影 + 排序内核：返回 `[(part, 屏幕多边形, dz), ...]`（**画家序**：远先画）。

        `dz` = 部件相机前向深度 − 角色中心前向深度（着色与排序同一口径，见下）。

        画家算法：按**相机坐标系前向深度**降序（远先画，近覆盖）。
        缺陷态（#175）：排序与着色都用固定 `part.depth`——只在默认相机
        （eye = center − 3·z）下「正 = 远离相机」成立；相机绕到背后时
        尾巴/四肢（depth 大）实际离相机更近却先画、被身体盖住，着色也反了。
        判据来源：docs/theory/世界模型与语义时空图_完整理论整理与实现路线.md
        §3.2 投影-关联（遮挡/尺度须按相机坐标 z，与 `Camera3D.project` 同一变换），
        与 world3d.Object3D.depth 同口径。

        部件按相机前向深度**只**取决于 `part.depth` 与相机（与关节角无关）
        ⇒ 同一机位下**画家序在帧间不变**——这是 `animate_svg` 能按「每个部件
        一条 `points` 动画」逐帧合成的前提之一（另一个前提是可见部件集合不变，
        见 `animate_svg` 的降级路径）。
        """
        cam = camera or self._default_camera()
        # 每个部件：旋转后顶点 → 3D → 屏幕多边形
        projected = []
        for part in self.parts:
            poly = []
            ok = True
            for p2 in self.transformed_points(part):
                p3 = self._to_3d(p2, part.depth)
                sp = cam.project(p3, screen_w, screen_h)
                if sp is None:
                    ok = False
                    break
                poly.append(sp)
            if ok and len(poly) >= 3:
                projected.append((part, poly))

        z_ref = cam.depth_of(self.center)
        view = []
        for part, poly in projected:
            dz = cam.depth_of(self._to_3d((0.0, 0.0), part.depth)) - z_ref
            view.append((part, poly, dz))
        return sorted(view, key=lambda x: -x[2])

    def _paint(self, draw, view, alpha: bool = False) -> None:
        """绘制内核：只调 `draw.polygon(...)`（与 `ImageDraw` / `SVGBackend` 同形）。

        `view` = `_view(...)` 的输出（画家序）；按序填充 + 描边，深度着色
        与既有实现同一口径。**位图出口与矢量出口共用本内核** ⇒ 绘制顺序、
        坐标、颜色不可能在两出口之间走偏。
        """
        for part, poly, dz in view:
            f = self._depth_factor(dz)
            fill = self._shade(part.color, f)
            if alpha:
                draw.polygon(poly, fill=(*fill, 255), outline=(*part.outline, 255))
            else:
                draw.polygon(poly, fill=fill, outline=part.outline)

    def render(self, screen_w: int = 400, screen_h: int = 500,
               camera: Optional[Camera3D] = None,
               background: Tuple[int, int, int] = (250, 248, 244),
               alpha: bool = False) -> "PIL.Image":
        """渲染：按深度降序（远先画）填充每个轮廓部件 + 描边。返回 PIL Image。

        alpha=True 时返回 RGBA（背景透明，部件不透明）——供 WorldModel 用
        paste 合成（C 级），替代逐像素循环。

        几何/排序在 `_view`，绘制在 `_paint`（与 `render_svg` 共用）——
        本方法只负责建位图 + `ImageDraw.Draw` 后调内核，**输出逐字节不变**。
        """
        from PIL import Image, ImageDraw

        mode = "RGBA" if alpha else "RGB"
        if alpha:
            img = Image.new("RGBA", (screen_w, screen_h), (*background, 0))
        else:
            img = Image.new("RGB", (screen_w, screen_h), background)
        draw = ImageDraw.Draw(img)
        self._paint(draw, self._view(screen_w, screen_h, camera), alpha=alpha)
        return img

    # ---- 矢量出口（SVG · 与位图出口同几何同顺序）----

    def render_svg_backend(self, screen_w: int = 400, screen_h: int = 500,
                           camera: Optional[Camera3D] = None,
                           background: Tuple[int, int, int] = (250, 248, 244),
                           alpha: bool = False) -> SVGBackend:
        """`render` 同参同几何，出口换成 `SVGBackend`（可查 `stats()`，不吞跳过计数）。

        `alpha=True` 时背景按 RGBA `(…, 0)` 发射（与 `render` 的
        `Image.new("RGBA", …, (*background, 0))` 同源）⇒ SVG 里表现为背景矩形的
        `fill-opacity="0"`（与位图路径「背景透明、部件不透明」同义，不静默丢弃
        透明度）。
        """
        bg = (*background, 0) if alpha else background
        b = SVGBackend(screen_w, screen_h, background=bg)
        self._paint(b, self._view(screen_w, screen_h, camera), alpha=alpha)
        self.last_svg_stats = b.stats()
        return b

    def render_svg(self, screen_w: int = 400, screen_h: int = 500,
                   camera: Optional[Camera3D] = None,
                   background: Tuple[int, int, int] = (250, 248, 244),
                   alpha: bool = False) -> str:
        """矢量出口：返回**自包含** SVG 文档（纯标准库，无需 PIL）。

        与 `render(...)` 逐元素同序（画家序），坐标/颜色由 `SVGBackend`
        按固定小数位格式化 ⇒ 同参两次导出**逐字节相同**。
        元素数与跳过计数见 `self.last_svg_stats`（或 `render_svg_backend().stats()`）。

        尺寸口径（如实）：**本出口与本类既有位图出口一样不设单边上限**——
        本件只换出口、不改既有行为，故未在此新增限流（`World3D` 侧另有
        `MAX_SCREEN_DIM = 4096`，#416；两类口径的收口属另一件事）。
        风险面也如实不同：位图按 `w*h*3` 字节分配（实测 6000² ⇒ +142.5 MB），
        矢量出口不做面积分配（实测 6000² ⇒ 9.6 KB 文档）。
        """
        return self.render_svg_backend(screen_w, screen_h, camera,
                                       background, alpha).to_svg()

    # ---- SVG 动画（SMIL · 帧参数由调用方给定 · 确定性）----
    #
    # 为什么选 **SMIL**（而不是 CSS keyframes）：
    #   · 本件要动的是**逐顶点的多边形几何**（尾巴各节的摆动）。SVG 1.1 里
    #     `points` 是 `<animate>` 可直接驱动的属性，且当两张点表顶点数相同时
    #     允许线性插值 ⇒ 摆尾是**真实几何**动画，而不是近似。
    #   · CSS keyframes **不能**可移植地驱动 `points`（它不是 CSS 属性；只有
    #     新版 `d` 等少数属性可用），用它只能把每个部件包一层 `<g>` 做
    #     `transform` 级别的近似位移——摆尾会变成整体平移，几何是错的。
    #   · 自包含性：SMIL 动画就写在元素内部（`<polygon>…<animate/></polygon>`），
    #     不依赖外部样式表/脚本；Chrome / Firefox / Safari / Inkscape 均支持。
    #     代价（如实）：较老的 SVG 精简渲染器只认静态首帧——那会看到帧 0 的样子。
    #
    # 帧间只有关节角（尾巴波幅）变化 ⇒ 部件集合与画家序不变（见 `_view`），
    # 故每个部件一条 `points` 动画、按帧 0 的元素顺序即可精确合成。

    @staticmethod
    def _frame_waves(frames, amplitude: float) -> List[float]:
        """帧参数 → 每帧的尾巴波幅序列（弧度）。

        · `frames` 是 int N ⇒ 在**一个完整正弦周期**上均匀采样 N 帧
          （`wave_i = amplitude·sin(2π(i+0.5)/N)`，i = 0…N−1；N=1 ⇒ 复位姿态 0.0）。
          取**半步偏移**而非 `sin(2πi/N)`：后者在偶数 N 下会把帧全采在正弦的
          零点上（N=2 时两帧波幅都是 0 ⇒ 动画退化为静止），半步偏移使相位
          从 π/N 走到 2π−π/N，首尾相差一个整步 ⇒ 循环连续且每帧都有位移。
        · `frames` 是序列 ⇒ **逐帧显式给定**的波幅（序列长度即帧数）。
        全部由入参决定，**不引入任何随机源** ⇒ 同参数两次导出逐字节相同。
        """
        if isinstance(frames, bool):
            raise TypeError("frames 不能是 bool：%r" % (frames,))
        if isinstance(frames, float) and frames.is_integer():
            return Silhouette3D._frame_waves(int(frames), amplitude)
        if isinstance(frames, int):
            n = int(frames)
            if n < 1:
                raise ValueError("frames 必须 ≥ 1，得到 %r" % (frames,))
            if n == 1:
                return [0.0]
            return [float(amplitude) * math.sin(2.0 * math.pi * (i + 0.5) / n)
                    for i in range(n)]
        seq = [float(w) for w in frames]
        if not seq:
            raise ValueError("frames 序列不能为空")
        return seq

    @staticmethod
    def _svg_polygons(doc: str):
        """扫描文档里的 `<polygon …/>` **自闭合**元素 → `[(起点, 终点, 元素文本), …]`。

        为什么不按行切（曾经的写法，抓到一个真缺陷）：元素是否独占一行属于
        `to_svg()` 的**排版细节**（背景 `<rect>` 的换行有无、缩进几个空格），
        不是契约——冻结后端一次排版改动就让"最后/最前一个元素被漏掉"，表现为
        该部件在动画里**卡住**。这里只依赖 `<polygon>` 自闭合这一**契约级**形态。
        出现非自闭合的 `<polygon …>`（或开标签不成对）⇒ 返回 None，调用方降级。
        """
        out = []
        i = 0
        while True:
            j = doc.find("<polygon", i)
            if j < 0:
                break
            k = doc.find(">", j)
            if k < 0:
                break
            if doc[k - 1] != "/":            # 非自闭合 ⇒ 不能安全注入子元素
                return None
            out.append((j, k + 1, doc[j:k + 1]))
            i = k + 1
        return out

    @staticmethod
    def _points_attr(elem: str) -> Optional[str]:
        """取元素上的 `points="…"` 值；无此属性（非多边形）⇒ None。"""
        elem = elem.strip()          # 行含 4 空格缩进，此处容忍
        if not elem.startswith("<polygon ") or not elem.endswith("/>"):
            return None
        if 'points="' not in elem:
            return None
        return elem.split('points="', 1)[1].split('"', 1)[0]

    def animate_svg(self, frames=24, screen_w: int = 400, screen_h: int = 500,
                    camera: Optional[Camera3D] = None,
                    background: Tuple[int, int, int] = (250, 248, 244),
                    alpha: bool = False, amplitude: float = 0.45,
                    period: float = 2.0, loop: bool = True) -> str:
        """自包含**动画 SVG**：用本模块已有的尾巴波形（`_set_tail_wave`）+ 既有深度着色。

        参数：
          · `frames`  int N ⇒ 一个完整正弦周期上均匀采样 N 帧；序列 ⇒ 逐帧显式波幅。
          · `amplitude` 正弦波幅（弧度，默认 0.45）；`period` 一个周期的秒数（dur）。
          · `loop=True` ⇒ `repeatCount="indefinite"`；`False` ⇒ `fill="freeze"`（停在末帧）。
          · 其余（尺寸/相机/背景/alpha）与 `render_svg` 同义。

        帧数语义：动画元素 `values="f0;f1;…;f_{N-1}"` ⇒ 以 `;` 切分的帧数**恒等于**请求的
        N，时间轴 `dur = period`。单帧（N=1）**退化为静态 SVG**（不发 `<animate>`，不崩）。

        诚实的降级（结果都在 `self.last_anim_info` 里，不静默）：
          · 帧间可见部件集合/元素数变化（例如某帧被近平面裁掉）⇒ 返回帧 0 的**静态** SVG，
            `mode="static"`、`reason="element_count_varies"` / `"part_set_varies"`；
          · 帧间几何完全相同（例如轮廓没有尾巴，波形驱动不到任何部件）⇒ 同样返回静态 SVG，
            `reason="no_motion"` —— 不伪造一个"其实没动"的动画。
        调用结束后**复原**全部关节角（不会把动画姿态留在对象上）。逐帧的
        发射/跳过合计见 `self.last_anim_info["skipped"]`（动画不改 `last_svg_stats`）。
        """
        waves = self._frame_waves(frames, amplitude)
        n = len(waves)
        if not math.isfinite(period) or period <= 0:
            raise ValueError("period 必须是有限正数（秒），得到 %r" % (period,))

        bg = (*background, 0) if alpha else background
        saved = [p.angle for p in self.parts]
        docs: List[str] = []
        names_per_frame: List[List[str]] = []
        skipped: dict = {}
        try:
            for w in waves:
                _set_tail_wave(self, w)
                b = SVGBackend(screen_w, screen_h, background=bg)
                view = self._view(screen_w, screen_h, camera)
                self._paint(b, view, alpha=alpha)
                docs.append(b.to_svg())
                names_per_frame.append([p.name for p, _poly, _dz in view])
                for k, v in b.stats()["skipped"].items():
                    skipped[k] = skipped.get(k, 0) + v
        finally:
            for p, a0 in zip(self.parts, saved):
                p.angle = a0

        e0 = self._svg_polygons(docs[0])
        spans = [self._svg_polygons(d) for d in docs]
        reason: Optional[str] = None
        animated = 0
        if n == 1:
            reason = "single_frame"
        elif e0 is None or any(sp is None for sp in spans):
            reason = "non_polygon_element"
        elif any(len(spans[k]) != len(names_per_frame[k]) for k in range(n)):
            # 后端跳过了元素（退化多边形等）⇒ 元素与部件不再一一对应，不硬猜
            reason = "element_count_varies"
        elif any(ns != names_per_frame[0] for ns in names_per_frame[1:]):
            reason = "part_set_varies"
        else:
            cols = [[self._points_attr(spans[k][i][2]) for k in range(n)]
                    for i in range(len(e0))]
            if any(None in c for c in cols):
                reason = "non_polygon_element"
            else:
                moving = [i for i, c in enumerate(cols) if len(set(c)) > 1]
                animated = len(moving)
                if not moving:
                    reason = "no_motion"

        if reason is not None:
            svg = docs[0]
        else:
            dur = "%gs" % float(period)
            out: List[str] = []
            pos = 0
            for i, (start, end, txt) in enumerate(e0):
                out.append(docs[0][pos:start])
                pos = end
                col = cols[i]
                if len(set(col)) <= 1:
                    out.append(txt)         # 静止部件：保持普通 `<polygon .../>`
                    continue
                child = ('<animate attributeName="points" values="%s" dur="%s" '
                         'calcMode="linear"%s/>'
                         % (";".join(col), dur,
                            ' repeatCount="indefinite"' if loop else ' fill="freeze"'))
                out.append(txt[:-2] + ">" + child + "</polygon>")
            out.append(docs[0][pos:])
            svg = "".join(out)

        mode = "static" if reason is not None else "smil"
        self.last_anim_info = {
            "frames": n, "mode": mode, "reason": reason,
            "animated_parts": animated if mode == "smil" else 0,
            "static_parts": (len(e0) - animated) if mode == "smil" else len(e0),
            "duration_s": float(period), "loop": bool(loop),
            "waves": list(waves), "elements_per_frame": len(e0),
            "skipped": skipped,
        }
        return svg


# ---------------------------------------------------------------------------
# 肥鱼（猫娘少女）轮廓模板
# ---------------------------------------------------------------------------


def _ellipse(center: Tuple[float, float], rx: float, ry: float,
             n: int = 14) -> List[Tuple[float, float]]:
    """椭圆轮廓点（center 中心，rx/ry 半轴，n 顶点）。"""
    cx, cy = center
    return [(cx + rx * math.cos(2 * math.pi * i / n),
             cy + ry * math.sin(2 * math.pi * i / n)) for i in range(n)]


def fatfish_skinned(center: Tuple[float, float, float] = (0, 0.85, 5.0),
                    height: float = 1.55, lod: int = 3) -> Silhouette3D:
    """肥鱼**蒙皮版**基础轮廓（无服装层）——从骨架关节坐标推导（骨架=基准，轮廓=面）。

    与 fatfish_skeleton（手调坐标）的关键区别：
    - 所有身体部件（头/躯干/四肢/尾巴）的轮廓点**从 fatfish_bone_skeleton
      关节的世界坐标直接推导**，保证骨架线条叠加时关节精确落在轮廓内
      （"身体骨架和外部轮廓一致"，去服装层/叠加骨架线时轮廓始终贴合）。
    - 基础层：全身皮肤色，不含服装；服装是后续可叠加的外层。

    用法：
        ff = fatfish_skinned()
        img = ff.render(...)          # 基础层
        img = ff.render(..., bones=True 由 WorldModel 叠骨架)  # 骨架+面
    """
    from .skeleton3d import fatfish_bone_skeleton

    h = height
    sk = fatfish_bone_skeleton(center=center, height=h)
    # 关节世界坐标（相对角色中心）
    def J(name: str) -> Tuple[float, float]:
        p = sk.world_pos(name)
        return (p[0] - center[0], p[1] - center[1])

    s = Silhouette3D(center=center, depth_span=0.30)
    skin = (246, 214, 198)
    fur = (250, 228, 214)
    line = (45, 35, 40)

    # ---- 头（head 关节中心，半径 = head→head_top 距离）----
    hc = J("head")
    head_r = abs(J("head_top")[1] - hc[1]) * 1.15
    s.add_part(SilhouettePart("head", _ellipse(hc, head_r * 0.9, head_r, 16),
                              depth=0.00, color=skin, outline=line))

    # ---- head 区特征（像素画/动漫经典规则：大眼+瞳孔+双高光+精致小嘴）----
    eye_dark = (60, 45, 60)          # 眼白深色（像素画上眼睑粗线感）
    pupil = (35, 25, 40)             # 瞳孔（更深的黑）
    eye_light = (255, 255, 255)
    blush = (255, 175, 190)
    mouth = (185, 85, 100)
    hair = (185, 150, 130)
    ey = hc[1] - head_r * 0.02       # 眼睛中心（动漫大眼：下置显可爱）
    eye_dx = head_r * 0.26           # 眼距（像素画：一瞳距）
    eye_rx = head_r * 0.14           # 眼半宽（占头宽 ~28%，动漫大眼）
    eye_ry = head_r * 0.19           # 眼半高（椭圆）
    # 左眼（上眼睑粗线 = 上部深色，下部浅一点营造渐变）
    for side in ("l", "r"):
        sgn = -1 if side == "l" else 1
        cx = hc[0] + sgn * eye_dx
        eye = [(cx - eye_rx, ey - eye_ry), (cx - eye_rx, ey + eye_ry * 0.5),
               (cx + eye_rx, ey + eye_ry * 0.5), (cx + eye_rx, ey - eye_ry)]
        s.add_part(SilhouettePart(f"eye_{side}", eye, depth=-0.005,
                                  color=eye_dark, outline=eye_dark))
        # 瞳孔（深色小圆，眼中心）
        s.add_part(SilhouettePart(f"pupil_{side}", _ellipse((cx, ey - eye_ry * 0.1),
                                                            eye_rx * 0.5, eye_ry * 0.55, 8),
                                  depth=-0.006, color=pupil, outline=pupil))
        # 双高光：大（左上）+ 小（右下）——像素画标志，加大更闪亮
        hl_big = head_r * 0.07
        hl_small = head_r * 0.035
        big = [(cx - eye_rx * 0.35 - hl_big, ey - eye_ry * 0.4 - hl_big),
               (cx - eye_rx * 0.35 + hl_big, ey - eye_ry * 0.4 - hl_big),
               (cx - eye_rx * 0.35 + hl_big, ey - eye_ry * 0.4 + hl_big),
               (cx - eye_rx * 0.35 - hl_big, ey - eye_ry * 0.4 + hl_big)]
        small = [(cx + eye_rx * 0.3 - hl_small, ey + eye_ry * 0.25 - hl_small),
                 (cx + eye_rx * 0.3 + hl_small, ey + eye_ry * 0.25 - hl_small),
                 (cx + eye_rx * 0.3 + hl_small, ey + eye_ry * 0.25 + hl_small),
                 (cx + eye_rx * 0.3 - hl_small, ey + eye_ry * 0.25 + hl_small)]
        s.add_part(SilhouettePart(f"highlight_{side}_big", big, depth=-0.007,
                                  color=eye_light, outline=eye_light))
        s.add_part(SilhouettePart(f"highlight_{side}_small", small, depth=-0.007,
                                  color=eye_light, outline=eye_light))
        # 瞳孔中心白点（第三高光——动漫眼睛"闪亮"灵魂）
        hl_center = head_r * 0.03
        ctr = [(cx - hl_center, ey - eye_ry * 0.1 - hl_center),
               (cx + hl_center, ey - eye_ry * 0.1 - hl_center),
               (cx + hl_center, ey - eye_ry * 0.1 + hl_center),
               (cx - hl_center, ey - eye_ry * 0.1 + hl_center)]
        s.add_part(SilhouettePart(f"highlight_{side}_center", ctr, depth=-0.007,
                                  color=eye_light, outline=eye_light))
    # 腮红（小圆，眼下外侧——像素画：两点腮红）
    bl_r = head_r * 0.09
    for side in ("l", "r"):
        sgn = -1 if side == "l" else 1
        bx = hc[0] + sgn * (eye_dx + eye_rx * 1.4)
        by = ey - head_r * 0.02
        s.add_part(SilhouettePart(f"blush_{side}", _ellipse((bx, by), bl_r, bl_r * 0.8, 8),
                                  depth=-0.004, color=blush, outline=blush))
    # 嘴（精致小嘴：微笑弧 + 嘴角上翘）
    mouth_pts = [(hc[0] - head_r * 0.07, ey - head_r * 0.28),
                 (hc[0] - head_r * 0.02, ey - head_r * 0.34),
                 (hc[0] + head_r * 0.02, ey - head_r * 0.34),
                 (hc[0] + head_r * 0.07, ey - head_r * 0.28),
                 (hc[0] + head_r * 0.04, ey - head_r * 0.24),
                 (hc[0] - head_r * 0.04, ey - head_r * 0.24)]
    s.add_part(SilhouettePart("mouth", mouth_pts, depth=-0.004, color=mouth, outline=mouth))
    # 刘海（额前发帘：头顶弧条）
    fringe = [(hc[0] - head_r * 0.95, hc[1] + head_r * 0.30),
              (hc[0] - head_r * 0.35, hc[1] + head_r * 0.78),
              (hc[0], hc[1] + head_r * 0.85),
              (hc[0] + head_r * 0.35, hc[1] + head_r * 0.78),
              (hc[0] + head_r * 0.95, hc[1] + head_r * 0.30),
              (hc[0] + head_r * 0.98, hc[1] + head_r * 0.10),
              (hc[0] - head_r * 0.98, hc[1] + head_r * 0.10)]
    s.add_part(SilhouettePart("fringe", fringe, depth=-0.002, color=hair, outline=(45, 35, 40)))

    # 眉毛（情绪表达，head 关节推导；眉在眼睛上方 = ey +）
    brow = (85, 70, 80)
    brow_y = ey + eye_ry * 1.6           # 眉在眼上方
    for side in ("l", "r"):
        sgn = -1 if side == "l" else 1
        bx = hc[0] + sgn * eye_dx
        brow_pts = [(bx - eye_rx * 1.2, brow_y + head_r * 0.03),
                    (bx - eye_rx * 0.3, brow_y - head_r * 0.02),
                    (bx + eye_rx * 0.5, brow_y - head_r * 0.02),
                    (bx + eye_rx * 1.2, brow_y + head_r * 0.03)]
        s.add_part(SilhouettePart(f"brow_{side}", brow_pts, depth=-0.006,
                                  color=brow, outline=brow))
    # 唇色分层（嘴下方区域 = ey -；上唇在嘴上方一点 = 靠近嘴）
    lip_up = (170, 80, 95)
    lip_lo = (205, 120, 130)
    for side in ("l", "r"):
        sgn = -1 if side == "l" else 1
        # 上唇（嘴角到唇珠，位于嘴线稍上方 = ey - 0.18r 区）
        up = [(hc[0] + sgn * head_r * 0.16, ey - head_r * 0.18),
              (hc[0] + sgn * head_r * 0.05, ey - head_r * 0.13),
              (hc[0], ey - head_r * 0.17),
              (hc[0] + sgn * head_r * 0.05, ey - head_r * 0.22)]
        s.add_part(SilhouettePart(f"lip_up_{side}", up, depth=-0.0045,
                                  color=lip_up, outline=lip_up))
    # 下唇高光（嘴下方 = ey - 0.30r 区）
    lip_gloss = [(hc[0] - head_r * 0.06, ey - head_r * 0.30),
                 (hc[0] + head_r * 0.06, ey - head_r * 0.30),
                 (hc[0] + head_r * 0.04, ey - head_r * 0.34),
                 (hc[0] - head_r * 0.04, ey - head_r * 0.34)]
    s.add_part(SilhouettePart("lip_gloss", lip_gloss, depth=-0.0045,
                              color=(235, 190, 195), outline=(235, 190, 195)))

    # ---- 躯干+腹部（hip→clavicle，腰腹曲线：肩→胸→腰→髋 环绕顺序）----
    hip = J("hip")
    clav = J("clavicle")
    waist_j = J("waist")
    chest_j = J("chest")
    hip_w = 0.10 * h                   # 髋部宽（骨盆）
    waist_w = 0.062 * h                # 腰部宽（最窄，腹部的"腰"）
    chest_w = 0.088 * h                # 胸廓宽
    shoulder_w = 0.082 * h             # 肩宽（≤胸廓宽，避免肩成凹点）
    torso = [
        (clav[0] - shoulder_w, clav[1] - 0.01 * h),    # 左肩
        (chest_j[0] - chest_w, chest_j[1]),            # 左胸侧
        (waist_j[0] - waist_w, waist_j[1]),            # 左腰（收窄）
        (hip[0] - hip_w, hip[1] - 0.02 * h),           # 左髋（放宽）
        (hip[0] + hip_w, hip[1] - 0.02 * h),           # 右髋
        (waist_j[0] + waist_w, waist_j[1]),            # 右腰
        (chest_j[0] + chest_w, chest_j[1]),            # 右胸侧
        (clav[0] + shoulder_w, clav[1] - 0.01 * h),    # 右肩
    ]
    s.add_part(SilhouettePart("torso", torso, depth=0.06, color=skin, outline=line))

    # ---- chest（注视偏好第二优先：胸廓体积，chest 关节推导）----
    chest_j2 = J("chest")
    bust_light = (232, 222, 240)         # 柔和粉白高光
    bust_shade = (170, 200, 235)         # 蓝调描边
    bust_r = chest_w * 0.72              # 胸椭圆半宽
    bust_ry = chest_w * 0.95             # 胸椭圆半高
    for side in ("l", "r"):
        sgn = -1 if side == "l" else 1
        bc = (chest_j2[0] + sgn * chest_w * 0.9, chest_j2[1] - bust_ry * 0.55)
        s.add_part(SilhouettePart(f"bust_{side}", _ellipse(bc, bust_r, bust_ry, 14),
                                  depth=0.058, color=bust_light, outline=bust_shade))
    # 胸廓高光条
    chest_mid = [(chest_j2[0] - chest_w * 0.12, chest_j2[1] - bust_ry * 0.1),
                 (chest_j2[0] + chest_w * 0.12, chest_j2[1] - bust_ry * 0.1),
                 (chest_j2[0] + chest_w * 0.09, chest_j2[1] + bust_ry * 0.5),
                 (chest_j2[0] - chest_w * 0.09, chest_j2[1] + bust_ry * 0.5)]
    s.add_part(SilhouettePart("chest_mid", chest_mid, depth=0.057,
                              color=(215, 228, 248), outline=(215, 228, 248)))
    # 胸廓间曲线（两胸廓间暗缝，增强体积分离感）
    cleft = [(chest_j2[0] - chest_w * 0.08, chest_j2[1] - bust_ry * 0.3),
             (chest_j2[0], chest_j2[1] - bust_ry * 0.75),
             (chest_j2[0] + chest_w * 0.08, chest_j2[1] - bust_ry * 0.3),
             (chest_j2[0], chest_j2[1] - bust_ry * 0.1)]
    s.add_part(SilhouettePart("cleft", cleft, depth=0.057,
                              color=(160, 185, 225), outline=(160, 185, 225)))
    # 胸廓下阴影（胸廓下方弧线，体积感）
    under_bust = [(chest_j2[0] - chest_w * 0.75, chest_j2[1] - bust_ry * 0.55),
                  (chest_j2[0] - chest_w * 0.9, chest_j2[1] - bust_ry * 0.15),
                  (chest_j2[0] - chest_w * 0.7, chest_j2[1] - bust_ry * 0.1)]
    s.add_part(SilhouettePart("under_bust_l", under_bust, depth=0.057,
                              color=(175, 200, 235), outline=(175, 200, 235)))
    under_bust_r = [(chest_j2[0] + chest_w * 0.75, chest_j2[1] - bust_ry * 0.55),
                    (chest_j2[0] + chest_w * 0.9, chest_j2[1] - bust_ry * 0.15),
                    (chest_j2[0] + chest_w * 0.7, chest_j2[1] - bust_ry * 0.1)]
    s.add_part(SilhouettePart("under_bust_r", under_bust_r, depth=0.057,
                              color=(175, 200, 235), outline=(175, 200, 235)))

    # ---- hip（注视偏好第三优先：髋部体积，hip 关节推导）----
    hip_shade = (105, 130, 195)          # 深蓝紫阴影（与裙身/肤色区分）
    hip_br = hip_w * 0.62                # 髋椭圆半宽
    hip_bry = hip_w * 0.95               # 髋椭圆半高
    for side in ("l", "r"):
        sgn = -1 if side == "l" else 1
        hc2 = (hip[0] + sgn * hip_w * 1.25, hip[1] - hip_bry * 0.55)
        s.add_part(SilhouettePart(f"hip_{side}", _ellipse(hc2, hip_br, hip_bry, 12),
                                  depth=0.058, color=hip_shade, outline=(80, 105, 165)))
    # 髋沟线（髋部之间，骨盆中线下方）
    buttock_cleft = [(hip[0] - hip_w * 0.25, hip[1] - hip_bry * 0.6),
                     (hip[0], hip[1] - hip_bry * 0.2),
                     (hip[0] + hip_w * 0.25, hip[1] - hip_bry * 0.6),
                     (hip[0], hip[1] - hip_bry * 0.75)]
    s.add_part(SilhouettePart("buttock_cleft", buttock_cleft, depth=0.058,
                              color=(95, 120, 180), outline=(95, 120, 180)))
    # 骨盆下缘曲线（身体完整性：骨盆到髋下缘的轮廓线，漫画风暗示）
    pelvis_low = [(hip[0] - hip_w * 0.85, hip[1] - hip_bry * 0.95),
                  (hip[0] - hip_w * 0.35, hip[1] - hip_bry * 1.15),
                  (hip[0], hip[1] - hip_bry * 1.2),
                  (hip[0] + hip_w * 0.35, hip[1] - hip_bry * 1.15),
                  (hip[0] + hip_w * 0.85, hip[1] - hip_bry * 0.95)]
    s.add_part(SilhouettePart("pelvis_low", pelvis_low, depth=0.059,
                              color=(120, 145, 200), outline=(120, 145, 200)))
    # 大腿根/腹股沟衔接（大腿与躯干的过渡轮廓，漫画风）
    groin_l = [(hip[0] - hip_w * 0.6, hip[1] - hip_bry * 1.0),
               (hip[0] - hip_w * 0.25, hip[1] - hip_bry * 1.05),
               (hip[0] - hip_w * 0.4, hip[1] - hip_bry * 1.3)]
    s.add_part(SilhouettePart("groin_l", groin_l, depth=0.058,
                              color=(110, 135, 190), outline=(110, 135, 190)))
    groin_r = [(hip[0] + hip_w * 0.6, hip[1] - hip_bry * 1.0),
               (hip[0] + hip_w * 0.25, hip[1] - hip_bry * 1.05),
               (hip[0] + hip_w * 0.4, hip[1] - hip_bry * 1.3)]
    s.add_part(SilhouettePart("groin_r", groin_r, depth=0.058,
                              color=(110, 135, 190), outline=(110, 135, 190)))

    # ---- 脖子（clavicle→neck，躯干到头之间的连续段）----
    clav = J("clavicle")
    nk = J("neck")
    neck_w = 0.035 * h
    neck = [(clav[0] - neck_w * 0.6, clav[1]),
            (clav[0] + neck_w * 0.6, clav[1]),
            (nk[0] + neck_w * 0.8, nk[1]),
            (nk[0] - neck_w * 0.8, nk[1])]
    s.add_part(SilhouettePart("neck", neck, depth=0.04, color=skin, outline=line))

    # ---- 手臂（肩→肘→腕 皮肤色，宽度 0.045h）----
    arm_w = 0.045 * h
    for side in ("l", "r"):
        sh = J(f"shoulder_{side}")
        el = J(f"elbow_{side}")
        wr = J(f"wrist_{side}")
        # 上臂（肩→肘）：围绕关节对称展开
        upper = [(sh[0] - arm_w * 0.7, sh[1]),
                 (sh[0] + arm_w * 0.7, sh[1]),
                 (el[0] + arm_w * 0.5, el[1]),
                 (el[0] - arm_w * 0.5, el[1])]
        s.add_part(SilhouettePart(f"upper_{side}", upper, depth=0.10, color=skin, outline=line,
                                  pivot=sh, angle=0.0))
        # 下臂（肘→腕）
        lower = [(el[0] - arm_w * 0.5, el[1]),
                 (el[0] + arm_w * 0.5, el[1]),
                 (wr[0] + arm_w * 0.35, wr[1]),
                 (wr[0] - arm_w * 0.35, wr[1])]
        s.add_part(SilhouettePart(f"lower_{side}", lower, depth=0.10, color=skin, outline=line,
                                  pivot=el, angle=0.0, parent=f"upper_{side}"))

    # ---- 腿（髋→膝→踝 皮肤色，含大腿上段连接躯干）----
    hip_j = J("hip")                     # 骨盆（躯干底）
    leg_w = 0.062 * h                    # 大腿宽
    calf_w = 0.045 * h                   # 小腿宽（比大腿细）
    hip_w_leg = 0.085 * h                # 大腿上段（胯部）宽
    for side in ("l", "r"):
        hp = J(f"thigh_{side}")
        kn = J(f"knee_{side}")
        an = J(f"ankle_{side}")
        # 大腿上段（骨盆→髋关节）：从躯干底过渡到大腿，覆盖整段
        thigh_top = [(hip_j[0] - hip_w_leg, hip_j[1] - 0.02 * h),
                     (hip_j[0] + hip_w_leg, hip_j[1] - 0.02 * h),
                     (hp[0] + leg_w, hp[1] - 0.005 * h),
                     (hp[0] - leg_w, hp[1] - 0.005 * h)]
        s.add_part(SilhouettePart(f"thigh_top_{side}", thigh_top, depth=0.13, color=skin,
                                  outline=line, pivot=hp, angle=0.0))
        # 大腿（髋→膝）
        thigh = [(hp[0] - leg_w, hp[1]),
                 (hp[0] + leg_w, hp[1]),
                 (kn[0] + leg_w * 0.8, kn[1]),
                 (kn[0] - leg_w * 0.8, kn[1])]
        s.add_part(SilhouettePart(f"thigh_{side}", thigh, depth=0.14, color=skin, outline=line,
                                  pivot=hp, angle=0.0))
        # 小腿（膝→踝，比大腿细）
        calf = [(kn[0] - calf_w, kn[1]),
                (kn[0] + calf_w, kn[1]),
                (an[0] + calf_w * 0.7, an[1]),
                (an[0] - calf_w * 0.7, an[1])]
        s.add_part(SilhouettePart(f"calf_{side}", calf, depth=0.14, color=skin, outline=line,
                                  pivot=kn, angle=0.0, parent=f"thigh_{side}"))

    # ---- 猫耳（head_top 关节，加大 + 内耳粉色——猫娘标志特征）----
    # 视觉验收修复（2026-08-29）：底边 0.045h 过宽呈喇叭状——收窄为立耳三角
    ear_inner = (255, 190, 200)          # 内耳粉（像素画猫娘特征）
    for side in ("l", "r"):
        sgn = -1 if side == "l" else 1
        eb = J(f"ear_{side}_base")
        et = J(f"ear_{side}_tip")
        # 外耳（立耳三角：底宽 0.028h，顶随骨架 tip 内收）
        ear = [(eb[0] + sgn * 0.028 * h, eb[1]),
               (et[0] - sgn * 0.012 * h, et[1] + 0.012 * h),
               (et[0] + sgn * 0.012 * h, et[1] + 0.012 * h),
               (eb[0] - sgn * 0.028 * h, eb[1])]
        s.add_part(SilhouettePart(f"ear_{side}", ear, depth=0.02, color=fur, outline=line,
                                  pivot=eb, angle=0.0))
        # 内耳（粉色小三角，猫耳标志）
        inner = [(eb[0] + sgn * 0.011 * h, eb[1] - 0.008 * h),
                 (et[0] - sgn * 0.005 * h, et[1] + 0.012 * h),
                 (et[0] + sgn * 0.005 * h, et[1] + 0.012 * h),
                 (eb[0] - sgn * 0.011 * h, eb[1] - 0.008 * h)]
        s.add_part(SilhouettePart(f"ear_inner_{side}", inner, depth=0.019,
                                  color=ear_inner, outline=ear_inner))

    # ---- 尾巴（tail0→tail4 关节链，5 节，宽度渐细）----
    tail_w = 0.028 * h
    prev = "tail0"
    tpts = [J(prev)]
    for i in range(1, 5):
        tpts.append(J(f"tail{i}"))
    for i in range(4):
        p0, p1 = tpts[i], tpts[i + 1]
        w0 = tail_w * (1 - i * 0.18)
        w1 = tail_w * (1 - (i + 1) * 0.18)
        seg = [(p0[0] - w0, p0[1]), (p0[0] + w0, p0[1]),
               (p1[0] + w1, p1[1]), (p1[0] - w1, p1[1])]
        s.add_part(SilhouettePart(f"tail{i + 1}", seg, depth=0.22 + i * 0.01,
                                  color=fur, outline=line, pivot=p0, angle=0.0,
                                  parent=(f"tail{i}" if i >= 1 else None)))
    # ---- LOD 细节层次（注意力深度 → 部件过滤）----
    # L3（高注意力/特写）：全部部件
    # L2（中）：保留关键特征（眼/嘴/胸廓髋部/躯干四肢），省略细部（眉/唇高光/骨盆线）
    # L1（低/整体）：只留轮廓大块（头/躯干/四肢/尾），五官省略（防挤成一团）
    if lod <= 2:
        _L2_OMIT = {"brow_l", "brow_r", "lip_up_l", "lip_up_r", "lip_gloss",
                    "cleft", "under_bust_l", "under_bust_r", "buttock_cleft",
                    "pelvis_low", "groin_l", "groin_r",
                    "pupil_l", "pupil_r",
                    "highlight_l_big", "highlight_r_big",
                    "highlight_l_small", "highlight_r_small",
                    "highlight_l_center", "highlight_r_center"}
        s.parts = [p for p in s.parts if p.name not in _L2_OMIT]
    if lod <= 1:
        _L1_OMIT = {"eye_l", "eye_r", "mouth", "blush_l", "blush_r",
                    "bust_l", "bust_r", "chest_mid", "hip_l", "hip_r",
                    "ear_inner_l", "ear_inner_r",
                    "ear_l", "ear_r", "tail1", "tail2", "tail3", "tail4"}
        s.parts = [p for p in s.parts if p.name not in _L1_OMIT]
    return s


def fatfish_skeleton(center: Tuple[float, float, float] = (0, 0.85, 5.0),
                     height: float = 1.55) -> Silhouette3D:
    """肥鱼猫娘轮廓：猫耳/头/裙体/手臂/腿/尾巴，深度分层（近亮远暗）。"""
    h = height
    s = Silhouette3D(center=center, depth_span=0.30)
    skin = (246, 214, 198)
    fur = (250, 228, 214)        # 猫耳/尾巴（奶白粉）
    dress = (150, 180, 235)      # 裙子（蓝）
    line = (45, 35, 40)

    # 头（最近 z=0.00）
    head_c = (0.0, h * 0.86)
    head_r = h * 0.115
    head_pts = []
    for i in range(16):
        a = 2 * math.pi * i / 16
        head_pts.append((head_c[0] + head_r * math.cos(a),
                         head_c[1] + head_r * math.sin(a)))
    s.add_part(SilhouettePart("head", head_pts, depth=0.00, color=skin, outline=line))

    # ---- head 区特征（玩家注视偏好第一优先：脸）----
    eye_dark = (70, 55, 70)          # 眼睛深色
    eye_light = (255, 255, 255)      # 高光
    blush = (255, 180, 190)          # 腮红
    mouth = (180, 90, 100)           # 嘴
    hair = (170, 140, 200)           # 刘海（淡紫，呼应裙子蓝？换暖色更搭）→ 用浅褐 (180,150,130)
    hair = (185, 150, 130)
    ey = h * 0.86                     # 眼睛中心高度
    eye_dx = h * 0.045                # 眼距
    eye_r = h * 0.020                 # 眼半径
    # 左眼（椭圆：宽>高，漫画风）
    eye_l = [(head_c[0] - eye_dx - eye_r * 1.3, ey - eye_r * 0.6),
             (head_c[0] - eye_dx - eye_r * 1.3, ey + eye_r * 0.6),
             (head_c[0] - eye_dx + eye_r * 1.3, ey + eye_r * 0.6),
             (head_c[0] - eye_dx + eye_r * 1.3, ey - eye_r * 0.6)]
    s.add_part(SilhouettePart("eye_l", eye_l, depth=-0.005, color=eye_dark, outline=eye_dark))
    # 右眼
    eye_rp = [(head_c[0] + eye_dx - eye_r * 1.3, ey - eye_r * 0.6),
              (head_c[0] + eye_dx - eye_r * 1.3, ey + eye_r * 0.6),
              (head_c[0] + eye_dx + eye_r * 1.3, ey + eye_r * 0.6),
              (head_c[0] + eye_dx + eye_r * 1.3, ey - eye_r * 0.6)]
    s.add_part(SilhouettePart("eye_r", eye_rp, depth=-0.005, color=eye_dark, outline=eye_dark))
    # 眼睛高光（白色小点，左上）
    hl = h * 0.008
    for side in ("l", "r"):
        cx = head_c[0] - eye_dx if side == "l" else head_c[0] + eye_dx
        hl_pts = [(cx - eye_r * 0.6 - hl, ey - eye_r * 0.3 - hl),
                  (cx - eye_r * 0.6 + hl, ey - eye_r * 0.3 - hl),
                  (cx - eye_r * 0.6 + hl, ey - eye_r * 0.3 + hl),
                  (cx - eye_r * 0.6 - hl, ey - eye_r * 0.3 + hl)]
        s.add_part(SilhouettePart(f"highlight_{side}", hl_pts, depth=-0.006,
                                  color=eye_light, outline=eye_light))
    # 腮红（粉，眼下外侧）
    bl_r = h * 0.016
    for side in ("l", "r"):
        bx = head_c[0] - eye_dx - eye_r * 2.2 if side == "l" else head_c[0] + eye_dx + eye_r * 2.2
        by = ey - h * 0.012
        blush_pts = [(bx - bl_r, by - bl_r * 0.5), (bx - bl_r, by + bl_r * 0.5),
                     (bx + bl_r, by + bl_r * 0.5), (bx + bl_r, by - bl_r * 0.5)]
        s.add_part(SilhouettePart(f"blush_{side}", blush_pts, depth=-0.004,
                                  color=blush, outline=blush))
    # 嘴（微笑小弧，用细长多边形）
    mouth_pts = [(head_c[0] - h * 0.018, ey - h * 0.045),
                 (head_c[0] + h * 0.018, ey - h * 0.045),
                 (head_c[0] + h * 0.008, ey - h * 0.055),
                 (head_c[0] - h * 0.008, ey - h * 0.055)]
    s.add_part(SilhouettePart("mouth", mouth_pts, depth=-0.004, color=mouth, outline=mouth))
    # 刘海（额前发帘：头顶弧条）
    fringe = [(head_c[0] - head_r * 0.95, head_c[1] + head_r * 0.30),
              (head_c[0] - head_r * 0.35, head_c[1] + head_r * 0.75),
              (head_c[0], head_c[1] + head_r * 0.82),
              (head_c[0] + head_r * 0.35, head_c[1] + head_r * 0.75),
              (head_c[0] + head_r * 0.95, head_c[1] + head_r * 0.30),
              (head_c[0] + head_r * 0.98, head_c[1] + head_r * 0.10),
              (head_c[0] - head_r * 0.98, head_c[1] + head_r * 0.10)]
    s.add_part(SilhouettePart("fringe", fringe, depth=-0.002, color=hair, outline=line))

    # 猫耳（z=0.02，稍后）
    ear_l = [(-0.155 * h, 0.97 * h), (-0.11 * h, 1.10 * h), (-0.02 * h, 0.94 * h)]
    ear_r = [(0.155 * h, 0.97 * h), (0.11 * h, 1.10 * h), (0.02 * h, 0.94 * h)]
    s.add_part(SilhouettePart("ear_l", ear_l, depth=0.02, color=fur, outline=line))
    s.add_part(SilhouettePart("ear_r", ear_r, depth=0.02, color=fur, outline=line))

    # 身体（A 字裙，z=0.06）——裙身轮廓带髋部鼓出弧度
    body = [(-0.17 * h, 0.72 * h), (0.17 * h, 0.72 * h),
            (0.27 * h, 0.55 * h),          # 裙摆右侧鼓起（髋）
            (0.26 * h, 0.42 * h),
            (0.19 * h, 0.40 * h),          # 裙摆下沿内收
            (-0.19 * h, 0.40 * h),
            (-0.26 * h, 0.42 * h),
            (-0.27 * h, 0.55 * h)]         # 裙摆左侧鼓起（髋）
    s.add_part(SilhouettePart("body", body, depth=0.06, color=dress, outline=line))

    # 躯干（皮肤色基础层，z=0.065 在裙子后）——脱裙也有身体，不是"空壳"
    # 顶部低于裙子领口线（0.72h），确保穿裙时完全被覆盖
    torso = [(-0.12 * h, 0.70 * h), (0.12 * h, 0.70 * h),
             (0.14 * h, 0.62 * h), (0.14 * h, 0.50 * h),
             (-0.14 * h, 0.50 * h), (-0.14 * h, 0.62 * h)]
    s.add_part(SilhouettePart("torso", torso, depth=0.065, color=skin, outline=line))

    # ---- chest 区（体积化：两个椭圆色块 + 高光，漫画风）----
    bust_light = (232, 222, 240)           # 胸廓高光色（柔和粉白）
    bust_shade = (170, 200, 235)           # 胸廓阴影（蓝调）
    # 左胸廓椭圆（相对身体上部，带弧度）
    bust_l = _ellipse((head_c[0] - h * 0.085, 0.66 * h),
                      h * 0.050, h * 0.065)
    s.add_part(SilhouettePart("bust_l", bust_l, depth=0.058,
                              color=bust_light, outline=bust_shade))
    # 右胸廓椭圆
    bust_r = _ellipse((head_c[0] + h * 0.085, 0.66 * h),
                      h * 0.050, h * 0.065)
    s.add_part(SilhouettePart("bust_r", bust_r, depth=0.058,
                              color=bust_light, outline=bust_shade))
    # 胸廓中间高光条（胸廓/领口下方，增强体积感）
    chest_mid = [(head_c[0] - h * 0.02, 0.70 * h),
                 (head_c[0] + h * 0.02, 0.70 * h),
                 (head_c[0] + h * 0.015, 0.62 * h),
                 (head_c[0] - h * 0.015, 0.62 * h)]
    s.add_part(SilhouettePart("chest_mid", chest_mid, depth=0.057,
                              color=(215, 228, 248), outline=(215, 228, 248)))

    # ---- hip（体积化：裙摆鼓起 + 髋沟线，漫画风）----
    # 裙摆鼓起色块（深蓝紫阴影色，与裙身明显区分，体现髋部体积）
    hip_shade = (105, 130, 195)            # 深一档（比裙身 150,180,235 暗很多）
    hip_l = _ellipse((head_c[0] - h * 0.215, 0.47 * h), h * 0.05, h * 0.075)
    hip_r = _ellipse((head_c[0] + h * 0.215, 0.47 * h), h * 0.05, h * 0.075)
    # 命名 hip_shade_*（非 hip_*）：下方「腿」段已用 hip_l/hip_r 命名大腿部件，
    # 重名会让 `joint("hip_l", …)` 命中先添加的阴影（pivot=None ⇒ 旋转空操作），
    # 大腿永远摆不动（#310）。此处只改名字，不动 joint()/add_part() 的行为。
    s.add_part(SilhouettePart("hip_shade_l", hip_l, depth=0.058,
                              color=hip_shade, outline=(80, 105, 165)))
    s.add_part(SilhouettePart("hip_shade_r", hip_r, depth=0.058,
                              color=hip_shade, outline=(80, 105, 165)))
    # 裙摆中心弧线（两腿之间的裙摆曲线，收束到腿部）
    skirt_mid = [(head_c[0] - h * 0.04, 0.44 * h),
                 (head_c[0], 0.42 * h),
                 (head_c[0] + h * 0.04, 0.44 * h)]
    s.add_part(SilhouettePart("skirt_mid", skirt_mid, depth=0.059,
                              color=(110, 145, 205), outline=(110, 145, 205)))

    # ---- 手臂（分节：上臂 + 下臂，pivot 肩/肘）----
    # 左臂（z=0.10）
    shoulder_l = (-0.18 * h, 0.68 * h)
    elbow_l = (-0.22 * h, 0.55 * h)
    hand_l = (-0.26 * h, 0.42 * h)
    arm_l_up = [(-0.16 * h, 0.70 * h), (-0.21 * h, 0.70 * h),
                (elbow_l[0] - 0.02 * h, elbow_l[1]), (elbow_l[0] + 0.02 * h, elbow_l[1])]
    arm_l_lo = [(elbow_l[0] - 0.02 * h, elbow_l[1]), (elbow_l[0] + 0.02 * h, elbow_l[1]),
                (hand_l[0] + 0.01 * h, hand_l[1]), (hand_l[0] - 0.03 * h, hand_l[1])]
    s.add_part(SilhouettePart("shoulder_l", arm_l_up, depth=0.10, color=skin, outline=line,
                              pivot=shoulder_l))
    s.add_part(SilhouettePart("elbow_l", arm_l_lo, depth=0.10, color=skin, outline=line,
                              pivot=elbow_l, parent="shoulder_l"))
    # 右臂
    shoulder_r = (0.18 * h, 0.68 * h)
    elbow_r = (0.22 * h, 0.55 * h)
    hand_r = (0.26 * h, 0.42 * h)
    arm_r_up = [(0.21 * h, 0.70 * h), (0.16 * h, 0.70 * h),
                (elbow_r[0] + 0.02 * h, elbow_r[1]), (elbow_r[0] - 0.02 * h, elbow_r[1])]
    arm_r_lo = [(elbow_r[0] + 0.02 * h, elbow_r[1]), (elbow_r[0] - 0.02 * h, elbow_r[1]),
                (hand_r[0] - 0.01 * h, hand_r[1]), (hand_r[0] + 0.03 * h, hand_r[1])]
    s.add_part(SilhouettePart("shoulder_r", arm_r_up, depth=0.10, color=skin, outline=line,
                              pivot=shoulder_r))
    s.add_part(SilhouettePart("elbow_r", arm_r_lo, depth=0.10, color=skin, outline=line,
                              pivot=elbow_r, parent="shoulder_r"))

    # ---- 腿（分节：大腿 + 小腿，pivot 髋/膝）----
    # 左腿（z=0.14）
    hip_l = (-0.08 * h, 0.42 * h)
    knee_l = (-0.10 * h, 0.26 * h)
    foot_l = (-0.10 * h, 0.10 * h)
    leg_l_up = [(hip_l[0] - 0.045 * h, hip_l[1]), (hip_l[0] + 0.01 * h, hip_l[1]),
                (knee_l[0] + 0.02 * h, knee_l[1]), (knee_l[0] - 0.02 * h, knee_l[1])]
    leg_l_lo = [(knee_l[0] - 0.02 * h, knee_l[1]), (knee_l[0] + 0.02 * h, knee_l[1]),
                (foot_l[0] + 0.015 * h, foot_l[1]), (foot_l[0] - 0.035 * h, foot_l[1])]
    s.add_part(SilhouettePart("hip_l", leg_l_up, depth=0.14, color=skin, outline=line,
                              pivot=hip_l))
    s.add_part(SilhouettePart("knee_l", leg_l_lo, depth=0.14, color=skin, outline=line,
                              pivot=knee_l))
    # 右腿
    hip_r = (0.08 * h, 0.42 * h)
    knee_r = (0.10 * h, 0.26 * h)
    foot_r = (0.10 * h, 0.10 * h)
    leg_r_up = [(hip_r[0] + 0.045 * h, hip_r[1]), (hip_r[0] - 0.01 * h, hip_r[1]),
                (knee_r[0] - 0.02 * h, knee_r[1]), (knee_r[0] + 0.02 * h, knee_r[1])]
    leg_r_lo = [(knee_r[0] + 0.02 * h, knee_r[1]), (knee_r[0] - 0.02 * h, knee_r[1]),
                (foot_r[0] - 0.015 * h, foot_r[1]), (foot_r[0] + 0.035 * h, foot_r[1])]
    s.add_part(SilhouettePart("hip_r", leg_r_up, depth=0.14, color=skin, outline=line,
                              pivot=hip_r))
    s.add_part(SilhouettePart("knee_r", leg_r_lo, depth=0.14, color=skin, outline=line,
                              pivot=knee_r))

    # ---- 尾巴（多节：3 节，pivot 逐节，能摆动）----
    tail_base = (-0.24 * h, 0.44 * h)
    t1 = (-0.30 * h, 0.52 * h)
    t2 = (-0.36 * h, 0.62 * h)
    t3 = (-0.40 * h, 0.74 * h)
    tail_sec1 = [(tail_base[0] - 0.02 * h, tail_base[1]), (tail_base[0] + 0.03 * h, tail_base[1]),
                 (t1[0] + 0.015 * h, t1[1]), (t1[0] - 0.015 * h, t1[1])]
    tail_sec2 = [(t1[0] - 0.015 * h, t1[1]), (t1[0] + 0.015 * h, t1[1]),
                 (t2[0] + 0.012 * h, t2[1]), (t2[0] - 0.012 * h, t2[1])]
    tail_sec3 = [(t2[0] - 0.012 * h, t2[1]), (t2[0] + 0.012 * h, t2[1]),
                 (t3[0] + 0.008 * h, t3[1]), (t3[0] - 0.008 * h, t3[1])]
    s.add_part(SilhouettePart("tail1", tail_sec1, depth=0.22, color=fur, outline=line,
                              pivot=tail_base))
    s.add_part(SilhouettePart("tail2", tail_sec2, depth=0.23, color=fur, outline=line,
                              pivot=t1, parent="tail1"))
    s.add_part(SilhouettePart("tail3", tail_sec3, depth=0.24, color=fur, outline=line,
                              pivot=t2, parent="tail2"))
    return s


# ---------------------------------------------------------------------------
# 姿态（状态 → 四肢/尾巴角度）
# ---------------------------------------------------------------------------

# 关节命名：`apply_pose` 的「声明」与「实现」共用同一份数据（#380）。
#
# 缺陷态：旧 docstring 声明「兼容 shoulder_l/elbow_l/hip_l/knee_l 与
# upper_l/lower_l/thigh_l/calf_l」，但五个姿态分支只给臂设值 —— 腿的 8 个名字
# 只出现在归零段，声明与实现不一致（腿只能被归零）。修复取向＝**收窄声明**：
# 把「仅归零」与「会被姿态驱动」拆成两张显式表，docstring 只声明后者。
#
# 判据来源：本件报告 #380（经验标定，追不到理论章节——仓内无「姿态应驱动哪些
# 关节」的理论规定）。**削掉的判别力**：不再宣称 `apply_pose` 会摆腿；若将来要
# 摆腿，须同时改本表与 docstring（守卫
# `tests/test_issue8_175_310_376_380_silhouette3d_guard.py::test_380_*`
# 会因「声明≠实现」变红）。

# 归零表：每次 `apply_pose` 先把这些关节角复位（覆盖两套模板命名）。
# 含腿——若调用方用 `joint()` 摆过腿，`apply_pose` 负责复位；但姿态本身不改写它们。
_ZERO_JOINTS = (
    "shoulder_l", "upper_l", "shoulder_r", "upper_r",
    "elbow_l", "lower_l", "elbow_r", "lower_r",
    "hip_l", "thigh_l", "hip_r", "thigh_r",
    "knee_l", "calf_l", "knee_r", "calf_r",
)

# 驱动表：姿态分支实际会**设值**的关节名（两模板命名并列；`_set_tail_wave` 驱动 tail1..3）。
# 腿不在其中——`hip_*`/`thigh_*`/`knee_*`/`calf_*` 仅归零。
_POSE_DRIVEN_JOINTS = {
    "shoulder": ("shoulder_l", "shoulder_r", "upper_l", "upper_r"),
    "elbow": ("elbow_l", "elbow_r", "lower_l", "lower_r"),
    "tail": ("tail1", "tail2", "tail3"),
}

# 尾巴摆动：绕自身 pivot 的旋转（级联：每节 = 上一节角度 + 自身增量）
def _set_tail_wave(s: Silhouette3D, wave: float) -> None:
    """设置尾巴摆动幅度（wave 弧度，正=向右甩）。"""
    s.joint("tail1", wave)
    s.joint("tail2", wave * 1.4)
    s.joint("tail3", wave * 1.8)


# ---------------------------------------------------------------------------
# 表情系统（玩家注视第一优先：脸 —— 眼型/嘴型随情绪变化）
# ---------------------------------------------------------------------------

_EYE_DARK = (70, 55, 70)
_EYE_LIGHT = (255, 255, 255)
_MOUTH = (180, 90, 100)


def _eye_poly(cx, cy, rx, ry, shape: str) -> List[Tuple[float, float]]:
    """生成眼型多边形。
    normal=椭圆 / happy=笑眼(上弯弧) / sad=哭眼(下垂) / surprise=惊眼(圆大) / squint=眯眼(横线)
    """
    if shape == "happy":       # 笑眼：上弯弧（月牙）
        return [(cx - rx, cy + ry * 0.2), (cx - rx * 0.5, cy - ry * 0.6),
                (cx, cy - ry * 0.8), (cx + rx * 0.5, cy - ry * 0.6),
                (cx + rx, cy + ry * 0.2), (cx, cy + ry * 0.5)]
    if shape == "sad":         # 哭眼：下垂弧（八字）
        return [(cx - rx, cy - ry * 0.3), (cx - rx * 0.5, cy + ry * 0.4),
                (cx, cy + ry * 0.5), (cx + rx * 0.5, cy + ry * 0.4),
                (cx + rx, cy - ry * 0.3), (cx, cy - ry * 0.6)]
    if shape == "surprise":    # 惊眼：圆大
        return _ellipse((cx, cy), rx * 1.35, ry * 1.35, 10)
    if shape == "squint":      # 眯眼：横线
        return [(cx - rx * 1.1, cy - ry * 0.15), (cx + rx * 1.1, cy - ry * 0.15),
                (cx + rx * 1.1, cy + ry * 0.15), (cx - rx * 1.1, cy + ry * 0.15)]
    return _ellipse((cx, cy), rx, ry, 10)     # normal


def _mouth_poly(cx, cy, rx, ry, shape: str) -> List[Tuple[float, float]]:
    """嘴型多边形。smile=微笑 / open=张嘴 / pout=嘟嘴 / sad=委屈下垂"""
    if shape == "open":        # 张嘴：椭圆
        return _ellipse((cx, cy), rx, ry * 1.3, 10)
    if shape == "pout":        # 嘟嘴：小圆
        return _ellipse((cx, cy), rx * 0.7, ry * 1.2, 10)
    if shape == "sad":         # 委屈：下垂弧
        return [(cx - rx, cy), (cx - rx * 0.4, cy + ry * 0.6),
                (cx, cy + ry * 0.8), (cx + rx * 0.4, cy + ry * 0.6),
                (cx + rx, cy), (cx + rx * 0.6, cy - ry * 0.3),
                (cx - rx * 0.6, cy - ry * 0.3)]
    return [(cx - rx, cy), (cx + rx, cy),
            (cx + rx * 0.5, cy + ry), (cx - rx * 0.5, cy + ry)]   # smile


# 腮红：基色 + shy「加深」用的通道系数
# 加深只在 G/B 通道压暗（颜色更红）；整体乘系数会把接近饱和的粉色推成纯白
_BLUSH_BASE = (255, 180, 190)
_BLUSH_DEEP_SCALE = (1.0, 0.72, 0.80)


def set_expression(s: Silhouette3D, expression: str) -> Silhouette3D:
    """表情（状态情绪）→ 眼型 + 嘴型 + 腮红深浅。返回同一对象。

    表达式: happy/sad/angry/surprised/shy/neutral
    - happy    → 笑眼 + 张嘴
    - sad      → 哭眼 + 委屈嘴
    - angry    → 眯眼 + 嘟嘴
    - surprised→ 惊眼 + 张嘴(大)
    - shy      → 笑眼(小) + 抿嘴 + 腮红加深
    - neutral  → 正常眼 + 微笑

    **幂等**（#376）：眼/嘴/高光的几何基准在首次调用时快照进 `s._expr_base`，
    每次调用都**从该基准重算**（而非从上一次调用后的几何迭代）。故同一对象上
    连调 N 次几何不变，且任意表情序列可复原。判据来源：本件报告 #376
    （经验标定，追不到理论章节；旧行为是以上次结果为输入的几何级数）。
    """
    eye_shape = {"happy": "happy", "sad": "sad", "angry": "squint",
                 "surprised": "surprise", "shy": "happy", "neutral": "normal"}.get(expression, "normal")
    mouth_shape = {"happy": "open", "sad": "sad", "angry": "pout",
                   "surprised": "open", "shy": "smile", "neutral": "smile"}.get(expression, "smile")

    # ---- 几何基准：首次调用时快照，之后一律从这里重算（#376 幂等）----
    if s._expr_base is None:
        s._expr_base = {
            p.name: [tuple(pt) for pt in p.points]
            for p in s.parts
            if p.name in ("eye_l", "eye_r", "mouth") or p.name.startswith("highlight_")
        }
    base = s._expr_base

    def _center(pts):
        return (sum(pt[0] for pt in pts) / len(pts), sum(pt[1] for pt in pts) / len(pts))

    def _half(pts):
        return ((max(pt[0] for pt in pts) - min(pt[0] for pt in pts)) / 2,
                (max(pt[1] for pt in pts) - min(pt[1] for pt in pts)) / 2)

    # 眼（从基准眼型取眼心/半轴，不再量上一次调用后的几何）
    if base.get("eye_l"):
        cy = _center(base["eye_l"])[1]
        rx, ry = _half(base["eye_l"])
        for side in ("l", "r"):
            bpts = base.get(f"eye_{side}")
            if not bpts:
                continue
            ecx = _center(bpts)[0]          # 该眼自身中心（不假设左右镜像）
            for p in s.parts:
                if p.name == f"eye_{side}":
                    p.points = _eye_poly(ecx, cy, rx, ry, eye_shape)
            # 高光跟随眼型（像素画高光是灵魂，始终保留，位置随眼动）：
            # 从**基准**高光点重算（眼心不变 ⇒ 只叠加上抬 dy）。
            dy = -ry * 0.3 if eye_shape in ("happy", "squint") else 0.0
            for p in s.parts:
                if p.name.startswith(f"highlight_{side}"):
                    bp = base.get(p.name)
                    if bp and len(bp) >= 4:
                        p.points = [(pt[0], pt[1] + dy) for pt in bp]

    # 嘴（同样从基准重算）
    if base.get("mouth"):
        mx, my = _center(base["mouth"])
        mrx, mry = _half(base["mouth"])
        for p in s.parts:
            if p.name == "mouth":
                p.points = _mouth_poly(mx, my, max(0.002, mrx), max(0.002, mry), mouth_shape)

    # 腮红深浅（shy 加深：压暗 G/B 通道走向深粉，而不是整体乘系数把颜色推成饱和纯白）
    for p in s.parts:
        if p.name.startswith("blush_"):
            if expression == "shy":
                p.color = tuple(int(c * f) for c, f in zip(_BLUSH_BASE, _BLUSH_DEEP_SCALE))
            else:
                p.color = _BLUSH_BASE
    return s


def apply_pose(s: Silhouette3D, pose: str) -> Silhouette3D:
    """姿态（状态）→ 手臂/尾巴关节角度。返回同一对象（可链式）。

    兼容两种部件命名（下表即 `_POSE_DRIVEN_JOINTS`——**声明与实现同一份数据**）：
    - 手调版 fatfish_skeleton：shoulder_l/elbow_l（+ tail1..3）
    - 蒙皮版 fatfish_skinned：upper_l/lower_l（+ tail1..3）

    **腿不在此列**（#380）：`hip_*`/`thigh_*`/`knee_*`/`calf_*` 属 `_ZERO_JOINTS`
    （每次调用先复位），但五个姿态分支从不给它们设值——旧 docstring 曾把它们
    列为「兼容命名」，与实现不符。需要腿部姿态时由调用方直接
    `Silhouette3D.joint(...)`；若要改由 `apply_pose` 驱动，须同步改
    `_POSE_DRIVEN_JOINTS` 与本 docstring（守卫会因两者不一致变红）。
    """
    # 部件名归一化（旧名 → 可作用于的关节名）
    def j(sil: Silhouette3D, *names: str) -> None:
        for n in names:
            if any(p.name == n for p in sil.parts):
                sil.joint(n, 0.0)
                return
    def setj(sil: Silhouette3D, val: float, *names: str) -> None:
        for n in names:
            if any(p.name == n for p in sil.parts):
                sil.joint(n, val)
                return
    # 默认（自然站立，尾巴微垂）：按归零表复位（含腿——只复位，不设姿态值）
    for n in _ZERO_JOINTS:
        j(s, n)
    _set_tail_wave(s, 0.15)

    if pose == "happy":
        # 开心：右臂举起挥手 + 尾巴快摆
        setj(s, 0.9, "shoulder_r", "upper_r")
        setj(s, -1.2, "elbow_r", "lower_r")
        setj(s, -0.2, "shoulder_l", "upper_l")
        _set_tail_wave(s, 0.5)
    elif pose == "shy":
        # 害羞：手臂收拢身前 + 尾巴下垂
        setj(s, 0.5, "shoulder_l", "upper_l")
        setj(s, -1.0, "elbow_l", "lower_l")
        setj(s, -0.5, "shoulder_r", "upper_r")
        setj(s, -1.0, "elbow_r", "lower_r")
        _set_tail_wave(s, -0.1)
    elif pose == "sad":
        # 难过：垂头垂尾
        setj(s, 0.2, "shoulder_l", "upper_l")
        setj(s, -0.2, "shoulder_r", "upper_r")
        _set_tail_wave(s, -0.4)
    elif pose == "angry":
        # 生气：双手叉腰状（手臂外张）+ 尾巴炸毛（大幅度）
        setj(s, -0.6, "shoulder_l", "upper_l")
        setj(s, 0.6, "shoulder_r", "upper_r")
        _set_tail_wave(s, 0.7)
    elif pose == "tired":
        # 疲惫：垂肩垂尾
        setj(s, 0.3, "shoulder_l", "upper_l")
        setj(s, -0.3, "shoulder_r", "upper_r")
        _set_tail_wave(s, -0.25)
    return s


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)

    center = (0.0, 0.85, 5.0)
    from PIL import Image

    # 3 视角：正面 / 微侧 / 侧视
    cams = [
        ("front", Camera3D.look_at(eye=(0.0, 1.0, 1.6), target=center, fov_deg=45)),
        ("side3", Camera3D.look_at(eye=(1.2, 1.0, 1.6), target=center, fov_deg=45)),
        ("side",  Camera3D.look_at(eye=(2.4, 1.0, 1.6), target=center, fov_deg=45)),
    ]
    for name, cam in cams:
        ff = fatfish_skeleton(center=center, height=1.55)
        img = ff.render(300, 400, camera=cam)
        p = os.path.join(out_dir, f"fatfish_{name}.png")
        img.save(p)
        print(f"fatfish {name} → {p}")
