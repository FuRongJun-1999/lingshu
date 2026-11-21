# -*- coding: utf-8 -*-
"""wm_bridge.factory.vec.geom2d · 3D→2D 投影 + 隐藏线消除（纯标准库 · 零 bpy · 零 numpy）
================================================================================
为什么需要它：
  `wm_bridge.factory.geo.spec` 是四条产线共用的**唯一输入契约**；建模/绘画产线把 spec
  变成 Blender 场景与位图。位图的问题是"分配分辨率的瞬间就定死了"：放到海报尺寸就糊，
  线稿也不可编辑。要出**矢量稿**，第一步必须是"把真实 3D 几何投到 2D"，第二步是
  "把被挡住的棱线去掉"——否则投影出来的线框会前后穿插，看起来像一坨铁丝。

本模块只做这两件事，且**只用标准库**（可在没有 Blender、没有 numpy 的机器上跑）：
  · `Camera`      —— 透视相机（与 Blender 的 eye/look_at/焦距口径对齐），投影 + 近平面裁剪
  · `DepthBuffer` —— 自己的**扫描线 z-buffer**（1/z 线性插值），做隐藏线判据
  · `split_hidden` —— 折线按深度缓冲切成"可见段 / 被遮挡段"，供后端分别画实线/虚线
  · `triangulate`  —— 多边形面 → 三角形（扇形），喂给 z-buffer 与画家序

判据与边界（如实）：
  · **不是**逐像素精确的 HLR：判据是"棱线采样点相对 z-buffer 的深度差"，采样步长默认
    1 像素。好处是纯标准库、确定、可复现；代价是**极细的遮挡（<1px）可能漏判**，
    公开写入 `stats()["hidden_ambiguous"]` 让调用方看得见。
  · 只做**均匀网格**采样，不做边缘抗锯齿——矢量稿本来就不需要抗锯齿。
  · 投影用**水平 FOV**（Blender 的 sensor_fit=AUTO 且宽>高时同口径）；竖直方向按
    像素正方形换算，故不引入非等比缩放。
  · 所有浮点输出最终由 `svg.py` 按固定小数位格式化；本模块**不做**格式化（保持数值精度）。
"""
from __future__ import annotations

import math

__all__ = [
    "Camera", "DepthBuffer", "ProjectedFace", "split_hidden", "triangulate",
    "face_normal", "bounds3", "HAVE_NUMPY",
]

try:                                    # 可选加速：有 numpy 就用，没有也行（判据不变）
    import numpy as _np
    HAVE_NUMPY = True
except Exception:                       # pragma: no cover - 环境相关
    _np = None
    HAVE_NUMPY = False

_EPS = 1e-12


# ─────────────────────────────────────────────────────────── 基础向量（纯 tuple）
def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _mul(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _len(a):
    return math.sqrt(_dot(a, a))


def _norm(a):
    n = _len(a)
    if n < _EPS:
        raise ValueError("零长度向量不能归一化：%r" % (a,))
    return (a[0] / n, a[1] / n, a[2] / n)


def bounds3(points):
    """3D 包围盒 → ((minx,miny,minz),(maxx,maxy,maxz))；空集 ⇒ None。"""
    pts = [p for p in points]
    if not pts:
        return None
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    for p in pts:
        for i in range(3):
            v = float(p[i])
            if v < lo[i]:
                lo[i] = v
            if v > hi[i]:
                hi[i] = v
    return (tuple(lo), tuple(hi))


def face_normal(pts):
    """多边形法线（Newell 法，对非平面四边形也稳）；退化 ⇒ (0,0,0)。"""
    nx = ny = nz = 0.0
    n = len(pts)
    for i in range(n):
        a = pts[i]
        b = pts[(i + 1) % n]
        nx += (a[1] - b[1]) * (a[2] + b[2])
        ny += (a[2] - b[2]) * (a[0] + b[0])
        nz += (a[0] - b[0]) * (a[1] + b[1])
    v = (nx, ny, nz)
    ln = _len(v)
    if ln < _EPS:
        return (0.0, 0.0, 0.0)
    return (v[0] / ln, v[1] / ln, v[2] / ln)


def triangulate(pts):
    """凸/近凸多边形 → 扇形三角化 `[(p0,pi,pi+1), ...]`；顶点 < 3 ⇒ 空列表。"""
    if len(pts) < 3:
        return []
    p0 = pts[0]
    return [(p0, pts[i], pts[i + 1]) for i in range(1, len(pts) - 1)]


# ─────────────────────────────────────────────────────────── 相机
class Camera:
    """透视相机：`eye` 看向 `look_at`，`up` 为上向；水平 FOV（度）定视锥。

    坐标口径与 Blender 一致：世界 **Z-up**。屏幕坐标 y **向下**（SVG 同向），
    原点在左上角，单位为像素。
    """

    def __init__(self, eye, look_at, up=(0.0, 0.0, 1.0), fov_deg=39.6,
                 width=1920, height=1080, near=0.05, far=1000.0):
        self.eye = tuple(float(x) for x in eye)
        self.look_at = tuple(float(x) for x in look_at)
        self.width = int(width)
        self.height = int(height)
        self.near = float(near)
        self.far = float(far)
        fwd = _sub(self.look_at, self.eye)
        if _len(fwd) < _EPS:
            raise ValueError("eye 与 look_at 重合，视线方向未定义")
        self.fwd = _norm(fwd)
        upv = tuple(float(x) for x in up)
        right = _cross(self.fwd, upv)
        if _len(right) < 1e-9:
            # up 与视线共线：换一个上向（与 Blender 的 track-to 行为一致：不报错，换轴）
            upv = (0.0, 1.0, 0.0) if abs(self.fwd[1]) < 0.9 else (1.0, 0.0, 0.0)
            right = _cross(self.fwd, upv)
        self.right = _norm(right)
        self.up = _cross(self.right, self.fwd)
        self.fov_deg = float(fov_deg)
        if not (0.0 < self.fov_deg < 179.0):
            raise ValueError("fov_deg 须在 (0,179)，实为 %r" % (fov_deg,))
        self.focal_px = (self.width / 2.0) / math.tan(math.radians(self.fov_deg) / 2.0)

    # -------------------------------------------------- 构造
    @classmethod
    def from_eye_look(cls, eye, look_at, **kw):
        return cls(eye, look_at, **kw)

    @classmethod
    def orbit(cls, target, radius, azimuth_deg, elevation_deg, **kw):
        """绕 `target` 的球面机位（方位角绕 +Z，仰角自水平面）。"""
        az = math.radians(float(azimuth_deg))
        el = math.radians(float(elevation_deg))
        r = float(radius)
        eye = (target[0] + r * math.cos(el) * math.cos(az),
               target[1] + r * math.cos(el) * math.sin(az),
               target[2] + r * math.sin(el))
        return cls(eye, tuple(float(x) for x in target), **kw)

    # -------------------------------------------------- 投影
    def to_camera(self, p):
        """世界点 → 相机坐标 `(xc, yc, zc)`；zc > 0 表示在相机前方。"""
        d = _sub(p, self.eye)
        return (_dot(d, self.right), _dot(d, self.up), _dot(d, self.fwd))

    def cam_to_screen(self, c):
        """相机坐标 → 屏幕像素 `(x, y)`；zc 必须 > 0（否则调用方先裁剪）。"""
        z = c[2]
        if z <= _EPS:
            raise ValueError("点在相机后方（zc=%r），调用方须先做近平面裁剪" % (z,))
        inv = self.focal_px / z
        return (self.width / 2.0 + c[0] * inv, self.height / 2.0 - c[1] * inv)

    def project(self, p):
        """世界点 → `(x, y, zc)`（不裁剪；`zc <= 0` 时 x/y 无意义）。"""
        c = self.to_camera(p)
        return (self.width / 2.0 + c[0] * self.focal_px / max(c[2], _EPS),
                self.height / 2.0 - c[1] * self.focal_px / max(c[2], _EPS),
                c[2])

    def in_view(self, p, margin=0.0):
        """点是否在视锥内（含 near/far 与画面边界，margin 为像素外扩）。"""
        x, y, z = self.project(p)
        if z < self.near or z > self.far:
            return False
        return (-margin <= x <= self.width + margin
                and -margin <= y <= self.height + margin)

    # -------------------------------------------------- 近平面裁剪
    def clip_polygon(self, pts):
        """多边形对**近平面**做 Sutherland-Hodgman 裁剪；返回裁剪后的点列。

        为什么必须做：`zc<=0` 的点投影会翻到画面另一侧，产生"穿过相机"的假面片
        （既有位图路径为此专门修过近平面裁剪；矢量路径同样不能漏）。
        """
        out = []
        n = len(pts)
        if n == 0:
            return out
        cam = [self.to_camera(p) for p in pts]
        for i in range(n):
            a = cam[i]
            b = cam[(i + 1) % n]
            az = a[2] - self.near
            bz = b[2] - self.near
            if az >= 0:
                out.append(a)
            if (az >= 0) != (bz >= 0):
                t = az / (az - bz) if abs(az - bz) > _EPS else 0.0
                out.append((a[0] + (b[0] - a[0]) * t,
                            a[1] + (b[1] - a[1]) * t,
                            a[2] + (b[2] - a[2]) * t))
        if not out:
            return []
        return [self._cam_to_world(c) for c in out]

    def _cam_to_world(self, c):
        return _add(self.eye, _add(_mul(self.right, c[0]),
                                   _add(_mul(self.up, c[1]), _mul(self.fwd, c[2]))))

    def pixel_extent(self, p_lo, p_hi):
        """3D 包围盒投影到屏幕的近似外接框（8 角点投影取包络）。"""
        xs, ys = [], []
        for i in (0, 1):
            for j in (0, 1):
                for k in (0, 1):
                    p = (p_lo[0] if i else p_hi[0],
                         p_lo[1] if j else p_hi[1],
                         p_lo[2] if k else p_hi[2])
                    x, y, z = self.project(p)
                    xs.append(x)
                    ys.append(y)
        return (min(xs), min(ys), max(xs), max(ys))

    def __repr__(self):                 # pragma: no cover - 调试用
        return ("Camera(eye=%r, look_at=%r, fov=%.2f, %dx%d)"
                % (self.eye, self.look_at, self.fov_deg, self.width, self.height))


class ProjectedFace:
    """一个已投影的面片：屏幕多边形 + 平均深度 + 来源标签/颜色。

    `depth` 是**屏幕空间重心处**的 `1/z` 的倒数（即 z），用于画家算法排序；
    `backface` 为真表示法线背离相机（默认由调用方剔除）。
    """

    __slots__ = ("pts", "depth", "label", "color", "backface", "face_id")

    def __init__(self, pts, depth, label="", color=None, backface=False, face_id=-1):
        self.pts = pts                  # [(x, y), ...] 屏幕像素
        self.depth = depth              # 平均 zc（越大越远）
        self.label = label
        self.color = color
        self.backface = backface
        self.face_id = face_id

    def __repr__(self):                 # pragma: no cover
        return "ProjectedFace(%s, n=%d, z=%.4f)" % (self.label, len(self.pts), self.depth)


# ─────────────────────────────────────────────────────────── 深度缓冲（自己的扫描线）
class DepthBuffer:
    """屏幕空间 z-buffer：存 `1/z`（越大越近），0 表示空。

    实现是**自己的扫描线**（边缘函数 + 逐行 y 区间），不是调库：
      · 逐三角形取屏幕外接框，按行扫；每行与三角形两条边求交得 x 区间；
      · 区间内按重心线性插值 `1/z`（透视下 `1/z` 在屏幕空间线性 ⇒ 无需除透视校正）；
      · 写入时取 max（最近者胜）。
    numpy 存在时按行向量化（判据不变，只是快）；不存在时退化为纯 python 逐像素。
    """

    def __init__(self, width, height):
        self.width = int(width)
        self.height = int(height)
        if HAVE_NUMPY:
            self._buf = _np.zeros((self.height, self.width), dtype=_np.float32)
        else:
            self._buf = [[0.0] * self.width for _ in range(self.height)]
        self.tris_drawn = 0
        self.tris_skipped = 0
        self.pixels_written = 0

    # -------------------------------------------------- 写入
    def add_triangles(self, tris):
        """`tris` = [(p0, p1, p2)]，每个 p 是**屏幕坐标 + zc**：`(x, y, z)`。

        返回实际写入的三角形数（屏幕外接框为空的会被跳过，不静默：计入返回值的差值）。
        """
        drawn = 0
        skipped = 0
        for t in tris:
            if self._add_one(t):
                drawn += 1
            else:
                skipped += 1
        self.tris_drawn += drawn
        self.tris_skipped += skipped
        return drawn

    def _add_one(self, t):
        (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = t
        if z0 <= _EPS or z1 <= _EPS or z2 <= _EPS:
            return False
        minx = max(0, int(math.floor(min(x0, x1, x2))))
        maxx = min(self.width - 1, int(math.ceil(max(x0, x1, x2))))
        miny = max(0, int(math.floor(min(y0, y1, y2))))
        maxy = min(self.height - 1, int(math.ceil(max(y0, y1, y2))))
        if maxx < minx or maxy < miny:
            return False
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(area) < 1e-12:
            return False
        w0 = 1.0 / z0
        w1 = 1.0 / z1
        w2 = 1.0 / z2
        inv_area = 1.0 / area
        rows = range(miny, maxy + 1)
        if HAVE_NUMPY:
            ys = _np.arange(miny, maxy + 1, dtype=_np.float64) + 0.5
            xa, xb, wa, wb = self._spans(x0, y0, w0, x1, y1, w1, x2, y2, w2, ys,
                                         minx, maxx)
            for k, y in enumerate(rows):
                if xa[k] > xb[k]:
                    continue
                lo = max(minx, int(math.ceil(xa[k] - 0.5)))
                hi = min(maxx, int(math.floor(xb[k] - 0.5)))
                if hi < lo:
                    continue
                xs = _np.arange(lo, hi + 1, dtype=_np.float64) + 0.5
                denom = xb[k] - xa[k]
                if abs(denom) < 1e-12:
                    tt = _np.zeros_like(xs)
                else:
                    tt = (xs - xa[k]) / denom
                wv = wa[k] + (wb[k] - wa[k]) * tt
                row = self._buf[y, lo:hi + 1]
                _np.maximum(row, wv.astype(_np.float32), out=row)
                self.pixels_written += (hi - lo + 1)
        else:                            # pragma: no cover - 无 numpy 环境
            for y in rows:
                yc = y + 0.5
                xa, xb, wa, wb = self._span_py(x0, y0, w0, x1, y1, w1,
                                               x2, y2, w2, yc)
                if xa is None or xa > xb:
                    continue
                lo = max(minx, int(math.ceil(xa - 0.5)))
                hi = min(maxx, int(math.floor(xb - 0.5)))
                if hi < lo:
                    continue
                denom = xb - xa
                for x in range(lo, hi + 1):
                    xc = x + 0.5
                    tt = 0.0 if abs(denom) < 1e-12 else (xc - xa) / denom
                    wv = wa + (wb - wa) * tt
                    if wv > self._buf[y][x]:
                        self._buf[y][x] = wv
                    self.pixels_written += 1
        return True

    @staticmethod
    def _spans(x0, y0, w0, x1, y1, w1, x2, y2, w2, ys, minx, maxx):
        """三边与各行求交 → 每行的 (xa, xb, wa, wb)。

        做法：对三角形的**每条边**，在该边 y 覆盖的行上解出 x 与 w；三条边各写一次，
        取"最小 x"和"最大 x"（凸三角形行截面是一条线段，端点在边上）。
        """
        n = len(ys)
        xa = _np.full(n, _np.inf)
        xb = _np.full(n, -_np.inf)
        wa = _np.zeros(n)
        wb = _np.zeros(n)
        for (ax, ay, aw, bx, by, bw) in ((x0, y0, w0, x1, y1, w1),
                                         (x1, y1, w1, x2, y2, w2),
                                         (x2, y2, w2, x0, y0, w0)):
            if abs(by - ay) < 1e-12:
                continue
            t = (ys - ay) / (by - ay)
            m = (t >= 0.0) & (t <= 1.0)
            if not m.any():
                continue
            xs_e = ax + (bx - ax) * t
            ws_e = aw + (bw - aw) * t
            sel = _np.nonzero(m)[0]
            for i in sel:
                x = xs_e[i]
                w = ws_e[i]
                if x < xa[i]:
                    xa[i] = x
                    wa[i] = w
                if x > xb[i]:
                    xb[i] = x
                    wb[i] = w
        return xa, xb, wa, wb

    @staticmethod
    def _span_py(x0, y0, w0, x1, y1, w1, x2, y2, w2, yc):
        xa = float("inf")
        xb = float("-inf")
        wa = wb = 0.0
        for (ax, ay, aw, bx, by, bw) in ((x0, y0, w0, x1, y1, w1),
                                         (x1, y1, w1, x2, y2, w2),
                                         (x2, y2, w2, x0, y0, w0)):
            if abs(by - ay) < 1e-12:
                continue
            t = (yc - ay) / (by - ay)
            if t < 0.0 or t > 1.0:
                continue
            x = ax + (bx - ax) * t
            w = aw + (bw - aw) * t
            if x < xa:
                xa, wa = x, w
            if x > xb:
                xb, wb = x, w
        if xa == float("inf"):
            return None, None, 0.0, 0.0
        return xa, xb, wa, wb

    # -------------------------------------------------- 查询
    def invz_at(self, x, y):
        """屏幕点（像素中心）的 `1/z`；空像素 ⇒ 0.0。"""
        ix = int(math.floor(x))
        iy = int(math.floor(y))
        if ix < 0 or iy < 0 or ix >= self.width or iy >= self.height:
            return 0.0
        if HAVE_NUMPY:
            return float(self._buf[iy, ix])
        return float(self._buf[iy][ix])       # pragma: no cover

    def stats(self):
        return {"width": self.width, "height": self.height,
                "tris_drawn": self.tris_drawn,
                "tris_skipped": self.tris_skipped,
                "pixels_written": self.pixels_written,
                "backend": "numpy" if HAVE_NUMPY else "pure-python"}


# ─────────────────────────────────────────────────────────── 隐藏线消除
def split_hidden(polyline3d, cam, zbuf, samples_per_px=1.0, bias=1.5e-3,
                 min_run=1):
    """把 3D 折线按深度缓冲切成**可见段 / 被遮挡段**。

    参数
    ----
    polyline3d : 世界坐标点列（≥2）
    cam        : `Camera`
    zbuf       : `DepthBuffer`（须已画完所有遮挡体）
    samples_per_px : 采样密度（1.0 = 每像素一个采样点；>1 更细但更慢）
    bias       : 相对深度容差（`1/z` 越小越远；`(|1/z_pt| * (1+bias)) >= buf` 视为不遮挡）。
                 为什么要相对容差：同一面片的棱线正好落在自己面上，绝对容差会在
                 远近都误判（近处过松、远处过紧）。
    min_run    : 短于该采样数的段并入邻段（避免碎成一堆 1 点虚线段）

    返回
    ----
    `(runs, stats)`，runs = `[(points, visible_bool), ...]`（相邻同态已合并）；
    stats 含 `samples` / `hidden_samples` / `ambiguous`（缓冲为空但落在画面内的采样数）。
    """
    pts = [tuple(float(c) for c in p) for p in polyline3d]
    if len(pts) < 2:
        return [], {"samples": 0, "hidden": 0, "ambiguous": 0}
    if cam.to_camera(pts[0])[2] <= cam.near and cam.to_camera(pts[-1])[2] <= cam.near:
        return [], {"samples": 0, "hidden": 0, "ambiguous": 0, "clipped": True}

    flags = []          # (world_point, visible, ambiguous)
    n_seg = len(pts) - 1
    for i in range(n_seg):
        a, b = pts[i], pts[i + 1]
        ca, cb = cam.to_camera(a), cam.to_camera(b)
        sa = cam.cam_to_screen(ca) if ca[2] > cam.near else None
        sb = cam.cam_to_screen(cb) if cb[2] > cam.near else None
        if sa is None or sb is None:
            # 有一段在近平面后：整段按"不可判"处理（不假装可见）——交由上层裁剪
            steps = 0
        else:
            d = math.hypot(sb[0] - sa[0], sb[1] - sa[1])
            steps = max(1, int(math.ceil(d * float(samples_per_px))))
        for k in range(steps + 1):
            if i > 0 and k == 0:
                continue                # 段首点已在上一段末处理过
            t = k / float(steps) if steps else 0.0
            p = (a[0] + (b[0] - a[0]) * t,
                 a[1] + (b[1] - a[1]) * t,
                 a[2] + (b[2] - a[2]) * t)
            cc = cam.to_camera(p)
            if cc[2] <= cam.near:
                flags.append((p, False, True))       # 近平面后：不可见且不画
                continue
            sx, sy = cam.cam_to_screen(cc)
            if not (0 <= sx < zbuf.width and 0 <= sy < zbuf.height):
                flags.append((p, True, True))        # 画面外：无遮挡证据，标不可判
                continue
            w = 1.0 / cc[2]
            buf = zbuf.invz_at(sx, sy)
            if buf <= 0.0:
                flags.append((p, True, True))        # 该像素没画过任何面：无遮挡证据
                continue
            vis = (w * (1.0 + bias)) >= buf
            flags.append((p, vis, False))

    # 合并成 runs
    runs = []
    for p, vis, amb in flags:
        if runs and runs[-1][1] == vis:
            runs[-1][0].append(p)
        else:
            runs.append(([p], vis))
    # 去掉过短的中断（把 noise 段并回邻段）
    if min_run > 1 and len(runs) > 2:
        out = [runs[0]]
        for r in runs[1:]:
            if len(r[0]) <= min_run and out and len(out) > 1:
                out[-1][0].extend(r[0])
            else:
                out.append(r)
        runs = out
    stats = {"samples": len(flags),
             "hidden": sum(1 for f in flags if not f[1]),
             "ambiguous": sum(1 for f in flags if f[2])}
    return runs, stats
