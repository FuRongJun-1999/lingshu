# -*- coding: utf-8 -*-
"""wm_bridge.factory.vec.raster · 自写 SVG 光栅器（**只覆盖本产线自用原语子集**）
================================================================================
为什么必须自己写：本机**没有任何 SVG 渲染器**（实测 `cairosvg` / `rsvg-convert` /
`inkscape` / `chrome` 全都不存在）。而"矢量与位图一致"这条验收（前景 IoU ≥ 0.90）
必须以同分辨率、同 alpha 口径把 SVG 变成像素才能算。所以这个文件的定位是
**判据工具**：它不必支持 SVG 全规范，只需精确支持 `svg.py` 实际发射的那个子集，
并且**如实报告自己不支持什么**（`report["unsupported"]`）——不支持 ≠ 假装画了。

支持面（与 `svg.py` 发射面一一对应）
  · 元素：`<g>` / `<polygon>` / `<polyline>` / `<path>`（M L H V Z）/ `<rect>` /
    `<circle>` / `<ellipse>` / `<line>`；`<title>/<desc>/<defs>` 忽略；`<text>` **不支持**
    （计入 unsupported，不画）；SMIL 动画元素在静态帧下忽略（造型取基础属性）。
  · 属性继承：`fill` / `stroke` / `stroke-width` / `opacity` / `fill-opacity` /
    `stroke-opacity` / `stroke-dasharray`；`transform` 支持 `translate/scale/rotate/matrix`。
  · `fill="url(#id)"`（渐变）：**取中间色标近似**并计数（不是真渐变渲染，如实标出）。
  · 填充规则：**even-odd**（本产线发射的都是简单多边形，两规则结果一致；自交图形会不同，故如实写明）。
  · 抗锯齿：**SSAA**（默认 2×2 超采样后盒式降采样），不是解析式覆盖。

实现是**自己的扫描线 + 胶囊描边**：逐元素在其包围盒内解出覆盖度，再按文档序
（画家算法）alpha 合成；没有调用任何第三方光栅器。
"""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET

try:
    import numpy as np
except ImportError:  # pragma: no cover - 纯标准库环境（parse_svg 几何解析仍可用）
    np = None

__all__ = ["parse_svg", "rasterize", "foreground_mask", "iou", "load_svg_text",
           "SUPPORTED_TAGS"]

SUPPORTED_TAGS = ("svg", "g", "polygon", "polyline", "path", "rect", "circle",
                  "ellipse", "line", "title", "desc", "defs", "linearGradient",
                  "radialGradient", "stop", "animate", "animateTransform",
                  "animateMotion", "set", "metadata")
#: 完全不画、也不影响几何的元素（结构/动画/元数据）
_INERT = ("title", "desc", "defs", "linearGradient", "radialGradient", "stop",
          "animate", "animateTransform", "animateMotion", "set", "metadata")
_NUM = r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?"
_PATH_TOKEN_RE = re.compile(
    r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?|[A-Za-z]|[^\s,]"
)


def _require_numpy(fn_name):
    if np is None:
        raise ModuleNotFoundError(
            "numpy is required for wm_bridge.factory.vec.raster.%s" % fn_name)


# ─────────────────────────────────────────────────────────── 解析
def load_svg_text(src):
    """路径或 SVG 文本 → 文本（两条路都收，方便脚本与交互）。"""
    if isinstance(src, str) and src.lstrip().startswith("<"):
        return src
    with open(src, "r", encoding="utf-8") as fh:
        return fh.read()


def _floats(s):
    import re
    return [float(x) for x in re.findall(_NUM, s or "")]


def _color(v, default=None):
    """CSS 颜色（本产线只发射 #rrggbb / none）→ (r,g,b,a) 浮点 0..1。"""
    if v is None:
        return default
    v = v.strip().lower()
    if v in ("none", "transparent"):
        return None
    if v.startswith("#"):
        h = v[1:]
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        if len(h) >= 6:
            try:
                return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0,
                        int(h[4:6], 16) / 255.0, 1.0)
            except ValueError:
                return default
    if v.startswith("rgb("):
        vals = _floats(v)
        if len(vals) >= 3:
            return (vals[0] / 255.0, vals[1] / 255.0, vals[2] / 255.0,
                    vals[3] if len(vals) > 3 else 1.0)
    return default


def _gradient_mid(doc_defs, ref):
    """`url(#id)` → 渐变**中间色标**的颜色（近似，非真渐变）。"""
    if not ref.startswith("url(#") or not ref.endswith(")"):
        return None, None
    gid = ref[5:-1]
    g = doc_defs.get(gid)
    if g is None:
        return None, None
    stops = g["stops"]
    if not stops:
        return None, None
    return stops[len(stops) // 2], gid


def _mat_identity():
    if np is not None:
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def _mat_mul(a, b):
    if np is not None:
        return a @ b
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
            for i in range(3)]


def _parse_transform(s):
    """`transform` 文本 → 3×3 仿射矩阵（SVG 语义：后写的先作用？不——按 SVG 顺序左乘）。"""
    m = _mat_identity()
    if not s:
        return m
    for name, args in re.findall(r"(\w+)\s*\(([^)]*)\)", s):
        v = _floats(args)
        t = _mat_identity()
        if name == "translate":
            t[0][2] = v[0] if v else 0.0
            t[1][2] = v[1] if len(v) > 1 else 0.0
        elif name == "scale":
            sx = v[0] if v else 1.0
            sy = v[1] if len(v) > 1 else sx
            t[0][0], t[1][1] = sx, sy
        elif name == "rotate":
            a = math.radians(v[0] if v else 0.0)
            ca, sa = math.cos(a), math.sin(a)
            t[0][0], t[0][1], t[1][0], t[1][1] = ca, -sa, sa, ca
            if len(v) >= 3:
                cx, cy = v[1], v[2]
                t[0][2] = cx - ca * cx + sa * cy
                t[1][2] = cy - sa * cx - ca * cy
        elif name == "matrix" and len(v) >= 6:
            t[0][0], t[0][1], t[0][2] = v[0], v[2], v[4]
            t[1][0], t[1][1], t[1][2] = v[1], v[3], v[5]
        m = _mat_mul(m, t)
    return m


def parse_svg(svg_text):
    """SVG 文本 → `(shapes, meta, report)`。

    shapes 是**绘制序**列表，每项：`{kind, subpaths(用户坐标), closed, subpath_closed,
    fill, stroke, stroke_width, dash, rule}`；meta 含画布尺寸与 viewBox 映射。
    """
    root = ET.fromstring(svg_text)
    report = {"unsupported": {}, "gradients_approximated": 0, "elements_seen": 0}
    # #2098：viewBox 的 min-x/min-y 不能丢——rasterize 的补偿项（-vx*s/-vy*s）
    # 就是按「meta 带全四元组」设计的，meta 恒置 0 让带偏移 viewBox 的 SVG 整幅
    # 错位一个偏移量（实测 min-x/min-y=(50,50) 时红色矩形错位 50px）。
    vx, vy, vw, vh = 0.0, 0.0, 300.0, 150.0
    vb = root.get("viewBox")
    if vb:
        f = _floats(vb)
        if len(f) == 4:
            vx, vy, vw, vh = f
    defs = {}

    def scan_defs(e):
        for c in e:
            tag = c.tag.split("}")[-1]
            if tag in ("linearGradient", "radialGradient"):
                gid = c.get("id")
                stops = []
                for s in c:
                    if s.tag.split("}")[-1] == "stop":
                        col = s.get("stop-color")
                        rgba = _color(col)
                        if rgba:
                            stops.append(rgba)
                if gid:
                    defs[gid] = {"stops": stops, "type": tag}
            scan_defs(c)

    scan_defs(root)

    shapes = []

    def emit(kind, subpaths, closed, style, subpath_closed=None):
        flags = (list(subpath_closed) if subpath_closed is not None
                 else [bool(closed)] * len(subpaths))
        shapes.append({"kind": kind, "subpaths": subpaths, "closed": bool(closed),
                       "subpath_closed": flags,
                       "fill": style.get("fill"), "stroke": style.get("stroke"),
                       "stroke_width": style.get("stroke-width", 1.0),
                       "dash": style.get("stroke-dasharray"),
                       "rule": "evenodd"})

    def walk(e, ctm, style):
        tag = e.tag.split("}")[-1]
        report["elements_seen"] += 1
        if tag not in SUPPORTED_TAGS:
            report["unsupported"][tag] = report["unsupported"].get(tag, 0) + 1
            return
        if tag in _INERT:
            return
        m = _mat_mul(ctm, _parse_transform(e.get("transform")))
        st = dict(style)
        for k in ("fill", "stroke", "stroke-width", "fill-opacity", "stroke-opacity",
                  "opacity", "stroke-dasharray"):
            if e.get(k) is not None:
                st[k] = e.get(k)
        # 颜色解析（含渐变近似）
        fill = st.get("fill")
        if fill and "url(" in fill:
            col, gid = _gradient_mid(defs, fill)
            if col is not None:
                st["fill"] = "#%02x%02x%02x" % tuple(int(round(c * 255)) for c in col[:3])
                report["gradients_approximated"] += 1
            else:
                st["fill"] = None
        if tag in ("g", "svg"):
            for c in e:
                walk(c, m, st)
            return
        if tag == "text":
            report["unsupported"]["text"] = report["unsupported"].get("text", 0) + 1
            return
        subs, closed_flags = _geometry(tag, e, per_subpath=True)
        if not subs:
            report["unsupported"][tag + ":empty"] = report["unsupported"].get(tag + ":empty", 0) + 1
            return
        subs = [_apply(m, sp) for sp in subs]
        emit(tag, subs, all(closed_flags), st, subpath_closed=closed_flags)

    walk(root, _mat_identity(), {})
    meta = {"width": float(_floats(root.get("width"))[0]) if root.get("width") else vw,
            "height": float(_floats(root.get("height"))[0]) if root.get("height") else vh,
            "viewbox": (vx, vy, vw, vh)}
    return shapes, meta, report


def _apply(m, pts):
    if np is None:
        return [(m[0][0] * float(x) + m[0][1] * float(y) + m[0][2],
                 m[1][0] * float(x) + m[1][1] * float(y) + m[1][2])
                for x, y in pts]
    arr = np.asarray(pts, dtype=np.float64)
    if arr.size == 0:
        return arr.reshape((0, 2))
    hom = np.concatenate([arr, np.ones((arr.shape[0], 1))], axis=1)
    out = hom @ m.T
    return out[:, :2]


def _geometry(tag, e, *, per_subpath=False):
    """元素 → `(subpaths, closed)`（用户坐标，未经 transform）。

    默认返回 `(subpaths, bool)`；`per_subpath=True` 时第二项为与 `subpaths`
    等长的 `list[bool]`，供 `parse_svg` 保留 `<path>` 混合闭合/开放子路径语义。
    """
    if tag == "polygon":
        f = _floats(e.get("points"))
        pts = [(f[i], f[i + 1]) for i in range(0, len(f) - 1, 2)]
        subs, closed = ([pts] if len(pts) >= 3 else []), True
    elif tag == "polyline":
        f = _floats(e.get("points"))
        pts = [(f[i], f[i + 1]) for i in range(0, len(f) - 1, 2)]
        subs, closed = ([pts] if len(pts) >= 2 else []), False
    elif tag == "rect":
        x = float(e.get("x") or 0)
        y = float(e.get("y") or 0)
        w = float(e.get("width") or 0)
        h = float(e.get("height") or 0)
        if w <= 0 or h <= 0:
            subs, closed = [], True
        else:
            subs, closed = [[(x, y), (x + w, y), (x + w, y + h), (x, y + h)]], True
    elif tag == "circle":
        cx = float(e.get("cx") or 0)
        cy = float(e.get("cy") or 0)
        r = float(e.get("r") or 0)
        if r <= 0:
            subs, closed = [], True
        else:
            subs, closed = [_circle_pts(cx, cy, r)], True
    elif tag == "ellipse":
        cx = float(e.get("cx") or 0)
        cy = float(e.get("cy") or 0)
        rx = float(e.get("rx") or 0)
        ry = float(e.get("ry") or 0)
        if rx <= 0 or ry <= 0:
            subs, closed = [], True
        else:
            subs, closed = [_circle_pts(cx, cy, rx, ry)], True
    elif tag == "line":
        x1 = float(e.get("x1") or 0)
        y1 = float(e.get("y1") or 0)
        x2 = float(e.get("x2") or 0)
        y2 = float(e.get("y2") or 0)
        subs, closed = [[(x1, y1), (x2, y2)]], False
    elif tag == "path":
        subs, flags = _path_subpaths(e.get("d") or "")
        if per_subpath:
            return subs, flags
        return subs, (bool(flags) and all(flags))
    else:
        subs, closed = [], False
    if per_subpath:
        return subs, [closed] * len(subs)
    return subs, closed


def _circle_pts(cx, cy, rx, ry=None, n=96):
    ry = rx if ry is None else ry
    return [(cx + rx * math.cos(2 * math.pi * i / n),
             cy + ry * math.sin(2 * math.pi * i / n)) for i in range(n)]


def _read_path_nums(tokens, i, count):
    if i + count > len(tokens):
        return None
    vals = []
    for k in range(count):
        tk = tokens[i + k]
        if tk.isalpha():
            return None
        try:
            v = float(tk)
        except ValueError:
            return None
        if not math.isfinite(v):
            return None
        vals.append(v)
    return vals


def _path_subpaths(d):
    """路径 → `(subpaths, closed_flags)`。

    **只实现 M/L/H/V/Z + 小写相对版**（本产线只发射这些）；
    遇 C/S/Q/T/A 或非法词元 fail-closed 返回 `([], [])`（由调用方计入 unsupported）。
    """
    tokens = _PATH_TOKEN_RE.findall(d or "")
    subs = []
    closed_flags = []
    cur = []
    x = y = 0.0
    sx = sy = 0.0
    i = 0
    cmd = None
    started = False
    need_arg = False

    def _flush(is_closed):
        nonlocal cur
        if is_closed and len(cur) > 2 and cur[-1] == cur[0]:
            cur = cur[:-1]
        if len(cur) >= 2:
            subs.append(cur)
            closed_flags.append(is_closed)
        cur = []

    while i < len(tokens):
        tk = tokens[i]
        if tk.isalpha():
            if need_arg:
                return [], []
            cmd = tk
            i += 1
            if not started and cmd not in "Mm":
                return [], []
            if cmd in "Zz":
                if cur:
                    _flush(True)
                x, y = sx, sy
                need_arg = False
            elif cmd in "MmLlHhVv":
                need_arg = True
            else:
                # C/S/Q/T/A 等不在自用子集内：整条路径跳过（计入 unsupported），不猜
                return [], []
            continue

        if cmd is None or cmd in "Zz":
            return [], []

        rel = cmd.islower()
        c = cmd.upper()
        if c == "M":
            vals = _read_path_nums(tokens, i, 2)
            if vals is None:
                return [], []
            x0, y0 = vals
            i += 2
            if rel:
                x0 += x
                y0 += y
            if cur:
                _flush(False)
            x, y = x0, y0
            sx, sy = x, y
            cur = [(x, y)]
            started = True
            cmd = "l" if rel else "L"       # 后续隐式 L/l
            need_arg = False
        elif c == "L":
            vals = _read_path_nums(tokens, i, 2)
            if vals is None:
                return [], []
            x0, y0 = vals
            i += 2
            if rel:
                x0 += x
                y0 += y
            if not cur:
                cur = [(x, y)]
            x, y = x0, y0
            cur.append((x, y))
            need_arg = False
        elif c == "H":
            vals = _read_path_nums(tokens, i, 1)
            if vals is None:
                return [], []
            x0 = vals[0]
            i += 1
            if not cur:
                cur = [(x, y)]
            x = x + x0 if rel else x0
            cur.append((x, y))
            need_arg = False
        elif c == "V":
            vals = _read_path_nums(tokens, i, 1)
            if vals is None:
                return [], []
            y0 = vals[0]
            i += 1
            if not cur:
                cur = [(x, y)]
            y = y + y0 if rel else y0
            cur.append((x, y))
            need_arg = False
        else:
            return [], []

    if need_arg:
        return [], []
    if cur:
        _flush(False)
    return subs, closed_flags


# ─────────────────────────────────────────────────────────── 光栅化
class _Ctx:
    def __init__(self, w, h, ss, bg):
        self.w, self.h, self.ss = w, h, ss
        self.dw, self.dh = int(w * ss), int(h * ss)
        self.img = np.zeros((self.dh, self.dw, 4), dtype=np.float32)
        self.img[..., 0:3] = np.asarray(bg[:3], dtype=np.float32)
        self.img[..., 3] = float(bg[3]) if len(bg) > 3 else 1.0
        self.painted = np.zeros((self.dh, self.dw), dtype=bool)

    # -- 颜色/透明度：把 style 解析成 (rgba, alpha) ------------------------
    @staticmethod
    def _style_color(st, key, opacity_key):
        col = _color(st.get(key))
        if col is None:
            return None, 0.0
        alpha = col[3]
        o = st.get(opacity_key)
        if o is not None:
            alpha *= float(o)
        go = st.get("opacity")
        if go is not None:
            alpha *= float(go)
        return col, max(0.0, min(1.0, alpha))

    def draw(self, shape):
        fill, fa = self._style_color(shape, "fill", "fill-opacity")
        stroke, sa = self._style_color(shape, "stroke", "stroke-opacity")
        if fill is None and stroke is None:
            return 0
        subpaths = shape["subpaths"]
        sub_closed = shape.get("subpath_closed")
        if sub_closed is None or len(sub_closed) != len(subpaths):
            sub_closed = [bool(shape["closed"])] * len(subpaths)
        else:
            sub_closed = [bool(c) for c in sub_closed]

        prepared = []
        for sp, is_closed in zip(subpaths, sub_closed):
            arr = np.asarray(sp, dtype=np.float64)
            if arr.size == 0:
                continue
            prepared.append((arr * self.ss, is_closed))
        if not prepared:
            return 0

        touched = 0
        # 填充：同元素全部闭合子路径合并做 even-odd 扫描线（支持多块与带洞复合路径）
        if fill is not None:
            fill_polys = [arr for arr, is_closed in prepared if is_closed and len(arr) >= 3]
            if fill_polys:
                x0 = min(float(arr[:, 0].min()) for arr in fill_polys)
                x1 = max(float(arr[:, 0].max()) for arr in fill_polys)
                y0 = min(float(arr[:, 1].min()) for arr in fill_polys)
                y1 = max(float(arr[:, 1].max()) for arr in fill_polys)
                pad = 2.0
                ix0 = max(0, int(math.floor(x0 - pad)))
                ix1 = min(self.dw - 1, int(math.ceil(x1 + pad)))
                iy0 = max(0, int(math.floor(y0 - pad)))
                iy1 = min(self.dh - 1, int(math.ceil(y1 + pad)))
                if ix1 >= ix0 and iy1 >= iy0:
                    cov = _fill_subpaths(fill_polys, ix0, ix1, iy0, iy1)
                    _composite(self.img, self.painted, cov, fill, fa, ix0, iy0)
                    touched += 1

        # 描边：逐子路径按各自闭合标记切段，避免开放子路径被误连回起点
        if stroke is not None:
            lw = float(shape["stroke_width"]) * self.ss
            if lw > 0:
                r = lw / 2.0
                pad = 2.0 + lw
                for arr, is_closed in prepared:
                    segs = _dash_split(arr, is_closed, shape["dash"], self.ss)
                    if not segs:
                        continue
                    xs, ys = arr[:, 0], arr[:, 1]
                    ix0 = max(0, int(math.floor(float(xs.min()) - pad)))
                    ix1 = min(self.dw - 1, int(math.ceil(float(xs.max()) + pad)))
                    iy0 = max(0, int(math.floor(float(ys.min()) - pad)))
                    iy1 = min(self.dh - 1, int(math.ceil(float(ys.max()) + pad)))
                    if ix1 < ix0 or iy1 < iy0:
                        continue
                    scov = np.zeros((iy1 - iy0 + 1, ix1 - ix0 + 1), dtype=np.float32)
                    for a, b in segs:
                        _capsule(scov, ix0, iy0, a, b, r)
                    _composite(self.img, self.painted, scov, stroke, sa, ix0, iy0)
                    touched += 1
        return touched

    def result(self):
        img = self.img
        ss = self.ss
        if ss > 1:
            h2, w2 = (self.dh // ss) * ss, (self.dw // ss) * ss
            img = img[:h2, :w2].reshape(h2 // ss, ss, w2 // ss, ss, 4).mean(axis=(1, 3))
        out = np.clip(img[..., :4] * 255.0 + 0.5, 0, 255).astype(np.uint8)
        return out, self.painted


def _composite(img, painted, cov, color, alpha, ix0, iy0):
    if alpha <= 0:
        return
    sub = img[iy0:iy0 + cov.shape[0], ix0:ix0 + cov.shape[1], :]
    a = (cov * alpha)[..., None]
    sub[..., 0:3] = sub[..., 0:3] * (1.0 - a) + np.asarray(color[:3], dtype=np.float32) * a
    sub[..., 3] = sub[..., 3] * (1.0 - a[..., 0]) + a[..., 0]
    painted[iy0:iy0 + cov.shape[0], ix0:ix0 + cov.shape[1]] |= (cov > 0.5)


def _fill_subpaths(polys, ix0, ix1, iy0, iy1):
    """**自己的扫描线 even-odd 填充**：汇总全部闭合子路径的边求交 → 按 (行, x)
    排序 → 相邻交点配对成 span → 差分数组累加。对**非凸**多边形与**带洞/多块**
    复合路径同样正确（min/max 近似会把它填爆）。"""
    h = iy1 - iy0 + 1
    w = ix1 - ix0 + 1
    ys = np.arange(iy0, iy1 + 1, dtype=np.float64) + 0.5
    rows_all, xs_all = [], []
    for pts in polys:
        p = np.asarray(pts, dtype=np.float64)
        n = len(p)
        if n < 3:
            continue
        for i in range(n):
            ax, ay = p[i]
            bx, by = p[(i + 1) % n]
            if ay == by:
                continue
            lo, hi = (ay, by) if ay < by else (by, ay)
            sel = (ys >= lo) & (ys < hi)
            if not sel.any():
                continue
            idx = np.nonzero(sel)[0]
            t = (ys[idx] - ay) / (by - ay)
            rows_all.append(idx)
            xs_all.append(ax + (bx - ax) * t)
    if not rows_all:
        return np.zeros((h, w), dtype=np.float32)
    rows = np.concatenate(rows_all)
    xs = np.concatenate(xs_all)
    order = np.lexsort((xs, rows))
    rows = rows[order]
    xs = xs[order]
    grp_start = np.concatenate([[0], np.nonzero(np.diff(rows))[0] + 1])
    grp_count = np.diff(np.concatenate([grp_start, [len(rows)]]))
    pos = np.arange(len(rows)) - np.repeat(grp_start, grp_count)
    pairable = (pos % 2 == 0) & (pos + 1 < np.repeat(grp_count, grp_count))
    left = xs[pairable]
    right = xs[np.nonzero(pairable)[0] + 1]
    rw = rows[pairable]
    a = np.ceil(left - 0.5) - ix0
    b = np.floor(right - 0.5) - ix0
    keep = (right > left) & (b >= a) & (b >= 0) & (a < w)
    a = np.clip(a[keep], 0, w - 1).astype(np.int64)
    b = np.clip(b[keep], 0, w - 1).astype(np.int64)
    rw = rw[keep]
    if len(a) == 0:
        return np.zeros((h, w), dtype=np.float32)
    diff = np.zeros((h, w + 1), dtype=np.float32)
    np.add.at(diff, (rw, a), 1.0)
    np.add.at(diff, (rw, b + 1), -1.0)
    cov = np.cumsum(diff[:, :w], axis=1)
    return (cov > 0.5).astype(np.float32)


def _fill_coverage(pts, ix0, ix1, iy0, iy1, closed=True):
    """单多边形扫描线 even-odd 填充（委托 `_fill_subpaths`）。"""
    return _fill_subpaths([pts], ix0, ix1, iy0, iy1)


def _capsule(cov, ox, oy, a, b, r):
    """把线段 a→b 以半径 r 的胶囊覆盖写进 cov（取 max，不是叠加）。"""
    ax, ay = a
    bx, by = b
    lo_x = int(max(0, math.floor(min(ax, bx) - r - 1) - ox))
    hi_x = int(min(cov.shape[1] - 1, math.ceil(max(ax, bx) + r + 1) - ox))
    lo_y = int(max(0, math.floor(min(ay, by) - r - 1) - oy))
    hi_y = int(min(cov.shape[0] - 1, math.ceil(max(ay, by) + r + 1) - oy))
    if hi_x < lo_x or hi_y < lo_y:
        return
    xs = np.arange(lo_x, hi_x + 1, dtype=np.float64) + ox + 0.5
    ys = np.arange(lo_y, hi_y + 1, dtype=np.float64) + oy + 0.5
    px, py = np.meshgrid(xs, ys)
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    if L2 < 1e-12:
        d = np.hypot(px - ax, py - ay)
    else:
        t = np.clip(((px - ax) * vx + (py - ay) * vy) / L2, 0.0, 1.0)
        d = np.hypot(px - (ax + t * vx), py - (ay + t * vy))
    m = np.clip(r + 0.5 - d, 0.0, 1.0).astype(np.float32)
    reg = cov[lo_y:hi_y + 1, lo_x:hi_x + 1]
    np.maximum(reg, m, out=reg)


def _dash_split(pts, closed, dash, ss):
    """折线 → 段列表（`dash` 为 None ⇒ 原样；否则按 SVG 语义做长度切分）。"""
    segs = []
    n = len(pts)
    if n < 2:
        return segs
    pairs = [(i, i + 1) for i in range(n - 1)]
    if closed:
        pairs.append((n - 1, 0))
    if not dash:
        return [(pts[i], pts[j]) for i, j in pairs]
    pat = [float(x) * ss for x in str(dash).replace(",", " ").split() if x.strip()]
    if not pat:
        return [(pts[i], pts[j]) for i, j in pairs]
    if len(pat) == 1:
        pat = pat * 2
    total_len = sum(pat)
    if total_len <= 0:
        return []
    pos = 0.0
    pi_ = 0
    on = True
    for i, j in pairs:
        a = np.asarray(pts[i], dtype=np.float64)
        b = np.asarray(pts[j], dtype=np.float64)
        seg_len = float(np.hypot(*(b - a)))
        if seg_len <= 0:
            continue
        cur = 0.0
        while cur < seg_len:
            remain_pat = pat[pi_] - pos
            step = min(remain_pat, seg_len - cur)
            if on:
                t0, t1 = cur / seg_len, (cur + step) / seg_len
                segs.append((a + (b - a) * t0, a + (b - a) * t1))
            cur += step
            pos += step
            if pos >= pat[pi_] - 1e-12:
                pos = 0.0
                pi_ = (pi_ + 1) % len(pat)
                on = not on
    return segs


def rasterize(src, width=None, height=None, ss=2, background=(0.0, 0.0, 0.0, 0.0)):
    """SVG（路径或文本）→ `(RGBA uint8 数组, report)`。

    `width/height` 缺省用 SVG 自身尺寸；`ss` 为超采样倍数（1 = 关抗锯齿）。
    """
    _require_numpy("rasterize")
    text = load_svg_text(src)
    shapes, meta, report = parse_svg(text)
    w = int(round(width or meta["width"]))
    h = int(round(height or meta["height"]))
    if w <= 0 or h <= 0:
        raise ValueError("画布尺寸非法：%dx%d" % (w, h))
    # viewBox → 画布 的等比缩放（preserveAspectRatio 默认 xMidYMid meet）
    vx, vy, vw, vh = meta["viewbox"]
    if vw > 0 and vh > 0:
        s = min(w / vw, h / vh)
        tx = (w - vw * s) / 2.0 - vx * s
        ty = (h - vh * s) / 2.0 - vy * s
    else:
        s, tx, ty = 1.0, 0.0, 0.0
    for sh in shapes:
        sh["subpaths"] = [_apply(np.array([[s, 0, tx], [0, s, ty], [0, 0, 1.0]]), sp)
                          for sp in sh["subpaths"]]
        sh["stroke_width"] = float(sh["stroke_width"]) * s
    ctx = _Ctx(w, h, ss, background)
    drawn = 0
    for sh in shapes:
        if ctx.draw(sh):
            drawn += 1
    img, painted = ctx.result()
    report.update({"width": w, "height": h, "ss": ss, "shapes_parsed": len(shapes),
                   "shapes_drawn": drawn})
    return img, report


def foreground_mask(src, width=None, height=None, ss=2, threshold=0.5):
    """前景掩码：alpha ≥ threshold 的像素（用于与 Blender 的 film_transparent 对齐）。"""
    img, report = rasterize(src, width=width, height=height, ss=ss)
    return (img[..., 3].astype(np.float32) / 255.0) >= threshold, img, report


def iou(mask_a, mask_b):
    """前景 IoU。两掩码尺寸须一致（不一致直接报错，不猜）。

    返回值 `(iou, detail)`。**空集语义**（#699 F3）：两掩码**都为空**时 IoU 在数学上是
    `0/0`（未定义）。旧版返回 `1.0`，把"两个空掩码"判成"完全重合"（假成功：双空掩码
    能骗过 `iou >= 0.90` 这类下限门）。现改为返回 `0.0` 并**在明细里显式标 `undefined`**
    —— 消费方据此可区分"真的算出来是 0"（`undefined` 缺席）与"无从判定"（`undefined=True`）。
    """
    _require_numpy("iou")
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("掩码尺寸不一致：%s vs %s" % (a.shape, b.shape))
    inter = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    if union == 0:
        #: 两掩码全空 ⇒ 未定义（不是"完全重合"，也不是"完全不同"）
        return 0.0, {"inter": 0, "union": 0, "a": 0, "b": 0,
                     "empty": True, "undefined": True}
    return inter / float(union), {"inter": inter, "union": union,
                                  "a": int(np.count_nonzero(a)),
                                  "b": int(np.count_nonzero(b))}
