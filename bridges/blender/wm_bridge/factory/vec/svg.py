# -*- coding: utf-8 -*-
"""wm_bridge.factory.vec.svg · SVG 文档生成器（纯标准库 · 确定性 · 零依赖）
================================================================================
为什么需要它：
  Blender 自带 `wm.grease_pencil_export_svg`（实测可用，见 `bpy_vec.py` 的读数），但它
  ①只出**线稿**（Line Art 笔画，纯黑填充、无分色）；②画布由笔画包围盒自动决定，**不可控**；
  ③只能表达"GP 时间线"这一种分层。要出**可编辑、分层、彩色、确定性**的矢量稿——
  也就是海报与设计稿那种"线稿 + 色块 + 图层"——需要自己的文档生成器。

设计要点（每条都有理由）：
  · **确定性**：坐标按固定小数位格式化、属性按固定序输出、无时间戳/无随机 ⇒
    同一场景两次导出**逐字节相同**（可哈希、可版本控制、可回归）。
  · **分层是一等公民**：`<g id="...">` 由调用方命名，空层也照样发射（图层结构稳定，
    设计师在 Illustrator/Inkscape 里看到的层序不随内容抖动）。
  · **不静默**：退化输入（顶点 <3 / 零尺寸 / NaN / 超大坐标）**不发射**该元素，
    但**计数**进 `stats()["skipped"]`——"不画可以，但要说"。
  · **画家算法天然成立**：SVG 按文档序绘制 ⇒ 与"远先画"同构，调用方不必额外排序。
  · **与 Xml 解析面对齐**：`count_elements()` 用 `xml.etree.ElementTree` 数元素，
    与 `stats()["emitted"]` 必须相等（这是本模块的对外判据之一，见 `verify()`）。
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET

__all__ = ["SVGDocument", "Layer", "SVGError", "PRECISION", "MAX_COORD",
           "count_elements", "fmt_number", "norm_color"]

#: 坐标输出小数位（3 位 = 亚像素，且输出短且稳定）
PRECISION = 3
#: 坐标绝对值上限。超过 ⇒ **不发射**并计数：这类数会写出 12+ 位文本，且在多数渲染器里
#: 触发"巨大画布/精度丢失"，是典型的"看起来成功其实画错"，故 fail-closed。
MAX_COORD = 1e6
#: 可绘制的元素（`stats()["emitted"]` 只数这些；`<g>`/`<defs>` 属于结构层，另计）
DRAWABLE_TAGS = ("polygon", "polyline", "path", "rect", "circle", "ellipse", "line", "text")
#: 属性输出的**固定顺序**（不在表里的按调用方插入序追加）
_ATTR_ORDER = ("id", "class", "x", "y", "x1", "y1", "x2", "y2", "width", "height",
               "cx", "cy", "r", "rx", "ry", "d", "points", "transform",
               "fill", "fill-opacity", "stroke", "stroke-width", "stroke-opacity",
               "stroke-dasharray", "stroke-linecap", "stroke-linejoin", "opacity")
#: 需要按数值格式化的属性
_NUM_ATTRS = frozenset(("x", "y", "x1", "y1", "x2", "y2", "width", "height",
                        "cx", "cy", "r", "rx", "ry", "stroke-width",
                        "fill-opacity", "stroke-opacity", "opacity"))


class SVGError(ValueError):
    """SVG 文档生成期的契约错误（非法画布/非法层引用等）。"""


def fmt_number(v, precision=PRECISION):
    """有限数值 → 确定的十进制文本；非有限 ⇒ `None`（调用方据此跳过并计数）。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    if f == 0.0:
        return "0"                      # −0.0 也归零，保证逐字节稳定
    s = ("%%.%df" % precision) % f
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


def norm_color(c):
    """颜色 → `(#rrggbb, opacity 或 None)`；认不出 ⇒ None。

    接受：`'#rrggbb'` / `'#rgb'` / `'none'`；灰度 int 0..255；3/4 元组（int 0..255 或
    float 0..1）。越界分量按惯例钳位（不抛错），但**计数**由调用方负责（本函数返回
    `clamped` 标志）。为什么钳位而不是拒绝：拒绝会让"同一个 spec 的两条出口"画出不同的
    画（位图路径会钳位），画面一致比严格更重要——与 `lingshu` 侧 svg_backend 的实测
    结论一致（越界分量按 PIL 行为钳位，同时计数）。
    """
    if c is None:
        return None
    if isinstance(c, str):
        s = c.strip().lower()
        if s == "none":
            return ("none", None, False)
        if s.startswith("#"):
            h = s[1:]
            if len(h) == 3:
                h = "".join(ch * 2 for ch in h)
            if len(h) == 6:
                try:
                    int(h, 16)
                except ValueError:
                    return None
                return ("#" + h, None, False)
        return None
    if isinstance(c, bool):
        return None
    if isinstance(c, (int, float)):
        v = min(255, max(0, int(c)))
        clamped = not (0 <= float(c) <= 255)
        return ("#%02x%02x%02x" % (v, v, v), None, clamped)
    if isinstance(c, (tuple, list)):
        if len(c) not in (3, 4):
            return None
        try:
            raw = [float(x) for x in c]
        except (TypeError, ValueError):
            return None
        unit = all(isinstance(x, float) for x in c) and all(0.0 <= x <= 1.0 for x in raw)
        vals = [x * 255.0 for x in raw] if unit else raw
        clamped = any(x < 0.0 or x > 255.0 for x in vals)
        vals = [min(255.0, max(0.0, x)) for x in vals]
        rgb = tuple(int(round(x)) for x in vals[:3])
        op = None
        if len(vals) == 4:
            a = vals[3] / 255.0 if not unit else raw[3]
            a = min(1.0, max(0.0, a))
            if a < 1.0:
                op = fmt_number(a, 3)
        return ("#%02x%02x%02x" % rgb, op, clamped)
    return None


def count_elements(svg_text):
    """语义化计数：用 ElementTree 解析 `svg_text`，返回 `{tag: n}` 与总数。

    这是"自报 emitted 与真实文档一致"的**独立**判据面——数的是解析后的元素，
    不是生成器自己的计数器。
    """
    root = ET.fromstring(svg_text)
    counts = {}
    stack = [root]
    total = 0
    while stack:
        e = stack.pop()
        tag = e.tag.split("}")[-1]
        counts[tag] = counts.get(tag, 0) + 1
        total += 1
        stack.extend(list(e))
    counts["__total__"] = total
    counts["__drawable__"] = sum(counts.get(t, 0) for t in DRAWABLE_TAGS)
    return counts


class Layer:
    """一个图层：`<g id="...">` + 若干元素（按加入序 = 绘制序 = 远先画）。"""

    __slots__ = ("id", "label", "attrs", "elements", "kind", "children")

    def __init__(self, id, label=None, kind="draw", attrs=None):
        self.id = id
        self.label = label
        self.kind = kind
        self.attrs = dict(attrs or {})
        self.elements = []              # [(tag, attrs_dict)]
        self.children = []              # SMIL 等：挂在 <g> 下的原样 XML 子元素

    def __len__(self):
        return len(self.elements)

    def __repr__(self):                 # pragma: no cover
        return "Layer(%r, %d elems)" % (self.id, len(self.elements))


class SVGDocument:
    """分层 SVG 文档生成器。

    用法::

        doc = SVGDocument(1920, 1080, background=(14, 18, 28))
        body = doc.layer("body", label="主体")
        doc.polygon([(10, 10), (30, 10), (30, 30)], layer=body,
                    fill=(220, 90, 60), stroke="#101010", stroke_width=1.5)
        svg = doc.to_svg()
    """

    def __init__(self, width, height, background=None, viewbox=None,
                 title=None, desc=None, precision=PRECISION):
        w, h = int(width), int(height)
        if w <= 0 or h <= 0:
            raise SVGError("画布尺寸须为正整数，实为 %dx%d" % (w, h))
        self.width = w
        self.height = h
        self.viewbox = tuple(float(v) for v in (viewbox or (0.0, 0.0, w, h)))
        if len(self.viewbox) != 4:
            raise SVGError("viewBox 须为 4 个数")
        self.title = title
        self.desc = desc
        self.precision = int(precision)
        self._defs = []
        self._layers = []
        self._by_id = {}
        self._emitted = 0
        self._skipped = {}
        self._clamped = 0
        self._unsupported = {}
        if background is not None:
            bg = self.layer("background", label="背景")
            col = norm_color(background)
            if col is None:
                self._skip("background_color")
            else:
                if col[2]:
                    self._clamped += 1
                self.rect([0, 0, w, h], layer=bg, fill=background)

    # ------------------------------------------------------------ 图层
    def layer(self, id, label=None, kind="draw", attrs=None):
        """建层（同名复用）。`kind='draw'` 为绘制层；`'frame'` 为版面/边框层。"""
        if id in self._by_id:
            return self._by_id[id]
        lay = Layer(id, label=label, kind=kind, attrs=attrs)
        self._layers.append(lay)
        self._by_id[id] = lay
        return lay

    def _resolve(self, layer):
        if layer is None:
            return None
        if isinstance(layer, Layer):
            return layer
        return self._by_id.get(layer) or self.layer(layer)

    def add_defs(self, raw):
        """追加 `<defs>` 片段（渐变/图案等）；调用方负责是合法 XML 片段。"""
        self._defs.append(str(raw))

    # ------------------------------------------------------------ 原语
    def polygon(self, points, layer=None, **style):
        pts = self._points(points, 3)
        if pts is None:
            return None
        return self._emit("polygon", {"points": pts}, layer, style)

    def polyline(self, points, layer=None, **style):
        pts = self._points(points, 2)
        if pts is None:
            return None
        return self._emit("polyline", {"points": pts}, layer, style)

    def path(self, d, layer=None, **style):
        if not isinstance(d, str) or not d.strip():
            self._skip("path_d")
            return None
        return self._emit("path", {"d": d}, layer, style)

    def rect(self, box, layer=None, **style):
        x0, y0, x1, y1 = self._box(box)
        if x0 is None:
            return None
        w = float(x1) - float(x0)
        h = float(y1) - float(y0)
        if w <= 0 or h <= 0:
            self._skip("rect_degenerate")
            return None
        return self._emit("rect", {"x": x0, "y": y0, "width": w, "height": h}, layer, style)

    def circle(self, center, r, layer=None, **style):
        cx, cy = center
        if not self._ok_pair(cx, cy):
            return None
        rr = self._ok_num(r)
        if rr is None:
            return None
        if rr <= 0:
            self._skip("circle_degenerate")
            return None
        return self._emit("circle", {"cx": cx, "cy": cy, "r": rr}, layer, style)

    def ellipse(self, box, layer=None, **style):
        x0, y0, x1, y1 = self._box(box)
        if x0 is None:
            return None
        rx = (float(x1) - float(x0)) / 2.0
        ry = (float(y1) - float(y0)) / 2.0
        if rx <= 0 or ry <= 0:
            self._skip("ellipse_degenerate")
            return None
        return self._emit("ellipse", {"cx": (float(x0) + float(x1)) / 2.0,
                                      "cy": (float(y0) + float(y1)) / 2.0,
                                      "rx": rx, "ry": ry}, layer, style)

    def line(self, p0, p1, layer=None, **style):
        if not self._ok_pair(*p0) or not self._ok_pair(*p1):
            return None
        if float(p0[0]) == float(p1[0]) and float(p0[1]) == float(p1[1]):
            self._skip("line_degenerate")
            return None
        return self._emit("line", {"x1": p0[0], "y1": p0[1], "x2": p1[0], "y2": p1[1]},
                          layer, style)

    def text(self, x, y, content, layer=None, **style):
        if not isinstance(content, str) or not content:
            self._skip("text_content")
            return None
        if not self._ok_pair(x, y):
            return None
        st = dict(style)
        st["_text"] = content
        return self._emit("text", {"x": x, "y": y}, layer, st)

    def raw(self, xml_fragment, drawable=True, layer=None):
        """直接塞一段已排版好的元素（逃生舱）；`drawable` 决定是否计入 emitted。"""
        lay = self._resolve(layer)
        if lay is None:
            raise SVGError("raw() 必须指定层")
        lay.elements.append(("__raw__", {"xml": str(xml_fragment)}))
        if drawable:
            self._emitted += 1
        return xml_fragment

    # ------------------------------------------------------------ 输出
    def to_svg(self):
        """导出完整 SVG 文档（确定性：同一调用序列 ⇒ 逐字节相同）。"""
        vb = " ".join(fmt_number(v, self.precision) or "0" for v in self.viewbox)
        head = ['<?xml version="1.0" encoding="UTF-8"?>',
                '<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
                'width="%d" height="%d" viewBox="%s">' % (self.width, self.height, vb)]
        if self.title:
            head.append("  <title>%s</title>" % _esc(self.title))
        if self.desc:
            head.append("  <desc>%s</desc>" % _esc(self.desc))
        if self._defs:
            head.append("  <defs>")
            for d in self._defs:
                head.append("    " + d)
            head.append("  </defs>")
        body = []
        for lay in self._layers:
            attrs = {"id": lay.id}
            if lay.label:
                attrs["data-label"] = lay.label
            attrs.update(lay.attrs)
            kids = [_element_str(tag, a, self.precision, 4) for tag, a in lay.elements]
            kids.extend(lay.children)
            body.append(_element_str("g", dict(attrs, _children=kids), self.precision, 2))
        return "\n".join(head + body + ["</svg>", ""])

    def write(self, path, encoding="utf-8"):
        svg = self.to_svg()
        with open(path, "w", encoding=encoding, newline="\n") as fh:
            fh.write(svg)
        return svg

    # ------------------------------------------------------------ 观测面
    def stats(self):
        """发射/跳过/钳位的完整观测面（跳过必须能看见）。"""
        groups = len(self._layers)
        return {"emitted": self._emitted, "groups": groups,
                "elements_in_layers": sum(len(l.elements) for l in self._layers),
                "skipped": dict(self._skipped), "clamped": self._clamped,
                "unsupported": dict(self._unsupported),
                "width": self.width, "height": self.height,
                "layers": [l.id for l in self._layers]}

    def verify(self, svg_text=None):
        """对拍：自报 `emitted` vs ElementTree 解析出的可绘制元素数。"""
        svg = svg_text if svg_text is not None else self.to_svg()
        c = count_elements(svg)
        return {"emitted": self._emitted, "parsed_drawable": c["__drawable__"],
                "parsed_total": c["__total__"], "groups": c.get("g", 0),
                "ok": self._emitted == c["__drawable__"], "by_tag": c}

    # ------------------------------------------------------------ 内部
    def _emit(self, tag, attrs, layer, style):
        st = dict(style)
        text = st.pop("_text", None)
        lay = self._resolve(layer)
        if lay is None:
            raise SVGError("%s 必须指定 layer（否则文档没有确定的分层）" % tag)
        fill = st.pop("fill", None)
        stroke = st.pop("stroke", None)
        sw = st.pop("stroke_width", st.pop("stroke-width", None))
        fo = st.pop("fill_opacity", st.pop("fill-opacity", None))
        so = st.pop("stroke_opacity", st.pop("stroke-opacity", None))
        op = st.pop("opacity", None)
        dash = st.pop("dash", st.pop("stroke_dasharray", st.pop("stroke-dasharray", None)))
        kids = st.pop("children", None)

        out = dict(attrs)
        for key, val in ((("fill-opacity"), fo), ("stroke-opacity", so), ("opacity", op)):
            if val is not None:
                out[key] = val
        # 颜色：None 与 'none' 都表示"不填/不描"
        if fill is None:
            out["fill"] = "none"
        else:
            col = norm_color(fill)
            if col is None:
                self._skip("fill_color")
                out["fill"] = "none"
            else:
                if col[2]:
                    self._clamped += 1
                out["fill"] = col[0]
                if col[1] is not None:
                    out["fill-opacity"] = col[1]
        if stroke is not None:
            col = norm_color(stroke)
            if col is None:
                self._skip("stroke_color")
            elif col[0] != "none":
                if col[2]:
                    self._clamped += 1
                out["stroke"] = col[0]
                out["stroke-width"] = float(sw) if sw is not None else 1.0
                if col[1] is not None:
                    out["stroke-opacity"] = col[1]
        if dash is not None:
            out["stroke-dasharray"] = dash
        passthrough = ("stroke_linecap", "stroke-linejoin", "fill_rule", "transform", "class",
                       "font_family", "font_size", "font_weight", "text_anchor")
        for k in passthrough:
            if k in st:
                out[k.replace("_", "-")] = st.pop(k)
        out.update(st)
        if kids:
            out["_children"] = list(kids)
        if text is not None:
            out["_text"] = text
        lay.elements.append((tag, out))
        self._emitted += 1
        return out

    def _skip(self, why):
        self._skipped[why] = self._skipped.get(why, 0) + 1

    def _ok_num(self, v):
        s = fmt_number(v, self.precision)
        if s is None:
            self._skip("nonfinite")
            return None
        if abs(float(v)) > MAX_COORD:
            self._skip("coord_overflow")
            return None
        return float(v)

    def _ok_pair(self, x, y):
        return self._ok_num(x) is not None and self._ok_num(y) is not None

    def _points(self, points, minimum):
        """点列 → `"x,y x,y ..."`；任何非法 ⇒ None（并计数），保证不发射坏元素。"""
        if points is None:
            self._skip("points_none")
            return None
        try:
            seq = list(points)
        except TypeError:
            self._skip("points_type")
            return None
        if seq and not isinstance(seq[0], (tuple, list)):
            if len(seq) % 2:
                self._skip("points_flat_odd")
                return None
            seq = [(seq[i], seq[i + 1]) for i in range(0, len(seq), 2)]
        out = []
        for p in seq:
            if not isinstance(p, (tuple, list)) or len(p) < 2:
                self._skip("points_shape")
                return None
            if self._ok_num(p[0]) is None or self._ok_num(p[1]) is None:
                return None
            xs = fmt_number(p[0], self.precision)
            ys = fmt_number(p[1], self.precision)
            if not out or (out[-1] != (xs, ys)):
                out.append((xs, ys))     # 去连续重复点（零长边不改变形状，只会让文件变脏）
        if len(out) < minimum:
            self._skip("points_too_few")
            return None
        return " ".join("%s,%s" % p for p in out)

    def _box(self, box):
        if box is None:
            self._skip("box_none")
            return (None,) * 4
        try:
            seq = list(box)
        except TypeError:
            self._skip("box_type")
            return (None,) * 4
        if len(seq) != 4:
            self._skip("box_shape")
            return (None,) * 4
        if self._ok_num(seq[0]) is None or self._ok_num(seq[1]) is None or \
                self._ok_num(seq[2]) is None or self._ok_num(seq[3]) is None:
            return (None,) * 4
        return seq


def _esc(s):
    # #2099：属性值双引号不转义 ⇒ label 含 `"` 产出 not well-formed 的 SVG
    # （verify() 直接 ParseError，实测）。与姊妹实现 lingshu/world/svg_backend.py
    # 的 _attr_escape 同款（#1091 已把「属性值破坏文档结构」定为红线）；文本面
    # 上 &quot; 语义等价，统一转义无副作用。
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _attrs_str(attrs, precision):
    """属性字典 → 文本（固定序 + 数值格式化 + 转义）。"""
    ordered = []
    seen = set()
    for k in _ATTR_ORDER:
        if k in attrs:
            ordered.append((k, attrs[k]))
            seen.add(k)
    for k, v in attrs.items():
        if k not in seen and k != "_text":
            ordered.append((k, v))
    parts = []
    for k, v in ordered:
        if v is None:
            continue
        if k in _NUM_ATTRS:
            s = fmt_number(v, precision)
            if s is None:
                continue
            parts.append('%s="%s"' % (k, s))
        else:
            parts.append('%s="%s"' % (k, _esc(v)))
    return " ".join(parts)


def _element_str(tag, attrs, precision, indent=0):
    """一个元素 → 文本（可多行）。`__raw__` 原样透出；`_children` 递归；`<text>` 有内容。"""
    if tag == "__raw__":
        return (" " * indent) + attrs["xml"]
    a = dict(attrs)
    text = a.pop("_text", None)
    kids = a.pop("_children", None)
    body = _attrs_str(a, precision)
    open_tag = "<%s%s>" % (tag, (" " + body) if body else "")
    if kids:
        pad = " " * (indent + 2)
        inner = "\n".join(pad + k.strip() for k in kids)
        return (" " * indent) + open_tag + "\n" + inner + "\n" + (" " * indent) + "</%s>" % tag
    if text is None:
        return (" " * indent) + (("<%s %s/>" % (tag, body)) if body else "<%s/>" % tag)
    return (" " * indent) + "<%s%s>%s</%s>" % (tag, (" " + body) if body else "", _esc(text), tag)
