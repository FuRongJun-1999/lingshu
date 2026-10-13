# -*- coding: utf-8 -*-
"""svg_backend · 矢量渲染后端（与 `ImageDraw` **同形**的绘图调用 → SVG 文档）
============================================================================
为什么需要它（缺陷/能力缺口）：
  呈现层（`world3d` / `silhouette3d` / `scene_model`）**只能出位图** —— 光栅化在
  分配位图时就固定了分辨率。这带来三件在渲染外层无解的事：
  ① 放大即糊：`World3D.render(1600,1600)` 与 `render(400,400)` 是两次不同的绘制，
     不是同一张画的两种尺度；
  ② 边缘是硬的：位图路径没有抗锯齿，斜边/圆弧全是台阶（实测 640px 下硬边像素 3633）；
  ③ 没法动：一张 PNG 内部不可寻址，任何"让某个部件动起来"都必须重画整张。

修法（**只加一层，不改既有绘制逻辑**）：把绘制侧已经在用的**四个原语**收在一层
后端后面 —— `polygon` / `rectangle` / `ellipse` / `line`（全管线只有 9 处调用）。
`ImageDraw` 与新后端**同形**，故绘制代码一行不用改；既有的 PIL 路径**逐字节不变**。

判据来源：绘制面实测（`grep -oE "draw\\.[a-z_]+\\(" ` ⇒ 仅上列 4 个原语）；
形状/盒体/轮廓的绘制顺序已由既有守卫钉住（#311 面剔除、#312 部件深度序、#403 近平面裁剪），
本件**不改这些顺序**，只换出口。

设计要点（这几条是刻意选的）：
  · **画家算法天然成立**：SVG 按文档顺序绘制，与既有"远先画"的顺序**同构**
    ⇒ 不需要另外给矢量后端做排序。
  · **纯标准库**：本模块不 import PIL（`world` 的位图路径依赖 pillow extra，
    矢量路径不该被它拖住）；也不 import 本仓其它模块。
  · **确定性输出**：坐标按固定小数位格式化 ⇒ 同样的场景两次导出**逐字节相同**
    （便于守卫按文本比对，也便于版本控制）。
  · **不静默**：退化输入（顶点数 < 3 的多边形、零面积椭圆、非有限坐标）**不发射**
    元素，并**计数**进 `stats()` —— 与"禁止静默给错"同一条纪律：不画可以，但要说。

边界（如实）：
  · 只覆盖上述 4 个原语。`scene_model` 的 `L` 模式蒙版（`fill=255`）按**灰度**填充
    处理，语义与原路径一致（用 `L` 图当蒙版），但本模块**不**做位图布尔运算。
  · 不做渐变/阴影/纹理——那属于呈现层的表现力，不在"换出口"这件事的范围。
  · **背景色只认 RGB/RGBA/灰度**（与元素同一套）。PIL 还认颜色名（`'black'`）与
    十六进制（`'#ff0000'`）；本后端不认 ⇒ 计入 `stats()["skipped"]["background_color"]`
    并退回黑（与 `background=None` 同），**不假装成功**。本管线只传三元组，不受影响。
  · 越界颜色**照 PIL 钳位**并计数（`stats()["clamped"]`）；反序矩形/椭圆**照 PIL 抛
    `ValueError`**（同文本）——这两条是按独立复核方的实测把两出口对齐，不是自定语义。
  · 输出的 SVG 是**用户单位像素坐标**（`viewBox="0 0 W H"`、`width/height` 同值），
    与位图后端的像素网格一一对应。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

__all__ = ["SVGBackend", "SVG_PRECISION"]

#: 坐标输出的小数位。3 位 = 亚像素级精度（远小于一个像素），且让输出稳定可复现。
SVG_PRECISION = 3


def _num(v) -> str:
    """把坐标/尺寸格式化成**确定**的十进制文本（去尾零，保留至多 SVG_PRECISION 位）。

    非有限值（NaN/inf）⇒ 返回 `None`，调用方据此**跳过**该元素（不发射非法 SVG）。
    """
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):     # NaN / ±inf
        return None
    s = ("%%.%df" % SVG_PRECISION) % f
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


def _color(c) -> Optional[Tuple[str, Optional[str], bool]]:
    """颜色 → (fill 属性值, opacity 属性值或 None, 是否被钳位)。支持 RGB / RGBA / 灰度整数。

    **越界分量按 PIL 的行为钳到 0..255**（实测：`ImageDraw.rectangle(fill=(999,-5,0))`
    ⇒ 落到 `(255,0,0)`，不抛错）。为什么不像早先那样"拒绝并不发射"：那会让两条出口
    **画出不同的画**（实测对抗用例里逐像素全等率掉到 0.4716 —— 本件最大分歧），
    而"同一条管线两个出口"的判据是**画面一致**。钳位的同时**计数**（`stats()["clamped"]`），
    所以既不静默、也不与位图分道。

    返回 None ⇒ 颜色**形状**不合法（不是 3/4 元组、也不是数值），调用方跳过并计数。
    """
    if c is None:
        return None
    if isinstance(c, (int, float)) and not isinstance(c, bool):
        v = int(c)
        vc = min(255, max(0, v))
        return ("rgb(%d,%d,%d)" % (vc, vc, vc), None, not (0 <= v <= 255))
    if isinstance(c, (tuple, list)):
        if len(c) not in (3, 4):
            return None
        try:
            vals = [int(x) for x in c]
        except (TypeError, ValueError):
            return None
        clamped = any(x < 0 or x > 255 for x in vals)
        vals = [min(255, max(0, x)) for x in vals]
        if len(vals) == 3:
            return ("rgb(%d,%d,%d)" % tuple(vals), None, clamped)
        r, g, b, a = vals
        if a >= 255:
            return ("rgb(%d,%d,%d)" % (r, g, b), None, clamped)
        return ("rgb(%d,%d,%d)" % (r, g, b), _num(a / 255.0), clamped)
    return None


def _attr_escape(v: str) -> str:
    """属性值转义：`&` 与引号（id 里出现引号会直接破坏文档结构，故 fail-closed 转义）。"""
    return (str(v).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


class SVGBackend:
    """与 `ImageDraw` 同形的绘图后端：把调用**按序**录成 SVG 元素。

    用法（与 `ImageDraw.Draw(img)` 完全同形，故绘制代码无需改动）::

        b = SVGBackend(640, 640, background=(20, 24, 40))
        b.rectangle([0, 288, 640, 640], fill=(30, 34, 50))
        b.polygon([(10, 10), (30, 10), (30, 30)], fill=(200, 80, 40), outline=(140, 56, 28))
        svg = b.to_svg()
    """

    def __init__(self, width: int, height: int,
                 background: Tuple[int, int, int] = (20, 24, 40)) -> None:
        self.width = int(width)
        self.height = int(height)
        self.background = background
        self._elements: List[str] = []
        #: 分层（`<g id="...">`）——**纯增量能力**：一次都不调用 `begin_group()`
        #: 时下面两个容器恒空，`to_svg()` 走的是与改动前**逐字节相同**的分支。
        #: 为什么用 mark 而不是往 `_elements` 里塞标记：`_elements` 保持"字符串列表"
        #: 的既有形态，`__len__` / `stats()["elements"]` 语义不变（零回归的前提）。
        self._marks: List[Tuple[int, str, str]] = []
        self._group_stack: List[str] = []
        self._group_ids: List[str] = []
        #: 观测面：发射/跳过的元素数（跳过的必须能看见——不静默）
        self._emitted = 0
        self._skipped: Dict[str, int] = {}
        #: 越界颜色被钳位的次数（与 PIL 同行为，但**不静默**）
        self._clamped = 0
        #: 背景色**在构造时校验一次**。为什么不在 `to_svg()` 里做（独立复核方实测出的三条）：
        #   · 放在 `to_svg()` 里 ⇒ 不导出就查不到、**导出两次计数翻倍**（输出幂等、计数不幂等）；
        #   · 调用方（如 `World3D.render_svg`）习惯在 `to_svg()` **之前**取 `stats()`，
        #     于是经那个公开入口**永远看不到**背景这一条（实测 `background=(300,-5,0)`
        #     走 render_svg 得 clamped=0，直连后端得 1）。
        #   `background=None` 照 PIL：`Image.new('RGB', size, None)` ⇒ 黑（实测）。
        if background is None:
            self._bg = _color((0, 0, 0))
        else:
            self._bg = _color(background)
            if self._bg is None:
                # 认不出的形态（如颜色名 'black'、'#ff0000'）：PIL 认，本后端不认
                # ⇒ **计数**（能看见）并退回黑，不假装成功。
                self._skip("background_color")
                self._bg = _color((0, 0, 0))
            elif self._bg[2]:
                self._clamped += 1

    # ------------------------------------------------------------ 分层（增量）
    def begin_group(self, gid: str, label: str = None, **attrs) -> str:
        """开一个图层 `<g id="gid" [data-label=...] [其它属性]>`，返回 `gid`。

        为什么矢量出口需要它：SVG 的**独特价值**是"一个文件里有可寻址的结构"——
        设计师在 Illustrator/Inkscape 里能按图层选中/改色/隐藏。位图出口没有对应物，
        所以这是矢量出口相对位图出口的**能力新增**，不是两条出口的差异来源
        （几何、投影、画家序仍与位图共用 `_paint` + `prepare_triangles`）。

        口径（fail-closed，不静默）：
          · `gid` 须为非空字符串且**全局唯一**（重复 id 会让编辑器的图层选择产生歧义）；
          · 允许嵌套；`to_svg()` 时仍有未闭合的层 ⇒ 抛 `ValueError`（不自动补 `</g>`——
            自动补会把"调用方写错了"变成"看起来正常的一份文档"）。
        """
        if not isinstance(gid, str) or not gid.strip():
            raise ValueError("图层 id 须为非空字符串，实为 %r" % (gid,))
        if gid in self._group_ids:
            raise ValueError("图层 id 重复：%r（下游按 id 寻址，重复会产生歧义）" % (gid,))
        body = 'id="%s"' % _attr_escape(gid)
        if label:
            body += ' data-label="%s"' % _attr_escape(label)
        for k in sorted(attrs):
            v = attrs[k]
            if v is None:
                continue
            body += ' %s="%s"' % (_attr_escape(str(k)), _attr_escape(str(v)))
        self._marks.append((len(self._elements), "open", body))
        self._group_stack.append(gid)
        self._group_ids.append(gid)
        return gid

    def end_group(self) -> str:
        """闭合最近一个未闭合图层；没有未闭合层 ⇒ 抛 `ValueError`（不静默吞掉）。"""
        if not self._group_stack:
            raise ValueError("end_group() 没有对应的 begin_group()")
        gid = self._group_stack.pop()
        self._marks.append((len(self._elements), "close", gid))
        return gid

    def groups(self) -> List[str]:
        """已开图层 id 列表（按开启序）——观测面。"""
        return list(self._group_ids)

    # ------------------------------------------------------------ 四个原语
    def polygon(self, xy: Sequence, fill=None, outline=None) -> None:
        """多边形（PIL 同形）：`xy` = 顶点序列 `[(x, y), ...]`（也接受扁平 `[x0,y0,x1,y1,...]`）。"""
        pts = self._points(xy)
        if pts is None or len(pts) < 3:
            self._skip("polygon_pts")
            return
        self._emit("polygon", self._fill_stroke(fill, outline) +
                   ' points="%s"' % " ".join("%s,%s" % p for p in pts))

    def rectangle(self, xy: Sequence, fill=None, outline=None, width: int = 1) -> None:
        """轴对齐矩形（PIL 同形）：`xy` = `[x0, y0, x1, y1]`。"""
        box = self._box(xy)
        if box is None:
            self._skip("rectangle_box")
            return
        x0, y0, x1, y1 = box
        self._emit("rect", ' x="%s" y="%s" width="%s" height="%s"' % (
            x0, y0, _num(float(x1) - float(x0)) or "0",
            _num(float(y1) - float(y0)) or "0") + self._fill_stroke(fill, outline, width))

    def ellipse(self, xy: Sequence, fill=None, outline=None, width: int = 1) -> None:
        """内切于给定矩形的椭圆（PIL 同形）：`xy` = `[x0, y0, x1, y1]`。

        零半径（退化）⇒ 跳过并计数；否则 `rx <= 0` 的 `<ellipse>` 在渲染器里
        要么不显示要么报错，都可能变成"静默看不见的东西"。
        """
        box = self._box(xy)
        if box is None:
            self._skip("ellipse_box")
            return
        x0, y0, x1, y1 = box
        rx = (float(x1) - float(x0)) / 2.0
        ry = (float(y1) - float(y0)) / 2.0
        if rx <= 0 or ry <= 0:
            self._skip("ellipse_degenerate")
            return
        cx = (float(x0) + float(x1)) / 2.0
        cy = (float(y0) + float(y1)) / 2.0
        self._emit("ellipse", ' cx="%s" cy="%s" rx="%s" ry="%s"' % (
            _num(cx), _num(cy), _num(rx), _num(ry)) + self._fill_stroke(fill, outline, width))

    def line(self, xy: Sequence, fill=None, width: int = 1) -> None:
        """折线（PIL 同形）：`xy` = 顶点序列；两点即线段。"""
        pts = self._points(xy)
        if pts is None or len(pts) < 2:
            self._skip("line_pts")
            return
        col = _color(fill)
        if col is None:
            self._skip("line_color")
            return
        c, op, was_clamped = col
        if was_clamped:
            self._clamped += 1
        # 照 PIL 实测：`width=0` ⇒ 什么也不画；`width=-3` ⇒ 按宽 3 画（PIL 取绝对值）。
        w = abs(int(width))
        if w == 0:
            self._skip("line_width")
            return
        self._emit("polyline", ' points="%s" fill="none" stroke="%s" stroke-width="%d"%s'
                   % (" ".join("%s,%s" % p for p in pts), c, w,
                      '' if op is None else ' stroke-opacity="%s"' % op))

    # ------------------------------------------------------------ 输出
    def to_svg(self) -> str:
        """导出 SVG 文档（确定性：同样调用序列 ⇒ 逐字节相同）。"""
        bg = self._bg
        if self._group_stack:
            raise ValueError("to_svg() 时仍有未闭合的图层：%s"
                             % ", ".join(self._group_stack))
        if self._marks:
            body = self._body_with_groups()
        else:
            body = "\n".join("    " + e for e in self._elements)
        head = ('<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
                'width="%d" height="%d" viewBox="0 0 %d %d">'
                % (self.width, self.height, self.width, self.height))
        # 结尾的 `\n` 不能少：少了它背景矩形会与**首个元素同行**（实测被同伴的按行
        # 解析踩到过），且排版随实现漂移。它同时是"背景底"与绘制元素的天然分隔。
        rect = ('  <rect x="0" y="0" width="%d" height="%d" fill="%s"%s/>\n'
                % (self.width, self.height, bg[0],
                   '' if bg[1] is None else ' fill-opacity="%s"' % bg[1])
                ) if bg is not None else ""
        return "%s\n%s%s\n</svg>\n" % (head, rect, body)

    def _body_with_groups(self) -> str:
        """按 mark 把元素嵌进 `<g>`（仅在用过 `begin_group()` 时走这条）。

        缩进口径：顶层 `<g>` 缩进 2 格（与既有的"元素 4 格"对齐 ⇒ 层内元素仍是 4 格），
        每多嵌一层加 2 格。

        ★ **必须按 mark 的插入序处理，不能按 (index, close 优先) 排序**——第一版实测踩到的
        缺陷：空图层（`begin_group` 与 `end_group` 之间一个元素都没发射，例如相机在物体
        内部 ⇒ 该物体没有可见面）会在同一 index 上产生 `open(A) open(B) close(B) close(A)`，
        按"close 优先"排序后变成 `close(B) close(A) open(A) open(B)` ⇒ **`</g>` 跑到 `<g>`
        前面**，ElementTree 报 `mismatched tag`。而按插入序处理时，调用顺序本来就是文档顺序：
        同 index 上的 `close(上一个兄弟)` 天然排在 `open(下一个兄弟)` 之前（先调用），
        空图层也天然自洽 ⇒ 不需要任何排序。

        ★ **`index == len(_elements)`（文档末尾的 mark，含"空文档里的整段图层"）必须走
        同一套 open/close 逻辑**——第二版把尾部只当"补 close"处理，于是
        `begin_group()`+`end_group()` 但**一个元素都没发射**（`len(_elements) == 0`）
        的文档里两个 mark 都落在尾部、头一个还是 `open` ⇒ 被误判成"未闭合"而抛错。
        现在改成先把 idx 走完 `0..len(elements)` 再统一收尾。
        """
        marks = self._marks                     # 插入序；index 单调不减
        lines: List[str] = []
        mi = 0
        depth = 0
        n = len(self._elements)
        for idx in range(n + 1):                # ★ 含末尾那一档
            while mi < len(marks) and marks[mi][0] == idx:
                _i, kind, payload = marks[mi]
                if kind == "open":
                    lines.append("  " + "  " * depth + "<g %s>" % payload)
                    depth += 1
                else:
                    depth -= 1
                    if depth < 0:
                        raise ValueError("图层闭合次数多于开启次数（文档结构被破坏）")
                    lines.append("  " + "  " * depth + "</g>")
                mi += 1
            if idx < n:
                lines.append("  " + "  " * depth + self._elements[idx])
        if mi != len(marks):
            raise ValueError("有 mark 的 index 超过元素数（内部状态损坏）")
        if depth != 0:
            raise ValueError("to_svg() 时仍有未闭合的图层（depth=%d）" % depth)
        return "\n".join(lines)

    def stats(self) -> Dict:
        """观测面：发射了几个元素、跳过了哪些（跳过必须能看见）。"""
        return {"emitted": self._emitted, "skipped": dict(self._skipped),
                "clamped": self._clamped, "elements": len(self._elements),
                "width": self.width, "height": self.height,
                # 只增字段（既有调用方按 key 取用，不受影响）：
                "groups": len(self._group_ids), "group_ids": list(self._group_ids)}

    def __len__(self) -> int:
        return len(self._elements)

    # ------------------------------------------------------------ 内部
    def _emit(self, tag: str, attrs: str) -> None:
        self._elements.append("<%s%s/>" % (tag, attrs))
        self._emitted += 1

    def _skip(self, why: str) -> None:
        self._skipped[why] = self._skipped.get(why, 0) + 1

    def _fill_stroke(self, fill, outline, width: int = 1) -> str:
        out = ""
        col = _color(fill)
        if col is not None:
            if col[2]:
                self._clamped += 1        # 越界分量照 PIL 钳位，但要能看见
            out += ' fill="%s"' % col[0]
            if col[1] is not None:
                out += ' fill-opacity="%s"' % col[1]
        else:
            # `fill=None` 是合法的"不填充"；**颜色形状不合法**则要能看见
            if fill is not None:
                self._skip("fill_color")
            out += ' fill="none"'
        st = _color(outline)
        if st is not None:
            if st[2]:
                self._clamped += 1
            out += ' stroke="%s" stroke-width="%d"' % (st[0], max(1, int(width)))
            if st[1] is not None:
                out += ' stroke-opacity="%s"' % st[1]
        elif outline is not None:
            self._skip("outline_color")
        return out

    @staticmethod
    def _points(xy) -> Optional[List[Tuple[str, str]]]:
        """顶点序列 → 已格式化的 `(x, y)` 文本对；非有限 ⇒ None。"""
        if xy is None:
            return None
        try:
            seq = list(xy)
        except TypeError:
            return None
        if not seq:
            return None
        # 扁平 [x0, y0, x1, y1, ...] 与 [(x, y), ...] 都接受
        if not isinstance(seq[0], (tuple, list)):
            if len(seq) % 2:
                return None
            seq = [(seq[i], seq[i + 1]) for i in range(0, len(seq), 2)]
        pts = []
        for p in seq:
            if not isinstance(p, (tuple, list)) or len(p) < 2:
                return None
            xs, ys = _num(p[0]), _num(p[1])
            if xs is None or ys is None:
                return None
            pts.append((xs, ys))
        return pts

    @staticmethod
    def _box(xy) -> Optional[Tuple[str, str, str, str]]:
        """`[x0, y0, x1, y1]` → 已格式化四元组；非有限 ⇒ None。"""
        if xy is None:
            return None
        try:
            seq = list(xy)
        except TypeError:
            return None
        if len(seq) == 4 and not isinstance(seq[0], (tuple, list)):
            vals = [_num(v) for v in seq]
            if any(v is None for v in vals):
                return None
            if float(seq[2]) < float(seq[0]) or float(seq[3]) < float(seq[1]):
                # 照抄 PIL：`ImageDraw.rectangle/ellipse` 对反序框抛
                # `ValueError: x1 must be greater than or equal to x0`（实测）。
                # 早先的实现会发射 **负 width** 的 `<rect>` —— SVG 1.1 里是 error，
                # 渲染器直接不画，于是与位图分道（且不计数）。
                raise ValueError("x1 must be greater than or equal to x0")
            return (vals[0], vals[1], vals[2], vals[3])
        # 也接受 [(x0,y0),(x1,y1)] 形式
        if len(seq) == 2 and all(isinstance(p, (tuple, list)) and len(p) >= 2 for p in seq):
            vals = [_num(seq[0][0]), _num(seq[0][1]), _num(seq[1][0]), _num(seq[1][1])]
            if any(v is None for v in vals):
                return None
            return (vals[0], vals[1], vals[2], vals[3])
        return None
