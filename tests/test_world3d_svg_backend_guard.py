# -*- coding: utf-8 -*-
"""World3D 接入 SVG 矢量后端 —— 守卫（缺陷 / 修法 / 判据 / 断言组 / 运行）

缺陷（本件 · 呈现层只能出位图）：
  `World3D.render()` 把「地面矩形 + 画家算法远→近的物体」的绘制**与 PIL 绑死**：
  入口里直接 `Image.new` / `ImageDraw.Draw`，绘图面没有第二个出口，于是
  ① 放大即糊（1600px 与 400px 是两次不同的绘制）；② 边缘全是台阶；③ 产物不可寻址。
  另一面：**没有任何测试**能证明「换一个同形后背端，绘制序列一字不差」——
  既有守卫只钉像素/顺序（#311/#312/#313/#403），钉不住"出口可替换"这件事本身。

修法（只搬出口 + 加矢量出口，不改绘制顺序）：
  ① 抽出渲染内核 `World3D._paint(backend, cam, w, h, ground_color)`：只经
     `draw.rectangle / draw.polygon / draw.ellipse` 三个与 `PIL.ImageDraw` **同形**的原语；
  ② `render()` 走 `ImageDraw`（绘制参数、调用次数、顺序**逐字节不变**）；
  ③ 新增 `render_svg(...) -> str` 走 `SVGBackend`（`lingshu/world/svg_backend.py`），
     同参同序，可选 `stats` 出参回吐 `stats()["skipped"]`（跳过必须可见，不静默）；
  ④ `#416` 尺寸上限抽到 `_check_screen_dim`，两条出口**同判据**。

判据来源：
  · 本件冻结契约 本件的冻结契约（四原语与 `ImageDraw` 同形）与五条验收；
  · 既有绘图顺序判据（`#311` 可见面剔除 / `#312` 部件深度序 / `#313` 塔尖投影 /
    `#403` 近平面裁剪）与 `#416` 上限判据——本件**不改**这些，只换出口；
  · 绘制面实测：全管线只有 `polygon / rectangle / ellipse` 三原语被 world3d 用到
    （`grep -oE "draw\\.[a-z_]+\\("`；`line` 属于 scene_model，本文件不涉及）；
  · 位图不变的真源：改前基线（修复前上游版本实测）
    （`w3d_12obj_640 / w3d_40obj_800 / w3d_3obj_320`），复验命令
    **跨版本字节相等**须由实跑取证；
    本文件内的 B 组只钉"位图出口没被矢量改造污染"这条**不变量**（可移植、不依赖
    Pillow 版本号，故不内嵌绝对哈希）。

断言组（抽掉修复 ⇒ 哪组红）：
  A 两出口同一条绘制序列（录制后端逐调用比对）＋画家序「远先画」独立校验
  B 位图出口不变：确定性、不触碰矢量后端、`render` == `Image.new+ImageDraw.Draw+_paint`、
    地面矩形参数原样
  C 退化输入不静默：非有限屏幕坐标 ⇒ 该元素不发射且计入 `stats()["skipped"]`，
    正常物体不受牵连
  D 背景底 / 地面矩形 / `viewBox` 与画布尺寸一致（含非正方画布）
  E `MAX_SCREEN_DIM` 在矢量出口同样 fail-closed；矢量出口零位图分配；
    `ElementTree` 可解析且 `元素数 == stats()["emitted"] + 1`（那 1 个是背景底，
    对应位图的 `Image.new` 底色，不是绘制调用）

  回退即红（实测，见报告"回退即红"节；变异在临时副本上做，不动出货树）：
  · M1 删掉 `render_svg`（入口只剩 `_svg_entry_removed`，`AttributeError`）
    ⇒ **14 failed / 4 passed**：A1/A2/A3、C1/C2/C3、D1×2、D2/D3/D4、E1/E2/E3 全红；
    仍绿的是 B1..B4（位图出口，本就不走矢量通道）。
  · M2 `render_svg` 在、但**不调 `_paint`**（空文档）
    ⇒ **8 failed / 10 passed**：A1/A2/A3、C1/C2/C3、D3/D4 红；
    仍绿的是 B1..B4、D1×2、D2、E1/E2/E3。
    注意 E3 在 M2 下**仍绿**：它判的是"文档与 `stats()` 自洽"，空文档也自洽
    （"到底画没画"由 A3/D3 负责）——这是 E3 的判别力边界，如实记录。

运行（仓根 = 含 `lingshu/` 的目录）：
    python -X utf8 -m pytest tests/test_world3d_svg_backend_guard.py -q --no-header
本文件用完整路径跑亦通（见 `_find_repo`）：
    python -X utf8 -m pytest tests/test_world3d_svg_backend_guard.py -q --no-header
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

SVG_NS = "{http://www.w3.org/2000/svg}"


def _find_repo() -> Path:
    """定位含 `lingshu/world/world3d.py` 的仓根（从本文件位置向上找标记文件）。"""
    here = Path(__file__).resolve()
    cands = []
    env = os.environ.get("LINGSHU_REPO")
    if env:
        cands.append(Path(env))
    for p in [here.parent, *here.parents]:
        cands.append(p)
    for c in cands:
        if (c / "lingshu" / "world" / "world3d.py").is_file():
            return c
    raise RuntimeError(
        "找不到 lingshu 仓根（可设环境变量 LINGSHU_REPO）：从 %s 起都试过" % here)


REPO = _find_repo()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lingshu.world import svg_backend as svg_backend_mod          # noqa: E402
from lingshu.world.svg_backend import SVGBackend                   # noqa: E402
from lingshu.world.world3d import (Camera3D, MAX_SCREEN_DIM,       # noqa: E402
                                   Object3D, World3D)

# ---------------------------------------------------------------------------
# 录制后端：把两条出口的「原语调用序列」录下来逐条比对（A 组的尺子）
# ---------------------------------------------------------------------------


def _xy_pts(xy):
    return [(float(p[0]), float(p[1])) for p in xy]


class _RecordingSVGBackend(SVGBackend):
    """`SVGBackend` 的录制子类：原样转发，同时记下 `(kind, xy, fill, outline)`。"""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.calls = []

    def polygon(self, xy, fill=None, outline=None):
        self.calls.append(("polygon", _xy_pts(xy), fill, outline))
        return super().polygon(xy, fill=fill, outline=outline)

    def rectangle(self, xy, fill=None, outline=None, width=1):
        self.calls.append(("rectangle", [float(v) for v in xy], fill, outline))
        return super().rectangle(xy, fill=fill, outline=outline, width=width)

    def ellipse(self, xy, fill=None, outline=None, width=1):
        self.calls.append(("ellipse", [float(v) for v in xy], fill, outline))
        return super().ellipse(xy, fill=fill, outline=outline, width=width)

    def line(self, xy, fill=None, width=1):
        self.calls.append(("line", _xy_pts(xy), fill, None))
        return super().line(xy, fill=fill, width=width)


class _RecordingDraw:
    """`ImageDraw.Draw` 的录制代理：只录四个原语，其余属性转发给真身。"""

    def __init__(self, inner):
        self.inner = inner
        self.calls = []

    def polygon(self, xy, fill=None, outline=None):
        self.calls.append(("polygon", _xy_pts(xy), fill, outline))
        return self.inner.polygon(xy, fill=fill, outline=outline)

    def rectangle(self, xy, fill=None, outline=None, width=1):
        self.calls.append(("rectangle", [float(v) for v in xy], fill, outline))
        return self.inner.rectangle(xy, fill=fill, outline=outline, width=width)

    def ellipse(self, xy, fill=None, outline=None, width=1):
        self.calls.append(("ellipse", [float(v) for v in xy], fill, outline))
        return self.inner.ellipse(xy, fill=fill, outline=outline, width=width)

    def line(self, xy, fill=None, width=1):
        self.calls.append(("line", _xy_pts(xy), fill, None))
        return self.inner.line(xy, fill=fill, width=width)

    def __getattr__(self, name):
        return getattr(self.inner, name)


@contextlib.contextmanager
def _svg_recorder():
    """with 体内 `render_svg` 构造出的后端被换成录制后端；退出即还原。"""
    real = svg_backend_mod.SVGBackend
    box = {}

    def factory(*a, **k):
        rec = _RecordingSVGBackend(*a, **k)
        box["rec"] = rec
        return rec

    svg_backend_mod.SVGBackend = factory
    try:
        yield box
    finally:
        svg_backend_mod.SVGBackend = real


@contextlib.contextmanager
def _raster_recorder():
    """with 体内 `render()` 用的 `ImageDraw.Draw` 被换成录制代理；退出即还原。"""
    from PIL import ImageDraw
    real = ImageDraw.Draw
    box = {}

    def factory(img, mode=None):
        inner = real(img) if mode is None else real(img, mode)
        rec = _RecordingDraw(inner)
        box["rec"] = rec
        return rec

    ImageDraw.Draw = factory
    try:
        yield box
    finally:
        ImageDraw.Draw = real


def _png_bytes(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _tag(node) -> str:
    return node.tag[len(SVG_NS):] if node.tag.startswith(SVG_NS) else node.tag


def _bg_rects(root, w, h):
    """文档首部的「背景底」矩形（= 位图 `Image.new` 的底色，不是绘制调用）。"""
    out = []
    for node in list(root):
        if _tag(node) != "rect":
            continue
        if (node.get("x") == "0" and node.get("y") == "0"
                and node.get("width") == str(w) and node.get("height") == str(h)):
            out.append(node)
    return out


# ---------------------------------------------------------------------------
# 场景
# ---------------------------------------------------------------------------

#: 覆盖「部件形状（tree/person）+ 单图元（box/pyramid/sphere）」的确定场景。
_SCENE_SPECS = (
    ("box", (-2.0, 0.65, 0.0), (0.7, 1.3, 0.7), (200, 80, 60), "box"),
    ("tree", (1.5, 2.0, -1.0), (2.0, 4.0, 2.0), (60, 140, 70), "pillar"),
    ("person", (0.0, 0.85, 2.0), (0.6, 1.7, 0.4), (200, 160, 140), "pillar"),
    ("crown", (-1.0, 1.5, -2.5), (0.3, 0.25, 0.3), (240, 210, 100), "pyramid"),
    ("ball", (2.2, 0.5, 1.5), (0.2, 0.2, 0.2), (240, 120, 60), "sphere"),
)


def _scene(camera=None) -> World3D:
    w = World3D(camera=camera or Camera3D.look_at((6.0, 4.0, 8.0), (0.0, 0.6, 0.0)))
    for cat, center, size, color, shape in _SCENE_SPECS:
        w.objects.append(Object3D(category=cat, center=center, size=size,
                                  color=color, shape=shape))
    return w


def _sphere(center, color=(240, 120, 60)) -> Object3D:
    return Object3D(category="ball", center=center, size=(0.2, 0.2, 0.2),
                    color=color, shape="sphere")


# ---------------------------------------------------------------------------
# A 两出口同一条绘制序列 + 画家序
# ---------------------------------------------------------------------------


def test_a1_两出口绘制调用序列逐条一致():
    """A1：`render()` 与 `render_svg()` 的原语调用序列（序、几何、颜色）逐条相同。

    「把 draw 换成 SVGBackend，绘制代码一行不用改」这句话的**直接**证据。
    """
    sw, sh = 400, 300
    with _raster_recorder() as rbox:
        img = _scene().render(sw, sh)
    with _svg_recorder() as sbox:
        svg = _scene().render_svg(sw, sh)

    raster = rbox["rec"].calls
    vector = sbox["rec"].calls
    assert len(raster) > 5, "A1 录制面太小，尺子没量到东西：%d 条" % len(raster)
    assert img.size == (sw, sh)
    assert svg.startswith("<svg")
    assert raster == vector, (
        "A1 两出口绘制序列不一致：raster=%d 条 / svg=%d 条；首处不同 %r"
        % (len(raster), len(vector), next(
            ((i, a, b) for i, (a, b) in enumerate(zip(raster, vector)) if a != b),
            None)))


def test_a2_画家序远先画():
    """A2：远物体的原语**先**出现（SVG 文档序 = 绘制序 ⇒ 与位图画家算法同构）。"""
    w = World3D(camera=Camera3D.look_at((0.0, 1.6, 8.0), (0.0, 1.0, 0.0)))
    w.objects.append(_sphere((0.0, 1.0, 1.0), color=(1, 2, 3)))     # 近
    w.objects.append(Object3D(category="moon", center=(0.0, 1.0, -6.0),
                              size=(3.0, 3.0, 0.1), color=(9, 9, 9), shape="disk"))  # 远
    with _svg_recorder() as sbox:
        w.render_svg(300, 200)
    calls = sbox["rec"].calls

    assert calls[0][0] == "rectangle", "A2 首个原语应是地面矩形：%r" % (calls[0],)
    far = next((i for i, c in enumerate(calls) if c[0] == "ellipse" and c[2] == (9, 9, 9)), None)
    near = next((i for i, c in enumerate(calls) if c[0] == "ellipse" and c[2] == (1, 2, 3)), None)
    assert far is not None and near is not None, "A2 两个球没都画出来：%r" % (calls,)
    assert far < near, "A2 画家序反了：远物体 idx=%d 应在近物体 idx=%d 之前" % (far, near)


def test_a3_发射计数等于真实绘制调用数():
    """A3：`stats()["emitted"]` == 录到的调用数 == 后端元素数 == 位图调用数。"""
    sw, sh = 320, 240
    with _raster_recorder() as rbox:
        _scene().render(sw, sh)
    st = {}
    with _svg_recorder() as sbox:
        _scene().render_svg(sw, sh, stats=st)
    rec = sbox["rec"]
    assert st["emitted"] == len(rec.calls) == len(rec) == len(rbox["rec"].calls), (
        "A3 计数不一致：stats=%r calls=%d len=%d raster=%d"
        % (st, len(rec.calls), len(rec), len(rbox["rec"].calls)))
    assert st["emitted"] > 5


# ---------------------------------------------------------------------------
# B 位图出口不变（不被矢量改造污染）
# ---------------------------------------------------------------------------


def test_b1_位图出口确定性且仍是位图():
    """B1：同场景两次 `render()` 的 PNG 逐字节相同；产物仍是 PIL 位图。"""
    a = _png_bytes(_scene().render(200, 150))
    b = _png_bytes(_scene().render(200, 150))
    assert a == b, "B1 位图出口不确定：%s vs %s" % (
        hashlib.sha256(a).hexdigest(), hashlib.sha256(b).hexdigest())
    img = _scene().render(200, 150)
    assert img.mode == "RGB" and img.size == (200, 150)


def test_b2_位图出口不触碰矢量后端(monkeypatch):
    """B2：`render()` 不得构造 `SVGBackend`（位图出口仍只依赖 PIL）。"""
    def _boom(*a, **k):
        raise AssertionError("B2 位图出口不应构造 SVGBackend：%r %r" % (a, k))

    monkeypatch.setattr(svg_backend_mod, "SVGBackend", _boom)
    img = _scene().render(160, 120)
    assert img.size == (160, 120)


def test_b3_render_是内核的薄包装():
    """B3：`render()` 与「Image.new + ImageDraw.Draw + `_paint`」逐字节相同。

    抽内核若搬错参数/少画一笔，这条会红。
    """
    from PIL import Image, ImageDraw

    sw, sh, bg, gc = 200, 150, (20, 24, 40), (30, 34, 50)
    got = _png_bytes(_scene().render(sw, sh, background=bg, ground_color=gc))

    w = _scene()
    manual = Image.new("RGB", (sw, sh), bg)
    w._paint(ImageDraw.Draw(manual), w.camera, sw, sh, gc)
    assert got == _png_bytes(manual), "B3 render() 不是 _paint 的薄包装（像素有差）"


def test_b4_地面矩形参数原样():
    """B4：第一个绘制调用是地面矩形，参数与地平线口径一致（int(0.45*h)）。"""
    sw, sh, gc = 200, 150, (30, 34, 50)
    with _raster_recorder() as rbox:
        _scene().render(sw, sh, ground_color=gc)
    first = rbox["rec"].calls[0]
    assert first == ("rectangle", [0.0, float(int(sh * 0.45)), float(sw), float(sh)], gc, None), (
        "B4 地面矩形参数变了：%r" % (first,))


# ---------------------------------------------------------------------------
# C 退化输入：不发射、但必须计数（禁止静默）
# ---------------------------------------------------------------------------


def test_c1_非有限屏幕坐标的球被跳过并计数():
    """C1：NaN 屏幕点的球 ⇒ 两个椭圆都不发射，且 `skipped["ellipse_box"] == 2`。"""
    cam = Camera3D.look_at((6.0, 4.0, 8.0), (0.0, 0.6, 0.0))
    w = World3D(camera=cam)
    w.objects.append(_sphere((float("nan"), 0.5, 5.0)))
    st = {}
    svg = w.render_svg(400, 300, stats=st)

    assert st["emitted"] == 1, "C1 只应发射地面矩形：%r" % (st,)
    assert st["skipped"] == {"ellipse_box": 2}, "C1 退化椭圆未被计数：%r" % (st,)
    ET.fromstring(svg)                     # 仍然良构（不发射非法元素）
    w.render(400, 300)                     # 位图出口同样不抛


def test_c2_非有限多边形被跳过并计数():
    """C2：非有限顶点的盒（`polygon` 路径）⇒ `skipped["polygon_pts"] > 0`。"""
    w = World3D()                          # 默认相机
    w.objects.append(Object3D(category="box", center=(0.0, 0.65, float("inf")),
                              size=(0.7, 1.3, 0.7), color=(200, 80, 60)))
    st = {}
    svg = w.render_svg(400, 300, stats=st)

    assert st["skipped"].get("polygon_pts", 0) > 0, "C2 退化多边形未被计数：%r" % (st,)
    assert st["emitted"] == 1, "C2 退化盒不该发射任何多边形：%r" % (st,)
    ET.fromstring(svg)
    w.render(400, 300)


def test_c3_退化不误伤正常物体():
    """C3：退化物体被跳过时，正常物体照常发射（skipped 非空**不是**失败）。"""
    cam = Camera3D.look_at((6.0, 4.0, 8.0), (0.0, 0.6, 0.0))
    w = World3D(camera=cam)
    w.objects.append(_sphere((2.2, 0.5, 1.5)))                    # 正常
    w.objects.append(_sphere((float("nan"), 0.5, 5.0), color=(9, 9, 9)))  # 退化
    st = {}
    w.render_svg(400, 300, stats=st)

    assert st["emitted"] == 3, "C3 正常球应照常发射（地面+2 椭圆）：%r" % (st,)
    assert st["skipped"] == {"ellipse_box": 2}, "C3 跳过计数不对：%r" % (st,)


# ---------------------------------------------------------------------------
# D 背景底 / 地面 / viewBox
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sw,sh", [(400, 300), (320, 240)])
def test_d1_viewBox与画布尺寸一致(sw, sh):
    """D1：`width`/`height`/`viewBox` 与请求的画布尺寸一致（含非正方画布）。"""
    root = ET.fromstring(_scene().render_svg(sw, sh))
    assert _tag(root) == "svg", "D1 根元素不是 svg：%r" % (root.tag,)
    assert root.get("width") == str(sw) and root.get("height") == str(sh)
    assert root.get("viewBox") == "0 0 %d %d" % (sw, sh), root.get("viewBox")


def test_d2_背景底铺满画布():
    """D2：首个子元素是铺满画布的背景底，颜色为 `background`。"""
    sw, sh, bg = 400, 300, (1, 2, 3)
    root = ET.fromstring(_scene().render_svg(sw, sh, background=bg))
    kids = list(root)
    bg_rects = _bg_rects(root, sw, sh)
    assert len(bg_rects) == 1, "D2 背景底缺失或重复：%d 个" % len(bg_rects)
    assert kids[0] is bg_rects[0], "D2 背景底不在首位（画家序会变）"
    assert bg_rects[0].get("fill") == "rgb(1,2,3)", bg_rects[0].get("fill")


def test_d3_地面矩形与地平线一致():
    """D3：第二个子元素是地面矩形，y=int(0.45*h)、高 h-y、颜色为 `ground_color`。"""
    sw, sh, gc = 400, 300, (4, 5, 6)
    root = ET.fromstring(_scene().render_svg(sw, sh, ground_color=gc))
    ground = list(root)[1]
    y = int(sh * 0.45)
    assert _tag(ground) == "rect", "D3 地面不是矩形：%r" % (_tag(ground),)
    assert ground.get("x") == "0" and ground.get("y") == str(y), ground.attrib
    assert ground.get("width") == str(sw) and ground.get("height") == str(sh - y), ground.attrib
    assert ground.get("fill") == "rgb(4,5,6)", ground.get("fill")


def test_d4_背景与地面可换色且如实铺底():
    """D4：非默认配色必须原样落到 SVG（防写死默认色）。"""
    root = ET.fromstring(_scene().render_svg(
        200, 160, background=(7, 8, 9), ground_color=(10, 11, 12)))
    kids = list(root)
    assert kids[0].get("fill") == "rgb(7,8,9)"
    assert kids[1].get("fill") == "rgb(10,11,12)"


# ---------------------------------------------------------------------------
# E 上限、零位图分配、XML 良构与元素数
# ---------------------------------------------------------------------------


def test_e1_尺寸上限在矢量出口同样生效():
    """E1：`MAX_SCREEN_DIM` 超出 ⇒ ValueError 点名 #416；边界值放行（两出口同判据）。"""
    w = World3D()
    with pytest.raises(ValueError) as ei:
        w.render_svg(MAX_SCREEN_DIM + 1, 10)
    assert "#416" in str(ei.value), "E1 拒绝理由未点名 #416：%r" % (str(ei.value),)
    with pytest.raises(ValueError):
        w.render_svg(10, MAX_SCREEN_DIM + 1)
    assert w.render_svg(MAX_SCREEN_DIM, 1).startswith("<svg")     # 边界放行（矢量不分配 w*h）


def test_e2_矢量出口零位图分配(monkeypatch):
    """E2：`render_svg` 全程不分配位图（`PIL.Image.new` 一次都不许被调用）。"""
    def _boom(*a, **k):
        raise AssertionError("E2 矢量出口分配了位图：%r" % (a,))

    monkeypatch.setattr("PIL.Image.new", _boom)
    svg = _scene().render_svg(320, 240)
    assert svg.startswith("<svg") and svg.endswith("</svg>\n")


def test_e3_XML良构且元素数等于emitted加背景底():
    """E3：`ElementTree` 可解析；`元素数 == emitted + 1`（+1 = 背景底，非绘制调用）。"""
    sw, sh = 400, 300
    st = {}
    svg = _scene().render_svg(sw, sh, stats=st)
    root = ET.fromstring(svg)
    kids = list(root)
    bg = _bg_rects(root, sw, sh)

    assert len(bg) == 1, "E3 背景底数异常：%d" % len(bg)
    assert len(kids) - len(bg) == st["emitted"], (
        "E3 元素数与 emitted 不一致：元素 %d - 背景底 %d != emitted %d"
        % (len(kids), len(bg), st["emitted"]))
    assert len(kids) == st["elements"] + 1, (
        "E3 元素数 != backend.elements + 背景底：%d vs %d"
        % (len(kids), st["elements"]))
    assert {_tag(n) for n in kids} <= {"rect", "polygon", "ellipse", "polyline"}, (
        "E3 出现了非契约元素：%r" % ({_tag(n) for n in kids},))
