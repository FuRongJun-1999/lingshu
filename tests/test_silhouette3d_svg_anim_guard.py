# -*- coding: utf-8 -*-
"""守卫 · SVG 矢量出口 + SVG 动画 · lingshu/world/silhouette3d.py（WS-2）
============================================================================
缺陷 / 能力缺口（取证 = 修复前的上游 `lingshu/world/
silhouette3d.py`，md5 `197368ba796e78fce04673546fa9a0b4`）：
  旧 `render`（`:145-200`）把「投影 + 画家序排序」与「画多边形」和
  `PIL.ImageDraw` 焊死在一个方法里（`from PIL import Image, ImageDraw` +
  `draw.polygon(...)` 两处）⇒
  ① **没有矢量出口**：轮廓只能出位图，放大即糊、斜边是台阶；
  ② **没法动**：一张 PNG 内部不可寻址——`_set_tail_wave(s, wave)` 已能驱动
     尾巴关节（`:817-821`），但每次产出都只是一张互不相关的静态位图，
     没有"同一张画里的尾巴在摆"这种产物。

修法（只抽一层内核 + 加出口，**不改绘制顺序、不改位图输出**）：
  ① `_view(...)`：投影 + 画家序（远先画）→ `[(part, 屏幕多边形, dz), …]`；
     `_paint(draw, view, alpha)`：内核，只调**与 `ImageDraw` 同形**的
     `draw.polygon(poly, fill=…, outline=…)`。
     · `render()`      = `Image.new` + `ImageDraw.Draw` + `_paint`（既有出口）
     · `render_svg()`  = `SVGBackend` + `_paint` + `to_svg()`（新矢量出口）
     ⇒ 两出口共用同一内核，「绘制顺序走偏」在结构上不可能发生。
  ② `animate_svg(frames=…, screen_w=…, screen_h=…)`：帧间只改尾巴波幅
     （复用既有 `_set_tail_wave`，**不引入新随机源**），逐帧渲染进
     `SVGBackend`，再把每个部件的 `points` 串成一条
     `<animate attributeName="points" values="f0;f1;…"/>`。
     **选 SMIL（而非 CSS keyframes）**：本件要动的是**逐顶点几何**（摆尾）；
     SVG 1.1 里 `points` 是 `<animate>` 可驱动、且顶点数相同时可线性插值的
     属性；CSS keyframes 不能可移植地驱动 `points`（它不是 CSS 属性），只能
     把部件包 `<g>` 做 `transform` 级近似——摆尾会退化成整体平移，几何是错的。
     代价（如实）：只认静态首帧的**老式 SVG 精简渲染器**看不到动画。

判据来源：冻结契约 本件冻结契约（4 原语与 `ImageDraw` 同形 / 画家序天然成立 /
确定性 / 跳过必须可见）+ §3.2（**既有位图输出逐字节不变**）+ §4（验收 ①–⑤）+
task-2 描述里的断言组 A–E。SVG 本体的元素级/坐标级判据依据
`world/svg_backend.py` 冻结契约（`points`/`fill`/`fill-opacity` 的属性形态）。

断言组（把修复抽掉 ⇒ 哪组红；实跑读数见提交说明）：
  A [同源]  同场景 raster 出口的 `polygon` 调用序列 == svg 出口的元素序列
            （多边形顶点逐值、顺序一致；fill 颜色一致）；元素数 ==
            `stats()["emitted"]`；SVG 文档可被 `xml.etree.ElementTree` 解析。
            ⇒ 把 `render_svg` 改成另写一套投影（不调 `_paint`/`_view`）或换排序口径：A 红。
  B [位图]  `render()` 的 PNG file-sha256 与像素 tobytes-sha256 恒等于**改前**
            基线（5 组：两模板 × alpha × 默认/侧视相机）。
            ⇒ 改 `_paint` 的 polygon 参数/绘制顺序：B 红（A 亦红）。
  C [动画]  `values` 以 `;` 切分帧数 == 请求帧数；`dur == period`；
            loop=True ⇒ `repeatCount="indefinite"`；loop=False ⇒ `fill="freeze"`
            且无 repeatCount；同参数两次导出**逐字节相同**；只有尾巴各节在动。
            ⇒ 去掉 `<animate>` 注入（退回静态帧 0）：C 红。
  D [alpha] `alpha=True` ⇒ 背景矩形带 `fill-opacity="0"`（RGBA 不被静默丢弃），
            `alpha=False` 不带。
            ⇒ `render_svg_backend` 忽略 alpha（背景恒不透明）：D 红。
  E [退化]  frames=1 ⇒ 静态 SVG（无 `<animate>`、`reason=="single_frame"`，
            且与 `render_svg` 逐字节相同）；无尾部件 ⇒ `reason=="no_motion"`，
            **不伪造**动画；`frames=0` ⇒ `ValueError`；`animate_svg` 结束后
            关节角与位图出口读数复原（不在调用方对象上留姿态）。
            ⇒ 单帧仍发 `<animate>`、或把无位移的帧也写成动画：E 红。

覆盖边界（如实，未验的部分）：
  · 环境里**没有任何 SVG 渲染器**（实测 `cairosvg`/`svglib`/`rsvg`/`cairo` 全
    ModuleNotFoundError，无 inkscape/imagemagick/chromium）⇒ 本守卫**不做**
    逐像素比对，全部为**元素级/坐标级**断言 + 位图侧的字节级读数（B 组）。
    矢量画面与位图画面的一致性由独立复核（svg-verifier 的自写小光栅器）负责。
  · 帧间可见部件集合变化（部件被相机近平面裁掉）时 `animate_svg` 走
    `element_count_varies`/`part_set_varies` 降级路径——本守卫只覆盖
    `single_frame` 与 `no_motion` 两条降级，裁切那条未验（如实声明）。

变异自证 · 实跑读数（变异脚本：备份 → 改 → 跑 → 还原，
还原后 md5 `57d6d4f40a909b7904beeb363ba178cd` 校验通过）：
  · 未突变对照                 24 passed；
  · M1 `render_svg` 不共用内核（改按固定 `part.depth` 另排序）⇒ 5 failed：
       test_a_raster_and_svg_share_same_view_and_order[4 个参数组] +
       test_e_single_frame_degrades_to_static_svg；
  · M2 `_paint` 改成正序（近先画）⇒ 5 failed：B 组 5 个参数组全红（位图变了）；
  · M3 `animate_svg` 不发 `<animate>`（退回静态帧 0）⇒ 8 failed：C 组 5 个帧数
       参数组 + test_c_timeline_and_loop_semantics +
       test_c_animation_is_deterministic_and_only_tails_move +
       test_e_wave_amplitude_is_frame_parameter_not_random；
  · M4 `render_svg_backend` 忽略 alpha ⇒ 1 failed：D 组；
  · M5 整件回退到上游版（无 `render_svg`/`render_svg_backend`/`animate_svg`）
       ⇒ 19 failed / 5 passed —— **只有 B 组 5 个参数组存活**（位图基线本就独立
       于本件修复），A/C/D/E 组全红。
  五组互不替代：M2 时 A 组仍绿而 B 组红；M5 时 B 组绿而其余全红。

  另记一条**真实缺陷**的回归读数（本守卫当场抓到，修的是实现不是测试）：
  冻结后端 `to_svg()` 把背景 `<rect>` 与**首个** `<polygon>` 排在同一行时，旧实现
  按"4 空格开头"切元素 ⇒ 漏掉画家序第一个多边形（一个尾节）⇒ `animated_parts=3`
  但几何侧 `moved=4`，该部件在动画里会**卡住**。改为扫描 `<polygon …/>` 自闭合
  元素（与排版无关）后：修前 `info=3 anims=3 moved=4` → 修后
  `info=4 anims=4 moved=4`（`_svg_polygons` 的注释里记了这段判据）。

运行（仓库根）：
    python -X utf8 -m pytest tests/test_silhouette3d_svg_anim_guard.py -q --no-header
（本文件被移入 `lingshu/tests/` 后同一命令为：
    python -X utf8 -m pytest tests/test_silhouette3d_svg_anim_guard.py -q --no-header
  —— 根目录探测兼容两种位置。）
"""
from __future__ import annotations

import hashlib
import io
import os
import sys
import xml.etree.ElementTree as ET

import pytest

SVG_NS = "{http://www.w3.org/2000/svg}"


def _find_fork_root() -> str:
    """向上找含 `lingshu/world/silhouette3d.py` 的**仓根**。"""
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        for cand in (d,):
            if os.path.isfile(os.path.join(cand, "lingshu", "world", "silhouette3d.py")):
                return cand
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError("找不到含 lingshu/world/silhouette3d.py 的仓根")
        d = parent


FORK = _find_fork_root()
if FORK not in sys.path:
    sys.path.insert(0, FORK)

from lingshu.world import silhouette3d as S          # noqa: E402
from lingshu.world.world3d import Camera3D             # noqa: E402

CENTER = (0.0, 0.85, 5.0)
SIDE_CAM = Camera3D.look_at(eye=(2.4, 1.0, 1.6), target=CENTER, fov_deg=45)

# ------------------------------------------------------------------ 改前基线
# 取证方式：同一脚本分别从**未改动的上游树**与本工作树
# 导入 `silhouette3d` 各跑一次（见提交说明的原始读数）——12/12 组读数一致。
# pix = `img.tobytes()` 的 sha256（像素面，与 PNG 编码器无关）；
# png = `img.save(PNG)` 的 file-sha256（任务要求的"改前/改后"读数）。
# (maker, w, h, camera, alpha) -> (pix_sha, png_sha)
BASELINE = {
    ("fatfish_skinned", 400, 500, None, False): (
        "c6f5f4b8756fd7318b58dc321cebc5554f4347c4a503e451a9f9ed8c9bc1a80a",
        "d6bc6a10d06d96925a33e5c522197186f8896c70145d012b356229ef44fc9d24"),
    ("fatfish_skinned", 400, 500, None, True): (
        "c8f82ceab8060a434bd12e4c97de1343f0abd16fcf1c9f3b3c319baa542ba3c4",
        "7e3ce0e8ebbdd09b4ae50d545e90086f7fba2e0665bdbd07c64b089c87269982"),
    ("fatfish_skeleton", 400, 500, None, False): (
        "f11264074e713510caa1b9d972747062144db03fb3070f27639dcb26a659eb21",
        "83ae1851b49d8d5bfb064c91551bb2168809f39c1df11224cc6696e58b1bcac0"),
    ("fatfish_skeleton", 400, 500, None, True): (
        "de3825eb782d9a9626148ee7b1c8b61cf54f80f7afd2eecc8d354e6aa2b7be34",
        "ec158c51bd003d2b056d3bbc8379106f1ff006bf1b31aeac687feb97b25b77bc"),
    ("fatfish_skinned", 400, 500, "side", False): (
        "78a7fc3baa350e14937269aa951e85adea86f8ddd9fc765890ef692e5d02a507",
        "ce887f55b829abc02e9a292d652255a32bdfc8f1cfa203824c29203364a787c8"),
}


def _cam(name):
    return SIDE_CAM if name == "side" else None


def _num3(v) -> str:
    """按冻结后端的 `SVG_PRECISION = 3` 规则格式化坐标（去尾零）。

    光栅出口给的是**全精度 float**，矢量出口按契约格式化成 3 位小数 ⇒ 坐标级
    比对必须过同一出口格式，否则比较的是"精度"而不是"几何"。
    """
    s = "%.3f" % float(v)
    return s.rstrip("0").rstrip(".") if "." in s else s


def _png_sha(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()


def _polys(svg: str):
    """解析 SVG → [(顶点列表, fill, stroke), …]（文档序）。"""
    root = ET.fromstring(svg)
    out = []
    for el in root:
        if el.tag == SVG_NS + "polygon":
            pts = [tuple(float(v) for v in pair.split(","))
                   for pair in el.get("points").split()]
            out.append((pts, el.get("fill"), el.get("stroke")))
    return out


def _raster_polygon_calls(s, w, h, cam=None, alpha=False):
    """拦截 `PIL.ImageDraw.ImageDraw.polygon`，记录 raster 出口的真实调用序列。"""
    import PIL.ImageDraw as ID

    orig = ID.ImageDraw.polygon
    calls = []

    def spy(self, xy, fill=None, outline=None, width=1):
        calls.append(([tuple(float(v) for v in p) for p in xy], fill, outline))
        return orig(self, xy, fill=fill, outline=outline, width=width)

    ID.ImageDraw.polygon = spy
    try:
        s.render(w, h, camera=cam, alpha=alpha)
    finally:
        ID.ImageDraw.polygon = orig
    return calls


# ===========================================================================
# A 组：两出口同源（同一内核 ⇒ 同几何、同顺序、同色）
# ===========================================================================
@pytest.mark.parametrize("maker,cam_name,alpha", [
    ("fatfish_skinned", "default", False),
    ("fatfish_skinned", "side", False),
    ("fatfish_skeleton", "default", False),
    ("fatfish_skinned", "default", True),
])
def test_a_raster_and_svg_share_same_view_and_order(maker, cam_name, alpha):
    """raster 出口的 polygon 调用序列 == svg 出口的多边形元素序列（逐顶点比对）。

    缺陷态对照：`render_svg` 若不调 `_paint`（自写一套投影/排序），顶点或顺序
    就会与 raster 不一致 ⇒ 本组红。
    """
    cam = _cam(cam_name)
    w, h = 300, 400
    calls = _raster_polygon_calls(getattr(S, maker)(), w, h, cam=cam, alpha=alpha)
    b = getattr(S, maker)().render_svg_backend(w, h, camera=cam, alpha=alpha)
    polys = _polys(b.to_svg())

    assert len(polys) == len(calls) == b.stats()["emitted"], (
        "两出口元素数应相同：svg=%d raster=%d stats=%r"
        % (len(polys), len(calls), b.stats()))
    for i, ((pts_svg, _f, _s), (pts_ras, _f2, _o2)) in enumerate(zip(polys, calls)):
        assert [(_num3(x), _num3(y)) for x, y in pts_svg] == \
               [(_num3(x), _num3(y)) for x, y in pts_ras], \
               "第 %d 个多边形顶点不一致（顺序/口径走偏）" % i
    # 颜色同源：svg 的 fill 必须等于按冻结后端规则格式化的 raster fill
    for i, ((_p, fill_svg, _s), (_p2, fill_ras, _o2)) in enumerate(zip(polys, calls)):
        if alpha:
            assert fill_svg is not None and fill_svg.startswith("rgb("), \
                "第 %d 个元素应有不透明填充，实测 %r" % (i, fill_svg)
            continue
        expect = "rgb(%d,%d,%d)" % tuple(int(c) for c in fill_ras)
        assert fill_svg == expect, "第 %d 个元素 fill 不一致：%r != %r" % (i, fill_svg, expect)


def test_a_painter_order_is_far_first_and_svg_parses():
    """画家序仍是「远先画」（dz 降序），且 SVG 可被 ElementTree 解析。

    缺陷态对照：把排序键改成正序（近先画）⇒ dz 单调性断言红 + A 组顶点序错位。
    """
    s = S.fatfish_skinned()
    view = s._view(300, 400, None)
    dzs = [dz for _p, _poly, dz in view]
    assert dzs == sorted(dzs, reverse=True), "画家序必须按 dz 降序（远先画）"
    doc = s.render_svg(300, 400)
    root = ET.fromstring(doc)                 # 不能解析即抛
    assert root.tag == SVG_NS + "svg"
    assert len(_polys(doc)) == s.last_svg_stats["emitted"]
    assert s.last_svg_stats["skipped"] == {}, \
        "默认场景不应有跳过（有则必须能看见，不能吞）"


# ===========================================================================
# B 组：位图出口逐字节不变（改前/改后 基线）
# ===========================================================================
@pytest.mark.parametrize("key", list(BASELINE))
def test_b_raster_output_byte_identical_to_pre_change_baseline(key):
    """`render()` 的像素 sha256 与 PNG file-sha256 恒等于改前基线。

    缺陷态对照：改 `_paint` 的 `draw.polygon(...)` 参数/顺序（例如先画近后画远、
    或改 fill 口径）⇒ 本组立刻红。
    """
    maker, w, h, cam_name, alpha = key
    pix_sha, png_sha = BASELINE[key]
    img = getattr(S, maker)().render(w, h, camera=_cam(cam_name), alpha=alpha)
    got_pix, got_png = hashlib.sha256(img.tobytes()).hexdigest(), _png_sha(img)
    assert got_pix == pix_sha, (
        "%s %dx%d cam=%s alpha=%s 像素变了：%s != %s"
        % (maker, w, h, cam_name, alpha, got_pix, pix_sha))
    assert got_png == png_sha, (
        "%s %dx%d cam=%s alpha=%s PNG 变了：%s != %s"
        % (maker, w, h, cam_name, alpha, got_png, png_sha))


# ===========================================================================
# C 组：动画帧数 / 时间轴 / 循环语义 / 确定性
# ===========================================================================
@pytest.mark.parametrize("frames", [2, 3, 8, 24, 4.0])
def test_c_animation_frame_count_matches_request(frames):
    """`values` 以 `;` 切分帧数 == 请求帧数；每个动画元素都如此（含整数型 float）。"""
    s = S.fatfish_skinned()
    doc = s.animate_svg(frames=frames, screen_w=300, screen_h=400)
    anims = ET.fromstring(doc).findall(".//" + SVG_NS + "animate")
    assert anims, "应有 SMIL <animate> 元素"
    assert s.last_anim_info["frames"] == frames
    assert s.last_anim_info["mode"] == "smil"
    for a in anims:
        assert a.get("attributeName") == "points"
        assert len(a.get("values").split(";")) == frames, \
            "帧数应与请求一致：%d != %d" % (len(a.get("values").split(";")), frames)


def test_c_timeline_and_loop_semantics():
    """`dur == period`；loop=True ⇒ repeatCount=indefinite；False ⇒ fill=freeze。"""
    s = S.fatfish_skinned()
    doc = s.animate_svg(frames=6, screen_w=300, screen_h=400, period=1.5)
    anims = ET.fromstring(doc).findall(".//" + SVG_NS + "animate")
    assert anims and all(a.get("dur") == "1.5s" for a in anims)
    assert all(a.get("repeatCount") == "indefinite" for a in anims)

    s2 = S.fatfish_skinned()
    fin = s2.animate_svg(frames=6, screen_w=300, screen_h=400, loop=False)
    anims2 = ET.fromstring(fin).findall(".//" + SVG_NS + "animate")
    assert anims2 and all(a.get("repeatCount") is None for a in anims2)
    assert all(a.get("fill") == "freeze" for a in anims2)


def test_c_animation_is_deterministic_and_only_tails_move():
    """同参数两次导出逐字节相同；在动的部件正是尾巴各节，且元素数与观测一致。"""
    s = S.fatfish_skinned()
    a = s.animate_svg(frames=12, screen_w=320, screen_h=420)
    b = S.fatfish_skinned().animate_svg(frames=12, screen_w=320, screen_h=420)
    assert a == b, "同参数两次导出必须逐字节相同（确定性）"

    # 几何侧独立判定（不看 SVG 文本）：只有尾巴部件在波幅下发生位移
    rest = {p.name: poly for p, poly, _dz in s._view(320, 420, None)}
    S._set_tail_wave(s, 0.45)                     # 复用模块的既有波形驱动
    swung = {p.name: poly for p, poly, _dz in s._view(320, 420, None)}
    S._set_tail_wave(s, 0.0)
    moved = {n for n in rest if rest[n] != swung[n]}
    assert moved and all(n.startswith("tail") for n in moved), \
        "波幅只应驱动尾巴各节，实测 %r" % (sorted(moved),)

    anims = ET.fromstring(a).findall(".//" + SVG_NS + "animate")
    assert s.last_anim_info["animated_parts"] == len(anims) == len(moved), \
        "动画元素数应与实际位移部件数一致：info=%r anims=%d moved=%d" % (
            s.last_anim_info["animated_parts"], len(anims), len(moved))


# ===========================================================================
# D 组：alpha=True 的 RGBA 在 SVG 里体现为 fill-opacity（不静默丢弃）
# ===========================================================================
def test_d_alpha_background_fill_opacity_is_not_dropped():
    """`alpha=True` ⇒ 背景矩形 `fill-opacity="0"`；`alpha=False` ⇒ 不带。"""
    s = S.fatfish_skinned()
    rgba_doc = s.render_svg(200, 250, alpha=True)
    rgb_doc = s.render_svg(200, 250, alpha=False)
    assert 'fill-opacity="0"' in rgba_doc, "alpha=True 的背景透明度被丢弃了"
    assert "fill-opacity" not in rgb_doc
    root = ET.fromstring(rgba_doc)
    rects = [el for el in root if el.tag == SVG_NS + "rect"]
    assert len(rects) == 1 and rects[0].get("fill-opacity") == "0"
    # 部件本身 alpha=255 ⇒ 不写 fill-opacity（只有背景那一个）
    assert rgba_doc.count("fill-opacity") == 1


# ===========================================================================
# E 组：退化路径不崩且不静默（单帧 / 无位移 / 非法帧数 / 状态复原）
# ===========================================================================
def test_e_single_frame_degrades_to_static_svg():
    """frames=1 ⇒ 静态 SVG：无 <animate>、reason=single_frame、与 render_svg 相同。"""
    s = S.fatfish_skinned()
    one = s.animate_svg(frames=1, screen_w=240, screen_h=300)
    assert s.last_anim_info["mode"] == "static"
    assert s.last_anim_info["reason"] == "single_frame"
    assert "<animate" not in one
    assert one == s.render_svg(240, 300), "单帧应退化为与静态出口逐字节相同"
    ET.fromstring(one)


def test_e_no_motion_is_reported_not_faked():
    """没有尾巴可驱动时**不伪造**动画，而是如实报告 reason=no_motion。"""
    s = S.Silhouette3D(center=(0.0, 1.0, 5.0))
    s.add_part(S.SilhouettePart("only", [(-0.2, -0.2), (0.2, -0.2),
                                         (0.2, 0.2), (-0.2, 0.2)],
                                depth=0.0, color=(255, 0, 0), outline=(0, 0, 0)))
    doc = s.animate_svg(frames=5, screen_w=120, screen_h=120)
    assert s.last_anim_info["mode"] == "static"
    assert s.last_anim_info["reason"] == "no_motion"
    assert "<animate" not in doc
    assert len(_polys(doc)) == 1


@pytest.mark.parametrize("bad", [0, -3])
def test_e_invalid_frame_count_raises(bad):
    """帧数 < 1 ⇒ ValueError（不静默给一张空/假动画）。"""
    with pytest.raises(ValueError):
        S.fatfish_skinned().animate_svg(frames=bad, screen_w=120, screen_h=120)


def test_e_animation_restores_joint_state_and_bitmap():
    """`animate_svg` 结束后关节角与位图读数复原（不留动画姿态）。"""
    s = S.fatfish_skinned()
    before = [p.angle for p in s.parts]
    before_png = _png_sha(s.render(240, 300))
    s.animate_svg(frames=7, screen_w=240, screen_h=300)
    assert [p.angle for p in s.parts] == before, "关节角未复原"
    assert _png_sha(s.render(240, 300)) == before_png, "位图读数被动画调用污染"


def test_e_wave_amplitude_is_frame_parameter_not_random():
    """帧参数来自调用方：显式波幅序列逐帧生效，且两次导出相同（无随机源）。"""
    waves = [0.0, 0.3, -0.3, 0.15]
    a = S.fatfish_skinned().animate_svg(frames=waves, screen_w=200, screen_h=250)
    b = S.fatfish_skinned().animate_svg(frames=waves, screen_w=200, screen_h=250)
    assert a == b
    s = S.fatfish_skinned()
    s.animate_svg(frames=waves, screen_w=200, screen_h=250)
    assert s.last_anim_info["frames"] == len(waves)
    assert [round(w, 12) for w in s.last_anim_info["waves"]] == waves
    anims = ET.fromstring(a).findall(".//" + SVG_NS + "animate")
    assert len(anims[0].get("values").split(";")) == len(waves)
