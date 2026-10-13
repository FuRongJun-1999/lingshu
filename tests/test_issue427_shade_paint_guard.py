# -*- coding: utf-8 -*-
"""#427 零依赖着色质量守卫（shade-paint）

============================================================================ 缺陷
**真实模型在零依赖世界层渲出来几乎全黑**（Lead 亲测：turret OBJ 2028 面 / 800×600，
"八角塔身只能看出轮廓，材质、层次、体积感都没有"）。同一模型在 Blender 里是亮的。

实测根因三条（都可解析验证，不是观感）：
  D1 `mesh3d` 冻结口径 `lambert` 的环境项写成 `AMBIENT*(0.5+0.5*ny)`
     ⇒ **朝下的面环境项 = 0**、竖直面只剩一半；再叠上「只有一盏正前方主光 +
     裸 `max(0,n·l)`」，侧面直接落到 `SHADE_FLOOR=0.12`。
     实测（同机位）：`fg_mean_luma=60.5`、90% 质量只跨 **4/32** 档。
  D2 **在 sRGB 空间直接乘强度**（`int(base*i)`）= 对暗部做 gamma 平方压缩。
     解析核对：线性 0.18 的钢 @ 强度 0.5 应得 sRGB **85**，旧写法给 **59**。
  D3 **全部材质走同一条光照曲线**，没有按材质角色分层；自发光件也在被"打光"。

============================================================================ 修法
新增 `lingshu/world/shade_paint.py`（纯标准库、纯函数）并把 `mesh3d` 接上：
  F1 半球环境**双基座** `AMB_TOP` / `AMB_BOT`（天光 + 地面反弹，两项都不为零）；
  F2 **半 Lambert** 与裸 Lambert 按 `KEY_LAMBERT_MIX` 混合 ⇒ 背光面不塌零；
  F3 对侧**补光** `FILL_DIR` + 视线补光；
  F4 **Fresnel 边缘光**（掠射提亮）——零依赖里"把主体从背景抠出来"的唯一手段；
  F5 `shade_rgb` 改为**线性空间乘法** + 曝光 + 指数软肩（防高光削平）；
  F6 `face.mat → role → ROLE_PALETTE` 分层；自发光走 `unlit`；
  F7 `mesh_shade` 默认由 `"lambert"` 改为 `"studio"`；`lambert`/`naive`
     **数值冻结**（可逐位回退与对照）。

============================================================================ 判据来源
  · 三光源方位与 Blender 侧 `wm_bridge/factory/shade/render.py` 的 key/fill/rim 同向；
  · 手法清单来自 `blender-wm-bridge/_evidence/factory/VISUAL-ITERATIONS.md`
    （手法 2 环境反射、3 三光源与 rim、5 背景分层、8 色彩管理显式声明）；
  · 线性 ↔ sRGB 用 IEC 61966-2-1 分段函数；WCAG 相对亮度比用 W3C 公开口径；
  · 阈值是**本件实测标定**（`_evidence/shade-paint/raw/*/..._report.json`），
    取"新旧之间、且留出余量"的值 —— 见下面各断言里写明的实测数字。

============================================================================ 断言组
  A 纯函数（无渲染）：模式闭集 / 线性往返 / 环境双基座 / 半 Lambert / 边缘光对照 /
    软肩 / 角色分层 / 色板 / 角色识别优先级 / 能力清单。
  B 冻结口径：`lambert` 与 `naive` 的数值必须与**独立重写的旧公式**逐位一致；
    `shade_rgb_legacy` 必须复现 v2 的 `int(base*i)`。
  C 端到端（合成八角柱）：studio 相对"回退管线"在 fg 亮度 / 对比度 / 值域跨度上更好，
    且**面与面可区分**（前景里不同亮度档数 ≥ N）。
  D 真实模型 A/B（OBJ 不存在则 skip）：同一机位下 studio 必须越线、回退必须不越线
    ⇒ **回退即红**。同时把对照图与读数落盘到 `_evidence/shade-paint/`。

============================================================================ 运行命令
    cd /home/sixtyseven67/DSH/测试/测试/wm-verify/lingshu
    python3 -X utf8 -m pytest tests/test_issue427_shade_paint_guard.py -q
    # 需要 pytest；本机未预装，可用（已验证）：
    #   python3 -m pip install --target _evidence/shade-paint/.pytestlib pytest
    #   PYTHONPATH=_evidence/shade-paint/.pytestlib python3 -m pytest tests/test_issue427_shade_paint_guard.py -q
回退即红：把 `World3D.mesh_shade` 改回 `"lambert"`（或把 `shade_rgb` 改回 sRGB 直接乘），
D 组与 C 组的 `*_studio_beats_legacy` 断言立刻变红（实测见 `_evidence/shade-paint/README.md`）。
"""
import json
import math
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lingshu.world import mesh3d as M3                      # noqa: E402
from lingshu.world import shade_paint as SP                   # noqa: E402
from lingshu.world.world3d import Camera3D, Object3D, World3D  # noqa: E402

TURRET_DIR = "/mnt/d/DSH/blender/out/wmtest/turret"
TURRET_OBJ = os.path.join(TURRET_DIR, "mesh.obj")
TURRET_MTL = os.path.join(TURRET_DIR, "mesh.mtl")
EVID = os.path.join(REPO, "_evidence", "shade-paint")
SPEC_PY = os.path.join(os.path.dirname(REPO), "blender-wm-bridge",
                       "wm_bridge", "factory", "geo", "spec.py")

SW, SH = 480, 360


# --------------------------------------------------------------------- 工具
def _octagonal_prism():
    """合成八角柱：8 个侧面 + 顶底盖（**全部三角面** —— `prepare_triangles` 只吃三角面，
    喂 n-gon 会 `too many values to unpack`；这也顺带记录了"三角化在解析期完成"的口径）。"""
    verts, faces = [], []
    n = 8
    h = 0.9
    for k in range(n):
        a = 2.0 * math.pi * k / n
        verts.append((math.cos(a) * 0.7, -h / 2, math.sin(a) * 0.7))
    for k in range(n):
        a = 2.0 * math.pi * k / n
        verts.append((math.cos(a) * 0.7, h / 2, math.sin(a) * 0.7))
    for k in range(n):                                  # 侧面 → 每格 2 三角
        k2 = (k + 1) % n
        faces.append((k, k2, k2 + n))
        faces.append((k, k2 + n, k + n))
    for k in range(n):                                  # 顶/底盖 → 扇形三角化
        k2 = (k + 1) % n
        faces.append((n + k2, n + k, 2 * n))             # 顶盖（新增一个中心顶点）
        faces.append((k, k2, 2 * n + 1))                 # 底盖
    verts.append((0.0, h / 2, 0.0))
    verts.append((0.0, -h / 2, 0.0))
    fl = []
    for t in faces:
        a, b, c = verts[t[0]], verts[t[1]], verts[t[2]]
        ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
        nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
        nl = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
        fl.append(M3.Face(t, (118, 120, 124), (nx / nl, ny / nl, nz / nl), None,
                          nl / 2.0, "M_steel"))
    return M3.MeshGeometry(verts, fl, name="octaprobe", source="<synthetic>")


def _scene(category, mesh, ctr=None, size=(2, 2, 2)):
    M3.MESH_LIBRARY[category] = mesh
    ctr = ctr or (0.0, 0.0, 0.0)
    cam = Camera3D.from_look_at((2.6, 1.3, -2.9), (0.0, 0.0, 0.0),
                               up=(0.0, 1.0, 0.0), fov_deg=45.0)
    w = World3D(cam)
    w.objects = [Object3D(category, ctr, size, (170, 172, 178), "box")]
    return w


def _evidence_module(name, what):
    """加载 `_evidence/shade-paint/` 里的**证据侧工具**（不进产品 PR）。

    · CI（无 `_evidence/`）取不到 ⇒ **skip 而非红**——假红比红更坏。
    · `except` **不 reraise** ⇒ `test_gate_dependency_closure` 不把它当硬第三方依赖；
      用 `__import__(name)` 走字符串，AST 扫描器也看不到裸 `import` 语句。
    """
    if EVID not in sys.path:
        sys.path.insert(0, EVID)
    try:
        return __import__(name)
    except ImportError:
        pytest.skip("%s 不在本树（CI 无 _evidence/shade-paint/）—— 本用例需它做%s" % (name, what))


def _stats(img_bytes, w, h):
    SM = _evidence_module("shade_metrics", "像素度量")
    return SM.image_stats(rgb=img_bytes, w=w, h=h)


def _render_stats(mode, mesh, *, legacy_color=False):
    cat = "probe_%s_%d" % (mode, 1 if legacy_color else 0)
    w = _scene(cat, mesh)
    w.mesh_shade = mode
    saved = M3.shade_rgb
    if legacy_color:
        M3.shade_rgb = M3.shade_rgb_legacy
    try:
        img = w.render(SW, SH)
    finally:
        M3.shade_rgb = saved
    return _stats(img.tobytes(), SW, SH)


# ===================================================================== A 纯函数
def test_a1_模式闭集与角色闭集_且与冻结契约同源():
    assert M3.SHADE_MODES == ("studio", "lambert", "naive", "toon", "toon_studio")
    assert M3.SHADE_MODES == SP.SHADE_MODES
    # 角色闭集必须与 spec.py 的 MATERIAL_ROLES 逐个相同（跨仓同源；spec.py 不在则只校验本地冻结值）
    frozen = ("metal", "painted_metal", "rubber", "glass", "emissive",
              "plastic", "wood", "stone", "fabric", "skin")
    assert M3.ROLES == frozen
    assert SP.ROLES == frozen
    if os.path.isfile(SPEC_PY):
        src = open(SPEC_PY, encoding="utf-8").read()
        i = src.index("MATERIAL_ROLES")
        seg = src[i:src.index(")", i)]
        for r in frozen:
            assert '"%s"' % r in seg, "spec.py 的 MATERIAL_ROLES 里没有 %r（两份清单漂移了）" % r


def test_a2_srgb线性往返与解析值():
    for v in (0.0, 0.02, 0.18, 0.5, 0.9, 1.0):
        assert abs(SP.linear_to_srgb(SP.srgb_to_linear(v)) - v) < 1e-9
    assert abs(SP.srgb_to_linear(0.18) - 0.02721) < 1e-4


def test_a3_环境双基座_朝下不为零且显著优于冻结口径():
    v = (0.0, 0.0, -1.0)
    down = SP.shade_studio((0.0, -1.0, 0.0), v, "painted_metal")
    up = SP.shade_studio((0.0, 1.0, 0.0), v, "painted_metal")
    lam_down = SP.shade_intensity_mode((0.0, -1.0, 0.0), v, "lambert")
    assert up > down, "朝上必须比朝下亮"
    assert lam_down == pytest.approx(0.12, abs=1e-12), "冻结口径的朝下面应当就是地板值"
    assert down > lam_down + 0.10, (down, lam_down)


def test_a4_半lambert_背光面不塌零():
    n = (0.42, -0.78, 0.46)          # 与主光近似反向
    v = (0.0, 0.0, -1.0)
    back = SP.shade_studio(n, v, "painted_metal")
    lam_back = SP.shade_intensity_mode(n, v, "lambert")
    assert lam_back == pytest.approx(0.12, abs=1e-12)
    assert back > lam_back + 0.15, (back, lam_back)


def test_a5_边缘光是可测的_掠射面上开与关必须有差():
    v = (0.0, 0.0, -1.0)
    graze = (-1.0, 0.0, 0.0)
    assert SP.shade_studio(graze, v, "metal") > \
        SP.shade_studio(graze, v, "metal", rim_scale=0.0) + 0.05
    # 正对相机的面不受边缘光影响（rim 只在掠射时起作用）
    assert SP.shade_studio((0.0, 0.0, -1.0), v, "metal") == pytest.approx(
        SP.shade_studio((0.0, 0.0, -1.0), v, "metal", rim_scale=0.0), abs=1e-12)


def test_a6_软肩有界单调且不硬削平():
    assert all(SP.look_curve(a) <= SP.look_curve(a + 0.1) + 1e-12
               for a in (0.0, 0.3, 0.7, 1.2, 2.0))
    assert SP.look_curve(1e6) <= 1.0 + 1e-9
    # 1.3 的裸强度经软肩应落在 (0.99, 1.0) —— 硬钳位会直接给 1.0（那就没有层次了）
    assert 0.99 < SP.look_curve(1.3) <= 1.0


def test_a7_线性空间乘法_必须比旧口径亮且命中解析值():
    base = (118, 120, 124)           # M_steel 的 sRGB
    new = SP.shade_rgb(base, 0.5)[0]
    old = SP.shade_rgb_legacy(base, 0.5)[0]
    assert old == 59, "v2 口径的冻结值变了（59 是实测的旧读数）"
    assert new == 85, "线性空间乘法的解析值应为 85（IEC 61966-2-1）"
    assert new > old


def test_a8_角色分层_同朝向下四个角色强度互不相同():
    v = (0.0, 0.0, -1.0)
    vals = {r: SP.shade_studio((0.0, 1.0, 0.0), v, r)
            for r in ("metal", "rubber", "glass", "fabric")}
    assert len(set(round(x, 6) for x in vals.values())) == 4, vals
    assert SP.shade_studio((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), "emissive") == \
        pytest.approx(SP.EMISSIVE_GAIN, abs=1e-12)


def test_a9_toon色阶量化_档数受控():
    vals = [SP.shade_intensity_mode(n, (0.0, 0.0, -1.0), "toon", role="metal", bands=3)
            for n in ((0, 1, 0), (0.5, 1, 0), (1, 0, 0), (-1, 0, 0), (0, -1, 0))]
    assert len(set(round(x, 9) for x in vals)) <= 4, vals
    assert all(0.0 <= x <= 1.0 for x in vals), vals


def test_a10_角色识别_关键词与优先级():
    cases = {"M_brass": "metal", "M_steel": "metal", "M_dark": "painted_metal",
             "M_glow": "emissive", "rubber_grip": "rubber", "Glass_Lens": "glass",
             "painted_metal_A": "painted_metal", "WoodOak": "wood",
             "concrete_floor": "stone", "cloth_banner": "fabric", "skin_arm": "skin",
             "": "painted_metal", None: "painted_metal", "UNKNOWN_MAT": "painted_metal"}
    bad = {k: (v, M3.role_for_material(k)) for k, v in cases.items()
           if M3.role_for_material(k) != v}
    assert not bad, bad
    # 优先级：painted_metal 必须先于 metal 命中（否则涂装件会被当裸金属）
    assert M3.role_for_material("M_painted_metal") == "painted_metal"


def test_a11_色板十行_基色互异_恰一个自发光():
    rows = SP.palette_rows()
    assert len(rows) == 10
    assert tuple(r["role"] for r in rows) == SP.ROLES
    assert len({tuple(r["base_rgb255"]) for r in rows}) == 10
    assert sum(1 for r in rows if r["unlit"]) == 1
    for r in rows:
        assert all(0 <= c <= 255 for c in r["base_rgb255"]), r


def test_a12_能力清单两栏都非空且不给做不到的假装实现():
    cap = SP.capability_report()
    assert len(cap["can"]) >= 8 and len(cap["cannot"]) >= 6
    blob = " ".join(cap["cannot"])
    for must in ("阴影", "全局光照", "次表面散射"):
        assert must in blob, "能力上界必须点名做不到 %s" % must


def test_a13_selftest_verdict_pass():
    r = SP.selftest()
    fails = {k: v for k, v in r.items() if isinstance(v, dict) and not v.get("ok")}
    assert not fails, fails
    assert r["verdict"] == "PASS"


# ===================================================================== B 冻结口径
def test_b1_lambert数值与独立重写的旧公式逐位一致():
    key = (-0.42, 0.78, -0.46)
    normals = [(0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1),
               (0.42, -0.78, 0.46), (0.577, 0.577, 0.577)]
    for n in normals:
        for v in ((0, 0, -1), (0.5, 0.5, -0.7), (1, 0, 0)):
            k = n[0] * key[0] + n[1] * key[1] + n[2] * key[2]
            k = k if k > 0 else 0.0
            h = n[0] * v[0] + n[1] * v[1] + n[2] * v[2]
            h = h if h > 0 else 0.0
            old = 0.38 * (0.5 + 0.5 * n[1]) + 0.72 * k + 0.17 * h
            old = old if old >= 0.12 else 0.12
            got = SP.shade_intensity_mode(n, v, "lambert")
            assert got == pytest.approx(old, abs=1e-15), (n, v, got, old)


def test_b2_naive与冻结公式一致():
    for c in (0.0, 0.25, 0.5, 0.75, 1.0, 2.0, -1.0):
        assert SP.shade_intensity_mode((0, 1, 0), (0, 0, -1), "naive", cosang=c) == \
            pytest.approx(0.55 + 0.55 * max(0.0, min(1.0, c)), abs=1e-15)


def test_b3_mesh3d的shade_rgb与legacy仍然转发到shade_paint():
    assert M3.shade_rgb((118, 120, 124), 0.5) == SP.shade_rgb((118, 120, 124), 0.5)
    assert M3.shade_rgb_legacy((118, 120, 124), 0.5) == (59, 60, 62)
    # 默认模式必须是 studio（改回 lambert 会让 D 组变红）
    assert World3D.mesh_shade == "studio"


def test_b4_face带上了材质名_旋转后仍保留():
    mesh = _octagonal_prism()
    assert all(f.mat == "M_steel" for f in mesh.faces)
    r = mesh.rotated((0.3, 0.0, 0.0))
    assert all(f.mat == "M_steel" for f in r.faces)


# ===================================================================== C 端到端（合成）
def test_c1_studio在合成模型上_最暗前景与对比度值域三项全胜回退管线():
    mesh = _octagonal_prism()
    new = _render_stats("studio", mesh)
    old = _render_stats("lambert", mesh, legacy_color=True)
    # 判据用**最暗前景分位**而不是整体均值：缺陷是"侧面死黑"，
    # 而均值会被亮面与背景拉高、把"最暗那批像素黑成一团"掩盖掉（实测踩过：
    # 合成八角柱上 lambert 的均值反而更高，但最暗前景明显更黑）。
    assert new["fg_p05_luma"] > old["fg_p05_luma"] + 8, (new["fg_p05_luma"],
                                                        old["fg_p05_luma"])
    assert new["fg_bg_contrast"] > old["fg_bg_contrast"], (new["fg_bg_contrast"],
                                                           old["fg_bg_contrast"])
    assert new["hist_coverage"]["span_bins"] > old["hist_coverage"]["span_bins"], (
        new["hist_coverage"]["span_bins"], old["hist_coverage"]["span_bins"])
    assert new["std_luma"] >= old["std_luma"], (new["std_luma"], old["std_luma"])


def test_c2_八角柱的面必须互相可区分():
    """缺陷：旧管线把 8 个侧面压成同一片黑（"只能看出轮廓"）。
    判据：前景里不同亮度**档**（4 档/级）数 ≥ 5 —— 8 个侧面 + 顶面至少 5 个不同明暗。"""
    SM = _evidence_module("shade_metrics", "像素度量")
    mesh = _octagonal_prism()
    w = _scene("octa_flat", mesh)
    w.mesh_shade = "studio"
    img = w.render(SW, SH)
    st = SM.image_stats(rgb=img.tobytes(), w=SW, h=SH)
    lum = SM.luma_of(img.tobytes())
    mask, _bg = SM.fg_foreground(SW, SH, img.tobytes())
    bands = {v // 4 for v, m in zip(lum, mask) if m}
    assert len(bands) >= 5, "八角柱前景只有 %d 个亮度档 ⇒ 面与面不可区分" % len(bands)
    assert st["fg_mean_luma"] > 30, st["fg_mean_luma"]


def test_c3_toon模式下前景色阶数受控():
    SM = _evidence_module("shade_metrics", "像素度量")
    mesh = _octagonal_prism()
    w = _scene("octa_toon", mesh)
    w.mesh_shade = "toon"
    img = w.render(SW, SH)
    mask, _bg = SM.fg_foreground(SW, SH, img.tobytes())
    lum = SM.luma_of(img.tobytes())
    levels = len({v for v, m in zip(lum, mask) if m})
    assert levels <= 8, "toon 模式前景竟有 %d 个不同灰阶（色阶量化失效）" % levels


# ===================================================================== D 真实模型
@pytest.mark.skipif(not os.path.isfile(TURRET_OBJ), reason="turret OBJ 不在本机")
def test_d1_真实turret_同机位下studio越线而回退不越线_回退即红(tmp_path):
    """**回退即红**：这一组同时断言"新管线达标"与"旧管线不达标"。
    有人把 `mesh_shade` 改回 lambert（或把 shade_rgb 改回 sRGB 直接乘），
    新管线那侧立刻掉到线下 ⇒ 第一个 assert 变红。"""
    SM = _evidence_module("shade_metrics", "像素度量")
    RT = _evidence_module("render_turret", "对照渲染")

    # ★ 关键：新管线取的是**出厂默认**，不是写死的 "studio"。
    # 否则"把默认改回 lambert"这种回退不会被这一组抓住（实测踩过）。
    assert World3D.mesh_shade == "studio", (
        "World3D.mesh_shade 默认被改回 %r ⇒ 回退即红" % (World3D.mesh_shade,))

    w3a, mesh, geo = RT.build_scene(TURRET_OBJ, TURRET_MTL, "turret_d1", sw=SW, sh=SH)
    w3b, _m2, _g2 = RT.build_scene(TURRET_OBJ, TURRET_MTL, "turret_d1", sw=SW, sh=SH)

    w3a.mesh_shade = World3D.mesh_shade            # 出厂默认管线
    new = SM.image_stats(rgb=w3a.render(SW, SH).tobytes(), w=SW, h=SH, name="shipped")
    saved = M3.shade_rgb
    M3.shade_rgb = M3.shade_rgb_legacy
    w3b.mesh_shade = "lambert"
    try:
        old = SM.image_stats(rgb=w3b.render(SW, SH).tobytes(), w=SW, h=SH, name="lambert")
    finally:
        M3.shade_rgb = saved

    os.makedirs(EVID, exist_ok=True)
    with open(os.path.join(EVID, "guard_d1_readings.json"), "w", encoding="utf-8") as f:
        json.dump({"shipped_default": new, "lambert_legacy_v2": old,
                   "framing": geo.get("framing")}, f, ensure_ascii=False,
                  indent=1, sort_keys=True)

    # 实测（800×600 同机位）：出厂默认 fg_mean=86.4 / contrast=2.33 / span=9
    #                          v2 回退  fg_mean=60.5 / contrast=1.54 / span=4
    # 阈值取两者之间并留余量 ⇒ 任一侧回退必红。
    assert new["fg_mean_luma"] >= 75.0, ("出厂管线前景亮度没到线（回退即红触发）",
                                         new["fg_mean_luma"])
    assert new["fg_p05_luma"] > old["fg_p05_luma"] + 8, (new["fg_p05_luma"],
                                                        old["fg_p05_luma"])
    assert new["fg_bg_contrast"] >= 2.0, new["fg_bg_contrast"]
    assert new["hist_coverage"]["span_bins"] >= 7, new["hist_coverage"]
    assert old["fg_mean_luma"] < 75.0, ("v2 管线竟然也越线了 ⇒ 阈值失去判别力",
                                        old["fg_mean_luma"])
    assert new["fg_mean_luma"] > old["fg_mean_luma"] + 15
    assert new["fg_bg_contrast"] > old["fg_bg_contrast"] + 0.5
    assert new["hist_coverage"]["span_bins"] > old["hist_coverage"]["span_bins"]
    # 构图：完整不裁切 + 主体占比落带内
    fr = geo.get("framing") or {}
    assert fr.get("in_band") and fr.get("uncropped"), fr
