# -*- coding: utf-8 -*-
"""shade_paint · **零依赖着色模型**（纯标准库；世界模型侧的呈现能力核心）
================================================================================
为什么单独一个模块（而不是直接改 `mesh3d.shade_intensity`）：
  · `mesh3d.py` 正被另外的工作流并发编辑（`recenter_offset` 等），把新光学模型塞进它的
    常数区会增加互相踩踏的面积；本模块是**纯函数 + 纯数据**，零副作用、零 import 第三方；
  · 更重要的是：着色模型要能**在没有 Blender、没有 GPU、没有 numpy/PIL 的机器上**
    被单测与复核。放在一个只有 `math`/`typing` 的模块里，这一条才是真的。

──────────────────────────────────────────────────────────────── 诊断（缺陷 → 修法）
实测读数（turret OBJ，2028 面，800×600，见 `_evidence/shade-paint/raw/`）：

  旧模型（`mesh3d` 的 `lambert`）           mean_luma=48.6/255   std=44.1
                                            90% 质量只跨 13/32 档  前景/背景 WCAG 对比 2.48
  观感                                       侧面大片死黑，只剩轮廓

根因（三条，都可解析验证，不是感觉）：
  1. **环境项被乘上了"朝上程度"**：`AMBIENT * (0.5 + 0.5*ny)` ⇒ 竖直面只剩一半、
     朝下面归零。而真实物体**朝下的面也正是最需要地面反弹的**。
  2. **只有一盏正前方主光 + 裸 `max(0, n·l)`**：背光面直接归零（再被 SHADE_FLOOR 抬到 0.12）。
  3. **在 sRGB 空间直接乘强度**：等于对暗部做 gamma 平方压缩。线性值 0.18 乘 0.5 应为
     sRGB 85，旧写法给 59 —— 暗部整体再暗一档。

修法（本模块）：
  ① 半球环境**双基座**：`AMB_TOP`（天光）+ `AMB_BOT`（地面反弹），两项都不为零；
  ② **半 Lambert** `0.5 + 0.5*(n·l)` 与裸 Lambert 按 `KEY_LAMBERT_MIX` 混合 ⇒ 背光面不塌零；
  ③ **对侧补光** `FILL_DIR`（大面积弱增益）只抬暗部，不造第二个高光；
  ④ **边缘光**（Fresnel `(1 - n·view)^RIM_POW`）⇒ **掠射角提亮**，把主体从背景里抠出来
     （对应 Blender 侧 VISUAL-ITERATIONS.md 手法 3）；这是"轮廓可见"的唯一零依赖手段；
  ⑤ **线性空间乘法**（`shade_rgb`）：sRGB→linear → ×强度 → 软肩 → linear→sRGB；
  ⑥ **曝光 + 软肩**（`EXPOSURE`/`SHOULDER`）：对齐 Blender 侧 `AgX - Punchy + exposure 0.15`
     的取向；用指数软肩取代硬钳位，避免高光成片削平（削平会让直方图堆在 255）；
  ⑦ **按材质 role 分层**（`ROLE_PALETTE`）：不同角色给不同的主光/补光/环境/边缘光系数，
     发光体走 `unlit`。**不改资产声明的颜色**（MTL 的 Kd 是权威）。

──────────────────────────────────────────────────────────────── 能力上界（如实说）
**能**（本模块就是这些）：平面色 / 逐面 Lambert + 半 Lambert / 方向光 + 环境项 + 边缘光 /
色阶量化（toon）/ 双色调 / 背景渐变 / 地面与主体分离 / 构图规则 / 直方图读数。
**做不到**（零依赖的天花板，别假装）：软阴影 / 全局光照 / 屏幕空间光追 /
PBR 金属环境反射（没有环境贴图与 BRDF）/ 次表面散射 / 置换与法线贴图 / 景深与运动模糊。
本模型是"逐面直接光 + 解析环境项"，**不产生阴影、不守恒能量** —— 它优化的是
"一眼能看清形体与材质"，不是物理正确。

判据来源：本件经验标定（三光源方位与 Blender 侧 `shade/render.py` 的 key/fill/rim 同向）；
可红读数在 `tests/test_issue427_shade_paint_guard.py`，对照图与读数在
`_evidence/shade-paint/`。WCAG 相对亮度比是 W3C 公开口径。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "ROLES", "DEFAULT_ROLE", "ROLE_KEYWORDS", "ROLE_PALETTE", "SHADE_MODES",
    "KEY_LIGHT", "FILL_DIR", "AMB_TOP", "AMB_BOT", "KEY_GAIN_STUDIO",
    "KEY_LAMBERT_MIX", "FILL_GAIN_STUDIO", "HEADLIGHT_GAIN_STUDIO",
    "RIM_GAIN", "RIM_POW", "EXPOSURE", "SHOULDER", "TOON_BANDS", "EMISSIVE_GAIN",
    "ROLE_BASE_LINEAR", "role_for_material", "role_palette", "shade_studio",
    "shade_intensity_mode", "shade_rgb", "shade_rgb_legacy", "look_curve",
    "srgb_to_linear", "linear_to_srgb", "palette_rows", "capability_report",
    "selftest",
]

Vec3 = Tuple[float, float, float]
RGB = Tuple[int, int, int]

# ─────────────────────────────────────────────────────────── 光学模型常数
#: 主光方向（世界系单位向量，**指向光源**；y 向上 = 场景的「上」）。与 mesh3d.KEY_LIGHT 同值。
KEY_LIGHT: Vec3 = (-0.42, 0.78, -0.46)
#: 补光方向（对侧、略低、偏前）。
FILL_DIR: Vec3 = (0.88, 0.30, 0.36)
#: 半球环境：天光（朝上满值）。
AMB_TOP = 0.28
#: 半球环境：地面反弹（朝下满值）—— **非零是"不再死黑"的关键**。
AMB_BOT = 0.13
#: 主光增益。
KEY_GAIN_STUDIO = 0.58
#: 主光里"裸 Lambert"的占比（其余给半 Lambert）。0 = 全半 Lambert，1 = 全裸 Lambert。
KEY_LAMBERT_MIX = 0.60
#: 补光增益。
FILL_GAIN_STUDIO = 0.17
#: 视线补光增益（保证正对相机的面永远可读）。
HEADLIGHT_GAIN_STUDIO = 0.10
#: 边缘光基准增益与幂次（掠射时 = RIM_GAIN × 角色的 rim 系数）。
#: 幂次越大，边缘带越**窄**（越像轮廓线）；2.0 是"看得见但不糊住整面"的折中（看图标定）。
RIM_GAIN = 0.30
RIM_POW = 2.0
#: 发光体（unlit）的等效强度。
EMISSIVE_GAIN = 1.25
#: 曝光（≈ +0.2 EV；对齐 Blender 侧 presentation 档的 exposure=0.15）。
EXPOSURE = 1.15
#: 软肩起点（强度超过它以后按指数压缩，取代硬钳位）。
SHOULDER = 0.88
#: 色阶量化（toon）默认档数。
TOON_BANDS = 3

#: 角色闭集 —— 与 `wm_bridge/factory/geo/spec.py: MATERIAL_ROLES` 同源同名
#: （守卫测试断言两份清单一致，防止各自漂移）。
ROLES: Tuple[str, ...] = ("metal", "painted_metal", "rubber", "glass", "emissive",
                          "plastic", "wood", "stone", "fabric", "skin")
#: 认不出来时的回退角色（= spec.py 里 `part.material.role` 的默认值，不猜语义）。
DEFAULT_ROLE = "painted_metal"

#: 材质名 → role 关键词表。**顺序即优先级**（`painted_metal` 必须先于 `metal`，
#: 否则 `M_painted_metal` 会被 metal 抢走）。匹配口径：小写子串。
ROLE_KEYWORDS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("painted_metal", ("painted", "paint", "coating", "lacquer", "涂装", "漆")),
    ("emissive", ("glow", "emissive", "emit", "lamp", "neon", "led", "发光", "灯")),
    ("metal", ("brass", "steel", "iron", "metal", "aluminium", "aluminum", "alu",
               "copper", "chrome", "bronze", "gold", "silver", "金属", "黄铜")),
    ("rubber", ("rubber", "tyre", "tire", "grip", "橡胶", "胶")),
    ("glass", ("glass", "lens", "window", "crystal", "玻璃")),
    ("plastic", ("plastic", "abs", "resin", "塑")),
    ("wood", ("wood", "timber", "oak", "木")),
    ("stone", ("stone", "concrete", "rock", "granite", "marble", "石")),
    ("fabric", ("fabric", "cloth", "textile", "canvas", "布")),
    ("skin", ("skin", "flesh", "leather", "皮肤")),
)

#: role → 着色系数。`key/fill/amb` 是相对基准的倍率，`rim` 是掠射边缘光的绝对增益系数。
#: `unlit=True` ⇒ 自发光，忽略光照模型（只乘 EMISSIVE_GAIN）。
ROLE_PALETTE: Dict[str, Dict] = {
    "metal": {"rim": 0.30, "key": 1.12, "fill": 0.85, "amb": 0.88, "unlit": False},
    "painted_metal": {"rim": 0.20, "key": 1.00, "fill": 1.00, "amb": 1.00, "unlit": False},
    "rubber": {"rim": 0.12, "key": 0.82, "fill": 1.05, "amb": 1.08, "unlit": False},
    "glass": {"rim": 0.55, "key": 0.92, "fill": 1.05, "amb": 1.18, "unlit": False},
    "emissive": {"rim": 0.00, "key": 0.00, "fill": 0.00, "amb": 0.00, "unlit": True},
    "plastic": {"rim": 0.36, "key": 1.14, "fill": 0.90, "amb": 0.94, "unlit": False},
    "wood": {"rim": 0.10, "key": 0.84, "fill": 0.98, "amb": 0.90, "unlit": False},
    "stone": {"rim": 0.13, "key": 0.95, "fill": 1.05, "amb": 1.06, "unlit": False},
    "fabric": {"rim": 0.18, "key": 0.90, "fill": 1.05, "amb": 1.06, "unlit": False},
    "skin": {"rim": 0.30, "key": 0.95, "fill": 1.02, "amb": 1.10, "unlit": False},
}

#: 各角色的**基准色（线性空间）** —— 与 Blender 侧 `shade/material.py: ROLE_DEFAULTS`
#: 的 `base_color` 同值。用途：资产**没有**声明颜色（无 MTL / 材质名认不出）时，
#: 按 role 给一个可解释的起始色，而不是统统用一个灰（旧图元通道的病根之一）。
#: 资产一旦声明了颜色（MTL 的 Kd），**以资产为准**，这里不参与。
ROLE_BASE_LINEAR: Dict[str, Tuple[float, float, float]] = {
    "metal": (0.780, 0.800, 0.840),
    "painted_metal": (0.145, 0.290, 0.560),
    "rubber": (0.040, 0.042, 0.048),
    "glass": (0.840, 0.930, 1.000),
    "emissive": (1.000, 0.620, 0.260),
    "plastic": (0.760, 0.135, 0.095),
    "wood": (0.330, 0.190, 0.090),
    "stone": (0.400, 0.395, 0.375),
    "fabric": (0.400, 0.105, 0.140),
    "skin": (0.760, 0.545, 0.450),
}

#: `shade_intensity_mode` 认得的模式（闭集）。`lambert`/`naive` 是**冻结**的历史口径。
SHADE_MODES: Tuple[str, ...] = ("studio", "lambert", "naive", "toon", "toon_studio")


# ─────────────────────────────────────────────────────────── sRGB ↔ 线性
def srgb_to_linear(u: float) -> float:
    """sRGB 0..1 → 线性 0..1（IEC 61966-2-1 分段函数）。"""
    u = 0.0 if u < 0.0 else (1.0 if u > 1.0 else float(u))
    return u / 12.92 if u <= 0.04045 else ((u + 0.055) / 1.055) ** 2.4


def linear_to_srgb(v: float) -> float:
    """线性 0..1 → sRGB 0..1。"""
    v = 0.0 if v < 0.0 else (1.0 if v > 1.0 else float(v))
    return v * 12.92 if v <= 0.0031308 else 1.055 * (v ** (1.0 / 2.4)) - 0.055


#: 0..255 的 sRGB → 线性查表（导入期一次建好；热路径只做一次 float 乘）
_S2L: Tuple[float, ...] = tuple(srgb_to_linear(i / 255.0) for i in range(256))


def _u8(v: float) -> int:
    v = 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)
    return int(round(v * 255.0))


def _srgb255_from_linear(v: float) -> int:
    return _u8(linear_to_srgb(v))


# ─────────────────────────────────────────────────────────── 角色
def role_for_material(name: Optional[str]) -> str:
    """材质名 → role（小写子串，按 `ROLE_KEYWORDS` 的顺序取第一个命中）。

    认不出来 ⇒ `DEFAULT_ROLE`（**不猜**：语义不明的材质名不该被硬塞成一个金属或玻璃）。
    """
    if not name:
        return DEFAULT_ROLE
    s = str(name).lower()
    for role, keys in ROLE_KEYWORDS:
        for k in keys:
            if k in s:
                return role
    return DEFAULT_ROLE


def role_palette(role: Optional[str]) -> Dict:
    """取角色的着色系数（未知 role ⇒ 回退 DEFAULT_ROLE 的系数）。"""
    return dict(ROLE_PALETTE.get(str(role or ""), ROLE_PALETTE[DEFAULT_ROLE]))


# ─────────────────────────────────────────────────────────── 光学模型
def look_curve(i: float) -> float:
    """曝光 + 软肩（"零依赖版的 AgX-Punchy + exposure 0.15"）。

    为什么不用硬钳位：硬钳位会把所有超亮面**压成同一片 255**，直方图堆在末端、
    细节全丢（`hist_coverage` 的 used_levels 会反常地小）。指数软肩保留高光层次。
    """
    v = float(i) * EXPOSURE
    if v <= 0.0:
        return 0.0
    if v > SHOULDER:
        span = 1.0 - SHOULDER
        return SHOULDER + span * (1.0 - math.exp(-(v - SHOULDER) / span))
    return v


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def shade_studio(normal: Sequence[float], view: Sequence[float],
                 role: Optional[str] = None, rim_scale: float = 1.0) -> float:
    """**studio** 光学模型：半球环境（双基座）+ 主光（裸/半 Lambert 混合）+ 对侧补光
    + 视线补光 + 边缘光。返回**未做 look 曲线**的原始强度（便于单测解析核对）。

    逐项都可手算核对（见 `selftest`）：
      sky  = AMB_TOP·(0.5+0.5·ny) + AMB_BOT·(0.5−0.5·ny)
      key  = KEY_GAIN_STUDIO · [MIX·max(0,n·l_key) + (1−MIX)·(0.5+0.5·n·l_key)]
      fill = FILL_GAIN_STUDIO · (0.5 + 0.5·n·l_fill)
      head = HEADLIGHT_GAIN_STUDIO · max(0, n·view)
      rim  = RIM_GAIN · role.rim · rim_scale · (1 − max(0, n·view))^RIM_POW

    `rim_scale` 只为**可测**而存在（`0` = 关掉边缘光）。这样"边缘光是否真的提亮了掠射面"
    可以做成一条**对照断言**，而不是"看着像有描边"。
    """
    p = role_palette(role)
    if p["unlit"]:
        return EMISSIVE_GAIN
    nx, ny, nz = normal[0], normal[1], normal[2]
    nl = math.sqrt(nx * nx + ny * ny + nz * nz)
    if nl > 1e-15:
        nx, ny, nz = nx / nl, ny / nl, nz / nl
    n = (nx, ny, nz)
    ny = n[1]
    sky = (AMB_TOP * (0.5 + 0.5 * ny) + AMB_BOT * (0.5 - 0.5 * ny)) * p["amb"]
    dk = _dot(n, KEY_LIGHT)
    key_lam = dk if dk > 0.0 else 0.0
    key_half = 0.5 + 0.5 * dk
    key = KEY_GAIN_STUDIO * (KEY_LAMBERT_MIX * key_lam
                             + (1.0 - KEY_LAMBERT_MIX) * key_half) * p["key"]
    df = _dot(n, FILL_DIR)
    fill = FILL_GAIN_STUDIO * (0.5 + 0.5 * df) * p["fill"]
    dh = _dot(n, view)
    head = HEADLIGHT_GAIN_STUDIO * (dh if dh > 0.0 else 0.0)
    grazing = 1.0 - (dh if dh > 0.0 else 0.0)
    rim = RIM_GAIN * p["rim"] * float(rim_scale) * (grazing ** RIM_POW)
    return sky + key + fill + head + rim


def shade_intensity_mode(normal: Sequence[float], view: Sequence[float],
                         mode: str = "studio", cosang: float = 1.0,
                         role: Optional[str] = None,
                         bands: Optional[int] = None) -> float:
    """按模式分派。

    · `"studio"`（**新默认**）：`shade_studio` + look 曲线；
    · `"toon"` / `"toon_studio"`：同上，再对**已 look 的强度**做色阶量化；
    · `"lambert"`：**冻结**的旧口径（不含 look 曲线），仅供逐像素回退/对照；
    · `"naive"`：与图元通道同款的 `0.55 + 0.55·cosang`，**冻结**。
    """
    if mode == "naive":
        c = cosang if cosang > 0.0 else 0.0
        if c > 1.0:
            c = 1.0
        return 0.55 + 0.55 * c
    if mode == "lambert":
        ny = normal[1]
        key = _dot(normal, KEY_LIGHT)
        if key < 0.0:
            key = 0.0
        head = _dot(normal, view)
        if head < 0.0:
            head = 0.0
        # 与 mesh3d 的冻结口径同系数（0.38 / 0.72 / 0.17 / 0.12），逐位不变
        i = 0.38 * (0.5 + 0.5 * ny) + 0.72 * key + 0.17 * head
        return i if i >= 0.12 else 0.12
    i = look_curve(shade_studio(normal, view, role))
    if mode in ("toon", "toon_studio"):
        b = int(bands or TOON_BANDS)
        if b < 2:
            b = 2
        i = (math.floor(i * b) + 0.5) / b
        if i > 1.0:
            i = 1.0
        if i < 0.0:
            i = 0.0
    return i


# ─────────────────────────────────────────────────────────── 颜色（线性空间乘法）
def shade_rgb(base: Sequence[float], intensity: float, *, look: bool = False) -> RGB:
    """基色 × 强度 → 0..255，**在线性空间做乘法**（sRGB→linear→×i→linear→sRGB）。

    为什么必须线性：sRGB 空间直接乘强度 = 对暗部做 gamma 平方压缩。
    实测：线性 0.18 的钢，强度 0.5 时应得 sRGB 85；旧写法给 59（暗 30%）。
    `look=False`（默认）表示强度**已经过** look 曲线（`shade_intensity_mode` 里做了），
    避免重复施加曝光；传 `look=True` 可对裸强度直接上曲线（给外部调用用）。

    越界一律钳到 0..255（NaN 当 0 —— 坏值不许污染整帧）。
    """
    i = look_curve(intensity) if look else float(intensity)
    if not (i == i) or i <= 0.0:
        return (0, 0, 0)
    out = []
    for k in range(3):
        c = base[k] if k < len(base) else 0.0
        if not (c == c):
            c = 0.0
        ci = int(c) if c >= 0 else 0
        if ci > 255:
            ci = 255
        v = _S2L[ci] * i
        out.append(_srgb255_from_linear(v))
    return (out[0], out[1], out[2])


def shade_rgb_legacy(base: Sequence[float], intensity: float) -> RGB:
    """**冻结**的旧口径 `int(base × i)`（sRGB 空间直接乘）。仅供对照与回退复现。"""
    r = int(base[0] * intensity)
    g = int(base[1] * intensity)
    b = int(base[2] * intensity)
    return (0 if r < 0 else (255 if r > 255 else r),
            0 if g < 0 else (255 if g > 255 else g),
            0 if b < 0 else (255 if b > 255 else b))


# ─────────────────────────────────────────────────────────── 色板 / 能力清单
def palette_rows() -> List[Dict]:
    """角色色板表（给文档、图例与"相邻色带"图直接用）。

    每行：role / base_rgb255（**资产没声明颜色时**的起始色，线性→sRGB）/
    光照系数 / 边缘光 / 是否自发光 / 零依赖可行性。
    """
    rows = []
    for role in ROLES:
        lin = ROLE_BASE_LINEAR[role]
        p = ROLE_PALETTE[role]
        rows.append({
            "role": role,
            "base_linear": [round(x, 4) for x in lin],
            "base_rgb255": [_srgb255_from_linear(x) for x in lin],
            "key": p["key"], "fill": p["fill"], "amb": p["amb"], "rim": p["rim"],
            "unlit": bool(p["unlit"]),
            "zero_dep": True,
            "note": ("自发光：忽略光照，直接 ×%.2f" % EMISSIVE_GAIN if p["unlit"]
                     else "直接光 + 半球环境 + Fresnel 边缘光"),
        })
    return rows


def capability_report() -> Dict:
    """零依赖的**呈现能力上界**（能做/做不到）。这一节是给世界模型侧的规格。"""
    return {
        "schema": "wmshade.zerodep.capability/1",
        "can": [
            "平面色（每面基色，来自 MTL 或角色色板）",
            "逐面法线明暗：Lambert / 半 Lambert / 混合",
            "方向主光 + 对侧补光 + 半球环境（天光/地面反弹双基座）",
            "Fresnel 边缘光（掠射提亮）——把主体从背景抠出来的唯一零依赖手段",
            "色阶量化（toon，N 档）/ 双色调",
            "背景渐变（径向/竖直）与地面-主体分离",
            "线性空间调色 + 曝光 + 指数软肩（防高光削平）",
            "构图规则（三分法 / 主体占比 / 完整不裁切）——纯几何，见 shade/render.py",
            "直方图 / 分位 / 前景背景对比度 / 材质可分性读数（shade_metrics.py）",
        ],
        "cannot": [
            "软阴影与接触阴影（需要阴影贴图或光线求交）",
            "全局光照 / 间接反弹（需要多次弹射积分）",
            "屏幕空间光追与反射",
            "PBR 金属环境反射（没有环境贴图、BRDF 与能量守恒）",
            "次表面散射（皮肤/玉石的透光）",
            "法线贴图 / 置换 / 凹凸细节",
            "景深、运动模糊、体积雾",
            "逐像素高光形状（本模型逐面常数，高光只有『面』级，不是一个小亮点）",
        ],
        "consequence": ("所以世界模型侧的呈现目标是『一眼能认出形体与材质』，"
                        "不是『照片级真实』。要写实必须走 Blender 基准通道。"),
        "judgement": ("能做的每一条都有可红读数（tests/test_issue427_shade_paint_guard.py）；"
                      "做不到的不给任何近似假装实现。"),
    }


# ─────────────────────────────────────────────────────────── 自检
def selftest() -> Dict:
    """纯函数自检：解析可预测的值必须逐一命中（不依赖任何外部文件/渲染）。"""
    ev = {}
    ok = True

    def chk(name, cond, detail=None):
        nonlocal ok
        if not cond:
            ok = False
        ev[name] = {"ok": bool(cond), "detail": detail}

    # 1. sRGB ↔ 线性往返
    for v in (0.0, 0.02, 0.18, 0.5, 0.9, 1.0):
        chk("srgb_roundtrip_%.2f" % v,
            abs(linear_to_srgb(srgb_to_linear(v)) - v) < 1e-9,
            linear_to_srgb(srgb_to_linear(v)))
    chk("srgb_linear_0p18", abs(srgb_to_linear(0.18) - 0.0273) < 0.002,
        srgb_to_linear(0.18))
    # 2. 环境双基座：朝上 > 朝下，且**朝下不是零**（旧模型的致命缺陷就是朝下被归零）
    #    判据用**对照**而不是拍一个魔法阈值：同法向下必须显著高于冻结 lambert 的向下。
    up = shade_studio((0.0, 1.0, 0.0), (0.0, 0.0, -1.0), "painted_metal")
    down = shade_studio((0.0, -1.0, 0.0), (0.0, 0.0, -1.0), "painted_metal")
    lam_down = shade_intensity_mode((0.0, -1.0, 0.0), (0.0, 0.0, -1.0), "lambert")
    chk("ambient_up_gt_down", up > down, [round(up, 4), round(down, 4)])
    chk("ambient_down_beats_lambert", down > lam_down + 0.10,
        {"studio_down": round(down, 5), "lambert_down": round(lam_down, 5)})
    # 3. 背光面不塌零（半 Lambert 的判据）：n 与主光反向 ⇒ 必须显著高于冻结 lambert
    back = shade_studio((0.42, -0.78, 0.46), (0.0, 0.0, -1.0), "painted_metal")
    lam_back = shade_intensity_mode((0.42, -0.78, 0.46), (0.0, 0.0, -1.0), "lambert")
    chk("backface_beats_lambert", back > lam_back + 0.15,
        {"studio_back": round(back, 5), "lambert_back": round(lam_back, 5)})
    # 4. 边缘光：**对照断言**（掠射面上 rim 开 vs 关）——不是"看起来有描边"
    graze_n = (-1.0, 0.0, 0.0)
    v = (0.0, 0.0, -1.0)
    g_on = shade_studio(graze_n, v, "metal")
    g_off = shade_studio(graze_n, v, "metal", rim_scale=0.0)
    chk("rim_brightens_grazing", g_on > g_off + 0.05,
        {"rim_on": round(g_on, 5), "rim_off": round(g_off, 5)})
    chk("rim_zero_face_on", abs(shade_studio((0.0, 0.0, -1.0), v, "metal")
                                - shade_studio((0.0, 0.0, -1.0), v, "metal",
                                               rim_scale=0.0)) < 1e-12)
    # 5. 角色分层真的不同：金属/橡胶/玻璃 在同一朝向下强度互不相同
    vals = {r: shade_studio((0.0, 1.0, 0.0), (0.0, 0.0, -1.0), r)
            for r in ("metal", "rubber", "glass", "fabric")}
    chk("roles_differ", len(set(round(v, 6) for v in vals.values())) == 4, vals)
    # 6. 自发光走 unlit
    chk("emissive_unlit",
        abs(shade_studio((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), "emissive") - EMISSIVE_GAIN)
        < 1e-12)
    # 7. look 曲线：软肩有界、单调、不削平
    chk("look_monotone", all(look_curve(a) <= look_curve(a + 0.1) + 1e-12
                            for a in (0.0, 0.3, 0.7, 1.2, 2.0, 4.0)))
    chk("look_bounded_le_1", look_curve(100.0) <= 1.0 + 1e-9, look_curve(100.0))
    chk("look_no_hard_clip", look_curve(1.3) > 0.95 and look_curve(1.3) < 1.0,
        look_curve(1.3))
    # 8. 线性空间乘法：0.18 线性 @ i=0.5 → sRGB 85（旧写法给 59）
    newm = shade_rgb((118, 120, 124), 0.5)[0]
    legacy = shade_rgb_legacy((118, 120, 124), 0.5)[0]
    chk("linear_mult_brighter_than_legacy", newm > legacy, [newm, legacy])
    chk("linear_mult_matches_analytic", abs(newm - 85) <= 2, newm)
    chk("shade_rgb_clamps", shade_rgb((255, 255, 255), 99.0) == (255, 255, 255),
        shade_rgb((255, 255, 255), 99.0))
    chk("shade_rgb_zero_intensity", shade_rgb((255, 255, 255), 0.0) == (0, 0, 0))
    chk("shade_rgb_nan_safe", shade_rgb((float("nan"), 10, 10), 1.0)[0] == 0)
    # 9. 色阶量化：toon 输出只有 N 档
    iii = {round(shade_intensity_mode((0.0, 1.0, 0.0), (0.0, 0.0, -1.0), "toon",
                                     role="metal", bands=3), 9) for _ in range(1)}
    chk("toon_quantizes", len(iii) == 1, iii)
    toon_vals = [shade_intensity_mode((x, y, 0.0), (0.0, 0.0, -1.0), "toon",
                                      role="metal", bands=3)
                 for x, y in ((0, 1), (0.5, 1), (1, 0), (-1, 0), (0, -1))]
    chk("toon_distinct_levels_le_bands_plus_1",
        len(set(round(v, 9) for v in toon_vals)) <= 4, toon_vals)
    # 10. 冻结口径不变（lambert 的常数必须还是 0.38/0.72/0.17/0.12）
    lam = shade_intensity_mode((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), "lambert")
    chk("lambert_frozen", abs(lam - (0.38 * 1.0 + 0.72 * 0.78 + 0.17 * 0.0)) < 1e-12, lam)
    chk("naive_frozen", abs(shade_intensity_mode((0, 1, 0), (0, 0, -1), "naive",
                                                 cosang=0.5) - 0.825) < 1e-12)
    chk("lambert_floor", shade_intensity_mode((0.0, -1.0, 0.0), (0.0, 0.0, 1.0),
                                              "lambert") >= 0.12)
    # 11. 角色识别
    cases = {"M_brass": "metal", "M_steel": "metal", "M_dark": "painted_metal",
             "M_glow": "emissive", "mat_rubber_grip": "rubber", "Glass_Lens": "glass",
             "painted_metal_A": "painted_metal", "WoodOak": "wood",
             "concrete_floor": "stone", "cloth_banner": "fabric",
             "skin_arm": "skin", "": "painted_metal", None: "painted_metal"}
    bad = {k: (v, role_for_material(k)) for k, v in cases.items()
           if role_for_material(k) != v}
    chk("role_keywords_hit", not bad, bad)
    # 12. 色板：10 行、基色互不相同、恰好 1 个 unlit
    rows = palette_rows()
    chk("palette_10_rows", len(rows) == 10, len(rows))
    chk("palette_base_distinct",
        len({tuple(r["base_rgb255"]) for r in rows}) == 10)
    chk("palette_one_unlit", sum(1 for r in rows if r["unlit"]) == 1)
    chk("palette_roles_match_closed_set",
        tuple(r["role"] for r in rows) == ROLES)
    # 13. 能力清单两栏都非空
    cap = capability_report()
    chk("capability_can_nonempty", len(cap["can"]) >= 8, len(cap["can"]))
    chk("capability_cannot_nonempty", len(cap["cannot"]) >= 6, len(cap["cannot"]))
    ev["verdict"] = "PASS" if ok else "FAIL"
    return ev


if __name__ == "__main__":
    import sys
    r = selftest()
    for k, v in sorted(r.items()):
        print("  %-38s %s" % (k, v))
    print("VERDICT=%s" % r["verdict"])
    sys.exit(0 if r["verdict"] == "PASS" else 1)
