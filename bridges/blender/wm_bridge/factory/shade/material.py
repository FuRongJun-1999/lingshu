# -*- coding: utf-8 -*-
"""wm_bridge.factory.shade.material · **材质配方** `wmshade.material/1`（纯标准库 · 零 bpy）

★ 这是绘画产线的**声明面**：把"某个 material.role 该长什么样"从 bpy 代码里提出来，
  变成一份可 diff、可哈希、可在无 Blender 机器上自测的**数据**。

-------------------------------------------------------------------------------- 为什么是"配方"而不是"建材质"
  既有 `runtime/material.py` 已经提供了 `material_build(name, preset, params)` —— 但它
  ① 需要 Blender（import bpy）；② preset 是**代码里的 if 链**，spec 的 role 与它没有映射；
  ③ 结果不可哈希（节点图在 bpy 里，拿不到"同一 role ⇒ 同一结果"的判据）。
  本模块只做一件事：`role (+ spec 覆写) → 一份确定的节点图配方 dict`。
  真正落到 bpy 的动作在 `bpy_shade.py`（`apply_recipe`），两边靠这份 dict 解耦 ——
  于是"配方对不对"可以在**没有 Blender 的 CI 上**逐字节核对。

-------------------------------------------------------------------------------- 契约
  · 纯函数：同输入 ⇒ 同输出；`recipe_digest` 只依赖内容（键排序 + 数值量化 1e-6）。
  · 闭集：节点类型、socket 名、角色都在白名单里；越界 ⇒ `RecipeError`（带字段路径）。
  · 不静默：`validate_recipe(recipe)` 返回 `(ok, reason)`，不抛异常也不改入参。
  · 输入面**只**来自 `wm_bridge.factory.geo.spec` 的 `part.material`：
        {"role":..., "color":..., "roughness":..., "metallic":...}
    spec 没给的字段**不猜**（用角色默认值，并把 `source` 标出来）。

-------------------------------------------------------------------------------- 两套风格
  style="pbr"  Principled BSDF 骨架 —— **GLB 可导出**（glTF 只认 Principled 的输入）。
  style="npr"  Diffuse → Shader to RGB → ColorRamp(CONSTANT 阶梯) —— 赛璐璐色阶量化。
                ⚠ 只在 EEVEE 有效（Shader to RGB 是 EEVEE 专属），且**不能进 GLB**；
                  导出用 pbr 配方，两张图共用同一份 spec（颜色/粗糙度不漂移）。
                ⚠ EEVEE 硬限制（docs/EEVEE-工作要点.md §2）：Shader to RGB 只含**直接光**，
                  没有直射光的暗部会纯黑 —— 所以 NPR 渲染必须自带补光（render.py 已给三光源）。

-------------------------------------------------------------------------------- 节点类型白名单（bpy_shade 解释器与本文档同源）
  ShaderNodeOutputMaterial / ShaderNodeBsdfPrincipled / ShaderNodeBsdfDiffuse
  ShaderNodeEmission / ShaderNodeShaderToRGB / ShaderNodeValToRGB
  ShaderNodeTexCoord / ShaderNodeMapping / ShaderNodeTexNoise / ShaderNodeTexVoronoi
  ShaderNodeBump / ShaderNodeMix / ShaderNodeMath / ShaderNodeNewGeometry
  ShaderNodeBsdfTransparent / ShaderNodeMixShader / ShaderNodeLayerWeight
"""
from __future__ import annotations

import hashlib
import json
import math

#: 配方契约标识（写进存档与所有下游产物）
MATERIAL_SCHEMA = "wmshade.material/1"

#: 数值量化位数（与 spec.py 的 QUANT 同口径：导出量子 1e-6）
QUANT = 6

#: 允许的节点类型（闭集；bpy_shade 的说明器只认这些）
NODE_TYPES = (
    "ShaderNodeOutputMaterial",
    "ShaderNodeBsdfPrincipled",
    "ShaderNodeBsdfDiffuse",
    "ShaderNodeBsdfTransparent",
    "ShaderNodeEmission",
    "ShaderNodeShaderToRGB",
    "ShaderNodeValToRGB",
    "ShaderNodeTexCoord",
    "ShaderNodeMapping",
    "ShaderNodeTexNoise",
    "ShaderNodeTexVoronoi",
    "ShaderNodeBump",
    "ShaderNodeMix",
    "ShaderNodeMath",
    "ShaderNodeNewGeometry",
    "ShaderNodeMixShader",
    "ShaderNodeLayerWeight",
)

#: 允许的风格（闭集）
STYLES = ("pbr", "npr")

__all__ = [
    "MATERIAL_SCHEMA", "QUANT", "NODE_TYPES", "STYLES", "RecipeError",
    "ROLE_DEFAULTS", "role_defaults", "role_detail", "material_recipe",
    "outline_recipe", "material_plan", "validate_recipe", "canonical_json",
    "recipe_digest", "recipe_summary", "ROLE_BOARD_ORDER",
]


class RecipeError(Exception):
    """配方不合法（fail-closed：不猜、不补、不静默降级）。"""


# ═══════════════════════════════════════════════════════════════ 角色默认值
# 设计口径：10 个角色必须在**同一布光下互相区分得开**（材质展板要一眼能认）。
# 区分靠三件事，缺一不可：① 基色色相分离 ② 粗糙度/金属度落在不同区间 ③ 有一个"招牌"现象
# （玻璃的透射、自发光的强度、布料的 Sheen、金属的拉丝各向异性、皮肤的次表面）。
# 颜色写**线性空间**（与 spec.material.color 同一口径，不在这里做 sRGB 转换）。
ROLE_DEFAULTS = {
    "metal": {
        "base_color": (0.780, 0.800, 0.840), "metallic": 1.0, "roughness": 0.18,
        "detail": "brushed", "signature": "金属度 1 + 拉丝（各向异性噪声压粗糙度）",
    },
    "painted_metal": {
        "base_color": (0.145, 0.290, 0.560), "metallic": 0.0, "roughness": 0.42,
        "detail": "orange_peel", "signature": "饱和蓝漆 + 桔皮微凹凸（粗糙 0.42）",
    },
    "rubber": {
        "base_color": (0.040, 0.042, 0.048), "metallic": 0.0, "roughness": 0.93,
        "detail": "matte_fine", "signature": "近黑 + 粗糙 0.93（几乎无高光）",
    },
    "glass": {
        "base_color": (0.840, 0.930, 1.000), "metallic": 0.0, "roughness": 0.035,
        "detail": "none", "signature": "Transmission 1.0 + IOR 1.45（真透射）",
        "transmission": 1.0, "ior": 1.45, "alpha_hint": 0.30,
    },
    "emissive": {
        "base_color": (0.020, 0.020, 0.024), "metallic": 0.0, "roughness": 0.55,
        "detail": "none", "signature": "Emission 12 W（自发光，照亮周边）",
        "emission_color": (1.000, 0.620, 0.260), "emission_strength": 12.0,
    },
    "plastic": {
        "base_color": (0.760, 0.135, 0.095), "metallic": 0.0, "roughness": 0.28,
        "detail": "matte_fine", "signature": "亮红 + Coat 清漆层（锐利高光）",
        "coat": 0.35,
    },
    "wood": {
        "base_color": (0.330, 0.190, 0.090), "metallic": 0.0, "roughness": 0.58,
        "detail": "grain", "signature": "各向异性条带噪声（顺纹理拉长）",
    },
    "stone": {
        "base_color": (0.400, 0.395, 0.375), "metallic": 0.0, "roughness": 0.86,
        "detail": "mottle", "signature": "Noise×Voronoi 斑驳 + 强凹凸（粗糙 0.86）",
    },
    "fabric": {
        "base_color": (0.400, 0.105, 0.140), "metallic": 0.0, "roughness": 0.90,
        "detail": "weave", "signature": "Sheen 0.55（绒面掠射泛白）+ 粗布纹",
        "sheen": 0.55,
    },
    "skin": {
        "base_color": (0.760, 0.545, 0.450), "metallic": 0.0, "roughness": 0.52,
        "detail": "none", "signature": "Subsurface 0.28（角质层的次表面透光）",
        "subsurface": 0.28,
    },
}

#: 材质展板推荐排布（人读顺序：先金属族，再漆/胶/塑，再木石布，最后玻璃/发光/皮肤）
ROLE_BOARD_ORDER = ("metal", "painted_metal", "rubber", "plastic", "wood",
                    "stone", "fabric", "skin", "glass", "emissive")

# ─────────────────────────────────────────────────────────────────────────────
# 复用件路由（CAPABILITY-MAP 纪律：造轮子前先找 runtime）
# `runtime/material.py` 的 `material_build(name, preset, params)` 已覆盖 11 个 preset。
# 10 个 role 里 **8 个能直接路由到它**（preset + params 由本表给出，颜色/粗糙度/金属度
# 一律从 spec 覆写值透传，所以"角色语义"仍由本模块（纯标准库、可哈希）定义）。
# 剩下 2 个 **runtime 无对应件**，回落到本模块配方的 bpy 解释器：
#   · rubber —— runtime 没有"近黑 + 粗糙 0.93 的哑光胶"preset（plastic 的粗糙度上限不够）
#   · skin   —— runtime 没有任何次表面散射（Subsurface Weight）preset
# NPR（Shader to RGB + ColorRamp 色阶）与反壳描边在 runtime 里**也都没有**，见报告"runtime 无此件"。
ROLE_RUNTIME = {
    "metal": {"preset": "metal_brushed", "params": {"scale": 620.0, "variation": 0.30}},
    "painted_metal": {"preset": "metal_paint", "params": {"scale": 260.0, "bump": 0.14}},
    "rubber": None,
    "glass": {"preset": "glass", "params": {"ior": 1.45, "transmission": 1.0}},
    "emissive": {"preset": "emission", "params": {"strength": 12.0, "mix": 1.0}},
    "plastic": {"preset": "plastic", "params": {"scale": 320.0, "bump": 0.05}},
    "wood": {"preset": "wood", "params": {"bump": 0.26}},
    "stone": {"preset": "concrete", "params": {"roughness": 0.86}},
    "fabric": {"preset": "fabric", "params": {"bump": 0.55}},
    "skin": None,
}


def role_defaults(role):
    """取某角色的默认值（**副本**，防止调用方改坏常量表）。"""
    if role not in ROLE_DEFAULTS:
        raise RecipeError("未知 material.role：%r（允许 %s）"
                          % (role, ", ".join(sorted(ROLE_DEFAULTS))))
    return dict(ROLE_DEFAULTS[role])


def role_detail(role):
    """该角色的"招牌现象"一句话说明（给报告与文档用）。"""
    d = role_defaults(role)
    return {"role": role, "signature": d["signature"], "detail": d["detail"],
            "metallic": d["metallic"], "roughness": d["roughness"],
            "base_color": list(d["base_color"])}


# ═══════════════════════════════════════════════════════════════ 基础工具
def _q(x, digits=QUANT):
    v = round(float(x), digits)
    return 0.0 if v == 0.0 else v


def _v(v, n, where):
    if not isinstance(v, (list, tuple)) or len(v) != n:
        raise RecipeError("%s 须为长度 %d 的序列，实为 %r" % (where, n, v))
    out = []
    for i, x in enumerate(v):
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
            raise RecipeError("%s[%d] 须为有限数值，实为 %r" % (where, i, x))
        out.append(round(float(x), QUANT))
    return tuple(out)


def _clamp01(x, where):
    f = float(x)
    if not math.isfinite(f):
        raise RecipeError("%s 须为有限数值，实为 %r" % (where, x))
    if f < 0.0 or f > 1.0:
        raise RecipeError("%s 须在 0..1，实为 %r" % (where, x))
    return round(f, QUANT)


def canonical_json(obj, digits=QUANT):
    """规范化 JSON 文本（键排序 / 数值量化 / -0.0 归零 / 无空格）。"""
    def conv(o):
        if isinstance(o, float):
            return _q(o, digits)
        if isinstance(o, dict):
            return {k: conv(o[k]) for k in sorted(o)}
        if isinstance(o, (list, tuple)):
            return [conv(x) for x in o]
        return o
    return json.dumps(conv(obj), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), default=str)


# ═══════════════════════════════════════════════════════════════ 细节链（PBR 质感）
def _detail_nodes(base_color, roughness, detail, role):
    """返回 (nodes, links, 末端 Color/Height 句柄)。

    句柄约定：("node_id", "socket")。约定所有细节链的输出都接到 bsdf 的指定 socket。
    """
    n, l = [], []
    if detail == "none":
        return n, l, None, None

    n.append({"id": "tc", "type": "ShaderNodeTexCoord", "loc": [-980, -260],
              "inputs": {}})
    mp_inputs = {}
    if detail == "grain":
        mp_inputs["Scale"] = [1.0, 9.0, 1.0]      # 沿 Y 拉长 → 木纹条带
    n.append({"id": "map", "type": "ShaderNodeMapping", "loc": [-790, -260],
              "inputs": mp_inputs})
    l.append(["tc", "Object", "map", "Vector"])

    if detail == "brushed":
        # 拉丝：把噪声在 X 方向拉长（Scale 的 x 远大于 z），噪声压粗糙度区间
        n.append({"id": "nz", "type": "ShaderNodeTexNoise", "loc": [-560, -260],
                  "inputs": {"Scale": 340.0, "Detail": 2.0, "Roughness": 0.30}})
        l.append(["map", "Vector", "nz", "Vector"])
        n.append({"id": "ramp", "type": "ShaderNodeValToRGB", "loc": [-330, -420],
                  "inputs": {},
                  "ramp": {"interpolation": "LINEAR",
                           "elements": [[0.30, [0.0, 0.0, 0.0, 1.0]],
                                        [0.70, [1.0, 1.0, 1.0, 1.0]]]}})
        l.append(["nz", "Factor", "ramp", "Factor"])
        n.append({"id": "mix", "type": "ShaderNodeMix", "loc": [-60, -300],
                  "props": {"data_type": "FLOAT", "blend_type": "MIX"},
                  "inputs": {"Factor": 0.45,
                             "A": round(max(0.03, roughness - 0.10), QUANT),
                             "B": round(min(0.95, roughness + 0.22), QUANT)}})
        l.append(["ramp", "Color", "mix", "Factor"])
        return n, l, ["mix", "Result"], None

    if detail in ("orange_peel", "matte_fine", "weave", "mottle", "grain"):
        scale = {"orange_peel": 190.0, "matte_fine": 520.0,
                 "weave": 260.0, "mottle": 42.0, "grain": 26.0}[detail]
        n.append({"id": "nz", "type": "ShaderNodeTexNoise", "loc": [-560, -260],
                  "inputs": {"Scale": scale,
                             "Detail": 5.0 if detail == "mottle" else 3.0,
                             "Roughness": 0.62 if detail == "mottle" else 0.45}})
        l.append(["map", "Vector", "nz", "Vector"])
        height = ["nz", "Factor"]
        if detail == "mottle":
            n.append({"id": "vor", "type": "ShaderNodeTexVoronoi", "loc": [-560, -560],
                      "inputs": {"Scale": 11.0}})
            l.append(["map", "Vector", "vor", "Vector"])
            n.append({"id": "mixv", "type": "ShaderNodeMix", "loc": [-330, -420],
                      "props": {"data_type": "RGBA", "blend_type": "MIX"},
                      "inputs": {"Factor": 0.45}})
            l.append(["nz", "Color", "mixv", "A"])
            l.append(["vor", "Color", "mixv", "B"])
            height = ["mixv", "Result"]
        strength = {"orange_peel": 0.12, "matte_fine": 0.06,
                    "weave": 0.30, "mottle": 0.45, "grain": 0.28}[detail]
        n.append({"id": "bump", "type": "ShaderNodeBump", "loc": [-60, -420],
                  "inputs": {"Strength": strength, "Distance": 0.02}})
        l.append([height[0], height[1], "bump", "Height"])
        return n, l, None, ["bump", "Normal"]

    raise RecipeError("未知 detail：%r" % (detail,))


# ═══════════════════════════════════════════════════════════════ PBR 配方
def _pbr_recipe(role, d, color, roughness, metallic, detail_on):
    n, l = [], []
    detail = d["detail"] if detail_on else "none"
    dn, dl, color_handle, normal_handle = _detail_nodes(d["base_color"], roughness, detail, role)
    n.extend(dn)
    l.extend(dl)

    out = {"id": "out", "type": "ShaderNodeOutputMaterial", "loc": [520, 0], "inputs": {}}
    bsdf_inputs = {"Base Color": list(color), "Metallic": metallic, "Roughness": roughness}
    if "ior" in d:
        bsdf_inputs["IOR"] = d["ior"]
    if "transmission" in d:
        bsdf_inputs["Transmission Weight"] = d["transmission"]
    if "coat" in d:
        bsdf_inputs["Coat Weight"] = d["coat"]
        bsdf_inputs["Coat Roughness"] = 0.04
    if "sheen" in d:
        bsdf_inputs["Sheen Weight"] = d["sheen"]
        bsdf_inputs["Sheen Roughness"] = 0.35
    if "subsurface" in d:
        bsdf_inputs["Subsurface Weight"] = d["subsurface"]
    if role == "emissive":
        bsdf_inputs["Emission Color"] = list(d["emission_color"])
        bsdf_inputs["Emission Strength"] = d["emission_strength"]

    bsdf = {"id": "bsdf", "type": "ShaderNodeBsdfPrincipled", "loc": [300, 0],
            "inputs": bsdf_inputs}
    n.append(bsdf)
    n.append(out)
    l.append(["bsdf", "BSDF", "out", "Surface"])

    if color_handle is not None:
        # 拉丝：噪声→粗糙度（不动基色，保持金属本色）
        l.append([color_handle[0], color_handle[1], "bsdf", "Roughness"])
    if normal_handle is not None:
        l.append([normal_handle[0], normal_handle[1], "bsdf", "Normal"])

    props = {"use_backface_culling": False}
    if role == "glass":
        props["surface_render_method"] = "BLENDED"   # 5.x：透明用 BLENDED（旧的 blend_method 已废）
        props["use_backface_culling"] = False
    return {
        "schema": MATERIAL_SCHEMA, "role": role, "style": "pbr",
        "nodes": n, "links": l, "props": props,
        "role_defaults": {"signature": d["signature"], "detail": d["detail"]},
    }


# ═══════════════════════════════════════════════════════════════ NPR 配方（赛璐璐）
def _npr_recipe(role, d, color, roughness, bands=3, shadow=0.30):
    """Diffuse → Shader to RGB → ColorRamp(CONSTANT) → 输出。

    色阶量化的**量化判据**：输出颜色只可能是 ColorRamp 的 N 个 stop 色之一
    （CONSTANT 插值下不会产生中间色）⇒ 前景里"主导色"个数 ≤ bands + 描边色 + 抗锯齿过渡色。
    """
    if not (2 <= bands <= 8):
        raise RecipeError("bands 须在 2..8，实为 %r" % (bands,))
    # #2100：shadow 的 0..1 契约此前零校验——>1 时「最暗档」比基色还亮（语义
    # 反转 + HDR 超白）、<0 静默钳 0 无报错；同函数 roughness/metallic 越界
    # fail-closed（_unit01），shadow 例外无任何声明。现对齐 fail-closed。
    try:
        _sh = float(shadow)
    except (TypeError, ValueError):
        raise RecipeError("material.shadow 须为有限数值，实为 %r" % (shadow,))
    if not math.isfinite(_sh) or _sh < 0.0 or _sh > 1.0:
        raise RecipeError("material.shadow 须在 0..1，实为 %r" % (shadow,))
    lo = round(_sh, QUANT)
    els = []
    for i in range(bands):
        t = i / float(bands - 1)
        k = lo + (1.0 - lo) * t                     # 亮度倍率
        els.append([round(i / float(bands), QUANT),
                    [round(color[0] * k, QUANT), round(color[1] * k, QUANT),
                     round(color[2] * k, QUANT), 1.0]])
    n = [
        {"id": "diff", "type": "ShaderNodeBsdfDiffuse", "loc": [-260, 0],
         "inputs": {"Color": [1.0, 1.0, 1.0, 1.0], "Roughness": 0.0}},
        {"id": "s2rgb", "type": "ShaderNodeShaderToRGB", "loc": [-60, 0], "inputs": {}},
        {"id": "ramp", "type": "ShaderNodeValToRGB", "loc": [160, 0], "inputs": {},
         "ramp": {"interpolation": "CONSTANT", "elements": els}},
        {"id": "out", "type": "ShaderNodeOutputMaterial", "loc": [520, 0], "inputs": {}},
    ]
    l = [
        ["diff", "BSDF", "s2rgb", "Shader"],
        ["s2rgb", "Color", "ramp", "Factor"],
        ["ramp", "Color", "out", "Surface"],
    ]
    return {
        "schema": MATERIAL_SCHEMA, "role": role, "style": "npr",
        "nodes": n, "links": l,
        "props": {"use_backface_culling": False, "surface_render_method": "DITHERED"},
        "npr": {"bands": bands, "shadow_floor": lo,
                "quantified_colors": len(els)},
        "role_defaults": {"signature": d["signature"], "detail": d["detail"]},
    }


# ═══════════════════════════════════════════════════════════════ 对外主函数
def material_recipe(role, color=None, roughness=None, metallic=None,
                    *, style="pbr", detail=True, bands=3, shadow=0.30):
    """`material.role`（+ spec 覆写）→ **确定的**节点图配方 dict。

    参数
      role        spec.MATERIAL_ROLES 之一（闭集）
      color       spec.material.color（线性 RGBA 或 RGB；None ⇒ 用角色默认基色）
      roughness   spec.material.roughness（0..1；None ⇒ 角色默认）
      metallic    spec.material.metallic（None ⇒ 角色默认；**玻璃/自发光强制 0**）
      style       "pbr" | "npr"
      detail      是否挂程序化细节链（噪声/凹凸）；关掉可得"纯色球"用于对照
      bands       NPR 色阶数（2..8）
      shadow      NPR 最暗一档的亮度下限（0..1）

    返回
      配方 dict（键序固定；`recipe_digest` 可核对）。**同输入 ⇒ 逐字节相同 JSON**。
    """
    if style not in STYLES:
        raise RecipeError("style 须为 %s 之一，实为 %r" % ("/".join(STYLES), style))
    d = role_defaults(role)

    if color is None:
        c = _v(d["base_color"], 3, "role_defaults[%s].base_color" % role)
        color_source = "role_default"
    else:
        c = _v(color, len(color) if isinstance(color, (list, tuple)) else 0,
               "material.color")
        if len(c) == 4:
            c = c[:3]
        elif len(c) != 3:
            raise RecipeError("material.color 须为 RGB 或 RGBA，实为 %r" % (color,))
        for i, x in enumerate(c):
            if x < 0.0 or x > 1.0:
                raise RecipeError("material.color[%d] 须在 0..1（线性空间），实为 %r" % (i, x))
        color_source = "spec"

    if roughness is None:
        r = round(float(d["roughness"]), QUANT)
        rough_source = "role_default"
    else:
        r = _clamp01(roughness, "material.roughness")
        rough_source = "spec"

    if role in ("glass", "emissive"):
        m = 0.0                                   # 玻璃/自发光谈"金属度"没有意义，强制 0（不静默沿用）
        metal_source = "forced_zero"
    elif metallic is None:
        m = round(float(d["metallic"]), QUANT)
        metal_source = "role_default"
    else:
        m = _clamp01(metallic, "material.metallic")
        metal_source = "spec"

    if style == "npr":
        rec = _npr_recipe(role, d, c, r, bands=bands, shadow=shadow)
    else:
        rec = _pbr_recipe(role, d, c, r, m, bool(detail))

    rec["resolved"] = {"base_color": list(c), "roughness": r, "metallic": m,
                       "sources": {"color": color_source, "roughness": rough_source,
                                   "metallic": metal_source}}
    # 复用件路由：把 spec 覆写值透传给 runtime/material.py 的 preset（None = runtime 无此件）
    hint = ROLE_RUNTIME.get(role)
    if style == "pbr" and hint:
        params = dict(hint.get("params") or {})
        params["base_color"] = [c[0], c[1], c[2], 1.0]
        params["roughness"] = r
        params["metallic"] = m
        if role == "glass":
            params["base_color"] = [c[0], c[1], c[2], 1.0]
        if role == "emissive":
            params["color"] = [c[0], c[1], c[2], 1.0]
        rec["runtime"] = {"module": "runtime.material", "fn": "material_build",
                          "preset": hint["preset"], "params": params}
    else:
        rec["runtime"] = None
    rec["runtime_fallback_reason"] = (None if rec["runtime"] else
                                      ("runtime/material.py 无 %s 的等价 preset（见 ROLE_RUNTIME 注释）"
                                       % role if style == "pbr"
                                       else "NPR 色阶（Shader to RGB + ColorRamp）runtime 无此件"))
    rec["digest"] = recipe_digest(rec)
    validate_recipe(rec)                          # 自查：build 出来的配方必须自洽
    return rec


def outline_recipe(color=(0.010, 0.010, 0.012), thickness_m=0.006, style="npr"):
    """轮廓描边（反壳法）的材质配方：Backface Culling + Emission（不受光 → 恒定的黑边）。

    为什么用反壳法而不是"节点里做边缘检测"：
      · 节点法的"描边宽度"是屏幕空间 Fresnel，粗细随视角漂移，**无法给出稳定的像素占比判据**；
      · 反壳法是**几何**宽度（Solidify 的 thickness_m），描边在图上的像素占比可解析预测
        ≈ 2*thickness_m / 物体特征尺寸 × 前景像素数，因此"描边像素占前景比例"才是个可信判据。
    返回的 props 里 `use_backface_culling=True` 是关键 —— 反壳的正面朝内，只有背面可见。
    """
    # #2101（姊妹面）：validate_recipe 只查 style 闭集不查 thickness_m——负宽度
    # 是反壳 Solidify 的反语义输入（修前照常入库），入口 fail-closed。
    try:
        _t = float(thickness_m)
    except (TypeError, ValueError):
        raise RecipeError("outline.thickness_m 须为有限数值，实为 %r" % (thickness_m,))
    if not math.isfinite(_t) or _t <= 0:
        raise RecipeError("outline.thickness_m 须为正有限值，实为 %r" % (thickness_m,))
    n = [
        {"id": "em", "type": "ShaderNodeEmission", "loc": [300, 0],
         "inputs": {"Color": list(_v(color, 3, "outline.color")), "Strength": 1.0}},
        {"id": "out", "type": "ShaderNodeOutputMaterial", "loc": [520, 0], "inputs": {}},
    ]
    l = [["em", "Emission", "out", "Surface"]]
    rec = {"schema": MATERIAL_SCHEMA, "role": "_outline", "style": style,
           "nodes": n, "links": l,
           "props": {"use_backface_culling": True, "surface_render_method": "DITHERED"},
           "outline": {"thickness_m": round(float(thickness_m), QUANT),
                       "method": "inverted_hull"}}
    rec["digest"] = recipe_digest(rec)
    # #2101：自检的 (False, reason) 修前被丢弃——任意 style 字符串 / 负 thickness
    # 静默返回带 digest 的无效配方，模块级「越界 ⇒ RecipeError（fail-closed）」
    # 承诺被破坏（实测 style='pbr2' 照常入库）。现对自检结果 fail-closed。
    _ok, _why = validate_recipe(rec)
    if not _ok:
        raise RecipeError("outline.%s" % _why)
    return rec


# ═══════════════════════════════════════════════════════════════ spec 驱动
def material_plan(spec, *, style="pbr", detail=True, bands=3, shadow=0.30, roles=None):
    """按 spec 产出**逐部件**的材质计划（纯函数；不碰 bpy）。

    返回
      {
        "schema": "wmshade.material/1", "spec_digest": <spec 内容摘要>,
        "style": ..., "entries": [ {link, part, role, color, roughness, metallic,
                                    recipe_digest, recipe} ],
        "by_role": {role: [条目下标...]}, "distinct_recipes": N
      }
    """
    from . import material as _self  # 自引用（只用于文档/静态分析友好）
    try:
        from ..geo import spec as _spec
    except Exception:                                   # 允许把 shade/ 单独拷出来用
        _spec = None

    if _spec is None:
        raise RecipeError("找不到 wm_bridge.factory.geo.spec（material_plan 需要冻结契约）")
    n = _spec.normalize_spec(spec, _copy=True)
    if roles is None:
        roles = None
    entries = []
    by_role = {}
    seen = {}
    for li, lk in enumerate(n["links"]):
        for p in lk["parts"]:
            m = p["material"]
            role = m["role"]
            if roles is not None and role not in roles:
                continue
            # #2114：spec 专门声明的 emission/emission_strength 修前被 material_plan
            # 完全丢弃（不进 key、不进 entries）——spec 加字段并写明理由（「发光件
            # 由 spec 声明」），shade 计划面让默认值赢过一切声明属镜像面漂移。现
            # 随条目透传并参与去重 key（声明不同的两部件不再被合并成同一计划条）；
            # recipe 节点图对 emission 的消费属执行面（bpy_shade），本票不动。
            em, em_s = m.get("emission"), m.get("emission_strength")
            key = (role, tuple(m["color"]) if m["color"] else None,
                   m.get("roughness"), m.get("metallic"), em, em_s,
                   style, bool(detail), bands, shadow)
            if key not in seen:
                seen[key] = material_recipe(
                    role, m["color"], m.get("roughness"), m.get("metallic"),
                    style=style, detail=detail, bands=bands, shadow=shadow)
            rec = seen[key]
            entries.append({
                "link": lk["name"], "part": p["name"], "role": role,
                "color": list(m["color"]) if m["color"] else None,
                "roughness": m.get("roughness"), "metallic": m.get("metallic"),
                "emission": em, "emission_strength": em_s,
                "recipe_digest": rec["digest"], "recipe": rec,
            })
            by_role.setdefault(role, []).append(len(entries) - 1)
    return {
        "schema": MATERIAL_SCHEMA, "spec_digest": _spec.spec_digest(n), "style": style,
        "entries": entries, "by_role": by_role, "distinct_recipes": len(seen),
    }


# ═══════════════════════════════════════════════════════════════ 校验 / 摘要
def validate_recipe(recipe):
    """只读体检：返回 `(ok: bool, reason: str)`。**不抛异常、不改入参。**"""
    try:
        _check(recipe)
    except RecipeError as e:
        return False, str(e)
    except Exception as e:                                   # noqa: BLE001
        return False, "非预期错误：%s: %s" % (type(e).__name__, e)
    return True, ""


def _check(r):
    if not isinstance(r, dict):
        raise RecipeError("配方须为对象，实为 %s" % type(r).__name__)
    if r.get("schema") != MATERIAL_SCHEMA:
        raise RecipeError("schema 不符：期望 %r，实为 %r" % (MATERIAL_SCHEMA, r.get("schema")))
    if r.get("style") not in STYLES:
        raise RecipeError("style 不在闭集：%r" % (r.get("style"),))
    nodes = r.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise RecipeError("nodes 须为非空列表")
    ids = set()
    for i, nd in enumerate(nodes):
        w = "nodes[%d]" % i
        if not isinstance(nd, dict):
            raise RecipeError("%s 须为对象" % w)
        nid = nd.get("id")
        if not isinstance(nid, str) or not nid:
            raise RecipeError("%s.id 须为非空字符串，实为 %r" % (w, nid))
        if nid in ids:
            raise RecipeError("节点 id 重复：%r" % (nid,))
        ids.add(nid)
        if nd.get("type") not in NODE_TYPES:
            raise RecipeError("%s.type 不在闭集内：%r" % (w, nd.get("type")))
        if "ramp" in nd:
            rp = nd["ramp"]
            if rp.get("interpolation") not in ("CONSTANT", "LINEAR"):
                raise RecipeError("%s.ramp.interpolation 须为 CONSTANT/LINEAR" % w)
            els = rp.get("elements")
            if not isinstance(els, list) or len(els) < 2:
                raise RecipeError("%s.ramp.elements 须至少 2 项" % w)
            prev = None
            for j, e in enumerate(els):
                if not isinstance(e, (list, tuple)) or len(e) != 2:
                    raise RecipeError("%s.ramp.elements[%d] 须为 [位置, RGBA]" % (w, j))
                pos, col = e
                if isinstance(pos, bool) or not isinstance(pos, (int, float)):
                    raise RecipeError("%s.ramp.elements[%d] 位置须为数值" % (w, j))
                if prev is not None and pos <= prev:
                    raise RecipeError("%s.ramp.elements 位置须严格递增" % w)
                prev = pos
                if not isinstance(col, (list, tuple)) or len(col) not in (3, 4):
                    raise RecipeError("%s.ramp.elements[%d] 颜色须为 RGB/RGBA" % (w, j))
    links = r.get("links")
    if not isinstance(links, list):
        raise RecipeError("links 须为列表")
    for i, lk in enumerate(links):
        w = "links[%d]" % i
        if not isinstance(lk, (list, tuple)) or len(lk) != 4:
            raise RecipeError("%s 须为 [from_id, from_socket, to_id, to_socket]" % w)
        a, asock, b, bsock = lk
        if a not in ids:
            raise RecipeError("%s 的源节点 %r 不存在" % (w, a))
        if b not in ids:
            raise RecipeError("%s 的目标节点 %r 不存在" % (w, b))
        if not isinstance(asock, str) or not isinstance(bsock, str):
            raise RecipeError("%s 的 socket 名须为字符串" % w)
    produce = [nid for nid in ids
               if any(nid == lk[2] and lk[3] == "Surface" for lk in links)]
    if not produce:
        raise RecipeError("没有节点连到 Surface（材质会渲染成空）")
    return True


def recipe_digest(recipe):
    """配方的内容摘要（sha256 hex）。**与文件路径、调用时刻无关。**"""
    r = {k: v for k, v in recipe.items() if k not in ("digest",)}
    return hashlib.sha256(canonical_json(r).encode("utf-8")).hexdigest()


def recipe_summary(recipe):
    """人读摘要（不参与哈希）。"""
    from collections import Counter
    ok, why = validate_recipe(recipe)
    cnt = Counter(n["type"] for n in (recipe.get("nodes") or []))
    return {
        "schema": recipe.get("schema"), "role": recipe.get("role"),
        "style": recipe.get("style"), "valid": ok, "invalid_reason": why or None,
        "nodes": len(recipe.get("nodes") or []), "links": len(recipe.get("links") or []),
        "node_kinds": dict(sorted(cnt.items())),
        "digest": recipe.get("digest") or recipe_digest(recipe),
        "resolved": recipe.get("resolved"),
    }
