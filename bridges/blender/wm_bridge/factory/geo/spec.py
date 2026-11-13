# -*- coding: utf-8 -*-
"""wm_bridge.factory.geo.spec · **统一生产规格** `wmfactory.spec/1`（纯标准库 · 零 bpy）

★ 这是四条产线（建模 / 绘画 / SVG / 动画）的**唯一输入契约**。
  一条 spec 同时驱动：几何生成 → 材质赋值 → 渲染 → 矢量导出 → 动画与真值导出。
  这样"同一件东西"在四处的姿态/尺寸/颜色**不会各自漂移**——这正是六轮返工里
  反复出现的病根（取帧泄漏、parts 姿态与 oracle 不一致、坐标零点口径不一）。

-------------------------------------------------------------------------- 坐标口径（必须与 geom.py 一致）
  · spec 里的**所有坐标/尺寸/旋转都在 Blender 空间（Z-up）**——因为 spec 是拿去
    **驱动 Blender** 的，而 Blender 是权威场景。
  · 跨到 lingshu 的换算**只在 `geom.to_lingshu` / `geom.axis_rotate` 一处发生**
    （真旋转 (x,y,z)->(x,z,-y)，det=+1）。本模块**不做**换算，也不重复实现。
  · 单位：米（1 场景单位 = 1 m，与 OBJ 头声明一致）。

-------------------------------------------------------------------------- 为什么要有 spec（而不是让脚本各写各的）
  旧 `hex-model/` 的 13 个脚本里，模型是**硬编码**的（`build_models.py` 直接
  `mk("LAMP_arm_%d", cyl(...))`）。后果：
    · 同一模型只存在于代码里，"它到底该是什么样"没有**可就地核对的声明**；
    · 换一个模型 = 改代码，规格（尺寸/关节/材质）无法单独评审；
    · 四条产线各自读代码，口径必然漂移。
  spec 把"该是什么样"从代码里**提出来**，成为可校验、可 diff、可存档的一份数据。

-------------------------------------------------------------------------- 关节语义
  `joint` 描述这个部件相对**父 link** 如何运动。`kind` 是**闭集**（拒收未知值）：
    static    不动（默认）
    spin      绕轴匀速转（period_s = 转一圈的秒数）
    swing     绕轴往复摆（period_s 一圈，amp_deg 振幅）
    prismatic 沿轴平移（amp_m 行程）
    ease      非周期缓动（给 keys 关键帧：[[t, value], ...]，value 单位见字段）
  这样"运动"是**声明**而非"某段代码碰巧跑了什么"——真值导出与渲染共用同一份声明。

-------------------------------------------------------------------------- 硬纪律（与 wm_bridge 一致）
  1. **纯标准库**，不 import bpy / numpy —— 本模块可在无 Blender 机器上完整自测。
  2. **不静默**：任何非法输入 ⇒ `SpecError`（带字段路径）；`validate_spec` 是
     不抛异常的查询面，两者都要有。
  3. **确定性**：`spec_digest()` 只依赖内容（键排序 + 数值量化），与文件路径无关。
"""
from __future__ import annotations

import hashlib
import math

#: spec 契约标识（写进存档与所有下游产物）
SPEC_SCHEMA = "wmfactory.spec/1"

#: 允许的关节类型（闭集）
JOINT_KINDS = ("static", "spin", "swing", "prismatic", "ease")

#: 关节插值语义（闭集）。`linear` = 匀速；`bezier` = Blender 默认缓入缓出（复现既有件用）
JOINT_EASINGS = ("linear", "bezier")

#: 允许的材质角色（绘画产线的输入面；角色→节点组的映射在 shade 侧）
MATERIAL_ROLES = (
    "metal", "painted_metal", "rubber", "glass", "emissive",
    "plastic", "wood", "stone", "fabric", "skin",
)

#: 数值量化位数（与 determinism 一致：导出量子 = 1e-6）
QUANT = 6

#: 合法的轴向名
AXES = ("x", "y", "z")

__all__ = [
    "SPEC_SCHEMA", "JOINT_KINDS", "JOINT_EASINGS", "MATERIAL_ROLES", "AXES", "SpecError",
    "validate_spec", "normalize_spec", "spec_digest", "spec_summary",
    "iter_parts", "iter_links", "joint_at",
]


class SpecError(Exception):
    """spec 不合法（fail-closed：不猜、不补、不静默降级）。"""

#: 各层的**声明键**白名单。违反即拒 —— 这是 fail-closed 的落点。
#: 为什么必须有：`fe(独立复核者)` 实测到把 `rot`（应为 `rot_deg`）这类**拼错的键名**
#: 会被 `dict.get` 静默忽略 ⇒ **几何/旋转无声改变**，而 spec 自称"不静默"。
#: 这是"契约不成立的字段被下游静默消费"的同一族病（见 contract.py 的立意）。
_KEYS_SPEC = ("schema", "name", "scene", "links", "joints", "notes")
_KEYS_SCENE = ("fps", "frame_start", "frame_end", "world_size", "units", "seed")
_KEYS_LINK = ("name", "anchor", "parent", "parts")
_KEYS_PART = ("name", "kind", "pos", "rot_deg", "size", "material", "axis",
              "segments", "segments_v", "bevel", "material_name")
_KEYS_MATERIAL = ("role", "color", "roughness", "metallic", "emission", "emission_strength")
_KEYS_JOINT = ("link", "kind", "axis", "anchor", "period_s", "amp_deg", "amp_m",
               "phase_deg", "keys", "unit", "easing", "bezier_step")


def _reject_unknown(d, allowed, where):
    """未声明的键 ⇒ raise。生效条件：`d` 是 dict 且出现白名单外的键。"""
    if not isinstance(d, dict):
        return
    extra = sorted(set(d) - set(allowed))
    if extra:
        raise SpecError(
            "%s 出现未声明的键：%s —— 允许的键是 %s。"
            "（拼错的键名会被 dict.get 静默忽略、几何/旋转无声改变，故 fail-closed）"
            % (where, ", ".join(extra), ", ".join(sorted(allowed))))


def _opt_str(d, key, default, where):
    """可选字符串：**缺键或显式 null ⇒ 用默认值**；给了别的类型或空串 ⇒ 拒。

    ⚠ 历史缺陷：原实现写 `d.get(key) or default`，于是 `units=""`、`kind=""` 这类
      **非法值会被 falsy 吞掉、静默变成默认值**。这里改成显式 `is None` 判定。
    """
    if key not in d or d[key] is None:
        return default
    v = d[key]
    if not isinstance(v, str) or not v.strip():
        raise SpecError("%s.%s 须为非空字符串，实为 %r" % (where, key, v))
    return v.strip()


def _opt_vec3(d, key, default, where):
    """可选三分量：**缺键或显式 null ⇒ 默认值**；给了别的（含 falsy 的 `[]`/`0`）⇒ 交给 `_vec3` 拒。"""
    if key not in d or d[key] is None:
        return _vec3(default, "%s.%s" % (where, key))
    return _vec3(d[key], "%s.%s" % (where, key))


def _opt_list(d, key, where):
    """可选列表：缺键或显式 null ⇒ `[]`；给了非列表（如 `0`/`""`）⇒ 拒。"""
    if key not in d or d[key] is None:
        return []
    v = d[key]
    if not isinstance(v, list):
        raise SpecError("%s.%s 须为列表，实为 %s" % (where, key, type(v).__name__))
    return v




# ─────────────────────────────────────────────────────────── 基础校验工具
def _is_num(v):
    """有限数值（bool 不算 —— bool 是 int 子类，放行会让 `True` 当成 1 米）。"""
    if isinstance(v, bool):
        return False
    return isinstance(v, (int, float)) and math.isfinite(float(v))


def _req_num(d, key, where):
    if key not in d:
        raise SpecError("%s 缺必填数值字段 %r" % (where, key))
    v = d[key]
    if not _is_num(v):
        raise SpecError("%s.%s 须为有限数值，实为 %r" % (where, key, v))
    return float(v)


def _opt_num(d, key, default, where):
    if key not in d or d[key] is None:
        # ⚠ 必须 `float(default)`：否则缺省路径返回 **int**、显式路径返回 **float**，
        # 同一个值在两次 normalize 之间**类型不同** ⇒ `spec_digest` **不幂等**
        # （`{"fps":24}` 与 `{"fps":24.0}` 哈希不同）。由 `anim-seq` 报告、Lead 定位。
        return float(default)
    v = d[key]
    if not _is_num(v):
        raise SpecError("%s.%s 须为有限数值，实为 %r" % (where, key, v))
    return float(v)


def _vec3(v, where):
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        raise SpecError("%s 须为长度 3 的序列，实为 %r" % (where, v))
    out = []
    for i, x in enumerate(v):
        if not _is_num(x):
            raise SpecError("%s[%d] 须为有限数值，实为 %r" % (where, i, x))
        out.append(float(x))
    return tuple(out)


# ─────────────────────────────────────────────────────────── 校验（不抛异常的查询面）
def validate_spec(spec):
    """校验一份 spec；返回 `(ok: bool, reason: str)`。**不抛异常、不改入参。**

    与 `normalize_spec` 的关系：本函数是**只读体检**（给调用方决定怎么办）；
    `normalize_spec` 是**契约执行**（不合法就 raise）。
    """
    try:
        normalize_spec(spec, _copy=True)
    except SpecError as e:
        return False, str(e)
    except Exception as e:                                  # noqa: BLE001
        return False, "非预期错误：%s: %s" % (type(e).__name__, e)
    return True, ""


# ─────────────────────────────────────────────────────────── 规范化（契约执行面）
def normalize_spec(spec, _copy=False):
    """校验并补全默认值；返回**新的**规范化 spec（不修改入参）。

    补全的默认值（每一项都是"不写时的确定含义"，不猜尺寸）：
      part.kind        -> "box"
      part.pos         -> (0,0,0)        （相对**所属 link 锚点**）
      part.rot_deg     -> (0,0,0)        （欧拉 XYZ，度；XYZ 顺序与 Blender 一致）
      part.material.role -> "painted_metal"
      part.material.color -> None        （None ⇒ 由角色决定，不在此处编造颜色）
      link.anchor      -> (0,0,0)        （世界坐标，Blender 空间）
      link.parent      -> None
      joint.kind       -> "static"
      scene.fps        -> 24
      scene.frame_start-> 0
      scene.frame_end  -> 0
      scene.world_size -> 100.0
      scene.units      -> "m"

    生效条件：合法 ⇒ 返回新 dict；不合法 ⇒ raise SpecError（消息含字段路径）。
    """
    if not isinstance(spec, dict):
        raise SpecError("spec 顶层须为对象，实为 %s" % type(spec).__name__)

    if spec.get("schema") != SPEC_SCHEMA:
        raise SpecError("schema 不符：期望 %r，实为 %r" % (SPEC_SCHEMA, spec.get("schema")))

    name = spec.get("name")
    if not isinstance(name, str) or not name.strip():
        raise SpecError("spec.name 须为非空字符串，实为 %r" % (name,))

    # ---- scene
    _reject_unknown(spec, _KEYS_SPEC, "spec 顶层")
    sc_in = spec.get("scene")
    if sc_in is None:
        sc_in = {}
    if not isinstance(sc_in, dict):
        raise SpecError("scene 须为对象，实为 %s" % type(sc_in).__name__)
    _reject_unknown(sc_in, _KEYS_SCENE, "scene")
    fps = _opt_num(sc_in, "fps", 24, "scene")
    if fps <= 0:
        raise SpecError("scene.fps 须 > 0，实为 %r" % (fps,))
    fs = int(_opt_num(sc_in, "frame_start", 0, "scene"))
    fe = int(_opt_num(sc_in, "frame_end", 0, "scene"))
    if fe < fs:
        raise SpecError("scene.frame_end(%d) 不得小于 frame_start(%d)" % (fe, fs))
    scene = {
        "fps": fps,
        "frame_start": fs,
        "frame_end": fe,
        "world_size": _opt_num(sc_in, "world_size", 100.0, "scene"),
        "units": _opt_str(sc_in, "units", "m", "scene"),
    }
    if sc_in.get("seed") is not None:
        if not isinstance(sc_in["seed"], int) or isinstance(sc_in["seed"], bool):
            raise SpecError("scene.seed 须为整数，实为 %r" % (sc_in["seed"],))
        scene["seed"] = int(sc_in["seed"])

    # ---- links
    links_in = spec.get("links")
    if not isinstance(links_in, list) or not links_in:
        raise SpecError("links 须为非空列表")

    seen = set()
    norm_links = []
    for li, lk in enumerate(links_in):
        where = "links[%d]" % li
        if not isinstance(lk, dict):
            raise SpecError("%s 须为对象" % where)
        _reject_unknown(lk, _KEYS_LINK, where)
        lname = lk.get("name")
        if not isinstance(lname, str) or not lname.strip():
            raise SpecError("%s.name 须为非空字符串，实为 %r" % (where, lname))
        if lname in seen:
            raise SpecError("link 名重复：%r（下游按名 join，重复会让对齐产生歧义）" % (lname,))
        seen.add(lname)

        parts_in = _opt_list(lk, "parts", where)
        if not isinstance(parts_in, list):
            raise SpecError("%s.parts 须为列表" % where)
        parts = []
        pseen = set()
        for pi, p in enumerate(parts_in):
            pw = "%s.parts[%d]" % (where, pi)
            if not isinstance(p, dict):
                raise SpecError("%s 须为对象" % pw)
            _reject_unknown(p, _KEYS_PART, pw)
            pn = p.get("name")
            if not isinstance(pn, str) or not pn.strip():
                raise SpecError("%s.name 须为非空字符串，实为 %r" % (pw, pn))
            if pn in pseen:
                raise SpecError("%s 部件名在同 link 内重复：%r" % (pw, pn))
            pseen.add(pn)

            kind = _opt_str(p, "kind", "box", pw)
            if not isinstance(kind, str) or not kind:
                raise SpecError("%s.kind 须为非空字符串，实为 %r" % (pw, kind))

            pos = _opt_vec3(p, "pos", (0, 0, 0), pw)
            rot = _opt_vec3(p, "rot_deg", (0, 0, 0), pw)

            # size：必填（不默认尺寸 —— 猜尺寸是"静默给错"）
            if "size" not in p:
                raise SpecError("%s 缺 size（尺寸不默认，避免静默给错）" % pw)
            size = _vec3(p["size"], "%s.size" % pw)
            if any(s <= 0 for s in size):
                raise SpecError("%s.size 三分量须 > 0，实为 %r" % (pw, size))

            # #2109：`or {}` 把 material=0/''/[] 等显式 falsy 值吞成默认空对象
            # ——模块自己的修复注释（_common.opt_str 一带）已把同款 `x or default`
            # 判为历史缺陷族。显式 None 才回落默认；其余非 dict 值走既有校验。
            mat_in = p.get("material")
            if mat_in is None:
                mat_in = {}
            if not isinstance(mat_in, dict):
                raise SpecError("%s.material 须为对象" % pw)
            _reject_unknown(mat_in, _KEYS_MATERIAL, "%s.material" % pw)
            role = _opt_str(mat_in, "role", "painted_metal", "%s.material" % pw)
            if role not in MATERIAL_ROLES:
                raise SpecError("%s.material.role 不在闭集内：%r（允许 %s）"
                                % (pw, role, ", ".join(MATERIAL_ROLES)))
            color = mat_in.get("color")
            if color is not None:
                c = _vec3(color, "%s.material.color" % pw)
                if any(x < 0.0 or x > 1.0 for x in c):
                    raise SpecError("%s.material.color 须在 0..1（线性空间），实为 %r" % (pw, c))
                color = c
            material = {"role": role, "color": color}
            if mat_in.get("roughness") is not None:
                r = _opt_num(mat_in, "roughness", 0.5, "%s.material" % pw)
                if not (0.0 <= r <= 1.0):
                    raise SpecError("%s.material.roughness 须在 0..1，实为 %r" % (pw, r))
                material["roughness"] = r
            if mat_in.get("metallic") is not None:
                material["metallic"] = _opt_num(mat_in, "metallic", 0.0, "%s.material" % pw)
            # ── G3：自发光（**spec 只声明，节点图映射归 shade 产线**）
            # 原脚本 `M_glow` 的基色 (0.9,0.5,0.06) 与发光色 (1.0,0.5,0.06) **不同**，
            # 单个 `color` 字段装不下 ⇒ 必须能分别声明，否则复现不出既有件的颜色。
            if mat_in.get("emission") is not None:
                em = _vec3(mat_in["emission"], "%s.material.emission" % pw)
                if any(x < 0.0 or x > 1.0 for x in em):
                    raise SpecError("%s.material.emission 须在 0..1（线性空间），实为 %r" % (pw, em))
                material["emission"] = em
            if mat_in.get("emission_strength") is not None:
                es = _opt_num(mat_in, "emission_strength", 0.0, "%s.material" % pw)
                if es < 0.0:
                    raise SpecError("%s.material.emission_strength 须 >= 0，实为 %r" % (pw, es))
                material["emission_strength"] = es

            part = {"name": pn, "kind": kind, "pos": pos, "rot_deg": rot,
                    "size": size, "material": material}
            # 可选：显式轴（圆柱/棱柱用），默认 z
            if p.get("axis") is not None:
                ax = p["axis"]
                if ax not in AXES:
                    raise SpecError("%s.axis 须为 x/y/z 之一，实为 %r" % (pw, ax))
                part["axis"] = ax
            # ── G1：图元分段数（**收进 spec**，因为它直接决定顶点集 ⇒ 决定 OBJ 字节）
            if p.get("segments") is not None:
                sg = p["segments"]
                if isinstance(sg, bool) or not isinstance(sg, int) or sg < 3:
                    raise SpecError("%s.segments 须为 >=3 的整数，实为 %r" % (pw, sg))
                part["segments"] = int(sg)
            if p.get("segments_v") is not None:            # 球/环的第二维（纬向）
                sv = p["segments_v"]
                if isinstance(sv, bool) or not isinstance(sv, int) or sv < 2:
                    raise SpecError("%s.segments_v 须为 >=2 的整数，实为 %r" % (pw, sv))
                part["segments_v"] = int(sv)
            # ── G2：倒角宽度（**收进 spec**：带/不带 bevel 的 ELF 顶点数不同 ⇒ 影响导出几何）
            if p.get("bevel") is not None:
                bv = p["bevel"]
                if not _is_num(bv) or float(bv) < 0:
                    raise SpecError("%s.bevel 须为 >=0 的数值，实为 %r" % (pw, bv))
                part["bevel"] = float(bv)
            # ── G4：OBJ/MTL 的材质名（收进 spec 才能让 spec 模式与既有件逐字节对拍）
            if p.get("material_name") is not None:
                mn = p["material_name"]
                if not isinstance(mn, str) or not mn.strip():
                    raise SpecError("%s.material_name 须为非空字符串，实为 %r" % (pw, mn))
                part["material_name"] = mn.strip()
            norm_parts = part
            parts.append(norm_parts)

        link = {
            "name": lname,
            "anchor": _opt_vec3(lk, "anchor", (0, 0, 0), where),
            "parent": lk.get("parent"),
            "parts": parts,
        }
        if link["parent"] is not None:
            if not isinstance(link["parent"], str):
                raise SpecError("%s.parent 须为字符串或 null" % where)
            link["parent"] = link["parent"].strip()
        norm_links.append(link)

    # 父引用必须存在且不成环（下游按 link 递归姿态，成环会无限递归）
    by_name = {l["name"]: l for l in norm_links}
    for l in norm_links:
        p = l["parent"]
        if p is not None and p not in by_name:
            raise SpecError("link %r 的 parent %r 不存在" % (l["name"], p))
    for l in norm_links:
        seen_chain = set()
        cur = l
        while cur is not None and cur["parent"] is not None:
            if cur["name"] in seen_chain:
                raise SpecError("link 父链成环：%r" % (l["name"],))
            seen_chain.add(cur["name"])
            cur = by_name[cur["parent"]]

    # ---- joints
    joints_in = _opt_list(spec, "joints", "spec")
    if not isinstance(joints_in, list):
        raise SpecError("joints 须为列表")
    joints = []
    for ji, j in enumerate(joints_in):
        jw = "joints[%d]" % ji
        if not isinstance(j, dict):
            raise SpecError("%s 须为对象" % jw)
        _reject_unknown(j, _KEYS_JOINT, jw)
        link_name = j.get("link")
        if link_name not in by_name:
            raise SpecError("%s.link %r 不是已声明的 link" % (jw, link_name))
        kind = _opt_str(j, "kind", "static", jw)
        if kind not in JOINT_KINDS:
            raise SpecError("%s.kind 不在闭集内：%r（允许 %s）"
                            % (jw, kind, ", ".join(JOINT_KINDS)))
        axis = _opt_str(j, "axis", "z", jw)
        if axis not in AXES:
            raise SpecError("%s.axis 须为 x/y/z 之一，实为 %r" % (jw, axis))
        nj = {"link": link_name, "kind": kind, "axis": axis}
        if j.get("anchor") is not None:
            nj["anchor"] = _vec3(j["anchor"], "%s.anchor" % jw)
        else:
            nj["anchor"] = by_name[link_name]["anchor"]

        if kind in ("spin", "swing"):
            p = _req_num(j, "period_s", jw)
            if p <= 0:
                raise SpecError("%s.period_s 须 > 0，实为 %r" % (jw, p))
            nj["period_s"] = p
            if kind == "swing":
                nj["amp_deg"] = _req_num(j, "amp_deg", jw)
            if j.get("phase_deg") is not None:
                nj["phase_deg"] = _opt_num(j, "phase_deg", 0.0, jw)
            # ★ 缓动：**这不是锦上添花，是复现既有件的必要字段**。
            # 实测（geo-modeling 在 hex-model/build_models.py 上量的）：原脚本的 turret
            # 每 6 帧打一次关键帧、用 Blender **默认 BEZIER** 插值 ⇒ tick 1/2/3 的 yaw 是
            #   **2.0126 / 7.4402 / 15.3678 度**，而**匀速**应为 7.5 / 15.0 / 22.5 度。
            # ⇒ 既有资产的"360° 转"**根本不是匀速**。`easing="linear"`（默认）是"匀速"的
            # 精确声明；要复现既有件必须显式写 `easing="bezier"`。
            if j.get("easing") is not None:
                ez = j["easing"]
                if ez not in JOINT_EASINGS:
                    raise SpecError("%s.easing 须为 %s 之一，实为 %r"
                                    % (jw, "/".join(JOINT_EASINGS), ez))
                nj["easing"] = ez
            else:
                nj["easing"] = "linear"
            # ★ bezier 的**离散化步长**也必须收进 spec：
            # 原脚本每 6 帧打一个关键帧（25 键 / 步长 6），`geo-modeling` 从 .blend 的
            # fcurve 里读出 `interpolation="BEZIER"` + `handle_left_type="AUTO_CLAMPED"`。
            # 若步长留在 spec 外（命令行旋钮），则**同一份 spec 会产出不同的 oracle** ——
            # 那违背"spec 是唯一真源"。默认 6 与既有资产一致（不写即复现既有件）。
            if nj["easing"] == "bezier":
                bstep = j.get("bezier_step", 6)
                if isinstance(bstep, bool) or not isinstance(bstep, int) or bstep < 1:
                    raise SpecError("%s.bezier_step 须为 >=1 的整数，实为 %r" % (jw, bstep))
                nj["bezier_step"] = int(bstep)
        elif kind == "prismatic":
            nj["amp_m"] = _req_num(j, "amp_m", jw)
            p = _req_num(j, "period_s", jw)
            if p <= 0:
                raise SpecError("%s.period_s 须 > 0，实为 %r" % (jw, p))
            nj["period_s"] = p
        elif kind == "ease":
            keys = j.get("keys")
            # ⚠ 收 `(list, tuple)`：本函数把 `keys` 存成 **tuple**（不可变更安全），
            # 若只认 `list`，则 `normalize_spec` **不幂等**（二次 normalize 会 raise），
            # 且任何"再 normalize 一次"的调用方（例如 `spec_summary` → `iter_parts`）**必然抛**。
            # 由 `anim-seq` 独立发现（2026-10-14）。
            if not isinstance(keys, (list, tuple)) or len(keys) < 2:
                raise SpecError("%s.keys 须为至少 2 个 [t, value] 的列表" % jw)
            ks = []
            prev_t = None
            for ki, k in enumerate(keys):
                if not isinstance(k, (list, tuple)) or len(k) != 2:
                    raise SpecError("%s.keys[%d] 须为 [t, value]" % (jw, ki))
                t, v = k
                if not _is_num(t) or not _is_num(v):
                    raise SpecError("%s.keys[%d] 须为两个有限数值" % (jw, ki))
                t = float(t)
                if prev_t is not None and t <= prev_t:
                    raise SpecError("%s.keys 的时间必须严格递增（第 %d 项 %r <= %r）"
                                    % (jw, ki, t, prev_t))
                prev_t = t
                ks.append((t, float(v)))
            nj["keys"] = tuple(ks)
            nj["unit"] = _opt_str(j, "unit", "deg", jw)
            if nj["unit"] not in ("deg", "m"):
                raise SpecError("%s.unit 须为 'deg' 或 'm'，实为 %r" % (jw, nj["unit"]))
        joints.append(nj)

    # 一个 link 最多一个关节（多个会让"该 link 的姿态"无定义）
    jl = [j["link"] for j in joints]
    dup = sorted({x for x in jl if jl.count(x) > 1})
    if dup:
        raise SpecError("同一 link 声明了多个关节：%s（该 link 的姿态会无定义）" % dup)

    out = {"schema": SPEC_SCHEMA, "name": name.strip(), "scene": scene,
           "links": norm_links, "joints": joints}
    if spec.get("notes") is not None:
        if not isinstance(spec["notes"], str):
            raise SpecError("notes 须为字符串")
        out["notes"] = spec["notes"]
    return out


# ─────────────────────────────────────────────────────────── 摘要 / 摘要哈希
def spec_digest(spec, digits=QUANT):
    """spec 的内容摘要（sha256 hex）。只依赖内容：`normalize_spec` 后按规范 JSON 哈希。

    ⚠ 与文件路径、mtime、调用时刻**无关** —— 这是"同一份规格"的可核对标识。
    """
    n = normalize_spec(spec, _copy=True)
    payload = _canonical_json(n, digits)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _q(x, digits=QUANT):
    v = round(float(x), digits)
    return 0.0 if v == 0.0 else v


def _canonical_json(obj, digits=QUANT):
    """规范化 JSON 文本（键排序 / LF / 数值量化 / -0.0 归零）。"""
    import json

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


def spec_summary(spec):
    """人读摘要（用于报告与守卫断言，不参与哈希）。"""
    n = normalize_spec(spec)
    nparts = sum(len(l["parts"]) for l in n["links"])
    kinds = {}
    roles = {}
    for _, p in iter_parts(n):
        kinds[p["kind"]] = kinds.get(p["kind"], 0) + 1
        r = p["material"]["role"]
        roles[r] = roles.get(r, 0) + 1
    jk = {}
    for j in n["joints"]:
        jk[j["kind"]] = jk.get(j["kind"], 0) + 1
    return {
        "schema": n["schema"], "name": n["name"],
        "links": len(n["links"]), "parts": nparts, "joints": len(n["joints"]),
        "kinds": kinds, "material_roles": roles, "joint_kinds": jk,
        "fps": n["scene"]["fps"],
        "frames": n["scene"]["frame_end"] - n["scene"]["frame_start"] + 1,
        "digest": spec_digest(n),
    }


# ─────────────────────────────────────────────────────────── 迭代面
def iter_links(spec):
    """按声明序产出 link。"""
    for l in normalize_spec(spec)["links"]:
        yield l


def iter_parts(spec):
    """产出 `(link, part)` 对（按 link 声明序、part 声明序）。"""
    for l in normalize_spec(spec)["links"]:
        for p in l["parts"]:
            yield l, p


def joint_at(spec, link_name):
    """取某 link 的关节声明（无则 None）。"""
    for j in normalize_spec(spec)["joints"]:
        if j["link"] == link_name:
            return j
    return None
