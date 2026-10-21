# -*- coding: utf-8 -*-
"""wm_bridge.factory.anim.rig · **骨骼 / 驱动器 / 形态键的声明式定义与校验**（纯标准库 · 零 bpy）

`spec` 是**冻结的输入契约**（`wmfactory.spec/1`，见 `factory/geo/spec.py`），它只声明
「哪些 link、怎么动（joints）」——**没有**骨骼、驱动器、形态键这些 Blender 侧概念，
也不该有（spec 要驱动四条产线，不该为动画一侧长出 Blender 专有字段）。

本模块就是那层**翻译**：把 spec 的 `links` + `joints` 编译成一份
`wmfactory.rig/1` 文档 —— 它有四个面：

    bones    一根骨头一个 link（head = 关节枢轴 P_L、tail = head + (0, L, 0)、roll = 0）
             ⚠ tail 取 +Y 且 roll=0 **不是随手写的**：它让骨头的静置旋转 = 单位阵，
               于是 Blender 的 `pose.matrix = parent.pose.matrix · T(P_L − P_parent) · basis`
               与 `eval.py` 的链式式**逐位同构**（见 eval.py 模块注释）。换一个 tail 方向，
               对拍就会整体偏一个静置旋转 —— 那正是"看着差不多"的陷阱。
    driven   解析式关节（spin / swing）⇒ **真 Blender 驱动器**（表达式由本模块从 joint 生成，
             与 eval.py 共用同一份声明 ⇒ 两边不可能各写一套相位公式）
    baked    prismatic / ease / static ⇒ 逐帧求值后插关键帧（形态不解析或非周期）
    morphs   形态键：spec 没有这个字段 ⇒ 由调用方显式声明（`build_rig(spec, morphs=[…])`），
             本模块只负责校验与生成驱动器表达式

`driver_expression()` 生成的字符串同时被两处消费：
    · `bpy_anim.py` 把它塞进 `driver.expression`（Blender 自己算）
    · 本模块的 `eval_driver_expr()` 用**受限命名空间**在纯 Python 里算同一个字符串
      ⇒ 上下游一致性可以在**没有 Blender** 的机器上被证明（见 `_det/gate_anim.py` G3）。
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import types

from ..geo import spec as _spec
from . import eval as _eval

__all__ = [
    "RIG_SCHEMA", "RigError", "DRIVEN_KINDS", "BAKED_KINDS", "MORPH_KINDS",
    "BONE_TAIL_UP", "build_rig", "validate_rig", "rig_digest", "rig_summary",
    "driver_expression", "driver_property_exprs", "morph_value_expr",
    "eval_driver_expr", "EXPR_NAMESPACE", "morph_value", "morph_offset",
]

#: rig 文档的 schema 标识
RIG_SCHEMA = "wmfactory.rig/1"

#: 走 Blender 驱动器的关节 kind（解析式，能写成表达式）
DRIVEN_KINDS = ("spin", "swing")
#: 逐帧烘焙关键帧的关节 kind
BAKED_KINDS = ("static", "prismatic", "ease")
#: 形态键形变闭集
MORPH_KINDS = ("squash_z", "scale", "skew_x")
#: 骨头 tail 的默认方向（**必须是 +Y**，理由见模块 docstring / eval.py）
BONE_TAIL_UP = (0.0, 1.0, 0.0)

_AXIS_UNIT = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}

#: Blender 驱动器表达式可用的命名空间（`frame` 是驱动器特有的全局量）。
#: 本模块用**同一份**命名空间在纯 Python 里复算表达式 ⇒ 表达式与求值器的一致性可离线证明。
EXPR_NAMESPACE = {
    "pi": math.pi,
    "e": math.e,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "sqrt": math.sqrt,
    "fabs": abs,
    "abs": abs,
    "floor": math.floor,
    "ceil": math.ceil,
    "round": round,
    "radians": math.radians,
    "degrees": math.degrees,
}


class RigError(Exception):
    """rig 声明不合法（fail-closed：不猜、不补、不静默降级）。"""


def _f(x):
    """Blender 表达式里的浮点字面量：`repr` 保证**往返无损**（不截断精度）。"""
    return repr(float(x))


# --------------------------------------------------------------- 驱动器表达式
def driver_expression(joint, fps, where="joint"):
    """解析式关节 → **角度表达式字符串**（弧度为单位的转角）。

    生成规则（与 `eval.joint_value` 的公式一字不差）：
        spin   θ = 2π·(frame/fps)/period_s + radians(phase_deg)
        swing  θ = radians(amp_deg) · sin(2π·(frame/fps)/period_s + radians(phase_deg))
    生效条件：kind ∈ {spin, swing}；否则 `RigError`（这类关节要走烘焙，不该来这里）。
    """
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(float(fps)) or fps <= 0:
        raise RigError("%s：fps 须为正有限值，得到 %r" % (where, fps))
    kind = joint.get("kind")
    p = float(joint["period_s"])
    if p <= 0:
        raise RigError("%s：period_s 须 > 0，得到 %r" % (where, p))
    ph = math.radians(float(joint.get("phase_deg") or 0.0))
    if kind == "spin":
        return "(%s*(frame/%s)/%s + %s)" % (_f(2.0 * math.pi), _f(fps), _f(p), _f(ph))
    if kind == "swing":
        amp = math.radians(float(joint["amp_deg"]))
        return "(%s*sin(%s*(frame/%s)/%s + %s))" % (
            _f(amp), _f(2.0 * math.pi), _f(fps), _f(p), _f(ph))
    raise RigError("%s：kind=%r 不是解析式关节（driven 闭集 %s）"
                   % (where, kind, ", ".join(DRIVEN_KINDS)))


def driver_property_exprs(joint, fps, where="joint"):
    """解析式关节 → `[(quaternion_index, expression), …]`（4 个分量，与 `rotation_mode='QUATERNION'` 配套）。

    四元数 = `(cos(θ/2), sin(θ/2)·axis)` —— 与 `eval.quat_from_axis_angle` 同一构造，
    所以 Blender 驱动器算出来的四元数与求值器**同源**，不是"另写一遍"。
    """
    theta = driver_expression(joint, fps, where)
    half = "0.5*%s" % theta
    axis = _AXIS_UNIT[joint.get("axis") or "z"]
    out = [(0, "cos(%s)" % half)]
    for i, c in enumerate(axis, start=1):
        out.append((i, "sin(%s)*%s" % (half, _f(c))))
    return out


def morph_value_expr(morph):
    """形态键值的驱动器表达式：`0.5 + 0.5·sin(2π·(frame/fps)/period + phase)` ∈ [0, 1]。"""
    fps = float(morph["fps"])
    p = float(morph["period_s"])
    if p <= 0:
        raise RigError("morph %r：period_s 须 > 0" % (morph.get("name"),))
    ph = math.radians(float(morph.get("phase_deg") or 0.0))
    return "(0.5 + 0.5*sin(%s*(frame/%s)/%s + %s))" % (
        _f(2.0 * math.pi), _f(fps), _f(p), _f(ph))


#: 驱动器表达式允许的 AST 节点（算术/比较/条件；**不含**属性访问、lambda、推导式、f-string）。
_DRIVER_AST_ALLOWED = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant,
    ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.IfExp, ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE,
    ast.Gt, ast.GtE, ast.And, ast.Or, ast.Not,
)


def _check_driver_ast(tree, ns, expr):
    """驱动器表达式 AST 白名单校验（**拒绝**属性访问 / lambda / 推导式 / f-string）。

    只查顶层 code object 的 `co_names` 不够：lambda 与推导式会生成**嵌套 code
    object**，它们内部的名字不在外层 co_names 里 —— `(lambda: ().__class__...)` 这类
    表达式能穿过旧的白名单执行任意代码（2026-10-18 外部审计 R13-O02）。因此以 AST 为准：
    只允许算术/比较/条件表达式与**白名单名字的函数调用**；属性访问（`__class__` /
    `__globals__` 之类）一律拒绝。
    """
    for node in ast.walk(tree):
        if not isinstance(node, _DRIVER_AST_ALLOWED):
            raise RigError("驱动器表达式含不允许的语法 %s：%r"
                           % (type(node).__name__, expr))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in ns:
                raise RigError("驱动器表达式调用了未登记的函数：%r" % (expr,))
            if node.keywords:
                raise RigError("驱动器表达式不接受关键字实参：%r" % (expr,))
        elif isinstance(node, ast.Name):
            if node.id not in ns:
                raise RigError("驱动器表达式用了未登记的名字 %r：%r" % (node.id, expr))
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise RigError("驱动器表达式只接受数值常量：%r" % (expr,))


def _check_driver_code_names(code, ns, expr):
    """纵深防御：递归检查 code object 图里的全部名字（含嵌套 code object）。"""
    bad = sorted(n for n in code.co_names if n not in ns)
    if bad:
        raise RigError("驱动器表达式用了未登记的名字 %s：%r" % (bad, expr))
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            _check_driver_code_names(const, ns, expr)


def eval_driver_expr(expr, frame, extra=None):
    """在**受限命名空间**里求值表达式（纯 Python 复算 Blender 驱动器的那条式子）。

    这是"驱动器表达式 = 求值器"这条论断的离线证据：没有 Blender 也能证明两者同值。
    生效条件：表达式只含算术/比较/条件与**白名单名字**（含 `frame` 与 `extra` 里的名字），
    其余一律 `RigError`（AST 白名单 + 递归名字检查 + 无 `__builtins__`，不静默降级）。
    """
    if not isinstance(expr, str):
        raise RigError("驱动器表达式须为字符串，得到 %r" % (type(expr).__name__,))
    ns = dict(EXPR_NAMESPACE)
    ns["frame"] = frame
    if extra:
        ns.update(extra)
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise RigError("驱动器表达式语法错误：%r（%s）" % (expr, exc)) from exc
    _check_driver_ast(tree, ns, expr)
    code = compile(tree, "<driver>", "eval")
    # 纵深防御：AST 白名单已排除嵌套 code object 的产生源，这里仍递归检查一遍名字，
    # 防未来有人放松 AST 白名单时把「只查顶层」的旧病带回来。
    _check_driver_code_names(code, ns, expr)
    return eval(code, {"__builtins__": {}}, ns)  # noqa: S307 —— AST 白名单 + 名字白名单双重强制


# ------------------------------------------------------------------- 形态键形变
def morph_value(morph, frame):
    """形态键在 frame 处的 value ∈ [0, 1]（与 `morph_value_expr` 同式）。"""
    t = float(frame) / float(morph["fps"])
    return 0.5 + 0.5 * math.sin(2.0 * math.pi * t / float(morph["period_s"])
                                + math.radians(float(morph.get("phase_deg") or 0.0)))


def morph_offset(kind, v, local, half):
    """形态键形变量 → 顶点**局部位移**。

    `local` 是顶点相对形变中心的坐标，`half` 是包围盒半尺寸（用于按比例缩放）。
    · squash_z  体积近似守恒的压扁/拉伸（z 压 ⇒ xy 鼓）
    · scale     各向同性胀缩
    · skew_x    绕 y 轴的剪切（沿 x 的位移随 z 线性变化）—— 做"甩动/鞭打"感
    """
    v = float(v)
    x, y, z = local
    if kind == "squash_z":
        s = 1.0 - 0.5 * v                       # v=0 ⇒ 1.0，v=1 ⇒ 0.5
        lat = 1.0 / math.sqrt(s) if s > 1e-6 else 1.0
        return (x * (lat - 1.0), y * (lat - 1.0), z * (s - 1.0))
    if kind == "scale":
        return (x * v, y * v, z * v)
    if kind == "skew_x":
        return (v * z / max(half[2], 1e-6) * half[0] * 0.5, 0.0, 0.0)
    raise RigError("未知形态键 kind %r（闭集 %s）" % (kind, ", ".join(MORPH_KINDS)))


# --------------------------------------------------------------------- 编译面
def build_rig(spec, morphs=None, bone_len=0.25, name=None):
    """spec → `wmfactory.rig/1` 文档（不改入参）。

    `morphs`：形态键声明列表，每项
        {"part": 部件名, "name": 形态键名, "kind": MORPH_KINDS 之一,
         "period_s": 周期秒, "phase_deg": 相位（可选）, "driven": bool（默认 True）}
      `driven=True` ⇒ 该形态键的 value 交给 Blender 驱动器；`False` ⇒ 逐帧烘焙关键帧。
    `bone_len`：骨头长度（只影响 Blender 里的骨头显示与 rest 长度，不影响姿态数学）。
    """
    nspec = _eval.normalize(spec)
    fps = nspec["scene"]["fps"]
    if isinstance(bone_len, bool) or not isinstance(bone_len, (int, float)) or float(bone_len) <= 0:
        raise RigError("bone_len 须为正数，得到 %r" % (bone_len,))

    joints = {j["link"]: j for j in nspec["joints"]}
    links = {l["name"]: l for l in nspec["links"]}
    bones = []
    driven = []
    baked = []
    for l in nspec["links"]:
        nm = l["name"]
        j = joints.get(nm)
        pivot = _eval.joint_pivot(j, l)
        parent = l.get("parent")
        p_pivot = _eval.joint_pivot(joints.get(parent), links[parent]) if parent else (0.0, 0.0, 0.0)
        head = pivot
        tail = (head[0] + BONE_TAIL_UP[0] * bone_len,
                head[1] + BONE_TAIL_UP[1] * bone_len,
                head[2] + BONE_TAIL_UP[2] * bone_len)
        bones.append({"name": nm, "link": nm, "parent": parent,
                      "head": head, "tail": tail, "roll": 0.0,
                      "pivot": pivot, "parent_pivot": p_pivot,
                      "rest_rot": "identity"})
        kind = (j or {}).get("kind") or "static"
        axis = (j or {}).get("axis") or "z"
        if kind in DRIVEN_KINDS:
            driven.append({
                "link": nm, "bone": nm, "kind": kind, "axis": axis,
                "period_s": float(j["period_s"]),
                "phase_deg": float(j.get("phase_deg") or 0.0),
                "amp_deg": float(j["amp_deg"]) if kind == "swing" else None,
                "theta_expr": driver_expression(j, fps, "joints[link=%s]" % nm),
                "channels": driver_property_exprs(j, fps, "joints[link=%s]" % nm),
            })
        else:
            baked.append({"link": nm, "bone": nm, "kind": kind, "axis": axis})

    # ---- 形态键
    parts_by_name = {}
    for l in nspec["links"]:
        for p in l["parts"]:
            parts_by_name[p["name"]] = (l["name"], p)
    nmorphs = []
    seen_morph = set()
    for i, m in enumerate(morphs or []):
        w = "morphs[%d]" % i
        if not isinstance(m, dict):
            raise RigError("%s 须为对象" % w)
        part = m.get("part")
        if part not in parts_by_name:
            raise RigError("%s.part %r 不在 spec 的部件里（可选：%s）"
                           % (w, part, ", ".join(sorted(parts_by_name)[:8]) + " …"))
        name_m = m.get("name")
        if not isinstance(name_m, str) or not name_m.strip():
            raise RigError("%s.name 须为非空字符串" % w)
        key = (part, name_m)
        if key in seen_morph:
            raise RigError("%s：(part=%s, name=%s) 重复声明" % (w, part, name_m))
        seen_morph.add(key)
        kind = m.get("kind")
        if kind not in MORPH_KINDS:
            raise RigError("%s.kind 不在闭集内：%r（允许 %s）" % (w, kind, ", ".join(MORPH_KINDS)))
        period = m.get("period_s")
        if isinstance(period, bool) or not isinstance(period, (int, float)) or float(period) <= 0:
            raise RigError("%s.period_s 须 > 0，得到 %r" % (w, period))
        if "amp" in m and m["amp"] is not None:
            amp = float(m["amp"])
            if not math.isfinite(amp) or amp <= 0:
                raise RigError("%s.amp 须 > 0 有限值，得到 %r" % (w, m["amp"]))
        else:
            amp = 1.0
        rec = {"part": part, "link": parts_by_name[part][0], "name": name_m,
               "kind": kind, "amp": amp, "period_s": float(period),
               "phase_deg": float(m.get("phase_deg") or 0.0), "fps": float(fps),
               "driven": bool(m.get("driven", True))}
        rec["value_expr"] = morph_value_expr(rec)
        nmorphs.append(rec)

    rig = {
        "schema": RIG_SCHEMA,
        "name": name or nspec["name"],
        "fps": float(fps),
        "spec_digest": _spec.spec_digest(spec),
        "bone_len": float(bone_len),
        "bones": bones,
        "driven": driven,
        "baked": baked,
        "morphs": nmorphs,
    }
    ok, why = validate_rig(rig)
    if not ok:
        raise RigError("自产的 rig 没过校验（bug）：%s" % why)
    return rig


def validate_rig(rig):
    """校验 rig 文档；返回 `(ok: bool, reason: str)`。**不抛异常、不改入参。**"""
    try:
        _check_rig(rig)
    except RigError as e:
        return False, str(e)
    except Exception as e:                                   # noqa: BLE001
        return False, "非预期错误：%s: %s" % (type(e).__name__, e)
    return True, ""


def _check_rig(rig):
    if not isinstance(rig, dict):
        raise RigError("rig 顶层须为对象，实为 %s" % type(rig).__name__)
    if rig.get("schema") != RIG_SCHEMA:
        raise RigError("schema 不符：期望 %r，实为 %r" % (RIG_SCHEMA, rig.get("schema")))
    fps = rig.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or float(fps) <= 0:
        raise RigError("fps 须 > 0，实为 %r" % (fps,))
    bones = rig.get("bones")
    if not isinstance(bones, list) or not bones:
        raise RigError("bones 须为非空列表")
    seen = set()
    for i, b in enumerate(bones):
        w = "bones[%d]" % i
        for k in ("name", "head", "tail", "parent"):
            if k not in b:
                raise RigError("%s 缺字段 %r" % (w, k))
        if b["name"] in seen:
            raise RigError("%s.name 重复：%r" % (w, b["name"]))
        seen.add(b["name"])
        for k in ("head", "tail"):
            v = b[k]
            if not isinstance(v, (list, tuple)) or len(v) != 3 or not all(
                    isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))
                    for x in v):
                raise RigError("%s.%s 须为 3 个有限数，实为 %r" % (w, k, v))
        if b["parent"] is not None and b["parent"] not in [x["name"] for x in bones]:
            raise RigError("%s.parent %r 不是已声明的骨头" % (w, b["parent"]))
    for i, b in enumerate(bones):
        seen_chain = set()
        cur = b
        by = {x["name"]: x for x in bones}
        while cur is not None and cur["parent"] is not None:
            if cur["name"] in seen_chain:
                raise RigError("bones 父链成环：%r" % (b["name"],))
            seen_chain.add(cur["name"])
            cur = by[cur["parent"]]
    for i, d in enumerate(rig.get("driven") or []):
        w = "driven[%d]" % i
        if d.get("kind") not in DRIVEN_KINDS:
            raise RigError("%s.kind %r 不在解析式闭集内" % (w, d.get("kind")))
        ch = d.get("channels")
        if not isinstance(ch, list) or sorted(c[0] for c in ch) != [0, 1, 2, 3]:
            raise RigError("%s.channels 必须恰好覆盖四元数四个分量，实为 %r" % (w, ch))
        for _, e in ch:
            eval_driver_expr(e, 0)                  # 表达式必须能在白名单命名空间里求值
        eval_driver_expr(d["theta_expr"], 0)
    for i, d in enumerate(rig.get("baked") or []):
        if d.get("kind") not in BAKED_KINDS:
            raise RigError("baked[%d].kind %r 不在闭集内" % (i, d.get("kind")))
    for i, m in enumerate(rig.get("morphs") or []):
        w = "morphs[%d]" % i
        if m.get("kind") not in MORPH_KINDS:
            raise RigError("%s.kind 不在闭集内：%r" % (w, m.get("kind")))
        if float(m.get("period_s") or 0) <= 0:
            raise RigError("%s.period_s 须 > 0" % w)
        eval_driver_expr(m["value_expr"], 0)
    return True


def _canon(obj):
    def conv(o):
        if isinstance(o, float):
            v = round(o, 9)
            return 0.0 if v == 0.0 else v
        if isinstance(o, dict):
            return {k: conv(o[k]) for k in sorted(o)}
        if isinstance(o, (list, tuple)):
            return [conv(x) for x in o]
        return o
    return json.dumps(conv(obj), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), default=str)


def rig_digest(rig):
    """rig 的内容摘要（sha256 hex，与文件路径/时刻无关）。"""
    return hashlib.sha256(_canon(rig).encode("utf-8")).hexdigest()


def rig_summary(rig):
    """人读摘要（用于报告与守卫断言，不参与哈希）。"""
    return {
        "schema": rig["schema"], "name": rig["name"], "fps": rig["fps"],
        "bones": len(rig["bones"]),
        "driven_kinds": sorted({d["kind"] for d in rig["driven"]}),
        "baked_kinds": sorted({d["kind"] for d in rig["baked"]}),
        "drivers": sum(len(d["channels"]) for d in rig["driven"]),
        "morphs": [{"part": m["part"], "name": m["name"], "kind": m["kind"],
                    "driven": m["driven"]} for m in rig["morphs"]],
        "digest": rig_digest(rig),
    }
