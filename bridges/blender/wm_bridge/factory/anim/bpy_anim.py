# -*- coding: utf-8 -*-
"""wm_bridge.factory.anim.bpy_anim · **Blender 侧对拍工具**（`import bpy` 只在函数体内）

定位（**降级为对拍工具**，不是产品路径）
--------------------------------------
用户澄清主线后：世界模型"自己产出动画"那条线在 `eval.py` / `physics.py` / `seq.py`（纯标准库）。
本文件的价值就一条 —— **把 `eval.eval_at` 的解析解落到真 Blender 里，逐帧回读
`matrix_world` 与它对比**，证明"求值器说的姿态"就是"Blender 演的姿态"。

它做三件事（严格一一对应 `eval.py` 的定义，不重复实现运动学）：
    1. `build_scene()`   按 `rig.build_rig` 建**骨骼**（head=枢轴 P_L、tail=head+(0,L,0)、roll=0
                         ⇒ 静置旋转 = 单位阵，这正是与 eval 链式式逐位同构的前提）
    2. **驱动器 / 关键帧**：解析式关节（spin/swing）走真 Blender 驱动器（表达式由 rig.py 生成）；
       `easing="bezier"` 走欧拉通道 + **显式 smoothstep 控制多边形手柄**（= 两键 AUTO_CLAMPED 的形状）；
       其余关节（prismatic/ease/static）逐帧烘焙关键帧。
    3. `pose_snapshot()` 逐帧回读 `arm.matrix_world @ pose.bones[L].matrix` → (pos, quat)

另外提供 `probe_auto_clamped()`：直接量"Blender 默认 BEZIER(AUTO_CLAMPED) 两键段
是不是就是 3x²−2x³" —— 这是 Lead 点名要的核对。

运行（无头）：
    blender.exe -b -P bpy_anim.py -- --spec <spec.json> --outdir <dir> --frames 0..47 --size 560
"""

import argparse
import json
import math
import os
import sys

BONE_MESH_KINDS = ("box", "cyl", "sphere")


class BpyAnimError(Exception):
    """Blender 侧建场/导出失败（fail-closed）。"""


def _bpy():
    try:
        import bpy                                              # noqa: F401
    except ImportError as exc:                                    # pragma: no cover
        raise BpyAnimError("bpy 不可用：本文件必须在 Blender 内运行（%s）" % exc)
    return bpy


def _pkg():
    """确保能 import wm_bridge（脚本可能以 -P 直接从别处启动）。"""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if root not in sys.path:
        sys.path.insert(0, root)


# ---------------------------------------------------------------- 场景构建
def _clear_scene(bpy):
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    for blk in (bpy.data.meshes, bpy.data.armatures, bpy.data.materials,
                bpy.data.cameras, bpy.data.lights):
        for it in list(blk):
            blk.remove(it)


def _role_color(part):
    col = part["material"].get("color")
    if col:
        return (col[0], col[1], col[2], 1.0)
    return (0.55, 0.55, 0.58, 1.0)


def _make_material(bpy, name, part):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    m = part["material"]
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = _role_color(part)
        bsdf.inputs["Roughness"].default_value = float(m.get("roughness", 0.45))
        bsdf.inputs["Metallic"].default_value = float(m.get("metallic",
                                                    0.85 if m["role"] in ("metal",) else 0.1))
        em = m.get("emission")
        if em is not None:
            for key in ("Emission Color", "Emission"):
                if key in bsdf.inputs:
                    bsdf.inputs[key].default_value = (em[0], em[1], em[2], 1.0)
                    break
            if "Emission Strength" in bsdf.inputs:
                bsdf.inputs["Emission Strength"].default_value = float(m.get("emission_strength", 2.0))
    elif m["role"] == "emissive":
        pass
    return mat


def _prim_mesh(bpy, name, kind, size, part):
    """用 bmesh 造图元（**不调 bpy.ops** —— 无头下最稳），缩放/居中后返回 mesh。"""
    import bmesh
    from mathutils import Matrix, Vector

    if kind not in BONE_MESH_KINDS:
        raise BpyAnimError("不支持的部件 kind %r（闭集 %s）" % (kind, ", ".join(BONE_MESH_KINDS)))
    bm = bmesh.new()
    sx, sy, sz = (float(x) for x in size)
    if kind == "box":
        bmesh.ops.create_cube(bm, size=1.0)
        bmesh.ops.scale(bm, vec=Vector((sx, sy, sz)), verts=bm.verts)
    elif kind == "cyl":
        seg = int(part.get("segments") or 20)
        bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=seg,
                              radius1=0.5, radius2=0.5, depth=1.0)
        bmesh.ops.scale(bm, vec=Vector((sx, sy, sz)), verts=bm.verts)
    else:
        u = int(part.get("segments") or 20)
        v = int(part.get("segments_v") or max(8, u // 2))
        bmesh.ops.create_uvsphere(bm, u_segments=u, v_segments=v, radius=0.5)
        bmesh.ops.scale(bm, vec=Vector((sx, sy, sz)), verts=bm.verts)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    del Matrix

    return me


def _euler_xyz_matrix(rx, ry, rz):
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    return ((cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz),
            (cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz),
            (-sy, sx * cy, cx * cy))


def build_scene(spec, morphs=None, name="ANIM", ground=True, lights=True):
    """按 spec + rig 建真骨骼场景：骨骼 / 部件（蒙皮）/ 地面 / **灯光** / 驱动器 / 形态键。

    ‣ `lights=True`（缺省）会调本模块的 `add_lights()` 并铺一层深色世界底色。
      ⚠ 为什么要补这一步（本处实测）：原 docstring 早就把「灯光」列进本函数该建的东西，
      但实现里**从没调用过 `add_lights()`** ⇒ 用本函数建出来的场景**一盏灯都没有**，
      渲染出来近乎全黑（实测：机身只剩轮廓、地面完全看不见），
      正是"到底建了没有"要靠**看图**才能发现的那类缺陷。
    ‣ `ground=True`（缺省）建一块地面（见 `add_ground`）。
    ‣ **相机与渲染设置仍由调用方负责**（`add_camera()` / `setup_render()`）——
      本函数不再在 docstring 里含糊地把它们算作自己的产物。

    返回 `{"armature","bones","rig","norm_spec","parts","morphs","driven_links","fps","frames"}`。
    （`bones` 曾是 `None`，与 docstring 承诺不符；已补为骨骼名列表。）
    """
    _pkg()
    bpy = _bpy()
    from mathutils import Matrix, Vector
    from wm_bridge.factory.anim import eval as E
    from wm_bridge.factory.anim import rig as R

    nspec = E.normalize(spec)
    rdoc = R.build_rig(spec, morphs=morphs)
    fps = nspec["scene"]["fps"]
    _clear_scene(bpy)
    sc = bpy.context.scene
    sc.render.fps = int(round(fps))
    sc.frame_start = int(nspec["scene"]["frame_start"])
    sc.frame_end = int(nspec["scene"]["frame_end"])

    # ---- 骨架
    arm_data = bpy.data.armatures.new(name + "_data")
    arm = bpy.data.objects.new(name, arm_data)
    sc.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    for b in rdoc["bones"]:
        eb = arm_data.edit_bones.new(b["name"])
        eb.head = Vector((b["head"][0], b["head"][1], b["head"][2]))
        eb.tail = Vector((b["tail"][0], b["tail"][1], b["tail"][2]))
        eb.roll = float(b.get("roll") or 0.0)
    for b in rdoc["bones"]:
        if b["parent"]:
            arm_data.edit_bones[b["name"]].parent = arm_data.edit_bones[b["parent"]]
            arm_data.edit_bones[b["name"]].use_connect = False
    bpy.ops.object.mode_set(mode="OBJECT")
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"

    # ---- 件：顶点直接落在**世界 bind 坐标**，用 armature 修改器 + 顶点组 100% 权重
    links = {l["name"]: l for l in nspec["links"]}
    #: link 的**枢轴**（= 该 link 骨骼的 head，也与 `eval_at` 的 `joint_pivot` 同源）
    #: ⚠ 必须加进顶点坐标：本函数的设计是**蒙皮**（顶点落在**世界 bind 坐标**，用
    #: armature 修改器 + 顶点组 100% 权重），而 Blender 的形变是
    #: `M_pose · M_rest⁻¹` —— 顶点若只放"部件在 link 内的偏移 `pos`"而漏掉 link 的锚点，
    #: 就等于让每个部件绕**错误的地方**旋转。实测后果：`body__hull` 的几何中心恰停在
    #: z=0.07（部件局部 z），而它该在 z=0.73；六条腿因此整片飞散、渲染里根本看不到腿。
    joint_of = {j["link"]: j for j in nspec["joints"]}
    parts = []
    for lname, lk in links.items():
        piv = E.joint_pivot(joint_of.get(lname), lk)
        for p in lk["parts"]:
            me = _prim_mesh(bpy, "%s__%s" % (lname, p["name"]), p["kind"], p["size"], p)
            ob = bpy.data.objects.new("%s__%s" % (lname, p["name"]), me)
            sc.collection.objects.link(ob)
            m = p["material"]
            me.materials.append(_make_material(bpy, "%s__%s__mat" % (lname, p["name"]), p))
            rot = [math.radians(float(x)) for x in p.get("rot_deg") or (0, 0, 0)]
            Rm = _euler_xyz_matrix(rot[0], rot[1], rot[2])
            pos = p["pos"]
            for vtx in me.vertices:
                x, y, z = vtx.co
                vtx.co = Vector((
                    piv[0] + pos[0] + Rm[0][0] * x + Rm[0][1] * y + Rm[0][2] * z,
                    piv[1] + pos[1] + Rm[1][0] * x + Rm[1][1] * y + Rm[1][2] * z,
                    piv[2] + pos[2] + Rm[2][0] * x + Rm[2][1] * y + Rm[2][2] * z))
            vg = ob.vertex_groups.new(name=lname)
            vg.add([v.index for v in me.vertices], 1.0, "REPLACE")
            mod = ob.modifiers.new("arm", "ARMATURE")
            mod.object = arm
            mod.use_vertex_groups = True
            ob["anim_link"] = lname
            ob["anim_part"] = p["name"]
            parts.append({"object": ob, "link": lname, "part": p["name"],
                          "kind": p["kind"], "size": p["size"]})
    # ---- 地面：`ground=True` 原来**什么都没建**（`ground` 形参从头到尾没被用过）
    #: 用本模块**已有的** `add_ground()`，不再手搓平面（第一次改的时候手搓的块还误插进了
    #: `_prim_mesh` 里 —— 那里没有 `ground`/`links`/`sc`，一调用就会崩）。
    if ground:
        add_ground(z=0.0)

    # ---- 灯光 + 背景（见 docstring 里那条"看图才发现"的说明）
    #: 背景取**项目自己的** `shade.render.world_recipe`（竖直 3D 渐变 + 冷色背景），
    #: 不自己拍一个近黑常量。为什么（实测 A/B，同机位同几何只差世界）：
    #:   自拍 flat 0.055 ⇒ 平灰背景，主体与背景**没有色相/明度分离**；
    #:   项目 recipe (top 0.055/0.105/0.165, bottom 0.010/0.020/0.034, strength 0.30)
    #:   ⇒ 冷青蓝渐变把**暖色主体**衬出来，正是 recipe 自己写的
    #:   「主体偏暖 ⇒ 背景取冷青蓝；明度也刻意压低一档」那一条意图。
    #: `world_recipe` 是**纯数据**函数（不依赖 bpy），故只在函数内惰性 import；
    #: 拿不到时退回一个中性冷色，**不因为背景取不到就让整场建不出来**。
    if lights:
        add_lights()
        top, bot, strength = (0.055, 0.105, 0.165), (0.010, 0.020, 0.034), 0.30
        try:
            from wm_bridge.factory.shade import render as _R
            rec = _R.world_recipe("hero")
            top = tuple(rec["top_color"][:3])
            bot = tuple(rec["bottom_color"][:3])
            strength = float(rec["strength"])
        except Exception as exc:                                 # noqa: BLE001
            #: 背景配方取不到**不是**致命错误：退回上面的中性冷色继续建场。
            #: 但**必须说话**（本仓 `silent_except_lint` 要求宽 `except` 有
            #: warn/log/print/raise，**注释不算豁免**）。
            print("[warn] world_recipe 不可用，退回中性冷色背景：%s" % exc)
        wd = bpy.data.worlds.new(name + "_world")
        wd.use_nodes = True
        _nt = wd.node_tree
        _nt.nodes.clear()
        _out = _nt.nodes.new("ShaderNodeOutputWorld")
        _bg = _nt.nodes.new("ShaderNodeBackground")
        _bg.inputs[1].default_value = strength
        _nt.links.new(_bg.outputs[0], _out.inputs[0])
        _geo = _nt.nodes.new("ShaderNodeNewGeometry")
        _sep = _nt.nodes.new("ShaderNodeSeparateXYZ")
        _mr = _nt.nodes.new("ShaderNodeMapRange")
        _mr.inputs["From Min"].default_value = -0.2
        _mr.inputs["From Max"].default_value = 1.2
        _ramp = _nt.nodes.new("ShaderNodeValToRGB")
        _ramp.color_ramp.elements[0].position = 0.0
        _ramp.color_ramp.elements[0].color = (bot[0], bot[1], bot[2], 1.0)
        _ramp.color_ramp.elements[1].position = 1.0
        _ramp.color_ramp.elements[1].color = (top[0], top[1], top[2], 1.0)
        _nt.links.new(_geo.outputs["Position"], _sep.inputs[0])
        _nt.links.new(_sep.outputs["Z"], _mr.inputs["Value"])
        _nt.links.new(_mr.outputs[0], _ramp.inputs["Fac"])
        _nt.links.new(_ramp.outputs["Color"], _bg.inputs[0])
        sc.world = wd

    del Matrix

    # ---- 形态键（Basis + 每个 morph 一个 key_block，位移在**世界 bind 坐标**里算）
    morph_info = []
    for m in rdoc["morphs"]:
        ob = next(o["object"] for o in parts
                  if o["part"] == m["part"] and o["link"] == m["link"])
        if ob.data.shape_keys is None:
            ob.shape_key_add(name="Basis", from_mix=False)
        kb = ob.data.shape_keys.key_blocks.get(m["name"]) or ob.shape_key_add(
            name=m["name"], from_mix=False)
        key = ob.data.shape_keys.key_blocks
        base = key["Basis"]
        c = ob["anim_part"]
        del c
        ctr = [0.0, 0.0, 0.0]
        for v in base.data:
            for i in range(3):
                ctr[i] += v.co[i]
        n = len(base.data)
        ctr = [x / n for x in ctr]
        half = [0.0, 0.0, 0.0]
        for v in base.data:
            for i in range(3):
                half[i] = max(half[i], abs(v.co[i] - ctr[i]))
        from wm_bridge.factory.anim import rig as _R
        for i, v in enumerate(base.data):
            local = (v.co[0] - ctr[0], v.co[1] - ctr[1], v.co[2] - ctr[2])
            d = _R.morph_offset(m["kind"], m["amp"], local, half)
            kb.data[i].co = (v.co[0] + d[0], v.co[1] + d[1], v.co[2] + d[2])
        morph_info.append({"object": ob, "morph": m, "blocks": list(key.keys())})

    # ---- 驱动器（解析式关节 → 真 Blender 驱动器）
    driven_links = []
    for d in rdoc["driven"]:
        pb = arm.pose.bones[d["bone"]]
        easing = (E._spec.joint_at(spec, d["link"]) or {}).get("easing") or "linear"
        if easing == "bezier":
            continue                        # bezier 走欧拉+手柄（见 bake_bezier）
        for idx, expr in d["channels"]:
            fc = pb.driver_add("rotation_quaternion", idx)
            fc.driver.type = "SCRIPTED"
            fc.driver.expression = expr
        driven_links.append({"link": d["link"], "mode": "driver",
                             "channels": [c[0] for c in d["channels"]]})

    # ---- 形态键驱动器
    for mi in morph_info:
        m = mi["morph"]
        if not m.get("driven"):
            continue
        kb = mi["object"].data.shape_keys.key_blocks[m["name"]]
        fc = kb.driver_add("value")
        fc.driver.type = "SCRIPTED"
        fc.driver.expression = m["value_expr"]

    #: ⚠ docstring 早就承诺返回 `bones`，实现里一直是 `None`（契约不符）⇒ 补上
    return {"armature": arm, "bones": [b["name"] for b in rdoc["bones"]],
            "rig": rdoc, "norm_spec": nspec, "parts": parts,
            "morphs": morph_info, "driven_links": driven_links, "fps": fps,
            "frames": [int(nspec["scene"]["frame_start"]), int(nspec["scene"]["frame_end"])]}


# ---------------------------------------------------------------- 关键帧
def _bezier_joint_links(nspec):
    """spec 里 `easing="bezier"` 的解析式关节 link 列表（spec 顺序）。

    #2097：rig 层把 bezier 解析式关节归入 **baked**（驱动器表达式只有线性/正弦
    形态，表达不了 eval 的 smoothstep 逐周期缓动）——本模块的 bezier 通道因此
    **不能**再从 `rdoc["driven"]` 里找 bezier 关节，改与 rig 分类解耦、直接读
    spec（修前修后的 rig 文档形态都覆盖）。
    """
    out = []
    for j in nspec["joints"]:
        if (j.get("kind") or "") not in ("spin", "swing"):
            continue
        if (j.get("easing") or "linear") == "bezier":
            out.append(j["link"])
    return out


def bake_keyframes(ctx, frames):
    """把 `eval_at` 的答案落到关键帧：烘焙关节 + bezier 关节（欧拉手柄）+ 非驱动形态键。"""
    _pkg()
    bpy = _bpy()
    from wm_bridge.factory.anim import eval as E
    arm = ctx["armature"]
    rdoc = ctx["rig"]
    nspec = ctx["norm_spec"]
    fps = ctx["fps"]
    joint_of = {j["link"]: j for j in nspec["joints"]}
    baked = {b["link"] for b in rdoc["baked"]}
    driven = {d["link"] for d in rdoc["driven"]}
    n_baked = n_keyed = 0

    for f in frames:
        bpy.context.scene.frame_set(int(f))
        pose = E.eval_at(nspec, int(f))
        for b in rdoc["bones"]:
            lname = b["name"]
            pb = arm.pose.bones[lname]
            if lname in driven:
                j = joint_of.get(lname) or {}
                if (j.get("easing") or "linear") != "bezier":
                    continue                     # 真驱动器算，不烘焙
            if lname in baked or lname in driven:
                pose_l = E._eval_normalized(nspec, int(f), {})
                p = pose_l[lname]
                p_parent = pose_l[b["parent"]] if b["parent"] else None
                # basis = parent^-1 · target（与 eval 的链式式同解）
                if p_parent is None:
                    bq = p["quat"]
                    bt = (p["pos"][0] - b["pivot"][0], p["pos"][1] - b["pivot"][1],
                          p["pos"][2] - b["pivot"][2])
                else:
                    inv = E.quat_conj(p_parent["quat"])
                    bq = E.quat_mul(inv, p["quat"])
                    rel = E.quat_rotate(inv, (p["pos"][0] - p_parent["pos"][0],
                                              p["pos"][1] - p_parent["pos"][1],
                                              p["pos"][2] - p_parent["pos"][2]))
                    d = (b["pivot"][0] - b["parent_pivot"][0],
                         b["pivot"][1] - b["parent_pivot"][1],
                         b["pivot"][2] - b["parent_pivot"][2])
                    bt = (rel[0] - d[0], rel[1] - d[1], rel[2] - d[2])
                pb.rotation_quaternion = bq
                pb.location = bt
                pb.keyframe_insert("rotation_quaternion", frame=int(f))
                pb.keyframe_insert("location", frame=int(f))
                n_keyed += 1
                if lname in baked:
                    n_baked += 1
        del pose

    # bezier 关节：欧拉通道 + **显式 smoothstep 控制多边形手柄**
    # #2097：遍历面从 `rdoc["driven"]` 改为 spec 的 bezier 关节（rig 层修后把
    # bezier 归 baked，driven 里不再有它们）。
    n_bez = 0
    for ln in _bezier_joint_links(nspec):
        j = joint_of.get(ln) or {}
        pb = arm.pose.bones[ln]
        pb.rotation_mode = "XYZ"
        axis_idx = {"x": 0, "y": 1, "z": 2}[j.get("axis") or "z"]
        period_f = float(j["period_s"]) * fps
        phase = float(j.get("phase_deg") or 0.0)
        lo = min(frames) - int(math.ceil(period_f)) - 1
        hi = max(frames) + int(math.ceil(period_f)) + 1
        path = "pose.bones[%r].rotation_euler" % ln
        k = math.floor(lo / period_f)
        keys = []
        while k * period_f <= hi:
            keys.append((int(round(k * period_f)), 360.0 * k + phase))
            k += 1
        for f, val in keys:
            sc_f = bpy.context.scene
            sc_f.frame_set(f)
            pb.rotation_euler[axis_idx] = math.radians(val)
            pb.keyframe_insert("rotation_euler", index=axis_idx, frame=f)
        for fc in _action_fcurves(arm):
            if fc.data_path != path or fc.array_index != axis_idx:
                continue
            for kp in fc.keyframe_points:
                kp.interpolation = "BEZIER"
                kp.handle_left_type = kp.handle_right_type = "FREE"
                nxt = None
                prv = None
                for other in fc.keyframe_points:
                    if other.co[0] > kp.co[0] and (nxt is None or other.co[0] < nxt.co[0]):
                        nxt = other
                    if other.co[0] < kp.co[0] and (prv is None or other.co[0] > prv.co[0]):
                        prv = other
                if nxt is not None:                 # 指向右邻的三次控制点（= 两键 AUTO_CLAMPED 形状）
                    dx = (nxt.co[0] - kp.co[0]) / 3.0
                    kp.handle_right = (kp.co[0] + dx, kp.co[1])
                if prv is not None:
                    dx = (kp.co[0] - prv.co[0]) / 3.0
                    kp.handle_left = (kp.co[0] - dx, kp.co[1])
            fc.update()
            n_bez += 1

    # 非驱动形态键：逐帧烘焙
    for mi in ctx["morphs"]:
        m = mi["morph"]
        if m.get("driven"):
            continue
        kb = mi["object"].data.shape_keys.key_blocks[m["name"]]
        from wm_bridge.factory.anim import rig as R
        for f in frames:
            bpy.context.scene.frame_set(int(f))
            kb.value = R.morph_value(m, int(f))
            kb.keyframe_insert("value", frame=int(f))
    return {"baked_keys": n_baked, "keyed_channels": n_keyed, "bezier_channels": n_bez}


# ---------------------------------------------------------------- 回读对拍
def pose_snapshot(frame):
    """回读该帧**骨骼世界位姿** → `{link: {"pos":[…], "quat":[w,x,y,z]}}`（与 eval 同口径）。"""
    bpy = _bpy()
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    bpy.context.scene.frame_set(int(frame))
    bpy.context.view_layer.update()
    out = {}
    mw = arm.matrix_world
    for pb in arm.pose.bones:
        m = mw @ pb.matrix
        t = m.to_translation()
        q = m.to_quaternion()
        out[pb.name] = {"pos": [t.x, t.y, t.z], "quat": [q.w, q.x, q.y, q.z]}
    return out


# ---------------------------------------------------------------- AUTO_CLAMPED 量测
def _action_fcurves(ob):
    """取对象 action 的 fcurve 列表 —— **兼容 Blender 5.x 的分层 Action（slotted Action）**。

    ⚠ 实测（Blender 5.2.2 LTS）：`Action` 对象**没有** `.fcurves` 属性了
    （`AttributeError: 'Action' object has no attribute 'fcurves'`），
    fcurve 挂在 `action.layers[*].strips[*].channelbags[*].fcurves`。旧写法在 4.x 可用、
    5.x 直接崩 —— 这个 helper 两条路径都走，取到就返回。
    """
    act = ob.animation_data.action
    out = list(getattr(act, "fcurves", []) or [])
    if out:
        return out
    for layer in getattr(act, "layers", []) or []:
        for strip in getattr(layer, "strips", []) or []:
            for cb in getattr(strip, "channelbags", []) or []:
                out.extend(list(getattr(cb, "fcurves", []) or []))
    return out


def probe_auto_clamped(fps=24, n=48, total_deg=360.0):
    """量"Blender 默认 BEZIER(AUTO_CLAMPED) 两键段"到底是不是 `3x²−2x³`（Lead 点名要）。"""
    bpy = _bpy()
    _clear_scene(bpy)
    ob = bpy.data.objects.new("probe", None)
    bpy.context.scene.collection.objects.link(ob)
    ob.rotation_mode = "XYZ"
    ob.rotation_euler[0] = 0.0
    ob.keyframe_insert("rotation_euler", index=0, frame=0)
    ob.rotation_euler[0] = math.radians(total_deg)
    ob.keyframe_insert("rotation_euler", index=0, frame=n)
    fcs = _action_fcurves(ob)
    if not fcs:
        raise BpyAnimError('probe_auto_clamped：拿不到 fcurve（Blender 版本差异）')
    fc = fcs[0]
    for kp in fc.keyframe_points:
        kp.interpolation = "BEZIER"
        kp.handle_left_type = kp.handle_right_type = "AUTO_CLAMPED"
    fc.update()
    rows = []
    worst = 0.0
    for f in range(n + 1):
        bpy.context.scene.frame_set(f)
        got = math.degrees(ob.rotation_euler[0])
        x = f / float(n)
        want = total_deg * (x * x * (3.0 - 2.0 * x))
        d = abs(got - want)
        worst = max(worst, d)
        rows.append({"frame": f, "blender_deg": got, "smoothstep_deg": want, "abs_diff": d})
    return {"fps": fps, "n": n, "total_deg": total_deg, "max_abs_diff_deg": worst,
            "rows": rows}


# ---------------------------------------------------------------- 渲染导出
def setup_render(size=(560, 560), samples=32, engine="BLENDER_EEVEE"):
    bpy = _bpy()
    sc = bpy.context.scene
    engines = [i.identifier for i in bpy.types.RenderSettings.bl_rna
               .properties["engine"].enum_items]
    if engine not in engines:
        raise BpyAnimError("引擎 %r 不在本 Blender 的枚举里：%s" % (engine, engines))
    sc.render.engine = engine
    sc.render.resolution_x, sc.render.resolution_y = int(size[0]), int(size[1])
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.film_transparent = False
    if hasattr(sc, "eevee"):
        for attr, val in (("taa_render_samples", int(samples)),
                          ("use_raytracing", True)):
            if hasattr(sc.eevee, attr):
                setattr(sc.eevee, attr, val)
    wd = bpy.data.worlds.new("W") if not bpy.data.worlds else bpy.data.worlds[0]
    sc.world = wd
    wd.use_nodes = True
    bg = wd.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs[0].default_value = (0.045, 0.06, 0.10, 1.0)
        bg.inputs[1].default_value = 1.0
    return sc


def add_camera(center=(0.0, 0.0, 0.55), dist=5.6, height=2.5, lens=50.0):
    bpy = _bpy()
    from mathutils import Vector
    cam_data = bpy.data.cameras.new("Cam")
    cam_data.lens = lens
    cam = bpy.data.objects.new("Cam", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    cam.location = Vector((dist * 0.72, -dist, height))
    direction = Vector(center) - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam
    return cam


def add_lights():
    bpy = _bpy()
    from mathutils import Vector
    specs = [("Key", "AREA", (3.4, -3.6, 4.4), 900.0, 2.2),
             ("Fill", "AREA", (-4.0, -2.6, 2.2), 260.0, 3.0),
             ("Rim", "AREA", (-1.2, 4.2, 3.4), 700.0, 2.0)]
    for name, kind, loc, power, size in specs:
        ld = bpy.data.lights.new(name, type=kind)
        ld.energy = power
        if kind == "AREA":
            ld.size = size
        ob = bpy.data.objects.new(name, ld)
        bpy.context.scene.collection.objects.link(ob)
        ob.location = Vector(loc)
        d = Vector((0.0, 0.0, 0.5)) - ob.location
        ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    sun = bpy.data.lights.new("Sun", type="SUN")
    sun.energy = 1.6
    so = bpy.data.objects.new("Sun", sun)
    bpy.context.scene.collection.objects.link(so)
    so.rotation_euler = (math.radians(52.0), 0.0, math.radians(38.0))


def add_ground(z=0.0, size=40.0):
    bpy = _bpy()
    import bmesh
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=size / 2.0)
    me = bpy.data.meshes.new("Ground")
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new("Ground", me)
    bpy.context.scene.collection.objects.link(ob)
    ob.location = (0.0, 0.0, z)
    mat = bpy.data.materials.new("GroundMat")
    mat.use_nodes = True
    b = mat.node_tree.nodes.get("Principled BSDF")
    if b is not None:
        b.inputs["Base Color"].default_value = (0.10, 0.12, 0.16, 1.0)
        b.inputs["Roughness"].default_value = 0.72
    me.materials.append(mat)
    return ob


def render_frames(frames, out_dir, prefix="frame", snapshots=True):
    """逐帧渲染 PNG（Blender 自己写），可选同时回读位姿快照。"""
    bpy = _bpy()
    os.makedirs(out_dir, exist_ok=True)
    sc = bpy.context.scene
    snaps = {}
    paths = []
    for i, f in enumerate(frames):
        sc.frame_set(int(f))
        p = os.path.join(out_dir, "%s_%04d.png" % (prefix, i))
        sc.render.filepath = p
        bpy.ops.render.render(write_still=True)
        paths.append(p)
        if snapshots:
            snaps[str(int(f))] = pose_snapshot(int(f))
    return {"paths": paths, "snapshots": snaps}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="spec JSON 路径")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--frames", default="0-47", help="如 0-47 或 0,2,4")
    ap.add_argument("--size", type=int, default=560)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--prefix", default="frame")
    ap.add_argument("--morphs", default="hero", help="hero|none")
    ap.add_argument("--probe-auto-clamped", action="store_true")
    args = ap.parse_args(argv)

    _pkg()
    if "-" in args.frames:
        a, b = args.frames.split("-")
        frames = list(range(int(a), int(b) + 1))
    else:
        frames = [int(x) for x in args.frames.split(",")]

    morphs = None
    if args.morphs == "hero":
        from wm_bridge.factory.anim.demo import hero_morphs
        morphs = hero_morphs()

    with open(args.spec, encoding="utf-8") as f:
        spec = json.load(f)

    ctx = build_scene(spec, morphs=morphs, name="ANIM")
    setup_render(size=(args.size, args.size), samples=args.samples)
    add_camera()
    add_lights()
    add_ground()
    stats = bake_keyframes(ctx, frames)
    res = render_frames(frames, args.outdir, prefix=args.prefix)
    out = {"schema": "wm_bridge.anim/1", "frames": frames, "size": args.size,
           "fps": ctx["fps"], "keyframe_stats": stats,
           "paths": res["paths"], "snapshots": res["snapshots"],
           "blender": bpy.app.version_string}
    if args.probe_auto_clamped:
        try:
            out["auto_clamped_probe"] = probe_auto_clamped(fps=ctx["fps"], n=48)
        except Exception as exc:                                    # noqa: BLE001
            # 探测失败**不掩盖**：记进产物，同时保住快照（快照才是对拍的料）
            out["auto_clamped_probe"] = {"error": "%s: %s" % (type(exc).__name__, exc)}
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "pose_snapshot.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print("BPYANIM " + json.dumps({"ok": True, "n_frames": len(frames),
                                   "keyframe_stats": stats,
                                   "outdir": args.outdir}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    import bpy                                                  # noqa: F401
    _argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    sys.exit(main(_argv))
