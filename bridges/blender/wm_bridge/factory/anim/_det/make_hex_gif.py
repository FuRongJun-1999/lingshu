# -*- coding: utf-8 -*-
"""make_hex_gif · 六足步态 GIF（**在 Blender 内跑**）

为什么要专门写它：原来的 GIF 是临时脚本生成的，没留档 —— 而它犯了一个**可量化的错**：
48 帧 × 40 ms = **1.920 s**，而步态周期是 **2.000 s** ⇒ **整段动画快了 4.00%**。
（`seq.py` 里本来就写明「GIF delay 是 16 位**厘秒**，毫秒只能是 10 的整数倍，非 10 的倍数
一律显式 raise」——所以毛病不在编码器，而在**调用方选了帧数/延时却不校验总时长**。）

本脚本把那条约束**前置成断言**：`帧数 × 每帧延时 == 周期`，不成立就**拒绝生成**。
"""
import json
import math
import os
import sys
import traceback

def _bridge_root():
    """桥根（含 `wm_bridge/` 的那一级）。

    ⚠ 不数 `..` 的层数：本文件先前在 `_evidence/lead/`，搬进 `wm_bridge/factory/anim/_det/`
    后层数就变了（`..` 从 2 个变成 4 个），**数层数的写法一搬家就断**。向上找标志目录更稳。
    """
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        if os.path.isdir(os.path.join(d, "wm_bridge")):
            return d
        up = os.path.dirname(d)
        if up == d:
            break
        d = up
    raise RuntimeError("找不到桥根（含 wm_bridge/ 的那一级）")


BRIDGE = _bridge_root()
sys.path.insert(0, BRIDGE)

OUT = r"<外部产物根>\wmopt\hexgif"
SIZE = (420, 300)          #: 与旧 GIF 同画幅，便于逐项对比

res = {"out": OUT, "size": list(SIZE)}
try:
    import bpy
    from wm_bridge.factory.anim import bpy_anim as BA
    from wm_bridge.factory.anim import demo as D
    from wm_bridge.factory.anim import seq as SEQ

    spec = D.hero_walker_spec()
    n = int(D.HERO_FRAMES)
    period_ms = int(round(D.GAIT_PERIOD_S * 1000))
    delay = period_ms // n
    #: ⚠ 两条硬断言：延时必须是 10 的整数倍（GIF 厘秒），且总时长必须**精确**等于周期
    assert delay % 10 == 0, "延时必须是 10 ms 的整数倍（GIF 厘秒粒度）：%d" % delay
    assert n * delay == period_ms, ("帧数×延时 != 周期：%d×%d=%d ms vs %d ms"
                                    % (n, delay, n * delay, period_ms))
    os.makedirs(OUT, exist_ok=True)
    ctx = BA.build_scene(spec)
    sc = bpy.context.scene
    sc.render.resolution_x, sc.render.resolution_y = SIZE

    #: ⚠ `build_scene` **不建相机**（它的 docstring 明确不声称建相机）⇒ 渲染前必须自己建，
    #: 否则 `bpy.ops.render.render()` 报 `Error: Cannot render, no camera`（实测踩过）。
    import mathutils
    pts = []
    for ob in sc.objects:
        #: ⚠ 排除地面：对象名就是 **`Ground`**（`add_ground` 里 `objects.new("Ground", …)`，
        #: 尺寸 40 m）。我第一版写成 `startswith("GROUND")`（全大写）⇒ **没排除掉**
        #: ⇒ 取景把 40 m 的地面收进去、模型缩成一个小点（实测 radius=28.3 而机体只有 ~0.7 m）。
        #: 教训：这种"名字大小写"的排除，用**不区分大小写**的匹配，别照抄常量名。
        if ob.type == "MESH" and "ground" not in ob.name.lower():
            for v in ob.data.vertices:
                pts.append(ob.matrix_world @ v.co)
    if not pts:
        raise RuntimeError("场景里没有可用的网格顶点（取景无从下手）")
    lo = mathutils.Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = mathutils.Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    ctr = (lo + hi) * 0.5
    radius = max((hi - lo).length * 0.5, 1e-3)
    cd = bpy.data.cameras.new("hex_cam")
    cd.lens = 42.0
    cam = bpy.data.objects.new("hex_cam", cd)
    #: 3/4 视角：方位 -58°、仰角 14°，距离 2.7×半径（够把六条腿都收进画面）
    az, el, dist = math.radians(-58.0), math.radians(14.0), radius * 2.7
    cam.location = ctr + mathutils.Vector((dist * math.cos(el) * math.cos(az),
                                           dist * math.cos(el) * math.sin(az),
                                           dist * math.sin(el)))
    d = ctr - cam.location
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    sc.collection.objects.link(cam)
    sc.camera = cam
    res["camera"] = {"loc": [round(v, 3) for v in cam.location],
                     "ctr": [round(v, 3) for v in ctr], "radius": round(radius, 3)}
    #: 2026-10-18 外部审计 R13-O01：baked 关节（ease/prismatic/static）的唯一落地路径是
    #: `bake_keyframes` —— 不调用它，渲染出来的是**静置姿态**，与 `gate_gait_contact`
    #: 验证的运动学不是同一套动作（交付物功能错误）。烘焙必须在渲染之前、取景之后。
    res["bake"] = BA.bake_keyframes(ctx, list(range(n)))

    #: ⚠ `render_frames` 返回的是 **dict** `{"paths": [...], "snapshots": {...}}`（**不是列表**）。
    #: #699 F16：旧版写 `paths = BA.render_frames(...)` 后 `len(paths)` 取的是**键数**(=2)、
    #: `for p in paths` 迭代的是**键名** ⇒ `rendered` 恒为 2、`frames` 记成 ["paths","snapshots"]。
    #: 按契约取 `["paths"]`。
    _frames = BA.render_frames(list(range(n)), OUT, prefix="f")
    res["rendered"] = len(_frames["paths"])

    #: ⚠ GIF 组装**不在 Blender 里做**：Blender 的 Python 没有 PIL
    #: （实测 `ModuleNotFoundError: No module named 'PIL'`）。这里只出帧；
    #: 装配交给 venv 侧的 `assemble_hex_gif.py`（那里有 PIL），并把两条断言带上。
    res["frames"] = [os.path.basename(p) for p in _frames["paths"]]
    res["delay_ms"] = delay
    res["expected_total_ms"] = n * delay
    res["ok"] = True
except Exception as e:
    res["ok"] = False
    res["error"] = "%s: %s" % (type(e).__name__, e)
    res["trace"] = traceback.format_exc()[-800:]
print("HEADLESS " + json.dumps(res, ensure_ascii=True))
