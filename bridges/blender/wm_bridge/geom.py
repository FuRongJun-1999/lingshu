# -*- coding: utf-8 -*-
"""wm_bridge.geom · 几何 / 部件导出（**纯标准库 · 零 bpy**）

本模块把「Blender 侧抽取出来的原始量」确定性转换成下游（lingshu 世界模型）能吃的三件产物：

    mesh.obj + mesh.mtl   身体侧几何：1 场景单位 = 1 m、**Z-up 原样、不做归一化**
    shapes.json           `lingshu.shapes/1` 部件描述（5 图元拟合 / 局部 OBB 尺寸 / 枢轴相对位姿）
    （真值走 oracle.py）

为什么导出必须收进桥：这三件的格式与口径此前散在 `hex-model/build_models.py` 与
`hex-model/export_shapes.py`（Lead 的临时脚本）里——**没有契约、没有守卫、改一处不知道要重测什么**。
本模块只做**纯函数式的转换**：Blender 侧只负责「抽原始量」（见 `bpy_dump.py`），
所以这里可以在没有 Blender 的机器上被完整自测。

---------------------------------------------------------------- 三条硬纪律
1. **轴换算必须是真旋转** `(bx,by,bz) -> (bx, bz, -by)`（`AXIS_MAP`，det = **+1**）。
   反面 `(x, z, y)`（`MIRROR_MAP`）是**镜像**（det = −1）：左右翻转、旋转矩阵行列式变 −1。
   实测（Lead 的 IoU 对照，见 `wm-verify/TEAM-BRIEF.md` §3.2）：
   真旋转 IoU **0.4873** vs 镜像 **0.4787**。位置、尺寸、旋转三样必须用**同一次**换算，
   故本模块把轴换算做成**唯一入口**（`to_lingshu` / `axis_rotate`），并只接受 det = +1 的矩阵。
2. **`ob.dimensions` 不能用**：它是**世界轴对齐**包围盒（世界 AABB），旋转件会**虚胖**
   （小腿 35° 时长度被算成投影后的复合值）。必须用**局部**包围盒 × 缩放 —— 即真 OBB。
   本模块只接受局部 bbox 尺寸，`from_dimensions_aabb()` 这个入口**故意不提供**。
3. **拟合判据是「体积比最接近 1」，不是「比值最大」**。取最大永远落在分母最小的候选
   （圆锥 `π/12·a·b·h`），于是 **108 个网格全被判成圆锥**（Lead 实测）。见 `fit_kind`。

---------------------------------------------------------------- OBJ 头的 fail-closed 理由
载入端 `lingshu.world.world3d.load_obj_mesh` 是 **fail-closed** 的：解析不到
`# normalize: scale=… center=(…)` 就**拒收**整个文件。所以写出侧把这一行当**契约**：
`write_obj` 会断言它在头部（`require_normalize`），少一行即 raise —— 宁可写不出去，
也不要产出一个下游读不进来的 OBJ。

生效条件总览（纯函数，无副作用除写文件）：
  · `write_obj(mtl_path, obj_path, meshes, translate=(0,0,0))` → 摘要 dict；
    头部缺 normalize 行、或 meshes 空、或顶点/面索引非法 ⇒ raise。
  · `write_shapes(path, links, source=…, export_frame=N)` → 摘要 dict；
    part 尺寸含非正值、或轴矩阵 det≠+1、或 export_frame 未给 ⇒ raise。
"""
from __future__ import annotations

import json
import math
import os

# ------------------------------------------------------------------ 常量
#: Blender(Z-up) → lingshu(Y-up / +Z 前向) 的**真旋转**： (x,y,z) -> (x, z, -y)
AXIS_MAP = ((1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, -1.0, 0.0))
#: 反面教材： (x,y,z) -> (x, z, y)。**镜像**（det = −1），只用于自测里证明它必须被拒。
MIRROR_MAP = ((1.0, 0.0, 0.0),
              (0.0, 0.0, 1.0),
              (0.0, 1.0, 0.0))
#: `write_obj` 头部必须逐字出现的一行（载入端 fail-closed 的钥匙）
NORMALIZE_LINE = "# normalize: scale=1.0 center=(0.000000,0.000000,0.000000)"
OBJ_HEAD_COMMENT = ("# wmtest geometry (bridge-side)  1 unit = 1 m, "
                    "Blender Z-up, no normalize")
AXIS_NOTE = "Blender Z-up -> lingshu Y-up: (x,y,z)->(x,z,-y)"
SHAPES_SCHEMA = "lingshu.shapes/1"
#: 部件拟合的图元集合（顺序即 schema 里的允许值）
FIT_KINDS = ("box", "pillar", "sphere", "pyramid")
#: 非 box 只有在「明显更像」时才采用（照抄既有口径，见 fit_kind 的说明）
FIT_MIN_RATIO = 0.65
FIT_BOX_MARGIN = 1.05


#: 姿态核对的容差地板（**导数出精度决定**，不是随手挑的）：
#: 本桥所有旋转/位置都按 **6 位小数**写出（`round(v, 6)`，与既有产物同形），
#: 而核对要把「欧拉/四元数 → 矩阵 → 再相乘」跑一遍，单次换算的误差上限约
#: 半个末位（5e-7），两次复合后可到 ~1e-6。若把阈值压在 1e-6，就会在
#: 「只差最后一粒」处产生**假失败**（实测：Lead 正向端到端时 `arm` 的
#: `ARM_shell/ARM_upper/ARM_upper_glow` 恰好差 0.000001，报了 `part_pose_mismatch`）。
#: 故取 **10 倍导出量子 = 1e-5** 作为地板；而真实缺陷的偏差是**度级**的
#: （位姿取错帧 = 70° ⇒ 矩阵元素差 ~1.15），离地板有两个数量级，
#: 所以放宽地板**不会**让门失效（selftest I2/I5/I6 + 变异 M10 逐条钉死这一点）。
POSE_TOL = 1e-5

#: 数据值（位置/四元数分量）的比对容差：两侧都是 6 位小数，读回无误差，
#: 只有"重算一遍"（quat→mat→quat）才会带上 ~6e-7，故 2e-6 足够。
DATA_TOL = 2e-6


class GeomError(Exception):
    """几何导出的契约违反（fail-closed：不修好就不产出）。"""


# ------------------------------------------------------------------ 3×3 线代（纯标准库）
def mat3_mul(a, b):
    """3×3 矩阵乘（行元组表示）。"""
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
                 for i in range(3))


def mat3_transpose(m):
    return tuple(tuple(m[j][i] for j in range(3)) for i in range(3))


def mat3_det(m):
    """行列式（轴换算的判据：真旋转 +1，镜像 −1）。"""
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))


def apply3(m, v):
    """m · v（列向量；v 为三元组）。"""
    return tuple(sum(m[i][k] * v[k] for k in range(3)) for i in range(3))


def axis_rotate(m3):
    """把 Blender 轴下的旋转矩阵换成 lingshu 轴： `M · R · Mᵀ`（相似变换）。

    生效条件：m3 是 3×3；`AXIS_MAP` 的 det == +1 时才返回，否则 raise
    （det = −1 的映射会把旋转变成「旋转 + 镜像」，行列式与朝向一起坏掉）。
    """
    assert_axis_map()
    return mat3_mul(mat3_mul(AXIS_MAP, tuple(tuple(r) for r in m3)),
                    mat3_transpose(AXIS_MAP))


def to_lingshu(v):
    """(bx,by,bz) -> (bx, bz, -by)（真旋转，唯一入口）。"""
    return apply3(AXIS_MAP, tuple(float(x) for x in v))


def mat3_to_euler_xyz(m):
    """Blender 口径的 `Matrix.to_euler("XYZ")`：解 `R = Rz(z)·Ry(y)·Rx(x)`。

    ⚠ 约定必须与 Blender 一致，否则 `rot` 字段整列都是错的。本实现的正确性由
    **实测数据**锚定：`build_models.py` 的机械臂在最后一帧的链路
    yaw(Rz 35°)·shoulder(Rx −40°) 在 lingshu 轴下是 `Ry(35°)·Rx(−40°)`，
    Blender 给出的欧拉是 `(−40°, 35°, 0)` —— 只有 `Rz·Ry·Rx` 这一种顺序满足
    （`Rx·Ry·Rz` 给的是另一个矩阵）。自测 E3/E4 把这个锚点固化下来。

    ⚠ **还必须复现 Blender 的"两解取小"选择**：同一个矩阵有两组欧拉解，
    Blender 的 `mat3_normalized_to_eulO` 取 **L1 范数更小**的那组
    （源码注释：*return best, which is just the one with lowest values in it*）。
    不这么做会在 `|y| > 90°` 时给出 `(π, −45°, π)` 而不是 `(0, −135°, 0)` ——
    实测踩过：炮塔四个 ±135° 的炮管件因此对不上（本仓 `shapes.json` 记的是后者）。
    """
    m00, m01, m02 = m[0]
    m10, m11, m12 = m[1]
    m20, m21, m22 = m[2]
    cy = math.hypot(m00, m10)
    if cy > 1e-12:
        s1 = (math.atan2(m21, m22), math.atan2(-m20, cy), math.atan2(m10, m00))
        s2 = (math.atan2(-m21, -m22), math.atan2(-m20, -cy), math.atan2(-m10, -m00))
    else:                                    # 万向锁：x 归零，信息全给 z
        a = math.atan2(-m12, m11)
        s1 = s2 = (a, math.atan2(-m20, cy), 0.0)
    a1 = abs(s1[0]) + abs(s1[1]) + abs(s1[2])
    a2 = abs(s2[0]) + abs(s2[1]) + abs(s2[2])
    return s1 if a1 <= a2 else s2


def mat3_from_euler_xyz(e):
    """`Rz(z)·Ry(y)·Rx(x)`（与 `mat3_to_euler_xyz` 互逆）。"""
    x, y, z = (float(v) for v in e)
    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    cz, sz = math.cos(z), math.sin(z)
    return ((cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz),
            (cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz),
            (-sy, sx * cy, cx * cy))


def mat3_to_quat(m):
    """3×3 真旋转矩阵（SO(3)：正交且 det = +1）→ [w,x,y,z]，与 `mathutils.Matrix.to_quaternion()` 同约定。

    用 Shepperd 分支法，取 w ≥ 0 的那一支（180° 附近 ±q 同解，比较时按旋转等价处理）。
    生效条件：m 为 3×3 有限数值矩阵、近似正交（MᵀM = I）且为真旋转（det(M) = +1）；
    否则 raise（含缩放或镜像反射 det = −1 的矩阵没有对应的单位旋转四元数）。
    """
    try:
        raw = tuple(tuple(row) for row in m)
    except TypeError as exc:
        raise GeomError("mat3_to_quat 需要 3×3：%r" % (m,)) from exc
    if len(raw) != 3 or any(len(r) != 3 for r in raw):
        raise GeomError("mat3_to_quat 需要 3×3：%r" % (raw,))
    rows = []
    for r in raw:
        row_f = []
        for x in r:
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
                raise GeomError("mat3_to_quat 收到非有限数值：%r" % (raw,))
            row_f.append(float(x))
        rows.append(tuple(row_f))
    m = tuple(rows)
    tr = m[0][0] + m[1][1] + m[2][2]
    for i in range(3):
        for j in range(3):
            if abs(sum(m[k][i] * m[k][j] for k in range(3)) - (1.0 if i == j else 0.0)) > 1e-6:
                raise GeomError("mat3_to_quat 的输入不是正交矩阵（含缩放？）：%r" % (m,))
    det = mat3_det(m)
    if abs(det - 1.0) > 1e-6:
        raise GeomError(
            "mat3_to_quat 的输入不是真旋转矩阵（镜像反射 det=%.6f≠+1）：%r" % (det, m)
        )
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2][1] - m[1][2]) / s
        y = (m[0][2] - m[2][0]) / s
        z = (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        w = (m[2][1] - m[1][2]) / s
        x = 0.25 * s
        y = (m[0][1] + m[1][0]) / s
        z = (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        w = (m[0][2] - m[2][0]) / s
        x = (m[0][1] + m[1][0]) / s
        y = 0.25 * s
        z = (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
        w = (m[1][0] - m[0][1]) / s
        x = (m[0][2] + m[2][0]) / s
        y = (m[1][2] + m[2][1]) / s
        z = 0.25 * s
    # 规范到「最短弧」：w ≥ 0（q 与 −q 是同一个旋转，但**记哪个**是口径）。
    # 实测锚点：出货 `turret/oracle.jsonl` 288 条里 w<0 的有 0 条；而 Shepperd 的
    # 分支在 yaw ∈ (180°, 360°) 的 23 条上会给出 w<0（即 −q）——不归一就对不上。
    if w < 0.0:
        w, x, y, z = -w, -x, -y, -z
    return [w, x, y, z]


def quat_to_mat3(q):
    """[w,x,y,z] → 3×3（`world_model` 侧真值里的 `rot_quat` 用同一约定）。

    生效条件：q 为 4 个有限数；否则 raise（NaN 会让下面的比较全部静默为 False）。
    """
    if not isinstance(q, (list, tuple)) or len(q) != 4:
        raise GeomError("rot_quat 须为 4 元序列，实为 %r" % (q,))
    for v in q:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise GeomError("rot_quat 含非有限数值：%r" % (q,))
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n <= 0.0:
        raise GeomError("rot_quat 零模长")
    w, x, y, z = w / n, x / n, y / n, z / n
    return ((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)))


def assert_axis_map(m=AXIS_MAP):
    """轴换算矩阵必须是**真旋转**（det = +1）。镜像解直接 raise，不给「尽力而为」。"""
    d = mat3_det(m)
    if abs(d - 1.0) > 1e-9:
        raise GeomError(
            "轴换算是镜像（det=%.6f≠+1）：位置/尺寸/旋转会一起坏掉；"
            "(x,y,z)->(x,z,y) 是反面教材，正解是 (x,y,z)->(x,z,-y)" % d)
    return d


# ------------------------------------------------------------------ OBJ
def obj_head():
    """OBJ 头部三行（含 fail-closed 的 normalize 契约行）。"""
    return [OBJ_HEAD_COMMENT, NORMALIZE_LINE]


def require_normalize(text, where="<obj>"):
    """载入端 fail-closed 的写侧对应物：文本里没有 `# normalize:` 行即 raise。

    生效条件：text 为 str；含 `# normalize:` 子串返回 True，否则 raise GeomError。
    只读，不改 text。
    """
    if not isinstance(text, str):
        raise GeomError("%s：不是文本（%s）" % (where, type(text).__name__))
    if "# normalize:" not in text:
        raise GeomError(
            "%s：缺 `# normalize:` 行 —— lingshu 的 load_obj_mesh 是 fail-closed 的，"
            "会整份拒收（写侧必须在这里先炸，而不是让下游拿到读不进去的文件）" % where)
    return True


def canonical_tris(tris):
    """面序规范化：**保绕向**地把三角形序列变成与输入顺序无关的确定序。

    两步（缺一不可）：
      1. 每个三角形按**环状顺序**旋转，使最小顶点索引打头：
         `k = ix.index(min(ix)); ix = ix[k:] + ix[:k]`
         —— 必须是**环状旋转**，不是 `sorted()`：`sorted()` 会翻转绕向、法线就反了。
      2. 全部三角形按该规范式 `sort()`。

    为什么这步骤是**必须的**（实测根因，Lead 在 `build_models.py` 上定位）：
    Blender 的 `me.calc_loop_triangles()` 对 **ngon**（`cyl(cap_ends=True)` 的端盖这种
    多边形）的三角化结果**按内存地址序迭代**，受 ASLR 影响 ⇒ 同一份输入（顶点流
    逐位相同，最大绝对差 0.000e+00）每跑一次进程，`f` 行的顺序就变一次：
        `lamp/mesh.obj   1e97fa3a… → 69497223… → bf54bbb1…`（三轮三个哈希）
        `turret/mesh.obj d4c0ca77… → 81b5657f… → cdea5e42…`
        `arm/mesh.obj    c2c51ad5… → c2c51ad5… → c2c51ad5…`（稳定：它的件里没有 ngon 端盖）
    ⇒ 教训：**顶点确定 ≠ 文件确定**，面序是独立的一维。不去掉这一维，
    任何"同输入 ⇒ 同文件"的判据都是概率性的（见 selftest 的 J 组与 M5 变异）。
    """
    out = []
    for t in tris:
        ix = [int(i) for i in t]
        if len(ix) != 3:
            raise GeomError("只支持三角面：%r" % (t,))
        k = ix.index(min(ix))
        out.append(tuple(ix[k:] + ix[:k]))
    out.sort()
    return out


def format_obj(meshes, translate=(0.0, 0.0, 0.0), head=None, mtl_name="mesh.mtl"):
    """把 mesh 批次渲染成 OBJ 文本（返回 (text, mtl_rows, summary)）。

    meshes: 可迭代，每项
        {"name": 组名, "verts": [(x,y,z)…], "tris": [(i,j,k)…]（0-based）,
         "material": usemtl 名（缺省 M_default）,
         "materials": [(材质名, [r,g,b])…]（收集进 MTL，去重）}
    口径照抄既有产物：顶点取**世界坐标**、按 `name` 升序、每个网格先写 `v` 再写
    `g`/`usemtl` 再写 `f`（面索引用 1-based 全局顶点偏移）。
    """
    meshes = list(meshes)
    if not meshes:
        raise GeomError("write_obj：没有任何网格（空导出会被下游当成坏件）")
    mats = {}
    lines = list(head if head is not None else obj_head())
    lines.append("mtllib " + mtl_name)
    body = []
    voff = 1
    nv = nf = 0
    for ob in sorted(meshes, key=lambda m: m["name"]):
        verts = [tuple(float(x) for x in v) for v in ob["verts"]]
        tris = [tuple(int(i) for i in t) for t in ob["tris"]]
        if not verts:
            raise GeomError("网格 %s 没有顶点" % ob["name"])
        for v in verts:
            if len(v) != 3 or not all(math.isfinite(c) for c in v):
                raise GeomError("网格 %s 顶点非法：%r" % (ob["name"], v))
            body.append("v %.6f %.6f %.6f" % (v[0] + translate[0],
                                              v[1] + translate[1],
                                              v[2] + translate[2]))
        for nm, col in (ob.get("materials") or []):
            mats[nm] = tuple(float(c) for c in col)
        body.append("g " + ob["name"].replace(" ", "_"))
        body.append("usemtl " + (ob.get("material") or "M_default"))
        for t in canonical_tris(tris):      # ← 规范面序（保绕向），去掉它文件就不可复现
            if len(t) != 3 or any(not (0 <= i < len(verts)) for i in t):
                raise GeomError("网格 %s 面索引越界：%r（顶点数 %d）"
                                % (ob["name"], t, len(verts)))
            body.append("f %d %d %d" % (t[0] + voff, t[1] + voff, t[2] + voff))
        voff += len(verts)
        nv += len(verts)
        nf += len(tris)
    text = "\n".join(lines + body) + "\n"
    require_normalize(text, "write_obj 的产物")
    mtl_rows = []
    for nm in sorted(mats):
        c = mats[nm]
        mtl_rows += ["newmtl " + nm, "Kd %.6f %.6f %.6f" % (c[0], c[1], c[2])]
    return text, mtl_rows, {"vertices": nv, "objects": len(meshes),
                            "faces": nf, "materials": len(mats)}


def write_obj(mtl_path, obj_path, meshes, translate=(0.0, 0.0, 0.0), head=None):
    """写 OBJ + MTL。

    ⚠ 头部恒含 `# normalize: scale=1.0 center=(0.000000,0.000000,0.000000)`
    （`NORMALIZE_LINE`）：**1 场景单位 = 1 m、Z-up 原样、不做归一化**。
    载入端 `load_obj_mesh` 解析不到这行就拒收，故写侧把它当契约断言
    （`require_normalize`）——少一行必须在这里失败。
    """
    text, mtl_rows, summary = format_obj(
        meshes, translate=translate, head=head,
        mtl_name=os.path.basename(mtl_path))
    for p in (mtl_path, obj_path):
        d = os.path.dirname(os.path.abspath(p))
        if d and not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)
    with open(obj_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    with open(mtl_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(mtl_rows) + "\n")
    summary["obj_bytes"] = os.path.getsize(obj_path)
    summary["mtl_bytes"] = os.path.getsize(mtl_path)
    summary["normalize_line"] = NORMALIZE_LINE
    return summary


def parse_obj(text):
    """只读解析 OBJ → {"head": [...], "meshes": [{"name","verts","tris","material"}],
    "mtllib": str}。**先过 `require_normalize`**（fail-closed 与载入端同口径）。

    写侧口径是「先写该网格的 `v`，再写它的 `g`/`usemtl`/`f`」，所以解析时
    顶点是**暂存**的：遇到下一个 `g` 才把暂存顶点归给新组；`f` 的索引是
    **全局 1-based**，解析后折回组内 0-based（与 `format_obj` 互逆，可逐字节往返）。
    """
    require_normalize(text, "parse_obj 的输入")
    meshes, cur, pending = [], None, []
    head, mtllib = [], ""
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            head.append(line)
            continue
        tag, _, rest = line.partition(" ")
        if tag == "mtllib":
            mtllib = rest.strip()
        elif tag == "v":
            pending.append(tuple(float(x) for x in rest.split()))
        elif tag == "g":
            if cur is not None and not cur["verts"] and pending is None:
                raise GeomError("上一个组 %s 没有顶点" % cur["name"])
            cur = {"name": rest.strip(), "verts": pending, "tris": [],
                   "material": "M_default", "_off": 0}
            meshes.append(cur)
            pending = []
        elif tag == "usemtl":
            if cur is None:
                raise GeomError("解析到 usemtl 但还没有 g 组")
            cur["material"] = rest.strip()
        elif tag == "f":
            idx = []
            for tok in rest.split():
                idx.append(int(tok.split("/")[0]) - 1)
            if cur is None or len(idx) != 3:
                raise GeomError("只支持三角面：%r" % line)
            cur["tris"].append(tuple(idx))
    # 把全局面索引折回组内 0-based（组顶点在文件里是连续的）
    off = 0
    for m in meshes:
        if any(i < off or i >= off + len(m["verts"]) for t in m["tris"] for i in t):
            raise GeomError("组 %s 的面索引不在本组顶点范围内（格式不符）" % m["name"])
        m["tris"] = [tuple(i - off for i in t) for t in m["tris"]]
        m.pop("_off", None)
        off += len(m["verts"])
    return {"head": head, "mtllib": mtllib, "meshes": meshes}


# ------------------------------------------------------------------ 部件拟合
def fit_kind(dims, vol):
    """在 5 图元里挑**体积比最接近 1** 的那个 → (kind, ratio)。

    ⚠ 坑一（**判据方向**）：必须是「|ratio − 1| 最小」，**不是「ratio 最大」**。
    取最大永远命中分母最小的候选 —— 圆锥 `π/12·a·b·h` 是四者里最小的分母，
    于是 **108 个网格全被判成圆锥**（Lead 实测）。自测 F5 用「最大解 ≠ 最接近解」
    的构造把这个方向钉死。
    ⚠ 坑二（**尺寸来源**）：dims 必须是**局部**包围盒尺寸 × 缩放（真 OBB）。
    传世界 AABB（`ob.dimensions`）会让旋转件**虚胖**（小腿 35° 时被算成投影复合值），
    于是 volume/dims 比值整体偏小、判据跟着偏 —— 本模块只收局部尺寸，不提供 AABB 入口。

    生效条件：dims 为 3 个非负有限数、vol ≥ 0；dims 任一为 0 时只用 box 候选；
    返回 (kind ∈ FIT_KINDS, ratio 有限正数)。
    """
    dims = tuple(float(d) for d in dims)
    if len(dims) != 3:
        raise GeomError("dims 须为 3 元：%r" % (dims,))
    if any((not math.isfinite(d)) or d < 0 for d in dims):
        raise GeomError("dims 非法（负数/NaN/Inf）：%r" % (dims,))
    vol = float(vol)
    if not math.isfinite(vol) or vol < 0:
        raise GeomError("vol 非法：%r" % (vol,))
    ds = sorted(dims)
    h, a, b = ds[2], ds[0], ds[1]
    d = sum(dims) / 3.0
    box = vol / max(1e-12, dims[0] * dims[1] * dims[2])
    cands = [("box", box)]
    if h > 0 and a > 0 and b > 0:
        cands.append(("pillar", vol / max(1e-12, math.pi / 4 * a * b * h)))
        cands.append(("pyramid", vol / max(1e-12, math.pi / 12 * a * b * h)))
    if d > 0:
        cands.append(("sphere", vol / max(1e-12, math.pi / 6 * d ** 3)))
    kind, ratio = min(cands, key=lambda kv: abs(kv[1] - 1.0))
    # 非 box 只有明显更像才换（避免把规矩的盒子判成柱）——既有口径，逐字保留
    if kind != "box" and (ratio < FIT_MIN_RATIO or ratio < box * FIT_BOX_MARGIN):
        return "box", box
    return kind, ratio


def max_ratio_kind(dims, vol):
    """**反面参照**：按「比值最大」挑会长什么样（自测 F5 用它证明判据方向不能反）。

    只用于自测与诊断，**生产路径不得调用**（调用即 GeomError）。
    """
    ds = sorted(tuple(float(d) for d in dims))
    h, a, b = ds[2], ds[0], ds[1]
    d = sum(ds) / 3.0
    vol = float(vol)
    cands = [("box", vol / max(1e-12, ds[0] * ds[1] * ds[2]))]
    if h > 0 and a > 0 and b > 0:
        cands.append(("pillar", vol / max(1e-12, math.pi / 4 * a * b * h)))
        cands.append(("pyramid", vol / max(1e-12, math.pi / 12 * a * b * h)))
    if d > 0:
        cands.append(("sphere", vol / max(1e-12, math.pi / 6 * d ** 3)))
    return max(cands, key=lambda kv: kv[1])


def part_from_blender(dims_local, vol, world3, world_trans, local_center,
                      anchor_world, color, kind=None):
    """Blender 侧原始量 → 一条 part 记录（轴换算的**唯一**落点）。

    dims_local : 局部包围盒尺寸 × 缩放（真 OBB；**不得**用世界 AABB）
    vol        : 网格体积 × |det(world3x3)|（体积是旋转不变量）
    world3     : Blender 轴下的世界旋转 3×3
    world_trans: Blender 世界平移
    local_center: 网格**局部**包围盒中心（用它做世界中心 = world @ local_center）
    anchor_world: 该 link 枢轴的 Blender 世界平移
    kind       : 缺省 None ⇒ 由 fit_kind 决定；显式给了就只校验 ∈ FIT_KINDS

    返回 {"kind","pos","size","color","rot","_ratio"}；`pos` 相对**枢轴**（`pos_origin="anchor"`），
    `size` 已换轴为 (dx, dz, dy)，`rot` 为 lingshu 轴下的欧拉 XYZ（弧度）。
    """
    dims = tuple(float(d) for d in dims_local)
    if min(dims) <= 0:
        raise GeomError("部件尺寸含非正值：%r（退化件不得进描述）" % (dims,))
    if kind is None:
        kind, ratio = fit_kind(dims, vol)
    else:
        if kind not in FIT_KINDS:
            raise GeomError("未知图元 %r（允许 %s）" % (kind, list(FIT_KINDS)))
        ratio = fit_kind(dims, vol)[1]
    wc = to_lingshu(_world_point(world3, world_trans, local_center))
    anc = to_lingshu(anchor_world)
    rel = tuple(wc[i] - anc[i] for i in range(3))
    rl = axis_rotate(world3)
    e = mat3_to_euler_xyz(rl)
    return {
        "kind": kind,
        "pos": [round(v, 6) for v in rel],
        "size": [round(dims[0], 6), round(dims[2], 6), round(dims[1], 6)],
        "color": list(color),
        "rot": [round(e[0], 6), round(e[1], 6), round(e[2], 6)],
        "_ratio": round(float(ratio), 6),
    }


def _world_point(world3, world_trans, local_pt):
    """world_matrix @ local_point = R·p + t。"""
    r = apply3(tuple(tuple(x) for x in world3), local_pt)
    return tuple(r[i] + float(world_trans[i]) for i in range(3))


# ------------------------------------------------------------------ shapes.json
def build_shapes_doc(links, source="blender/wmtest", export_frame=None):
    """`links` → `lingshu.shapes/1` 文档。

    links: {link_name: {"anchor": [x,y,z]（llingshu 轴，仅作留痕）, "parts": [part…]}}
    export_frame: **必填**（int ≥ 0）。为什么必填见 `check_pose_consistency` 的说明：
    位姿随帧变，不记帧的位姿描述**不可复核**（实测踩过：机械臂的部件位姿一度是
    动画最后一帧的位姿，而 oracle 的 tick 0 是另一个位姿）。
    """
    if export_frame is None:
        raise GeomError("build_shapes_doc 必须显式给 export_frame（位姿随帧变，不记帧不可复核）")
    ef = int(export_frame)
    if ef < 0:
        raise GeomError("export_frame 不得为负：%r" % (export_frame,))
    assert_axis_map()
    shapes = {}
    for lname in sorted(links):
        rec = links[lname] or {}
        parts = []
        for p in (rec.get("parts") or []):
            q = {k: p[k] for k in ("kind", "pos", "size", "color", "rot")}
            if q["kind"] not in FIT_KINDS:
                raise GeomError("link %s 的部件图元非法：%r" % (lname, q["kind"]))
            if len(q["size"]) != 3 or min(q["size"]) <= 0:
                raise GeomError("link %s 的部件尺寸非法：%r" % (lname, q["size"]))
            parts.append(q)
        if parts:
            shapes["wmtest/" + lname] = parts
    if not shapes:
        raise GeomError("shapes 文档为空（没有可导出的部件）")
    return {"schema": SHAPES_SCHEMA, "source": source, "pos_origin": "anchor",
            "axis_note": AXIS_NOTE, "export_frame": ef, "shapes": shapes}


def validate_shapes_doc(doc):
    """shapes 文档的结构自校。返回 (ok, why)。**fail-closed 判据的单一落点。**"""
    if not isinstance(doc, dict):
        return False, "顶层不是对象"
    if doc.get("schema") != SHAPES_SCHEMA:
        return False, "schema 不符：%r" % (doc.get("schema"),)
    if doc.get("pos_origin") not in ("anchor", "bbox_center"):
        return False, "pos_origin 非法：%r" % (doc.get("pos_origin"),)
    if not isinstance(doc.get("export_frame"), int) or isinstance(doc.get("export_frame"), bool):
        return False, "缺 export_frame（int）—— 位姿描述必须记帧"
    if "# normalize" in json.dumps(doc.get("axis_note") or ""):
        return False, "axis_note 混进了 OBJ 头部契约行"
    if doc.get("axis_note") != AXIS_NOTE:
        return False, "axis_note 与桥的轴换算不一致：%r" % (doc.get("axis_note"),)
    shapes = doc.get("shapes")
    if not isinstance(shapes, dict) or not shapes:
        return False, "shapes 须为非空对象"
    for cat, parts in shapes.items():
        if not isinstance(parts, list) or not parts:
            return False, "%s 的部件列表为空" % cat
        for i, p in enumerate(parts):
            for k in ("kind", "pos", "size", "color", "rot"):
                if k not in p:
                    return False, "%s[%d] 缺键 %s" % (cat, i, k)
            if p["kind"] not in FIT_KINDS:
                return False, "%s[%d] kind 非法：%r" % (cat, i, p["kind"])
            for k, n in (("pos", 3), ("size", 3), ("rot", 3), ("color", 3)):
                v = p[k]
                if not isinstance(v, list) or len(v) != n:
                    return False, "%s[%d].%s 须为 %d 元：%r" % (cat, i, k, n, v)
                for x in v:
                    if isinstance(x, bool) or not isinstance(x, (int, float)) \
                            or not math.isfinite(x):
                        return False, "%s[%d].%s 含非有限数值：%r" % (cat, i, k, x)
            if min(p["size"]) <= 0:
                return False, "%s[%d].size 含非正值：%r" % (cat, i, p["size"])
    return True, ""


def write_shapes(path, links, source="blender/wmtest", export_frame=None):
    """写 shapes.json（`sort_keys=True`、无缩进、无尾换行 —— 与既有产物逐字节同形）。"""
    doc = build_shapes_doc(links, source=source, export_frame=export_frame)
    ok, why = validate_shapes_doc(doc)
    if not ok:
        raise GeomError("shapes 自校不通过：%s" % why)
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, ensure_ascii=False, sort_keys=True)
    return {"links": len(doc["shapes"]),
            "parts": sum(len(v) for v in doc["shapes"].values()),
            "kinds": {k: sum(1 for v in doc["shapes"].values()
                             for p in v if p["kind"] == k) for k in FIT_KINDS},
            "export_frame": doc["export_frame"],
            "bytes": os.path.getsize(path)}


def read_shapes(path, legacy_frame=None):
    """读回 shapes.json 并自校（坏件当场 raise，不进下游）。

    旧件（本桥登记 `export_frame` 之前导出的）缺该键：**不许猜** —— 必须由调用方
    显式 `legacy_frame=N` 声明"这批部件位姿取自哪一 tick"，否则 raise。
    """
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if "export_frame" not in doc:
        if legacy_frame is None:
            raise GeomError(
                "%s：缺 export_frame（旧口径件）。要读它必须显式声明 legacy_frame=N"
                "（该件的部件位姿取自哪一 tick）—— 本桥不替调用方猜帧" % path)
        doc["export_frame"] = int(legacy_frame)
        doc["_legacy_frame_declared"] = True
    ok, why = validate_shapes_doc(doc)
    if not ok:
        raise GeomError("shapes 坏件：%s" % why)
    return doc


def emit_shapes_doc(doc, path):
    """把一份**已校验**的 shapes 文档按既有字节形态写出（sort_keys、无缩进、无尾换行）。"""
    ok, why = validate_shapes_doc(doc)
    if not ok:
        raise GeomError("shapes 自校不通过：%s" % why)
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
    out = {k: doc[k] for k in ("schema", "source", "pos_origin", "axis_note",
                               "export_frame", "shapes") if k in doc}
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, sort_keys=True)
    return {"links": len(out["shapes"]),
            "parts": sum(len(v) for v in out["shapes"].values()),
            "export_frame": out["export_frame"],
            "bytes": os.path.getsize(path)}


# ------------------------------------------------------------------ 跨件一致性守卫
def meshes_from_dump(dump):
    """dump（Blender 侧抽取的原始量）→ `write_obj` 的 meshes 批次。

    ⚠ OBJ **不做轴换算**：它就是「Blender Z-up 原样、1 单位 = 1 m」，换算发生在
    lingshu 载入端（`load_obj_mesh` 按头部 normalize 归一化）。所以这里只做形状校验。
    """
    out = []
    for m in (dump.get("meshes") or []):
        verts = m.get("verts") or []
        if not verts:
            raise GeomError("dump 里的网格 %s 没有顶点" % m.get("name"))
        mats = []
        for pair in (m.get("materials") or []):
            nm, col = pair[0], pair[1]
            mats.append((nm, tuple(float(c) for c in col)))
        out.append({"name": m["name"],
                    "verts": [tuple(float(c) for c in v) for v in verts],
                    "tris": [tuple(int(i) for i in t) for t in (m.get("tris") or [])],
                    "material": m.get("material"),
                    "materials": mats})
    if not out:
        raise GeomError("dump 里没有任何网格")
    return out


def links_from_dump(dump):
    """dump → `write_shapes` 的 links（**轴换算唯一落点**：pos 相对枢轴、size 换轴、rot 换轴）。"""
    links = {}
    for lk in (dump.get("links") or []):
        anchor = lk.get("anchor")
        if not anchor or len(anchor) != 3:
            raise GeomError("link %s 缺 anchor（枢轴世界坐标）" % lk.get("name"))
        parts = []
        for p in (lk.get("parts") or []):
            parts.append(part_from_blender(
                p["dims"], p["vol"], p["world3"], p["world_trans"],
                p["local_center"], anchor, p.get("color") or [170, 170, 175],
                kind=p.get("kind")))
        links[lk["name"]] = {"anchor": list(to_lingshu(anchor)), "parts": parts}
    if not links:
        raise GeomError("dump 里没有任何 link")
    return links


def anchors_from_dump(dump):
    """{link: lingshu 轴枢轴世界坐标} —— 跨件核对用的对照面。"""
    return {lk["name"]: list(to_lingshu(lk["anchor"]))
            for lk in (dump.get("links") or []) if lk.get("anchor")}


def check_pose_consistency(shapes_doc, oracle_doc, dump, tol=None):
    """**跨件守卫**：位姿描述 `shapes.json` 与真值 `oracle` 必须描述**同一帧的同一个姿态**。

    两条判据（都要过）：
      A. **枢轴**：`shapes_doc["export_frame"]` 那一 tick，oracle 里每个 link 实体的
         `pos` 必须等于 dump 记的该 link 枢轴世界坐标（lingshu 轴）。
      B. **件级姿态链**：每个件的世界旋转必须等于「该 link 在该帧的世界旋转 ×
         该件的 link 局部旋转」—— `R_link(frame) · local3`。
         这条才是**能抓住"位姿取了别的帧"**的判据：A 只看平移，静态根连平移都不动，
         位姿换了帧也照样通过；B 比较的是旋转，换帧必炸。

    为什么需要这两条：实测踩过一次 —— `shapes` 的位姿是**动画循环最后一次赋值**
    留下的位姿（机械臂 = u=1：yaw **+35°**），而 `oracle` 的 `tick 0` 是 u=0
    （yaw **−35°**）；一次导出里两件产物描述了**两个姿态**，而单看任何一件都发现不了。
    还有一层：旧件的 link 名（`wmtest/<模型>`）与 oracle 实体名（`<模型>_root`）
    **根本对不上**，没有任何 key 能把两件 join 起来 —— 本函数会如实报 `missing_tick`。

    返回 (ok, problems[])；problems 每条 {"link","kind","detail"}。
    """
    tol = POSE_TOL if tol is None else float(tol)   # 缺省从模块常量取（单点可断言）
    problems = []
    frame = shapes_doc.get("export_frame")
    if not isinstance(frame, int) or isinstance(frame, bool):
        return False, [{"link": "-", "kind": "frame", "detail": "shapes 缺 export_frame"}]
    if not isinstance(dump, dict):
        return False, [{"link": "-", "kind": "dump", "detail": "缺 dump（无从核对）"}]
    tick = {}
    for r in (oracle_doc.get("tracks") or []):
        if r.get("tick") == frame:
            tick[r.get("entity")] = r
    cats = set((shapes_doc.get("shapes") or {}).keys())
    seen_cats = set()
    for lk in (dump.get("links") or []):
        name = lk.get("name")
        seen_cats.add("wmtest/" + str(name))
        rec = tick.get(name)
        if rec is None:
            problems.append({"link": name, "kind": "missing_tick",
                             "detail": "oracle 在 tick=%d 没有 %s 的记录 —— "
                                       "形状件与真值件**没有共同的 join key**，"
                                       "跨件核对失效（旧件正是如此：link 名 "
                                       "wmtest/<模型> vs 实体 <模型>_root）" % (frame, name)})
            continue
        want = to_lingshu(lk.get("anchor") or (0.0, 0.0, 0.0))
        g = rec.get("pos") or {}
        for i, ax in enumerate(("x", "y", "z")):
            if abs(float(g.get(ax, 0.0)) - float(want[i])) > tol:
                problems.append({"link": name, "kind": "anchor_mismatch",
                                 "detail": "tick=%d 的 %s：oracle=%r vs dump=%.6f"
                                           % (frame, ax, g.get(ax), float(want[i]))})
        r_link = quat_to_mat3(rec.get("rot_quat")) if rec.get("rot_quat") else None
        if r_link is None:
            problems.append({"link": name, "kind": "no_quat",
                             "detail": "oracle tick=%d 的 %s 没有 rot_quat，"
                                       "件级姿态链无法核对" % (frame, name)})
            continue
        for p in (lk.get("parts") or []):
            if "world3" not in p or "local3" not in p:
                continue
            got = axis_rotate(p["world3"])
            exp = mat3_mul(r_link, axis_rotate(p["local3"]))
            dev = max(abs(got[i][k] - exp[i][k]) for i in range(3) for k in range(3))
            if dev > tol:
                problems.append({
                    "link": name, "kind": "part_pose_mismatch",
                    "detail": "%s：tick=%d 的 R_link·local3 与 dump 记的 world3 差 %.6f"
                              "（>%.0e）⇒ 几何/部件位姿与真值**不在同一帧**"
                              % (p.get("src") or "?", frame, dev, tol)})
    # shapes 里出现的分类若在 dump 里没有对应 link，也要点名（否则"核对了"是假的）
    for cat in sorted(cats - seen_cats):
        problems.append({"link": cat, "kind": "shapes_link_not_in_dump",
                         "detail": "%s 在 dump 的 links 里没有同名 link（无法核对）" % cat})
    return (not problems), problems


def write_from_dump(dump, out_dir=None, obj=None, mtl=None, shapes=None,
                    oracle_path=None):
    """dump → 三件产物（CLI `export-*` 共用的单一实现；便于自测整体调用）。

    只写显式给出的路径；返回 {"obj":…, "shapes":…, "oracle":…}（未要的为 None）。
    真值那件走 `oracle.build_doc/write_oracle` 自校（此处不重复实现）。
    """
    frame = dump.get("frame")
    if not isinstance(frame, int) or isinstance(frame, bool):
        raise GeomError("dump 缺 frame（导出必须声明取哪一帧的位姿）")
    res = {"obj": None, "shapes": None, "oracle": None}
    if obj:
        res["obj"] = write_obj(mtl or (obj + ".mtl"), obj,
                               meshes_from_dump(dump),
                               translate=tuple(dump.get("translate") or (0.0, 0.0, 0.0)))
    if shapes:
        res["shapes"] = write_shapes(shapes, links_from_dump(dump),
                                     source=dump.get("source") or "blender/wmtest",
                                     export_frame=frame)
    if oracle_path:
        from . import oracle as O
        doc = O.doc_from_dump(dump)
        res["oracle"] = O.write_oracle(oracle_path, doc)
    return res
