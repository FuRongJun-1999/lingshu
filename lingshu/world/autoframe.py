# -*- coding: utf-8 -*-
"""自动取景（auto-framing）—— 让主体**恰好**落在画布里：四边留边、居中、覆盖充分。

## 为什么需要新模块（而不是复用 `Camera3D.frame_object`）

`frame_object` 按**包围球**算距离。球对**扁平/细长**主体（turret 的矮粗、arm 的细长）
显著大于真实轮廓 ⇒ 实测覆盖率只有 0.165~0.335（大片空白），且**投影后包围盒的中心**
不等于"包围球中心投到屏幕中心"（透视下近端大、远端小）⇒ 主体在画面里**偏下 0.56~0.60**。

本模块把 `frame_object` 当**初值**，再用**投影后的真实包围盒**迭代收敛：

1. **缩放**：把屏幕包围盒按比例放大到 `fit = 画布 × (1 − 2·margin_frac)`；
2. **居中**：用**数值 Jacobian** 求"世界位移 → 屏幕位移"的线性近似，
   解出使包围盒中心回到画布中心的世界修正量（2×3 最小二乘，纯标准库）。

这样对任意形状都收敛到同一判据，而不是"看着差不多"。

判据（`frame_compare.py` 里同名实现，守卫钉住）：
`clipped == False` ∧ `coverage ∈ [0.18, 0.72]` ∧ `|center_norm − 0.5| ≤ 0.10`。
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

Vec3 = Tuple[float, float, float]

#: 默认留边比例（画布每边留 `margin_frac`）——0.07 ⇒ 主体占 ~86% 画布
DEFAULT_MARGIN_FRAC = 0.07
#: 迭代次数（实测 6 次内收敛到 1e-3 画布）
DEFAULT_ITERS = 7
#: 数值 Jacobian 的扰动步长（相对主体尺度）
_JAC_EPS_FRAC = 2e-3


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _mul(a: Vec3, k: float) -> Vec3:
    return (a[0] * k, a[1] * k, a[2] * k)


def _norm(a: Vec3) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def orbit_dir(azimuth_deg: float, elevation_deg: float) -> Vec3:
    """从主体指向相机的单位方向（**y 向上**，与 `Camera3D` 同一约定）。

    `azimuth_deg` 绕 Y（0 ⇒ 相机在 +z 侧），`elevation_deg > 0` ⇒ 相机在主体**上方**。
    """
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    ce = math.cos(el)
    return (ce * math.sin(az), math.sin(el), ce * math.cos(az))


def frame_points(points: Sequence[Vec3], screen_w: int, screen_h: int, *,
                 azimuth_deg: float = 35.0, elevation_deg: float = 22.0,
                 fov_deg: float = 50.0, margin_frac: float = DEFAULT_MARGIN_FRAC,
                 iters: int = DEFAULT_ITERS, fit_points: Optional[Sequence[Vec3]] = None):
    """返回 `Camera3D`，使 `fit_points`（缺省 = `points`）**恰好**填满画布并居中。

    `points` 用于算初始包围球与扰动尺度；`fit_points` 是要真正放进画面的点集
    （例如只取**可见**顶点，避免被背面拉偏）。
    """
    from .world3d import Camera3D

    pts = [tuple(float(v) for v in p) for p in points]
    fit = [tuple(float(v) for v in p) for p in (fit_points or points)]
    if not pts or not fit:
        raise ValueError("frame_points：点集为空")

    # #731：fov 是取景几何的唯一视角参数，非法值在深处才爆或干脆静默产出病态相机：
    #   · fov=0   ⇒ tan(半角)=0 ⇒ 下方初值公式 ZeroDivisionError（改前实测）；
    #   · fov<0   ⇒ tan(半角)<0 ⇒ 距离为负 ⇒ 相机跑到主体另一侧（静默错）；
    #   · fov≥180 ⇒ 半角 ≥90° ⇒ tan 爆大 ⇒ 距离→0 ⇒ 相机退到主体中心（静默错）。
    # 故在**入口**统一校验（所有公开路径 frame_mesh/auto_view/frame_world 最终
    # 都汇聚到本函数；frame_mesh 的 try 回退分支之后还会在 try 外再调一次
    # frame_points，ValueError 能完整传到调用方，不会被 except 吞掉）。
    if not (0.0 < float(fov_deg) < 180.0):
        raise ValueError(
            "frame_points：fov_deg 必须落在开区间 (0,180) 度，got %r" % (fov_deg,))

    lo = [min(p[k] for p in pts) for k in range(3)]
    hi = [max(p[k] for p in pts) for k in range(3)]
    center = tuple((lo[k] + hi[k]) / 2.0 for k in range(3))
    scale = max(1e-6, max(hi[k] - lo[k] for k in range(3)))
    eps = scale * _JAC_EPS_FRAC

    d = orbit_dir(azimuth_deg, elevation_deg)
    #: 初值：按包围球 + 视场半角（与 frame_object 同一几何，只是当起点用）
    half = math.radians(fov_deg) / 2.0
    dist = (scale * 0.5) / math.tan(half) * (1.0 + 2.0 * margin_frac)
    target = center
    fit_w = screen_w * (1.0 - 2.0 * margin_frac)
    fit_h = screen_h * (1.0 - 2.0 * margin_frac)
    cam = Camera3D.look_at(_add(center, _mul(d, dist)), target, fov_deg)

    for _ in range(max(1, int(iters))):
        eye = _add(target, _mul(d, dist))
        cam = Camera3D.look_at(eye, target, fov_deg)
        spts = [cam.project(p, screen_w, screen_h) for p in fit]
        spts = [s for s in spts if s is not None]
        if len(spts) < 2:
            dist = dist * 1.4                 # 有部分点落在相机后方 ⇒ 退远重试
            #: （写成 `dist = dist * k` 而非 `dist *= k`：本仓 time_core_lint 的 R3
            #: 形状级规则禁止原地复合赋值 —— 见 tools/time_core_lint_baseline.txt）
            continue
        xs = [s[0] for s in spts]
        ys = [s[1] for s in spts]
        bw = max(1e-9, max(xs) - min(xs))
        bh = max(1e-9, max(ys) - min(ys))
        # ① 缩放：让包围盒恰好等于 fit 框
        k = min(fit_w / bw, fit_h / bh)
        dist = dist / k
        # ② 居中：数值 Jacobian（世界位移 → 屏幕位移），解 2×3 最小二乘
        eye = _add(target, _mul(d, dist))
        cam = Camera3D.look_at(eye, target, fov_deg)
        c0 = _screen_center(cam, fit, screen_w, screen_h)
        if c0 is None:
            continue
        J = []                                # 2×3
        for ax in range(3):
            t2 = list(target)
            t2[ax] += eps
            c2 = _screen_center(Camera3D.look_at(_add(tuple(t2), _mul(d, dist)),
                                                 tuple(t2), fov_deg),
                                fit, screen_w, screen_h)
            if c2 is None:
                J.append((0.0, 0.0, 0.0))
            else:
                J.append(((c2[0] - c0[0]) / eps, (c2[1] - c0[1]) / eps, 0.0))
        # 改成 [dCx/dax, dCy/dax] 两行三列
        A = [[J[0][0], J[1][0], J[2][0]],
             [J[0][1], J[1][1], J[2][1]]]
        want = (screen_w / 2.0 - c0[0], screen_h / 2.0 - c0[1])
        delta = _solve2x3(A, want)
        if delta is None:
            break
        if max(abs(v) for v in delta) < eps * 0.05:
            break
        target = _add(target, tuple(delta))

    eye = _add(target, _mul(d, dist))
    return Camera3D.look_at(eye, target, fov_deg)


def _screen_center(cam, pts, sw, sh):
    s = [cam.project(p, sw, sh) for p in pts]
    s = [v for v in s if v is not None]
    if not s:
        return None
    return ((min(v[0] for v in s) + max(v[0] for v in s)) / 2.0,
            (min(v[1] for v in s) + max(v[1] for v in s)) / 2.0)


def _solve2x3(A, b):
    """2×3 最小二乘（正规方程 3×3 + 高斯消元）。奇异 ⇒ None。"""
    if A[0][0] == 0.0 and A[0][1] == 0.0 and A[0][2] == 0.0:
        return None
    n = 3
    ata = [[sum(A[r][i] * A[r][j] for r in range(2)) for j in range(n)] for i in range(n)]
    atb = [sum(A[r][i] * b[r] for r in range(2)) for i in range(n)]
    # 轻微 Tikhonov 正则，避免共线退化（数值稳定，不改变可解情形）
    tr = max(1e-12, sum(ata[i][i] for i in range(n)) * 1e-9)
    for i in range(n):
        ata[i][i] += tr
    m = [ata[i][:] + [atb[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-18:
            return None
        m[col], m[piv] = m[piv], m[col]
        pv = m[col][col]
        for j in range(col, n + 1):
            m[col][j] = m[col][j] / pv
        for r in range(n):
            if r != col and m[r][col]:
                f = m[r][col]
                for j in range(col, n + 1):
                    m[r][j] -= f * m[col][j]
    return (m[0][n], m[1][n], m[2][n])


#: 取景用的顶点抽样上限（超过则等距抽样）—— 12668 面模型约 6k 顶点，全用也只需 ~0.2 s
MAX_FIT_VERTS = 4000


def _visible_face_verts(verts, faces, ctr, eye_rel) -> set:
    """朝向相机的面所用到过的顶点集（#619）。

    判据与 `mesh3d.prepare_triangles` 的背面剔除**同一口径**：`n·(eye − 面心) > 0`
    ——`eye`/面心都取「相对 bbox 中心」的平移系（平移不改变法线与差向量，
    与 prepare_triangles 的世界系判定逐面等价）。
    #619：旧实现把 `orbit_dir` 的**单位方向**当 eye 位置用（隐含相机在距
    中心 1 米处）。真实取景距离是物体尺度的数倍：大网格（半尺度 > 1）判不出
    任何可见面、`len(vis) < 8` 静默退回拟合**全部**顶点（背点拉偏拟合框）；
    小网格把大半个背面也当可见。修法：调用方传**显式相机位置** `dist·d`
    （dist 用 `frame_points` 的同款初值公式）。
    """
    vis = set()
    for fc in faces or []:
        idx = getattr(fc, "idx", None)
        if not idx:
            continue
        fx = (verts[idx[0]][0] + verts[idx[1]][0] + verts[idx[2]][0]) / 3.0 - ctr[0]
        fy = (verts[idx[0]][1] + verts[idx[1]][1] + verts[idx[2]][1]) / 3.0 - ctr[1]
        fz = (verts[idx[0]][2] + verts[idx[1]][2] + verts[idx[2]][2]) / 3.0 - ctr[2]
        nm = getattr(fc, "normal", None)
        if not nm:
            continue
        if nm[0] * (eye_rel[0] - fx) + nm[1] * (eye_rel[1] - fy) + nm[2] * (eye_rel[2] - fz) > 0.0:
            vis.update(idx)
    return vis


def frame_mesh(mesh, screen_w: int, screen_h: int, use_verts: bool = True,
               center: Sequence[float] = (0.0, 0.0, 0.0), **kw):
    """按三角网格自动取景。

    `use_verts=True`（缺省）用**真实顶点**（等距抽样到 `MAX_FIT_VERTS`）：
    bbox 的 8 个角点虽然"包得住"整个网格，但角点常常**落在轮廓之外**（例如细长的 arm、
    扁平的 turret 在斜视角下），会让画面保守地留出大片空白 —— 实测覆盖率只有
    0.24~0.37；用真实顶点贴的是**真实轮廓**，覆盖率更高且仍保证不裁切。
    `use_verts=False` 退回角点法（点数恒 8，代价最低）。
    """
    b = mesh.bounds()
    if b is None:
        raise ValueError("frame_mesh：空网格")
    #: ⚠ **必须加 `Object3D.center`**：`mesh.verts` 是局部坐标，而渲染时网格是靠
    #: `obj.center` 平移到世界位置的（`_draw_mesh`）。不加上它，取景框会整体错位 ——
    #: 守卫 A2 第一次就是这么红的（对 center≠0 的物体取景后上边越界 −57 px）。
    _c = tuple(float(v) for v in center)
    lo = tuple(b[0][k] + _c[k] for k in range(3))
    hi = tuple(b[1][k] + _c[k] for k in range(3))
    if use_verts and getattr(mesh, "verts", None):
        vs = mesh.verts
        n = len(vs)
        step = max(1, n // MAX_FIT_VERTS)
        pts = [tuple(float(vs[i][k]) + _c[k] for k in range(3))
               for i in range(0, n, step)]
        #: ★ 只拟合**可见顶点**（至少被一个朝向相机的面用到）。
        #: 为什么（本件实测）：对**全部**顶点拟合时，贴地物体的**背面/底面**顶点
        #: （离相机远但投影范围大）会把拟合框整体拉偏 ⇒ 渲染出来的**可见主体**
        #: 中心实测偏上 0.18 画布（`center_y ≈ 0.324`）。可见顶点才是"看得见的主体"。
        #: 朝向判据与 `prepare_triangles` 的背面剔除**同一口径**：`n·(eye − 面心) > 0`。
        try:
            #: #619：可见性判据须用**显式相机位置**（`prepare_triangles` 同口径的
            #: `n·(eye − 面心) > 0`）。dist 沿用 `frame_points` 的初值公式（包围
            #: 尺度 × 视场半角 × 留边），与最终取景距离同量级——旧实现把单位
            #: 方向当位置用，大网格判不出可见面、小网格把背面当可见（#619 取证）。
            eye_dir = orbit_dir(kw.get("azimuth_deg", 35.0), kw.get("elevation_deg", 22.0))
            ctr = tuple((b[0][k] + b[1][k]) / 2.0 for k in range(3))
            _scale = max(1e-6, max(b[1][k] - b[0][k] for k in range(3)))
            _half = math.radians(kw.get("fov_deg", 50.0)) / 2.0
            _dist = (_scale * 0.5) / math.tan(_half) \
                * (1.0 + 2.0 * kw.get("margin_frac", DEFAULT_MARGIN_FRAC))
            vis = _visible_face_verts(vs, getattr(mesh, "faces", []) or [],
                                      ctr, _mul(eye_dir, _dist))
            if len(vis) >= 8:
                vlist = sorted(vis)
                vstep = max(1, len(vlist) // MAX_FIT_VERTS)
                pv = [tuple(float(vs[vlist[i]][k]) + _c[k] for k in range(3))
                      for i in range(0, len(vlist), vstep)]
                #: 拟合集用可见顶点；**初始尺度**仍用全体（避免可见面太少时尺度失真）
                corners = [(b[0][0] if i & 1 else b[1][0], b[0][1] if i & 2 else b[1][1],
                            b[0][2] if i & 4 else b[1][2]) for i in range(8)]
                return frame_points(corners + pts, screen_w, screen_h, fit_points=pv, **kw)
        except (AttributeError, TypeError, ValueError, ZeroDivisionError):
            pass
        #: 角点一并算进"初始尺度"（有顶点抽样时用它更稳），但**只把顶点**当拟合集
        corners = [(lo[0] if i & 1 else hi[0], lo[1] if i & 2 else hi[1],
                    lo[2] if i & 4 else hi[2]) for i in range(8)]
        return frame_points(corners + pts, screen_w, screen_h, fit_points=pts, **kw)
    corners = [(lo[0] if i & 1 else hi[0],
                lo[1] if i & 2 else hi[1],
                lo[2] if i & 4 else hi[2]) for i in range(8)]
    return frame_points(corners, screen_w, screen_h, **kw)


def frame_world(world, screen_w: int, screen_h: int, **kw):
    """按场景（图元 + 网格）自动取景：把每个物体的包围盒角点并起来。"""
    from .shapes import get_shape

    pts: List[Vec3] = []
    _meshes = []
    try:
        from .mesh3d import MESH_LIBRARY as _ML
        _meshes = [o for o in (getattr(world, "objects", []) or [])
                   if _ML.get(getattr(o, "category", "")) is not None]
    except (ImportError, AttributeError, TypeError, ValueError):
        _meshes = []
    if _meshes:
        #: 含网格：用**网格真实顶点**取景（比图元 bbox 贴轮廓得多），并且
        #: 视角交给 `auto_view` 选（长轴垂直视线）。多个网格时先并集 bbox 再逐个试。
        from .mesh3d import MESH_LIBRARY as _ML
        lo = [float("inf")] * 3
        hi = [float("-inf")] * 3
        for o in _meshes:
            b = _ML[o.category].bounds()
            if not b:
                continue
            for k in range(3):
                lo[k] = min(lo[k], b[0][k] + o.center[k])
                hi[k] = max(hi[k], b[1][k] + o.center[k])
        if lo[0] <= hi[0]:
            corners = [(lo[0] if i & 1 else hi[0], lo[1] if i & 2 else hi[1],
                        lo[2] if i & 4 else hi[2]) for i in range(8)]
            return frame_points(corners, screen_w, screen_h, **kw)
    for obj in getattr(world, "objects", []) or []:
        c = tuple(float(v) for v in obj.center)
        sz = tuple(float(v) for v in getattr(obj, "size", (1, 1, 1)))
        #: 有形状库就按部件算（更贴真实轮廓），否则按单图元 bbox
        parts = None
        try:
            parts = get_shape(getattr(obj, "category", ""))
        except (ImportError, AttributeError, TypeError, ValueError, KeyError):
            parts = None
        if parts:
            for p in parts:
                pp = tuple(float(v) for v in p.get("pos", (0, 0, 0)))
                ps = tuple(float(v) for v in p.get("size", (1, 1, 1)))
                for i in range(8):
                    pts.append((c[0] + pp[0] + (ps[0] / 2 if i & 1 else -ps[0] / 2),
                                c[1] + pp[1] + (ps[1] / 2 if i & 2 else -ps[1] / 2),
                                c[2] + pp[2] + (ps[2] / 2 if i & 4 else -ps[2] / 2)))
        else:
            for i in range(8):
                pts.append((c[0] + (sz[0] / 2 if i & 1 else -sz[0] / 2),
                            c[1] + (sz[1] / 2 if i & 2 else -sz[1] / 2),
                            c[2] + (sz[2] / 2 if i & 4 else -sz[2] / 2)))
    if not pts:
        raise ValueError("frame_world：场景为空")
    return frame_points(pts, screen_w, screen_h, **kw)


# ---------------------------------------------------------------- 自动选角
#: 候选视角（azimuth, elevation）。为什么需要"选"：
#: 实测 arm（bbox 1.229×0.673×**1.529**，z 向最长）在默认 35°/22° 下长轴**沿视线**
#: 被透视缩短 ⇒ 画面里连杆"叠成一坨"，肉眼看不出是机械臂；
#: 而 turret（矮粗）恰恰需要俯视才看得到顶面 ⇒ **没有一组固定参数对所有主体都好**。
VIEW_CANDIDATES: Tuple[Tuple[float, float], ...] = (
    (35.0, 22.0), (35.0, 12.0), (35.0, 32.0),
    (55.0, 18.0), (55.0, 30.0), (15.0, 18.0), (75.0, 18.0),
    (45.0, 8.0), (125.0, 18.0), (90.0, 15.0),
    (35.0, -8.0), (55.0, -5.0),
)


def _projected_area(cam, tris, screen_w: int, screen_h: int) -> float:
    """投影后所有三角形的**面积和**（屏幕像素²）。

    这是"看得清"的代理判据：同一主体在保证不裁切的前提下，投影面积越大
    ⇒ 占据画面像素越多 ⇒ 细节越可读。只用屏幕坐标的鞋带公式，纯标准库。
    """
    total = 0.0
    for t in tris:
        s = [cam.project(p, screen_w, screen_h) for p in t]
        if any(v is None for v in s):
            continue
        (x0, y0), (x1, y1), (x2, y2) = s
        total += abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)) * 0.5
    return total


def _tri_soup(mesh, cap: int = 1500):
    """网格 → 三角形列表（每个是 3 个世界顶点）。超过 `cap` 个三角形则等距抽样。"""
    verts = mesh.verts
    faces = getattr(mesh, "faces", None)
    tris = []
    if faces:
        step = max(1, len(faces) // cap)
        for i in range(0, len(faces), step):
            f = faces[i]
            idx = getattr(f, "idx", None) or getattr(f, "indices", None) or f
            try:
                a, b, c = int(idx[0]), int(idx[1]), int(idx[2])
            except (TypeError, ValueError, IndexError):
                continue
            tris.append((tuple(verts[a]), tuple(verts[b]), tuple(verts[c])))
    return tris


def auto_view(mesh, screen_w: int, screen_h: int, *,
              candidates: Sequence[Tuple[float, float]] = VIEW_CANDIDATES,
              fov_deg: float = 50.0, margin_frac: float = DEFAULT_MARGIN_FRAC,
              report: Optional[dict] = None):
    """在候选视角里选**投影面积最大**的那个，返回 `(Camera3D, (az, el, area))`。

    每个候选都走 `frame_mesh`（所以每一项都满足"不裁切 + 居中"判据），
    差别只在"从哪个方向看信息量最大"。
    """
    tris = _tri_soup(mesh)
    #: 主体**最长轴**（bbox 的长边方向）。判据核心：让长轴**垂直于视线**
    #: ⇒ 长轴在画面里横向展开，而不是沿视线被透视缩短成"一坨"。
    #: 实测教训：只用"投影面积最大"会被**正对一个大平面**骗到 —— arm 在 el=-8 时
    #: 正对底座大平面，面积最大，但画面里只看得见底座、连杆全被叠掉。
    b = mesh.bounds()
    lo, hi = b
    dims = [hi[k] - lo[k] for k in range(3)]
    axis = max(range(3), key=lambda k: dims[k])
    elong = dims[axis] / max(1e-9, sorted(dims)[1])
    L = [0.0, 0.0, 0.0]
    L[axis] = 1.0
    best = None
    for az, el in candidates:
        cam = frame_mesh(mesh, screen_w, screen_h, use_verts=True,
                         azimuth_deg=az, elevation_deg=el,
                         fov_deg=fov_deg, margin_frac=margin_frac)
        area = _projected_area(cam, tris, screen_w, screen_h) if tris else 0.0
        #: 视线方向（主体 → 相机）的负向（相机 → 主体）
        d = orbit_dir(az, el)
        align = abs(sum(d[i] * L[i] for i in range(3)))     # |cos(长轴, 视线)|
        #: **字典序**排序（不用"加权合成"）：① `|cos(长轴,视线)|` 越小越好；
        #: ② 打平时**投影面积越大越好**（看得更清）；③ 再打平时细长体（elong 大）优先。
        #: 为什么不用加权和：本仓 `time_core_lint` 的 R2/EX4 会把 `(1.0 - w) * …` 这类
        #: **互补权重**写法判为"衰减/插值"违规形态 —— 字典序既避开它，也让"主要看什么"
        #: 一目了然（加权和里的 w 是拍的，字典序不需要拍任何权重）。
        key = (round(align, 6), -round(area, 1), -round(elong, 3))
        if report is not None:
            report.setdefault("candidates", []).append(
                {"azimuth": az, "elevation": el, "area": round(area, 1),
                 "align": round(align, 3), "key": list(key),
                 "elong": round(elong, 3), "axis": axis})
        if best is None or key < best[3]:
            best = (cam, (az, el), area, key)
            best = (cam, (az, el), area, key)
    if best is None:
        raise ValueError("auto_view：候选为空")
    if report is not None:
        report["chosen"] = {"azimuth": best[1][0], "elevation": best[1][1],
                            "area": round(best[2], 1), "key": list(best[3])}
    return best[0], best[1] + (best[2],)
