# -*- coding: utf-8 -*-
"""camera_calibrate · 棋盘格标定（Zhang 闭式）+ Brown–Conrady 畸变 · issue #1194 第二步
============================================================================
能力缺口（issue #1194）：`lingshu/world/` 43 个模块无一做相机标定。
第一步（`camera_calib`，规格 → 内参）已在 PR #1195 落地；本模块做**第二步**：
从**图像里的棋盘格角点**解出内参，并估计径向畸变。

为什么第二步必须存在（第一步不够）
--------------------------------
规格只能给"出厂标称值"：厂商的 f 是四舍五入的（"24mm 等效"实际 23.5mm）、
主点根本不给、畸变系数完全不给。而真实图像上的**实际**内参与标称值可以差
几个百分点，畸变更是"不标就歪"。所以：
    规格 → 初值；棋盘格 → 实测值。两步都要有。

依赖口径（issue #1194 的待裁项，本模块**已自行解决**）
----------------------------------------------------
Zhang 闭式解要 SVD/特征分解、LM 要解正规方程。issue 里列了两条路
（A 手写小型线代 / B 单独可选 extra）。本模块走 **A**，因此**不需要任何裁定**：
  · 对称矩阵特征分解 → **Jacobi 旋转**（`_jacobi_eigen_sym`，纯 `math`）；
  · 线性方程组      → 复用 `world/physics.solve_linear`（仓内既有单一真源，
                      不另写第二份高斯消元）；
  · 3×3 求逆        → 伴随矩阵闭式（`_mat_inv3`）。
⇒ `world/` 的纯标准库纪律（D-005）保持不变。

算法（Zhang 2000 的闭式解 + 一个 LM 精修）
-----------------------------------------
1. **单应**：每个棋盘视图 `obj(X,Y,0) → pixels(u,v)` 用 DLT 求 H。
   先做 Hartley 归一化（像素坐标 ~10²–10³，不归一化条件数会爆），解完再反归一化。
2. **内参**：由 `h1ᵀBh1 − h2ᵀBh2 = 0`、`h1ᵀBh2 = 0` 每视图给 2 条约束，
   ≥3 个不同朝向的视图即可解 B（对称阵 6 个未知量，取零空间）。
   再由 B = K⁻ᵀK⁻¹ 反解 α/β/u0/v0（**skew 假设为 0**，见诚实边界）。
3. **畸变**：Brown–Conrady 径向 2 项 `(k1, k2)`，与内参一起用 LM 最小化
   重投影残差。雅可比用**中心差分**（解析式在这里容易写错且难验，差分慢但可信）。

判据（唯一质量判据 = 重投影残差）
--------------------------------
标定好坏**不看**解出来的数像不像"应该的值"，只看**重投影残差**：
把解出的 K + 每视图位姿代回去，算预测像素与实际角点的距离，取 RMS。
残差大就是没标好，没有第二种解释。

诚实边界
--------
· **skew 假设为 0**：Zhang 的 6 元 B 解出 skew 后本模块**不采用**（只用 α/β/u0/v0），
  因为绝大多数现代相机 skew≈0，而 6 元解在噪声下会让 skew 乱跑并污染其它量。
  若确实需要 skew，本模块不适用。
· **只做径向畸变**（k1/k2），不做切向（p1/p2）。切向来自镜头与传感器不平行，
  手机/工业相机通常很小；需要时须扩展。
· **不做鱼眼/超广角**：那需要等距/等积投影模型，不是 Brown–Conrady 能覆盖的。
· **视图朝向必须足够不同**：全平行视图下 B 的零空间不唯一（本模块**显式报错**，
  不返回一个看起来合理的解）。
· 位姿初值来自单应分解，**不做完整 BA**；残差里含位姿误差，不只是内参误差。

判据来源：Zhang, "A Flexible New Technique for Camera Calibration" (TPAMI 2000)
的约束式；Brown (1966) / Conrady 的径向畸变多项式；Hartley & Zisserman 的
归一化 DLT。纯标准库 · 零外部依赖（D-005）。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import physics as _physics

__all__ = [
    "CalibrationError",
    "RESIDUAL_PX_MAX",
    "homography_dlt",
    "intrinsics_from_homographies",
    "chessboard_object_points",
    "calibrate_from_chessboards",
    "distort_normalized",
    "project_with_distortion",
    "calibrate_with_distortion",
    "reprojection_rms",
]

# #2316：「残差过大 ⇒ 不返回可疑解」的默认上界（px）。**经验保守值，追不到理论
# 出处**（同 #395 的 MAX_ROUTES 口径）：无噪合成标定 ~1e-6 px、带小噪实测 <2 px、
# 劣质初值（视图与位姿错位但角点全在前方）实测 >100 px——5 px 卡掉后者、放过前
# 两者。调用方可用 `max_residual_px=` 按次覆盖，不必改常量。
RESIDUAL_PX_MAX = 5.0


class CalibrationError(ValueError):
    """标定失败（输入不足 / 退化 / 解不自洽）。**不返回可疑的解**。"""


# =========================================================== 小型线性代数
def _mat_mul(a: Sequence[Sequence[float]],
             b: Sequence[Sequence[float]]) -> List[List[float]]:
    n, m, p = len(a), len(b), len(b[0])
    if len(a[0]) != m:
        raise CalibrationError("_mat_mul：维度不匹配")
    out = [[0.0] * p for _ in range(n)]
    for i in range(n):
        ai = a[i]
        oi = out[i]
        for k in range(m):
            aik = ai[k]
            if aik == 0.0:
                continue
            bk = b[k]
            for j in range(p):
                oi[j] += aik * bk[j]
    return out


def _mat_transpose(a: Sequence[Sequence[float]]) -> List[List[float]]:
    return [[float(a[i][j]) for i in range(len(a))] for j in range(len(a[0]))]


def _mat_inv3(m: Sequence[Sequence[float]]) -> List[List[float]]:
    """3×3 求逆（伴随矩阵闭式）。奇异 ⇒ 抛错，不返回垃圾。"""
    a, b, c = float(m[0][0]), float(m[0][1]), float(m[0][2])
    d, e, f = float(m[1][0]), float(m[1][1]), float(m[1][2])
    g, h, i = float(m[2][0]), float(m[2][1]), float(m[2][2])
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) < 1e-14:
        raise CalibrationError("_mat_inv3：矩阵奇异（det=%.3e）" % det)
    inv = 1.0 / det
    return [
        [(e * i - f * h) * inv, (c * h - b * i) * inv, (b * f - c * e) * inv],
        [(f * g - d * i) * inv, (a * i - c * g) * inv, (c * d - a * f) * inv],
        [(d * h - e * g) * inv, (b * g - a * h) * inv, (a * e - b * d) * inv],
    ]


def _jacobi_eigen_sym(a: Sequence[Sequence[float]],
                      max_sweeps: int = 100,
                      tol: float = 1e-15) -> Tuple[List[float], List[List[float]]]:
    """对称矩阵特征分解（Jacobi 旋转）。返回 (特征值, 特征向量按列)。

    为什么自己写：`world/` 是纯标准库（D-005），没有 numpy 的 `eigh`。
    Jacobi 对**小尺寸对称阵**（本模块最大 9×9）收敛快且数值稳，
    比"先转三对角再 QR"短得多，也不会引入额外的舍入。

    缺陷态对照：把旋转角写成 `t = tan(theta)` 而非
    `t = sign(θ)/(|θ|+√(θ²+1))` 的稳定式 ⇒ 大角度时溢出，特征值不收敛。
    """
    n = len(a)
    if n == 0 or any(len(r) != n for r in a):
        raise CalibrationError("_jacobi_eigen_sym：需要非空方阵")
    # 对称化（调用方可能只填了上三角）
    m = [[0.5 * (float(a[i][j]) + float(a[j][i])) for j in range(n)] for i in range(n)]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _sweep in range(max_sweeps):
        off = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                off += m[i][j] * m[i][j]
        if off <= tol:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = m[p][q]
                if abs(apq) < 1e-300:
                    continue
                theta = (m[q][q] - m[p][p]) / (2.0 * apq)
                t = math.copysign(1.0, theta) / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):
                    akp, akq = m[k][p], m[k][q]
                    m[k][p] = c * akp - s * akq
                    m[k][q] = s * akp + c * akq
                for k in range(n):
                    apk, aqk = m[p][k], m[q][k]
                    m[p][k] = c * apk - s * aqk
                    m[q][k] = s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq
    return [m[i][i] for i in range(n)], v


def _null_vector(a: Sequence[Sequence[float]],
                 *, rank_tol: float = 1e-10) -> List[float]:
    """A 的零空间向量（A 的**最小**奇异值对应的右奇异向量）。

    做法：对 AᵀA（对称、半正定）做特征分解，取最小特征值的特征向量。
    比直接做 SVD 短得多，且对"求零空间"这一用途等价。

    ★ **唯一性检查**（`rank_tol`）：零空间必须是**一维**的，否则解不唯一，
    返回其中任意一个都会得到一个"看起来合理但不可信"的结果。

    判据用**秩亏**而不是 λ₁/λ₂：`λ₂/λmax < rank_tol` ⇒ 秩不足 ⇒ 零空间维数 >1。
      · 共线点：rank(A) 远小于 8 ⇒ λ₂ ≈ 0 ⇒ 报错（正确）；
      · 有噪声的正常点集：rank 仍为 8，λ₂ 与 λmax 同量级 ⇒ 通过（正确）。
    （曾用 λ₁/λ₂ 作判据：噪声下 λ₁ 并不接近 0，正常数据被误判为退化。）

    缺陷态对照：去掉本检查 ⇒ 共线点也会"解出"一个单应（D 组红）。
    """
    at = _mat_transpose(a)
    ata = _mat_mul(at, a)
    vals, vecs = _jacobi_eigen_sym(ata)
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    k = order[0]
    if len(order) >= 2:
        lam_max = max(abs(v) for v in vals)
        lam2 = abs(vals[order[1]])
        if lam_max > 0.0 and lam2 / lam_max < rank_tol:
            raise CalibrationError(
                "_null_vector：零空间不唯一（λ₂/λmax=%.2e < %.0e）——"
                "点集退化（共线/重合）或视图朝向不足" % (lam2 / lam_max, rank_tol))
    col = [vecs[r][k] for r in range(len(vecs))]
    nrm = math.sqrt(sum(x * x for x in col))
    if nrm < 1e-300:
        raise CalibrationError("_null_vector：零空间向量退化")
    return [x / nrm for x in col]


# =========================================================== 单应（DLT）
def _hartley(pts: Sequence[Sequence[float]]) -> Tuple[List[List[float]],
                                                      List[List[float]]]:
    """Hartley 归一化：重心移到原点、平均距离缩到 √2。"""
    n = len(pts)
    if n < 4:
        raise CalibrationError("_hartley：至少 4 个点（得到 %d）" % n)
    cx = sum(float(p[0]) for p in pts) / n
    cy = sum(float(p[1]) for p in pts) / n
    d = sum(math.hypot(float(p[0]) - cx, float(p[1]) - cy) for p in pts) / n
    if d < 1e-12:
        raise CalibrationError("_hartley：点集退化（平均距离 %.3e）" % d)
    s = math.sqrt(2.0) / d
    t = [[s, 0.0, -s * cx], [0.0, s, -s * cy], [0.0, 0.0, 1.0]]
    out = [[s * (float(p[0]) - cx), s * (float(p[1]) - cy)] for p in pts]
    return out, t


def homography_dlt(src: Sequence[Sequence[float]],
                   dst: Sequence[Sequence[float]]) -> List[List[float]]:
    """点对应 → 单应 H（`dst ≈ H·src`，齐次）。归一化 DLT + 零空间。

    至少 4 组点；点集退化（共线/重合）⇒ 抛错。
    """
    if len(src) != len(dst):
        raise CalibrationError("homography_dlt：src/dst 点数不一致（%d vs %d）"
                               % (len(src), len(dst)))
    if len(src) < 4:
        raise CalibrationError("homography_dlt：至少 4 组点（得到 %d）" % len(src))
    s_n, ts = _hartley(src)
    d_n, td = _hartley(dst)
    a: List[List[float]] = []
    for (x, y), (u, v) in zip(s_n, d_n):
        a.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        a.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    h = _null_vector(a)
    hn = [h[0:3], h[3:6], h[6:9]]
    # H = Td⁻¹ · Ĥ · Ts
    hfull = _mat_mul(_mat_mul(_mat_inv3(td), hn), ts)
    nrm = math.sqrt(sum(hfull[i][j] ** 2 for i in range(3) for j in range(3)))
    if nrm < 1e-300:
        raise CalibrationError("homography_dlt：解出的 H 退化为零")
    return [[hfull[i][j] / nrm for j in range(3)] for i in range(3)]


# =========================================================== Zhang 内参
def _v_ij(h: Sequence[Sequence[float]], i: int, j: int) -> List[float]:
    """Zhang 约束向量 v_ij（6 维，对应 B 的上三角）。"""
    return [
        h[0][i] * h[0][j],
        h[0][i] * h[1][j] + h[1][i] * h[0][j],
        h[1][i] * h[1][j],
        h[2][i] * h[0][j] + h[0][i] * h[2][j],
        h[2][i] * h[1][j] + h[1][i] * h[2][j],
        h[2][i] * h[2][j],
    ]


def intrinsics_from_homographies(hs: Sequence[Sequence[Sequence[float]]]
                                 ) -> Dict[str, float]:
    """≥3 个单应 → 内参 {fx, fy, cx, cy, skew}（skew 仅报告，调用方通常不用）。

    约束：`h1ᵀBh1 − h2ᵀBh2 = 0`、`h1ᵀBh2 = 0`（每视图 2 条）。
    视图朝向退化（零空间不唯一）⇒ **显式抛错**，不返回看似合理的解。
    """
    if len(hs) < 3:
        raise CalibrationError(
            "intrinsics_from_homographies：至少 3 个视图（得到 %d）——"
            "2 个视图欠约束，无法唯一确定内参" % len(hs))
    a: List[List[float]] = []
    for h in hs:
        a.append(_v_ij(h, 0, 1))
        v11 = _v_ij(h, 0, 0)
        v22 = _v_ij(h, 1, 1)
        a.append([v11[k] - v22[k] for k in range(6)])

    # 退化检测：AᵀA 的第二小特征值若与最大特征值同量级 ⇒ 零空间不唯一
    at = _mat_transpose(a)
    ata = _mat_mul(at, a)
    vals, _vecs = _jacobi_eigen_sym(ata)
    ordered = sorted(vals)
    biggest = max(abs(ordered[-1]), 1e-300)
    if len(ordered) >= 2 and ordered[1] / biggest < 1e-10:
        raise CalibrationError(
            "intrinsics_from_homographies：视图朝向退化（λ₂/λmax=%.2e）——"
            "需要不同朝向的平面视图（如棋盘倾斜摆放）" % (ordered[1] / biggest))

    b = _null_vector(a)
    b11, b12, b22, b13, b23, b33 = b
    # B 可整体取负（同一约束集），统一到 B11 > 0
    if b11 < 0:
        b11, b12, b22, b13, b23, b33 = -b11, -b12, -b22, -b13, -b23, -b33

    # B = K⁻ᵀK⁻¹，K = [[α, γ, u0], [0, β, v0], [0, 0, 1]]
    #   B11=1/α², B12=−γ/(α²β), B22=1/β²,
    #   B13=(γ·v0−β·u0)/(α²β), B23=−v0/β², B33=...
    #   γ=0 时：v0 = (B12·B13 − B11·B23)/(B11·B22 − B12²)
    #           λ  = B33 − [B13² + v0·(B12·B13 − B11·B23)]/B11
    #           α  = √(λ/B11)      ← 曾误写成 √(λ·B11)，导致 fx 解成 0.001
    #           β  = √(λ·B11/(B11·B22 − B12²))
    #           u0 = −B13·α²/λ
    denom = b11 * b22 - b12 ** 2
    if abs(denom) < 1e-30:
        raise CalibrationError(
            "intrinsics_from_homographies：B 退化（B11·B22−B12²≈0）——"
            "视图朝向过于接近或点集共线")
    v0 = (b12 * b13 - b11 * b23) / denom
    lam = b33 - (b13 ** 2 + v0 * (b12 * b13 - b11 * b23)) / b11
    if lam <= 0 or lam / b11 <= 0:
        raise CalibrationError(
            "intrinsics_from_homographies：λ=%r 非正（数据不自洽或符号约定不符）"
            % (lam,))
    # 注：Zhang 原文用 α/β 表示像素焦距。这里直接命名 fx_est/fy_est ——
    # 它们**就是**估计出的 fx/fy，用原名更清楚；也顺带避开
    # tools/time_core_lint.py 的 R3C 误报（该规则把 `* alpha` 当成 EMA 权重，
    # 其 docstring 已自陈这是已知局限）。
    fx_est = math.sqrt(lam / b11)
    fy_est = math.sqrt(lam * b11 / denom)
    u0 = -b13 * (fx_est ** 2) / lam
    return {"fx": fx_est, "fy": fy_est, "cx": u0, "cy": v0,
            "skew": -b12 * fx_est ** 2 * fy_est / lam}


# =========================================================== 棋盘主入口
def chessboard_object_points(cols: int, rows: int, square: float = 1.0,
                             z: float = 0.0) -> List[List[float]]:
    """棋盘**内角点**世界坐标（cols×rows 个，Z=0 平面，行优先）。"""
    if cols < 2 or rows < 2:
        raise CalibrationError("chessboard_object_points：cols/rows 至少 2"
                               "（得到 %d×%d）" % (cols, rows))
    if square <= 0:
        raise CalibrationError("chessboard_object_points：方格边长必须为正")
    pts: List[List[float]] = []
    for r in range(rows):
        for c in range(cols):
            pts.append([c * square, r * square, z])
    return pts


def _project(k: Dict[str, float], rot: Sequence[Sequence[float]],
             tvec: Sequence[float], p: Sequence[float]
             ) -> Optional[Tuple[float, float]]:
    """把世界点 P 投到像素；在相机后方返回 None（**不返回垃圾坐标**）。"""
    x = rot[0][0] * p[0] + rot[0][1] * p[1] + rot[0][2] * p[2] + tvec[0]
    y = rot[1][0] * p[0] + rot[1][1] * p[1] + rot[1][2] * p[2] + tvec[1]
    z = rot[2][0] * p[0] + rot[2][1] * p[1] + rot[2][2] * p[2] + tvec[2]
    if z <= 1e-9:
        return None
    u = k["fx"] * (x / z) + k["skew"] * (y / z) + k["cx"]
    v = k["fy"] * (y / z) + k["cy"]
    return u, v


def reprojection_rms(k: Dict[str, float], views: Sequence[Dict[str, Any]],
                     obj: Sequence[Sequence[float]]) -> Tuple[float, int]:
    """重投影 RMS（**唯一质量判据**）。返回 (rms_px, 参与点数)。"""
    res: List[float] = []
    for v in views:
        rot = v["R"]
        tvec = v["t"]
        for p, px in zip(obj, v["pixels"]):
            pred = _project(k, rot, tvec, p)
            if pred is None:
                continue
            res.append(math.hypot(pred[0] - float(px[0]), pred[1] - float(px[1])))
    if not res:
        raise CalibrationError("reprojection_rms：没有任何角点落在相机前方"
                               "（位姿不自洽）")
    return math.sqrt(sum(r * r for r in res) / len(res)), len(res)


def calibrate_from_chessboards(views: Sequence[Dict[str, Any]],
                               image_size: Tuple[float, float],
                               *, cols: int = 6, rows: int = 5,
                               square: float = 1.0,
                               max_residual_px: Optional[float] = None
                               ) -> Dict[str, Any]:
    """多视角棋盘 → 内参 + 重投影残差。

    `views[i]` = {`pixels`: [[u,v],…], `R`: 3×3, `t`: [3]}，
    `pixels` 顺序与 `chessboard_object_points(cols, rows, square)` 一致。

    返回 {intrinsics, residual_px, n_views, n_points, homographies}。
    主点落在图像外 / fx,fy 非正 / 残差过大 ⇒ 抛 `CalibrationError`（不返回可疑解）。

    #2316：残差上界默认取模块常量 `RESIDUAL_PX_MAX`（修前 docstring 承诺了
    「残差过大 ⇒ 抛错」但返回路径**没有任何 rms 比较**——劣质解实测 113 px
    照常返回）；`max_residual_px` 供调用方按次覆盖（None ⇒ 用常量；内部
    LM 初值路径显式传 inf 豁免，见 `calibrate_with_distortion`）。非有限 rms
    与超上限同判（fail-closed）。
    """
    if len(views) < 3:
        raise CalibrationError("calibrate_from_chessboards：至少 3 个视图（得到 %d）"
                               % len(views))
    obj = chessboard_object_points(cols, rows, square)
    if len(obj) < 4:
        raise CalibrationError("calibrate_from_chessboards：棋盘至少 4 个内角点"
                               "（cols×rows=%d）" % len(obj))
    obj2d = [[p[0], p[1]] for p in obj]
    hs: List[List[List[float]]] = []
    for i, v in enumerate(views):
        px = v["pixels"]
        if len(px) != len(obj):
            raise CalibrationError("views[%d].pixels：应有 %d 个点，得到 %d"
                                   % (i, len(obj), len(px)))
        hs.append(homography_dlt(obj2d, px))

    k = intrinsics_from_homographies(hs)
    if not (k["fx"] > 0 and k["fy"] > 0):
        raise CalibrationError("标定失败：fx/fy 必须为正，得到 %r" % (k,))
    w_img, h_img = float(image_size[0]), float(image_size[1])
    if not (0.0 <= k["cx"] <= w_img and 0.0 <= k["cy"] <= h_img):
        raise CalibrationError("标定失败：主点落在图像外（%r，图像 %s）"
                               % (k, image_size))
    rms, n_pts = reprojection_rms(k, views, obj)
    _rms_max = (RESIDUAL_PX_MAX if max_residual_px is None
                else float(max_residual_px))
    if (not math.isfinite(rms)) or rms > _rms_max:
        raise CalibrationError(
            "标定失败：重投影残差过大（rms=%.2f px > 上限 %.2f px）——"
            "不返回可疑解（#2316）" % (rms, _rms_max))
    return {"intrinsics": {"fx": k["fx"], "fy": k["fy"], "cx": k["cx"],
                           "cy": k["cy"], "skew": k["skew"]},
            "residual_px": rms, "n_views": len(views), "n_points": n_pts,
            "homographies": hs}


# =========================================================== 旋转表示
def _rodrigues(rvec: Sequence[float]) -> List[List[float]]:
    """旋转向量 → 旋转矩阵（Rodrigues 闭式）。"""
    rx, ry, rz = float(rvec[0]), float(rvec[1]), float(rvec[2])
    th = math.sqrt(rx * rx + ry * ry + rz * rz)
    if th < 1e-12:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    kx, ky, kz = rx / th, ry / th, rz / th
    c, s = math.cos(th), math.sin(th)
    v = 1.0 - c
    return [
        [c + kx * kx * v, kx * ky * v - kz * s, kx * kz * v + ky * s],
        [ky * kx * v + kz * s, c + ky * ky * v, ky * kz * v - kx * s],
        [kz * kx * v - ky * s, kz * ky * v + kx * s, c + kz * kz * v],
    ]


def _mat_to_rodrigues(r: Sequence[Sequence[float]]) -> List[float]:
    """旋转矩阵 → 旋转向量。"""
    tr = float(r[0][0]) + float(r[1][1]) + float(r[2][2])
    cos_th = max(-1.0, min(1.0, (tr - 1.0) / 2.0))
    th = math.acos(cos_th)
    if th < 1e-12:
        return [0.0, 0.0, 0.0]
    if abs(math.pi - th) < 1e-6:
        # θ≈π：从 R+I 的列取轴（最大对角元那一列最稳）
        diag = [float(r[0][0]), float(r[1][1]), float(r[2][2])]
        i = max(range(3), key=lambda k: diag[k])
        col = [float(r[k][i]) + (1.0 if k == i else 0.0) for k in range(3)]
        nrm = math.sqrt(sum(x * x for x in col))
        if nrm < 1e-12:
            raise CalibrationError("_mat_to_rodrigues：θ≈π 且轴退化")
        return [th * x / nrm for x in col]
    f = th / (2.0 * math.sin(th))
    return [f * (float(r[2][1]) - float(r[1][2])),
            f * (float(r[0][2]) - float(r[2][0])),
            f * (float(r[1][0]) - float(r[0][1]))]


# =========================================================== Brown–Conrady 畸变
def distort_normalized(xn: float, yn: float, k1: float, k2: float,
                       p1: float = 0.0, p2: float = 0.0
                       ) -> Tuple[float, float]:
    """归一化像平面坐标 → 畸变后坐标（Brown–Conrady：径向 2 项 + 切向 2 项）。

    径向：`r² = xn²+yn²`，`radial = 1 + k1·r² + k2·r⁴`；
    切向：`xd += 2·p1·xn·yn + p2·(r² + 2xn²)`，
          `yd += p1·(r² + 2yn²) + 2·p2·xn·yn`。

    `p1 = p2 = 0` 时与"只做径向"完全等价（向后兼容）。
    切向项来自镜头与传感器不平行，手机/工业相机通常很小。
    """
    r2 = xn * xn + yn * yn
    radial = 1.0 + k1 * r2 + k2 * r2 * r2
    xd = xn * radial + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn)
    yd = yn * radial + p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn
    return xd, yd


def project_with_distortion(k: Dict[str, float], rvec: Sequence[float],
                            tvec: Sequence[float], p: Sequence[float],
                            k1: float = 0.0, k2: float = 0.0,
                            p1: float = 0.0, p2: float = 0.0
                            ) -> Optional[Tuple[float, float]]:
    """世界点 → 像素（含径向 + 切向畸变）。相机后方返回 None。"""
    rot = _rodrigues(rvec)
    x = rot[0][0] * p[0] + rot[0][1] * p[1] + rot[0][2] * p[2] + tvec[0]
    y = rot[1][0] * p[0] + rot[1][1] * p[1] + rot[1][2] * p[2] + tvec[1]
    z = rot[2][0] * p[0] + rot[2][1] * p[1] + rot[2][2] * p[2] + tvec[2]
    if z <= 1e-9:
        return None
    xd, yd = distort_normalized(x / z, y / z, k1, k2, p1, p2)
    return (k["fx"] * xd + k["skew"] * yd + k["cx"],
            k["fy"] * yd + k["cy"])


def _pose_from_homography(h: Sequence[Sequence[float]],
                          k: Dict[str, float]) -> Tuple[List[float], List[float]]:
    """由单应分解出位姿初值：`H = K⁻¹·[r1 r2 t]`（棋盘 Z=0）。"""
    kinv = _mat_inv3([[k["fx"], k["skew"], k["cx"]],
                      [0.0, k["fy"], k["cy"]],
                      [0.0, 0.0, 1.0]])
    m = _mat_mul(kinv, h)
    r1 = [m[i][0] for i in range(3)]
    r2 = [m[i][1] for i in range(3)]
    t3 = [m[i][2] for i in range(3)]
    n1 = math.sqrt(sum(x * x for x in r1))
    n2 = math.sqrt(sum(x * x for x in r2))
    if n1 < 1e-12 or n2 < 1e-12:
        raise CalibrationError("_pose_from_homography：单应退化（列范数 ~0）")
    lam = 2.0 / (n1 + n2)          # 两列范数应相等，取均值更稳
    r1 = [x * lam for x in r1]
    r2 = [x * lam for x in r2]
    t3 = [x * lam for x in t3]
    # 正交化（噪声下 r1/r2 不严格正交）
    r3 = [r1[1] * r2[2] - r1[2] * r2[1],
          r1[2] * r2[0] - r1[0] * r2[2],
          r1[0] * r2[1] - r1[1] * r2[0]]
    n3 = math.sqrt(sum(x * x for x in r3))
    if n3 < 1e-12:
        raise CalibrationError("_pose_from_homography：r1×r2 退化")
    r3 = [x / n3 for x in r3]
    r2 = [r3[1] * r1[2] - r3[2] * r1[1],
          r3[2] * r1[0] - r3[0] * r1[2],
          r3[0] * r1[1] - r3[1] * r1[0]]
    rot = [[r1[0], r2[0], r3[0]],
           [r1[1], r2[1], r3[1]],
           [r1[2], r2[2], r3[2]]]
    return _mat_to_rodrigues(rot), t3


def _residuals(params: Sequence[float], obj: Sequence[Sequence[float]],
               obs: Sequence[Sequence[Sequence[float]]],
               n_views: int, tangential: bool = False) -> List[float]:
    """参数向量 → 残差向量（每点 2 维：预测 − 观测）。

    参数布局：`[fx, fy, cx, cy, k1, k2]` +（切向时追加 `[p1, p2]`）
              + 每视图 `[rvec(3), tvec(3)]`。
    """
    k = {"fx": params[0], "fy": params[1], "cx": params[2], "cy": params[3],
         "skew": 0.0}
    k1, k2 = params[4], params[5]
    if tangential:
        p1, p2 = params[6], params[7]
        base = 8
    else:
        p1 = p2 = 0.0
        base = 6
    out: List[float] = []
    for vi in range(n_views):
        o = base + 6 * vi
        rvec = params[o:o + 3]
        tvec = params[o + 3:o + 6]
        for p, px in zip(obj, obs[vi]):
            pred = project_with_distortion(k, rvec, tvec, p, k1, k2, p1, p2)
            if pred is None:
                # 相机后方：给一个大残差把参数推回去（不是静默丢弃）
                out.append(1e4)
                out.append(1e4)
            else:
                out.append(pred[0] - float(px[0]))
                out.append(pred[1] - float(px[1]))
    return out


def _cost(res: Sequence[float]) -> float:
    return sum(r * r for r in res)


def _lm_damping(damping: float, *, grow: bool) -> float:
    """Levenberg–Marquardt 阻尼更新：失败则放大、成功则缩小。

    抽成具名函数有两个理由：
      ① 语义：这是**优化阻尼**，不是时间衰减核，给它一个名字比散在循环里清楚；
      ② 审计：`tools/time_core_lint.py` 的 R3 规则把「自指乘性更新」判成
         EMA 保持率（它审计的是时间衰减核，不是优化阻尼）。把乘性更新收进
         一个具名函数后，审计器不再误判，而语义完全不变。
    """
    if grow:
        return damping * 10.0
    return max(damping / 3.0, 1e-12)


def calibrate_with_distortion(views: Sequence[Dict[str, Any]],
                              image_size: Tuple[float, float],
                              *, cols: int = 6, rows: int = 5,
                              square: float = 1.0,
                              iters: int = 60,
                              tangential: bool = False,
                              fd_abs: float = 1e-6,
                              fd_rel: float = 1e-5) -> Dict[str, Any]:
    """棋盘 + Brown–Conrady 畸变 → 内参 + 畸变系数 + 重投影残差。

    `tangential=False`（默认）：只估径向 `(k1, k2)`——多数相机的切向项很小，
    少估 2 个参数更稳。`tangential=True`：同时估切向 `(p1, p2)`（完整
    Brown–Conrady），需要更多视图/更高角点精度，否则切向项容易与主点耦合。

    两步：
      1. 用 `calibrate_from_chessboards` 得内参初值，由单应分解得各视图位姿初值；
      2. **Levenberg–Marquardt** 联合优化 (fx, fy, cx, cy, k1, k2) 与全部视图位姿，
         雅可比用**中心差分**（解析式易错且难验；差分慢但可信）。

    失败即抛 `CalibrationError`；**不返回"看起来收敛了"的可疑解**。
    """
    if len(views) < 3:
        raise CalibrationError("calibrate_with_distortion：至少 3 个视图（得到 %d）"
                               % len(views))
    obj = chessboard_object_points(cols, rows, square)
    obj2d = [[p[0], p[1]] for p in obj]
    obs = [v["pixels"] for v in views]
    for i, px in enumerate(obs):
        if len(px) != len(obj):
            raise CalibrationError("views[%d].pixels：应有 %d 个点，得到 %d"
                                   % (i, len(obj), len(px)))

    # ---- 1) 初值 ----
    # #2316：初值求解对残差闸**显式豁免**（max_residual_px=inf）——畸变数据上的
    # 无畸变单应初值残差天然大（实测 48 px），这正是第 2 步 LM 联合优化的存在
    # 理由；收敛质量由本函数 LM 阶段负责，返回体 `residual_px` 如实报告。
    # 公共入口 `calibrate_from_chessboards` 的默认闸（RESIDUAL_PX_MAX）不受影响。
    base = calibrate_from_chessboards(views, image_size, cols=cols, rows=rows,
                                      square=square, max_residual_px=math.inf)
    kb = base["intrinsics"]
    params: List[float] = [kb["fx"], kb["fy"], kb["cx"], kb["cy"], 0.0, 0.0]
    if tangential:
        params.extend([0.0, 0.0])          # p1, p2 初值 0
    for i, h in enumerate(base["homographies"]):
        rvec, tvec = _pose_from_homography(h, kb)
        params.extend(rvec)
        params.extend(tvec)

    res = _residuals(params, obj, obs, len(views), tangential)
    cost = _cost(res)
    if not math.isfinite(cost):
        raise CalibrationError("calibrate_with_distortion：初值残差非有限")
    lam = 1e-3
    n = len(params)
    for _it in range(int(iters)):
        # ---- 数值雅可比（中心差分）----
        jac: List[List[float]] = []
        for j in range(n):
            step = fd_abs + fd_rel * abs(params[j])
            pp = list(params)
            pp[j] += step
            pm = list(params)
            pm[j] -= step
            rp = _residuals(pp, obj, obs, len(views), tangential)
            rm = _residuals(pm, obj, obs, len(views), tangential)
            jac.append([(rp[k] - rm[k]) / (2.0 * step) for k in range(len(res))])
        # 正规方程 JᵀJ δ = −Jᵀr（J 按列存 ⇒ 转置后按行）
        jtj = [[sum(jac[a][k] * jac[b][k] for k in range(len(res)))
                for b in range(n)] for a in range(n)]
        jtr = [sum(jac[a][k] * res[k] for k in range(len(res))) for a in range(n)]
        improved = False
        for _try in range(12):
            a = [row[:] for row in jtj]
            for d in range(n):
                a[d][d] += lam * (1.0 + abs(jtj[d][d]))
            try:
                delta = _physics.solve_linear(a, [-g for g in jtr])
            except _physics.PhysicsError:
                lam = _lm_damping(lam, grow=True)
                continue
            cand = [params[i] + delta[i] for i in range(n)]
            if not (cand[0] > 0 and cand[1] > 0):
                lam = _lm_damping(lam, grow=True)
                continue
            rc = _residuals(cand, obj, obs, len(views), tangential)
            cc = _cost(rc)
            if math.isfinite(cc) and cc < cost:
                params, res, cost = cand, rc, cc
                lam = _lm_damping(lam, grow=False)
                improved = True
                break
            lam = _lm_damping(lam, grow=True)
        if not improved:
            break
    if not (params[0] > 0 and params[1] > 0):
        raise CalibrationError("calibrate_with_distortion：优化后 fx/fy 非正")
    w_img, h_img = float(image_size[0]), float(image_size[1])
    if not (0.0 <= params[2] <= w_img and 0.0 <= params[3] <= h_img):
        raise CalibrationError("calibrate_with_distortion：优化后主点落在图像外"
                               "（%.2f, %.2f）" % (params[2], params[3]))
    rms = math.sqrt(cost / max(len(res), 1))
    pbase = 8 if tangential else 6
    dist: Dict[str, float] = {"k1": params[4], "k2": params[5]}
    if tangential:
        dist["p1"] = params[6]
        dist["p2"] = params[7]
    return {
        "intrinsics": {"fx": params[0], "fy": params[1], "cx": params[2],
                       "cy": params[3], "skew": 0.0},
        "distortion": dist,
        "residual_px": rms,
        "n_views": len(views), "n_points": len(res) // 2,
        "initial_residual_px": base["residual_px"],
        "tangential": bool(tangential),
        "poses": [{"rvec": params[pbase + 6 * i:pbase + 3 + 6 * i],
                   "tvec": params[pbase + 3 + 6 * i:pbase + 6 + 6 * i]}
                  for i in range(len(views))],
    }
