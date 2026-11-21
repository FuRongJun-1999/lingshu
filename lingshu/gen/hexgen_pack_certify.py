"""**同格多件的铺位穷举证书**（R313，旁路新建；白箱：纯标准库 + numpy）。

**为什么需要它**：R311/R312 把「判决带缩径」从 9 条压到 2 条，残余 2 条（`{303:47, 405:32}`）的
上界此前只能报**两条启发式搜索的带内一致**（生产 47 vs 参照 48；生产 32 vs 参照 27）——这是
「两个搜索都没找到」而不是「不存在」。本文件给一个**穷举证书**：在 **1px 网格**上判定
「半径 r 下是否存在**两两满足方向相关分离要求**的合法摆位」：

  * **枚举 p1**（第一件的质心）遍历**合法区域**的每个整数像素；
  * 固定 p1 后，`S1` ＝「与 p1 兼容且自身合法」的位置集合（合法 ∧ 距离 ≥ `need(p1,·)`）；
  * **p2/p3 的穷举靠差集**：`(S1 − S1) ∋ δ` ⟺ `S1` 与 `S1` 平移 δ 后相交 ⇒ 用**自相关**（FFT）
    一次性算出全部位移的**配对计数**，再取 `max_{δ∈差集, δ≠0} (|δ| − need(δ))` ＝ 该 p1 下的最优余量。
  * 判据：**存在 p1 使该余量 ≥ 0** ⟺ 可行。若全部 p1 的最优余量 < 0 ⇒ **该半径无解（证书）**。

**为什么这不是「又一个启发式」**：p1 是**穷举**的（1px 网格 = 生产实际允许的位置），p2/p3 通过差集
**穷举**（自相关给出所有位移对的计数，凡 `计数 > 0` 的位移都被检查）⇒ 结论是**判定**而非搜索。
代价：每个 p1 一次 1024² 的 FFT（≈ 10–20ms）⇒ 单个 (题面, 半径) 约 1–5 分钟。

**与生产的判据面严格同源**：`_support_vec`/`_need_vec` 与 `hexgen_self_source.paint_support`/
`_need_between` 同式（守门逐点比对，见 `tests/test_hexgen_pack_certify_r313.py`）。

用法：
    python -X utf8 -m lingshu.gen.hexgen_pack_certify --prompt 303 --r 47 --r 48
"""
import argparse
from collections.abc import Mapping
import json
import math
import os
import sys

import numpy as np

from . import hexgen_self_source as S          # noqa: E402

SIZE = S.SIZE
_MESH = {}
_VALID_CELLS = frozenset(f"r{i}" for i in range(9))
_ANISOTROPIC_SHAPES = frozenset(("square", "rectangle", "diamond"))


def _require_finite_real(val, name):
    """要求有限实数并拒绝 bool / np.bool_，违者抛 ValueError。"""
    if isinstance(val, (bool, np.bool_)):
        raise ValueError(f"{name} must not be bool, got {val!r}")
    if not isinstance(val, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a real number, got {type(val).__name__}")
    out = float(val)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {val!r}")
    return out


def _require_pos_finite(val, name):
    """要求严格正有限实数（> 0），拒绝 bool / 非有限 / <= 0。"""
    out = _require_finite_real(val, name)
    if out <= 0.0:
        raise ValueError(f"{name} must be > 0, got {val!r}")
    return out


def _require_nonneg_finite(val, name):
    """要求非负有限实数（>= 0），拒绝 bool / 非有限 / < 0。"""
    out = _require_finite_real(val, name)
    if out < 0.0:
        raise ValueError(f"{name} must be >= 0, got {val!r}")
    return out


def _require_int(val, name):
    """要求整数（拒绝 bool / np.bool_ / float / 非数值）。"""
    if isinstance(val, (bool, np.bool_)):
        raise ValueError(f"{name} must not be bool, got {val!r}")
    if not isinstance(val, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {type(val).__name__}")
    return int(val)


def _require_pos_int(val, name):
    """要求正整数（>= 1），拒绝 bool / 非整数 / <= 0。"""
    out = _require_int(val, name)
    if out <= 0:
        raise ValueError(f"{name} must be > 0, got {val!r}")
    return out


def _require_n(n):
    """要求同格件数 n ∈ {2, 3}（拒绝 bool 与越界值）。"""
    val = _require_int(n, "n")
    if val not in (2, 3):
        raise ValueError(f"n must be 2 or 3, got {n!r}")
    return val


def _validate_shape(shape):
    """校验并归一形状名；未知或非字符串抛 ValueError。"""
    if not isinstance(shape, str):
        raise ValueError(f"shape must be a string, got {type(shape).__name__}")
    norm = S._v3_shape(shape)
    if norm not in S.EXTENT_FRAC:
        raise ValueError(f"unknown shape: {shape!r}")
    return norm


def _validate_cells(cells):
    """校验格集合：非空且元素均为 'r0'..'r8'。"""
    if cells is None or isinstance(cells, (str, bytes)):
        raise ValueError(f"cells must be a non-string iterable, got {type(cells).__name__}")
    try:
        items = list(cells)
    except TypeError as exc:
        raise ValueError(f"cells must be iterable, got {type(cells).__name__}") from exc
    if not items:
        raise ValueError("cells must not be empty")
    for c in items:
        if not isinstance(c, str) or c not in _VALID_CELLS:
            raise ValueError(f"invalid cell {c!r}; expected 'r0'..'r8'")
    return tuple(items)


def _validate_zone_calib(zone_calib):
    """校验方位标定切点（仅 None 或空 Mapping 回落精确三等分；否则要求有限非 bool 且 0<=c0<=c1<=1）。"""
    if zone_calib is None or (isinstance(zone_calib, Mapping) and not zone_calib):
        return [1 / 3.0, 2 / 3.0], [1 / 3.0, 2 / 3.0]
    if not isinstance(zone_calib, Mapping):
        raise ValueError(f"zone_calib must be a mapping or None, got {type(zone_calib).__name__}")
    parsed = {}
    for axis in ("row", "col"):
        sub = zone_calib.get(axis)
        if not isinstance(sub, Mapping) or "cuts" not in sub:
            raise ValueError(f"zone_calib[{axis!r}] must be a mapping containing 'cuts'")
        raw_cuts = sub["cuts"]
        if raw_cuts is None or isinstance(raw_cuts, (str, bytes)):
            raise ValueError(f"zone_calib[{axis!r}]['cuts'] must be a 2-element sequence")
        try:
            cuts_list = list(raw_cuts)
        except TypeError as exc:
            raise ValueError(f"zone_calib[{axis!r}]['cuts'] must be iterable") from exc
        if len(cuts_list) != 2:
            raise ValueError(f"zone_calib[{axis!r}]['cuts'] must have length 2, got {len(cuts_list)}")
        c0 = _require_finite_real(cuts_list[0], f"zone_calib[{axis!r}]['cuts'][0]")
        c1 = _require_finite_real(cuts_list[1], f"zone_calib[{axis!r}]['cuts'][1]")
        if not (0.0 <= c0 <= c1 <= 1.0):
            raise ValueError(
                f"zone_calib[{axis!r}]['cuts'] must satisfy 0 <= c0 <= c1 <= 1, got {[c0, c1]!r}"
            )
        parsed[axis] = [c0, c1]
    return parsed["row"], parsed["col"]


def _has_bool_elem(val):
    if isinstance(val, (bool, np.bool_)):
        return True
    if isinstance(val, (list, tuple)):
        return any(_has_bool_elem(x) for x in val)
    return False


def _require_finite_array(val, name):
    """将标量或数组转为 float64 ndarray，拒绝 bool / np.bool_ 及任何非有限元素。"""
    if _has_bool_elem(val):
        raise ValueError(f"{name} must not contain bool")
    raw = np.asarray(val)
    if np.issubdtype(raw.dtype, np.bool_):
        raise ValueError(f"{name} must not have bool dtype")
    if not (np.issubdtype(raw.dtype, np.integer) or np.issubdtype(raw.dtype, np.floating)):
        raise ValueError(f"{name} must be real numeric scalar or array, got dtype={raw.dtype}")
    arr = raw.astype(np.float64, copy=False)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _validate_mask(mask):
    """校验合成 mask：2D 布尔或 0/1 有限数组，拒绝 NaN/inf 静默转 True。"""
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"mask must be 2D, got ndim={arr.ndim}")
    if np.issubdtype(arr.dtype, np.bool_):
        return arr.copy()
    if np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating):
        if np.issubdtype(arr.dtype, np.floating) and not np.all(np.isfinite(arr)):
            raise ValueError("mask contains non-finite values")
        if not np.all((arr == 0) | (arr == 1)):
            raise ValueError("numeric mask must contain only 0/1 values")
        return arr.astype(bool)
    raise ValueError(f"mask must be boolean or 0/1 numeric array, got dtype={arr.dtype}")


def _mesh(size=SIZE):
    size = _require_pos_int(size, "size")
    if size not in _MESH:
        y, x = np.mgrid[0:size, 0:size]
        _MESH[size] = (x.astype(np.int32), y.astype(np.int32))
    return _MESH[size]


def _rank_ok_vec(zone_calib, cells, size=SIZE):
    """逐像素「判决格 ∈ 允许集合」的布尔掩膜（与 `S._cell_of_pt` 同口径：冻结切点，缺省精确三等分）。"""
    size = _require_pos_int(size, "size")
    cells = _validate_cells(cells)
    rc, cc = _validate_zone_calib(zone_calib)
    x, y = _mesh(size)
    fy = y / float(size)
    fx = x / float(size)
    row = np.where(fy < rc[0], 0, np.where(fy < rc[1], 1, 2))
    col = np.where(fx < cc[0], 0, np.where(fx < cc[1], 1, 2))
    cell = row * 3 + col
    allow = np.zeros(9, dtype=bool)
    for c in cells:
        allow[int(c[1])] = True
    return allow[cell]


def legal_mask(cells, shape, r, zone_calib, size=SIZE):
    """**合法区域**：实物整体在画布内 ∧ 质心的判决格 ∈ 允许集合（＝生产 `fits` 的同一谓词）。"""
    cells = _validate_cells(cells)
    shape = _validate_shape(shape)
    r = _require_pos_finite(r, "r")
    size = _require_pos_int(size, "size")
    _validate_zone_calib(zone_calib)
    e = S.paint_extent(shape, r) + 2.0
    x, y = _mesh(size)
    m = (x >= e) & (x <= size - e) & (y >= e) & (y <= size - e)
    return m & _rank_ok_vec(zone_calib, cells, size)


def support_vec(shape, r, dx, dy, dist):
    """`S.paint_support` 的向量化版本（逐点等价，守门比对）。

    #2131：入口先 `_v3_shape` 归一——生产侧 paint_support 首行就做
    （自家名 `stripe` ≡ 真实名 `square`，SHAPE_SELF2V3）。修前 stripe 落到
    末行的 paint_extent 各向同性回退（r=40、对角给 41.0），生产面归一后走
    L1 支撑给 57.57——「与生产的判据面严格同源」声明被打破，可行方向判决
    不健全（证书 41 而生产 57.57，slack 判错号）。同模块 legal_mask 在
    paint_extent 前已有归一——支撑侧一张归一、一张不归一，模块内自相矛盾。
    """
    shape = _validate_shape(shape)
    r = _require_pos_finite(r, "r")
    dx_arr = _require_finite_array(dx, "dx")
    dy_arr = _require_finite_array(dy, "dy")
    dist_arr = _require_finite_array(dist, "dist")
    if np.any(dist_arr < 0.0):
        raise ValueError("dist must be >= 0")
    if shape in _ANISOTROPIC_SHAPES:
        if np.any(dist_arr <= 0.0):
            raise ValueError(f"dist must be > 0 for anisotropic shape {shape!r}")
        ax = np.abs(dx_arr) / dist_arr
        ay = np.abs(dy_arr) / dist_arr
        if shape == "square":
            return r * (ax + ay) + 1.0
        if shape == "rectangle":
            return 1.35 * r * ax + 0.62 * r * ay + 1.0
        return 1.15 * r * np.maximum(ax, ay) + 1.0
    if shape == "circle":
        return np.zeros_like(dist_arr, dtype=np.float64) + (r + 1.0)
    return np.full_like(dist_arr, S.paint_extent(shape, r), dtype=np.float64)


def slack_vec(shape, r, dx, dy,
              sep=S.SEP_GAP, pad=2.0):
    """`|δ| − need(δ)` 的向量化版本；`δ = 0` 处置 `-inf`（同一位置不算一对）。"""
    shape = _validate_shape(shape)
    r = _require_pos_finite(r, "r")
    sep = _require_nonneg_finite(sep, "sep")
    pad = _require_nonneg_finite(pad, "pad")
    dx_arr = _require_finite_array(dx, "dx")
    dy_arr = _require_finite_array(dy, "dy")
    dx_arr, dy_arr = np.broadcast_arrays(dx_arr, dy_arr)
    dist = np.hypot(dx_arr, dy_arr)
    out = np.full(dist.shape, -np.inf, dtype=np.float64)
    nz = dist > 0
    if np.any(nz):
        d = dist[nz]
        need = (support_vec(shape, r, dx_arr[nz], dy_arr[nz], d) * 2.0 + sep + pad)
        out[nz] = d - need
    return out


def _cert_p1_core(x1, y1, L, n, shape, r, size):
    """单个 p1 的最优余量（与 `certify_radius` **共用**同一实现 ⇒ 抽样与全量不可漂移）。"""
    ys, xs = np.nonzero(L)
    if ys.size == 0:
        #   合法区域为空 ⇒ 不存在合法 p1 ⇒ 判「无解」（`max_slack=-inf`、`n1=0`），
        #   不让 `ys.min()` 抛 ValueError。判据来源：本文件 :12 证书语义「全部 p1 的最优余量 < 0
        #   ⇒ 该半径无解」的退化极值。
        return float("-inf"), 0
    by0, by1 = int(ys.min()), int(ys.max())
    bx0, bx1 = int(xs.min()), int(xs.max())
    hh, ww = by1 - by0 + 1, bx1 - bx0 + 1
    Lc = L[by0:by1 + 1, bx0:bx1 + 1]
    ngh, ngw = 2 * hh, 2 * ww
    yy, xx = np.mgrid[by0:by1 + 1, bx0:bx1 + 1]
    sl = slack_vec(shape, r, xx - int(x1), yy - int(y1))
    S1 = Lc & np.isfinite(sl) & (sl >= 0.0)
    if n == 2:
        return (float(sl[S1].max()) if S1.any() else float("-inf")), int(S1.sum())
    if int(S1.sum()) < 2:
        return float("-inf"), int(S1.sum())
    F = np.fft.rfft2(S1.astype(np.float32), s=(ngh, ngw))
    ac = np.fft.irfft2(F * np.conj(F), s=(ngh, ngw))
    lagy = np.arange(ngh)
    lagy = np.where(lagy < hh, lagy, lagy - ngh).astype(np.float64)
    lagx = np.arange(ngw)
    lagx = np.where(lagx < ww, lagx, lagx - ngw).astype(np.float64)
    dyy, dxx = np.meshgrid(lagy, lagx, indexing="ij")
    sl_grid = slack_vec(shape, r, dxx, dyy)
    cand = np.where((ac > 0.5) & np.isfinite(sl_grid) & (sl_grid >= 0.0), sl_grid, -np.inf)
    return (float(np.max(cand)) if cand.size else float("-inf")), int(S1.sum())


def _validate_cert_inputs(cells, n, shape, r, zone_calib, size):
    """统一校验证书入口共用参数（cells / n / shape / r / zone_calib / size）。"""
    n_val = _require_n(n)
    cells_val = _validate_cells(cells)
    shape_val = _validate_shape(shape)
    r_val = _require_pos_finite(r, "r")
    size_val = _require_pos_int(size, "size")
    _validate_zone_calib(zone_calib)
    return cells_val, n_val, shape_val, r_val, size_val


def certify_radius(cells, n, shape, r, zone_calib, size=SIZE, verbose=False, max_p1=None, mask=None):
    """**穷举证书**：返回 dict(`feasible`, `max_slack`, `best_p1`, `n_p1`, `per_p1`)。

    ⚠ **`max_slack` 的语义**（R313 守门对拍时澄清）：它是「**配对项** slack 的最大值」——固定 p1 后
    只要求 `p2/p3` 与 p1 **相容**（slack ≥ 0），报告的是 `(p2,p3)` 那一对的余量，**不要求**它与 p1 的
    余量也大。因此 `max_slack ≥ 三点的「最大最小余量」`（是它的上界量），**判定只看 `feasible`**；
    `-inf` 表示「每个 p1 都找不到两个与自己相容的位置」＝更强的结构性结论。
    `n ∈ {2, 3}`（同格 k 件；R313 只服务残余的 k=3，k=2 走直判路径）。
    `mask` 只供**守门**用：用小合成合法掩膜把本机制与暴力穷举对拍。"""
    cells, n, shape, r, size = _validate_cert_inputs(cells, n, shape, r, zone_calib, size)
    if not isinstance(verbose, (bool, np.bool_)):
        raise ValueError(f"verbose must be bool, got {type(verbose).__name__}")
    if max_p1 is not None:
        max_p1 = _require_pos_int(max_p1, "max_p1")
    L = legal_mask(cells, shape, r, zone_calib, size) if mask is None else _validate_mask(mask)
    if not L.any():
        #   合法区域为空 ⇒ 该半径下不存在任何合法 p1 ⇒ **判「无解」**（`feasible=False`）而不是让
        #   `ys.min()` 抛 ValueError。判据来源：本文件 :12 证书语义「若全部 p1 的最优余量 < 0
        #   ⇒ 该半径无解（证书）」；:68 合法区域定义。`per_p1=[]` ⇒ `n_p1=0`（零个可枚举 p1）。
        return dict(feasible=False, max_slack=float("-inf"), best_p1=None,
                    n_p1=0, per_p1=[])
    ys, xs = np.nonzero(L)
    #   ⚠ **R313 提速②**：**包围盒裁剪**——`S1 ⊆ L` ⇒ 任何可实现的位移都落在合法区域的包围盒内
    #   （分量 ≤ 边长−1）⇒ 自相关只需在裁剪后的尺寸上做（合法区域通常远小于 512²，实测单次证书
    #   由 ~6 分钟降到 ~1 分钟），而不是整幅 512² 的 1024² FFT。
    by0, by1 = int(ys.min()), int(ys.max())
    bx0, bx1 = int(xs.min()), int(xs.max())
    hh, ww = by1 - by0 + 1, bx1 - bx0 + 1
    Lc = L[by0:by1 + 1, bx0:bx1 + 1]
    ngh, ngw = 2 * hh, 2 * ww                        # 自相关要线性卷积 ⇒ 补零到 2×
    per_p1, best, best_p1 = [], -np.inf, None
    yy, xx = np.mgrid[by0:by1 + 1, bx0:bx1 + 1]
    yy = yy.astype(np.int32)
    xx = xx.astype(np.int32)
    ys, xs = np.nonzero(Lc)                          # 裁剪系下的 p1
    #   ⚠ **R313 提速①**（实现落在 `_cert_p1_core`）：位移网格上的 `slack(δ)` 与 `admissible(δ)`
    #   只取决于 (shape, r)、与 p1 无关；配合**包围盒裁剪**后单个 p1 的位移网格只有 2h×2w（实测
    #   405 的窗口 184×64 ⇒ 128×368）⇒ 单次证书从「十余分钟」降到**分钟级**。
    for k in range(len(xs)):
        if max_p1 is not None and k >= max_p1:
            break
        x1, y1 = int(xs[k]) + bx0, int(ys[k]) + by0
        if n == 2:
            sl = slack_vec(shape, r, xx - x1, yy - y1)
            S1 = Lc & np.isfinite(sl) & (sl >= 0.0)
            #   k=2：只要存在一个与 p1 兼容的合法位置即成（差集退化为单点）
            per_p1.append((x1, y1, float(sl[S1].max()) if S1.any() else float("-inf")))
            if S1.any() and per_p1[-1][2] > best:
                best, best_p1 = per_p1[-1][2], (x1, y1)
            continue
        v, _n1 = _cert_p1_core(x1, y1, L, n, shape, r, size)
        if v <= float("-inf") and n == 3 and _n1 < 2:
            per_p1.append((x1, y1, float("-inf")))
            continue
        per_p1.append((x1, y1, v))
        if v > best:
            best, best_p1 = v, (x1, y1)
        if verbose and k % 500 == 0:
            print(f"  p1 {k}/{len(xs)} @({x1},{y1}) 当前最优余量 {best:.3f}", flush=True)
    if n == 2:
        #   k=2：只需「存在一条兼容对」⇒ 任一 p1 的 S1 非空即可（差集退化为单点）
        best = max((v for _, _, v in per_p1), default=-np.inf)
        feas = any(v > -np.inf for _, _, v in per_p1)
        return dict(feasible=bool(feas), max_slack=float(best), best_p1=best_p1,
                    n_p1=len(per_p1), per_p1=per_p1)
    return dict(feasible=bool(best >= 0.0), max_slack=float(best), best_p1=best_p1,
                n_p1=len(per_p1), per_p1=per_p1)


def certify_p1(x1, y1, cells, n, shape, r, zone_calib, size=SIZE, mask=None):
    """**单个 p1** 的最优余量（＝`certify_radius` 的内层；供守门抽样复算与诊断）。"""
    x1 = _require_int(x1, "x1")
    y1 = _require_int(y1, "y1")
    cells, n, shape, r, size = _validate_cert_inputs(cells, n, shape, r, zone_calib, size)
    L = legal_mask(cells, shape, r, zone_calib, size) if mask is None else _validate_mask(mask)
    return _cert_p1_core(x1, y1, L, n, shape, r, size)


def witness_at(cells, n, shape, r, zone_calib, size=SIZE, mask=None):
    """在半径 r 上**取出一组可行摆位**（证书内层：p1 遍历 + 差集找 δ + 回找 p2）——返回 pts 或 None。

    ⚠ 只在证书判定**可行**时有意义；用途是「把最优摆位的结构看出来」与作为记录证据。"""
    cells, n, shape, r, size = _validate_cert_inputs(cells, n, shape, r, zone_calib, size)
    L = legal_mask(cells, shape, r, zone_calib, size) if mask is None else _validate_mask(mask)
    if not L.any():
        #   合法区域为空 ⇒ 无可行摆位 ⇒ 返回 None（本函数语义：pts 或 None），不让 `ys.min()` 抛。
        #   判据来源：本文件 :197 docstring「返回 pts 或 None」+ :12 证书语义（空合法区域 ⇒ 无解）。
        return None
    ys, xs = np.nonzero(L)
    by0, by1 = int(ys.min()), int(ys.max())
    bx0, bx1 = int(xs.min()), int(xs.max())
    hh, ww = by1 - by0 + 1, bx1 - bx0 + 1
    Lc = L[by0:by1 + 1, bx0:bx1 + 1]
    yy, xx = np.mgrid[by0:by1 + 1, bx0:bx1 + 1]
    ys, xs = np.nonzero(Lc)          #   ⚠ 必须换成**裁剪系**索引（首版沿用全局索引又加一次偏移 ⇒ p1 跑到画布外）
    ngh, ngw = 2 * hh, 2 * ww
    lagy = np.arange(ngh)
    lagy = np.where(lagy < hh, lagy, lagy - ngh)
    lagx = np.arange(ngw)
    lagx = np.where(lagx < ww, lagx, lagx - ngw)
    dyy, dxx = np.meshgrid(lagy.astype(np.float64), lagx.astype(np.float64), indexing="ij")
    sl_grid = slack_vec(shape, r, dxx, dyy)
    adm = np.isfinite(sl_grid) & (sl_grid >= 0.0)
    for k in range(len(xs)):
        x1, y1 = int(xs[k]) + bx0, int(ys[k]) + by0
        sl = slack_vec(shape, r, xx - x1, yy - y1)
        S1 = Lc & np.isfinite(sl) & (sl >= 0.0)
        if int(S1.sum()) < (n - 1):
            continue
        if n == 2:
            yc, xc = np.nonzero(S1)
            return [(x1, y1), (int(xc[0]) + bx0, int(yc[0]) + by0)]
        F = np.fft.rfft2(S1.astype(np.float32), s=(ngh, ngw))
        ac = np.fft.irfft2(F * np.conj(F), s=(ngh, ngw))
        good = (ac > 0.5) & adm
        if not good.any():
            continue
        iy, ix = np.unravel_index(int(np.argmax(np.where(good, sl_grid, -np.inf))), good.shape)
        dy, dx = int(lagy[iy]), int(lagx[ix])
        #   ⚠ **回找必须用回绕后的位移**（`lagy/lagx`），不能用数组下标：首版写成 `r0 + iy - hh`
        #   ⇒ 拿一个**不存在的位移**去找 p3 ⇒ 回找到重叠的假见证（实测 303@51 给出距离 1px 的三点、
        #   最小余量 −97.5）。**证书判定本身不受影响**（它用 `sl_grid` 按回绕 lag 索引，已被 48 例
        #   暴力对拍验证），错的只是见证提取这条路。
        for r0 in range(hh):
            for c0 in np.nonzero(S1[r0])[0]:
                yy2, xx2 = r0 + dy, c0 + dx
                if 0 <= yy2 < hh and 0 <= xx2 < ww and S1[yy2, xx2]:
                    return [(x1, y1), (int(c0) + bx0, int(r0) + by0),
                            (int(xx2) + bx0, int(yy2) + by0)]
    return None


def check_witness(pts, cells, n, shape, r, zone_calib, size=SIZE):
    """**用判据面直接校验一组摆位**（正对照）：返回 (合法, 最小余量)。"""
    cells, n, shape, r, size = _validate_cert_inputs(cells, n, shape, r, zone_calib, size)
    if pts is None or isinstance(pts, (str, bytes)):
        raise ValueError(f"pts must be a sequence of {n} points")
    try:
        pts_raw = list(pts)
    except TypeError as exc:
        raise ValueError(f"pts must be iterable, got {type(pts).__name__}") from exc
    if len(pts_raw) != n:
        raise ValueError(f"pts length must match n={n}, got {len(pts_raw)}")
    pts_valid = []
    for idx, pt in enumerate(pts_raw):
        if pt is None or isinstance(pt, (str, bytes)):
            raise ValueError(f"pts[{idx}] must be a (x, y) pair")
        try:
            xy = list(pt)
        except TypeError as exc:
            raise ValueError(f"pts[{idx}] must be a (x, y) pair") from exc
        if len(xy) != 2:
            raise ValueError(f"pts[{idx}] must have length 2, got {len(xy)}")
        pts_valid.append((
            _require_finite_real(xy[0], f"pts[{idx}][0]"),
            _require_finite_real(xy[1], f"pts[{idx}][1]"),
        ))
    e = S.paint_extent(shape, r) + 2.0
    ok = True
    for x, y in pts_valid:
        if not (e <= x <= size - e and e <= y <= size - e):
            ok = False
        if S._cell_of_pt(int(x), int(y), size, zone_calib) not in cells:
            ok = False
    sl = math.inf
    for i in range(n):
        for j in range(i + 1, n):
            dx = pts_valid[i][0] - pts_valid[j][0]
            dy = pts_valid[i][1] - pts_valid[j][1]
            sl = min(sl, math.hypot(dx, dy) - S._need_between(shape, r, dx, dy))
    return ok, sl


def _load_base():
    from . import hexgen_align_probe as P, hexgen_c1_real as C
    fit, _ = P.blackbox_reader()
    return fit.get("zone"), {it["id"]: it for it in C.load_items(C.CORPUS) if it["truth"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=int, action="append", required=True)
    ap.add_argument("--r", type=int, action="append", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    zc, base = _load_base()
    res = {}
    for pid in a.prompt:
        t = base[pid]["truth"]
        cells, n, shape = tuple(sorted(t["cells"])), t["n"], t["shape"]
        for r in a.r:
            key = f"{pid}@{r}"
            got = certify_radius(cells, n, shape, r, zc, verbose=True)
            got.pop("per_p1", None)
            res[key] = dict(got, cells=list(cells), n=n, shape=shape)
            print(key, json.dumps(res[key], ensure_ascii=False), flush=True)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
