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
    PYTHONPATH=. python experiments/hexgen_pack_certify.py --prompt 303 --r 47 --r 48
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from . import hexgen_self_source as S          # noqa: E402

SIZE = S.SIZE
_MESH = {}


def _mesh(size=SIZE):
    if size not in _MESH:
        y, x = np.mgrid[0:size, 0:size]
        _MESH[size] = (x.astype(np.int32), y.astype(np.int32))
    return _MESH[size]


def _rank_ok_vec(zone_calib, cells, size=SIZE):
    """逐像素「判决格 ∈ 允许集合」的布尔掩膜（与 `S._cell_of_pt` 同口径：冻结切点，缺省精确三等分）。"""
    x, y = _mesh(size)
    if zone_calib:
        rc = list(zone_calib["row"]["cuts"])
        cc = list(zone_calib["col"]["cuts"])
    else:
        rc = [size / 3.0, 2 * size / 3.0]
        cc = [size / 3.0, 2 * size / 3.0]
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
    e = S.paint_extent(shape, r) + 2.0
    x, y = _mesh(size)
    m = (x >= e) & (x <= size - e) & (y >= e) & (y <= size - e)
    return m & _rank_ok_vec(zone_calib, cells, size)


def support_vec(shape, r, dx, dy, dist):
    """`S.paint_support` 的向量化版本（逐点等价，守门比对）。"""
    ax = np.abs(dx) / dist
    ay = np.abs(dy) / dist
    if shape == "square":
        return r * (ax + ay) + 1.0
    if shape == "rectangle":
        return 1.35 * r * ax + 0.62 * r * ay + 1.0
    if shape == "diamond":
        return 1.15 * r * np.maximum(ax, ay) + 1.0
    if shape == "circle":
        return np.zeros_like(dist) + (r + 1.0)
    return np.full_like(dist, S.paint_extent(shape, r))


def slack_vec(shape, r, dx, dy,
              sep=S.SEP_GAP, pad=2.0):
    """`|δ| − need(δ)` 的向量化版本；`δ = 0` 处置 `-inf`（同一位置不算一对）。"""
    dist = np.hypot(dx, dy)
    out = np.full(dist.shape, -np.inf, dtype=np.float64)
    nz = dist > 0
    d = dist[nz]
    need = (support_vec(shape, r, dx[nz], dy[nz], d) * 2.0 + sep + pad)
    out[nz] = d - need
    return out


def _cert_p1_core(x1, y1, L, n, shape, r, size):
    """单个 p1 的最优余量（与 `certify_radius` **共用**同一实现 ⇒ 抽样与全量不可漂移）。"""
    ys, xs = np.nonzero(L)
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


def certify_radius(cells, n, shape, r, zone_calib, size=SIZE, verbose=False, max_p1=None, mask=None):
    """**穷举证书**：返回 dict(`feasible`, `max_slack`, `best_p1`, `n_p1`, `per_p1`)。

    ⚠ **`max_slack` 的语义**（R313 守门对拍时澄清）：它是「**配对项** slack 的最大值」——固定 p1 后
    只要求 `p2/p3` 与 p1 **相容**（slack ≥ 0），报告的是 `(p2,p3)` 那一对的余量，**不要求**它与 p1 的
    余量也大。因此 `max_slack ≥ 三点的「最大最小余量」`（是它的上界量），**判定只看 `feasible`**；
    `-inf` 表示「每个 p1 都找不到两个与自己相容的位置」＝更强的结构性结论。
    `n ∈ {2, 3}`（同格 k 件；R313 只服务残余的 k=3，k=2 走直判路径）。
    `mask` 只供**守门**用：用小合成合法掩膜把本机制与暴力穷举对拍。"""
    assert n in (2, 3), n
    L = legal_mask(cells, shape, r, zone_calib, size) if mask is None else np.array(mask, bool)
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
    L = legal_mask(cells, shape, r, zone_calib, size) if mask is None else np.array(mask, bool)
    return _cert_p1_core(x1, y1, L, n, shape, r, size)


def witness_at(cells, n, shape, r, zone_calib, size=SIZE, mask=None):
    """在半径 r 上**取出一组可行摆位**（证书内层：p1 遍历 + 差集找 δ + 回找 p2）——返回 pts 或 None。

    ⚠ 只在证书判定**可行**时有意义；用途是「把最优摆位的结构看出来」与作为记录证据。"""
    L = legal_mask(cells, shape, r, zone_calib, size) if mask is None else np.array(mask, bool)
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
    e = S.paint_extent(shape, r) + 2.0
    ok = True
    for x, y in pts:
        if not (e <= x <= size - e and e <= y <= size - e):
            ok = False
        if S._cell_of_pt(int(x), int(y), size, zone_calib) not in cells:
            ok = False
    sl = math.inf
    for i in range(n):
        for j in range(i + 1, n):
            dx = pts[i][0] - pts[j][0]
            dy = pts[i][1] - pts[j][1]
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
