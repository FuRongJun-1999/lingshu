# -*- coding: utf-8 -*-
"""hex_ortho · 正交残差金字塔 + 信息差自适应深度(HEX-CNN-R3 显式化)
====================================================================================
来源(外部对照,两处独立来源同构):
  VCP-ToolBox ResidualPyramid: 语义方向逐层 Gram-Schmidt 剥离,残差能量占比
    低于 minEnergyRatio(0.1)即停 → 输出「语义能量谱 + 最终残差」。
  本仓《子部件提取_理论稿_v0.4》§〇.A.4/A.5 + §六:残差=信息差 → 自适应阈值 →
    死区停;深度上限不是超参,而是「可分性条件自然耗尽」。
两处说的是同一件事,但本仓此前只有理论、没有接口(recursive_search 的
min_share 声明了却从未生效)。本模块把它显式化为可调用、可审计的接口。

同构关系(白箱版,不引入黑箱 embedding):
  VCP   : 在「向量空间」对语义方向做 Gram-Schmidt 正交化(数值版,方向来自模型)。
  本模块: 在「特征(晶格)空间」对**核算子**做正交匹配剥离(白箱版,方向=手写核,
          每层的解释度/系数/残差/正交性证据全部留痕)。
  数学同一: 原子 a_c = P_c F₀(核 c 在原始特征上的卷积响应),
          残差 r_k = F₀ − Σ_j β_j a_j,β 由最小二乘重解 → r_k ⊥ span{a_j}。
          「正交」= 每个子部件的解释与其余子部件的解释不重叠(能量不重复计入)。

两组显式化增量(与对照分析 ACCEPT 项 #3 对应):
  ① 正交性约束: gram_schmidt_kernels 逐核算「新信息占比」,占比 < tol 的核判为
     重复解释(如 edge_{d+3} = −edge_d 同轴、coarse≈smooth 同向),从候选池剔除
     —— 同一份语义能量不被两遍计入;剥离后残差对**全部**已解释方向正交。
  ② 信息差自适应深度: 层数**由残差能量占比决定**(residual_share < dead_zone 即停),
     max_layers 降级为工程保险上界而非语义深度;停因显式可查(stop_reason)。

能量守恒(可验证): ‖F₀‖² = ‖r_k‖² + ‖Σβ_j a_j‖²,故 cumulative_share = 1 − residual_share² ≤ 1。

纯 numpy(复用 hex_conv,零 LLM,D-005)。
"""
from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ALGO = "hex_ortho-0.1"

from .hex_cnn import hex_conv

# 残差能量占比死区(信息差入死区即停)——VCP minEnergyRatio=0.1 的白箱同构物。
DEAD_ZONE = 0.10
# 核族正交去重阈值:新信息占比 < tol → 该核可由已保留核线性解释(重复解释)。
ORTHO_TOL = 0.20
# 层数工程保险上界(非语义深度;语义深度由死区决定,此值只防病态输入打爆算力)。
MAX_LAYERS = 8

# 默认核名表(与 extended_prior_family 的构造顺序一致;换核族时用 names 覆盖)。
KERNEL_NAMES: List[str] = (["edge_d%d" % d for d in range(6)]
                           + ["smooth", "coarse", "laplacian", "center_surround"]
                           + ["noise_k%d" % i for i in range(8)])


def kname(i: int, names: Optional[Sequence[str]] = None) -> str:
    nm = names if names else KERNEL_NAMES
    return nm[i] if i < len(nm) else f"k{i}"


# ==================== ① 正交性约束(核族去重) ====================

def gram_schmidt_kernels(kernels: Sequence[np.ndarray],
                         tol: float = ORTHO_TOL,
                         names: Optional[Sequence[str]] = None) -> List[Dict]:
    """核族 Gram-Schmidt:逐核算「新信息占比」→ 显式判定是否重复解释。

    new_info_ratio = ‖k − Σ_已保留 proj‖ / ‖k‖ ∈ [0,1]
      ≈1 → 该核带来全新解释方向(保留);
      ≈0 → 该核完全可由已保留核线性表达(重复解释,剔除)。
    返回每核报告(idx/name/norm/new_info_ratio/redundant/explained_by)。
    """
    kept: List[np.ndarray] = []          # 已保留的正交化向量
    kept_idx: List[int] = []
    report: List[Dict] = []
    for i, k in enumerate(kernels):
        v = np.asarray(k, dtype=np.float64).ravel().copy()
        n0 = float(np.linalg.norm(v))
        explainers: List[int] = []
        for j, u in zip(kept_idx, kept):
            if n0 < 1e-12:
                break
            proj = float(v @ u)
            if abs(proj) > 1e-9:
                explainers.append(j)
            v = v - proj * u
        n1 = float(np.linalg.norm(v))
        ratio = n1 / n0 if n0 > 1e-12 else 0.0
        redundant = ratio < tol
        if not redundant:
            kept.append(v / (n1 + 1e-12))
            kept_idx.append(i)
        report.append({"idx": i, "name": kname(i, names),
                       "norm": round(n0, 4), "new_info_ratio": round(ratio, 4),
                       "redundant": bool(redundant),
                       "explained_by": explainers if redundant else []})
    return report


def orthogonal_pool(kernels: Sequence[np.ndarray], tol: float = ORTHO_TOL,
                    names: Optional[Sequence[str]] = None
                    ) -> Tuple[List[int], List[Dict]]:
    """正交候选池:返回 (保留核下标, 全量报告)。剔除的核=重复解释,不参与金字塔。"""
    report = gram_schmidt_kernels(kernels, tol, names)
    keep = [r["idx"] for r in report if not r["redundant"]]
    return keep, report


# ==================== ② 正交残差金字塔(自适应深度) ====================

def residual_pyramid(lattice: np.ndarray, kernels: Sequence[np.ndarray],
                     dead_zone: float = DEAD_ZONE, max_layers: int = MAX_LAYERS,
                     tol: float = ORTHO_TOL, use_orthogonal_pool: bool = True,
                     names: Optional[Sequence[str]] = None) -> Dict:
    """正交残差金字塔:逐层选「解释度最高」的核,全体已解释方向最小二乘重解,
    残差能量占比入死区即停。

    第 k 层:原子 a_c = P_c F₀;在残差 r_{k-1} 上选 |⟨r,a_c⟩|/‖a_c‖ 最大的核;
      β = argmin‖F₀ − Σβ_j a_j‖²(对**全部**已选方向重解);
      r_k = F₀ − Σβ_j a_j,满足 ⟨r_k, a_j⟩ = 0 ∀j(正交:解释不重叠)。
    停止(自适应深度,顺序判定):
      card_exhausted       候选核耗尽
      orthogonal_exhausted 最优 cos² < tol(残差与所有核近乎正交=无处可解释)
      dead_zone            残差能量占比 < dead_zone(信息差入死区)
      max_layers           达工程保险上界
    返回 layers(逐层证据)/depth/stop_reason/energy_spectrum/正交审计。
    """
    import time
    t0 = time.time()
    F0 = np.asarray(lattice, dtype=np.float64)
    if F0.ndim == 4:                     # (B,r,c,C) → 取第 0 张(单实例金字塔)
        F0 = F0[0]
    flat0 = F0.ravel()
    norm0 = float(np.linalg.norm(flat0)) + 1e-12
    keep, ortho_report = (orthogonal_pool(kernels, tol, names)
                          if use_orthogonal_pool
                          else (list(range(len(kernels))), []))
    pool = list(keep)
    atoms: List[np.ndarray] = []         # 已选解释方向(基于 F₀ 的固定原子)
    sel_idx: List[int] = []
    layers: List[Dict] = []
    spectrum: List[float] = []
    r = flat0.copy()
    stop = "max_layers"
    while True:
        if not pool:
            stop = "card_exhausted"
            break
        if len(layers) >= max_layers:
            stop = "max_layers"
            break
        rn2 = float((r * r).sum())
        best: Optional[Dict] = None
        for ci in pool:
            a = hex_conv(F0.astype(np.float32), kernels[ci]).astype(np.float64)
            an2 = float((a * a).sum())
            if an2 < 1e-12:
                continue
            inner = abs(float((r * a.ravel()).sum()))
            gain = inner / np.sqrt(an2)              # 加入后残差下降量
            cos2 = (inner * inner) / (rn2 * an2 + 1e-12)   # 解释度 ∈[0,1]
            if best is None or gain > best["gain"]:
                best = {"idx": ci, "gain": gain, "cos2": cos2,
                        "a": a.ravel().copy()}
        if best is None:
            stop = "card_exhausted"
            break
        if best["cos2"] < tol:                       # 与所有核近乎正交 → 无处可解释
            stop = "orthogonal_exhausted"
            break
        atoms.append(best["a"])
        sel_idx.append(best["idx"])
        A = np.stack(atoms, axis=1)                  # (N,k)
        G = A.T @ A
        b = A.T @ flat0
        try:
            coef = np.linalg.solve(G + 1e-9 * np.eye(len(atoms)), b)
        except np.linalg.LinAlgError:                # 病态 → 最小二乘回退
            coef = np.linalg.lstsq(G, b, rcond=None)[0]
        r = flat0 - A @ coef
        residual_share = float(np.linalg.norm(r)) / norm0
        cumulative = 1.0 - residual_share ** 2        # 能量守恒:‖F₀‖²=‖r‖²+‖解释‖²
        prev_cum = layers[-1]["cumulative_share"] if layers else 0.0
        # 正交性证据:残差与全部已解释方向的最大残余重叠(理论值 0)
        ov = 0.0
        for a in atoms:
            na = float(np.linalg.norm(a)) + 1e-12
            ov = max(ov, abs(float((r * a).sum())) / (float(np.linalg.norm(r)) * na + 1e-12))
        layers.append({
            "layer": len(layers), "kernel_idx": best["idx"],
            "name": kname(best["idx"], names),
            "coef": round(float(coef[-1]), 5), "cos2": round(best["cos2"], 4),
            "explained_share": round(cumulative - prev_cum, 4),
            "cumulative_share": round(cumulative, 4),
            "residual_share": round(residual_share, 4),
            "orthogonality_gain": round(ov, 6)})
        spectrum.append(round(cumulative - prev_cum, 4))
        pool.remove(best["idx"])                      # 该方向已解释,不重复剥离
        if residual_share < dead_zone:                # 信息差入死区 → 停
            stop = "dead_zone"
            break
    return {"algo": ALGO, "layers": layers, "depth": len(layers),
            "stop_reason": stop, "energy_spectrum": spectrum,
            "residual_share": round(float(np.linalg.norm(r)) / norm0, 4),
            "cumulative_share": round(1.0 - (float(np.linalg.norm(r)) / norm0) ** 2, 4),
            "initial_energy": round(norm0, 4),
            "selected_kernels": sel_idx,
            "orthogonality": {"pool_size": len(keep),
                              "dropped": len(kernels) - len(keep),
                              "tol": tol, "report": ortho_report},
            "dead_zone": dead_zone, "max_layers": max_layers,
            "latency_ms": round((time.time() - t0) * 1000, 1)}


# ==================== 便捷入口 ====================

def pyramid_report(lattice: np.ndarray, kernels: Sequence[np.ndarray], **kw) -> Dict:
    """金字塔 + 一行式摘要(层数/停因/能量谱/去重数)——审计与日志用。"""
    r = residual_pyramid(lattice, kernels, **kw)
    return {"depth": r["depth"], "stop_reason": r["stop_reason"],
            "residual_share": r["residual_share"],
            "cumulative_share": r["cumulative_share"],
            "energy_spectrum": r["energy_spectrum"],
            "dropped_redundant": r["orthogonality"]["dropped"],
            "layers": [{"layer": l["layer"], "name": l["name"], "cos2": l["cos2"],
                        "residual_share": l["residual_share"]}
                       for l in r["layers"]]}


if __name__ == "__main__":
    import json
    from .hex_cnn import extended_prior_family, image_to_grid
    from .hex_hier import make_scene_dataset
    from .hex_train import normalize_lattices
    imgs, _ = make_scene_dataset(2, 48, seed=3)
    lat = normalize_lattices(np.stack([image_to_grid(im, 32)[0] / 255.0
                                       for im in imgs]))
    ks = extended_prior_family(12)
    print(json.dumps(pyramid_report(lat, ks), ensure_ascii=False, indent=1))
