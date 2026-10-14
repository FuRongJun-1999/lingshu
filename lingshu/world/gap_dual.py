# -*- coding: utf-8 -*-
"""gap_dual · 信息差符号区分 D_task / D_meta（智能论 v3.4 · 2.7.0）
============================================================================
v3.4 符号区分：
  - D_task：图上操作性距离——当前任务可缩小的信息差（可增可减，学习前提）
  - D_meta：未被建模的现实总量——d(D_meta)/dt ≥ 0（结构性论证）

映射声明（DEV-002）：v3.3 D_norm（四维度综合度量）对应 D_task——
系统当前可计算的信息差度量；D_meta 不纳入 D_norm 四维度公式，
仅作结构性边界（认知永远有活干的供给）。

D_task 操作性定义（v1.2 图上定义）：
    D_task(t) = min_{ p ∈ G, p 可用 } dist( C_task(t), CondSet(p) )
  - C_task(t)：任务条件实例
  - CondSet(p)：知识路径 p 的条件满足集
  - min 定义在空集上 → D_task = ∞ → BLINDSPOT 合法停止

纯标准库 · 零外部依赖（D-005）
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional

# 统一时间核（钉死批条款3：衰减核形状唯一，实现收口 aeis/time_core.py）
try:
    from .time_core import cred_blend
except ImportError:  # 直跑 fallback（裸名互导）
    from .time_core import cred_blend


class GapDual:
    """信息差双层度量（D_task / D_meta 分离）。

    实现：
      - D_task：基于知识图路径可达性的图上距离（0=任务可立即执行，∞=BLINDSPOT）
      - D_meta：结构性增长量（无干预时 d/dt≥0；本实现为观测统计，非真实测量）
    输入：
      - graph_paths：{path_id: {"available": bool, "dist": float, "cond_set": set}}
        由条件路由图（graph_retrieve）提供候选路径
      - C_task：当前任务条件实例
    输出：
      - D_task：min 可达路径距离（空集 → inf → BLINDSPOT）
      - D_meta_est：未建模估计（路径不可达/缺失条件的累积）
    """

    def __init__(self, decay: float = 0.1):
        self.decay = decay          # λ_gap 衰减参数
        self._meta_history: List[float] = []
        self._last_meta = 0.0
        self._prev_meta = 0.0       # #124：上一读数（trend 按方向判，非固定阈值）

    def compute_d_task(self, graph_paths: Optional[Dict] = None,
                       task_conditions: Optional[set] = None) -> Dict:
        """计算 D_task（图上操作性距离）。

        graph_paths: {path_id: {"available": bool, "dist": float, "cond_set": set}}
        task_conditions: 任务条件实例（集合）
        返回：{d_task, path_count, usable_paths, blindspot, cond_gap}
        """
        if not graph_paths:
            return {"d_task": float("inf"), "path_count": 0,
                    "usable_paths": 0, "blindspot": True,
                    "note": "无可用知识路径 → D_task=∞ → BLINDSPOT 合法停止"}

        usable = [(pid, p) for pid, p in graph_paths.items()
                  if p.get("available")]
        if not usable:
            return {"d_task": float("inf"), "path_count": len(graph_paths),
                    "usable_paths": 0, "blindspot": True,
                    "note": "全部路径不可达 → BLINDSPOT"}

        # #518：dist **缺失 / NaN / None** 的可用路径不参与选路——其距离无定义，
        # 混进 `min` 会引出三种病（改前实测）：① cond_gap=0 的无 dist 路径按
        # 字典序压过有限距离路径 ⇒ 明明存在有限 D_task 却报 BLINDSPOT；
        # ② NaN 参与比较恒 False ⇒ 结果随 dict 插入顺序漂移；③ None 直接
        # TypeError。选路面＝可用且 dist 为有限数的路径；该面为空 ⇒ 按 :16/:44
        # 判 BLINDSPOT（与 #124「唯一可用路径缺 dist ⇒ ∞ ⇒ BLINDSPOT」同一口径，
        # 从单路径推广到多路径）。
        def _finite_dist(p: Dict):
            try:
                d = float(p.get("dist"))
            except (TypeError, ValueError):
                return None
            return d if math.isfinite(d) else None

        valid = [(pid, p) for pid, p in usable if _finite_dist(p) is not None]
        if not valid:
            return {"d_task": float("inf"), "path_count": len(graph_paths),
                    "usable_paths": len(usable), "blindspot": True,
                    "cond_gap": None, "best_path": "",
                    "note": "可用路径的 dist 全部缺失/非有限 → D_task=∞ → BLINDSPOT（#518）"}

        # #124：D_task = min_{p 可用} dist(C_task, CondSet(p))（模块头 :12-13 的
        # v1.2 图上定义）——「条件缺口」是这条距离的一部分：缺任务的条件的路径
        # 更远。选路先比缺口（小者优先），缺口相同再比 dist。旧实现只按 dist
        # 选路，`cond_gap` 在选完才算且不回写 ⇒ 注释 :73 承诺的「条件缺口惩罚」
        # 从未生效（复现：p_far 完整覆盖 {a,b} 却输给缺 b 的 p_close）。
        # `best_path` 取 path_id——docstring 规定 id 在**键**上，旧实现读
        # `best["id"]`（值里没有）故任何合法输入下恒为空串。
        # 判据来源：模块头 :12-13（v1.2 图上定义）+ triage #124 §问题 2/3。
        def _cond_gap(p: Dict) -> int:
            if task_conditions:
                return len(set(task_conditions) - set(p.get("cond_set") or ()))
            return 0

        best_id, best = min(
            valid, key=lambda kv: (_cond_gap(kv[1]),
                                   kv[1].get("dist", float("inf"))))
        d_task = float(best.get("dist", float("inf")))
        cond_gap = _cond_gap(best)
        # #124：可用路径缺 dist ⇒ d_task=∞，按 :16/:44 规定即为 BLINDSPOT
        # （旧实现恒返回 blindspot=False，下游按 blindspot 分支会误判为可执行）。
        blindspot = not math.isfinite(d_task)

        return {"d_task": round(d_task, 4), "path_count": len(graph_paths),
                "usable_paths": len(usable), "blindspot": blindspot,
                "cond_gap": cond_gap, "best_path": best_id,
                "note": "D_task 可增可减——学习后路径可用性提升则下降"}

    def update_meta(self, d_task: float, blindspot: bool,
                    new_conditions_seen: int = 0) -> float:
        """D_meta 结构性更新：d(D_meta)/dt ≥ 0（无干预时）。

        观测代理：盲区事件（BLINDSPOT）+ 新条件暴露 + 时间衰减。
        返回当前 D_meta 估计（结构性边界，非数值计算进 D_norm）。

        #124：D_meta 是「未被建模的现实总量」，契约方向是**单调不减**（模块头
        :6）。旧实现把 `cred_blend` 当更新式直接写回：无盲区、无新条件（即
        「无干预」，growth=0）时估计值按 retain=1-decay 几何衰减，10 步后只剩
        34.87% —— 方向与声明正好相反。现取**单调包络**：平滑值只作候选，
        估计值取历史最大 ⇒ 无事件时 d/dt=0（持平，不再下降），有事件时 d/dt>0。
        `decay` 仍参与动力学：增量须超过 `last·decay` 才能抬高估计值。
        判据来源：模块头 :6 与 :91 docstring 的结构性论证（d(D_meta)/dt ≥ 0）。
        削掉的判别力（如实声明）：`decay` 不再能拉低已有估计值——「不做功则
        衰减」的旧行为被契约方向取代，指数衰减在读数上不可再观测。
        """
        # 结构性增长：盲区/新条件暴露 → 增长（时间衰减只决定吸收速度）
        growth = 0.0
        if blindspot:
            growth += 1.0                     # 盲区暴露 → 未建模总量增加
        growth += new_conditions_seen * 0.5   # 新条件暴露
        self._prev_meta = self._last_meta     # trend 按方向判（#124）
        # 单调包络：平滑候选与历史最大取大 ⇒ 无事件不下降
        self._last_meta = max(self._last_meta,
                              cred_blend(self._last_meta, growth,
                                         retain=1.0 - self.decay))
        self._meta_history.append(self._last_meta)
        if len(self._meta_history) > 1000:
            self._meta_history = self._meta_history[-1000:]
        return self._last_meta

    def state(self) -> Dict:
        """当前状态快照（D_task 计算 + D_meta 估计）。"""
        # #124：trend 按**方向**判（与上一读数比较），不再用固定阈值 0.1——
        # 旧判据在读数下降时仍报 "growing"（0.9→0.81 也报 growing）。
        if self._last_meta > self._prev_meta:
            trend = "growing"
        elif self._last_meta < self._prev_meta:
            trend = "declining"
        else:
            trend = "stable"
        return {
            "d_meta_est": round(self._last_meta, 4),
            "d_meta_trend": trend,
            "meta_samples": len(self._meta_history),
            "note": "D_meta 为结构性边界估计（类比层），不纳入 D_norm 数值计算",
        }
