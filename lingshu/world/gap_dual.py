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

        usable = [p for p in graph_paths.values() if p.get("available")]
        if not usable:
            return {"d_task": float("inf"), "path_count": len(graph_paths),
                    "usable_paths": 0, "blindspot": True,
                    "note": "全部路径不可达 → BLINDSPOT"}

        # D_task = min 可用路径的距离（条件缺口惩罚：任务条件未覆盖则 dist 加权）
        best = min(usable, key=lambda p: p.get("dist", float("inf")))
        d_task = float(best.get("dist", float("inf")))

        # 条件缺口：任务条件 vs 最佳路径条件集
        cond_gap = 0
        if task_conditions and best.get("cond_set"):
            missing = task_conditions - set(best["cond_set"])
            cond_gap = len(missing)
        elif task_conditions:
            cond_gap = len(task_conditions)

        return {"d_task": round(d_task, 4), "path_count": len(graph_paths),
                "usable_paths": len(usable), "blindspot": False,
                "cond_gap": cond_gap, "best_path": best.get("id", ""),
                "note": "D_task 可增可减——学习后路径可用性提升则下降"}

    def update_meta(self, d_task: float, blindspot: bool,
                    new_conditions_seen: int = 0) -> float:
        """D_meta 结构性更新：d(D_meta)/dt ≥ 0（无干预时）。

        观测代理：盲区事件（BLINDSPOT）+ 新条件暴露 + 时间衰减。
        返回当前 D_meta 估计（结构性边界，非数值计算进 D_norm）。
        """
        # 结构性增长：盲区/新条件暴露 → 增长；时间衰减 → 缓慢下降（需做功维持）
        growth = 0.0
        if blindspot:
            growth += 1.0                     # 盲区暴露 → 未建模总量增加
        growth += new_conditions_seen * 0.5   # 新条件暴露
        # 衰减（不做功则增长；做了功才可维持/下降）——统一时间核指数平滑
        self._last_meta = max(0.0, cred_blend(self._last_meta, growth,
                                              retain=1.0 - self.decay))
        self._meta_history.append(self._last_meta)
        if len(self._meta_history) > 1000:
            self._meta_history = self._meta_history[-1000:]
        return self._last_meta

    def state(self) -> Dict:
        """当前状态快照（D_task 计算 + D_meta 估计）。"""
        return {
            "d_meta_est": round(self._last_meta, 4),
            "d_meta_trend": "growing" if self._last_meta > 0.1 else "stable",
            "meta_samples": len(self._meta_history),
            "note": "D_meta 为结构性边界估计（类比层），不纳入 D_norm 数值计算",
        }
