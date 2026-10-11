# -*- coding: utf-8 -*-
"""channel_credibility · 通道可信度注册表（智能论 v3.4 · 2.9.1a）
============================================================================
v3.4 新增机制：置信度 ≠ 可信度。
  - 置信度（confidence）= 单次判断的内部一致性（"这次像不像"）
  - 可信度（credibility）= 通道历史验证命中率（"这个通道可不可信"）

贝叶斯 Beta 后验更新（伪样本量加权，GLM 评审 P3 修正）：
    c_i ~ Beta( a_i0 + n_effective_hit, b_i0 + n_effective_miss )
    n_effective = conf * n_base          # 连续置信度作为伪样本量调节
    credibility_i = a_i / (a_i + b_i)    # 后验均值

锚定分级（v3.4 2.9.1b）：弱验证（通道互裁 α=慢）/ 强验证（行动裁决 α=快）
  - 通过 alpha 参数体现：弱验证 alpha 高（慢更新，证据权重低）
  - 强验证 alpha 低（快更新，证据权重高）

偏离声明（#119④ · 设计者裁决 C-12 · 2026-10-09）——**有意现状，非缺陷**：
  本面与理想设计存在三处**已知偏离**，经设计者裁定保留（不引入遗忘，改文档如实声明）：
  ① **无时间动力学**：`a`/`b` 只累加、**无衰减项**（无 `exp(-γ·t)`）⇒ **无遗忘机制**。
     理论 3.2.2 的「统一时间动力学（防高置信过期事实）」未延伸到本面
     （stable 租约/confidence 面另按 DEV-005 声明）。上文"快更新"只指**单次证据
     权重高**（α 低 ⇒ `n_eff` 大），**不含时间维度**。
  ② **O(N) 惰性**：后验惯性随历史**线性增长**——N 次观测后单次反证只动后验
     `n_eff/(a+b+n_eff)`（N≈200 时约 0.5%）；要翻转需与历史**同量级**（≈N）的
     反证 ⇒ 对**非平稳环境**响应惰性 O(N)。"快"只相对**先验**成立，不相对已积历史成立。
  ③ **`e_weight` 趋零**：本面后验饱和 ⇒ 信任 T 趋于稳定 ⇒ 信任二阶变化率
     （`core.SelfModel._compute_e_weight`，理论 1.6.5 的 `E_weight`）长期趋零，
     即该驱动力机制在长期读数上失去输入。
  边界（P1-003）：理论 3.2「不可遗忘」的对象是**记录**而非状态，本条不越界。
  **重新评估条件**（出现任一即重开 C-12）：① 实测出现「过期高置信压舱」导致的误判；
  ② `e_weight` 由「仅记录」转为参与决策（v1.3 开放）；③ 出现「变化环境须快速翻转」的
  实测需求——届时按 C-12 备选 (b) 引入**状态**遗忘（滑窗/指数衰减），记录仍保留。

通道集合（对应感知机端口 + 预测/行动）：
  visual（see） / tactile（control） / audio / action（输出端口） /
  prediction（第六感） / search（条件路由）

纯标准库 · 零外部依赖（D-005）
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Dict, List, Optional

# 默认通道（对应 v3.4 三端口架构）
DEFAULT_CHANNELS = ["visual", "tactile", "audio", "action", "prediction", "search"]

# Beta 先验参数（弱先验：a=b=2，Beta(2,2) 中心 0.5）
PRIOR_A = 2.0
PRIOR_B = 2.0

# 伪样本量基数（每单位置信度折算的样本数）
N_BASE = 10.0

# 锚定分级：弱验证慢更新（α高） / 强验证快更新（α低）
ALPHA_WEAK = 0.95   # 通道互裁（感知通道间）——慢更新
ALPHA_STRONG = 0.7  # 行动/世界裁决（输出端口）——快更新

# 降权/检修阈值（工程默认值，待实测校准——DEV-004）
DOWNWEIGHT_THRESHOLD = 0.3   # 低于此 → 降权（保留但权重低）
MAINTENANCE_THRESHOLD = 0.15 # 低于此 → 标记需检修（check 探测）


class ChannelCredibilityRegistry:
    """通道可信度注册表（v3.4 2.9.1a）。

    每个通道维护 Beta 后验 (a, b)：
      - record_hit(channel, conf, strong)：验证命中 → a += n_effective
      - record_miss(channel, conf, strong)：验证未命中 → b += n_effective
      - credibility(channel)：后验均值 a/(a+b)
    持久化：JSON 文件（可选的 persist_path），热数据可选。
    """

    def __init__(self, persist_path: Optional[str] = None):
        self.persist_path = persist_path
        self._state: Dict[str, Dict[str, float]] = {}
        for ch in DEFAULT_CHANNELS:
            self._state[ch] = {"a": PRIOR_A, "b": PRIOR_B, "hits": 0.0, "misses": 0.0}
        self._load()

    # ---- 持久化 ----

    def _load(self) -> None:
        if not self.persist_path or not os.path.exists(self.persist_path):
            return
        try:
            with open(self.persist_path, encoding="utf-8") as f:
                data = json.load(f)
            for ch, st in data.items():
                if ch in self._state and isinstance(st, dict):
                    self._state[ch].update({k: float(st.get(k, self._state[ch][k])) for k in ("a", "b", "hits", "misses")})
        except Exception:
            pass

    def save(self) -> None:
        if not self.persist_path:
            return
        try:
            os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
            with open(self.persist_path, "w", encoding="utf-8") as f:
                json.dump(self._state, f, ensure_ascii=False, indent=1)
        except Exception:
            pass

    # ---- 更新 ----

    def record_hit(self, channel: str, conf: float = 1.0, strong: bool = False) -> Dict:
        """验证命中：a += n_effective（伪样本量加权）。strong=True 用强验证 α。"""
        return self._update(channel, conf, strong, hit=True)

    def record_miss(self, channel: str, conf: float = 1.0, strong: bool = False) -> Dict:
        """验证未命中：b += n_effective。"""
        return self._update(channel, conf, strong, hit=False)

    def _update(self, channel: str, conf: float, strong: bool, hit: bool) -> Dict:
        # #402：NaN/±inf 不得被钳制表达式放行——实测 Python 的 min(1.0, nan)=1.0
        #   会把 NaN 置信度抬成 1.0，注入满额伪样本量（n_eff）。非有限值一律
        #   fail-closed 归零（零伪样本量，后验不动）。
        conf = float(conf)
        if not math.isfinite(conf):
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        alpha = ALPHA_STRONG if strong else ALPHA_WEAK
        # 伪样本量：置信度 × 基数 × 锚定权重
        # 强验证（α低=快更新）：证据权重高；弱验证（α高=慢更新）：证据权重低
        # anchor_w = 证据强度 = 1 - α（强验证 α=0.7 → w=0.3；弱验证 α=0.95 → w=0.05）
        evidence = 1.0 - alpha
        anchor_w = evidence * 20.0  # 放大到可比尺度
        anchor_w = min(anchor_w, 6.0)
        n_eff = conf * N_BASE * anchor_w
        st = self._state.setdefault(channel, {"a": PRIOR_A, "b": PRIOR_B, "hits": 0.0, "misses": 0.0})
        if hit:
            st["a"] += n_eff
            st["hits"] += 1.0
        else:
            st["b"] += n_eff
            st["misses"] += 1.0
        if self.persist_path:
            self.save()
        return self.channel_state(channel)

    # ---- 查询 ----

    def credibility(self, channel: str) -> float:
        st = self._state.get(channel, {"a": PRIOR_A, "b": PRIOR_B})
        return st["a"] / (st["a"] + st["b"])

    def channel_state(self, channel: str) -> Dict:
        st = self._state.get(channel, {"a": PRIOR_A, "b": PRIOR_B, "hits": 0.0, "misses": 0.0})
        cred = st["a"] / (st["a"] + st["b"])
        status = "ok"
        if cred < MAINTENANCE_THRESHOLD:
            status = "maintenance"
        elif cred < DOWNWEIGHT_THRESHOLD:
            status = "downweighted"
        return {
            "channel": channel, "credibility": round(cred, 4),
            "a": round(st["a"], 2), "b": round(st["b"], 2),
            "hits": int(st["hits"]), "misses": int(st["misses"]),
            "status": status,
        }

    def registry(self) -> Dict:
        """全通道注册表快照（B(t) 端口状态的一部分）。"""
        return {ch: self.channel_state(ch) for ch in self._state}

    # ---- 重置 ----

    def reset(self, channel: Optional[str] = None) -> None:
        if channel:
            self._state[channel] = {"a": PRIOR_A, "b": PRIOR_B, "hits": 0.0, "misses": 0.0}
        else:
            for ch in self._state:
                self._state[ch] = {"a": PRIOR_A, "b": PRIOR_B, "hits": 0.0, "misses": 0.0}
        if self.persist_path:
            self.save()
