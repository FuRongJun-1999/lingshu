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
import warnings
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
        except Exception:
            # 只兜「文件级」失败（不可读 / 非 JSON）——整体保持构造器先验。
            return
        if not isinstance(data, dict):
            return  # A parseable non-object is still an invalid calibration file.
        for ch, st in data.items():
            if ch not in self._state or not isinstance(st, dict):
                continue
            # #601：逐通道防护。旧实现把整段循环包进同一个 try——某个通道的
            # 值不可解析（ValueError/TypeError）时循环当场中断：坏值之前的通道
            # 已落状态、**之后的健康通道被静默弃养**（保持先验），且随后任一次
            # _update 触发的全量 save() 会把这份残缺状态固化回盘（永久丢失）。
            # 与 _sanitize 的纪律同口径：单个通道有毒**只丢该通道**（保持构造器
            # 先验），不牵连其他通道。dict 推导在异常时整体失败、update 不会
            # 写入半套值，故毒通道不会留下「一半新一半旧」的混合状态。
            try:
                self._state[ch].update({k: float(st.get(k, self._state[ch][k])) for k in ("a", "b", "hits", "misses")})
            except (TypeError, ValueError, OverflowError):
                continue
        self._sanitize()

    def _sanitize(self) -> None:
        """把持久化带入的毒状态挡在查询面之外。

        非对象根按坏文件处理；正的有限 Beta 参数、有限正总量与非负有限计数
        是公开查询的前置条件。坏通道只复位自身，载入不回写文件。
        `_update` 只做加法、`reset` 复位到先验，所以 `a+b<=0` / 非有限**只可能
        来自被篡改或损坏的持久化文件**。修前这些值一路带到
        `credibility()`/`channel_state()`/`registry()` 才炸（`ZeroDivisionError`，
        或 NaN 可信度配 `status="ok"`）——炸点离写入点很远，且 `_load` 期的
        `except Exception: pass` 只兜 JSON 解析、不兜语义。
        此处 fail-closed：单个通道有毒就**只丢该通道**（回落先验），不牵连其他通道，
        也不静默修改调用方传入的文件。
        """
        for ch, st in self._state.items():
            a, b = st.get("a"), st.get("b")
            values = (a, b, st.get("hits"), st.get("misses"))
            # Beta parameters must be positive; counts must be nonnegative.
            # Validate all four fields before public registry() converts counts to int.
            valid = all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)
            if not valid or a <= 0 or b <= 0 or not math.isfinite(a + b) \
                    or st["hits"] < 0 or st["misses"] < 0:
                st["a"], st["b"] = PRIOR_A, PRIOR_B
                st["hits"], st["misses"] = 0.0, 0.0

    def save(self) -> None:
        if not self.persist_path:
            return
        try:
            os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
            # #651：落盘前过滤非注册键——_load 重启后只认 DEFAULT_CHANNELS
            #（构造器先建的就是这六个键），写下去也只写这批，盘面与重载面
            # 严格一致。_state 里混入的其它键（直改内部字典 / 旧版本残留）
            # 不落盘：盘干净是第一道防线，_load 的 allowlist 是第二道。
            payload = {ch: st for ch, st in self._state.items() if ch in DEFAULT_CHANNELS}
            with open(self.persist_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=1)
        except Exception:
            pass

    # ---- 更新 ----

    def _registered_channels(self) -> set:
        """合法通道集合（#651 写入闸口径）：DEFAULT_CHANNELS ∪ 注册表既有键。

        与 ``_load`` 的 allowlist（``ch not in self._state: continue``）同口径：
        构造期 ``_state`` 的键就是 DEFAULT_CHANNELS，故「已在 _state」与「在
        DEFAULT_CHANNELS」在正常运行下重合——即**重启后真正能活下来的通道名**。
        修前写入侧（_update 的 setdefault / reset 的直接赋值）对通道名零校验，
        任意名静默建档：known=True、进 registry()，reset 带 persist 还会把自造键
        固化到盘上——与同文件 _load 的过滤纪律相反（#651）。
        """
        return set(DEFAULT_CHANNELS) | set(self._state)

    def record_hit(self, channel: str, conf: float = 1.0, strong: bool = False) -> Dict:
        """验证命中：a += n_effective（伪样本量加权）。strong=True 用强验证 α。

        通道名不在合法集合时按 #651 写入闸处理：warn + 跳过本次记账（见
        ``_update`` docstring）。
        """
        return self._update(channel, conf, strong, hit=True)

    def record_miss(self, channel: str, conf: float = 1.0, strong: bool = False) -> Dict:
        """验证未命中：b += n_effective。

        通道名不在合法集合时按 #651 写入闸处理：warn + 跳过本次记账（见
        ``_update`` docstring）。
        """
        return self._update(channel, conf, strong, hit=False)

    def _update(self, channel: str, conf: float, strong: bool, hit: bool) -> Dict:
        """记账一次验证（hit→a / miss→b），带写入侧通道名闸（#651）。

        未知通道（不在 DEFAULT_CHANNELS ∪ 注册表既有键）：**warn + 跳过本次
        记账**——不建档、不落盘，返回该名的中性先验快照（known=False，与查询
        侧 #539 的「未知通道可区分」口径衔接）。选 warn 而非 raise：本方法是
        数据驱动路径（anchored_verification / anchor_verify 把事件自带的通道名
        透传进来），一条坏通道名的一次 hit 不应打断整条验证更新流；但完全
        静默又等于把拼错/自造的通道当真通道统计（known=True 污染本进程的
        B(t) 端口快照），故至少要有声响。
        """
        # #651 写入闸：未知通道 warn + 跳过本次记账（口径理由见上）。
        if channel not in self._registered_channels():
            warnings.warn(
                "通道 %r 未注册，本次验证记账已跳过（合法通道：%s）——"
                "拼错/自造的通道名不再静默建档（#651）"
                % (channel, sorted(self._registered_channels())),
                stacklevel=3,  # 指向 record_hit/record_miss 的真正调用方
            )
            return self.channel_state(channel)
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
        # #119 ②：折扣 Beta（遗忘）——α 至此才真正是「保持率」，而不只是算权重的参数。
        #   旧实现只做 `st["a"] += n_eff` 累加、**从不折旧** ⇒ 历史越长越难被发现失灵：
        #   issue 实测 50 次强命中后要连续 117 次强未命中才降权，且该数随命中次数 N
        #   **线性增长**，文件头承诺的「降权 / 检修」于是只在最不需要的新通道上灵敏。
        #   本件按 issue 修复建议 2 加入折扣：更新前先 `a ← PRIOR_A + α(a - PRIOR_A)`
        #   （b 同理）⇒ 后验有上界 `a* = PRIOR_A + n_eff/(1-α)`，降权所需未命中次数
        #   与 N 解耦（即 issue 守卫判据「不超过某个与 N 无关的上界 K」）。
        #   边界（如实标注）：折扣**只施加于本次被更新的通道**（issue 原文「每次更新
        #   前先执行」指该通道），不遍历全部通道；`hits`/`misses` 仍为**累计计数**
        #   ——它们是观测面、不是后验伪样本量，不参与折扣（#119 往返判据要求这两个
        #   字段 save/load 后逐字段相等）。
        st["a"] = PRIOR_A + alpha * (st["a"] - PRIOR_A)
        st["b"] = PRIOR_B + alpha * (st["b"] - PRIOR_B)
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
        """该通道的可信度（后验均值 a/(a+b)）。

        分母为零或非有限时回落到先验 0.5——`a+b<=0` / `NaN` 只可能来自被篡改或
        损坏的持久化文件（`_update` 只做加法、`reset` 复位到先验，正常运行到不了
        这个状态）。修前此处直接除，`credibility()`/`channel_state()`/`registry()`
        会抛 `ZeroDivisionError`，而毒值在 `_load` 期已静默入库、炸点离写入点很远。
        """
        st = self._state.get(channel, {"a": PRIOR_A, "b": PRIOR_B})
        denom = st["a"] + st["b"]
        if not math.isfinite(denom) or denom <= 0:
            return PRIOR_A / (PRIOR_A + PRIOR_B)
        return st["a"] / denom

    def channel_state(self, channel: str) -> Dict:
        raw = self._state.get(channel, {"a": PRIOR_A, "b": PRIOR_B, "hits": 0.0, "misses": 0.0})
        denom = raw["a"] + raw["b"]
        if not math.isfinite(denom) or denom <= 0:
            # 分母不可用：回落先验，且**快照里的 a/b 一并报先验**——否则 NaN 会从
            # 本字典漏给调用方（调用方自己再除一次仍得 NaN）。
            a, b = PRIOR_A, PRIOR_B
        else:
            a, b = raw["a"], raw["b"]
        cred = a / (a + b)
        status = "ok"
        if cred < MAINTENANCE_THRESHOLD:
            status = "maintenance"
        elif cred < DOWNWEIGHT_THRESHOLD:
            status = "downweighted"
        return {
            "channel": channel, "credibility": round(cred, 4),
            "a": round(a, 2), "b": round(b, 2),
            "hits": int(raw["hits"]), "misses": int(raw["misses"]),
            # `known`：该通道是否真的在注册表里。未知通道名（拼错/大小写/空串）
            # 修前与「健康中性态」逐位不可区分——credibility 恒 0.5、status 恒 ok。
            "known": channel in self._state,
            "status": status,
        }

    def registry(self) -> Dict:
        """全通道注册表快照（B(t) 端口状态的一部分）。"""
        return {ch: self.channel_state(ch) for ch in self._state}

    # ---- 重置 ----

    def reset(self, channel: Optional[str] = None) -> None:
        """复位通道到先验，带写入侧通道名闸（#651）。

        · ``channel=None``：全量复位（既有行为不变）；
        · ``channel`` 非法（非 None 且不在 DEFAULT_CHANNELS ∪ 注册表既有键）：
          **raise ValueError**。reset 是显式管理 API，调用方传错通道名属编程
          错误——静默为任意名建档（known=True 进 registry、persist 落盘）会把
          错误藏进状态里，与 normalize_part / load_shapes 的「非法即 raise」
          纪律同口径（fail-closed）。空串同样 raise：修前 ``if channel:`` 把
          ``""`` 当无参走全量复位，属于同一个零校验缺口。
        """
        if channel is not None:
            if channel not in self._registered_channels():
                raise ValueError(
                    "未知通道 %r：reset 只接受已注册通道（%s）或 None（全量复位）"
                    % (channel, sorted(self._registered_channels()))
                )
            self._state[channel] = {"a": PRIOR_A, "b": PRIOR_B, "hits": 0.0, "misses": 0.0}
        else:
            for ch in self._state:
                self._state[ch] = {"a": PRIOR_A, "b": PRIOR_B, "hits": 0.0, "misses": 0.0}
        if self.persist_path:
            self.save()
