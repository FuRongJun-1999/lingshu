# -*- coding: utf-8 -*-
"""wm_simloop · 世界模型推演循环（WM-SIMLOOP-REV1 · 世界模型阶段5 · 里程碑 M5.1）
============================================================================
核心（荣 2026-09-16 指令路线）：**世界状态 = 存算一体 IR 的推演执行**——
把「世界图 ↔ 认知图」之间的推演循环从隐式散布升为显式一等结构：

  装载环 load_priors()  认知图现在时节点 → 世界图实体先验（位置+类别）
  推演环 step()/run()   ①generate 生成先验 → ②settle 物理结算（SceneSimulator，
                        模型不可见内部）→ ③perceive 观察（可注入感知遮蔽）
                        → ④verify 命中率 → ⑤growth 拓扑生长/回退 → ⑥WAL 留痕
  回写环 flush_payloads()  WAL → 认知图 contextual 节点载荷（对齐 ARC AGI
                        游戏记忆先例格式；不旁路写真源，摄取走 mdcg 正规通道）

拓扑生长（Overmann 拓扑可塑性 · 最小可验证实例）：
  - 触发（死区门控，对齐 hex 反传语义）：实体 A 连续 K tick 内 anomaly ≥ deadzone
    （标定发现：D2 宽随机边界使直线/温和追逐不可辨识——线性外推已完美解释，
    无信息差不生长是机制的诚实行为；生长仅在窄边界预测持续失准时触发）
  - 生长（确定性外推，零 rng）：A 的位移射线收敛 → 假设实体 H（隐藏目标：
    感知遮蔽/盲区中的存在）+ seek(A→H) 关系边
  - 验证窗口 N tick：趋向占比（位移与 A→H 方向余弦 > 0.2 的占比）≥ 半窗
    → H 固化（置信升至 0.7 封顶）；A 持续背离 → REVERT（验证不成立不固化，
    anomaly 留痕不删）

WAL（append-only 单调）：{seq, ts, tick, kind ∈ episode|growth|confirm|revert|verify}。
回放一致性：顺序重放 episode 事件重建的世界图与当前 graph() 逐节点逐边一致
（WAL = 世界图演化的可逆投影；被 REVERT 的假设在快照消失但 revert 事件留痕）。

设计稿：docs/世界模型推演循环显式化_设计稿_v0.1.md
（含 Nature co-design 八特性 checklist 审计与 v0.1 诚实边界）

纯标准库 · 零外部依赖 · 零 LLM（D-005）· SimLoop 自身零 rng（确定性外推）
"""
from __future__ import annotations

import json
import math
import os
import time
import uuid
from typing import Dict, List, Optional, Tuple

try:
    from .world_model import UnifiedWorldModel, WMNode, WMEdge
except ImportError:
    from .world_model import UnifiedWorldModel, WMNode, WMEdge
try:
    from .scene_simulator import SceneSimulator
except ImportError:
    from .scene_simulator import SceneSimulator


# ==================== 确定性几何（拓扑生长外推 · 零 rng）====================

def _ray_intersect_2d(q1, u1, q2, u2) -> Optional[Tuple[float, float]]:
    """射线 q1+s·u1 与 q2+t·u2 的 2D 交点（平行/近平行返回 None）。

    判据：交点须落在**两条**射线的正向前方（s≥0 且 t≥0）。只校验 s 会接受
    「在射线 1 前方、却在射线 2 身后」的交点——那是两直线相交，不是两射线
    相交：对发散轨迹（背离型）会给出假目标点。
    """
    denom = u1[0] * (-u2[1]) - (-u2[0]) * u1[1]
    if abs(denom) < 1e-9:
        return None
    w = (q2[0] - q1[0], q2[1] - q1[1])
    s = (w[0] * (-u2[1]) - (-u2[0]) * w[1]) / denom
    t = (w[1] * u1[0] - w[0] * u1[1]) / denom
    if s < -1e-6 or t < -1e-6:         # 交点在任一条射线身后（背离运动方向）→ 无效
        return None
    x = q1[0] + s * u1[0]
    z = q1[1] + s * u1[1]
    return (x, z)


def estimate_hidden_target(traj: List[Tuple[float, float, float]],
                           size: int,
                           min_turn_cos: float = 0.966,
                           extension: float = 6.0
                           ) -> Optional[Tuple[Tuple[float, float, float], float]]:
    """从轨迹推断隐藏目标位置（确定性）：位移射线两两交汇（转角>15°）取质心；
    射线平行（直线追逐，目标距离不可观）→ 沿平均航向外推固定距离。

    返回 (位置, 适应度)；适应度 = 各位移与「位移起点→候选点」方向的平均余弦，
    ≤0.5 视为轨迹不由该点解释（不生长）。
    """
    if len(traj) < 3:
        return None
    rays: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for i in range(1, len(traj)):
        dx, dz = traj[i][0] - traj[i - 1][0], traj[i][2] - traj[i - 1][2]
        n = math.hypot(dx, dz)
        if n > 1e-6:
            rays.append(((traj[i - 1][0], traj[i - 1][2]), (dx / n, dz / n)))
    if len(rays) < 2:
        return None
    inter: List[Tuple[float, float]] = []
    for i in range(len(rays)):
        for j in range(i + 1, len(rays)):
            c = rays[i][1][0] * rays[j][1][0] + rays[i][1][1] * rays[j][1][1]
            if c < min_turn_cos:       # 转角 > ~15° 的射线对才有交点信息
                p = _ray_intersect_2d(rays[i][0], rays[i][1],
                                      rays[j][0], rays[j][1])
                if p is not None and 0.5 <= p[0] <= size - 0.5 \
                        and 0.5 <= p[1] <= size - 0.5:
                    inter.append(p)
    if inter:
        px = sum(p[0] for p in inter) / len(inter)
        pz = sum(p[1] for p in inter) / len(inter)
    else:
        ux = sum(r[1][0] for r in rays) / len(rays)
        uz = sum(r[1][1] for r in rays) / len(rays)
        n = math.hypot(ux, uz)
        if n < 1e-6:
            return None
        px = traj[-1][0] + ux / n * extension
        pz = traj[-1][2] + uz / n * extension
    # 适应度：位移是否持续指向候选点
    cos_sum, cnt = 0.0, 0
    for i in range(1, len(traj)):
        dx, dz = traj[i][0] - traj[i - 1][0], traj[i][2] - traj[i - 1][2]
        dl = math.hypot(dx, dz)
        tx, tz = px - traj[i - 1][0], pz - traj[i - 1][2]
        tl = math.hypot(tx, tz)
        if dl > 1e-6 and tl > 1e-6:
            cos_sum += (dx * tx + dz * tz) / (dl * tl)
            cnt += 1
    quality = round(cos_sum / cnt, 3) if cnt else 0.0
    return (round(px, 2), traj[-1][1], round(pz, 2)), quality


# ==================== 推演循环 ====================

class SimulationLoop:
    """世界模型推演循环：UnifiedWorldModel（世界图三端口）× SceneSimulator
    （物理侧）× WAL（单调留痕）× 拓扑生长（Overmann 拓扑可塑性）。

    小脑（确定性结算）+ 新皮层（认知图沉淀）之间的显式推演主循环。
    """

    def __init__(self, size: int = 24, seed: int = 42,
                 world_model: Optional[UnifiedWorldModel] = None,
                 scene: Optional[SceneSimulator] = None,
                 wal_path: Optional[str] = None,
                 mask_eids: Optional[set] = None,
                 growth_window: int = 6,      # W：轨迹窗口
                 anomaly_window: int = 4,     # K：anomaly 计数窗口
                 deadzone: int = 2,           # 死区：K tick 内 anomaly 阈值
                 verify_window: int = 8,      # 假设验证窗口 N
                 confirm_ratio: float = 0.5,  # 窗口内趋向占比 ≥ 此比例 → 固化
                 persistence_window: int = 80,   # 方向持续性窗口（整窗；见 _persistence）
                 persistence_min: float = 0.45,  # 随机游走 null 之上 → 有未解释结构
                 clock=None):
        self.wm = world_model or UnifiedWorldModel(
            size=size, seed=seed, world=scene)
        self.scene = scene or self.wm.world
        self.wal_path = wal_path
        self._mask = set(mask_eids or set())
        self.W = int(growth_window)
        self.K = int(anomaly_window)
        self.deadzone = int(deadzone)
        self.verify_window = int(verify_window)
        self.confirm_ratio = float(confirm_ratio)
        self.persistence_window = int(persistence_window)
        self.persistence_min = float(persistence_min)
        self._clock = clock or time.time
        # WAL（内存 + 可选落盘，append-only）
        self.wal: List[Dict] = []
        self._seq = 0
        self.ticks_stepped = 0             # 推演环推进的 tick 数（不含外部 perceive）
        self._alias: Dict[str, str] = {}   # WAL 稳定别名（SceneEntity 用 uuid4 随机
        #                                    id，别名保证同配置两次运行 WAL 逐字节
        #                                    一致——单调审计可比性的前提）
        self._hypotheses: Dict[str, Dict] = {}   # subject_eid → 假设状态
        self._growth_log: List[Dict] = []
        self._rolling_rates: List[float] = []
        self._flush_count = 0

    def _alias_of(self, eid: str) -> str:
        """eid → 稳定别名（首次出现顺序分配 e1,e2,…——确定性）。"""
        if eid not in self._alias:
            self._alias[eid] = f"e{len(self._alias) + 1}"
        return self._alias[eid]

    # ---------- WAL ----------

    def _wal(self, kind: str, payload: Dict) -> None:
        rec = {"seq": self._seq, "ts": round(self._clock(), 3),
               "tick": self.wm.tick, "kind": kind, "payload": payload}
        self._seq += 1
        self.wal.append(rec)
        if self.wal_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.wal_path)),
                        exist_ok=True)
            with open(self.wal_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---------- 感知（可注入遮蔽：模型看不到的实体=盲区）----------

    def _observe(self) -> List[Dict]:
        return [{"eid": eid, "category": e.category, "pos": tuple(e.pos)}
                for eid, e in self.scene.entities.items()
                if eid not in self._mask]

    # ---------- 推演环 ----------

    def step(self, n: int = 1, external=None) -> Dict:
        """推进 n tick：generate → settle → perceive → verify → growth → WAL。

        external(scene, tick)：物理世界的外部演化钩子（遮蔽实体的脚本化运动/
        外部事件），在物理结算前应用——世界怎么演化是世界的自由，模型只观察。
        """
        for _ in range(max(0, int(n))):
            self.ticks_stepped += 1
            self.wm.generate(horizon=1)               # ① 生成先验
            if external is not None:
                external(self.scene, self.wm.tick + 1)
            self.scene.step(n=1)                      # ② 物理结算（模型不可见内部）
            self.wm.perceive(observations=self._observe())  # ③ 观察（遮蔽注入点）
            v = self.wm.verify()                      # ④ 验证（外部观察者）
            self._rolling_rates.append(v["hit_rate"])
            self._growth_tick(v)                      # ⑤ 拓扑生长/回退
            self._wal("episode", {                    # ⑥ 留痕（全量快照）
                "nodes": self._snapshot_nodes(),
                "edges": self._snapshot_edges(),
                "hit_rate": v["hit_rate"]})
        return self.state()

    def run(self, n: int = 10) -> Dict:
        """推演 n tick + 汇总（verify 事件落 WAL）。"""
        self.step(n=n)
        summary = {
            "ticks": int(n), "tick": self.wm.tick,
            "rolling_hit_rate": round(sum(self._rolling_rates)
                                      / len(self._rolling_rates), 4)
            if self._rolling_rates else 1.0,
            "growth_summary": {
                "born": sum(1 for g in self._growth_log
                            if g["event"] in ("growth", "re-grow")),
                "confirmed": sum(1 for g in self._growth_log
                                 if g["event"] == "confirm"),
                "reverted": sum(1 for g in self._growth_log
                                if g["event"] == "revert"),
                "active": len(self._hypotheses)}}
        self._wal("verify", summary)
        return {"status": "ok", **summary}

    # ---------- 拓扑生长（Overmann：结构生长非权重调整）----------

    def _anomaly_count(self, eid: str) -> int:
        """实体 eid 在最近 K tick 的 anomaly 次数（死区门控输入）。"""
        lo = self.wm.tick - self.K
        return sum(1 for a in self.wm._anomalies
                   if a["entity"] == eid and a["tick"] > lo)

    def _persistence(self, eid: str) -> Optional[float]:
        """方向持续性 = 净位移/路径长（定向运动 →1，随机游走 →0）。

        判据来源：经验标定（2026-10-10，本仓随机游走 null 采样；无理论章节）。
        net/path 的 null 分布强烈依赖样本数 n——n=10 时 p99≈0.76、max≈0.92，
        固定阈值 0.45 落在 null 尾部之内（21% 随机游走样本越过，60/60 seed
        触发通道②）；n=80 时 null p99≈0.25、max≈0.37（84400 样本），0.45 才
        落在 null 之上。故本估计器要求**整窗**样本（n == persistence_window）：
        样本不足（早期 tick / 遮蔽缺链）⇒ None，不给宽尾不可靠估计。窗口取
        80 是本判据成立的条件（`persistence_window` 调小会让 0.45 重回 null
        尾内、通道②对随机游走误报）。"""
        traj = [rec["entities"][eid]
                for rec in self.wm.history[-self.persistence_window:]
                if eid in rec["entities"]]
        if len(traj) < self.persistence_window:
            return None
        path = sum(math.dist(traj[i - 1], traj[i])
                   for i in range(1, len(traj)))
        if path < 1e-6:
            return None
        return math.dist(traj[0], traj[-1]) / path

    def _hit_for(self, verify_result: Dict, eid: str) -> Optional[bool]:
        for d in verify_result.get("details", []):
            if d["entity"] == eid:
                return bool(d["hit"])
        return None

    def _traj(self, eid: str) -> List[Tuple[float, float, float]]:
        return [tuple(rec["entities"][eid])
                for rec in self.wm.history[-self.W:]
                if eid in rec["entities"]]

    def _growth_tick(self, verify_result: Dict) -> None:
        tick = self.wm.tick
        # 1) 已有假设：验证窗口推进（confirm / revert）
        for subject in list(self._hypotheses.keys()):
            h = self._hypotheses[subject]
            H = self.wm.nodes.get(h["node"])
            if H is None:                       # 被外部移除 → 清账
                self._hypotheses.pop(subject, None)
                continue
            hit = self._hit_for(verify_result, subject)
            node_A = self.wm.nodes.get(subject)
            away = False
            if node_A is not None:
                # 位移相对「假设目标方向」的余弦：>0.2 趋向 / <-0.2 背离
                dx = node_A.pos[0] - h["last_pos"][0]
                dz = node_A.pos[2] - h["last_pos"][2]
                tx, tz = H.pos[0] - h["last_pos"][0], H.pos[2] - h["last_pos"][2]
                dl, tl = math.hypot(dx, dz), math.hypot(tx, tz)
                if dl > 1e-6 and tl > 1e-6:
                    cos = (dx * tx + dz * tz) / (dl * tl)
                    if cos > 0.2:
                        h["toward"] += 1
                    if cos < -0.2:
                        h["away_streak"] += 1
                        away = True
                    else:
                        h["away_streak"] = 0
                else:
                    h["away_streak"] = 0
                h["last_pos"] = node_A.pos
            if h["phase"] == "candidate":
                h["ticks"] += 1
                if hit:
                    h["hits"] += 1
                need = self.verify_window * self.confirm_ratio
                if h["away_streak"] >= 3:
                    self._revert(h, "主体持续背离假设目标")
                elif h["ticks"] >= self.verify_window:
                    if h["toward"] >= need:
                        h["phase"] = "confirmed"
                        H.confidence = 0.5
                        ev = {"tick": tick, "event": "confirm",
                              "subject": self._alias_of(subject),
                              "hypothesis": self._alias_of(h["node"]),
                              "pos": list(H.pos),
                              "toward": h["toward"], "hits": h["hits"],
                              "window": h["ticks"],
                              "criterion": f"toward≥{need}（假设持续解释主体行为）"}
                        self._growth_log.append(ev)
                        self._wal("confirm", ev)
                    else:
                        self._revert(h, f"验证窗口趋向占比不足 "
                                        f"({h['toward']}/{h['ticks']}<{need})")
            else:  # confirmed：置信爬升至 0.7 封顶；位置持续跟踪；持续背离 → 回退
                H.confidence = min(0.7, H.confidence + 0.05)
                est = estimate_hidden_target(self._traj(subject), self.wm.size)
                if est is not None and est[1] > 0.5:
                    ex, ey, ez = est[0]
                    H.pos = (round(H.pos[0] + 0.5 * (ex - H.pos[0]), 2),
                             H.pos[1],
                             round(H.pos[2] + 0.5 * (ez - H.pos[2]), 2))
                    H.last_seen = tick
                if h["away_streak"] >= 4:
                    self._revert(h, "固化后主体持续背离（假设失效）")
        # 2) 新生长评估（无假设的已观测实体；双通道触发，确定性）
        for eid, n in list(self.wm.nodes.items()):
            if eid in self._hypotheses or n.attrs.get("hypothesis"):
                continue
            if n.last_seen != tick:
                continue
            # 通道①窄边界失准：anomaly 死区；通道②宽边界下的未解释结构：
            # 模型按随机行为预测（cons<阈值=承认无知）但方向持续性超随机游走
            trig = None
            if self._anomaly_count(eid) >= self.deadzone:
                trig = f"anomaly≥{self.deadzone}/last{self.K}ticks"
            else:
                cons, _ = self.wm._motion_stats(eid, 8)
                per = self._persistence(eid)
                if (cons < self.wm.entropy_threshold and per is not None
                        and per >= self.persistence_min):
                    trig = f"stochastic_mask+structure(cons={cons},per={per:.2f})"
            if trig is None:
                continue
            est = estimate_hidden_target(self._traj(eid), self.wm.size)
            if est is None or est[1] <= 0.5:
                continue
            pos, quality = est
            self._grow(eid, pos, quality, trigger=trig)

    def _grow(self, subject: str, pos: Tuple[float, float, float],
              quality: float, trigger: str = "") -> None:
        """生长假设实体 H + seek(subject→H) 关系边（拓扑事件，WAL 留痕）。

        结构生长非权重调整：若主体已有出边（旧关系解释），新边生长时旧边
        被取代（superseded 留痕——纠正链语义，不做静默修改）；若主体已有
        假设（re-grow），旧假设先按同一纪律摘除。
        """
        tick = self.wm.tick
        hid = f"hyp_{subject}_{tick}"
        old = self._hypotheses.pop(subject, None)
        if old is not None:                     # re-grow：摘除旧假设
            self.wm.nodes.pop(old["node"], None)
            self.wm._conditions.pop(old["node"], None)
            # 摘除**所有与该假设关联**的边（入边 + 出边）。只删 target==hid
            # 会漏掉假设节点自身长出的出边（world_model.infer_patterns 在假设
            # 节点进历史时会为它推关系边），残留悬空边（源/目标已不在 nodes）。
            self.wm.edges = [e for e in self.wm.edges
                             if e.target != old["node"]
                             and e.source != old["node"]]
        superseded = [e.to_dict() for e in self.wm.edges if e.source == subject]
        self.wm.nodes[hid] = WMNode(
            eid=hid, category="hidden_target", pos=pos, confidence=0.3,
            first_seen=tick, last_seen=tick,
            attrs={"hypothesis": True, "subject": subject,
                   "born_tick": tick, "phase": "candidate"})
        self.wm._conditions[hid] = {
            "observation_tool": "simloop_growth",
            "inferred_from": "ray_convergence",
            "time_window": [tick, tick],
            "existence_constraint": "假设存在中（验证窗口内）",
            "fitness": quality}
        self.wm.edges = [e for e in self.wm.edges if e.source != subject]
        self.wm.edges.append(WMEdge(source=subject, relation="seek",
                                    target=hid, confidence=0.4,
                                    evidence="inferred"))
        node_A = self.wm.nodes[subject]
        self._hypotheses[subject] = {
            "_subject": subject, "node": hid, "phase": "candidate",
            "born_tick": tick, "ticks": 0, "hits": 0, "toward": 0,
            "away_streak": 0, "last_pos": tuple(node_A.pos)}
        ev = {"tick": tick, "event": "growth",
              "subject": self._alias_of(subject),
              "hypothesis": self._alias_of(hid),
              "pos": list(pos), "fitness": quality,
              "trigger": trigger,
              "superseded": superseded}
        self._growth_log.append(ev)
        self._wal("growth", ev)

    def _revert(self, h: Dict, reason: str) -> None:
        """回退假设（验证不成立不固化）；anomaly 留痕不删。

        摘除**所有与该假设关联**的边（入边 + 出边）：只删 target==hid 会漏掉
        假设节点自身的出边，残留悬空边（源/目标已不在 nodes，graph()/WAL
        快照仍会导出它）。
        """
        hid = h["node"]
        ev = {"tick": self.wm.tick, "event": "revert",
              "subject": self._alias_of(h.get("_subject", "")),
              "hypothesis": self._alias_of(hid),
              "pos": list(self.wm.nodes[hid].pos)
              if hid in self.wm.nodes else None,
              "reason": reason}
        self.wm.nodes.pop(hid, None)
        self.wm._conditions.pop(hid, None)
        self.wm.edges = [e for e in self.wm.edges
                         if e.target != hid and e.source != hid]
        self._hypotheses.pop(h.get("_subject", ""), None)
        self._growth_log.append(ev)
        self._wal("revert", ev)

    # ---------- 快照与回放 ----------

    def _snapshot_nodes(self) -> Dict:
        out = {}
        for eid, n in self.wm.nodes.items():
            key = self._alias_of(eid)
            out[key] = {"category": n.category, "pos": list(n.pos),
                        "confidence": round(n.confidence, 3),
                        "behavior_inferred": n.behavior_inferred,
                        "hypothesis": bool(n.attrs.get("hypothesis"))}
            if n.attrs.get("hypothesis"):
                out[key]["phase"] = self._hypotheses.get(
                    n.attrs.get("subject", ""), {}).get("phase", "unknown")
        return out

    def _snapshot_edges(self) -> List[Dict]:
        return [{"source": self._alias_of(e.source), "relation": e.relation,
                 "target": self._alias_of(e.target),
                 "confidence": round(e.confidence, 3),
                 "evidence": e.evidence} for e in self.wm.edges]

    def replay(self, records: Optional[List[Dict]] = None) -> Dict:
        """WAL 顺序重放：episode 全量快照覆盖式推进 → 重建终态。
        一致性判据：重建终态 == 当前 graph()（WAL 忠实记录世界图演化）。"""
        state = {"tick": 0, "nodes": {}, "edges": [], "episodes": 0}
        for r in (records if records is not None else self.wal):
            if r.get("kind") == "episode":
                state = {"tick": r["tick"], "nodes": r["payload"]["nodes"],
                         "edges": r["payload"]["edges"],
                         "episodes": state["episodes"] + 1}
        return state

    def replay_consistent(self) -> bool:
        """回放重建终态与当前世界图逐节点逐边一致。"""
        state = self.replay()
        live = self._snapshot_nodes()
        if set(state["nodes"]) != set(live):
            return False
        for eid, n in live.items():
            s = state["nodes"].get(eid, {})
            if (s.get("category") != n["category"]
                    or s.get("pos") != n["pos"]
                    or s.get("behavior_inferred") != n["behavior_inferred"]
                    or s.get("hypothesis") != n["hypothesis"]):
                return False
        live_edges = {(e["source"], e["relation"], e["target"])
                      for e in self._snapshot_edges()}
        wal_edges = {(e["source"], e["relation"], e["target"])
                     for e in state["edges"]}
        return live_edges == wal_edges

    # ---------- 装载环（认知图 → 世界图先验）----------

    @staticmethod
    def parse_seed_entity(line: str):
        """解析 flush 载荷实体行：`- entity: {category} id={eid} pos=(x, y, z) …`。

        flush 写出的类别是裸文本（无引号、无转义），类别可含空格。按分隔标记
        定位（` id=` / ` pos=(` / `)`）而非按空格裸切——裸切会把多词类别截成
        首词、错位字段（写读不对称 = 持久化往返损坏）。
        """
        if not line.strip().startswith("- entity:"):
            return None
        try:
            seg = line.strip()[len("- entity:"):].strip()
            if " id=" not in seg or " pos=(" not in seg:
                return None
            category, rest = seg.split(" id=", 1)
            eid, rest = rest.split(" pos=(", 1)
            pos_seg = rest.split(")", 1)[0]
            pos = tuple(float(v) for v in pos_seg.split(","))
            if len(pos) != 3:
                return None
            return {"category": category.strip(), "eid": eid.strip(),
                    "pos": (pos[0], pos[1], pos[2])}
        except (IndexError, ValueError):
            return None

    def load_priors(self, mdcg_root: str, limit: int = 20) -> Dict:
        """认知图 contextual 层（tags 含 wm_simloop 的节点）→ 世界图实体先验。

        诚实边界：v1 只重建位置+类别先验（行为不重建，落 wander 缺省）。
        """
        root = os.environ.get("MDCG_ROOT", mdcg_root or "")
        seeded = []
        ctx = os.path.join(root, "contextual")
        if root and os.path.isdir(ctx):
            for fn in sorted(os.listdir(ctx)):
                if len(seeded) >= limit:
                    break
                path = os.path.join(ctx, fn)
                if not fn.endswith(".md") or not os.path.isfile(path):
                    continue
                try:
                    text = open(path, encoding="utf-8").read()
                except OSError:
                    continue
                fm = text.split("---")[1] if text.startswith("---") else ""
                if "wm_simloop" not in fm.split("tags:")[-1][:200]:
                    continue
                for line in text.splitlines():
                    e = self.parse_seed_entity(line)
                    if e:
                        new_id = self.scene.add_entity(
                            e["category"], behavior="wander", pos=e["pos"])
                        seeded.append({"source": fn, **e, "seeded_id": new_id})
        return {"status": "ok", "mdcg_root": root, "seeded": len(seeded),
                "entities": seeded,
                "note": "v1 只重建位置+类别先验（行为不重建）"}

    # ---------- 回写环（WAL → 认知图 contextual 载荷）----------

    def flush_payloads(self, out_path: Optional[str] = None) -> Dict:
        """推演结果 → 认知图 contextual 节点载荷（frontmatter + 正文）。

        对齐 data/mdcg/contextual/arc_agi_ls20_*.md 先例格式；
        v1 不旁路写真源——真实摄取走 mdcg 正规通道（cg write/ingest）。

        issue #324：`out_path` 落盘改**原子写**（同目录临时文件 + `os.replace`）——
        旧实现 `open(out_path, "w")` 先截断再写，中途失败即留下截断坏档。
        路径**白名单**不在此层：本方法是调用方已持有对象时的库 API；模型驱动面
        经 `SpacetimeMemoryEngine.wm_simloop("flush")` 进入，越界闸在门面侧
        （`_confine_simloop_out_path`）。边界（如实标注）：`os.replace` 保证
        「目标文件要么是旧内容、要么是完整新内容」，**不**保证掉电后元数据已落盘
        （未做目录 fsync；Windows 上亦无法对目录取句柄）。
        """
        self._flush_count += 1
        g = self.wm.graph()
        wid = f"simloop-{self._flush_count}"
        fm = {
            "id": f"wm_simloop_{wid}_t{self.wm.tick}",
            "layer": "contextual", "modality": "text",
            "tags": ["wm_simloop", f"world:{wid}", f"tick:{self.wm.tick}"],
            "condition_space": json.dumps({
                "time_window": [round(self._clock(), 3), round(self._clock(), 3)],
                "observation_tool": "wm_simloop",
                "wal_seq_range": [0, max(0, self._seq - 1)]}, ensure_ascii=False),
            "confidence": 0.6, "importance": 0.5,
            "created_at": round(self._clock(), 3),
            "edges": [], "evidence_count": 0, "positive_evidence": 0,
            "negative_evidence": 0, "protected": False,
            "sensitivity": "internal", "verification_basis": "wal_replay",
            "non_applicable_conditions": []}
        lines = [f"# 世界模型推演循环记忆 - {wid} (tick {self.wm.tick})", "",
                 f"## 世界快照 (tick={self.wm.tick})"]
        for eid, n in g["nodes"].items():
            tag = "hypothesis" if n.get("attrs", {}).get("hypothesis") else "entity"
            lines.append(f"- {tag}: {n['category']} id={eid} "
                         f"pos=({n['pos'][0]}, {n['pos'][1]}, {n['pos'][2]}) "
                         f"confidence={round(n['confidence'], 3)}")
        rate = round(self._rolling_rates[-1], 4) if self._rolling_rates else 1.0
        lines += ["", f"命中率: {rate} | 节点 {g['node_count']} | "
                  f"边 {g['edge_count']} | WAL {self._seq} 条",
                  f"growth: born={sum(1 for x in self._growth_log if x['event'] in ('growth', 're-grow'))} "
                  f"confirmed={sum(1 for x in self._growth_log if x['event'] == 'confirm')} "
                  f"reverted={sum(1 for x in self._growth_log if x['event'] == 'revert')}"]
        if self._growth_log:
            lines.append("")
            for ev in self._growth_log[-5:]:
                lines.append(f"- {ev['event']}@{ev['tick']}: "
                             f"{ev.get('subject')}→{ev.get('hypothesis')} "
                             f"{ev.get('reason', ev.get('criterion', ''))}")
        body = "\n".join(lines)
        md = ("---\n" + "\n".join(
            f"{k}: {json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else json.dumps(v, ensure_ascii=False)}"
            for k, v in fm.items()) + "\n---\n" + body + "\n")
        payload = {"id": fm["id"], "frontmatter": fm, "body_md": md}
        written = None
        if out_path:
            # issue #324（数据丢失面）：旧实现 `open(out_path, "w")` **先截断再写**
            # ⇒ 写中途失败即把目标文件毁成空/半截（先毁后写），且无原子性。
            # 修法：同目录临时文件写完再 `os.replace` 换入（原子、覆盖式语义保留）。
            # 越界/白名单闸在门面侧（`SpacetimeMemoryEngine._confine_simloop_out_path`）——
            # 本方法是被 Python 直接持有的库 API，不做路径白名单（调用方已持对象）。
            # 判据来源：经验标定（本件 #324）——口径取「临时文件 + os.replace」；
            # 追不到更早出处。
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            tmp_path = "%s.tmp-%s" % (out_path, uuid.uuid4().hex[:8])
            try:
                with open(tmp_path, "w", encoding="utf-8") as f:
                    f.write(md)
                os.replace(tmp_path, out_path)
            except BaseException:
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except OSError:
                    pass
                raise
            written = out_path
        return {"status": "ok", "payload": payload, "written": written,
                "note": "载荷已生成；真实摄取走 mdcg 正规通道（不旁路写真源）"}

    # ---------- 报告 ----------

    def growth_events(self, limit: int = 20) -> List[Dict]:
        return self._growth_log[-max(1, int(limit)):]

    def report(self) -> Dict:
        g = self.wm.graph()
        return {"status": "ok", "tick": self.wm.tick,
                "node_count": g["node_count"], "edge_count": g["edge_count"],
                "anomaly_count": g["anomaly_count"],
                "rolling_hit_rate": round(sum(self._rolling_rates)
                                          / len(self._rolling_rates), 4)
                if self._rolling_rates else 1.0,
                "hypotheses_active": len(self._hypotheses),
                "growth_events": len(self._growth_log),
                "wal_records": len(self.wal),
                "replay_consistent": self.replay_consistent()}

    def state(self) -> Dict:
        return {"status": "ok", "tick": self.wm.tick,
                "nodes": len(self.wm.nodes), "edges": len(self.wm.edges),
                "wal": len(self.wal),
                "hypotheses": {k: {"node": v["node"], "phase": v["phase"]}
                               for k, v in self._hypotheses.items()},
                "masked": sorted(self._mask)}


if __name__ == "__main__":
    # 自测：漂移正弦遮蔽目标（确定性外部世界，模型只见追逐者轨迹）
    # 通道②要求整窗（persistence_window=80）样本，故场景须让追逐者持续移动
    # 至少一个窗口——场地取 48、速度降到 0.5，避免 ~27 tick 就顶到边界静止
    # （静止后 path≈0、持续性无估计，通道②无从触发）。
    import math as _m
    sl = SimulationLoop(size=48, wal_path=None)
    sl.scene.create_scene(trees=0, water=False)
    t = sl.scene.add_entity("rabbit", behavior="wander", pos=(6, 1.5, 13), speed=0.0)
    a = sl.scene.add_entity("wolf", behavior="seek", goal=t,
                            pos=(2, 1.5, 13), speed=0.5)
    sl._mask = {t}
    sl.wm.perceive(observations=sl._observe())

    def drifting(scene, tick):
        scene.entities[t].pos = (3.0 + 0.3 * tick
                                 + 2.5 * _m.sin(tick * 0.45), 1.5, 13.0)

    sl.step(n=1)
    sl.step(n=140, external=drifting)
    rep = sl.report()
    H = [n for n in sl.wm.nodes.values() if n.attrs.get("hypothesis")]
    print("replay_consistent:", rep["replay_consistent"],
          "| hypotheses:", rep["hypotheses_active"],
          "| growth:", [(g["event"], g["tick"], g.get("trigger", ""))[:2]
                        for g in sl._growth_log])
    for h in H:
        print("H:", h.eid, h.pos, "conf:", h.confidence,
              "| dist(H,T):", round(_m.dist(h.pos, sl.scene.entities[t].pos), 2))
    print("T:", sl.scene.entities[t].pos, "| A:", sl.scene.entities[a].pos)
    print("SELF_TEST_DONE")
