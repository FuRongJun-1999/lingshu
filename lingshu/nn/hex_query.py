# -*- coding: utf-8 -*-
"""hex_query · 连接查询式识别(HEX-CNN-M4.16 · 荣指令:识别的本质是查询)
====================================================================================
荣 2026-09-08:「修改架构——更大分辨率,更多连接组合,识别方式为已有的
连接相似性加上条件判断。要更快,更精确。识别的本质是查询。」

架构(查询式 vs 拟合式):
  连接库:预建的结构化连接记录(每条=已知物体的完整关系网络);
  关系提取:新图 → 部件检出 + 部件间关系对(递归搜索输出增强);
  查询:新图关系描述 → 在库中匹配相似连接(比对操作,零梯度);
  条件判断:四态筛选(ACCEPT / DEFER / REJECT / BLINDSPOT)——判据见 query_match #121:
    ACCEPT    最优 score ≥ th_accept(默认 0.6)
    DEFER     th_defer ≤ 最优 score < th_accept(默认 0.3)
    REJECT    最优与次优 score **相等**(margin==0)⇒ 两条配方不可区分,冲突,拒绝作答
    BLINDSPOT 最优 score < th_defer ⇒ 库中无相关配方(知识盲区,非冲突)
    score = (conn 命中 + 关系命中) / (conn 需求 + 关系需求 + **未被匹配消费的检测数**)
            末项为 #121 补上的 precision 侧:旧口径只有前两项(纯 recall),
            故「配方全命中 + 多余部件」与「恰好全命中」同分(实测均 1.0)。
  输出:物体判定 + 置信 + 缺失关系预测(补全)。

F06 关系判定(recipe role → detection 一对一绑定):
  配方 tags.relations 的真实格式 = [{"from": 角色, "to": 角色, "rel": 关系名}, ...]
  (见 build_default_library 的 car/snowman/flag)。判定链:
    _match_binding   最大匹配命中数 + 规范检测序上的稳定 tie-break → conn→detection
                     绑定(一对一;未满足的 conn 记 -1;绑定表用**原输入下标**);
    _bind_roles      绑定表 → 角色→检测下标(同一检测至多属一个角色,确定性);
    _relation_criteria 两检测的位置 → 逐关系判据真值表(每条关系各按自己的 row/col
                     判据消费;斜向可同时 above 与 left_of;同格不存在方向关系);
    _relation_fact   同一对的单一主词投影(历史证据字段 fact;不参与满足判定);
    _eval_relations  逐条关系判定 + 逐条证据(witness/observed/原因)。
  评分 = (conn 命中数 + 关系满足数) / (conn 总数 + 关系条数)——关系是新的约束项;
  无 relations 的旧配方走原口径(命中/conn 总数),评分与返回结构逐位不变;
  未知关系名/畸形关系条目/角色不在配方/角色未绑定 ⇒ 一律不计满足(fail-closed),
  并逐条落原因——不把「不可判」伪装成「已满足」。
  **顺序无关**:同一条配方下 detections 的任意排列必须给出同一 score/verdict/角色
  绑定/关系证据;规范序只用于挑选,挑选后立刻折算回原输入下标,不会错指。

速度:查询=比对操作(毫秒级),非梯度计算(分钟级)。
精度:关系结构匹配比单点分类更鲁棒(多证据联合裁决)。
"""
from __future__ import annotations
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

ALGO = "hex_query-0.1"

from .hex_hier import OBJ, SHAPES, COLORS, POS

# ==================== 连接库(声明式,零训练) ====================

class ConnectionLibrary:
    """连接库:已知物体的结构化连接记录集合。
    每条记录 = 一个物体的完整关系网络(部件+部件间连接)。
    加新物体 = 加一条记录(声明式,零训练)。"""

    def __init__(self):
        self.records: List[Dict] = []

    def add_record(self, name: str, connections: List[Dict],
                   tags: Optional[Dict] = None):
        """添加一条连接记录。
        connections: [{type: 部件类型, zone: 方位, color: 颜色或"*"}]
        tags: 附加属性(物体级属性如体色/纹理)。"""
        rec = {"name": name, "connections": connections,
               "tags": tags or {}, "id": len(self.records)}
        self.records.append(rec)

    def save(self, path: str):
        """Publish a complete library, preserving the previous file on failure.

        Each call owns one exclusive temporary file. Concurrent saves publish
        in replace order; contents are not merged. Preserve the old target's
        mode and symlink destination, as the previous in-place writer did.
        File fsync does not promise directory durability after power loss.
        """
        import uuid

        target = os.path.realpath(path)
        temporary = os.path.join(os.path.dirname(target),
                                 f".lingshu-query.tmp-{os.getpid()}-{uuid.uuid4().hex}")
        owned = False
        try:
            with open(temporary, "x", encoding="utf-8") as f:
                owned = True
                json.dump(self.records, f, ensure_ascii=False, indent=1)
                f.flush()
                os.fsync(f.fileno())
            try:
                os.chmod(temporary, os.stat(target).st_mode & 0o7777)
            except FileNotFoundError:
                pass
            os.replace(temporary, target)
        except BaseException:
            if owned:
                try:
                    if os.path.exists(temporary):
                        try:
                            os.chmod(temporary, 0o600)
                        except OSError:
                            pass
                    os.unlink(temporary)
                except OSError:
                    pass
            raise

    @classmethod
    def load(cls, path: str) -> "ConnectionLibrary":
        lib = cls()
        with open(path, encoding="utf-8") as f:
            lib.records = json.load(f)
        return lib

    def __len__(self):
        return len(self.records)


# ==================== 默认连接库(复合体配方) ====================

_ZONES = {
    "top": {"r0", "r1", "r2"}, "bottom": {"r6", "r7", "r8"},
    "left": {"r0", "r3", "r6"}, "right": {"r2", "r5", "r8"},
    "top-left": {"r0"}, "top-center": {"r1"}, "top-right": {"r2"},
    "center": {"r4"}, "bottom-left": {"r6"}, "bottom-center": {"r7"},
    "bottom-right": {"r8"},
    "upper-left": {"r0", "r3"}, "upper-right": {"r2", "r5"},
    "lower-left": {"r6", "r7"}, "lower-right": {"r7", "r8"},
}


def build_default_library() -> ConnectionLibrary:
    """构建默认连接库(复合体配方——零训练,声明式)。"""
    lib = ConnectionLibrary()
    # 单部件物体
    for s in SHAPES:
        for c in COLORS:
            lib.add_record(
                f"{s}_{c}",
                [{"type": s, "zone": "*", "color": c}],
                tags={"single": True})
    # 复合体(多部件+关系)
    lib.add_record("car", [
        {"type": "stripe", "zone": "top", "color": "*", "role": "body"},
        {"type": "circle", "zone": "bottom-left", "color": "*", "role": "wheel"},
        {"type": "circle", "zone": "bottom-right", "color": "*", "role": "wheel"},
    ], tags={"relations": [
        {"from": "body", "to": "wheel", "rel": "above"},
        {"from": "wheel", "to": "wheel", "rel": "side_by_side"}]})
    lib.add_record("snowman", [
        {"type": "circle", "zone": "top", "color": "*", "role": "head"},
        {"type": "circle", "zone": "bottom", "color": "*", "role": "body"},
    ], tags={"relations": [{"from": "head", "to": "body", "rel": "above"}]})
    lib.add_record("flag", [
        {"type": "stripe", "zone": "left", "color": "*", "role": "pole"},
        {"type": "triangle", "zone": "top-right", "color": "*", "role": "banner"},
    ], tags={"relations": [{"from": "pole", "to": "banner", "rel": "right_of"}]})
    return lib


# ==================== 关系提取(从递归搜索输出增强) ====================

def extract_relations(detections: List[Dict]) -> List[Dict]:
    """从部件检出清单 → 部件间关系对(增强输出)。"""
    relations = []
    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            a, b = detections[i], detections[j]
            qa, qax = divmod(int(a["pos"][1:]), 3)
            qb, qbx = divmod(int(b["pos"][1:]), 3)
            rel = {"from": a["pos"], "to": b["pos"],
                   "type_a": a["obj"].split("|")[0],
                   "type_b": b["obj"].split("|")[0],
                   "color_a": a["obj"].split("|")[1],
                   "color_b": b["obj"].split("|")[1]}
            # 空间关系
            if qa < qb:
                rel["spatial"] = "above"
            elif qa > qb:
                rel["spatial"] = "below"
            elif qax < qbx:
                rel["spatial"] = "left_of"
            else:
                rel["spatial"] = "right_of"
            relations.append(rel)
    return relations


def _det_axes(det: Dict) -> Optional[Tuple[int, int]]:
    """检测 pos → (row, col)（9 宫格 `r{row*3+col}` 口径，与 extract_relations 同源）。

    畸形/缺失 pos ⇒ None（不可判：不猜行列，调用方按「不可判」处置——既不进
    行列判据，也不编造方向）。
    """
    try:
        q = int(str(det.get("pos", ""))[1:])
    except (TypeError, ValueError):
        return None
    return q // 3, q % 3


# ==================== 查询式识别 ====================

def query_match(detections: List[Dict], lib: ConnectionLibrary,
                th_accept: float = 0.6, th_defer: float = 0.3
                ) -> List[Dict]:
    """查询式识别:新图关系 → 连接库匹配 → 条件判断(四态)。
    零梯度,纯比对——毫秒级。"""
    if not detections:
        return [{"verdict": "BLINDSPOT", "note": "无检出"}]
    # dsh-bounty #1800①：知识侧的对称退化——空连接库（`ConnectionLibrary()`
    # 或 save→load 往返后的 0 记录库）是合法状态，恰是本模块对 BLINDSPOT 的
    # 定义（「库中无相关配方」）。旧实现在 results[0] 处 IndexError，四态契约
    # 被崩溃打破；与数据侧兜底同形状返回。
    if not lib.records:
        return [{"verdict": "BLINDSPOT", "note": "连接库为空（无已登记配方）"}]
    results = []
    for rec in lib.records:
        score, report = _score_record(rec, detections)
        entry = {"name": rec["name"], "score": round(score, 3), "id": rec["id"]}
        if report is not None:
            # 关系证据先挂在私有键上，等 best 选定后再摘出来：all_scores 快照
            # 必须保持 {name, score, id} 旧形状——排序快照是既有契约。
            entry["_relations_report"] = report
        results.append(entry)
    results.sort(key=lambda x: -x["score"])
    best = results[0]
    second = results[1] if len(results) > 1 else {"score": 0}
    # #121：四态齐备。旧实现只按 best 分数三档，docstring 承诺的「冲突 REJECT」
    # 在函数体里连字面量都没有、不可达；而 margin（最优−次优）算了却从不参与判定。
    # 现按 margin==0（最优与次优**不可区分**）判冲突 —— 不引入新阈值。
    if best["score"] < th_defer:
        v = "BLINDSPOT"                  # 无配方达下界：知识盲区（非冲突）
    elif len(results) > 1 and best["score"] == second["score"]:
        v = "REJECT"                     # 冲突：两条配方不可区分，拒绝作答
    elif best["score"] >= th_accept:
        v = "ACCEPT"
    else:
        v = "DEFER"
    # Copy the ranking entries before annotating ``best``.  Otherwise the
    # first entry in ``all_scores`` would reference ``best`` itself and make
    # the returned report impossible to serialize as JSON.
    all_scores = [{k: val for k, val in item.items()
                   if not k.startswith("_")} for item in results[:5]]
    best["verdict"] = v
    best["margin"] = round(best["score"] - second["score"], 3)
    best["all_scores"] = all_scores
    # F06：命中配方带 tags.relations 时，回报角色绑定与逐条关系证据；无 relations
    # 的旧配方不加任何键（返回结构与旧版逐位一致）。
    report = best.pop("_relations_report", None)
    if report is not None:
        best["roles"] = report["roles"]
        best["role_detections"] = report["role_detections"]
        best["relations"] = report["relations"]
        best["conn_hits"] = report["conn_hits"]
        best["conn_total"] = report["conn_total"]
        best["relation_hits"] = report["relation_hits"]
        best["relation_total"] = report["relation_total"]
    return [best]


def _conn_matches(conn: Dict, det: Dict) -> bool:
    """单条 conn 与单个检测是否匹配（类型/方位/颜色判据）。"""
    det_obj = det["obj"].split("|")
    det_shape, det_color, det_pos = det_obj[0], det_obj[1], det["pos"]
    if det_shape != conn["type"]:
        return False
    zone = conn.get("zone", "*")
    if zone != "*" and det_pos not in _ZONES.get(zone, set()):
        return False
    color = conn.get("color", "*")
    if color != "*" and det_color != color:
        return False
    return True


def _canonical_det_order(detections: List[Dict]) -> List[int]:
    """检测的**规范全序** → 原下标表（键只取检测内容，不取输入位置）。

    规范序 = ((row, col), obj 串) 升序。同内容 ⇒ 同序 ⇒ 匹配绑定与关系证据不随
    detections 的排列变化。行列不可解析者排到最后；内容完全相同的检测互为等价
    （roles/证据回报的是位置），故用原下标兜底 tie-break 不改变任何可观测输出。
    """
    return sorted(range(len(detections)),
                  key=lambda i: (_det_axes(detections[i]) or (9, 9),
                                 str(detections[i].get("obj", ""))))


def _max_matching_size(conns: List[Dict], det_indices: List[int],
                       detections: List[Dict],
                       skip_conns: Optional[set] = None) -> int:
    """子图（给定 conn × 给定检测下标）上的**最大匹配命中数**（增广路，只数命中）。

    命中数只取决于「哪些 conn 能匹配哪些检测」，与两侧枚举顺序无关——命中数的
    顺序无关性由此保证（顺序只影响拿到哪一个最大匹配，而匹配的挑选由
    _match_binding 的稳定 tie-break 负责）。
    """
    skip = skip_conns or ()
    # match_det[di] = 分配给检测 di 的 conn 下标（-1 表示未匹配）
    match_det = {di: -1 for di in det_indices}

    def try_assign(ci: int, seen: set) -> bool:
        """为 conn ci 寻找增广路（把它匹配到某个检测，允许改配已有 conn）。"""
        for di in det_indices:
            if di in seen or not _conn_matches(conns[ci], detections[di]):
                continue
            seen.add(di)
            if match_det[di] == -1 or try_assign(match_det[di], seen):
                match_det[di] = ci
                return True
        return False

    hits = 0
    for ci in range(len(conns)):
        if ci in skip:
            continue
        if try_assign(ci, set()):
            hits += 1
    return hits


def _match_binding(rec: Dict, detections: List[Dict]) -> Tuple[int, List[int]]:
    """既有匹配（最大匹配 + 稳定 tie-break）→ (命中 conn 数, conn→detection 绑定表)。

    conn→detection 为二部图，命中数 = **最大匹配**：只取决于「有哪些 conn 能被
    哪些检测满足」，与 detections 的输入顺序无关。原实现为贪心 first-fit（取第一个
    匹配的未用检测即 break），通配 conn 会抢占具体 conn 需要的检测，使命中数随
    detections 顺序变化（verdict 可在 ACCEPT/DEFER 间翻转）。最大匹配保证命中数
    最优且顺序无关。

    **绑定本身**也必须与输入顺序无关：纯增广路的「后一个 conn 抢走靠前检测」会让
    等价输入得到不同的角色↔检测对应，F06 的 role 绑定/关系证据随之翻转（同一配方
    评分 0.667 vs 1.0）。这里改为在**规范检测序**（_canonical_det_order，只取检测
    内容）上取「最大匹配里 conn 序 × 规范检测序的字典序最小者」：

      · 先算最大命中数 total（顺序无关）；
      · 再按 conn 声明顺序逐条试定候选检测，并用同一套增广路检查「其余未定 conn
        仍能补足 total」——可行才定下（lexicographic greedy with feasibility）。

    复杂度 O(conn² × 检测² × 增广)，配方规模（conn 个位数、检测 ≤ 9 宫格）下毫秒级；
    不引入通用图匹配或指数搜索。规范序**只用于挑选**，绑定表立即折算回**原输入
    下标**：assign[ci] = 分给 conn ci 的原检测下标（-1 = 该 conn 未满足），一对一由
    匹配本身保证，回报的角色/证据永远指向原检测，不会错指。
    """
    conns = rec["connections"]
    if not detections or not conns:
        return 0, [-1] * len(conns)
    order = _canonical_det_order(detections)
    total = _max_matching_size(conns, order, detections)
    assign = [-1] * len(conns)
    used: set = set()
    matched = 0
    for ci in range(len(conns)):
        for di in order:
            if di in used or not _conn_matches(conns[ci], detections[di]):
                continue
            rest = [d for d in order if d not in used and d != di]
            if _max_matching_size(conns, rest, detections,
                                  skip_conns=set(range(ci + 1))) >= total - matched - 1:
                assign[ci] = di
                used.add(di)
                matched += 1
                break
    return total, assign


def _match_score(rec: Dict, detections: List[Dict]) -> float:
    """单条记录的匹配度 = 被支持的连接数 / 总连接数（conn 侧口径，保持不变）。

    带 tags.relations 的配方走 _score_record（关系计入命中与分母）；无 relations
    时两者逐位一致——本函数仍是 conn 口径的公开小件。
    """
    conns = rec["connections"]
    if not detections or not conns:
        return 0.0
    hits, _ = _match_binding(rec, detections)
    return hits / len(conns)


# ==================== F06 · 关系判定（角色绑定 + 逐关系证据）====================

#: 库内可判的关系名（未列入者一律不判为满足——fail-closed，不猜语义）。
#: 前四个与 extract_relations 的 spatial 词汇同源；side_by_side 是库内声明词
#: （build_default_library 的 car: wheel→wheel），语义 = 同排左右。
_KNOWN_RELATIONS = ("above", "below", "left_of", "right_of", "side_by_side")
#: 对称关系（a 对 b 成立 ⇔ b 对 a 成立）：同 role 枚举有向对时按规范序确定性去重。
_SYMMETRIC_RELATIONS = ("side_by_side",)


def _relation_criteria(a: Dict, b: Dict) -> Optional[Dict[str, bool]]:
    """两检测之间的**逐关系判据真值表**（每条声明关系按自己的 row/col 判据消费）。

    位置 = 9 宫格 (row, col)。五种关系词各自独立取真值：
      above ⇔ row_a < row_b；below ⇔ row_a > row_b；
      left_of ⇔ col_a < col_b；right_of ⇔ col_a > col_b；
      side_by_side ⇔ 同一 row 且不同 col（同排左右）。
    于是一对斜向检测（如 r0 与 r8）可以**同时**满足 above 与 left_of——这不是自造
    词汇，而是每条关系各按自己的行列判据取真。原实现只取 extract_relations 的单一
    主词（行优先 above/below），列判据永远判不到，导致 col0<col2 的 left_of 被误判
    为不成立。

    同一格（row 与 col 都相同）⇒ 不存在任何方向关系（**不编造左右**）⇒ None；
    pos 不可解析 ⇒ None（不可判）。两者都不作为满足证据（fail-closed）。
    """
    ax, bx = _det_axes(a), _det_axes(b)
    if ax is None or bx is None:
        return None
    ra, ca = ax
    rb, cb = bx
    if ra == rb and ca == cb:
        return None
    return {"above": ra < rb, "below": ra > rb,
            "left_of": ca < cb, "right_of": ca > cb,
            "side_by_side": ra == rb and ca != cb}


def _relation_fact(a: Dict, b: Dict) -> Optional[str]:
    """extract_relations 的**单一主词投影**（历史证据字段 fact 的口径，未变）。

    只用于读数展示：extract_relations 对每对检测只给一个 spatial 词（行优先），
    斜向时它会用 above/below 掩盖 left_of/right_of。**满足判定一律走
    _relation_criteria 的真值表，不经过本函数**。同格/不可解析 ⇒ None。
    """
    if a.get("pos") == b.get("pos"):
        return None
    if _det_axes(a) is None or _det_axes(b) is None:
        return None
    facts = extract_relations([a, b])
    return facts[0]["spatial"] if facts else None


def _criteria_true(criteria: Optional[Dict[str, bool]]) -> List[str]:
    """真值表 → 成立的判据名清单（按 _KNOWN_RELATIONS 词序，确定性）。"""
    if not criteria:
        return []
    return [name for name in _KNOWN_RELATIONS if criteria.get(name)]


def _relation_satisfied(declared: str, criteria: Optional[Dict[str, bool]]) -> bool:
    """声明的关系名 vs 判据真值表；不可判（None）与未知关系名一律不算满足。"""
    if not criteria:
        return False
    return criteria.get(declared) is True


def _relations_declared(rec: Dict) -> List:
    """配方声明的 relations 清单（tags.relations 的真实格式 = list[dict]）。

    缺 tags / 缺 relations ⇒ []（旧配方：评分与返回结构保持原口径）；
    非列表的畸形写法**不算「没有 relations」**——按 1 条坏关系落账并判不满足
    （fail-closed：坏写法不得被当成「无约束」而白拿满分）。
    """
    tags = rec.get("tags")
    if not isinstance(tags, dict):
        return []
    raw = tags.get("relations")
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    return [raw]


def _recipe_roles(rec: Dict) -> List[str]:
    """配方 connections 里声明的角色名（按首次出现顺序去重）。"""
    roles: List[str] = []
    for conn in rec["connections"]:
        role = conn.get("role")
        if isinstance(role, str) and role not in roles:
            roles.append(role)
    return roles


def _bind_roles(rec: Dict, assign: List[int],
                rank: Optional[Dict[int, int]] = None) -> Dict[str, List[int]]:
    """角色 → 已绑定检测下标（按**规范检测序**排列）；绑定取自既有匹配的绑定表。

    未满足的 conn（assign=-1）不产出绑定 ⇒ 该角色可能为空集（后续判据按
    「角色未绑定」落账，不拿别的检测顶替）。一对一：一个检测只属于一个 conn，
    因而只属于一个角色——重复 role（如 car 的两个 wheel）得到的是不同检测。
    排列顺序按 rank（规范检测序，只取检测内容）而不是输入下标：否则同一角色在
    等价输入下会报出 [r6, r8] / [r8, r6] 两种次序，证据面看似不稳定。返回值仍是
    **原输入下标**，只是次序确定。
    """
    roles: Dict[str, List[int]] = {}
    for ci, conn in enumerate(rec["connections"]):
        role = conn.get("role")
        di = assign[ci] if ci < len(assign) else -1
        if not isinstance(role, str) or di < 0:
            continue
        roles.setdefault(role, []).append(di)
    order = rank or {}
    return {role: sorted(dets, key=lambda i: order.get(i, i))
            for role, dets in roles.items()}


def _relation_witness(name: str, from_role: str, to_role: str,
                      f_dets: List[int], t_dets: List[int],
                      detections: List[Dict], rank: Dict[int, int]
                      ) -> Tuple[Optional[Dict], List[Dict]]:
    """在已绑定检测里找满足声明的证据对 → (witness 或 None, 实际观测表)。

    枚举**两个不同检测之间的有序对**（fi ≠ ti）；两侧都按规范检测序（rank）排，
    枚举顺序只取决于检测内容 ⇒ 换序后证据逐位稳定。同 role 关系（wheel→wheel）的
    两个 slot 语义对称：有向关系（above/below/left_of/right_of）必须把两个方向都
    算上，否则「后者相对前者成立」的合法检测对被漏掉（原实现只取 fi<ti，wheel→wheel
    的 right_of/above 反向永远判不成立）；对称关系（side_by_side）在同一无序对上
    两个方向同真，按规范序只留一次（确定性去重，不重复消费同一条关系）。跨 role 的
    两个检测分属不同 conn（一对一绑定），不会重复，无需去重。

    每条证据同时给两个字段：fact = extract_relations 单一主词投影（历史字段），
    facts = 逐关系判据真值（判定依据）。二者不一致时以 facts 为准。
    """
    observed: List[Dict] = []
    witness: Optional[Dict] = None
    f_list = sorted(set(f_dets), key=lambda i: rank.get(i, i))
    t_list = sorted(set(t_dets), key=lambda i: rank.get(i, i))
    symmetric = name in _SYMMETRIC_RELATIONS
    for fi in f_list:
        for ti in t_list:
            if fi == ti:
                continue
            if from_role == to_role and symmetric and \
                    rank.get(fi, fi) > rank.get(ti, ti):
                continue
            criteria = _relation_criteria(detections[fi], detections[ti])
            item = {"from": detections[fi]["pos"], "to": detections[ti]["pos"],
                    "fact": _relation_fact(detections[fi], detections[ti]),
                    "facts": _criteria_true(criteria)}
            observed.append(item)
            if witness is None and _relation_satisfied(name, criteria):
                witness = item
    return witness, observed


def _eval_relations(rels: List, roles: Dict[str, List[int]],
                    detections: List[Dict], recipe_roles: List[str],
                    rank: Dict[int, int]) -> Tuple[List[Dict], int]:
    """逐条关系判定 → (逐条证据, 满足条数)。未知/坏关系一律不计满足。"""
    report: List[Dict] = []
    hits = 0
    for k, rel in enumerate(rels):
        if not isinstance(rel, dict):
            report.append({"index": k, "raw": repr(rel), "satisfied": False,
                           "reason": "关系条目非字典"})
            continue
        f_role, t_role, name = rel.get("from"), rel.get("to"), rel.get("rel")
        item: Dict = {"index": k, "from": f_role, "to": t_role, "rel": name}
        bad_shape = (not isinstance(f_role, str) or not isinstance(t_role, str)
                     or not isinstance(name, str))
        if bad_shape:
            item.update({"satisfied": False,
                         "reason": "缺 from/to/rel 或字段非字符串"})
            report.append(item)
            continue
        if name not in _KNOWN_RELATIONS:
            item.update({"satisfied": False,
                         "reason": f"未知关系名 {name!r}（不判为满足）"})
            report.append(item)
            continue
        missing = [r for r in (f_role, t_role) if r not in recipe_roles]
        if missing:
            item.update({"satisfied": False,
                         "reason": f"角色不在配方 connections.role 里：{missing}"})
            report.append(item)
            continue
        f_dets, t_dets = roles.get(f_role, []), roles.get(t_role, [])
        if not f_dets or not t_dets:
            item.update({"satisfied": False, "reason": "角色未绑定到任何检测",
                         "bound": {f_role: f_dets, t_role: t_dets}})
            report.append(item)
            continue
        witness, observed = _relation_witness(name, f_role, t_role, f_dets, t_dets,
                                              detections, rank)
        item["observed"] = observed
        if witness is None:
            reason = ("已绑定检测之间的实际关系与声明不符" if observed else
                      "无成对检测：该关系需要两个不同的已绑定检测")
            item.update({"satisfied": False, "reason": reason})
        else:
            item.update({"satisfied": True, "witness": witness})
            hits += 1
        report.append(item)
    return report, hits


def _score_record(rec: Dict, detections: List[Dict]) -> Tuple[float, Optional[Dict]]:
    """(评分, 关系报告)。报告 None ⇒ 该配方没有 relations（评分 = 旧口径）。

    #121：分母含**未被本次匹配消费的检测**（`unmatched`）——旧口径只算 recall
    （命中/配方需求），多余部件对分数**零影响**，于是「2 命中 + 1 多余」与
    「恰好 2 命中」同分（实测均 1.0，逐位相同）。补上 precision 侧后，
    score=1.0 当且仅当「配方需求全命中**且无多余检测**」。
    """
    conns = rec["connections"]
    rels = _relations_declared(rec)
    if not detections:
        hits, assign = 0, [-1] * len(conns)
    else:
        hits, assign = _match_binding(rec, detections)
    # 匹配是单射：被消费的检测数 == 命中的 conn 数；其余为多余检测（precision 侧）
    unmatched = max(0, len(detections) - hits) if detections else 0
    if not rels:
        if not conns:
            return 0.0, None
        total = len(conns) + unmatched
        return ((hits / total) if detections and total else 0.0), None
    recipe_roles = _recipe_roles(rec)
    # 规范序（只取检测内容）→ 角色排列/关系证据的枚举与去重顺序，与输入排列无关
    rank = {di: k for k, di in enumerate(_canonical_det_order(detections))}
    roles = _bind_roles(rec, assign, rank)
    rel_report, rel_hits = _eval_relations(rels, roles, detections, recipe_roles, rank)
    total = len(conns) + len(rels) + unmatched
    score = (hits + rel_hits) / total if total else 0.0
    report = {
        "roles": {role: [detections[i]["pos"] for i in roles.get(role, [])]
                  for role in recipe_roles},
        # 角色 → **原输入检测下标**（assign 自始即原下标；规范序只用于挑选）
        "role_detections": {role: list(roles.get(role, []))
                            for role in recipe_roles},
        "relations": rel_report,
        "conn_hits": hits, "conn_total": len(conns),
        "relation_hits": rel_hits, "relation_total": len(rels),
    }
    return score, report


def save_report(report: Dict, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=str)
    return path
