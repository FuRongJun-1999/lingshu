# -*- coding: utf-8 -*-
"""hex_unit · 单元化语义头(荣神经元模型:连接声明+感知偏差+有界增益)
====================================================================================
荣 2026-09-07:「大脑 800 亿神经元自身参数就几个——识别连接,感知偏差,
    传递信号。语义在固定的连接网络组合和结构里。」

两阶段:
  阶段一(结构发现,已由 base 网+剪枝完成):宽松训练 → 每卡 |w| top-k
    连接声明(方向=符号)——每条连接获得语义身份「该卡的第 N 证据源」;
  阶段二(结构固化,本模块):冻结连接结构,只调每卡两个有界属性——
    感知偏差 θ(汇聚信号的响应阈值)与传递增益 g(0≤g≤2)。
    参数 18 个(9 卡×2),纯 numpy 有限差分即可(秒级)。
白箱:每条连接可直读("条纹卡 ← 特征7,方向+,g=0.8,θ=0.2")。
"""
from __future__ import annotations
import math
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .hex_cnn import extended_prior_family, hex_conv, image_to_grid

# #197：参数向量「读-改-写」的原子化锁。
# get_vec/set_vec 是**两次独立调用**，各自内部加锁并不能让裸用的 RMW 成为原子操作
# （两线程可在 get 与 set 之间交错）。故提供 apply_delta() 把整段收进同一临界区；
# 本锁为**模块级**（同模块内跨实例串行化），代价是不同实例之间也会互斥。
_VEC_RMW_LOCK = threading.Lock()

ALGO = "hex_unit-0.1"


class UnitHeadNet:
    """冻结 L1/L1.5 前端(base 网)+单元化语义头。"""

    GAIN_BOUND = 2.0

    def __init__(self, base_net, top_k: int = 4):
        self.base = base_net
        head = base_net.head.copy()
        n_card, n_feat = head.shape
        self.n_card, self.n_feat = n_card, n_feat
        # 阶段一产物:连接声明(0/1)+方向符号(±1)
        self.mask = np.zeros_like(head)
        self.dir = np.zeros_like(head)
        for c in range(n_card):
            idx = np.argsort(-np.abs(head[c]))[:top_k]
            for j in idx:
                self.mask[c, j] = 1.0
                self.dir[c, j] = 1.0 if head[c, j] > 0 else -1.0
        # 阶段二可学习属性:每条声明连接一个增益(简单关系量,有界;
        # 初始=阶段一幅度按卡归一保留——幅度差异是连接语义的一部分),
        # 每卡一个感知偏差 θ
        self.gain = np.zeros_like(head)
        for c in range(n_card):
            idx = np.where(self.mask[c] > 0)[0]
            if len(idx):
                m = np.abs(head[c, idx]).max() + 1e-9
                self.gain[c, idx] = np.abs(head[c, idx]) / m * 1.5  # 幅度，方向只由 dir 承担
        self.theta = np.zeros(n_card, dtype=np.float64)

    # ---- 参数视图(有限差分统一索引:声明连接增益 ++ theta) ----
    def get_vec(self) -> np.ndarray:
        """取参数向量**副本**（UnitHeadNet）。

        ⚠ #197：本方法与 `set_vec` 是**两次独立调用**，「get → 改 → set」这一范式
        在多线程下**非原子** —— 两线程可各自取到同一快照、各自改、后写者覆盖先写者，
        静默丢一次更新（不抛异常）。单线程下无影响。
        并发场景请改用 `apply_delta()`（RMW 在同一临界区内完成）。
        """
        return np.concatenate([self.gain[self.mask > 0],
                               self.theta])

    THETA_BOUND = 2.0   # #2168：与 GAIN_BOUND 对称——类 docstring 声明「两个
    # 有界属性」，修前 theta 一个闸都没有（set_vec(1e6) → theta=1e6、
    # set_vec(nan) → theta=nan 原样写入）

    def set_vec(self, v: np.ndarray):
        # #2520：长度与有限性闸。此前长度不符时静默改形状（短 3 ⇒ theta
        # 变 (6,)，错误延迟到下次前向的 broadcast 报错）；`np.clip(nan)` 仍是
        # NaN（clip 不做有限性检查）⇒ #2168 的「有界」声明对 NaN 不成立。
        v = np.asarray(v, dtype=np.float64)
        k = int(self.mask.sum())
        _want = k + self.n_card
        if v.shape != (_want,):
            raise ValueError("UnitHeadNet.set_vec: 需 shape=(%d,)，实得 %r"
                             % (_want, v.shape))
        if not np.isfinite(v).all():
            raise ValueError("UnitHeadNet.set_vec: 参数含 NaN/inf")
        self.gain[self.mask > 0] = np.clip(v[:k], 0.0, self.GAIN_BOUND)
        # #2168：theta 与 gain 同口径 clip（修前原样写入，无任何边界）
        self.theta = np.clip(np.asarray(v[k:], dtype=np.float64),
                             -self.THETA_BOUND, self.THETA_BOUND).copy()  # #557: 复制

    def apply_delta(self, delta):
        """原子读改写（#197）：在同一临界区内完成 get_vec → 应用 → set_vec。

        `delta` 两种形态：
          · 可调用对象 `fn(vec) -> ndarray | None`：收到当前向量的副本，
            返回新向量；返回 None 表示已原地修改传入的副本。
          · ndarray：与当前向量逐元素相加的增量（形状须一致）。

        并发下多次 `apply_delta` 不会互相覆盖 —— RMW 整段被 `_VEC_RMW_LOCK` 串行化。
        ⚠ 本方法只保证「apply_delta 之间」原子：`get_vec`/`set_vec` 的**成对裸用**
          仍然非原子（两次调用之间可被交错）。并发场景请改用本方法，
          或由调用方自行串行化。返回写入后的向量副本。
        """
        with _VEC_RMW_LOCK:
            vec = self.get_vec()
            if callable(delta):
                out = delta(vec)
                new = vec if out is None else np.asarray(out)
            else:
                _d = np.asarray(delta)
                # #2520：形状必须一致——`(1,)` 增量此前被广播到全部参数
                # （375 个全 +1），调用方无法从返回值看出发生了广播。
                if _d.shape != vec.shape:
                    raise ValueError("apply_delta: delta 需 shape=%r，实得 %r"
                                     % (vec.shape, _d.shape))
                new = vec + _d
            self.set_vec(new)
            return self.get_vec()

    def n_params(self) -> int:
        return int(self.mask.sum()) + self.n_card

    # ---- 前端(冻结)+ 单元头 ----
    def features(self, lat: np.ndarray) -> np.ndarray:
        sf, cf = self.base.l2_features(lat)
        return np.concatenate([sf, cf], axis=1)          # (B, n_feat)

    def logits(self, feat: np.ndarray) -> np.ndarray:
        # 汇聚(每连接:方向×自己的增益)→ 感知偏差 θ
        agg = (self.dir * self.gain) @ feat.T            # (n_card, B)
        return (agg - self.theta[:, None]).T

    def loss(self, lat: np.ndarray, obj_idx: np.ndarray) -> float:
        feat = self.features(lat)
        logits = self.logits(feat)
        z = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        return float(-np.log(p[np.arange(len(obj_idx)), obj_idx] + 1e-12).mean())

    def accuracy(self, lat: np.ndarray, obj_idx: np.ndarray) -> float:
        feat = self.features(lat)
        pred = self.logits(feat).argmax(axis=1)
        return float((pred == obj_idx).mean())

    # ---- 兼容接口(recursive_search/eval_composite 直接可用) ----
    def l1(self, lat):
        return self.base.l1(lat)

    def l2_features(self, lat):
        return self.base.l2_features(lat)

    def l3_logits(self, shape_feat, colorfeat):
        feat = np.concatenate([shape_feat, colorfeat], axis=1)
        return self.logits(feat)

    def describe(self, card: int, feat_names: Optional[List[str]] = None) -> List[str]:
        """白箱直读:某卡的全部连接声明(语义身份)。"""
        lines = []
        for j in range(self.n_feat):
            if self.mask[card, j]:
                name = feat_names[j] if feat_names else f"feat{j}"
                lines.append(f"{name} 方向{'+' if self.dir[card, j] > 0 else '-'} "
                             f"增益={self.gain[card, j]:.2f} θ={self.theta[card]:.2f}")
        return lines


def finetune_units(unit: UnitHeadNet, crops: np.ndarray, obj_idx: np.ndarray,
                   steps: int = 80, lr: float = 0.1, eps: float = 1e-3,
                   batch: int = 96, seed: int = 7,
                   feat_cache: Optional[np.ndarray] = None) -> List[float]:
    """阶段二:只调 g/θ(有限差分符号更新)。
    前端冻结→特征缓存一次(feat_cache),训练循环只算单元头(微秒级)——
    否则每次差分都是 Python 全网前向(M4.3i 超时的根因)。"""
    rng = np.random.default_rng(seed)
    feat = feat_cache if feat_cache is not None else unit.features(crops)
    curve = []
    for step in range(steps):
        idx = rng.permutation(len(feat))[:batch]
        xf, yb = feat[idx], obj_idx[idx]
        vec = unit.get_vec()

        def loss(v: np.ndarray) -> float:
            unit.set_vec(v)
            logits = unit.logits(xf)
            z = logits - logits.max(axis=1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(axis=1, keepdims=True)
            return float(-np.log(p[np.arange(len(yb)), yb] + 1e-12).mean())

        d = loss(vec)
        curve.append(d)
        for pi in range(len(vec)):
            vp, vm = vec.copy(), vec.copy()
            vp[pi] += eps; vm[pi] -= eps
            dp = loss(vp); dm = loss(vm)
            g = (dp - dm) / (2 * eps)
            if abs(g) > 1e-6:
                vec[pi] -= lr * np.sign(g)
        unit.set_vec(vec)
    return curve


# ============================================================================
# M5 汉字模态：内置零依赖 IDS 拆解 + 矢量字形渲染 → 蜂窝晶格 HexConv →
# 独体部件条件卡与六书零样本合体字组合判定（「休=人+木」原生实证）
# ============================================================================

IDS_OPERATORS: Tuple[str, ...] = ("⿰", "⿱")
HANZI_VERDICTS: Tuple[str, ...] = ("ACCEPT", "DEFER", "BLINDSPOT", "REJECT")

#: 19 个核心独体部件的归一化矢量笔画表（[0,1]×[0,1] 坐标系，每条笔画为 (x0, y0, x1, y1)）
RADICAL_STROKES: Dict[str, Tuple[Tuple[float, float, float, float], ...]] = {
    "人": ((0.52, 0.08, 0.14, 0.90), (0.46, 0.42, 0.88, 0.90)),
    "木": (
        (0.12, 0.34, 0.88, 0.34),
        (0.50, 0.08, 0.50, 0.92),
        (0.50, 0.36, 0.14, 0.86),
        (0.50, 0.36, 0.86, 0.86),
    ),
    "口": (
        (0.18, 0.18, 0.82, 0.18),
        (0.82, 0.18, 0.82, 0.84),
        (0.82, 0.84, 0.18, 0.84),
        (0.18, 0.84, 0.18, 0.18),
    ),
    "日": (
        (0.22, 0.12, 0.78, 0.12),
        (0.78, 0.12, 0.78, 0.88),
        (0.78, 0.88, 0.22, 0.88),
        (0.22, 0.88, 0.22, 0.12),
        (0.22, 0.50, 0.78, 0.50),
    ),
    "目": (
        (0.24, 0.10, 0.76, 0.10),
        (0.76, 0.10, 0.76, 0.90),
        (0.76, 0.90, 0.24, 0.90),
        (0.24, 0.90, 0.24, 0.10),
        (0.24, 0.37, 0.76, 0.37),
        (0.24, 0.63, 0.76, 0.63),
    ),
    "月": (
        (0.28, 0.10, 0.28, 0.62),
        (0.28, 0.62, 0.16, 0.90),
        (0.28, 0.10, 0.76, 0.10),
        (0.76, 0.10, 0.76, 0.90),
        (0.76, 0.90, 0.62, 0.82),
        (0.28, 0.38, 0.76, 0.38),
        (0.28, 0.62, 0.76, 0.62),
    ),
    "火": (
        (0.20, 0.26, 0.28, 0.48),
        (0.82, 0.24, 0.68, 0.44),
        (0.50, 0.10, 0.46, 0.56),
        (0.46, 0.56, 0.14, 0.90),
        (0.48, 0.52, 0.86, 0.90),
    ),
    "土": (
        (0.22, 0.44, 0.78, 0.44),
        (0.50, 0.10, 0.50, 0.88),
        (0.10, 0.88, 0.90, 0.88),
    ),
    "十": (
        (0.10, 0.48, 0.90, 0.48),
        (0.50, 0.10, 0.50, 0.90),
    ),
    "田": (
        (0.14, 0.14, 0.86, 0.14),
        (0.86, 0.14, 0.86, 0.86),
        (0.86, 0.86, 0.14, 0.86),
        (0.14, 0.86, 0.14, 0.14),
        (0.50, 0.14, 0.50, 0.86),
        (0.14, 0.50, 0.86, 0.50),
    ),
    "女": (
        (0.44, 0.10, 0.26, 0.58),
        (0.26, 0.58, 0.76, 0.84),
        (0.72, 0.36, 0.20, 0.88),
        (0.10, 0.42, 0.90, 0.38),
    ),
    "子": (
        (0.20, 0.14, 0.78, 0.14),
        (0.78, 0.14, 0.50, 0.36),
        (0.50, 0.36, 0.54, 0.88),
        (0.54, 0.88, 0.38, 0.80),
        (0.12, 0.52, 0.88, 0.52),
    ),
    "山": (
        (0.50, 0.10, 0.50, 0.86),
        (0.18, 0.42, 0.18, 0.86),
        (0.18, 0.86, 0.82, 0.86),
        (0.82, 0.42, 0.82, 0.86),
    ),
    "大": (
        (0.12, 0.38, 0.88, 0.38),
        (0.50, 0.10, 0.46, 0.50),
        (0.46, 0.50, 0.14, 0.90),
        (0.50, 0.46, 0.86, 0.90),
    ),
    "力": (
        (0.18, 0.34, 0.82, 0.34),
        (0.82, 0.34, 0.74, 0.88),
        (0.74, 0.88, 0.58, 0.78),
        (0.50, 0.10, 0.16, 0.88),
    ),
    "心": (
        (0.16, 0.48, 0.12, 0.68),
        (0.28, 0.38, 0.34, 0.82),
        (0.34, 0.82, 0.82, 0.82),
        (0.82, 0.82, 0.86, 0.64),
        (0.48, 0.24, 0.58, 0.42),
        (0.80, 0.36, 0.90, 0.54),
    ),
    "二": (
        (0.22, 0.28, 0.78, 0.28),
        (0.10, 0.76, 0.90, 0.76),
    ),
    "工": (
        (0.16, 0.20, 0.84, 0.20),
        (0.50, 0.20, 0.50, 0.84),
        (0.12, 0.84, 0.88, 0.84),
    ),
    "王": (
        (0.18, 0.16, 0.82, 0.16),
        (0.22, 0.50, 0.78, 0.50),
        (0.12, 0.86, 0.88, 0.86),
        (0.50, 0.16, 0.50, 0.86),
    ),
}

RADICAL_VOCAB: Tuple[str, ...] = tuple(RADICAL_STROKES.keys())

#: 34 个双部件合体字内置 IDS 词典（覆盖左右 ⿰ 与上下 ⿱ 两类标准结构）
HANZI_IDS_DICT: Dict[str, str] = {
    # 左右结构 ⿰ (20)
    "休": "⿰人木",
    "明": "⿰日月",
    "林": "⿰木木",
    "好": "⿰女子",
    "相": "⿰木目",
    "仁": "⿰人二",
    "吐": "⿰口土",
    "叶": "⿰口十",
    "仙": "⿰人山",
    "伙": "⿰人火",
    "加": "⿰力口",
    "从": "⿰人人",
    "朋": "⿰月月",
    "杜": "⿰木土",
    "旺": "⿰日王",
    "杠": "⿰木工",
    "肚": "⿰月土",
    "肋": "⿰月力",
    "功": "⿰工力",
    "如": "⿰女口",
    # 上下结构 ⿱ (14)
    "昌": "⿱日日",
    "炎": "⿱火火",
    "吕": "⿱口口",
    "圭": "⿱土土",
    "呆": "⿱口木",
    "杏": "⿱木口",
    "早": "⿱日十",
    "古": "⿱十口",
    "男": "⿱田力",
    "思": "⿱田心",
    "李": "⿱木子",
    "态": "⿱大心",
    "全": "⿱人王",
    "仝": "⿱人工",
}

IDS_TRIPLE_TO_HANZI: Dict[Tuple[str, str, str], str] = {}


def _m5_require_int(value: Any, name: str, *, min_value: Optional[int] = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("%s 必须是 int（拒绝 bool/非整数），收到 %r" % (name, value))
    iv = int(value)
    if min_value is not None and iv < min_value:
        raise ValueError("%s 必须 >= %d，收到 %r" % (name, min_value, value))
    return iv


def _m5_require_finite(
    value: Any,
    name: str,
    *,
    min_value: Optional[float] = None,
    min_exclusive: Optional[float] = None,
    max_value: Optional[float] = None,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError("%s 必须是有限数值（拒绝 bool/非数值），收到 %r" % (name, value))
    fv = float(value)
    if not math.isfinite(fv):
        raise ValueError("%s 必须是有限数值（拒绝 NaN/inf），收到 %r" % (name, value))
    if min_value is not None and fv < min_value:
        raise ValueError("%s 必须 >= %r，收到 %r" % (name, min_value, value))
    if min_exclusive is not None and fv <= min_exclusive:
        raise ValueError("%s 必须 > %r，收到 %r" % (name, min_exclusive, value))
    if max_value is not None and fv > max_value:
        raise ValueError("%s 必须 <= %r，收到 %r" % (name, max_value, value))
    return fv


def _m5_require_pair(pair: Any, name: str) -> Tuple[float, float]:
    if isinstance(pair, (str, bytes, dict)):
        raise ValueError("%s 必须是二元组 (dx, dy)，收到 %r" % (name, pair))
    try:
        items = list(pair)
    except TypeError as exc:
        raise ValueError("%s 必须是二元组 (dx, dy)：%s" % (name, exc)) from exc
    if len(items) != 2:
        raise ValueError("%s 必须含 2 个分量，收到 %r" % (name, pair))
    return (
        _m5_require_finite(items[0], "%s[0]" % name),
        _m5_require_finite(items[1], "%s[1]" % name),
    )


def parse_ids(expr: str) -> Tuple[str, str, str]:
    """解析标准双部件 IDS 表达式（如 `'⿰人木'` -> `('⿰', '人', '木')`）。"""
    if not isinstance(expr, str) or len(expr) != 3:
        raise ValueError("IDS 表达式必须是 3 字符字符串（如 '⿰人木'），收到 %r" % (expr,))
    op, left_or_top, right_or_bottom = expr[0], expr[1], expr[2]
    if op not in IDS_OPERATORS:
        raise ValueError("不支持的 IDS 结构符 %r（支持 %r）" % (op, IDS_OPERATORS))
    if left_or_top not in RADICAL_STROKES:
        raise ValueError("未知部件 %r（可用：%r）" % (left_or_top, RADICAL_VOCAB))
    if right_or_bottom not in RADICAL_STROKES:
        raise ValueError("未知部件 %r（可用：%r）" % (right_or_bottom, RADICAL_VOCAB))
    return (op, left_or_top, right_or_bottom)


for _hz, _expr in HANZI_IDS_DICT.items():
    IDS_TRIPLE_TO_HANZI[parse_ids(_expr)] = _hz


def _draw_segment(
    canvas: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    thickness: float,
) -> None:
    """纯 numpy 确定性抗锯齿线段光栅化（把笔画写入 [0,1] 灰度画布）。"""
    h, w = canvas.shape
    radius = max(0.6, float(thickness) * 0.5)
    min_x = max(0, int(math.floor(min(x0, x1) - radius - 1.0)))
    max_x = min(w - 1, int(math.ceil(max(x0, x1) + radius + 1.0)))
    min_y = max(0, int(math.floor(min(y0, y1) - radius - 1.0)))
    max_y = min(h - 1, int(math.ceil(max(y0, y1) + radius + 1.0)))
    if min_x > max_x or min_y > max_y:
        return
    yy, xx = np.mgrid[min_y:max_y + 1, min_x:max_x + 1]
    vx = x1 - x0
    vy = y1 - y0
    seg_len2 = vx * vx + vy * vy
    if seg_len2 <= 1e-12:
        dist = np.hypot(xx - x0, yy - y0)
    else:
        t = ((xx - x0) * vx + (yy - y0) * vy) / seg_len2
        t = np.clip(t, 0.0, 1.0)
        proj_x = x0 + t * vx
        proj_y = y0 + t * vy
        dist = np.hypot(xx - proj_x, yy - proj_y)
    ink = np.clip(1.0 - np.maximum(0.0, dist - radius * 0.6) / max(0.6, radius * 0.6), 0.0, 1.0)
    patch = canvas[min_y:max_y + 1, min_x:max_x + 1]
    canvas[min_y:max_y + 1, min_x:max_x + 1] = np.maximum(patch, ink.astype(np.float32))


def _render_strokes_into_box(
    canvas: np.ndarray,
    strokes: Sequence[Tuple[float, float, float, float]],
    box: Tuple[float, float, float, float],
    stroke_width: float,
    shift_px: Tuple[float, float],
    scale: float,
) -> None:
    bx0, by0, bx1, by1 = box
    bw = bx1 - bx0
    bh = by1 - by0
    cx = bx0 + bw * 0.5 + shift_px[0]
    cy = by0 + bh * 0.5 + shift_px[1]
    for sx0, sy0, sx1, sy1 in strokes:
        px0 = cx + (sx0 - 0.5) * bw * scale
        py0 = cy + (sy0 - 0.5) * bh * scale
        px1 = cx + (sx1 - 0.5) * bw * scale
        py1 = cy + (sy1 - 0.5) * bh * scale
        _draw_segment(canvas, px0, py0, px1, py1, stroke_width)


def render_radical(
    radical: str,
    size: int = 48,
    *,
    stroke_width: float = 2.4,
    shift: Tuple[float, float] = (0.0, 0.0),
    scale: float = 1.0,
) -> np.ndarray:
    """无字体依赖确定性渲染单个独体部件 → `(size, size, 3)` uint8 图像（白字黑底）。"""
    if not isinstance(radical, str) or radical not in RADICAL_STROKES:
        raise ValueError("未知独体部件 %r（可用：%r）" % (radical, RADICAL_VOCAB))
    sz = _m5_require_int(size, "size", min_value=16)
    sw = _m5_require_finite(stroke_width, "stroke_width", min_exclusive=0.0)
    sh = _m5_require_pair(shift, "shift")
    sc = _m5_require_finite(scale, "scale", min_exclusive=0.0, max_value=2.0)

    gray = np.zeros((sz, sz), dtype=np.float32)
    margin = sz * 0.08
    _render_strokes_into_box(
        gray,
        RADICAL_STROKES[radical],
        (margin, margin, sz - margin, sz - margin),
        sw,
        sh,
        sc,
    )
    rgb = np.clip(np.round(gray * 255.0), 0, 255).astype(np.uint8)
    return np.stack([rgb, rgb, rgb], axis=-1)


def render_ids(
    expr: str,
    size: int = 64,
    *,
    stroke_width: float = 2.2,
    shift: Tuple[float, float] = (0.0, 0.0),
    scale: float = 1.0,
) -> np.ndarray:
    """按 IDS 结构（`⿰` 左右 / `⿱` 上下）零依赖渲染双部件合体字 → `(size, size, 3)` uint8。"""
    op, rad_a, rad_b = parse_ids(expr)
    sz = _m5_require_int(size, "size", min_value=16)
    sw = _m5_require_finite(stroke_width, "stroke_width", min_exclusive=0.0)
    sh = _m5_require_pair(shift, "shift")
    sc = _m5_require_finite(scale, "scale", min_exclusive=0.0, max_value=2.0)

    gray = np.zeros((sz, sz), dtype=np.float32)
    if op == "⿰":
        box_a = (sz * 0.04, sz * 0.08, sz * 0.46, sz * 0.92)
        box_b = (sz * 0.54, sz * 0.08, sz * 0.96, sz * 0.92)
    else:
        box_a = (sz * 0.08, sz * 0.04, sz * 0.92, sz * 0.46)
        box_b = (sz * 0.08, sz * 0.54, sz * 0.92, sz * 0.96)
    _render_strokes_into_box(gray, RADICAL_STROKES[rad_a], box_a, sw, sh, sc)
    _render_strokes_into_box(gray, RADICAL_STROKES[rad_b], box_b, sw, sh, sc)
    rgb = np.clip(np.round(gray * 255.0), 0, 255).astype(np.uint8)
    return np.stack([rgb, rgb, rgb], axis=-1)


def render_hanzi(
    hanzi: str,
    size: int = 64,
    *,
    stroke_width: float = 2.2,
    shift: Tuple[float, float] = (0.0, 0.0),
    scale: float = 1.0,
) -> np.ndarray:
    """渲染内置 IDS 词典中的合体汉字 → `(size, size, 3)` uint8。"""
    if not isinstance(hanzi, str) or hanzi not in HANZI_IDS_DICT:
        raise ValueError("未知汉字 %r（未在内置 IDS 词典中）" % (hanzi,))
    return render_ids(
        HANZI_IDS_DICT[hanzi],
        size=size,
        stroke_width=stroke_width,
        shift=shift,
        scale=scale,
    )


def _validate_hanzi_image(img: Any, name: str = "img") -> np.ndarray:
    if img is None or isinstance(img, (str, bytes, dict)):
        raise ValueError("%s 必须是图像数组，收到 %r" % (name, type(img).__name__))
    arr = np.asarray(img)
    if arr.dtype == np.bool_:
        raise ValueError("%s 拒绝 bool 数组" % (name,))
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError("%s 必须是数值图像数组，收到 dtype=%r" % (name, arr.dtype))
    arr_f = arr.astype(np.float32, copy=False)
    if not np.isfinite(arr_f).all():
        raise ValueError("%s 含 NaN/inf 非有限像素值" % (name,))
    if arr_f.ndim == 2:
        h, w = arr_f.shape
        rgb = np.stack([arr_f, arr_f, arr_f], axis=-1)
    elif arr_f.ndim == 3 and arr_f.shape[2] in (1, 3):
        h, w = arr_f.shape[:2]
        if arr_f.shape[2] == 1:
            g = arr_f[:, :, 0]
            rgb = np.stack([g, g, g], axis=-1)
        else:
            rgb = arr_f
    else:
        raise ValueError("%s 维度须为 (H,W) 或 (H,W,1|3)，收到 shape=%s" % (name, tuple(arr_f.shape)))
    if h < 12 or w < 12:
        raise ValueError("%s 尺寸至少 12×12，收到 %dx%d" % (name, h, w))
    max_val = float(rgb.max())
    if max_val <= 1.0 and float(rgb.min()) >= 0.0:
        rgb = rgb * 255.0
    return np.clip(rgb, 0.0, 255.0).astype(np.float32)


def _pool_subgrid_6x6(sub_feat: np.ndarray, ink_sub: np.ndarray, active_thr: float) -> Tuple[Optional[np.ndarray], float]:
    """在槽位有效墨迹包围框内池化为 6×6×C + 6×6 墨迹密度归一化特征（尺度/平移不变）。"""
    ink_energy = float(ink_sub.mean())
    mask = ink_sub > active_thr
    if not mask.any():
        return None, ink_energy
    ys, xs = np.nonzero(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop_f = sub_feat[y0:y1, x0:x1]
    crop_i = ink_sub[y0:y1, x0:x1]
    ch, cw, n_ch = crop_f.shape
    bins = 6
    pooled_f = np.zeros((bins, bins, n_ch), dtype=np.float64)
    pooled_i = np.zeros((bins, bins), dtype=np.float64)
    for bi in range(bins):
        ry0 = (bi * ch) // bins
        ry1 = max(ry0 + 1, ((bi + 1) * ch) // bins)
        for bj in range(bins):
            rx0 = (bj * cw) // bins
            rx1 = max(rx0 + 1, ((bj + 1) * cw) // bins)
            pooled_f[bi, bj] = crop_f[ry0:ry1, rx0:rx1].mean(axis=(0, 1))
            pooled_i[bi, bj] = float(crop_i[ry0:ry1, rx0:rx1].mean())
    vec = np.concatenate([pooled_f.reshape(-1), pooled_i.reshape(-1) * 2.0])
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return None, ink_energy
    return vec / norm, ink_energy


def _hex_conv_feature_map(rgb: np.ndarray, cells_across: int, kernels: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """图像 → 蜂窝晶格 `image_to_grid` → 10 通道 `hex_conv` 方向边缘/拉普拉斯/中心拮抗特征。"""
    grid_rgb, _ = image_to_grid(rgb, cells_across=cells_across)
    lum = grid_rgb.mean(axis=-1, keepdims=True) / 255.0
    chans = [np.abs(hex_conv(lum, k)[:, :, 0]) for k in kernels]
    feat = np.stack(chans, axis=-1).astype(np.float32)
    return feat, lum[:, :, 0]


class HanziRadicalCardBank:
    """独体部件蜂窝条件卡字典（只在独体部件上校准，从未在任何合体字上训练）。"""

    def __init__(
        self,
        radicals: Optional[Sequence[str]] = None,
        *,
        cells_across: int = 32,
        p_threshold: float = 0.65,
        margin_threshold: float = 0.02,
        defer_floor: float = 0.40,
        active_threshold: float = 0.03,
        calib_size: int = 64,
    ) -> None:
        self.cells_across = _m5_require_int(cells_across, "cells_across", min_value=8)
        self.p_threshold = _m5_require_finite(
            p_threshold, "p_threshold", min_exclusive=0.0, max_value=1.0
        )
        self.margin_threshold = _m5_require_finite(
            margin_threshold, "margin_threshold", min_value=0.0, max_value=1.0
        )
        self.defer_floor = _m5_require_finite(
            defer_floor, "defer_floor", min_value=0.0, max_value=self.p_threshold
        )
        self.active_threshold = _m5_require_finite(
            active_threshold, "active_threshold", min_exclusive=0.0, max_value=1.0
        )
        self.calib_size = _m5_require_int(calib_size, "calib_size", min_value=16)

        if radicals is None:
            vocab = list(RADICAL_VOCAB)
        else:
            vocab = []
            for r in radicals:
                if not isinstance(r, str) or r not in RADICAL_STROKES:
                    raise ValueError("未知部件 %r" % (r,))
                if r not in vocab:
                    vocab.append(r)
            if not vocab:
                raise ValueError("radicals 不能为空")

        self.radicals: Tuple[str, ...] = tuple(vocab)
        self.kernels: List[np.ndarray] = extended_prior_family(10)
        self.cards: Dict[str, np.ndarray] = {}
        self.slot_cards: Dict[str, Dict[str, np.ndarray]] = {"full": {}, "⿰": {}, "⿱": {}}
        self.trained_on_compounds = False
        self._calibrate_single_radicals()

    def _calibrate_single_radicals(self) -> None:
        sz = self.calib_size
        slot_configs = {
            "full": [((sz * 0.08, sz * 0.08, sz * 0.92, sz * 0.92), "full")],
            "⿰": [
                ((sz * 0.04, sz * 0.08, sz * 0.46, sz * 0.92), "first"),
                ((sz * 0.54, sz * 0.08, sz * 0.96, sz * 0.92), "second"),
            ],
            "⿱": [
                ((sz * 0.08, sz * 0.04, sz * 0.92, sz * 0.46), "first"),
                ((sz * 0.08, sz * 0.54, sz * 0.92, sz * 0.96), "second"),
            ],
        }
        for slot_kind, configs in slot_configs.items():
            for rad in self.radicals:
                vecs = []
                for box, which in configs:
                    for sw in (2.0, 2.5):
                        gray = np.zeros((sz, sz), dtype=np.float32)
                        _render_strokes_into_box(
                            gray, RADICAL_STROKES[rad], box, sw, (0.0, 0.0), 1.0
                        )
                        rgb_u8 = np.clip(np.round(gray * 255.0), 0, 255).astype(np.uint8)
                        img = np.stack([rgb_u8, rgb_u8, rgb_u8], axis=-1)
                        feat, lum = _hex_conv_feature_map(
                            img.astype(np.float32), self.cells_across, self.kernels
                        )
                        if slot_kind == "⿰":
                            mid = feat.shape[1] // 2
                            if which == "first":
                                feat, lum = feat[:, :mid, :], lum[:, :mid]
                            else:
                                feat, lum = feat[:, mid:, :], lum[:, mid:]
                        elif slot_kind == "⿱":
                            mid = feat.shape[0] // 2
                            if which == "first":
                                feat, lum = feat[:mid, :, :], lum[:mid, :]
                            else:
                                feat, lum = feat[mid:, :, :], lum[mid:, :]
                        v, _ = _pool_subgrid_6x6(feat, lum, self.active_threshold)
                        if v is not None:
                            vecs.append(v)
                proto = np.mean(vecs, axis=0)
                proto = proto / max(1e-12, float(np.linalg.norm(proto)))
                self.slot_cards[slot_kind][rad] = proto
        self.cards = dict(self.slot_cards["full"])

    def match_slot(
        self,
        slot_feat: np.ndarray,
        slot_lum: np.ndarray,
        *,
        slot_kind: str = "full",
        p_threshold: Optional[float] = None,
        margin_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """用独体部件条件卡（p_threshold / margin）对单个结构槽位给出四态判定。"""
        if slot_kind not in self.slot_cards:
            raise ValueError("未知槽位形态 %r" % (slot_kind,))
        p_thr = (
            self.p_threshold
            if p_threshold is None
            else _m5_require_finite(p_threshold, "p_threshold", min_exclusive=0.0, max_value=1.0)
        )
        m_thr = (
            self.margin_threshold
            if margin_threshold is None
            else _m5_require_finite(margin_threshold, "margin_threshold", min_value=0.0, max_value=1.0)
        )
        vec, energy = _pool_subgrid_6x6(slot_feat, slot_lum, self.active_threshold)
        if vec is None or energy < 0.01:
            return {
                "radical": None,
                "score": 0.0,
                "second_radical": None,
                "second_score": 0.0,
                "margin": 0.0,
                "ink_energy": round(energy, 5),
                "verdict": "BLINDSPOT",
                "reason": "empty_or_low_energy_slot",
            }
        bank = self.slot_cards[slot_kind]
        scores = {rad: float(np.dot(vec, proto)) for rad, proto in bank.items()}
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_r, best_s = ranked[0]
        second_r, second_s = ranked[1] if len(ranked) > 1 else (None, 0.0)
        margin = best_s - second_s

        if best_s >= p_thr and margin >= m_thr:
            verdict = "ACCEPT"
            reason = "card_threshold_and_margin_satisfied"
        elif best_s >= self.defer_floor:
            verdict = "DEFER"
            reason = "insufficient_confidence_or_margin"
        else:
            verdict = "BLINDSPOT"
            reason = "out_of_radical_vocabulary"
        return {
            "radical": best_r,
            "score": round(best_s, 5),
            "second_radical": second_r,
            "second_score": round(second_s, 5),
            "margin": round(margin, 5),
            "ink_energy": round(energy, 5),
            "verdict": verdict,
            "reason": reason,
        }


def _evaluate_split(
    feat: np.ndarray,
    lum: np.ndarray,
    op: str,
    bank: HanziRadicalCardBank,
    *,
    p_threshold: Optional[float] = None,
    margin_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    rows, cols, _ = feat.shape
    if op == "⿰":
        mid = cols // 2
        s1_f, s1_l = feat[:, :mid, :], lum[:, :mid]
        s2_f, s2_l = feat[:, mid:, :], lum[:, mid:]
        bridge = float(lum[:, max(0, mid - 1):min(cols, mid + 1)].mean())
    else:
        mid = rows // 2
        s1_f, s1_l = feat[:mid, :, :], lum[:mid, :]
        s2_f, s2_l = feat[mid:, :, :], lum[mid:, :]
        bridge = float(lum[max(0, mid - 1):min(rows, mid + 1), :].mean())

    m1 = bank.match_slot(
        s1_f, s1_l, slot_kind=op, p_threshold=p_threshold, margin_threshold=margin_threshold
    )
    m2 = bank.match_slot(
        s2_f, s2_l, slot_kind=op, p_threshold=p_threshold, margin_threshold=margin_threshold
    )
    quality = (m1["score"] + m2["score"]) * 0.5 - bridge * 0.35
    return {
        "op": op,
        "slot1": m1,
        "slot2": m2,
        "center_bridge": round(bridge, 5),
        "split_quality": round(quality, 5),
    }


def recognize_hanzi(
    img: Any,
    bank: Optional[HanziRadicalCardBank] = None,
    *,
    op: Optional[str] = None,
    declared_hanzi: Optional[str] = None,
    p_threshold: Optional[float] = None,
    margin_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """六书部件条件组合卡零样本汉字判定（支持自动 ⿰/⿱ 拆槽与四态诚实判决）。"""
    rgb = _validate_hanzi_image(img, "img")
    card_bank = bank if bank is not None else HanziRadicalCardBank()
    if not isinstance(card_bank, HanziRadicalCardBank):
        raise ValueError("bank 必须是 HanziRadicalCardBank 实例，收到 %r" % (bank,))
    if op is not None and (not isinstance(op, str) or op not in IDS_OPERATORS):
        raise ValueError("op 必须是 %r 之一，收到 %r" % (IDS_OPERATORS, op))
    if declared_hanzi is not None and (
        not isinstance(declared_hanzi, str) or declared_hanzi not in HANZI_IDS_DICT
    ):
        raise ValueError("declared_hanzi %r 不在内置 IDS 词典中" % (declared_hanzi,))

    feat, lum = _hex_conv_feature_map(rgb, card_bank.cells_across, card_bank.kernels)
    candidates = (
        [_evaluate_split(feat, lum, op, card_bank, p_threshold=p_threshold, margin_threshold=margin_threshold)]
        if op is not None
        else [
            _evaluate_split(feat, lum, cand_op, card_bank, p_threshold=p_threshold, margin_threshold=margin_threshold)
            for cand_op in IDS_OPERATORS
        ]
    )
    best = max(candidates, key=lambda c: c["split_quality"])
    chosen_op = best["op"]
    slot1, slot2 = best["slot1"], best["slot2"]
    r1, r2 = slot1["radical"], slot2["radical"]
    triple = (chosen_op, r1, r2) if (r1 is not None and r2 is not None) else None
    matched_hanzi = IDS_TRIPLE_TO_HANZI.get(triple) if triple is not None else None
    ids_expr = (chosen_op + r1 + r2) if (r1 is not None and r2 is not None) else None

    if slot1["verdict"] == "BLINDSPOT" or slot2["verdict"] == "BLINDSPOT":
        verdict = "BLINDSPOT"
        reason = "slot_blindspot"
    elif slot1["verdict"] == "DEFER" or slot2["verdict"] == "DEFER":
        if declared_hanzi is not None and matched_hanzi is not None and matched_hanzi != declared_hanzi:
            verdict = "REJECT"
            reason = "declared_hanzi_conflict"
        else:
            verdict = "DEFER"
            reason = "slot_confidence_or_margin_insufficient"
    elif matched_hanzi is None:
        if declared_hanzi is not None:
            verdict = "REJECT"
            reason = "declared_hanzi_conflict_unregistered_composition"
        else:
            verdict = "BLINDSPOT"
            reason = "composition_not_in_ids_dictionary"
    elif declared_hanzi is not None and matched_hanzi != declared_hanzi:
        verdict = "REJECT"
        reason = "declared_hanzi_conflict"
    else:
        verdict = "ACCEPT"
        reason = "ids_composition_matched"

    return {
        "verdict": verdict,
        "reason": reason,
        "hanzi": matched_hanzi,
        "ids": ids_expr,
        "op": chosen_op,
        "radicals": (r1, r2),
        "slot1": slot1,
        "slot2": slot2,
        "split_quality": best["split_quality"],
        "trained_on_compounds": card_bank.trained_on_compounds,
    }


def evaluate_hanzi_m5(
    bank: Optional[HanziRadicalCardBank] = None,
    *,
    characters: Optional[Sequence[str]] = None,
    size: int = 64,
    stroke_width: float = 2.2,
    shift: Tuple[float, float] = (0.0, 0.0),
    scale: float = 1.0,
) -> Dict[str, Any]:
    """M5 批量零样本双部件汉字组合判定评测（返回部件判定率与整字判定率）。"""
    card_bank = bank if bank is not None else HanziRadicalCardBank()
    if not isinstance(card_bank, HanziRadicalCardBank):
        raise ValueError("bank 必须是 HanziRadicalCardBank 实例")
    chars = list(HANZI_IDS_DICT.keys()) if characters is None else list(characters)
    if not chars:
        raise ValueError("characters 不能为空")

    radical_hits = 0
    radical_total = 0
    char_hits = 0
    accept_count = 0
    details = []
    for ch in chars:
        exp_op, exp_a, exp_b = parse_ids(HANZI_IDS_DICT[ch])
        img = render_hanzi(ch, size=size, stroke_width=stroke_width, shift=shift, scale=scale)
        res = recognize_hanzi(img, bank=card_bank)
        r1, r2 = res["radicals"]
        radical_total += 2
        if r1 == exp_a:
            radical_hits += 1
        if r2 == exp_b:
            radical_hits += 1
        if res["verdict"] == "ACCEPT":
            accept_count += 1
        if res["verdict"] == "ACCEPT" and res["hanzi"] == ch and res["op"] == exp_op:
            char_hits += 1
        details.append(
            {
                "expected": ch,
                "expected_ids": HANZI_IDS_DICT[ch],
                "predicted": res["hanzi"],
                "predicted_ids": res["ids"],
                "verdict": res["verdict"],
                "slot1_score": res["slot1"]["score"],
                "slot2_score": res["slot2"]["score"],
            }
        )

    n = len(chars)
    return {
        "total_chars": n,
        "radical_hits": radical_hits,
        "radical_total": radical_total,
        "radical_accuracy": round(radical_hits / float(radical_total), 4),
        "char_hits": char_hits,
        "char_accuracy": round(char_hits / float(n), 4),
        "accept_count": accept_count,
        "trained_on_compounds": card_bank.trained_on_compounds,
        "details": details,
    }
