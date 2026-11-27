# -*- coding: utf-8 -*-
"""stcnn.py · 白箱自举第三阶段·多模态主线原型 v1——3D 时空尺度 CNN（零 LLM）
理论：《白箱自举·LLM替代与3D时空多模态》（§3 3D时空尺度 = 时间×空间统一卷积）
核心：连续帧 → 时空体素 (T×H×W) → 3D 卷积（核跨时间+空间）→ 时空特征 → 时空原语
  （运动方向/速度/周期/形状变化）→ 时空记忆图写入（看见→记住→回忆 白箱闭环）
零依赖：纯 numpy 确定性实现（D-005 核心零外部依赖哲学）。
"""
import math
import sys
from copy import deepcopy

import numpy as np


# ============ 一、时空体素化（连续帧 → 3D 体素） ============
def frames_to_voxel(frames, width=None, height=None):
    """连续帧（灰度/彩色）→ 时空体素 (T,H,W)。
    帧序列 = 时空体在时间轴的切片（2D 是 3D 透视下的情况——灵枢原则）"""
    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim == 4:  # (T,H,W,C) → 灰度
        arr = arr.mean(axis=-1)
    if width is not None or height is not None:
        h, w = arr.shape[1], arr.shape[2]
        arr = arr[:, : (height or h), : (width or w)]
    # 归一化 0~1。#352：旧实现按**整段** min/max 归一 ⇒ 任意小噪声被拉到满量程
    # （静止画面 + σ=1e-6 噪声的 motion_magnitude=236.9 > 真实滚动小球 90.0——
    # 运动检测与真实运动幅度无关）。现锚定**绝对量纲**：输入在 [0,1] 或 [0,255]
    # （帧来自 uint8 图像/本仓合成器，量纲有定义）时按已知量纲缩放（已 0..1 则
    # 原样、0..255 则 /255）——微小噪声保持微小；仅当输入明显超出两个已知域
    # （自造浮点域、无先验量纲）才退回整段 min-max（该分支仍带 #352 界，如实注明）。
    amin, amax = float(arr.min()), float(arr.max())
    # F23-18（#942 清单）如实标注（审计面，判据不变）：三分支判定是**零容差
    # 精确比较**——本应落在 [0,1] 但带 -1e-9 浮点误差的输入会翻转进 min-max
    # 分支（被重归一）。这不是「明显超出」的模糊判据；输入域由上游（uint8
    # 图像/本仓合成器）保证时无影响。是否加容差属行为面裁定，不在本标注内。
    if 0.0 <= amin and amax <= 1.0:
        pass                                   # 已是 0..1 量纲
    elif 0.0 <= amin and 0.0 < amax <= 255.0:
        arr = arr / 255.0                      # uint8 图像量纲
    elif amax > amin:
        arr = (arr - amin) / (amax - amin)     # 自造浮点域：退回 min-max（带 #352 界）
    return arr


# ============ 二、3D 卷积（时间+空间统一卷积核） ============
def _require_finite(arr, name):
    """入口闸（#788）：输入含 NaN/inf ⇒ fail-closed（raise ValueError）。

    改前 conv3d/max_pool3d 对输入体素的有限性**零校验**——含 NaN 的体素
    静默产出含 NaN 的输出（numpy 静默传播），NaN 一旦进入卷积核 → 特征图
    全 NaN → 下游训练梯度/loss 全 NaN 且**无崩溃**，回溯定位困难。现入口
    `np.isfinite` 校验，非有限时抛 ValueError，消息含输入 shape 与非有限
    位置计数（NaN/inf 分列），帮助定位上游泄漏源（如 physics.predict_at
    对极端 k 值的 e^{-k·t} 下溢/上溢到 NaN）。
    同仓先例：world/physics.py `_check_samples`（#662，位置/时间有限性）、
    core/core.py `_finite_clamp`（#574，importance 值域）——本件堵时空体素输入面。"""
    a = np.asarray(arr)
    bad = ~np.isfinite(a)
    if bad.any():
        raise ValueError(
            "%s含非有限值（NaN/inf）：shape=%s，非有限位置数=%d/%d"
            "（nan=%d，inf=%d）——fail-closed 拒绝静默传播（#788）"
            % (name, tuple(a.shape), int(bad.sum()), a.size,
               int(np.isnan(a).sum()), int(np.isinf(a).sum())))


def conv3d(voxel, kernel, stride=(1, 1, 1)):
    """3D 卷积：核 (kt, kh, kw) 同时跨时间+空间滑窗 → 时空特征图。
    核可自定义为时空滤波器（如运动检测核 = 时间差分×空间平滑）。

    输入闸（#788）：voxel 含 NaN/inf 时 raise ValueError（消息含 shape 与
    非有限位置计数，见 _require_finite）——不再静默产出含 NaN 的特征图。

    边界口径（#598）：输入任一维小于核（含 stride 放大后 ot<0 的情形）时，
    输出维截断到 0——返回空张量 (0,oh,ow) 等，**不抛异常**；与同文件
    max_pool3d 的 T//kt 非负截断同口径（下游拿到空特征图可照常 continue）。
    此前 ot=(T-kt)//st+1 对 T<kt 因 Python // 地板取整得负数，
    np.zeros 抛 ValueError: negative dimensions。"""
    _require_finite(voxel, "conv3d 输入体素")
    T, H, W = voxel.shape
    kt, kh, kw = kernel.shape
    st, sh, sw = stride
    ot = max(0, (T - kt) // st + 1)
    oh = max(0, (H - kh) // sh + 1)
    ow = max(0, (W - kw) // sw + 1)
    out = np.zeros((ot, oh, ow), dtype=np.float32)
    for t in range(ot):
        for i in range(oh):
            for j in range(ow):
                patch = voxel[t * st:t * st + kt, i * sh:i * sh + kh, j * sw:j * sw + kw]
                out[t, i, j] = float((patch * kernel).sum())
    return out


def max_pool3d(feat, size=(2, 2, 2)):
    """3D 最大池化（降采样时空特征图）。

    输入闸（#788）：feat 含 NaN/inf 时 raise ValueError（消息含 shape 与
    非有限位置计数，见 _require_finite）——与 conv3d 同口径，不再静默
    传播 NaN（max 对 NaN 的行为是污染整个池化窗，且不可回溯）。"""
    _require_finite(feat, "max_pool3d 输入特征图")
    T, H, W = feat.shape
    kt, kh, kw = size
    ot, oh, ow = T // kt, H // kh, W // kw
    out = np.zeros((ot, oh, ow), dtype=np.float32)
    for t in range(ot):
        for i in range(oh):
            for j in range(ow):
                out[t, i, j] = feat[t * kt:t * kt + kt,
                                    i * kh:i * kh + kh,
                                    j * kw:j * kw + kw].max()
    return out


# 时空滤波器核
# 运动核：时间差分（t+1 - t）× 空间平均 → 响应=该处亮度随时间变化（运动）
MOTION_KERNEL = np.zeros((2, 3, 3), dtype=np.float32)
MOTION_KERNEL[1] = 1.0 / 9.0   # 后帧平均
MOTION_KERNEL[0] = -1.0 / 9.0  # 前帧平均（差分 → 变化检测）
# 空间平滑核：单时间点空间平均（形状基线）
SPATIAL_KERNEL = np.zeros((1, 3, 3), dtype=np.float32)
SPATIAL_KERNEL[0] = 1.0 / 9.0

# #352 重判（残余）：`moving` 的**强度**阈值。
# 旧判据 `motion_mag > 0.5` 把阈值加在 motion_mag 上，而 motion_mag = |conv3d|.sum()
# 是**广延量**（∝ 输出元素数 (T-1)(H-2)(W-2)），于是同一噪声在不同帧尺寸下判定不同：
# 实测 σ=0.01 在 8×8 上 motion_mag=0.76（刚越阈）、在 128×128 上 296.2；而 σ=1e-4
# 这种 0.086% 满量程的纯噪声，在 64×64/128×128 上也越 0.5 被判为 moving=True。
# 现改为**强度量**（每元素平均响应），并与本模块对「运动」的既有定义同量纲：
# frame_diff 的逐像素阈值 0.08 ÷ MOTION_KERNEL 的 3×3 空间平均面积 9 = 0.0089。
# 依据：真运动的每元素响应恰为 1.0/9 = 0.1111（满对比度边沿经 3×3 平均），
# σ=0.01 噪声为 0.0038 —— 阈值 0.0089 对噪声留 2.3× 余量、对真运动留 12.5× 余量，
# 且跨尺寸稳定（8..128 上噪声 0.0037–0.0042、真运动 0.0159–0.1111）。
_MOVING_INTENSITY = 0.08 / 9.0


# ============ 三、时空原语提取（运动/方向/速度/周期） ============
def frame_diff(frames, threshold=0.08):
    """帧间差分 → 运动区域掩码序列（|f(t+1)-f(t)| > 阈值）。

    注意（#76）：这是**绝对值**差分，「暗→亮」与「亮→暗」各出一个沿——单个亮灭
    周期因此产生**双沿**。掩码用于运动区域/质心是正确口径，但**不能**把它当作周期
    信号喂给 detect_period（双沿 ⇒ 读出的周期是真实周期的一半）；周期信号须用
    `frame_rise`（每个亮灭周期只保留亮起前沿）。
    """
    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim == 4:                        # #256：(T,H,W,C) → 灰度
        arr = arr.mean(axis=-1)              #   与 frames_to_voxel:20-21 同口径
    diffs = np.abs(np.diff(arr, axis=0))
    return (diffs > threshold).astype(np.float32)


def frame_rise(frames, threshold=0.08):
    """帧间**正向**亮度变化 → 亮起前沿信号（每个亮灭周期恰一个沿）。

    #76：`frame_diff` 取 |Δ|，对「暗→亮」与「亮→暗」各出一个沿 ⇒ 单个亮灭周期双沿，
    把它交给 detect_period 读出的周期是真实周期的**一半**。周期检测只应看一个方向
    的沿，这里只保留正向变化（亮起前沿）；`detect_period` 的前沿检测正是按此设计的。
    """
    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim == 4:                        # #256：(T,H,W,C) → 灰度
        arr = arr.mean(axis=-1)              #   与 frames_to_voxel:20-21 同口径
    rise = np.clip(np.diff(arr, axis=0), 0.0, None)
    rise[rise <= threshold] = 0.0
    return rise


def motion_direction(centroids):
    """运动方向：质心位移 → 主方向（右/左/下/上/静止）"""
    if len(centroids) < 2:
        return "静止"
    dx = centroids[-1][0] - centroids[0][0]
    dy = centroids[-1][1] - centroids[0][1]
    total = abs(dx) + abs(dy)
    if total < 1.0:
        return "静止"
    if abs(dx) >= abs(dy):
        return "向右" if dx > 0 else "向左"
    return "向下" if dy > 0 else "向上"


def motion_speed(centroids, frames_per_unit=1.0):
    """速度：总位移 / 总时间（帧数）"""
    if len(centroids) < 2:
        return 0.0
    total_disp = 0.0
    for i in range(1, len(centroids)):
        total_disp += abs(centroids[i][0] - centroids[i - 1][0]) \
            + abs(centroids[i][1] - centroids[i - 1][1])
    return total_disp / (len(centroids) - 1) / frames_per_unit


def detect_period(signal, min_period=2):
    """周期检测：运动/亮度信号 → 周期（帧），无周期返回 None。
    用「前沿」检测（信号从低→高的跳变位置，间隔即周期）——
    平台型信号（1,1 相邻）无严格局部极大，局部极大法会漏检。

    #57（本件）：`min_period` 此前在函数体内**零引用**（声明参数不兑现）——
    实测 `min_period=2/3/5/99` 对同一信号都返回同一周期。现让它参与判定：
    读出的候选周期 `period < min_period` 时返回 `None`（视作「低于最小可辨
    周期 ⇒ 无有效周期」），与既有 `if count < 2: return None` 的返回 `None`
    语义一致（该分支同样以 `None` 表「无周期」）。默认 `min_period=2` 下，
    既有回归读数不变（最小真周期本就 ≥2）。
    判据来源：公开 issue #57（声明参数零引用）＋ 本函数既有 `None`=无周期口径。
    边界（削掉的判别力）：只做下界过滤，**不**校验 `min_period` 自身合法性
    （`min_period<=0` 等价于不过滤）；同型面全仓共 29 处，本件只修有实害的 3 处。
    """
    sig = np.asarray(signal, dtype=np.float32)
    edges = []
    for i in range(1, len(sig)):
        if sig[i] > 0.05 and sig[i - 1] <= 0.05:
            edges.append(i)  # 前沿：低→高
    if len(edges) < 2:
        return None
    intervals = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]
    from collections import Counter
    period, count = Counter(intervals).most_common(1)[0]
    if count < 2:
        return None  # 无重复间隔 → 非周期（此前 count=1 且间隔多时误判为周期）
    if period < min_period:
        return None  # #57：低于声明的最小可辨周期 ⇒ 无有效周期
    return period


def extract_spatiotemporal_primitives(frames):
    """时空原语提取：运动/方向/速度/周期（白箱确定性）

    #798：入口 fail-closed 校验——空帧列表 / 少于 2 帧时 raise ValueError（领域消息），
    不再把 numpy 内部错误（``zero-size array to reduction operation minimum which
    has no identity``，出自 ``frames_to_voxel`` 对空数组取 min/max 归一化）漏给调用方。
    最小帧数取 **2**（判据）：本函数的全部时序算子——``MOTION_KERNEL``（kt=2）、
    ``frame_diff``/``frame_rise`` 的 ``np.diff(axis=0)``、``detect_period`` 的前沿
    间隔——都以「相邻两帧」为最小可分辨单元；单帧的时间维为 1，conv3d 输出空特征
    图、diff/rise 全空 ⇒ 一切时空原语都是「未测量」却伪装成「测得 0」（静止/无周期），
    与其静默返回伪读数，不如 fail-closed（与 physics._check_samples 的
    「样本不足：需要 >= N 个」同款口径，#662 先例）。"""
    n_frames = len(frames) if hasattr(frames, "__len__") else None
    if n_frames is None or n_frames < 2:
        raise ValueError(
            "extract_spatiotemporal_primitives：帧列表为空/不足（至少需要 2 帧，"
            "得到 %s 帧）——时空原语的时序算子（运动核/帧差/前沿检测）以相邻两帧"
            "为最小可分辨单元" % ("?" if n_frames is None else n_frames)
        )
    voxel = frames_to_voxel(frames)
    # ① 3D 卷积：运动核 → 时空特征图（运动响应）
    motion_feat = conv3d(voxel, MOTION_KERNEL)
    motion_mag = np.abs(motion_feat).sum()
    # ② 帧差运动区域 + 质心轨迹
    diffs = frame_diff(frames)
    centroids = []
    motion_frames = []          # #304：有运动的帧差下标（diffs[t] 对应帧 t→t+1）
    for t in range(diffs.shape[0]):
        region = diffs[t]
        ys, xs = np.nonzero(region)
        if len(xs) > 2:
            motion_frames.append(t)
            centroids.append((float(xs.mean()), float(ys.mean())))
        elif centroids:
            centroids.append(centroids[-1])
    # ③ 运动信号（每帧**亮起前沿**）→ 周期检测
    #    #76：此处原用 frame_diff 的 |Δ| 掩码，单个亮灭周期有亮/灭两个沿 ⇒ 周期减半。
    #    周期信号改用 frame_rise（只保留暗→亮的前沿），每个周期恰一个沿。
    motion_signal = [float(region.sum()) for region in frame_rise(frames)]
    period = detect_period(motion_signal)
    # ④ 原语汇总。#304：`t_first` / `t_last` = 片段内运动实际发生的帧区间
    # （闭区间；diffs[t] 对应帧 t→t+1，判定口径与质心轨迹一致：>2 个变化像素）。
    # diff 下标 t 覆盖帧 [t, t+1] ⇒ 有运动的 diff 下标 [i..j] 对应帧区间
    # [i, j+1]（迟动球实测 diffs 39..42 ⇒ 帧 39..43——球在 40..43，diff 在 39
    # 也触发是因为 39→40 的边界变化属于运动起点的过渡帧）。
    # 无任何运动帧 ⇒ 取整段 [0, n_frames-1]（报告口径：无运动时区间无信息，
    # 记整段不算谎）。`remember` 用它们把 clip 首帧绝对时刻换算成真实事件区间。
    n_frames = len(frames)
    if motion_frames:
        t_first, t_last = motion_frames[0], motion_frames[-1] + 1
    else:
        t_first, t_last = 0, max(0, n_frames - 1)
    prims = {
        "motion_magnitude": round(float(motion_mag), 4),
        "direction": motion_direction(centroids),
        "speed": round(motion_speed(centroids), 4),
        "period": period,
        "moving": bool(motion_mag / motion_feat.size > _MOVING_INTENSITY) if motion_feat.size else False,
        "trajectory_len": len(centroids),
        "t_first": t_first,
        "t_last": t_last,
    }
    return prims, motion_feat


# ============ 四、时空记忆图写入（看见→记住） ============
# 时空锚点必备字段（remember 的产物契约；缺失即视为损坏/幻觉事件）
_EVENT_FIELDS = ("t_start", "t_end", "direction", "speed", "period", "moving", "label")


class SpatiotemporalMemory:
    """时空记忆图：事件 → 时空锚点（时间/位置/运动模式）+ 记忆边。
    零 LLM 确定性记忆（对接灵枢语义时空图的感知侧入口）。"""

    def __init__(self):
        """时空卷积网络层级参数初始化。"""
        self.events = []  # {t_start, t_end, direction, speed, period, label}
        self._written = []  # Independent snapshots; mutable recall aliases are not a baseline.

    def remember(self, prims, label, t_start=0):
        """把一次「看见」的时空原语写为记忆事件（时空锚点）。

        #304：`t_start` 视为**片段首帧**的绝对时刻；事件区间 = 片段内运动实际
        发生的帧区间（原语 `t_first`/`t_last`，闭区间）平移到绝对时刻 ⇒
        事件 [t_start+t_first, t_start+t_last]（如「迟动球」clip 帧 200..243、
        运动发生在 240..243 ⇒ 事件 [240, 243]，不再误记 [200, 202]）。
        原语没有这两个键（手工构造的旧口径原语）⇒ 保持原公式
        `t_start + trajectory_len - 1` 不变（向后兼容）。
        """
        if "t_first" in prims and "t_last" in prims:
            ev_start = t_start + prims["t_first"]
            ev_end = t_start + prims["t_last"]
        else:
            ev_start = t_start
            ev_end = t_start + max(1, prims.get("trajectory_len", 1) - 1)
        event = {
            "t_start": ev_start,
            "t_end": max(ev_start + 1, ev_end),
            "direction": prims.get("direction", "静止"),
            "speed": prims.get("speed", 0.0),
            "period": prims.get("period"),
            "moving": prims.get("moving", False),
            "label": label,
        }
        written = deepcopy(event)  # Copy before either list is published.
        self.events.append(event)
        self._written.append(written)
        return event

    #: recall 可查询的字段白名单＝事件契约字段面（_EVENT_FIELDS 全集）。
    #: #162（本件）：旧白名单只有四键（direction/moving/period/label）——
    #  `speed`/`t_start`/`t_end` 在事件里存了值却**不可查询**（静默忽略＝
    #  假全量）。白名单对齐事件契约全集后，查询键与召回键一一对应。
    _QUERY_KEYS = frozenset(_EVENT_FIELDS)

    def recall(self, query=None):
        """回忆：按条件查询时空记忆（白箱召回，无 LLM）。

        #162（本件）：可查询键白名单对齐 `_EVENT_FIELDS` 全集——旧白名单
        漏掉 `t_start`/`t_end`/`speed` 等事件契约字段，按这些键查询被静默
        忽略（返回全量＝假命中）。
        """
        if query is not None:
            unknown = set(query) - self._QUERY_KEYS
            if unknown:
                raise ValueError(
                    "recall：查询键 %r 不在可查询字段 %s 内——旧实现静默忽略"
                    "（返回全量＝假命中）；如需按新键查询请先扩展事件契约"
                    % (sorted(unknown), sorted(self._QUERY_KEYS)))
        out = []
        for e in self.events:
            if query is None:
                out.append(e)
            else:
                hit = all(e.get(k) == v for k, v in query.items())
                if hit:
                    out.append(e)
        return out
    def verify_consistency(self):
        """自校验：回忆 vs 已记事件一致性（记忆无幻觉）。

        旧实现 `len(self.recall()) == len(self.events)` 恒真——recall() 无查询时
        就是把 self.events 逐条 append 出来，两侧长度是同一个数的两次读数，
        任何「记了却回忆不出 / 回忆出没记过的」都发现不了（#302）。

        新判据（按「看见→记住→回忆」白箱闭环的可观测环节逐项核对）：
          ① 全量回忆与已记事件的**条数与顺序**一致（recall(None) 的召回完整性）；
          ② 每条已记事件具备时空锚点必备字段（_EVENT_FIELDS）；
          ③ 每条已记事件都能被 recall 以该事件自身字段查询**命中**（召回可达性）；
          ④ 每条回忆出的事件确实是 self.events 中的同一对象（无凭空多出的幻觉事件）。
        ⑤ 事件与 remember 时的独立深快照逐条相同（#302 残余）：改写、丢失、
          换序或完整伪事件不能通过自检。快照是进程内基准，不是持久审计/安全防篡改。
        任一条不成立即判不一致（返回 False），不再恒真。
        """
        if self.events != self._written:
            return False
        recalled = self.recall()
        if len(recalled) != len(self.events):
            return False
        if any(a is not b for a, b in zip(recalled, self.events)):
            return False
        for e in self.events:
            if not all(k in e for k in _EVENT_FIELDS):
                return False
            hits = self.recall(dict(e))
            if not any(h is e for h in hits):
                return False
        return True


# ============ 五、合成时空场景（演示数据，无真实视频） ============
def synth_ball_rolling(frames=10, size=32, start=3, speed_px=2, period=None):
    """合成：小球水平匀速滚动（行固定中间，列随帧递增 → 向右）。
    可选周期闪烁——球周期性可见（时隐时现）。"""
    frames_list = []
    row = size // 2  # 球行固定（中间）
    for t in range(frames):
        frame = np.zeros((size, size), dtype=np.float32)
        if period is None or (t % period) < period // 2:  # 周期性可见
            col = min(start + speed_px * t, size - 4)
            frame[row:row + 3, col:col + 3] = 1.0
        frames_list.append(frame)
    return frames_list


def synth_static(frames=6, size=32):
    """合成：静止场景（无运动基线）"""
    return [np.ones((size, size), dtype=np.float32) * 0.5 for _ in range(frames)]


def synth_blinking(frames=12, size=32, period=3):
    """合成：周期性闪烁（全屏亮/暗交替）"""
    out = []
    for t in range(frames):
        frame = np.zeros((size, size), dtype=np.float32)
        if (t % period) < period // 2:
            frame[:] = 1.0
        out.append(frame)
    return out


# ============ 五.B、M7 时序运动轨迹提取、未来位置预测与 4D 时空体素桥接 ============
MOTION_MODES = ("static", "linear", "bounce_or_turn", "blindspot")


def _m7_require_int(value, name, *, min_value=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("%s 必须是 int（拒绝 bool/非整数），收到 %r" % (name, value))
    iv = int(value)
    if min_value is not None and iv < min_value:
        raise ValueError("%s 必须 >= %d，收到 %r" % (name, min_value, value))
    return iv


def _m7_require_finite(value, name, *, min_value=None, min_exclusive=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError("%s 必须是有限数值（拒绝 bool/非数值），收到 %r" % (name, value))
    fv = float(value)
    if not math.isfinite(fv):
        raise ValueError("%s 必须是有限数值（拒绝 NaN/inf），收到 %r" % (name, value))
    if min_value is not None and fv < min_value:
        raise ValueError("%s 必须 >= %r，收到 %r" % (name, min_value, value))
    if min_exclusive is not None and fv <= min_exclusive:
        raise ValueError("%s 必须 > %r，收到 %r" % (name, min_exclusive, value))
    return fv


def _m7_validate_frames(frames, name, *, min_frames=2):
    if frames is None or isinstance(frames, (str, bytes, dict)):
        raise ValueError("%s 必须是帧序列 (T, H, W[, C])，收到 %r" % (name, type(frames).__name__))
    try:
        n = len(frames)
    except TypeError as exc:
        raise ValueError("%s 必须是可求长度的帧序列：%s" % (name, exc)) from exc
    if n < min_frames:
        raise ValueError("%s 帧数不足（至少需要 %d 帧，得到 %d 帧）" % (name, min_frames, n))
    arr = np.asarray(frames)
    if arr.dtype == np.bool_:
        raise ValueError("%s 拒绝 bool 帧数组（须数值像素）" % (name,))
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError("%s 须为数值像素数组，收到 dtype=%r" % (name, arr.dtype))
    arr = arr.astype(np.float32, copy=False)
    _require_finite(arr, name)
    if arr.ndim not in (3, 4):
        raise ValueError("%s 维度须为 3 (T,H,W) 或 4 (T,H,W,C)，收到 shape=%s" % (name, tuple(arr.shape)))
    if arr.shape[1] < 2 or arr.shape[2] < 2:
        raise ValueError("%s 空间尺寸至少 2×2，收到 shape=%s" % (name, tuple(arr.shape)))
    if arr.ndim == 4 and arr.shape[3] < 1:
        raise ValueError("%s 通道维不能为空，收到 shape=%s" % (name, tuple(arr.shape)))
    return frames_to_voxel(arr)


def _largest_fg_component(delta_frame, thr):
    """在单帧前景残差图上选出总能量最高的 8-连通前景簇，抗孤立背景噪点。"""
    mask = delta_frame > thr
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None, None
    h, w = delta_frame.shape
    visited = np.zeros((h, w), dtype=bool)
    best_energy = -1.0
    best_pts = None
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if visited[sy, sx]:
            continue
        stack = [(sy, sx)]
        visited[sy, sx] = True
        comp = []
        energy = 0.0
        while stack:
            cy, cx = stack.pop()
            comp.append((cy, cx))
            energy += float(delta_frame[cy, cx])
            for dy in (-1, 0, 1):
                ny = cy + dy
                if ny < 0 or ny >= h:
                    continue
                for dx in (-1, 0, 1):
                    nx = cx + dx
                    if nx < 0 or nx >= w or visited[ny, nx] or not mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if energy > best_energy:
            best_energy = energy
            best_pts = comp
    if not best_pts:
        return None, None
    total_w = 0.0
    sum_x = 0.0
    sum_y = 0.0
    x_min = w
    x_max = -1
    y_min = h
    y_max = -1
    for py, px in best_pts:
        wt = float(delta_frame[py, px])
        total_w += wt
        sum_x += px * wt
        sum_y += py * wt
        if px < x_min:
            x_min = px
        if px > x_max:
            x_max = px
        if py < y_min:
            y_min = py
        if py > y_max:
            y_max = py
    if total_w <= 0.0:
        return None, None
    return (sum_x / total_w, sum_y / total_w), (int(x_min), int(y_min), int(x_max), int(y_max))


def _extract_foreground_series(voxel, min_contrast):
    bg_val = float(np.median(voxel))
    fg_delta = np.abs(voxel - bg_val)
    peak = float(fg_delta.max())
    if peak < min_contrast:
        t_len = voxel.shape[0]
        return [None] * t_len, [None] * t_len
    noise_mad = float(np.median(fg_delta))
    thr = max(float(min_contrast), 5.0 * 1.4826 * noise_mad)
    if thr >= peak:
        thr = max(float(min_contrast), 0.5 * peak)
    centroids = []
    bboxes = []
    for t in range(voxel.shape[0]):
        c, bb = _largest_fg_component(fg_delta[t], thr)
        centroids.append(c)
        bboxes.append(bb)
    return centroids, bboxes


def _conv3d_directional_consistency(motion_feat, centroids):
    """计算 3D 时空卷积正/负响应瓣质心向量与前景位移的一致性（[0, 1]）。"""
    if motion_feat.size == 0:
        return 0.0, 0.0
    total_energy = float(np.abs(motion_feat).sum())
    scores = []
    for t in range(motion_feat.shape[0]):
        c0, c1 = centroids[t], centroids[t + 1]
        if c0 is None or c1 is None:
            continue
        dx = c1[0] - c0[0]
        dy = c1[1] - c0[1]
        step_norm = math.hypot(dx, dy)
        if step_norm < 1e-6:
            continue
        mf = motion_feat[t]
        pos_y, pos_x = np.nonzero(mf > 1e-4)
        neg_y, neg_x = np.nonzero(mf < -1e-4)
        if len(pos_x) == 0 or len(neg_x) == 0:
            continue
        pw = mf[pos_y, pos_x]
        nw = np.abs(mf[neg_y, neg_x])
        p_sum = float(pw.sum())
        n_sum = float(nw.sum())
        if p_sum <= 0.0 or n_sum <= 0.0:
            continue
        cx_p = float((pos_x * pw).sum()) / p_sum
        cy_p = float((pos_y * pw).sum()) / p_sum
        cx_n = float((neg_x * nw).sum()) / n_sum
        cy_n = float((neg_y * nw).sum()) / n_sum
        cdx = cx_p - cx_n
        cdy = cy_p - cy_n
        c_norm = math.hypot(cdx, cdy)
        if c_norm < 1e-6:
            continue
        cos_sim = (dx * cdx + dy * cdy) / (step_norm * c_norm)
        scores.append(max(0.0, min(1.0, cos_sim)))
    if not scores:
        return 0.0, total_energy
    return sum(scores) / len(scores), total_energy


def _classify_and_estimate_velocity(voxel, centroids, visible_idx, motion_feat, static_tol):
    """根据逐帧可见质心与 3D 卷积能量判定运动模式并估计当前速度 (vx, vy)。"""
    t_len = voxel.shape[0]
    if len(visible_idx) < 2:
        temporal_diff = float(np.max(np.abs(np.diff(voxel, axis=0))))
        has_content = float(np.max(np.abs(voxel))) > 1e-6
        if len(visible_idx) == 0 and has_content and temporal_diff <= 1e-6:
            return (0.0, 0.0), "static", 1.0
        return (0.0, 0.0), "blindspot", 0.0

    steps = []
    for k in range(len(visible_idx) - 1):
        t0, t1 = visible_idx[k], visible_idx[k + 1]
        dt = float(t1 - t0)
        c0, c1 = centroids[t0], centroids[t1]
        vx = (c1[0] - c0[0]) / dt
        vy = (c1[1] - c0[1]) / dt
        steps.append((t0, t1, vx, vy, math.hypot(vx, vy)))

    max_step_speed = max(s[4] for s in steps)
    c_first = centroids[visible_idx[0]]
    c_last = centroids[visible_idx[-1]]
    net_disp = math.hypot(c_last[0] - c_first[0], c_last[1] - c_first[1])
    mean_intensity = float(np.abs(motion_feat).mean()) if motion_feat.size else 0.0

    if max_step_speed < static_tol and net_disp < 1.0 and mean_intensity <= _MOVING_INTENSITY:
        vis_ratio = len(visible_idx) / float(t_len)
        return (0.0, 0.0), "static", round(min(1.0, vis_ratio), 4)

    moving_steps = [s for s in steps if s[4] >= static_tol]
    if not moving_steps:
        return (0.0, 0.0), "static", round(len(visible_idx) / float(t_len), 4)

    # 检测是否出现明显转向或边界反弹
    turn_idx = -1
    for i in range(1, len(moving_steps)):
        _, _, vx_a, vy_a, sp_a = moving_steps[i - 1]
        _, _, vx_b, vy_b, sp_b = moving_steps[i]
        cos_ab = (vx_a * vx_b + vy_a * vy_b) / (sp_a * sp_b)
        if cos_ab < 0.7:
            turn_idx = i

    if turn_idx >= 0:
        # 反弹或转向：取最近一致直线段估计当前瞬时速度
        seg_steps = moving_steps[turn_idx:]
        t_start = seg_steps[0][0]
        t_end = seg_steps[-1][1]
        vx_est = (centroids[t_end][0] - centroids[t_start][0]) / float(t_end - t_start)
        vy_est = (centroids[t_end][1] - centroids[t_start][1]) / float(t_end - t_start)
        mode = "bounce_or_turn"
        dir_coherence = 0.85
    else:
        t_start = visible_idx[0]
        t_end = visible_idx[-1]
        vx_est = (centroids[t_end][0] - centroids[t_start][0]) / float(t_end - t_start)
        vy_est = (centroids[t_end][1] - centroids[t_start][1]) / float(t_end - t_start)
        mode = "linear"
        est_norm = math.hypot(vx_est, vy_est)
        if est_norm > 1e-9:
            cos_list = [
                max(0.0, (vx * vx_est + vy * vy_est) / (sp * est_norm))
                for _, _, vx, vy, sp in moving_steps
            ]
            dir_coherence = sum(cos_list) / len(cos_list)
        else:
            dir_coherence = 0.0

    vis_ratio = len(visible_idx) / float(t_len)
    conv_cons, _ = _conv3d_directional_consistency(motion_feat, centroids)
    conf = 0.5 * dir_coherence + 0.3 * conv_cons + 0.2 * vis_ratio
    return (round(vx_est, 6), round(vy_est, 6)), mode, round(max(0.0, min(1.0, conf)), 4)


def extract_motion_trajectory(frames, *, min_contrast=0.15, static_tol=0.1):
    """M7：从连续帧 (T, H, W[, C]) 提取前景质心/包围框、3D 卷积能量一致性与速度向量。"""
    mc = _m7_require_finite(min_contrast, "min_contrast", min_exclusive=0.0)
    st = _m7_require_finite(static_tol, "static_tol", min_exclusive=0.0)
    voxel = _m7_validate_frames(frames, "frames", min_frames=2)

    motion_feat = conv3d(voxel, MOTION_KERNEL)
    centroids, bboxes = _extract_foreground_series(voxel, mc)
    visible_idx = [i for i, c in enumerate(centroids) if c is not None]
    velocity, mode, confidence = _classify_and_estimate_velocity(
        voxel, centroids, visible_idx, motion_feat, st
    )
    conv_cons, motion_energy = _conv3d_directional_consistency(motion_feat, centroids)

    last_centroid = centroids[visible_idx[-1]] if visible_idx else None
    last_bbox = bboxes[visible_idx[-1]] if visible_idx else None
    if last_bbox is not None:
        obj_size = (
            float(last_bbox[2] - last_bbox[0] + 1),
            float(last_bbox[3] - last_bbox[1] + 1),
        )
    else:
        obj_size = (0.0, 0.0)

    prims, _ = extract_spatiotemporal_primitives(frames)
    return {
        "frame_shape": tuple(int(v) for v in voxel.shape),
        "centroids": [
            (round(c[0], 4), round(c[1], 4)) if c is not None else None for c in centroids
        ],
        "bboxes": bboxes,
        "visible_indices": visible_idx,
        "velocity": velocity,
        "speed": round(math.hypot(velocity[0], velocity[1]), 6),
        "mode": mode,
        "confidence": confidence,
        "conv_directional_consistency": round(conv_cons, 4),
        "motion_energy": round(motion_energy, 4),
        "last_centroid": (
            (round(last_centroid[0], 4), round(last_centroid[1], 4))
            if last_centroid is not None
            else None
        ),
        "last_bbox": last_bbox,
        "object_size": obj_size,
        "primitives": prims,
    }


def _reflect_1d(val, lo, hi, vel):
    if hi <= lo:
        return lo, 0.0
    pos = val
    cur_v = vel
    while pos < lo or pos > hi:
        if pos < lo:
            pos = lo + (lo - pos)
            cur_v = -cur_v
        elif pos > hi:
            pos = hi - (pos - hi)
            cur_v = -cur_v
    return pos, cur_v


def _bbox_from_center(cx, cy, obj_w, obj_h, w, h):
    hw = max(0.0, (obj_w - 1.0) / 2.0)
    hh = max(0.0, (obj_h - 1.0) / 2.0)
    x0 = int(round(cx - hw))
    y0 = int(round(cy - hh))
    x1 = int(round(cx + hw))
    y1 = int(round(cy + hh))
    x0 = min(max(0, x0), w - 1)
    y0 = min(max(0, y0), h - 1)
    x1 = min(max(x0, x1), w - 1)
    y1 = min(max(y0, y1), h - 1)
    return (x0, y0, x1, y1)


def predict_motion_trajectory(
    frames,
    horizon=5,
    *,
    bounce=False,
    min_contrast=0.15,
    static_tol=0.1,
):
    """M7：基于连续帧轨迹外推未来 horizon 帧平移/反弹物体位置与像素包围框（零未来泄漏）。"""
    h_steps = _m7_require_int(horizon, "horizon", min_value=1)
    if not isinstance(bounce, bool):
        raise ValueError("bounce 必须是 bool，收到 %r" % (bounce,))
    traj = extract_motion_trajectory(frames, min_contrast=min_contrast, static_tol=static_tol)

    _, height, width = traj["frame_shape"]
    last_c = traj["last_centroid"]
    if traj["mode"] == "blindspot" or last_c is None:
        return {
            "horizon": h_steps,
            "bounce": bounce,
            "predictable": False,
            "mode": traj["mode"],
            "velocity": traj["velocity"],
            "confidence": traj["confidence"],
            "trajectory": traj,
            "predicted_positions": [None] * h_steps,
            "predicted_bboxes": [None] * h_steps,
            "naive_baseline_positions": [None] * h_steps,
        }

    obj_w, obj_h = traj["object_size"]
    hw = max(0.0, (obj_w - 1.0) / 2.0)
    hh = max(0.0, (obj_h - 1.0) / 2.0)
    x_lo, x_hi = hw, max(hw, float(width - 1) - hw)
    y_lo, y_hi = hh, max(hh, float(height - 1) - hh)

    cx, cy = float(last_c[0]), float(last_c[1])
    vx, vy = float(traj["velocity"][0]), float(traj["velocity"][1])
    pred_positions = []
    pred_bboxes = []

    for _ in range(h_steps):
        nx = cx + vx
        ny = cy + vy
        if bounce:
            nx, vx = _reflect_1d(nx, x_lo, x_hi, vx)
            ny, vy = _reflect_1d(ny, y_lo, y_hi, vy)
        else:
            if nx < 0.0:
                nx = 0.0
            elif nx > float(width - 1):
                nx = float(width - 1)
            if ny < 0.0:
                ny = 0.0
            elif ny > float(height - 1):
                ny = float(height - 1)
        cx, cy = nx, ny
        pred_positions.append((round(cx, 4), round(cy, 4)))
        pred_bboxes.append(_bbox_from_center(cx, cy, obj_w, obj_h, width, height))

    naive_pos = [(round(last_c[0], 4), round(last_c[1], 4))] * h_steps
    return {
        "horizon": h_steps,
        "bounce": bounce,
        "predictable": True,
        "mode": traj["mode"],
        "velocity": traj["velocity"],
        "confidence": traj["confidence"],
        "trajectory": traj,
        "predicted_positions": pred_positions,
        "predicted_bboxes": pred_bboxes,
        "naive_baseline_positions": naive_pos,
    }


def evaluate_motion_prediction(
    observed_frames,
    future_frames,
    *,
    bounce=False,
    min_contrast=0.15,
    static_tol=0.1,
):
    """M7：独立评测未来帧位置预测误差，并与零速静态基线（naive hold-last）对比。"""
    mc = _m7_require_finite(min_contrast, "min_contrast", min_exclusive=0.0)
    st = _m7_require_finite(static_tol, "static_tol", min_exclusive=0.0)
    obs_voxel = _m7_validate_frames(observed_frames, "observed_frames", min_frames=2)
    fut_voxel = _m7_validate_frames(future_frames, "future_frames", min_frames=1)
    if obs_voxel.shape[1:] != fut_voxel.shape[1:]:
        raise ValueError(
            "observed_frames %s 与 future_frames %s 空间尺寸不一致"
            % (tuple(obs_voxel.shape[1:]), tuple(fut_voxel.shape[1:]))
        )

    horizon = int(fut_voxel.shape[0])
    pred = predict_motion_trajectory(
        observed_frames,
        horizon=horizon,
        bounce=bounce,
        min_contrast=mc,
        static_tol=st,
    )
    fut_centroids, _ = _extract_foreground_series(fut_voxel, mc)

    per_step_errors = []
    naive_errors = []
    for p_pos, n_pos, a_pos in zip(
        pred["predicted_positions"], pred["naive_baseline_positions"], fut_centroids
    ):
        if p_pos is None or a_pos is None:
            per_step_errors.append(None)
        else:
            per_step_errors.append(round(math.hypot(p_pos[0] - a_pos[0], p_pos[1] - a_pos[1]), 6))
        if n_pos is None or a_pos is None:
            naive_errors.append(None)
        else:
            naive_errors.append(round(math.hypot(n_pos[0] - a_pos[0], n_pos[1] - a_pos[1]), 6))

    valid_errs = [e for e in per_step_errors if e is not None]
    valid_naive = [e for e in naive_errors if e is not None]
    mean_err = round(sum(valid_errs) / len(valid_errs), 6) if valid_errs else None
    max_err = round(max(valid_errs), 6) if valid_errs else None
    naive_mean = round(sum(valid_naive) / len(valid_naive), 6) if valid_naive else None
    improvement = (
        round(naive_mean - mean_err, 6)
        if (mean_err is not None and naive_mean is not None)
        else None
    )

    return {
        "horizon": horizon,
        "mode": pred["mode"],
        "velocity": pred["velocity"],
        "confidence": pred["confidence"],
        "predictable": pred["predictable"],
        "predicted_positions": pred["predicted_positions"],
        "actual_positions": [
            (round(c[0], 4), round(c[1], 4)) if c is not None else None for c in fut_centroids
        ],
        "per_step_errors": per_step_errors,
        "mean_pixel_error": mean_err,
        "max_pixel_error": max_err,
        "naive_mean_pixel_error": naive_mean,
        "improvement_over_naive": improvement,
    }


def _pixel_to_voxel_pos(cx, cy, width, height, world_size, y_val):
    sx = 0.5 + (cx / float(max(1, width - 1))) * float(max(1, world_size - 1))
    sz = 0.5 + (cy / float(max(1, height - 1))) * float(max(1, world_size - 1))
    return (round(sx, 4), float(y_val), round(sz, 4))


def sync_motion_to_voxel_world(
    frames,
    voxel_world,
    *,
    horizon=1,
    label="stcnn_motion",
    y_level=None,
    bounce=False,
    min_contrast=0.15,
    static_tol=0.1,
):
    """M7：把观测帧轨迹与未来预测占位同步写入鸭子类型 VoxelWorld 四维 (x, y, z, t) 时空网格。"""
    for req_attr in ("spawn_entity", "move_entity", "occupancy_at", "trail", "simulate"):
        if not callable(getattr(voxel_world, req_attr, None)):
            raise ValueError("voxel_world 缺少必需方法 %s" % req_attr)
    if not isinstance(label, str) or not label.strip():
        raise ValueError("label 必须是非空字符串，收到 %r" % (label,))
    world_size = _m7_require_int(getattr(voxel_world, "size", 0), "voxel_world.size", min_value=2)
    ground_level = _m7_require_finite(getattr(voxel_world, "ground_level", 1), "voxel_world.ground_level")
    y_val = (
        ground_level + 0.5
        if y_level is None
        else _m7_require_finite(y_level, "y_level")
    )

    pred = predict_motion_trajectory(
        frames,
        horizon=horizon,
        bounce=bounce,
        min_contrast=min_contrast,
        static_tol=static_tol,
    )
    traj = pred["trajectory"]
    t_len, height, width = traj["frame_shape"]

    if not pred["predictable"]:
        return {
            "synced": False,
            "mode": pred["mode"],
            "observed_entity_id": None,
            "predicted_entity_id": None,
            "observed_ticks": [],
            "predicted_ticks": [],
            "prediction": pred,
        }

    obs_entity_id = None
    observed_ticks = []
    centroids = traj["centroids"]
    for t in range(t_len):
        c = centroids[t]
        if t == 0:
            cur_tick = int(getattr(voxel_world, "_step", 0))
            if c is not None:
                vpos = _pixel_to_voxel_pos(c[0], c[1], width, height, world_size, y_val)
                obs_entity_id = voxel_world.spawn_entity(label, vpos, velocity=(0.0, 0.0, 0.0))
                observed_ticks.append(cur_tick)
            continue
        if obs_entity_id is None:
            voxel_world.simulate(1)
            cur_tick = int(getattr(voxel_world, "_step", t))
            if c is not None:
                vpos = _pixel_to_voxel_pos(c[0], c[1], width, height, world_size, y_val)
                obs_entity_id = voxel_world.spawn_entity(label, vpos, velocity=(0.0, 0.0, 0.0))
                observed_ticks.append(cur_tick)
        else:
            if c is not None:
                vpos = _pixel_to_voxel_pos(c[0], c[1], width, height, world_size, y_val)
                voxel_world.move_entity(obs_entity_id, vpos, record=False)
            voxel_world.simulate(1)
            cur_tick = int(getattr(voxel_world, "_step", t))
            if c is not None:
                observed_ticks.append(cur_tick)

    last_c = traj["last_centroid"]
    last_vpos = _pixel_to_voxel_pos(last_c[0], last_c[1], width, height, world_size, y_val)
    pred_entity_id = voxel_world.spawn_entity(
        "%s_predicted" % label, last_vpos, velocity=(0.0, 0.0, 0.0)
    )
    predicted_ticks = []
    for p_pos in pred["predicted_positions"]:
        if p_pos is not None:
            pv = _pixel_to_voxel_pos(p_pos[0], p_pos[1], width, height, world_size, y_val)
            voxel_world.move_entity(pred_entity_id, pv, record=False)
        voxel_world.simulate(1)
        cur_tick = int(getattr(voxel_world, "_step", 0))
        if p_pos is not None:
            predicted_ticks.append(cur_tick)

    return {
        "synced": True,
        "mode": pred["mode"],
        "observed_entity_id": obs_entity_id,
        "predicted_entity_id": pred_entity_id,
        "observed_ticks": observed_ticks,
        "predicted_ticks": predicted_ticks,
        "prediction": pred,
    }


def synth_moving_object(
    frames=12,
    size=32,
    start=(3, 14),
    velocity=(2, 0),
    box_size=3,
    *,
    bounce=False,
    noise_std=0.0,
    seed=42,
):
    """M7：确定性合成运动场景（支持任意二维速度、边界反弹与低噪背景）。"""
    n_frames = _m7_require_int(frames, "frames", min_value=2)
    img_size = _m7_require_int(size, "size", min_value=4)
    b_size = _m7_require_int(box_size, "box_size", min_value=1)
    if b_size >= img_size:
        raise ValueError("box_size (%d) 必须小于 size (%d)" % (b_size, img_size))
    if not isinstance(bounce, bool):
        raise ValueError("bounce 必须是 bool，收到 %r" % (bounce,))
    ns = _m7_require_finite(noise_std, "noise_std", min_value=0.0)
    sd = _m7_require_int(seed, "seed", min_value=0)

    if (isinstance(start, (str, bytes, dict))
            or not callable(getattr(start, "__getitem__", None))):
        raise ValueError("start 必须是可下标访问的 (x, y) 二元组，收到 %r" % (start,))
    try:
        start_len = len(start)
    except TypeError as exc:
        raise ValueError("start 必须是可求长度的 (x, y) 二元组") from exc
    if start_len != 2:
        raise ValueError("start 必须是 (x, y) 二元组，收到 %r" % (start,))
    if (isinstance(velocity, (str, bytes, dict))
            or not callable(getattr(velocity, "__getitem__", None))):
        raise ValueError("velocity 必须是可下标访问的 (vx, vy) 二元组，收到 %r" % (velocity,))
    try:
        velocity_len = len(velocity)
    except TypeError as exc:
        raise ValueError("velocity 必须是可求长度的 (vx, vy) 二元组") from exc
    if velocity_len != 2:
        raise ValueError("velocity 必须是 (vx, vy) 二元组，收到 %r" % (velocity,))
    sx = _m7_require_finite(start[0], "start[0]", min_value=0.0)
    sy = _m7_require_finite(start[1], "start[1]", min_value=0.0)
    vx = _m7_require_finite(velocity[0], "velocity[0]")
    vy = _m7_require_finite(velocity[1], "velocity[1]")

    rng = np.random.default_rng(sd) if ns > 0.0 else None
    max_origin = float(img_size - b_size)
    x, y = min(sx, max_origin), min(sy, max_origin)
    out = []
    for _ in range(n_frames):
        frame = np.zeros((img_size, img_size), dtype=np.float32)
        if rng is not None:
            frame = np.clip(frame + rng.normal(0.0, ns, size=(img_size, img_size)).astype(np.float32), 0.0, 1.0)
        ix = int(round(x))
        iy = int(round(y))
        ix = min(max(0, ix), img_size - b_size)
        iy = min(max(0, iy), img_size - b_size)
        frame[iy:iy + b_size, ix:ix + b_size] = 1.0
        out.append(frame)
        nx, ny = x + vx, y + vy
        if bounce:
            nx, vx = _reflect_1d(nx, 0.0, max_origin, vx)
            ny, vy = _reflect_1d(ny, 0.0, max_origin, vy)
        else:
            if nx < 0.0:
                nx = 0.0
            elif nx > max_origin:
                nx = max_origin
            if ny < 0.0:
                ny = 0.0
            elif ny > max_origin:
                ny = max_origin
        x, y = nx, ny
    return out


# ============ 六、主演示 ============
if __name__ == "__main__":
    # issue #147：编码钉法只在直跑时生效。模块顶层裸调会在导入期改写全局
    # stdout 编码（使 PYTHONIOENCODING 失效），且 stdout 被重定向时
    # （_io.StringIO 无 reconfigure 属性）直接 AttributeError ⇒ 导入即崩。
    # 守卫形态与 lingshu/world/scenario_cli.py:150-152 同源。
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8")
    print("=== 白箱自举·多模态主线：3D 时空 CNN × 时空记忆图（零 LLM） ===\n")
    mem = SpatiotemporalMemory()

    # ① 球向右滚动
    frames = synth_ball_rolling(frames=10, speed_px=2)
    prims, feat = extract_spatiotemporal_primitives(frames)
    ev = mem.remember(prims, label="球")
    print("① 球向右滚动")
    print(f"   时空原语: 方向={prims['direction']} 速度={prims['speed']}/帧 "
          f"运动量={prims['motion_magnitude']} 周期={prims['period']}")
    print(f"   3D卷积特征图尺寸: {feat.shape}（T×H×W 时空压缩）")

    # ② 周期闪烁（灯）
    frames2 = synth_blinking(frames=12, period=3)
    prims2, _ = extract_spatiotemporal_primitives(frames2)
    mem.remember(prims2, label="灯")
    print("\n② 周期闪烁（灯）")
    print(f"   时空原语: 周期={prims2['period']}帧 方向={prims2['direction']} "
          f"运动量={prims2['motion_magnitude']}")

    # ③ 静止场景（基线）
    frames3 = synth_static(frames=6)
    prims3, _ = extract_spatiotemporal_primitives(frames3)
    mem.remember(prims3, label="背景")
    print("\n③ 静止场景")
    print(f"   时空原语: 方向={prims3['direction']} 运动量={prims3['motion_magnitude']} "
          f"移动={prims3['moving']}")

    # ④ 回忆（看见→记住→回忆 白箱闭环）
    print("\n=== 时空记忆图回忆（白箱召回） ===")
    print(f"全部事件: {len(mem.events)} 个 | 一致性自校验: "
          f"{'✔' if mem.verify_consistency() else '✘'}")
    balls = mem.recall({"label": "球"})
    moving = mem.recall({"moving": True})
    periodic = mem.recall({"period": 3})
    print(f"回忆[球]: {len(balls)} 个（方向={balls[0]['direction'] if balls else '?'}）")
    print(f"回忆[运动中]: {len(moving)} 个")
    print(f"回忆[周期3帧]: {len(periodic)} 个（灯闪烁规律）")

    # 判定
    ok_dir = prims['direction'] == "向右"
    ok_period = prims2['period'] == 3
    ok_static = not prims3['moving']
    ok_mem = mem.verify_consistency()
    total = 4
    passed = sum([ok_dir, ok_period, ok_static, ok_mem])
    print(f"\n=== 判定 ===\n时空识别: {passed}/{total}（方向/周期/静止基线/记忆一致性）")
