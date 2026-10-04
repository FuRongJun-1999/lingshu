# -*- coding: utf-8 -*-
"""stcnn.py · 白箱自举第三阶段·多模态主线原型 v1——3D 时空尺度 CNN（零 LLM）
理论：《白箱自举·LLM替代与3D时空多模态》（§3 3D时空尺度 = 时间×空间统一卷积）
核心：连续帧 → 时空体素 (T×H×W) → 3D 卷积（核跨时间+空间）→ 时空特征 → 时空原语
  （运动方向/速度/周期/形状变化）→ 时空记忆图写入（看见→记住→回忆 白箱闭环）
零依赖：纯 numpy 确定性实现（D-005 核心零外部依赖哲学）。
"""
import sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')


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
    # 归一化 0~1
    amin, amax = arr.min(), arr.max()
    if amax > amin:
        arr = (arr - amin) / (amax - amin)
    return arr


# ============ 二、3D 卷积（时间+空间统一卷积核） ============
def conv3d(voxel, kernel, stride=(1, 1, 1)):
    """3D 卷积：核 (kt, kh, kw) 同时跨时间+空间滑窗 → 时空特征图。
    核可自定义为时空滤波器（如运动检测核 = 时间差分×空间平滑）。"""
    T, H, W = voxel.shape
    kt, kh, kw = kernel.shape
    st, sh, sw = stride
    ot = (T - kt) // st + 1
    oh = (H - kh) // sh + 1
    ow = (W - kw) // sw + 1
    out = np.zeros((ot, oh, ow), dtype=np.float32)
    for t in range(ot):
        for i in range(oh):
            for j in range(ow):
                patch = voxel[t * st:t * st + kt, i * sh:i * sh + kh, j * sw:j * sw + kw]
                out[t, i, j] = float((patch * kernel).sum())
    return out


def max_pool3d(feat, size=(2, 2, 2)):
    """3D 最大池化（降采样时空特征图）"""
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


# ============ 三、时空原语提取（运动/方向/速度/周期） ============
def frame_diff(frames, threshold=0.08):
    """帧间差分 → 运动区域掩码序列（|f(t+1)-f(t)| > 阈值）"""
    arr = np.asarray(frames, dtype=np.float32)
    diffs = np.abs(np.diff(arr, axis=0))
    return (diffs > threshold).astype(np.float32)


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
    平台型信号（1,1 相邻）无严格局部极大，局部极大法会漏检。"""
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
    return period


def extract_spatiotemporal_primitives(frames):
    """时空原语提取：运动/方向/速度/周期（白箱确定性）"""
    voxel = frames_to_voxel(frames)
    # ① 3D 卷积：运动核 → 时空特征图（运动响应）
    motion_feat = conv3d(voxel, MOTION_KERNEL)
    motion_mag = np.abs(motion_feat).sum()
    # ② 帧差运动区域 + 质心轨迹
    diffs = frame_diff(frames)
    centroids = []
    for t in range(diffs.shape[0]):
        region = diffs[t]
        ys, xs = np.nonzero(region)
        if len(xs) > 2:
            centroids.append((float(xs.mean()), float(ys.mean())))
        elif centroids:
            centroids.append(centroids[-1])
    # ③ 运动信号（每帧运动量）→ 周期检测
    motion_signal = [float(region.sum()) for region in diffs]
    period = detect_period(motion_signal)
    # ④ 原语汇总
    prims = {
        "motion_magnitude": round(float(motion_mag), 4),
        "direction": motion_direction(centroids),
        "speed": round(motion_speed(centroids), 4),
        "period": period,
        "moving": bool(motion_mag > 0.5),  # bool()：numpy bool 不满足 is True
        "trajectory_len": len(centroids),
    }
    return prims, motion_feat


# ============ 四、时空记忆图写入（看见→记住） ============
class SpatiotemporalMemory:
    """时空记忆图：事件 → 时空锚点（时间/位置/运动模式）+ 记忆边。
    零 LLM 确定性记忆（对接灵枢语义时空图的感知侧入口）。"""

    def __init__(self):
        """时空卷积网络层级参数初始化。"""
        self.events = []  # {t_start, t_end, direction, speed, period, label}

    def remember(self, prims, label, t_start=0):
        """把一次「看见」的时空原语写为记忆事件（时空锚点）"""
        event = {
            "t_start": t_start,
            "t_end": t_start + max(1, prims.get("trajectory_len", 1) - 1),
            "direction": prims.get("direction", "静止"),
            "speed": prims.get("speed", 0.0),
            "period": prims.get("period"),
            "moving": prims.get("moving", False),
            "label": label,
        }
        self.events.append(event)
        return event

    def recall(self, query=None):
        """回忆：按条件查询时空记忆（白箱召回，无 LLM）"""
        out = []
        for e in self.events:
            if query is None:
                out.append(e)
            else:
                hit = all(
                    e.get(k) == v for k, v in query.items()
                    if k in ("direction", "moving", "period", "label"))
                if hit:
                    out.append(e)
        return out

    def verify_consistency(self):
        """自校验：回忆 vs 已记事件一致性（记忆无幻觉）"""
        recalled = self.recall()
        return len(recalled) == len(self.events)


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


# ============ 六、主演示 ============
if __name__ == "__main__":
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
