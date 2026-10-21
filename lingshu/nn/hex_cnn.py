# -*- coding: utf-8 -*-
"""hex_cnn · 自研蜂窝 CNN + 并行条件路由神经网络（HEX-CNN-REV1 · 传感器层原生架构）
====================================================================================
荣 2026-09-06 裁定（四条，grill 访谈）：
  1. CNN×条件路由一体——CNN 是条件路由的特征前端，路由是 CNN 的语义判定层；
  2. 自监督+白箱固化——核参数可学习，结构/损失/梯度全程白箱可审计，
     学完固化即白箱（世界模型理论_白箱固化_v1 命题：LLM/训练引导→固化后零训练）；
  3. 统一蜂窝晶格+共享底座——各模态（图像/语音/视频/文字）前端编码为统一
     蜂窝晶格表示，共享 HexConv 底座与并行条件路由——同一架构多模态；
  4. 视觉先行——第一里程碑用图像域验证（img0/示范图，对接子部件/条件卡管线）。

设计根基：
  - 蜂窝形感知（白箱身体补充计划 §六）：6邻等距各向同性/最优镶嵌/更圆感受野/
    生物视网膜六角/匹配各向同性 3D 世界模型——HexConv 6邻 stencil（非 3x3 8邻）。
  - cnn_inpaint 先例：卷积核手写可解释（零训练黑箱）——本模块延续：核可学习但
    每步白箱（权重直读/响应图/损失曲线/梯度数值可验证）。
  - condition_route：条件卡+四态判定（ACCEPT/REJECT/DEFER/BLINDSPOT）——本模块
    把谓词参数化为滤波器组，多卡并行评估特征响应，四态语义不变。

纯 numpy + PIL（视觉域依赖惯例，D-005 零 LLM）。
"""
from __future__ import annotations
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

ALGO = "hex_cnn-0.1"

# 蜂窝六邻方向（axial 坐标 ±q/±r）：E, NE, NW, W, SW, SE（pointy-top 约定）
HEX_DIRS = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]


# ==================== 蜂窝晶格（axial 坐标 ↔ 像素） ====================

class HexGrid:
    """蜂窝晶格：axial 坐标 (q, r) 存储为「错行矩形」数组（row=r, col=q+offset）。
    pointy-top：六边形尖朝上下；行间水平错位半格，垂直间距 3/4·size。"""

    def __init__(self, width_cells: int, height_cells: int, cell_size: float = 1.0):
        self.W = width_cells      # 列数（q 方向）
        self.H = height_cells     # 行数（r 方向）
        self.size = float(cell_size)

    # ---- 像素坐标（cell 中心，图像坐标系 y 向下） ----
    def center_px(self, q: int, r: int) -> Tuple[float, float]:
        x = self.size * math.sqrt(3) * (q + r / 2.0)
        y = self.size * 1.5 * r
        return (x, y)

    def pixel_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        """像素 → 最近 cell（axial round：立方坐标四舍五入）。"""
        # 逆变换到轴向浮点
        qf = (math.sqrt(3) / 3 * x - y / 3.0) / self.size
        rf = (2.0 / 3.0 * y) / self.size
        return self._axial_round(qf, rf)

    def _axial_round(self, qf: float, rf: float) -> Tuple[int, int]:
        sf = -qf - rf
        q, r, s = round(qf), round(rf), round(sf)
        dq, dr, ds = abs(q - qf), abs(r - rf), abs(s - sf)
        if dq > dr and dq > ds:
            q = -r - s
        elif dr > ds:
            r = -q - s
        return (q, r)

    def neighbors(self, q: int, r: int) -> List[Tuple[int, int]]:
        return [(q + dq, r + dr) for dq, dr in HEX_DIRS]

    def in_bounds(self, q: int, r: int) -> bool:
        """错行矩形存储：r 全宽有效，q 按列数（边界 cell 邻接由 replicate padding 处理）。"""
        return 0 <= r < self.H and 0 <= q < self.W


def image_to_grid(img_arr: np.ndarray, cells_across: int = 48) -> Tuple[np.ndarray, HexGrid]:
    """图像 (H,W,3) uint8 → 蜂窝晶格特征 (rows, cols, 3) float32（每 cell 区域均值）。
    错行布局：第 r 行中心 x = √3·s·(q + 0.5·(r%2) + 0.5)，y = 1.5·s·r + s。
    s 为外接圆半径；横向节距 √3·s = W/(cells_across+0.5)——cells_across 列（含奇行半格
    错位）恰铺满画幅宽 W，每个 cell 的采样窗都落在画内。"""
    H, W = img_arr.shape[:2]
    s3 = W / (cells_across + 0.5)       # 横向节距 √3·s（此前误作 s 用：晶格横向越出画幅 √3 倍）
    cell = s3 / math.sqrt(3)
    rows = max(1, int(round(H / (cell * 1.5)) - 1)) or 1
    grid = HexGrid(cells_across, rows, cell)
    out = np.zeros((rows, cells_across, img_arr.shape[2]), dtype=np.float32)
    for r in range(rows):
        xoff = s3 * (0.5 * (r % 2) + 0.5)
        for q in range(cells_across):
            cx = s3 * q + xoff
            cy = 1.5 * cell * r + cell
            # cell 采样窗（六边形内切圆近似：方形窗均值；内切圆半径 √3·s/2 = s3/2）
            rad = max(1, int(s3 * 0.5))
            x0, x1 = int(cx - rad), int(cx + rad + 1)
            y0, y1 = int(cy - rad), int(cy + rad + 1)
            x0c, y0c = max(0, x0), max(0, y0)
            x1c, y1c = min(W, x1), min(H, y1)
            if x1c > x0c and y1c > y0c:
                out[r, q] = img_arr[y0c:y1c, x0c:x1c].reshape(-1, img_arr.shape[2]).mean(axis=0)
    return out, grid


def grid_to_image(grid_feat: np.ndarray, grid: HexGrid, scale: int = 6) -> np.ndarray:
    """蜂窝晶格 → 图像（六边形填充可视化；错行公式与 image_to_grid 一致）。"""
    rows, cols = grid_feat.shape[:2]
    cell = grid.size
    s = cell * scale
    Wpx = int(s * math.sqrt(3) * (cols + 1)) + 8
    Hpx = int(1.5 * s * (rows - 1) + 2 * s) + 8
    canvas = np.zeros((Hpx, Wpx, 3), dtype=np.uint8)
    for r in range(rows):
        xoff = s * math.sqrt(3) * (0.5 * (r % 2) + 0.5)
        for q in range(cols):
            cx = s * math.sqrt(3) * q + xoff + 4
            cy = 1.5 * s * r + s + 4
            color = np.clip(grid_feat[r, q], 0, 255).astype(np.uint8)
            _fill_hex(canvas, cx, cy, s * 0.94, color)
    return canvas


def _fill_hex(canvas: np.ndarray, cx: float, cy: float, size: float, color: np.ndarray):
    """pointy-top 六边形填充：|dy|≤size/2 满宽(√3/2·size)，下半段线性收尖(√3·(size-|dy|))。"""
    H, W = canvas.shape[:2]
    half_full = size * math.sqrt(3) / 2
    for dy in range(int(-size), int(size) + 1):
        y = int(cy) + dy
        if not (0 <= y < H):
            continue
        ady = abs(dy)
        half = half_full if ady <= size / 2 else max(0.0, (size - ady) * math.sqrt(3))
        x0, x1 = int(cx - half), int(cx + half) + 1
        if x1 > x0:
            canvas[y, max(0, x0):min(W, x1)] = color


# ==================== HexConv · 6 邻 stencil 卷积 ====================

def hex_conv(feature: np.ndarray, kernel7: np.ndarray,
             channels: Optional[List[int]] = None) -> np.ndarray:
    """蜂窝卷积：每 cell = w0·自身 + Σ w_i·六邻（6 邻 stencil，7 参数）。
    feature: (rows, cols, C)；kernel7: (7,) 或 (C,7)——逐通道核。
    边界：出界邻取自身值（edge-replicate 的蜂窝版，保结构）。"""
    rows, cols, C = feature.shape
    k = np.asarray(kernel7, dtype=np.float32)
    if k.ndim == 1:
        k = np.tile(k, (C, 1))          # (C,7) 共享核
    out = np.zeros_like(feature, dtype=np.float32)
    padded = _hex_pad_replicate(feature)
    # padded: (rows+2, cols+2, C)——错行阵列的六邻偏移
    # 行内水平邻居: (r, q±1)；斜向: (r, q∓1+2*(r%2)-1 + ...)——统一用错行表：
    # 错行存储下偶数行六邻: (q-1,r)(q+1,r)(q-1,r-1)(q,r-1)(q-1,r+1)(q,r+1)
    #                奇数行六邻: (q-1,r)(q+1,r)(q,r-1)(q+1,r-1)(q,r+1)(q+1,r+1)
    even_off = [(-1, 0), (1, 0), (-1, -1), (0, -1), (-1, 1), (0, 1)]
    odd_off = [(-1, 0), (1, 0), (0, -1), (1, -1), (0, 1), (1, 1)]
    acc = k[:, 0][None, None, :] * padded[1:-1, 1:-1, :]
    for i in range(6):
        dq, dr = even_off[i]
        e = padded[1 + dr: 1 + dr + rows, 1 + dq: 1 + dq + cols, :]
        acc = acc + k[:, i + 1][None, None, :] * e
        dq2, dr2 = odd_off[i]
        o = padded[1 + dr2: 1 + dr2 + rows, 1 + dq2: 1 + dq2 + cols, :]
        # 奇数行用 odd 偏移，偶数行用 even 偏移——按行 mask 混合
        rowmask = np.zeros((rows, 1, 1), dtype=np.float32)
        rowmask[1::2] = 1.0
        acc = acc + rowmask * k[:, i + 1][None, None, :] * (o - e)
    out[:] = acc
    return out


def _hex_pad_replicate(feature: np.ndarray) -> np.ndarray:
    rows, cols, C = feature.shape
    top = feature[0:1]      # (1, cols, C)
    bot = feature[-1:]
    left = feature[:, 0:1]  # (rows, 1, C)
    right = feature[:, -1:]
    tl = feature[0:1, 0:1]
    tr = feature[0:1, -1:]
    bl = feature[-1:, 0:1]
    br = feature[-1:, -1:]
    mid = np.concatenate([left, feature, right], axis=1)
    topm = np.concatenate([tl, top, tr], axis=1)
    botm = np.concatenate([bl, bot, br], axis=1)
    return np.concatenate([topm, mid, botm], axis=0)


# ---- 手写可解释核（7 参数：[自身, E, NE, NW, W, SW, SE] 按奇偶行混合语义） ----

def kernel_smooth() -> np.ndarray:
    """各向同性平滑：7 点均匀（方阵 3x3 做不到的等距——蜂窝独有优势）。"""
    return np.array([1.0, 1, 1, 1, 1, 1, 1], dtype=np.float32) / 7.0


def kernel_edge(vertical: bool = True) -> np.ndarray:
    """方向边缘：水平向差分（vertical=True 检测纵向边缘）或纵向向差分。"""
    if vertical:
        # 左右差：W 侧 - E 侧（近似）
        return np.array([0.0, -1, 0, 0, 1, 0, 0], dtype=np.float32)
    return np.array([0.0, 0, -1, -1, 0, 1, 1], dtype=np.float32) / 2.0


def kernel_laplacian() -> np.ndarray:
    """各向同性二阶（蜂窝离散 Laplacian：邻和×(1/6) − 自身）——蜂窝标准差分。"""
    return np.array([-1.0, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6], dtype=np.float32)


def kernel_center_surround() -> np.ndarray:
    """中心- surround（视网膜感受野拮抗——生物视觉六角先例）。"""
    return np.array([1.0, -1 / 6, -1 / 6, -1 / 6, -1 / 6, -1 / 6, -1 / 6], dtype=np.float32)


def extended_prior_family(n: int) -> List[np.ndarray]:
    """n 个互异白箱先验核(每核唯一,无对称重复)——扩容的单元先验族。
    构成:六方向 edge(六边形天然各向同性完备)+smooth+粗平滑+laplacian
    +center-surround;超出时按序加微噪变体(仍互异)。"""
    fam: List[np.ndarray] = []
    for d in range(6):                                   # 六方向 edge
        k = np.zeros(7, dtype=np.float32)
        k[0] = 0.0
        k[1 + d] = -1.0
        k[1 + (d + 3) % 6] = 1.0
        fam.append(k)
    fam.append(kernel_smooth())                          # 各向同性平滑
    coarse = np.array([1.0] + [1.5] * 6, dtype=np.float32)
    fam.append(coarse / coarse.sum())                    # 粗平滑
    fam.append(kernel_laplacian())
    fam.append(kernel_center_surround())
    i = 0
    while len(fam) < n:                                  # 微噪变体(互异)
        base = fam[i % 10].copy() + (i + 1) * 1e-3
        fam.append(base / (np.abs(base).sum() + 1e-9))
        i += 1
    return fam[:n]


DEFAULT_KERNELS: Dict[str, np.ndarray] = {
    "smooth": kernel_smooth(), "edge_v": kernel_edge(True),
    "edge_h": kernel_edge(False), "laplacian": kernel_laplacian(),
    "center_surround": kernel_center_surround(),
}


# ==================== 自监督微调 + 白箱固化 ====================

def selfsup_finetune(img_arr: np.ndarray, kernel_name: str = "center_surround",
                     cells_across: int = 48, epochs: int = 30, lr: float = 0.05,
                     mask_ratio: float = 0.25, seed: int = 7) -> Dict:
    """自监督：掩码重建——随机遮蔽部分 cell，用当前核从邻域预测被遮 cell，
    以重建误差的数值梯度微调 7 参数（有限差分——梯度本身可审计）。
    返回 {kernel, loss_curve, init_loss, final_loss}。

    `init_loss` / `final_loss` 是**同一个固定评估掩码**下的训练前／训练后读数
    （评估掩码在训练前抽一次、全程不变 ⇒ 两个读数同口径、可比）。训练用的
    每-epoch 掩码照旧逐轮重抽，`loss_curve` 仍是**各 epoch 训练掩码下**的曲线
    （监控用途，口径不变）。

    #593：此前 `final_loss` 取 `loss_curve[-1]`——那是末 epoch 训练掩码下的读数，
    与训练前抽的 `init_loss` 掩码是两次**独立抽样**，两者不可相互比较。修法把
    训练前那一次抽样提为评估掩码、训练后在同一掩码下重算，故训练路径逐位不变。

    `epochs <= 0` 表示「不训练」：`loss_curve` 为空，`final_loss` 回落为
    `init_loss`（没有训练就不存在「训练后」的读数），核保持为输入核。
    """
    gray = (0.299 * img_arr[..., 0] + 0.587 * img_arr[..., 1]
            + 0.114 * img_arr[..., 2]).astype(np.float32) if img_arr.ndim == 3 else img_arr.astype(np.float32)
    feat, grid = image_to_grid(gray[..., None].repeat(3, axis=2), cells_across)
    lum = feat[..., :1]                                  # 亮度通道 (rows,cols,1)
    rng = np.random.default_rng(seed)
    # #593：评估掩码——训练前抽一次、全程固定，`init_loss` / `final_loss` 都在它下面
    # 取值 ⇒ 同口径可比。此处正是 #593 之前 `init_loss` 抽的那一次 ⇒ `rng` 的抽样
    # 序列（进而各 epoch 训练掩码、核、`loss_curve`）与改前**逐位不变**；此后各
    # epoch 的掩码仍由 `rng` 逐轮重抽（本行不再额外消耗 `rng`）。
    eval_hole = rng.random(lum.shape[:2]) < mask_ratio
    rows, cols = lum.shape[:2]
    w = DEFAULT_KERNELS[kernel_name].copy()
    loss_curve = []

    def recon_loss(wv: np.ndarray, hole: np.ndarray) -> float:
        """给定掩码下的重建误差。

        掩码由调用方显式传入：同一步内的 base / w+eps / w-eps / 曲线记录必须
        共用**同一**掩码（common random numbers），否则中心差分算的是
        「两次换掩码带来的损失波动」，而不是核参数的真实曲率。
        """
        masked = lum.copy()
        masked[hole] = 0.0
        pred = hex_conv(masked, wv)[..., :1]
        return float(((pred[hole] - lum[hole]) ** 2).mean())

    init_loss = recon_loss(w, eval_hole)                 # #593：评估掩码（非训练掩码）
    for _ in range(epochs):
        hole = rng.random(lum.shape[:2]) < mask_ratio      # 每步抽一次；步内共用
        grads = np.zeros(7, dtype=np.float32)
        eps = 1e-3
        for i in range(7):                                # 有限差分数值梯度（白箱可验证）
            wp = w.copy(); wp[i] += eps
            wm = w.copy(); wm[i] -= eps
            grads[i] = (recon_loss(wp, hole) - recon_loss(wm, hole)) / (2 * eps)
        w -= lr * grads
        w /= (abs(w).sum() + 1e-8)                        # 归一化（核能量守恒——白箱约束）
        loss_curve.append(recon_loss(w, hole))
    # #593：训练后读数在**同一评估掩码**下重算（不再取 `loss_curve[-1]`——那是训练
    # 掩码下的读数，与 `init_loss` 不可比）。零轮训练 ⇒ 无「训练后」读数，回落 init
    # （沿用 #591 的「空曲线即无训练后读数」表述）。
    final_loss = recon_loss(w, eval_hole) if loss_curve else init_loss
    return {"kernel": w, "kernel_name": kernel_name, "init_loss": round(init_loss, 5),
            "final_loss": round(final_loss, 5),
            "loss_curve": loss_curve, "algo": ALGO, "epochs": epochs, "lr": lr}


def save_consolidated(kernels: Dict[str, np.ndarray], meta: Dict, path: str) -> str:
    """白箱固化：学完的核参数 + 训练元数据落盘（JSON——可直读可审计）。

    原子落盘：先写同目录临时档再 os.replace 顶替目标档——写入中途失败（含
    元数据不可序列化）时目标档保持原样，不会被先截断成空档/半截坏档。
    每次保存独占一个定长短名临时档，仅清理自己成功创建的文件；同进程并发保存互不
    消费临时档。最终内容取最后一次成功 replace，不承诺调用顺序或合并。

    #498（#393 修复的副作用）：`os.replace` 把目标换成**新 inode** ⇒ ①新档权限
    取 `0o666 & ~umask`，不继承旧档刻意收紧的权限位（`0600` 固化档被静默放宽成
    `0644`）；②目标是符号链接时替换的是**链接本身**（链接被换成普通档，真实
    固化位置不再更新）。旧实现 `open(path, "w")` 截断写同一 inode，两者都保留。
    修法：目标先 `realpath` 穿透链接（临时档建在真实目标同目录 ⇒ 同卷原子性
    不丢），`os.replace` 前把旧档权限位拷到临时档（目标不存在则跳过，用 umask
    默认）。用 `& 0o7777` 字面量而不在顶层 `import stat`（避免打掉 #156 组件
    发现的「未知裸名」判据 D2/D4a）。边界：只搬权限位，不搬属主/属组。
    """
    import uuid

    payload = {"algo": ALGO, "consolidated": True,
               "kernels": {k: [round(float(x), 6) for x in v] for k, v in kernels.items()},
               **meta}
    target = os.path.realpath(path)
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    tmp = os.path.join(os.path.dirname(target), f".lingshu-save.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    owns_tmp = False
    try:
        with open(tmp, "x", encoding="utf-8") as f:
            owns_tmp = True
            json.dump(payload, f, ensure_ascii=False, indent=1)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.chmod(tmp, os.stat(target).st_mode & 0o7777)
        except FileNotFoundError:
            pass
        os.replace(tmp, target)
    except BaseException:
        if not owns_tmp:
            raise
        try:
            # #498 反向腿：临时档可能刚被 chmod 成旧档的**只读位**——Windows 上只读
            # 档删不掉（`os.unlink` ⇒ PermissionError）⇒ 失败路径留下 `.tmp-` 残件。
            # 清理前先尽力恢复可写（POSIX 上 unlink 只看目录权限，此步无害）。
            if os.path.exists(tmp):
                try:
                    os.chmod(tmp, 0o600)
                except OSError:
                    pass
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def load_consolidated(path: str) -> Dict[str, np.ndarray]:
    """加载固化核（确定性模块——零训练依赖运行）。"""
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return {k: np.array(v, dtype=np.float32) for k, v in payload["kernels"].items()}


# ==================== 并行条件路由 NN 层 ====================

class ConditionFilterCard:
    """条件卡 → 滤波器组：谓词参数化为「核+通道+响应阈值」。
    响应 = |HexConv 特征| 在卡关注区域的强度 → 四态判定证据。"""

    def __init__(self, card_id: str, kernel: np.ndarray, threshold: float,
                 semantic: str, channel: int = 0, region: Optional[Tuple] = None):
        self.card_id = card_id
        self.kernel = np.asarray(kernel, dtype=np.float32)
        self.threshold = float(threshold)
        self.semantic = semantic
        self.channel = channel
        self.region = region        # (r0,c0,r1,c1) 关注域；None=全域

    def response(self, feature: np.ndarray) -> float:
        r = hex_conv(feature, self.kernel)[..., self.channel]
        if self.region:
            r0, c0, r1, c1 = self.region
            r = r[r0:r1, c0:c1]
        return float(abs(r).mean())

    def verdict(self, feature: np.ndarray) -> Dict:
        """四态判定（语义与 condition_route 一致，证据=响应强度/阈值）。"""
        resp = self.response(feature)
        ratio = resp / max(1e-6, self.threshold)
        if resp >= self.threshold:
            v, reason = "ACCEPT", f"响应 {resp:.4f} ≥ 阈值 {self.threshold:.4f}"
        elif resp >= 0.5 * self.threshold:
            v, reason = "DEFER", f"响应不足 {resp:.4f}/{self.threshold:.4f}（证据弱,待验）"
        elif resp <= 0.1 * self.threshold:
            v, reason = "REJECT", f"响应趋零 {resp:.4f}（条件冲突）"
        else:
            v, reason = "BLINDSPOT", f"响应 {resp:.4f} 落入不可判带"
        return {"card": self.card_id, "semantic": self.semantic, "verdict": v,
                "reason": reason, "response": round(resp, 5),
                "threshold": self.threshold, "ratio": round(ratio, 3)}


def parallel_route(feature: np.ndarray, cards: List[ConditionFilterCard]) -> List[Dict]:
    """并行条件路由：全部卡在同一特征图上评估（numpy 顺序循环但无状态依赖——
    语义上并行，每卡独立证据；批量核卷积可再加速，V0 先保清晰）。"""
    return [card.verdict(feature) for card in cards]
