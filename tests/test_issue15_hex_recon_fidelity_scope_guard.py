# -*- coding: utf-8 -*-
"""守卫 · hex_recon 的保真度声明必须与「只存包围盒」这一实现事实一致（issue #15）

缺陷（锚定 50ee03b2，`lingshu/nn/hex_recon.py`）：
    模块 docstring 声称「**99% 逐像素**」与「应 **100% 可复原**」，但
    `regions_to_connections()` 只序列化 center/size/color/pixels/type 五个字段；
    真实像素集合 `mask_indices` 被标注「内部使用,不序列化」。
    ⇒ `reconstruct()` 只能按 center/size 画**轴对齐矩形**：
        · 圆形 r=8..18 整图匹配率 **0.9609–0.9900**（10/11 组低于 99%）
        · 正方形（轴对齐）0.9932–0.9964（**也不是 1.0000**，来自 round/int 取整漂移）
        · 满幅纯色 1.000000
    ⇒ 声明的适用面**窄于**实现的能力，属「记录撒谎」类缺陷。

处置（任务卡三选一）：**改声明**（如实收窄），**不改数值行为**。
    理由：改判据（换分母）会把真实的保真度损失藏起来 —— 那是不诚实测量；
    要真正达标须序列化形状（mask/contour/RLE），属**格式变更**，需设计裁定。

本守卫四条：
  A 声明如实：docstring 不得再出现无条件的「应 100% 可复原」，且须带收窄范围常量
  B 事实钉住：非矩形（圆形）的匹配率**确实**低于 99%（证明收窄有据）
  C 边界钉住：轴对齐矩形/满幅纯色**确实**≥99%（证明收窄没削掉真实能力）
  D 根因钉住：序列化键**恰为**那 6 个、不含任何形状字段（声明的依据不可漂移）

运行（仓根）：
  PYTHONUTF8=1 python -X utf8 -m pytest tests/test_issue15_hex_recon_fidelity_scope_guard.py -q --no-header
"""
from __future__ import annotations

import inspect
import os
import sys

_THIS = os.path.abspath(__file__)
REPO = os.environ.get("LINGSHU_ROOT") or os.path.dirname(os.path.dirname(_THIS))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

from lingshu.nn import hex_recon as HR  # noqa: E402

# 常量在修复后才存在：缺失时用哨兵，让「修复前」的红来自断言而非导入崩溃
try:
    SHAPE_FIDELITY_SCOPE = HR.SHAPE_FIDELITY_SCOPE
except AttributeError:  # pragma: no cover - 仅修复前
    SHAPE_FIDELITY_SCOPE = None

EXPECTED_KEYS = {"id", "type", "color", "center", "size", "pixels"}
FORBIDDEN_SHAPE_KEYS = ("mask", "contour", "rle", "polygon", "shape")
#: 修复前那句**无条件**声明（原文逐字）。注意：本守卫钉的是「声明被加了限定」，
#: 不是「这些词永不出现」——`荣 2026-09-08` 的历史原文引用仍可保留（那是目标与出处，
#: 且已标注为原文），所以只禁这一条未加限定的完整句式。
UNCONDITIONAL_SENTENCE = "渲染→应 100% 可复原"
SCOPE_MARKER = "轴对齐矩形"


def _canvas(size=96, bg=128):
    return np.full((size, size, 3), bg, dtype=np.uint8)


def _circle(img, cy, cx, r, color=(220, 60, 60)):
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    img[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = color
    return img


def _square(img, y0, x0, side, color=(60, 200, 60)):
    img[y0:y0 + side, x0:x0 + side] = color
    return img


def _accuracy(img):
    regions, _ = HR.extract_regions(img)
    conns = HR.regions_to_connections(regions, img.shape)
    return HR.pixel_accuracy(img, HR.reconstruct(conns))


# ---------------- A 声明如实 ----------------

def test_a_declaration_is_narrowed():
    """A docstring 不得再无条件下断言 100% 可复原，且须带可核对的收窄范围。"""
    doc = HR.__doc__ or ""
    assert UNCONDITIONAL_SENTENCE not in doc, (
        "A docstring 仍含无条件句式 %r —— 而实现只存包围盒、圆形实测 10/11 组 <99%%：见 #15"
        % (UNCONDITIONAL_SENTENCE,))
    assert SHAPE_FIDELITY_SCOPE is not None, (
        "A 未找到 `SHAPE_FIDELITY_SCOPE`：保真度适用范围无从机器核对")
    assert SHAPE_FIDELITY_SCOPE == "axis_aligned_rectangles_only", (
        "A 保真度范围常量不诚实：%r（实现只出轴对齐矩形）" % (SHAPE_FIDELITY_SCOPE,))
    assert SCOPE_MARKER in doc, (
        "A docstring 未出现限定词 %r —— 100%% 声明没有被任何适用范围限定住" % (SCOPE_MARKER,))
    assert "99%" in doc or "0.99" in doc or "匹配率" in doc, (
        "A docstring 须保留实际达成的口径描述，不能一删了之")


# ---------------- B 事实钉住：非矩形确实低于 99% ----------------

def test_b_non_rectangular_is_below_99_percent():
    """B 圆形（非矩形）的匹配率必须**确实**低于 99% —— 收窄声明的事实依据。"""
    accs = [_accuracy(_circle(_canvas(), 48, 48, r)) for r in (10, 14, 18)]
    assert all(a < 0.99 for a in accs), (
        "B 圆形匹配率并未低于 99%%（实测 %r）—— 若如此，收窄声明失去依据，"
        "本守卫须重新审视" % (accs,))
    # 并钉住机制：丢的恰是包围盒四角 ⇒ 解析式 1-((2r+1)^2-pi*r^2)/(W*H)
    for r in (14, 18):
        measured = _accuracy(_circle(_canvas(), 48, 48, r))
        analytic = 1.0 - ((2 * r + 1) ** 2 - np.pi * r * r) / (96 * 96)
        assert abs(measured - analytic) < 0.005, (
            "B r=%d 实测 %.6f 与解析式 %.6f 偏差过大 —— 机制不再是「丢包围盒四角」"
            % (r, measured, analytic))


# ---------------- C 边界钉住：轴对齐矩形不低 ----------------

def test_c_rectangular_and_solid_still_meet_99_percent():
    """C 轴对齐矩形与满幅纯色必须 ≥99% —— 收窄不得削掉真实能力。"""
    for side in (16, 24, 32):
        a = _accuracy(_square(_canvas(), 48 - side // 2, 48 - side // 2, side))
        assert a >= 0.99, (
            "C 正方形 side=%d 匹配率 %.6f < 0.99 —— 声明收窄过头，削掉了真实能力"
            % (side, a))
    solid = _accuracy(_canvas(bg=200))
    assert solid == 1.0, "C 满幅纯色应为 1.0，实测 %.6f" % (solid,)


# ---------------- D 根因钉住：序列化不含形状 ----------------

def test_d_serialized_keys_contain_no_shape_field():
    """D 序列化键恰为 6 个已知字段，不含任何形状字段 —— 声明依据不可漂移。"""
    img = _circle(_canvas(), 48, 48, 12)
    regions, _ = HR.extract_regions(img)
    conns = HR.regions_to_connections(regions, img.shape)
    assert conns["connections"], "D 未提取到区域，守卫前提失效"
    keys = set(conns["connections"][0].keys())
    assert keys == EXPECTED_KEYS, (
        "D 序列化键变了：%r（期望 %r）—— 若新增了形状字段，"
        "`SHAPE_FIDELITY_SCOPE` 与 A 的收窄声明必须同步更新"
        % (sorted(keys), sorted(EXPECTED_KEYS)))
    for bad in FORBIDDEN_SHAPE_KEYS:
        assert bad not in keys, "D 序列化竟含形状字段 %r —— 收窄声明已过时" % (bad,)
    # mask_indices 仍在 region 内部（作为「未序列化」这一事实的另一面）
    assert "mask_indices" in regions[0], (
        "D `mask_indices` 不再存在 —— 本守卫对「形状未序列化」的刻画须重写")
    src = inspect.getsource(HR.regions_to_connections)
    assert "mask_indices" not in src, (
        "D `regions_to_connections` 已开始序列化 mask_indices —— 收窄声明须重审")