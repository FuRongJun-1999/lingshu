# -*- coding: utf-8 -*-
"""守卫 · extract_spatiotemporal_primitives 对空/不足帧列表 fail-closed（lingshu issue #798）

缺陷（HEAD 修复前，`lingshu/nn/stcnn.py`）
    `extract_spatiotemporal_primitives([])` 抛
    `ValueError: zero-size array to reduction operation minimum which has no identity`
    ——numpy 内部错误（出自 `frames_to_voxel` 对空数组取 min/max 做归一化），
    不含任何「输入为空帧列表」的领域信息，调用方无法从消息定位是输入格式问题
    还是内部 bug。改前实测读数：

        extract_spatiotemporal_primitives([])
        → ValueError: zero-size array to reduction operation minimum which has no identity

    单帧（1 帧）改前**不抛**——返回全零退化原语（静止/无周期/trajectory_len=0），
    属「未测量伪装成测得 0」，本件一并收口。

修复
    入口补校验：`len(frames) < 2` 即 raise
    `ValueError("extract_spatiotemporal_primitives：帧列表为空/不足（至少需要 2 帧，…）")`。

判据来源
    · 题面读数即判据：公开 issue #798 的空列表实测。
    · 最小帧数取 2 的依据：本函数全部时序算子——MOTION_KERNEL（kt=2）、
      frame_diff/frame_rise 的 np.diff(axis=0)、detect_period 的前沿间隔——
      都以「相邻两帧」为最小可分辨单元；2 帧是时序信号存在的最小输入
      （实测 2 帧正常产出原语，1 帧时间维为 1、conv3d 输出空特征图）。
    · 消息口径与 physics._check_samples（#662）「样本不足：需要 >= N 个」同族：
      含函数名、所需量、实得量三要素。
    · 无帧数 >= 2 的既有行为变化：仓内调用面（stcnn 主演示 6/10/12 帧、
      test_stcnn_period_edges 8/12 帧）全部 >= 2。

定点变异自证（抽掉修复 ⇒ 必红）
    把入口校验段整体删除 ⇒ G1 红（回到裸 numpy 错）、G2 红（单帧静默返回伪读数）、
    G3 红（消息不再含函数名/帧数）；恢复校验 ⇒ 全绿。

运行（仓根）：python -X utf8 -m pytest tests/test_issue798_stcnn_min_frames_guard.py -q --no-header
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lingshu.nn.stcnn import extract_spatiotemporal_primitives  # noqa: E402


# ---------- G1：空帧列表 → 领域 ValueError，不再漏 numpy 内部错误 ----------
def test_g1_empty_frames_raises_domain_valueerror():
    with pytest.raises(ValueError) as ei:
        extract_spatiotemporal_primitives([])
    msg = str(ei.value)
    # 领域消息三要素：函数名 + 所需帧数 + 实得帧数；绝无 numpy 内部措辞
    assert "extract_spatiotemporal_primitives" in msg
    assert "至少需要 2 帧" in msg
    assert "得到 0 帧" in msg
    assert "zero-size array" not in msg
    assert "reduction operation" not in msg


# ---------- G2：单帧（时间维无任何可分辨信号）→ 同款 fail-closed ----------
def test_g2_single_frame_raises_too():
    one = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError) as ei:
        extract_spatiotemporal_primitives([one])
    assert "至少需要 2 帧" in str(ei.value)
    assert "得到 1 帧" in str(ei.value)


# ---------- G3：2 帧（最小合法输入）照常产出原语——闸不误伤 ----------
def test_g3_two_frames_still_work():
    frames = [np.zeros((4, 4), dtype=np.float32), np.ones((4, 4), dtype=np.float32)]
    prims, feat = extract_spatiotemporal_primitives(frames)
    assert prims["direction"] in ("向右", "向左", "向上", "向下", "静止")
    assert isinstance(prims["moving"], bool)
    assert feat.shape[0] >= 1  # 时间维非空：MOTION_KERNEL kt=2 在 T=2 下 ot=1


if __name__ == "__main__":
    rc = pytest.main([__file__, "-q", "--no-header"])
    sys.exit(rc)
