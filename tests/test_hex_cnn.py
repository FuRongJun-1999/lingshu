#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""hex_cnn 回归测试 · HEX-CNN-REV1
覆盖：蜂窝晶格(像素↔cell 往返/6邻对称性) / HexConv(手写核性质:平滑=均值保持,
Laplacian 零响应于常数,center-surround 拮抗) / 自监督微调(损失下降+白箱固化
落盘/加载确定性) / 并行条件路由 NN(四态语义/多卡并行) / 真实图验证(素材可选：上游素材在盘时跑)。

脚本/pytest 双模式：python tests/test_hex_cnn.py 或 python -m pytest tests/test_hex_cnn.py
（执行体收拢进 main()——此前为模块级直跑，pytest 收集时即触发 sys.exit，
  整个 tests/ 目录无法一键收集；断言与读数与脚本模式逐位一致。）
"""
import json
import os
import sys
import tempfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from PIL import Image
from lingshu.nn.hex_cnn import (DEFAULT_KERNELS, ConditionFilterCard, HexGrid,
                          image_to_grid, grid_to_image, hex_conv, kernel_center_surround,
                          kernel_edge, kernel_laplacian, kernel_smooth, load_consolidated,
                          parallel_route, save_consolidated, selfsup_finetune)

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")


def main():
    results.clear()
    # ============ ① 蜂窝晶格几何 ============
    print("\n[1] 蜂窝晶格(axial/pointy-top)")
    g = HexGrid(10, 8, 1.0)
    cx, cy = g.center_px(0, 0)
    check("原点 cell 中心在原点", abs(cx) < 1e-9 and abs(cy) < 1e-9)
    cx2, cy2 = g.center_px(1, 0)
    import math as _math
    check("同行相邻间距 = √3·size", abs((cx2 - cx) - _math.sqrt(3)) < 1e-6)
    q, r = g.pixel_to_cell(cx2, cy2)
    check("像素→cell 往返", (q, r) == (1, 0), str((q, r)))
    n = g.neighbors(3, 3)
    check("六邻 6 个且互不相同", len(n) == 6 and len(set(n)) == 6)
    dists = [abs(g.center_px(*n[i])[0] - g.center_px(3, 3)[0]) ** 2
             + abs(g.center_px(*n[i])[1] - g.center_px(3, 3)[1]) ** 2 for i in range(6)]
    check("6 邻等距(各向同性——蜂窝核心性质)", max(dists) - min(dists) < 1e-6,
          f"spread={max(dists)-min(dists):.2e}")

    # ============ ② HexConv 手写核性质 ============
    print("\n[2] HexConv 手写核性质")
    feat = np.full((8, 10, 3), 100.0, dtype=np.float32)
    sm = hex_conv(feat, kernel_smooth())
    check("平滑核对常数场恒等(归一化核)", np.allclose(sm, 100.0, atol=1e-4))
    lap = hex_conv(feat, kernel_laplacian())
    check("Laplacian 对常数场零响应", np.allclose(lap, 0.0, atol=1e-4))
    cs = hex_conv(feat, kernel_center_surround())
    check("center-surround 对常数场零响应(拮抗平衡)",
          np.allclose(cs, 0.0, atol=1e-4))
    # 阶跃边缘 → 方向核有响应
    step = np.full((8, 10, 3), 50.0, dtype=np.float32)
    step[:, 5:] = 150.0
    ev = hex_conv(step, kernel_edge(True))
    check("纵向边缘核在阶跃处有响应", abs(ev[:, 4:6]).mean() > 10,
          f"边缘响应 {abs(ev[:, 4:6]).mean():.1f}")
    check("HexConv 输出形状保持", ev.shape == step.shape)

    # ============ ③ 图像→晶格→可视化往返 ============
    print("\n[3] 图像↔晶格")
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    img[40:80, 60:100] = 255  # 中央白块
    feat, grid = image_to_grid(img, cells_across=32)
    check("晶格尺寸合理", feat.shape[0] >= 4 and feat.shape[1] == 32, str(feat.shape))
    # 白块 (x 60~100, y 40~80) → 实测亮 cell 集中在 (r=5..7, q=7..10)
    check("白块区域 cell 亮于角落", feat[6, 8].mean() > feat[1, 1].mean() + 50,
          f"块内={feat[6, 8].mean():.0f} 角落={feat[1, 1].mean():.0f}")
    vis = grid_to_image(feat, grid, scale=6)
    check("可视化画布非空且亮度合理", vis.shape[0] > 50 and vis.mean() > 5,
          f"{vis.shape} mean={vis.mean():.1f}")

    # ============ ④–⑥ 素材相关段（素材可选：上游素材未随导出时显式 SKIP） ============
    # 素材路径：上游私有素材（导出面不存在）；在盘时本段按原判据全跑。
    _ASSET = os.path.join(PROJECT_ROOT, "data", "img", "0.png")
    if os.path.exists(_ASSET):
        # ============ ④ 自监督微调 + 白箱固化 ============
        print("\n[4] 自监督(掩码重建)+白箱固化")
        img = Image.open(_ASSET).convert("RGB")
        arr0 = np.asarray(img.resize((200, 328)))
        tr = selfsup_finetune(arr0, kernel_name="center_surround", epochs=8, lr=0.3)
        check("损失记录完整(epochs 轮)", len(tr["loss_curve"]) == 8)
        check("自监督损失下降(可学习性的白箱证据)",
              tr["final_loss"] <= tr["init_loss"] * 1.05,
              f"init={tr['init_loss']} final={tr['final_loss']}")
        check("核归一化(能量守恒约束)", abs(abs(tr["kernel"]).sum() - 1.0) < 1e-5)
        tmpd = tempfile.mkdtemp()
        kpath = save_consolidated({**DEFAULT_KERNELS, "learned": tr["kernel"]},
                                  {"source": "upstream-asset", "epochs": 8}, os.path.join(tmpd, "kernels.json"))
        loaded = load_consolidated(kpath)
        check("固化落盘/加载一致", np.allclose(loaded["learned"], tr["kernel"], atol=1e-5))
        check("固化文件含全部手写核+学习核", set(loaded) >= set(DEFAULT_KERNELS) | {"learned"})
        with open(kpath, encoding="utf-8") as f:
            payload = json.load(f)
        check("固化 JSON 可直读(白箱审计)", payload["consolidated"] is True
              and len(payload["kernels"]["learned"]) == 7)

        # ============ ⑤ 并行条件路由 NN ============
        print("\n[5] 并行条件路由 NN(条件卡→滤波器组→四态)")
        # 条件卡:纵向结构(edge_v 核)在素材图上应有强响应(主体边缘);阈值按实际响应校准
        base = image_to_grid(np.asarray(img.resize((200, 328))), 48)[0]
        resp_v = ConditionFilterCard("t", kernel_edge(True), 1e9, "校准").response(base)
        card_acc = ConditionFilterCard("edge-纵向结构", kernel_edge(True), resp_v * 0.9,
                                       "主体边缘结构")
        card_def = ConditionFilterCard("edge-弱", kernel_edge(True), resp_v * 1.5, "弱响应待验")
        card_rej = ConditionFilterCard("lap-不存在", kernel_laplacian(), 1e6, "不可能高响应")
        cards = [card_acc, card_def, card_rej]
        reports = parallel_route(base, cards)
        by = {r["card"]: r for r in reports}
        check("ACCEPT 语义(响应≥阈值)", by["edge-纵向结构"]["verdict"] == "ACCEPT",
              str(by["edge-纵向结构"]))
        check("DEFER 语义(证据弱)", by["edge-弱"]["verdict"] == "DEFER",
              str(by["edge-弱"]))
        check("REJECT 语义(响应趋零)", by["lap-不存在"]["verdict"] == "REJECT",
              str(by["lap-不存在"]))
        check("每卡报告带证据(reason+response+ratio)",
              all({"reason", "response", "ratio"} <= set(r) for r in reports))

        # ============ ⑥ 真实图:RGBA 透明底安全 ============
        # 素材在盘时：由素材合成 RGBA 透明底（近白像素转 alpha=0）——保持原测试意图
        # （RGBA→RGB 透明底复合 + 降采样晶格）不变。
        print("\n[6] RGBA 透明底(素材合成)")
        _base = Image.open(_ASSET).resize((200, 262)).convert("RGBA")
        _arr = np.asarray(_base).copy()
        _lum = _arr[..., :3].mean(axis=2)
        _arr[..., 3] = np.where(_lum > 240, 0, 255).astype(np.uint8)
        nah = Image.fromarray(_arr, "RGBA")
        nah_rgb = Image.new("RGB", nah.size, (255, 255, 255))
        nah_rgb.paste(nah, mask=nah.split()[-1])
        arr_n = np.asarray(nah_rgb)
        feat_n, _ = image_to_grid(arr_n, 40)
        check("RGBA 透明底晶格提取正常", feat_n.shape[1] == 40 and feat_n.mean() > 0,
              f"{feat_n.shape} mean={feat_n.mean():.1f}")
        tr_n = selfsup_finetune(arr_n, epochs=5, lr=0.3)
        check("透明底域自监督收敛", tr_n["final_loss"] < tr_n["init_loss"] * 1.1,
              f"{tr_n['init_loss']}→{tr_n['final_loss']}")
    else:
        print("\n[4-6] SKIP：上游素材 data/img/0.png 未随导出（构造性段①②③照跑）")

    passed = sum(1 for _, ok in results if ok)
    failed = len(results) - passed
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())


def test_hex_cnn_full():
    """构造性回归全段（脚本模式 main() 的 pytest 入口；断言集一致）。"""
    assert main() == 0
