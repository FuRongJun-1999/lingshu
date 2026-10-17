# -*- coding: utf-8 -*-
"""wm_bridge.factory.film.mv._det · MV 执行器（P2–P6）的确定性实现

与 `film/_det/` 的关系：那一层是**判据**（P1，纯标准库、只吃输入束），本层是**执行器**
（P2–P6，真的调 `lingshu.world` 渲染、合成音频、编码视频）。判据仍由
`film/_det/gate_all` 裁决，本层只负责把产物装配成它要的输入束（见 `gates.py`）。

模块一览（每个模块一个阶段）：
    audio.py      P3  乐谱 + 程序化音轨（wmmv.score/1）
    scene.py      P2  场景与角色（mesh_gen / skeleton3d / physics / shade_paint）
    camera.py     P2  CameraRig（取景标定：compose 契约的落地）
    render.py     P2  逐帧渲染 + 绘画后处理（天空/雾/调色/bloom/vignette/颗粒）
    timeline.py   P4/P5  绑定 · 镜头表 · 剪辑 · 文本（bind/edit/text 三契约）
    oracle.py     §六 视听真值回读（腿 C：渲染器 → 判据库）
    gates.py      适配  执行器输入束 → P1 输入束（含 Y-up↔Z-up 口径桥）
    encode.py     P6  视频/清单出口（手写 MVI/MJPG + 可选 MP4 + GIF）
    preview.py    P6  小体积预览 GIF
    pipeline.py   ——  编排（brief → 全片 → 判据 → 产物）
    acceptance.py ——  冻结态验收（复算，不采信转述）
    paths.py      ——  上游定位（复用 factory/_deps）
    contracts.py  ——  契约族委托层（不重复实现校验）
"""
from __future__ import annotations

__all__ = ["audio", "scene", "camera", "render", "timeline", "oracle",
           "gates", "encode", "preview", "pipeline", "acceptance",
           "paths", "contracts"]
