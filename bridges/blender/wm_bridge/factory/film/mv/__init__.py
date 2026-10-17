# -*- coding: utf-8 -*-
"""wm_bridge.factory.film.mv · **MV 制作插件**（影视层 P2–P6 的执行器）

`film/` 立的是**契约与判据**（P0/P1：七份 `wmfilm.*`/`wmmv.*` + K 族/E1 闸）。
本插件把那份"可判定"变成**可交付的成片**：

    brief.json（创意简报，LLM 编译期的唯一产物）
        │
        ├─ P3 audio     乐谱（拍点/强拍/段落/能量）+ 程序化音轨        → wmmv.score/1
        ├─ P4 bind      能量 → 景别/ASL/运动量 + 卡点规则              → wmmv.bind/1
        ├─ P5 edit      全片镜头表（切点落强拍）+ 剪辑时间轴            → wmfilm.shot/1 × N, wmmv.edit/1
        ├─ P5 text      歌词上屏（CPS/行宽/安全区由契约层硬拦）         → wmmv.text/1
        ├─ P2 scene     建模（mesh_gen）· 表演（skeleton3d）· 物理（physics）· 着色（shade_paint）
        ├─ P2 camera    取景标定：`compose` 契约 → 逐帧 `Camera3D`
        ├─ P2 render    逐帧渲染 + 绘画后处理（天空/雾/调色/bloom/vignette/颗粒）+ 字幕叠层
        ├─ §六 oracle   从**渲出来的像素**回读真值（覆盖率/包围盒/剪影 ΔL/直方图/色温）
        ├─ P1 judge     把输入束交给 `film/_det/gate_all` 裁决         → wmfilm.oracle/1
        └─ P6 encode    AVI(MJPG) / 可选 MP4(H.264+AAC) / GIF / SRT / manifest
                                        ↓
                             一支完整 MV（视频 + 音轨 + 字幕 + 清单 + 真值裁定）

三条不可让的纪律（与 `film/__init__.py` 同源）：

* **意图与执行分离**：LLM 只在编译期（写 `brief.json`）出现；执行期零 LLM。
* **真值外部性**：判据吃**渲染器回读的事实**，不吃声明。执行器不给自己打分 ——
  分数由 P1 的判据门出（见 `_det/gates.py`）。
* **四态贯穿**：缺料一律 BLINDSPOT，绝不静默 ACCEPT。

边界：B1 不改 `lingshu/world/`；B2 不把 bpy 带进 lingshu；B3 原版插件零改动；
B4 契约层（`film/*.py` 与本包的 `__init__`）**纯标准库可导入** —— 执行器
（`_det/*`）才依赖上游渲染所需的 PIL / numpy，且通过 `factory/_deps.py` 显式定位。
"""
from __future__ import annotations

import os

__all__ = ["PLUGIN_ID", "VERSION", "PIPELINE", "DEFAULT_BRIEF",
           "make_mv", "judge", "accept", "summarize"]

PLUGIN_ID = "wmfilm.mv"
VERSION = "1.0.0"

#: 阶段登记表：(阶段, 模块, 产出契约, 做什么)
PIPELINE = (
    ("P3", "audio", "wmmv.score/1", "乐谱 + 程序化音轨"),
    ("P4", "timeline", "wmmv.bind/1", "能量 → 景别/ASL/运动量"),
    ("P2/P5", "timeline", "wmfilm.shot/1", "全片镜头表（切点落强拍）"),
    ("P5", "timeline", "wmmv.edit/1", "剪辑时间轴（首尾相接、对齐切点）"),
    ("P5", "timeline", "wmmv.text/1", "歌词上屏"),
    ("P2", "scene", "-", "建模 / 表演 / 物理 / 着色"),
    ("P2", "camera", "-", "取景标定（compose 契约 → 逐帧相机）"),
    ("P2", "render", "-", "逐帧渲染 + 绘画后处理 + 字幕叠层"),
    ("§六", "oracle", "-", "视听真值回读"),
    ("P1", "gates", "wmfilm.oracle/1", "交给 P1 判据门裁决（复用 film/_det/gate_all）"),
    ("P6", "encode", "-", "AVI / MP4 / GIF / SRT / manifest"),
)

DEFAULT_BRIEF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "briefs", "night_walk.json")


def _load_brief(brief):
    """接受 dict 或 JSON 路径。"""
    import json
    if isinstance(brief, dict):
        return brief
    with open(brief, "r", encoding="utf-8") as f:
        return json.load(f)


def make_mv(brief=None, out=None, size=None, jobs=1, ssaa=1, gif=True,
            log=print, **kw):
    """产出一支完整 MV。`brief` 缺省用 `briefs/night_walk.json`。

    返回 `pipeline.build_all` 的结果字典（frames / manifest / oracle / artifacts / tally）。
    """
    from ._det import pipeline
    b = _load_brief(brief if brief is not None else DEFAULT_BRIEF)
    return pipeline.build_all(b, out or _default_out(), size=size, jobs=jobs,
                              ssaa=ssaa, do_gif=gif, log=log, **kw)


def judge(out=None, brief=None, log=print):
    """只重跑判据（不重渲）：从已存帧重算真值与裁定。"""
    from ._det import pipeline
    b = _load_brief(brief if brief is not None else DEFAULT_BRIEF)
    return pipeline.regates(b, out or _default_out(), log=log)


def accept(out=None, brief=None, sample=5):
    """冻结态验收（复算清单/确定性/判据不变式/字幕/契约）。"""
    from ._det import acceptance
    b = _load_brief(brief if brief is not None else DEFAULT_BRIEF)
    return acceptance.run(out or _default_out(), b, sample)


def summarize(oracle_doc):
    """一行汇总（判据票型 + 是否全绿）。"""
    from ._det import gates
    counts = {}
    for c in oracle_doc.get("checks") or []:
        counts[c.get("verdict")] = counts.get(c.get("verdict"), 0) + 1
    return {"counts": counts,
            "blocking": [c["name"] for c in oracle_doc.get("checks") or []
                         if c.get("verdict") in ("REJECT",)],
            "thresholds": len(gates.all_thresholds())}


def _default_out() -> str:
    from ._det import paths
    return paths.out_dir()
