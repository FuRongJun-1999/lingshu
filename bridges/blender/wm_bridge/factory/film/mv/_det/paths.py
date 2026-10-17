# -*- coding: utf-8 -*-
"""wm_bridge.factory.film.mv._det.paths · 运行时路径（上游定位 / 输出目录 / 字体）

**为什么单独一层**：执行器要 import 上游 `lingshu.world.*` 才能真正渲染，而本桥的立身之本
是"不 import lingshu，只通过工件对接"。二者不矛盾 —— 但依赖必须**显式、可配置、找不到就报错**，
所以这里**不写机器绝对路径**，而是复用 `factory/_deps.py` 已有的三级解析：

    `WM_LINGSHU_ROOT`（环境变量）
    → 仓库相对候选（逐级向上找 `<祖先>/wm-verify/lingshu`）
    → 都找不到 ⇒ `DepError`（明确失败，不静默降级）

输出目录同理：`MV_OUT` 环境变量优先，缺省 `<cwd>/mv_out`（**不往仓库里写产物**）。
"""
from __future__ import annotations

import os
import sys

from .... import _deps                 # wm_bridge.factory._deps

__all__ = ["DepError", "lingshu_root", "ensure_lingshu", "out_dir", "font_path"]

#: 依赖解析失败时抛的就是它（转导出，调用方不必再 import _deps）
DepError = _deps.DepError

_NOTO = "/usr/share/fonts/opentype/noto"
_DEJAVU = "/usr/share/fonts/truetype/dejavu"


def lingshu_root() -> str:
    """上游 `lingshu` 仓根（绝对路径）。

    解析顺序：`_deps.lingshu_root()`（`WM_LINGSHU_ROOT` → 仓内候选）→ **仓内回退**。
    为什么需要回退：本插件在 CI 与"就在 lingshu 仓里开发"这两种场景下，**当前仓根就是上游**
    （CI 检出的就是 lingshu），此时 `_deps` 的 `<祖先>/wm-verify/lingshu` 候选必然落空。
    若只认 `_deps`，插件在 CI 上会整体 skip —— 守门测试变成永远不跑。
    回退仍不写死路径：直接问已安装/已可导入的 `lingshu` 包它的仓根在哪。
    """
    try:
        return _deps.lingshu_root()
    except _deps.DepError:
        try:
            import lingshu as _pkg
        except ImportError:
            raise
        return os.path.dirname(os.path.dirname(os.path.abspath(_pkg.__file__)))


def ensure_lingshu() -> str:
    """把上游仓根注入 `sys.path`（幂等）并返回它。"""
    root = lingshu_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def out_dir(*parts: str) -> str:
    """产物目录。`MV_OUT` 优先；缺省 `<cwd>/mv_out`。额外段按需拼接。"""
    base = os.environ.get("MV_OUT") or os.path.join(os.getcwd(), "mv_out")
    p = os.path.join(base, *parts) if parts else base
    os.makedirs(p, exist_ok=True)
    return p


def font_path(script: str = "latin", bold: bool = True) -> str:
    """字幕字体。`script` ∈ {"latin","cjk"}。缺字体 ⇒ 返回空串（调用方**如实回退**）。"""
    if script == "cjk":
        cands = [
            os.path.join(_NOTO, "NotoSansCJK-Bold.ttc" if bold else "NotoSansCJK-Regular.ttc"),
            os.path.join(_NOTO, "NotoSerifCJK-Bold.ttc"),
        ]
    else:
        cands = [
            os.path.join(_DEJAVU, "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
            os.path.join(_NOTO, "NotoSansCJK-Regular.ttc"),
        ]
    for c in cands:
        if os.path.isfile(c):
            return c
    return ""
