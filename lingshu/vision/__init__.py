# -*- coding: utf-8 -*-
"""vision · 图像与音频身体感知。

导出符号按需加载：只用标准库 WAV 分析时不导入 numpy/Pillow；图像线稿首次访问
再装载重依赖模块，保持轻核和可选 extras 的边界。
"""
from __future__ import annotations

from importlib import import_module

__all__ = [
    "darkline", "linework", "AudioPerceptionError", "analyze_wav",
    "format_audio_summary", "perceive_wav",
]


def __getattr__(name):
    if name in {"darkline", "linework"}:
        return getattr(import_module(".lineart", __name__), name)
    if name in {"AudioPerceptionError", "analyze_wav", "format_audio_summary", "perceive_wav"}:
        return getattr(import_module(".audio", __name__), name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def __dir__():
    return sorted(set(globals()) | set(__all__))
