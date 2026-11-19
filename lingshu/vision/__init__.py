# -*- coding: utf-8 -*-
"""vision · 惰性图像和视频分析公共入口。"""
from __future__ import annotations

from importlib import import_module

__all__ = ["darkline", "linework", "VideoAnalysisError", "analyze_video_frames", "ffmpeg_video_available"]


def __getattr__(name):
    if name in {"darkline", "linework"}:
        return getattr(import_module(".lineart", __name__), name)
    if name in {"VideoAnalysisError", "analyze_video_frames", "ffmpeg_video_available"}:
        return getattr(import_module(".video", __name__), name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def __dir__():
    return sorted(set(globals()) | set(__all__))
