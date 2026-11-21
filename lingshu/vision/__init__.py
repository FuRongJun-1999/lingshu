# -*- coding: utf-8 -*-
"""vision · 图像、音频与视频帧身体感知。

导出符号按需加载：只用标准库 WAV 分析或视频帧差工具时不导入 numpy/Pillow；图像线稿
与像素统计首次访问再装载重依赖模块，保持轻核和可选 extras 的边界。
"""
from __future__ import annotations

from importlib import import_module

__all__ = ['darkline', 'linework', 'AudioPerceptionError', 'AudioCodecUnavailable', 'analyze_wav', 'analyze_audio', 'decode_compressed_audio', 'ffmpeg_available', 'segment_audio_events', 'infer_audio_activity_period', 'format_audio_summary', 'perceive_wav', 'perceive_audio_file', 'VideoAnalysisError', 'analyze_video_frames', 'ffmpeg_video_available', 'analyze_image_pixels']


def __getattr__(name):
    if name in {"darkline", "linework"}:
        return getattr(import_module(".lineart", __name__), name)
    if name in {"AudioPerceptionError", "AudioCodecUnavailable", "analyze_wav", "analyze_audio",
                "decode_compressed_audio", "ffmpeg_available", "segment_audio_events",
                "infer_audio_activity_period", "format_audio_summary",
                "perceive_wav", "perceive_audio_file"}:
        return getattr(import_module(".audio", __name__), name)
    if name in {"VideoAnalysisError", "analyze_video_frames", "ffmpeg_video_available"}:
        return getattr(import_module(".video", __name__), name)
    if name == "analyze_image_pixels":
        return getattr(import_module(".image_stats", __name__), name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def __dir__():
    return sorted(set(globals()) | set(__all__))
