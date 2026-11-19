# -*- coding: utf-8 -*-
"""Bounded, optional-FFmpeg grayscale video-frame difference analysis.

This reports pixel changes on a fixed sampling grid only. It does not identify scenes,
objects, speech, actions, or persistent video memories. Frames stay in bounded memory and
are never returned or written to disk.
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
from typing import Any, Dict, Union

__all__ = ["VideoAnalysisError", "ffmpeg_video_available", "analyze_video_frames"]

_MAX_INPUT_BYTES = 64 * 1024 * 1024
_MAX_DURATION_SECONDS = 600.0
_MAX_TIMEOUT_SECONDS = 600.0
_FRAME_WIDTH = 64
_FRAME_HEIGHT = 64
_FRAME_BYTES = _FRAME_WIDTH * _FRAME_HEIGHT
_MAX_SAMPLE_FRAMES = 120
_VIDEO_DEMUXERS = {
    ".mp4": "mov", ".m4v": "mov", ".mov": "mov",
    ".mkv": "matroska", ".webm": "matroska",
    ".avi": "avi", ".mpeg": "mpeg", ".mpg": "mpeg", ".ts": "mpegts",
}


class VideoAnalysisError(ValueError):
    """Invalid, unsupported, or over-budget local video input."""


def ffmpeg_video_available(ffmpeg: str = "ffmpeg") -> bool:
    """Return whether an optional FFmpeg executable is resolvable on PATH."""
    return isinstance(ffmpeg, str) and bool(ffmpeg.strip()) and shutil.which(ffmpeg) is not None


def _demuxer_from_signature(data: bytes) -> str:
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return "mov"
    if data.startswith(b"\x1a\x45\xdf\xa3"):
        return "matroska"
    if len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"AVI ":
        return "avi"
    if data.startswith(b"\x00\x00\x01\xba") or data.startswith(b"\x00\x00\x01\xb3"):
        return "mpeg"
    if ((len(data) >= 376 and data[0] == 0x47 and data[188] == 0x47)
            or (len(data) >= 388 and data[4] == 0x47 and data[196] == 0x47)):
        return "mpegts"
    raise VideoAnalysisError("无法识别受支持的本地视频容器")


def _read_bounded_source(source: Union[str, os.PathLike, bytes, bytearray, memoryview]) -> tuple[bytes, str]:
    if isinstance(source, (bytes, bytearray, memoryview)):
        size = source.nbytes if isinstance(source, memoryview) else len(source)
        if size == 0:
            raise VideoAnalysisError("视频 bytes 不能为空")
        if size > _MAX_INPUT_BYTES:
            raise VideoAnalysisError("视频输入超过 64 MiB 上限")
        data = bytes(source)
        return data, _demuxer_from_signature(data)
    if not isinstance(source, (str, os.PathLike)):
        raise TypeError("source 必须是本地视频路径或 bytes-like 数据")
    path = os.fsdecode(os.fspath(source))
    if not path.strip():
        raise VideoAnalysisError("视频路径不能为空")
    if "://" in path or not os.path.isfile(path):
        raise VideoAnalysisError("视频 source 必须是存在的本地文件")
    extension = os.path.splitext(path)[1].lower()
    demuxer = _VIDEO_DEMUXERS.get(extension)
    if demuxer is None:
        raise VideoAnalysisError("不支持的视频扩展名：%s" % (extension or "(无扩展名)"))
    try:
        with open(path, "rb") as stream:
            data = stream.read(_MAX_INPUT_BYTES + 1)
    except OSError as exc:
        raise VideoAnalysisError("无法读取本地视频文件") from exc
    if len(data) > _MAX_INPUT_BYTES:
        raise VideoAnalysisError("视频输入超过 64 MiB 上限")
    signature_demuxer = _demuxer_from_signature(data)
    if signature_demuxer != demuxer and not (demuxer == "mov" and signature_demuxer == "mov"):
        raise VideoAnalysisError("视频容器签名与文件扩展名不匹配")
    return data, demuxer


def analyze_video_frames(
    source: Union[str, os.PathLike, bytes, bytearray, memoryview],
    *,
    ffmpeg: str = "ffmpeg",
    sample_fps: float = 1.0,
    max_frames: int = 120,
    max_duration_s: float = 60.0,
    timeout_s: float = 60.0,
) -> Dict[str, Any]:
    """Sample local video onto a bounded 64x64 grayscale grid and measure frame changes.

    FFmpeg is an optional executable dependency. This function requires a recognized local
    container with a video stream, restricts FFmpeg to stdin/pipe, and discards stderr so
    paths/metadata are not returned. It returns JSON-safe statistics only; decoded pixels
    remain in memory. Sampling timestamps are grid times (index / sample_fps), not source
    packet presentation timestamps.
    """
    if not isinstance(ffmpeg, str) or not ffmpeg.strip():
        raise TypeError("ffmpeg 必须是非空可执行文件名或路径")
    fps = float(sample_fps) if isinstance(sample_fps, (int, float)) and not isinstance(sample_fps, bool) else math.nan
    if not math.isfinite(fps) or not 0.1 <= fps <= 10.0:
        raise ValueError("sample_fps 必须是 [0.1, 10] 内的有限数值")
    if isinstance(max_frames, bool) or not isinstance(max_frames, int) or not 1 <= max_frames <= _MAX_SAMPLE_FRAMES:
        raise ValueError("max_frames 必须是 [1, 120] 内的整数")
    duration = float(max_duration_s) if isinstance(max_duration_s, (int, float)) and not isinstance(max_duration_s, bool) else math.nan
    if not math.isfinite(duration) or not 0.1 <= duration <= _MAX_DURATION_SECONDS:
        raise ValueError("max_duration_s 必须是 [0.1, 600] 内的有限数值")
    timeout = float(timeout_s) if isinstance(timeout_s, (int, float)) and not isinstance(timeout_s, bool) else math.nan
    if not math.isfinite(timeout) or not 0.1 <= timeout <= _MAX_TIMEOUT_SECONDS:
        raise ValueError("timeout_s 必须是 [0.1, 600] 内的有限数值")

    data, demuxer = _read_bounded_source(source)
    executable = shutil.which(ffmpeg)
    if executable is None:
        raise VideoAnalysisError("视频帧分析需要可选 FFmpeg 可执行文件")
    filter_chain = (
        "fps=fps=%.6f:start_time=0,"
        "scale=%d:%d:force_original_aspect_ratio=decrease,"
        "pad=%d:%d:(ow-iw)/2:(oh-ih)/2,format=gray"
    ) % (fps, _FRAME_WIDTH, _FRAME_HEIGHT, _FRAME_WIDTH, _FRAME_HEIGHT)
    requested_frames = max_frames + 1
    output_limit = _FRAME_BYTES * requested_frames
    command = [
        executable, "-nostdin", "-hide_banner", "-loglevel", "error",
        "-protocol_whitelist", "pipe", "-f", demuxer, "-i", "pipe:0",
        "-map", "0:v:0", "-an", "-sn", "-dn", "-t", "%.6f" % duration,
        "-vf", filter_chain, "-frames:v", str(requested_frames),
        "-fps_mode", "passthrough", "-pix_fmt", "gray", "-f", "rawvideo",
        "-fs", str(output_limit), "pipe:1",
    ]
    try:
        completed = subprocess.run(command, input=data, stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL, timeout=timeout,
                                   check=False, shell=False)
    except subprocess.TimeoutExpired as exc:
        raise VideoAnalysisError("视频帧采样超过时间限制") from exc
    except OSError as exc:
        raise VideoAnalysisError("FFmpeg 可执行文件无法启动") from exc
    if completed.returncode != 0:
        raise VideoAnalysisError("FFmpeg 无法从该视频提取帧")
    raw = completed.stdout
    if len(raw) > output_limit:
        raise VideoAnalysisError("抽取帧字节超过预算")
    if len(raw) % _FRAME_BYTES:
        raise VideoAnalysisError("FFmpeg 输出了不完整的灰度帧")
    decoded_count = len(raw) // _FRAME_BYTES
    if decoded_count == 0:
        raise VideoAnalysisError("视频没有可读取的视频帧")
    truncated = decoded_count > max_frames
    frame_count = min(decoded_count, max_frames)
    frames = [memoryview(raw)[i * _FRAME_BYTES:(i + 1) * _FRAME_BYTES]
              for i in range(frame_count)]
    differences = []
    for index in range(1, frame_count):
        previous, current = frames[index - 1], frames[index]
        total = sum(abs(a - b) for a, b in zip(previous, current))
        differences.append(total / (_FRAME_BYTES * 255.0))
    mean_difference = (sum(differences) / len(differences)) if differences else None
    sample_interval = 1.0 / fps
    return {
        "status": "ok",
        "container_demuxer": demuxer,
        "frame_count": frame_count,
        "sample_fps": fps,
        "sample_interval_s": round(sample_interval, 9),
        "sample_time_basis": "sampling_grid_seconds",
        "sample_times_s": [round(i * sample_interval, 6) for i in range(frame_count)],
        "frame_width": _FRAME_WIDTH,
        "frame_height": _FRAME_HEIGHT,
        "grayscale": True,
        "mean_adjacent_frame_difference": (round(mean_difference, 6)
                                            if mean_difference is not None else None),
        "adjacent_frame_differences": [
            {"from_s": round(i * sample_interval, 6),
             "to_s": round((i + 1) * sample_interval, 6),
             "mean_abs_diff_0_1": round(value, 6)}
            for i, value in enumerate(differences)
        ],
        "truncated_by_frame_limit": truncated,
        "decoded_frame_byte_count": frame_count * _FRAME_BYTES,
    }

