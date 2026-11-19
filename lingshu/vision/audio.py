# -*- coding: utf-8 -*-
"""标准库 WAV 声学摘要与身体感知接入。

仅读取未压缩 PCM WAV，按有限时长流式计算声学包络、响度与静音区间；不做语音
识别，不保存或上传原始样本。perceive_wav 可把确定性摘要写入引擎的 audio 模态记忆，
原文件路径不会写入记忆节点或返回值。
"""
from __future__ import annotations

import math
import os
import struct
import wave
from typing import Any, Dict, Iterator, List, Union

__all__ = ["AudioPerceptionError", "analyze_wav", "format_audio_summary", "perceive_wav"]

_MAX_DURATION_SECONDS = 3600.0
_MAX_PCM_BYTES = 64 * 1024 * 1024
_MAX_ENVELOPE_POINTS = 1200
_MAX_SILENCE_INTERVALS = 64


class AudioPerceptionError(ValueError):
    """WAV 内容、格式或资源预算不满足身体感知契约。"""


def _finite_number(name: str, value: Any, *, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("%s 必须是有限实数，收到 %r" % (name, value))
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("%s 必须是有限实数，收到 %r" % (name, value))
    if result < low or result > high:
        raise ValueError("%s 须在 [%s, %s] 内，收到 %r" % (name, low, high, value))
    return result


def _pcm_samples(raw: bytes, width: int) -> Iterator[int]:
    """解码 little-endian PCM 样本；调用方已按 sample width 校验数据长度。"""
    if width == 1:
        for value in raw:
            yield value - 128
    elif width == 2:
        count = len(raw) // 2
        yield from struct.unpack("<%dh" % count, raw)
    elif width == 3:
        for i in range(0, len(raw), 3):
            value = raw[i] | (raw[i + 1] << 8) | (raw[i + 2] << 16)
            yield value - (1 << 24) if value & (1 << 23) else value
    elif width == 4:
        count = len(raw) // 4
        yield from struct.unpack("<%di" % count, raw)
    else:  # pragma: no cover - format gate prevents this branch
        raise AudioPerceptionError("不支持的 PCM sample width: %d" % width)


def _dbfs(amplitude: float) -> float:
    if amplitude <= 0.0:
        return -120.0
    return max(-120.0, round(20.0 * math.log10(amplitude), 2))


def analyze_wav(
    path: Union[str, os.PathLike],
    *,
    window_ms: int = 100,
    silence_dbfs: float = -45.0,
    max_duration_s: float = 300.0,
) -> Dict[str, Any]:
    """流式分析未压缩 PCM WAV，返回有界、可 JSON 序列化的声学特征。

    支持 8/16/24/32-bit PCM，最多 64 MiB，默认最多分析 5 分钟。静音按每个
    分析窗的 RMS 与 silence_dbfs 比较。输出不含路径、文件名或原始音频，不做转写。
    """
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError("path 必须是 WAV 文件路径，收到 %r" % (path,))
    path_text = os.fsdecode(os.fspath(path))
    if not path_text.strip():
        raise ValueError("path 不能为空")
    if isinstance(window_ms, bool) or not isinstance(window_ms, int):
        raise TypeError("window_ms 必须是整数，收到 %r" % (window_ms,))
    if not 10 <= window_ms <= 1000:
        raise ValueError("window_ms 须在 [10, 1000] 内，收到 %r" % (window_ms,))
    threshold_db = _finite_number("silence_dbfs", silence_dbfs, low=-120.0, high=0.0)
    duration_cap = _finite_number("max_duration_s", max_duration_s, low=0.01,
                                  high=_MAX_DURATION_SECONDS)

    try:
        with wave.open(path_text, "rb") as reader:
            channels = reader.getnchannels()
            sample_rate = reader.getframerate()
            width = reader.getsampwidth()
            frame_count = reader.getnframes()
            if reader.getcomptype() != "NONE":
                raise AudioPerceptionError("仅支持未压缩 PCM WAV")
            if channels < 1 or sample_rate < 1 or frame_count < 1:
                raise AudioPerceptionError("WAV 必须包含正的声道数、采样率和音频帧")
            if channels > 8:
                raise AudioPerceptionError("WAV 声道数超过支持上限 8：%d" % channels)
            if width not in (1, 2, 3, 4):
                raise AudioPerceptionError("PCM sample width 仅支持 8/16/24/32-bit")
            duration = frame_count / sample_rate
            if duration > duration_cap:
                raise AudioPerceptionError("音频时长 %.3fs 超出限制 %.3fs" % (duration, duration_cap))
            expected_bytes = frame_count * channels * width
            if expected_bytes > _MAX_PCM_BYTES:
                raise AudioPerceptionError("PCM 数据超过 64 MiB 读取上限")

            window_frames = max(1, int(round(sample_rate * window_ms / 1000.0)))
            scale = float(1 << (8 * width - 1))
            silent_threshold = 10.0 ** (threshold_db / 20.0)
            frame_pos = 0
            silent_frames = 0
            sample_total = 0
            square_total = 0.0
            peak = 0.0
            clipped = 0
            envelope: List[float] = []
            intervals: List[Dict[str, float]] = []
            interval_count = 0
            silence_start = None

            while frame_pos < frame_count:
                nframes = min(window_frames, frame_count - frame_pos)
                raw = reader.readframes(nframes)
                expected = nframes * channels * width
                if len(raw) != expected:
                    raise AudioPerceptionError("WAV 数据截断：期望 %d 字节，实际 %d" % (expected, len(raw)))
                window_sq = 0.0
                window_samples = 0
                window_peak = 0.0
                window_clipped = 0
                for sample in _pcm_samples(raw, width):
                    value = sample / scale
                    magnitude = abs(value)
                    window_sq += value * value
                    window_peak = max(window_peak, magnitude)
                    if sample >= int(scale) - 1 or sample <= -int(scale):
                        window_clipped += 1
                    window_samples += 1
                if window_samples == 0:
                    raise AudioPerceptionError("WAV 音频帧为空")
                window_rms = math.sqrt(window_sq / window_samples)
                envelope.append(_dbfs(window_rms))
                square_total += window_sq
                sample_total += window_samples
                peak = max(peak, window_peak)
                clipped += window_clipped
                is_silent = window_rms <= silent_threshold
                if is_silent:
                    silent_frames += nframes
                    if silence_start is None:
                        silence_start = frame_pos
                elif silence_start is not None:
                    interval_count += 1
                    if len(intervals) < _MAX_SILENCE_INTERVALS:
                        intervals.append({"start_s": round(silence_start / sample_rate, 4),
                                          "end_s": round(frame_pos / sample_rate, 4)})
                    silence_start = None
                frame_pos += nframes

            if silence_start is not None:
                interval_count += 1
                if len(intervals) < _MAX_SILENCE_INTERVALS:
                    intervals.append({"start_s": round(silence_start / sample_rate, 4),
                                      "end_s": round(frame_count / sample_rate, 4)})
    except (wave.Error, EOFError, struct.error) as exc:
        raise AudioPerceptionError("无效或不支持的 WAV 文件：%s" % exc) from exc

    stride = max(1, int(math.ceil(len(envelope) / _MAX_ENVELOPE_POINTS)))
    envelope = envelope[::stride]
    rms = math.sqrt(square_total / sample_total) if sample_total else 0.0
    silence_ratio = silent_frames / frame_count
    return {
        "format": "wav-pcm",
        "duration_s": round(duration, 4),
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "sample_width_bits": width * 8,
        "frames": frame_count,
        "window_ms": window_ms,
        "rms_dbfs": _dbfs(rms),
        "peak_dbfs": _dbfs(peak),
        "silence_threshold_dbfs": threshold_db,
        "silence_ratio": round(silence_ratio, 6),
        "activity_ratio": round(1.0 - silence_ratio, 6),
        "clipped_sample_ratio": round(clipped / sample_total, 8),
        "silence_intervals": intervals,
        "silence_interval_count": interval_count,
        "silence_intervals_truncated": interval_count > len(intervals),
        "energy_envelope_dbfs": envelope,
        "energy_envelope_stride_windows": stride,
    }


def format_audio_summary(features: Dict[str, Any]) -> str:
    """把确定性特征转成短摘要，不含文件名或路径。"""
    return (
        "WAV 声学摘要：时长 %.2f 秒，采样率 %d Hz，%d 声道，%d-bit PCM；"
        "RMS %s dBFS，峰值 %s dBFS，活动比例 %.1f%%，静音区间 %d 段。"
        % (features["duration_s"], features["sample_rate_hz"], features["channels"],
           features["sample_width_bits"], features["rms_dbfs"], features["peak_dbfs"],
           100.0 * features["activity_ratio"], features["silence_interval_count"])
    )


def perceive_wav(
    engine: Any,
    path: Union[str, os.PathLike],
    *,
    importance: float = 0.4,
    window_ms: int = 100,
    silence_dbfs: float = -45.0,
    max_duration_s: float = 300.0,
) -> Dict[str, Any]:
    """分析 WAV 并把声学摘要写入引擎的 audio 模态记忆。

    只存摘要，不存原文件、文件名或路径。engine 需实现 add_perception 接口。
    """
    if not callable(getattr(engine, "add_perception", None)):
        raise TypeError("engine 必须提供 add_perception(content, modality=...) 方法")
    importance_value = _finite_number("importance", importance, low=0.0, high=1.0)
    features = analyze_wav(path, window_ms=window_ms, silence_dbfs=silence_dbfs,
                           max_duration_s=max_duration_s)
    summary = format_audio_summary(features)
    node = engine.add_perception(summary, modality="audio", importance=importance_value,
                                  tags=["audio_summary"], source="body:wav-acoustic-summary")
    return {"status": "ok", "node_id": getattr(node, "id", None),
            "summary": summary, "features": features, "raw_audio_stored": False}
