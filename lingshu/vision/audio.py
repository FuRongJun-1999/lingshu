# -*- coding: utf-8 -*-
"""标准库 WAV/可选 FFmpeg 音频声学与事件画像身体感知。

WAV 以标准库流式读取；压缩格式按需调用可选 FFmpeg，结果仅在内存解码为有界 PCM；
segment_audio_events 从包络生成粗粒度 activity/silence 片段，明确报告时间分辨率；
不做语音识别或语义事件分类，不保存或上传原始样本。
"""
from __future__ import annotations

import io
import math
import os
import shutil
import struct
import subprocess
import wave
from typing import Any, Dict, Iterator, List, Union

__all__ = ["AudioPerceptionError", "AudioCodecUnavailable", "analyze_wav", "analyze_audio", "decode_compressed_audio", "ffmpeg_available", "segment_audio_events", "infer_audio_activity_period", "format_audio_summary", "perceive_wav", "perceive_audio_file"]

_MAX_DURATION_SECONDS = 3600.0
_MAX_PCM_BYTES = 64 * 1024 * 1024
_MAX_ENVELOPE_POINTS = 1200
_MAX_SILENCE_INTERVALS = 64


class AudioPerceptionError(ValueError):
    """WAV/音频内容、格式或资源预算不满足身体感知契约。"""


class AudioCodecUnavailable(AudioPerceptionError):
    """可选压缩音频解码器不可用。"""


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
    path: Union[str, os.PathLike, bytes, bytearray, memoryview],
    *,
    window_ms: int = 100,
    silence_dbfs: float = -45.0,
    max_duration_s: float = 300.0,
) -> Dict[str, Any]:
    """流式分析未压缩 PCM WAV，返回有界、可 JSON 序列化的声学特征。

    path 支持文件路径或 WAV bytes/bytearray/memoryview（适合内存采集流，无需临时文件）。
    支持 8/16/24/32-bit PCM，最多 64 MiB，默认最多分析 5 分钟。输出不含原始音频，
    不做转写。
    """
    if isinstance(path, (bytes, bytearray, memoryview)):
        byte_count = path.nbytes if isinstance(path, memoryview) else len(path)
        if byte_count == 0:
            raise ValueError("WAV bytes 不能为空")
        if byte_count > _MAX_PCM_BYTES:
            raise AudioPerceptionError("WAV bytes 超过 64 MiB 读取上限")
        wav_source = io.BytesIO(bytes(path))
    else:
        if not isinstance(path, (str, os.PathLike)):
            raise TypeError("path 必须是 WAV 路径或二进制 WAV 数据，收到 %r" % (path,))
        path_text = os.fsdecode(os.fspath(path))
        if not path_text.strip():
            raise ValueError("path 不能为空")
        wav_source = path_text
    if isinstance(window_ms, bool) or not isinstance(window_ms, int):
        raise TypeError("window_ms 必须是整数，收到 %r" % (window_ms,))
    if not 10 <= window_ms <= 1000:
        raise ValueError("window_ms 须在 [10, 1000] 内，收到 %r" % (window_ms,))
    threshold_db = _finite_number("silence_dbfs", silence_dbfs, low=-120.0, high=0.0)
    duration_cap = _finite_number("max_duration_s", max_duration_s, low=0.01,
                                  high=_MAX_DURATION_SECONDS)

    try:
        with wave.open(wav_source, "rb") as reader:
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
        "analysis_window_frames": window_frames,
        "analysis_window_s": round(window_frames / sample_rate, 9),
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


_MAX_COMPRESSED_INPUT_BYTES = 64 * 1024 * 1024
_MAX_DECODED_PCM_BYTES = 64 * 1024 * 1024
# Force a known demuxer. Besides producing predictable decoding, this avoids letting
# arbitrary playlist/container inputs make FFmpeg open secondary local/network resources.
_COMPRESSED_DEMUXERS = {
    ".mp3": "mp3", ".m4a": "ipod", ".mp4": "ipod", ".aac": "aac",
    ".ogg": "ogg", ".opus": "ogg", ".flac": "flac",
    ".aif": "aiff", ".aiff": "aiff", ".wma": "asf", ".webm": "matroska",
}


def _is_wav_bytes(data: bytes) -> bool:
    return len(data) >= 12 and data[:4] in (b"RIFF", b"RIFX", b"RF64") and data[8:12] == b"WAVE"


def _compressed_demuxer_from_path(path: str) -> str:
    extension = os.path.splitext(path)[1].lower()
    demuxer = _COMPRESSED_DEMUXERS.get(extension)
    if demuxer is None:
        raise AudioPerceptionError("不支持的压缩音频扩展名：%s" % (extension or "(无扩展名)"))
    return demuxer


def _compressed_demuxer_from_bytes(data: bytes) -> str:
    if data.startswith(b"ADIF") or (
        len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xF6) == 0xF0
    ):
        return "aac"
    if data.startswith(b"ID3") or (
        len(data) >= 2
        and data[0] == 0xFF
        and (data[1] & 0xE0) == 0xE0
        and (data[1] & 0x18) != 0x08
        and (data[1] & 0x06) != 0
    ):
        return "mp3"
    if data.startswith(b"fLaC"):
        return "flac"
    if data.startswith(b"OggS"):
        return "ogg"
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return "ipod"
    if data.startswith(b"FORM") and data[8:12] in (b"AIFF", b"AIFC"):
        return "aiff"
    if data.startswith(b"\x1a\x45\xdf\xa3"):
        return "matroska"
    if data.startswith(b"\x30\x26\xb2\x75\x8e\x66\xcf\x11"):
        return "asf"
    raise AudioPerceptionError("无法从音频 bytes 识别受支持的压缩格式")


def decode_compressed_audio(
    source: Union[str, os.PathLike, bytes, bytearray, memoryview],
    *,
    ffmpeg: str = "ffmpeg",
    max_duration_s: float = 300.0,
    timeout_s: float = 90.0,
) -> bytes:
    """Decode a local compressed audio file/bytes to bounded mono 8 kHz PCM WAV.

    FFmpeg is an optional executable dependency, never imported by the WAV-only path.
    No shell is used; network-style path inputs are rejected and ffmpeg protocols are
    restricted to file/pipe. Decoder stderr is discarded so user paths/metadata are not
    echoed into logs or raised exceptions. Returned audio is in memory only.
    """
    if not isinstance(ffmpeg, str) or not ffmpeg.strip():
        raise TypeError("ffmpeg 必须是非空可执行文件名或路径")
    duration_cap = _finite_number("max_duration_s", max_duration_s, low=0.01,
                                  high=_MAX_DURATION_SECONDS)
    timeout_value = _finite_number("timeout_s", timeout_s, low=0.1, high=600.0)
    if isinstance(source, (bytes, bytearray, memoryview)):
        source_size = source.nbytes if isinstance(source, memoryview) else len(source)
        if source_size > _MAX_COMPRESSED_INPUT_BYTES:
            raise AudioPerceptionError("压缩音频输入超过 64 MiB 上限")
        data = bytes(source)
        if not data:
            raise ValueError("压缩音频 bytes 不能为空")
        if len(data) > _MAX_COMPRESSED_INPUT_BYTES:
            raise AudioPerceptionError("压缩音频输入超过 64 MiB 上限")
        demuxer = _compressed_demuxer_from_bytes(data)
    else:
        if not isinstance(source, (str, os.PathLike)):
            raise TypeError("source 必须是本地音频路径或 bytes-like 数据")
        source_path = os.fsdecode(os.fspath(source))
        if not source_path.strip():
            raise ValueError("source 路径不能为空")
        if "://" in source_path or not os.path.isfile(source_path):
            raise AudioPerceptionError("压缩音频 source 必须是存在的本地文件")
        try:
            with open(source_path, "rb") as fh:
                data = fh.read(_MAX_COMPRESSED_INPUT_BYTES + 1)
        except OSError as exc:
            raise AudioPerceptionError("无法读取压缩音频输入") from exc
        if len(data) > _MAX_COMPRESSED_INPUT_BYTES:
            raise AudioPerceptionError("压缩音频输入超过 64 MiB 上限")
        demuxer = _compressed_demuxer_from_bytes(data)
    resolved = shutil.which(ffmpeg)
    if resolved is None:
        raise AudioCodecUnavailable("压缩音频需可选 FFmpeg 可执行文件；WAV PCM 不需要 FFmpeg")
    # Read the local file into the already bounded buffer and feed stdin. FFmpeg then
    # has no file/network protocol available to follow secondary playlist references.
    command = [resolved, "-nostdin", "-hide_banner", "-loglevel", "error",
               "-protocol_whitelist", "pipe", "-f", demuxer, "-i", "pipe:0",
               "-map", "0:a:0", "-vn", "-sn", "-dn", "-t", "%.6f" % duration_cap,
               "-ac", "1", "-ar", "8000", "-c:a", "pcm_s16le",
               "-fs", str(_MAX_DECODED_PCM_BYTES - 65536), "-f", "s16le", "pipe:1"]
    try:
        completed = subprocess.run(command, input=data, stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL, timeout=timeout_value,
                                   check=False, shell=False)
    except subprocess.TimeoutExpired as exc:
        raise AudioPerceptionError("FFmpeg 音频解码超过时间限制") from exc
    except OSError as exc:
        raise AudioCodecUnavailable("FFmpeg 可执行文件无法启动") from exc
    if completed.returncode != 0:
        raise AudioPerceptionError("FFmpeg 无法解码该压缩音频")
    pcm = completed.stdout
    if not pcm or len(pcm) > _MAX_DECODED_PCM_BYTES or len(pcm) % 2:
        raise AudioPerceptionError("解码后的 PCM 为空、长度无效或超过 64 MiB 上限")
    # FFmpeg WAV-to-pipe output may carry an unknown-size sentinel header. Wrap the
    # bounded raw PCM ourselves so the standard-library reader sees an exact frame count.
    output = io.BytesIO()
    try:
        with wave.open(output, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(8000)
            writer.writeframes(pcm)
    except (wave.Error, OSError) as exc:
        raise AudioPerceptionError("无法构造解码后的 PCM WAV") from exc
    return output.getvalue()


def analyze_audio(
    source: Union[str, os.PathLike, bytes, bytearray, memoryview],
    *,
    ffmpeg: str = "ffmpeg",
    window_ms: int = 100,
    silence_dbfs: float = -45.0,
    max_duration_s: float = 300.0,
    timeout_s: float = 90.0,
) -> Dict[str, Any]:
    """Analyze WAV directly, or optionally decode compressed local audio via FFmpeg.

    WAV paths and RIFF/RIFX/RF64 bytes remain on the standard-library path. Other
    local audio files/bytes are decoded to mono 8 kHz PCM in memory before analysis.
    """
    is_wav = False
    demuxer = "wav"
    if isinstance(source, (bytes, bytearray, memoryview)):
        source_size = source.nbytes if isinstance(source, memoryview) else len(source)
        if source_size > _MAX_COMPRESSED_INPUT_BYTES:
            raise AudioPerceptionError("音频输入超过 64 MiB 上限")
        is_wav = _is_wav_bytes(bytes(source))
        if not is_wav:
            demuxer = _compressed_demuxer_from_bytes(bytes(source))
    elif isinstance(source, (str, os.PathLike)):
        source_path = os.fsdecode(os.fspath(source))
        extension = os.path.splitext(source_path)[1].lower()
        is_wav = extension in (".wav", ".wave")
        if not is_wav:
            demuxer = _compressed_demuxer_from_path(source_path)
    else:
        raise TypeError("source 必须是本地音频路径或 bytes-like 数据")
    if is_wav:
        result = analyze_wav(source, window_ms=window_ms, silence_dbfs=silence_dbfs,
                              max_duration_s=max_duration_s)
        result["decoded_from_compressed"] = False
        result["source_audio_format"] = "wav"
        return result
    decoded = decode_compressed_audio(source, ffmpeg=ffmpeg,
                                      max_duration_s=max_duration_s, timeout_s=timeout_s)
    result = analyze_wav(decoded, window_ms=window_ms, silence_dbfs=silence_dbfs,
                         max_duration_s=max_duration_s)
    result["decoded_from_compressed"] = True
    result["source_audio_format"] = demuxer
    result["decode_output_sample_rate_hz"] = 8000
    result["decode_output_channels"] = 1
    return result


def perceive_audio_file(
    engine: Any,
    source: Union[str, os.PathLike, bytes, bytearray, memoryview],
    *,
    importance: float = 0.4,
    ffmpeg: str = "ffmpeg",
    window_ms: int = 100,
    silence_dbfs: float = -45.0,
    max_duration_s: float = 300.0,
    timeout_s: float = 90.0,
) -> Dict[str, Any]:
    """Analyze WAV or optional FFmpeg-supported audio and persist only its summary."""
    if not callable(getattr(engine, "add_perception", None)):
        raise TypeError("engine 必须提供 add_perception(content, modality=...) 方法")
    importance_value = _finite_number("importance", importance, low=0.0, high=1.0)
    features = analyze_audio(source, ffmpeg=ffmpeg, window_ms=window_ms,
                             silence_dbfs=silence_dbfs, max_duration_s=max_duration_s,
                             timeout_s=timeout_s)
    features["event_profile"] = segment_audio_events(features, silence_dbfs=silence_dbfs)
    features["activity_period"] = infer_audio_activity_period(features, silence_dbfs=silence_dbfs)
    summary = format_audio_summary(features)
    node = engine.add_perception(summary, modality="audio", importance=importance_value,
                                  tags=["audio_summary"], source="body:audio-summary")
    return {"status": "ok", "node_id": getattr(node, "id", None),
            "summary": summary, "features": features, "raw_audio_stored": False}


def ffmpeg_available(ffmpeg: str = "ffmpeg") -> bool:
    """Whether the optional FFmpeg executable can be resolved on this host."""
    return isinstance(ffmpeg, str) and bool(ffmpeg.strip()) and shutil.which(ffmpeg) is not None


def segment_audio_events(features: Dict[str, Any], *, silence_dbfs: float = -45.0) -> Dict[str, Any]:
    """Convert the bounded RMS envelope into ordered acoustic activity segments.

    When analyze_wav has downsampled the envelope, boundaries describe the
    retained window grid rather than sample-exact transitions. The returned
    resolution and approximate flag make that limitation explicit. This is
    an amplitude-profile segmenter, not speech or semantic event recognition.
    Boundaries use the frame-aligned analysis window when available; older
    feature mappings fall back to window_ms. All endpoints are duration-bounded.
    """
    if not isinstance(features, dict):
        raise TypeError("features 必须是 analyze_wav 返回的映射")
    required = ("duration_s", "window_ms", "energy_envelope_stride_windows", "energy_envelope_dbfs")
    missing = [key for key in required if key not in features]
    if missing:
        raise ValueError("features 缺少字段：%s" % ", ".join(missing))
    duration = _finite_number("features.duration_s", features["duration_s"],
                              low=0.001, high=_MAX_DURATION_SECONDS)
    envelope_duration = duration
    frame_count = features.get("frames")
    sample_rate = features.get("sample_rate_hz")
    if frame_count is not None or sample_rate is not None:
        if (isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < 1
                or isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate < 1):
            raise ValueError("features.frames 与 features.sample_rate_hz 必须是正整数")
        measured_duration = frame_count / sample_rate
        if round(measured_duration, 4) != duration:
            raise ValueError("features.duration_s 与 frames/sample_rate_hz 不一致")
        envelope_duration = measured_duration
    window_ms = features["window_ms"]
    if isinstance(window_ms, bool) or not isinstance(window_ms, int) or not 10 <= window_ms <= 1000:
        raise ValueError("features.window_ms 超出有效范围")
    stride = features["energy_envelope_stride_windows"]
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("features.energy_envelope_stride_windows 必须是正整数")
    if "analysis_window_s" in features:
        window_s = _finite_number("features.analysis_window_s", features["analysis_window_s"],
                                  low=0.00001, high=1.0)
    else:
        window_s = window_ms / 1000.0
    threshold_dbfs = _finite_number("silence_dbfs", silence_dbfs, low=-120.0, high=0.0)
    envelope = features["energy_envelope_dbfs"]
    if not isinstance(envelope, (list, tuple)) or len(envelope) > _MAX_ENVELOPE_POINTS:
        raise ValueError("features.energy_envelope_dbfs 必须是有界数值序列")
    envelope = [_finite_number("energy_envelope_dbfs item", item, low=-120.0, high=0.0)
                for item in envelope]
    if not envelope:
        return {"segments": [], "resolution_s": 0.0, "approximate": False,
                "segment_count": 0, "segments_truncated": False}
    resolution_s = window_s * stride
    # Use precise frame metadata for the envelope count when available.
    envelope = envelope[:max(1, int(math.ceil(envelope_duration / resolution_s)))]

    def boundary(index: int) -> float:
        return min(duration, round(index * resolution_s, 4))

    segments = []
    start = 0
    total_segments = 0
    current_active = envelope[0] > threshold_dbfs
    for index, level in enumerate(envelope[1:], 1):
        active = level > threshold_dbfs
        if active != current_active:
            total_segments += 1
            if len(segments) < _MAX_SILENCE_INTERVALS * 2:
                segments.append({"kind": "activity" if current_active else "silence",
                                 "start_s": boundary(start),
                                 "end_s": boundary(index),
                                 "mean_dbfs": round(sum(envelope[start:index]) / (index - start), 2)})
            start = index
            current_active = active
    total_segments += 1
    if len(segments) < _MAX_SILENCE_INTERVALS * 2:
        segments.append({"kind": "activity" if current_active else "silence",
                         "start_s": boundary(start),
                         "end_s": boundary(len(envelope)),
                         "mean_dbfs": round(sum(envelope[start:]) / (len(envelope) - start), 2)})
    return {"segments": segments, "resolution_s": round(resolution_s, 9),
            "approximate": stride > 1, "segment_count": total_segments,
            "segments_truncated": total_segments > len(segments)}


def infer_audio_activity_period(
    features: Dict[str, Any],
    *,
    silence_dbfs: float = -45.0,
    tolerance: float = 0.2,
    min_fronts: int = 4,
) -> Dict[str, Any]:
    """Conservatively estimate a repeated period from RMS silence-to-activity fronts.

    This estimates only periodic acoustic activity, not a musical beat, speech cadence,
    source identity, or semantic/world event. Downsampled envelopes are rejected because
    sparse sampling can miss short bursts. A period is returned only with enough fronts,
    at least three supported intervals, and adequate time resolution.
    """
    if not isinstance(features, dict):
        raise TypeError("features 必须是 analyze_wav/analyze_audio 返回的映射")
    required = ("duration_s", "analysis_window_s", "energy_envelope_stride_windows",
                "energy_envelope_dbfs")
    missing = [key for key in required if key not in features]
    if missing:
        raise ValueError("features 缺少字段：%s" % ", ".join(missing))
    duration = _finite_number("features.duration_s", features["duration_s"],
                              low=0.001, high=_MAX_DURATION_SECONDS)
    window_s = _finite_number("features.analysis_window_s", features["analysis_window_s"],
                              low=0.00001, high=1.0)
    stride = features["energy_envelope_stride_windows"]
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("features.energy_envelope_stride_windows 必须是正整数")
    if isinstance(min_fronts, bool) or not isinstance(min_fronts, int) or not 4 <= min_fronts <= 64:
        raise ValueError("min_fronts 必须是 [4, 64] 内的整数")
    tolerance_value = _finite_number("tolerance", tolerance, low=0.01, high=1.0)
    threshold = _finite_number("silence_dbfs", silence_dbfs, low=-120.0, high=0.0)
    envelope = features["energy_envelope_dbfs"]
    if not isinstance(envelope, (list, tuple)) or len(envelope) > _MAX_ENVELOPE_POINTS:
        raise ValueError("features.energy_envelope_dbfs 必须是有界数值序列")
    envelope = [_finite_number("energy_envelope_dbfs item", v, low=-120.0, high=0.0)
                for v in envelope]
    resolution = window_s * stride

    def result(status: str, reason: str, **values: Any) -> Dict[str, Any]:
        return {"status": status, "reason": reason, "period_s": None,
                "candidate_period_s": None, "front_count": 0,
                "support_interval_count": 0, "support_ratio": 0.0,
                "jitter_ratio": None, "resolution_s": round(resolution, 9),
                "approximate": stride > 1, **values}

    if stride > 1:
        return result("insufficient_resolution", "downsampled_envelope")
    if not envelope:
        return result("insufficient_evidence", "empty_envelope")

    # A high first window is not a measured silence-to-activity transition; require
    # an observed low-to-high crossing rather than inventing an onset at t=0.
    fronts = []
    for i in range(1, len(envelope)):
        if envelope[i - 1] <= threshold and envelope[i] > threshold:
            fronts.append(i)
    if len(fronts) < min_fronts:
        has_activity = any(level > threshold for level in envelope)
        reason = ("no_activity" if not has_activity else
                  "no_observed_fronts" if not fronts else "too_few_activity_fronts")
        return result("insufficient_evidence", reason, front_count=len(fronts))

    intervals = [(fronts[i] - fronts[i - 1]) * resolution
                 for i in range(1, len(fronts))]
    ordered = sorted(intervals)
    mid = len(ordered) // 2
    candidate = (ordered[mid] if len(ordered) % 2
                 else (ordered[mid - 1] + ordered[mid]) / 2.0)
    deviations = [abs(interval - candidate) / candidate for interval in intervals]
    supported = [d <= tolerance_value for d in deviations]
    support_count = sum(supported)
    support_ratio = support_count / len(intervals)
    jitter = max(deviations)
    common = {"front_count": len(fronts),
              "front_times_s": [round(i * resolution, 6) for i in fronts],
              "intervals_s": [round(v, 6) for v in intervals],
              "candidate_period_s": round(candidate, 6),
              "support_interval_count": support_count,
              "support_ratio": round(support_ratio, 4),
              "jitter_ratio": round(jitter, 4)}
    if candidate < 4.0 * resolution:
        return result("insufficient_resolution", "period_below_four_windows", **common)
    if support_count < 3 or support_ratio < 0.75:
        return result("not_periodic", "intervals_not_stable", **common)
    return result("periodic", "stable_activity_front_intervals",
                  period_s=round(candidate, 6), **common)


def format_audio_summary(features: Dict[str, Any]) -> str:
    """把确定性特征与粗粒度声学片段转成短摘要，不含文件名或路径。"""
    segments = features.get("event_profile", {}).get("segments", [])
    activity_segments = sum(1 for item in segments if item["kind"] == "activity")
    silence_segments = sum(1 for item in segments if item["kind"] == "silence")
    profile = features.get("event_profile", {})
    detail = "活动片段 %d 段、静音片段 %d 段" % (activity_segments, silence_segments)
    shown = segments[:8]
    if shown:
        detail += "；片段序列：" + "；".join(
            "%s %.2f–%.2f秒（%s dBFS）" %
            ("活动" if item["kind"] == "activity" else "静音", item["start_s"],
             item["end_s"], item["mean_dbfs"])
            for item in shown)
    omitted = max(0, int(profile.get("segment_count", len(segments))) - len(shown))
    if omitted:
        detail += "；其余 %d 段省略" % omitted
    if profile.get("approximate"):
        detail += "（分段为降采样包络近似）"
    periodic = features.get("activity_period", {})
    if periodic.get("status") == "periodic":
        detail += "；声学活动重复间隔约 %.3f 秒（支持 %d 个间隔）" % (
            periodic["period_s"], periodic["support_interval_count"])
    source_format = features.get("source_audio_format", "wav")
    media_label = "WAV" if source_format == "wav" else "压缩音频(%s)" % source_format
    return (
        "%s 声学摘要：时长 %.2f 秒，采样率 %d Hz，%d 声道，%d-bit PCM；"
        "RMS %s dBFS，峰值 %s dBFS，活动比例 %.1f%%，静音区间 %d 段；%s。"
        % (media_label, features["duration_s"], features["sample_rate_hz"],
           features["channels"], features["sample_width_bits"], features["rms_dbfs"],
           features["peak_dbfs"], 100.0 * features["activity_ratio"],
           features["silence_interval_count"], detail)
    )


def perceive_wav(
    engine: Any,
    path: Union[str, os.PathLike, bytes, bytearray, memoryview],
    *,
    importance: float = 0.4,
    window_ms: int = 100,
    silence_dbfs: float = -45.0,
    max_duration_s: float = 300.0,
) -> Dict[str, Any]:
    """分析 WAV 文件路径或内存 bytes-like 数据，并写入 audio 模态摘要。

    内存数据无需临时文件；只存声学摘要，不存原音频、文件名或路径。
    engine 需实现 add_perception 接口。
    """
    if not callable(getattr(engine, "add_perception", None)):
        raise TypeError("engine 必须提供 add_perception(content, modality=...) 方法")
    importance_value = _finite_number("importance", importance, low=0.0, high=1.0)
    features = analyze_wav(path, window_ms=window_ms, silence_dbfs=silence_dbfs,
                           max_duration_s=max_duration_s)
    features["event_profile"] = segment_audio_events(features, silence_dbfs=silence_dbfs)
    features["activity_period"] = infer_audio_activity_period(features, silence_dbfs=silence_dbfs)
    summary = format_audio_summary(features)
    node = engine.add_perception(summary, modality="audio", importance=importance_value,
                                  tags=["audio_summary"], source="body:wav-acoustic-summary")
    return {"status": "ok", "node_id": getattr(node, "id", None),
            "summary": summary, "features": features, "raw_audio_stored": False}
