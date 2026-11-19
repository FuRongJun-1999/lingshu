# -*- coding: utf-8 -*-
"""身体 WAV 感知：PCM 解析、静音分段、资源界限与 audio 模态闭环。"""
from __future__ import annotations

import os
import subprocess
import sys
import wave

import pytest

from lingshu.core.core import SpacetimeMemoryEngine
from lingshu.vision import (
    AudioCodecUnavailable, AudioPerceptionError, analyze_audio, analyze_wav,
    ffmpeg_available, perceive_wav, segment_audio_events,
)
from lingshu.vision import audio as audio_module


def _encode_sample(value: int, width: int) -> bytes:
    if width == 1:
        return bytes((value + 128,))
    return int(value).to_bytes(width, "little", signed=True)


def _write_wav(path, values, *, width=2, channels=1, rate=8000):
    """values 是逐帧 signed integer；每帧恰含 channels 个样本。"""
    data = b"".join(b"".join(_encode_sample(x, width) for x in frame) for frame in values)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(width)
        out.setframerate(rate)
        out.writeframes(data)
    return path


def _tone_and_silence(path):
    rate = 8000
    silence = [(0,)] * (rate // 5)
    tone = [(8192,)] * (rate * 3 // 5)
    tail = [(0,)] * (rate // 5)
    return _write_wav(path, silence + tone + tail, rate=rate)


def test_wav_features_include_rms_peak_activity_and_two_silence_intervals(tmp_path):
    path = _tone_and_silence(tmp_path / "source.wav")
    got = analyze_wav(path, window_ms=100, silence_dbfs=-45)

    assert got["format"] == "wav-pcm"
    assert got["duration_s"] == pytest.approx(1.0)
    assert got["sample_rate_hz"] == 8000 and got["channels"] == 1
    assert got["sample_width_bits"] == 16 and got["frames"] == 8000
    assert got["peak_dbfs"] == pytest.approx(-12.04, abs=0.03)
    assert got["rms_dbfs"] == pytest.approx(-14.26, abs=0.08)
    assert got["silence_ratio"] == pytest.approx(0.4)
    assert got["activity_ratio"] == pytest.approx(0.6)
    assert got["silence_interval_count"] == 2
    assert got["silence_intervals"] == [
        {"start_s": 0.0, "end_s": 0.2},
        {"start_s": 0.8, "end_s": 1.0},
    ]
    assert len(got["energy_envelope_dbfs"]) == 10
    assert not got["silence_intervals_truncated"]


def test_event_profile_turns_envelope_into_explainable_segments(tmp_path):
    path = _tone_and_silence(tmp_path / "profile.wav")
    features = analyze_wav(path, window_ms=100, silence_dbfs=-45)
    profile = segment_audio_events(features, silence_dbfs=-45)

    assert profile["segment_count"] == 3
    assert profile["segments"] == [
        {"kind": "silence", "start_s": 0.0, "end_s": 0.2, "mean_dbfs": -120.0},
        {"kind": "activity", "start_s": 0.2, "end_s": 0.8, "mean_dbfs": -12.04},
        {"kind": "silence", "start_s": 0.8, "end_s": 1.0, "mean_dbfs": -120.0},
    ]
    assert profile["resolution_s"] == pytest.approx(0.1)
    assert profile["approximate"] is False
    assert profile["segments_truncated"] is False


def test_event_profile_marks_downsampled_resolution_as_approximate(tmp_path):
    features = analyze_wav(_tone_and_silence(tmp_path / "coarse.wav"), window_ms=100)
    features["energy_envelope_stride_windows"] = 4
    profile = segment_audio_events(features)
    assert profile["resolution_s"] == pytest.approx(0.4)
    assert profile["approximate"] is True


def test_event_profile_validates_input_and_bounds_segments():
    with pytest.raises(ValueError, match="缺少字段"):
        segment_audio_events({})

    malformed = {
        "duration_s": 3.0,
        "window_ms": 10,
        "energy_envelope_stride_windows": 1,
        "energy_envelope_dbfs": [-120.0, float("nan")],
    }
    with pytest.raises(ValueError, match="有限实数"):
        segment_audio_events(malformed)

    noisy = {
        "duration_s": 3.0,
        "window_ms": 10,
        "energy_envelope_stride_windows": 1,
        "energy_envelope_dbfs": [-120.0 if i % 2 else 0.0 for i in range(200)],
    }
    profile = segment_audio_events(noisy)
    assert profile["segment_count"] == 200
    assert len(profile["segments"]) == 128
    assert profile["segments_truncated"] is True
    features = {
        "duration_s": 3.0, "sample_rate_hz": 8000, "channels": 1,
        "sample_width_bits": 16, "rms_dbfs": -20.0, "peak_dbfs": -12.0,
        "activity_ratio": 0.5, "silence_interval_count": 100,
        "event_profile": profile,
    }
    from lingshu.vision import format_audio_summary
    summary = format_audio_summary(features)
    assert "片段序列：" in summary
    assert "其余 192 段省略" in summary


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_supported_pcm_widths_are_decoded(width, tmp_path):
    scale = 1 << (8 * width - 1)
    sample = max(1, scale // 4)
    path = _write_wav(tmp_path / ("pcm-%d.wav" % width), [(sample,)] * 800,
                      width=width, rate=8000)
    got = analyze_wav(path)
    assert got["sample_width_bits"] == width * 8
    assert got["peak_dbfs"] == pytest.approx(-12.04, abs=0.04)
    assert got["rms_dbfs"] == pytest.approx(got["peak_dbfs"], abs=0.02)


def test_stereo_clipping_is_reported_without_leaking_file_path(tmp_path):
    path = _write_wav(tmp_path / "private-name.wav", [(32767, -32768)] * 80,
                      width=2, channels=2, rate=8000)
    got = analyze_wav(path)
    assert got["channels"] == 2
    assert got["clipped_sample_ratio"] == 1.0
    assert "private-name.wav" not in repr(got)


def test_invalid_or_over_budget_wav_fails_before_memory_write(tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not a wave file")
    with pytest.raises(AudioPerceptionError, match="WAV"):
        analyze_wav(bad)

    truncated = _tone_and_silence(tmp_path / "truncated.wav")
    truncated.write_bytes(truncated.read_bytes()[:-4])
    with pytest.raises(AudioPerceptionError, match="截断"):
        analyze_wav(truncated)

    short = _tone_and_silence(tmp_path / "long-enough.wav")
    engine = SpacetimeMemoryEngine(":memory:")
    before = engine.store.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    with pytest.raises(AudioPerceptionError, match="时长"):
        perceive_wav(engine, short, max_duration_s=0.5)
    after = engine.store.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    assert after == before, "资源预算拒绝前不应写入摘要节点"
    with pytest.raises(ValueError, match="silence_dbfs"):
        analyze_wav(short, silence_dbfs=1.0)


def test_perceive_wav_adds_only_audio_summary_to_memory(tmp_path):
    path = _tone_and_silence(tmp_path / "do-not-store-this-name.wav")
    engine = SpacetimeMemoryEngine(":memory:")
    result = engine.perceive_audio(path, importance=0.4)

    caps = engine.get_body_capabilities()
    assert caps["modalities"]["audio"] is True
    assert caps["audio"]["available"] is True
    assert caps["audio"]["compressed_decode"] == "ffmpeg-optional"
    assert caps["audio"]["transcription"] is False
    assert caps["audio"]["raw_audio_persisted"] is False
    assert "perceive_audio" in caps["tools"]
    assert result["status"] == "ok"
    assert result["raw_audio_stored"] is False
    assert result["node_id"]
    assert "do-not-store-this-name.wav" not in repr(result)
    node = engine.store.get_node(result["node_id"])
    assert node is not None and node.modality == "audio"
    assert "WAV 声学摘要" in node.content
    assert node.content == result["summary"]
    assert "活动片段 1 段、静音片段 2 段" in node.content
    assert "片段序列：静音 0.00–0.20秒" in node.content
    profile = result["features"]["event_profile"]
    assert profile["segment_count"] == 3
    assert profile["resolution_s"] == pytest.approx(0.1)
    assert profile["approximate"] is False
    assert "audio_summary" in node.tags




def test_empty_or_oversized_memory_wav_bytes_fail_closed():
    from lingshu.vision import AudioPerceptionError

    with pytest.raises(ValueError, match="bytes 不能为空"):
        analyze_wav(b"")
    with pytest.raises(AudioPerceptionError, match="64 MiB"):
        analyze_wav(bytes(64 * 1024 * 1024 + 1))


def test_memory_wav_bytes_match_file_analysis_and_reach_memory(tmp_path):
    wav_path = _tone_and_silence(tmp_path / "buffer-source.wav")
    payload = wav_path.read_bytes()
    from_path = analyze_wav(wav_path)
    from_bytes = analyze_wav(payload)
    from_view = analyze_wav(memoryview(payload))
    assert from_bytes == from_path == from_view

    engine = SpacetimeMemoryEngine(":memory:")
    result = engine.perceive_audio(payload)
    assert result["status"] == "ok"
    assert result["features"]["duration_s"] == pytest.approx(1.0)
    assert "buffer-source.wav" not in result["summary"]
    assert engine.store.get_node(result["node_id"]).modality == "audio"


def test_wav_stays_on_standard_library_path_without_ffmpeg(tmp_path, monkeypatch):
    wav_path = _tone_and_silence(tmp_path / "standard.wav")

    def forbidden(*args, **kwargs):
        pytest.fail("WAV input must not start the optional FFmpeg process")

    monkeypatch.setattr(audio_module.shutil, "which", lambda _: None)
    monkeypatch.setattr(audio_module.subprocess, "run", forbidden)
    features = analyze_audio(wav_path.read_bytes())
    assert features["decoded_from_compressed"] is False
    assert features["source_audio_format"] == "wav"


def test_compressed_audio_input_budget_and_path_policy_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(audio_module, "_MAX_COMPRESSED_INPUT_BYTES", 8)
    with pytest.raises(AudioPerceptionError, match="64 MiB"):
        analyze_audio(b"ID3" + b"oversized")
    with pytest.raises(AudioPerceptionError, match="本地文件"):
        analyze_audio("https://example.invalid/audio.mp3")
    with pytest.raises(AudioPerceptionError, match="扩展名"):
        analyze_audio(tmp_path / "playlist.m3u")


def test_compressed_audio_uses_bounded_ffmpeg_pipe_and_reaches_memory(tmp_path, monkeypatch):
    wav_path = _tone_and_silence(tmp_path / "decoded.wav")
    with wave.open(str(wav_path), "rb") as reader:
        decoded_pcm = reader.readframes(reader.getnframes())
    commands = []

    def fake_run(command, **kwargs):
        commands.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout=decoded_pcm, stderr=b"")

    monkeypatch.setattr(audio_module.shutil, "which", lambda _: "ffmpeg-test")
    monkeypatch.setattr(audio_module.subprocess, "run", fake_run)
    compressed = b"ID3" + b"synthetic compressed payload"
    engine = SpacetimeMemoryEngine(":memory:")
    before = engine.store.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    result = engine.perceive_audio(compressed, max_duration_s=2.0, timeout_s=3.0)

    assert result["status"] == "ok"
    assert result["raw_audio_stored"] is False
    assert result["features"]["decoded_from_compressed"] is True
    assert result["features"]["decode_output_sample_rate_hz"] == 8000
    assert result["features"]["duration_s"] == pytest.approx(1.0)
    assert "压缩音频(mp3) 声学摘要" in result["summary"]
    assert engine.store.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] == before + 1
    command, kwargs = commands[0]
    assert command[command.index("-protocol_whitelist") + 1] == "pipe"
    assert command[command.index("-f") + 1] == "mp3"
    assert command[command.index("-t") + 1] == "2.000000"
    assert int(command[command.index("-fs") + 1]) <= 64 * 1024 * 1024
    assert kwargs["input"] == compressed and kwargs["timeout"] == 3.0
    assert kwargs["shell"] is False and kwargs["stderr"] == subprocess.DEVNULL
    assert "synthetic compressed payload" not in result["summary"]


def test_compressed_audio_missing_ffmpeg_fails_before_memory_write(monkeypatch):
    monkeypatch.setattr(audio_module.shutil, "which", lambda _: None)
    engine = SpacetimeMemoryEngine(":memory:")
    before = engine.store.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    with pytest.raises(AudioCodecUnavailable, match="可选 FFmpeg"):
        engine.perceive_audio(b"ID3synthetic")
    assert engine.store.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] == before
    with pytest.raises(AudioPerceptionError, match="识别"):
        analyze_audio(b"not a supported audio signature")


def test_compressed_audio_decoded_pcm_limit_fails_before_memory_write(monkeypatch):
    monkeypatch.setattr(audio_module.shutil, "which", lambda _: "ffmpeg-test")
    monkeypatch.setattr(audio_module, "_MAX_DECODED_PCM_BYTES", 2)
    monkeypatch.setattr(
        audio_module.subprocess, "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout=b"\x00\x00\x00\x00", stderr=b""),
    )
    engine = SpacetimeMemoryEngine(":memory:")
    before = engine.store.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    with pytest.raises(AudioPerceptionError, match="超过 64 MiB"):
        engine.perceive_audio(b"ID3synthetic")
    assert engine.store.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] == before


def test_compressed_audio_timeout_is_reported_without_memory_write(monkeypatch):
    monkeypatch.setattr(audio_module.shutil, "which", lambda _: "ffmpeg-test")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(audio_module.subprocess, "run", timeout)
    with pytest.raises(AudioPerceptionError, match="超过时间限制"):
        analyze_audio(b"ID3synthetic", timeout_s=0.2)


@pytest.mark.skipif(not ffmpeg_available(), reason="optional FFmpeg is not installed")
def test_real_ffmpeg_mp3_round_trip_to_audio_memory(tmp_path):
    wav_path = _tone_and_silence(tmp_path / "input.wav")
    mp3_path = tmp_path / "input.mp3"
    encoded = subprocess.run(
        [audio_module.shutil.which("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error",
         "-y", "-i", str(wav_path), "-t", "1", "-c:a", "libmp3lame", "-b:a", "64k", str(mp3_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, check=False, shell=False)
    if encoded.returncode != 0:
        pytest.skip("installed FFmpeg does not provide the test MP3 encoder")
    engine = SpacetimeMemoryEngine(":memory:")
    result = engine.perceive_audio(mp3_path, max_duration_s=2)
    assert result["status"] == "ok"
    assert result["features"]["decoded_from_compressed"] is True
    assert result["features"]["channels"] == 1
    assert result["features"]["sample_rate_hz"] == 8000
    assert mp3_path.name not in repr(result)
    assert engine.store.get_node(result["node_id"]).modality == "audio"


def test_perceive_wav_rejects_invalid_engine_before_reading(tmp_path):
    with pytest.raises(TypeError, match="add_perception"):
        perceive_wav(object(), tmp_path / "missing.wav")


def test_audio_public_import_does_not_load_numpy_or_pillow():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = r"""
import sys
sys.path.insert(0, sys.argv[1])
class DenyOptional:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "numpy" or fullname.startswith("numpy.") or fullname == "PIL" or fullname.startswith("PIL."):
            raise RuntimeError("optional image dependency was imported: " + fullname)
sys.meta_path.insert(0, DenyOptional())
from lingshu.vision import (analyze_wav, analyze_audio, decode_compressed_audio,
    ffmpeg_available, perceive_audio_file, perceive_wav, segment_audio_events)
assert all(callable(item) for item in (analyze_wav, analyze_audio, decode_compressed_audio,
    ffmpeg_available, perceive_audio_file, perceive_wav, segment_audio_events))
assert "numpy" not in sys.modules and "PIL" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", script, repo],
        cwd=repo, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
