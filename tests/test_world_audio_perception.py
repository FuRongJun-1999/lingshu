# -*- coding: utf-8 -*-
"""身体 WAV 感知：PCM 解析、静音分段、资源界限与 audio 模态闭环。"""
from __future__ import annotations

import os
import subprocess
import sys
import wave

import pytest

from lingshu.core.core import SpacetimeMemoryEngine
from lingshu.vision import AudioPerceptionError, analyze_wav, perceive_wav


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
    assert caps["audio"] == {
        "provider": "wav-pcm", "available": True,
        "transcription": False, "raw_audio_persisted": False,
    }
    assert "perceive_audio" in caps["tools"]
    assert result["status"] == "ok"
    assert result["raw_audio_stored"] is False
    assert result["node_id"]
    assert "do-not-store-this-name.wav" not in repr(result)
    node = engine.store.get_node(result["node_id"])
    assert node is not None and node.modality == "audio"
    assert "WAV 声学摘要" in node.content
    assert node.content == result["summary"]
    assert "audio_summary" in node.tags


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
from lingshu.vision import analyze_wav, perceive_wav
assert callable(analyze_wav) and callable(perceive_wav)
assert "numpy" not in sys.modules and "PIL" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", script, repo],
        cwd=repo, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
