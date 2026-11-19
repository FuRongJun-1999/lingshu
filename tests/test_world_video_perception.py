# -*- coding: utf-8 -*-
"""Bounded video gray-frame sampling and adjacent pixel-difference tests."""
from __future__ import annotations

import subprocess

import pytest

from lingshu.vision import VideoAnalysisError, analyze_video_frames, ffmpeg_video_available
from lingshu.vision import video as video_module

_FRAME_BYTES = 64 * 64


def _mock_frames(*values):
    return b"".join(bytes([value]) * _FRAME_BYTES for value in values)


def test_gray_frames_report_adjacent_difference_without_returning_pixels(monkeypatch):
    payload = _mock_frames(0, 0, 255)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout=payload, stderr=b"")

    monkeypatch.setattr(video_module.shutil, "which", lambda _: "ffmpeg-test")
    monkeypatch.setattr(video_module.subprocess, "run", fake_run)
    # MP4 file signature: ftyp box. Its synthetic payload is accepted by signature routing.
    result = analyze_video_frames(b"\x00\x00\x00\x18ftypisom" + bytes(24), sample_fps=2)

    assert result["status"] == "ok"
    assert result["container_demuxer"] == "mov"
    assert result["frame_count"] == 3
    assert result["sample_interval_s"] == pytest.approx(0.5)
    assert result["sample_times_s"] == [0.0, 0.5, 1.0]
    assert result["frame_width"] == 64 and result["frame_height"] == 64
    assert result["adjacent_frame_differences"] == [
        {"from_s": 0.0, "to_s": 0.5, "mean_abs_diff_0_1": 0.0},
        {"from_s": 0.5, "to_s": 1.0, "mean_abs_diff_0_1": 1.0},
    ]
    assert result["mean_adjacent_frame_difference"] == 0.5
    assert result["decoded_frame_byte_count"] == 3 * _FRAME_BYTES
    assert b"\x00" * _FRAME_BYTES not in repr(result).encode()
    command, kwargs = calls[0]
    assert command[command.index("-protocol_whitelist") + 1] == "pipe"
    assert command[command.index("-map") + 1] == "0:v:0"
    assert command[command.index("-t") + 1] == "60.000000"
    assert command[-1] == "pipe:1"
    assert kwargs["shell"] is False and kwargs["stderr"] == subprocess.DEVNULL
    assert kwargs["input"].startswith(b"\x00\x00\x00\x18ftyp")


def test_single_static_frame_has_no_adjacent_difference(monkeypatch):
    monkeypatch.setattr(video_module.shutil, "which", lambda _: "ffmpeg-test")
    monkeypatch.setattr(video_module.subprocess, "run", lambda command, **kwargs:
                    subprocess.CompletedProcess(command, 0, stdout=_mock_frames(64), stderr=b""))
    result = analyze_video_frames(b"\x00\x00\x00\x18ftypisom" + bytes(24), max_frames=1)
    assert result["frame_count"] == 1
    assert result["mean_adjacent_frame_difference"] is None
    assert result["adjacent_frame_differences"] == []
    assert result["truncated_by_frame_limit"] is False


def test_frame_sampling_budget_caps_returned_stats(monkeypatch):
    monkeypatch.setattr(video_module.shutil, "which", lambda _: "ffmpeg-test")
    payload = _mock_frames(0, 64, 128, 192)
    monkeypatch.setattr(video_module.subprocess, "run", lambda command, **kwargs:
                    subprocess.CompletedProcess(command, 0, stdout=payload, stderr=b""))
    result = analyze_video_frames(b"\x00\x00\x00\x18ftypisom" + bytes(24), max_frames=3)
    assert result["frame_count"] == 3
    assert result["truncated_by_frame_limit"] is True
    assert len(result["adjacent_frame_differences"]) == 2


def test_core_video_utility_is_discoverable_but_not_video_memory(monkeypatch):
    from lingshu.core.core import SpacetimeMemoryEngine

    monkeypatch.setattr(video_module.shutil, "which", lambda _: "ffmpeg-test")
    monkeypatch.setattr(video_module.subprocess, "run", lambda command, **kwargs:
                    subprocess.CompletedProcess(command, 0, stdout=_mock_frames(0, 255), stderr=b""))
    engine = SpacetimeMemoryEngine(":memory:")
    caps = engine.get_body_capabilities()
    assert caps["modalities"]["video"] is False
    assert caps["video_frame_analysis"]["available"] is True
    assert caps["video_frame_analysis"]["semantic_perception"] is False
    assert caps["video_frame_analysis"]["frames_persisted"] is False
    assert "analyze_video_frames" in caps["tools"]
    before = engine.store.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    result = engine.analyze_video_frames(b"\x00\x00\x00\x18ftypisom" + bytes(24))
    after = engine.store.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    assert result["status"] == "ok"
    assert result["mean_adjacent_frame_difference"] == 1.0
    assert after == before, "frame statistics must not masquerade as persisted video memory"


def test_video_frame_analysis_rejects_invalid_sources_and_limits(tmp_path):
    with pytest.raises(VideoAnalysisError, match="本地文件"):
        analyze_video_frames("https://example.invalid/video.mp4")
    with pytest.raises(VideoAnalysisError, match="签名|识别"):
        analyze_video_frames(b"not a supported video container")
    with pytest.raises(VideoAnalysisError, match="64 MiB"):
        analyze_video_frames(memoryview(bytes(64 * 1024 * 1024 + 1)))
    with pytest.raises(ValueError, match="sample_fps"):
        analyze_video_frames(b"\x00\x00\x00\x18ftypisom" + bytes(24), sample_fps=99)
    with pytest.raises(ValueError, match="max_frames"):
        analyze_video_frames(b"\x00\x00\x00\x18ftypisom" + bytes(24), max_frames=121)


def test_video_public_import_does_not_load_numpy_or_pillow():
    import os
    import sys

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = r"""
import sys
sys.path.insert(0, sys.argv[1])
class DenyOptional:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "numpy" or fullname.startswith("numpy.") or fullname == "PIL" or fullname.startswith("PIL."):
            raise RuntimeError("optional image dependency was imported: " + fullname)
sys.meta_path.insert(0, DenyOptional())
from lingshu.vision import analyze_video_frames, ffmpeg_video_available
assert callable(analyze_video_frames) and callable(ffmpeg_video_available)
assert "numpy" not in sys.modules and "PIL" not in sys.modules
"""
    result = subprocess.run([sys.executable, "-S", "-c", script, repo], cwd=repo,
                            capture_output=True, text=True, encoding="utf-8", errors="replace",
                            timeout=30)
    assert result.returncode == 0, result.stderr


def test_video_frame_analysis_missing_ffmpeg_fails_explicitly(monkeypatch):
    monkeypatch.setattr(video_module.shutil, "which", lambda _: None)
    with pytest.raises(VideoAnalysisError, match="FFmpeg"):
        analyze_video_frames(b"\x00\x00\x00\x18ftypisom" + bytes(24))


def test_video_frame_analysis_ffmpeg_failure_and_timeout_are_distinct(monkeypatch):
    monkeypatch.setattr(video_module.shutil, "which", lambda _: "ffmpeg-test")
    monkeypatch.setattr(video_module.subprocess, "run", lambda command, **kwargs:
                    subprocess.CompletedProcess(command, 1, stdout=b"", stderr=b""))
    with pytest.raises(VideoAnalysisError, match="无法从该视频"):
        analyze_video_frames(b"\x00\x00\x00\x18ftypisom" + bytes(24))

    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])
    monkeypatch.setattr(video_module.subprocess, "run", timeout)
    with pytest.raises(VideoAnalysisError, match="超过时间限制"):
        analyze_video_frames(b"\x00\x00\x00\x18ftypisom" + bytes(24), timeout_s=0.2)


@pytest.mark.skipif(not ffmpeg_video_available(), reason="optional FFmpeg is not installed")
def test_real_ffmpeg_video_frame_sampling(tmp_path):
    ffmpeg = video_module.shutil.which("ffmpeg")
    video_path = tmp_path / "synthetic.mp4"
    encoded = subprocess.run(
        [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc2=size=96x64:rate=4:duration=2",
         "-c:v", "mpeg4", "-pix_fmt", "yuv420p", str(video_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, check=False, shell=False)
    if encoded.returncode != 0:
        pytest.skip("installed FFmpeg does not provide mpeg4 test encoding")
    result = analyze_video_frames(video_path, sample_fps=2, max_frames=8, max_duration_s=3)
    assert result["status"] == "ok"
    assert result["frame_count"] >= 3
    assert result["container_demuxer"] == "mov"
    assert result["frame_width"] == result["frame_height"] == 64
    assert result["mean_adjacent_frame_difference"] is not None
    assert result["decoded_frame_byte_count"] <= 8 * _FRAME_BYTES
    assert "synthetic.mp4" not in repr(result)
