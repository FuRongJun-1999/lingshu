# -*- coding: utf-8 -*-
"""Bounded video gray-frame sampling and adjacent pixel-difference tests."""
from __future__ import annotations

import io
import struct
from types import SimpleNamespace
import subprocess
import sys
import time

import pytest

from lingshu.vision import VideoAnalysisError, analyze_video_frames, ffmpeg_video_available
from lingshu.vision import video as video_module

_FRAME_BYTES = 64 * 64


def _mock_frames(*values):
    return b"".join(bytes([value]) * _FRAME_BYTES for value in values)


def _mock_mp4(payload=b"data"):
    # Well-formed top-level ftyp + moov + mdat boxes (faststart, pipe seekless).
    ftyp = struct.pack(">I4s8s", 16, b"ftyp", b"isom\x00\x00\x02\x00")
    moov = struct.pack(">I4s", 8, b"moov")
    mdat = struct.pack(">I4s", 8 + len(payload), b"mdat") + payload
    return ftyp + moov + mdat


def _mock_mp4_tail_moov(payload=b"data", free_size=1024 * 1024):
    ftyp = struct.pack(">I4s8s", 16, b"ftyp", b"isom\x00\x00\x02\x00")
    mdat = struct.pack(">I4s", 8 + len(payload), b"mdat") + payload
    free = struct.pack(">I4s", free_size, b"free") + bytes(free_size - 8)
    moov = struct.pack(">I4s", 8, b"moov")
    return ftyp + mdat + free + moov


class _InputPipe:
    def __init__(self):
        self.data = bytearray()
        self.closed = False

    def write(self, chunk):
        self.data.extend(chunk)
        return len(chunk)

    def close(self):
        self.closed = True


class _FakePopen:
    def __init__(self, command, output=b"", returncode=0, **kwargs):
        self.command = command
        self.stdin = _InputPipe()
        self.stdout = io.BytesIO(output)
        self.returncode = returncode
        self.kwargs = kwargs

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.returncode = -9


def _install_fake_popen(monkeypatch, output, returncode=0):
    calls = []
    def factory(command, **kwargs):
        process = _FakePopen(command, output=output, returncode=returncode, **kwargs)
        calls.append((command, kwargs, process))
        return process
    monkeypatch.setattr(video_module.subprocess, "Popen", factory)
    return calls


def test_gray_frames_report_adjacent_difference_without_returning_pixels(monkeypatch):
    payload = _mock_frames(0, 0, 255)
    calls = _install_fake_popen(monkeypatch, payload)

    monkeypatch.setattr(video_module.shutil, "which", lambda _: "ffmpeg-test")
    # MP4 file signature: ftyp box. Its synthetic payload is accepted by signature routing.
    result = analyze_video_frames(_mock_mp4(), sample_fps=2)

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
    command, process_kwargs, process = calls[0]
    assert command[command.index("-protocol_whitelist") + 1] == "pipe"
    assert command[command.index("-map") + 1] == "0:v:0"
    assert command[command.index("-t") + 1] == "60.000000"
    assert command[-1] == "pipe:1"
    assert bytes(process.stdin.data).startswith(_mock_mp4())
    assert process_kwargs["shell"] is False and process_kwargs["stderr"] is video_module.subprocess.DEVNULL


def test_single_static_frame_has_no_adjacent_difference(monkeypatch):
    monkeypatch.setattr(video_module.shutil, "which", lambda _: "ffmpeg-test")
    monkeypatch.setattr(video_module, "_run_ffmpeg_bounded", lambda command, data, timeout, limit:
                    subprocess.CompletedProcess(command, 0, stdout=_mock_frames(64), stderr=None))
    result = analyze_video_frames(_mock_mp4(), max_frames=1)
    assert result["frame_count"] == 1
    assert result["mean_adjacent_frame_difference"] is None
    assert result["adjacent_frame_differences"] == []
    assert result["truncated_by_frame_limit"] is False


def test_frame_sampling_budget_caps_returned_stats(monkeypatch):
    monkeypatch.setattr(video_module.shutil, "which", lambda _: "ffmpeg-test")
    payload = _mock_frames(0, 64, 128, 192)
    monkeypatch.setattr(video_module, "_run_ffmpeg_bounded", lambda command, data, timeout, limit:
                    subprocess.CompletedProcess(command, 0, stdout=payload, stderr=None))
    result = analyze_video_frames(_mock_mp4(), max_frames=3)
    assert result["frame_count"] == 3
    assert result["truncated_by_frame_limit"] is True
    assert len(result["adjacent_frame_differences"]) == 2


def test_core_video_utility_is_discoverable_but_not_video_memory(monkeypatch):
    from lingshu.core.core import SpacetimeMemoryEngine

    monkeypatch.setattr(video_module.shutil, "which", lambda _: "ffmpeg-test")
    monkeypatch.setattr(video_module, "_run_ffmpeg_bounded", lambda command, data, timeout, limit:
                    subprocess.CompletedProcess(command, 0, stdout=_mock_frames(0, 255), stderr=None))
    engine = SpacetimeMemoryEngine(":memory:")
    caps = engine.get_body_capabilities()
    assert caps["modalities"]["video"] is False
    assert caps["video_frame_analysis"]["available"] is True
    assert caps["video_frame_analysis"]["semantic_perception"] is False
    assert caps["video_frame_analysis"]["frames_persisted"] is False
    assert "analyze_video_frames" in caps["tools"]
    before = engine.store.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    result = engine.analyze_video_frames(_mock_mp4())
    after = engine.store.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    assert result["status"] == "ok"
    assert result["mean_adjacent_frame_difference"] == 1.0
    assert after == before, "frame statistics must not masquerade as persisted video memory"


def test_pipe_timeout_covers_blocked_stdin_write():
    command = [sys.executable, "-c", "import time; time.sleep(3)"]
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        video_module._run_ffmpeg_bounded(command, b"x" * (1024 * 1024), 0.2, 1024)
    assert time.monotonic() - started < 1.5
    assert caught.value.output is None, "timeout must never expose partial raw frame bytes"


def test_unc_video_path_is_rejected_before_isfile_or_open(monkeypatch):
    monkeypatch.setattr(video_module, "_is_windows", lambda: True)
    monkeypatch.setattr(video_module.os.path, "abspath", lambda path: path)
    monkeypatch.setattr(video_module, "_windows_drive_type", lambda drive: 4)

    def forbidden(*args, **kwargs):
        pytest.fail("network path must be rejected before file-system access")

    monkeypatch.setattr(video_module.os.path, "isfile", forbidden)
    with pytest.raises(VideoAnalysisError, match="UNC"):
        video_module._read_bounded_source(r"\\server\share\video.mp4")
    with pytest.raises(VideoAnalysisError, match="UNC"):
        video_module._read_bounded_source("Z:\\mapped\\video.mp4")


def test_mov_box_count_budget(monkeypatch):
    monkeypatch.setattr(video_module, "_MAX_MOV_TOP_LEVEL_BOXES", 8)
    free = struct.pack(">I4s", 8, b"free")
    video_module._require_seekless_mov(_mock_mp4() + free * 5)
    with pytest.raises(VideoAnalysisError, match="too many top-level boxes"):
        video_module._require_seekless_mov(_mock_mp4() + free * 6)


@pytest.mark.parametrize("source", ["video.mp4", r"\video.mp4", r"Z:video.mp4"])
def test_relative_mapped_drive_is_rejected_before_probe(monkeypatch, source):
    monkeypatch.setattr(video_module, "_is_windows", lambda: True)
    monkeypatch.setattr(video_module.os.path, "abspath", lambda path: r"Z:\cwd\video.mp4")
    monkeypatch.setattr(video_module, "_windows_drive_type", lambda drive: 4)
    monkeypatch.setattr(video_module.os.path, "isfile",
                        lambda path: pytest.fail("network path reached isfile"))
    with pytest.raises(VideoAnalysisError, match="UNC"):
        video_module._read_bounded_source(source)


def test_windows_drive_allowlist_is_fail_closed(monkeypatch):
    monkeypatch.setattr(video_module, "_is_windows", lambda: True)
    drive_types = {"fixed": 3, "removable": 2, "remote": 4, "unknown": 0, "cdrom": 5}
    for name, drive_type in drive_types.items():
        monkeypatch.setattr(video_module, "_windows_drive_type", lambda drive, kind=drive_type: kind)
        result = video_module._is_network_path("C:\\sample.mp4")
        assert result is (name not in ("fixed", "removable")), name


def test_relative_path_on_unc_cwd_is_resolved_then_rejected(monkeypatch):
    monkeypatch.setattr(video_module, "_is_windows", lambda: True)
    monkeypatch.setattr(video_module.os.path, "abspath", lambda path: r"\\server\share\cwd\video.mp4")
    monkeypatch.setattr(video_module.os.path, "isfile",
                        lambda *args, **kwargs: pytest.fail("must reject UNC cwd before isfile"))
    with pytest.raises(VideoAnalysisError, match="UNC"):
        video_module._read_bounded_source("video.mp4")


def test_reparse_component_is_rejected_before_isfile(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(video_module, "_is_windows", lambda: True)
    monkeypatch.setattr(video_module, "_windows_drive_type", lambda drive: 3)
    monkeypatch.setattr(video_module.os.path, "abspath", lambda path: r"C:\safe\junction\video.mp4")
    monkeypatch.setattr(video_module.os, "lstat",
                        lambda path: SimpleNamespace(st_mode=video_module.stat.S_IFDIR,
                                                     st_file_attributes=0x400))
    monkeypatch.setattr(video_module.os.path, "isfile",
                        lambda *args, **kwargs: pytest.fail("must reject reparse component before isfile"))
    with pytest.raises(VideoAnalysisError, match="重解析点"):
        video_module._read_bounded_source(r"C:\safe\junction\video.mp4")


def test_video_frame_analysis_rejects_invalid_sources_and_limits(tmp_path):
    # UNC shares are rejected before any filesystem call (also covered by a Windows-path stub below).
    with pytest.raises(VideoAnalysisError, match="本地文件"):
        analyze_video_frames("https://example.invalid/video.mp4")
    with pytest.raises(VideoAnalysisError, match="签名|识别"):
        analyze_video_frames(b"not a supported video container")
    with pytest.raises(VideoAnalysisError, match="64 MiB"):
        analyze_video_frames(memoryview(bytes(64 * 1024 * 1024 + 1)))
    with pytest.raises(ValueError, match="sample_fps"):
        analyze_video_frames(_mock_mp4(), sample_fps=99)
    with pytest.raises(ValueError, match="max_frames"):
        analyze_video_frames(_mock_mp4(), max_frames=121)


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
        analyze_video_frames(_mock_mp4())


def test_video_frame_analysis_ffmpeg_failure_and_timeout_are_distinct(monkeypatch):
    monkeypatch.setattr(video_module.shutil, "which", lambda _: "ffmpeg-test")
    monkeypatch.setattr(video_module, "_run_ffmpeg_bounded", lambda command, data, timeout, limit:
                    subprocess.CompletedProcess(command, 1, stdout=b"", stderr=None))
    with pytest.raises(VideoAnalysisError, match="无法从该视频"):
        analyze_video_frames(_mock_mp4())

    def worker_timeout(command, data, timeout, limit):
        error = TimeoutError("simulated worker deadline")
        error.sensitive_partial_bytes = b"worker output must not escape"
        raise error

    monkeypatch.setattr(video_module, "_run_ffmpeg_bounded", worker_timeout)
    with pytest.raises(VideoAnalysisError, match="超过时间限制") as worker_caught:
        analyze_video_frames(_mock_mp4(), timeout_s=0.2)
    assert isinstance(worker_caught.value.__cause__, subprocess.TimeoutExpired)
    assert worker_caught.value.__cause__.output is None
    assert worker_caught.value.__cause__.stderr is None
    assert worker_caught.value.__context__ is None

    def backend_timeout(command, data, timeout, limit):
        raise subprocess.TimeoutExpired(command, timeout, output=b"backend frame bytes")

    monkeypatch.setattr(video_module, "_run_ffmpeg_bounded", backend_timeout)
    with pytest.raises(VideoAnalysisError, match="超过时间限制") as backend_caught:
        analyze_video_frames(_mock_mp4(), timeout_s=0.2)
    assert isinstance(backend_caught.value.__cause__, subprocess.TimeoutExpired)
    assert backend_caught.value.__cause__.output is None
    assert backend_caught.value.__cause__.stderr is None
    assert backend_caught.value.__context__ is None
    assert b"backend frame bytes" not in repr(backend_caught.value).encode()


def _encode_mp4(path, duration, *, faststart):
    ffmpeg = video_module.shutil.which("ffmpeg")
    command = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
               "-f", "lavfi", "-i", "testsrc2=size=96x64:rate=4:duration=%s" % duration,
               "-c:v", "mpeg4", "-pix_fmt", "yuv420p"]
    if faststart:
        command.extend(["-movflags", "+faststart"])
    command.append(str(path))
    encoded = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             timeout=30, check=False, shell=False)
    if encoded.returncode != 0:
        pytest.skip("installed FFmpeg does not provide mpeg4 test encoding")
    return path.read_bytes()


def _box_positions(data):
    offset = 0
    boxes = []
    while offset + 8 <= len(data):
        size = int.from_bytes(data[offset:offset + 4], "big")
        kind = data[offset + 4:offset + 8]
        header = 8
        if size == 1:
            size = int.from_bytes(data[offset + 8:offset + 16], "big")
            header = 16
        elif size == 0:
            size = len(data) - offset
        if size < header or offset + size > len(data):
            break
        boxes.append((kind, offset))
        offset += size
    return boxes


@pytest.mark.skipif(not ffmpeg_video_available(), reason="optional FFmpeg is not installed")
def test_real_ffmpeg_video_frame_sampling(tmp_path):
    video_path = tmp_path / "synthetic.mp4"
    _encode_mp4(video_path, 2, faststart=True)
    result = analyze_video_frames(video_path, sample_fps=2, max_frames=8, max_duration_s=3)
    assert result["status"] == "ok"
    assert result["frame_count"] >= 3
    assert result["container_demuxer"] == "mov"
    assert result["frame_width"] == result["frame_height"] == 64
    assert result["mean_adjacent_frame_difference"] is not None
    assert result["decoded_frame_byte_count"] <= 8 * _FRAME_BYTES
    assert "synthetic.mp4" not in repr(result)


@pytest.mark.skipif(not ffmpeg_video_available(), reason="optional FFmpeg is not installed")
def test_real_tail_moov_with_megabyte_padding_is_rejected_predecode(tmp_path, monkeypatch):
    video_path = tmp_path / "tail-moov-padded.mp4"
    data = _encode_mp4(video_path, 1, faststart=False)
    boxes = _box_positions(data)
    mdat = next(position for kind, position in boxes if kind == b"mdat")
    moov = next(position for kind, position in boxes if kind == b"moov")
    assert mdat < moov
    free_payload = bytes(1024 * 1024)
    free_box = struct.pack(">I4s", len(free_payload) + 8, b"free") + free_payload
    tail_moov = data[:moov] + free_box + data[moov:]
    monkeypatch.setattr(video_module.subprocess, "Popen",
                        lambda *args, **kwargs: pytest.fail("tail moov must be rejected before FFmpeg"))
    with pytest.raises(VideoAnalysisError, match="trailing moov|faststart"):
        analyze_video_frames(tail_moov)


@pytest.mark.skipif(not ffmpeg_video_available(), reason="optional FFmpeg is not installed")
def test_real_ffmpeg_subsecond_video_keeps_zero_time_frame(tmp_path):
    video_path = tmp_path / "short.mp4"
    _encode_mp4(video_path, 0.25, faststart=True)
    result = analyze_video_frames(video_path, sample_fps=1.0, max_frames=8, max_duration_s=1.0)
    assert result["status"] == "ok"
    assert result["frame_count"] == 1
    assert result["sample_times_s"] == [0.0]
    assert result["adjacent_frame_differences"] == []
