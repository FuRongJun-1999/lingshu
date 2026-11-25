# -*- coding: utf-8 -*-
"""守卫测试：ADTS AAC 帧同步头必须在 MP3 同步掩码之前识别，且 MP3 须校验合法 MPEG layer/version。

根因背景
--------
``lingshu/vision/audio.py`` 的 ``_compressed_demuxer_from_bytes()`` 原先将
``data[0] == 0xFF and data[1] & 0xE0 == 0xE0``（MP3 裸帧同步）置于 ADTS AAC 判据
``data[0] == 0xFF and data[1] & 0xF6 == 0xF0`` 之前。因 ADTS 次字节的 4 个合法取值
（``0xF0 / 0xF1 / 0xF8 / 0xF9``）全部满足 ``& 0xE0 == 0xE0``，原 ADTS 分支为死代码，
导致所有 ADTS AAC 字节流及本地 ``.aac`` 文件（``decode_compressed_audio`` 按字节探测）
均被误判为 ``"mp3"`` 并以 ``-f mp3`` 传给 FFmpeg，100% 解码失败。

守卫断言
--------
  A1  ADTS AAC 全部 4 种次字节（0xF0/0xF1/0xF8/0xF9）及 ADIF 均识别为 ``"aac"``
  A2  Mock FFmpeg 下 ``decode_compressed_audio / analyze_audio / perceive_audio_file``
      对 ADTS bytes/bytearray/memoryview 与本地 ``.aac`` 文件均传 ``-f aac``
  A3  ID3v2、合法 MP3/MPEG 裸帧同步头（含 0xFB/0xFA/0xF3/0xF2/0xE3/0xE2）及其他压缩/WAV
      格式识别零回归
  A4  非法 MPEG reserved layer (``layer == 00``: 0xFF 0xE0 / 0xFF 0xE1) 及未知字节被拒绝
  A5  非法 MPEG reserved version (``version == 01``: 0xFF 0xE8..0xEF) 被拒绝
  A6  AST 源码序守卫：``_compressed_demuxer_from_bytes`` 中 ``"aac"`` 分支严格先于 ``"mp3"``
  A7  若宿主机安装了 FFmpeg 且支持 ``aac`` 编码器，端到端编码 0.3s ``tone.aac``（及无 ID3 裸 MP3）
      并验证 ``analyze_audio`` 路径与字节流双路真实解码通过（无 FFmpeg 时自动静默跳过）

运行方式
--------
  pytest:   python -X utf8 -m pytest tests/test_audio_adts_aac_demux.py -v
  直跑:     python -X utf8 tests/test_audio_adts_aac_demux.py
  变异自证: python -X utf8 tests/test_audio_adts_aac_demux.py --mutate
"""
from __future__ import annotations

import ast
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from unittest import mock

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from lingshu.vision import audio as audio_module  # noqa: E402
from lingshu.vision.audio import (  # noqa: E402
    AudioPerceptionError,
    analyze_audio,
    decode_compressed_audio,
    ffmpeg_available,
    perceive_audio_file,
)

_AUDIO_FILE = _REPO / "lingshu" / "vision" / "audio.py"


class _FakeNode:
    def __init__(self, node_id: str) -> None:
        self.id = node_id


class _RecordingEngine:
    def __init__(self) -> None:
        self.records = []

    def add_perception(self, content: str, **kwargs):
        self.records.append((content, kwargs))
        return _FakeNode("audio-node-%d" % len(self.records))


def _make_pcm_s16le(duration_s: float = 0.3, rate: int = 8000, amplitude: int = 8192) -> bytes:
    frames = int(round(duration_s * rate))
    return struct.pack("<h", amplitude) * frames


def _write_tone_wav(path: Path, *, duration_s: float = 0.3, rate: int = 8000) -> Path:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(_make_pcm_s16le(duration_s=duration_s, rate=rate))
    return path


def _real_ffmpeg_aac_capable() -> bool:
    """探测当前宿主机 FFmpeg 是否可用且自带 aac 编码器（无则不阻塞 CI）。"""
    if not ffmpeg_available():
        return False
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        return False
    with tempfile.TemporaryDirectory() as td:
        wav_path = _write_tone_wav(Path(td) / "probe.wav", duration_s=0.1)
        aac_path = Path(td) / "probe.aac"
        proc = subprocess.run(
            [
                ffmpeg_bin, "-nostdin", "-hide_banner", "-loglevel", "error",
                "-y", "-i", str(wav_path), "-c:a", "aac", "-f", "adts", str(aac_path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
            shell=False,
        )
        return proc.returncode == 0 and aac_path.is_file() and aac_path.stat().st_size > 0


# ----------------------------------------------------------------- A1..A7
def test_adts_aac_sync_headers_and_adif_detect_aac():
    """A1: 全部 4 种 ADTS 次字节（MPEG-4/2 × CRC 开/关）及 ADIF 必须识别为 aac。"""
    for second_byte in (0xF0, 0xF1, 0xF8, 0xF9):
        minimal = bytes([0xFF, second_byte])
        frame = bytes([0xFF, second_byte, 0x50, 0x80, 0x00, 0x1F, 0xFC])
        got_min = audio_module._compressed_demuxer_from_bytes(minimal)
        got_frame = audio_module._compressed_demuxer_from_bytes(frame)
        assert got_min == "aac", "0xFF 0x%02X (2B) 误判为 %r" % (second_byte, got_min)
        assert got_frame == "aac", "0xFF 0x%02X (7B) 误判为 %r" % (second_byte, got_frame)
    assert audio_module._compressed_demuxer_from_bytes(b"ADIF\x00\x00\x00\x00") == "aac"


def test_adts_aac_decode_and_analyze_pass_aac_demuxer_to_ffmpeg():
    """A2: Mock FFmpeg 下 bytes/bytearray/memoryview 与 .aac 文件均传 -f aac。"""
    pcm = _make_pcm_s16le(duration_s=0.3, rate=8000, amplitude=8192)
    seen_commands = []

    def fake_run(command, **kwargs):
        seen_commands.append((list(command), dict(kwargs)))
        return subprocess.CompletedProcess(command, 0, stdout=pcm, stderr=b"")

    with mock.patch.object(audio_module.shutil, "which", return_value="ffmpeg-mock"), \
            mock.patch.object(audio_module.subprocess, "run", side_effect=fake_run):
        for second_byte in (0xF0, 0xF1, 0xF8, 0xF9):
            payload = bytes([0xFF, second_byte, 0x6C, 0x40, 0x01, 0x1F, 0xFC, 0x00])
            for variant in (payload, bytearray(payload), memoryview(payload)):
                seen_commands.clear()
                wav_out = decode_compressed_audio(variant)
                assert wav_out.startswith(b"RIFF") and wav_out[8:12] == b"WAVE"
                cmd, _ = seen_commands[-1]
                assert cmd[cmd.index("-f") + 1] == "aac", (
                    "ADTS 0xFF 0x%02X 传给 FFmpeg 的 demuxer 为 %r 而非 'aac'"
                    % (second_byte, cmd[cmd.index("-f") + 1])
                )

            seen_commands.clear()
            features = analyze_audio(payload)
            assert features["decoded_from_compressed"] is True
            assert features["source_audio_format"] == "aac"
            assert features["sample_rate_hz"] == 8000
            assert features["channels"] == 1
            assert abs(features["duration_s"] - 0.3) < 1e-4
            cmd, _ = seen_commands[-1]
            assert cmd[cmd.index("-f") + 1] == "aac"

        with tempfile.TemporaryDirectory() as td:
            aac_path = Path(td) / "tone.aac"
            aac_path.write_bytes(b"\xFF\xF1\x6C\x40\x01\x1F\xFC\x00")
            seen_commands.clear()
            engine = _RecordingEngine()
            perceived = perceive_audio_file(engine, aac_path)
            assert perceived["status"] == "ok"
            assert perceived["raw_audio_stored"] is False
            assert perceived["features"]["source_audio_format"] == "aac"
            assert "压缩音频(aac) 声学摘要" in perceived["summary"]
            cmd, _ = seen_commands[-1]
            assert cmd[cmd.index("-f") + 1] == "aac", (
                ".aac 文件经 decode_compressed_audio 传给 FFmpeg 的 demuxer 为 %r"
                % (cmd[cmd.index("-f") + 1],)
            )


def test_valid_mp3_sync_headers_and_other_formats_remain_intact():
    """A3: ID3v2、全部合法 MPEG Audio 同步头以及其他受支持容器/WAV 零回归。"""
    assert audio_module._compressed_demuxer_from_bytes(b"ID3\x03\x00\x00\x00\x00\x00\x00") == "mp3"

    # 覆盖 MPEG-1 (0xFA..0xFF)、MPEG-2 (0xF2..0xF7)、MPEG-2.5 (0xE2..0xE7) 全部合法层
    valid_mp3_second_bytes = [
        b1 for b1 in range(0xE0, 0x100)
        if (b1 & 0x18) != 0x08 and (b1 & 0x06) != 0
    ]
    for explicit in (0xFB, 0xFA, 0xF3, 0xF2, 0xE3, 0xE2):
        assert explicit in valid_mp3_second_bytes
    for b1 in valid_mp3_second_bytes:
        got = audio_module._compressed_demuxer_from_bytes(bytes([0xFF, b1, 0x90, 0x00]))
        assert got == "mp3", "合法 MPEG 头 0xFF 0x%02X 误判为 %r" % (b1, got)

    other_signatures = [
        (b"fLaC\x00\x00\x00\x22", "flac"),
        (b"OggS\x00\x02\x00\x00", "ogg"),
        (b"\x00\x00\x00\x1cftypM4A \x00\x00\x00\x00", "ipod"),
        (b"FORM\x00\x00\x01\x00AIFF", "aiff"),
        (b"FORM\x00\x00\x01\x00AIFC", "aiff"),
        (b"\x1a\x45\xdf\xa3\x9f\x42\x86\x81", "matroska"),
        (b"\x30\x26\xb2\x75\x8e\x66\xcf\x11\xa6\xd9", "asf"),
    ]
    for sample, expected in other_signatures:
        got = audio_module._compressed_demuxer_from_bytes(sample)
        assert got == expected, "签名 %r 期望 %r，实得 %r" % (sample[:8], expected, got)

    # WAV 路径不经 FFmpeg
    with tempfile.TemporaryDirectory() as td:
        wav_path = _write_tone_wav(Path(td) / "pure.wav", duration_s=0.2)
        with mock.patch.object(audio_module.shutil, "which", return_value=None):
            got = analyze_audio(wav_path)
        assert got["decoded_from_compressed"] is False
        assert got["source_audio_format"] == "wav"


def test_reserved_mpeg_layer_zero_not_classified_as_mp3():
    """A4: 非 ADTS 的 MPEG 2.5 保留层 (layer == 00: 0xFF 0xE0 / 0xFF 0xE1) 与短/未知头须拒绝。"""
    for bad in (
        b"\xFF\xE0",
        b"\xFF\xE1",
        b"\xFF\xE0\x90\x00\x00\x00",
        b"\xFF\xE1\x90\x00\x00\x00",
        b"",
        b"\xFF",
        b"\xFF\x00",
        b"\xFF\xDF",
        b"not-an-audio-stream",
    ):
        try:
            got = audio_module._compressed_demuxer_from_bytes(bad)
        except AudioPerceptionError:
            continue
        raise AssertionError("非法/保留层字节 %r 不应识别为 %r" % (bad, got))


def test_reserved_mpeg_version_not_classified_as_mp3():
    """A5: MPEG 保留版本位 (version == 01: 0xFF 0xE8..0xEF) 须拒绝，不得判为 mp3。"""
    for b1 in range(0xE8, 0xF0):
        sample = bytes([0xFF, b1, 0x90, 0x00])
        try:
            got = audio_module._compressed_demuxer_from_bytes(sample)
        except AudioPerceptionError:
            continue
        raise AssertionError("MPEG 保留版本 0xFF 0x%02X 不应识别为 %r" % (b1, got))


def test_adts_branch_precedes_mp3_branch_in_ast():
    """A6: AST 源码序守卫——_compressed_demuxer_from_bytes 中 'aac' 分支必须严格先于 'mp3'。"""
    tree = ast.parse(_AUDIO_FILE.read_text(encoding="utf-8"))
    func = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_compressed_demuxer_from_bytes":
            func = node
            break
    assert func is not None, "未找到 _compressed_demuxer_from_bytes 定义"

    branch_order = []
    for stmt in func.body:
        if isinstance(stmt, ast.If) and len(stmt.body) == 1:
            ret = stmt.body[0]
            if isinstance(ret, ast.Return) and isinstance(ret.value, ast.Constant):
                if isinstance(ret.value.value, str):
                    branch_order.append(ret.value.value)
    assert "aac" in branch_order and "mp3" in branch_order, "分支缺失：%r" % (branch_order,)
    assert branch_order.index("aac") < branch_order.index("mp3"), (
        "ADTS AAC 分支未置于 MP3 之前：%r" % (branch_order,)
    )


def test_real_ffmpeg_adts_aac_and_raw_mp3_roundtrip_if_available():
    """A7: 有 FFmpeg 时实测 0.3s tone.aac（路径+bytes）及无 ID3 裸 MP3 端到端解码；无则跳过。"""
    if not ffmpeg_available():
        return
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        return

    with tempfile.TemporaryDirectory() as td:
        wav_path = _write_tone_wav(Path(td) / "tone.wav", duration_s=0.3, rate=8000)
        aac_path = Path(td) / "tone.aac"
        encoded_aac = subprocess.run(
            [
                ffmpeg_bin, "-nostdin", "-hide_banner", "-loglevel", "error",
                "-y", "-i", str(wav_path), "-c:a", "aac", "-f", "adts", str(aac_path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
            shell=False,
        )
        if encoded_aac.returncode == 0 and aac_path.is_file() and aac_path.stat().st_size > 0:
            raw_aac = aac_path.read_bytes()
            assert raw_aac[0] == 0xFF and (raw_aac[1] & 0xF6) == 0xF0, (
                "FFmpeg 生成的 ADTS AAC 头部异常: %s" % raw_aac[:4].hex()
            )
            assert audio_module._compressed_demuxer_from_bytes(raw_aac) == "aac"
            for res in (analyze_audio(aac_path), analyze_audio(raw_aac)):
                assert res["decoded_from_compressed"] is True
                assert res["source_audio_format"] == "aac"
                assert res["sample_rate_hz"] == 8000
                assert res["channels"] == 1
                assert res["duration_s"] > 0.2
                assert res["peak_dbfs"] > -30.0

        # 顺验无 ID3 裸帧 MP3（8 kHz 下 libmp3lame 生成 MPEG-2.5 0xFF 0xE3）
        mp3_path = Path(td) / "tone_raw.mp3"
        encoded_mp3 = subprocess.run(
            [
                ffmpeg_bin, "-nostdin", "-hide_banner", "-loglevel", "error",
                "-y", "-i", str(wav_path), "-c:a", "libmp3lame",
                "-write_xing", "0", "-id3v2_version", "0", str(mp3_path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
            shell=False,
        )
        if encoded_mp3.returncode == 0 and mp3_path.is_file() and mp3_path.stat().st_size > 0:
            raw_mp3 = mp3_path.read_bytes()
            assert raw_mp3[0] == 0xFF and not raw_mp3.startswith(b"ID3")
            assert audio_module._compressed_demuxer_from_bytes(raw_mp3) == "mp3"
            res_mp3 = analyze_audio(raw_mp3)
            assert res_mp3["decoded_from_compressed"] is True
            assert res_mp3["source_audio_format"] == "mp3"
            assert res_mp3["duration_s"] > 0.2


# ----------------------------------------------------------------- 直跑与 --mutate 变异自证
_ASSERTION_LIST = [
    ("A1", test_adts_aac_sync_headers_and_adif_detect_aac),
    ("A2", test_adts_aac_decode_and_analyze_pass_aac_demuxer_to_ffmpeg),
    ("A3", test_valid_mp3_sync_headers_and_other_formats_remain_intact),
    ("A4", test_reserved_mpeg_layer_zero_not_classified_as_mp3),
    ("A5", test_reserved_mpeg_version_not_classified_as_mp3),
    ("A6", test_adts_branch_precedes_mp3_branch_in_ast),
    ("A7", test_real_ffmpeg_adts_aac_and_raw_mp3_roundtrip_if_available),
]

_FIXED_HEAD = (
    '    if data.startswith(b"ADIF") or (\n'
    '        len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xF6) == 0xF0\n'
    '    ):\n'
    '        return "aac"\n'
    '    if data.startswith(b"ID3") or (\n'
    '        len(data) >= 2\n'
    '        and data[0] == 0xFF\n'
    '        and (data[1] & 0xE0) == 0xE0\n'
    '        and (data[1] & 0x18) != 0x08\n'
    '        and (data[1] & 0x06) != 0\n'
    '    ):\n'
    '        return "mp3"\n'
)

_REVERTED_HEAD = (
    '    if data.startswith(b"ID3") or (len(data) >= 2 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0):\n'
    '        return "mp3"\n'
    '    if data.startswith(b"ADIF") or (\n'
    '        len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xF6) == 0xF0\n'
    '    ):\n'
    '        return "aac"\n'
)

_REORDERED_ONLY_HEAD = (
    '    if data.startswith(b"ID3") or (\n'
    '        len(data) >= 2\n'
    '        and data[0] == 0xFF\n'
    '        and (data[1] & 0xE0) == 0xE0\n'
    '        and (data[1] & 0x18) != 0x08\n'
    '        and (data[1] & 0x06) != 0\n'
    '    ):\n'
    '        return "mp3"\n'
    '    if data.startswith(b"ADIF") or (\n'
    '        len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xF6) == 0xF0\n'
    '    ):\n'
    '        return "aac"\n'
)

_NO_LAYER_HEAD = _FIXED_HEAD.replace("        and (data[1] & 0x06) != 0\n", "")
_NO_VERSION_HEAD = _FIXED_HEAD.replace("        and (data[1] & 0x18) != 0x08\n", "")

# 每条变异臂：(名称描述, 旧片段, 新片段, 必红核心断言集合, 有真实 FFmpeg 时同步变红的断言集合)
_MUTATION_ARMS = {
    "M1": (
        "回退为缺陷原状（无 layer/version 校验的 MP3 置于 ADTS 之前）",
        _FIXED_HEAD,
        _REVERTED_HEAD,
        {"A1", "A2", "A4", "A5", "A6"},
        {"A7"},
    ),
    "M2": (
        "将 MP3 判据挪回 ADTS 之前（保留掩码校验，检验分支序守卫）",
        _FIXED_HEAD,
        _REORDERED_ONLY_HEAD,
        {"A6"},
        set(),
    ),
    "M3": (
        "去掉 MP3 的 layer != 0 校验",
        _FIXED_HEAD,
        _NO_LAYER_HEAD,
        {"A4"},
        set(),
    ),
    "M4": (
        "去掉 MP3 的 MPEG version != 01 保留位校验",
        _FIXED_HEAD,
        _NO_VERSION_HEAD,
        {"A5"},
        set(),
    ),
}


def _clear_audio_pyc() -> None:
    cache_dir = _AUDIO_FILE.parent / "__pycache__"
    if cache_dir.is_dir():
        for pyc in cache_dir.glob("audio.*.pyc"):
            pyc.unlink(missing_ok=True)


def _run_assertions() -> int:
    failed = 0
    for code, fn in _ASSERTION_LIST:
        try:
            fn()
        except Exception as exc:
            failed += 1
            print("FAIL %s (%s): %s: %s" % (code, fn.__name__, type(exc).__name__, exc))
        else:
            print("PASS %s (%s)" % (code, fn.__name__))
    print("通过 %d / 失败 %d" % (len(_ASSERTION_LIST) - failed, failed))
    return failed


def _subprocess_fail_codes():
    _clear_audio_pyc()
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-B", "-X", "utf8", str(Path(__file__).resolve()), "--assert-only"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(_REPO),
        env=env,
        timeout=120,
        check=False,
        shell=False,
    )
    codes = set(re.findall(r"(?m)^FAIL\s+(A\d+)\b", proc.stdout or ""))
    return codes, (proc.stdout or "") + (proc.stderr or "")


def _run_mutations(argv) -> int:
    want = ""
    for arg in argv:
        if arg == "--mutate":
            want = want or "ALL"
        elif arg.startswith("--mutate="):
            want = arg.split("=", 1)[1]
    if not want:
        return 0

    ids = sorted(_MUTATION_ARMS) if want == "ALL" else [want]
    has_real_aac = _real_ffmpeg_aac_capable()
    mismatches = 0
    for mid in ids:
        if mid not in _MUTATION_ARMS:
            print("[ANCHOR-MISS] %s: 未登记该变异臂" % mid)
            return 2
        desc, old_snippet, new_snippet, expect_core, expect_ff = _MUTATION_ARMS[mid]
        expect = set(expect_core) | (set(expect_ff) if has_real_aac else set())
        orig_bytes = _AUDIO_FILE.read_bytes()
        try:
            text = orig_bytes.decode("utf-8")
            nl = "\r\n" if "\r\n" in text else "\n"
            norm = text.replace("\r\n", "\n")
            hits = norm.count(old_snippet)
            if hits != 1:
                print("[ANCHOR-MISS] %s: 锚点命中 %d 次（期望 1）" % (mid, hits))
                return 2
            mutated = norm.replace(old_snippet, new_snippet, 1).replace("\n", nl)
            _AUDIO_FILE.write_bytes(mutated.encode("utf-8"))
            actual, out = _subprocess_fail_codes()
            if actual == expect:
                print("[MUTATE-OK]   %s (%s): 恰好转红 %s" % (mid, desc, sorted(actual)))
            else:
                print(
                    "[MUTATE-FAIL] %s (%s): 期望红 %s，实际红 %s"
                    % (mid, desc, sorted(expect), sorted(actual))
                )
                for line in out.splitlines():
                    if line.startswith("FAIL "):
                        print("    " + line)
                mismatches += 1
        finally:
            _AUDIO_FILE.write_bytes(orig_bytes)
            _clear_audio_pyc()
    print(
        "变异自证汇总：%d 条变异，%s（FFmpeg ADTS 端到端=%s）"
        % (
            len(ids),
            "全部精确命中" if mismatches == 0 else "%d 条偏差" % mismatches,
            "已启用" if has_real_aac else "未安装自动跳过",
        )
    )
    return 1 if mismatches else 0


if __name__ == "__main__":
    failures = _run_assertions()
    if "--assert-only" in sys.argv:
        sys.exit(1 if failures else 0)
    mut_rc = _run_mutations(sys.argv[1:])
    sys.exit(1 if (failures or mut_rc) else 0)
