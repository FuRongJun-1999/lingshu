"""Issue #1507 LS-03: event boundaries follow the sampled WAV window grid."""
from __future__ import annotations

import io
import struct
import wave

import pytest

from lingshu.vision.audio import analyze_wav, segment_audio_events


def _wav(silent_frames, active_frames, rate):
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(b"\x00\x00" * silent_frames
                           + struct.pack("<h", 32767) * active_frames)
    return output.getvalue()


def _assert_ordered_within_duration(profile, duration):
    previous_end = 0.0
    for segment in profile["segments"]:
        assert previous_end == segment["start_s"]
        assert 0.0 <= segment["start_s"] <= segment["end_s"] <= duration
        previous_end = segment["end_s"]


@pytest.mark.parametrize("rate,window_ms,window_frames,tail_frames", [
    (11025, 10, 110, 1),
    (11025, 30, 331, 37),
    (22050, 10, 220, 1),
    (8000, 100, 800, 17),
])
def test_real_wav_transition_matches_sample_boundary(rate, window_ms, window_frames, tail_frames):
    silent_frames = 101 * window_frames
    features = analyze_wav(_wav(silent_frames, tail_frames, rate), window_ms=window_ms)
    assert features["analysis_window_frames"] == window_frames

    profile = segment_audio_events(features)
    silence, activity = profile["segments"]
    expected_transition = round(silent_frames / rate, 4)
    assert silence["kind"] == "silence" and activity["kind"] == "activity"
    assert silence["end_s"] == activity["start_s"] == expected_transition
    assert activity["end_s"] == features["duration_s"]
    assert profile["resolution_s"] == pytest.approx(window_frames / rate, abs=1e-9)
    assert profile["approximate"] is False
    _assert_ordered_within_duration(profile, features["duration_s"])


def test_rounding_duration_does_not_drop_real_partial_window():
    rate = 48000
    window_frames = 480
    features = analyze_wav(_wav(window_frames, 1, rate), window_ms=10)
    assert features["duration_s"] == 0.01
    assert features["frames"] == window_frames + 1

    profile = segment_audio_events(features)
    assert [segment["kind"] for segment in profile["segments"]] == ["silence", "activity"]
    assert profile["segments"][-1]["start_s"] == profile["segments"][-1]["end_s"] == 0.01


def test_real_downsampled_envelope_uses_actual_window_grid():
    rate = 11025
    # 2,401 windows retain every third window, including the short active tail.
    silent_frames = 2400 * 110
    features = analyze_wav(_wav(silent_frames, 7, rate), window_ms=10)
    assert features["energy_envelope_stride_windows"] == 3

    profile = segment_audio_events(features)
    silence, activity = profile["segments"]
    assert silence["end_s"] == activity["start_s"] == round(silent_frames / rate, 4)
    assert activity["end_s"] == features["duration_s"]
    assert profile["resolution_s"] == pytest.approx(3 * 110 / rate, abs=1e-9)
    assert profile["approximate"] is True
    _assert_ordered_within_duration(profile, features["duration_s"])


def test_long_downsampled_envelope_keeps_tail_despite_rounded_window_seconds():
    # This mapping describes a ~61 MiB WAV within the configurable input budget.
    rate = 48001
    window_frames = 480
    frame_count = 63612 * window_frames + 1
    features = {
        "duration_s": round(frame_count / rate, 4),
        "sample_rate_hz": rate,
        "frames": frame_count,
        "window_ms": 10,
        "analysis_window_frames": window_frames,
        "analysis_window_s": round(window_frames / rate, 9),
        "energy_envelope_stride_windows": 54,
        "energy_envelope_dbfs": [-120.0] * 1178 + [-10.0],
    }
    profile = segment_audio_events(features)
    assert [s["kind"] for s in profile["segments"]] == ["silence", "activity"]
    assert profile["segments"][-1]["start_s"] == min(
        features["duration_s"], round(63612 * window_frames / rate, 4))
    _assert_ordered_within_duration(profile, features["duration_s"])


def test_legacy_features_keep_nominal_window_and_bound_every_endpoint():
    features = {
        "duration_s": 0.01506,
        "window_ms": 10,
        "energy_envelope_stride_windows": 1,
        "energy_envelope_dbfs": [-120.0, -10.0, -120.0],
    }
    profile = segment_audio_events(features)
    assert [(s["start_s"], s["end_s"]) for s in profile["segments"]] == [
        (0.0, 0.01), (0.01, 0.01506),
    ]
    assert profile["resolution_s"] == 0.01
    _assert_ordered_within_duration(profile, features["duration_s"])


@pytest.mark.parametrize("window_s,error", [
    (float("nan"), ValueError),
    (float("inf"), ValueError),
    (0.0, ValueError),
    (-0.01, ValueError),
    (1.1, ValueError),
    (True, TypeError),
    ("0.01", TypeError),
    (None, TypeError),
])
def test_invalid_actual_window_is_rejected_instead_of_using_nominal(window_s, error):
    features = {
        "duration_s": 0.02,
        "window_ms": 10,
        "analysis_window_s": window_s,
        "energy_envelope_stride_windows": 1,
        "energy_envelope_dbfs": [-120.0, -10.0],
    }
    with pytest.raises(error, match="analysis_window_s"):
        segment_audio_events(features)
