# -*- coding: utf-8 -*-
"""Camera3D / observation contract round-trips.

This is intentionally a test-only contribution.  Atta's useful boundary here is
"facts before pixels": camera metadata and projection inverses must be stable
before a render is treated as evidence.  The tests exercise the already-public
world3d contract without changing the production implementation.
"""
from __future__ import annotations

import math

import pytest

from lingshu.world.world3d import (
    Camera3D,
    CameraMetaError,
    format_camera_observation,
    parse_camera_observation,
)


def _assert_meta_close(left, right):
    """Compare JSON-shaped camera metadata without accepting missing fields."""
    assert set(left) == set(right)
    for key in left:
        if key == "basis":
            assert len(left[key]) == len(right[key]) == 3
            for arow, brow in zip(left[key], right[key]):
                assert arow == pytest.approx(brow, abs=1e-14)
        else:
            assert left[key] == pytest.approx(right[key], abs=1e-14)


def test_camera_meta_roundtrip_preserves_roll_and_explicit_basis():
    """Optional orientation fields must survive the persisted metadata path."""
    cam = Camera3D(
        fov_deg=53.25,
        cx=1.25,
        cy=-0.75,
        cz=3.5,
        yaw=0.2,
        pitch=-0.31,
        roll=0.37,
    )
    restored = Camera3D.from_meta(cam.to_meta(), strict=True)
    _assert_meta_close(restored.to_meta(), cam.to_meta())

    basis_cam = Camera3D(
        fov_deg=61.0,
        cx=-2.0,
        cy=1.0,
        cz=4.0,
        basis=((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)),
    )
    basis_restored = Camera3D.from_meta(basis_cam.to_meta(), strict=True)
    _assert_meta_close(basis_restored.to_meta(), basis_cam.to_meta())


def test_project_unproject_roundtrip_with_roll():
    """The projection inverse remains exact for a rolled camera."""
    cam = Camera3D(
        fov_deg=48.0,
        cx=0.5,
        cy=1.2,
        cz=-0.25,
        yaw=-0.22,
        pitch=0.17,
        roll=0.41,
    )
    point = (2.25, 1.8, 7.5)
    px = cam.project(point, 1280, 720)
    assert px is not None
    depth = cam.depth_of(point)
    restored = cam.unproject(px[0], px[1], depth, 1280, 720)
    assert restored == pytest.approx(point, abs=1e-12)


def test_project_unproject_roundtrip_with_explicit_basis():
    """The explicit right-handed basis path has the same inverse contract."""
    cam = Camera3D(
        fov_deg=60.0,
        cx=1.0,
        cy=2.0,
        cz=3.0,
        basis=((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)),
    )
    point = (-4.0, 2.5, 1.5)
    px = cam.project(point, 800, 600)
    assert px is not None
    restored = cam.unproject(px[0], px[1], cam.depth_of(point), 800, 600)
    assert restored == pytest.approx(point, abs=1e-12)


def test_project_unproject_roundtrip_with_basis_and_ignored_roll():
    cam = Camera3D(
        fov_deg=57.0,
        cx=1.0,
        cy=2.0,
        cz=3.0,
        roll=0.41,
        basis=((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)),
    )
    point = (-4.0, 2.5, 1.5)
    px = cam.project(point, 800, 600)
    assert px is not None
    restored = cam.unproject(px[0], px[1], cam.depth_of(point), 800, 600)
    assert restored == pytest.approx(point, abs=1e-12)


def test_format_parse_camera_observation_roundtrip():
    """The persisted observation carries the same camera and scalar fields."""
    cam = Camera3D(
        fov_deg=55.0,
        cx=0.3,
        cy=1.4,
        cz=-1.0,
        yaw=0.4,
        pitch=-0.2,
        roll=0.15,
    )
    packed = format_camera_observation(
        "flange",
        (10.25, 20.5, 70.75, 90.125),
        cam,
        640,
        480,
        eid="part-7",
        tick=12,
        confidence=0.875,
        ts=1234.5,
        source="atta-replay",
    )
    parsed = parse_camera_observation(packed["content"])
    assert parsed is not None
    assert parsed["v"] == "lingshu.camera.v1"
    assert parsed["category"] == "flange"
    assert parsed["bbox"] == pytest.approx([10.25, 20.5, 70.75, 90.125])
    assert parsed["eid"] == "part-7"
    assert parsed["tick"] == 12
    assert parsed["screen_w"] == 640
    assert parsed["screen_h"] == 480
    assert parsed["confidence"] == pytest.approx(0.875)
    assert parsed["ts"] == pytest.approx(1234.5)
    assert parsed["source"] == "atta-replay"
    _assert_meta_close(parsed["camera"], cam.to_meta())


def test_observation_without_camera_block_is_not_a_bad_pose():
    """No block is a missing observation, not malformed metadata."""
    assert parse_camera_observation("plain visual observation") is None


def _mutate_meta(content, mutate):
    import json
    prefix, body = content.split("```lingshu-camera\n", 1)
    raw, suffix = body.split("\n```", 1)
    doc = json.loads(raw)
    mutate(doc)
    return prefix + "```lingshu-camera\n" + json.dumps(doc) + "\n```" + suffix


@pytest.mark.parametrize(
    "mutator, expected",
    [
        (lambda m: _mutate_meta(m, lambda d: d.update(v="unknown")), "版本"),
        (lambda m: _mutate_meta(m, lambda d: d["camera"].pop("fov_deg")), "缺字段"),
        (lambda m: _mutate_meta(m, lambda d: d["bbox"].__setitem__(0, float("nan"))), "有限"),
        (lambda m: m.rsplit("\n```", 1)[0], "未闭合"),
    ],
)
def test_parse_camera_observation_rejects_malformed_blocks(mutator, expected):
    cam = Camera3D()
    packed = format_camera_observation("box", (1, 2, 3, 4), cam, 320, 240)
    broken = mutator(packed["content"])
    with pytest.raises(CameraMetaError) as exc:
        parse_camera_observation(broken)
    assert expected in str(exc.value), str(exc.value)


def test_parse_camera_observation_rejects_ambiguous_multiple_anchors():
    cam = Camera3D()
    packed = format_camera_observation("box", (1, 2, 3, 4), cam, 320, 240)
    head, block = packed["content"].split("\n```lingshu-camera\n", 1)
    ambiguous = head + "；" + head + "\n```lingshu-camera\n" + block
    with pytest.raises(CameraMetaError, match="视觉锚点"):
        parse_camera_observation(ambiguous)


@pytest.mark.parametrize(
    "bad",
    [
        {"fov_deg": 60.0, "cx": 0.0, "cy": 0.0, "cz": 0.0, "yaw": 0.0},
        {"fov_deg": 60.0, "cx": math.nan, "cy": 0.0, "cz": 0.0, "yaw": 0.0, "pitch": 0.0},
        {"fov_deg": 60.0, "cx": 0.0, "cy": 0.0, "cz": 0.0, "yaw": 0.0, "pitch": math.inf},
        {"fov_deg": 180.0, "cx": 0.0, "cy": 0.0, "cz": 0.0, "yaw": 0.0, "pitch": 0.0},
    ],
)
def test_from_meta_is_fail_closed_for_missing_or_nonfinite_pose(bad):
    with pytest.raises(CameraMetaError):
        Camera3D.from_meta(bad, strict=True)
