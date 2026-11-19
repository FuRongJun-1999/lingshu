# -*- coding: utf-8 -*-
"""Synthetic, Pillow-only guards for bounded image pixel measurements."""
from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

from PIL import Image
import pytest

from lingshu.vision.image_stats import analyze_image_pixels


REPO = Path(__file__).resolve().parents[1]


def _histogram(*values):
    counts = [0] * 16
    for value in values:
        counts[value // 16] += 1
    return counts


def _subprocess(code):
    return subprocess.run(
        [sys.executable, "-B", "-I", "-c", code],
        cwd=REPO,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )


def test_solid_color_and_original_dimensions():
    result = analyze_image_pixels(Image.new("RGB", (3, 2), (15, 16, 255)))
    assert (result["width"], result["height"], result["pixel_count"]) == (3, 2, 6)
    assert result["source_mode"] == "RGB"
    assert result["visible_rgb"] == {
        "r": _histogram(*([15] * 6)),
        "g": _histogram(*([16] * 6)),
        "b": _histogram(*([255] * 6)),
    }
    assert result["alpha"] == {
        "min": 255, "max": 255, "mean": 255.0,
        "fully_transparent_pixels": 0, "partially_transparent_pixels": 0,
        "opaque_pixels": 6,
    }
    assert result["luminance"]["stddev"] == 0.0


def test_red_blue_blocks_and_encoded_luminance():
    image = Image.new("RGB", (5, 2), "red")
    image.paste("blue", (3, 0, 5, 2))
    result = analyze_image_pixels(image)
    assert result["visible_rgb"] == {
        "r": _histogram(*([255] * 6 + [0] * 4)),
        "g": _histogram(*([0] * 10)),
        "b": _histogram(*([0] * 6 + [255] * 4)),
    }
    luma = result["luminance"]
    assert luma["min"] == 29
    assert luma["max"] == 76
    assert luma["mean"] == pytest.approx(57.2)
    assert luma["stddev"] == pytest.approx(math.sqrt(0.6 * 0.4 * 47 ** 2))
    assert luma["histogram"] == _histogram(*([76] * 6 + [29] * 4))


def test_rgb_histogram_boundaries_include_last_bin():
    values = [0, 15, 16, 31, 32, 239, 240, 255]
    image = Image.new("RGB", (len(values), 1))
    for x, value in enumerate(values):
        image.putpixel((x, 0), (value, values[-x - 1], value))
    result = analyze_image_pixels(image)
    expected = [2, 2, 1] + [0] * 11 + [1, 2]
    assert expected == _histogram(*values)
    assert all(hist == expected for hist in result["visible_rgb"].values())


def test_grayscale_luminance_population_standard_deviation():
    image = Image.new("L", (4, 1))
    image.putdata([0, 64, 128, 255])
    luma = analyze_image_pixels(image)["luminance"]
    mean = (0 + 64 + 128 + 255) / 4
    assert luma == {
        "min": 0, "max": 255, "mean": mean,
        "stddev": pytest.approx(math.sqrt(sum((v - mean) ** 2 for v in (0, 64, 128, 255)) / 4)),
        "histogram": _histogram(0, 64, 128, 255),
    }


def test_rgba_transparency_is_composited_over_white():
    image = Image.new("RGBA", (3, 1))
    image.putdata([(255, 0, 0, 0), (255, 0, 0, 128), (0, 0, 255, 255)])
    result = analyze_image_pixels(image)
    assert result["visible_rgb"] == {
        "r": _histogram(255, 255, 0),
        "g": _histogram(255, 127, 0),
        "b": _histogram(255, 127, 255),
    }
    assert result["alpha"] == {
        "min": 0, "max": 255, "mean": pytest.approx(383 / 3),
        "fully_transparent_pixels": 1, "partially_transparent_pixels": 1,
        "opaque_pixels": 1,
    }
    # Pillow's rounded encoded L values for white, composited red, and blue.
    assert result["luminance"]["histogram"] == _histogram(255, 165, 29)


def test_alpha_classification_and_compositing_rounding_at_endpoints():
    image = Image.new("RGBA", (5, 1))
    image.putdata([(0, 0, 0, alpha) for alpha in (0, 1, 128, 254, 255)])
    result = analyze_image_pixels(image)
    assert result["alpha"] == {
        "min": 0, "max": 255, "mean": 127.6,
        "fully_transparent_pixels": 1, "partially_transparent_pixels": 3,
        "opaque_pixels": 1,
    }
    expected = _histogram(255, 254, 127, 1, 0)
    assert all(hist == expected for hist in result["visible_rgb"].values())
    assert result["luminance"]["histogram"] == expected


def test_palette_transparency_and_source_mode():
    image = Image.new("P", (3, 1))
    image.putpalette([255, 0, 0, 0, 255, 0, 0, 0, 255] + [0] * (768 - 9))
    image.putdata([0, 1, 2])
    image.info["transparency"] = bytes([0, 128, 255])
    result = analyze_image_pixels(image)
    assert result["source_mode"] == "P"
    assert result["visible_rgb"] == {
        "r": _histogram(255, 127, 0),
        "g": _histogram(255, 255, 0),
        "b": _histogram(255, 127, 255),
    }
    assert result["alpha"]["partially_transparent_pixels"] == 1


@pytest.mark.parametrize("mode,color,visible", [
    ("LA", (80, 128), 167),
    ("L", 80, 80),
    ("1", 1, 255),
    ("CMYK", (0, 0, 0, 255), 0),
])
def test_pillow_mode_conversion(mode, color, visible):
    result = analyze_image_pixels(Image.new(mode, (2, 1), color))
    assert result["source_mode"] == mode
    assert all(hist == _histogram(visible, visible) for hist in result["visible_rgb"].values())
    assert result["luminance"]["mean"] == visible
    assert result["alpha"]["mean"] == (128 if mode == "LA" else 255)


class _SizedImage(Image.Image):
    """Report dimensions without allocating a correspondingly sized buffer."""

    def __init__(self, size):
        super().__init__()
        self._size = size


@pytest.mark.parametrize("size", [(4_194_305, 1), (2048, 2049), (1, 10 ** 30)])
def test_pixel_budget_rejected_before_conversion_or_allocation(monkeypatch, size):
    image = _SizedImage(size)
    convert = Mock(side_effect=AssertionError("conversion must not happen"))
    allocate = Mock(side_effect=AssertionError("allocation must not happen"))
    monkeypatch.setattr(image, "convert", convert)
    monkeypatch.setattr(Image, "new", allocate)
    with pytest.raises(ValueError, match="4,194,304"):
        analyze_image_pixels(image)
    convert.assert_not_called()
    allocate.assert_not_called()


def test_exact_pixel_budget_reaches_conversion_without_resizing(monkeypatch):
    image = _SizedImage((2048, 2048))
    convert = Mock(side_effect=RuntimeError("reached conversion"))
    monkeypatch.setattr(image, "convert", convert)
    with pytest.raises(RuntimeError, match="reached conversion"):
        analyze_image_pixels(image)
    convert.assert_called_once_with("RGBA")


@pytest.mark.parametrize("size", [
    (0, 1), (1, 0), (-1, 2), (2, -1), (1.0, 2), (2, 1.5),
    (True, 2), (2, False), ("2", 1), (1, None),
])
def test_invalid_dimensions_rejected_before_conversion(monkeypatch, size):
    image = _SizedImage(size)
    convert = Mock(side_effect=AssertionError("conversion must not happen"))
    monkeypatch.setattr(image, "convert", convert)
    with pytest.raises(ValueError, match="positive integers"):
        analyze_image_pixels(image)
    convert.assert_not_called()


@pytest.mark.parametrize("bad_input", [None, 1, "image.png", b"pixels", {}, object()])
def test_non_pillow_inputs_rejected(bad_input):
    with pytest.raises(TypeError, match="PIL.Image.Image"):
        analyze_image_pixels(bad_input)


def test_output_is_bounded_json_safe_and_input_is_unchanged():
    image = Image.new("RGBA", (7, 3), (12, 90, 240, 128))
    image.info.update(icc_profile=b"ignored profile", gamma=0.5, filename="private")
    before = (image.mode, image.size, image.tobytes(), image.info.copy())
    result = analyze_image_pixels(image)
    assert set(result) == {
        "width", "height", "pixel_count", "source_mode", "visible_rgb", "luminance", "alpha",
    }
    assert set(result["visible_rgb"]) == {"r", "g", "b"}
    assert set(result["luminance"]) == {"min", "max", "mean", "stddev", "histogram"}
    histograms = list(result["visible_rgb"].values()) + [result["luminance"]["histogram"]]
    for histogram in histograms:
        assert len(histogram) == 16
        assert all(type(count) is int and 0 <= count <= 21 for count in histogram)
        assert sum(histogram) == result["pixel_count"]
    alpha = result["alpha"]
    assert sum(alpha[key] for key in (
        "fully_transparent_pixels", "partially_transparent_pixels", "opaque_pixels",
    )) == result["pixel_count"]
    encoded = json.dumps(result, allow_nan=False)
    assert json.loads(encoded) == result
    assert len(encoded) < 2000
    assert "private" not in encoded
    assert (image.mode, image.size, image.tobytes(), image.info) == before
    assert analyze_image_pixels(Image.new("RGBA", image.size, (12, 90, 240, 128))) == result


def test_module_import_is_lazy_and_measurement_does_not_need_numpy():
    source = REPO / "lingshu" / "vision" / "image_stats.py"
    code = f"""
import importlib.abc
import importlib.util
import sys
class RejectNumpy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'numpy' or fullname.startswith('numpy.'):
            raise AssertionError('NumPy must not be imported')
sys.meta_path.insert(0, RejectNumpy())
spec = importlib.util.spec_from_file_location('isolated_image_stats', {str(source)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert not any(name == 'PIL' or name.startswith('PIL.') for name in sys.modules)
from PIL import Image
result = module.analyze_image_pixels(Image.new('RGB', (1, 1), 'red'))
assert result['pixel_count'] == 1
"""
    completed = _subprocess(code)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_public_vision_import_does_not_load_pillow_or_numpy():
    # Intentionally fails until the public initializer uses lazy exports.
    code = f"""
import sys
sys.path.insert(0, {str(REPO)!r})
import lingshu.vision
loaded = sorted(name for name in sys.modules if
    name in ('PIL', 'numpy') or name.startswith(('PIL.', 'numpy.')))
assert not loaded, loaded
"""
    completed = _subprocess(code)
    assert completed.returncode == 0, completed.stdout + completed.stderr
