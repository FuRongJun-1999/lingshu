"""Transparent input pixels must use the line-art API's white background."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lingshu.vision import darkline, linework


def _rgba_line(alpha=255):
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    ImageDraw.Draw(image).line((8, 16, 23, 16), fill=(0, 0, 0, alpha), width=1)
    return image


def _palette_line():
    image = Image.new("P", (32, 32), 0)
    image.putpalette([0, 0, 0, 0, 0, 0] + [0, 0, 0] * 254)
    image.info["transparency"] = 0
    ImageDraw.Draw(image).line((8, 16, 23, 16), fill=1, width=1)
    return image


def _white_composite(image):
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")


@pytest.mark.parametrize("factory", [_rgba_line, _palette_line], ids=["rgba", "palette"])
def test_transparent_pixels_are_composited_over_white(factory):
    image = factory()
    assert int(np.count_nonzero(np.asarray(image.convert("RGBA"))[..., 3])) == 16

    expected = darkline(_white_composite(image))
    actual = darkline(image)
    assert np.array_equal(np.asarray(actual), np.asarray(expected)), \
        "alpha input must be composited over white"
    assert int(np.count_nonzero(np.asarray(actual) == 0)) == 36
    assert np.array_equal(np.asarray(linework(image)), np.asarray(expected.convert("RGB")))


def test_opaque_rgba_keeps_rgb_results():
    rgb = Image.new("RGB", (32, 32), "white")
    ImageDraw.Draw(rgb).line((8, 16, 23, 16), fill="black", width=1)
    assert np.array_equal(np.asarray(darkline(rgb)), np.asarray(darkline(rgb.convert("RGBA"))))


def test_partial_alpha_is_composited_over_white():
    image = _rgba_line(alpha=128)
    before = (image.tobytes(), image.info.copy())
    actual = darkline(image)
    expected = darkline(_white_composite(image))
    assert np.array_equal(np.asarray(actual), np.asarray(expected))
    assert (np.asarray(actual) == 0).any()
    assert (image.tobytes(), image.info) == before
