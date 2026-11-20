"""Transparent input pixels must use the line-art API's white background."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lingshu.vision import darkline, linework


ALPHA_MODES = ["RGBA", "RGBa", "LA", "La", "PA", "P"]
ALPHA_IDS = ["rgba", "premultiplied_rgba", "la", "premultiplied_la", "pa", "palette"]


def _alpha_line(mode, alpha=255, opaque=False):
    if mode == "P":
        image = Image.new("P", (32, 32), 0)
        background = [255, 255, 255] if opaque else [0, 0, 0]
        image.putpalette(background + [0, 0, 0] * 255)
        image.info["transparency"] = bytes([255 if opaque else 0, alpha] + [255] * 254)
        ImageDraw.Draw(image).line((8, 16, 23, 16), fill=1, width=1)
        return image
    background = (255, 255, 255, 255) if opaque else (0, 0, 0, 0)
    image = Image.new("RGBA", (32, 32), background)
    ImageDraw.Draw(image).line((8, 16, 23, 16), fill=(0, 0, 0, alpha), width=1)
    if mode == "La":
        return image.convert("LA").convert("La")
    return image.convert(mode)


def _white_composite(image):
    # Pillow cannot convert premultiplied luminance directly to RGBA.
    straight = image.convert("LA") if image.mode == "La" else image
    rgba = straight.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")


def _snapshot(image):
    return image.mode, image.size, image.tobytes(), image.info.copy(), image.getpalette()


@pytest.mark.parametrize("mode", ALPHA_MODES, ids=ALPHA_IDS)
def test_transparent_pixels_are_composited_over_white(mode):
    image = _alpha_line(mode)
    assert int(np.count_nonzero(np.asarray(_white_composite(image)) < 255)) == 48

    expected = darkline(_white_composite(image))
    actual = darkline(image)
    assert np.array_equal(np.asarray(actual), np.asarray(expected)), \
        "alpha input must be composited over white"
    assert int(np.count_nonzero(np.asarray(actual) == 0)) == 36
    assert np.array_equal(np.asarray(linework(image)), np.asarray(expected.convert("RGB")))


def test_palette_transparency_index_is_composited_over_white():
    image = _alpha_line("P")
    image.info["transparency"] = 0
    actual = darkline(image)
    expected = darkline(_white_composite(image))
    assert np.array_equal(np.asarray(actual), np.asarray(expected)), \
        "palette transparency index must be composited over white"


@pytest.mark.parametrize("mode", ALPHA_MODES, ids=ALPHA_IDS)
def test_opaque_alpha_keeps_rgb_results(mode):
    rgb = Image.new("RGB", (32, 32), "white")
    ImageDraw.Draw(rgb).line((8, 16, 23, 16), fill="black", width=1)
    image = _alpha_line(mode, opaque=True)
    assert np.array_equal(np.asarray(darkline(rgb)), np.asarray(darkline(image))), \
        "opaque alpha input must preserve RGB results"


@pytest.mark.parametrize("mode", ALPHA_MODES, ids=ALPHA_IDS)
def test_partial_alpha_is_composited_over_white(mode):
    image = _alpha_line(mode, alpha=128)
    actual = darkline(image)
    expected = darkline(_white_composite(image))
    assert np.array_equal(np.asarray(actual), np.asarray(expected)), \
        "partial alpha input must be composited over white"
    assert (np.asarray(actual) == 0).any()
    assert np.array_equal(np.asarray(darkline(image, white_bg=False)), 255 - np.asarray(expected))


@pytest.mark.parametrize("mode", ALPHA_MODES, ids=ALPHA_IDS)
def test_alpha_input_is_not_modified(mode):
    image = _alpha_line(mode, alpha=128)
    image.info["lineart_test"] = "preserve metadata"
    before = _snapshot(image)
    darkline(image)
    linework(image)
    assert _snapshot(image) == before, "alpha input must remain unchanged"
