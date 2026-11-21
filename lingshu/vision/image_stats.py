# -*- coding: utf-8 -*-
"""Bounded pixel measurements for an already decoded Pillow image.

Pillow is imported only when analyze_image_pixels is called. No file or
network access, resizing, model calls, or ICC/gamma color management is used.
"""
from __future__ import annotations

__all__ = ["analyze_image_pixels"]

_MAX_PIXELS = 4_194_304


def analyze_image_pixels(image: "PIL.Image.Image") -> dict:
    """Measure the input at its original size and return JSON-safe summaries.

    visible_rgb maps r, g, and b to 16-bin count lists after RGBA conversion
    and alpha compositing over white. luminance measures that visible RGB
    image using Pillow's encoded 8-bit convert("L") samples, with population
    standard deviation. alpha describes the original converted RGBA alpha
    channel. Histogram bins cover 0..15 through 240..255; no pixels or image
    metadata are returned.

    Raises TypeError for non-Pillow inputs, or ValueError for non-positive,
    non-integer dimensions and images exceeding 4,194,304 pixels. These size
    checks happen before image conversion or pixel-buffer allocation. Lazy or
    closed images are rejected: the caller must load the current frame first.
    """
    from PIL import Image, ImageStat

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")

    width, height = image.size
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive integers")
    pixel_count = width * height
    if pixel_count > _MAX_PIXELS:
        raise ValueError("image exceeds the 4,194,304 pixel limit")

    # Pillow Image.open() is lazy: convert() would otherwise read its file/stream
    # and mutate the caller's decode state. Inspect the stored core without calling
    # load()/getdata()/im (which can trigger decoding). Pillow 12.3 is the tested floor.
    core = getattr(image, "_im", None)
    if core is None or isinstance(core, Image.DeferredError) or getattr(image, "tile", ()):
        raise ValueError("image must already be decoded; call image.load() before analysis")

    source_mode = image.mode
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    alpha_histogram = alpha.histogram()
    alpha_stat = ImageStat.Stat(alpha)
    visible = Image.alpha_composite(
        Image.new("RGBA", (width, height), (255, 255, 255, 255)), rgba
    ).convert("RGB")
    rgb_histogram = visible.histogram()
    luminance = visible.convert("L")
    luminance_stat = ImageStat.Stat(luminance)
    luminance_histogram = luminance.histogram()

    return {
        "width": width,
        "height": height,
        "pixel_count": pixel_count,
        "source_mode": source_mode,
        "visible_rgb": {
            channel: [
                sum(rgb_histogram[offset + start:offset + start + 16])
                for start in range(0, 256, 16)
            ]
            for channel, offset in (("r", 0), ("g", 256), ("b", 512))
        },
        "luminance": {
            "min": int(luminance_stat.extrema[0][0]),
            "max": int(luminance_stat.extrema[0][1]),
            "mean": float(luminance_stat.mean[0]),
            "stddev": float(luminance_stat.stddev[0]),
            "histogram": [
                sum(luminance_histogram[start:start + 16])
                for start in range(0, 256, 16)
            ],
        },
        "alpha": {
            "min": int(alpha_stat.extrema[0][0]),
            "max": int(alpha_stat.extrema[0][1]),
            "mean": float(alpha_stat.mean[0]),
            "fully_transparent_pixels": alpha_histogram[0],
            "partially_transparent_pixels": sum(alpha_histogram[1:255]),
            "opaque_pixels": alpha_histogram[255],
        },
    }
