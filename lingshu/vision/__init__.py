# -*- coding: utf-8 -*-
"""vision · 可选的图像统计和线稿工具公共入口。"""
from __future__ import annotations

from importlib import import_module

__all__ = ["darkline", "linework", "analyze_image_pixels"]


def __getattr__(name):
    if name in {"darkline", "linework"}:
        return getattr(import_module(".lineart", __name__), name)
    if name == "analyze_image_pixels":
        return getattr(import_module(".image_stats", __name__), name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def __dir__():
    return sorted(set(globals()) | set(__all__))
