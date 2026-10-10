# -*- coding: utf-8 -*-
"""#416 world3d 渲染尺寸无上限直接进 Image.new —— 守卫

缺陷（lingshu issue #416 · [core/world3d 门面][资源耗尽]）：
  `world3d` 门面把用户参数 `screen_w/screen_h` 原样透传
  （`sw = int(p.get("screen_w", 800))`），`World3D.render` 直接
  `Image.new("RGB", (screen_w, screen_h))` —— 位图按 w*h*3 字节分配，
  任意大尺寸（如 10^6×10^6 ≈ 3TB）即内存耗尽。旧实现不设上限、静默分配。

修法（fail-closed，上限真源 `world3d.MAX_SCREEN_DIM = 4096`）：
  ① `World3D.render` 在 `Image.new` **之前**判上限，超限抛 `ValueError`（点名 #416）；
  ② `world3d` 门面在分配前先判，超限返回 `{"status": "error"}`（与门面既有校验
     惯例一致）。**不**静默钳到上限（静默钳会让调用方以为拿到了请求的尺寸）。

判据来源：经验标定（本件 #416）——仓内无规定渲染尺寸上限的理论章节，追不到。

断言组（抽掉修复＝删掉两处上限判据 ⇒ A/B/C 全红）：
  A 门面超限 ⇒ error dict 点名 #416，且 `PIL.Image.new` **零调用**（未分配）
  B 直调 `World3D.render` 超限 ⇒ `ValueError` 点名 #416，且零分配
  C 防误杀：800×600 正常渲染出 (800,600) 位图；边界值 MAX 放行、MAX+1 拒绝

运行（仓根）：python -X utf8 -m pytest tests/test_issue416_world3d_screen_cap_guard.py -q --no-header
"""
from __future__ import annotations

import os
import sys
from unittest import mock

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.core.core import SpacetimeMemoryEngine  # noqa: E402
from lingshu.world.world3d import World3D, MAX_SCREEN_DIM  # noqa: E402

HUGE = 10 ** 6


def test_a_门面超限报错且零分配():
    """A：门面超限 ⇒ error dict 点名 #416；`PIL.Image.new` 一次都没被调用。"""
    eng = SpacetimeMemoryEngine(":memory:")
    with mock.patch("PIL.Image.new", side_effect=AssertionError("发生了位图分配")) as m:
        for params in ({"screen_w": HUGE}, {"screen_h": HUGE},
                       {"screen_w": HUGE, "screen_h": HUGE}):
            r = eng.world3d("render", params)
            assert r.get("status") == "error", (
                "A 门面未拒绝超限尺寸（旧缺陷行为）：%r -> %r" % (params, r))
            assert "#416" in str(r.get("error", "")), (
                "A 拒绝理由未点名 #416：%r" % (r,))
        assert m.call_count == 0, "A 拒绝前已分配位图：Image.new 调用 %d 次" % m.call_count


def test_b_直调render超限抛错且零分配():
    """B：直调 `World3D.render` 超限 ⇒ ValueError 点名 #416；零分配。"""
    w = World3D()
    with mock.patch("PIL.Image.new", side_effect=AssertionError("发生了位图分配")) as m:
        for kw in ({"screen_w": HUGE, "screen_h": 600},
                   {"screen_w": 600, "screen_h": HUGE}):
            try:
                w.render(**kw)
                raise AssertionError("B 直调 render 未拒超限（旧缺陷行为）：%r" % (kw,))
            except ValueError as e:
                assert "#416" in str(e), "B 拒绝理由未点名 #416：%r" % (str(e),)
        assert m.call_count == 0, "B 拒绝前已分配位图：Image.new 调用 %d 次" % m.call_count


def test_c_正常尺寸与边界照旧():
    """C：防误杀——800×600 正常渲染；边界 MAX 放行、MAX+1 拒绝。"""
    w = World3D()
    img = w.render(800, 600)
    assert img.size == (800, 600), "C 正常渲染尺寸退化：%r" % (img.size,)

    # 边界：正好 MAX 放行（用 1 像素高避免大分配），MAX+1 拒绝
    ok = w.render(MAX_SCREEN_DIM, 1)
    assert ok.size == (MAX_SCREEN_DIM, 1), "C 边界 MAX 被误拒：%r" % (ok.size,)
    try:
        w.render(MAX_SCREEN_DIM + 1, 1)
        raise AssertionError("C 边界 MAX+1 未被拒（旧缺陷行为）")
    except ValueError as e:
        assert "#416" in str(e), "C 边界拒绝理由未点名 #416：%r" % (str(e),)

    # 门面正常路径照旧
    eng = SpacetimeMemoryEngine(":memory:")
    r = eng.world3d("render", {"screen_w": 320, "screen_h": 240})
    assert r.get("status") == "ok" and r.get("in_memory"), (
        "C 门面正常渲染退化：%r" % (r,))
