# -*- coding: utf-8 -*-
"""test_issue2399_anchor_channel_closed_set_guard —— 验证通道键闭集守卫。

缺陷（dsh-bounty #2399）
============================================================================
`anchor_verify.add_channel_evidence` 对 channel 零校验，任意字符串直接入账；
`verify_anchor` 的通道一致闸 `len(ev) >= MIN_CHANNELS` 按**字典键数**计数 ⇒
同一台视觉传感器把通道名拼错两次（visaul / vsiual）即伪造出「多通道一致」
ACCEPT_weak。而 #160 引入 MIN_CHANNELS 的全部目的就是挡「单通道自证陷阱」
（模块头 7-10）——闸的语义从「独立通道数」退化为「不同字符串数」：拼对
`visual` 只算 1 通道 → NOT_ACCEPTED；拼错两次 → ACCEPT_weak（方向与
fail-closed 相反，且无任何告警）。

实现侧口径（本守卫的判据）
--------------------------------------------------------------------------
通道键闭集 `KNOWN_CHANNELS = STRONG_CHANNELS | WEAK_CHANNELS`（覆盖 docstring
七通道）；域外键 `add_channel_evidence` raise ValueError（对齐同仓
physics.MODEL_KINDS / shapes.PART_KINDS 闭集 fail-loud 惯例）；门面
`world3d("verify")` 转 error dict（同门面 fail-closed 惯例，不把 ValueError
抛给模型驱动面）。

守卫用例
--------------------------------------------------------------------------
G1 缺陷态：同一锚点两次拼错通道（visaul/vsiual）⇒ 门面回 error，不再
   ACCEPT_weak（改前实测 ACCEPT_weak）。
G2 底层：add_channel_evidence 对域外键 raise ValueError（文案点名坏键与闭集）。
G3 正对照：七合法通道全部可入账（防一刀切假绿）；两真实通道仍能 ACCEPT。
G4 范围控制：拼错被拒后零残留——坏键不进 channel_evidence，后续合法两通道
   的判定不受影响（不因一次错拼永久降级）。
G5 语义面：拼对单通道仍 NOT_ACCEPTED（#160 自证陷阱闸原样有效）。

运行（lingshu 仓根）：
    python -X utf8 -m pytest tests/test_issue2399_anchor_channel_closed_set_guard.py -q
"""
from __future__ import annotations

import pytest

from lingshu.core.core import SpacetimeMemoryEngine
from lingshu.world.anchor_verify import KNOWN_CHANNELS, STRONG_CHANNELS, WEAK_CHANNELS


def _engine_with_anchor():
    e = SpacetimeMemoryEngine(":memory:")
    r = e.world3d("add", {"category": "box", "bbox": [380, 280, 420, 320]})
    assert r.get("status") == "ok", r
    g = e._world3d
    g.build_anchor_graph()
    return e, list(g._anchor_graph.anchors)[0]


# ---------------------------------------------------------------- G1

def test_2399_typo_channels_no_longer_accept():
    """改前：visaul + vsiual 两个拼错键 ⇒ ACCEPT_weak（伪造多通道）。"""
    e, aid = _engine_with_anchor()
    r1 = e.world3d("verify", {"anchor_id": aid, "channel": "visaul", "evidence": 0.9})
    r2 = e.world3d("verify", {"anchor_id": aid, "channel": "vsiual", "evidence": 0.9})
    assert r1.get("status") == "error", (
        "#2399 拼错通道 'visaul' 仍被受理：%r" % (r1,))
    assert r2.get("status") == "error", (
        "#2399 拼错通道 'vsiual' 仍被受理（改前两次错拼凑出 ACCEPT_weak）：%r" % (r2,))


# ---------------------------------------------------------------- G2

def test_2399_low_level_raises_with_readable_message():
    from lingshu.world.anchor_verify import AnchorVerification
    v = AnchorVerification()
    with pytest.raises(ValueError) as ei:
        v.add_channel_evidence("A", "visaul", 0.9)
    msg = str(ei.value)
    assert "visaul" in msg and "visual" in msg, (
        "错误文案须点名坏键与闭集（可判读）：%r" % (msg,))


# ---------------------------------------------------------------- G3

def test_2399_all_known_channels_accepted():
    """七合法通道全部可入账（防一刀切假绿）。"""
    from lingshu.world.anchor_verify import AnchorVerification
    assert KNOWN_CHANNELS == STRONG_CHANNELS | WEAK_CHANNELS
    v = AnchorVerification()
    for ch in sorted(KNOWN_CHANNELS):
        v.add_channel_evidence("A", ch, 0.9)
    rec = v._anchors["A"]["channel_evidence"]
    assert set(rec) == set(KNOWN_CHANNELS), (
        "合法通道入账集合应等于闭集：%r" % (sorted(rec),))


def test_2399_two_real_channels_still_accept():
    e, aid = _engine_with_anchor()
    assert e.world3d("verify", {"anchor_id": aid, "channel": "visual",
                                "evidence": 0.9})["status"] == "ok"
    r = e.world3d("verify", {"anchor_id": aid, "channel": "tactile",
                             "evidence": 0.9})
    assert r["verification"]["confirmation"] in ("ACCEPT_weak", "ACCEPT_strong",
                                                 "ACCEPT_stable"), r


# ---------------------------------------------------------------- G4

def test_2399_rejected_key_leaves_no_residue():
    """错拼被拒后零残留：坏键不进 channel_evidence；后续合法两通道照常判定。"""
    e, aid = _engine_with_anchor()
    e.world3d("verify", {"anchor_id": aid, "channel": "visaul", "evidence": 0.9})
    v = e._world3d._anchor_graph._verifier
    rec = v._anchors.get(aid, {})
    assert "visaul" not in rec.get("channel_evidence", {}), (
        "坏键残留进 channel_evidence：%r" % (sorted(rec.get("channel_evidence", {})),))
    e.world3d("verify", {"anchor_id": aid, "channel": "visual", "evidence": 0.9})
    r = e.world3d("verify", {"anchor_id": aid, "channel": "tactile", "evidence": 0.9})
    assert r["verification"]["confirmation"] != "NOT_ACCEPTED", r


# ---------------------------------------------------------------- G5

def test_2399_single_channel_still_not_accepted():
    """#160 自证陷阱闸原样有效：拼对的单通道仍 NOT_ACCEPTED。"""
    e, aid = _engine_with_anchor()
    r = e.world3d("verify", {"anchor_id": aid, "channel": "visual", "evidence": 0.9})
    assert r["verification"]["confirmation"] == "NOT_ACCEPTED", (
        "#160 单通道不得确认：%r" % (r["verification"],))
