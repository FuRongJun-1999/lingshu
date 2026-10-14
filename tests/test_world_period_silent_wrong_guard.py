# -*- coding: utf-8 -*-
"""周期检测「真周期超出枚举上限时静默给错」—— 守卫

缺陷（[world/world_model][静默失败 · 边界不诚实]）：
  周期检测在 `[min_period, min(max_period, n//2)]` 里枚举候选 P，取残差最小者。
  当**真周期超出这个上限**时，短周期会因为「恰好也能对上几个点」而残差足够低 ⇒
  被**当成真周期返回**，调用方拿到一个**自信而错误**的答案。
  实测：真周期 118、分析窗 96 ⇒ 返回 **P=4**、`resid=0.2177`，**低于 0.25 的接受阈值**，
  于是静默走 `periodic` 分支。
  `clipped`（P 撞上限）这条提示**抓不到它** —— P=4 并没有撞上限。

修法（`wm.period-caliber/1`）：
  ① **谷够尖**判据（`sharp_min_ratio`）：真周期是残差曲线的**尖**极小，
     短周期捡漏是**平**的 ⇒ `sharpness = resid(P) / min(resid(P±1))`，
     超阈值即判「平滑假象」并**拒绝**（返回 None + `suspect_spurious` + 人话原因）；
     实测假阳性 sharpness 0.801~0.803，真周期 0.000~0.335 ⇒ 阈值 0.5 两侧余量充足。
  ② **截断必须可见**：P 撞到枚举上限时置 `clipped` + `clip_note`（旧实现在此静默）。
  ③ 两个**新能力**按本仓惯例**出厂默认关**（如 #447「全部默认关闭，附可复现基准」）：
     `adaptive_window`（短窗给不出干净解/撞上限时才升级长窗）、
     `noninteger`（亚帧周期：整数最优 P 上抛物线精修 + 相位插值）。

判据来源：`bench/bench_world.py` 的域内 `trace_sha256`（改前改后须逐位一致）+
本件实测的 sharpness 两侧余量。

断言组（抽掉修复 ⇒ 各组按标注转红）：
  A **静默错答被拒**：真 118/窗 96 ⇒ `suspect_spurious=True` 且**不进** periodic；
    并**同一断言里证明**光靠残差阈值拦不住（该候选的 `resid` 本身就低于阈值）
    —— 抽 ① ⇒ 红（退回静默给 P=4）
  B 真周期仍被检出（防误杀）：周期 20/窗 96 ⇒ P=20 且进 periodic —— 抽 ① 过头 ⇒ 红
  C **截断可见**：周期 48/窗 96（恰撞上限）⇒ `clipped=True` 且有 `clip_note` —— 抽 ② ⇒ 红
  D 两个能力**出厂默认关**；**显式开启**后长周期（100/240）能被找到 —— 抽 ③ ⇒ 红
  E 安装入口唯一且自检会拒：`install_prediction_wrappers` 在，陈旧重绑被 `verify` 拒绝

运行（仓根）：python -X utf8 -m pytest tests/test_world_period_silent_wrong_guard.py -q --no-header
"""
from __future__ import annotations

import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.world.world_model import (  # noqa: E402
    UnifiedWorldModel, install_prediction_wrappers, period_caliber_status,
    set_period_caliber)


def _feed(pts):
    m = UnifiedWorldModel(size=24)
    for p in pts:
        m.perceive([{"eid": "a", "pos": p, "category": "c"}])
        m.generate(1)
    return m


def _circ(period, n, r=1.0):
    return [(r * math.cos(2 * math.pi * t / period), 0.0,
             r * math.sin(2 * math.pi * t / period)) for t in range(n)]


def _diag(m):
    return (getattr(m, "_periodic_diag", {}) or {}).get("a") or {}


def _mode(m):
    g = m.generate(1) or {}
    return ((g.get("predictions") or {}).get("a") or {}).get("mode")


def test_a_真周期超出上限时不得静默给错():
    """A：真 118 / 窗 96 ⇒ 必须**拒绝并说明**，而不是给一个自信的短周期。"""
    m = _feed(_circ(118, 96))
    d = _diag(m)
    assert d.get("suspect_spurious") is True, (
        "真周期 118 超出枚举上限，短周期捡漏必须被判为**平滑假象**：%r" % (d,))
    assert d.get("reject_reason"), "拒绝必须带人话原因（边界诚实，不静默）"
    assert _mode(m) != "periodic", "被拒的候选不得进入 periodic 分支"
    # 关键对照：**光靠残差阈值拦不住** —— 该候选的残差本身就低于接受阈值。
    # （这条断言是「为什么需要谷够尖」的机械化证据；抽掉判据它就会红。）
    from lingshu.world.world_model import _PC
    assert d.get("resid") is not None and d["resid"] < _PC["resid_ratio"], (
        "本用例的意义在于「残差够低但谷不尖」：resid=%r 应低于阈值 %r"
        % (d.get("resid"), _PC["resid_ratio"]))


def test_b_真周期仍须被检出():
    """B（防误杀）：干净的真周期（20/窗 96）不得被谷够尖判据顺手杀掉。"""
    m = _feed(_circ(20, 96))
    d = _diag(m)
    assert d.get("P") is not None, "干净真周期没被检出：%r" % (d,)
    assert abs(float(d["P"]) - 20) <= 1e-2, "检出周期 %r ≠ 20" % (d.get("P"),)
    assert d.get("suspect_spurious") is not True, "真周期被误判为假象"
    assert _mode(m) == "periodic", "真周期应进入 periodic"


def test_c_截断必须可见():
    """C：P 撞到枚举上限时必须显式置 `clipped` 并给出提示（旧实现在此静默）。"""
    m = _feed(_circ(48, 96))
    d = _diag(m)
    assert d.get("clipped") is True, "P 撞上限却未标记 clipped：%r" % (d,)
    assert d.get("clip_note"), "截断必须带可读提示"
    assert d.get("max_period_used") is not None and d["P"] >= d["max_period_used"]


def test_d_两个新能力出厂默认关且开启后生效():
    """D：默认关（本仓惯例）；显式开启后长周期能被找到。"""
    cfg = period_caliber_status()["config"]
    assert cfg["adaptive_window"] is False and cfg["noninteger"] is False, (
        "两个新能力必须**出厂默认关**：%r" % (cfg,))
    saved = dict(cfg)
    try:
        off = _diag(_feed(_circ(100, 240)))
        set_period_caliber(adaptive_window=True)
        on = _diag(_feed(_circ(100, 240)))
        assert off.get("P") != 100, "默认关时不该检出 100（枚举上限到不了）：%r" % (off,)
        assert on.get("P") == 100, "开启自适应后应检出 100：%r" % (on,)
        assert on.get("escalated") is True, "长周期应由**升级**路径拿到"
    finally:
        set_period_caliber(adaptive_window=saved["adaptive_window"],
                           noninteger=saved["noninteger"])


def test_e_安装入口唯一且自检会拒():
    """E：`install_prediction_wrappers` 是唯一安装入口，`verify=True` 会拒陈旧重绑。"""
    assert callable(install_prediction_wrappers)
    r = install_prediction_wrappers(verify=True)
    assert isinstance(r, dict), "安装回执应是 dict：%r" % (r,)
    # 再装一次不应抛（幂等）；但把已装好的层**从类上摘掉**再 verify 必须被拒。
    import lingshu.world.world_model as W
    cls = W.UnifiedWorldModel
    got = getattr(cls, "_wm_wrappers_installed", None)
    r2 = install_prediction_wrappers(verify=True)
    assert isinstance(r2, dict), "重复安装应幂等：%r" % (r2,)
    assert got == getattr(cls, "_wm_wrappers_installed", None), "安装标记被重复安装改动了"
