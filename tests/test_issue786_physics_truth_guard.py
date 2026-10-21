# -*- coding: utf-8 -*-
"""test_issue786_physics_truth_guard —— physics 解析真值三项守卫（#786）

缺陷 / 修法（三项均经 #817 维护审核独立复跑判定成立）
----------------------------------------------------
    · 缺陷一（P1）`predict_at` circular 分支 center 约定错位：
      `fit_circular` 存的 `center` 是**世界序**（`_unhv` 按 up 轴展开），
      旧 `predict_at` 却按打包序读 `center[0]/center[1]/center[2]`
      （水平对 + 竖直）⇒ up∈{0,1} 时 fit→predict roundtrip 必错
      （#786/#817 实测 up=1 误差恰 = |cz−vert| = 2.5，y/z 互换；
      up=2 两约定重合故漏网；圆心恰在掩盖条件「竖直=某水平」时误差 0）。
      修法（#817 处置建议①，不改模型字典面）：`predict_at` 按轴取
      水平对 `center[idx[0]]/center[idx[1]]`、竖直 `center[up]`。
    · 缺陷二（P2）`Reflector.apply` 负初速轴速度方向恒错：
      旧码 `abs(vel[i])·sign` 抹掉调用方速度方向 ⇒ 未触壁的下行轴
      输出 +2.7（#817 读数 -2.7→+2.7）；位置面完全正确。
      修法：`vel[i]·sign`（一行）。
    · 缺陷三（P2）`projectile_truth.apex_height` 宣称「顶点高度」实为
      「相对起点位移」：旧返回值恒等于 v0_up²/2g，与姊妹函数
      `projectile_state` 的世界坐标口径差 p0_up（#786 算例差恰 10.0）。
      修法（#786 §8 探针 R2 的判据 = 数值世界最高点 p0_up+v0²/2g）：
      公式补 `+ p_up`，与 `projectile_state` / `environment_truth` 同口径。

运行命令
--------
    cd <仓根>
    python -X utf8 -m pytest tests/test_issue786_physics_truth_guard.py -q

断言组（抽掉修复 ⇒ 哪组红）
--------------------------------
    G1 circular roundtrip   up∈{0,1,2} 的 fit→predict_at(t_ref) 复原首样本；
                            生产链 classify→predict_at 复现真位置；
                            fit center 仍为世界序（防按打包序改 fit 返回面）。
                            → 回退 predict_at 打包序读法 ⇒ G1 全红（up=2 除外的
                              参数组合红；center 面断言在修法②下红）
    G2 Reflector 负初速     未触壁下行轴速度原号折回；正初速轴零回归；
                            逐步反射仿真对照（位置 + 速度方向）；
                            predict_trajectory 反射通路同口径。
                            → 回退 abs(...)·sign ⇒ G2 红
    G3 apex 世界高度        apex_height == p0[up]+v0_up²/2g == projectile_state
                            顶点世界坐标 == 数值扫描最高点；其余字段零回归。
                            → 回退去掉 +p_up ⇒ G3 红
"""
from __future__ import annotations

import math

import pytest

from lingshu.world import physics as P

G = 9.80665

# —— 缺陷一参数（与 #786 §2.3 / #817 复核读数一致）：r=2、ω=1.5、圆在 x-z 平面，
#    圆心世界序 (2, 0.5, 3) —— 竖直 0.5 ≠ 两水平 2/3，避开掩盖条件 ——
_TS = [round(0.05 * i, 2) for i in range(25)]  # 0.00..1.20


def _circ_samples(up):
    """up 轴参数化的圆周 + 零漂移合成轨迹（世界序圆心 (2, 0.5, 3)，r=2，ω=1.5）。

    up=1 ⇒ 圆在 x-z 平面（issue 主复现）；up=0 ⇒ 圆在 y-z 平面；
    up=2 ⇒ 圆在 x-y 平面（两约定重合的反面对照）。
    """
    def pos(t):
        out = [None, None, None]
        idx = [i for i in range(3) if i != up]
        out[idx[0]] = 2.0 + 2.0 * math.cos(1.5 * t)
        out[idx[1]] = 3.0 + 2.0 * math.sin(1.5 * t)
        out[up] = 0.5
        return tuple(out)
    return [(t, pos(t)) for t in _TS]


@pytest.mark.parametrize("up", [0, 1, 2])
def test_G1_circular_roundtrip_first_sample(up):
    """fit→`predict_at(t_ref)` 必须复原**自身首样本**（世界序 center 的直接后果）。

    #786 修前实测：up=1 预测 (4.0, 3.0, 0.5) vs 首样本 (4.0, 0.5, 3.0)
    （y/z 互换，maxdiff = 2.500000）；up=0 同错（maxdiff = 3.5）；
    up=2 两约定重合恒对（1.73e-14）—— 本参数避开「竖直=某水平」的掩盖条件。
    """
    smp = _circ_samples(up)
    m = P.fit_model("circular", smp, gravity=G, up=up)
    first = smp[0][1]
    p, v = P.predict_at(m, float(m["t_ref"]))
    assert all(abs(p[i] - first[i]) < 1e-9 for i in range(3)), (
        "circular fit→predict_at(t_ref) 未复原首样本（#786 约定错位）："
        "up=%d 预测 %r vs 首样本 %r" % (up, tuple(round(x, 6) for x in p), first))
    # 全时段 roundtrip（不只 t_ref）：位置 + 速度（切向 r·ω 旋转 90°，v_up=0 ⇒ up 轴速度 0）
    idx = [i for i in range(3) if i != up]
    for t, q in smp:
        th = 1.5 * t
        v_true = [0.0, 0.0, 0.0]
        v_true[idx[0]] = -2.0 * 1.5 * math.sin(th)
        v_true[idx[1]] = 2.0 * 1.5 * math.cos(th)
        p_t, v_t = P.predict_at(m, t)
        assert all(abs(p_t[i] - q[i]) < 1e-9 for i in range(3)), (
            "circular predict_at(%r) 偏离样本（#786）：预测 %r vs %r" % (t, p_t, q))
        assert all(abs(v_t[i] - v_true[i]) < 1e-9 for i in range(3)), (
            "circular predict_at(%r) 速度偏离切向真值（#786）：%r vs %r" % (t, v_t, v_true))


def test_G1_circular_center_is_world_order():
    """fit 的返回面不动：`center` 仍为**世界序**（修法① 的边界锚点）。

    若有人按修法②把 `fit_circular` 改存打包序（会改 `wmfactory.physics/1`
    已持久化模型的字段含义），此断言先红。Kåsa 拟合本身无瑕（#786 §2.3）。
    """
    m = P.fit_model("circular", _circ_samples(1), gravity=G, up=1)
    assert all(abs(m["center"][i] - (2.0, 0.5, 3.0)[i]) < 1e-9 for i in range(3)), (
        "fit_circular 的 center 必须保持世界序（#786 修法①边界）：得到 %r" % (m["center"],))
    assert abs(m["radius"] - 2.0) < 1e-9 and abs(m["omega"] - 1.5) < 1e-9
    assert m["residual_rms"] < 1e-9


def test_G1_production_chain_classify_predict():
    """生产链 `classify→predict_at`（physics_predict.py 的调用形态）复现真位置。

    #786 修前：判型对（residual_rms 是 fit 自身世界序残差）而预测错 2.5 ——
    缺陷被完整传导到下游；这是 P1 的生产影响读数。
    """
    smp = _circ_samples(1)
    got = P.classify(smp, up=1)
    assert got["kind"] == "circular", got["reason"]
    t_probe = 0.4
    idx = [0, 2]
    true_p = (2.0 + 2.0 * math.cos(1.5 * t_probe), 0.5, 3.0 + 2.0 * math.sin(1.5 * t_probe))
    p, _ = P.predict_at(got["model"], t_probe)
    dp = max(abs(p[i] - true_p[i]) for i in range(3))
    assert dp < 1e-6, "classify→predict_at(%r) 位置偏差 %.6f（#786 修前 = 2.500000）" % (t_probe, dp)


# —— 缺陷二参数（与 #786 §3.2 / #817 读数一致）——
_BOX_LO, _BOX_HI = (0.0, 0.0, 0.0), (10.0, 8.0, 6.0)
_V0_NEG = (3.3, -2.7, 1.9)          # 轴1 负初速（下行）
_P0_BOX = (1.0, 1.0, 1.0)


def test_G2_reflector_negative_v0_velocity_sign():
    """未触壁、正在下行的轴：速度必须**原号**折回（位置原样）。

    #786 修前实测（逐字）：apply(0.1s, pos=(1.0,0.73,1.0), vel=(0.0,-2.7,0.0))
    ⇒ 位置原样、速度 +2.7（反号）。正初速轴修前修后一致（反面对照）。
    """
    rf = P.Reflector(_BOX_LO, _BOX_HI, up=1)
    pos, vel = rf.apply(0.0, 0.1, (1.0, 0.73, 1.0), (0.0, -2.7, 0.0))
    assert all(abs(pos[i] - (1.0, 0.73, 1.0)[i]) < 1e-12 for i in range(3))
    assert all(abs(vel[i] - (0.0, -2.7, 0.0)[i]) < 1e-12 for i in range(3)), (
        "未触壁的下行轴速度被反号（#786）：得到 %r，期望 (0.0, -2.7, 0.0)" % (vel,))
    # 正初速对照（缺陷严格局限于负初速轴）
    _, vel_pos = rf.apply(0.0, 0.1, (1.0, 0.73, 1.0), (0.0, 2.7, 0.0))
    assert all(abs(vel_pos[i] - (0.0, 2.7, 0.0)[i]) < 1e-12 for i in range(3))


def test_G2_reflector_stepwise_simulation_crosscheck():
    """`Reflector.apply` vs 逐步反射仿真（位置 + 远壁速度方向，#786 §3.2 的轻量版）。

    #786 修前：位置 8/8 全对（max|dp| 2.55e-11）而轴1（v0=-2.7）速度方向 0/8 一致。
    """
    rf = P.Reflector(_BOX_LO, _BOX_HI, up=1)
    dts = 1e-3
    t_end = 2.4
    n = int(round(t_end / dts))
    pos, vel = list(_P0_BOX), list(_V0_NEG)
    snaps = {}
    for s in range(n + 1):
        if s % 100 == 0:
            snaps[round(s * dts, 6)] = (tuple(pos), tuple(vel))
        for i in range(3):
            pos[i] += vel[i] * dts
            if pos[i] > _BOX_HI[i]:
                pos[i] = _BOX_HI[i] - (pos[i] - _BOX_HI[i]); vel[i] = -vel[i]
            elif pos[i] < _BOX_LO[i]:
                pos[i] = _BOX_LO[i] + (_BOX_LO[i] - pos[i]); vel[i] = -vel[i]
    bad_dir, max_dp = [], 0.0
    for t, (p_sim, v_sim) in sorted(snaps.items()):
        p_free = tuple(_P0_BOX[i] + _V0_NEG[i] * t for i in range(3))
        p_an, v_an = rf.apply(0.0, t, p_free, _V0_NEG)
        max_dp = max(max_dp, max(abs(p_an[i] - p_sim[i]) for i in range(3)))
        for i in range(3):
            # 近壁转向点（±3 步内）有步长歧义，不计符号
            if min(abs(p_an[i] - _BOX_LO[i]), abs(_BOX_HI[i] - p_an[i])) <= 3 * dts * 3.3:
                continue
            if v_an[i] * v_sim[i] <= 0:
                bad_dir.append((t, i))
    assert max_dp < 8 * dts * 3.3, "位置面漂移 max|dp|=%.2e（修前量级 1e-11，不应变大）" % max_dp
    assert not bad_dir, "远离壁面处速度方向与逐步仿真不一致（#786）：%r" % (bad_dir[:6],)


def test_G2_predict_trajectory_reflector_path():
    """`predict_trajectory(model, times, reflector)` 反射通路同口径（生产面）。"""
    rf = P.Reflector(_BOX_LO, _BOX_HI, up=1)
    mdl = {"kind": "uniform", "p0": _P0_BOX, "v0": _V0_NEG, "t_ref": 0.0}
    rows = P.predict_trajectory(mdl, [0.0, 0.3, 0.6, 0.9, 1.2], reflector=rf)
    for t, p, v in rows:
        for i in range(3):
            assert _BOX_LO[i] - 1e-9 <= p[i] <= _BOX_HI[i] + 1e-9, "反射通路穿透（#786 邻域）"
            # 未折返（自由位置仍在盒内）⇒ 速度分量必须与 v0 同号
            free = _P0_BOX[i] + _V0_NEG[i] * t
            if _BOX_LO[i] <= free <= _BOX_HI[i]:
                assert v[i] * _V0_NEG[i] > 0 or abs(v[i]) < 1e-12, (
                    "t=%r 轴%d 未折返而速度变号（#786）：v=%r" % (t, i, v))


# —— 缺陷三参数（与 #786 §4.1 / #817 读数一致）：p0=(1,10,2)、v0=(3,5,-1) ——
_APX_P0 = (1.0, 10.0, 2.0)


@pytest.mark.parametrize("up,v0", [(1, (3.0, 5.0, -1.0)), (2, (3.0, 5.0, 4.0))])
def test_G3_apex_height_world_caliber(up, v0):
    """apex_height = 世界坐标顶点高度 = p0[up] + v0_up²/2g（与 projectile_state 同口径）。

    #786 修前实测（up=1, v0=(3,5,-1)）：apex_height = 1.274645266（恒等于 v0²/2g
    位移口径），而 projectile_state 同一时刻给 11.274645266 —— 差恰 p0_up = 10.0。
    up=2（Z-up）用正竖直初速复核同型口径。
    """
    p0 = _APX_P0
    tr = P.projectile_truth(p0, v0, gravity=G, up=up, ground=None)
    m = {"p0": p0, "v0": v0, "up": up, "t0": 0.0}
    world_apex = P.projectile_state(m, tr["t_apex"])[0][up]
    expect = p0[up] + v0[up] ** 2 / (2 * G)
    assert abs(tr["apex_height"] - expect) < 1e-12, (
        "apex_height 须为世界高度 p0[up]+v0²/2g（#786）：得到 %.12f，期望 %.12f"
        % (tr["apex_height"], expect))
    assert abs(tr["apex_height"] - world_apex) < 1e-12, (
        "apex_height 与 projectile_state 顶点世界坐标不符（#786）：%.12f vs %.12f"
        % (tr["apex_height"], world_apex))
    # 数值扫描最高点（粗扫 40 万点，分辨率远小于断言容差）
    best = max(p0[up] + v0[up] * t - 0.5 * G * t * t
               for t in (i * 2.0 / 400000 for i in range(400001)))
    assert abs(tr["apex_height"] - best) < 1e-4


def test_G3_projectile_truth_other_fields_no_regression():
    """防过修：同函数其余字段（t_land/range/v_land_up）与独立复算逐位一致（#786 §4.1）。"""
    p0, v0 = _APX_P0, (3.0, 5.0, -1.0)
    tr = P.projectile_truth(p0, v0, gravity=G, up=1, ground=0.0)
    h = p0[1] - 0.0
    disc = v0[1] ** 2 + 2 * G * h
    t_land = (v0[1] + math.sqrt(disc)) / G
    assert abs(tr["t_land"] - t_land) < 1e-12
    assert abs(tr["range"] - math.hypot(v0[0], v0[2]) * t_land) < 1e-12
    assert abs(tr["v_land_up"] - (v0[1] - G * t_land)) < 1e-12
    assert abs(tr["t_apex"] - v0[1] / G) < 1e-12
    # v_up<=0（无未来顶点）⇒ 两字段仍为 None（不虚构顶点）
    tr_dn = P.projectile_truth(p0, (3.0, -5.0, -1.0), gravity=G, up=1)
    assert tr_dn["t_apex"] is None and tr_dn["apex_height"] is None


def test_G3_environment_truth_projectile_apex_world_caliber():
    """`environment_truth('projectile')` 透出面同口径（公开 API 使用者拿世界高度）。"""
    med = P.make_environment()
    v0 = (3.0, 5.0, -1.0)
    tr = P.environment_truth(med, "projectile", p0=_APX_P0, v0=v0)
    assert abs(tr["apex_height"] - (_APX_P0[1] + v0[1] ** 2 / (2 * P.DEFAULT_G))) < 1e-9
