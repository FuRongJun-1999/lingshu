# -*- coding: utf-8 -*-
"""表情/渲染守卫：set_expression 不得让眼睛高光重叠，也不得把腮红推成纯白。

覆盖 lingshu/world/silhouette3d.py 的 set_expression：
  A 高光几何（两个模板 × 六种表情）：只做「整体平移」——两两距离保持、同侧位移一致、
    非上抬表情零位移、上抬表情只向上位移、高光互不重合
  B 腮红颜色：任何表情都不是纯白；shy 确实「加深」；非 shy 保持基色；描边与填充一致
  C 像素级（真实渲染 900x1100）：纯白像素全部来自高光（腮红没有变白）；shy 与 neutral
    的白块数一致；白斑形状与原始一致
  D 幂等与可复原：重复调用、切换表情后颜色稳定

standalone（无 pytest）：python -X utf8 tests/test_expression_render_guard.py
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.world import silhouette3d as S  # noqa: E402

_PASS = []
_FAIL = []
EXPRS = ("neutral", "happy", "sad", "angry", "surprised", "shy")
MAKERS = ("fatfish_skinned", "fatfish_skeleton")
# 与 silhouette3d.set_expression 内的眼型映射保持一致
EYE_SHAPE = {"happy": "happy", "sad": "sad", "angry": "squint",
             "surprised": "surprise", "shy": "happy", "neutral": "normal"}
RAISED = ("happy", "squint")          # 这些眼型会把高光整体上抬
BLUSH_BASE = (255, 180, 190)
W, H = 900, 1100


def ok(cond, msg, extra=""):
    line = msg + (("   " + extra) if extra else "")
    (_PASS if cond else _FAIL).append(line)
    if not cond:
        print("  [FAIL] " + line)


def cen(obj):
    pts = obj.points if hasattr(obj, "points") else obj
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def part(s, name):
    return next((p for p in s.parts if p.name == name), None)


def dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


# ---------------------------------------------------------------- A 高光几何

def group_a():
    n0, f0 = len(_PASS), len(_FAIL)
    for maker in MAKERS:
        for expr in EXPRS:
            s = getattr(S, maker)()
            before = {}
            for side in ("l", "r"):
                before[side] = {p.name: cen(p) for p in s.parts
                                if p.name.startswith(f"highlight_{side}")}
            S.set_expression(s, expr)

            worst_d, worst_pair = 0.0, ""
            worst_s, worst_side = 0.0, ""
            min_dist = None
            for side in ("l", "r"):
                names = sorted(before[side])
                for i in range(len(names)):
                    for j in range(i + 1, len(names)):
                        d0 = dist(before[side][names[i]], before[side][names[j]])
                        d1 = dist(cen(part(s, names[i])), cen(part(s, names[j])))
                        if abs(d0 - d1) > worst_d:
                            worst_d = abs(d0 - d1)
                            worst_pair = "%s↔%s 前=%.6f 后=%.6f" % (names[i], names[j], d0, d1)
                        min_dist = d1 if min_dist is None else min(min_dist, d1)
                # 同侧每个高光的位移向量必须完全一致（红版会把它们搬到各自的名字位置）
                shifts = [(cen(part(s, nm))[0] - before[side][nm][0],
                           cen(part(s, nm))[1] - before[side][nm][1]) for nm in names]
                if shifts:
                    spread = max(dist(sh, shifts[0]) for sh in shifts)
                    if spread > worst_s:
                        worst_s = spread
                        worst_side = "眼%s 位移=%s" % (
                            side, [(round(sh[0], 9), round(sh[1], 9)) for sh in shifts])
                    dx, dy = shifts[0]
                    ok(abs(dx) <= 1e-6, f"{maker}/{expr}/眼{side} 高光零水平位移", "dx=%.9f" % dx)
                    if EYE_SHAPE[expr] in RAISED:
                        ok(dy < -1e-9, f"{maker}/{expr}/眼{side} 上抬表情高光整体上移",
                           "dy=%.9f" % dy)
                    else:
                        ok(abs(dy) <= 1e-9, f"{maker}/{expr}/眼{side} 非上抬表情高光零位移",
                           "dy=%.9f" % dy)

            ok(worst_d <= 1e-9, f"{maker}/{expr} 高光两两距离保持（整体平移）", worst_pair)
            ok(worst_s <= 1e-9, f"{maker}/{expr} 同侧高光共享同一位移向量", worst_side)
            if min_dist is not None:
                ok(min_dist > 1e-6, f"{maker}/{expr} 高光互不重合", "最小距离=%.6f" % min_dist)
    print("  [A] 高光几何断言 %d 条，失败 %d 条" % (len(_PASS) - n0, len(_FAIL) - f0))


# ---------------------------------------------------------------- B 腮红颜色

def group_b():
    n0, f0 = len(_PASS), len(_FAIL)
    for maker in MAKERS:
        for expr in EXPRS:
            s = getattr(S, maker)()
            raw = part(s, "blush_l").color
            S.set_expression(s, expr)
            c = part(s, "blush_l").color
            ok(c != (255, 255, 255), f"{maker}/{expr} 腮红不是纯白（不与高光同色）", str(c))
            if expr == "shy":
                ok(c[1] < raw[1] and c[2] < raw[2] and sum(c) < sum(raw),
                   f"{maker}/shy 腮红确实是「加深」", f"{raw} → {c}")
            else:
                ok(c == BLUSH_BASE, f"{maker}/{expr} 非 shy 保持腮红基色", str(c))
    print("  [B] 腮红断言 %d 条，失败 %d 条" % (len(_PASS) - n0, len(_FAIL) - f0))


# ---------------------------------------------------------------- C 渲染像素

def _mask(img, rgb):
    import numpy as np
    a = np.asarray(img.convert("RGB"))
    return (a[:, :, 0] == rgb[0]) & (a[:, :, 1] == rgb[1]) & (a[:, :, 2] == rgb[2])


def blobs(mask):
    import numpy as np
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if seen[y0, x0]:
            continue
        stack = [(y0, x0)]
        seen[y0, x0] = True
        n = 0
        xa = xb = x0
        ya = yb = y0
        while stack:
            y, x = stack.pop()
            n += 1
            xa = min(xa, x); xb = max(xb, x)
            ya = min(ya, y); yb = max(yb, y)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
        out.append((n, (xa, ya, xb, yb)))
    out.sort(key=lambda t: t[1][0])
    return out


def shot(maker, expr=None):
    """返回 (纯白像素掩码, 高光像素掩码, 白块列表)。

    高光掩码用「把高光临时改成品红再渲染一次」得到，因此不依赖投影内部实现。
    """
    s = getattr(S, maker)()
    if expr:
        S.set_expression(s, expr)
    m_white = _mask(s.render(screen_w=W, screen_h=H), (255, 255, 255))
    for p in s.parts:
        if p.name.startswith("highlight_"):
            p.color = (255, 0, 255)
            p.outline = (255, 0, 255)
    m_hi = _mask(s.render(screen_w=W, screen_h=H), (255, 0, 255))
    return m_white, m_hi, blobs(m_white)


def group_c():
    n0, f0 = len(_PASS), len(_FAIL)
    for maker in MAKERS:
        mw_or, mh_or, b_or = shot(maker)
        mw_nu, mh_nu, b_nu = shot(maker, "neutral")
        mw_sh, mh_sh, b_sh = shot(maker, "shy")

        for tag, mw, mh in (("neutral", mw_nu, mh_nu), ("shy", mw_sh, mh_sh)):
            stray = int((mw & ~mh).sum())
            ok(stray == 0, f"{maker} 渲染：{tag} 的纯白像素全部来自高光（腮红没有变白）",
               "越界白像素=%d" % stray)
            missing = int((mh & ~mw).sum())
            ok(missing == 0, f"{maker} 渲染：{tag} 的高光像素全部是纯白", "非白高光像素=%d" % missing)

        ok(len(b_sh) == len(b_nu), f"{maker} 渲染：shy 与 neutral 白色连通块数一致",
           "neutral=%d shy=%d" % (len(b_nu), len(b_sh)))
        ok(len(b_or) == len(b_nu), f"{maker} 渲染：原始与 neutral 白色块数一致",
           "orig=%d neutral=%d" % (len(b_or), len(b_nu)))

        def dims(bs):
            return sorted((bb[2] - bb[0] + 1, bb[3] - bb[1] + 1) for _, bb in bs)

        ok(dims(b_or) == dims(b_nu), f"{maker} 渲染：高光白斑尺寸与原始一致",
           "%s vs %s" % (dims(b_or), dims(b_nu)))
    print("  [C] 像素断言 %d 条，失败 %d 条" % (len(_PASS) - n0, len(_FAIL) - f0))


# ---------------------------------------------------------------- D 幂等/复原

def group_d():
    n0, f0 = len(_PASS), len(_FAIL)
    for maker in MAKERS:
        s = getattr(S, maker)()
        S.set_expression(s, "shy")
        first = part(s, "blush_l").color
        S.set_expression(s, "shy")
        ok(part(s, "blush_l").color == first, f"{maker}: shy 重复调用颜色稳定", str(first))
        S.set_expression(s, "neutral")
        ok(part(s, "blush_l").color == BLUSH_BASE, f"{maker}: shy → neutral 腮红复原基色",
           str(part(s, "blush_l").color))

        s2 = getattr(S, maker)()
        S.set_expression(s2, "surprised")
        S.set_expression(s2, "surprised")
        names = sorted(p.name for p in s2.parts if p.name.startswith("highlight_l"))
        if len(names) > 1:
            ds = [dist(cen(part(s2, names[i])), cen(part(s2, names[j])))
                  for i in range(len(names)) for j in range(i + 1, len(names))]
            ok(min(ds) > 1e-6, f"{maker}: surprised 连调两次高光仍互异", "最小距离=%.6f" % min(ds))
    print("  [D] 幂等断言 %d 条，失败 %d 条" % (len(_PASS) - n0, len(_FAIL) - f0))


def main():
    print("=" * 72)
    print("表情/渲染守卫：set_expression 高光几何 + 腮红颜色 + 渲染像素")
    print("=" * 72)
    group_a()
    group_b()
    group_c()
    group_d()
    total = len(_PASS) + len(_FAIL)
    print("\n===== SUMMARY %d/%d 通过 =====" % (len(_PASS), total))
    if _FAIL:
        print("失败项：")
        for m in _FAIL:
            print("  - " + m)
        return 1
    print("VERDICT=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
