# Copyright 2026 灵枢 (Lingshu) · MIT
"""守卫：Skeleton3D.world_pos 的父链旋转合成顺序（issue 见 PR 说明）。

三条独立判据（全部只用模块自身的语义，不引入外部约定）：

  A. 自洽性：同一个父链上，任意关节偏移被「自身旋转 + 父链旋转」作用后的累积算子
     必须是同一个关于链的函数。深度 1（父=根）时 A = R_parent · R_child；深度 ≥2 时
     也必须如此（父在左、根在右），否则同一链条对父偏移与子偏移用了两种顺序。

  B. 刚体不变量：根关节旋转时，父骨与子骨的夹角（局部弯曲角）与各关节相对间距必须不变。
     顺序被反转时该夹角会随根关节角度摆动（实测 58.2°–121.8°，极差 63.5°）。

  C. 参照实现一致：按 A_parent·R_child 递归独立复算的世界坐标，必须与 world_pos 逐位一致
     （含真实模板 fatfish_bone_skeleton 的混合轴姿态，47 个关节 max 偏差 < 1e-9）。

standalone（不依赖 pytest）：python -X utf8 tests/test_skeleton_fk_order_guard.py
退出码 0 = 全过；1 = 有失败。
"""
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.world.skeleton3d import Skeleton3D, fatfish_bone_skeleton  # noqa: E402

_PASS = []
_FAIL = []


def ok(cond: bool, msg: str, extra: str = "") -> bool:
    ( _PASS if cond else _FAIL).append(msg)
    if not cond:
        print("[FAIL] %s%s" % (msg, ("   " + extra) if extra else ""))
    return bool(cond)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def mat(joint):
    return joint._rot_matrix()


def mm(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def mv(m, v):
    return tuple(sum(m[i][k] * v[k] for k in range(3)) for i in range(3))


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def norm(v):
    return math.sqrt(sum(c * c for c in v))


def angle(u, v):
    nu, nv = norm(u), norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return float("nan")
    c = sum(a * b for a, b in zip(u, v)) / (nu * nv)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def ref_world(sk):
    """独立参照：A_root = R_root；A_child = A_parent · R_child；
    world(child) = world(parent) + A_parent · (R_child · v_child)。"""
    A, out = {}, {}

    def acc(name):
        if name not in A:
            j = sk.joints[name]
            A[name] = mat(j) if j.parent is None else mm(acc(j.parent), mat(j))
        return A[name]

    def pos(name):
        if name not in out:
            j = sk.joints[name]
            if j.parent is None:
                out[name] = tuple(sk.center[i] + mv(mat(j), j.pos)[i] for i in range(3))
            else:
                p = pos(j.parent)
                local = mv(acc(j.parent), mv(mat(j), j.pos))
                out[name] = tuple(p[i] + local[i] for i in range(3))
        return out[name]

    for n in list(sk.joints):
        pos(n)
    return out


def chain_skel():
    """root → mid → leaf，root/mid 旋转轴不同（不交换）。"""
    sk = Skeleton3D(center=(0.0, 0.0, 5.0))
    sk.add_joint("root", (0.0, 0.0, 0.0))
    sk.add_joint("mid", (0.1, 0.0, 0.0), parent="root")
    sk.add_joint("leaf", (0.0, -0.3, 0.0), parent="mid")
    return sk


# ---------------------------------------------------------------------------
# A. 自洽性：深度 1 与深度 2 必须用同一种合成顺序
# ---------------------------------------------------------------------------
def group_a():
    sk = chain_skel()
    sk.joints["root"].rot = (0.0, 0.0, 1.0)
    sk.joints["mid"].rot = (0.6, 0.0, 0.0)

    # A1：深度 1 —— world(mid) 的累积算子是 R_root · R_mid
    want_mid = tuple(sk.center[i] + mv(mm(mat(sk.joints["root"]), mat(sk.joints["mid"])),
                                       sk.joints["mid"].pos)[i] for i in range(3))
    got_mid = sk.world_pos("mid")
    d1 = norm(sub(got_mid, want_mid))
    ok(d1 < 1e-9, "A1 深度 1 的累积旋转 = R_root·R_mid（父在左）", "偏差=%.12f" % d1)

    # A2：深度 2 —— 若顺序自洽，world(leaf) 必须等于参照
    ref = ref_world(sk)
    got_leaf = sk.world_pos("leaf")
    d2 = norm(sub(got_leaf, ref["leaf"]))
    ok(d2 < 1e-9, "A2 深度 2 的世界坐标与参照 A_parent·R_child 一致", "偏差=%.12f" % d2)

    # A3：两种顺序在同一链上不可互换（反证：差不为 0，故必须选对）
    prod_std = mm(mat(sk.joints["root"]), mat(sk.joints["mid"]))
    prod_rev = mm(mat(sk.joints["mid"]), mat(sk.joints["root"]))
    diff = max(abs(prod_std[i][j] - prod_rev[i][j]) for i in range(3) for j in range(3))
    ok(diff > 1e-3, "A3 该姿态下两种顺序确实不等价（夹具有效）", "max 元素差=%.6f" % diff)


# ---------------------------------------------------------------------------
# B. 刚体不变量
# ---------------------------------------------------------------------------
def group_b():
    sk = chain_skel()
    sk.joints["mid"].rot = (0.6, 0.0, 0.0)
    sk.joints["leaf"].rot = (1.2, 0.0, 0.0)
    angs, spans = [], []
    for i in range(13):
        a = i * math.pi / 6
        sk.joints["root"].rot = (0.0, 0.0, a)
        w = {n: sk.world_pos(n) for n in ("root", "mid", "leaf")}
        angs.append(angle(sub(w["mid"], w["root"]), sub(w["leaf"], w["mid"])))
        spans.append(norm(sub(w["leaf"], w["mid"])))
    ok(max(angs) - min(angs) < 1e-6,
       "B1 根关节扫 0..2π 时父骨/子骨夹角恒定", "极差=%.6f 度" % (max(angs) - min(angs)))
    ok(abs(angs[0] - 90.0) < 1e-6, "B2 夹角等于姿态设定值 90°", "实测=%.6f 度" % angs[0])
    ok(max(spans) - min(spans) < 1e-9,
       "B3 根关节旋转不改变子骨长度", "极差=%.3e m" % (max(spans) - min(spans)))


def group_b2():
    """真实模板：hip(Rz) 旋转时整条手臂刚性（混合轴父链）。"""
    sk = fatfish_bone_skeleton(center=(0.0, 0.85, 5.0), height=1.55)
    sk.joints["clavicle"].rot = (0.35, 0.0, 0.0)
    sk.joints["shoulder_l"].rot = (0.0, 0.0, 0.6)
    sk.joints["elbow_l"].rot = (1.2, 0.0, 0.0)
    angs = []
    for i in range(13):
        sk.joints["hip"].rot = (0.0, 0.0, i * math.pi / 6)
        w = {n: sk.world_pos(n) for n in ("clavicle", "shoulder_l", "elbow_l")}
        angs.append(angle(sub(w["shoulder_l"], w["clavicle"]), sub(w["elbow_l"], w["shoulder_l"])))
    ok(max(angs) - min(angs) < 1e-6,
       "B4 fatfish 骨架：hip 旋转时上臂/前臂夹角恒定", "极差=%.6f 度" % (max(angs) - min(angs)))


# ---------------------------------------------------------------------------
# C. 与参照实现逐位一致（混合轴真实姿态）
# ---------------------------------------------------------------------------
def group_c():
    sk = fatfish_bone_skeleton(center=(0.0, 0.85, 5.0), height=1.55)
    pose = {
        "hip": (0.0, 0.0, 0.8), "waist": (0.0, 0.5, 0.0), "chest": (0.3, 0.0, 0.0),
        "clavicle": (0.0, 0.0, 0.25), "shoulder_l": (0.0, 0.0, 0.6), "elbow_l": (1.2, 0.0, 0.0),
        "shoulder_r": (0.0, 0.0, -0.4), "elbow_r": (0.9, 0.0, 0.0), "neck": (0.2, 0.0, 0.0),
        "thigh_l": (0.5, 0.0, 0.0), "knee_l": (-0.7, 0.0, 0.0),
        "tail0": (0.2, 0.0, 0.0), "tail1": (0.3, 0.0, 0.0), "tail2": (0.3, 0.0, 0.0),
    }
    for k, v in pose.items():
        sk.joints[k].rot = v
    ref = ref_world(sk)
    worst, name = 0.0, ""
    for n in sk.joints:
        d = norm(sub(sk.world_pos(n), ref[n]))
        if d > worst:
            worst, name = d, n
    ok(worst < 1e-9, "C1 47 关节混合轴姿态与参照一致", "max=%.3e m @ %s" % (worst, name))

    # C2：子骨骼长度与姿态无关（刚性）
    sk.joints["hip"].rot = (0.0, 0.0, 0.0)
    base = norm(sub(sk.world_pos("elbow_l"), sk.world_pos("shoulder_l")))
    sk.joints["hip"].rot = (0.7, 0.3, 0.2)
    moved = norm(sub(sk.world_pos("elbow_l"), sk.world_pos("shoulder_l")))
    ok(abs(base - moved) < 1e-9, "C2 前臂长度与祖先姿态无关", "%.9f vs %.9f" % (base, moved))


# ---------------------------------------------------------------------------
# D. 已发布调用点的姿态不受影响（同轴链）
# ---------------------------------------------------------------------------
def group_d():
    # scene_model.build()：skeleton 呈现时 happy → shoulder_r/elbow_r 绕 z；tired → neck 绕 x
    sk = fatfish_bone_skeleton(center=(0.0, 0.85, 5.0), height=1.55)
    sk.joints["shoulder_r"].rot = (0.0, 0.0, 1.2)
    sk.joints["elbow_r"].rot = (0.0, 0.0, -1.0)
    ref = ref_world(sk)
    worst = max(norm(sub(sk.world_pos(n), ref[n])) for n in sk.joints)
    ok(worst < 1e-9, "D1 happy 挥手姿态（同轴链）与参照一致", "max=%.3e m" % worst)

    sk2 = fatfish_bone_skeleton(center=(0.0, 0.85, 5.0), height=1.55)
    sk2.joints["neck"].rot = (0.5, 0.0, 0.0)
    ref2 = ref_world(sk2)
    worst2 = max(norm(sub(sk2.world_pos(n), ref2[n])) for n in sk2.joints)
    ok(worst2 < 1e-9, "D2 tired 低头姿态（单关节）与参照一致", "max=%.3e m" % worst2)

    # D3：内置走路 demo（thigh/shin 同轴 rx + shoulder rz）刚性
    sk3 = fatfish_bone_skeleton(center=(0.0, 0.85, 5.0), height=1.55)
    sk3.joints["thigh_l"].rot = (0.4, 0.0, 0.0)
    sk3.joints["knee_l"].rot = (-0.3, 0.0, 0.0)
    ref3 = ref_world(sk3)
    worst3 = max(norm(sub(sk3.world_pos(n), ref3[n])) for n in sk3.joints)
    ok(worst3 < 1e-9, "D3 走路姿态（同轴链）与参照一致", "max=%.3e m" % worst3)


def main() -> int:
    print("=" * 70)
    print("Skeleton3D 父链旋转合成顺序守卫（issue 见 PR 说明）")
    print("=" * 70)
    print("repo = %s" % REPO)
    group_a()
    group_b()
    group_b2()
    group_c()
    group_d()
    total = len(_PASS) + len(_FAIL)
    print("=" * 70)
    print("===== SUMMARY %d/%d 通过 =====" % (len(_PASS), total))
    if _FAIL:
        print("失败 %d 条：" % len(_FAIL))
        for m in _FAIL:
            print("   - %s" % m)
        return 1
    print("VERDICT=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
