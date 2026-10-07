# -*- coding: utf-8 -*-
"""test_hex_gen · P1 白箱文生图最小闭环验收
（P1① 条件兑现率≥95% 构造性口径 + P1③ 变换不变性≥95% + zone 映射表交付物）
脚本/pytest 双模式。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from lingshu.nn.hex_gen import (ATTR_CALIB_96, PATTERN_WORDS, SIZE_WORDS,
                          ZONE_TRANSFORMS, compile_description,
                          generate_from_text, rc_zone, render_relations,
                          shift_relations, transform_relations,
                          verify_constructive, verify_readback, zone_rc)
from lingshu.nn.hex_text import POS, SHAPES, COLORS
from lingshu.nn.hex_composite import PATTERNS

_results = [0, 0]


def check(name, cond, detail=""):
    if cond:
        _results[0] += 1
        print(f"  PASS {name}")
    else:
        _results[1] += 1
        print(f"  FAIL {name} {detail}")


def run_checks():
    # ---------- 编译器 ----------
    c = compile_description("左上有红色圆形,右下有绿色三角")
    check("两子句编译", len(c["parts"]) == 2
          and c["parts"][0]["shape"] == "circle"
          and c["parts"][0]["color"] == "red"
          and c["parts"][0]["zone"] == "r0"
          and c["parts"][1]["shape"] == "triangle"
          and c["parts"][1]["color"] == "green"
          and c["parts"][1]["zone"] == "r8",
          str(c["parts"]))
    check("三元组齐全时无 shape/color/zone 缺省", not any(
        d["field"] in ("shape", "color", "zone")
        for d in c["defaults_applied"]), str(c["defaults_applied"]))
    c2 = compile_description("上面有圆")
    # R43：单字方位词「上」→r1（此前未识别默认落r4——期望随行为更新）
    check("缺省补全且落账", c2["parts"][0]["color"] == "red"
          and c2["parts"][0]["zone"] == "r1"
          and any(d["field"] == "color" for d in c2["defaults_applied"]),
          str(c2))

    # ---------- P1① 条件兑现率（构造性验证，主验收）----------
    rng = np.random.default_rng(11)
    for size in (48, 96):
        n_ok = n_all = 0
        for t in range(40):
            k = int(rng.integers(1, 4))          # 1-3 部件
            parts = [{"shape": SHAPES[rng.integers(3)],
                      "color": COLORS[rng.integers(3)],
                      "zone": POS[rng.integers(9)]} for _ in range(k)]
            img, log = render_relations(parts, size=size, seed=t)
            check_img = img.shape == (size, size, 3) and img.mean() > 0
            v = verify_constructive(parts, log)
            n_ok += v["matched"]
            n_all += v["total"]
            if not check_img:
                _results[1] += 1
                print(f"  FAIL 渲染形状 size={size} t={t}")
        rate = n_ok / n_all
        check(f"P1① 兑现率@{size}px ≥95%（构造性）", rate >= 0.95,
              f"rate={rate:.3f} ({n_ok}/{n_all})")
        print(f"    [数字] {size}px 兑现率 = {rate:.1%} ({n_ok}/{n_all})")

    # ---------- P1③ zone 映射表交付物 ----------
    # 映射表数学自检：rot90cw∘rot90ccw=恒等；rot180∘rot180=恒等；fliph∘fliph=恒等
    ident = all(ZONE_TRANSFORMS["rot90cw"](*ZONE_TRANSFORMS["rot90ccw"](r, c)) == (r, c)
                and ZONE_TRANSFORMS["rot180"](*ZONE_TRANSFORMS["rot180"](r, c)) == (r, c)
                and ZONE_TRANSFORMS["fliph"](*ZONE_TRANSFORMS["fliph"](r, c)) == (r, c)
                for r in range(3) for c in range(3))
    check("zone 映射表群性质（逆变换复合=恒等）", ident)
    # 角点样例（映射表逐格显式）
    check("rot90cw 角点映射", ZONE_TRANSFORMS["rot90cw"](0, 0) == (0, 2)
          and ZONE_TRANSFORMS["rot90cw"](2, 0) == (0, 0))
    zone_words = {rc_zone(*ZONE_TRANSFORMS[op](r, c))
                  for op in ZONE_TRANSFORMS
                  for r in range(3) for c in range(3)}
    check("映射封闭于 9 宫格词汇", zone_words <= set(POS), str(zone_words))

    # ---------- P1③ 变换不变性（关系层变换 + 构造性回读）----------
    for op in ZONE_TRANSFORMS:
        n_ok = n_all = 0
        for t in range(8):
            k = int(rng.integers(1, 4))
            parts = [{"shape": SHAPES[rng.integers(3)],
                      "color": COLORS[rng.integers(3)],
                      "zone": POS[rng.integers(9)]} for _ in range(k)]
            tp = transform_relations(parts, op)
            img, log = render_relations(tp, size=48, seed=t)
            v = verify_constructive(tp, log)
            n_ok += v["matched"]
            n_all += v["total"]
        rate = n_ok / n_all
        check(f"P1③ 变换不变性 {op} ≥95%", rate >= 0.95,
              f"rate={rate:.3f}")
        print(f"    [数字] {op} 不变率 = {rate:.1%}")

    # ---------- 端到端 ----------
    g = generate_from_text("左上有红色圆形,右下有绿色三角", size=96, seed=3)
    check("端到端兑现", g["constructive"]["rate"] == 1.0,
          str(g["constructive"]))
    g2 = generate_from_text("左上有红色圆形", size=96, seed=3,
                            transform="rot90cw")
    check("端到端变换（r0→rot90cw→r2）",
          g2["parts"][0]["zone"] == "r2"
          and g2["constructive"]["rate"] == 1.0, str(g2["parts"]))
    check("确定性（同 seed 同输出）",
          np.array_equal(g["image"], generate_from_text(
              "左上有红色圆形,右下有绿色三角", size=96, seed=3)["image"]))
    try:
        transform_relations([{"shape": "circle", "color": "red", "zone": "r4"}],
                            "zoom")
        check("未知变换报错", False)
    except ValueError:
        check("未知变换报错", True)

    # ---------- R2 · P1② 属性词解析 ----------
    ca = compile_description("左上有条纹圆,右下有大三角,中间有实心小圆")
    p0, p1, p2 = ca["parts"]
    check("花纹词解析（条纹→striped/实心→solid）",
          p0["pattern"] == "striped" and p2["pattern"] == "solid",
          str(ca["parts"]))
    check("尺寸词解析（大→large/小→small）",
          p1["size"] == "large" and p2["size"] == "small", str(ca["parts"]))
    check("词表完整（3 花纹×2 尺寸）",
          len(PATTERN_WORDS) == 3 and len(SIZE_WORDS) == 2)
    check("无属性词时缺省落账",
          compile_description("左上有红色圆形")["parts"][0]["pattern"] == "solid"
          and any(d["field"] == "pattern" for d in
                  compile_description("左上有红色圆形")["defaults_applied"]))

    # ---------- R2 · 带属性兑现率（构造性，主验收）----------
    rng2 = np.random.default_rng(23)
    for size in (48, 96):
        n_ok = n_all = 0
        for t in range(30):
            k = int(rng2.integers(1, 4))
            parts = [{"shape": SHAPES[rng2.integers(3)],
                      "color": COLORS[rng2.integers(3)],
                      "zone": POS[rng2.integers(9)],
                      "pattern": PATTERNS[rng2.integers(3)],
                      "size": ["small", "medium", "large"][rng2.integers(3)]}
                     for _ in range(k)]
            _, log = render_relations(parts, size=size, seed=t)
            v = verify_constructive(parts, log)
            n_ok += v["matched"]
            n_all += v["total"]
        rate = n_ok / n_all
        check(f"R2 带属性兑现率@{size}px ≥95%", rate >= 0.95,
              f"rate={rate:.3f}")
        print(f"    [数字] {size}px 带属性兑现率 = {rate:.1%} ({n_ok}/{n_all})")

    # ---------- R2 · 像素回读次级指标（96px，不卡线如实记录）----------
    rng3 = np.random.default_rng(31)
    pat_stat = {p: [0, 0] for p in PATTERNS}     # pattern -> [ok, n]
    size_ok = size_n = 0
    for t in range(30):
        k = int(rng3.integers(1, 4))
        parts = [{"shape": SHAPES[rng3.integers(3)],
                  "color": COLORS[rng3.integers(3)],
                  "zone": POS[rng3.integers(9)],
                  "pattern": PATTERNS[rng3.integers(3)],
                  "size": ["small", "medium", "large"][rng3.integers(3)]}
                 for _ in range(k)]
        img, log = render_relations(parts, size=96, seed=100 + t)
        for i, p in enumerate(parts):
            e = next(x for x in log if x["part"] == i)
            from lingshu.nn.hex_composite import extract_attributes
            a = extract_attributes(img, e["zone_landed"], p["color"],
                                   ATTR_CALIB_96, shape_hint=p["shape"])
            pat_stat[p["pattern"]][1] += 1
            pat_stat[p["pattern"]][0] += int(a.get("pattern") == p["pattern"])
            if p["shape"] == "circle" and p["size"] in ("small", "large"):
                size_n += 1
                size_ok += int(a.get("size") == p["size"])
    pat_all_ok = sum(v[0] for v in pat_stat.values())
    pat_all_n = sum(v[1] for v in pat_stat.values())
    per_pat = {k: f"{v[0]}/{v[1]}" for k, v in pat_stat.items()}
    print(f"    [数字] 96px 回读：花纹总 {pat_all_ok}/{pat_all_n} = "
          f"{pat_all_ok/pat_all_n:.1%}（分项 {per_pat}）；"
          f"尺寸 {size_ok}/{size_n} = {size_ok/max(1,size_n):.1%}")
    check("R2 回读花纹 ≥80%（次级，如实记录）",
          pat_all_ok / pat_all_n >= 0.80, f"{pat_all_ok}/{pat_all_n} {per_pat}")
    check("R2 回读尺寸（圆） ≥80%", size_ok / max(1, size_n) >= 0.80,
          f"{size_ok}/{size_n}")
    # 端到端含属性词 + 回读
    g3 = generate_from_text("左上有大的条纹圆,右下有小的点纹三角", size=96, seed=5)
    check("带属性端到端兑现",
          g3["constructive"]["rate"] == 1.0
          and g3["parts"][0]["pattern"] == "striped"
          and g3["parts"][0]["size"] == "large", str(g3["parts"]))

    # ---------- R3 · P1③ 补全：组合变换群性质 + 关系层平移 ----------
    # 组合律：rot90cw∘rot90cw == rot180；fliph∘flipv == rot180（逐格验证）
    comp_ok = all(
        ZONE_TRANSFORMS["rot90cw"](*ZONE_TRANSFORMS["rot90cw"](r, c))
        == ZONE_TRANSFORMS["rot180"](r, c)
        and ZONE_TRANSFORMS["fliph"](*ZONE_TRANSFORMS["flipv"](r, c))
        == ZONE_TRANSFORMS["rot180"](r, c)
        for r in range(3) for c in range(3))
    check("组合律（rot90²=rot180；fliph∘flipv=rot180）", comp_ok)
    # 平移：入界 100% 构造性 + 出界诚实落账
    parts = [{"shape": "circle", "color": "red", "zone": z}
             for z in ("r0", "r4", "r8")]
    sh = shift_relations(parts, 1, 1)
    check("平移入界映射（r0→r4 r4→r8）",
          [p["zone"] for p in sh["parts"]] == ["r4", "r8"]
          and len(sh["dropped"]) == 1 and sh["dropped"][0]["from"] == "r8",
          str(sh))
    sh2 = shift_relations(parts, -1, 0)
    check("平移出界落账（r0 出界丢弃；r4→r1 r8→r5）",
          len(sh2["parts"]) == 2
          and [p["zone"] for p in sh2["parts"]] == ["r1", "r5"]
          and len(sh2["dropped"]) == 1
          and sh2["dropped"][0]["from"] == "r0",
          str(sh2))
    rng4 = np.random.default_rng(41)
    n_ok = n_all = n_drop_expect = n_drop_got = 0
    for t in range(12):
        k = int(rng4.integers(1, 4))
        base_parts = [{"shape": SHAPES[rng4.integers(3)],
                       "color": COLORS[rng4.integers(3)],
                       "zone": POS[rng4.integers(9)]} for _ in range(k)]
        s = shift_relations(base_parts, int(rng4.integers(-2, 3)),
                            int(rng4.integers(-2, 3)))
        _, log = render_relations(s["parts"], size=48, seed=t)
        v = verify_constructive(s["parts"], log)
        n_ok += v["matched"]
        n_all += v["total"]
        n_drop_expect += sum(1 for p in base_parts
                             if not (0 <= zone_rc(p["zone"])[0] < 3))
        n_drop_got += len(s["dropped"])
    rate = n_ok / n_all
    check("平移后构造性兑现率 ≥95%", rate >= 0.95, f"rate={rate:.3f}")
    print(f"    [数字] 平移（入界部件）兑现率 = {rate:.1%}；"
          f"出界落账 {n_drop_got} 条")
    return _results


def test_hex_gen_p1():
    p, f = run_checks()
    assert f == 0, f"{f} checks failed"


if __name__ == "__main__":
    run_checks()
    total = _results[0] + _results[1]
    print(f"\n===== HEX-GEN P1 回归: {_results[0]}/{total} 通过 =====")
    sys.exit(1 if _results[1] else 0)
