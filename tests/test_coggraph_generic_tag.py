# -*- coding: utf-8 -*-
"""test_coggraph_generic_tag · issue #77 守卫：GENERIC_TAG 短前缀分支必须带非字母边界
============================================================================
缺陷（lingshu issue #77 · 报告人 123WP-a）：`tools/coggraph/derive_edges.py` 的
`GENERIC_TAG` 三个短前缀分支只锚定了开头（`^sha` / `^precise` / `^v\d`）⇒
`shape` / `shared` / `shadow` / `sharding` / `precisely` / `v2model` 这类**真实内容标签**
被 `content_tags()` 当作结构性标签剔除，R3 标签共现边随之少算（边被静默丢弃）。

修法：三个分支各补一个**非字母边界** `(?![a-z])`。取 `(?![a-z])` 而不是 `($|:)`，
是为了让 `sha256` / `precise_pose` / `v2.1` 这些**原有结构性剔除保持不变**——
只收紧误杀面，不放开任何既有口径（C 组断言即为此设）。

断言组：
  A 组（阳性对照）：结构性标签必须仍被剔除（含 sha256 / precise_pose / v2.1 等原口径）
  B 组（阴性对照）：内容标签不得再被误杀（issue #77 列举的 6 个 + 同族）
  C 组（收紧边界）：新旧判据的「剔除集合」差异必须恰好等于 B 组（无额外放宽/收紧）
  D 组（端到端 · content_tags）：共现用的标签集合符合预期；normalize 仍生效
  E 组（端到端 · R3 边）：最小 graph.json 走 `main()` 全链路，修复后 R3 共现边成立
  F 组（夹具自证）：同一夹具在**旧判据**下共享内容标签数 < min_shared ⇒ 边不成立
                     （证明 E 组夹具确有辨别力，不是恒真断言）

运行（lingshu 仓根）：python -X utf8 tests/test_coggraph_generic_tag.py
退出码：0 = 全过；1 = 有断言失败
"""
from __future__ import annotations

import io
import json
import os
import re
import shutil
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools", "coggraph"))

# 经包路径导入（`tools` 在依赖闭包守卫 tests/test_gate_dependency_closure.py 的 LOCAL_TOP 白名单内）；
# 裸 `import derive_edges` 会被该守卫判为「未声明的第三方模块」——本地 coggraph 模块须走包路径或 spec 加载。
from tools.coggraph import derive_edges as D  # noqa: E402

#: 上游（修复前）判据——只在本件里作对照读数用，不改仓内文件
OLD_GENERIC_TAG = re.compile(
    r"^(doc|md|node|unit|skill|md$|case)$|^doc:|^level:|^lang:|^sha|^precise|^v\d")

#: 结构性/形态标签（阳性对照：修复前后都必须剔除）
STRUCTURAL_TAGS = (
    "doc", "md", "node", "unit", "skill", "case",          # 裸形态
    "doc:spec", "level:2", "lang:zh",                       # 命名空间式
    "sha", "sha256", "sha:abc123",                          # 短前缀 + 非字母
    "precise", "precise_pose",                              # 短前缀 + 非字母
    "v1", "v2", "v10", "v2.1",                              # 版本形态
)

#: 内容标签（阴性对照：修复后必须保留）——前 6 个即 issue #77 的误杀清单
CONTENT_TAGS = (
    "shape", "shared", "shadow", "sharding", "precisely", "v2model",
    "preciseness", "v2abc", "shampoo",
    "circle", "stripe", "red", "square", "triangle",        # 几何/颜色（原本就保留）
)

#: 预期「收紧」清单：旧判据剔除、新判据保留的标签
EXPECTED_UNKILLED = (
    "shape", "shared", "shadow", "sharding", "precisely", "v2model",
    "preciseness", "v2abc", "shampoo",
)

#: E 组夹具：两节点共享 4 个内容标签（修复后），旧判据下只剩 1 个 ⇒ 边得失可见
FIXTURE_NODES = [
    {"id": "n1", "title": "alpha", "layer": "contextual",
     "tags": ["shape", "shadow", "sharding", "circle", "red"]},
    {"id": "n2", "title": "beta", "layer": "contextual",
     "tags": ["shape", "shadow", "sharding", "circle", "blue"]},
    {"id": "n3", "title": "gamma", "layer": "contextual",
     "tags": ["doc", "md", "v1"]},          # 全是形态标签 ⇒ 无内容标签
]

_PASS = []
_FAIL = []


def ok(cond, msg, extra=""):
    (_PASS if cond else _FAIL).append(msg)
    print(("  PASS " if cond else "  FAIL ") + msg
          + (("  ← " + str(extra)) if (extra and not cond) else ""))


def old_content_tags(tags):
    """按**旧**判据算内容标签（对照用，复刻 content_tags 的语义）。"""
    out = []
    for t in (tags or []):
        nt = str(t).strip().lower()
        if not nt or OLD_GENERIC_TAG.match(nt):
            continue
        out.append(nt)
    return frozenset(out)


def group_a_structural_still_killed():
    for t in STRUCTURAL_TAGS:
        ok(bool(D.GENERIC_TAG.match(t)), "A 结构性标签仍被剔除：%s" % t)


def group_b_content_not_killed():
    for t in CONTENT_TAGS:
        ok(not D.GENERIC_TAG.match(t), "B 内容标签不再被误杀：%s" % t)


def group_c_tightening_is_exact():
    sample = list(STRUCTURAL_TAGS) + list(CONTENT_TAGS) + ["knowledge", "verification"]
    old_killed = {t for t in sample if OLD_GENERIC_TAG.match(t)}
    new_killed = {t for t in sample if D.GENERIC_TAG.match(t)}
    extra_killed = sorted(new_killed - old_killed)
    unkilled = sorted(old_killed - new_killed)
    ok(not extra_killed, "C 新判据没有多杀任何标签（只收紧、不放开）", extra_killed)
    ok(unkilled == sorted(EXPECTED_UNKILLED),
       "C 放开清单恰好等于预期误杀清单", "got=%s" % (unkilled,))
    print("  [读数] 样本 %d 个：旧判据剔除 %d 个 → 新判据剔除 %d 个（放开 %d 个）"
          % (len(sample), len(old_killed), len(new_killed), len(unkilled)))


def group_d_content_tags_end_to_end():
    got = D.content_tags(["shape", "shadow", "doc", "v1"])
    ok(got == frozenset({"shape", "shadow"}),
       "D content_tags 只剔除形态标签", "got=%s" % (sorted(got),))
    ok(D.content_tags([" ShaPe "]) == frozenset({"shape"}),
       "D norm_tag 的 strip+lower 仍生效",
       "got=%s" % (sorted(D.content_tags([" ShaPe "])),))
    ok(D.content_tags(["", "shape", "  "]) == frozenset({"shape"}),
       "D 空标签被跳过、不进入内容标签集",
       "got=%s" % (sorted(D.content_tags(["", "shape", "  "])),))


def run_pipeline(nodes, outdir, with_old_judgement=False):
    """把夹具图交给 derive_edges.main() 全链路跑一遍，返回 meta.rules。"""
    os.makedirs(outdir, exist_ok=True)
    gpath = os.path.join(outdir, "graph.json")
    with io.open(gpath, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"meta": {"source_sha": "fixture"}, "nodes": nodes, "edges": []},
                  f, ensure_ascii=False)
    saved = D.GENERIC_TAG
    if with_old_judgement:
        D.GENERIC_TAG = OLD_GENERIC_TAG
    try:
        rc = D.main(["--graph", gpath, "--out", os.path.join(outdir, "out"),
                     "--write", "--sample", "5"])
    finally:
        D.GENERIC_TAG = saved
    ok(rc == 0, "E/F main() 正常返回", "rc=%s" % (rc,))
    with io.open(os.path.join(outdir, "out", "derived_edges.json"), encoding="utf-8") as f:
        out = json.load(f)
    return out["meta"]["rules"], out


def group_e_r3_edge_present():
    tmp = tempfile.mkdtemp(prefix="coggraph_r3_")
    try:
        rules, out = run_pipeline(FIXTURE_NODES, os.path.join(tmp, "new"))
        r3 = int(rules.get("R3_tag_jaccard", 0))
        ok(r3 == 1, "E 修复后 R3 共现边成立（shape/shadow/sharding 参与共现）",
           "R3=%s rules=%s" % (r3, rules))
        jac = [e["weight"] for e in out["edges"] if e["rule"] == "R3_tag_jaccard"]
        ok(jac and abs(jac[0] - 0.667) < 0.002, "E R3 边权重=Jaccard(4/6)", "w=%s" % (jac,))
        ok(not [e for e in out["edges"] if e["s"] == "n3" or e["t"] == "n3"],
           "E 纯形态标签节点 n3 不产生任何推导边",
           [e["rule"] for e in out["edges"] if "n3" in (e["s"], e["t"])])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def group_f_fixture_has_discriminating_power():
    tmp = tempfile.mkdtemp(prefix="coggraph_r3_old_")
    try:
        rules, _ = run_pipeline(FIXTURE_NODES, os.path.join(tmp, "old"), with_old_judgement=True)
        r3_old = int(rules.get("R3_tag_jaccard", 0))
        ok(r3_old == 0, "F 旧判据下同一夹具不产生 R3 边（夹具具辨别力）",
           "R3_old=%s rules=%s" % (r3_old, rules))
        a = old_content_tags(FIXTURE_NODES[0]["tags"])
        b = old_content_tags(FIXTURE_NODES[1]["tags"])
        ok(len(a & b) == 1 and len(a & b) < 3,
           "F 旧判据下共享内容标签数 < min_shared(3)", "shared=%s" % sorted(a & b))
        a_new = D.content_tags(FIXTURE_NODES[0]["tags"])
        b_new = D.content_tags(FIXTURE_NODES[1]["tags"])
        ok(len(a_new & b_new) == 4,
           "F 新判据下共享内容标签数 = 4", "shared=%s" % sorted(a_new & b_new))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    print("== A 组：阳性对照（结构性标签仍被剔除） ==")
    group_a_structural_still_killed()
    print("== B 组：阴性对照（内容标签不再被误杀） ==")
    group_b_content_not_killed()
    print("== C 组：收紧边界恰好等于预期 ==")
    group_c_tightening_is_exact()
    print("== D 组：content_tags 端到端 ==")
    group_d_content_tags_end_to_end()
    print("== E 组：R3 共现边端到端（修复后成立） ==")
    group_e_r3_edge_present()
    print("== F 组：夹具自证（旧判据下边不成立） ==")
    group_f_fixture_has_discriminating_power()
    print()
    print("===== SUMMARY %d/%d 通过 =====" % (len(_PASS), len(_PASS) + len(_FAIL)))
    if _FAIL:
        print("FAILS:", _FAIL)
        return 1
    print("VERDICT=PASS（issue #77 守卫：GENERIC_TAG 短前缀分支带非字母边界，"
          "只收紧误杀面、不放开既有口径）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
