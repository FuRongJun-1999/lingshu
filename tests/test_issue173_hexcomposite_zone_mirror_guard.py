# -*- coding: utf-8 -*-
"""守卫 · #173：`hex_composite._ZONES["bottom-right"]` 的左右镜像不成立

缺陷（main 40fceb6）
    `lingshu/nn/hex_composite.py` 与 `lingshu/gen/hex_composite.py` 的 `_ZONES` 里
    `"bottom-right": {"r5", "r8"}`，而 `"bottom-left": {"r6", "r7"}`。
    3×3 编码为 `r{row*3+col}`，左右镜像映射是 `col -> 2-col`：
    `r6 -> r8`、`r7 -> r7`，故 `bottom-right` 应为 `{"r7", "r8"}`。
    旧值 `{"r5","r8"}` 里的 `r5` 属中右行（`r3,r4,r5` 的右端），是错配。

两文件同改（硬约束）
    `lingshu/nn/hex_composite.py` 与 `lingshu/gen/hex_composite.py` 全文件逐字节相同
    （LF 归一 sha256 = `111796661833fe52982a4feceb85d996faac9bba05014381aa22e25e4ad4ec37`），
    且既有守卫 `test_issue171_hex_composite_run_count.py::test_g4_dual_copies_byte_identical`
    与 `test_issue196_nonfinite_gate.py::TestHexCompositeDualCopyIdentical::test_byte_identical`
    断言两份原始字节 sha256 相等 ⇒ 只改一份会直接点红这两条 G4。
    两模块互不 import，各自定义 `_ZONES`（`core.py:3750` 明写副本合一属架构改动，不在本件范围）。

断言组（`@pytest.mark.parametrize` 覆盖两份模块）
    Z1 两份模块的 `bottom-right` 都等于 `{"r7","r8"}`（镜像真值）
    Z2 两份模块的 `bottom-left` 仍是 `{"r6","r7"}`，且 `bottom-right` 恰为其左右镜像
       （`col -> 2-col` 逐元素映射）
    Z3 反向不变式：`top`/`bottom`/`left`/`right`/`top-left`/`top-right` 六个 zone 不变
       （防止「为了修镜像把整行整列改坏」）

变异自证（本文件自带，非仓库原生机制）
    python -X utf8 tests/test_issue173_hexcomposite_zone_mirror_guard.py --mutate
    M1 只改 nn（gen 保持缺陷值）      ⇒ Z1/Z2 的 gen 参数红（+ 两份 G4 红）
    M2 只改 gen（nn 保持缺陷值）      ⇒ Z1/Z2 的 nn 参数红（+ 两份 G4 红）——M2 绿即 SHADOW_SITE
    M3 两处都改对                      ⇒ 全绿
    M4 nn 改对、gen 保留旧值           ⇒ 同 M2（证明「只改一处」不可能蒙混）
    每臂跑完按字节 sha256 校验两文件原样还原。

运行（仓根）：python -X utf8 -m pytest tests/test_issue173_hexcomposite_zone_mirror_guard.py -q --no-header
"""
from __future__ import annotations

import hashlib
import importlib
import os
import re
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

MODS = ("lingshu.nn.hex_composite", "lingshu.gen.hex_composite")
NN_PATH = os.path.join(REPO, "lingshu", "nn", "hex_composite.py")
GEN_PATH = os.path.join(REPO, "lingshu", "gen", "hex_composite.py")
RUN_TIMEOUT = 300

_BAD_LINE = '"bottom-right": {"r5", "r8"},'
_GOOD_LINE = '"bottom-right": {"r7", "r8"},'

#: 反向不变式：其余六个 zone 的期望值（与 40fceb6 一致，本件不得改动）
_UNCHANGED = {
    "top": {"r0", "r1", "r2"},
    "bottom": {"r6", "r7", "r8"},
    "left": {"r0", "r3", "r6"},
    "right": {"r2", "r5", "r8"},
    "top-left": {"r0", "r3"},
    "top-right": {"r2", "r5"},
}


def _zones(mod_name):
    mod = importlib.import_module(mod_name)
    return mod._ZONES


def _mirror(cells):
    """左右镜像：r{row*3+col} -> r{row*3+(2-col)}。"""
    out = set()
    for c in cells:
        assert c.startswith("r") and c[1:].isdigit(), "非预期编码：%r" % (c,)
        n = int(c[1:])
        row, col = divmod(n, 3)
        out.add("r%d" % (row * 3 + (2 - col)))
    return out


@pytest.mark.parametrize("mod_name", MODS)
def test_z1_bottom_right_is_mirror_value(mod_name):
    """Z1：两份模块的 bottom-right 都必须是镜像真值 {"r7","r8"}。"""
    z = _zones(mod_name)
    assert z["bottom-right"] == {"r7", "r8"}, (
        "%s 的 bottom-right = %r，期望 {'r7','r8'}（#173：r5 属中右行，是错配）"
        % (mod_name, z["bottom-right"]))


@pytest.mark.parametrize("mod_name", MODS)
def test_z2_bottom_pair_mirrors(mod_name):
    """Z2：bottom-left 不变，且 bottom-right 恰为其 col->2-col 镜像。"""
    z = _zones(mod_name)
    assert z["bottom-left"] == {"r6", "r7"}, (
        "%s 的 bottom-left = %r（不应被改动）" % (mod_name, z["bottom-left"]))
    assert z["bottom-right"] == _mirror(z["bottom-left"]), (
        "%s：bottom-right %r 不是 bottom-left %r 的左右镜像"
        % (mod_name, z["bottom-right"], z["bottom-left"]))


@pytest.mark.parametrize("mod_name", MODS)
def test_z3_other_zones_unchanged(mod_name):
    """Z3：反向不变式——其余六个 zone 逐字不变（防整行整列被改坏）。"""
    z = _zones(mod_name)
    bad = {k: (z.get(k), v) for k, v in _UNCHANGED.items() if z.get(k) != v}
    assert not bad, "%s 的其他 zone 被改动：%r" % (mod_name, bad)


# ---------------------------------------------------------------- --mutate

_MUTATIONS = (
    # name, {path: line_to_write}, expected red nodeids
    ("M1_nn_only", {NN_PATH: _GOOD_LINE, GEN_PATH: _BAD_LINE},
     {"test_z1_bottom_right_is_mirror_value[lingshu.gen.hex_composite]",
      "test_z2_bottom_pair_mirrors[lingshu.gen.hex_composite]"}),
    ("M2_gen_only", {NN_PATH: _BAD_LINE, GEN_PATH: _GOOD_LINE},
     {"test_z1_bottom_right_is_mirror_value[lingshu.nn.hex_composite]",
      "test_z2_bottom_pair_mirrors[lingshu.nn.hex_composite]"}),
    ("M3_both_good", {NN_PATH: _GOOD_LINE, GEN_PATH: _GOOD_LINE}, set()),
    ("M4_one_good_one_bad", {NN_PATH: _GOOD_LINE, GEN_PATH: _BAD_LINE},
     {"test_z1_bottom_right_is_mirror_value[lingshu.gen.hex_composite]",
      "test_z2_bottom_pair_mirrors[lingshu.gen.hex_composite]"}),
)


def _set_bottom_right(path, line):
    """把该文件的 bottom-right 行改成给定行（其余不动）。"""
    raw = open(path, "rb").read().decode("utf-8")
    cur = _GOOD_LINE if _GOOD_LINE in raw else _BAD_LINE
    assert raw.count(cur) == 1, "%s 里 bottom-right 行不唯一：%r" % (path, cur)
    new = raw.replace(cur, line, 1)
    open(path, "wb").write(new.encode("utf-8"))


def _run_pytest():
    cmd = [sys.executable, "-X", "utf8", "-m", "pytest", os.path.abspath(__file__),
           "-q", "--no-header", "-p", "no:cacheprovider"]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=RUN_TIMEOUT)
    out = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
    red = set()
    for line in out.splitlines():
        m = re.match(r"^(?:FAILED|ERROR) .*::(\S+)", line)
        if m:
            red.add(m.group(1))
    summary = [l for l in out.splitlines() if " passed" in l or " failed" in l]
    return red, (summary[-1] if summary else "")


def _mutate():
    orig = {p: open(p, "rb").read() for p in (NN_PATH, GEN_PATH)}
    digests = {p: hashlib.sha256(d).hexdigest() for p, d in orig.items()}
    bad = 0
    for name, spec, expect in _MUTATIONS:
        try:
            for path, line in spec.items():
                _set_bottom_right(path, line)
            red, summary = _run_pytest()
            print("  %-22s 期望点红=%d 实际点红=%d %s"
                  % (name, len(expect), len(red), summary))
            if red != expect:
                bad += 1
                print("    VERDICT=FAIL 多红=%r 漏红=%r"
                      % (sorted(red - expect), sorted(expect - red)))
            else:
                print("    VERDICT=PASS")
        finally:
            for path, data in orig.items():
                open(path, "wb").write(data)
            for path, dg in digests.items():
                now = hashlib.sha256(open(path, "rb").read()).hexdigest()
                assert now == dg, "%s 未原样还原" % path
    print("  OVERALL=%s" % ("PASS" if bad == 0 else "FAIL",))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    if "--mutate" in sys.argv:
        sys.exit(_mutate())
    sys.exit(0)
