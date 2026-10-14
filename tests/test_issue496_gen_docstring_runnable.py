# -*- coding: utf-8 -*-
"""test_issue496_gen_docstring_runnable · gen 侧 docstring 命令行须指向真实可跑路径（#496）
======================================================================================
缺陷（#496）：4 个 gen 模块 docstring 的「命令行」仍指本仓**不存在**的
``experiments/`` 目录（模块化落仓前的旧路径照抄），而同文件下方 #108 落仓
注释又自己写明「``experiments/`` 目录在本仓不存在」——同一文件自相矛盾；
读者照 docstring 跑即 ``No such file``。#108 的落仓守卫只收口了 ``sys.path``
半，对 docstring 行零判别力（本件补上）。

修法：docstring 命令行改为包内真实入口 ``python -X utf8 -m lingshu.gen.<name>``
（四件均有 ``main()`` + argparse + ``__main__``；``-m --help`` 实测 rc=0）。

判据（本件）：
  G1 全 gen 面扫描：docstring 命令行不得再出现 ``experiments/``（正则锚
     ``python …experiments/`` / ``PYTHONPATH=…experiments/``；改前命中 6 行）；
  G2 修复后的命令真实可跑：四件 ``-m … --help`` 退出码 0（文档承诺可兑现）；
  G3 定点变异：向扫描器喂含 ``python experiments/x.py`` 的合成 docstring ⇒
     必命中（证明 G1 的 0 是「扫过且无」而非「没扫到」——正是 #496 对旧守卫
     的批评形态）。

运行（仓根）：python -X utf8 -m pytest tests/test_issue496_gen_docstring_runnable.py -q
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GEN = REPO / "lingshu" / "gen"

# docstring 命令行形态的 experiments/ 引用：行内出现「python <…>experiments/….py」
# 即命中（不限行首——真实文档有「命令行：python …」带前缀形态与
# 「PYTHONPATH=. python …」前缀形态；解释性注释只提目录名、不含该命令形状）
RE_EXPERIMENTS_CMD = re.compile(r"python\s+\S*experiments/\S*\.py")

MODULES = ("hexgen_self_source", "hexgen_align_probe",
           "hexgen_multi_seed", "hexgen_pack_certify")


def _cmd_lines(text: str):
    return [ln.strip() for ln in text.splitlines()
            if RE_EXPERIMENTS_CMD.search(ln)]


def test_no_experiments_cmdline_left_in_gen_docstrings():
    """G1：gen 全面包内，docstring 命令行零 ``experiments/`` 残留（改前 6 行）。"""
    hits = {}
    for p in sorted(GEN.glob("*.py")):
        lines = _cmd_lines(p.read_text(encoding="utf-8"))
        if lines:
            hits[p.name] = lines
    assert hits == {}, (
        "#496 docstring 命令行仍指向本仓不存在的 experiments/（同文件注释已自证"
        "该目录不存在，读者照跑即 No such file）：\n"
        + "\n".join("  · %s: %r" % (f, ls) for f, ls in hits.items())
        + "\n\n修法：改为包内真实入口 `python -X utf8 -m lingshu.gen.<name>`。")


def test_documented_module_entry_actually_runs():
    """G2：四件 `-m … --help` 退出码 0——docstring 承诺的命令可兑现。"""
    for m in MODULES:
        r = subprocess.run(
            [sys.executable, "-X", "utf8", "-m", "lingshu.gen." + m, "--help"],
            cwd=str(REPO), capture_output=True, timeout=120)
        assert r.returncode == 0, (
            "#496 docstring 指定的入口跑不起来：%s → rc=%d stderr=%r"
            % (m, r.returncode, r.stderr.decode("utf-8", "replace")[-200:]))


def test_mutation_scanner_catches_experiments_cmdline():
    """G3（定点变异自证）：合成旧形态 docstring ⇒ 扫描器必命中（含带前缀形态）。"""
    synthetic = (
        '"""示例模块。\n'
        '命令行：python experiments/example.py --gen\n'
        '    python experiments/example.py --measure\n'
        '    PYTHONPATH=. python experiments/example.py --prompt 1\n'
        '"""\n')
    hits = _cmd_lines(synthetic)
    assert len(hits) == 3, (
        "扫描器应命中合成旧形态的全部 3 行（含「命令行：」带前缀与 PYTHONPATH 前缀）"
        "——实测 %r；若不命中，G1 判据空转" % hits)
    # 对照：修复后形态不命中；解释性注释（只提目录名）不命中
    ok = ('"""示例。\n命令行：python -X utf8 -m lingshu.gen.example\n'
          '# experiments/ 目录在本仓不存在（解释性注释，非命令行）\n"""\n')
    assert _cmd_lines(ok) == [], "修复后形态/解释性注释不应命中"


if __name__ == "__main__":
    raise SystemExit(subprocess.call(
        [sys.executable, "-X", "utf8", "-m", "pytest", __file__, "-q", "--no-header"]))
