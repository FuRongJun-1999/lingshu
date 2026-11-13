# -*- coding: utf-8 -*-
"""test_issue1960_multi_seed_gen_only · #1960 守卫：`--gen` 只生成模式必须写出报告
============================================================================
本文件只钉「缺陷不再存在」，不是「代码能跑」。

#1960（成立）· 第 ④ 行打印落在 `if a.measure or do_both:` 块**外**
    现场：`81f68ccc` 把 `print(f"    ④ 空读件 {s['n_img_empty']}/...")` 从 8 格缩进
    改成 4 格 ⇒ 该行无条件执行，而它引用的 `s`（`rep["summary"]`）与 `er`
    只在 `if a.measure or do_both:` 块内赋值。于是：
      · `--gen`（只生成，模块 docstring 第 30 行承诺可单跑）：`s` 从未赋值 ⇒
        `UnboundLocalError`，**进程崩溃且 JSON 报告一字节不写**（写报告在其后），
        已渲染成果全丢；
      · `--measure`/无参：`s` 已赋值能跑过，但该行脱离了量测块，逻辑上不再是
        「量测结果的第 ④ 项」。
    判据来源：本模块 docstring 的命令行契约（`--gen` 只生成、`--measure` 只量测）。

    断言（真跑 `main()`，不做源码字符串匹配）：
      1. `--gen` 必须**正常返回**且 `--out` 报告已写出、含 `generation` 段、
         **不含** `summary`/`within_vs_between`（只生成不量测）；
      2. 反向对照：`--gen` 下不得打印 ④ 行——挡住「把 `s` 预先初始化成 None」一类
         只消错、仍让量测行无条件执行的假修复（那样 ④ 会在只生成模式下打印）；
      3. `--measure` 反向对照：④ 行必须照旧打印（不把整行删掉迁就崩溃）。
    抽掉修复（缩进回 4 格）⇒ 断言 1 先炸成 `UnboundLocalError` ⇒ 本组判红。

环境依赖的隔离：`gen` 需要本地 Qwen-Image-2.1 权重 + torch，`measure` 需要标定语料，
两者都不进测试环境；本件只替换这两处**环境**依赖（`M.gen` / `M.load_multi`+
`M.measure`+`M.summarize`），argparse、分支判定、print、`os.makedirs`、json 写报告
全部走原样实现——缺陷正落在这条原样链上。

运行（lingshu 仓根）：
    python -X utf8 -m pytest tests/test_issue1960_multi_seed_gen_only.py -q --no-header
"""
from __future__ import annotations

import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.gen import hexgen_multi_seed as M       # noqa: E402

FAKE_PROMPTS = [{"sid": "vs_s000_circle_red",
                 "prompt": "flat vector illustration, one red circle(s) in the center"}]


def _fake_gen(prompts, seeds=M.SEEDS, outdir=M.OUTDIR):
    """假 gen：与真实现同键（见模块 `gen()` 返回体），代表「渲染已成功完成」。"""
    rows = [{"sid": it["sid"], "prompt": it["prompt"], "seed": s,
             "path": os.path.join(outdir, it["sid"], "s%d.png" % s)}
            for it in prompts for s in seeds]
    return {"load_s": 0.0, "gen_s": 0.0, "n_new": len(rows), "n_img": len(rows),
            "steps": M.STEPS, "size": M.SIZE, "rows": rows}


def _fake_summary():
    return {"n_img": 0, "n_prompt": 0, "stability": None, "honored": None,
            "n_img_empty": 0, "empty_rate": None,
            "within_vs_between": {"within_prompt_defined": False,
                                  "median_ratio": None, "worst_dims": []}}


def _run(monkeypatch, tmp_path, *args):
    out = tmp_path / "report.json"
    monkeypatch.setattr(M, "pick_prompts", lambda n: list(FAKE_PROMPTS))
    monkeypatch.setattr(M, "gen", _fake_gen)
    monkeypatch.setattr(sys, "argv", ["hexgen_multi_seed.py", *args, "--out", str(out)])
    M.main()
    return out


def test_gen_only_writes_report_and_does_not_measure(monkeypatch, capsys, tmp_path):
    """#1960 核心：`--gen` 只生成必须正常返回并写出报告（缺陷形态在此抛
    `UnboundLocalError`，报告文件不存在）。"""
    monkeypatch.setattr(M, "summarize",
                        lambda *a, **k: pytest.fail("--gen 只生成模式不得调用 summarize"))

    out = _run(monkeypatch, tmp_path, "--gen", "--prompts", "1", "--seeds", "2")

    assert out.is_file(), "--gen 崩溃点之后才写报告 ⇒ 报告一个字节都没写出（#1960）"
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["generation"]["n_img"] == 2, rep.get("generation")
    assert len(rep["rows"]) == 2, rep.get("rows")
    assert "summary" not in rep, "只生成模式不应有量测段"
    assert "within_vs_between" not in rep, "只生成模式不应有量测段"


def test_gen_only_does_not_print_measure_line(monkeypatch, capsys, tmp_path):
    """反向对照：④「空读件」是**量测结果的第 ④ 项**，只生成模式不得打印它。

    挡住假修复：把 `s` 提前初始化（如 `s = {}`）虽能消掉 `UnboundLocalError`，
    但该行仍在块外 ⇒ 只生成模式会打印 ④（并可能读空 dict 再炸）⇒ 本断言判红。
    """
    _run(monkeypatch, tmp_path, "--gen", "--prompts", "1", "--seeds", "2")
    out_txt = capsys.readouterr().out
    assert "生成：新增 2/2 张" in out_txt, out_txt
    assert "④" not in out_txt, "只生成模式打印了量测第 ④ 项（行仍在 if 块外）：" + out_txt


def test_measure_mode_still_prints_line_four(monkeypatch, capsys, tmp_path):
    """反向对照：`--measure` 必须照旧打印 ④（不得把整行删掉迁就崩溃）。"""
    monkeypatch.setattr(M, "load_multi", lambda rows: [])
    monkeypatch.setattr(M, "measure", lambda items, d=4: ([], None))
    monkeypatch.setattr(M, "summarize", lambda reads, items: _fake_summary())

    out = _run(monkeypatch, tmp_path, "--measure", "--prompts", "1", "--seeds", "2")

    assert out.is_file(), "只量测模式也必须写出报告"
    out_txt = capsys.readouterr().out
    assert "④ 空读件 0/0（无样本）" in out_txt, out_txt
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["summary"]["empty_rate"] is None
