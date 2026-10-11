# -*- coding: utf-8 -*-
"""test_issue198_hexgen_multi_seed_stable_id · 多 seed 语料 id 必须跨进程可复现
============================================================================
背景（lingshu issue #198）：
  `hexgen_multi_seed.load_multi` 给每张渲染件造的 id 是

      "id": hash((r["sid"], r["seed"])) % 10 ** 9          # 缺陷形态

  内建 `hash()` 对 `str` 施加 **PYTHONHASHSEED 随机化**（CPython 3 默认逐进程
  随机）⇒ 同一份语料在两个进程里得到**不同 id**，报告不可复现、跨进程对不上
  （本仓实测：PYTHONHASHSEED=1/2/3 分别得 883364599 / 620007010 / 312349447）。
  本模块头部红线「固定 seed ⇒ 可复跑」以及下游按 id 排序/索引的用法都要求 id
  与进程无关。

修复：改用 `hashlib.blake2b` 摘要造 id（`_stable_id`，同仓
  `lingshu/world/semantic_anchor_graph.py:90` 亦以哈希摘要造 id），数值域仍
  `% 10**9`，下游只当不透明标识。

断言组（回退 / 放宽即红）：
  G1 `_stable_id(sid, seed)` == blake2b("sid|seed") 前 8 字节 % 10**9——钉死
     修复后的**确定性公式**（缺陷形态为内建 hash，值随 PYTHONHASHSEED 变）。
  G2 子进程实跑：在 `PYTHONHASHSEED=0/1/2` 三个环境里各起一次解释器，
     `load_multi` 对同一 rows 给出的 id **逐个相等**（缺陷形态三个进程给三个值）。
  G3 id 落在 [0, 10**9)（数值域与旧式一致，下游无需改）。

判据来源（逐条写清，不编造）：
  · 「id 必须可复现」= 本模块头部红线「生成侧只用本地管线…固定 seed ⇒ 可复跑」
    （`hexgen_multi_seed.py` 文件头第 22-23 行）与「量测只读复用确定性实现」的
    同一条要求；**理论章节无此定义**，属模块自身口径。
  · `% 10**9` 的数值域 = 被替换式原文；本件**未改**该数值域。
  · `blake2b` 造 id 先例：`lingshu/world/semantic_anchor_graph.py:90`。

不适用条件 / 已知边界：
  · 本守卫只钉 id 的**跨进程可复现性**；`load_multi` 的读图口径（#357）与
    量测链路（#58/#390）不在本件范围。
  · 旧语料 json 里已落盘的旧 id 不会被本件回填（本件只保证**今后重跑**一致）。

定点变异自证（抽掉修复 ⇒ 必红）：
  把 `load_multi` 的 id 改回 `hash((r["sid"], r["seed"])) % 10 ** 9` ⇒
  G2 报三个进程 id 互不相等（逐字读数见报告）。

运行（仓根）：python -X utf8 tests/test_issue198_hexgen_multi_seed_stable_id.py
             / python -X utf8 -m pytest tests/test_issue198_hexgen_multi_seed_stable_id.py
退出码：0 = 全过；1 = 有断言失败
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lingshu.gen import hexgen_multi_seed as M            # noqa: E402

ROWS = [{"sid": "vs_s1_circle_red", "seed": 11},
        {"sid": "vs_s2_triangle_blue", "seed": 22},
        {"sid": "vs_s1_circle_red", "seed": 33}]

# 子进程脚本：读 rows（JSON 经 argv）→ load_multi → 打印 id 列表
_CHILD = r"""
import json, os, sys
sys.path.insert(0, os.environ["LS_REPO"])
from lingshu.gen import hexgen_multi_seed as M
rows = json.loads(sys.argv[1])
ids = [it["id"] for it in M.load_multi(rows)]
print(json.dumps(ids))
"""


def _tmp_rows(tmpdir):
    """把 ROWS 配上真实 PNG 路径（load_multi 会 open 图片）。"""
    p = os.path.join(tmpdir, "s11.png")
    Image.fromarray(np.zeros((6, 7, 3), np.uint8)).save(p)
    return [{"sid": r["sid"], "seed": r["seed"], "path": p,
             "prompt": "flat vector illustration, one red circle(s) in the center"}
            for r in ROWS]


def _child_ids(rows, hashseed):
    env = dict(os.environ, PYTHONHASHSEED=str(hashseed), LS_REPO=REPO,
               PYTHONUTF8="1")
    out = subprocess.run([sys.executable, "-X", "utf8", "-c", _CHILD,
                          json.dumps(rows)],
                         cwd=REPO, env=env, capture_output=True, text=True)
    assert out.returncode == 0, f"子进程失败：{out.stderr}"
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_g1_stable_id_matches_blake2b_formula():
    """G1：id 公式 = blake2b('sid|seed') 前 8 字节 % 10**9。"""
    for sid, seed in (("vs_s1_circle_red", 11), ("vs_s2_triangle_blue", 22)):
        want = int.from_bytes(
            hashlib.blake2b(("%s|%s" % (sid, seed)).encode("utf-8"),
                            digest_size=8).digest(), "big") % 10 ** 9
        got = M._stable_id(sid, seed)
        assert got == want, f"{sid}/{seed}: id={got}（应 {want}）"
        # 且不得等于内建 hash 的口径（同一进程内也可能偶然相等，故只钉公式）
        assert 0 <= got < 10 ** 9


def test_g2_ids_identical_across_pythonhashseed():
    """G2：PYTHONHASHSEED=0/1/2 三个子进程给出的 id 逐个相等。"""
    with tempfile.TemporaryDirectory() as td:
        rows = _tmp_rows(td)
        runs = {s: _child_ids(rows, s) for s in (0, 1, 2)}
    base = runs[0]
    assert len(base) == len(ROWS)
    for s in (1, 2):
        assert runs[s] == base, (
            f"PYTHONHASHSEED={s} 的 id {runs[s]} != seed=0 的 {base}"
            "（内建 hash 的 id 随进程变，不可复现）")


def test_g3_ids_in_range_and_in_process_matches_child():
    """G3：id ∈ [0,10**9)，且本进程与子进程读数一致。"""
    with tempfile.TemporaryDirectory() as td:
        rows = _tmp_rows(td)
        inproc = [it["id"] for it in M.load_multi(rows)]
        child = _child_ids(rows, 7)
    assert all(0 <= i < 10 ** 9 for i in inproc), inproc
    assert inproc == child, f"本进程 {inproc} != 子进程 {child}"


if __name__ == "__main__":
    test_g1_stable_id_matches_blake2b_formula()
    test_g2_ids_identical_across_pythonhashseed()
    test_g3_ids_in_range_and_in_process_matches_child()
    print("OK #198 守卫全过")
