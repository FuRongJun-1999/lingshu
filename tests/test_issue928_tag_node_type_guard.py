# -*- coding: utf-8 -*-
"""#928 守卫：`tag_node` 的 tag 必须是非空字符串——list 入参不得嵌套污染 tags 数组。

缺陷面（公开 issue #928）：`tag_node(nid, ["甲"])` 不校验类型，整个列表被当作
**一个元素** append 进 `nodes.tags` ⇒ 串行两次得嵌套 `[["甲"],["乙"]]`（跨进程/
同进程同样成立——损坏源是类型未校验，不是进程边界）。跨进程+字符串 tag 的路径
由 #344 的 BEGIN IMMEDIATE 保证，无恙。

本件锁：
  ① list/tuple/dict/数字/空串 ⇒ ValueError（fail-closed，在**任何写之前**抛出）；
  ② 拒收后 tags 原样（不落半截数据）；
  ③ 合法字符串 tag 跨进程串行 ⇒ 扁平数组按调用序累积（跨进程面回归）；
  ④ 幂等：同 tag 二次打标不重复。
定点变异：去掉类型闸 ⇒ ① 红。
"""
import json
import os
import sqlite3
import subprocess
import sys

import pytest

from lingshu.core.core import SpacetimeMemoryEngine

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- ① 类型闸

@pytest.mark.parametrize("bad", [
    ["甲"], ("乙",), {"tag": 1}, 123, None, "",
])
def test_non_string_tag_rejected(bad):
    e = SpacetimeMemoryEngine(":memory:")
    nid = e.add_perception("类型闸探针").id
    with pytest.raises(ValueError):
        e.store.tag_node(nid, bad)
    node = e.store.get_node(nid)
    assert node.tags == [t for t in node.tags if t], "拒收后不得落半截数据"


# ---------------------------------------------------------------- ② 拒收后原样

def test_rejection_leaves_tags_untouched():
    e = SpacetimeMemoryEngine(":memory:")
    nid = e.add_perception("先打合法标签").id
    e.store.tag_node(nid, "合法")
    with pytest.raises(ValueError):
        e.store.tag_node(nid, ["嵌套"])
    node = e.store.get_node(nid)
    assert node.tags == ["合法"], "拒收后既有标签不得变动：%r" % (node.tags,)


# ---------------------------------------------------------------- ③ 跨进程串行

def test_cross_process_string_tags_stay_flat(tmp_path):
    db = os.path.join(str(tmp_path), "a.db")
    tmpl = (
        "import sys, warnings; warnings.filterwarnings('ignore')\n"
        "sys.path.insert(0, r'%s')\n"
        "from lingshu.core.core import SpacetimeMemoryEngine\n"
        "step, db = sys.argv[1], sys.argv[2]\n"
        "nid = sys.argv[3] if len(sys.argv) > 3 else None\n"
        "tag = sys.argv[4] if len(sys.argv) > 4 else None\n"
        "e = SpacetimeMemoryEngine(db)\n"
        "if step == '1':\n"
        "    print(e.add_perception('跨进程探针').id)\n"
        "else:\n"
        "    e.store.tag_node(nid, tag)\n"
        "    print('tagged')\n"
    ) % REPO

    def run(step, *extra):
        p = subprocess.run(
            [sys.executable, "-X", "utf8", "-c", tmpl, step, db, *extra],
            capture_output=True, text=True, encoding="utf-8")
        assert p.returncode == 0, p.stderr[-300:]
        return p.stdout.strip()

    nid = run("1")
    run("2", nid, "甲")
    run("3", nid, "乙")
    tags = json.loads(sqlite3.connect(db).execute(
        "SELECT tags FROM nodes WHERE id=?", (nid,)).fetchone()[0])
    assert tags == ["甲", "乙"], "跨进程串行字符串 tag 应为扁平数组：%r" % (tags,)


# ---------------------------------------------------------------- ④ 幂等

def test_same_tag_twice_no_duplicate():
    e = SpacetimeMemoryEngine(":memory:")
    nid = e.add_perception("幂等探针").id
    e.store.tag_node(nid, "甲")
    e.store.tag_node(nid, "甲")
    assert e.store.get_node(nid).tags == ["甲"]
