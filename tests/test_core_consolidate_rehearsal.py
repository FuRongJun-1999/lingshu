# -*- coding: utf-8 -*-
"""consolidate_cycle 演练提权节流 —— 判据须用「本次增量后」的访问计数。

语义：高重要度节点每次演练把访问计数 +1，每满 10 次演练提权一次
（access_count % 10 == 0 → importance += gain）。故第 1 轮后计数才到 1（从未被访问）
⇒ 不得提权；第 10 轮后计数刚好到 10 ⇒ 必须提权。本件只断言这两条外部可观察行为
（提权与否 = importance 变化），不读内部字段。

必须用文件库：increment_access 自开一条连接写库，:memory: 与主库不同源。

运行（仓根）：python -X utf8 -m pytest tests/test_core_consolidate_rehearsal.py -v
"""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lingshu.core.core import SpacetimeMemoryEngine  # noqa: E402

GAIN = 0.01   # consolidate_cycle 默认提权步长（容差由它推出，不写死阈值）


@pytest.fixture()
def engine(tmp_path):
    eng = SpacetimeMemoryEngine(str(tmp_path / "consolidate.db"))
    try:
        yield eng
    finally:
        eng.close()   # 关主连接，临时目录才能回收


def _boost_rounds(engine, node_id, rounds):
    """逐轮巩固，返回发生提权的轮次（1 起）；只看外部可见的 importance。"""
    boosted = []
    for rnd in range(1, rounds + 1):
        before = engine.store.get_node(node_id).importance
        engine.consolidate_cycle()
        if engine.store.get_node(node_id).importance > before + GAIN / 2:
            boosted.append(rnd)
    return boosted


def test_first_boost_lands_on_tenth_rehearsal(engine):
    """第 1 轮不得提权；首次提权应落在第 10 次演练。"""
    node = engine.add_perception("高重要度记忆", importance=0.85, skip_dedup=True)
    boosted = _boost_rounds(engine, node.id, 12)
    assert 1 not in boosted, (
        "第 1 轮不应提权：初始访问计数 0 只表示『从未被访问』，不是『已演练 10 次』；"
        "实际提权轮次=%s" % boosted)
    assert boosted and boosted[0] == 10, (
        "首次提权应恰好落在第 10 次演练（此时访问计数=10）；实际提权轮次=%s" % boosted)
