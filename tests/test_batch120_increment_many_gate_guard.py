# -*- coding: utf-8 -*-
"""批一百二十守卫：#2635 `increment_access_many` 批量入口类型闸。

覆盖
--------------------------------------------------------------------------
#1396 类型闸镜像给 `increment_access`（单点）补了非空 str 闸，但**它委派的
批量入口 `increment_access_many` 零闸**——三条静默失效（实测）：

  ① `increment_access_many([123])` 把 int 当 SQL 参数，UPDATE 命中 0 行、
     不抛，调用方以为刷新成功；
  ② `increment_access_many("ab")`（忘包 list）按**逐字符**迭代，把 `a`/`b`
     两个单字符 id 各 +1、目标 `ab` 纹丝不动——**静默写错实体**；
  ③ 混批里的非 str/空串被 `if i` 静默丢弃，调用方无从得知少记了几条。

    G1 行为级：批量含非 str 元素 ⇒ 可读 TypeError 且零写入
    G2 行为级：单个字符串（忘包 list）⇒ TypeError，且**不得**逐字符写错实体
    G3 行为级：混批含非法项 ⇒ TypeError（不再静默丢弃）
    G4 正对照：None/空序列保持静默；list/tuple 合法路径照常计数；单点闸不回归
    G5 源码级：闸位于 ids 过滤之前（容器 + 元素两层）

变异判别（撤入口闸 ⇒ G1/G2/G3 红）：
    python -X utf8 tests/test_batch120_increment_many_gate_guard.py --mutate

运行（lingshu 仓根）：
    python -X utf8 -m pytest tests/test_batch120_increment_many_gate_guard.py -q
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lingshu.core.core import (ConditionSpace, LayeredStore, MemoryLayer,  # noqa: E402
                               STNode)

_CORE = os.path.join(_ROOT, "lingshu", "core", "core.py")

_GATE_SRC = '''        if node_ids is None:
            return
        if not isinstance(node_ids, (list, tuple)):
            raise TypeError(
                "increment_access_many 的 node_ids 必须是 list/tuple（不是单个字符串——"
                "逐字符迭代会更新错误实体），收到 %r" % (type(node_ids).__name__,))
        bad = [i for i in node_ids if not isinstance(i, str) or not i.strip()]
        if bad:
            raise TypeError(
                "increment_access_many 的 node_ids 每项必须是非空字符串，收到非法项 %r"
                % (bad[:3],))
'''


def _store():
    return LayeredStore(os.path.join(tempfile.mkdtemp(), "g.db"))


def _mk(i):
    cs = ConditionSpace(observation_position=None, observation_tool=None,
                        time_window=(0.0, 0.0), existence_constraint=None)
    return STNode(id=i, content="C" + i, modality="text", spatial_coordinates={},
                  temporal_coordinate=0.0, condition_space=cs,
                  layer=MemoryLayer.KNOWLEDGE)


def test_g1_non_str_element_fails_loud():
    """G1：批量含非 str 元素 ⇒ 可读 TypeError（不是静默 0 行 UPDATE）。"""
    s = _store()
    s.add_node(_mk("n1"))
    try:
        s.increment_access_many([123])
    except TypeError as e:
        assert "node_ids" in str(e) and "非空字符串" in str(e), \
            "错误消息不指参数名: %s" % e
        assert s.get_node("n1").access_count == 0, "非法批次不应写任何节点"
        return
    raise AssertionError("increment_access_many([123]) 未抛 TypeError——静默吞掉")


def test_g2_bare_string_rejected():
    """G2：单个字符串（忘包 list）⇒ TypeError，且不得逐字符更新错实体。"""
    s = _store()
    for i in ("a", "b", "ab"):
        s.add_node(_mk(i))
    try:
        s.increment_access_many("ab")
    except TypeError as e:
        assert "list/tuple" in str(e), "错误消息未指明容器要求: %s" % e
        counts = [s.get_node(x).access_count for x in ("a", "b", "ab")]
        assert counts == [0, 0, 0], \
            "被拒调用仍写了计数: %r（逐字符迭代复发）" % (counts,)
        return
    raise AssertionError("increment_access_many('ab') 未抛 TypeError——逐字符写错实体")


def test_g3_mixed_batch_rejected():
    """G3：混批含非法项 ⇒ TypeError（不再静默丢弃）。"""
    s = _store()
    s.add_node(_mk("n1"))
    for bad in (["n1", None], ["n1", ""], ["n1", "  "], ["n1", 0]):
        try:
            s.increment_access_many(bad)
        except TypeError as e:
            assert "非法项" in str(e), "错误消息未列出非法项: %s" % e
        else:
            raise AssertionError("混批 %r 未抛 TypeError——静默少记" % (bad,))
    assert s.get_node("n1").access_count == 0, "被拒批次不应部分写入"


def test_g4_positive_paths_intact():
    """G4：正对照——None/空序列静默；合法 list/tuple 照常；单点闸不回归。"""
    s = _store()
    for i in ("n1", "n2", "n3"):
        s.add_node(_mk(i))
    s.increment_access_many(None)
    s.increment_access_many([])
    assert s.get_node("n1").access_count == 0, "None/[] 不应写计数"
    s.increment_access_many(["n1"])
    s.increment_access_many(("n2", "n3"))
    assert s.get_node("n1").access_count == 1, "list 路径失效"
    assert s.get_node("n2").access_count == 1, "tuple 路径失效"
    assert s.get_node("n3").access_count == 1, "tuple 路径失效"
    try:
        s.increment_access(123)
    except TypeError:
        pass
    else:
        raise AssertionError("increment_access(123) 未抛 TypeError——单点闸回归")


def test_g5_source_anchor():
    """G5：源码锚——闸在 ids 过滤之前、含容器与元素两层校验。"""
    with open(_CORE, encoding="utf-8") as f:
        src = f.read()
    idx = src.find("def increment_access_many")
    assert idx >= 0, "找不到 increment_access_many"
    body = src[idx:idx + 4000]
    assert "isinstance(node_ids, (list, tuple))" in body, "缺容器类型闸"
    assert "非空字符串" in body, "缺逐元素非空 str 闸"
    gate = body.find("isinstance(node_ids, (list, tuple))")
    filt = body.find("ids = [i for i in")
    assert 0 <= gate < filt, "闸未置于 ids 过滤之前"


def _run_plain():
    import pytest
    return pytest.main([os.path.abspath(__file__), "-q"])


def _mutate():
    """撤入口闸 ⇒ 子进程复跑必须变红（本进程已 import，改源码对它无效）。"""
    with open(_CORE, encoding="utf-8") as f:
        src = f.read()
    if src.count(_GATE_SRC) != 1:
        print("[ANCHOR-MISS] M1  命中 %d 处" % src.count(_GATE_SRC))
        return 1
    try:
        with open(_CORE, "w", encoding="utf-8", newline="") as f:
            f.write(src.replace(_GATE_SRC, ""))
        proc = subprocess.run(
            [sys.executable, "-X", "utf8", "-m", "pytest", os.path.abspath(__file__), "-q"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=_ROOT)
        red = proc.returncode != 0
        out = (proc.stdout or "") + (proc.stderr or "")
    finally:
        with open(_CORE, "w", encoding="utf-8", newline="") as f:
            f.write(src)
    if red:
        arms = [a for a in ("test_g1", "test_g2", "test_g3") if a in out]
        print("[MUTATE-OK] M1  撤闸后变红：%s" % (",".join(arms) or "（无具名臂）"))
        return 0
    print("[MUTATE-FAIL] M1  撤闸后守卫仍绿——守卫无牙")
    return 1


if __name__ == "__main__":
    if "--mutate" in sys.argv:
        sys.exit(_mutate())
    sys.exit(_run_plain())
