# -*- coding: utf-8 -*-
"""批一百二十二守卫：#2638 三个公开入口类型闸（同族批量补全）。

覆盖
--------------------------------------------------------------------------
批量扫描 `SpacetimeMemoryEngine` 176 个公开入口的 None 入参健壮性，命中
三条裸异常泄漏（同 #1396 类型闸族纪律）：

  `ingest_frame(None)`                  → `AttributeError: 'NoneType' object has no attribute 'get'`
  `record_action_sequence(None,"o",True)` → `TypeError: 'NoneType' object is not subscriptable`
  `what_happened_at(None)`              → `TypeError: unsupported operand type(s) for -: 'NoneType' and 'float'`

    G1 行为级：ingest_frame 非 dict ⇒ 可读 TypeError
    G2 行为级：record_action_sequence actions 非 list/tuple / 元素非 str /
              outcome 非 str ⇒ 可读 TypeError
    G3 行为级：what_happened_at timestamp/tolerance 非数值 ⇒ 可读 TypeError
              （bool 亦拒——`True` 是 int 子类，非时间戳语义）
    G4 正对照：三条合法路径照常工作（ingest_frame({}) / record_action_sequence 合法 /
              what_happened_at(0.0)）
    G5 源码级：三处闸均在位

变异判别（撤三闸 ⇒ G1/G2/G3/G5 红）：
    python -X utf8 tests/test_batch122_entry_type_gates_guard.py --mutate

运行（lingshu 仓根）：
    python -X utf8 -m pytest tests/test_batch122_entry_type_gates_guard.py -q
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

from lingshu.core.core import SpacetimeMemoryEngine  # noqa: E402

_CORE = os.path.join(_ROOT, "lingshu", "core", "core.py")


def _engine():
    return SpacetimeMemoryEngine(os.path.join(tempfile.mkdtemp(), "g.db"))


def _expect_typeerror(fn, name, *needles):
    try:
        fn()
    except TypeError as ex:
        msg = str(ex)
        for needle in needles:
            assert needle in msg, "%s 错误消息缺 %r: %s" % (name, needle, msg)
        return
    raise AssertionError("%s 未抛 TypeError" % name)


def test_g1_ingest_frame_gate():
    """G1：ingest_frame 非 dict ⇒ 可读 TypeError。"""
    e = _engine()
    for bad in (None, 123, ["x"], "s"):
        _expect_typeerror(lambda b=bad: e.ingest_frame(b), "ingest_frame(%r)" % (bad,),
                          "ingest_frame", "frame_data", "必须是 dict")


def test_g2_record_action_sequence_gate():
    """G2：record_action_sequence 三处闸（actions 容器 / 元素 / outcome）。"""
    e = _engine()
    for bad in (None, 123, "abc"):
        _expect_typeerror(lambda b=bad: e.record_action_sequence(b, "o", True),
                          "record_action_sequence(actions=%r)" % (bad,),
                          "record_action_sequence", "actions")
    _expect_typeerror(lambda: e.record_action_sequence([None, "a"], "o", True),
                      "record_action_sequence(元素非 str)", "每项必须是字符串")
    _expect_typeerror(lambda: e.record_action_sequence(["a"], None, True),
                      "record_action_sequence(outcome=None)", "outcome")


def test_g3_what_happened_at_gate():
    """G3：what_happened_at timestamp/tolerance 非数值 ⇒ 可读 TypeError（bool 亦拒）。"""
    e = _engine()
    for bad in (None, "abc", True, ["x"]):
        _expect_typeerror(lambda b=bad: e.what_happened_at(b),
                          "what_happened_at(%r)" % (bad,), "timestamp", "数值")
    _expect_typeerror(lambda: e.what_happened_at(0.0, None),
                      "what_happened_at(tolerance=None)", "tolerance", "数值")
    _expect_typeerror(lambda: e.what_happened_at(0.0, True),
                      "what_happened_at(tolerance=True)", "tolerance")


def test_g4_positive_paths_intact():
    """G4：三条合法路径照常工作。"""
    e = _engine()
    node = e.ingest_frame({})
    assert node is not None, "ingest_frame({}) 应返回节点"
    got = e.what_happened_at(0.0)
    assert isinstance(got, list), "what_happened_at(0.0) 应返回 list"
    # record_action_sequence 合法路径（不强制成功：无 TMS/技能组件时允许 None）
    e.record_action_sequence(["open", "save"], "ok", True)


def test_g5_source_anchors():
    """G5：三处闸均在位。"""
    with open(_CORE, encoding="utf-8") as f:
        src = f.read()
    checks = [
        ("def ingest_frame", 'isinstance(frame_data, dict)'),
        ("def record_action_sequence", 'isinstance(actions, (list, tuple))'),
        ("def what_happened_at", 'isinstance(timestamp, (int, float))'),
    ]
    for anchor, gate in checks:
        idx = src.find(anchor)
        assert idx >= 0, "找不到 %s" % anchor
        body = src[idx:idx + 6000]
        assert gate in body, "%s 缺闸 %s" % (anchor, gate)


def _mutate():
    """撤三闸 ⇒ 子进程复跑必须变红。"""
    gates = [
        '''        if not isinstance(frame_data, dict):
            raise TypeError("ingest_frame 的 frame_data 必须是 dict，收到 %r"
                            % (type(frame_data).__name__,))
''',
        '''        if not isinstance(actions, (list, tuple)):
            raise TypeError("record_action_sequence 的 actions 必须是 list/tuple，收到 %r"
                            % (type(actions).__name__,))
        if not all(isinstance(a, str) for a in actions):
            raise TypeError("record_action_sequence 的 actions 每项必须是字符串，收到 %r"
                            % ([type(a).__name__ for a in actions][:3],))
        if not isinstance(outcome, str):
            raise TypeError("record_action_sequence 的 outcome 必须是字符串，收到 %r"
                            % (type(outcome).__name__,))
''',
        '''        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
            raise TypeError("what_happened_at 的 timestamp 必须是数值，收到 %r"
                            % (type(timestamp).__name__,))
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
            raise TypeError("what_happened_at 的 tolerance 必须是数值，收到 %r"
                            % (type(tolerance).__name__,))
''',
    ]
    with open(_CORE, encoding="utf-8") as f:
        src = f.read()
    for i, g in enumerate(gates, 1):
        if src.count(g) != 1:
            print("[ANCHOR-MISS] M%d  命中 %d 处" % (i, src.count(g)))
            return 1
    try:
        stripped = src
        for g in gates:
            stripped = stripped.replace(g, "")
        with open(_CORE, "w", encoding="utf-8", newline="") as f:
            f.write(stripped)
        proc = subprocess.run(
            [sys.executable, "-X", "utf8", "-m", "pytest", os.path.abspath(__file__), "-q"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=_ROOT)
        red = proc.returncode != 0
        out = (proc.stdout or "") + (proc.stderr or "")
    finally:
        with open(_CORE, "w", encoding="utf-8", newline="") as f:
            f.write(src)
    if red:
        arms = [a for a in ("test_g1", "test_g2", "test_g3", "test_g5") if a in out]
        print("[MUTATE-OK] M1-M3  撤闸后变红：%s" % (",".join(arms) or "（无具名臂）"))
        return 0
    print("[MUTATE-FAIL] 撤闸后守卫仍绿——守卫无牙")
    return 1


if __name__ == "__main__":
    if "--mutate" in sys.argv:
        sys.exit(_mutate())
    import pytest
    sys.exit(pytest.main([os.path.abspath(__file__), "-q"]))
