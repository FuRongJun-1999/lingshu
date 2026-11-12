# -*- coding: utf-8 -*-
"""批一百二十一守卫：#2637 `search_skills` 类型闸（`recall_skill` 公开入口直通面）。

覆盖
--------------------------------------------------------------------------
`recall_skill`（`SpacetimeMemoryEngine` 公开 API）原样转发到
`LayeredStore.search_skills`，而后者 `query.strip()` 裸调用：

  `recall_skill(None)`  → `AttributeError: 'NoneType' object has no attribute 'strip'`
  `recall_skill(123)`   → `AttributeError: 'int' object has no attribute 'strip'`
  `recall_skill(['x'])` → `AttributeError: 'list' object has no attribute 'strip'`

不可读且不指参数名。同文件 `search_content` 早有
`"search_content 的 query 必须是字符串"` 闸（#1396 类型闸族同款纪律），
本法零闸属同族漏网。

    G1 行为级：`recall_skill(None/123/list/bytes)` ⇒ 可读 TypeError 且指参数名
    G2 行为级：store 直调 `search_skills` 同闸（绕不开放映层）
    G3 正对照：空串/纯空白保持既有「列出前 100 张」语义（返 list，不报错）；
              合法查询照常返回
    G4 源码级：闸在 `q = query.strip()` 之前

变异判别（撤闸 ⇒ G1/G2 红）：
    python -X utf8 tests/test_batch121_search_skills_gate_guard.py --mutate

运行（lingshu 仓根）：
    python -X utf8 -m pytest tests/test_batch121_search_skills_gate_guard.py -q
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

from lingshu.core.core import LayeredStore, SpacetimeMemoryEngine  # noqa: E402

_CORE = os.path.join(_ROOT, "lingshu", "core", "core.py")

_GATE_SRC = '''        if not isinstance(query, str):
            raise TypeError("search_skills 的 query 必须是字符串，收到 %r"
                            % (type(query).__name__,))
'''


def _engine():
    return SpacetimeMemoryEngine(os.path.join(tempfile.mkdtemp(), "g.db"))


def test_g1_recall_skill_fails_loud():
    """G1：`recall_skill` 收到非 str ⇒ 可读 TypeError 且指参数名。"""
    e = _engine()
    for bad in (None, 123, ["x"], b"b", 3.5):
        try:
            e.recall_skill(bad)
        except TypeError as ex:
            msg = str(ex)
            assert "search_skills" in msg and "query" in msg, \
                "错误消息不指参数名: %s" % msg
            assert "必须是字符串" in msg, "错误消息未说明期望类型: %s" % msg
        else:
            raise AssertionError("recall_skill(%r) 未抛 TypeError——裸 AttributeError 泄漏" % (bad,))


def test_g2_store_direct_gate():
    """G2：store 直调同样受闸（绕不开放映层）。"""
    db = os.path.join(tempfile.mkdtemp(), "g.db")
    s = LayeredStore(db)
    for bad in (None, 123, ["x"]):
        try:
            s.search_skills(bad)
        except TypeError as ex:
            assert "必须是字符串" in str(ex), "store 层消息不对: %s" % ex
        else:
            raise AssertionError("search_skills(%r) 未抛 TypeError" % (bad,))


def test_g3_positive_paths_intact():
    """G3：空串/纯空白保持既有语义（返 list，不报错）；合法查询照常返回。"""
    e = _engine()
    empty = e.recall_skill("")
    blank = e.recall_skill("   ")
    assert isinstance(empty, list) and isinstance(blank, list), \
        "空串/纯空白应返回 list（既有「列出前 100 张」语义）"
    got = e.recall_skill("备份")
    assert isinstance(got, list), "合法查询路径失效"


def test_g4_source_anchor():
    """G4：源码锚——闸在 `q = query.strip()` 之前。"""
    with open(_CORE, encoding="utf-8") as f:
        src = f.read()
    idx = src.find("def search_skills")
    assert idx >= 0, "找不到 search_skills"
    body = src[idx:idx + 10000]
    gate = body.find('isinstance(query, str)')
    strip = body.find("q = query.strip()")
    assert gate >= 0, "缺 query 类型闸"
    assert strip >= 0, "找不到 q = query.strip()"
    assert gate < strip, "闸未置于 query.strip() 之前"


def _mutate():
    """撤闸 ⇒ 子进程复跑必须变红。"""
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
        arms = [a for a in ("test_g1", "test_g2", "test_g4") if a in out]
        print("[MUTATE-OK] M1  撤闸后变红：%s" % (",".join(arms) or "（无具名臂）"))
        return 0
    print("[MUTATE-FAIL] M1  撤闸后守卫仍绿——守卫无牙")
    return 1


if __name__ == "__main__":
    if "--mutate" in sys.argv:
        sys.exit(_mutate())
    import pytest
    sys.exit(pytest.main([os.path.abspath(__file__), "-q"]))
