# -*- coding: utf-8 -*-
"""守卫 · 幻影旋钮的终裁记录必须如实标注「无读取点」（issue #500）

缺陷（锚定 50ee03b2，`lingshu/core/core.py` `adjudicate_verifier_standard`）：
    终裁通过时写结构层记录
        `deviation_threshold=<值> 终裁通过（<终裁人>）`
    —— 该串读作「此标准已生效」。但全仓对该参数**零读取点**（普查见 docstring 末），
    写入后行为完全不变（行为实验：应用 0.99 后 `_reflect_verify` 偏差判定、
    `check_escalation` 对 ESC-004 条件串、`_dedup_static` 逐项相同）。
    ⇒ 记录在撒谎：它声称生效，实际只登记。

本件只钉事实（**不让该参数真的去影响判据** —— 那属设计裁定）：
  A 无读取点参数的终裁记录必须带如实标注；
  B 防误伤：有读取点的参数（`dedup_static`）记录**不得**被加标注；
  C 钉住普查事实：`deviation_threshold` 在 `lingshu/` 内零「读取形态」引用（AST Load），
    且登记表与之一致 —— 若将来有人真接了读取点，本守卫会红，提示同步更新登记与标注。

判据来源：issue #500 正文（「结构层记录是否声称它已生效」）；
    普查口径 `git grep deviation_threshold` = 19 处 / 6 文件，无一为读取。

运行（仓根）：python -X utf8 -m pytest tests/test_issue500_phantom_knob_record_guard.py -q --no-header
"""
from __future__ import annotations

import ast
import os
import sys
import tempfile

# 不写死沙箱路径：优先环境变量，否则从 __file__ 上溯定位仓库根（本文件在 <repo>/tests/ 下）
_THIS = os.path.abspath(__file__)
_HERE = os.path.dirname(_THIS)
REPO = os.environ.get("LINGSHU_REPO") or os.path.dirname(_HERE)
sys.path.insert(0, REPO)

from lingshu.core.core import SpacetimeMemoryEngine  # noqa: E402

# 登记表在修复后才有。缺失时用哨兵而非让整份守卫在**导入期**崩掉——
# 这样「修复前」的红来自断言（真实缺陷），而不是 collection error（弱证据）。
try:
    from lingshu.core.core import _VERIFIER_PARAMS_WITH_READERS  # noqa: E402
except ImportError:  # pragma: no cover - 仅在修复前的源码上走到
    _VERIFIER_PARAMS_WITH_READERS = None

KEY = "GUARD-500-KEY"
PARAM_NO_READER = "deviation_threshold"
PARAM_WITH_READER = "dedup_static"


def _db(name):
    return os.path.join(tempfile.mkdtemp(prefix="lingshu500_"), name + ".db")


def _approve(e, param, value):
    vid = e.store.add_verifier_standard("标准", param, value, "理由", "提议")
    assert e.store.review_verifier_standard(vid, "复核", True) is True
    assert e.store.cs_review_verifier_standard(vid, "空间", True) is True
    assert e.adjudicate_verifier_standard(vid, "终裁", True, designer_key=KEY) is True
    return vid


def _structure_record_for(e, param):
    """取最近一条提到该参数的结构层记录正文。

    记录经 `_record_structure_event` 写成 layer=STRUCTURE 的记忆节点（不可遗忘），
    此处按 engines 自己的读面取回；取不到即断言失败（不允许静默通过）。
    """
    rows = e.store.conn.execute(
        "SELECT content FROM nodes WHERE content LIKE ? ORDER BY rowid DESC LIMIT 5",
        (f"%{param}%",),
    ).fetchall()
    texts = [r[0] for r in rows if r[0]]
    assert texts, (
        "未取到含 %r 的结构层记录 —— 守卫的读取面失效，不能据此判通过" % (param,))
    return texts[0]


# ---------------- A 无读取点参数：记录必须如实标注 ----------------

def test_a_no_reader_param_record_is_annotated(monkeypatch):
    """A `deviation_threshold` 终裁记录必须标出「无读取点」，不得读作已生效。"""
    monkeypatch.setenv("AEIS_DESIGNER_KEY", KEY)
    e = SpacetimeMemoryEngine(_db("a"))
    try:
        _approve(e, PARAM_NO_READER, 0.42)
        text = _structure_record_for(e, PARAM_NO_READER)
        assert "终裁通过" in text, "记录形态变了，守卫前提失效：%r" % (text,)
        assert "无读取点" in text, (
            "A 记录未如实标注「无读取点」——它读起来像已生效，而该参数全仓零读取点：%r"
            % (text,))
    finally:
        e.close()


# ---------------- B 有读取点参数：不得误加标注 ----------------

def test_b_reader_param_record_not_annotated(monkeypatch):
    """B 防误伤：`dedup_static` 确有读取点，记录**不得**被加「无读取点」标注。"""
    monkeypatch.setenv("AEIS_DESIGNER_KEY", KEY)
    e = SpacetimeMemoryEngine(_db("b"))
    try:
        _approve(e, PARAM_WITH_READER, 0.6)
        text = _structure_record_for(e, PARAM_WITH_READER)
        assert "终裁通过" in text, "记录形态变了，守卫前提失效：%r" % (text,)
        assert "无读取点" not in text, (
            "B 误伤：`dedup_static` 有读取点（会改变 `_dedup_static`），记录不该标注无读取点：%r"
            % (text,))
    finally:
        e.close()


# ---------------- C 钉住普查事实（AST Load 为零） ----------------

def _load_names_for(path, name):
    """返回该文件里以「读取形态」出现的 name 的次数（AST Load 上下文）。

    读取形态 = 表达式里取值（Name/Attribute 的 Load），以及 dict.get(name) 这类
    以字符串取键的读取。写入（Store）、字符串字面量、注释都不计。
    """
    try:
        tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
    except SyntaxError:
        return 0
    hits = 0
    for node in ast.walk(tree):
        # Name / Attribute 的 Load 上下文
        if isinstance(node, ast.Name) and node.id == name:
            if isinstance(node.ctx, ast.Load):
                hits += 1
        elif isinstance(node, ast.Attribute) and node.attr == name:
            if isinstance(node.ctx, ast.Load):
                hits += 1
        # dict.get("name") / ["name"] 这类字符串取键的读取
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr == "get" and node.args:
                a0 = node.args[0]
                if isinstance(a0, ast.Constant) and a0.value == name:
                    hits += 1
    return hits


def test_c_census_zero_read_points_for_phantom_param():
    """C `deviation_threshold` 在 `lingshu/` 内必须零读取形态引用（钉住普查事实）。"""
    src_root = os.path.join(REPO, "lingshu")
    assert os.path.isdir(src_root), "定位不到 lingshu/ 源码根：%r" % (src_root,)

    offenders = []
    for dirpath, dirnames, filenames in os.walk(src_root):
        dirnames[:] = [d for d in dirnames if d not in ("__pycache__", ".git")]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            p = os.path.join(dirpath, fn)
            n = _load_names_for(p, PARAM_NO_READER)
            if n:
                offenders.append((os.path.relpath(p, REPO), n))

    assert not offenders, (
        "C `%s` 出现了读取形态引用 —— 普查事实已被推翻，须同步更新 "
        "`_VERIFIER_PARAMS_WITH_READERS` 与记录标注：%r" % (PARAM_NO_READER, offenders))

    assert _VERIFIER_PARAMS_WITH_READERS is not None, (
        "C 未找到读取点登记表 `_VERIFIER_PARAMS_WITH_READERS`："
        "无读取点参数的记录标注无从判定")
    assert PARAM_NO_READER not in _VERIFIER_PARAMS_WITH_READERS, (
        "C 登记表把无读取点参数列为「有读取点」：%r" % (_VERIFIER_PARAMS_WITH_READERS,))
    assert PARAM_WITH_READER in _VERIFIER_PARAMS_WITH_READERS, (
        "C 登记表漏了确有读取点的参数：%r" % (_VERIFIER_PARAMS_WITH_READERS,))