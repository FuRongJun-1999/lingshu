# -*- coding: utf-8 -*-
"""TMS 真值维护（JTMS 式依赖撤回）的机械化守卫 —— 把 `lingshu/core/tms.py` 的声明钉成判据。

为什么需要
----------
`tms.py`（#447 引入，默认关闭、经 `SpacetimeMemoryEngine.enable_tms()` 装配）在 docstring
与实现里写了四条**可判**的声明，但此前**没有任何机器看着它们**
（`tools/coverage_map_lint.py` 把它报为「零引用模块」）：

  ① `belief(c) = own(c)`（c 无依据，或 own 状态为「已佐证」）；
     否则 `belief(c) = min(own(c), jsup(c))`，其中 `jsup(c) = max_j min_{p∈j} belief(p)`
     ⇒ **TMS 只撤回、不放大**（"推导不是证据"：前提再可靠也不替结论增信）。
  ② 传播：沿**下游闭包**迭代至**不动点**（`len(nodes)+1` 轮上限），**环可处理**；
     `own` 取自**证据账本**（`ledger.assess`），闭包**外**的前提取 `nodes.confidence`。
  ③ `undermined` 标记：前提被推翻时点名列出。
  ④ 输入校验：空前提 / 结论自身当前提 / 节点不存在 ⇒ 一律拒绝。

本文件把 ①–④ 钉住，并为正向判据配**定点变异**证明判据**真的会红**
（没有"能红"演示的守卫按本仓口径算**假绿**）。

⚠ 测试口径（踩过）：`propagate` 的 `own(n)` 来自 `ledger.assess(n)`，**不看**
`nodes.confidence`；后者只影响**闭包外**前提的取值（`_external_belief`）。
所以本文件用 `_own()` 把 `ledger.assess` 包一层来**精确控制 own**（真实 store/TMS 代码不变），
而前提值用 SQL 写 `nodes.confidence` —— 这正是被测的两条取数路径。

纯标准库 + 真实引擎（临时库），不依赖 numpy / PIL / Blender。
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import warnings

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.core.core import SpacetimeMemoryEngine          # noqa: E402
from lingshu.core import tms as TMS                           # noqa: E402
from lingshu.core.evidence import REFUTED_BELOW               # noqa: E402

_KEY = "k-tms-guard"


@pytest.fixture()
def engine():
    os.environ["AEIS_DESIGNER_KEY"] = _KEY
    d = tempfile.mkdtemp()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        e = SpacetimeMemoryEngine(os.path.join(d, "tms.db"))
    try:
        yield e
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _node(engine, text):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return engine.add_perception(text, importance=0.6).id


def _set_conf(engine, nid, value):
    """写 `nodes.confidence` —— 闭包**外**前提的取值路径。"""
    engine.store.conn.execute("UPDATE nodes SET confidence=? WHERE id=?", (value, nid))
    engine.store.conn.commit()


def _conf(engine, nid):
    return float(engine.store.get_node(nid).confidence)


def _own(tms, values):
    """把 `ledger.assess` 包一层 ⇒ 精确控制 `propagate` 用的 `own(n)`。

    只替换 `confidence` 一项，`status` 等仍走真实账本（TMS 靠 `status=="corroborated"`
    决定是否豁免前提撤回 —— 那条分支也要保持真实）。
    """
    real = tms.ledger.assess

    def assess(nid):
        d = dict(real(nid))
        if nid in values:
            d["confidence"] = values[nid]
        return d

    tms.ledger.assess = assess


# ═══════════════════════════════════════════ ① 只撤回、不放大
def test_t1_belief_never_exceeds_own_confidence(engine):
    """前提**高于**结论时，结论**不得被抬高**：`belief = min(own, jsup)` 的上界侧。"""
    tms = engine.enable_tms()
    a, b, c = _node(engine, "前提A"), _node(engine, "前提B"), _node(engine, "结论C")
    tms.add_justification(c, [a, b])
    _set_conf(engine, a, 0.9)          # 闭包外 ⇒ 直接是 jsup 的输入
    _set_conf(engine, b, 0.9)
    _own(tms, {c: 0.1})                # own(c) = 0.1
    got = tms.propagate([c])
    assert abs(got[c] - 0.1) < 1e-9, (
        "own=0.1 而前提 0.9/0.9 ⇒ 结论应保持 0.1（不得被抬高），实测 %r" % got.get(c))


def test_t1b_belief_is_pulled_down_to_weakest_premise(engine):
    """前提**低于**结论时，结论被**撤回**到 `jsup = min(前提)`，并写回库。"""
    tms = engine.enable_tms()
    a, b, c = _node(engine, "前提A"), _node(engine, "前提B"), _node(engine, "结论C")
    tms.add_justification(c, [a, b])
    _set_conf(engine, a, 0.1)
    _set_conf(engine, b, 0.2)
    _own(tms, {c: 0.9})
    got = tms.propagate([c])
    assert abs(got[c] - 0.1) < 1e-9, "合取应取最弱前提 0.1，实测 %r" % got.get(c)
    assert abs(_conf(engine, c) - 0.1) < 1e-6, "belief 未写回 nodes.confidence：%r" % _conf(engine, c)


# ═══════════════════════════════════════════ ② 析取 max / 合取 min
def test_t2_disjunction_takes_max_over_justifications(engine):
    """多条依据（析取）⇒ `jsup = max_j min_{p∈j}`：取**最强的那条依据**。"""
    tms = engine.enable_tms()
    a, b, c, d = (_node(engine, "a"), _node(engine, "b"),
                  _node(engine, "结论c"), _node(engine, "d"))
    tms.add_justification(c, [a, b])          # min(0.1, 0.9) = 0.1
    tms.add_justification(c, [d])             # min(0.5)      = 0.5  ⇒ jsup = 0.5
    _set_conf(engine, a, 0.1)
    _set_conf(engine, b, 0.9)
    _set_conf(engine, d, 0.5)
    _own(tms, {c: 1.0})
    got = tms.propagate([c])
    assert abs(got[c] - 0.5) < 1e-9, "析取应取 max(0.1, 0.5)=0.5，实测 %r" % got.get(c)


# ═══════════════════════════════════════════ ③ 闭包与不动点（含环）
def test_t3_propagation_is_idempotent_on_cycle(engine):
    """带**环**的图：二次传播必须幂等（已到不动点），且值域有限。"""
    tms = engine.enable_tms()
    x, y, z = _node(engine, "X"), _node(engine, "Y"), _node(engine, "Z")
    tms.add_justification(y, [x])             # Y ← X
    tms.add_justification(x, [y])             # X ← Y（2-环）
    tms.add_justification(z, [x])             # Z ← X
    _own(tms, {x: 0.8, y: 0.6, z: 0.7})
    first = dict(tms.propagate([x]))
    second = dict(tms.propagate([x]))
    assert set(first) == set(second), "闭包不稳定：%r vs %r" % (sorted(first), sorted(second))
    for n, v in second.items():
        assert 0.0 <= v <= 1.0, "环上出现越界信念：%s=%r" % (n, v)
    for n in first:
        assert abs(second[n] - first[n]) < 1e-9, "未到不动点：%s %r→%r" % (n, first[n], second[n])
    assert len(first) >= 3, "闭包应含 X/Y/Z 三个节点，实测 %r" % sorted(first)


# ═══════════════════════════════════════════ ④ undermined 标记 + 输入校验
def test_t4_undermined_lists_refuting_premises(engine):
    tms = engine.enable_tms()
    a, b, c = _node(engine, "前提A"), _node(engine, "前提B"), _node(engine, "结论C")
    tms.add_justification(c, [a, b])
    _set_conf(engine, a, 0.1)
    _set_conf(engine, b, 0.2)
    tms.propagate([c])
    ev = tms.assess(c)
    assert ev["status"] == "undermined", "前提已被推翻但状态未标：%r" % ev.get("status")
    assert sorted(ev["undermined_by"]) == sorted([a, b]), (
        "undermined_by 应列出推翻它的前提（<%r）；实测 %r" % (REFUTED_BELOW, ev.get("undermined_by")))


def test_t5_input_validation_rejects_bad_justifications(engine):
    tms = engine.enable_tms()
    a, c = _node(engine, "前提A"), _node(engine, "结论C")
    with pytest.raises(ValueError):
        tms.add_justification(c, [])                        # 空前提
    with pytest.raises(ValueError):
        tms.add_justification(c, [c])                       # 结论当前提
    with pytest.raises(KeyError):
        tms.add_justification(c, ["node_does_not_exist"])   # 不存在节点
    assert tms.justifications_of(c) == [], "被拒的依据不得残留"
    assert tms.add_justification(c, [a])                    # 合法路径仍可用


# ═══════════════════════════════════════════ 定点变异：证明上面真的会红
def test_t6_mutation_amplify_turns_t1_red(engine, monkeypatch):
    """把 `_solve_beliefs` 改成**会放大**（`max` 而非 `min`）⇒ T1 的不放大判据必须转红。"""
    tms = engine.enable_tms()
    a, b, c = _node(engine, "前提A"), _node(engine, "前提B"), _node(engine, "结论C")
    tms.add_justification(c, [a, b])
    _set_conf(engine, a, 0.9)
    _set_conf(engine, b, 0.9)
    _own(tms, {c: 0.1})

    def amplified(self, nodes, own):
        belief = {n: own[n]["confidence"] for n in nodes}
        for n in nodes:
            o = own[n]
            js = self._just_cache.get(n)
            if not js or o["status"] == "corroborated":
                continue

            def bp(p):
                return belief[p] if p in belief else self._external_belief(p)
            jsup = max(min(bp(p) for p in j) for j in js)
            belief[n] = max(o["confidence"], jsup)        # ← 变异：放大
        return belief

    monkeypatch.setattr(TMS.TruthMaintenance, "_solve_beliefs", amplified)
    got = tms.propagate([c])
    assert got[c] > 0.1 + 1e-9, (
        "变异后结论未被抬高 ⇒ T1 没有判别力（假绿）：%r" % got.get(c))


def test_t7_mutation_root_only_closure_turns_t3_red(engine, monkeypatch):
    """把闭包砍成**只有根** ⇒ 下游节点不再被重算 ⇒ T3 的闭包/幂等判据必须转红。"""
    tms = engine.enable_tms()
    x, y, z = _node(engine, "X"), _node(engine, "Y"), _node(engine, "Z")
    tms.add_justification(y, [x])
    tms.add_justification(x, [y])
    tms.add_justification(z, [x])
    _own(tms, {x: 0.8, y: 0.6, z: 0.7})
    good = dict(tms.propagate([x]))
    assert len(good) >= 3, "前置：正常闭包应含 3 个节点"

    def root_only(self, roots):
        self._closure = lambda rs: list(dict.fromkeys(rs))       # ← 变异：闭包只剩根
        try:
            return orig_propagate(self, roots)
        finally:
            del self._closure

    orig_propagate = TMS.TruthMaintenance.propagate
    monkeypatch.setattr(TMS.TruthMaintenance, "propagate", root_only)
    bad = dict(tms.propagate([x]))
    assert len(bad) < len(good), (
        "闭包砍成只有根后传播面没变 ⇒ T3 没有判别力（假绿）：good=%d bad=%d"
        % (len(good), len(bad)))
