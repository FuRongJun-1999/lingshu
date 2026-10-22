# -*- coding: utf-8 -*-
"""守卫 · acquire(history=...) 必须在 state() 读数面可见、可保真、不可遗忘（lingshu #859）

缺陷（锚定提交 f6b85abfac16ff894a4a620da9ece67afdfc3b1f，`lingshu/world/stable_lease.py:58-71`）
    **定位依据**：本缺陷的定位结论（`stable_lease.py:58` 附近 + 「函数体零引用 history」）
    见 dsh-bounty #1480（`evt_e5a4e8c3283a`）；本文件仅据此编写守卫，**不主张发现与定位**。

    `acquire(self, key, ttl=None, confidence=1.0, history=None)` 的签名与 docstring 声明
    `history` 是「验证历史记录（不可遗忘，仅记录不删除）」，但**函数体零引用** `history`
    （`grep -n history` 全文件只命中 59/60 两行，即签名与 docstring）；`:63-69` 整条替换
    `self._leases[key]`，新字典不含 history；`:70` 注释自述「历史由调用方持有，这里仅标记」
    —— **连标记也没有落**。

    修前实测（本沙箱，冻结时钟，见 evidence/repro-before-*.log、baseline-keysets-*.json）：
        带 history 与不带 history 的 `acquire()` 返回**逐字段相同**（8 键，无 history）；
        AST 扫描 acquire 函数体：引用名字里没有 'history'。
    对照面（证明不是「统一不存」）：同函数 `ttl` / `confidence` 真实落库并由 `state()` 报出。

读数面（本文件的唯一判据面）
    **只认 `state(key)`**。`check()` / `degrade()` 的返回体是各自定制的 dict（degrade 带
    reason/note），往里塞 history 属改契约面 ⇒ 本件不要求、也不允许；只断言它们的**键集不变**。
    `acquire()` 的返回值就是 `state(key)`，故自动包含。

冻结时钟
    源码只有三处 `time.time()`（acquire / renew / _stable_now），故对**模块属性**打桩：
        monkeypatch.setattr(slmod, "time", types.SimpleNamespace(time=lambda: T))
    不改 time 模块本身、不 sleep；「过期」靠把 T 往后拨。

判据来源
    · 契约面：模块头 P1-003「验证历史记录仍属 3.2 节不可遗忘记录范畴，降级 ≠ 删除」
      + `acquire` docstring「不可遗忘，仅记录不删除」。
    · 语义（v2 §C）：`history is None` ⇒ 不改动既有 history（新 key 则无该键）；
      `history` 非 None（含 `[]`，`[]` 为 no-op 追加）⇒ **追加**。
      依据＝「不可遗忘 / 仅记录不删除」+ 不改变未传参调用者的现状。
      issue 正文「建议处置·兑现契约」原文：「`acquire` 把 `history` 并入 `self._leases[key]`
      （只增不删，P1-003 同口径），`state()` 透出」——与上面两条一致，已抄进 changes.md。

运行（仓根）：python -X utf8 -m pytest tests/test_issue859_stable_lease_history.py -q --no-header
"""
import types
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lingshu.world.stable_lease as slmod  # noqa: E402
from lingshu.world.stable_lease import StableLease  # noqa: E402

#: 冻结时刻（金标准采集用的同一 T，见 evidence/baseline-keysets-*.json 的 "frozen_T"）
T = 1_700_000_000.0

#: 修复前的 `state()` 键集（8 键；锚定提交实测，见 baseline-keysets-*.json）
BASE = ["confidence", "expires_at", "key", "state", "t_verified",
        "ttl", "ttl_expires_at", "weak_at"]
#: 修复前的 `check()` 键集（在租约内分支）
BASE_CHECK = ["age", "confidence", "decayed_confidence", "in_lease", "key",
              "remaining", "state"]
#: `degrade()` 返回体键集
DEGRADE_KEYS = ["key", "note", "reason", "state"]
#: 未知 key 的 `state()` / `check()` 键集
UNKNOWN_STATE = ["key", "state"]
UNKNOWN_CHECK = ["in_lease", "key", "reason", "state"]

#: 修复前 `acquire("k")` 在冻结时钟下的**逐字段**读数（S2 修前快照字面量）
BASE_STATE_LITERAL = {
    "confidence": 1.0, "expires_at": 1700002407.9, "key": "k", "state": "stable",
    "t_verified": 1700000000.0, "ttl": 3600.0, "ttl_expires_at": 1700003600.0,
    "weak_at": 1700002407.9,
}


@pytest.fixture
def clock(monkeypatch):
    """冻结时钟：返回一个可拨表的 setter。"""
    def set_t(t):
        monkeypatch.setattr(slmod, "time", types.SimpleNamespace(time=lambda: t))
    set_t(T)
    return set_t


def _lease(clock):
    return StableLease()


# ------------------------------------------------------------------ ① 可见且保真

def test_859_history_visible_via_state(clock):
    """① `state(key)` 必须读到 history：逐条等值 + 保序 + **保重复**。

    断言走公开读数面 `state(key)`（不是内部 dict）；用 `[1, 1, 2]` 同时验保序与保重复——
    用 set / sorted / len 之类比较都会漏掉「去重」或「重排」这类半修。
    """
    h = [1, 1, 2]
    lease = _lease(clock)
    lease.acquire("k", history=h)

    got = lease.state("k")["history"]
    assert got == h, "逐条不等值/丢序/丢重复：%r" % (got,)
    assert list(got) == [1, 1, 2]

    # 结构体条目也逐字段保真
    h2 = [{"round": 1, "by": "A"}, {"round": 2, "by": "B"}]
    lease2 = _lease(clock)
    lease2.acquire("k", history=h2)
    assert lease2.state("k")["history"] == h2


# ------------------------------------------- ② None / [] / 未知 key / check / degrade

def test_859_none_and_empty_unchanged(clock):
    """② `history=None` 与 `history=[]`（新 key）读数面**与修复前逐格相等**。

    · `acquire()` 与 `acquire(history=None)`：state() 键集 == BASE，且**无** history 键；
    · `acquire(history=[])`：同上（`[]` 是 no-op 追加，不得凭空长出空列表键）；
    · 未知 key：state() == {key, state}；check() == {in_lease, key, reason, state}；
    · `check()` 键集 == BASE_CHECK；`degrade()` 键集 == {key, state, reason, note}
      （本件**不**往这两个定制 dict 里塞 history）。
    """
    for kwargs in ({}, {"history": None}, {"history": []}):
        lease = _lease(clock)
        lease.acquire("k", **kwargs)
        st = lease.state("k")
        assert sorted(st) == BASE, "%r 的 state() 键集变了：%s" % (kwargs, sorted(st))
        assert "history" not in st, "%r 不该长出 history 键：%r" % (kwargs, st)

    empty = _lease(clock)
    assert sorted(empty.state("nope")) == UNKNOWN_STATE
    assert sorted(empty.check("nope")) == UNKNOWN_CHECK

    lease = _lease(clock)
    lease.acquire("k")
    assert sorted(lease.check("k")) == BASE_CHECK, sorted(lease.check("k"))
    assert sorted(lease.degrade("k")) == DEGRADE_KEYS, sorted(lease.degrade("k"))


# ------------------------------------------------------------ ③ 五个转换后不可遗忘

def test_859_survives_transitions(clock):
    """③ renew / degrade / check / renew(evidence=True) / **过期后 check 落 weak** 五格，
    每格经 `state()` 读到的 history 与传入**逐条相同**。

    过期格专抓「check() 的 weak 分支重建 dict 时丢 history」——那条路径不经过 renew，
    只测 renew 是测不到的。
    """
    h = [{"round": 1}, {"round": 2}]
    lease = _lease(clock)
    lease.acquire("k", ttl=10.0, history=h)

    def expect(where):
        got = lease.state("k")["history"]
        assert got == h, "%s 之后 history 变了：%r" % (where, got)

    lease.check("k")
    expect("check()")
    lease.renew("k")
    expect("renew()")
    lease.degrade("k", reason="冲突")
    assert lease.state("k")["state"] == "weak"
    expect("degrade()")
    lease.renew("k", evidence=True)
    assert lease.state("k")["state"] == "stable"
    expect("renew(evidence=True)")

    clock(T + 100.0)                       # 拨表越过 TTL=10（不 sleep）
    assert lease.check("k")["state"] == "weak"
    assert lease.state("k")["state"] == "weak"
    expect("过期后 check 落 weak")


# ------------------------------------------------------------------ ④ 防御性拷贝

def test_859_defensive_copy(clock):
    """④ 三个方向都不得泄漏：改调用方的 list、改 `state()` 读出来的 list、改**嵌套对象**。

    浅拷贝（`list(history)`）能挡住第一层的 append，但挡不住 `h[0]["t"] = 9`；
    故这里必须同时验嵌套。另断言两次 `state()` 返回的不是同一个对象。
    """
    # 方向一：改调用方自己的 list（追加 + 改嵌套）
    mine = [{"t": 1}]
    lease = _lease(clock)
    lease.acquire("k", history=mine)
    mine.append({"t": 2})
    mine[0]["t"] = 99
    assert lease.state("k")["history"] == [{"t": 1}], "调用方改入参污染了库内"

    # 方向二：改 state() 读出来的 list（追加 + 改嵌套）
    leaked = lease.state("k")["history"]
    leaked.append({"t": 3})
    leaked[0]["t"] = 77
    assert lease.state("k")["history"] == [{"t": 1}], "改 state() 返回值污染了库内"

    # 方向三：两次 state() 返回的 list 与条目都不是同一对象
    a = lease.state("k")["history"]
    b = lease.state("k")["history"]
    assert a is not b, "两次 state() 返回了同一个 list 对象"
    assert a[0] is not b[0], "两次 state() 返回了同一个条目对象（只拷了外层）"


# ------------------------------------------------------------- ⑤ 二次 acquire 语义

def test_859_reacquire_semantics(clock):
    """⑤ `history is None` 不改动既有记录；非 None（含 `[]`）追加（`[]` 为 no-op）。

    · `acquire(k, [1,2])` → `acquire(k, history=None)`：history 仍在且仍是 [1,2]；
    · 再 `acquire(k, [3])`：追加为 [1,2,3]（**不覆盖**）；
    · `acquire(k, [])`：no-op，仍是 [1,2,3]。
    """
    lease = _lease(clock)
    lease.acquire("k", history=[1, 2])
    assert lease.state("k")["history"] == [1, 2]

    lease.acquire("k", history=None)
    assert lease.state("k")["history"] == [1, 2], "history=None 把既有记录清掉了"

    lease.acquire("k")
    assert lease.state("k")["history"] == [1, 2], "缺省 history 把既有记录清掉了"

    lease.acquire("k", history=[3])
    assert lease.state("k")["history"] == [1, 2, 3], "二次 acquire 未追加（或被覆盖）"

    lease.acquire("k", history=[])
    assert lease.state("k")["history"] == [1, 2, 3], "history=[] 不该是清空"

    # 「只增不删」必须机械可判：**不得去重**（去重 = 变相删除）、不得重排
    dedup = _lease(clock)
    dedup.acquire("k", history=[1, 2])
    dedup.acquire("k", history=[1])
    assert dedup.state("k")["history"] == [1, 2, 1],         "重复条目被去重（=变相删除）：%r" % (dedup.state("k")["history"],)

    dup = _lease(clock)
    dup.acquire("k", history=[1, 1, 2])
    dup.acquire("k", history=[2, 1])
    assert dup.state("k")["history"] == [1, 1, 2, 2, 1],         "重复/顺序未被原样保留：%r" % (dup.state("k")["history"],)

    # None 之后：条数与顺序逐条相等（不是「集合相等」）
    before = dedup.state("k")["history"]
    dedup.acquire("k", history=None)
    after = dedup.state("k")["history"]
    assert after == before and len(after) == len(before),         "history=None 后条数/顺序变了：%r -> %r" % (before, after)


# ------------------------------------------------- ⑥ 字段差集恰为 {"history"}

def test_859_delta_is_only_history(clock):
    """⑥ 冻结时钟下，带 history 与不带的 `state()` **字段差集恰为 {"history"}**。

    防「顺手多改」：除了多出 history，其余键的键名与值必须逐一相等。
    """
    a = _lease(clock).acquire("k1")
    b = _lease(clock).acquire("k2", history=[1])

    only_a = set(a) - set(b)
    only_b = set(b) - set(a)
    assert only_b == {"history"}, "多出的键不止 history：%s" % sorted(only_b)
    assert only_a == set(), "带 history 反而少了键：%s" % sorted(only_a)

    shared = set(a) & set(b)
    assert shared == set(BASE), sorted(shared)
    for k in sorted(shared):
        if k == "key":
            continue                        # key 值按设计不同（k1 / k2）
        assert a[k] == b[k], "字段 %s 被顺带改了：%r != %r" % (k, a[k], b[k])


# -------------------------------------------------- ⑦ 与修前快照逐字段相等

def test_859_none_matches_baseline(clock):
    """⑦ `history=None` 的 `state()` 与 **S2 修前快照逐字段相等**（字面量写进测试）。

    字面量取自锚定提交、冻结 T 下的实测（evidence/baseline-keysets-*.json）；本测试用同一 T。
    """
    lease = _lease(clock)
    lease.acquire("k")
    assert lease.state("k") == BASE_STATE_LITERAL, lease.state("k")
    assert lease.acquire("k2") == dict(BASE_STATE_LITERAL, key="k2")


# ------------------------------------------------- ⑧ 保重复保序（拆解员 2026-10-22 追加）

def test_history_keeps_duplicates_and_order(clock):
    """⑧ `history` 是**记录**不是**集合**：不得去重、不得排序（docstring「仅记录不删除」）。

    这条专防「用 `set(...)` / 去重式写法糊过去」的半修——那种实现能让 ① 的
    `[1,1,2]` 之外的用例全绿，却把「重复的验证轮次」静默吃掉。
    """
    lease = _lease(clock)
    lease.acquire("dup", history=[1, 1, 2])
    assert lease.state("dup")["history"] == [1, 1, 2], \
        "去重或排序了：%r" % (lease.state("dup")["history"],)

    # 顺序敏感：倒序输入必须原样保留（不许被排序成升序）
    lease2 = _lease(clock)
    lease2.acquire("dup", history=[3, 1, 2])
    assert lease2.state("dup")["history"] == [3, 1, 2], \
        "被排序了：%r" % (lease2.state("dup")["history"],)


# ------------------------------------------- ⑨ 读数面只认 state()：check/degrade 不加宽

def test_check_and_degrade_read_surfaces_unchanged(clock):
    """⑨ `check()` / `degrade()` 的返回体**不得**因 history 而加宽（读数面只认 `state()`）。

    键集写成**字面量**（不引用常量），确保「有人顺手往这两个定制 dict 里塞 history」时当场红。
    """
    key = "read-surface"
    lease = _lease(clock)
    lease.acquire(key, history=[{"round": 1}])

    assert sorted(lease.check(key)) == [
        "age", "confidence", "decayed_confidence", "in_lease",
        "key", "remaining", "state",
    ], "check() 返回体被加宽了：%s" % sorted(lease.check(key))

    assert sorted(lease.degrade(key)) == ["key", "note", "reason", "state"], \
        "degrade() 返回体被加宽了：%s" % sorted(lease.degrade(key))

    assert sorted(lease.state("never-seen")) == ["key", "state"], \
        "未知 key 分支被加宽了：%s" % sorted(lease.state("never-seen"))
