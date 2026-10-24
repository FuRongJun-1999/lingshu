# -*- coding: utf-8 -*-
"""守卫 · _novelty 必须识别「同模板换 ASCII token」（issue #573）

缺陷（锚定 50ee03b2，`lingshu/core/longterm_gate.py::_novelty`）：
    判据是核心词新颖比例 `novel_grams / len(grams)`，阈值 NOVEL_TRIGGER=0.75。
    同模板只换一个 ASCII token 时，**新增 gram 与消失 gram 数量相等**，
    比例上相互抵消，且上限随句长下降 —— 实测 25 字句上限仅 6/25 = 0.2400。
    修前读数：端口/主机/账号/CVE/域名/IP 六族 **6/6 不触发**
    （0.2400 / 0.3158 / 0.1429 / 0.0588 / 0.3529 / 0.3077）。

修法（#213 §六·建议2 落地）：比例之外**并列一条模板判据**——
    把 ASCII token 抹成占位符得「句型模板」，同模板而 token 集合不同 ⇒ 内容确实变了 ⇒ 1.0；
    同模板同 token ⇒ 真重复 ⇒ 保持比例值；模板未命中 ⇒ 保持比例值。

五条：
  A 同模板换 token（6 族）必须全部 ≥ NOVEL_TRIGGER
  B 真重复（同模板同 token）必须**不**触发 —— 防「一律返回 1.0」的假修
  C 全新句型仍走比例判据，且比例语义未被破坏（含一个应触发的正对照）
  D 模板判据不得因异常拖垮主判据（_template_of / _known_templates 抛错时仍返回比例值）
  E existing_id 排除自身：同一节点自评时不应被自己「抹平」成已知

运行（仓根）：
  PYTHONUTF8=1 python -X utf8 -m pytest tests/test_issue573_novelty_template_guard.py -q --no-header
"""
from __future__ import annotations

import os
import sys
import tempfile

_THIS = os.path.abspath(__file__)
REPO = os.environ.get("LINGSHU_ROOT") or os.path.dirname(os.path.dirname(_THIS))
sys.path.insert(0, REPO)

from lingshu.core.core import SpacetimeMemoryEngine          # noqa: E402
from lingshu.core.longterm_gate import LongTermMemoryGate    # noqa: E402

TRIGGER = LongTermMemoryGate.NOVEL_TRIGGER

#: 同模板、只换 ASCII token 的六族（取自 issue #573 正文的 case 表）
SAME_TEMPLATE = [
    ("数据库 prod-01 的连接端口是 5432", "数据库 prod-01 的连接端口是 6379"),
    ("主机 prod-01 运行着订单服务", "主机 prod-07 运行着订单服务"),
    ("收款账号是 6222 8888 0000 1234", "收款账号是 6222 9999 0000 5678"),
    ("这个漏洞的编号是 CVE-2021-44228", "这个漏洞的编号是 CVE-2023-99999"),
    ("服务部署在 api.example.com 域名下", "服务部署在 api.othersite.net 域名下"),
    ("服务器地址是 192.168.1.10", "服务器地址是 10.0.0.77"),
]


def _gate_with(content: str):
    """新库 + 写入一条已知内容，返回 gate。"""
    d = tempfile.mkdtemp(prefix="g573_")
    e = SpacetimeMemoryEngine(os.path.join(d, "p.db"))
    gate = LongTermMemoryGate(engine=e)
    e.add_perception(content)
    return e, gate


# ---------------- A 同模板换 token ----------------

def test_a_same_template_new_token_triggers():
    """A 同模板只换 ASCII token ⇒ 内容确实变了 ⇒ 必须触发 prefeed。"""
    bad = []
    for known, probe in SAME_TEMPLATE:
        _e, gate = _gate_with(known)
        n = gate._novelty(probe)
        if n < TRIGGER:
            bad.append((known[:14], probe[:14], round(n, 4)))
    assert not bad, (
        "A 同模板换 token 仍未触发（阈值 %.2f）：%r\n"
        "   —— 比例判据的新增/消失 gram 相互抵消，必须靠模板判据兜住（#573）"
        % (TRIGGER, bad))


# ---------------- B 真重复不得触发 ----------------

def test_b_exact_repeat_does_not_trigger():
    """B 完全重复（同模板同 token）必须判为已知——防「一律 1.0」的假修。"""
    for known, _probe in SAME_TEMPLATE:
        _e, gate = _gate_with(known)
        n = gate._novelty(known)
        assert n < TRIGGER, (
            "B 完全重复的内容被判为新（novelty=%.4f ≥ %.2f）：%r\n"
            "   —— 模板判据把「同模板同 token」也放过了（假修）"
            % (n, TRIGGER, known[:20]))


# ---------------- C 全新句型仍走比例判据 ----------------

def test_c_brand_new_content_still_triggers():
    """C 完全不同的内容必须触发（比例判据语义未坏）。"""
    _e, gate = _gate_with("数据库 prod-01 的连接端口是 5432")
    n = gate._novelty("今天下午团队开会讨论了明年的预算规划")
    assert n >= TRIGGER, "C 全新内容未触发（novelty=%.4f）：比例判据被改坏" % (n,)


# ---------------- D 模板判据异常不得拖垮主判据 ----------------

def test_d_template_failure_does_not_break_ratio():
    """D _known_templates 抛错时，_novelty 仍须返回比例值而不是 0.5/异常。"""
    _e, gate = _gate_with("数据库 prod-01 的连接端口是 5432")

    def _boom(*a, **k):
        raise RuntimeError("boom")

    original = gate._known_templates
    try:
        gate._known_templates = _boom
        n = gate._novelty("数据库 prod-01 的连接端口是 6379")
        # 模板判据失效 ⇒ 退回比例值 0.2400（旧行为），不是 0.5、也不抛错
        assert 0.0 <= n < TRIGGER, (
            "D 模板判据抛错后应退回比例值（实测 %.4f）—— 不得变成 0.5 或抛异常" % (n,))
    finally:
        gate._known_templates = original


# ---------------- E existing_id 排除自身 ----------------

def test_e_existing_id_excludes_self():
    """E 同一节点自评（existing_id=自己）不应被自己的内容抹平。

    ★ 注意 `add_perception` 返回的是 **STNode 对象**（不是 id 字符串），
    必须取 `.id`；直接把对象传进 existing_id 会 `str(obj)` 化、匹配不到任何节点，
    于是排除逻辑静默失效（这类假通过正是本守卫要挡的）。

    ★ 口径边界（实测，非本件改动）：`_known_grams(exclude_id=X)` 在**库里只有 X
    这一个节点**时返回空集 ⇒ `_novelty` 命中既有的「无参照：中性」分支返回 0.5
    （这是 v1.27 就有的语义，不是本件引入）。故本守卫放**第二个节点**，
    保证排除自身后参考集非空——否则测的是那条中性分支，与本件无关。
    """
    d = tempfile.mkdtemp(prefix="g573e_")
    e = SpacetimeMemoryEngine(os.path.join(d, "p.db"))
    gate = LongTermMemoryGate(engine=e)
    node = e.add_perception("数据库 prod-01 的连接端口是 5432")
    e.add_perception("缓存过期时间是 300 秒")      # 保证排除自身后参考集非空
    node_id = getattr(node, "id", None) or str(node)
    assert node_id and node_id.startswith("node_"), (
        "E 未能取得节点 id（拿到 %r）—— add_perception 的返回体口径变了？" % (node_id,))
    n = gate._novelty("数据库 prod-01 的连接端口是 6379", existing_id=node_id)
    assert n >= TRIGGER, (
        "E 带 existing_id 的同模板换 token 未触发（novelty=%.4f）—— 自身排除把变更抹平了"
        % (n,))


def test_e2_single_node_library_falls_back_to_neutral():
    """E2 已知口径钉住：库中只有被排除的那一个节点时返回 0.5（中性），不是 1.0。

    v1.27 起 `_known_grams(exclude_id=X)` 只剔除「仅出现在 X」的 gram；
    若库里只有 X，参考集为空 ⇒ 返回 0.5「无参照」。本守卫**故意钉住**该行为，
    防止有人为修 #573 而把这条中性分支改成 1.0（那会让所有单节点库的写入
    都被判成高新奇，是反向假修）。
    """
    d = tempfile.mkdtemp(prefix="g573e2_")
    e = SpacetimeMemoryEngine(os.path.join(d, "p.db"))
    gate = LongTermMemoryGate(engine=e)
    node = e.add_perception("数据库 prod-01 的连接端口是 5432")
    nid = getattr(node, "id", None) or str(node)
    n = gate._novelty("数据库 prod-01 的连接端口是 6379", existing_id=nid)
    assert abs(n - 0.5) < 1e-9, (
        "E2 单节点库排除自身后应回中性 0.5（实测 %.4f）—— 参考集为空时的语义被改了"
        % (n,))
