# -*- coding: utf-8 -*-
"""test_issue196_nonfinite_gate · 非有限输入不得产出「通过」态（lingshu issue #196）
============================================================================
背景（lingshu issue #196）：
  `NaN` 与任何比较都返回 False。三处判据共同采用「先比阈值、最后兜底 return
  成功态」的写法，且入口不校验数值有限性 ⇒ NaN 输入下所有 `x < 阈值` 都不成立，
  控制流直落函数末尾的成功分支——损坏/未初始化的数值被静默判为「通过」：

    ① `lingshu/core/verdict.py`   assess(part_fg_px=NaN) → ACCEPT
    ② `lingshu/nn/hex_composite.py`（及 gen/ 双副本）judge_density(density=NaN) → ACCEPT
    ③ `lingshu/world/stable_lease.py`  StableLease(confidence/ttl=NaN) → stable

  缺陷形态在 HEAD 实测（本仓 Python 3.12.10）：
    assess({'type':'head'}, 清晰域, nan, 10000)  → {'verdict':'ACCEPT',...,'fg_ratio':nan}
    judge_density({'composite':'car','score':1.0}, nan, 3.0) → {'state':'ACCEPT',...}
    StableLease().acquire('k', confidence=nan)   → {'state':'stable',...}

断言组（回退 / 放宽即红）：
  G1 `verdict.assess` 非有限输入 → **BLINDSPOT**（reason 钉死 'non-finite input'），
     绝非 ACCEPT；正对照：有限输入照旧走 REJECT / BLINDSPOT / ACCEPT 各档。
  G2 `judge_density`（nn 与 gen **两份**副本）非有限 density/th_dense → **DEFER**，
     绝非 ACCEPT；正对照：有限输入照旧 ACCEPT / DEFER(依据不足)。
  G3 `StableLease` 非有限 confidence / ttl / gamma / weak_threshold（经 acquire /
     renew / check 三条入口）→ 一律 **weak** 且 `in_lease is False`，绝非 stable；
     正对照：全有限参数照旧 stable；降级只作用于状态、记录不删（P1-003）。
  G4 双副本 `lingshu/gen/hex_composite.py` 与 `lingshu/nn/hex_composite.py`
     逐字节相同（防只修一份）。

判据来源（逐条写清出处，不编造理论依据）：
  · verdict → BLINDSPOT：理论只给了四态语义——`docs/theory/自研蜂窝CNN_理论稿_v0.1.md:33`
    「BLINDSPOT(不可判带)」、`lingshu/core/core.py:3915`「没有则诚实声明 BLINDSPOT」；
    「非有限输入 ⇒ 不可判 ⇒ 不给 ACCEPT」这一步**理论未规定 NaN 语义（追不到）**，
    属工程 fail-closed 约定，与同仓 `tests/test_issue402_nan_clamp_guard.py` 头部
    「非有限输入不得被当作有效证据/支持」同源。
  · stable_lease → weak：判据在本仓源码内可核——`lingshu/world/stable_lease.py`
    头部 3.2.2 定义 `S ∈ stable ⟺ (t-t_verified) < TTL ∧ ¬conflict`；TTL/置信度
    非有限时该合取式无法成立，故不得判 stable。降级口径见同文件 P1-003 边界
    （降级作用于状态，不删除记录）。
  · judge_density → DEFER：判据是 `judge_density` 自身 docstring 的既有四态口径
    「依据不足 ⇒ DEFER」；非有限密度即「依据不可判」，退到 DEFER 而非 ACCEPT。
  · 阈值本体（0.10 / 0.30 / th_dense / TTL）**一律未改**——本件只加输入有限性闸，
    不放宽任何判据。

不适用条件 / 已知边界：
  · 本守卫只钉「非有限数值输入」这一类；不覆盖其它上游污染形态（越界、类型错）。
  · `verdict.assess` 的 `image_domains` 文案（clothing/occlusion/contrast）不在
    本件范围；只钉数值入口。

定点变异自证（抽掉修复 ⇒ 必红）：
  删 `verdict.py:assess` 入口的 `math.isfinite` 闸 ⇒ G1 红（报 ACCEPT）；
  删 `hex_composite.judge_density` 的同名闸 ⇒ G2 红（两份副本，报 ACCEPT）；
  删 `stable_lease.check` 的非有限闸 ⇒ G3 红（报 stable）。

运行（仓根）：python -X utf8 tests/test_issue196_nonfinite_gate.py
             / python -X utf8 -m pytest tests/test_issue196_nonfinite_gate.py
退出码：0 = 全过；1 = 有断言失败
"""
from __future__ import annotations

import hashlib
import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lingshu.core.verdict import assess                        # noqa: E402
from lingshu.gen.hex_composite import judge_density as gen_jd   # noqa: E402
from lingshu.nn.hex_composite import judge_density as nn_jd     # noqa: E402
from lingshu.world.stable_lease import StableLease             # noqa: E402

NAN = float("nan")
INF = float("inf")
NINF = float("-inf")
DOMAINS = {"clothing": "无", "occlusion": "无遮挡", "contrast": "清晰",
           "line_edge": "淡线稿"}


class TestVerdictNonFiniteGate(unittest.TestCase):
    """G1：verdict.assess 非有限输入 → BLINDSPOT（#196）。"""

    def test_nan_fg_px_is_not_accept(self):
        r = assess({"type": "head"}, DOMAINS, NAN, 10000)
        self.assertEqual(r["verdict"], "BLINDSPOT", r)
        self.assertEqual(r["reason"], "non-finite input", r)

    def test_nan_area_is_not_accept(self):
        r = assess({"type": "head"}, DOMAINS, 9000, NAN)
        self.assertEqual(r["verdict"], "BLINDSPOT", r)
        self.assertEqual(r["reason"], "non-finite input", r)

    def test_infinities_are_not_accept(self):
        for bad in (INF, NINF):
            with self.subTest(bad=bad):
                r = assess({"type": "head"}, DOMAINS, bad, 10000)
                self.assertEqual(r["verdict"], "BLINDSPOT", r)
                r2 = assess({"type": "head"}, DOMAINS, 9000, bad)
                self.assertEqual(r2["verdict"], "BLINDSPOT", r2)

    def test_positive_control_finite_paths_intact(self):
        """正对照（防误杀）：有限输入仍走原四态各档。"""
        self.assertEqual(assess({"type": "head"}, DOMAINS, 9000, 10000)["verdict"],
                         "ACCEPT")
        self.assertEqual(assess({"type": "head"}, DOMAINS, 100, 10000)["verdict"],
                         "REJECT")
        self.assertEqual(assess({"type": "head"}, DOMAINS, 2000, 10000)["verdict"],
                         "BLINDSPOT")
        d = dict(DOMAINS, clothing="常服")
        self.assertEqual(assess({"type": "torso"}, d, 9000, 10000)["verdict"],
                         "DEFER")


class TestJudgeDensityNonFiniteGate(unittest.TestCase):
    """G2：judge_density（nn 与 gen 双副本）非有限输入 → DEFER（#196）。"""

    def _check_copy(self, jd, label):
        self.assertEqual(jd({"composite": "car", "score": 1.0}, NAN, 3.0)["state"],
                         "DEFER", label)
        self.assertEqual(jd({"composite": "car", "score": 1.0}, 5.0, NAN)["state"],
                         "DEFER", label)
        self.assertEqual(jd({"composite": "car", "score": 1.0}, INF, 3.0)["state"],
                         "DEFER", label)
        # 正对照：有限输入照旧 ACCEPT / DEFER(依据不足)
        self.assertEqual(jd({"composite": "car", "score": 1.0}, 5.0, 3.0)["state"],
                         "ACCEPT", label)
        j2 = jd({"composite": "car", "score": 1.0}, 1.0, 3.0)
        self.assertEqual(j2["state"], "DEFER", label)
        self.assertEqual(j2["reason"], "证据密度不足", j2)

    def test_nn_copy(self):
        self._check_copy(nn_jd, "nn")

    def test_gen_copy(self):
        self._check_copy(gen_jd, "gen")


class TestStableLeaseNonFiniteGate(unittest.TestCase):
    """G3：StableLease 非有限输入 → weak（#196）。"""

    def _assert_not_stable(self, lease, key):
        st = lease.check(key)
        self.assertEqual(st["state"], "weak", st)
        self.assertIs(st["in_lease"], False, st)

    def test_nan_confidence(self):
        lease = StableLease()
        self.assertEqual(lease.acquire("k", confidence=NAN)["state"], "weak")
        self._assert_not_stable(lease, "k")

    def test_nan_ttl(self):
        lease = StableLease()
        self.assertEqual(lease.acquire("k", ttl=NAN)["state"], "weak")
        self._assert_not_stable(lease, "k")

    def test_infinite_ttl(self):
        lease = StableLease()
        self.assertEqual(lease.acquire("k", ttl=INF)["state"], "weak")
        self._assert_not_stable(lease, "k")

    def test_nan_gamma(self):
        lease = StableLease(gamma=NAN)
        self.assertEqual(lease.acquire("k")["state"], "weak")
        self._assert_not_stable(lease, "k")

    def test_nan_weak_threshold(self):
        lease = StableLease(weak_threshold=NAN)
        self.assertEqual(lease.acquire("k")["state"], "weak")
        self._assert_not_stable(lease, "k")

    def test_renew_with_nan_is_not_stable(self):
        lease = StableLease()
        self.assertEqual(lease.renew("k", confidence=NAN)["state"], "weak")
        self._assert_not_stable(lease, "k")

    def test_renew_on_existing_stable_lease_with_nan_not_stable(self):
        """已有有效租约再以 NaN 续期 ⇒ 不得回到 stable（钉 renew 自身那道闸）。

        注：上一次调用的租约是 None，走 acquire 腿；这一条先 acquire 出 stable，
        再 renew(NaN)——若只修 acquire 而漏 renew，此处必红。
        """
        lease = StableLease()
        self.assertEqual(lease.acquire("k")["state"], "stable")
        self.assertEqual(lease.renew("k", confidence=NAN)["state"], "weak")
        self._assert_not_stable(lease, "k")
        self.assertEqual(lease.state("k")["state"], "weak")

    def test_positive_control_finite_is_stable(self):
        """正对照（防误杀）：全有限参数照旧 stable，且记录保留（P1-003）。"""
        lease = StableLease()
        self.assertEqual(lease.acquire("k")["state"], "stable")
        st = lease.check("k")
        self.assertEqual(st["state"], "stable", st)
        self.assertIs(st["in_lease"], True, st)
        # 降级只作用于状态、不删除记录（P1-003）
        lease2 = StableLease()
        lease2.acquire("k", confidence=NAN)
        self.assertEqual(lease2.state("k")["state"], "weak")
        self.assertEqual(lease2.state("k")["key"], "k")


class TestHexCompositeDualCopyIdentical(unittest.TestCase):
    """G4：gen/ 与 nn/ 的 hex_composite.py 逐字节相同（防只修一份）。"""

    def test_byte_identical(self):
        gen = os.path.join(REPO, "lingshu", "gen", "hex_composite.py")
        nn = os.path.join(REPO, "lingshu", "nn", "hex_composite.py")
        with open(gen, "rb") as f:
            hg = hashlib.sha256(f.read()).hexdigest()
        with open(nn, "rb") as f:
            hn = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(hg, hn, "双副本已漂移：只修了一份？")


if __name__ == "__main__":
    print("MUTATION：抽掉修复后本件必红——")
    print("  verdict.py:assess 去 isfinite 闸  ⇒ G1 报 ACCEPT")
    print("  hex_composite.judge_density 去闸   ⇒ G2 报 ACCEPT（两份副本）")
    print("  stable_lease.check 去闸            ⇒ G3 报 stable")
    print("  stable_lease.acquire/renew 去参数闸 ⇒ G3 报 stable")
    unittest.main(verbosity=2)
