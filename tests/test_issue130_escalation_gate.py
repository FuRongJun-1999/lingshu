# -*- coding: utf-8 -*-
"""test_issue130_escalation_gate · 升级点匹配修正 + 授权闸 + 审计留痕（issue #130）
============================================================================
缺陷（#130 · 部分成立）：
  ① 匹配异常（`check_escalation`）：判据 `signal_type in point["trigger"] or
     signal_type in point["condition"]` **方向反了**——本意应是「触发词出现在信号
     里」。旁证读数：精确触发词命中 1、含词长句命中 0、**空串命中全部 6 条**。
  ② `set_escalation_enabled` 无授权、无审计（`action_logs` 前后 0→0）。

修法（最小 · 复用既有先例）：
  ① 空串/纯空白短路返回 `[]`；包含方向双向判定（保留既有调用方以 token 作信号的
     用法，如私有侧 `check_escalation("deviation", d_norm)`）。
  ② `set_escalation_enabled` 加 `designer_key` + `verify_designer`（与
     `adjudicate_promotion` 同款 fail-closed）+ 落 `action_logs`。

断言组（回退即红）：
  A 匹配：空串 / 纯空白 ⇒ 0；含词长句命中触发词（方向修正）；既有 token 用法不回归。
  B 无密钥 / 错误密钥 ⇒ `PermissionError`，且开关**未变**、无审计行。
  C 正确密钥 ⇒ 开关生效 + `action_logs` 新增一行且 `outcome` 可查。
  D 修前读数对照（改坏方向即红）：空串不得命中全部。

运行（仓根）：python -X utf8 tests/test_issue130_escalation_gate.py
或：python -X utf8 -m pytest tests/test_issue130_escalation_gate.py -v
退出码：0 = 全过；1 = 有断言失败
"""
from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingshu.core.core import SpacetimeMemoryEngine  # noqa: E402

DUMMY_KEY = "GOV-IMPL-DUMMY-KEY"
SEEDED = 6  # ESC-001..ESC-006


class EscalationMatchTests(unittest.TestCase):
    """匹配修正（无需密钥）。"""

    def setUp(self):
        self.engine = SpacetimeMemoryEngine(":memory:", identity="guard-130m")
        self.addCleanup(self.engine.close)

    def _codes(self, matches):
        return {p["code"] for p in matches}

    def test_seed_count_unchanged(self):
        # 前置：默认播种 6 条（空串命中的「6」正是它们）
        self.assertEqual(len(self.engine.list_escalation_points(enabled_only=False)),
                         SEEDED)

    def test_empty_and_whitespace_signal_match_nothing(self):
        # 核心修正：空串不得命中全部（修前 = 6）
        self.assertEqual(self.engine.check_escalation(""), [])
        self.assertEqual(self.engine.check_escalation("   "), [])

    def test_exact_trigger_word_hits_its_point(self):
        self.assertEqual(self._codes(self.engine.check_escalation("自维持迹象")),
                         {"ESC-001"})

    def test_long_sentence_containing_trigger_now_hits(self):
        # 方向修正：修前命中 0；触发词出现在信号里 ⇒ 命中 ESC-001
        self.assertIn("ESC-001",
                      self._codes(self.engine.check_escalation("关于自维持迹象的报告")))

    def test_existing_token_usage_does_not_regress(self):
        # 回归：私有侧既有调用方以 token 作信号（signal in condition）
        self.assertIn("ESC-004", self._codes(self.engine.check_escalation("deviation")))
        self.assertIn("ESC-002", self._codes(self.engine.check_escalation("P0")))
        self.assertIn("ESC-002", self._codes(self.engine.check_escalation("结构威胁")))

    def test_unknown_signal_matches_nothing(self):
        self.assertEqual(self.engine.check_escalation("绝不相干的信号XYZ"), [])


class EscalationGateAuditTests(unittest.TestCase):
    """授权闸 + 审计留痕。"""

    def setUp(self):
        self._saved = os.environ.pop("AEIS_DESIGNER_KEY", None)
        self.addCleanup(self._restore)
        self.engine = SpacetimeMemoryEngine(":memory:", identity="guard-130g")
        self.addCleanup(self.engine.close)
        self.target = self.engine.list_escalation_points(enabled_only=False)[0]["id"]

    def _restore(self):
        os.environ.pop("AEIS_DESIGNER_KEY", None)
        if self._saved is not None:
            os.environ["AEIS_DESIGNER_KEY"] = self._saved

    def _enabled(self, esc_id):
        for p in self.engine.list_escalation_points(enabled_only=False):
            if p["id"] == esc_id:
                return p["enabled"]
        raise AssertionError("升级点不存在: " + esc_id)

    def _log_rows(self):
        c = self.engine.store.conn.cursor()
        c.execute("SELECT action_type, summary, outcome FROM action_logs"
                  " ORDER BY id")
        return c.fetchall()

    # ---- B 无密钥 / 错误密钥 ⇒ 拒 + 未变 + 无审计 ----
    def test_no_key_rejected_and_state_unchanged_no_audit(self):
        with self.assertRaises(PermissionError):
            self.engine.set_escalation_enabled(self.target, False)
        self.assertTrue(self._enabled(self.target))
        self.assertEqual(len(self._log_rows()), 0)

    def test_wrong_key_rejected_no_audit(self):
        os.environ["AEIS_DESIGNER_KEY"] = DUMMY_KEY
        with self.assertRaises(PermissionError):
            self.engine.set_escalation_enabled(self.target, False,
                                               designer_key="WRONG-KEY")
        self.assertTrue(self._enabled(self.target))
        self.assertEqual(len(self._log_rows()), 0)

    # ---- C 正确密钥 ⇒ 生效 + 审计可查 ----
    def test_correct_key_applies_and_writes_audit(self):
        os.environ["AEIS_DESIGNER_KEY"] = DUMMY_KEY
        self.engine.set_escalation_enabled(self.target, False,
                                           designer_key=DUMMY_KEY)
        self.assertFalse(self._enabled(self.target))
        rows = self._log_rows()
        self.assertEqual(len(rows), 1)
        action_type, summary, outcome = rows[0]
        self.assertEqual(action_type, "escalation_toggle")
        self.assertIn(self.target, summary)
        self.assertEqual(json.loads(outcome),
                         {"escalation_id": self.target, "enabled": False})

    def test_reenable_also_audited(self):
        os.environ["AEIS_DESIGNER_KEY"] = DUMMY_KEY
        self.engine.set_escalation_enabled(self.target, False,
                                           designer_key=DUMMY_KEY)
        self.engine.set_escalation_enabled(self.target, True,
                                           designer_key=DUMMY_KEY)
        self.assertTrue(self._enabled(self.target))
        rows = self._log_rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual(json.loads(rows[1][2])["enabled"], True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
