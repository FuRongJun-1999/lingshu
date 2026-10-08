# -*- coding: utf-8 -*-
"""test_issue109_external_anchor_gate · 外部锚点写入授权闸守卫（lingshu issue #109）
============================================================================
缺陷（#109 · 写边界）：`SpacetimeMemoryEngine.register_external_anchor` 的
PRIMARY 分支经 `add_structure_node` 直接写**结构层**（`confidence=1.0`、不可删），
**无需任何密钥**，且绕过「提案→复核→终裁」全链（`propose_promotion` 对共享层直接
拒绝）。对照真闸 `LayeredStore.adjudicate_promotion`（同为「写结构层」）无密钥即
fail-closed —— 两条路待遇相反。

修法（最小 · 复用既有先例）：给 `register_external_anchor` 加 `designer_key`
形参 + `verify_designer` 闸（与 `adjudicate_promotion` **同款** fail-closed）。

断言组（回退即红）：
  A 无密钥 / 密钥未配置 ⇒ `PermissionError`（fail-closed，给拒的形态），且**零落库**。
  B 错误密钥 ⇒ 同样拒绝（不是「有值即过」）。
  C 正确密钥 ⇒ 照旧生效（PRIMARY 写 STRUCTURE / confidence=1.0）。
  D 拒绝是 `PermissionError` 而非「静默降级到知识层」——闸不可绕过。

运行（仓根）：python -X utf8 tests/test_issue109_external_anchor_gate.py
或：python -X utf8 -m pytest tests/test_issue109_external_anchor_gate.py -v
退出码：0 = 全过；1 = 有断言失败
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingshu.core.core import (  # noqa: E402
    SpacetimeMemoryEngine, MemoryLayer,
)

#: 自设假值（不读取、不打印任何真实凭据；verify_designer 判据=传入值==本进程环境值）
DUMMY_KEY = "GOV-IMPL-DUMMY-KEY"


class ExternalAnchorGateTests(unittest.TestCase):
    def setUp(self):
        # 环境隔离：本守卫自管 AEIS_DESIGNER_KEY，测毕复原
        self._saved = os.environ.pop("AEIS_DESIGNER_KEY", None)
        self.addCleanup(self._restore)

    def _restore(self):
        os.environ.pop("AEIS_DESIGNER_KEY", None)
        if self._saved is not None:
            os.environ["AEIS_DESIGNER_KEY"] = self._saved

    def _engine(self):
        eng = SpacetimeMemoryEngine(":memory:", identity="guard-109")
        self.addCleanup(eng.close)
        return eng

    # ---- A 无密钥 ⇒ 拒 + 零落库 ----
    def test_no_key_is_rejected_fail_closed(self):
        eng = self._engine()  # AEIS_DESIGNER_KEY 未配置
        with self.assertRaises(PermissionError):
            eng.register_external_anchor("external_calibration", "EXT-PAYLOAD")
        # 拒绝即零副作用：结构层不得新增节点
        self.assertEqual(eng.store.count_layer(MemoryLayer.STRUCTURE), 0)

    def test_missing_key_argument_against_configured_env_is_rejected(self):
        os.environ["AEIS_DESIGNER_KEY"] = DUMMY_KEY
        eng = self._engine()
        with self.assertRaises(PermissionError):
            eng.register_external_anchor("introspection", "EXT-PAYLOAD")
        self.assertEqual(eng.store.count_layer(MemoryLayer.STRUCTURE), 0)

    # ---- B 错误密钥 ⇒ 拒 ----
    def test_wrong_key_is_rejected(self):
        os.environ["AEIS_DESIGNER_KEY"] = DUMMY_KEY
        eng = self._engine()
        with self.assertRaises(PermissionError):
            eng.register_external_anchor("pre_access_stance", "EXT-PAYLOAD",
                                         designer_key="WRONG-KEY")
        self.assertEqual(eng.store.count_layer(MemoryLayer.STRUCTURE), 0)

    # ---- C 正确密钥 ⇒ 照旧生效 ----
    def test_correct_key_still_writes_structure_layer(self):
        os.environ["AEIS_DESIGNER_KEY"] = DUMMY_KEY
        eng = self._engine()
        node = eng.register_external_anchor("external_calibration", "EXT-PAYLOAD",
                                            designer_key=DUMMY_KEY)
        self.assertEqual(node.layer, MemoryLayer.STRUCTURE)
        self.assertEqual(node.confidence, 1.0)
        self.assertEqual(eng.store.count_layer(MemoryLayer.STRUCTURE), 1)
        # 照旧：不可删（不可遗忘层）
        self.assertFalse(eng.store.delete_node(node.id))
        self.assertIsNotNone(eng.store.get_node(node.id))

    # ---- D 闸不可绕过：拒绝形态是 PermissionError，不是静默降级 ----
    def test_rejection_is_permission_error_not_silent_degrade(self):
        eng = self._engine()
        try:
            node = eng.register_external_anchor("external_calibration", "EXT-PAYLOAD")
        except PermissionError:
            node = None
        # 无论是否抛错，都不得出现「落到知识层」的静默放行
        self.assertIsNone(node)
        self.assertEqual(eng.store.count_layer(MemoryLayer.KNOWLEDGE), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
