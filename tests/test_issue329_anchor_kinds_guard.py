# -*- coding: utf-8 -*-
"""test_issue329_anchor_kinds_guard · 锚点层写入面 fail-closed 守卫（lingshu issue #329）
============================================================================
背景（lingshu issue #329）：
  锚点层（`MemoryLayer.ANCHOR`）与结构层同为**不可遗忘**的共享层；其中
  `register_external_anchor` 是「外部内容 → 不可遗忘共享层」的唯一入口，其
  `ANCHOR_KINDS` 只认三种 kind。外部观察报称：版本迭代记录无 kind 可走，且
  「结构层重要事件」的记录惯例在仓内**四处内联复制**、其中 `pending_sync` 标签
  只一处带（另两处漏）⇒ 同惯例已漂移。

裁定（维护者 2026-10-10，原话）：
  「关于锚点层确实是极度重要的信息，需要收紧。」
  ⇒ **维持三 kind 白名单、不放行第四种**；锚点层是不可变声明层，写入面
  **一律 fail-closed**。本件把这条取向**钉成可判红的守卫**（防松动）。

断言组（回退 / 放宽即红）：
  G1 白名单**恰好**是这三项（钉死取值与顺序——**不是**「至少包含」）。
  G2 全仓**不存在第二份硬编码 kind 列表**（AST 扫描：含 ≥2 个 kind 字面量的
     容器字面量必须唯一，且落在 `core.py` 的 `ANCHOR_KINDS` 定义处）。
  G3 写入门槛仍在：非白名单 kind ⇒ `ValueError`，且**零落库**（结构层/知识层
     都不新增）；白名单 kind + 正确密钥 ⇒ 照旧写结构层（正对照，防误杀）。
  G4 记录惯例**已收归单一真源**：四处内联复制不存在——`pending_sync` 字面量在
     `core.py` 只出现于常量 `SUB_PENDING_SYNC_TAG` 的定义；三处记录点均经
     `_write_structure_record`；SUB 分支写出的节点**一律**带 `pending_sync`
     （行为断言：`register_external_anchor` 的 SUB 腿）。

定点变异自证（见文件末 `__main__` 打印的 MUTATION 提示）：
  向 `ANCHOR_KINDS` 注入第四种 kind ⇒ G1（并连带 G2）必红并点名；
  恢复后复绿。

运行（仓根）：python -X utf8 tests/test_issue329_anchor_kinds_guard.py
             / python -X utf8 -m pytest tests/test_issue329_anchor_kinds_guard.py
退出码：0 = 全过；1 = 有断言失败
"""
from __future__ import annotations

import ast
import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lingshu.core.core import (  # noqa: E402
    SpacetimeMemoryEngine, MemoryLayer, Role,
)

#: 裁定钉死的白名单（**恰好**这三项，含顺序）。改动本行 = 松动裁定，守卫必红。
PINNED_ANCHOR_KINDS = ("pre_access_stance", "introspection", "external_calibration")

#: 自设假值（不读取、不打印任何真实凭据）
DUMMY_KEY = "ANCHOR-GUARD-DUMMY-KEY"

CORE_PY = os.path.join(REPO, "lingshu", "core", "core.py")


def _scan_kind_containers():
    """AST 扫描 `lingshu/**/*.py`：返回含 ≥2 个 kind 字面量的容器字面量清单。

    每项 = (相对路径, 行号, 命中的 kind 名集合)。这是「第二份硬编码 kind 列表」
    的判据——真正的白名单只应出现一次（`core.py` 的 `ANCHOR_KINDS` 定义处）。
    """
    kinds = set(PINNED_ANCHOR_KINDS)
    found = []
    pkg = os.path.join(REPO, "lingshu")
    for dp, _dn, fn in os.walk(pkg):
        if "__pycache__" in dp:
            continue
        for f in fn:
            if not f.endswith(".py"):
                continue
            p = os.path.join(dp, f)
            rel = os.path.relpath(p, REPO).replace("\\", "/")
            try:
                with open(p, encoding="utf-8") as fh:
                    tree = ast.parse(fh.read())
            except SyntaxError:  # pragma: no cover
                continue
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Tuple, ast.List, ast.Set)):
                    continue
                elems = {e.value for e in node.elts
                         if isinstance(e, ast.Constant) and isinstance(e.value, str)}
                hit = elems & kinds
                if len(hit) >= 2:
                    found.append((rel, getattr(node, "lineno", 0), frozenset(hit)))
    return found


class AnchorKindsGuardTests(unittest.TestCase):
    # ---- 环境隔离：本守卫自管 AEIS_DESIGNER_KEY ----
    def setUp(self):
        self._saved = os.environ.pop("AEIS_DESIGNER_KEY", None)
        self.addCleanup(self._restore)

    def _restore(self):
        os.environ.pop("AEIS_DESIGNER_KEY", None)
        if self._saved is not None:
            os.environ["AEIS_DESIGNER_KEY"] = self._saved

    def _engine(self, role=Role.PRIMARY):
        eng = SpacetimeMemoryEngine(":memory:", identity="guard-329", role=role)
        self.addCleanup(eng.close)
        return eng

    # ---- G1 白名单恰好三项（钉死取值与顺序）----
    def test_g1_anchor_kinds_is_exactly_the_pinned_triple(self):
        actual = tuple(SpacetimeMemoryEngine.ANCHOR_KINDS)
        self.assertEqual(
            actual, PINNED_ANCHOR_KINDS,
            "ANCHOR_KINDS 漂移：actual=%r pinned=%r。\n"
            "锚点层是不可变声明层（维护者 2026-10-10 裁定「极度重要的信息，需要收紧」）"
            "⇒ 白名单是**封闭集**，新增 kind 属设计取舍、须经裁定，"
            "不得由实施者顺手放宽。" % (actual, PINNED_ANCHOR_KINDS))
        self.assertEqual(len(set(actual)), len(actual), "白名单含重复项：%r" % (actual,))

    # ---- G2 全仓不存在第二份硬编码 kind 列表 ----
    def test_g2_no_second_hardcoded_kind_list(self):
        containers = _scan_kind_containers()
        self.assertTrue(
            containers,
            "扫描器没扫到任何 kind 容器 ⇒ 它坏了（本判据此时无判别力，会静默通过）。")
        bad = [c for c in containers if c[0] != "lingshu/core/core.py"]
        self.assertEqual(
            bad, [],
            "发现**第二份硬编码 kind 列表**（单一真源 = core.py 的 ANCHOR_KINDS）：\n"
            + "\n".join("  · %s:%d  含 %s" % (r, ln, sorted(hit)) for r, ln, hit in bad))
        # 单文件内也只应有一处（定义处）；core.py 里其它位置不得复制该列表
        self.assertEqual(
            len(containers), 1,
            "core.py 内 kind 容器字面量不止一处（应只在 ANCHOR_KINDS 定义处）：%r"
            % (containers,))

    # ---- G3 写入门槛仍在（fail-closed）----
    def test_g3a_non_whitelist_kind_is_rejected_and_writes_nothing(self):
        os.environ["AEIS_DESIGNER_KEY"] = DUMMY_KEY
        eng = self._engine()
        for kind in ("version_iteration", "release", "version", "release_record", ""):
            with self.assertRaises(ValueError, msg="非白名单 kind %r 未被拒" % kind):
                eng.register_external_anchor(kind, "v0.0.2 骨架", designer_key=DUMMY_KEY)
        # 拒绝即零副作用：结构层 / 知识层都不得新增
        self.assertEqual(eng.store.count_layer(MemoryLayer.STRUCTURE), 0)
        self.assertEqual(eng.store.count_layer(MemoryLayer.KNOWLEDGE), 0)

    def test_g3b_whitelist_kind_with_key_still_writes_structure(self):
        """正对照：白名单内 kind + 正确密钥 ⇒ 照旧生效（防过度收紧误杀）。"""
        os.environ["AEIS_DESIGNER_KEY"] = DUMMY_KEY
        eng = self._engine()
        for kind in PINNED_ANCHOR_KINDS:
            node = eng.register_external_anchor(kind, "payload", designer_key=DUMMY_KEY)
            self.assertEqual(node.layer, MemoryLayer.STRUCTURE)
            self.assertEqual(node.confidence, 1.0)
        self.assertEqual(eng.store.count_layer(MemoryLayer.STRUCTURE), len(PINNED_ANCHOR_KINDS))

    def test_g3c_gate_is_fail_closed_without_key(self):
        """无密钥/未配置 ⇒ PermissionError（闸在 kind 判定之前，不可绕过）。"""
        eng = self._engine()
        with self.assertRaises(PermissionError):
            eng.register_external_anchor(PINNED_ANCHOR_KINDS[0], "payload")
        self.assertEqual(eng.store.count_layer(MemoryLayer.STRUCTURE), 0)

    # ---- G4 记录惯例已收归单一真源 ----
    def test_g4a_pending_sync_has_single_source(self):
        with open(CORE_PY, encoding="utf-8") as fh:
            src = fh.read()
        tree = ast.parse(src)
        # 字符串字面量 "pending_sync" 只应出现一次（SUB_PENDING_SYNC_TAG 的定义）
        literals = [n.lineno for n in ast.walk(tree)
                    if isinstance(n, ast.Constant) and n.value == "pending_sync"]
        self.assertEqual(
            len(literals), 1,
            "`pending_sync` 字面量在 core.py 出现 %d 次（行 %r）——应只作为常量 "
            "`SUB_PENDING_SYNC_TAG` 定义一次，其余引用一律走该常量（防同惯例再次内联漂移）。"
            % (len(literals), literals))
        self.assertIn("SUB_PENDING_SYNC_TAG = \"pending_sync\"", src,
                      "未找到单一真源常量 `SUB_PENDING_SYNC_TAG = \"pending_sync\"`")

    def test_g4b_all_record_sites_route_through_single_helper(self):
        with open(CORE_PY, encoding="utf-8") as fh:
            src = fh.read()
        tree = ast.parse(src)
        # 收集 `_write_structure_record(...)` 调用点 → 其第二实参（tag 字面量）
        helper_tags = set()
        calls = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = getattr(fn, "id", None) or getattr(fn, "attr", None)
            if name != "_write_structure_record":
                continue
            calls += 1
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                helper_tags.add(node.args[1].value)
        self.assertIn("def _write_structure_record(", src,
                      "缺单一收口辅助 `_write_structure_record`")
        self.assertGreaterEqual(calls, 3,
                                "`_write_structure_record` 调用点不足三处（calls=%d）" % calls)
        for tag in ("attention_weight", "migration", "verifier_standard"):
            self.assertIn(tag, helper_tags,
                          "记录点 %s 未接到辅助（实际 tag 集 = %r）" % (tag, sorted(helper_tags)))
        # 旧的四处内联标签列表形态必须消失
        for stale in ('tags=["attention_weight"]',
                      'tags=["migration", "v1.7"]',
                      'tags=["verifier_standard", "pending_sync"]'):
            self.assertNotIn(stale, src, "旧内联复制仍在：%s" % stale)
        # 记录惯例的角色分支只应存在于单一辅助内（+ 外部锚点自身分支）
        self.assertEqual(
            src.count("if self.role == Role.PRIMARY:"), 2,
            "`if self.role == Role.PRIMARY:` 出现次数 > 2 ⇒ 同惯例又被内联复制")

    def test_g4c_sub_writes_carry_pending_sync(self):
        """行为断言：SUB 角色经 register_external_anchor 写知识层副本 ⇒ 带 pending_sync。"""
        os.environ["AEIS_DESIGNER_KEY"] = DUMMY_KEY
        eng = self._engine(role=Role.SUB)
        node = eng.register_external_anchor(PINNED_ANCHOR_KINDS[0], "payload",
                                            designer_key=DUMMY_KEY)
        self.assertEqual(node.layer, MemoryLayer.KNOWLEDGE)
        self.assertIn("pending_sync", node.tags,
                      "SUB 知识层副本未带 pending_sync（同惯例漂移复发）：%r" % (node.tags,))


if __name__ == "__main__":
    print("# 定点变异自证提示：向 ANCHOR_KINDS 注入第四种 kind ⇒ G1/G2 必红并点名；恢复后复绿。")
    unittest.main(verbosity=2)
