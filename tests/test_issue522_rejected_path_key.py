# -*- coding: utf-8 -*-
"""test_issue522_rejected_path_key · mark_rejected_path_consumed 须设计者密钥（#522）
==================================================================================
缺陷（#522，P1）：#216 要求「被否决内容重提须 override + 设计者密钥」，但
公开的 mark_rejected_path_consumed **无密钥**即可把否决行从 status='open'
翻成 'consumed'——而 #216 闸只读 status='open' ⇒ 翻完即可重提，绕过整个
负记忆闸。

修法：mark_rejected_path_consumed 补 designer_key 参数并过 verify_designer
（fail-closed，与 resolve_blindspot 同款）——消费否决行＝解除负记忆闸，
不得无密钥调用。

判据：
  G1 无密钥调用 ⇒ PermissionError；
  G2 有正确密钥 ⇒ 正常消费（不误伤合法路径）；
  G3 消费后 status 变为 consumed。

运行：python -X utf8 -m pytest tests/test_issue522_rejected_path_key.py -q
"""
from __future__ import annotations

import os

import pytest

from lingshu.core.core import LayeredStore


def _store_with_rejected():
    s = LayeredStore(":memory:")
    rid = s.add_rejected_path("promotion", "测试否决路径", "设计者终裁否决")
    return s, rid


def test_g1_no_key_rejected():
    s, rid = _store_with_rejected()
    try:
        with pytest.raises(PermissionError, match=r"#522"):
            s.mark_rejected_path_consumed(rid)
    finally:
        s.close()


def test_g2_with_key_consumed():
    os.environ["AEIS_DESIGNER_KEY"] = "test_key_522"
    try:
        s, rid = _store_with_rejected()
        try:
            s.mark_rejected_path_consumed(rid, designer_key="test_key_522")
            paths = s.list_rejected_paths()
            assert all(p["status"] == "consumed" for p in paths), paths
        finally:
            s.close()
    finally:
        del os.environ["AEIS_DESIGNER_KEY"]


if __name__ == "__main__":
    import subprocess
    import sys
    raise SystemExit(subprocess.call(
        [sys.executable, "-X", "utf8", "-m", "pytest", __file__, "-q", "--no-header"]))
