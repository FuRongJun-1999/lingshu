# -*- coding: utf-8 -*-
"""#51 守卫：`verify_designer` 对非 ASCII 密钥必须**干净拒绝/干净接受**，不得抛 TypeError。

缺陷面（公开 issue #51）：`hmac.compare_digest(str, str)` 对非 ASCII str 抛
TypeError（CPython 文档化行为）⇒ `AEIS_DESIGNER_KEY` 为中文（或任何非 ASCII）时，
6 条终裁通路（晋升/验证标准/危机/盲区裁决及其透传）从「拒绝」变成「崩」。

本件锁（fail-closed 语义不变，只修「崩」）：
  ① 中文密钥正确 ⇒ True；
  ② 中文密钥错误 ⇒ False（不抛）；
  ③ 非字符串入参（数字等）⇒ False（str() 归一后比较）；
  ④ 未配置环境变量 ⇒ False；
  ⑤ ASCII 密钥路径行为不变（正确 True / 错误 False）。
定点变异：比较改回 `compare_digest(str, str)` ⇒ ①② 红。
"""
import pytest

from lingshu.core import core as C


@pytest.fixture
def cn_key(monkeypatch):
    monkeypatch.setenv("AEIS_DESIGNER_KEY", "设计者密钥测试")


@pytest.fixture
def ascii_key(monkeypatch):
    monkeypatch.setenv("AEIS_DESIGNER_KEY", "designer-key-ascii-123")


# ---------------------------------------------------------------- ①② 中文密钥

def test_unicode_key_correct_accepts(cn_key):
    assert C.verify_designer("设计者密钥测试") is True


def test_unicode_key_wrong_rejects_without_raise(cn_key):
    assert C.verify_designer("错误密钥") is False


# ---------------------------------------------------------------- ③ 非字符串

def test_non_string_input_rejects(cn_key):
    assert C.verify_designer(12345) is False
    assert C.verify_designer(None) is False


# ---------------------------------------------------------------- ④ 未配置

def test_unconfigured_env_rejects(monkeypatch):
    monkeypatch.delenv("AEIS_DESIGNER_KEY", raising=False)
    assert C.verify_designer("任意") is False


# ---------------------------------------------------------------- ⑤ ASCII 不变

def test_ascii_key_paths_unchanged(ascii_key):
    assert C.verify_designer("designer-key-ascii-123") is True
    assert C.verify_designer("wrong") is False
    assert C.verify_designer("") is False
