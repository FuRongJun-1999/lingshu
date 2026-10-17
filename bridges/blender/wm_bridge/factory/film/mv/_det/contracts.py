# -*- coding: utf-8 -*-
"""wm_bridge.factory.film.mv._det.contracts · **契约族委托层**（不重复实现校验）

P0 的七份契约与校验器已在 `factory/film/`（`shot.py` / `score.py` / `bind.py` /
`edit.py` / `text.py` / `oracle.py`）。执行器**不得**另写一套校验 ——
否则同一份 schema 会有两套判定，跑久了必然走偏。

本模块只做一件事：把 P0 的一个个 `normalize_xxx` 收敛成执行器内部统一取用的
`normalize(kind, doc)` / `digest(kind, doc)` 面，并转导出执行器需要的常量与助手。

fail-closed 口径沿用 P0：未知键即抛、闭集即拒、`ContractError` 上抛不吞。
"""
from __future__ import annotations

import hashlib
import json
import os

from ... import _common as C                 # 契约助手（QUANT/ContractError/canonical_json/…）
from ... import bind as _bind
from ... import edit as _edit
from ... import oracle as _oracle
from ... import score as _score
from ... import shot as _shot
from ... import text as _text

__all__ = ["ContractError", "KINDS", "SCHEMAS", "normalize", "validate", "canonical",
           "digest", "write_json", "read_json", "QUANT", "ASPECT", "TITLE_SAFE",
           "ACTION_SAFE", "MOVE_KINDS", "SHOT_SCALES", "SUBJECT_ANCHORS",
           "TRANSITION_KINDS", "DEFAULT_EASING"]

ContractError = C.ContractError
QUANT = C.QUANT

#: 执行器要用的六份（`perform` 由 P0 契约面直接承载，执行器不产它）
KINDS = ("score", "shot", "bind", "edit", "text", "oracle")

SCHEMAS = {
    "score": _score.SCORE_SCHEMA,
    "shot": _shot.SHOT_SCHEMA,
    "bind": _bind.BIND_SCHEMA,
    "edit": _edit.EDIT_SCHEMA,
    "text": _text.TEXT_SCHEMA,
    "oracle": _oracle.ORACLE_SCHEMA,
}

_NORMALIZE = {
    "score": _score.normalize_score,
    "shot": _shot.normalize_shot,
    "bind": _bind.normalize_bind,
    "edit": _edit.normalize_edit,
    "text": _text.normalize_text,
    "oracle": _oracle.normalize_oracle,
}

# 执行器内部默认值（与 P0 常量同源，不另立数字）
ASPECT = _shot.ASPECT_16_9 if hasattr(_shot, "ASPECT_16_9") else 16.0 / 9.0
TITLE_SAFE = 0.80
ACTION_SAFE = 0.90
MOVE_KINDS = _shot.MOVE_KINDS
TRANSITION_KINDS = _shot.TRANSITION_KINDS
SUBJECT_ANCHORS = _shot.SUBJECT_ANCHORS
DEFAULT_EASING = _shot.DEFAULT_EASING
SHOT_SCALES = _bind.SHOT_SCALES


def normalize(kind: str, doc):
    """委托到 P0 的 `normalize_xxx`。未知种类 ⇒ ContractError。"""
    fn = _NORMALIZE.get(kind)
    if fn is None:
        raise ContractError("未知契约种类 %r；可选：%s" % (kind, ", ".join(KINDS)))
    return fn(doc)


def validate(kind: str, doc):
    """查询面：返回 `(ok, reason)`，**不抛异常**、不改入参。"""
    try:
        normalize(kind, doc)
        return True, "ok"
    except (C.ContractError, TypeError, ValueError, KeyError) as e:
        return False, "%s: %s" % (type(e).__name__, e)


def canonical(doc) -> str:
    """规范化 JSON 文本（键序无关、浮点量化到 QUANT）——digest 的唯一真源。"""
    return C.canonical_json(doc)


def digest(kind: str, doc) -> str:
    """内容摘要：**完整 sha256(64 位)**（对规范化后的 canonical JSON 取摘要）。

    为什么必须是 64 位：P0 各契约（`shot.spec_ref` / `score.source` / `bind.score_ref` /
    `edit.shot_ref` / `oracle.subject_ref`）的 `_digest_str` 有一条硬规矩 ——
    **形如十六进制就必须是 64 位 sha256**，否则拒收（防止"随手写个 id 冒充摘要"）。
    摘要只依赖内容（键序无关、浮点量化），故同内容必同摘要。
    """
    return hashlib.sha256(canonical(normalize(kind, doc)).encode("utf-8")).hexdigest()


def _quant(o):
    """递归量化（与 `canonical_json` 同口径），供**美化落盘**用。"""
    if isinstance(o, float):
        return C.q(o)
    if isinstance(o, dict):
        return {k: _quant(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_quant(v) for v in o]
    return o


def write_json(path: str, doc) -> str:
    """写 canonical JSON（缩进 2、键排序、末行换行），返回文件 sha256(64)。"""
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    text = json.dumps(_quant(doc), sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
