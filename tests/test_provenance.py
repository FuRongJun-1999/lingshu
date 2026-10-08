# -*- coding: utf-8 -*-
"""provenance.from_legacy 回归 · 验证状态反推不得静默降级

背景：from_legacy 的验证状态回填循环漏掉过 "verified"，导致带 `verified`
标签的旧记录被反推为 `unverified`——声明强度由 weak 静默掉为 none，
全程无异常、无日志。信任层的静默降级是最高代价的一类缺陷。

本件守两条性质：
  1) VERIFY_STATUS 的每一个取值都必须可被反推（不得漏项）；
  2) 反推后的强度不得低于标签所声明的强度。

运行：python tests/test_provenance.py
      python -m pytest tests/test_provenance.py -v
"""
import os as _os
import sys as _sys

_RP = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # 仓根
_sys.path.insert(0, _RP)

import json

from lingshu.core.provenance import VERIFY_STATUS, from_legacy

_TS = 1700000000.0


def _legacy(tags):
    return from_legacy(tags=json.dumps(tags, ensure_ascii=False),
                       created_at=_TS, condition_space="{}")


# ==================== ① 不得漏项 ====================

def test_every_verify_status_is_inferable():
    """VERIFY_STATUS 的全量取值都必须能从 legacy 标签反推出来。"""
    for s in VERIFY_STATUS:
        assert _legacy([s]).verify_status == s, f"{s} 无法被反推"


def test_verified_is_not_downgraded():
    """`verified` 不得被反推成 `unverified`（历史回归点）。"""
    assert _legacy(["verified"]).verify_status == "verified"


# ==================== ② 强度不得丢失 ====================

def test_strength_not_lost_for_whitebox_verified():
    """`verified` + 白箱校验 → 含 whitebox_code → 强度 strong。"""
    p = _legacy(["verified", "白箱校验"])
    assert p.verify_status == "verified"
    assert "whitebox_code" in p.verify_methods
    assert p.strength() == "strong"


def test_verified_without_strong_method_is_weak():
    p = _legacy(["verified"])
    assert p.verify_status == "verified"
    assert p.strength() == "weak"


# ==================== ③ 取更强的一个（顺序无关） ====================

def test_anchored_wins_over_verified():
    assert _legacy(["verified", "anchored"]).verify_status == "anchored"
    assert _legacy(["anchored", "verified"]).verify_status == "anchored"


# ==================== ④ 既有行为不变 ====================

def test_unverified_path_unchanged():
    """无验证标签 → 维持既有口径：unverified / strength == none。"""
    p = _legacy([])
    assert p.verify_status == "unverified"
    assert p.strength() == "none"


if __name__ == "__main__":
    import traceback

    failed = 0
    for _name in sorted(n for n in list(globals()) if n.startswith("test_")):
        _fn = globals()[_name]
        try:
            _fn()
            print(f"  PASS {_name}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL {_name}: {exc}")
            traceback.print_exc()
    print(f"\n{'FAILED' if failed else 'OK'} — {failed} failed")
    _sys.exit(1 if failed else 0)
