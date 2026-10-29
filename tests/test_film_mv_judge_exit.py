# -*- coding: utf-8 -*-
"""MV 判据闸的退出码契约 —— `--judge` 必须把「不可放行」透出到进程退出码

运行方式：
    python -X utf8 -m pytest tests/test_film_mv_judge_exit.py -v

缺陷（本守卫防的是什么）：
    `make.py` 的 `--judge` 分支只打印 `tally` 后返回 0——判据出 REJECT / BLINDSPOT
    时进程退出码仍是 0。README §一 把 `--judge` 定位成"不重渲就能跑"的判据入口，
    于是任何拿退出码判闸的流水线都会**假绿**：判据有结论，闸不拦。

    同一处还有第二份口径：`mv.summarize()` 的 `blocking` 只认 REJECT，而仓内
    `film.oracle.gates_passed()` 的放行规则是"无 REJECT 且无 BLINDSPOT"，
    `_det/gate_all.summarize()` 也按后者的口径。两套口径并存 ⇒ 判据层说"拦"、
    CLI 层说"过"。

断言组（抽掉修复 ⇒ 哪条红）：
    [A] 可放行的裁定 ⇒ 退出码 0                   退回 `return 0` ⇒ B 不红、A 仍绿（故须 B 兜底）
    [B] BLINDSPOT 的裁定 ⇒ 退出码 1 且 blocking 点名  退回 `return 0` ⇒ B 红
    [C] summarize 与 film.oracle.gates_passed 同源   只认 REJECT ⇒ C 红
"""
import importlib.util
import json
import os
import sys

_HERE = os.path.abspath(os.path.dirname(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
_PKG_ROOT = os.path.abspath(os.path.join(_REPO, "bridges", "blender"))
for _p in (_REPO, _PKG_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest  # noqa: E402

from wm_bridge.factory.film import mv                                   # noqa: E402
from wm_bridge.factory.film import oracle as FO                         # noqa: E402
from wm_bridge.factory.film.mv import make as MAKE                      # noqa: E402
from wm_bridge.factory.film.mv._det import gates as MVG                 # noqa: E402


def _sibling_tests():
    """复用 `test_film_mv_plugin._bundle()` 的输入束，避免第二套夹具漂移。"""
    path = os.path.join(_HERE, "test_film_mv_plugin.py")
    spec = importlib.util.spec_from_file_location("_mv_plugin_tests", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _oracle(bundle, oracle_id="oracle_main"):
    return MVG.build_oracle(bundle, oracle_id, "mv-judge-contract", "a" * 64)


def _releasable(doc):
    """只保留 ACCEPT 的裁定：用于「可放行 ⇒ 退出码 0」这一侧。

    默认夹具（以及默认简报的小样）在 K2/K3/K5 上是 BLINDSPOT，按仓内放行规则本就
    不可放行 —— 那是判据该有的结论，不是本守卫要修的东西。这里只切出可放行的子集，
    两侧都仍走 `film.oracle` 的契约校验，不自造文档形状。
    """
    out = dict(doc)
    out["checks"] = [c for c in doc["checks"] if c["verdict"] == "ACCEPT"]
    return out


def _tally(doc):
    out = {}
    for c in doc["checks"]:
        out[c["verdict"]] = out.get(c["verdict"], 0) + 1
    return out


@pytest.fixture(scope="module")
def docs():
    sib = _sibling_tests()
    ok = _releasable(_oracle(sib._bundle()))
    blind = _oracle(sib._bundle(axis=False))     # 主体沿视线前行 ⇒ K1 按设计 BLINDSPOT
    return ok, blind


@pytest.fixture(scope="module")
def judge_stub(docs):
    ok, blind = docs
    return {"ok": {"oracle": ok, "tally": _tally(ok)},
            "blind": {"oracle": blind, "tally": _tally(blind)}}


# ══════════════════════════════════════════════════ [A] 可放行 ⇒ 0
def test_a_releasable_oracle_exits_zero(monkeypatch, tmp_path, judge_stub):
    ok, _ = FO.gates_passed(judge_stub["ok"]["oracle"])
    assert ok, "夹具前提：该输入束应可放行"
    monkeypatch.setattr(mv, "judge", lambda **kw: judge_stub["ok"])
    assert MAKE.main(["--judge", "--out", str(tmp_path)]) == 0


# ══════════════════════════════════════════════════ [B] BLINDSPOT ⇒ 1
def test_b_blindspot_oracle_exits_one(monkeypatch, tmp_path, judge_stub):
    ok, bad = FO.gates_passed(judge_stub["blind"]["oracle"])
    assert not ok and bad, "夹具前提：该输入束应被 BLINDSPOT 拦下"
    monkeypatch.setattr(mv, "judge", lambda **kw: judge_stub["blind"])
    assert MAKE.main(["--judge", "--out", str(tmp_path)]) == 1


def test_b2_cli_reports_which_checks_block(monkeypatch, tmp_path, judge_stub, capsys):
    monkeypatch.setattr(mv, "judge", lambda **kw: judge_stub["blind"])
    assert MAKE.main(["--judge", "--out", str(tmp_path)]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["gates_passed"] is False
    assert payload["blocking"], "不可放行时必须点名是哪几条判据"
    assert payload["tally"] == judge_stub["blind"]["tally"], "tally 仍须原样透出"


# ══════════════════════════════════════════════════ [C] 口径同源
def test_c_summarize_shares_gates_passed(docs):
    ok_doc, blind_doc = docs
    for doc in (ok_doc, blind_doc):
        ok, bad = FO.gates_passed(doc)
        s = mv.summarize(doc)
        assert s["gates_passed"] is ok
        assert s["blocking"] == bad
        assert s["counts"] == FO.counts_by_verdict(doc)


def test_c2_blindspot_is_blocking(docs):
    _, blind_doc = docs
    s = mv.summarize(blind_doc)
    verdicts = {c["name"]: c["verdict"] for c in blind_doc["checks"]}
    blind_names = sorted(n for n, v in verdicts.items() if v == "BLINDSPOT")
    assert blind_names, "夹具前提：应有 BLINDSPOT 判据"
    assert set(blind_names) <= set(s["blocking"]), "BLINDSPOT 必须计入 blocking"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
