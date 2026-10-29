# -*- coding: utf-8 -*-
"""MV 判据入口的退出码契约 —— `--judge` 与 `--accept` 同口径，`--strict` 才是发布放行

运行方式：
    python -X utf8 -m pytest tests/test_film_mv_judge_exit.py -q

缺陷（本守卫防的是什么）：
    `make.py` 的 `--judge` 分支只打印 `tally` 后返回 0——判据出 REJECT 时进程退出码
    仍是 0。README §一 把 `--judge` 定位成"不重渲就能跑"的判据入口，于是拿退出码
    判闸的流水线会**假绿**：判据有结论，闸不拦。

两档口径必须分开（仓内既有，不是本件的发明）：

  · `_det/acceptance.py` E 组：失败条件只有 REJECT；K2 的 BLINDSPOT 是**必须如实
    出现**才过（默认简报在 K2/K3/K5 上本就 BLINDSPOT）。
  · `film/oracle.py:gates_passed()`：**发布放行**要"无 REJECT 且无 BLINDSPOT"。

所以 `--judge` 用第一档（与 `--accept` 同结论），`--strict` 用第二档。若把发布放行
硬塞进 `--judge`，默认简报的小样会恒判红，而同一目录 `--accept` 判定通过——同一支
片子两个结论，正是 `_det/gates.py` 开头警告的那件事。

断言组（抽掉修复 ⇒ 哪条红）：
    [A] 可放行（只有 ACCEPT）⇒ --judge 退出码 0
    [B] 有 BLINDSPOT、无 REJECT ⇒ --judge 0（与验收同口径）；--strict 1
    [C] 有 REJECT ⇒ --judge 1 且点名；--strict 1
    [D] summarize 的两档各自与 gates_passed / REJECT 名单一致
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


def _only_accept(doc):
    """只保留 ACCEPT 的裁定：可放行样本。"""
    out = dict(doc)
    out["checks"] = [c for c in doc["checks"] if c["verdict"] == "ACCEPT"]
    return out


def _with_reject(doc):
    """把一个 ACCEPT 判据改成 REJECT（保留 measured/threshold ⇒ 仍是合法裁定）。

    这里测的是 CLI 的退出码映射，不是判据本身能否判红：判据的判红能力由
    `tests/test_film_mv_plugin.py` 与 gates_k/gates_e 自己的用例覆盖。
    """
    checks = [dict(c) for c in doc["checks"]]
    for c in checks:
        if c["verdict"] == "ACCEPT":
            c["verdict"] = "REJECT"
            break
    else:                                                      # pragma: no cover
        raise AssertionError("夹具前提：样本里应有 ACCEPT 判据可改")
    out = dict(doc)
    out["checks"] = checks
    return out


def _tally(doc):
    out = {}
    for c in doc["checks"]:
        out[c["verdict"]] = out.get(c["verdict"], 0) + 1
    return out


@pytest.fixture(scope="module")
def docs():
    sib = _sibling_tests()
    blind = _oracle(sib._bundle(axis=False))    # K1 按设计 BLINDSPOT，其余是默认夹具票型
    return {"blind": blind,
            "releasable": _only_accept(blind),
            "reject": _with_reject(blind)}


@pytest.fixture(scope="module")
def stubs(docs):
    return {k: {"oracle": v, "tally": _tally(v)} for k, v in docs.items()}


def _run(monkeypatch, stubs, key, tmp_path, *flags):
    monkeypatch.setattr(mv, "judge", lambda **kw: stubs[key])
    return MAKE.main([*flags, "--out", str(tmp_path)])


# ══════════════════════════════════════════ [A] 可放行 ⇒ 0
def test_a_releasable_exits_zero(monkeypatch, tmp_path, stubs):
    assert _run(monkeypatch, stubs, "releasable", tmp_path, "--judge") == 0


# ══════════════════════════════════════════ [B] 只有 BLINDSPOT ⇒ judge 0、strict 1
def test_b_blindspot_only_is_not_a_judge_failure(monkeypatch, tmp_path, docs, stubs):
    ok, _ = FO.gates_passed(docs["blind"])
    assert not ok, "夹具前提：该样本在发布放行档下应被拦"
    assert _run(monkeypatch, stubs, "blind", tmp_path, "--judge") == 0, (
        "BLINDSPOT 不是验收失败（acceptance E 组只把 REJECT 当失败）")
    assert _run(monkeypatch, stubs, "blind", tmp_path, "--strict") == 1, (
        "发布放行档必须拦 BLINDSPOT")


# ══════════════════════════════════════════ [C] 有 REJECT ⇒ 1
def test_c_reject_exits_one(monkeypatch, tmp_path, docs, stubs):
    reject = FO.failed_checks(docs["reject"])
    assert reject, "夹具前提：样本里应有 REJECT"
    assert _run(monkeypatch, stubs, "reject", tmp_path, "--judge") == 1
    assert _run(monkeypatch, stubs, "reject", tmp_path, "--strict") == 1


def test_c2_payload_names_blocking_checks(monkeypatch, tmp_path, docs, stubs, capsys):
    expected = FO.failed_checks(docs["reject"])
    assert _run(monkeypatch, stubs, "reject", tmp_path, "--judge") == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["blocking"] == expected
    assert payload["tally"] == stubs["reject"]["tally"], "tally 仍须原样透出"
    assert payload["gates_passed"] is False
    assert payload["strict"] is False


# ══════════════════════════════════════════ [D] 两档各连各的口径
def test_d_summarize_lanes_match_repo_rules(docs):
    for key, doc in docs.items():
        s = mv.summarize(doc)
        assert s["blocking"] == FO.failed_checks(doc), "%s：judge 档应为 REJECT 名单" % key
        assert s["counts"] == FO.counts_by_verdict(doc)
        ok, release_bad = FO.gates_passed(doc)
        assert s["gates_passed"] is ok, "%s：strict 档应与 gates_passed 一致" % key
        assert s["release_blocking"] == release_bad


def test_d2_strict_lane_names_blindspot(docs):
    s = mv.summarize(docs["blind"])
    blind_names = sorted(c["name"] for c in docs["blind"]["checks"]
                         if c["verdict"] == "BLINDSPOT")
    assert blind_names, "夹具前提：应有 BLINDSPOT 判据"
    assert set(blind_names) <= set(s["release_blocking"])
    assert not (set(blind_names) & set(s["blocking"])), "BLINDSPOT 不该进 --judge 档"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
