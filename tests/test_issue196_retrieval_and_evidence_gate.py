# -*- coding: utf-8 -*-
"""test_issue196_retrieval_and_evidence_gate · 检索与证据账本非有限/防 bool 闸（#196 第十三批）
====================================================================================
缺陷形态（改前基线 main 796ab89）：
  1. `lingshu/core/retrieval.py`:
     - `FTSIndex.blend(jaccard, bm25_pos)`：因 Python `max(nan, 0.0) == nan` 且
       `min(1.0, nan) == 1.0`，`blend(nan, 0.0)` 静默返回 `1.0`（直接击穿 M5
       `>= 0.95` 近重复去重阈值）；`blend(0.0, inf)` 因 `inf/(inf+10)=nan` 静默
       将 BM25 抹零；`blend(inf, 0.0) == 1.0`；`blend(True, False) == 1.0`；
       超大整数 `10**400` 抛裸 `OverflowError`。
     - `FTSIndex.temporal_neighbors(..., window=...)`：`window=nan` 使 `abs(dt)<=nan`
       恒假静默丢光邻居；`window=inf` 静默取消会话边界；`window=True`/`-5.0` 静默放行。
     - `FTSIndex.sync(batch=...)`：`batch=0` 静默返回 `0` 且残留脏队列不排空；
       `batch=True`/`-1`/`float`/非整数未校验。
  2. `lingshu/core/evidence.py`:
     - `_check_weight(True)` 因 `float(True) == 1.0` 静默放行 `bool`。
     - `EvidenceLedger.record(..., polarity=True, weight=True)` 因 `True in (1, -1)`
       与 `float(True) == 1.0` 静默写入一条 SUPPORT 证据并抬高 `nodes.confidence`；
       `polarity=1.0`/`-1.0` 亦绕过检查。
     - `EvidenceLedger.assess(...)`：当 `evidence` 表存在损坏的非有限/非正权重（如
       SQLite 写入 `9e999` → `+inf`）时，`(1+inf)/(2+inf)=nan` 绕过 `conf < REFUTED_BELOW`
       返回 `{'confidence': nan, 'status': 'corroborated'}`，并经 `sync()` 将 `NULL`
       写回 `nodes.confidence`。

断言组（G1–G5）：
  - G1 `FTSIndex.blend` 拒绝 `NaN`/`±inf`/`bool`/非数值/超大整型，有限有效输入逐位不变
  - G2 `FTSIndex.temporal_neighbors` 拒绝 `NaN`/`±inf`/`bool`/负数/非数值 `window`，合法对照不变
  - G3 `FTSIndex.sync` 拒绝 `0`/负数/`bool`/浮点/非整数 `batch` 且零副作用，正整数排空正常
  - G4 `_check_weight` 与 `EvidenceLedger.record` 拒绝非法 `polarity`/`weight`（含 `bool`/浮点）
       且零部分写入（`evidence` 表、`nodes.confidence`、`on_change` 回调、事务状态均不变）
  - G5 `EvidenceLedger.assess` 与 `sync` 拒绝损坏库权重（`±inf`/`0`/`<0`/文本/`bool`/求和溢出）
       而非传播 `NaN`；合法多源/同权反证/自产忽略回归不变

运行（仓根）：
  python -X utf8 -m pytest tests/test_issue196_retrieval_and_evidence_gate.py -v
  python -X utf8 tests/test_issue196_retrieval_and_evidence_gate.py
  python -X utf8 tests/test_issue196_retrieval_and_evidence_gate.py --mutate
  python -X utf8 tests/test_issue196_retrieval_and_evidence_gate.py --mutate=ALL
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.append(str(_REPO))

from lingshu.core.core import SpacetimeMemoryEngine              # noqa: E402
from lingshu.core.evidence import CONTRADICT, SUPPORT, _check_weight  # noqa: E402
from lingshu.core import retrieval as RT                         # noqa: E402

NAN = float("nan")
INF = float("inf")
NINF = float("-inf")
HUGE_INT = 10 ** 400


def _assert_raises_value_error(fn, *args, **kwargs):
    try:
        res = fn(*args, **kwargs)
    except ValueError as exc:
        return exc
    except Exception as exc:  # pragma: no cover
        raise AssertionError(
            "期望抛出 ValueError，实际抛出 %s: %s" % (type(exc).__name__, exc)
        ) from exc
    raise AssertionError("期望抛出 ValueError，但调用静默返回 %r" % (res,))


@contextmanager
def _fts_engine():
    prev_fts = os.environ.get("LINGSHU_FTS")
    os.environ["LINGSHU_FTS"] = "1"
    engine = SpacetimeMemoryEngine(":memory:")
    try:
        assert engine.store._fts is not None, "测试环境需启用 SQLite FTS5"
        yield engine
    finally:
        engine.close()
        if prev_fts is None:
            os.environ.pop("LINGSHU_FTS", None)
        else:
            os.environ["LINGSHU_FTS"] = prev_fts


def _ledger_engine():
    engine = SpacetimeMemoryEngine(":memory:")
    engine.enable_evidence_ledger()
    assert engine._evidence is not None
    return engine


def test_g1_blend_gate_and_regressions():
    """G1：FTSIndex.blend 拦截 NaN/±inf/bool/非数值/超大整数，并保持有限输入逐位不变。"""
    bad_vals = (NAN, INF, NINF, True, False, "0.5", None, 1 + 2j, [0.5], HUGE_INT)
    for bad in bad_vals:
        _assert_raises_value_error(RT.FTSIndex.blend, bad, 0.0)
        _assert_raises_value_error(RT.FTSIndex.blend, 0.0, bad)
    # 明确覆盖改前最典型绕过组合
    _assert_raises_value_error(RT.FTSIndex.blend, True, False)
    _assert_raises_value_error(RT.FTSIndex.blend, NAN, INF)

    # 合法有限输入回归（包括 bm25_pos <= 0 与 jaccard 夹断）
    assert RT.FTSIndex.blend(0.0, 0.0) == 0.0
    assert RT.FTSIndex.blend(0.0, -5.0) == 0.0
    assert RT.FTSIndex.blend(-0.5, 0.0) == 0.0
    assert math.isclose(RT.FTSIndex.blend(0.0, RT.BM25_K), RT.BM25_CAP * 0.5, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(RT.FTSIndex.blend(0, 10), RT.BM25_CAP * 0.5, rel_tol=0.0, abs_tol=1e-12)
    assert RT.FTSIndex.blend(0.96, 0.0) == 0.96
    assert RT.FTSIndex.blend(1.5, 0.0) == 1.0
    high_bm25 = RT.FTSIndex.blend(0.1, 1e9)
    assert 0.89 < high_bm25 <= RT.BM25_CAP < 0.95


def test_g2_temporal_neighbors_window_gate_and_regressions():
    """G2：FTSIndex.temporal_neighbors 拦截 NaN/±inf/bool/负数/非数值 window。"""
    with _fts_engine() as engine:
        fts = engine.store._fts
        n0 = engine.add_perception("第一条时序记录：北港码头", skip_dedup=True)
        n1 = engine.add_perception("第二条时序记录：灯塔守夜", skip_dedup=True)
        n2 = engine.add_perception("第三条时序记录：潮汐测绘", skip_dedup=True)
        engine.store.conn.execute(
            "UPDATE nodes SET temporal_coordinate = ? WHERE id = ?", (1000.0, n0.id)
        )
        engine.store.conn.execute(
            "UPDATE nodes SET temporal_coordinate = ? WHERE id = ?", (1000.0, n1.id)
        )
        engine.store.conn.execute(
            "UPDATE nodes SET temporal_coordinate = ? WHERE id = ?", (1005.0, n2.id)
        )
        engine.store.conn.commit()

        bad_windows = (NAN, INF, NINF, True, False, -1e-9, -5.0, "10", None, [10.0], HUGE_INT)
        for bad in bad_windows:
            _assert_raises_value_error(fts.temporal_neighbors, [n1.id], window=bad)
            # 空种子列表也必须先校验 window（不得静默返回 []）
            _assert_raises_value_error(fts.temporal_neighbors, [], window=bad)

        # 合法有限非负 window 回归
        within_10 = fts.temporal_neighbors([n1.id], window=10.0)
        assert sorted(row[0] for _, row in within_10) == sorted([n0.id, n2.id])

        same_tick = fts.temporal_neighbors([n1.id], window=0.0)
        assert [row[0] for _, row in same_tick] == [n0.id]

        within_4 = fts.temporal_neighbors([n1.id], window=4.0)
        assert [row[0] for _, row in within_4] == [n0.id]


def test_g3_sync_batch_gate_and_regressions():
    """G3：FTSIndex.sync 拦截非法 batch（0/负/bool/浮点/非整型）且零副作用，正整数正常排空。"""
    with _fts_engine() as engine:
        fts = engine.store._fts
        node = engine.add_perception("原始检索正文：初始索引条目", skip_dedup=True)
        # 绕过 add_node 裸 SQL 更新，在 nodes_fts_dirty 留下 1 条脏记录
        engine.store.conn.execute(
            "UPDATE nodes SET content = ? WHERE id = ?",
            ("更新后的检索正文：待同步目标", node.id),
        )
        engine.store.conn.commit()
        dirty_before = engine.store.conn.execute(
            "SELECT COUNT(*) FROM nodes_fts_dirty"
        ).fetchone()[0]
        assert dirty_before == 1

        bad_batches = (0, -1, -100, True, False, 1.0, 1.5, NAN, INF, NINF, "10", None, [1])
        for bad in bad_batches:
            _assert_raises_value_error(fts.sync, batch=bad)
            dirty_after = engine.store.conn.execute(
                "SELECT COUNT(*) FROM nodes_fts_dirty"
            ).fetchone()[0]
            assert dirty_after == 1, "非法 batch=%r 不应改变脏队列" % (bad,)

        # 合法 batch=1 排空脏队列并使新内容可检索
        synced = fts.sync(batch=1)
        assert synced == 1
        dirty_final = engine.store.conn.execute(
            "SELECT COUNT(*) FROM nodes_fts_dirty"
        ).fetchone()[0]
        assert dirty_final == 0
        hits = engine.store.search_content("待同步目标", limit=3)
        assert hits and hits[0][0].id == node.id

        # 合法超大 Python int（超过 SQLite 64 位上限）不抛 OverflowError
        assert fts.sync(batch=10 ** 25) == 0


def test_g4_evidence_record_and_check_weight_gate():
    """G4：_check_weight 与 EvidenceLedger.record 拒绝非法 polarity/weight 且零部分写入。"""
    for bad_w in (True, False, NAN, INF, NINF, 0.0, -0.5, 1.01, HUGE_INT, "nan", "inf", "x", None):
        _assert_raises_value_error(_check_weight, bad_w)
    assert _check_weight(1.0) == 1.0
    assert _check_weight(0.5) == 0.5
    assert _check_weight(1) == 1.0
    assert _check_weight("0.8") == 0.8

    engine = _ledger_engine()
    try:
        ledger = engine._evidence
        node = engine.add_perception("关键事实：北港潮位基准面为 2.4 米", source="sensor:tide")
        before_rep = ledger.assess(node.id)
        before_conf = engine.store.get_node(node.id).confidence
        before_count = engine.store.conn.execute(
            "SELECT COUNT(*) FROM evidence WHERE node_id = ?", (node.id,)
        ).fetchone()[0]
        assert before_count == 1
        assert before_rep["status"] == "single_source"

        cb_calls = []
        ledger.on_change = cb_calls.append

        # 非法 polarity（含 bool 与 float 1.0/-1.0）
        bad_polarities = (True, False, 1.0, -1.0, 0, 2, -2, "1", None, NAN, INF, NINF)
        for bad_pol in bad_polarities:
            _assert_raises_value_error(ledger.record, node.id, "src:bad_pol", polarity=bad_pol, weight=1.0)
            # 即使节点不存在，也应在开事务前以 ValueError 拦截而非 KeyError
            _assert_raises_value_error(ledger.record, "nonexistent_node", "src:bad_pol", polarity=bad_pol)

        # 改前最典型绕过：polarity=True, weight=True
        _assert_raises_value_error(ledger.record, node.id, "src:both_bool", polarity=True, weight=True)

        # 非法 weight（含 bool / NaN / ±inf / 越界 / 超大整数）
        bad_weights = (True, False, NAN, INF, NINF, 0.0, -0.1, 1.5, HUGE_INT, "bad", None)
        for bad_w in bad_weights:
            _assert_raises_value_error(ledger.record, node.id, "src:bad_w", polarity=SUPPORT, weight=bad_w)
            _assert_raises_value_error(ledger.record, "nonexistent_node", "src:bad_w", polarity=SUPPORT, weight=bad_w)

        # 零部分写入断言：evidence 行数、nodes.confidence、assess 状态、on_change 回调、事务状态均未变动
        after_count = engine.store.conn.execute(
            "SELECT COUNT(*) FROM evidence WHERE node_id = ?", (node.id,)
        ).fetchone()[0]
        assert after_count == before_count == 1
        assert engine.store.get_node(node.id).confidence == before_conf
        assert ledger.assess(node.id) == before_rep
        assert cb_calls == []
        assert not engine.store.conn.in_transaction

        # 正对照：合法 SUPPORT(1) 与 CONTRADICT(-1) 正常工作
        eid_s = ledger.record(node.id, "sensor:buoy", polarity=SUPPORT, weight=1.0)
        assert isinstance(eid_s, str) and len(eid_s) == 32
        rep_corr = ledger.assess(node.id)
        assert rep_corr["status"] == "corroborated"
        assert math.isclose(rep_corr["confidence"], 0.75, rel_tol=0.0, abs_tol=1e-6)
        assert cb_calls == [node.id]

        eid_c = ledger.record(node.id, "sensor:manual", polarity=CONTRADICT, weight=1.0)
        assert isinstance(eid_c, str) and len(eid_c) == 32
        rep_cont = ledger.assess(node.id)
        assert rep_cont["status"] == "contested"
        assert math.isclose(rep_cont["confidence"], 0.6, rel_tol=0.0, abs_tol=1e-6)
        assert cb_calls == [node.id, node.id]
    finally:
        engine.close()


def test_g5_evidence_assess_corrupt_db_weight_gate():
    """G5：EvidenceLedger.assess 与 sync 拦截损坏库权重，防 NaN 传播进 corroborated/confidence。"""
    corrupt_sql_weights = ("9e999", "-9e999", "0.0", "-0.5", "'nan'", "'inf'", "'corrupt'")
    for sql_w in corrupt_sql_weights:
        engine = _ledger_engine()
        try:
            ledger = engine._evidence
            node = engine.add_perception("待评估观测：" + sql_w, source="src:clean")
            conf_before = engine.store.get_node(node.id).confidence
            # 直接在 SQLite 注入损坏权重行（模拟绕过 record 的损坏持久化状态）
            engine.store.conn.execute(
                f"INSERT INTO evidence (id, node_id, source, polarity, weight, "
                f"self_generated, note, created_at) VALUES (?, ?, ?, ?, {sql_w}, 0, '', 1.0)",
                ("corrupt_" + sql_w, node.id, "src:corrupt", SUPPORT),
            )
            engine.store.conn.commit()
            _assert_raises_value_error(ledger.assess, node.id)
            _assert_raises_value_error(ledger.sync, node.id)
            # sync 被拒后不得把 NULL/NaN 写回 nodes.confidence
            assert engine.store.get_node(node.id).confidence == conf_before
        finally:
            engine.close()

    # 多个有限超大库权重求和溢出到 inf（包括 S 溢出、C 溢出及 S+C 分母溢出导致 conf 静默下溢为 0.0）
    overflow_cases = (
        ("support_sum_overflow", [(SUPPORT, 1e308), (SUPPORT, 1e308)]),
        ("symmetric_denom_overflow", [(SUPPORT, 1e308), (CONTRADICT, 1e308)]),
        ("contradict_sum_overflow", [(CONTRADICT, 1e308), (CONTRADICT, 1e308)]),
    )
    for case_name, extra_rows in overflow_cases:
        engine = _ledger_engine()
        try:
            ledger = engine._evidence
            node = engine.add_perception("求和溢出测试节点：" + case_name, source="src:s0")
            conf_before = engine.store.get_node(node.id).confidence
            for idx, (pol, w_val) in enumerate(extra_rows, start=1):
                engine.store.conn.execute(
                    "INSERT INTO evidence (id, node_id, source, polarity, weight, "
                    "self_generated, note, created_at) VALUES (?, ?, ?, ?, ?, 0, '', 1.0)",
                    (f"{case_name}_{idx}", node.id, f"src:huge{idx}", pol, w_val),
                )
            engine.store.conn.commit()
            _assert_raises_value_error(ledger.assess, node.id)
            _assert_raises_value_error(ledger.sync, node.id)
            assert engine.store.get_node(node.id).confidence == conf_before
        finally:
            engine.close()

    # entries() 返回 bool/None 权重也必须被 assess 拦截
    engine = _ledger_engine()
    try:
        ledger = engine._evidence
        node = engine.add_perception("替身 entries bool/None 权重测试", source="src:base")
        real_entries = ledger.entries
        for bad_w in (True, False, None, NAN, INF, 0.0, -1.0):
            ledger.entries = lambda _nid, _bw=bad_w: [
                {
                    "id": "mock1",
                    "source": "src:mock",
                    "polarity": SUPPORT,
                    "weight": _bw,
                    "self_generated": False,
                    "note": "",
                    "created_at": 1.0,
                }
            ]
            _assert_raises_value_error(ledger.assess, node.id)
        ledger.entries = real_entries

        # 正对照：同源同权反证胜出（#2533）与自产忽略保持不变
        ledger.record(node.id, "src:base", polarity=CONTRADICT, weight=1.0)
        ledger.record(node.id, "self:auto", polarity=SUPPORT, weight=1.0, self_generated=True)
        rep = ledger.assess(node.id)
        assert rep["status"] == "refuted"
        assert rep["self_generated_ignored"] == 1
        assert math.isclose(rep["confidence"], 1.0 / 3.0, rel_tol=0.0, abs_tol=1e-6)
    finally:
        engine.close()


_TESTS = (
    test_g1_blend_gate_and_regressions,
    test_g2_temporal_neighbors_window_gate_and_regressions,
    test_g3_sync_batch_gate_and_regressions,
    test_g4_evidence_record_and_check_weight_gate,
    test_g5_evidence_assess_corrupt_db_weight_gate,
)


def _run_direct() -> int:
    failed = 0
    for fn in _TESTS:
        try:
            fn()
            print("  PASS %s" % fn.__name__)
        except BaseException as exc:
            failed += 1
            print("  FAIL %s  %s: %s" % (fn.__name__, type(exc).__name__, exc))
    print("SUMMARY pass=%d fail=%d" % (len(_TESTS) - failed, failed))
    return 1 if failed else 0


_MUTATION_ARMS = {
    "M1_blend_gate": (
        Path("lingshu") / "core" / "retrieval.py",
        [(
            "    @staticmethod\n"
            "    def blend(jaccard: float, bm25_pos: float) -> float:\n"
            "        j = _require_finite_real(jaccard, \"jaccard\")\n"
            "        b = _require_finite_real(bm25_pos, \"bm25_pos\")\n"
            "        r = b / (b + BM25_K) if b > 0 else 0.0\n"
            "        return min(1.0, max(j, BM25_CAP * r))",
            "    @staticmethod\n"
            "    def blend(jaccard: float, bm25_pos: float) -> float:\n"
            "        r = bm25_pos / (bm25_pos + BM25_K) if bm25_pos > 0 else 0.0\n"
            "        return min(1.0, max(jaccard, BM25_CAP * r))",
        )],
        ["test_g1_blend_gate_and_regressions"],
    ),
    "M2_temporal_neighbors_gate": (
        Path("lingshu") / "core" / "retrieval.py",
        [(
            "        win = _require_finite_real(window, \"window\")\n"
            "        if win < 0.0:\n"
            "            raise ValueError(f\"window 必须是非负有限实数，收到 {window!r}\")",
            "        win = window",
        )],
        ["test_g2_temporal_neighbors_window_gate_and_regressions"],
    ),
    "M3_sync_batch_gate": (
        Path("lingshu") / "core" / "retrieval.py",
        [(
            "        batch = _require_positive_int(batch, \"batch\")\n"
            "        sql_limit = min(batch, _SQLITE_MAX_LIMIT)",
            "        sql_limit = batch",
        )],
        ["test_g3_sync_batch_gate_and_regressions"],
    ),
    "M4_check_weight_bool_gate": (
        Path("lingshu") / "core" / "evidence.py",
        [(
            "    if isinstance(weight, bool):\n"
            "        raise ValueError(f\"evidence weight 必须是数值（不能为 bool），收到 {weight!r}\")\n",
            "",
        )],
        ["test_g4_evidence_record_and_check_weight_gate"],
    ),
    "M5_record_polarity_gate": (
        Path("lingshu") / "core" / "evidence.py",
        [(
            "        if type(polarity) is not int or polarity not in (SUPPORT, CONTRADICT):",
            "        if polarity not in (SUPPORT, CONTRADICT):",
        )],
        ["test_g4_evidence_record_and_check_weight_gate"],
    ),
    "M6_assess_corrupt_weight_gate": (
        Path("lingshu") / "core" / "evidence.py",
        [(
            "            w_new = _check_stored_weight(e[\"weight\"])",
            "            w_new = e[\"weight\"]",
        ), (
            "        if not (math.isfinite(S) and math.isfinite(C)\n"
            "                and math.isfinite(numer) and math.isfinite(denom)):\n"
            "            raise ValueError(f\"evidence 聚合权重溢出非有限：S={S!r}, C={C!r}\")",
            "        pass",
        ), (
            "        if not math.isfinite(conf):\n"
            "            raise ValueError(f\"evidence 聚合置信度非有限：S={S!r}, C={C!r}\")",
            "        pass",
        )],
        ["test_g5_evidence_assess_corrupt_db_weight_gate"],
    ),
    "M7_assess_sum_overflow_gate": (
        Path("lingshu") / "core" / "evidence.py",
        [(
            "        if not (math.isfinite(S) and math.isfinite(C)\n"
            "                and math.isfinite(numer) and math.isfinite(denom)):\n"
            "            raise ValueError(f\"evidence 聚合权重溢出非有限：S={S!r}, C={C!r}\")",
            "        pass",
        )],
        ["test_g5_evidence_assess_corrupt_db_weight_gate"],
    ),
}


def _run_subtest(temp_root: Path) -> tuple[int, list[str], str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(temp_root) + os.pathsep + str(_REPO)
    proc = subprocess.run(
        [sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--assert-only"],
        cwd=str(temp_root),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    failed = [
        line.split()[1]
        for line in (proc.stdout or "").splitlines()
        if line.strip().startswith("FAIL ")
    ]
    return proc.returncode, failed, out


def _run_mutate(argv: list[str]) -> int:
    want = ""
    for arg in argv:
        if arg == "--mutate":
            want = want or "ALL"
        elif arg.startswith("--mutate="):
            want = arg.split("=", 1)[1]
    if not want:
        return 0
    ids = sorted(_MUTATION_ARMS) if want == "ALL" else [want]
    for mid in ids:
        if mid not in _MUTATION_ARMS:
            print("[UNKNOWN-ARM] %s" % mid)
            return 2

    tracked = [
        _REPO / "lingshu" / "core" / "retrieval.py",
        _REPO / "lingshu" / "core" / "evidence.py",
    ]
    before_sha = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}

    with tempfile.TemporaryDirectory(prefix="issue196_ret_ev_mut_") as td:
        tmp_root = Path(td)
        shutil.copytree(
            _REPO / "lingshu",
            tmp_root / "lingshu",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "assets"),
        )
        # 确认隔离副本优先被加载
        rc_chk = subprocess.run(
            [
                sys.executable, "-X", "utf8", "-c",
                "import pathlib, lingshu.core.retrieval as r, lingshu.core.evidence as e; "
                "root = pathlib.Path(%r).resolve(); "
                "assert pathlib.Path(r.__file__).resolve().is_relative_to(root), r.__file__; "
                "assert pathlib.Path(e.__file__).resolve().is_relative_to(root), e.__file__"
                % str(tmp_root),
            ],
            cwd=str(tmp_root),
            env={**os.environ, "PYTHONPATH": str(tmp_root) + os.pathsep + str(_REPO)},
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        assert rc_chk.returncode == 0, "隔离副本未优先加载: %s" % (rc_chk.stderr or rc_chk.stdout)

        all_ok = True
        rc0, failed0, out0 = _run_subtest(tmp_root)
        ok0 = (rc0 == 0 and failed0 == [])
        all_ok = all_ok and ok0
        print("  %-30s rc=%d(expect=0) failed=%s VERDICT=%s" % (
            "M0_clean_baseline", rc0, failed0, "PASS" if ok0 else "FAIL"
        ))
        if not ok0:
            print(out0)

        for mid in ids:
            rel, edits, expect_failed = _MUTATION_ARMS[mid]
            target = tmp_root / rel
            orig_bytes = target.read_bytes()
            try:
                norm = orig_bytes.decode("utf-8").replace("\r\n", "\n")
                for old, new in edits:
                    if norm.count(old) != 1:
                        print("[ANCHOR-MISS] %s (count=%d)" % (mid, norm.count(old)))
                        return 2
                    norm = norm.replace(old, new)
                target.write_text(norm, encoding="utf-8", newline="\n")
                # 清理副本 pyc 避免缓存干扰
                for pyc in (tmp_root / "lingshu" / "core" / "__pycache__").glob("*.pyc"):
                    pyc.unlink(missing_ok=True)
                rc_m, failed_m, out_m = _run_subtest(tmp_root)
                ok_m = (rc_m != 0 and failed_m == expect_failed)
                all_ok = all_ok and ok_m
                print("  %-30s rc=%d(expect=1) failed=%s(expect=%s) VERDICT=%s" % (
                    mid, rc_m, failed_m, expect_failed, "PASS" if ok_m else "FAIL"
                ))
                if not ok_m:
                    print(out_m)
            finally:
                target.write_bytes(orig_bytes)
                for pyc in (tmp_root / "lingshu" / "core" / "__pycache__").glob("*.pyc"):
                    pyc.unlink(missing_ok=True)

        rc_end, failed_end, out_end = _run_subtest(tmp_root)
        ok_end = (rc_end == 0 and failed_end == [])
        all_ok = all_ok and ok_end
        print("  %-30s rc=%d(expect=0) failed=%s VERDICT=%s" % (
            "M_restored_baseline", rc_end, failed_end, "PASS" if ok_end else "FAIL"
        ))
        if not ok_end:
            print(out_end)

    after_sha = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}
    untouched = (before_sha == after_sha)
    overall = all_ok and untouched
    print("OVERALL=%s (arms=%d repo_files_untouched=%s)" % (
        "PASS" if overall else "FAIL", len(ids), untouched
    ))
    return 0 if overall else 1


if __name__ == "__main__":
    args = sys.argv[1:]
    if any(a == "--mutate" or a.startswith("--mutate=") for a in args):
        raise SystemExit(_run_mutate(args))
    raise SystemExit(_run_direct())
