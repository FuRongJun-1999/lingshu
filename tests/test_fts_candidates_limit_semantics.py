# -*- coding: utf-8 -*-
"""Guard for ``FTSIndex.candidates(k=...)`` limit semantics.

Contract (aligned with ``#658`` / ``#1795`` / ``#1474`` and ``core._normalize_limit``):
  - ``k > 0``: return at most ``k`` hits ordered by BM25 descending.
  - ``k == 0``: return ``[]`` (still returns ``None`` if query has no usable bigrams;
    still checks FTS tracking epoch).
  - ``k is None`` or ``k < 0``: return all matching hits (SQLite ``LIMIT -1`` without
    Python negative slicing ``rows[:-1]`` that previously dropped trailing rows).
  - ``bool`` or any non-int type (``True``, ``False``, ``1.5``, ``nan``, ``inf``,
    ``"x"``, etc.): fail-loud ``TypeError`` naming ``k``.

Supports:
  - pytest: ``python -X utf8 -m pytest tests/test_fts_candidates_limit_semantics.py -q``
  - direct run (stdlib only): ``python -X utf8 tests/test_fts_candidates_limit_semantics.py``
  - mutation self-check: ``python -X utf8 tests/test_fts_candidates_limit_semantics.py --mutate``
"""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from lingshu.core.core import MemoryLayer, SpacetimeMemoryEngine  # noqa: E402


@contextmanager
def _fts_engine():
    prev_fts = os.environ.get("LINGSHU_FTS")
    os.environ["LINGSHU_FTS"] = "1"
    engine = SpacetimeMemoryEngine(":memory:")
    try:
        assert engine.store._fts is not None, "validation requires SQLite FTS5"
        yield engine
    finally:
        engine.close()
        if prev_fts is None:
            os.environ.pop("LINGSHU_FTS", None)
        else:
            os.environ["LINGSHU_FTS"] = prev_fts


def _seed_knowledge(engine: SpacetimeMemoryEngine, count: int = 5) -> None:
    for i in range(count):
        engine.add_perception(
            "needle knowledge item %d retrieval probe" % i,
            skip_dedup=True,
        )


def test_positive_and_zero_limits():
    with _fts_engine() as engine:
        _seed_knowledge(engine, 5)
        fts = engine.store._fts

        full = fts.candidates(["needle"], k=10)
        assert full is not None and len(full) == 5
        scores = [s for _, s in full]
        assert scores == sorted(scores, reverse=True), "BM25 order must be descending"

        top2 = fts.candidates(["needle"], k=2)
        assert len(top2) == 2
        assert [row[0] for row, _ in top2] == [row[0] for row, _ in full[:2]]

        top1 = fts.candidates(["needle"], k=1)
        assert len(top1) == 1
        assert top1[0][0][0] == full[0][0][0]

        assert fts.candidates(["needle"], k=0) == []
        assert fts.candidates(["？！。"], k=0) is None


def test_negative_and_none_limits_return_all_hits():
    with _fts_engine() as engine:
        _seed_knowledge(engine, 5)
        fts = engine.store._fts

        baseline = fts.candidates(["needle"], k=10)
        baseline_ids = [row[0] for row, _ in baseline]
        assert len(baseline_ids) == 5

        for k in (-1, -2, -5, -100, None):
            got = fts.candidates(["needle"], k=k)
            assert got is not None and len(got) == 5, (
                "k=%r must return all 5 hits, got %r" % (k, len(got) if got is not None else None)
            )
            assert [row[0] for row, _ in got] == baseline_ids, (
                "k=%r order/content diverged from full baseline" % (k,)
            )

        huge = fts.candidates(["needle"], k=10 ** 20)
        assert huge is not None and len(huge) == 5
        assert [row[0] for row, _ in huge] == baseline_ids


def test_layers_filter_with_negative_none_zero_and_positive_limits():
    with _fts_engine() as engine:
        for i in range(5):
            engine.add_perception(
                "needle knowledge item %d retrieval probe" % i,
                skip_dedup=True,
            )
        for i in range(5):
            engine.add_context("needle context item %d retrieval probe" % i)
        # Extra KNOWLEDGE nodes so positive k=2 with layers=[CONTEXT] exercises inner expansion
        for i in range(20):
            engine.add_perception(
                "needle needle needle knowledge burst %d" % i,
                skip_dedup=True,
            )

        fts = engine.store._fts
        layer_col = [
            r[1] for r in engine.store.conn.execute("PRAGMA table_info(nodes)").fetchall()
        ].index("layer")

        for target_layer, expected_count in (
            (MemoryLayer.KNOWLEDGE, 25),
            (MemoryLayer.CONTEXT, 5),
            (MemoryLayer.SELF, 0),
        ):
            for k in (-1, -2, -10, None):
                got = fts.candidates(["needle"], layers=[target_layer], k=k)
                assert got is not None and len(got) == expected_count, (
                    "layer=%s k=%r expected %d, got %d"
                    % (target_layer.value, k, expected_count, len(got))
                )
                assert all(row[layer_col] == target_layer.value for row, _ in got)

            assert fts.candidates(["needle"], layers=[target_layer], k=0) == []

        ctx_top2 = fts.candidates(["needle"], layers=[MemoryLayer.CONTEXT], k=2)
        assert len(ctx_top2) == 2
        assert all(row[layer_col] == MemoryLayer.CONTEXT.value for row, _ in ctx_top2)


def test_bool_and_non_integer_k_fail_loud():
    with _fts_engine() as engine:
        _seed_knowledge(engine, 3)
        fts = engine.store._fts

        bad_values = (
            True,
            False,
            1.5,
            0.0,
            float("nan"),
            float("inf"),
            -float("inf"),
            "5",
            "x",
            [1],
        )
        for bad in bad_values:
            for query in (["needle"], ["？！。"]):
                try:
                    fts.candidates(query, k=bad)
                except TypeError as exc:
                    assert "k" in str(exc), "TypeError message must name k: %s" % (exc,)
                else:
                    raise AssertionError(
                        "candidates(%r, k=%r) failed to raise TypeError" % (query, bad)
                    )


def test_zero_limit_preserves_lifecycle_epoch_check():
    with _fts_engine() as engine:
        _seed_knowledge(engine, 2)
        fts = engine.store._fts
        engine.store.conn.execute("DROP TRIGGER nodes_fts_ai")
        try:
            fts.candidates(["needle"], k=0)
        except sqlite3.OperationalError:
            pass
        else:
            raise AssertionError(
                "candidates(..., k=0) bypassed FTS lifecycle tracking check"
            )


_TESTS = (
    test_positive_and_zero_limits,
    test_negative_and_none_limits_return_all_hits,
    test_layers_filter_with_negative_none_zero_and_positive_limits,
    test_bool_and_non_integer_k_fail_loud,
    test_zero_limit_preserves_lifecycle_epoch_check,
)


def _run_assertions() -> int:
    fails = 0
    for fn in _TESTS:
        try:
            fn()
            print("  PASS %s" % fn.__name__)
        except BaseException as exc:
            fails += 1
            print("  FAIL %s  %s: %s" % (fn.__name__, type(exc).__name__, exc))
    print("passed %d / failed %d" % (len(_TESTS) - fails, fails))
    return fails


_MUTATION_ARMS = {
    "M1": (
        "lingshu/core/retrieval.py",
        [(
            "    if k is None or k < 0:\n        return None",
            "    if False:\n        return None",
        )],
        {
            "test_negative_and_none_limits_return_all_hits",
            "test_layers_filter_with_negative_none_zero_and_positive_limits",
        },
    ),
    "M2": (
        "lingshu/core/retrieval.py",
        [(
            "    if k is not None and (isinstance(k, bool) or not isinstance(k, int)):",
            "    if False:",
        )],
        {
            "test_bool_and_non_integer_k_fail_loud",
        },
    ),
    "M3": (
        "lingshu/core/retrieval.py",
        [(
            "        selected = rows if want is None else rows[:want]",
            "        selected = rows[:-1] if want is None else rows[:want]",
        )],
        {
            "test_negative_and_none_limits_return_all_hits",
            "test_layers_filter_with_negative_none_zero_and_positive_limits",
        },
    ),
}


def _subprocess_fail_names():
    for cache in (_REPO / "lingshu").rglob("__pycache__"):
        for pyc in cache.glob("*.pyc"):
            pyc.unlink(missing_ok=True)
    proc = subprocess.run(
        [sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--assert-only"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(_REPO),
        timeout=300,
    )
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return set(re.findall(r"(?m)^\s*FAIL\s+(\S+)", out)), out


def _mutate(argv) -> int:
    want = ""
    for arg in argv:
        if arg == "--mutate":
            want = want or "ALL"
        elif arg.startswith("--mutate="):
            want = arg.split("=", 1)[1]
    if not want:
        return 0
    ids = sorted(_MUTATION_ARMS) if want == "ALL" else [want]
    red = 0
    for mid in ids:
        if mid not in _MUTATION_ARMS:
            print("[UNKNOWN-ARM] %s" % mid)
            return 2
        rel, edits, expect = _MUTATION_ARMS[mid]
        target = _REPO / rel
        orig = target.read_bytes()
        try:
            text = orig.decode("utf-8")
            nl = "\r\n" if "\r\n" in text else "\n"
            norm = text.replace("\r\n", "\n")
            for old, new in edits:
                if norm.count(old) != 1:
                    print("[ANCHOR-MISS] %s (count=%d)" % (mid, norm.count(old)))
                    return 2
                norm = norm.replace(old, new)
            target.write_bytes(norm.replace("\n", nl).encode("utf-8"))
            actual, out = _subprocess_fail_names()
            if actual == expect:
                print("[MUTATE-OK] %s %s" % (mid, sorted(actual)))
            else:
                print(
                    "[MUTATE-FAIL] %s expect=%s got=%s"
                    % (mid, sorted(expect), sorted(actual))
                )
                print(out)
                red += 1
        finally:
            target.write_bytes(orig)
    print("mutate: %d, %s" % (len(ids), "all-hit" if red == 0 else "%d missing" % red))
    return 1 if red else 0


if __name__ == "__main__":
    fails = _run_assertions()
    if "--assert-only" in sys.argv:
        sys.exit(1 if fails else 0)
    sys.exit(1 if (fails or _mutate(sys.argv[1:])) else 0)
