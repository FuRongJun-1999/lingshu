"""A context write uses the persisted capacity and rolls back as one unit."""
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import pytest

from lingshu.core.core import MemoryLayer, SpacetimeMemoryEngine


pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture
def engine(tmp_path, monkeypatch):
    for key in ("LINGSHU_EVIDENCE_LEDGER", "LINGSHU_TMS"):
        monkeypatch.delenv(key, raising=False)
    value = SpacetimeMemoryEngine(str(tmp_path / "context.db"))
    try:
        yield value
    finally:
        value.close()


def _seed(engine, count=2):
    return [engine.add_context(f"original context {i}").id for i in range(count)]


def _count(engine):
    return engine.store.count_layer(MemoryLayer.CONTEXT)


def _reject_trim(engine):
    engine.store.conn.execute("CREATE TRIGGER reject_context_trim BEFORE DELETE ON nodes "
        "BEGIN SELECT RAISE(ABORT, 'context trim interrupted'); END")


@pytest.mark.parametrize("old_cap,new_cap,count", [(2, 5, 4), (5, 2, 2)], ids=["increased", "decreased"])
def test_existing_instance_uses_current_database_capacity(engine, old_cap, new_cap, count):
    engine.set_context_cap(old_cap)
    other = SpacetimeMemoryEngine(engine.store.db_path)
    try:
        engine.set_context_cap(new_cap)
        original = _seed(engine, count)
        added = other.add_context("write from an already-open second instance")
        expected = min(count + 1, new_cap)
        assert _count(engine) == expected, "context write used a stale or default capacity"
        assert engine.store.get_node(added.id) is not None
        retained = original[-(new_cap - 1):] if new_cap > 1 else []
        assert all(engine.store.get_node(nid) is not None for nid in retained), "current capacity lost retained context"
        assert engine.store.get_meta("context_max")["context_max"] == str(new_cap)
    finally:
        other.close()


def test_existing_process_keeps_context_after_capacity_increase(engine, tmp_path):
    engine.set_context_cap(2)
    program = '''import json, sys, warnings
from pathlib import Path
warnings.simplefilter("ignore")
from lingshu.core.core import MemoryLayer, SpacetimeMemoryEngine
engine = SpacetimeMemoryEngine(sys.argv[1])
try:
    Path(sys.argv[2]).write_text("ready", encoding="utf-8")
    if sys.stdin.readline().strip() != "write":
        raise RuntimeError("missing test command")
    node = engine.add_context("context from a second real process")
    print(json.dumps({"count": engine.store.count_layer(MemoryLayer.CONTEXT), "node": node.id}), flush=True)
finally:
    engine.close()
'''
    repo = Path(__file__).resolve().parents[1]
    ready_path = tmp_path / "process-ready"
    # Constructor diagnostics can exceed a Windows pipe buffer before ready.
    # Keep stderr in a file and bound readiness independently of stdout.
    diagnostic = (tmp_path / "process-stderr.log").open("w+", encoding="utf-8")
    process = subprocess.Popen([sys.executable, "-B", "-X", "utf8", "-u", "-c", program,
        engine.store.db_path, str(ready_path)],
        cwd=repo, env={**os.environ, "PYTHONPATH": str(repo), "PYTHONDONTWRITEBYTECODE": "1"},
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=diagnostic, text=True, encoding="utf-8")
    try:
        deadline = time.monotonic() + 15
        while not ready_path.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        diagnostic.seek(0)
        assert ready_path.exists(), diagnostic.read()
        engine.set_context_cap(5)
        original = _seed(engine, 4)
        stdout, _ = process.communicate("write\n", timeout=15)
        diagnostic.seek(0)
        assert process.returncode == 0, diagnostic.read()
        result = json.loads(stdout)
        assert result["count"] == 5, "already-open process erased context using its old capacity"
        assert all(engine.store.get_node(nid) is not None for nid in original)
        assert engine.store.get_node(result["node"]) is not None
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
        for pipe in (process.stdin, process.stdout, process.stderr):
            if pipe is not None:
                pipe.close()
        diagnostic.close()


def test_fifo_failure_undoes_the_new_context_on_disk(engine):
    engine.set_context_cap(2)
    original = _seed(engine)
    _reject_trim(engine)
    with pytest.raises(sqlite3.IntegrityError, match="context trim interrupted"):
        engine.add_context("failed context write")
    with closing(sqlite3.connect(engine.store.db_path)) as observer:
        count = observer.execute("SELECT COUNT(*) FROM nodes WHERE layer='context'").fetchone()[0]
    assert count == 2, "failed context write was committed before FIFO failed"
    assert all(engine.store.get_node(nid) is not None for nid in original)
    assert not engine.store.conn.in_transaction, "failed context write left an active transaction"


def test_fifo_failure_undoes_only_this_call_in_a_caller_transaction(engine):
    engine.set_context_cap(2)
    original = _seed(engine)
    _reject_trim(engine)
    engine.store.conn.execute("BEGIN IMMEDIATE")
    try:
        engine.store.update_node_importance(original[1], 0.33)
        assert engine.store.get_node(original[1]).importance == pytest.approx(0.73)
        with pytest.raises(sqlite3.IntegrityError, match="context trim interrupted"):
            engine.add_context("failed write inside caller transaction")
        assert engine.store.conn.in_transaction, "context write ended the caller transaction"
        assert _count(engine) == 2, "failed context call left a partial write in caller transaction"
        assert engine.store.get_node(original[1]).importance == pytest.approx(0.73)
        with closing(sqlite3.connect(engine.store.db_path)) as observer:
            assert observer.execute("SELECT importance FROM nodes WHERE id=?", (original[1],)).fetchone()[0] == pytest.approx(0.4)
    finally:
        engine.store.conn.rollback()
    assert engine.store.get_node(original[1]).importance == pytest.approx(0.4)


def test_successful_context_call_can_be_rolled_back_by_caller(engine):
    engine.set_context_cap(2)
    original = _seed(engine)
    engine.store.conn.execute("BEGIN IMMEDIATE")
    try:
        added = engine.add_context("pending caller context")
        assert _count(engine) == 2, "successful context write did not trim FIFO inside caller transaction"
        assert engine.store.get_node(original[0]) is None
        assert engine.store.get_node(added.id) is not None
        assert engine.store.conn.in_transaction, "context write committed its caller"
    finally:
        engine.store.conn.rollback()
    assert all(engine.store.get_node(nid) is not None for nid in original)
    assert engine.store.get_node(added.id) is None


def test_float_capacity_and_existing_protection_policy_are_retained(engine):
    engine.set_context_cap(2.0)
    original = _seed(engine)
    engine.store.protect_node(original[0], "keep original context")
    added = engine.add_context("new context after protected prefix")
    assert _count(engine) == 3
    assert all(engine.store.get_node(nid) is not None for nid in [*original, added.id])
