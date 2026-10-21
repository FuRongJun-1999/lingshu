"""Public export entry points must preserve their own active SQLite files.

All cases use disposable files. The ordinary backup overwrite contract stays on.
"""
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lingshu.core.core import SpacetimeMemoryEngine  # noqa: E402


@pytest.fixture
def database(tmp_path, monkeypatch):
    monkeypatch.delenv("LINGSHU_EXPORT_OUT_ROOT", raising=False)
    path = tmp_path / "source.db"
    engine = SpacetimeMemoryEngine(str(path))
    node = engine.add_perception("synthetic durable source fact", skip_dedup=True)
    engine.store.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    try:
        yield engine, path, node.id
    finally:
        # A deliberately unfixed baseline may have destroyed this disposable DB.
        try:
            engine.close()
        except sqlite3.DatabaseError:
            pass


def _export(engine, method, target):
    if method == "all":
        return engine.export_all(str(target))
    if method == "structured":
        return engine.export_structured_trajectories(output_path=str(target))
    return engine.export_trajectory(output_path=str(target))


def _reject(engine, method, target, label):
    try:
        _export(engine, method, target)
    except ValueError as exc:
        assert "SQLite" in str(exc), f"{label} rejection must identify source protection"
    except OSError as exc:
        raise AssertionError(f"{label} source path was not rejected before filesystem write") from exc
    else:
        raise AssertionError(f"{label} source path was not rejected before filesystem write")


def _healthy(path, node_id):
    assert path.read_bytes()[:16] == b"SQLite format 3\x00"
    with closing(sqlite3.connect(path, timeout=0.1)) as observer:
        assert observer.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert observer.execute("SELECT id FROM nodes WHERE id=?", (node_id,)).fetchone()


@pytest.mark.parametrize("method", ["all", "trajectory", "structured"])
def test_source_database_rejected_before_write(database, method):
    """G1: default configuration must not let an export replace/truncate its DB."""
    engine, path, node_id = database
    before = path.read_bytes()
    _reject(engine, method, path, "G1")
    assert path.read_bytes() == before
    assert not list(path.parent.glob("*.tmp-*"))
    assert not engine.store.conn.in_transaction
    _healthy(path, node_id)


@pytest.mark.parametrize("method", ["all", "trajectory"])
@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_source_sidecars_rejected_even_if_absent(database, method, suffix):
    """G2: live WAL/SHM and the currently absent rollback journal are protected."""
    engine, path, node_id = database
    target = Path(str(path) + suffix)
    before = target.read_bytes() if target.exists() else None
    _reject(engine, method, target, "G2")
    assert (target.read_bytes() if target.exists() else None) == before
    _healthy(path, node_id)


def test_source_inside_allowed_root_is_still_rejected(database, monkeypatch):
    """G3: opting into an export root does not opt into database destruction."""
    engine, path, node_id = database
    monkeypatch.setenv("LINGSHU_EXPORT_OUT_ROOT", str(path.parent))
    _reject(engine, "all", path, "G3")
    _healthy(path, node_id)


def test_source_hardlink_alias_is_rejected(database):
    """G4: realpath alone cannot identify a hard link used by direct writing."""
    engine, path, node_id = database
    target = path.with_name("source-alias.json")
    try:
        os.link(path, target)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"hard links unavailable: {exc}")
    _reject(engine, "trajectory", target, "G4")
    _healthy(path, node_id)
    assert os.path.samefile(path, target)


def test_source_symlink_alias_is_rejected(database):
    """G5: a symlink to the main DB is not an ordinary backup destination."""
    engine, path, node_id = database
    target = path.with_name("source-link.json")
    try:
        target.symlink_to(path)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symbolic links unavailable: {exc}")
    _reject(engine, "all", target, "G5")
    assert target.is_symlink()
    _healthy(path, node_id)


def test_connection_path_survives_working_directory_change(tmp_path, monkeypatch):
    """G6: the active file, not the current interpretation of db_path, is protected."""
    monkeypatch.delenv("LINGSHU_EXPORT_OUT_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    with closing(SpacetimeMemoryEngine("relative.db")) as engine:
        other = tmp_path / "other"
        other.mkdir()
        monkeypatch.chdir(other)
        _reject(engine, "all", tmp_path / "relative.db", "G6")


def test_attached_database_is_protected(database):
    """G7: all file-backed databases actually used by this connection are sources."""
    engine, path, node_id = database
    attached = path.with_name("attached.db")
    engine.store.conn.execute("ATTACH DATABASE ? AS aux", (str(attached),))
    engine.store.conn.execute("CREATE TABLE aux.synthetic (value TEXT)")
    before = attached.read_bytes()
    _reject(engine, "all", attached, "G7")
    assert attached.read_bytes() == before
    _healthy(path, node_id)


@pytest.mark.parametrize("method", ["all", "trajectory"])
def test_normal_backup_overwrite_contract_is_preserved(database, method):
    """G8 positive control: normal JSON backups can still be created/overwritten."""
    engine, path, node_id = database
    target = path.with_name("backup.json")
    first = _export(engine, method, target)
    assert first["overwrote_existing"] is False
    engine.add_context("synthetic later fact")
    second = _export(engine, method, target)
    assert second["overwrote_existing"] is True, "G8 normal backup overwrite was disabled"
    content = json.loads(target.read_text(encoding="utf-8"))
    if method == "all":
        assert len(content["nodes"]) == second["exported_nodes"] == 2
    else:
        assert "cycles" in content
    _healthy(path, node_id)


def test_memory_database_can_export_to_normal_file(tmp_path):
    """G9 positive control: an in-memory DB has no source file to forbid."""
    with closing(SpacetimeMemoryEngine(":memory:")) as engine:
        result = engine.export_all(str(tmp_path / "memory-backup.json"))
        assert result["exported_nodes"] == 0
        assert (tmp_path / "memory-backup.json").exists()
