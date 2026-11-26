# -*- coding: utf-8 -*-
"""Regression & isolated-sandbox mutation guards for structure_permissions/service/writer."""
from __future__ import annotations

import json
import os
import shutil
import ssl
import stat as stat_mod
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

_REPO = Path(__file__).resolve().parent.parent


def _mk_writer(tmp_path: Path):
    from lingshu.core.core import LayeredStore, Role
    from lingshu.core.structure_writer import StructureWriter

    store = LayeredStore(str(tmp_path / "body.sqlite3"), role=Role.PRIMARY)
    store.migrate_v17_coordinates()
    return store, StructureWriter(store)


def test_permissions_rejects_hardlink_and_directory_target(tmp_path):
    from lingshu.core.structure_permissions import check_protected_directory

    storage = tmp_path / "storage"
    storage.mkdir(mode=0o700)
    if os.name == "posix":
        os.chmod(storage, 0o700)
    secret = storage / "secret.pem"
    secret.write_text("keydata", encoding="utf-8")

    # Subdirectory masquerading as protected file must be rejected
    sub = storage / "subdir"
    sub.mkdir()
    with pytest.raises(PermissionError, match="regular file"):
        check_protected_directory(storage, (sub,))

    # Hardlink alias outside storage must be rejected on both POSIX and Windows
    outside = tmp_path / "outside.pem"
    os.link(secret, outside)
    try:
        with pytest.raises(PermissionError, match="hardlink"):
            check_protected_directory(storage, (secret,))
    finally:
        outside.unlink(missing_ok=True)

    # Without hardlink, regular file inside protected directory passes (on POSIX with 0o644 too)
    if os.name == "posix":
        os.chmod(secret, 0o644)
        assert check_protected_directory(storage, (secret,)) == storage.resolve()
        # World-writable file must be rejected
        os.chmod(secret, 0o646)
        with pytest.raises(PermissionError, match="world-writable"):
            check_protected_directory(storage, (secret,))
        os.chmod(secret, 0o600)


def test_permissions_posix_checks_file_owner_and_mode_via_mock(tmp_path):
    from lingshu.core import structure_permissions as sp

    storage = tmp_path / "storage"
    storage.mkdir()
    target = storage / "service.json"
    target.write_text("{}", encoding="utf-8")

    root_st = SimpleNamespace(st_uid=1000, st_mode=stat_mod.S_IFDIR | 0o700, st_nlink=2)
    foreign_file = SimpleNamespace(st_uid=1001, st_mode=stat_mod.S_IFREG | 0o600, st_nlink=1)
    world_w_file = SimpleNamespace(st_uid=1000, st_mode=stat_mod.S_IFREG | 0o602, st_nlink=1)
    good_file = SimpleNamespace(st_uid=1000, st_mode=stat_mod.S_IFREG | 0o644, st_nlink=1)

    orig_stat = Path.stat
    root_res = storage.resolve()
    target_res = target.resolve()

    def run_with(file_st):
        def fake_stat(self, *args, **kwargs):
            if self == root_res:
                return root_st
            if self == target_res:
                return file_st
            return orig_stat(self, *args, **kwargs)

        host_path_cls = type(storage)
        with mock.patch.object(sp, "Path", host_path_cls), \
             mock.patch.object(sp.os, "name", "posix"), \
             mock.patch.object(sp.os, "geteuid", lambda: 1000, create=True), \
             mock.patch.object(host_path_cls, "stat", fake_stat):
            return sp.check_protected_directory(storage, (target,))

    assert run_with(good_file) == root_res
    with pytest.raises(PermissionError, match="own protected files"):
        run_with(foreign_file)
    with pytest.raises(PermissionError, match="world-writable"):
        run_with(world_w_file)


def test_writer_rejects_non_string_occurred_at_cleanly(tmp_path):
    from lingshu.core.structure_writer import ActorContext, EventInput, WriteError

    store, writer = _mk_writer(tmp_path)
    try:
        actor = ActorContext("s1", "ns1", "src1", frozenset({"events.write"}), frozenset())
        for bad in ({"bad": 1}, ["2026-01-01"], 12345, True):
            with pytest.raises(WriteError, match="occurred_at"):
                writer.record_event(actor, EventInput(f"ev_{type(bad).__name__}", "k", {}, occurred_at=bad))
        # Malformed string timestamp still records with clarification warning
        res = writer.record_event(actor, EventInput("ev_str", "k", {}, occurred_at="yesterday"))
        assert "occurred_at_needs_clarification" in res["warnings"]
        assert res["occurred_at"] is None
    finally:
        store.close()


def test_service_send_nan_and_server_and_load_validation(tmp_path):
    from lingshu.core import structure_service as ss
    from lingshu.core.structure_writer import WriteError

    class FakeConn:
        def sendall(self, _data):
            pass

    # _send must raise WriteError (not naked ValueError/TypeError) for NaN / non-JSON
    with pytest.raises(WriteError, match="strict JSON"):
        ss._send(FakeConn(), {"ok": True, "result": float("nan")})
    with pytest.raises(WriteError, match="strict JSON"):
        ss._send(FakeConn(), {"ok": True, "result": object()})

    # StructureServer validates max_connections
    tls = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    tls.verify_mode = ssl.CERT_REQUIRED
    for bad_max in (0, -1, True, False, 1.5, "4"):
        with pytest.raises(WriteError, match="max_connections"):
            ss.StructureServer(("127.0.0.1", 0), None, tls, {}, max_connections=bad_max, bind_and_activate=False)

    # load_service rejects malformed JSON, non-dict, empty/dir path, bad principals, bad listen
    storage = tmp_path / "svc"
    storage.mkdir(mode=0o700)
    if os.name == "posix":
        os.chmod(storage, 0o700)
    cfg = storage / "service.json"

    for payload in (
        "{not json",
        "[]",
        json.dumps({"database": "", "certificate": "c", "key": "k", "ca": "a", "principals": []}),
        json.dumps({"database": ".", "certificate": "c", "key": "k", "ca": "a", "principals": []}),
        json.dumps({"database": "db.sqlite", "certificate": "c", "key": "k", "ca": "a", "principals": "notlist"}),
        json.dumps({"database": "db.sqlite", "certificate": "c", "key": "k", "ca": "a", "principals": [], "listen": ["127.0.0.1", True]}),
        json.dumps({"database": "db.sqlite", "certificate": "c", "key": "k", "ca": "a", "principals": [], "listen": ["127.0.0.1", 70000]}),
    ):
        cfg.write_text(payload, encoding="utf-8")
        with pytest.raises(WriteError):
            ss.load_service(cfg, bind_and_activate=False)


def test_client_call_rejects_malformed_envelope():
    from lingshu.core import structure_service as ss
    from lingshu.core.structure_writer import WriteError

    client = object.__new__(ss.StructureClient)
    client.address = ("127.0.0.1", 7443)
    client.server_name = "localhost"
    client.timeout = 1
    client.tls = mock.MagicMock()

    with mock.patch("socket.create_connection") as m_conn, \
         mock.patch.object(ss, "_send"), \
         mock.patch.object(ss, "_receive", return_value=["not-a-dict"]):
        m_conn.return_value.__enter__.return_value = mock.MagicMock()
        with pytest.raises(WriteError, match="invalid response envelope"):
            client.call("current", aggregate_key="k")

    with mock.patch("socket.create_connection") as m_conn, \
         mock.patch.object(ss, "_send"), \
         mock.patch.object(ss, "_receive", return_value={"ok": True}):
        m_conn.return_value.__enter__.return_value = mock.MagicMock()
        with pytest.raises(WriteError, match="missing result"):
            client.call("current", aggregate_key="k")


_MUTATION_ARMS = {
    "M1_hardlink_gate": (
        "lingshu/core/structure_permissions.py",
        'if fstat.st_nlink > 1:\n                raise PermissionError("protected file must not have hardlink aliases")',
        "if False:\n                raise PermissionError(\"protected file must not have hardlink aliases\")",
        "test_permissions_rejects_hardlink_and_directory_target",
    ),
    "M2_occurred_at_type_gate": (
        "lingshu/core/structure_writer.py",
        'if event.occurred_at is not None and not isinstance(event.occurred_at, str):\n            raise WriteError("occurred_at must be an ISO-8601 string or None")',
        "if False:\n            raise WriteError(\"occurred_at must be an ISO-8601 string or None\")",
        "test_writer_rejects_non_string_occurred_at_cleanly",
    ),
    "M3_send_strict_json_gate": (
        "lingshu/core/structure_service.py",
        'except (ValueError, TypeError) as exc:\n        raise WriteError("response cannot be encoded as strict JSON") from exc',
        "except ZeroDivisionError as exc:\n        raise WriteError(\"response cannot be encoded as strict JSON\") from exc",
        "test_service_send_nan_and_server_and_load_validation",
    ),
    "M4_load_service_dict_gate": (
        "lingshu/core/structure_service.py",
        'if not isinstance(config, dict):\n        raise WriteError("service configuration must be a JSON object")',
        "if False:\n        raise WriteError(\"service configuration must be a JSON object\")",
        "test_service_send_nan_and_server_and_load_validation",
    ),
}


def _run_mutate(want: str) -> int:
    arms = list(_MUTATION_ARMS) if want == "ALL" else [want]
    failed = 0
    for arm in arms:
        if arm not in _MUTATION_ARMS:
            print(f"[ANCHOR-MISS] unknown arm {arm}")
            return 2
        rel, old, new, test_name = _MUTATION_ARMS[arm]
        src_path = _REPO / rel
        orig_bytes = src_path.read_bytes()
        tmp = Path(tempfile.mkdtemp(prefix=f"mut_l1_{arm}_"))
        try:
            shutil.copytree(_REPO / "lingshu", tmp / "lingshu",
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            (tmp / "tests").mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(__file__), tmp / "tests" / Path(__file__).name)
            target = tmp / rel
            text = target.read_bytes().decode("utf-8").replace("\r\n", "\n")
            if text.count(old) < 1:
                print(f"[ANCHOR-MISS] {arm}: anchor not found in {rel}")
                return 2
            target.write_bytes(text.replace(old, new).encode("utf-8"))
            env = os.environ.copy()
            env["PYTHONPATH"] = str(tmp)
            r = subprocess.run(
                [sys.executable, "-X", "utf8", "-m", "pytest", "-q",
                 str(tmp / "tests" / Path(__file__).name), "-k", test_name],
                cwd=str(tmp), env=env, capture_output=True, text=True,
                encoding="utf-8", errors="replace",
            )
            untouched = src_path.read_bytes() == orig_bytes
            if r.returncode != 0 and untouched:
                print(f"[MUTATE-OK] {arm} killed (rc={r.returncode}, repo_files_untouched={untouched})")
            else:
                print(f"[MUTATE-FAIL] {arm} survived (rc={r.returncode}, repo_files_untouched={untouched})")
                failed += 1
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    want = None
    for a in sys.argv[1:]:
        if a == "--mutate":
            want = "ALL"
        elif a.startswith("--mutate="):
            want = a.split("=", 1)[1]
    if want is not None:
        raise SystemExit(_run_mutate(want))
    raise SystemExit(pytest.main(["-q", __file__]))
