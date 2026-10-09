"""Business governance and authenticated subprocess integration; synthetic data only."""
import json
import os
from pathlib import Path
import queue
import shutil
import socket
import subprocess
import sys
import threading

import pytest

from lingshu.core.core import LayeredStore, MemoryLayer, Role, SpacetimeMemoryEngine
from lingshu.core.structure_permissions import check_protected_directory
from lingshu.core.structure_service import StructureClient
from lingshu.core.structure_writer import (ActorContext, Conflict, EventInput, PermissionDenied,
                                         ProposalInput, StructureWriter, VersionResult, WriteError)

ROOT = Path(__file__).resolve().parents[1]


def host(path):
    engine = SpacetimeMemoryEngine.__new__(SpacetimeMemoryEngine)
    engine.role = Role.PRIMARY
    engine.store = LayeredStore(str(path), role=Role.PRIMARY)
    return engine


def actor(subject, *capabilities):
    return ActorContext(subject, "lab", subject, frozenset(capabilities))


def test_approved_snapshot_revisions_recovery_and_atomic_failure(tmp_path):
    engine = host(tmp_path / "body.sqlite3")
    writer = StructureWriter(engine)
    proposer = actor("proposer", "events.write", "structure.propose", "structure.read")
    reviewer = actor("reviewer", "structure.review", "structure.read")
    approver = actor("approver", "structure.approve", "structure.read")
    try:
        event = writer.record_event(proposer, EventInput("event-1", "calibration", {"text": "reference"}))
        assert not engine.get_current_structure_nodes()  # history does not authorize rules
        repeated = writer.record_event(proposer, EventInput("event-1", "calibration", {"text": "reference"}, metadata={"note": "extra"}))
        assert repeated["replayed"] and repeated["id"] == event["id"]
        proposal = writer.propose(proposer, ProposalInput("p1", "reference", "anchor", {"weight": 0.8}, 0, event_id=event["id"]))
        saved = writer.get_proposal(reviewer, proposal["proposal_id"])
        saved["payload"]["weight"] = 99  # reviewer sees a detached saved snapshot
        assert writer.get_proposal(reviewer, proposal["proposal_id"])["payload"] == {"weight": 0.8}
        with pytest.raises(PermissionDenied):
            writer.review(actor("proposer", "structure.review"), proposal["proposal_id"], "approve")
        writer.review(reviewer, proposal["proposal_id"], "approve")
        with pytest.raises(PermissionDenied):
            writer.approve(actor("reviewer", "structure.approve"), proposal["proposal_id"])
        engine.store.conn.execute("CREATE TRIGGER reject_commit BEFORE INSERT ON action_logs WHEN NEW.action_type='structure.committed' BEGIN SELECT RAISE(ABORT,'synthetic failure'); END")
        engine.store.conn.commit()
        import sqlite3
        with pytest.raises(sqlite3.IntegrityError):
            writer.approve(approver, proposal["proposal_id"])
        assert writer.target_status(approver, "reference")["revision"] == 0
        assert writer.get_proposal(reviewer, proposal["proposal_id"])["state"] == "reviewed"
        assert engine.store.count_layer(MemoryLayer.STRUCTURE) == 1  # source event only
        engine.store.conn.execute("DROP TRIGGER reject_commit")
        engine.store.conn.commit()
        approved = writer.approve(approver, proposal["proposal_id"])
        assert writer.current(proposer, "reference")["payload"] == {"weight": 0.8}
        assert [n.id for n in engine.get_current_structure_nodes()] == [approved["id"]]
        assert writer.approve(approver, proposal["proposal_id"])["id"] == approved["id"]
        stale = writer.propose(proposer, ProposalInput("stale", "reference", "anchor", {"weight": 0.9}, 0))
        writer.review(reviewer, stale["proposal_id"], "approve")
        with pytest.raises(Conflict):
            writer.approve(approver, stale["proposal_id"])
        revoke = writer.propose(proposer, ProposalInput("revoke", "reference", "anchor", {}, 1, operation="revoke"))
        writer.review(reviewer, revoke["proposal_id"], "approve")
        writer.approve(approver, revoke["proposal_id"])
        assert writer.current(proposer, "reference") is None
        assert writer.target_status(proposer, "reference")["revision"] == 2
        assert engine.get_current_structure_nodes() == []
        backup = tmp_path / "backup.sqlite3"
        writer.backup(backup)
    finally:
        engine.store.close()
    restored = host(backup)
    try:
        reader = StructureWriter(restored)
        assert reader.target_status(proposer, "reference")["status"] == "revoked"
        assert len(reader.structure_history(proposer, "reference")) == 2
        assert reader.event_history(proposer)[0]["id"] == event["id"]
    finally:
        restored.store.close()


@pytest.fixture
def service(tmp_path):
    executable = shutil.which("openssl")
    assert executable, "OpenSSL CLI is required to generate ephemeral TLS test certificates"
    storage = tmp_path / "service"
    storage.mkdir(mode=0o700)
    # Only a temporary fixture directory is changed; no persistent user ACLs.
    if os.name == "nt":
        account = subprocess.check_output(["whoami", "/user", "/fo", "csv", "/nh"], text=True)
        import csv
        sid = next(csv.reader([account.strip()]))[1]
        subprocess.run(["icacls", str(storage), "/inheritance:r", "/grant:r", f"*{sid}:(OI)(CI)F"], check=True, capture_output=True)
    else:
        storage.chmod(0o700)

    def openssl(*args):
        subprocess.run([executable, *map(str, args)], check=True, capture_output=True, cwd=storage)

    openssl("req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1", "-subj", "/CN=Ephemeral test CA", "-keyout", "ca.key", "-out", "ca.pem", "-addext", "basicConstraints=critical,CA:TRUE", "-addext", "keyUsage=critical,keyCertSign,cRLSign")
    for name in ("server", "worker", "reviewer", "approver", "executor"):
        openssl("req", "-new", "-newkey", "rsa:2048", "-nodes", "-subj", f"/CN={name}", "-keyout", f"{name}.key", "-out", f"{name}.csr")
        extension = storage / f"{name}.ext"
        san = "DNS:localhost,IP:127.0.0.1" if name == "server" else f"URI:urn:lingshu:{name}"
        purpose = "serverAuth" if name == "server" else "clientAuth"
        extension.write_text(f"basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\nextendedKeyUsage={purpose}\nsubjectAltName={san}\nsubjectKeyIdentifier=hash\nauthorityKeyIdentifier=keyid,issuer\n", encoding="utf-8")
        openssl("x509", "-req", "-in", f"{name}.csr", "-CA", "ca.pem", "-CAkey", "ca.key", "-CAcreateserial", "-out", f"{name}.pem", "-days", "1", "-extfile", extension)
    grants = {
        "worker": ["events.write", "structure.propose", "structure.read"],
        "reviewer": ["structure.review", "structure.read"],
        "approver": ["structure.approve", "structure.read"],
        "executor": ["versions.report", "structure.read"],
    }
    config = {"database": "body.sqlite3", "certificate": "server.pem", "key": "server.key", "ca": "ca.pem", "listen": ["127.0.0.1", 0],
              "principals": [{"identity": f"urn:lingshu:{name}", "subject": name, "namespace": "lab", "source": name, "capabilities": caps, "instances": ["instance-1"] if name == "executor" else []} for name, caps in grants.items()]}
    config_file = storage / "service.json"
    config_file.write_text(json.dumps(config), encoding="utf-8")
    check_protected_directory(storage, (config_file,))
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUTF8="1")
    child = subprocess.Popen([sys.executable, "-X", "utf8", "-m", "lingshu.core.structure_service", str(config_file)], cwd=tmp_path, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
    ready = queue.Queue()
    threading.Thread(target=lambda: ready.put(child.stdout.readline()), daemon=True).start()
    try:
        announcement = ready.get(timeout=15)
        assert announcement, child.stderr.read()
        address = json.loads(announcement)["listening"]
        clients = {name: StructureClient(*address, ca=str(storage / "ca.pem"), certificate=str(storage / f"{name}.pem"), key=str(storage / f"{name}.key")) for name in grants}
        yield clients, storage
    finally:
        child.terminate()
        child.wait(timeout=10)
        child.stdout.close()
        child.stderr.close()


def test_authenticated_process_business_workflow(service):
    clients, storage = service
    worker, reviewer, approver, executor = [clients[name] for name in ("worker", "reviewer", "approver", "executor")]
    event = worker.call("record_event", source_event_id="source-1", kind="version.planned", payload={"version": "0.2"})
    assert event["origin"] == "reported"
    with pytest.raises(WriteError):
        worker.call("record_event", source_event_id="spoof", kind="calibration", payload={}, namespace="admin")
    with pytest.raises(PermissionDenied):
        worker.call("record_version_result", report_id="fake", run_id="run", instance_id="instance-1", to_version="0.2", outcome="applied")
    with pytest.raises(PermissionDenied):
        executor.call("record_version_result", report_id="wrong-instance", run_id="run", instance_id="other", to_version="0.2", outcome="applied")
    result = executor.call("record_version_result", report_id="result-1", run_id="run", instance_id="instance-1", to_version="0.2", outcome="applied")
    assert result["origin"] == "executor_report"
    proposal = worker.call("propose", request_id="proposal-1", aggregate_key="calibration", kind="rule", payload={"weight": 0.8}, expected_revision=0, event_id=event["id"])
    with pytest.raises(PermissionDenied):
        worker.call("approve", proposal_id=proposal["proposal_id"])
    assert reviewer.call("get_proposal", proposal_id=proposal["proposal_id"])["payload"] == {"weight": 0.8}
    reviewer.call("review", proposal_id=proposal["proposal_id"], decision="approve")
    approved = approver.call("approve", proposal_id=proposal["proposal_id"])
    assert approved["revision"] == 1
    assert worker.call("current", aggregate_key="calibration")["id"] == approved["id"]
    assert len(worker.call("event_history")) == 2
    # Administrative backup works while the other process owns the listening port.
    snapshot = storage / "snapshot.sqlite3"
    subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "lingshu.core.structure_service",
         str(storage / "service.json"), "--backup", str(snapshot)],
        cwd=storage.parent, env=dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUTF8="1"),
        check=True, capture_output=True, timeout=15)
    restored = host(snapshot)
    try:
        saved = StructureWriter(restored)
        read_actor = actor("worker", "structure.read")
        assert saved.current(read_actor, "calibration")["id"] == approved["id"]
        assert len(saved.event_history(read_actor)) == 2
    finally:
        restored.store.close()


def test_permission_boundary_rejects_public_storage(tmp_path):
    if os.name == "nt":
        # The default temporary directory inherits broader ACLs; test explicit grants.
        account = subprocess.check_output(["whoami", "/user", "/fo", "csv", "/nh"], text=True)
        import csv
        sid = next(csv.reader([account.strip()]))[1]
        subprocess.run(["icacls", str(tmp_path), "/inheritance:r", "/grant:r", f"*{sid}:(OI)(CI)F", "*S-1-1-0:(OI)(CI)R"], check=True, capture_output=True)
    else:
        tmp_path.chmod(0o755)
    with pytest.raises(PermissionError):
        check_protected_directory(tmp_path)
