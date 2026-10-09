"""Persistent structural history, immutable proposals and approved current heads.

Standard library only. Use through structure_service for untrusted workers;
this in-process API is reserved for the trusted host.
"""
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import sqlite3
import uuid

from .core import Role


class WriteError(Exception):
    """A request the caller must fix or resubmit explicitly."""


class PermissionDenied(WriteError):
    pass


class Conflict(WriteError):
    pass


@dataclass(frozen=True)
class ActorContext:
    """Internal connection context; constructing one is not authentication."""
    subject: str
    namespace: str
    source: str
    capabilities: frozenset[str]
    instances: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class EventInput:
    source_event_id: str
    kind: str
    payload: dict
    occurred_at: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ProposalInput:
    request_id: str
    aggregate_key: str
    kind: str
    payload: dict
    expected_revision: int
    operation: str = "set"
    event_id: str | None = None
    prior_proposal_id: str | None = None
    valid_until: str | None = None


@dataclass(frozen=True)
class VersionResult:
    report_id: str
    run_id: str
    instance_id: str
    to_version: str
    outcome: str
    from_version: str | None = None
    occurred_at: str | None = None
    details: dict = field(default_factory=dict)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _id() -> str:
    return uuid.uuid4().hex


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WriteError(f"{name} must be nonempty text")
    return value


def _json(value: dict) -> str:
    if not isinstance(value, dict):
        raise WriteError("payload must be a JSON object")
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise WriteError("payload must be JSON serializable") from exc


def _time(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            raise ValueError("timezone required")
        return stamp.astimezone(timezone.utc).isoformat(timespec="microseconds")
    except (AttributeError, TypeError, ValueError) as exc:
        raise WriteError("time must identify a timezone, e.g. +08:00 or Z") from exc


_SCHEMA = """
CREATE TABLE IF NOT EXISTS structure_events (
    id TEXT PRIMARY KEY REFERENCES nodes(id), namespace TEXT NOT NULL, source TEXT NOT NULL,
    source_event_id TEXT NOT NULL, subject TEXT NOT NULL, kind TEXT NOT NULL,
    payload TEXT NOT NULL, metadata TEXT NOT NULL, origin TEXT NOT NULL, occurred_at TEXT,
    raw_occurred_at TEXT, recorded_at TEXT NOT NULL, warnings TEXT NOT NULL,
    UNIQUE(namespace, source, source_event_id)
);
CREATE TABLE IF NOT EXISTS structure_proposals (
    id TEXT PRIMARY KEY, namespace TEXT NOT NULL, proposer TEXT NOT NULL,
    request_id TEXT NOT NULL, aggregate_key TEXT NOT NULL, kind TEXT NOT NULL,
    operation TEXT NOT NULL CHECK(operation IN ('set','revoke')),
    payload TEXT NOT NULL, expected_revision INTEGER NOT NULL,
    event_id TEXT REFERENCES structure_events(id),
    prior_proposal_id TEXT REFERENCES structure_proposals(id),
    valid_until TEXT, created_at TEXT NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('pending','reviewed','rejected','committed','superseded')),
    UNIQUE(namespace, proposer, request_id)
);
CREATE TABLE IF NOT EXISTS structure_reviews (
    id TEXT PRIMARY KEY, proposal_id TEXT NOT NULL REFERENCES structure_proposals(id),
    reviewer TEXT NOT NULL, decision TEXT NOT NULL, note TEXT NOT NULL,
    recorded_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS structure_reviews_proposal ON structure_reviews(proposal_id);
CREATE TABLE IF NOT EXISTS structure_records (
    id TEXT PRIMARY KEY REFERENCES nodes(id), namespace TEXT NOT NULL, aggregate_key TEXT NOT NULL,
    revision INTEGER NOT NULL, proposal_id TEXT NOT NULL UNIQUE REFERENCES structure_proposals(id),
    approver TEXT NOT NULL, committed_at TEXT NOT NULL,
    UNIQUE(namespace, aggregate_key, revision)
);
CREATE TABLE IF NOT EXISTS structure_heads (
    namespace TEXT NOT NULL, aggregate_key TEXT NOT NULL, kind TEXT NOT NULL,
    revision INTEGER NOT NULL, record_id TEXT NOT NULL REFERENCES structure_records(id),
    PRIMARY KEY(namespace, aggregate_key)
);
"""


class StructureWriter:
    """Governance inside the protected host, sharing its existing LayeredStore.

    ActorContext comes from the authenticated service, never from wire data.
    Event/approval projections and decisions commit in the same database.
    """

    def __init__(self, host):
        self.store = getattr(host, "store", host)
        if self.store.role != Role.PRIMARY:
            raise PermissionDenied("governance requires the protected PRIMARY host")
        self._lock = self.store._lock
        if self.store.conn.in_transaction:
            raise WriteError("finish the caller transaction before initializing governance")
        self.store._structure_writer_active = True
        self.store.conn.execute("PRAGMA foreign_keys=ON")
        version = self.store.get_meta("structure_writer_schema").get("structure_writer_schema")
        if version is not None and version != "1":
            raise WriteError("unsupported structure writer schema")
        if version is not None:
            tables = {row[0] for row in self._db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            required = {"structure_events", "structure_proposals", "structure_reviews",
                        "structure_records", "structure_heads"}
            if not required <= tables:
                raise WriteError("incomplete governance database; restore a full SQLite backup")
        if version is None:
            with self._transaction():
                # execute individually: executescript would commit a caller transaction.
                for statement in _SCHEMA.split(";"):
                    if statement.strip():
                        self._db.execute(statement)
                self._db.execute(
                    "INSERT INTO engine_meta(key,value) VALUES ('structure_writer_schema','1')")

    @property
    def _db(self):
        return self.store.conn

    @contextmanager
    def _transaction(self):
        with self.store._structure_event_transaction():
            yield

    def backup(self, path):
        """Trusted administrative API; deliberately not an RPC command.

        SQLite backup includes governance and legacy tables; M13 node JSON
        alone is not a full governance backup. Call outside a transaction.
        """
        from pathlib import Path
        if self.store.db_path != ":memory:" and Path(path).resolve() == Path(self.store.db_path).resolve():
            raise WriteError("backup must not overwrite the live database")
        with self._lock:
            if self._db.in_transaction:
                raise WriteError("finish the caller transaction before backup")
            with closing(sqlite3.connect(path)) as destination:
                self._db.backup(destination)

    @staticmethod
    def _permit(actor: ActorContext, capability: str):
        if capability not in actor.capabilities:
            raise PermissionDenied(capability)

    def _audit(self, actor: ActorContext, action: str, object_id: str, note=""):
        import time
        self._db.execute(
            "INSERT INTO action_logs(ts,action_type,summary,node_ids,outcome,context) "
            "VALUES (?,?,?,?,?,?)",
            (time.time(), action, note, json.dumps([object_id]), "{}",
             json.dumps({"namespace": actor.namespace, "subject": actor.subject})))

    def _proposal(self, actor: ActorContext, proposal_id: str):
        row = self._db.execute("SELECT * FROM structure_proposals WHERE id=? AND namespace=?",
                               (proposal_id, actor.namespace)).fetchone()
        if row is None:
            raise WriteError("proposal not found in this namespace")
        return row

    @staticmethod
    def _event_view(row, replayed=False):
        item = dict(row)
        item["payload"] = json.loads(item["payload"])
        item["metadata"] = json.loads(item["metadata"])
        item["warnings"] = json.loads(item["warnings"])
        item["replayed"] = replayed
        return item

    def record_event(self, actor: ActorContext, event: EventInput) -> dict:
        self._permit(actor, "events.write")
        return self._record_event(actor, event, "reported")

    def _record_event(self, actor: ActorContext, event: EventInput, origin: str) -> dict:
        _text(event.source_event_id, "source_event_id")
        _text(event.kind, "kind")
        payload = _json(event.payload)
        metadata = _json(event.metadata)
        warnings = []
        try:
            occurred_at = _time(event.occurred_at)
        except WriteError:
            occurred_at = None
            warnings.append("occurred_at_needs_clarification")
        with self._transaction():
            old = self._db.execute(
                "SELECT * FROM structure_events WHERE namespace=? AND source=? AND source_event_id=?",
                (actor.namespace, actor.source, event.source_event_id)).fetchone()
            if old is not None:
                if (old["kind"] != event.kind or old["origin"] != origin
                        or json.loads(old["payload"]) != json.loads(payload)):
                    raise Conflict("event ID holds different content; append a correction")
                return self._event_view(old, replayed=True)
            node = self.store._record_structure_event(
                event.kind, json.dumps(event.payload, ensure_ascii=False),
                source=actor.source, confidence=0.8,
                metadata={"namespace": actor.namespace, "subject": actor.subject,
                          "source_event_id": event.source_event_id, "origin": origin,
                          "payload": event.payload, "occurred_at": occurred_at,
                          "raw_occurred_at": event.occurred_at, "warnings": warnings,
                          "details": event.metadata}, _cursor=self._db.cursor())
            event_id = node.id
            self._db.execute(
                "INSERT INTO structure_events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (event_id, actor.namespace, actor.source, event.source_event_id,
                 actor.subject, event.kind, payload, metadata, origin, occurred_at,
                 event.occurred_at, _now(), json.dumps(warnings)))
            self._audit(actor, "event.recorded", event_id)
            row = self._db.execute("SELECT * FROM structure_events WHERE id=?", (event_id,)).fetchone()
            return self._event_view(row)

    def record_version_result(self, actor: ActorContext, result: VersionResult) -> dict:
        """Record an authenticated executor's actual result; no code is executed."""
        self._permit(actor, "versions.report")
        if result.instance_id not in actor.instances:
            raise PermissionDenied("executor is not assigned this instance")
        if result.outcome not in {"applied", "failed", "rolled_back", "uncertain"}:
            raise WriteError("unknown execution outcome")
        _text(result.run_id, "run_id")
        _text(result.to_version, "to_version")
        payload = {
            "run_id": result.run_id, "instance_id": result.instance_id,
            "from_version": result.from_version, "to_version": result.to_version,
            "outcome": result.outcome}
        return self._record_event(
            actor, EventInput(result.report_id, "version." + result.outcome,
                              payload, result.occurred_at, {"details": result.details}), "executor_report")

    def propose(self, actor: ActorContext, request: ProposalInput) -> dict:
        self._permit(actor, "structure.propose")
        _text(request.request_id, "request_id")
        _text(request.aggregate_key, "aggregate_key")
        if request.kind not in {"anchor", "rule"} or request.operation not in {"set", "revoke"}:
            raise WriteError("use an explicit anchor/rule and set/revoke operation")
        if type(request.expected_revision) is not int or request.expected_revision < 0:
            raise WriteError("expected_revision must be a nonnegative integer")
        payload = _json(request.payload)
        end = _time(request.valid_until)
        snapshot = (
            request.aggregate_key, request.kind, request.operation, json.loads(payload),
            request.expected_revision, request.event_id, request.prior_proposal_id, end)
        with self._transaction():
            old = self._db.execute(
                "SELECT * FROM structure_proposals WHERE namespace=? AND proposer=? AND request_id=?",
                (actor.namespace, actor.subject, request.request_id)).fetchone()
            if old is not None:
                previous = (
                    old["aggregate_key"], old["kind"], old["operation"],
                    json.loads(old["payload"]), old["expected_revision"],
                    old["event_id"], old["prior_proposal_id"], old["valid_until"])
                if previous != snapshot:
                    raise Conflict("request ID already holds another proposal")
                return {"proposal_id": old["id"], "state": old["state"], "replayed": True}
            if request.event_id:
                event = self._db.execute(
                    "SELECT 1 FROM structure_events WHERE id=? AND namespace=?",
                    (request.event_id, actor.namespace)).fetchone()
                if event is None:
                    raise WriteError("source event not found in this namespace")
            if request.prior_proposal_id:
                prior = self._proposal(actor, request.prior_proposal_id)
                if (prior["aggregate_key"] != request.aggregate_key
                        or prior["event_id"] != request.event_id
                        or prior["kind"] != request.kind):
                    raise WriteError("prior proposal must belong to this event/target/kind")
                if prior["state"] in {"pending", "reviewed"}:
                    if prior["proposer"] != actor.subject:
                        raise PermissionDenied("only the proposer can replace an open proposal")
                    self._db.execute("UPDATE structure_proposals SET state='superseded' WHERE id=?",
                                     (prior["id"],))
                    self._audit(actor, "proposal.superseded", prior["id"], request.request_id)
            proposal_id = _id()
            self._db.execute(
                "INSERT INTO structure_proposals VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (proposal_id, actor.namespace, actor.subject, request.request_id,
                 request.aggregate_key, request.kind, request.operation, payload,
                 request.expected_revision, request.event_id,
                 request.prior_proposal_id, end, _now(), "pending"))
            self._audit(actor, "proposal.created", proposal_id)
            return {"proposal_id": proposal_id, "state": "pending", "replayed": False}

    def get_proposal(self, actor: ActorContext, proposal_id: str) -> dict:
        """Return the saved approval object; changing this copy cannot change it."""
        self._permit(actor, "structure.read")
        with self._lock:
            item = dict(self._proposal(actor, proposal_id))
            item["payload"] = json.loads(item["payload"])
            return item

    def review(self, actor: ActorContext, proposal_id: str, decision: str, note="") -> dict:
        self._permit(actor, "structure.review")
        if decision not in {"approve", "needs_info", "reject"}:
            raise WriteError("unknown review decision")
        with self._transaction():
            proposal = self._proposal(actor, proposal_id)
            if proposal["state"] in {"rejected", "committed", "superseded"}:
                raise Conflict("proposal is closed; create a new attempt")
            if actor.subject == proposal["proposer"]:
                raise PermissionDenied("proposer cannot review this proposal")
            state = {"approve": "reviewed", "needs_info": "pending", "reject": "rejected"}[decision]
            self._db.execute(
                "INSERT INTO structure_reviews VALUES (?,?,?,?,?,?)",
                (_id(), proposal_id, actor.subject, decision, note, _now()))
            self._db.execute("UPDATE structure_proposals SET state=? WHERE id=?", (state, proposal_id))
            self._audit(actor, "proposal.reviewed", proposal_id, decision)
            return {"proposal_id": proposal_id, "state": state}

    def approve(self, actor: ActorContext, proposal_id: str) -> dict:
        self._permit(actor, "structure.approve")
        with self._transaction():
            proposal = self._proposal(actor, proposal_id)
            if proposal["state"] == "committed":
                return dict(self._db.execute(
                    "SELECT * FROM structure_records WHERE proposal_id=?", (proposal_id,)).fetchone())
            if proposal["state"] != "reviewed":
                raise Conflict("proposal needs an approving review")
            review = self._db.execute(
                "SELECT * FROM structure_reviews WHERE proposal_id=? ORDER BY rowid DESC LIMIT 1",
                (proposal_id,)).fetchone()
            if actor.subject in {proposal["proposer"], review["reviewer"]}:
                raise PermissionDenied("proposer, reviewer and approver must be distinct")
            head = self._db.execute(
                "SELECT * FROM structure_heads WHERE namespace=? AND aggregate_key=?",
                (actor.namespace, proposal["aggregate_key"])).fetchone()
            revision = head["revision"] if head else 0
            if revision != proposal["expected_revision"]:
                raise Conflict(
                    f"target changed: expected {proposal['expected_revision']}, current {revision}")
            if head and head["kind"] != proposal["kind"]:
                raise Conflict("target kind changed; use a distinct aggregate key")
            if not head and proposal["operation"] == "revoke":
                raise Conflict("target does not exist")
            if proposal["valid_until"] and _now() >= proposal["valid_until"]:
                raise Conflict("proposal has expired; create a new proposal")
            node = self.store._record_structure_event(
                proposal["kind"], proposal["payload"], source=actor.source,
                confidence=1.0, record_class="normative",
                metadata={"namespace": actor.namespace,
                          "aggregate_key": proposal["aggregate_key"],
                          "revision": revision + 1, "proposal_id": proposal_id,
                          "operation": proposal["operation"],
                          "valid_until": proposal["valid_until"]},
                _cursor=self._db.cursor())
            record_id, committed_at = node.id, _now()
            self._db.execute(
                "INSERT INTO structure_records VALUES (?,?,?,?,?,?,?)",
                (record_id, actor.namespace, proposal["aggregate_key"], revision + 1,
                 proposal_id, actor.subject, committed_at))
            if head:
                self._db.execute(
                    "UPDATE structure_heads SET revision=?,record_id=? "
                    "WHERE namespace=? AND aggregate_key=?",
                    (revision + 1, record_id, actor.namespace, proposal["aggregate_key"]))
            else:
                self._db.execute(
                    "INSERT INTO structure_heads VALUES (?,?,?,?,?)",
                    (actor.namespace, proposal["aggregate_key"], proposal["kind"], 1, record_id))
            self._db.execute("UPDATE structure_proposals SET state='committed' WHERE id=?", (proposal_id,))
            self._audit(actor, "structure.committed", record_id)
            return {
                "id": record_id, "namespace": actor.namespace,
                "aggregate_key": proposal["aggregate_key"], "revision": revision + 1,
                "proposal_id": proposal_id, "approver": actor.subject,
                "committed_at": committed_at}

    def target_status(self, actor: ActorContext, aggregate_key: str) -> dict:
        """Keep the target revision visible even after revocation or expiry."""
        self._permit(actor, "structure.read")
        with self._lock:
            row = self._db.execute(
                "SELECT r.*,p.kind,p.operation,p.payload,p.valid_until "
                "FROM structure_heads h JOIN structure_records r ON r.id=h.record_id "
                "JOIN structure_proposals p ON p.id=r.proposal_id "
                "WHERE h.namespace=? AND h.aggregate_key=?",
                (actor.namespace, aggregate_key)).fetchone()
            if row is None:
                return {"aggregate_key": aggregate_key, "revision": 0,
                        "status": "missing", "record": None}
            status = "active"
            if row["operation"] == "revoke":
                status = "revoked"
            elif row["valid_until"] and _now() >= row["valid_until"]:
                status = "expired"
            item = dict(row)
            item["payload"] = json.loads(item["payload"])
            return {"aggregate_key": aggregate_key, "revision": row["revision"],
                    "status": status, "record": item}

    def current(self, actor: ActorContext, aggregate_key: str) -> dict | None:
        """Latest active approval; callers must separately evaluate runtime scope."""
        target = self.target_status(actor, aggregate_key)
        return target["record"] if target["status"] == "active" else None

    def event_history(self, actor: ActorContext, *, limit=100, before_rowid=None) -> list[dict]:
        self._permit(actor, "structure.read")
        limit = max(1, min(limit, 200))
        with self._lock:
            rows = self._db.execute(
                "SELECT rowid AS cursor,* FROM structure_events WHERE namespace=? "
                "AND (? IS NULL OR rowid<?) ORDER BY rowid DESC LIMIT ?",
                (actor.namespace, before_rowid, before_rowid, limit)).fetchall()
            return [self._event_view(row) for row in rows]

    def structure_history(self, actor: ActorContext, aggregate_key: str, *,
                          limit=100, before_revision=None) -> list[dict]:
        self._permit(actor, "structure.read")
        limit = max(1, min(limit, 200))
        with self._lock:
            rows = self._db.execute(
                "SELECT r.*,p.kind,p.operation,p.payload FROM structure_records r "
                "JOIN structure_proposals p ON p.id=r.proposal_id "
                "WHERE r.namespace=? AND r.aggregate_key=? "
                "AND (? IS NULL OR r.revision<?) ORDER BY r.revision DESC LIMIT ?",
                (actor.namespace, aggregate_key, before_revision, before_revision, limit)).fetchall()
            return [dict(row, payload=json.loads(row["payload"])) for row in rows]

