"""Protected structural writer process and mTLS JSON client (standard library).

Run `python -m lingshu.core.structure_service service.json` under the dedicated
service account. Workers receive only their own client certificates, never the
server key, designer key, configuration or database.
"""
import argparse
import json
import logging
from pathlib import Path
import socket
import socketserver
import sqlite3
import ssl
import struct
import threading

from .core import LayeredStore, Role
from .structure_permissions import check_protected_directory
from .structure_writer import (ActorContext, Conflict, EventInput, PermissionDenied,
                               ProposalInput, StructureWriter, VersionResult, WriteError)

MAX_MESSAGE = 1024 * 1024


def _read_exact(connection, size):
    parts = []
    while size:
        part = connection.recv(size)
        if not part:
            raise EOFError("connection ended before the response")
        parts.append(part)
        size -= len(part)
    return b"".join(parts)


def _receive(connection):
    size = struct.unpack("!I", _read_exact(connection, 4))[0]
    if size > MAX_MESSAGE:
        raise WriteError("message exceeds the service limit; use history pagination")
    return json.loads(_read_exact(connection, size).decode("utf-8"))


def _send(connection, value):
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(payload) > MAX_MESSAGE:
        raise WriteError("response exceeds the service limit; request a smaller page")
    connection.sendall(struct.pack("!I", len(payload)) + payload)


def dispatch(writer, actor, request):
    """Explicit business commands; no SQL, paths, imports or caller-supplied roles."""
    if not isinstance(request, dict) or set(request) != {"command", "arguments"}:
        raise WriteError("send command and arguments")
    commands = {
        "record_event": (writer.record_event, EventInput),
        "record_version_result": (writer.record_version_result, VersionResult),
        "propose": (writer.propose, ProposalInput),
        "get_proposal": (writer.get_proposal, None),
        "review": (writer.review, None), "approve": (writer.approve, None),
        "current": (writer.current, None), "target_status": (writer.target_status, None),
        "event_history": (writer.event_history, None),
        "structure_history": (writer.structure_history, None),
    }
    args = request["arguments"]
    if not isinstance(request["command"], str):
        raise WriteError("command must be text")
    entry = commands.get(request["command"])
    if entry is None or not isinstance(args, dict):
        raise WriteError("unknown command or invalid arguments")
    method, value_type = entry
    try:
        result = method(actor, value_type(**args)) if value_type else method(actor, **args)
        if request["command"] in {"record_event", "record_version_result"}:
            # A durable write returns a small receipt; large payloads are read
            # separately, so response size cannot make a successful write look lost.
            return {key: result[key] for key in ("id", "origin", "recorded_at", "replayed", "warnings")}
        return result
    except TypeError as exc:
        raise WriteError("arguments do not match this business command") from exc


class _Handler(socketserver.BaseRequestHandler):
    def handle(self):
        self.request.settimeout(5)
        try:
            with self.server.tls.wrap_socket(self.request, server_side=True) as connection:
                names = [value for kind, value in connection.getpeercert().get("subjectAltName", ())
                         if kind == "URI" and value.startswith("urn:lingshu:")]
                actor = self.server.principals.get(names[0]) if len(names) == 1 else None
                if actor is None:
                    raise PermissionDenied("client certificate has no configured identity")
                try:
                    result = dispatch(self.server.writer, actor, _receive(connection))
                    response = {"ok": True, "result": result}
                except (WriteError, ValueError, UnicodeError) as exc:
                    response = {"ok": False, "error": type(exc).__name__, "message": str(exc)}
                except sqlite3.Error:
                    logging.exception("structural storage request failed")
                    response = {"ok": False, "error": "StorageError",
                                "message": "storage failed; retry the same request ID"}
                try:
                    _send(connection, response)
                except WriteError as exc:
                    _send(connection, {"ok": False, "error": "WriteError", "message": str(exc)})
        except (OSError, EOFError, PermissionDenied):
            # Authentication/transport failures have no business receipt.
            logging.info("structural connection rejected or interrupted")
        finally:
            self.server.writer.store.close_current_thread()


class StructureServer(socketserver.ThreadingTCPServer):
    """Bounded connections; one authenticated command per connection."""
    allow_reuse_address = True
    daemon_threads = False
    block_on_close = True

    def __init__(self, address, writer, tls, principals, *, max_connections=16, bind_and_activate=True):
        if tls.verify_mode != ssl.CERT_REQUIRED:
            raise PermissionError("mutual TLS client authentication is required")
        self.writer, self.tls, self.principals = writer, tls, dict(principals)
        self._capacity = threading.BoundedSemaphore(max_connections)
        super().__init__(address, _Handler, bind_and_activate=bind_and_activate)

    def process_request(self, request, client_address):
        if not self._capacity.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._capacity.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._capacity.release()


class StructureClient:
    def __init__(self, host, port, *, ca, certificate, key, server_name="localhost", timeout=5):
        self.address, self.server_name, self.timeout = (host, port), server_name, timeout
        self.tls = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=ca)
        self.tls.keylog_filename = None
        self.tls.minimum_version = ssl.TLSVersion.TLSv1_2
        self.tls.load_cert_chain(certificate, key)

    def call(self, command, **arguments):
        with socket.create_connection(self.address, timeout=self.timeout) as raw:
            with self.tls.wrap_socket(raw, server_hostname=self.server_name) as connection:
                _send(connection, {"command": command, "arguments": arguments})
                response = _receive(connection)
        if not response["ok"]:
            errors = {"PermissionDenied": PermissionDenied, "Conflict": Conflict}
            raise errors.get(response["error"], WriteError)(response["message"])
        return response["result"]


def load_service(config_file, *, bind_and_activate=True):
    """Trusted startup: check permissions once and pin certificate grants."""
    config_file = Path(config_file).resolve(strict=True)
    root = check_protected_directory(config_file.parent, (config_file,))
    config = json.loads(config_file.read_text(encoding="utf-8"))
    files = {name: root / config[name] for name in ("database", "certificate", "key", "ca")}
    extra = [Path(str(files["database"]) + suffix) for suffix in ("-wal", "-shm", "-journal")]
    check_protected_directory(root, [*files.values(), *extra])
    principals = {}
    for grant in config["principals"]:
        identity = grant["identity"]
        if not identity.startswith("urn:lingshu:") or identity in principals:
            raise WriteError("configure a unique certificate identity")
        if not all(isinstance(grant.get(name), str) and grant[name].strip()
                   for name in ("subject", "namespace", "source")):
            raise WriteError("grant subject, namespace and source must be configured")
        principals[identity] = ActorContext(
            grant["subject"], grant["namespace"], grant["source"],
            frozenset(grant["capabilities"]), frozenset(grant.get("instances", ())))
    tls = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH, cafile=str(files["ca"]))
    tls.keylog_filename = None
    tls.minimum_version = ssl.TLSVersion.TLSv1_2
    tls.verify_mode = ssl.CERT_REQUIRED
    tls.load_cert_chain(str(files["certificate"]), str(files["key"]))
    # Do not assemble optional plugins in the privileged process.
    store = LayeredStore(str(files["database"]), role=Role.PRIMARY)
    try:
        store.migrate_v17_coordinates()
        writer = StructureWriter(store)
        server = StructureServer(tuple(config.get("listen", ["127.0.0.1", 7443])),
                                 writer, tls, principals, bind_and_activate=bind_and_activate)
    except BaseException:
        store.close()
        raise
    return server


def main():
    parser = argparse.ArgumentParser(description="Protected structural writer service")
    parser.add_argument("config", help="service-owned configuration JSON")
    parser.add_argument("--backup", help="one-shot full SQLite backup inside service storage")
    args = parser.parse_args()
    server = load_service(args.config, bind_and_activate=not args.backup)
    try:
        with server:
            if args.backup:
                destination = Path(args.backup).resolve()
                check_protected_directory(Path(args.config).resolve().parent, (destination,))
                server.writer.backup(destination)
            else:
                logging.info("structural writer listening on %s", server.server_address)
                print(json.dumps({"listening": list(server.server_address)}), flush=True)
                server.serve_forever()
    finally:
        server.writer.store.close()


if __name__ == "__main__":
    main()
