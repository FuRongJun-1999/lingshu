"""帧读取的绝对预算与本地真实 mTLS 连接槽回收；仅使用合成数据。"""
import json
from pathlib import Path
import shutil
import socket
import ssl
import struct
import subprocess
import threading

import pytest

from lingshu.core import structure_service as service
from lingshu.core.core import LayeredStore, Role
from lingshu.core.structure_writer import ActorContext, StructureWriter, WriteError


class Clock:
    def __init__(self):
        self.value = 0.0

    def monotonic(self):
        return self.value


class Fragments:
    def __init__(self, clock, chunks, timeout=0.5):
        self.clock, self.chunks, self.timeout = clock, list(chunks), timeout

    def gettimeout(self):
        return self.timeout

    def settimeout(self, value):
        self.timeout = value

    def recv(self, size):
        if not self.chunks:
            return b""
        cost, part = self.chunks.pop(0)
        if self.timeout is not None and cost > self.timeout:
            self.clock.value += self.timeout
            raise TimeoutError("fragment exceeds remaining socket budget")
        self.clock.value += cost
        assert len(part) <= size
        return part


def _clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(service, "time", clock, raising=False)
    return clock


def test_fragmented_header_cannot_renew_the_deadline(monkeypatch):
    clock = _clock(monkeypatch)
    wire = Fragments(clock, [(0.2, bytes([b])) for b in struct.pack("!I", 2)] + [(0, b"{}")])
    with pytest.raises(TimeoutError):
        service._receive(wire)
    assert wire.gettimeout() == 0.5


def test_fragmented_body_uses_remaining_budget(monkeypatch):
    clock = _clock(monkeypatch)
    wire = Fragments(clock, [(0, struct.pack("!I", 2)), (0.3, b"{"), (0.3, b"}")])
    with pytest.raises(TimeoutError):
        service._receive(wire)
    assert wire.gettimeout() == 0.5


def test_header_and_body_share_one_budget(monkeypatch):
    clock = _clock(monkeypatch)
    wire = Fragments(clock, [(0.3, struct.pack("!I", 2)), (0.3, b"{}")])
    with pytest.raises(TimeoutError):
        service._receive(wire)


def test_original_timeout_is_restored_after_a_complete_frame(monkeypatch):
    clock = _clock(monkeypatch)
    wire = Fragments(clock, [(0.1, struct.pack("!I", 2)), (0.1, b"{}")])
    assert service._receive(wire) == {}
    assert wire.gettimeout() == 0.5, "framed reader changed the caller socket timeout"


@pytest.mark.parametrize("timeout", [None, 0.0], ids=["blocking", "nonblocking"])
def test_no_positive_timeout_retains_its_existing_semantics(monkeypatch, timeout):
    clock = _clock(monkeypatch)
    wire = Fragments(clock, [(0, struct.pack("!I", 2)), (0, b"{}")], timeout)
    assert service._receive(wire) == {} and wire.gettimeout() is timeout


def test_oversized_frame_is_rejected_before_body_read(monkeypatch):
    clock = _clock(monkeypatch)
    wire = Fragments(clock, [(0, struct.pack("!I", service.MAX_MESSAGE + 1))])
    with pytest.raises(WriteError, match="message exceeds"):
        service._receive(wire)
    assert wire.gettimeout() == 0.5


def test_incomplete_frame_retains_eof_and_restores_timeout(monkeypatch):
    clock = _clock(monkeypatch)
    wire = Fragments(clock, [(0.1, struct.pack("!I", 2)), (0.1, b"{")])
    with pytest.raises(EOFError):
        service._receive(wire)
    assert wire.gettimeout() == 0.5


def test_real_mtls_slow_peer_releases_the_only_connection_slot(tmp_path):
    executable = shutil.which("openssl")
    if executable is None:
        pytest.skip("真实 mTLS 守卫需要 OpenSSL CLI 生成临时证书")

    def openssl(*args):
        subprocess.run([executable, *map(str, args)], cwd=tmp_path, check=True, capture_output=True)

    openssl("req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1", "-subj", "/CN=Temporary deadline CA",
            "-keyout", "ca.key", "-out", "ca.pem", "-addext", "basicConstraints=critical,CA:TRUE",
            "-addext", "keyUsage=critical,keyCertSign,cRLSign")
    for name in ("server", "worker"):
        openssl("req", "-new", "-newkey", "rsa:2048", "-nodes", "-subj", "/CN=" + name,
                "-keyout", name + ".key", "-out", name + ".csr")
        san = "DNS:localhost,IP:127.0.0.1" if name == "server" else "URI:urn:lingshu:worker"
        purpose = "serverAuth" if name == "server" else "clientAuth"
        extension = tmp_path / (name + ".ext")
        extension.write_text("basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\n"
                             + "extendedKeyUsage=" + purpose + "\nsubjectAltName=" + san
                             + "\nsubjectKeyIdentifier=hash\nauthorityKeyIdentifier=keyid,issuer\n", encoding="utf-8")
        openssl("x509", "-req", "-in", name + ".csr", "-CA", "ca.pem", "-CAkey", "ca.key", "-CAcreateserial",
                "-out", name + ".pem", "-days", "1", "-extfile", extension)
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH, cafile=str(tmp_path / "ca.pem"))
    context.keylog_filename = None
    context.verify_mode = ssl.CERT_REQUIRED
    context.load_cert_chain(str(tmp_path / "server.pem"), str(tmp_path / "server.key"))
    established, finished, stop = threading.Event(), threading.Event(), threading.Event()

    class RealTLSWithShortReadBudget:
        verify_mode = ssl.CERT_REQUIRED

        def wrap_socket(self, *args, **kwargs):
            connection = context.wrap_socket(*args, **kwargs)
            # 只缩短量具的正读取预算；真实证书认证与原 Handler/dispatch 不替换。
            connection.settimeout(0.2)
            established.set()
            return connection

    class ObservedServer(service.StructureServer):
        def process_request_thread(self, request, address):
            try:
                super().process_request_thread(request, address)
            finally:
                finished.set()

    store = LayeredStore(str(tmp_path / "body.sqlite3"), role=Role.PRIMARY)
    writer = StructureWriter(store)
    actor = ActorContext("worker", "lab", "worker", frozenset({"structure.read"}))
    server = ObservedServer(("127.0.0.1", 0), writer, RealTLSWithShortReadBudget(),
                            {"urn:lingshu:worker": actor}, max_connections=1)
    serving = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.02))
    serving.start()
    raw = slow = sender = None
    sent, failures = [], []

    def drip():
        try:
            slow.sendall(struct.pack("!I", service.MAX_MESSAGE))
            # 最多几百字节，stop 后立即退出；不补齐正文，不进行任何数据库操作。
            for _ in range(100):
                if stop.is_set():
                    break
                slow.sendall(b" ")
                sent.append(1)
                stop.wait(0.04)
        except OSError:
            pass
        except BaseException as exc:
            failures.append(exc)

    try:
        client = service.StructureClient(*server.server_address, ca=str(tmp_path / "ca.pem"),
            certificate=str(tmp_path / "worker.pem"), key=str(tmp_path / "worker.key"), timeout=3)
        raw = socket.create_connection(server.server_address, timeout=3)
        slow = client.tls.wrap_socket(raw, server_hostname="localhost")
        sender = threading.Thread(target=drip)
        sender.start()
        assert established.wait(3)
        assert finished.wait(2), "slow TLS peer retained the service connection slot beyond its read budget"
        assert not failures and 0 < len(sent) < 100
        stop.set()
        sender.join(timeout=2)
        assert not sender.is_alive()
        assert client.call("event_history") == [], "normal authenticated traffic did not recover"
        assert store.conn.execute("SELECT COUNT(*) FROM structure_events").fetchone()[0] == 0
    finally:
        stop.set()
        if slow is not None:
            slow.close()
        elif raw is not None:
            raw.close()
        if sender is not None:
            sender.join(timeout=3)
        server.shutdown()
        serving.join(timeout=3)
        server.server_close()
        store.close()
    assert not serving.is_alive() and not sender.is_alive()
