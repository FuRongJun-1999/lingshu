"""Real stdio subprocess guards for issue #143; synthetic data only."""
from concurrent.futures import ThreadPoolExecutor
import sys
import threading

import pytest

from lingshu.world.brain_store import BrainError, MCPClient, connect


SERVER = '''import json, os, sys, time
mode = sys.argv[1] if len(sys.argv) > 1 else "normal"
def send(obj):
    print(json.dumps(obj), flush=True)
def log(size, suffix=""):
    sys.stderr.buffer.write(b"W" * size + suffix.encode("utf-8"))
    sys.stderr.buffer.flush()
for line in sys.stdin:
    message = json.loads(line)
    if message.get("method") == "initialize":
        if mode == "initial-silent":
            time.sleep(60)
        if mode == "initial-log":
            log(200000)
        info = {} if mode == "no-info" else {"name": "transport-fixture"}
        send({"jsonrpc": "2.0", "id": message["id"], "result": {"serverInfo": info}})
        if mode == "blocked-stdin":
            time.sleep(60)
    elif message.get("method") == "tools/call":
        name = message["params"]["name"]
        args = message["params"]["arguments"]
        if name == "silent":
            log(0, "timeout-diagnostic")
            time.sleep(60)
        if name == "chatter":
            while True:
                print("not JSON", flush=True)
                time.sleep(0.01)
        if name == "exit-error":
            log(200000, "诊断尾部-END-MARKER")
            raise SystemExit(1)
        if name == "close-stdout":
            log(0, "stdout-closed")
            os.close(sys.stdout.fileno())
            time.sleep(60)
        if name == "invalid-utf8":
            sys.stderr.buffer.write(b"\\xff\\xfe" * 100000)
            sys.stderr.buffer.flush()
        if name == "slow":
            time.sleep(0.3)
        log(args.get("stderr_bytes", 0))
        if name == "rpc-error":
            send({"jsonrpc": "2.0", "id": message["id"], "error": {"code": -32000, "message": "fixture-error"}})
        else:
            text = "not JSON" if name == "bad-json" else json.dumps(args)
            send({"jsonrpc": "2.0", "id": message["id"], "result": {
                "isError": name == "tool-error", "content": [{"text": text}]}})
if mode == "stubborn-close":
    time.sleep(60)
'''


def bounded(operation, client):
    """Fail old blocking implementations without leaving their test process alive."""
    outcome = []

    def run():
        try:
            outcome.append((True, operation()))
        except BaseException as exc:
            outcome.append((False, exc))

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout=3)
    if worker.is_alive():
        process = getattr(client, "_p", None)
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=3)
        worker.join(timeout=3)
        pytest.fail("MCP operation remained blocked for three seconds")
    success, value = outcome[0]
    if not success:
        raise value
    return value


@pytest.fixture
def transport(tmp_path):
    script = tmp_path / "mcp_fixture.py"
    script.write_text(SERVER, encoding="utf-8")
    clients = []

    def create(mode="normal", **kwargs):
        client = MCPClient.__new__(MCPClient)
        clients.append(client)
        bounded(lambda: client.__init__(python=sys.executable, pythonpath=str(tmp_path),
                                       root=str(tmp_path), args=[str(script), mode], **kwargs), client)
        return client

    create.clients = clients
    yield create
    for client in clients:
        process = getattr(client, "_p", None)
        if process is None:
            continue
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)
        client.close()
        for stream in (process.stdin, process.stdout, process.stderr):
            stream.close()


@pytest.mark.parametrize("size", [1000, 200000, 2000000])
def test_tool_response_survives_stderr_backpressure(transport, size):
    client = transport()
    args = {"stderr_bytes": size, "token": "tool-response"}
    assert bounded(lambda: client.call("echo", args), client) == args


def test_initialize_survives_stderr_backpressure(transport):
    client = transport("initial-log")
    assert client.server_info["name"] == "transport-fixture"
    assert bounded(lambda: client.call("echo", {"token": "after-init"}), client) == {"token": "after-init"}


def test_repeated_large_logs_keep_responses_separate(transport):
    client = transport()
    for i in range(3):
        args = {"stderr_bytes": 200000, "token": i}
        assert bounded(lambda: client.call("echo", args), client) == args


def test_concurrent_callers_keep_their_own_response(transport):
    client = transport()
    start = threading.Barrier(4)

    def call(token):
        start.wait(timeout=3)
        args = {"stderr_bytes": 200000, "token": token}
        return bounded(lambda: client.call("echo", args), client)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(call, range(4)))
    assert [result["token"] for result in results] == list(range(4))


def test_process_error_includes_recent_unicode_diagnostic(transport):
    client = transport()
    with pytest.raises(BrainError) as error:
        bounded(lambda: client.call("exit-error", {}), client)
    assert "诊断尾部-END-MARKER" in str(error.value)
    assert len(str(error.value)) < 500


def test_invalid_utf8_stderr_does_not_block_rpc(transport):
    client = transport()
    args = {"token": "binary-log"}
    assert bounded(lambda: client.call("invalid-utf8", args), client) == args


@pytest.mark.parametrize("name", ["rpc-error", "tool-error", "bad-json"])
def test_tool_errors_leave_transport_usable(transport, name):
    client = transport()
    with pytest.raises(BrainError):
        bounded(lambda: client.call(name, {}), client)
    assert bounded(lambda: client.call("echo", {"token": "after-error"}), client) == {"token": "after-error"}


@pytest.mark.parametrize("circular", [False, True])
def test_unserializable_arguments_leave_transport_usable(transport, circular):
    client = transport()
    args = {}
    args["bad"] = args if circular else object()
    with pytest.raises(ValueError if circular else TypeError):
        bounded(lambda: client.call("echo", args), client)
    assert bounded(lambda: client.call("echo", {"token": "after-input-error"}), client) == {
        "token": "after-input-error"}


def test_close_reaps_process_and_is_idempotent(transport):
    client = transport()
    client.close()
    client.close()
    assert client._p.poll() is not None
    assert all(s.closed for s in (client._p.stdin, client._p.stdout, client._p.stderr))


def test_failed_initialize_reaps_process(transport):
    with pytest.raises(BrainError, match="serverInfo"):
        transport("no-info")
    assert transport.clients[-1]._p.poll() is not None


def test_stubborn_close_reaps_process(transport):
    client = transport("stubborn-close")
    bounded(client.close, client)
    assert client._p.poll() is not None


def test_deadline_initialize_reaps_unresponsive_process(transport):
    with pytest.raises(BrainError, match="超时"):
        transport("initial-silent", request_timeout=0.2)
    assert transport.clients[-1]._p.poll() is not None


@pytest.mark.parametrize("name", ["silent", "chatter", "close-stdout"])
def test_deadline_failed_call_reaps_client_and_prevents_reuse(transport, name):
    client = transport(request_timeout=0.2)
    with pytest.raises(BrainError) as error:
        bounded(lambda: client.call(name, {}), client)
    if name in {"silent", "chatter"}:
        assert "超时" in str(error.value)
    if name == "silent":
        assert "timeout-diagnostic" in str(error.value)
    assert client._p.poll() is not None
    with pytest.raises(BrainError, match="关闭"):
        bounded(lambda: client.call("echo", {"token": "late"}), client)


def test_deadline_also_bounds_blocked_request_write(transport):
    client = transport("blocked-stdin", request_timeout=0.2)
    with pytest.raises(BrainError, match="超时"):
        bounded(lambda: client.call("echo", {"payload": "x" * 2000000}), client)
    assert client._p.poll() is not None


def test_deadline_none_allows_slow_valid_response(transport):
    client = transport(request_timeout=None)
    args = {"token": "slow-valid"}
    assert bounded(lambda: client.call("slow", args), client) == args


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_deadline_invalid_budget_does_not_spawn_process(transport, timeout):
    with pytest.raises(ValueError, match="request_timeout"):
        transport(request_timeout=timeout)
    assert not hasattr(transport.clients[-1], "_p")


def test_deadline_connect_forwards_budget(tmp_path, transport):
    package = tmp_path / "md_cg"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "mcp_server.py").write_text(SERVER, encoding="utf-8")
    agent = connect(python=sys.executable, pythonpath=str(tmp_path), root=str(tmp_path), request_timeout=0.2)
    client = agent.store.client
    transport.clients.append(client)
    with pytest.raises(BrainError, match="超时"):
        bounded(lambda: client.call("silent", {}), client)
    assert client._p.poll() is not None
