"""真实启动解析与业务授权；TLS/ACL 的原回归在邻接批次单列。"""
import json
from pathlib import Path
import ssl

import pytest

from lingshu.core import structure_service as service
from lingshu.core.structure_writer import PermissionDenied, WriteError


@pytest.fixture
def configured(tmp_path, monkeypatch):
    root = tmp_path / "service"
    root.mkdir()
    contexts, servers = [], []

    class ConfigurationTLS:
        verify_mode = ssl.CERT_REQUIRED

        def load_cert_chain(self, *args):
            pass

    def tls_factory(*args, **kwargs):
        contexts.append(True)
        return ConfigurationTLS()

    # 量具只隔离操作系统 ACL 与证书装配；真实 load_service、ActorContext、
    # LayeredStore、StructureWriter 和 dispatch 不替换，不宣称本夹具验证 TLS。
    monkeypatch.setattr(service, "check_protected_directory", lambda path, files=(): Path(path).resolve())
    monkeypatch.setattr(service.ssl, "create_default_context", tls_factory)
    config = {"database": "body.sqlite3", "certificate": "server.pem", "key": "server.key", "ca": "ca.pem",
              "principals": [{"identity": "urn:lingshu:executor", "subject": "executor", "namespace": "lab",
                              "source": "executor", "capabilities": ["versions.report"], "instances": ["instance-1"]}]}

    def load(field=None, value=None, omit=False):
        grant = config["principals"][0]
        if field is not None:
            if omit:
                grant.pop(field, None)
            else:
                grant[field] = value
        path = root / "service.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        server = service.load_service(path, bind_and_activate=False)
        servers.append(server)
        return server

    yield load, root, contexts, config
    for server in servers:
        server.server_close()
        server.writer.store.close()


@pytest.mark.parametrize("field,value", [
    ("instances", "instance-1"), ("instances", {"instance-1": True}),
    ("instances", [["instance-1"]]), ("instances", None), ("instances", [""]),
    ("capabilities", "versions.report"), ("capabilities", {"versions.report": True}),
    ("capabilities", [["versions.report"]]), ("capabilities", None), ("capabilities", [" "]),
], ids=["instances-string", "instances-object", "instances-nested", "instances-null", "instances-empty",
        "capabilities-string", "capabilities-object", "capabilities-nested", "capabilities-null", "capabilities-empty"])
def test_malformed_grant_is_rejected_before_startup(configured, field, value):
    load, root, contexts, config = configured
    with pytest.raises(WriteError, match=field):
        load(field, value)
    assert contexts == [], "malformed grant reached TLS initialization"
    assert not (root / "body.sqlite3").exists(), "malformed grant opened persistent storage"


def _report(server, instance, report_id):
    actor = server.principals["urn:lingshu:executor"]
    return service.dispatch(server.writer, actor, {"command": "record_version_result", "arguments": {
        "report_id": report_id, "run_id": "run-1", "instance_id": instance,
        "to_version": "v2", "outcome": "applied"}})


def test_executor_is_authorized_for_exact_instance_only(configured):
    load, root, contexts, config = configured
    server = load()
    receipt = _report(server, "instance-1", "correct")
    assert receipt["origin"] == "executor_report" and not receipt["replayed"]
    with pytest.raises(PermissionDenied):
        _report(server, "i", "wrong")
    assert server.writer.store.conn.execute("SELECT COUNT(*) FROM structure_events").fetchone()[0] == 1


def test_omitted_instances_grants_no_instance(configured):
    load, root, contexts, config = configured
    server = load("instances", omit=True)
    assert not server.principals["urn:lingshu:executor"].instances
    with pytest.raises(PermissionDenied):
        _report(server, "instance-1", "no-instance")


def test_array_names_are_not_trimmed_or_rewritten(configured):
    load, root, contexts, config = configured
    names = [" instance-1 ", " instance-1 "]
    server = load("instances", names)
    assert server.principals["urn:lingshu:executor"].instances == frozenset({" instance-1 "})
    assert config["principals"][0]["instances"] == names
    with pytest.raises(PermissionDenied):
        _report(server, "instance-1", "trimmed")
    assert _report(server, " instance-1 ", "exact-spaces")["origin"] == "executor_report"
