# -*- coding: utf-8 -*-
"""coggraph 查看器两处安全缺陷的回归守卫（#240 Host 头 / #276 路径包含性）。

两处同属「本地服务 + 不可信 `viewer_graph.json`」的组合面，故断言**缺陷形态不存在**：

  A 组（#276 路径包含性）：`viewer_serve.tmpl::node_detail` 读正文前必须做包含性校验 ——
    `path` 含 `../`（逃出 MDCG_ROOT）或为绝对路径（`os.path.join` 直接丢弃 MDCG）时，
    均**不得**返回库外文件内容；库内路径必须照常读到（防把校验写成一刀切）。
  B 组（#240 Host 头）：服务只绑 127.0.0.1，但需挡 DNS rebinding ⇒ 非本机 Host
    必须 403；本机 Host（127.0.0.1 / localhost）必须照常 200。

回退本修复 ⇒ A/B 两组按标注转红。运行：
    python -X utf8 -m pytest tests/test_coggraph_viewer_host_and_path_guard.py -q
"""
from __future__ import annotations

import http.client
import importlib.util
import io
import json
import os
import socketserver
import sys
import threading
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COG = os.path.join(HERE, "tools", "coggraph")
TMPL = os.path.join(COG, "viewer_serve.tmpl")

INSIDE = "INSIDE-MDCG-CONTENT"
SECRET = "SECRET-OUTSIDE-MDCG"

#: 每个图都额外挂一个「库内对照节点」——用于**自证 MDCG_ROOT 真生效**。
#: 没有这条前置断言时，若 MDCG 被宿主环境覆盖成空，逃逸断言会因为「根本没读文件」而**假通过**。
CTRL_ID = "__ctrl_inside__"
CTRL_NODE = {"id": CTRL_ID, "path": "inside.md"}


def _load_serve(tmp_path, nodes, mdcg):
    """按 serve.py 的真实加载路径装配模块：tmpl 去掉 __main__ 段、其余逐字不变。"""
    src = io.open(TMPL, encoding="utf-8").read()
    head = src.split('if __name__ == "__main__":')[0]
    sp = os.path.join(str(tmp_path), "serve.py")
    with io.open(sp, "w", encoding="utf-8") as f:
        f.write(head)
    with io.open(os.path.join(str(tmp_path), "viewer_graph.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps({"nodes": list(nodes) + [CTRL_NODE]}))
    os.environ["MDCG_ROOT"] = mdcg
    # serve.py 把 `sys.argv[2]` 也当 MDCG_ROOT（命令行用法）。pytest 的宿主 argv
    # 长度 >2 ⇒ 会把 MDCG 覆盖成 pytest 的参数。临时收敛 argv，加载后还原。
    old_argv = sys.argv
    sys.argv = [sp]
    try:
        spec = importlib.util.spec_from_file_location("serve_guard_probe", sp)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
    finally:
        sys.argv = old_argv
    return m


def _fixture(tmp_path):
    sb = str(tmp_path)
    mdcg = os.path.join(sb, "mdcg")
    os.makedirs(mdcg, exist_ok=True)
    with io.open(os.path.join(mdcg, "inside.md"), "w", encoding="utf-8") as f:
        f.write(INSIDE)
    with io.open(os.path.join(sb, "secret_outside.txt"), "w", encoding="utf-8") as f:
        f.write(SECRET)
    return sb, mdcg


def _assert_mdcg_effective(m):
    """前置自证：MDCG_ROOT 必须真生效（对照组读得到库内内容）。

    没有这一步，「逃逸被拦」的断言在 MDCG 为空时也会通过（假通过）。
    """
    got = (m.node_detail(CTRL_ID).get("content") or "")
    assert INSIDE in got, (
        "前置失败：MDCG_ROOT 未生效，对照组读不到库内正文 -> %r（这是测试环境问题，不是缺陷结论）" % got)


# ---------------------------------------------------------------- A 组 #276
def test_relative_escape_is_blocked(tmp_path):
    """A1：`path = ../secret_outside.txt` 不得读到 MDCG 之外。"""
    sb, mdcg = _fixture(tmp_path)
    m = _load_serve(tmp_path, [{"id": "n1", "path": "../secret_outside.txt"}], mdcg)
    _assert_mdcg_effective(m)
    out = m.node_detail("n1")
    assert SECRET not in (out.get("content") or ""), \
        "路径穿越未被拦：读到了 MDCG 之外的内容 -> %r" % out.get("content")


def test_absolute_path_is_blocked(tmp_path):
    """A2：`path` 为绝对路径时 `os.path.join` 会丢弃 MDCG ⇒ 同样必须拦。"""
    sb, mdcg = _fixture(tmp_path)
    secret_abs = os.path.join(sb, "secret_outside.txt").replace("\\", "/")
    m = _load_serve(tmp_path, [{"id": "n1", "path": secret_abs}], mdcg)
    _assert_mdcg_effective(m)
    out = m.node_detail("n1")
    assert SECRET not in (out.get("content") or ""), \
        "绝对路径未被拦：读到了 MDCG 之外的内容 -> %r" % out.get("content")


def test_inside_path_still_reads(tmp_path):
    """A3（对照，防一刀切）：库内路径必须照常读到正文。"""
    sb, mdcg = _fixture(tmp_path)
    m = _load_serve(tmp_path, [], mdcg)
    _assert_mdcg_effective(m)


def test_sibling_prefix_not_treated_as_inside(tmp_path):
    """A4：`mdcg` 与 `mdcg_evil` 这种「同前缀不同目录」不得被判为库内。"""
    sb, mdcg = _fixture(tmp_path)
    evil = os.path.join(sb, "mdcg_evil")
    os.makedirs(evil, exist_ok=True)
    with io.open(os.path.join(evil, "x.md"), "w", encoding="utf-8") as f:
        f.write(SECRET)
    m = _load_serve(tmp_path, [{"id": "n1", "path": "../mdcg_evil/x.md"}], mdcg)
    _assert_mdcg_effective(m)
    out = m.node_detail("n1")
    assert SECRET not in (out.get("content") or ""), \
        "同前缀目录被误判为库内 -> %r" % out.get("content")


# ---------------------------------------------------------------- B 组 #240
class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def _with_server(tmp_path, nodes, mdcg, fn):
    m = _load_serve(tmp_path, nodes, mdcg)
    _assert_mdcg_effective(m)
    httpd = _Server(("127.0.0.1", 0), m.H)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        time.sleep(0.2)
        return fn(port)
    finally:
        httpd.shutdown()
        httpd.server_close()


def _get(port, path, host=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("GET", path, headers=({"Host": host} if host else {}))
    r = c.getresponse()
    return r.status, r.read().decode("utf-8", "replace")


def test_foreign_host_is_403(tmp_path):
    """B1：非本机 Host 必须 403（DNS rebinding 面）。"""
    sb, mdcg = _fixture(tmp_path)

    def run(port):
        return _get(port, "/api/node?id=" + CTRL_ID, host="rebind.attacker.example:%d" % port)

    st, body = _with_server(tmp_path, [], mdcg, run)
    assert st == 403, "伪造 Host 未被拒：status=%s" % st
    assert INSIDE not in body, "403 响应里仍带出了正文"


def test_local_host_still_works(tmp_path):
    """B2（对照）：本机 Host 必须照常 200 并返回正文。"""
    sb, mdcg = _fixture(tmp_path)

    def run(port):
        return _get(port, "/api/node?id=" + CTRL_ID, host="127.0.0.1:%d" % port)

    st, body = _with_server(tmp_path, [], mdcg, run)
    assert st == 200, "本机 Host 被误拒：status=%s" % st
    assert INSIDE in body, "本机 Host 未返回正文"


def test_localhost_host_still_works(tmp_path):
    """B3（对照）：`localhost` 同样在允许集内。"""
    sb, mdcg = _fixture(tmp_path)

    def run(port):
        return _get(port, "/api/node?id=" + CTRL_ID, host="localhost:%d" % port)

    st, body = _with_server(tmp_path, [], mdcg, run)
    assert st == 200, "localhost 被误拒：status=%s" % st
    assert INSIDE in body


def test_static_file_also_host_checked(tmp_path):
    """B4：静态文件（viewer_graph.json）同样受 Host 校验保护。"""
    sb, mdcg = _fixture(tmp_path)

    def run(port):
        return _get(port, "/viewer_graph.json", host="rebind.attacker.example:%d" % port)

    st, _body = _with_server(tmp_path, [], mdcg, run)
    assert st == 403, "静态文件未受 Host 校验：status=%s" % st
