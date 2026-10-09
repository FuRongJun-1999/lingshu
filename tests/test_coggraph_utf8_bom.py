#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""coggraph BOM 守卫：带 UTF-8 BOM 的记忆文件必须与无 BOM 同等解析（issue #412）

standalone，无 pytest 依赖：
    python -X utf8 tests/test_coggraph_utf8_bom.py

缺陷：parse_md 用 encoding="utf-8" 读文件（不剥 BOM），FM_RE = r"^---\n..." 要求
文件首字符是 `-`。文件以 U+FEFF 开头时匹配失败 ⇒ 节点计入 skipped_md、退出码仍 0，
下游 derive/naming/viewer 整条链路看不到这些记忆。

修复：四处读取点改 encoding="utf-8-sig"——
  export_coggraph.parse_md（记忆 .md 入口）
  derive_edges R5（读正文参与互引，BOM 会让 FMHEAD 失配、前言混进正文）
  viewer_serve.tmpl::node_detail（BOM 让 startswith("---") 判否，前言整段展示）
  session_chains（读 _audit.jsonl，BOM 让首行 json.loads 抛异常被吞，首条事件丢失）

断言组：
  A 组  export：LF / CRLF / UTF8-BOM 三种编码读数一致（nodes=2 edges=1 skipped_md=0）
  B 组  BOM 文件的 title/tags 正确解析（BOM 不残留进 title）
  C 组  derive：BOM 记忆的 R3/R4 链路与无 BOM 一致（不被整篇跳过）
  D 组  session_chains：带 BOM 的 _audit.jsonl 首条 add 事件不丢（会话节点数一致）
  E 组  夹具自证：修复前语义（纯 utf-8 重读同文件）BOM 文件解析失败 ⇒ 可区分
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(REPO, "tools", "coggraph", "export_coggraph.py")
DER = os.path.join(REPO, "tools", "coggraph", "derive_edges.py")
SESS = os.path.join(REPO, "tools", "coggraph", "session_chains.py")

_PASS, _FAIL = [], []


def ok(cond, msg, extra=""):
    (_PASS if cond else _FAIL).append(msg)
    if not cond:
        print("[FAIL] %s%s" % (msg, ("   " + str(extra)) if extra else ""))


def run(cmd, **kw):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    # PyYAML 非内置依赖：优先用仓库环境；本测试自带 lab 目录时通过 LAB_PYLIBS 注入
    lab_libs = os.environ.get("LAB_PYLIBS")
    if lab_libs:
        env["PYTHONPATH"] = lab_libs + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", cwd=REPO, env=env, **kw)


MD1 = "---\nid: n1\nlayer: knowledge\nbucket: kb\ntags: [shape, red]\ncreated: 1759930000\nedges:\n  - {type: similar, target: n2}\n---\n# 功能名：笔记一\n正文一\n"
MD2 = "---\nid: n2\nlayer: knowledge\nbucket: kb\ntags: [shape, blue]\ncreated: 1759930100\n---\n# 功能名：笔记二\n正文二\n"


def make_root(base, bom=False, tag=""):
    root = os.path.join(base, "root" + ("_bom" if bom else "") + tag)
    os.makedirs(os.path.join(root, "knowledge"), exist_ok=True)
    for fn, s in (("n1.md", MD1), ("n2.md", MD2)):
        data = s.encode("utf-8")
        if bom:
            data = b"\xef\xbb\xbf" + data
        with open(os.path.join(root, "knowledge", fn), "wb") as f:
            f.write(data)
    return root


def export(root, out):
    p = run([sys.executable, "-X", "utf8", EXP, "--root", root, "--out", out, "--write"])
    doc = None
    gp = os.path.join(out, "graph.json")
    if os.path.isfile(gp):
        with open(gp, encoding="utf-8") as f:
            doc = json.load(f)
    return p, doc


def a_group(tmp):
    """LF / CRLF / UTF8-BOM 三种编码导出读数一致。"""
    readings = {}
    for label, conv in (("LF", lambda s: s.encode("utf-8")),
                        ("CRLF", lambda s: s.replace("\n", "\r\n").encode("utf-8")),
                        ("UTF8-BOM", lambda s: b"\xef\xbb\xbf" + s.encode("utf-8"))):
        root = os.path.join(tmp, "root_" + label)
        out = os.path.join(tmp, "out_" + label)
        os.makedirs(os.path.join(root, "knowledge"))
        for fn, s in (("n1.md", MD1), ("n2.md", MD2)):
            with open(os.path.join(root, "knowledge", fn), "wb") as f:
                f.write(conv(s))
        p, doc = export(root, out)
        ok(p.returncode == 0, "A %s 导出退出码 0" % label, p.stderr[-200:])
        c = doc["meta"]["counts"] if doc else {}
        readings[label] = (c.get("nodes"), c.get("edges"), c.get("skipped_md"))
        ok(readings[label] == (2, 1, 0),
           "A %s 读数 nodes=2 edges=1 skipped_md=0（got %s）" % (label, readings[label]), readings[label])
    ok(readings["UTF8-BOM"] == readings["LF"],
       "A BOM 与 LF 读数完全一致", (readings["UTF8-BOM"], readings["LF"]))


def b_group(tmp):
    """BOM 文件的 title 正确解析、不残留 U+FEFF。"""
    root = make_root(tmp, bom=True)
    out = os.path.join(tmp, "out_b")
    p, doc = export(root, out)
    titles = sorted(n["title"] for n in doc["nodes"]) if doc else []
    ok(titles == ["笔记一", "笔记二"], "B BOM 文件 title 正常解析（TITLE_RE 取「功能名：」后内容）", titles)
    ok(not any("\ufeff" in (n.get("title") or "") for n in (doc["nodes"] if doc else [])),
       "B title 无 U+FEFF 残留")


def c_group(tmp):
    """derive：BOM 记忆与无 BOM 记忆产出一致（R2 同桶边存在）。"""
    for bom in (False, True):
        root = make_root(tmp, bom=bom, tag="_c")
        out1 = os.path.join(tmp, "c_out1_%s" % bom)
        out2 = os.path.join(tmp, "c_out2_%s" % bom)
        p1, doc = export(root, out1)
        p2 = run([sys.executable, "-X", "utf8", DER, "--graph", os.path.join(out1, "graph.json"),
                  "--out", out2, "--write", "--small-bucket", "8"])
        ok(p1.returncode == 0 and p2.returncode == 0, "C bom=%s 管线退出码 0" % bom, p2.stderr[-200:])
        with open(os.path.join(out2, "derived_edges.json"), encoding="utf-8") as f:
            d = json.load(f)
        rules = d.get("meta", {}).get("rules", {})
        ok(rules.get("R1_same_bucket") == 2, "C bom=%s R1 同桶边=2（记忆未被跳过）" % bom,
           rules.get("R1_same_bucket"))


def d_group(tmp):
    """session_chains：带 BOM 的 _audit.jsonl 首条 add 事件不丢。"""
    audit = os.path.join(tmp, "_audit.jsonl")
    lines = [
        json.dumps({"op": "add", "id": "n1", "t": 1759930000, "session": "s1"}, ensure_ascii=False),
        json.dumps({"op": "add", "id": "n2", "t": 1759930100, "session": "s1"}, ensure_ascii=False),
    ]
    with open(audit, "wb") as f:
        f.write(b"\xef\xbb\xbf" + ("\n".join(lines) + "\n").encode("utf-8"))
    root = make_root(tmp, bom=True)
    out1 = os.path.join(tmp, "d_out1")
    export(root, out1)
    out2 = os.path.join(tmp, "d_out2")
    p = run([sys.executable, "-X", "utf8", SESS, "--audit", audit,
             "--graph", os.path.join(out1, "graph.json"), "--out", out2, "--write", "--min-chain", "2"])
    ok(p.returncode == 0, "D session_chains 退出码 0", p.stderr[-200:])
    sp = os.path.join(out2, "session_chains.json")
    if os.path.isfile(sp):
        with open(sp, encoding="utf-8") as f:
            sc = json.load(f)
        n1 = [c["nodes"] for c in sc.get("chains", []) if c["session"] == "s1"]
        ok(n1 and n1[0] == 2, "D BOM 首行事件未丢：会话 s1 含 2 节点（修复前为 1）", n1)
        # E 组自证：按旧语义（纯 utf-8）重读，首行应解析失败
        with open(audit, "rb") as f:
            raw = f.read()
        first = raw.split(b"\n")[0].decode("utf-8")
        try:
            json.loads(first)
            old_ok = True
        except Exception:
            old_ok = False
        ok(not old_ok, "E 自证：旧口径（纯 utf-8）下 BOM 首行确实解析失败", first[:40])


def main():
    tmp = tempfile.mkdtemp(prefix="coggraph_bom_")
    try:
        a_group(tmp)
        b_group(tmp)
        c_group(tmp)
        d_group(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    total = len(_PASS) + len(_FAIL)
    print("")
    print("===== SUMMARY %d/%d 通过 =====" % (len(_PASS), total))
    if _FAIL:
        print("失败项：")
        for m in _FAIL:
            print("  - " + m)
        return 1
    print("VERDICT=PASS（UTF-8 BOM 记忆与无 BOM 同等解析）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
