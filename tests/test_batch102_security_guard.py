# -*- coding: utf-8 -*-
"""批一百零二守卫：两处安全面收口（#2563 / #2564）。

覆盖
--------------------------------------------------------------------------
#2563 脑侧 MCP 子进程必须以 `-P` 启动（不把 cwd 插入 `sys.path[0]`）
    G1 源码级：`Popen` 的 argv 含 `-P`
       （改前 `[python, "-X", "utf8", "-m", "md_cg.mcp_server"]` —— `-m` 把
        进程 cwd 插到 `sys.path[0]`，排在 `PYTHONPATH` 之前 ⇒ cwd 里的毒
        `md_cg` 抢先执行，即使调用方正确指定了 `--brain-path`）
    G2 行为级：在含毒 `md_cg` 的 cwd 下，带 `-P` 启动必须**不执行**毒包
       （改前 `MARKER_ATTACKER_EXECUTED` 被打印）
    G3 正对照：合法 `PYTHONPATH` 指向真实包时，带 `-P` 仍能找到模块

#2564 `_brain_env_file` 的保留键检查必须大小写不敏感
    G4 `{"MDCG_root": ...}` / `{"MDCG_STG_state": ...}` 必须 ValueError
       （改前放行 ⇒ Windows 环境块名不区分大小写 ⇒ 脑端库根被重定向）
    G5 正对照：精确保留键仍拒、合法 `MDCG_TENANT` 仍通过

运行（lingshu 仓根）：
    python -X utf8 -m pytest tests/test_batch102_security_guard.py -q
"""
from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import tempfile

import pytest

from lingshu.world import brain_store as BS
from lingshu.world import scenario_cli as SC


# ============================================ #2563 脑侧子进程 -P
def test_G1_popen_has_isolated_flag():
    """G1：`Popen` 的 argv 必须含 `-P`（改前无 ⇒ cwd 插到 sys.path[0]）。"""
    src = inspect.getsource(BS.MCPClient.__init__)
    i = src.find("subprocess.Popen")
    assert i > 0, "未找到 Popen"
    seg = src[i:i + 400]
    assert '"-P"' in seg, "Popen argv 缺 -P：%r" % seg[:220]


def test_G2_poisoned_cwd_not_executed(tmp_path):
    """G2：含毒 `md_cg` 的 cwd 下，带 `-P` 启动必须不执行毒包。"""
    pkg = tmp_path / "md_cg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "mcp_server.py").write_text(
        "print('MARKER_ATTACKER_EXECUTED')\n", encoding="utf-8")
    # 干净 PYTHONPATH（空目录），确保只有 cwd 里能找到 md_cg
    empty = tmp_path / "empty"
    empty.mkdir()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(empty)
    r = subprocess.run(
        [sys.executable, "-X", "utf8", "-P", "-m", "md_cg.mcp_server"],
        cwd=str(tmp_path), capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=env, timeout=30)
    assert "MARKER_ATTACKER_EXECUTED" not in (r.stdout or ""), \
        "带 -P 仍执行了 cwd 毒包：%r" % (r.stdout,)
    assert r.returncode != 0, "带 -P 应因找不到模块而失败"


def test_G3_pythonpath_still_works(tmp_path):
    """G3 正对照：合法 `PYTHONPATH` 指向真实包时，带 `-P` 仍能找到模块。"""
    pkg = tmp_path / "legitpkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "mod.py").write_text("print('LEGIT_OK')\n", encoding="utf-8")
    env = dict(os.environ)
    # PYTHONPATH 必须指向**包父目录**（含 legitpkg/ 的那层），不是包目录本身
    env["PYTHONPATH"] = str(tmp_path)
    r = subprocess.run(
        [sys.executable, "-X", "utf8", "-P", "-m", "legitpkg.mod"],
        cwd=str(tmp_path), capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=env, timeout=30)
    assert "LEGIT_OK" in (r.stdout or ""), (r.returncode, r.stdout, r.stderr)


# ============================================ #2564 保留键大小写
@pytest.mark.parametrize("payload", [
    {"MDCG_root": "/evil"},
    {"MDCG_STG_state": "0"},
    {"MDCG_Root": "/evil"},
])
def test_G4_case_variant_reserved_keys_rejected(tmp_path, payload):
    """G4：保留键的大小写变体必须拒绝（改前小写变体被放行）。"""
    p = tmp_path / "env.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError) as ei:
        SC._brain_env_file(str(p))
    assert "reserved" in str(ei.value), str(ei.value)


def test_G5_exact_keys_and_legal_still_work(tmp_path):
    """G5 正对照：精确保留键仍拒、合法 `MDCG_TENANT` 仍通过。"""
    p = tmp_path / "a.json"
    p.write_text(json.dumps({"MDCG_ROOT": "/evil"}), encoding="utf-8")
    with pytest.raises(ValueError):
        SC._brain_env_file(str(p))
    p2 = tmp_path / "b.json"
    p2.write_text(json.dumps({"MDCG_TENANT": "t1"}), encoding="utf-8")
    assert SC._brain_env_file(str(p2)) == {"MDCG_TENANT": "t1"}
