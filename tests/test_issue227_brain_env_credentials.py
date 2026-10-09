# -*- coding: utf-8 -*-
"""test_issue227_brain_env_credentials · lingshu issue #227 守卫（P0 · 凭据泄露族）
============================================================================
缺陷（lingshu issue #227）：`brain_store._brain_env` 构造脑侧子进程 env 时只
**剔除**继承的 `MDCG_*`（黑名单），其余环境变量原样交给子进程。脑侧子进程**由
模型驱动**，于是 `AEIS_DESIGNER_KEY` / `*_API_KEY` / `*_TOKEN` / `*_SECRET` /
`*_PASSWORD` 等凭据只要名字不带 `MDCG_` 就落进子进程可读面（`os.environ`）。

复现读数（修复前，本机真实环境）：
  子进程 env 含 `AEIS_DESIGNER_KEY` / `DEEPSEEK_API_KEY` / `OPENROUTER_API_KEY`
  / `BOCHA_API_KEY` / `YOUR_STEP_API_KEY` / `ZAI_OAUTH_CLIENT_ID` / `IGCCSVC_DB`
  —— 均为父进程真实凭据，原样透传。

修法：**白名单放行**（fail-closed）替代黑名单剔除——只继承
`_ENV_ALLOW_NAMES`（进程运行基座：PATH/PYTHON*/临时目录/区域… ＋ 两个**具名**部署
配置项 `HIVE_JOBS_DIR`/`HIVE_EXE`）与 `_ENV_ALLOW_PREFIXES`（`PYTHON` / `LINGSHU_` /
`LC_`）；继承 `MDCG_*` 仍剔除（防在役库被无意命中）。凭据若确需传入，必须经
`extra_env` **显式**声明。选白名单而非「黑名单拦 `*_KEY/*_TOKEN/...`」的理由：黑名单
对凭据面天然 fail-open——新出现的凭据名只要不撞模式就照旧泄露（本例 `IGCCSVC_DB`、
`YOUR_STEP_API_KEY` 已证）；白名单是「不声明就不给」。

#227 **续**（本轮扩展）：原修法把 `HIVE_` 整条列入 `_ENV_ALLOW_PREFIXES`，前缀过宽
⇒ `HIVE_API_KEY` / `HIVE_ORCH_TOKEN` / `HIVE_ORCH_TOKEN_FILE` /
`HIVE_SUBAGENT_API_KEY` 这些**凭据形状名**整体放行给脑侧子进程。复核读数
**4 failed / 3 passed**（W2/W3/W6/W7 红并逐名点名）。收敛：移除 `HIVE_` 前缀，
只**具名**放行脑侧真读的两个非凭据名 `HIVE_JOBS_DIR`/`HIVE_EXE`。判定证据见 W8。

断言组：
  W1 判据单测：`_env_is_allowed` 的形状表（凭据名一律 False，运行基座 True）。
  W2 `_brain_env` 直读：注入的凭据哨兵一个都不出现。
  W3 **真实父环境动态扫描**：父环境里任何「凭据形状」的名字都不得出现在子进程 env。
  W4 **反向断言**（防修死）：必需项仍在——PATH / PYTHONPATH（给定值）/
     PYTHONUTF8 / PYTHONIOENCODING / MDCG_ROOT（给定值）/ MDCG_STG_STATE="1"，
     且具名 `HIVE_JOBS_DIR`/`HIVE_EXE` 与 `LC_*` / `PYTHON*` 前缀面仍透传。
  W5 `extra_env` **显式**注入口仍有效（含 MDCG_* 注入与 None 删除）。
  W6 **端到端**：`MCPClient` 真起子进程，子进程把自己 `os.environ` 报回，
     断言里面无凭据（证明封的是「真被 spawn 的子进程」而不只是 `_brain_env` 的返回值）。
  W7 **判据有牙自证**：用旧法（只剔 `MDCG_*`）构造的 env 必被 `_credential_leaks`
     点名（否则本守卫对缺陷无感）。
  W8 **#227 续·HIVE_ 前缀收敛**（另立一条并写明理由）：四个凭据形状的 `HIVE_*`
     名不进子进程 env；脑侧真读的具名 `HIVE_JOBS_DIR`/`HIVE_EXE` 仍透传。

运行（lingshu 仓根）：python -X utf8 tests/test_issue227_brain_env_credentials.py
                    / python -X utf8 -m pytest tests/test_issue227_brain_env_credentials.py
退出码：0 = 全过；1 = 有断言失败
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lingshu.world.brain_store import (  # noqa: E402
    _ENV_ALLOW_NAMES, _ENV_ALLOW_PREFIXES, _brain_env, _env_is_allowed)

#: 凭据形状判据——**只列不会误伤白名单名的标记**（白名单名无一含这些词）。
_CRED_TOKENS = frozenset({"KEY", "KEYS", "TOKEN", "SECRET", "SECRETS",
                          "PASSWORD", "PASSWD", "CREDENTIAL", "CREDENTIALS"})
_CRED_PREFIXES = ("AEIS_",)


def _looks_like_credential(name: str) -> bool:
    """名字是否「凭据形状」（整词命中敏感词 / AEIS_ 前缀）。"""
    up = str(name).upper()
    if up.startswith(_CRED_PREFIXES):
        return True
    return any(tok in up.split("_") for tok in _CRED_TOKENS)


def _credential_leaks(env: dict) -> list:
    """env 中「凭据形状」的名字列表（空 = 无凭据面）。"""
    return sorted(k for k in env if _looks_like_credential(k))


#: 本守卫自用的凭据哨兵（**假值**，绝不碰真密钥）。
#: 末四项为 #227 续新增：`HIVE_` 前缀曾被整条放行，这四个凭据形状名即被透传
#: 的实证对象（W2/W3/W6 由此覆盖它们；W8 再另立一条具名核对）。
_SENTINELS = {
    "AEIS_DESIGNER_KEY": "SENTINEL-DESIGNER",
    "AEIS_SIGNING_TOKEN": "SENTINEL-SIGNING",
    "MY_SERVICE_TOKEN": "SENTINEL-SERVICE",
    "DB_PASSWORD": "SENTINEL-DB",
    "OPENAI_API_KEY": "SENTINEL-OPENAI",
    "SOME_SECRET": "SENTINEL-SECRET",
    "AWS_SESSION_CREDENTIAL": "SENTINEL-AWS",
    "HIVE_API_KEY": "SENTINEL-HIVE-API-KEY",
    "HIVE_ORCH_TOKEN": "SENTINEL-HIVE-ORCH-TOKEN",
    "HIVE_ORCH_TOKEN_FILE": "SENTINEL-HIVE-ORCH-TOKEN-FILE",
    "HIVE_SUBAGENT_API_KEY": "SENTINEL-HIVE-SUBAGENT-API-KEY",
}

#: #227 续：`HIVE_` 前缀收敛的**具名**对照表。
_HIVE_CRED_NAMES = ("HIVE_API_KEY", "HIVE_ORCH_TOKEN", "HIVE_ORCH_TOKEN_FILE",
                    "HIVE_SUBAGENT_API_KEY")
#: 脑侧子进程真读的两个**非凭据** `HIVE_*`（`md_cg/units.py` 的 ENV_JOBS_DIR /
#: ENV_EXE，经 `cg(op=ccg, action=units)` 工具面可达）——反向腿须仍透传。
_HIVE_CONFIG_NAMES = ("HIVE_JOBS_DIR", "HIVE_EXE")


def test_w1_env_is_allowed_shape_table():
    """W1 判据单测：凭据名一律 False；运行基座与显式配置面 True。"""
    for name in _SENTINELS:
        assert _env_is_allowed(name) is False, name
    for name in ("DEEPSEEK_API_KEY", "IGCCSVC_DB", "MY_RANDOM_VAR",
                 "ZAI_OAUTH_CLIENT_ID", "GITHUB_PAT"):
        assert _env_is_allowed(name) is False, name
    for name in ("PATH", "PATHEXT", "COMSPEC", "SYSTEMROOT", "TEMP", "TMP",
                 "HOME", "USERPROFILE", "LANG", "LC_CTYPE", "PYTHONPATH",
                 "PYTHONUTF8", "PYTHONIOENCODING",
                 "HIVE_JOBS_DIR", "HIVE_EXE",          # #227 续：具名放行（见 W8）
                 "LINGSHU_COMPONENT_ROOT"):
        assert _env_is_allowed(name) is True, name
    # #227 续：`HIVE_` 前缀已移除——凭据形状的 `HIVE_*` 与 hive 服务/执行器侧角色名
    # （`HIVE_ROLE`/`HIVE_NODE_ID` 由 hive/verify_runner.py 读，**不在**脑侧子进程面）
    # 一律 False；`HIVE_EXEC_PY` 等执行器配置名同理，确需者走 extra_env。
    for name in _HIVE_CRED_NAMES + ("HIVE_ROLE", "HIVE_NODE_ID", "HIVE_ANY",
                                    "HIVE_EXEC_PY", "HIVE_ORCH_PY"):
        assert _env_is_allowed(name) is False, name
    # 大小写无关（Windows 上 os.environ 键为大写；POSIX 上原样）
    assert _env_is_allowed("pythonpath") is True
    assert _env_is_allowed("aeis_designer_key") is False


def test_w2_brain_env_drops_credential_sentinels():
    """W2 `_brain_env` 直读：注入的凭据哨兵一个都不出现在返回值里。"""
    saved = {}
    try:
        for k, v in _SENTINELS.items():
            saved[k] = os.environ.get(k)
            os.environ[k] = v
        env = _brain_env("/iso/root", "/iso/brain-pkg")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    leaked = _credential_leaks(env)
    assert leaked == [], "子进程 env 仍带凭据：%r" % (leaked,)
    for name, value in _SENTINELS.items():
        assert name not in env, "%s 仍透传（值=%r）" % (name, env.get(name))
        assert value not in env.values(), "%s 的哨兵值仍透传" % name


def test_w3_real_parent_env_has_no_credential_shaped_leak():
    """W3 真实父环境动态扫描：父里任何凭据形状名都不得进子进程 env。"""
    env = _brain_env("/iso/root", "/iso/brain-pkg")
    parent_cred = [k for k in os.environ if _looks_like_credential(k)]
    leaked = sorted(k for k in parent_cred if k in env)
    assert leaked == [], "父进程凭据原样透传：%r" % (leaked,)
    # 反向腿：父里非凭据的随机变量也不该透传（白名单 = 不声明就不给）
    assert "MY_RANDOM_VAR_227_DOES_NOT_EXIST" not in env


def test_w4_required_items_survive():
    """W4 反向断言：必需项仍在——别把功能修死。"""
    env = _brain_env("/iso/root", "/iso/brain-pkg",
                     extra_env={"MDCG_LEGACY_ENV_AUTH": "1"})
    assert env["PYTHONPATH"] == "/iso/brain-pkg"
    assert env["MDCG_ROOT"] == "/iso/root"
    assert env["MDCG_STG_STATE"] == "1"
    assert env["PYTHONUTF8"] == "1"
    assert env["PYTHONIOENCODING"] == "utf-8"
    assert env.get("PATH"), "PATH 必须放行（子进程运行基座）"
    assert env["MDCG_LEGACY_ENV_AUTH"] == "1"          # extra_env 显式注入
    # 前缀面：`PYTHON*` 仍透传；`HIVE_` 前缀已移除（#227 续）——改由具名项承担
    assert _env_is_allowed("PYTHONANY") is True
    assert _env_is_allowed("HIVE_ANY") is False, "HIVE_ 前缀不得再放行"
    # 运行基座具名项确实在白名单里（逐项核对，防止误删）
    for name in ("PATH", "PATHEXT", "SYSTEMROOT", "TEMP", "HOME", "LANG"):
        assert name in _ENV_ALLOW_NAMES, name
    # #227 续：脑侧真读的两个非凭据 HIVE_* 具名在册（反向腿，防把功能修死）
    for name in _HIVE_CONFIG_NAMES:
        assert name in _ENV_ALLOW_NAMES, name
    assert "PYTHON" in _ENV_ALLOW_PREFIXES and "HIVE_" not in _ENV_ALLOW_PREFIXES
    assert not any(p.startswith("MDCG") for p in _ENV_ALLOW_PREFIXES), \
        "MDCG_ 不得进前缀白名单（继承的 MDCG_* 仍须剔除）"


def test_w5_inherited_mdcg_is_still_purged_and_extra_env_can_remove():
    """W5 继承 MDCG_* 仍剔除；extra_env 显式注入与 None 删除均有效。"""
    saved = os.environ.get("MDCG_BRAIN_PYTHONPATH")
    os.environ["MDCG_BRAIN_PYTHONPATH"] = "SENTINEL-INHERITED"
    try:
        env = _brain_env("/iso/root", "/iso/brain-pkg")
    finally:
        if saved is None:
            os.environ.pop("MDCG_BRAIN_PYTHONPATH", None)
        else:
            os.environ["MDCG_BRAIN_PYTHONPATH"] = saved
    assert "MDCG_BRAIN_PYTHONPATH" not in env, "继承的 MDCG_* 必须剔除"
    env2 = _brain_env("/iso/root", "/iso/brain-pkg",
                      extra_env={"MDCG_TENANT": "t1", "MDCG_ACTOR": None})
    assert env2["MDCG_TENANT"] == "t1"
    assert "MDCG_ACTOR" not in env2


_ENV_DUMP_SERVER = '''import json, os, sys
def send(obj):
    print(json.dumps(obj), flush=True)
for line in sys.stdin:
    msg = json.loads(line)
    if msg.get("method") == "initialize":
        send({"jsonrpc": "2.0", "id": msg["id"],
              "result": {"serverInfo": {"name": "env-dump-fixture", "version": "0.1"}}})
    elif msg.get("method") == "tools/call":
        send({"jsonrpc": "2.0", "id": msg["id"], "result": {
            "content": [{"text": json.dumps({"env": dict(os.environ)})}]}})
'''


def test_w6_spawned_child_env_carries_no_credentials():
    """W6 端到端：MCPClient 真起子进程，子进程报回 os.environ ⇒ 无凭据。"""
    from lingshu.world.brain_store import MCPClient
    tmp = tempfile.mkdtemp(prefix="i227_env_")
    script = os.path.join(tmp, "env_dump.py")
    with open(script, "w", encoding="utf-8") as f:
        f.write(_ENV_DUMP_SERVER)
    saved = {}
    try:
        for k, v in _SENTINELS.items():
            saved[k] = os.environ.get(k)
            os.environ[k] = v
        client = MCPClient(python=sys.executable, pythonpath=tmp, root=tmp,
                           args=[script], request_timeout=30.0)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    try:
        child_env = client.call("dump_env", {})["env"]
    finally:
        client.close()
    leaked = _credential_leaks(child_env)
    assert leaked == [], "真子进程 env 仍带凭据：%r" % (leaked,)
    for name in _SENTINELS:
        assert name not in child_env, "%s 出现在真子进程 env" % name
    # 反向腿：真子进程仍能正常启动并读到运行基座
    assert child_env.get("PATH")
    assert child_env.get("MDCG_ROOT") == tmp
    assert child_env.get("MDCG_STG_STATE") == "1"


def test_w7_predicate_has_teeth():
    """W7 判据有牙自证：旧法（只剔 MDCG_*）构造的 env 必被点名。"""
    legacy = {k: v for k, v in os.environ.items() if not k.startswith("MDCG_")}
    legacy["AEIS_DESIGNER_KEY"] = "SENTINEL-DESIGNER"
    legacy["MY_SERVICE_TOKEN"] = "SENTINEL-SERVICE"
    leaked = _credential_leaks(legacy)
    assert "AEIS_DESIGNER_KEY" in leaked, "判据对 AEIS_DESIGNER_KEY 无感（无牙）"
    assert "MY_SERVICE_TOKEN" in leaked, "判据对 *_TOKEN 无感（无牙）"
    # 且修复后的白名单 env 判据为空（同一判据两侧都给对读数）
    assert _credential_leaks(_brain_env("/iso/root", "/iso/brain-pkg")) == []


def test_w8_hive_prefix_converged_named_only():
    """W8 · #227 续·`HIVE_` 前缀收敛（另立一条并写明理由）。

    为什么另立：W1/W4 原有两处断言（`_env_is_allowed("HIVE_ROLE") is True`、
    `"HIVE_" in _ENV_ALLOW_PREFIXES`）**编码的正是本轮要收敛的过宽策略**，与修复
    目标相反；按「不改既有断言换绿灯」的口径，此处不改判据语义去凑绿，而是新增本条
    具名核对并把理由写在这里（W1/W4 对应两行只做同步收紧＋就地标注）。

    判定证据（脑包 `dsh-memory-main`）：brain_store 起的是
    `python -X utf8 -m md_cg.mcp_server`（`brain_store.py` 的 `MCPClient`）。该进程
    经 `cg(op=ccg, action=units)`（`md_cg/mcp_server.py` 内 `from . import units`）
    读 `md_cg/units.py` 的 `ENV_JOBS_DIR="HIVE_JOBS_DIR"` / `ENV_EXE="HIVE_EXE"`
    （`HIVE_API_KEY` 仅 `doctor()` 的 `api_key_set` 报告字段）。`hive/serve_start.py`
    的 `HIVE_CONFIG/HIVE_CARGO/...` 与 `hive/exec.py`、`hive/orch.py` 的
    `HIVE_ORCH_TOKEN(_FILE)`/`HIVE_SUBAGENT_API_KEY` 属蜂巢服务/执行器侧，**不在**
    该子进程读取面。⇒ 移除前缀，仅具名放行两个非凭据配置项。
    """
    # 腿 1：四个凭据形状名不进「白名单判据」
    for name in _HIVE_CRED_NAMES:
        assert _env_is_allowed(name) is False, "%s 不该被放行" % name
    assert "HIVE_" not in _ENV_ALLOW_PREFIXES, "HIVE_ 前缀不得保留"
    # 腿 2：注入假哨兵后 `_brain_env` 直读里四名一个都不出现
    saved = {}
    try:
        for name in _HIVE_CRED_NAMES:
            saved[name] = os.environ.get(name)
            os.environ[name] = _SENTINELS[name]
        env = _brain_env("/iso/root", "/iso/brain-pkg")
    finally:
        for name, v in saved.items():
            if v is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = v
    for name in _HIVE_CRED_NAMES:
        assert name not in env, "%s 仍透传子进程 env（值=%r）" % (name, env.get(name))
        assert _SENTINELS[name] not in env.values(), "%s 的哨兵值仍透传" % name
    # 腿 3（反向腿·防修死）：脑侧真读的两个具名非凭据项仍透传
    saved_cfg = {}
    try:
        for name in _HIVE_CONFIG_NAMES:
            saved_cfg[name] = os.environ.get(name)
            os.environ[name] = "/iso/" + name.lower()
        env2 = _brain_env("/iso/root", "/iso/brain-pkg")
    finally:
        for name, v in saved_cfg.items():
            if v is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = v
    for name in _HIVE_CONFIG_NAMES:
        assert env2.get(name) == "/iso/" + name.lower(), \
            "%s 应仍透传（脑侧 units 真读，别把功能修死）" % name


_PASS: list = []
_FAIL: list = []


def main() -> int:
    tests = [
        test_w1_env_is_allowed_shape_table,
        test_w2_brain_env_drops_credential_sentinels,
        test_w3_real_parent_env_has_no_credential_shaped_leak,
        test_w4_required_items_survive,
        test_w5_inherited_mdcg_is_still_purged_and_extra_env_can_remove,
        test_w6_spawned_child_env_carries_no_credentials,
        test_w7_predicate_has_teeth,
        test_w8_hive_prefix_converged_named_only,
    ]
    for t in tests:
        try:
            t()
            _PASS.append(t.__name__)
            print("  PASS " + t.__name__)
        except AssertionError as exc:
            _FAIL.append((t.__name__, str(exc)))
            print("  FAIL %s ← %s" % (t.__name__, exc))
        except Exception as exc:  # noqa: BLE001
            _FAIL.append((t.__name__, repr(exc)))
            print("  ERROR %s ← %r" % (t.__name__, exc))
    print()
    print(f"===== SUMMARY {len(_PASS)}/{len(_PASS) + len(_FAIL)} 通过 =====")
    if _FAIL:
        print("FAILS:", _FAIL)
        return 1
    print("VERDICT=PASS（issue #227：凭据不再随继承环境进入脑侧子进程；"
          "白名单放行 + 反向断言放行项仍在 + extra_env 显式口 + 真子进程端到端 + 判据有牙）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
