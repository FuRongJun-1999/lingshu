# Copyright 2026 灵枢 (Lingshu) · MIT
"""brain_store — 脑版 store/engine 适配器：语义时空图 ↔ 灵枢认知图（MCP 通路）。

分位与四条裁定（`dsh-memory` 仓 `docs/plans/灵枢身体×脑_世界模型对接设计_v0.1.md`
§四；使用者 2026-10-06 批准，按各条推荐执行）：

1. **适配器归属＝身体侧**（本件）——章程「脑为包依赖、身侧件进身仓」；
2. **3D 坐标＝`spatial.coords3d` 自定义键直存**（米制；可选 bbox 投影后补，
   不改既有 `spatial.bbox` 的 2D 语义）；
3. **tag 过滤＝适配器侧读后过滤**（零脑改；探针验证后再议是否进脑）；
4. 组合冒烟已进对端发布门禁（12 腿之一），本件由身侧测试复跑。

两条最小闭环（与 `scene_model` **零改动**对接）：

- **M1 读向（身体重建世界）**：`BrainStore.get_nodes_by_tag` →
  `cg(op=read)` 取候选 + 按 tags 过滤 + 坐标取 `frontmatter.spatial.coords3d`
  + 状态取**槽位投影**（`stg(op=state_chain)`，slot=「状态」）→
  `scene_model.load_world_from_memory(brain_store)` 直接可用。
- **M2 写向（身体观测入脑）**：`BrainEngine.add_perception` → `mdcg_remember`
  （tags + `spatial.coords3d` 直存）；状态写入经 `store.conn` 垫片翻译为
  `cg(op=state_event)` **记账**（事件是源、槽位是投影）→
  `scene_model.ingest_scene(brain_agent, desc)` 直接可用（含状态）。

传输：**MCP stdio**（逐行 JSON-RPC，与「身体件经 MCP 连脑的真实通路」一致）。
脑包位置与库根一律经参数/环境注入——**本仓不写本机路径字面量**（章程六）。

连接安全（缺省 fail-closed）：
- 子进程 env **白名单放行**（issue #227）：只透传进程运行基座与部署方显式配置面
  （见 `_ENV_ALLOW_NAMES` / `_ENV_ALLOW_PREFIXES`），其余继承变量一律**不透传**——
  `AEIS_DESIGNER_KEY` / `*_API_KEY` / `*_TOKEN` / `*_SECRET` / `*_PASSWORD` 等
  凭据**不再随继承环境交给脑侧子进程**（旧法「只剔 `MDCG_*`」是黑名单，对凭据面
  fail-open）。继承的 `MDCG_*` 仍被剔除（防在役库被无意命中）；凭据若确需传入，
  必须经 `extra_env` **显式**声明（有意为之、可审计）。再注入下面的显式覆盖；
- `pythonpath` 必须显式给出（参数或 `BRAIN_PYTHONPATH`），否则 `ValueError`；
- `root` 未给出且 `BRAIN_ROOT` 为空时，脑侧将回落到其缺省根——**调用方务必
  显式给隔离/目标根**（测试与生产都不例外）；
- `MDCG_STG_STATE=1` 由本层缺省注入（M1 状态读依赖 state_chain 开关）。

用法（隔离示例；测试件见 `tests/test_brain_store.py`）：

    from lingshu.world.brain_store import connect
    from lingshu.world.scene_model import ingest_scene, load_world_from_memory

    agent = connect(root=<隔离根>, pythonpath=<脑包目录>, extra_env={...身份...})
    ingest_scene(agent, "肥鱼|fatfish|0,0.85,5|shy\\n桌子|table|1.5,0.45,6|neutral")
    wm = load_world_from_memory(agent.store)
    print(wm.scene_text())

诚实边界（MVP）：
- 读候选受 `cg(op=read)` 召回面与 `limit` 约束（非全库枚举；`limit` 译为脑端
  条数参数 `k`——issue #2）——场景规模大时由 `seed_query`（缺省「场景实体」
  ＝ingest_scene 的内容约定词）保证召回头部；
- `record_state` 的 `old` 恒为 None（legacy `ingest_scene` 只传新值；变迁史由
  事件序承担，撤回语义走脑侧既有 projection）；
- 仅翻译 `UPDATE nodes SET state_attributes` 一条 legacy 语句形态，其它 SQL 抛错
  （不静默）。
"""
from __future__ import annotations

import json
import math
import os
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

#: M1/M2 状态槽位约定（与 ingest_scene 的 state → 呈现语义对应）
STATE_SLOT = "状态"
#: 读候选种子词（ingest_scene 写入的正文定式首词）
SEED_QUERY = "场景实体"

#: 子进程 env **白名单·具名**（issue #227 · 凭据泄露族）——跨平台「进程运行基座」：
#: 路径 / 临时目录 / 区域 / 主机自省。**不含任何凭据名**。
_ENV_ALLOW_NAMES = frozenset({
    # 可执行查找 / 命令解释器
    "PATH", "PATHEXT", "COMSPEC",
    # Windows 目录基座（Python 解释器与 stdlib 启动期读用）
    "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR",
    "PROGRAMDATA", "ALLUSERSPROFILE", "PUBLIC",
    "PROGRAMFILES", "PROGRAMFILES(X86)", "COMMONPROGRAMFILES",
    "COMMONPROGRAMFILES(X86)", "PROGRAMW6432", "COMMONPROGRAMW6432",
    # 临时目录
    "TEMP", "TMP", "TMPDIR",
    # 家目录 / 账户
    "HOME", "HOMEDRIVE", "HOMEPATH", "USERPROFILE",
    "APPDATA", "LOCALAPPDATA",
    "USER", "USERNAME", "USERDOMAIN", "LOGNAME",
    # 区域 / 终端
    "LANG", "LANGUAGE", "TZ", "TERM",
    # 宿主自省（解释器/编译扩展常用）
    "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "PROCESSOR_IDENTIFIER",
    "PROCESSOR_LEVEL", "PROCESSOR_REVISION", "OS",
})

#: 子进程 env **白名单·前缀**（部署方显式配置面）。`PYTHON*` 覆盖 PYTHONPATH /
#: PYTHONUTF8 / PYTHONHOME / PYTHONIOENCODING / PYTHONDONTWRITEBYTECODE 等。
#: 刻意**不含 `MDCG_`**——继承的 `MDCG_*` 仍须剔除（防在役库被无意命中），
#: `MDCG_ROOT` / `MDCG_STG_STATE` 由 `_brain_env` 显式注入。
_ENV_ALLOW_PREFIXES = ("PYTHON", "HIVE_", "LINGSHU_", "LC_")


def _env_is_allowed(name: str) -> bool:
    """继承的环境变量名是否在白名单内（大小写无关：Windows 上 `os.environ` 键为大写）。"""
    up = str(name).upper()
    if up in _ENV_ALLOW_NAMES:
        return True
    return any(up.startswith(p) for p in _ENV_ALLOW_PREFIXES)


class BrainError(RuntimeError):
    """脑侧 MCP 传输、超时或工具结果错误时抛出（不静默）。"""


# ---------------------------------------------------------------------------
# 最小 MCP stdio 客户端（stdlib only）
# ---------------------------------------------------------------------------


def _brain_env(root: Optional[str], pythonpath: Optional[str],
               extra_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """构造脑子进程 env（隔离/连接安全单点）。

    **白名单放行**（issue #227）：只继承 `_ENV_ALLOW_NAMES` / `_ENV_ALLOW_PREFIXES`
    命中的变量；其余（含全部凭据与继承 `MDCG_*`）一律不透传 ⇒ 再注入显式覆盖。
    子进程由模型驱动，凭据一旦透传即落在其可读面内（`os.environ`），故本层
    fail-closed：需要什么，谁需要谁**显式**给（`extra_env`），不靠继承顺带。
    """
    env = {k: v for k, v in os.environ.items() if _env_is_allowed(k)}
    if root:
        env["MDCG_ROOT"] = root
    if pythonpath:
        env["PYTHONPATH"] = pythonpath
    env["MDCG_STG_STATE"] = "1"          # M1 状态读依赖 state_chain（新面默认关）
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    for k, v in (extra_env or {}).items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = str(v)
    return env


class MCPClient:
    """脑侧 MCP stdio 客户端（单飞请求，逐行 JSON-RPC）。

    stderr 持续消费，仅缓存最近 4096 字节供错误诊断。`request_timeout` 是每次
    initialize / tools/call 写入及等待响应的秒数，缺省 30；None 显式关闭超时。
    传输失败或超时会关闭连接，调用方须重新连接，避免迟到响应串入后续调用。
    """

    def __init__(self, python: Optional[str] = None, pythonpath: Optional[str] = None,
                 root: Optional[str] = None, extra_env: Optional[Dict[str, str]] = None,
                 args: Optional[List[str]] = None,
                 request_timeout: Optional[float] = 30.0):
        if request_timeout is not None and (
                not math.isfinite(request_timeout) or request_timeout <= 0):
            raise ValueError("request_timeout 必须是正的有限秒数或 None")
        python = python or os.environ.get("BRAIN_PYTHON") or sys.executable
        pythonpath = pythonpath or os.environ.get("BRAIN_PYTHONPATH")
        root = root or os.environ.get("BRAIN_ROOT")
        if not pythonpath:
            raise ValueError(
                "必须显式提供脑包目录（pythonpath 参数或 BRAIN_PYTHONPATH）"
                "——本仓不写本机路径字面量（章程六）")
        env = _brain_env(root, pythonpath, extra_env)
        self.root = root
        self._id = 0
        self._request_timeout = request_timeout
        self._request_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._closed = False
        self._rpc_worker = None
        self._stderr_lock = threading.Lock()
        self._stderr_tail = b""
        self.server_info: Dict = {}
        self._p = subprocess.Popen(
            [python, "-X", "utf8", *(args or ["-m", "md_cg.mcp_server"])],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, text=True, encoding="utf-8", errors="replace")
        self._stderr_worker = threading.Thread(target=self._drain_stderr, daemon=True)
        try:
            self._stderr_worker.start()  # 必须早于握手；启动日志也会填满管道。
            init = self._exchange({"jsonrpc": "2.0", "method": "initialize",
                                   "params": {"protocolVersion": "2024-11-05",
                                              "capabilities": {},
                                              "clientInfo": {"name": "lingshu-brain-store",
                                                             "version": "0.1"}}})
            self.server_info = (init.get("result") or {}).get("serverInfo") or {}
            if not self.server_info:
                raise BrainError("脑侧握手无 serverInfo：%r" % (init,))
            self._rpc({"jsonrpc": "2.0", "method": "notifications/initialized"})
        except BaseException:
            self.close()
            raise

    # ---- 低层 ----
    def _next(self) -> int:
        self._id += 1
        return self._id

    def _rpc(self, obj: dict) -> None:
        self._p.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._p.stdin.flush()

    def _drain_stderr(self) -> None:
        # read1 不等换行或填满整块；原始字节也不会因非法 UTF-8 中止消费。
        try:
            while True:
                chunk = self._p.stderr.buffer.read1(4096)
                if not chunk:
                    return
                with self._stderr_lock:
                    self._stderr_tail = (self._stderr_tail + chunk)[-4096:]
        except (OSError, ValueError):  # 管道关闭；诊断保留已读尾部。
            return

    def _diagnostic(self) -> str:
        with self._stderr_lock:
            return self._stderr_tail.decode("utf-8", errors="replace")[-400:]

    def _read(self) -> dict:
        while True:
            line = self._p.stdout.readline()
            if not line:
                # stdout EOF 与最后一块 stderr 的到达没有先后保证；短等尾部，
                # 但不对仍存活、只关闭 stdout 的子进程调用阻塞式 stderr.read。
                self._stderr_worker.join(timeout=0.1)
                raise BrainError("脑进程关闭：" + self._diagnostic())
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if isinstance(msg, dict) and ("result" in msg or "error" in msg):
                return msg

    def _exchange(self, obj: dict) -> dict:
        with self._request_lock:
            if self._closed:
                raise BrainError("脑侧 MCP 连接已关闭；请重新连接")
            obj["id"] = self._next()
            outcome = queue.Queue(maxsize=1)

            def run():
                try:
                    self._rpc(obj)
                    outcome.put((True, self._read()))
                except Exception as exc:
                    outcome.put((False, exc))

            worker = threading.Thread(target=run, daemon=True)
            self._rpc_worker = worker
            worker.start()
            try:
                success, value = outcome.get(timeout=self._request_timeout)
            except queue.Empty as exc:
                # 写入也在 worker 中：子进程不读 stdin 时同样受期限约束。
                self.close()
                raise BrainError("脑侧 MCP %s 超时（%s 秒）：%s"
                                 % (obj["method"], self._request_timeout,
                                    self._diagnostic())) from exc
            except BaseException:
                self.close()
                raise
            worker.join()
            self._rpc_worker = None
            if not success:
                if isinstance(value, (BrainError, OSError)):
                    self.close()
                    raise BrainError("脑侧 MCP 传输失败：" + str(value)) from value
                raise value
            return value

    # ---- 调用面 ----
    def call(self, name: str, arguments: dict) -> dict:
        """tools/call → 解析 content[0].text（JSON）；错误面抛 BrainError。"""
        msg = self._exchange({"jsonrpc": "2.0", "method": "tools/call",
                              "params": {"name": name, "arguments": arguments}})
        if "error" in msg and msg["error"]:
            raise BrainError("MCP error：%r" % (msg["error"],))
        res = msg.get("result") or {}
        text = ((res.get("content") or [{}])[0]).get("text", "")
        if res.get("isError"):
            raise BrainError("工具 %s 返回 isError：%s" % (name, text[:400]))
        try:
            return json.loads(text)
        except ValueError as e:
            raise BrainError("工具 %s 返回非 JSON：%s" % (name, text[:400])) from e

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            # 活跃 writer 可持有 stdin 的 I/O 锁；先停进程，不能先 close(stdin)。
            if self._request_lock.locked() and self._p.poll() is None:
                self._p.terminate()
            else:
                try:
                    self._p.stdin.close()
                except (OSError, ValueError):
                    pass
            try:
                self._p.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._p.kill()
                self._p.wait(timeout=2)
            for worker in (self._rpc_worker, self._stderr_worker):
                if worker is not None and worker.ident is not None:
                    worker.join(timeout=1)
            # 后代若继承管道，reader 仍可能等待；不能 close 它持锁的 stream。
            if self._rpc_worker is None or not self._rpc_worker.is_alive():
                for stream in (self._p.stdin, self._p.stdout):
                    try:
                        stream.close()
                    except (OSError, ValueError):
                        pass
            if not self._stderr_worker.is_alive():
                self._p.stderr.close()


# ---------------------------------------------------------------------------
# store / engine 协议实现（scene_model 消费面）
# ---------------------------------------------------------------------------


@dataclass
class BrainNode:
    """store.get_nodes_by_tag 的返回单元——字段名与身体侧 store 协议一致。"""
    id: str
    content: str
    tags: List[str] = field(default_factory=list)
    spatial_coordinates: Optional[Dict[str, float]] = None   # {"x","y","z"}（米）
    state_attributes: Optional[Dict[str, str]] = None        # {"state": ...}


@dataclass
class NodeRef:
    """engine.add_perception 的返回单元（legacy 代码访问 `.id`）。"""
    id: str


class _ConnShim:
    """legacy `ingest_scene` 直写语句 → 脑侧状态事件 的翻译垫片（单点）。

    契约：只认 `UPDATE nodes SET state_attributes=? WHERE id=?` 一条形态——
    payload 为 `{"state": ...}` JSON、第二参数为节点 id；翻译为
    `cg(op=state_event)`（subject=该节点的实体名、slot=「状态」、kind=acquisition）。
    其它 SQL 抛 NotImplementedError（不静默吞）；状态事件未入账时抛
    `BrainError`（同样不静默吞）。legacy 侧 `commit()` 为无操作。
    """

    UPDATE_PREFIX = "UPDATE NODES SET STATE_ATTRIBUTES"

    def __init__(self, store: "BrainStore"):
        self.store = store

    def execute(self, sql, params=None):
        if not isinstance(sql, str) or not sql.strip().upper().startswith(self.UPDATE_PREFIX):
            raise NotImplementedError("brain conn 垫片只翻译状态写语句；收到：%r"
                                      % (str(sql)[:80],))
        payload = json.loads(params[0]) if params and params[0] else {}
        node_id = str(params[1])
        state = (payload or {}).get("state")
        ok = self.store.record_state(node_id, state)
        if state and not ok:
            raise BrainError(
                "状态事件未入账（record_state→False）：node=%s state=%r；"
                "节点正文已写入、台账未落，重建世界会与节点内容不一致"
                % (node_id, state))
        return self

    def commit(self):        # noqa: D102 —— 事件已即时落账，无事务
        pass


class BrainStore:
    """脑版 store：`get_nodes_by_tag`（M1 读）+ 状态事件写（M2 经垫片）。"""

    def __init__(self, client: MCPClient, scene_tag: str = "spatial",
                 seed_query: str = SEED_QUERY):
        self.client = client
        self.scene_tag = scene_tag
        self.seed_query = seed_query
        self.conn = _ConnShim(self)          # legacy ingest_scene 的直写出口
        self._node_meta: Dict[str, Dict[str, str]] = {}   # node_id → {"ent": 名}

    # ---- M1 读向 ----
    def _state_map(self) -> Dict[str, str]:
        """活动状态槽位投影：subject → value（slot=「状态」；一次查询全量取回）。"""
        resp = self.client.call("stg", {"op": "state_chain", "limit": 500})
        m: Dict[str, str] = {}
        for u in (resp.get("items") or []):
            if u.get("slot") == STATE_SLOT and u.get("state") == "active":
                m[str(u.get("subject"))] = u.get("value")
        return m

    #: 允许进入世界重建的脑端资格态（cg(op=read) 返回体 item["state"]，四态之一）。
    #: 默认只踢 REJECT：场景实体节点本身缺 CCG 六要素 ⇒ 实测恒为 BLINDSPOT，
    #: 若只收 ACCEPT 世界重建会清零；世界是消费端，取保守纪律——只拒明确否决者
    #: （REJECT），BLINDSPOT（证据不足）/DEFER（待定）不拒。
    #: 调用方可显式收紧（accept_states={"ACCEPT"}）或传 set() 关闭过滤（旧行为）。
    DEFAULT_ACCEPT_STATES = frozenset({"ACCEPT", "BLINDSPOT", "DEFER"})

    def get_nodes_by_tag(self, tag: str, limit: int = 200,
                         accept_states: Optional[set] = None) -> List[BrainNode]:
        """cg(op=read) 取候选 → **适配器侧按 tag 过滤**（裁定三：零脑改）→ BrainNode。

        坐标取 `frontmatter.spatial.coords3d`（裁定二）；状态取槽位投影。

        `limit` 译为脑端条数参数 `k`（issue #2）：脑端 `cg(op=read)` 的条数口径
        是 `k`（`_int_arg(a, "k", 20)`）；`limit` 在非 `budget_tokens` 路径**不被
        识别**、静默回落脑端缺省 20——写入 21 个场景实体会只重建 20 个。

        `accept_states`：资格态白名单（缺省 `DEFAULT_ACCEPT_STATES`＝只踢
        REJECT；传 `set()` 关闭过滤保留旧行为）。REJECT 来自脑端资格判据
        `judge_qualification`——不适用条件命中情境（query/context）即明确否决，
        不应作为可信几何注入世界重建；state 缺失（如脑端换判据面）不剔除
        （未知≠否决，fail-open，防静默清空）。
        """
        if accept_states is None:
            accept_states = self.DEFAULT_ACCEPT_STATES
        resp = self.client.call("cg", {"op": "read", "query": self.seed_query,
                                       "k": max(1, int(limit))})
        states = self._state_map()
        out: List[BrainNode] = []
        for item in (resp.get("results") or resp.get("items") or []):
            node = item.get("node") or item
            fm = node.get("frontmatter") or item.get("frontmatter") or {}
            tags = list(fm.get("tags") or [])
            if tag not in tags:
                continue                      # 裁定三：过滤在适配器侧
            if accept_states:
                st = item.get("state")
                if st and st not in accept_states:
                    continue                  # 明确否决态不注入世界（缺省踢 REJECT）
            sp = ((fm.get("spatial") or {}).get("coords3d") or {})
            coords = None
            if isinstance(sp, dict) and sp:
                coords = {"x": float(sp.get("x", 0.0)),
                          "y": float(sp.get("y", 0.0)),
                          "z": float(sp.get("z", 0.0))}
            ent = ""
            for t in tags:
                if str(t).startswith("ent:"):
                    ent = str(t)[4:]
                    break
            state = states.get(ent)
            out.append(BrainNode(
                id=str(node.get("id") or fm.get("id") or ""),
                content=str(node.get("content") or ""),
                tags=tags,
                spatial_coordinates=coords,
                state_attributes=({"state": state} if state else None)))
        return out

    # ---- M2 写向（垫片目标）----
    def record_state(self, node_id: str, state: Optional[str]) -> bool:
        """把 legacy 的状态写翻译成 `cg(op=state_event)` 记账（subject=实体名）。"""
        if not state:
            return False
        ent = (self._node_meta.get(node_id) or {}).get("ent") or node_id
        r = self.client.call("cg", {"op": "state_event", "subject": ent,
                                    "slot": STATE_SLOT, "old": None,
                                    "new": str(state), "kind": "acquisition",
                                    "evidence": "ingest_scene→brain(conn 垫片)"})
        return bool(r.get("ok"))


class BrainEngine:
    """脑版 engine：`add_perception` → `mdcg_remember`（含 spatial.coords3d 直存）。"""

    def __init__(self, store: BrainStore):
        self.store = store

    def add_perception(self, content: str, importance: float = 0.6,
                       spatial_coordinates: Optional[Dict[str, float]] = None,
                       tags: Optional[List[str]] = None,
                       entities: Optional[List[str]] = None, **_kw) -> NodeRef:
        args = {"content": content, "layer": "contextual", "gated": False,
                "importance": float(importance), "tags": list(tags or [])}
        if spatial_coordinates:
            args["spatial"] = {"coords3d": {k: float(v)
                                            for k, v in spatial_coordinates.items()}}
        r = self.store.client.call("mdcg_remember", args)
        if not r.get("ok") or not r.get("id"):
            raise BrainError("mdcg_remember 未成功：%r" % (r,))
        nid = str(r["id"])
        ent = (entities or [None])[0]
        if ent:
            self.store._node_meta[nid] = {"ent": str(ent)}
        return NodeRef(id=nid)


class BrainAgent:
    """agent 形（scene_model 同时取 `.store` 与 `.engine`）。"""

    def __init__(self, store: BrainStore):
        self.store = store
        self.engine = BrainEngine(store)


def connect(python: Optional[str] = None, pythonpath: Optional[str] = None,
            root: Optional[str] = None,
            extra_env: Optional[Dict[str, str]] = None,
            request_timeout: Optional[float] = 30.0) -> BrainAgent:
    """建立到脑的 MCP 连接并返回 BrainAgent。

    每次请求缺省限时 30 秒；慢工具可传更长 request_timeout 或 None。
    调用方负责 close：`agent.store.client.close()`；超时后须重新 connect。
    """
    client = MCPClient(python=python, pythonpath=pythonpath, root=root,
                       extra_env=extra_env, request_timeout=request_timeout)
    return BrainAgent(BrainStore(client))


# ---------------------------------------------------------------------------
# 便捷函数（惰性导入 scene_model，保持本模块核心零重依赖）
# ---------------------------------------------------------------------------


def load_world_from_brain(agent_or_store, tag: str = "spatial"):
    """M1 便捷：脑为底重建世界模型（scene_model.load_world_from_memory 零改动）。"""
    from .scene_model import load_world_from_memory
    return load_world_from_memory(agent_or_store, tag=tag)


def ingest_scene_to_brain(agent: BrainAgent, scene_desc: str, tag: str = "spatial"):
    """M2 便捷：场景描述入脑（scene_model.ingest_scene 零改动；状态经 conn 垫片入账）。"""
    from .scene_model import ingest_scene
    return ingest_scene(agent, scene_desc, tag=tag)
