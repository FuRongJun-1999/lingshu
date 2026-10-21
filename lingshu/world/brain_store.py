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
  （tags + `spatial.coords3d` 直存；密级 `sensitivity` / 来源 `role` /
  `condition_space` 显式透传，未具名的关键字一律 `TypeError` 不静默吞——
  issue #364）；状态写入经 `store.conn` 垫片翻译为
  `cg(op=state_event)` **记账**（事件是源、槽位是投影；**subject＝节点 id**，
  非实体名——实体名跨场景不唯一，issue #366）→
  `scene_model.ingest_scene(brain_agent, desc)` 直接可用（含状态）。

传输：**MCP stdio**（逐行 JSON-RPC，与「身体件经 MCP 连脑的真实通路」一致）。
脑包位置与库根一律经参数/环境注入——**本仓不写本机路径字面量**（章程六）。

连接安全（缺省 fail-closed）：
- 子进程 env **白名单放行**（issue #227）：只透传进程运行基座与部署方显式配置面
  （见 `_ENV_ALLOW_NAMES` / `_ENV_ALLOW_PREFIXES`），其余继承变量一律**不透传**——
  `AEIS_DESIGNER_KEY` / `*_API_KEY` / `*_TOKEN` / `*_SECRET` / `*_PASSWORD` 等
  凭据**不再随继承环境交给脑侧子进程**（旧法「只剔 `MDCG_*`」是黑名单，对凭据面
  fail-open）。**该断言现已无旁路**：原先仅余的 `LC_*` 宽前缀会顺带放行
  `LC_TOKEN` / `LC_PASSWORD` 这类凭据形状名（#495，复核 #680 §一.3），已按同一
  手法收敛为 13 个区域名具名（见 `_ENV_ALLOW_NAMES`）⇒ 前缀白名单为空。继承的 `MDCG_*` 仍被剔除（防在役库被无意命中）；凭据若确需传入，
  必须经 `extra_env` **显式**声明（有意为之、可审计）。**`HIVE_` 前缀不再放行**
  （#227 续）：脑侧子进程（`python -X utf8 -m md_cg.mcp_server`）只在
  `cg(op=ccg, action=units)` 工具面上经 `md_cg/units.py` 具名读 `HIVE_JOBS_DIR` /
  `HIVE_EXE`（二者已入 `_ENV_ALLOW_NAMES`），凭据形状的 `HIVE_API_KEY` /
  `HIVE_ORCH_TOKEN(_FILE)` / `HIVE_SUBAGENT_API_KEY` 一律不透传。**`PYTHON` /
  `LINGSHU_` 前缀亦不再放行**（#652③）：`PYTHONSTARTUP` / `PYTHONHOME` 等解释器
  启动劫持名、以及**身侧**逃生口 `LINGSHU_ALLOW_CWD_IMPORTS` 原被宽前缀连带放行；
  已按同一手法改具名（见 `_ENV_ALLOW_NAMES`）。再注入下面的显式覆盖；
- `pythonpath` 必须显式给出（参数或 `BRAIN_PYTHONPATH`），否则 `ValueError`；
- `root` 未给出且 `BRAIN_ROOT` 为空时，脑侧将回落到其缺省根——**调用方务必
  显式给隔离/目标根**（测试与生产都不例外）；
- `MDCG_STG_STATE=1` 由本层缺省注入（M1 状态读依赖 state_chain 开关）。

**父进程配置面（#652①，勿与上面的白名单混淆）**：`BRAIN_PYTHON` /
`BRAIN_PYTHONPATH` / `BRAIN_ROOT` 是在**本进程**里、**`_brain_env` 之前**被读的
（`MCPClient.__init__`）——它们决定「拉起哪个解释器 / 从哪个目录取模块 / 库根在哪」。
上面的 `_ENV_ALLOW_NAMES` / `_ENV_ALLOW_PREFIXES` **管不到它们**：白名单只决定
「往子进程 env 里放什么」。故「白名单拦住了 `BRAIN_*` 不进子进程」**不能**说明这
三个变量安全。处置：显式参数优先；env 兜底时 `BRAIN_PYTHON` / `BRAIN_PYTHONPATH`
做 fail-closed 校验（绝对路径＋存在性＋**解释器形状/脑包结构**＋**受信根**，见
`_require_env_interpreter` / `_require_env_dir`；#748 收窄）。**受信根默认必填**：
未声明 `LINGSHU_BRAIN_PYTHON_ROOTS` / `LINGSHU_BRAIN_PYTHONPATH_ROOTS` 时，
env 兜底面一律**拒**（一段继承/残留的 `BRAIN_PYTHONPATH` 不再静默决定被执行的
模块）——要用 env 面就先显式声明受信根。可达性（如实）：前提是已能设置本进程
环境变量 ⇒ 对「已能设置 env 的攻击者」而言仍属**提权/持久化放大器**，非默认可达
RCE；受信根收的是**默认配置**这一档，不是「env 面＝安全边界」。

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
import re
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

#: M1/M2 状态槽位约定（与 ingest_scene 的 state → 呈现语义对应）
STATE_SLOT = "状态"
#: 读候选种子词（ingest_scene 写入的正文定式首词）
SEED_QUERY = "场景实体"
#: 状态槽位投影的单页条数（`stg(op=state_chain)` 的 `limit`）。脑端返回体是
#: `items=units[:limit]` 且带 `truncated`（`md_cg/stg.py:state_chain`）——固定读
#: 一页且不看 `truncated` 时，排在投影序后面的主体会被静默读成「无值」
#: （issue #365）。超出本页时 `_state_map` 按主体补读。
_STATE_PAGE = 500
#: 脑端 `cg(op=read)` 的候选面全局上限（脑包 `md_cg/mdcg.py:94` 的 `GLOBAL_CAP
#: = 500`，`cut_by_relevance` 在此截断候选）——`k` 再大也不会多给候选。本层据此
#: 把「候选面已取尽」与「被脑端截断」区分开（issue #367）。
BRAIN_CANDIDATE_CAP = 500

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
    # 部署方显式配置面（**具名**，非前缀）——#227 续收敛：脑侧子进程
    # `python -X utf8 -m md_cg.mcp_server` 的 `cg(op=ccg, action=units)` 工具面
    # 经 `md_cg/units.py`（`ENV_JOBS_DIR` / `ENV_EXE`）读这两个名字定位蜂巢作业池
    # 与可执行文件；缺省会回落到**脑包仓内**路径（`units.jobs_dir` / `units.exe_path`），
    # 故非必需——重定向部署方可继承，或按需经 `extra_env` 显式声明。
    # **不设 `HIVE_` 前缀**：前缀会顺带放行 `HIVE_API_KEY` / `HIVE_ORCH_TOKEN` /
    # `HIVE_ORCH_TOKEN_FILE` / `HIVE_SUBAGENT_API_KEY` 等凭据形状名，而这些名字只在
    # hive 服务/执行器侧（`hive/serve_start.py` / `hive/exec.py` / `hive/orch.py`，
    # 由蜂巢自身进程树拉起）被读，**不在** `md_cg.mcp_server` 的读取面上。
    "HIVE_JOBS_DIR", "HIVE_EXE",
    # #652③：原 `PYTHON` / `LINGSHU_` **宽前缀**收敛为**具名**（前缀过宽会把
    # 解释器启动/行为劫持名与**身侧**逃生口一并放行给脑侧子进程）。具名清单＝
    # 「脑侧子进程真读的」＋「部署方继承同义配置的余地」两类：
    #   · CPython 配置/行为面：`PYTHONPATH` / `PYTHONUTF8` / `PYTHONIOENCODING`
    #     三者由 `_brain_env` 显式注入（此处保留是为部署方继承同义配置）；
    #     其余为纯行为开关（字节码缓存 / 缓冲 / 告警 / 故障处理器 / 哈希种子 /
    #     用户 site / 安全路径 / 开发模式）——**均非「启动期执行任意代码」面**；
    #   · 脑侧真读的 `LINGSHU_*` 只有 `LINGSHU_KB` / `LINGSHU_DB`
    #     （`md_cg/whitebox_kb/**` 白盒引擎的知识库/数据库路径）。
    # **不收**：`PYTHONSTARTUP` / `PYTHONHOME` / `PYTHONINSPECT` /
    # `PYTHONEXECUTABLE`（解释器启动期劫持面）与 `LINGSHU_ALLOW_CWD_IMPORTS`
    # （`lingshu/_pathguard.py` 的**身侧**逃生口，脑侧零读取点 ⇒ 透传纯属扩大面）。
    "PYTHONPATH", "PYTHONUTF8", "PYTHONIOENCODING",
    "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED", "PYTHONWARNINGS",
    "PYTHONFAULTHANDLER", "PYTHONHASHSEED", "PYTHONNOUSERSITE",
    "PYTHONSAFEPATH", "PYTHONDEVMODE",
    "LINGSHU_KB", "LINGSHU_DB",
    # #495（复核 #680 §一.3 残留）：原 `("LC_",)` 是**宽前缀**，顺带放行任意
    # `LC_*` 名——含凭据形状的 `LC_TOKEN` / `LC_PASSWORD`，即上文「`*_TOKEN` /
    # `*_PASSWORD` 不再透传」对 LC_ 族**不成立**。按 #227 续 / #652③ 同一手法
    # 第三次收敛为**具名**：POSIX + glibc 的 13 个区域名（`LC_` 前缀面随之清空）。
    # 非标准 `LC_*` 名本无区域语义，确需者走 `extra_env` 显式声明。
    "LC_ALL", "LC_COLLATE", "LC_CTYPE", "LC_MESSAGES", "LC_MONETARY",
    "LC_NUMERIC", "LC_TIME",
    "LC_PAPER", "LC_NAME", "LC_ADDRESS", "LC_TELEPHONE", "LC_MEASUREMENT",
    "LC_IDENTIFICATION",
})

#: 子进程 env **白名单·前缀**（部署方显式配置面）。刻意**不含 `MDCG_`**——继承的
#: `MDCG_*` 仍须剔除（防在役库被无意命中），`MDCG_ROOT` / `MDCG_STG_STATE` 由
#: `_brain_env` 显式注入。**不含 `HIVE_`**（#227 续）：前缀过宽会把凭据形状名整体
#: 放行；脑侧确需的两个非凭据名 `HIVE_JOBS_DIR` / `HIVE_EXE` 走上面的
#: `_ENV_ALLOW_NAMES` **具名**放行。**不含 `PYTHON` / `LINGSHU_`**（#652③ 续收敛）：
#: 同为「前缀过宽」——`PYTHON` 连带放行 `PYTHONSTARTUP` / `PYTHONHOME` 等解释器
#: 启动劫持名，`LINGSHU_` 连带放行身侧逃生口 `LINGSHU_ALLOW_CWD_IMPORTS`；两条
#: 前缀已按 #227 续的同一手法改**具名**（见上）。**`LC_` 亦已清空**（#495）：
#: 同一「前缀过宽」失效模式第三次命中——前缀面把 `LC_TOKEN` / `LC_PASSWORD`
#: 这类凭据形状名一并放行，故 13 个区域名改入 `_ENV_ALLOW_NAMES` 具名。
#: 结论：**前缀白名单为空**——「不声明就不给」现在是完整的（无前缀旁路）。
_ENV_ALLOW_PREFIXES = ()

#: 子进程 env **黑名单·拒传**（issue #750 残余面 ③ · **deny 优先于 allow**）：
#: 这两个名字决定子进程的**导入安全态 / 组件解析根**——`LINGSHU_ALLOW_CWD_
#: IMPORTS` 是 `lingshu/_pathguard.py` 的身侧逃生口（开了等于放弃 cwd 解析面
#: 保护），`LINGSHU_COMPONENT_ROOT` 是 `component_resolver` 的显式组件根
#: （改写白名单组件的解析面）。父进程一旦把它们透传给脑子子进程，等于**改写
#: 孙进程的保护态**（#750 原报告实测：`LINGSHU_` 宽前缀时代曾整体透传）。
#: #652③/#495 已把前缀白名单收敛为空、二者不再透传；本层再把「永不透传」
#: 钉成**结构化黑名单**——未来白名单若再放宽（重引入前缀 / 具名误收），
#: deny 仍先行拦截。脑侧子进程（`md_cg.mcp_server`）对二者零读取点；
#: 确需给子进程重定向组件根的部署走 `extra_env` **显式**声明。
_ENV_DENY_NAMES = frozenset({
    "LINGSHU_ALLOW_CWD_IMPORTS",
    "LINGSHU_COMPONENT_ROOT",
})


def _env_is_allowed(name: str) -> bool:
    """继承的环境变量名是否在白名单内（大小写无关：Windows 上 `os.environ` 键为大写）。

    #750 残余面 ③：`_ENV_DENY_NAMES` **优先**于白名单——黑名单名一律拒传
    （防白名单未来再放宽时连带复活「子进程保护态被父进程改写」面）。
    """
    up = str(name).upper()
    if up in _ENV_DENY_NAMES:
        return False
    if up in _ENV_ALLOW_NAMES:
        return True
    return any(up.startswith(p) for p in _ENV_ALLOW_PREFIXES)


#: **父进程配置面**（issue #652①）——这三个名字在 `MCPClient.__init__` 里、**在
#: `_brain_env` 之前**被读，用于决定「拉起哪个解释器 / 从哪个目录取模块 / 库根在
#: 哪」。它们**不受 `_ENV_ALLOW_NAMES` / `_ENV_ALLOW_PREFIXES` 约束**：白名单只
#: 决定「往子进程 env 里放什么」，管不到父进程读取的这三个变量。因此
#: 「白名单拦住了 `BRAIN_*` 不进子进程」**不能**说明这三个变量安全。
#: 可达性（如实）：前提是已能设置本进程环境变量；具备该能力者本已可影响本进程
#: 启动（如 `PYTHONPATH`），故本面属**提权/持久化放大器**，非默认可达 RCE。
#: 处置：`BRAIN_PYTHON` / `BRAIN_PYTHONPATH` 走**显式参数优先**，env 兜底时做
#: fail-closed 校验（见 `_require_env_interpreter` / `_require_env_dir`；**受信根
#: 默认必填**——未声明即拒）；`BRAIN_ROOT` 属数据根（脑侧可自建），只做声明不做
#: 存在性校验。
_ENV_PARENT_CONFIG = ("BRAIN_PYTHON", "BRAIN_PYTHONPATH", "BRAIN_ROOT")

#: 部署方声明的**受信根**面（`os.pathsep` 多根）。**默认必填**（#748）：env 兜底
#: 值未声明受信根即拒——否则「存在且含 md_cg 的目录」就够格，而造这样一个目录
#: 零成本（攻击者侧），报告里的 payload 原样仍成立。声明后，值的归一化路径必须
#: 落在其中某个根之下，否则 fail-closed。
_BRAIN_PYTHON_ROOTS_ENV = "LINGSHU_BRAIN_PYTHON_ROOTS"
_BRAIN_PYTHONPATH_ROOTS_ENV = "LINGSHU_BRAIN_PYTHONPATH_ROOTS"

#: 可接受的解释器**文件名形状**（`BRAIN_PYTHON` 兜底值）。
_PY_EXE_RE = re.compile(r"^python(\d+(\.\d+)?)?(\.exe)?$", re.IGNORECASE)


def _require_under_trusted_roots(value: str, roots_env: str, var: str,
                                 required: bool = False) -> str:
    """受信根校验（#748）。

    `required=True`（env 兜底面的默认处置）：`roots_env` **未声明 ⇒ 拒**——
    env 给出的执行面（解释器 / 模块搜索目录）在**默认配置下不被采信**，
    部署方要么走显式参数 `python=` / `pythonpath=`（推荐路径），要么显式声明
    受信根把 env 面打开。`required=False` 只做「声明了就必须落在根内」。

    声明了 `roots_env` 时：`value` 的归一化路径须落在其中某个根之下，否则
    fail-closed。归一化 = `os.path.normcase(os.path.realpath(...))`（两侧同口径）。

    残留面（如实，**本件不消除**）：能设置本进程 env 者同样能设置该受信根变量
    ⇒ 对「已能设置 env 的攻击者」而言本面仍属**提权/持久化放大器**（与 main
    既有定性一致），非默认可达 RCE。本件收的是**默认配置**这一档：一段从父环境
    继承/残留的 `BRAIN_PYTHONPATH` 不再静默决定被执行的模块。
    """
    declared = os.environ.get(roots_env)
    if not declared:
        if required:
            raise ValueError(
                "%s 来自 env，但未声明受信根 %s ⇒ 默认不采信 env 给出的执行面"
                "（#748 fail-closed）。请改用显式参数（`python=` / `pythonpath=`），"
                "或把该目录加入 %s 后再试：%r" % (var, roots_env, roots_env, value))
        return value
    real = os.path.normcase(os.path.realpath(value))
    for root in declared.split(os.pathsep):
        if not root:
            continue
        root_real = os.path.normcase(os.path.realpath(root))
        if real == root_real or real.startswith(root_real.rstrip("\\/") + os.sep):
            return value
    raise ValueError(
        "%s 的值不在受信根内：%r（%s 已声明受信根；#748 fail-closed）"
        % (var, value, roots_env))


def _require_env_interpreter(value: str) -> str:
    """`BRAIN_PYTHON` 兜底值校验（#652①，fail-closed；#748 收窄）。

    该值决定**被拉起的可执行程序**，故要求绝对路径且是已存在的文件——
    相对路径会按本进程 cwd 解析（cwd 可受外部影响），是比绝对路径更弱的面。
    显式参数 `python=`（含 CLI `--brain-python`）不经本校验：那是调用方的显式
    配置面，允许裸命令名（由 PATH 解析）。

    #748 在上述两条之后追加两道（只收窄 env 兜底面）：

    1. **解释器形状**：basename 须命中 `_PY_EXE_RE`（`python` / `python3` /
       `python3.12` / `python.exe` / `python312.exe` …）。「存在性」不是白名单：
       攻击者的毒目录本身存在、毒 `.cmd` 本身存在 ⇒ 旧口径放行任意可执行
       （`.cmd` / `.bat` / 任意名），实测仍跑到攻击者的 `md_cg/mcp_server.py`
       （#748：RCE 半成立）。
    2. **受信根**（**默认 fail-closed**）：env 兜底面**必须**声明
       `LINGSHU_BRAIN_PYTHON_ROOTS`，且归一化路径须落在其中某个根之下，
       否则拒。未声明受信根 ⇒ 不采信 env 给出的解释器（改用显式参数 `python=`，
       或声明受信根后再试）。

    残留面（如实）：能设置本进程 env 者同样能设置受信根变量 ⇒ 对「已能设置 env
    的攻击者」而言本面仍属**提权/持久化放大器**（与 main 既有定性一致），非默认
    可达 RCE。形状校验挡住的是「任意可执行（如 `.cmd`）」这一档；受信根把默认
    配置收成 fail-closed（一段继承/残留的 `BRAIN_PYTHON` 不再静默决定被执行的程序）。
    """
    if not os.path.isabs(value):
        raise ValueError(
            "BRAIN_PYTHON 必须是绝对路径（相对路径按本进程 cwd 解析，"
            "该 cwd 不受本模块控制；#652① fail-closed）。显式参数 `python=` "
            "不受此限，可传裸命令名")
    if not os.path.isfile(value):
        raise ValueError(
            "BRAIN_PYTHON 指向的不是已存在的文件：%r（#652① fail-closed）"
            % (value,))
    if not _PY_EXE_RE.match(os.path.basename(value)):
        raise ValueError(
            "BRAIN_PYTHON 指向的不是解释器（解释器形状不符，须形如 python / "
            "python3 / python3.12 / python.exe）：%r（#748 fail-closed）"
            % (value,))
    return _require_under_trusted_roots(value, _BRAIN_PYTHON_ROOTS_ENV,
                                        "BRAIN_PYTHON", required=True)


def _require_env_dir(value: str, var: str) -> str:
    """`BRAIN_PYTHONPATH` 兜底值校验（#652①，fail-closed；#748 收窄）。

    该值经 `_brain_env` 写进子进程 `PYTHONPATH`，CPython 启动期即按它解析模块
    （含自动导入 `sitecustomize`）⇒ 等于「子进程启动期执行哪个目录的代码」。
    故要求绝对路径且是已存在的目录。显式参数 `pythonpath=` 不经本校验。

    #748 追加两道（只收窄 env 兜底面）：

    1. **脑包结构**：目录下须解析出 `md_cg/__init__.py` 与
       `md_cg/mcp_server.py`。同样地「存在性」不是白名单——攻击者的毒目录本身
       就是一个存在的目录（#748）。
    2. **受信根**（**默认 fail-closed**）：env 兜底面**必须**声明
       `LINGSHU_BRAIN_PYTHONPATH_ROOTS`，且归一化路径须落在其中某个根之下，
       否则拒。未声明受信根 ⇒ 不采信 env 给出的模块搜索目录（改用显式参数
       `pythonpath=`，或声明受信根后再试）。

    残留面（如实）：能设置本进程 env 者同样能设置受信根变量 ⇒ 对「已能设置 env
    的攻击者」而言本面仍属**提权/持久化放大器**（与 main 既有定性一致），非默认
    可达 RCE；结构校验挡住的是「任意存在的目录」这一档，受信根把默认配置收成
    fail-closed。
    """
    if not os.path.isabs(value):
        raise ValueError(
            "%s 必须是绝对路径（#652① fail-closed；显式参数 `pythonpath=` 不受此限）"
            % (var,))
    if not os.path.isdir(value):
        raise ValueError(
            "%s 指向的不是已存在的目录：%r（#652① fail-closed）" % (var, value))
    if not (os.path.isfile(os.path.join(value, "md_cg", "__init__.py"))
            and os.path.isfile(os.path.join(value, "md_cg", "mcp_server.py"))):
        raise ValueError(
            "%s 指向的目录下未解析出 md_cg 脑包（须含 md_cg/__init__.py 与 "
            "md_cg/mcp_server.py）：%r（#748 fail-closed）" % (var, value))
    return _require_under_trusted_roots(value, _BRAIN_PYTHONPATH_ROOTS_ENV, var,
                                        required=True)


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
        # #652①：显式参数优先；env 兜底值**不可信**（父进程 env 决定被执行的
        # 程序与模块）⇒ fail-closed 校验（绝对路径＋存在性＋形状/结构，
        # #748）。见模块 docstring「父进程配置面」。
        if not python:
            env_python = os.environ.get("BRAIN_PYTHON")
            python = (_require_env_interpreter(env_python) if env_python
                      else sys.executable)
        if not pythonpath:
            env_pp = os.environ.get("BRAIN_PYTHONPATH")
            if env_pp:
                pythonpath = _require_env_dir(env_pp, "BRAIN_PYTHONPATH")
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
    temporal_coordinate: Optional[float] = None


@dataclass
class NodeRef:
    """engine.add_perception 的返回单元（legacy 代码访问 `.id`）。"""
    id: str


class _ConnShim:
    """legacy `ingest_scene` 直写语句 → 脑侧状态事件 的翻译垫片（单点）。

    契约：只认 `UPDATE nodes SET state_attributes=? WHERE id=?` 一条形态——
    payload 为 `{"state": ...}` JSON、第二参数为节点 id；翻译为
    `cg(op=state_event)`（subject=**该节点 id**、slot=「状态」、kind=acquisition
    ——issue #366：实体名跨场景不唯一，不作台账键）。
    其它 SQL 抛 NotImplementedError（不静默吞）；状态事件未入账时抛
    `BrainError`（同样不静默吞）。legacy 侧 `commit()` 为无操作。

    **能力声明（issue #596）**：本垫片补齐 `DBAPI` 形状的连接协议
    （`execute` / `executemany` / `commit` / `rollback`）并**显式声明边界**
    ——`atomic_batch = False`（不保证批量原子性）。消费方
    （`scene_model.ingest_scene` 的写入段）据此选路，不再用 AttributeError
    做能力探测：声明为假 ⇒ 直接逐条，不走「批量→失败→回滚→重试」。

    **状态输入的三种语义**（falsy 状态 = 「无状态可写」合法静默 / 非空且未落账
    必抛 `BrainError` / 非空且落账成功正常返回）见 `execute` 的 docstring
    ——那里写清了"为什么静默是对的"，免得下一轮把它当缺陷再报一次。
    """

    UPDATE_PREFIX = "UPDATE NODES SET STATE_ATTRIBUTES"

    #: 能力声明：`False` = 本连接**不保证** `executemany` 的批量原子性。
    #: 真 sqlite 连接没有这个属性 ⇒ 消费方 `getattr(conn, "atomic_batch", True)`
    #: 取缺省真，走批量；本垫片声明为假，消费方直接逐条。
    #: 为什么必须声明而不是"补上 executemany 就等于有事务"：见 `rollback`。
    atomic_batch = False

    def __init__(self, store: "BrainStore"):
        self.store = store

    def execute(self, sql, params=None):
        """翻译一条 legacy 状态写语句 → 脑侧状态事件。

        **三态语义（钉在这里；别当缺陷再报一次）** —— 对齐仓内既有口径
        `tests/test_state_write_observable.py` 的 T3：「空状态（`名称|类别|坐标|`）
        属**无状态可写**，不报警、重建回落 neutral」。

        | 输入 | 行为 | 告警 | 台账 |
        |---|---|---|---|
        | **falsy 状态**（`None` / `""`，或 payload 无 `state` 键） | 「**无状态可写**」⇒ **合法静默**：`BrainStore.record_state` 第一句就 `if not state: return False`（**不发 RPC**），本方法**不抛** | **无** | **不落** |
        | **非空状态 + `record_state` 返回 False** | 账该落却没落（脑端 RPC 回 `ok=False`）⇒ 抛 `BrainError` | **必须可见**（调用方 `ingest_scene` 转 `RuntimeWarning`，见 T2） | 不落 |
        | **非空状态 + 落账成功** | 正常返回 `self`（DBAPI 形状） | 无 | 落 1 条 |

        为什么 falsy **不**等同于"写失败"（这是仓内既有口径，不是本件新发明）：
        `ingest_scene` 对**第 4 字段留空**的行给出 `state=""` —— 那是**格式合法**
        的输入，语义是"该实体本轮没有状态"，不是"状态写失败了"（T3 的既有断言）。
        若把它当失败，则所有不带状态的场景行每次导入都刷 `RuntimeWarning`
        ⇒ 告警失去判别力，T2「台账不可写必须告警」也就跟着不可信了。
        反过来，**非空**状态失败必须响亮——这正是上面第二行的抛错。
        """
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

    def executemany(self, sql, seq_of_params):
        """逐条翻译落账（与逐条 `execute` 路径**逐位等价**），返回处理条数。

        为什么"批量"在这里只能是逐条（issue #596）：本垫片背后是**跨进程 RPC**
        （`store.record_state` → `client.call("cg", {op: state_event})`），物理上
        **不存在真事务**——第 k 条失败时，前 k−1 条已即时落进 append-only 事件台账，
        撤不回来。故此处逐条调用 `execute`，语义与逐条路径完全相同，只是把连接
        **协议的形状**补齐，让消费方不必再用 AttributeError 探测能力。

        任一条抛错立即向上抛（不吞、不假装部分成功）。调用方判断"能否靠批量兜底"
        的唯一依据是 `atomic_batch`——本类恒 `False`，所以 `ingest_scene` 在脑侧
        直接走逐条，不会先批量再回退。
        """
        n = 0
        for params in (seq_of_params or ()):
            self.execute(sql, params)
            n += 1
        return n

    def rollback(self):
        """**显式无操作** —— 这不是"假装回滚成功"，是确无可回滚。

        物理事实（决定本方法只能是 no-op）：`record_state` → `client.call("cg", …)`
        是对**另一进程（脑端 MCP）**的 RPC，每条状态事件在 `execute` 返回时就已
        即时落账（append-only 台账），本层**没有隐式事务可撤销**。

        写进协议是为了让调用方不必用 AttributeError 探测能力（issue #596），
        **不是**承诺"批量失败可以靠它回退"。调用方必须看 `atomic_batch`：本类为
        `False` ⇒ 批量部分失败不可回退 ⇒ 消费方**不得**走"批量→失败→rollback→
        逐条重试"——那会把已成功落账的事件**重复落账**（append-only），且重试若
        成功连告警都不会发（实测 3 实体、第 2 条失败 ⇒ 5 次 state_event、
        台账 4 行、首实体 2 行、0 告警）。
        """
        return None

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

    # ---- M1 读向 ----
    def read_content(self, node_id: str) -> str:
        """按已知记忆 ID 经 MCP 完整读取正文，包含脑端的续读页。"""
        return self._content(self.client.call("cg", {"op": "read", "node_id": node_id}))

    def _content(self, node: Dict) -> str:
        parts = [str(node.get("content") or "")]
        page = node
        while page.get("truncated"):
            page = self.client.call("cg", {"op": "read", "node_id": str(node["id"]),
                                            "offset": page["next_offset"]})
            parts.append(page["content"])
        return "\n".join(part.rstrip("\n") for part in parts)

    def _state_map(self, subjects: Optional[List[str]] = None) -> Dict[str, str]:
        """活动状态槽位投影：subject → value（slot=「状态」）。

        subject **＝节点 id**（不是实体名，issue #366）：台账是全局 append-only
        的，实体名在同一脑根内的多个场景间不唯一——用实体名作键时，另一场景的
        同名实体会覆盖本场景的状态（后写盖先写）。节点 id 全局唯一，故状态按
        「哪个节点」记账，读回时按节点 id 取（见 get_nodes_by_tag / record_state）。

        **截断纪律（issue #365）**：脑端 `stg(op=state_chain)` 的 `items` 是
        `units[:limit]`、命中总数在 `count`、`truncated = count > len(items)`
        （`md_cg/stg.py:state_chain`；投影按 `(subject, slot)` 字典序稳定排序，
        `md_cg/state_slots.py`）。固定 `limit=500` 且不看 `truncated` 时，全库状态
        槽位超过一页的场景，其主体若排在 500 名之后，状态会被静默读成 None
        （世界重建回落 neutral）。故本方法：① 只投影本层真正读的槽位
        （`slot=「状态」`下推，缩小命中集）；② 返回体 `truncated` 为真时按
        **调用方给出的 subject** 逐个精确补读——`subject`+`slot` 过滤后每个
        `(subject, slot)` 至多一个单元，不受全局截断影响；未给主体清单而无从
        补读时抛 `BrainError`（「没读到」不得当作「没有值」）。
        """
        m: Dict[str, str] = {}

        def _absorb(items) -> None:
            for u in (items or []):
                if u.get("slot") == STATE_SLOT and u.get("state") == "active":
                    m[str(u.get("subject"))] = u.get("value")

        resp = self.client.call("stg", {"op": "state_chain", "slot": STATE_SLOT,
                                        "limit": _STATE_PAGE})
        if not resp.get("truncated"):
            _absorb(resp.get("items"))
            return m
        if subjects is None:
            raise BrainError(
                "状态槽位投影被截断（count=%s > limit=%s）且未给出主体清单，"
                "无法取全——不静默补 None（issue #365）"
                % (resp.get("count"), _STATE_PAGE))
        for sid in subjects:
            r = self.client.call("stg", {"op": "state_chain", "subject": str(sid),
                                         "slot": STATE_SLOT, "limit": 1})
            if r.get("truncated"):
                raise BrainError("按主体 %r 的状态投影仍被截断：%r（issue #365）"
                                 % (str(sid), r))
            _absorb(r.get("items"))
        return m

    #: 允许进入世界重建的脑端资格态（cg(op=read) 返回体 item["state"]，四态之一）。
    #: 默认只踢 REJECT：场景实体节点本身缺 CCG 六要素 ⇒ 实测恒为 BLINDSPOT，
    #: 若只收 ACCEPT 世界重建会清零；世界是消费端，取保守纪律——只拒明确否决者
    #: （REJECT），BLINDSPOT（证据不足）/DEFER（待定）不拒。
    #: 调用方可显式收紧（accept_states={"ACCEPT"}）或传 set() 关闭过滤（旧行为）。
    DEFAULT_ACCEPT_STATES = frozenset({"ACCEPT", "BLINDSPOT", "DEFER"})

    def get_nodes_by_tag(self, tag: str, limit: int = 200,
                         accept_states: Optional[set] = None,
                         seed_query: Optional[str] = None,
                         view: Optional[str] = None) -> List[BrainNode]:
        """cg(op=read) 取候选 → **适配器侧按 tag 过滤**（裁定三：零脑改）→ BrainNode。

        坐标取 `frontmatter.spatial.coords3d`（裁定二）；状态取槽位投影。
        limit 为标签过滤前的候选预算；保留既有 max(1, int(limit)) 地板，
        所以 limit=0 仍请求一个候选，不表示空查询。状态投影另受 500 条预算约束。

        `limit` 译为脑端条数参数 `k`（issue #2）：脑端 `cg(op=read)` 的条数口径
        是 `k`（`_int_arg(a, "k", 20)`）；`limit` 在非 `budget_tokens` 路径**不被
        识别**、静默回落脑端缺省 20——写入 21 个场景实体会只重建 20 个。

        `accept_states`：资格态白名单（缺省 `DEFAULT_ACCEPT_STATES`＝只踢
        REJECT；传 `set()` 关闭过滤保留旧行为）。REJECT 来自脑端资格判据
        `judge_qualification`——不适用条件命中情境（query/context）即明确否决，
        不应作为可信几何注入世界重建；state 缺失（如脑端换判据面）不剔除
        （未知≠否决，fail-open，防静默清空）。

        **候选面取全再过滤（issue #367）**：脑端 `cg(op=read)` 回的是词法召回
        top-k；只取一页（`k=limit`）再在适配器侧按 tag 过滤时，别的场景把这一页
        占满，本场景就返回 0（`load_world_from_memory` 重建出空世界）。故此处按
        脑端返回体 `meta` 的候选总数（`pre_cap`＝截断**前**命中数，
        `md_cg/mdcg.py:4149,4638`）逐轮加大 `k` 直到覆盖整个候选面（上限＝脑端
        `GLOBAL_CAP=500`），再按 tag 过滤；候选面本身被脑端上限截断、`limit` 取不
        满时抛 `BrainError`——「无匹配」与「被截断」必须可分辨，不静默返回残缺
        世界。
        """
        if accept_states is None:
            accept_states = self.DEFAULT_ACCEPT_STATES
        want = max(1, int(limit))
        k = want
        while True:
            args = {"op": "read", "query": self.seed_query if seed_query is None else seed_query, "k": k}
            if view is not None:
                args["view"] = view
            resp = self.client.call("cg", args)
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
                out.append(BrainNode(
                    id=str(node.get("id") or fm.get("id") or ""),
                    content=self._content(node),
                    tags=tags,
                    spatial_coordinates=coords,
                    state_attributes=None,
                    temporal_coordinate=fm.get("created_at")))
            if len(out) >= want:
                break                             # 本场景候选已够 limit
            meta = resp.get("meta") or {}
            _pre = meta.get("pre_cap")
            total = int(_pre if _pre is not None
                        else (meta.get("candidates") or 0))
            nxt = min(total, BRAIN_CANDIDATE_CAP)
            if nxt <= k:
                if total > BRAIN_CANDIDATE_CAP:
                    raise BrainError(
                        "候选面被脑端上限截断：tag=%r 需 %d 条，取到 %d 条，"
                        "候选总数 %d > 脑端上限 %d——不静默返回残缺世界（issue #367）"
                        % (tag, want, len(out), total, BRAIN_CANDIDATE_CAP))
                break                             # 候选面已取尽：真的只有这么多匹配
            k = nxt
        # 状态按**节点 id** 取（#366）；按 id 精确补读（#365：全量投影会截断）。
        states = self._state_map([n.id for n in out])
        for n in out:
            state = states.get(n.id)
            if state:
                n.state_attributes = {"state": state}
        return out[:want]

    # ---- M2 写向（垫片目标）----
    def record_state(self, node_id: str, state: Optional[str]) -> bool:
        """把 legacy 的状态写翻译成 `cg(op=state_event)` 记账（subject=**节点 id**）。

        subject 用节点 id 而非实体名（issue #366）：台账是全局 append-only 的，
        实体名在同一脑根内的多场景间不唯一，用实体名会让另一场景的同名实体改掉
        本场景的状态；节点 id 全局唯一，状态按节点记账、按节点读回
        （_state_map / get_nodes_by_tag 同口径）。
        """
        if not state:
            return False
        r = self.client.call("cg", {"op": "state_event", "subject": str(node_id),
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
                       entities: Optional[List[str]] = None,
                       sensitivity: Optional[str] = None,
                       role: Optional[str] = None,
                       condition_space: Optional[dict] = None,
                       skip_dedup: bool = False,
                       **_kw) -> NodeRef:
        """`mdcg_remember` 直写一条观测（含 spatial.coords3d 直存）。

        **密级/来源显式透传**（issue #364）：`sensitivity`（密级）与 `role`
        （来源证据：user=外部惊奇 / command·tool-output=内部确定性）、
        `condition_space`（条件空间）**原样进 `mdcg_remember` 的 args**。此前本
        方法用 `**_kw` 把这三个名字静默吞掉：调用方声明 `private` 的观测按脑端
        缺省 `internal` 落盘（脑端 `docindex.DEFAULT_SENSITIVITY`），无令牌访客
        即可检索重建该实体——隐私边界被**静默降级**，且调用方无从察觉。

        **未知关键字 fail-closed**（同 #364）：凡未在上面具名的关键字一律
        `TypeError`，不得静默丢弃——「声明了却落不了盘」正是本缺陷的成因。
        新增调用方参数须在此显式声明并透传。

        `skip_dedup`：legacy 调用形（`scene_model.ingest_scene`）透传的 M5 去重
        开关。脑侧本层是**直写**（`gated=False`），脑端直写路径不做同构去重
        （脑包 `md_cg/mcp_server.py`「直写不去重是文档化现状」），故此处显式
        接收并**如实记录为脑侧无操作**——不静默吞掉名字，也不假装它生效。
        """
        if _kw:
            raise TypeError(
                "BrainEngine.add_perception 不接受未透传的关键字参数 %s——"
                "它们不会进 mdcg_remember（静默吞参＝声明静默失效，issue #364）；"
                "请在方法签名里显式声明并透传"
                % (sorted(_kw),))
        args = {"content": content, "layer": "contextual", "gated": False,
                "importance": float(importance), "tags": list(tags or [])}
        if sensitivity is not None:
            args["sensitivity"] = sensitivity
        if role is not None:
            args["role"] = role
        if condition_space is not None:
            args["condition_space"] = condition_space
        if spatial_coordinates:
            args["spatial"] = {"coords3d": {k: float(v)
                                            for k, v in spatial_coordinates.items()}}
        r = self.store.client.call("mdcg_remember", args)
        if not r.get("ok") or not r.get("id"):
            raise BrainError("mdcg_remember 未成功：%r" % (r,))
        # `entities` 保留以兼容 legacy 调用形（scene_model.ingest_scene 传实体名）；
        # 状态台账已改按节点 id 记账（issue #366），不再需要实体名映射。
        return NodeRef(id=str(r["id"]))


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


def load_world_from_brain(agent_or_store, tag: str = "spatial", limit: int = 200):
    """M1 便捷：按显式候选预算重建世界模型（默认 200，非全库枚举）。"""
    from .scene_model import load_world_from_memory
    return load_world_from_memory(agent_or_store, tag=tag, limit=limit)


def ingest_scene_to_brain(agent: BrainAgent, scene_desc, tag: str = "spatial",
                          relations=None):
    """M2 便捷：场景描述入脑（scene_model.ingest_scene 零改动；状态经 conn 垫片入账）。"""
    from .scene_model import ingest_scene
    return ingest_scene(agent, scene_desc, tag=tag, relations=relations)
