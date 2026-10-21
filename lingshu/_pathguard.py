# -*- coding: utf-8 -*-
"""_pathguard · 导入期路径护栏（lingshu issue #156 v2 · #273 裁定：路 B）
============================================================================
问题（issue #156 第二版 · 对抗性复核已证）：第一版只在 `core.py` 顶部装了
**组件解析守卫**（meta_path finder），它只对 15 个白名单组件名生效，且晚于
`core.py:17 import json`——`json` / `hashlib` / `hmac` / `sqlite3` / `threading`
/ `uuid` / `dataclasses` / `typing` / `inspect` / `ast` / `contextlib` /
`datetime` / `enum` / `collections` 这批**标准库裸名**仍可由 cwd / 空串
`sys.path` 条目抢先解析并**在引擎进程内执行**（独立复核实测 14 个名字逐个
`executed=True`，且毒模块同进程读到环境变量）。

机制（本模块 · **路 B · issue #273 裁定**）：

  ① **不删条目**——`scrub()` **不再从 `sys.path` 移除任何条目**：宿主进程的
     `sys.path` 保持原样（**不改写宿主进程的全局状态**、零副作用）。旧实现
     （v2 原版）在 `import lingshu` 时把 cwd / 空串条目**永久删除且不回滚**；
     裁定判定其「作用域与目的不匹配」——一个库导入不应永久改写宿主进程的全局
     状态（本件主诉的语义根）。
  ② **解析期拒绝**——`install_route_finder()` 在 `sys.meta_path` 装一个 finder
     （`_CwdRouteFinder`），其拒绝判据钉死为**通用判据**：「**解析结果落在
     cwd / 空串条目内**」。**不得**写成「名字 ∈ `sys.stdlib_module_names` ∪
     组件白名单」——`numpy` 不在前者、也不在后者，而 `lingshu/` 下
     `import numpy as np` 出现在 10 个以上文件 ⇒ 名字白名单会留一个**同型口子**
     （cwd 下的 `numpy.py` 会优先于 site-packages 真件被解析并在引擎进程内执行，
     正是 v1 漏洞形态换了一批名字）；已由 143B 负对照实测坐实（名字面下
     `__version__ == "POISON"`、MARKER 出现）。
  ③ **命中判据后的形态 = 「跳过该条目、在净化路径上继续解析」**（＝143B 实测的
     「臂 B」），**不是**「命中即抛」（＝143B 的「臂 A」）。臂 A 实测：连标准库
     `json` 本身都拿不到（`core.py:23 import json` 经 finder 命中 cwd 毒
     `json.py` ⇒ 命中即抛 ⇒ 引擎硬崩；156 守卫 94/113，15 条行为面条目 `rc=1`
     而 `marker` 全为 `False`）——即臂 A 的形态是**可用性崩，不是保护失效**；
     且臂 A 拒绝了**多于**「现状删除集合」的解析（连合法的 site-packages 同名件
     也拒），违背裁定帖理由 2「拒绝集合与删除集合**逐条等价**」。臂 B 实测
     109/113，转红 4 条逐条都是实现断言（`'' 不在 sys.path` /
     `cwd_entries_now == 0` / `removed_count ≥ 1` / `cwd_excluded_count ≥ 1`），
     **行为断言零转红**。

精确范围声明（**解析期拒绝「启动期快照内」的 cwd/空串 `sys.path` 面**）：
  - 覆盖：一切经 `sys.path` 的裸名导入（含标准库、含包内子模块）——解析结果落在
    **启动期快照内**的 cwd / 空串条目内即**跳过该条目**，改在**净化路径**
    （`clean_search_path()` = 快照内剔除 cwd/空串面 ＋ 快照之后新增的条目）上继续解析。
  - 只拒 `''` 与「归一后等于**启动期** cwd」的条目（**且须在快照内**）；
    `PYTHONPATH` 的**非 cwd 条目** / 显式配置根 `LINGSHU_COMPONENT_ROOT` /
    site-packages / `.zip` / egg / 命名空间包条目**不动**
    （`_is_cwd_entry` 返回 None 的条目不参与拒绝面）。
  - ★**拒绝面冻结为启动期快照**（本件 150A 收口 · #273 裁定帖理由 2）：
    `import lingshu`（装 finder）那一刻 `sys.path` 的**完整快照**被冻结，拒绝面
    只含**快照内**的 cwd / 空串条目那若干次出现；宿主在 import **之后**新增的条目
    （含 `sys.path.insert(0, <仓根绝对路径>)` 这类**运行期新增的 cwd 面条目**）
    **全部保留**。理由：基线（`3f1e944`）的删除面是「import 时把**当时**的 cwd
    面条目物理删除（一次）」——scrub 之后新增的条目**不受影响**；`47f364b` 的
    「每次解析动态重判当前 `sys.path`」使拒绝面成为删除面的**真超集**（S(t) ⊋ S₀），
    把 `tests/` 下 `sys.path.insert(0, REPO_ROOT)` ＋ `from tools import ...`
    （`tools` 是仓根**顶层包**）一并拒掉 ⇒ 10 件收集 ERROR。收口后二者**逐条等价**。
  - 快照**缺失**（捕获失败）⇒ 退回保守形态（按值剔除当前全部 cwd / 空串条目），
    行为不弱于现状。
  - ★**`PYTHONPATH` 归一后等于 cwd 的条目，不论来源都落入拒绝面**（须在快照内）。
    裁定帖「两项一并裁定」第 1 条：`sys.path` 条目**无来源标记**，实现层无法区分
    「解释器隐式插入的 cwd」与「部署方显式 `PYTHONPATH=<cwd>`」——`_is_cwd_entry`
    只拿到路径值。故本模块**不再**声称「`PYTHONPATH` 条目一律不动」；如实声明为：
    **显式 `PYTHONPATH=<cwd>` 与本层保护互斥，二者只能取一**。
  - 幂等：重复 import（模块缓存）不重复执行；重复调用 `scrub()` /
    `install_route_finder()` 安全、不报错。

仍**不覆盖**（如实）：
  - 进程已被**预置投毒**：`sys.modules` 预置、`sys.modules` 别名冒充（私域别名面，
    设计所需）；
  - ★**`import lingshu` 之后**由宿主新增的 cwd / 空串 `sys.path` 条目**不再被拒**
    ——这是与基线删除面等价的**刻意取舍**（安全面只许等价、不许放宽，也**不许**
    扩到超集）：`import lingshu` 之后才 `sys.path.insert(0, <cwd>)` 再导入 cwd 同名
    模块，本层不拦。代价面明确记录于此；
  - **逃生口** `LINGSHU_ALLOW_CWD_IMPORTS=1` 被显式打开（见下；路 B 下其语义为
    **不装 finder**，等于放弃本层）；
  - `python <script>.py` 且**脚本目录 != cwd** 时，脚本目录条目**不落入拒绝面**
    ——**脚本目录 ≠ cwd 时本模块不拒它、这一面保留不动（#421）**，属部署/脚本面；
    该面下的**白名单组件**由 `component_resolver` 另行排除
    （`safe_search_path` 排除启动面）；#750 残余收口起，白名单名在组件层
    受控面未命中即失败、**不再交还标准 `PathFinder`**，故该面也不再解析
    白名单名（宿主帧同理）；本模块对**非白名单名**维持这一边界不变——脚本
    目录≈宿主自有代码面，全面拒绝会拆掉宿主相邻模块导入（#275）；
    `python -m` 形态下解释器注入的**启动 cwd（包根）**同属此列（issue #421：
    启动面按**启动期快照**判定，不吃判定时 cwd）；
  - `import lingshu` **之前**已发生的 cwd 面解析（finder 尚未在位）；
  - `python -S` 极简启动（`os` 可能尚未加载，护栏降级为只处理 `''`/`.`）。

**持续在位**（裁定帖代价②）：finder 装在 `sys.meta_path` 上**持续生效**，故
`import lingshu` **之后**才发生的导入（含函数内延迟导入）同样落入拒绝面——
但拒绝面本身冻结在启动期快照（见上）：**快照内**的 cwd 条目持续被拒，
import **之后**新增的 cwd 条目不再被拒（与基线删除面等价）。

逃生口：`LINGSHU_ALLOW_CWD_IMPORTS` ∈ {1,true,yes,on}（大小写无关）⇒
**不装 finder**（`scrub()` 记录状态 + 发 `CwdImportsAllowed` 告警；
`pathguard_report()` 的 `escape_hatch_open=True` / `kept_cwd_kinds` 保持可观测）。
**打开它等于放弃本层保护**（cwd 同名模块，含标准库名，仍可在本进程内执行）。
该开关**只作用于本层**（一切裸名导入的 cwd/空串解析面）；白名单 15 个组件名
的受控面收口（`component_resolver`）自 #750 残余修复起**不随它放开**——受控面
未命中即按「组件缺失」失败，逃生口开也一样（见该模块 v4 段）。

与 `#288`/D-2 的**同源口径**（裁定帖存疑节点名要求「避免两份口径漂移」）：
本模块的拒绝面（cwd / 空串条目集合）与 `startup_controlled_entries()` /
`_is_cwd_entry` 的既有判定**同一来源**——finder 与受控面都不另写路径判定，
统一走 `_is_cwd_entry`（`''` / `.` / `./` / `.\\` / 归一后等于 cwd）；
净化路径亦由同一判定导出（`clean_search_path()`）。

零外部依赖 · 纯标准库（D-005）。本模块顶层只触碰 `sys`（内建）与已加载的
`os`，且在任何其它 import 之前完成状态判定；finder 亦**不 import 任何模块**
（标准 `PathFinder` 从 `sys.meta_path` 里取，见 `_std_path_finder()`）。
"""

import sys as _sys

# os 在解释器常规启动期已加载（site/io 链依赖它）；优先取用已加载实例，
# 避免在本模块顶层触发任何可能被 cwd 抢先解析的 import。
_os = _sys.modules.get("os")
if _os is None:  # 兜底：-S 等极简启动（此路径下 os 可能经 sys.path 解析，降级）
    try:  # noqa: E402
        import os as _os  # noqa: E402,F401
    except Exception:  # pragma: no cover - 环境异常
        _os = None

# 逃生口环境变量（等于显式放弃本层保护）
ALLOW_ENV = "LINGSHU_ALLOW_CWD_IMPORTS"

_APPLIED = False        # 是否已执行过一轮状态判定（sticky）
_ESCAPE = False         # 逃生口是否打开
_REMOVED_COUNT = 0      # 累计移除条目数（路 B 下恒 0——保留字段供观测面兼容）
_REMOVED_KINDS = []     # 累计移除条目「种类」（路 B 下恒空）
_KEPT_KINDS = []        # 逃生口打开时被保留的 cwd 类条目「种类」
_WARN_EMITTED = False   # 逃生口告警是否已发出

# ---- 解析期拒绝 finder（#273 裁定 · 路 B · 形态 = 143B 的「臂 B」）-----------
_ROUTE_FINDER = None          # 已安装的 finder 实例（未装为 None）
_ROUTE_REFUSED_IMPORTS = 0    # 累计被拒的导入次数（诊断；只报计数）
_ROUTE_REFUSED_NAMES = []     # 累计被拒的顶层名（只存名字，不含任何路径；上限 32）

# ---- 受控面启动期快照（issue #288 · D-2）-----------------------------------
# 受控面 = 白名单组件**允许**被解析到的面。它必须在**进程启动时刻**（本模块
# 首次 import 时，即 lingshu 包的最早落点）一次性判定并冻结，**不再随判定时的
# cwd 变化**；相对路径条目（其目标由 cwd 决定）**不**计入受控面。
_SNAPSHOT_DONE = False
_STARTUP_CWD = None          # 启动时刻的 cwd（归一；供分类，不外泄）
_STARTUP_SCRIPT_DIR = None   # 启动时刻的脚本目录（原始路径，供分类）
_CONTROLLED_ENTRIES = ()     # 启动时刻判定的受控面条目（绝对、非 cwd、非脚本目录、非相对）


# ---- 启动期 `sys.path` 快照（#273 裁定 · 拒绝面 ⊋ 删除面的收口 · 本件 150A）----
# 基线（`3f1e944`）的**删除面**＝「`import lingshu` 时把**当时** `sys.path` 里的
# cwd / 空串条目**物理删除**（一次）」；删除之后宿主新增的条目（含同值 cwd 条目）
# **不受影响**。`47f364b` 的 finder 改为「每次解析动态重判当前 `sys.path`」⇒
# 拒绝面 = S(t) ⊋ S₀，超出的正是「运行期新增的 cwd 面条目」——`tests/` 下多件
# `sys.path.insert(0, REPO_ROOT)` ＋ `from tools import ...`（`tools` 是仓根**顶层
# 包**、只存在于仓根）因此被误拒（跨件顺序依赖：件序上只要先有件 `import lingshu`
# ⇒ finder 在位 ⇒ 后续件的顶层包导入即被拒）。
#
# 收口口径（本件）：拒绝面**冻结为启动期快照**——
#   · `_PATH_SNAPSHOT`＝装 finder 那一刻（`import lingshu`）`sys.path` 的**完整
#     列表**（值与顺序）；
#   · `_PATH_SNAPSHOT_CWD`＝其中判为 cwd / 空串面的 `[(entry, kind), ...]`，
#     **启动期一次判定并冻结**（不吃判定时 cwd——与 `_STARTUP_CWD` 同源口径）；
#   · `_PATH_SNAPSHOT_CWD_COUNTS`＝上述条目的**值→出现次数**（多重集）。
# 净化路径只剔除「快照内 cwd 面条目」那**若干次出现**；快照之后新增的条目
# （按**多重集差**：当前某值出现次数超出快照计数的那部分）**全部保留**。
# ⇒ 拒绝面与删除面**逐条等价**（#273 裁定帖理由 2）；安全面只许等价、不许放宽
# （**不是**「净化路径找不到就回退返回原始 spec」——那等于取消拒绝面）。
#
# 实测依据（本件 150A 探针 · `wt150A @ 47f364b` · `pytest tests/ --collect-only`）：
# `import lingshu` 那一刻快照含仓根绝对路径 3 次；逐模块收集时 pytest 再插
# ⇒ `import tools` 时当前计数 6～40 次 ⇒ 多重集差非空 ⇒ 净化路径仍含仓根
# ⇒ `import tools` 成功。而 `-c` / `-m` 形态无新增 ⇒ 差为空 ⇒ 快照内 cwd 条目
# 全被剔除 ⇒ F1/F2 的「毒 cwd 文件未被执行」判据原样保住。
_PATH_SNAPSHOT = ()              # 装 finder 那一刻的 `sys.path` 完整快照（值+顺序）
_PATH_SNAPSHOT_CWD = ()          # 快照中判为 cwd / 空串面的 [(entry, kind), ...]
_PATH_SNAPSHOT_CWD_COUNTS = {}   # 上述条目的 值→出现次数（多重集；值均为 str）


# ---- 启动期「解释器注入的启动面」快照（issue #421）--------------------------
# `sys.path[0]` 由解释器在**进程启动期**按启动形态写定，是本进程里唯一不随
# 「判定时 cwd」漂移的启动面记录：
#   · `python <script>.py` ⇒ 脚本目录；· `python -m <mod>` ⇒ 启动 cwd（包根）；
#   · `python -c` / REPL ⇒ `''`（非绝对 ⇒ 不计入）。
# 必须在状态判定**之前**取值：`-m` 形态下该条目**等于启动 cwd**（旧实现会把它
# 移出 `sys.path`；路 B 下条目不再被移除，但取值仍按启动期快照语义钉死）。
# `-P` / `PYTHONSAFEPATH=1`（`sys.flags.safe_path`）下解释器**不注入**任何条目 ⇒
# 此时 `sys.path[0]` 是部署方条目（PYTHONPATH / site-packages），**不可**当启动面。
def _capture_startup_launch_dir():
    """冻结启动期 `sys.path[0]`（解释器注入面）；不可判定时 None。"""
    try:
        if bool(getattr(_sys.flags, "safe_path", False)):
            return None
        p0 = _sys.path[0] if _sys.path else None
    except Exception:  # pragma: no cover - sys.path / sys.flags 异常
        return None
    if not isinstance(p0, str) or not p0 or _os is None:
        return None
    try:
        if not _os.path.isabs(p0):
            return None
    except Exception:  # pragma: no cover - 非法路径
        return None
    return p0


_STARTUP_LAUNCH_DIR = _capture_startup_launch_dir()


# issue #487②：宿主可能在 `import lingshu` **之前** `sys.path.insert(0, vendor)`，
# 使 `sys.path[0]` 不再等于真实启动面（脚本目录被挤到后面）——只按 `sys.path[0]`
# 冻结启动面会把 vendor 当启动面、把真实脚本目录留进受控面（其下同名毒组件被
# 引擎构造期执行）。`sys.argv[0]` 仍是解释器收到的脚本路径，优先用它定位脚本
# 目录；`-m` 形态 argv[0] 是模块文件（`__main__.py`），其目录不是启动面，跳过；
# `-c` / REPL 的 `-c` / 空值同样跳过（回退 `sys.path[0]` 快照）。
def _capture_argv_script_dir():
    try:
        argv0 = _sys.argv[0] if _sys.argv else None
    except Exception:  # pragma: no cover - argv 异常
        return None
    if not isinstance(argv0, str) or not argv0 or argv0.startswith("-"):
        return None
    if _os is None:
        return None
    try:
        ap = _os.path.abspath(argv0)
        if not (_os.path.isfile(ap) and _os.path.basename(ap) != "__main__.py"):
            return None
        d = _os.path.dirname(ap)
        # `-m app.mod` 形态：argv[0] 是模块文件（不是 `__main__.py`），其目录
        # 通常**不在** sys.path（`-m` 注入的是启动 cwd）；脚本形态 argv[0] 目录
        # 必在 sys.path（解释器注入，可能被宿主前插挤到后面）⇒ 以此区分两者。
        keys = set()
        for e in _sys.path:
            if isinstance(e, str) and e:
                try:
                    keys.add(_os.path.normcase(_os.path.realpath(e)))
                except Exception:
                    pass
        if _os.path.normcase(_os.path.realpath(d)) not in keys:
            return None
        return d
    except Exception:  # pragma: no cover - 非法路径
        return None


_STARTUP_ARGV_SCRIPT_DIR = _capture_argv_script_dir()


class CwdImportsAllowed(UserWarning):
    """逃生口 `LINGSHU_ALLOW_CWD_IMPORTS` 被打开（cwd/空串解析面未被拒绝）。"""


def _escape_hatch_open() -> bool:
    """逃生口是否被显式打开（默认关）。"""
    if _os is None:
        return False
    try:
        v = _os.environ.get(ALLOW_ENV)
    except Exception:  # pragma: no cover
        return False
    if v is None:
        return False
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _norm(path: str) -> str:
    """路径归一（大小写/**符号链接**/相对/尾分隔符无关）。"""
    if _os is None:
        return path
    try:
        # issue #343：`realpath`（非 `abspath`）——同一目录经「软链/junction 路径」与
        # 「真实路径」两种别名出现时归一到同一键。判定「脚本目录」两侧必须用**同一**
        # 归一：`sys.path[0]` 由解释器在启动期写定、**保留** argv0 的未解析形态
        # （实测：经 junction 启动时 `sys.path[0]` 仍是 junction 路径），若只把
        # `main_script_dir` 一侧改 realpath，反而会把原本相符的两侧改成不符 ⇒ 护栏失效。
        return _os.path.normcase(_os.path.realpath(path))
    except Exception:  # 防御：非法路径按原样比较
        return path


def _is_cwd_entry(entry):
    """`sys.path` 条目是否为「当前工作目录」解析面（**拒绝面的唯一判据**）。

    返回种类描述（`"''"` / `"'.'"` / `"cwd"`）或 None（= 不落入拒绝面、条目不动）。
    仅 `''`、`.`、`./`、`.\\` 与「归一后等于 cwd」的条目判为命中。

    ★本函数是**唯一**的 cwd 面判定来源：`#273` 裁定要求 finder 的拒绝面与
    `#288`/D-2 受控面口径**同源**、避免两份口径漂移 ⇒ `_cwd_entries()` /
    `clean_search_path()` / `startup_controlled_entries()` 均复用它，不另写判定。
    """
    if not isinstance(entry, str):
        return None
    if entry == "":
        return "''"
    if entry in (".", "./", ".\\"):
        return repr(entry)
    if _os is None:
        return None
    try:
        cwd = _os.getcwd()
    except Exception:  # cwd 不可用（目录被删）时无法判定
        return None
    try:
        if _norm(entry) == _norm(cwd):
            return "cwd"
    except Exception:  # pragma: no cover
        return None
    return None


def _cwd_entries() -> list:
    """当前 `sys.path` 里的 cwd / 空串条目 `[(entry, kind), ...]`（**判定时**口径）。

    与 `_is_cwd_entry` **同一来源**（#273 裁定 §1.4：避免两份口径漂移）。
    ★本函数是「条目**此刻仍在** `sys.path` 里」的观测口径（`cwd_entries_now` 用它），
    **不是**拒绝面口径——拒绝面见 `_cwd_snapshot_entries()`（启动期快照冻结）。
    """
    try:
        cur = list(_sys.path)
    except Exception:  # pragma: no cover - sys.path 异常
        return []
    out = []
    for e in cur:
        kind = _is_cwd_entry(e)
        if kind is not None:
            out.append((e, kind))
    return out


def _capture_path_snapshot() -> None:
    """冻结装 finder 那一刻的 `sys.path` 完整快照（值与顺序）；幂等。

    ★这是**拒绝面的唯一口径来源**：`import lingshu`（装 finder）时一次判定，
    此后无论宿主如何增删/重排 `sys.path`，**快照内**的 cwd / 空串条目集合不变
    （与 `_STARTUP_CWD` / `_capture_controlled_surface()` 同款「启动期冻结」语义）。

    kind 用**捕获时**的 cwd 判定（即启动期 cwd）——宿主之后 `os.chdir()` 不会
    改写快照内条目的 kind（否则拒绝面会随判定时 cwd 漂移，正是本件要收掉的缺陷）。
    """
    global _PATH_SNAPSHOT, _PATH_SNAPSHOT_CWD, _PATH_SNAPSHOT_CWD_COUNTS
    if _PATH_SNAPSHOT:
        return
    _PATH_SNAPSHOT = tuple(_sys.path)
    items = []
    counts = {}
    for e in _PATH_SNAPSHOT:
        kind = _is_cwd_entry(e)
        if kind is None:
            continue
        items.append((e, kind))
        counts[e] = counts.get(e, 0) + 1
    _PATH_SNAPSHOT_CWD = tuple(items)
    _PATH_SNAPSHOT_CWD_COUNTS = counts


def _cwd_snapshot_entries() -> list:
    """**启动期快照内**的 cwd / 空串条目 `[(entry, kind), ...]`（**拒绝面**口径）。

    与 `_cwd_entries()` 的差别只在时间轴：本函数返回的是**冻结快照**——
    宿主在 `import lingshu` **之后**插入的 cwd / 空串条目**不计入**（那些条目
    与基线删除面下的新增条目同待遇：不受影响）。
    """
    _capture_path_snapshot()
    return list(_PATH_SNAPSHOT_CWD)


def clean_search_path() -> tuple:
    """**净化路径**：`sys.path` 剔除「**启动期快照内**的 cwd / 空串条目」后的结果。

    #273 裁定 §1.4：本函数与拒绝面同源（同一 `_is_cwd_entry` 判定），不另写一套；
    本件（150A）再收一层：**拒绝面冻结为启动期快照**，与基线删除面逐条等价——

      ① 快照**内**的 cwd / 空串条目（`_PATH_SNAPSHOT_CWD_COUNTS` 里的那若干次
         出现）⇒ **剔除**（F1 `-c` 的 `''`、F2 `-m` 的启动 cwd 绝对路径、pytest
         启动期插入的 rootdir 均在此列）；
      ② 当前 `sys.path` 里**超出快照计数**的条目（多重集差）⇒ **保留**——这正是
         基线删除面下「scrub 之后宿主新增的条目」（例如测试件
         `sys.path.insert(0, REPO_ROOT)` ＋ `from tools import ...`）；
      ③ 非 cwd / 空串面的条目 ⇒ 一律保留（与现状一致）。

    多重集差（而非「前段逐位置对齐」）是刻意的：pytest 会把 rootdir **反复前插**
    （实测同一值出现 3 → 6 → 40 次），位置对齐在宿主重排/前插时必然失配 ⇒ 会退回
    「按值剔除当前全部 cwd 条目」＝本件要修的缺陷形态。按值计数差对重排免疫。

    快照捕获失败（`_PATH_SNAPSHOT` 为空）⇒ 退回保守形态（按值剔除当前全部
    cwd / 空串条目）——行为不弱于现状。
    """
    try:
        cur = list(_sys.path)
    except Exception:  # pragma: no cover - sys.path 异常
        return ()
    if not _PATH_SNAPSHOT:
        # 快照缺失 ⇒ 保守形态（按值剔除当前全部 cwd / 空串条目；不弱于现状）。
        return tuple(e for e in cur if _is_cwd_entry(e) is None)
    budget = dict(_PATH_SNAPSHOT_CWD_COUNTS)
    out = []
    for e in cur:
        if isinstance(e, str) and budget.get(e, 0) > 0:
            budget[e] -= 1       # 该次出现属「快照内的 cwd 面条目」⇒ 剔除
            continue
        out.append(e)            # 快照之后新增 / 非 cwd 面条目 ⇒ 保留
    return tuple(out)


def _emit_escape_warning(n_kept: int):
    """逃生口告警：优先用已加载的 warnings；否则直接写 stderr（不新增 import）。"""
    global _WARN_EMITTED
    _WARN_EMITTED = True
    msg = ("[issue156-v2] %s=1 ⇒ 不装 cwd/空串解析面 finder（%d 个 cwd 类条目），"
           "本层保护已放弃：cwd 同名模块（含标准库名）仍可在本进程内执行。"
           % (ALLOW_ENV, n_kept))
    w = _sys.modules.get("warnings")
    if w is not None:
        try:
            w.warn(msg, CwdImportsAllowed, stacklevel=3)
            return
        except Exception:  # 告警通道不得改变导入语义（如 -W error）
            pass
    try:
        _sys.stderr.write("WARNING: CwdImportsAllowed: %s\n" % msg)
    except Exception:  # pragma: no cover
        pass


def scrub() -> bool:
    """判定并记录 cwd 面状态（**路 B：不改写宿主 `sys.path`**）。

    `#273` 裁定：本函数**不再从 `sys.path` 移除任何条目**——宿主的 `sys.path`
    保持原样（零副作用）。cwd / 空串条目的解析拒绝由 `install_route_finder()`
    装的 `sys.meta_path` finder 在**解析期**完成（见模块头）。

    幂等（可重复调用）。路 B 下**恒返回 `False`**（无任何移除动作）。
    逃生口打开时记录状态 + 发告警（且 `install_route_finder()` 不装 finder）。
    """
    global _APPLIED, _ESCAPE, _KEPT_KINDS, _WARN_EMITTED

    _APPLIED = True
    entries = _cwd_entries()

    if _escape_hatch_open():
        _ESCAPE = True
        _KEPT_KINDS[:] = [kind for _, kind in entries]
        if not _WARN_EMITTED:
            _emit_escape_warning(len(entries))
        return False

    _ESCAPE = False
    _KEPT_KINDS[:] = []
    return False


def _std_path_finder():
    """取标准 `PathFinder` 类（**不 import 任何模块**：从 `sys.meta_path` 里找）。

    `_pathguard` 的零依赖约束：顶层不得触发任何裸名 import（会被 cwd 抢先解析）。
    标准启动下 `PathFinder` 一定在 `sys.meta_path` 里（`__name__ == "PathFinder"`，
    实测 `_frozen_importlib_external.PathFinder`）。
    """
    try:
        for f in _sys.meta_path:
            if getattr(f, "__name__", "") == "PathFinder":
                return f
    except Exception:  # pragma: no cover - meta_path 异常
        return None
    return None


def _note_refusal(fullname: str) -> None:
    """记录一次「解析被拒」（只存顶层名，**不存任何路径**）。"""
    global _ROUTE_REFUSED_IMPORTS
    _ROUTE_REFUSED_IMPORTS += 1
    try:
        if len(_ROUTE_REFUSED_NAMES) < 32 and fullname not in _ROUTE_REFUSED_NAMES:
            _ROUTE_REFUSED_NAMES.append(fullname)
    except Exception:  # pragma: no cover
        pass


class _CwdRouteFinder:
    """`sys.meta_path` finder（issue #273 裁定 · 路 B · 形态 = 143B 的「臂 B」）。

    拒绝判据钉死为**通用判据**：「`PathFinder` 的解析结果落在 cwd / 空串条目内」
    ⇒ **跳过该条目、在净化路径上继续解析**（`clean_search_path()`）。

    **不是**「命中即抛」（臂 A）：臂 A 连标准库 `json` 本身都拿不到
    （`core.py:23 import json` 经 finder 命中 cwd 毒 `json.py` ⇒ 抛 ⇒ 引擎硬崩，
    实测 156 守卫 94/113、15 条行为面条目 `rc=1` 而 `marker=False`），且拒绝了
    多于「现状删除集合」的解析（连合法的 site-packages 同名件也拒），违背裁定帖
    理由 2「拒绝集合与删除集合逐条等价」。臂 B 才是逐条等价形态。

    **不是**「名字 ∈ `sys.stdlib_module_names` ∪ 组件白名单」（裁定帖理由 1）：
    `numpy` 等第三方裸名会留口子；143B 负对照实测：名字面下 cwd 毒 `numpy.py`
    被抢先解析并执行、`__version__ == "POISON"`。

    子模块（`path is not None`）一律 `return None`，随父包 `__path__` 走。
    """

    __slots__ = ()

    def find_spec(self, fullname, path=None, target=None):
        if path is not None:
            return None                      # 子模块随父包 __path__（裁定帖 §1.3）
        # 拒绝面＝**启动期快照内**的 cwd / 空串条目（本件 150A 收口）。快照为空
        # （捕获失败）⇒ 退回保守形态：当前无 cwd/空串条目才零干预。
        if not _PATH_SNAPSHOT:
            if not _cwd_entries():
                return None
        elif not _PATH_SNAPSHOT_CWD:
            return None                      # 快照内无 cwd/空串条目 ⇒ 拒绝面为空
        pf = _std_path_finder()
        if pf is None:
            return None
        # ① 先在**净化路径**上解析（跳过 cwd/空串条目）——臂 B 的核心。
        try:
            spec = pf.find_spec(fullname, list(clean_search_path()))
        except Exception:  # 解析异常不得改变导入语义
            spec = None
        if spec is not None:
            return spec
        # ② 净化路径没有 ⇒ 看原始 `sys.path` 是否**只有** cwd 面才有它。
        try:
            orig = pf.find_spec(fullname, list(_sys.path))
        except Exception:  # pragma: no cover
            orig = None
        if orig is None:
            return None                      # 两面都没有 ⇒ 交还后续 finder（原生缺失）
        _note_refusal(fullname)
        err = ModuleNotFoundError("No module named '%s'" % fullname)
        err.name = fullname
        err.cwd_surface_refused = True       # 诊断标记（不改 str(err) 文本）
        raise err

    def invalidate_caches(self):
        return None


def install_route_finder():
    """装**解析期拒绝** finder（路 B 的主体；幂等）。逃生口开 ⇒ **不装**。

    装在 `sys.meta_path[0]`（早于标准 `PathFinder`）；`component_resolver` 的
    组件 finder 随后插入其前（`meta_path[0]`）⇒ 白名单组件仍先走受控面。
    """
    global _ROUTE_FINDER, _ESCAPE

    if _escape_hatch_open():
        _ESCAPE = True
        return None

    # ★快照钉在「装 finder 那一刻」（＝`import lingshu` 时；幂等，模块 import 时已捕获）。
    _capture_path_snapshot()

    if _ROUTE_FINDER is None:
        _ROUTE_FINDER = _CwdRouteFinder()
    try:
        if _ROUTE_FINDER not in _sys.meta_path:
            _sys.meta_path.insert(0, _ROUTE_FINDER)
    except Exception:  # pragma: no cover - meta_path 异常
        return None
    return _ROUTE_FINDER


def is_route_finder_installed() -> bool:
    """解析期拒绝 finder 是否在位。"""
    try:
        return _ROUTE_FINDER is not None and _ROUTE_FINDER in _sys.meta_path
    except Exception:  # pragma: no cover
        return False


def refused_cwd_entry_count() -> int:
    """**被 finder 拒绝解析**的 cwd / 空串条目数（**启动期快照**口径）。

    逃生口关（finder 在位）⇒ = 装 finder 那一刻 `sys.path` 里判为 cwd / 空串面的
    条目数（≥1，若快照内有的话）；**不含** `import lingshu` 之后新增的 cwd 面条目
    （那些条目与基线删除面下的新增条目同待遇：不受影响）。
    逃生口开（不装 finder）⇒ **0**（无拒绝动作）。
    与 `cwd_entries_now`（条目**仍在** `sys.path` 的**判定时**计数）语义不同、不可相加。
    """
    if not is_route_finder_installed():
        return 0
    return len(_cwd_snapshot_entries())


def refused_cwd_kinds() -> list:
    """被拒解析的 cwd 类条目「种类」（`''` / `'.'` / `cwd`；**不含绝对路径**）。

    与 `refused_cwd_entry_count()` 同口径（启动期快照内的 cwd 面条目）。
    """
    if not is_route_finder_installed():
        return []
    out = []
    for _, kind in _cwd_snapshot_entries():
        if kind not in out:
            out.append(kind)
    return out


def ensure_scrubbed() -> bool:
    """幂等入口：确保已执行一轮状态判定（可被其它模块再调用）。返回 `scrub()` 结果。"""
    return scrub()


def main_script_dir():
    """启动期由解释器**注入**的「启动面」目录（或 None）。

    语义 = 「解释器在进程启动期隐式塞进 `sys.path` 的那个条目所指向的目录」：
      · `python <script>.py` ⇒ 脚本目录；
      · `python -m <mod>` ⇒ **启动 cwd（包根）**——`-m` 形态下 `sys.path[0]` 就是它，
        而非模块文件所在目录；
      · `-c` / REPL（`sys.path[0] == ''`）与 `-P` / `PYTHONSAFEPATH=1`
        （解释器不注入）⇒ None。

    这些均属「解释器隐式塞入的解析面」、不属部署方显式声明面 ⇒
    `component_resolver.safe_search_path()` 亦将其排除。

    issue #421：该值在**本模块 import 时**（状态判定之前）一次性冻结为
    `_STARTUP_LAUNCH_DIR`，**不再**由判定时的 `sys.argv[0]` / `os.getcwd()` /
    `sys.path[0]` 现算。旧实现在「import 前 chdir」时三者同时漂移：`-m` 形态下
    `sys.path[0]` 是启动 cwd，chdir 后它既不再等于 cwd（逃过旧 scrub）、又不等同
    `dirname(argv0)`（逃过脚本目录排除）⇒ 启动目录被当受控面保留，其下同名毒组件
    被导入执行。冻结后启动面与判定时刻无关（快照语义）。

    issue #343：快照经 `realpath` 归一（与 `_norm` 同一归一），故「别名路径启动」
    与「真实路径」两侧可比；`argv0` 不可解析时不再退回**判定时**的 `sys.path[0]`
    （#421 的漂移面），而用启动期快照。
    """
    # issue #487②：脚本形态优先用 argv[0] 的目录——它不受宿主 import 前
    # `sys.path.insert(0, vendor)` 影响；`-m` / `-c` / REPL 形态由上面的
    # `_capture_argv_script_dir()` 返回 None，自动回退 sys.path[0] 快照。
    if _STARTUP_ARGV_SCRIPT_DIR is not None:
        try:
            return _os.path.realpath(_STARTUP_ARGV_SCRIPT_DIR)
        except Exception:  # pragma: no cover - 非法路径
            return _STARTUP_ARGV_SCRIPT_DIR
    if _STARTUP_LAUNCH_DIR is not None:
        try:
            return _os.path.realpath(_STARTUP_LAUNCH_DIR)
        except Exception:  # pragma: no cover - 非法路径
            return _STARTUP_LAUNCH_DIR
    # 快照缺失（-P / PYTHONSAFEPATH / sys.path 异常）⇒ 退回 argv0 判定（原行为）。
    argv = _sys.argv
    if not argv:
        return None
    argv0 = argv[0]
    if not isinstance(argv0, str) or not argv0 or argv0.startswith("-"):
        return None
    if _os is None:
        return None
    try:
        if _os.path.isfile(argv0):
            return _os.path.dirname(_os.path.realpath(argv0))
    except Exception:
        return None
    # argv0 已不可解析（典型：import 前 chdir ⇒ 相对 argv0 相对新 cwd 找不到）。
    # 退回 `sys.path[0]`：`python <script>.py` 形态下即启动期写定的脚本目录绝对
    # 路径（与 cwd 无关）；`-m` 形态为启动 cwd（与 cwd 面重合，保守多排除一项，
    # 无害）；`-c`/REPL 形态为 `''` ⇒ 仍返回 None。
    try:
        p0 = _sys.path[0] if _sys.path else None
    except Exception:  # pragma: no cover - sys.path 异常
        return None
    if not isinstance(p0, str) or not p0:
        return None
    try:
        return _os.path.realpath(p0)
    except Exception:  # pragma: no cover
        return None


def is_script_dir_entry(entry) -> bool:
    """条目是否等于启动期由解释器**注入**的启动面目录（`main_script_dir()`）。

    issue #421 起该面含 `python -m <mod>` 的**启动 cwd（包根）**，不再只指
    `python <script>.py` 的脚本目录；判定用启动期冻结快照，不吃判定时 cwd。

    #273 路 B **不改变这一面**（脚本目录 ≠ cwd 时本模块不拒它）——保留不动。
    """
    sd = main_script_dir()
    if sd is None or not isinstance(entry, str) or entry == "":
        return False
    try:
        return _norm(entry) == _norm(sd)
    except Exception:  # pragma: no cover
        return False


def _is_relative_entry(entry) -> bool:
    """条目是否为**相对路径**（含空串 / `.` / `./` / `.\\`）。

    相对条目的目标目录由**判定时的 cwd** 决定 ⇒ 不可信、不属部署方显式声明面
    ⇒ 一律不计入受控面（issue #288 · D-2）。空串与 `.` 同属此列。
    """
    if not isinstance(entry, str):
        return True
    if entry == "" or entry in (".", "./", ".\\"):
        return True
    if _os is None:
        return True
    try:
        return not _os.path.isabs(entry)
    except Exception:  # pragma: no cover
        return True


def startup_controlled_entries() -> tuple:
    """**启动期快照**的受控面条目（issue #288 · D-2）。

    返回在本模块首次 import 时一次性判定的元组，判定依据：
      · **绝对路径**（相对条目目标由 cwd 决定 ⇒ 不计入）；
      · **非 cwd**（`_is_cwd_entry` 命中的条目——`''` / `.` / 启动期 cwd）；
      · **非脚本目录**（`python <script>.py` 的 `sys.path[0]` 脚本目录面）。

    **关键性质**：本函数返回的是**冻结快照**，与调用时刻的 cwd 无关——
    进程启动后 `os.chdir()` 到任何位置都不会改变其内容（D-2 的核心诉求：
    受控面由启动时刻或显式配置（`LINGSHU_COMPONENT_ROOT`）确定，不吃 cwd 判定）。

    #273 裁定 §1.4：本函数与 finder 的拒绝面**同源**（同一 `_is_cwd_entry`）。
    """
    _capture_controlled_surface()
    return _CONTROLLED_ENTRIES


def startup_cwd():
    """启动时刻的 cwd（归一）；不可用时为 None。仅作诊断，不外泄绝对路径。"""
    _capture_controlled_surface()
    return _STARTUP_CWD


def _capture_controlled_surface() -> None:
    """在进程启动期（本模块 import 时）一次性判定并冻结受控面。幂等。"""
    global _SNAPSHOT_DONE, _STARTUP_CWD, _STARTUP_SCRIPT_DIR, _CONTROLLED_ENTRIES
    if _SNAPSHOT_DONE:
        return
    _SNAPSHOT_DONE = True
    if _os is None:
        _CONTROLLED_ENTRIES = ()
        return
    try:
        _STARTUP_CWD = _norm(_os.getcwd())
    except Exception:  # pragma: no cover - cwd 不可用
        _STARTUP_CWD = None
    _STARTUP_SCRIPT_DIR = main_script_dir()
    out = []
    for e in list(_sys.path):
        if not isinstance(e, str) or e == "":
            continue                      # 空串 = cwd 解析面
        if _is_relative_entry(e):
            continue                      # 相对条目：cwd 依赖 ⇒ 非受控面
        if _is_cwd_entry(e) is not None:
            continue                      # 启动期 cwd 面：非受控面（#273 路 B 下条目
                                          # 仍留在 sys.path 里，但解析由 finder 拒绝）
        if _STARTUP_SCRIPT_DIR is not None:
            try:
                if _norm(e) == _norm(_STARTUP_SCRIPT_DIR):
                    continue              # 脚本目录面：非部署方显式声明
            except Exception:  # pragma: no cover
                pass
        out.append(e)
    _CONTROLLED_ENTRIES = tuple(out)


def pathguard_report() -> dict:
    """路径护栏只读摘要（供 component_resolver / self_check 观测）。

    只增字段；**不含本机绝对路径**（条目只报「种类」）。
    """
    cwd_now = _cwd_entries()
    return {
        "applied": _APPLIED,
        "cwd_scrub_active": _APPLIED and not _ESCAPE,
        "escape_hatch_env": ALLOW_ENV,
        "escape_hatch_open": _ESCAPE,
        # 路 B（#273 裁定）：不删条目 ⇒ 恒 0 / 空。字段保留供观测面兼容；
        # 「已排除」的新语义读数见下方 refused_cwd_entry_count。
        "removed_count": _REMOVED_COUNT,
        "removed_kinds": list(_REMOVED_KINDS),
        "kept_cwd_kinds": list(_KEPT_KINDS),
        "warning_emitted": _WARN_EMITTED,
        # 条目**仍在** sys.path 里的计数（路 B 下逃生口关时亦 ≥1）。
        "cwd_entries_now": len(cwd_now),
        # #273 路 B 新增：解析期拒绝面（与 cwd_entries_now 语义相反、不可相加）。
        "route_finder_installed": is_route_finder_installed(),
        "refused_cwd_entry_count": refused_cwd_entry_count(),
        "refused_cwd_kinds": refused_cwd_kinds(),
        "route_refused_imports": _ROUTE_REFUSED_IMPORTS,
        # issue #288 · D-2：受控面 = 启动期快照（不吃判定时 cwd）；只报计数与
        # 启动期 cwd 是否可用，不含任何绝对路径。
        "controlled_surface_count": len(_CONTROLLED_ENTRIES),
        "controlled_surface_snapshot": _SNAPSHOT_DONE,
        "startup_cwd_available": _STARTUP_CWD is not None,
    }


# 导入本模块即执行状态判定（`lingshu/__init__.py` 首行 import 本模块 ⇒ 最早落点）。
# 路 B（#273 裁定）：`scrub()` **不改写** `sys.path`——只判定/记录 cwd 面状态；
# 解析期拒绝由 `lingshu/__init__.py` 紧接着调用的 `install_route_finder()` 承担。
_IMPORT_REMOVED = scrub()
# issue #288 · D-2：在**启动期**一次性冻结受控面快照——此后无论 cwd 如何变化，
# 受控面判定不变。
_capture_controlled_surface()
# #273 裁定 · 本件 150A：在**装 finder 那一刻**（＝本模块 import 时，紧随
# `lingshu/__init__.py` 的 `install_route_finder()`）冻结 `sys.path` 完整快照，
# 把拒绝面钉死为「快照内的 cwd / 空串条目」——与基线删除面逐条等价（见模块头）。
_capture_path_snapshot()
