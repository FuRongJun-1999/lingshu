# -*- coding: utf-8 -*-
"""_pathguard · 导入期路径护栏（lingshu issue #156 v2）
============================================================================
问题（issue #156 第二版 · 对抗性复核已证）：第一版只在 `core.py` 顶部装了
**组件解析守卫**（meta_path finder），它只对 15 个白名单组件名生效，且晚于
`core.py:17 import json`——`json` / `hashlib` / `hmac` / `sqlite3` / `threading`
/ `uuid` / `dataclasses` / `typing` / `inspect` / `ast` / `contextlib` /
`datetime` / `enum` / `collections` 这批**标准库裸名**仍可由 cwd / 空串
`sys.path` 条目抢先解析并**在引擎进程内执行**（独立复核实测 14 个名字逐个
`executed=True`，且毒模块同进程读到环境变量）。

机制（本模块）：在 `lingshu` 包被**首次 import** 时（Python 语义：`import
lingshu.core.core` 必先执行 `lingshu/__init__.py` ⇒ 早于 `core.py` 的一切裸名
import），把 `sys.path` 中**解释器隐式塞入的 cwd / 空串条目**移除。这是审计
实测**唯一**能一次封住「所有 cwd 同名模块」（含标准库）的开关
（`PYTHONSAFEPATH=1` 等价物，进程内实现形式）。

精确范围声明（**消除 cwd/空串 `sys.path` 解析**）：
  - 覆盖：一切经 `sys.path` 的裸名导入（含标准库、含包内子模块）。
  - 只移 `''` 与「等于当前工作目录」的条目；`PYTHONPATH` / 显式配置根
    `LINGSHU_COMPONENT_ROOT` / site-packages / `.zip` / egg / 命名空间包条目
    一律**不动**（`_is_cwd_entry` 返回 None 的条目不参与移除）。
  - 幂等：重复 import（模块缓存）不重复执行；重复调用 `scrub()` 安全、不报错。

仍**不覆盖**（如实）：
  - 进程已被**预置投毒**：`sys.modules` 预置、进程内 `sys.path.insert`、
    `sys.modules` 别名冒充（私域别名面，设计所需）；
  - **逃生口** `LINGSHU_ALLOW_CWD_IMPORTS=1` 被显式打开（见下，等于放弃本层）；
  - `python <script>` 且**脚本目录 != cwd** 时，脚本目录条目（本模块不移除它，
    属部署/脚本面）；该面下的**白名单组件**由 `component_resolver` 另行排除
    （`safe_search_path` 排除脚本目录）；
  - cwd 条目在 `lingshu` import **之后**被重新插入 `sys.path`（本模块只在
    import 期一次性消除；无持续守卫）；
  - `python -S` 极简启动（`os` 可能尚未加载，护栏降级为只处理 `''`/`.`）。

逃生口：`LINGSHU_ALLOW_CWD_IMPORTS` ∈ {1,true,yes,on}（大小写无关）⇒ 跳过本轮
移除（保留 cwd 条目），发出 `CwdImportsAllowed` 告警，并在 `pathguard_report()`
中记 `escape_hatch_open=True` / `kept_cwd_kinds`。**打开它等于放弃本层保护**
（cwd 同名模块，含标准库名，仍可在本进程内执行）。

零外部依赖 · 纯标准库（D-005）。本模块顶层只触碰 `sys`（内建）与已加载的
`os`，且在任何其它 import 之前完成移除。
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

_APPLIED = False        # 是否已执行过一轮护栏（sticky）
_ESCAPE = False         # 逃生口是否打开
_REMOVED_COUNT = 0      # 累计移除条目数
_REMOVED_KINDS = []     # 累计移除条目「种类」（'' / '.' / cwd；不含本机绝对路径）
_KEPT_KINDS = []        # 逃生口打开时被保留的 cwd 类条目「种类」
_WARN_EMITTED = False   # 逃生口告警是否已发出


class CwdImportsAllowed(UserWarning):
    """逃生口 `LINGSHU_ALLOW_CWD_IMPORTS` 被打开（cwd/空串解析面未被消除）。"""


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
    """`sys.path` 条目是否为「当前工作目录」解析面。

    返回种类描述（`"''"` / `"'.'"` / `"cwd"`）或 None（= 不动该条目）。
    仅 `''`、`.`、`./`、`.\\` 与「归一后等于 cwd」的条目判为 True。
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


def _emit_escape_warning(n_kept: int):
    """逃生口告警：优先用已加载的 warnings；否则直接写 stderr（不新增 import）。"""
    global _WARN_EMITTED
    _WARN_EMITTED = True
    msg = ("[issue156-v2] %s=1 ⇒ 保留 cwd/空串 sys.path 条目（%d 个），"
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
    """默认执行：从 `sys.path` 移除 `''` 与「等于当前工作目录」的条目。

    幂等（可重复调用；重复调用只在再次发现 cwd 条目时补扫）。返回本次是否
    有移除动作。逃生口打开时不移除（记状态 + 告警）。
    """
    global _APPLIED, _ESCAPE, _REMOVED_COUNT, _WARN_EMITTED

    _APPLIED = True

    if _escape_hatch_open():
        _ESCAPE = True
        kept = [k for k in (_is_cwd_entry(e) for e in list(_sys.path)) if k]
        _KEPT_KINDS[:] = kept
        if not _WARN_EMITTED:
            _emit_escape_warning(len(kept))
        return False

    _ESCAPE = False
    removed = []
    kept_path = []
    for e in list(_sys.path):
        kind = _is_cwd_entry(e)
        if kind is not None:
            removed.append((e, kind))
        else:
            kept_path.append(e)
    if not removed:
        return False
    _sys.path[:] = kept_path
    _REMOVED_COUNT += len(removed)
    for _, kind in removed:
        if kind not in _REMOVED_KINDS:
            _REMOVED_KINDS.append(kind)
    return True


def ensure_scrubbed() -> bool:
    """幂等入口：确保已执行一轮（可被其它模块再调用）。返回 `scrub()` 结果。"""
    return scrub()


def main_script_dir():
    """`python <script>.py` 形态下 `sys.path[0]` 所指向的**脚本目录**（或 None）。

    `-c`（argv0 == '-c'）/ REPL 形态返回 None。`-m`（argv0 为模块文件，其目录
    通常不在 sys.path）返回该模块文件所在目录——两者均非 `sys.path[0]` 的脚本
    目录面，故不构成误排除。脚本目录同属「解释器隐式塞入的解析面」，不属部署方
    显式声明面 ⇒ `component_resolver.safe_search_path()` 亦将其排除。

    issue #343：路径经 `realpath` 归一（与 `_norm` 同一归一，两侧方可比），
    且当 argv0 **已不可解析**时退回进程启动期由解释器写定的 `sys.path[0]`——
    否则「启动期相对 argv0 + import 前 chdir」会让 `abspath(argv0)` 相对**新**
    cwd 解析（`isfile` 落空 ⇒ 旧实现返回 None）⇒ 脚本目录不被排除 ⇒ 同目录
    毒组件被导入执行。
    """
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
    """条目是否等于 `python <script>.py` 的脚本目录。"""
    sd = main_script_dir()
    if sd is None or not isinstance(entry, str) or entry == "":
        return False
    try:
        return _norm(entry) == _norm(sd)
    except Exception:  # pragma: no cover
        return False


def pathguard_report() -> dict:
    """路径护栏只读摘要（供 component_resolver / self_check 观测）。

    只增字段；**不含本机绝对路径**（移除条目只报「种类」）。
    """
    entries = [e for e in _sys.path if isinstance(e, str)]
    return {
        "applied": _APPLIED,
        "cwd_scrub_active": _APPLIED and not _ESCAPE,
        "escape_hatch_env": ALLOW_ENV,
        "escape_hatch_open": _ESCAPE,
        "removed_count": _REMOVED_COUNT,
        "removed_kinds": list(_REMOVED_KINDS),
        "kept_cwd_kinds": list(_KEPT_KINDS),
        "warning_emitted": _WARN_EMITTED,
        "cwd_entries_now": sum(1 for e in entries if _is_cwd_entry(e) is not None),
    }


# 导入本模块即执行护栏（`lingshu/__init__.py` 首行 import 本模块 ⇒ 最早落点）。
_IMPORT_REMOVED = scrub()
