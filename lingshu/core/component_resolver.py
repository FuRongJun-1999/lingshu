# -*- coding: utf-8 -*-
"""
component_resolver · 组件发现面收口（lingshu issue #156）
============================================================================
背景：核心引擎按**裸模块名**（`from <name> import ...`）解析一批可选组件。
当 `sys.modules` 没有该名时，Python 会沿 `sys.path` 搜索——而 `python -c` /
`python -m` / REPL / pytest 形态下 `sys.path[0]` 是**当前工作目录**（空串
`''` 或 cwd 字面量）。于是「在工作目录放一个同名 `.py`」就能在引擎**构造期**
进程内执行任意代码（拿到活引擎与 `AEIS_DESIGNER_KEY`、伪造终裁、关停升级点）。

本模块把「这批组件的解析」收口到**部署方可控**的面，不再放任解释器隐式
塞进来的 cwd / 空串条目：

  ① `sys.modules` 已注册的别名——Python 导入语义天然最优先，本模块不干预。
     先例：AEIS 私域 `aeis/__init__.py` 的「命名空间注册」（`sys.modules
     ["entity_registry"] = aeis.entities` 等），值置 `None` 表示部署方显式
     声明「不可用」（`import` 将抛 `None in sys.modules`，语义不变）。
  ② 显式配置根 `LINGSHU_COMPONENT_ROOT`（os.pathsep 分隔，可多目录；
     沿用 `activation.py` 的 `LINGSHU_DATA_ROOT` 式显式根约定）。
  ③ `sys.path` 中**非 cwd** 的条目（`PYTHONPATH` / site-packages / 脚本目录
     等——均为部署方显式设置的面；部署方可控 ≠ 攻击者可控）。

三条之外即「解释器按当前目录解析」这一隐式面，被本模块关闭：白名单名字在
受控面上找不到时，按「组件缺失」抛**与原行为逐字同文本**的
`ModuleNotFoundError`（调用方既有 `try/except` 降级语义不变），并发出
`ComponentDiscoveryWarning`——装配失败此前完全静默（core.py 全文无
print/logging/warnings，构造期 stderr 为空），现在有可观测出口。

v2 补强（issue #156 第二版）：
  - `sys.path` 的 cwd/空串条目由 `lingshu._pathguard` 在**包首次 import 时装
    解析期拒绝 finder**（早于 core.py 的 `import json`）——#273 裁定（路 B）
    起**不删条目**、只拒绝解析（命中 cwd 面 ⇒ 跳过该条目、在净化路径上继续
    解析）。本模块的 `_is_cwd_entry` 仍保留，用于受控面判定。cwd / 空串 /
    脚本目录条目对白名单名**一律不是解析面**（**#750 残余修复起不再按调用方
    分档**）：受控面未命中即按「组件缺失」失败，与逃生口状态无关（见 v4 段）。
  - **脚本目录**（`python <script>.py` 形态的 `sys.path[0]`）同属「解释器隐式
    塞入的解析面」，不再被当作受控面 ⇒ 白名单名不从脚本目录解析（见
    `_is_script_dir_entry` / `_pathguard.main_script_dir`）。
  - **启动面**（issue #421）：`python -m <mod>` 形态下解释器注入的是**启动 cwd
    （包根）**，同属此列；`main_script_dir()` 按**启动期快照**返回该面 ⇒ 判定
    不随 import 后的 `chdir` 漂移（旧实现吃判定时 cwd/argv0 ⇒ 启动目录被保留）。

范围边界（精确）：本模块封的是**组件发现面**（白名单 15 名的裸名解析不经
cwd/空串/脚本目录条目）；「一切经 `sys.path` 的裸名解析（含标准库）的 cwd 面
**解析期拒绝**」由 `lingshu._pathguard` 承担（#273 裁定 · 路 B：不删条目）。
同进程代码可读 `os.environ`（密钥面）与 `sys.modules` 预置/别名冒充属**设计级**
事项，不在本模块处理。

v3 收窄（issue #275 · #156 v2 的回归）：上一版 finder 对白名单 15 个**通用顶层名**
不区分调用方，找不到即 `raise` ⇒ 宿主项目里同名的自有模块（`vision.py` /
`body.py` / `game_web/` …）在 `import lingshu.core` 之后一律 `ModuleNotFoundError`，
逃生口也无效。现按 `_importer_is_lingshu()` 只对**包内发起的**组件导入保留上述
收口；宿主发起的同名导入 `return None` 交还后续 finder。**削掉的判别力**：本模块
不再保证「宿主进程里任何代码 import 这些名字都拿不到 cwd/脚本目录里的同名模块」
——那本就不是本模块要保护的面（组件是引擎内部约定，非对外 API）；引擎自身的
组件导入仍逐字维持 v2 收口。

v4 残余收口（issue #750 残余 · EQRS #817 复核方向①）：v3/#782 的「宿主帧
`return None`」把整条 `sys.path`（含 cwd / 脚本目录条目）交还标准 `PathFinder`
⇒ A 臂（逃生口开 + cwd 毒件）、D 臂（逃生口关 + 脚本目录≠cwd + 脚本目录在
`sys.path[0]`）、E 臂（开 + 脚本目录≠cwd）的毒白名单组件仍在引擎进程内执行
（#782 只关掉「受控面有同名实现」的 B 臂）。本版起**白名单名的解析面只剩
`safe_search_path()`**（受控面）：未命中即按「组件缺失」失败（原生同文本
`ModuleNotFoundError` + 告警），**不再交还 `PathFinder`**——与调用方（包内 /
宿主）、与逃生口状态均无关。#275 的「宿主自有同名模块必须可解析」保证随之
**按名收窄**为两条支持径：① 放在受控面上（`LINGSHU_COMPONENT_ROOT` /
`PYTHONPATH` 显式声明 / site-packages）；② 早于本 finder 生效（先于
`import lingshu.core` 导入、或 `sys.modules` 预注册——Python 导入语义天然最
优先，本模块不干预）。逃生口 `LINGSHU_ALLOW_CWD_IMPORTS` 仍保留其**原有语义**
（只作用于 `_pathguard` 层非白名单名的 cwd 解析面），不再恢复白名单名的
cwd / 脚本目录解析。取舍依据见 issue #750 线程（TRIAGE-750 §四 / MERGE-782
-750-RESIDUAL §四「收口点」；EQRS #817 §一处置方向①）。

零外部依赖 · 纯标准库（D-005）。
"""

from __future__ import annotations

import importlib.machinery
import os
import sys
import warnings

from .._pathguard import (
    main_script_dir as _main_script_dir,
    pathguard_report as _pathguard_report,
    startup_controlled_entries as _startup_controlled_entries,
)

# 白名单：仓内以裸名解析的全部组件（含 AEIS `aeis/__init__.py` 注册别名集）。
# 新增裸名导入时须同步登记，守卫 tests/test_issue156_component_discovery.py
# 的 D 组以 AST 扫描**全仓裸名**并按「标准库 / 组件 / 已知第三方 / 未知」分类，
# 「未知」即判红（v2：旧 D 组先按本表过滤 ⇒ 恒真空转，已修真）。
COMPONENT_NAMES = frozenset({
    # —— 构造期（SpacetimeMemoryEngine.__init__ 的 _setup_* 调用链）——
    "entity_registry",
    "cognitive_orchestrator",
    "blindspot_learning_loop",
    "semantic_space",
    "attention_policy",
    "prediction_engine",
    "lifecycle_engine",
    "flywheel_engine",
    "self_cognition_engine",
    "body",
    # —— 调用期（同类形态）——
    "vision",
    "pattern_separation",
    "scene_reconstruction",
    "spacetime_memory_core",
    "game_web",
})

# 显式配置根（部署方声明组件所在目录；os.pathsep 分隔可多目录）
COMPONENT_ROOT_ENV = "LINGSHU_COMPONENT_ROOT"


class ComponentDiscoveryWarning(UserWarning):
    """组件未在受控解析面找到（装配失败）告警。

    不阻塞导入流程、不改变 `*_error` 字段文本；仅提供可观测出口。
    """


def _norm(path: str) -> str:
    """路径归一（大小写/**符号链接**/相对/尾分隔符无关），用于判定 cwd / 脚本目录条目。

    issue #343：改用 `realpath` 并与 `_pathguard._norm` **同归一**——`sys.path[0]`
    由解释器写定、保留 argv0 的未解析形态（经 junction/symlink 启动时即别名路径），
    故「脚本目录」两侧必须同样解析软链，否则同一目录的两种别名互相判不等 ⇒ 排除失效。
    """
    try:
        return os.path.normcase(os.path.realpath(path))
    except Exception:  # 防御：非法路径按原样比较
        return path


def _is_cwd_entry(entry) -> bool:
    """`sys.path` 条目是否为「当前工作目录」（`''` / `.` / cwd 字面量）。

    这三类即解释器在 -c/-m/REPL/pytest 下隐式塞入的 cwd 解析面。
    """
    if not isinstance(entry, str):
        return False
    if entry == "":
        return True
    if entry in (".", "./", ".\\"):
        return True
    try:
        cwd = os.getcwd()
    except Exception:  # cwd 不可用（目录被删）时无法判定
        return False
    return _norm(entry) == _norm(cwd)


def _is_script_dir_entry(entry) -> bool:
    """`sys.path` 条目是否为 `python <script>.py` 形态的**脚本目录**。

    脚本目录同属「解释器隐式塞入」的解析面（非部署方显式声明），故亦不视为
    受控面（v2 补强：上一版只判 cwd/空串 ⇒ 脚本目录 != cwd 时毒白名单组件
    仍被执行）。issue #421：`python -m <mod>` 形态下解释器注入的是**启动 cwd
    （包根）**，`_pathguard.main_script_dir()` 现按**启动期快照**返回该面 ⇒ 本
    函数同时覆盖 `-m` 启动目录（判定不随 import 后的 chdir 漂移）。
    """
    if not isinstance(entry, str) or entry == "":
        return False
    sd = _main_script_dir()
    if not sd:
        return False
    return _norm(entry) == _norm(sd)


def configured_roots() -> list:
    """显式配置的组件根（`LINGSHU_COMPONENT_ROOT`，按顺序去空项）。"""
    raw = os.environ.get(COMPONENT_ROOT_ENV, "")
    return [p for p in raw.split(os.pathsep) if p.strip()]


def _is_relative_entry(entry) -> bool:
    """条目是否为**相对路径**（含空串 / `.` / `./` / `.\\`）。

    issue #288 · D-2：相对条目的目标目录由**判定时的 cwd** 决定，不属部署方显式
    声明面 ⇒ 一律不计入受控面。旧实现靠 `_is_cwd_entry`（吃判定时 cwd）间接排除
    「恰好等于当前 cwd 的相对条目」——cwd 一变，同一条目的判定就变（#273/#275 的
    根因面）。本函数**不看 cwd**，只按「是否绝对路径」判定，故与判定时刻无关。
    """
    if not isinstance(entry, str):
        return True
    if entry == "" or entry in (".", "./", ".\\"):
        return True
    return not os.path.isabs(entry)


def safe_search_path() -> list:
    """受控搜索面：**启动期快照**的受控面条目 ＋ 显式配置根（去重保序）。

    issue #288 · D-2：受控面不再由**判定时**的 `sys.path` 与 cwd 现算，而是取
    `_pathguard` 在**进程启动期**冻结的快照（`startup_controlled_entries`）——
    启动后 `os.chdir()` 到任何位置都不改变受控面（旧实现吃 cwd，见 `_is_cwd_entry`）。
    快照只含**绝对路径**条目且已排除 cwd / 脚本目录；**相对路径条目一律不计入**
    （其目标由 cwd 决定，属 #273/#275 的核心面）。

    另注意：显式配置根（`LINGSHU_COMPONENT_ROOT`）**先加且永不受** cwd/脚本目录
    判定影响 ⇒ 部署方声明的组件位置始终可达（正对照，防过度收紧）。
    """
    roots: list = []
    seen: set = set()

    def _add(p: str):
        key = _norm(p)
        if key not in seen:
            seen.add(key)
            roots.append(p)

    for p in configured_roots():
        _add(p)
    for entry in _startup_controlled_entries():
        if not isinstance(entry, str) or entry == "":
            continue
        if _is_relative_entry(entry):          # 防御：快照不应含相对条目
            continue
        _add(entry)
    return roots


# 本包根目录（用于判定调用栈帧是否来自 lingshu 包内）
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _importer_is_lingshu() -> bool:
    """当前这次裸名导入是否由 **lingshu 包内的代码**发起（issue #275）。

    判定：沿调用栈向上找第一个「有归属」的帧，看它属于 lingshu 包内还是宿主。

      · 包内：模块名以 `lingshu` 开头，或**代码对象**来自本包根目录之下 ⇒ True
        （引擎自己的组件导入，须继续走 #156 的受控面收口）；
      · 宿主 / `__main__` / 其它库 ⇒ False（v3/#782 时代据此 `return None`
        交还后续 finder；v4 起该差别不再改变解析结果，仅作诊断观测）。

    无法取得调用栈（非 CPython 实现等）时返回 True —— 保守维持 #156 的收口
    （宁可多收，不可放松安全面）。#750 残余收口（v4）后 `find_spec` 对**所有**
    调用方一视同仁（受控面未命中即失败），本函数不再决定「raise 还是 return
    None」，保留为判据面自身（#652④ 伪造方向守卫）与诊断观测——「宿主帧」
    与「包内帧」的**解析结果**自此没有差别，判别结果不再影响任何安全判据。

    **判据取值面（#652④）**：第二判据取 `frame.f_code.co_filename`——它是**代码
    对象**的编译期事实（这段代码编译自哪个文件），而**不是**模块全局
    `__file__` / `__name__`（调用方随时可写）。旧实现取 `f_globals["__file__"]`，
    于是「危险方向」可伪造：**包内文件执行的代码**把 `__file__` 改成包外路径、
    `__name__` 改成宿主名 ⇒ 本函数返回 False ⇒ 交还 `PathFinder` ⇒ 该次裸名
    导入逃出受控面（读 cwd/sys.path 上的同名模块）。实测读数见
    `tests/test_issue652_importer_identity_guard.py`。取代码对象事实后，包内文件
    无论怎样改模块全局都**不可能**再把自己判成宿主（正向伪造 lingshu 名仍为
    True = fail-closed 方向，无害）。
    """
    try:
        frame = sys._getframe(1)
    except Exception:  # pragma: no cover - 非 CPython / 帧不可用
        return True
    while frame is not None:
        name = frame.f_globals.get("__name__")
        if isinstance(name, str) and name and not name.startswith(
                ("importlib", "lingshu.core.component_resolver")):
            if name == "lingshu" or name.startswith("lingshu."):
                return True
            f = frame.f_code.co_filename
            if isinstance(f, str) and f:
                try:
                    if os.path.abspath(f).startswith(_PKG_ROOT + os.sep):
                        return True
                except Exception:  # pragma: no cover - 非法路径
                    pass
            return False
        frame = frame.f_back
    return False


class _ComponentFinder:
    """meta_path finder：只对 `COMPONENT_NAMES` 生效，不走 cwd/空串/脚本目录条目。

    被询问时（即 `sys.modules` 未命中——别名缺失）**一律**在受控面
    （`safe_search_path()`）上查找；受控面命中即返回 spec——**不论调用方是包内
    还是宿主、不论逃生口开或关**（#782 起；#750 残余收口起对未命中分支同样
    不再分档）。受控面**未命中**时按「组件缺失」抛原生同文本
    `ModuleNotFoundError`（**不**回落到 cwd / 脚本目录解析面，**不**交还
    `PathFinder`），并告警。

    issue #750 残余（#817 复核坐实的 A/D/E 三臂）：v3/#782 在受控面未命中且
    调用栈来自宿主时 `return None` 交还后续 finder ⇒ 标准 `PathFinder` 按**整条
    `sys.path`**（含 cwd / 脚本目录条目）解析——A（逃生口开+cwd 毒件执行）、
    D（逃生口关+脚本目录≠cwd，脚本目录条目在 `sys.path[0]`，毒件执行）、
    E（开+脚本目录≠cwd，毒件执行）。本类自 v4 起封死该交还：白名单 15 名的
    裸名解析**只**发生在受控面上。宿主确需同名自有模块的两条支持径见模块头
    v4 段（受控面显式声明 / 早于本 finder 导入）。
    """

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in COMPONENT_NAMES:
            return None
        # issue #750 残余收口（EQRS #817 方向①）：白名单名**只**在受控面
        # safe_search_path() 上解析——受控面未命中时不再 `return None` 交还
        # 标准 PathFinder（那会把整条 sys.path、含 cwd / 脚本目录条目一并
        # 交回：A=逃生口开+cwd 毒件、D=逃生口关+脚本目录≠cwd 且脚本目录在
        # sys.path[0]、E=开+脚本目录≠cwd，三臂的毒模块正是在引擎进程内执行
        # 的那几条路）。#782 已关掉「受控面有同名实现」的 B 臂（先查受控面、
        # 命中即返回）；本处关掉其余三臂：未命中即按「组件缺失」失败，与
        # 调用方（包内/宿主）、与逃生口状态均无关。#275 的宿主自有同名模块
        # 保证按名收窄——两条支持径（受控面显式声明 / 早于本 finder 导入）
        # 见模块头 v4 段。
        roots = safe_search_path()
        spec = importlib.machinery.PathFinder.find_spec(fullname, roots)
        if spec is not None:
            return spec
        # 受控面缺失 ⇒ 按「组件缺失」失败（原生同文本异常 + 可观测告警）
        try:
            warnings.warn(
                "[issue156] 组件 %r 未在受控解析面找到（已查 %d 个条目；%s %s；"
                "cwd/空串/脚本目录 sys.path 条目已排除）。若确需该组件，请用 %s 或 "
                "PYTHONPATH 显式声明其所在目录。" % (
                    fullname, len(roots), COMPONENT_ROOT_ENV,
                    "已设置" if configured_roots() else "未设置",
                    COMPONENT_ROOT_ENV),
                ComponentDiscoveryWarning, stacklevel=2)
        except Exception:  # 告警通道不得改变导入语义（如 -W error 场景）
            pass
        err = ModuleNotFoundError("No module named '%s'" % fullname)
        err.name = fullname
        err.component_search_roots = roots  # 诊断用（不改 str(err) 文本）
        raise err

    def invalidate_caches(self):
        return None


_FINDER = None


def install() -> "_ComponentFinder":
    """安装组件解析守卫（幂等，可重复调用；仅首次调用注册告警过滤规则）。"""
    global _FINDER
    if _FINDER is None:
        _FINDER = _ComponentFinder()
    if _FINDER not in sys.meta_path:
        sys.meta_path.insert(0, _FINDER)
        try:  # 确保本类告警默认可观测（即便外部 -W error 也不改变导入语义）
            warnings.filterwarnings("default", category=ComponentDiscoveryWarning)
        except Exception:
            pass
    return _FINDER


def is_installed() -> bool:
    """守卫是否已安装。"""
    return _FINDER is not None and _FINDER in sys.meta_path


def discovery_report() -> dict:
    """组件发现面只读摘要（供 self_check / 诊断；不含本机绝对路径）。"""
    try:
        os.getcwd()
        cwd_ok = True
    except Exception:
        cwd_ok = False
    entries = [p for p in sys.path if isinstance(p, str)]
    _pg = _pathguard_report()
    _cwd_live = sum(1 for p in entries if _is_cwd_entry(p))
    return {
        "guard_installed": is_installed(),
        "component_names": sorted(COMPONENT_NAMES),
        "configured_root_env": COMPONENT_ROOT_ENV,
        "configured_roots_set": bool(configured_roots()),
        "search_entries": len(safe_search_path()),
        # cwd 类条目被挡在受控面之外的计数 = **被 `_pathguard` 的解析期拒绝
        # finder 拒绝解析**的 cwd / 空串条目数（`path_scrub.refused_cwd_entry_count`）
        # ——路 B（#273 裁定）下这才是字面意义上的「已排除」：条目**仍留在**
        # `sys.path` 里（宿主状态零改写），但其上的解析被跳过。
        # issue #338：旧式 `removed_count + _cwd_live` 把「已消除」与「仍存活于
        # sys.path、只是被本模块忽略」（逃生口打开时）两个**语义相反**的量相加，
        # 使「护栏生效」与「逃生口打开（放弃防护）」两状态下该字段读数相同（实测
        # 均为 3）⇒ 计数无法反映防护是否生效。#273 路 B 后旧口径
        # （`removed_count`）恒为 0 ⇒ 两状态不可区分，本字段改取「被拒解析」口径：
        # 逃生口关（finder 在位）⇒ ≥1；逃生口开（不装 finder）⇒ 0。
        # 仍存活的 cwd 条目由下一字段 `cwd_entries_in_sys_path` 单独给出，二者
        # 语义不同、不可相加。
        "cwd_excluded_count": _pg["refused_cwd_entry_count"],
        "cwd_entries_in_sys_path": _cwd_live,
        "script_dir_excluded": _main_script_dir() is not None,
        "script_dir_excluded_count": sum(1 for p in entries if _is_script_dir_entry(p)),
        "cwd_available": cwd_ok,
        # issue #288 · D-2：受控面 = 启动期快照（不吃判定时 cwd）。
        # 只增字段，不含本机绝对路径。
        "relative_entries_excluded": sum(1 for p in entries if _is_relative_entry(p)),
        # issue #156 v2：路径护栏状态（cwd/空串条目是否已从 sys.path 消除、
        # 逃生口是否被打开、移除了哪些「种类」）——只增字段，不含本机绝对路径。
        "path_scrub": _pg,
    }
