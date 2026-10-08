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
  - `sys.path` 的 cwd/空串条目由 `lingshu._pathguard` 在**包首次 import 时**
    全局消除（早于 core.py 的 `import json`）——本模块的 `_is_cwd_entry` 仍保留，
    用于逃生口打开（cwd 条目仍在）时白名单名**依旧不**从 cwd 解析。
  - **脚本目录**（`python <script>.py` 形态的 `sys.path[0]`）同属「解释器隐式
    塞入的解析面」，不再被当作受控面 ⇒ 白名单名不从脚本目录解析（见
    `_is_script_dir_entry` / `_pathguard.main_script_dir`）。

范围边界（精确）：本模块封的是**组件发现面**（白名单 15 名的裸名解析不经
cwd/空串/脚本目录条目）；「消除一切经 `sys.path` 的裸名解析（含标准库）」由
`lingshu._pathguard` 承担。同进程代码可读 `os.environ`（密钥面）与
`sys.modules` 预置/别名冒充属**设计级**事项，不在本模块处理。

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
    """路径归一（大小写/相对/尾分隔符无关），用于判定 cwd 条目。"""
    try:
        return os.path.normcase(os.path.abspath(path))
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
    仍被执行）。
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


def safe_search_path() -> list:
    """受控搜索面：显式配置根在前，其后为 sys.path 的非 cwd/非脚本目录条目（去重保序）。"""
    roots: list = []
    seen: set = set()

    def _add(p: str):
        key = _norm(p)
        if key not in seen:
            seen.add(key)
            roots.append(p)

    for p in configured_roots():
        _add(p)
    for entry in list(sys.path):
        if _is_cwd_entry(entry) or _is_script_dir_entry(entry):
            continue
        _add(entry)
    return roots


class _ComponentFinder:
    """meta_path finder：只对 `COMPONENT_NAMES` 生效，不走 cwd/空串条目。

    被询问时（即 `sys.modules` 未命中——别名缺失）在受控面上查找；找不到即
    抛原生同文本 `ModuleNotFoundError`（**不**回落到 cwd 解析面），并告警。
    """

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in COMPONENT_NAMES:
            return None
        roots = safe_search_path()
        spec = importlib.machinery.PathFinder.find_spec(fullname, roots)
        if spec is not None:
            return spec
        # 受控面缺失 ⇒ 按「组件缺失」失败（原生同文本异常 + 可观测告警）
        try:
            warnings.warn(
                "[issue156] 组件 %r 未在受控解析面找到（已查 %d 个条目；%s %s；"
                "cwd/空串 sys.path 条目已排除）。若确需该组件，请用 %s 或 "
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
        # cwd 类条目被挡在受控面之外的计数 = 已由 _pathguard 从 sys.path 消除的
        # （`path_scrub.removed_count`）+ 仍在 sys.path 但被本模块忽略的（逃生口打开时）。
        "cwd_excluded_count": _pg["removed_count"] + _cwd_live,
        "cwd_entries_in_sys_path": _cwd_live,
        "script_dir_excluded": _main_script_dir() is not None,
        "script_dir_excluded_count": sum(1 for p in entries if _is_script_dir_entry(p)),
        "cwd_available": cwd_ok,
        # issue #156 v2：路径护栏状态（cwd/空串条目是否已从 sys.path 消除、
        # 逃生口是否被打开、移除了哪些「种类」）——只增字段，不含本机绝对路径。
        "path_scrub": _pg,
    }
