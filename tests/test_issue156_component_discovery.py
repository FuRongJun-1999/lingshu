# -*- coding: utf-8 -*-
"""test_issue156_component_discovery · lingshu issue #156 守卫（v2）
============================================================================
缺陷（lingshu issue #156 · 安全：cwd 投毒 → 引擎进程内执行）：
  引擎按**裸模块名**解析组件与一众标准库；`python -c` / `python -m` / REPL /
  pytest 形态下 `sys.path[0]` 是**当前工作目录**（空串 `''` 或 cwd 字面量）。
  cwd 放同名 `.py` 即可在构造期进程内执行任意代码（活引擎、读
  `AEIS_DESIGNER_KEY`、伪造 D-007 终裁），且**完全静默**。

v1 的漏（独立对抗性复核已证，见报告 ①）：
  · 只封了 15 个**白名单组件名**；`core.py:17 import json` 等**标准库裸名**在
    组件守卫（`core.py:40`）之前就经 cwd 解析 ⇒ 实测 `json/hashlib/hmac/
    sqlite3/threading/uuid/dataclasses/typing/inspect/ast/contextlib/datetime/
    enum/collections` 14 名逐个 `executed=True`（PWN.txt 生成、rc=0、静默装配）。
  · `python <script>.py` 且**脚本目录 != cwd** 时，脚本目录里的白名单同名毒文件
    被执行（脚本目录被当成"受控面"）。
  · 旧 D 组自称「AST 扫描断言全仓裸名 ⊆ 白名单」，实则**先按白名单过滤** ⇒
    恒真空转（注入新裸名守卫仍 81/81 全绿）。

v2 修法：
  ① **最早落点**（`lingshu/__init__.py` 首行 + `core.py` 顶部）：在包首次 import
     时由 `lingshu/_pathguard.py` 从 `sys.path` 移除 **cwd/空串条目** ⇒ 一切经
     `sys.path` 的裸名导入（含标准库）不再落到 cwd。幂等、可观测、带逃生口
     `LINGSHU_ALLOW_CWD_IMPORTS`（打开即放弃本层保护，留告警 + 状态）。
  ② 组件解析收口（`component_resolver.py`）：白名单名只从 sys.modules 别名 /
     显式根 `LINGSHU_COMPONENT_ROOT` / sys.path 非 cwd / **非脚本目录** 条目解析；
     受控面缺失即抛原生同文本 `ModuleNotFoundError`（不回落 cwd）。
  ③ 装配失败告警出口：ComponentDiscoveryWarning + `self_check()["component_assembly"]`
     （只增字段，含 `discovery.path_scrub`）。
  ④ 密钥声明收窄到事实（同进程可读，属设计级事项）。

断言组（除 D 外全部走子进程，隔离环境；毒目录一律 tempfile.mkdtemp）：
  A 组（组件注入面关闭）：逐名毒 cwd 在 `-c` / pytest / 链式 / 调用期 / 齐上阵
     形态下均不执行；受控面缺失即原生错误。
  B 组（告警 + 装配状态可观）：ComponentDiscoveryWarning（stderr）+ self_check 字段。
  C 组（装配成功路径未坏 · 反向腿）：显式根 / PYTHONPATH / sys.modules 别名 /
     包+子模块 / **非 cwd 合法条目（含 ghost .zip/egg）保留**。
  D 组（全仓裸名分类 · 有牙）：AST 扫描**全仓**裸名 → {标准库 / 组件 / 已知第三方 /
     未知}；**未知即红**；并**自证有牙**（沙箱副本注入新裸名 ⇒ 必被检出）。
  F 组（标准库劫持面 · v2 核心）：逐名毒 cwd（json/hashlib/.../collections）
     `-c` 形态不执行；另 `-m` 形态一条（证绝对 cwd 条目亦被消除）。
  G 组（逃生口）：关＝cwd 条目被消除；开＝保留 + 告警 + 状态标 `escape_hatch_open`。
  S 组（脚本目录边界）：`python <script>.py` 且脚本目录 != cwd 时，脚本目录里的
     白名单毒组件不被执行（v1 漏点）。
  E 组（边界诚实）：verify_designer docstring 收窄到事实。

运行（lingshu 仓根）：python -X utf8 tests/test_issue156_component_discovery.py
退出码：0 = 全过；1 = 有断言失败
"""
from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lingshu.core.component_resolver import COMPONENT_NAMES  # noqa: E402

# ------------------------------------------------------------------ 分类白名单
# (a) 标准库白名单——**显式列出**仓内实际出现的标准库顶层名（每类附理由）。
#     理由：v2「消除 cwd/空串 sys.path 解析」覆盖一切经 sys.path 的裸名，故标准库
#     名同样受保护；本表只用于 AST 分类判据（把标准库与「未知」分开）。
#     下方 D0b 断言本表 ⊆ sys.stdlib_module_names（解释器权威清单），避免误收。
STDLIB_ALLOWED = frozenset({
    # 解释器/未来特性
    "__future__",
    # 序列化 / 编码
    "json", "base64", "hashlib", "hmac",
    # 容器 / 迭代 / 类型 / 数据类 / 枚举
    "collections", "itertools", "typing", "dataclasses", "enum",
    # 运行时 / 导入 / 告警
    "sys", "os", "importlib", "warnings", "pathlib", "contextlib",
    "ctypes", "datetime", "logging", "socket", "socketserver", "ssl", "struct",
    # 并发 / 时间 / 唯一定名（queue：2026-10-09 采纳 PR #260 时 brain_store.py 引入——
    # 标准库、与 threading 同族；守卫按设计判红「未知裸名」，此处补分类而非放宽判据）
    "threading", "queue", "time", "uuid",
    # 存储
    "sqlite3",
    # IO / 文本 / 数值 / 随机
    "io", "re", "math", "random",
    # 命令行 / 子进程
    "argparse", "subprocess",
})
# (b) 已知第三方依赖（pyproject extras 声明；显式列出 + 理由）
THIRD_PARTY_ALLOWED = frozenset({
    "numpy",    # world/nn/gen extras
    "PIL",      # nn/gen extras（Pillow）
    "scipy",    # activation 可选稀疏
    "torch",    # gen 可选
    "diffusers",  # gen 可选
    "pyarrow",  # nn/hex_train 可选
})
# (c) 本包自名（绝对自导入）
SELF_ALLOWED = frozenset({"lingshu"})

# 构造期裸名（SpacetimeMemoryEngine.__init__ 的 _setup_* 调用链）→ 失败信息字段
CONSTRUCT_TIME = [
    ("entity_registry", "_v13_error"),
    ("cognitive_orchestrator", "_v13_error"),
    ("blindspot_learning_loop", "_v13_error"),
    ("semantic_space", "_semantic_error"),
    ("attention_policy", "_attention_error"),
    ("prediction_engine", "_prediction_error"),
    ("lifecycle_engine", "_lifecycle_error"),
    ("flywheel_engine", "_flywheel_error"),
    ("self_cognition_engine", "_self_cognition_error"),
    ("body", "_body_error"),
]
# 白名单全集（扫描面必须覆盖的最小集）
WHITELIST_EXPECTED = frozenset(
    [n for n, _ in CONSTRUCT_TIME]
    + ["vision", "pattern_separation", "scene_reconstruction",
       "spacetime_memory_core", "game_web"])

# v2 核心判据：引擎 import 图内、v1 下被 cwd 劫持并执行的标准库名（逐个验）
STDLIB_HIJACK_NAMES = [
    "json", "hashlib", "hmac", "sqlite3", "threading", "uuid", "dataclasses",
    "typing", "inspect", "ast", "contextlib", "datetime", "enum", "collections",
]

_PASS: list = []
_FAIL: list = []


def ok(cond, msg, extra=""):
    (_PASS if cond else _FAIL).append(msg)
    print(("  PASS " if cond else "  FAIL ") + msg
          + (("  ← " + str(extra)) if (extra and not cond) else ""))


def sub_env(path=None, extra=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = path or REPO
    env["PYTHONUTF8"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"     # 不写任何 .pyc（含毒目录与仓内）
    env.pop("LINGSHU_COMPONENT_ROOT", None)  # 清宿主干扰
    env.pop("LINGSHU_ALLOW_CWD_IMPORTS", None)
    env.pop("AEIS_DESIGNER_KEY", None)       # 本守卫不碰真密钥
    if extra:
        env.update(extra)
    return env


def run_py(code, cwd, path=None, extra=None, args=None):
    argv = [sys.executable, "-B"] + (args or ["-c", code])
    return subprocess.run(argv, cwd=cwd, env=sub_env(path, extra),
                          capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


POISON_SYMBOLS = {
    "entity_registry": "class EntityRegistry:\n    def __init__(self, *a, **k): pass\n",
    "cognitive_orchestrator": "class CognitiveOrchestrator:\n    def __init__(self, *a, **k): pass\n",
    "blindspot_learning_loop": "class BlindSpotLearningLoop:\n    def __init__(self, *a, **k): pass\n",
    "semantic_space": "class SemanticSpaceProvider:\n    def __init__(self, *a, **k): pass\n",
    "attention_policy": "class AttentionPolicy:\n    def __init__(self, *a, **k): pass\n",
    "prediction_engine": "class PredictionEngine:\n    def __init__(self, *a, **k): pass\n",
    "lifecycle_engine": "class LifecycleEngine:\n    def __init__(self, *a, **k): pass\n",
    "flywheel_engine": "class FlywheelEngine:\n    def __init__(self, *a, **k): pass\n",
    "self_cognition_engine": "class SelfCognitionEngine:\n    def __init__(self, *a, **k): pass\n",
    "body": "def build_default_registry(workspace=\"\"):\n    return None\n",
}

POISON_TEMPLATE = """# poison component for issue156 guard (top-level code must NOT run)
import os as _os
with open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "MARKER.txt"),
          "w", encoding="utf-8") as _f:
    _f.write("EXECUTED cwd=%s\\n" % _os.getcwd())
{symbols}"""

# 标准库毒名模板：顶层写 MARKER，并（若是可运行的）留一个可导入的替身
STDLIB_POISON_TEMPLATE = '''# poison stdlib for issue156 v2 guard (top-level code must NOT run)
import os as _os
with open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "MARKER.txt"),
          "w", encoding="utf-8") as _f:
    _f.write("EXECUTED stdlib=" + _os.path.basename(__file__) + "\\n")
'''


def make_poison_dir(base, name, tag="", with_pytest_file=False):
    # tag 使每组各自独立目录（避免前一组残留 MARKER 污染后一组判据）
    d = os.path.join(base, ("poison_%s_%s" % (tag, name)) if tag else ("poison_" + name))
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name + ".py"), "w", encoding="utf-8") as f:
        f.write(POISON_TEMPLATE.format(symbols=POISON_SYMBOLS[name]))
    if with_pytest_file:
        with open(os.path.join(d, "test_poison_form.py"), "w", encoding="utf-8") as f:
            f.write(textwrap.dedent("""
                import os

                def test_component_discovery_guard():
                    from lingshu.core.core import SpacetimeMemoryEngine
                    SpacetimeMemoryEngine(":memory:")
                    here = os.path.dirname(os.path.abspath(__file__))
                    assert not os.path.exists(os.path.join(here, "MARKER.txt")), \\
                        "cwd/poison component executed under pytest form"
                """))
    return d


def make_stdlib_poison_dir(base, name, tag):
    d = os.path.join(base, "stdlib_%s_%s" % (tag, name))
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name + ".py"), "w", encoding="utf-8") as f:
        f.write(STDLIB_POISON_TEMPLATE)
    return d


CONSTRUCT_DUMP = """
import json, os, sys
from lingshu.core.core import SpacetimeMemoryEngine
e = SpacetimeMemoryEngine(":memory:")
here = os.getcwd()
print("##JSON##" + json.dumps({
    "marker": os.path.exists(os.path.join(here, "MARKER.txt")),
    "err": repr(getattr(e, %r, "<missing>")),
    "in_modules": %r in sys.modules,
}, ensure_ascii=False))
"""


# ---------------------------------------------------------------- A 组
def group_a_bash_form(base):
    print("== A 组：逐名毒 cwd —— -c 形态构造（MARKER 不得出现） ==")
    for name, err_field in CONSTRUCT_TIME:
        d = make_poison_dir(base, name, tag="bash")
        r = run_py(CONSTRUCT_DUMP % (err_field, name), cwd=d)
        j = None
        for line in (r.stdout or "").splitlines():
            if line.startswith("##JSON##"):
                j = json.loads(line[len("##JSON##"):])
        ok(r.returncode == 0 and j is not None, f"A1[{name}] -c 构造 rc=0 且有读数",
           f"rc={r.returncode} stdout={r.stdout!r} stderr={r.stderr[:200]!r}")
        if j:
            ok(j["marker"] is False, f"A2[{name}] 毒同名 .py 未被执行（MARKER 不存在）", j)
            ok(j["in_modules"] is False, f"A3[{name}] 毒模块未进入 sys.modules", j)
            ok(isinstance(j["err"], str) and "No module named" in j["err"],
               f"A4[{name}] 该组件未装配且错误文本为原生形态（{err_field}）", j)


def group_a_pytest_form(base):
    print("== A 组：pytest 形态构造（毒目录内含 test 文件，pytest 收集即 sys.path[0]=毒目录） ==")
    for name, _ in CONSTRUCT_TIME:
        d = make_poison_dir(base, name, tag="pytest", with_pytest_file=True)
        r = subprocess.run(
            [sys.executable, "-B", "-m", "pytest", d, "-q", "-p", "no:cacheprovider"],
            cwd=d, env=sub_env(REPO), capture_output=True, text=True,
            encoding="utf-8", errors="replace")
        marker = os.path.exists(os.path.join(d, "MARKER.txt"))
        ok(r.returncode == 0 and not marker,
           f"A5[{name}] pytest 形态：毒组件未执行、守卫测试绿",
           f"rc={r.returncode} marker={marker} out={(r.stdout or '')[-300:]!r}")


def group_a_chain(base):
    """A7：v13 块第 2/3 行位置的组件也走受控面（前一导入成功时不得回落 cwd）。

    显式根提供 entity_registry（第一行成功）→ 毒 cwd 提供 cognitive_orchestrator /
    blindspot_learning_loop（第二/三行）⇒ 后者不得从 cwd 解析执行。"""
    print("== A 组：链式位置收口——前一行成功时后续裸名仍不走 cwd ==")
    root = os.path.join(base, "chainroot")
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "entity_registry.py"), "w", encoding="utf-8") as f:
        f.write("class EntityRegistry:\n    FAKE = True\n"
                "    def __init__(self, store): self.store = store\n")
    d = os.path.join(base, "poison_chain")
    os.makedirs(d, exist_ok=True)
    for nm in ("cognitive_orchestrator", "blindspot_learning_loop"):
        with open(os.path.join(d, nm + ".py"), "w", encoding="utf-8") as f:
            f.write(POISON_TEMPLATE.format(symbols=POISON_SYMBOLS[nm]))
    code = textwrap.dedent("""
        import glob, os
        from lingshu.core.core import SpacetimeMemoryEngine
        e = SpacetimeMemoryEngine(":memory:")
        here = os.getcwd()
        print("CHAIN_MARKERS:", sorted(glob.glob(os.path.join(here, "MARKER*.txt"))))
        print("CHAIN_COG:", e._cognition, "CHAIN_LOOP:", e._learning_loop)
        print("CHAIN_ERR:", repr(e._v13_error))
        """)
    r = run_py(code, cwd=d, extra={"LINGSHU_COMPONENT_ROOT": root})
    out = r.stdout or ""
    ok(r.returncode == 0 and "CHAIN_MARKERS: []" in out,
       "A7a 链式毒文件（cognitive_orchestrator/blindspot_learning_loop）未被执行", out[:400])
    ok("CHAIN_COG: None CHAIN_LOOP: None" in out and "No module named" in out,
       "A7b 后续裸名在受控面缺失 ⇒ 原生缺失错误（不回落 cwd）", out[:400])

    # A7-ii：显式根一次给全 v13 三件套 ⇒ 装配链完整可用（修法未破坏成功路径）
    root2 = os.path.join(base, "chainroot_full")
    os.makedirs(root2, exist_ok=True)
    with open(os.path.join(root2, "entity_registry.py"), "w", encoding="utf-8") as f:
        f.write("class EntityRegistry:\n    def __init__(self, store): self.store = store\n")
    for nm, sym in (("cognitive_orchestrator", "CognitiveOrchestrator"),
                    ("blindspot_learning_loop", "BlindSpotLearningLoop")):
        with open(os.path.join(root2, nm + ".py"), "w", encoding="utf-8") as f:
            f.write("class %s:\n    def __init__(self, *a, **k): pass\n" % sym)
    code2 = textwrap.dedent("""
        import os
        from lingshu.core.core import SpacetimeMemoryEngine
        e = SpacetimeMemoryEngine(":memory:")
        print("FULL_READY:", e._v13_ready, repr(e._v13_error))
        print("FULL_SLOTS:", e.entity_registry is not None,
              e._cognition is not None, e._learning_loop is not None)
        print("FULL_MARKER:", os.path.exists(os.path.join(os.getcwd(), "MARKER.txt")))
        """)
    r2 = run_py(code2, cwd=d, extra={"LINGSHU_COMPONENT_ROOT": root2})
    out2 = r2.stdout or ""
    ok(r2.returncode == 0 and "FULL_READY: True ''" in out2 and "FULL_SLOTS: True True True" in out2,
       "A7c 显式根给全 v13 三件套 ⇒ _v13_ready=True、三槽位齐备（成功路径未破坏）", out2[:400])
    ok("FULL_MARKER: False" in out2,
       "A7d 链完整装配时毒 cwd 仍不执行", out2[:400])


def group_a_call_time(base):
    """A8：调用期同类裸名（vision / pattern_separation / scene_reconstruction /
    game_web）——引擎运行期触发时同样不得从 cwd 解析执行。"""
    print("== A 组：调用期同类裸名——毒 cwd 在首次调用触发时也不执行 ==")
    d = os.path.join(base, "poison_calltime")
    os.makedirs(d, exist_ok=True)
    poison_files = {
        "vision.py": "def create_vision_provider():\n    return None\n"
                     "def perceive_image(*a, **k):\n    return {}\n",
        "pattern_separation.py": "class PatternSeparation:\n    def __init__(self, *a, **k): pass\n"
                                 "    def scan(self, limit=0): return {}\n",
        "scene_reconstruction.py": "class SceneReconstruction:\n    def __init__(self, *a, **k): pass\n"
                                   "    def reconstruct(self, *a, **k): return {}\n",
    }
    for fn, body in poison_files.items():
        with open(os.path.join(d, fn), "w", encoding="utf-8") as f:
            f.write(POISON_TEMPLATE.format(symbols=body))
    gw = os.path.join(d, "game_web")
    os.makedirs(gw, exist_ok=True)
    open(os.path.join(gw, "__init__.py"), "w").close()
    with open(os.path.join(gw, "generate.py"), "w", encoding="utf-8") as f:
        f.write(POISON_TEMPLATE.format(
            symbols="class WorldGenerator:\n    def __init__(self, *a, **k): pass\n"
                    "    def scene_from_text(self, t): return {}\n"))

    code = textwrap.dedent("""
        import glob, os
        from lingshu.core.core import SpacetimeMemoryEngine
        e = SpacetimeMemoryEngine(":memory:")
        r1 = e.perceive_image("no_such_file.png")
        r2 = e.pattern_separation_scan(limit=5)
        r3 = e.reconstruct_scene("测试线索", depth=1)
        r4 = e.world_generator("scene", {"text": "森林"})
        here = os.getcwd()
        print("CT_VISION:", r1.get("status"), "|", r2.get("error") is not None,
              "|", r3.get("error") is not None, "|", r4.get("status"))
        print("CT_MARKERS:", sorted(glob.glob(os.path.join(here, "MARKER*.txt"))),
              sorted(glob.glob(os.path.join(here, "game_web", "MARKER*.txt"))))
        """)
    r = run_py(code, cwd=d)
    out = r.stdout or ""
    ok(r.returncode == 0 and "CT_MARKERS: [] []" in out,
       "A8a 调用期四类毒文件（vision/pattern_separation/scene_reconstruction/game_web）均未被执行",
       out[:400])
    ok("CT_VISION: vision_unavailable" in out,
       "A8b perceive_image 触发 vision 裸名 ⇒ 未装配降级（不落 cwd）", out[:400])
    ok("| True | True | gen_not_ready" in out,
       "A8c pattern_separation/scene_reconstruction/game_web 触发 ⇒ 原生缺失降级", out[:400])


def group_a_all_at_once(base):
    print("== A 组：10 名毒文件同时在场 → 一次构造全部不执行 ==")
    d = os.path.join(base, "poison_all")
    os.makedirs(d, exist_ok=True)
    for name, _ in CONSTRUCT_TIME:
        with open(os.path.join(d, name + ".py"), "w", encoding="utf-8") as f:
            f.write(POISON_TEMPLATE.format(symbols=POISON_SYMBOLS[name]))
    code = ("import os\n"
            "from lingshu.core.core import SpacetimeMemoryEngine\n"
            "SpacetimeMemoryEngine(':memory:')\n"
            "import glob\n"
            "print('MARKERS:', sorted(glob.glob(os.path.join(os.getcwd(), 'MARKER*.txt'))))\n"
            "print('M_%s:', os.path.exists(os.path.join(os.getcwd(), 'MARKER.txt')))\n")
    r = run_py(code, cwd=d)
    ok(r.returncode == 0 and "M_True:" not in r.stdout,
       "A6 10 名毒文件同时在场：无一个被执行（MARKER.txt 不存在）",
       f"rc={r.returncode} out={r.stdout!r}")


# ---------------------------------------------------------------- B 组
def group_b_alert(base):
    print("== B 组：装配失败告警出口（warnings + self_check 字段） ==")
    clean = os.path.join(base, "clean_warn")
    os.makedirs(clean, exist_ok=True)

    # B1：-W always ⇒ stderr 有 ComponentDiscoveryWarning（此前完全静默）
    code1 = textwrap.dedent("""
        from lingshu.core.core import SpacetimeMemoryEngine
        SpacetimeMemoryEngine(":memory:")
        print("CONSTRUCTED")
        """)
    r = run_py(code1, cwd=clean, args=["-W", "always", "-c", code1])
    warn_txt = (r.stderr or "")
    n_warn = warn_txt.count("ComponentDiscoveryWarning")
    ok(r.returncode == 0 and "CONSTRUCTED" in r.stdout, "B1a -W always 构造成功 rc=0",
       f"rc={r.returncode} err={warn_txt[:200]!r}")
    ok(n_warn >= 8 and "[issue156]" in warn_txt,
       "B1b 告警通道可观测（stderr ≥8 条 ComponentDiscoveryWarning，含 [issue156]）",
       f"n_warn={n_warn}")
    ok("self_cognition_engine" in warn_txt and "entity_registry" in warn_txt,
       "B1c 告警文本点名具体组件（此前装配失败完全静默）")

    # B2：self_check()["component_assembly"] 覆盖构造期 10 名 + path_scrub 可观
    code2 = textwrap.dedent("""
        import json
        from lingshu.core.core import SpacetimeMemoryEngine
        e = SpacetimeMemoryEngine(":memory:")
        ca = e.self_check()["component_assembly"]
        d = ca["discovery"]
        print("##JSON##" + json.dumps({
            "failed": sorted(ca["failed"].keys()),
            "ok": ca["ok"], "lazy": ca["lazy"],
            "guard": d["guard_installed"],
            "cwd_excluded": d["cwd_excluded_count"],
            "names": d["component_names"],
            "path_scrub": d.get("path_scrub"),
        }, ensure_ascii=False))
        """)
    r = run_py(code2, cwd=clean)
    j = None
    for line in (r.stdout or "").splitlines():
        if line.startswith("##JSON##"):
            j = json.loads(line[len("##JSON##"):])
    ok(r.returncode == 0 and j is not None, "B2a self_check() 可取 component_assembly",
       f"rc={r.returncode} err={r.stderr[:200]!r}")
    if j:
        need = {n for n, _ in CONSTRUCT_TIME}
        ok(need <= set(j["failed"]),
           "B2b component_assembly.failed 覆盖全部 10 个构造期组件",
           f"missing={sorted(need - set(j['failed']))}")
        ok(j["guard"] is True, "B2c discovery.guard_installed=True（解析守卫在场）")
        ok(j["cwd_excluded"] >= 1, "B2d discovery 报告已排除 cwd/空串条目", j)
        ok(set(j["names"]) >= WHITELIST_EXPECTED,
           "B2e 白名单含全部已知 15 名", f"missing={sorted(WHITELIST_EXPECTED - set(j['names']))}")
        ps = j.get("path_scrub") or {}
        ok(ps.get("applied") is True and ps.get("removed_count", 0) >= 1
           and ps.get("escape_hatch_open") is False,
           "B2f discovery.path_scrub 可观测（applied + removed_count≥1 + 逃生口关）", ps)

    # B3：-W ignore（压掉告警）⇒ 构造不崩，self_check 通道仍完整（两路独立）
    r = run_py(code2, cwd=clean, args=["-W", "ignore", "-c", code2])
    j3 = None
    for line in (r.stdout or "").splitlines():
        if line.startswith("##JSON##"):
            j3 = json.loads(line[len("##JSON##"):])
    ok(r.returncode == 0 and j3 is not None and len(j3["failed"]) >= 10,
       "B3 告警被压制（-W ignore）时：构造不崩、self_check 通道仍报告全部失败（两路独立）",
       f"rc={r.returncode} failed={None if j3 is None else len(j3['failed'])}")


# ---------------------------------------------------------------- C 组
def group_c_install_paths(base):
    print("== C 组：装配成功路径未被修坏（显式根 / PYTHONPATH / sys.modules 别名 / 合法条目保留） ==")
    root = os.path.join(base, "comproot")
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "self_cognition_engine.py"), "w", encoding="utf-8") as f:
        f.write("class SelfCognitionEngine:\n    FAKE = True\n"
                "    def __init__(self, engine): self.engine = engine\n")
    with open(os.path.join(root, "body.py"), "w", encoding="utf-8") as f:
        f.write("def build_default_registry(workspace=\"\"):\n    return \"FAKE_BODY\"\n")
    with open(os.path.join(root, "semantic_space.py"), "w", encoding="utf-8") as f:
        f.write("class SemanticSpaceProvider:\n    FAKE = True\n"
                "    def __init__(self, *a, **k): pass\n")

    poison = make_poison_dir(base, "self_cognition_engine", tag="c1")  # 毒 cwd 仍在场

    code = textwrap.dedent("""
        import os
        from lingshu.core.core import SpacetimeMemoryEngine
        e = SpacetimeMemoryEngine(":memory:")
        print("SCE:", type(e._self_cognition).__name__, getattr(e._self_cognition, "FAKE", None),
              repr(e._self_cognition_error))
        print("BODY:", repr(e._body_registry), repr(e._body_error))
        print("SEM:", type(e._semantic_provider).__name__,
              getattr(e._semantic_provider, "FAKE", None), repr(e._semantic_error))
        print("POISON_MARKER:", os.path.exists(os.path.join(os.getcwd(), "MARKER.txt")))
        """)

    r = run_py(code, cwd=poison, extra={"LINGSHU_COMPONENT_ROOT": root})
    out = r.stdout or ""
    ok(r.returncode == 0 and "SCE: SelfCognitionEngine True ''" in out,
       "C1a 显式根（LINGSHU_COMPONENT_ROOT）：self_cognition_engine 正常装上", out[:400])
    ok("BODY: 'FAKE_BODY' ''" in out, "C1b 显式根：body 正常装上", out[:400])
    ok("SEM: SemanticSpaceProvider True ''" in out, "C1c 显式根：semantic_space 正常装上", out[:400])
    ok("POISON_MARKER: False" in out, "C1d 显式根在场时毒 cwd 依旧不执行", out[:400])

    # C2：PYTHONPATH（非 cwd）面（独立毒目录：避免 C1 残留 MARKER 干扰）
    poison2 = make_poison_dir(base, "self_cognition_engine", tag="c2")
    r2 = run_py("from lingshu.core.core import SpacetimeMemoryEngine\n"
                "e = SpacetimeMemoryEngine(':memory:')\n"
                "print('SCE2:', type(e._self_cognition).__name__, getattr(e._self_cognition, 'FAKE', None))\n",
                cwd=poison2, path=root + os.pathsep + REPO)
    ok(r2.returncode == 0 and "SCE2: SelfCognitionEngine True" in (r2.stdout or ""),
       "C2 PYTHONPATH（部署方显式面，非 cwd）：组件正常装上", (r2.stdout or "")[:300])

    # C3：sys.modules 别名（AEIS 私域 __init__ 命名空间注册形态）
    code3 = textwrap.dedent("""
        import os, sys, types
        m = types.ModuleType("self_cognition_engine")
        class SelfCognitionEngine:
            ALIAS = True
            def __init__(self, engine): self.engine = engine
        m.SelfCognitionEngine = SelfCognitionEngine
        sys.modules["self_cognition_engine"] = m
        from lingshu.core.core import SpacetimeMemoryEngine
        e = SpacetimeMemoryEngine(":memory:")
        print("ALIAS:", type(e._self_cognition).__name__,
              getattr(e._self_cognition, "ALIAS", None), repr(e._self_cognition_error))
        print("ALIAS_POISON_MARKER:", os.path.exists(os.path.join(os.getcwd(), "MARKER.txt")))
        """)
    # C3 用独立毒目录：其 marker 判据只反映本子进程是否执行了毒文件
    poison3 = make_poison_dir(base, "self_cognition_engine", tag="c3")
    r3 = run_py(code3, cwd=poison3)
    ok(r3.returncode == 0 and "ALIAS: SelfCognitionEngine True ''" in (r3.stdout or ""),
       "C3a sys.modules 别名（AEIS 命名空间注册形态）：命中别名，装配成功", (r3.stdout or "")[:300])
    ok("ALIAS_POISON_MARKER: False" in (r3.stdout or ""),
       "C3b 别名优先时毒 cwd 不执行", (r3.stdout or "")[:300])

    # C4：受控面提供「包 + 子模块」形态（game_web.generate）⇒ 随父包 __path__ 解析
    root3 = os.path.join(base, "comproot_gw")
    gw = os.path.join(root3, "game_web")
    os.makedirs(gw, exist_ok=True)
    with open(os.path.join(gw, "__init__.py"), "w", encoding="utf-8") as f:
        f.write("")
    with open(os.path.join(gw, "generate.py"), "w", encoding="utf-8") as f:
        f.write("class WorldGenerator:\n    def __init__(self, *a, **k): pass\n"
                "    def scene_from_text(self, t): return {\"scene\": \"FAKE\", \"text\": t}\n")
    code4 = textwrap.dedent("""
        from lingshu.core.core import SpacetimeMemoryEngine
        e = SpacetimeMemoryEngine(":memory:")
        r = e.world_generator("scene", {"text": "森林"})
        print("GW:", r.get("status"), (r.get("scene") or {}).get("scene"))
        """)
    poison4 = make_poison_dir(base, "self_cognition_engine", tag="c4")
    r4 = run_py(code4, cwd=poison4, extra={"LINGSHU_COMPONENT_ROOT": root3})
    ok(r4.returncode == 0 and "GW: ok FAKE" in (r4.stdout or ""),
       "C4 显式根提供 game_web 包（含子模块 generate）⇒ world_generator 正常装上",
       (r4.stdout or "")[:300])

    # C5：非 cwd 合法条目（含 ghost .zip / egg 路径）保留——至少不误删
    clean = os.path.join(base, "clean_keep")
    os.makedirs(clean, exist_ok=True)
    ghosts = [os.path.join(base, "ghost.zip"), os.path.join(base, "ghost.egg"),
              os.path.join(base, "ns_root")]
    code5 = textwrap.dedent("""
        import os, sys
        from lingshu.core.core import SpacetimeMemoryEngine
        SpacetimeMemoryEngine(":memory:")
        ghosts = %r
        kept = [g for g in ghosts if any(str(e).lower() == g.lower() for e in sys.path)]
        repo_kept = any(isinstance(e, str) and os.path.abspath(e) == os.path.abspath(%r)
                        for e in sys.path)
        print("GHOSTS_KEPT:", len(kept), "REPO_KEPT:", repo_kept)
        """) % (ghosts, REPO)
    r5 = run_py(code5, cwd=clean, path=REPO + os.pathsep + os.pathsep.join(ghosts))
    out5 = r5.stdout or ""
    ok(r5.returncode == 0 and "GHOSTS_KEPT: 3" in out5 and "REPO_KEPT: True" in out5,
       "C5 非 cwd 合法条目（PYTHONPATH 的 ghost .zip/.egg/ns 目录 + 仓根）未被误删",
       out5[:300])


# ---------------------------------------------------------------- D 组（AST · 有牙）
def _classify_top(top):
    if top in COMPONENT_NAMES:
        return "component"
    if top in STDLIB_ALLOWED:
        return "stdlib"
    if top in THIRD_PARTY_ALLOWED:
        return "third_party"
    if top in SELF_ALLOWED:
        return "self"
    return "unknown"


def scan_naked_imports(pkg_dir):
    """AST 扫描 <pkg_dir> 下全部 `.py` 的**裸名** import（不含相对导入）。

    返回 {top: [(abs_path, lineno, modname), ...]}——**不过滤**，含未知名。
    """
    found = {}
    for root, dirs, files in os.walk(pkg_dir):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fn in files:
            if not fn.endswith(".py"):
                continue
            p = os.path.join(root, fn)
            try:
                tree = ast.parse(open(p, encoding="utf-8").read())
            except Exception as exc:  # noqa: BLE001
                found.setdefault("__PARSE_ERROR__", []).append((p, 0, str(exc)))
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for a in node.names:
                        top = a.name.split(".")[0]
                        found.setdefault(top, []).append((p, node.lineno, a.name))
                elif isinstance(node, ast.ImportFrom):
                    if node.level and node.level > 0:
                        continue
                    top = (node.module or "").split(".")[0]
                    if top:
                        found.setdefault(top, []).append((p, node.lineno, node.module))
    return found


def _classify_report(pkg_dir):
    """对 <pkg_dir> 做「标准库/组件/已知第三方/未知」分类；返回 (unknown, all_tops, hits)。"""
    found = scan_naked_imports(pkg_dir)
    unknown = {}
    hits = 0
    for top, sites in found.items():
        hits += len(sites)
        if _classify_top(top) == "unknown":
            unknown[top] = sites
    return unknown, set(found), hits


def group_d_whitelist():
    print("== D 组：全仓裸名分类（有牙）——标准库/组件/已知第三方/未知 ==")
    ok(STDLIB_ALLOWED <= set(sys.stdlib_module_names),
       "D0a 标准库白名单 ⊆ sys.stdlib_module_names（解释器权威清单校验）",
       f"非法={sorted(STDLIB_ALLOWED - set(sys.stdlib_module_names))}")
    unknown, all_tops, hits = _classify_report(os.path.join(REPO, "lingshu"))
    ok(WHITELIST_EXPECTED <= all_tops and hits >= 20,
       "D1 全仓裸名扫描：已知 15 名全部出现（≥20 处引用）",
       f"found={sorted(all_tops)} hits={hits}")
    ok(not unknown,
       "D2 未知裸名 = ∅（出现未知即红：须登记为组件白名单/标准库/已知第三方）",
       "未知=" + json.dumps({k: [os.path.relpath(s[0], REPO) + ":" + str(s[1])
                                for s in v] for k, v in unknown.items()},
                              ensure_ascii=False))
    ok(set(COMPONENT_NAMES) >= WHITELIST_EXPECTED,
       "D3 白名单未缩水（≥已知 15 名）",
       f"missing={sorted(WHITELIST_EXPECTED - set(COMPONENT_NAMES))}")

    # D4：**自证有牙**——沙箱副本注入新裸名 ⇒ 判据必检为「未知」；干净副本 ⇒ 空
    sandbox = tempfile.mkdtemp(prefix="f156_d_sandbox_")
    try:
        copy_root = os.path.join(sandbox, "lingshu")
        shutil.copytree(os.path.join(REPO, "lingshu"), copy_root,
                        ignore=shutil.ignore_patterns("__pycache__"))
        u_clean, _, _ = _classify_report(copy_root)
        ok(not u_clean, "D4a 干净副本：未知裸名 = ∅（基线）", f"unknown={sorted(u_clean)}")
        probe = os.path.join(copy_root, "_inject_probe_156.py")
        with open(probe, "w", encoding="utf-8") as f:
            f.write("# 沙箱变异：注入新裸名（守卫必须检为「未知」）\n"
                    "import totally_new_component_xyz_156\n")
        u_mut, _, _ = _classify_report(copy_root)
        ok("totally_new_component_xyz_156" in u_mut,
           "D4b **有牙自证**：注入新裸名 totally_new_component_xyz_156 ⇒ 判据检为「未知」（必红）",
           f"unknown={sorted(u_mut)}")
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)


# ---------------------------------------------------------------- F 组（标准库劫持面）
def group_f_stdlib_hijack(base):
    print("== F 组：标准库劫持面（v2 核心）——逐名毒 cwd 在 -c 形态下不执行 ==")
    ct = ("from lingshu.core.core import SpacetimeMemoryEngine\n"
          "SpacetimeMemoryEngine(':memory:')\n"
          "print('CONSTRUCTED')\n")
    n_exec = 0
    for name in STDLIB_HIJACK_NAMES:
        d = make_stdlib_poison_dir(base, name, tag="c")
        r = run_py(ct, cwd=d)
        marker = os.path.exists(os.path.join(d, "MARKER.txt"))
        n_exec += 1 if marker else 0
        ok(r.returncode == 0 and not marker and "CONSTRUCTED" in (r.stdout or ""),
           f"F1[{name}] -c 形态：cwd 毒 {name}.py 未执行、引擎照常装配 rc=0",
           f"rc={r.returncode} marker={marker} out={(r.stdout or '')[:120]!r} err={(r.stderr or '')[:120]!r}")
    ok(n_exec == 0, f"F1 汇总：{len(STDLIB_HIJACK_NAMES)} 个标准库毒名全部未执行（executed=0/%d）"
       % len(STDLIB_HIJACK_NAMES), f"n_exec={n_exec}")

    # F2：`-m` 形态（sys.path[0] = cwd **绝对路径**）—— 证「等于 cwd 的绝对条目」亦被消除
    runner_root = os.path.join(base, "runner_mod")
    pkg = os.path.join(runner_root, "f156mod")
    os.makedirs(pkg, exist_ok=True)
    with open(os.path.join(pkg, "__init__.py"), "w", encoding="utf-8") as f:
        f.write("")
    with open(os.path.join(pkg, "__main__.py"), "w", encoding="utf-8") as f:
        f.write(ct)
    d = make_stdlib_poison_dir(base, "json", tag="m")
    r = run_py("", cwd=d, path=REPO + os.pathsep + runner_root, args=["-m", "f156mod"])
    marker = os.path.exists(os.path.join(d, "MARKER.txt"))
    ok(r.returncode == 0 and not marker and "CONSTRUCTED" in (r.stdout or ""),
       "F2 -m 形态（sys.path[0]=cwd 绝对路径）：cwd 毒 json.py 未执行、引擎照常装配",
       f"rc={r.returncode} marker={marker} out={(r.stdout or '')[:160]!r} err={(r.stderr or '')[:160]!r}")


# ---------------------------------------------------------------- G 组（逃生口）
def group_g_escape_hatch(base):
    print("== G 组：逃生口 LINGSHU_ALLOW_CWD_IMPORTS（默认关；开＝放弃本层保护） ==")
    clean = os.path.join(base, "clean_hatch")
    os.makedirs(clean, exist_ok=True)
    code = textwrap.dedent("""
        import json, os, sys
        import lingshu._pathguard as pg
        from lingshu.core.core import SpacetimeMemoryEngine
        SpacetimeMemoryEngine(":memory:")
        print("##JSON##" + json.dumps({
            "empty_in_path": ("" in sys.path),
            "scrub": pg.pathguard_report(),
        }, ensure_ascii=False))
        """)

    # G1：默认（逃生口关）⇒ '' 被消除、状态标 escape_hatch_open=False
    r = run_py(code, cwd=clean)
    j = None
    for line in (r.stdout or "").splitlines():
        if line.startswith("##JSON##"):
            j = json.loads(line[len("##JSON##"):])
    ok(r.returncode == 0 and j is not None, "G1a 逃生口关：子进程取到状态", (r.stdout or "")[:160])
    if j:
        ok(j["empty_in_path"] is False, "G1b 逃生口关：'' 已从 sys.path 消除")
        s = j["scrub"]
        ok(s["escape_hatch_open"] is False and s["cwd_scrub_active"] is True
           and s["removed_count"] >= 1 and s["cwd_entries_now"] == 0,
           "G1c 逃生口关：状态标 cwd_scrub_active=True / removed_count≥1 / cwd_entries_now=0", s)
        ok(all(":" not in k for k in s["removed_kinds"]),
           "G1d removed_kinds 不含本机绝对路径（只记种类）", s["removed_kinds"])

    # G2：逃生口开（LINGSHU_ALLOW_CWD_IMPORTS=1）⇒ 保留 cwd 条目 + 告警 + 状态可观测
    r = run_py(code, cwd=clean, extra={"LINGSHU_ALLOW_CWD_IMPORTS": "1"})
    j = None
    for line in (r.stdout or "").splitlines():
        if line.startswith("##JSON##"):
            j = json.loads(line[len("##JSON##"):])
    ok(r.returncode == 0 and j is not None, "G2a 逃生口开：子进程取到状态", (r.stdout or "")[:160])
    if j:
        s = j["scrub"]
        ok(j["empty_in_path"] is True,
           "G2b 逃生口开：'' 被**保留**（本层保护关闭）")
        ok(s["escape_hatch_open"] is True and s["escape_hatch_env"] == "LINGSHU_ALLOW_CWD_IMPORTS"
           and s["cwd_entries_now"] >= 1 and len(s["kept_cwd_kinds"]) >= 1,
           "G2c 逃生口开：状态标 escape_hatch_open=True / kept_cwd_kinds 非空", s)
    ok("CwdImportsAllowed" in (r.stderr or "") and "[issue156-v2]" in (r.stderr or ""),
       "G2d 逃生口开：stderr 有明确告警（CwdImportsAllowed + [issue156-v2]）",
       (r.stderr or "")[:200])

    # G3：逃生口开 + cwd 毒 json.py ⇒ MARKER 出现（记录性断言：开＝放弃保护）
    d = make_stdlib_poison_dir(base, "json", tag="hatch")
    r = run_py("from lingshu.core.core import SpacetimeMemoryEngine; SpacetimeMemoryEngine(':memory:')",
               cwd=d, extra={"LINGSHU_ALLOW_CWD_IMPORTS": "1"})
    marker = os.path.exists(os.path.join(d, "MARKER.txt"))
    ok(marker, "G3 逃生口开 + cwd 毒 json.py ⇒ MARKER 出现（如实记录：开＝放弃本层保护）",
       f"marker={marker} rc={r.returncode}")


# ---------------------------------------------------------------- S 组（脚本目录边界）
def group_s_script_dir(base):
    print("== S 组：脚本目录边界——`python <script>.py` 且脚本目录 != cwd ==")
    A = os.path.join(base, "script_dir_A")
    B = os.path.join(base, "cwd_B")
    os.makedirs(A, exist_ok=True)
    os.makedirs(B, exist_ok=True)
    with open(os.path.join(A, "self_cognition_engine.py"), "w", encoding="utf-8") as f:
        f.write(POISON_TEMPLATE.format(symbols=POISON_SYMBOLS["self_cognition_engine"]))
    with open(os.path.join(A, "run_probe.py"), "w", encoding="utf-8") as f:
        f.write(textwrap.dedent("""
            import os
            from lingshu.core.core import SpacetimeMemoryEngine
            e = SpacetimeMemoryEngine(":memory:")
            here = os.path.dirname(os.path.abspath(__file__))
            print("SCE_SLOT:", e._self_cognition, "ERR:", repr(e._self_cognition_error))
            print("SCRIPTDIR_MARKER:", os.path.exists(os.path.join(here, "MARKER.txt")))
            """))
    r = run_py("", cwd=B, args=[os.path.join(A, "run_probe.py")])
    out = r.stdout or ""
    ok(r.returncode == 0 and "SCRIPTDIR_MARKER: False" in out,
       "S1 脚本目录（!= cwd）里的白名单毒组件未被执行（v1 漏点）", out[:300])
    ok("SCE_SLOT: None" in out and "No module named" in out,
       "S2 脚本目录不被当作受控面 ⇒ 白名单名原生缺失（不回落脚本目录）", out[:300])


# ---------------------------------------------------------------- E 组
def group_e_disclosure():
    print("== E 组：边界诚实——密钥声明收窄到事实（v2 精确措辞） ==")
    from lingshu.core.core import verify_designer
    doc = verify_designer.__doc__ or ""
    ok("同进程" in doc, "E1 docstring 点明「同进程」可读（原「永远无法读取」不成立）")
    ok("不成立" in doc, "E2 docstring 显式给出「原表述不成立」的边界", doc[:160])
    ok("设计级" in doc and "sys.path" in doc and "消除" in doc,
       "E3 docstring 精确表述本批范围（消除 cwd/空串 sys.path 解析）且同进程属设计级")
    ok("逃生口" in doc, "E4 docstring 明示仍不覆盖的情形（逃生口被显式打开）")


def main():
    base = tempfile.mkdtemp(prefix="f156_guard_")
    print(f"# guard temp base: {base}")
    try:
        group_a_bash_form(base)
        group_a_pytest_form(base)
        group_a_chain(base)
        group_a_call_time(base)
        group_a_all_at_once(base)
        group_b_alert(base)
        group_c_install_paths(base)
        group_d_whitelist()
        group_f_stdlib_hijack(base)
        group_g_escape_hatch(base)
        group_s_script_dir(base)
        group_e_disclosure()
    finally:
        shutil.rmtree(base, ignore_errors=True)
    print()
    print(f"===== SUMMARY {len(_PASS)}/{len(_PASS) + len(_FAIL)} 通过 =====")
    if _FAIL:
        print("FAILS:")
        for m in _FAIL:
            print("  -", m)
        return 1
    print("VERDICT=PASS（issue #156 v2 守卫：cwd/空串 sys.path 解析面已消除——白名单组件"
          "（逐名 × 两形态）+ 标准库（逐名）毒 cwd 均不执行；脚本目录边界亦收口；"
          "AST 分类判据有牙（未知即红）；逃生口与 path_scrub 状态可观测；"
          "显式根/PYTHONPATH/别名/非 cwd 合法条目四条装配路径未被修坏）")
    return 0


def test_issue156_component_discovery():
    """pytest 入口：与脚本式 main() 同一套断言。"""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
