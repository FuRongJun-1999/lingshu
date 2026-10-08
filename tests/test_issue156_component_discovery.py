# -*- coding: utf-8 -*-
"""test_issue156_component_discovery · lingshu issue #156 守卫
============================================================================
缺陷（lingshu issue #156 · 安全：组件发现面）：
  引擎按**裸模块名**解析一批可选组件（`from entity_registry import ...` 等，
  构造期即刻触发 10 处）；当 sys.modules 无别名时，`python -c` / `python -m` /
  REPL / pytest 形态下 `sys.path[0]` 是**当前工作目录**（空串 `''` 或 cwd）
  ⇒ 在工作目录放同名 `.py` 即可在**构造期**进程内执行任意代码：拿到活引擎、
  读 `AEIS_DESIGNER_KEY`（同进程可读）、伪造 D-007 终裁、`set_escalation_enabled`
  关停全部升级点。且装配失败**完全静默**（core.py 无 print/logging/warnings、
  构造期 stderr 为空，仅有事后 `*_error` 字符串）。
  （`python xxx.py` 形态 sys.path[0] 是脚本目录，不在本守卫断言面内。）

修法（本批）：
  ① 组件解析收口（`lingshu/core/component_resolver.py`）：白名单名字只从
     sys.modules 别名 / 显式根 `LINGSHU_COMPONENT_ROOT` / sys.path 非 cwd 条目
     解析；受控面缺失即抛原生同文本 ModuleNotFoundError（不回落到 cwd），
     且不再依赖 cwd/空串条目。
  ② 装配失败告警出口：ComponentDiscoveryWarning（stderr，标准库）+ 引擎
     `self_check()["component_assembly"]`（只增字段）；既有 `*_error` 字段
     文本逐字不变。
  ③ core.py 顶部密钥声明收窄到事实（同进程可读，属设计级事项）。

断言组（全部走子进程，隔离环境；毒目录一律 tempfile.mkdtemp）：
  A 组（注入面关闭）：对**每个**构造期裸名，毒目录放同名 .py（顶层写 MARKER）
     ⇒ ①`-c` 形态构造引擎：MARKER 不得出现、该名字不得进 sys.modules、
     对应 *_error 非空；②pytest 形态（毒目录内含 test_ 文件）构造引擎：
     pytest 不得因 MARKER 出现而失败（即毒文件未执行）；③10 名毒文件同时在场
     ⇒ 一次构造，全部 MARKER 不出现。
  B 组（告警出口）：干净环境构造 ⇒ ①`-W always` 时 stderr 出现
     ComponentDiscoveryWarning 文本（≥8 条，含构造期组件名）；②self_check()
     的 component_assembly.failed 覆盖 10 个构造期名字且 guard_installed=True；
     ③`-W ignore`（告警被压掉）下构造不崩、failed 字段仍完整（两路独立）。
  C 组（装配成功路径没被修坏）：①显式根 LINGSHU_COMPONENT_ROOT 放假组件
     （self_cognition_engine/body/semantic_space）⇒ 正常装上且毒 cwd 仍不执行；
     ②PYTHONPATH（非 cwd）放组件 ⇒ 装上；③sys.modules 别名（AEIS 私域
     `aeis/__init__.py` 命名空间注册形态）⇒ 命中别名、cwd 毒文件不执行。
  D 组（白名单完整性 · 回归护栏）：AST 扫描全仓裸名导入 ⇒ 顶层名集合
     ⊆ COMPONENT_NAMES，且包含已知 15 名（新增裸名须同步登记白名单）。
  E 组（边界诚实）：verify_designer docstring 不再声称「永远无法读取」，
     明确点出「同进程」可读与「本批只封注入面」。

运行（lingshu 仓根）：python -X utf8 tests/test_issue156_component_discovery.py
退出码：0 = 全过；1 = 有断言失败
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import textwrap

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lingshu.core.component_resolver import COMPONENT_NAMES  # noqa: E402

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

    # B2：self_check()["component_assembly"] 覆盖构造期 10 名
    code2 = textwrap.dedent("""
        import json
        from lingshu.core.core import SpacetimeMemoryEngine
        e = SpacetimeMemoryEngine(":memory:")
        ca = e.self_check()["component_assembly"]
        print("##JSON##" + json.dumps({
            "failed": sorted(ca["failed"].keys()),
            "ok": ca["ok"], "lazy": ca["lazy"],
            "guard": ca["discovery"]["guard_installed"],
            "cwd_excluded": ca["discovery"]["cwd_excluded_count"],
            "names": ca["discovery"]["component_names"],
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
    print("== C 组：装配成功路径未被修坏（显式根 / PYTHONPATH / sys.modules 别名） ==")
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


# ---------------------------------------------------------------- D 组
def group_d_whitelist():
    print("== D 组：白名单完整性——全仓裸名 ⊆ COMPONENT_NAMES ==")
    found = set()
    hits = []
    for root, dirs, files in os.walk(os.path.join(REPO, "lingshu")):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fn in files:
            if not fn.endswith(".py"):
                continue
            p = os.path.join(root, fn)
            try:
                tree = ast.parse(open(p, encoding="utf-8").read())
            except Exception as exc:
                ok(False, f"D0 解析失败 {fn}", exc)
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for a in node.names:
                        top = a.name.split(".")[0]
                        if top in WHITELIST_EXPECTED:
                            found.add(top)
                            hits.append((p, node.lineno, a.name))
                elif isinstance(node, ast.ImportFrom):
                    if node.level and node.level > 0:
                        continue
                    top = (node.module or "").split(".")[0]
                    if top in WHITELIST_EXPECTED:
                        found.add(top)
                        hits.append((p, node.lineno, node.module))
    ok(WHITELIST_EXPECTED <= found and len(hits) >= 20,
       "D1 全仓裸名扫描：已知 15 名全部出现（≥20 处引用）",
       f"found={sorted(found)} hits={len(hits)}")
    ok(found <= set(COMPONENT_NAMES),
       "D2 扫描结果 ⊆ COMPONENT_NAMES（新增裸名须登记白名单）",
       f"extra={sorted(found - set(COMPONENT_NAMES))}")
    ok(set(COMPONENT_NAMES) >= WHITELIST_EXPECTED,
       "D3 白名单未缩水（≥已知 15 名）",
       f"missing={sorted(WHITELIST_EXPECTED - set(COMPONENT_NAMES))}")


# ---------------------------------------------------------------- E 组
def group_e_disclosure():
    print("== E 组：边界诚实——密钥声明收窄到事实 ==")
    from lingshu.core.core import verify_designer
    doc = verify_designer.__doc__ or ""
    # 反面判据：原 docstring（“密钥仅存在于服务环境变量…永远无法冒充”）
    # 不含下列任一词 ⇒ 若回退到旧表述，本组必红（见下方“原表述不成立”的否定语）。
    ok("同进程" in doc, "E1 docstring 点明「同进程」可读（原「永远无法读取」不成立）")
    ok("不成立" in doc, "E2 docstring 显式给出「原表述不成立」的边界", doc[:160])
    ok("设计级" in doc and "注入面" in doc,
       "E3 docstring 注明本批只封注入面、同进程可信性属设计级事项")


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
        group_e_disclosure()
    finally:
        import shutil
        shutil.rmtree(base, ignore_errors=True)
    print()
    print(f"===== SUMMARY {len(_PASS)}/{len(_PASS) + len(_FAIL)} 通过 =====")
    if _FAIL:
        print("FAILS:")
        for m in _FAIL:
            print("  -", m)
        return 1
    print("VERDICT=PASS（issue #156 守卫：cwd 同名组件在构造期不执行（逐名 × 两形态）；"
          "装配失败有可观测告警出口；显式根/PYTHONPATH/别名三条装配路径未被修坏）")
    return 0


def test_issue156_component_discovery():
    """pytest 入口：与脚本式 main() 同一套断言。"""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
