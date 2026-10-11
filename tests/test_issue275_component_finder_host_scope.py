# -*- coding: utf-8 -*-
"""test_issue275_component_finder_host_scope · 组件 finder 不得接管宿主的同名顶层模块
============================================================================
背景（lingshu issue #275 · #156 v2 的回归）：
  `core.py` 在模块导入时执行 `component_resolver.install()`，向 `sys.meta_path[0]`
  插入 `_ComponentFinder`。该 finder 对白名单 15 个**通用顶层名**（`vision` /
  `body` / `game_web` / `semantic_space` …）生效，**不区分调用方**，受控面找不到
  即 `raise ModuleNotFoundError`、**不交还**后续 finder。于是宿主项目里同名的
  自有模块，只要在 `import lingshu.core` 之后导入，一律 `ModuleNotFoundError`；
  逃生口 `LINGSHU_ALLOW_CWD_IMPORTS=1` 也无效（排除 cwd/脚本目录是**本 finder**
  自身的行为，与 `_pathguard` 无关）。

  缺陷形态在 HEAD 实测（issue 正文复现脚本 [A]/[B]）：`vision` / `body` /
  `game_web` / `semantic_space` 四行**全部** `ModuleNotFoundError`。

修复：`_ComponentFinder.find_spec` 先判「本次导入是否由 lingshu 包内代码发起」
  （`_importer_is_lingshu()`：沿调用栈取第一个有归属的帧，`lingshu` 包内 ⇒ True）。
  · 包内（引擎自己的组件导入）⇒ 维持 #156 的严格收口（受控面缺失即原生
    `ModuleNotFoundError` + 告警）；
  · 宿主 / `__main__` / 其它库 ⇒ `return None` 交还后续 finder，由标准
    `PathFinder` 按 `sys.path` 解析宿主自己的模块。

断言组（回退 / 放宽即红）：
  G1 复现脚本 [A]：宿主 app 目录放自有 `vision.py`/`body.py`/`game_web.py`/
     `semantic_space.py`，脚本先 `import lingshu.core.core` 再逐个 `__import__`
     ⇒ 四行全 `ok`，且取到的模块 `OWNER == 'host'`（是宿主自己的模块）。
  G2 同上 + `LINGSHU_ALLOW_CWD_IMPORTS=1`（逃生口）⇒ 四行全 `ok`。
  G3 反向腿（防过度放松 · 钉 #156 收口仍在）：毒 cwd 里放白名单组件
     `self_cognition_engine.py`，从该 cwd 构造引擎 ⇒ 毒文件**不执行**
     （MARKER 不存在）且 `_self_cognition_error` 含 `No module named`
     ——证明本修复只把「宿主发起的同名导入」交还，没有拆掉引擎侧的组件收口。
  G4 引擎仍能从**显式根**装配组件（`LINGSHU_COMPONENT_ROOT`，成功路径未坏）。

判据来源（逐条写清，不编造）：
  · 判据 = lingshu issue #275 正文「复现」的 [A]/[B] 两段脚本，及「判据」节：
    「复现脚本 [A]、[B] 中 8 行全部为 `ok`；`tests/test_issue156_component_discovery.py`
    中『cwd / 脚本目录里的同名毒组件不会在引擎构造期执行』的断言仍然通过」
    （源文档 `.tmp/bodies_rest/275.md`，本仓外，属编排侧给的线索）。
  · 「宿主自有模块必须可导入」的理论依据：Python 导入语义——顶层名解析由
    `sys.meta_path` 链共同决定，任一 finder `return None` 即交还后续 finder；
    lingshu 的组件白名单是**本引擎的内部约定**，不应对外部进程施加。
  · 本件**未改**受控面判定（`safe_search_path` / `_is_cwd_entry` /
    `_is_script_dir_entry` / `_is_relative_entry` 一行未动），只改「找不到时
    raise 还是 return None」。

不适用条件 / 已知边界：
  · 本守卫只钉「宿主发起的同名顶层导入不被接管」；「宿主把 lingshu 当作组件根
    时引擎会把宿主同名模块当组件加载」（#275 正文第 3 点「撞名挡不住」）不在
    本件范围。
  · 调用栈不可用时 `_importer_is_lingshu()` 保守返回 True（维持 #156 收口）。

定点变异自证（抽掉修复 ⇒ 必红）：
  删 `find_spec` 里的 `if not _importer_is_lingshu(): return None` 两行 ⇒
  G1/G2 报四行 `ModuleNotFoundError`（逐字读数见报告）。

运行（仓根）：python -X utf8 tests/test_issue275_component_finder_host_scope.py
             / python -X utf8 -m pytest tests/test_issue275_component_finder_host_scope.py
退出码：0 = 全过；1 = 有断言失败
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

HOST_NAMES = ("vision", "body", "game_web", "semantic_space")

# 宿主 app 脚本（与 #275 正文复现脚本逐字同构，另打印模块归属以钉「是宿主自己的」）
_HOST_MAIN = (
    "import sys, lingshu.core.core\n"
    "for m in (%r, %r, %r, %r):\n"
    "    try:\n"
    "        mod = __import__(m)\n"
    "        print('   import %%-15s -> ok owner=%%s' %% (m, getattr(mod, 'OWNER', None)))\n"
    "    except ModuleNotFoundError:\n"
    "        print('   import %%-15s -> ModuleNotFoundError' %% m)\n"
) % HOST_NAMES


def _mk_host_app():
    """宿主项目：app 目录里有自己的同名模块 + main.py。"""
    app = tempfile.mkdtemp(prefix="f275_host_")
    for name in HOST_NAMES:
        with open(os.path.join(app, name + ".py"), "w", encoding="utf-8") as f:
            f.write("OWNER = 'host'\n")
    main = os.path.join(app, "main.py")
    with open(main, "w", encoding="utf-8") as f:
        f.write(_HOST_MAIN)
    return app, main


def _run_host(main, extra_env=None):
    env = {k: v for k, v in os.environ.items()
           if k not in ("LINGSHU_ALLOW_CWD_IMPORTS", "LINGSHU_COMPONENT_ROOT")}
    env["PYTHONPATH"] = REPO
    env["PYTHONUTF8"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if extra_env:
        env.update(extra_env)
    return subprocess.run([sys.executable, "-B", "-W", "ignore", main],
                          cwd=tempfile.gettempdir(), env=env,
                          capture_output=True, text=True, encoding="utf-8")


def _assert_all_ok(out, label):
    for name in HOST_NAMES:
        line = "import %-15s -> ok owner=host" % name
        assert line in out, f"{label}: 缺 `{line}`\n实际输出：\n{out}"
    assert "ModuleNotFoundError" not in out, (
        f"{label}: 宿主自有同名模块被组件 finder 接管（应全部可导入）\n{out}")


def test_g1_host_own_modules_importable_after_lingshu():
    """G1（#275 [A]）：先 import lingshu.core 后，宿主自有同名模块仍可导入。"""
    app, main = _mk_host_app()
    r = _run_host(main)
    assert r.returncode == 0, f"宿主脚本 rc={r.returncode}\nstderr={r.stderr[-400:]}"
    _assert_all_ok(r.stdout, "[A]")


def test_g2_escape_hatch_also_ok():
    """G2（#275 [B]）：逃生口打开时同样全 ok（缺陷形态逃生口无效）。"""
    app, main = _mk_host_app()
    r = _run_host(main, {"LINGSHU_ALLOW_CWD_IMPORTS": "1"})
    assert r.returncode == 0, f"宿主脚本 rc={r.returncode}\nstderr={r.stderr[-400:]}"
    _assert_all_ok(r.stdout, "[B]")


def test_g3_engine_side_strictness_intact():
    """G3 反向腿：引擎自己的组件导入仍走受控面（毒 cwd 组件不执行）。"""
    d = tempfile.mkdtemp(prefix="f275_poison_")
    with open(os.path.join(d, "self_cognition_engine.py"), "w", encoding="utf-8") as f:
        f.write("import os as _os\n"
                "open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),"
                " 'MARKER.txt'), 'w').write('EXECUTED')\n"
                "class SelfCognitionEngine:\n    def __init__(self, *a, **k): pass\n")
    code = ("from lingshu.core.core import SpacetimeMemoryEngine\n"
            "e = SpacetimeMemoryEngine(':memory:')\n"
            "print('ERR:', repr(e._self_cognition_error))\n")
    env = {k: v for k, v in os.environ.items()
           if k not in ("LINGSHU_ALLOW_CWD_IMPORTS", "LINGSHU_COMPONENT_ROOT")}
    env["PYTHONPATH"] = REPO
    env["PYTHONUTF8"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    r = subprocess.run([sys.executable, "-B", "-W", "ignore", "-c", code], cwd=d,
                       env=env, capture_output=True, text=True, encoding="utf-8")
    marker = os.path.exists(os.path.join(d, "MARKER.txt"))
    assert r.returncode == 0, f"rc={r.returncode} err={r.stderr[-400:]}"
    assert not marker, "毒 cwd 组件被执行——#156 收口被拆（过度放松）"
    assert "No module named" in r.stdout, (
        f"引擎侧组件收口失效：_self_cognition_error={r.stdout!r}")


def test_g4_explicit_root_still_assembles():
    """G4：显式根（LINGSHU_COMPONENT_ROOT）装配路径未被修坏。"""
    root = tempfile.mkdtemp(prefix="f275_root_")
    with open(os.path.join(root, "self_cognition_engine.py"), "w", encoding="utf-8") as f:
        f.write("class SelfCognitionEngine:\n    FAKE = True\n"
                "    def __init__(self, engine): self.engine = engine\n")
    code = ("from lingshu.core.core import SpacetimeMemoryEngine\n"
            "e = SpacetimeMemoryEngine(':memory:')\n"
            "print('SCE:', type(e._self_cognition).__name__,"
            " getattr(e._self_cognition, 'FAKE', None), repr(e._self_cognition_error))\n")
    clean = tempfile.mkdtemp(prefix="f275_clean_")
    env = {k: v for k, v in os.environ.items()
           if k not in ("LINGSHU_ALLOW_CWD_IMPORTS", "LINGSHU_COMPONENT_ROOT")}
    env["PYTHONPATH"] = REPO
    env["PYTHONUTF8"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["LINGSHU_COMPONENT_ROOT"] = root
    r = subprocess.run([sys.executable, "-B", "-W", "ignore", "-c", code], cwd=clean,
                       env=env, capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, f"rc={r.returncode} err={r.stderr[-400:]}"
    assert "SCE: SelfCognitionEngine True ''" in r.stdout, (
        f"显式根装配路径被修坏：{r.stdout!r}")


if __name__ == "__main__":
    test_g1_host_own_modules_importable_after_lingshu()
    test_g2_escape_hatch_also_ok()
    test_g3_engine_side_strictness_intact()
    test_g4_explicit_root_still_assembles()
    print("OK #275 守卫全过")
