# -*- coding: utf-8 -*-
"""test_issue10_intake_registration_guard · 两份登记一致 + 脱敏路径操作数行为级校验（#10）
====================================================================================
缺陷（分诊表 `p2_groups.json` → `docs/intake/body-export-v0.1/_export_manifest.json`
行 #10，判定成立）分两半：

  ① **两份登记不一致**：v0.1 导出登记有两处——机器生成的
     `docs/intake/body-export-v0.1/_export_manifest.json`（core.py 条目
     `externals`＝11 项，含 `body`/`pattern_separation`；`hidden_refs` 另含
     `scene_reconstruction`）与人工撰写的同目录 `README.md` §三（自称列
     「10 个未实现扩展点名义」，含 `scene_reconstruction` 却**漏列**
     `body`/`pattern_separation`）。两侧名表互不相等 ⇒ 读者按 README 会以为
     `body`/`pattern_separation` 不是未装即降级的扩展点。

  ② **脱敏把路径操作数变成字面量后静默失效**：导出脱敏把
     `lingshu/nn/rust_bridge.py` 探测链里两个**本机字面量候选路径**改写成中性
     占位符 `<local-path>`；但它们是**路径操作数**（进 `Path(c).is_file()`），
     脱敏后两候选恒 False、探测链静默失效。清单里的 `sanitized: 4` 只是计数，
     **没有任何行为级校验**去发现「脱敏把代码改坏了」。

判据来源：
  · ① 三方同源：`core.py` 的 AST 裸名导入面（实测 14 个根名）＝
    `_export_manifest.json` core.py 条目 `externals` ∪ `hidden_refs` 的根名集合
    ＝ README §三 点名的组件名集合；组件名白名单取本仓自有真源
    `lingshu/core/component_resolver.py::COMPONENT_NAMES`（不另造词表）。
  · ② 行为级：探测链必须**环境变量驱动**——README §三.③ 的导出精修先例
    「本机字面量 → 环境变量（缺省 …）」（`activation.py` 的 `LINGSHU_DATA_ROOT`）。
    无理论章节规定该探测链口径，属经验标定（本件 #10）。

断言（每条钉「缺陷不再存在」，不是「代码能跑」）：
  G1 三方名表一致：README §三 点名的组件名集合 == core.py 裸名导入集合 ==
     清单 `externals`∪`hidden_refs` 根名集合。任一侧漏一个名即红。
  G2a 形态：`lingshu/nn/rust_bridge.py` 的**代码字面量**（非 docstring/注释）里
     不得再出现脱敏占位符 `<local-path>`——它是死路径操作数。
  G2b 行为：子进程里 `_locate_hex_exe()` 在 `AEIS_HEX_NN` 未设时返回的路径不得含
     占位符；把 `AEIS_HEX_NN` 指向真实文件时必须返回该文件（证明探测链真在按
     环境变量工作，而不是被脱敏改成了常量假路径）。

作用域（如实）：G2 覆盖 #10 取证列在办面 `lingshu/nn/rust_bridge.py`。
  `lingshu/core/core.py:6391/6408` 的同类占位符（`_kb = r'<local-path> Files\\...'`
  后 `sys.path.insert`）**不在本守卫覆盖内**——该文件由另一批在办（编排规则：
  同文件会撞）；`lingshu/gen/hexgen_multi_seed.py:53` 的 `QWEN` 同类形态属另一组
  （#355/#356）。**本守卫绿不等于全仓已清净**，勿据此推定。

运行（lingshu 仓根）：python -X utf8 -m pytest tests/test_issue10_intake_registration_guard.py -q --no-header
"""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CORE_PY = os.path.join(REPO, "lingshu", "core", "core.py")
README = os.path.join(REPO, "docs", "intake", "body-export-v0.1", "README.md")
MANIFEST = os.path.join(REPO, "docs", "intake", "body-export-v0.1",
                        "_export_manifest.json")
RUST_BRIDGE = os.path.join(REPO, "lingshu", "nn", "rust_bridge.py")

#: 脱敏中性化占位符（导出脱敏把本机路径替成的标记）。
PLACEHOLDER_RE = re.compile(r"<local-path>")


def _stdlib_names():
    return getattr(sys, "stdlib_module_names", frozenset())


def _code_bare_roots(path):
    """AST 扫一文件里**代码级**裸名 import 的根名（排除标准库与本包自名）。"""
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                roots.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    return {r for r in roots if r not in _stdlib_names() and r != "lingshu"}


def _component_names():
    """本仓自有组件名白名单（唯一词表真源）。"""
    sys.path.insert(0, REPO)
    from lingshu.core.component_resolver import COMPONENT_NAMES
    return set(COMPONENT_NAMES)


def _readme_section3_component_roots():
    """README §三 里点名的组件根名（`x.y` 取根；只保留白名单内的词）。"""
    with open(README, encoding="utf-8") as fh:
        text = fh.read()
    sec = text[text.index("## 三、"):text.index("## 四、")]
    vocab = _component_names()
    roots = set()
    for m in re.finditer(r"`([^`]+)`", sec):
        root = m.group(1).split(".")[0]
        if root in vocab:
            roots.add(root)
    return roots


def _manifest_component_roots():
    """清单 core.py 条目 `externals` ∪ `hidden_refs` 的根名（hidden_refs 形如 `5031:name`）。"""
    with open(MANIFEST, encoding="utf-8") as fh:
        payload = json.load(fh)
    core = [f for f in payload["blocks"]["core"]["files"] if f["file"] == "core.py"]
    assert core, "清单 core 块缺 core.py 条目"
    entry = core[0]
    roots = set()
    for name in entry.get("externals") or []:
        roots.add(name.split(".")[0])
    for ref in entry.get("hidden_refs") or []:
        name = ref.split(":", 1)[1] if ":" in ref else ref
        roots.add(name.split(".")[0])
    return roots


# ---------------------------------------------------------------- G1

def test_g1_three_way_component_name_tables_agree():
    """G1：README §三 / core.py 裸名导入 / 清单 externals∪hidden_refs 三方名表必须相等。"""
    vocab = _component_names()
    code = _code_bare_roots(CORE_PY) & vocab
    readme = _readme_section3_component_roots()
    manifest = _manifest_component_roots() & vocab

    problems = []
    if readme != code:
        problems.append("README §三 漏列：%s ；多列：%s"
                        % (sorted(code - readme), sorted(readme - code)))
    if manifest != code:
        problems.append("清单 externals∪hidden_refs 漏列：%s ；多列：%s"
                        % (sorted(code - manifest), sorted(manifest - code)))
    assert not problems, (
        "v0.1 导出登记的两份名表（README §三 vs _export_manifest.json）与 core.py "
        "实有裸名导入不一致（#10）：\n" + "\n".join("  · " + p for p in problems)
        + "\n\n修法：以 core.py 的 AST 裸名导入面为真源，同步 README §三 与清单条目。"
    )


# ---------------------------------------------------------------- G2

def _non_docstring_string_literals(path):
    """文件里**非 docstring** 的字符串字面量（docstring 是模块/类/函数首语句）。"""
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and \
                    isinstance(body[0].value, ast.Constant) and \
                    isinstance(body[0].value.value, str):
                docstrings.add(id(body[0].value))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and id(node) not in docstrings:
            out.append((node.lineno, node.value))
    return out


def test_g2a_no_placeholder_path_operand_in_rust_bridge_code():
    """G2a：rust_bridge.py 的**代码**字面量里不得再有脱敏占位符（死路径操作数）。"""
    offenders = ["rust_bridge.py:%d = %r" % (ln, s)
                 for ln, s in _non_docstring_string_literals(RUST_BRIDGE)
                 if PLACEHOLDER_RE.search(s)]
    assert not offenders, (
        "脱敏把路径操作数改成了字面量占位符（#10）：\n"
        + "\n".join("  · " + o for o in offenders)
        + "\n\n修法：把本机字面量候选改为**环境变量驱动**（README §三.③ 先例）。"
    )


_SNIPPET = r'''
import json, os, sys
from lingshu.nn import rust_bridge as RB
print(json.dumps({"hex_exe": str(RB.HEX_EXE)}))
'''


def _run_locator(env_extra):
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONPATH"] = REPO
    env.pop("AEIS_HEX_NN", None)
    env.pop("AEIS_HEX_NN_DIR", None)
    env.update(env_extra)
    p = subprocess.run([sys.executable, "-X", "utf8", "-c", _SNIPPET],
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       cwd=REPO, env=env)  # #455
    assert p.returncode == 0, "子进程 import rust_bridge 失败：\n%s" % p.stderr[-2000:]
    return json.loads(p.stdout.strip().splitlines()[-1])["hex_exe"]


def test_g2b_locator_is_env_driven_and_never_a_placeholder(tmp_path):
    """G2b（行为级）：探测链按环境变量工作；缺省返回值不得是占位符假路径。"""
    # 未设环境变量：返回值不得含占位符
    default_exe = _run_locator({})
    assert not PLACEHOLDER_RE.search(default_exe), (
        "脱敏占位符被当成了探测链缺省值（#10）：HEX_EXE=%r" % default_exe)

    # 把 AEIS_HEX_NN 指向真实文件：必须原样返回它（证明探测链真在按环境变量工作）
    real = tmp_path / "hex_nn.exe"
    real.write_bytes(b"MZ-fake")
    got = _run_locator({"AEIS_HEX_NN": str(real)})
    assert os.path.normcase(os.path.abspath(got)) == \
        os.path.normcase(os.path.abspath(str(real))), (
        "AEIS_HEX_NN 指向真实文件时探测链没返回它（环境变量驱动失效）：%r" % got)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
