# -*- coding: utf-8 -*-
"""test_issue253_extras_declared_guard · 各层第三方导入必须有声明面（lingshu issue #253）
============================================================================
缺陷（#253，成立）：
    `pyproject.toml` 的 extras 只声明了 numpy / Pillow，而身体侧代码里还有四处
    第三方导入**没有任何声明面**，且 `world` extra 连 Pillow 都没带：

      · `world = ["numpy>=2.3.5,<2.6"]`——**缺 Pillow**，而 world 的 4 个
        `render()` 都在函数内 `from PIL import Image`：
        `skeleton3d.py:128` / `silhouette3d.py:154` / `scene_model.py:234,263` /
        `world3d.py:374` ⇒ 只装 `.[world]` 时「装上世界模型却画不出图」。
      · `nn/hex_train.py:40`  `import pyarrow.parquet as pq`（parquet 语料加载）
      · `gen/hexgen_multi_seed.py:88-89` `import torch` / `from diffusers import ...`
      · `core/activation.py:56` `from scipy.sparse import csr_matrix`（可选稀疏后端）
      后三者与 Pillow 不同：它们**只在单个函数内**按需 import、缺则调用点
      `ModuleNotFoundError`，且体积大 ⇒ 不应进 `full`，而应有各自的 sub-extra。

修复：`world` extra 补 Pillow；新增三个 sub-extra（`core-sparse` / `nn-train` /
    `gen-diffusion`），命名约定 `<层>-<用途>`，**不进 full**。

判据（本件，三层）：
  A 量具自检：AST 扫描必须真扫到 numpy / PIL / scipy / pyarrow / torch / diffusers，
    且 world 里引用 PIL 的模块集合恰为 4 个 render 件（扫描器坏了即红，防假绿）。
  B 主判据（本件机制）：`lingshu/<层>/**` 的**全部非标准库、非本地、非组件**顶层
    导入，必须能在「该层 extra ∪ 该层 `层-*` sub-extra」里找到声明面（PIL→Pillow
    别名）；任一未声明即红并点名「谁 import 的、在第几行」。
  C 定点夹具（只让本件机制生效）：把 world extra 改回不带 Pillow 的形态喂给同一
    检查器，B 的机制必须当场报 `world: PIL`；并断言**既有**的
    `tests/test_gate_dependency_closure.py` **看不见** lingshu（其 `SCAN_DIRS`
    仅 tests/tools/scripts）⇒ 两守卫不可能互相代偿（本件是唯一读 `lingshu/` 导入面
    的声明面守卫）。

判据来源：
  · 「`.[full]` 的常规路径含 numpy / Pillow」——`README.md:52`（`pip install -e
    ".[full]"  # 含 numpy / Pillow（world / nn / gen 的常规路径）`）与
    `docs/intake/body-export-v0.1/README.md:30-33` 的依赖分层表（world 的重载为
    `PIL`、gen 的可选重载为 `torch`）——即「world 的常规路径本就含渲染」是既有声明。
  · 「可选重载不进 full」——`docs/intake/body-export-v0.1/README.md:33` 把 torch
    标为「**仅** hexgen_multi_seed 的语料生成函数内」，且 `pyproject.toml:27-33`
    既有先例（PyYAML 落 dev 而非 full，理由即「只在开发场景被拉起」）。
  · sub-extra 的**版本区间**：scipy / pyarrow / torch / diffusers 未入 README
    「已测版本矩阵」⇒ 如实不设区间（设区间须先补实测矩阵）。判据来源＝经验标定
    （本件 #253），追不到更早出处。

不适用条件 / 已知边界：
  · 本件只判「声明面是否存在」，**不判**声明版本区间是否够窄（那是
    `tests/test_gate_python_floor_and_matrix.py` / `tests/test_hex_text_thresholds.py`
    的 D 组面）；也不真去 pip 装 torch/diffusers（本机未装，装它不在本件范围）。
  · 本件不覆盖 `tests/` `tools/` `scripts/` 的硬导入（那是既有
    `tests/test_gate_dependency_closure.py` 的面），二者扫描目录不重叠。

定点变异自证（抽掉修复 ⇒ 必红）：
  把 `pyproject.toml` 的 `world = [..., "Pillow>=12.2,<13"]` 改回
  `world = ["numpy>=2.3.5,<2.6"]` ⇒ B/C 两组当场红（`world: PIL ←
  lingshu/world/skeleton3d.py:128, …`）；把 `core-sparse`/`nn-train`/`gen-diffusion`
  三个 sub-extra 删掉 ⇒ B 组报 core: scipy / nn: pyarrow / gen: torch, diffusers。

运行（lingshu 仓根）：
    python -X utf8 -m pytest tests/test_issue253_extras_declared_guard.py -q --no-header
退出码：0 = 全过；1 = 有断言失败
"""
from __future__ import annotations

import ast
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYPROJECT = os.path.join(REPO, "pyproject.toml")
PKG = os.path.join(REPO, "lingshu")

LAYERS = ("core", "world", "nn", "gen")

# 发行名别名（import 名 → PyPI 发行名，小写）
DIST_ALIAS = {"pil": "pillow", "yaml": "pyyaml", "pytest": "pytest"}

# world 渲染面：引用 PIL 的模块集合（缺陷 #253 取证点；只钉集合与 render 存在，
# 不钉行号——行号随提交漂移）
WORLD_PIL_MODULES = frozenset({
    "skeleton3d.py", "silhouette3d.py", "scene_model.py", "world3d.py",
})

# #253 点名的六个第三方顶层名（量具自检用：扫不到即扫描器坏了）
KNOWN_THIRD_PARTY = ("numpy", "PIL", "scipy", "pyarrow", "torch", "diffusers")


def _local_tops():
    """本地顶层名：本包自名 + 引擎的裸名组件（非第三方）。

    组件名从 `component_resolver.COMPONENT_NAMES` 取（AST 读，避免 import 副作用）。
    """
    tops = {"lingshu"}
    path = os.path.join(PKG, "core", "component_resolver.py")
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "COMPONENT_NAMES":
            for elt in node.value.args:
                tops.update(ast.literal_eval(elt))
    return tops


def _third_party_imports(layer):
    """`lingshu/<layer>/**` 的非标准库 / 非本地顶层导入 → {top: [relpath:行, …]}。"""
    local = _local_tops()
    found: dict = {}
    root = os.path.join(PKG, layer)
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d != "__pycache__"]
        for f in sorted(fn):
            if not f.endswith(".py"):
                continue
            p = os.path.join(dp, f)
            rel = os.path.relpath(p, REPO).replace("\\", "/")
            tree = ast.parse(open(p, encoding="utf-8").read())
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name.split(".")[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names = [node.module.split(".")[0]]
                for top in names:
                    if (top in sys.stdlib_module_names or top in local
                            or top.startswith("_")):
                        continue
                    found.setdefault(top, []).append("%s:%d" % (rel, node.lineno))
    return found


def _declared_extras(pyproject_text):
    """{extra 名: {发行名小写}}。"""
    data = _parse_toml(pyproject_text)
    proj = data.get("project", {})
    out = {}
    for extra, items in (proj.get("optional-dependencies") or {}).items():
        names = set()
        for spec in items or []:
            head = spec.split("[")[0]
            for sep in (">", "=", "<", "!", ";", " "):
                head = head.split(sep)[0]
            names.add(head.strip().lower())
        out[extra] = names
    return out


def _parse_toml(text):
    # 本仓 requires-python >= 3.12（pyproject.toml:15）⇒ tomllib 恒可用，不引第三方
    # 解析器（否则 tests/ 面多一个未声明的硬依赖，会撞
    # tests/test_gate_dependency_closure.py）。
    import tomllib
    return tomllib.loads(text)


def _layer_extras(extra_names, layer):
    """该层的声明面 = `layer` extra ∪ 全部 `layer-*` sub-extra。"""
    return {e for e in extra_names if e == layer or e.startswith(layer + "-")}


def _check(pyproject_text, repo_root=REPO):
    """返回 {层: {未声明的第三方 top: [位置]}}（空 dict = 全有声明面）。"""
    extras = _declared_extras(pyproject_text)
    bad = {}
    for layer in LAYERS:
        declared = set()
        for e in _layer_extras(extras, layer):
            declared |= extras[e]
        missing = {}
        for top, places in _third_party_imports(layer).items():
            dist = DIST_ALIAS.get(top.lower(), top.lower())
            if dist not in declared:
                missing[top] = places
        if missing:
            bad[layer] = missing
    return bad


# ---------------------------------------------------------------- A 量具自检

def test_scanner_is_alive():
    """A：扫描器必须真扫到六个已知第三方名，且 world 的 PIL 面恰为 4 个 render 件。"""
    seen: dict = {}
    for layer in LAYERS:
        for top, places in _third_party_imports(layer).items():
            seen.setdefault(top, []).extend(places)
    absent = [m for m in KNOWN_THIRD_PARTY if m not in seen]
    assert not absent, (
        "扫描器没扫到 %s ⇒ 它坏了（或 LAYERS 失效），本判据会**静默通过**。实扫：%s"
        % (absent, sorted(seen)))

    pil_mods = {p.split("/")[-1].split(":")[0] for p in seen["PIL"]}
    # `anim.py` 走**并集**而不是 `WORLD_PIL_MODULES`：后者在下方被断言"每个都有
    # `def render(`"（那是 world 的 4 个渲染件），而 anim.py 的面是 `frames/to_gif`。
    assert pil_mods == set(WORLD_PIL_MODULES) | {"hex_gen.py", "hex_train.py",
                                                 "hexgen_c1_real.py", "hexgen_multi_seed.py",
                                                 "anim.py"}, (
        "引用 PIL 的模块集合与预期不符（扫描器或代码面已变）：%s" % sorted(pil_mods))

    # world 的 4 个 render 件确实各有 render()（Pillow 是它们的**常规路径**依赖）
    for fn in sorted(WORLD_PIL_MODULES):
        src = open(os.path.join(PKG, "world", fn), encoding="utf-8").read()
        assert "def render(" in src, (
            "%s 不再有 render()（Pillow 依赖面已变）——本件判据须重标定" % fn)


# ---------------------------------------------------------------- B 主判据

def test_every_layer_third_party_import_has_declaration():
    """B：各层全部第三方导入都能在「层 extra ∪ 层-* sub-extra」里找到声明面。"""
    text = open(PYPROJECT, encoding="utf-8").read()
    bad = _check(text)
    assert not bad, (
        "以下第三方导入**没有 pyproject 声明面**——`pip install -e \".[<层>]\"` 不会装它们，"
        "走到这些位置必 `ModuleNotFoundError`（#253）：\n"
        + "\n".join("  · %s: %s   ← %s" % (layer, top, ", ".join(sorted(places)))
                    for layer, tops in sorted(bad.items())
                    for top, places in sorted(tops.items()))
        + "\n\n已声明 extras = "
        + repr({k: sorted(v) for k, v in sorted(_declared_extras(text).items())})
        + "\n修法：把发行名加进该层的 extra，或新建 `层-用途` sub-extra（如 core-sparse）。")


def test_world_extra_declares_pillow():
    """B′：world extra **本层**必须带 Pillow（缺陷 #253 的正身；不接受靠 sub-extra 顶）。"""
    extras = _declared_extras(open(PYPROJECT, encoding="utf-8").read())
    assert "world" in extras, "pyproject 没有 world extra"
    assert "pillow" in extras["world"], (
        "world extra 缺 Pillow——而 world 的 4 个 render()（skeleton3d / silhouette3d / "
        "scene_model / world3d）在函数内 `from PIL import Image` ⇒ 只装 `.[world]` 时"
        "「装上世界模型却画不出图」（#253）。实读 world extra = %s" % sorted(extras["world"]))
    assert "numpy" in extras["world"], "world extra 缺 numpy（回归）"


# ---------------------------------------------------------------- C 定点夹具

def test_mutation_world_without_pillow_is_red():
    """C-1：把 world extra 改回不带 Pillow 的形态 ⇒ 本件机制当场报 world: PIL。

    这是**只让本件机制生效**的夹具：既有 `test_gate_dependency_closure.py` 的
    SCAN_DIRS 仅 tests/tools/scripts（见 C-2），根本看不见 `lingshu/` 的导入面，
    故它不可能代偿本条——抽掉修复后唯一能变红的就是本件。
    """
    text = open(PYPROJECT, encoding="utf-8").read()
    assert 'world = ["numpy>=2.3.5,<2.6", "Pillow>=12.2,<13"]' in text, (
        "pyproject 的 world extra 形态与预期不符，本夹具须重写（实读文本已变）")
    mutated = text.replace(
        'world = ["numpy>=2.3.5,<2.6", "Pillow>=12.2,<13"]',
        'world = ["numpy>=2.3.5,<2.6"]')
    bad = _check(mutated)
    assert "world" in bad and "PIL" in bad["world"], (
        "变异自证失败：把 world extra 的 Pillow 抽掉后，检查器**没**报 world: PIL ⇒ "
        "本件判据无判别力（代偿或量具坏了）。实报 = %r" % bad)


def test_mutation_dropping_subextras_is_red():
    """C-2：删掉三个 sub-extra ⇒ 检查器必报 core: scipy / nn: pyarrow / gen: torch+diffusers。"""
    text = open(PYPROJECT, encoding="utf-8").read()
    mutated = "\n".join(
        l for l in text.splitlines()
        if not l.startswith(("core-sparse =", "nn-train =", "gen-diffusion =")))
    bad = _check(mutated)
    for layer, top in (("core", "scipy"), ("nn", "pyarrow"),
                       ("gen", "torch"), ("gen", "diffusers")):
        assert layer in bad and top in bad[layer], (
            "变异自证失败：删掉 sub-extra 后检查器**没**报 %s: %s ⇒ 无判别力。实报 = %r"
            % (layer, top, bad))


def test_existing_gate_guard_cannot_compensate():
    """C-3：既有 `test_gate_dependency_closure.py` 的扫描面不含 lingshu ⇒ 不可能代偿。"""
    p = os.path.join(REPO, "tests", "test_gate_dependency_closure.py")
    tree = ast.parse(open(p, encoding="utf-8").read())
    scan_dirs = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "SCAN_DIRS":
            scan_dirs = ast.literal_eval(node.value)
    assert scan_dirs is not None, "既有守卫里没找到 SCAN_DIRS ⇒ 本夹具须重写"
    assert "lingshu" not in scan_dirs, (
        "既有守卫现在也扫 lingshu（SCAN_DIRS=%r）⇒ 与本件机制重叠、可能互相代偿，"
        "须重新设计只让本件生效的夹具" % (scan_dirs,))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
