# -*- coding: utf-8 -*-
r"""#608 import_direction_lint 四形态守卫：core→world 闸不再被「改写法」绕过

缺陷（lingshu issue #608，P2）：
  tools/import_direction_lint.py 的 RE_CORE_TO_WORLD 旧口径只认相对导入字面形态
  （`^\s*(?:from|import)\s+\.\.world\b`）。同语义的反向依赖全数不到：
    · `from lingshu.world.prediction import Oracle`（绝对导入）
    · `from .. import world`（含 `from .. import world, x`）
    · `import lingshu.world.prediction`
    · `importlib.import_module("lingshu.world.prediction")`（动态导入字符串）
  后果：把一行相对导入改写成绝对导入（依赖还在、只是换写法）⇒ 计数 1→0、
  退出码 0→0 恒绿，baseline「只准减不准增」被绕过；baseline 里
  `grep -n "from ..world" lingshu/core/` 为空的字面终局判据同样失明。
  改前读数（题面对照实验在本仓复现成立）：夹具 lingshu/core/xxx.py 六种写法、
  基线配额 1，第 4 行改绝对导入后旧工具计数 1→0、exit 0→0、报
  「基线中已失效条目 1 个（可清理）」。

修法：RE_CORE_TO_WORLD 扩为四条正则列表（CORE_TO_WORLD_PATTERNS），行级口径
不变（一行命中任一形态计 1、多形态同行不重复计）；`lingshu.core` 内部互引不算；
字符串形态只认形如 "lingshu.world…" 的模块路径。仓内实测（375cf55 分支，
2026-10-15）：扩形态前后 core_to_world 均 28 行（全在 core.py）、stmc_refs 均
4 行、exit 0——四条新形态在 main 无存量实例，基线数据行不变（只更正注释）。

断言组（抽掉任一修复即红；正则回退旧单形态 ⇒ G1/G3 必红）：
  G1 四种新形态各构造一处 ⇒ 全部计数（含 `from .. import world, x` 混名形态、
     裸调 `import_module("lingshu.world…")`）；四形态同文件逐行归属。
  G2 防过修：既有 `from ..world.x` 相对形态计数不变；core 内部互引（绝对
     `lingshu.core…`、相对 `..core…`、动态 `import_module("lingshu.core…")`）、
     前缀撞名（`lingshu.worldly` / `"lingshu.world_extra"`）、行中字符串
     （core.py 报错文案里的 `from lingshu.world`）、world 侧绝对导入 ⇒ 全不计数。
  G3 端到端（题面对照实验改后）：
     · 单行配额 1：相对行改写成绝对导入 ⇒ 计数**不归零**（仍 1）、无 stale——
       旧口径此处为 0＋stale，即绕过点本体；
     · 六形态夹具（配额 1 为旧盲口径所记）：改写后计数 6 > 1 ⇒ exit 1 判红；
     · 干净基线上新增任一新形态 ⇒ exit 1 并点名 core_to_world。
  G4 真实仓读数：出货基线 exit 0、合计=基线合计、无 stale；输出与基线注释的
     终局判据均改为「工具 core_to_world 计数 = 0」（#608），不再以字面 grep 为准。

运行（仓根）：python -X utf8 -m pytest tests/test_issue608_import_direction_abs_guard.py -q --no-header
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "tools" / "import_direction_lint.py"
BASELINE = REPO_ROOT / "tools" / "import_direction_lint_baseline.txt"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import import_direction_lint as I  # noqa: E402


def run_lint(*args, cwd=None, timeout=300):
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "-X", "utf8", str(TOOL), *args],
        cwd=str(cwd or REPO_ROOT), capture_output=True, timeout=timeout, env=env)


def payload(res):
    assert res.stdout, "工具没有输出：stderr=%r" % res.stderr[:2000]
    return json.loads(res.stdout.decode("utf-8"))


def write_module(directory: Path, name: str, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(body, encoding="utf-8")
    return path


def scan_one(body: str, tmp_path, sub="core"):
    """合成仓（lingshu/core/xxx.py = body）后调用 I.scan，返回 core_to_world 计数。"""
    root = tmp_path / "repo"
    write_module(root / "lingshu" / sub, "xxx.py", body)
    counts, errors = I.scan(root, root / "lingshu")
    assert errors == [], errors
    return counts[I.DIR_CORE_TO_WORLD].get("lingshu/core/xxx.py", 0), counts


# --------------------------------------------------- G1 四种新形态全部计数
def test_g1_from_parent_import_world_counted(tmp_path):
    """形态 b：`from .. import world`（#608 新增，旧口径数不到）。"""
    n, _ = scan_one("def m():\n    from .. import world as _w\n    return _w\n", tmp_path)
    assert n == 1


def test_g1_from_parent_import_world_with_sibling_counted(tmp_path):
    """形态 b 变体：`from .. import world, x`——world 混在多个导入名里，匹配该行。"""
    n, _ = scan_one("def m():\n    from .. import world, core as _pair\n    return _pair\n",
                    tmp_path)
    assert n == 1


def test_g1_absolute_from_import_counted(tmp_path):
    """形态 c：`from lingshu.world.prediction import Oracle`（绝对导入）。"""
    n, _ = scan_one("def m():\n    from lingshu.world.prediction import Oracle\n    return Oracle\n",
                    tmp_path)
    assert n == 1


def test_g1_absolute_plain_import_counted(tmp_path):
    """形态 c 变体：`import lingshu.world.prediction`。"""
    n, _ = scan_one("def m():\n    import lingshu.world.prediction as _p\n    return _p\n",
                    tmp_path)
    assert n == 1


def test_g1_importlib_import_module_string_counted(tmp_path):
    """形态 d：`importlib.import_module("lingshu.world…")` 动态导入字符串。"""
    body = ("import importlib\n"
            "def m():\n"
            "    return importlib.import_module(\"lingshu.world.prediction\")\n")
    n, _ = scan_one(body, tmp_path)
    assert n == 1


def test_g1_bare_import_module_call_counted(tmp_path):
    """形态 d 变体：`from importlib import import_module` 后裸调
    `import_module("lingshu.world…")`（调用可出现在行中，非行首锚定）。"""
    body = ("from importlib import import_module\n"
            "def m(name):\n"
            "    mod = import_module(\"lingshu.world.prediction\")\n"
            "    return mod\n")
    n, _ = scan_one(body, tmp_path)
    assert n == 1


def test_g1_all_four_forms_in_one_file_attributed_per_line(tmp_path):
    """四形态同文件 ⇒ 逐行归属共 4 行（一行一计，不按形态重复也不互相吞并）。"""
    body = (
        "import importlib\n"
        "from importlib import import_module as _im\n"
        "\n"
        "\n"
        "def m1():\n"
        "    from .. import world as _w\n"                       # b
        "    return _w\n"
        "\n"
        "\n"
        "def m2():\n"
        "    from lingshu.world.prediction import Oracle\n"      # c
        "    return Oracle\n"
        "\n"
        "\n"
        "def m3():\n"
        "    import lingshu.world.prediction as _p\n"            # c
        "    return _p\n"
        "\n"
        "\n"
        "def m4():\n"
        "    return importlib.import_module(\"lingshu.world.prediction\")\n")  # d
    n, _ = scan_one(body, tmp_path)
    assert n == 4


# --------------------------------------------------- G2 防过修：误报边界
def test_g2_relative_form_still_counted_unchanged(tmp_path):
    """既有 `from ..world.x` 相对形态计数不变（#178 口径原样保持，两行计 2）。"""
    n, _ = scan_one("def m():\n    from ..world.a import X\n    from ..world.b import Y\n    return X, Y\n",
                    tmp_path)
    assert n == 2


def test_g2_core_internal_and_unrelated_not_counted(tmp_path):
    """core 内部互引 / 前缀撞名 / 行中字符串 / 裸 importlib ⇒ 全不计数。"""
    body = (
        "import importlib\n"
        "import importlib.machinery\n"
        "from lingshu.core.component_resolver import X\n"      # core 内部（绝对）
        "import lingshu.core.core\n"                           # core 内部（绝对）
        "from ..core.core import LayeredStore\n"               # core 内部（相对）
        "spec = importlib.import_module(\"lingshu.core.component_resolver\")\n"  # 内部动态
        "from lingshu.worldly import x\n"                      # 前缀撞名（\\b 挡）
        "mod = import_module(\"lingshu.world_extra\")\n"       # 前缀撞名（_ 是词字符）
        "return {\"error\": \"初始化步骤：from lingshu.world \"\n"  # 行中字符串（非行首）
        "        \"import register_world_process;\"}\n")
    n, counts = scan_one(body, tmp_path)
    assert n == 0, counts[I.DIR_CORE_TO_WORLD]


def test_g2_world_side_absolute_import_not_counted(tmp_path):
    """world 侧的绝对导入（如 mesh_svg.py 形态）不在 lingshu/core/ 下 ⇒ 不计。"""
    root = tmp_path / "repo"
    write_module(root / "lingshu" / "world", "mesh_svg_like.py",
                 "def m():\n    from lingshu.world import mesh3d\n    return mesh3d\n")
    counts, errors = I.scan(root, root / "lingshu")
    assert errors == []
    assert counts[I.DIR_CORE_TO_WORLD] == {}, counts


# --------------------------------------------------- G3 端到端：写法变化不再绕过
def test_g3_rewrite_to_absolute_keeps_count_no_stale(tmp_path):
    """★题面对照实验（单行配额 1）：相对行改写成绝对导入 ⇒ 计数**不归零**（仍 1）、
    无 stale、exit 0（配额内维持）。旧口径此处 = 计数 0 ＋ stale 条目——绕过点本体；
    正则回退旧形态 ⇒ 本条必红（计数变 0 / stale 非空）。"""
    root = tmp_path / "repo"
    engine = write_module(root / "lingshu" / "core", "xxx.py",
                          "def m():\n    from ..world.prediction import Oracle\n    return Oracle\n")
    assert run_lint("--repo-root", str(root), "--update-baseline").returncode == 0
    base = root / "tools" / "import_direction_lint_baseline.txt"
    assert "core_to_world\tlingshu/core/xxx.py\t1" in base.read_text(encoding="utf-8")

    # ← 变异：把唯一的相对导入改写成绝对导入（依赖还在、只是换写法）
    engine.write_text("def m():\n    from lingshu.world.prediction import Oracle\n    return Oracle\n",
                      encoding="utf-8")
    data = payload(run_lint("--repo-root", str(root), "--json"))
    assert data["core_to_world_total"] == 1, data          # 不归零（旧口径 = 0）
    assert data["stale_baseline_entries"] == [], data      # 不伪造失效（旧口径 = 1 条 stale）
    assert data["exit_code"] == 0, data                    # 配额内维持，非绿灯假象


def test_g3_issue_fixture_six_forms_rewrite_turns_red(tmp_path):
    """★题面夹具：六种写法、基线配额 1（旧盲口径只数得出相对那行）。改写第 4 行
    （相对形态）为绝对导入后：计数 6（改写行仍计）> 配额 1 ⇒ exit 1 判红——
    门禁不再被写法变化绕过。旧口径此处 = 计数 0、exit 0 恒绿。"""
    six = (
        "import importlib\n"
        "\n"
        "\n"
        "def m1():\n"
        "    from ..world.prediction import Oracle          # 相对（旧口径唯一计数）\n"
        "    return Oracle\n"
        "\n"
        "\n"
        "def m2():\n"
        "    from .. import world as _w\n"
        "    return _w\n"
        "\n"
        "\n"
        "def m3():\n"
        "    from lingshu.world.prediction import Oracle2\n"
        "    return Oracle2\n"
        "\n"
        "\n"
        "def m4():\n"
        "    import lingshu.world.prediction as _p\n"
        "    return _p\n"
        "\n"
        "\n"
        "def m5():\n"
        "    return importlib.import_module(\"lingshu.world.prediction\")\n"
        "\n"
        "\n"
        "def m6():\n"
        "    from .. import world, core as _pair\n"
        "    return _pair\n")
    root = tmp_path / "repo"
    engine = write_module(root / "lingshu" / "core", "xxx.py", six)
    base_dir = root / "tools"
    base_dir.mkdir(parents=True)
    (base_dir / "import_direction_lint_baseline.txt").write_text(
        "core_to_world\tlingshu/core/xxx.py\t1\n", encoding="utf-8")

    # 旧盲口径下六形态文件只数 1（配额 1 恰好绿）；四形态口径如实数 6 ⇒ 已红
    data0 = payload(run_lint("--repo-root", str(root), "--json"))
    assert data0["core_to_world_total"] == 6, data0

    # ← 变异：第 4 行起的相对形态改写成绝对导入
    engine.write_text(six.replace(
        "    from ..world.prediction import Oracle          # 相对（旧口径唯一计数）",
        "    from lingshu.world.prediction import Oracle    # 改写成绝对导入（#608）"),
        encoding="utf-8")
    data = payload(run_lint("--repo-root", str(root), "--json"))
    assert data["core_to_world_total"] == 6, data          # 改写行仍计，计数不归零
    assert data["exit_code"] == 1, "改写法后守卫竟然没红：%r" % data
    assert any(o["direction"] == I.DIR_CORE_TO_WORLD and o["excess"] == 5
               for o in data["new_over_baseline"]), data


def test_g3_new_form_additions_red_on_clean_baseline(tmp_path):
    """干净（空）基线上 core 里新增任一新形态 ⇒ exit 1 并点名 core_to_world。"""
    body = ("import importlib\n"
            "from importlib import import_module as _im\n"
            "\n"
            "\n"
            "def m1():\n"
            "    from .. import world\n"
            "    return world\n"
            "\n"
            "\n"
            "def m2():\n"
            "    from lingshu.world.prediction import Oracle\n"
            "    return Oracle\n"
            "\n"
            "\n"
            "def m3():\n"
            "    import lingshu.world.prediction\n"
            "\n"
            "\n"
            "def m4():\n"
            "    return importlib.import_module(\"lingshu.world.prediction\")\n")
    root = tmp_path / "repo"
    write_module(root / "lingshu" / "core", "xxx.py", body)
    base = root / "tools" / "import_direction_lint_baseline.txt"
    base.parent.mkdir(parents=True)
    base.write_text("", encoding="utf-8")
    res = run_lint("--repo-root", str(root), "--json")
    data = payload(res)
    assert res.returncode == 1, data
    assert [(o["direction"], o["file"], o["count"])
            for o in data["new_over_baseline"]] == [(I.DIR_CORE_TO_WORLD,
                                                     "lingshu/core/xxx.py", 4)], data


# --------------------------------------------------- G4 真实仓读数（改后对比）
def test_g4_real_repo_readings_unchanged_by_wider_patterns():
    """扩形态后仓内实测与基线一致：exit 0、合计=基线合计、无超项、无 stale——
    四条新形态在 main 无存量实例（题面断言），扩宽正则未引入误报。"""
    data = payload(run_lint("--json"))
    assert data["exit_code"] == 0, data
    assert data["new_over_baseline"] == [], data
    assert data["core_to_world_total"] == data["baseline_total"]["core_to_world"], data
    assert data["stmc_refs_total"] == data["baseline_total"]["stmc_refs"], data
    assert data["stale_baseline_entries"] == [], data


def test_g4_terminal_criterion_no_longer_literal_grep():
    """终局判据同步更正（#608）：工具输出与 baseline 注释都不再以字面
    `grep "from ..world"` 为准，改按工具 core_to_world 计数 = 0。"""
    res = run_lint()
    out = res.stdout.decode("utf-8")
    assert res.returncode == 0
    assert "#608 四形态口径" in out, out
    assert "不以 \"from ..world\" 字面为准" in out, out
    text = BASELINE.read_text(encoding="utf-8")
    assert "#608" in text, "baseline 未留痕 #608"
    assert "计数 = 0" in text, "baseline 终局判据未改为工具计数口径"
    entries, errors = I.parse_baseline(text)
    assert errors == []
    # 数据行：#434 清零 world/prediction.py 的 2 处 stmc_refs（4 → 2）
    assert entries == {"core_to_world:lingshu/core/core.py": 28,
                       "stmc_refs:lingshu/core/component_resolver.py": 1,
                       "stmc_refs:lingshu/core/core.py": 1}, entries
