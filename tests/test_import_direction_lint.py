# -*- coding: utf-8 -*-
"""批次 F / issue #178 · `tools/import_direction_lint.py` 的验收测试（守卫）

本件机制：**core↔world 双向依赖计数只准减、不准增**（防新增）。
  · `core_to_world`（lingshu/core/ 里 `from ..world`/`import ..world` 的行数）> 基线 ⇒ 红；
  · `stmc_refs`（lingshu/ 里含旧包名 `spacetime_memory_core` 的行数）> 基线 ⇒ 红。

★★本批（P2 批次 F）**只上尺子＋定基线（防新增）**：
  · **11 个世界层外观方法迁到 `lingshu/facade/world_api.py` 的重构未做**；
  · issue #178 的终局目标 `grep -n "from ..world" lingshu/core/` 为空 **本批达不到**；
  · 守卫只保证计数不再增长，不保证终局目标。

断言组
------
A 量具自证（lint 必须真在数）：
   · 合成 core 模块里的 `from ..world.x import Y` ⇒ core_to_world ≥ 1；
   · 合成含 `spacetime_memory_core` 的行 ⇒ stmc_refs ≥ 1；
   · core 目录外的 `from ..world`（如 world 内部相对导入）不计入 core_to_world。
B 真实仓库（出货基线）⇒ 退出码 0、无超基线项、合计 = 基线合计。
C 定点变异（把基线数字改小一格 ⇒ 守卫必红）：临时副本基线，不动出货基线。
D 基线外新增（core 里多一行 `from ..world`）⇒ 退出 1 并点名。
E stale 可观测（v2 · 修死条件恒空）：基线登记的 (方向,文件) 消失（删文件/清引用）
   ⇒ stale_baseline_entries 必须点名；B 组补「实测合计 == 基线合计」防下降漂移静默。

判据来源：公开 issue #178「core↔world 双向依赖」＋ 本仓 tools/time_core_lint.py 先例。

本文件只读仓库源码、只写 tmp_path；不改 lingshu/ 下任何文件。
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


# --------------------------------------------------- A 量具自证
def test_core_to_world_lazy_import_counted(tmp_path):
    root = tmp_path / "repo"
    write_module(root / "lingshu" / "core", "engine.py",
                 "def method(self):\n"
                 "    from ..world.prediction import PredictionEngine\n"
                 "    return PredictionEngine\n")
    counts, errors = I.scan(root, root / "lingshu")
    assert errors == []
    assert counts[I.DIR_CORE_TO_WORLD] == {"lingshu/core/engine.py": 1}, counts


def test_stmc_old_name_counted(tmp_path):
    root = tmp_path / "repo"
    write_module(root / "lingshu" / "world", "prediction.py",
                 "try:\n"
                 "    from spacetime_memory_core import LayeredStore as _L\n"
                 "except Exception:\n"
                 "    from ..core.core import LayeredStore as _L\n")
    counts, errors = I.scan(root, root / "lingshu")
    assert errors == []
    assert counts[I.DIR_STMC_REFS] == {"lingshu/world/prediction.py": 1}, counts


def test_world_side_relative_import_not_core_to_world(tmp_path):
    """`lingshu/world/` 内部的 `from ..core` 不计入 core_to_world（只在 core/ 里数）。"""
    root = tmp_path / "repo"
    write_module(root / "lingshu" / "world", "mod.py",
                 "from ..core.core import X\n")
    counts, _ = I.scan(root, root / "lingshu")
    assert counts[I.DIR_CORE_TO_WORLD] == {}, counts


# --------------------------------------------------- B 真实仓库：出货基线绿
def test_real_repo_shipped_baseline_exit_zero():
    res = run_lint("--json")
    data = payload(res)
    assert res.returncode == 0, data
    assert data["new_over_baseline"] == []
    assert data["summary"]["new"] == 0
    # 基线里确实登记了真实计数（否则规则可能失效）
    assert data["core_to_world_total"] >= 1, "core_to_world 基线为 0，规则可能失效"
    assert data["stmc_refs_total"] >= 1, "stmc_refs 基线为 0，规则可能失效"
    # 合计与基线合计一致（与姊妹守卫 test_silent_except_lint 同口径）——计数
    # **下降**方向（外观方法迁出 core ⇒ 终局目标推进）的基线漂移不得静默。
    assert data["core_to_world_total"] == data["baseline_total"]["core_to_world"], data
    assert data["stmc_refs_total"] == data["baseline_total"]["stmc_refs"], data
    # 出货树上没有失效基线条目（文件被删/改名 ⇒ stale 必须可观测，见 E 组）
    assert data["stale_baseline_entries"] == [], data


def test_shipped_baseline_file_has_provenance_and_terminal_goal_note():
    text = BASELINE.read_text(encoding="utf-8")
    assert "基线来源" in text and "HEAD" in text
    # ★必须如实写明终局目标本批未达成（任务书强制）
    assert "facade/world_api.py" in text, "基线未写明外观方法迁移重构未做"
    assert "本批达不到" in text or "本批未达成" in text or "只上尺子" in text
    entries, errors = I.parse_baseline(text)
    assert errors == []
    assert entries, "基线清单为空"


# --------------------------------------------------- C 定点变异：改小一格 ⇒ 红
def test_mutation_shrink_baseline_by_one_turns_red(tmp_path):
    """★ 定点变异：把 core_to_world 基线改小一格 ⇒ 守卫必红并点名。"""
    data0 = payload(run_lint("--json"))
    assert data0["exit_code"] == 0
    cw = data0["core_to_world_total"]
    assert cw >= 1, "core_to_world 计数为 0，无法做变异"

    # 复刻当前全部基线，但把 core_to_world 的 core.py 条目减 1
    lines = []
    for rec in data0["per_key"]:
        n = rec["count"]
        if rec["direction"] == I.DIR_CORE_TO_WORLD:
            n -= 1                            # ← 变异：改小一格
        lines.append("%s\t%s\t%d" % (rec["direction"], rec["file"], n))
    mutated = tmp_path / "shrunk_baseline.txt"
    mutated.write_text("\n".join(lines) + "\n", encoding="utf-8")

    res = run_lint("--baseline", str(mutated), "--json")
    data = payload(res)
    assert res.returncode == 1, "基线改小一格后守卫竟然没红：%r" % data
    over = {(o["direction"], o["excess"]) for o in data["new_over_baseline"]}
    assert (I.DIR_CORE_TO_WORLD, 1) in over, over


def test_mutation_empty_baseline_turns_red(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    res = run_lint("--baseline", str(empty))
    assert res.returncode == 1
    assert "超基线" in res.stdout.decode("utf-8")


# --------------------------------------------------- D 基线外新增 ⇒ 红
def test_new_core_to_world_import_turns_red(tmp_path):
    root = tmp_path / "repo"
    write_module(root / "lingshu" / "core", "engine.py",
                 "def m(self):\n    from ..world.a import X\n")
    baseline = root / "tools" / "import_direction_lint_baseline.txt"
    baseline.parent.mkdir(parents=True, exist_ok=True)
    baseline.write_text("", encoding="utf-8")
    res = run_lint("--repo-root", str(root), "--json")
    data = payload(res)
    assert res.returncode == 1, data
    assert any(o["direction"] == I.DIR_CORE_TO_WORLD
               for o in data["new_over_baseline"]), data


def test_no_new_import_stays_green(tmp_path):
    root = tmp_path / "repo"
    write_module(root / "lingshu" / "core", "engine.py",
                 "def m(self):\n    return 1\n")
    baseline = root / "tools" / "import_direction_lint_baseline.txt"
    baseline.parent.mkdir(parents=True, exist_ok=True)
    baseline.write_text("", encoding="utf-8")
    assert run_lint("--repo-root", str(root), "--json").returncode == 0


# --------------------------------------------------- baseline 解析 / 更新往返
def test_baseline_parse_and_render_roundtrip():
    counts = {I.DIR_CORE_TO_WORLD: {"lingshu/core/a.py": 2},
              I.DIR_STMC_REFS: {"lingshu/world/b.py": 1}}
    text = I.render_baseline(counts)
    parsed, errors = I.parse_baseline(text)
    assert errors == []
    assert parsed == {"core_to_world:lingshu/core/a.py": 2,
                      "stmc_refs:lingshu/world/b.py": 1}


def test_update_baseline_roundtrip(tmp_path):
    root = tmp_path / "repo"
    write_module(root / "lingshu" / "core", "engine.py",
                 "def m(self):\n    from ..world.a import X\n")
    res = run_lint("--repo-root", str(root), "--update-baseline")
    assert res.returncode == 0
    baseline = root / "tools" / "import_direction_lint_baseline.txt"
    assert baseline.exists()
    assert "core_to_world\tlingshu/core/engine.py\t1" in baseline.read_text(encoding="utf-8")
    assert run_lint("--repo-root", str(root)).returncode == 0


def test_usage_errors_return_two(tmp_path):
    assert run_lint("--repo-root", str(tmp_path / "no-root")).returncode == 2
    assert run_lint("--baseline", str(tmp_path / "nope.txt")).returncode == 2
    bad = tmp_path / "bad.txt"
    bad.write_text("这不是一条合法条目\n", encoding="utf-8")
    assert run_lint("--baseline", str(bad)).returncode == 2
    bad2 = tmp_path / "bad2.txt"
    bad2.write_text("unknown_dir\tlingshu/a.py\t1\n", encoding="utf-8")
    assert run_lint("--baseline", str(bad2)).returncode == 2
    assert run_lint("--update-baseline", "--json").returncode == 2


def test_json_byte_identical_across_runs(tmp_path):
    first = run_lint("--json")
    second = run_lint("--json")
    assert first.stdout == second.stdout
    assert first.returncode == second.returncode == 0
    third = run_lint("--json", cwd=tmp_path)
    assert third.stdout == first.stdout


# ------------------------------------------- 登记面（documented-exemption surface）
def test_registration_surface_is_preserved_by_update_baseline(tmp_path):
    """★「登记面」用户登记段须被 --update-baseline 逐字保留（仅作文档，无强制力）。"""
    root = tmp_path / "repo"
    write_module(root / "lingshu" / "core", "engine.py",
                 "def m(self):\n    from ..world.a import X\n")
    baseline = root / "tools" / "import_direction_lint_baseline.txt"
    baseline.parent.mkdir(parents=True, exist_ok=True)
    baseline.write_text(
        "%s\n%s\n%s\n" % (I.REG_BEGIN,
                          "# 2026-10-11 deadbee core_to_world lingshu/core/engine.py  外观方法待迁 facade（#178）",
                          I.REG_END),
        encoding="utf-8")
    res = run_lint("--repo-root", str(root), "--update-baseline")
    assert res.returncode == 0
    text = baseline.read_text(encoding="utf-8")
    assert "外观方法待迁 facade（#178）" in text, text
    assert "core_to_world\tlingshu/core/engine.py\t1" in text, text


def test_registration_surface_has_no_force():
    text = (I.REG_BEGIN + "\n"
            + "# 2026-10-11 deadbee core_to_world lingshu/core/x.py  允许\n"
            + I.REG_END + "\n")
    entries, errors = I.parse_baseline(text)
    assert errors == []
    assert entries == {}, "登记面注释不应被 parse_baseline 当作计数条目"


# ------------------------------------- E stale 可观测（v2 · 修死条件恒空）
def test_stale_baseline_entries_list_deleted_file(tmp_path):
    """★v2：基线登记的 (方向,文件) 从仓库消失 ⇒ stale_baseline_entries 必须点名。

    v1 形态实测：该字段恒为 []（生成器条件「方向前缀 ∉ DIRECTIONS」被
    parse_baseline 的方向校验提前挡死，永假）——文件删除/改名后基线残留行
    不可观测。本断言即定点变异：把 v2 的「按实测键反查」撤掉 ⇒ 本条必红。
    """
    root = tmp_path / "repo"
    write_module(root / "lingshu" / "core", "engine.py",
                 "def m(self):\n    from ..world.a import X\n    return X\n")
    res = run_lint("--repo-root", str(root), "--update-baseline")
    assert res.returncode == 0, res.stdout + res.stderr
    base = root / "tools" / "import_direction_lint_baseline.txt"
    assert "core_to_world\tlingshu/core/engine.py\t1" in base.read_text(encoding="utf-8")

    (root / "lingshu" / "core" / "engine.py").unlink()      # ← 变异：删掉基线登记的文件
    res2 = run_lint("--repo-root", str(root), "--json")
    data = payload(res2)
    assert data["stale_baseline_entries"] == ["core_to_world:lingshu/core/engine.py"], data
    # 失效不等于违规（退出仍 0——清理基线是维护动作，不是守卫红线）
    assert res2.returncode == 0, data


def test_stale_baseline_entries_list_zeroed_count(tmp_path):
    """★v2：计数清零（文件还在、引用行删光）同样进 stale——两种漂移形态都点名。"""
    root = tmp_path / "repo"
    engine = write_module(root / "lingshu" / "core", "engine.py",
                          "def m(self):\n    from ..world.a import X\n    return X\n")
    res = run_lint("--repo-root", str(root), "--update-baseline")
    assert res.returncode == 0
    engine.write_text("def m(self):\n    return 1\n", encoding="utf-8")   # 引用行删光
    res2 = run_lint("--repo-root", str(root), "--json")
    data = payload(res2)
    assert data["stale_baseline_entries"] == ["core_to_world:lingshu/core/engine.py"], data
