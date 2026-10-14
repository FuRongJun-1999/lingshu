# -*- coding: utf-8 -*-
"""批次 F / issue #170 · `tools/silent_except_lint.py` 的验收测试（守卫）

本件机制：**静默宽 except 计数只准减、不准增**。
  · 逐文件静默数 > 基线 ⇒ 退出 1（红）；
  · 出现基线之外、且带静默宽 except 的新文件 ⇒ 退出 1（红）；
  · 基线内的既有静默允许存在（本批**不**逐处整改，只上尺子）。

断言组
------
A 量具自证（lint 必须真在数）：
   · 合成「静默宽 except」源 ⇒ 必被数出（silent ≥ 1）；
   · 合成「有 raise / log」的宽 except ⇒ 不被数（silent == 0）；
   · 合成「窄 except」⇒ 不算宽 except（wide == 0）。
B 真实仓库（出货基线）⇒ 退出码 0、无超基线项、合计 = 基线合计。
C 定点变异（把基线数字改小一格 ⇒ 守卫必红）：用**临时副本基线**模拟，
   不改出货基线文件；断言「静默数 > 基线」时退出 1 且点名该文件。
D 基线外新文件带静默宽 except ⇒ 退出 1。
E unparsable fail-closed（v2 · #408 同构收口）：语法错文件 ⇒ 退出 1；
   --update-baseline 遇 unparsable ⇒ 退出 2 且不写盘（防计数无痕丢失）。

判据来源：公开 issue #170「建议 1/2」＋ 本仓 tools/time_core_lint.py 先例
（同款 `--baseline` / `--update-baseline` / 退出码语义）。

本文件只读仓库源码、只写 tmp_path；不改 lingshu/ 下任何文件。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "tools" / "silent_except_lint.py"
BASELINE = REPO_ROOT / "tools" / "silent_except_lint_baseline.txt"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import silent_except_lint as S  # noqa: E402


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
def test_silent_wide_except_is_counted():
    """合成静默宽 except ⇒ 必被数出。"""
    src = (
        "def f():\n"
        "    try:\n"
        "        g()\n"
        "    except Exception:\n"
        "        pass\n"
    )
    res = S.scan_source(src)
    assert res["wide"] == 1, res
    assert res["silent"] == 1, res


def test_bare_except_is_wide_and_counted():
    src = (
        "def f():\n"
        "    try:\n"
        "        g()\n"
        "    except:\n"
        "        return None\n"
    )
    res = S.scan_source(src)
    assert res["wide"] == 1 and res["silent"] == 1, res


def test_audited_wide_except_not_counted():
    """宽 except 但体内有 raise / log / warn ⇒ 不算静默。"""
    for token in ("raise", "logging.warning", "log.error", "audit"):
        src = ("def f():\n"
               "    try:\n"
               "        g()\n"
               "    except Exception:\n"
               "        %s('x')\n" % token)
        res = S.scan_source(src)
        assert res["wide"] == 1, (token, res)
        assert res["silent"] == 0, (token, res)


def test_narrow_except_not_wide():
    """窄 except 不计入宽 except。"""
    src = (
        "def f():\n"
        "    try:\n"
        "        g()\n"
        "    except KeyError:\n"
        "        pass\n"
    )
    res = S.scan_source(src)
    assert res["wide"] == 0 and res["silent"] == 0, res


# --------------------------------------------------- B 真实仓库：出货基线绿
def test_real_repo_shipped_baseline_exit_zero():
    res = run_lint("--json")
    data = payload(res)
    assert res.returncode == 0, data
    assert data["new_over_baseline"] == []
    assert data["actual_total"] == data["baseline_total"], data
    assert data["summary"]["new"] == 0
    # v2 fail-closed 的在仓面：出货树必须零 unparsable（否则审计面不完整仍会绿）
    assert data["unparsable"] == [], data
    # 基线里确实登记了真实命中（否则规则可能失效）
    assert data["baseline_total"] >= 1, "基线里没有任何真实命中，规则可能失效"


def test_shipped_baseline_file_has_provenance_and_entries():
    text = BASELINE.read_text(encoding="utf-8")
    assert "基线来源" in text and "HEAD" in text
    counts, errors = S.parse_baseline(text)
    assert errors == []
    assert counts, "基线清单为空"
    assert sum(counts.values()) >= 1


# --------------------------------------------------- C 定点变异：基线改小一格 ⇒ 红
def test_mutation_shrink_baseline_by_one_turns_red(tmp_path):
    """★ 定点变异：把某文件基线静默数改小一格 ⇒ 守卫必红并点名该文件。

    用**临时副本基线**模拟（不动出货基线）；对真实仓库跑一次空/收缩基线，
    断言退出码 1 且该文件出现在 new_over_baseline。
    """
    data0 = payload(run_lint("--json"))
    assert data0["exit_code"] == 0
    # 找一个静默数 >= 1 的文件
    target = next(r for r in data0["per_file"] if r["silent"] >= 1)
    rel, silent = target["file"], target["silent"]

    lines = []
    for r in data0["per_file"]:
        base = r["silent"]
        if r["file"] == rel:
            base -= 1                       # ← 变异：改小一格
        lines.append("%s\t%d" % (r["file"], base))
    mutated = tmp_path / "shrunk_baseline.txt"
    mutated.write_text("\n".join(lines) + "\n", encoding="utf-8")

    res = run_lint("--baseline", str(mutated), "--json")
    data = payload(res)
    assert res.returncode == 1, "基线改小一格后守卫竟然没红：%r" % data
    over = {(o["file"], o["excess"]) for o in data["new_over_baseline"]}
    assert (rel, 1) in over, (rel, over)


def test_mutation_empty_baseline_turns_red(tmp_path):
    """空基线 ⇒ 全部静默命中都算新增 ⇒ 退出 1。"""
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    res = run_lint("--baseline", str(empty))
    assert res.returncode == 1
    assert "超基线" in res.stdout.decode("utf-8")


# --------------------------------------------------- D 基线外新文件 ⇒ 红
def test_new_file_with_silent_except_turns_red(tmp_path):
    root = tmp_path / "repo"
    write_module(root / "lingshu" / "pkg", "mod.py",
                 "def f():\n"
                 "    try:\n"
                 "        g()\n"
                 "    except Exception:\n"
                 "        pass\n")
    # 空基线：新文件带静默 ⇒ 红
    res = run_lint(str(root / "lingshu"), "--repo-root", str(root),
                   "--baseline", str(tmp_path / "none.txt"))
    assert res.returncode == 2  # 显式 baseline 不存在 = 用法错误
    baseline = root / "tools" / "silent_except_lint_baseline.txt"
    baseline.parent.mkdir(parents=True, exist_ok=True)
    baseline.write_text("", encoding="utf-8")
    res2 = run_lint(str(root / "lingshu"), "--repo-root", str(root), "--json")
    data2 = payload(res2)
    assert res2.returncode == 1, data2
    assert any(o["file"] == "lingshu/pkg/mod.py" for o in data2["new_over_baseline"])


# --------------------------------------------------- baseline 解析 / 更新往返
def test_baseline_parse_and_render_roundtrip():
    counts = {"lingshu/a.py": 3, "lingshu/b.py": 0}
    text = S.render_baseline(counts)
    parsed, errors = S.parse_baseline(text)
    assert errors == []
    assert parsed == counts


def test_update_baseline_roundtrip(tmp_path):
    root = tmp_path / "repo"
    write_module(root / "lingshu" / "pkg", "mod.py",
                 "def f():\n"
                 "    try:\n"
                 "        g()\n"
                 "    except Exception:\n"
                 "        pass\n")
    res = run_lint(str(root / "lingshu"), "--repo-root", str(root), "--update-baseline")
    assert res.returncode == 0
    baseline = root / "tools" / "silent_except_lint_baseline.txt"
    assert baseline.exists()
    assert "lingshu/pkg/mod.py" in baseline.read_text(encoding="utf-8")
    res2 = run_lint(str(root / "lingshu"), "--repo-root", str(root))
    assert res2.returncode == 0


def test_usage_errors_return_two(tmp_path):
    assert run_lint("no/such/path").returncode == 2
    assert run_lint("--baseline", str(tmp_path / "nope.txt")).returncode == 2
    bad = tmp_path / "bad.txt"
    bad.write_text("这不是一条合法条目\n", encoding="utf-8")
    assert run_lint("--baseline", str(bad)).returncode == 2
    assert run_lint("--update-baseline", "--json").returncode == 2
    assert run_lint("--repo-root", str(tmp_path / "no-root")).returncode == 2


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
    write_module(root / "lingshu" / "pkg", "mod.py",
                 "def f():\n"
                 "    try:\n"
                 "        g()\n"
                 "    except Exception:\n"
                 "        pass\n")
    baseline = root / "tools" / "silent_except_lint_baseline.txt"
    baseline.parent.mkdir(parents=True, exist_ok=True)
    baseline.write_text(
        "%s\n%s\n%s\n" % (S.REG_BEGIN, "# 2026-10-11 deadbee lingshu/pkg/mod.py:4  刻意的 fail-safe",
                          S.REG_END),
        encoding="utf-8")
    res = run_lint(str(root / "lingshu"), "--repo-root", str(root), "--update-baseline")
    assert res.returncode == 0
    text = baseline.read_text(encoding="utf-8")
    assert "# 2026-10-11 deadbee lingshu/pkg/mod.py:4  刻意的 fail-safe" in text, text
    # 计数行仍在（登记段不影响判定）
    assert "lingshu/pkg/mod.py\t1" in text, text


def test_registration_surface_has_no_force():
    """★如实标注：登记面**不被工具解析、不产生豁免**（只是注释，不进 parse_baseline）。"""
    text = (S.REG_BEGIN + "\n"
            + "# 2026-10-11 deadbee lingshu/x.py:1  允许静默\n"
            + S.REG_END + "\n")
    counts, errors = S.parse_baseline(text)
    assert errors == []
    assert counts == {}, "登记面注释不应被 parse_baseline 当作计数条目"


# ------------------------------------- E unparsable fail-closed（v2 · #408 同构收口）
def test_unparsable_file_fails_closed(tmp_path):
    """★v2：unparsable 文件 ⇒ 退出 1——「语法错 ＋ N 处静默宽 except」不得整文件逃出审计。

    v1 形态实测：同样输入退出 0（绿）、计数 0——fail-open（verify_manifest #408
    的同构残留）。本断言即定点变异：把 v2 的 fail-closed 撤掉 ⇒ 本条必红。
    """
    root = tmp_path / "repo"
    write_module(root / "lingshu" / "pkg", "newmod.py",
                 "def f():\n"
                 "    try:\n"
                 "        g()\n"
                 "    except Exception:\n"
                 "        pass\n"
                 "\n"
                 "def broken(:\n"          # ← 语法错：ast.parse 失败 ⇒ 整文件 unparsable
                 "    pass\n")
    baseline = root / "tools" / "silent_except_lint_baseline.txt"
    baseline.parent.mkdir(parents=True, exist_ok=True)
    baseline.write_text("", encoding="utf-8")
    res = run_lint(str(root / "lingshu"), "--repo-root", str(root), "--json")
    data = payload(res)
    assert res.returncode == 1, ("unparsable 文件必须 fail-closed（退出 1）", data)
    assert data["unparsable"] and "newmod.py" in data["unparsable"][0], data
    assert data["summary"]["silent_total"] == 0, data   # 计数不含该文件（口径不变）
    # 文本模式必须把「不可判」点名为退出 1 的原因（可读性）
    res_text = run_lint(str(root / "lingshu"), "--repo-root", str(root), "--quiet")
    assert "不可判" in res_text.stdout.decode("utf-8"), res_text.stdout


def test_unparsable_update_baseline_refuses_to_write(tmp_path):
    """★v2：--update-baseline 遇 unparsable ⇒ 退出 2 且**不写盘**（防计数无痕丢失）。

    v1 形态实测：静默把 unparsable 文件从基线里丢掉、退出 0。本断言即定点变异。
    """
    root = tmp_path / "repo"
    write_module(root / "lingshu" / "pkg", "good.py",
                 "def f():\n"
                 "    try:\n"
                 "        g()\n"
                 "    except Exception:\n"
                 "        pass\n")
    write_module(root / "lingshu" / "pkg", "bad.py", "def broken(:\n    pass\n")
    baseline = root / "tools" / "silent_except_lint_baseline.txt"
    baseline.parent.mkdir(parents=True, exist_ok=True)
    baseline.write_text("lingshu/pkg/good.py\t1\n", encoding="utf-8")   # 既有基线
    res = run_lint(str(root / "lingshu"), "--repo-root", str(root), "--update-baseline")
    assert res.returncode == 2, res.stdout + res.stderr
    # 基线原样未动（good.py 的计数 1 仍在，未被静默重写/清空）
    assert baseline.read_text(encoding="utf-8") == "lingshu/pkg/good.py\t1\n", \
        baseline.read_text(encoding="utf-8")
