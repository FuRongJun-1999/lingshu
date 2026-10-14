# -*- coding: utf-8 -*-
"""import_direction_lint · core↔world 双向依赖机械化审计器（P2 批次 F / issue #178 新增）

背景（公开 issue #178「架构/分层：`core` 与 `world` 双向依赖」）
----------------------------------------------------------------
理想分层是单向的：`world` 依赖 `core`，`core` 不应反向依赖 `world`。实测本仓
**双向依赖**：

  · 正方向（`core` → `world`）：`lingshu/core/core.py` 的 `SpacetimeMemoryEngine`
    内含多个**世界层外观方法**（在 issue 原基线约 780 行），方法体内以
    `from ..world.xxx import ...` **惰性导入**世界层模块——即 core 反向依赖 world。
  · 反方向（`world` → `core` 旧名）：`lingshu/world/prediction.py` 用**旧包名**
    `from spacetime_memory_core import LayeredStore`（该名在本仓**不存在**，
    是私域 AEIS 的命名空间别名；本仓形态靠 `except` 回落仓内 `..core.core`）。

issue 的**终局目标**：`grep -n "from ..world" lingshu/core/` 为**空**——即把 core
里的世界层外观方法全部迁到 `lingshu/facade/world_api.py`（外观层），core 只保留
引擎本体，依赖方向回到单向。

★★本批（P2 批次 F）**只做机械可判的那一半**——「上尺子 ＋ 定基线（防新增）」：
  · 新增本工具与基线 `tools/import_direction_lint_baseline.txt`（基线＝本次实测计数）；
  · 新增守卫 `tests/test_import_direction_lint.py`（**计数 > 基线 ⇒ 红**）；
  · **11 个世界层外观方法迁到 `lingshu/facade/world_api.py` 的重构未做**——
    那是搬运约 780 行代码的大重构，属另一件且需设计者裁方向，**不在本批**。
  · `world/prediction.py` 的旧名 `spacetime_memory_core`：本工具**如实计数**
    （HEAD b846086 实测该引用**仍在**，见基线）；是否清除留待另一件。

检测口径
--------
两个方向、按文件聚合行计数（行级正则，宽松下界）：

  DIR "core_to_world"（正方向，issue 点名）：
      扫描 `lingshu/core/*.py`，计**行首**为 `from ..world` 或 `import ..world`
      的行（`^\\s*(from|import)\\s+\\.\\.world`）。一行计 1。
  DIR "stmc_refs"（反方向旧名）：
      扫描 `lingshu/**/*.py`，计含子串 `spacetime_memory_core` 的行（含注释/字符串，
      因为旧名出现本身就是分层/命名面的事实）。一行计 1。

已知局限（如实登记）
--------------------
L1 行级正则、非 AST：把 `from ..world` 写在字符串/注释里也会计数（假阳）；
   `from .. world import`（空格）等异常写法漏报（假阴）。核心用法（逐方法体
   `from ..world.x import Y`）两种口径一致。
L2 只数**行数**，不数"外观方法个数"：issue 说的"11 个外观方法"需人工按方法边界
   归属，本工具不自动判定（那需要 AST 方法归属 + 外观语义判定）。
L3 `stmc_refs` 把注释里的提及也计入（如 `core.py` 文件头 docstring、
   `prediction.py` 的注释）——因为本工具要防的是"旧名是否还在仓里出现"，
   注释也算出现；若要区分代码/注释需另加 AST 口径（本批从简）。
L4 基线按 `"<dir>:<file>" -> 行数` 登记；同文件内"删一行、加一行"会相互抵消——
   故守卫同时校验**总计数**（见守卫）与逐文件计数。

用法
----
    python tools/import_direction_lint.py [--json] [--baseline FILE]
                                          [--update-baseline] [--quiet] [--repo-root PATH]
    默认扫描根 = <仓库根>/lingshu
    默认 baseline = <仓库根>/tools/import_direction_lint_baseline.txt
    仓库根 = 本脚本上两级目录（由 __file__ 推导，**不依赖 cwd**）
    `--baseline` 的相对路径**按仓库根解析**；绝对路径原样使用。

baseline 格式
-------------
    每行 `<方向>\t<仓内相对路径>\t<行数>`；`#` 起为注释；空行忽略。
    方向 ∈ {core_to_world, stmc_refs}。缺失的 (方向,文件) 按 0 计（不得新增）。

退出码（冻结）
--------------
0  无新增（每 (方向,文件) 行数 ≤ 基线）
1  存在超基线（某 (方向,文件) 行数 > 基线，或出现基线外的新计数）
2  用法 / 路径错误（扫描根不存在、baseline 格式错、显式 baseline 缺失、
   `--update-baseline` 与 `--json` 同用；argparse 自身错误同样返回 2）
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

LINT_VERSION = 1

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCAN_DIR = "lingshu"
DEFAULT_BASELINE_REL = "tools/import_direction_lint_baseline.txt"

# 「登记面」用户登记段标记（--update-baseline 逐字保留其间内容；工具不解析、不豁免）
REG_BEGIN = "# ── 登记面 · 用户登记（--update-baseline 保留本段逐字）──"
REG_END = "# ── 登记面 · 结束 ──"

DIR_CORE_TO_WORLD = "core_to_world"
DIR_STMC_REFS = "stmc_refs"
DIRECTIONS: Tuple[str, ...] = (DIR_CORE_TO_WORLD, DIR_STMC_REFS)

# 正方向：行首 `from ..world` / `import ..world`
RE_CORE_TO_WORLD = re.compile(r"^\s*(?:from|import)\s+\.\.world\b")
# 反方向旧名
STMC_TOKEN = "spacetime_memory_core"


# ---------------------------------------------------------------- 扫描
def _read_source(path: Path) -> str:
    data = path.read_bytes()
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace")


def normalize_report_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return Path(os.path.abspath(str(path))).as_posix()


def scan(root: Path, scan_dir: Path) -> Tuple[Dict[str, Dict[str, int]], List[str]]:
    """返回 ({方向: {相对路径: 行数}}, 错误列表)。"""
    errors: List[str] = []
    if not scan_dir.is_dir():
        errors.append("扫描根不存在：%s" % scan_dir)
        return {}, errors
    out: Dict[str, Dict[str, int]] = {d: {} for d in DIRECTIONS}
    core_dir = scan_dir / "core"
    for path in sorted(scan_dir.rglob("*.py"), key=lambda q: str(q)):
        rel = normalize_report_path(path, root)
        try:
            text = _read_source(path)
        except OSError as exc:
            errors.append("无法读取 %s：%s" % (rel, exc))
            continue
        # stmc_refs：全扫描根
        n = sum(1 for line in text.splitlines() if STMC_TOKEN in line)
        if n:
            out[DIR_STMC_REFS][rel] = n
        # core_to_world：仅 lingshu/core/ 下
        if core_dir.is_dir():
            try:
                path.resolve().relative_to(core_dir.resolve())
            except ValueError:
                continue
            m = sum(1 for line in text.splitlines() if RE_CORE_TO_WORLD.search(line))
            if m:
                out[DIR_CORE_TO_WORLD][rel] = m
    return out, errors


# ---------------------------------------------------------------- baseline
def parse_baseline(text: str) -> Tuple[Dict[str, int], List[str]]:
    entries: Dict[str, int] = {}
    errors: List[str] = []
    for no, raw in enumerate(text.split("\n"), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 3:
            errors.append("第 %d 行无法解析（期望 <方向> <路径> <行数>）：%s" % (no, raw.strip()))
            continue
        direction, path, num = parts
        if direction not in DIRECTIONS:
            errors.append("第 %d 行方向未知（应为 %s）：%s"
                          % (no, "/".join(DIRECTIONS), raw.strip()))
            continue
        if not num.isdigit():
            errors.append("第 %d 行行数非整数：%s" % (no, raw.strip()))
            continue
        entries["%s:%s" % (direction, path.replace("\\", "/"))] = int(num)
    return entries, errors


def _extract_registration(existing_text: Optional[str]) -> List[str]:
    """取出「用户登记段」逐字内容（--update-baseline 保留）；无标记则返回空。"""
    if not existing_text:
        return []
    lines = existing_text.split("\n")
    try:
        i = next(k for k, ln in enumerate(lines) if REG_BEGIN in ln)
        j = next(k for k, ln in enumerate(lines) if REG_END in ln and k > i)
    except StopIteration:
        return []
    return lines[i + 1:j]


def render_baseline(counts: Dict[str, Dict[str, int]],
                    existing_text: Optional[str] = None) -> str:
    lines = [
        "# import_direction_lint baseline —— core↔world 双向依赖计数（issue #178）",
        "# 格式：<方向><TAB><仓内相对路径><TAB><行数>   （`#` 起为注释）",
        "# 方向：core_to_world = lingshu/core/ 里 `from ..world`/`import ..world` 的行数；",
        "#       stmc_refs     = lingshu/ 里含旧包名 `spacetime_memory_core` 的行数。",
        "# 语义：每 (方向,文件) 行数**只准减不准增**；超基线或出现基线外新计数 ⇒ 退出 1。",
        "# 基线来源：HEAD b846086 实测（python -X utf8 tools/import_direction_lint.py --update-baseline）。",
        "#   两组读数对表：issue 原实测 = core→world 约 22 处惰性导入；本 worktree b846086 实测 = 28 行（全在 core.py）。",
        "#   stmc_refs 实测 = 4 行（component_resolver.py:1 / core.py:1 / world/prediction.py:2）——旧名仍在，未清。",
        "#   ★终局目标（issue #178）= `grep -n \"from ..world\" lingshu/core/` 为空——**本批达不到**：",
        "#     本批只上尺子＋定基线（防新增）；11 个外观方法迁 lingshu/facade/world_api.py 的重构未做。",
        "# ★基线可移植性：本值是 b846086 快照。合入其它在飞批次后若计数上升，先判因——",
        "#   真·新增反向依赖 ⇒ 改代码；刻意为之 ⇒ 更新基线并在下方登记段追加一行带日期记录。",
        "#   重生成后请更新上一行 HEAD 号。口径 / 已知局限 / 退出码见工具头部。",
        "",
        "# ── 登记面（documented-exemption surface；★工具**不解析、不豁免**，仅作文档）──────",
        "# 格式：<日期> <commit> <方向> <路径>  <理由>；本段不被工具解析，仅作文档。",
        REG_BEGIN,
    ]
    lines.extend(_extract_registration(existing_text))
    lines.append(REG_END)
    lines.append("")
    rows: List[Tuple[str, str, int]] = []
    for direction in DIRECTIONS:
        for rel in sorted(counts.get(direction, {})):
            rows.append((direction, rel, counts[direction][rel]))
    for direction, rel, n in sorted(rows):
        lines.append("%s\t%s\t%d" % (direction, rel, n))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- 主流程
def _load_baseline_file(path: Path, explicit: bool
                        ) -> Tuple[Dict[str, int], List[str], bool]:
    if not path.exists():
        if explicit:
            return {}, ["baseline 文件不存在：%s" % path], False
        return {}, [], False
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    entries, errors = parse_baseline(text)
    return entries, errors, True


def run(argv: Optional[Sequence[str]] = None) -> Tuple[int, str, Dict]:
    parser = argparse.ArgumentParser(
        prog="import_direction_lint.py",
        description="core↔world 双向依赖审计器（issue #178；只上尺子＋定基线）")
    parser.add_argument("--json", action="store_true", help="输出确定性 JSON（sort_keys=True）")
    parser.add_argument("--baseline", metavar="FILE", default=None,
                        help="baseline 文件（默认 <仓库根>/%s）" % DEFAULT_BASELINE_REL)
    parser.add_argument("--update-baseline", action="store_true",
                        help="把当前双向依赖计数写回 baseline")
    parser.add_argument("--quiet", action="store_true", help="只打印超基线项与摘要")
    parser.add_argument("--repo-root", metavar="PATH", default=str(REPO_ROOT),
                        help="仓库根（默认由 __file__ 推导）")
    parser.add_argument("--scan-dir", metavar="PATH", default=None,
                        help="扫描根（默认 <仓库根>/%s）" % DEFAULT_SCAN_DIR)
    args = parser.parse_args(argv)

    root = Path(args.repo_root).resolve()
    if not root.is_dir():
        return 2, "仓库根不存在：%s\n" % root, {"exit_code": 2}
    scan_dir = Path(args.scan_dir) if args.scan_dir else root / DEFAULT_SCAN_DIR

    counts, errors = scan(root, scan_dir)
    if errors:
        return 2, "\n".join(errors) + "\n", {"exit_code": 2}

    if args.baseline is None:
        baseline_path = root / DEFAULT_BASELINE_REL
        explicit_baseline = False
    else:
        arg_path = Path(args.baseline)
        baseline_path = arg_path if arg_path.is_absolute() else root / arg_path
        explicit_baseline = True
    baseline_entries, baseline_errors, baseline_found = _load_baseline_file(
        baseline_path, explicit_baseline)
    if baseline_errors:
        return 2, "\n".join(baseline_errors) + "\n", {"exit_code": 2}

    over: List[Dict] = []
    per_key: List[Dict] = []
    for direction in DIRECTIONS:
        for rel in sorted(counts.get(direction, {})):
            actual = counts[direction][rel]
            base = baseline_entries.get("%s:%s" % (direction, rel), 0)
            per_key.append({"direction": direction, "file": rel,
                            "count": actual, "baseline": base,
                            "excess": actual - base})
            if actual > base:
                over.append({"direction": direction, "file": rel,
                             "count": actual, "baseline": base,
                             "excess": actual - base})

    def total(direction: str) -> int:
        return sum(counts.get(direction, {}).values())

    baseline_total = {d: sum(v for k, v in baseline_entries.items() if k.startswith(d + ":"))
                      for d in DIRECTIONS}
    actual_total = {d: total(d) for d in DIRECTIONS}
    # stale = 基线里登记、本次扫描已数不到的 (方向,文件)——文件被删/改名或计数清零。
    # （v1 误写成「方向前缀 ∉ DIRECTIONS」——该条件被 parse_baseline 的方向校验
    #   提前挡死，恒为空；姊妹件 silent_except / coverage_map 均为「按实测键反查」
    #   的正确形态，此处对齐。）
    actual_keys = {"%s:%s" % (rec["direction"], rec["file"]) for rec in per_key}
    missing = sorted(k for k in baseline_entries if k not in actual_keys)

    exit_code = 1 if over else 0
    payload = {
        "baseline_file": normalize_report_path(baseline_path, root),
        "baseline_found": baseline_found,
        "baseline_total": baseline_total,
        "core_to_world_total": actual_total[DIR_CORE_TO_WORLD],
        "exit_code": exit_code,
        "new_over_baseline": over,
        "per_key": per_key,
        "stale_baseline_entries": missing,
        "stmc_refs_total": actual_total[DIR_STMC_REFS],
        "summary": {
            "core_to_world": actual_total[DIR_CORE_TO_WORLD],
            "new": len(over),
            "stmc_refs": actual_total[DIR_STMC_REFS],
        },
        "version": LINT_VERSION,
    }

    if args.update_baseline:
        if args.json:
            return 2, "--update-baseline 不能与 --json 同用\n", {"exit_code": 2}
        text = render_baseline(
            counts,
            existing_text=(baseline_path.read_text(encoding="utf-8-sig", errors="replace")
                           if baseline_path.exists() else None))
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(text, encoding="utf-8")
        msg = "" if args.quiet else (
            "已写入 core_to_world=%d / stmc_refs=%d → %s\n"
            % (actual_total[DIR_CORE_TO_WORLD], actual_total[DIR_STMC_REFS], baseline_path))
        return 0, msg, payload

    if args.json:
        return exit_code, json.dumps(payload, sort_keys=True,
                                     ensure_ascii=False, indent=2) + "\n", payload

    lines: List[str] = []
    if not baseline_found and not args.quiet:
        lines.append("提示：baseline 不存在（%s），本次按空基线运行——所有计数都算新增"
                     % normalize_report_path(baseline_path, root))
    if not args.quiet:
        lines.append("%-16s %-44s %6s %6s" % ("方向", "文件", "计数", "基线"))
        for rec in per_key:
            lines.append("%-16s %-44s %6d %6d%s"
                         % (rec["direction"], rec["file"], rec["count"], rec["baseline"],
                            "  ← 超基线 +%d" % rec["excess"] if rec["excess"] > 0 else ""))
    for rec in over:
        lines.append("[新] %s %s：%d > 基线 %d（超 %d）"
                     % (rec["direction"], rec["file"], rec["count"],
                        rec["baseline"], rec["excess"]))
    if missing and not args.quiet:
        lines.append("基线中已失效条目 %d 个（可清理）：%s" % (len(missing), ", ".join(missing)))
    if not args.quiet:
        lines.append("")
        lines.append("core→world 惰性导入 %d 行（基线 %d）；旧名 stmc_refs %d 行（基线 %d）；超基线项 %d"
                     % (actual_total[DIR_CORE_TO_WORLD], baseline_total[DIR_CORE_TO_WORLD],
                        actual_total[DIR_STMC_REFS], baseline_total[DIR_STMC_REFS], len(over)))
        lines.append("★终局目标（grep \"from ..world\" lingshu/core/ 为空）本批未达成——只上尺子防新增。")
        lines.append("退出码 %d（%s）" % (exit_code, "存在超基线" if over else "无新增"))
    return exit_code, "\n".join(lines) + ("\n" if lines else ""), payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", newline="\n")
        except Exception:      # pragma: no cover - 平台相关
            pass
    code, out, _payload = run(argv)
    if out:
        sys.stdout.write(out)
    return code


if __name__ == "__main__":
    sys.exit(main())
