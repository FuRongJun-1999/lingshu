# -*- coding: utf-8 -*-
"""silent_except_lint · 静默宽 except 机械化审计器（P2 批次 F / issue #170 新增）

背景（公开 issue #170「静默吞错是系统性口径」）
------------------------------------------------
在 commit a215d8b 上，全仓 `lingshu/**/*.py` 共 **108 处宽 `except`**
（`except Exception` 或裸 `except`），其中 **81 处无任何 log/warn/raise**——
即"吞掉异常、不记审计、不改窄、不留注释"。这是**系统性口径**而非单点缺陷。

issue 建议的口径（本工具头部登记，供维护者后续整改时对齐）
----------------------------------------------------------
一处宽 `except` 要"不违规"，至少满足下列之一：
  (a) **记可审计错误** —— 在 handler 体内写日志 / 审计 / 上报（`warn` / `log` /
      `audit` / `print` / `raise` 等可检索动作）；
  (b) **改窄到具体异常类型** —— 不用 `except Exception` / 裸 `except`，改捕获
      具体类型（`except KeyError:` / `except (ValueError, TypeError):`）；
  (c) **带注释说明为何允许静默** —— 就地写明静默的正当理由（如"best-effort 清理"）。

★本批（P2 批次 F）只做 **上尺子 ＋ 定基线**：
  · 新增本工具（AST 扫描）与基线文件 `tools/silent_except_lint_baseline.txt`；
  · 新增守卫 `tests/test_silent_except_lint.py`（**计数 > 基线 ⇒ 红**，防新增）；
  · **本批不逐处整改**——114 处静默宽 except 是另一个量级的重构（改窄异常类型
    / 补日志 / 补注释，逐处都要判语义），不在本批范围。工具只保证"不会再变多"。

基线可移植性（★合入其它在飞批次前必读）
----------------------------------------
本基线是在 **worktree `b846086`** 上取的实测值（`151` 宽 / `114` 静默）。基线是
**时间点快照**，不是"永不增长"的承诺。合入其它在飞批次（#243/#248/#183/#299…）
后若计数上升，**必须先判因，再决定动作**：
  · 若是**新增的真缺陷**（新代码吞错、不记审计、未写明理由）⇒ **修代码**（收窄
    异常类型 / 补日志 / 补注释），不得为了让守卫变绿而放宽基线；
  · 若是**刻意静默且已写明理由**（如"读坏档 fail-safe，只吞 json 解析错"）⇒
    **更新基线**并在基线文件「登记面」追加**一行带日期与 commit 的记录**说明
    为何允许——**不要只改数字**。本工具的"登记面"是**文档性**的（见下），
    工具**不**据此豁免（与 `time_core_lint_baseline.txt` 的既有形态一致：
    那里也是「登记 + 带判据注释」，判定仍按计数/位置）。

登记面（documented-exemption surface）
--------------------------------------
基线文件支持「登记面」注释段：用 `#` 注释行登记「某处为何允许静默 / 为何未整改」。
★如实标注：**这些注释不被本工具解析、也不产生豁免**——它们只是人工复核与迁移
判断的依据（`time_core_lint.py` 的基线同样只有「计数/位置登记」这一种强制机制，
没有独立的豁免机制可照搬）。要真正"豁免"某处，只能收窄异常类型或补审计动作，
让工具**数不到**它；或把它写进基线计数（本批即如此）。

检测口径（**照抄 issue 的统计脚本**，为的是基线可与 issue 旧读数对表）
----------------------------------------------------------------------
遍历 AST，对每个 `ast.ExceptHandler`：
  · 宽 except 判定：`n.type is None`（裸 except）或 `ast.unparse(n.type)` 含子串
    `"Exception"`（覆盖 `Exception` / `BaseException` / 元组里含 Exception）；
  · 静默判定：`ast.unparse(n.body)` 中**不含**下列任一子串：
    `warn` / `log` / `print` / `raise` / `audit` / `error`。
命中数按文件聚合：`wide`（宽 except 数）、`silent`（其中静默数）。

已知局限（如实登记）
--------------------
L1 静默判定是**子串启发式**，不是数据流分析：
   · 假阴（把"有记录"判成静默的可能性）：`raise` 出现在 handler 外的同体表达式、
     或 `logging` 经别名（`lg.info(...)` 不含 `log` 子串时）——本口径会**少算**静默；
   · 假阳（把"其实没记录"判成有记录）：handler 体内出现**恰好叫 `errors` 的变量**、
     或 `print` 只出现在字符串里——本口径会**多算**"有记录"。
   故本工具是"可机械复现的下界估计"，不是精确语义判定。
L2 宽 except 的"改窄到具体类型"建议**不**由本工具判定——工具只数"有没有记录动作"，
   不判 handler 捕获的类型是否够窄（那需要逐处人工裁量）。
L3 按**文件**聚合（非行号锚）：基线存"每文件静默数"，同一文件内增删行不会顶漂；
   但同一文件内"消一处、增一处"会相互抵消——故守卫同时校验**总计数**（见守卫）。
L4 解析失败（语法 / 编码）的文件：如实记为 `unparsable` 并在报告中列出，**不计入**
   宽/静默计数（不为了跑通去改被扫文件——见批次 F 铁律 2）。★v2 起 unparsable
   非空 ⇒ **退出 1**（fail-closed）＋ `--update-baseline` 拒写（退出 2）——v1 的
   「不计入 ⇒ 照常退出 0」使「语法错 ＋ N 处静默宽 except」的新文件可整文件逃出
   审计而守卫恒绿，属 #408 同构 fail-open，已收口（见头部「退出码」）。

用法
----
    python tools/silent_except_lint.py [路径...] [--json] [--baseline FILE]
                                       [--update-baseline] [--quiet] [--repo-root PATH]
    默认扫描路径 = <仓库根>/lingshu/**/*.py
    默认 baseline = <仓库根>/tools/silent_except_lint_baseline.txt
    仓库根 = 本脚本上两级目录（由 __file__ 推导，**不依赖 cwd**）
    `--baseline` 的相对路径**按仓库根解析**；绝对路径原样使用。

baseline 格式
-------------
    每行 `<仓内相对路径>\t<静默数>`；`#` 起为注释；空行忽略。
    缺失的文件按 0 计（即"该文件不得新增静默宽 except"）。

退出码（v2 修订——unparsable 由静默放行改为 fail-closed）
--------------------------------------------------------
0  无新增（每文件静默数 ≤ 基线，且总静默数 ≤ 基线总和，且无 unparsable 文件）
1  存在超基线（某文件静默数 > 基线，或出现基线外的新文件），**或存在 unparsable
   文件**——审计面不完整 ⇒ 计数不可信，不得以退出 0 放行（fail-closed；与
   `verify_manifest` 修 #408 的「扫描面被清空不得静默变绿」刻意语义同构）。
   v1 曾对 unparsable 静默退出 0：新文件「语法错 ＋ N 处静默宽 except」可整文件
   逃出审计而守卫恒绿（实测复现：2 处静默＋语法错 ⇒ exit 0 / 计数 0；只修语法错
   ⇒ exit 1）——本版收口。
2  用法 / 路径错误（路径不存在、baseline 格式错、显式 baseline 缺失、
   `--update-baseline` 与 `--json` 同用、**`--update-baseline` 遇 unparsable 文件**
   （拒绝写基线——防该文件既有计数无痕丢失）；argparse 自身错误同样返回 2）
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

LINT_VERSION = 2

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCAN_DIR = "lingshu"
DEFAULT_BASELINE_REL = "tools/silent_except_lint_baseline.txt"

# 判定"有记录动作"的子串（照抄 issue 统计脚本，勿改——基线可对表）
AUDIT_TOKENS: Tuple[str, ...] = ("warn", "log", "print", "raise", "audit", "error")

MAX_SNIPPET = 160

# 「登记面」用户登记段标记（--update-baseline 逐字保留其间内容；工具不解析、不豁免）
REG_BEGIN = "# ── 登记面 · 用户登记（--update-baseline 保留本段逐字）──"
REG_END = "# ── 登记面 · 结束 ──"


# ---------------------------------------------------------------- 单文件扫描
def scan_source(text: str) -> Dict:
    """扫描一段源码，返回 {wide, silent, hits, unparsable}。

    hits 为静默宽 except 的定位（行号 + 异常类型），仅用于人工阅读与 --json 展示；
    基线/守卫按**文件聚合计数**判定（见头部 L3）。
    """
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return {"wide": 0, "silent": 0, "hits": [],
                "unparsable": "SyntaxError: %s" % exc}
    lines = text.split("\n")
    wide = silent = 0
    hits: List[Dict] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        ty = ast.unparse(node.type) if node.type is not None else "bare"
        if not (node.type is None or "Exception" in ty):
            continue
        wide += 1
        body = ast.unparse(node.body)
        if any(tok in body for tok in AUDIT_TOKENS):
            continue
        silent += 1
        ln = getattr(node, "lineno", 0)
        snippet = lines[ln - 1].strip() if 0 < ln <= len(lines) else ""
        if len(snippet) > MAX_SNIPPET:
            snippet = snippet[:MAX_SNIPPET - 3] + "..."
        hits.append({"line": ln, "type": ty, "code_snippet": snippet})
    return {"wide": wide, "silent": silent, "hits": hits, "unparsable": None}


# ---------------------------------------------------------------- baseline
def parse_baseline(text: str) -> Tuple[Dict[str, int], List[str]]:
    """解析 baseline；返回 ({相对路径: 静默数}, 错误列表)。"""
    counts: Dict[str, int] = {}
    errors: List[str] = []
    for no, raw in enumerate(text.split("\n"), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            errors.append("第 %d 行无法解析（期望 <路径> <静默数>）：%s" % (no, raw.strip()))
            continue
        path, num = parts
        if not num.isdigit():
            errors.append("第 %d 行静默数非整数：%s" % (no, raw.strip()))
            continue
        counts[path.replace("\\", "/")] = int(num)
    return counts, errors


def _extract_registration(existing_text: Optional[str]) -> List[str]:
    """从既有基线文本中取出「用户登记段」逐字内容（--update-baseline 保留它）。

    用户登记段由 `REG_BEGIN` 行开启、`REG_END` 行结束；返回其间的行（不含两个标记行）。
    找不到标记时返回空列表（不报错——旧版基线没有该段）。
    """
    if not existing_text:
        return []
    lines = existing_text.split("\n")
    try:
        i = next(k for k, ln in enumerate(lines) if REG_BEGIN in ln)
        j = next(k for k, ln in enumerate(lines) if REG_END in ln and k > i)
    except StopIteration:
        return []
    return lines[i + 1:j]


def render_baseline(counts: Dict[str, int],
                    existing_text: Optional[str] = None) -> str:
    lines = [
        "# silent_except_lint baseline —— 已登记的既有静默宽 except（issue #170）",
        "# 格式：<仓内相对路径><TAB><静默数>   （`#` 起为注释）",
        "# 语义：每文件**允许存在**的静默宽 except 上限；超上限或出现基线外新文件 ⇒ 退出 1。",
        "# 基线来源：HEAD b846086 实测（python -X utf8 tools/silent_except_lint.py --update-baseline）。",
        "#   口径 = issue #170 统计脚本逐字（宽 except=Exception/裸；静默=handler 体无 warn/log/print/raise/audit/error）。",
        "#   两组读数对表：issue 原基线 a215d8b = 108/81；本 worktree b846086 实测 = 151 宽 / 114 静默。",
        "#   （并行批次已改源码的那棵树会更高——本基线取**本工作区** b846086 的值，"
        "不与其它工作区混比。）",
        "#   重生成后请更新上一行的 HEAD 号。",
        "# ★基线可移植性：本值是 b846086 快照。合入其它在飞批次（#243/#248/#183/#299…）后",
        "#   若计数上升，先判因——新增的真缺陷 ⇒ 改代码；刻意静默且已写明理由 ⇒ 更新本基线",
        "#   并在下方「用户登记段」追加一行带日期/commit 的记录（不要只改数字）。",
        "# 口径 / 已知局限 / 退出码见 tools/silent_except_lint.py 头部。",
        "",
        "# ── 登记面（documented-exemption surface；★工具**不解析、不豁免**，仅作文档）──────",
        "# 用途：登记「某处为何允许静默 / 为何未整改」，供人工复核与迁移判断。",
        "# 格式：<日期> <commit> <路径>:<行>  <理由>",
        "# ★强制力声明：本段**不被工具解析、不产生豁免**——它只是文档（time_core_lint 的",
        "#   基线同样只有「计数/位置登记」一种强制机制，无独立豁免机制可照搬）。要让某处",
        "#   真正不再被计数，只能改窄异常类型或补 warn/log/raise/audit 动作。",
        REG_BEGIN,
    ]
    lines.extend(_extract_registration(existing_text))
    lines.append(REG_END)
    lines.append("")
    total = 0
    for path in sorted(counts):
        lines.append("%s\t%d" % (path, counts[path]))
        total += counts[path]
    lines.append("")
    lines.append("# TOTAL %d" % total)
    return "\n".join(lines) + "\n"


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


def collect_targets(paths: Sequence[str], root: Path) -> Tuple[List[Path], List[str]]:
    targets: List[Path] = []
    errors: List[str] = []
    for raw in paths:
        p = Path(raw)
        if not p.exists():
            alt = root / raw
            if alt.exists():
                p = alt
            else:
                errors.append("路径不存在：%s" % raw)
                continue
        if p.is_dir():
            targets.extend(sorted(p.rglob("*.py"), key=lambda q: str(q)))
        else:
            targets.append(p)
    uniq: List[Path] = []
    seen = set()
    for t in targets:
        key = str(t.resolve())
        if key not in seen:
            seen.add(key)
            uniq.append(t)
    return uniq, errors


def repository_scan(root: Path, paths: Sequence[str]) -> Tuple[Dict, List[str]]:
    targets, errors = collect_targets(paths, root)
    if errors:
        return {}, errors
    files: Dict[str, Dict] = {}
    unparsable: List[str] = []
    for path in targets:
        rel = normalize_report_path(path, root)
        try:
            text = _read_source(path)
        except OSError as exc:
            errors.append("无法读取 %s：%s" % (rel, exc))
            continue
        res = scan_source(text)
        if res["unparsable"]:
            unparsable.append("%s: %s" % (rel, res["unparsable"]))
            continue
        if res["wide"] or res["silent"]:
            files[rel] = res
    return {"files": files, "files_scanned": len(targets),
            "unparsable": unparsable}, errors


# ---------------------------------------------------------------- 主流程
def _load_baseline_file(path: Path, explicit: bool
                        ) -> Tuple[Dict[str, int], List[str], bool]:
    if not path.exists():
        if explicit:
            return {}, ["baseline 文件不存在：%s" % path], False
        return {}, [], False
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    counts, errors = parse_baseline(text)
    return counts, errors, True


def run(argv: Optional[Sequence[str]] = None) -> Tuple[int, str, Dict]:
    parser = argparse.ArgumentParser(
        prog="silent_except_lint.py",
        description="静默宽 except 机械化审计器（issue #170；只上尺子＋定基线）")
    parser.add_argument("paths", nargs="*", default=None,
                        help="要扫描的文件或目录（默认 <仓库根>/%s/**/*.py）" % DEFAULT_SCAN_DIR)
    parser.add_argument("--json", action="store_true", help="输出确定性 JSON（sort_keys=True）")
    parser.add_argument("--baseline", metavar="FILE", default=None,
                        help="baseline 文件（默认 <仓库根>/%s）" % DEFAULT_BASELINE_REL)
    parser.add_argument("--update-baseline", action="store_true",
                        help="把当前每文件静默数写回 baseline（不修改源码）")
    parser.add_argument("--quiet", action="store_true", help="只打印超基线项与摘要")
    parser.add_argument("--repo-root", metavar="PATH", default=str(REPO_ROOT),
                        help="仓库根（默认由 __file__ 推导）")
    args = parser.parse_args(argv)

    root = Path(args.repo_root).resolve()
    if not root.is_dir():
        return 2, "仓库根不存在：%s\n" % root, {"exit_code": 2}

    scan_paths = list(args.paths) if args.paths else [str(root / DEFAULT_SCAN_DIR)]
    report, errors = repository_scan(root, scan_paths)
    if errors:
        return 2, "\n".join(errors) + "\n", {"exit_code": 2}

    if args.baseline is None:
        baseline_path = root / DEFAULT_BASELINE_REL
        explicit_baseline = False
    else:
        arg_path = Path(args.baseline)
        baseline_path = arg_path if arg_path.is_absolute() else root / arg_path
        explicit_baseline = True
    baseline_counts, baseline_errors, baseline_found = _load_baseline_file(
        baseline_path, explicit_baseline)
    if baseline_errors:
        return 2, "\n".join(baseline_errors) + "\n", {"exit_code": 2}

    files: Dict[str, Dict] = report["files"]
    baseline_total = sum(baseline_counts.values())
    actual_total = sum(v["silent"] for v in files.values())

    per_file: List[Dict] = []
    over: List[Dict] = []
    for rel in sorted(files):
        silent = files[rel]["silent"]
        base = baseline_counts.get(rel, 0)
        rec = {"file": rel, "silent": silent, "wide": files[rel]["wide"],
               "baseline": base, "excess": silent - base,
               "hits": sorted(files[rel]["hits"], key=lambda h: h["line"])}
        per_file.append(rec)
        if silent > base:
            over.append({"file": rel, "silent": silent, "baseline": base,
                         "excess": silent - base})
    # 基线里登记了但本次扫描不到的路径：不算违规，仅提示（可能是文件被删/改名）
    missing = sorted(p for p in baseline_counts if p not in files)

    # v2 fail-closed（见头部「退出码」）：unparsable ⇒ 审计面不完整，不得以 0 放行。
    unparsable = report["unparsable"]
    exit_code = 1 if (over or unparsable) else 0
    payload = {
        "actual_total": actual_total,
        "baseline_file": normalize_report_path(baseline_path, root),
        "baseline_found": baseline_found,
        "baseline_total": baseline_total,
        "exit_code": exit_code,
        "files_scanned": report["files_scanned"],
        "new_over_baseline": over,
        "over_total": actual_total - baseline_total if actual_total > baseline_total else 0,
        "per_file": per_file,
        "stale_baseline_entries": missing,
        "summary": {
            "files_with_wide": len(files),
            "new": len(over),
            "silent_total": actual_total,
            "wide_total": sum(v["wide"] for v in files.values()),
        },
        "unparsable": unparsable,
        "version": LINT_VERSION,
    }

    if args.update_baseline:
        if args.json:
            return 2, "--update-baseline 不能与 --json 同用\n", {"exit_code": 2}
        if unparsable:
            # 拒写（v2）：unparsable 文件数不进 counts，写盘会把它的既有基线计数
            # 无痕丢掉——先修复解析失败（语法错/解释器版本差异），再重取基线。
            return 2, ("存在 unparsable 文件，拒绝写基线（防计数无痕丢失）：\n  "
                       + "\n  ".join(unparsable)
                       + "\n"), {"exit_code": 2}
        counts = {rel: files[rel]["silent"] for rel in sorted(files)}
        prev = (baseline_path.read_text(encoding="utf-8-sig", errors="replace")
                if baseline_path.exists() else None)
        text = render_baseline(counts, existing_text=prev)
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(text, encoding="utf-8")
        msg = "" if args.quiet else (
            "已写入 %d 个文件的静默数（合计 %d）→ %s\n"
            % (len(counts), actual_total, baseline_path))
        return 0, msg, payload

    if args.json:
        return exit_code, json.dumps(payload, sort_keys=True,
                                     ensure_ascii=False, indent=2) + "\n", payload

    lines: List[str] = []
    if not baseline_found and not args.quiet:
        lines.append("提示：baseline 不存在（%s），本次按空基线运行——所有静默命中都算新增"
                     % normalize_report_path(baseline_path, root))
    if not args.quiet:
        lines.append("%-52s %6s %6s %8s" % ("文件", "宽", "静默", "基线"))
        for rec in per_file:
            lines.append("%-52s %6d %6d %8d%s"
                         % (rec["file"], rec["wide"], rec["silent"], rec["baseline"],
                            "  ← 超基线 +%d" % rec["excess"] if rec["excess"] > 0 else ""))
    for rec in over:
        lines.append("[新] %s：静默 %d > 基线 %d（超 %d）"
                     % (rec["file"], rec["silent"], rec["baseline"], rec["excess"]))
    if unparsable:
        lines.append("[不可判] 解析失败 %d 个（退出 1 的原因——审计面不完整，v2 fail-closed）："
                     % len(unparsable))
        lines.extend("  " + u for u in unparsable)
    if not args.quiet:
        lines.append("")
        lines.append("扫描 %d 个文件；静默宽 except 合计 %d（基线 %d）；超基线文件 %d"
                     % (report["files_scanned"], actual_total, baseline_total, len(over)))
        why = "存在超基线" if over else ("存在 unparsable 文件" if unparsable else "无新增")
        lines.append("退出码 %d（%s）" % (exit_code, why))
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
