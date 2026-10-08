#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_pr_title — 校验 PR 标题符合本仓提交约定。纯标准库；不写任何文件。

约定（与仓内既有提交同款）：`type(scope): 摘要`
  · type    ∈ ALLOWED_TYPES（下表）
  · scope   可选；给出时必须非空且不含括号/空白（如 `docs(deps): …`）
  · `!`     可选；紧跟 scope/type 之后，表示破坏性变更（如 `feat(world)!: …`）
  · `: `    冒号后至少一个空格，再接**非空**摘要
  · 首尾不得有空白；标题建议 ≤ 160 字符（超长只告警，不判负）

用法（argv）：
    python -X utf8 scripts/check_pr_title.py "<PR 标题>"
    python -X utf8 scripts/check_pr_title.py --title "<PR 标题>"
    python -X utf8 scripts/check_pr_title.py                 # 读环境变量 PR_TITLE
    python -X utf8 scripts/check_pr_title.py --list-types     # 打印允许的 type
    python -X utf8 scripts/check_pr_title.py --selftest       # 自检（正负例）

退出码：0 = 合规；1 = 不合规；2 = 用法错误（缺标题 / 参数非法）。
"""
from __future__ import annotations

import os
import re
import sys

#: 允许的 type 白名单。
#:
#: 来源①「本仓 git log 实有前缀」（2026-10-08 实测统计；`git log --pretty=%s` 前缀计数）：
#:     docs×10 · chore×4 · fix×2 · tools×2 · code×2 · feat×1 · init×1 · theory×1
#:     ⇒ {init, feat, fix, docs, chore, tools, code, theory}
#: 来源② 通用 Conventional Commits 型（本仓 log 暂无实例，但为向后兼容预留——
#:     「A 档＝CI 加固」自身的 `ci:` 提交、以及后续新增 test/perf/refactor 均需可用）：
#:     {test, perf, refactor, style, build, ci, revert}
#:
#: ⚠️ **设计者裁定点**：若只要「本仓实有」的窄口径，删掉来源②七个即可。
#:    （见报告「落仓前须知」。）
ALLOWED_TYPES = (
    # 来源① —— 本仓 git log 实有
    "init", "feat", "fix", "docs", "chore", "tools", "code", "theory",
    # 来源② —— 通用型预留
    "test", "perf", "refactor", "style", "build", "ci", "revert",
)

MAX_LEN_WARN = 160          # 超过仅告警；阈值取 160 —— 仓内既有标题最长 154 字符（2026-10-08 实测）⇒ 不误报既有风格
MAX_LEN_FAIL = 300          # 荒谬超长判负

_TITLE_RE = re.compile(
    r"^(?P<type>[a-z][a-z0-9]*)"
    r"(?:\((?P<scope>[^()\s]+)\))?"
    r"(?P<breaking>!)?"
    r":[ ]+(?P<summary>\S(?:.*\S)?)$"
)

#: GitHub 界面「Revert」按钮生成的标题形如 `Revert "docs: …"`——放行并提示，
#: 避免把合法回滚 PR 卡红（若要严格禁止，删掉此特例即可）。
_REVERT_RE = re.compile(r'^Revert "[^"]+"$')


def _is_actions() -> bool:
    return os.environ.get("GITHUB_ACTIONS") == "true"


def _fail(msg: str, hints=None) -> int:
    print(("::error::" if _is_actions() else "FAIL: ") + msg)
    for h in hints or []:
        print("  · " + h)
    return 1


def check(title: str) -> int:
    if title is None:
        return _fail("未提供 PR 标题（缺 argv 参数且环境变量 PR_TITLE 为空）。",
                     ["用法：python -X utf8 scripts/check_pr_title.py \"<PR 标题>\""])
    if title != title.strip():
        return _fail("标题首尾含多余空白。", [f"实际标题：{title!r}"])
    if not title:
        return _fail("标题为空。")
    if _REVERT_RE.match(title):
        print(f"OK（Revert 特例放行）：{title}")
        return 0

    m = _TITLE_RE.match(title)
    if not m:
        return _fail(
            f"标题不符合约定 `type(scope): 摘要`：{title!r}",
            ["示例：`fix(world): BrainStore 读条数参数 limit→k`",
             "示例：`docs(deps): 声明已测版本区间与矩阵`",
             "注意：冒号后要有空格，且摘要不得为空；type 用英文小写。",
             "允许的 type：" + ", ".join(ALLOWED_TYPES)])

    t = m.group("type")
    if t not in ALLOWED_TYPES:
        return _fail(
            f"type {t!r} 不在白名单内。",
            ["允许的 type：" + ", ".join(ALLOWED_TYPES),
             "如确需新 type，请先与维护者确认后在 ALLOWED_TYPES 登记（勿临时放宽）。"])

    if len(title) > MAX_LEN_FAIL:
        return _fail(f"标题过长（{len(title)} > {MAX_LEN_FAIL} 字符）。",
                     ["请把细节放进 PR 正文，标题只留一句摘要。"])
    if len(title) > MAX_LEN_WARN:
        print(f"WARN: 标题偏长（{len(title)} > {MAX_LEN_WARN} 字符）——建议精简。")

    scope = m.group("scope")
    print(f"OK: type={t} scope={scope or '-'} breaking={bool(m.group('breaking'))} "
          f"summary={m.group('summary')}")
    return 0


def _selftest() -> int:
    """正负例自检；返回 0 = 全部符合预期。"""
    good = [
        "fix(world): BrainStore 读条数参数 limit→k——21 实体场景不再漏 1",
        "docs(deps): 声明已测版本区间与矩阵——numpy>=2.3.5,<2.6 / Pillow>=12.2,<13",
        "init: 灵枢聚合仓骨架",
        "feat(world)!: 破坏性改动示例",
        "chore(privacy): 公开面前进式脱敏",
        "ci(gate): 新增 CI 门禁",
        'Revert "docs: 某次改动"',
    ]
    bad = [
        "修复了一个 bug",              # 无 type
        "fix:nospace",                 # 冒号后无空格
        "unknown: 摘要",               # type 不在白名单
        "fix(): 摘要",                 # scope 为空
        "fix: ",                       # 摘要为空
        "Fix(world): 摘要",            # type 大写
        " fix: 摘要",                  # 首尾空白
        "fix(world):",                 # 无摘要
    ]
    fails = 0
    for s in good:
        if check(s) != 0:
            print(f"  SELFTEST-FAIL 应放行：{s!r}")
            fails += 1
    for s in bad:
        if check(s) != 1:
            print(f"  SELFTEST-FAIL 应判负：{s!r}")
            fails += 1
    # 用法错误
    if check(None) != 1:
        print("  SELFTEST-FAIL None 应判负")
        fails += 1
    print(f"selftest: {'OK' if fails == 0 else f'{fails} 例不符预期'}"
          f"（正例 {len(good)} · 负例 {len(bad)}）")
    return 0 if fails == 0 else 1


def main(argv) -> int:
    args = list(argv[1:])
    if not args:
        return check(os.environ.get("PR_TITLE"))
    if args[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if args[0] == "--list-types":
        print("\n".join(ALLOWED_TYPES))
        return 0
    if args[0] == "--selftest":
        return _selftest()
    if args[0] == "--title":
        if len(args) < 2:
            print("用法错误：--title 后缺少标题。", file=sys.stderr)
            return 2
        return check(args[1])
    if args[0].startswith("-"):
        print(f"用法错误：未知参数 {args[0]!r}（见 --help）。", file=sys.stderr)
        return 2
    return check(args[0])


if __name__ == "__main__":
    sys.exit(main(sys.argv))
