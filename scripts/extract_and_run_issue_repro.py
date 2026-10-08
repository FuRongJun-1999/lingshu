#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""extract_and_run_issue_repro — issue 复现脚本：抽取 → 落盘 → 限时执行 → 读数报告。

供 `.github/workflows/repro-bot.yml` 调用的**执行核**（Actions 只当外壳）；也可本机干跑
（`--body-file` / `--dry-run`）。

定位：外部报告人附了复现脚本的 issue，维护者打 `run-repro` 标签后自动跑出读数并回贴。
**只贴读数、不作结论**——结论仍走人工/模型侧的既有流程。

为什么它能跑不可信代码（安全前提，改动前先读）：
  · 本脚本对被测子进程的**环境做清洗**：剔除 `GITHUB_*` / `GH_*` / `ACTIONS_*` /
    `RUNNER_*` / `INPUT_*` 前缀的键，以及名字含 TOKEN|SECRET|PASSWORD|CREDENTIAL
    的键——被测脚本拿不到 job 令牌，也拿不到 `$GITHUB_ENV`/`$GITHUB_OUTPUT` 等
    CI 注入面的路径（workflow 侧另有配合：取正文与回评步骤才带 GH_TOKEN，
    执行步骤不带；checkout 用 persist-credentials: false 不落 git 凭据）。
  · 独立临时目录（默认 mkdtemp；不写仓内）、限时执行（默认 600 s，超时 kill）。
  · 报告中的路径一律中性化（`<临时目录>`），因为报告要原样贴到公开 issue。
  · 解释器用 `python -X utf8`；子进程 env 强制 PYTHONUTF8=1（对齐仓工作纪律第 15 条）。

抽取规则（与仓内 fenced 口径对齐：scripts/link_check.py::_FENCE_OPEN /
md_cg/docindex.py::_FENCE，即 CommonMark 围栏语义）：
  · 开启：行首（允许前导空白）`{3,}` 或 ~{3,}；info string = 同行其后剩余文本。
  · 闭合：同行定界符字符、长度不短于开启者、且该行除定界符外无其它字符
    （围栏内写的 ` ```js ` 只算代码内容，不算闭合）。
  · 未闭合（到文末）：按「到文末收尾」收编（CommonMark 语义）。
  · **只选第一个 info 首词为 python|py|py3（大小写不敏感）的围栏块**；
    正文含多个 python 块时也只用第一个——报告里如实给出全部块的语言分布。
  · 无 python 块 ⇒ 显式失败（退出码 2），**不执行任何东西**。

用法（argv）：
    python -X utf8 scripts/extract_and_run_issue_repro.py --repo OWNER/REPO --issue N [选项]
    python -X utf8 scripts/extract_and_run_issue_repro.py --body-file PATH [--source-desc "..."] [选项]
    python -X utf8 scripts/extract_and_run_issue_repro.py --selftest

选项：
    --repo OWNER/REPO        取正文来源仓（与 --issue 配套；内部走 `gh api`）
    --issue N                取该 issue 正文
    --body-file PATH         直接读本地文件当正文（离线/干跑；与 --repo/--issue 互斥）
    --source-desc TEXT       报告「来源」行用（缺省：body 文件名 / OWNER/REPO#N）
    --timeout SECONDS        执行上限，默认 600
    --max-output-bytes N     stdout/stderr 各自截断上限（字节），默认 20480（20 KiB）
    --out-dir PATH           脚本落盘目录（默认新建 mkdtemp 临时目录）
    --env KEY=VALUE          透传给被测脚本的环境变量（可重复；KEY 合法且 VALUE 非空才生效）
    --report-file PATH       报告落盘路径（同时总是打印到 stdout）
    --dry-run                只抽取+落盘，不执行
    --selftest               抽取器正负例自检（对齐 check_pr_title.py 的先例）
    -h / --help              显示本帮助

退出码：0 = 被测脚本退出码 0；
        1 = 被测脚本退出码非 0（含超时终止）；
        2 = 抽取失败 / 取正文失败 / 用法错误。
"""
from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
import tempfile

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

#: 围栏开启行（对齐 scripts/link_check.py 的 _FENCE_OPEN）。
_FENCE_OPEN = re.compile(r"^\s*(`{3,}|~{3,})")

#: info string 首词命中这些即视为「python 块」（大小写不敏感，取小写后比对）。
_PY_LANGS = {"python", "py", "py3"}

#: 被测子进程环境的剔除面（前缀 / 词）——令牌与 CI 注入面。
_ENV_DROP_PREFIXES = ("GITHUB_", "GH_", "ACTIONS_", "RUNNER_", "INPUT_")
_ENV_DROP_WORDS = ("TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")

DEFAULT_TIMEOUT = 600
DEFAULT_MAX_OUTPUT_BYTES = 20480

#: 报告末行固定声明（**不要**改动措辞——它是「只贴读数不作结论」的对外契约）。
REPORT_FOOTER = "本评论仅贴读数，不作结论。"


# ---------------------------------------------------------------------------
# 抽取
# ---------------------------------------------------------------------------

def _extract_fenced_blocks(text):
    """按 CommonMark 围栏口径抽取全部围栏块。

    返回 [(lang, code_first_line, code_last_line, code)]（行号为正文内 1-based，
    code_first/last 指块内**代码行**范围；空块记 first>last）。lang 为 info 首词
    小写；无 info 记 ""。
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks = []
    i, n = 0, len(lines)
    while i < n:
        m = _FENCE_OPEN.match(lines[i])
        if not m:
            i += 1
            continue
        delim = m.group(1)
        info = lines[i][m.end():].strip()
        lang = info.split()[0].lower() if info else ""
        j = i + 1
        closed = False
        while j < n:
            s = lines[j].strip()
            if (s and s[0] == delim[0] and len(s) >= len(delim)
                    and all(ch == delim[0] for ch in s)):
                closed = True
                break
            j += 1
        code_end = j if closed else n          # 0-based，代码行止于 code_end-1
        code = "\n".join(lines[i + 1:code_end])
        blocks.append((lang, i + 2, code_end, code))
        i = (j + 1) if closed else n
    return blocks


def _choose_python_block(blocks):
    """第一个 info 首词为 python|py|py3 的块；无则 None。"""
    for b in blocks:
        if b[0] in _PY_LANGS:
            return b
    return None


# ---------------------------------------------------------------------------
# 取正文
# ---------------------------------------------------------------------------

def _fetch_issue_body(repo, issue, env=None):
    """`gh api repos/OWNER/REPO/issues/N` 取正文（gh 输出 JSON；直接解析，不经 shell）。"""
    cmd = ["gh", "api", "repos/%s/issues/%s" % (repo, issue)]
    run_env = dict(os.environ)
    run_env["PYTHONUTF8"] = "1"
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env or run_env)
    if r.returncode != 0:
        raise RuntimeError("gh api 失败（退出码 %d）：%s"
                           % (r.returncode, (r.stderr or "").strip()[:800]))
    try:
        obj = json.loads(r.stdout)
    except ValueError as e:
        raise RuntimeError("gh api 返回非 JSON：%s" % e) from e
    return obj.get("body") or ""


# ---------------------------------------------------------------------------
# 执行
# ---------------------------------------------------------------------------

def _child_env(extra):
    """构造被测子进程 env：清掉令牌/CI 注入面，再注入显式覆盖。"""
    env = {}
    for k, v in os.environ.items():
        if k.startswith(_ENV_DROP_PREFIXES):
            continue
        if any(w in k.upper() for w in _ENV_DROP_WORDS):
            continue
        env[k] = v
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env.update(extra)
    return env


def _run_script(script_path, cwd, timeout, extra_env):
    """限时执行；返回 (status, exit_line, stdout, stderr)。

    status：0 通过 / 1 非 0 或超时。超时经 subprocess.run 的 kill 语义终止
    （直接子进程；其孙进程可能短暂残留——一次性 runner，已在文档接受）。
    """
    cmd = [sys.executable, "-X", "utf8", script_path]
    try:
        r = subprocess.run(cmd, cwd=cwd, env=_child_env(extra_env),
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired as e:
        out = e.stdout if isinstance(e.stdout, str) else (e.stdout or b"").decode("utf-8", "replace")
        err = e.stderr if isinstance(e.stderr, str) else (e.stderr or b"").decode("utf-8", "replace")
        return (1, "超时终止（%d s）" % timeout, out or "", err or "")
    except OSError as e:
        return (1, "启动失败（%s）" % e, "", "")
    rc = r.returncode
    return (0 if rc == 0 else 1, str(rc), r.stdout or "", r.stderr or "")


# ---------------------------------------------------------------------------
# 报告渲染（固定模板）
# ---------------------------------------------------------------------------

def _truncate(text, limit):
    """按 UTF-8 字节截断（不切半个字符）；返回 (text, 原始字节数, 是否截断)。"""
    data = (text or "").encode("utf-8")
    if len(data) <= limit:
        return text or "", len(data), False
    return data[:limit].decode("utf-8", "ignore"), len(data), True


def _md_fence(text):
    """围栏长度 = 内容里最长反引号串 + 1（至少 3）——防输出内容撑破 markdown。"""
    longest = 0
    for m in re.finditer(r"`+", text or ""):
        longest = max(longest, len(m.group(0)))
    return "`" * max(3, longest + 1)


def _env_line():
    py = "Python %s（%s）" % (platform.python_version(), platform.python_implementation())
    osdesc = "%s %s" % (platform.system(), platform.release())
    if os.environ.get("GITHUB_ACTIONS") == "true":
        where = "GitHub Actions 托管 runner"
    else:
        where = "本机（非 Actions）"
    return "%s · %s · %s" % (py, osdesc, where)


def _render_report(source, extract_info, cmd_line, exit_line, stdout_txt, stderr_txt,
                   max_bytes, note=None):
    """固定模板：来源/抽取/环境/执行/退出码 + 原始输出 + 末行固定声明。"""
    parts = []
    parts.append("### 自动复现读数（repro-bot）")
    parts.append("")
    parts.append("> 由 `repro-bot` 在 `run-repro` 标签触发的自动执行——**仅**呈现原始读数。")
    parts.append("")
    parts.append("- **来源**：%s" % source)
    parts.append("- **抽取**：%s" % extract_info)
    parts.append("- **环境**：%s" % _env_line())
    parts.append("- **执行**：%s" % cmd_line)
    parts.append("- **退出码**：%s" % exit_line)
    if note:
        parts.append("")
        parts.append("> %s" % note)
    parts.append("")
    for title, body in (("原始输出（stdout）", stdout_txt), ("原始错误（stderr）", stderr_txt)):
        shown, orig_n, cut = _truncate(body, max_bytes)
        fence = _md_fence(shown)
        parts.append("#### %s —— 上限 %d B" % (title, max_bytes))
        parts.append("")
        parts.append(fence + "text")
        if shown.strip():
            parts.append(shown.rstrip("\n"))
            if cut:
                parts.append("…（已截断：原始 %d B，仅保留前 %d B）" % (orig_n, max_bytes))
        else:
            parts.append("(空)")
        parts.append(fence)
        parts.append("")
    parts.append("---")
    parts.append(REPORT_FOOTER)
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------

def _selftest():
    """抽取器正负例自检；返回 0 = 全部符合预期。"""
    good = [
        ("前文\n```python\nprint(1)\n```\n后文", "print(1)"),
        ("```py\nx = 1\n```", "x = 1"),
        ("```py3\nx = 2\n```", "x = 2"),
        ("~~~python\nx = 3\n~~~", "x = 3"),
        ("```text\na\n```\n```python\nx = 4\n```", "x = 4"),        # 跳过非 python 块
        ("```python\nx = 5\n```\n```python\nx = 6\n```", "x = 5"),  # 取第一个 python 块
        ("````text\n```\n````\n```Python\nx = 7\n```", "x = 7"),    # 4 反引号包裹 + 大写 info
        ("  ```python\nx = 8\n  ```", "x = 8"),                     # 缩进围栏
        ("行内 `code` 不成块\n```python\nx = 9\n```", "x = 9"),      # 行内代码不干扰
        ("```python\na = 1\n```js\nb = 2\n```", "a = 1\n```js\nb = 2"),  # 围栏内 ```js 不闭合
        ("```python\nx = 10\n（无闭合）", "x = 10\n（无闭合）"),       # 未闭合收编到文末
    ]
    fails = 0
    for text, want in good:
        blocks = _extract_fenced_blocks(text)
        chosen = _choose_python_block(blocks)
        got = chosen[3].strip("\n") if chosen else None
        if got != want:
            print("  SELFTEST-FAIL 应抽到 %r：实得 %r（text=%r）" % (want, got, text))
            fails += 1
    bad = [
        "",                                    # 空正文
        "只有文字，没有围栏",                   # 无围栏
        "```text\n只有 text 块\n```",           # 无 python 块
        "```\n无 info 的块\n```",               # info 为空
        "```diff\n-a\n+b\n```",                # 只有 diff 块
    ]
    for text in bad:
        chosen = _choose_python_block(_extract_fenced_blocks(text))
        if chosen is not None:
            print("  SELFTEST-FAIL 应抽不到（None）：实得 %r（text=%r）" % (chosen[3], text))
            fails += 1
    # 行号口径：代码行范围 1-based
    b = _choose_python_block(_extract_fenced_blocks("a\n```python\nx\n```\n"))
    if not b or (b[1], b[2]) != (3, 3):
        print("  SELFTEST-FAIL 行号口径应为 (3,3)：实得 %r" % (b,))
        fails += 1
    # 截断口径：不切半个多字节字符
    cut, orig, flag = _truncate("中" * 100, 10)
    if not flag or orig != 300 or len(cut.encode("utf-8")) > 10:
        print("  SELFTEST-FAIL 截断口径：%r %d %s" % (cut, orig, flag))
        fails += 1
    print("selftest: %s（正例 %d · 负例 %d · 附加 2）"
          % ("OK" if fails == 0 else "%d 例不符预期" % fails, len(good), len(bad)))
    return 0 if fails == 0 else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _usage(msg):
    print("用法错误：%s（见 --help）" % msg, file=sys.stderr)
    return 2


def main(argv):
    args = list(argv[1:])
    if any(a in ("-h", "--help") for a in args):
        print(__doc__)
        return 0
    if "--selftest" in args:
        return _selftest()

    opt = {
        "repo": None, "issue": None, "body_file": None, "source_desc": None,
        "timeout": DEFAULT_TIMEOUT, "max_bytes": DEFAULT_MAX_OUTPUT_BYTES,
        "out_dir": None, "report_file": None, "dry_run": False, "env": [],
    }
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--dry-run":
            opt["dry_run"] = True
            i += 1
            continue
        if a in ("--repo", "--issue", "--body-file", "--source-desc", "--timeout",
                 "--max-output-bytes", "--out-dir", "--env", "--report-file"):
            if i + 1 >= len(args):
                return _usage("%s 后缺参数" % a)
            val = args[i + 1]
            i += 2
            if a == "--repo":
                opt["repo"] = val
            elif a == "--issue":
                if not val.isdigit():
                    return _usage("--issue 应为数字：%r" % val)
                opt["issue"] = val
            elif a == "--body-file":
                opt["body_file"] = val
            elif a == "--source-desc":
                opt["source_desc"] = val
            elif a == "--timeout":
                if not val.isdigit() or int(val) <= 0:
                    return _usage("--timeout 应为正整数秒：%r" % val)
                opt["timeout"] = int(val)
            elif a == "--max-output-bytes":
                if not val.isdigit() or int(val) <= 0:
                    return _usage("--max-output-bytes 应为正整数：%r" % val)
                opt["max_bytes"] = int(val)
            elif a == "--out-dir":
                opt["out_dir"] = val
            elif a == "--env":
                if val:
                    opt["env"].append(val)
            elif a == "--report-file":
                opt["report_file"] = val
            continue
        return _usage("未知参数 %r" % a)

    # ── 输入互斥校验 ──
    if opt["body_file"]:
        if opt["repo"] or opt["issue"]:
            return _usage("--body-file 与 --repo/--issue 互斥")
    elif not (opt["repo"] and opt["issue"]):
        return _usage("需给 --body-file，或 --repo 与 --issue 配套")

    # ── 取正文 ──
    if opt["body_file"]:
        try:
            with open(opt["body_file"], encoding="utf-8", errors="replace") as f:
                body = f.read()
        except OSError as e:
            print("FAIL：读正文文件失败：%s" % e, file=sys.stderr)
            return 2
        source = opt["source_desc"] or ("本地文件 `%s`" % os.path.basename(opt["body_file"]))
    else:
        try:
            body = _fetch_issue_body(opt["repo"], opt["issue"])
        except RuntimeError as e:
            report = _render_report(
                source="`%s#%s`（gh api）" % (opt["repo"], opt["issue"]),
                extract_info="未抽取（取正文失败）",
                cmd_line="未执行（取正文失败）",
                exit_line="—",
                stdout_txt="", stderr_txt=str(e),
                max_bytes=opt["max_bytes"],
                note="repro-bot 未能取到 issue 正文，未执行任何代码。")
            if opt["report_file"]:
                with open(opt["report_file"], "w", encoding="utf-8", newline="\n") as f:
                    f.write(report)
            print(report)
            return 2
        source = opt["source_desc"] or ("`%s#%s`" % (opt["repo"], opt["issue"]))

    # ── 抽取 ──
    blocks = _extract_fenced_blocks(body)
    tally = {}
    for b in blocks:
        tally[b[0] or "(无标注)"] = tally.get(b[0] or "(无标注)", 0) + 1
    tally_line = " / ".join("%s×%d" % (k, v) for k, v in tally.items()) or "无"

    def _fail_report(extract_info, note, stderr_txt=""):
        report = _render_report(source=source, extract_info=extract_info,
                                cmd_line="未执行（抽取失败）", exit_line="—",
                                stdout_txt="", stderr_txt=stderr_txt,
                                max_bytes=opt["max_bytes"], note=note)
        if opt["report_file"]:
            with open(opt["report_file"], "w", encoding="utf-8", newline="\n") as f:
                f.write(report)
        print(report)
        return 2

    if not blocks:
        return _fail_report("未抽取（正文没有任何围栏代码块）",
                            "issue 正文里没有围栏代码块，无可执行内容；未执行任何代码。")
    chosen = _choose_python_block(blocks)
    if chosen is None:
        return _fail_report("未抽取（正文无 python 围栏块；块语言：%s）" % tally_line,
                            "issue 正文里没有 info 标注为 python/py/py3 的围栏代码块；"
                            "未执行任何代码。请报告人补充 python 复现脚本。")
    lang, first, last, code = chosen
    nlines = code.count("\n") + 1 if code else 0
    extract_info = ("第 %d 个围栏块（正文第 %d–%d 行的 python 块，共 %d 行；"
                    "正文围栏块 %d 个：%s）"
                    % (blocks.index(chosen) + 1, first, last, nlines, len(blocks), tally_line))

    # ── 落盘（独立临时目录） ──
    out_dir = opt["out_dir"] or tempfile.mkdtemp(prefix="repro_issue_")
    os.makedirs(out_dir, exist_ok=True)
    script_name = "repro_issue_%s.py" % opt["issue"] if opt["issue"] else "repro_issue_local.py"
    script_path = os.path.join(out_dir, script_name)
    with open(script_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(code + "\n")

    cmd_line = ("`python -X utf8 %s`（cwd=`<临时目录>`；独立临时目录，不写仓内；"
                "超时 %d s；脚本内容=抽出的代码块原文，未做任何改动）"
                % (script_name, opt["timeout"]))

    if opt["dry_run"]:
        print("DRY-RUN：已抽取并落盘")
        print("  " + extract_info)
        print("  " + cmd_line.replace("`", ""))
        print("  note：脚本只写进临时目录：%s" % out_dir)
        return 0

    # ── 限时执行 ──
    extra_env = {}
    for kv in opt["env"]:
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        if k and v:
            extra_env[k] = v
    status, exit_line, out, err = _run_script(script_path, out_dir, opt["timeout"], extra_env)

    note = None
    if status != 0:
        note = ("被测脚本未正常结束（%s）。读数照贴、不作结论——"
                "脚本可能依赖运行方预置（如变量/环境），或与当前 HEAD 行为不符。" % exit_line)
    report = _render_report(source=source, extract_info=extract_info, cmd_line=cmd_line,
                            exit_line=exit_line, stdout_txt=out, stderr_txt=err,
                            max_bytes=opt["max_bytes"], note=note)
    if opt["report_file"]:
        with open(opt["report_file"], "w", encoding="utf-8", newline="\n") as f:
            f.write(report)
    print(report)
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv))
