# -*- coding: utf-8 -*-
"""test_issue120_239_362_repro_exec · repro-bot 执行核三件 P2 缺陷守卫
============================================================================
被测件：`scripts/extract_and_run_issue_repro.py`（执行核）＋
        `.github/workflows/repro-bot.yml`（外壳）。
与既有的 `tests/test_issue363_235_241_repro_bot.py` / `test_issue144_repro_bot_toctou.py`
**不共用机制**：那两件管的是注入面（#363 info string 行内代码化）、HTML 注释内围栏
（#235）、job 隔离（#241/#144）；本件管的是**入口编码**、**读数可复现性**与
**抽取/回收口径**——任一件抽掉修复时，只本件的那条断言变红，不靠对方兜底。

#120（成立）——入口在 Windows cp936 下 `--help` 直接崩（UnicodeEncodeError）。
    复现（修复前实跑）：`python scripts/extract_and_run_issue_repro.py --help`，env 去掉
    `PYTHONUTF8`/`PYTHONIOENCODING`、stdout 为管道 ⇒ stdout 按 cp936 编码，而模块文档含
    `⇒`(U+21D2) / `⚠`(U+26A0) 等 GBK **编不出**的字符 ⇒
    `UnicodeEncodeError: 'gbk' codec can't encode character '\u26a0' …` 退出码 1、stdout 0 字节。
    判据：仓工作纪律「全仓文本与路径一律 UTF-8…控制台乱码属显示层（取用走 python）」
    ——显示层乱码可接受，**崩溃不可**；故入口把自身 stdio 显式切 UTF-8（`_force_utf8_stdio`）。
    断言：G120 子进程按 cp936 管道跑 `--help` ⇒ 退出码 0、stdout 可按 UTF-8 解码且含 `⇒`/`⚠`。

#239（成立）——读数报告不记被测提交与依赖版本；路径中性化承诺未兑现；`--selftest` 未接 CI。
    复现（修复前实跑）：报告只有「环境：Python/OS/宿主」，无 commit、无 numpy/Pillow；
    docstring 写「报告中的路径一律中性化（`<临时目录>`）」而实测 stdout/stderr 里
    `C:\\Users\\…\\repro_issue_xxx\\repro_issue_local.py` 原样透出；
    `grep -rn selftest .github/workflows/*.yml` 仅命中 `pr-lint.yml`（check_pr_title）。
    判据：① 读数要可复现须钉住被测代码与依赖版本（经验标定＝本件 #239）；
          ② 路径口径＝仓工作纪律「公开面路径写相对路径，本机绝对路径以中性占位代」；
          ③ 自检入口须真被 CI 执行——先例＝`pr-lint.yml` 跑 `check_pr_title.py --selftest`
             （同族盲区教训：不被执行的判据等于不存在）。
    断言：G239a 端到端报告含「被测提交」行（40 位十六进制 sha）与「依赖」行（numpy 版本）；
          G239b 被测输出里的落盘目录/仓根**不再**以原文出现，而写作 `<临时目录>`/`<仓根>`；
          G239c 至少一个 workflow 的 `run:` 真跑 `extract_and_run_issue_repro.py --selftest`。

#362（成立）——把合法复现判成失败（缩进围栏 / 引用块围栏 / 后台进程占管道超时）。
    复现（修复前实跑）：
      · `  ```python` + `  print(1)` ⇒ 抽出带 2 空格缩进的代码，落盘即
        `IndentationError: unexpected indent`，退出码 1（合法复现被判失败）；
      · `> ```python` + `> print(...)` ⇒ `_FENCE_OPEN` 不认 `>`，整块抽不到 ⇒ rc 2「未抽取」；
      · 脚本 `subprocess.Popen([...sleep 20...])` 后 `print('DONE')` 立即退出 ⇒
        `capture_output=True` 等的是**管道 EOF**（孙进程占着写端）⇒ 8 s 后
        `TimeoutExpired`，报告写「超时终止（8 s）」，退出码 1——**父进程其实已 exit 0**。
    判据：CommonMark 围栏/块引用语义（围栏缩进量从块内代码行剥除；块引用标记在判块前
    剥除）＋ 超时口径应看**进程退出**而非管道关闭；三处均属经验标定（本件 #362），
    仓内无对应理论章节。
    断言：G362a 缩进围栏端到端 rc 0 且 stdout 见标记；G362b 引用块围栏（含嵌套）端到端
          rc 0，且**非**引用块里代码行首的 `>` 是字面量不被误剥；G362c 派生后台进程的
          脚本在 `--timeout 3` 下仍报退出码 0（不得出现「超时终止」）；G362d `--selftest` 绿。

运行（仓根）：python -X utf8 -m pytest tests/test_issue120_239_362_repro_exec.py -q --no-header
退出码：0 = 全过；1 = 有断言失败。
"""
from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT = os.path.join(REPO, "scripts", "extract_and_run_issue_repro.py")
_WORKFLOWS = os.path.join(REPO, ".github", "workflows")
_SHA40 = re.compile(r"\b[0-9a-f]{40}\b")


def _load_module():
    spec = importlib.util.spec_from_file_location("_repro_exec_core_p2", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_MOD = _load_module()


def _run_main(body, argv_extra=()):
    """把 body 写临时文件、跑执行核 main()（in-process）；返回 (rc, stdout)。"""
    import contextlib
    import io
    d = tempfile.mkdtemp(prefix="repro_p2_")
    p = os.path.join(d, "body.md")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(body)
    argv = ["extract_and_run_issue_repro.py", "--body-file", p,
            "--source-desc", "guard", *argv_extra]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = _MOD.main(argv)
    return rc, buf.getvalue()


def _child_env_cp936():
    """剥掉 UTF-8 强制项、把子进程 stdio 编码钉到 cp936（复现 #120 的宿主条件）。"""
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONUTF8", "PYTHONIOENCODING")}
    env["PYTHONIOENCODING"] = "cp936"
    return env


# ── 量具自检 ────────────────────────────────────────────────────────────────
def test_gauge_is_alive():
    """量具自检：三件各自依赖的函数/文件真在场，且 cp936 编不出 `⇒`/`⚠`（#120 前提）。"""
    assert callable(getattr(_MOD, "_force_utf8_stdio", None)), \
        "执行核缺 `_force_utf8_stdio` ⇒ #120 的入口兜底不存在"
    assert callable(getattr(_MOD, "_neutralize_paths", None)), \
        "执行核缺 `_neutralize_paths` ⇒ #239② 的路径中性化不存在"
    assert callable(getattr(_MOD, "_git_head_line", None)), \
        "执行核缺 `_git_head_line` ⇒ #239① 的被测提交行不存在"
    assert callable(getattr(_MOD, "_bq_strip", None)), \
        "执行核缺 `_bq_strip` ⇒ #362② 的引用块支持不存在"
    assert os.path.isfile(_SCRIPT) and os.path.isdir(_WORKFLOWS)
    # 缺陷前提：这两个字符在 cp936 下确实编不出（否则 #120 无从复现，本判据无判别力）
    for ch in ("\u21d2", "\u26a0"):
        try:
            ch.encode("cp936")
        except UnicodeEncodeError:
            pass
        else:
            raise AssertionError("cp936 竟能编码 %r ⇒ #120 的前提不成立，本件判据需重审" % ch)
    # 模块文档确实含这两个字符（缺陷的触发面）
    doc = _MOD.__doc__ or ""
    assert "\u21d2" in doc and "\u26a0" in doc, \
        "模块文档不再含 ⇒/⚠ ⇒ `print(__doc__)` 不再触发 #120，本件判据需重审"


# ── #120 ────────────────────────────────────────────────────────────────────
def test_120_help_survives_cp936_pipe():
    """#120：cp936 管道下 `--help` 必须退出码 0 且 stdout 是完整 UTF-8。

    变异点：删掉 `main()` 里的 `_force_utf8_stdio()` 调用（或删该函数）⇒ 本断言红：
    子进程 stdout 按 cp936 编码、`⚠`/`⇒` 编不出 ⇒ `UnicodeEncodeError`、退出码 1。
    """
    r = subprocess.run([sys.executable, _SCRIPT, "--help"],
                       capture_output=True, env=_child_env_cp936(), timeout=60)
    err = r.stderr.decode("utf-8", "replace")
    assert r.returncode == 0, (
        "#120：cp936 下 `--help` 非 0 退出（%d）⇒ 入口仍会崩\nstderr 尾：%s"
        % (r.returncode, err[-600:]))
    assert b"UnicodeEncodeError" not in r.stderr, (
        "#120：stderr 出现 UnicodeEncodeError：%s" % err[-600:])
    text = r.stdout.decode("utf-8")            # 解不开即红（说明不是 UTF-8 输出）
    assert "⇒" in text and "⚠" in text, (
        "#120：`--help` 输出里 `⇒`/`⚠` 缺失 ⇒ 输出被截断或替换，读数不可信")
    assert "用法（argv）" in text, "#120：`--help` 输出不完整（缺用法段）"


def test_120_stdio_helper_is_idempotent_and_safe():
    """#120 辅助口径：`_force_utf8_stdio` 对不支持 reconfigure 的流静默跳过（不抛）。"""
    import io
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout = io.StringIO()             # 无 reconfigure ⇒ 必须静默跳过
        sys.stderr = io.StringIO()
        _MOD._force_utf8_stdio()               # 抛异常即红
    finally:
        sys.stdout, sys.stderr = old_out, old_err


# ── #239 ────────────────────────────────────────────────────────────────────
def test_239_report_records_commit_and_deps():
    """#239①：报告须含「被测提交」（40 位 sha）与「依赖」（numpy 版本）两行。

    变异点：把 `- **被测提交**：` / `- **依赖**：` 两行从 `_render_report` 删掉
    ⇒ 本断言红。
    """
    rc, report = _run_main("```python\nprint('OK')\n```\n")
    assert rc == 0, "探针脚本应 rc 0，实得 %r\n%s" % (rc, report)
    head_lines = [ln for ln in report.splitlines() if ln.startswith("- **被测提交**：")]
    dep_lines = [ln for ln in report.splitlines() if ln.startswith("- **依赖**：")]
    assert head_lines, "#239①：报告无「被测提交」行\n%s" % report
    assert dep_lines, "#239①：报告无「依赖」行\n%s" % report
    assert _SHA40.search(head_lines[0]) or "未知" in head_lines[0], (
        "#239①：「被测提交」行既无 40 位 sha 也无「未知」标注：%r" % head_lines[0])
    assert "numpy" in dep_lines[0], (
        "#239①：「依赖」行未记 numpy 版本：%r" % dep_lines[0])


def test_239_paths_are_neutralized_in_report():
    """#239②：被测输出里的落盘目录/仓根必须中性化（不得原样进公开 issue）。

    变异点：`_render_report` 里把 `_neutralize_paths(body, neutral_dirs)` 改回 `body`
    ⇒ 本断言红（报告出现本机绝对路径原文）。
    """
    body = ("```python\n"
            "import os\n"
            "print('CWD=' + os.getcwd())\n"
            "print('FILE=' + __file__)\n"
            "print('REPO=' + r'%s')\n"
            "```\n") % REPO
    rc, report = _run_main(body)
    assert rc == 0, "探针脚本应 rc 0，实得 %r\n%s" % (rc, report)
    assert "CWD=<临时目录>" in report, (
        "#239②：落盘目录未中性化为 `<临时目录>`\n%s" % report)
    assert "<仓根>" in report, "#239②：仓根未中性化为 `<仓根>`\n%s" % report
    # 反向：报告里不得再出现本机仓根原文
    assert REPO not in report, (
        "#239②：报告里出现仓根绝对路径原文（承诺未兑现）\n%s" % report)
    assert "repro_issue_" + os.sep not in report and "repro_p2_" not in report, (
        "#239②：报告里出现临时目录路径片段原文\n%s" % report)


def test_239_selftest_is_wired_into_ci():
    """#239③：至少一个 workflow 的 `run:` 真跑执行核 `--selftest`（先例：pr-lint.yml）。

    变异点：删掉 repro-bot.yml 里那一步 ⇒ 本断言红。
    """
    hits = []
    for name in sorted(os.listdir(_WORKFLOWS)):
        if not name.endswith((".yml", ".yaml")):
            continue
        with open(os.path.join(_WORKFLOWS, name), encoding="utf-8") as f:
            text = f.read()
        for ln in text.splitlines():
            if "extract_and_run_issue_repro.py" in ln and "--selftest" in ln:
                hits.append("%s: %s" % (name, ln.strip()))
    assert hits, (
        "#239③：没有任何 workflow 跑 `extract_and_run_issue_repro.py --selftest` ⇒ "
        "抽取器自检不在门禁面（同族盲区：不被执行的判据等于不存在）")


def test_239_selftest_exit_code_is_zero():
    """#239③ 前置：`--selftest` 自身必须绿（红的话接进 CI 也只会全仓红）。"""
    r = subprocess.run([sys.executable, "-X", "utf8", _SCRIPT, "--selftest"],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=120)
    assert r.returncode == 0, (
        "`--selftest` 非 0（%d）\nstdout：%s\nstderr：%s"
        % (r.returncode, r.stdout, r.stderr))
    assert "selftest: OK" in r.stdout, "自检输出未见 OK 结论：%r" % r.stdout


# ── #362 ────────────────────────────────────────────────────────────────────
def test_362a_indented_fence_is_dedented():
    """#362①：缩进围栏的块内代码须按开启围栏的缩进量剥除，不得落盘成 IndentationError。

    变异点：`_extract_fenced_blocks` 里把 `_dedent(raw, indent)` 改回 `raw`
    ⇒ 抽出带 2 空格缩进的代码 ⇒ 端到端 rc 1 + IndentationError，本断言红。
    """
    blocks = _MOD._extract_fenced_blocks("  ```python\n  print(1)\n  ```\n")
    assert len(blocks) == 1 and blocks[0][3] == "print(1)", (
        "#362①：缩进围栏的代码未去缩进，抽出 %r" % (blocks,))
    # 只剥开启行的缩进量，更深的缩进（块内结构）保持
    b2 = _MOD._extract_fenced_blocks("   ```python\n   if True:\n       y = 1\n   ```\n")
    assert b2[0][3] == "if True:\n    y = 1", (
        "#362①：去缩进口径错（应只剥 3 格），抽出 %r" % (b2[0][3],))
    rc, report = _run_main("  ```python\n  print('INDENT-OK')\n  ```\n")
    assert rc == 0, "#362①：缩进围栏的合法复现被判失败（rc=%r）\n%s" % (rc, report)
    assert "INDENT-OK" in report, "#362①：stdout 未见标记\n%s" % report
    assert "IndentationError" not in report, "#362①：仍报 IndentationError\n%s" % report


def test_362b_blockquote_fence_is_extracted():
    """#362②：引用块（`>`，可嵌套）内的围栏须被抽取；非引用块里的 `>` 保持字面量。

    变异点：`_extract_fenced_blocks` 里把 `_bq_strip(...)` 改回取原文行
    ⇒ 引用块围栏整块抽不到 ⇒ rc 2「未抽取」，本断言红。
    """
    bq = _MOD._extract_fenced_blocks("> ```python\n> print(1)\n> ```\n")
    assert len(bq) == 1 and bq[0][3] == "print(1)", \
        "#362②：引用块围栏未被抽取：%r" % (bq,)
    bq2 = _MOD._extract_fenced_blocks("> > ```python\n> > x = 1\n> > ```\n")
    assert len(bq2) == 1 and bq2[0][3] == "x = 1", \
        "#362②：嵌套引用块围栏未被抽取：%r" % (bq2,)
    # 反向：非引用块的代码里行首 `>` 是字面量，不得被误剥
    fb = _MOD._extract_fenced_blocks("```python\n> literal\n```")
    assert fb[0][3] == "> literal", "#362②：非引用块里代码行的 `>` 被误剥：%r" % (fb,)
    rc, report = _run_main("> ```python\n> print('QUOTE-OK')\n> ```\n")
    assert rc == 0, "#362②：引用块里的合法复现被判失败（rc=%r）\n%s" % (rc, report)
    assert "QUOTE-OK" in report, "#362②：stdout 未见标记\n%s" % report


def test_362c_background_child_does_not_fake_a_timeout():
    """#362③：脚本派生占着 stdout 的后台孙进程、自己 exit 0 ⇒ 必须报退出码 0。

    变异点：`_run_script` 改回 `subprocess.run(capture_output=True, timeout=…)`
    ⇒ 等管道 EOF（孙进程占 8 s > timeout 3）⇒ 报告「超时终止（3 s）」、rc 1，本断言红。
    """
    body = ("```python\n"
            "import subprocess, sys\n"
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(8)'])\n"
            "print('DONE-BG')\n"
            "```\n")
    rc, report = _run_main(body, ("--timeout", "3"))
    exit_lines = [ln for ln in report.splitlines() if ln.startswith("- **退出码**：")]
    assert exit_lines, "报告无退出码行\n%s" % report
    assert "超时终止" not in exit_lines[0], (
        "#362③：脚本已 exit 0 仍被判超时（后台孙进程占着管道）⇒ %r\n%s"
        % (exit_lines[0], report))
    assert rc == 0 and "DONE-BG" in report, (
        "#362③：rc=%r，stdout 未见标记\n%s" % (rc, report))
    assert exit_lines[0].strip().endswith("：0"), (
        "#362③：退出码行非 0：%r" % exit_lines[0])


def test_362d_real_timeout_still_reported():
    """#362③ 反向（防「把超时判据一并删掉」）：真不结束的脚本仍须判超时终止。"""
    body = "```python\nimport time\ntime.sleep(30)\n```\n"
    rc, report = _run_main(body, ("--timeout", "2"))
    exit_lines = [ln for ln in report.splitlines() if ln.startswith("- **退出码**：")]
    assert exit_lines and "超时终止" in exit_lines[0], (
        "#362③：真超时未被识别（判据被放宽？）%r\n%s" % (exit_lines, report))
    assert rc == 1, "#362③：超时的退出码应为 1，实得 %r" % rc
