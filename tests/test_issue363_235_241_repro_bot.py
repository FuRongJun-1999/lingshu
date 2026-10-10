# -*- coding: utf-8 -*-
"""test_issue363_235_241_repro_bot · issue 复现机器人的三件 P1 缺陷守卫
============================================================================
被测件：`scripts/extract_and_run_issue_repro.py`（执行核）＋
        `.github/workflows/repro-bot.yml`（外壳）。报告要**原样贴到公开 issue**，
        正文（含围栏 info string）完全由外部报告人控制 ⇒ 注入面必须封死。

三件缺陷与判据（各给一条断言组，各自可单独变异变红）：

#363 —— 正文围栏的 **info string** 原样写进围栏**之外**的报告行。
    取证（缺陷态）：`tally_line = " / ".join("%s×%d" % (k, v) …)`，k 即 info 首词
    原文；拼进 `- **抽取**：…`（固定模板的**行外**，不在任何围栏里）。于是
    ` ```[x](https://evil.example) ` 会让报告渲染出活外链、`@everyone` 会成真提及、
    `<img …>` 会成 HTML。
    判据：一切来自正文的 info string 入报告前必须包成**行内代码**（`_inline_code`）
    ⇒ 其中的 markdown 只作字面量。
    断言：G2 端到端跑一次，`- **抽取**：` 行里注入串必须**带反引号**出现。

#235 —— 抽取器不识别 HTML 注释：`<!-- … -->` 里的 python 块网页不可见却被优先执行。
    取证（缺陷态）：`_extract_fenced_blocks` 全程按原文行匹配 `_FENCE_OPEN`，注释
    里的围栏照抽；实测注释里的 python 块被抽出并执行（stdout 打印 HIDDEN-EXECUTED）。
    判据：按 CommonMark 块级语义单趟扫描——**行首在注释内**的行不参与围栏判定；
    反之围栏内（含代码里的 `<!--`）一律按原文、不认注释。
    断言：G3 注释内 python 块 ⇒ 无块可抽（等价「未抽取」）；可见块与注释块混排时
    只抽可见块，且行号仍指正文真实行；围栏代码里的 `<!--` 不被误当注释。

#241 —— 环境清洗挡不住**同 job 后续步骤**：可经父进程 environ 写回
    `$GITHUB_ENV`/`$GITHUB_PATH`；残留孙进程可在回贴前改写报告。
    取证（缺陷态）：模块 docstring 断言「被测脚本拿不到 job 令牌，也拿不到
    `$GITHUB_ENV`/`$GITHUB_OUTPUT` 等」，但 `_child_env` 只清**子进程自身** env
    （同 uid 可读 Linux `/proc/<ppid>/environ` 取回），`_run_script` 只用
    `subprocess.run(timeout=…)` 回收直接子进程；而 workflow 里「回贴读数」步骤
    以 `GH_TOKEN` 在**同一个 job/同一台 runner** 上运行 `gh issue comment`。
    判据：真正的边界是**执行与回评分属不同 job/runner**——执行核的 env 清洗只是
    「封子进程自己的 env」，不得被当成 job 隔离。
    断言：G4 结构断言 workflow 分三 job（prep/exec/report），跑不可信代码的
    `exec` job **无** `issues: write` 且其任一步骤**不出现** `GH_TOKEN`；回评
    只出现在 `report` job。G5 断言执行核 docstring 已如实写明清洗的作用域局限
    （不再声称 job 级隔离）。

运行（lingshu 仓根）：python -X utf8 -m pytest tests/test_issue363_235_241_repro_bot.py -q
退出码：0 = 全过；1 = 有断言失败。
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import sys
import tempfile

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT = os.path.join(REPO, "scripts", "extract_and_run_issue_repro.py")
_WORKFLOW = os.path.join(REPO, ".github", "workflows", "repro-bot.yml")


def _load_module():
    spec = importlib.util.spec_from_file_location("_repro_exec_core", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_MOD = _load_module()


def _run_main(body, argv_extra=()):
    """把 body 写临时文件，跑执行核 main()，返回 (rc, stdout)。"""
    d = tempfile.mkdtemp(prefix="repro_guard_")
    p = os.path.join(d, "body.md")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(body)
    argv = ["extract_and_run_issue_repro.py", "--body-file", p,
            "--source-desc", "guard", *argv_extra]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = _MOD.main(argv)
    return rc, buf.getvalue()


def _extract_line(report):
    for ln in report.splitlines():
        if ln.startswith("- **抽取**："):
            return ln
    raise AssertionError("报告里没有 `- **抽取**：` 行：\n" + report)


# ── 量具自检 ────────────────────────────────────────────────────────────────
def test_gauge_is_alive():
    """量具自检：模块真加载到了、三组断言各自依赖的函数/文件真在场。"""
    assert hasattr(_MOD, "_inline_code"), "执行核缺 `_inline_code` ⇒ #363 的封口不存在"
    assert hasattr(_MOD, "_advance_comment_state"), "执行核缺 `_advance_comment_state` ⇒ #235 的封口不存在"
    assert os.path.isfile(_WORKFLOW), "workflow 文件不在：%s" % _WORKFLOW
    assert callable(_MOD.main) and callable(_MOD._extract_fenced_blocks)


# ── #363 ────────────────────────────────────────────────────────────────────
def test_inline_code_wraps_and_escapes_backticks():
    """`_inline_code` 的基本口径：包进反引号；内容含反引号时定界符加长。"""
    assert _MOD._inline_code("python") == "`python`"
    assert _MOD._inline_code("[x](u)") == "`[x](u)`"
    assert _MOD._inline_code("a`b") == "``a`b``"


def test_363_info_string_is_inert_in_report():
    """#363：正文 info string 里的 markdown/提及/HTML 必须只作字面量入报告。

    变异点：把 `_inline_code(k)` 改回裸 `k` ⇒ 本断言红（注入串不带反引号出现）。
    """
    payloads = {
        "link": "[click](https://evil.example)",
        "mention": "@everyone",
        "html": "<img>",
    }
    body = ("正文。\n"
            "```%s\n```\n" % payloads["link"]
            + "```%s\n```\n" % payloads["mention"]
            + "```%s\n```\n" % payloads["html"])
    rc, report = _run_main(body)
    line = _extract_line(report)
    assert rc == 2, "无 python 块应显式失败（退出码 2），实得 %r\n%s" % (rc, report)
    for name, pl in payloads.items():
        assert ("`%s`" % pl) in line, (
            "#363：%s 注入串未以行内代码入报告（可被渲染）⇒ 判据失守\n抽取行：%r"
            % (name, line))
        # 反向：不得存在**未**被反引号包住的裸形态
        assert ("：%s×" % pl) not in line and ("/ %s×" % pl) not in line, (
            "#363：%s 出现了未包反引号的裸形态\n抽取行：%r" % (name, line))


# ── #235 ────────────────────────────────────────────────────────────────────
def test_235_fence_inside_html_comment_is_invisible():
    """#235：HTML 注释里的围栏不算块（网页不可见 ⇒ 不得被抽取执行）。

    变异点：`_extract_fenced_blocks` 去掉注释状态跟踪（改回按原文行无脑匹配）
    ⇒ 本断言红。
    """
    hidden_only = ("可见文字。\n"
                   "<!-- 说明\n"
                   "```python\n"
                   "print('HIDDEN')\n"
                   "```\n"
                   "-->\n")
    assert _MOD._extract_fenced_blocks(hidden_only) == [], (
        "#235：HTML 注释里的围栏被当成块 ⇒ 网页不可见的代码会被执行")
    rc, report = _run_main(hidden_only)
    assert rc == 2, "注释内 python 块不得被抽取执行，实得 rc=%r\n%s" % (rc, report)
    assert "未抽取" in _extract_line(report)

    # 混排：只抽可见块；注释块不参与计数；行号仍指正文真实行
    mixed = ("```python\nprint('VISIBLE')\n```\n"
             "<!--\n"
             "```python\nprint('HIDDEN')\n```\n"
             "-->\n")
    blocks = _MOD._extract_fenced_blocks(mixed)
    assert len(blocks) == 1, "应只有 1 个可见块，实得 %r" % (blocks,)
    lang, first, last, code = blocks[0]
    assert code == "print('VISIBLE')" and (first, last) == (2, 2)
    chosen = _MOD._choose_python_block(blocks)
    assert chosen is not None and "HIDDEN" not in chosen[3]

    # 单行闭合的注释也须生效
    assert _MOD._extract_fenced_blocks("<!-- ```python -->\nx\n") == [], (
        "同一行内闭合的注释未掩 ⇒ 注释里的围栏仍被当块")

    # 反向：围栏**代码里**的 `<!--` 是字面量、不得被当注释（否则会吞掉真块）
    in_fence = '```python\nx = "<!--"\ny = 2\n```\n'
    fb = _MOD._extract_fenced_blocks(in_fence)
    assert len(fb) == 1 and fb[0][3] == 'x = "<!--"\ny = 2', (
        "围栏代码里的 `<!--` 被误当注释 ⇒ 真块被吞：%r" % (fb,))


# ── #241 ────────────────────────────────────────────────────────────────────
def _workflow():
    yaml = pytest.importorskip("yaml", reason="需要 PyYAML（dev extra）解析 workflow")
    with open(_WORKFLOW, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_241_exec_job_holds_no_issue_token():
    """#241：跑不可信代码的 job 必须与持 issue 令牌的 job 分属不同 runner。

    变异点：把 workflow 改回单 job（回评步骤与执行步骤同 job）⇒ 本断言红。
    """
    d = _workflow()
    jobs = d["jobs"]
    assert {"prep", "exec", "report"} <= set(jobs), (
        "#241：未分 job（需 prep/exec/report 三 job），实得 %r" % (list(jobs),))

    exec_job = jobs["exec"]
    perms = exec_job.get("permissions") or {}
    assert perms.get("issues") != "write", (
        "#241：exec job 持有 issues: write ⇒ 被执行代码所在 runner 能改 issue")
    exec_text = repr(exec_job)
    assert "GH_TOKEN" not in exec_text, (
        "#241：exec job 的步骤里出现 GH_TOKEN ⇒ 令牌与不可信代码同 job\n%s" % exec_text)

    # 回评/打标签只在 report job；且它 needs exec（不同 runner）
    assert "gh issue comment" in repr(jobs["report"]), "report job 应负责回评"
    assert "gh issue comment" not in exec_text, "exec job 不得回评（那是持令牌的动作）"
    assert jobs["report"].get("needs") == "exec"
    assert (jobs["report"].get("permissions") or {}).get("issues") == "write"

    # 正文/报告只经 artifact 跨 job 交换（exec 不直连 prep 的文件面）
    assert "actions/download-artifact" in exec_text
    assert "actions/upload-artifact" in repr(jobs["prep"])
    assert "actions/upload-artifact" in exec_text


def test_241_cleanup_scope_is_documented_honestly():
    """#241：执行核 docstring 须如实写明 env 清洗的作用域（不得再声称 job 级隔离）。

    变异点：把 docstring 里「只清子进程自己的 env / 同 job 不隔离」的声明删回
    「拿不到 $GITHUB_ENV」的原断言 ⇒ 本断言红。
    """
    doc = _MOD.__doc__ or ""
    assert "/proc/" in doc, (
        "#241：docstring 未点明同 uid 可经父进程 environ（/proc/<ppid>/environ）取回")
    assert "同 job" in doc, "#241：docstring 未点明「同 job 内不隔离」"
    # 执行核仍必须真的清掉子进程自身的 CI 注入面键（原功能不得回退）
    os.environ["GITHUB_ENV"] = "/tmp/ge"
    os.environ["GITHUB_PATH"] = "/tmp/gp"
    os.environ["SOME_TOKEN"] = "t"
    ce = _MOD._child_env({})
    assert "GITHUB_ENV" not in ce and "GITHUB_PATH" not in ce and "SOME_TOKEN" not in ce
    assert ce.get("PYTHONUTF8") == "1"
