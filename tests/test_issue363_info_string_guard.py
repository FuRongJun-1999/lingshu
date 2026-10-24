"""#363 守卫：正文围栏 info string 不得在报告**围栏之外**产出可渲染的
粗体 / 外链 / @提及 / 原始 HTML。

判据取 issue #363 的原文口径：**围栏之外**不得出现粗体 / 外链 / @提及（三项全 False）。
注意这里刻意**不**先剥行内代码 span —— 判据要的是「报告文本里有没有这些字符」，
不是「Markdown 渲染器会不会把它变成活链接」。理由：封口若只靠 `_inline_code` 包一层，
一旦某次改动漏包一处（历史已发生：tally 与 extract_info 是两处调用点），
裸串就会直接落到围栏外。白名单在**数据源头**掐掉非白名单 info string，
使报告里根本不存在这些字符，判据因此对「包没包」不敏感。

变异点（实测见 evidence/issue-363/verify.md 第 ④ 项）：
把 `_safe_lang_label` 的白名单判定改回直接返回 `lang` ⇒ 本文件断言成片变红。
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_SCRIPT = os.path.join(_REPO, "scripts", "extract_and_run_issue_repro.py")

_FENCE_LINE = re.compile(r"^\s*(`{3,}|~{3,})")


def _load_exec_core():
    """按**文件路径**加载执行核。

    不用 `sys.path.insert(scripts) + import extract_and_run_issue_repro`：
    该顶层名不在 `tests/test_gate_dependency_closure.py` 的 `LOCAL_TOP` 里，
    静态导入会被判成「未声明的第三方依赖」而使 gate 变红（实测 `50ee03b` 上
    `test_every_hard_third_party_import_is_declared` 红）。按路径加载则不进静态面。
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("_issue363_exec_core", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
_TEMPLATE_BOLD = ("抽取", "来源", "环境", "被测提交", "依赖", "执行", "退出码", "仅")


def outside_fence(report):
    """报告里**围栏块之外**的全部文本（含行内代码，不做任何剥除）。"""
    out, inside, ch = [], False, None
    for ln in report.splitlines():
        m = _FENCE_LINE.match(ln)
        if m:
            if not inside:
                inside, ch = True, m.group(1)[0]
                continue
            if m.group(1)[0] == ch:
                inside = False
                continue
        if not inside:
            out.append(ln)
    return "\n".join(out)


def hazards(report):
    """围栏之外的危险面：粗体 / 外链 / @提及 / 原始 HTML。"""
    z = outside_fence(report)
    for k in _TEMPLATE_BOLD:          # 模板自身固定的 **抽取** 等标签不算注入
        z = z.replace("**%s**" % k, "")
    return {
        "bold": re.findall(r"\*\*[^*\n]+\*\*", z),
        "link": re.findall(r"\[[^\]\n]*\]\([^)\n]*\)", z),
        "mention": re.findall(r"@[A-Za-z0-9_-]{2,}", z),
        "html": re.findall(r"</?[A-Za-z][A-Za-z0-9]*(\s[^>\n]*)?/?>", z),
    }


def _run(body):
    """用 --body-file 干跑执行核，返回 (rc, report)。"""
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8", newline="\n") as f:
        f.write(body)
        path = f.name
    try:
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        r = subprocess.run(
            [sys.executable, "-X", "utf8", _SCRIPT, "--body-file", path, "--dry-run"],
            cwd=_REPO, capture_output=True, text=True, encoding="utf-8",
            errors="replace", env=env, timeout=120,
        )
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


#: 攻击向量：info string 里塞粗体 / 外链 / @提及 / HTML，且首词伪装成 python。
PAYLOADS = [
    'python" **伪结论** [x](https://evil.example) @victim',
    "python**[伪结论](https://evil.example)**@victim",
    "python`[x](https://evil.example)`",
    "python<script>alert(1)</script> **伪结论**",
    "py**粗**@victim",
    "[x](https://evil.example)",
    "@everyone",
    "<img src=x onerror=alert(1)>",
]


def test_gauge_detects_bare_hazard():
    """量具自检：裸注入必须被检出（否则本守卫是空转的）。"""
    h = hazards("- a：**[B](u)**@mm <i>x</i>\n")
    assert h["bold"] and h["link"] and h["mention"] and h["html"], "量具失灵：裸注入没被检出"


@pytest.mark.parametrize("payload", PAYLOADS)
def test_363_no_renderable_hazard_outside_fence(payload):
    """#363：恶意 info string 不得在围栏之外产出可渲染内容（三项全 False）。"""
    body = "正文。\n\n```%s\nprint(1)\n```\n" % payload
    rc, report = _run(body)
    assert "### 自动复现读数" in report, "报告没生成，量具前提不成立：\n%s" % report
    h = hazards(report)
    assert not h["bold"], "#363：围栏外出现粗体 %r\n---\n%s" % (h["bold"], report)
    assert not h["link"], "#363：围栏外出现外链 %r\n---\n%s" % (h["link"], report)
    assert not h["mention"], "#363：围栏外出现 @提及 %r\n---\n%s" % (h["mention"], report)
    assert not h["html"], "#363：围栏外出现原始 HTML %r\n---\n%s" % (h["html"], report)


def test_363_legit_langs_still_recognized():
    """反向：白名单内的合法 info string 仍被正确识别为 python 块（不许修成一律不认）。"""
    for lang in ("python", "py", "py3", "python3", "Python"):
        body = "```%s\nprint('ok')\n```\n" % lang
        rc, report = _run(body)
        assert rc == 0, "%s 块应被抽取执行，实得 rc=%r\n%s" % (lang, rc, report)
        assert "DRY-RUN：已抽取并落盘" in report


def test_363_safe_label_is_whitelisted():
    """白名单口径：合法 info 首词原样保留，非白名单一律落成「未知」。"""
    mod = _load_exec_core()
    assert hasattr(mod, "_safe_lang_label"), "执行核缺 `_safe_lang_label` ⇒ #363 的白名单不存在"
    for ok in ("python", "py", "py3", "python3", "js", "c++", "a-b_c", "A1"):
        assert mod._safe_lang_label(ok) == ok, "白名单误杀合法 info：%r" % ok
    for bad in ('python"', "py**b**", "[x](u)", "@m", "<i>", "a b", "x" * 21, ""):
        assert mod._safe_lang_label(bad) == "未知", "非白名单 info 未被中和：%r ⇒ %r" % (
            bad, mod._safe_lang_label(bad))
