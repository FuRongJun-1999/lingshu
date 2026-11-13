# -*- coding: utf-8 -*-
"""#80 守卫：intake 文档声明的「复跑命令」必须在本仓内可执行。

缺陷态（issue #80 · #8）：
`docs/intake/body-export-v0.2/README.md` §六「复跑」的 ```bash 代码块里写着
    python <AEIS>/tools/public_export/smoke.py <本仓根>
`<AEIS>` 是**未定义的占位符**，且该工具不随本仓导出 ⇒ **代码块里的命令不可执行**。
即使旁边加了「本仓内不可执行」的注释，**代码块本身仍是给读者复制粘贴的东西**——
与同文档「结论均以能被独立复跑为准」直接冲突。

★ 判据口径（这条是判别力的关键）：
   · **代码块内** = 读者会直接复制执行的命令 ⇒ **一律不得含 `<AEIS>` 占位符**（无例外）
   · **表格/散文** = 说明性引用 ⇒ 允许出现，但必须标注「不可执行 / 需上游工具」

修前实测（锚 6146c780）：回滚 README 后本守卫 4 条红；
修后 6 条全绿。判别力已三段验证（见 evidence/issue-80/guard_discrimination.txt）。

运行：`python -X utf8 -m pytest -q tests/test_issue80_intake_rerun_executable_guard.py`
"""
import os as _os
import re
import subprocess
import sys as _sys

_RP = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # 仓根
_sys.path.insert(0, _RP)

import pytest  # noqa: E402

README = _os.path.join(_RP, "docs", "intake", "body-export-v0.2", "README.md")
SMOKE = _os.path.join(_RP, "tools", "intake_smoke_min.py")


def _read_readme():
    assert _os.path.exists(README), "README 不存在：%s" % README
    with open(README, encoding="utf-8") as fh:
        return fh.read()


def _code_blocks(txt):
    """返回 [(起始行号, [行...])] —— 全部 ``` 围栏块的内容（不含围栏行）。"""
    out, cur, start = [], None, 0
    for i, line in enumerate(txt.splitlines(), 1):
        if line.strip().startswith("```"):
            if cur is None:
                cur, start = [], i + 1
            else:
                out.append((start, cur))
                cur = None
        elif cur is not None:
            cur.append((i, line))
    return out


# ---------------------------------------------------------------- ① 核心：代码块
def test_no_aeis_placeholder_inside_any_code_block():
    """★ 核心判据：任何代码块内都不得出现 `<AEIS>` —— 那是不可执行的命令。

    修前 §六 代码块第 80 行即 `python <AEIS>/tools/public_export/smoke.py <本仓根>`。
    这条**不因旁边有「不可执行」注释而放过**：代码块是给读者复制的。
    """
    txt = _read_readme()
    offenders = []
    for start, lines in _code_blocks(txt):
        for ln, line in lines:
            if "<AEIS>" in line:
                offenders.append((ln, line.strip()[:120]))
    assert not offenders, (
        "代码块内仍有 <AEIS> 占位符 ⇒ 读者复制后无法执行（这些行：%r）" % (offenders,))


def test_rerun_code_block_is_executable_in_repo():
    """§六 复跑代码块里，每条 `python <路径>` 的目标文件必须在本仓内存在。"""
    txt = _read_readme()
    m = re.search(r"## 六、复跑(.*?)```(.*?)```", txt, re.S)
    assert m, "README 未找到 §六 的 bash 代码块"
    block = m.group(2)

    missing = []
    for line in block.splitlines():
        line = line.strip()
        if not line.startswith("python "):
            continue
        target = None
        for p in line.split()[1:]:
            if p.startswith("-"):
                continue
            target = p
            break
        if not target or not target.endswith(".py"):
            continue  # python -c / -m 等
        if not _os.path.exists(_os.path.join(_RP, target.replace("/", _os.sep))):
            missing.append(target)
    assert not missing, (
        "§六 复跑命令引用了本仓内不存在的脚本 ⇒ 读者无法复跑：%r" % (missing,))


# ---------------------------------------------------------------- ② 散文/表格：允许但须标注
def test_prose_aeis_occurrences_are_marked_unexecutable():
    """代码块**之外**的 `<AEIS>` 允许存在（说明性），但必须标注不可执行。"""
    txt = _read_readme()
    in_block = set()
    for start, lines in _code_blocks(txt):
        for ln, _ in lines:
            in_block.add(ln)

    for i, line in enumerate(txt.splitlines(), 1):
        if "<AEIS>" not in line or i in in_block:
            continue
        assert ("不可执行" in line or "需上游工具" in line), (
            "第 %d 行（非代码块）含 <AEIS> 但未标注不可执行：%r" % (i, line.strip()[:140]))


# ---------------------------------------------------------------- ③ 本仓脚本
def test_inrepo_smoke_script_exists():
    """本仓内等价复跑脚本必须存在（README 点名了它）。"""
    assert _os.path.exists(SMOKE), (
        "README 点名的本仓复跑脚本不存在：%s ⇒ 「复跑命令」再次落空" % SMOKE)


def test_readme_points_to_inrepo_smoke_script():
    """README 必须真的点名本仓脚本，而不只是「碰巧有个脚本」。"""
    assert "tools/intake_smoke_min.py" in _read_readme(), (
        "README 未点名本仓内复跑脚本 tools/intake_smoke_min.py")


def test_inrepo_smoke_script_runs_and_passes():
    """「可执行」= 真跑通且返回 0，不是「文件存在」。"""
    env = dict(_os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [_sys.executable, "-X", "utf8", SMOKE, "--quiet"],
        cwd=_RP, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=900, env=env)
    assert proc.returncode == 0, (
        "本仓冒烟脚本返回 %d（应 0）\nstdout=%s\nstderr=%s"
        % (proc.returncode, (proc.stdout or "")[-1500:], (proc.stderr or "")[-1500:]))
    assert "fail=0" in (proc.stdout or ""), (
        "冒烟输出未见 fail=0：%r" % ((proc.stdout or "")[-400:],))
