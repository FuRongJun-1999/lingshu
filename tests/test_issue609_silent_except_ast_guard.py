# -*- coding: utf-8 -*-
"""issue #609 · silent_except_lint 静默判定 AST 化的守卫测试

缺陷（#609，P1）：scan_source 原对 `ast.unparse(handler 体)` 整体做 AUDIT_TOKENS
子串匹配 ⇒ 标识符里**恰好含** token 子串的**纯赋值**全被判"已记录"放行：
`fingerprint = None`（print）/ `catalog = []`（log）/ `logger = None`（log，
连调用都没有）/ `error_count += 1`（error，仅自增计数器）……
端到端对照（改前实测，HEAD 375cf55）：两个各含 10 个宽 except 的文件，
体写 `pass` 判红 exit 1、写 `fingerprint = None` 放行 exit 0——门禁对两者
无区分力。

修复（本守卫锁定的行为）：静默判定改为 **AST 结构判定**（handler_has_audit_action）：
  a) ast.Call 且被调函数**名字链**（Name.id / 各级 Attribute.attr）任一环节含
     AUDIT_TOKENS 任一子串；
  b) ast.Raise 语句。
基线已随判据收紧重测（core.py 46→76、TOTAL 114→144，见
tools/silent_except_lint_baseline.txt 的 #609 重测记录）。

断言组
------
G1 子串放行面（纯赋值伪记录必须判**静默**）：8 个标识符含 token 子串的
   纯赋值/自增 handler，全部须 silent == 1；
G2 真记录面（须判**有记录动作**放行）：print / logger.warning / raise /
   logging.error / self.log 全部须 silent == 0；
G3 端到端：改前读数的 d_many/e_pass 对照——修复后 d_many 也必须 exit 1；
G4 全仓零超基线：出货基线下完整跑一次必须 exit 0。

变异锚点：把 handler_has_audit_action 的调用改回 unparse 子串匹配
（文本替换回 `if any(tok in ast.unparse(node.body) for tok in AUDIT_TOKENS):
continue`）⇒ G1 全红、G3 的 d_many 断言红；恢复 ⇒ 全绿。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "tools" / "silent_except_lint.py"

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


def module_with_handler(body: str) -> str:
    """构造一个恰好含 1 个宽 except（体 = body）的模块源码字符串。"""
    return ("def f():\n"
            "    try:\n"
            "        g()\n"
            "    except Exception:\n"
            "        " + body + "\n")


def write_probe(directory: Path, name: str, body: str, count: int = 10) -> Path:
    """构造含 count 个宽 except（体均为 body）的 probe 文件（G3 端到端用）。"""
    directory.mkdir(parents=True, exist_ok=True)
    lines = ["def f():", "    try:", "        g()"]
    for _ in range(count):
        lines += ["    except Exception:", "        " + body]
    path = directory / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------- G1 子串放行面：纯赋值伪记录必判静默
G1_BODIES = [
    "fingerprint = None",    # "print" ⊂ fingerprint
    "footprint = ''",        # "print" ⊂ footprint
    "printing = True",       # "print" ⊂ printing
    "catalog = []",          # "log" ⊂ catalog
    "logic = 0",             # "log" ⊂ logic
    "logger = None",         # "log" ⊂ logger——连调用都没有，最尖锐
    "warning = 'x'",         # "warn" ⊂ warning
    "error_count += 1",      # "error" ⊂ error_count——仅自增计数器
]

# 真仓实读补充（#609 重测中 core.py 新计入的 30 条即此类）：token 只出现在
# 字典键名/字符串字面量/无 token 调用名里，无任何日志/告警/上抛。
G1_EXTRA_BODIES = [
    "self._vision_error = str(e)",                                        # 纯属性赋值
    "return {'status': 'error', 'error': str(e)}",                        # 字符串字面量/键名
    "info.update(status='failed', error=str(e))",                         # 调用名 update 无 token
    "x = 'please log this'",                                              # token 只在字符串里
]


def test_g1_substring_false_positive_assignments_are_silent():
    """★G1：标识符含 token 子串的纯赋值/自增 ⇒ 必须判静默（旧口径全放行 = 缺陷面）。"""
    for body in G1_BODIES + G1_EXTRA_BODIES:
        res = S.scan_source(module_with_handler(body))
        assert res["wide"] == 1, (body, res)
        assert res["silent"] == 1, (
            "纯赋值伪记录被放行（#609 缺陷复发）：%r ⇒ %r" % (body, res))


# ---------------------------------------------- G2 真记录面：须判有记录动作放行
G2_BODIES = [
    "print('x')",             # Name 链 ["print"]
    "logger.warning('x')",    # 名字链 ["logger", "warning"]——attr 命中 "warn"
    "raise",                  # ast.Raise 结构性判定
    "logging.error('x')",     # 名字链 ["logging", "error"]
    "self.log('x')",          # 属性调用 attr "log" 命中
]

G2_EXTRA_BODIES = [
    "logging.exception('x')",                     # 名字链经 "logging"（含 "log"）命中
    "raise ValueError('x')",                      # 带异常类的 raise
    "self._vision_error = str(e)\n        print(e)",  # 纯赋值 + 真调用混排 ⇒ 有记录
    "get_printer()('x')",                         # 内层调用名 get_printer 含 "print" ⇒ 有记录
]


def test_g2_real_recording_actions_are_not_silent():
    """★G2：真调用/真 raise ⇒ 必须判有记录动作（silent == 0），不得因收紧而误伤。"""
    for body in G2_BODIES + G2_EXTRA_BODIES:
        src = module_with_handler(body)
        res = S.scan_source(src)
        assert res["wide"] == 1, (body, res)
        assert res["silent"] == 0, ("真记录动作被误判静默（收紧过头）：%r ⇒ %r"
                                    % (body, res))


def test_g2_call_without_token_name_is_silent():
    """对照：调用存在但名字链无 token（如 info.update）⇒ 仍判静默（收紧方向正确）。"""
    res = S.scan_source(module_with_handler("info.update(status='failed')"))
    assert res["wide"] == 1 and res["silent"] == 1, res


# ---------------------------------------------- G3 端到端：d_many 修复后必须判红
def test_g3_d_many_fingerprint_now_fails_closed(tmp_path):
    """★G3：改前读数对照——体写 `fingerprint = None` 的 10 宽 except 文件，
    修复前放行 exit 0（静默 0），修复后必须 exit 1（静默 10 > 基线 0）。
    """
    root = tmp_path / "repo"
    write_probe(root / "lingshu" / "probe", "d_many.py", "fingerprint = None")
    write_probe(root / "lingshu" / "probe", "e_pass.py", "pass")
    baseline = root / "tools" / "silent_except_lint_baseline.txt"
    baseline.parent.mkdir(parents=True, exist_ok=True)
    baseline.write_text("", encoding="utf-8")   # 空基线：静默即违规

    for name in ("d_many.py", "e_pass.py"):
        res = run_lint(str(root / "lingshu" / "probe" / name),
                       "--repo-root", str(root), "--json")
        assert res.returncode == 1, ("%s 必须判红（exit 1）" % name, res.stdout)
        data = json.loads(res.stdout.decode("utf-8"))
        rel = "lingshu/probe/%s" % name
        assert any(o["file"] == rel and o["excess"] == 10
                   for o in data["new_over_baseline"]), data
        assert data["summary"]["silent_total"] == 10, data


# ---------------------------------------------- G4 全仓零超基线
def test_g4_whole_repo_zero_over_baseline():
    """★G4：出货基线（#609 重测后）下全仓跑 ⇒ exit 0、无超基线、无 unparsable。"""
    res = run_lint("--json")
    assert res.stdout, res.stderr[:2000]
    data = json.loads(res.stdout.decode("utf-8"))
    assert res.returncode == 0, data
    assert data["new_over_baseline"] == [], data
    assert data["actual_total"] == data["baseline_total"], data
    #: ★2026-10-15（PR #615）146 → 144：**真实修复导致的下降**，不是放宽判据——
    #: `lingshu/core/core.py` 78→77（FTS 索引挂载腿补 `warnings.warn`）、
    #: `lingshu/core/retrieval.py` 1→0（`_doc()` 宽 except 改窄为 `(TypeError, ValueError)`）。
    #: ★2026-10-16（`286b428` · issue #292 追加变体）144 → 140：同款**真实修复**——
    #: `lingshu/core/longterm_gate.py` 11→7（`prefeed` 主写入改 fail-closed 回报、
    #: 建边/`protect_node` 两处降级腿与 `write_snapshot` 的「已存在」腿各补 `_log.warning`）。
    #: 反向保护未动：上面 `actual_total == baseline_total` 与 `new_over_baseline == []` 仍是硬判据
    #: （把计数调回高值或新增静默宽 except 都会立刻判红）。
    #: 该地板值**跟随每一次真实下降同步**：改基线文件而不动本行 ⇒ 本断言当场判红
    #: （`286b428` 就漏了这一步 ⇒ 三条 job 同时红，已补）。
    #: ★2026-11-06（#170 清偿第一批）140 → 139：`lingshu/world/prediction.py`
    #: 14 → 0（全 14 处补 logger 审计，控制流零变化；基线文件同步删行）。
    #: ★2026-11-06（#170 清偿第二批）129 → 128：小额四文件 11 处
    #: （模块级回落 6 处补 logger.debug ＋ provenance 4 处收窄
    #: (ValueError, TypeError)——基线文件同步删四行）。
    assert data["actual_total"] >= 128, ("基线合计低于当前登记值 128"
                                         "（#170 清偿两批累计 −16：第一批 prediction.py −14＋第二批小额 11 处中净 −2 计入合计口径差），"
                                         "判据可能被偷偷放宽")
    assert data["unparsable"] == [], data
    # #609 重测的落点：core.py 计数已同步进基线（46→76；#653 再 76→78，
    # `_tolerant_edges` 边侧逐行容错新增 2 条，见基线文件 #653 登记行）
    core = next(r for r in data["per_file"] if r["file"] == "lingshu/core/core.py")
    #: 78 → 77：同上（本笔给 FTS 挂载腿补 `warnings.warn` ⇒ 该处不再静默）。
    assert core["silent"] == core["baseline"] == 77, core


def test_g4_baseline_file_documents_609_retest():
    """基线文件保留 #609 重测记录（历史口径可追溯，不只有数字）。"""
    text = (REPO_ROOT / "tools" / "silent_except_lint_baseline.txt").read_text(
        encoding="utf-8")
    assert "#609 判定 AST 化后重测" in text, text
    # #653 起基线值 76→78（`_tolerant_edges` 新增 2，见基线 #653 登记行）——
    # 本断言跟随当前基线值，追溯链 = #609 历史区「46→76」＋ #653 登记行「76→78」。
    #: 78 → 77：本笔真实修复（FTS 挂载腿补 `warnings.warn`）；追溯链 = #609「46→76」
    #: ＋ #653「76→78」＋ 本笔「78→77（真实修复）」。
    assert "lingshu/core/core.py\t77" in text, text
