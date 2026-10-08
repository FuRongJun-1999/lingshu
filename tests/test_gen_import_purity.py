# -*- coding: utf-8 -*-
"""test_gen_import_purity · `lingshu/gen` 导入期不得改写进程级 `sys.path`
============================================================================
来历（两份上游报告 + 本仓自陈缺口）：
  · **issue #108**（`123WP-a`，基线 `d7c6b33`）：`lingshu/gen` 多个模块在 **import 期**
    把相对路径 `"."` / `"experiments"` 写入 `sys.path`；作者如实声明
    **「不主张已造成任何实际错误行为（未做该实验）」**、**「曾尝试检验但无法区分因果」**。
  · **issue #156 v2**：`_pathguard` 在 `import lingshu` 时消除 cwd/空串条目，以阻止
    「cwd 同名模块在进程内执行」；其自陈的「仍不覆盖」里逐字写着
    **「cwd 条目在 `lingshu` import **之后**被重新插入 `sys.path`（无持续守卫）」**。

本件把上述两件接上，并给出**可执行的判据**：
  A 导入 `lingshu/gen` 任一模块，`sys.path` **不得变化**（该模块自身即因变量）。
  B 导入这些模块后，**cwd 解析面不得被重新打开**——即放在 cwd 里的模块名
    仍必须 `ModuleNotFoundError`（`_pathguard` 的收口不得被同仓模块撤销）。

为什么用子进程：判据 A/B 都是"导入期对**进程全局状态**的作用"，必须在干净解释器里观测；
在 pytest 进程内做会污染 pytest 自身，且五个模块互相导入会产生累积读数（实测同一进程连导可净增 6 条）。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_TIMEOUT = 300
_MARK = "__R__"

# 5 个文件、8 处 `sys.path.insert`（issue #108 列出 7 处，实测第 8 处为
# `hexgen_subject_split.py`，该行在 #108 的基线 d7c6b33 上已存在）
_GEN_MODULES = (
    "lingshu.gen.hexgen_align_probe",
    "lingshu.gen.hexgen_multi_seed",
    "lingshu.gen.hexgen_pack_certify",
    "lingshu.gen.hexgen_self_source",
    "lingshu.gen.hexgen_subject_split",
)

_PURITY_PROBE = '''\
import json, sys
import lingshu                      # 先让 _pathguard 完成其**既定**收口（issue #156，非本件射程）
before = list(sys.path)             # 以"收口之后"为基线 ⇒ 只量**该 gen 模块自己**的改动
try:
    import %(mod)s
    err = None
except Exception as exc:
    err = type(exc).__name__ + ": " + str(exc)[:160]
after = list(sys.path)
print("%(mark)s" + json.dumps(
    {"added": [p for p in after if p not in before],
     "removed": [p for p in before if p not in after],
     "err": err}, ensure_ascii=False))
'''

_REOPEN_PROBE = '''\
import json, sys, os
import lingshu                      # 触发 _pathguard：消除 cwd/空串解析面
def probe(m):
    try:
        __import__(m); return "ok"
    except Exception as e:
        return type(e).__name__
closed = probe("cwdonly")           # 收口生效 => ModuleNotFoundError
err = None
try:
    import %(mod)s
except Exception as exc:
    err = type(exc).__name__ + ": " + str(exc)[:160]
after_reopen = probe("cwdonly")
print("%(mark)s" + json.dumps({"closed": closed, "reopen": after_reopen, "err": err},
                              ensure_ascii=False))
'''


def _child(source: str, cwd: str):
    env = {k: v for k, v in os.environ.items()
           if k not in ("LINGSHU_ALLOW_CWD_IMPORTS", "LINGSHU_COMPONENT_ROOT")}
    env["PYTHONPATH"] = str(REPO)
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", source],
        cwd=cwd, env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=_TIMEOUT,
    )
    line = next((ln for ln in (proc.stdout or "").splitlines() if ln.startswith(_MARK)), None)
    assert line is not None, (
        "探测子进程未产出读数。\n"
        f"  cwd={cwd}\n  rc={proc.returncode}\n  stdout={proc.stdout!r}\n  stderr={proc.stderr[-500:]!r}"
    )
    payload = line[len(_MARK):]          # ★ 必须按 marker 实际长度切（曾误写 [4:] ⇒ 假失败）
    try:
        return json.loads(payload)
    except Exception as exc:  # 脚手架自身故障要吵，不要伪装成判据失败
        raise AssertionError(
            f"探测读数无法解析（脚手架故障，非判据失败）：{exc}\n  payload={payload[:200]!r}"
        ) from exc


def test_importing_gen_modules_does_not_mutate_sys_path():
    """判据 A（issue #108）：导入 `lingshu/gen` 模块本身不得改动 `sys.path`。

    基线取 `import lingshu` **之后**——那一步的移除是 `_pathguard` 的**既定**行为
    （issue #156，本件不动它）；本判据只量该 gen 模块**自己**新增/移除了什么。
    """
    with tempfile.TemporaryDirectory(prefix="lingshu-gen-") as tmp:
        bad = {}
        for mod in _GEN_MODULES:
            r = _child(_PURITY_PROBE % {"mod": mod, "mark": _MARK}, cwd=tmp)
            # ★ 也要判 import 本身是否成功：否则"删多了导致导入失败"会静默变绿
            if r["added"] or r["removed"] or r["err"]:
                bad[mod] = r
        assert not bad, (
            "导入 lingshu/gen 模块改动了进程级 `sys.path`（issue #108）或导入失败：\n"
            + "\n".join(
                "  %s\n     新增=%s 移除=%s%s"
                % (m, v["added"], v["removed"], ("  ← import 失败: %s" % v["err"]) if v["err"] else "")
                for m, v in sorted(bad.items()))
            + "\n契约：库的导入不得改动宿主进程的全局状态；这些写入对本包的相对导入毫无必要。"
        )


def test_gen_import_does_not_reopen_cwd_surface():
    """判据 B（issue #156 自陈缺口）：导入 `lingshu/gen` 后 cwd 解析面仍须关闭。"""
    with tempfile.TemporaryDirectory(prefix="lingshu-cwd-") as tmp:
        # ★ 必须是**合法** python（曾误写 "\n" 为字面反斜杠 ⇒ 读数成 SyntaxError）
        (Path(tmp) / "cwdonly.py").write_text("MARK = 'host-cwd'\n", encoding="utf-8")
        for mod in _GEN_MODULES:
            r = _child(_REOPEN_PROBE % {"mod": mod, "mark": _MARK}, cwd=tmp)
            assert r["closed"] == "ModuleNotFoundError", (
                f"前置条件不成立：`import lingshu` 之后 cwd 面本应关闭，实测 {r['closed']!r}"
                "（若非 ModuleNotFoundError，本判据无区分力，须先查 _pathguard 行为）"
            )
            assert r["reopen"] == "ModuleNotFoundError", (
                f"导入 {mod} 之后，cwd 解析面被重新打开（issue #156 自陈缺口 + #108）：\n"
                f"  导入前：{r['closed']!r} → 导入后：{r['reopen']!r}\n"
                "含义：cwd 里与 sys.path 同名的模块又可被解析 ⇒ `_pathguard` 的收口被同仓模块撤销。"
            )


# 每模块一个**已知可用**的代表性入口（签名取自本仓实测；只断言"可调用、不抛异常"，
# ★ 不断言具体数值——那会把判据绑死在 numpy 版本上）
_SMOKE = {
    "hexgen_align_probe": ("main", None),
    "hexgen_pack_certify": ("support_vec", ("triangle", 10, 0.0, 0.0, 0.0)),
    "hexgen_self_source": ("determinism_check", ()),        # 该模块自带的确定性自检
    "hexgen_subject_split": ("proto_features", (32,)),
    "hexgen_multi_seed": ("summarize", ([], [])),
}


def test_gen_modules_representative_entrypoints_still_work():
    """判据 C：删除 import 期 `sys.path` 写入后，各模块的代表性入口仍可调用。

    本判据防的是"删多了把功能弄坏"，与 A/B 是不同关注点：
    A/B 只保证"导入干净"，**不保证模块仍可用**——故补此条。
    """
    src = (
        "import json, sys, warnings\n"
        "warnings.simplefilter('ignore')\n"
        "import importlib\n"
        f"SPEC = {_SMOKE!r}\n"
        "out = {}\n"
        "for short, (attr, args) in SPEC.items():\n"
        "    name = 'lingshu.gen.' + short\n"
        "    try:\n"
        "        m = importlib.import_module(name)\n"
        "    except Exception as exc:\n"
        "        out[name] = 'IMPORT_ERR:' + type(exc).__name__ + ':' + str(exc)[:80]\n"
        "        continue\n"
        "    fn = getattr(m, attr, None)\n"
        "    if not callable(fn):\n"
        "        out[name] = 'NO_CALLABLE:' + attr\n"
        "        continue\n"
        "    if args is None:\n"
        "        out[name] = 'ok(callable)'\n"
        "        continue\n"
        "    try:\n"
        "        fn(*args)\n"
        "        out[name] = 'ok'\n"
        "    except Exception as exc:\n"
        "        out[name] = 'CALL_ERR:' + type(exc).__name__ + ':' + str(exc)[:80]\n"
        "print('__R__' + json.dumps(out, ensure_ascii=False))\n"
    )
    with tempfile.TemporaryDirectory(prefix="lingshu-gen-smoke-") as tmp:
        r = _child(src, cwd=tmp)
    bad = {k: v for k, v in r.items() if not str(v).startswith("ok")}
    assert not bad, (
        "删除 `sys.path` 写入后，下列 gen 模块的代表性入口不可用：\n"
        + "\n".join("  %s: %s" % (k, v) for k, v in sorted(bad.items()))
        + "\n含义：本改动只应删除 import 期的路径写入，不应影响任何功能入口。"
    )
