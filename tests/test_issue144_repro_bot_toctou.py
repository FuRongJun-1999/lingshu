# -*- coding: utf-8 -*-
"""#144 守卫：repro-bot 安全模型两处——① TOCTOU（正文须取打标签时刻快照）、
② 令牌隔离的表述不得过度（不得再宣称「最坏后果=状态标签打反」/「令牌不进进程
环境」即可作安全前提）。

缺陷态（取证 · 修复前基线 `8a2de8f` 落仓版）：
  · ① `.github/workflows/repro-bot.yml` 的「取 issue 正文」步骤在**运行时**调
    `gh api "repos/…/issues/…" --jq .body` 重取正文——该步排在 checkout/setup-python/
    `pip install`/clone 脑端**之后**（数十秒～一两分钟），报告人可在「维护者审完并
    打标签 → 执行」之间改正文，把被执行的脚本换成另一份（维护者审的 ≠ 机器人跑的）。
    无 `updated_at` 比对、无正文哈希。
  · ② `docs/repro-bot.md` 与 yml 头注把「令牌不进被测脚本的进程环境」当成安全前提，
    并把它推导成「最坏后果=状态标签打反」（记为接受项）。实际隔离只有两样：执行
    步骤不挂 `GH_TOKEN` + 执行核按键名清洗子进程 env（`_ENV_DROP_PREFIXES` /
    `_child_env`）——**只挡「读环境变量」一条路**；同 job 后续步骤（`$GITHUB_ENV`/
    `$GITHUB_PATH` 注入）与残留孙进程仍在。

判据（#144 报告与维护者 2026-10-09 回应）：
  ① 正文须取**打标签那一刻的事件载荷快照**（`$GITHUB_EVENT_PATH` 的 `.issue.body`），
     不在运行时重取；「审的正文」与「跑的正文」必须是同一份。
  ② 安全表述须与真实隔离面一致：不得再出现「最坏后果=状态标签打反」这类接受项；
     须如实写明 env 清洗**不是 job 隔离**（真实边界是执行/回评分 job，见 #241）。
  来源：issue #144（报告人 cyd471，2026-10-08）＋维护者回应；属**经验标定，
  追不到理论出处**。

变异点：把 `prep` 的正文来源改回运行时 `gh api … --jq .body`（或在 `prep` 挂
`GH_TOKEN`）⇒ 本文件 test_body_comes_from_label_time_snapshot / test_prep_holds_no_token
必红；把 docs 的「最坏后果=状态标签打反」接受项加回 ⇒ test_no_overclaim_in_docs 红。

运行（lingshu 仓根）：python -X utf8 -m pytest tests/test_issue144_repro_bot_toctou.py -q
"""
from __future__ import annotations

import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WORKFLOW = os.path.join(REPO, ".github", "workflows", "repro-bot.yml")
_DOCS = os.path.join(REPO, "docs", "repro-bot.md")


def _workflow_text():
    with open(_WORKFLOW, encoding="utf-8") as f:
        return f.read()


def _docs_text():
    with open(_DOCS, encoding="utf-8") as f:
        return f.read()


def _yaml():
    yaml = pytest.importorskip("yaml", reason="需要 PyYAML（dev extra）解析 workflow")
    with open(_WORKFLOW, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _steps_text(job):
    """把 job 的 steps 渲染成可搜索文本（含 run/env）。"""
    return repr(job.get("steps") or [])


# ── 量具自检 ────────────────────────────────────────────────────────────────
def test_gauge_is_alive():
    d = _yaml()
    assert "prep" in d["jobs"], "prep job 不在（#241 的分 job 结构）"
    assert os.path.isfile(_DOCS)


# ── ① TOCTOU：正文取打标签时刻快照，不在运行时重取 ─────────────────────────
def test_body_comes_from_label_time_snapshot():
    """prep 的正文须来自 `$GITHUB_EVENT_PATH`（打标签时刻载荷），不得运行时重取。

    变异点：改回 `gh api "repos/…/issues/…" --jq .body` ⇒ 本断言红。
    """
    d = _yaml()
    prep = d["jobs"]["prep"]
    txt = _steps_text(prep)
    assert "GITHUB_EVENT_PATH" in txt, (
        "#144①：prep 未从打标签时刻的事件载荷（$GITHUB_EVENT_PATH）取正文 ⇒ "
        "审的正文与跑的正文可能不是同一份")
    assert ".issue.body" in txt, (
        "#144①：prep 未从事件载荷读 `.issue.body`")
    # 事件载荷文本不得经 shell 内插进 run（对齐 pr-lint.yml 的注入面纪律）
    assert "${{ github.event.issue.body }}" not in txt, (
        "#144①：正文被直接内插进 shell 正文 ⇒ 注入面（应用 jq/python 读文件）")


def test_no_runtime_refetch_of_issue_body():
    """workflow 的**任何 run 步骤**都不得在运行时调 `gh api … /issues/…` 重取正文。

    （只看 yaml 的 `run:` 正文，不误伤头注里对旧缺陷的说明文字。）
    """
    d = _yaml()
    runs = []
    for job in d["jobs"].values():
        for st in (job.get("steps") or []):
            if isinstance(st, dict) and isinstance(st.get("run"), str):
                runs.append(st["run"])
    joined = "\n".join(runs)
    assert "--jq .body" not in joined, (
        "#144①：仍在运行时用 `gh api … --jq .body` 重取正文 ⇒ TOCTOU 窗口未闭")
    assert "issues/${{ github.event.issue.number }}" not in joined, (
        "#144①：仍按 issue 号调 API 取正文 ⇒ 未用事件载荷快照")
    # 正文须来自事件载荷
    assert "GITHUB_EVENT_PATH" in joined and ".issue.body" in joined, (
        "#144①：run 步骤里没有从事件载荷读正文的动作")


def test_prep_holds_no_token():
    """prep 只读本机事件载荷 ⇒ 不应持有任何令牌（变异点：给 prep 挂 GH_TOKEN）。"""
    d = _yaml()
    prep = d["jobs"]["prep"]
    txt = _steps_text(prep)
    assert "GH_TOKEN" not in txt, (
        "#144②：prep 挂了 GH_TOKEN，但它只读本机事件载荷文件、无需令牌 ⇒ "
        "令牌面被无谓放大")
    assert (prep.get("permissions") or {}).get("issues") != "write", (
        "#144②：prep 持有 issues: write（它只落盘正文）")


# ── ② 表述：不得再拿「进程环境」当安全前提 ─────────────────────────────────
def test_exec_job_has_no_issue_token():
    """exec（跑不可信代码）不得持 issues: write、任一步骤不得挂 GH_TOKEN。"""
    d = _yaml()
    exec_job = d["jobs"]["exec"]
    assert (exec_job.get("permissions") or {}).get("issues") != "write", (
        "#144②：exec job 持有 issues: write ⇒ 令牌与不可信代码同 runner")
    assert "GH_TOKEN" not in _steps_text(exec_job), (
        "#144②：exec job 的步骤里出现 GH_TOKEN")


def test_no_overclaim_in_docs():
    """docs 不得再把「令牌不进进程环境」推导成「最坏后果=状态标签打反」接受项。

    变异点：把「最坏后果=状态标签打反」接受项加回 ⇒ 本断言红。
    """
    docs = _docs_text()
    assert "最坏后果=状态标签打反" not in docs and "状态标签打反" not in docs, (
        "#144②：docs 仍宣称「最坏后果=状态标签打反」——真实影响远不止（令牌可经"
        "同 job 后续步骤/残留孙进程/runner 进程内存触达）")
    # 必须如实写明 env 清洗不是 job 隔离
    assert "不是 job 隔离" in docs, (
        "#144②：docs 未点明执行核的 env 清洗**不是** job 隔离（真实边界是分 job）")


def test_docs_record_token_residual_honestly():
    """docs 须如实记录 exec runner 自身作业令牌这条残余（#144② 路径(b)）。"""
    docs = _docs_text()
    assert "Runner.Worker" in docs, (
        "#144②：docs 未记录 exec runner 自身作业令牌存于 runner 进程内存这条残余")
    assert "contents: read" in docs and "残余" in docs, (
        "#144②：残余项未标注影响上界（exec 只有 contents: read，无 issues: write）")


if __name__ == "__main__":
    raise SystemExit(__import__("pytest").main([__file__, "-q"]))
