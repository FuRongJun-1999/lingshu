# -*- coding: utf-8 -*-
"""wm_bridge.factory.film._det.iter_loop · P7 负记忆台账与回归拦截门（纯标准库 · 零 bpy）

★ 这一层实现什么（理论架构 §一 命题 6 + §七 P7）
    1. **声明式判据求值** (`evaluate_custom_criterion`)：
       从 `scene.truth[].measured` 回读实测字段（如 `coverage_actual`、
       `silhouette_delta_L`、`color_temp_est_k`、`histogram.p50` 等），按 `wmfilm.iteration/1`
       中声明的比较/区间算子输出四态：
         - `G1` 硬门：通过 ⇒ `ACCEPT`，越界 ⇒ `REJECT`
         - `G2` 软门：通过 ⇒ `ACCEPT`，越界 ⇒ `ACCEPT` + `WARN[G2 软门告警]`（不拦截）
         - `G3` 风格：只登记 `DEFER`（含空 `measured`）
         - 缺实测字段 / 缺 `truth` ⇒ **严守 §7.2.1 核心不变式：`BLINDSPOT`，绝不静默变成 `ACCEPT`**
    2. **判别力双相自证** (`verify_iteration_discrimination`)：
       新判据归档前，必须在历史缺陷快照 `bad_scene` 上触发 `REJECT`（或 G2 告警），
       且在修复快照 `fixed_scene` 上转为无告警 `ACCEPT`；否则拒绝归档（防假判据）。
    3. **防篡改负记忆归档链** (`append_iteration_ledger` / `verify_ledger_chain`)：
       每笔归档记录携带 `prev_digest`、`iteration_digest` 与 `entry_digest` 构成 sha256 哈希链。
    4. **确定性重放门禁** (`replay_ledger_against_scene`)：
       将归档台账在任意场景快照上重放，装配为通过 `wmfilm.oracle/1` 校验的裁定文档。
"""
from __future__ import annotations

from .. import _common as C
from .. import iteration as IT
from .. import oracle as OR
from . import gate_all as GA
from . import scene as S

#: 创世前驱摘要（64 位 0，仅 seq=0 使用）
GENESIS_DIGEST = "0" * 64

_ENTRY_KEYS = ("seq", "prev_digest", "iteration_digest", "entry_digest",
               "iteration", "discrimination")

__all__ = [
    "GENESIS_DIGEST", "IterLoopError",
    "criterion_thresholds", "is_soft_warning",
    "evaluate_custom_criterion", "verify_iteration_discrimination",
    "assert_ledger_valid", "verify_ledger_chain",
    "append_iteration_ledger", "replay_ledger_against_scene",
]


class IterLoopError(C.ContractError):
    """P7 迭代闭环执行/归档失败（fail-closed）。"""


def is_soft_warning(check):
    """判断一条 oracle check 是否为 G2 软门越界告警。"""
    if not isinstance(check, dict):
        return False
    return (check.get("verdict") == "ACCEPT"
            and str(check.get("reason", "")).startswith("WARN[G2"))


def criterion_thresholds(criterion):
    """将规范化判据展开为 `wmfilm.oracle/1` 可登记的阈值条目列表。"""
    nc = IT.normalize_criterion(criterion)
    if nc["gate"] == "G3" or "op" not in nc:
        return []
    tname = nc["threshold_name"]
    op = nc["op"]
    if op in ("between", "outside"):
        lo, hi = nc["range"]
        return [
            {"name": tname, "value": lo,
             "source": "%s [区间下界 %r..%r]" % (nc["source"], lo, hi)},
            {"name": "%s_hi" % tname, "value": hi,
             "source": "%s [区间上界 %r..%r]" % (nc["source"], lo, hi)},
        ]
    return [{"name": tname, "value": nc["threshold"], "source": nc["source"]}]


def _resolve_metric(measured_map, metric_path):
    """按点路径从 truth.measured 解析数值；返回 (found: bool, valid_num: bool, value)."""
    cur = measured_map
    for part in metric_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False, False, None
        cur = cur[part]
    if cur is None:
        return False, False, None
    if not C.is_num(cur):
        return True, False, cur
    return True, True, float(cur)


def _eval_op(val, nc):
    op = nc["op"]
    if op == "ge":
        return val >= nc["threshold"] - 1e-12
    if op == "gt":
        return val > nc["threshold"]
    if op == "le":
        return val <= nc["threshold"] + 1e-12
    if op == "lt":
        return val < nc["threshold"]
    if op == "eq":
        return abs(val - nc["threshold"]) <= 1e-9
    lo, hi = nc["range"]
    if op == "between":
        return (lo - 1e-12) <= val <= (hi + 1e-12)
    if op == "outside":
        return val < lo or val > hi
    raise IterLoopError("未知算子 %r" % (op,))


def _format_rule(nc):
    op = nc["op"]
    if op in ("between", "outside"):
        return "%s [%r, %r]" % (op, nc["range"][0], nc["range"][1])
    return "%s %r" % (op, nc["threshold"])


def evaluate_custom_criterion(criterion, sc, target_shot_id=None):
    """在输入束 `sc` 上对单条声明式判据求值，返回 `wmfilm.oracle/1` check 条目。"""
    S.reject_unknown_scene(sc)
    nc = IT.normalize_criterion(criterion)
    name = nc["name"]

    if nc["gate"] == "G3":
        desc = nc.get("description") or nc["source"]
        return {
            "name": name,
            "verdict": "DEFER",
            "measured": [],
            "reason": "G3 风格判据（nature=style）仅登记供人裁，不拦截：%s" % desc,
        }

    sid = target_shot_id or nc.get("shot_id")
    if sid is not None:
        entries = S.truth_at(sc, sid)
    else:
        raw_truth = S.truth_of(sc)
        for i, e in enumerate(raw_truth):
            if not isinstance(e, dict):
                raise S.SceneError("scene.truth[%d] 须为对象，实为 %s" % (i, type(e).__name__))
            C.reject_unknown(e, S.TRUTH_KEYS, "scene.truth[%d]" % i)
        entries = raw_truth

    metric = nc["metric"]
    if not entries:
        return {
            "name": name,
            "verdict": "BLINDSPOT",
            "measured": [],
            "reason": "缺少真值条目（scene.truth 缺失或无目标镜头 %s），无法读取实测字段 %s"
                      % (sid or "*", metric),
        }

    readings = []
    missing_count = 0
    bad_vals = []
    for e in entries:
        m = S.measured(e)
        found, valid, val = _resolve_metric(m, metric)
        if not found:
            missing_count += 1
        elif not valid:
            bad_vals.append(val)
        else:
            readings.append(val)

    if bad_vals:
        return {
            "name": name,
            "verdict": "BLINDSPOT",
            "measured": readings,
            "reason": "实测字段 %s 含非有限数值 %r，无法裁定" % (metric, bad_vals[0]),
        }

    if missing_count > 0 or not readings:
        return {
            "name": name,
            "verdict": "BLINDSPOT",
            "measured": readings,
            "reason": "真值缺少实测字段 %s（§7.2.1 核心不变式：缺料绝不静默变成 ACCEPT）" % metric,
        }

    failed = [v for v in readings if not _eval_op(v, nc)]
    tname = nc["threshold_name"]
    rule_str = _format_rule(nc)
    if not failed:
        return {
            "name": name,
            "verdict": "ACCEPT",
            "measured": readings,
            "threshold": tname,
            "evidence": "实测 %s=%r 满足 %s" % (metric, readings, rule_str),
        }

    if nc["gate"] == "G1":
        return {
            "name": name,
            "verdict": "REJECT",
            "measured": readings,
            "threshold": tname,
            "reason": "违反 G1 硬门判据 %s：%s=%r 未满足 %s" % (name, metric, failed, rule_str),
            "evidence": "G1 硬门拦截（%s）" % nc["source"],
        }

    # G2 软门：越界只告警 + 带实测读数，不 REJECT
    return {
        "name": name,
        "verdict": "ACCEPT",
        "measured": readings,
        "threshold": tname,
        "reason": "WARN[G2 软门告警] %s 越界：%s=%r 未满足 %s（只告警不拦截）"
                  % (name, metric, failed, rule_str),
        "evidence": "G2 软门告警（%s）" % nc["source"],
    }


def verify_iteration_discrimination(iter_doc, bad_scene, fixed_scene):
    """验证迭代产出的判据在历史缺陷快照上必拦/告警，在修复快照上转绿放行。

    防止把没有判别力的假判据归档进负记忆台账。
    """
    n_iter = IT.normalize_iteration(iter_doc)
    S.reject_unknown_scene(bad_scene)
    S.reject_unknown_scene(fixed_scene)

    details = []
    discriminated = []
    for crit in n_iter["outcomes"]["new_criteria"]:
        c_bad = evaluate_custom_criterion(crit, bad_scene)
        c_fix = evaluate_custom_criterion(crit, fixed_scene)
        details.append({
            "name": crit["name"],
            "gate": crit["gate"],
            "bad_verdict": c_bad["verdict"],
            "fixed_verdict": c_fix["verdict"],
            "bad_warn": is_soft_warning(c_bad),
            "fixed_warn": is_soft_warning(c_fix),
        })

        if crit["gate"] == "G1":
            discriminated.append(crit["name"])
            if c_bad["verdict"] != "REJECT":
                raise IterLoopError(
                    "判据 %s（G1）在缺陷快照 bad_scene 上未触发 REJECT（实为 %s：%s），"
                    "无判别力，拒绝归档"
                    % (crit["name"], c_bad["verdict"], c_bad.get("reason", "")))
            if c_fix["verdict"] != "ACCEPT":
                raise IterLoopError(
                    "判据 %s（G1）在修复快照 fixed_scene 上未转为 ACCEPT（实为 %s：%s），"
                    "拒绝归档"
                    % (crit["name"], c_fix["verdict"], c_fix.get("reason", "")))
        elif crit["gate"] == "G2":
            discriminated.append(crit["name"])
            if c_bad["verdict"] == "BLINDSPOT" or not is_soft_warning(c_bad):
                raise IterLoopError(
                    "判据 %s（G2 软门）在缺陷快照 bad_scene 上未触发软告警（实为 %s：%s），"
                    "无判别力，拒绝归档"
                    % (crit["name"], c_bad["verdict"], c_bad.get("reason", "")))
            if c_fix["verdict"] != "ACCEPT" or is_soft_warning(c_fix):
                raise IterLoopError(
                    "判据 %s（G2 软门）在修复快照 fixed_scene 上未清除告警（实为 %s：%s），"
                    "拒绝归档"
                    % (crit["name"], c_fix["verdict"], c_fix.get("reason", "")))

    return {
        "passed": True,
        "iteration_id": n_iter["id"],
        "iteration_digest": IT.iteration_digest(n_iter),
        "discriminated_criteria": discriminated,
        "details": details,
    }


def _compute_entry_digest(seq, prev_digest, iteration_digest, discrimination):
    return C.digest_of({
        "seq": seq,
        "prev_digest": prev_digest,
        "iteration_digest": iteration_digest,
        "discrimination": discrimination,
    })


def assert_ledger_valid(ledger):
    """校验归档哈希链完整性；任何篡改 / 断链 / 重排 ⇒ 抛 IterLoopError（fail-closed）。"""
    if not isinstance(ledger, list):
        raise IterLoopError("ledger 须为列表，实为 %s" % type(ledger).__name__)
    expected_prev = GENESIS_DIGEST
    seen_ids = set()
    for idx, entry in enumerate(ledger):
        where = "ledger[%d]" % idx
        if not isinstance(entry, dict):
            raise IterLoopError("%s 须为对象，实为 %s" % (where, type(entry).__name__))
        C.reject_unknown(entry, _ENTRY_KEYS, where)
        for req_k in _ENTRY_KEYS:
            if req_k not in entry:
                raise IterLoopError("%s 缺必填字段 %r" % (where, req_k))

        seq = entry["seq"]
        if isinstance(seq, bool) or not isinstance(seq, int) or seq != idx:
            raise IterLoopError("%s.seq 篡改或错序：期望 %d，实为 %r" % (where, idx, seq))

        prev_d = entry["prev_digest"]
        if prev_d != expected_prev:
            raise IterLoopError(
                "%s.prev_digest 断链：期望 %s，实为 %r" % (where, expected_prev, prev_d))

        n_iter = IT.normalize_iteration(entry["iteration"])
        it_id = n_iter["id"]
        if it_id in seen_ids:
            raise IterLoopError("%s 出现重复迭代 id %r" % (where, it_id))
        seen_ids.add(it_id)

        real_it_digest = IT.iteration_digest(n_iter)
        if entry["iteration_digest"] != real_it_digest:
            raise IterLoopError(
                "%s.iteration_digest 与 iteration 内容不匹配（内容已被篡改）："
                "期望 %s，实为 %r" % (where, real_it_digest, entry["iteration_digest"]))

        if not isinstance(entry["discrimination"], dict):
            raise IterLoopError("%s.discrimination 须为对象" % where)

        real_entry_digest = _compute_entry_digest(
            seq, prev_d, real_it_digest, entry["discrimination"])
        if entry["entry_digest"] != real_entry_digest:
            raise IterLoopError(
                "%s.entry_digest 校验失败（哈希链被篡改）：期望 %s，实为 %r"
                % (where, real_entry_digest, entry["entry_digest"]))

        expected_prev = real_entry_digest
    return True


def verify_ledger_chain(ledger):
    """查询面：校验哈希链完整性，返回 (ok: bool, reason: str)。"""
    try:
        assert_ledger_valid(ledger)
    except (IterLoopError, C.ContractError) as exc:
        return False, str(exc)
    return True, ""


def append_iteration_ledger(ledger, iter_doc, bad_scene, fixed_scene):
    """验证判别力并追加到防篡改哈希链，返回新台账列表（不修改入参）。"""
    base = [] if ledger is None else list(ledger)
    assert_ledger_valid(base)

    n_iter = IT.normalize_iteration(iter_doc)
    existing_ids = {e["iteration"]["id"] for e in base}
    if n_iter["id"] in existing_ids:
        raise IterLoopError("台账已存在同 id 迭代记录 %r，禁止重复归档" % n_iter["id"])

    disc = verify_iteration_discrimination(n_iter, bad_scene, fixed_scene)
    seq = len(base)
    prev_digest = GENESIS_DIGEST if seq == 0 else base[-1]["entry_digest"]
    it_digest = IT.iteration_digest(n_iter)
    entry_digest = _compute_entry_digest(seq, prev_digest, it_digest, disc)

    entry = {
        "seq": seq,
        "prev_digest": prev_digest,
        "iteration_digest": it_digest,
        "entry_digest": entry_digest,
        "iteration": n_iter,
        "discrimination": disc,
    }
    return base + [entry]


def replay_ledger_against_scene(
    ledger,
    sc,
    oracle_id="oracle_p7_replay",
    subject_name="p7_replay",
    subject_digest=None,
    generated_by="wm_bridge.factory.film._det.iter_loop",
    include_base_gates=False,
):
    """将负记忆台账中的判据在场景 `sc` 上重放，装配为通过校验的 `wmfilm.oracle/1`。"""
    entries = [] if ledger is None else ledger
    assert_ledger_valid(entries)
    S.reject_unknown_scene(sc)

    active_by_name = {}
    order = []
    negatives_count = 0
    for entry in entries:
        oc = entry["iteration"]["outcomes"]
        negatives_count += len(oc["new_negatives"])
        for crit in oc["new_criteria"]:
            nm = crit["name"]
            if nm not in active_by_name:
                order.append(nm)
            active_by_name[nm] = crit

    checks = []
    thresholds = []
    seen_th = set()
    for nm in order:
        crit = active_by_name[nm]
        checks.append(evaluate_custom_criterion(crit, sc))
        for th in criterion_thresholds(crit):
            if th["name"] not in seen_th:
                seen_th.add(th["name"])
                thresholds.append(th)

    if include_base_gates:
        base_checks, _ = GA.run_all(sc)
        custom_names = set(order)
        for chk in base_checks:
            if chk["name"] not in custom_names:
                checks.append(chk)
        for th in GA.all_thresholds():
            if th["name"] not in seen_th:
                seen_th.add(th["name"])
                thresholds.append(th)

    subj_dg = subject_digest or (entries[-1]["entry_digest"] if entries else GENESIS_DIGEST)
    doc = {
        "schema": OR.ORACLE_SCHEMA,
        "id": oracle_id,
        "subject_ref": {"name": subject_name, "digest": subj_dg},
        "thresholds": thresholds,
        "checks": checks,
        "generated_by": generated_by,
        "notes": "P7 负记忆重放：%d 条迭代记录，%d 条激活判据，%d 条不适用条件"
                 % (len(entries), len(order), negatives_count),
    }
    return OR.normalize_oracle(doc)
