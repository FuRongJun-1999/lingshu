# -*- coding: utf-8 -*-
"""wm_bridge.factory.film.iteration · P7 负记忆迭代闭环契约 wmfilm.iteration/1（纯标准库 · 零 bpy）

★ 这一份契约要回答什么（理论架构 §一 命题 6 + §七 P7）
    审美改进不靠「更努力」，靠**把每次「哪里难看」固化成负记忆 + 新判据**：
        现象（哪里难看）→ 根因（为什么难看）→ 新判据 / 新不适用条件 → 归档 → 下次自动拦住

    本契约把 `factory/_lead/hero_render.py` 末段的自由文本 `ITERATIONS` 正式化为
    可校验、可哈希、可自动回归拦截的结构化文档 `wmfilm.iteration/1`，并把两道硬铁律
    写进校验器：
      1. **P7 硬验收判据**：每次迭代必须产出 ≥1 条新判据 `new_criteria` 或新不适用条件
         `new_negatives`；两者皆空 ⇒ fail-closed 拒收（杜绝「空转归档」）。
      2. **§4.1 三级门铁律**：`G1` 硬门只收 `standard` / `consensus` 性质，试图把
         `style`（或 `empirical`）塞进 `G1` 硬门拦创作 ⇒ fail-closed 拒收。

★ B4 硬边界
    纯标准库实现，不 import bpy / numpy / PIL / mathutils，可在无 Blender 环境完整自测。
"""
from __future__ import annotations

from . import _common as C
from . import oracle as OR

#: 契约标识
ITERATION_SCHEMA = "wmfilm.iteration/1"

#: 根因落点层（闭集：§3.1 架构七层对应模块）
LAYERS = ("geo", "scene", "perform", "shot", "score", "bind", "edit", "text", "oracle")

#: §4.1 三级门
GATES = ("G1", "G2", "G3")

#: 参数性质（§4.1：[标准] / [共识] / [经验] / [风格]）
NATURES = ("standard", "consensus", "empirical", "style")

#: §4.1 门-性质准入矩阵（唯一权威）
GATE_NATURES = {
    "G1": ("standard", "consensus"),
    "G2": ("empirical",),
    "G3": ("style",),
}

#: 声明式数值比较 / 区间算子（闭集）
OPS = ("ge", "gt", "le", "lt", "eq", "between", "outside")
_SCALAR_OPS = ("ge", "gt", "le", "lt", "eq")
_RANGE_OPS = ("between", "outside")

__all__ = [
    "ITERATION_SCHEMA", "LAYERS", "GATES", "NATURES", "GATE_NATURES", "OPS",
    "IterationError",
    "normalize_criterion", "normalize_iteration", "validate_iteration",
    "iteration_digest", "iteration_summary",
]

_KEYS_ITERATION = ("schema", "id", "phenomenon", "root_cause", "outcomes",
                   "verification", "notes")
_KEYS_PHENOMENON = ("shot_id", "frame_range", "observed_readings", "description")
_KEYS_ROOT_CAUSE = ("layer", "mechanism", "component")
_KEYS_OUTCOMES = ("new_criteria", "new_negatives")
_KEYS_CRITERION = ("name", "family", "gate", "nature", "metric", "op", "threshold",
                   "range", "source", "threshold_name", "description", "shot_id")
_KEYS_NEGATIVE = ("target", "condition", "reason", "fallback")
_KEYS_VERIFICATION = ("before_oracle_ref", "after_oracle_ref",
                      "before_verdict", "after_verdict", "notes")
_KEYS_REF = ("name", "digest")


class IterationError(C.ContractError):
    """迭代闭环契约不合法（fail-closed：不猜、不补、不静默降级）。"""


def _is_sha256_hex(v):
    return (isinstance(v, str) and len(v) == 64
            and all(ch in "0123456789abcdefABCDEF" for ch in v))


def _req_sha256_ref(parent, key, where):
    ref = C.req_map(parent, key, where)
    sub_where = "%s.%s" % (where, key)
    C.reject_unknown(ref, _KEYS_REF, sub_where)
    name = C.req_str(ref, "name", sub_where)
    dg = C.req_str(ref, "digest", sub_where)
    if not _is_sha256_hex(dg):
        raise IterationError(
            "%s.digest 须为 64 位 sha256 十六进制摘要，实为 %r" % (sub_where, dg))
    return {"name": name, "digest": dg.lower()}


def _normalize_frame_range(ph, where):
    fr = C.req_list(ph, "frame_range", where)
    if len(fr) != 2:
        raise IterationError("%s.frame_range 须为 [start, end] 两元素，实为 %r" % (where, fr))
    vals = []
    for i, x in enumerate(fr):
        if not C.is_num(x) or not float(x).is_integer():
            raise IterationError(
                "%s.frame_range[%d] 须为整数（禁 bool/NaN/Inf/小数），实为 %r"
                % (where, i, x))
        vals.append(int(x))
    if vals[0] < 0 or vals[1] < vals[0]:
        raise IterationError(
            "%s.frame_range 须满足 0 <= start <= end，实为 %r" % (where, vals))
    return vals


def _normalize_observed_readings(ph, where):
    raw = C.req_map(ph, "observed_readings", where)
    if not raw:
        raise IterationError("%s.observed_readings 不得为空（须记录可观测指标读数）" % where)
    out = {}
    for k in sorted(raw):
        if not isinstance(k, str) or not k.strip():
            raise IterationError("%s.observed_readings 键须为非空字符串，实为 %r" % (where, k))
        v = raw[k]
        if not C.is_num(v):
            raise IterationError(
                "%s.observed_readings[%r] 须为有限数值（禁 bool/NaN/Inf），实为 %r"
                % (where, k, v))
        out[k.strip()] = float(v)
    return out


def normalize_criterion(c, where="criterion"):
    """校验并规范化单条新判据（对齐 §4.1 三级门与 `oracle.family_of`）。"""
    try:
        return _normalize_criterion_impl(c, where=where)
    except IterationError:
        raise
    except C.ContractError as exc:
        raise IterationError(str(exc)) from exc


def _normalize_criterion_impl(c, where="criterion"):
    if not isinstance(c, dict):
        raise IterationError("%s 须为对象，实为 %s" % (where, type(c).__name__))
    C.reject_unknown(c, _KEYS_CRITERION, where)

    name = C.req_str(c, "name", where)
    family = OR.family_of(name)
    if c.get("family") is not None and c.get("family") != family:
        raise IterationError(
            "%s.family=%r 与判据名 %r 所属族 %r 不一致"
            % (where, c.get("family"), name, family))
    gate = C.enum(C.req_str(c, "gate", where), GATES, where + ".gate")
    nature = C.enum(C.req_str(c, "nature", where), NATURES, where + ".nature")

    allowed_natures = GATE_NATURES[gate]
    if nature not in allowed_natures:
        raise IterationError(
            "%s 违反 §4.1 三级门：%s 门只允许 nature ∈ %r，实为 %r"
            "（严禁把 style/empirical 塞进 G1 硬门拦创作）"
            % (where, gate, allowed_natures, nature))

    source = C.req_str(c, "source", where)
    metric = C.opt_str(c, "metric", None, where)
    op = c.get("op")
    if gate in ("G1", "G2"):
        if metric is None:
            raise IterationError("%s（%s 门）缺必填字段 'metric'" % (where, gate))
        if op is None:
            raise IterationError("%s（%s 门）缺必填字段 'op'" % (where, gate))
    if metric is not None and any(ch.isspace() for ch in metric):
        raise IterationError("%s.metric 不得含空白字符，实为 %r" % (where, metric))
    if op is not None:
        if metric is None:
            raise IterationError("%s 声明了 op=%r 但缺 metric" % (where, op))
        op = C.enum(C.req_str(c, "op", where), OPS, where + ".op")

    has_thr = c.get("threshold") is not None
    has_rng = c.get("range") is not None

    out = {
        "name": name,
        "family": family,
        "gate": gate,
        "nature": nature,
        "source": source,
    }
    if metric is not None:
        out["metric"] = metric
    if op is not None:
        out["op"] = op
        if op in _SCALAR_OPS:
            if has_rng:
                raise IterationError(
                    "%s.op=%r 为单阈值算子，不得同时提供 range" % (where, op))
            out["threshold"] = C.req_num(c, "threshold", where)
        else:
            if has_thr:
                raise IterationError(
                    "%s.op=%r 为区间算子，不得同时提供 threshold" % (where, op))
            if not has_rng:
                raise IterationError("%s.op=%r 缺必填区间字段 'range'" % (where, op))
            lo, hi = C.vec2(c["range"], where + ".range")
            if lo > hi:
                raise IterationError(
                    "%s.range 须满足 lo <= hi，实为 [%r, %r]" % (where, lo, hi))
            out["range"] = [lo, hi]
    elif has_thr or has_rng:
        raise IterationError("%s 未声明 op，不得携带 threshold 或 range" % where)

    tname = C.opt_str(c, "threshold_name", None, where)
    if tname is not None:
        out["threshold_name"] = tname
    elif metric is not None:
        out["threshold_name"] = "%s_%s" % (name, metric)

    desc = C.opt_str(c, "description", None, where)
    if desc is not None:
        out["description"] = desc
    sid = C.opt_str(c, "shot_id", None, where)
    if sid is not None:
        out["shot_id"] = sid
    return out


def _normalize_negative(neg, where):
    if not isinstance(neg, dict):
        raise IterationError("%s 须为对象，实为 %s" % (where, type(neg).__name__))
    C.reject_unknown(neg, _KEYS_NEGATIVE, where)
    out = {
        "target": C.req_str(neg, "target", where),
        "condition": C.req_str(neg, "condition", where),
        "reason": C.req_str(neg, "reason", where),
    }
    fb = C.opt_str(neg, "fallback", None, where)
    if fb is not None:
        out["fallback"] = fb
    return out


# ─────────────────────────────────────────────────────────── 规范化（执行面）
def normalize_iteration(it):
    """校验并规范化 `wmfilm.iteration/1`；返回新的 dict（不修改入参）。"""
    try:
        return _normalize_iteration_impl(it)
    except IterationError:
        raise
    except C.ContractError as exc:
        raise IterationError(str(exc)) from exc


def _normalize_iteration_impl(it):
    if not isinstance(it, dict):
        raise IterationError("iteration 顶层须为对象，实为 %s" % type(it).__name__)
    C.check_schema(it, ITERATION_SCHEMA, "iteration")
    C.reject_unknown(it, _KEYS_ITERATION, "iteration 顶层")

    out = {
        "schema": ITERATION_SCHEMA,
        "id": C.req_str(it, "id", "iteration"),
    }

    # 1. phenomenon（现象）
    ph = C.req_map(it, "phenomenon", "iteration")
    C.reject_unknown(ph, _KEYS_PHENOMENON, "iteration.phenomenon")
    out["phenomenon"] = {
        "shot_id": C.req_str(ph, "shot_id", "iteration.phenomenon"),
        "frame_range": _normalize_frame_range(ph, "iteration.phenomenon"),
        "observed_readings": _normalize_observed_readings(ph, "iteration.phenomenon"),
        "description": C.req_str(ph, "description", "iteration.phenomenon"),
    }

    # 2. root_cause（根因）
    rc = C.req_map(it, "root_cause", "iteration")
    C.reject_unknown(rc, _KEYS_ROOT_CAUSE, "iteration.root_cause")
    rc_out = {
        "layer": C.enum(C.req_str(rc, "layer", "iteration.root_cause"),
                        LAYERS, "iteration.root_cause.layer"),
        "mechanism": C.req_str(rc, "mechanism", "iteration.root_cause"),
    }
    comp = C.opt_str(rc, "component", None, "iteration.root_cause")
    if comp is not None:
        rc_out["component"] = comp
    out["root_cause"] = rc_out

    # 3. outcomes（P7 硬判据：new_criteria / new_negatives 至少一项非空）
    oc = C.req_map(it, "outcomes", "iteration")
    C.reject_unknown(oc, _KEYS_OUTCOMES, "iteration.outcomes")
    raw_criteria = C.opt_list(oc, "new_criteria", "iteration.outcomes")
    raw_negatives = C.opt_list(oc, "new_negatives", "iteration.outcomes")
    if not raw_criteria and not raw_negatives:
        raise IterationError(
            "P7 硬验收判据违反：iteration.outcomes.new_criteria 与 new_negatives "
            "不得同时为空 —— 每次迭代必须产出 >= 1 条新判据或新不适用条件")

    criteria_out, seen_names = [], set()
    for i, c in enumerate(raw_criteria):
        nc = normalize_criterion(c, where="iteration.outcomes.new_criteria[%d]" % i)
        if nc["name"] in seen_names:
            raise IterationError(
                "iteration.outcomes.new_criteria 重复声明判据 %r" % nc["name"])
        seen_names.add(nc["name"])
        criteria_out.append(nc)

    negatives_out = [
        _normalize_negative(neg, "iteration.outcomes.new_negatives[%d]" % i)
        for i, neg in enumerate(raw_negatives)
    ]
    out["outcomes"] = {
        "new_criteria": criteria_out,
        "new_negatives": negatives_out,
    }

    # 4. verification（回归对拍：绑定 before/after oracle 64 位 sha256）
    vr = C.req_map(it, "verification", "iteration")
    C.reject_unknown(vr, _KEYS_VERIFICATION, "iteration.verification")
    vr_out = {
        "before_oracle_ref": _req_sha256_ref(
            vr, "before_oracle_ref", "iteration.verification"),
        "after_oracle_ref": _req_sha256_ref(
            vr, "after_oracle_ref", "iteration.verification"),
    }
    bv = C.opt_str(vr, "before_verdict", None, "iteration.verification")
    if bv is not None:
        vr_out["before_verdict"] = C.enum(
            bv, OR.VERDICTS, "iteration.verification.before_verdict")
    av = C.opt_str(vr, "after_verdict", None, "iteration.verification")
    if av is not None:
        vr_out["after_verdict"] = C.enum(
            av, OR.VERDICTS, "iteration.verification.after_verdict")
    vnotes = C.opt_str(vr, "notes", None, "iteration.verification")
    if vnotes is not None:
        vr_out["notes"] = vnotes
    out["verification"] = vr_out

    notes = C.opt_str(it, "notes", None, "iteration")
    if notes is not None:
        out["notes"] = notes
    return out


# ─────────────────────────────────────────────────────────── 查询面与摘要
def validate_iteration(it):
    """查询面：返回 (ok: bool, reason: str)。不抛异常、不改入参。"""
    return C.validate_with(normalize_iteration, it)


def iteration_digest(it):
    """迭代记录的内容摘要（sha256 hex）。只依赖内容。"""
    return C.digest_of(normalize_iteration(it))


def iteration_summary(it):
    """人读摘要（用于报告与守卫断言，不参与哈希）。"""
    n = normalize_iteration(it)
    return {
        "schema": n["schema"],
        "id": n["id"],
        "shot_id": n["phenomenon"]["shot_id"],
        "layer": n["root_cause"]["layer"],
        "criteria_count": len(n["outcomes"]["new_criteria"]),
        "negatives_count": len(n["outcomes"]["new_negatives"]),
        "criteria_names": [c["name"] for c in n["outcomes"]["new_criteria"]],
        "before_digest": n["verification"]["before_oracle_ref"]["digest"],
        "after_digest": n["verification"]["after_oracle_ref"]["digest"],
        "digest": iteration_digest(n),
    }
