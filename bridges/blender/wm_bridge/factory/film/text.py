# -*- coding: utf-8 -*-
"""wm_bridge.factory.film.text · 文本层契约 wmmv.text/1（纯标准库 · 零 bpy）

★ 这一份契约要回答什么
    歌词、字幕、字卡是 MV 里**唯一必须被人读懂**的元素，但它们此前没有任何
    可核对的声明：什么时候出现、停留多久、每行几个字、CPS 多少、是否落在安全区。
    wmmv.text/1 把这几件事写成数据 —— 于是「字幕太快来不及读」不再是主观抱怨，
    而是一条可以算出来的读数。

★ 口径来源（**不新造**，逐条取证）
    安全区：EBU R95 —— title_safe = 0.80 x 画幅、action_safe = 0.90 x 画幅
            （1920x1080 下为 1536x864 / 1728x972 px）。见调研附录第一部分。
    可读性：Netflix Timed Text Style Guide —— CPS <= 20（成人）、每行 <= 42 字符（拉丁）、
            同屏 <= 2 行。见调研附录第一部分④。
    ★ 拉丁口径是**附录已录**的；CJK 口径附录未录，故本版**不给默认值** ——
      若做中文歌词，必须由使用者在 limits 里**显式声明**上限（不编造标准）。

★ 画幅（已裁 D4）
    16:9 横屏。竖屏 9:16 **不在本版范围**，安全区不得复用（§3.4 设计要点 8）。
"""
from __future__ import annotations

from . import TITLE_SAFE, ACTION_SAFE, ASPECT_16_9
from . import _common as C

#: 契约标识
TEXT_SCHEMA = "wmmv.text/1"

#: 文本条目类型（闭集）
CUE_KINDS = ("lyric", "subtitle", "title_card", "credit")

#: 书写系统（闭集）。CJK 无附录依据 ⇒ 必须显式给 limits
SCRIPTS = ("latin", "cjk")

#: 拉丁口径的默认上限（附录已录：Netflix Timed Text Style Guide）
LATIN_LIMITS = {"max_cps": 20.0, "max_chars_per_line": 42, "max_lines": 2}

#: 常用停留时长参考（秒）—— 只作提示，**不硬拒**（不同平台口径不同）
DWELL_HINT_S = (5.0 / 6.0, 7.0)

__all__ = ["TEXT_SCHEMA", "CUE_KINDS", "SCRIPTS", "LATIN_LIMITS", "DWELL_HINT_S",
           "TextError", "normalize_text", "validate_text", "text_digest",
           "text_summary", "cps_of", "safe_area_px", "cue_at"]

_KEYS_TEXT = ("schema", "id", "frame_base", "aspect_ratio", "safe_area",
              "limits", "cues", "notes")
_KEYS_FRAME_BASE = ("fps", "frame_start", "frame_end")
_KEYS_SAFE = ("title_safe", "action_safe")
_KEYS_LIMITS = ("max_cps", "max_chars_per_line", "max_lines")
_KEYS_CUE = ("kind", "start_frame", "end_frame", "text", "lines", "script")


class TextError(C.ContractError):
    """文本层契约不合法（fail-closed：不猜、不补、不静默降级）。"""


def _norm_frame_base(fb):
    C.reject_unknown(fb, _KEYS_FRAME_BASE, "text.frame_base")
    fs = C.req_int(fb, "frame_start", "text.frame_base")
    fe = C.req_int(fb, "frame_end", "text.frame_base")
    if fe <= fs:
        raise TextError("text.frame_base 的 frame_end(%d) 须晚于 frame_start(%d)"
                        % (fe, fs))
    return {"fps": C.positive(C.req_num(fb, "fps", "text.frame_base"),
                              "text.frame_base.fps"),
            "frame_start": fs, "frame_end": fe}


# ─────────────────────────────────────────────────────────── 规范化（执行面）
def normalize_text(tx):
    """校验并补全默认值；返回**新的**规范化 text（不修改入参）。

    补全的默认值：
      aspect_ratio  -> 16/9（已裁 D4）
      safe_area     -> {"title_safe": 0.80, "action_safe": 0.90}   （EBU R95）
      limits        -> LATIN_LIMITS（**拉丁侧默认**，附录已录；script="cjk"
                       时三项上限必须显式齐全 —— 空/None/缺项即拒，绝不把
                       拉丁口径当使用者的 CJK 声明套用）
      cue.script    -> "latin"
      cue.lines     -> 由 text 按换行切分（显式给出时须与 text 的实际换行
                       逐行一致 —— lines 只是可选的预分行，行数/行宽/CPS
                       三项硬门与 cps_of/summary 读数都以 text 正文为准）

    生效条件：合法 ⇒ 返回新 dict；不合法 ⇒ raise TextError（消息含字段路径）。
    """
    if not isinstance(tx, dict):
        raise TextError("text 顶层须为对象，实为 %s" % type(tx).__name__)
    C.check_schema(tx, TEXT_SCHEMA, "text")
    C.reject_unknown(tx, _KEYS_TEXT, "text 顶层")

    out = {"schema": TEXT_SCHEMA}
    out["id"] = C.req_str(tx, "id", "text")
    out["frame_base"] = _norm_frame_base(C.req_map(tx, "frame_base", "text"))

    ar = C.opt_num(tx, "aspect_ratio", ASPECT_16_9, "text")
    if not (ar > 0.0):
        raise TextError("text.aspect_ratio 须为正数，实为 %r" % (ar,))
    out["aspect_ratio"] = ar

    sa = C.opt_map(tx, "safe_area", "text")
    C.reject_unknown(sa, _KEYS_SAFE, "text.safe_area")
    ts = C.frac01(C.opt_num(sa, "title_safe", TITLE_SAFE, "text.safe_area"),
                  "text.safe_area.title_safe")
    ac = C.frac01(C.opt_num(sa, "action_safe", ACTION_SAFE, "text.safe_area"),
                  "text.safe_area.action_safe")
    if not (ts <= ac):
        raise TextError("title_safe(%r) 不得大于 action_safe(%r) —— 字幕区是动作区的子集"
                        % (ts, ac))
    out["safe_area"] = {"title_safe": ts, "action_safe": ac}

    lm = C.opt_map(tx, "limits", "text")
    C.reject_unknown(lm, _KEYS_LIMITS, "text.limits")
    lim = {"max_cps": C.positive(C.opt_num(lm, "max_cps", LATIN_LIMITS["max_cps"],
                                           "text.limits"), "text.limits.max_cps"),
           "max_chars_per_line": C.opt_int(lm, "max_chars_per_line",
                                           LATIN_LIMITS["max_chars_per_line"],
                                           "text.limits"),
           "max_lines": C.opt_int(lm, "max_lines", LATIN_LIMITS["max_lines"],
                                  "text.limits")}
    if lim["max_chars_per_line"] < 1:
        raise TextError("text.limits.max_chars_per_line 须 >= 1，实为 %r"
                        % (lim["max_chars_per_line"],))
    if lim["max_lines"] < 1:
        raise TextError("text.limits.max_lines 须 >= 1，实为 %r" % (lim["max_lines"],))
    out["limits"] = lim

    fb = out["frame_base"]
    cues = []
    for i, c in enumerate(C.req_list(tx, "cues", "text")):
        w = "text.cues[%d]" % i
        if not isinstance(c, dict):
            raise TextError("%s 须为对象，实为 %s" % (w, type(c).__name__))
        C.reject_unknown(c, _KEYS_CUE, w)
        kind = C.enum(C.req_str(c, "kind", w), CUE_KINDS, w + ".kind")
        st = C.req_int(c, "start_frame", w)
        en = C.req_int(c, "end_frame", w)
        if en <= st:
            raise TextError("%s 的 end_frame(%d) 须晚于 start_frame(%d)" % (w, en, st))
        if st < fb["frame_start"] or en > fb["frame_end"]:
            raise TextError("%s 的 [%d, %d] 越出 frame_base [%d, %d]"
                            % (w, st, en, fb["frame_start"], fb["frame_end"]))

        body = C.req_str(c, "text", w)
        # 正文的实际分行（按 \n 切分、去每行首尾空白）—— 行数 / 行宽 / CPS
        # 三项可读性硬门与 cps_of/summary 读数都钉在这份「实际上屏正文」上；
        # lines 只是可选的预分行，显式给出时必须与 text 的实际换行逐行一致。
        canon = [s.strip() for s in body.split("\n")]
        for j, ln in enumerate(canon):
            if not ln:
                raise TextError("%s.text 第 %d 行为空或纯空白（正文按 \\n 分行后"
                                "须为非空行），实为 %r" % (w, j, body.split("\n")[j]))
        lines = c.get("lines")
        if lines is None:
            clean = canon
        else:
            if not isinstance(lines, list) or not lines:
                raise TextError("%s.lines 须为非空列表，实为 %r" % (w, lines))
            clean = []
            for j, ln in enumerate(lines):
                if not isinstance(ln, str) or not ln.strip():
                    raise TextError("%s.lines[%d] 须为非空字符串，实为 %r" % (w, j, ln))
                clean.append(ln.strip())
            if clean != canon:
                raise TextError(
                    "%s.lines 与 text 不一致 —— lines 只是可选的预分行，"
                    "可读性硬门按 text 的实际换行计：lines 合计 %d 行 / %d 字符，"
                    "text 实际 %d 行 / %d 字符（fail-closed：不猜哪份是对的）"
                    % (w, len(clean), sum(len(x) for x in clean),
                       len(canon), sum(len(x) for x in canon)))
        if len(clean) > lim["max_lines"]:
            raise TextError(
                "%s 有 %d 行，超过 limits.max_lines=%d（附录已录：同屏 <= 2 行）"
                % (w, len(clean), lim["max_lines"]))
        for j, ln in enumerate(clean):
            if len(ln) > lim["max_chars_per_line"]:
                raise TextError(
                    "%s.lines[%d] 有 %d 字符，超过 limits.max_chars_per_line=%d"
                    "（附录已录：拉丁每行 <= 42 字符）"
                    % (w, j, len(ln), lim["max_chars_per_line"]))

        script = C.opt_enum(c, "script", "latin", w, SCRIPTS)
        if script == "cjk":
            # 「显式声明」判据落在内容上：三项上限缺任一项（含空对象 /
            # 显式 null / 只给部分）即拒 —— LATIN_LIMITS 是附录已录的拉丁
            # 默认，不是使用者的 CJK 声明，绝不让它冒名顶替（#825 ①）。
            missing = [k for k in _KEYS_LIMITS if lm.get(k) is None]
            if missing:
                raise TextError(
                    "%s.script=\"cjk\" 但 text.limits 未显式声明 %s —— "
                    "CJK 的 CPS/行宽口径**附录未录**，本版不给默认值（不编造标准）；"
                    "请按你的平台规范显式给出 max_cps / max_chars_per_line / max_lines"
                    % (w, " / ".join(missing)))

        cues.append({"kind": kind, "start_frame": st, "end_frame": en,
                     "text": body, "lines": clean, "script": script})

    for a, b in zip(cues, cues[1:]):
        if b["start_frame"] <= a["end_frame"]:
            raise TextError(
                "文本条目重叠：%r 到第 %d 帧，但下一条起于第 %d 帧 —— "
                "同屏两条会互相遮挡" % (a["text"][:12], a["end_frame"], b["start_frame"]))
    out["cues"] = cues

    # ---- CPS 硬校验（附录已录：成人 <= 20）
    fps = fb["fps"]
    for i, c in enumerate(cues):
        v = cps_of(c, fps)
        if v > lim["max_cps"] + 1e-9:
            raise TextError(
                "text.cues[%d] 的 CPS=%.2f 超过 limits.max_cps=%.2f"
                "（附录已录：成人内容 CPS <= 20）—— 观众读不完"
                % (i, v, lim["max_cps"]))

    if tx.get("notes") is not None:
        out["notes"] = C.req_str(tx, "notes", "text")
    return out


# ─────────────────────────────────────────────────────────── 查询面与摘要
def cps_of(cue, fps):
    """单条文本的 CPS（每秒字符数）。字符数按**所有行合计**（含空格）。

    行按 text 的实际换行计（lines 只是预分行，规范化后须与 text 一致）——
    读数钉在实际上屏正文上，不受 lines 单方面改写的影响（#825 ②）。
    """
    dur_s = (cue["end_frame"] - cue["start_frame"] + 1) / float(fps)
    if dur_s <= 0.0:
        raise TextError("cps_of 的时长须为正，实为 %r" % (dur_s,))
    n = sum(len(ln) for ln in cue["text"].split("\n"))
    return n / dur_s


def validate_text(tx):
    """查询面：返回 (ok: bool, reason: str)。**不抛异常、不改入参。**"""
    return C.validate_with(normalize_text, tx)


def text_digest(tx):
    """文本层契约的内容摘要（sha256 hex）。只依赖内容。"""
    return C.digest_of(normalize_text(tx))


def safe_area_px(tx, width, height):
    """安全区在给定分辨率下的像素矩形（EBU R95 口径；16:9 下 1920x1080 -> 1536x864）。"""
    n = normalize_text(tx)
    sa = n["safe_area"]
    return {"title_safe": (int(round(width * sa["title_safe"])),
                           int(round(height * sa["title_safe"]))),
            "action_safe": (int(round(width * sa["action_safe"])),
                            int(round(height * sa["action_safe"])))}


def cue_at(tx, frame):
    """取该帧上屏的文本条目（无 ⇒ None）。"""
    for c in normalize_text(tx)["cues"]:
        if c["start_frame"] <= frame <= c["end_frame"]:
            return c
    return None


def text_summary(tx):
    """人读摘要（用于报告与守卫断言，不参与哈希）。"""
    n = normalize_text(tx)
    return {"schema": n["schema"], "id": n["id"], "cues": len(n["cues"]),
            "kinds": sorted({c["kind"] for c in n["cues"]}),
            "title_safe": n["safe_area"]["title_safe"],
            "action_safe": n["safe_area"]["action_safe"],
            "limits": n["limits"],
            "max_cps": round(max([cps_of(c, n["frame_base"]["fps"]) for c in n["cues"]]
                                 or [0.0]), 3),
            "digest": text_digest(n)}
