# -*- coding: utf-8 -*-
"""wm_bridge.factory.film.score · 乐谱契约 wmmv.score/1（纯标准库 · 零 bpy · 零 numpy）

★ 这一份契约要回答什么
    MV 的「音」侧此前没有任何**可核对**的声明：拍点、段落、能量、调性都只存在于
    某一次 librosa 调用的返回值里。换一台机器、换一个版本，同一个音频会得到
    略有差异的读数 —— 而剪辑判据（E 族）要拿它判「切点是否落在 downbeat」。
    wmmv.score/1 把离线分析的**结论固化**成数据：分析只做一次、存下来、
    之后所有人读同一份 —— 这是命题 3「时间化单一真源」在音频侧的落地。

★ 为什么存指纹而不存音频
    音频是**输入**，不是本层产物；把音频塞进契约会让 digest 与版权、体积纠缠。
    本契约只存 source.digest（音频文件的 sha256）＋分析参数，从而保证
    「同一份音频 + 同一套参数 ⇒ 同一份乐谱」这件事**可被核对**。

★ 依赖（§3.5 已裁 D2）
    librosa 作 film-audio sub-extra，**不在本模块 import** ——
    本模块是**纯标准库**的契约与校验器。分析实现落在后续件里，
    且必须走函数内 import（不进 full、不污染轻核，见 §3.5.2）。
"""
from __future__ import annotations

from . import _common as C

#: 契约标识
SCORE_SCHEMA = "wmmv.score/1"

#: 十二平均律音名（闭集）
TONICS = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

#: 调式（闭集）
MODES = ("major", "minor")

#: 段落类型（闭集）。用闭集而非自由字符串：否则「chorus」与「Chorus」「副歌」
#: 会各写各的，段落判据按名取用时只命中一部分 —— 静默漏判。
SECTION_KINDS = ("intro", "verse", "pre_chorus", "chorus", "bridge", "break",
                 "instrumental", "solo", "outro")

#: 拍号（闭集）
METERS = ((4, 4), (3, 4), (6, 8), (2, 4), (5, 4), (7, 8))

#: 拍点归属容差（秒）—— downbeat 必须**是**某个 beat，浮点比较用
BEAT_EPS = 1e-6

__all__ = [
    "SCORE_SCHEMA", "TONICS", "MODES", "SECTION_KINDS", "METERS", "BEAT_EPS",
    "ScoreError", "normalize_score", "validate_score", "score_digest",
    "score_summary", "beats_in", "nearest_beat", "section_at", "energy_at",
]

_KEYS_SCORE = ("schema", "id", "source", "tempo", "beats", "downbeats",
               "sections", "energy", "key", "loudness", "notes")
_KEYS_SOURCE = ("path", "digest", "duration_s", "sample_rate", "channels")
_KEYS_TEMPO = ("bpm", "meter", "offset_s")
_KEYS_SECTION = ("kind", "start_s", "end_s", "energy")
_KEYS_ENERGY = ("hop_s", "metric", "values")
_KEYS_KEY = ("tonic", "mode", "confidence")
_KEYS_LOUDNESS = ("integrated_lufs", "true_peak_db")

#: 能量包络可用的度量（闭集）
ENERGY_METRICS = ("rms", "spectral_flux", "onset_strength", "loudness_lufs")


class ScoreError(C.ContractError):
    """乐谱契约不合法（fail-closed：不猜、不补、不静默降级）。"""


def _is_hex(s):
    return bool(s) and all(ch in "0123456789abcdefABCDEF" for ch in s)


def _digest_str(d, where):
    v = C.req_str(d, "digest", where)
    if _is_hex(v) and len(v) != 64:
        raise ScoreError(
            "%s.digest 形如十六进制但非 sha256（须 64 位），实为 %r" % (where, v))
    return v


def _time_list(v, where):
    """时间序列：须为有限、非负、**严格递增**的列表。"""
    if not isinstance(v, list):
        raise ScoreError("%s 须为列表，实为 %s" % (where, type(v).__name__))
    out, prev = [], None
    for i, t in enumerate(v):
        w = "%s[%d]" % (where, i)
        if not C.is_num(t):
            raise ScoreError("%s 须为有限数值，实为 %r" % (w, t))
        t = float(t)
        if t < 0.0:
            raise ScoreError("%s 须非负（秒），实为 %r" % (w, t))
        if prev is not None and not (t > prev):
            raise ScoreError("%s 须严格递增，实为 %r 不大于前一点 %r" % (w, t, prev))
        prev = t
        out.append(t)
    return out


# ─────────────────────────────────────────────────────────── 规范化（执行面）
def normalize_score(sc):
    """校验并补全默认值；返回**新的**规范化 score（不修改入参）。

    补全的默认值：
      source.channels  -> 2
      tempo.offset_s   -> 0.0
      downbeats        -> []
      sections         -> []
      energy           -> 省略（无包络则不声明，不编造空包络）
      key              -> 省略（分析不出调性时不编造）
      loudness         -> 省略

    生效条件：合法 ⇒ 返回新 dict；不合法 ⇒ raise ScoreError（消息含字段路径）。
    """
    if not isinstance(sc, dict):
        raise ScoreError("score 顶层须为对象，实为 %s" % type(sc).__name__)
    C.check_schema(sc, SCORE_SCHEMA, "score")
    C.reject_unknown(sc, _KEYS_SCORE, "score 顶层")

    out = {"schema": SCORE_SCHEMA}
    out["id"] = C.req_str(sc, "id", "score")

    # ---- source：只存指纹与参数，不存音频
    sr = C.req_map(sc, "source", "score")
    C.reject_unknown(sr, _KEYS_SOURCE, "score.source")
    dur = C.positive(C.req_num(sr, "duration_s", "score.source"),
                     "score.source.duration_s")
    out["source"] = {
        "digest": _digest_str(sr, "score.source"),
        "duration_s": dur,
        "sample_rate": C.positive(C.req_num(sr, "sample_rate", "score.source"),
                                  "score.source.sample_rate"),
        "channels": C.opt_int(sr, "channels", 2, "score.source"),
    }
    if sr.get("path") is not None:
        out["source"]["path"] = C.req_str(sr, "path", "score.source")
    if out["source"]["channels"] < 1:
        raise ScoreError("score.source.channels 须 >= 1，实为 %r"
                         % (out["source"]["channels"],))

    # ---- tempo
    tp = C.req_map(sc, "tempo", "score")
    C.reject_unknown(tp, _KEYS_TEMPO, "score.tempo")
    bpm = C.positive(C.req_num(tp, "bpm", "score.tempo"), "score.tempo.bpm")
    meter = C.req_list(tp, "meter", "score.tempo")
    if len(meter) != 2 or not all(C.is_num(x) and float(x).is_integer() for x in meter):
        raise ScoreError("score.tempo.meter 须为 [分子, 分母] 两个整数，实为 %r" % (meter,))
    meter_v = (int(meter[0]), int(meter[1]))
    if meter_v not in METERS:
        raise ScoreError("score.tempo.meter 须为闭集之一 %s，实为 %r"
                         % (" / ".join("%d/%d" % m for m in METERS), meter_v))
    # 规范化输出必须是**自己的合法输入**（幂等契约，2026-10-18 外部审计 R12-01）：
    # 这里落 list 而非 tuple —— req_list 只收 list，落 tuple 会让第二次 normalize
    # 以「meter 须为列表」报错，E1 在文档声明的输入形态下恒为 BLINDSPOT。
    out["tempo"] = {"bpm": bpm, "meter": list(meter_v),
                    "offset_s": C.opt_num(tp, "offset_s", 0.0, "score.tempo")}

    # ---- beats / downbeats
    beats = _time_list(C.req_list(sc, "beats", "score"), "score.beats")
    if not beats:
        raise ScoreError("score.beats 不得为空（没有拍点则剪辑判据无从对齐）")
    out["beats"] = beats

    dbs = _time_list(C.opt_list(sc, "downbeats", "score"), "score.downbeats")
    for i, t in enumerate(dbs):
        if nearest_beat(beats, t) is None:
            raise ScoreError(
                "score.downbeats[%d]=%r 不在 score.beats 中 —— "
                "downbeat 必须**是**某一个 beat（强拍是拍点的子集），容差 %g s"
                % (i, t, BEAT_EPS))
    out["downbeats"] = dbs

    # ---- sections：首尾覆盖全曲、不重叠
    secs = []
    for i, s in enumerate(C.opt_list(sc, "sections", "score")):
        w = "score.sections[%d]" % i
        if not isinstance(s, dict):
            raise ScoreError("%s 须为对象，实为 %s" % (w, type(s).__name__))
        C.reject_unknown(s, _KEYS_SECTION, w)
        kind = C.enum(C.req_str(s, "kind", w), SECTION_KINDS, w + ".kind")
        st = C.req_num(s, "start_s", w)
        en = C.req_num(s, "end_s", w)
        if st < 0.0 or en <= st:
            raise ScoreError("%s 的 [start_s, end_s)=[%r, %r) 须非负且非空" % (w, st, en))
        secs.append({"kind": kind, "start_s": st, "end_s": en,
                     "energy": C.frac01(C.opt_num(s, "energy", 0.0, w), w + ".energy")})
    for a, b in zip(secs, secs[1:]):
        if abs(a["end_s"] - b["start_s"]) > BEAT_EPS:
            raise ScoreError(
                "段落不连续：%s 结束于 %r，但下一段 %s 起于 %r —— 段落须首尾衔接"
                % (a["kind"], a["end_s"], b["kind"], b["start_s"]))
    if secs and abs(secs[0]["start_s"]) > BEAT_EPS:
        raise ScoreError("首段须起于 0.0，实为 %r" % (secs[0]["start_s"],))
    if secs and abs(secs[-1]["end_s"] - dur) > 1e-3:
        raise ScoreError(
            "末段须止于 source.duration_s=%r，实为 %r —— 段落须覆盖全曲" % (dur, secs[-1]["end_s"]))
    out["sections"] = secs

    # ---- energy：包络长度必须与时长自洽
    if sc.get("energy") is not None:
        eg = C.req_map(sc, "energy", "score")
        C.reject_unknown(eg, _KEYS_ENERGY, "score.energy")
        hop = C.positive(C.req_num(eg, "hop_s", "score.energy"), "score.energy.hop_s")
        vals = C.req_list(eg, "values", "score.energy")
        if not vals:
            raise ScoreError("score.energy.values 不得为空")
        clean = []
        for i, x in enumerate(vals):
            if not C.is_num(x):
                raise ScoreError("score.energy.values[%d] 须为有限数值，实为 %r" % (i, x))
            clean.append(float(x))
        want = int(dur / hop) + 1
        if len(clean) != want:
            raise ScoreError(
                "score.energy.values 长度(%d) 与 duration_s/hop_s(%d) 不自洽 —— "
                "包络长度必须由 hop_s 与时长唯一决定，否则读数无法复算" % (len(clean), want))
        out["energy"] = {"hop_s": hop,
                         "metric": C.opt_enum(eg, "metric", "rms", "score.energy",
                                              ENERGY_METRICS),
                         "values": clean}

    # ---- key / loudness（可选：分析不出就不写，不编造）
    if sc.get("key") is not None:
        kd = C.req_map(sc, "key", "score")
        C.reject_unknown(kd, _KEYS_KEY, "score.key")
        out["key"] = {"tonic": C.enum(C.req_str(kd, "tonic", "score.key"), TONICS,
                                      "score.key.tonic"),
                      "mode": C.enum(C.req_str(kd, "mode", "score.key"), MODES,
                                     "score.key.mode"),
                      "confidence": C.frac01(
                          C.opt_num(kd, "confidence", 0.0, "score.key"),
                          "score.key.confidence")}
    if sc.get("loudness") is not None:
        ld = C.req_map(sc, "loudness", "score")
        C.reject_unknown(ld, _KEYS_LOUDNESS, "score.loudness")
        out["loudness"] = {
            "integrated_lufs": C.req_num(ld, "integrated_lufs", "score.loudness"),
            "true_peak_db": C.req_num(ld, "true_peak_db", "score.loudness")}

    if sc.get("notes") is not None:
        out["notes"] = C.req_str(sc, "notes", "score")
    return out


# ─────────────────────────────────────────────────────────── 查询面与摘要
def validate_score(sc):
    """查询面：返回 (ok: bool, reason: str)。**不抛异常、不改入参。**"""
    return C.validate_with(normalize_score, sc)


def score_digest(sc):
    """乐谱契约的内容摘要（sha256 hex）。只依赖内容。"""
    return C.digest_of(normalize_score(sc))


def nearest_beat(beats, t, eps=BEAT_EPS):
    """在 beats 中找与 t 相等的拍点（容差 eps）；无 ⇒ None。"""
    for b in beats:
        if abs(float(b) - float(t)) <= eps:
            return b
    return None


def beats_in(sc, start_s, end_s):
    """取 [start_s, end_s) 区间内的拍点。"""
    n = normalize_score(sc)
    return [b for b in n["beats"] if start_s <= b < end_s]


def section_at(sc, t):
    """取时刻 t 所属的段落（无段落声明或越界 ⇒ None）。"""
    for s in normalize_score(sc)["sections"]:
        if s["start_s"] <= t < s["end_s"]:
            return s
    return None


def energy_at(sc, t):
    """取时刻 t 的能量读数（无包络或越界 ⇒ None）。**不插值** —— 最近帧取值。"""
    n = normalize_score(sc)
    eg = n.get("energy")
    if not eg:
        return None
    idx = int(round(float(t) / eg["hop_s"]))
    if idx < 0 or idx >= len(eg["values"]):
        return None
    return eg["values"][idx]


def score_summary(sc):
    """人读摘要（用于报告与守卫断言，不参与哈希）。"""
    n = normalize_score(sc)
    durs = [s["end_s"] - s["start_s"] for s in n["sections"]]
    return {"schema": n["schema"], "id": n["id"],
            "duration_s": n["source"]["duration_s"], "bpm": n["tempo"]["bpm"],
            "meter": "%d/%d" % n["tempo"]["meter"],
            "beats": len(n["beats"]), "downbeats": len(n["downbeats"]),
            "sections": [s["kind"] for s in n["sections"]],
            "section_durations": durs,
            "has_energy": "energy" in n, "key": n.get("key"),
            "digest": score_digest(n)}
