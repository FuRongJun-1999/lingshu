# -*- coding: utf-8 -*-
"""确定性音乐合成 + `wmmv.score/1` 乐谱（**纯标准库**：wave / struct / math / array）。

依据 `mv/INTERFACES.md` §3.2（冻结签名）与 §4（契约要点）；
架构依据 `advance/film-docs/理论架构_v0.2.md` §4.2 E1（卡点：切点必须落在 downbeat，≤1 帧）
与 §七 P3（乐谱 `wmmv.score/1`：分析后固化为文档，执行期零依赖）。

设计要点
--------
· **合成**：44.1 kHz / 16-bit / 立体声。低音线（bass）+ 和弦垫（pad）
  + 鼓组（kick / snare / hihat，4/4 网格落拍）+ 副歌主音（lead）。
  段落编排随 `kind` 变化：intro 稀疏（pad + 稀疏 kick）→ verse 加 bass/hihat
  → pre_chorus 加密（kick/snare + 末尾军鼓滚奏）→ chorus 全开
  → bridge 收（只剩 pad + lead）→ 末段 chorus 最满（lead 八度加倍 + crash）。
· **确定性**：全程不用 `time` / `random`；噪声用内置 32 位 LCG（固定种子）。
  同一 brief 两次渲染 ⇒ WAV **逐字节相同**（sha256 相同）。
  且 `synthesize(make(...)["score"], p)` 能**逐字节复现** `make` 写出的 WAV。
· **能量同源**：`energy.values[i] = _energy_map(rms_i)`（`rms/ENERGY_REF`，顶端 tanh 软膝）；
  `sections[i].energy` = 该段所覆盖 hop 的**均值** ⇒ 段落能量与实际 RMS 同源、可复算，
  副歌必然高于主歌（编排保证）。
· **拍点闭式**：`beats[i] = offset_s + i*60/bpm`；`downbeats` 取每小节第 1 拍
  （直接复用 beats 的同一浮点值 ⇒ 必为子集）。
· **fail-closed**：缺字段 / WAV 与 brief 段落跨度不符 / 非 16-bit PCM ⇒ 抛异常，不静默降级。
"""
from __future__ import annotations

import array
import hashlib
import math
import os
import struct
import sys
import wave

from . import contracts as C

__all__ = ["make", "synthesize", "analyze", "AudioError",
           "SAMPLE_RATE", "CHANNELS", "SAMPWIDTH", "DEFAULT_HOP_S"]

# ---------------------------------------------------------------- 全片常数

SAMPLE_RATE = 44100
CHANNELS = 2
SAMPWIDTH = 2                      # 16-bit
DEFAULT_HOP_S = 0.05               # 能量包络步长

#: 能量包络标定：`values[i] = _energy_map(rms_i)`（见下）。
#: 取 0.20 使本简报的 chorus ≈ 0.75、verse ≈ 0.38、pre_chorus ≈ 0.57、
#: intro ≈ 0.14、bridge ≈ 0.13 —— 与 `brief.json` 的创作目标
#: （0.22 / 0.42 / 0.58 / 0.86 / 0.50 / 0.90）同量级且同序。
ENERGY_REF = 0.20
#: 线性段上限：v <= KNEE 时严格等于 rms/REF；其上用 tanh 软膝（单调、有界、不削平）
ENERGY_KNEE = 0.85

#: 主输出峰值（线性，满幅 = 1.0）
PEAK_TARGET = 0.90

#: 16 分音符网格（每小节 16 步）
STEPS_PER_BAR = 16

#: 音名 → 音高类
_PC = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5,
       "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}

#: 和弦进行（相对主音的半音偏移 + 三和弦性质）
_PROG = {
    "minor": ((0, "min"), (-4, "maj"), (3, "maj"), (-2, "maj")),      # i VI III VII
    "major": ((0, "maj"), (7, "maj"), (9, "min"), (5, "maj")),        # I V vi IV
}

#: 三和弦音程
_THIRD = {"min": 3, "maj": 4}

# ---------------------------------------------------------------- 段落编排表

#: 各段落乐器增益（0 = 不发声）。索引即 `contracts.SECTION_KINDS` 的语义。
_KIND_GAIN = {
    "intro":      {"pad": 0.85, "bass": 0.00, "kick": 0.45, "snare": 0.00, "hat": 0.00, "lead": 0.00},
    "verse":      {"pad": 0.62, "bass": 0.62, "kick": 0.50, "snare": 0.00, "hat": 0.30, "lead": 0.00},
    "pre_chorus": {"pad": 0.80, "bass": 0.88, "kick": 0.70, "snare": 0.65, "hat": 0.55, "lead": 0.35},
    "chorus":     {"pad": 1.45, "bass": 1.00, "kick": 1.00, "snare": 0.95, "hat": 0.90, "lead": 1.15},
    "bridge":     {"pad": 0.95, "bass": 0.00, "kick": 0.00, "snare": 0.00, "hat": 0.00, "lead": 0.90},
    "break":      {"pad": 0.70, "bass": 0.40, "kick": 0.50, "snare": 0.00, "hat": 0.30, "lead": 0.30},
    "drop":       {"pad": 1.45, "bass": 1.00, "kick": 1.00, "snare": 0.95, "hat": 0.90, "lead": 1.15},
    "outro":      {"pad": 0.85, "bass": 0.00, "kick": 0.40, "snare": 0.00, "hat": 0.00, "lead": 0.35},
}

#: 每小节的 16 分网格落点（步号 0..15）
_KICK_STEPS = {
    "intro": (0,), "verse": (0, 8), "pre_chorus": (0, 8, 14), "chorus": (0, 4, 8, 12),
    "bridge": (), "break": (0, 8), "drop": (0, 4, 8, 12), "outro": (0,),
}
_SNARE_STEPS = {
    "intro": (), "verse": (), "pre_chorus": (4, 12), "chorus": (4, 12),
    "bridge": (), "break": (), "drop": (4, 12), "outro": (),
}
_HAT_STEPS = {
    "intro": (), "verse": (0, 2, 4, 6, 8, 10, 12, 14), "pre_chorus": (0, 2, 4, 6, 8, 10, 12, 14),
    "chorus": tuple(range(STEPS_PER_BAR)), "bridge": (), "break": (2, 6, 10, 14),
    "drop": tuple(range(STEPS_PER_BAR)), "outro": (),
}
#: bass 走 8 分音符（步号偶数）
_BASS_STEPS = (0, 2, 4, 6, 8, 10, 12, 14)
#: lead 落点：整拍（chorus / pre_chorus / drop）或半小节（bridge / outro / break）
_LEAD_STEPS_BEAT = (0, 4, 8, 12)
_LEAD_STEPS_HALF = (0, 8)

# 乐器混音电平（峰值归一化前；相对比例决定段落能量差）
_GAIN = {"pad": 0.105, "bass": 0.300, "kick": 0.560, "snare": 0.360,
         "hat": 0.085, "lead": 0.150, "crash": 0.045}


class AudioError(Exception):
    """音频层 fail-closed 异常（缺料 / 不一致 / 非法 WAV）。"""


# ---------------------------------------------------------------- 基础工具

class _LCG:
    """内置 32 位线性同余噪声源（确定性；不依赖 `random`）。"""

    __slots__ = ("_s",)

    def __init__(self, seed: int) -> None:
        self._s = int(seed) & 0xFFFFFFFF

    def next(self) -> float:
        self._s = (1664525 * self._s + 1013904223) & 0xFFFFFFFF
        return self._s / 2147483648.0 - 1.0        # [-1, 1)


def _midi_hz(m: float) -> float:
    return 440.0 * (2.0 ** ((m - 69.0) / 12.0))


def _tonic_midi(tonic: str) -> int:
    """主音落在大字组 A2(45) 附近（A → 45）。"""
    if tonic not in _PC:
        raise AudioError("未知主音 %r（可选 %s）" % (tonic, "/".join(sorted(_PC))))
    return 45 + ((_PC[tonic] - _PC["A"]) % 12)


def _chord_at(bar_index: int, key: dict) -> tuple:
    """返回 (root_midi, quality)；进行按小节循环。"""
    mode = key["mode"]
    if mode not in _PROG:
        raise AudioError("未知调式 %r" % (mode,))
    prog = _PROG[mode]
    off, quality = prog[bar_index % len(prog)]
    return _tonic_midi(key["tonic"]) + off, quality


def _chord_tones(root: int, quality: str) -> tuple:
    third = _THIRD[quality]
    return (root, root + third, root + 7)


# ---------------------------------------------------------------- 振荡器 / 鼓件

def _tone(freq: float, n: int, sr: int, gain: float,
          atk: float, dec: float, sus: float, rel: float,
          harms=((1.0, 1.0),)) -> list:
    """ADSR 简化包络 + 正弦谐波叠加。返回 gain 已折算的浮点缓冲。"""
    out = [0.0] * n
    if n <= 0:
        return out
    w = 2.0 * math.pi * freq / sr
    rel_n = max(1, int(rel * sr))
    rel_start = n - rel_n
    inv_atk = 1.0 / atk if atk > 0 else 0.0
    inv_dec = 1.0 / dec if dec > 0 else 0.0
    for i in range(n):
        t = i / sr
        if t < atk:
            e = t * inv_atk
        elif t < atk + dec:
            e = 1.0 + (sus - 1.0) * ((t - atk) * inv_dec)
        else:
            e = sus
        if i >= rel_start:
            e *= (n - i) / rel_n
        if e <= 0.0:
            continue
        p = w * i
        s = 0.0
        for mult, amp in harms:
            s += amp * math.sin(mult * p)
        out[i] = gain * e * s
    return out


def _pad(freqs, n: int, sr: int, gain: float, detune: float, atk=0.35, rel=0.30) -> list:
    """和弦垫：每个和弦音两路轻微失谐（宽声场），缓起缓落。"""
    out = [0.0] * n
    if n <= 0:
        return out
    ws = [2.0 * math.pi * f * (1.0 + detune) / sr for f in freqs]
    rel_n = max(1, int(rel * sr))
    rel_start = n - rel_n
    inv_atk = 1.0 / atk
    for i in range(n):
        if i < atk * sr:
            e = i / sr * inv_atk
        else:
            e = 1.0
        if i >= rel_start:
            e *= (n - i) / rel_n
        if e <= 0.0:
            continue
        s = 0.0
        for w in ws:
            s += math.sin(w * i)
        out[i] = gain * e * s / len(ws)
    return out


def _kick(n: int, sr: int, gain: float) -> list:
    """底鼓：频率包络 140→45 Hz + 极短击打噪声。"""
    out = [0.0] * n
    ph = 0.0
    rng = _LCG(0x5EED01)
    for i in range(n):
        t = i / sr
        f = 45.0 + 95.0 * math.exp(-t / 0.030)
        ph += 2.0 * math.pi * f / sr
        e = math.exp(-t / 0.170)
        click = rng.next() * math.exp(-t / 0.004)
        out[i] = gain * (e * math.sin(ph) + 0.25 * click)
    return out


def _snare(n: int, sr: int, gain: float, seed: int) -> list:
    """军鼓：190 Hz 鼓皮 + 一阶高通噪声。"""
    out = [0.0] * n
    ph = 0.0
    rng = _LCG(seed)
    prev = 0.0
    w = 2.0 * math.pi * 190.0 / sr
    for i in range(n):
        t = i / sr
        ph += w
        x = rng.next()
        hp = x - prev
        prev = x
        out[i] = gain * (0.45 * math.sin(ph) * math.exp(-t / 0.100)
                         + 1.30 * hp * math.exp(-t / 0.075))
    return out


def _hat(n: int, sr: int, gain: float, seed: int, decay: float = 0.030) -> list:
    """踩镲 / 吊镲：二阶差分噪声（亮）。`decay` 大 ⇒ 开镲 / crash。"""
    out = [0.0] * n
    rng = _LCG(seed)
    p1 = 0.0
    p2 = 0.0
    for i in range(n):
        t = i / sr
        x = rng.next()
        y = x - p1
        p1 = x
        z = y - p2
        p2 = y
        out[i] = gain * z * math.exp(-t / decay)
    return out


# ---------------------------------------------------------------- 混音

def _add(dst: list, pos: int, src: list, limit: int = None) -> None:
    """把 src 叠进 dst[pos:]。`limit` 截断（段末），截断处做 5 ms 淡出防爆音。"""
    n = len(src)
    if limit is not None and limit < n:
        n = limit
    if n <= 0:
        return
    if pos + n > len(dst):
        n = len(dst) - pos
        if n <= 0:
            return
    if n < len(src):
        f = min(n, 220)
        for j in range(f):
            src[n - f + j] *= (f - j) / f
    seg = dst[pos:pos + n]
    dst[pos:pos + n] = [a + b for a, b in zip(seg, src[:n])]


def _pan_gains(pan: float) -> tuple:
    """线性声像：pan ∈ [-1,1] ⇒ (左, 右)，中心 (1,1)。"""
    return (1.0 - max(0.0, pan), 1.0 + min(0.0, pan))


def _place(master: list, pos: int, src: list, pan: float, limit: int = None) -> None:
    gl, gr = _pan_gains(pan)
    if gl:
        _add(master[0], pos, src if gl == 1.0 else [v * gl for v in src], limit)
    if gr:
        _add(master[1], pos, src if gr == 1.0 else [v * gr for v in src], limit)


# ---------------------------------------------------------------- 计划（plan）

def _norm_sections(raw, duration_s: float) -> list:
    """段落结构校验 + 末段对齐到真实时长（无缝无叠，从 0 开始）。"""
    if not isinstance(raw, (list, tuple)) or not raw:
        raise AudioError("brief.sections 不得为空")
    out = []
    prev_end = None
    for i, s in enumerate(raw):
        if not isinstance(s, dict):
            raise AudioError("sections[%d] 须为对象" % i)
        kind = s.get("kind")
        if kind not in _KIND_GAIN:
            raise AudioError("sections[%d].kind 非法：%r" % (i, kind))
        try:
            a = float(s["start_s"])
            b = float(s["end_s"])
        except (KeyError, TypeError, ValueError):
            raise AudioError("sections[%d] 缺 start_s/end_s" % i)
        if b <= a:
            raise AudioError("sections[%d] end_s 须 > start_s" % i)
        if prev_end is not None and abs(a - prev_end) > 1e-6:
            raise AudioError("sections[%d] 必须无缝衔接（前段 end=%r，本段 start=%r）"
                             % (i, prev_end, a))
        prev_end = b
        out.append({"kind": kind, "start_s": a, "end_s": b})
    if out[0]["start_s"] > 1e-6:
        raise AudioError("sections 须从 0 开始（得到 %r）" % out[0]["start_s"])
    if abs(out[-1]["end_s"] - duration_s) > 1e-3:
        raise AudioError("sections 末段 end_s(%r) 与时长(%r) 不符（> 1e-3）"
                         % (out[-1]["end_s"], duration_s))
    out[-1]["end_s"] = float(duration_s)          # 末段严格等于真实时长
    return out


def _tempo_of(brief: dict) -> dict:
    tp = brief.get("tempo")
    if not isinstance(tp, dict):
        raise AudioError("brief.tempo 缺失")
    try:
        bpm = float(tp["bpm"])
    except (KeyError, TypeError, ValueError):
        raise AudioError("brief.tempo.bpm 缺失或非数值")
    if not (20.0 <= bpm <= 400.0):
        raise AudioError("bpm 越界：%r" % bpm)
    meter = tp.get("meter", [4, 4])
    if not (isinstance(meter, (list, tuple)) and len(meter) == 2):
        raise AudioError("meter 须为 [分子, 分母]：%r" % (meter,))
    meter = [int(meter[0]), int(meter[1])]
    if meter[0] < 1 or meter[1] < 1:
        raise AudioError("meter 非法：%r" % (meter,))
    offset = float(tp.get("offset_s", 0.0))
    return {"bpm": bpm, "meter": meter, "offset_s": offset}


def _key_of(obj: dict) -> dict:
    k = obj.get("key")
    if k is None:
        return {"tonic": "A", "mode": "minor", "source": "default_aminor"}
    if not isinstance(k, dict):
        raise AudioError("key 须为对象：%r" % (k,))
    tonic = k.get("tonic")
    mode = k.get("mode")
    if tonic not in _PC:
        raise AudioError("key.tonic 非法：%r" % (tonic,))
    if mode not in _PROG:
        raise AudioError("key.mode 非法：%r" % (mode,))
    return {"tonic": tonic, "mode": mode, "source": "doc"}


def _plan_from_brief(brief: dict) -> dict:
    if not isinstance(brief, dict):
        raise AudioError("brief 须为对象")
    tempo = _tempo_of(brief)
    secs = brief.get("sections")
    if not isinstance(secs, (list, tuple)) or not secs:
        raise AudioError("brief.sections 缺失")
    duration = brief.get("duration_s")
    duration = float(duration) if duration is not None else float(secs[-1]["end_s"])
    if duration <= 0:
        raise AudioError("duration_s 须 > 0：%r" % duration)
    sections = _norm_sections(secs, duration)
    for i, s in enumerate(sections):
        s["index"] = i
        s["is_last"] = (i == len(sections) - 1)
    return {"duration_s": duration, "tempo": tempo, "key": _key_of(brief),
            "sections": sections, "sample_rate": SAMPLE_RATE, "channels": CHANNELS}


def _plan_from_score(score_doc: dict) -> dict:
    """从（已校验的）乐谱文档还原渲染计划 —— 与 `_plan_from_brief` 同构，
    故 `synthesize(make(brief))` 能逐字节复现 `make` 的 WAV。"""
    doc = C.normalize("score", score_doc)
    src = doc["source"]
    if src["sample_rate"] != SAMPLE_RATE:
        raise AudioError("仅支持 %d Hz（文档为 %r）" % (SAMPLE_RATE, src["sample_rate"]))
    if src["channels"] != CHANNELS:
        raise AudioError("仅支持 %d 声道（文档为 %r）" % (CHANNELS, src["channels"]))
    tempo = {"bpm": float(doc["tempo"]["bpm"]),
             "meter": [int(x) for x in doc["tempo"]["meter"]],
             "offset_s": float(doc["tempo"]["offset_s"])}
    sections = []
    for i, s in enumerate(doc["sections"]):
        sections.append({"kind": s["kind"], "start_s": float(s["start_s"]),
                         "end_s": float(s["end_s"]), "index": i,
                         "is_last": (i == len(doc["sections"]) - 1)})
    return {"duration_s": float(src["duration_s"]), "tempo": tempo,
            "key": _key_of(doc), "sections": sections,
            "sample_rate": SAMPLE_RATE, "channels": CHANNELS}


# ---------------------------------------------------------------- 渲染

_PAD_CACHE = {}


def _pad_cached(freqs, n: int, sr: int, gain: float, detune: float) -> list:
    key = (tuple(round(f, 6) for f in freqs), n, sr, round(gain, 6), round(detune, 6))
    buf = _PAD_CACHE.get(key)
    if buf is None:
        buf = _pad(freqs, n, sr, gain, detune)
        _PAD_CACHE[key] = buf
    return list(buf)


def _render(plan: dict, wav_path: str) -> dict:
    sr = int(plan["sample_rate"])
    ch = int(plan["channels"])
    duration = float(plan["duration_s"])
    total = int(round(duration * sr))
    if total <= 0:
        raise AudioError("渲染帧数须 > 0")
    tempo = plan["tempo"]
    bpm = float(tempo["bpm"])
    meter = [int(x) for x in tempo["meter"]]
    offset = float(tempo["offset_s"])
    key = plan["key"]

    beat_s = 60.0 / bpm
    bar_s = meter[0] * (4.0 / meter[1]) * beat_s
    step_s = bar_s / STEPS_PER_BAR
    sections = plan["sections"]

    master = [[0.0] * total, [0.0] * total]

    def sec_limit(t: float, end_s: float) -> int:
        return max(0, int(round((end_s - t) * sr)))

    # ---- 逐段落铺乐器
    for sec in sections:
        kind = sec["kind"]
        g = dict(_KIND_GAIN[kind])
        t0, t1 = float(sec["start_s"]), float(sec["end_s"])
        # 末段副歌「最满」：lead 加倍 + crash + hat 全开
        lead_double = bool(sec["is_last"] and kind in ("chorus", "drop"))
        if lead_double:
            g["lead"] = min(1.0, g["lead"] + 0.15)
            g["hat"] = 1.0
        crash = bool(kind in ("chorus", "drop"))

        # --- 和弦垫：每个小节边界起一个和弦（段首若不在小节线上则从段首起）
        if g["pad"] > 0:
            b = int(math.ceil((t0 - offset) / bar_s - 1e-9))
            starts = []
            if offset + b * bar_s > t0 + 1e-9:
                starts.append(t0)
            while offset + b * bar_s < t1 - 1e-9:
                starts.append(offset + b * bar_s)
                b += 1
            for st in starts:
                bar_index = int(math.floor((st - offset) / bar_s + 1e-9))
                root, quality = _chord_at(bar_index, key)
                freqs = [_midi_hz(m) for m in _chord_tones(root + 12, quality)]
                n = min(int(round(bar_s * sr)), sec_limit(st, t1))
                if n <= 0:
                    continue
                pos = int(round(st * sr))
                buf = _pad_cached(freqs, n, sr, _GAIN["pad"] * g["pad"], 0.0035)
                _place(master, pos, buf, 0.0, sec_limit(st, t1))
                buf = _pad_cached(freqs, n, sr, _GAIN["pad"] * g["pad"], -0.0028)
                _place(master, pos, buf, 0.0, sec_limit(st, t1))

        # --- 16 分网格上的鼓 / bass / lead
        k = int(math.ceil((t0 - offset) / step_s - 1e-9))
        while True:
            t = offset + k * step_s
            if t >= t1 - 1e-9:
                break
            if t >= t0 - 1e-9:
                bar_index = int(math.floor((t - offset) / bar_s + 1e-9))
                st = k - bar_index * STEPS_PER_BAR
                root, quality = _chord_at(bar_index, key)
                pos = int(round(t * sr))
                lim = sec_limit(t, t1)
                if lim > 0:
                    if st in _KICK_STEPS[kind]:
                        _place(master, pos, _kick(min(int(0.30 * sr), lim), sr,
                                                  _GAIN["kick"] * g["kick"]), 0.0, lim)
                    if st in _SNARE_STEPS[kind]:
                        vel = g["snare"]
                        if kind == "pre_chorus" and bar_index == int(
                                math.floor((t1 - offset - step_s) / bar_s + 1e-9)) and st >= 8:
                            vel *= 0.45 + 0.55 * ((st - 8) / 7.0)     # 段末军鼓滚奏渐强
                        _place(master, pos, _snare(min(int(0.25 * sr), lim), sr,
                                                   _GAIN["snare"] * vel, 0x51A2 + st * 7), 0.0, lim)
                    if st in _HAT_STEPS[kind]:
                        acc = 1.0 if st % 4 == 0 else 0.62
                        _place(master, pos, _hat(min(int(0.06 * sr), lim), sr,
                                                 _GAIN["hat"] * g["hat"] * acc, 0x1A7 + st * 13),
                               0.18, lim)
                    if crash and st == 0:
                        _place(master, pos, _hat(min(int(1.30 * sr), lim), sr,
                                                 _GAIN["crash"], 0x0C4A5 + bar_index * 31, 0.900),
                               0.0, lim)
                    if g["bass"] > 0 and st in _BASS_STEPS:
                        j = st // 2
                        bnote = root - 12 + (0 if j % 2 == 0 else 7)
                        if j == 7:
                            bnote = root - 12 + 12
                        nb = min(int(0.42 * sr), lim)
                        _place(master, pos, _tone(_midi_hz(bnote), nb, sr,
                                                  _GAIN["bass"] * g["bass"],
                                                  0.008, 0.10, 0.72, 0.06,
                                                  ((1.0, 1.0), (2.0, 0.22), (3.0, 0.08))),
                               0.0, lim)
                    if g["lead"] > 0:
                        steps = _LEAD_STEPS_BEAT if kind in ("chorus", "drop", "pre_chorus") \
                            else _LEAD_STEPS_HALF
                        if st in steps:
                            idx = steps.index(st)
                            third = _THIRD[quality]
                            motif = (0, 7, third, 12)
                            lnote = root + 24 + motif[idx % len(motif)]
                            nl = min(int(0.55 * sr), lim)
                            _place(master, pos, _tone(_midi_hz(lnote), nl, sr,
                                                      _GAIN["lead"] * g["lead"],
                                                      0.020, 0.18, 0.60, 0.14,
                                                      ((1.0, 1.0), (2.0, 0.18), (3.0, 0.10))),
                                   0.22, lim)
                            if lead_double:
                                _place(master, pos, _tone(_midi_hz(lnote + 12), nl, sr,
                                                          _GAIN["lead"] * g["lead"] * 0.42,
                                                          0.020, 0.18, 0.55, 0.14,
                                                          ((1.0, 1.0), (3.0, 0.12))),
                                       0.22, lim)
            k += 1

    # ---- 首尾淡入淡出（防爆音；末段收尾）
    fade_in = min(total, int(0.020 * sr))
    fade_out = min(total, int(0.150 * sr))
    for i in range(fade_in):
        e = i / fade_in
        master[0][i] *= e
        master[1][i] *= e
    for i in range(fade_out):
        e = i / fade_out
        master[0][total - 1 - i] *= e
        master[1][total - 1 - i] *= e

    # ---- 软限幅（tanh）→ 峰值归一化到 PEAK_TARGET
    peak = 0.0
    for c in (0, 1):
        chn = master[c]
        for v in chn:
            av = v if v >= 0.0 else -v
            if av > peak:
                peak = av
    drive = (0.95 / peak) if peak > 1e-12 else 1.0
    k_scale = PEAK_TARGET / math.tanh(0.95)
    n_samples = total * ch
    pcm = array.array("h", bytes(2 * n_samples))
    idx = 0
    for i in range(total):
        for c in (0, 1):
            y = math.tanh(master[c][i] * drive) * k_scale
            iv = int(y * 32767.0 + (0.5 if y >= 0.0 else -0.5))
            if iv > 32767:
                iv = 32767
            elif iv < -32768:
                iv = -32768
            pcm[idx] = iv
            idx += 1
    if sys.byteorder != "little":
        pcm.byteswap()
    payload = pcm.tobytes()

    d = os.path.dirname(os.path.abspath(wav_path))
    if d:
        os.makedirs(d, exist_ok=True)
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(ch)
        wf.setsampwidth(SAMPWIDTH)
        wf.setframerate(sr)
        wf.writeframes(payload)

    with open(wav_path, "rb") as f:                    # 摘要以**真实文件**为准
        blob = f.read()
    return {"path": wav_path,
            "sha256": hashlib.sha256(blob).hexdigest(),
            "frames": total, "sample_rate": sr, "channels": ch,
            "duration_s": total / float(sr),
            "samples": n_samples,
            "bytes": len(blob)}


# ---------------------------------------------------------------- 读取 / 分析

def _read_wav(path: str) -> dict:
    if not os.path.isfile(path):
        raise AudioError("WAV 不存在：%r" % path)
    try:
        with wave.open(path, "rb") as wf:
            ch = wf.getnchannels()
            sw = wf.getsampwidth()
            sr = wf.getframerate()
            n = wf.getnframes()
            raw = wf.readframes(n)
    except (wave.Error, EOFError, struct.error) as e:
        raise AudioError("WAV 无法解析：%s（%s）" % (path, e))
    if sw != SAMPWIDTH:
        raise AudioError("仅支持 16-bit PCM（%r 为 %d-bit）" % (path, sw * 8))
    if ch < 1:
        raise AudioError("声道数非法：%r" % ch)
    if len(raw) < n * ch * sw:
        raise AudioError("WAV 数据不足：%r" % path)
    a = array.array("h")
    a.frombytes(raw[:n * ch * sw])
    if sys.byteorder != "little":
        a.byteswap()
    mono = [0.0] * n
    if ch == 1:
        for i in range(n):
            mono[i] = a[i] / 32768.0
    else:
        for i in range(n):
            s = 0.0
            for c in range(ch):
                s += a[i * ch + c]
            mono[i] = s / (ch * 32768.0)
    return {"frames": n, "sample_rate": sr, "channels": ch, "mono": mono,
            "path": path}


def _energy_map(rms: float) -> float:
    """RMS(线性) → 能量 ∈ [0,1)：`v = rms/ENERGY_REF`，`v > ENERGY_KNEE` 处转 tanh 软膝。

    线性段严格等于 `rms/ENERGY_REF`（便于复算）；软膝段单调、有界 < 1，
    避免少数强瞬态 hop 被硬削平成一片 1.0（那会抹掉峰值结构）。
    """
    v = rms / ENERGY_REF
    if v <= ENERGY_KNEE:
        return v if v > 0.0 else 0.0
    k = 1.0 - ENERGY_KNEE
    return ENERGY_KNEE + k * math.tanh((v - ENERGY_KNEE) / k)


def _rms_envelope(mono: list, sr: int, hop_s: float, n_hops: int) -> list:
    hop = max(1, int(round(hop_s * sr)))
    vals = []
    total = len(mono)
    for i in range(n_hops):
        a = i * hop
        b = min(total, a + hop)
        if b <= a:
            vals.append(0.0)
            continue
        s = 0.0
        for j in range(a, b):
            v = mono[j]
            s += v * v
        vals.append(math.sqrt(s / (b - a)))
    return vals


def _beats_of(tempo: dict, duration_s: float) -> tuple:
    bpm = float(tempo["bpm"])
    offset = float(tempo["offset_s"])
    num = int(tempo["meter"][0])
    beat_s = 60.0 / bpm
    beats = []
    i = 0
    while True:
        t = offset + i * 60.0 / bpm
        if t > duration_s + 1e-9:
            break
        beats.append(t)
        i += 1
        if i > 100000:
            raise AudioError("beats 数量异常（bpm=%r）" % bpm)
    if not beats:
        raise AudioError("beats 为空（duration=%r bpm=%r offset=%r）" % (duration_s, bpm, offset))
    downbeats = [beats[i] for i in range(0, len(beats), num)]
    if not downbeats:
        raise AudioError("downbeats 为空")
    return beats, downbeats


def _section_energy(values: list, hop_s: float, start_s: float, end_s: float) -> float:
    """段落能量 = 该段覆盖 hop 的均值（与实际 RMS 同源）。"""
    acc = 0.0
    n = 0
    for i, v in enumerate(values):
        a = i * hop_s
        b = a + hop_s
        if a < end_s - 1e-9 and b > start_s + 1e-9:
            acc += v
            n += 1
    if n == 0:
        return 0.0
    return acc / n


def analyze(wav_path: str, brief: dict) -> dict:
    """离线分析 WAV + 结合 brief 的结构 ⇒ `wmmv.score/1` 文档（过契约校验）。

    段落的 `energy` **由实际 RMS 包络测得**（不是照抄 brief 的目标值），
    故副歌高于主歌是音频事实而非声明。
    """
    if not isinstance(brief, dict):
        raise AudioError("brief 须为对象")
    tempo = _tempo_of(brief)
    info = _read_wav(wav_path)
    sr = info["sample_rate"]
    duration = info["frames"] / float(sr)
    secs = brief.get("sections")
    if not isinstance(secs, (list, tuple)) or not secs:
        raise AudioError("brief.sections 缺失")
    sections = _norm_sections(secs, duration)
    if duration <= 0:
        raise AudioError("WAV 时长为 0：%r" % wav_path)

    hop_s = float(brief.get("energy_hop_s", DEFAULT_HOP_S))
    if not (1e-4 <= hop_s <= 1.0):
        raise AudioError("energy_hop_s 越界：%r" % hop_s)
    #: 包络长度必须由 hop_s 与时长**唯一决定**（P0 `wmmv.score/1` 的硬约束）：
    #: `len(values) == round(duration/hop_s) + 1` —— **含端点**（t=0 与 t=duration 各一个）。
    #: 首版用 `ceil(duration/hop_s)`（=480）差一，被契约层挡下：「读数无法复算」。
    n_hops = int(round(duration / hop_s)) + 1
    raw = _rms_envelope(info["mono"], sr, hop_s, n_hops)
    values = [_energy_map(v) for v in raw]

    out_sections = []
    for s in sections:
        e = _section_energy(values, hop_s, s["start_s"], s["end_s"])
        out_sections.append({"kind": s["kind"],
                             "start_s": s["start_s"], "end_s": s["end_s"],
                             "energy": min(1.0, max(0.0, e))})

    beats, downbeats = _beats_of(tempo, duration)
    key = _key_of(brief)
    with open(wav_path, "rb") as f:
        file_digest = hashlib.sha256(f.read()).hexdigest()
    doc = {
        "schema": C.SCHEMAS["score"],
        "id": "score_main",
        "source": {"path": os.path.basename(wav_path),
                   "digest": file_digest,
                   "duration_s": duration,
                   "sample_rate": sr, "channels": info["channels"]},
        "tempo": {"bpm": tempo["bpm"], "meter": list(tempo["meter"]),
                  "offset_s": tempo["offset_s"]},
        "key": {"tonic": key["tonic"], "mode": key["mode"],
                "confidence": 0.9 if key["source"] == "doc" else 0.3},
        "beats": beats,
        "downbeats": downbeats,
        "sections": out_sections,
        "energy": {"hop_s": hop_s, "metric": "rms", "values": values},
    }
    return C.normalize("score", doc)


# ---------------------------------------------------------------- 公开面

def synthesize(score_doc: dict, wav_path: str) -> dict:
    """按乐谱文档渲染 WAV（纯函数：同 doc ⇒ 逐字节同文件）。返回 wav info dict。"""
    plan = _plan_from_score(score_doc)
    return _render(plan, wav_path)


def make(brief: dict, out_dir: str) -> dict:
    """产线入口：合成 `<out_dir>/music.wav` + 分析出 `wmmv.score/1` 乐谱。

    返回 `{"score": <score doc>, "wav": {"path","sha256","frames","sample_rate",
    "channels","duration_s",...}}`。
    """
    from . import paths
    paths.ensure_lingshu()                     # 环境契约（幂等；B1 只读边界）
    if not out_dir:
        raise AudioError("out_dir 不得为空")
    os.makedirs(out_dir, exist_ok=True)
    wav_path = os.path.join(out_dir, "music.wav")
    plan = _plan_from_brief(brief)
    info = _render(plan, wav_path)
    doc = analyze(wav_path, brief)
    if doc["source"]["digest"] != info["sha256"]:
        raise AudioError("乐谱 source.digest 与 WAV 实际摘要不一致（自证失败）")
    return {"score": doc, "wav": info}


if __name__ == "__main__":                     # pragma: no cover - 手动探针
    import json

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "brief.json"), encoding="utf-8") as f:
        _brief = json.load(f)
    _out = os.path.join(here, "out", "_audio_probe")
    _r = make(_brief, _out)
    print(json.dumps({"wav": _r["wav"],
                      "sections": _r["score"]["sections"]}, ensure_ascii=False, indent=2))
