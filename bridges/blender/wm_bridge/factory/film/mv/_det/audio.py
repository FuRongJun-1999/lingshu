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
import json
import math
import os
import struct
import sys
import wave

from ... import score as _score
from . import contracts as C

__all__ = ["make", "synthesize", "analyze",
           "analyze_external_audio", "consolidate_score", "load_consolidated_score",
           "AudioError", "SAMPLE_RATE", "CHANNELS", "SAMPWIDTH", "DEFAULT_HOP_S"]

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
        # #2113：修前在 src 上**原地**做截断淡出——_place 的 pan=0 路径把同一个
        # list 先后传给左右两个 _add，右声道拿到已被左声道淡出一次的缓冲、再淡出
        # 一次（居中乐器截断点尾部差 220 倍，实测 R/L=1/220）。淡出改在副本上做，
        # src 保持只读契约。
        tail = list(src[n - f:n])
        for j in range(f):
            tail[j] *= (f - j) / f
        src = src[:n - f] + tail
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
    # #2119：「从 0 开始」修前只拦正偏移（`> 1e-6`）——负 start_s 静默通过，
    # 渲染期整段样点被负索引切片静默丢弃（fail-open，模块 docstring 声明要杜绝
    # 的「不静默降级」）。改用与衔接缝同款的绝对值判等。
    if abs(out[0]["start_s"]) > 1e-6:
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


# ---------------------------------------------------------------- 外部音频离线分析与固化

_TONICS = _score.TONICS
_KS_MAJOR = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
_KS_MINOR = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)


def _require_finite_num(v, where: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise AudioError("%s 须为有限数值，实为 %r" % (where, v))
    fv = float(v)
    if not math.isfinite(fv):
        raise AudioError("%s 须为有限数值，实为 %r" % (where, v))
    return fv


def _validate_meter(meter) -> tuple[int, int]:
    if not isinstance(meter, (tuple, list)) or len(meter) != 2:
        raise AudioError("meter 须为长度为 2 的序列，实为 %r" % (meter,))
    for idx, x in enumerate(meter):
        if isinstance(x, bool) or not isinstance(x, int):
            raise AudioError("meter[%d] 须为整数（不得为 bool），实为 %r" % (idx, x))
    mv = (int(meter[0]), int(meter[1]))
    if mv not in _score.METERS:
        raise AudioError("meter 须属于闭集 %s，实为 %r" % (_score.METERS, mv))
    return mv


def _validate_hop_s(energy_hop_s) -> float:
    hop = _require_finite_num(energy_hop_s, "energy_hop_s")
    if not (1e-4 <= hop <= 1.0):
        raise AudioError("energy_hop_s 越界（须在 [1e-4, 1.0]）：%r" % (energy_hop_s,))
    return hop


def _validate_score_id(score_id) -> str:
    if not isinstance(score_id, str) or not score_id.strip():
        raise AudioError("score_id 须为非空字符串，实为 %r" % (score_id,))
    return score_id.strip()


def _validate_audio_path(audio_path) -> str:
    if not isinstance(audio_path, str) or not audio_path.strip():
        raise AudioError("audio_path 须为非空文件路径，实为 %r" % (audio_path,))
    if not os.path.isfile(audio_path):
        raise AudioError("音频文件不存在：%r" % (audio_path,))
    return audio_path


def _norm_sections_external(raw, duration_s: float) -> list[dict]:
    if not isinstance(raw, (list, tuple)) or not raw:
        raise AudioError("brief.sections 不得为空")
    out = []
    prev_end = None
    for i, s in enumerate(raw):
        if not isinstance(s, dict):
            raise AudioError("sections[%d] 须为对象" % i)
        kind = s.get("kind")
        if kind not in _score.SECTION_KINDS:
            raise AudioError("sections[%d].kind 非法：%r" % (i, kind))
        if "start_s" not in s or "end_s" not in s:
            raise AudioError("sections[%d] 缺 start_s/end_s" % i)
        a = _require_finite_num(s["start_s"], "sections[%d].start_s" % i)
        b = _require_finite_num(s["end_s"], "sections[%d].end_s" % i)
        if a < 0.0 or b <= a:
            raise AudioError("sections[%d] 须非负且 end_s > start_s（实为 [%r, %r)）" % (i, a, b))
        if prev_end is not None and abs(a - prev_end) > 1e-6:
            raise AudioError("sections[%d] 必须无缝衔接（前段 end=%r，本段 start=%r）"
                             % (i, prev_end, a))
        prev_end = b
        out.append({"kind": kind, "start_s": a, "end_s": b})
    if abs(out[0]["start_s"]) > 1e-6:
        raise AudioError("sections 须从 0 开始（得到 %r）" % out[0]["start_s"])
    if abs(out[-1]["end_s"] - duration_s) > 1e-3:
        raise AudioError("sections 末段 end_s(%r) 与时长(%r) 不符（> 1e-3）"
                         % (out[-1]["end_s"], duration_s))
    out[-1]["end_s"] = float(duration_s)
    return out


def _build_sections_output(brief: dict | None, duration_s: float,
                           values: list[float], hop_s: float) -> list[dict]:
    if brief is not None and "sections" in brief and brief["sections"] is not None:
        secs = _norm_sections_external(brief["sections"], duration_s)
        out = []
        for s in secs:
            e = _section_energy(values, hop_s, s["start_s"], s["end_s"])
            out.append({"kind": s["kind"], "start_s": s["start_s"],
                        "end_s": s["end_s"], "energy": min(1.0, max(0.0, e))})
        return out
    mean_e = sum(values) / float(len(values)) if values else 0.0
    return [{"kind": "instrumental", "start_s": 0.0, "end_s": float(duration_s),
             "energy": min(1.0, max(0.0, mean_e))}]


def _pearson_corr12(a: list[float], b: tuple[float, ...]) -> float:
    ma = sum(a) / 12.0
    mb = sum(b) / 12.0
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da > 0.0 and db > 0.0 else 0.0


def _estimate_key_from_chroma(chroma: list[float],
                              open_chroma: list[float] | None = None) -> dict | None:
    total = sum(chroma)
    if total <= 0.0 or not math.isfinite(total):
        return None
    cn = [v / total for v in chroma]
    candidates = []
    for ti in range(12):
        rot = [cn[(ti + j) % 12] for j in range(12)]
        candidates.append((_pearson_corr12(rot, _KS_MAJOR), ti, "major"))
        candidates.append((_pearson_corr12(rot, _KS_MINOR), ti, "minor"))
    candidates.sort(reverse=True)
    best_r, best_ti, best_mode = candidates[0]

    if open_chroma is not None:
        o_tot = sum(open_chroma)
        if o_tot > 0.0 and math.isfinite(o_tot):
            on = [v / o_tot for v in open_chroma]
            if best_mode == "major":
                rel_ti = (best_ti + 9) % 12
                maj_sc = on[best_ti] + on[(best_ti + 4) % 12] + 1.2 * on[(best_ti + 7) % 12]
                min_sc = 1.2 * on[rel_ti] + on[best_ti] + on[(best_ti + 4) % 12]
                if min_sc > maj_sc * 1.05:
                    best_ti, best_mode = rel_ti, "minor"
            else:
                rel_ti = (best_ti + 3) % 12
                min_sc = 1.2 * on[best_ti] + on[rel_ti] + on[(best_ti + 7) % 12]
                maj_sc = on[rel_ti] + on[(rel_ti + 4) % 12] + 1.2 * on[(rel_ti + 7) % 12]
                if maj_sc > min_sc * 1.05:
                    best_ti, best_mode = rel_ti, "major"

    conf = max(0.0, min(1.0, 0.5 * (best_r + 1.0)))
    return {"tonic": _TONICS[best_ti], "mode": best_mode, "confidence": conf}


def _detect_beats_pcm(mono: list[float], sr: int, meter: tuple[int, int]) -> tuple[float, float, list[float], list[float]]:
    """纯标准库确定性波形节拍/BPM/强拍实测（无 `brief.tempo` 盲测）。"""
    dur = len(mono) / float(sr)
    stride = max(1, sr // 11025)
    ds = mono[::stride]
    sr_ds = sr / float(stride)
    n_ds = len(ds)
    if n_ds < 32:
        raise AudioError("音频过短，无法实测节拍（样点数=%d）" % n_ds)

    # 低通（捕获底鼓/贝斯）+ 一阶差分高通（捕获军鼓/踩镲/瞬态）
    lp = [0.0] * n_ds
    hp = [0.0] * n_ds
    alpha = 0.14
    prev_lp = 0.0
    prev_x = 0.0
    for i, x in enumerate(ds):
        prev_lp = prev_lp + alpha * (x - prev_lp)
        lp[i] = prev_lp
        hp[i] = x - prev_x
        prev_x = x

    hop_n = max(1, int(round(0.005 * sr_ds)))
    dt = hop_n / sr_ds
    n_f = n_ds // hop_n
    if n_f < 8:
        raise AudioError("音频过短，无法形成节拍包络（帧数=%d）" % n_f)

    el = [0.0] * n_f
    eh = [0.0] * n_f
    ef = [0.0] * n_f
    for k in range(n_f):
        a = k * hop_n
        b = min(n_ds, a + hop_n)
        sl = sh = sf = 0.0
        for j in range(a, b):
            vl = lp[j]
            vh = hp[j]
            vf = ds[j]
            sl += vl * vl
            sh += vh * vh
            sf += vf * vf
        inv = 1.0 / (b - a)
        el[k] = math.sqrt(sl * inv)
        eh[k] = math.sqrt(sh * inv)
        ef[k] = math.sqrt(sf * inv)

    # 因果半波整流对数通量：比较 k 后 15ms 与 k 前 15ms，消除起音延迟偏移
    flux = [0.0] * n_f
    flux_low = [0.0] * n_f
    w = min(3, max(1, n_f // 4))
    for k in range(n_f):
        if k == 0:
            la = sum(el[:w]) / w
            ha = sum(eh[:w]) / w
            fa = sum(ef[:w]) / w
            flux_low[0] = math.log1p(25.0 * la)
            flux[0] = 1.6 * flux_low[0] + math.log1p(25.0 * ha) + 0.8 * math.log1p(25.0 * fa)
            continue
        b0 = max(0, k - w)
        a1 = min(n_f, k + w)
        lb = sum(el[b0:k]) / (k - b0)
        la = sum(el[k:a1]) / (a1 - k)
        hb = sum(eh[b0:k]) / (k - b0)
        ha = sum(eh[k:a1]) / (a1 - k)
        fb = sum(ef[b0:k]) / (k - b0)
        fa = sum(ef[k:a1]) / (a1 - k)
        dl = math.log1p(25.0 * la) - math.log1p(25.0 * lb)
        dh = math.log1p(25.0 * ha) - math.log1p(25.0 * hb)
        df = math.log1p(25.0 * fa) - math.log1p(25.0 * fb)
        fl = dl if dl > 0.0 else 0.0
        fh = dh if dh > 0.0 else 0.0
        ff = df if df > 0.0 else 0.0
        flux_low[k] = fl
        flux[k] = 1.6 * fl + 1.0 * fh + 0.8 * ff

    rad = 15
    pref = [0.0] * (n_f + 1)
    for k in range(n_f):
        pref[k + 1] = pref[k] + flux[k]
    flux_odf = [0.0] * n_f
    for k in range(n_f):
        lo = max(0, k - rad)
        hi = min(n_f, k + rad + 1)
        m = (pref[hi] - pref[lo]) / (hi - lo)
        v = flux[k] - m
        flux_odf[k] = v if v > 0.0 else 0.0
    flux_odf[0] = max(flux_odf[0], flux[0] * 0.5)

    def _prior(bpm_val: float) -> float:
        return math.exp(-0.5 * (math.log2(bpm_val / 115.0) / 0.85) ** 2)

    min_lag = max(2, int(round((60.0 / 190.0) / dt)))
    max_lag = min(max(min_lag + 1, n_f // 2), int(round((60.0 / 55.0) / dt)))
    acf_max = min(n_f - 1, 2 * max_lag + 2)
    acf = [0.0] * (acf_max + 1)
    mean_f = sum(flux_odf) / n_f
    centered = [v - mean_f for v in flux_odf]
    var_f = sum(v * v for v in centered)
    if var_f <= 1e-18:
        raise AudioError("音频缺乏节拍瞬态特征，无法检出稳定节拍")
    for lag in range(min(acf_max, min_lag // 2), acf_max + 1):
        s = 0.0
        lim = n_f - lag
        for i in range(lim):
            s += centered[i] * centered[i + lag]
        acf[lag] = s / (lim * var_f / n_f)

    cands = []
    b = 55.0
    while b <= 190.0:
        lag_f = (60.0 / b) / dt
        l0 = int(round(lag_f))
        l2 = int(round(2.0 * lag_f))
        sc = acf[l0] if l0 < len(acf) else 0.0
        if l2 < len(acf):
            sc = 0.65 * sc + 0.35 * acf[l2]
        sc *= _prior(b)
        cands.append((sc, b))
        b += 0.5
    cands.sort(reverse=True)

    peaks = []
    for _sc, b in cands:
        if all(abs(b - pb) >= 5.0 for pb in peaks):
            peaks.append(b)
            if len(peaks) >= 4:
                break
    if not peaks:
        peaks = [120.0]

    sm = [0.0] * n_f
    for k in range(n_f):
        s = 0.60 * flux_odf[k]
        if k >= 1:
            s += 0.25 * flux_odf[k - 1]
        if k + 1 < n_f:
            s += 0.25 * flux_odf[k + 1]
        sm[k] = s

    best_comb = (-1.0, peaks[0], 0.0)
    for bp0 in peaks:
        b_lo = max(40.0, bp0 - 1.5)
        b_hi = min(220.0, bp0 + 1.5)
        nb = int(round((b_hi - b_lo) / 0.05)) + 1
        for bi in range(nb):
            bpm_cand = b_lo + bi * 0.05
            period = 60.0 / bpm_cand
            n_ph = max(1, int(round(period / dt)))
            pr = _prior(bpm_cand)
            for pi in range(n_ph):
                off = pi * dt
                tot = 0.0
                i = 0
                while True:
                    t = off + i * period
                    if t > dur - 0.01:
                        break
                    idx = int(round(t / dt))
                    if 0 <= idx < n_f:
                        tot += sm[idx]
                    i += 1
                score = tot * pr
                if score > best_comb[0]:
                    best_comb = (score, bpm_cand, off)

    bpm_est = round(best_comb[1], 4)
    off_est = best_comb[2]
    period = 60.0 / bpm_est
    if period - off_est < 0.025:
        off_est = 0.0
    off_est = round(off_est, 6)

    beats = []
    i = 0
    while True:
        t = off_est + i * period
        if t > dur + 1e-9:
            break
        beats.append(t)
        i += 1
    if not beats:
        raise AudioError("未检出有效拍点")

    num = int(meter[0])
    best_p = 0
    best_pscore = -1.0
    for p in range(num):
        ps = 0.0
        pc = 0
        for idx in range(p, len(beats), num):
            k = int(round(beats[idx] / dt))
            s = 0.0
            for dk in range(-2, 4):
                if 0 <= k + dk < n_f:
                    s += flux_low[k + dk]
            ps += s
            pc += 1
        val = ps / pc if pc else 0.0
        if p == 0:
            val *= 1.05
        if val > best_pscore:
            best_pscore = val
            best_p = p
    downbeats = [beats[idx] for idx in range(best_p, len(beats), num)]
    if not downbeats:
        downbeats = [beats[0]]
    return bpm_est, off_est, beats, downbeats


def _detect_key_pcm(mono: list[float], sr: int, offset_s: float) -> dict | None:
    """Goertzel 12 音高类色度向量 + Krumhansl-Schmuckler 调性估计（纯标准库）。"""
    stride_c = max(1, sr // 4410)
    csig = mono[::stride_c]
    sr_c = sr / float(stride_c)
    if len(csig) < 64:
        return None
    n_win = min(2048, len(csig))
    if n_win < 64:
        return None
    hann = [0.5 * (1.0 - math.cos(2.0 * math.pi * j / (n_win - 1))) for j in range(n_win)]
    coeffs = [
        (m % 12, m, 2.0 * math.cos(2.0 * math.pi * (440.0 * (2.0 ** ((m - 69.0) / 12.0))) / sr_c))
        for m in range(36, 84)
    ]

    n_f = 24
    span = max(0, len(csig) - n_win)
    step = max(1, span // max(1, n_f - 1)) if span > 0 else 1
    chroma = [0.0] * 12
    for fi in range(n_f):
        st = fi * step
        if st + n_win > len(csig):
            break
        win_s = [csig[st + j] * hann[j] for j in range(n_win)]
        for pc, _m, coeff in coeffs:
            s1 = 0.0
            s2 = 0.0
            for x in win_s:
                s0 = x + coeff * s1 - s2
                s2 = s1
                s1 = s0
            pwr = s1 * s1 + s2 * s2 - coeff * s1 * s2
            if pwr > 0.0:
                chroma[pc] += math.sqrt(pwr)

    open_chroma = [0.0] * 12
    st0 = min(max(0, len(csig) - n_win), max(0, int(round((offset_s + 0.35) * sr_c))))
    if st0 + n_win <= len(csig):
        win0 = [csig[st0 + j] * hann[j] for j in range(n_win)]
        for pc, m, coeff in coeffs:
            if m < 45:
                continue
            s1 = 0.0
            s2 = 0.0
            for x in win0:
                s0 = x + coeff * s1 - s2
                s2 = s1
                s1 = s0
            pwr = s1 * s1 + s2 * s2 - coeff * s1 * s2
            if pwr > 0.0:
                open_chroma[pc] += math.sqrt(pwr)

    return _estimate_key_from_chroma(chroma, open_chroma)


def _resample_envelope(vals: list[float], target_len: int) -> list[float]:
    if target_len <= 0:
        raise AudioError("包络目标长度须 > 0：%r" % (target_len,))
    if not vals:
        raise AudioError("包络原始序列为空")
    if len(vals) == target_len:
        return list(vals)
    if target_len == 1 or len(vals) == 1:
        return [vals[0]] * target_len
    out = []
    scale = (len(vals) - 1) / float(target_len - 1)
    for i in range(target_len):
        pos = i * scale
        lo = int(math.floor(pos))
        hi = min(len(vals) - 1, lo + 1)
        frac = pos - lo
        out.append(vals[lo] * (1.0 - frac) + vals[hi] * frac)
    return out


def _as_float_list(seq, where: str) -> list[float]:
    raw = seq.tolist() if hasattr(seq, "tolist") else list(seq)
    out = []
    for i, v in enumerate(raw):
        out.append(_require_finite_num(v, "%s[%d]" % (where, i)))
    return out


def _analyze_via_librosa(audio_path: str, brief: dict | None,
                         meter: tuple[int, int], hop_s: float,
                         score_id: str) -> dict:
    try:
        import librosa  # noqa: PLC0415
    except ImportError as exc:
        raise AudioError(
            "缺少可选音频分析依赖 librosa：请安装 `pip install -e \".[film-audio]\"` "
            "或显式切用 backend=\"pcm_wave\"（%s）" % exc
        ) from exc

    loaded = librosa.load(audio_path, sr=None, mono=True)
    if not isinstance(loaded, (tuple, list)) or len(loaded) != 2:
        raise AudioError("librosa.load 返回形状非法：%r" % (type(loaded).__name__,))
    y_raw, sr_raw = loaded
    sr_f = _require_finite_num(sr_raw, "sample_rate")
    if sr_f <= 0.0:
        raise AudioError("sample_rate 须 > 0：%r" % (sr_raw,))
    sr = int(round(sr_f))
    samples = _as_float_list(y_raw, "samples")
    if not samples:
        raise AudioError("音频样本为空：%r" % audio_path)
    if max(abs(v) for v in samples) <= 1e-12:
        raise AudioError("音频全静音，无法分析：%r" % audio_path)

    duration = len(samples) / float(sr)
    if duration <= 0.0:
        raise AudioError("音频时长须 > 0：%r" % audio_path)

    channels = 1
    try:
        with wave.open(audio_path, "rb") as wf:
            channels = max(1, int(wf.getnchannels()))
    except (wave.Error, EOFError, struct.error, OSError):
        channels = 1

    hop_length = max(1, int(round(hop_s * sr)))
    tempo_raw, beat_frames_raw = librosa.beat.beat_track(y=y_raw, sr=sr, hop_length=hop_length)
    if hasattr(tempo_raw, "__len__") and not isinstance(tempo_raw, (str, bytes)):
        t_seq = _as_float_list(tempo_raw, "tempo")
        if not t_seq:
            raise AudioError("librosa.beat.beat_track 返回空 tempo")
        bpm = t_seq[0]
    else:
        bpm = _require_finite_num(tempo_raw, "tempo")
    if not (20.0 <= bpm <= 400.0):
        raise AudioError("librosa 检出 bpm 越界：%r" % bpm)

    if hasattr(librosa, "frames_to_time"):
        beats_raw = librosa.frames_to_time(beat_frames_raw, sr=sr, hop_length=hop_length)
        beats = _as_float_list(beats_raw, "beats")
    else:
        beat_frames = _as_float_list(beat_frames_raw, "beat_frames")
        beats = [f * hop_length / float(sr) for f in beat_frames]
    if not beats:
        raise AudioError("librosa 未检出拍点")
    for idx, t in enumerate(beats):
        if t < 0.0 or (idx > 0 and not (t > beats[idx - 1])):
            raise AudioError("librosa 拍点须非负且严格递增：%r" % (beats,))

    n_hops = int(round(duration / hop_s)) + 1
    rms_feat = getattr(getattr(librosa, "feature", None), "rms", None)
    if callable(rms_feat):
        rms_raw = rms_feat(y=y_raw, frame_length=min(len(samples), max(2048, hop_length * 2)),
                           hop_length=hop_length)
        rms_rows = rms_raw.tolist() if hasattr(rms_raw, "tolist") else list(rms_raw)
        if not rms_rows:
            raise AudioError("librosa.feature.rms 返回空结果")
        row0 = rms_rows[0] if isinstance(rms_rows[0], (list, tuple)) else rms_rows
        rms_vals = _as_float_list(row0, "rms")
        rms_aligned = _resample_envelope(rms_vals, n_hops)
    else:
        rms_aligned = _rms_envelope(samples, sr, hop_s, n_hops)
    values = [_energy_map(max(0.0, v)) for v in rms_aligned]

    num = meter[0]
    best_p = 0
    best_ps = -1.0
    for p in range(num):
        idxs = range(p, len(beats), num)
        sc = 0.0
        cnt = 0
        for bi in idxs:
            hi = min(len(values) - 1, max(0, int(round(beats[bi] / hop_s))))
            sc += values[hi]
            cnt += 1
        avg = sc / cnt if cnt else 0.0
        if p == 0:
            avg *= 1.05
        if avg > best_ps:
            best_ps = avg
            best_p = p
    downbeats = [beats[i] for i in range(best_p, len(beats), num)]
    if not downbeats:
        downbeats = [beats[0]]

    key_doc = None
    chroma_feat = getattr(getattr(librosa, "feature", None), "chroma_stft", None)
    if callable(chroma_feat):
        chroma_raw = chroma_feat(y=y_raw, sr=sr, hop_length=hop_length)
        c_rows = chroma_raw.tolist() if hasattr(chroma_raw, "tolist") else list(chroma_raw)
        if len(c_rows) != 12:
            raise AudioError("librosa.feature.chroma_stft 须返回 12 个音高类，实为 %d" % len(c_rows))
        chroma12 = []
        open12 = []
        for pc in range(12):
            row = c_rows[pc]
            vals_pc = _as_float_list(row, "chroma[%d]" % pc) if isinstance(row, (list, tuple)) else [_require_finite_num(row, "chroma[%d]" % pc)]
            chroma12.append(sum(max(0.0, x) for x in vals_pc))
            open12.append(sum(max(0.0, x) for x in vals_pc[:max(1, len(vals_pc) // 8)]))
        key_doc = _estimate_key_from_chroma(chroma12, open12)

    out_sections = _build_sections_output(brief, duration, values, hop_s)
    with open(audio_path, "rb") as f:
        file_digest = hashlib.sha256(f.read()).hexdigest()

    doc = {
        "schema": C.SCHEMAS["score"],
        "id": score_id,
        "source": {
            "path": os.path.basename(audio_path),
            "digest": file_digest,
            "duration_s": duration,
            "sample_rate": sr,
            "channels": channels,
        },
        "tempo": {
            "bpm": bpm,
            "meter": [meter[0], meter[1]],
            "offset_s": float(beats[0]),
        },
        "beats": beats,
        "downbeats": downbeats,
        "sections": out_sections,
        "energy": {"hop_s": hop_s, "metric": "rms", "values": values},
    }
    if key_doc is not None:
        doc["key"] = key_doc
    return C.normalize("score", doc)


def _analyze_via_pcm_wave(audio_path: str, brief: dict | None,
                          meter: tuple[int, int], hop_s: float,
                          score_id: str) -> dict:
    info = _read_wav(audio_path)
    sr = info["sample_rate"]
    frames = info["frames"]
    mono = info["mono"]
    if frames <= 0 or sr <= 0 or not mono:
        raise AudioError("WAV 音频为空或采样率非法：%r" % audio_path)
    if max(abs(v) for v in mono) <= 1e-9:
        raise AudioError("WAV 音频全静音，无法分析节拍与调性：%r" % audio_path)
    duration = frames / float(sr)

    n_hops = int(round(duration / hop_s)) + 1
    raw_rms = _rms_envelope(mono, sr, hop_s, n_hops)
    values = [_energy_map(v) for v in raw_rms]

    bpm, offset_s, beats, downbeats = _detect_beats_pcm(mono, sr, meter)
    key_doc = _detect_key_pcm(mono, sr, offset_s)
    out_sections = _build_sections_output(brief, duration, values, hop_s)

    with open(audio_path, "rb") as f:
        file_digest = hashlib.sha256(f.read()).hexdigest()

    doc = {
        "schema": C.SCHEMAS["score"],
        "id": score_id,
        "source": {
            "path": os.path.basename(audio_path),
            "digest": file_digest,
            "duration_s": duration,
            "sample_rate": sr,
            "channels": info["channels"],
        },
        "tempo": {
            "bpm": bpm,
            "meter": [meter[0], meter[1]],
            "offset_s": offset_s,
        },
        "beats": beats,
        "downbeats": downbeats,
        "sections": out_sections,
        "energy": {"hop_s": hop_s, "metric": "rms", "values": values},
    }
    if key_doc is not None:
        doc["key"] = key_doc
    return C.normalize("score", doc)


def analyze_external_audio(
    audio_path: str,
    brief: dict | None = None,
    *,
    backend: str = "librosa",
    meter: tuple[int, int] = (4, 4),
    energy_hop_s: float = DEFAULT_HOP_S,
    score_id: str = "score_main",
) -> dict:
    """离线分析外部音频（不依赖 `brief.tempo`），产出通过 `wmmv.score/1` 的乐谱文档。

    · `backend="librosa"`：仅在函数体内按需导入 `librosa`；缺失时 fail-closed 抛 `AudioError`。
    · `backend="pcm_wave"`：纯标准库 16-bit PCM WAV 波形实测器（无外部依赖）。
    """
    path = _validate_audio_path(audio_path)
    if brief is not None and not isinstance(brief, dict):
        raise AudioError("brief 须为对象或 None，实为 %s" % type(brief).__name__)
    meter_v = _validate_meter(meter)
    hop_s = _validate_hop_s(energy_hop_s)
    sid = _validate_score_id(score_id)

    if backend == "librosa":
        return _analyze_via_librosa(path, brief, meter_v, hop_s, sid)
    if backend == "pcm_wave":
        return _analyze_via_pcm_wave(path, brief, meter_v, hop_s, sid)
    raise AudioError("不支持的分析后端 %r（可选：'librosa' / 'pcm_wave'）" % (backend,))


def consolidate_score(score_doc: dict, out_json_path: str) -> dict:
    """将 `wmmv.score/1` 乐谱固化落盘为确定性 JSON，后续执行期零外部音频依赖。"""
    if not isinstance(score_doc, dict):
        raise AudioError("score_doc 须为对象，实为 %s" % type(score_doc).__name__)
    if not isinstance(out_json_path, str) or not out_json_path.strip():
        raise AudioError("out_json_path 须为非空路径，实为 %r" % (out_json_path,))

    norm = C.normalize("score", score_doc)
    file_sha256 = C.write_json(out_json_path, norm)
    verified = C.normalize("score", C.read_json(out_json_path))
    score_digest = C.digest("score", verified)
    return {
        "score": verified,
        "path": out_json_path,
        "digest": score_digest,
        "file_sha256": file_sha256,
    }


def load_consolidated_score(json_path: str) -> dict:
    """读取并校验已固化的 `wmmv.score/1` JSON（纯标准库，零外部音频依赖）。"""
    if not isinstance(json_path, str) or not json_path.strip():
        raise AudioError("json_path 须为非空路径，实为 %r" % (json_path,))
    if not os.path.isfile(json_path):
        raise AudioError("固化乐谱 JSON 不存在：%r" % (json_path,))

    def _reject_nonfinite(tok: str) -> None:
        raise AudioError("固化乐谱 JSON 含非法非有限常量：%r" % (tok,))

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f, parse_constant=_reject_nonfinite)
    except AudioError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AudioError("读取固化乐谱失败：%s（%s）" % (json_path, exc)) from exc

    return C.normalize("score", raw)


if __name__ == "__main__":                     # pragma: no cover - 手动探针
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "brief.json"), encoding="utf-8") as f:
        _brief = json.load(f)
    _out = os.path.join(here, "out", "_audio_probe")
    _r = make(_brief, _out)
    print(json.dumps({"wav": _r["wav"],
                      "sections": _r["score"]["sections"]}, ensure_ascii=False, indent=2))
