# -*- coding: utf-8 -*-
"""影视层 P3 守卫 —— `film-audio` sub-extra 与外部音频离线分析/固化闭环
===========================================================================
依据 `docs/plans/影视级视听语言_理论架构_v0.1.md` §3.5.2（已裁 D2）与 §七 P3：
  1. `pyproject.toml` 声明 `film-audio = ["librosa"]`（不进 `full`、不设版本区间）；
  2. `mv/_det/audio.py` 新增 `analyze_external_audio` / `consolidate_score` /
     `load_consolidated_score`：
     - `backend="librosa"`：函数体内按需导入 `librosa`（`# noqa: PLC0415`）；
       缺失时 fail-closed 抛 `AudioError`，严禁静默降级猜拍点；可用（含受控桩）时
       产出通过 `wmmv.score/1` 契约校验的乐谱文档；
     - `backend="pcm_wave"`：纯标准库 16-bit PCM WAV 波形实测器，不依赖 `brief.tempo`，
       在真实合成音乐 WAV（96 / 120 BPM）上盲测恢复 BPM（误差 <= 1.0 BPM）、
       拍点/强拍（对齐误差 <= 1 帧 @24fps、`downbeats ⊆ beats`）、
       段落能量（`chorus > verse`）与调性（`tonic` / `mode`）；
     - 固化后执行期零依赖：`consolidate_score` 落盘 64 位 sha256 一致，在屏蔽
       `librosa` 的环境中通过 `load_consolidated_score` 驱动 `timeline` +
       `gates_e.check_e1` 全绿（`ACCEPT`）；
     - 全入口对 `NaN`/`±inf`/`bool`、坏 `meter`、空/静音/损坏 WAV、坏 JSON fail-closed；
     - 100% 保留既有 `make`/`synthesize`/`analyze` 行为不变。

运行方式：
    python -X utf8 -m pytest tests/test_film_p3_external_audio.py -v
    python -X utf8 tests/test_film_p3_external_audio.py
    python -X utf8 tests/test_film_p3_external_audio.py --mutate
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
import tempfile
import tomllib
import types
import wave

_HERE = os.path.abspath(os.path.dirname(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
_PKG_ROOT = os.path.join(_REPO, "bridges", "blender")
for _p in (_REPO, _PKG_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from wm_bridge.factory.film import score as SC                           # noqa: E402
from wm_bridge.factory.film._det import gates_e as GE                    # noqa: E402
from wm_bridge.factory.film.mv._det import audio as AU                   # noqa: E402
from wm_bridge.factory.film.mv._det import contracts as MVC              # noqa: E402
from wm_bridge.factory.film.mv._det import timeline as TL                # noqa: E402

PYPROJECT_PATH = os.path.join(_REPO, "pyproject.toml")
AUDIO_PY_PATH = os.path.join(
    _REPO, "bridges", "blender", "wm_bridge", "factory", "film", "mv", "_det", "audio.py"
)


def _raises(exc_types, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_types as exc:
        return exc
    raise AssertionError("预期抛出 %r 但未抛出" % (exc_types,))


def _make_wav_fixture(root_dir: str, bpm: float, tonic: str, mode: str,
                      duration_s: float, sections: list, offset_s: float = 0.0) -> tuple[str, dict]:
    brief = {
        "duration_s": duration_s,
        "fps": 24,
        "tempo": {"bpm": bpm, "meter": [4, 4], "offset_s": offset_s},
        "key": {"tonic": tonic, "mode": mode},
        "sections": sections,
    }
    res = AU.make(brief, root_dir)
    return res["wav"]["path"], res


def _fake_librosa_module(bpm: float = 120.0, sr: int = 22050, duration_s: float = 4.0):
    """构造受控假 `librosa` 模块，验证契约装配与特征提取路径。"""
    mod = types.ModuleType("librosa")
    beat_ns = types.ModuleType("librosa.beat")
    feat_ns = types.ModuleType("librosa.feature")

    n_samples = int(round(sr * duration_s))
    samples = [0.2 if (i % 200) < 40 else 0.02 for i in range(n_samples)]

    def _load(path, sr=None, mono=True):
        return samples, 22050

    def _beat_track(y=None, sr=22050, hop_length=512):
        step_s = 60.0 / bpm
        frames = [int(round((k * step_s) * sr / hop_length)) for k in range(int(duration_s / step_s))]
        return [bpm], frames

    def _frames_to_time(frames, sr=22050, hop_length=512):
        return [float(f) * hop_length / float(sr) for f in frames]

    def _rms(y=None, frame_length=2048, hop_length=512):
        n_frames = max(2, int(round(len(y) / hop_length)) + 1)
        return [[0.08 if i < n_frames // 2 else 0.16 for i in range(n_frames)]]

    def _chroma_stft(y=None, sr=22050, hop_length=512):
        # A minor: A(9)=1.0, C(0)=0.8, E(4)=0.7, D(2)=0.3, F(5)=0.4, G(7)=0.35
        weights = [0.80, 0.02, 0.30, 0.02, 0.70, 0.40, 0.02, 0.35, 0.02, 1.00, 0.02, 0.10]
        return [[w] * 8 for w in weights]

    beat_ns.beat_track = _beat_track
    feat_ns.rms = _rms
    feat_ns.chroma_stft = _chroma_stft
    mod.load = _load
    mod.frames_to_time = _frames_to_time
    mod.beat = beat_ns
    mod.feature = feat_ns
    return mod


# ══════════════════════════════════════════════════════ [A] pyproject 声明面与函数内 import
def test_a1_pyproject_declares_film_audio_sub_extra_not_in_full():
    with open(PYPROJECT_PATH, "rb") as fh:
        data = tomllib.load(fh)
    proj = data["project"]
    deps = proj.get("dependencies") or []
    assert not deps, "轻核 dependencies 必须保持为空：%r" % (deps,)
    extras = proj.get("optional-dependencies") or {}
    assert "film-audio" in extras, "pyproject.toml 缺 film-audio sub-extra"
    assert extras["film-audio"] == ["librosa"], (
        "film-audio 必须如实声明为 ['librosa']（不设版本区间），实为 %r" % (extras["film-audio"],)
    )
    for other_extra, specs in extras.items():
        if other_extra == "film-audio":
            continue
        joined = " ".join(specs or []).lower()
        assert "librosa" not in joined, (
            "librosa 严禁放进 %s（只能留在 film-audio）：%r" % (other_extra, specs)
        )


def test_a2_librosa_import_is_lazy_inside_function_only():
    src = open(AUDIO_PY_PATH, encoding="utf-8").read()
    tree = ast.parse(src)
    # 顶层语句不得出现 import librosa
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] != "librosa", "模块顶层禁止 import librosa"
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] != "librosa", "模块顶层禁止 from librosa import ..."

    # 函数体内必须有 import librosa 且带 # noqa: PLC0415
    found_lazy = False
    lines = src.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Import) and any(a.name.split(".")[0] == "librosa" for a in sub.names):
                    found_lazy = True
                    line_txt = lines[sub.lineno - 1]
                    assert "noqa: PLC0415" in line_txt, (
                        "函数内 import librosa 须带 `# noqa: PLC0415`：%r" % line_txt
                    )
    assert found_lazy, "audio.py 未在函数体内按需 import librosa"


# ══════════════════════════════════════════════════════ [B] librosa 缺失 fail-closed 与受控桩
def test_b1_missing_librosa_fails_closed():
    with tempfile.TemporaryDirectory() as td:
        wav_path, _ = _make_wav_fixture(
            td, 120.0, "A", "minor", 4.0,
            [{"kind": "intro", "start_s": 0.0, "end_s": 2.0, "energy": 0.2},
             {"kind": "chorus", "start_s": 2.0, "end_s": 4.0, "energy": 0.8}],
        )
        old = sys.modules.get("librosa", ...)
        sys.modules["librosa"] = None
        try:
            exc = _raises(AU.AudioError, AU.analyze_external_audio, wav_path, backend="librosa")
            msg = str(exc)
            assert "film-audio" in msg or "pcm_wave" in msg, (
                "缺失 librosa 报错应指明 `.[film-audio]` 或 `backend='pcm_wave'`：%r" % msg
            )
        finally:
            if old is ...:
                sys.modules.pop("librosa", None)
            else:
                sys.modules["librosa"] = old


def test_b2_mocked_librosa_produces_valid_score():
    with tempfile.TemporaryDirectory() as td:
        wav_path, _ = _make_wav_fixture(
            td, 120.0, "A", "minor", 4.0,
            [{"kind": "intro", "start_s": 0.0, "end_s": 2.0, "energy": 0.2},
             {"kind": "chorus", "start_s": 2.0, "end_s": 4.0, "energy": 0.8}],
        )
        fake = _fake_librosa_module(bpm=120.0, sr=22050, duration_s=4.0)
        old = sys.modules.get("librosa", ...)
        sys.modules["librosa"] = fake
        try:
            doc = AU.analyze_external_audio(
                wav_path,
                brief={"sections": [
                    {"kind": "verse", "start_s": 0.0, "end_s": 2.0},
                    {"kind": "chorus", "start_s": 2.0, "end_s": 4.0},
                ]},
                backend="librosa",
                meter=(4, 4),
                energy_hop_s=0.05,
                score_id="score_librosa_stub",
            )
        finally:
            if old is ...:
                sys.modules.pop("librosa", None)
            else:
                sys.modules["librosa"] = old

        ok, why = SC.validate_score(doc)
        assert ok, "受控 librosa 桩产出应过 wmmv.score/1 校验：%s" % why
        assert doc["id"] == "score_librosa_stub"
        assert abs(doc["tempo"]["bpm"] - 120.0) <= 1e-6
        assert len(doc["energy"]["values"]) == int(round(4.0 / 0.05)) + 1
        assert doc["sections"][1]["energy"] > doc["sections"][0]["energy"]
        assert doc["key"]["tonic"] == "A" and doc["key"]["mode"] == "minor"
        with open(wav_path, "rb") as f:
            assert doc["source"]["digest"] == hashlib.sha256(f.read()).hexdigest()


# ══════════════════════════════════════════════════════ [C] pcm_wave 纯标准库盲测实测
def test_c1_pcm_wave_blind_120bpm_aminor_and_sections():
    with tempfile.TemporaryDirectory() as td:
        sections_gt = [
            {"kind": "intro", "start_s": 0.0, "end_s": 2.0, "energy": 0.2},
            {"kind": "verse", "start_s": 2.0, "end_s": 8.0, "energy": 0.4},
            {"kind": "chorus", "start_s": 8.0, "end_s": 16.0, "energy": 0.85},
        ]
        wav_path, made = _make_wav_fixture(td, 120.0, "A", "minor", 16.0, sections_gt)
        # 盲测：故意不传 brief.tempo 与 brief.key（只传可选段落边界）
        blind_brief = {
            "sections": [
                {"kind": "intro", "start_s": 0.0, "end_s": 2.0},
                {"kind": "verse", "start_s": 2.0, "end_s": 8.0},
                {"kind": "chorus", "start_s": 8.0, "end_s": 16.0},
            ]
        }
        doc = AU.analyze_external_audio(wav_path, brief=blind_brief, backend="pcm_wave")
        ok, why = SC.validate_score(doc)
        assert ok, "pcm_wave 产出未过 score 契约：%s" % why
        assert abs(doc["tempo"]["bpm"] - 120.0) <= 1.0, "120 BPM 盲测误差 > 1.0: %r" % doc["tempo"]["bpm"]

        one_frame_s = 1.0 / 24.0
        for b_gt in made["score"]["beats"]:
            nearest = min(doc["beats"], key=lambda x: abs(x - b_gt))
            assert abs(nearest - b_gt) <= one_frame_s, (
                "拍点偏差 > 1 帧(@24fps)：gt=%r nearest=%r" % (b_gt, nearest)
            )
        for db in doc["downbeats"]:
            assert SC.nearest_beat(doc["beats"], db) is not None, "downbeat 必须是 beats 子集"
        for db_gt in made["score"]["downbeats"]:
            nearest_db = min(doc["downbeats"], key=lambda x: abs(x - db_gt))
            assert abs(nearest_db - db_gt) <= one_frame_s, (
                "强拍偏差 > 1 帧(@24fps)：gt=%r nearest=%r" % (db_gt, nearest_db)
            )

        want_n = int(round(16.0 / AU.DEFAULT_HOP_S)) + 1
        assert len(doc["energy"]["values"]) == want_n
        by_kind = {s["kind"]: s["energy"] for s in doc["sections"]}
        assert by_kind["chorus"] > by_kind["verse"] > by_kind["intro"], by_kind
        assert doc["key"]["tonic"] == "A" and doc["key"]["mode"] == "minor", doc["key"]


def test_c2_pcm_wave_blind_96bpm_cmajor_and_no_brief():
    with tempfile.TemporaryDirectory() as td:
        sections_gt = [
            {"kind": "verse", "start_s": 0.0, "end_s": 5.0, "energy": 0.4},
            {"kind": "chorus", "start_s": 5.0, "end_s": 10.0, "energy": 0.85},
        ]
        wav_path, made = _make_wav_fixture(td, 96.0, "C", "major", 10.0, sections_gt)
        # 完全不传 brief（brief=None）
        doc = AU.analyze_external_audio(wav_path, brief=None, backend="pcm_wave")
        ok, why = SC.validate_score(doc)
        assert ok, "brief=None 盲测未过 score 契约：%s" % why
        assert abs(doc["tempo"]["bpm"] - 96.0) <= 1.0, "96 BPM 盲测误差 > 1.0: %r" % doc["tempo"]["bpm"]

        one_frame_s = 1.0 / 24.0
        for b_gt in made["score"]["beats"]:
            nearest = min(doc["beats"], key=lambda x: abs(x - b_gt))
            assert abs(nearest - b_gt) <= one_frame_s
        for db_gt in made["score"]["downbeats"]:
            nearest_db = min(doc["downbeats"], key=lambda x: abs(x - db_gt))
            assert abs(nearest_db - db_gt) <= one_frame_s
        assert doc["sections"][0]["kind"] == "instrumental"
        assert doc["key"]["tonic"] == "C" and doc["key"]["mode"] == "major", doc["key"]


# ══════════════════════════════════════════════════════ [D] 固化后零依赖重放过 E1
def test_d1_consolidate_and_zero_dep_replay_passes_e1():
    with tempfile.TemporaryDirectory() as td:
        sections_tl = [
            {"kind": "intro", "start_s": 0.0, "end_s": 2.0, "energy": 0.2,
             "shot_scale": "extreme_wide", "move_bias": 0.15},
            {"kind": "verse", "start_s": 2.0, "end_s": 8.0, "energy": 0.4,
             "shot_scale": "wide", "move_bias": 0.30},
            {"kind": "chorus", "start_s": 8.0, "end_s": 16.0, "energy": 0.85,
             "shot_scale": "medium", "move_bias": 0.70},
        ]
        wav_path, _ = _make_wav_fixture(td, 120.0, "A", "minor", 16.0, sections_tl)
        blind_brief = {"duration_s": 16.0, "fps": 24, "sections": sections_tl}
        score_raw = AU.analyze_external_audio(wav_path, brief=blind_brief, backend="pcm_wave")

        json_path = os.path.join(td, "consolidated_score.json")
        cons = AU.consolidate_score(score_raw, json_path)
        assert len(cons["digest"]) == 64 and len(cons["file_sha256"]) == 64
        assert cons["digest"] == MVC.digest("score", cons["score"])

        # 执行期屏蔽 librosa 与原音频依赖，仅从固化 JSON 重放
        old = sys.modules.get("librosa", ...)
        sys.modules["librosa"] = None
        try:
            reloaded = AU.load_consolidated_score(json_path)
            assert MVC.digest("score", reloaded) == cons["digest"]
            bind = TL.build_bind(blind_brief, reloaded)
            shots = TL.plan_shot_table(blind_brief, reloaded, bind)
            edit = TL.build_edit(blind_brief, shots, reloaded, bind)
            e1 = GE.check_e1({"edit": edit, "score": reloaded})
            assert e1["verdict"] == "ACCEPT", e1
        finally:
            if old is ...:
                sys.modules.pop("librosa", None)
            else:
                sys.modules["librosa"] = old


# ══════════════════════════════════════════════════════ [E] 全入口 fail-closed 边界门禁
def test_e1_fail_closed_on_nonfinite_bool_bad_meter_and_corrupt_files():
    with tempfile.TemporaryDirectory() as td:
        wav_path, _ = _make_wav_fixture(
            td, 120.0, "A", "minor", 4.0,
            [{"kind": "intro", "start_s": 0.0, "end_s": 4.0, "energy": 0.3}],
        )
        # 参数非法：NaN / ±inf / bool / 坏 meter / 坏 backend / 空 score_id
        for bad_hop in (float("nan"), float("inf"), float("-inf"), True, False, 0.0, 2.0):
            _raises((AU.AudioError, ValueError), AU.analyze_external_audio,
                    wav_path, backend="pcm_wave", energy_hop_s=bad_hop)
        for bad_meter in ((True, 4), (4, False), (0, 4), (4, 5), (4,), "4/4"):
            _raises((AU.AudioError, ValueError), AU.analyze_external_audio,
                    wav_path, backend="pcm_wave", meter=bad_meter)
        _raises(AU.AudioError, AU.analyze_external_audio, wav_path, backend="invalid_backend")
        _raises(AU.AudioError, AU.analyze_external_audio, wav_path, backend="pcm_wave", score_id="")
        _raises(AU.AudioError, AU.analyze_external_audio, os.path.join(td, "missing.wav"), backend="pcm_wave")

        # brief.sections 非有限 / bool
        for bad_val in (float("nan"), float("inf"), True):
            _raises((AU.AudioError, ValueError), AU.analyze_external_audio,
                    wav_path, backend="pcm_wave",
                    brief={"sections": [{"kind": "intro", "start_s": bad_val, "end_s": 4.0}]})

        # 空 WAV（0 帧）
        empty_wav = os.path.join(td, "empty.wav")
        with wave.open(empty_wav, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            wf.writeframes(b"")
        _raises(AU.AudioError, AU.analyze_external_audio, empty_wav, backend="pcm_wave")

        # 静音 WAV
        silent_wav = os.path.join(td, "silent.wav")
        with wave.open(silent_wav, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            wf.writeframes(b"\x00\x00" * 44100)
        _raises(AU.AudioError, AU.analyze_external_audio, silent_wav, backend="pcm_wave")

        # 损坏 WAV 头
        bad_wav = os.path.join(td, "corrupt.wav")
        with open(bad_wav, "wb") as f:
            f.write(b"NOT_A_RIFF_WAV_HEADER")
        _raises(AU.AudioError, AU.analyze_external_audio, bad_wav, backend="pcm_wave")

        # 固化 JSON 非有限 / 损坏
        nan_json = os.path.join(td, "nan.json")
        with open(nan_json, "w", encoding="utf-8") as f:
            f.write('{"schema": "wmmv.score/1", "x": NaN}')
        _raises((AU.AudioError, ValueError), AU.load_consolidated_score, nan_json)
        _raises(AU.AudioError, AU.load_consolidated_score, os.path.join(td, "nope.json"))
        _raises((AU.AudioError, ValueError), AU.consolidate_score, {"schema": "wrong"}, os.path.join(td, "bad.json"))


# ══════════════════════════════════════════════════════ [F] 既有 make/synthesize/analyze 不变
def test_f1_existing_make_synthesize_analyze_unchanged():
    with tempfile.TemporaryDirectory() as td:
        brief = {
            "duration_s": 2.0,
            "fps": 24,
            "tempo": {"bpm": 120.0, "meter": [4, 4], "offset_s": 0.0},
            "key": {"tonic": "A", "mode": "minor"},
            "sections": [
                {"kind": "intro", "start_s": 0.0, "end_s": 1.0, "energy": 0.3},
                {"kind": "chorus", "start_s": 1.0, "end_s": 2.0, "energy": 0.9},
            ],
        }
        r1 = AU.make(brief, os.path.join(td, "m1"))
        r2 = AU.make(brief, os.path.join(td, "m2"))
        assert r1["wav"]["sha256"] == r2["wav"]["sha256"]
        synth_path = os.path.join(td, "synth_dir", "music.wav")
        w_synth = AU.synthesize(r1["score"], synth_path)
        assert w_synth["sha256"] == r1["wav"]["sha256"]
        re_an = AU.analyze(synth_path, brief)
        assert MVC.digest("score", re_an) == MVC.digest("score", r1["score"])


# ══════════════════════════════════════════════════════ --mutate 变异自证
def _run_mutations() -> int:
    """至少 3 条定向变异点红自证（进程内受控替换，不污染仓库文件）。"""
    global PYPROJECT_PATH
    results = []

    # M1：从 pyproject.toml 删掉 film-audio ⇒ test_a1 必红
    orig_pyp = PYPROJECT_PATH
    with tempfile.TemporaryDirectory() as td:
        mut_pyp = os.path.join(td, "pyproject.toml")
        src_pyp = open(orig_pyp, encoding="utf-8").read()
        with open(mut_pyp, "w", encoding="utf-8") as fh:
            fh.write("\n".join(l for l in src_pyp.splitlines() if not l.startswith("film-audio =")))
        PYPROJECT_PATH = mut_pyp
        try:
            test_a1_pyproject_declares_film_audio_sub_extra_not_in_full()
            results.append(("M1_drop_film_audio_extra", False, "未点红"))
        except AssertionError:
            results.append(("M1_drop_film_audio_extra", True, "如期点红"))
        finally:
            PYPROJECT_PATH = orig_pyp

    # M2：把缺失 librosa 的 fail-closed 替换成静默返回伪乐谱 ⇒ test_b1 必红
    orig_librosa_fn = AU._analyze_via_librosa
    AU._analyze_via_librosa = lambda path, brief, meter, hop_s, sid: {"schema": "wmmv.score/1"}
    try:
        test_b1_missing_librosa_fails_closed()
        results.append(("M2_silent_fallback_missing_librosa", False, "未点红"))
    except AssertionError:
        results.append(("M2_silent_fallback_missing_librosa", True, "如期点红"))
    finally:
        AU._analyze_via_librosa = orig_librosa_fn

    # M3：把 bpm 盲测估计故意偏移 +5.0 BPM ⇒ test_c2 必红
    orig_detect = AU._detect_beats_pcm
    def _bad_bpm_detect(mono, sr, meter):
        bpm, off, beats, dbs = orig_detect(mono, sr, meter)
        return bpm + 5.0, off, beats, dbs
    AU._detect_beats_pcm = _bad_bpm_detect
    try:
        test_c2_pcm_wave_blind_96bpm_cmajor_and_no_brief()
        results.append(("M3_corrupt_blind_bpm_detection", False, "未点红"))
    except AssertionError:
        results.append(("M3_corrupt_blind_bpm_detection", True, "如期点红"))
    finally:
        AU._detect_beats_pcm = orig_detect

    # M4：放行 bool 传入 energy_hop_s（Python 中 True==1.0 落在 [1e-4, 1.0] 内）⇒ test_e1 必红
    orig_hop_fn = AU._validate_hop_s
    def _allow_bool_hop(v):
        import math
        fv = float(v)
        if not math.isfinite(fv) or not (1e-4 <= fv <= 1.0):
            raise AU.AudioError("out of range")
        return fv
    AU._validate_hop_s = _allow_bool_hop
    try:
        test_e1_fail_closed_on_nonfinite_bool_bad_meter_and_corrupt_files()
        results.append(("M4_bypass_bool_hop_guard", False, "未点红"))
    except AssertionError:
        results.append(("M4_bypass_bool_hop_guard", True, "如期点红"))
    finally:
        AU._validate_hop_s = orig_hop_fn

    for name, ok, note in results:
        print(f"MUTATION {name}: {'RED_OK' if ok else 'FAILED_TO_TRIGGER'} ({note})")
    all_ok = all(ok for _, ok, _ in results)
    print("MUTATE_OVERALL=%s (%d/%d)" % ("PASS" if all_ok else "FAIL",
                                         sum(1 for _, ok, _ in results if ok), len(results)))
    return 0 if all_ok else 1


if __name__ == "__main__":
    if "--mutate" in sys.argv[1:]:
        sys.exit(_run_mutations())
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for fn in tests:
        try:
            fn()
            passed += 1
            print("PASS", fn.__name__)
        except Exception as exc:                               # noqa: BLE001
            failed += 1
            print("FAIL", fn.__name__, type(exc).__name__, exc)
    print("SUMMARY pass=%d fail=%d" % (passed, failed))
    sys.exit(1 if failed else 0)
