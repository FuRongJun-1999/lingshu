# -*- coding: utf-8 -*-
"""issue #825 守卫：wmmv.text 的 CJK limits 判据与可读性硬门必须落在正文上。

病象（修前）：
  ① script="cjk" 只查顶层「limits 键是否存在」——{} / null / 只给 1 项都能
     过闸，随后被 LATIN_LIMITS 兜成拉丁口径（模块头明说 CJK 不给默认值）；
  ② lines 可独立于 text 给出，三项可读性硬门读 lines：47 字符的 text 配上
     lines=["ok"] 即 ACCEPT，落库正文 47 字符、真实 CPS=376，而 cps_of()
     与 text_summary() 都报 16.0。

修复语义：CJK 必须显式给出三项上限；lines 若给出必须是 text 的等价分片；
cps_of 按上屏正文 text 的换行拆分合计。

运行：python -X utf8 tests/test_issue825_text_gates.py
自证：python -X utf8 tests/test_issue825_text_gates.py --mutate
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "bridges" / "blender"))

from wm_bridge.factory.film import text as TX  # noqa: E402

BASE = {"schema": "wmmv.text/1", "id": "t1",
        "frame_base": {"fps": 24, "frame_start": 0, "frame_end": 95}}
CJK = [{"kind": "lyric", "start_frame": 0, "end_frame": 95,
        "text": "一二三四五六七八九十", "script": "cjk"}]
LONG = "This line is far too long to read in two frames"   # 47 字符 > 42

_CHECKS = []
FAILS = []


def _check(name, cond, detail=""):
    if name not in _CHECKS:
        _CHECKS.append(name)
    if cond:
        print("  PASS " + name)
    else:
        print("  FAIL " + name + "  " + str(detail)[:240])
        FAILS.append(name)


def _validate(**kw):
    d = dict(BASE)
    d.update(kw)
    return TX.validate_text(d)


def _assertions():
    print("== A 组：CJK limits 必须三项显式 ==")
    ok, why = _validate(cues=CJK)
    _check("A01 前提：CJK 无 limits 键 ⇒ REJECT（既有拒绝面）",
           ok is False and "limits" in why, why)

    ok, why = _validate(cues=CJK, limits={})
    _check("A02 ★行：limits={} ⇒ REJECT（空对象不算显式声明）",
           ok is False and "未显式给出" in why, why)

    ok, why = _validate(cues=CJK, limits=None)
    _check("A03 ★行：limits=None ⇒ REJECT", ok is False and "未显式给出" in why, why)

    ok, why = _validate(cues=CJK, limits={"max_cps": 20})
    _check("A04 ★行：三项只给 1 项 ⇒ REJECT 且点名缺项",
           ok is False and "max_chars_per_line" in why and "max_lines" in why, why)

    ok, why = _validate(cues=CJK, limits={"max_cps": 8,
                                          "max_chars_per_line": 16,
                                          "max_lines": 2})
    norm = TX.normalize_text(dict(BASE, cues=CJK, limits={"max_cps": 8,
                                                          "max_chars_per_line": 16,
                                                          "max_lines": 2}))
    _check("A05 对照：三项全给 ⇒ ACCEPT 且 limits 为声明值",
           ok is True and norm["limits"] == {"max_cps": 8.0,
                                             "max_chars_per_line": 16,
                                             "max_lines": 2},
           (ok, norm.get("limits") if ok else why))

    print("== B 组：可读性硬门读 text ==")
    ok, why = _validate(cues=[{"kind": "lyric", "start_frame": 0,
                               "end_frame": 2, "text": LONG}])
    _check("B01 前提：47 字符 text 单独 ⇒ REJECT（既有行宽判据）",
           ok is False and "42" in why, why)

    ok, why = _validate(cues=[{"kind": "lyric", "start_frame": 0,
                               "end_frame": 2, "text": LONG,
                               "lines": ["ok"]}])
    _check("B02 ★行：text 47 字符 + lines=['ok'] ⇒ REJECT（lines 不得独立）",
           ok is False and "不一致" in why, why)

    ok, why = _validate(cues=[{"kind": "lyric", "start_frame": 0,
                               "end_frame": 95, "text": "aa\nbb\ncc"}])
    _check("B03 前提：3 行 text 单独 ⇒ REJECT（既有 max_lines）",
           ok is False and "3 行" in why, why)

    ok, why = _validate(cues=[{"kind": "lyric", "start_frame": 0,
                               "end_frame": 95, "text": "aa\nbb\ncc",
                               "lines": ["aa", "bb"]}])
    _check("B04 ★行：3 行 text + lines 只给 2 行 ⇒ REJECT（行数不可被 lines 掩盖）",
           ok is False and "不一致" in why, why)

    ok, why = _validate(cues=[{"kind": "lyric", "start_frame": 0,
                               "end_frame": 95, "text": "aa\nbb",
                               "lines": ["aa", "bb"]}])
    _check("B05 对照：lines 与 text 逐行一致 ⇒ ACCEPT",
           ok is True, why)

    cue = {"text": LONG, "lines": ["ok"], "start_frame": 0, "end_frame": 2}
    v = TX.cps_of(cue, 24.0)
    _check("B06 ★行：cps_of 按 text 合计（47 字符/3 帧 = 376.0）",
           abs(v - 376.0) < 1e-6, v)

    d = dict(BASE, cues=[{"kind": "lyric", "start_frame": 0, "end_frame": 95,
                          "text": "x" * 40}])
    summ = TX.text_summary(d)
    _check("B07 ★行：text_summary 的 max_cps 与 cps_of 同口径（40 字符/4 秒 = 10.0）",
           abs(summ["max_cps"] - 10.0) < 1e-6, summ["max_cps"])

    print("== E 组：源码钉子 ==")

    def _src(rel):
        try:
            return (_REPO / rel).read_text(encoding="utf-8")
        except OSError:
            return ""

    src = _src("bridges/blender/wm_bridge/factory/film/text.py")
    _check("E01 ★钉子：CJK 按三项缺项判据", "missing = [k for k in _KEYS_LIMITS" in src)
    _check("E02 ★钉子：lines 与 text 一致性检查", "if clean != body_lines:" in src)
    _check("E03 ★钉子：cps_of 读 text", 'body = cue.get("text")' in src)


def _run_assertions():
    global FAILS
    FAILS = []
    del _CHECKS[:]
    _assertions()
    print("通过 %d / 失败 %d" % (len(_CHECKS) - len(FAILS), len(FAILS)))
    return len(FAILS)


_ARMS = {
    "M1": (
        "bridges/blender/wm_bridge/factory/film/text.py",
        [("            if missing:", "            if False:")],
        {"A01", "A02", "A03", "A04"},
    ),
    "M2": (
        "bridges/blender/wm_bridge/factory/film/text.py",
        [("            if clean != body_lines:", "            if False:")],
        {"B02", "B04", "E02"},
    ),
    "M3": (
        "bridges/blender/wm_bridge/factory/film/text.py",
        [('    body = cue.get("text")', "    body = None")],
        {"B06", "E03"},
    ),
}


def _subprocess_fail_names():
    for cache in _REPO.rglob("__pycache__"):
        for pyc in cache.glob("text.*.pyc"):
            pyc.unlink(missing_ok=True)
    r = subprocess.run([sys.executable, "-X", "utf8", __file__, "--assert-only"],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=str(_REPO), timeout=900)
    names = set(re.findall(r"(?m)^\s*FAIL\s+([A-Z]\d+)", r.stdout or ""))
    return names, (r.stdout or "") + (r.stderr or "")


def _mutate(argv):
    want = ""
    for a in argv:
        if a == "--mutate":
            want = want or "ALL"
        elif a.startswith("--mutate="):
            want = a.split("=", 1)[1]
    if not want:
        return 0
    ids = sorted(_ARMS) if want == "ALL" else [want]
    red = 0
    for mid in ids:
        if mid not in _ARMS:
            print("[ANCHOR-MISS] %s：未登记该变异臂" % mid)
            return 2
        rel, edits, expect = _ARMS[mid]
        target = _REPO / rel
        orig = target.read_bytes()
        try:
            text = orig.decode("utf-8")
            nl = "\r\n" if "\r\n" in text else "\n"
            norm = text.replace("\r\n", "\n")
            for old, new in edits:
                cnt = norm.count(old)
                if cnt != 1:
                    print("[ANCHOR-MISS] %s：锚点命中 %d 次（期望 1）" % (mid, cnt))
                    return 2
                norm = norm.replace(old, new)
            target.write_bytes(norm.replace("\n", nl).encode("utf-8"))
            actual, out = _subprocess_fail_names()
            if actual == expect:
                print("[MUTATE-OK]   %s：指定断言转红 %s" % (mid, sorted(actual)))
            else:
                print("[MUTATE-FAIL] %s：期望红 %s，实际红 %s"
                      % (mid, sorted(expect), sorted(actual)))
                for line in out.splitlines():
                    if line.strip().startswith("FAIL"):
                        print("    " + line)
                red += 1
        finally:
            target.write_bytes(orig)
    print("反证结果：%d 条变异，%s" % (len(ids), "全部恰好命中" if red == 0 else "%d 条未命中" % red))
    return 1 if red else 0


if __name__ == "__main__":
    _f = _run_assertions()
    if "--assert-only" in sys.argv:
        sys.exit(1 if _f else 0)
    _m = _mutate(sys.argv[1:])
    sys.exit(1 if (_f or _m) else 0)
