# -*- coding: utf-8 -*-
"""#802 守卫：导出 JSON 落盘剥除 BiDi 控制/零宽字符——渲染无反转（lingshu/core/core.py）。

缺陷定性（lingshu issue #802，SEC·P2 · CWE-176）：
  BiDi 控制字符（U+202A-U+202E 嵌入/覆写、U+2066-U+2069 隔离、U+200E/U+200F
  方向标记）与零宽族（U+200B-U+200D / U+FEFF / U+2060）视觉不可见但被 BiDi
  渲染器解释——RTLO（U+202E）使后续文本**反向显示**：`file\\u202etxt` 落盘后
  显示成 `txt.exe`。`json.dump`（`ensure_ascii=False`）只转义 ASCII 控制符，
  非 ASCII 的 U+202E 原样落盘 ⇒ 导出 JSON 在下游渲染面（编辑器/终端/web UI/
  日志查看器）存在反转文本（spoofing 面）＋被遮蔽文本的人工审计绕过。

修法（本件，最小 · 仅导出序列化面）：`export_all` / `export_trajectory`
  **写出前**对整棵导出树过 `_sanitize_export_obj` → `_strip_export_controls`
  （剥除；与 #803/PR #807 判同键剥除同款风格）。落库正文与 `import_all`
  回读面不动（判据只管导出面）。

改前读数（基线 40fceb6 · Windows 11 · Python 3.12 · 本 worktree 实测）：
  add_perception('安全文件: file\\u202etxt')            → export_all
  落盘 raw text 含 U+202E: True；parsed（nodes.content）含 U+202E: True；
  LRO(U+202D)/RLI(U+2066)/ZWJ(U+200D) 同样原样保留；import_all 回读不炸
  （integrity_ok=True）但反转文本随备份继续传播。
  修后（本件）：以上四类字符在落盘 JSON 的 raw 文本与解析树中均不再出现，
  正文长度相应变短（RTLO 载荷 14→13），CJK/emoji/真 RTL 文字逐字节保留，
  回读与再导出（第二次 export）同样干净。

断言组（回退修复＝`_strip_export_controls` 改恒等返回 ⇒ A/B/D/E 必红）：
  A RTLO 载荷（issue 原复现 + `important\\u202esecurity\\u202enotice`）：落盘
    raw 文本与**解析后**的任意字符串值/键均无 U+202E；正文其余部分保留
  B 全控制集：LRE/RLE/PDF/LRO/RTO + LRM/RLM + LRI/RLI/FSI/PDI +
    ZWSP/ZWNJ/ZWJ/WJ/BOM ⇒ 落盘 raw 文本一个不剩
  C 不误伤：CJK / emoji / 真正的 RTL 文字（阿拉伯/希伯来）导出后逐字节保留
  D 往返一致性（剥除口径）：export → import_all 回读成功（integrity_ok）、
    回读库再 export ⇒ 落盘同样干净（反转文本不随备份二传）
  E `export_trajectory` 落盘分支同款剥除（cycle_record_full 载荷含 RTLO）

运行（仓根）：python -X utf8 -m pytest tests/test_issue802_export_bidi_stripped.py -q --no-header
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.core.core import SpacetimeMemoryEngine  # noqa: E402

# ---- 载荷字符（名字钉 Unicode 码点，防编辑器/终端再渲染反转）----
RTLO = "‮"  # U+202E（issue 原复现字符）
LRO = "‭"   # U+202D
LRE = "‪"   # U+202A
RLE = "‫"   # U+202B
PDF = "‬"   # U+202C
LRM = "‎"   # U+200E
RLM = "‏"   # U+200F
LRI = "⁦"   # U+2066
RLI = "⁧"   # U+2067
FSI = "⁨"   # U+2068
PDI = "⁩"   # U+2069
ZWSP = "​"  # U+200B
ZWNJ = "‌"  # U+200C
ZWJ = "‍"   # U+200D
WJ = "⁠"    # U+2060
BOM = "﻿"   # U+FEFF

ALL_CONTROLS = [RTLO, LRO, LRE, RLE, PDF, LRM, RLM,
                LRI, RLI, FSI, PDI, ZWSP, ZWNJ, ZWJ, WJ, BOM]


def _engine():
    return SpacetimeMemoryEngine(":memory:")


def _export_raw(eng, out):
    eng.export_all(out)
    with open(out, encoding="utf-8") as f:
        return f.read()


def _all_strings(obj):
    """解析树里所有字符串（含 dict 键）拉平——「解析后也不含」的判据面。"""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str):
                yield k
            yield from _all_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _all_strings(v)


# ---------------- A：RTLO 载荷——raw 与解析树均无 U+202E ----------------

def test_a_rtlo_载荷_落盘与解析树均剥除():
    td = tempfile.mkdtemp()
    eng = _engine()
    eng.add_perception("安全文件: file" + RTLO + "txt", skip_dedup=True)
    eng.add_perception("important" + RTLO + "security" + RTLO + "notice",
                       skip_dedup=True)
    out = os.path.join(td, "export.json")
    raw = _export_raw(eng, out)
    assert RTLO not in raw, "落盘 raw 文本仍含裸 U+202E——反转渲染面未除"
    parsed = json.loads(raw)
    strs = list(_all_strings(parsed))
    assert all(RTLO not in s for s in strs), "解析树仍有字符串含 U+202E"
    contents = [n.get("content", "") for n in parsed.get("nodes", [])]
    assert any("安全文件: filetxt" == c for c in contents), \
        "剥除应只去掉控制符，正文其余部分（CJK/ASCII）逐字保留"
    assert any(c == "importantsecuritynotice" for c in contents)


# ---------------- B：全控制集一个不剩 ----------------

def test_b_全控制集_落盘raw一个不剩():
    td = tempfile.mkdtemp()
    eng = _engine()
    eng.add_perception("".join("x" + c for c in ALL_CONTROLS), skip_dedup=True)
    out = os.path.join(td, "export.json")
    raw = _export_raw(eng, out)
    for c in ALL_CONTROLS:
        assert c not in raw, f"控制符 U+{ord(c):04X} 仍原样落盘"
    assert "x" * len(ALL_CONTROLS) in json.dumps(  # 剩下的 x 一个不少
        [n.get("content") for n in json.loads(raw).get("nodes", [])],
        ensure_ascii=False)


# ---------------- C：不误伤——CJK/emoji/真 RTL 文字逐字节保留 ----------------

def test_c_正常文本不受影响():
    td = tempfile.mkdtemp()
    eng = _engine()
    normal = "中文内容保持原样ＡＢＣ 123 🚀🧠 שלום سلام"
    eng.add_perception(normal, skip_dedup=True)
    out = os.path.join(td, "export.json")
    raw = _export_raw(eng, out)
    parsed = json.loads(raw)
    contents = [n.get("content", "") for n in parsed.get("nodes", [])]
    assert normal in contents, "正常 CJK/emoji/阿拉伯/希伯来文字必须逐字节保留"


# ---------------- D：往返一致性（剥除口径）----------------

def test_d_导入回读与再导出同样干净():
    td = tempfile.mkdtemp()
    eng = _engine()
    eng.add_perception("安全文件: file" + RTLO + "txt", skip_dedup=True)
    out = os.path.join(td, "export.json")
    _export_raw(eng, out)
    # 回读：剥除后的正文长度变短，import_all 不得因此炸
    eng2 = SpacetimeMemoryEngine(os.path.join(td, "t2.db"))
    r = eng2.import_all(out)
    assert r.get("integrity_ok") is True
    assert r["imported"]["nodes"] == 1
    # 二传：回读库再导出 ⇒ 落盘同样干净（反转文本不随备份链传播）
    out2 = os.path.join(td, "export2.json")
    raw2 = _export_raw(eng2, out2)
    assert RTLO not in raw2
    # 回读正文的可读部分保留
    contents = [n.get("content", "") for n in json.loads(raw2).get("nodes", [])]
    assert any("安全文件: filetxt" == c for c in contents)


# ---------------- E：export_trajectory 落盘分支同款剥除 ----------------

def test_e_trajectory_落盘分支剥除_rtlo():
    td = tempfile.mkdtemp()
    eng = _engine()
    payload = json.dumps(
        {"cycle_id": "c1", "note": "audit" + RTLO + "me",
         RTLO + "key": "反向键"}, ensure_ascii=False)
    eng.add_perception(payload, tags=["cycle:c1", "cycle_record_full"],
                       skip_dedup=True)
    out = os.path.join(td, "traj.json")
    eng.export_trajectory("c1", output_path=out)
    raw = open(out, encoding="utf-8").read()
    assert RTLO not in raw, "export_trajectory 落盘仍含裸 U+202E"
    parsed = json.loads(raw)
    assert all(RTLO not in s for s in _all_strings(parsed))
    # 值/键的其余部分保留（载荷 JSON 经剥除后仍可解析、内容可读）
    flat = " ".join(_all_strings(parsed))
    assert "auditme" in flat
