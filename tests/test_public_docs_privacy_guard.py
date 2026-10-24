# -*- coding: utf-8 -*-
"""test_public_docs_privacy_guard · 公开面**文档面**入库隐私闸
============================================================================
背景（2026-10-17 取证）：
    仓内已有「双清单审计」约定（见 docs/intake/*/AUDIT.md）——
      内容政策：凭据/令牌、本机绝对路径、邮箱、本地 IP
      隐私面  ：记忆数据、私有模块名、私有资产引用
    但该约定只对**导出批**逐批执行（docs/intake/** 与随批新增面），
    文档面从未纳入机械扫描 —— 于是私有仓名经设计稿/理论稿流入公开面，
    且仓内**没有任何**能拦住它的入库闸（全仓无 POLICY_DENY / policy_scan 实现）。
    本件即补该闸。

扫描面（v2 放宽）：
    docs/** 递归，扩展名 .md 与 .json。首版只扫 docs/plans/**；
    放宽后 docs/plans 面保持 0 命中，其余文档面以**逐条豁免表**记录既有项。

判据（五类，逐行）：
    A 凭据/令牌  ：sk-… / ghp_… / tk_<hex> / mdcg1. / (api_key|secret|password|token) 赋值
    B 本机绝对路径：盘符 + 分隔符、UNC、POSIX 家目录与挂载根
    C 邮箱
    D 本地 IP
    E 私有面标识 ：私有仓名及其全部环境变量/常量前缀

豁免机制（本件的核心设计）：
    _EXEMPT 为 (路径 glob, 类别, 行内须含子串, 理由) 四元组表。
    仅抑制**同时满足**三条件的行 —— 于是：
      · 豁免是**逐条、可复核**的（每行都有人写的理由），不是整文件白名单；
      · **同文件的新增泄漏仍会被判红**（新行不含豁免短语 ⇒ 不被抑制）；
      · 非陈化断言（test_exemptions_are_not_stale）要求每条豁免**至少命中一次**，
        否则判红 —— 防豁免表随内容演进而悄悄失效、变成无声的大洞。
    口径对齐 docs/intake/*/AUDIT.md 既有约定：「命中项须逐条复核：真命中=修，
    误报=记录理由」。既有 32 处命中的分类见下表注释。

边界（如实，防本件被读成「全仓已干净」）：
    · 扫描面**仅** docs/**（文档面）。同一私有标识在**源码面是公开运行时接口**
      （lingshu/core/core.py 以该环境变量名 os.environ.get 读取），仓内 60+ 文件含它；
      本件**不**对其判红 —— 该面是否收口是独立议题（改名即破坏公开接口兼容性）。
    · 本件自身含该标识字面量（判据表 E 类与豁免表）。若日后放宽扫描面至仓根，
      须同时排除本件路径，否则自触。
    · 本件只做**机械**判据；「私有模块名」这类需人判的项仍走 AUDIT.md 人工复核。

量具自检（同族纪律，防恒绿）：
    · 扫描器必须能从**合成文本**读出每一类命中 —— test_scanner_detects_known_positives；
    · 对合法文本不得误杀 —— test_no_false_positive_on_benign_text；
    · 扫描面不得为空（空集恒绿是坏扫描器掩盖的假绿）—— test_scan_surface_is_nonempty；
    · 豁免表不得陈化 / 形状不得残缺 —— test_exemptions_are_not_stale / _are_wellformed；
    · 本件自身必须可被 pytest 收集（否则落在 gate 盲区）—— test_guard_itself_is_collectable。

运行：pytest tests/test_public_docs_privacy_guard.py -q
      （或 python -X utf8 tests/test_public_docs_privacy_guard.py 直接打报告）
"""
from __future__ import annotations

import fnmatch
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAN_ROOT = os.path.join(REPO, "docs")
SCAN_EXTS = (".md", ".json")

#: A 凭据/令牌 —— 口径逐字对齐 docs/intake/brain-store-v0.1/AUDIT.md 判据表，另补 ghp_。
_CRED = (
    ("sk_token", re.compile(r"sk-[A-Za-z0-9_-]{16,}")),
    ("gh_token", re.compile(r"ghp_[A-Za-z0-9]{20,}")),
    ("tk_token", re.compile(r"tk_[0-9a-fA-F]{16,}")),
    ("mdcg_token", re.compile(r"mdcg1\.[A-Za-z0-9]{8,}")),
    ("kv_secret", re.compile(r"(?i)(?:api[_-]?key|secret|password|passwd|token)\s*[=:]\s*\S")),
)

#: B 本机绝对路径 —— 盘符腿带左界，防 class _Block:\n 这类伪影（先例见 intake AUDIT）。
_PATH = (
    ("win_drive", re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]")),
    ("unc", re.compile(r"\\\\[A-Za-z0-9._-]+\\")),
    ("posix_home", re.compile(r"(?:/home|/Users|/root|/mnt)/[A-Za-z0-9._-]+/")),
)

#: C 邮箱
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

#: D 本地 IP —— 各私有段**段数不同**（10 是 1 段前缀、192.168 与 172.x 是 2 段），
#: 故分开写；一律要求**完整地址**，避免把 10.0.0 / 1.12.1 这类版本号误判。
_IP = re.compile(
    r"(?<![\d.])(?:"
    r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
    r"|192\.168\.\d{1,3}\.\d{1,3}"
    r"|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}"
    r"|127\.0\.0\.1"
    r")(?![\d.])"
)

#: E 私有面标识 —— 私有仓名及其全部环境变量/常量前缀。
_PRIVATE_NAME = re.compile(r"(?<![A-Za-z0-9_])AEIS[A-Za-z0-9_]*")

CATEGORIES = ("凭据/令牌", "本机绝对路径", "邮箱", "本地IP", "私有面标识")

_CATEGORIES = (
    ("凭据/令牌", _CRED),
    ("本机绝对路径", _PATH),
)

# ------------------------------------------------------------------ 豁免表
# 四元组：(路径 glob（相对 docs/，POSIX 分隔）, 类别, 行内须含子串, 理由)
# 逐条复核结论（2026-10-17，共 38 处命中：误报 6 / 有意公开 26 / 冻结历史 6；
#   2026-10-23 复核：architecture.md「身体侧真源（AEIS）」已脱敏 ⇒ 该条豁免陈化已删，现 37 处 / 有意公开 25）：
#   误报 6   = 服务回环地址 2（structure-events.md 配置示例，先例见 architecture.md
#              引入登记「CDN 资源 URL / npm 包名 / 服务回环地址」判 CLEAN）
#              + 审计判据示意表自指 3（AUDIT.md 把判据正则本身写在表里）
#              + 单字符盘符伪影 1（f://enum；intake AUDIT 已记录为误报）
#   有意公开 26 = 导出来源标注 / 冻结规格 schema 字段 / 理论稿原文照录
#   冻结历史 6 = docs/plans 面原文照录（裁定「已有的历史不再修改」，回退前进式脱敏后按历史保留）
_EXEMPT = (
    # ---- 误报：服务回环地址（配置示例）
    ("structure-events.md", "本地IP", "127.0.0.1",
     "误报：结构事件服务的回环地址配置示例，非本机对外地址；先例见 architecture.md 引入登记（服务回环地址判 CLEAN）"),
    # ---- 误报：审计判据示意表自指（闸扫到自己的判据表）
    ("intake/*/AUDIT.md", "凭据/令牌", "mdcg1.",
     "误报：AUDIT.md 判据示意表把正则样例本身写进表里，非真实凭据"),
    ("intake/*/AUDIT.md", "本机绝对路径", "盘符 + 分隔符 + 目录名",
     "误报：同上，示意占位符 X:\\dir / X:/dir 是判据描述不是真实路径"),
    ("intake/*/AUDIT.md", "本地IP", "10.x / 192.168.x",
     "误报：同上，判据描述里的内网段示意"),
    # ---- 误报：单字符盘符 + 非路径串（intake AUDIT 已记录同族）
    ("intake/*/README.md", "本机绝对路径", "f://enum",
     "误报：单字符盘符 + 非路径串（误报复核先例），见 intake AUDIT 记录"),
    # ---- 有意公开：导出来源标注（architecture.md 引入登记表逐批登记）
    ("intake/*/README.md", "私有面标识", "身体侧真源（AEIS）",
     "有意公开：导出来源标注，architecture.md 引入登记表已逐批登记（该批双清单判 CLEAN）"),
    ("intake/*/README.md", "私有面标识", "tools/public_export/smoke.py",
     "有意公开：上游工具路径标注，README 同句明示「该命令需上游工具，本仓内不可执行」"),
    ("intake/*/README.md", "私有面标识", "AEIS 侧字节数更大",
     "有意公开：否证记录（「上游过时」前提被实测否证），属审计结论文本"),
    ("intake/*/README.md", "私有面标识", "上游 AEIS 同跑同红",
     "有意公开：上游对照复现记录，属审计结论文本"),
    # ---- 有意公开：冻结规格件（改字段=动冻结契约）
    ("intake/*/_export_manifest.json", "私有面标识", "\"aeis_root\": \"AEIS\"",
     "有意公开：导出清单**冻结规格的 schema 字段**；tests/test_manifest_integrity.py 断言 18-20 把清单条目数冻结在 64，改字段即动冻结契约"),
    ("intake/*/_export_manifest.json", "私有面标识", "上游 AEIS 为真源",
     "有意公开：清单 note 的来源说明，属登记文本"),
    ("intake/*/_export_manifest.json", "私有面标识", "非 AEIS 导出件",
     "有意公开：清单 origin 原生来源标记，受 test_manifest_integrity.py 断言 18 复核"),
    ("intake/*/_export_manifest.json", "私有面标识", "『AEIS 侧字节数更大",
     "有意公开：清单 note 的否证记录，属审计结论文本"),
    # ---- 有意公开：理论稿原文照录（architecture.md 登记「六份原文照录零改动」）
    ("theory/*.md", "私有面标识", "灵枢（AEIS）",
     "有意公开：理论稿题注原文照录，architecture.md 引入登记「六份原文照录零改动」"),
    ("theory/*.md", "私有面标识", "与 AEIS 实现",
     "有意公开：理论稿定位句原文照录（同上）"),
    ("theory/*.md", "私有面标识", "AEIS 私有",
     "有意公开：理论稿立场/范围声明原文照录（同上）"),
    ("theory/*.md", "私有面标识", "工程载体（AEIS）",
     "有意公开：理论稿表头原文照录（同上）"),
    ("theory/*.md", "私有面标识", "AEIS/aeis/",
     "有意公开：理论稿资产位置标注原文照录（同上）"),
    # ---- 有意公开：docs 根级登记件
    ("world_process_usage.md", "私有面标识", "非 AEIS 导出件",
     "有意公开：来源标注（说明本仓原生开发），与清单 origin 标记同源"),
    ("world_process_usage.md", "私有面标识", "是 AEIS",
     "有意公开：src 字段语义说明（src 仅用于导出件），属接口文档"),
    # ---- 冻结历史：docs/plans 面原文照录（裁定 2026-10-17：已有的历史不再修改，回退前进式脱敏）
    ("plans/待裁清单_v0.1.md", "私有面标识", "AEIS_DB",
     "冻结历史：该行记录私域仓配置标识，属历史原文照录，按裁定不改写"),
    ("plans/待裁清单_v0.1.md", "私有面标识", "私域 `AEIS`",
     "冻结历史：该行记录私域仓已推改动清单，属历史原文照录"),
    ("plans/治理与写入边界_修复设计_v0.1.md", "私有面标识", "私域仓 `AEIS`",
     "冻结历史：取证对象标注原文照录，属历史原文不改写"),
    ("plans/治理与写入边界_修复设计_v0.1.md", "私有面标识", "AEIS_DESIGNER_KEY",
     "冻结历史：该标识符本就是公开源码接口名（core.py 经 os.environ 读取），遮蔽零收益，按历史原文保留"),
)


# ------------------------------------------------------------------ 扫描

def _scan_files(root):
    """递归列出 root 下 .md/.json（跳过 __pycache__）；空集须由调用方显式判红。"""
    out = []
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d != "__pycache__"]
        for f in sorted(fn):
            if f.lower().endswith(SCAN_EXTS):
                out.append(os.path.join(dp, f))
    return sorted(out)


def _read(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def _violations_in_text(text):
    """单份文本 -> [(类别, 行号, 行内容)]（**不含**豁免判定）。量具自检直接喂合成文本。"""
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        cat = None
        for name, pats in _CATEGORIES:
            if any(p.search(line) for _n, p in pats):
                cat = name
                break
        if cat is None:
            if _EMAIL.search(line):
                cat = "邮箱"
            elif _IP.search(line):
                cat = "本地IP"
            elif _PRIVATE_NAME.search(line):
                cat = "私有面标识"
        if cat is not None:
            out.append((cat, i, line.strip()))
    return out


def _match_exempt(rel_posix, cat, txt):
    """命中豁免表 -> 返回该条豁免四元组；否则 None。"""
    for entry in _EXEMPT:
        glob, e_cat, needle = entry[0], entry[1], entry[2]
        if cat == e_cat and needle in txt and fnmatch.fnmatch(rel_posix, glob):
            return entry
    return None


def _scan(root=None):
    """扫描面全量 -> (未豁免命中, 豁免命中)。"""
    root = SCAN_ROOT if root is None else root
    hits, exempted = [], []
    for p in _scan_files(root):
        rel = os.path.relpath(p, root).replace(os.sep, "/")
        for cat, ln, txt in _violations_in_text(_read(p)):
            ent = _match_exempt(rel, cat, txt)
            if ent is None:
                hits.append((rel, ln, cat, txt))
            else:
                exempted.append((rel, ln, cat, txt, ent))
    return hits, exempted


def _violations(root=None):
    """仅未豁免命中（对外主判据）。"""
    return _scan(root)[0]


def _exempt_usage(root=None):
    """每条豁免的命中次数（按表内顺序）。非陈化断言读它。"""
    _hits, exempted = _scan(root)
    used = {i: 0 for i in range(len(_EXEMPT))}
    for _rel, _ln, _cat, _txt, ent in exempted:
        used[_EXEMPT.index(ent)] += 1
    return used


def _report(root=None):
    hits, exempted = _scan(root)
    lines = ["docs/** 文档面隐私闸：未豁免命中 %d 处，已豁免 %d 处"
             % (len(hits), len(exempted))]
    for rel, ln, cat, txt in hits:
        lines.append("  [%s] %s L%d  %s" % (cat, rel, ln, txt[:110]))
    for i, ent in enumerate(_EXEMPT):
        n = sum(1 for e in exempted if e[4] is ent)
        lines.append("  豁免#%02d %-34s %-8s 命中 %d 次" % (i + 1, ent[0][:34], ent[1], n))
    return (1 if hits else 0), "\n".join(lines)


# ------------------------------------------------------------------ 断言组

def test_scan_surface_is_nonempty():
    """扫描面为空 = 空集恒绿，是坏扫描器掩盖的假绿。"""
    assert os.path.isdir(SCAN_ROOT), "扫描面不存在：%s" % SCAN_ROOT
    files = _scan_files(SCAN_ROOT)
    assert files, "扫描面下无 .md/.json —— 拒绝以空集判绿"


def test_scanner_detects_known_positives():
    """量具自检：每一类都必须能从合成文本读出，否则 0 命中不可信。"""
    samples = (
        ("凭据/令牌", "key = sk-abcdefghijklmnopqrst"),
        ("凭据/令牌", "ghp_" + "a" * 30),
        ("凭据/令牌", "tk_" + "0" * 20),
        ("凭据/令牌", "mdcg1.abcdefgh"),
        ("凭据/令牌", "api_key=xxx"),
        ("凭据/令牌", "password: yyy"),
        ("本机绝对路径", "D:" + chr(92) + "program" + chr(92) + "x"),
        ("本机绝对路径", "C:/tmp/x"),
        ("本机绝对路径", chr(92) * 2 + "srv" + chr(92) + "share"),
        ("本机绝对路径", "/home/alice/proj/f"),
        ("邮箱", "a@b.com"),
        ("本地IP", "192.168.1.5"),
        ("本地IP", "10.1.2.3"),
        ("本地IP", "127.0.0.1"),
        ("私有面标识", "私域 AEIS 镜像"),
        ("私有面标识", "AEIS_DESIGNER_KEY"),
        ("私有面标识", "AEIS_DB"),
    )
    for want, s in samples:
        got = {c for c, _i, _t in _violations_in_text(s)}
        assert want in got, "扫描器读不出 [%s] 类命中：%r -> %s" % (want, s, got)


def test_no_false_positive_on_benign_text():
    """防误杀：仓内真实存在的合法文本不得被判红。"""
    benign = (
        "16:9 横屏画幅（ASPECT_16_9）",
        "class _Block:" + chr(92) + "n",
        "版本 10.0.0 / 1.12.1",
        "PRAISE 词族（不含私有标识）",
        "docs/plans/待裁清单_v0.1.md",
        "见 §4.2 E 族表 E1 · D-007 fail-closed",
        "https://github.com/FuRongJun-1999/lingshu/pull/737",
        "0.80 / 0.90 安全区 · 1920x1080",
        "| 2026-10-09 | 引入件 | CLEAN |",
    )
    for s in benign:
        v = _violations_in_text(s)
        assert v == [], "误杀：%r -> %s" % (s, v)


def test_docs_face_has_no_violations():
    """主判据：docs/** 文档面除豁免表外 0 命中。"""
    hits, _exempted = _scan()
    assert hits == [], "docs/** 未豁免命中 %d 处：\n%s" % (
        len(hits), "\n".join("  [%s] %s L%d %s" % t for t in hits))


def test_exemptions_are_wellformed():
    """豁免表形状：四元组齐备、类别属闭集、glob/子串/理由非空。"""
    for i, ent in enumerate(_EXEMPT):
        assert len(ent) == 4, "豁免#%d 不是四元组" % (i + 1)
        glob, cat, needle, reason = ent
        assert cat in CATEGORIES, "豁免#%d 类别 %r 不在闭集 %s" % (i + 1, cat, CATEGORIES)
        assert glob and needle and reason, "豁免#%d 有空字段" % (i + 1)
        assert len(reason) >= 10, "豁免#%d 理由过短（须可复核）" % (i + 1)


def test_exemptions_are_not_stale():
    """非陈化：每条豁免必须至少命中一次 —— 否则它已是无声的大洞，判红。"""
    used = _exempt_usage()
    stale = [i + 1 for i, n in used.items() if n == 0]
    assert not stale, "豁免表第 %s 条已陈化（0 命中）——内容已变，须逐条复核后删除" % stale


def test_guard_itself_is_collectable():
    """自证：本件必须能被 pytest 收集，否则它自己就落在 gate 盲区。"""
    assert os.path.basename(__file__).startswith("test_")
    assert os.path.isfile(os.path.abspath(__file__))
    for fn in (test_scan_surface_is_nonempty, test_scanner_detects_known_positives,
               test_no_false_positive_on_benign_text, test_docs_face_has_no_violations,
               test_exemptions_are_wellformed, test_exemptions_are_not_stale):
        assert callable(fn)


if __name__ == "__main__":
    _code, _text = _report()
    print(_text)
    sys.exit(_code)
