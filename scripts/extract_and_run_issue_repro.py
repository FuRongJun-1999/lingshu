#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""extract_and_run_issue_repro — issue 复现脚本：抽取 → 落盘 → 限时执行 → 读数报告。

供 `.github/workflows/repro-bot.yml` 调用的**执行核**（Actions 只当外壳）；也可本机干跑
（`--body-file` / `--dry-run`）。

定位：外部报告人附了复现脚本的 issue，维护者打 `run-repro` 标签后自动跑出读数并回贴。
**只贴读数、不作结论**——结论仍走人工/模型侧的既有流程。

为什么它能跑不可信代码（安全前提，改动前先读）：
  · 本脚本对被测子进程的**环境做清洗（白名单制，issue #682）**：只透传
    `_ENV_ALLOW_EXACT` / `_ENV_ALLOW_PREFIXES` 命中的「进程运行基座」键
    （路径 / 临时目录 / 家目录 / 区域 / 宿主自省 / CPython 配置），**其余继承键
    一律丢弃**——被测脚本的**进程 env 里**拿不到 job 令牌，也拿不到
    `$GITHUB_ENV`/`$GITHUB_OUTPUT` 等 CI 注入面的路径，更拿不到 `*_KEY` /
    `*_URL` / `SSH_AUTH_SOCK` / `KUBECONFIG` / `DOCKER_HOST` / `NETRC` /
    `*_PROXY` 这类**能力凭据**（旧实现是词表丢弃，漏掉这一整类；workflow 侧另有
    配合：取正文与回评步骤才带 GH_TOKEN，执行步骤不带；checkout 用
    persist-credentials: false 不落 git 凭据）。
    ⚠️ **本清洗只清「被测子进程自己的 env」，不等于同 job 隔离**（#241）：同 uid
    的被测代码仍可读父进程 environ（如 Linux `/proc/<ppid>/environ`）把被清洗的
    键取回，也可写 `$GITHUB_ENV`/`$GITHUB_PATH` 影响**后续步骤**。真正的边界是
    workflow 侧那条**「执行步骤不带令牌、回评步骤才带令牌」的分步隔离**，不是本
    清洗；同 job 内不可信代码与持令牌步骤共存即无隔离。要真隔离须让执行与回评
    分属不同 job（仅**回评**那一步的正文来自被测输出，故可让回评 job 只做
    `gh issue comment --body-file`）。
  · 独立临时目录（默认 mkdtemp；不写仓内）、限时执行（默认 600 s，超时 kill）。
    被测输出的回收**不走管道**（#362③）：脚本派生的孙/后台进程会一直占着管道写端，
    用 `subprocess.run(capture_output=True)` 会等管道 EOF 而非进程退出 ⇒ 脚本已
    exit 0 也被判「超时终止」。改为重定向到临时文件、等**进程退出**。
  · 报告里**可枚举的**本机路径一律中性化（`_neutralize_paths`，判据＝仓工作纪律
    「公开面路径写相对路径／本机绝对路径以中性占位代」）：报告自己给出的
    落盘路径写 `<临时目录>`，被测输出里出现的落盘目录、执行核所在仓根也一并替换。
    被测输出里它自己探到的**其它**绝对路径无法枚举，如实保留——本承诺不含它们
    （#239②）。
  · 报告记录**被测提交**（执行核所在仓的 `git rev-parse HEAD` ＋工作树是否干净）
    与**依赖版本**（numpy / Pillow / lingshu）——读数要可复现，必须钉住被测代码与
    环境的版本（#239①）。
  · 本脚本自身的 stdout/stderr 显式切 UTF-8（`_force_utf8_stdio`）：Windows 上
    stdout 被重定向（管道/文件）时按控制台代码页（常为 cp936/gbk）编码，而本模块
    文档含 `⇒` / `⚠` 等 GBK 编不出的字符 ⇒ `--help` 当场 UnicodeEncodeError 崩
    （#120）。控制台乱码属显示层（仓工作纪律不做控制台兼容），**崩溃不是**。
  · **报告里一切来自正文的文本都经「反注入」处理**：正文围栏 info string 属攻击者
    可控，块语言分布一律包成**行内代码**（`_inline_code`）再入报告，使其中的
    `[x](url)` / `@提及` / `<html>` 只作字面量、不被渲染（#363）；stdout/stderr
    走 `_md_fence` 动态围栏防撑破。报告正文只由**固定模板**拼装。
  · 解释器用 `python -X utf8`；子进程 env 强制 PYTHONUTF8=1（对齐仓工作纪律「统一 UTF-8」）。

抽取规则（CommonMark 围栏语义——本仓无该解析器，口径以本节为准；参考实现见
脑端公开仓 `md_cg/docindex.py::_FENCE`，门禁会 clone 该仓供 test_brain_store）：
  · 开启：行首（允许前导空白）`{3,}` 或 ~{3,}；info string = 同行其后剩余文本。
  · 闭合：同行定界符字符、长度不短于开启者、且该行除定界符外无其它字符
    （围栏内写的 ` ```js ` 只算代码内容，不算闭合）。
  · **代码行去缩进**：开启围栏前有 N 列缩进（空格 1 列、tab 展开到下一制表位——
    CommonMark §2.2 列宽口径）时，块内每行去掉行首至多 N **列**。旧实现只数/只剥
    空格字符 ⇒ `  ```python` / `  print(1)` 抽出的代码带缩进落盘即成
    `IndentationError`（#362①）；tab 缩进的围栏更是一个字符都不剥、整块带 tab 落盘
    同样 `IndentationError`——合法复现被判失败（#599）。
  · **引用块（`>`）内的围栏算块**：行首 `>` 标记（每层后可跟一个空格，可嵌套）
    剥掉后再判围栏与代码行（CommonMark 块引用语义）。旧实现不认 `>` ⇒ 报告人写在
    引用块里的复现脚本**整块抽不到**（#362②）。
  · 未闭合（到文末）：按「到文末收尾」收编（CommonMark 语义）。
  · **HTML 注释（`<!-- … -->`，可跨行）内的围栏不算块**——注释在网页上不可见，
    其中的 python 块不得被抽取执行（#235）。检测按「行首是否处于注释内」判（单趟
    扫描，围栏内代码里的 `<!--` 不认作注释），抽出的代码仍取原文行。
  · **嵌套容器（#612）**：HTML 块起点/结束的判定剥的是行首**全部**容器前缀
    （块引用 `> ` 与列表标记可嵌套可交错，如 `- > <div`、`  - <div`）——旧实现
    每行只剥一层 ⇒ 嵌套容器内的 `<div title="` 起点判不出，块内缩进围栏被当真
    围栏抽出执行，而渲染下整段被吞进属性值（页面上看不见的代码被执行）。
    起点行自带**可打断段落**的列表标记（圆弹/`1.`）时，**⑦ 型**起点视同前行空白
    （CommonMark 5.2：列表标记开新列表项、打断前行段落）；**⑥ 型恒可打断段落**
    （CommonMark 4.6 / #595）⇒ 不依赖此豁免。
  · **只选第一个 info 首词为 python|py|py3（大小写不敏感）的围栏块**；
    正文含多个 python 块时也只用第一个——报告里如实给出全部块的语言分布。
  · 无 python 块 ⇒ 显式失败（退出码 2），**不执行任何东西**。

用法（argv）：
    python -X utf8 scripts/extract_and_run_issue_repro.py --repo OWNER/REPO --issue N [选项]
    python -X utf8 scripts/extract_and_run_issue_repro.py --body-file PATH [--source-desc "..."] [选项]
    python -X utf8 scripts/extract_and_run_issue_repro.py --selftest

选项：
    --repo OWNER/REPO        取正文来源仓（与 --issue 配套；内部走 `gh api`）
    --issue N                取该 issue 正文
    --body-file PATH         直接读本地文件当正文（离线/干跑；与 --repo/--issue 互斥）
    --source-desc TEXT       报告「来源」行用（缺省：body 文件名 / OWNER/REPO#N）
    --timeout SECONDS        执行上限，默认 600
    --max-output-bytes N     stdout/stderr 各自截断上限（字节），默认 20480（20 KiB）
    --out-dir PATH           脚本落盘目录（默认新建 mkdtemp 临时目录）
    --env KEY=VALUE          透传给被测脚本的环境变量（可重复；KEY 合法且 VALUE 非空才生效）
    --report-file PATH       报告落盘路径（同时总是打印到 stdout）
    --dry-run                只抽取+落盘，不执行
    --selftest               抽取器正负例自检（对齐 check_pr_title.py 的先例）
    -h / --help              显示本帮助

退出码：0 = 被测脚本退出码 0；
        1 = 被测脚本退出码非 0（含超时终止）；
        2 = 抽取失败 / 取正文失败 / 用法错误。
"""
from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import tempfile


# ---------------------------------------------------------------------------
# 入口护栏
# ---------------------------------------------------------------------------

def _force_utf8_stdio():
    """把本进程 stdout/stderr 显式切到 UTF-8（#120）。

    为什么必须：Windows 上 stdout **被重定向**（管道 / 文件 / CI 捕获）时按控制台
    代码页（本机 cp936＝gbk）编码，而本模块文档含 `⇒`(U+21D2) / `⚠`(U+26A0) 等 GBK
    **编不出**的字符 ⇒ `--help`（`print(__doc__)`）当场 `UnicodeEncodeError` 退出码 1。
    控制台乱码属显示层（仓工作纪律不做控制台兼容），**崩溃不是**——故这里
    只把编码钉成 UTF-8；`errors="replace"` 兜底：某端不支持 reconfigure（如测试里被
    换成 StringIO）时静默跳过，绝不因编码问题崩。只动本进程自身 stdio，不碰被测
    子进程（后者另有 `_child_env` 注 `PYTHONIOENCODING=utf-8`）。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

#: 围栏开启行（CommonMark 语义；参考实现见脑端公开仓 md_cg/docindex.py::_FENCE）。
_FENCE_OPEN = re.compile(r"^\s*(`{3,}|~{3,})")

#: info string 首词命中这些即视为「python 块」（大小写不敏感，取小写后比对）。
#: `python3` 同认（#409）：GFM 高亮语法里 ```python3 是合法 info string，
#: 旧集合漏收 ⇒ 合法复现被判「正文无 python 围栏块」不执行。
_PY_LANGS = {"python", "python3", "py", "py3"}

# ---------------------------------------------------------------- HTML 块（#520）
#: CommonMark 4.6 HTML 块七型（与 cmark/GitHub 渲染同口径）：HTML 块内的围栏
#: **不是代码围栏**——网页上它们被吞进标签/属性值（<div title="```python…```">）
#: 或 script/pre 正文，维护者看不见，抽取器不得把它们当可执行块。
#: 七型中 ③（<!-- 注释）由既有 `_advance_comment_state` 承担；这里补
#: ①（script/pre/style/textarea，**四者任一**闭标签收——#604 对齐 CommonMark
#: ①型终止条款）、②（<? … ?>）、④（<! 声明，行内含 `>` 收——裸 `>` 行在
#: 原始行上判定，#604）、⑤（<![CDATA[ … ]]>）、⑥（已知块标签，空行收；
#: 起点只需 `<tag` + 空白/>/EOL，
#: **属性可跨行未闭**——#520 的攻击形态正落在这）、⑦（行首完整孤立标签，
#: 空行收）。**⑦ 型不得打断段落**（CommonMark 4.6）⇒ 仅在前行空白时生效；
#: **⑥ 型可以打断段落**（#595 纠正：CommonMark 只规定 ⑦ 型不能打断）⇒ 恒生效。
_HTML1_TAGS = ("script", "pre", "style", "textarea")
_RE_HTML1_START = re.compile(
    r"^ {0,3}<(%s)(?:\s|>|/>|$)" % "|".join(_HTML1_TAGS), re.IGNORECASE)
#: ① 型**结束条件**（#604）：行内出现 script/pre/style/textarea **四者任一**的
#: 闭标签即收——与开标签不必同名（CommonMark 4.6 ①型：`<script>…</pre>` 用
#: `</pre>` 收是合法结束）。旧「同名收」漏收 ⇒ 有围栏的正文被谎报无块。
#: `</div>` 等四者之外的闭标签不收（与渲染真值一致）。大小写不敏感与起点同口径。
_RE_HTML1_END = re.compile(
    r"</(?:%s)\s*>" % "|".join(_HTML1_TAGS), re.IGNORECASE)
#: ⑥ 的已知块标签表（CommonMark 列表）。
_HTML6_TAGS = ("address", "article", "aside", "base", "basefont", "blockquote",
               "body", "caption", "center", "col", "colgroup", "dd", "details",
               "dialog", "dir", "div", "dl", "dt", "fieldset", "figcaption",
               "figure", "footer", "form", "frame", "frameset", "h1", "h2", "h3",
               "h4", "h5", "h6", "head", "header", "hr", "html", "iframe",
               "legend", "li", "link", "main", "menu", "menuitem", "nav",
               "noframes", "ol", "optgroup", "option", "p", "param", "search",
               "section", "summary", "table", "tbody", "td", "tfoot", "th",
               "thead", "title", "tr", "track", "ul")
_RE_HTML6_START = re.compile(
    r"^ {0,3}</?(?:%s)(?:\s|>|/>|$)" % "|".join(_HTML6_TAGS), re.IGNORECASE)
#: ⑦ 完整开/闭标签独占一行（属性须闭合——⑦与⑥的关键差：⑥ 起点允许属性跨行未闭）。
_RE_HTML7_START = re.compile(
    r"^ {0,3}(?:<[A-Za-z][A-Za-z0-9-]*(?:\s+[A-Za-z_:][A-Za-z0-9_:.-]*"
    r"(?:\s*=\s*(?:[^\s\"'=<>`]+|'[^']*'|\"[^\"]*\"))?"
    r")*\s*/?>|</[A-Za-z][A-Za-z0-9-]*\s*>)\s*$")


def _html_block_start(line, prev_blank):
    """判定一行是否开启 HTML 块（CommonMark ①②④⑤⑥⑦；③注释走既有状态机）。

    返回 HTML 块状态：("tag", 标签名)（①：四标签**任一**闭标签收，#604；
    存名仅供报告）、("marker", 终串)（②④⑤：`?>` / `>` / `]]>` 收）、
    ("blank",)（⑥⑦：空行收）；非块行返回 None。
    ⑦ 不打断段落（CommonMark 4.6）⇒ 前行非空白文本时不启动；
    ⑥ 型**可以**打断段落（#595：`文本行` 紧跟 `<div title="` 照样开块）⇒ 不受此前置。
    """
    m1 = _RE_HTML1_START.match(line)
    if m1:
        return ("tag", m1.group(1).lower())
    if re.match(r"^ {0,3}<\?", line):
        return ("marker", "?>")
    if re.match(r"^ {0,3}<![A-Za-z]", line):          # ④（<!DOCTYPE 等；<!-- 是③）
        return ("marker", ">")
    if re.match(r"^ {0,3}<!\[CDATA\[", line):
        return ("marker", "]]>")
    # #595：⑥ 型可打断段落（CommonMark 4.6：1–6 型都可；只有 ⑦ 型不可）——
    # 旧实现在 `if prev_blank:` 门下判 ⑥ ⇒ 正文「一行文字」紧跟 `<div title="`
    # 时不认块，块内围栏被当真围栏抽出执行，而 GitHub（cmark-gfm）把整段吞进
    # 属性值 ⇒ 页面上看不见的代码被执行（人审的代码 ≠ runner 跑的代码）。
    if _RE_HTML6_START.match(line):
        return ("blank",)
    if prev_blank and _RE_HTML7_START.match(line):   # 只有 ⑦ 型不能打断段落
        return ("blank",)
    return None

#: 被测子进程环境的**白名单·具名**（#682）——跨平台「进程运行基座」：可执行查找 /
#: 命令解释器 / 临时目录 / 家目录 / 区域 / 宿主自省 / CPython 配置面。
#: **不含任何凭据或能力形状名**（`*_KEY` / `*_TOKEN` / `*_URL` / `*_URI` / `*_DSN`
#: / `SSH_AUTH_SOCK` / `KUBECONFIG` / `DOCKER_HOST` / `NETRC` / `*_PROXY` 等）。
#: 口径与脑侧 `lingshu/world/brain_store.py::_ENV_ALLOW_NAMES`（issue #227/#652）一致。
_ENV_ALLOW_EXACT = frozenset({
    # 可执行查找 / 命令解释器
    "PATH", "PATHEXT", "COMSPEC",
    # Windows 目录基座（Python 解释器与 stdlib 启动期读用）
    "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR",
    "PROGRAMDATA", "ALLUSERSPROFILE", "PUBLIC",
    "PROGRAMFILES", "PROGRAMFILES(X86)", "COMMONPROGRAMFILES",
    "COMMONPROGRAMFILES(X86)", "PROGRAMW6432", "COMMONPROGRAMW6432",
    # 临时目录
    "TEMP", "TMP", "TMPDIR",
    # 家目录 / 账户
    "HOME", "HOMEDRIVE", "HOMEPATH", "USERPROFILE",
    "APPDATA", "LOCALAPPDATA",
    "USER", "USERNAME", "USERDOMAIN", "LOGNAME",
    # 区域 / 终端
    "LANG", "LANGUAGE", "TZ", "TERM",
    # 宿主自省（解释器/编译扩展常用）
    "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "PROCESSOR_IDENTIFIER",
    "PROCESSOR_LEVEL", "PROCESSOR_REVISION", "OS",
    # CPython 配置/行为面（**不含** PYTHONHOME / PYTHONSTARTUP / PYTHONINSPECT /
    # PYTHONEXECUTABLE——解释器启动期劫持面，见脑侧 #652③；被测脚本按
    # `python -X utf8 <script>` 启动，不需要它们）。
    "PYTHONPATH", "PYTHONUTF8", "PYTHONIOENCODING",
    "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED", "PYTHONWARNINGS",
    "PYTHONFAULTHANDLER", "PYTHONHASHSEED", "PYTHONNOUSERSITE",
    "PYTHONSAFEPATH", "PYTHONDEVMODE",
    # 区域/编码面（#495，复核 #680 §一.3 残留）：原 `("LC_",)` 宽前缀顺带放行
    # 任意 `LC_*` 名，含凭据形状的 `LC_TOKEN` / `LC_PASSWORD`——与上面「不含任何
    # 凭据或能力形状名」的声明相悖。按脑侧 #227 续 / #652③ 同一手法收敛为
    # POSIX + glibc 的 13 个区域名具名（非标准 `LC_*` 名本无区域语义）。
    "LC_ALL", "LC_COLLATE", "LC_CTYPE", "LC_MESSAGES", "LC_MONETARY",
    "LC_NUMERIC", "LC_TIME",
    "LC_PAPER", "LC_NAME", "LC_ADDRESS", "LC_TELEPHONE", "LC_MEASUREMENT",
    "LC_IDENTIFICATION",
})

#: 白名单·前缀（#682）：**为空**（#495）——原仅余 `LC_`，该前缀会顺带放行
#: `LC_TOKEN` / `LC_PASSWORD` 这类凭据形状名，13 个区域名已改具名（见上）。
_ENV_ALLOW_PREFIXES = ()


DEFAULT_TIMEOUT = 600
DEFAULT_MAX_OUTPUT_BYTES = 20480

#: 报告末行固定声明（**不要**改动措辞——它是「只贴读数不作结论」的对外契约）。
REPORT_FOOTER = "本评论仅贴读数，不作结论。"


# ---------------------------------------------------------------------------
# 抽取
# ---------------------------------------------------------------------------

def _advance_comment_state(line, in_comment):
    """逐字符推进 HTML 注释状态，返回处理完该行后的 `in_comment`。

    HTML 注释不可嵌套：`<!--` 开启、`-->` 闭合；同行可多次开关。注释跨行时
    后续行的**行首**即处于注释内——那正是围栏检测要用的判据。
    """
    k = 0
    while k < len(line):
        if not in_comment:
            s = line.find("<!--", k)
            if s == -1:
                return False
            in_comment = True
            # #604：闭串搜索起点 s+2 允许 `-->` 与 `<!--` **重叠**——空注释
            # `<!-->` / `<!--->` 的 `-->` 分别落在偏移 2/3，从 s+4 起搜会漏掉、
            # 误判「真未闭合」把后续围栏全吞（rc=2 谎报无块）。CommonMark 4.6
            # ③型按字面找 `-->`、不排除与开串共享字符；`<!--` 真未闭合与
            # `<!-- -->` 正常注释的行为不变。
            k = s + 2
        else:
            e = line.find("-->", k)
            if e == -1:
                return True
            in_comment = False
            k = e + 3
    return in_comment


#: 块引用标记（CommonMark）：行首至多 3 个空格 + `>`，标记后可跟一个空格。
_BQ_MARKER = re.compile(r"^ {0,3}> ?")


def _bq_strip(line, max_depth=None):
    """剥掉行首的块引用标记，返回 (余下文本, 实际剥掉的层数)。

    `>` 前允许至多 3 个空格、标记后可跟一个空格（CommonMark 块引用语义）；
    `> > x` 为两层（可嵌套）。`max_depth` 给定时至多剥这么多层——用于让块内
    代码行/闭合行与开启行**对齐同一引用深度**，从而「不在引用块里的围栏」其
    代码行里的 `>` 保持字面量、不被误剥（#362②）。
    """
    depth = 0
    s = line
    while max_depth is None or depth < max_depth:
        m = _BQ_MARKER.match(s)
        if not m:
            break
        s = s[m.end():]
        depth += 1
    return s, depth


def _indent_width(s):
    """行首缩进的**列宽**（#599；CommonMark §2.2「制表符」口径）。

    空格记 1 列；制表符展开到下一制表位（按 4 列宽制表位：`w += 4 - (w % 4)`）；
    遇其它字符即停。为什么按列不按字符：`_FENCE_OPEN` 用 `\\s` 认围栏（tab 缩进的
    围栏**合法**），而旧实现 `len(head) - len(head.lstrip(" "))` 只数空格 ⇒ tab
    缩进的围栏 indent=0、`_dedent` 一个字符都不剥 ⇒ 块内每行带着 tab 落盘即
    `IndentationError`，合法复现被静默判死（#599）。
    """
    w = 0
    for ch in s:
        if ch == " ":
            w += 1
        elif ch == "\t":
            w += 4 - (w % 4)
        else:
            break
    return w


def _dedent(line, n):
    """去掉行首至多 n **列**缩进（#599；CommonMark 围栏语义＋§2.2 列宽口径）。

    按列宽推进（空格 1 列、tab 展开到下一制表位），剥到累计列宽 ≥ n 或遇非缩进
    字符为止——与 `_indent_width` 同口径，tab 缩进的围栏才能把块内 tab 一并剥掉。
    `n <= 0` 直接返回原文（一个字符都不剥）。制表符跨过 n 列边界时整枚剥除
    （列宽口径下不存在「剥半个 tab」）。
    """
    if n <= 0:
        return line
    k, w = 0, 0
    while k < len(line):
        ch = line[k]
        if ch == " " and w < n:
            w += 1
        elif ch == "\t" and w < n:
            w += 4 - (w % 4)
        else:
            break
        k += 1
    return line[k:]

#: 列表项标记（CommonMark）：`- ` / `* ` / `+ ` / `N. `（N≥1 数字）。
_RE_LIST_MARKER = re.compile(r"^ {0,3}(?:[-*+]|\d{1,9}[.)])(?:\s+|$)")

#: **可打断段落**的列表标记（CommonMark 5.2「列表打断段落」条件）：圆弹标记
#: 恒可（其后内容非空即打断——本函数只服务 ⑥⑦ 起点判定，起点匹配时标记后
#: 必有 `<tag`，内容非空自动成立）；有序标记仅起始号为 1 可（`2.` 起头不得
#: 打断——起点行仍按段落延续处理，其后围栏按 CommonMark 是**真围栏**）。
#: markdown-it（commonmark 预设）对表验证：`para\n- <div title="` 与
#: `para\n1. <div title="` 均开 HTML 块吞围栏；`para\n2. <div title="` 不开、
#: 围栏照常渲染。
_RE_LIST_MARKER_INTERRUPT = re.compile(r"^ {0,3}(?:[-*+]|1[.)])(?:\s+|$)")

#: _container_strip 循环剥除的层数上限（#612）：防死循环护栏。每层至少剥 1
#: 字符（标记本身），循环实为「剥不动即停」，上限是双保险（正常嵌套远不到 8 层）。
_MAX_CONTAINER_LAYERS = 8


def _container_strip(line):
    """剥掉行首的**全部**容器前缀（块引用 `> ` ＋ 列表项标记，可嵌套可交错）。

    返回 (剥净后的内容, 前缀里是否含「可打断段落」的列表标记)。

    #612（#520/#555/#568 同族残留，维护者复核指出）：旧实现每行只剥**一层**
    （`_bq_strip` 一遍 ＋ 列表标记 sub 一次）⇒ HTML 块起点行处于**嵌套容器**
    时判不出「HTML 块开始」——形态 A（`- > <div title="` / `- > > <div title="`，
    列表标记后还跟块引用标记，剥一层后行首仍是容器标记）；形态 B（`  - <div
    title="`，外层列表项内容缩进内的嵌套列表项——剥一层后虽露出 `<div`，但
    前行是外层列表项的段落文字、⑥ 型起点按「不打断段落」被拒（**#595 前**的
    旧口径；⑥ 型自 #595 起恒可打断段落 ⇒ 形态 B 的 ⑥ 起点已不依赖此豁免，
    ⑦ 型起点仍依赖，见 `_extract_fenced_blocks` 的 prev_blank 处理）。两种形态下块内缩进围栏都被
    当真围栏抽出执行，而 CommonMark 渲染（markdown-it commonmark 预设实测）
    整段被吞进 `<div title="…">` 属性值——页面上看不见的代码被执行。

    修法：反复应用「剥引用标记 → 剥列表标记」直到行首不再是容器标记（每层
    前导空格至多 3 个，与 _BQ_MARKER/_RE_LIST_MARKER 既有口径一致——形态 B
    的嵌套列表项即「前导容器缩进 ＋ 标记」的完整一层）。#555 的单层语义
    （`> <div` / `- <div`）是本循环一层的特例，行为不变。

    第二返回值（可打断段落的列表标记在起点行上）：CommonMark 5.2 列表可打断
    段落 ⇒ 标记开**新列表项**、打断前行段落 ⇒ 新容器首行内容没有「开放段落」，
    **⑦ 型**起点对该行视同前行空白（⑥ 型恒可打断段落——#595；形态 B 的 ⑥ 起点
    在 #595 后已不依赖它，⑦ 型起点仍依赖，见调用点）。块引用标记
    **不**给此豁免：`> 文本` 后同行首的 `> <div` 属同一引用内的段落延续、
    `> 文本` 后**不带**标记的行是 lazy continuation——扁平行扫描没有容器开闭
    状态，两者都不可与「引用新开」区分；且 GitHub（cmark）下 `> 文本` 后的
    ⑥ 起点确属段落延续、其后围栏是**可见**代码。宁缺勿错：漏判 ⑥ = 隐藏代码
    被执行（本缺陷的破坏方向），此处保守多抽。

    边界：剥的只是容器标记及其前 0–3 个前导空格，**不碰代码缩进的剥除**——
    `_dedent` 的缩进量在围栏路径上按 `_bq_strip` 后的原始行计算（head/indent
    均不经本函数），两口径互不影响。
    """
    s = line
    interrupt = False
    for _ in range(_MAX_CONTAINER_LAYERS):
        before = s
        s, _ = _bq_strip(s)
        m = _RE_LIST_MARKER_INTERRUPT.match(s)
        if m:
            s = s[m.end():]
            interrupt = True
        else:
            m = _RE_LIST_MARKER.match(s)
            if m:
                s = s[m.end():]
        if s == before:                    # 本层剥不动 ⇒ 行首已不是容器标记
            break
    return s, interrupt


def _extract_fenced_blocks(text):
    """按 CommonMark 围栏口径抽取全部围栏块。

    返回 [(lang, code_first_line, code_last_line, code)]（行号为正文内 1-based，
    code_first/last 指块内**代码行**范围；空块记 first>last）。lang 为 info 首词
    小写；无 info 记 ""。

    HTML 注释（`<!-- … -->`）内的围栏**不算块**（#235）——注释在网页上不可见，
    其中的代码不得被抽取执行。按 CommonMark 块级语义单趟扫描：**行首在注释内**
    的行不参与围栏判定；反之，围栏内（含代码里的 `<!--`）一律按原文、不认注释。
    两条构造互不越界（`<!--` 出现在围栏内不被当注释、围栏出现在注释内不被当块）。

    容器与缩进（#362①②，CommonMark 围栏/块引用语义）：
      · **块引用**：行首 `>` 标记（可嵌套）剥掉后再判围栏与代码行；开启行的引用
        深度即该块的深度，代码行/闭合行只剥到同深度（非引用块里代码行的 `>` 是
        字面量，不剥）。
      · **缩进**：开启围栏前有 N 列缩进（空格 1 列、tab 展开到制表位，CommonMark
        §2.2 列宽口径）时，块内每行去掉行首至多 N 列——旧实现照抄原文行，
        `  ```python` / `  print(1)` 抽出的代码带缩进、落盘即 `IndentationError`
        （#362①）；tab 缩进围栏则一个字符都不剥、整块带 tab 落盘（#599）。
      · **嵌套容器（#612）**：HTML 块的起点/结束在剥**净**容器前缀后的内容上判定
        （多层标记循环剥除）；起点行自带可打断段落的列表标记时 ⑥⑦ 视同前行空白
        ——嵌套列表项/列表套块引用内的 `<div title="` 不再漏判，块内围栏不得抽出。
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks = []
    i, n = 0, len(lines)
    in_comment = False
    html_state = None          # #520：HTML 块状态（None=不在块内）
    prev_blank = True          # 行 0 视同「前行空白」——⑥⑦ 打断段落判据用
    while i < n:
        if html_state is not None:
            # HTML 块内：围栏不算块、注释不算注释——只找本型的结束条件。
            # #555 残留：容器（引用块/列表项）内的 HTML 块，其结束条件也在
            # 剥容器后的内容上判定（`> ` 后的空行也是空行）。
            # #612：嵌套容器（如 `- > `、`  - ` 多层前缀）剥**净**后判定。
            content, _ = _container_strip(lines[i])
            kind = html_state[0]
            if kind == "blank":                       # ⑥⑦：空行收（空行不属块）
                if not content.strip():
                    html_state = None
                    prev_blank = True
                else:
                    prev_blank = False
            elif kind == "tag":                       # ①：四者任一闭标签即收（含本行）
                # #604：旧判定按「同名」`</tag>` 收，与 CommonMark ①型不符——
                # 终止条件是 script/pre/style/textarea 任一闭标签（`<script>` 用
                # `</pre>` 收合法），同名收漏收 ⇒ 围栏全被吞、谎报无块。
                if _RE_HTML1_END.search(content):
                    html_state = None
                prev_blank = not content.strip()
            else:                                     # ②④⑤：出现终串即收（含本行）
                # #604：④型终止符 `>` **独占一行**时会被 `_container_strip` 当
                # 块引用前缀剥掉（_container_strip('')===''），剥净行上判不出 ⇒
                # 裸 `>` 行改在**原始行**上判定（CommonMark ④型：行内含 `>` 即
                # 收）。②⑤（`?>`/`]]>`）不是容器前缀、剥净行判定不变（行为保持）。
                if (html_state[1] in content
                        or (html_state[1] == ">"
                            and lines[i].strip() == ">")):
                    html_state = None
                prev_blank = not content.strip()
            i += 1
            continue
        if in_comment:
            # 行首已在注释内 ⇒ 本行不可能开启围栏；只推进注释状态。
            in_comment = _advance_comment_state(lines[i], True)
            prev_blank = not lines[i].strip()
            i += 1
            continue
        head, bq_depth = _bq_strip(lines[i])
        m = _FENCE_OPEN.match(head)
        if not m:
            # #520：HTML 块起点判定先于注释推进（①②④⑤⑥⑦；③注释走下方既有
            # 状态机）——`<div title="` 这类起点行里出现 `<!--` 时，块归属按
            # CommonMark 起点次序：行首是什么就开什么型。
            # #555 残留：起点判定在**剥容器后**的内容上跑（与围栏判定同口径）。
            # #612：剥的是**全部**嵌套容器前缀（`- > <div` / `  - <div` 等）。
            content, list_marker_seen = _container_strip(lines[i])
            if list_marker_seen:
                # #612 形态 B：起点行自带**可打断段落**的列表标记（⑦ 型起点用；
                # ⑥ 型自 #595 起恒可打断段落，不再依赖此豁免）（见
                # _container_strip 的 _RE_LIST_MARKER_INTERRUPT）⇒ 标记开新
                # 列表项、打断前行段落 ⇒ 新容器首行内容没有「开放段落」，
                # **⑦ 型**起点对该行视同前行空白。prev_blank 随后在下方两分支
                # 都会按本行内容重算，此处置位只影响本次起点判定。
                prev_blank = True
            st = _html_block_start(content, prev_blank)
            if st is not None:
                html_state = st
                # 同行即含结束条件（如 `<?xml?>`、`<div></div>` 单行）⇒ 立即收。
                if st[0] == "marker" and st[1] in content:
                    html_state = None
                elif st[0] == "tag" and _RE_HTML1_END.search(content):
                    html_state = None
                prev_blank = not content.strip()
                i += 1
                continue
            in_comment = _advance_comment_state(lines[i], False)
            # #612：prev_blank 按**剥净容器后**的有效行计（`> ` / `- ` 独占的
            # 行按容器内空白计——其后新容器的 ⑦ 起点无开放段落可打断；⑥ 型
            # 不受此前置）。
            prev_blank = not content.strip()
            i += 1
            continue
        # 围栏开启行本身不改变段落空白判据（块级构造之后的行都是「块后」行——
        # 保守按非空白处理；⑥⑦ 打断段落的判据只在文本段落场景起作用）。
        prev_blank = False
        delim = m.group(1)
        indent = _indent_width(head)   # #599：按列宽（tab→制表位），与 _dedent 同口径
        info = head[m.end():].strip()
        lang = info.split()[0].lower() if info else ""
        j = i + 1
        closed = False
        while j < n:
            cand, cand_depth = _bq_strip(lines[j], bq_depth)
            s = cand.strip()
            if (cand_depth == bq_depth and s and s[0] == delim[0]
                    and len(s) >= len(delim)
                    and all(ch == delim[0] for ch in s)):
                closed = True
                break
            j += 1
        code_end = j if closed else n          # 0-based，代码行止于 code_end-1
        code_lines = []
        for k in range(i + 1, code_end):
            raw, _ = _bq_strip(lines[k], bq_depth)
            code_lines.append(_dedent(raw, indent))
        code = "\n".join(code_lines)
        blocks.append((lang, i + 2, code_end, code))
        i = (j + 1) if closed else n
    return blocks


def _choose_python_block(blocks):
    """第一个 info 首词为 python|py|py3 的块；无则 None。"""
    for b in blocks:
        if b[0] in _PY_LANGS:
            return b
    return None


# ---------------------------------------------------------------------------
# 取正文
# ---------------------------------------------------------------------------

def _fetch_issue_body(repo, issue, env=None):
    """`gh api repos/OWNER/REPO/issues/N` 取正文（gh 输出 JSON；直接解析，不经 shell）。"""
    cmd = ["gh", "api", "repos/%s/issues/%s" % (repo, issue)]
    run_env = dict(os.environ)
    run_env["PYTHONUTF8"] = "1"
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env or run_env)
    if r.returncode != 0:
        raise RuntimeError("gh api 失败（退出码 %d）：%s"
                           % (r.returncode, (r.stderr or "").strip()[:800]))
    try:
        obj = json.loads(r.stdout)
    except ValueError as e:
        raise RuntimeError("gh api 返回非 JSON：%s" % e) from e
    return obj.get("body") or ""


# ---------------------------------------------------------------------------
# 执行
# ---------------------------------------------------------------------------

def _env_is_allowed(name):
    """继承的 env 键是否在白名单内（大小写无关：Windows 上 `os.environ` 键为大写）。"""
    up = str(name).upper()
    if up in _ENV_ALLOW_EXACT:
        return True
    return any(up.startswith(p) for p in _ENV_ALLOW_PREFIXES)


def _child_env(extra):
    """构造被测子进程 env：**默认丢弃 + 白名单透传**，再注入显式覆盖。

    #682：旧实现是「默认透传 + 词表丢弃」——词表（TOKEN|SECRET|PASSWORD|
    CREDENTIAL）**不含** `KEY`/`URL`/`URI`/`DSN`，且盖不住 `SSH_AUTH_SOCK` /
    `KUBECONFIG` / `DOCKER_HOST` / `NETRC` / `*_PROXY` 这类**能力凭据**，被测
    （不可信）脚本能原样拿到（如 `AEIS_DESIGNER_KEY` / `AWS_ACCESS_KEY_ID` /
    `DATABASE_URL`）。改为白名单后，父进程 env 里除 `_ENV_ALLOW_EXACT` /
    `_ENV_ALLOW_PREFIXES` 命中的键外一律不透传；白名单只含「进程运行基座」，
    不含任何凭据。

    `PYTHONUTF8`/`PYTHONIOENCODING` 仍强制注入、`extra` 仍可覆盖（既有契约不变）。

    ⚠️ 作用域**仅限被测子进程自己的 env**（#241）：同 uid 的被测代码仍可读父
    进程 environ（如 Linux `/proc/<ppid>/environ`）把被清洗的键取回，也可写
    `$GITHUB_ENV`/`$GITHUB_PATH` 影响**同一 job 的后续步骤**。本清洗不是隔离
    边界——隔离边界在 workflow 的分步安排（执行步骤不带令牌）。见模块 docstring。
    """
    env = {}
    for k, v in os.environ.items():
        if _env_is_allowed(k):
            env[k] = v
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env.update(extra)
    return env


def _run_script(script_path, cwd, timeout, extra_env):
    """限时执行；返回 (status, exit_line, stdout, stderr)。

    status：0 通过 / 1 非 0 或超时。

    ⚠️ 输出回收**不走管道**（#362③）：`subprocess.run(capture_output=True)` 读管道
    直到 **EOF**——被测脚本若派生后台/孙进程且它继承了 stdout/stderr 写端，管道在
    孙进程存活期间不关闭 ⇒ 父进程**已 exit 0** 也被判 `TimeoutExpired`，合法复现
    被当失败。改为把子进程 stdout/stderr 重定向到临时文件、用 `Popen.wait(timeout)`
    等**进程退出**，随后读文件（同样按 UTF-8 解码）。

    超时则 kill 直接子进程（其孙进程可能短暂残留——一次性 runner，已在文档接受；
    它们此后只写文件、不再阻塞读数）。
    """
    cmd = [sys.executable, "-X", "utf8", script_path]
    env = _child_env(extra_env)
    out_fd, out_path = tempfile.mkstemp(prefix="repro_out_", suffix=".txt")
    err_fd, err_path = tempfile.mkstemp(prefix="repro_err_", suffix=".txt")
    os.close(out_fd)
    os.close(err_fd)

    def _read(path):
        try:
            with open(path, "rb") as f:
                return f.read().decode("utf-8", "replace")
        except OSError:
            return ""

    rc, timed_out, out, err = None, False, "", ""
    try:
        with open(out_path, "wb") as of, open(err_path, "wb") as ef:
            try:
                proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=of, stderr=ef)
            except OSError as e:
                return (1, "启动失败（%s）" % e, "", "")
            try:
                rc = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                proc.kill()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
        out, err = _read(out_path), _read(err_path)
    finally:
        for p in (out_path, err_path):
            try:
                os.unlink(p)
            except OSError:
                pass
    if timed_out:
        return (1, "超时终止（%d s）" % timeout, out, err)
    return (0 if rc == 0 else 1, str(rc), out, err)


# ---------------------------------------------------------------------------
# 报告渲染（固定模板）
# ---------------------------------------------------------------------------

def _truncate(text, limit):
    """按 UTF-8 字节截断（不切半个字符）；返回 (text, 原始字节数, 是否截断)。"""
    data = (text or "").encode("utf-8")
    if len(data) <= limit:
        return text or "", len(data), False
    return data[:limit].decode("utf-8", "ignore"), len(data), True


def _md_fence(text):
    """围栏长度 = 内容里最长反引号串 + 1（至少 3）——防输出内容撑破 markdown。"""
    longest = 0
    for m in re.finditer(r"`+", text or ""):
        longest = max(longest, len(m.group(0)))
    return "`" * max(3, longest + 1)


def _inline_code(text):
    """把任意**单行**文本包成 CommonMark 行内代码（定界反引号数 = 内容里最长
    反引号串 + 1）。

    为什么必须包：报告要原样贴到公开 issue，而「块语言」来自正文围栏的 info
    string——**攻击者可控**。原样透出到围栏**之外**的 `- **抽取**：` 行，等于
    把 `[x](url)` / `@提及` / `<html>` 直接渲染成活的 markdown（伪结论/外链/
    提及注入，#363）。行内代码里这些字符一律为字面量，从根上不可渲染。

    首/尾是反引号时按 CommonMark 规则加一个空格（渲染时被剥掉，内容不变）。
    调用点保证 text 无空白（info 首词），故不涉及行内代码不能跨行的限制。
    """
    text = text or ""
    longest = max((len(m.group(0)) for m in re.finditer(r"`+", text)), default=0)
    ticks = "`" * (longest + 1)
    pad = " " if (text.startswith("`") or text.endswith("`")) else ""
    return ticks + pad + text + pad + ticks


def _env_line():
    py = "Python %s（%s）" % (platform.python_version(), platform.python_implementation())
    osdesc = "%s %s" % (platform.system(), platform.release())
    if os.environ.get("GITHUB_ACTIONS") == "true":
        where = "GitHub Actions 托管 runner"
    else:
        where = "本机（非 Actions）"
    return "%s · %s · %s" % (py, osdesc, where)


def _git_head_line():
    """被测提交：执行核所在仓的 `git rev-parse HEAD`（#239①）。

    读数要可复现，必须钉住**被测代码**的版本——旧报告只有 Python/OS，同一份
    正文在不同 HEAD 上跑出的读数无法区分。取不到（无 git / 非仓）如实写「未知」，
    不编造 SHA。工作树有未提交改动时追加标注——HEAD 相同但内容不同会误导。
    """
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        r = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=15)
    except (OSError, subprocess.SubprocessError) as e:
        return "未知（git 不可用：%s）" % e
    if r.returncode != 0:
        return "未知（git rev-parse 退出码 %d）" % r.returncode
    sha = (r.stdout or "").strip() or "未知"
    try:
        st = subprocess.run(["git", "-C", repo, "status", "--porcelain"],
                            capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=15)
        if st.returncode == 0 and (st.stdout or "").strip():
            return "%s（工作树有未提交改动）" % sha
    except (OSError, subprocess.SubprocessError):
        pass
    return sha


def _deps_line():
    """被测环境的依赖版本（#239①）：numpy / Pillow / 本包；未装如实写「未安装」。"""
    parts = []
    for dist in ("numpy", "Pillow", "lingshu"):
        try:
            parts.append("%s %s" % (dist, importlib.metadata.version(dist)))
        except importlib.metadata.PackageNotFoundError:
            parts.append("%s 未安装" % dist)
        except Exception as e:                      # 元数据面异常不拖垮读数
            parts.append("%s 版本未知（%s）" % (dist, e))
    return " · ".join(parts)


def _neutralize_paths(text, extra_dirs=()):
    """把**可枚举的**本机绝对路径替换成中性占位（#239②）。

    判据来源：仓工作纪律「公开面路径写相对路径——本机绝对路径以中性占位代
    （仓内 `<仓根>`／家目录 `~/`／临时 `%TEMP%`）」。报告要原样贴到公开 issue，故
    执行核**自己知道**的那几条路径一律替换：落盘临时目录（`<临时目录>`）、执行核
    所在仓根（`<仓根>`）、解释器目录（`<Python>`）。被测输出里它自己探到的**其它**
    绝对路径无法枚举，如实保留——本承诺**不含**它们（模块 docstring 已写明）。

    削掉的判别力：旧 docstring 写「报告中的路径一律中性化」，实测 stdout/stderr
    里的路径原样透出（承诺未兑现）。本函数只兑现**可枚举**的那部分；「一律」一词
    已按实测收窄，不宣称覆盖被测输出自探的路径。
    """
    for d in extra_dirs:
        if d:
            text = (text or "").replace(d, "<临时目录>")
    text = text or ""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo:
        text = text.replace(repo, "<仓根>")
    py_dir = os.path.dirname(os.path.abspath(sys.executable))
    if py_dir:
        text = text.replace(py_dir, "<Python>")
    return text


def _render_report(source, extract_info, cmd_line, exit_line, stdout_txt, stderr_txt,
                   max_bytes, note=None, neutral_dirs=()):
    """固定模板：来源/抽取/环境/被测提交/依赖/执行/退出码 + 原始输出 + 末行固定声明。"""
    parts = []
    parts.append("### 自动复现读数（repro-bot）")
    parts.append("")
    parts.append("> 由 `repro-bot` 在 `run-repro` 标签触发的自动执行——**仅**呈现原始读数。")
    parts.append("")
    parts.append("- **来源**：%s" % source)
    parts.append("- **抽取**：%s" % extract_info)
    parts.append("- **环境**：%s" % _env_line())
    parts.append("- **被测提交**：%s" % _git_head_line())
    parts.append("- **依赖**：%s" % _deps_line())
    parts.append("- **执行**：%s" % cmd_line)
    parts.append("- **退出码**：%s" % exit_line)
    if note:
        parts.append("")
        parts.append("> %s" % note)
    parts.append("")
    for title, body in (("原始输出（stdout）", stdout_txt), ("原始错误（stderr）", stderr_txt)):
        shown, orig_n, cut = _truncate(_neutralize_paths(body, neutral_dirs), max_bytes)
        fence = _md_fence(shown)
        parts.append("#### %s —— 上限 %d B" % (title, max_bytes))
        parts.append("")
        parts.append(fence + "text")
        if shown.strip():
            parts.append(shown.rstrip("\n"))
            if cut:
                parts.append("…（已截断：原始 %d B，仅保留前 %d B）" % (orig_n, max_bytes))
        else:
            parts.append("(空)")
        parts.append(fence)
        parts.append("")
    parts.append("---")
    parts.append(REPORT_FOOTER)
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------

def _selftest():
    """抽取器正负例自检；返回 0 = 全部符合预期。"""
    good = [
        ("前文\n```python\nprint(1)\n```\n后文", "print(1)"),
        ("```py\nx = 1\n```", "x = 1"),
        ("```py3\nx = 2\n```", "x = 2"),
        ("~~~python\nx = 3\n~~~", "x = 3"),
        ("```text\na\n```\n```python\nx = 4\n```", "x = 4"),        # 跳过非 python 块
        ("```python\nx = 5\n```\n```python\nx = 6\n```", "x = 5"),  # 取第一个 python 块
        ("````text\n```\n````\n```Python\nx = 7\n```", "x = 7"),    # 4 反引号包裹 + 大写 info
        ("  ```python\nx = 8\n  ```", "x = 8"),                     # 缩进围栏（0 缩进代码）
        ("行内 `code` 不成块\n```python\nx = 9\n```", "x = 9"),      # 行内代码不干扰
        ("```python\na = 1\n```js\nb = 2\n```", "a = 1\n```js\nb = 2"),  # 围栏内 ```js 不闭合
        ("```python\nx = 10\n（无闭合）", "x = 10\n（无闭合）"),       # 未闭合收编到文末
        # #362① 缩进围栏：块内代码行须剥掉开启围栏的缩进量（否则 IndentationError）
        ("  ```python\n  x = 11\n  ```", "x = 11"),
        ("   ```python\n   if True:\n       y = 1\n   ```", "if True:\n    y = 1"),
        # #362② 引用块围栏：剥掉 `>` 标记后按围栏判（含嵌套）
        ("> ```python\n> x = 12\n> ```", "x = 12"),
        ("> 说明\n> ```python\n> x = 13\n> ```\n> 尾", "x = 13"),
        ("> > ```python\n> > x = 14\n> > ```", "x = 14"),
        # #612 嵌套容器：嵌套列表项（形态 B）/列表套块引用（形态 A）内的 ⑥ 型
        # HTML 起点不再漏判——隐藏块被吞，chosen 落到正文可见块
        ("- 场景说明\n  - <div title=\"\n    ```python\n    print('HIDDEN')\n    ```\n"
         "    \"></div>\n\n```python\nprint('VISIBLE')\n```", "print('VISIBLE')"),
        ("- > <div title=\"\n  > ```python\n  > print('HIDDEN2')\n  > ```\n"
         "  > \"></div>\n\n```python\nprint('VISIBLE2')\n```", "print('VISIBLE2')"),
    ]
    fails = 0
    for text, want in good:
        blocks = _extract_fenced_blocks(text)
        chosen = _choose_python_block(blocks)
        got = chosen[3].strip("\n") if chosen else None
        if got != want:
            print("  SELFTEST-FAIL 应抽到 %r：实得 %r（text=%r）" % (want, got, text))
            fails += 1
    bad = [
        "",                                    # 空正文
        "只有文字，没有围栏",                   # 无围栏
        "```text\n只有 text 块\n```",           # 无 python 块
        "```\n无 info 的块\n```",               # info 为空
        "```diff\n-a\n+b\n```",                # 只有 diff 块
        "> 只是引用文字，没有围栏",              # 引用块但不含围栏
        "> ```text\n> 引用块里的非 python 块\n> ```",  # 引用块里只有 text 块
    ]
    for text in bad:
        chosen = _choose_python_block(_extract_fenced_blocks(text))
        if chosen is not None:
            print("  SELFTEST-FAIL 应抽不到（None）：实得 %r（text=%r）" % (chosen[3], text))
            fails += 1
    # 行号口径：代码行范围 1-based
    b = _choose_python_block(_extract_fenced_blocks("a\n```python\nx\n```\n"))
    if not b or (b[1], b[2]) != (3, 3):
        print("  SELFTEST-FAIL 行号口径应为 (3,3)：实得 %r" % (b,))
        fails += 1
    # 截断口径：不切半个多字节字符
    cut, orig, flag = _truncate("中" * 100, 10)
    if not flag or orig != 300 or len(cut.encode("utf-8")) > 10:
        print("  SELFTEST-FAIL 截断口径：%r %d %s" % (cut, orig, flag))
        fails += 1
    # #362② 反向：非引用块的代码里行首 `>` 是字面量，不得被剥掉
    fb = _choose_python_block(_extract_fenced_blocks("```python\n> literal\n```"))
    if not fb or fb[3] != "> literal":
        print("  SELFTEST-FAIL 非引用块里代码行的 `>` 被误剥：%r" % (fb,))
        fails += 1
    # #239① 报告须带被测提交与依赖版本两行
    rep = _render_report("src", "extract", "cmd", "0", "out", "err", 1024)
    for field in ("- **被测提交**：", "- **依赖**："):
        if field not in rep:
            print("  SELFTEST-FAIL 报告缺 %r 行" % field)
            fails += 1
    if "numpy" not in rep:
        print("  SELFTEST-FAIL 依赖行未记 numpy 版本")
        fails += 1
    # #239② 可枚举路径须中性化
    if _neutralize_paths("写进 " + tempfile.gettempdir(), ("%s" % tempfile.gettempdir(),)) != "写进 <临时目录>":
        print("  SELFTEST-FAIL 路径中性化未生效")
        fails += 1
    print("selftest: %s（正例 %d · 负例 %d · 附加 6）"
          % ("OK" if fails == 0 else "%d 例不符预期" % fails, len(good), len(bad)))
    return 0 if fails == 0 else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _usage(msg):
    print("用法错误：%s（见 --help）" % msg, file=sys.stderr)
    return 2


def main(argv):
    _force_utf8_stdio()
    args = list(argv[1:])
    if any(a in ("-h", "--help") for a in args):
        print(__doc__)
        return 0
    if "--selftest" in args:
        return _selftest()

    opt = {
        "repo": None, "issue": None, "body_file": None, "source_desc": None,
        "timeout": DEFAULT_TIMEOUT, "max_bytes": DEFAULT_MAX_OUTPUT_BYTES,
        "out_dir": None, "report_file": None, "dry_run": False, "env": [],
    }
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--dry-run":
            opt["dry_run"] = True
            i += 1
            continue
        if a in ("--repo", "--issue", "--body-file", "--source-desc", "--timeout",
                 "--max-output-bytes", "--out-dir", "--env", "--report-file"):
            if i + 1 >= len(args):
                return _usage("%s 后缺参数" % a)
            val = args[i + 1]
            i += 2
            if a == "--repo":
                opt["repo"] = val
            elif a == "--issue":
                if not val.isdigit():
                    return _usage("--issue 应为数字：%r" % val)
                opt["issue"] = val
            elif a == "--body-file":
                opt["body_file"] = val
            elif a == "--source-desc":
                opt["source_desc"] = val
            elif a == "--timeout":
                if not val.isdigit() or int(val) <= 0:
                    return _usage("--timeout 应为正整数秒：%r" % val)
                opt["timeout"] = int(val)
            elif a == "--max-output-bytes":
                if not val.isdigit() or int(val) <= 0:
                    return _usage("--max-output-bytes 应为正整数：%r" % val)
                opt["max_bytes"] = int(val)
            elif a == "--out-dir":
                opt["out_dir"] = val
            elif a == "--env":
                if val:
                    opt["env"].append(val)
            elif a == "--report-file":
                opt["report_file"] = val
            continue
        return _usage("未知参数 %r" % a)

    # ── 输入互斥校验 ──
    if opt["body_file"]:
        if opt["repo"] or opt["issue"]:
            return _usage("--body-file 与 --repo/--issue 互斥")
    elif not (opt["repo"] and opt["issue"]):
        return _usage("需给 --body-file，或 --repo 与 --issue 配套")

    # 报告中性化面（#239②）：落盘目录在抽取成功后才定，此处先留可变列表。
    neutral_dirs = []

    # ── 取正文 ──
    if opt["body_file"]:
        try:
            with open(opt["body_file"], encoding="utf-8", errors="replace") as f:
                body = f.read()
        except OSError as e:
            print("FAIL：读正文文件失败：%s" % e, file=sys.stderr)
            return 2
        source = opt["source_desc"] or ("本地文件 `%s`" % os.path.basename(opt["body_file"]))
    else:
        try:
            body = _fetch_issue_body(opt["repo"], opt["issue"])
        except RuntimeError as e:
            report = _render_report(
                source="`%s#%s`（gh api）" % (opt["repo"], opt["issue"]),
                extract_info="未抽取（取正文失败）",
                cmd_line="未执行（取正文失败）",
                exit_line="—",
                stdout_txt="", stderr_txt=str(e),
                max_bytes=opt["max_bytes"],
                note="repro-bot 未能取到 issue 正文，未执行任何代码。")
            if opt["report_file"]:
                with open(opt["report_file"], "w", encoding="utf-8", newline="\n") as f:
                    f.write(report)
            print(report)
            return 2
        source = opt["source_desc"] or ("`%s#%s`" % (opt["repo"], opt["issue"]))

    # ── 抽取 ──
    blocks = _extract_fenced_blocks(body)
    tally = {}
    for b in blocks:
        tally[b[0] or "(无标注)"] = tally.get(b[0] or "(无标注)", 0) + 1
    # 块语言来自正文 info string（攻击者可控）⇒ 一律包进行内代码再入报告（#363）。
    tally_line = " / ".join("%s×%d" % (_inline_code(k), v) for k, v in tally.items()) or "无"

    def _fail_report(extract_info, note, stderr_txt=""):
        report = _render_report(source=source, extract_info=extract_info,
                                cmd_line="未执行（抽取失败）", exit_line="—",
                                stdout_txt="", stderr_txt=stderr_txt,
                                max_bytes=opt["max_bytes"], note=note,
                                neutral_dirs=neutral_dirs)
        if opt["report_file"]:
            with open(opt["report_file"], "w", encoding="utf-8", newline="\n") as f:
                f.write(report)
        print(report)
        return 2

    if not blocks:
        return _fail_report("未抽取（正文没有任何围栏代码块）",
                            "issue 正文里没有围栏代码块，无可执行内容；未执行任何代码。")
    chosen = _choose_python_block(blocks)
    if chosen is None:
        return _fail_report("未抽取（正文无 python 围栏块；块语言：%s）" % tally_line,
                            "issue 正文里没有 info 标注为 python/py/py3 的围栏代码块；"
                            "未执行任何代码。请报告人补充 python 复现脚本。")
    lang, first, last, code = chosen
    nlines = code.count("\n") + 1 if code else 0
    extract_info = ("第 %d 个围栏块（正文第 %d–%d 行的 python 块，共 %d 行；"
                    "正文围栏块 %d 个：%s）"
                    % (blocks.index(chosen) + 1, first, last, nlines, len(blocks), tally_line))

    # ── 落盘（独立临时目录） ──
    out_dir = opt["out_dir"] or tempfile.mkdtemp(prefix="repro_issue_")
    os.makedirs(out_dir, exist_ok=True)
    neutral_dirs.append(out_dir)          # 被测输出里的落盘目录一并中性化（#239②）
    script_name = "repro_issue_%s.py" % opt["issue"] if opt["issue"] else "repro_issue_local.py"
    script_path = os.path.join(out_dir, script_name)
    with open(script_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(code + "\n")

    cmd_line = ("`python -X utf8 %s`（cwd=`<临时目录>`；独立临时目录，不写仓内；"
                "超时 %d s；脚本内容=抽出的代码块，围栏缩进量与引用块标记已按 "
                "CommonMark 剥除、其余逐字未改）"
                % (script_name, opt["timeout"]))

    if opt["dry_run"]:
        print("DRY-RUN：已抽取并落盘")
        print("  " + extract_info)
        print("  " + cmd_line.replace("`", ""))
        print("  note：脚本只写进临时目录：%s" % out_dir)
        return 0

    # ── 限时执行 ──
    extra_env = {}
    for kv in opt["env"]:
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        if k and v:
            extra_env[k] = v
    status, exit_line, out, err = _run_script(script_path, out_dir, opt["timeout"], extra_env)

    note = None
    if status != 0:
        note = ("被测脚本未正常结束（%s）。读数照贴、不作结论——"
                "脚本可能依赖运行方预置（如变量/环境），或与当前 HEAD 行为不符。" % exit_line)
    report = _render_report(source=source, extract_info=extract_info, cmd_line=cmd_line,
                            exit_line=exit_line, stdout_txt=out, stderr_txt=err,
                            max_bytes=opt["max_bytes"], note=note,
                            neutral_dirs=neutral_dirs)
    if opt["report_file"]:
        with open(opt["report_file"], "w", encoding="utf-8", newline="\n") as f:
            f.write(report)
    print(report)
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv))
