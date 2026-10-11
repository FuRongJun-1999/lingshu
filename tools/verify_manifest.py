#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""verify_manifest.py —— 核验 docs/intake/*/_export_manifest.json 的逐件登记指纹。

用途
    灵枢仓库的 intake 导出清单（`docs/intake/*/_export_manifest.json`）里逐件登记了
    文件 sha256，但仓库原先**没有任何工具**能核验这些登记。本工具遍历清单、实算
    盘上文件的 sha256、按规则逐条判定，并给出可机器消费的退出码。

登记基准（LF 归一 · 与检出无关）
    **登记基准 = LF 归一后的字节，与检出 CRLF/LF 无关。**
    实算 sha256（与 `bytes` 大小）前，先把读到的字节里所有 `\\r\\n` 归一为 `\\n`
    （孤立 `\\r` 不动）。原因：登记件是文本、上游真源在 Linux 侧，而本仓
    `core.autocrlf=true` 会让 Windows 工作树落成 CRLF、CI（Linux）检出成 LF——
    若按「盘上现文件字节」记录，同一份登记在两种检出下实算值不同，CI 必红。
    归一后同一份登记在 CRLF 工作树与 LF 检出上给出**同一个**实算值（`--json`
    里的 `actual_sha256` / `actual_bytes` 同步为该口径），登记才与平台无关。
    这也是 `bytes` 字段的口径：**LF 归一后的字节数**（不是 `stat` 的盘上大小）。

用法
    python tools/verify_manifest.py [--repo-root PATH] [--json] [--strict] [--quiet]

    --repo-root PATH   仓库根；缺省 = 本脚本上两级目录（即 <repo>/tools/verify_manifest.py
                       的 <repo>）。在任何 cwd 下都可显式指定。
    --json             输出单个 JSON 对象（stdout），供 CI/脚本消费。
    --strict           出现 MISMATCH 时退出 3（缺省时 MISMATCH 不算失败，见下）。
    --quiet            文本模式只打印末行总表（不打印每清单小节与 bytes 样例）。

路径解析规则（两批清单都支持，取自冻结规格）
    - 清单含 `blocks`（v0.1 形态）：条目路径 = `lingshu/<block>/<file>`
    - 否则（v0.2 形态）：条目路径 = `file`（已是仓内相对路径），
      条目取自 `artifacts` 与 `tests` 两个数组
    - 输出路径一律规范化：折叠内部 `/./` 与重复斜杠；**不**折叠 `..`
      （折叠 `..` 会改变「是否越界」的语义；越界由下面的边界校验负责）。

逐条判定
    MISSING    文件不在盘
    MALFORMED  `sha256` 缺失、非小写 hex、或长度不在 {16, 64}；
               或登记路径本身不可信 —— 绝对路径（reason `path-absolute`）、
               空路径（`path-empty`）、解析后越出仓库根（`path-outside-repo`）。
               **绝对路径条目在输出里一律脱敏为 `<redacted:absolute-path>`**，
               因此 `--json` 里永不出现本机绝对路径（旧版只有「真实清单恰好不含
               绝对路径」才成立，现已在输入可控时也成立），也堵住「清单把工具
               当任意文件读取器」去读仓外文件。
    MATCH/MISMATCH  实算 sha256（**LF 归一后**，见上「登记基准」）与记录值做
               **同长度前缀**比较（记录 16 位 → 比实算前 16 位；记录 64 位 → 全比）；
               `MATCH` 有正例测试（tests/test_manifest_integrity.py::test_9）。
    `bytes` 存在时记录**LF 归一后**的字节数与差值，仅作信息，不影响判定。

    判定优先级（四类计数是条目集的一个划分）：
    绝对路径 → MALFORMED ＞ 越界路径 → MALFORMED ＞ 文件不在盘 → MISSING ＞
    指纹形状 → MALFORMED ＞ MATCH / MISMATCH。

退出码（四码 · 规格作者确认）
    0  成功：所有清单可解析，且无 MISSING、无 MALFORMED；非 strict 下 MISMATCH
       **不算失败**（清单未声明指纹指向哪一侧，属既有事实）。`--strict` 且零 MISMATCH
       同样是 0。
    1  结构性 / 内容失败：存在 MISSING / MALFORMED / 清单不可解析 / **清单结构损坏**
       （容器类型错、条目非对象、缺 `file` 或 `file` 非字符串——均记入 `summary` 的
       `unparseable` 一列）；**目录里零清单、或清单在盘却零条目，也是 1**——这是刻意
       语义，不是空真：「没发现任何清单 / 一条也没登记出来」必须与「清单全部通过」
       在退出码上可区分（否则扫描面被清空时会静默变绿）。
    2  用法 / 路径错误：未知参数、多余位置参数、`--repo-root` 不存在或不是目录
       （与同批 `tools/time_core_lint.py` 的 `2` **完全一致**）。
       `_Parser.error` 覆写为退出 2，避免 argparse 默认码与本表冲突。
    3  `--strict` 且存在 MISMATCH：把「口径分歧」（strict 落差）与「结构损坏」（1）
       分开，调用方无需解析 stdout 就能区分两者。`2` 与 `3` 语义互不重叠。

输出
    - 文本模式：每个清单一段（条目数、四类计数、sha256 长度分布、bytes 差值样例）
      + 末行总表。
    - `--json`：`{"manifests":[…], "summary":{…}, "exit_code":n}`，`sort_keys=True`，
      **LF 换行**（`reconfigure(newline="\\n")`，Windows 上不产生 CRLF，跨平台字节一致）。
    - 确定性：不含时间戳、不含绝对路径、列表显式排序；同一仓库两次运行逐字节相同
      （`tests/test_manifest_integrity.py` 有断言）。
    - **失败路径也有 JSON**：`--json` 时用法错误 / `--repo-root` 不可用会输出错误信封
      `{"error":…, "exit_code":n, "manifests":[], "summary":{…零计数…}}`，
      消费方（如 `--json | jq`）不会拿到空输入。信封**不含**任何路径。

边界（如实）
    - 只读：不写任何文件、不联网、不调用 git。只读盘上被登记文件来实算 sha256。
    - EOL 归一**只做 `\\r\\n`→`\\n`**：孤立 `\\r`（老 Mac 行尾）原样保留，
      不改变非文本件（二进制）的字节——它们本就不含 `\\r\\n` 序列。
    - 只处理 `docs/intake/<batch>/_export_manifest.json` 这一 glob；更深嵌套（如
      `docs/intake/x/y/_export_manifest.json`）不扫 ⇒ 该目录下无清单 ⇒ 退出 1
      （同样是上面「零清单 = 1」的刻意语义）。
    - 文本模式同样只用仓内相对路径（绝对路径条目脱敏），不泄露机器绝对路径。
    - `..` 越界条目的**字面相对路径**会保留在输出里（如 `../../../../etc/passwd`）：
      它不是机器绝对路径、不构成路径泄漏（已判 `MALFORMED / path-outside-repo`），
      保留原样便于定位是哪条登记写歪了；只有绝对路径条目脱敏。
    - MISMATCH 计数大不是本工具的缺陷，是既有事实（登记指纹与仓内现文件不一致）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

MANIFEST_GLOB = "docs/intake/*/_export_manifest.json"
HEX_LOWER_RE = re.compile(r"^[0-9a-f]+$")
ABS_ENTRY_RE = re.compile(r"^(?:/|[A-Za-z]:)")
ALLOWED_SHA_LENGTHS = (16, 64)
BYTES_SAMPLE_LIMIT = 5
REDACTED_ABSOLUTE = "<redacted:absolute-path>"

EXIT_OK = 0
EXIT_DEFECT = 1
EXIT_USAGE = 2
EXIT_STRICT_MISMATCH = 3

STATUS_MATCH = "MATCH"
STATUS_MISMATCH = "MISMATCH"
STATUS_MISSING = "MISSING"
STATUS_MALFORMED = "MALFORMED"


class _Parser(argparse.ArgumentParser):
    """用法错误退出 2（与 tools/time_core_lint.py 同口径）；`--json` 时补错误信封。"""

    json_requested = False

    def error(self, message):  # noqa: D102 - argparse hook
        self.print_usage(sys.stderr)
        sys.stderr.write("usage-error: %s\n" % message)
        if self.json_requested:
            emit_json_error("usage-error", EXIT_USAGE)
        raise SystemExit(EXIT_USAGE)


def configure_stdout() -> None:
    """stdout 固定 UTF-8 + LF：Windows 上 `--json` 不产生 CRLF，跨平台字节一致。"""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", newline="\n")
        except (OSError, ValueError, TypeError):  # pragma: no cover - 非标准 stdout
            pass


def empty_summary() -> dict:
    return {"manifests": 0, "entries": 0, "match": 0, "mismatch": 0,
            "missing": 0, "malformed": 0, "unparseable": 0}


def write_json(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    sys.stdout.write("\n")


def emit_json_error(kind: str, exit_code: int) -> None:
    """失败路径的 JSON 信封；只带错误种类与退出码，不含任何路径。"""
    write_json({
        "error": kind,
        "exit_code": exit_code,
        "manifests": [],
        "summary": empty_summary(),
    })


CHUNK = 1 << 20


def lf_digest(path: Path):
    """实算文件的 (LF 归一字节数, sha256 hex)。

    **登记基准 = LF 归一后的字节，与检出 CRLF/LF 无关**：读到的字节里所有
    `\\r\\n` 先归一为 `\\n` 再喂给哈希（孤立 `\\r` 不动）。这样同一份登记在
    Windows（autocrlf=true ⇒ 工作树 CRLF）与 Linux（检出 LF）上实算值相同。

    流式实现（大文件不吃内存），且与「整文件 `replace(b"\\r\\n", b"\\n")`」
    逐字节等价：跨 chunk 的 `\\r\\n` 由 1 字节 carry 处理——chunk 末尾若为 `\\r`
    则留作下一轮前缀，不与下一 chunk 的 `\\n` 被拆散。
    """
    digest = hashlib.sha256()
    size = 0
    carry = b""
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            data = carry + chunk
            if data.endswith(b"\r"):
                carry, data = b"\r", data[:-1]
            else:
                carry = b""
            data = data.replace(b"\r\n", b"\n")
            digest.update(data)
            size += len(data)
    if carry:
        digest.update(carry)
        size += len(carry)
    return size, digest.hexdigest()


def is_unsafe_absolute(raw: str) -> bool:
    """登记值是否为绝对路径（POSIX `/…`、UNC `//…`、Windows `C:…`）。"""
    text = str(raw).strip().replace("\\", "/")
    return bool(ABS_ENTRY_RE.match(text))


def normalize_relpath(raw: str) -> str:
    """规范化仓内相对路径：折叠内部 `/./` 与重复斜杠，保留 `..` 段。"""
    text = str(raw).replace("\\", "/").strip()
    return "/".join(seg for seg in text.split("/") if seg not in ("", "."))


def iter_records(payload):
    """按冻结的路径解析规则产出 `(records, defects)`。

    `records`：`(raw_path, recorded_sha256, recorded_bytes, absolute)` 元组列表。
    `defects`：**结构损坏**的原因串列表（容器类型错 / 条目非对象 / 缺 `file` 或
    `file` 非字符串）。`absolute` 为真时 `raw_path` 只用于内部判定，绝不进输出（输出脱敏）。

    #408：旧实现对上述损坏一律 `continue` **静默跳过** ⇒ 一份被改坏（如 `files`
    成了字符串）的清单被当成「零条目」而 `summary` 无 missing/malformed ⇒ 退出码 0
    （「清单全部通过」）。现在每一处跳过都**记账**为 `defects`，由 `inspect_manifest`
    升级为清单级错误——结构损坏不得被当作零条目，否则扫描面被清空时会静默变绿
    （见模块头「退出码」1 的刻意语义）。
    """
    records = []
    defects = []
    blocks = payload.get("blocks")
    if blocks is not None and not isinstance(blocks, dict):
        defects.append("blocks 不是对象（%s）" % type(blocks).__name__)
    if isinstance(blocks, dict):
        for block in sorted(blocks):
            node = blocks[block]
            files = node.get("files") if isinstance(node, dict) else None
            if not isinstance(files, list):
                defects.append("blocks.%s.files 不是数组" % block)
                continue
            for idx, item in enumerate(files):
                if not isinstance(item, dict):
                    defects.append("blocks.%s.files[%d] 不是对象" % (block, idx))
                    continue
                name = item.get("file")
                if not isinstance(name, str) or not name:
                    defects.append("blocks.%s.files[%d].file 缺失或非字符串"
                                   % (block, idx))
                    continue
                records.append(("lingshu/%s/%s" % (block, name), item.get("sha256"),
                                item.get("bytes"), is_unsafe_absolute(name)))
        return records, defects

    for key in ("artifacts", "tests"):
        array = payload.get(key)
        if array is None:
            continue
        if not isinstance(array, list):
            defects.append("%s 不是数组（%s）" % (key, type(array).__name__))
            continue
        for idx, item in enumerate(array):
            if not isinstance(item, dict):
                defects.append("%s[%d] 不是对象" % (key, idx))
                continue
            name = item.get("file")
            if not isinstance(name, str) or not name:
                defects.append("%s[%d].file 缺失或非字符串" % (key, idx))
                continue
            records.append((name, item.get("sha256"), item.get("bytes"),
                            is_unsafe_absolute(name)))
    return records, defects


def classify(root: Path, root_resolved: Path, raw_path: str, recorded_sha,
             recorded_bytes, absolute_entry: bool) -> dict:
    """判定单条登记，返回确定性 detail 字典（绝不写入绝对路径）。"""
    detail = {
        "path": REDACTED_ABSOLUTE if absolute_entry else normalize_relpath(raw_path),
        "status": None,
        "reason": None,
        "recorded_sha256": recorded_sha if isinstance(recorded_sha, str) else None,
        "actual_sha256": None,
        "recorded_bytes": recorded_bytes if isinstance(recorded_bytes, int) else None,
        "actual_bytes": None,
        "bytes_delta": None,
    }

    # ① 绝对路径条目：不 stat、不读取，直接判 MALFORMED 并脱敏
    if absolute_entry:
        detail["status"] = STATUS_MALFORMED
        detail["reason"] = "path-absolute"
        return detail
    # ② 空路径
    if not detail["path"]:
        detail["status"] = STATUS_MALFORMED
        detail["reason"] = "path-empty"
        return detail
    # ③ 越出仓库根（含 `..`）：不读取仓外文件
    target = root / detail["path"]
    try:
        resolved = target.resolve()
    except (OSError, RuntimeError):
        resolved = None
    if resolved is None or not resolved.is_relative_to(root_resolved):
        detail["status"] = STATUS_MALFORMED
        detail["reason"] = "path-outside-repo"
        return detail

    if not target.is_file():
        detail["status"] = STATUS_MISSING
        detail["reason"] = "missing-file"
        return detail

    # 实算一次（LF 归一）：同时得到 bytes 与 sha256 两个实盘量
    actual_bytes, actual = lf_digest(target)
    detail["actual_bytes"] = actual_bytes
    if detail["recorded_bytes"] is not None:
        detail["bytes_delta"] = detail["actual_bytes"] - detail["recorded_bytes"]

    if not isinstance(recorded_sha, str) or not recorded_sha:
        detail["status"] = STATUS_MALFORMED
        detail["reason"] = "sha256-absent"
        return detail
    if not HEX_LOWER_RE.match(recorded_sha):
        detail["status"] = STATUS_MALFORMED
        detail["reason"] = "sha256-not-lowercase-hex"
        return detail
    if len(recorded_sha) not in ALLOWED_SHA_LENGTHS:
        detail["status"] = STATUS_MALFORMED
        detail["reason"] = "sha256-length-%d" % len(recorded_sha)
        return detail

    detail["actual_sha256"] = actual
    if actual[: len(recorded_sha)] == recorded_sha:
        detail["status"] = STATUS_MATCH
    else:
        detail["status"] = STATUS_MISMATCH
    return detail


def inspect_manifest(root: Path, manifest_path: Path) -> dict:
    """读一个清单并产出确定性结果块。"""
    rel = normalize_relpath(manifest_path.relative_to(root).as_posix())
    block = {
        "path": rel,
        "entries": 0,
        "match": 0,
        "mismatch": 0,
        "missing": 0,
        "malformed": 0,
        "sha256_lengths": [],
        "details": [],
    }
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        block["error"] = "unparseable-manifest: %s" % type(exc).__name__
        return block

    if not isinstance(payload, dict):
        block["error"] = "unparseable-manifest: root-not-object"
        return block

    root_resolved = root.resolve()
    records, defects = iter_records(payload)
    details = [classify(root, root_resolved, raw, sha, size, absolute)
               for raw, sha, size, absolute in records]
    details.sort(key=lambda item: item["path"])

    # 长度分布 = 所有「登记值确为字符串」的条目长度集合（与文本模式同一口径）
    lengths = {len(item["recorded_sha256"]) for item in details
               if isinstance(item["recorded_sha256"], str)}

    block["entries"] = len(details)
    block["match"] = sum(1 for item in details if item["status"] == STATUS_MATCH)
    block["mismatch"] = sum(1 for item in details if item["status"] == STATUS_MISMATCH)
    block["missing"] = sum(1 for item in details if item["status"] == STATUS_MISSING)
    block["malformed"] = sum(1 for item in details if item["status"] == STATUS_MALFORMED)
    block["sha256_lengths"] = sorted(lengths)
    block["details"] = details

    # #408：结构损坏（容器类型错 / 条目非对象 / 缺 file）或零条目不得静默当「全通过」。
    # 前者给出损坏位置，后者兜住「清单被清空/容器名写错」这类没有逐条损坏的形态。
    if defects:
        block["error"] = "malformed-manifest: " + "; ".join(defects[:4])
    elif not details:
        block["error"] = "empty-manifest: 清单登记零条目（结构损坏或被清空）"
    return block


def collect(root: Path):
    """遍历所有清单 → (blocks, summary_counts)。"""
    manifest_paths = sorted(root.glob(MANIFEST_GLOB), key=lambda p: p.as_posix())
    blocks = [inspect_manifest(root, path) for path in manifest_paths]
    summary = empty_summary()
    summary["manifests"] = len(blocks)
    for key in ("entries", "match", "mismatch", "missing", "malformed"):
        summary[key] = sum(b[key] for b in blocks)
    summary["unparseable"] = sum(1 for b in blocks if "error" in b)
    return blocks, summary


def decide_exit(summary: dict, strict: bool) -> int:
    """四码退出判定：0 成功 / 1 结构损坏（含零清单、零条目，刻意语义）/ 3 strict 口径分歧。

    用法与路径错误（2）不经过本函数，由 `_Parser.error` 与 `main` 的 repo-root 校验直接返回。
    """
    # #408：`manifests == 0`（零清单）与 `entries == 0`（清单在盘却一条也没登记出来）
    # 都要判 1——否则清单被清空/改坏时 summary 无 missing/malformed，会静默变绿。
    if summary["manifests"] == 0 or summary["entries"] == 0:
        return EXIT_DEFECT
    if summary["unparseable"] or summary["missing"] or summary["malformed"]:
        return EXIT_DEFECT
    if strict and summary["mismatch"]:
        return EXIT_STRICT_MISMATCH
    return EXIT_OK


def _length_distribution(block: dict) -> str:
    counted = {}
    for item in block["details"]:
        recorded = item["recorded_sha256"]
        if isinstance(recorded, str):
            counted[len(recorded)] = counted.get(len(recorded), 0) + 1
    if not counted:
        return "none"
    return ", ".join("%d=%d" % (length, counted[length]) for length in sorted(counted))


def _bytes_lines(block: dict) -> list:
    lines = []
    with_bytes = [item for item in block["details"] if item["recorded_bytes"] is not None]
    deltas = [item["bytes_delta"] for item in with_bytes if item["bytes_delta"] is not None]
    positives = sum(1 for d in deltas if d > 0)
    negatives = sum(1 for d in deltas if d < 0)
    zeros = sum(1 for d in deltas if d == 0)
    lines.append("  bytes: records_with_bytes=%d/%d  delta(+%d/-%d/zero%d)"
                 % (len(with_bytes), block["entries"], positives, negatives, zeros))
    for item in with_bytes[:BYTES_SAMPLE_LIMIT]:
        lines.append("    %s: recorded=%s actual=%s delta=%s"
                     % (item["path"], item["recorded_bytes"],
                        item["actual_bytes"], item["bytes_delta"]))
    return lines


def render_text(blocks, summary, exit_code, quiet) -> str:
    lines = []
    if not quiet:
        for block in blocks:
            lines.append("[manifest] %s" % block["path"])
            if "error" in block:
                lines.append("  ! %s" % block["error"])
                continue
            lines.append("  entries=%d match=%d mismatch=%d missing=%d malformed=%d"
                         % (block["entries"], block["match"], block["mismatch"],
                            block["missing"], block["malformed"]))
            lines.append("  sha256 lengths: %s" % _length_distribution(block))
            lines.extend(_bytes_lines(block))
    lines.append("SUMMARY manifests=%d entries=%d match=%d mismatch=%d missing=%d "
                 "malformed=%d unparseable=%d exit_code=%d"
                 % (summary["manifests"], summary["entries"], summary["match"],
                    summary["mismatch"], summary["missing"], summary["malformed"],
                    summary["unparseable"], exit_code))
    return "\n".join(lines)


def build_parser() -> _Parser:
    parser = _Parser(description="核验 docs/intake/*/_export_manifest.json 的登记指纹")
    parser.add_argument("--repo-root", default=None,
                        help="仓库根；缺省 = 本脚本上两级目录")
    parser.add_argument("--json", action="store_true", help="输出单个 JSON 对象")
    parser.add_argument("--strict", action="store_true",
                        help="存在 MISMATCH 时退出 3")
    parser.add_argument("--quiet", action="store_true",
                        help="文本模式只打印末行总表")
    return parser


def main(argv=None) -> int:
    configure_stdout()

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    parser.json_requested = "--json" in raw_argv
    args = parser.parse_args(raw_argv)

    if args.repo_root:
        root = Path(args.repo_root).expanduser()
    else:
        root = Path(__file__).resolve().parents[1]

    if not root.is_dir():
        sys.stderr.write("repo-root not a directory: %s\n" % root)
        if args.json:
            emit_json_error("repo-root-not-a-directory", EXIT_USAGE)
        return EXIT_USAGE
    root = root.resolve()

    blocks, summary = collect(root)
    exit_code = decide_exit(summary, args.strict)

    if args.json:
        write_json({
            "manifests": blocks,
            "summary": summary,
            "exit_code": exit_code,
        })
    else:
        sys.stdout.write(render_text(blocks, summary, exit_code, args.quiet))
        sys.stdout.write("\n")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
