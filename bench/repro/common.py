# -*- coding: utf-8 -*-
"""复现脚本公用部分：路径、数据加载、hive-memory-bench 检索机械层口径。

口径（自建，与 hmb 仓库自带脚本不同，只用于同一口径内的前后对比）：
  切A = 第 1–8 章（111 块），切B = 第 1–16 章（203 块）；块 = 章内自然段依次合并到 ≥150 字。
  金标 = 题卡 supporting_evidence / evidence_pool 中落在切内的引文（A 58 题 301 条，B 84 题 467 条）。
  证据保留 = 引文所在块进 top-10 的比例；块级命中 = 引文所在章有块进 top-10 的比例；
  另一侧覆盖 = 与题面最像的引文块 + 异章里最不像的引文块同时进 top-10 的题数。
"""
import glob, json, os, random, re, sys, tempfile, time, warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(HERE, "data")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import yaml  # noqa: E402  (pyyaml；仓库测试依赖已含)

from lingshu.core.core import SpacetimeMemoryEngine  # noqa: E402

_CJK = re.compile(r"[^\u4e00-\u9fff0-9A-Za-z]")


def need_data():
    if not os.path.isdir(os.path.join(DATA, "hmb", "cards")):
        sys.exit("缺少语料：先运行 python bench/repro/fetch_data.py")


def norm(s):
    return _CJK.sub("", s or "")


def bigrams(s):
    s = norm(s).lower()
    return [s[i:i + 2] for i in range(len(s) - 1)]


def jac(a, b):
    A, B = set(bigrams(a)), set(bigrams(b))
    return len(A & B) / len(A | B) if A and B else 0.0


_chs = None


def chapters():
    global _chs
    if _chs is None:
        need_data()
        _chs = []
        for i in (1, 2, 3):
            _chs += json.load(open(os.path.join(DATA, "hmb", "corpus", f"u0{i}.json"), encoding="utf-8"))["chapters"]
    return _chs


def chunks_for(maxch, minlen=150):
    out = []
    for ch in chapters():
        if ch["chapter_no"] > maxch:
            continue
        buf = ""
        for p in ch["text"].split("\n"):
            p = p.strip()
            if not p:
                continue
            buf = (buf + "\n" + p) if buf else p
            if len(buf) >= minlen:
                out.append((ch["cid"], ch["chapter_no"], buf))
                buf = ""
        if buf:
            out.append((ch["cid"], ch["chapter_no"], buf))
    return out


def _locate(quote, chunks):
    q = norm(quote)
    if not q:
        return None
    for i, (_, _, t) in enumerate(chunks):
        if q in norm(t):
            return i
    n = min(10, len(q))
    for i, (_, _, t) in enumerate(chunks):
        nt = norm(t)
        if any(q[j:j + n] in nt for j in range(0, len(q) - n + 1)):
            return i
    return None


def build_set(maxch):
    """返回 (chunks, questions)；maxch=8 为切A，16 为切B，99 为全书。"""
    chunks = chunks_for(maxch)
    cids = {c["cid"] for c in chapters() if c["chapter_no"] <= maxch}
    qs = []
    for f in sorted(glob.glob(os.path.join(DATA, "hmb", "cards", "*.yaml"))):
        c = yaml.safe_load(open(f, encoding="utf-8"))
        ev = c.get("supporting_evidence") or c.get("evidence_pool") or []
        evs = []
        for e in ev:
            if e.get("cid") in cids:
                i = _locate(e["quote"], chunks)
                if i is not None:
                    evs.append((e["quote"], e["cid"], i))
        if not evs:
            continue
        srt = sorted(evs, key=lambda e: -jac(c["question"], e[0]))
        near = srt[0]
        far = next((e for e in reversed(srt) if e[1] != near[1]), None)
        qs.append(dict(qid=c["qid"], q=c["question"], ev=evs, pair=(near[2], far[2]) if far else None))
    return chunks, qs


def evaluate(chunks, qs, fn):
    ret = tot = chap = pc = pn = 0
    for x in qs:
        top = fn(x["q"])[:10]
        S = set(top)
        Ch = {chunks[i][0] for i in top}
        for (_, cid, i) in x["ev"]:
            tot += 1
            ret += i in S
            chap += cid in Ch
        if x["pair"]:
            pn += 1
            pc += x["pair"][0] in S and x["pair"][1] in S
    return {"证据保留": round(100 * ret / tot, 1), "另一侧覆盖": f"{pc}/{pn}",
            "块级命中": round(100 * chap / tot, 1), "题数": len(qs), "证据数": tot}


def kdconv(n, seed=7):
    need_data()
    u = [json.loads(l)["t"] for l in open(os.path.join(DATA, "kdconv_utts.jsonl"), encoding="utf-8")]
    random.Random(seed).shuffle(u)
    return u[:n]


def locomo():
    need_data()
    d = os.path.join(DATA, "locomo")
    corpus = [json.loads(l) for l in open(os.path.join(d, "corpus567.jsonl"), encoding="utf-8")]
    qs = [json.loads(l) for l in open(os.path.join(d, "questions500.jsonl"), encoding="utf-8")]
    return corpus, qs


def engine():
    return SpacetimeMemoryEngine(os.path.join(tempfile.mkdtemp(), "b.db"))


def flags():
    keys = ("LINGSHU_FTS", "LINGSHU_FTS_CONTEXT", "LINGSHU_RETRIEVAL_NO_TOUCH",
            "LINGSHU_CONTRADICTION", "LINGSHU_TMS")
    return {k.replace("LINGSHU_", "").lower(): os.environ.get(k, "0") for k in keys}


def pct(a, p):
    a = sorted(a)
    return a[min(len(a) - 1, int(len(a) * p))]


def emit(d):
    d.update(flags=flags())
    print(json.dumps(d, ensure_ascii=False), flush=True)


__all__ = ["build_set", "evaluate", "kdconv", "locomo", "engine", "emit", "pct", "time", "random", "os", "json"]
