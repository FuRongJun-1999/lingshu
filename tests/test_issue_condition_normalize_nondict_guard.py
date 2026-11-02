# -*- coding: utf-8 -*-
"""#自主发现 · condition_normalize 非字典输入契约守卫（脚本式，无 test_* 函数）。

缺陷（修复前实测，origin/main @ 91cb152）：
  · canonical_condition(None) -> TypeError: argument of type 'NoneType' is not iterable
    （不可读；且连带 cond_hash / validate_edges / canonical_sourced 全崩）；
  · canonical_condition(["a"]) -> 静默返回全 __UNASSERTED__（list 的 in 是逐元素
    比较，被当成空映射）——非法容器被静默改写成「未声明」，与 #1063②
    「类型不对应落 INVALID/报出，不静默」相反；
  · invalid_domains(None) -> AttributeError（'NoneType' object has no attribute 'get'）。

契约（模块头 domains: Dict[str, str]）：None = 无条件声明（与空 dict 同义，
P2-001「未声明无可校验」）；其余非映射类型是调用方 bug，显式 TypeError 说明期望类型。

判据来源：模块内 #138 / #1063 既有口径（类型不对必须可被调用方看见，不得静默折叠）
+ 本机 origin/main 实跑读数。
"""
import re
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_CHECKS = []
FAILS = []


def _check(name, ok, detail=""):
    _CHECKS.append(name)
    if ok:
        print("  PASS %s" % name)
    else:
        FAILS.append(name)
        print("  FAIL %s  %s" % (name, detail))


def _safe(fn):
    try:
        return fn()
    except Exception as exc:   # noqa: BLE001 - 断言用：把异常转成可判定值
        return "__EXC__%s" % type(exc).__name__


def _raises(fn, exc):
    try:
        fn()
    except exc as e:
        return str(e)
    except Exception as e:      # noqa: BLE001 - 断言用：错误类型必须精确
        return "__WRONG__%s:%s" % (type(e).__name__, e)
    return None



def _assertions():
    from lingshu.core import condition_normalize as CN

    empty = CN.canonical_condition({})
    none_canon = _safe(lambda: CN.canonical_condition(None))
    _check("A01 None 与空 dict 同义（全未声明）",
           none_canon == empty
           and set(empty.values()) == {CN.UNASSERTED}
           and len(empty) == len(CN.IMAGE_DOMAINS),
           none_canon)

    msg = _raises(lambda: CN.canonical_condition(["a"]), TypeError)
    _check("A02 非映射容器必须显式 TypeError（不静默当空映射）",
           msg is not None and not msg.startswith("__WRONG__")
           and ("映射" in msg or "Mapping" in msg),
           msg)

    _check("A03 cond_hash(None) == cond_hash({})",
           _safe(lambda: CN.cond_hash(None)) == CN.cond_hash({}))

    _check("A04 validate_edges(None) 与空 dict 同结论",
           _safe(lambda: CN.validate_edges(None)) == CN.validate_edges({})
           and CN.validate_edges({})["violations"] == [])

    _check("A05 invalid_domains(None) 为空（不崩溃）",
           _safe(lambda: CN.invalid_domains(None)) == {})

    src = _safe(lambda: CN.canonical_sourced(None))
    _check("A06 canonical_sourced(None) 全未声明且三字段齐",
           isinstance(src, dict) and len(src) == len(CN.IMAGE_DOMAINS)
           and src["style"]["value"] == CN.UNASSERTED
           and src["style"]["source"] == "asserted",
           src.get("style") if isinstance(src, dict) else src)

    from collections import UserDict
    ud = UserDict({"style": "可爱动漫"})
    _check("B01 映射子类（UserDict）按契约归一",
           CN.canonical_condition(ud)["style"] == "动漫"
           and CN.canonical_condition(ud)["pose"] == CN.UNASSERTED)

    text = (_REPO / "lingshu/core/condition_normalize.py").read_text(encoding="utf-8")
    _check("E01 钉子：非映射类型闸（isinstance Mapping）",
           "isinstance(domains, Mapping)" in text)
    _check("E02 钉子：None 等价空 dict", "if domains is None:" in text)
    _check("E03 钉子：invalid_domains 用 None 安全的 source",
           "source = domains if isinstance(domains, Mapping) else {}" in text)


def _run_assertions():
    global FAILS
    FAILS = []
    del _CHECKS[:]
    _assertions()
    print("通过 %d / 失败 %d" % (len(_CHECKS) - len(FAILS), len(FAILS)))
    return len(FAILS)

_ARMS = {
    "M1": ("lingshu/core/condition_normalize.py",
           [("    if domains is None:\n        domains = {}\n    elif not isinstance(domains, Mapping):",
             "    if not isinstance(domains, Mapping):")],
           {"A01", "A03", "A04", "A05", "A06", "E02"}),
    "M2": ("lingshu/core/condition_normalize.py",
           [('        raise TypeError("domains 必须是映射（dict/Mapping），收到 %s"\n'
             '                        % type(domains).__name__)',
             "        domains = {}")],
           {"A02"}),
    "M3": ("lingshu/core/condition_normalize.py",
           [("    source = domains if isinstance(domains, Mapping) else {}",
             "    source = domains")],
           {"E03"}),
}


def _subprocess_fail_names():
    for cache in _REPO.rglob("__pycache__"):
        for pyc in cache.glob("condition_normalize.*.pyc"):
            pyc.unlink(missing_ok=True)
    r = subprocess.run([sys.executable, "-X", "utf8", __file__, "--assert-only"],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=str(_REPO), timeout=900)
    out = (r.stdout or "") + (r.stderr or "")
    return set(re.findall(r"(?m)^\s*FAIL\s+([A-Z]\d+)", out)), out


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
