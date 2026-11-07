# -*- coding: utf-8 -*-
"""自主发现 · add_path points 参数可读化（脚本式）。"""
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


def _sim():
    from lingshu.world.scene_simulator import SceneSimulator
    s = SceneSimulator(size=24, ground_level=1)
    s.create_scene(trees=0, water=False)
    return s


def _err(fn, exc):
    try:
        fn()
    except exc as ex:
        return str(ex)
    except Exception as ex:
        return "__WRONG__%s:%s" % (type(ex).__name__, ex)
    return None


def _assertions():
    s = _sim()
    msg = _err(lambda: s.add_path("p", "x"), TypeError)
    _check("A01 非数值 points -> 可读 TypeError（含 points）",
           msg is not None and not msg.startswith("__WRONG__") and "points" in msg, msg)
    s2 = _sim()
    msg2 = _err(lambda: s2.add_path("p", [(1.0, 1.5, 1.0), "bad"]), TypeError)
    _check("A02 单点非数值 -> 同一个可读 TypeError",
           msg2 is not None and not msg2.startswith("__WRONG__") and "points" in msg2, msg2)
    s3 = _sim()
    s3.add_path("patrol", [(1.0, 1.5, 1.0), (2.0, 1.5, 2.0)])
    _check("A03 正对照：合法点列正常注册", list(s3.paths) == ["patrol"] and len(s3.paths["patrol"]) == 2, list(s3.paths))

    text = (_REPO / "lingshu/world/scene_simulator.py").read_text(encoding="utf-8")
    _check("E01 钉子：可读文案", "points 必须是 [(x, y, z), ...] 的实数点列" in text)


def _run_assertions():
    global FAILS
    FAILS = []
    del _CHECKS[:]
    _assertions()
    print("通过 %d / 失败 %d" % (len(_CHECKS) - len(FAILS), len(FAILS)))
    return len(FAILS)


_ARMS = {
    "M1": ("lingshu/world/scene_simulator.py",
           [('            raise TypeError("points 必须是 [(x, y, z), ...] 的实数点列，收到 %r" % (points,)) from exc',
             '            raise ValueError("points 必须是 [(x, y, z), ...] 的实数点列，收到 %r" % (points,)) from exc')],
           {"A01", "A02"}),
}


def _subprocess_fail_names():
    for cache in _REPO.rglob("__pycache__"):
        for pyc in cache.glob("scene_simulator.*.pyc"):
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
            print("[ANCHOR-MISS] %s" % mid)
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
                    print("[ANCHOR-MISS] %s: %d" % (mid, cnt))
                    return 2
                norm = norm.replace(old, new)
            target.write_bytes(norm.replace("\n", nl).encode("utf-8"))
            actual, out = _subprocess_fail_names()
            if actual == expect:
                print("[MUTATE-OK]   %s：%s" % (mid, sorted(actual)))
            else:
                print("[MUTATE-FAIL] %s：期望 %s 实际 %s" % (mid, sorted(expect), sorted(actual)))
                red += 1
        finally:
            target.write_bytes(orig)
    print("反证：%d 条，%s" % (len(ids), "全部命中" if red == 0 else "%d 未命中" % red))
    return 1 if red else 0


if __name__ == "__main__":
    _f = _run_assertions()
    if "--assert-only" in sys.argv:
        sys.exit(1 if _f else 0)
    _m = _mutate(sys.argv[1:])
    sys.exit(1 if (_f or _m) else 0)
