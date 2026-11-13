# -*- coding: utf-8 -*-
"""自主发现 · scene_simulator.add_entity 速度值域守卫（脚本式）。

缺陷（origin/main @ 91cb152）：speed=-1 被接受，实体反向移动（(6,6)->(5.72,6.96)），
并会污染 world_learner.speed_est。契约：speed 非负有限；越界 ValueError 且不注册。
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
    msg = _err(lambda: s.add_entity(category="cat", behavior="wander",
                                    pos=(6.0, 1.5, 6.0), speed=-1.0, goal=""), ValueError)
    _check("A01 负速度拒绝且不注册", 
           msg is not None and not msg.startswith("__WRONG__") and len(s.entities) == 0, msg)

    s2 = _sim()
    msg2 = _err(lambda: s2.add_entity(category="cat", behavior="wander",
                                      pos=(6.0, 1.5, 6.0), speed=float("nan"), goal=""), ValueError)
    _check("A02 NaN 速度拒绝", msg2 is not None and not msg2.startswith("__WRONG__"), msg2)

    s3 = _sim()
    eid = s3.add_entity(category="cat", behavior="wander",
                        pos=(6.0, 1.5, 6.0), speed=0.3, goal="")
    ent = s3.entities[eid]
    p0 = ent.pos
    s3.step(1)
    _check("A03 正对照：0.3 正常注册并移动", len(s3.entities) == 1 and ent.pos != p0, (p0, ent.pos))

    s4 = _sim()
    eid4 = s4.add_entity(category="cat", behavior="wander",
                         pos=(6.0, 1.5, 6.0), speed=0.0, goal="")
    s4.step(1)
    _check("A04 合法 0（静止）接受且不动",
           s4.entities[eid4].pos == (6.0, 1.5, 6.0), s4.entities[eid4].pos)

    text = (_REPO / "lingshu/world/scene_simulator.py").read_text(encoding="utf-8")
    _check("E01 钉子：可读拒绝文案", "speed 必须是非负有限数" in text)
    _check("E02 钉子：非有限与负值同闸",
           "not math.isfinite(_speed) or _speed < 0.0" in text)


def _run_assertions():
    global FAILS
    FAILS = []
    del _CHECKS[:]
    _assertions()
    print("通过 %d / 失败 %d" % (len(_CHECKS) - len(FAILS), len(FAILS)))
    return len(FAILS)


_ARMS = {
    "M1": ("lingshu/world/scene_simulator.py",
           [("        if not math.isfinite(_speed) or _speed < 0.0:", "        if False:")],
           {"A01", "A02", "E02"}),
    "M2": ("lingshu/world/scene_simulator.py",
           [("        if not math.isfinite(_speed) or _speed < 0.0:", "        if _speed < 0.0:")],
           {"A02", "E02"}),
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
