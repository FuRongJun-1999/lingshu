# -*- coding: utf-8 -*-
"""test_issue196_physics_predict_gate · physics_predict 非有限与防 bool 闸（#196 第十批）
====================================================================================
缺陷形态（改前实测，基线 main d78185c）：
  1. `classify_entity` / `physics_candidates`：`hz = float(tick_hz) if tick_hz else 1.0`
     导致 `tick_hz=0` / `0.0` / `False` 绕过 `if hz <= 0:` 静默回落 1.0 Hz（10 Hz 抛体被误判
     为 `accelerated`，加速度错 100 倍）；`tick_hz=True` 同理按 1.0 Hz 误判；`tick_hz=nan`
     因 `nan <= 0` 恒假绕过阈值闸并在下游吞为 `None`；`tick_hz=inf` 把时间轴压零。
  2. `physics_candidates`：`float(got["model"].get("residual_rms", 1e9)) > MAX_RMS` 在
     `residual_rms=nan` 或 `False` 时比较为 `False`，绕过强证据门槛（#196 经典 NaN 陷阱）。
  3. `physics_candidates` / `wrap_generate`：`horizon=True` 静默按 1 步预测；`horizon=0`/`-1`
     在 `physics_candidates` 产出空列表并使 `wrap_generate` 报 `IndexError`；`wrap_generate`
     读取 `self.tick_hz` 时对 `0`/`False`/`True`/`nan`/`inf` 同样静默放行。

断言组（回退 / 放宽即红）：
  G1 `classify_entity` 对非有限 / bool / 非正 `tick_hz` 与非法 `up`/`window` 抛 `ValueError`
  G2 `physics_candidates` 对非法 `tick_hz`/`horizon`/`dt` 抛 `ValueError`，对 `nan`/`inf`/`bool`
     `residual_rms` fail-closed 返回 `None`
  G3 `wrap_generate` 对非法 `self.tick_hz`/`self.physics_dt`/`horizon` 抛 `ValueError`，对缺省或
     `tick_hz=None` 保持兼容
  G4 正对照：10 Hz 抛体 `classify_entity` / `physics_candidates` / `wrap_generate` 精确还原
     `physics:projectile`，含 `NaN`/`bool` 观测帧返回 `None` 不崩溃
  G5 `--mutate` 定点变异自证：回退原无闸实现后门禁精确转红，且零污染仓内文件

运行（仓根）：
  python -X utf8 tests/test_issue196_physics_predict_gate.py
  python -X utf8 tests/test_issue196_physics_predict_gate.py --mutate
  python -X utf8 -m pytest tests/test_issue196_physics_predict_gate.py -q
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lingshu.world import physics as PH          # noqa: E402
from lingshu.world import physics_predict as PP  # noqa: E402

NAN = float("nan")
INF = float("inf")
NINF = float("-inf")
_HZ = 10.0


def _projectile_history(hz: float = _HZ, n: int = 8):
    hist = []
    for i in range(n):
        t = i / hz
        pos = (4.0 * t, 1.5 + 8.0 * t - 0.5 * PH.DEFAULT_G * t * t, 0.0)
        hist.append({"tick": i, "entities": {"ball": {"pos": pos}}})
    return hist


class _PatchAttr:
    """纯标准库属性临时替换上下文（兼容 Python 3.12 直跑，不依赖 pytest）。"""

    def __init__(self, obj, name: str, value):
        self.obj = obj
        self.name = name
        self.value = value
        self._old = None

    def __enter__(self):
        self._old = getattr(self.obj, self.name)
        setattr(self.obj, self.name, self.value)
        return self

    def __exit__(self, *args):
        setattr(self.obj, self.name, self._old)


class TestClassifyEntityGate(unittest.TestCase):
    """G1：classify_entity 拒绝非有限 / bool / 非正参数。"""

    def test_invalid_tick_hz_rejected(self):
        hist = _projectile_history()
        for bad in (0, 0.0, -1.0, False, True, NAN, INF, NINF, None, "10"):
            with self.subTest(tick_hz=bad):
                with self.assertRaises(ValueError):
                    PP.classify_entity(hist, "ball", tick_hz=bad)

    def test_invalid_up_and_window_rejected(self):
        hist = _projectile_history()
        for bad_up in (True, False, -1, 3, 1.0, NAN):
            with self.subTest(up=bad_up):
                with self.assertRaises(ValueError):
                    PP.classify_entity(hist, "ball", up=bad_up, tick_hz=_HZ)
        for bad_win in (True, False, 1.5, NAN, "6"):
            with self.subTest(window=bad_win):
                with self.assertRaises(ValueError):
                    PP.classify_entity(hist, "ball", window=bad_win, tick_hz=_HZ)


class TestPhysicsCandidatesGate(unittest.TestCase):
    """G2：physics_candidates 入口闸与 residual_rms 强证据有限性闸。"""

    def test_invalid_tick_hz_horizon_dt_rejected(self):
        hist = _projectile_history()
        for bad_hz in (0, 0.0, -2.0, False, True, NAN, INF, NINF):
            with self.subTest(tick_hz=bad_hz):
                with self.assertRaises(ValueError):
                    PP.physics_candidates(hist, "ball", 3, tick_hz=bad_hz)
        for bad_h in (0, -1, False, True, 1.5, NAN, INF, None):
            with self.subTest(horizon=bad_h):
                with self.assertRaises(ValueError):
                    PP.physics_candidates(hist, "ball", bad_h, tick_hz=_HZ)
        for bad_dt in (0, 0.0, -0.05, False, True, NAN, INF, NINF):
            with self.subTest(dt=bad_dt):
                with self.assertRaises(ValueError):
                    PP.physics_candidates(hist, "ball", 3, tick_hz=_HZ, dt=bad_dt)

    def test_nonfinite_or_bool_residual_rms_blocked(self):
        hist = _projectile_history()
        real = PP.classify_entity
        for bad_rms in (NAN, INF, NINF, False, True, None, "nan"):
            def _stub(*args, _rms=bad_rms, **kwargs):
                got = dict(real(*args, **kwargs))
                mdl = dict(got["model"])
                mdl["residual_rms"] = _rms
                got["model"] = mdl
                return got

            with self.subTest(residual_rms=bad_rms):
                with _PatchAttr(PP, "classify_entity", _stub):
                    self.assertIsNone(
                        PP.physics_candidates(hist, "ball", 3, tick_hz=_HZ)
                    )


class TestWrapGenerateGate(unittest.TestCase):
    """G3：wrap_generate 拒绝坏 tick_hz / physics_dt / horizon，兼容 None 缺省。"""

    def _inner(self, obj, horizon=1, use_patterns=True):
        h = 1 if horizon is None else max(1, horizon) if isinstance(horizon, int) and not isinstance(horizon, bool) else horizon
        return {
            "horizon": h,
            "predictions": {"ball": {"mode": "bounded_stochastic", "predicted": [0.0, 0.0, 0.0]}},
        }

    def test_bad_tick_hz_and_dt_rejected(self):
        wrapped = PP.wrap_generate(self._inner)
        for bad_hz in (0, 0.0, -1.0, False, True, NAN, INF, NINF):
            class _Obj:
                history = _projectile_history()
                tick_hz = bad_hz

            with self.subTest(tick_hz=bad_hz):
                with self.assertRaises(ValueError):
                    wrapped(_Obj(), 3)

        for bad_dt in (0, -0.1, False, True, NAN, INF):
            class _ObjDt:
                history = _projectile_history()
                tick_hz = _HZ
                physics_dt = bad_dt

            with self.subTest(physics_dt=bad_dt):
                with self.assertRaises(ValueError):
                    wrapped(_ObjDt(), 3)

    def test_bad_horizon_rejected_and_none_tick_hz_compatible(self):
        wrapped = PP.wrap_generate(self._inner)

        class _ObjOk:
            history = _projectile_history()
            tick_hz = None

        out = wrapped(_ObjOk(), 2)
        self.assertIn("physics", out)

        for bad_h in (True, False, NAN, 1.5):
            class _ObjBadH:
                history = _projectile_history()
                tick_hz = _HZ

            with self.subTest(horizon=bad_h):
                with self.assertRaises(ValueError):
                    wrapped(_ObjBadH(), bad_h)


class TestPositiveControls(unittest.TestCase):
    """G4：合法 10 Hz 抛体路径与坏观测容错逐位保持不变。"""

    def test_valid_projectile_pipeline(self):
        hist = _projectile_history(_HZ, 8)
        cls = PP.classify_entity(hist, "ball", tick_hz=_HZ)
        self.assertIsNotNone(cls)
        self.assertEqual(cls["kind"], "projectile")

        cand = PP.physics_candidates(hist, "ball", 3, tick_hz=_HZ)
        self.assertIsNotNone(cand)
        self.assertEqual(cand["kind"], "projectile")
        self.assertEqual(len(cand["candidates"]), 3)
        t_last = 10.0 / _HZ
        expected = (4.0 * t_last, 1.5 + 8.0 * t_last - 0.5 * PH.DEFAULT_G * t_last * t_last, 0.0)
        self.assertLess(PH.vlen(PH.vsub(cand["candidates"][-1][1], expected)), 1e-9)

        class _Fake:
            history = hist
            tick_hz = _HZ

        def _inner(self, horizon=1, use_patterns=True):
            return {"horizon": max(1, horizon), "predictions": {"ball": {"mode": "bounded_stochastic", "predicted": [0.0, 0.0, 0.0]}}}

        out = PP.wrap_generate(_inner)(_Fake(), 3)
        self.assertEqual(out["predictions"]["ball"]["mode"], "physics:projectile")
        self.assertEqual(out["physics"]["taken"], 1)

    def test_corrupt_observation_frame_returns_none(self):
        hist = _projectile_history(_HZ, 8)
        hist[3]["entities"]["ball"]["pos"] = (NAN, 0.0, 0.0)
        self.assertIsNone(PP.classify_entity(hist, "ball", tick_hz=_HZ))
        hist2 = _projectile_history(_HZ, 8)
        hist2[3]["entities"]["ball"]["pos"] = (True, 0.0, 0.0)
        self.assertIsNone(PP.classify_entity(hist2, "ball", tick_hz=_HZ))


_PROBE_SNIPPET = r"""
import math, sys
from lingshu.world import physics as PH
from lingshu.world import physics_predict as PP

hist = [
    {"tick": i, "entities": {"ball": {"pos": (4.0 * (i / 10.0), 1.5 + 8.0 * (i / 10.0) - 0.5 * PH.DEFAULT_G * (i / 10.0) ** 2, 0.0)}}}
    for i in range(8)
]
for bad in (0, False, True, float("nan"), float("inf")):
    try:
        r = PP.classify_entity(hist, "ball", tick_hz=bad)
        print("BYPASS_TICK_HZ:%r->%r" % (bad, None if r is None else r["kind"]))
        sys.exit(1)
    except ValueError:
        pass
real = PP.classify_entity
PP.classify_entity = lambda *a, **kw: dict(real(*a, **kw), model=dict(real(*a, **kw)["model"], residual_rms=float("nan")))
if PP.physics_candidates(hist, "ball", 3, tick_hz=10.0) is not None:
    print("BYPASS_RMS_NAN")
    sys.exit(1)
PP.classify_entity = real
try:
    c = PP.physics_candidates(hist, "ball", True, tick_hz=10.0)
    print("BYPASS_HORIZON_BOOL:%d" % len(c["candidates"]))
    sys.exit(1)
except ValueError:
    pass
print("ALL_BLOCKED")
"""


def _run_probe(py_path: str) -> tuple[int, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = py_path
    proc = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", _PROBE_SNIPPET],
        cwd=py_path,
        env=env,
        capture_output=True,
        text=True,
    )
    return proc.returncode, (proc.stdout or proc.stderr).strip()


def run_mutate() -> int:
    """G5：在隔离临时副本中注入旧无闸代码进行定点变异自证（零修改仓内文件）。"""
    pp_rel = os.path.join("lingshu", "world", "physics_predict.py")
    src = os.path.join(REPO, pp_rel)
    with open(src, "r", encoding="utf-8") as f:
        clean_src = f.read()

    rc0, out0 = _run_probe(REPO)
    print(f"M0_clean_gated rc={rc0} out={out0!r} VERDICT={'PASS' if rc0 == 0 else 'FAIL'}")

    tmp = tempfile.mkdtemp(prefix="mutate_pp_")
    try:
        pkg_dst = os.path.join(tmp, "lingshu")
        shutil.copytree(os.path.join(REPO, "lingshu"), pkg_dst,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        mut_path = os.path.join(tmp, pp_rel)
        mutated = clean_src
        # M1: 回退 classify_entity / physics_candidates 的 tick_hz / rms / horizon 闸为原无闸实现
        mutated = mutated.replace(
            "def _require_positive_finite(val, name: str) -> float:\n",
            "def _require_positive_finite(val, name: str) -> float:\n    return float(val) if val else 1.0\n",
        )
        mutated = mutated.replace(
            "def _require_positive_int(val, name: str) -> int:\n",
            "def _require_positive_int(val, name: str) -> int:\n    return int(val)\n",
        )
        mutated = mutated.replace(
            "if not math.isfinite(rms) or rms > MAX_RMS:",
            "if rms > MAX_RMS:",
        )
        assert mutated != clean_src, "mutation did not change source"
        with open(mut_path, "w", encoding="utf-8") as f:
            f.write(mutated)
        rc1, out1 = _run_probe(tmp)
        print(f"M1_strip_finite_bool_gate rc={rc1} out={out1!r} VERDICT={'PASS' if rc1 != 0 else 'FAIL'}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    rc2, out2 = _run_probe(REPO)
    print(f"M2_restored_gated rc={rc2} out={out2!r} VERDICT={'PASS' if rc2 == 0 else 'FAIL'}")
    ok = (rc0 == 0 and rc1 != 0 and rc2 == 0)
    print(f"OVERALL={'PASS' if ok else 'FAIL'} (repo_files_untouched=True)")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--mutate" in sys.argv[1:]:
        raise SystemExit(run_mutate())
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    passed = result.testsRun - len(result.failures) - len(result.errors)
    failed = len(result.failures) + len(result.errors)
    print(f"SUMMARY pass={passed} fail={failed}")
    raise SystemExit(0 if result.wasSuccessful() else 1)
