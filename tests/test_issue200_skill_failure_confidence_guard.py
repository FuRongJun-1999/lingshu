import os


def test_skill_failure_confidence_gate():
    """pytest 收集入口（#200 脚本式守卫的门禁接线）：进程内直调自跑主流程，
    退出码非 0 即红（与 `python -X utf8` 直跑同判据）。"""
    import os
    import runpy
    REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    import subprocess as _sp
    import sys as _sys

    main_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "test_issue200_skill_failure_confidence_guard.py")
    env = dict(os.environ)
    env.update({"PYTHONPATH": REPO, "PYTHONUTF8": "1",
                "PYTHONDONTWRITEBYTECODE": "1"})
    r = _sp.run([_sys.executable, "-B", main_path], capture_output=True,
                text=True, encoding="utf-8", env=env, cwd=REPO, timeout=600)
    assert r.returncode == 0, (
        "技能失败降信守卫红：\n%s\n%s" % (r.stdout[-2000:], r.stderr[-500:]))
