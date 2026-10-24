"""在隔离副本组合两个世界模型增量；不修改检出，不替代完整合并验收。

左侧提供行动反馈实现，右侧提供身份关联实现；core.py 做真实三方文本合并。
只运行两项增量的具名守卫，不复制或改写清单指纹，不承诺完整 Git 合并无冲突。
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    left = git("rev-parse", "--verify", args.left + "^{commit}").decode().strip()
    right = git("rev-parse", "--verify", args.right + "^{commit}").decode().strip()
    base = git("merge-base", left, right).decode().strip()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="owned-composition-", dir=output) as directory:
        sandbox = Path(directory).resolve()
        assert sandbox.is_relative_to(output)
        archive = git("archive", "--format=zip", left, "lingshu")
        with zipfile.ZipFile(io.BytesIO(archive)) as files:
            assert all((sandbox / p.filename).resolve().is_relative_to(sandbox) for p in files.infolist())
            files.extractall(sandbox)
        core = "lingshu/core/core.py"
        inputs = []
        for name, revision in (("left", left), ("base", base), ("right", right)):
            path = sandbox / (name + "-core.py")
            path.write_bytes(git("show", revision + ":" + core))
            inputs.append(path)
        merge = subprocess.run(["git", "merge-file", "-p", *map(str, inputs)],
                               capture_output=True, cwd=ROOT)
        if merge.returncode:
            raise RuntimeError("core.py 三方文本合并失败；不能以覆盖一侧的方式伪造组合通过")
        (sandbox / core).write_bytes(merge.stdout)
        paths = ["lingshu/world/world_model.py", "lingshu/world/identity.py", "lingshu/world/wm_simloop.py"]
        for path in paths:
            (sandbox / path).write_bytes(git("show", right + ":" + path))
        guards = ["tests/test_world_qualified_feedback.py", "tests/test_world_identity_qualification.py"]
        (sandbox / "tests").mkdir()
        for path, revision in zip(guards, (left, right)):
            (sandbox / path).write_bytes(git("show", revision + ":" + path))
        env = {**os.environ, "PYTHONPATH": str(sandbox), "PYTHONDONTWRITEBYTECODE": "1"}
        if os.name != "nt":
            env["PYTEST_ADDOPTS"] = env.get("PYTEST_ADDOPTS", "") + " --capture=no"
        result = subprocess.run([sys.executable, "-B", "-X", "utf8", "-m", "pytest", *guards,
            "-q", "--tb=short", "-p", "no:cacheprovider", "--basetemp=" + str(sandbox / "temp"),
            "--junitxml=" + str(output / "composition.xml")], cwd=sandbox, env=env,
            capture_output=True, timeout=180)
        raw = (result.stdout + result.stderr).decode("utf-8", errors="replace").replace("\r\n", "\n")
        (output / "composition.log").write_text(raw, encoding="utf-8")
        tests = ET.parse(output / "composition.xml").findall(".//testcase")
        failures = [t.get("name") for t in tests if t.find("failure") is not None or t.find("error") is not None]
        skipped = [t.get("name") for t in tests if t.find("skipped") is not None]
        hashes = {p: hashlib.sha256((sandbox / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                  for p in [core, "lingshu/world/seven_layer_loop.py", *paths, *guards]}
    summary = {"left": left, "right": right, "base": base, "core_merge_returncode": merge.returncode,
               "returncode": result.returncode, "tests": len(tests), "failures": failures,
               "skipped": skipped, "sha256_lf": hashes, "owned_temp_cleaned": not sandbox.exists(),
               "scope": "两项增量守卫的源码组合；不是全套或清单/Git合并验收"}
    (output / "composition.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if result.returncode == 0 and not failures and not skipped else 1


if __name__ == "__main__":
    raise SystemExit(main())
