#!/usr/bin/env python3
"""Run the opt-in historical, sparse-backend and installed-wheel acceptance legs.

This uses only the standard library. The contracts leg expects the declared
``dev,core-sparse`` extras in its interpreter. The wheel leg creates its own
light/full virtual environments; it never installs into the caller's environment.
All output belongs in a new directory outside the source checkout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import venv
import xml.etree.ElementTree as ET
import zipfile

BASE_COMMIT = "124ea433fe9a2ad2fd66a7360df85f6d61bab793"
BASE_TREE = "fb2cefe048c0c4798c9016afd7256d86f0e664b8"
PHYSICS_NAMES = (
    "test_G2_no_env_is_field_identical_to_legacy",
    "test_G6_mutation_new_kinds_in_first_pass_turns_red",
    "test_G7_mutation_env_used_key_turns_red",
    "test_G2c_classify_field_set_frozen_against_baseline",
)
SPARSE_NAMES = {
    "test_activation_graph_refresh": (
        "test_new_node_becomes_a_seed_on_the_next_activation[scipy]",
        "test_new_edge_propagates_to_an_existing_node[scipy]",
        "test_replaced_edge_uses_its_current_relation_decay[scipy]",
        "test_removed_edge_stops_propagating[scipy]",
        "test_deleted_node_cannot_reenter_through_previous_workset[scipy]",
        "test_unchanged_graph_preserves_propagation_self_condition_and_audit[scipy]",
    ),
    "test_issue229_280_activation_suppression_bounded": (
        "test_opposite_edge_does_not_excite_its_target[scipy]",
        "test_opposite_edge_subtracts_from_excitatory_activation[scipy]",
        "test_opposite_suppression_is_clamped_at_zero[scipy]",
        "test_propagation_is_max_plus_not_sum_across_sources[scipy]",
        "test_self_conditioned_activation_stays_bounded_across_rounds[scipy]",
        "test_audit_max_act_is_bounded_under_self_condition[scipy]",
    ),
    "test_issue7_161_293_314_53_activation_contract": (
        "test_cond_match_actually_gates_propagation[scipy]",
        "test_cond_match_factor_is_the_only_difference[scipy]",
        "test_cond_match_half_overlap_scales_decay[scipy]",
        "test_cond_match_backends_agree",
    ),
}
ASSET = "lingshu/nn/assets/kernels_img0.json"


class AcceptanceError(RuntimeError):
    """Required execution evidence is absent or contradicts the contract."""


def require(condition, message):
    if not condition:
        raise AcceptanceError(message)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
                    + "\n", encoding="utf-8", newline="\n")


def clean_env():
    # Prevent an inherited deployment or source path from changing the acceptance.
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("PYTHON", "PYTEST_", "LINGSHU_", "MDCG_"))
           and key not in {"VIRTUAL_ENV", "__PYVENV_LAUNCHER__", "PIP_TARGET", "PIP_PREFIX",
                           "PIP_USER", "PIP_ROOT", "PIP_CONFIG_FILE"}}
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1",
               PIP_CONFIG_FILE=os.devnull)
    return env


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], timeout=120)


def blob_sha(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data,
                        usedforsecurity=False).hexdigest()


def verify_baseline_export(root, expected):
    """Check exact Git blob identities, not a plausible directory name or timestamp."""
    require(root.is_dir(), "frozen baseline directory is missing")
    actual = set()
    for path in root.rglob("*"):
        require(not path.is_symlink(), "frozen baseline contains a symlink")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    require(actual == set(expected), "frozen baseline file set differs from the pinned tree")
    for name, expected_blob in expected.items():
        require(blob_sha((root / name).read_bytes()) == expected_blob,
                "frozen baseline bytes differ: " + name)
    return {"files": len(actual), "verified_git_blobs": len(expected)}


def validate_junit(path, required):
    """Require successful real cases, including every named guard; SKIP is not pass."""
    require(path.is_file(), "pytest result is missing: " + path.name)
    try:
        tree = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise AcceptanceError("pytest result is not valid XML") from exc
    cases = list(tree.iter("testcase"))
    require(bool(cases), "pytest result has no executed test cases")
    names = [case.attrib.get("classname", "") + "::" + case.attrib.get("name", "")
             for case in cases]
    require(len(names) == len(set(names)), "pytest result repeats a test identity")
    require(set(required) <= set(names), "pytest result omits required guards: "
            + ", ".join(sorted(set(required) - set(names))))
    for case, name in zip(cases, names):
        require(not any(child.tag in {"skipped", "failure", "error"} for child in case),
                "guard did not pass: " + name)
    return {"passed": len(cases), "skipped": 0, "required_passed": sorted(required),
            "junit_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def source_identity(repo):
    return {"head": git(repo, "rev-parse", "HEAD").decode().strip(),
            "tree": git(repo, "rev-parse", "HEAD^{tree}").decode().strip(),
            "status": git(repo, "status", "--porcelain").decode().strip()}


def verify_source_identity(before, after):
    require(set(before) == {"head", "tree", "status"} and before == after
            and not before["status"], "source commit, tree or status changed during acceptance")


class Runner:
    def __init__(self, repo, output):
        self.repo, self.output = repo, output
        self.env = clean_env()
        initial = source_identity(repo)
        self.report = {"source_head": initial["head"], "source_tree": initial["tree"],
                       "source_status": initial["status"],
                       "python": sys.version, "steps": [], "accepted": False}
        self.save()
        require(not self.report["source_status"], "acceptance requires a clean committed source checkout")

    def save(self):
        write_json(self.output / "summary.json", self.report)

    def run(self, name, command, *, cwd=None, env=None, timeout=600):
        stdout, stderr = self.output / (name + ".stdout.log"), self.output / (name + ".stderr.log")
        started = time.monotonic()
        entry = {"name": name, "command": [str(arg) for arg in command]}
        self.report["steps"].append(entry)
        self.save()
        try:
            with stdout.open("wb") as out, stderr.open("wb") as err:
                result = subprocess.run(command, cwd=cwd or self.repo, env=env or self.env,
                                        stdout=out, stderr=err, timeout=timeout)
            entry["returncode"] = result.returncode
        except subprocess.TimeoutExpired as exc:
            entry.update(returncode="timeout", seconds=round(time.monotonic() - started, 3))
            self.save()
            raise AcceptanceError("acceptance step timed out: " + name) from exc
        entry.update(seconds=round(time.monotonic() - started, 3),
                     stdout_sha256=hashlib.sha256(stdout.read_bytes()).hexdigest(),
                     stderr_sha256=hashlib.sha256(stderr.read_bytes()).hexdigest())
        self.save()
        require(result.returncode == 0, "acceptance step failed: " + name + " (see raw logs)")
        return stdout


def export_baseline(runner, fetch):
    try:
        git(runner.repo, "cat-file", "-e", BASE_COMMIT + "^{commit}")
    except subprocess.CalledProcessError:
        require(fetch, "pinned baseline commit is missing; use --fetch-baseline")
        runner.run("fetch-baseline", ["git", "-C", str(runner.repo), "fetch", "--no-tags",
                                       "--depth=1", "origin", BASE_COMMIT])
    tree = git(runner.repo, "rev-parse", BASE_COMMIT + "^{tree}").decode().strip()
    require(tree == BASE_TREE, "pinned baseline tree does not match")
    records = git(runner.repo, "ls-tree", "-r", "-z", BASE_COMMIT, "--", "lingshu")
    expected = {}
    for record in records.split(b"\0"):
        if not record:
            continue
        metadata, name = record.split(b"\t", 1)
        mode, kind, oid = metadata.decode().split()
        require(mode == "100644" and kind == "blob", "baseline requires regular source files")
        expected[name.decode("utf-8")] = oid
    require("lingshu/world/physics.py" in expected, "pinned tree has no physics implementation")
    archive = runner.output / "physics-baseline.tar"
    # Archive conversion otherwise inherits Windows core.autocrlf. The verifier
    # deliberately compares exact pinned blobs, including their original LF bytes.
    runner.run("archive-baseline", ["git", "-c", "core.autocrlf=false", "-c", "core.eol=lf",
                                   "-C", str(runner.repo), "archive", "--format=tar",
                                   "--output=" + str(archive), BASE_COMMIT, "lingshu"])
    destination = runner.output / "physics-baseline"
    destination.mkdir()
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            path = PurePosixPath(member.name)
            require(not path.is_absolute() and ".." not in path.parts
                    and path.parts[0] == "lingshu" and (member.isfile() or member.isdir()),
                    "unexpected path in pinned archive")
        tar.extractall(destination, filter="data")
    verification = verify_baseline_export(destination, expected)
    runner.report["physics_baseline"] = {"commit": BASE_COMMIT, "tree": tree,
                                        "file_blobs": expected, **verification}
    runner.save()
    return destination, expected


def contracts(runner, fetch):
    baseline, expected = export_baseline(runner, fetch)
    env = dict(runner.env, LINGSHU_BASE_TREE=str(baseline), PYTHONPATH=str(runner.repo))
    physics_required = {"tests.test_physics_predict::" + name for name in PHYSICS_NAMES}
    physics_nodes = ["tests/test_physics_predict.py::" + name for name in PHYSICS_NAMES]
    junit = runner.output / "physics.xml"
    runner.run("physics", [sys.executable, "-B", "-X", "utf8", "-m", "pytest", *physics_nodes,
                           "-q", "-rs", "--tb=short", "-p", "no:cacheprovider", "--junitxml=" + str(junit)],
               env=env)
    runner.report["physics"] = validate_junit(junit, physics_required)
    verify_baseline_export(baseline, expected)
    runner.save()
    junit = runner.output / "sparse.xml"
    files = ["tests/" + name + ".py" for name in SPARSE_NAMES]
    required = {"tests." + module + "::" + name
                for module, names in SPARSE_NAMES.items() for name in names}
    runner.run("sparse", [sys.executable, "-B", "-X", "utf8", "-m", "pytest", *files,
                          "-q", "-rs", "--tb=short", "-p", "no:cacheprovider", "--junitxml=" + str(junit)],
               env=env)
    runner.report["sparse"] = validate_junit(junit, required)
    runner.save()


def venv_python(path):
    return path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def wheel(runner):
    # Build in a disposable copy: setuptools must not leave build/egg-info in source.
    source = runner.output / "build-source"
    source.mkdir()
    names = git(runner.repo, "ls-files", "-z").split(b"\0")
    for name in names:
        if not name:
            continue
        relative = Path(name.decode("utf-8"))
        original = runner.repo / relative
        require(original.is_file() and not original.is_symlink(), "wheel source must be regular files")
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
    dist = runner.output / "dist"
    dist.mkdir()
    runner.run("build-wheel", [sys.executable, "-I", "-X", "utf8", "-m", "pip", "wheel",
                               "--no-deps", "--wheel-dir", str(dist), str(source)], cwd=runner.output)
    wheels = list(dist.glob("lingshu-*.whl"))
    require(len(wheels) == 1, "wheel build did not produce exactly one lingshu wheel")
    built = wheels[0]
    with zipfile.ZipFile(built) as archive:
        require(ASSET in archive.namelist(), "built wheel is missing the declared nn asset")
        asset_hash = hashlib.sha256(archive.read(ASSET)).hexdigest()
    runner.report["wheel"] = {"name": built.name, "sha256": hashlib.sha256(built.read_bytes()).hexdigest(),
                              "asset_sha256": asset_hash}
    probe = runner.output / "installed_wheel_probe.py"
    shutil.copyfile(Path(__file__).with_name("installed_wheel_probe.py"), probe)
    for phase in ("light", "full"):
        environment = runner.output / (phase + "-venv")
        venv.EnvBuilder(with_pip=True).create(environment)
        interpreter = venv_python(environment)
        spec = str(built) + ("[full]" if phase == "full" else "")
        runner.run("install-" + phase, [str(interpreter), "-I", "-X", "utf8", "-m", "pip",
                                        "install", *(["--no-deps"] if phase == "light" else []), spec],
                   cwd=runner.output)
        runner.run("pip-check-" + phase, [str(interpreter), "-I", "-X", "utf8", "-m", "pip", "check"],
                   cwd=runner.output)
        result = runner.output / (phase + "-journey")
        result.mkdir()
        runner.run("probe-" + phase, [str(interpreter), "-I", "-B", "-X", "utf8", str(probe),
                                      "--phase", phase, "--source", str(runner.repo),
                                      "--output", str(result)], cwd=result)
        report = result / "report.json"
        require(report.is_file(), "installed wheel probe produced no report")
        data = json.loads(report.read_text(encoding="utf-8"))
        require(data.get("accepted") is True, "installed wheel probe was not accepted")
        if phase == "full":
            require(data.get("asset_sha256") == asset_hash, "installed nn asset differs from built wheel")
        runner.report[phase] = data
        runner.save()


def selftest():
    """Exercise rejection evidence directly, without changing the product or the repo."""
    rejected = []
    with tempfile.TemporaryDirectory(prefix="lingshu-acceptance-selftest-") as folder:
        root = Path(folder)
        result = root / "result.xml"
        wanted = {"test.guard::required"}
        checks = [
            ("missing_result", None),
            ("empty_result", "<testsuites/>"),
            ("omitted_guard", '<testsuite><testcase classname="test.guard" name="other"/></testsuite>'),
            ("skipped_guard", '<testsuite><testcase classname="test.guard" name="required"><skipped/></testcase></testsuite>'),
            ("failed_guard", '<testsuite><testcase classname="test.guard" name="required"><failure/></testcase></testsuite>'),
        ]
        for name, xml in checks:
            if xml is not None:
                result.write_text(xml, encoding="utf-8")
            try:
                validate_junit(result, wanted)
            except AcceptanceError:
                rejected.append(name)
            else:
                raise AcceptanceError("false-green selftest was accepted: " + name)
        result.write_text('<testsuite><testcase classname="test.guard" name="required"/></testsuite>', encoding="utf-8")
        require(validate_junit(result, wanted)["passed"] == 1, "positive report selftest failed")
        baseline = root / "baseline"
        expected = {"lingshu/world/physics.py": blob_sha(b"pinned source\n")}
        for name in ("missing_baseline", "tampered_baseline"):
            if name == "tampered_baseline":
                target = baseline / "lingshu/world/physics.py"
                target.parent.mkdir(parents=True)
                target.write_bytes(b"different source\n")
            try:
                verify_baseline_export(baseline, expected)
            except AcceptanceError:
                rejected.append(name)
            else:
                raise AcceptanceError("false baseline selftest was accepted: " + name)
        target.write_bytes(b"pinned source\n")
        require(verify_baseline_export(baseline, expected)["files"] == 1, "positive baseline selftest failed")
        before = {"head": "first", "tree": "first-tree", "status": ""}
        verify_source_identity(before, dict(before))
        try:
            verify_source_identity(before, {"head": "second", "tree": "second-tree", "status": ""})
        except AcceptanceError:
            rejected.append("clean_source_commit_changed")
        else:
            raise AcceptanceError("clean source commit change was accepted")
    return {"rejected": rejected, "positive_controls_passed": 3}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("leg", choices=("selftest", "contracts", "wheel"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fetch-baseline", action="store_true")
    args = parser.parse_args()
    if args.leg == "selftest":
        print(json.dumps(selftest(), indent=2))
        return 0
    require(args.output is not None, "--output is required")
    repo, output = args.repo.resolve(), args.output.resolve()
    require(not output.is_relative_to(repo), "acceptance output must be outside source")
    require(not output.exists() or (output.is_dir() and not any(output.iterdir())),
            "acceptance output must be new or empty; existing evidence is preserved")
    output.mkdir(parents=True, exist_ok=True)
    runner = Runner(repo, output)
    try:
        runner.report["selftest"] = selftest()
        if args.leg == "contracts":
            contracts(runner, args.fetch_baseline)
        else:
            wheel(runner)
        before = {key: runner.report["source_" + key] for key in ("head", "tree", "status")}
        after = source_identity(repo)
        runner.report.update(source_after=after, source_unchanged=(before == after))
        verify_source_identity(before, after)
        runner.report["accepted"] = True
    except (AcceptanceError, subprocess.SubprocessError, OSError, ValueError) as exc:
        runner.report["error"] = str(exc)
        raise
    finally:
        runner.save()
    print(json.dumps({"accepted": True, "leg": args.leg, "report": str(output / "summary.json")}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AcceptanceError as error:
        print("acceptance failed: " + str(error), file=sys.stderr)
        raise SystemExit(1)
