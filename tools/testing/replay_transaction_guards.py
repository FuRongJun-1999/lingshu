"""Replay one committed guard/spec against fixed, baseline and named mutants.

The real working tree is not edited. Each run copies production code and the
unchanged guard to a fresh directory; raw stdout and JUnit are kept in output.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]


def under(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("spec paths must be relative and stay inside the checkout")
    target = (root / path).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("spec path escaped its checkout")
    return target


def mutate(source, change):
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == change["class"])
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == change["method"])
    lines = source.splitlines(keepends=True)
    body = "".join(lines[method.lineno - 1:method.end_lineno])
    old = change["old"]
    if not old or body.count(old) != 1:
        raise ValueError("mutation target must occur exactly once in the named method")
    return ("".join(lines[:method.lineno - 1]) + body.replace(old, change["new"], 1)
            + "".join(lines[method.end_lineno:]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--mutate", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    spec = json.loads(under(ROOT, args.spec).read_text(encoding="utf-8"))
    guard = spec["guard"]
    output = (args.output or ROOT / "artifacts" / Path(args.spec).stem).resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = {path: under(ROOT, path).read_text(encoding="utf-8")
               for path in spec["sources"]}
    cases = [("fixed", sources, [], "")]
    if args.baseline:
        baseline = {path: subprocess.run(
            ["git", "show", f"{args.baseline}:{path}"], cwd=ROOT,
            capture_output=True, check=True).stdout.decode("utf-8") for path in sources}
        cases.append(("baseline", baseline, spec["baseline_failures"], ""))
    if args.mutate:
        for item in spec["mutations"]:
            candidate = dict(sources)
            for change in item["changes"]:
                path = change["file"]
                candidate[path] = mutate(candidate[path], change)
            cases.append((item["name"], candidate, [item["test"]], item["assertion"]))
    records = []
    for name, candidate, failures, assertion in cases:
        if re.fullmatch(r"[a-z][a-z0-9_]*", name) is None:
            raise ValueError("case names must be safe artifact filenames")
        with tempfile.TemporaryDirectory(prefix="guard-", dir=output) as directory:
            sandbox = Path(directory).resolve()
            assert sandbox.is_relative_to(output)
            shutil.copytree(ROOT / "lingshu", sandbox / "lingshu",
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            for path, text in candidate.items():
                under(sandbox, path).write_text(text, encoding="utf-8", newline="\n")
            target = under(sandbox, guard)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(under(ROOT, guard), target)
            junit = output / (name + ".xml")
            selected = guard + ("::" + failures[0] if assertion else "")
            command = [sys.executable, "-B", "-X", "utf8", "-m", "pytest", selected,
                       "-q", "--tb=short", "--disable-warnings", "-p", "no:cacheprovider",
                       f"--junitxml={junit}", f"--basetemp={sandbox / 'tmp'}"]
            env = {**os.environ, "PYTHONPATH": str(sandbox), "PYTHONDONTWRITEBYTECODE": "1"}
            result = subprocess.run(command, cwd=sandbox, env=env,
                                    capture_output=True, timeout=120)
            raw = (result.stdout + result.stderr).decode("utf-8", errors="replace")
            raw = raw.replace("\r\n", "\n")
            (output / (name + ".log")).write_text(raw, encoding="utf-8", newline="\n")
            tests = ET.parse(junit).findall(".//testcase")
            actual = sorted(t.get("name") for t in tests if t.find("failure") is not None)
            errors = sum(t.find("error") is not None for t in tests)
            skipped = sum(t.find("skipped") is not None for t in tests)
            passed = len(tests) - len(actual) - errors - skipped
            expected_count = 1 if assertion else spec["total"]
            ok = (actual == sorted(failures) and not errors and not skipped
                  and len(tests) == expected_count
                  and result.returncode == (1 if failures else 0)
                  and (not assertion or assertion in raw))
            summaries = re.findall(r"^.*\b(?:passed|failed)\b.* in [0-9.]+s\s*$", raw, re.M)
            record = {"case": name, "ok": ok, "returncode": result.returncode,
                      "passed": passed, "failed_tests": actual, "errors": errors,
                      "skipped": skipped, "expected_failures": failures,
                      "expected_assertion": assertion,
                      "pytest_summary": summaries[-1].strip() if summaries else "",
                      "source_sha256": {p: hashlib.sha256(s.encode("utf-8")).hexdigest()
                                        for p, s in candidate.items()}}
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    summary = {"python": platform.python_version(), "platform": platform.system(),
               "sqlite": sqlite3.sqlite_version, "baseline": args.baseline,
               "guard_sha256": hashlib.sha256(under(ROOT, guard).read_bytes().replace(
                   b"\r\n", b"\n")).hexdigest(), "records": records}
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    return 0 if all(record["ok"] for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
