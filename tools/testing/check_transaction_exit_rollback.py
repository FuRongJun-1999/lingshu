"""Replay frozen baseline and named transaction-exit mutations in disposable copies.

Run from the repository root (pytest is a dev dependency):
  python tools/testing/check_transaction_exit_rollback.py --baseline <base-sha> --mutate
The working tree is never edited. Raw pytest/JUnit logs remain in --output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
CORE = Path("lingshu/core/core.py")
GUARD = Path("tests/test_transaction_exit_rollback.py")
G1 = "test_commit_failure_restores_subtree_and_allows_durable_followup"
G2 = "test_release_failure_rolls_back_only_its_caller_owned_savepoint"
G3 = "test_release_failure_can_be_caught_inside_managed_outer_transaction"
G4 = "test_body_failure_still_rolls_back_and_resets_depth[ValueError]"


def _once(source, old, new):
    if source.count(old) != 1:
        raise AssertionError("mutation target must occur exactly once")
    return source.replace(old, new, 1)


def _mutations(source):
    begin = source.index("    @contextlib.contextmanager\n    def _tx(self):")
    end = source.index("\n    @_transactional\n    def _init_tables", begin)
    original = source[begin:end]
    finish = '''            if nested:
                conn.execute("RELEASE SAVEPOINT %s" % savepoint)
            else:
                conn.commit()
        except BaseException:'''
    boundary = "            raise\n        finally:"

    commit_outside = _once(original, finish, '''            if nested:
                conn.execute("RELEASE SAVEPOINT %s" % savepoint)
        except BaseException:''')
    commit_outside = _once(commit_outside, boundary, '''            raise
        else:
            if not nested:
                conn.commit()
        finally:''')
    release_outside = _once(original, finish, '''            if not nested:
                conn.commit()
        except BaseException:''')
    release_outside = _once(release_outside, boundary, '''            raise
        else:
            if nested:
                conn.execute("RELEASE SAVEPOINT %s" % savepoint)
        finally:''')
    caller_rollback = _once(original, '''                    conn.execute("ROLLBACK TO SAVEPOINT %s" % savepoint)
                    conn.execute("RELEASE SAVEPOINT %s" % savepoint)''', "                    conn.rollback()")
    depth_leak = _once(original, "            self._tx_depth -= 1", "            self._tx_depth -= 0")
    for name, changed, node, message in (
        ("commit_outside_rollback_boundary", commit_outside, G1,
         "G1 failed COMMIT left a transaction pending"),
        ("release_outside_rollback_boundary", release_outside, G2,
         "G2 failed RELEASE left its inner write pending"),
        ("rollback_caller_instead_of_savepoint", caller_rollback, G2,
         "G2 caller transaction was ended by inner recovery"),
        ("transaction_depth_not_reset", depth_leak, G4,
         "G4 transaction depth was not reset after a body failure"),
    ):
        yield name, source[:begin] + changed + source[end:], [node], message


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", help="Git revision used only for the core.py baseline")
    parser.add_argument("--mutate", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/transaction_exit_rollback")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = (ROOT / CORE).read_text(encoding="utf-8")
    cases = [("fixed", source, [], "")]
    if args.baseline:
        old = subprocess.run(["git", "show", f"{args.baseline}:{CORE.as_posix()}"],
                             cwd=ROOT, capture_output=True, check=True).stdout.decode("utf-8")
        cases.append(("baseline", old, [G1, G2, G3], ""))
    if args.mutate:
        cases.extend(_mutations(source))
    records = []
    for name, candidate, failures, assertion in cases:
        with tempfile.TemporaryDirectory(prefix="tx-exit-", dir=output) as directory:
            sandbox = Path(directory)
            shutil.copytree(ROOT / "lingshu", sandbox / "lingshu",
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            (sandbox / CORE).write_text(candidate, encoding="utf-8")
            (sandbox / GUARD).parent.mkdir(parents=True)
            shutil.copy2(ROOT / GUARD, sandbox / GUARD)
            junit = output / f"{name}.xml"
            command = [sys.executable, "-B", "-X", "utf8", "-m", "pytest",
                       str(GUARD) + ("::" + failures[0] if assertion else ""),
                       "-q", "--disable-warnings", f"--junitxml={junit}",
                       f"--basetemp={sandbox / 'tmp'}"]
            env = {**os.environ, "PYTHONPATH": str(sandbox), "PYTHONDONTWRITEBYTECODE": "1"}
            result = subprocess.run(command, cwd=sandbox, env=env,
                                    capture_output=True, timeout=120)
            raw = (result.stdout + result.stderr).decode("utf-8", errors="replace")
            (output / f"{name}.log").write_text(raw, encoding="utf-8")
            tree = ET.parse(junit)
            tests = tree.findall(".//testcase")
            actual = sorted(t.get("name") for t in tests if t.find("failure") is not None)
            bad_other = any(t.find("error") is not None or t.find("skipped") is not None for t in tests)
            passed = sum(t.find("failure") is None and t.find("error") is None and
                         t.find("skipped") is None for t in tests)
            expected_count = 1 if assertion else 6
            ok = (actual == sorted(failures) and not bad_other and len(tests) == expected_count
                  and result.returncode == (1 if failures else 0)
                  and (not assertion or assertion in raw))
            record = {"case": name, "ok": ok, "returncode": result.returncode,
                      "passed": passed, "failed_tests": actual, "expected_failures": failures,
                      "expected_assertion": assertion, "core_sha256": hashlib.sha256(
                          candidate.encode("utf-8")).hexdigest()}
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    summary = {"python": sys.version, "platform": platform.platform(), "sqlite": sqlite3.sqlite_version,
               "baseline": args.baseline, "guard_sha256": hashlib.sha256(
                   (ROOT / GUARD).read_bytes()).hexdigest(), "records": records}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if all(record["ok"] for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
