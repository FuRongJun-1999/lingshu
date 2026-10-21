"""Replay live-source export protection against a frozen baseline and mutations.

Run: python -X utf8 tools/testing/check_export_source_path_guard.py --baseline <sha> --mutate
Every database and modified source lives in a disposable copy under --output.
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
GUARD = Path("tests/test_export_source_path_guard.py")
G1 = "test_source_database_rejected_before_write"
G2 = "test_source_sidecars_rejected_even_if_absent"
BASELINE_FAILURES = ([f"{G1}[{m}]" for m in ("all", "trajectory", "structured")]
                     + [f"{G2}[{s}-{m}]" for s in ("-wal", "-shm", "-journal")
                        for m in ("all", "trajectory")]
                     + ["test_source_inside_allowed_root_is_still_rejected",
                        "test_source_hardlink_alias_is_rejected", "test_source_symlink_alias_is_rejected",
                        "test_connection_path_survives_working_directory_change",
                        "test_attached_database_is_protected"])

READINGS = '''
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import warnings
from lingshu.core.core import SpacetimeMemoryEngine
warnings.simplefilter("ignore")
records = []
for method in ("all", "trajectory"):
    path = Path("readings-" + method + ".db").resolve()
    engine = SpacetimeMemoryEngine(str(path))
    node = engine.add_perception("synthetic durable fact", skip_dedup=True)
    engine.store.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    record = {"entry": method, "before_header_hex": path.read_bytes()[:16].hex()}
    try:
        if method == "all":
            engine.export_all(str(path))
        else:
            engine.export_trajectory(output_path=str(path))
        record["returned_success"] = True
        record["error"] = None
    except Exception as exc:
        record["returned_success"] = False
        # Keep the real error codes/message, without publishing local filenames.
        detail = exc.strerror if isinstance(exc, OSError) else str(exc)
        record["error"] = type(exc).__name__ + ": " + str(detail)
        if isinstance(exc, OSError):
            record["errno"] = exc.errno
            record["winerror"] = getattr(exc, "winerror", None)
    try:
        engine.close()
    except sqlite3.DatabaseError as exc:
        record["close_error"] = str(exc)
    record["after_header_hex"] = path.read_bytes()[:16].hex()
    try:
        with closing(sqlite3.connect(path)) as observer:
            record["integrity"] = observer.execute("PRAGMA integrity_check").fetchone()[0]
            record["fact_survived"] = bool(observer.execute(
                "SELECT id FROM nodes WHERE id=?", (node.id,)).fetchone())
        record["reopen_error"] = None
    except sqlite3.DatabaseError as exc:
        record["reopen_error"] = type(exc).__name__ + ": " + str(exc)
        record["fact_survived"] = False
    records.append(record)
print(json.dumps(records, ensure_ascii=False, indent=2))
'''


def _once(source, old, new):
    if source.count(old) != 1:
        raise AssertionError("mutation target must occur exactly once")
    return source.replace(old, new, 1)


def _mutations(source):
    for name, old, new, node, label in (
        ("export_all_not_wired", "        self._guard_export_source_path(target_path)\n", "",
         f"{G1}[all]", "G1"),
        ("trajectory_not_wired", "            self._guard_export_source_path(output_path)\n", "",
         f"{G1}[trajectory]", "G1"),
        ("sidecars_not_protected", 'for suffix in ("", "-wal", "-shm", "-journal"):',
         'for suffix in ("",):', f"{G2}[-journal-all]", "G2"),
        ("hardlink_identity_not_checked", "aliases_source = os.path.samefile(out_path, source)",
         "aliases_source = False", "test_source_hardlink_alias_is_rejected", "G4"),
        ("attached_database_not_enumerated", 'for _, _, database_path in self.store.conn.execute("PRAGMA database_list"):',
         "for database_path in (self.store.db_path,):",
         "test_attached_database_is_protected", "G7"),
    ):
        yield name, _once(source, old, new), [node], f"{label} source path was not rejected before filesystem write"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline")
    parser.add_argument("--mutate", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/export_source_path")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = (ROOT / CORE).read_text(encoding="utf-8")
    cases = [("fixed", source, [], "")]
    if args.baseline:
        old = subprocess.run(["git", "show", f"{args.baseline}:{CORE.as_posix()}"],
                             cwd=ROOT, capture_output=True, check=True).stdout.decode("utf-8")
        cases.append(("baseline", old, BASELINE_FAILURES, ""))
    if args.mutate:
        cases.extend(_mutations(source))
    records = []
    for name, candidate, failures, assertion in cases:
        with tempfile.TemporaryDirectory(prefix="export-source-", dir=output) as directory:
            sandbox = Path(directory)
            shutil.copytree(ROOT / "lingshu", sandbox / "lingshu",
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            (sandbox / CORE).write_text(candidate, encoding="utf-8")
            (sandbox / GUARD).parent.mkdir(parents=True)
            shutil.copy2(ROOT / GUARD, sandbox / GUARD)
            env = {**os.environ, "PYTHONPATH": str(sandbox), "PYTHONDONTWRITEBYTECODE": "1"}
            env.pop("LINGSHU_EXPORT_OUT_ROOT", None)
            readings = None
            if name in ("fixed", "baseline"):
                probe = subprocess.run([sys.executable, "-B", "-X", "utf8", "-c", READINGS],
                                       cwd=sandbox, env=env, capture_output=True, timeout=60)
                (output / f"{name}-readings.txt").write_text(
                    (probe.stdout + probe.stderr).decode("utf-8", errors="replace").replace("\r\n", "\n"),
                    encoding="utf-8", newline="\n")
                if probe.returncode:
                    raise AssertionError(f"{name} readings failed, see raw output")
                readings = json.loads(probe.stdout)
                if name == "fixed":
                    assert all(r["error"].startswith("ValueError:") and r["fact_survived"]
                               and r["reopen_error"] is None for r in readings)
            junit = output / f"{name}.xml"
            command = [sys.executable, "-B", "-X", "utf8", "-m", "pytest",
                       str(GUARD) + ("::" + failures[0] if assertion else ""),
                       "-q", "--disable-warnings", "--tb=short", f"--junitxml={junit}",
                       f"--basetemp={sandbox / 'tmp'}"]
            result = subprocess.run(command, cwd=sandbox, env=env, capture_output=True, timeout=120)
            raw = (result.stdout + result.stderr).decode("utf-8", errors="replace").replace("\r\n", "\n")
            (output / f"{name}.log").write_text(raw, encoding="utf-8", newline="\n")
            tests = ET.parse(junit).findall(".//testcase")
            actual = sorted(t.get("name") for t in tests if t.find("failure") is not None)
            bad_other = any(t.find("error") is not None or t.find("skipped") is not None for t in tests)
            passed = sum(t.find("failure") is None and t.find("error") is None and
                         t.find("skipped") is None for t in tests)
            ok = (actual == sorted(failures) and not bad_other and len(tests) == (1 if assertion else 17)
                  and result.returncode == (1 if failures else 0) and (not assertion or assertion in raw))
            record = {"case": name, "ok": ok, "returncode": result.returncode, "passed": passed,
                      "failed_tests": actual, "expected_failures": failures, "expected_assertion": assertion,
                      "core_sha256": hashlib.sha256(candidate.encode("utf-8")).hexdigest(),
                      "readings": readings}
            records.append(record)
            print(json.dumps({k: v for k, v in record.items() if k != "readings"}, ensure_ascii=False), flush=True)
    summary = {"python": sys.version, "platform": platform.platform(), "sqlite": sqlite3.sqlite_version,
               "baseline": args.baseline, "guard_sha256": hashlib.sha256(
                   (ROOT / GUARD).read_bytes()).hexdigest(), "records": records}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if all(record["ok"] for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
