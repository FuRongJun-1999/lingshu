"""Compare shared statistics with a supplied, unchanged baseline checkout.

Uses the baseline's original scalar functions, not a second copy of the new
implementation. Run with --baseline PATH (cb3a6ea for this iteration).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
from types import SimpleNamespace
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lingshu.world.motion import ObservationWindow


def baseline_method(root, filename, cls, method):
    source = root / 'lingshu/world' / filename
    module = ast.parse(source.read_text(encoding='utf-8'))
    definition = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == cls)
    function = next(node for node in definition.body if isinstance(node, ast.FunctionDef) and node.name == method)
    namespace = {'math': math, 'Optional': Optional, 'Tuple': Tuple}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace[method]


def check(baseline):
    spec = importlib.util.spec_from_file_location('baseline_motion', baseline / 'lingshu/world/motion.py')
    motion = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(motion)
    old_flat = baseline_method(baseline, 'world_model.py', 'UnifiedWorldModel', '_pair_stats')
    old_nested = baseline_method(baseline, 'world_learner.py', 'WorldLearner', '_pair_tendency')
    checks = 0
    for seed in range(200):
        rng = random.Random(seed)
        flat, tick = [], 0
        for _ in range(rng.randrange(1, 80)):
            tick += rng.randrange(1, 4)
            flat.append({'tick': tick, 'entities': {
                str(eid): (rng.randrange(8), 1.5, rng.randrange(8))
                for eid in range(6) if rng.random() < (seed % 5) / 4}})
        for nested in (False, True):
            history = ([{'tick': f['tick'], 'entities': {eid: {'pos': p} for eid, p in f['entities'].items()}}
                        for f in flat] if nested else flat)
            for width in (1, 2, 6, 8, 100):
                view = ObservationWindow(history, width, nested)
                stub = SimpleNamespace(history=history, window=width)
                for a in map(str, range(7)):
                    assert view.motion(a) == motion.estimate_motion(history, a, width, nested), (seed, nested, width, a)
                    checks += 1
                    for b in map(str, range(7)):
                        got = view.pair(a, b)
                        expected = old_nested(stub, a, b, width) if nested else old_flat(stub, a, b, width)
                        assert (got[:2] if nested else (got[0], got[2])) == expected, (seed, nested, width, a, b)
                        checks += 1
    files = ('motion.py', 'world_learner.py', 'world_model.py')
    return {'seeds': 200, 'widths': [1, 2, 6, 8, 100], 'checks': checks, 'mismatches': 0,
            'baseline_lf_sha256': {name: hashlib.sha256((baseline / 'lingshu/world' / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                                   for name in files},
            'candidate_motion_lf_sha256': hashlib.sha256((ROOT / 'lingshu/world/motion.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = check(args.baseline.resolve())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
