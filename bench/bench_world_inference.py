"""World-model inference cost, fixed histories and observable result hashes.

Run this unchanged in the before/after checkout. Timings are measurements,
not CI thresholds; missing/currently stationary observations remain distinct.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lingshu.world.world_learner import LNNode, WorldLearner
from lingshu.world.world_model import UnifiedWorldModel, WMNode


def make_history(count, ticks, policy, seed=0):
    rng = random.Random(seed)
    ids = ['e' + str(i) for i in range(count)]
    frames = []
    for tick in range(1, ticks + 1):
        if policy == 'dense' or tick == 1:
            chosen = range(count)
        elif policy == 'single':
            chosen = [tick % count]
        elif policy == 'random_four':
            chosen = rng.sample(range(count), min(4, count))
        else:  # A fully observed past followed by an empty window.
            chosen = range(count) if tick <= ticks - 16 else []
        positions = {ids[i]: [3 + i % 8 * 2 + (tick % 20) * 0.02,
                             1.5, 3 + i // 8 * 0.5] for i in chosen}
        frames.append({'tick': tick, 'entities': positions})
    return frames


def make_model(kind, frames):
    model = UnifiedWorldModel() if kind == 'unified' else WorldLearner()
    last = {}
    first = {}
    for frame in frames:
        for eid, pos in frame['entities'].items():
            first.setdefault(eid, frame['tick'])
            last[eid] = (frame['tick'], tuple(pos))
    node_class = WMNode if kind == 'unified' else LNNode
    model.nodes = {eid: node_class(eid=eid, category=eid, pos=pos,
                                  first_seen=first[eid], last_seen=tick)
                   for eid, (tick, pos) in last.items()}
    if kind == 'unified':
        model.history = frames
    else:
        model.history = [{'tick': f['tick'], 'entities': {eid: {'pos': tuple(pos), 'category': eid}
                         for eid, pos in f['entities'].items()}} for f in frames]
    model.tick = frames[-1]['tick']
    return model


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def run(repeats):
    rows = []
    specs = [(16, 512, 'dense'), (64, 512, 'dense'), (128, 512, 'dense'),
             (64, 2000, 'single'), (64, 8000, 'single'),
             (64, 2000, 'random_four'), (64, 2000, 'empty_tail')]
    for count, ticks, policy in specs:
        frames = make_history(count, ticks, policy)
        history_hash = digest(frames)
        for kind in ('learner', 'unified'):
            seconds = []
            hashes = []
            for _ in range(repeats):
                model = make_model(kind, frames)
                start = time.perf_counter()
                report = model.infer_patterns(force=True) if kind == 'unified' else model.learn()
                seconds.append(time.perf_counter() - start)
                hashes.append(digest(report))
            assert len(set(hashes)) == 1, 'Inference is not deterministic'
            row = {'model': kind, 'entities': count, 'ticks': ticks, 'policy': policy,
                   'history_sha256': history_hash, 'result_sha256': hashes[0],
                   'relations': len(report['relations']), 'seconds': seconds,
                   'median_seconds': statistics.median(seconds)}
            rows.append(row)
            print(json.dumps(row), flush=True)
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True,
                            text=True, encoding='utf-8', errors='replace')
    sources = ['motion.py', 'world_learner.py', 'world_model.py']
    return {'python': platform.python_version(), 'head': commit.stdout.strip(), 'repeats': repeats,
            'benchmark_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'source_sha256': {name: hashlib.sha256((ROOT/'lingshu/world'/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                              for name in sources}, 'cases': rows}


def compare(before, after):
    assert before['benchmark_sha256'] == after['benchmark_sha256'], 'Benchmark code differs'
    assert before['repeats'] == after['repeats'], 'Measurement repeats differ'
    assert len(before['cases']) == len(after['cases']), 'Case sets differ'
    for b, a in zip(before['cases'], after['cases']):
        for key in ('model', 'entities', 'ticks', 'policy', 'history_sha256'):
            assert b[key] == a[key], (key, b[key], a[key])
        print(f"{a['model']:7} {a['policy']:12} E={a['entities']:3} H={a['ticks']:5} "
              f"{b['median_seconds']:.6f} -> {a['median_seconds']:.6f} s "
              f"ratio={b['median_seconds']/a['median_seconds']:.2f} "
              f"same_result={a['result_sha256'] == b['result_sha256']} "
              f"relations={b['relations']} -> {a['relations']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--compare', nargs=2, type=Path)
    args = parser.parse_args()
    if args.compare:
        compare(*(json.loads(p.read_text(encoding='utf-8')) for p in args.compare))
    else:
        result = run(args.repeats)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
