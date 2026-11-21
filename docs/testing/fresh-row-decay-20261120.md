# Fresh rows in a time-based decay interval

Baseline: `d74848fb33dbff1d719959df19d1266a1ac60f44`.

The store's global `_last_decay_at` cursor can predate a newly created row.
Before this fix, the next clock-mode decay applied that entire interval to
every eligible context node, knowledge node and unverified edge. A context
received milliseconds after a restart could consequently be hard-deleted by
the catch-up decay for the preceding week. This is separate from #176's
original call-count decay and its clock-rollback follow-up.

## Public-API reproduction

```python
import time
from lingshu.core.core import SpacetimeMemoryEngine, EdgeType

engine = SpacetimeMemoryEngine(":memory:")
engine.decay_cycle(now=time.time() - 7 * 86400)
context = engine.add_context("just received live context", importance=0.5)
a = engine.add_perception("live source event", skip_dedup=True)
b = engine.add_perception("unrelated fresh observation", skip_dedup=True)
edge = engine.add_edge(a.id, b.id, EdgeType.CAUSAL)
now = time.time()
engine.decay_cycle(now=now)
print(engine.store.get_node(context.id))
print(engine.store.get_node(a.id).confidence)
print(engine.store.get_edge(edge.id))
engine.close()
```

Measured on the baseline: a 2.999 ms-old context was deleted; a 2.001 ms-old
knowledge node changed from `0.5` to `1.8105578254347584e-89`; a 0.499 ms-old
edge was deleted. The independent post-birth formula gives approximately
`0.4999994951`, `0.4999996632`, and `0.4999999159`, respectively. Millisecond
values vary with scheduling; the week-versus-milliseconds mismatch does not.

## Fix and clock boundary

For a previously established, finite cursor, a finite creation timestamp in
`(previous_tick, now]` establishes that the row was born during this interval.
Use `min(global_dt, now - created_at)` for that row. Older rows retain the full
interval. The same rule applies to all three decay legs; thresholds, protected
rows, immutable endpoints, and OPPOSITE-edge exemptions are unchanged.

An exactly zero interval is the identity, including `factor=1`: no elapsed
time must not delete a row born exactly at `now`.

This is deliberately a fix for **comparable birth timestamps**, not a new
universal clock model. Existing APIs allow an injected decay clock while
creation still uses `time.time()`; the original #176 guard explicitly uses
an injected clock starting at `5000`. Missing, invalid, nonfinite or future
creation timestamps do not establish a comparable birth and retain the
existing global interval. The first clock tick retains its documented
one-period bootstrap, and `now=None` retains discrete-step behavior. A clock
on another epoch cannot reveal actual birth age from the current metadata.
No clock settings, schema, or stored creation timestamps are rewritten.

## Verification

```sh
python -X utf8 -B -m pytest tests/test_fresh_row_decay.py tests/test_issue176_time_based_decay_guard.py tests/test_issue176_clock_rollback_decay_guard.py -q
python -X utf8 -B tools/testing/replay_transaction_guards.py --spec tests/fixtures/fresh_row_decay_mutations.json --baseline d74848fb33dbff1d719959df19d1266a1ac60f44 --mutate --output artifacts/fresh-row-decay
python -X utf8 -B bench/bench_fresh_row_decay.py --root . --output artifacts/fresh-row-decay-benchmark.json
```

On Windows, Python 3.12.10 / SQLite 3.49.1:

- New guard: **48 passed**. The identical guard on isolated baseline source:
  **25 failed / 23 passed**, with no errors or skips.
- Original time-based and clock-rollback guards: **11 passed** unchanged.
- Nine named mutants: **9/9 killed**, each by the specified behavioral
  assertion with one failed test, no errors and no skips. They cover missing
  clipping; each individual decay leg; future-time misclassification;
  freezing newborns; zero-age decay; changing bootstrap behavior; and
  disabling old-row catch-up.
- The benchmark's independent closed-form oracle checked **27,450 values**
  per source across empty / 100 / 1,000 / 5,000-node workloads. Fixed source
  has **0 failures**; baseline has **4,575 failures**, all from the mixed-birth
  workloads. The setup and oracle are outside measured regions.

## Cost measurements

Each workload retains three warmups, nine latency samples and three separate
Python-allocation peak probes. The largest workload has 5,000 nodes and 2,500
edges. The measurements are from a shared Windows host, not a performance
acceptance threshold:

| Workload | Baseline median ms | Fixed median ms | Baseline traced peak bytes | Fixed traced peak bytes |
| --- | ---: | ---: | ---: | ---: |
| Clock, all old | 81.5249 | 89.5016 | 2,504,322 | 2,611,290 |
| Clock, mixed births | 118.6533 | 103.8121 | 2,504,322 | 2,611,314 |
| Discrete, mixed births | 115.8040 | 119.7324 | 2,503,524 | 2,610,335 |

The fix adds one cursor read per clock-mode cycle, a constant-time age check
per eligible row, and one creation-timestamp column to the two node reads.
It retains the same asymptotic work and makes no per-node queries. Peak
figures are `tracemalloc` Python allocations, **not process RSS**. The latency
samples contain visible scheduler variation; neither the faster mixed
median nor a hardware-independent slowdown is claimed as a conclusion.

Full repository and cross-platform results are recorded separately by the
delivery validation; the figures above describe the local scoped checks.
