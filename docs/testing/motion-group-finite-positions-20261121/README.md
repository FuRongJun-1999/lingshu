# Motion-group coordinate validation

Motion-group fitting previously accepted a NaN in the first shared offset.
Subsequent drift comparisons were false, so a history with relative offsets
`[NaN, 2, 98, -100]` produced a `common_translation` group. It reported a
maximum offset drift of 0.0 and a maximum step drift of 198.0. A NaN Y value
could also appear in a successful `world_process("predict_groups")` result,
which failed strict JSON serialization.

The motion-group module now rejects non-finite X, Y, and Z coordinates in the
selected history window and in the visible anchors of the group being predicted.
The regression suite exercises both registered core actions and checks
that finite numeric strings still work and prediction leaves the input intact.

## Reproduction

```sh
python -X utf8 -m pytest tests/test_motion_group_finite_positions.py -q
python -X utf8 tools/verify_manifest.py --json --strict
python -X utf8 bench/bench_motion_group_finite_positions.py
```

The guard suite passes 18 cases, including two named mutations. On the
pre-fix revision `c15c12ea90ac4c7733ff2ef10e6fc586eab9b211`, the same
behavior tests give 14 failures and 2 passes. To replay, copy the new test
and its JSON fixture into a worktree at that revision and run the first command
with `-k 'not test_named_mutation'`. The two mutation cases require the fixed
source. Removing the finite-value check reproduces the false group
(`MOTION_GROUP_ACCEPTED_NAN_BASELINE`). Bypassing the conversion of visible
anchors reproduces the NaN prediction (`MOTION_GROUP_PREDICTED_NAN_ANCHOR`).

Strict export manifest verification passes 64 of 64 entries on the baseline
and after this change. The motion-group entry records the new source hash.

## Benchmark

The [raw Windows CPython 3.13 samples](benchmark-windows.json) cover five
synthetic workloads. Every finite result matched the pre-fix result.
Each version has 15 alternating paired measurements per workload.

| Workload | Baseline median | Fixed median | Fixed / baseline | Python allocation peak |
| --- | ---: | ---: | ---: | ---: |
| Detect, 1 entity × 10,000 ticks | 9.788 ms | 11.454 ms | 1.170× | 2,936,064 → 2,936,064 B |
| Detect, 8 entities × 512 ticks | 4.590 ms | 4.957 ms | 1.080× | 430,896 → 430,896 B |
| Detect, 24 entities × 128 ticks | 12.640 ms | 12.880 ms | 1.019× | 364,752 → 364,752 B |
| Predict, 1 visible anchor of 2 members | 4.149 μs | 4.380 μs | 1.056× | 1,616 → 1,776 B |
| Predict, 4 visible anchors of 8 members and 24 unrelated observations | 20.600 μs | 21.133 μs | 1.026× | 3,752 → 4,104 B |

Timings describe this host. Allocation peaks come from `tracemalloc`.
The validation covers input coordinates. Arithmetic overflow on extreme finite
values and custom group models supplied by a caller are outside this change.
